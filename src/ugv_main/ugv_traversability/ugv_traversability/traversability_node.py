"""ROS 2 node: point clouds -> rolling elevation map -> traversability cost.

Subscribes
  <cloud_topics>   sensor_msgs/PointCloud2  (default /scan_cloud = 16-ch 3D lidar)
  <imu_topic>      sensor_msgs/Imu          (roll/pitch compensation of 2D odometry)
Publishes
  ~/grid              nav_msgs/OccupancyGrid  cost 0..100 (100 = lethal), -1 unknown, frame = fixed_frame
  ~/elevation_cloud   sensor_msgs/PointCloud2 x,y,z(elevation),intensity(cost) for RViz, frame = fixed_frame
  ~/obstacles         sensor_msgs/PointCloud2 lethal cells, frame = robot_frame   -> Nav2 obstacle_layer (marking)
  ~/ground            sensor_msgs/PointCloud2 traversable cells, frame = robot_frame -> Nav2 obstacle_layer (clearing)
"""

import time

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Imu, PointCloud2, PointField
from tf2_ros import Buffer, TransformException, TransformListener

from .cloud_utils import apply, cloud_to_xyz, quat_to_rot, rot_to_rpy, rpy_to_rot, transform_to_matrix, \
    xyz_to_cloud_fields
from .elevation_map import RollingElevationMap
from .features import TraversabilityParams, compute_traversability


class TraversabilityNode(Node):

    def __init__(self):
        super().__init__('terrain_traversability')
        d = self.declare_parameter
        self.fixed_frame = d('fixed_frame', 'odom').value
        self.robot_frame = d('robot_frame', 'base_footprint').value
        topics = list(d('cloud_topics', ['/scan_cloud']).value)
        strides = list(d('cloud_strides', [1]).value)
        # Per-topic reliability.  Large organised clouds (depth camera: 640x480x32 B = 9.8 MB) are
        # split into many UDP fragments; with best-effort QoS losing one fragment drops the whole
        # message, so in practice none arrive.  Use 'reliable' for those (publisher must offer it).
        qos_names = list(d('cloud_qos', ['best_effort']).value)
        self.imu_topic = d('imu_topic', '/imu').value
        self.use_imu_attitude = bool(d('use_imu_attitude', True).value)
        self.min_range = float(d('min_range', 0.25).value)
        self.max_range = float(d('max_range', 6.0).value)
        # Optional per-topic max range (<= 0 -> use max_range).  A low-mounted depth camera sees
        # the ground at a grazing angle: beyond ~2 m its sample rows are decimetres apart and its
        # height error grows with range, so it is best used only for the lidar's near-field gap.
        max_ranges = [float(v) for v in d('cloud_max_ranges', [0.0]).value]
        self.max_height_above_robot = float(d('max_height_above_robot', 0.9).value)
        self.max_depth_below_robot = float(d('max_depth_below_robot', 2.0).value)
        self.self_filter = list(d('self_filter_box', [-0.16, 0.16, -0.13, 0.13, 0.35]).value)
        self.footprint_radius = float(d('footprint_radius', 0.12).value)
        # fill the footprint on the robot's (tilted) base plane rather than a flat disc
        self.footprint_follow_slope = bool(d('footprint_follow_slope', True).value)
        # place clouds only with the pose at their own timestamp (park + retry up to pose_wait_s)
        self.exact_pose = bool(d('exact_pose', True).value)
        self.pose_wait_s = float(d('pose_wait_s', 0.3).value)
        self.pending = []
        self.publish_rate = float(d('publish_rate', 5.0).value)
        self.obstacle_cost = int(d('obstacle_cost_threshold', 100).value)
        self.ground_stride = max(1, int(d('ground_cloud_stride', 2).value))
        self.tf_timeout = float(d('tf_timeout', 0.0).value)

        p = TraversabilityParams()
        p.resolution = float(d('resolution', p.resolution).value)
        p.plane_radius = float(d('plane_radius', p.plane_radius).value)
        p.step_radius = float(d('step_radius', p.step_radius).value)
        p.min_valid_ratio = float(d('min_valid_ratio', p.min_valid_ratio).value)
        p.slope_safe_deg = float(d('slope_safe_deg', p.slope_safe_deg).value)
        p.slope_critical_deg = float(d('slope_critical_deg', p.slope_critical_deg).value)
        p.step_safe = float(d('step_safe', p.step_safe).value)
        p.step_critical = float(d('step_critical', p.step_critical).value)
        p.roughness_safe = float(d('roughness_safe', p.roughness_safe).value)
        p.roughness_critical = float(d('roughness_critical', p.roughness_critical).value)
        p.weights = {'slope': float(d('weight_slope', 1.0).value),
                     'step': float(d('weight_step', 1.0).value),
                     'roughness': float(d('weight_roughness', 1.0).value)}
        self.params = p
        self.map = RollingElevationMap(length=float(d('map_length', 8.0).value), resolution=p.resolution,
                                       alpha=float(d('fusion_alpha', 0.4).value),
                                       max_age=float(d('max_age', 20.0).value),
                                       recenter_dist=float(d('recenter_distance', 0.5).value))

        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.static_cache = {}
        self.imu_rp = None
        self.z_est = 0.0
        self.last_xy = None
        self.odom_is_3d = False
        self.tf_z0 = None
        self.last_pose = None  # (stamp_msg, T_fixed_robot_compensated, T_fixed_robot_tf)
        self.stats = {'clouds': 0, 'points': 0, 'proc_ms': 0.0, 'pub_ms': 0.0, 'pubs': 0, 'tf_drop': 0}
        self.create_timer(0.02, self.retry_pending)
        self.frames_seen = {}

        if len(strides) < len(topics):
            strides += [strides[-1] if strides else 1] * (len(topics) - len(strides))
        if len(qos_names) < len(topics):
            qos_names += [qos_names[-1] if qos_names else 'best_effort'] * (len(topics) - len(qos_names))
        if len(max_ranges) < len(topics):
            max_ranges += [0.0] * (len(topics) - len(max_ranges))
        max_ranges = [mr if mr > 0.0 else self.max_range for mr in max_ranges]
        for t, s, qn, mr in zip(topics, strides, qos_names, max_ranges):
            if qn == 'reliable':
                qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                                 durability=QoSDurabilityPolicy.VOLATILE)
            elif qn == 'best_effort':
                qos = qos_profile_sensor_data
            else:
                raise ValueError(f"cloud_qos entries must be 'reliable' or 'best_effort', got {qn!r}")
            self.create_subscription(PointCloud2, t,
                                     lambda m, s=int(s), mr=float(mr): self.on_cloud(m, s, mr), qos)
            self.get_logger().info(f'listening to {t} (stride {s}, {qn}, max range {mr:.1f} m)')
        self.create_subscription(Imu, self.imu_topic, self.on_imu, qos_profile_sensor_data)

        latched = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_grid = self.create_publisher(OccupancyGrid, '~/grid', latched)
        self.pub_elev = self.create_publisher(PointCloud2, '~/elevation_cloud', 2)
        self.pub_obst = self.create_publisher(PointCloud2, '~/obstacles', 2)
        self.pub_ground = self.create_publisher(PointCloud2, '~/ground', 2)
        self.create_timer(1.0 / self.publish_rate, self.on_timer)
        self.create_timer(10.0, self.report)

    # ------------------------------------------------------------------ inputs
    def on_imu(self, msg):
        q = msg.orientation
        if q.x == q.y == q.z == q.w == 0.0:
            return
        r, p, _ = rot_to_rpy(quat_to_rot(q.x, q.y, q.z, q.w))
        self.imu_rp = (r, p)

    def lookup(self, target, source, stamp, exact=False):
        # Never block the executor: try the exact stamp (optionally waiting tf_timeout, default 0),
        # else (exact=False) fall back to the latest transform.
        try:
            if self.tf_timeout > 0.0:
                return self.tf_buffer.lookup_transform(target, source, stamp,
                                                       timeout=Duration(seconds=self.tf_timeout))
            return self.tf_buffer.lookup_transform(target, source, stamp)
        except TransformException:
            if exact:
                return None
            try:
                return self.tf_buffer.lookup_transform(target, source, Time())
            except TransformException as e:
                self.get_logger().warn(f'TF {target}<-{source}: {e}', throttle_duration_sec=5.0)
                return None

    def robot_pose(self, stamp, exact=False):
        """(T_compensated, T_tf, tf_msg).  With use_imu_attitude the 2D odometry pose is
        upgraded to 3D using IMU roll/pitch and integrated height."""
        tfm = self.lookup(self.fixed_frame, self.robot_frame, stamp, exact=exact)
        if tfm is None:
            return None
        T_tf = transform_to_matrix(tfm.transform)
        if self.use_imu_attitude and not self.odom_is_3d:
            # Auto-detect a TF that is already 3D (Gazebo diff_drive with odometry_source=world,
            # robot_localization 3D, RTAB-Map ...): then compensating again would double-count.
            r_tf, p_tf, _ = rot_to_rpy(T_tf[:3, :3])
            if self.tf_z0 is None:
                self.tf_z0 = T_tf[2, 3]
            if max(abs(r_tf), abs(p_tf)) > np.radians(0.5) or abs(T_tf[2, 3] - self.tf_z0) > 0.01:
                self.odom_is_3d = True
                self.get_logger().info(f'TF {self.fixed_frame}->{self.robot_frame} is already 3D '
                                       '(roll/pitch/z present): IMU attitude compensation disabled')
        if not self.use_imu_attitude or self.odom_is_3d or self.imu_rp is None:
            return T_tf, T_tf, tfm
        x, y = T_tf[0, 3], T_tf[1, 3]
        _, _, yaw = rot_to_rpy(T_tf[:3, :3])
        roll, pitch = self.imu_rp
        R = rpy_to_rot(roll, pitch, yaw)
        if self.last_xy is not None:
            dxy = np.array([x, y]) - self.last_xy
            s = dxy @ np.array([np.cos(yaw), np.sin(yaw)])  # signed wheel travel
            if abs(s) < 1.0:  # ignore jumps (re-localisation)
                self.z_est += s * R[2, 0] / max(np.hypot(R[0, 0], R[1, 0]), 1e-3)
        self.last_xy = np.array([x, y])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = (x, y, T_tf[2, 3] + self.z_est)
        return T, T_tf, tfm

    def retry_pending(self):
        if not self.pending:
            return
        keep = []
        for t_recv, msg, stride, max_range in self.pending:
            if self.robot_pose(Time.from_msg(msg.header.stamp), exact=True) is not None:
                self.on_cloud(msg, stride, max_range, retry=True)
            elif time.monotonic() - t_recv > self.pose_wait_s:
                self.stats['tf_drop'] += 1
            else:
                keep.append((t_recv, msg, stride, max_range))
        self.pending = keep

    def on_cloud(self, msg, stride, max_range=None, retry=False):
        """A cloud is only ever placed with the robot pose AT ITS TIMESTAMP. When that pose is not in
        the TF buffer yet (the cloud arrived before the odometry of the same instant) the cloud is
        parked and retried; it is never placed with the latest pose instead: while the robot turns
        (Nav2 rotate-in-place / spin recovery) that mismatch smears the ground into fake steps and
        filled the whole map window with lethal cells."""
        if self.exact_pose and not retry:
            if self.robot_pose(Time.from_msg(msg.header.stamp), exact=True) is None:
                self.pending.append((time.monotonic(), msg, stride, max_range))
                if len(self.pending) > 8:
                    self.pending.pop(0)
                    self.stats['tf_drop'] = self.stats.get('tf_drop', 0) + 1
                return
        t0 = time.perf_counter()
        frame = msg.header.frame_id
        key = frame
        if key not in self.static_cache:
            tf_rs = self.lookup(self.robot_frame, frame, Time())
            if tf_rs is None:
                return
            self.static_cache[key] = transform_to_matrix(tf_rs.transform)
        T_rs = self.static_cache[key]
        pose = self.robot_pose(Time.from_msg(msg.header.stamp), exact=self.exact_pose)
        if pose is None:
            return
        T_fr, T_tf, tfm = pose
        pts = cloud_to_xyz(msg, stride)
        if len(pts) == 0:
            return
        r = np.linalg.norm(pts, axis=1)
        pts = pts[(r > self.min_range) & (r < (max_range or self.max_range))]
        pts_r = apply(T_rs, pts)
        x0, x1, y0, y1, z1 = self.self_filter
        inbox = (pts_r[:, 0] > x0) & (pts_r[:, 0] < x1) & (pts_r[:, 1] > y0) & (pts_r[:, 1] < y1) & \
                (pts_r[:, 2] < z1)
        pts_f = apply(T_fr, pts_r[~inbox])
        zr = T_fr[2, 3]
        pts_f = pts_f[(pts_f[:, 2] < zr + self.max_height_above_robot) &
                      (pts_f[:, 2] > zr - self.max_depth_below_robot)]
        stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        self.map.move_to(T_fr[0, 3], T_fr[1, 3])
        self.map.insert(pts_f, stamp)
        self.map.set_footprint(T_fr[0, 3], T_fr[1, 3], zr, self.footprint_radius, stamp,
                               normal=T_fr[:3, 2] if self.footprint_follow_slope else None)
        self.last_pose = (tfm.header.stamp, T_fr, T_tf)
        self.stats['clouds'] += 1
        self.frames_seen[frame] = self.frames_seen.get(frame, 0) + 1
        self.stats['points'] += len(pts_f)
        self.stats['proc_ms'] += (time.perf_counter() - t0) * 1e3

    # ------------------------------------------------------------------ outputs
    def on_timer(self):
        if self.map.origin is None or self.last_pose is None:
            return
        t0 = time.perf_counter()
        now = self.get_clock().now()
        self.map.expire(now.nanoseconds * 1e-9)
        layers = compute_traversability(self.map.h, self.params)
        cost = layers['cost']
        stamp_tf, _, T_tf = self.last_pose
        stamp = stamp_tf

        g = OccupancyGrid()
        g.header.frame_id = self.fixed_frame
        g.header.stamp = stamp
        g.info.resolution = self.map.res
        g.info.width = g.info.height = self.map.n
        g.info.origin.position.x = float(self.map.origin[0])
        g.info.origin.position.y = float(self.map.origin[1])
        g.info.origin.orientation.w = 1.0
        g.data = cost.ravel().tolist()
        self.pub_grid.publish(g)

        X, Y = self.map.cell_centers()
        known = cost >= 0
        if self.pub_elev.get_subscription_count():
            pts = np.stack([X[known], Y[known], self.map.h[known]], axis=1)
            self.pub_elev.publish(self.make_cloud(pts, self.fixed_frame, stamp, cost[known].astype(np.float32)))

        # Nav2 clouds in robot frame (sensor origin = robot -> correct raytracing)
        T_inv = np.linalg.inv(T_tf)
        lethal = cost >= self.obstacle_cost
        pts = np.stack([X[lethal], Y[lethal], self.map.h[lethal]], axis=1)
        self.pub_obst.publish(self.make_cloud(apply(T_inv, pts), self.robot_frame, stamp))
        free = known & ~lethal
        sub = np.zeros_like(free)
        sub[::self.ground_stride, ::self.ground_stride] = True
        free &= sub
        pts = np.stack([X[free], Y[free], self.map.h[free]], axis=1)
        self.pub_ground.publish(self.make_cloud(apply(T_inv, pts), self.robot_frame, stamp))
        self.stats['pub_ms'] += (time.perf_counter() - t0) * 1e3
        self.stats['pubs'] += 1

    @staticmethod
    def make_cloud(pts, frame, stamp, intensity=None):
        data, step, fields = xyz_to_cloud_fields(pts, intensity)
        m = PointCloud2()
        m.header.frame_id = frame
        m.header.stamp = stamp
        m.height = 1
        m.width = len(pts)
        m.fields = [PointField(name=n, offset=o, datatype=dt, count=c) for n, o, dt, c in fields]
        m.is_bigendian = False
        m.point_step = step
        m.row_step = step * len(pts)
        m.is_dense = True
        m.data = data
        return m

    def report(self):
        s = self.stats
        if s['clouds'] == 0:
            self.get_logger().warn('no point clouds processed yet (check cloud_topics / TF)',
                                   throttle_duration_sec=30.0)
            return
        known = np.isfinite(self.map.h).mean() * 100
        self.get_logger().info(
            f"clouds {s['clouds']} ({s['points'] / s['clouds']:.0f} pts avg, {s['proc_ms'] / s['clouds']:.1f} ms), "
            f"maps {s['pubs']} ({s['pub_ms'] / max(s['pubs'], 1):.1f} ms), known {known:.0f} %, "
            f"z_est {self.z_est:+.2f} m, per frame {self.frames_seen}, "
            f"clouds dropped (no pose at their stamp) {s['tf_drop']}")
        self.frames_seen = {}
        for k in s:
            s[k] = 0 if isinstance(s[k], int) else 0.0


def main(args=None):
    rclpy.init(args=args)
    node = TraversabilityNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl-C / launch shutdown: exit cleanly (code 0)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
