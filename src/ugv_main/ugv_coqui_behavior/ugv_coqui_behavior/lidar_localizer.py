#!/usr/bin/env python3
"""GPS-denied localization: map -> odom from the 3D lidar (see lidar_localizer_core.py).

The 3D EKF (wheels + IMU, config/ekf_odom.yaml) gives a smooth odom -> base_footprint that drifts. This node matches what
the lidar sees of trunks, bushes and grass edges with everything it has seen before and publishes the correction as
map -> odom. It needs no GPS and no prior map: the map starts at the trailhead pose given in the profile (initial_pose).

    in   /scan_cloud (sensor_msgs/PointCloud2), TF odom -> base_footprint
    out  TF map -> odom, /lidar_localizer/status (std_msgs/String, JSON, 1 Hz)
"""
import json
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
import tf2_ros

from .lidar_localizer_core import CorrectionSmoother, DEFAULTS, StructureLocalizer


def transform_matrix(t):
    q, p = t.rotation, t.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    M = np.eye(4)
    M[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    M[:3, 3] = (p.x, p.y, p.z)
    return M


def cloud_xyz(msg):
    """(N, 3) xyz of a PointCloud2 with float32 x, y, z first."""
    off = {f.name: f.offset for f in msg.fields}
    dt = np.dtype({'names': ['x', 'y', 'z'], 'formats': ['<f4'] * 3, 'offsets': [off['x'], off['y'], off['z']],
                   'itemsize': msg.point_step})
    a = np.frombuffer(msg.data, dtype=dt, count=msg.width * msg.height)
    return np.stack([a['x'], a['y'], a['z']], axis=1).astype(float)


class LidarLocalizer(Node):
    def __init__(self):
        super().__init__('lidar_localizer')
        d = self.declare_parameter
        self.map_frame = d('map_frame', 'map').value
        self.odom_frame = d('odom_frame', 'odom').value
        self.base_frame = d('base_frame', 'base_footprint').value
        start = [float(v) for v in d('initial_pose', [0.0, 0.0, 0.0, 0.0]).value]     # x, y, z, yaw of odom in map
        self.every = max(1, int(d('process_every', 2).value))      # use every n-th scan (10 Hz lidar)
        self.pose_wait = float(d('pose_wait_s', 0.6).value)
        self.postdate = float(d('transform_tolerance', 0.2).value)  # as AMCL: the correction is good for a moment ahead
        params = {k: type(v)(d(k, v).value) for k, v in DEFAULTS.items()}
        smooth = {k: float(d(k, v).value) for k, v in (('max_mps', 0.25), ('max_zps', 0.10), ('max_dps', 6.0))}
        self.loc = StructureLocalizer(start, params)
        self.smooth = CorrectionSmoother(start, **smooth)
        self.tf_buf = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_lis = tf2_ros.TransformListener(self.tf_buf, self)
        self.tf_br = tf2_ros.TransformBroadcaster(self)
        self.status_pub = self.create_publisher(String, '/lidar_localizer/status', 5)
        self.T_base_lidar = {}
        self.parked, self.n, self.pivot, self.last_pub = [], 0, (0.0, 0.0), None
        self.stats = {'scans': 0, 'matched': 0, 'faults': 0, 'no_pose': 0, 'ms': 0.0}
        self.create_subscription(PointCloud2, d('cloud_topic', '/scan_cloud').value, self.on_cloud,
                                 qos_profile_sensor_data)
        self.create_timer(0.05, self.on_timer)
        self.create_timer(1.0, self.on_status)
        self.get_logger().info(f'lidar localizer started at {start} (GPS-denied, no prior map)')

    # ------------------------------------------------------------------------------------- input
    def on_cloud(self, msg):
        self.n += 1
        if self.n % self.every == 0:
            self.parked.append((time.monotonic(), msg))
            self.parked = self.parked[-5:]
            self.work()

    def work(self):
        """A scan is only used with the odometry pose AT ITS STAMP; it waits until that pose is in the TF buffer."""
        keep = []
        for t, msg in self.parked:
            try:
                T = transform_matrix(self.tf_buf.lookup_transform(
                    self.odom_frame, self.base_frame, Time.from_msg(msg.header.stamp)).transform)
            except tf2_ros.TransformException:
                if time.monotonic() - t < self.pose_wait:
                    keep.append((t, msg))
                else:
                    self.stats['no_pose'] += 1
                continue
            self.use(msg, T)
        self.parked = keep

    def use(self, msg, T):
        f = msg.header.frame_id
        if f not in self.T_base_lidar:
            try:
                self.T_base_lidar[f] = transform_matrix(
                    self.tf_buf.lookup_transform(self.base_frame, f, Time()).transform)
            except tf2_ros.TransformException:
                return
        t0 = time.perf_counter()
        pts = cloud_xyz(msg)
        pts = pts[np.isfinite(pts).all(1)]
        M = self.T_base_lidar[f]
        st = self.loc.process(pts @ M[:3, :3].T + M[:3, 3], T)
        self.pivot = (T[0, 3], T[1, 3])
        s = self.stats
        s['scans'] += 1
        s['matched'] += bool(st['matched'])
        s['faults'] += bool(st['fault'])
        s['ms'] += (time.perf_counter() - t0) * 1e3

    # ------------------------------------------------------------------------------------ output
    def on_timer(self):
        self.work()
        now = self.get_clock().now()
        dt = 0.0 if self.last_pub is None else (now - self.last_pub).nanoseconds * 1e-9
        if self.last_pub is not None and dt <= 0.0:
            return      # the (simulated) clock has not moved
        self.last_pub = now
        x, y, z, yaw = self.smooth.step(self.loc.C, dt, self.pivot)
        t = TransformStamped()
        t.header.stamp = (now + Duration(seconds=self.postdate)).to_msg()
        t.header.frame_id, t.child_frame_id = self.map_frame, self.odom_frame
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = float(x), float(y), float(z)
        t.transform.rotation.z, t.transform.rotation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.tf_br.sendTransform(t)

    def on_status(self):
        s, n = self.stats, max(self.stats['scans'], 1)
        lost = self.loc.misses >= int(self.loc.p['lost_after'])
        m = String()
        m.data = json.dumps({'correction': [round(float(v), 3) for v in self.loc.C], 'lost': lost,
                             'misses': self.loc.misses, 'score': self.loc.status.get('score'),
                             'points': self.loc.status.get('points'), 'scans': s['scans'], 'matched': s['matched'],
                             'faults': s['faults'], 'no_pose': s['no_pose'], 'ms': round(s['ms'] / n, 1)})
        self.status_pub.publish(m)
        if lost:
            self.get_logger().warn('lidar localizer: no match for a while, running on odometry and searching wider',
                                   throttle_duration_sec=10.0)


def main(args=None):
    from rclpy._rclpy_pybind11 import RCLError
    rclpy.init(args=args)
    node = LidarLocalizer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RCLError:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
