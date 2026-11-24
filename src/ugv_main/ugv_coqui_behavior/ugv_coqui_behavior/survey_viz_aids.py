"""Viewer aids for watching the Coqui survey in RViz (robot side: only the robot's own sensors and TF).

Seen map      There is no SLAM map in this setup. One in `every_n` lidar clouds (cloud_topic, points with a
              sensor-frame range in [min_range_m, max_range_m]) is placed in target_frame with the TF at the
              cloud's stamp and accumulated in voxel_size voxels (at most max_voxels):
                  /coqui_survey/seen_cloud  sensor_msgs/PointCloud2, x y z float32, latched, re-published every
                                            publish_period_s when it changed = what the robot has seen, in 3D.
Lifted grids  Nav2's costmaps and the traversability grid have their origin at z = 0, metres under this
              forest's ground (z 2.6 .. 3.3 m), so a 3D view hides them below the terrain. Latched copies with
              origin z = (base_frame z in the grid's frame) + grid_z_offset_m are republished:
                  grid_inputs[i] -> grid_outputs[i]   (defaults: GRID_INPUTS -> GRID_OUTPUTS below)
Lifted paths  Nav2's 2D plans have every pose at z = 0 for the same reason. Copies of flat plans (all z = 0)
              with every pose at (base_frame z in the plan's frame) + path_z_offset_m are republished; plans that
              already carry heights pass through unchanged:
                  path_inputs[i] -> path_outputs[i]   (defaults: PATH_INPUTS -> PATH_OUTPUTS below)

    ros2 run ugv_coqui_behavior survey_viz_aids --ros-args -p use_sim_time:=true
"""
import array
import sys

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformException, TransformListener

from .voxel_map import VoxelSet, crop_range, transform_points

SEEN_TOPIC = '/coqui_survey/seen_cloud'
GRID_INPUTS = ['/global_costmap/costmap', '/local_costmap/costmap', '/terrain_traversability/grid']
GRID_OUTPUTS = ['/coqui_survey/viz/global_costmap', '/coqui_survey/viz/local_costmap',
                '/coqui_survey/viz/traversability_grid']
PATH_INPUTS = ['/plan', '/local_plan']
PATH_OUTPUTS = ['/coqui_survey/viz/plan', '/coqui_survey/viz/local_plan']
# sensor_msgs/PointField datatype -> numpy type
PF_DTYPES = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16, 5: np.int32, 6: np.uint32, 7: np.float32,
             8: np.float64}


def cloud_xyz(msg):
    """(N, 3) float32 x, y, z of a PointCloud2 (NaN / inf rows are kept)."""
    fields = {f.name: f for f in msg.fields}
    if not all(k in fields for k in 'xyz'):
        raise ValueError(f'cloud has no x, y, z fields: {sorted(fields)}')
    if msg.width * msg.height == 0:
        return np.empty((0, 3), dtype=np.float32)
    same_type = all(f.datatype == msg.fields[0].datatype for f in msg.fields)
    if same_type and msg.row_step == msg.width * msg.point_step:
        return point_cloud2.read_points_numpy(msg, field_names=('x', 'y', 'z')).astype(np.float32, copy=False)
    # fields of mixed types (e.g. a uint16 'ring', which read_points_numpy refuses) or padded rows: read directly
    endian = '>' if msg.is_bigendian else '<'
    dt = np.dtype({'names': ['x', 'y', 'z'], 'offsets': [fields[k].offset for k in 'xyz'],
                   'formats': [np.dtype(PF_DTYPES[fields[k].datatype]).newbyteorder(endian) for k in 'xyz'],
                   'itemsize': msg.point_step})
    arr = np.ndarray((msg.height, msg.width), dtype=dt, buffer=np.frombuffer(msg.data, dtype=np.uint8),
                     strides=(msg.row_step, msg.point_step))
    return np.stack([arr['x'].ravel(), arr['y'].ravel(), arr['z'].ravel()], axis=1).astype(np.float32)


def xyz_cloud(points, frame, stamp):
    """PointCloud2 with float32 x, y, z fields from N x 3 points."""
    pts = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
    msg = PointCloud2()
    msg.header.frame_id = frame
    msg.header.stamp = stamp
    msg.height = 1
    msg.width = len(pts)
    msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate('xyz')]
    msg.is_bigendian = sys.byteorder == 'big'
    msg.point_step = 12
    msg.row_step = 12 * len(pts)
    msg.is_dense = True
    msg.data = array.array('B', pts.tobytes())  # array.array: assigning bytes is checked byte by byte (slow)
    return msg


class SurveyVizAids(Node):

    def __init__(self):
        super().__init__('survey_viz_aids')
        d = self.declare_parameter
        self.target_frame = d('target_frame', 'map').value
        self.base_frame = d('base_frame', 'base_footprint').value
        self.cloud_topic = d('cloud_topic', '/scan_cloud').value
        self.every_n = max(1, int(d('every_n', 3).value))
        self.min_range = float(d('min_range_m', 0.3).value)
        self.max_range = float(d('max_range_m', 20.0).value)
        tf_timeout = float(d('tf_timeout_s', 0.1).value)
        self.tf_timeout = Duration(seconds=tf_timeout)
        publish_period = float(d('publish_period_s', 4.0).value)
        self.voxels = VoxelSet(float(d('voxel_size', 0.18).value), int(d('max_voxels', 1500000).value))
        grid_inputs = list(d('grid_inputs', GRID_INPUTS).value)
        grid_outputs = list(d('grid_outputs', GRID_OUTPUTS).value)
        self.grid_z_offset = float(d('grid_z_offset_m', -0.05).value)
        path_inputs = list(d('path_inputs', PATH_INPUTS).value)
        path_outputs = list(d('path_outputs', PATH_OUTPUTS).value)
        self.path_z_offset = float(d('path_z_offset_m', 0.05).value)
        status_period = float(d('status_period_s', 30.0).value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)  # has its own reentrant callback group
        latched = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        # Seen map. The cloud callback (which may wait tf_timeout_s for TF) and the timers that read the voxel
        # set share one mutually exclusive group; the grid relays have their own group, so with the
        # multi-threaded executor they never wait behind a TF wait.
        seen_group = MutuallyExclusiveCallbackGroup()
        self.stats = {'received': 0, 'used': 0, 'no_tf': 0, 'errors': 0}
        self.changed = False
        self.last_stamp = None
        self.cap_logged = False
        self.pub_seen = self.create_publisher(PointCloud2, SEEN_TOPIC, latched)
        self.create_subscription(PointCloud2, self.cloud_topic, self.on_cloud, qos_profile_sensor_data,
                                 callback_group=seen_group)
        self.create_timer(publish_period, self.publish_seen, callback_group=seen_group)
        self.create_timer(status_period, self.report, callback_group=seen_group)

        # Lifted grids (Nav2 and the traversability node latch theirs: subscribe transient_local).
        grid_group = MutuallyExclusiveCallbackGroup()
        if len(grid_inputs) != len(grid_outputs):
            self.get_logger().error(f'grid_inputs ({len(grid_inputs)}) and grid_outputs ({len(grid_outputs)}) '
                                    'differ in length: only the first pairs are relayed')
        pairs = list(zip(grid_inputs, grid_outputs))
        self.base_zs = {}  # grid / path header frame -> last known base_frame z in that frame
        self.grids_relayed = {out: 0 for _, out in pairs}
        for src, out in pairs:
            pub = self.create_publisher(OccupancyGrid, out, latched)
            self.create_subscription(OccupancyGrid, src, lambda m, pub=pub, out=out: self.on_grid(m, pub, out),
                                     latched, callback_group=grid_group)

        # Lifted paths (Nav2 publishes /plan and /local_plan reliable + volatile, depth 1).
        if len(path_inputs) != len(path_outputs):
            self.get_logger().error(f'path_inputs ({len(path_inputs)}) and path_outputs ({len(path_outputs)}) '
                                    'differ in length: only the first pairs are relayed')
        path_pairs = list(zip(path_inputs, path_outputs))
        for src, out in path_pairs:
            pub = self.create_publisher(Path, out, 1)
            self.create_subscription(Path, src, lambda m, pub=pub, out=out: self.on_path(m, pub, out), 1,
                                     callback_group=grid_group)

        self.get_logger().info(
            f'seen map: {self.cloud_topic} (1 in {self.every_n} clouds, range {self.min_range:g}..'
            f'{self.max_range:g} m, TF wait {tf_timeout:g} s) -> {self.voxels.voxel:g} m voxels in '
            f'{self.target_frame} (max {self.voxels.max_voxels}) -> {SEEN_TOPIC} every {publish_period:g} s; '
            f'lifted grids (origin z = {self.base_frame} z {self.grid_z_offset:+g} m): '
            + ', '.join(f'{src} -> {out}' for src, out in pairs)
            + f'; lifted flat paths ({self.base_frame} z {self.path_z_offset:+g} m): '
            + ', '.join(f'{src} -> {out}' for src, out in path_pairs))

    # ------------------------------------------------------------------ seen map
    def on_cloud(self, msg):
        self.stats['received'] += 1
        if self.stats['received'] % self.every_n:
            return
        try:
            self.add_cloud(msg)
        except Exception as e:  # a viewer aid must never die on a bad message
            self.stats['errors'] += 1
            self.get_logger().warn(f'cloud ignored: {e!r}', throttle_duration_sec=10.0)

    def add_cloud(self, msg):
        frame = msg.header.frame_id
        try:
            tf = self.tf_buffer.lookup_transform(self.target_frame, frame, Time.from_msg(msg.header.stamp),
                                                 timeout=self.tf_timeout)
        except TransformException as e:
            self.stats['no_tf'] += 1
            self.get_logger().warn(f'cloud skipped, no TF {self.target_frame} <- {frame} at its stamp: {e}',
                                   throttle_duration_sec=10.0)
            return
        t, q = tf.transform.translation, tf.transform.rotation
        pts = crop_range(cloud_xyz(msg), self.min_range, self.max_range)
        added = self.voxels.add(transform_points(pts, (t.x, t.y, t.z), (q.x, q.y, q.z, q.w)))
        self.stats['used'] += 1
        self.last_stamp = msg.header.stamp
        self.changed = self.changed or added > 0
        if self.stats['used'] == 1:
            self.get_logger().info(f'first cloud used: {msg.width * msg.height} points in {frame}, '
                                   f'{len(pts)} in range -> {added} voxels')
        if self.voxels.full and not self.cap_logged:
            self.cap_logged = True
            self.get_logger().warn(f'seen map reached max_voxels ({self.voxels.max_voxels}): it keeps what it has, '
                                   'new voxels are dropped')

    def publish_seen(self):
        if not self.changed or self.last_stamp is None:
            return
        try:
            self.pub_seen.publish(xyz_cloud(self.voxels.centers(), self.target_frame, self.last_stamp))
            self.changed = False
        except Exception as e:
            self.get_logger().warn(f'{SEEN_TOPIC} not published: {e!r}', throttle_duration_sec=10.0)

    def report(self):
        s = self.stats
        n = len(self.voxels)
        grids = ', '.join(f'{out} {k}' for out, k in self.grids_relayed.items())
        zs = ', '.join(f'{frame} {z:.2f}' for frame, z in dict(self.base_zs).items()) or 'none yet'
        self.get_logger().info(
            f'seen map {n} voxels ({n * 12 / 1e6:.1f} MB), clouds received {s["received"]}, used {s["used"]}, '
            f'skipped (no TF) {s["no_tf"]}, errors {s["errors"]}; grids relayed: {grids}; robot z: {zs}')

    # ------------------------------------------------------------------ lifted grids and paths
    def base_z(self, frame, what):
        """base_frame z in `frame` from the latest TF (no waiting); without TF the last known value (0.0 at first)."""
        try:
            tf = self.tf_buffer.lookup_transform(frame, self.base_frame, Time())
            self.base_zs[frame] = tf.transform.translation.z
        except TransformException as e:
            self.get_logger().warn(f'{what}: no TF {frame} <- {self.base_frame} ({e}), using z '
                                   f'{self.base_zs.get(frame, 0.0):.2f}', throttle_duration_sec=10.0)
        return self.base_zs.get(frame, 0.0)

    def on_grid(self, msg, pub, out):
        try:
            frame = msg.header.frame_id
            msg.info.origin.position.z = float(self.base_z(frame, out) + self.grid_z_offset)
            pub.publish(msg)
            if self.grids_relayed[out] == 0:
                self.get_logger().info(f'first grid on {out}: {msg.info.width}x{msg.info.height} cells in {frame}, '
                                       f'origin z {msg.info.origin.position.z:.2f}')
            self.grids_relayed[out] += 1
        except Exception as e:
            self.get_logger().warn(f'{out}: grid not relayed: {e!r}', throttle_duration_sec=10.0)

    def on_path(self, msg, pub, out):
        try:
            if msg.poses and all(abs(p.pose.position.z) < 1e-6 for p in msg.poses):  # a flat 2D plan
                z = float(self.base_z(msg.header.frame_id, out) + self.path_z_offset)
                for p in msg.poses:
                    p.pose.position.z = z
            pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f'{out}: path not relayed: {e!r}', throttle_duration_sec=10.0)


def main(args=None):
    rclpy.init(args=args)
    node = SurveyVizAids()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl-C / launch shutdown: exit cleanly
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
