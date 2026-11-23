"""Save the current traversability grid (and elevation) to an .npz file.

  ros2 run ugv_traversability traversability_snapshot --ros-args -p out:=/tmp/trav.npz -p wait:=5.0
"""
import time

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2

from .cloud_utils import cloud_to_xyz


class Snapshot(Node):
    def __init__(self):
        super().__init__('traversability_snapshot')
        self.out = self.declare_parameter('out', '/tmp/traversability_snapshot.npz').value
        self.wait = float(self.declare_parameter('wait', 5.0).value)
        ns = self.declare_parameter('node_name', '/terrain_traversability').value
        self.grid = self.elev = self.gt = None
        q = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                       durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, ns + '/grid', lambda m: setattr(self, 'grid', m), q)
        self.create_subscription(PointCloud2, ns + '/elevation_cloud', self._elev, 2)
        self.create_subscription(Odometry, '/ugv_gt_odom', lambda m: setattr(self, 'gt', m), qos_profile_sensor_data)

    def _elev(self, m):
        # read x,y,z,intensity
        f = {x.name: x.offset for x in m.fields}
        a = np.frombuffer(bytes(m.data), dtype=np.float32).reshape(-1, m.point_step // 4)
        self.elev = a[:, [f['x'] // 4, f['y'] // 4, f['z'] // 4, f['intensity'] // 4]].copy()


def main():
    rclpy.init()
    n = Snapshot()
    t0 = time.time()
    while rclpy.ok() and (time.time() - t0 < n.wait or n.grid is None):
        rclpy.spin_once(n, timeout_sec=0.1)
        if time.time() - t0 > n.wait + 30:
            break
    g = n.grid
    if g is None:
        n.get_logger().error('no grid received')
        return
    cost = np.array(g.data, dtype=np.int8).reshape(g.info.height, g.info.width)
    extra = {}
    if n.elev is not None:
        extra['elevation_points'] = n.elev
    if n.gt is not None:
        p = n.gt.pose.pose
        extra['gt_pose'] = np.array([p.position.x, p.position.y, p.position.z, p.orientation.x, p.orientation.y,
                                     p.orientation.z, p.orientation.w])
    np.savez_compressed(n.out, cost=cost, resolution=g.info.resolution,
                        origin=np.array([g.info.origin.position.x, g.info.origin.position.y]),
                        frame=g.header.frame_id, **extra)
    n.get_logger().info(f'saved {n.out}: {cost.shape}, known {np.mean(cost >= 0) * 100:.0f} %, '
                        f'lethal {np.mean(cost == 100) * 100:.1f} %')
    n.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
