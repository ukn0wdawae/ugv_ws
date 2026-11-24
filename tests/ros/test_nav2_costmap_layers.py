#!/usr/bin/env python3
"""Live ROS 2 Humble verification of Nav2 global_costmap slope + obstacle layer combination.

Runs inside the ROS 2 container (no Gazebo required, ~10 s) on an isolated ROS_DOMAIN_ID.
Proves empirically against the real `nav2_costmap_2d` C++ node that:
  1. Unknown (-1) cells in /terrain_traversability/global_grid do NOT erase raytraced free space
     from obstacle_layer (fixing the 21:31 StaticLayer use_maximum + track_unknown_space bug!).
  2. Obstacles marked by /terrain_traversability/obstacles stay lethal (100) even where global_grid is -1.
  3. Graded slope costs (1..90) in global_grid survive when /terrain_traversability/ground clears over them.
  4. Truly unseen space stays unknown (-1) so frontier exploration continues to work.

Usage:
  python3 tests/ros/test_nav2_costmap_layers.py [--nav2-params src/ugv_main/ugv_coqui_behavior/config/nav2_survey_base.yaml]
"""
import argparse
import array
import os
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
import yaml

import rclpy
from geometry_msgs.msg import TransformStamped
from lifecycle_msgs.srv import ChangeState
from lifecycle_msgs.msg import Transition
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
import tf2_ros


def make_pc2(pts, frame, stamp):
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    m = PointCloud2()
    m.header.frame_id = frame
    m.header.stamp = stamp
    m.height = 1
    m.width = len(pts)
    m.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    m.is_bigendian = False
    m.point_step = 12
    m.row_step = 12 * len(pts)
    m.is_dense = True
    m.data = pts.tobytes()
    return m


def rewrite_for_standalone_costmap(nav2_path):
    """Extract global_costmap params into a standalone nav2_costmap_2d node YAML with wall time."""
    doc = yaml.safe_load(open(nav2_path))
    gc = dict(doc['global_costmap']['global_costmap']['ros__parameters'])
    gc['use_sim_time'] = False
    gc['width'] = 20
    gc['height'] = 20
    gc['resolution'] = 0.1
    gc['update_frequency'] = 5.0
    gc['publish_frequency'] = 5.0
    out = {'costmap': {'costmap': {'ros__parameters': gc}}}
    fd, path = tempfile.mkstemp(prefix='test_gc_', suffix='.yaml')
    with os.fdopen(fd, 'w') as f:
        yaml.safe_dump(out, f, sort_keys=False)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nav2-params', default='src/ugv_main/ugv_coqui_behavior/config/nav2_survey_base.yaml')
    args = ap.parse_args()

    params_path = rewrite_for_standalone_costmap(args.nav2_params)
    proc = subprocess.Popen(
        ['ros2', 'run', 'nav2_costmap_2d', 'nav2_costmap_2d',
         '--ros-args', '--params-file', params_path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    rclpy.init()
    node = Node('test_nav2_costmap_layers')
    try:
        latched = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        pub_global = node.create_publisher(OccupancyGrid, '/terrain_traversability/global_grid', latched)
        pub_obs = node.create_publisher(PointCloud2, '/terrain_traversability/obstacles', 10)
        pub_gnd = node.create_publisher(PointCloud2, '/terrain_traversability/ground', 10)
        tf_br = tf2_ros.TransformBroadcaster(node)

        latest = {'msg': None}
        node.create_subscription(OccupancyGrid, '/costmap/costmap', lambda m: latest.__setitem__('msg', m), 5)

        # Activate the lifecycle costmap node
        cli = node.create_client(ChangeState, '/costmap/costmap/change_state')
        if not cli.wait_for_service(timeout_sec=12.0):
            raise RuntimeError('nav2_costmap_2d lifecycle service did not appear')

        def send_tf():
            now = node.get_clock().now().to_msg()
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = 'map'
            t.child_frame_id = 'base_footprint'
            t.transform.rotation.w = 1.0
            tf_br.sendTransform(t)
            return now

        for trans_id in (Transition.TRANSITION_CONFIGURE, Transition.TRANSITION_ACTIVATE):
            send_tf()
            req = ChangeState.Request()
            req.transition.id = trans_id
            fut = cli.call_async(req)
            t0 = time.time()
            while not fut.done() and time.time() - t0 < 8.0:
                send_tf()
                rclpy.spin_once(node, timeout_sec=0.05)
            assert fut.done() and fut.result().success, f'lifecycle transition {trans_id} failed'

        # Build a 20x20 m global slope grid @ 0.2 m, mostly UNKNOWN (-1), with a known 55-cost slope patch
        # in x in [1.0, 2.6], y in [-1.0, 1.0]. Everything else in global_grid is -1!
        res_g, n_g = 0.2, 100
        gdata = np.full((n_g, n_g), -1, dtype=np.int8)
        xs = -10.0 + (np.arange(n_g) + 0.5) * res_g
        ys = -10.0 + (np.arange(n_g) + 0.5) * res_g
        Xg, Yg = np.meshgrid(xs, ys)
        gdata[(Xg >= 1.0) & (Xg <= 2.6) & (Yg >= -1.0) & (Yg <= 1.0)] = 55

        # Publish ground rays clearing x in [-3.5, +3.5], y in [-0.4, +0.4] (crosses BOTH unknown global_grid
        # at x=-2.0 and sloped global_grid at x=+1.8!) + two obstacles at (-2.5, 0.0) [unknown in global_grid]
        # and (+2.0, 0.8) [inside sloped patch].
        gnd_pts = []
        for gx in np.linspace(-3.5, 3.5, 36):
            for gy in np.linspace(-0.4, 0.4, 5):
                if abs(gx) > 0.3:
                    gnd_pts.append([gx, gy, 0.0])
        obs_pts = [
            [-2.55, 0.05, 0.2],
            [2.05, 0.85, 0.2],
        ]

        t0 = time.time()
        while time.time() - t0 < 4.5:
            now = send_tf()
            gg = OccupancyGrid()
            gg.header.frame_id = 'map'
            gg.header.stamp = now
            gg.info.resolution = res_g
            gg.info.width = gg.info.height = n_g
            gg.info.origin.position.x = -10.0
            gg.info.origin.position.y = -10.0
            gg.info.origin.orientation.w = 1.0
            buf = array.array('b')
            buf.frombytes(np.ascontiguousarray(gdata, dtype=np.int8).tobytes())
            gg.data = buf
            pub_global.publish(gg)
            pub_gnd.publish(make_pc2(gnd_pts, 'base_footprint', now))
            pub_obs.publish(make_pc2(obs_pts, 'base_footprint', now))
            rclpy.spin_once(node, timeout_sec=0.08)

        msg = latest['msg']
        assert msg is not None, 'did not receive /costmap/costmap'
        arr = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        ox, oy, r = msg.info.origin.position.x, msg.info.origin.position.y, msg.info.resolution

        def val(wx, wy):
            ix = int(np.floor((wx - ox) / r))
            iy = int(np.floor((wy - oy) / r))
            return int(arr[iy, ix])

        def val_max3(wx, wy):
            ix = int(np.floor((wx - ox) / r))
            iy = int(np.floor((wy - oy) / r))
            return int(arr[max(0, iy - 1):iy + 2, max(0, ix - 1):ix + 2].max())

        free_in_unk_global = val(-1.5, 0.0)          # raytraced free where global_grid == -1
        obs_in_unk_global = val_max3(-2.55, 0.05)    # obstacle where global_grid == -1
        obs_in_slope_global = val_max3(2.05, 0.85)   # obstacle where global_grid == 55
        slope_preserved = val(1.8, 0.0)              # sloped cell (55) raytraced free by ground_cloud
        truly_unseen = val(0.0, 7.0)                 # unknown in BOTH layers

        print(f'RESULT: free_in_unk_global={free_in_unk_global}, obs_in_unk_global={obs_in_unk_global}, '
              f'obs_in_slope_global={obs_in_slope_global}, slope_preserved={slope_preserved}, '
              f'truly_unseen={truly_unseen}, known_cells={(arr >= 0).sum()}')

        assert free_in_unk_global == 0, f'global slope layer erased raytraced free space! got {free_in_unk_global}'
        assert obs_in_unk_global == 100, f'global slope layer erased obstacle! got {obs_in_unk_global}'
        assert obs_in_slope_global == 100, f'obstacle on slope not lethal! got {obs_in_slope_global}'
        assert 30 <= slope_preserved <= 80, f'graded slope cost not preserved! got {slope_preserved}'
        assert truly_unseen == -1, f'unseen frontier cell lost unknown (-1)! got {truly_unseen}'
        print('PASS: all 5 Nav2 global_costmap slope+obstacle invariants verified live!')
    finally:
        node.destroy_node()
        rclpy.shutdown()
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=4)
        except subprocess.TimeoutExpired:
            proc.kill()
        if os.path.exists(params_path):
            os.remove(params_path)


if __name__ == '__main__':
    main()
