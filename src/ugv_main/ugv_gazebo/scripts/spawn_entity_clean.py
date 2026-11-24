#!/usr/bin/env python3
"""Clean Gazebo entity spawner that eliminates false-positive timeout errors on heavy worlds.

In Gazebo Classic 11 (`libgazebo_ros_factory.so`), `/spawn_entity` pushes the SDF onto Gazebo's
insert queue and polls `world->ModelByName()` for only ~0.5 s before returning `success=False` with:
  "Entity pushed to spawn queue, but spawn service timed outwaiting for entity to appear..."
When spawning `ugv_pro_audio` (3 GPU Ogre sensors + 4 ROS plugins + meshes) into a 50k-line forest
or hill world, Ogre sensor creation takes ~0.7-1.5 s, so upstream `spawn_entity.py` logs a false
`[ERROR] Spawn service failed. Exiting.` (exit code 1) right before the robot finishes appearing.

This node:
  1. Waits briefly for `/clock` so `gzserver` has finished loading the world and started stepping.
  2. Calls `/spawn_entity`.
  3. If Gazebo reports the entity was pushed to the spawn queue, verifies that the robot actually
     finishes initializing in the simulation (via `/wheel/odom`, `/ugv_gt_odom`, `/joint_states`,
     or `gz model`) before exiting cleanly with code 0.
"""
import argparse
import math
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState


def _euler_to_quat(roll: float, pitch: float, yaw: float):
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


class CleanEntitySpawner(Node):
    def __init__(self):
        super().__init__('spawn_entity')
        self.got_clock = False
        self.robot_alive = False
        self.create_subscription(Clock, '/clock', self._on_clock, qos_profile_sensor_data)
        self.create_subscription(JointState, '/joint_states', self._on_alive, 10)
        self.create_subscription(Odometry, '/wheel/odom', self._on_alive, 10)
        self.create_subscription(Odometry, '/ugv_gt_odom', self._on_alive, 10)
        self.cli = self.create_client(SpawnEntity, '/spawn_entity')

    def _on_clock(self, _msg):
        self.got_clock = True

    def _on_alive(self, _msg):
        self.robot_alive = True

    def spawn(self, entity: str, file_path: str, x: float, y: float, z: float,
              roll: float, pitch: float, yaw: float, timeout_s: float) -> int:
        try:
            xml_parsed = ET.parse(file_path).getroot()
            entity_xml = ET.tostring(xml_parsed, encoding='unicode')
        except Exception as exc:
            self.get_logger().error(f'Failed to load SDF {file_path}: {exc}')
            return 1

        # Wait for /spawn_entity service AND first /clock tick so world loading is complete
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and time.monotonic() < deadline:
            if self.cli.service_is_ready() and self.got_clock:
                break
            rclpy.spin_once(self, timeout_sec=0.1)

        if not self.cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f'/spawn_entity service not available after {timeout_s:.0f}s')
            return 1

        req = SpawnEntity.Request()
        req.name = entity
        req.xml = entity_xml
        req.robot_namespace = ''
        req.reference_frame = 'world'
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = x, y, z
        qx, qy, qz, qw = _euler_to_quat(roll, pitch, yaw)
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = qx, qy, qz, qw
        req.initial_pose = pose

        self.get_logger().info(f'Spawning [{entity}] at ({x:.2f}, {y:.2f}, {z:.2f}, yaw={yaw:.2f})')
        fut = self.cli.call_async(req)
        while rclpy.ok() and not fut.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

        if not fut.done() or fut.result() is None:
            self.get_logger().error(f'/spawn_entity call timed out for [{entity}]')
            return 1

        res = fut.result()
        msg = (res.status_message or '').strip()
        if res.success or 'already exists' in msg.lower():
            self.get_logger().info(f'Successfully spawned entity [{entity}]')
            return 0

        if 'pushed to spawn queue' in msg.lower():
            # Gazebo queued insertion while compiling GPU sensor shaders; confirm it finishes appearing
            verify_end = time.monotonic() + 25.0
            next_gz_check = time.monotonic() + 1.0
            while rclpy.ok() and time.monotonic() < verify_end:
                rclpy.spin_once(self, timeout_sec=0.1)
                if self.robot_alive:
                    self.get_logger().info(f'Successfully spawned entity [{entity}] (sensors active)')
                    return 0
                if time.monotonic() >= next_gz_check:
                    next_gz_check = time.monotonic() + 1.5
                    chk = subprocess.run(
                        ['gz', 'model', '-m', entity, '-i'],
                        capture_output=True, text=True, timeout=3.0, check=False,
                    )
                    if f'name: "{entity}"' in chk.stdout:
                        self.get_logger().info(f'Successfully spawned entity [{entity}]')
                        return 0

        self.get_logger().error(f'Spawn failed for [{entity}]: {msg}')
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-entity', required=True)
    parser.add_argument('-file', required=True)
    parser.add_argument('-x', type=float, default=0.0)
    parser.add_argument('-y', type=float, default=0.0)
    parser.add_argument('-z', type=float, default=0.0)
    parser.add_argument('-R', type=float, default=0.0)
    parser.add_argument('-P', type=float, default=0.0)
    parser.add_argument('-Y', type=float, default=0.0)
    parser.add_argument('-timeout', type=float, default=120.0)
    parsed, ros_args = parser.parse_known_args(argv)

    rclpy.init(args=ros_args)
    node = CleanEntitySpawner()
    rc = 0
    try:
        rc = node.spawn(
            parsed.entity, parsed.file,
            parsed.x, parsed.y, parsed.z,
            parsed.R, parsed.P, parsed.Y,
            parsed.timeout,
        )
    except (KeyboardInterrupt, ExternalShutdownException):
        rc = 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(rc)


if __name__ == '__main__':
    main()
