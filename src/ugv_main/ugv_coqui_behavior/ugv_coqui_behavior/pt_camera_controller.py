#!/usr/bin/env python3
"""2-Axis Pan-Tilt Camera Controller for ugv_pro_audio.

Points the Pan-Tilt camera (`pt_camera_link`) at any 3D point in `map`, `odom`,
or `base_footprint` coordinates by computing:
  - Pan angle (`pt_base_link_to_pt_link1`, axis = 0 0 -1)
  - Tilt angle (`pt_link1_to_pt_link2`, axis = 0 0 1 in pt_link2, +tilt = pitch UP)

Publishes both:
  1. `trajectory_msgs/msg/JointTrajectory` on `/set_joint_trajectory` (for Gazebo's
     `libgazebo_ros_joint_pose_trajectory.so` plugin)
  2. `sensor_msgs/msg/JointState` on `/ugv/joint_states` (for `joint_state_publisher`
     and `ugv_driver` hardware interface)
"""

import math
import numpy as np

import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import PointStamped, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tf2_ros


PT_PIVOT_IN_FOOTPRINT = np.array([-0.008, 0.0, 0.211], dtype=np.float64)


def quat_to_rot_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


class PtCameraControllerNode(Node):
    def __init__(self):
        super().__init__('pt_camera_controller')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.robot_pos = np.zeros(3, dtype=np.float64)
        self.robot_rot = np.eye(3, dtype=np.float64)

        self.target_pan = 0.0
        self.target_tilt = 0.0
        self.current_pan = 0.0
        self.current_tilt = 0.0
        self.last_sent_pan = None
        self.last_sent_tilt = None

        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.create_subscription(
            PointStamped, '/coqui_behavior/look_at_point', self._look_at_cb, 10
        )
        self.create_subscription(
            Vector3, '/coqui_behavior/pan_tilt_cmd', self._pan_tilt_cmd_cb, 10
        )

        self.traj_pub = self.create_publisher(JointTrajectory, '/set_joint_trajectory', 10)
        self.js_pub = self.create_publisher(JointState, '/ugv/joint_states', 10)
        self.state_pub = self.create_publisher(Vector3, '/coqui_behavior/pan_tilt_state', 10)

        self.timer = self.create_timer(0.05, self._control_step)
        self.get_logger().info('2-Axis Pan-Tilt Camera Controller node ready.')

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.robot_pos = np.array([p.x, p.y, p.z], dtype=np.float64)
        self.robot_rot = quat_to_rot_matrix(q.x, q.y, q.z, q.w)

    def _pan_tilt_cmd_cb(self, msg: Vector3):
        self.target_pan = float(np.clip(msg.x, -3.14, 3.14))
        self.target_tilt = float(np.clip(msg.y, -0.60, 1.52))

    def _look_at_cb(self, msg: PointStamped):
        target_pt = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=np.float64)
        frame = msg.header.frame_id.strip() or 'map'

        if frame == 'base_footprint':
            rel_body = target_pt - PT_PIVOT_IN_FOOTPRINT
        else:
            # Lookup transform from base_footprint to target frame
            pos_fp = self.robot_pos
            rot_fp = self.robot_rot
            try:
                tf = self.tf_buffer.lookup_transform(
                    frame, 'base_footprint', rclpy.time.Time(), timeout=Duration(seconds=0.05)
                )
                t = tf.transform.translation
                q = tf.transform.rotation
                pos_fp = np.array([t.x, t.y, t.z], dtype=np.float64)
                rot_fp = quat_to_rot_matrix(q.x, q.y, q.z, q.w)
            except Exception:
                pass
            pivot_world = pos_fp + rot_fp @ PT_PIVOT_IN_FOOTPRINT
            rel_body = rot_fp.T @ (target_pt - pivot_world)

        dx, dy, dz = float(rel_body[0]), float(rel_body[1]), float(rel_body[2])
        horiz_dist = math.hypot(dx, dy)

        # In ugv_pro_audio, `pt_base_link_to_pt_link1` has axis (0, 0, -1):
        # so positive pan rotates clockwise (-yaw in base_footprint), hence pan = -atan2(dy, dx).
        pan = -math.atan2(dy, dx)
        # In `pt_link1_to_pt_link2`, positive angle pitches the camera UP (+Z):
        tilt = math.atan2(dz, max(horiz_dist, 0.05))

        self.target_pan = float(np.clip(pan, -3.14, 3.14))
        self.target_tilt = float(np.clip(tilt, -0.60, 1.52))

    def _control_step(self):
        # Smooth interpolation toward target pan/tilt
        alpha = 0.40
        self.current_pan += alpha * (self.target_pan - self.current_pan)
        self.current_tilt += alpha * (self.target_tilt - self.current_tilt)
        if abs(self.target_pan - self.current_pan) < 0.0015:
            self.current_pan = self.target_pan
        if abs(self.target_tilt - self.current_tilt) < 0.0015:
            self.current_tilt = self.target_tilt

        now = self.get_clock().now().to_msg()
        joint_names = ['pt_base_link_to_pt_link1', 'pt_link1_to_pt_link2']
        positions = [float(self.current_pan), float(self.current_tilt)]

        # 1. Only publish JointTrajectory to Gazebo when pan/tilt actually changes!
        # Calling ODEJoint::SetPosition at 20Hz while driving perturbs wheel-ground contact velocities.
        need_traj_send = (
            self.last_sent_pan is None
            or abs(self.current_pan - self.last_sent_pan) > 0.002
            or abs(self.current_tilt - self.last_sent_tilt) > 0.002
        )
        if need_traj_send:
            traj = JointTrajectory()
            traj.header.stamp = now
            traj.header.frame_id = 'base_footprint'
            traj.joint_names = joint_names
            pt = JointTrajectoryPoint()
            pt.positions = positions
            pt.time_from_start.sec = 0
            pt.time_from_start.nanosec = 20_000_000
            traj.points = [pt]
            self.traj_pub.publish(traj)
            self.last_sent_pan = float(self.current_pan)
            self.last_sent_tilt = float(self.current_tilt)

        # 2. Publish JointState for joint_state_publisher / TF / RViz
        js = JointState()
        js.header.stamp = now
        js.name = joint_names
        js.position = positions
        self.js_pub.publish(js)

        # 3. Publish current state vector (rad)
        st = Vector3(
            x=float(self.current_pan),
            y=float(self.current_tilt),
            z=float(math.hypot(self.target_pan - self.current_pan, self.target_tilt - self.current_tilt)),
        )
        self.state_pub.publish(st)


def main(args=None):
    rclpy.init(args=args)
    node = PtCameraControllerNode()
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

