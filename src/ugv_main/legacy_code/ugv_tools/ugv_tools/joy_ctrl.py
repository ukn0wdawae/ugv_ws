#!/usr/bin/env python3
# encoding: utf-8

import getpass
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy, JointState
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


def get_joystick_names():
    if not HAS_PYGAME:
        return []
    try:
        pygame.init()
        pygame.joystick.init()
        joystick_names = []
        for i in range(pygame.joystick.get_count()):
            joystick = pygame.joystick.Joystick(i)
            joystick.init()
            joystick_names.append(joystick.get_name())
        pygame.quit()
        return joystick_names
    except Exception:
        return []


class JoyTeleop(Node):
    def __init__(self, name):
        super().__init__(name)
        self.Joy_active = True
        self.user_name = getpass.getuser()
        self.linear_Gear = 1.0
        self.angular_Gear = 1.0

        # Pan-Tilt state (radians) for the 2-axis Pan-Tilt camera turret
        self.pt_pan = 0.0
        self.pt_tilt = 0.0

        # Publishers
        self.pub_cmdVel = self.create_publisher(Twist, 'cmd_vel', 10)
        self.pub_JoyState = self.create_publisher(Bool, 'JoyState', 10)
        self.pub_pt_traj = self.create_publisher(JointTrajectory, '/set_joint_trajectory', 10)
        self.pub_pt_js = self.create_publisher(JointState, '/ugv/joint_states', 10)

        # Subscriber
        self.sub_Joy = self.create_subscription(Joy, 'joy', self.buttonCallback, 10)

        # Speed limit parameters
        self.declare_parameter('xspeed_limit', 0.5)
        self.declare_parameter('yspeed_limit', 0.5)
        self.declare_parameter('angular_speed_limit', 1.2)
        self.xspeed_limit = self.get_parameter('xspeed_limit').get_parameter_value().double_value
        self.yspeed_limit = self.get_parameter('yspeed_limit').get_parameter_value().double_value
        self.angular_speed_limit = self.get_parameter('angular_speed_limit').get_parameter_value().double_value

        joysticks = get_joystick_names()
        self.joysticks = joysticks[0] if len(joysticks) != 0 else 'default'
        # [linear_gear_btn, angular_gear_btn, turn_axis]
        self.switch_dict = {
            'Xbox 360 Controller': [9, 10, 3],
            'SHANWAN Android Gamepad': [13, 14, 2],
            'default': [9, 10, 3],
        }
        self.get_logger().info(
            'Joystick Teleop ready: Left stick Y = forward/back, Right stick X (or Left stick X) = turn, '
            'D-pad = Pan-Tilt camera, Button B/Circle = E-stop.'
        )

    def _get_mapping(self):
        return self.switch_dict.get(self.joysticks, self.switch_dict['default'])

    def buttonCallback(self, joy_data: Joy):
        if not isinstance(joy_data, Joy):
            return

        lin_btn, ang_btn, turn_axis = self._get_mapping()
        buttons = joy_data.buttons
        axes = joy_data.axes

        # Linear gear cycling (L3 / stick press)
        if len(buttons) > lin_btn and buttons[lin_btn] == 1:
            if self.linear_Gear == 1.0:
                self.linear_Gear = 1.0 / 3.0
            elif abs(self.linear_Gear - 1.0 / 3.0) < 1e-3:
                self.linear_Gear = 2.0 / 3.0
            else:
                self.linear_Gear = 1.0

        # Angular gear cycling (R3 / stick press)
        if len(buttons) > ang_btn and buttons[ang_btn] == 1:
            if self.angular_Gear == 1.0:
                self.angular_Gear = 0.25
            elif abs(self.angular_Gear - 0.25) < 1e-3:
                self.angular_Gear = 0.50
            elif abs(self.angular_Gear - 0.50) < 1e-3:
                self.angular_Gear = 0.75
            else:
                self.angular_Gear = 1.0

        # Forward/backward on Left Stick vertical (axes[1])
        ax_fwd = axes[1] if len(axes) > 1 else 0.0
        # Steering on Right Stick horizontal (axes[3] or axes[2]) OR Left Stick horizontal (axes[0])
        ax_turn = 0.0
        if len(axes) > turn_axis and abs(axes[turn_axis]) >= 0.2:
            ax_turn = axes[turn_axis]
        elif len(axes) > 2 and turn_axis != 2 and abs(axes[2]) >= 0.2 and abs(axes[2]) < 0.99:
            ax_turn = axes[2]
        elif len(axes) > 0 and abs(axes[0]) >= 0.2:
            ax_turn = axes[0]

        xlinear_speed = self.filter_data(ax_fwd) * self.xspeed_limit * self.linear_Gear
        angular_speed = self.filter_data(ax_turn) * self.angular_speed_limit * self.angular_Gear

        # Emergency stop on Button 1 (B / Circle)
        if len(buttons) > 1 and buttons[1] == 1:
            xlinear_speed = 0.0
            angular_speed = 0.0

        xlinear_speed = max(-self.xspeed_limit, min(self.xspeed_limit, xlinear_speed))
        angular_speed = max(-self.angular_speed_limit, min(self.angular_speed_limit, angular_speed))

        twist = Twist()
        twist.linear.x = float(xlinear_speed)
        twist.angular.z = float(angular_speed)
        if self.Joy_active:
            self.pub_cmdVel.publish(twist)

        # D-pad (axes[6], axes[7]) controls the 2-Axis Pan-Tilt Camera Turret
        pt_changed = False
        if len(axes) > 6 and abs(axes[6]) > 0.1:
            self.pt_pan = max(-3.14, min(3.14, self.pt_pan - 0.08 * axes[6]))
            pt_changed = True
        if len(axes) > 7 and abs(axes[7]) > 0.1:
            self.pt_tilt = max(-0.60, min(1.52, self.pt_tilt + 0.08 * axes[7]))
            pt_changed = True
        # Button 0 (A / Cross) resets Pan-Tilt camera to center (0, 0)
        if len(buttons) > 0 and buttons[0] == 1:
            self.pt_pan = 0.0
            self.pt_tilt = 0.0
            pt_changed = True

        if pt_changed:
            joint_names = ['pt_base_link_to_pt_link1', 'pt_link1_to_pt_link2']
            positions = [float(self.pt_pan), float(self.pt_tilt)]

            traj = JointTrajectory()
            traj.header.stamp.sec = 0
            traj.header.stamp.nanosec = 0
            traj.header.frame_id = 'base_footprint'
            traj.joint_names = joint_names
            pt = JointTrajectoryPoint()
            pt.positions = positions
            pt.time_from_start.sec = 0
            pt.time_from_start.nanosec = 50_000_000
            traj.points = [pt]
            self.pub_pt_traj.publish(traj)

            js = JointState()
            js.header.stamp = self.get_clock().now().to_msg()
            js.name = joint_names
            js.position = positions
            self.pub_pt_js.publish(js)

    def filter_data(self, value):
        if abs(value) < 0.2:
            value = 0.0
        return value


def main(args=None):
    rclpy.init(args=args)
    joy_ctrl = JoyTeleop('joy_ctrl')
    try:
        rclpy.spin(joy_ctrl)
    except KeyboardInterrupt:
        pass
    finally:
        joy_ctrl.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
