#!/usr/bin/env python3
"""Wheel odometry -> a body velocity the 3D EKF can use (/wheel/odom -> /wheel/odom_3d).

The wheels only measure forward speed. A wheeled robot also does not fly or slide sideways, and a 3D filter needs to be
told so: with vy and vz left unmeasured the estimated height wanders off. This relay republishes the twist with
  vx  as measured, its uncertainty growing with speed and with turning (skid-steer wheels slip when they turn)
  vy  0, loosely (the wheels do slide a little sideways)
  vz  0, tightly (the base stays on the ground; climbing comes from the pitch of the body, not from vz)
The yaw rate is dropped: six skid-steering wheels measure it badly and the gyro measures it well.
"""
import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

BIG = 1e6


def body_twist_covariance(vx, wz, sigma_vx=0.02, slip_frac=0.06, turn_slip=0.04, sigma_vy=0.03, sigma_vz=0.01):
    """Diagonal (vx, vy, vz, wx, wy, wz) variances of the wheel measurement."""
    s_vx = sigma_vx + slip_frac * abs(vx) + turn_slip * abs(wz)
    s_vy = sigma_vy + turn_slip * abs(wz)
    return s_vx ** 2, s_vy ** 2, sigma_vz ** 2, BIG, BIG, BIG


class WheelOdomRelay(Node):
    def __init__(self):
        super().__init__('wheel_odom_relay')
        d = self.declare_parameter
        self.scale = float(d('vx_scale', 1.0).value)     # calibrate on hardware: true / measured straight-line distance
        self.kw = {k: float(d(k, v).value) for k, v in
                   (('sigma_vx', 0.02), ('slip_frac', 0.06), ('turn_slip', 0.04), ('sigma_vy', 0.03), ('sigma_vz', 0.01))}
        self.pub = self.create_publisher(Odometry, '/wheel/odom_3d', qos_profile_sensor_data)
        self.create_subscription(Odometry, '/wheel/odom', self.on_odom, qos_profile_sensor_data)

    def on_odom(self, msg):
        tw = msg.twist.twist
        tw.linear.x *= self.scale
        tw.linear.y = tw.linear.z = 0.0
        cov = [0.0] * 36
        for i, v in enumerate(body_twist_covariance(tw.linear.x, tw.angular.z, **self.kw)):
            cov[7 * i] = float(v)
        msg.twist.covariance = cov
        self.pub.publish(msg)


def main(args=None):
    from rclpy._rclpy_pybind11 import RCLError
    rclpy.init(args=args)
    node = WheelOdomRelay()
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
