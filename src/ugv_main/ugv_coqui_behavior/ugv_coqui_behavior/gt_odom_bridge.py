#!/usr/bin/env python3
"""Optional simulation-only ground-truth odometry bridge for `localization:=perfect` ablation runs.

When `ugv_pro_audio` runs with realistic wheel encoder odometry (`/wheel/odom`, publish_odom_tf=false),
default `localization:=ekf` and `localization:=slam` fuse `/wheel/odom` + Madgwick `/imu` through
robot_localization's EKF. This bridge is ONLY started when the user explicitly requests
`localization:=perfect` to isolate planning/behavior bugs from localization drift.
"""
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
import tf2_ros


class GtOdomBridge(Node):
    def __init__(self):
        super().__init__('gt_odom_bridge')
        self.odom_frame = self.declare_parameter('odom_frame', 'odom').value
        self.base_frame = self.declare_parameter('base_frame', 'base_footprint').value
        self.tf_br = tf2_ros.TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 20)
        self.imu_pub = self.create_publisher(Imu, '/imu', 20)
        self.create_subscription(Odometry, '/ugv_gt_odom', self._on_gt, qos_profile_sensor_data)
        self.create_subscription(Imu, '/imu/raw', self._on_imu, qos_profile_sensor_data)

    def _on_gt(self, msg: Odometry):
        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.odom_frame
        out.child_frame_id = self.base_frame
        out.pose = msg.pose
        out.twist = msg.twist
        self.odom_pub.publish(out)

        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation
        self.tf_br.sendTransform(t)

    def _on_imu(self, msg: Imu):
        self.imu_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GtOdomBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
