#!/usr/bin/env python3
# encoding: utf-8
import sys
import select
import termios
import tty

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

msg = """
Control Your UGV & 2-Axis Pan-Tilt Camera!
---------------------------
Driving around:
   u    i    o
   j    k    l
   m    ,    .

2-Axis Pan-Tilt Camera Turret:
   a / d : Pan camera Left / Right
   r / f : Tilt camera Up / Down
   g     : Reset / Level camera to center (0, 0)

Speed adjustments:
   q/z : increase/decrease max speeds by 10%
   w/x : increase/decrease only linear speed by 10%
   e/c : increase/decrease only angular speed by 10%
   t/T : x and y speed switch
   s/S : toggle stop keyboard control
   space, k : force stop

CTRL-C to quit
"""

moveBindings = {
    'i': (1, 0),
    'o': (1, -1),
    'j': (0, 1),
    'l': (0, -1),
    'u': (1, 1),
    ',': (-1, 0),
    '.': (-1, 1),
    'm': (-1, -1),
    'I': (1, 0),
    'O': (1, -1),
    'J': (0, 1),
    'L': (0, -1),
    'U': (1, 1),
    'M': (-1, -1),
}

speedBindings = {
    'Q': (1.1, 1.1),
    'Z': (.9, .9),
    'W': (1.1, 1),
    'X': (.9, 1),
    'E': (1, 1.1),
    'C': (1, .9),
    'q': (1.1, 1.1),
    'z': (.9, .9),
    'w': (1.1, 1),
    'x': (.9, 1),
    'e': (1, 1.1),
    'c': (1, .9),
}

# Pan-Tilt step bindings: (delta_pan_rad, delta_tilt_rad)
# Note: in ugv_pro_audio, axis of pt_base_link_to_pt_link1 is (0, 0, -1),
# so negative pan rotates left (+yaw) and positive pan rotates right (-yaw).
# Axis of pt_link1_to_pt_link2 is +Z in pt_link2, so positive tilt pitches UP.
ptBindings = {
    'a': (-0.10, 0.0),
    'A': (-0.10, 0.0),
    'd': (0.10, 0.0),
    'D': (0.10, 0.0),
    'r': (0.0, 0.10),
    'R': (0.0, 0.10),
    'f': (0.0, -0.10),
    'F': (0.0, -0.10),
}


class ugv_Keyboard(Node):
    def __init__(self, name):
        super().__init__(name)
        self.pub = self.create_publisher(Twist, 'cmd_vel', 1)
        self.pub_pt_traj = self.create_publisher(JointTrajectory, '/set_joint_trajectory', 10)
        self.pub_pt_js = self.create_publisher(JointState, '/ugv/joint_states', 10)

        self.declare_parameter("linear_speed_limit", 1.0)
        self.declare_parameter("angular_speed_limit", 1.0)
        self.linenar_speed_limit = self.get_parameter("linear_speed_limit").get_parameter_value().double_value
        self.angular_speed_limit = self.get_parameter("angular_speed_limit").get_parameter_value().double_value
        self.settings = termios.tcgetattr(sys.stdin)

        self.pt_pan = 0.0
        self.pt_tilt = 0.0

    def publish_pan_tilt(self):
        joint_names = ['pt_base_link_to_pt_link1', 'pt_link1_to_pt_link2']
        positions = [float(self.pt_pan), float(self.pt_tilt)]

        # Use stamp=0 so Gazebo's libgazebo_ros_joint_pose_trajectory.so executes immediately
        # even when keyboard_ctrl is launched without --ros-args -p use_sim_time:=true
        traj = JointTrajectory()
        traj.header.stamp.sec = 0
        traj.header.stamp.nanosec = 0
        traj.header.frame_id = 'base_footprint'
        traj.joint_names = joint_names
        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = 20_000_000
        traj.points = [pt]
        self.pub_pt_traj.publish(traj)

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = joint_names
        js.position = positions
        self.pub_pt_js.publish(js)

    def getKey(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def vels(self, speed, turn):
        return "currently:\tspeed %.2f\tturn %.2f\tpan %.2f rad\ttilt %.2f rad" % (
            speed, turn, self.pt_pan, self.pt_tilt
        )


def main():
    rclpy.init()
    ugv_keyboard = ugv_Keyboard("keyboard_ctrl")
    xspeed_switch = True
    (speed, turn) = (0.2, 0.5)
    (x, th) = (0, 0)
    status = 0
    stop = False
    count = 0
    twist = Twist()
    try:
        print(msg)
        print(ugv_keyboard.vels(speed, turn))
        # Level the Pan-Tilt camera on startup
        ugv_keyboard.publish_pan_tilt()

        while True:
            key = ugv_keyboard.getKey()
            if key == "t" or key == "T":
                xspeed_switch = not xspeed_switch
            elif key == "s" or key == "S":
                print("stop keyboard control: {}".format(not stop))
                stop = not stop
            elif key in ptBindings:
                d_pan, d_tilt = ptBindings[key]
                ugv_keyboard.pt_pan = max(-3.14, min(3.14, ugv_keyboard.pt_pan + d_pan))
                ugv_keyboard.pt_tilt = max(-0.60, min(1.52, ugv_keyboard.pt_tilt + d_tilt))
                ugv_keyboard.publish_pan_tilt()
                print(ugv_keyboard.vels(speed, turn))
            elif key in ('g', 'G'):
                ugv_keyboard.pt_pan = 0.0
                ugv_keyboard.pt_tilt = 0.0
                ugv_keyboard.publish_pan_tilt()
                print("Pan-Tilt camera reset to level (0.0, 0.0)")

            if key in moveBindings.keys():
                x = moveBindings[key][0]
                th = moveBindings[key][1]
                count = 0
            elif key in speedBindings.keys():
                speed = speed * speedBindings[key][0]
                turn = turn * speedBindings[key][1]
                count = 0
                if speed > ugv_keyboard.linenar_speed_limit:
                    speed = ugv_keyboard.linenar_speed_limit
                    print("Linear speed limit reached!")
                if turn > ugv_keyboard.angular_speed_limit:
                    turn = ugv_keyboard.angular_speed_limit
                    print("Angular speed limit reached!")
                print(ugv_keyboard.vels(speed, turn))
                if status == 14:
                    print(msg)
                status = (status + 1) % 15
            elif key == ' ' or key == 'k' or key == 'K':
                (x, th) = (0, 0)
            else:
                count = count + 1
                if count > 4:
                    (x, th) = (0, 0)
                if key == '\x03':
                    break

            if xspeed_switch:
                twist.linear.x = float(speed * x)
            else:
                twist.linear.y = float(speed * x)
            twist.angular.z = float(turn * th)

            if not stop:
                ugv_keyboard.pub.publish(twist)
            else:
                ugv_keyboard.pub.publish(Twist())
    except Exception as e:
        print(e)
    finally:
        ugv_keyboard.pub.publish(Twist())
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, ugv_keyboard.settings)
    ugv_keyboard.destroy_node()
    rclpy.shutdown()
