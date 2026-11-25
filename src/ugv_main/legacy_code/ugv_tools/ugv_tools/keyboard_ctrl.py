#!/usr/bin/env python3
# encoding: utf-8
"""Keyboard teleop for the UGV + 2-axis pan-tilt camera.

Driving behaves exactly like ros2 teleop_twist_keyboard (same keys, same default
speeds 0.5 m/s / 1.0 rad/s, the last command is held until another key is pressed).
The only addition is the pan-tilt camera keys a/d/r/f/g, which do NOT change or stop
the current driving command.
"""
import sys
import threading

import rclpy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import termios
import tty

msg = """
This node takes keypresses from the keyboard and publishes them
as Twist messages. It works best with a US keyboard layout.
---------------------------
Moving around:
   u    i    o
   j    k    l
   m    ,    .

For Holonomic mode (strafing), hold down the shift key:
---------------------------
   U    I    O
   J    K    L
   M    <    >

t : up (+z)
b : down (-z)

anything else : stop

q/z : increase/decrease max speeds by 10%
w/x : increase/decrease only linear speed by 10%
e/c : increase/decrease only angular speed by 10%

Pan-tilt camera (does not affect driving):
---------------------------
a/d : pan left / right
r/f : tilt up / down
g   : reset camera to straight ahead, level

CTRL-C to quit
"""

# Identical to teleop_twist_keyboard: (x, y, z, th)
moveBindings = {
    'i': (1, 0, 0, 0),
    'o': (1, 0, 0, -1),
    'j': (0, 0, 0, 1),
    'l': (0, 0, 0, -1),
    'u': (1, 0, 0, 1),
    ',': (-1, 0, 0, 0),
    '.': (-1, 0, 0, 1),
    'm': (-1, 0, 0, -1),
    'O': (1, -1, 0, 0),
    'I': (1, 0, 0, 0),
    'J': (0, 1, 0, 0),
    'L': (0, -1, 0, 0),
    'U': (1, 1, 0, 0),
    '<': (-1, 0, 0, 0),
    '>': (-1, -1, 0, 0),
    'M': (-1, 1, 0, 0),
    't': (0, 0, 1, 0),
    'b': (0, 0, -1, 0),
}

speedBindings = {
    'q': (1.1, 1.1),
    'z': (.9, .9),
    'w': (1.1, 1),
    'x': (.9, 1),
    'e': (1, 1.1),
    'c': (1, .9),
}

# Pan-tilt steps: (delta_pan_rad, delta_tilt_rad).
# In ugv_pro_audio the pan joint axis is (0, 0, -1), so +pan turns the camera right;
# the tilt joint's +angle pitches the camera up.
ptBindings = {
    'a': (-0.10, 0.0),
    'd': (0.10, 0.0),
    'r': (0.0, 0.10),
    'f': (0.0, -0.10),
}
PT_RESET_KEY = 'g'
PAN_LIMIT = 3.14
TILT_MIN, TILT_MAX = -0.60, 1.52
PT_JOINTS = ['pt_base_link_to_pt_link1', 'pt_link1_to_pt_link2']


def getKey(settings):
    tty.setraw(sys.stdin.fileno())
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def vels(speed, turn):
    return 'currently:\tspeed %.2f\tturn %.2f ' % (speed, turn)


def publish_pan_tilt(node, traj_pub, js_pub, pan, tilt):
    # stamp 0 -> Gazebo's libgazebo_ros_joint_pose_trajectory executes it immediately
    traj = JointTrajectory()
    traj.header.frame_id = 'base_footprint'
    traj.joint_names = PT_JOINTS
    pt = JointTrajectoryPoint()
    pt.positions = [float(pan), float(tilt)]
    pt.time_from_start.nanosec = 20_000_000
    traj.points = [pt]
    traj_pub.publish(traj)

    # real robot (ugv_driver) / joint_state_publisher
    js = JointState()
    js.header.stamp = node.get_clock().now().to_msg()
    js.name = PT_JOINTS
    js.position = [float(pan), float(tilt)]
    js_pub.publish(js)


def main():
    settings = termios.tcgetattr(sys.stdin)

    rclpy.init()
    node = rclpy.create_node('keyboard_ctrl')

    speed = node.declare_parameter('speed', 0.5).value
    turn = node.declare_parameter('turn', 1.0).value

    pub = node.create_publisher(Twist, 'cmd_vel', 10)
    traj_pub = node.create_publisher(JointTrajectory, '/set_joint_trajectory', 10)
    js_pub = node.create_publisher(JointState, '/ugv/joint_states', 10)

    spinner = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spinner.start()

    x = y = z = th = 0.0
    pan = tilt = 0.0
    status = 0
    twist = Twist()

    try:
        print(msg)
        print(vels(speed, turn))
        while True:
            key = getKey(settings)
            if key in moveBindings:
                x, y, z, th = moveBindings[key]
            elif key in speedBindings:
                speed = speed * speedBindings[key][0]
                turn = turn * speedBindings[key][1]
                print(vels(speed, turn))
                if status == 14:
                    print(msg)
                status = (status + 1) % 15
            elif key in ptBindings or key == PT_RESET_KEY:
                if key == PT_RESET_KEY:
                    pan = tilt = 0.0
                else:
                    pan = max(-PAN_LIMIT, min(PAN_LIMIT, pan + ptBindings[key][0]))
                    tilt = max(TILT_MIN, min(TILT_MAX, tilt + ptBindings[key][1]))
                publish_pan_tilt(node, traj_pub, js_pub, pan, tilt)
                print('camera:\t\tpan %+.2f rad\ttilt %+.2f rad' % (pan, tilt))
                continue  # camera keys leave the driving command untouched
            else:
                x = y = z = th = 0.0
                if key == '\x03':
                    break

            twist.linear.x = x * speed
            twist.linear.y = y * speed
            twist.linear.z = z * speed
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = th * turn
            pub.publish(twist)

    except Exception as e:
        print(e)

    finally:
        pub.publish(Twist())
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        rclpy.shutdown()


if __name__ == '__main__':
    main()
