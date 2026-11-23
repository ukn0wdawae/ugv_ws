#!/usr/bin/env python3
"""Closed-loop driving trials in Gazebo (run inside the sim container).

For every trial in a JSON file: teleport the robot (gazebo_ros_state
/set_entity_state), let it settle, publish a constant cmd_vel for `duration`
seconds of *sim* time and record the ground-truth trajectory (p3d plugin,
/ugv_gt_odom).  Writes a JSON result per trial:

  progress_m   distance travelled along the commanded heading
  efficiency   progress / (v * duration)
  max_pitch / max_roll (deg), final pose, stuck flag (<35 % efficiency)

The world must load libgazebo_ros_state.so (the test harness injects it into a
temporary copy of the world).

  ros2 run ... / python3 sim_drive_test.py trials.json results.json [--entity ugv_pro_audio] [--z-key z_fixed]
"""

import argparse
import json
import math
import time

import rclpy
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


def yaw_to_quat(yaw):
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)


def quat_to_rpy(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


class Runner(Node):
    def __init__(self, entity):
        super().__init__('sim_drive_test', parameter_overrides=[
            rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.entity = entity
        self.cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.gt = None
        self.create_subscription(Odometry, '/ugv_gt_odom', self._gt, qos_profile_sensor_data)
        self.cli = self.create_client(SetEntityState, '/set_entity_state')

    def _gt(self, msg):
        self.gt = msg

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def spin_for(self, sim_seconds, cb=None, rate=20.0):
        t0 = self.now_s()
        while rclpy.ok() and self.now_s() - t0 < sim_seconds:
            rclpy.spin_once(self, timeout_sec=1.0 / rate)
            if cb:
                cb()

    def teleport(self, x, y, z, yaw):
        if not self.cli.wait_for_service(timeout_sec=30.0):
            raise RuntimeError('/set_entity_state not available (gazebo_ros_state plugin missing?)')
        req = SetEntityState.Request()
        st = EntityState()
        st.name = self.entity
        st.pose.position.x, st.pose.position.y, st.pose.position.z = x, y, z
        qx, qy, qz, qw = yaw_to_quat(yaw)
        st.pose.orientation.x, st.pose.orientation.y = qx, qy
        st.pose.orientation.z, st.pose.orientation.w = qz, qw
        st.reference_frame = 'world'
        req.state = st
        fut = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        if not fut.result() or not fut.result().success:
            raise RuntimeError(f'teleport failed: {fut.result()}')

    def run_trial(self, t, z_key):
        self.cmd.publish(Twist())
        self.teleport(t['x'], t['y'], t[z_key], t['yaw'])
        self.spin_for(2.0, cb=lambda: self.cmd.publish(Twist()))
        # wait for a fresh gt
        self.gt = None
        while rclpy.ok() and self.gt is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        p0 = self.gt.pose.pose.position
        start = (p0.x, p0.y, p0.z)
        traj = [start]
        maxp = maxr = 0.0
        tw = Twist()
        tw.linear.x = float(t['v'])
        tw.angular.z = float(t.get('w', 0.0))
        state = {'maxp': 0.0, 'maxr': 0.0}

        def tick():
            self.cmd.publish(tw)
            if self.gt is not None:
                p = self.gt.pose.pose.position
                if math.dist((p.x, p.y), traj[-1][:2]) > 0.05:
                    traj.append((p.x, p.y, p.z))
                r, pi, _ = quat_to_rpy(self.gt.pose.pose.orientation)
                state['maxp'] = max(state['maxp'], abs(math.degrees(pi)))
                state['maxr'] = max(state['maxr'], abs(math.degrees(r)))

        wall0 = time.time()
        self.spin_for(float(t['duration']), cb=tick)
        wall = time.time() - wall0
        self.cmd.publish(Twist())
        p1 = self.gt.pose.pose.position
        r, pi, yaw = quat_to_rpy(self.gt.pose.pose.orientation)
        dx, dy = p1.x - start[0], p1.y - start[1]
        prog = dx * math.cos(t['yaw']) + dy * math.sin(t['yaw'])
        nominal = float(t['v']) * float(t['duration'])
        maxp, maxr = state['maxp'], state['maxr']
        res = dict(name=t['name'], category=t['category'], progress_m=round(prog, 3),
                   efficiency=round(prog / nominal, 3), lateral_m=round(-dx * math.sin(t['yaw']) + dy * math.cos(t['yaw']), 3),
                   dz_m=round(p1.z - start[2], 3), max_pitch_deg=round(maxp, 1), max_roll_deg=round(maxr, 1),
                   final=[round(p1.x, 3), round(p1.y, 3), round(p1.z, 3)], stuck=prog / nominal < 0.35,
                   flipped=abs(math.degrees(r)) > 60 or abs(math.degrees(pi)) > 60,
                   rtf=round(float(t['duration']) / max(wall, 1e-3), 2),
                   traj=[[round(a, 3) for a in q] for q in traj])
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trials')
    ap.add_argument('out')
    ap.add_argument('--entity', default='ugv_pro_audio')
    ap.add_argument('--z-key', default='z_fixed')
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    trials = json.load(open(a.trials))
    if a.only:
        trials = [t for t in trials if t['name'] in a.only or t['category'] in a.only]
    rclpy.init()
    n = Runner(a.entity)
    # wait for sim clock + gt
    while rclpy.ok() and (n.gt is None or n.now_s() < 1.0):
        rclpy.spin_once(n, timeout_sec=0.2)
    out = []
    for t in trials:
        try:
            r = n.run_trial(t, a.z_key)
        except Exception as e:  # keep going, record the failure
            r = dict(name=t['name'], category=t['category'], error=str(e))
        print(json.dumps({k: v for k, v in r.items() if k != 'traj'}), flush=True)
        out.append(r)
        json.dump(out, open(a.out, 'w'), indent=1)
    n.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
