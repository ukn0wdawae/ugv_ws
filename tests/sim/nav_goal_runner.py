#!/usr/bin/env python3
"""Pooled closed-loop Nav2 goal runner (generalized from docs/forest_traversability/test_harness/nav_goal_test.py).

Sends a sequence of NavigateToPose goals (world XY) to ANY active Nav2 stack, records ground-truth
track, roll/pitch, fall/capsize events, goal success rate, path length, and final position error.

Usage:
  python3 tests/sim/nav_goal_runner.py --out result.json --goals 24.5,6.2 24.0,3.7 [--timeout 180]
"""
import argparse
import json
import math
import time

import rclpy
from action_msgs.msg import GoalStatus
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener


def rp(q):
    roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
    return math.degrees(roll), math.degrees(pitch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--goals', nargs='+', required=True, help='list of x,y map/world coordinates')
    ap.add_argument('--input-frame', default='map', choices=['map', 'odom'],
                    help='frame of --goals (default: map; use odom for legacy world-odom worlds)')
    ap.add_argument('--timeout', type=float, default=180.0, help='per-goal sim-time timeout (s)')
    args = ap.parse_args()
    goals = [tuple(float(v) for v in g.split(',')) for g in args.goals]

    rclpy.init()
    n = Node('nav_goal_runner', parameter_overrides=[
        rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
    track = []
    n.create_subscription(Odometry, '/ugv_gt_odom', lambda m: track.append(
        (m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, m.pose.pose.position.x, m.pose.pose.position.y,
         m.pose.pose.position.z, *rp(m.pose.pose.orientation))), qos_profile_sensor_data)
    tfb = Buffer()
    TransformListener(tfb, n)
    ac = ActionClient(n, NavigateToPose, 'navigate_to_pose')
    if not ac.wait_for_server(timeout_sec=180):
        json.dump({'error': 'no navigate_to_pose server'}, open(args.out, 'w'), indent=2)
        return

    cli = n.create_client(GetState, '/bt_navigator/get_state')
    wall0 = time.time()
    while time.time() - wall0 < 240:
        if cli.wait_for_service(timeout_sec=2.0):
            f = cli.call_async(GetState.Request())
            rclpy.spin_until_future_complete(n, f, timeout_sec=5.0)
            if f.result() is not None and f.result().current_state.label == 'active':
                break
        rclpy.spin_once(n, timeout_sec=0.5)

    # Let costmaps populate
    t_fill = time.time() + 10.0
    while time.time() < t_fill:
        rclpy.spin_once(n, timeout_sec=0.2)

    results = []
    for gx, gy in goals:
        g = NavigateToPose.Goal()
        g.pose.header.frame_id = 'map'
        cx, cy = (track[-1][1], track[-1][2]) if track else (gx, gy)
        yaw = math.atan2(gy - cy, gx - cx)
        if args.input_frame == 'odom':
            try:
                tf = tfb.lookup_transform('map', 'odom', Time())
                q = tf.transform.rotation
                th = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                tx, ty = tf.transform.translation.x, tf.transform.translation.y
            except Exception:
                tx, ty, th = 0.0, 0.0, 0.0
            mx = tx + math.cos(th) * gx - math.sin(th) * gy
            my = ty + math.sin(th) * gx + math.cos(th) * gy
            yaw += th
        else:
            mx, my = gx, gy
        g.pose.pose.position.x, g.pose.pose.position.y = mx, my
        g.pose.pose.orientation.z, g.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)

        i0 = len(track)
        t0 = n.get_clock().now().nanoseconds * 1e-9
        fut = ac.send_goal_async(g)
        rclpy.spin_until_future_complete(n, fut, timeout_sec=20)
        h = fut.result()
        if h is None or not h.accepted:
            results.append({'goal': [gx, gy], 'status': 'REJECTED'})
            continue
        rf = h.get_result_async()
        w0 = time.time()
        while rclpy.ok() and not rf.done():
            rclpy.spin_once(n, timeout_sec=0.2)
            if n.get_clock().now().nanoseconds * 1e-9 - t0 > args.timeout or time.time() - w0 > args.timeout * 3:
                h.cancel_goal_async()
                break
        status = rf.result().status if rf.done() else -1
        seg = track[i0:]
        end = seg[-1] if seg else None
        dist = sum(math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(seg, seg[1:]))
        max_r = max((abs(s[4]) for s in seg), default=0.0)
        max_p = max((abs(s[5]) for s in seg), default=0.0)
        z_drop = (max(s[3] for s in seg) - min(s[3] for s in seg)) if seg else 0.0
        r = {
            'goal': [gx, gy],
            'status': {GoalStatus.STATUS_SUCCEEDED: 'SUCCEEDED', GoalStatus.STATUS_ABORTED: 'ABORTED',
                       GoalStatus.STATUS_CANCELED: 'CANCELED'}.get(status, 'TIMEOUT'),
            'sim_time_s': round(n.get_clock().now().nanoseconds * 1e-9 - t0, 1),
            'final_gt': [round(v, 3) for v in end[1:4]] if end else None,
            'final_error_m': round(math.hypot(end[1] - gx, end[2] - gy), 3) if end else None,
            'path_length_m': round(dist, 2),
            'max_abs_roll_deg': round(max_r, 1),
            'max_abs_pitch_deg': round(max_p, 1),
            'z_span_m': round(z_drop, 2),
            'capsized': max(max_r, max_p) >= 65.0,
        }
        print(json.dumps(r), flush=True)
        results.append(r)

    summary = {
        'goals_total': len(goals),
        'goals_succeeded': sum(1 for r in results if r['status'] == 'SUCCEEDED'),
        'any_capsized': any(r.get('capsized') for r in results),
        'results': results,
    }
    json.dump(summary, open(args.out, 'w'), indent=2)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
