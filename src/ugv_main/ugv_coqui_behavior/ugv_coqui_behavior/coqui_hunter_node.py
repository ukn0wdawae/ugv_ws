#!/usr/bin/env python3
"""Autonomous Coqui Frog Hunter Behavior Node with 3D/2.5D Nav2 & RViz Visualization.

State Machine:
  1. INIT:
     Records the robot's initial 3D position and heading in `map`/`odom` frame so the
     robot always returns home after spotting a Coqui frog.
  2. EXPLORING:
     Explores waypoints around the map (sending goals to 3D/2.5D Nav2 `NavigateToPose`
     with obstacle-aware local controller support) until the 5-channel microphone array
     detects a Coqui frog call.
  3. SOUND_FOLLOWING_AND_TRIANGULATING:
     Locks onto the calling frog, computes diagonal baseline steps from the 3D acoustic
     Direction-of-Arrival (DOA) bearing, and sends sound-based navigation goals to Nav2
     while accumulating multi-position rays to triangulate the frog's 3D position (x, y, z).
  4. APPROACHING_VANTAGE_POINT:
     Evaluates traversability and camera line-of-sight around the triangulated frog using
     the 2.5D/3D Nav2 costmap (`/global_costmap/costmap` and `/map`) and 3D LiDAR (`/scan_cloud`):
       - Enforces a strict obstacle keepout zone around the frog so the robot NEVER drives
         through or over the frog.
       - If the frog is behind a tree or behind an object (bench/rock/log), selects a clear
         vantage point on the unobstructed side of the obstacle and plans around it via Nav2.
       - If the frog is inside an untraversable bush thicket, navigates to the closest
         traversable vantage point outside the bush boundary.
       - If the frog is high on a tree branch, stands back at ~2.15m standoff for upward tilt.
  5. AIMING_AND_CAPTURING_PHOTO:
     With the robot completely stopped at the vantage point, rotates to face the frog,
     aims the 2-axis Pan-Tilt camera (`pt_camera`) at the 3D triangulated coordinates
     (fx, fy, fz), performs OpenCV HSV detection of the small (~4.8cm) Coqui frog,
     captures an annotated photograph saved to `/root/ugv_ws/coqui_captures/` and
     published on `/coqui_behavior/captured_image`, and resets the camera to (0, 0).
  6. RETURNING_TO_INITIAL_POSE:
     Navigates back to the initial position `(x0, y0, yaw0)` via Nav2. Once home, if more
     unvisited frogs remain on the map, resets triangulation and heads out for the next frog;
     otherwise completes the mission.
"""

import json
import math
import os
import time
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from action_msgs.msg import GoalStatus
from cv_bridge import CvBridge
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Twist, Vector3, Vector3Stamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import Image, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from nav2_msgs.action import NavigateToPose
import tf2_ros


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_from_yaw(yaw: float):
    return (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))


class CoquiHunterBehaviorNode(Node):
    def __init__(self):
        super().__init__('coqui_hunter_node')

        self.declare_parameter('use_nav2_if_available', True)
        self.declare_parameter('single_frog_mission', False)
        self.declare_parameter('capture_dir', '/root/ugv_ws/coqui_captures')
        self.declare_parameter('obstacle_stop_dist_m', 0.68)
        self.declare_parameter('frog_keepout_radius_m', 1.05)

        self.use_nav2_pref = bool(self.get_parameter('use_nav2_if_available').value)
        self.single_frog_mission = bool(self.get_parameter('single_frog_mission').value)
        self.capture_dir = self.get_parameter('capture_dir').value
        if not os.path.isdir('/root/ugv_ws') and self.capture_dir.startswith('/root/ugv_ws'):
            self.capture_dir = os.path.abspath('coqui_captures')
        os.makedirs(self.capture_dir, exist_ok=True)
        self.obstacle_stop_dist = float(self.get_parameter('obstacle_stop_dist_m').value)
        self.frog_keepout_radius = float(self.get_parameter('frog_keepout_radius_m').value)

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Robot state
        self.have_pose = False
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_z = 0.0
        self.robot_yaw = 0.0
        self.robot_speed = 0.0

        self.initial_pose = None  # (x0, y0, z0, yaw0)
        self.explore_waypoints = []
        self.current_wp_idx = 0

        # Sensor & perception state
        self.latest_cloud_pts = None
        self.latest_map = None
        self.latest_costmap = None
        self.latest_pt_image = None
        self.latest_front_image = None
        self.sound_detected = False
        self.latest_doa = None
        self.tri_status = None
        self.active_frog_info = {}
        self.pt_state = Vector3()
        self.known_frog_obstacles = []  # list of (fx, fy, fz)

        # Nav2NavigateToPose Action Client
        self.nav2_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.nav2_goal_handle = None
        self.nav2_goal_status = None
        self.nav2_sent_xy = None
        self.nav2_sent_time = 0.0
        self.last_progress_dist = None
        self.last_progress_time = 0.0

        # State machine variables
        self.state = 'INIT'
        self.state_enter_time = time.monotonic()
        self.baseline_maneuver_stage = 0
        self.baseline_target_xy = None
        self.vantage_target = None  # (vx, vy, vyaw, standoff_m)
        self.vantage_reach_status = 'TRAVERSABLE_VANTAGE_REACHED'
        self.current_goal_label = 'Initializing'
        self.current_goal_xy_yaw = None
        self.aim_substage = 0
        self.aim_timer_start = 0.0
        self.scan_offsets = [(0.0, 0.0), (-0.14, 0.0), (0.14, 0.0), (0.0, 0.10), (0.0, -0.10)]
        self.scan_idx = 0
        self.capture_count = 0

        # Subscriptions
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.create_subscription(PointCloud2, '/scan_cloud', self._scan_cloud_cb, 10)
        self.create_subscription(OccupancyGrid, '/map', self._map_cb, 10)
        self.create_subscription(OccupancyGrid, '/global_costmap/costmap', self._costmap_cb, 10)
        self.create_subscription(Bool, '/ugv_audio/sound_detected', self._sound_det_cb, 10)
        self.create_subscription(Vector3Stamped, '/ugv_audio/doa_bearing', self._doa_cb, 10)
        self.create_subscription(String, '/ugv_audio/triangulation_status', self._tri_status_cb, 10)
        self.create_subscription(String, '/coqui/active_frog_info', self._frog_info_cb, 10)
        self.create_subscription(Image, '/pt_camera/image_raw', self._pt_image_cb, 10)
        self.create_subscription(Image, '/camera/image_raw', self._front_image_cb, 10)
        self.create_subscription(Vector3, '/coqui_behavior/pan_tilt_state', self._pt_state_cb, 10)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.explore_resume_pub = self.create_publisher(Bool, '/explore/resume', 10)
        self.lock_caller_pub = self.create_publisher(Bool, '/coqui/lock_active_caller', 10)
        self.reset_tri_pub = self.create_publisher(String, '/coqui/reset_triangulation', 10)
        self.look_at_pub = self.create_publisher(PointStamped, '/coqui_behavior/look_at_point', 10)
        self.pt_cmd_pub = self.create_publisher(Vector3, '/coqui_behavior/pan_tilt_cmd', 10)
        self.captured_img_pub = self.create_publisher(Image, '/coqui_behavior/captured_image', 10)
        self.report_pub = self.create_publisher(String, '/coqui_behavior/detection_report', 10)

        # RViz Goal & Path Visualization Publishers
        self.nav_goal_pub = self.create_publisher(PoseStamped, '/coqui_behavior/sound_nav_goal', 10)
        self.planned_path_pub = self.create_publisher(Path, '/coqui_behavior/planned_path', 10)
        self.viz_marker_pub = self.create_publisher(MarkerArray, '/coqui_behavior/visualization_markers', 10)

        self.timer = self.create_timer(0.1, self._tick)
        self.get_logger().info('Coqui Hunter Behavior Node (with 3D/2.5D Nav2 & RViz Visualization) ready.')

    # -------------------- Callbacks --------------------
    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        self.robot_speed = math.hypot(v.x, v.y)
        try:
            tf = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time(), timeout=Duration(seconds=0.02)
            )
            t = tf.transform.translation
            r = tf.transform.rotation
            self.robot_x = float(t.x)
            self.robot_y = float(t.y)
            self.robot_z = float(t.z)
            self.robot_yaw = yaw_from_quat(r.x, r.y, r.z, r.w)
        except Exception:
            self.robot_x = float(p.x)
            self.robot_y = float(p.y)
            self.robot_z = float(p.z)
            self.robot_yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        self.have_pose = True

    def _scan_cloud_cb(self, msg: PointCloud2):
        try:
            step = msg.point_step // 4
            if step >= 3 and len(msg.data) >= msg.point_step:
                arr = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, step)[:, :3]
                valid = (
                    np.isfinite(arr[:, 0])
                    & np.isfinite(arr[:, 1])
                    & np.isfinite(arr[:, 2])
                    & (arr[:, 2] >= 0.02)
                    & (arr[:, 2] <= 1.20)
                )
                pts = arr[valid]
                if len(pts) > 0:
                    r = np.hypot(pts[:, 0], pts[:, 1])
                    mask = (r >= 0.22) & (r <= 30.0)
                    pts = pts[mask]
                    r = r[mask]
                    ang = np.arctan2(pts[:, 1], pts[:, 0])
                    self.latest_cloud_pts = (r, ang)
                else:
                    self.latest_cloud_pts = (np.empty(0), np.empty(0))
        except Exception:
            pass

    def _map_cb(self, msg: OccupancyGrid):
        self.latest_map = msg

    def _costmap_cb(self, msg: OccupancyGrid):
        self.latest_costmap = msg

    def _sound_det_cb(self, msg: Bool):
        self.sound_detected = bool(msg.data)

    def _doa_cb(self, msg: Vector3Stamped):
        self.latest_doa = msg

    def _tri_status_cb(self, msg: String):
        try:
            self.tri_status = json.loads(msg.data)
        except Exception:
            pass

    def _frog_info_cb(self, msg: String):
        try:
            self.active_frog_info = json.loads(msg.data)
        except Exception:
            pass

    def _pt_image_cb(self, msg: Image):
        self.latest_pt_image = msg

    def _front_image_cb(self, msg: Image):
        self.latest_front_image = msg

    def _pt_state_cb(self, msg: Vector3):
        self.pt_state = msg

    # -------------------- Nav2 & Obstacle-Aware Navigation --------------------
    def _transition(self, new_state: str):
        self.get_logger().info(f'[STATE] {self.state} -> {new_state}')
        self.state = new_state
        self.state_enter_time = time.monotonic()
        self.nav2_sent_xy = None
        self.last_progress_dist = None

    def _stop_robot(self):
        self.cmd_vel_pub.publish(Twist())
        if self.nav2_goal_handle is not None:
            try:
                self.nav2_goal_handle.cancel_goal_async()
            except Exception:
                pass
            self.nav2_goal_handle = None
        self.nav2_sent_xy = None

    def _send_nav2_goal(self, tx: float, ty: float, tyaw: float, label: str):
        """Publish the sound-based goal to RViz and send `NavigateToPose` to 3D/2.5D Nav2."""
        self.current_goal_label = label
        self.current_goal_xy_yaw = (tx, ty, tyaw)

        now_msg = self.get_clock().now().to_msg()
        qx, qy, qz, qw = quat_from_yaw(tyaw)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = now_msg
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = float(tx)
        pose_msg.pose.position.y = float(ty)
        pose_msg.pose.position.z = float(self.robot_z)
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw
        self.nav_goal_pub.publish(pose_msg)

        # Publish curved/standoff planned path preview for RViz
        self._publish_planned_path_preview(tx, ty)

        if not self.use_nav2_pref:
            return
        if not self.nav2_client.server_is_ready():
            return

        # Only send a new action goal to Nav2 if target changed by > 0.22m
        if self.nav2_sent_xy is not None:
            if math.hypot(tx - self.nav2_sent_xy[0], ty - self.nav2_sent_xy[1]) <= 0.22:
                return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose_msg
        self.nav2_sent_xy = (tx, ty)
        self.nav2_sent_time = time.monotonic()
        self.nav2_goal_status = None

        future = self.nav2_client.send_goal_async(goal_msg)
        future.add_done_callback(self._nav2_goal_response_cb)
        self.get_logger().info(
            f"[Nav2 Goal Sent] {label}: ({tx:.2f}, {ty:.2f}, yaw={math.degrees(tyaw):.0f} deg)"
        )

    def _nav2_goal_response_cb(self, future):
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.nav2_goal_handle = None
                return
            self.nav2_goal_handle = goal_handle
            res_future = goal_handle.get_result_async()
            res_future.add_done_callback(self._nav2_result_cb)
        except Exception:
            self.nav2_goal_handle = None

    def _nav2_result_cb(self, future):
        try:
            self.nav2_goal_status = future.result().status
        except Exception:
            self.nav2_goal_status = GoalStatus.STATUS_ABORTED

    def _publish_planned_path_preview(self, tx: float, ty: float):
        """Publish an obstacle-aware path from robot to (tx, ty) on `/coqui_behavior/planned_path`."""
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'map'
        steps = 12
        # Check if a triangulated frog or obstacle sits near the straight line and curve around it
        obs_list = list(self.known_frog_obstacles)
        if self.tri_status is not None and 'triangulated_pos' in self.tri_status:
            tp = self.tri_status['triangulated_pos']
            obs_list.append((float(tp[0]), float(tp[1]), float(tp[2])))

        for i in range(steps + 1):
            s = i / float(steps)
            px = self.robot_x + s * (tx - self.robot_x)
            py = self.robot_y + s * (ty - self.robot_y)
            # Push intermediate path points outside frog keepout radius
            if 0 < i < steps:
                for ox, oy, _oz in obs_list:
                    d_obs = math.hypot(px - ox, py - oy)
                    if d_obs < self.frog_keepout_radius and d_obs > 1e-3:
                        push = (self.frog_keepout_radius - d_obs) * math.sin(math.pi * s)
                        px += push * ((px - ox) / d_obs)
                        py += push * ((py - oy) / d_obs)
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(px)
            ps.pose.position.y = float(py)
            ps.pose.position.z = float(self.robot_z + 0.05)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.planned_path_pub.publish(path)

    def _get_sector_min_range(self, center_angle_rad: float, half_width_rad: float = 0.45) -> float:
        """Return minimum valid 3D LiDAR obstacle range in [center - half_width, center + half_width]."""
        if self.latest_cloud_pts is None:
            return 10.0
        r, ang = self.latest_cloud_pts
        if len(r) == 0:
            return 10.0
        diff = np.arctan2(np.sin(ang - center_angle_rad), np.cos(ang - center_angle_rad))
        in_sector = r[np.abs(diff) <= half_width_rad]
        if len(in_sector) == 0:
            return 10.0
        return float(np.min(in_sector))

    def _is_cell_traversable_in_map(self, x: float, y: float, clearance_m: float = 0.38) -> bool:
        """Check both `/global_costmap/costmap` and `/map` so we never place a goal inside an obstacle."""
        # Also enforce keepout around all known/triangulated frogs
        for ox, oy, _oz in self.known_frog_obstacles:
            if math.hypot(x - ox, y - oy) < self.frog_keepout_radius:
                return False

        for grid in (self.latest_costmap, self.latest_map):
            if grid is None:
                continue
            info = grid.info
            res = info.resolution
            if res <= 0.0:
                continue
            cx = int((x - info.origin.position.x) / res)
            cy = int((y - info.origin.position.y) / res)
            rad_cells = max(1, int(clearance_m / res))
            w, h = info.width, info.height
            data = grid.data
            thresh = 75 if grid is self.latest_costmap else 55
            for dy in range(-rad_cells, rad_cells + 1):
                for dx in range(-rad_cells, rad_cells + 1):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        val = data[ny * w + nx]
                        if val >= thresh:
                            return False
        return True

    def _has_clear_line_of_sight_in_map(self, vx: float, vy: float, fx: float, fy: float) -> bool:
        """Raycast in `/map` from vantage `(vx, vy)` toward frog `(fx, fy)` (stopping 0.42m before frog)
        to verify no tree trunk, bench, rock, or log blocks the camera view."""
        if self.latest_map is None:
            return True
        info = self.latest_map.info
        res = info.resolution
        if res <= 0.0:
            return True
        dist = math.hypot(fx - vx, fy - vy)
        if dist <= 0.55:
            return True
        max_s = max(0.1, (dist - 0.42) / dist)
        w, h = info.width, info.height
        data = self.latest_map.data
        for s in np.linspace(0.15, max_s, 10):
            px = vx + s * (fx - vx)
            py = vy + s * (fy - vy)
            cx = int((px - info.origin.position.x) / res)
            cy = int((py - info.origin.position.y) / res)
            if 0 <= cx < w and 0 <= cy < h:
                if data[cy * w + cx] >= 60:
                    return False
        return True

    def _navigate_toward_goal(self, tx: float, ty: float, tyaw: float, stop_tol: float, label: str) -> str:
        """Send goal to 3D/2.5D Nav2 (`NavigateToPose`) and assist with obstacle-aware control if needed."""
        dx = tx - self.robot_x
        dy = ty - self.robot_y
        dist = math.hypot(dx, dy)
        if dist <= stop_tol:
            self._stop_robot()
            return 'REACHED'

        # 1. Send / update goal on Nav2 & RViz
        self._send_nav2_goal(tx, ty, tyaw, label)

        # Check front & flank LiDAR clearance
        front_r = self._get_sector_min_range(0.0, 0.42)
        left_r = self._get_sector_min_range(0.58, 0.42)
        right_r = self._get_sector_min_range(-0.58, 0.42)
        desired_yaw = math.atan2(dy, dx)
        yaw_err = normalize_angle(desired_yaw - self.robot_yaw)

        # If close to target (< 2.0m) and blocked by an untraversable bush or obstacle directly ahead
        if dist < 2.0 and front_r < self.obstacle_stop_dist and abs(yaw_err) < 0.55:
            self._stop_robot()
            return 'BLOCKED'

        # Check if Nav2 is actively driving the robot (`robot_speed > 0.04`)
        nav2_active = (
            self.use_nav2_pref
            and self.nav2_goal_handle is not None
            and self.nav2_goal_status not in (GoalStatus.STATUS_ABORTED, GoalStatus.STATUS_CANCELED)
            and (time.monotonic() - self.nav2_sent_time < 2.5 or self.robot_speed > 0.04)
            and front_r >= self.obstacle_stop_dist
        )
        if nav2_active:
            return 'MOVING'

        # Closed-loop obstacle- & frog-keepout-aware local velocity control (with repulsive field around frogs)
        repulse_yaw = 0.0
        if left_r < 0.90:
            repulse_yaw -= 0.55 * (0.90 - left_r)
        if right_r < 0.90:
            repulse_yaw += 0.55 * (0.90 - right_r)

        # Repulsive steering away from triangulated frog keepout zone so robot NEVER drives through frog
        if self.tri_status is not None and 'triangulated_pos' in self.tri_status:
            tp = self.tri_status['triangulated_pos']
            dfx = float(tp[0]) - self.robot_x
            dfy = float(tp[1]) - self.robot_y
            d_frog = math.hypot(dfx, dfy)
            if d_frog < self.frog_keepout_radius + 0.35:
                ang_to_frog = normalize_angle(math.atan2(dfy, dfx) - self.robot_yaw)
                if abs(ang_to_frog) < 1.1:
                    repulse_yaw += (-0.85 if ang_to_frog >= 0.0 else 0.85)

        cmd = Twist()
        if abs(yaw_err) > 0.55:
            cmd.linear.x = 0.0
            cmd.angular.z = float(np.clip(1.35 * yaw_err, -0.70, 0.70))
        elif front_r < self.obstacle_stop_dist:
            steer_dir = 1.0 if left_r >= right_r else -1.0
            cmd.linear.x = 0.04
            cmd.angular.z = 0.65 * steer_dir
        else:
            cmd.linear.x = float(np.clip(0.30 * dist, 0.12, 0.34))
            cmd.angular.z = float(np.clip(1.15 * yaw_err + repulse_yaw, -0.68, 0.68))

        self.cmd_vel_pub.publish(cmd)
        return 'MOVING'

    def _rotate_to_yaw(self, target_yaw: float, tol_rad: float = 0.08) -> bool:
        err = normalize_angle(target_yaw - self.robot_yaw)
        if abs(err) <= tol_rad:
            self.cmd_vel_pub.publish(Twist())
            return True
        cmd = Twist()
        cmd.angular.z = float(np.clip(1.45 * err, -0.62, 0.62))
        if abs(cmd.angular.z) < 0.16:
            cmd.angular.z = math.copysign(0.16, err)
        self.cmd_vel_pub.publish(cmd)
        return False

    def _compute_best_vantage_point(self, fx: float, fy: float, fz: float):
        """Compute the best traversable, line-of-sight-clear standoff vantage point around `(fx, fy, fz)`."""
        elev_above_robot = fz - self.robot_z
        placement = self.active_frog_info.get('placement', 'open')
        hint_angle = self.active_frog_info.get('vantage_hint_angle', None)
        cfg_standoff = float(self.active_frog_info.get('standoff_m', 1.55))

        if elev_above_robot > 0.90 or placement == 'on_tree':
            preferred_standoff = max(2.15, cfg_standoff)
        elif placement == 'inside_bush':
            preferred_standoff = max(1.85, cfg_standoff)
        else:
            preferred_standoff = max(1.48, cfg_standoff)

        approach_angle = math.atan2(self.robot_y - fy, self.robot_x - fx)
        # If the frog is behind a tree or behind an object, prioritize the unobstructed side angle!
        if placement in ('behind_tree', 'behind_object') and hint_angle is not None:
            primary_angle = float(hint_angle)
        else:
            primary_angle = approach_angle

        angle_offsets = [0.0, 0.35, -0.35, 0.70, -0.70, 1.05, -1.05, 1.45, -1.45, 1.85, -1.85, 2.35, -2.35, 3.14]
        radii = [preferred_standoff, preferred_standoff + 0.40, preferred_standoff + 0.80]

        best_fallback = None
        for r in radii:
            for dtheta in angle_offsets:
                ang = normalize_angle(primary_angle + dtheta)
                vx = fx + r * math.cos(ang)
                vy = fy + r * math.sin(ang)
                if self._is_cell_traversable_in_map(vx, vy, clearance_m=0.38):
                    vyaw = math.atan2(fy - vy, fx - vx)
                    if best_fallback is None:
                        best_fallback = (vx, vy, vyaw, r)
                    if self._has_clear_line_of_sight_in_map(vx, vy, fx, fy):
                        return vx, vy, vyaw, r

        if best_fallback is not None:
            return best_fallback

        vx = fx + preferred_standoff * math.cos(primary_angle)
        vy = fy + preferred_standoff * math.sin(primary_angle)
        vyaw = math.atan2(fy - vy, fx - vx)
        return vx, vy, vyaw, preferred_standoff

    # -------------------- RViz Visualization Markers --------------------
    def _publish_rviz_visualization(self):
        """Publish rich 3D markers on `/coqui_behavior/visualization_markers` explaining sound goals & Nav2."""
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()

        # 1. Floating Robot Status HUD above robot
        hud = Marker()
        hud.header.frame_id = 'map'
        hud.header.stamp = now
        hud.ns = 'robot_status_hud'
        hud.id = 0
        hud.type = Marker.TEXT_VIEW_FACING
        hud.action = Marker.ADD
        hud.pose.position.x = float(self.robot_x)
        hud.pose.position.y = float(self.robot_y)
        hud.pose.position.z = float(self.robot_z) + 0.65
        hud.pose.orientation.w = 1.0
        hud.scale.z = 0.22
        hud.color.r, hud.color.g, hud.color.b, hud.color.a = (1.0, 1.0, 1.0, 0.98)
        doa_str = ''
        if self.tri_status is not None:
            doa_str = f" | DOA: {self.tri_status.get('azimuth_deg', 0):.0f} deg, El: {self.tri_status.get('elevation_deg', 0):.0f} deg"
        hud.text = f"[{self.state}]{doa_str}\n{self.current_goal_label}"
        ma.markers.append(hud)

        # 2. Current Sound-Based Nav2 Goal Arrow & Beacon
        if self.current_goal_xy_yaw is not None:
            gx, gy, gyaw = self.current_goal_xy_yaw
            qx, qy, qz, qw = quat_from_yaw(gyaw)

            arrow = Marker()
            arrow.header.frame_id = 'map'
            arrow.header.stamp = now
            arrow.ns = 'sound_nav_goal'
            arrow.id = 1
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.pose.position.x = float(gx)
            arrow.pose.position.y = float(gy)
            arrow.pose.position.z = float(self.robot_z) + 0.12
            arrow.pose.orientation.x = qx
            arrow.pose.orientation.y = qy
            arrow.pose.orientation.z = qz
            arrow.pose.orientation.w = qw
            arrow.scale.x = 0.65
            arrow.scale.y = 0.14
            arrow.scale.z = 0.14
            arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = (1.0, 0.15, 0.85, 0.95)
            ma.markers.append(arrow)

            beacon = Marker()
            beacon.header.frame_id = 'map'
            beacon.header.stamp = now
            beacon.ns = 'sound_nav_goal_beacon'
            beacon.id = 2
            beacon.type = Marker.CYLINDER
            beacon.action = Marker.ADD
            beacon.pose.position.x = float(gx)
            beacon.pose.position.y = float(gy)
            beacon.pose.position.z = float(self.robot_z) + 0.45
            beacon.pose.orientation.w = 1.0
            beacon.scale.x = 0.12
            beacon.scale.y = 0.12
            beacon.scale.z = 0.90
            beacon.color.r, beacon.color.g, beacon.color.b, beacon.color.a = (1.0, 0.2, 0.9, 0.45)
            ma.markers.append(beacon)

            gtxt = Marker()
            gtxt.header.frame_id = 'map'
            gtxt.header.stamp = now
            gtxt.ns = 'sound_nav_goal_label'
            gtxt.id = 3
            gtxt.type = Marker.TEXT_VIEW_FACING
            gtxt.action = Marker.ADD
            gtxt.pose.position.x = float(gx)
            gtxt.pose.position.y = float(gy)
            gtxt.pose.position.z = float(self.robot_z) + 1.05
            gtxt.pose.orientation.w = 1.0
            gtxt.scale.z = 0.20
            gtxt.color.r, gtxt.color.g, gtxt.color.b, gtxt.color.a = (1.0, 0.4, 0.95, 0.95)
            gtxt.text = f"Nav2 Goal: {self.current_goal_label}\n({gx:.1f}, {gy:.1f})"
            ma.markers.append(gtxt)

        # 3. Standoff Ring & Obstacle Keepout Ring around Triangulated Frog
        if self.tri_status is not None and 'triangulated_pos' in self.tri_status:
            tp = self.tri_status['triangulated_pos']
            fx, fy, fz = float(tp[0]), float(tp[1]), float(tp[2])

            keepout = Marker()
            keepout.header.frame_id = 'map'
            keepout.header.stamp = now
            keepout.ns = 'frog_keepout_zone'
            keepout.id = 4
            keepout.type = Marker.CYLINDER
            keepout.action = Marker.ADD
            keepout.pose.position.x = fx
            keepout.pose.position.y = fy
            keepout.pose.position.z = float(self.robot_z) + 0.02
            keepout.pose.orientation.w = 1.0
            keepout.scale.x = self.frog_keepout_radius * 2.0
            keepout.scale.y = self.frog_keepout_radius * 2.0
            keepout.scale.z = 0.04
            keepout.color.r, keepout.color.g, keepout.color.b, keepout.color.a = (1.0, 0.25, 0.1, 0.28)
            ma.markers.append(keepout)

            # Camera line-of-sight vector from robot/vantage to 3D frog
            los = Marker()
            los.header.frame_id = 'map'
            los.header.stamp = now
            los.ns = 'camera_sightline'
            los.id = 5
            los.type = Marker.LINE_STRIP
            los.action = Marker.ADD
            los.scale.x = 0.03
            los.color.r, los.color.g, los.color.b, los.color.a = (1.0, 0.95, 0.1, 0.85)
            los.points.append(Point(x=float(self.robot_x), y=float(self.robot_y), z=float(self.robot_z + 0.21)))
            los.points.append(Point(x=fx, y=fy, z=fz))
            ma.markers.append(los)

        self.viz_marker_pub.publish(ma)

    # -------------------- Visual Frog Detection & Photo Capture --------------------
    def _detect_coqui_visually(self, bgr_img: np.ndarray):
        """Detect the small (~4.8cm) Coqui frog (vivid lime-green body + golden-yellow vocal sac) in BGR image."""
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv, np.array([36, 120, 100]), np.array([80, 255, 255]))
        mask_sac = cv2.inRange(hsv, np.array([16, 150, 150]), np.array([36, 255, 255]))

        kernel = np.ones((7, 7), np.uint8)
        sac_dilated = cv2.dilate(mask_sac, kernel, iterations=2)
        combined = cv2.bitwise_or(mask_green, mask_sac)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_cnt = None
        best_score = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 8 or area > 40000:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            roi_sac = sac_dilated[y:y + h, x:x + w]
            roi_grn = mask_green[y:y + h, x:x + w]
            sac_pixels = int(np.count_nonzero(roi_sac))
            grn_pixels = int(np.count_nonzero(roi_grn))
            if grn_pixels >= 5 and sac_pixels >= 3:
                score = float(area) + 3.0 * sac_pixels
                if score > best_score:
                    best_score = score
                    best_cnt = (x, y, w, h, int(area))

        if best_cnt is not None:
            x, y, w, h, area = best_cnt
            return True, (x, y, w, h), (x + w // 2, y + h // 2), area
        return False, None, None, 0

    def _capture_and_save_photo(self, fx: float, fy: float, fz: float, reached_status: str):
        """Capture image from `/pt_camera/image_raw` (or `/camera/image_raw`), annotate, and save."""
        img_msg = self.latest_pt_image if self.latest_pt_image is not None else self.latest_front_image
        if img_msg is not None:
            frame = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8').copy()
        else:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        vis_detected, bbox, _center, _area = self._detect_coqui_visually(frame)
        dist_2d = math.hypot(fx - self.robot_x, fy - self.robot_y)
        pan_deg = math.degrees(self.pt_state.x)
        tilt_deg = math.degrees(self.pt_state.y)
        placement = self.active_frog_info.get('placement', 'open')
        frog_name = self.active_frog_info.get('frog_name', 'coqui_frog')

        h, w = frame.shape[:2]
        cv2.line(frame, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), (200, 200, 200), 1)
        cv2.line(frame, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20), (200, 200, 200), 1)

        if vis_detected and bbox is not None:
            bx, by, bw, bh = bbox
            pad = 6
            cv2.rectangle(
                frame,
                (max(0, bx - pad), max(0, by - pad)),
                (min(w - 1, bx + bw + pad), min(h - 1, by + bh + pad)),
                (0, 255, 0), 2
            )
            cv2.putText(
                frame, f'COQUI FROG ({placement.upper()})', (max(5, bx - 10), max(22, by - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2
            )
            status_banner = f'PHOTO CAPTURED: FROG SPOTTED ({placement.upper()})'
            banner_color = (0, 220, 0)
        else:
            status_banner = (
                f'TRIANGULATED AT ({fx:.1f}, {fy:.1f}, {fz:.1f}) - '
                f'OCCLUDED / UNTRAVERSABLE ({placement.upper()})'
            )
            banner_color = (0, 180, 255)

        cv2.rectangle(frame, (0, 0), (w, 68), (20, 20, 20), -1)
        cv2.putText(frame, status_banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, banner_color, 2)
        cv2.putText(
            frame,
            f'Triangulated 3D: ({fx:.2f}, {fy:.2f}, {fz:.2f})m | Standoff: {dist_2d:.2f}m | '
            f'PTZ Pan: {pan_deg:+.1f} deg, Tilt: {tilt_deg:+.1f} deg',
            (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1
        )

        self.capture_count += 1
        env_name = self.active_frog_info.get('env', 'env')
        filename = f'coqui_{env_name}_{placement}_{self.capture_count}.png'
        filepath = os.path.join(self.capture_dir, filename)
        cv2.imwrite(filepath, frame)

        latest_path = os.path.join(self.capture_dir, 'latest_coqui_capture.png')
        cv2.imwrite(latest_path, frame)

        out_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.header.frame_id = 'pt_camera_link'
        self.captured_img_pub.publish(out_msg)

        # Record this frog position as a permanent keepout obstacle
        self.known_frog_obstacles.append((fx, fy, fz))

        report = {
            'event': 'COQUI_CAPTURE_COMPLETE',
            'env': env_name,
            'frog_name': frog_name,
            'placement': placement,
            'visually_confirmed_in_camera': bool(vis_detected),
            'vantage_reachability': reached_status,
            'triangulated_3d_map': [round(fx, 3), round(fy, 3), round(fz, 3)],
            'robot_vantage_pose': [round(self.robot_x, 3), round(self.robot_y, 3), round(self.robot_yaw, 3)],
            'camera_pan_deg': round(pan_deg, 1),
            'camera_tilt_deg': round(tilt_deg, 1),
            'saved_image_path': filepath,
        }
        rep_msg = String()
        rep_msg.data = json.dumps(report)
        self.report_pub.publish(rep_msg)

        self.get_logger().info(
            f"[CAPTURE #{self.capture_count}] Saved Coqui image to {filepath} | "
            f"frog={frog_name} ({placement}), visual_confirmed={vis_detected}, "
            f"vantage={reached_status}, pan={pan_deg:+.1f} deg, tilt={tilt_deg:+.1f} deg"
        )

    # -------------------- Main State Machine Tick --------------------
    def _tick(self):
        if not self.have_pose:
            return

        now = time.monotonic()
        self._publish_rviz_visualization()

        # ==================== 1. INIT ====================
        if self.state == 'INIT':
            self.initial_pose = (self.robot_x, self.robot_y, self.robot_z, self.robot_yaw)
            x0, y0, _z0, yaw0 = self.initial_pose
            fwd_x, fwd_y = math.cos(yaw0), math.sin(yaw0)
            lft_x, lft_y = -math.sin(yaw0), math.cos(yaw0)
            self.explore_waypoints = [
                (x0 + 2.6 * fwd_x + 1.4 * lft_x, y0 + 2.6 * fwd_y + 1.4 * lft_y),
                (x0 + 4.4 * fwd_x - 1.4 * lft_x, y0 + 4.4 * fwd_y - 1.4 * lft_y),
                (x0 + 6.0 * fwd_x + 1.6 * lft_x, y0 + 6.0 * fwd_y + 1.6 * lft_y),
                (x0 + 3.2 * fwd_x - 2.4 * lft_x, y0 + 3.2 * fwd_y - 2.4 * lft_y),
            ]
            self.current_wp_idx = 0
            self.pt_cmd_pub.publish(Vector3(x=0.0, y=0.0, z=0.0))
            self.explore_resume_pub.publish(Bool(data=True))
            self.get_logger().info(
                f"Recorded initial pose: x={x0:.2f}, y={y0:.2f}, yaw={yaw0:.2f}. Starting exploration..."
            )
            self._transition('EXPLORING')
            return

        # ==================== 2. EXPLORING ====================
        if self.state == 'EXPLORING':
            if self.sound_detected and self.tri_status is not None:
                self._stop_robot()
                self.explore_resume_pub.publish(Bool(data=False))
                self.lock_caller_pub.publish(Bool(data=True))
                self.baseline_maneuver_stage = 0
                self.baseline_target_xy = None
                self.get_logger().info(
                    f"Heard Coqui call ({self.active_frog_info.get('frog_name', 'frog')} - "
                    f"{self.active_frog_info.get('placement', 'unknown')})! "
                    f"Initial DOA={self.tri_status.get('azimuth_deg')} deg, SNR={self.tri_status.get('snr_db')} dB. "
                    f"Following sound & triangulating..."
                )
                self._transition('SOUND_FOLLOWING_AND_TRIANGULATING')
                return

            wp_x, wp_y = self.explore_waypoints[self.current_wp_idx]
            wp_yaw = math.atan2(wp_y - self.robot_y, wp_x - self.robot_x)
            status = self._navigate_toward_goal(
                wp_x, wp_y, wp_yaw, stop_tol=0.45,
                label=f'EXPLORE WAYPOINT #{self.current_wp_idx + 1}'
            )
            if status in ('REACHED', 'BLOCKED') or (now - self.state_enter_time) > 12.0:
                self.current_wp_idx = (self.current_wp_idx + 1) % len(self.explore_waypoints)
                self.state_enter_time = now
                self.nav2_sent_xy = None
            return

        # ==================== 3. SOUND_FOLLOWING_AND_TRIANGULATING ====================
        if self.state == 'SOUND_FOLLOWING_AND_TRIANGULATING':
            if self.tri_status is None:
                return
            tri_pos = self.tri_status.get('triangulated_pos', [self.robot_x, self.robot_y, self.robot_z])
            fx, fy, fz = float(tri_pos[0]), float(tri_pos[1]), float(tri_pos[2])

            converged = bool(self.tri_status.get('converged', False))
            num_rays = int(self.tri_status.get('num_rays', 0))
            baseline = float(self.tri_status.get('baseline_m', 0.0))

            if converged or (num_rays >= 4 and baseline >= 0.45) or (now - self.state_enter_time > 13.0):
                self._stop_robot()
                vx, vy, vyaw, standoff = self._compute_best_vantage_point(fx, fy, fz)
                self.vantage_target = (vx, vy, vyaw, standoff)
                placement = self.active_frog_info.get('placement', 'open')
                self.get_logger().info(
                    f"3D Triangulation locked on '{placement}' frog at ({fx:.2f}, {fy:.2f}, {fz:.2f})m "
                    f"(rays={num_rays}, baseline={baseline:.2f}m, unc={self.tri_status.get('uncertainty_m')}m). "
                    f"Sending 3D/2.5D Nav2 vantage goal ({vx:.2f}, {vy:.2f}) at standoff={standoff:.2f}m..."
                )
                self._transition('APPROACHING_VANTAGE_POINT')
                return

            # Compute a baseline step offset (+28 deg or -28 deg from acoustic bearing)
            # while staying strictly outside the frog keepout zone
            if self.baseline_target_xy is None:
                bearing_world = math.atan2(fy - self.robot_y, fx - self.robot_x)
                dist_to_frog = math.hypot(fx - self.robot_x, fy - self.robot_y)
                step_len = min(1.35, max(0.55, dist_to_frog - 1.65))
                offset_ang = 0.50 if self.baseline_maneuver_stage % 2 == 0 else -0.50
                step_angle = bearing_world + offset_ang
                bx = self.robot_x + step_len * math.cos(step_angle)
                by = self.robot_y + step_len * math.sin(step_angle)
                self.baseline_target_xy = (bx, by, bearing_world)

            bx, by, byaw = self.baseline_target_xy
            res = self._navigate_toward_goal(
                bx, by, byaw, stop_tol=0.32,
                label=f'SOUND BASELINE STEP #{self.baseline_maneuver_stage + 1} (3D DOA Triangulation)'
            )
            if res in ('REACHED', 'BLOCKED'):
                self.baseline_maneuver_stage += 1
                self.baseline_target_xy = None
            return

        # ==================== 4. APPROACHING_VANTAGE_POINT ====================
        if self.state == 'APPROACHING_VANTAGE_POINT':
            tri_pos = self.tri_status.get('triangulated_pos', [self.robot_x, self.robot_y, self.robot_z])
            fx, fy, fz = float(tri_pos[0]), float(tri_pos[1]), float(tri_pos[2])

            vx, vy, vyaw, standoff = self._compute_best_vantage_point(fx, fy, fz)
            self.vantage_target = (vx, vy, vyaw, standoff)
            placement = self.active_frog_info.get('placement', 'open')

            res = self._navigate_toward_goal(
                vx, vy, vyaw, stop_tol=0.38,
                label=f'VANTAGE GOAL ({placement.upper()}, standoff={standoff:.2f}m)'
            )
            if res == 'REACHED':
                self._stop_robot()
                self.vantage_reach_status = (
                    'STOPPED_OUTSIDE_UNTRAVERSABLE_BUSH' if placement == 'inside_bush'
                    else 'TRAVERSABLE_VANTAGE_REACHED'
                )
                self.aim_substage = 0
                self.scan_idx = 0
                self._transition('AIMING_AND_CAPTURING_PHOTO')
                return
            elif res == 'BLOCKED' or (now - self.state_enter_time > 22.0):
                self._stop_robot()
                self.vantage_reach_status = 'STOPPED_AT_UNTRAVERSABLE_OBSTACLE_BOUNDARY'
                self.aim_substage = 0
                self.scan_idx = 0
                self.get_logger().info(
                    'Stopped at closest traversable boundary outside obstacle/bush; aiming PTZ camera...'
                )
                self._transition('AIMING_AND_CAPTURING_PHOTO')
                return
            return

        # ==================== 5. AIMING_AND_CAPTURING_PHOTO ====================
        if self.state == 'AIMING_AND_CAPTURING_PHOTO':
            # Keep wheels completely still while panning/tilting camera
            self.cmd_vel_pub.publish(Twist())

            tri_pos = self.tri_status.get('triangulated_pos', [self.robot_x, self.robot_y, self.robot_z])
            fx, fy, fz = float(tri_pos[0]), float(tri_pos[1]), float(tri_pos[2])

            # Substage 0: Rotate chassis in place to face the frog before tilting PTZ camera
            if self.aim_substage == 0:
                target_yaw = math.atan2(fy - self.robot_y, fx - self.robot_x)
                self.current_goal_label = f'ORIENTING CHASSIS TOWARD FROG ({math.degrees(target_yaw):.0f} deg)'
                aligned = self._rotate_to_yaw(target_yaw, tol_rad=0.09)
                if aligned or (now - self.state_enter_time > 5.5):
                    self._stop_robot()
                    self.aim_substage = 1
                    self.aim_timer_start = now
                return

            # Substage 1: Point 2-axis Pan-Tilt camera at triangulated 3D frog location (fx, fy, fz)
            if self.aim_substage == 1:
                self.current_goal_label = f'AIMING PTZ CAMERA AT ({fx:.1f}, {fy:.1f}, {fz:.1f})'
                d_pan, d_tilt = self.scan_offsets[self.scan_idx]
                pt_msg = PointStamped()
                pt_msg.header.stamp = self.get_clock().now().to_msg()
                pt_msg.header.frame_id = 'map'
                dist_xy = math.hypot(fx - self.robot_x, fy - self.robot_y)
                perp_x = -(fy - self.robot_y) / max(dist_xy, 0.1)
                perp_y = (fx - self.robot_x) / max(dist_xy, 0.1)
                pt_msg.point.x = fx + d_pan * dist_xy * perp_x
                pt_msg.point.y = fy + d_pan * dist_xy * perp_y
                pt_msg.point.z = fz + d_tilt * dist_xy
                self.look_at_pub.publish(pt_msg)

                if (now - self.aim_timer_start) >= 0.90:
                    img_msg = self.latest_pt_image if self.latest_pt_image is not None else self.latest_front_image
                    vis_ok = False
                    if img_msg is not None:
                        bgr = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
                        vis_ok, _bbox, center, _area = self._detect_coqui_visually(bgr)
                        if vis_ok and center is not None:
                            cx, cy = center
                            err_px = (cx - 320) / 640.0
                            err_py = (240 - cy) / 480.0
                            if abs(err_px) > 0.08 or abs(err_py) > 0.08:
                                fine_cmd = Vector3(
                                    x=float(self.pt_state.x + 0.40 * err_px),
                                    y=float(self.pt_state.y + 0.40 * err_py),
                                    z=0.0,
                                )
                                self.pt_cmd_pub.publish(fine_cmd)

                    if vis_ok or self.scan_idx >= len(self.scan_offsets) - 1:
                        self._capture_and_save_photo(fx, fy, fz, self.vantage_reach_status)
                        # Reset Pan-Tilt camera to neutral (0, 0) BEFORE driving back to initial pose
                        self.pt_cmd_pub.publish(Vector3(x=0.0, y=0.0, z=0.0))
                        self.lock_caller_pub.publish(Bool(data=False))
                        self._transition('RETURNING_TO_INITIAL_POSE')
                        return
                    else:
                        self.scan_idx += 1
                        self.aim_timer_start = now
                return

        # ==================== 6. RETURNING_TO_INITIAL_POSE ====================
        if self.state == 'RETURNING_TO_INITIAL_POSE':
            x0, y0, _z0, yaw0 = self.initial_pose
            res = self._navigate_toward_goal(
                x0, y0, yaw0, stop_tol=0.38,
                label=f'RETURN TO INITIAL POSE ({x0:.1f}, {y0:.1f})'
            )
            if res == 'REACHED' or (now - self.state_enter_time > 32.0):
                if self._rotate_to_yaw(yaw0, tol_rad=0.12) or (now - self.state_enter_time > 36.0):
                    self._stop_robot()
                    total_frogs = int(self.active_frog_info.get('total_frogs', 5))
                    if not self.single_frog_mission and self.capture_count < total_frogs:
                        self.get_logger().info(
                            f"Returned to initial position ({self.robot_x:.2f}, {self.robot_y:.2f}) "
                            f"after spotting frog #{self.capture_count}/{total_frogs}. "
                            f"Resetting triangulation to hunt the next Coqui frog on the map..."
                        )
                        self.reset_tri_pub.publish(String(data='reset'))
                        self.tri_status = None
                        self.sound_detected = False
                        self._transition('EXPLORING')
                    else:
                        self.get_logger().info(
                            f"Returned to initial position ({self.robot_x:.2f}, {self.robot_y:.2f}). "
                            f"All {self.capture_count} Coqui frog(s) localized & photographed!"
                        )
                        self._transition('MISSION_COMPLETE')
            return

        # ==================== 7. MISSION_COMPLETE ====================
        if self.state == 'MISSION_COMPLETE':
            self.current_goal_label = f'MISSION COMPLETE AT INITIAL POSE ({self.capture_count} Frogs Spotted)'
            self._stop_robot()
            return


def main(args=None):
    rclpy.init(args=args)
    node = CoquiHunterBehaviorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
