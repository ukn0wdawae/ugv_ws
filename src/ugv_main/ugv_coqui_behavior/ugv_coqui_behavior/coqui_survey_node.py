#!/usr/bin/env python3
"""Coqui survey mission: explore an area, find calling Coqui frogs by ear, photograph them, record them.

    EXPLORE      drive between listening stations that cover the survey area (stop and listen at each)
      -> a call is heard whose bearing does not point at a frog already recorded
    HUNT         triangulate that one frog: stop and listen until IT has been heard from here, then move to the
                 reachable spot that adds the most bearing parallax (chosen from the costmap, checked with
                 Nav2's planner), repeat. Other frogs' calls never move the robot (acoustic_hunt.py)
      -> triangulation converged, or the frog becomes a lead to come back to
    VANTAGE      pick a camera viewpoint around the estimate (free, line of sight on the costmap,
                 reachable), standoff from the estimated height (on the ground vs perched)
    PHOTO        aim the pan-tilt camera at the estimate, search a few offsets, centre on the frog
    RECORD       append the frog (map xyz + photo) to the survey record, back to EXPLORE
    REVISIT      go back to leads (frogs heard but not located) before calling the area covered
    HOME         when the area is covered or time is up

Design rules
  * Nav2 is the only thing that moves the wheels. Every goal is first checked with ComputePathToPose;
    when Nav2 fails the node picks a different goal, it never steers by itself.
  * Only robot-side information is used: the costmap, the robot's own pose, the acoustic bearings /
    bearings from mic_array_localizer and the cameras. No GPS (the forest is GPS-denied), no simulation truth.
  * Everything runs on ROS time (sim time in Gazebo).

What it shows in RViz (watching only, nothing reads these back)
  /coqui_survey/markers   the complete scene (build_scene): area, stations, recorded frogs, the current
                          estimate, the hunt's bearing rays, the Nav2 goal, the camera aim and a status HUD
  /coqui_survey/hearing   one short-lived arrow per bearing heard, coloured by what the mission did with it
  /coqui_survey/photo     the last annotated frog photo
"""
import itertools
import json
import math
import os
import threading
import time
import uuid

import cv2
import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration as MsgDuration, Time as MsgTime
from lifecycle_msgs.srv import GetState
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Twist, Vector3
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros

from . import acoustic_hunt as ah
from . import slope_recovery as sr
from . import survey_planning as sp
from .survey_store import SurveyStore, load_known
from .survey_vision import annotate, detect_coqui
from .survey_viz import hud_text, ray_segments, station_status


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class MissionAbort(Exception):
    pass


PARAMS = {
    # area / coverage
    'survey_area': [20.0, 50.0, -14.0, 14.0],      # xmin, xmax, ymin, ymax in map frame
    'use_frontiers': True,
    'min_frontier_cells': 4,
    'frontier_min_dist_m': 1.5,
    'frontier_handoff_m': 2.0,
    'frontier_commit_m': 2.5,
    'frontier_commit_sec': 8.0,
    'frontier_commit_bonus_m': 3.0,
    'explore_listen_every_m': 8.0,
    'frontier_size_weight': 0.15,
    'frontier_heading_penalty_m': 2.0,
    'whisper_bias_m': 5.0,
    'trajectory_cover_step_m': 1.5,
    'moving_cover_radius_m': 3.5,
    'station_spacing_m': 6.0,
    'cover_radius_m': 7.0,
    'station_listen_sec': 4.0,
    'snap_search_m': 2.0,
    'failed_radius_m': 1.0,
    'failed_retry_sec': 90.0,
    'max_cooldown_resets': 2,
    # costmap
    'costmap_topic': '/global_costmap/costmap',
    'traversability_grid_topic': '/terrain_traversability/grid',
    'odom_topic': '/odom',
    'lethal_cost': 65,
    'clearance_m': 0.18,
    'terrain_cost_weight': 0.08,
    'path_slope_weight': 3.0,
    # detection / hunt
    'min_detection_snr_db': 7.0,
    'min_coherence': 0.85,
    'loudness_ref_db': 0.0,
    'loudness_gate_db': 10.0,
    'loudness_spread_db': 5.0,
    'hunt_trigger_range_m': 12.0,
    'moving_tf_max_age_s': 0.5,
    'anchor_wait_sec': 25.0,
    'anchor_listen_sec': 5.0,
    'loudness_prior_sd_db': 4.0,
    'passive_window_sec': 600.0,
    'detections_to_trigger': 2,
    'known_frog_radius_m': 2.0,
    'known_bearing_radius_m': 1.1,
    'frog_keepout_m': 0.9,
    'stationary_only_triangulation': True,
    'stationary_max_vx': 0.05,
    'stationary_max_wz': 0.12,
    'stationary_hold_sec': 0.25,
    'hunt_listen_max_sec': 45.0,
    'hunt_min_new_rays': 3,
    'min_step_separation_m': 0.7,
    'group_link_deg': 3.0,
    'partial_min_parallax_deg': 12.0,
    'lead_max_attempts': 3,
    'lead_retry_sec': 120.0,
    'lead_detour_m': 8.0,
    'lead_listen_sec': 25.0,
    'reject_empty_open_photo': True,
    'pose_sigma_m': 0.08,
    'adaptive_steps': True,
    'step_min_m': 0.8,
    'step_max_m': 1.8,
    'step_angles': 12,
    'keepout_m': 1.2,
    'approach_until_m': 5.0,
    'approach_gain_deg_per_m': 6.0,
    'step_cost_deg_per_m': 4.0,
    'max_steps': 8,
    'min_rays': 6,
    'min_baseline_m': 1.0,
    'min_parallax_deg': 15.0,
    'min_places': 3,
    'max_uncertainty_m': 0.6,
    'min_rays_per_place': 2,
    'max_level_sd_db': 3.0,
    'ray_inlier_deg': 2.0,
    'anchor_tol_deg': 6.0,
    'anchor_slack_max_deg': 8.0,
    'min_fix_range_m': 0.6,
    'max_range_m': 25.0,
    'doa_sigma_deg': 1.0,
    'bout_gap_s': 1.2,
    'nominal_range_max_m': 12.0,
    'hunt_timeout_sec': 420.0,
    'suppress_bearing_sec': 180.0,
    # vantage / camera
    'standoff_ground_m': 1.4,
    'standoff_elevated_m': 2.2,
    'elevated_threshold_m': 0.8,
    'vantage_angles': 16,
    'vantage_extra_radii_m': [0.0, 0.5, 1.0, 2.0],
    'los_skip_end_m': 0.5,
    'los_cost': 99,
    'no_los_penalty': 10.0,
    'extra_radius_penalty': 1.0,
    'camera_settle_sec': 2.0,
    'still_wait_sec': 4.0,
    'aim_tolerance_deg': 6.0,
    'vantage_listen_sec': 8.0,
    'vantage_confirm_sec': 45.0,
    'elevation_topic': '/terrain_traversability/elevation_cloud',
    'camera_scan_offsets': [0.0, 0.0, 0.15, 0.0, -0.15, 0.0, 0.0, 0.12, 0.0, -0.12],  # (dpan, dtilt) pairs
    'image_topic': '/pt_camera/image_raw',
    'camera_info_topic': '/pt_camera/camera_info',
    # mission
    'max_mission_sec': 3600.0,
    'return_home': True,
    'home_tolerance_m': 1.2,
    'home_max_stages': 20,
    'output_dir': '/root/ugv_ws/coqui_surveys',
    'run_id': '',
    'resume_survey': '',
    'geo_ref': '',       # '' = no lat/lon (GPS-denied). Operator-surveyed 'lat,lon,map_x,map_y,heading_deg' otherwise
    'nav_goal_timeout_sec': 90.0,
    'plan_check_max': 4,
    'global_frame': 'map',
    'base_frame': 'base_footprint',
    'array_frame': 'audio_array_base_link',
    # IMU steep-slope stall detection, obstacle-avoiding slow descent, & continuous safety watchdog
    'slope_recovery_enabled': True,
    'safety_monitor_enabled': True,
    'imu_topic': '/imu',
    'obstacle_grid_topic': '/terrain_traversability/obstacle_grid',
    'slope_cmd_vel_topic': '/cmd_vel_nav',
    'cmd_vel_monitor_topic': '/cmd_vel',
    **sr.SLOPE_DEFAULTS,
}


class CoquiSurveyNode(Node):
    def __init__(self):
        super().__init__('coqui_survey')
        for k, v in PARAMS.items():
            self.declare_parameter(k, v)
        self.cfg = {k: self.get_parameter(k).value for k in PARAMS}
        c = self.cfg
        self.gf, self.bf = c['global_frame'], c['base_frame']
        self.cb = ReentrantCallbackGroup()
        self.bridge = CvBridge()
        self.tf = tf2_ros.Buffer()
        self.tfl = tf2_ros.TransformListener(self.tf, self)
        self.lock = threading.Lock()

        # latest sensor data (written by callbacks, read by the mission thread)
        self.grid_msg = None
        self.obs_grid_msg = None
        self.imu_slope = None       # latest sr.imu_slope_info() dict
        self.imu_wx = 0.0
        self.imu_wy = 0.0
        self.imu_wz = 0.0           # latest IMU yaw angular velocity (rad/s)
        self.odom_vx = 0.0
        self.odom_wz = 0.0
        self._stationary_since = None
        self.last_cmd_vel = (0.0, 0.0, 0.0)   # (t, vx, wz)
        self.slope_detector = sr.SlopeStallDetector(c)
        self.safety = sr.SafetyMonitor(c)
        self.safety_abort_reason = None
        self.mission_active = False
        self.status = None          # last triangulation status (dict) since the last reset
        self.status_seq = 0
        self.ray_origins = []       # ray origins received since the last reset
        self.reset_time = 0.0
        self.image = None
        self.image_stamp = 0.0
        self.cam_info = None
        self.pt_state = Vector3()
        self.geo_ref = sp.parse_geo_ref(c['geo_ref'])   # (lat, lon, map_x, map_y, heading_deg) or None
        self.recent_doa = []        # [(t, bearing, loudness, origin_xy, obs, stationary)] near calls of the last 4 s
        self.passive_obs = []       # [(t, origin xyz, dir xyz, loudness)] stationary bearings, for triangulation
        self.whisper_rays = []      # [(t, origin_xy, bearing, loudness)] every bearing (incl. moving) for frontier bias
        self.listening_for_new_calls = False
        self.collecting = False     # True during a hunt: stationary bearings go into self.obs_inbox
        self.obs_inbox = []         # bearings heard since the hunt last looked [(t, origin xyz, dir xyz, loudness)]
        self.pending_call = None    # (t, bearing, origin_xy, obs (t, o, d, loudness), stationary)
        self.suppressed = []        # [(until_t, origin_xy, bearing)] calls we gave up on
        self.leads = ah.LeadBook(c)  # frogs heard but not located
        self.heard = {'calls': 0, 'moving': 0, 'no_tf': 0, 'blended': 0, 'near': 0}   # since the last hearing_stats event
        self._explore_goal = None   # frontier being driven to (commitment)
        self._since_listen_m = 0.0  # metres driven since the robot last stood and listened
        self._last_odo_xy = None

        self.known = load_known(c['resume_survey'])
        run_id = c['run_id'] or time.strftime('survey_%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:4]
        self.store = SurveyStore(c['output_dir'], run_id, {
            'frame': self.gf, 'survey_area_xy': list(c['survey_area']), 'resumed_from': c['resume_survey'] or None,
            'known_frogs_at_start': len(self.known),
            'note': 'pos is in the map frame (GPS-denied: no GPS is used); lat/lon only when geo_ref was given'})
        self.state = 'INIT'
        self.listened = []          # stationary listening spots (full cover_radius_m)
        self.moving_listened = []   # moving pass samples (narrower moving_cover_radius_m)
        self.failed_ts = []         # [(x, y, retry_after_t)] failed exploration goals with cooldown
        self.cooldown_resets = 0
        self.good_spots = []        # places Nav2 has driven the robot to successfully
        self.nav_fail_streak = 0
        self._active_nav_handle = None
        self._last_cover_xy = None
        self._last_passive_check = 0.0
        self._last_passive_t = -1.0
        self.rejected_fixes = []    # fixes a photo contradicted
        # what RViz shows (publish_markers / build_scene); None = not shown. Own lock: the mission thread and
        # the 1 Hz timer both publish the scene.
        self.viz_lock = threading.Lock()
        self.viz = {'goal': None, 'estimate': None, 'rays': None, 'anchor': None, 'aim': None,
                    'hud_state': self.state, 'hud_info': {}, 'hud_since': 0.0}
        self.hearing_ids = itertools.count()

        sensor = qos_profile_sensor_data
        latched = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, c['costmap_topic'], self._grid_cb, 1, callback_group=self.cb)
        self.create_subscription(OccupancyGrid, c['traversability_grid_topic'], self._trav_grid_cb, latched,
                                 callback_group=self.cb)
        self.create_subscription(OccupancyGrid, c['obstacle_grid_topic'], self._obs_grid_cb, latched,
                                 callback_group=self.cb)
        self.create_subscription(Odometry, c['odom_topic'], self._odom_cb, sensor, callback_group=self.cb)
        self.create_subscription(Imu, c['imu_topic'], self._imu_cb, sensor, callback_group=self.cb)
        self.create_subscription(Twist, c['cmd_vel_monitor_topic'], self._cmd_vel_cb, 10, callback_group=self.cb)
        self.create_subscription(String, '/ugv_audio/triangulation_status', self._status_cb, 20,
                                 callback_group=self.cb)
        self.create_subscription(String, '/ugv_audio/call', self._call_cb, 50, callback_group=self.cb)
        self.elev_msg = None
        self.create_subscription(PointCloud2, c['elevation_topic'], self._elev_cb, 1, callback_group=self.cb)
        self.create_subscription(Image, c['image_topic'], self._image_cb, sensor, callback_group=self.cb)
        self.create_subscription(CameraInfo, c['camera_info_topic'], self._caminfo_cb, sensor, callback_group=self.cb)
        self.create_subscription(Vector3, '/coqui_behavior/pan_tilt_state', self._pt_cb, 10, callback_group=self.cb)
        self.slope_cmd_pub = self.create_publisher(Twist, c['slope_cmd_vel_topic'], 10)
        self.emergency_cmd_pub = self.create_publisher(Twist, c['cmd_vel_monitor_topic'], 10)
        self.reset_pub = self.create_publisher(String, '/coqui/reset_triangulation', 10)
        self.look_pub = self.create_publisher(PointStamped, '/coqui_behavior/look_at_point', 10)
        self.ptcmd_pub = self.create_publisher(Vector3, '/coqui_behavior/pan_tilt_cmd', 10)
        self.state_pub = self.create_publisher(String, '/coqui_survey/state', latched)
        self.marker_pub = self.create_publisher(MarkerArray, '/coqui_survey/markers', latched)
        self.hearing_pub = self.create_publisher(
            MarkerArray, '/coqui_survey/hearing',
            QoSProfile(depth=20, reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.VOLATILE))
        self.photo_pub = self.create_publisher(Image, '/coqui_survey/photo', latched)
        self.clear_global = self.create_client(ClearEntireCostmap, '/global_costmap/clear_entirely_global_costmap',
                                               callback_group=self.cb)
        self.clear_local = self.create_client(ClearEntireCostmap, '/local_costmap/clear_entirely_local_costmap',
                                              callback_group=self.cb)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose', callback_group=self.cb)
        self.planner = ActionClient(self, ComputePathToPose, 'compute_path_to_pose', callback_group=self.cb)
        self.create_timer(1.0, self._viz_timer_cb, callback_group=self.cb)
        self.create_timer(0.1, self._safety_timer_cb, callback_group=self.cb)
        self.create_timer(60.0, self._hearing_stats_cb, callback_group=self.cb)
        self.get_logger().info(f'Coqui survey -> {self.store.dir} (known frogs at start: {len(self.known)})')

    # ===================================================================================== callbacks
    def _grid_cb(self, msg):
        self.grid_msg = msg
        self.safety.record_sensor('costmap', self.now())

    def _trav_grid_cb(self, _msg):
        self.safety.record_sensor('traversability', self.now())

    def _obs_grid_cb(self, msg):
        self.obs_grid_msg = msg
        self.safety.record_sensor('traversability', self.now())

    def _odom_cb(self, msg):
        t = self.now()
        vx = float(msg.twist.twist.linear.x)
        wz = float(msg.twist.twist.angular.z)
        self.odom_vx = vx
        self.odom_wz = wz
        still = (abs(vx) <= float(self.cfg['stationary_max_vx'])
                 and abs(wz) <= float(self.cfg['stationary_max_wz'])
                 and abs(self.imu_wz) <= float(self.cfg['stationary_max_wz']))
        if still:
            if self._stationary_since is None:
                self._stationary_since = t
        else:
            self._stationary_since = None

    def _is_stationary(self, now_t=None):
        if now_t is None:
            now_t = self.now()
        if self._stationary_since is None:
            return False
        return (now_t - self._stationary_since) >= float(self.cfg['stationary_hold_sec'])

    def _imu_cb(self, msg):
        q = msg.orientation
        a = msg.linear_acceleration
        info = sr.imu_slope_info(q.x, q.y, q.z, q.w, a.x, a.y, a.z)
        self.imu_slope = info
        self.imu_wx = float(msg.angular_velocity.x)
        self.imu_wy = float(msg.angular_velocity.y)
        self.imu_wz = float(msg.angular_velocity.z)
        self.slope_detector.update_tilt(info['tilt_deg'], roll_deg=info['roll_deg'])
        self.safety.record_sensor('imu', self.now())
        if abs(self.imu_wz) > float(self.cfg['stationary_max_wz']):
            self._stationary_since = None

    def _safety_timer_cb(self):
        """Continuous 10 Hz mission-wide safety monitor (runs in EVERY state, not just Nav2 goals)."""
        if not self.mission_active or not self.cfg.get('safety_monitor_enabled', True):
            return
        if self.safety_abort_reason is not None:
            self._emergency_stop()
            return
        now_t = self.now()
        p = self.pose(quiet=True)
        if p is not None:
            self.safety.record_sensor('tf', now_t)
        tilt = self.imu_slope['tilt_deg'] if self.imu_slope is not None else None
        code, detail = self.safety.update(
            now_t,
            pose_xyz=p[:3] if p is not None else None,
            tilt_deg=tilt,
            imu_wx=self.imu_wx,
            imu_wy=self.imu_wy,
        )
        if code is not None:
            self._trigger_safety_abort(f'{code}: {detail}')
            return
        stale = self.safety.check_stale_sensors(now_t)
        if stale is not None:
            self._trigger_safety_abort(f'stale_sensor: {stale}')

    def _emergency_stop(self):
        zero = Twist()
        try:
            self.slope_cmd_pub.publish(zero)
            self.emergency_cmd_pub.publish(zero)
        except Exception:
            pass

    def _trigger_safety_abort(self, reason):
        if self.safety_abort_reason is not None:
            return
        self.safety_abort_reason = reason
        self.get_logger().error(f'[SAFETY ABORT] {reason} — stopping robot and aborting mission!')
        self._emergency_stop()
        try:
            self.store.log_event('safety_abort', reason=reason, t=round(self.now(), 2))
        except Exception:
            pass
        h = self._active_nav_handle
        if h is not None:
            try:
                h.cancel_goal_async()
            except Exception:
                pass

    def _check_safety_abort(self):
        if self.safety_abort_reason is not None:
            self._cancel_active_nav()
            self._emergency_stop()
            raise MissionAbort(f'safety stop ({self.safety_abort_reason})')

    def _cmd_vel_cb(self, msg):
        self.last_cmd_vel = (self.now(), float(msg.linear.x), float(msg.angular.z))

    def _status_cb(self, msg):
        try:
            st = json.loads(msg.data)
        except ValueError:
            return
        with self.lock:
            if self.now() < self.reset_time:
                return
            self.status = st
            self.status_seq += 1
            o = st.get('ray_origin')
            if o and all(math.hypot(o[0] - p[0], o[1] - p[1]) > 0.3 for p in self.ray_origins):
                self.ray_origins.append((o[0], o[1]))

    def _bearing_tf(self, frame, stamp, stationary):
        """TF of the microphone array for a call heard at `stamp`. The call often arrives a few ms before the pose of the
        same instant; a driving robot then takes the newest pose (at most moving_tf_max_age_s old), which is plenty to
        notice a frog. Returning None here is what made the robot deaf whenever its wheels turned."""
        try:
            return self.tf.lookup_transform(self.gf, frame, stamp)
        except Exception:
            pass
        try:
            tf = self.tf.lookup_transform(self.gf, frame, Time())
        except Exception:
            return None
        if stationary or stamp.nanoseconds == 0:
            return tf
        tf_t = Time.from_msg(tf.header.stamp).nanoseconds
        age = abs(stamp.nanoseconds - tf_t) * 1e-9
        return tf if tf_t == 0 or age <= float(self.cfg['moving_tf_max_age_s']) else None

    def _call_cb(self, msg):
        """One detected coqui call from mic_array_localizer (JSON, see its docstring)."""
        try:
            call = json.loads(msg.data)
            az, el, loud = float(call['az']), float(call['el']), float(call['level_db'])
            snr, coherence = float(call['snr_db']), float(call.get('coherence', 1.0))
        except (ValueError, KeyError, TypeError):
            return
        if snr < self.cfg['min_detection_snr_db']:
            return
        active = self.listening_for_new_calls or self.collecting
        t = float(call.get('t', 0.0)) or self.now()
        stationary = self._is_stationary(t)   # standing when the call was captured, not when it got here
        frame = call.get('frame') or self.cfg['array_frame']
        tf = self._bearing_tf(frame, Time(nanoseconds=int(t * 1e9)), stationary)
        with self.lock:
            self.heard['calls'] += 1
            self.heard['moving'] += 0 if stationary else 1
            self.heard['no_tf'] += 1 if tf is None else 0
        if tf is None:
            return
        # 3D bearing in the map frame (array may be tilted on a slope)
        q = tf.transform.rotation
        R = np.array([[1 - 2 * (q.y * q.y + q.z * q.z), 2 * (q.x * q.y - q.z * q.w), 2 * (q.x * q.z + q.y * q.w)],
                      [2 * (q.x * q.y + q.z * q.w), 1 - 2 * (q.x * q.x + q.z * q.z), 2 * (q.y * q.z - q.x * q.w)],
                      [2 * (q.x * q.z - q.y * q.w), 2 * (q.y * q.z + q.x * q.w), 1 - 2 * (q.x * q.x + q.y * q.y)]])
        d = R @ np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        o = np.array([tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z])
        bearing = math.atan2(d[1], d[0])
        origin = (o[0], o[1])
        clean = coherence >= float(self.cfg['min_coherence'])
        # positions come only from a standing robot and from calls that are one frog: two frogs calling at once blend
        # into a bearing that points at neither
        use_for_pos = clean and (stationary or not self.cfg.get('stationary_only_triangulation', True))
        # every bearing is drawn in RViz (show_bearing), coloured by what happens to it below
        if not clean:
            with self.lock:
                self.heard['blended'] += 1
        # not listening right now (photographing, driving home, ...), or two frogs at once while standing. While driving
        # the motors cost coherence too: such a bearing may still stop the robot, it just never becomes a position.
        if not active or (not clean and (stationary or coherence < 0.5 * float(self.cfg['min_coherence']))):
            self.show_bearing(o, d, 'ignored')
            return
        obs = (t, o, d, loud)
        br_radius = float(self.cfg.get('known_bearing_radius_m', self.cfg['known_frog_radius_m']))
        if sp.bearing_points_at_known(origin, bearing, self.known, br_radius, loudness=loud, cfg=self.cfg):
            self.show_bearing(o, d, 'ignored')
            return
        if self.collecting:
            if use_for_pos:
                with self.lock:
                    self.obs_inbox.append(obs)
            self.show_bearing(o, d, 'hunt' if use_for_pos else 'ignored')
            return
        near = sp.call_is_near(loud, self.cfg)
        with self.lock:
            win = self.cfg['passive_window_sec']
            self.whisper_rays = [r for r in self.whisper_rays if t - r[0] < 120.0][-59:] + [(t, origin, bearing, loud)]
            if use_for_pos:
                self.passive_obs = [r for r in self.passive_obs if t - r[0] < win][-399:] + [obs]
            self.heard['near'] += 1 if near else 0
        for until, s_xy, s_bearing in self.suppressed:   # calls we already gave up on from near here
            if t < until and math.hypot(s_xy[0] - origin[0], s_xy[1] - origin[1]) < 3.0 and \
                    abs(sp.wrap(s_bearing - bearing)) < math.radians(20):
                self.show_bearing(o, d, 'ignored')
                return
        lead = self.leads.match(origin, bearing, loud)
        if lead is not None and (self.leads.exhausted(lead) or self.now() - lead['t'] < self.cfg['lead_retry_sec']):
            self.show_bearing(o, d, 'ignored')   # already tried lately; its bearings are banked for the revisit
            return
        self.show_bearing(o, d, 'new')
        # Two tiers by distance (from loudness, not SNR: SNR also falls when the motors are loud): far calls bias
        # frontier exploration and are banked when stationary; near ones stop the robot for a hunt.
        if not near:
            return
        with self.lock:
            self.recent_doa = [r for r in self.recent_doa if t - r[0] < 6.0] + \
                [(t, bearing, loud, origin, obs, use_for_pos)]
            agree = [r for r in self.recent_doa if abs(sp.wrap(r[1] - bearing)) < math.radians(15)
                     and abs(r[2] - loud) <= 6.0]
            if len(agree) >= self.cfg['detections_to_trigger'] and self.pending_call is None:
                self.pending_call = (t, bearing, origin, obs, use_for_pos)

    def _elev_cb(self, msg):
        self.elev_msg = msg     # read on demand (ground_z)

    def ground_z(self, x, y, radius=0.45):
        """Measured terrain height (map) around (x, y), or None where the robot has not seen the ground."""
        msg = self.elev_msg
        if msg is None or msg.width == 0:
            return None
        try:
            tf = self.tf.lookup_transform(self.gf, msg.header.frame_id, Time()).transform
        except Exception:
            return None
        a = np.frombuffer(msg.data, np.float32).reshape(-1, msg.point_step // 4)[:, :3].astype(float)
        yaw = 2.0 * math.atan2(tf.rotation.z, tf.rotation.w)
        cs, sn = math.cos(yaw), math.sin(yaw)
        px = cs * a[:, 0] - sn * a[:, 1] + tf.translation.x
        py = sn * a[:, 0] + cs * a[:, 1] + tf.translation.y
        m = np.hypot(px - x, py - y) <= radius
        # low percentile: grass and bushes stand on the ground, they are not the ground
        return float(np.percentile(a[m, 2], 20) + tf.translation.z) if int(m.sum()) >= 5 else None

    def _image_cb(self, msg):
        self.image = msg
        self.image_stamp = self.now()

    def _caminfo_cb(self, msg):
        self.cam_info = msg

    def _pt_cb(self, msg):
        self.pt_state = msg

    def _viz_timer_cb(self):
        self.publish_markers()   # 1 Hz: keeps the RViz HUD clock and state live (never waits)

    def _hearing_stats_cb(self):
        """Once a minute in events.jsonl: what the microphones gave the mission (is the robot deaf while driving?)."""
        if not self.mission_active:
            return
        with self.lock:
            st, self.heard = self.heard, {k: 0 for k in self.heard}
        self.store.log_event('hearing_stats', t=round(self.now(), 1), **st)

    # ======================================================================================= helpers
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def sleep(self, sec, abort_on_call=False):
        end = self.now() + sec
        while rclpy.ok() and self.now() < end:
            self._check_safety_abort()
            if abort_on_call and self.pending_call is not None:
                return False
            time.sleep(0.05)
        return True

    def pose(self, quiet=False):
        return self.pose_in_frame(self.gf, quiet=quiet)

    def pose_in_frame(self, target_frame, quiet=False):
        try:
            tf = self.tf.lookup_transform(target_frame, self.bf, Time())
        except Exception as e:
            if not quiet:
                self.get_logger().warn(f'no {target_frame}->{self.bf}: {e}', throttle_duration_sec=5.0)
            return None
        t = tf.transform.translation
        return t.x, t.y, t.z, yaw_of(tf.transform.rotation)

    def grid(self):
        return sp.Grid.from_msg(self.grid_msg) if self.grid_msg is not None else None

    def set_state(self, s, **info):
        self.state = s
        info.update(state=s, t=round(self.now(), 1), frogs_found=len(self.store.frogs))
        self.state_pub.publish(String(data=json.dumps(info)))
        self.get_logger().info(f'[{s}] ' + ', '.join(f'{k}={v}' for k, v in info.items() if k not in ('state', 't')))
        self.store.log_event('state', **info)
        hud = dict(info, listen_s=self.cfg['station_listen_sec']) if s == 'LISTEN' else dict(info)
        self.publish_markers(hud_state=s, hud_info=hud, hud_since=self.now())   # RViz status HUD

    def time_left(self):
        return self.cfg['max_mission_sec'] - (self.now() - self.t_start)

    def _wait(self, fut, timeout_wall):
        end = time.time() + timeout_wall
        while not fut.done():
            if time.time() > end or not rclpy.ok():
                return None
            time.sleep(0.02)
        return fut.result()

    def _goal_pose(self, x, y, yaw):
        ps = PoseStamped()
        ps.header.frame_id = self.gf
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y = float(x), float(y)
        ps.pose.orientation.z, ps.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        return ps

    def plan_length(self, x, y):
        """Path length Nav2's planner finds from the robot to (x, y), or None if unreachable."""
        g = ComputePathToPose.Goal()
        g.goal = self._goal_pose(x, y, 0.0)
        g.use_start = False
        h = self._wait(self.planner.send_goal_async(g), 10.0)
        if h is None or not h.accepted:
            return None
        r = self._wait(h.get_result_async(), 15.0)
        if r is None or r.status != GoalStatus.STATUS_SUCCEEDED or not r.result.path.poses:
            return None
        pts = [(p.pose.position.x, p.pose.position.y) for p in r.result.path.poses]
        if math.hypot(pts[-1][0] - x, pts[-1][1] - y) > 0.6:   # planner tolerance landed elsewhere
            return None
        return sp.path_terrain_cost(pts, self.grid(), self.cfg['path_slope_weight'])

    def best_reachable(self, candidates, extra_cost=None, fast_accept=False):
        """candidates: [(x, y, ...)] in preference order. Plans to the first plan_check_max and
        returns (candidate, path_len) with the lowest (rank + path) cost, or (None, None).
        When fast_accept=True (continuous frontier exploration), accept the first candidate whose
        planned path is close to straight-line distance instead of making extra RPC round-trips."""
        p = self.pose(quiet=True) if fast_accept else None
        best, best_cost = None, None
        for rank, cand in enumerate(candidates[:self.cfg['plan_check_max']]):
            L = self.plan_length(cand[0], cand[1])
            if L is None:
                continue
            if fast_accept and p is not None:
                straight = math.hypot(cand[0] - p[0], cand[1] - p[1])
                if L <= 1.6 * straight + 2.5:
                    return cand, L
            cost = L + (extra_cost(cand, rank) if extra_cost else 0.0)
            if best_cost is None or cost < best_cost:
                best, best_cost = (cand, L), cost
        return best if best else (None, None)

    def _active_failed(self):
        now = self.now()
        return [(x, y) for x, y, until in self.failed_ts if now < until]

    def _mark_failed(self, pt):
        self.failed_ts.append((float(pt[0]), float(pt[1]), self.now() + self.cfg['failed_retry_sec']))

    def _weak_rays(self):
        """Recent passive bearings [(origin_xy, bearing_rad, snr)] not pointing at known/suppressed frogs."""
        t = self.now()
        win = 120.0
        br_radius = float(self.cfg.get('known_bearing_radius_m', self.cfg['known_frog_radius_m']))
        with self.lock:
            snap = list(self.whisper_rays)
        out = []
        for rt, origin, br, loud in snap:
            if t - rt > win:
                continue
            if sp.bearing_points_at_known(origin, br, self.known, br_radius, loudness=loud, cfg=self.cfg):
                continue
            if any(t < until and math.hypot(s_xy[0] - origin[0], s_xy[1] - origin[1]) < 3.0 and
                   abs(sp.wrap(s_b - br)) < math.radians(20) for until, s_xy, s_b in self.suppressed):
                continue
            out.append((origin, br, loud))
        return out[-20:]

    def _usable_passive(self, limit=200):
        """Banked stationary bearings that do not point at recorded frogs (newest last)."""
        t = self.now()
        win = self.cfg['passive_window_sec']
        br_radius = float(self.cfg.get('known_bearing_radius_m', self.cfg['known_frog_radius_m']))
        with self.lock:
            recent = [r for r in self.passive_obs if t - r[0] < win][-limit:]
        return [r for r in recent if not sp.bearing_points_at_known(
            (float(r[1][0]), float(r[1][1])), math.atan2(float(r[2][1]), float(r[2][0])), self.known, br_radius,
            loudness=r[3], cfg=self.cfg)]

    def _check_passive_trigger(self):
        """Start a hunt if the banked stationary bearings already locate an unrecorded frog completely."""
        t = self.now()
        if self.pending_call is not None or t - self._last_passive_check < 1.5:
            return
        self._last_passive_check = t
        obs = self._usable_passive()
        if len(obs) < int(self.cfg['min_rays']) or obs[-1][0] == self._last_passive_t:
            return
        self._last_passive_t = obs[-1][0]
        # anchor on each recent group of bearings in turn: in a chorus the newest bearing is any frog's
        tried = []
        for k in range(len(obs) - 1, max(-1, len(obs) - 12), -1):
            ob = obs[k]
            br = math.atan2(float(ob[2][1]), float(ob[2][0]))
            if any(abs(sp.wrap(br - b)) < math.radians(6) and math.hypot(ob[1][0] - o[0], ob[1][1] - o[1]) < 0.5
                   for b, o in tried):
                continue
            tried.append((br, ob[1]))
            if any(t < until and math.hypot(s_xy[0] - ob[1][0], s_xy[1] - ob[1][1]) < 3.0 and
                   abs(sp.wrap(s_b - br)) < math.radians(20) for until, s_xy, s_b in self.suppressed):
                continue
            tri, a_obs = sp.check_passive_triangulation(obs[:k] + obs[k + 1:] + [ob], self.known, self.cfg)
            if tri is not None and a_obs is not None and not self._rejected_spot(tri['pos']):
                with self.lock:
                    if self.pending_call is None:
                        self.pending_call = (a_obs[0], br, (float(a_obs[1][0]), float(a_obs[1][1])), a_obs, True)
                return

    def _rejected_spot(self, pos):
        """A photo already showed there is no frog at pos."""
        return any(math.hypot(pos[0] - r[0], pos[1] - r[1]) < 0.8 for r in self.rejected_fixes)

    def _record_cover_pose(self, p, stationary=False):
        if p is None or not self.listening_for_new_calls:
            return
        if stationary:
            self.listened.append((p[0], p[1]))
            self._last_cover_xy = (p[0], p[1])
            self._since_listen_m = 0.0
            return
        step = self.cfg['trajectory_cover_step_m']
        if self._last_cover_xy is None or math.hypot(p[0] - self._last_cover_xy[0], p[1] - self._last_cover_xy[1]) >= step:
            self._last_cover_xy = (p[0], p[1])
            self.moving_listened.append((p[0], p[1]))

    def _cancel_active_nav(self):
        h = self._active_nav_handle
        self._active_nav_handle = None
        if h is not None:
            try:
                self._wait(h.cancel_goal_async(), 3.0)
            except Exception:
                pass

    def _track_distance(self, p):
        """Metres driven since the robot last stood and listened."""
        if self._last_odo_xy is not None:
            self._since_listen_m += math.hypot(p[0] - self._last_odo_xy[0], p[1] - self._last_odo_xy[1])
        self._last_odo_xy = (p[0], p[1])

    def navigate(self, x, y, yaw, label, abort_on_call=False, handoff_frontier=False):
        """Drive with Nav2. Returns 'SUCCEEDED', 'HANDOFF' (preemptive frontier handoff while moving),
        'FAILED', 'CALL' (cancelled because a new call was heard) or 'TIMEOUT'."""
        self.publish_markers(goal=(x, y, yaw, label))
        try:
            res = self._navigate(x, y, yaw, abort_on_call, handoff_frontier)
        finally:
            if not handoff_frontier or self._active_nav_handle is None:
                self.publish_markers(goal=None)
        if isinstance(res, tuple) and res[0] == 'SLOPE_STALL':
            if res[1] == 'capsized':
                self._emergency_stop()
                raise MissionAbort('capsized during navigation')
            self.slope_descent(reason=res[1], bad_goal=(x, y))
            return 'FAILED'
        return res

    def _navigate(self, x, y, yaw, abort_on_call, handoff_frontier=False):
        self._check_safety_abort()
        g = NavigateToPose.Goal()
        g.pose = self._goal_pose(x, y, yaw)
        h = self._wait(self.nav.send_goal_async(g), 10.0)
        if h is None or not h.accepted:
            self._active_nav_handle = None
            self.nav_fail_streak += 1
            return 'FAILED'
        self._active_nav_handle = h
        res_f = h.get_result_async()
        t_start = self.now()
        t_end = t_start + self.cfg['nav_goal_timeout_sec']
        self.slope_detector.reset_motion()
        p_start = self.pose(quiet=True)
        while rclpy.ok() and not res_f.done():
            self._check_safety_abort()
            p = self.pose(quiet=True)
            now_t = self.now()
            if p is not None:
                self._track_distance(p)
            if self.cfg['slope_recovery_enabled'] and p is not None and self.imu_slope is not None:
                cmd_t, cmd_vx, cmd_wz = self.last_cmd_vel
                if now_t - cmd_t > 0.6:
                    cmd_vx, cmd_wz = 0.0, 0.0
                stall = self.slope_detector.update(
                    now_t, p[0], p[1], p[3],
                    self.imu_slope['tilt_deg'], self.imu_wz, cmd_vx, cmd_wz,
                    active=True, roll_deg=self.imu_slope['roll_deg'])
                if stall is not None:
                    self.get_logger().warn(
                        f'IMU detected {stall} on {self.slope_detector.smooth_tilt_deg:.1f} deg slope '
                        f'at ({p[0]:.2f}, {p[1]:.2f}); cancelling Nav2')
                    self._cancel_active_nav()
                    return ('SLOPE_STALL', stall)
            if abort_on_call:
                self._record_cover_pose(p, stationary=False)
                self._check_passive_trigger()
                if self.pending_call is not None:
                    self._cancel_active_nav()
                    return 'CALL'
                if self.listening_for_new_calls and self._since_listen_m >= self.cfg['explore_listen_every_m']:
                    self._cancel_active_nav()
                    return 'LISTEN'
            if handoff_frontier and p is not None and now_t - t_start > 0.8:
                dist_goal = math.hypot(x - p[0], y - p[1])
                driven = 0.0 if p_start is None else math.hypot(p[0] - p_start[0], p[1] - p_start[1])
                if sp.frontier_handoff(dist_goal, driven, now_t - t_start,
                                       sp.frontier_still_unknown((x, y), self.grid(), 1.2), self.cfg,
                                       None if p_start is None else math.hypot(x - p_start[0], y - p_start[1])):
                    # keep robot rolling at speed while explore_step computes and preempts with the next frontier
                    self.nav_fail_streak = 0
                    self.good_spots.append((p[0], p[1]))
                    return 'HANDOFF'
            if now_t > t_end:
                self._cancel_active_nav()
                self.nav_fail_streak += 1
                return 'TIMEOUT'
            time.sleep(0.05)
        self._active_nav_handle = None
        r = res_f.result()
        ok = r is not None and r.status == GoalStatus.STATUS_SUCCEEDED
        if ok:
            self.nav_fail_streak = 0
            p = self.pose(quiet=True)
            if p is not None:
                self.good_spots.append((p[0], p[1]))
        else:
            if self.cfg['slope_recovery_enabled'] and self.slope_detector.is_capsized():
                return ('SLOPE_STALL', 'capsized')
            if self.cfg['slope_recovery_enabled'] and self.slope_detector.is_steep():
                return ('SLOPE_STALL', 'nav_failed_on_slope')
            self.nav_fail_streak += 1
        return 'SUCCEEDED' if ok else 'FAILED'

    def _publish_slope_cmd(self, vx, wz):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.angular.z = float(wz)
        self.slope_cmd_pub.publish(msg)

    def slope_descent(self, reason='slope_stall', bad_goal=None):
        """IMU-guided slow downhill descent when skid-steer turning stalls on a steep slope.

        Cancels Nav2 and drives steadily down the IMU gravity fall line (forward if nose-down,
        reverse if nose-up — never spinning in place) while steering around physical obstacles in
        /terrain_traversability/obstacle_grid until the IMU confirms flat ground.
        Returns True if flat ground was reached.
        """
        self._cancel_active_nav()
        self.slope_detector.reset_motion()
        p0 = self.pose(quiet=True)
        if p0 is None or self.imu_slope is None:
            return False
        if bad_goal is not None:
            self._mark_failed(bad_goal)
        self._mark_failed((p0[0], p0[1]))
        start_tilt = float(self.slope_detector.smooth_tilt_deg)
        t0 = self.now()
        flat_since = None
        drive_sign = None
        prev_wz = 0.0
        dist_traveled = 0.0
        last_xy = (p0[0], p0[1])
        prog_xy = (p0[0], p0[1])
        prog_t = t0
        last_wedge_t = t0
        blocked_since = None
        wedge_until = 0.0
        wedge_dir = 1.0
        wedge_sign = 1
        last_hud_t = 0.0
        outcome = 'limit'

        # Brief settle so any lingering Nav2 velocity command clears before we drive /cmd_vel_nav
        for _ in range(3):
            self._publish_slope_cmd(0.0, 0.0)
            time.sleep(0.05)

        self.set_state('SLOPE_DESCENT', reason=reason, tilt_deg=round(start_tilt, 1),
                       at=[round(p0[0], 2), round(p0[1], 2)])
        try:
            while rclpy.ok():
                now_t = self.now()
                elapsed = now_t - t0
                p_map = self.pose(quiet=True)
                sinfo = self.imu_slope
                if p_map is None or sinfo is None:
                    time.sleep(0.06)
                    continue

                step_d = math.hypot(p_map[0] - last_xy[0], p_map[1] - last_xy[1])
                if step_d < 1.0:
                    dist_traveled += step_d
                last_xy = (p_map[0], p_map[1])
                if math.hypot(p_map[0] - prog_xy[0], p_map[1] - prog_xy[1]) >= 0.08:
                    prog_xy = (p_map[0], p_map[1])
                    prog_t = now_t

                smooth_tilt = float(self.slope_detector.smooth_tilt_deg)
                if smooth_tilt >= float(self.cfg['slope_max_drivable_deg']):
                    outcome = 'capsized'
                    break
                if smooth_tilt <= float(self.cfg['slope_flat_deg']):
                    if flat_since is None:
                        flat_since = now_t
                    elif now_t - flat_since >= float(self.cfg['slope_flat_confirm_sec']):
                        outcome = 'flat_ground'
                        break
                else:
                    flat_since = None

                if elapsed >= float(self.cfg['slope_descent_max_sec']) or \
                        dist_traveled >= float(self.cfg['slope_descent_max_dist_m']):
                    outcome = 'limit'
                    break

                obs_msg = self.obs_grid_msg
                obs_grid = sp.Grid.from_msg(obs_msg) if obs_msg is not None else None
                obs_frame = obs_msg.header.frame_id if obs_msg is not None else self.gf
                p_obs = self.pose_in_frame(obs_frame, quiet=True) if obs_frame != self.gf else p_map
                obs_to_map = None
                if p_obs is None:
                    p_obs, obs_grid, obs_frame = p_map, None, self.gf
                elif obs_frame != self.gf:
                    try:
                        tfm_om = self.tf.lookup_transform(self.gf, obs_frame, Time())
                        obs_to_map = (tfm_om.transform.translation.x, tfm_om.transform.translation.y,
                                      yaw_of(tfm_om.transform.rotation))
                    except Exception:
                        # Fallback rigid transform inferred from p_map and p_obs
                        dyaw = sp.wrap(p_map[3] - p_obs[3])
                        c_d, s_d = math.cos(dyaw), math.sin(dyaw)
                        tx = p_map[0] - (c_d * p_obs[0] - s_d * p_obs[1])
                        ty = p_map[1] - (s_d * p_obs[0] + c_d * p_obs[1])
                        obs_to_map = (tx, ty, dyaw)
                area_arg = self.cfg['survey_area']

                self._check_safety_abort()
                if now_t < wedge_until:
                    vx = wedge_sign * float(self.cfg['slope_descent_speed_min'])
                    wz = wedge_dir * 0.55
                    dbg = {'blocked': False, 'wedge_escape': True}
                else:
                    vx, wz, drive_sign, dbg = sr.choose_descent_cmd(
                        sinfo, (p_obs[0], p_obs[1], p_obs[3]), obs_grid,
                        prev_sign=drive_sign, prev_wz=prev_wz, cfg=self.cfg,
                        survey_area=area_arg, obs_to_map=obs_to_map)
                    if dbg.get('blocked'):
                        if blocked_since is None:
                            blocked_since = now_t
                    else:
                        blocked_since = None
                    # If wedged on a boulder edge (no physical motion for 3.5 s or all arcs blocked 2.0 s),
                    # briefly arc backward/diagonally away from the rock and re-evaluate downhill.
                    if now_t - prog_t >= 12.0:
                        outcome = 'blocked'
                        break
                    wedged = ((now_t - prog_t >= 3.5 and now_t - last_wedge_t >= 3.5)
                              or (blocked_since is not None and now_t - blocked_since >= 2.0))
                    if wedged:
                        wedge_sign = -(drive_sign or 1)
                        wedge_dir = -wedge_dir
                        wedge_until = now_t + 1.2
                        last_wedge_t = now_t
                        blocked_since = None
                        vx = wedge_sign * float(self.cfg['slope_descent_speed_min'])
                        wz = wedge_dir * 0.55

                prev_wz = wz
                self._publish_slope_cmd(vx, wz)
                if now_t - last_hud_t >= 1.0:
                    last_hud_t = now_t
                    direction = 'forward' if vx >= 0.0 else 'reverse'
                    self.publish_markers(
                        hud_state='SLOPE_DESCENT',
                        hud_info={'reason': reason, 'tilt_deg': round(smooth_tilt, 1), 'direction': direction},
                        hud_since=t0)
                time.sleep(0.065)
        finally:
            for _ in range(5):
                self._publish_slope_cmd(0.0, 0.0)
                time.sleep(0.05)
            self.slope_detector.reset_motion()

        p_end = self.pose(quiet=True) or p0
        end_tilt = float(self.slope_detector.smooth_tilt_deg)
        self.get_logger().info(
            f'[SLOPE_DESCENT] {outcome}: tilt {start_tilt:.1f}->{end_tilt:.1f} deg, '
            f'dist={dist_traveled:.1f} m in {self.now() - t0:.1f} s, '
            f'({p0[0]:.2f},{p0[1]:.2f},{p0[2]:.2f}) -> ({p_end[0]:.2f},{p_end[1]:.2f},{p_end[2]:.2f})')
        self.store.log_event(
            'slope_descent', reason=reason, outcome=outcome,
            start_pose=[round(v, 2) for v in p0[:3]], end_pose=[round(v, 2) for v in p_end[:3]],
            start_tilt_deg=round(start_tilt, 1), end_tilt_deg=round(end_tilt, 1),
            dist_m=round(dist_traveled, 2), duration_s=round(self.now() - t0, 1))
        if outcome == 'capsized':
            self._emergency_stop()
            raise MissionAbort(f'capsized during slope descent (tilt {end_tilt:.1f} deg)')
        if outcome == 'flat_ground':
            self.nav_fail_streak = 0
            self.good_spots.append((p_end[0], p_end[1]))
            self._record_cover_pose(p_end, stationary=False)
            if self.clear_local.wait_for_service(timeout_sec=1.5):
                self._wait(self.clear_local.call_async(ClearEntireCostmap.Request()), 3.0)
            self.sleep(0.8)
            return True
        return False

    def unstick(self):
        """After repeated Nav2 failures: if tilted steeply, descend downhill to flat ground via IMU;
        otherwise drive back (with Nav2) to the most recent place the robot reached before.
        Raises MissionAbort if that fails too."""
        self._cancel_active_nav()
        self._check_safety_abort()
        p = self.pose()
        if self.cfg['slope_recovery_enabled'] and self.slope_detector.is_capsized():
            self._emergency_stop()
            raise MissionAbort('capsized before unstick')
        if self.cfg['slope_recovery_enabled'] and self.slope_detector.is_steep():
            if self.slope_descent(reason='unstick_on_slope'):
                return
        back = [s for s in reversed(self.good_spots) if math.hypot(s[0] - p[0], s[1] - p[1]) > 0.8]
        self.set_state('UNSTICK', fails=self.nav_fail_streak, at=[round(p[0], 2), round(p[1], 2)])
        for spot in back[:3]:
            if self.navigate(spot[0], spot[1], math.atan2(spot[1] - p[1], spot[0] - p[0]), 'unstick') == 'SUCCEEDED':
                return
        raise MissionAbort(f'stuck at ({p[0]:.2f}, {p[1]:.2f}): Nav2 cannot move the robot')

    def dump_costmap(self, tag):
        """Save the global costmap + robot pose (diagnostics for planning failures)."""
        try:
            g, p = self.grid(), self.pose(quiet=True)
            if g is None:
                return
            d = os.path.join(self.store.dir, 'diag')
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f'{tag}_{self.now():.0f}.npz')
            np.savez_compressed(path, cost=g.data, origin=[g.origin_x, g.origin_y], res=g.resolution,
                                pose=p if p else [])
            self.store.log_event('costmap_dump', tag=tag, file=path)
        except Exception as e:
            self.get_logger().warn(f'costmap dump failed: {e}')

    def clear_costmaps(self, include_global=False):
        """Clear local costmap (and only optionally global as a true last resort so frontiers survive)."""
        clients = (self.clear_local, self.clear_global) if include_global else (self.clear_local,)
        for cli in clients:
            if cli.wait_for_service(timeout_sec=2.0):
                self._wait(cli.call_async(ClearEntireCostmap.Request()), 5.0)
        self.sleep(2.0)   # let the sensors repopulate around the robot

    def reset_triangulation(self):
        with self.lock:
            self.reset_time = self.now() + 0.2
            self.status = None
            self.ray_origins = []
        self.reset_pub.publish(String(data='reset'))
        self.sleep(0.3)

    # ======================================================================================= mission
    def run(self):
        try:
            self._run()
        except MissionAbort as e:
            self.mission_active = False
            self._cancel_active_nav()
            self._emergency_stop()
            self.get_logger().error(f'mission aborted: {e}')
            self.store.set_mission(result=f'aborted: {e}')
            try:
                self.set_state('DONE', result=f'aborted: {e}')
            except Exception:
                pass
        except Exception as e:  # keep the record even on a bug (ignore rclpy teardown on Ctrl-C)
            self.mission_active = False
            if not rclpy.ok():
                self.get_logger().info('mission stopped during ROS shutdown')
                return
            self.get_logger().error(f'mission crashed: {e!r}')
            self.store.set_mission(result=f'crashed: {e!r}')
            raise

    def return_home_staged(self):
        """Drive back to `self.home` in rolling-window stages (handles distances >> global_costmap size)."""
        self.set_state('HOME')
        self.listening_for_new_calls = False
        tol = float(self.cfg.get('home_tolerance_m', 1.2))
        max_stages = int(self.cfg.get('home_max_stages', 20))
        home_failed = []
        for stage in range(max_stages):
            if not rclpy.ok():
                break
            self._check_safety_abort()
            p = self.pose(quiet=True)
            if p is None:
                break
            d_home = math.hypot(self.home[0] - p[0], self.home[1] - p[1])
            if d_home <= tol:
                return True, round(d_home, 2)
            cands = sp.next_home_waypoints(
                (p[0], p[1]), (self.home[0], self.home[1]), self.good_spots, self.grid(), self.cfg,
                failed=home_failed)
            goal, _ = self.best_reachable(cands)
            if goal is None:
                self.clear_costmaps(include_global=False)
                goal, _ = self.best_reachable(cands)
            if goal is None:
                break
            is_final = math.hypot(goal[0] - self.home[0], goal[1] - self.home[1]) < 0.5
            yaw = self.home[3] if is_final else math.atan2(goal[1] - p[1], goal[0] - p[0])
            res = self.navigate(goal[0], goal[1], yaw, 'home' if is_final else f'home stage {stage + 1}')
            if res != 'SUCCEEDED':
                home_failed.append(goal)
        p = self.pose(quiet=True)
        final_d = math.hypot(self.home[0] - p[0], self.home[1] - p[1]) if p is not None else float('inf')
        return (final_d <= tol), (round(final_d, 2) if math.isfinite(final_d) else None)

    def _run(self):
        self.set_state('WAIT_FOR_NAV2')
        while rclpy.ok() and (self.pose(quiet=True) is None or self.grid_msg is None
                              or not self.nav.wait_for_server(timeout_sec=1.0)
                              or not self.planner.wait_for_server(timeout_sec=1.0)):
            time.sleep(0.5)
        # Nav2 accepts goals only once bt_navigator is active; a successful plan proves the planner is up
        cli = self.create_client(GetState, '/bt_navigator/get_state', callback_group=self.cb)
        while rclpy.ok():
            if cli.wait_for_service(timeout_sec=1.0):
                r = self._wait(cli.call_async(GetState.Request()), 5.0)
                if r is not None and r.current_state.label == 'active':
                    break
            time.sleep(1.0)
        p = self.pose()
        wait_plan_end = self.now() + 12.0
        while rclpy.ok() and self.now() < wait_plan_end and \
                self.plan_length(p[0] + 0.5 * math.cos(p[3]), p[1] + 0.5 * math.sin(p[3])) is None:
            if self.cfg['slope_recovery_enabled'] and self.slope_detector.is_steep():
                break
            self.sleep(2.0)
        self.t_start = self.now()
        self.home = self.pose()
        now_t = self.now()
        for s_name in ('imu', 'costmap', 'traversability', 'tf'):
            if s_name not in self.safety.sensor_stamps:
                self.safety.record_sensor(s_name, now_t)
        self.mission_active = True
        self.good_spots.append((self.home[0], self.home[1]))
        self.cells = sp.coverage_cells(self.cfg['survey_area'], self.cfg['station_spacing_m'])
        self.store.set_mission(start_pose=[round(v, 3) for v in self.home], result='running')
        self.set_state('START', home=[round(v, 2) for v in self.home[:2]], stations_to_cover=len(self.cells))

        self.listen_at_station()
        while rclpy.ok() and self.time_left() > 0:
            self._check_safety_abort()
            if self.pending_call is not None:
                self.hunt()
                continue
            if not self.explore_step():
                break
        if self.time_left() <= 0:
            result = 'time limit'
        elif self.failed_ts and any(
            not sp.is_cell_covered(c, self.listened, self.cfg['cover_radius_m'],
                                   self.moving_listened, self.cfg['moving_cover_radius_m'])
            for c in self.cells
        ):
            result = 'no reachable stations left'
        else:
            result = 'area covered'
        if self.leads.leads:
            result = f'{result} ({len(self.leads.leads)} frogs heard but not located)'
        self._cancel_active_nav()
        returned_home, home_dist_m = None, None
        if self.cfg['return_home']:
            returned_home, home_dist_m = self.return_home_staged()
            if not returned_home and home_dist_m is not None:
                result = f'{result} (return home failed, {home_dist_m:.1f} m from home)'
        self.mission_active = False
        self.store.set_mission(result=result, duration_s=round(self.now() - self.t_start, 1),
                               frogs_found=len(self.store.frogs), stations_listened=len(self.listened),
                               returned_home=returned_home, home_dist_m=home_dist_m)
        self.set_state('DONE', result=result)

    # ------------------------------------------------------------------------------------- explore
    def listen_at_station(self):
        p = self.pose()
        self.set_state('LISTEN', at=[round(p[0], 1), round(p[1], 1)])
        self.listening_for_new_calls = True
        heard = not self.sleep(self.cfg['station_listen_sec'], abort_on_call=True)
        if not heard:
            self._check_passive_trigger()
            heard = self.pending_call is not None
        if not heard:
            self._record_cover_pose(p, stationary=True)
            self.publish_markers()

    def explore_step(self):
        """Continuous frontier exploration with fallback to uncovered listening stations.
        Returns False when the survey area is covered."""
        self._check_safety_abort()
        if self.nav_fail_streak >= 3:
            self.unstick()
        p = self.pose()
        grid = self.grid()
        failed = self._active_failed()
        self.listening_for_new_calls = True

        # A frog heard nearby but not located is worth a short detour before driving on
        near_leads = self.leads.due((p[0], p[1]), self.now(), max_dist=self.cfg['lead_detour_m'])
        if near_leads:
            self.revisit_lead(near_leads[0])
            return True

        # Check remaining uncovered stations so we know when the full survey box is covered
        all_st_cands = sp.station_candidates(
            (p[0], p[1]), self.cells, self.listened, self.cfg['cover_radius_m'],
            failed, grid, self.cfg, known=self.known,
            moving_listened=self.moving_listened, moving_cover_radius=self.cfg['moving_cover_radius_m'])
        if not all_st_cands:
            # Distinguish truly covered vs every remaining uncovered station stuck in cooldown!
            uncovered_any = sp.station_candidates(
                (p[0], p[1]), self.cells, self.listened, self.cfg['cover_radius_m'],
                [], grid, self.cfg, known=self.known,
                moving_listened=self.moving_listened, moving_cover_radius=self.cfg['moving_cover_radius_m'])
            if uncovered_any and self.cooldown_resets < int(self.cfg.get('max_cooldown_resets', 2)):
                self.cooldown_resets += 1
                self.failed_ts.clear()
                self.get_logger().warn(
                    f'All {len(uncovered_any)} remaining stations were in cooldown; clearing cooldowns '
                    f'({self.cooldown_resets}/{self.cfg["max_cooldown_resets"]}) and refreshing local costmap')
                self.clear_costmaps(include_global=False)
                return True
            self._cancel_active_nav()
            # the area is covered; go back to every frog that was heard but not located before saying so
            leads = self.leads.due((p[0], p[1]), float('inf'))
            if leads:
                self.revisit_lead(leads[0])
                return True
            return False
        st_cands = [c for c in all_st_cands if grid is None or grid.inside(*c[0])]

        # 1) Primary: industry-standard continuous frontier exploration (biased toward Tier-1 whispers)
        goal, L, mode = None, None, 'station'
        if self.cfg['use_frontiers'] and grid is not None:
            fr_cands = sp.frontier_candidates((p[0], p[1]), p[3], self.cfg['survey_area'], grid, failed,
                                              self.cfg, weak_rays=self._weak_rays(),
                                              uncovered_xy=[c for _, c in all_st_cands],
                                              cover_radius=self.cfg['cover_radius_m'],
                                              known=self.known, prev_goal=self._explore_goal)
            if fr_cands:
                fr_goals = [xy for _, xy, _ in fr_cands]
                (goal, L) = self.best_reachable(fr_goals, fast_accept=True)
                if goal is not None:
                    mode = 'frontier'

        # 2) Fallback: uncovered listening stations (projecting onto rolling-window edge if outside grid)
        goals = [(g[0], g[1]) for g, _ in st_cands]
        if not goals and grid is not None:
            tx, ty = all_st_cands[0][0]
            dx, dy = tx - p[0], ty - p[1]
            d = max(math.hypot(dx, dy), 1e-6)
            step_r = min(d, 0.4 * min(grid.width, grid.height) * grid.resolution)
            gx, gy = p[0] + dx * step_r / d, p[1] + dy * step_r / d
            snapped = grid.nearest_free(gx, gy, self.cfg['clearance_m'], self.cfg['lethal_cost'], 4.0)
            goals = [snapped] if snapped is not None else [(gx, gy)]
        if goal is None:
            (goal, L) = self.best_reachable(goals, fast_accept=True)
        if goal is None:
            # If the robot is boxed into lethal slope cells on a steep hillside, descend to flat ground
            # immediately via IMU rather than trying to replan back uphill.
            self._cancel_active_nav()
            if self.cfg['slope_recovery_enabled'] and self.slope_detector.is_steep():
                if self.slope_descent(reason='boxed_in_on_slope'):
                    return True
            # every candidate failed to plan: first suspect the local costmap around the robot, not the area
            self.dump_costmap('stations_unreachable')
            self.set_state('RECOVER', reason='no station plannable', candidates=len(goals))
            self.clear_costmaps(include_global=False)
            (goal, L) = self.best_reachable(goals)
            if goal is None and len(self.good_spots) > 1:
                try:
                    self.unstick()
                except MissionAbort:
                    pass
                (goal, L) = self.best_reachable(goals)
        if goal is None:
            # Only mark the single closest candidate failed (never bulk-fail 16 stations in 10 s when boxed in!)
            if goals:
                self._mark_failed(goals[0])
            self.nav_fail_streak += 1
            self.store.log_event('stations_unreachable', goals=[list(map(lambda v: round(v, 2), gg))
                                                                for gg in goals[:self.cfg['plan_check_max']]])
            return True

        self.set_state('EXPLORE', mode=mode, goal=[round(goal[0], 1), round(goal[1], 1)], path_m=round(L, 1),
                       uncovered=len(all_st_cands))
        yaw = math.atan2(goal[1] - p[1], goal[0] - p[0])
        self._explore_goal = (goal[0], goal[1]) if mode == 'frontier' else None
        res = self.navigate(goal[0], goal[1], yaw, mode, abort_on_call=True,
                            handoff_frontier=(mode == 'frontier'))
        if res in ('CALL', 'HANDOFF'):
            return True
        if res == 'LISTEN':   # it has been driving for a while: stand and listen (the only bearings good for positions)
            self.listen_at_station()
            return True
        if res == 'SUCCEEDED':
            if mode == 'frontier':
                self._record_cover_pose(self.pose(quiet=True), stationary=False)
            else:
                self.listen_at_station()
        else:
            self._explore_goal = None
            self._mark_failed(goal)
            self.store.log_event('station_failed', mode=mode, goal=[round(goal[0], 2), round(goal[1], 2)], result=res)
        return True

    def revisit_lead(self, lead):
        """Go and listen near a frog that was heard but not located. A call from it starts a normal hunt."""
        p = self.pose()
        goal, L = self.best_reachable(self.leads.listening_places(lead, (p[0], p[1]), self.grid(), self.cfg, self.known))
        self.set_state('REVISIT', lead=lead['id'], attempt=lead['attempts'] + 1,
                       pos=[round(v, 1) for v in lead['pos']], goal=None if goal is None else [round(v, 1) for v in goal])
        lead['t'] = -float('inf')       # its calls may start a hunt now
        self.suppressed = [(u, o, b) for u, o, b in self.suppressed if abs(sp.wrap(
            b - math.atan2(lead['pos'][1] - o[1], lead['pos'][0] - o[0]))) > math.radians(25)]
        self.listening_for_new_calls = True
        if goal is not None:
            yaw = math.atan2(lead['pos'][1] - goal[1], lead['pos'][0] - goal[0])
            if self.navigate(goal[0], goal[1], yaw, lead['id'], abort_on_call=True) == 'CALL':
                return
            self.sleep(self.cfg['lead_listen_sec'], abort_on_call=True)
            if self.pending_call is None:
                self._check_passive_trigger()
            if self.pending_call is not None:
                return
        lead['attempts'] += 1
        lead['t'] = self.now()
        lead['reason'] = 'no reachable listening place' if goal is None else 'silent on revisit'
        self.store.set_unresolved(self.leads.records())

    # ---------------------------------------------------------------------------------------- hunt
    def _take_inbox(self):
        with self.lock:
            obs, self.obs_inbox = self.obs_inbox, []
        return obs

    def _find_anchor(self, rough):
        """The call that started the hunt was heard while driving (or blended with another): stand still until the same
        frog is heard cleanly. Returns (anchor, every clean bearing heard meanwhile)."""
        t0 = self.now()
        heard, t_end = [], t0 + float(self.cfg['anchor_wait_sec'])
        while rclpy.ok() and self.now() < t_end:
            self._check_safety_abort()
            heard += self._take_inbox()
            anchor = ah.pick_anchor(heard, rough, self.cfg)
            if anchor is not None and self.now() - t0 >= float(self.cfg['anchor_listen_sec']):
                return anchor, heard     # not the first call that fits: another frog's may come before the right one's
            time.sleep(0.1)
        return None, heard

    def _hunt_seed(self, anchor, heard):
        """Bearings that may help this hunt: banked ones from the last minutes within 15 m, plus `heard`."""
        seed, seen = [], set()
        for r in self._usable_passive() + list(heard):
            if math.hypot(float(r[1][0]) - float(anchor[1][0]), float(r[1][1]) - float(anchor[1][1])) <= 15.0 \
                    and round(r[0], 3) not in seen:
                seen.add(round(r[0], 3))
                seed.append(r)
        return seed

    def hunt(self):
        """Locate the one frog whose call is pending (acoustic_hunt.HuntTracker decides; this drives)."""
        self._cancel_active_nav()
        t_call, bearing, origin, call, clean = self.pending_call
        self.listening_for_new_calls = False
        self.reset_triangulation()   # localizer's own estimate (RViz only) starts fresh too
        t0 = self.now()
        self._take_inbox()
        self.collecting = True
        self.set_state('HUNT', bearing_deg=round(math.degrees(bearing)), heard_at=[round(float(v), 1) for v in origin])
        anchor, heard = (call, []) if clean else self._find_anchor(call)
        if anchor is None:
            self._end_hunt(None, 'call not heard again after stopping', t0, call, heard)
            return
        tracker = ah.HuntTracker(anchor, self.cfg, self.now(), known=self.known, seed=self._hunt_seed(anchor, heard))
        p = self.pose()
        tracker.at_place((p[0], p[1]), self.now())
        plan_fail, step_failed, shown, outcome = 0, [], -1, 'stopped'
        while rclpy.ok():
            self._check_safety_abort()
            for o in self._take_inbox():
                tracker.add(o)
            action, why = tracker.decide(self.now())
            if len(tracker.obs) != shown:
                shown = len(tracker.obs)
                self.show_hunt(tracker.obs, tracker.triangulate(), (anchor[1], anchor[2]))
            if action == 'end':
                outcome = why
                break
            if action == 'listen':
                time.sleep(0.2)
                continue
            p = self.pose()
            grid = self.grid()
            cands = tracker.step_candidates((p[0], p[1]), grid, avoid=step_failed)
            if not cands:   # nothing seen-free around: allow unseen ground
                cands = tracker.step_candidates((p[0], p[1]), grid, allow_unknown=True, avoid=step_failed)
            goal, L = self.best_reachable([c for _, c in cands], extra_cost=lambda c, rank: 0.5 * rank)
            tracker.steps += 1
            tri, est = tracker.triangulate(), tracker.estimate_xy()
            info = {'step': tracker.steps, 'why': why, 'obs': len(tracker.obs), 'est': [round(v, 2) for v in est]}
            if tri:
                info.update(inliers=tri['inliers'], places=tri['n_origins'], parallax_deg=tri['parallax_deg'],
                            unc_m=tri['uncertainty_m'], level_db=tri['source_level_db'])
            self.set_state('HUNT', **info)
            if goal is None:
                plan_fail += 1
                if plan_fail >= 3:
                    outcome = tracker.finish('no reachable step')
                    break
                self.sleep(2.0)     # the costmap may still be filling in
                tracker.at_place((p[0], p[1]), self.now())
                continue
            yaw = math.atan2(est[1] - goal[1], est[0] - goal[0])
            if self.navigate(goal[0], goal[1], yaw, f'baseline step {tracker.steps}') != 'SUCCEEDED':
                step_failed.append(goal)    # Nav2 could not get there: never pick that spot again
                if self.nav_fail_streak >= 3:
                    outcome = tracker.finish('robot stuck')
                    break
            p = self.pose()
            tracker.at_place((p[0], p[1]), self.now())
        self._end_hunt(tracker, outcome, t0, anchor, heard)

    def _end_hunt(self, tracker, outcome, t0, anchor, heard):
        """Log the hunt, photograph a converged fix, and remember an unresolved frog as a lead."""
        self.collecting = False
        tri = None if tracker is None else tracker.triangulate()
        obs = list(heard) if tracker is None else tracker.obs
        leftover = obs if tracker is None else tracker.leftover()
        with self.lock:   # bearings this hunt did not use are other frogs': keep them for the next hunt
            have = {round(r[0], 3) for r in self.passive_obs}
            self.passive_obs = sorted(self.passive_obs + [o for o in leftover if round(o[0], 3) not in have],
                                      key=lambda o: o[0])[-400:]
        steps = 0 if tracker is None else tracker.steps
        self.store.log_event('hunt_end', t=round(self.now(), 1), outcome=outcome, steps=steps,
                             tri=None if tri is None else {k: v for k, v in tri.items() if k != 'inlier_mask'},
                             duration_s=round(self.now() - t0, 1),
                             places=[] if tracker is None else [[round(v, 2) for v in q] for q in tracker.places],
                             anchor=[round(anchor[0], 2), [round(float(v), 3) for v in anchor[1]],
                                     [round(float(v), 4) for v in anchor[2]], round(float(anchor[3]), 1)],
                             obs=[[round(o[0], 2)] + [round(float(v), 3) for v in o[1]] +
                                  [round(float(v), 4) for v in o[2]] + [round(float(o[3]), 1)] for o in obs])
        self.pending_call = None
        self.recent_doa = []
        recorded = False
        if outcome.startswith('converged') and tri is not None:
            recorded, why = self.visit_and_photograph(tracker, steps, self.now() - t0)
            tri = tracker.triangulate() or tri
            if not recorded:
                outcome = f'rejected ({why})'
                if why in ('photo_contradicts_open_ground', 'unseen_and_not_heard_at_the_viewpoint'):
                    self.rejected_fixes.append(tuple(tri['pos']))
        if tracker is not None:
            self.listened += ah.coverage_places(tracker.places, recorded)
        if recorded:
            self.leads.resolve_near(self.store.frogs[-1]['pos'], self.cfg['known_frog_radius_m'])
        else:
            a_o = (float(anchor[1][0]), float(anchor[1][1]))
            a_b = math.atan2(float(anchor[2][1]), float(anchor[2][0]))
            self.suppressed.append((self.now() + self.cfg['suppress_bearing_sec'], a_o, a_b))
            if outcome != 'already recorded frog':
                if tracker is None:
                    r = min(max(sp.implied_range_m(anchor[3], self.cfg), 2.0), self.cfg['nominal_range_max_m'])
                    est, spread = (a_o[0] + r * math.cos(a_b), a_o[1] + r * math.sin(a_b)), 0.4 * r
                else:
                    est, spread = tracker.estimate_xy(), 3.0 if tri is None else tri['uncertainty_m']
                ld = self.leads.note(anchor, est, spread, self.now(), outcome)
                self.store.log_event('lead', t=round(self.now(), 1), id=ld['id'], attempts=ld['attempts'],
                                     pos=[round(v, 2) for v in ld['pos']], spread_m=round(ld['spread_m'], 2), why=outcome)
            self.set_state('HUNT_END', outcome=outcome)
        self.store.set_unresolved(self.leads.records())
        self.publish_markers(estimate=None, rays=None, anchor=None)   # RViz: this hunt is over
        self.listening_for_new_calls = True

    # -------------------------------------------------------------------------------- vantage/photo
    def visit_and_photograph(self, tracker, steps, hunt_s):
        """Drive to a viewpoint, listen there, photograph. Returns (recorded, why)."""
        st = tracker.triangulate()
        est = st['pos']
        p = self.pose()
        kind = sp.classify_height(est[2], p[2], self.cfg['elevated_threshold_m'])
        self.set_state('VANTAGE', est=[round(v, 2) for v in est], height=kind)
        vantage, vantage_los, failed_v, staged = 'not reached', False, [], False
        for attempt in range(4):
            self._check_safety_abort()
            p = self.pose()
            grid = self.grid()
            cands = sp.vantage_candidates(est, (p[0], p[1]), kind, grid, self.cfg,
                                          allow_unknown=False, known=self.known)
            cands = [c for c in cands if all(math.hypot(c[1][0] - f[0], c[1][1] - f[1]) > 0.5 for f in failed_v)]
            if not cands and not staged:
                # nothing around the frog has been seen yet: get within ~3 m so the sensors map it first
                staged = True
                d = math.hypot(p[0] - est[0], p[1] - est[1])
                if d > self.cfg['standoff_ground_m'] + 1.6:
                    r = self.cfg['standoff_ground_m'] + 1.5
                    sx, sy = est[0] + r * (p[0] - est[0]) / d, est[1] + r * (p[1] - est[1]) / d
                    snap = grid.nearest_free(sx, sy, self.cfg['clearance_m'], self.cfg['lethal_cost'], 1.0) \
                        if grid is not None else (sx, sy)
                    if snap is not None and not sp.in_frog_keepout(snap, self.known, self.cfg['frog_keepout_m']):
                        self.set_state('VANTAGE', step='staging', goal=[round(snap[0], 2), round(snap[1], 2)])
                        self.navigate(snap[0], snap[1], math.atan2(est[1] - snap[1], est[0] - snap[0]), 'staging')
                self.sleep(2.0)
                continue
            if not cands:
                cands = sp.vantage_candidates(est, (p[0], p[1]), kind, grid, self.cfg,
                                              allow_unknown=True, known=self.known)
                cands = [c for c in cands
                         if all(math.hypot(c[1][0] - f[0], c[1][1] - f[1]) > 0.5 for f in failed_v)]
            pts = [(v[0], v[1], v[2], los, r) for _, v, r, los in cands]
            (goal, L) = self.best_reachable(pts, extra_cost=lambda c, rank: 0.0 if c[3] else self.cfg['no_los_penalty'])
            if goal is None:
                self.dump_costmap('vantage_unreachable')
                break
            self.set_state('VANTAGE', attempt=attempt + 1, goal=[round(goal[0], 2), round(goal[1], 2)],
                           line_of_sight=goal[3], path_m=round(L, 1))
            label = f'viewpoint {attempt + 1}' + ('' if goal[3] else ' (no line of sight)')   # RViz only
            if self.navigate(goal[0], goal[1], goal[2], label) == 'SUCCEEDED':
                vantage_los = bool(goal[3])
                vantage = 'reached' + ('' if vantage_los else ' (no clear line of sight on costmap)')
                break
            failed_v.append(goal)
        self.wait_still()
        heard = self.listen_at_viewpoint(tracker, self.cfg['vantage_listen_sec']) if vantage.startswith('reached') else None
        self.set_state('PHOTO', vantage=vantage, heard_here=heard)
        shot = self.photograph(tracker, vantage)
        if not shot['found'] and heard == 0:
            # not seen and not heard: give it the longest pause a frog takes before deciding it is not there
            heard = self.listen_at_viewpoint(tracker, self.cfg['vantage_confirm_sec'], enough=2)
            if heard:
                shot = self.photograph(tracker, vantage)
        return self.record(tracker, shot, vantage, vantage_los, None if heard is None else heard > 0, steps, hunt_s)

    def wait_still(self):
        """Nav2 reports a goal as reached while the robot is still turning (about 1 rad/s): wait until it stands."""
        t_end = self.now() + float(self.cfg['still_wait_sec'])
        while rclpy.ok() and self.now() < t_end and not self._is_stationary():
            self._check_safety_abort()
            time.sleep(0.05)

    def listen_at_viewpoint(self, tracker, max_sec, enough=3):
        """Collect bearings at the viewpoint, where a frog is loud and a bearing is worth centimetres. Only bearings that
        are the frog at the fix refine it (tracker.refine); the others are banked. Returns how many were it (stops after
        `enough`)."""
        p = self.pose()
        tracker.at_place((p[0], p[1]), self.now())
        self._take_inbox()
        self.collecting = True
        heard, other, t_end = 0, [], self.now() + float(max_sec)
        while rclpy.ok() and self.now() < t_end and heard < enough:
            self._check_safety_abort()
            new = self._take_inbox()
            used = tracker.refine(new, (p[0], p[1]))
            heard += len(used)
            other += [o for o in new if not any(o is u for u in used)]
            time.sleep(0.1)
        self.collecting = False
        with self.lock:
            self.passive_obs = (self.passive_obs + other)[-400:]
        tri = tracker.triangulate()
        self.show_hunt(tracker.obs, tri, (tracker.anchor[1], tracker.anchor[2]))
        self.store.log_event('viewpoint_listen', t=round(self.now(), 1), heard=heard, others=len(other),
                             est=[round(v, 3) for v in tri['pos']], unc_m=tri['uncertainty_m'])
        return heard

    def frog_point(self, tri):
        """(x, y, z, kind) to aim at and to record. A frog that is not clearly perched sits on the ground the robot
        measured there; the array's own height estimate is off by decimetres."""
        est = tri['pos']
        p = self.pose()
        g = self.ground_z(est[0], est[1])
        base = p[2] if g is None else g
        kind = sp.classify_height(est[2], base, self.cfg['elevated_threshold_m'])
        z = est[2] if kind == 'elevated' or g is None else g + 0.05
        return float(est[0]), float(est[1]), float(z), kind


    def aim(self, est):
        """Point the camera at est and make sure it does: the aim is computed from one pose, so a robot that still turned
        (or was corrected) while the camera moved leaves it pointing somewhere else."""
        for attempt in range(4):
            self.wait_still()
            self._aim_once(est)
            p = self.pose(quiet=True)
            err = 0.0 if p is None else math.degrees(abs(sp.pan_error((p[0], p[1]), p[3], self.pt_state.x, est)))
            if err <= float(self.cfg['aim_tolerance_deg']):
                return
            self.get_logger().info(f'camera is {err:.0f} deg off after aiming: aiming again')

    def _aim_once(self, est):
        ps = PointStamped()
        ps.header.frame_id = self.gf
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.point = Point(x=float(est[0]), y=float(est[1]), z=float(est[2]) + 0.05)
        self.look_pub.publish(ps)
        p = self.pose(quiet=True)   # RViz: line from the pan-tilt camera to the aim point while photographing
        if p is not None:
            self.publish_markers(aim=((p[0], p[1], p[2] + 0.35), (ps.point.x, ps.point.y, ps.point.z)))
        self._settle()

    def _settle(self):
        t_end = self.now() + 6.0
        self.sleep(0.3)
        while self.now() < t_end and self.pt_state.z > 0.01:
            time.sleep(0.05)
        self.sleep(self.cfg['camera_settle_sec'])

    def point_camera(self, pan, tilt):
        self.ptcmd_pub.publish(Vector3(x=float(pan), y=float(tilt), z=0.0))
        self._settle()

    def grab(self):
        t_req = self.now()
        while self.now() - t_req < 3.0:
            if self.image is not None and self.image_stamp >= t_req:
                return self.bridge.imgmsg_to_cv2(self.image, desired_encoding='bgr8')
            time.sleep(0.05)
        return None if self.image is None else self.bridge.imgmsg_to_cv2(self.image, desired_encoding='bgr8')

    def photograph(self, tracker, vantage):
        """Aim at the fix, search a little around it, centre on the frog when it is seen."""
        x, y, z, kind = self.frog_point(tracker.triangulate())
        est = (x, y, z)
        self.aim(est)
        pan0, tilt0 = self.pt_state.x, self.pt_state.y
        offs = self.cfg['camera_scan_offsets']
        found, img, bbox, centre = False, None, None, None
        for dp, dt in zip(offs[0::2], offs[1::2]):
            if dp or dt:
                self.point_camera(pan0 + dp, tilt0 + dt)
            img = self.grab()
            if img is None:
                continue
            found, bbox, centre, _ = detect_coqui(img)
            if found:
                break
        if found and self.cam_info is not None and centre is not None:
            fx, fy = self.cam_info.k[0], self.cam_info.k[4]
            cx, cy = self.cam_info.k[2], self.cam_info.k[5]
            pan = self.pt_state.x + math.atan2(centre[0] - cx, fx)     # +pan turns the camera right
            tilt = self.pt_state.y + math.atan2(cy - centre[1], fy)    # +tilt pitches up
            self.point_camera(pan, tilt)
            img2 = self.grab()
            if img2 is not None:
                f2, b2, c2, _ = detect_coqui(img2)
                if f2:
                    img, bbox, centre = img2, b2, c2
        return {'found': found, 'img': img, 'bbox': bbox, 'est': est, 'kind': kind}

    def record(self, tracker, shot, vantage, vantage_los, heard_near, steps, hunt_s):
        """Save the frog if it was seen, or heard well enough. Returns (recorded, why)."""
        found, img, bbox, est, kind = (shot[k] for k in ('found', 'img', 'bbox', 'est', 'kind'))
        st = tracker.triangulate()
        p = self.pose()
        accept, accept_reason = sp.should_accept_record(st, found, vantage, vantage_los, self.grid(), self.cfg,
                                                        heard_near=heard_near)
        if not accept:
            self.get_logger().warn(
                f'Rejecting unconfirmed frog fix at {[round(float(v), 2) for v in est]} '
                f'({accept_reason}, vantage={vantage}, los={vantage_los})')
            self.store.log_event('frog_rejected', reason=accept_reason,
                                 est=[round(float(v), 3) for v in est], vantage=vantage, los=vantage_los)
            self.publish_markers(aim=None)
            self.point_camera(0.0, 0.0)
            return False, accept_reason
        rec = {
            'pos': [round(float(v), 3) for v in est],
            'uncertainty_m': st.get('uncertainty_m'),
            'source_level_db': st.get('source_level_db'),
            'height_class': kind,
            'visually_confirmed': bool(found),
            'heard_at_viewpoint': heard_near,
            'accept_reason': accept_reason,
            'vantage': vantage,
            'acoustic': {'bearings_used': st.get('inliers'), 'bearings_heard': st.get('n_obs'),
                         'listening_places': st.get('n_origins'), 'baseline_m': st.get('baseline_m'),
                         'parallax_deg': st.get('parallax_deg'), 'rms_deg': st.get('rms_deg'),
                         'hunt_steps': steps, 'hunt_s': round(hunt_s, 1)},
            'robot_pose_at_photo': [round(v, 3) for v in p],
            'camera_pan_deg': round(math.degrees(self.pt_state.x), 1),
            'camera_tilt_deg': round(math.degrees(self.pt_state.y), 1),
            'sim_time_s': round(self.now(), 1),
            'mission_time_s': round(self.now() - self.t_start, 1),
            'wall_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        if self.geo_ref is not None:
            lat, lon, mx, my, heading = self.geo_ref
            rec['lat'], rec['lon'] = [round(v, 8) for v in sp.local_to_latlon(lat, lon, (mx, my), est[:2], heading)]
        fid = f'coqui_{len(self.store.frogs) + 1:03d}'
        rec['photo'] = None
        if img is not None:
            lines = [(f'{fid}  {"FROG IN VIEW" if found else "frog not visible (occluded?)"}  [{kind}]',
                      (0, 255, 0) if found else (0, 180, 255)),
                     (f'map ({est[0]:.2f}, {est[1]:.2f}, {est[2]:.2f}) +/-{st.get("uncertainty_m", 0):.2f} m',
                      (235, 235, 235)),
                     (f'vantage: {vantage}', (235, 235, 235))]
            out = annotate(img, found, bbox, lines)
            path = self.store.photo_path(fid)
            cv2.imwrite(path, out)
            rec['photo'] = path
            self.photo_pub.publish(self.bridge.cv2_to_imgmsg(out, encoding='bgr8'))
        rec['id'] = fid
        self.store.add_frog(rec)
        self.known.append(rec)
        br_radius = float(self.cfg.get('known_bearing_radius_m', self.cfg['known_frog_radius_m']))
        with self.lock:
            self.passive_obs = [
                r for r in self.passive_obs
                if not sp.bearing_points_at_known(
                    (float(r[1][0]), float(r[1][1])),
                    math.atan2(float(r[2][1]), float(r[2][0])),
                    self.known, br_radius, loudness=r[3], cfg=self.cfg)
            ]
        self.publish_markers(aim=None)   # RViz: the new frog; the camera stops aiming at it
        self.point_camera(0.0, 0.0)
        self.set_state('RECORDED', id=fid, pos=rec['pos'], visual=found, vantage=vantage)
        return True, accept_reason

    # ------------------------------------------------------------------------------------- markers
    def publish_markers(self, **changes):
        """Update what RViz shows and re-publish the complete scene on /coqui_survey/markers (the 1 Hz timer
        calls it without changes, which keeps the HUD clock live). changes (None hides an item):
        goal (x, y, yaw, label), estimate (pos, uncertainty_m), rays (ray_segments()), anchor (origin, dir),
        aim (from xyz, to xyz), hud_state / hud_info / hud_since (from set_state).
        Never raises and never waits: a visualization bug must not stop or stall the mission."""
        try:
            with self.viz_lock:
                unknown = set(changes) - set(self.viz)
                if unknown:
                    raise KeyError(f'unknown scene items {sorted(unknown)}')
                self.viz.update(changes)
                ma = build_scene(self._scene_data(), self.get_clock().now().to_msg(), self.gf)
                self.marker_pub.publish(ma)
        except Exception as e:  # visualization must never disturb the mission
            self.get_logger().warn(f'survey markers not published: {e!r}', throttle_duration_sec=10.0)

    def _scene_data(self):
        """Plain-data snapshot of the scene for build_scene() (called with viz_lock held)."""
        c, v = self.cfg, self.viz
        home = getattr(self, 'home', None)
        cells = getattr(self, 'cells', None) or []
        # failed goals are station goals, snapped up to snap_search_m away from their cell centre
        stations = station_status(cells, list(self.listened), c['cover_radius_m'], self._active_failed(),
                                  c['failed_radius_m'] + c['snap_search_m'])
        t_start = getattr(self, 't_start', None)
        if t_start is None:
            mission_s = 0.0
        else:   # the clock stops when the mission is done
            mission_s = (v['hud_since'] if v['hud_state'] == 'DONE' else self.now()) - t_start
        covered = sum(1 for s in stations if s[2] == 'covered')
        hud = hud_text(v['hud_state'], v['hud_info'], len(self.store.frogs), mission_s, covered, len(cells))
        return {'area': c['survey_area'], 'z': home[2] if home else None, 'stations': stations,
                'frogs': list(self.known), 'estimate': v['estimate'], 'rays': v['rays'], 'anchor': v['anchor'],
                'goal': v['goal'], 'aim': v['aim'], 'hud': hud, 'base_frame': self.bf}

    def show_hunt(self, obs, tri, anchor):
        """RViz, after each triangulation of a hunt: the bearing rays (green agree with the estimate, red
        rejected, orange before the first fix), the anchor bearing that started the hunt (white) and the
        estimate (yellow)."""
        try:
            rays = ray_segments(obs, None if tri is None else tri['inlier_mask'], None if tri is None else tri['pos'])
            estimate = None if tri is None else (tri['pos'], tri['uncertainty_m'])
        except Exception as e:  # visualization must never disturb the mission
            self.get_logger().warn(f'hunt rays not drawn: {e!r}', throttle_duration_sec=10.0)
            rays, estimate = None, None
        self.publish_markers(rays=rays, anchor=anchor, estimate=estimate)

    def show_bearing(self, origin, direction, use):
        """RViz: one heard bearing on /coqui_survey/hearing (see hearing_marker); use 'new', 'hunt' or 'ignored'."""
        try:
            m = hearing_marker(origin, direction, use, next(self.hearing_ids) % 50,
                               self.get_clock().now().to_msg(), self.gf)
            self.hearing_pub.publish(MarkerArray(markers=[m]))
        except Exception as e:  # visualization must never disturb the mission
            self.get_logger().warn(f'hearing marker not published: {e!r}', throttle_duration_sec=10.0)


# ------------------------------------------------------------------------------------------------ RViz
# Colour language (r, g, b, a): what the robot recorded is green, its current estimate yellow, the Nav2 goal blue.
WHITE = (1.0, 1.0, 1.0, 1.0)
FROG_PHOTO_RGBA = (0.1, 1.0, 0.2, 0.9)          # recorded and seen in the photo
FROG_ACOUSTIC_RGBA = (0.6, 0.9, 0.2, 0.9)       # recorded by ear only
ESTIMATE_RGB = (1.0, 0.85, 0.0)
GOAL_RGB = (0.2, 0.5, 1.0)
ANCHOR_RGBA = (1.0, 1.0, 1.0, 0.9)
AIM_RGBA = (1.0, 0.6, 0.1, 0.9)
RAY_RGBA = {'inlier': (0.1, 1.0, 0.2, 0.9), 'outlier': (1.0, 0.25, 0.2, 0.6), 'pending': (1.0, 0.6, 0.1, 0.9)}
STATION_RGBA = {'covered': (0.6, 0.6, 0.6, 0.5), 'open': (0.55, 0.8, 1.0, 0.35), 'unreachable': (1.0, 0.25, 0.2, 0.6)}
HEARING_RGBA = {'new': (0.0, 0.9, 1.0, 0.9), 'hunt': (1.0, 0.6, 0.1, 0.9), 'ignored': (0.6, 0.6, 0.6, 0.5)}


def build_scene(data, stamp, gf):
    """The complete /coqui_survey/markers scene from plain data (no node needed): DELETEALL first, then every
    marker, all in frame gf except the HUD. data keys, all optional:
      area [xmin, xmax, ymin, ymax] and stations [(x, y, 'covered'|'open'|'unreachable')], drawn at
      z (the ground height; None before the start: not drawn), frogs [survey records], estimate (pos,
      uncertainty_m), rays [(start, end, 'inlier'|'outlier'|'pending')], anchor (origin, dir),
      goal (x, y, yaw, label), aim (from xyz, to xyz), hud [lines] drawn above the robot in base_frame."""
    ma = MarkerArray()
    clear = Marker()
    clear.header.frame_id, clear.header.stamp, clear.action = gf, stamp, Marker.DELETEALL
    ma.markers.append(clear)

    def mk(ns, i, typ, rgba, scale):
        m = Marker()
        m.header.frame_id, m.header.stamp, m.ns, m.id, m.type = gf, stamp, ns, i, typ
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = (float(v) for v in scale)
        m.color.r, m.color.g, m.color.b, m.color.a = (float(v) for v in rgba)
        return m

    def pt(v, dz=0.0):
        return Point(x=float(v[0]), y=float(v[1]), z=float(v[2]) + dz)

    z = None if data.get('z') is None else float(data['z'])
    if z is not None:
        if data.get('area') is not None:
            xmin, xmax, ymin, ymax = (float(v) for v in data['area'])
            area = mk('area', 0, Marker.LINE_STRIP, (1.0, 1.0, 1.0, 0.6), (0.08, 0.0, 0.0))
            area.points = [Point(x=x, y=y, z=z) for x, y in
                           ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin))]
            ma.markers.append(area)
        for i, (x, y, status) in enumerate(data.get('stations') or []):
            s = mk('stations', i, Marker.CYLINDER, STATION_RGBA.get(status, STATION_RGBA['open']), (0.6, 0.6, 0.05))
            s.pose.position = Point(x=float(x), y=float(y), z=z + 0.05)
            ma.markers.append(s)
    for i, f in enumerate(data.get('frogs') or []):
        try:
            pos = pt(f['pos'])
        except (KeyError, IndexError, TypeError, ValueError):   # malformed record from a resumed survey file
            continue
        seen = bool(f.get('visually_confirmed'))
        s = mk('frogs', i, Marker.SPHERE, FROG_PHOTO_RGBA if seen else FROG_ACOUSTIC_RGBA, (0.35, 0.35, 0.35))
        s.pose.position = pos
        t = mk('frog_labels', i, Marker.TEXT_VIEW_FACING, WHITE, (0.0, 0.0, 0.20))
        t.pose.position = Point(x=pos.x, y=pos.y, z=pos.z + 0.38)
        note, kind = ('photo' if seen else 'no photo'), f.get('height_class')
        t.text = f"{f.get('id', 'frog')}\n" + (f'{kind}, {note}' if kind else note)
        ma.markers += [s, t]
    est_pos = None
    if data.get('estimate') is not None:
        est_pos, unc = data['estimate']
        unc = float('nan') if unc is None else float(unc)
        size = max(0.25, min(unc, 2.0)) if math.isfinite(unc) else 1.0
        s = mk('estimate', 0, Marker.SPHERE, ESTIMATE_RGB + (0.5,), (size, size, size))
        s.pose.position = pt(est_pos)
        t = mk('estimate_label', 0, Marker.TEXT_VIEW_FACING, ESTIMATE_RGB + (1.0,), (0.0, 0.0, 0.20))
        t.pose.position = pt(est_pos, size / 2 + 0.22)
        t.text = f'estimate +/-{unc:.2f} m' if math.isfinite(unc) else 'estimate'
        ma.markers += [s, t]
    rays = data.get('rays') or []
    for i, kind in enumerate(('inlier', 'outlier', 'pending')):
        points = [pt(p) for seg in rays if seg[2] == kind for p in seg[:2]]
        if points:
            m = mk('rays', i, Marker.LINE_LIST, RAY_RGBA[kind], (0.02, 0.0, 0.0))
            m.points = points
            ma.markers.append(m)
    if data.get('anchor') is not None:
        seg = ray_segments([(0.0, data['anchor'][0], data['anchor'][1], 0.0)], None, est_pos)
        if seg:
            m = mk('anchor_ray', 0, Marker.LINE_LIST, ANCHOR_RGBA, (0.04, 0.0, 0.0))
            m.points = [pt(seg[0][0]), pt(seg[0][1])]
            ma.markers.append(m)
    if data.get('goal') is not None:
        x, y, yaw, label = data['goal']
        gz = 0.0 if z is None else z
        a = mk('goal', 0, Marker.ARROW, GOAL_RGB + (0.9,), (0.6, 0.12, 0.12))
        a.pose.position = Point(x=float(x), y=float(y), z=gz + 0.3)
        a.pose.orientation.z, a.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        t = mk('goal_label', 0, Marker.TEXT_VIEW_FACING, GOAL_RGB + (1.0,), (0.0, 0.0, 0.20))
        t.pose.position = Point(x=float(x), y=float(y), z=gz + 0.6)
        t.text = str(label)
        ma.markers += [a, t]
    if data.get('aim') is not None:
        m = mk('aim', 0, Marker.LINE_STRIP, AIM_RGBA, (0.03, 0.0, 0.0))
        m.points = [pt(data['aim'][0]), pt(data['aim'][1])]
        ma.markers.append(m)
    hud = data.get('hud')
    if hud:
        t = mk('hud', 0, Marker.TEXT_VIEW_FACING, WHITE, (0.0, 0.0, 0.18))
        t.header.frame_id = data.get('base_frame') or 'base_footprint'
        t.header.stamp = MsgTime()     # latest TF, re-applied every frame: the text rides above the robot
        t.frame_locked = True
        t.pose.position.z = 1.15
        t.text = hud if isinstance(hud, str) else '\n'.join(hud)
        ma.markers.append(t)
    return ma


def hearing_marker(origin, direction, use, idx, stamp, gf, length=4.0):
    """One /coqui_survey/hearing arrow (ns 'doa', id idx, lives 2 s): a bearing drawn from the mic array origin
    `length` m along its map-frame direction, coloured by what the mission did with it: 'new' cyan (considered
    as a new call), 'hunt' orange (collected for the current hunt), anything else grey (ignored)."""
    o = [float(v) for v in origin[:3]]
    e = [o[k] + length * float(direction[k]) for k in range(3)]
    m = Marker()
    m.header.frame_id, m.header.stamp, m.ns, m.id, m.type = gf, stamp, 'doa', int(idx), Marker.ARROW
    m.pose.orientation.w = 1.0
    m.scale.x, m.scale.y, m.scale.z = 0.03, 0.1, 0.2       # shaft diameter, head diameter, head length
    m.color.r, m.color.g, m.color.b, m.color.a = HEARING_RGBA.get(use, HEARING_RGBA['ignored'])
    m.lifetime = MsgDuration(sec=2)
    m.points = [Point(x=o[0], y=o[1], z=o[2]), Point(x=e[0], y=e[1], z=e[2])]
    return m


def main(args=None):
    rclpy.init(args=args)
    node = CoquiSurveyNode()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    worker = threading.Thread(target=node.run, daemon=True)
    worker.start()
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.store.log_event('shutdown')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
