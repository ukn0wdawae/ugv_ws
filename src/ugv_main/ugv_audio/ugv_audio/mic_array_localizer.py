#!/usr/bin/env python3
"""5-Channel 3D Microphone Array Processor & Acoustic Triangulator for ugv_pro_audio.

Processes raw 5-channel audio frames from `/ugv_audio/mic_array/audio_raw`:
  1. Bandpass spectral filtering in the Coqui frequency band (1000 Hz - 2400 Hz)
     and dual-peak ("Co" ~1160 Hz + "Qui" ~1820-2180 Hz) call detection.
  2. Coarse-to-Fine 3D Steered Response Power with Phase Transform (SRP-PHAT)
     across the 5-microphone 3D array to estimate 3D Direction of Arrival
     (azimuth and elevation) in `audio_array_base_link`.
  3. Multi-position 3D Ray-Intersection Least-Squares Triangulation in `map` frame
     using TF2 (`map` <- `audio_array_base_link`) to triangulate the Coqui frog's
     3D coordinates (x, y, z) and uncertainty radius.

Every detected call is also published on `/ugv_audio/call` (std_msgs/String, JSON), which is what the survey uses:
  t          capture time (s, ROS clock)      frame      frame of az / el (the array)
  az, el     direction of arrival (rad)       snr_db     over the noise floor: says how audible the call is
  level_db   received level, 0 dB = a nominal call 1 m away, -20 dB = 10 m: says roughly how far the frog is
  coherence  0..1, share of the call that one direction explains; low = two frogs at once, bearing not usable
"""

import json
import math
import numpy as np

import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import Bool, Float32MultiArray, String
from geometry_msgs.msg import Point, PointStamped, Vector3Stamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros

from .audio_core import (
    COQUI_MASK,
    FRAME_SAMPLES,
    MIC_POSITIONS_BODY,
    SAMPLE_RATE,
    call_band_level_db,
    compute_coqui_snr_db,
    estimate_doa,
    level_to_range_m,
    quat_to_rot_matrix,
)


class MicArrayLocalizerNode(Node):
    def __init__(self):
        super().__init__('mic_array_localizer')

        self.declare_parameter('snr_threshold_db', 6.0)
        self.declare_parameter('min_baseline_for_convergence_m', 0.45)
        self.declare_parameter('min_rays_for_convergence', 3)
        self.declare_parameter('max_stored_rays', 30)
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('array_frame', 'audio_array_base_link')
        # Ray gating (0 = off). Once the buffer holds rays, a new bearing is only used if it points within
        # this many degrees of the current estimate, so a different frog calling mid-hunt does not get
        # mixed into the triangulation. Not applied when the estimate is closer than gate_min_range_m.
        self.declare_parameter('ray_gate_deg', 0.0)
        self.declare_parameter('gate_min_range_m', 1.5)

        self.snr_threshold_db = float(self.get_parameter('snr_threshold_db').value)
        self.min_baseline = float(self.get_parameter('min_baseline_for_convergence_m').value)
        self.min_rays = int(self.get_parameter('min_rays_for_convergence').value)
        self.max_rays = int(self.get_parameter('max_stored_rays').value)
        self.global_frame = self.get_parameter('global_frame').value
        self.array_frame = self.get_parameter('array_frame').value
        self.ray_gate = math.radians(float(self.get_parameter('ray_gate_deg').value))
        self.gate_min_range = float(self.get_parameter('gate_min_range_m').value)
        self.rays_rejected = 0

        # TF2 buffer & listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Fallback odometry if map->audio_array_base_link TF is still warming up
        self.latest_odom_pos = np.array([0.0, 0.0, 0.115], dtype=np.float64)
        self.latest_odom_rot = np.eye(3, dtype=np.float64)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)

        # Stored acoustic bearing rays: list of dicts {'origin': np(3), 'dir': np(3), 'range_est': float, 'weight': float}
        self.rays = []
        self.triangulated_pos = None
        self.uncertainty_m = 9.99

        # Subscriptions & Publishers
        self.create_subscription(
            Float32MultiArray, '/ugv_audio/mic_array/audio_raw', self._audio_cb, 10
        )
        self.create_subscription(
            String, '/coqui/reset_triangulation', self._reset_cb, 10
        )

        self.detected_pub = self.create_publisher(Bool, '/ugv_audio/sound_detected', 10)
        self.doa_pub = self.create_publisher(Vector3Stamped, '/ugv_audio/doa_bearing', 10)
        self.call_pub = self.create_publisher(String, '/ugv_audio/call', 50)
        self.frog_pt_pub = self.create_publisher(PointStamped, '/ugv_audio/triangulated_frog', 10)
        self.status_pub = self.create_publisher(String, '/ugv_audio/triangulation_status', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/ugv_audio/triangulation_markers', 10)

        self.get_logger().info('5-Channel 3D Microphone Array Localizer & Triangulator ready.')

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.latest_odom_pos = np.array([p.x, p.y, p.z + 0.115], dtype=np.float64)
        self.latest_odom_rot = quat_to_rot_matrix(q.x, q.y, q.z, q.w)

    def _reset_cb(self, _msg: String):
        self.rays.clear()
        self.triangulated_pos = None
        self.uncertainty_m = 9.99
        self.rays_rejected = 0
        ma = MarkerArray()
        m_del = Marker()
        m_del.action = Marker.DELETEALL
        ma.markers.append(m_del)
        self.marker_pub.publish(ma)
        self.get_logger().info('Reset acoustic triangulation ray buffer.')

    def _get_array_pose_in_map(self):
        """Return (origin_3d, rot_3x3) of `audio_array_base_link` in `map` frame."""
        for target_frame in (self.global_frame, 'odom'):
            try:
                tf = self.tf_buffer.lookup_transform(
                    target_frame, self.array_frame, rclpy.time.Time(), timeout=Duration(seconds=0.05)
                )
                t = tf.transform.translation
                q = tf.transform.rotation
                return (
                    np.array([t.x, t.y, t.z], dtype=np.float64),
                    quat_to_rot_matrix(q.x, q.y, q.z, q.w),
                )
            except Exception:
                pass
        return self.latest_odom_pos.copy(), self.latest_odom_rot.copy()

    def _solve_3d_triangulation(self):
        """Solve weighted 3D ray-intersection least squares + range prior regularization."""
        if not self.rays:
            return None, 9.99, 0.0

        # Compute maximum baseline between ray origins
        origins = np.array([r['origin'] for r in self.rays], dtype=np.float64)
        if len(origins) >= 2:
            diffs = origins[:, None, :] - origins[None, :, :]
            baseline = float(np.max(np.linalg.norm(diffs[:, :, :2], axis=2)))
        else:
            baseline = 0.0

        # Accumulate ray projection matrices: sum w_k (I - d_k d_k^T) x = sum w_k (I - d_k d_k^T) p_k
        A = np.zeros((3, 3), dtype=np.float64)
        b = np.zeros(3, dtype=np.float64)
        prior_pts = []
        prior_weights = []

        for idx, r in enumerate(self.rays):
            d = r['dir']
            p = r['origin']
            # Give slightly higher weight to more recent measurements and higher SNR
            recency_w = 0.5 + 0.5 * ((idx + 1) / len(self.rays))
            w = float(r['weight']) * recency_w
            P_orth = np.eye(3) - np.outer(d, d)
            A += w * P_orth
            b += w * (P_orth @ p)
            prior_pts.append(p + float(r['range_est']) * d)
            prior_weights.append(w)

        prior_weights = np.array(prior_weights)
        x_prior = np.average(np.array(prior_pts), axis=0, weights=prior_weights)

        # Regularization decays as baseline parallax grows
        reg_xy = 0.35 / (1.0 + (baseline / 0.45) ** 2)
        # Vertical z has slightly stronger prior from elevation ray + range because robot moves mostly in 2D
        reg_z = 0.45 / (1.0 + (baseline / 0.80) ** 2)
        R_reg = np.diag([reg_xy, reg_xy, reg_z])

        x_est = np.linalg.solve(A + R_reg, b + R_reg @ x_prior)

        # Estimate uncertainty from residual perpendicular distances + baseline
        perp_Residuals = []
        for r in self.rays[-min(len(self.rays), 10):]:
            v = x_est - r['origin']
            d = r['dir']
            perp = v - np.dot(v, d) * d
            perp_Residuals.append(np.linalg.norm(perp))
        rms_perp = float(np.mean(perp_Residuals)) if perp_Residuals else 1.0
        uncertainty = rms_perp + 1.2 / (1.0 + 2.5 * baseline + 0.3 * len(self.rays))
        return x_est, float(uncertainty), baseline

    def _audio_cb(self, msg: Float32MultiArray):
        data = np.array(msg.data, dtype=np.float64)
        num_mics = MIC_POSITIONS_BODY.shape[0]
        if data.size != num_mics * FRAME_SAMPLES:
            return
        signals = data.reshape((num_mics, FRAME_SAMPLES))

        # 1. Harmonic-free, motor-noise-resilient spectral SNR & Coqui call detection
        snr_db, specs, power_spec = compute_coqui_snr_db(signals, SAMPLE_RATE)

        is_detected = snr_db >= self.snr_threshold_db
        det_msg = Bool()
        det_msg.data = bool(is_detected)
        self.detected_pub.publish(det_msg)

        if not is_detected:
            return

        # 2. Estimate 3D Direction of Arrival (DOA) in `audio_array_base_link`
        spec_coqui = specs[:, COQUI_MASK]
        az_rad, el_rad, u_body, coherence = estimate_doa(spec_coqui)
        level_db = call_band_level_db(specs, power_spec)

        now_msg = self.get_clock().now().to_msg()
        doa_msg = Vector3Stamped()
        doa_msg.header.stamp = now_msg
        doa_msg.header.frame_id = self.array_frame
        doa_msg.vector.x = float(az_rad)
        doa_msg.vector.y = float(el_rad)
        doa_msg.vector.z = float(snr_db)
        self.doa_pub.publish(doa_msg)
        call = String()
        call.data = json.dumps({
            't': now_msg.sec + now_msg.nanosec * 1e-9, 'frame': self.array_frame,
            'az': round(az_rad, 4), 'el': round(el_rad, 4), 'snr_db': round(snr_db, 2),
            'level_db': round(level_db, 2), 'coherence': round(coherence, 3)})
        self.call_pub.publish(call)

        # 3. Transform DOA ray into `map` frame and update 3D triangulation
        origin_map, rot_map = self._get_array_pose_in_map()
        dir_map = rot_map @ u_body
        dir_map = dir_map / np.linalg.norm(dir_map)

        range_est = float(np.clip(level_to_range_m(level_db), 0.5, 25.0))   # prior until there is parallax

        if self.ray_gate > 0.0 and self.rays:
            if self.triangulated_pos is not None:
                ref = np.asarray(self.triangulated_pos, dtype=np.float64)
            else:
                r0 = self.rays[0]
                ref = r0['origin'] + float(r0['range_est']) * r0['dir']
            to_ref = ref[:2] - origin_map[:2]
            if np.linalg.norm(to_ref) >= self.gate_min_range and np.linalg.norm(dir_map[:2]) > 1e-6:
                dev = abs(math.atan2(math.sin(math.atan2(dir_map[1], dir_map[0]) - math.atan2(to_ref[1], to_ref[0])),
                                     math.cos(math.atan2(dir_map[1], dir_map[0]) - math.atan2(to_ref[1], to_ref[0]))))
                if dev > self.ray_gate:
                    self.rays_rejected += 1
                    return

        # Store ray (avoid storing duplicate rays from the exact same spot if we already have many)
        weight = float(np.clip((snr_db - self.snr_threshold_db) / 15.0, 0.2, 2.5))
        if len(self.rays) >= self.max_rays:
            # Drop the oldest ray unless it maximizes baseline
            self.rays.pop(0)
        self.rays.append({
            'origin': origin_map,
            'dir': dir_map,
            'range_est': range_est,
            'weight': weight,
        })

        self.triangulated_pos, self.uncertainty_m, baseline = self._solve_3d_triangulation()
        converged = (
            len(self.rays) >= self.min_rays
            and (baseline >= self.min_baseline or range_est < 2.6)
            and self.uncertainty_m < 0.85
        )

        # Publish triangulated 3D point in map frame
        pt_msg = PointStamped()
        pt_msg.header.stamp = now_msg
        pt_msg.header.frame_id = self.global_frame
        pt_msg.point.x = float(self.triangulated_pos[0])
        pt_msg.point.y = float(self.triangulated_pos[1])
        pt_msg.point.z = float(self.triangulated_pos[2])
        self.frog_pt_pub.publish(pt_msg)

        # Publish detailed JSON status
        status = {
            'detected': True,
            'converged': bool(converged),
            'triangulated_pos': [
                round(float(self.triangulated_pos[0]), 3),
                round(float(self.triangulated_pos[1]), 3),
                round(float(self.triangulated_pos[2]), 3),
            ],
            'uncertainty_m': round(float(self.uncertainty_m), 3),
            'baseline_m': round(float(baseline), 3),
            'num_rays': len(self.rays),
            'azimuth_deg': round(math.degrees(az_rad), 1),
            'elevation_deg': round(math.degrees(el_rad), 1),
            'range_est_m': round(range_est, 2),
            'snr_db': round(snr_db, 1),
            'rays_rejected': int(self.rays_rejected),
            'ray_origin': [round(float(v), 3) for v in origin_map],
            'ray_dir': [round(float(v), 4) for v in dir_map],
        }
        st_msg = String()
        st_msg.data = json.dumps(status)
        self.status_pub.publish(st_msg)

        self._publish_triangulation_markers(now_msg)

    def _publish_triangulation_markers(self, stamp):
        if self.triangulated_pos is None:
            return
        ma = MarkerArray()

        # 1. Triangulated Frog 3D Sphere
        m_est = Marker()
        m_est.header.frame_id = self.global_frame
        m_est.header.stamp = stamp
        m_est.ns = 'coqui_triangulation'
        m_est.id = 0
        m_est.type = Marker.SPHERE
        m_est.action = Marker.ADD
        m_est.pose.position.x = float(self.triangulated_pos[0])
        m_est.pose.position.y = float(self.triangulated_pos[1])
        m_est.pose.position.z = float(self.triangulated_pos[2])
        m_est.pose.orientation.w = 1.0
        s = max(0.25, min(float(self.uncertainty_m), 1.5))
        m_est.scale.x = s
        m_est.scale.y = s
        m_est.scale.z = s
        m_est.color.r = 1.0
        m_est.color.g = 0.85
        m_est.color.b = 0.0
        m_est.color.a = 0.65
        ma.markers.append(m_est)

        # 2. Recent DOA Bearing Rays
        m_rays = Marker()
        m_rays.header.frame_id = self.global_frame
        m_rays.header.stamp = stamp
        m_rays.ns = 'coqui_doa_rays'
        m_rays.id = 1
        m_rays.type = Marker.LINE_LIST
        m_rays.action = Marker.ADD
        m_rays.scale.x = 0.035
        m_rays.color.r = 0.0
        m_rays.color.g = 0.92
        m_rays.color.b = 1.0
        m_rays.color.a = 0.80
        for r in self.rays[-10:]:
            p0 = r['origin']
            p1 = p0 + min(float(r['range_est']) * 1.15, 9.0) * r['dir']
            m_rays.points.append(Point(x=float(p0[0]), y=float(p0[1]), z=float(p0[2])))
            m_rays.points.append(Point(x=float(p1[0]), y=float(p1[1]), z=float(p1[2])))
        ma.markers.append(m_rays)

        # 3. 3D Text label above the triangulated frog estimate
        m_txt = Marker()
        m_txt.header.frame_id = self.global_frame
        m_txt.header.stamp = stamp
        m_txt.ns = 'coqui_triangulation_label'
        m_txt.id = 2
        m_txt.type = Marker.TEXT_VIEW_FACING
        m_txt.action = Marker.ADD
        m_txt.pose.position.x = float(self.triangulated_pos[0])
        m_txt.pose.position.y = float(self.triangulated_pos[1])
        m_txt.pose.position.z = float(self.triangulated_pos[2]) + 0.68
        m_txt.pose.orientation.w = 1.0
        m_txt.scale.z = 0.20
        m_txt.color.r = 1.0
        m_txt.color.g = 0.90
        m_txt.color.b = 0.1
        m_txt.color.a = 0.95
        m_txt.text = (
            f"TRIANGULATED ({self.triangulated_pos[0]:.1f}, "
            f"{self.triangulated_pos[1]:.1f}, {self.triangulated_pos[2]:.1f}) "
            f"+/-{self.uncertainty_m:.2f}m"
        )
        ma.markers.append(m_txt)

        self.marker_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = MicArrayLocalizerNode()
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

