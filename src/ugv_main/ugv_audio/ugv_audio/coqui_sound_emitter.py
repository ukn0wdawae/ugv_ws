#!/usr/bin/env python3
"""Multi-Frog 3D Coqui Acoustic Environment Simulator for ugv_pro_audio.

All Coqui frogs placed across the map (in the open, behind a tree, inside an
untraversable bush thicket, behind an object, and perched on a tree) are active
in the environment simultaneously without any `scenario` tag.

Synthesizes authentic Puerto Rican Coqui frog calls ("Co-Qui": ~1160 Hz 'Co' note
followed by ~1820->2180 Hz upward frequency sweep 'Qui' note) and simulates
3D acoustic wave propagation from the calling Coqui frog in Gazebo to each of the
5 microphones on `ugv_pro_audio`:
  - mic_fl_link  (+0.095, +0.075, 0.000)
  - mic_fr_link  (+0.095, -0.075, 0.000)
  - mic_rl_link  (-0.095, +0.075, 0.000)
  - mic_rr_link  (-0.095, -0.075, 0.000)
  - mic_top_link (-0.065,  0.000, 0.040)

Applies sub-sample Fourier fractional time-of-flight delays (tau_i = d_i / 343.0 m/s),
inverse-distance spherical spreading attenuation, foliage/obstacle occlusion attenuation,
and background noise, publishing 5-channel audio frames on `/ugv_audio/mic_array/audio_raw`.
"""

import json
import math
import os
import wave
import yaml
import numpy as np

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Bool, Float32MultiArray, MultiArrayDimension, String
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray


SPEED_OF_SOUND = 343.0  # m/s in air at ~20 C
SAMPLE_RATE = 16000     # Hz
FRAME_SAMPLES = 2048    # ~128 ms per audio frame

# Microphone coordinates in `audio_array_base_link` frame (meters)
MIC_POSITIONS_BODY = np.array([
    [0.095,  0.075, 0.000],  # 0: Front-Left (mic_fl_link)
    [0.095, -0.075, 0.000],  # 1: Front-Right (mic_fr_link)
    [-0.095, 0.075, 0.000],  # 2: Rear-Left (mic_rl_link)
    [-0.095, -0.075, 0.000], # 3: Rear-Right (mic_rr_link)
    [-0.065, 0.000, 0.040],  # 4: Top-Center (mic_top_link)
], dtype=np.float64)

ARRAY_BASE_OFFSET_IN_FOOTPRINT = np.array([0.0, 0.0, 0.115], dtype=np.float64)


def quat_to_rot_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to 3x3 rotation matrix."""
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def synthesize_coqui_call_frame(num_samples: int, sample_rate: int, phase_seed: float = 0.0) -> np.ndarray:
    """Synthesize a single frame containing the two-note 'Co-Qui' call."""
    t = np.arange(num_samples, dtype=np.float64) / float(sample_rate)
    signal = np.zeros_like(t)

    # Note 1: "CO" (~1160 Hz constant tone with Hann envelope in first 38% of frame)
    n_co_start = int(0.05 * num_samples)
    n_co_end = int(0.40 * num_samples)
    t_co = t[n_co_start:n_co_end] - t[n_co_start]
    env_co = np.sin(np.pi * np.linspace(0.0, 1.0, n_co_end - n_co_start)) ** 2
    signal[n_co_start:n_co_end] = 0.75 * env_co * (
        np.sin(2.0 * np.pi * 1160.0 * t_co + phase_seed)
        + 0.25 * np.sin(2.0 * np.pi * 2320.0 * t_co + 0.5 * phase_seed)
    )

    # Note 2: "QUI" (~1820 Hz -> 2180 Hz upward chirp in 48%..92% of frame)
    n_qui_start = int(0.48 * num_samples)
    n_qui_end = int(0.92 * num_samples)
    t_qui = t[n_qui_start:n_qui_end] - t[n_qui_start]
    dur_qui = max(t_qui[-1], 1e-4)
    f0, f1 = 1820.0, 2180.0
    chirp_phase = 2.0 * np.pi * (f0 * t_qui + 0.5 * ((f1 - f0) / dur_qui) * (t_qui ** 2))
    env_qui = np.sin(np.pi * np.linspace(0.0, 1.0, n_qui_end - n_qui_start)) ** 1.5
    signal[n_qui_start:n_qui_end] = 1.0 * env_qui * (
        np.sin(chirp_phase + phase_seed)
        + 0.30 * np.sin(1.5 * chirp_phase)
    )
    return signal


def fractional_delay_fft(signal: np.ndarray, delay_samples: float) -> np.ndarray:
    """Apply exact sub-sample time delay via Fourier linear phase shift."""
    n = len(signal)
    spec = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0)
    phase_shift = np.exp(-1j * 2.0 * np.pi * freqs * delay_samples)
    return np.fft.irfft(spec * phase_shift, n=n)


class CoquiSoundEmitterNode(Node):
    def __init__(self):
        super().__init__('coqui_sound_emitter')

        self.declare_parameter('env', 'park')
        self.declare_parameter('call_period_sec', 0.75)
        self.declare_parameter('start_delay_sec', 4.5)
        self.declare_parameter('max_audible_range_m', 22.0)
        self.declare_parameter('noise_std', 0.008)

        self.env_name = self.get_parameter('env').value.strip().lower()
        self.call_period = float(self.get_parameter('call_period_sec').value)
        self.start_delay = float(self.get_parameter('start_delay_sec').value)
        self.max_range = float(self.get_parameter('max_audible_range_m').value)
        self.noise_std = float(self.get_parameter('noise_std').value)

        pkg_share = get_package_share_directory('ugv_audio')
        cfg_path = os.path.join(pkg_share, 'config', 'coqui_environments.yaml')
        with open(cfg_path, 'r') as f:
            full_cfg = yaml.safe_load(f)

        env_cfg = full_cfg['environments'].get(self.env_name, full_cfg['environments']['park'])
        self.spawn_pose = env_cfg['spawn']
        raw_frogs = env_cfg['frogs']
        if isinstance(raw_frogs, dict):
            self.frogs = list(raw_frogs.values())
        else:
            self.frogs = list(raw_frogs)

        self.visited_frog_names = set()
        self.locked_frog_idx = None
        self.current_caller_idx = 0
        self.pause_until_sec = 0.0

        # Robot pose in Gazebo world frame (initialized from default spawn pose)
        self.robot_pos_world = np.array(self.spawn_pose[:3], dtype=np.float64)
        yaw0 = float(self.spawn_pose[3])
        self.robot_rot_world = np.array([
            [math.cos(yaw0), -math.sin(yaw0), 0.0],
            [math.sin(yaw0),  math.cos(yaw0), 0.0],
            [0.0,             0.0,            1.0],
        ], dtype=np.float64)
        self.have_gt_odom = False

        self._save_reference_wav()

        # Subscriptions
        self.create_subscription(Odometry, '/ugv_gt_odom', self._gt_odom_cb, 10)
        self.create_subscription(Odometry, '/odom', self._odom_fallback_cb, 10)
        self.create_subscription(Bool, '/coqui/lock_active_caller', self._lock_caller_cb, 10)
        self.create_subscription(String, '/coqui_behavior/detection_report', self._report_cb, 10)

        # Publishers
        self.audio_pub = self.create_publisher(
            Float32MultiArray, '/ugv_audio/mic_array/audio_raw', 10
        )
        self.info_pub = self.create_publisher(
            String, '/coqui/active_frog_info', 10
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, '/coqui/emitter_markers', 10
        )

        self.start_time = self.get_clock().now()
        self.call_counter = 0
        self.timer = self.create_timer(self.call_period, self._emit_cycle)

        frog_summary = ', '.join(
            f"{f['name']} ({f.get('placement', 'open')} @ {f['pos']})" for f in self.frogs
        )
        self.get_logger().info(
            f"Multi-Frog Coqui Sound Emitter started in env='{self.env_name}' "
            f"with {len(self.frogs)} frogs on map: {frog_summary}"
        )

    def _save_reference_wav(self):
        try:
            out_dir = '/root/ugv_ws/coqui_captures'
            if not os.path.isdir('/root/ugv_ws'):
                out_dir = os.path.expanduser('~/coqui_captures')
            os.makedirs(out_dir, exist_ok=True)
            wav_path = os.path.join(out_dir, 'coqui_call_reference.wav')
            if not os.path.exists(wav_path):
                call = synthesize_coqui_call_frame(SAMPLE_RATE // 2, SAMPLE_RATE)
                pcm = np.clip(call * 30000.0, -32767, 32767).astype(np.int16)
                with wave.open(wav_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(pcm.tobytes())
        except Exception as exc:
            self.get_logger().debug(f"Could not write reference wav: {exc}")

    def _gt_odom_cb(self, msg: Odometry):
        self.have_gt_odom = True
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.robot_pos_world = np.array([p.x, p.y, p.z], dtype=np.float64)
        self.robot_rot_world = quat_to_rot_matrix(q.x, q.y, q.z, q.w)

    def _odom_fallback_cb(self, msg: Odometry):
        if self.have_gt_odom:
            return
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.robot_pos_world = np.array([p.x, p.y, p.z], dtype=np.float64)
        self.robot_rot_world = quat_to_rot_matrix(q.x, q.y, q.z, q.w)

    def _lock_caller_cb(self, msg: Bool):
        if msg.data:
            self.locked_frog_idx = self.current_caller_idx
        else:
            self.locked_frog_idx = None

    def _report_cb(self, msg: String):
        """When the hunter node finishes capturing a frog, mark it visited and advance."""
        try:
            rep = json.loads(msg.data)
            fname = rep.get('frog_name')
            if fname:
                self.visited_frog_names.add(fname)
                self.locked_frog_idx = None
                # Quiet period while robot starts returning to initial position
                elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
                self.pause_until_sec = elapsed + 6.0
                self.get_logger().info(
                    f"Marked frog '{fname}' as spotted/visited ({len(self.visited_frog_names)}/{len(self.frogs)})."
                )
        except Exception:
            pass

    def _select_active_calling_frog(self, array_origin_world: np.ndarray) -> int:
        """Select which unvisited frog is currently calling on the map."""
        if self.locked_frog_idx is not None and 0 <= self.locked_frog_idx < len(self.frogs):
            f_locked = self.frogs[self.locked_frog_idx]
            if f_locked['name'] not in self.visited_frog_names:
                return self.locked_frog_idx

        unvisited = [
            i for i, f in enumerate(self.frogs) if f['name'] not in self.visited_frog_names
        ]
        if not unvisited:
            # All frogs visited — reset visited set so continuous testing can continue if desired
            self.visited_frog_names.clear()
            unvisited = list(range(len(self.frogs)))

        # Pick the nearest unvisited frog to the robot's current exploration position
        best_i = unvisited[0]
        best_d = 1e9
        for i in unvisited:
            fpos = np.array(self.frogs[i]['pos'], dtype=np.float64)
            d = float(np.linalg.norm(array_origin_world[:2] - fpos[:2]))
            if d < best_d:
                best_d = d
                best_i = i
        self.current_caller_idx = best_i
        return best_i

    def _emit_cycle(self):
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9

        # Compute 3D world positions of the 5 microphones
        array_origin_world = (
            self.robot_pos_world + self.robot_rot_world @ ARRAY_BASE_OFFSET_IN_FOOTPRINT
        )
        mic_world_pts = array_origin_world + (MIC_POSITIONS_BODY @ self.robot_rot_world.T)

        caller_idx = self._select_active_calling_frog(array_origin_world)
        active = self.frogs[caller_idx]
        frog_pos = np.array(active['pos'], dtype=np.float64)

        dists = np.linalg.norm(mic_world_pts - frog_pos[None, :], axis=1)
        center_dist = float(np.linalg.norm(array_origin_world - frog_pos))

        # Synthesize source waveform if past start_delay, not in return-home pause, and in range
        is_calling = (
            (elapsed >= self.start_delay)
            and (elapsed >= self.pause_until_sec)
            and (center_dist <= self.max_range)
        )
        num_mics = MIC_POSITIONS_BODY.shape[0]
        multi_ch = np.random.normal(0.0, self.noise_std, size=(num_mics, FRAME_SAMPLES))

        if is_calling:
            self.call_counter += 1
            base_wave = synthesize_coqui_call_frame(
                FRAME_SAMPLES, SAMPLE_RATE, phase_seed=0.37 * self.call_counter
            )
            tau_center = center_dist / SPEED_OF_SOUND
            foliage_gain = float(active.get('foliage_attenuation', 1.0))

            for i in range(num_mics):
                tau_i = float(dists[i]) / SPEED_OF_SOUND
                rel_delay_samples = (tau_i - tau_center) * float(SAMPLE_RATE)
                attenuation = foliage_gain / max(float(dists[i]), 0.35)
                delayed = fractional_delay_fft(base_wave, rel_delay_samples)
                multi_ch[i, :] += attenuation * delayed

        # Publish 5-channel audio frame
        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label='channels', size=num_mics, stride=num_mics * FRAME_SAMPLES),
            MultiArrayDimension(label='samples', size=FRAME_SAMPLES, stride=FRAME_SAMPLES),
        ]
        msg.layout.data_offset = 0
        msg.data = multi_ch.astype(np.float32).ravel().tolist()
        self.audio_pub.publish(msg)

        # Publish metadata for RViz / diagnostics
        info = {
            'env': self.env_name,
            'frog_id': active.get('id', active['name']),
            'frog_name': active['name'],
            'placement': active.get('placement', 'open'),
            'description': active.get('description', ''),
            'vantage_hint_angle': float(active.get('vantage_hint_angle', 0.0)),
            'standoff_m': float(active.get('standoff_m', 1.55)),
            'calling': bool(is_calling),
            'distance_m': round(center_dist, 3),
            'visited_count': len(self.visited_frog_names),
            'total_frogs': len(self.frogs),
        }
        info_msg = String()
        info_msg.data = json.dumps(info)
        self.info_pub.publish(info_msg)

        self._publish_markers(caller_idx, is_calling)

    def _publish_markers(self, caller_idx: int, is_calling: bool):
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()
        for idx, fcfg in enumerate(self.frogs):
            fname = fcfg['name']
            placement = fcfg.get('placement', 'open')
            is_active = (idx == caller_idx)
            is_visited = (fname in self.visited_frog_names)

            # 1. Frog position marker sphere
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = now
            m.ns = 'coqui_frogs_env'
            m.id = idx
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(fcfg['pos'][0])
            m.pose.position.y = float(fcfg['pos'][1])
            m.pose.position.z = float(fcfg['pos'][2]) + 0.08
            m.pose.orientation.w = 1.0
            s = 0.28 if (is_active and is_calling) else 0.18
            m.scale.x = s
            m.scale.y = s
            m.scale.z = s
            if is_visited:
                m.color.r, m.color.g, m.color.b, m.color.a = (0.1, 0.85, 1.0, 0.85)
            elif is_active and is_calling:
                m.color.r, m.color.g, m.color.b, m.color.a = (0.1, 1.0, 0.2, 0.95)
            else:
                m.color.r, m.color.g, m.color.b, m.color.a = (0.65, 0.85, 0.2, 0.65)
            ma.markers.append(m)

            # 2. Text label above each frog showing placement type & state
            lbl = Marker()
            lbl.header.frame_id = 'map'
            lbl.header.stamp = now
            lbl.ns = 'coqui_frogs_labels'
            lbl.id = 100 + idx
            lbl.type = Marker.TEXT_VIEW_FACING
            lbl.action = Marker.ADD
            lbl.pose.position.x = float(fcfg['pos'][0])
            lbl.pose.position.y = float(fcfg['pos'][1])
            lbl.pose.position.z = float(fcfg['pos'][2]) + 0.42
            lbl.pose.orientation.w = 1.0
            lbl.scale.z = 0.22
            state_tag = 'SPOTTED' if is_visited else ('CALLING' if (is_active and is_calling) else 'QUIET')
            lbl.text = f"{placement.upper()} [{state_tag}]"
            lbl.color.r = 1.0
            lbl.color.g = 1.0 if not is_active else 0.95
            lbl.color.b = 1.0 if not is_active else 0.2
            lbl.color.a = 0.95
            ma.markers.append(lbl)

            # 3. Expanding sound wave ring around the actively calling frog
            if is_active and is_calling:
                ring = Marker()
                ring.header.frame_id = 'map'
                ring.header.stamp = now
                ring.ns = 'coqui_sound_wave'
                ring.id = 200
                ring.type = Marker.CYLINDER
                ring.action = Marker.ADD
                ring.pose.position.x = float(fcfg['pos'][0])
                ring.pose.position.y = float(fcfg['pos'][1])
                ring.pose.position.z = float(fcfg['pos'][2]) + 0.04
                ring.pose.orientation.w = 1.0
                pulse = 0.55 + 0.35 * (self.call_counter % 4)
                ring.scale.x = pulse * 2.0
                ring.scale.y = pulse * 2.0
                ring.scale.z = 0.03
                ring.color.r, ring.color.g, ring.color.b, ring.color.a = (0.2, 1.0, 0.3, 0.32)
                ma.markers.append(ring)

        self.marker_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = CoquiSoundEmitterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
