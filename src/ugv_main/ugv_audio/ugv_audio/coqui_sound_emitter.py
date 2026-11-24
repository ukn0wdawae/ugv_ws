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
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Bool, Float32MultiArray, MultiArrayDimension, String
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray

from .audio_core import (
    ARRAY_BASE_OFFSET_IN_FOOTPRINT,
    FRAME_SAMPLES,
    MIC_POSITIONS_BODY,
    SAMPLE_RATE,
    SPEED_OF_SOUND,
    fractional_delay_fft,
    propagate_sources_to_array,
    quat_to_rot_matrix,
    synthesize_coqui_call_frame,
)

# Survey mode: viewer-only simulation truth layer on /sim/coqui_truth_markers (latched complete scene)
TRUTH_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
TRUTH_RGB = (1.0, 0.2, 0.9)   # magenta


def make_marker(stamp, ns: str, mid: int, mtype: int, xyz, scale, rgba) -> Marker:
    """A map-frame ADD marker, every field set with Python floats."""
    m = Marker()
    m.header.frame_id = 'map'
    m.header.stamp = stamp
    m.ns = ns
    m.id = int(mid)
    m.type = mtype
    m.action = Marker.ADD
    m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in xyz)
    m.pose.orientation.w = 1.0
    m.scale.x, m.scale.y, m.scale.z = (float(v) for v in scale)
    m.color.r, m.color.g, m.color.b, m.color.a = (float(v) for v in rgba)
    return m


class CoquiSoundEmitterNode(Node):
    def __init__(self):
        super().__init__('coqui_sound_emitter')

        self.declare_parameter('env', 'park')
        self.declare_parameter('call_period_sec', 0.75)
        self.declare_parameter('start_delay_sec', 4.5)
        self.declare_parameter('max_audible_range_m', 22.0)
        self.declare_parameter('noise_std', 0.008)
        # mode 'legacy' : compact coqui worlds + coqui_hunter_node (robot locks/marks frogs, frog hints
        #                 published on /coqui/active_frog_info).
        # mode 'survey' : frogs from `truth_file`, calling in bouts on their own schedule. The emitter
        #                 ignores the robot and publishes nothing but audio (no hints / answer key), plus,
        #                 only with publish_truth_markers, a viewer-only truth layer on /sim/coqui_truth_markers.
        self.declare_parameter('mode', 'legacy')
        self.declare_parameter('truth_file', '')
        self.declare_parameter('bout_calls_min', 10)
        self.declare_parameter('bout_calls_max', 18)
        self.declare_parameter('bout_gap_sec', 2.0)
        self.declare_parameter('seed', 0)
        self.declare_parameter('publish_truth_markers', False)
        # Realism parameters: motor self-noise, overlapping chorus calls, multipath echo, ambient insects
        self.declare_parameter('motor_noise_gain', 0.025)
        self.declare_parameter('overlap_prob', 0.25)
        self.declare_parameter('echo_gain', 0.12)
        self.declare_parameter('echo_delay_ms', 18.0)
        self.declare_parameter('ambient_clutter_prob', 0.20)

        self.env_name = self.get_parameter('env').value.strip().lower()
        self.call_period = float(self.get_parameter('call_period_sec').value)
        self.start_delay = float(self.get_parameter('start_delay_sec').value)
        self.max_range = float(self.get_parameter('max_audible_range_m').value)
        self.noise_std = float(self.get_parameter('noise_std').value)
        self.mode = str(self.get_parameter('mode').value).strip().lower()
        self.survey = self.mode == 'survey'
        self.bout_calls = (int(self.get_parameter('bout_calls_min').value),
                           int(self.get_parameter('bout_calls_max').value))
        self.bout_gap = float(self.get_parameter('bout_gap_sec').value)
        self.publish_truth_markers = bool(self.get_parameter('publish_truth_markers').value)
        self.motor_noise_gain = float(self.get_parameter('motor_noise_gain').value)
        self.overlap_prob = float(self.get_parameter('overlap_prob').value)
        self.echo_gain = float(self.get_parameter('echo_gain').value)
        self.echo_delay_ms = float(self.get_parameter('echo_delay_ms').value)
        self.ambient_clutter_prob = float(self.get_parameter('ambient_clutter_prob').value)
        self.rng = np.random.default_rng(int(self.get_parameter('seed').value))
        self.robot_speed = 0.0
        self.robot_wz = 0.0

        if self.survey:
            truth_file = str(self.get_parameter('truth_file').value)
            if not truth_file:
                truth_file = os.path.join(get_package_share_directory('ugv_audio'), 'config', 'sim_truth',
                                          'forest_survey.yaml')
            with open(truth_file, 'r') as f:
                truth = yaml.safe_load(f)
            self.spawn_pose = truth['spawn']
            # body/call point is 5 cm above the model origin; habitat = short label for the truth layer
            self.frogs = [{'name': fr['id'], 'pos': [fr['pos'][0], fr['pos'][1], fr['pos'][2] + 0.05],
                           'foliage_attenuation': float(fr.get('attenuation', 1.0)),
                           'habitat': str(fr.get('habitat', '')).split('(')[0].strip()[:30].rstrip()}
                          for fr in truth['frogs']]
            self.env_name = truth.get('world', 'survey')
            self.bout_frog = None        # frog index of the current calling bout
            self.bout_left = 0           # calls left in the bout
            self.bout_gap_until = 0.0    # sim time until which nobody calls (between bouts)
        else:
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
        if not self.survey:
            self.create_subscription(Bool, '/coqui/lock_active_caller', self._lock_caller_cb, 10)
            self.create_subscription(String, '/coqui_behavior/detection_report', self._report_cb, 10)

        # Publishers
        self.audio_pub = self.create_publisher(
            Float32MultiArray, '/ugv_audio/mic_array/audio_raw', 10
        )
        self.info_pub = None if self.survey else self.create_publisher(
            String, '/coqui/active_frog_info', 10
        )
        # In survey mode the true frog positions are only drawn when explicitly asked for, on a
        # clearly-named /sim/ topic, for a human watching RViz. Nothing on the robot subscribes to it.
        self.marker_pub = None
        if not self.survey:
            self.marker_pub = self.create_publisher(MarkerArray, '/coqui/emitter_markers', 10)
        elif self.publish_truth_markers:
            self.marker_pub = self.create_publisher(MarkerArray, '/sim/coqui_truth_markers', TRUTH_QOS)

        self.start_time = self.get_clock().now()
        self.call_counter = 0
        self.timer = self.create_timer(self.call_period, self._emit_cycle)

        # Survey truth layer: (index of the frog calling now or None, its call reaches the robot)
        self.truth_state = (None, False)
        self.truth_calls = 0    # calls of the calling frogs, drives the pulse of the truth ring
        if self.survey and self.marker_pub is not None:
            # 1 Hz: refreshes the ring pulse and reaches RViz displays subscribed as volatile
            self.truth_timer = self.create_timer(1.0, self._publish_truth)
            self._publish_truth()

        if self.survey:
            # do not print positions: the console is visible to whoever is testing the robot
            truth = 'drawn only on /sim/coqui_truth_markers (RViz)' if self.marker_pub is not None else 'not published'
            self.get_logger().info(
                f"Coqui Sound Emitter (survey mode) started: {len(self.frogs)} frogs calling in bouts, "
                f"audible to {self.max_range:.0f} m. True positions are {truth}.")
        else:
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
        tw = msg.twist.twist
        self.robot_speed = math.hypot(tw.linear.x, tw.linear.y)
        self.robot_wz = float(tw.angular.z)

    def _odom_fallback_cb(self, msg: Odometry):
        tw = msg.twist.twist
        if not self.have_gt_odom:
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            self.robot_pos_world = np.array([p.x, p.y, p.z], dtype=np.float64)
            self.robot_rot_world = quat_to_rot_matrix(q.x, q.y, q.z, q.w)
            self.robot_speed = math.hypot(tw.linear.x, tw.linear.y)
            self.robot_wz = float(tw.angular.z)

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

    def _survey_schedule(self, elapsed: float):
        """Frogs call in bouts (a run of calls from one frog, then a short silence, then a random
        frog), independent of the robot. Returns (frog index, frog is calling this cycle)."""
        if self.bout_left <= 0:
            if self.bout_frog is not None:
                self.bout_gap_until = elapsed + self.bout_gap
                self.bout_frog = None
            if elapsed < self.bout_gap_until:
                return 0, False
            choices = list(range(len(self.frogs)))
            if len(choices) > 1 and getattr(self, 'last_bout_frog', None) is not None:
                choices.remove(self.last_bout_frog)
            self.bout_frog = int(self.rng.choice(choices))
            self.last_bout_frog = self.bout_frog
            self.bout_left = int(self.rng.integers(self.bout_calls[0], self.bout_calls[1] + 1))
        self.bout_left -= 1
        return self.bout_frog, True

    def _emit_cycle(self):
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9

        # Compute 3D world positions of the 5 microphones
        array_origin_world = (
            self.robot_pos_world + self.robot_rot_world @ ARRAY_BASE_OFFSET_IN_FOOTPRINT
        )

        if self.survey:
            if elapsed < self.start_delay:
                caller_idx, in_bout = 0, False
            else:
                caller_idx, in_bout = self._survey_schedule(elapsed)
        else:
            caller_idx = self._select_active_calling_frog(array_origin_world)
            in_bout = True
        active = self.frogs[caller_idx]
        frog_pos = np.array(active['pos'], dtype=np.float64)
        center_dist = float(np.linalg.norm(array_origin_world - frog_pos))

        # Synthesize source waveform(s) if past start_delay, not in return-home pause, and in range
        is_calling = (
            in_bout
            and (elapsed >= self.start_delay)
            and (elapsed >= self.pause_until_sec)
            and (center_dist <= self.max_range)
        )
        num_mics = MIC_POSITIONS_BODY.shape[0]
        sources = []
        if is_calling:
            self.call_counter += 1
            pitch_scale = 0.97 + 0.012 * (caller_idx % 6)
            base_wave = synthesize_coqui_call_frame(
                FRAME_SAMPLES, SAMPLE_RATE, phase_seed=0.37 * self.call_counter, pitch_scale=pitch_scale
            )
            sources.append({
                'pos': frog_pos,
                'waveform': base_wave,
                'foliage_attenuation': float(active.get('foliage_attenuation', 1.0)),
            })
            # Overlapping chorus calls (survey mode): a second frog may call in the same window
            if self.survey and len(self.frogs) > 1 and self.overlap_prob > 0.0:
                if float(self.rng.random()) < self.overlap_prob:
                    others = [i for i in range(len(self.frogs)) if i != caller_idx]
                    idx2 = int(self.rng.choice(others))
                    f2 = self.frogs[idx2]
                    p2 = np.array(f2['pos'], dtype=np.float64)
                    if float(np.linalg.norm(array_origin_world - p2)) <= self.max_range:
                        wave2 = synthesize_coqui_call_frame(
                            FRAME_SAMPLES, SAMPLE_RATE,
                            phase_seed=0.53 * self.call_counter + idx2,
                            pitch_scale=0.97 + 0.012 * (idx2 % 6),
                            start_frac=float(self.rng.uniform(0.05, 0.25)),
                        )
                        sources.append({
                            'pos': p2,
                            'waveform': wave2,
                            'foliage_attenuation': float(f2.get('foliage_attenuation', 1.0)),
                        })

        clutter = self.survey and (float(self.rng.random()) < self.ambient_clutter_prob)
        multi_ch = propagate_sources_to_array(
            sources, array_origin_world, self.robot_rot_world,
            noise_std=self.noise_std,
            motor_speed=self.robot_speed,
            motor_wz=self.robot_wz,
            motor_noise_gain=self.motor_noise_gain if self.survey else 0.0,
            echo_gain=self.echo_gain if self.survey else 0.0,
            echo_delay_ms=self.echo_delay_ms,
            ambient_clutter=clutter,
            rng=self.rng,
        )

        # Publish 5-channel audio frame
        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label='channels', size=num_mics, stride=num_mics * FRAME_SAMPLES),
            MultiArrayDimension(label='samples', size=FRAME_SAMPLES, stride=FRAME_SAMPLES),
        ]
        msg.layout.data_offset = 0
        msg.data = multi_ch.astype(np.float32).ravel().tolist()
        self.audio_pub.publish(msg)

        if self.survey:
            if self.marker_pub is not None:
                self._update_truth(caller_idx if in_bout else None, is_calling)
            return

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

    def _update_truth(self, caller, audible: bool):
        """Survey mode, once per call cycle: `caller` = index of the frog calling (in its bout) or None,
        `audible` = its call reaches the robot (within max_audible_range_m). Re-publishes the truth scene
        when either changes; the 1 Hz timer re-publishes it in between."""
        if caller is not None:
            self.truth_calls += 1
        state = (caller, bool(audible))
        if state != self.truth_state:
            self.truth_state = state
            self._publish_truth()

    def _publish_truth(self):
        self.marker_pub.publish(self._truth_scene(*self.truth_state))

    def _truth_scene(self, caller, audible: bool = True) -> MarkerArray:
        """Survey mode, viewer-only simulation truth as one complete scene (DELETEALL first): every frog as a
        magenta sphere + label, and a pulsing ring around the frog calling now (`caller` index, None: nobody)."""
        now = self.get_clock().now().to_msg()
        clear = Marker()
        clear.header.frame_id = 'map'
        clear.header.stamp = now
        clear.action = Marker.DELETEALL
        ma = MarkerArray(markers=[clear])
        for i, fr in enumerate(self.frogs):
            x, y, z = fr['pos']    # call point, 5 cm above the frog model origin
            calling = i == caller
            ma.markers.append(make_marker(now, 'truth_frogs', i, Marker.SPHERE, (x, y, z), (0.22, 0.22, 0.22),
                                          TRUTH_RGB + (0.45,)))
            text = fr['name'] + ' (truth)'
            if calling:
                text += ' CALLING' if audible else ' CALLING (out of earshot)'
            if fr['habitat']:
                text += '\n' + fr['habitat']
            label = make_marker(now, 'truth_labels', i, Marker.TEXT_VIEW_FACING, (x, y, z + 0.45),
                                (0.22, 0.22, 0.22), (TRUTH_RGB + (1.0,)) if calling else (1.0, 1.0, 1.0, 0.9))
            label.text = text
            ma.markers.append(label)
        if caller is not None:
            d = 0.8 + 0.4 * (self.truth_calls % 4)    # diameter pulses 0.8 .. 2.0 m with the calls
            ma.markers.append(make_marker(now, 'truth_call', 0, Marker.CYLINDER, self.frogs[caller]['pos'],
                                          (d, d, 0.02), TRUTH_RGB + (0.25,)))
        return ma


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
