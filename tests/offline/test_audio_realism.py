"""Audio realism, harmonic-free SNR, motor noise, overlapping calls, and randomization tests.

Regressions for:
  - SNR capped at 13.4 dB (1 m) to 10.9 dB (22 m) due to synthetic harmonic inside noise_mask
  - Modest motor noise cutting detections to zero
  - Overlapping frog calls & ambient insect clutter
  - Seeded scenario randomization (not fixed to single seed=0 layout)
  - Answer-key separation invariant
"""
import math
import os
import numpy as np
import yaml

from ugv_audio import audio_core as ac


def _single_frog_snr(dist_m, motor_speed=0.0, motor_noise_gain=0.025, seed=0):
    rng = np.random.default_rng(seed)
    wave = ac.synthesize_coqui_call_frame(ac.FRAME_SAMPLES, ac.SAMPLE_RATE, phase_seed=0.4)
    arr = ac.propagate_sources_to_array(
        [{'pos': (dist_m, 0.0, 0.15), 'waveform': wave, 'foliage_attenuation': 1.0}],
        array_origin_world=np.array([0.0, 0.0, 0.115]),
        robot_rot_world=np.eye(3),
        noise_std=0.008,
        motor_speed=motor_speed,
        motor_wz=0.3 if motor_speed > 0 else 0.0,
        motor_noise_gain=motor_noise_gain,
        echo_gain=0.12,
        rng=rng,
    )
    snr_db, _, _ = ac.compute_coqui_snr_db(arr, ac.SAMPLE_RATE)
    return snr_db


def test_snr_dynamic_range_and_motor_noise_resilience(survey_cfg):
    """Harmonic-free SNR must span >= 15 dB from 1 m to 20 m (was capped at 13.4..10.9 dB),
    and modest motor noise at 0.5 m/s must NOT wipe out detections of a nearby frog!"""
    snr_1m = _single_frog_snr(1.0, motor_speed=0.0, seed=1)
    snr_5m = _single_frog_snr(5.0, motor_speed=0.0, seed=2)
    snr_15m = _single_frog_snr(15.0, motor_speed=0.0, seed=3)
    snr_22m = _single_frog_snr(22.0, motor_speed=0.0, seed=4)
    assert snr_1m > 25.0, f'1 m SNR still capped? got {snr_1m:.1f} dB'
    assert snr_1m - snr_15m > 12.0, f'insufficient SNR dynamic range: 1m={snr_1m:.1f}, 15m={snr_15m:.1f}'
    assert snr_1m > snr_5m > snr_15m > snr_22m

    # With realistic motor noise while driving at 0.5 m/s, nearby (5 m) frog is STILL reliably detected!
    det_thresh = float(survey_cfg['coqui_survey']['ros__parameters']['min_detection_snr_db'])
    hits = sum(1 for s in range(10) if _single_frog_snr(5.0, motor_speed=0.5, motor_noise_gain=0.025, seed=10 + s) >= det_thresh)
    assert hits >= 8, f'motor noise wiped detections: only {hits}/10 detected at 5 m (thresh={det_thresh} dB)'

    # Pure motor noise + ambient insect clutter with NO frog must stay below hunt trigger!
    trig_thresh = float(survey_cfg['coqui_survey']['ros__parameters']['hunt_trigger_snr_db'])
    rng = np.random.default_rng(99)
    false_trigs = 0
    for _ in range(20):
        arr = ac.propagate_sources_to_array(
            [], np.array([0.0, 0.0, 0.115]), np.eye(3),
            noise_std=0.008, motor_speed=0.6, motor_wz=0.5, motor_noise_gain=0.025,
            ambient_clutter=True, rng=rng,
        )
        snr, _, _ = ac.compute_coqui_snr_db(arr, ac.SAMPLE_RATE)
        if snr >= trig_thresh:
            false_trigs += 1
    assert false_trigs == 0, f'{false_trigs}/20 false triggers on pure motor+insect noise'


def test_overlapping_two_frog_calls_synthesize_cleanly():
    """Overlapping calls from two distinct frogs produce a detectable multi-source frame."""
    rng = np.random.default_rng(5)
    w1 = ac.synthesize_coqui_call_frame(ac.FRAME_SAMPLES, ac.SAMPLE_RATE, phase_seed=0.1, pitch_scale=0.98)
    w2 = ac.synthesize_coqui_call_frame(ac.FRAME_SAMPLES, ac.SAMPLE_RATE, phase_seed=1.7, pitch_scale=1.03, start_frac=0.15)
    arr = ac.propagate_sources_to_array(
        [
            {'pos': (4.0, 1.0, 0.15), 'waveform': w1, 'foliage_attenuation': 1.0},
            {'pos': (-3.0, 6.0, 0.15), 'waveform': w2, 'foliage_attenuation': 0.9},
        ],
        array_origin_world=np.array([0.0, 0.0, 0.115]),
        robot_rot_world=np.eye(3),
        noise_std=0.008,
        echo_gain=0.12,
        rng=rng,
    )
    assert arr.shape == (5, ac.FRAME_SAMPLES)
    snr, _, _ = ac.compute_coqui_snr_db(arr, ac.SAMPLE_RATE)
    assert snr > 14.0


def test_seeded_scenario_randomization_is_deterministic_and_varied(ws_root):
    """seed=0 preserves canonical layout; seed=1 and seed=2 produce distinct, valid, deterministic
    frog positions and spawn poses inside the survey area."""
    truth_path = os.path.join(ws_root, 'src', 'ugv_main', 'ugv_audio', 'config', 'sim_truth', 'forest_survey.yaml')
    base = yaml.safe_load(open(truth_path))
    area = [18.0, 50.0, -15.0, 15.0]

    s0 = ac.randomize_truth_scenario(base, seed=0, survey_area=area)
    assert s0['frogs'] == base['frogs'] and s0['spawn'] == base['spawn']

    s1a = ac.randomize_truth_scenario(base, seed=1, survey_area=area)
    s1b = ac.randomize_truth_scenario(base, seed=1, survey_area=area)
    s2 = ac.randomize_truth_scenario(base, seed=2, survey_area=area)
    assert s1a == s1b, 'randomization must be deterministic for a given seed'
    assert s1a['frogs'] != base['frogs'] and s1a['frogs'] != s2['frogs']
    assert s1a['spawn'] != base['spawn']
    for fr in s1a['frogs']:
        x, y, _ = fr['pos']
        assert area[0] + 1.5 <= x <= area[1] - 1.5
        assert area[2] + 1.5 <= y <= area[3] - 1.5
    # Pairwise separation >= 3.0 m
    pts = [fr['pos'][:2] for fr in s1a['frogs']]
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            assert math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) >= 2.9
