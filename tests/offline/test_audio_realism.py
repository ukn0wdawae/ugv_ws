"""Audio realism, harmonic-free SNR, motor noise, overlapping calls, and randomization tests.

Regressions for:
  - SNR capped at 13.4 dB (1 m) to 10.9 dB (22 m) due to synthetic harmonic inside noise_mask
  - Motor noise cutting detections to zero, or costing almost nothing (2.5 dB)
  - Loudness -> range, coherence of overlapping calls, chorus schedule, level scatter
  - Overlapping frog calls & ambient insect clutter
  - Seeded scenario randomization (not fixed to single seed=0 layout)
  - Answer-key separation invariant
"""
import math
import os
import numpy as np
import yaml

from ugv_audio import audio_core as ac
from ugv_coqui_behavior import survey_planning as sp


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

    # With the motor noise of the simulation (0.06) at 0.5 m/s, a nearby (5 m) frog is STILL reliably detected,
    # and the motors cost a realistic amount of SNR (the old 0.025 cost 2.5 dB: too easy)
    det_thresh = float(survey_cfg['coqui_survey']['ros__parameters']['min_detection_snr_db'])
    moving = [_single_frog_snr(5.0, motor_speed=0.5, motor_noise_gain=0.06, seed=10 + s) for s in range(10)]
    assert sum(v >= det_thresh for v in moving) >= 8, f'motor noise wiped detections at 5 m: {moving}'
    assert 5.0 < snr_5m - float(np.mean(moving)) < 14.0


def test_noise_alone_never_triggers_a_hunt(survey_cfg):
    """Pure motor noise + ambient insect clutter with NO frog must never look like a near call."""
    cfg = survey_cfg['coqui_survey']['ros__parameters']
    rng = np.random.default_rng(99)
    for _ in range(40):
        arr = ac.propagate_sources_to_array(
            [], np.array([0.0, 0.0, 0.115]), np.eye(3),
            noise_std=0.008, motor_speed=0.6, motor_wz=0.5, motor_noise_gain=0.06,
            ambient_clutter=True, rng=rng,
        )
        snr, specs, power = ac.compute_coqui_snr_db(arr, ac.SAMPLE_RATE)
        level = ac.call_band_level_db(specs, power)
        assert not (snr >= cfg['min_detection_snr_db'] and sp.call_is_near(level, cfg)), (snr, level)


def _call(dist, az=0.7, seed=0, others=(), **kw):
    rng = np.random.default_rng(seed)
    src = [{'pos': (d * math.cos(a), d * math.sin(a), 0.15), 'foliage_attenuation': 1.0,
            'waveform': ac.synthesize_coqui_call_frame(ac.FRAME_SAMPLES, ac.SAMPLE_RATE, phase_seed=rng.uniform(0, 6),
                                                       start_frac=f)}
           for d, a, f in [(dist, az, 0.0), *others]]
    arr = ac.propagate_sources_to_array(src, np.array([0.0, 0.0, 0.115]), np.eye(3), noise_std=0.008, echo_gain=0.12,
                                        rng=rng, **kw)
    snr, specs, power = ac.compute_coqui_snr_db(arr, ac.SAMPLE_RATE)
    az_est, _, _, coh = ac.estimate_doa(specs[:, ac.COQUI_MASK])
    return ac.call_band_level_db(specs, power), az_est, coh, snr


def test_call_level_says_how_far_the_frog_is():
    """0 dB = a nominal call at 1 m, spherical spreading, also while the motors run (they must not add loudness)."""
    for d in (1.0, 3.0, 6.0, 10.0, 18.0):
        level = np.mean([_call(d, seed=s)[0] for s in range(6)])
        assert abs(level + 20 * math.log10(d)) < 0.7, (d, level)
        assert abs(ac.level_to_range_m(level) - d) < 0.1 * d
    moving = np.mean([_call(6.0, seed=s, motor_speed=0.5, motor_wz=0.3, motor_noise_gain=0.06)[0] for s in range(10)])
    assert abs(moving + 20 * math.log10(6.0)) < 1.5


def test_coherence_tells_one_frog_from_two_at_once(survey_cfg):
    """Two frogs calling in the same instant blend into a bearing that may point at neither: the robot must notice."""
    gate = float(survey_cfg['coqui_survey']['ros__parameters']['min_coherence'])
    one = [_call(d, az=a, seed=k)[2] for k, (d, a) in enumerate([(1, 0), (4, 1), (8, 2), (15, -2), (22, -1)])]
    assert min(one) > gate + 0.05, one
    rng = np.random.default_rng(3)
    bad = []
    for k in range(40):
        az2 = 0.7 + rng.uniform(0.8, 2 * math.pi - 0.8)
        _, az, coh, _ = _call(6.0, seed=k, others=[(6.0, az2, rng.uniform(0.05, 0.25))])
        err = min(abs(math.degrees(math.remainder(az - a, 2 * math.pi))) for a in (0.7, az2))
        if coh >= gate and err > 6.0:
            bad.append((coh, err))
    assert len(bad) <= 1, bad    # an accepted blend must at least point at one of the two frogs


def test_chorus_is_a_chorus_and_turns_are_turns():
    """Every frog keeps calling on its own rhythm (no 75 s waits for a frog's turn), neighbours mostly avoid overlap."""
    ch = ac.ChorusSchedule(6, np.random.default_rng(1), start_sec=0.0)
    calls = [(0.25 * k, i) for k in range(2400) for i, _ in ch.step(0.25 * k)]      # 10 minutes
    for f in range(6):
        t = [a for a, i in calls if i == f]
        assert 80 < len(t) < 300, (f, len(t))
        assert 1.5 < np.median(np.diff(t)) < 4.5 and max(np.diff(t)) < 45.0
    frames = {}
    for t, i in calls:
        frames.setdefault(t, []).append(i)
    overlap = np.mean([len(v) > 1 for v in frames.values()])
    assert 0.01 < overlap < 0.35, overlap
    again = ac.ChorusSchedule(6, np.random.default_rng(1), start_sec=0.0)
    assert [(0.25 * k, i) for k in range(2400) for i, _ in again.step(0.25 * k)] == calls, 'same seed, same chorus'


def test_call_levels_scatter_but_not_wildly(survey_cfg):
    """What the loudness gates rely on: calls of one frog, corrected for distance, scatter less than
    loudness_spread_db between places, and no frog is further than loudness_gate_db from nominal."""
    cfg = survey_cfg['coqui_survey']['ros__parameters']
    for seed in range(8):
        var = ac.CallVariation(np.random.default_rng([seed, 2]), 6, [1.0, 0.9, 0.78, 0.85, 0.94, 1.0])
        for f in range(6):
            lv = [20 * math.log10(var.gain(f, (0, 0), (r * math.cos(a), r * math.sin(a))))
                  for r in (2, 5, 9) for a in np.linspace(0, 2 * math.pi, 8) for _ in range(3)]
            assert abs(np.mean(lv)) < float(cfg['loudness_gate_db']) - 2.0
            assert np.std(lv) < 0.5 * float(cfg['loudness_spread_db'])


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


def test_forest_and_hill_randomization_respects_3d_terrain_bushes_and_trees(ws_root):
    """Every randomized forest run must place frogs in genuinely different spots across survey_area,
    including inside real bushes and on top of / perched on real trees, with exact 3D terrain Z."""
    forest_truth = yaml.safe_load(open(os.path.join(
        ws_root, 'src', 'ugv_main', 'ugv_audio', 'config', 'sim_truth', 'forest_survey.yaml')))
    hill_truth = yaml.safe_load(open(os.path.join(
        ws_root, 'src', 'ugv_main', 'ugv_audio', 'config', 'sim_truth', 'hill_survey.yaml')))
    f_area = [18.0, 50.0, -15.0, 15.0]
    h_area = [-39.0, 48.0, -16.5, 16.5]

    scenes = ac.load_world_scenes()
    assert 'forest_grid_z' in scenes and 'forest_bushes' in scenes and 'forest_tree_tops' in scenes
    bush_xy = scenes['forest_bushes'][:, :2]
    top_xyz = scenes['forest_tree_tops'][:, :3]
    perch_xyz = scenes['forest_tree_perches'][:, :3]

    layouts = []
    for seed in (11, 22, 33, 44, 55):
        doc = ac.randomize_truth_scenario(forest_truth, seed=seed, survey_area=f_area, spawn_jitter_m=0.0)
        # spawn_jitter_m=0.0 keeps robot spawn aligned with Terminal 2 static EKF initial_pose
        assert doc['spawn'] == forest_truth['spawn']
        layouts.append([tuple(fr['pos']) for fr in doc['frogs']])

        n_bush, n_top, n_elev = 0, 0, 0
        for fr in doc['frogs']:
            x, y, z = fr['pos']
            gz = ac.ground_elevation_z('forest', x, y)
            hab = fr['habitat']
            if 'on top of' in hab:
                n_top += 1
                n_elev += 1
                assert z - gz >= 1.0, f'tree-top frog must be elevated >= 1 m above ground: {fr}'
                near_top = np.min(np.linalg.norm(top_xyz - np.array([x, y, z]), axis=1))
                assert near_top <= 0.05, f'tree-top frog not on real tree crown/trunk surface: {fr}'
            elif 'perched' in hab:
                n_elev += 1
                assert z - gz >= 1.1, f'perched tree frog must be elevated >= 1.1 m: {fr}'
                near_perch = np.min(np.linalg.norm(perch_xyz - np.array([x, y, z]), axis=1))
                assert near_perch <= 0.05, f'perched frog not on real pine trunk bark: {fr}'
            else:
                assert abs(z - gz) <= 0.02, f'ground/bush/grass frog floating or buried (z={z}, gz={gz:.3f}): {fr}'
                if 'bush' in hab:
                    n_bush += 1
                    near_bush = np.min(np.hypot(bush_xy[:, 0] - x, bush_xy[:, 1] - y))
                    assert near_bush <= 0.35, f'bush frog not inside a real bush_pine footprint: {fr}'

        assert n_bush >= 2, f'seed {seed}: expected >= 2 bush frogs, got {n_bush}'
        assert n_top >= 1, f'seed {seed}: expected >= 1 on-top-of-tree frog, got {n_top}'
        assert n_elev >= 2, f'seed {seed}: expected >= 2 elevated tree frogs, got {n_elev}'

    # Every consecutive run places frogs at distinct locations
    for i in range(len(layouts) - 1):
        assert layouts[i] != layouts[i + 1]

    # Hill world 11 m elevation span also snaps ground/bush/grass frogs onto hill terrain
    hdoc = ac.randomize_truth_scenario(hill_truth, seed=17, survey_area=h_area, spawn_jitter_m=0.0)
    assert hdoc['spawn'] == hill_truth['spawn']
    for fr in hdoc['frogs']:
        x, y, z = fr['pos']
        gz = ac.ground_elevation_z('hill', x, y)
        if 'perched' in fr['habitat'] or 'on top of' in fr['habitat']:
            assert z - gz >= 1.0
        else:
            assert abs(z - gz) <= 0.05
