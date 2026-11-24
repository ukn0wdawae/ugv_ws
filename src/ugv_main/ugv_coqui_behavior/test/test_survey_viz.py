"""Offline tests for what the Coqui survey shows in RViz (survey_viz: HUD text, bearing rays, stations; no ROS).

    cd src/ugv_main/ugv_coqui_behavior && python3 -m pytest test/test_survey_viz.py -q
"""
import ast
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ugv_coqui_behavior import survey_planning as sp  # noqa: E402
from ugv_coqui_behavior import survey_viz as sv  # noqa: E402

# every state coqui_survey_node.set_state() uses, plus the initial one
STATES = ['INIT', 'WAIT_FOR_NAV2', 'START', 'LISTEN', 'EXPLORE', 'RECOVER', 'UNSTICK', 'HUNT', 'HUNT_END',
          'VANTAGE', 'PHOTO', 'RECORDED', 'HOME', 'DONE']


def check_hud(lines):
    assert isinstance(lines, list) and 1 <= len(lines) <= 3
    for line in lines:
        assert isinstance(line, str) and line
        assert len(line) <= sv.HUD_LINE_CHARS, line
        assert line.isascii(), line
        assert not any(c in line for c in '\n\r\t'), line


def test_survey_viz_has_no_ros_imports():
    """Pure python + numpy, so it runs (and is tested) without ROS."""
    tree = ast.parse(open(sv.__file__, encoding='utf-8').read())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split('.')[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or '').split('.')[0])
    assert mods <= {'math', 'numpy'}, mods


# ------------------------------------------------------------------------------------------ clock
@pytest.mark.parametrize('seconds,text', [
    (0, '00:00'), (59.9, '00:59'), (61, '01:01'), (331.2, '05:31'), (3599, '59:59'),
    (3600, '1:00:00'), (3725.4, '1:02:05'), (36000, '10:00:00'),
    (None, '00:00'), (-5, '00:00'), (float('nan'), '00:00'), (float('inf'), '00:00'), ('abc', '00:00'),
    (np.float64(75.0), '01:15'),
])
def test_mmss(seconds, text):
    assert sv.mmss(seconds) == text


# -------------------------------------------------------------------------------------------- HUD
def test_hud_examples():
    explore = sv.hud_text('EXPLORE', {'goal': [31.0, -4.0], 'path_m': 12.34, 'uncovered': 9}, 2, 331.2, 7, 23)
    assert explore == ['EXPLORE -> station (31.0, -4.0), path 12.3 m', 'frogs recorded: 2   stations 7/23   05:31']
    frontier = sv.hud_text('EXPLORE', {'mode': 'frontier', 'goal': [31.0, -4.0], 'path_m': 5.2}, 2, 331.2, 7, 23)
    assert frontier[0] == 'EXPLORE -> frontier (31.0, -4.0), path 5.2 m'
    hunt = sv.hud_text('HUNT', {'step': 3, 'obs': 14, 'est': [31.24, -4.0, 3.1], 'inliers': 9, 'places': 3,
                                'parallax_deg': 12.4, 'unc_m': 0.8}, 1, 100, 3, 23)
    assert hunt[:2] == ['HUNT step 3: 9 bearings from 3 places, parallax 12 deg',
                        'estimate +/-0.80 m at (31.2, -4.0, 3.1)']
    first = sv.hud_text('HUNT', {'bearing_deg': 135, 'heard_at': [40.1, 2.0]}, 0, 50, 1, 23)
    assert first[0] == 'HUNT new call: bearing 135 deg at (40.1, 2.0)'
    no_fix = sv.hud_text('HUNT', {'step': 1, 'obs': 4, 'est': [30.0, 1.0, 2.9]}, 0, 60, 1, 23)
    assert no_fix[0] == 'HUNT step 1: 4 bearings, no cross-bearing fix yet'
    assert sv.hud_text('LISTEN', {'at': [40.12, 2.0], 'listen_s': 8.0}, 0, 0, 0, 23)[0] == 'LISTEN 8 s at (40.1, 2.0)'
    assert sv.hud_text('PHOTO', {'vantage': 'reached'}, 1, 0, 0, 23)[:2] == [
        'PHOTO coqui_002: aiming the pan-tilt camera', 'vantage: reached']
    assert sv.hud_text('RECORDED', {'id': 'coqui_002', 'pos': [31.2, -4.0, 3.1], 'visual': True}, 2, 0, 0, 23)[:2] \
        == ['RECORDED coqui_002 (photo)', 'at (31.2, -4.0, 3.1)']
    assert sv.hud_text('RECORDED', {'id': 'coqui_002', 'visual': False}, 2, 0, 0, 23)[0] \
        == 'RECORDED coqui_002 (no photo)'
    assert sv.hud_text('DONE', {'result': 'area covered'}, 6, 3700, 23, 23) == [
        'DONE: area covered', 'frogs recorded: 6   stations 23/23   1:01:40']


def test_hud_accepts_info_as_set_state_publishes_it():
    """set_state() adds state, t and frogs_found to the info; they must not disturb the HUD."""
    info = {'goal': [31.0, -4.0], 'path_m': 12.3, 'uncovered': 9, 'state': 'EXPLORE', 't': 1234.5, 'frogs_found': 2}
    assert sv.hud_text('EXPLORE', info, 2, 331, 7, 23)[0] == 'EXPLORE -> station (31.0, -4.0), path 12.3 m'


@pytest.mark.parametrize('state', STATES + ['SOMETHING_NEW', '', None])
def test_hud_missing_info_keys(state):
    for info in ({}, None, {'state': state, 't': 12.0, 'frogs_found': 0}):
        lines = sv.hud_text(state, info, 0, 0.0, 0, 0)
        check_hud(lines)
        assert lines[-1] == 'frogs recorded: 0   00:00'      # no station count before the stations exist


def test_hud_malformed_values():
    bad = {'goal': 'abc', 'path_m': None, 'at': [1.0], 'home': 5, 'est': [1.0, float('nan'), 2.0], 'step': 'x',
           'inliers': 'many', 'places': [], 'parallax_deg': float('inf'), 'unc_m': 'small', 'heard_at': {},
           'bearing_deg': object(), 'attempt': None, 'line_of_sight': None, 'id': None, 'pos': None,
           'listen_s': None, 'reason': None, 'outcome': None, 'result': None}
    for state in STATES:
        check_hud(sv.hud_text(state, bad, None, None, None, None))
    assert sv.hud_text('EXPLORE', bad, 0, 0, 0, 0)[0] == 'EXPLORE -> station'
    assert sv.hud_text('LISTEN', bad, 0, 0, 0, 0)[0] == 'LISTEN'
    assert sv.hud_text('DONE', bad, 0, 0, 0, 0)[0] == 'DONE'


def test_hud_is_plain_ascii_single_lines_and_cut_to_length():
    lines = sv.hud_text('DONE', {'result': 'área cubierta ✓\nsecond line'}, 1, 10, 1, 2)
    check_hud(lines)
    assert lines[0] == 'DONE: ?rea cubierta ? second line'
    cut = sv.hud_text('HUNT_END', {'outcome': 'x' * 300}, 0, 0, 0, 0)
    check_hud(cut)
    assert cut[0].endswith('...') and len(cut[0]) == sv.HUD_LINE_CHARS


def test_hud_lines_stay_short_with_large_numbers():
    big = {'step': 123, 'obs': 4567, 'est': [-1234.5, -1234.5, -123.4], 'inliers': 4567, 'places': 99,
           'parallax_deg': 179.9, 'unc_m': 123.456, 'goal': [-1234.5, -1234.5], 'path_m': 12345.6,
           'at': [-1234.5, -1234.5], 'home': [-1234.5, -1234.5], 'heard_at': [-1234.5, -1234.5],
           'bearing_deg': -179.9, 'stations_to_cover': 9999, 'attempt': 4, 'line_of_sight': False,
           'candidates': 6, 'fails': 12, 'id': 'coqui_123', 'pos': [-1234.5, -1234.5, -123.4], 'visual': True,
           'vantage': 'reached (no clear line of sight on costmap)', 'listen_s': 8.0,
           'outcome': 'converged (robot stuck)', 'result': 'no reachable stations left'}
    for state in STATES:
        check_hud(sv.hud_text(state, big, 999, 10 ** 6, 999, 999))


# ----------------------------------------------------------------------------------- bearing rays
def obs_from(origin, dirs):
    """Observations shaped like the node's self.obs: (t, origin np xyz, unit dir np xyz, snr)."""
    return [(float(i), np.array(origin, float), np.array(d, float) / np.linalg.norm(d), 20.0)
            for i, d in enumerate(dirs)]


def plain_floats(segs):
    return all(type(v) is float for s in segs for v in s[0] + s[1])


def test_rays_empty():
    assert sv.ray_segments([], None, None) == []
    assert sv.ray_segments([], np.zeros(0, bool), [1.0, 2.0, 3.0]) == []


def test_rays_before_the_first_fix_are_pending_with_default_length():
    obs = obs_from((1.0, 2.0, 3.0), [(1, 0, 0), (0, 2, 0)])
    segs = sv.ray_segments(obs, None, None)
    assert [s[2] for s in segs] == ['pending', 'pending']
    assert segs[0][0] == (1.0, 2.0, 3.0)
    assert segs[0][1] == pytest.approx((9.0, 2.0, 3.0))
    assert segs[1][1] == pytest.approx((1.0, 10.0, 3.0))
    assert plain_floats(segs)
    # mask None stays pending even when an estimate is known; length then follows the estimate
    segs = sv.ray_segments(obs, None, (5.0, 2.0, 3.0))
    assert [s[2] for s in segs] == ['pending', 'pending']
    assert segs[0][1] == pytest.approx((1.0 + 4.0 + 1.5, 2.0, 3.0))


def test_rays_are_classified_by_the_mask_and_reach_just_past_the_estimate():
    est = (5.0, 0.0, 3.0)
    obs = obs_from((0.0, 0.0, 3.0), [(1, 0, 0)]) + obs_from((0.0, 4.0, 3.0), [(5, -4, 0)]) \
        + obs_from((0.0, 4.0, 3.0), [(1, 1, 0)])
    segs = sv.ray_segments(obs, np.array([True, True, False]), est)
    assert [s[2] for s in segs] == ['inlier', 'inlier', 'outlier']
    assert segs[0][1] == pytest.approx((6.5, 0.0, 3.0))
    for s in segs:
        length = math.dist(s[0], s[1])
        assert length == pytest.approx(math.dist(s[0], est) + 1.5)
    # the agreeing rays pass through the estimate
    d = np.subtract(segs[1][1], segs[1][0]) / math.dist(segs[1][0], segs[1][1])
    assert np.cross(d, np.subtract(est, segs[1][0])) == pytest.approx(np.zeros(3), abs=1e-9)
    assert plain_floats(segs)
    # a plain list works as a mask too
    assert [s[2] for s in sv.ray_segments(obs, [False, True, True], est)] == ['outlier', 'inlier', 'inlier']


def test_rays_are_capped_and_a_short_mask_leaves_the_rest_pending():
    obs = obs_from((0.0, 0.0, 0.0), [(1, 0, 0), (0, 1, 0), (0, -1, 0)])
    segs = sv.ray_segments(obs, [True], (100.0, 0.0, 0.0), max_len=25.0)
    assert [s[2] for s in segs] == ['inlier', 'pending', 'pending']
    assert all(math.dist(s[0], s[1]) == pytest.approx(25.0) for s in segs)


def test_rays_skip_bad_observations_and_ignore_a_bad_estimate():
    good = obs_from((0.0, 0.0, 0.0), [(0, 0, 1)])[0]
    obs = [good,
           (1.0, np.zeros(3), np.zeros(3), 20.0),                                  # zero direction
           (2.0, np.array([np.nan, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 20.0),   # bad origin
           (3.0, np.array([0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 20.0),            # 2D origin
           (4.0, np.zeros(3), np.array([np.inf, 0.0, 0.0]), 20.0)]                 # bad direction
    segs = sv.ray_segments(obs, None, None)
    assert len(segs) == 1 and segs[0][1] == pytest.approx((0.0, 0.0, 8.0))
    for bad_est in ([1.0, 2.0], [np.nan, 0.0, 0.0], [[1.0, 2.0, 3.0, 4.0]]):
        segs = sv.ray_segments([good], [True], bad_est)
        assert segs[0][1] == pytest.approx((0.0, 0.0, 8.0)) and segs[0][2] == 'inlier'


# --------------------------------------------------------------------------------------- stations
def test_station_status():
    cells = [(21.0, -12.0), (27.0, -12.0), (33.0, -12.0), (39.0, -12.0)]
    listened = [(21.5, -11.0)]
    failed = [(27.5, -13.0), (21.0, -12.5)]          # the second lies on a covered cell: covered wins
    assert sv.station_status(cells, listened, 2.0, failed, 3.0) == [
        (21.0, -12.0, 'covered'), (27.0, -12.0, 'unreachable'), (33.0, -12.0, 'open'), (39.0, -12.0, 'open')]


def test_station_status_edge_cases():
    assert sv.station_status([], [(0.0, 0.0)], 7.0, [(0.0, 0.0)], 1.0) == []
    assert sv.station_status([(1, 2)], [], 7.0, [], 1.0) == [(1.0, 2.0, 'open')]
    assert sv.station_status([(0.0, 0.0)], [(7.0, 0.0)], 7.0, [], 1.0) == [(0.0, 0.0, 'open')]   # strictly inside
    st = sv.station_status([(np.float64(3.0), np.float64(4.0))], [], 7.0, [], 1.0)
    assert st == [(3.0, 4.0, 'open')] and type(st[0][0]) is float


def test_station_status_agrees_with_the_planner():
    """'open' stations are exactly those survey_planning.station_candidates still offers (no costmap, no
    failures), so the RViz view shows what the mission will do."""
    cells = sp.coverage_cells([18.0, 50.0, -15.0, 15.0], 6.0)
    listened = [(40.0, 2.0), (30.0, -5.0), (22.0, 10.0), (47.9, 2.1)]
    offered = {c for _, c in sp.station_candidates((48.0, 2.1), cells, listened, 7.0, [], None,
                                                   {'failed_radius_m': 1.0})}
    status = sv.station_status(cells, listened, 7.0, [], 1.0)
    assert {(x, y) for x, y, s in status if s == 'open'} == offered
    assert {s for _, _, s in status} == {'open', 'covered'}


# ------------------------------------------------------------------------------- paths on the terrain
def test_terrain_heights_keep_the_newest_surface():
    from ugv_coqui_behavior.survey_viz import TerrainHeights
    t = TerrainHeights(cell=0.2)
    t.add([[1.05, 2.05, 2.0], [1.06, 2.06, 2.1], [float('nan'), 0, 0]])
    assert t.at(1.1, 2.1) == 2.1                 # the highest point of a cell
    t.add([[1.1, 2.1, 1.7]])
    assert t.at(1.1, 2.1) == 1.7                 # re-measured
    assert t.at(1.25, 2.1) == 1.7                # neighbour
    assert t.at(5.0, 5.0) is None and t.at(5.0, 5.0, default=3.0) == 3.0


def test_paths_follow_the_terrain_not_the_robot_plane():
    """Regression (2026-11-24): paths and arrows were drawn on a flat plane and vanished under the terrain."""
    from ugv_coqui_behavior.survey_viz import TerrainHeights, lay_path
    t = TerrainHeights()
    xs = [0.1 * k for k in range(40)]
    t.add([[x, 0.0, 0.3 * x] for x in xs[:30]])   # a ramp; the last metre was not seen
    z = lay_path([(x, 0.0) for x in xs], t, fallback_z=0.0, lift=0.05)
    assert all(zz >= 0.3 * x for zz, x in zip(z[:29], xs[:29])), 'path under the ground'
    assert all(zz <= 0.3 * x + 0.15 for zz, x in zip(z[:29], xs[:29]))
    assert z[31] > 0.8                            # a short unseen gap keeps the last height
    assert abs(lay_path([(50.0, 50.0)], t, fallback_z=2.6, lift=0.05)[0] - 2.65) < 1e-9
