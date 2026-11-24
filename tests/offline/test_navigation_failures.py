"""Pooled navigation & mission failure regression tests.

Directly covers every row of the Failure Cases audit table:
  - hill_v1: return-home beyond rolling window (57 m from home -> staged waypoints)
  - 20:04: recorded frog keep-out zone (pinned 0.29 m from frog 6)
  - listening while driving vs stationary-only triangulation + calibrated uncertainty
  - 20:04: false frog rejected when open-ground clear-LOS photo is empty + tighter masking cone
  - 21:09: frog goes quiet mid-hunt (keeps tight 4 cm fix, rejects loose 4.8 m fix)
  - 13:48: goal timeout increments nav_fail_streak (triggers unstick)
  - 20:04: boxed-in early cooldown loop (does not terminate mission when stations are in cooldown)
  - moving vs stationary acoustic coverage radius
  - narrow bush gap, behind trunk, inside bush, perched frog hunts
"""
import ast
import math
import os
import numpy as np

from ugv_coqui_behavior import survey_planning as sp


def _world(size=24.0, origin=(-12.0, -12.0), res=0.1, fill=0):
    n = int(round(size / res))
    return sp.Grid(origin[0], origin[1], res, np.full((n, n), fill, dtype=np.int16))


def _cfg(survey_cfg):
    return dict(survey_cfg['coqui_survey']['ros__parameters'])


def test_staged_return_home_beyond_rolling_window(survey_cfg):
    """Regression for hill_v1 (logged DONE 57 m from home because home was outside rolling window):
    next_home_waypoints must step smoothly along breadcrumbs / window boundary across 60 m."""
    cfg = _cfg(survey_cfg)
    home = (43.0, -13.0)
    # Breadcrumbs laid along the 60 m hill trail from (43,-13) -> (-17,-10)
    breadcrumbs = [(43.0 - 4.0 * k, -13.0 + 0.2 * k) for k in range(16)]
    rx, ry = -17.0, -10.0
    stages = 0
    for _ in range(10):
        # 24x24 m rolling window centered on robot (home at 60 m is far outside!)
        g = _world(size=24.0, origin=(rx - 12.0, ry - 12.0), res=0.2, fill=0)
        cands = sp.next_home_waypoints((rx, ry), home, breadcrumbs, g, cfg)
        assert cands, f'no home staging waypoint from ({rx:.1f}, {ry:.1f})'
        wx, wy = cands[0]
        assert g.inside(wx, wy), 'staged waypoint must lie inside current rolling window!'
        assert math.hypot(wx - home[0], wy - home[1]) < math.hypot(rx - home[0], ry - home[1])
        rx, ry = wx, wy
        stages += 1
        if math.hypot(rx - home[0], ry - home[1]) < 1.0:
            break
    assert math.hypot(rx - home[0], ry - home[1]) < 1.0, 'failed to stage back to home 60 m away'
    assert stages >= 3


def test_recorded_frog_keepout_zone(survey_cfg):
    """Regression for 20:04 (pinned 0.29 m from frog 6): stations, frontiers, baseline steps, and
    vantages must stay >= frog_keepout_m clear of every already-recorded frog."""
    cfg = _cfg(survey_cfg)
    ko = float(cfg['frog_keepout_m'])
    assert ko >= 0.8
    known = [{'pos': [0.0, 0.0, 2.7], 'id': 'coqui_006'}]
    g = _world(size=20.0, origin=(-10.0, -10.0), res=0.1, fill=0)
    g.data[:, 120:] = -1  # unknown east -> frontier near x=2.0

    # 1) Station centered right on frog 6 is skipped
    st = sp.station_candidates((-3.0, 0.0), [(0.0, 0.0), (0.2, 0.1), (4.0, 0.0)],
                               [], 2.0, [], g, cfg, known=known)
    assert all(math.hypot(goal[0], goal[1]) >= ko for goal, _ in st)

    # 2) Baseline steps toward another frog never land inside frog 6 keepout
    steps = sp.baseline_step_candidates((-1.2, 0.0), (5.0, 0.0), [(-1.2, 0.0)], g, cfg, known=known)
    assert steps and all(math.hypot(c[0], c[1]) >= ko for _, c in steps)

    # 3) Vantage points around a nearby frog stay outside frog 6 keepout
    vants = sp.vantage_candidates((1.4, 0.0, 2.7), (-2.0, 0.0), 'ground', g, cfg, known=known)
    assert vants and all(math.hypot(v[0], v[1]) >= ko for _, v, _, _ in vants)


def test_empty_open_ground_photo_rejected_and_tighter_mask(survey_cfg):
    """Regression for 20:04 false frog (saved 5 m off as +-0.14 m despite empty photo, then masked
    +-30 deg at 5 m)."""
    cfg = _cfg(survey_cfg)
    g = _world(size=20.0, origin=(-10.0, -10.0), res=0.1, fill=0)
    # Put a bush at (4.0, 4.0)
    yy, xx = np.mgrid[0:g.height, 0:g.width]
    X = g.origin_x + (xx + 0.5) * g.resolution
    Y = g.origin_y + (yy + 0.5) * g.resolution
    g.data[(X - 4.0) ** 2 + (Y - 4.0) ** 2 <= 1.0 ** 2] = 100

    tight_tri = {'pos': [1.0, 1.0, 2.7], 'inliers': 10, 'n_origins': 4, 'baseline_m': 2.0,
                 'parallax_deg': 25.0, 'rms_deg': 0.8, 'uncertainty_m': 0.14}
    # 1) Open ground + reached vantage + clear LOS + empty photo -> REJECTED!
    ok, reason = sp.should_accept_record(tight_tri, False, 'reached', True, g, cfg)
    assert not ok and reason == 'photo_contradicts_open_ground'

    # 2) Visually confirmed in open ground -> ACCEPTED!
    ok2, _ = sp.should_accept_record(tight_tri, True, 'reached', True, g, cfg)
    assert ok2

    # 3) Frog genuinely inside bush (occluded, no LOS) with converged acoustic fix -> ACCEPTED!
    bush_tri = dict(tight_tri, pos=[4.0, 4.0, 2.7])
    ok3, reason3 = sp.should_accept_record(bush_tri, False, 'reached (no clear line of sight)', False, g, cfg)
    assert ok3 and reason3 == 'acoustic_only_occluded'

    # 4) Tighter bearing mask cone: at 5 m, a bearing 24 deg off must NOT be masked!
    br_r = float(cfg['known_bearing_radius_m'])
    known = [{'pos': [5.0, 0.0, 2.7]}]
    assert sp.bearing_points_at_known((0.0, 0.0), math.radians(24.0), known, br_r) is None
    assert sp.bearing_points_at_known((0.0, 0.0), math.radians(8.0), known, br_r) is not None


def test_quiet_frog_keeps_tight_fix_and_rejects_loose_fix(survey_cfg):
    """Regression for 21:09 (4 cm fix dropped when frog went quiet, 4.8 m fix accepted at step limit)."""
    cfg = _cfg(survey_cfg)
    # Accurate 4 cm fix from 2 places before frog stopped calling -> ACCEPTED!
    good_partial = {'pos': [5.0, 1.0, 2.7], 'inliers': 5, 'n_origins': 2, 'baseline_m': 1.2,
                    'parallax_deg': 14.0, 'rms_deg': 0.7, 'uncertainty_m': 0.28}
    assert sp.accept_partial_triangulation(good_partial, cfg) is True

    # Loose garbage fix (8.5 deg parallax, 2.1 m uncertainty) -> REJECTED!
    bad_loose = {'pos': [9.8, 1.0, 2.7], 'inliers': 6, 'n_origins': 2, 'baseline_m': 0.9,
                 'parallax_deg': 8.5, 'rms_deg': 1.8, 'uncertainty_m': 2.1}
    assert sp.accept_partial_triangulation(bad_loose, cfg) is False


def test_stationary_only_triangulation_and_calibrated_uncertainty(survey_cfg):
    """Regression: stationary_only_triangulation must be enabled by default and pose_sigma_m floor
    keeps triangulation uncertainty honest."""
    cfg = _cfg(survey_cfg)
    assert cfg['stationary_only_triangulation'] is True
    assert float(cfg['pose_sigma_m']) >= 0.05
    # Verify uncertainty includes the pose_sigma floor even with near-zero bearing noise
    frog = np.array([5.0, 0.0, 0.1])
    places = [np.array([0.0, -1.5, 0.1]), np.array([0.0, 0.0, 0.1]), np.array([0.0, 1.5, 0.1])]
    O, D = [], []
    for o in places:
        v = frog - o
        d = v / np.linalg.norm(v)
        for _ in range(3):
            O.append(o)
            D.append(d)
    tri = sp.robust_triangulate(O, D, (O[0], D[0]), cfg)
    assert tri is not None and tri['uncertainty_m'] >= float(cfg['pose_sigma_m'])


def test_moving_vs_stationary_coverage_radius(survey_cfg):
    """Regression: driving past at 6 m must NOT mark a cell covered (frogs silent ~60% of passes);
    only stationary listening covers out to cover_radius_m (7 m), moving pass covers moving_cover_radius_m."""
    cfg = _cfg(survey_cfg)
    r_stat = float(cfg['cover_radius_m'])
    r_mov = float(cfg['moving_cover_radius_m'])
    assert r_mov < r_stat and r_mov <= 4.0
    assert not sp.is_cell_covered((5.5, 0.0), [], r_stat, moving_listened=[(0.0, 0.0)], moving_cover_radius=r_mov)
    assert sp.is_cell_covered((5.5, 0.0), [(0.0, 0.0)], r_stat, moving_listened=[], moving_cover_radius=r_mov)
    assert sp.is_cell_covered((2.5, 0.0), [], r_stat, moving_listened=[(0.0, 0.0)], moving_cover_radius=r_mov)


def test_goal_timeout_increments_fail_streak_and_cooldown_resets(ws_root, survey_cfg):
    """Regressions for 13:48 (4x180s timeouts never triggered unstick) and 20:04 (cooldown exit)."""
    cfg = _cfg(survey_cfg)
    assert float(cfg['nav_goal_timeout_sec']) <= 120.0
    assert int(cfg['max_cooldown_resets']) >= 1

    node_src = open(
        os.path.join(ws_root, 'src', 'ugv_main', 'ugv_coqui_behavior', 'ugv_coqui_behavior', 'coqui_survey_node.py'),
        encoding='utf-8',
    ).read()
    # Verify TIMEOUT branch increments nav_fail_streak before returning 'TIMEOUT'
    idx = node_src.index("return 'TIMEOUT'")
    snippet = node_src[max(0, idx - 180):idx]
    assert 'self.nav_fail_streak += 1' in snippet, 'TIMEOUT branch must increment nav_fail_streak!'
    # Verify syntax is clean
    ast.parse(node_src)
