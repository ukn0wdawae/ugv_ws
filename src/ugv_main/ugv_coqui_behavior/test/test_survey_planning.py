"""Offline scenario tests for the Coqui survey decision logic (no ROS, no Gazebo; ~seconds).

    cd src/ugv_main/ugv_coqui_behavior && python3 -m pytest test/test_survey_planning.py -q

Scenarios are synthetic costmaps (0 free, 100 lethal, -1 unknown) at 5 cm. `GridPlanner` stands in for
Nav2's planner (A* with the robot clearance); `simulate_hunt` replays the node's hunt loop with noisy
acoustic bearings and least-squares triangulation, using the same functions the node uses.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ugv_coqui_behavior import survey_planning as sp  # noqa: E402

CFG = {
    'clearance_m': 0.25, 'lethal_cost': 65, 'snap_search_m': 2.0, 'failed_radius_m': 1.0,
    'step_min_m': 0.8, 'step_max_m': 1.8, 'step_angles': 12, 'keepout_m': 1.2,
    'approach_until_m': 5.0, 'approach_gain_deg_per_m': 6.0, 'step_cost_deg_per_m': 4.0,
    'min_rays': 6, 'min_baseline_m': 1.0, 'max_uncertainty_m': 0.6, 'min_parallax_deg': 15.0, 'min_places': 3,
    'ray_inlier_deg': 6.0, 'anchor_tol_deg': 12.0, 'max_range_m': 25.0, 'doa_sigma_deg': 3.0,
    'standoff_ground_m': 1.4, 'standoff_elevated_m': 2.2, 'vantage_angles': 16,
    'vantage_extra_radii_m': [0.0, 0.5, 1.0, 2.0], 'los_skip_end_m': 0.5,
    'no_los_penalty': 10.0, 'extra_radius_penalty': 1.0,
}
RES = 0.05


def world(size=20.0, origin=(-10.0, -10.0)):
    n = int(size / RES)
    return sp.Grid(origin[0], origin[1], RES, np.zeros((n, n), dtype=np.int16))


def disc(g, cx, cy, r, val=100):
    yy, xx = np.mgrid[0:g.height, 0:g.width]
    X = g.origin_x + (xx + 0.5) * RES
    Y = g.origin_y + (yy + 0.5) * RES
    g.data[(X - cx) ** 2 + (Y - cy) ** 2 <= r * r] = val


def rect(g, x0, x1, y0, y1, val=100):
    yy, xx = np.mgrid[0:g.height, 0:g.width]
    X = g.origin_x + (xx + 0.5) * RES
    Y = g.origin_y + (yy + 0.5) * RES
    g.data[(X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)] = val


# ------------------------------------------------------------------------------------- basics
def test_grid_queries():
    g = world()
    disc(g, 2.0, 0.0, 0.5)
    assert g.cost(2.0, 0.0) == 100 and g.cost(0.0, 0.0) == 0 and g.cost(50.0, 0.0) == -1
    assert not g.is_free(1.6, 0.0, 0.25, 65) and g.is_free(0.0, 0.0, 0.25, 65)
    assert g.line_blocked((0, 0), (4, 0), 65) and not g.line_blocked((0, 1), (4, 1), 65)
    fx, fy = g.nearest_free(2.0, 0.0, 0.25, 65, max_search=1.5)
    assert math.hypot(fx - 2.0, fy) >= 0.7


def test_bearing_attribution_to_known_frog():
    known = [{'pos': [6.0, 0.0, 0.0]}]
    assert sp.bearing_points_at_known((0, 0), 0.0, known, 2.0)
    assert sp.bearing_points_at_known((0, 0), math.radians(20), known, 2.0)      # within atan(2/6)+8 deg
    assert sp.bearing_points_at_known((0, 0), math.radians(35), known, 2.0) is None
    assert sp.near_known((5.0, 1.0), known, 2.0) and sp.near_known((3.0, 0.0), known, 2.0) is None


def test_height_classification():
    assert sp.classify_height(4.3, 2.8, 0.8) == 'elevated'
    assert sp.classify_height(3.0, 2.8, 0.8) == 'ground'


# ----------------------------------------------------------------------------- narrow bush gap
def bush_wall_with_gap(gap):
    """Two bushes forming a wall at x=2 with a gap of `gap` metres centred on y=0."""
    g = world()
    rect(g, 1.5, 2.5, -8.0, -gap / 2)
    rect(g, 1.5, 2.5, gap / 2, 8.0)
    return g


@pytest.mark.parametrize('gap,passable', [(0.3, False), (0.45, False), (0.8, True)])
def test_narrow_gap_between_bushes_is_planned_around_not_through(gap, passable):
    g = bush_wall_with_gap(gap)
    planner = sp.GridPlanner(g, CFG['clearance_m'], CFG['lethal_cost'])
    L = planner.plan((0.0, 0.0), (4.0, 0.0))
    assert L is not None                       # the wall ends at y=+-8, so there is always a way round
    if passable:
        assert L == pytest.approx(4.0, abs=0.5)    # straight through the gap
    else:
        assert L > 15.0                             # around the end of the wall, never through the gap


def test_baseline_step_never_proposes_a_spot_in_or_beyond_an_impassable_gap():
    """Frog behind a bush wall with a 0.3 m gap: the best-parallax steps are across the wall,
    but no candidate inside the bush is proposed and the planner rejects those beyond it, so the
    robot steps sideways on its own side instead."""
    g = bush_wall_with_gap(0.3)
    planner = sp.GridPlanner(g, CFG['clearance_m'], CFG['lethal_cost'])
    frog = (5.0, 0.0)
    cands = sp.baseline_step_candidates((0.5, 0.0), frog, [(0.5, 0.0)], g, CFG)
    assert cands
    for _, c in cands:
        assert g.is_free(c[0], c[1], CFG['clearance_m'], CFG['lethal_cost'])
    reachable = [c for _, c in cands if planner.plan((0.5, 0.0), c) is not None]
    assert reachable and all(c[0] < 1.5 for c in reachable)
    best = reachable[0]
    assert sp.parallax_deg(frog, [(0.5, 0.0)], best) > 8.0


# ---------------------------------------------------------------------------------- vantage
def test_vantage_frog_inside_bush_stops_outside_and_flags_no_line_of_sight():
    g = world()
    disc(g, 4.0, 0.0, 1.6)                     # dense bush, frog ~at its centre
    frog = (4.0, 0.0, 0.0)
    cands = sp.vantage_candidates(frog, (0.0, 0.0), 'ground', g, CFG)
    assert cands, 'some viewpoint outside the bush must exist'
    for _, (vx, vy, _), r, los in cands:
        assert math.hypot(vx - 4.0, vy) > 1.6 + CFG['clearance_m'] - 1e-6
        assert not los                          # the bush hides a frog sitting inside it
    planner = sp.GridPlanner(g, CFG['clearance_m'], CFG['lethal_cost'])
    assert planner.plan((0.0, 0.0), cands[0][1][:2]) is not None


def test_vantage_frog_behind_trunk_picks_a_side_with_line_of_sight():
    g = world()
    disc(g, 3.0, 0.0, 0.25)                    # trunk between robot and frog
    frog = (3.6, 0.0, 0.0)
    cands = sp.vantage_candidates(frog, (0.0, 0.0), 'ground', g, CFG)
    best = cands[0]
    assert best[3], 'best viewpoint must have line of sight'
    assert not g.line_blocked(best[1][:2], frog[:2], 65, skip_start=0.25, skip_end=0.5)
    # the straight-ahead viewpoint (in line with the trunk) is not the chosen one
    assert abs(best[1][1]) > 0.4


def test_vantage_elevated_frog_uses_longer_standoff_and_ignores_low_obstacles():
    g = world()
    disc(g, 3.0, 0.0, 0.25)                    # its trunk
    rect(g, 1.0, 1.4, -3, 3)                    # low bush row between robot and tree
    frog = (3.0, -0.3, 1.5)
    cands = sp.vantage_candidates(frog, (0.0, 0.0), 'elevated', g, CFG)
    assert cands and all(c[3] for c in cands)
    assert min(c[2] for c in cands) == pytest.approx(CFG['standoff_elevated_m'])


# ------------------------------------------------------------------------- station coverage
def test_station_candidates_skip_covered_failed_and_snap_to_free():
    g = world(size=40.0, origin=(-20.0, -20.0))
    disc(g, 3.0, 3.0, 1.0)                      # bush sitting on a cell centre
    cells = sp.coverage_cells([-6.0, 6.0, -6.0, 6.0], 6.0)     # centres at +-3
    out = sp.station_candidates((0, 0), cells, listened=[(-3.0, -3.0)], cover_radius=2.0,
                                failed=[(-3.0, 3.0)], grid=g, cfg=CFG)
    goals = [gl for gl, _ in out]
    assert len(goals) == 2                      # (-3,-3) covered, (-3,3) failed
    snapped = [gl for gl, c in out if c == (3.0, 3.0)][0]
    assert g.is_free(snapped[0], snapped[1], CFG['clearance_m'], 65)
    assert math.hypot(snapped[0] - 3.0, snapped[1] - 3.0) <= CFG['snap_search_m'] + 1e-6


# --------------------------------------------------------------------- robust triangulation
def bearing(o, target, rng, noise_deg):
    a = math.atan2(target[1] - o[1], target[0] - o[0]) + math.radians(rng.normal(0, noise_deg))
    dz = target[2] - o[2]
    el = math.atan2(dz, math.hypot(target[0] - o[0], target[1] - o[1])) + math.radians(rng.normal(0, noise_deg))
    return np.array([math.cos(el) * math.cos(a), math.cos(el) * math.sin(a), math.sin(el)])


def test_robust_triangulation_ignores_a_second_frog_calling_in_between():
    rng = np.random.default_rng(3)
    target, other = (6.0, 2.0, 0.05), (3.0, -5.0, 0.05)
    places = [(0, 0, 0.1), (0.8, -0.6, 0.1), (1.4, -1.4, 0.1), (0.4, 1.0, 0.1)]
    O, D, label = [], [], []
    for p in places:
        for _ in range(5):
            O.append(p)
            D.append(bearing(p, target, rng, 2.5))
            label.append('target')
        for _ in range(5):                     # the other frog calls just as often
            O.append(p)
            D.append(bearing(p, other, rng, 2.5))
            label.append('other')
    label = np.array(label)
    anchor = (np.array(places[0]), bearing(places[0], target, rng, 0.0))
    tri = sp.robust_triangulate(O, D, anchor, CFG)
    assert tri is not None
    assert not np.any(tri['inlier_mask'] & (label == 'other')), "the other frog's bearings must not be used"
    assert np.sum(tri['inlier_mask'] & (label == 'target')) >= 17
    # same answer as triangulating the target's bearings alone (noise limits accuracy, not the mixing)
    alone = sp.robust_triangulate([o for o, lb in zip(O, label) if lb == 'target'],
                                  [d for d, lb in zip(D, label) if lb == 'target'], anchor, CFG)
    assert math.hypot(tri['pos'][0] - alone['pos'][0], tri['pos'][1] - alone['pos'][1]) < 0.05
    assert math.hypot(tri['pos'][0] - 6.0, tri['pos'][1] - 2.0) < 1.2   # ~25 deg parallax, 2.5 deg noise
    assert tri['n_origins'] == 4 and tri['parallax_deg'] > 15
    # anchoring on the other frog finds the other frog from the same bearings
    anchor2 = (np.array(places[0]), bearing(places[0], other, rng, 0.0))
    tri2 = sp.robust_triangulate(O, D, anchor2, CFG)
    assert not np.any(tri2['inlier_mask'] & (label == 'target'))
    assert math.hypot(tri2["pos"][0] - 3.0, tri2["pos"][1] + 5.0) < 1.2


def test_robust_triangulation_needs_two_places_and_recovers_height():
    rng = np.random.default_rng(0)
    frog = (5.0, 0.0, 1.6)                      # perched 1.5 m above the robot's ground
    one = [(0, 0, 0.1)] * 6
    D1 = [bearing(p, frog, rng, 2.0) for p in one]
    anchor = (np.array(one[0]), D1[0])
    assert sp.robust_triangulate(one, D1, anchor, CFG) is None
    two = one + [(0.5, 1.2, 0.1)] * 6 + [(1.0, -1.2, 0.1)] * 6
    D2 = D1 + [bearing(p, frog, rng, 2.0) for p in two[6:]]
    tri = sp.robust_triangulate(two, D2, anchor, CFG)
    assert tri is not None and math.hypot(tri['pos'][0] - 5.0, tri['pos'][1]) < 0.5
    assert abs(tri['pos'][2] - 1.6) < 0.5
    assert sp.classify_height(tri['pos'][2], 0.0, 0.8) == 'elevated'


# ---------------------------------------------------------------------- simulated hunt loop
def simulate_hunt(g, robot, frog, noise_deg=3.0, seed=0, max_steps=8, rays_per_stop=4, distractor=None):
    """Replays the node's HUNT loop: listen (noisy bearings, optionally a second frog calling too),
    robust triangulation anchored on the first call, pick the best reachable baseline step, move.
    Returns (estimate, steps, robot path)."""
    rng = np.random.default_rng(seed)
    planner = sp.GridPlanner(g, CFG['clearance_m'], CFG['lethal_cost'])
    frog3 = (frog[0], frog[1], 0.05)
    O, D, path = [], [], [tuple(robot)]
    anchor = (np.array([robot[0], robot[1], 0.1]), bearing((robot[0], robot[1], 0.1), frog3, rng, noise_deg))
    tri = None
    for step in range(max_steps + 1):
        o = (robot[0], robot[1], 0.1)
        for _ in range(rays_per_stop):
            O.append(o)
            D.append(bearing(o, frog3, rng, noise_deg))
            if distractor is not None:
                O.append(o)
                D.append(bearing(o, (distractor[0], distractor[1], 0.05), rng, noise_deg))
        tri = sp.robust_triangulate(O, D, anchor, CFG)
        if tri is None:
            est = (anchor[0][0] + 6.0 * anchor[1][0], anchor[0][1] + 6.0 * anchor[1][1])
            places = [tuple(robot)]
        else:
            est = tri['pos'][:2]
            places = [tuple(oo[:2]) for oo, m in zip(O, tri['inlier_mask']) if m]
            if sp.triangulation_done(tri, CFG):
                break
        cands = sp.baseline_step_candidates(robot, est, places, g, CFG)
        nxt = None
        for _, c in cands[:6]:
            if planner.plan(robot, c) is not None:
                nxt = c
                break
        assert nxt is not None, 'no reachable baseline step'
        assert g.is_free(nxt[0], nxt[1], CFG['clearance_m'], CFG['lethal_cost'])
        robot = nxt
        path.append(tuple(robot))
    return (tri['pos'] if tri else est), step, path


@pytest.mark.parametrize('name,setup,frog', [
    ('open', lambda g: None, (5.0, 1.0)),
    ('behind_trunk', lambda g: disc(g, 4.4, 1.0, 0.25), (5.0, 1.0)),
    ('inside_bush', lambda g: disc(g, 5.0, 1.0, 1.5), (5.0, 1.0)),
    ('bush_wall_with_small_gap', lambda g: (rect(g, 2.5, 3.2, -8, 0.85), rect(g, 2.5, 3.2, 1.15, 8)), (5.0, 1.0)),
])
@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('distractor', [None, (2.0, -6.0)])
def test_simulated_hunt_converges_without_entering_obstacles(name, setup, frog, seed, distractor):
    g = world()
    setup(g)
    est, steps, path = simulate_hunt(g, (0.0, 0.0), frog, seed=seed, distractor=distractor)
    err = math.hypot(est[0] - frog[0], est[1] - frog[1])
    assert err < 0.8, f'{name}: estimate {est} is {err:.2f} m off after {steps} steps'
    assert steps <= 6
    for p in path:
        assert g.is_free(p[0], p[1], CFG['clearance_m'], CFG['lethal_cost'])


# ------------------------------------------------------------------ recorded Gazebo regression
def test_recorded_gazebo_hunt_bush_frog_with_base_parameters():
    """Bearings recorded in Gazebo while hunting the frog inside the bush. With the first parameters
    (6 deg window, inlier counting) a neighbouring frog's bout pulled the estimate 0.95 m off; with the
    base parameters (2 deg, MSAC scoring, bouts) it must land on the frog."""
    import json
    import yaml
    fx = json.load(open(os.path.join(os.path.dirname(__file__), 'data', 'hunt_bush_frog_gazebo.json')))
    base = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), '..', 'config', 'survey_base.yaml')))
    cfg = base['coqui_survey']['ros__parameters']
    obs = np.array(fx['obs'])
    anchor = (np.array(fx['anchor'][0]), np.array(fx['anchor'][1]))
    groups = sp.bout_ids(list(obs[:, 0]), cfg['bout_gap_s'])
    tri = sp.robust_triangulate(obs[:, 1:4], obs[:, 4:7], anchor, cfg, groups=groups)
    true = fx['true_frog']
    assert math.hypot(tri['pos'][0] - true[0], tri['pos'][1] - true[1]) < 0.3
    assert abs(tri['pos'][2] - true[2]) < 0.3


def test_vantage_prefers_ground_already_seen_free_over_unknown():
    """Next to a bush the unseen cells usually turn out to be bush: only seen-free viewpoints are
    offered unless asked otherwise."""
    g = world()
    disc(g, 4.0, 0.0, 1.0)                     # bush with the frog in it
    rect(g, 4.0, 10.0, -10.0, 10.0, val=-1)    # far side of the bush not seen yet
    frog = (4.0, 0.0, 0.0)
    seen = sp.vantage_candidates(frog, (0.0, 0.0), 'ground', g, CFG)
    assert seen and all(v[0] < 4.0 for _, v, _, _ in seen)
    anyc = sp.vantage_candidates(frog, (0.0, 0.0), 'ground', g, CFG, allow_unknown=True)
    assert any(v[0] > 4.0 for _, v, _, _ in anyc)


def test_baseline_steps_prefer_ground_already_seen_free():
    g = world()
    rect(g, 0.6, 10.0, -10.0, 10.0, val=-1)    # everything beyond x=0.6 unseen (e.g. a bush edge)
    cands = sp.baseline_step_candidates((0.0, 0.0), (5.0, 0.0), [(0.0, 0.0)], g, CFG)
    assert cands and all(c[0] <= 0.6 for _, c in cands)
    anyc = sp.baseline_step_candidates((0.0, 0.0), (5.0, 0.0), [(0.0, 0.0)], g, CFG, allow_unknown=True)
    assert any(c[0] > 0.6 for _, c in anyc)


# ----------------------------------------------------------------------- frontier & two-tier acoustics
def test_frontier_candidates_cluster_heading_and_whisper_bias():
    g = world(size=20.0, origin=(-10.0, -10.0))
    g.data[:, :] = -1
    # robot mapped a 6 m square around (0, 0), leaving frontiers to east (+x) and west (-x)
    rect(g, -3.0, 3.0, -3.0, 3.0, val=0)
    # obstacle blocking north and south borders
    rect(g, -3.5, 3.5, 2.8, 3.5, val=100)
    rect(g, -3.5, 3.5, -3.5, -2.8, val=100)
    area = [-9.0, 9.0, -9.0, 9.0]
    # facing east (yaw=0): east frontier wins via heading continuity
    cands = sp.frontier_candidates((0.0, 0.0), 0.0, area, g, [], CFG)
    assert len(cands) >= 2
    assert cands[0][1][0] > 2.0
    # weak Tier-1 whisper ray pointing west (pi): flips preference to west frontier!
    biased = sp.frontier_candidates((0.0, 0.0), 0.0, area, g, [], CFG, weak_rays=[((0.0, 0.0), math.pi, 11.5)])
    assert biased[0][1][0] < -2.0
    assert sp.frontier_still_unknown(biased[0][1], g, 1.0)
    # filtering to uncovered cells drops frontiers in already-covered areas (e.g. tree shadows)
    west_only = sp.frontier_candidates((0.0, 0.0), 0.0, area, g, [], CFG,
                                       uncovered_xy=[(-5.0, 0.0)], cover_radius=4.0)
    assert west_only and all(xy[0] < 0.0 for _, xy, _ in west_only)


def test_adaptive_baseline_steps_fit_tight_corridor_and_scale_with_range():
    g = world()
    # narrow 0.9 m corridor along x, walls at y = +-0.45
    rect(g, -8.0, 8.0, 0.45, 8.0, val=100)
    rect(g, -8.0, 8.0, -8.0, -0.45, val=100)
    acfg = dict(CFG, clearance_m=0.18, adaptive_steps=True)
    # close frog off to the side: nimble 0.45 m steps along the narrow corridor are offered
    close_cands = sp.baseline_step_candidates((0.0, 0.0), (0.0, 2.5), [(0.0, 0.0)], g, acfg)
    assert close_cands
    dists = [math.hypot(c[0], c[1]) for _, c in close_cands]
    assert min(dists) < 0.6
    # distant frog (12 m): longer steps > 2.0 m are offered for faster parallax
    far_cands = sp.baseline_step_candidates((0.0, 0.0), (0.0, 12.0), [(0.0, 0.0)], world(), acfg)
    assert max(math.hypot(c[0], c[1]) for _, c in far_cands) > 2.2


def test_passive_triangulation_catches_quiet_frog_while_ignoring_known():
    rng = np.random.default_rng(7)
    frog = (6.0, 3.0, 2.8)
    obs = []
    t = 0.0
    for ox, oy in [(0.0, 0.0), (2.5, 0.0), (5.0, 0.0)]:
        for _ in range(3):
            o = np.array([ox, oy, 2.8])
            d = bearing(o, frog, rng, 1.0)
            obs.append((t, o, d, 11.8))  # quiet (< hunt_trigger_snr_db)
            t += 0.5
        t += 2.0  # distinct bouts
    cfg = dict(CFG, known_frog_radius_m=2.0, bout_gap_s=1.2)
    tri, anchor = sp.check_passive_triangulation(obs, [], cfg)
    assert tri is not None and anchor is not None
    assert math.hypot(tri['pos'][0] - frog[0], tri['pos'][1] - frog[1]) < 0.6
    # once recorded as known, passive check ignores it
    tri2, _ = sp.check_passive_triangulation(obs, [{'pos': list(frog)}], cfg)
    assert tri2 is None


def test_intrabount_moving_triangulation_and_area_boundary_margin():
    rng = np.random.default_rng(11)
    frog = (40.5, -2.5, 3.0)
    # Single continuous calling bout (dt=0.75s < bout_gap_s=1.2s) while fast robot moves 2.7 m
    O = [np.array([46.8 - 0.1 * k, 0.0 - 0.25 * k, 2.8]) for k in range(11)]
    D = [bearing(o, frog, rng, 0.8) for o in O]
    groups = sp.bout_ids([0.75 * k for k in range(11)], 1.2)
    assert len(set(groups)) == 1
    tri = sp.robust_triangulate(O, D, (O[0], D[0]), CFG, groups=groups)
    assert tri is not None
    assert math.hypot(tri['pos'][0] - frog[0], tri['pos'][1] - frog[1]) < 0.6
    # Boundary safety margin keeps baseline steps and vantages strictly inside survey_area
    g = world(size=30.0, origin=(30.0, -15.0))
    bcfg = dict(CFG, survey_area=[30.0, 50.0, -15.0, 15.0], area_margin_m=1.2)
    cands = sp.baseline_step_candidates((48.6, -6.0), (47.5, -9.0), [(48.6, -6.0)], g, bcfg)
    assert cands and all(c[0] <= 48.8 + 1e-9 for _, c in cands)
    vcands = sp.vantage_candidates((48.5, -6.0, 3.0), (46.0, -6.0), 'ground', g, bcfg)
    assert vcands and all(v[0] <= 48.8 + 1e-9 for _, v, _, _ in vcands)
    # Close-range elevation rays outweigh noisy distant rays in z estimation
    elev_frog = (47.5, -9.5, 4.4)
    O_far = [np.array([24.0 + 0.3 * k, -8.5, 3.5]) for k in range(6)]
    D_far = [np.array([elev_frog[0] - o[0], elev_frog[1] - o[1], -1.5]) for o in O_far]
    D_far = [d / np.linalg.norm(d) for d in D_far]
    O_near = [np.array([46.5, -3.5 + 0.5 * k, 2.8]) for k in range(3)]
    D_near = [bearing(o, elev_frog, rng, 0.2) for o in O_near]
    tri_z = sp.robust_triangulate(O_far + O_near, D_far + D_near, (O_near[-1], D_near[-1]), CFG)
    assert tri_z is not None and abs(tri_z['pos'][2] - 4.4) < 0.3


def test_refinement_cannot_drift_behind_anchor_onto_collinear_distant_frog():
    """When the robot drives west past a close foreground frog while earlier bouts pointed straight
    west at a distant collinear frog, the distant frog's rays pass near the anchor. Refinement must
    reject any hypothesis that pulls the estimate off or behind the anchor bearing."""
    rng = np.random.default_rng(19)
    near_frog = (30.0, -13.9, 2.6)
    far_frog = (16.1, -11.5, 2.7)
    O, D, times = [], [], []
    # Bout 0: robot driving west from x=32.4 to 30.2 hearing the distant west frog
    for k in range(10):
        o = np.array([32.4 - 0.24 * k, -13.8 + 0.04 * k, 2.65])
        O.append(o)
        D.append(bearing(o, far_frog, rng, 0.5))
        times.append(0.75 * k)
    # Bout 1: robot passes the close frog from x=30.1 to 28.2, anchor points SE (+x, -y)
    for k in range(10):
        o = np.array([30.1 - 0.21 * k, -13.4 + 0.08 * k, 2.70])
        O.append(o)
        D.append(bearing(o, near_frog, rng, 1.0))
        times.append(10.0 + 0.75 * k)
    anchor = (O[14], D[14])
    groups = sp.bout_ids(times, 1.2)
    cfg = dict(CFG, ray_inlier_deg=2.0, anchor_tol_deg=8.0)
    tri = sp.robust_triangulate(O, D, anchor, cfg, groups=groups)
    assert tri is not None
    assert math.hypot(tri['pos'][0] - near_frog[0], tri['pos'][1] - near_frog[1]) < 0.35


def test_low_slope_preference_in_vantage_frontiers_and_path_cost():
    g = world(size=20.0, origin=(-10.0, -10.0))
    # North half (y > 0) is a 18 deg sloped hillside (cost=55); south half (y <= 0) is flat trail (cost=0)
    rect(g, -10.0, 10.0, 0.2, 10.0, val=55)
    # 1) Vantage around an elevated tree frog at (0, 0): robot starts slightly uphill at (0, 1.5),
    #    yet flat trail viewpoints (cost=0, y <= 0) must outrank closer tilted hillside viewpoints (y > 0.2, cost=55)
    vcands = sp.vantage_candidates((0.0, 0.0, 1.5), (0.0, 1.5), 'elevated', g, CFG)
    assert vcands and vcands[0][1][1] <= 0.01
    assert g.cost(vcands[0][1][0], vcands[0][1][1]) == 0
    # 2) Terrain-weighted path cost: 5 m flat detour beats 3 m steep hillside shortcut
    flat_path = [(0.0, -1.0), (5.0, -1.0)]
    steep_path = [(0.0, 1.0), (3.0, 1.0)]
    assert sp.path_terrain_cost(flat_path, g, slope_weight=3.0) < sp.path_terrain_cost(steep_path, g, slope_weight=3.0)
