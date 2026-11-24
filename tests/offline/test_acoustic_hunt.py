"""Hunting one frog in a chorus: regression for the mission of 2026-11-24 and closed-loop hunts.

The recorded mission (tests/cases/two_frog_phantoms_20261124.json) circled between a frog in tall grass and one in the open
for minutes, intersected bearings of different frogs, saved two frogs that do not exist and photographed empty ground.
What must stay fixed:
  * bearings of two frogs never become one fix (loudness, per-place support, honest uncertainty);
  * an unseen frog is recorded only on a complete fix, "the view was blocked" is not enough;
  * the robot leaves a listening place only after the frog it hunts was heard there, and never shuttles between places;
  * a hunt that fails leaves a lead and does not count as covered ground;
  * exploring commits to a frontier.
"""
import json
import math
import os

import numpy as np
import pytest

from ugv_coqui_behavior import acoustic_hunt as ah
from ugv_coqui_behavior import survey_planning as sp

import hunt_sim as H


@pytest.fixture(scope='module')
def cfg(survey_cfg):
    return dict(survey_cfg['coqui_survey']['ros__parameters'])


@pytest.fixture(scope='module')
def case(ws_root):
    c = json.load(open(os.path.join(ws_root, 'tests', 'cases', 'two_frog_phantoms_20261124.json')))
    # that build logged SNR instead of level; a nominal call at 1 m measured 45 dB
    for h in c['hunts']:
        h['rays'] = [(o[0], np.array(o[1:4]), np.array(o[4:7]), o[7] - 45.0) for o in h['obs']]
        a_o, a_d = np.array(h['anchor'][0]), np.array(h['anchor'][1])
        h['anchor_ray'] = min(h['rays'], key=lambda r: np.linalg.norm(r[1][:2] - a_o[:2]) + abs(sp.wrap(
            math.atan2(r[2][1], r[2][0]) - math.atan2(a_d[1], a_d[0]))))
    return c


def _fix(h, cfg):
    O, D, L = [r[1] for r in h['rays']], [r[2] for r in h['rays']], [r[3] for r in h['rays']]
    a = h['anchor_ray']
    return sp.robust_triangulate(O, D, (a[1], a[2]), cfg, groups=sp.ray_groups(O, D, L), loudness=L, anchor_loudness=a[3])


def _nearest(case, pos):
    return min(math.hypot(x - pos[0], y - pos[1]) for x, y in case['true_frogs'].values())


# --------------------------------------------------------------------------------- the recorded mission
def test_case_really_contains_the_failure(case):
    """The fixture is the run as it happened: two saved frogs, both far from any real one, both claiming < 0.2 m."""
    saved = [h for h in case['hunts'] if h['photo'] and h['photo']['result'] == 'recorded']
    assert len(saved) == 2
    for h in saved:
        assert _nearest(case, h['reported']['pos']) > 1.5
        assert h['reported']['uncertainty_m'] < 0.2
        assert not h['photo']['visual']


def test_recorded_phantoms_are_not_recorded_again(case, cfg):
    """Replaying that run's bearings: nothing it heard may be saved without being seen."""
    for k, h in enumerate(case['hunts']):
        tri = _fix(h, cfg)
        ok, why = sp.should_accept_record(tri, False, 'reached (no clear line of sight on costmap)', False, None, cfg) \
            if tri is not None else (False, 'no fix')
        assert not ok, f'hunt {k} would be saved unseen: {why} {tri["pos"]}, {_nearest(case, tri["pos"]):.2f} m from a frog'


def test_circling_hunts_no_longer_produce_a_fix(case, cfg):
    """Hunts 2 and 3 orbited a crossing of a western and an eastern frog's bearings: loudness rules the crossing out."""
    for k in (1, 2):
        tri = _fix(case['hunts'][k], cfg)
        assert tri is None or not sp.accept_partial_triangulation(tri, cfg), (k, tri and tri['pos'])


def test_uncertainty_of_the_recorded_fixes_is_honest(case, cfg):
    """Any fix still produced from that run must not claim centimetres: its poses were up to 1.7 m off."""
    for k, h in enumerate(case['hunts']):
        tri = _fix(h, cfg)
        if tri is not None and sp.triangulation_done(tri, cfg):
            assert _nearest(case, tri['pos']) < 1.7 + 2 * tri['uncertainty_m'], (k, tri['pos'], tri['uncertainty_m'])


def test_recorded_robot_left_before_hearing_its_frog(case, cfg):
    """In that run most baseline steps were taken without one call of the hunted frog from the place being left. The
    tracker, fed the same bearings, must ask to stay there."""
    stays = 0
    for h in case['hunts']:
        trk = ah.HuntTracker(h['anchor_ray'], cfg, h['anchor_ray'][0])
        for s in h['steps'][1:]:
            xy = min((r for r in h['rays'] if r[0] <= s['t']), key=lambda r: s['t'] - r[0])[1]
            trk.at_place(xy[:2], s['t'])
            for r in h['rays']:
                if r[0] <= s['t'] and not any(q[0] == r[0] for q in trk.obs):
                    trk.add(r)
            if trk.decide(s['t'] + 0.1)[0] == 'listen':
                stays += 1
    assert stays >= 8, stays


# ----------------------------------------------------------------------------------- two-frog geometry
def _rays(frog, places, n=4, sd=0.6, loud=0.0, seed=0, t0=0.0):
    rng = np.random.default_rng(seed)
    out = []
    for i, p in enumerate(places):
        for k in range(n):
            b = math.atan2(frog[1] - p[1], frog[0] - p[0]) + rng.normal(0, math.radians(sd))
            d = math.hypot(frog[0] - p[0], frog[1] - p[1])
            out.append((t0 + 10 * i + k, np.array([p[0], p[1], 0.1]), np.array([math.cos(b), math.sin(b), 0.0]),
                        loud - 20 * math.log10(d) + rng.normal(0, 1.0)))
    return out


def _tri(rays, anchor, cfg):
    O, D, L = [r[1] for r in rays], [r[2] for r in rays], [r[3] for r in rays]
    return sp.robust_triangulate(O, D, (anchor[1], anchor[2]), cfg, groups=sp.ray_groups(O, D, L), loudness=L,
                                 anchor_loudness=anchor[3])


def test_frogs_on_both_sides_do_not_make_a_frog_under_the_robot(cfg):
    """One frog 9 m west, one 9 m east: their bearings cross next to the robot wherever it stands."""
    places = [(0.0, 0.0), (0.4, 1.4), (0.9, 2.8)]
    west, east = _rays((-9.0, 0.6), places, seed=1), _rays((9.0, 0.4), places, seed=2)
    tri = _tri(west + east, west[0], cfg)
    assert tri is None or math.hypot(tri['pos'][0] + 9.0, tri['pos'][1] - 0.6) < 2.5, tri['pos']
    # with bearings only, the same data is a convincing phantom: that is what loudness is for
    O, D = [r[1] for r in west[:4] + east[4:8]], [r[2] for r in west[:4] + east[4:8]]
    cross = sp._intersect2d(O[0], D[0], O[-1], D[-1])
    assert cross is not None


def test_two_frogs_on_one_bearing_are_told_apart(cfg):
    """A loud frog 4 m away hides a farther one behind it on the same bearing: the fix is the one that was anchored."""
    places = [(0.0, 0.0), (0.6, 2.0), (1.4, -2.2)]
    near, far = _rays((4.0, 0.3), places, seed=3), _rays((11.0, 0.6), places, seed=4)
    for anchor, frog in ((near[0], (4.0, 0.3)), (far[0], (11.0, 0.6))):
        tri = _tri(near + far, anchor, cfg)
        assert tri is not None and math.hypot(tri['pos'][0] - frog[0], tri['pos'][1] - frog[1]) < 1.0, (frog, tri['pos'])


def test_a_faint_call_behind_a_recorded_frog_is_not_masked(cfg):
    known = [{'pos': [4.0, 0.0, 0.0], 'source_level_db': 0.0}]
    same = -20 * math.log10(4.0)
    behind = -20 * math.log10(11.0)
    r = cfg['known_bearing_radius_m']
    assert sp.bearing_points_at_known((0, 0), 0.0, known, r, loudness=same, cfg=cfg) is not None
    assert sp.bearing_points_at_known((0, 0), 0.0, known, r, loudness=behind, cfg=cfg) is None


def test_stray_bearings_do_not_count_as_places(cfg):
    """Six bearings from three places are not enough when two of the places heard the frog once."""
    tri = {'inliers': 7, 'n_origins': 3, 'baseline_m': 4.0, 'parallax_deg': 60.0, 'uncertainty_m': 0.15, 'rms_deg': 0.6,
           'level_sd_db': 1.0, 'rays_per_place': [5, 1, 1]}
    assert not sp.triangulation_done(tri, cfg)
    assert sp.triangulation_done(dict(tri, rays_per_place=[3, 2, 2]), cfg)
    assert not sp.triangulation_done(dict(tri, rays_per_place=[3, 2, 2], level_sd_db=5.0), cfg)


def test_fix_error_grows_when_the_places_are_in_line_with_the_frog():
    good = sp.fix_sigma_m((5, 0), [(0, 0), (0, 3), (2, 3)], math.radians(1.0))
    line = sp.fix_sigma_m((5, 0), [(0, 0), (1, 0.02), (2, 0)], math.radians(1.0))
    opposite = sp.fix_sigma_m((5, 0), [(0, 0), (10, 0.05)], math.radians(1.0))
    assert good < 0.3 < 1.0 < line and opposite > 1.0


def test_unseen_frog_needs_a_complete_fix(cfg):
    part = {'pos': [0, 0, 0], 'inliers': 6, 'n_origins': 2, 'baseline_m': 2.0, 'parallax_deg': 40.0, 'uncertainty_m': 0.2,
            'rms_deg': 0.5, 'rays_per_place': [3, 3], 'level_sd_db': 0.5}
    full = dict(part, n_origins=3, rays_per_place=[3, 3, 3], inliers=9)
    blocked = 'reached (no clear line of sight on costmap)'
    assert not sp.should_accept_record(part, False, blocked, False, None, cfg)[0]
    assert sp.should_accept_record(full, False, blocked, False, None, cfg)[0]
    assert sp.should_accept_record(part, True, 'reached', True, None, cfg)[0]      # seen is seen


# ---------------------------------------------------------------------------------- closed-loop hunts
def _run(seed, cfg, **kw):
    start = [(38, -5.5), (43, -6), (38, 2), (30, -6), (45, -4)][seed % 5]
    return H.hunt(H.Ears(H.FOREST, seed, drift_m=0.3, cover=H.COVER, **kw), start, H.open_world(start), cfg)


@pytest.mark.parametrize('seed', range(5))
def test_hunts_in_the_forest_chorus_find_a_real_frog(seed, cfg):
    r = _run(seed, cfg)
    assert r['outcome'].startswith('converged'), r['outcome']
    assert r['err_m'] < 0.6, (r['nearest'], r['err_m'])
    assert r['duration'] < 180 and r['steps'] <= 6


def test_robot_never_leaves_before_hearing_its_frog_and_never_shuttles(cfg):
    for seed in range(8):
        r = _run(seed, cfg)
        assert all(s['heard_target_before_step'] >= 1 for s in r['log']), seed
        pl = r['places']
        for i in range(len(pl)):
            for j in range(i + 1, len(pl)):
                assert math.hypot(pl[i][0] - pl[j][0], pl[i][1] - pl[j][1]) >= 0.5, (seed, pl)


def test_dense_chorus_yields_almost_no_phantoms(cfg):
    """12 frogs in 26 x 26 m, worse bearings: a complete fix must be a real frog."""
    complete, phantom = 0, 0
    for seed in range(14):
        rng = np.random.default_rng(1000 + seed)
        frogs = {f'f{k}': (rng.uniform(-13, 13), rng.uniform(-13, 13)) for k in range(12)}
        start = (rng.uniform(-2, 2), rng.uniform(-2, 2))
        r = H.hunt(H.Ears(frogs, seed, bearing_sd_deg=0.9, drift_m=0.3), start, H.open_world(start), cfg)
        if r['outcome'].startswith('converged') and sp.triangulation_done(r['tri'], cfg):
            complete += 1
            phantom += r['err_m'] > 1.0
    assert complete >= 10 and phantom <= 1, (complete, phantom)


def test_robot_waits_for_a_frog_that_pauses(cfg):
    """The target stops calling for half a minute mid-hunt while another frog keeps calling: stay, do not wander."""
    rays = _rays((6.0, 0.0), [(0.0, 0.0)], n=3)
    other = _rays((-5.0, 3.0), [(0.0, 0.0)], n=30, seed=9)
    trk = ah.HuntTracker(rays[0], cfg, 0.0)
    for r in rays[1:]:
        trk.add(r)
    trk.at_place((0.0, 0.0), 2.5)
    assert trk.decide(3.0)[0] == 'step'
    trk.steps += 1
    trk.at_place((0.4, 2.0), 10.0)
    for k, o in enumerate(other):
        trk.add((11.0 + k, np.array([0.4, 2.0, 0.1]), o[2], o[3]))
    assert trk.decide(40.0)[0] == 'listen'
    assert trk.decide(10.0 + cfg['hunt_listen_max_sec'] + 1)[0] == 'end'     # ... but not forever


# ------------------------------------------------------------------------------- viewpoint and photo
def test_listening_at_the_viewpoint_refines_the_fix_and_neighbours_do_not_move_it(cfg):
    frog = (8.0, 1.0)
    rays = _rays(frog, [(0.0, 0.0), (0.8, 2.0), (2.0, -1.5)], sd=1.5, seed=8)
    trk = ah.HuntTracker(rays[0], cfg, 0.0, seed=rays)
    before = trk.triangulate()
    err0 = math.hypot(before['pos'][0] - frog[0], before['pos'][1] - frog[1])
    view = (6.6, 1.0)
    used = trk.refine(_rays(frog, [view], n=4, sd=1.0, seed=9, t0=100.0) +
                      _rays((view[0] + 0.5, view[1] - 9.0), [view], n=6, seed=10, t0=110.0), view)
    assert len(used) == 4, 'only the frog at the fix is used'
    after = trk.triangulate()
    assert math.hypot(after['pos'][0] - frog[0], after['pos'][1] - frog[1]) <= max(err0, 0.12)
    assert trk.refine(_rays((view[0], view[1] + 8.0), [view], n=5, seed=11, t0=120.0), view) == []


def test_phantom_is_silent_at_its_own_viewpoint(cfg):
    """Nothing calls where a phantom is: standing 1.5 m from it, every call comes from somewhere else (or too faint)."""
    view, phantom = (0.0, 0.0), (1.5, 0.0)
    elsewhere = _rays((0.0, 6.0), [view], n=4, seed=1) + _rays((-5.0, -3.0), [view], n=4, seed=2)
    behind = _rays((12.0, 0.2), [view], n=4, seed=3)       # on its bearing, but 18 dB too faint to be 1.5 m away
    assert sp.heard_from_here(elsewhere + behind, view, phantom, cfg) == []
    assert len(sp.heard_from_here(_rays(phantom, [view], n=3, seed=4), view, phantom, cfg)) == 3


def test_unseen_frog_not_heard_at_the_viewpoint_is_not_recorded(cfg):
    full = {'pos': [0, 0, 0], 'inliers': 9, 'n_origins': 3, 'baseline_m': 2.0, 'parallax_deg': 40.0, 'uncertainty_m': 0.2,
            'rms_deg': 0.5, 'rays_per_place': [3, 3, 3], 'level_sd_db': 0.5}
    blocked = 'reached (no clear line of sight on costmap)'
    assert not sp.should_accept_record(full, False, blocked, False, None, cfg, heard_near=False)[0]
    assert sp.should_accept_record(full, False, blocked, False, None, cfg, heard_near=True)[0]
    assert sp.should_accept_record(full, True, 'reached', True, None, cfg, heard_near=False)[0]


def test_camera_aim_is_checked_after_the_robot_stopped(cfg):
    """Regression (2026-11-24): Nav2 said "reached" while the robot turned at 1 rad/s; aimed once, the camera ended up
    28 and 60 degrees off and photographed empty ground. Numbers from that run."""
    frog, view = (35.29, -2.61), (36.19, 0.71)
    aimed_yaw, final_yaw = math.radians(-108.6), math.radians(-80.4)
    pan = -sp.wrap(math.atan2(frog[1] - view[1], frog[0] - view[0]) - aimed_yaw)
    assert abs(math.degrees(sp.pan_error(view, aimed_yaw, pan, frog))) < 1e-6
    assert abs(math.degrees(sp.pan_error(view, final_yaw, pan, frog))) > cfg['aim_tolerance_deg']
    src = open(os.path.join(os.path.dirname(sp.__file__), 'coqui_survey_node.py')).read()
    aim = src[src.index('    def aim(self, est)'):src.index('    def _aim_once')]
    assert 'wait_still()' in aim and 'pan_error' in aim and 'aim_tolerance_deg' in aim


# ---------------------------------------------------------------------------------------- leads, coverage
def test_failed_hunt_is_not_coverage():
    assert ah.coverage_places([(0, 0), (1, 1)], recorded=False) == []
    assert ah.coverage_places([(0, 0), (1, 1)], recorded=True) == [(0, 0), (1, 1)]


def test_lead_book_remembers_retries_and_resolves(cfg):
    book = ah.LeadBook(cfg)
    a = _rays((6.0, 0.0), [(0.0, 0.0)], n=1)[0]
    ld = book.note(a, (6.2, 0.3), 1.5, now=0.0, reason='target went quiet')
    assert book.due((0, 0), 10.0) == []                                 # not straight away
    assert book.due((0, 0), cfg['lead_retry_sec'] + 1)[0] is ld
    book.note(a, (5.8, -0.2), 1.0, now=200.0, reason='step limit')       # same frog: one lead, two attempts
    assert len(book.leads) == 1 and book.leads[0]['attempts'] == 2
    book.note(a, (6.0, 0.0), 1.0, now=400.0, reason='step limit')
    assert book.exhausted(book.leads[0]) and book.due((0, 0), float('inf')) == []
    assert book.records()[0]['attempts'] == 3
    assert book.resolve_near((6.1, 0.1), cfg['known_frog_radius_m']) and book.leads == []


def test_lead_listening_places_are_new_free_and_outside_keepouts(cfg):
    book = ah.LeadBook(cfg)
    c = dict(cfg, survey_area=[-20, 20, -20, 20])
    ld = book.note(_rays((6.0, 0.0), [(0.0, 0.0)], n=1)[0], (6.0, 0.0), 2.0, 0.0, 'x')
    g = H.open_world((0, 0))
    H.put_disc(g, 6.0, 4.0, 1.5)
    places = book.listening_places(ld, (0.0, 0.0), g, c)
    assert places and all(g.is_free(x, y, c['clearance_m'], c['lethal_cost'], allow_unknown=False) for x, y in places)
    best = math.degrees(abs(sp.wrap(math.atan2(places[0][1], places[0][0] - 6.0) - math.pi)))
    assert best > 10.0, 'the first place should see the lead from a new direction'


def test_call_heard_on_the_move_is_anchored_standing(cfg):
    """A bearing taken while driving is only good enough to stop for; the hunt starts from a clean standing one."""
    rough = _rays((8.0, 1.0), [(-1.0, 0.0)], n=1, sd=5.0, seed=4)[0]
    stand = _rays((8.0, 1.0), [(0.0, 0.0)], n=2, seed=5)
    other = _rays((-6.0, 5.0), [(0.0, 0.0)], n=2, seed=6)
    picked = ah.pick_anchor(other + stand, rough, cfg)
    assert any(picked is s for s in stand)
    assert ah.pick_anchor(other, rough, cfg) is None


# ------------------------------------------------------------------------------------------- exploring
def test_frontier_commitment(case, cfg):
    """The recorded run picked 29 frontiers in one 65 s drive because a frontier counts as seen a second after it is
    chosen. Now it drives 2.5 m or 8 s towards it first (unless it has arrived)."""
    assert not sp.frontier_handoff(6.0, 0.4, 1.0, False, cfg)
    assert sp.frontier_handoff(6.0, 2.6, 5.0, False, cfg)
    assert sp.frontier_handoff(6.0, 1.0, 9.0, False, cfg)
    assert not sp.frontier_handoff(6.0, 5.0, 20.0, True, cfg)
    assert sp.frontier_handoff(1.5, 4.2, 9.0, True, cfg, dist_start=6.0)
    assert not sp.frontier_handoff(1.9, 0.4, 0.9, True, cfg, dist_start=2.4)     # a near frontier is driven to, too
    assert sp.frontier_handoff(0.9, 1.5, 4.0, True, cfg, dist_start=2.4)
    gaps = np.diff([g['t'] for g in case['explore_goals']])
    assert np.median(gaps) < 3.0      # the fixture does contain the churn


# ----------------------------------------------------------------------------------------- GPS-denied
def test_survey_never_uses_gps(ws_root):
    """The forest is GPS-denied: nothing on the robot side of the survey may read a GPS."""
    beh = os.path.join(ws_root, 'src', 'ugv_main', 'ugv_coqui_behavior')
    files = [os.path.join(beh, 'ugv_coqui_behavior', f) for f in (
        'coqui_survey_node.py', 'survey_planning.py', 'survey_store.py', 'acoustic_hunt.py', 'lidar_localizer_core.py',
        'lidar_localizer.py', 'wheel_odom_relay.py')]
    files += [os.path.join(beh, 'launch', 'coqui_survey.launch.py'), os.path.join(beh, 'config', 'ekf_odom.yaml'),
              os.path.join(beh, 'config', 'survey_base.yaml')]
    for p in files:
        txt = open(p, encoding='utf-8').read()
        for bad in ('/gps', 'NavSatFix', 'navsat_transform', 'sensor_msgs.msg import NavSat'):
            assert bad not in txt, f'{os.path.basename(p)} uses {bad}'
    assert sp.parse_geo_ref('') is None
    assert sp.parse_geo_ref('18.3,-65.8,1,2') == (18.3, -65.8, 1.0, 2.0, 0.0)
