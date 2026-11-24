"""Closed-loop hunts without ROS or Gazebo: a chorus of frogs, a robot with realistic ears and pose error, and the real
HuntTracker / LeadBook / triangulation code. Used by test_acoustic_hunt.py."""
import math

import numpy as np

from ugv_audio import audio_core as ac
from ugv_coqui_behavior import acoustic_hunt as ah
from ugv_coqui_behavior import survey_planning as sp

FOREST = {'coqui_01': (37.5, -9.5), 'coqui_02': (39.35, 8.5), 'coqui_03': (40.5, -2.5), 'coqui_04': (35.5, -2.6),
          'coqui_05': (47.4, -9.2), 'coqui_06': (24.0, -6.0)}
COVER = {'coqui_01': 1.0, 'coqui_02': 0.9, 'coqui_03': 0.78, 'coqui_04': 0.85, 'coqui_05': 0.94, 'coqui_06': 1.0}


def open_world(center, size=60.0, res=0.1):
    n = int(size / res)
    return sp.Grid(center[0] - size / 2, center[1] - size / 2, res, np.zeros((n, n), np.int16))


def put_disc(g, x, y, r, val=100):
    yy, xx = np.mgrid[0:g.height, 0:g.width]
    gx, gy = g.origin_x + (xx + 0.5) * g.resolution, g.origin_y + (yy + 0.5) * g.resolution
    g.data[(gx - x) ** 2 + (gy - y) ** 2 <= r * r] = val


class Ears:
    """What the microphone array reports: (t, origin, direction, loudness) of every clean call."""

    def __init__(self, frogs, seed, bearing_sd_deg=0.6, drift_m=0.0, cover=None, overlap_prob=0.25):
        self.names = list(frogs)
        self.pos = np.array([frogs[k] for k in self.names], float)
        self.rng = np.random.default_rng([seed, 10])
        self.chorus = ac.ChorusSchedule(len(frogs), np.random.default_rng([seed, 1]), overlap_prob=overlap_prob)
        self.var = ac.CallVariation(np.random.default_rng([seed, 2]), len(frogs),
                                    [1.0 if cover is None else cover.get(k, 1.0) for k in self.names])
        self.sd = math.radians(bearing_sd_deg)
        self.bias = np.random.default_rng([seed, 3]).normal(0, math.radians(0.4), 2)   # placement of the microphones
        self.drift_m = drift_m      # position error gathered per 100 m driven
        self.err = np.zeros(2)

    def drive(self, dist):
        self.err = self.err + self.rng.normal(0, self.drift_m * math.sqrt(dist / 100.0), 2)

    def listen(self, t, xy):
        """Observations of one audio frame from true position xy."""
        calls = [(i, self.var.gain(i, self.pos[i], xy)) for i, _ in self.chorus.step(t)]
        out = []
        for i, g in calls:
            d = float(np.hypot(*(self.pos[i] - xy)))
            lv = 20 * math.log10(g) - 20 * math.log10(max(d, 0.35))
            if d < 30 and lv > -36:
                out.append((lv, i, d))
        if not out or (len(out) > 1 and sorted(out)[-1][0] - sorted(out)[-2][0] < 8):   # a blend is rejected (coherence)
            return []
        lv, i, d = max(out)
        b = math.atan2(*(self.pos[i] - xy)[::-1])
        b += self.rng.normal(0, self.sd) + self.bias[0] * math.cos(b) + self.bias[1] * math.sin(b)
        o = np.array([xy[0] + self.err[0], xy[1] + self.err[1], 0.115])
        return [((t, o, np.array([math.cos(b), math.sin(b), 0.0]), lv), self.names[i])]


def _seen(grid, places, r):
    """The part of the map a robot has seen from `places` (r metres around each); the rest is unknown."""
    g = sp.Grid(grid.origin_x, grid.origin_y, grid.resolution, np.full_like(grid.data, -1))
    yy, xx = np.mgrid[0:g.height, 0:g.width]
    gx, gy = g.origin_x + (xx + 0.5) * g.resolution, g.origin_y + (yy + 0.5) * g.resolution
    m = np.zeros_like(g.data, bool)
    for x, y in places:
        m |= (gx - x) ** 2 + (gy - y) ** 2 <= r * r
    g.data[m] = grid.data[m]
    return g


def hunt(ears, start, grid, cfg, t=0.0, target=None, known=(), speed=0.35, seed_obs=(), seen_m=None):
    """One hunt, like the node's loop. Returns a dict with what happened. target: name of the frog to anchor on
    (None: the first frog that sounds near enough, as the node does). seen_m: the robot only knows the map this far around
    the places it has been (as in the forest), so it cannot step anywhere it likes."""
    xy = np.array(start, float)
    anchor, banked = None, list(seed_obs)
    t_end = t + 120
    while anchor is None and t < t_end:
        t += 0.25
        for ob, who in ears.listen(t, xy):
            if (target is None and sp.call_is_near(ob[3], cfg)) or who == target:
                anchor, anchor_frog = ob, who
            else:
                banked.append(ob)
    if anchor is None:
        return {'outcome': 'never heard', 't': t}
    trk = ah.HuntTracker(anchor, cfg, t, known=known, seed=banked)
    trk.at_place(xy + ears.err, t)
    log, failed, plan_fail, driven, t0 = [], [], 0, 0.0, t
    while True:
        action, why = trk.decide(t)
        if action == 'end':
            break
        if action == 'listen':
            t += 0.25
            for ob, _ in ears.listen(t, xy):
                trk.add(ob)
            continue
        here = len(trk.target_here())
        g = grid if seen_m is None else _seen(grid, trk.places + [tuple(xy + ears.err)], seen_m)
        cands = trk.step_candidates(tuple(xy + ears.err), g, avoid=failed)
        goal = next((c for _, c in cands if not grid.line_blocked(tuple(xy + ears.err), c, cfg['lethal_cost'])), None)
        if goal is None and cands:
            goal = cands[0][1]
        trk.steps += 1
        if goal is None:
            plan_fail += 1
            if plan_fail >= 3:
                why = trk.finish('no reachable step')
                break
            trk.at_place(xy + ears.err, t)
            continue
        new = np.array(goal) - ears.err
        d = float(np.hypot(*(new - xy)))
        ears.drive(d)
        driven += d
        for _ in range(int(d / speed / 0.25)):     # calls on the way are not used (moving)
            t += 0.25
            ears.chorus.step(t)
        xy = new
        trk.at_place(xy + ears.err, t)
        log.append({'heard_target_before_step': here, 'place': tuple(xy)})
    tri = trk.triangulate()
    r = {'outcome': why, 'anchor_frog': anchor_frog, 't': t, 'duration': t - t0, 'steps': trk.steps, 'driven': driven,
         'log': log, 'places': trk.places, 'tri': tri, 'tracker': trk, 'xy': xy, 'err_m': None, 'nearest': None}
    if tri is not None:
        e = {k: float(np.hypot(p[0] - tri['pos'][0], p[1] - tri['pos'][1])) for k, p in zip(ears.names, ears.pos)}
        r['nearest'] = min(e, key=e.get)
        r['err_m'] = e[r['nearest']]
        r['err_target_m'] = e[anchor_frog]
    return r
