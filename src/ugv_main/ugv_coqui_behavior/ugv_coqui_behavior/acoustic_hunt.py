"""Hunting one calling frog among several, free of ROS so it can be tested offline.

HuntTracker   one hunt = one frog. Decides, from the bearings heard so far, whether to keep listening here,
              move to another listening place, or stop. Two rules keep the robot from wandering between two
              calling frogs:
                * it never moves until the TARGET has been heard from the current place. Other frogs' calls are
                  kept (they help the next hunt) but they never count as progress on this one;
                * it remembers every place it has listened from, so it never shuttles between two spots.
LeadBook      frogs that were heard but not located. A hunt that ends without a record leaves a lead, the mission
              goes back to leads before it calls the area covered, and leads that stay unresolved are reported
              instead of being forgotten or recorded at a wrong place.

Bearings are tuples (t, origin xyz, unit direction xyz, loudness dB), in the map frame. Loudness is the received call
level with cfg['loudness_ref_db'] = a nominal call heard 1 m away (see survey_planning).
"""
import math

import numpy as np

from . import survey_planning as sp


def _bearing(d):
    return math.atan2(float(d[1]), float(d[0]))


def _xy(o):
    return float(o[0]), float(o[1])


def same_caller(obs, anchor, cfg):
    """Could bearing `obs` be the frog that was heard on bearing `anchor` (both (t, o, d, loudness))?
    From the same place: same direction and loudness. From another place: the two bearings must cross in front of both
    places, at a distance each call's loudness agrees with."""
    o, d, loud = obs[1], obs[2], float(obs[3])
    a_o, a_d, a_loud = anchor[1], anchor[2], float(anchor[3])
    spread = float(cfg.get('loudness_spread_db', 5.0))
    tol = math.radians(float(cfg['anchor_tol_deg']))
    d_b = abs(sp.wrap(_bearing(d) - _bearing(a_d)))
    sep = math.hypot(float(o[0]) - float(a_o[0]), float(o[1]) - float(a_o[1]))
    if sep < 0.3:
        return d_b <= tol and abs(loud - a_loud) <= spread
    x = sp._intersect2d(np.asarray(a_o, float), np.asarray(a_d, float), np.asarray(o, float), np.asarray(d, float))
    if x is None or np.linalg.norm(x - np.asarray(a_o[:2], float)) > float(cfg['max_range_m']):
        # (nearly) parallel bearings cross too far away to tell: the frog is far, so it must sound the same
        return d_b <= tol and abs(loud - a_loud) <= spread
    va, vo = x - np.asarray(a_o[:2], float), x - np.asarray(o[:2], float)
    if float(va @ np.asarray(a_d[:2], float)) <= 0 or float(vo @ np.asarray(d[:2], float)) <= 0:
        # they diverge: the same frog only if they are parallel within the bearing noise (a step straight towards it)
        return d_b <= 2.0 * math.radians(float(cfg['ray_inlier_deg'])) and abs(loud - a_loud) <= spread
    ra, ro = float(np.linalg.norm(va)), float(np.linalg.norm(vo))
    if min(ra, ro) < float(cfg.get('min_fix_range_m', 0.6)):
        return False
    sa, so = sp.source_level_db(a_loud, ra), sp.source_level_db(loud, ro)
    return abs(sa - so) <= spread and \
        abs(0.5 * (sa + so) - float(cfg.get('loudness_ref_db', 0.0))) <= float(cfg.get('loudness_gate_db', 10.0))


class HuntTracker:
    """The state of one hunt. Feed it bearings with add(), tell it when the robot has stopped somewhere with at_place(),
    and ask decide() what to do."""

    def __init__(self, anchor, cfg, now, known=(), seed=()):
        self.cfg = cfg
        self.anchor = tuple(anchor)
        self.known = list(known)
        self.t0 = float(now)
        self.obs = sorted([tuple(o) for o in seed], key=lambda o: o[0])
        if not any(abs(o[0] - anchor[0]) < 1e-3 for o in self.obs):
            self.obs.append(self.anchor)
        self.places = []              # every place listened from in this hunt, in order
        self.place_xy = None
        self.place_t0 = float(now)
        self.steps = 0
        self.tri = None
        self._tri_n = -1

    # ----------------------------------------------------------------------------------- input
    def add(self, obs):
        self.obs.append(tuple(obs))

    def at_place(self, xy, now):
        """The robot has stopped at xy and starts listening."""
        self.place_xy = (float(xy[0]), float(xy[1]))
        self.place_t0 = float(now)
        if all(math.hypot(self.place_xy[0] - p[0], self.place_xy[1] - p[1]) > 0.3 for p in self.places):
            self.places.append(self.place_xy)

    def refine(self, new, robot_xy):
        """At a viewpoint next to a finished fix: use those bearings of `new` that are the frog there, and only if the fix
        then stays where it is (a neighbour must not take it over). Returns the ones used."""
        before = self.triangulate()
        if before is None:
            return []
        used = [o for o in sp.heard_from_here(new, robot_xy, before['pos'], self.cfg)
                if before['source_level_db'] is None or abs(sp.source_level_db(o[3], math.hypot(
                    before['pos'][0] - robot_xy[0], before['pos'][1] - robot_xy[1])) - before['source_level_db'])
                <= float(self.cfg.get('loudness_spread_db', 5.0)) + 1.0]
        if not used:
            return []
        obs = self.obs
        self.obs = obs + [tuple(o) for o in used]
        after = self.triangulate()
        if after is None or math.hypot(after['pos'][0] - before['pos'][0], after['pos'][1] - before['pos'][1]) > 1.0:
            self.obs, self.tri, self._tri_n = obs, before, len(obs)
        return used

    # -------------------------------------------------------------------------------- estimate
    def triangulate(self):
        if len(self.obs) != self._tri_n:
            self._tri_n = len(self.obs)
            origins = [o[1] for o in self.obs]
            dirs = [o[2] for o in self.obs]
            loud = [o[3] for o in self.obs]
            self.tri = sp.robust_triangulate(
                origins, dirs, (self.anchor[1], self.anchor[2]), self.cfg,
                groups=sp.ray_groups(origins, dirs, loud, link_deg=float(self.cfg.get('group_link_deg', 3.0))),
                loudness=loud, anchor_loudness=self.anchor[3])
        return self.tri

    def estimate_xy(self):
        """Where the target is believed to be: the fix, or along the anchor bearing as far as its loudness says."""
        tri = self.triangulate()
        if tri is not None:
            return tri['pos'][0], tri['pos'][1]
        r = min(max(sp.implied_range_m(self.anchor[3], self.cfg), 2.0), float(self.cfg.get('nominal_range_max_m', 12.0)))
        b = _bearing(self.anchor[2])
        return float(self.anchor[1][0]) + r * math.cos(b), float(self.anchor[1][1]) + r * math.sin(b)

    def on_target(self, obs):
        """Is this bearing a call of the frog being hunted?"""
        tri = self.triangulate()
        if tri is not None:
            v = np.asarray(tri['pos'][:2]) - np.asarray(obs[1][:2], float)
            r = float(np.linalg.norm(v))
            slack = min(math.atan2(min(tri['uncertainty_m'], 2.0), max(r, 0.5)), math.radians(12.0))
            if abs(sp.wrap(math.atan2(v[1], v[0]) - _bearing(obs[2]))) <= math.radians(2.0 * float(self.cfg['ray_inlier_deg'])) + slack \
                    and abs(sp.source_level_db(obs[3], r) - tri['source_level_db']) <= float(self.cfg['loudness_spread_db']):
                return True
            if sp.triangulation_done(tri, self.cfg):
                return False
        return same_caller(obs, self.anchor, self.cfg)

    def target_here(self):
        """Bearings of the target heard since the robot stopped at the current place. The first place also counts the
        calls that started the hunt."""
        t_from = self.place_t0 - (4.0 if len(self.places) <= 1 else 0.0)
        return [o for o in self.obs if o[0] >= t_from and
                (self.place_xy is None or math.hypot(float(o[1][0]) - self.place_xy[0],
                                                     float(o[1][1]) - self.place_xy[1]) < 0.5)
                and self.on_target(o)]

    # -------------------------------------------------------------------------------- decision
    def finish(self, reason):
        """Outcome of a hunt that has to stop because of `reason`: still 'converged (...)' when the fix is good enough
        to be worth a photo."""
        return f'converged ({reason})' if sp.accept_partial_triangulation(self.triangulate(), self.cfg) else reason

    def decide(self, now):
        """(action, why): 'listen' (stay and wait for the target), 'step' (go to another listening place) or 'end'.
        For 'end', `why` is the outcome; outcomes that start with 'converged' are worth a photo."""
        c = self.cfg
        tri = self.triangulate()
        if tri is not None and tri['n_origins'] >= 2 and \
                sp.near_known(tri['pos'], self.known, float(c['known_frog_radius_m'])):
            return 'end', 'already recorded frog'
        if sp.triangulation_done(tri, c):
            return 'end', 'converged'

        if now - self.t0 > float(c['hunt_timeout_sec']):
            return 'end', self.finish('hunt time limit')
        here = len(self.target_here())
        waited = now - self.place_t0
        if here >= int(c['hunt_min_new_rays']):
            if self.steps >= int(c['max_steps']):
                return 'end', self.finish('step limit')
            return 'step', f'{here} calls of the target from here'
        if waited < float(c['hunt_listen_max_sec']):
            return 'listen', f'{here} of {int(c["hunt_min_new_rays"])} calls of the target from here'
        if here >= 1 and self.steps < int(c['max_steps']):
            return 'step', 'target calls rarely: one bearing from here has to do'
        return 'end', self.finish('target went quiet')

    def step_origins(self):
        """Listening places a new one must add parallax to: where the target was heard, and everywhere this hunt has
        already stood (so the best next step is never a place it just left)."""
        out = list(self.places)
        tri = self.triangulate()
        for p in (tri['places'] if tri is not None else []):
            if all(math.hypot(p[0] - q[0], p[1] - q[1]) > 0.3 for q in out):
                out.append(p)
        return out

    def step_candidates(self, robot_xy, grid, cfg=None, allow_unknown=False, avoid=()):
        """Reachability-unchecked next listening places, best first [(score, (x, y))]. Never within
        min_step_separation_m of a place this hunt has already listened from."""
        c = cfg or self.cfg
        sep = float(c.get('min_step_separation_m', 0.7))
        cands = sp.baseline_step_candidates(robot_xy, self.estimate_xy(), self.step_origins(), grid, c,
                                            allow_unknown=allow_unknown, known=self.known)
        return [(s, p) for s, p in cands
                if all(math.hypot(p[0] - q[0], p[1] - q[1]) >= sep for q in self.places)
                and all(math.hypot(p[0] - q[0], p[1] - q[1]) > 0.6 for q in avoid)]

    def leftover(self):
        """Bearings this hunt did not use: other frogs' calls (or all of them when nothing was located)."""
        tri = self.triangulate()
        if tri is None:
            return list(self.obs)
        return [o for o, m in zip(self.obs, tri['inlier_mask']) if not m]


def pick_anchor(heard, rough, cfg):
    """The clean standing bearing that is the frog of a rough (moving or blended) one, or None. `heard`: bearings heard
    since the robot stopped, a metre or so further on. A rough bearing is good to ~15 degrees, and the two are too close
    together to cross usefully, so it takes a similar direction and a similar loudness; the best match of both wins. A
    call that sounds out of hunting range is never the one that stopped the robot."""
    spread = float(cfg.get('loudness_spread_db', 5.0))
    r = max(sp.implied_range_m(rough[3], cfg), 1.5)
    far = 1.3 * float(cfg.get('hunt_trigger_range_m', 12.0))
    best = None
    for o in heard:
        sep = math.hypot(float(o[1][0]) - float(rough[1][0]), float(o[1][1]) - float(rough[1][1]))
        tol = math.radians(max(15.0, float(cfg['anchor_tol_deg']))) + min(math.atan2(sep, r), math.radians(45.0))
        d_b, d_l = abs(sp.wrap(_bearing(o[2]) - _bearing(rough[2]))), abs(float(o[3]) - float(rough[3]))
        if d_b > tol or d_l > spread or sp.implied_range_m(o[3], cfg) > far:
            continue
        misfit = (d_b / (0.4 * tol)) ** 2 + (d_l / (0.5 * spread)) ** 2
        if best is None or misfit < best[0]:
            best = (misfit, o)
    return None if best is None else best[1]


def coverage_places(places, recorded):
    """Listening places of a hunt that count as survey coverage. A hunt that did not end in a record has not covered
    its area: the frog that started it is still out there."""
    return list(places) if recorded else []


class LeadBook:
    """Frogs heard but not located."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.leads = []
        self._n = 0

    def _tol(self, lead, robot_xy):
        d = math.hypot(lead['pos'][0] - robot_xy[0], lead['pos'][1] - robot_xy[1])
        return math.atan2(lead['spread_m'], max(d, 1.0)) + math.radians(6.0), d

    def match(self, origin, bearing, loudness):
        """The lead a heard call belongs to, or None."""
        for ld in self.leads:
            tol, d = self._tol(ld, _xy(origin))
            b = math.atan2(ld['pos'][1] - float(origin[1]), ld['pos'][0] - float(origin[0]))
            if abs(sp.wrap(b - bearing)) > min(tol, math.radians(20.0)):
                continue
            r = sp.implied_range_m(loudness, self.cfg)
            if d < 1.0 or 0.4 <= r / max(d, 0.5) <= 2.5:
                return ld
        return None

    def note(self, anchor, estimate_xy, spread_m, now, reason):
        """A hunt ended without a record: remember the frog (or count another try on a lead it already was)."""
        x, y = float(estimate_xy[0]), float(estimate_xy[1])
        spread = float(min(max(spread_m, 0.5), 4.0))
        for ld in self.leads:
            if math.hypot(ld['pos'][0] - x, ld['pos'][1] - y) <= max(2.0, 0.7 * max(spread, ld['spread_m'])):
                ld['attempts'] += 1
                ld['t'] = float(now)
                ld['reason'] = reason
                ld['origins'].append(_xy(anchor[1]))
                if spread < ld['spread_m']:
                    ld['pos'], ld['spread_m'] = (x, y), spread
                return ld
        self._n += 1
        ld = {'id': f'lead_{self._n:02d}', 'pos': (x, y), 'spread_m': spread, 'attempts': 1, 't': float(now),
              'reason': reason, 'origins': [_xy(anchor[1])], 'bearing_deg': round(math.degrees(_bearing(anchor[2])), 1),
              'loudness_db': round(float(anchor[3]), 1)}
        self.leads.append(ld)
        return ld

    def resolve_near(self, pos, radius):
        """A frog was recorded at pos: drop the leads it explains."""
        gone = [ld for ld in self.leads
                if math.hypot(ld['pos'][0] - pos[0], ld['pos'][1] - pos[1]) <= radius + 0.5 * ld['spread_m']]
        self.leads = [ld for ld in self.leads if ld not in gone]
        return gone

    def exhausted(self, ld):
        return ld['attempts'] >= int(self.cfg.get('lead_max_attempts', 3))

    def due(self, robot_xy, now, max_dist=None):
        """Leads worth another try now, nearest first."""
        wait = float(self.cfg.get('lead_retry_sec', 120.0))
        out = [ld for ld in self.leads if not self.exhausted(ld) and now - ld['t'] >= wait]
        d = {ld['id']: math.hypot(ld['pos'][0] - robot_xy[0], ld['pos'][1] - robot_xy[1]) for ld in out}
        if max_dist is not None:
            out = [ld for ld in out if d[ld['id']] <= max_dist]
        return sorted(out, key=lambda ld: d[ld['id']])

    def listening_places(self, ld, robot_xy, grid, cfg, known=()):
        """Free places 3..6.5 m from a lead to listen from, best first: new directions, short drives."""
        out = []
        area = cfg.get('survey_area')
        for r in (3.5, 5.0, 6.5):
            for k in range(16):
                a = 2 * math.pi * k / 16
                p = (ld['pos'][0] + r * math.cos(a), ld['pos'][1] + r * math.sin(a))
                if area is not None and not (area[0] + 1 <= p[0] <= area[1] - 1 and area[2] + 1 <= p[1] <= area[3] - 1):
                    continue
                if grid is not None and not grid.is_free(p[0], p[1], cfg['clearance_m'], cfg['lethal_cost'],
                                                         allow_unknown=False):
                    continue
                if sp.in_frog_keepout(p, list(known), float(cfg.get('frog_keepout_m', 0.9))):
                    continue
                score = sp.parallax_deg(ld['pos'], ld['origins'], p) \
                    - 6.0 * math.hypot(p[0] - robot_xy[0], p[1] - robot_xy[1])
                out.append((score, p))
        out.sort(key=lambda t: -t[0])
        return [p for _, p in out]

    def records(self):
        """For survey.yaml: what was heard but never located."""
        return [{'id': ld['id'], 'approx_pos_xy': [round(ld['pos'][0], 2), round(ld['pos'][1], 2)],
                 'spread_m': round(ld['spread_m'], 2), 'attempts': ld['attempts'], 'why': ld['reason'],
                 'heard_from_xy': [[round(v, 2) for v in o] for o in ld['origins']]} for ld in self.leads]
