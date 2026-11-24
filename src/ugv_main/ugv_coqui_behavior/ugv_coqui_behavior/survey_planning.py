"""Decision logic for the Coqui survey, free of ROS so it can be tested offline (test/test_survey_planning.py).

Everything here works on a `Grid` (a Nav2 costmap / OccupancyGrid: 0..100 cost, -1 unknown) and on
what the robot itself has measured (its poses, its acoustic rays). Nothing reads simulation truth.

The node turns each candidate list into Nav2 `ComputePathToPose` requests and only drives (via
`NavigateToPose`) to a candidate the planner can actually reach, so these functions only have to
rank candidates; reachability is decided by Nav2 (or by `GridPlanner` in the offline tests).
"""
import collections
import heapq
import math
from dataclasses import dataclass

import numpy as np


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


# --------------------------------------------------------------------------------------------- grid
@dataclass
class Grid:
    """Row-major costmap: data[row=y][col=x], cost 0..100, -1 = unknown."""
    origin_x: float
    origin_y: float
    resolution: float
    data: np.ndarray  # (height, width) int16

    @classmethod
    def from_msg(cls, msg):
        d = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        return cls(msg.info.origin.position.x, msg.info.origin.position.y, msg.info.resolution, d)

    @property
    def width(self):
        return self.data.shape[1]

    @property
    def height(self):
        return self.data.shape[0]

    def cell(self, x, y):
        return int(math.floor((x - self.origin_x) / self.resolution)), \
            int(math.floor((y - self.origin_y) / self.resolution))

    def inside(self, x, y):
        cx, cy = self.cell(x, y)
        return 0 <= cx < self.width and 0 <= cy < self.height

    def cost(self, x, y):
        """Cost at a point; -1 when unknown or outside the grid."""
        cx, cy = self.cell(x, y)
        if 0 <= cx < self.width and 0 <= cy < self.height:
            return int(self.data[cy, cx])
        return -1

    def max_cost_disc(self, x, y, radius):
        """Highest known cost inside a disc (-1 if every cell is unknown/outside)."""
        cx, cy = self.cell(x, y)
        r = max(0, int(math.ceil(radius / self.resolution)))
        x0, x1 = max(cx - r, 0), min(cx + r + 1, self.width)
        y0, y1 = max(cy - r, 0), min(cy + r + 1, self.height)
        if x0 >= x1 or y0 >= y1:
            return -1
        sub = self.data[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        vals = sub[disc]
        return int(vals.max()) if vals.size else -1

    def is_free(self, x, y, clearance, lethal, allow_unknown=True):
        if not allow_unknown and self.cost(x, y) < 0:
            return False
        return self.max_cost_disc(x, y, clearance) < lethal

    def line_blocked(self, a, b, lethal, skip_start=0.0, skip_end=0.0):
        """True if a lethal cell lies on segment a->b (ignoring the first/last metres given)."""
        ax, ay = a
        bx, by = b
        L = math.hypot(bx - ax, by - ay)
        if L <= skip_start + skip_end:
            return False
        n = max(2, int(L / (self.resolution * 0.5)))
        for t in np.linspace(skip_start / L, 1.0 - skip_end / L, n):
            if self.cost(ax + t * (bx - ax), ay + t * (by - ay)) >= lethal:
                return True
        return False

    def nearest_free(self, x, y, clearance, lethal, max_search=1.5, step=None):
        """Closest free point to (x, y) within max_search, spiralling out; None if none."""
        if self.is_free(x, y, clearance, lethal):
            return x, y
        step = step or max(self.resolution * 2, 0.1)
        r = step
        while r <= max_search + 1e-9:
            n = max(8, int(2 * math.pi * r / step))
            best = None
            for k in range(n):
                a = 2 * math.pi * k / n
                px, py = x + r * math.cos(a), y + r * math.sin(a)
                if self.is_free(px, py, clearance, lethal):
                    best = (px, py)
                    break
            if best:
                return best
            r += step
        return None


# ------------------------------------------------------------------------------------ known frogs
def bearing_points_at_known(robot_xy, bearing, known, radius, margin_deg=5.0, max_cone_deg=22.0):
    """Does an acoustic bearing from robot_xy point at an already-recorded frog?
    Returns that frog (dict with 'pos') or None. A bearing is attributed to a known frog when it passes
    within `radius` of it (plus an angular margin for DOA noise, capped at `max_cone_deg` beyond 2.5 m
    so one recorded frog cannot blind a wide sector of the survey area)."""
    for fr in known:
        fx, fy = fr['pos'][0], fr['pos'][1]
        d = math.hypot(fx - robot_xy[0], fy - robot_xy[1])
        if d < 1e-3:
            return fr
        tol = math.atan2(radius, d) + math.radians(margin_deg)
        if d >= 2.5 and max_cone_deg is not None:
            tol = min(tol, math.radians(max_cone_deg))
        if abs(wrap(math.atan2(fy - robot_xy[1], fx - robot_xy[0]) - bearing)) <= tol:
            return fr
    return None


def near_known(xy, known, radius):
    for fr in known:
        if math.hypot(fr['pos'][0] - xy[0], fr['pos'][1] - xy[1]) < radius:
            return fr
    return None


def in_frog_keepout(xy, known, keepout_m):
    """True if (x, y) is closer than `keepout_m` to any already-recorded frog."""
    if not known or keepout_m <= 0.0:
        return False
    return near_known(xy, known, keepout_m) is not None


def classify_height(frog_z, ground_z, elevated_threshold):
    """'elevated' (perched on a trunk/branch) or 'ground', from the triangulated height above the
    ground the robot is standing on."""
    return 'elevated' if (frog_z - ground_z) >= elevated_threshold else 'ground'


# --------------------------------------------------------------------------- listening stations
def coverage_cells(area, spacing):
    """Centres of a regular grid over the survey area [xmin, xmax, ymin, ymax]."""
    xmin, xmax, ymin, ymax = area
    xs = np.arange(xmin + spacing / 2, xmax, spacing)
    ys = np.arange(ymin + spacing / 2, ymax, spacing)
    return [(float(x), float(y)) for x in xs for y in ys]


def is_cell_covered(cell_xy, listened, cover_radius, moving_listened=None, moving_cover_radius=None):
    """True if `cell_xy` has been covered by a stationary listening stop (within `cover_radius`)
    or a close moving pass (within `moving_cover_radius`)."""
    cx, cy = cell_xy
    if any(math.hypot(cx - p[0], cy - p[1]) < cover_radius for p in listened):
        return True
    if moving_listened and moving_cover_radius and moving_cover_radius > 0.0:
        if any(math.hypot(cx - p[0], cy - p[1]) < moving_cover_radius for p in moving_listened):
            return True
    return False


def station_candidates(robot_xy, cells, listened, cover_radius, failed, grid, cfg,
                       known=None, moving_listened=None, moving_cover_radius=None):
    """Uncovered coverage cells, snapped to a free spot, nearest first.
    listened: positions where the robot has already listened (stations, hunt steps).
    failed:   goals Nav2 could not reach (skipped within failed_radius).
    known:    recorded frogs to keep clear of (frog_keepout_m)."""
    keepout = float(cfg.get('frog_keepout_m', 0.9))
    out = []
    for c in cells:
        if is_cell_covered(c, listened, cover_radius, moving_listened, moving_cover_radius):
            continue
        goal = c
        if grid is not None and grid.inside(*c):
            goal = grid.nearest_free(c[0], c[1], cfg['clearance_m'], cfg['lethal_cost'], cfg['snap_search_m'])
            if goal is None:
                continue
        if in_frog_keepout(goal, known, keepout):
            continue
        if any(math.hypot(goal[0] - f[0], goal[1] - f[1]) < cfg['failed_radius_m'] for f in failed):
            continue
        out.append((math.hypot(goal[0] - robot_xy[0], goal[1] - robot_xy[1]), goal, c))
    out.sort(key=lambda t: t[0])
    return [(g, c) for _, g, c in out]


# -------------------------------------------------------------------------- frontier exploration
def frontier_candidates(robot_xy, robot_yaw, area, grid, failed, cfg, weak_rays=None,
                        uncovered_xy=None, cover_radius=None, known=None):
    """Industry-standard frontier detection inside `area` [xmin, xmax, ymin, ymax]: clusters of
    known-free cells bordering unknown (-1) cells. Ranked lowest-cost first combining travel distance,
    cluster size (information gain), heading continuity (avoid oscillating), and optional Tier-1
    acoustic whisper rays [(origin_xy, bearing_rad, snr)] that pull exploration toward faint frogs.
    When `uncovered_xy` and `cover_radius` are given, frontiers that are not near any remaining
    uncovered survey cell (e.g. rolling-window/tree shadows in already-covered areas) are skipped.
    Returns [(score, (x, y), cluster_size)]."""
    if grid is None:
        return []
    H, W = grid.height, grid.width
    if H < 3 or W < 3:
        return []
    xmin, xmax, ymin, ymax = (float(v) for v in area)
    margin = float(cfg.get('area_margin_m', 1.0))
    if xmax - xmin > 2 * margin:
        xmin, xmax = xmin + margin, xmax - margin
    if ymax - ymin > 2 * margin:
        ymin, ymax = ymin + margin, ymax - margin
    c0x = max(1, int(math.floor((xmin - grid.origin_x) / grid.resolution)))
    c1x = min(W - 1, int(math.ceil((xmax - grid.origin_x) / grid.resolution)))
    c0y = max(1, int(math.floor((ymin - grid.origin_y) / grid.resolution)))
    c1y = min(H - 1, int(math.ceil((ymax - grid.origin_y) / grid.resolution)))
    if c0x >= c1x or c0y >= c1y:
        return []

    sub = grid.data[c0y:c1y, c0x:c1x]
    free = (sub >= 0) & (sub < cfg['lethal_cost'])
    unk = grid.data < 0
    has_unk = (unk[c0y - 1:c1y - 1, c0x:c1x] | unk[c0y + 1:c1y + 1, c0x:c1x]
               | unk[c0y:c1y, c0x - 1:c1x - 1] | unk[c0y:c1y, c0x + 1:c1x + 1])
    fmask = free & has_unk
    fy, fx = np.nonzero(fmask)
    if len(fx) == 0:
        return []

    # 8-connected clustering of frontier cells
    fset = set(zip(fx.tolist(), fy.tolist()))
    min_cells = int(cfg.get('min_frontier_cells', 4))
    min_dist = float(cfg.get('frontier_min_dist_m', 1.2))
    size_w = float(cfg.get('frontier_size_weight', 0.15))
    head_w = float(cfg.get('frontier_heading_penalty_m', 2.0))
    whisper_w = float(cfg.get('whisper_bias_m', 5.0))
    clearance = float(cfg['clearance_m'])
    lethal = int(cfg['lethal_cost'])
    failed_r = float(cfg['failed_radius_m'])

    clusters = []
    while fset:
        seed = fset.pop()
        q = collections.deque([seed])
        comp = [seed]
        while q:
            ux, uy = q.popleft()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (ux + dx, uy + dy)
                    if nb in fset:
                        fset.remove(nb)
                        q.append(nb)
                        comp.append(nb)
        if len(comp) >= min_cells:
            clusters.append(comp)

    out = []
    rx, ry = float(robot_xy[0]), float(robot_xy[1])
    for comp in clusters:
        mx = grid.origin_x + (c0x + sum(p[0] for p in comp) / len(comp) + 0.5) * grid.resolution
        my = grid.origin_y + (c0y + sum(p[1] for p in comp) / len(comp) + 0.5) * grid.resolution
        # pick the free frontier cell closest to the centroid (fall back to nearest_free around centroid)
        best_pt = None
        best_d2 = 1e18
        for px_i, py_i in comp:
            wx = grid.origin_x + (c0x + px_i + 0.5) * grid.resolution
            wy = grid.origin_y + (c0y + py_i + 0.5) * grid.resolution
            d2 = (wx - mx) ** 2 + (wy - my) ** 2
            if d2 < best_d2 and grid.is_free(wx, wy, clearance, lethal, allow_unknown=False):
                best_d2, best_pt = d2, (wx, wy)
        if best_pt is None:
            best_pt = grid.nearest_free(mx, my, clearance, lethal, max_search=1.0)
            if best_pt is None or grid.cost(*best_pt) < 0:
                continue
        gx, gy = best_pt
        if in_frog_keepout((gx, gy), known, float(cfg.get('frog_keepout_m', 0.9))):
            continue
        if uncovered_xy is not None and cover_radius is not None:
            if not any(math.hypot(gx - ux, gy - uy) <= cover_radius for ux, uy in uncovered_xy):
                continue
        dist = math.hypot(gx - rx, gy - ry)
        if dist < min_dist:
            continue
        if any(math.hypot(gx - f[0], gy - f[1]) < failed_r for f in failed):
            continue
        heading_err = abs(wrap(math.atan2(gy - ry, gx - rx) - float(robot_yaw))) / math.pi
        terrain_w = float(cfg.get('terrain_cost_weight', 0.08))
        cell_cost = max(0, grid.cost(gx, gy))
        score = dist - size_w * min(len(comp), 40) + head_w * heading_err + terrain_w * cell_cost
        if weak_rays:
            for (ox, oy), br, _ in weak_rays:
                vx, vy = gx - ox, gy - oy
                if math.hypot(vx, vy) > 0.5 and abs(wrap(math.atan2(vy, vx) - br)) < math.radians(25.0):
                    score -= whisper_w
                    break
        out.append((score, (float(gx), float(gy)), len(comp)))
    out.sort(key=lambda t: t[0])
    return out


def frontier_still_unknown(goal_xy, grid, radius_m=1.2):
    """True if there are still unknown (-1) cells within `radius_m` of a frontier goal."""
    if grid is None or not grid.inside(*goal_xy):
        return False
    cx, cy = grid.cell(*goal_xy)
    r = max(1, int(math.ceil(radius_m / grid.resolution)))
    x0, x1 = max(cx - r, 0), min(cx + r + 1, grid.width)
    y0, y1 = max(cy - r, 0), min(cy + r + 1, grid.height)
    sub = grid.data[y0:y1, x0:x1]
    return bool(np.any(sub < 0))


def check_passive_triangulation(passive_obs, known, cfg):
    """Check whether passive bearings [(t, origin xyz, dir xyz, snr)] collected while exploring
    already cross-triangulate an unrecorded frog. Anchors on the newest bearing.
    Returns (tri_dict, anchor_obs) or (None, None)."""
    if len(passive_obs) < 4:
        return None, None
    anchor_obs = passive_obs[-1]
    a_o, a_d = anchor_obs[1], anchor_obs[2]
    a_bearing = math.atan2(float(a_d[1]), float(a_d[0]))
    if bearing_points_at_known((float(a_o[0]), float(a_o[1])), a_bearing, known, cfg['known_frog_radius_m']):
        return None, None
    times = [o[0] for o in passive_obs]
    origins = [o[1] for o in passive_obs]
    dirs = [o[2] for o in passive_obs]
    groups = bout_ids(times, cfg.get('bout_gap_s', 1.2))
    tri = robust_triangulate(origins, dirs, (a_o, a_d), cfg, groups=groups)
    if tri is None:
        return None, None
    if tri['n_origins'] < 2 or tri['parallax_deg'] < 12.0 or tri['inliers'] < 4:
        return None, None
    if near_known(tri['pos'], known, cfg['known_frog_radius_m']):
        return None, None
    return tri, anchor_obs


# ------------------------------------------------------------------------------ triangulation
def parallax_deg(frog_xy, origins, cand_xy):
    """Smallest angle (deg) at the frog between the candidate and any existing ray origin: how much
    new geometry a bearing from cand_xy adds to the triangulation."""
    if not origins:
        return 90.0
    a = math.atan2(cand_xy[1] - frog_xy[1], cand_xy[0] - frog_xy[0])
    best = 180.0
    for o in origins:
        b = math.atan2(o[1] - frog_xy[1], o[0] - frog_xy[0])
        best = min(best, abs(math.degrees(wrap(a - b))))
    return min(best, 90.0)


def baseline_step_candidates(robot_xy, frog_xy, origins, grid, cfg, allow_unknown=False, known=None):
    """Places to move to for the next bearing, best first: free, outside the frog keep-out (and any
    recorded frog's keep-out), adding parallax, and (while the frog is still far) getting closer.
    Returns [(score, (x, y))].
    allow_unknown=False only offers ground already seen to be free (steps toward a frog in a bush
    otherwise lead under the bush edge before the sensors have mapped it, and the robot gets boxed in)."""
    d_now = math.hypot(frog_xy[0] - robot_xy[0], frog_xy[1] - robot_xy[1])
    if cfg.get('adaptive_steps', False):
        r_min = max(0.45, min(cfg['step_min_m'], 0.15 * d_now))
        r_max = max(cfg['step_max_m'], min(3.0, 0.35 * d_now))
        radii = np.linspace(r_min, r_max, 4)
    else:
        radii = np.linspace(cfg['step_min_m'], cfg['step_max_m'], 3)
    area = cfg.get('survey_area')
    margin = float(cfg.get('area_margin_m', 1.2))
    frog_ko = float(cfg.get('frog_keepout_m', 0.9))
    out = []
    for r in radii:
        for k in range(cfg['step_angles']):
            a = 2 * math.pi * k / cfg['step_angles']
            c = (robot_xy[0] + r * math.cos(a), robot_xy[1] + r * math.sin(a))
            if area is not None:
                xmin, xmax, ymin, ymax = (float(v) for v in area)
                if not (xmin + margin <= c[0] <= xmax - margin and ymin + margin <= c[1] <= ymax - margin):
                    continue
            d_c = math.hypot(frog_xy[0] - c[0], frog_xy[1] - c[1])
            if d_c < cfg['keepout_m']:
                continue
            if in_frog_keepout(c, known, frog_ko):
                continue
            if grid is not None and not grid.is_free(c[0], c[1], cfg['clearance_m'], cfg['lethal_cost'],
                                                     allow_unknown=allow_unknown):
                continue
            score = parallax_deg(frog_xy, origins, c)
            if d_now > cfg['approach_until_m']:
                score += cfg['approach_gain_deg_per_m'] * (d_now - d_c)
            score -= cfg['step_cost_deg_per_m'] * r
            if grid is not None:
                score -= 0.20 * max(0, grid.cost(c[0], c[1]))
            out.append((score, c))
    out.sort(key=lambda t: -t[0])
    return out


def triangulation_done(tri, cfg):
    """Convergence test on a robust_triangulate() result."""
    return (tri is not None
            and tri['inliers'] >= cfg['min_rays']
            and tri['n_origins'] >= cfg.get('min_places', 3)
            and tri['baseline_m'] >= cfg['min_baseline_m']
            and tri['parallax_deg'] >= cfg['min_parallax_deg']
            and tri['uncertainty_m'] <= cfg['max_uncertainty_m'])


def accept_partial_triangulation(tri, cfg):
    """Stricter acceptance gate when a hunt ends early (frog went quiet, step limit, or robot stuck).

    Keeps genuinely tight fixes (e.g. 4 cm error from 2+ places with low RMS and uncertainty <=
    max_uncertainty_m) while rejecting loose/unbounded fixes (e.g. 4.8 m off at 8 deg parallax).
    """
    if tri is None:
        return False
    min_inliers = max(4, int(cfg.get('min_rays', 6)) - 2)
    min_par = float(cfg.get('partial_min_parallax_deg', 12.0))
    max_unc = float(cfg['max_uncertainty_m'])
    max_rms = float(cfg.get('ray_inlier_deg', 2.0))
    return (tri['n_origins'] >= 2
            and tri['inliers'] >= min_inliers
            and tri['baseline_m'] >= 0.7 * float(cfg.get('min_baseline_m', 1.0))
            and tri['parallax_deg'] >= min_par
            and tri['uncertainty_m'] <= max_unc
            and tri['rms_deg'] <= max_rms)


def should_accept_record(tri, visually_confirmed, vantage, los_clear, grid, cfg):
    """Decide whether a photographed frog estimate should be saved into the survey record.

    Prevents recording a false frog (e.g. 5 m off in open ground with an empty photo) which would
    otherwise poison `known` and blind bearings in a wide cone around the false position.
    Returns (accept: bool, reason: str).
    """
    if visually_confirmed:
        return True, 'visually_confirmed'
    if tri is None or 'pos' not in tri:
        return False, 'no_triangulation'
    est = tri['pos']
    reject_open = bool(cfg.get('reject_empty_open_photo', True))
    open_ground = False
    if grid is not None and grid.inside(est[0], est[1]):
        cell_cost = grid.cost(est[0], est[1])
        open_ground = (0 <= cell_cost < int(cfg['lethal_cost']))
    if reject_open and vantage == 'reached' and los_clear and open_ground:
        return False, 'photo_contradicts_open_ground'
    # Occluded or vantage unreachable: require solid acoustic convergence
    if triangulation_done(tri, cfg) or accept_partial_triangulation(tri, cfg):
        return True, 'acoustic_only_occluded'
    return False, 'weak_unconfirmed_fix'


def _angles_to(x, origins, dirs):
    """Horizontal angle between each ray direction and the direction from its origin to x (rad), and
    the horizontal range to x."""
    v = x[None, :2] - origins[:, :2]
    rng = np.linalg.norm(v, axis=1)
    a_ray = np.arctan2(dirs[:, 1], dirs[:, 0])
    a_x = np.arctan2(v[:, 1], v[:, 0])
    return np.abs(np.arctan2(np.sin(a_ray - a_x), np.cos(a_ray - a_x))), rng


def bout_ids(times, gap_s):
    """Split bearings into calling bouts: a new bout starts after a silence longer than gap_s."""
    ids, b = [], 0
    for i, t in enumerate(times):
        if i and t - times[i - 1] > gap_s:
            b += 1
        ids.append(b)
    return np.asarray(ids)


def _inlier_mask(ang, r, tol, max_r, groups):
    ok = (r <= max_r) & (r > 0.2)
    if groups is None:
        return (ang <= tol) & ok
    m = np.zeros(len(ang), bool)
    for g in np.unique(groups):
        sel = groups == g
        if np.median(ang[sel]) <= tol:          # the bout as a whole points at x
            m |= sel & (ang <= 2.5 * tol) & ok
    return m


def robust_triangulate(origins, dirs, anchor, cfg, rng_seed=0, groups=None):
    """Locate ONE frog from bearings that may include calls of other frogs.

    origins (N,3), dirs (N,3 unit): every acoustic bearing heard during the hunt (map frame).
    groups (N,) optional calling-bout ids (bout_ids): a bout is one frog, so it is accepted or rejected
            as a whole by its median angle (a neighbouring frog's bout a few degrees off is dropped).
    anchor: (origin, dir) of the call that started the hunt; the frog must lie along it (within
            cfg['anchor_tol_deg']), which is what ties the hunt to that frog.
    Hypotheses are pairwise intersections of rays from different places; the one most bearings agree
    with (within cfg['ray_inlier_deg']) wins and is refined by least squares on its inliers. Height is
    the median of the inliers' elevation rays at their range.
    Returns None until bearings from two places intersect along the anchor, else a dict:
      pos [x,y,z], inliers, n_origins, baseline_m, parallax_deg, rms_deg, uncertainty_m, inlier_mask."""
    origins = np.asarray(origins, float)
    dirs = np.asarray(dirs, float)
    n = len(origins)
    if n < 2:
        return None
    tol = math.radians(cfg['ray_inlier_deg'])
    a_o, a_d = np.asarray(anchor[0], float), np.asarray(anchor[1], float)
    a_tol = math.radians(cfg['anchor_tol_deg'])
    max_r = cfg['max_range_m']
    rng = np.random.default_rng(rng_seed)
    # candidate pairs: different places (>= 0.3 m apart), non-parallel
    I, J = np.triu_indices(n, 1)
    sep = np.linalg.norm(origins[I, :2] - origins[J, :2], axis=1)
    keep = sep >= 0.3
    I, J, sep = I[keep], J[keep], sep[keep]
    if len(I) == 0:
        return None
    if groups is not None:   # pairs from different bouts, or same bout once robot moved >= 1.0 m
        keep = (np.asarray(groups)[I] != np.asarray(groups)[J]) | (sep >= 1.0)
        I, J = I[keep], J[keep]
        if len(I) == 0:
            return None
    if len(I) > 600:
        sel = rng.choice(len(I), 600, replace=False)
        I, J = I[sel], J[sel]
    def _on_anchor(pt):
        va = pt - a_o[:2]
        dist = float(np.linalg.norm(va))
        if dist > max_r or float(np.dot(va, a_d[:2])) <= 0:
            return False
        return abs(wrap(math.atan2(va[1], va[0]) - math.atan2(a_d[1], a_d[0]))) <= a_tol + math.atan2(0.5, dist)

    scored = []
    for i, j in zip(I, J):
        x = _intersect2d(origins[i], dirs[i], origins[j], dirs[j])
        if x is None:
            continue
        # in front of both rays and within audible range
        if np.dot(x - origins[i, :2], dirs[i, :2]) <= 0 or np.dot(x - origins[j, :2], dirs[j, :2]) <= 0:
            continue
        # on the anchor bearing
        if not _on_anchor(x):
            continue
        ang, r = _angles_to(np.r_[x, 0.0], origins, dirs)
        inl = _inlier_mask(ang, r, tol, max_r, groups)
        # MSAC score: squared angle error, capped at the tolerance (lower is better). Unlike counting
        # inliers it prefers the point most bearings agree with tightly over one that many bearings
        # (including a neighbouring frog's) agree with loosely.
        cost = float(np.sum(np.minimum(np.where(r <= max_r, ang, tol), tol) ** 2))
        if inl.sum() >= 2:
            scored.append((cost, x, inl))
    if not scored:
        return None
    # Deduplicate top pairwise seeds (>= 0.5 m apart) and refine each while enforcing the anchor
    # constraint: a distant collinear frog's rays can pass near the anchor and score well before
    # refinement, only to pull the least-squares solution off (or behind) the anchor bearing.
    scored.sort(key=lambda t: t[0])
    seeds = []
    for cost, x0, inl0 in scored:
        if all(float(np.linalg.norm(x0 - s[1])) > 0.5 for s in seeds):
            seeds.append((cost, x0, inl0))
            if len(seeds) >= 8:
                break
    best = None
    for _, x0, inl0 in seeds:
        x, inl = x0, inl0
        ok = True
        for _ in range(3):   # refine on inliers (angular least squares), re-select inliers
            x = _refine_angular(_ls2d(origins[inl], dirs[inl]), origins[inl], dirs[inl])
            if not _on_anchor(x):
                ok = False
                break
            ang, r = _angles_to(np.r_[x, 0.0], origins, dirs)
            inl = _inlier_mask(ang, r, tol, max_r, groups)
            if inl.sum() < 2:
                ok = False
                break
        if not ok:
            continue
        rcost = float(np.sum(np.minimum(np.where(r <= max_r, ang, tol), tol) ** 2))
        if best is None or rcost < best[0]:
            best = (rcost, x, inl, ang)
    if best is None:
        return None
    _, x, inl, ang = best
    o_in, d_in = origins[inl], dirs[inl]
    hr = np.linalg.norm(x[None, :] - o_in[:, :2], axis=1)
    horiz = np.linalg.norm(d_in[:, :2], axis=1)
    z_rays = o_in[:, 2] + hr * d_in[:, 2] / np.maximum(horiz, 1e-6)
    near = hr <= max(10.0, 1.5 * float(np.min(hr)))
    z = float(np.median(z_rays[near]))
    # distinct listening places among inliers
    places = []
    for o in o_in:
        if all(math.hypot(o[0] - p[0], o[1] - p[1]) > 0.3 for p in places):
            places.append(o)
    base = max((math.hypot(a[0] - b[0], a[1] - b[1]) for a in places for b in places), default=0.0)
    par = 0.0
    for a in places:
        for b in places:
            ua = math.atan2(a[1] - x[1], a[0] - x[0])
            ub = math.atan2(b[1] - x[1], b[0] - x[0])
            par = max(par, abs(math.degrees(wrap(ua - ub))))
    rms = float(np.degrees(np.sqrt(np.mean(ang[inl] ** 2))))
    mean_r = float(np.mean(hr))
    sigma = math.radians(max(rms, cfg['doa_sigma_deg']))
    # along-range error of two bearings sigma apart seen with parallax `par` (deliberately conservative:
    # no averaging credit, DOA errors of one listening place are correlated) plus optional pose sigma floor
    pose_sigma = float(cfg.get('pose_sigma_m', 0.05))
    unc_ang = mean_r * sigma / max(math.sin(math.radians(min(par, 90.0))), math.sin(math.radians(3.0)))
    unc = math.hypot(unc_ang, pose_sigma)
    return {'pos': [float(x[0]), float(x[1]), z], 'inliers': int(inl.sum()), 'n_obs': int(n),
            'n_origins': len(places), 'baseline_m': round(base, 3), 'parallax_deg': round(par, 1),
            'rms_deg': round(rms, 2), 'uncertainty_m': round(unc, 3), 'inlier_mask': inl}


def _intersect2d(o1, d1, o2, d2):
    a = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], float)
    det = np.linalg.det(a)
    if abs(det) < math.sin(math.radians(2.0)) * np.linalg.norm(d1[:2]) * np.linalg.norm(d2[:2]):
        return None
    t = np.linalg.solve(a, np.array([o2[0] - o1[0], o2[1] - o1[1]], float))
    return np.array([o1[0] + t[0] * d1[0], o1[1] + t[0] * d1[1]])


def _refine_angular(x, origins, dirs, iters=8):
    """Gauss-Newton on bearing-angle residuals (the quantity the microphones actually measure).
    Perpendicular-distance least squares (_ls2d) is biased toward the listeners when parallax is small;
    this removes most of that bias."""
    a_meas = np.arctan2(dirs[:, 1], dirs[:, 0])
    x = np.array(x, float)
    for _ in range(iters):
        v = x[None, :] - origins[:, :2]
        d2 = np.maximum((v ** 2).sum(1), 1e-6)
        res = np.arctan2(np.sin(np.arctan2(v[:, 1], v[:, 0]) - a_meas), np.cos(np.arctan2(v[:, 1], v[:, 0]) - a_meas))
        J = np.stack([-v[:, 1] / d2, v[:, 0] / d2], axis=1)
        H = J.T @ J + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, -J.T @ res)
        x = x + step
        if np.linalg.norm(step) < 1e-4:
            break
    return x


def _ls2d(origins, dirs):
    A = np.zeros((2, 2))
    b = np.zeros(2)
    for o, d in zip(origins, dirs):
        dd = d[:2] / max(np.linalg.norm(d[:2]), 1e-9)
        P = np.eye(2) - np.outer(dd, dd)
        A += P
        b += P @ o[:2]
    return np.linalg.solve(A + 1e-9 * np.eye(2), b)


# ------------------------------------------------------------------------------- vantage point
def vantage_candidates(frog_xyz, robot_xy, kind, grid, cfg, allow_unknown=False, known=None):
    """Camera viewpoints around a frog, best first: [(score, (x, y, yaw_to_frog), standoff, los_clear)].
    `kind` is 'ground' or 'elevated' (from classify_height). Line of sight is checked on the costmap
    (lethal cells between viewpoint and frog block the view of a frog on the ground); for an elevated
    frog low obstacles do not block, so only the trunk itself (skipped) is near it.
    allow_unknown=False keeps only viewpoints the robot has already seen to be free: next to a bush the
    unseen cells usually turn out to be bush once the robot gets close."""
    fx, fy = frog_xyz[0], frog_xyz[1]
    standoff = cfg['standoff_elevated_m'] if kind == 'elevated' else cfg['standoff_ground_m']
    area = cfg.get('survey_area')
    margin = float(cfg.get('area_margin_m', 1.0))
    frog_ko = float(cfg.get('frog_keepout_m', 0.9))
    out = []
    for extra in cfg['vantage_extra_radii_m']:
        r = standoff + extra
        for k in range(cfg['vantage_angles']):
            a = 2 * math.pi * k / cfg['vantage_angles']
            vx, vy = fx + r * math.cos(a), fy + r * math.sin(a)
            if area is not None:
                xmin, xmax, ymin, ymax = (float(v) for v in area)
                if not (xmin + margin <= vx <= xmax - margin and ymin + margin <= vy <= ymax - margin):
                    continue
            if in_frog_keepout((vx, vy), known, frog_ko):
                continue
            if grid is not None and not grid.is_free(vx, vy, cfg['clearance_m'], cfg['lethal_cost'],
                                                     allow_unknown=allow_unknown):
                continue
            los = True
            if grid is not None and kind == 'ground':
                los = not grid.line_blocked((vx, vy), (fx, fy), cfg['lethal_cost'],
                                            skip_start=cfg['clearance_m'], skip_end=cfg['los_skip_end_m'])
            yaw = math.atan2(fy - vy, fx - vx)
            terrain_w = float(cfg.get('terrain_cost_weight', 0.08))
            cell_cost = max(0, grid.cost(vx, vy)) if grid is not None else 0
            score = (0.0 if los else cfg['no_los_penalty']) + cfg['extra_radius_penalty'] * extra \
                + 0.3 * math.hypot(vx - robot_xy[0], vy - robot_xy[1]) + terrain_w * cell_cost
            out.append((score, (vx, vy, yaw), r, los))
    out.sort(key=lambda t: t[0])
    return out


# ------------------------------------------------------------------------- staged return-home
def next_home_waypoints(robot_xy, home_xy, breadcrumbs, grid, cfg, failed=None):
    """Compute intermediate return-home waypoints inside the current rolling costmap window.

    When `home_xy` lies outside the rolling `global_costmap` window (e.g. 57 m away on the hill),
    Nav2 cannot plan directly to `home_xy`. This returns an ordered list of reachable in-window
    staging candidates [(x, y), ...] stepping along previously visited `breadcrumbs` (`good_spots`)
    that make progress toward home, plus direct rolling-window projections toward `home_xy`.
    """
    rx, ry = float(robot_xy[0]), float(robot_xy[1])
    hx, hy = float(home_xy[0]), float(home_xy[1])
    failed = failed or []
    failed_r = float(cfg.get('failed_radius_m', 1.0))
    clearance = float(cfg['clearance_m'])
    lethal = int(cfg['lethal_cost'])

    def _inside_margin(px, py, pad=1.0):
        if grid is None:
            return True
        x0 = grid.origin_x + pad
        x1 = grid.origin_x + grid.width * grid.resolution - pad
        y0 = grid.origin_y + pad
        y1 = grid.origin_y + grid.height * grid.resolution - pad
        return x0 <= px <= x1 and y0 <= py <= y1

    if _inside_margin(hx, hy, pad=0.5):
        return [(hx, hy)]

    d_home = math.hypot(hx - rx, hy - ry)
    cands = []
    # 1) Prefer recorded breadcrumbs (known-drivable trail!) that lie inside the rolling window and
    #    advance substantially toward home, farthest-along-route (closest to home) first.
    if breadcrumbs:
        bc_scored = []
        for bx, by in breadcrumbs:
            bx, by = float(bx), float(by)
            if not _inside_margin(bx, by, pad=1.0):
                continue
            d_step = math.hypot(bx - rx, by - ry)
            d_rem = math.hypot(hx - bx, hy - by)
            if d_step < 1.5 or d_rem >= d_home - 1.5:
                continue
            if any(math.hypot(bx - fx, by - fy) < failed_r for fx, fy in failed):
                continue
            pt = (bx, by)
            if grid is not None:
                snapped = grid.nearest_free(bx, by, clearance, lethal, max_search=2.0)
                if snapped is None:
                    continue
                pt = snapped
            bc_scored.append((d_rem, pt))
        bc_scored.sort(key=lambda t: t[0])
        for _, pt in bc_scored:
            if all(math.hypot(pt[0] - c[0], pt[1] - c[1]) > 1.5 for c in cands):
                cands.append(pt)
                if len(cands) >= 6:
                    break

    # 2) Projection onto the rolling-window rim along robot -> home (and slight +/- angle offsets)
    if grid is not None:
        max_r = max(2.0, 0.38 * min(grid.width, grid.height) * grid.resolution)
    else:
        max_r = 15.0
    step_r = min(d_home, max_r)
    base_ang = math.atan2(hy - ry, hx - rx)
    for da_deg in (0.0, 20.0, -20.0, 40.0, -40.0):
        a = base_ang + math.radians(da_deg)
        px, py = rx + step_r * math.cos(a), ry + step_r * math.sin(a)
        pt = (px, py)
        if grid is not None:
            snapped = grid.nearest_free(px, py, clearance, lethal, max_search=3.5)
            if snapped is None:
                continue
            pt = snapped
        if any(math.hypot(pt[0] - fx, pt[1] - fy) < failed_r for fx, fy in failed):
            continue
        if all(math.hypot(pt[0] - c[0], pt[1] - c[1]) > 1.0 for c in cands):
            cands.append(pt)
    return cands


# ------------------------------------------------------------------------- offline path planner
class GridPlanner:
    """8-connected A* on a Grid (lethal cells inflated by `clearance`), standing in for Nav2's planner
    in offline tests. plan() returns the path length in metres or None."""

    def __init__(self, grid, clearance, lethal, allow_unknown=True):
        self.g = grid
        blocked = grid.data >= lethal
        if not allow_unknown:
            blocked |= grid.data < 0
        r = int(math.ceil(clearance / grid.resolution))
        infl = blocked.copy()
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        for dy, dx in zip(*np.nonzero(xx ** 2 + yy ** 2 <= r * r)):
            dy, dx = dy - r, dx - r
            infl |= np.roll(np.roll(blocked, dy, 0), dx, 1)
        self.blocked = infl

    def plan(self, start, goal):
        g = self.g
        s, t = g.cell(*start), g.cell(*goal)
        H, W = self.blocked.shape
        if not (0 <= t[0] < W and 0 <= t[1] < H) or self.blocked[t[1], t[0]]:
            return None
        if not (0 <= s[0] < W and 0 <= s[1] < H):
            return None
        openq = [(0.0, 0.0, s)]
        best = {s: 0.0}
        steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        while openq:
            _, cost, c = heapq.heappop(openq)
            if c == t:
                return cost * g.resolution
            if cost > best.get(c, 1e18):
                continue
            for dx, dy, w in steps:
                n = (c[0] + dx, c[1] + dy)
                if not (0 <= n[0] < W and 0 <= n[1] < H) or self.blocked[n[1], n[0]]:
                    continue
                nc = cost + w
                if nc < best.get(n, 1e18):
                    best[n] = nc
                    heapq.heappush(openq, (nc + math.hypot(t[0] - n[0], t[1] - n[1]), nc, n))
        return None


def path_length(points):
    return float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:])))


def path_terrain_cost(points, grid, slope_weight=3.0):
    """Terrain-weighted path length (metres equivalent): flat ground (cost 0) weighs 1.0x, while
    sloped/rough cells weigh up to (1 + slope_weight)x so easier low-slope routes beat steep detours."""
    if grid is None or slope_weight <= 0.0:
        return path_length(points)
    total = 0.0
    for a, b in zip(points, points[1:]):
        ds = math.hypot(b[0] - a[0], b[1] - a[1])
        c = grid.cost(0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
        mult = 1.0 + (slope_weight * c / 100.0 if c > 0 else 0.0)
        total += ds * mult
    return float(total)


# ----------------------------------------------------------------------------------- geodesy
EARTH_R = 6378137.0


def local_to_latlon(ref_lat, ref_lon, ref_xy, xy, heading_deg=0.0):
    """Map point -> WGS84, from one reference pair (map xy where the GPS read ref_lat/lon).
    Assumes map x = east, y = north rotated by heading_deg (0 in Gazebo ENU worlds). Good to cm over a
    few hundred metres."""
    dx, dy = xy[0] - ref_xy[0], xy[1] - ref_xy[1]
    h = math.radians(heading_deg)
    e = dx * math.cos(h) - dy * math.sin(h)
    n = dx * math.sin(h) + dy * math.cos(h)
    lat = ref_lat + math.degrees(n / EARTH_R)
    lon = ref_lon + math.degrees(e / (EARTH_R * math.cos(math.radians(ref_lat))))
    return lat, lon
