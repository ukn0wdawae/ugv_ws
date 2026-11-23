"""Decision logic for the Coqui survey, free of ROS so it can be tested offline (test/test_survey_planning.py).

Everything here works on a `Grid` (a Nav2 costmap / OccupancyGrid: 0..100 cost, -1 unknown) and on
what the robot itself has measured (its poses, its acoustic rays). Nothing reads simulation truth.

The node turns each candidate list into Nav2 `ComputePathToPose` requests and only drives (via
`NavigateToPose`) to a candidate the planner can actually reach, so these functions only have to
rank candidates; reachability is decided by Nav2 (or by `GridPlanner` in the offline tests).
"""
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
def bearing_points_at_known(robot_xy, bearing, known, radius, margin_deg=8.0):
    """Does an acoustic bearing from robot_xy point at an already-recorded frog?
    Returns that frog (dict with 'pos') or None. A bearing is attributed to a known frog when it passes
    within `radius` of it (plus an angular margin for DOA noise)."""
    for fr in known:
        fx, fy = fr['pos'][0], fr['pos'][1]
        d = math.hypot(fx - robot_xy[0], fy - robot_xy[1])
        if d < 1e-3:
            return fr
        tol = math.atan2(radius, d) + math.radians(margin_deg)
        if abs(wrap(math.atan2(fy - robot_xy[1], fx - robot_xy[0]) - bearing)) <= tol:
            return fr
    return None


def near_known(xy, known, radius):
    for fr in known:
        if math.hypot(fr['pos'][0] - xy[0], fr['pos'][1] - xy[1]) < radius:
            return fr
    return None


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


def station_candidates(robot_xy, cells, listened, cover_radius, failed, grid, cfg):
    """Uncovered coverage cells, snapped to a free spot, nearest first.
    listened: positions where the robot has already listened (stations, hunt steps).
    failed:   goals Nav2 could not reach (skipped within failed_radius)."""
    out = []
    for c in cells:
        if any(math.hypot(c[0] - p[0], c[1] - p[1]) < cover_radius for p in listened):
            continue
        goal = c
        if grid is not None and grid.inside(*c):
            goal = grid.nearest_free(c[0], c[1], cfg['clearance_m'], cfg['lethal_cost'], cfg['snap_search_m'])
            if goal is None:
                continue
        if any(math.hypot(goal[0] - f[0], goal[1] - f[1]) < cfg['failed_radius_m'] for f in failed):
            continue
        out.append((math.hypot(goal[0] - robot_xy[0], goal[1] - robot_xy[1]), goal, c))
    out.sort(key=lambda t: t[0])
    return [(g, c) for _, g, c in out]


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


def baseline_step_candidates(robot_xy, frog_xy, origins, grid, cfg, allow_unknown=False):
    """Places to move to for the next bearing, best first: free, outside the frog keep-out, adding
    parallax, and (while the frog is still far) getting closer. Returns [(score, (x, y))].
    allow_unknown=False only offers ground already seen to be free (steps toward a frog in a bush
    otherwise lead under the bush edge before the sensors have mapped it, and the robot gets boxed in)."""
    d_now = math.hypot(frog_xy[0] - robot_xy[0], frog_xy[1] - robot_xy[1])
    radii = np.linspace(cfg['step_min_m'], cfg['step_max_m'], 3)
    out = []
    for r in radii:
        for k in range(cfg['step_angles']):
            a = 2 * math.pi * k / cfg['step_angles']
            c = (robot_xy[0] + r * math.cos(a), robot_xy[1] + r * math.sin(a))
            d_c = math.hypot(frog_xy[0] - c[0], frog_xy[1] - c[1])
            if d_c < cfg['keepout_m']:
                continue
            if grid is not None and not grid.is_free(c[0], c[1], cfg['clearance_m'], cfg['lethal_cost'],
                                                     allow_unknown=allow_unknown):
                continue
            score = parallax_deg(frog_xy, origins, c)
            if d_now > cfg['approach_until_m']:
                score += cfg['approach_gain_deg_per_m'] * (d_now - d_c)
            score -= cfg['step_cost_deg_per_m'] * r
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
    I, J = I[keep], J[keep]
    if len(I) == 0:
        return None
    if groups is not None:   # pairs from different bouts only
        keep = np.asarray(groups)[I] != np.asarray(groups)[J]
        I, J = I[keep], J[keep]
        if len(I) == 0:
            return None
    if len(I) > 600:
        sel = rng.choice(len(I), 600, replace=False)
        I, J = I[sel], J[sel]
    best = None
    for i, j in zip(I, J):
        x = _intersect2d(origins[i], dirs[i], origins[j], dirs[j])
        if x is None:
            continue
        # in front of both rays and within audible range
        if np.dot(x - origins[i, :2], dirs[i, :2]) <= 0 or np.dot(x - origins[j, :2], dirs[j, :2]) <= 0:
            continue
        # on the anchor bearing
        va = x - a_o[:2]
        if np.linalg.norm(va) > max_r or np.dot(va, a_d[:2]) <= 0:
            continue
        if abs(wrap(math.atan2(va[1], va[0]) - math.atan2(a_d[1], a_d[0]))) > a_tol + math.atan2(0.5, np.linalg.norm(va)):
            continue
        ang, r = _angles_to(np.r_[x, 0.0], origins, dirs)
        inl = _inlier_mask(ang, r, tol, max_r, groups)
        # MSAC score: squared angle error, capped at the tolerance (lower is better). Unlike counting
        # inliers it prefers the point most bearings agree with tightly over one that many bearings
        # (including a neighbouring frog's) agree with loosely.
        cost = float(np.sum(np.minimum(np.where(r <= max_r, ang, tol), tol) ** 2))
        if inl.sum() >= 2 and (best is None or cost < best[0]):
            best = (cost, x, inl)
    if best is None:
        return None
    _, x, inl = best
    for _ in range(3):   # refine on inliers (angular least squares), re-select inliers
        x = _refine_angular(_ls2d(origins[inl], dirs[inl]), origins[inl], dirs[inl])
        ang, r = _angles_to(np.r_[x, 0.0], origins, dirs)
        inl = _inlier_mask(ang, r, tol, max_r, groups)
        if inl.sum() < 2:
            return None
    o_in, d_in = origins[inl], dirs[inl]
    hr = np.linalg.norm(x[None, :] - o_in[:, :2], axis=1)
    horiz = np.linalg.norm(d_in[:, :2], axis=1)
    z = float(np.median(o_in[:, 2] + hr * d_in[:, 2] / np.maximum(horiz, 1e-6)))
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
    # no averaging credit, DOA errors of one listening place are correlated)
    unc = mean_r * sigma / max(math.sin(math.radians(min(par, 90.0))), math.sin(math.radians(3.0)))
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
def vantage_candidates(frog_xyz, robot_xy, kind, grid, cfg, allow_unknown=False):
    """Camera viewpoints around a frog, best first: [(score, (x, y, yaw_to_frog), standoff, los_clear)].
    `kind` is 'ground' or 'elevated' (from classify_height). Line of sight is checked on the costmap
    (lethal cells between viewpoint and frog block the view of a frog on the ground); for an elevated
    frog low obstacles do not block, so only the trunk itself (skipped) is near it.
    allow_unknown=False keeps only viewpoints the robot has already seen to be free: next to a bush the
    unseen cells usually turn out to be bush once the robot gets close."""
    fx, fy = frog_xyz[0], frog_xyz[1]
    standoff = cfg['standoff_elevated_m'] if kind == 'elevated' else cfg['standoff_ground_m']
    out = []
    for extra in cfg['vantage_extra_radii_m']:
        r = standoff + extra
        for k in range(cfg['vantage_angles']):
            a = 2 * math.pi * k / cfg['vantage_angles']
            vx, vy = fx + r * math.cos(a), fy + r * math.sin(a)
            if grid is not None and not grid.is_free(vx, vy, cfg['clearance_m'], cfg['lethal_cost'],
                                                     allow_unknown=allow_unknown):
                continue
            los = True
            if grid is not None and kind == 'ground':
                los = not grid.line_blocked((vx, vy), (fx, fy), cfg['lethal_cost'],
                                            skip_start=cfg['clearance_m'], skip_end=cfg['los_skip_end_m'])
            yaw = math.atan2(fy - vy, fx - vx)
            score = (0.0 if los else cfg['no_los_penalty']) + cfg['extra_radius_penalty'] * extra \
                + 0.3 * math.hypot(vx - robot_xy[0], vy - robot_xy[1])
            out.append((score, (vx, vy, yaw), r, los))
    out.sort(key=lambda t: t[0])
    return out


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
