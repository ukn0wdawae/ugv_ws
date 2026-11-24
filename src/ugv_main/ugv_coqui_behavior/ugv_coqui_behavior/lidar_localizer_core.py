"""GPS-denied localization in a forest: scan-to-map matching on vertical structure (numpy + OpenCV, free of ROS so it can be
tested offline, test/test_lidar_localizer.py).

Wheels slip and the gyro drifts; there is no GPS under the canopy. Trunks, bushes and grass edges do not move, so the 3D
lidar can tell the robot where it is relative to everything it has already seen:

  1. the scan is levelled with the odometry's roll and pitch and reduced to points on VERTICAL structure. Ground returns are
     dropped: the rings a lidar paints on the ground move with the robot, so registering them says "nothing moved"
     (that is what a plain ICP did here: 16 m of error);
  2. the structure points are matched to a 2D map of all structure seen so far, searching a small window around the
     odometry's prediction; the result moves the map -> odom correction (x, y, yaw) part of the way;
  3. ground points do the same for the height;
  4. every ~0.3 m the scan is added to the map.

The map keeps everything, so coming back to a place corrects the drift gathered since.
"""
import math

import cv2
import numpy as np

DEFAULTS = {
    'resolution_m': 0.05,
    'initial_size_m': 80.0,        # the map grows by itself when the robot nears its edge
    'blur_sigma_m': 0.12,          # how far off a point may be and still agree with the map
    'min_range_m': 0.8,
    'max_range_m': 18.0,           # beyond this the rings are too far apart to see a trunk twice
    'steep_min_dz_m': 0.05,
    'min_points': 40,
    'search_m': 0.3,
    'search_deg': 2.0,
    'lost_search_m': 1.0,          # wider search after lost_after scans without a match
    'lost_search_deg': 5.0,
    'lost_after': 6,
    'min_score': 0.15,
    'gain_xy': 0.5,
    'gain_yaw': 0.2,
    'gain_z': 0.2,
    'insert_every_m': 0.3,
    'insert_every_deg': 8.0,
    'still_m': 0.03,               # when standing still a few matches are enough; more would only follow noise
    'still_deg': 0.5,
    'still_matches': 3,
    'ground_cell_m': 0.5,
    'shell_fraction': 0.3,         # a scan with this many returns at one and the same range is a sensor fault
}


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw_of(R):
    return math.atan2(R[1, 0], R[0, 0])


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def structure_mask(pts_level, ranges, cfg=DEFAULTS):
    """Which points of a levelled scan (N, 3, robot at the origin, z up) lie on vertical structure: neighbouring rings of the
    same azimuth that are steeper than 45 degrees. Terrain up to 45 degrees never qualifies, whatever its height."""
    n = len(pts_level)
    if n < 2:
        return np.zeros(n, bool)
    az = np.floor(np.degrees(np.arctan2(pts_level[:, 1], pts_level[:, 0])) + 180.0).astype(int) % 360
    hr = np.hypot(pts_level[:, 0], pts_level[:, 1])
    el = np.arctan2(pts_level[:, 2], hr)
    o = np.lexsort((el, az))
    h, z, b = hr[o], pts_level[o, 2], az[o]
    dz = z[1:] - z[:-1]
    steep = (b[1:] == b[:-1]) & (dz > cfg['steep_min_dz_m']) & (np.abs(h[1:] - h[:-1]) < dz)
    m = np.zeros(n, bool)
    m[1:] |= steep
    m[:-1] |= steep
    out = np.zeros(n, bool)
    out[o] = m
    return out & (ranges > cfg['min_range_m']) & (ranges < cfg['max_range_m'])


def has_range_shell(ranges, cfg=DEFAULTS):
    """A scan whose returns pile up at one range all around is not the world (a clipping fault, a dirty window)."""
    r = ranges[(ranges > 2.0) & (ranges < cfg['max_range_m'])]
    if len(r) < 200:
        return False
    hist, _ = np.histogram(r, bins=np.arange(2.0, cfg['max_range_m'] + 0.1, 0.1))
    return float((hist[:-1] + hist[1:]).max()) / max(len(ranges), 1) > cfg['shell_fraction']


class StructureLocalizer:
    """Keeps the map -> odom correction. Poses: `T_odom_base` 4x4 from odometry (x, y, z, full attitude)."""

    def __init__(self, start_xyz_yaw=(0.0, 0.0, 0.0, 0.0), params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.res = float(self.p['resolution_m'])
        self.C = np.array(start_xyz_yaw, float)      # map <- odom: x, y, z, yaw
        n = int(self.p['initial_size_m'] / self.res)
        self.origin = None                           # map xy of the lower-left corner
        self._n0 = n
        self.hits = self.field = None
        self.sig = float(self.p['blur_sigma_m']) / self.res
        self.ground = {}                             # (ix, iy) of ground_cell_m cells -> [z, weight]
        self.last_insert = None
        self.last_match_odom = None
        self.still_count = 0
        self.misses = 0
        self.status = {}

    # ------------------------------------------------------------------------------------ poses
    def to_map(self, T_odom_base):
        """Robot pose in the map (x, y, z, yaw) for an odometry pose."""
        x, y, z = T_odom_base[:3, 3]
        c, s = math.cos(self.C[3]), math.sin(self.C[3])
        return (c * x - s * y + self.C[0], s * x + c * y + self.C[1], z + self.C[2],
                wrap(yaw_of(T_odom_base[:3, :3]) + self.C[3]))

    def _set_pose(self, T_odom_base, x, y, yaw):
        """Choose the correction that puts the odometry pose at (x, y, yaw) in the map."""
        self.C[3] = wrap(yaw - yaw_of(T_odom_base[:3, :3]))
        c, s = math.cos(self.C[3]), math.sin(self.C[3])
        ox, oy = T_odom_base[0, 3], T_odom_base[1, 3]
        self.C[0], self.C[1] = x - (c * ox - s * oy), y - (s * ox + c * oy)

    # -------------------------------------------------------------------------------------- map
    def _ensure(self, x, y):
        if self.origin is None:
            half = self._n0 * self.res / 2
            self.origin = np.array([x - half, y - half])
            self.hits = np.zeros((self._n0, self._n0), np.float32)
            self.field = np.zeros((self._n0, self._n0), np.float32)
        margin = float(self.p['max_range_m']) + 5.0
        pad = int(40.0 / self.res)
        h, w = self.hits.shape
        l = pad if x - self.origin[0] < margin else 0
        b = pad if y - self.origin[1] < margin else 0
        r = pad if self.origin[0] + w * self.res - x < margin else 0
        t = pad if self.origin[1] + h * self.res - y < margin else 0
        if l or b or r or t:
            self.hits = np.pad(self.hits, ((b, t), (l, r)))
            self.field = np.pad(self.field, ((b, t), (l, r)))
            self.origin = self.origin - np.array([l, b]) * self.res

    def _idx(self, xy):
        return np.floor((xy - self.origin) / self.res).astype(np.int64)

    def _insert(self, xy):
        i = self._idx(xy)
        h, w = self.hits.shape
        i = np.unique(i[(i[:, 0] >= 0) & (i[:, 0] < w) & (i[:, 1] >= 0) & (i[:, 1] < h)], axis=0)
        if not len(i):
            return
        self.hits[i[:, 1], i[:, 0]] = np.minimum(self.hits[i[:, 1], i[:, 0]] + 1.0, 6.0)
        k = int(3 * self.sig) + 2
        x0, y0 = np.maximum(i.min(0) - 2 * k, 0)
        x1, y1 = np.minimum(i.max(0) + 2 * k + 1, [w, h])
        g = cv2.GaussianBlur(np.minimum(self.hits[y0:y1, x0:x1], 3.0) / 3.0, (0, 0), self.sig)
        g = np.minimum(g * (math.pi * self.sig ** 2), 1.0)
        sl = np.s_[k:g.shape[0] - k, k:g.shape[1] - k]
        self.field[y0 + k:y1 - k, x0 + k:x1 - k] = g[sl]

    def _scores(self, xy, shifts):
        i = self._idx(xy)
        h, w = self.field.shape
        out = np.zeros(len(shifts))
        for k, (dx, dy) in enumerate(shifts):
            jx, jy = i[:, 0] + dx, i[:, 1] + dy
            ok = (jx >= 0) & (jx < w) & (jy >= 0) & (jy < h)
            out[k] = float(self.field[jy[ok], jx[ok]].sum()) / len(i)
        return out

    def _match(self, pts, x, y, yaw, win_m, win_rad):
        """(score, dx, dy, dyaw): coarse then fine correlative search around (x, y, yaw)."""
        best = (-1.0, 0.0, 0.0, 0.0)
        levels = ((2, win_m, win_rad, math.radians(0.5)), (1, 0.1, math.radians(0.5), math.radians(0.125)))
        for lvl, (step, rad, yr, ys) in enumerate(levels):
            k = int(round(rad / (step * self.res)))
            shifts = [(i * step, j * step) for i in range(-k, k + 1) for j in range(-k, k + 1)]
            bx, by, bt = (0.0, 0.0, 0.0) if lvl == 0 else best[1:]
            cand = (-1.0, bx, by, bt)
            for a in np.arange(-yr, yr + 1e-9, ys):
                t = yaw + bt + a
                R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
                sc = self._scores(pts @ R.T + [x + bx, y + by], shifts)
                m = int(sc.argmax())
                # ties go to the smaller correction
                s = sc[m] - 0.02 * self.res * (abs(shifts[m][0]) + abs(shifts[m][1])) - 0.02 * abs(a)
                if s > cand[0]:
                    cand = (float(s), bx + shifts[m][0] * self.res, by + shifts[m][1] * self.res, bt + float(a))
            best = cand
        return best

    # ----------------------------------------------------------------------------------- update
    def process(self, pts_base, T_odom_base):
        """One lidar scan: points (N, 3) in the robot base frame and the odometry pose at its stamp.
        Updates the correction and returns the status dict."""
        pts_base = np.asarray(pts_base, float)
        pts_base = pts_base[np.isfinite(pts_base).all(1)]
        rng = np.linalg.norm(pts_base, axis=1)
        st = {'points': 0, 'matched': False, 'score': None, 'shift_m': 0.0, 'fault': False}
        self.status = st
        if has_range_shell(rng, self.p):
            st['fault'] = True
            return st
        R = T_odom_base[:3, :3]
        lvl = pts_base @ (rot_z(-yaw_of(R)) @ R).T      # heading kept as x, roll and pitch removed
        sm = structure_mask(lvl, rng, self.p)
        pts = lvl[sm, :2]
        if len(pts):
            pts = np.unique(np.round(pts / 0.1).astype(int), axis=0) * 0.1     # even out near and far structure
        st['points'] = len(pts)
        x, y, z, yaw = self.to_map(T_odom_base)
        self._ensure(x, y)
        if self.last_insert is not None and len(pts) >= self.p['min_points'] and self._should_match(T_odom_base):
            lost = self.misses >= int(self.p['lost_after'])
            win = float(self.p['lost_search_m' if lost else 'search_m'])
            deg = float(self.p['lost_search_deg' if lost else 'search_deg'])
            score, dx, dy, dth = self._match(pts, x, y, yaw, win, math.radians(deg))
            c, s = math.cos(yaw), math.sin(yaw)
            here = self._scores(pts @ np.array([[c, -s], [s, c]]).T + [x, y], [(0, 0)])[0]
            st['score'] = round(score, 3)
            if score >= float(self.p['min_score']) * (2.0 if lost else 1.0) and score >= here:
                gx, gt = (1.0, 1.0) if lost else (float(self.p['gain_xy']), float(self.p['gain_yaw']))
                x, y, yaw = x + gx * dx, y + gx * dy, wrap(yaw + gt * dth)
                self._set_pose(T_odom_base, x, y, yaw)
                st['matched'], st['shift_m'] = True, round(math.hypot(dx, dy), 3)
                self.misses = 0
            else:
                self.misses += 1
        self._height(lvl, rng, sm, x, y, yaw, T_odom_base)
        li = self.last_insert
        if len(pts) >= 20 and (li is None or math.hypot(x - li[0], y - li[1]) > self.p['insert_every_m']
                               or abs(wrap(yaw - li[2])) > math.radians(self.p['insert_every_deg'])) \
                and self.misses < int(self.p['lost_after']):
            c, s = math.cos(yaw), math.sin(yaw)
            self._insert(pts @ np.array([[c, -s], [s, c]]).T + [x, y])
            self.last_insert = (x, y, yaw)
        return st

    def _should_match(self, T):
        o = (T[0, 3], T[1, 3], yaw_of(T[:3, :3]))
        m = self.last_match_odom
        still = m is not None and math.hypot(o[0] - m[0], o[1] - m[1]) < self.p['still_m'] \
            and abs(wrap(o[2] - m[2])) < math.radians(self.p['still_deg'])
        if not still:
            self.last_match_odom = o
            self.still_count = 0
            return True
        self.still_count += 1
        return self.still_count < int(self.p['still_matches'])

    def _height(self, lvl, rng, sm, x, y, yaw, T_odom_base):
        """Ground seen near the robot vs the ground heights already mapped there: corrects the height drift."""
        m = ~sm & (rng > 0.6) & (rng < 6.0)
        if int(m.sum()) < 50:
            return
        c, s = math.cos(yaw), math.sin(yaw)
        q = lvl[m]
        gx = c * q[:, 0] - s * q[:, 1] + x
        gy = s * q[:, 0] + c * q[:, 1] + y
        gz = q[:, 2] + T_odom_base[2, 3] + self.C[2]
        cell = float(self.p['ground_cell_m'])
        ix, iy = np.floor(gx / cell).astype(int), np.floor(gy / cell).astype(int)
        key = ix * 100003 + iy
        order = np.argsort(key)
        uniq, start = np.unique(key[order], return_index=True)
        cells = {}
        for k, idx in zip(uniq, np.split(order, start[1:])):
            if len(idx) >= 3:
                cells[(int(ix[idx[0]]), int(iy[idx[0]]))] = float(np.percentile(gz[idx], 20))
        err = [self.ground[k][0] - v for k, v in cells.items() if k in self.ground and self.ground[k][1] >= 2]
        if len(err) >= 8:
            dz = float(np.clip(np.median(err), -0.10, 0.10)) * float(self.p['gain_z'])
            self.C[2] += dz
            cells = {k: v + dz for k, v in cells.items()}
        for k, v in cells.items():
            g = self.ground.get(k)
            if g is None:
                self.ground[k] = [v, 1]
            elif abs(v - g[0]) < 0.25:
                w = min(g[1] + 1, 20)
                g[0] += (v - g[0]) / w
                g[1] = w


class CorrectionSmoother:
    """The map -> odom correction as published: follows the localizer's, at most this fast. A correction is knowledge, not
    motion; released all at once it looks like a fall or a runaway to the safety monitor and smears the costmaps."""

    def __init__(self, start, max_mps=0.25, max_zps=0.10, max_dps=6.0):
        self.c = np.array(start, float)
        self.v = float(max_mps)
        self.vz = float(max_zps)
        self.w = math.radians(max_dps)

    def step(self, target, dt, pivot_xy=(0.0, 0.0)):
        """Move towards `target` (x, y, z, yaw) for dt seconds. pivot_xy: the robot in odom; a yaw correction turns about
        the robot, so it never swings the robot sideways."""
        dt = max(0.0, min(float(dt), 0.5))
        px, py = float(pivot_xy[0]), float(pivot_xy[1])

        def robot(c):
            cs, sn = math.cos(c[3]), math.sin(c[3])
            return np.array([cs * px - sn * py + c[0], sn * px + cs * py + c[1]])

        here, want = robot(self.c), robot(target)
        dyaw = max(-self.w * dt, min(self.w * dt, wrap(target[3] - self.c[3])))
        d = want - here
        n = float(np.hypot(d[0], d[1]))
        if n > self.v * dt:
            d = d * (self.v * dt / n)
        yaw = wrap(self.c[3] + dyaw)
        cs, sn = math.cos(yaw), math.sin(yaw)
        new = here + d
        self.c[0], self.c[1] = new[0] - (cs * px - sn * py), new[1] - (sn * px + cs * py)
        self.c[2] += max(-self.vz * dt, min(self.vz * dt, float(target[2]) - self.c[2]))
        self.c[3] = yaw
        return self.c
