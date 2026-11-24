"""Robot-centric rolling 2.5D elevation map (pure numpy, ROS-free)."""

import numpy as np


class RollingElevationMap:
    """Square height grid that follows the robot in a fixed (odom) frame.

    * arrays are ``[iy, ix]``; cell (ix, iy) centre is
      ``origin + (ix + 0.5, iy + 0.5) * resolution``
    * each scan contributes the **max** height of its points per cell (so
      trunks/walls show their top) which is then fused into the map with an
      exponential filter (``alpha``) to suppress sensor noise
    * cells not refreshed for ``max_age`` seconds are forgotten
    """

    def __init__(self, length=8.0, resolution=0.05, alpha=0.4, max_age=20.0, recenter_dist=0.5):
        self.res = float(resolution)
        self.n = int(round(length / resolution))
        if self.n % 2:
            self.n += 1
        self.alpha = float(alpha)
        self.max_age = float(max_age)
        self.recenter_dist = float(recenter_dist)
        self.h = np.full((self.n, self.n), np.nan, dtype=np.float64)
        self.stamp = np.full((self.n, self.n), -np.inf, dtype=np.float64)
        self.hits = np.zeros((self.n, self.n), dtype=np.int32)
        self.origin = None  # (x, y) of the lower-left corner

    # ------------------------------------------------------------------ geometry
    @property
    def length(self):
        return self.n * self.res

    def center(self):
        return self.origin + self.length / 2.0

    def cell_centers(self):
        ix = np.arange(self.n)
        xs = self.origin[0] + (ix + 0.5) * self.res
        ys = self.origin[1] + (ix + 0.5) * self.res
        return np.meshgrid(xs, ys)  # X[iy, ix], Y[iy, ix]

    def world_to_index(self, x, y):
        ix = np.floor((x - self.origin[0]) / self.res).astype(np.int64)
        iy = np.floor((y - self.origin[1]) / self.res).astype(np.int64)
        return ix, iy

    # ------------------------------------------------------------------ rolling
    def move_to(self, x, y):
        """Re-centre on (x, y) if the robot moved more than ``recenter_dist``."""
        snap = lambda v: np.floor(v / self.res) * self.res  # noqa: E731
        new_origin = np.array([snap(x) - self.length / 2.0, snap(y) - self.length / 2.0])
        if self.origin is None:
            self.origin = new_origin
            return
        if np.hypot(*(self.center() - (x, y))) < self.recenter_dist:
            return
        d = np.round((new_origin - self.origin) / self.res).astype(int)  # cells moved (dx, dy)
        self.h = self._shift(self.h, d, np.nan)
        self.stamp = self._shift(self.stamp, d, -np.inf)
        self.hits = self._shift(self.hits, d, 0)
        self.origin = self.origin + d * self.res

    def _shift(self, a, d, fill):
        dx, dy = int(d[0]), int(d[1])
        out = np.full_like(a, fill)
        n = self.n
        if abs(dx) >= n or abs(dy) >= n:
            return out
        # new[iy, ix] = old[iy + dy, ix + dx]
        out[max(0, -dy):n - max(0, dy), max(0, -dx):n - max(0, dx)] = \
            a[max(0, dy):n - max(0, -dy), max(0, dx):n - max(0, -dx)]
        return out

    # ------------------------------------------------------------------ update
    @staticmethod
    def _estimate_scan_tilt(ux, uy, dz, sx, sy, max_tilt_rad=0.055, max_dz=0.05):
        """Estimate rigid scan-to-map attitude/height offset (a*dx + b*dy + c) via 2-pass Huber IRLS.

        Cancels per-scan IMU roll/pitch jitter (0.5-1.5 deg) that otherwise creates multi-centimetre
        washboard steps between adjacent lidar rings, while ignoring local curbs/boulders/trunks.
        """
        dx = ux - float(sx)
        dy = uy - float(sy)
        # Keep plausible ground overlaps (exclude walls/trunks > 8 cm above ground)
        m = np.abs(dz) < 0.08
        if int(m.sum()) < 20:
            return 0.0, 0.0, 0.0
        # Require 2D spatial spread (reject narrow front-only depth-camera wedges where dx and 1 are collinear)
        dxm, dym = dx[m], dy[m]
        cov = np.cov(dxm, dym)
        if cov.shape != (2, 2) or float(np.linalg.eigvalsh(cov)[0]) < 0.15:
            return 0.0, 0.0, 0.0
        X = np.stack([dxm, dym, np.ones(int(m.sum()), dtype=np.float64)], axis=1)
        y = dz[m]
        w = np.ones_like(y)
        sol = np.zeros(3, dtype=np.float64)
        for _ in range(3):
            Xw = X * w[:, None]
            H = Xw.T @ X
            if abs(np.linalg.det(H)) < 1e-6:
                return 0.0, 0.0, 0.0
            sol = np.linalg.solve(H, Xw.T @ y)
            res = np.abs(y - X @ sol)
            # Huber weight with 1.5 cm transition so a real 4-5 cm curb has minimal leverage
            w = np.where(res <= 0.015, 1.0, 0.015 / np.maximum(res, 1e-6))
        a = float(np.clip(sol[0], -max_tilt_rad, max_tilt_rad))
        b = float(np.clip(sol[1], -max_tilt_rad, max_tilt_rad))
        c = float(np.clip(sol[2], -max_dz, max_dz))
        return a, b, c

    def insert(self, pts, stamp, sensor_xy=None, range_weight_scale=0.0, stabilize_attitude=False):
        """Fuse an (N, 3) array of points expressed in the map frame.

        When `sensor_xy` and `stabilize_attitude=True` are given, estimates and removes the rigid
        per-scan roll/pitch/z offset against overlapping map cells before fusing (suppresses IMU
        attitude noise without blurring curbs, trunks, or true terrain slope).
        When `range_weight_scale > 0`, per-cell fusion weight decays with horizontal distance `r`.
        """
        if self.origin is None or len(pts) == 0:
            return 0
        ix, iy = self.world_to_index(pts[:, 0], pts[:, 1])
        ok = (ix >= 0) & (iy >= 0) & (ix < self.n) & (iy < self.n) & np.isfinite(pts[:, 2])
        if not ok.any():
            return 0
        flat = iy[ok] * self.n + ix[ok]
        z = pts[ok, 2]
        order = np.argsort(flat, kind='stable')
        flat, z = flat[order], z[order]
        uniq, start = np.unique(flat, return_index=True)
        zmax = np.maximum.reduceat(z, start)
        cnt = np.diff(np.append(start, len(flat)))

        hv = self.h.ravel()
        cur = hv[uniq]
        ux = uy = None
        if sensor_xy is not None and (stabilize_attitude or range_weight_scale > 0.0):
            ux = self.origin[0] + ((uniq % self.n) + 0.5) * self.res
            uy = self.origin[1] + ((uniq // self.n) + 0.5) * self.res

        if stabilize_attitude and sensor_xy is not None:
            ov = np.isfinite(cur)
            if int(ov.sum()) >= 20:
                a, b, c = self._estimate_scan_tilt(ux[ov], uy[ov], zmax[ov] - cur[ov], sensor_xy[0], sensor_xy[1])
                zmax = zmax - (a * (ux - float(sensor_xy[0])) + b * (uy - float(sensor_xy[1])) + c)

        if sensor_xy is not None and range_weight_scale > 0.0:
            r2 = (ux - float(sensor_xy[0])) ** 2 + (uy - float(sensor_xy[1])) ** 2
            alpha_eff = self.alpha / (1.0 + r2 / (float(range_weight_scale) ** 2))
            alpha_eff = np.maximum(alpha_eff, min(self.alpha, 0.08))
        else:
            alpha_eff = self.alpha
        new = np.where(np.isfinite(cur), cur + alpha_eff * (zmax - cur), zmax)
        hv[uniq] = new
        self.stamp.ravel()[uniq] = stamp
        self.hits.ravel()[uniq] += cnt.astype(np.int32)
        return len(uniq)

    def ground_height_at(self, x, y, normal=None, r_min=0.18, r_max=0.65, min_cells=15):
        """Robustly estimate the map's ground elevation directly under (x, y) from a nearby ring.

        Projects surrounding known map heights back to (x, y) along the robot's ground plane and
        takes the median (immune to adjacent trunks/bushes), allowing 2D-odometry z_est to stay
        anchored to the elevation map without open-loop drift on rolling terrain and side-slopes.
        """
        if self.origin is None:
            return None
        X, Y = self.cell_centers()
        r2 = (X - float(x)) ** 2 + (Y - float(y)) ** 2
        m = np.isfinite(self.h) & (r2 >= float(r_min) ** 2) & (r2 <= float(r_max) ** 2)
        if int(m.sum()) < int(min_cells):
            return None
        z_proj = self.h[m]
        if normal is not None and float(normal[2]) > 0.5:
            z_proj = z_proj + (float(normal[0]) * (X[m] - float(x)) +
                               float(normal[1]) * (Y[m] - float(y))) / float(normal[2])
        return float(np.median(z_proj))

    def set_footprint(self, x, y, z, radius, stamp, normal=None):
        """Mark cells under the robot as ground: the plane the robot stands on, through (x, y, z) with
        the robot's up axis `normal` (flat if None). A flat disc on a side slope is centimetres off the
        real ground at its rim, which reads as a step ring and left a lethal trail behind the robot."""
        if self.origin is None:
            return
        X, Y = self.cell_centers()
        m = (X - x) ** 2 + (Y - y) ** 2 <= radius ** 2
        zp = np.full(int(m.sum()), z, dtype=float)
        if normal is not None and normal[2] > 0.5:
            zp = z - (normal[0] * (X[m] - x) + normal[1] * (Y[m] - y)) / normal[2]
        # Only fill unknown (NaN) cells or LOWER impossible obstacles sticking > 3 cm ABOVE the
        # robot's underside; never RAISE clean measured ground into a fake step plateau when z_est drifts.
        cur = self.h[m]
        keep = np.isfinite(cur) & (cur <= zp + 0.03) & (cur >= zp - 0.15)
        self.h[m] = np.where(keep, cur, zp)
        self.stamp[m] = stamp

    def expire(self, now):
        old = (now - self.stamp) > self.max_age
        self.h[old] = np.nan
        self.hits[old] = 0


class RollingCostGrid:
    """Coarse persistent rolling cost grid (int8, -1 unknown, 0..max_cost) for Nav2 global_costmap.

    Accumulates continuous non-lethal slope/roughness costs across a wide area (e.g. 110 m @ 0.20 m)
    so Nav2's global planner prefers low-slope flat trails over steep hillsides without ever painting
    un-clearable lethal cells (lethal obstacles remain owned by Nav2's obstacle_layer).
    """

    def __init__(self, length=110.0, resolution=0.20, recenter_dist=5.0, alpha=0.5, max_cost=90):
        self.res = float(resolution)
        self.n = int(round(length / resolution))
        if self.n % 2:
            self.n += 1
        self.recenter_dist = float(recenter_dist)
        self.alpha = float(alpha)
        self.max_cost = int(np.clip(max_cost, 1, 95))
        self.data = np.full((self.n, self.n), -1, dtype=np.int8)
        self.origin = None

    @property
    def length(self):
        return self.n * self.res

    def center(self):
        return self.origin + self.length / 2.0

    def move_to(self, x, y):
        snap = lambda v: np.floor(v / self.res) * self.res  # noqa: E731
        new_origin = np.array([snap(x) - self.length / 2.0, snap(y) - self.length / 2.0])
        if self.origin is None:
            self.origin = new_origin
            return
        if np.hypot(*(self.center() - (x, y))) < self.recenter_dist:
            return
        d = np.round((new_origin - self.origin) / self.res).astype(int)
        dx, dy = int(d[0]), int(d[1])
        n = self.n
        out = np.full((n, n), -1, dtype=np.int8)
        if abs(dx) < n and abs(dy) < n:
            out[max(0, -dy):n - max(0, dy), max(0, -dx):n - max(0, dx)] = \
                self.data[max(0, dy):n - max(0, -dy), max(0, dx):n - max(0, -dx)]
        self.data = out
        self.origin = self.origin + d * self.res

    def update_from_local(self, X, Y, local_cost):
        """Fuse known cells of a local cost grid into the coarse persistent cost grid."""
        if self.origin is None:
            return 0
        known = local_cost >= 0
        if not known.any():
            return 0
        xs = X[known]
        ys = Y[known]
        vals = np.clip(local_cost[known].astype(np.float32), 0.0, float(self.max_cost))
        ix = np.floor((xs - self.origin[0]) / self.res).astype(np.int64)
        iy = np.floor((ys - self.origin[1]) / self.res).astype(np.int64)
        ok = (ix >= 0) & (iy >= 0) & (ix < self.n) & (iy < self.n)
        if not ok.any():
            return 0
        flat = iy[ok] * self.n + ix[ok]
        vals = vals[ok]
        uniq, inv = np.unique(flat, return_inverse=True)
        sums = np.bincount(inv, weights=vals)
        cnts = np.bincount(inv)
        mean_v = sums / np.maximum(cnts, 1)
        dv = self.data.ravel()
        cur = dv[uniq]
        blended = np.where(cur >= 0, cur.astype(np.float32) + self.alpha * (mean_v - cur.astype(np.float32)), mean_v)
        dv[uniq] = np.clip(np.round(blended), 0, self.max_cost).astype(np.int8)
        return len(uniq)


def height_shift(T_fr, T_tf, odom_2d):
    """How much lower the published terrain must be than the internal map, so that it fits the TF the viewer and Nav2 see.
    With 3D odometry the map keeps its own (drift-free) height and the difference is the odometry's drift. With 2D odometry
    (flat TF) there is no height in the TF to fit; the map is published as it is."""
    return 0.0 if odom_2d else float(T_fr[2, 3] - T_tf[2, 3])
