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
    def insert(self, pts, stamp):
        """Fuse an (N, 3) array of points expressed in the map frame."""
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
        new = np.where(np.isfinite(cur), cur + self.alpha * (zmax - cur), zmax)
        hv[uniq] = new
        self.stamp.ravel()[uniq] = stamp
        self.hits.ravel()[uniq] += cnt.astype(np.int32)
        return len(uniq)

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
        # only lower obstacles that would be impossible under the robot; keep
        # real measurements that agree with the footprint height
        cur = self.h[m]
        self.h[m] = np.where(np.isfinite(cur) & (np.abs(cur - zp) < 0.03), cur, zp)
        self.stamp[m] = stamp

    def expire(self, now):
        old = (now - self.stamp) > self.max_age
        self.h[old] = np.nan
        self.hits[old] = 0
