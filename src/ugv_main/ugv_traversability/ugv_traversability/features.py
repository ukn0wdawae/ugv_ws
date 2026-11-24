"""Pure-numpy terrain feature extraction on a 2.5D elevation grid.

Everything here is ROS-free so it can be unit-tested on any machine.

Grid convention: arrays are indexed ``[iy, ix]`` (row = y), exactly like
``nav_msgs/OccupancyGrid.data`` reshaped to ``(height, width)``.  Unknown cells
are ``NaN``.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TraversabilityParams:
    """Thresholds default to the Waveshare UGV Rover / ugv_pro_audio geometry.

    wheel radius 0.040 m, chassis clearance 0.060 m, wheel base 0.171 m,
    track 0.175 m, mu(rubber/ground) ~0.75.
    """

    resolution: float = 0.05          # m / cell
    plane_radius: float = 0.15        # half window for slope/roughness plane fit (m)
    step_radius: float = 0.10         # half window for step detection (m)
    step_outlier_trim: int = 0        # trim k highest/lowest cells in step window when >= 2k+3 cells valid
    min_valid_ratio: float = 0.30     # fraction of window cells needed for a plane fit

    slope_safe_deg: float = 12.0      # below: no cost
    slope_critical_deg: float = 25.0  # at/above: lethal
    step_safe: float = 0.020          # m
    step_critical: float = 0.045      # m  (~ wheel radius)
    roughness_safe: float = 0.010     # m  (std of plane residual)
    roughness_critical: float = 0.030

    weights: dict = field(default_factory=lambda: {'slope': 1.0, 'step': 1.0, 'roughness': 1.0})


def _box_sum(a, k):
    """Sum of ``a`` over a (2k+1)x(2k+1) window centred on every cell (zero padded)."""
    p = np.pad(a, ((k + 1, k), (k + 1, k)), mode='constant')
    c = p.cumsum(0).cumsum(1)
    w = 2 * k + 1
    return c[w:, w:] - c[:-w, w:] - c[w:, :-w] + c[:-w, :-w]


def _shift(a, dy, dx, fill=np.nan):
    """out[y, x] = a[y + dy, x + dx] (NaN outside)."""
    out = np.full_like(a, fill)
    ny, nx = a.shape
    ys = slice(max(0, -dy), min(ny, ny - dy))
    xs = slice(max(0, -dx), min(nx, nx - dx))
    ys2 = slice(max(0, dy), min(ny, ny + dy))
    xs2 = slice(max(0, dx), min(nx, nx + dx))
    out[ys, xs] = a[ys2, xs2]
    return out


def fit_planes(h, res, radius, min_valid_ratio):
    """Least-squares plane z = a*x + b*y + c around every cell.

    Returns (a, b, z_fit_at_centre, residual_std, ok_mask). Uses integral images
    so cost is O(N) regardless of window size.
    """
    k = max(1, int(round(radius / res)))
    ny, nx = h.shape
    V = np.isfinite(h).astype(np.float64)
    Z = np.where(V > 0, h, 0.0).astype(np.float64)
    # local metric coordinates; the plane solve is translation-invariant so use
    # coordinates relative to the grid centre for conditioning
    X = (np.arange(nx) - nx / 2.0)[None, :] * res * np.ones((ny, 1))
    Y = (np.arange(ny) - ny / 2.0)[:, None] * res * np.ones((1, nx))
    S1 = _box_sum(V, k)
    Sx = _box_sum(V * X, k)
    Sy = _box_sum(V * Y, k)
    Sz = _box_sum(Z, k)
    Sxx = _box_sum(V * X * X, k)
    Syy = _box_sum(V * Y * Y, k)
    Sxy = _box_sum(V * X * Y, k)
    Sxz = _box_sum(X * Z, k)
    Syz = _box_sum(Y * Z, k)
    Szz = _box_sum(Z * Z, k)

    need = min_valid_ratio * (2 * k + 1) ** 2
    ok = (S1 >= max(need, 4)) & (V > 0)
    a = np.full(h.shape, np.nan)
    b = np.full(h.shape, np.nan)
    zc = np.full(h.shape, np.nan)
    rough = np.full(h.shape, np.nan)
    idx = np.nonzero(ok)
    if idx[0].size:
        M = np.empty((idx[0].size, 3, 3))
        M[:, 0, 0] = Sxx[idx]
        M[:, 0, 1] = M[:, 1, 0] = Sxy[idx]
        M[:, 0, 2] = M[:, 2, 0] = Sx[idx]
        M[:, 1, 1] = Syy[idx]
        M[:, 1, 2] = M[:, 2, 1] = Sy[idx]
        M[:, 2, 2] = S1[idx]
        v = np.stack([Sxz[idx], Syz[idx], Sz[idx]], axis=1)
        det = np.linalg.det(M)
        good = np.abs(det) > 1e-12
        sol = np.full((idx[0].size, 3), np.nan)
        if good.any():
            sol[good] = np.linalg.solve(M[good], v[good][..., None])[..., 0]
        sse = Szz[idx] - np.einsum('ij,ij->i', sol, v)
        a[idx], b[idx] = sol[:, 0], sol[:, 1]
        zc[idx] = sol[:, 0] * X[idx] + sol[:, 1] * Y[idx] + sol[:, 2]
        rough[idx] = np.sqrt(np.clip(sse, 0, None) / S1[idx])
        bad = ~good
        if bad.any():
            ok[tuple(i[bad] for i in idx)] = False
    return a, b, zc, rough, ok


def robust_gradient(h, res, radius, decimate=2):
    """Median of the finite-difference gradients in a window around every cell.

    Unlike a least-squares plane, the median ignores the few large differences
    produced by a curb/step, so it recovers the *underlying* terrain slope next
    to a discontinuity.  Returns (gx, gy) in m/m (NaN where unknown).

    The underlying slope varies slowly, so the median is evaluated on every
    ``decimate``-th cell in each direction and copied to its neighbours
    (~decimate^2 faster; the median is by far the most expensive feature).
    """
    from numpy.lib.stride_tricks import sliding_window_view as swv
    k = max(1, int(round(radius / res)))
    q = max(1, int(decimate))
    ny, nx = h.shape
    gx = np.full(h.shape, np.nan)
    gy = np.full(h.shape, np.nan)
    gx[:, :-1] = (h[:, 1:] - h[:, :-1]) / res
    gy[:-1, :] = (h[1:, :] - h[:-1, :]) / res
    valid = np.isfinite(h)
    # evaluation centres: the decimated lattice, wherever any cell of its q x q block is known
    blk = np.zeros(((ny + q - 1) // q, (nx + q - 1) // q), dtype=bool)
    for dy in range(q):
        for dx in range(q):
            v = valid[dy::q, dx::q]
            blk[:v.shape[0], :v.shape[1]] |= v
    out = []
    for g in (gx, gy):
        p = np.pad(g, k, mode='constant', constant_values=np.nan)
        w = swv(p, (2 * k + 1, 2 * k + 1))[::q, ::q][blk].reshape(-1, (2 * k + 1) ** 2)
        # NaN-aware median via sort (NaNs sort last) - ~4x faster than nanmedian
        s = np.sort(w, axis=1)
        n = np.isfinite(s).sum(axis=1)
        lo = np.clip((n - 1) // 2, 0, None)
        hi = np.clip(n // 2, 0, None)
        rows = np.arange(len(s))
        med = 0.5 * (s[rows, lo] + s[rows, hi])
        med[n == 0] = np.nan
        coarse = np.full(blk.shape, np.nan)
        coarse[blk] = med
        o = np.repeat(np.repeat(coarse, q, axis=0), q, axis=1)[:ny, :nx]
        o[~valid] = np.nan
        out.append(o)
    return out[0], out[1]


def step_heights(h, res, radius, gx, gy, trim=0):
    """Spread of slope-corrected heights within the step window.

    For every centre cell the neighbours' heights are corrected by the robust
    local gradient (``gx``, ``gy``) and the spread (max - min, or trimmed spread
    when ``trim > 0`` and enough cells are valid) of what remains is the step a
    wheel has to climb.  Smooth slopes give ~0, a curb gives ~its full height
    even on a slope, while single-cell attitude/lidar spikes are rejected when
    ``trim >= 1``.  Where no gradient is available the raw spread is used.
    """
    k = max(1, int(round(radius / res)))
    gxx = np.nan_to_num(gx)
    gyy = np.nan_to_num(gy)
    trim = max(0, int(trim))
    if trim == 0:
        rmax = np.full(h.shape, -np.inf)
        rmin = np.full(h.shape, np.inf)
        cnt = np.zeros(h.shape)
        for dy in range(-k, k + 1):
            for dx in range(-k, k + 1):
                hs = _shift(h, dy, dx)
                r = hs - res * (gxx * dx + gyy * dy)
                f = np.isfinite(r)
                rmax = np.where(f, np.maximum(rmax, r), rmax)
                rmin = np.where(f, np.minimum(rmin, r), rmin)
                cnt += f
        step = rmax - rmin
        step[(cnt < 2) | ~np.isfinite(h)] = np.nan
        return step

    w = 2 * k + 1
    stack = np.empty((w * w, h.shape[0], h.shape[1]), dtype=np.float64)
    idx = 0
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            hs = _shift(h, dy, dx)
            stack[idx] = hs - res * (gxx * dx + gyy * dy)
            idx += 1
    # Sort along window axis; NaNs sort to the end
    s = np.sort(stack, axis=0)
    cnt = np.isfinite(s).sum(axis=0)
    # Trim `trim` extremes from each end only when enough cells exist (>= 2*trim + 3)
    eff_trim = np.where(cnt >= 2 * trim + 3, trim, 0)
    hi_idx = np.clip(cnt - 1 - eff_trim, 0, w * w - 1)
    lo_idx = np.clip(eff_trim, 0, w * w - 1)
    yy, xx = np.indices(h.shape)
    step = s[hi_idx, yy, xx] - s[lo_idx, yy, xx]
    step[(cnt < 2) | ~np.isfinite(h)] = np.nan
    return step


def _norm(v, safe, crit):
    return np.clip((v - safe) / max(crit - safe, 1e-9), 0.0, 1.0)


def compute_traversability(h, p: TraversabilityParams):
    """Return dict of layers: slope_deg, step, roughness, cost (int8, -1 unknown)."""
    a, b, zc, rough, ok = fit_planes(h, p.resolution, p.plane_radius, p.min_valid_ratio)
    slope = np.degrees(np.arctan(np.hypot(a, b)))
    gx, gy = robust_gradient(h, p.resolution, p.plane_radius)
    step = step_heights(h, p.resolution, p.step_radius, gx, gy, trim=p.step_outlier_trim)

    known = np.isfinite(h) & (np.isfinite(step) | ok)
    s_n = np.where(np.isfinite(slope), _norm(slope, p.slope_safe_deg, p.slope_critical_deg), 0.0)
    t_n = np.where(np.isfinite(step), _norm(step, p.step_safe, p.step_critical), 0.0)
    r_n = np.where(np.isfinite(rough), _norm(rough, p.roughness_safe, p.roughness_critical), 0.0)
    w = p.weights
    danger = np.maximum.reduce([w.get('slope', 1.0) * s_n, w.get('step', 1.0) * t_n,
                                w.get('roughness', 1.0) * r_n])
    lethal = ((np.nan_to_num(slope) >= p.slope_critical_deg) |
              (np.nan_to_num(step) >= p.step_critical) |
              (np.nan_to_num(rough) >= p.roughness_critical))
    cost = np.round(99.0 * np.clip(danger, 0, 1)).astype(np.int16)
    cost[lethal] = 100
    cost[~known] = -1

    # Non-slope geometric obstacle cost (boulders, trunks, bushes, vertical faces >= 50 deg):
    # stays 0 on smooth steep dirt slopes and concave hill-bottom transitions so slope-descent
    # recovery can steer around real physical obstacles without mistaking the hillside for a wall.
    wall_deg = max(50.0, 2.0 * p.slope_critical_deg)
    obs_danger = np.maximum(w.get('step', 1.0) * t_n, w.get('roughness', 1.0) * r_n)
    obs_lethal = ((np.nan_to_num(step) >= p.step_critical) |
                  (np.nan_to_num(rough) >= p.roughness_critical) |
                  (np.nan_to_num(slope) >= wall_deg))
    obstacle_cost = np.round(99.0 * np.clip(obs_danger, 0, 1)).astype(np.int16)
    obstacle_cost[obs_lethal] = 100
    obstacle_cost[~known] = -1

    # Continuous non-lethal terrain cost (slope + roughness, capped at 90) for global_costmap so
    # Nav2's global planner strongly prefers low-slope flat trails without locking in lethal cells.
    terr_danger = np.maximum(w.get('slope', 1.0) * s_n, w.get('roughness', 1.0) * r_n)
    terrain_cost = np.round(90.0 * np.clip(terr_danger, 0, 1)).astype(np.int16)
    terrain_cost[~known] = -1

    return {
        'slope_deg': slope, 'step': step, 'roughness': rough,
        'cost': cost.astype(np.int8), 'obstacle_cost': obstacle_cost.astype(np.int8),
        'terrain_cost': terrain_cost.astype(np.int8), 'plane_ok': ok,
    }


def clamp_footprint_cost(cost, X, Y, rx, ry, radius, max_cost=50):
    """Clamp cells inside the robot's own footprint so they are never lethal.

    ``compute_traversability`` evaluates a +/-0.15 m window around every cell,
    so a bush or steep curb right next to the chassis can bleed lethal cost
    (100) onto the robot's standing spot and deadlock Nav2 DWB / BackUp at t=0.
    Clamping cells inside ``radius`` to at most ``max_cost`` preserves real
    obstacles outside the footprint (and preserves slope cost up to max_cost
    inside it) while keeping the robot's own pose legally escapable.
    """
    if radius <= 0.0:
        return cost
    out = cost.copy()
    cap = int(np.clip(max_cost, 0, 99))
    inside = ((X - float(rx)) ** 2 + (Y - float(ry)) ** 2) <= float(radius) ** 2
    out[inside & (out > cap)] = cap
    return out
