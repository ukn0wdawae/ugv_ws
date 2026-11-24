"""Unit tests for the pure-numpy traversability core (no ROS needed).

Run:  python3 -m pytest -q src/ugv_main/ugv_traversability/test
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ugv_traversability.cloud_utils import apply, quat_to_rot, rot_to_rpy, rpy_to_rot  # noqa: E402
from ugv_traversability.elevation_map import RollingElevationMap  # noqa: E402
from ugv_traversability.features import (TraversabilityParams, clamp_footprint_cost,  # noqa: E402
                                         compute_traversability, fit_planes, robust_gradient, step_heights)

RES = 0.05
P = TraversabilityParams()


def grid(n=80):
    x = (np.arange(n) - n / 2) * RES
    X, Y = np.meshgrid(x, x)  # [iy, ix]
    return X, Y


def interior(a, m=6):
    return a[m:-m, m:-m]


# ----------------------------------------------------------------------------- features
def test_flat_ground_is_free():
    X, _ = grid()
    h = np.zeros_like(X)
    L = compute_traversability(h, P)
    c = interior(L['cost'])
    assert (c == 0).all()
    assert np.nanmax(interior(L['slope_deg'])) < 0.1


@pytest.mark.parametrize('deg,expect', [(5, 'free'), (10, 'free'), (18, 'costly'), (30, 'lethal')])
def test_planar_slopes(deg, expect):
    X, Y = grid()
    th = np.radians(deg)
    # slope along a diagonal direction to exercise both gradient components
    h = np.tan(th) * (X * np.cos(0.6) + Y * np.sin(0.6))
    L = compute_traversability(h, P)
    s = interior(L['slope_deg'])
    assert np.allclose(s, deg, atol=0.2), (s.min(), s.max())
    assert np.nanmax(interior(L['step'])) < 0.005  # a smooth slope is not a step
    c = interior(L['cost'])
    if expect == 'free':
        assert (c == 0).all()
    elif expect == 'costly':
        assert ((c > 0) & (c < 100)).all()
    else:
        assert (c == 100).all()


@pytest.mark.parametrize('height,lethal', [(0.01, False), (0.03, False), (0.05, True), (0.08, True)])
def test_curb_step(height, lethal):
    X, _ = grid()
    h = np.where(X > 0.0, height, 0.0)
    L = compute_traversability(h, P)
    col = np.argmin(np.abs(X[0] - 0.0))
    band = L['cost'][10:-10, col - 1:col + 2]
    far = L['cost'][10:-10, :col - 8]
    assert (far == 0).all()
    st = np.nanmax(L['step'][10:-10, col - 1:col + 2])
    assert st == pytest.approx(height, abs=0.012)
    if lethal:
        assert (band.max(axis=1) == 100).all()
    else:
        assert (band < 100).all()


def test_curb_on_slope_detected():
    """A 5 cm curb on a 10 deg slope must still be lethal, the slope alone not."""
    X, _ = grid()
    h = np.tan(np.radians(10)) * X + np.where(X > 0, 0.05, 0.0)
    L = compute_traversability(h, P)
    col = np.argmin(np.abs(X[0]))
    assert (L['cost'][10:-10, col - 1:col + 2].max(axis=1) == 100).all()
    assert (L['cost'][10:-10, 5:col - 8] == 0).all()


def test_box_obstacle_and_roughness():
    X, Y = grid()
    rng = np.random.default_rng(0)
    h = rng.normal(0, 0.002, X.shape)  # sensor-noise level roughness -> free
    L = compute_traversability(h, P)
    assert (interior(L['cost']) < 20).all()
    # rubble: 3 cm std
    h2 = rng.normal(0, 0.03, X.shape)
    L2 = compute_traversability(h2, P)
    assert np.mean(interior(L2['cost']) == 100) > 0.9
    # 20 cm trunk 15x15 cm
    h3 = np.zeros_like(X)
    h3[(np.abs(X) < 0.075) & (np.abs(Y) < 0.075)] = 0.2
    L3 = compute_traversability(h3, P)
    c = L3['cost']
    n = X.shape[0] // 2
    assert c[n, n] == 100 and c[n, n + 3] == 100
    assert c[n, n + 12] == 0


def test_unknown_cells():
    X, _ = grid()
    h = np.zeros_like(X)
    h[:, :30] = np.nan
    L = compute_traversability(h, P)
    assert (L['cost'][:, :28] == -1).all()
    assert (interior(L['cost'][:, 36:]) == 0).all()


def test_sparse_rings_like_lidar():
    """Every 3rd row known (lidar ring pattern) still gives a valid slope estimate."""
    X, Y = grid()
    h = np.tan(np.radians(8)) * X
    h[np.arange(h.shape[0]) % 3 != 0, :] = np.nan
    a, b, zc, r, ok = fit_planes(h, RES, P.plane_radius, P.min_valid_ratio)
    s = np.degrees(np.arctan(np.hypot(a, b)))[ok]
    assert ok.mean() > 0.2
    assert np.allclose(s, 8, atol=0.3)


# ----------------------------------------------------------------------------- elevation map
def test_map_insert_and_roll():
    m = RollingElevationMap(length=4.0, resolution=0.1, alpha=1.0, max_age=5.0)
    m.move_to(0.0, 0.0)
    pts = np.array([[0.55, 0.55, 1.0], [0.56, 0.54, 1.2], [1.05, -0.95, 0.3], [50, 50, 9]])
    assert m.insert(pts, 0.0) == 2
    ix, iy = m.world_to_index(np.array([0.55]), np.array([0.55]))
    assert m.h[iy[0], ix[0]] == pytest.approx(1.2)  # max of the two points
    m.move_to(1.0, 0.0)  # shift by 10 cells in x
    ix, iy = m.world_to_index(np.array([0.55]), np.array([0.55]))
    assert m.h[iy[0], ix[0]] == pytest.approx(1.2)
    ix, iy = m.world_to_index(np.array([1.05]), np.array([-0.95]))
    assert m.h[iy[0], ix[0]] == pytest.approx(0.3)
    assert np.isfinite(m.h).sum() == 2
    m.expire(10.0)
    assert np.isfinite(m.h).sum() == 0


def test_map_fusion_filters_noise():
    m = RollingElevationMap(length=2.0, resolution=0.1, alpha=0.4)
    m.move_to(0, 0)
    rng = np.random.default_rng(1)
    for k in range(40):
        z = 0.5 + rng.normal(0, 0.01)
        m.insert(np.array([[0.05, 0.05, z]]), float(k))
    ix, iy = m.world_to_index(np.array([0.05]), np.array([0.05]))
    assert abs(m.h[iy[0], ix[0]] - 0.5) < 0.012


def test_footprint_marks_ground():
    m = RollingElevationMap(length=2.0, resolution=0.05)
    m.move_to(0, 0)
    m.set_footprint(0.0, 0.0, 0.3, 0.12, 0.0)
    assert np.nanmax(m.h) == pytest.approx(0.3)
    assert np.isfinite(m.h).sum() >= 15


# ----------------------------------------------------------------------------- transforms
def test_rotations_roundtrip():
    for r, p, y in [(0.1, -0.2, 2.5), (0, 0, 0), (-0.3, 0.25, -1.0)]:
        R = rpy_to_rot(r, p, y)
        assert np.allclose(rot_to_rpy(R), (r, p, y))
        # quaternion of the same rotation
        qw = np.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
        qx = (R[2, 1] - R[1, 2]) / (4 * qw)
        qy = (R[0, 2] - R[2, 0]) / (4 * qw)
        qz = (R[1, 0] - R[0, 1]) / (4 * qw)
        assert np.allclose(quat_to_rot(qx, qy, qz, qw), R)
    T = np.eye(4)
    T[:3, :3] = rpy_to_rot(0, 0, np.pi / 2)
    T[:3, 3] = (1, 2, 3)
    assert np.allclose(apply(T, np.array([[1.0, 0, 0]])), [[1, 3, 3]])


def test_decimated_robust_gradient_matches_full():
    # 12 deg ramp with a 6 cm curb and some noise: decimating the median must not change
    # the recovered slope or which cells become lethal
    rng = np.random.default_rng(3)
    n, res = 120, 0.05
    x = np.arange(n) * res
    h = np.tile(np.tan(np.radians(12.0)) * x, (n, 1)) + rng.normal(0, 0.002, (n, n))
    h[:, 70:] += 0.06
    h[rng.random((n, n)) < 0.15] = np.nan
    full = robust_gradient(h, res, 0.15, decimate=1)
    dec = robust_gradient(h, res, 0.15, decimate=2)
    inner = np.s_[10:-10, 10:60]
    assert np.nanmedian(dec[0][inner]) == pytest.approx(np.tan(np.radians(12.0)), abs=0.01)
    assert np.nanmedian(np.abs(dec[0] - full[0])) < 0.01
    st_full = step_heights(h, res, 0.10, *full)
    st_dec = step_heights(h, res, 0.10, *dec)
    lethal_full = np.nan_to_num(st_full) >= 0.045
    lethal_dec = np.nan_to_num(st_dec) >= 0.045
    assert (lethal_full != lethal_dec).mean() < 0.005
    assert lethal_dec[:, 69:71].mean() > 0.8  # the curb is still found


def test_footprint_on_side_slope_follows_the_slope():
    """Robot on a 10 deg side slope, cells under it unseen (lidar is blind within ~0.85 m, camera looks
    ahead): the footprint fill must lie on the robot's plane, not on a flat disc that is ~2 cm off at the
    rim and makes a step ring against the measured ground around it."""
    slope = np.tan(np.radians(10.0))
    normal = np.array([0.0, -np.sin(np.radians(10.0)), np.cos(np.radians(10.0))])
    errs = {}
    for name, nrm in (('plane', normal), ('flat', None)):
        m = RollingElevationMap(length=2.0, resolution=0.02, alpha=1.0)
        m.move_to(0.0, 0.0)
        X, Y = m.cell_centers()
        m.set_footprint(0.0, 0.0, 1.0, 0.12, 0.0, normal=nrm)
        disc = X ** 2 + Y ** 2 <= 0.12 ** 2
        errs[name] = np.max(np.abs(m.h[disc] - (1.0 + slope * Y[disc])))
    assert errs['plane'] < 1e-6
    assert errs['flat'] > 0.015


def test_clamp_footprint_cost_prevents_adjacent_obstacle_bleed():
    """An obstacle right outside the chassis bleeds lethal cost onto the robot's standing cell
    via the +/-0.15 m feature window. Clamping the footprint disc prevents Nav2 deadlock at t=0
    while keeping all obstacle cells outside the footprint strictly lethal (100)."""
    X, Y = grid()
    h = np.zeros_like(X)
    # 30 cm bush/curb starting just outside the 0.15 m footprint radius
    h[X > 0.16] = 0.30
    cost = compute_traversability(h, P)['cost']
    c = X.shape[0] // 2
    assert cost[c, c + 1] == 100  # without clamping, the +/-0.15 m window bleeds lethal inside the footprint

    clamped = clamp_footprint_cost(cost, X, Y, 0.0, 0.0, radius=0.15, max_cost=50)
    inside = (X ** 2 + Y ** 2) <= 0.15 ** 2
    assert (clamped[inside] <= 50).all()
    # every cell outside the footprint disc is completely untouched, and the curb wall stays lethal (100)
    assert np.array_equal(clamped[~inside], cost[~inside])
    assert (clamped[(X >= 0.20) & (X <= 0.30) & (np.abs(Y) < 0.5)] == 100).all()
    # unknown (-1), free (0), and moderate cost (< 50) inside the footprint are untouched
    mixed = np.array([[-1, 0, 30, 80, 100]], dtype=np.int8)
    mx = np.array([[0.0, 0.02, 0.04, 0.06, 0.25]])
    my = np.zeros_like(mx)
    out = clamp_footprint_cost(mixed, mx, my, 0.0, 0.0, radius=0.15, max_cost=50)
    assert out.tolist() == [[-1, 0, 30, 50, 100]]


def test_obstacle_cost_ignores_smooth_steep_slope_and_flags_boulder():
    """On a smooth 32 deg hillside, full traversability `cost` marks the hillside lethal (100),
    whereas `obstacle_cost` stays near 0 on the smooth dirt slope and concave hill-bottom transition
    while marking a boulder sitting on that slope lethal (100)."""
    X, Y = grid()
    # Smooth 32 deg ramp for X > 0 transitioning smoothly into flat ground for X <= 0
    s = np.tan(np.radians(32.0))
    h = np.where(X > 0.0, s * (X - 0.5 * (1.0 - np.exp(-2.0 * X))), 0.0)
    # Add a 0.40 m tall boulder on the steep ramp at (1.5, 0.0)
    boulder = ((X - 1.5) ** 2 + Y ** 2) <= 0.30 ** 2
    h = h + np.where(boulder, 0.40, 0.0)

    layers = compute_traversability(h, P)
    smooth_steep = (X > 1.0) & (X < 2.2) & (np.abs(Y) > 0.8) & (np.abs(Y) < 1.8)
    assert (layers['cost'][smooth_steep] == 100).all()
    assert (layers['obstacle_cost'][smooth_steep] == 0).all()
    # Smooth concave transition near X=0 stays non-lethal in obstacle_cost
    concave = (np.abs(X) < 0.3) & (np.abs(Y) < 1.0)
    assert (layers['obstacle_cost'][concave] < 70).all()
    # Boulder edge on the steep slope is strictly lethal (100) in obstacle_cost
    boulder_ring = ((X - 1.5) ** 2 + Y ** 2 <= 0.35 ** 2) & ((X - 1.5) ** 2 + Y ** 2 >= 0.22 ** 2)
    assert (layers['obstacle_cost'][boulder_ring] == 100).mean() > 0.8


def test_rolling_cost_grid_persists_non_lethal_slope_costs():
    from ugv_traversability.elevation_map import RollingCostGrid
    g = RollingCostGrid(length=40.0, resolution=0.20, recenter_dist=5.0, alpha=0.5, max_cost=90)
    g.move_to(0.0, 0.0)
    X, Y = grid()
    local = np.zeros_like(X, dtype=np.int8)
    local[Y > 0.5] = 70    # steep hillside north of trail
    local[Y > 1.5] = 100   # lethal slope (must be capped at max_cost=90 in global_grid)
    assert g.update_from_local(X, Y, local) > 0
    # Move 10 m away (triggers recenter) and confirm scanned hillside cost persists
    g.move_to(10.0, 0.0)
    ix_flat = int(np.floor((0.0 - g.origin[0]) / g.res))
    iy_flat = int(np.floor((0.0 - g.origin[1]) / g.res))
    iy_hill = int(np.floor((1.0 - g.origin[1]) / g.res))
    iy_steep = int(np.floor((1.8 - g.origin[1]) / g.res))
    assert g.data[iy_flat, ix_flat] == 0
    assert 60 <= g.data[iy_hill, ix_flat] <= 80
    assert g.data[iy_steep, ix_flat] == 90  # strictly < 100 so StaticLayer never locks lethal cells
