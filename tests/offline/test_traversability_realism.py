"""Traversability realism & robustness regression tests.

Covers:
  - Attitude error sensitivity (0 deg, 0.5 deg, 1.0 deg at 1-2 m and 2-4 m) on obstacle-free ground
  - Curb, boulder, and tree-trunk recall with trimmed step & range-weighted fusion enabled
  - Low-branch clearance (max_height_above_robot <= 0.5 m instead of 0.9 m)
  - Boxed-in early footprint clamping & RollingCostGrid global persistence
"""
import math
import numpy as np
import pytest

from ugv_traversability.cloud_utils import apply, rpy_to_rot
from ugv_traversability.elevation_map import RollingCostGrid, RollingElevationMap
from ugv_traversability.features import TraversabilityParams, clamp_footprint_cost, compute_traversability


def _params_from_yaml(trav_cfg):
    rp = trav_cfg['terrain_traversability']['ros__parameters']
    p = TraversabilityParams(
        resolution=float(rp.get('resolution', 0.05)),
        plane_radius=float(rp.get('plane_radius', 0.15)),
        step_radius=float(rp.get('step_radius', 0.10)),
        step_outlier_trim=int(rp.get('step_outlier_trim', 1)),
        min_valid_ratio=float(rp.get('min_valid_ratio', 0.3)),
        slope_safe_deg=float(rp.get('slope_safe_deg', 8.0)),
        slope_critical_deg=float(rp.get('slope_critical_deg', 24.0)),
        step_safe=float(rp.get('step_safe', 0.02)),
        step_critical=float(rp.get('step_critical', 0.045)),
        roughness_safe=float(rp.get('roughness_safe', 0.010)),
        roughness_critical=float(rp.get('roughness_critical', 0.030)),
    )
    return p, rp


def _sim_lidar_flat_ground(n_scans, attitude_sigma_deg, trav_cfg, seed=0):
    """Simulate a 16-ch lidar (-16..+16 deg) scanning obstacle-free flat ground while the robot
    moves slowly with Gaussian roll/pitch attitude error of `attitude_sigma_deg`."""
    p, rp = _params_from_yaml(trav_cfg)
    rng = np.random.default_rng(seed)
    m = RollingElevationMap(
        length=float(rp.get('map_length', 8.0)),
        resolution=p.resolution,
        alpha=float(rp.get('fusion_alpha', 0.35)),
        max_age=float(rp.get('max_age', 12.0)),
    )
    rw_scale = float(rp.get('range_weight_scale', 1.5))
    stab = bool(rp.get('stabilize_attitude', True))
    az = np.linspace(-math.pi, math.pi, 360, endpoint=False)
    # 8 beams below horizon (-16 .. -2 deg) hitting flat ground z=0 from lidar height 0.15 m
    elev_deg = np.linspace(-16.0, -2.0, 8)
    z_lidar = 0.15
    base_pts = []
    for el in elev_deg:
        r = z_lidar / math.tan(math.radians(-el))
        if 0.3 <= r <= 4.5:
            xs = r * np.cos(az)
            ys = r * np.sin(az)
            zs = np.full_like(xs, -z_lidar)  # in lidar frame
            base_pts.append(np.stack([xs, ys, zs], axis=1))
    pts_lidar = np.vstack(base_pts)

    rx, ry = 0.0, 0.0
    m.move_to(rx, ry)
    for k in range(n_scans):
        t = k * 0.1
        rx = 0.04 * k  # slowly advancing 0.4 m/s
        m.move_to(rx, ry)
        # Noisy lidar range (~6 mm) + attitude estimation error (roll, pitch)
        noisy = pts_lidar + rng.normal(0.0, 0.006, pts_lidar.shape)
        er = math.radians(rng.normal(0.0, attitude_sigma_deg))
        ep = math.radians(rng.normal(0.0, attitude_sigma_deg))
        T = np.eye(4)
        T[:3, :3] = rpy_to_rot(er, ep, 0.0)
        T[:3, 3] = (rx, ry, z_lidar)
        pts_w = apply(T, noisy)
        m.insert(pts_w, t, sensor_xy=(rx, ry), range_weight_scale=rw_scale, stabilize_attitude=stab)
        m.set_footprint(rx, ry, 0.0, 0.12, t)

    L = compute_traversability(m.h, p)
    X, Y = m.cell_centers()
    cost = clamp_footprint_cost(L['cost'], X, Y, rx, ry, 0.20, 50)
    dist = np.hypot(X - rx, Y - ry)
    known = cost >= 0
    near_m = known & (dist >= 1.0) & (dist <= 2.0)
    far_m = known & (dist > 2.0) & (dist <= 4.0)
    near_lethal = float((cost[near_m] >= 100).mean()) if near_m.any() else 0.0
    far_lethal = float((cost[far_m] >= 100).mean()) if far_m.any() else 0.0
    return near_lethal, far_lethal


def test_attitude_error_sensitivity_stays_bounded(trav_cfg):
    """Regression for Realism #1 (before fix: 0.5 deg -> 11-18% near / 39-47% far;
    1.0 deg -> 54-60% near / 76-84% far false lethal cells on obstacle-free ground!).
    With range-weighted fusion + trimmed step + smoother alpha, false lethal rates drop dramatically."""
    near0, far0 = _sim_lidar_flat_ground(25, 0.0, trav_cfg, seed=0)
    near05, far05 = _sim_lidar_flat_ground(25, 0.5, trav_cfg, seed=1)
    near10, far10 = _sim_lidar_flat_ground(25, 1.0, trav_cfg, seed=2)
    assert near0 < 0.01 and far0 < 0.01, f'0 deg noise false lethal: near={near0:.2%}, far={far0:.2%}'
    assert near05 < 0.05, f'0.5 deg near false lethal too high: {near05:.2%} (was 11-18%)'
    assert far05 < 0.15, f'0.5 deg far false lethal too high: {far05:.2%} (was 39-47%)'
    assert near10 < 0.15, f'1.0 deg near false lethal too high: {near10:.2%} (was 54-60%)'
    assert far10 < 0.35, f'1.0 deg far false lethal too high: {far10:.2%} (was 76-84%)'


def test_trimmed_step_still_catches_real_curbs_and_trunks(trav_cfg):
    """Trimming single-cell attitude spikes must NEVER miss a real 5 cm curb or 15 cm tree trunk."""
    p, _ = _params_from_yaml(trav_cfg)
    n = 80
    x = (np.arange(n) - n / 2) * p.resolution
    X, Y = np.meshgrid(x, x)

    # 1) 5.5 cm curb
    h_curb = np.where(X > 0.0, 0.055, 0.0)
    L_curb = compute_traversability(h_curb, p)
    col = np.argmin(np.abs(X[0]))
    assert (L_curb['cost'][10:-10, col - 1:col + 2].max(axis=1) == 100).all()

    # 2) 15x15 cm trunk
    h_trunk = np.zeros_like(X)
    h_trunk[(np.abs(X) < 0.075) & (np.abs(Y) < 0.075)] = 0.25
    L_trunk = compute_traversability(h_trunk, p)
    c = n // 2
    assert L_trunk['cost'][c, c] == 100

    # 3) Single-cell noisy 6 cm spike on flat ground IS rejected by step_outlier_trim >= 1
    h_spike = np.zeros_like(X)
    h_spike[c, c] = 0.06
    L_spike = compute_traversability(h_spike, p)
    assert L_spike['step'][c, c] < p.step_critical


def test_low_branches_ignored_above_chassis_clearance(trav_cfg):
    """Regression: max_height_above_robot=0.9 m treated overhead branches 0.5-0.9 m above the
    27 cm tall rover as impassable walls. Must be <= 0.50 m."""
    _, rp = _params_from_yaml(trav_cfg)
    max_h = float(rp['max_height_above_robot'])
    assert 0.30 <= max_h <= 0.50, f'max_height_above_robot={max_h} m (must clear branches >= 0.5 m)'


def test_global_cost_grid_caps_non_lethal_and_keeps_unseen_unknown():
    """RollingCostGrid must cap slope costs <= 90 (<100 lethal) and leave unseen cells -1."""
    g = RollingCostGrid(length=40.0, resolution=0.20, recenter_dist=5.0, alpha=0.5, max_cost=90)
    g.move_to(0.0, 0.0)
    x = np.linspace(-2.0, 2.0, 81)
    X, Y = np.meshgrid(x, x)
    local = np.full_like(X, -1, dtype=np.int8)
    local[(X > -1.0) & (X < 1.0) & (Y > -1.0) & (Y < 1.0)] = 45
    local[(X > 0.5) & (X < 1.0) & (Y > 0.5) & (Y < 1.0)] = 100
    g.update_from_local(X, Y, local)
    assert int(g.data.max()) <= 90
    assert (g.data == -1).mean() > 0.90  # outside scanned patch stays unknown (-1)


def test_footprint_never_raises_ground_into_fake_step_when_z_est_drifts():
    """Regression for live forest drive stall at (42.0, 1.85) on an 8 deg side-slope:
    when open-loop z_est drifted +5 cm above scanned ground, set_footprint raised real ground
    cells up to z_est, stamping a fake 5 cm lethal curb directly behind the wheels."""
    m = RollingElevationMap(length=4.0, resolution=0.05, alpha=1.0)
    m.move_to(0.0, 0.0)
    X, Y = m.cell_centers()
    slope = np.tan(np.radians(8.0))
    m.h[:, :] = 0.070 + slope * Y
    m.stamp[:, :] = 1.0
    normal = np.array([0.0, -np.sin(np.radians(8.0)), np.cos(np.radians(8.0))])

    z_map = m.ground_height_at(0.0, 0.0, normal=normal)
    assert z_map is not None and abs(z_map - 0.070) < 0.005

    m.set_footprint(0.0, 0.0, 0.123, 0.12, 2.0, normal=normal)
    L = compute_traversability(m.h, TraversabilityParams(step_outlier_trim=1))
    near = (np.hypot(X, Y) < 1.0) & (L['cost'] >= 0)
    assert int((L['cost'][near] >= 100).sum()) == 0, 'set_footprint stamped a fake lethal curb on drifted z_est'


def test_published_terrain_fits_the_robot_tf_with_3d_odometry():
    """Regression (2026-11-24): in RViz the terrain floated above the robot. With 3D odometry the map keeps its own
    drift-free height; what is published must be moved by the drift, so it touches the wheels in the viewer's TF."""
    from ugv_traversability.elevation_map import height_shift
    T_tf, T_fr = np.eye(4), np.eye(4)
    T_tf[2, 3], T_fr[2, 3] = 2.40, 2.65           # odometry has drifted 0.25 m down
    ground_in_map = 2.65
    assert abs(ground_in_map - height_shift(T_fr, T_tf, odom_2d=False) - T_tf[2, 3]) < 1e-9
    assert height_shift(T_fr, T_tf, odom_2d=True) == 0.0       # 2D odometry: no height in the TF to fit
