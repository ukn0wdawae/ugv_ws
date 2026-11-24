"""GPS-denied localization: scan-to-map matching on vertical structure (lidar_localizer_core) and the 3D odometry inputs.

Regression for the mission of 2026-11-24, which drifted 4.7 m and 19 deg on dead reckoning:
  * the localizer stays within 0.3 m of truth while odometry drifts by metres and degrees, also up and down hills;
  * ground is never matched (the rings a lidar paints on the ground move with the robot);
  * a lidar scan spoiled by the depth camera's 5 m clip (about 1 scan in 20 in Gazebo) is thrown away;
  * corrections reach TF at walking speed, not as a jump that looks like a fall;
  * the EKF is 3D and fed by the wheel relay, which never claims a yaw rate or a height from the wheels.
"""
import math
import os

import numpy as np
import pytest
import yaml

from ugv_coqui_behavior import lidar_localizer_core as lc

SENSOR = np.array([0.04, 0.0, 0.138])
RINGS = np.deg2rad(np.linspace(-16, 16, 16))
AZ = np.deg2rad(np.arange(0, 360, 1.0))


def terrain(x, y):
    return 0.35 * np.sin(0.35 * x) + 0.25 * np.cos(0.4 * y + 0.5 * x)


def forest(seed=0, n=90, half=25.0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-half, half, (n, 2))
    xy = xy[np.hypot(xy[:, 0], xy[:, 1]) > 2.0]
    return np.c_[xy, rng.uniform(0.12, 0.35, len(xy))]


def rotation(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def scan(pose, trunks, rng=None):
    """Points of a 16-ring lidar in the robot base frame; trunks stand on the terrain, the ground is bumpy."""
    x, y, z, roll, pitch, yaw = pose
    R = rotation(roll, pitch, yaw)
    p0 = np.array([x, y, z]) + R @ SENSOR
    el, az = np.meshgrid(RINGS, AZ, indexing='ij')
    d = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1).reshape(-1, 3) @ R.T
    best = np.full(len(d), np.inf)
    h = np.hypot(d[:, 0], d[:, 1])
    for tx, ty, tr in trunks:
        ox, oy = p0[0] - tx, p0[1] - ty
        b = (ox * d[:, 0] + oy * d[:, 1]) / np.maximum(h, 1e-9)
        c = ox * ox + oy * oy - tr * tr
        disc = b * b - c
        t = (-b - np.sqrt(np.maximum(disc, 0))) / np.maximum(h, 1e-9)
        hit = (disc > 0) & (t > 0.3)
        zc = p0[2] + t * d[:, 2]
        ground = terrain(p0[0] + t * d[:, 0], p0[1] + t * d[:, 1])
        ok = hit & (zc > ground) & (zc < ground + 2.5) & (t < best)
        best[ok] = t[ok]
    for t in np.arange(0.4, 20.0, 0.25):     # coarse march onto the ground
        g = (best > t) & (p0[2] + t * d[:, 2] < terrain(p0[0] + t * d[:, 0], p0[1] + t * d[:, 1]))
        best[g] = np.minimum(best[g], t)
    r = best[np.isfinite(best) & (best < 25)]
    dd = d[np.isfinite(best) & (best < 25)]
    if rng is not None:
        r = r + rng.normal(0, 0.008, len(r))
    return ((p0 + dd * r[:, None]) - np.array([x, y, z])) @ R


def matrix(x, y, z, roll, pitch, yaw):
    T = np.eye(4)
    T[:3, :3] = rotation(roll, pitch, yaw)
    T[:3, 3] = (x, y, z)
    return T


def drive(n=110, step=0.35):
    """A loop through the forest: (x, y, z, roll, pitch, yaw) on the terrain."""
    out, x, y, yaw = [], 0.0, 0.0, 0.0
    for k in range(n):
        yaw += 0.06 * math.sin(k / 9.0) + 2 * math.pi / n
        x, y = x + step * math.cos(yaw), y + step * math.sin(yaw)
        e = 0.2
        gx = (terrain(x + e, y) - terrain(x - e, y)) / (2 * e)
        gy = (terrain(x, y + e) - terrain(x, y - e)) / (2 * e)
        pitch = -math.atan(gx * math.cos(yaw) + gy * math.sin(yaw))
        roll = math.atan(-gx * math.sin(yaw) + gy * math.cos(yaw))
        out.append((x, y, float(terrain(x, y)), roll, pitch, yaw))
    return out


def run_localizer(seed=0, yaw_drift=2e-3, scale=0.96, z_drift=0.004):
    rng = np.random.default_rng(seed)
    trunks, poses = forest(seed), drive()
    loc = lc.StructureLocalizer((poses[0][0], poses[0][1], poses[0][2], poses[0][5]))
    err, odo, zerr = [], [], []
    ox, oy, oyaw, oz = 0.0, 0.0, 0.0, 0.0
    prev = poses[0]
    for k, p in enumerate(poses):
        if k:
            dx, dy = p[0] - prev[0], p[1] - prev[1]
            c, s = math.cos(prev[5]), math.sin(prev[5])
            fwd, side = c * dx + s * dy, -s * dx + c * dy
            dyaw = lc.wrap(p[5] - prev[5]) + yaw_drift + rng.normal(0, 1e-3)
            mid = oyaw + 0.5 * dyaw
            ox += scale * fwd * math.cos(mid) - side * math.sin(mid)
            oy += scale * fwd * math.sin(mid) + side * math.cos(mid)
            oyaw += dyaw
            oz += (p[2] - prev[2]) + z_drift
        prev = p
        loc.process(scan(p, trunks, rng), matrix(ox, oy, oz, p[3], p[4], oyaw))
        x, y, z, th = loc.to_map(matrix(ox, oy, oz, p[3], p[4], oyaw))
        err.append(math.hypot(x - p[0], y - p[1]))
        zerr.append(abs(z - p[2]))
        c0, s0 = math.cos(poses[0][5]), math.sin(poses[0][5])
        mx, my = c0 * ox - s0 * oy + poses[0][0], s0 * ox + c0 * oy + poses[0][1]
        odo.append(math.hypot(mx - p[0], my - p[1]))
    return np.array(err), np.array(odo), np.array(zerr), loc


@pytest.mark.parametrize('seed', [0, 1])
def test_localizer_holds_while_odometry_drifts(seed):
    """Gyro drift of 2 deg/min, 4 % wheel slip, height drift 4 mm per step (the recorded robot: 1 deg, 3 %)."""
    err, odo, zerr, loc = run_localizer(seed, yaw_drift=5e-4, scale=0.96)
    assert odo.max() > 0.6, f'the drift in this test is too small to mean anything ({odo.max():.2f} m)'
    assert err.max() < 0.45 and err.mean() < 0.22, (err.mean(), err.max())
    assert zerr.max() < 0.25


def test_localizer_recovers_after_a_bad_stretch():
    """8 deg/min of heading drift: it lags by a few tenths, then coming back to where it started removes the drift."""
    err, odo, _, _ = run_localizer(0, yaw_drift=2e-3, scale=0.96)
    assert odo.max() > 1.5 and err.max() < 1.0
    assert err[-5:].mean() < 0.3, err[-5:]


def test_ground_is_not_structure():
    """A scan of open ground, level and on a 15 degree slope, contains no structure; a trunk does."""
    for pitch in (0.0, math.radians(15)):
        rng = np.random.default_rng(0)
        g = []
        for el in RINGS[RINGS < -0.03]:
            r = 0.138 / math.sin(-el) if pitch == 0 else 0.138 / math.sin(-el) * 1.2
            g.append(np.c_[r * np.cos(AZ), r * np.sin(AZ), np.full(len(AZ), -0.138) + 0.0 * AZ])
        pts = np.concatenate(g) + rng.normal(0, 0.005, (sum(len(a) for a in g), 3))
        pts[:, 2] += pts[:, 0] * math.tan(pitch)
        m = lc.structure_mask(pts, np.linalg.norm(pts, axis=1))
        assert m.mean() < 0.02, (pitch, m.mean())
    trunk = np.array([[3.0, 0.0, z] for z in np.arange(-0.1, 1.5, 0.1)])
    assert lc.structure_mask(trunk, np.linalg.norm(trunk, axis=1)).sum() >= 10


def test_scan_with_a_range_shell_is_rejected():
    """The Gazebo fault: beams above the ground all return 4.75 m (the depth camera's far clip)."""
    rng = np.random.default_rng(0)
    good = scan((0, 0, float(terrain(0, 0)), 0, 0, 0), forest(0), rng)
    assert not lc.has_range_shell(np.linalg.norm(good, axis=1))
    shell = rng.normal(0, 1, (3000, 3))
    shell = shell / np.linalg.norm(shell, axis=1)[:, None] * 4.75
    bad = np.r_[good, shell]
    assert lc.has_range_shell(np.linalg.norm(bad, axis=1))
    loc = lc.StructureLocalizer((0, 0, 0, 0))
    st = loc.process(bad, np.eye(4))
    assert st['fault'] and loc.hits is None        # not matched, not added to the map


def test_correction_reaches_tf_at_walking_speed():
    s = lc.CorrectionSmoother((0, 0, 0, 0), max_mps=0.25, max_zps=0.1, max_dps=6.0)
    pivot = (10.0, 4.0)
    prev = None
    for _ in range(40):
        c = s.step((0.8, -0.3, 0.2, math.radians(4)), 0.1, pivot)
        cs, sn = math.cos(c[3]), math.sin(c[3])
        robot = np.array([cs * pivot[0] - sn * pivot[1] + c[0], sn * pivot[0] + cs * pivot[1] + c[1], c[2]])
        if prev is not None:
            assert np.hypot(*(robot - prev)[:2]) <= 0.25 * 0.1 + 1e-9, 'robot moved faster than it may'
            assert abs(robot[2] - prev[2]) <= 0.1 * 0.1 + 1e-9
        prev = robot
    assert np.allclose(s.c, (0.8, -0.3, 0.2, math.radians(4)), atol=1e-6)


def test_correction_is_not_seen_as_a_fall_by_the_safety_monitor():
    """Worst case: a 1 m correction after being lost, published through the smoother while the robot stands still."""
    from ugv_coqui_behavior import slope_recovery as sr
    mon = sr.SafetyMonitor() if hasattr(sr, 'SafetyMonitor') else None
    if mon is None:
        pytest.skip('SafetyMonitor not available')
    s = lc.CorrectionSmoother((0, 0, 0, 0))
    for k in range(80):
        c = s.step((0.8, 0.6, 0.3, 0.1), 0.1)
        code, _ = mon.update(0.1 * k, pose_xyz=(c[0], c[1], c[2]), tilt_deg=0.0, imu_wx=0.0, imu_wy=0.0)
        assert code is None, (k, code)


# ----------------------------------------------------------------------------------------- odometry
def test_ekf_is_3d_and_uses_no_yaw_or_height_from_the_wheels(ws_root):
    cfg = os.path.join(ws_root, 'src', 'ugv_main', 'ugv_coqui_behavior', 'config')
    e = yaml.safe_load(open(os.path.join(cfg, 'ekf_odom.yaml')))['ekf_filter_node']['ros__parameters']
    assert e['two_d_mode'] is False
    assert e['odom0'] == '/wheel/odom_3d'
    assert e['odom0_config'][6:9] == [True, True, True] and not any(e['odom0_config'][:6] + e['odom0_config'][9:])
    assert e['imu0_config'][3:5] == [True, True] and e['imu0_config'][5] is False     # Madgwick's yaw is unobserved
    assert e['imu0_config'][9:12] == [True, True, True]
    assert yaml.safe_load(open(os.path.join(cfg, 'imu_filter.yaml')))['imu_filter_madgwick']['ros__parameters']['gain'] <= 0.03


def test_wheel_relay_twist_is_honest(monkeypatch):
    import sys
    import types
    for mod in ('rclpy', 'rclpy.executors', 'rclpy.node', 'rclpy.qos', 'nav_msgs', 'nav_msgs.msg'):
        if mod not in sys.modules:
            m = types.ModuleType(mod)
            m.Node, m.Odometry, m.ExternalShutdownException, m.qos_profile_sensor_data = object, object, Exception, None
            monkeypatch.setitem(sys.modules, mod, m)
    from ugv_coqui_behavior.wheel_odom_relay import BIG, body_twist_covariance
    slow, fast, turn = body_twist_covariance(0.0, 0.0), body_twist_covariance(0.5, 0.0), body_twist_covariance(0.5, 1.0)
    assert slow[0] < fast[0] < turn[0]            # slips more the faster and the harder it turns
    assert slow[2] < 1e-3                         # the wheels stay on the ground
    assert slow[3] == slow[4] == slow[5] == BIG   # no rates from the wheels


def test_default_localization_is_gps_free_lidar(ws_root):
    p = os.path.join(ws_root, 'src', 'ugv_main', 'ugv_coqui_behavior', 'launch', 'coqui_survey.launch.py')
    txt = open(p).read()
    assert "'localization', default_value='lidar'" in txt
    assert "executable='lidar_localizer'" in txt and "executable='wheel_odom_relay'" in txt


def test_no_camera_clips_the_lidar_in_simulation(ws_root):
    """A camera's far clip leaks into gpu_ray scans (fake walls at 0.95 x far): none may be shorter than the lidar range."""
    import re
    txt = open(os.path.join(ws_root, 'src', 'ugv_main', 'ugv_gazebo', 'models', 'ugv_pro_audio', 'model.sdf')).read()
    far = [float(v) for v in re.findall(r'<far>([0-9.]+)</far>', txt)]
    lidar = re.search(r"<sensor name='lidar_3d'.*?<range>.*?<max>([0-9.]+)</max>", txt, re.S)
    assert far and lidar and min(far) >= float(lidar.group(1)), (far, lidar and lidar.group(1))
