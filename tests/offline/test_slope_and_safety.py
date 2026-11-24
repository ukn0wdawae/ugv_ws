"""Pooled slope, rollover, fall, capsize, escape-maneuver, and stale-sensor regression tests.

Covers historical runs: slope_test_1, slope_test_2, slope_test_3 (21:01 9.6 m fall @ 2.4 m/s),
bush_slope_test, and continuous safety monitoring outside Nav2 goals.
"""
import math
import numpy as np
import pytest

from ugv_coqui_behavior import slope_recovery as sr
from ugv_coqui_behavior import survey_planning as sp


def _rpy_to_quat(roll, pitch, yaw=0.0):
    cr, sr_ = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp_ = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr_ * cp * cy - cr * sp_ * sy,
        cr * sp_ * cy + sr_ * cp * sy,
        cr * cp * sy - sr_ * sp_ * cy,
        cr * cp * cy + sr_ * sp_ * sy,
    )


def _make_grid(size=20.0, origin=(-10.0, -10.0), res=0.05, fill=0):
    n = int(round(size / res))
    return sp.Grid(origin[0], origin[1], res, np.full((n, n), fill, dtype=np.int16))


def test_safety_monitor_catches_2101_fall_and_capsize():
    """Regression for 21:01 slope_test_3: rover dropped 9.6 m in 16 s at up to 2.4 m/s while Nav2
    reported 'Goal succeeded'. SafetyMonitor must catch BOTH the drop and capsize immediately,
    even when no Nav2 goal is active."""
    # 1) Replay a 2.4 m/s downhill tumble dropping 2.0 m in 1.2 s
    mon = sr.SafetyMonitor()
    triggered = None
    for k in range(30):
        t = k * 0.1
        z = 12.37 - 1.6 * t
        x = -11.20 + 1.8 * t
        code, detail = mon.update(t, pose_xyz=(x, 10.5, z), tilt_deg=35.0, imu_wx=0.4, imu_wy=0.5)
        if code is not None:
            triggered = (t, code, detail)
            break
    assert triggered is not None, 'SafetyMonitor missed the 21:01 slope fall!'
    assert triggered[0] < 1.5, f'fall caught too late ({triggered[0]:.2f} s)'
    assert triggered[1] in ('fall_drop', 'fall_speed', 'runaway_tumble')

    # 2) Capsize (> 70 deg) must ALWAYS trigger abort — never ignored!
    mon2 = sr.SafetyMonitor()
    cap = None
    for k in range(20):
        t = k * 0.05
        code, _ = mon2.update(t, pose_xyz=(0.0, 0.0, 2.5), tilt_deg=95.0)
        if code is not None:
            cap = (t, code)
            break
    assert cap is not None and cap[1] == 'capsized' and cap[0] <= 0.6

    # 3) SlopeStallDetector also reports 'capsized' even when active=False (e.g. during LISTEN/PHOTO)
    det = sr.SlopeStallDetector()
    res = None
    for k in range(20):
        res = det.update(k * 0.05, 0.0, 0.0, 0.0, tilt_deg=85.0,
                         imu_wz=0.0, cmd_vx=0.0, cmd_wz=0.0, active=False)
        if res is not None:
            break
    assert res == 'capsized' and det.is_capsized() and not det.is_steep()


def test_safety_monitor_does_not_false_trigger_on_normal_hill_climb():
    """Steady 0.55 m/s climb/descent on a 18 deg switchback must NEVER trigger fall or capsize."""
    mon = sr.SafetyMonitor()
    for k in range(200):  # 20 s
        t = k * 0.1
        x = 0.55 * t
        z = 12.0 - 0.55 * math.sin(math.radians(18.0)) * t  # ~0.17 m/s controlled descent
        code, detail = mon.update(t, pose_xyz=(x, 0.0, z), tilt_deg=18.0, imu_wx=0.1, imu_wy=0.1)
        assert code is None, f'false safety trigger on normal hill drive: {code} ({detail})'


def test_safety_monitor_aborts_on_stale_sensors():
    """Regression: if terrain_traversability, lidar, IMU, or TF dies, mission must stop instead of
    driving blindly on old data."""
    mon = sr.SafetyMonitor({'sensor_stale_sec': 5.0})
    for name in ('imu', 'costmap', 'traversability', 'tf'):
        mon.record_sensor(name, 10.0)
    assert mon.check_stale_sensors(12.0) is None
    # Keep imu/costmap/tf fresh, let traversability die
    for name in ('imu', 'costmap', 'tf'):
        mon.record_sensor(name, 16.0)
    stale = mon.check_stale_sensors(16.0)
    assert stale is not None and 'traversability' in stale


def test_escape_maneuver_bounded_blocks_unknown_and_enforces_area(survey_cfg):
    """Regression for risky escape maneuver:
    - max distance <= 8 m (was 30 m!)
    - unknown ground (-1) is BLOCKED (not treated as free!)
    - survey_area is strictly enforced even when obs_grid is in odom frame (obs_to_map != identity)."""
    params = survey_cfg['coqui_survey']['ros__parameters']
    assert float(params['slope_descent_max_dist_m']) <= 8.0
    assert float(params['slope_descent_max_sec']) <= 45.0
    assert params['slope_descent_allow_unknown'] is False

    # 1) Unknown drop-off straight downhill (+X) -> must NOT drive straight into unknown (-1)!
    g = _make_grid(size=10.0, origin=(-5.0, -5.0), fill=0)
    yy, xx = np.mgrid[0:g.height, 0:g.width]
    X = g.origin_x + (xx + 0.5) * g.resolution
    g.data[X > 0.35] = -1  # unseen cliff / drop-off right ahead!
    s_down = sr.imu_slope_info(*_rpy_to_quat(0.0, math.radians(28.0)))  # downhill = +X
    vx, wz, sign, dbg = sr.choose_descent_cmd(s_down, (0.0, 0.0, 0.0), g)
    # Must either block or reverse/skirt away from the unknown zone (never drive straight forward into X>0.35)
    assert dbg['blocked'] or vx <= 0.0 or abs(wz) > 0.3

    # 2) Survey area boundary enforced even when obs_grid is in shifted/rotated odom frame!
    g_open = _make_grid(size=20.0, origin=(-10.0, -10.0), fill=0)
    # odom origin is shifted by (+40, 0) in map frame -> robot at odom (8.5, 0) is at map (48.5, 0),
    # right next to survey_area xmax=50.0 (margin=1.0 -> max allowed map x=49.0)!
    area = [18.0, 50.0, -15.0, 15.0]
    vx2, wz2, _, dbg2 = sr.choose_descent_cmd(
        s_down, (8.5, 0.0, 0.0), g_open, survey_area=area, area_margin_m=1.0, obs_to_map=(40.0, 0.0, 0.0))
    # Straight forward (+X) would exit map x=49.0 within 0.5 m -> must not command straight +X exit
    assert dbg2['blocked'] or vx2 <= 0.0 or abs(wz2) > 0.35


@pytest.mark.parametrize('init_yaw_deg', [-90.0, 90.0, -35.0, 135.0])
def test_bush_slope_closed_loop_descent_weaves_boulders_safely(init_yaw_deg):
    """Pooled bush_slope_test: 30 deg hillside with boulders in the fall line; controlled descent
    reaches flat ground safely within <= 6 m-equivalent envelope without colliding or spinning in place."""
    g = _make_grid(size=24.0, origin=(-12.0, -12.0), fill=0)
    yy, xx = np.mgrid[0:g.height, 0:g.width]
    X = g.origin_x + (xx + 0.5) * g.resolution
    Y = g.origin_y + (yy + 0.5) * g.resolution
    boulders = [(0.0, 2.6, 0.65), (-0.7, 0.4, 0.55), (0.8, -1.0, 0.60)]
    for bx, by, br in boulders:
        g.data[(X - bx) ** 2 + (Y - by) ** 2 <= br * br] = 100

    x, y, yaw = 0.0, 5.0, math.radians(init_yaw_deg)
    dt, sign, prev_wz = 0.1, None, 0.0
    min_clr = 1e9
    reached_flat = False
    for _ in range(550):
        slope_deg = 30.0 if y > -2.5 else max(5.0, 30.0 * (y + 3.5))
        if slope_deg <= 14.0:
            reached_flat = True
            break
        sinfo = {'tilt_deg': slope_deg, 'roll_deg': 0.0, 'pitch_deg': 0.0,
                 'down_x': -math.sin(yaw), 'down_y': -math.cos(yaw), 'down_angle': 0.0}
        vx, wz, sign, dbg = sr.choose_descent_cmd(sinfo, (x, y, yaw), g, prev_sign=sign, prev_wz=prev_wz)
        assert not dbg['blocked'] and abs(vx) >= 0.11
        prev_wz = wz
        x += vx * math.cos(yaw) * dt
        y += vx * math.sin(yaw) * dt
        yaw = sp.wrap(yaw + wz * dt)
        for bx, by, br in boulders:
            min_clr = min(min_clr, math.hypot(x - bx, y - by) - br)
    assert reached_flat
    assert min_clr > 0.14
