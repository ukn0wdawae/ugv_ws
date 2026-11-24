"""Offline unit & closed-loop tests for IMU steep-slope stall detection and boulder-avoiding descent."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ugv_coqui_behavior import slope_recovery as sr  # noqa: E402
from ugv_coqui_behavior import survey_planning as sp  # noqa: E402
from ugv_coqui_behavior import survey_viz as sv  # noqa: E402


def rpy_to_quat(roll, pitch, yaw=0.0):
    cr, sr_ = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp_ = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    qx = sr_ * cp * cy - cr * sp_ * sy
    qy = cr * sp_ * cy + sr_ * cp * sy
    qz = cr * cp * sy - sr_ * sp_ * cy
    qw = cr * cp * cy + sr_ * sp_ * sy
    return qx, qy, qz, qw


def make_grid(size=20.0, origin=(-10.0, -10.0), res=0.05):
    n = int(round(size / res))
    return sp.Grid(origin[0], origin[1], res, np.zeros((n, n), dtype=np.int16))


def add_boulder(g, cx, cy, r, val=100):
    yy, xx = np.mgrid[0:g.height, 0:g.width]
    X = g.origin_x + (xx + 0.5) * g.resolution
    Y = g.origin_y + (yy + 0.5) * g.resolution
    g.data[(X - cx) ** 2 + (Y - cy) ** 2 <= r * r] = val


# ----------------------------------------------------------------------------- IMU slope geometry
def test_imu_slope_info_directions_and_accel_fallback():
    # Nose tilted DOWN 30 deg (+pitch around +Y) -> downhill is forward (+X)
    info = sr.imu_slope_info(*rpy_to_quat(0.0, math.radians(30.0), 0.7))
    assert info['tilt_deg'] == pytest.approx(30.0, abs=0.1)
    assert info['down_x'] == pytest.approx(1.0, abs=1e-3)
    assert abs(info['down_y']) < 1e-3

    # Nose tilted UP 28 deg (-pitch around +Y) -> downhill is backward (-X)
    info_up = sr.imu_slope_info(*rpy_to_quat(0.0, math.radians(-28.0), -1.2))
    assert info_up['tilt_deg'] == pytest.approx(28.0, abs=0.1)
    assert info_up['down_x'] == pytest.approx(-1.0, abs=1e-3)

    # Right side down / left side up (+roll 25 deg around +X) -> downhill is right (-Y)
    info_roll = sr.imu_slope_info(*rpy_to_quat(math.radians(25.0), 0.0, 0.3))
    assert info_roll['tilt_deg'] == pytest.approx(25.0, abs=0.1)
    assert info_roll['down_y'] == pytest.approx(-1.0, abs=1e-3)

    # Accelerometer-only fallback (zero quaternion): specific force pointing back-and-up means nose down
    s = math.sin(math.radians(30.0))
    c = math.cos(math.radians(30.0))
    info_acc = sr.imu_slope_info(0, 0, 0, 0, ax=-9.81 * s, ay=0.0, az=9.81 * c)
    assert info_acc['tilt_deg'] == pytest.approx(30.0, abs=0.1)
    assert info_acc['down_x'] == pytest.approx(1.0, abs=1e-3)


# ----------------------------------------------------------------------------- IMU stall detector
def test_stall_detector_catches_skid_steer_turn_stall_on_steep_slope():
    det = sr.SlopeStallDetector()
    reason = None
    for k in range(80):  # 4.0 s at 20 Hz
        t = k * 0.05
        # Tilted 23 deg (below rollover_guard 28 deg), Nav2 commands spin wz=1.2 rad/s, robot slips (imu_wz ~ 0.05)
        reason = det.update(t, x=0.0, y=0.0, yaw=0.01 * math.sin(k), tilt_deg=23.0,
                            imu_wz=0.05, cmd_vx=-0.05, cmd_wz=1.2, active=True)
        if reason is not None:
            break
    assert reason == 'turn_stall'
    assert t <= 3.2   # catches quickly (~2.5 s) instead of grinding for a minute


def test_stall_detector_does_not_false_trigger_during_normal_hill_climb_or_flat_spin():
    det = sr.SlopeStallDetector()
    # 1) Normal steady climb at 0.45 m/s up a 24 deg hill switchback for 8 s -> never triggers
    for k in range(160):
        t = k * 0.05
        res = det.update(t, x=0.45 * t, y=0.0, yaw=0.0, tilt_deg=24.0,
                         imu_wz=0.15, cmd_vx=0.45, cmd_wz=0.20, active=True)
        assert res is None
    # 2) In-place spin on flat ground (6 deg) -> never triggers
    det.reset_motion()
    for k in range(120):
        t = 10.0 + k * 0.05
        res = det.update(t, x=3.6, y=0.0, yaw=0.0, tilt_deg=6.0,
                         imu_wz=0.0, cmd_vx=0.0, cmd_wz=1.2, active=True)
        assert res is None
    # 3) Parked/listening on a 26 deg slope (active=False) -> never triggers
    for k in range(120):
        t = 20.0 + k * 0.05
        res = det.update(t, x=3.6, y=0.0, yaw=0.0, tilt_deg=26.0,
                         imu_wz=0.0, cmd_vx=0.0, cmd_wz=0.0, active=False)
        assert res is None
    # 4) Capsized / upside down (>= 65 deg) -> returns 'capsized' (never ignored!), is_steep() is False
    det.reset_motion()
    cap = None
    for k in range(30):
        t = 30.0 + k * 0.05
        cap = det.update(t, x=0.0, y=0.0, yaw=0.0, tilt_deg=115.0,
                         imu_wz=0.0, cmd_vx=0.0, cmd_wz=1.0, active=True)
        if cap is not None:
            break
    assert cap == 'capsized'
    assert det.is_capsized()
    assert not det.is_steep()


# --------------------------------------------------------- obstacle-avoiding slow downhill descent
def test_descent_nose_down_goes_forward_and_nose_up_reverses_without_spinning():
    g = make_grid()
    # Nose down (+pitch 30 deg): forward slow descent, no in-place spin
    s_down = sr.imu_slope_info(*rpy_to_quat(0.0, math.radians(30.0)))
    vx, wz, sign, dbg = sr.choose_descent_cmd(s_down, (0.0, 0.0, 0.0), g)
    assert sign == +1 and vx > 0.15 and abs(wz) < 0.15 and not dbg['blocked']

    # Nose up (-pitch 30 deg): reverse slow descent (tail first!), never tries to spin 180 deg
    s_up = sr.imu_slope_info(*rpy_to_quat(0.0, math.radians(-30.0)))
    vx2, wz2, sign2, dbg2 = sr.choose_descent_cmd(s_up, (0.0, 0.0, 0.0), g)
    assert sign2 == -1 and vx2 < -0.15 and abs(wz2) < 0.15 and not dbg2['blocked']


@pytest.mark.parametrize('init_yaw_deg,expected_sign', [
    (-90.0, +1),   # facing straight downhill (-Y) -> slow forward descent
    (90.0, -1),    # facing straight uphill (+Y) -> slow reverse descent (tail downhill)
    (-35.0, +1),   # angled diagonally downhill -> arcs smoothly down fall line
    (135.0, -1),   # angled diagonally uphill -> reverses smoothly down fall line
])
def test_closed_loop_descent_weaves_around_boulders_to_flat_ground(init_yaw_deg, expected_sign):
    """Simulate a 30 deg hillside sloping toward -Y from y=+5 down to flat ground at y<=-3,
    with boulders directly in the fall line. The controller must descend slowly, avoid all
    boulders with clearance, and reach flat ground cleanly."""
    g = make_grid(size=24.0, origin=(-12.0, -12.0), res=0.05)
    boulders = [(0.0, 2.6, 0.65), (-0.7, 0.4, 0.55), (0.8, -1.0, 0.60)]
    for bx, by, br in boulders:
        add_boulder(g, bx, by, br)

    x, y, yaw = 0.0, 5.0, math.radians(init_yaw_deg)
    dt = 0.1
    sign, prev_wz = None, 0.0
    first_sign = None
    min_boulder_clearance = 1e9
    reached_flat = False

    for _ in range(550):  # up to 55 s
        # World slope: 30 deg toward -Y for y > -3.0, tapering smoothly to 5 deg flat ground
        slope_deg = 30.0 if y > -2.5 else max(5.0, 30.0 * (y + 3.5))
        if slope_deg <= 14.0:
            reached_flat = True
            break
        # Downhill world vector is (0, -1); express in body frame
        down_bx = -math.sin(yaw)
        down_by = -math.cos(yaw)
        sinfo = {
            'tilt_deg': slope_deg,
            'roll_deg': 0.0,
            'pitch_deg': 0.0,
            'down_x': down_bx,
            'down_y': down_by,
            'down_angle': math.atan2(down_by, down_bx),
        }
        vx, wz, sign, dbg = sr.choose_descent_cmd(sinfo, (x, y, yaw), g, prev_sign=sign, prev_wz=prev_wz)
        if first_sign is None:
            first_sign = sign
        assert not dbg['blocked']
        assert abs(vx) >= 0.11  # always moving steadily, never spinning in place on the slope
        prev_wz = wz
        x += vx * math.cos(yaw) * dt
        y += vx * math.sin(yaw) * dt
        yaw = sp.wrap(yaw + wz * dt)
        for bx, by, br in boulders:
            min_boulder_clearance = min(min_boulder_clearance, math.hypot(x - bx, y - by) - br)

    assert first_sign == expected_sign
    assert reached_flat, f'did not reach flat ground; ended at ({x:.2f}, {y:.2f})'
    assert min_boulder_clearance > 0.14, f'collided with boulder (min clearance {min_boulder_clearance:.2f} m)'


def test_hud_formats_slope_descent_state():
    lines = sv.hud_text('SLOPE_DESCENT', {'tilt_deg': 27.4, 'direction': 'reverse'}, 2, 185.0, 10, 70)
    assert any('SLOPE_DESCENT' in line and '27 deg' in line and 'reverse' in line for line in lines)
    assert all(len(line) <= sv.HUD_LINE_CHARS for line in lines)


def test_rollover_guard_catches_side_hill_and_steep_climb_without_false_trigger():
    det = sr.SlopeStallDetector()
    # Normal 21 deg straight climb with moderate 12 deg roll -> never triggers rollover guard
    for k in range(60):
        t = k * 0.05
        res = det.update(t, x=0.45 * t, y=0.0, yaw=0.0, tilt_deg=21.0,
                         imu_wz=0.1, cmd_vx=0.45, cmd_wz=0.1, active=True, roll_deg=12.0)
        assert res is None
    # Brief 0.15 s roll spike (bump) -> ignored
    for k in range(3):
        t = 3.0 + k * 0.05
        assert det.update(t, x=1.5, y=0.0, yaw=0.0, tilt_deg=25.0,
                          imu_wz=0.0, cmd_vx=0.3, cmd_wz=0.2, active=True, roll_deg=25.0) is None
    # Sustained dangerous 25 deg side-hill roll -> immediately fires rollover_guard within < 1 s
    det.reset_motion()
    hit = None
    for k in range(30):
        t = 5.0 + k * 0.05
        hit = det.update(t, x=1.5 + 0.2 * k * 0.05, y=0.0, yaw=0.0, tilt_deg=26.0,
                         imu_wz=0.2, cmd_vx=0.3, cmd_wz=0.4, active=True, roll_deg=25.0)
        if hit is not None:
            break
    assert hit == 'rollover_guard'
    assert t - 5.0 < 1.0
