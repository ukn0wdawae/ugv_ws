"""IMU-guided steep-slope stall detection and obstacle-avoiding slow descent (ROS-free).

When a skid-steer rover gets onto a steep slope and Nav2 tries to spin or turn back uphill, the
tracks/wheels lose yaw authority and stall. This module provides two pure, unit-testable pieces:

  1. SlopeStallDetector  Uses the IMU attitude + yaw rate against commanded velocities and pose
                         progress to recognize when the robot is on a steep slope and cannot turn
                         or make progress.
  2. choose_descent_cmd  Computes slow, constant-curvature downhill (v, w) arcs aligned with the
                         IMU gravity fall line (forward when nose-down, reverse when nose-up —
                         never spinning in place on the slope) while steering around physical
                         obstacles (rocks, trees, bushes) in the non-slope obstacle grid.
"""
from collections import deque
import math

import numpy as np

from .survey_planning import wrap

SLOPE_DEFAULTS = {
    'slope_stall_deg': 18.0,            # IMU tilt above which turn-stall detection & downhill escape activate
    'slope_steep_deg': 22.0,            # IMU tilt above which general no-progress stall activates
    'slope_rollover_roll_deg': 22.0,    # side-hill |roll| threshold for proactive anti-rollover abort
    'slope_rollover_tilt_deg': 28.0,    # total inclination threshold for proactive anti-rollover abort
    'slope_rollover_confirm_sec': 0.40, # hold time before rollover guard fires (filters brief bumps)
    'slope_max_drivable_deg': 65.0,     # above this the rover has capsized (wheels off ground -> abort mission!)
    'slope_capsize_confirm_sec': 0.40,  # hold time before capsize abort fires
    'slope_flat_deg': 14.0,             # IMU tilt below this (held for slope_flat_confirm_sec) = flat ground
    'slope_flat_confirm_sec': 1.0,
    'slope_turn_stall_sec': 2.5,        # blocked skid-steer turn duration on slope before taking over
    'slope_stall_sec': 4.0,             # general no-progress duration on steep slope before taking over
    'slope_turn_cmd_min': 0.30,         # rad/s minimum |cmd_wz| considered a turn attempt
    'slope_turn_ratio_max': 0.25,       # |imu_wz| / |cmd_wz| below this = turn stalled
    'slope_progress_min_m': 0.22,       # XY progress below this over the stall window = stuck
    'slope_tilt_alpha': 0.25,           # EMA smoothing on IMU tilt
    'slope_descent_speed_max': 0.24,    # m/s slow open-slope descent speed
    'slope_descent_speed_min': 0.12,    # m/s cautious speed while weaving past obstacles
    'slope_descent_max_wz': 0.85,       # rad/s max gentle arc steering rate during descent
    'slope_descent_clearance_m': 0.22,  # footprint radius checked against non-slope obstacle grid
    'slope_descent_stop_dist_m': 0.45,  # minimum clear arc distance required before hitting an obstacle
    'slope_descent_lookahead_m': 1.60,  # arc rollout distance for boulder avoidance
    'slope_descent_lethal_cost': 70,    # obstacle_grid cost >= this is treated as a physical obstacle
    'slope_descent_allow_unknown': False,  # never blindly drive downhill into unseen ground / drop-offs
    'slope_descent_max_sec': 35.0,      # safety time cap for one controlled descent
    'slope_descent_max_dist_m': 6.0,    # safety distance cap for one controlled descent
    # Continuous mission-wide safety monitor (fall, capsize, tumble, stale sensors)
    'fall_drop_m': 1.5,                 # vertical drop inside fall_window_sec that triggers fall abort
    'fall_window_sec': 4.0,
    'fall_vz_mps': 1.2,                 # sustained downward vertical speed (m/s) -> fall abort
    'runaway_speed_mps': 1.5,           # sustained 3D speed > 2x max cmd_vel -> tumble/slide abort
    'tumble_rate_rps': 2.5,             # sustained |wx| or |wy| (rad/s) -> rolling tumble abort
    'fall_confirm_sec': 0.40,
    'sensor_stale_sec': 5.0,            # stop & abort if critical sensor stream stops for this long
}


def imu_slope_info(qx, qy, qz, qw, ax=0.0, ay=0.0, az=0.0):
    """Extract total terrain inclination and body-frame downhill direction from an IMU reading.

    Uses the fused orientation quaternion (body -> world ENU, +X forward, +Y left, +Z up) when
    non-zero; falls back to normalized linear acceleration (stationary specific force points UP).

    Returns a dict:
      tilt_deg    total inclination from horizontal in [0, 180] deg
      roll_deg    REP-103 roll (+ = right side down / left side up)
      pitch_deg   REP-103 pitch (+ = nose down)
      down_x, down_y   unit 2D downhill vector in robot body XY frame (0, 0 when flat)
      down_angle       atan2(down_y, down_x) in (-pi, pi] (0 = forward, +-pi = backward)
      up_body          (ux, uy, uz) world-up unit vector in body frame
    """
    qn = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if qn > 1e-6:
        x, y, z, w = qx / qn, qy / qn, qz / qn, qw / qn
        # 3rd row of body-to-world rotation matrix R_wb = world +Z expressed in body frame
        ux = 2.0 * (x * z - y * w)
        uy = 2.0 * (y * z + x * w)
        uz = 1.0 - 2.0 * (x * x + y * y)
    else:
        an = math.sqrt(ax * ax + ay * ay + az * az)
        if an < 1e-6:
            return {'tilt_deg': 0.0, 'roll_deg': 0.0, 'pitch_deg': 0.0,
                    'down_x': 0.0, 'down_y': 0.0, 'down_angle': 0.0, 'up_body': (0.0, 0.0, 1.0)}
        ux, uy, uz = ax / an, ay / an, az / an

    uz_c = max(-1.0, min(1.0, uz))
    tilt_deg = math.degrees(math.acos(uz_c))
    roll_deg = math.degrees(math.atan2(uy, uz))
    pitch_deg = math.degrees(math.atan2(-ux, math.hypot(uy, uz)))
    horiz = math.hypot(ux, uy)
    if horiz > 1e-4:
        down_x, down_y = -ux / horiz, -uy / horiz
        down_angle = math.atan2(down_y, down_x)
    else:
        down_x, down_y, down_angle = 0.0, 0.0, 0.0
    return {
        'tilt_deg': tilt_deg,
        'roll_deg': roll_deg,
        'pitch_deg': pitch_deg,
        'down_x': down_x,
        'down_y': down_y,
        'down_angle': down_angle,
        'up_body': (ux, uy, uz),
    }


class SlopeStallDetector:
    """Detects when the robot is on a steep slope and cannot skid-steer turn or make progress."""

    def __init__(self, cfg=None):
        self.cfg = dict(SLOPE_DEFAULTS)
        if cfg:
            for k in SLOPE_DEFAULTS:
                if k in cfg:
                    self.cfg[k] = cfg[k]
        self.smooth_tilt_deg = 0.0
        self.smooth_roll_deg = 0.0
        self._has_tilt = False
        self._last_t = None
        self._turn_stall_accum = 0.0
        self._steep_stall_accum = 0.0
        self._rollover_accum = 0.0
        self._capsize_accum = 0.0
        self._history = deque()   # (t, x, y, yaw, cmd_wz)

    def reset_motion(self):
        """Clear accumulated stall timers/history while keeping the latest smoothed IMU tilt."""
        self._last_t = None
        self._turn_stall_accum = 0.0
        self._steep_stall_accum = 0.0
        self._rollover_accum = 0.0
        self._capsize_accum = 0.0
        self._history.clear()

    def update_tilt(self, tilt_deg, roll_deg=None):
        alpha = float(self.cfg['slope_tilt_alpha'])
        if not self._has_tilt:
            self.smooth_tilt_deg = float(tilt_deg)
            self.smooth_roll_deg = abs(float(roll_deg)) if roll_deg is not None else 0.0
            self._has_tilt = True
        else:
            self.smooth_tilt_deg = (1.0 - alpha) * self.smooth_tilt_deg + alpha * float(tilt_deg)
            if roll_deg is not None:
                self.smooth_roll_deg = (1.0 - alpha) * self.smooth_roll_deg + alpha * abs(float(roll_deg))
        return self.smooth_tilt_deg

    def is_steep(self):
        return (self._has_tilt
                and float(self.cfg['slope_stall_deg']) <= self.smooth_tilt_deg < float(self.cfg['slope_max_drivable_deg']))

    def is_capsized(self):
        return self._has_tilt and self.smooth_tilt_deg >= float(self.cfg['slope_max_drivable_deg'])

    def update(self, t, x, y, yaw, tilt_deg, imu_wz, cmd_vx, cmd_wz, active=True, roll_deg=None):
        """Feed one control/monitor tick. Returns None, 'capsized', 'rollover_guard', 'turn_stall', or 'slope_stall'."""
        tilt = self.update_tilt(tilt_deg, roll_deg=roll_deg)
        c = self.cfg
        dt = 0.0 if self._last_t is None else max(0.0, min(0.5, float(t) - self._last_t))
        self._last_t = float(t)

        # Capsize check runs ALWAYS — even when active=False (e.g. listening/stopped) or tilt >= 65 deg!
        if tilt >= float(c['slope_max_drivable_deg']):
            self._capsize_accum += dt
            self._turn_stall_accum = 0.0
            self._steep_stall_accum = 0.0
            self._rollover_accum = 0.0
            if self._capsize_accum >= float(c['slope_capsize_confirm_sec']):
                return 'capsized'
            return None
        self._capsize_accum = max(0.0, self._capsize_accum - 2.0 * dt)

        if not active:
            self.reset_motion()
            self._last_t = float(t)
            return None

        self._history.append((float(t), float(x), float(y), float(yaw), float(cmd_wz)))
        max_win = max(float(c['slope_turn_stall_sec']), float(c['slope_stall_sec'])) + 0.5
        while self._history and (t - self._history[0][0]) > max_win:
            self._history.popleft()

        if tilt < float(c['slope_stall_deg']):
            self._turn_stall_accum = max(0.0, self._turn_stall_accum - 2.0 * dt)
            self._steep_stall_accum = max(0.0, self._steep_stall_accum - 2.0 * dt)
            self._rollover_accum = 0.0
            return None

        # 0) Proactive anti-rollover guard: abort Nav2 before side-hilling or steep climb tips the rover
        high_roll = self.smooth_roll_deg >= float(c['slope_rollover_roll_deg'])
        high_tilt = tilt >= float(c['slope_rollover_tilt_deg'])
        if high_roll or high_tilt:
            self._rollover_accum += dt
            if self._rollover_accum >= float(c['slope_rollover_confirm_sec']):
                return 'rollover_guard'
        else:
            self._rollover_accum = max(0.0, self._rollover_accum - 2.0 * dt)

        def _window_stats(win_s):
            ref = None
            cmd_yaw_int = 0.0
            prev_t = None
            for ht, hx, hy, hyaw, hcwz in self._history:
                if t - ht <= win_s:
                    if ref is None:
                        ref = (ht, hx, hy, hyaw)
                    if prev_t is not None:
                        cmd_yaw_int += abs(hcwz) * max(0.0, ht - prev_t)
                    prev_t = ht
            if ref is None:
                return 0.0, 0.0, 0.0, 0.0
            span = t - ref[0]
            disp = math.hypot(x - ref[1], y - ref[2])
            dyaw = abs(wrap(yaw - ref[3]))
            return span, disp, dyaw, cmd_yaw_int

        prog_min = float(c['slope_progress_min_m'])
        turn_win = float(c['slope_turn_stall_sec'])
        span_t, disp_t, dyaw_t, cmd_yaw_t = _window_stats(turn_win)

        # 1) Skid-steer turn stall on a steep slope: commanding rotation, yaw rate blocked or reversed
        cmd_turning = abs(cmd_wz) >= float(c['slope_turn_cmd_min'])
        yaw_blocked = (imu_wz * cmd_wz < -0.02) or (abs(imu_wz) < float(c['slope_turn_ratio_max']) * abs(cmd_wz))
        if disp_t >= prog_min and span_t >= 0.8:
            self._turn_stall_accum = 0.0
        elif cmd_turning and yaw_blocked:
            self._turn_stall_accum += dt
        else:
            self._turn_stall_accum = max(0.0, self._turn_stall_accum - 0.5 * dt)

        if self._turn_stall_accum >= turn_win and disp_t < prog_min:
            return 'turn_stall'
        # Also catch yaw oscillation / rocking back-and-forth while trying to turn on the slope
        if span_t >= turn_win - 0.15 and cmd_yaw_t >= 1.5 and dyaw_t < 0.22 and disp_t < prog_min:
            return 'turn_stall'

        # 2) General steep-slope stall / wheelspin (e.g. unable to climb or reverse out of a steep pocket)
        steep_win = float(c['slope_stall_sec'])
        span_s, disp_s, _, _ = _window_stats(steep_win)
        commanding = (abs(cmd_vx) >= 0.04) or (abs(cmd_wz) >= 0.25)
        if tilt >= float(c['slope_steep_deg']) and commanding:
            if disp_s >= prog_min and span_s >= 1.0:
                self._steep_stall_accum = 0.0
            else:
                self._steep_stall_accum += dt
        else:
            self._steep_stall_accum = max(0.0, self._steep_stall_accum - 0.5 * dt)

        if self._steep_stall_accum >= steep_win and span_s >= steep_win - 0.2 and disp_s < prog_min:
            return 'slope_stall'
        return None


class SafetyMonitor:
    """Continuous mission-wide safety monitor (runs in ALL states, not only during Nav2 goals).

    Detects:
      - capsize (sustained tilt >= slope_max_drivable_deg)
      - fall / drop (rapid altitude loss or high downward vertical velocity)
      - runaway tumble / slide (3D speed far above commanded speed or violent roll/pitch rates)
      - stale critical sensors (IMU, global costmap, traversability grid, TF)
    """

    def __init__(self, cfg=None):
        self.cfg = dict(SLOPE_DEFAULTS)
        if cfg:
            for k in SLOPE_DEFAULTS:
                if k in cfg:
                    self.cfg[k] = cfg[k]
        self._last_t = None
        self._pose_hist = deque()     # (t, x, y, z)
        self._capsize_accum = 0.0
        self._vz_accum = 0.0
        self._speed_accum = 0.0
        self._tumble_accum = 0.0
        self.sensor_stamps = {}       # name -> last_seen_t

    def record_sensor(self, name, t):
        self.sensor_stamps[name] = float(t)

    def check_stale_sensors(self, now_t, required=('imu', 'costmap', 'traversability', 'tf')):
        """Return name of the first stale sensor, or None if all required sensors are fresh."""
        limit = float(self.cfg['sensor_stale_sec'])
        for name in required:
            last = self.sensor_stamps.get(name)
            if last is None:
                return f'{name} (never received)'
            if float(now_t) - last > limit:
                return f'{name} (stale {float(now_t) - last:.1f}s > {limit:.1f}s)'
        return None

    def update(self, t, pose_xyz=None, tilt_deg=None, imu_wx=0.0, imu_wy=0.0):
        """Feed one continuous monitor tick. Returns (reason_code, detail) or (None, None)."""
        t = float(t)
        c = self.cfg
        dt = 0.0 if self._last_t is None else max(0.0, min(0.5, t - self._last_t))
        self._last_t = t

        # 1) Capsize check (tilt >= slope_max_drivable_deg, never ignored!)
        if tilt_deg is not None and math.isfinite(float(tilt_deg)):
            if float(tilt_deg) >= float(c['slope_max_drivable_deg']):
                self._capsize_accum += dt
                if self._capsize_accum >= float(c['slope_capsize_confirm_sec']):
                    return 'capsized', f'tilt {float(tilt_deg):.1f} deg >= {float(c["slope_max_drivable_deg"]):.1f} deg'
            else:
                self._capsize_accum = max(0.0, self._capsize_accum - 2.0 * dt)

        # 2) Extreme roll/pitch angular rate (tumbling end-over-end or sideways)
        rp_rate = max(abs(float(imu_wx)), abs(float(imu_wy)))
        if rp_rate >= float(c['tumble_rate_rps']):
            self._tumble_accum += dt
            if self._tumble_accum >= float(c['fall_confirm_sec']):
                return 'tumble_rate', f'roll/pitch rate {rp_rate:.2f} rad/s'
        else:
            self._tumble_accum = max(0.0, self._tumble_accum - 2.0 * dt)

        # 3) Altitude drop & runaway speed checks from 3D pose history
        if pose_xyz is not None and all(math.isfinite(float(v)) for v in pose_xyz[:3]):
            px, py, pz = float(pose_xyz[0]), float(pose_xyz[1]), float(pose_xyz[2])
            if self._pose_hist:
                pt0, x0, y0, z0 = self._pose_hist[-1]
                pdt = t - pt0
                if pdt > 0.02:
                    vz = (pz - z0) / pdt
                    spd = math.sqrt((px - x0) ** 2 + (py - y0) ** 2 + (pz - z0) ** 2) / pdt
                    if vz <= -float(c['fall_vz_mps']):
                        self._vz_accum += min(pdt, 0.5)
                        if self._vz_accum >= float(c['fall_confirm_sec']):
                            return 'fall_speed', f'downward speed {-vz:.2f} m/s'
                    else:
                        self._vz_accum = max(0.0, self._vz_accum - 2.0 * min(pdt, 0.5))
                    if spd >= float(c['runaway_speed_mps']):
                        self._speed_accum += min(pdt, 0.5)
                        if self._speed_accum >= float(c['fall_confirm_sec']):
                            return 'runaway_tumble', f'3D speed {spd:.2f} m/s'
                    else:
                        self._speed_accum = max(0.0, self._speed_accum - 2.0 * min(pdt, 0.5))
            self._pose_hist.append((t, px, py, pz))
            win = float(c['fall_window_sec'])
            while self._pose_hist and (t - self._pose_hist[0][0]) > win:
                self._pose_hist.popleft()
            if len(self._pose_hist) >= 2:
                z_peak = max(h[3] for h in self._pose_hist)
                drop = z_peak - pz
                if drop >= float(c['fall_drop_m']):
                    return 'fall_drop', f'dropped {drop:.2f} m in <= {win:.1f} s'

        return None, None


def _disc_max_cost(grid, x, y, radius):
    """Maximum cell cost inside a circle of `radius` around (x, y) in `grid`, or -1 if outside/any unknown."""
    if grid is None:
        return 0
    ix = int(math.floor((x - grid.origin_x) / grid.resolution))
    iy = int(math.floor((y - grid.origin_y) / grid.resolution))
    k = int(math.ceil(radius / grid.resolution))
    if ix + k < 0 or iy + k < 0 or ix - k >= grid.width or iy - k >= grid.height:
        return -1
    x0, x1 = max(0, ix - k), min(grid.width, ix + k + 1)
    y0, y1 = max(0, iy - k), min(grid.height, iy + k + 1)
    sub = grid.data[y0:y1, x0:x1]
    if sub.size == 0:
        return -1
    yy, xx = np.mgrid[y0:y1, x0:x1]
    cx = grid.origin_x + (xx + 0.5) * grid.resolution
    cy = grid.origin_y + (yy + 0.5) * grid.resolution
    m = (cx - x) ** 2 + (cy - y) ** 2 <= radius * radius
    if not m.any():
        return int(grid.cost(x, y))
    vals = sub[m]
    vmax = int(vals.max())
    if vmax >= 70:
        return vmax
    if int(vals.min()) < 0:
        return -1
    return vmax


def _arc_point(v, wz, s):
    """Body-frame (x, y) reached after travelling arc length `s >= 0` at linear vel `v` and angular vel `wz`."""
    speed = abs(v)
    if speed < 1e-6:
        return 0.0, 0.0
    t = s / speed
    if abs(wz) < 1e-4:
        return v * t, 0.0
    r = v / wz
    return r * math.sin(wz * t), r * (1.0 - math.cos(wz * t))


def choose_descent_cmd(slope_info, robot_xy_yaw, obs_grid, prev_sign=None, prev_wz=0.0, cfg=None,
                       survey_area=None, area_margin_m=1.0, obs_to_map=None):
    """Choose a slow, obstacle-avoiding (vx, wz) command to descend along the IMU fall line.

    Never commands an in-place skid-steer spin: always moves steadily forward (+1) or in reverse (-1)
    along a collision-free circular arc aligned with downhill while steering around boulders/trees in
    `obs_grid` (non-slope geometric obstacle costmap).

    `obs_to_map`: optional (tx, ty, yaw_rad) rigid 2D transform from obs_grid frame -> map frame so
    `survey_area` (always in map frame) is strictly enforced even when obs_grid is in odom frame.

    Returns (vx, wz, drive_sign, debug_dict).
    """
    c = dict(SLOPE_DEFAULTS)
    if cfg:
        for k in SLOPE_DEFAULTS:
            if k in cfg:
                c[k] = cfg[k]

    rx, ry, ryaw = float(robot_xy_yaw[0]), float(robot_xy_yaw[1]), float(robot_xy_yaw[2])
    cy, sy = math.cos(ryaw), math.sin(ryaw)
    dx_down, dy_down = float(slope_info['down_x']), float(slope_info['down_y'])
    v_max = float(c['slope_descent_speed_max'])
    v_min = float(c['slope_descent_speed_min'])
    wz_max = float(c['slope_descent_max_wz'])
    r_obs = float(c['slope_descent_clearance_m'])
    r_wide = r_obs + 0.16
    stop_d = float(c['slope_descent_stop_dist_m'])
    look_d = float(c['slope_descent_lookahead_m'])
    lethal = int(c['slope_descent_lethal_cost'])
    allow_unknown = bool(c.get('slope_descent_allow_unknown', False))
    steps_s = np.arange(0.15, look_d + 1e-6, 0.12)
    wz_samples = np.linspace(-wz_max, wz_max, 19)

    if obs_to_map is not None:
        m_tx, m_ty, m_yaw = float(obs_to_map[0]), float(obs_to_map[1]), float(obs_to_map[2])
        m_c, m_s = math.cos(m_yaw), math.sin(m_yaw)
    else:
        m_tx = m_ty = 0.0
        m_c, m_s = 1.0, 0.0

    def _in_bounds(wx, wy):
        if survey_area is None:
            return True
        mx = m_tx + m_c * wx - m_s * wy
        my = m_ty + m_s * wx + m_c * wy
        x0, x1, y0, y1 = survey_area
        m = float(area_margin_m)
        return (x0 + m <= mx <= x1 - m) and (y0 + m <= my <= y1 - m)

    def _eval_direction(sign, allow_cross_slope=False):
        v_nom = sign * v_max
        min_align = -0.30 if allow_cross_slope else -0.05
        best = None
        for wz in wz_samples:
            wz_f = float(wz)
            clear_d = look_d
            wide_d = look_d
            soft_sum = 0.0
            soft_n = 0
            dir_pt = None
            for s in steps_s:
                sf = float(s)
                bx, by = _arc_point(v_nom, wz_f, sf)
                wx = rx + cy * bx - sy * by
                wy = ry + sy * bx + cy * by
                if not _in_bounds(wx, wy):
                    clear_d = min(clear_d, sf)
                    wide_d = min(wide_d, sf)
                    break
                mc = _disc_max_cost(obs_grid, wx, wy, r_obs)
                if mc >= lethal or (not allow_unknown and obs_grid is not None and mc < 0):
                    clear_d = sf
                    wide_d = min(wide_d, sf)
                    break
                if wide_d == look_d:
                    wc = _disc_max_cost(obs_grid, wx, wy, r_wide)
                    if wc >= lethal or (not allow_unknown and obs_grid is not None and wc < 0):
                        wide_d = sf
                if mc > 0:
                    soft_sum += mc
                soft_n += 1
                if sf <= 0.85:
                    dir_pt = (bx, by)
            if clear_d < stop_d or dir_pt is None:
                continue
            dn = math.hypot(dir_pt[0], dir_pt[1])
            if dn < 1e-6:
                continue
            down_align = (dir_pt[0] / dn) * dx_down + (dir_pt[1] / dn) * dy_down
            if down_align < min_align:
                continue
            mean_soft = (soft_sum / soft_n) if soft_n else 0.0
            score = (
                2.5 * down_align
                + 1.6 * (clear_d / look_d)
                + 0.45 * (wide_d / look_d)
                - 0.35 * (mean_soft / 100.0)
                - 0.20 * (abs(wz_f) / max(wz_max, 1e-6))
                - 0.25 * (abs(wz_f - prev_wz) / max(wz_max, 1e-6))
            )
            if best is None or score > best[0]:
                frac = max(0.0, min(1.0, (clear_d - stop_d) / max(look_d - stop_d, 0.2)))
                speed = v_min + (v_max - v_min) * frac
                scale = speed / v_max
                best = (score, sign * speed, wz_f * scale, sign, {
                    'blocked': False,
                    'clear_dist': round(clear_d, 2),
                    'down_align': round(down_align, 2),
                    'cross_slope': allow_cross_slope,
                    'score': round(score, 3),
                })
        return best

    # Direction selection with hysteresis (forward when nose-down, reverse when nose-up)
    if prev_sign in (+1, -1):
        # Stick with current end unless the downhill vector flipped strongly (> 117 deg) to the other end
        primary = prev_sign if (prev_sign * dx_down >= -0.45) else -prev_sign
    elif abs(dx_down) >= 0.25:
        primary = +1 if dx_down > 0.0 else -1
    else:
        # Sideways on the slope: pick whichever end gives the better downhill obstacle-free arc
        f_best = _eval_direction(+1, allow_cross_slope=False)
        r_best = _eval_direction(-1, allow_cross_slope=False)
        if f_best is not None and (r_best is None or f_best[0] >= r_best[0]):
            return f_best[1], f_best[2], f_best[3], f_best[4]
        if r_best is not None:
            return r_best[1], r_best[2], r_best[3], r_best[4]
        primary = +1

    # 1) Preferred direction, downhill-aligned arcs
    cand = _eval_direction(primary, allow_cross_slope=False)
    if cand is not None:
        return cand[1], cand[2], cand[3], cand[4]

    # 2) Preferred direction blocked straight downhill: compare wider cross-slope skirt vs reversing downhill
    alt_rev = _eval_direction(-primary, allow_cross_slope=False)
    alt_skirt = _eval_direction(primary, allow_cross_slope=True)
    if alt_rev is not None and (alt_skirt is None or alt_rev[0] > alt_skirt[0] + 0.15):
        return alt_rev[1], alt_rev[2], alt_rev[3], alt_rev[4]
    if alt_skirt is not None:
        return alt_skirt[1], alt_skirt[2], alt_skirt[3], alt_skirt[4]

    # 3) Last resort: reverse with wider cross-slope skirt to back clear of a boulder cluster
    alt_rev_skirt = _eval_direction(-primary, allow_cross_slope=True)
    if alt_rev_skirt is not None:
        return alt_rev_skirt[1], alt_rev_skirt[2], alt_rev_skirt[3], alt_rev_skirt[4]

    return 0.0, 0.0, primary, {'blocked': True, 'clear_dist': 0.0, 'down_align': 0.0, 'cross_slope': True}
