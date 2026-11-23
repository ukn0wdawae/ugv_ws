"""What the Coqui survey shows in RViz, free of ROS so it can be tested offline (test/test_survey_viz.py).

coqui_survey_node.build_scene() turns these into visualization_msgs markers:
  hud_text()        the status lines that ride above the robot (what the mission does, frogs, stations, clock)
  ray_segments()    the hunt's acoustic bearings as line segments (agreeing / rejected / before the first fix)
  station_status()  listening stations: covered, open or unreachable
Like survey_planning, this only uses what the robot itself knows: its poses, the bearings it heard, its own
estimates and records.
"""
import math

import numpy as np

HUD_LINE_CHARS = 64     # longest HUD line; longer text is cut with '...'


# ------------------------------------------------------------------------------------------ HUD
def mmss(seconds):
    """Clock text 'MM:SS', or 'H:MM:SS' from one hour on ('00:00' for a missing or negative time)."""
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return '00:00'
    if not math.isfinite(s) or s < 0:
        return '00:00'
    s = int(s)
    h, m, sec = s // 3600, s // 60 % 60, s % 60
    return f'{h}:{m:02d}:{sec:02d}' if h else f'{m:02d}:{sec:02d}'


def hud_text(state, info, frogs_found, mission_s, stations_covered, stations_total):
    """Status for the HUD above the robot: at most 3 lines of at most HUD_LINE_CHARS plain ASCII characters,
    what the mission is doing now (1-2 lines, from the state and the info passed to set_state) and a summary
    'frogs recorded: 2   stations 7/23   05:31'. Missing or malformed info keys are left out."""
    info = info if isinstance(info, dict) else {}
    head, detail = _headline(str(state or '?'), info, frogs_found)
    summary = f'frogs recorded: {_int(frogs_found)}'
    if _int(stations_total) > 0:
        summary += f'   stations {_int(stations_covered)}/{_int(stations_total)}'
    summary += '   ' + mmss(mission_s)
    return [_clean(line) for line in (head, detail, summary) if line]


def _headline(state, info, frogs_found):
    """(headline, detail) for one mission state; detail may be ''."""
    g = info.get
    if state == 'WAIT_FOR_NAV2':
        return 'WAIT_FOR_NAV2: waiting for Nav2, costmap and TF', ''
    if state == 'START':
        return ('START' + _num(g('stations_to_cover'), '.0f', ': ', ' stations to cover') + _xy(g('home'), ' from '),
                '')
    if state == 'LISTEN':
        return 'LISTEN' + _num(g('listen_s'), '.0f', ' ', ' s') + _xy(g('at'), ' at '), ''
    if state == 'EXPLORE':
        return 'EXPLORE -> station' + _xy(g('goal'), ' ') + _num(g('path_m'), '.1f', ', path ', ' m'), ''
    if state == 'RECOVER':
        return ('RECOVER' + _txt(g('reason'), ': ') + _num(g('candidates'), '.0f', ' (', ' candidates)'),
                'clearing the costmaps, then planning again')
    if state == 'UNSTICK':
        return ('UNSTICK' + _num(g('fails'), '.0f', ': ', ' failed goals') + _xy(g('at'), ' at '),
                'driving back to a place reached before')
    if state == 'HUNT':
        if g('step') is None:   # the call that starts the hunt
            return ('HUNT new call' + _num(g('bearing_deg'), '.0f', ': bearing ', ' deg') + _xy(g('heard_at'), ' at '),
                    'triangulating: listen, step sideways, listen again')
        head = 'HUNT step' + _num(g('step'), '.0f', ' ') + ': '
        if g('inliers') is None:   # bearings from one place only so far
            return head + _num(g('obs'), '.0f', '', ' bearings, ') + 'no cross-bearing fix yet', ''
        head += (_num(g('inliers'), '.0f', '', ' bearings') + _num(g('places'), '.0f', ' from ', ' places')
                 + _num(g('parallax_deg'), '.0f', ', parallax ', ' deg'))
        return head, 'estimate' + _num(g('unc_m'), '.2f', ' +/-', ' m') + _xy(g('est'), ' at ', 3)
    if state == 'VANTAGE':
        if g('step') == 'staging':
            return 'VANTAGE staging' + _xy(g('goal'), ' -> '), 'getting close so the sensors map the spot first'
        if g('attempt') is not None:
            los = g('line_of_sight')
            return ('VANTAGE try' + _num(g('attempt'), '.0f', ' ') + _xy(g('goal'), ' -> ')
                    + _num(g('path_m'), '.1f', ', path ', ' m'),
                    '' if los is None else 'line of sight on the costmap: ' + ('yes' if los else 'no'))
        return 'VANTAGE: camera viewpoint' + _txt(g('height'), ' (', ' frog)'), _xy(g('est'), 'estimate ', 3)
    if state == 'PHOTO':
        fid = g('id') or f'coqui_{_int(frogs_found) + 1:03d}'
        return f'PHOTO {fid}: aiming the pan-tilt camera', _txt(g('vantage'), 'vantage: ')
    if state == 'RECORDED':
        return f"RECORDED {g('id') or 'frog'} " + ('(photo)' if g('visual') else '(no photo)'), _xy(g('pos'), 'at ', 3)
    if state == 'HUNT_END':
        return 'HUNT_END' + _txt(g('outcome'), ': '), ''
    if state == 'HOME':
        return 'HOME: driving back to the start', ''
    if state == 'DONE':
        return 'DONE' + _txt(g('result'), ': '), ''
    return state, ''


def _finite(v):
    """v as a float, or None when it is missing or not a finite number."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _int(v):
    f = _finite(v)
    return int(f) if f is not None else 0


def _num(v, fmt, pre='', post=''):
    """pre + v formatted with fmt + post, or '' when v is missing or not a finite number."""
    f = _finite(v)
    return '' if f is None else f'{pre}{f:{fmt}}{post}'


def _xy(v, pre='', n=2):
    """pre + '(x, y)' (or '(x, y, z)' with n=3), one decimal, or '' when v is missing or malformed."""
    try:
        vals = [_finite(v[i]) for i in range(n)]
    except (TypeError, IndexError, KeyError):
        return ''
    if any(f is None for f in vals):
        return ''
    return pre + '(' + ', '.join(f'{f:.1f}' for f in vals) + ')'


def _txt(v, pre='', post=''):
    """pre + str(v) + post, or '' when v is missing or empty."""
    s = '' if v is None else str(v)
    return f'{pre}{s}{post}' if s else ''


def _clean(line):
    """One line of plain ASCII, at most HUD_LINE_CHARS long."""
    line = str(line).replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
    line = line.encode('ascii', 'replace').decode('ascii')
    return line if len(line) <= HUD_LINE_CHARS else line[:HUD_LINE_CHARS - 3] + '...'


# --------------------------------------------------------------------------------- bearing rays
def ray_segments(obs, inlier_mask, estimate, default_len=8.0, max_len=25.0):
    """The hunt's bearings as line segments [(start xyz, end xyz, kind)] (plain floats).

    obs: [(t, origin xyz, unit dir xyz, snr)] like the node's self.obs (map frame).
    inlier_mask: per-observation bools from robust_triangulate ('inlier' / 'outlier'), or None before the first
        fix ('pending' for all; also for observations beyond the end of the mask).
    estimate: xyz the rays should reach, or None. Each ray is as long as the distance from its origin to the
        estimate + 1.5 m (so agreeing rays visibly cross there), else default_len; never longer than max_len.
    Observations with a zero or non-finite direction are skipped."""
    est = None
    if estimate is not None:
        est = np.asarray(estimate, dtype=float).ravel()
        if est.shape != (3,) or not np.all(np.isfinite(est)):
            est = None
    out = []
    for i, ob in enumerate(obs):
        o = np.asarray(ob[1], dtype=float).ravel()[:3]
        d = np.asarray(ob[2], dtype=float).ravel()[:3]
        n = float(np.linalg.norm(d)) if d.shape == (3,) else 0.0
        if o.shape != (3,) or not (n > 1e-9 and math.isfinite(n) and np.all(np.isfinite(o))):
            continue
        length = default_len if est is None else float(np.linalg.norm(est - o)) + 1.5
        end = o + min(length, max_len) * d / n
        if inlier_mask is None or i >= len(inlier_mask):
            kind = 'pending'
        else:
            kind = 'inlier' if inlier_mask[i] else 'outlier'
        out.append((tuple(float(v) for v in o), tuple(float(v) for v in end), kind))
    return out


# ------------------------------------------------------------------------------------ stations
def station_status(cells, listened, cover_radius, failed, failed_radius):
    """Listening stations as the RViz view shows them: [(x, y, status)] in the order of `cells`.
    'covered'      the robot has listened within cover_radius (as survey_planning.station_candidates counts it)
    'unreachable'  not covered, and a Nav2 goal within failed_radius of it failed
    'open'         still to visit"""
    out = []
    for c in cells:
        x, y = float(c[0]), float(c[1])
        if any(math.hypot(x - p[0], y - p[1]) < cover_radius for p in listened):
            status = 'covered'
        elif any(math.hypot(x - f[0], y - f[1]) < failed_radius for f in failed):
            status = 'unreachable'
        else:
            status = 'open'
        out.append((x, y, status))
    return out
