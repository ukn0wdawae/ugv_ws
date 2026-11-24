"""Pure-numpy Coqui acoustic synthesis, motor noise, harmonic-free SNR, SRP-PHAT DOA,
and seeded scenario randomization (ROS-free so offline tests can exercise all acoustic physics).
"""
import copy
import functools
import math
import os
import numpy as np


SPEED_OF_SOUND = 343.0  # m/s in air at ~20 C
SAMPLE_RATE = 16000     # Hz
FRAME_SAMPLES = 2048    # ~128 ms per audio frame

# Microphone coordinates in `audio_array_base_link` frame (meters)
MIC_POSITIONS_BODY = np.array([
    [0.095,  0.075, 0.000],  # 0: Front-Left (mic_fl_link)
    [0.095, -0.075, 0.000],  # 1: Front-Right (mic_fr_link)
    [-0.095, 0.075, 0.000],  # 2: Rear-Left (mic_rl_link)
    [-0.095, -0.075, 0.000], # 3: Rear-Right (mic_rr_link)
    [-0.065, 0.000, 0.040],  # 4: Top-Center (mic_top_link)
], dtype=np.float64)

ARRAY_BASE_OFFSET_IN_FOOTPRINT = np.array([0.0, 0.0, 0.115], dtype=np.float64)

# Frequency masks for Coqui call detection (harmonic-free!).
# Co note ~1160 Hz (2nd harmonic 2320 Hz); Qui chirp ~1820->2180 Hz (1.5x harmonic 2730..3270 Hz,
# 2x harmonic 3640..4360 Hz). The noise guard bands sit strictly outside BOTH fundamentals AND
# harmonics so SNR scales cleanly with distance (~28+ dB at 1 m down to ~7 dB at 22 m) instead of
# hitting a synthetic 13.4 dB harmonic ceiling.
FREQS = np.fft.rfftfreq(FRAME_SAMPLES, d=1.0 / SAMPLE_RATE)
CO_BAND_MASK = (FREQS >= 1060.0) & (FREQS <= 1260.0)
QUI_BAND_MASK = (FREQS >= 1760.0) & (FREQS <= 2260.0)
COQUI_MASK = CO_BAND_MASK | QUI_BAND_MASK
NOISE_GUARD_MASK = (
    ((FREQS >= 550.0) & (FREQS <= 920.0))
    | ((FREQS >= 1380.0) & (FREQS <= 1620.0))
    | ((FREQS >= 4800.0) & (FREQS <= 6500.0))
)


def quat_to_rot_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to 3x3 rotation matrix."""
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def synthesize_coqui_call_frame(num_samples: int, sample_rate: int, phase_seed: float = 0.0,
                                pitch_scale: float = 1.0, start_frac: float = 0.0) -> np.ndarray:
    """Synthesize a single frame containing the two-note 'Co-Qui' call.

    `pitch_scale` adds realistic per-frog vocal variation (typically 0.96..1.04).
    `start_frac` shifts the call onset within the frame (used for overlapping calls).
    """
    t = np.arange(num_samples, dtype=np.float64) / float(sample_rate)
    signal = np.zeros_like(t)
    shift = int(np.clip(start_frac, -0.04, 0.35) * num_samples)

    # Note 1: "CO" (~1160 Hz constant tone with Hann envelope)
    n_co_start = max(0, int(0.05 * num_samples) + shift)
    n_co_end = min(num_samples, int(0.40 * num_samples) + shift)
    if n_co_end > n_co_start + 8:
        t_co = t[n_co_start:n_co_end] - t[n_co_start]
        env_co = np.sin(np.pi * np.linspace(0.0, 1.0, n_co_end - n_co_start)) ** 2
        f_co = 1160.0 * pitch_scale
        signal[n_co_start:n_co_end] += 0.75 * env_co * (
            np.sin(2.0 * np.pi * f_co * t_co + phase_seed)
            + 0.25 * np.sin(2.0 * np.pi * (2.0 * f_co) * t_co + 0.5 * phase_seed)
        )

    # Note 2: "QUI" (~1820 Hz -> 2180 Hz upward chirp)
    n_qui_start = max(0, int(0.48 * num_samples) + shift)
    n_qui_end = min(num_samples, int(0.92 * num_samples) + shift)
    if n_qui_end > n_qui_start + 8:
        t_qui = t[n_qui_start:n_qui_end] - t[n_qui_start]
        dur_qui = max(t_qui[-1], 1e-4)
        f0, f1 = 1820.0 * pitch_scale, 2180.0 * pitch_scale
        chirp_phase = 2.0 * np.pi * (f0 * t_qui + 0.5 * ((f1 - f0) / dur_qui) * (t_qui ** 2))
        env_qui = np.sin(np.pi * np.linspace(0.0, 1.0, n_qui_end - n_qui_start)) ** 1.5
        signal[n_qui_start:n_qui_end] += 1.0 * env_qui * (
            np.sin(chirp_phase + phase_seed)
            + 0.25 * np.sin(2.0 * chirp_phase)
        )
    return signal


def fractional_delay_fft(signal: np.ndarray, delay_samples: float) -> np.ndarray:
    """Apply exact sub-sample time delay via Fourier linear phase shift."""
    n = len(signal)
    spec = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0)
    phase_shift = np.exp(-1j * 2.0 * np.pi * freqs * delay_samples)
    return np.fft.irfft(spec * phase_shift, n=n)


def synthesize_motor_noise(num_mics: int, num_samples: int, sample_rate: int,
                           speed_mps: float, yaw_rate_rps: float, gain: float,
                           rng: np.random.Generator) -> np.ndarray:
    """Synthesize realistic DC gearmotor + tire-on-dirt self-noise across the mic array.

    Dominated by low-frequency chassis rumble (<750 Hz), gearmesh tones (~650/950 Hz),
    and a weaker broadband floor that leaks into the Coqui band when driving fast.
    """
    activity = abs(float(speed_mps)) + 0.35 * abs(float(yaw_rate_rps))
    if gain <= 0.0 or activity < 0.02:
        return np.zeros((num_mics, num_samples), dtype=np.float64)
    level = gain * min(activity / 0.5, 2.0)
    t = np.arange(num_samples, dtype=np.float64) / float(sample_rate)
    freqs = np.fft.rfftfreq(num_samples, d=1.0 / sample_rate)
    # Pink/low-pass shaped rumble (1/f rolloff above 250 Hz, strong chassis coupling <700 Hz)
    rumble_shape = 1.0 / np.sqrt(1.0 + (freqs / 320.0) ** 2)
    rumble_shape[0] = 0.0
    common = rng.normal(0.0, 1.0, size=num_samples)
    common_filt = np.fft.irfft(np.fft.rfft(common) * rumble_shape, n=num_samples)
    common_filt /= max(float(np.std(common_filt)), 1e-9)
    # Gearmesh whine scaled with wheel speed (stays mostly below 1000 Hz)
    f_gear = 280.0 + 380.0 * min(activity, 1.2)
    whine = 0.35 * np.sin(2.0 * np.pi * f_gear * t + rng.uniform(0.0, 2.0 * np.pi))
    out = np.empty((num_mics, num_samples), dtype=np.float64)
    for i in range(num_mics):
        indep = rng.normal(0.0, 1.0, size=num_samples)
        indep_filt = np.fft.irfft(np.fft.rfft(indep) * rumble_shape, n=num_samples)
        indep_filt /= max(float(np.std(indep_filt)), 1e-9)
        broad = rng.normal(0.0, 0.18, size=num_samples)
        out[i] = level * (0.65 * common_filt + 0.35 * indep_filt + whine + broad)
    return out


def synthesize_ambient_clutter(num_mics: int, num_samples: int, sample_rate: int,
                               rng: np.random.Generator, amp: float = 0.03) -> np.ndarray:
    """Synthesize non-Coqui forest insect/cricket trill (>3.2 kHz) so detector must be band-selective."""
    t = np.arange(num_samples, dtype=np.float64) / float(sample_rate)
    f_insect = rng.uniform(3300.0, 5200.0)
    env = 0.5 * (1.0 + np.sin(2.0 * np.pi * rng.uniform(25.0, 45.0) * t))
    trill = amp * env * np.sin(2.0 * np.pi * f_insect * t + rng.uniform(0.0, 2.0 * np.pi))
    return np.tile(trill[None, :], (num_mics, 1))


def compute_coqui_snr_db(signals: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """Compute harmonic-free, motor-noise-resilient Coqui SNR (dB) and complex spectra.

    Compares peak-band energy inside the narrow Co + Qui windows (1060-1260 Hz, 1760-2260 Hz)
    against the **median** noise power in quiet guard bands (550-920, 1380-1620, 4800-6500 Hz).
    Using the median of guard bins rejects narrowband gear whine or insect chirps while giving
    clean distance-dependent SNR (~28+ dB at 1 m down to ~7 dB at 22 m).
    Returns (snr_db, specs, power_spec).
    """
    n_samples = signals.shape[1]
    win = np.hanning(n_samples)[None, :]
    specs = np.fft.rfft(signals * win, axis=1)
    power_spec = np.mean(np.abs(specs) ** 2, axis=0)
    if n_samples == FRAME_SAMPLES and sample_rate == SAMPLE_RATE:
        co_m, qui_m, guard_m = CO_BAND_MASK, QUI_BAND_MASK, NOISE_GUARD_MASK
    else:
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
        co_m = (freqs >= 1060.0) & (freqs <= 1260.0)
        qui_m = (freqs >= 1760.0) & (freqs <= 2260.0)
        guard_m = (((freqs >= 550.0) & (freqs <= 920.0))
                   | ((freqs >= 1380.0) & (freqs <= 1620.0))
                   | ((freqs >= 4800.0) & (freqs <= 6500.0)))

    # Use top-quartile energy inside Co and Qui sub-bands (narrowband tonal call vs broadband noise)
    co_bins = np.sort(power_spec[co_m])
    qui_bins = np.sort(power_spec[qui_m])
    co_sig = float(np.mean(co_bins[max(0, len(co_bins) // 2):])) if len(co_bins) else 0.0
    qui_sig = float(np.mean(qui_bins[max(0, len(qui_bins) // 2):])) if len(qui_bins) else 0.0
    # Require energy in BOTH notes (or strong harmonic pair) so pure motor whine in one bin is damped
    dual_sig = 0.5 * (co_sig + qui_sig)
    # Use high-band + mid-band median guard floor (robust to low-freq motor rumble < 800 Hz!)
    guard_vals = power_spec[guard_m]
    noise_floor = max(float(np.median(guard_vals)) * 1.44, 1e-12)  # 1.44 = 1/ln(2) converts Rayleigh median->mean
    snr_db = 10.0 * math.log10(max(dual_sig / noise_floor, 1e-6))
    return float(snr_db), specs, power_spec


# Band level (dB) of a nominal call 1 m away, as call_band_level_db measures it (Hann window, Co + Qui bands).
# Calibrate on hardware: play a reference call 1 m from the array.
CALL_LEVEL_AT_1M_DB = -13.0


def call_band_level_db(specs: np.ndarray, power_spec: np.ndarray) -> float:
    """Received level of the call in dB relative to a nominal call heard 1 m away (0 = 1 m, -20 = 10 m, ...).

    The noise floor of the guard bands is subtracted, so the motors make a call less audible but not louder.
    specs / power_spec are what compute_coqui_snr_db returned for a FRAME_SAMPLES frame."""
    floor = float(np.median(power_spec[NOISE_GUARD_MASK])) * 1.44
    n = specs.shape[1]
    # Parseval for a one-sided spectrum: mean square = 2 * sum|X|^2 / N^2
    band = float(np.sum(np.maximum(power_spec[COQUI_MASK] - floor, 0.0)))
    ms = 2.0 * band / float((2 * (n - 1)) ** 2)
    return 10.0 * math.log10(max(ms, 1e-14)) - CALL_LEVEL_AT_1M_DB


def level_to_range_m(level_db: float) -> float:
    """Distance at which a nominal call has this level (spherical spreading)."""
    return 10.0 ** (-float(level_db) / 20.0)


_AZ0 = np.deg2rad(np.linspace(-180.0, 175.0, 72))
_EL0 = np.deg2rad(np.linspace(-15.0, 65.0, 17))
_COARSE_AZ, _COARSE_EL = (a.ravel() for a in np.meshgrid(_AZ0, _EL0, indexing='ij'))


def _unit_dirs(az, el):
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)


_COARSE_PROJ = _unit_dirs(_COARSE_AZ, _COARSE_EL) @ MIC_POSITIONS_BODY.T
_COQUI_FREQS = FREQS[COQUI_MASK]


def estimate_doa(spec_coqui: np.ndarray):
    """Direction of arrival by coarse-to-fine SRP-PHAT. spec_coqui: complex (5, F) in the Coqui passband.

    Returns (azimuth, elevation, unit direction in the array frame, coherence). Coherence is the share of the
    call's energy that the best direction explains (1 = one plane wave). One frog gives 0.97 and more at any
    distance; two frogs calling at once give 0.6..0.95, and then the direction may point at neither."""
    mag = np.abs(spec_coqui)
    phat = spec_coqui / np.maximum(mag ** 0.7, 1e-9)     # keeps some magnitude so the strong Co/Qui bins dominate
    top = np.argsort(np.mean(mag, axis=0))[-40:]
    f, sub = _COQUI_FREQS[top], phat[:, top]
    k = -2.0 * np.pi / SPEED_OF_SOUND

    def power(proj):
        steered = np.sum(np.exp(1j * k * proj[:, :, None] * f[None, None, :]) * sub[None, :, :], axis=1)
        return np.sum(np.abs(steered) ** 2, axis=1)

    i = int(np.argmax(power(_COARSE_PROJ)))
    az = float(_COARSE_AZ[i]) + np.deg2rad(np.linspace(-6.0, 6.0, 25))
    el = np.clip(float(_COARSE_EL[i]) + np.deg2rad(np.linspace(-6.0, 6.0, 25)), np.deg2rad(-25.0), np.deg2rad(75.0))
    az, el = (a.ravel() for a in np.meshgrid(az, el, indexing='ij'))
    dirs = _unit_dirs(az, el)
    p = power(dirs @ MIC_POSITIONS_BODY.T)
    j = int(np.argmax(p))
    coherence = float(p[j] / max(MIC_POSITIONS_BODY.shape[0] * float(np.sum(np.abs(sub) ** 2)), 1e-18))
    return math.atan2(math.sin(az[j]), math.cos(az[j])), float(el[j]), dirs[j] / np.linalg.norm(dirs[j]), coherence


def propagate_sources_to_array(sources, array_origin_world, robot_rot_world,
                               noise_std=0.008, motor_speed=0.0, motor_wz=0.0,
                               motor_noise_gain=0.0, echo_gain=0.0, echo_delay_ms=18.0,
                               ambient_clutter=False, rng=None, mic_error=None):
    """Simulate 5-channel microphone waveforms for 0, 1, or multiple overlapping frog calls.

    `sources`: list of dicts {'pos': (x,y,z), 'waveform': np.ndarray(FRAME_SAMPLES), 'foliage_attenuation': float}
    `mic_error`: (position error (5, 3) m, gain (5,)) of the real microphones, see make_mic_error; None = ideal.
    Returns (5, FRAME_SAMPLES) float64 array.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    num_mics = MIC_POSITIONS_BODY.shape[0]
    mic_body = MIC_POSITIONS_BODY if mic_error is None else MIC_POSITIONS_BODY + mic_error[0]
    mic_world_pts = array_origin_world + (mic_body @ robot_rot_world.T)
    multi_ch = rng.normal(0.0, noise_std, size=(num_mics, FRAME_SAMPLES))

    if motor_noise_gain > 0.0:
        multi_ch += synthesize_motor_noise(
            num_mics, FRAME_SAMPLES, SAMPLE_RATE, motor_speed, motor_wz, motor_noise_gain, rng)
    if ambient_clutter:
        multi_ch += synthesize_ambient_clutter(num_mics, FRAME_SAMPLES, SAMPLE_RATE, rng)

    for src in sources:
        fpos = np.asarray(src['pos'], dtype=np.float64)
        base_wave = src['waveform']
        foliage_gain = float(src.get('foliage_attenuation', 1.0))
        dists = np.linalg.norm(mic_world_pts - fpos[None, :], axis=1)
        center_dist = float(np.linalg.norm(array_origin_world - fpos))
        tau_center = center_dist / SPEED_OF_SOUND
        for i in range(num_mics):
            tau_i = float(dists[i]) / SPEED_OF_SOUND
            rel_delay_samples = (tau_i - tau_center) * float(SAMPLE_RATE)
            attenuation = foliage_gain / max(float(dists[i]), 0.35)
            delayed = fractional_delay_fft(base_wave, rel_delay_samples)
            multi_ch[i, :] += attenuation * delayed
            if echo_gain > 0.0:
                echo_samples = rel_delay_samples + (echo_delay_ms * 1e-3) * float(SAMPLE_RATE)
                echo_wave = fractional_delay_fft(base_wave, echo_samples)
                multi_ch[i, :] += (echo_gain * attenuation) * echo_wave
    if mic_error is not None:
        multi_ch *= np.asarray(mic_error[1], dtype=np.float64)[:, None]
    return multi_ch


def make_mic_error(rng, pos_sigma_m=0.0015, gain_sigma_db=0.7):
    """How the microphones of one real robot differ from the drawing: a fixed position and gain error each.
    The robot does not know it; it bends every bearing by a direction-dependent part of a degree."""
    n = MIC_POSITIONS_BODY.shape[0]
    return rng.normal(0.0, pos_sigma_m, (n, 3)), 10.0 ** (rng.normal(0.0, gain_sigma_db, n) / 20.0)


class CallVariation:
    """How loud the calls of one frog arrive, beyond spherical spreading.

    The frog: a fixed source level of its own, and a call-to-call variation. The path: foliage takes some dB per
    metre, and the cover the frog sits in (`cover_attenuation`, amplitude 0..1) muffles it more in one direction
    (the trunk or the bush is on one side of it) than in the other."""

    def __init__(self, rng, n_frogs, cover_attenuation, frog_sigma_db=1.5, call_sigma_db=1.2, foliage_db_per_m=0.08,
                 cover_directivity=0.5):
        self.rng = rng
        self.frog_db = np.clip(rng.normal(0.0, frog_sigma_db, n_frogs), -2.5 * frog_sigma_db, 2.5 * frog_sigma_db)
        self.cover_db = [-20.0 * math.log10(max(float(a), 0.05)) for a in cover_attenuation]
        self.cover_dir = rng.uniform(-math.pi, math.pi, n_frogs)
        self.call_sigma = float(call_sigma_db)
        self.foliage = float(foliage_db_per_m)
        self.directivity = float(cover_directivity)

    def gain(self, i, frog_xy, array_xy):
        """Amplitude factor for one call of frog i heard at array_xy."""
        dx, dy = float(array_xy[0]) - float(frog_xy[0]), float(array_xy[1]) - float(frog_xy[1])
        cover = self.cover_db[i] * (1.0 + self.directivity * math.cos(math.atan2(dy, dx) - self.cover_dir[i]))
        db = self.frog_db[i] + self.rng.normal(0.0, self.call_sigma) - cover - self.foliage * math.hypot(dx, dy)
        return 10.0 ** (db / 20.0)


class ChorusSchedule:
    """Every frog calls on its own rhythm, like a real chorus: a call every 2..4 s for a minute or two, then a pause.

    Neighbours avoid calling over each other most of the time (as coquis do), but not always: `overlap_prob` of the
    calls that would collide are let through. step(t) returns [(frog index, onset shift 0..0.25)] for this frame."""

    def __init__(self, n_frogs, rng, interval_sec=(2.0, 3.6), bout_sec=(45.0, 120.0), pause_sec=(8.0, 35.0),
                 overlap_prob=0.25, start_sec=0.0):
        self.rng = rng
        self.n = int(n_frogs)
        self.interval = rng.uniform(interval_sec[0], interval_sec[1], self.n)
        self.bout, self.pause = bout_sec, pause_sec
        self.overlap_prob = float(overlap_prob)
        self.next_call = start_sec + rng.uniform(0.0, 4.0, self.n)
        self.bout_end = self.next_call + rng.uniform(bout_sec[0] * 0.5, bout_sec[1], self.n)

    def step(self, t):
        out = []
        due = [i for i in range(self.n) if t >= self.next_call[i]]
        self.rng.shuffle(due)
        for i in due:
            if out and float(self.rng.random()) >= self.overlap_prob:
                self.next_call[i] = t + self.rng.uniform(0.25, 0.8)      # wait for the neighbour to finish
                continue
            out.append((i, 0.0 if not out else float(self.rng.uniform(0.05, 0.25))))
            gap = self.interval[i] * self.rng.uniform(0.9, 1.15)
            if t >= self.bout_end[i]:
                gap = self.rng.uniform(*self.pause)
                self.bout_end[i] = t + gap + self.rng.uniform(*self.bout)
            self.next_call[i] = t + gap
        return out


def _world_scenes_path() -> str:
    """Locate precomputed 3D terrain & vegetation geometry (`world_scenes.npz`) in source or install tree."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(os.path.join(here, '..', 'config', 'sim_truth', 'world_scenes.npz')),
    ]
    try:
        from ament_index_python.packages import get_package_share_directory
        candidates.append(os.path.join(
            get_package_share_directory('ugv_audio'), 'config', 'sim_truth', 'world_scenes.npz'))
    except Exception:
        pass
    for p in candidates:
        if os.path.isfile(p):
            return p
    return ''


@functools.lru_cache(maxsize=1)
def load_world_scenes() -> dict:
    """Load cached 3D terrain elevation grids and real bush/tree/grass catalogs."""
    p = _world_scenes_path()
    if not p:
        return {}
    with np.load(p, allow_pickle=False) as zf:
        return {k: zf[k] for k in zf.files}


def _scene_prefix(world_name: str) -> str:
    w = (world_name or '').lower()
    if 'forest' in w:
        return 'forest'
    if 'hill' in w:
        return 'hill'
    return ''


def ground_elevation_z(world_name: str, x: float, y: float, fallback_z: float = 0.0) -> float:
    """Return exact bilinearly interpolated terrain mesh elevation `z(x, y)` for `forest` or `hill`."""
    scenes = load_world_scenes()
    pref = _scene_prefix(world_name)
    if not pref or f'{pref}_grid_z' not in scenes:
        return float(fallback_z)
    grid = scenes[f'{pref}_grid_z']
    x0, y0, dx, dy = (float(v) for v in scenes[f'{pref}_grid_meta'])
    ny, nx = grid.shape
    fx = float(np.clip((float(x) - x0) / dx, 0.0, nx - 1.0001))
    fy = float(np.clip((float(y) - y0) / dy, 0.0, ny - 1.0001))
    ix, iy = int(math.floor(fx)), int(math.floor(fy))
    tx, ty = fx - ix, fy - iy
    z00, z10 = float(grid[iy, ix]), float(grid[iy, ix + 1])
    z01, z11 = float(grid[iy + 1, ix]), float(grid[iy + 1, ix + 1])
    return (1.0 - ty) * ((1.0 - tx) * z00 + tx * z10) + ty * ((1.0 - tx) * z01 + tx * z11)


def _is_clear_of_obstacles(x: float, y: float, obstacles: np.ndarray, extra_pad: float = 0.45) -> bool:
    if obstacles is None or len(obstacles) == 0:
        return True
    d = np.hypot(obstacles[:, 0] - x, obstacles[:, 1] - y)
    return bool(np.all(d >= obstacles[:, 2] + extra_pad))


def _respects_bounds_and_sep(x: float, y: float, bounds, placed_xy, min_sep: float,
                             spawn_xy=None, min_spawn_sep: float = 0.0) -> bool:
    if bounds is not None:
        xmin, xmax, ymin, ymax = bounds
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            return False
    if spawn_xy is not None and min_spawn_sep > 0.0:
        if math.hypot(x - spawn_xy[0], y - spawn_xy[1]) < min_spawn_sep:
            return False
    return all(math.hypot(x - px, y - py) >= min_sep for px, py in placed_xy)


def _randomize_with_scene(out: dict, pref: str, scenes: dict, rng: np.random.Generator,
                          bounds, min_frog_sep_m: float, spawn_xy, min_spawn_sep_m: float) -> bool:
    """Place frogs across real 3D bushes, tree tops, tree bark perches, tall grass, and open glades."""
    frogs = out.get('frogs', [])
    if not frogs:
        return True
    world_name = out.get('world', pref)
    bushes = scenes.get(f'{pref}_bushes', np.zeros((0, 4)))
    tree_tops = scenes.get(f'{pref}_tree_tops', np.zeros((0, 6)))
    tree_perches = scenes.get(f'{pref}_tree_perches', np.zeros((0, 5)))
    grass = scenes.get(f'{pref}_grass', np.zeros((0, 3)))
    obstacles = scenes.get(f'{pref}_obstacles', np.zeros((0, 4)))

    n = len(frogs)
    # Build habitat schedule ensuring every 6-frog forest run has:
    #   - 2 frogs inside distinct dense bushes
    #   - 2 elevated tree frogs (at least 1 literally on top of a tree crown/trunk + 1 perched up a pine trunk)
    #   - 1 frog inside tall grass
    #   - 1 frog on open forest floor / behind a pine trunk
    if pref == 'forest':
        base_roles = ['bush', 'bush', 'tree_top', 'tree_perch', 'grass', 'open_or_behind']
        if rng.random() < 0.35 and len(tree_tops) >= 2:
            base_roles[3] = 'tree_top'
    else:
        base_roles = ['bush', 'bush', 'tree_perch', 'grass', 'open_or_behind', 'open_or_behind']
    roles = [base_roles[i % len(base_roles)] for i in range(n)]
    # Shuffle which frog ID (`coqui_01..06`) gets which habitat role so IDs are not predictable
    role_order = rng.permutation(n)

    placed_xy = []
    used_groups = set()
    assignments = [None] * n

    # Place constrained feature-bound roles (`tree_top`, `bush`, `tree_perch`, `grass`) before open ground
    priority = {'tree_top': 0, 'bush': 1, 'tree_perch': 2, 'grass': 3, 'open_or_behind': 4}
    slot_indices = sorted(range(n), key=lambda idx: priority[roles[idx]])

    for slot_idx in slot_indices:
        role = roles[slot_idx]
        chosen = None

        if role == 'tree_top' and len(tree_tops) > 0:
            perm = rng.permutation(len(tree_tops))
            # Balance crown-top (subtype 0) and fallen-trunk-top (subtype 1) across seeds
            prefer_sub = 0.0 if rng.random() < 0.65 else 1.0
            ordered = sorted(perm.tolist(), key=lambda i: 0 if tree_tops[i, 5] == prefer_sub else 1)
            for idx in ordered:
                cx, cy, cz, cyaw, grp, subtype = (float(v) for v in tree_tops[idx])
                if int(grp) in used_groups:
                    continue
                if not _respects_bounds_and_sep(cx, cy, bounds, placed_xy, min_frog_sep_m, spawn_xy, min_spawn_sep_m):
                    continue
                used_groups.add(int(grp))
                g_z = ground_elevation_z(world_name, cx, cy, cz - 1.6)
                dh = max(1.0, cz - g_z)
                hab = (f'on top of young pine tree crown ({dh:.1f} m high)'
                       if subtype < 0.5 else f'on top of fallen pine tree trunk ({dh:.1f} m high)')
                chosen = {
                    'pos': [round(cx, 3), round(cy, 3), round(cz, 3)],
                    'yaw': round(cyaw, 3),
                    'habitat': hab,
                    'attenuation': round(float(rng.uniform(0.90, 0.97)), 3),
                }
                break
            if chosen is None:
                role = 'tree_perch'

        if role == 'bush' and len(bushes) > 0:
            perm = rng.permutation(len(bushes))
            for idx in perm:
                bx, by, _bz, grp = (float(v) for v in bushes[idx])
                if int(grp) in used_groups:
                    continue
                jx = bx + float(rng.uniform(-0.18, 0.18))
                jy = by + float(rng.uniform(-0.18, 0.18))
                if not _respects_bounds_and_sep(jx, jy, bounds, placed_xy, min_frog_sep_m, spawn_xy, min_spawn_sep_m):
                    continue
                used_groups.add(int(grp))
                jz = ground_elevation_z(world_name, jx, jy, _bz)
                chosen = {
                    'pos': [round(jx, 3), round(jy, 3), round(jz, 3)],
                    'habitat': 'inside a dense bush (not traversable)',
                    'attenuation': round(float(rng.uniform(0.75, 0.83)), 3),
                }
                break

        if role == 'tree_perch' and len(tree_perches) > 0:
            perm = rng.permutation(len(tree_perches))
            for idx in perm:
                tx, ty, tz, tyaw, grp = (float(v) for v in tree_perches[idx])
                if (1000 + int(grp)) in used_groups:
                    continue
                if not _respects_bounds_and_sep(tx, ty, bounds, placed_xy, min_frog_sep_m, spawn_xy, min_spawn_sep_m):
                    continue
                used_groups.add(1000 + int(grp))
                g_z = ground_elevation_z(world_name, tx, ty, tz - 1.6)
                eff_z = max(tz, g_z + 1.35)
                dh = eff_z - g_z
                chosen = {
                    'pos': [round(tx, 3), round(ty, 3), round(eff_z, 3)],
                    'yaw': round(tyaw, 3),
                    'habitat': f'perched {dh:.1f} m up a pine tree trunk',
                    'attenuation': round(float(rng.uniform(0.91, 0.97)), 3),
                }
                break

        if role == 'grass' and len(grass) > 0:
            perm = rng.permutation(len(grass))
            for idx in perm:
                gx, gy, _gz = (float(v) for v in grass[idx])
                jx = gx + float(rng.uniform(-0.25, 0.25))
                jy = gy + float(rng.uniform(-0.25, 0.25))
                if not _is_clear_of_obstacles(jx, jy, obstacles, extra_pad=0.35):
                    continue
                if not _respects_bounds_and_sep(jx, jy, bounds, placed_xy, min_frog_sep_m, spawn_xy, min_spawn_sep_m):
                    continue
                jz = ground_elevation_z(world_name, jx, jy, _gz)
                chosen = {
                    'pos': [round(jx, 3), round(jy, 3), round(jz, 3)],
                    'habitat': 'inside a tall-grass patch',
                    'attenuation': round(float(rng.uniform(0.83, 0.89)), 3),
                }
                break

        if chosen is None:
            # Either `open_or_behind` or fallback if a habitat pool was saturated
            trunks_only = obstacles[obstacles[:, 3] == 0.0] if len(obstacles) > 0 else np.zeros((0, 4))
            if len(trunks_only) > 0 and rng.random() < 0.50:
                tperm = rng.permutation(len(trunks_only))
                for idx in tperm:
                    tcx, tcy, tr, _ = (float(v) for v in trunks_only[idx])
                    # Place on the side opposite to robot spawn ("behind a pine trunk")
                    away = math.atan2(tcy - spawn_xy[1], tcx - spawn_xy[0]) if spawn_xy is not None else 0.0
                    ang = away + float(rng.uniform(-0.45, 0.45))
                    dist = tr + float(rng.uniform(0.55, 0.95))
                    px, py = tcx + dist * math.cos(ang), tcy + dist * math.sin(ang)
                    if not _is_clear_of_obstacles(px, py, obstacles, extra_pad=0.35):
                        continue
                    if not _respects_bounds_and_sep(px, py, bounds, placed_xy, min_frog_sep_m,
                                                    spawn_xy, min_spawn_sep_m):
                        continue
                    pz = ground_elevation_z(world_name, px, py, 2.8)
                    chosen = {
                        'pos': [round(px, 3), round(py, 3), round(pz, 3)],
                        'habitat': 'behind a pine tree trunk',
                        'attenuation': round(float(rng.uniform(0.87, 0.93)), 3),
                    }
                    break
            if chosen is None and bounds is not None:
                xmin, xmax, ymin, ymax = bounds
                for _ in range(250):
                    px = float(rng.uniform(xmin, xmax))
                    py = float(rng.uniform(ymin, ymax))
                    if not _is_clear_of_obstacles(px, py, obstacles, extra_pad=0.50):
                        continue
                    if not _respects_bounds_and_sep(px, py, bounds, placed_xy, min_frog_sep_m,
                                                    spawn_xy, min_spawn_sep_m):
                        continue
                    pz = ground_elevation_z(world_name, px, py, 2.8)
                    chosen = {
                        'pos': [round(px, 3), round(py, 3), round(pz, 3)],
                        'habitat': 'open forest glade' if pref == 'forest' else 'open hillside trail',
                        'attenuation': round(float(rng.uniform(0.95, 1.00)), 3),
                    }
                    break

        if chosen is None:
            return False
        placed_xy.append((chosen['pos'][0], chosen['pos'][1]))
        assignments[slot_idx] = chosen

    for slot_idx, frog_idx in enumerate(role_order):
        fr = frogs[int(frog_idx)]
        spec = assignments[slot_idx]
        fr['pos'] = spec['pos']
        fr['habitat'] = spec['habitat']
        fr['attenuation'] = spec['attenuation']
        if 'yaw' in spec:
            fr['yaw'] = spec['yaw']
        else:
            fr.pop('yaw', None)
    return True


def randomize_truth_scenario(truth_doc: dict, seed: int, survey_area=None,
                             frog_jitter_m: float = 2.5, spawn_jitter_m: float = 1.5,
                             min_frog_sep_m: float = 3.2, min_spawn_sep_m: float = 4.0) -> dict:
    """Create a deterministic randomized frog & spawn layout from a base truth YAML and `seed`.

    - seed == 0: returns a copy of `truth_doc` unchanged (reproducible canonical baseline).
    - seed != 0:
      * When 3D world scene geometry (`world_scenes.npz`) is available for `forest` or `hill`,
        samples genuinely new locations across `survey_area` every seed: inside real dense bushes,
        on top of real tree crowns / fallen tree trunks, perched up real pine trunks, in tall-grass
        patches, and in open glades, with exact 3D surface elevation `z(x, y)`.
      * When `spawn_jitter_m > 0`, perturbs robot spawn XY (avoiding obstacles and snapping Z to
        terrain); pass `spawn_jitter_m=0.0` when Terminal 2 uses a fixed EKF trailhead `initial_pose`.
    """
    out = copy.deepcopy(truth_doc)
    if int(seed) == 0:
        return out
    rng = np.random.default_rng(int(seed) * 10007 + 42)
    margin = 1.8
    world_name = out.get('world', '')
    pref = _scene_prefix(world_name)
    scenes = load_world_scenes()
    obstacles = scenes.get(f'{pref}_obstacles', np.zeros((0, 4))) if pref else np.zeros((0, 4))

    if survey_area is None and pref == 'forest':
        survey_area = [18.0, 50.0, -15.0, 15.0]
    elif survey_area is None and pref == 'hill':
        survey_area = [-39.0, 48.0, -16.5, 16.5]

    bounds = None
    if survey_area is not None and len(survey_area) == 4:
        xmin, xmax, ymin, ymax = (float(v) for v in survey_area)
        bounds = (xmin + margin, xmax - margin, ymin + margin, ymax - margin)

    # Optionally perturb spawn pose along clear trailhead ground
    spawn_xy = None
    if 'spawn' in out and len(out['spawn']) >= 4:
        sx, sy, sz, syaw = (float(v) for v in out['spawn'][:4])
        if spawn_jitter_m > 0.0:
            nx, ny = sx, sy
            for _ in range(30):
                cand_x = sx + float(rng.uniform(-spawn_jitter_m, spawn_jitter_m))
                cand_y = sy + float(rng.uniform(-0.5 * spawn_jitter_m, 0.5 * spawn_jitter_m))
                if bounds is not None:
                    cand_x = float(np.clip(cand_x, bounds[0], bounds[1]))
                    cand_y = float(np.clip(cand_y, bounds[2], bounds[3]))
                if _is_clear_of_obstacles(cand_x, cand_y, obstacles, extra_pad=0.6):
                    nx, ny = cand_x, cand_y
                    break
            dyaw = float(rng.uniform(-0.4, 0.4))
            nyaw = math.atan2(math.sin(syaw + dyaw), math.cos(syaw + dyaw))
            nz = ground_elevation_z(world_name, nx, ny, sz - 0.08) + 0.08 if pref else sz
            out['spawn'] = [round(nx, 3), round(ny, 3), round(nz, 3), round(nyaw, 3)]
        spawn_xy = (float(out['spawn'][0]), float(out['spawn'][1]))

    if pref and f'{pref}_grid_z' in scenes and bounds is not None:
        if _randomize_with_scene(out, pref, scenes, rng, bounds, min_frog_sep_m, spawn_xy, min_spawn_sep_m):
            out['seed'] = int(seed)
            return out

    # Generic fallback when custom truth_doc has no precomputed 3D world scene
    placed = []
    for fr in out.get('frogs', []):
        fx, fy, fz = (float(v) for v in fr['pos'][:3])
        best_xy = (fx, fy)
        for _ in range(60):
            if bounds is not None:
                cx = float(rng.uniform(bounds[0], bounds[1]))
                cy = float(rng.uniform(bounds[2], bounds[3]))
            else:
                r = float(rng.uniform(0.5, frog_jitter_m))
                ang = float(rng.uniform(-math.pi, math.pi))
                cx, cy = fx + r * math.cos(ang), fy + r * math.sin(ang)
            if _respects_bounds_and_sep(cx, cy, bounds, placed, min_frog_sep_m, spawn_xy, min_spawn_sep_m):
                best_xy = (cx, cy)
                break
        placed.append(best_xy)
        nz = ground_elevation_z(world_name, best_xy[0], best_xy[1], fz) if pref else fz
        fr['pos'] = [round(best_xy[0], 3), round(best_xy[1], 3), round(nz, 3)]
        att = float(fr.get('attenuation', 1.0))
        fr['attenuation'] = round(float(np.clip(att * rng.uniform(0.90, 1.05), 0.65, 1.0)), 3)
    out['seed'] = int(seed)
    return out
