"""Pure-numpy Coqui acoustic synthesis, motor noise, harmonic-free SNR, SRP-PHAT DOA,
and seeded scenario randomization (ROS-free so offline tests can exercise all acoustic physics).
"""
import copy
import math
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


def propagate_sources_to_array(sources, array_origin_world, robot_rot_world,
                               noise_std=0.008, motor_speed=0.0, motor_wz=0.0,
                               motor_noise_gain=0.0, echo_gain=0.0, echo_delay_ms=18.0,
                               ambient_clutter=False, rng=None):
    """Simulate 5-channel microphone waveforms for 0, 1, or multiple overlapping frog calls.

    `sources`: list of dicts {'pos': (x,y,z), 'waveform': np.ndarray(FRAME_SAMPLES), 'foliage_attenuation': float}
    Returns (5, FRAME_SAMPLES) float64 array.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    num_mics = MIC_POSITIONS_BODY.shape[0]
    mic_world_pts = array_origin_world + (MIC_POSITIONS_BODY @ robot_rot_world.T)
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
    return multi_ch


def randomize_truth_scenario(truth_doc: dict, seed: int, survey_area=None,
                             frog_jitter_m: float = 2.5, spawn_jitter_m: float = 1.5,
                             min_frog_sep_m: float = 3.0) -> dict:
    """Create a deterministic randomized frog & spawn layout from a base truth YAML and `seed`.

    - seed == 0: returns a copy of `truth_doc` unchanged (reproducible canonical baseline).
    - seed != 0: perturbs each frog's XY position within `frog_jitter_m` (clamped inside `survey_area`
      with margin and enforcing `min_frog_sep_m` pairwise separation), jitters attenuation slightly,
      and perturbs robot spawn XY (within `spawn_jitter_m`) and yaw (within +-0.4 rad).
    """
    out = copy.deepcopy(truth_doc)
    if int(seed) == 0:
        return out
    rng = np.random.default_rng(int(seed) * 10007 + 42)
    margin = 2.0
    xmin, xmax, ymin, ymax = (None, None, None, None)
    if survey_area is not None and len(survey_area) == 4:
        xmin, xmax, ymin, ymax = (float(v) for v in survey_area)

    # Perturb spawn pose slightly along safe trail margin
    if 'spawn' in out and len(out['spawn']) >= 4:
        sx, sy, sz, syaw = (float(v) for v in out['spawn'][:4])
        dx = float(rng.uniform(-spawn_jitter_m, spawn_jitter_m))
        dy = float(rng.uniform(-0.5 * spawn_jitter_m, 0.5 * spawn_jitter_m))
        nx, ny = sx + dx, sy + dy
        if xmin is not None:
            nx = float(np.clip(nx, xmin + margin, xmax - margin))
            ny = float(np.clip(ny, ymin + margin, ymax - margin))
        nyaw = math.atan2(math.sin(syaw + rng.uniform(-0.4, 0.4)),
                          math.cos(syaw + rng.uniform(-0.4, 0.4)))
        out['spawn'] = [round(nx, 3), round(ny, 3), round(sz, 3), round(nyaw, 3)]

    placed = []
    for fr in out.get('frogs', []):
        fx, fy, fz = (float(v) for v in fr['pos'][:3])
        best_xy = (fx, fy)
        for _ in range(30):
            r = float(rng.uniform(0.5, frog_jitter_m))
            ang = float(rng.uniform(-math.pi, math.pi))
            cx, cy = fx + r * math.cos(ang), fy + r * math.sin(ang)
            if xmin is not None:
                cx = float(np.clip(cx, xmin + margin, xmax - margin))
                cy = float(np.clip(cy, ymin + margin, ymax - margin))
            if all(math.hypot(cx - px, cy - py) >= min_frog_sep_m for px, py in placed):
                best_xy = (cx, cy)
                break
        placed.append(best_xy)
        fr['pos'] = [round(best_xy[0], 3), round(best_xy[1], 3), round(fz, 3)]
        att = float(fr.get('attenuation', 1.0))
        fr['attenuation'] = round(float(np.clip(att * rng.uniform(0.90, 1.05), 0.65, 1.0)), 3)
    out['seed'] = int(seed)
    return out
