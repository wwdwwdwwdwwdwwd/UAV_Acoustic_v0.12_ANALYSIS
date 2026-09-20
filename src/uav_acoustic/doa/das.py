from __future__ import annotations
import numpy as np
from scipy.signal import get_window

from ..coordinates import az_el_to_unit


def _unit_vectors(azimuth_deg: np.ndarray, elevation_deg: np.ndarray) -> np.ndarray:
    return az_el_to_unit(azimuth_deg, elevation_deg)


def narrowband_steered_power(audio: np.ndarray, fs: int, mic_xyz: np.ndarray, freq_hz: float, az_grid: np.ndarray, el_grid: np.ndarray, c: float = 343.0) -> np.ndarray:
    """Simple frequency-domain delay-and-sum scan for validation/debug.

    Returns power[el, az]. This is intentionally simple rather than optimized.
    """
    audio = np.asarray(audio, float)
    # A bounded central segment retains the phase/TDOA physics while avoiding
    # repeated 160k-point FFTs during multi-harmonic scans.
    if audio.shape[1] > 32768:
        start = (audio.shape[1] - 32768) // 2
        audio = audio[:, start:start + 32768]
    n = audio.shape[1]
    freqs = np.fft.rfftfreq(n, 1/fs)
    k = int(np.argmin(np.abs(freqs - freq_hz)))
    X = np.fft.rfft(audio, axis=1)[:, k]
    out = np.zeros((len(el_grid), len(az_grid)), dtype=float)
    pos = mic_xyz - mic_xyz.mean(axis=0, keepdims=True)
    for ie, el in enumerate(el_grid):
        dirs = _unit_vectors(np.asarray(az_grid), np.full_like(az_grid, el, dtype=float))
        delays = -(dirs @ pos.T) / c  # (A,M)
        steer = np.exp(1j * 2*np.pi*freq_hz*delays)
        y = steer @ X
        out[ie] = np.abs(y)**2
    return out


def peak_from_map(power: np.ndarray, az_grid: np.ndarray, el_grid: np.ndarray) -> tuple[float,float,float]:
    ie, ia = np.unravel_index(np.argmax(power), power.shape)
    return float(az_grid[ia]), float(el_grid[ie]), float(power[ie,ia])


def map_diagnostics(
    power: np.ndarray,
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    *,
    top_k: int = 5,
    exclusion_deg: float = 10.0,
) -> dict:
    """Explain peak strength and ambiguity without hiding secondary lobes."""
    p = np.asarray(power, float)
    finite = np.where(np.isfinite(p), p, 0.0)
    peak = float(np.max(finite))
    floor = float(np.median(finite))
    candidates = []
    work = finite.copy()
    da = float(np.median(np.diff(az_grid))) if len(az_grid) > 1 else 1.0
    de = float(np.median(np.diff(el_grid))) if len(el_grid) > 1 else 1.0
    ra = max(1, int(round(exclusion_deg / abs(da))))
    re = max(1, int(round(exclusion_deg / abs(de))))
    for _ in range(top_k):
        ie, ia = np.unravel_index(np.argmax(work), work.shape)
        value = float(work[ie, ia])
        if not np.isfinite(value) or value < 0:
            break
        candidates.append({
            "azimuth_deg": float(az_grid[ia]), "elevation_deg": float(el_grid[ie]),
            "normalized_power": value / max(peak, 1e-30),
        })
        work[max(0, ie-re):ie+re+1, max(0, ia-ra):ia+ra+1] = -np.inf
    second = candidates[1]["normalized_power"] if len(candidates) > 1 else 0.0
    half = floor + 0.5 * max(0.0, peak-floor)
    width_fraction = float(np.mean(finite >= half))
    sharpness = float(np.clip((peak - np.percentile(finite, 90)) / max(peak-floor, 1e-30), 0, 1))
    psr_db = float(10*np.log10(max(peak, 1e-30) / max(floor, 1e-30)))
    return {
        "peak_to_sidelobe_db": psr_db,
        "normalized_peak_sharpness": sharpness,
        "top1_top2_difference": float(1.0-second),
        "half_power_area_fraction": width_fraction,
        "ambiguous": bool(second >= 0.90 or width_fraction >= 0.15),
        "top_candidates": candidates,
    }


def farfield_broadband_das(
    audio: np.ndarray,
    fs: int,
    mic_xyz: np.ndarray,
    *,
    azimuth_grid_deg: np.ndarray,
    elevation_deg: float = 0.0,
    freq_range=(1200.0, 2000.0),
    nfft: int = 4096,
    c: float = 343.0,
) -> dict:
    """Direction-only conventional broadband delay-and-sum beamformer.

    Steering uses only microphone offsets and a unit direction.  There is no
    focal plane, source range, or spherical-wave distance in this API.
    """
    a = np.asarray(audio, float)
    xyz = np.asarray(mic_xyz, float)
    azimuth = np.asarray(azimuth_grid_deg, float)
    if a.ndim != 2 or a.shape[0] != len(xyz):
        raise ValueError("audio/microphone layout mismatch")
    if azimuth.ndim != 1 or azimuth.size < 2:
        raise ValueError("azimuth_grid_deg must contain at least two directions")
    n = min(a.shape[1], int(nfft))
    start = max(0, (a.shape[1]-n)//2)
    segment = a[:, start:start+n] * get_window("hann", n)[None, :]
    spectrum = np.fft.rfft(segment, n=nfft, axis=1)
    frequency = np.fft.rfftfreq(nfft, 1/fs)
    bins = np.flatnonzero((frequency >= float(freq_range[0])) &
                          (frequency <= min(float(freq_range[1]), fs/2)))
    if bins.size == 0:
        raise ValueError(f"no FFT bins in far-field DAS band {freq_range}")
    bins = bins[np.linspace(0, bins.size-1, min(96, bins.size)).astype(int)]
    position = xyz - xyz.mean(axis=0, keepdims=True)
    directions = az_el_to_unit(azimuth, np.full(azimuth.shape, float(elevation_deg)))
    geometric_delay = directions @ position.T / float(c)  # directions x microphones
    score = np.zeros(azimuth.size, float)
    for k in bins:
        # Far-field recording phase is +w*u.r/c; steering removes that phase.
        steering = np.exp(-2j*np.pi*frequency[k]*geometric_delay)
        beam = steering @ spectrum[:, k]
        score += np.abs(beam)**2
    power = score[None, :]
    diagnostics = map_diagnostics(power, azimuth, np.asarray([float(elevation_deg)]),
                                  exclusion_deg=0.5)
    peak_index = int(np.argmax(score))
    normalized = score / max(float(np.max(score)), 1e-30)
    return {
        "azimuth_deg": float(azimuth[peak_index]),
        "elevation_deg": float(elevation_deg),
        "frequency_band_hz": [float(freq_range[0]), float(freq_range[1])],
        "beam_peak_power": float(score[peak_index]),
        "beam_response": normalized.tolist(),
        "azimuth_grid_deg": azimuth.tolist(),
        "diagnostics": diagnostics,
        "implementation": "farfield_frequency_domain_delay_and_sum_v1",
        "range_model": "direction_only_no_source_distance",
    }


def vendor_nearfield_das(
    audio: np.ndarray,
    fs: int,
    mic_xyz: np.ndarray,
    *,
    x_grid=None,
    y_grid=None,
    focal_z_m: float = 2.0,
    azimuth_fov_deg: float = 70.0,
    elevation_fov_deg: float = 45.0,
    freq_range=(1000.0, 5000.0),
    nfft: int = 4096,
    c: float = 343.0,
) -> dict:
    """Frequency-domain equivalent of vendor near-field delay-and-sum.

    It uses the vendor geometry, X/Y grid, +Z focal plane and equal weights.
    Only the requested band is integrated, making it practical offline while
    retaining the same physical propagation-delay model as DAS_20cm.m.
    """
    a = np.asarray(audio, float)
    if focal_z_m <= 0:
        raise ValueError("focal_z_m must be positive")
    # The former fixed +/-1 m focal plane silently capped the available angle
    # to +/-26.6 deg at z=2 m and +/-1.9 deg at z=30 m.  Scale the Cartesian
    # vendor focal plane from the declared angular FOV so synthetic +/-60 deg
    # remains representable at every focal distance.
    if x_grid is None:
        x_extent = focal_z_m * np.tan(np.deg2rad(float(azimuth_fov_deg)))
        x_grid = np.linspace(-x_extent, x_extent, 97)
    if y_grid is None:
        y_extent = focal_z_m * np.tan(np.deg2rad(float(elevation_fov_deg)))
        y_grid = np.linspace(-y_extent, y_extent, 65)
    n = min(a.shape[1], nfft)
    start = max(0, (a.shape[1]-n)//2)
    segment = a[:, start:start+n] * get_window("hann", n)[None, :]
    X = np.fft.rfft(segment, n=nfft, axis=1)
    frequencies = np.fft.rfftfreq(nfft, 1/fs)
    bins = np.flatnonzero((frequencies >= freq_range[0]) & (frequencies <= freq_range[1]))
    if bins.size == 0:
        raise ValueError(f"no FFT bins in vendor DAS band {freq_range}")
    # Cap frequency samples without changing band endpoints/physics.
    bins = bins[np.linspace(0, bins.size-1, min(96, bins.size)).astype(int)]
    xx, yy = np.meshgrid(np.asarray(x_grid, float), np.asarray(y_grid, float))
    focal = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, focal_z_m)], axis=1)
    distances = np.linalg.norm(focal[:, None, :] - np.asarray(mic_xyz)[None, :, :], axis=2)
    distances -= distances.min(axis=1, keepdims=True)
    score = np.zeros(focal.shape[0])
    for k in bins:
        # Recorded propagation contributes exp(-j*w*t); compensating delay is +j*w*t.
        steer = np.exp(1j * 2*np.pi*frequencies[k] * distances / c)
        score += np.abs(np.sum(steer * X[:, k][None, :], axis=1))**2
    heatmap = score.reshape(yy.shape)
    diag = map_diagnostics(heatmap, np.asarray(x_grid), np.asarray(y_grid), exclusion_deg=0.2)
    iy, ix = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    point = np.array([x_grid[ix], y_grid[iy], focal_z_m], float)
    from ..coordinates import xyz_to_az_el
    az, el = xyz_to_az_el(point)
    return {
        "focal_point_m": point.tolist(), "azimuth_deg": float(az), "elevation_deg": float(el),
        "normalized_heatmap": (heatmap / max(float(heatmap.max()), 1e-30)).tolist(),
        "x_grid_m": np.asarray(x_grid, float).tolist(), "y_grid_m": np.asarray(y_grid, float).tolist(),
        "diagnostics": diag,
        "confidence": {"score": float(np.clip(0.4*diag["normalized_peak_sharpness"] +
                                    1.5*diag["top1_top2_difference"], 0, 1)),
                       "status": "LOW_CONFIDENCE" if diag["ambiguous"] else "VALID"},
    }


def vendor_matlab_time_das(
    audio: np.ndarray,
    fs: int,
    mic_xyz: np.ndarray,
    *,
    x_grid=np.linspace(-1.0, 1.0, 21),
    y_grid=np.linspace(-1.0, 1.0, 17),
    focal_z_m: float = 2.0,
    frame_samples: int = 4096,
    c: float = 343.0,
) -> dict:
    """Literal integer-delay equivalent of the core DAS_20cm.m loop."""
    a = np.asarray(audio, float)
    n = min(int(frame_samples), a.shape[1])
    start = max(0, (a.shape[1] - n) // 2)
    data = a[:, start : start + n].T  # MATLAB dataNew: samples x microphones
    xx, yy = np.meshgrid(np.asarray(x_grid, float), np.asarray(y_grid, float))
    focal = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, focal_z_m)], axis=1)
    distance = np.linalg.norm(focal[:, None, :] - np.asarray(mic_xyz)[None, :, :], axis=2)
    delay = np.rint(distance / c * fs).astype(int)
    delay = delay - int(delay.min()) + 1
    max_delay = int(delay.max())
    buffer = np.zeros((n + max_delay, data.shape[1]), dtype=float)
    buffer[max_delay : max_delay + n] = data
    score = np.zeros(len(focal), float)
    base = np.arange(n)
    for b in range(len(focal)):
        summed = np.zeros(n, float)
        for m in range(data.shape[1]):
            summed += buffer[delay[b, m] + base, m]
        score[b] = np.dot(summed, summed)
    heatmap = score.reshape(xx.shape)
    iy, ix = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    point = np.array([x_grid[ix], y_grid[iy], focal_z_m], float)
    from ..coordinates import xyz_to_az_el
    az, el = xyz_to_az_el(point)
    return {"focal_point_m": point.tolist(), "azimuth_deg": float(az), "elevation_deg": float(el),
            "normalized_heatmap": (heatmap / max(float(heatmap.max()), 1e-30)).tolist(),
            "x_grid_m": np.asarray(x_grid, float).tolist(), "y_grid_m": np.asarray(y_grid, float).tolist(),
            "implementation": "literal_DAS_20cm_integer_delay"}
