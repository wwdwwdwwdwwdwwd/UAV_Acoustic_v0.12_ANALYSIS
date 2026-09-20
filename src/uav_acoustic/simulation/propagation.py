from __future__ import annotations
import numpy as np
from scipy.interpolate import interp1d
from ..types import AcousticFrame
from ..coordinates import az_el_to_unit


def az_el_range_to_xyz(az_deg: float, el_deg: float, distance_m: float) -> np.ndarray:
    """Convert the documented +Z-front angles to a physical XYZ point."""
    return distance_m * az_el_to_unit(az_deg, el_deg)


def simulate_free_field(
    source_signal: np.ndarray,
    fs: int,
    mic_xyz: np.ndarray,
    source_xyz: np.ndarray,
    c: float = 343.0,
    snr_db: float | None = 10.0,
    seed: int = 0,
) -> AcousticFrame:
    """Generate multichannel free-field data with fractional propagation delay.

    The common absolute propagation time is removed, preserving relative TDOA.
    1/r attenuation is retained relative to the nearest microphone.
    """
    rng = np.random.default_rng(seed)
    mic_xyz = np.asarray(mic_xyz, float)
    source_xyz = np.asarray(source_xyz, float)
    distances = np.linalg.norm(source_xyz[None, :] - mic_xyz, axis=1)
    delays = (distances - distances.min()) / c
    t = np.arange(source_signal.size) / fs
    out = np.zeros((mic_xyz.shape[0], source_signal.size), dtype=np.float32)
    for m, (dist, delay) in enumerate(zip(distances, delays)):
        f = interp1d(t, source_signal, kind="linear", bounds_error=False, fill_value=0.0)
        shifted = f(t - delay)
        # Normalize attenuation relative to nearest microphone to avoid tiny numbers.
        shifted *= distances.min() / max(dist, 1e-6)
        out[m] = shifted.astype(np.float32)
    if snr_db is not None:
        p_sig = float(np.mean(out**2)) + 1e-12
        p_noise = p_sig / (10 ** (snr_db / 10.0))
        out += rng.normal(scale=np.sqrt(p_noise), size=out.shape).astype(np.float32)
    return AcousticFrame(audio=out, fs=fs, mic_xyz=mic_xyz.copy())
