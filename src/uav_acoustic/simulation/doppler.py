from __future__ import annotations
import numpy as np
from scipy.signal import resample_poly


def doppler_shift(signal: np.ndarray, fs: int, radial_velocity_mps: float, c: float = 343.0) -> np.ndarray:
    """Approximate moving-source Doppler by time scaling.

    Positive radial_velocity_mps means approaching, so observed frequency rises.
    """
    if abs(radial_velocity_mps) >= c:
        raise ValueError("radial velocity magnitude must be below sound speed")
    factor = c / (c - radial_velocity_mps)
    # Rational approximation for robust resampling.
    from fractions import Fraction
    frac = Fraction(float(factor)).limit_denominator(10000)
    # To raise frequency by factor, compress the time axis. The old implementation
    # used the inverse operation and therefore made an approaching source lower.
    y = resample_poly(signal, frac.denominator, frac.numerator)
    if y.size >= signal.size:
        return y[: signal.size].astype(np.float32)
    return np.pad(y, (0, signal.size-y.size)).astype(np.float32)


def generate_radial_doppler_tone(
    duration_s: float,
    fs: int,
    source_frequency_hz: float,
    radial_velocity_mps: np.ndarray | float,
    c: float = 343.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a continuous tone with a prescribed radial-velocity trajectory.

    Positive velocity means approaching. This is an algorithm regression signal,
    not a model of WaveFrag sampling-clock or field accuracy.
    """
    n = int(round(duration_s * fs))
    velocity = np.broadcast_to(np.asarray(radial_velocity_mps, float), (n,)).copy()
    if np.any(np.abs(velocity) >= c):
        raise ValueError("radial velocity magnitude must be below sound speed")
    observed_hz = source_frequency_hz * c / (c - velocity)
    phase = 2.0 * np.pi * np.cumsum(observed_hz) / fs
    return np.sin(phase).astype(np.float32), observed_hz
