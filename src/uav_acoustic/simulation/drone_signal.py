from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt


def generate_drone_signal(
    duration_s: float,
    fs: int,
    bpf_hz: float = 220.0,
    n_harmonics: int = 14,
    harmonic_decay: float = 0.72,
    broadband_level: float = 0.18,
    seed: int = 0,
) -> np.ndarray:
    """Synthetic drone-like source: BPF harmonic comb + 1-5 kHz broadband energy.

    This is a functional test signal, not a validated physical UAV model.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    t = np.arange(n) / fs
    sig = np.zeros(n, dtype=float)
    # Slight RPM modulation prevents an unrealistically stationary comb.
    mod = 1.0 + 0.012 * np.sin(2 * np.pi * 0.7 * t)
    phase_base = 2 * np.pi * np.cumsum((bpf_hz * mod) / fs)
    for h in range(1, n_harmonics + 1):
        if h * bpf_hz >= 0.45 * fs:
            break
        sig += (harmonic_decay ** (h - 1)) * np.sin(h * phase_base + rng.uniform(0, 2*np.pi))
    noise = rng.standard_normal(n)
    lo, hi = 1000.0, min(5000.0, fs * 0.45)
    if hi > lo:
        sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
        noise = sosfiltfilt(sos, noise)
        noise /= np.std(noise) + 1e-12
        sig += broadband_level * noise
    sig /= np.max(np.abs(sig)) + 1e-12
    return sig.astype(np.float32)
