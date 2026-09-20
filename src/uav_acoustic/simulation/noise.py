from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt


def vehicle_like_noise(n_samples: int, fs: int, seed: int = 0) -> np.ndarray:
    """Simple vehicle/engine surrogate for algorithm stress tests.

    Contains low-frequency engine harmonics plus broadband fan/road noise.
    It is deliberately not claimed to reproduce a specific truck.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fs
    base = rng.uniform(55, 95)
    y = np.zeros(n_samples)
    for h in range(1, 12):
        f = h * base
        if f >= fs/2:
            break
        y += (0.8**(h-1)) * np.sin(2*np.pi*f*t + rng.uniform(0,2*np.pi))
    w = rng.standard_normal(n_samples)
    hi = min(6000, fs*0.45)
    sos = butter(4, hi, btype="lowpass", fs=fs, output="sos")
    w = sosfiltfilt(sos, w)
    w /= np.std(w)+1e-12
    y += 1.5*w
    y /= np.std(y)+1e-12
    return y.astype(np.float32)
