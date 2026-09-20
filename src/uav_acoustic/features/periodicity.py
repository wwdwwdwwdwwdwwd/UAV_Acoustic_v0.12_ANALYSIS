from __future__ import annotations
import numpy as np
from scipy.signal import correlate


def periodicity_score(signal: np.ndarray, fs: int, freq_range=(80.0, 500.0)) -> dict:
    x = np.asarray(signal, float)
    if x.ndim != 1 or x.size < 3:
        raise ValueError("signal must be a one-dimensional array with at least three samples")
    x = x - np.mean(x)
    if np.std(x) < 1e-12:
        return {"score": 0.0, "period_hz": None}
    ac = correlate(x, x, mode="full", method="fft")[len(x)-1:]
    ac /= ac[0] + 1e-12
    lag_min = max(1, int(fs / freq_range[1]))
    lag_max = min(len(ac)-1, int(fs / freq_range[0]))
    if lag_max <= lag_min:
        return {"score": 0.0, "period_hz": None}
    k = lag_min + int(np.argmax(ac[lag_min:lag_max+1]))
    return {"score": float(ac[k]), "period_hz": float(fs / k)}
