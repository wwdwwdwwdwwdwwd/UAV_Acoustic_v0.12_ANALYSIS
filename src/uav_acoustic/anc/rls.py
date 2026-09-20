from __future__ import annotations
import numpy as np


def rls_cancel(
    primary: np.ndarray,
    reference: np.ndarray,
    taps: int = 32,
    forgetting: float = 0.995,
    delta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """RLS simulation baseline for a synchronized reference microphone."""
    if not 0.0 < forgetting <= 1.0:
        raise ValueError("forgetting must be in (0, 1]")
    d = np.asarray(primary, float)
    x = np.asarray(reference, float)
    n = min(d.size, x.size)
    w = np.zeros(taps)
    p = np.eye(taps) / delta
    hist = np.zeros(taps)
    error = np.zeros(n)
    for i in range(n):
        hist[1:] = hist[:-1]
        hist[0] = x[i]
        ph = p @ hist
        gain = ph / (forgetting + hist @ ph)
        prediction = w @ hist
        error[i] = d[i] - prediction
        w += gain * error[i]
        p = (p - np.outer(gain, hist) @ p) / forgetting
    return error.astype(np.float32), w
