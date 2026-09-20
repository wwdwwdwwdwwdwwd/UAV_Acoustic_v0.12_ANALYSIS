from __future__ import annotations
import numpy as np


def nlms_cancel(primary: np.ndarray, reference: np.ndarray, taps: int = 128, mu: float = 0.3, eps: float = 1e-8):
    """Basic NLMS adaptive noise cancellation baseline.

    primary = desired UAV + correlated vehicle noise
    reference = correlated vehicle noise reference
    Real performance depends critically on reference microphone placement.
    """
    d = np.asarray(primary, float)
    x = np.asarray(reference, float)
    n = min(len(d), len(x))
    w = np.zeros(taps)
    e = np.zeros(n)
    hist = np.zeros(taps)
    for i in range(n):
        hist[1:] = hist[:-1]
        hist[0] = x[i]
        y = np.dot(w, hist)
        e[i] = d[i] - y
        w += (mu * e[i] / (eps + np.dot(hist,hist))) * hist
    return e.astype(np.float32), w
