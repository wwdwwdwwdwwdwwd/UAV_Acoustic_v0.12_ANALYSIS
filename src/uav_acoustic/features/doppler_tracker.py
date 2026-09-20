from __future__ import annotations
import numpy as np
from scipy.signal import stft


def track_peak_frequency(signal: np.ndarray, fs: int, band=(100.0, 3000.0), window_s: float = 0.5, hop_s: float = 0.1) -> dict:
    signal = np.asarray(signal, float)
    if signal.ndim != 1 or signal.size < 4:
        raise ValueError("signal must be a one-dimensional array with at least four samples")
    nperseg = min(signal.size, max(256, int(round(window_s * fs))))
    noverlap = max(0, nperseg - int(round(hop_s * fs)))
    f, t, Z = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, boundary=None)
    mask = (f >= band[0]) & (f <= band[1])
    fb = f[mask]
    mag = np.abs(Z[mask])
    if fb.size == 0:
        raise ValueError(f"tracking band {band} has no STFT bins")
    peaks = np.zeros(mag.shape[1])
    for i in range(mag.shape[1]):
        k = int(np.argmax(mag[:, i]))
        # Parabolic interpolation in log-magnitude for sub-bin estimate.
        if 0 < k < len(fb)-1:
            y = np.log(mag[k-1:k+2, i] + 1e-12)
            denom = y[0] - 2*y[1] + y[2]
            delta = 0.5*(y[0]-y[2])/denom if abs(denom) > 1e-12 else 0.0
            df = fb[1]-fb[0]
            peaks[i] = fb[k] + delta*df
        else:
            peaks[i] = fb[k]
    return {"time_s": t, "peak_hz": peaks}
