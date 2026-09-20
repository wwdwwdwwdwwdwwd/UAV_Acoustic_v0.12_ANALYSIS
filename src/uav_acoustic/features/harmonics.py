from __future__ import annotations
import numpy as np
from scipy.signal import get_window


def mean_spectrum(audio: np.ndarray, fs: int, nfft: int = 8192):
    """Return frequency axis and mean magnitude spectrum across channels."""
    audio = np.asarray(audio, float)
    if audio.ndim == 1:
        audio = audio[None, :]
    if audio.ndim != 2 or audio.shape[1] < 2:
        raise ValueError("audio must have shape (channels, samples) with at least two samples")
    n = min(audio.shape[1], nfft)
    x = audio[:, :n] * get_window("hann", n)[None, :]
    X = np.fft.rfft(x, n=nfft, axis=1)
    mag = np.mean(np.abs(X), axis=0)
    f = np.fft.rfftfreq(nfft, 1/fs)
    return f, mag


def harmonic_comb_score(audio: np.ndarray, fs: int, bpf_candidates=np.arange(100.0, 501.0, 2.0), max_hz: float = 3000.0, nfft: int = 8192):
    """Find BPF candidate whose harmonic comb captures the most spectral energy."""
    f, mag = mean_spectrum(audio, fs, nfft=nfft)
    mag = mag / (np.median(mag) + 1e-12)
    scores = []
    for bpf in bpf_candidates:
        hs = np.arange(1, int(max_hz // bpf)+1) * bpf
        bins = np.searchsorted(f, hs)
        bins = np.clip(bins, 1, len(mag)-2)
        local = np.array([np.max(mag[k-1:k+2]) for k in bins])
        # Remove the local floor. Without this contrast term, broadband spectra
        # can receive a misleadingly strong comb score simply by summation.
        local_floor = np.array([np.median(mag[max(0, k-4):min(len(mag), k+5)]) for k in bins])
        local = np.maximum(0.0, local - local_floor)
        # Give higher weight to lower harmonics but retain upper harmonics.
        weights = 1.0 / np.sqrt(np.arange(1, len(local)+1))
        scores.append(float(np.sum(local * weights) / np.sum(weights)))
    scores = np.asarray(scores)
    i = int(np.argmax(scores))
    return {
        "bpf_hz": float(np.asarray(bpf_candidates)[i]),
        "score": float(scores[i]),
        "candidates_hz": np.asarray(bpf_candidates),
        "scores": scores,
    }
