"""NumPy/SciPy broadband SRP-PHAT; pyroomacoustics is not required."""
from __future__ import annotations

import numpy as np
from scipy.signal import stft

from ..coordinates import az_el_to_unit, CONVENTION
from .das import map_diagnostics


def _spanning_pairs(
    mic_xyz: np.ndarray, max_pairs: int, *, max_unaliased_spacing_m: float | None = None
) -> np.ndarray:
    """Deterministic local/mid-baseline subset with spatial-alias control."""
    m = len(mic_xyz)
    i, j = np.triu_indices(m, 1)
    distance = np.linalg.norm(mic_xyz[i] - mic_xyz[j], axis=1)
    eligible = np.arange(len(distance))
    if max_unaliased_spacing_m is not None:
        local = eligible[distance <= max_unaliased_spacing_m]
        if len(local) >= max(32, max_pairs // 4):
            eligible = local
    order = eligible[np.argsort(distance[eligible])]
    if len(order) > max_pairs:
        # Half nearest-neighbour pairs, half evenly across the still-unaliased set.
        nearest = order[: max_pairs // 2]
        rest = order[max_pairs // 2 :]
        sampled = rest[np.linspace(0, len(rest) - 1, max_pairs - len(nearest)).astype(int)]
        order = np.concatenate([nearest, sampled])
    return np.stack([i[order], j[order]], axis=1)


def srp_phat_map(
    audio: np.ndarray,
    fs: int,
    mic_xyz: np.ndarray,
    *,
    freq_range=(1000.0, 5000.0),
    nfft: int = 1024,
    azimuth_grid_deg=np.arange(-90.0, 90.1, 4.0),
    elevation_grid_deg=np.arange(-60.0, 60.1, 4.0),
    max_pairs: int = 768,
    c: float = 343.0,
) -> np.ndarray:
    audio = np.asarray(audio, float)
    mic_xyz = np.asarray(mic_xyz, float)
    if audio.ndim != 2 or audio.shape[0] != len(mic_xyz):
        raise ValueError("audio/microphone layout mismatch")
    if audio.shape[1] > 32768:
        start = (audio.shape[1]-32768)//2
        audio = audio[:, start:start+32768]
    # Limit frame count for bounded CPU while sampling the whole recording.
    freq, _, Z = stft(audio, fs=fs, window="hann", nperseg=nfft,
                      noverlap=nfft//2, nfft=nfft, axis=-1, boundary=None, padded=False)
    bins = np.flatnonzero((freq >= freq_range[0]) & (freq <= min(freq_range[1], fs/2)))
    if bins.size == 0:
        raise ValueError(f"no STFT bins in SRP-PHAT band {freq_range}")
    bins = bins[np.linspace(0, bins.size-1, min(72, bins.size)).astype(int)]
    frames = np.linspace(0, Z.shape[-1]-1, min(24, Z.shape[-1])).astype(int)
    # Half-wavelength bound prevents the old long-baseline/5 kHz grating-lobe lock.
    max_spacing = c / (2.0 * float(freq_range[1]))
    pairs = _spanning_pairs(mic_xyz, max_pairs, max_unaliased_spacing_m=max_spacing)
    cross = Z[pairs[:, 0]][:, bins][:, :, frames] * np.conj(Z[pairs[:, 1]][:, bins][:, :, frames])
    cross = np.mean(cross / np.maximum(np.abs(cross), 1e-12), axis=2)  # pairs x bins
    az = np.asarray(azimuth_grid_deg, float)
    el = np.asarray(elevation_grid_deg, float)
    aa, ee = np.meshgrid(az, el)
    directions = az_el_to_unit(aa.ravel(), ee.ravel())
    delta = mic_xyz[pairs[:, 0]] - mic_xyz[pairs[:, 1]]
    tau = directions @ delta.T / c  # directions x pairs
    scores = np.zeros(len(directions))
    # X_i conj(X_j) has +w*tau; use the negative steering phase.
    for start in range(0, len(directions), 256):
        t = tau[start:start+256]
        phase = np.exp(-2j*np.pi * t[:, :, None] * freq[bins][None, None, :])
        scores[start:start+256] = np.real(np.einsum("dpf,pf->d", phase, cross, optimize=True))
    scores -= scores.min()
    return scores.reshape(len(el), len(az))


def estimate_srp_phat(
    audio: np.ndarray,
    fs: int,
    mic_xyz: np.ndarray,
    freq_range=(1000.0, 5000.0),
    nfft: int = 1024,
    num_src: int = 1,
    azimuth_grid_deg=np.arange(-90.0, 90.1, 4.0),
    elevation_grid_deg=np.arange(-60.0, 60.1, 4.0),
    **kwargs,
) -> dict:
    del num_src  # top-K candidates expose ambiguity without pretending source count is known.
    power = srp_phat_map(audio, fs, mic_xyz, freq_range=freq_range, nfft=nfft,
                         azimuth_grid_deg=azimuth_grid_deg,
                         elevation_grid_deg=elevation_grid_deg, **kwargs)
    diag = map_diagnostics(power, np.asarray(azimuth_grid_deg), np.asarray(elevation_grid_deg))
    best = diag["top_candidates"][0]
    spatial_confidence = float(np.clip(0.4*diag["normalized_peak_sharpness"] +
                                       1.5*diag["top1_top2_difference"], 0, 1))
    return {
        "azimuth_deg": best["azimuth_deg"], "elevation_deg": best["elevation_deg"],
        "frequency_band_hz": [float(freq_range[0]), float(freq_range[1])],
        "diagnostics": diag, "confidence": {"score": spatial_confidence,
            "status": "LOW_CONFIDENCE" if diag["ambiguous"] else "VALID"},
        "normalized_heatmap": (power/max(power.max(), 1e-30)).tolist(),
        "azimuth_grid_deg": np.asarray(azimuth_grid_deg, float).tolist(),
        "elevation_grid_deg": np.asarray(elevation_grid_deg, float).tolist(),
        "implementation": "core_numpy_scipy_srp_phat", "convention": CONVENTION,
    }
