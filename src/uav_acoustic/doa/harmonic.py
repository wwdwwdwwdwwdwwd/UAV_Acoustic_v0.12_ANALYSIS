from __future__ import annotations

import numpy as np

from ..coordinates import az_el_to_unit, xyz_to_az_el
from .das import narrowband_steered_power, map_diagnostics
from .confidence import confidence_and_status


def _spectrum(audio: np.ndarray, fs: int, nfft: int = 32768) -> tuple[np.ndarray, np.ndarray]:
    n = min(audio.shape[1], nfft)
    window = np.hanning(n)
    spec = np.mean(np.abs(np.fft.rfft(audio[:, :n]*window, n=nfft, axis=1)), axis=0)
    freq = np.fft.rfftfreq(nfft, 1/fs)
    return freq, spec


def _tone_snr_from_spectrum(freq: np.ndarray, spec: np.ndarray, target: float) -> tuple[float, float]:
    search = np.flatnonzero(np.abs(freq-target) <= max(8.0, target*0.015))
    k = int(search[np.argmax(spec[search])])
    noise = (np.abs(freq-freq[k]) >= 20) & (np.abs(freq-freq[k]) <= 250)
    floor = np.median(spec[noise]) if np.any(noise) else 1e-15
    return float(freq[k]), float(20*np.log10((spec[k]+1e-15)/(floor+1e-15)))


def harmonic_multiband_doa(
    audio: np.ndarray,
    fs: int,
    mic_xyz: np.ndarray,
    bpf_hz: float,
    *,
    max_frequency_hz: float = 5000.0,
    minimum_snr_db: float = 8.0,
    azimuth_grid_deg=np.arange(-90.0, 90.1, 4.0),
    elevation_grid_deg=np.arange(-60.0, 60.1, 4.0),
) -> dict:
    if bpf_hz <= 0:
        raise ValueError("BPF must be positive")
    harmonics = []
    spectrum_frequency, spectrum_magnitude = _spectrum(np.asarray(audio, float), fs)
    for order in range(1, int(max_frequency_hz//bpf_hz)+1):
        requested = order*bpf_hz
        measured, snr = _tone_snr_from_spectrum(spectrum_frequency, spectrum_magnitude, requested)
        if snr < minimum_snr_db:
            continue
        power = narrowband_steered_power(audio, fs, mic_xyz, measured,
                                         np.asarray(azimuth_grid_deg), np.asarray(elevation_grid_deg))
        diag = map_diagnostics(power, np.asarray(azimuth_grid_deg), np.asarray(elevation_grid_deg))
        conf = confidence_and_status(diag, snr)
        best = diag["top_candidates"][0]
        harmonics.append({"order": order, "requested_frequency_hz": requested,
                          "measured_frequency_hz": measured, "snr_db": snr,
                          "azimuth_deg": best["azimuth_deg"], "elevation_deg": best["elevation_deg"],
                          "confidence": conf, "diagnostics": diag})
    usable = [h for h in harmonics if h["confidence"]["score"] >= 0.25]
    if not usable:
        return {"estimated_bpf_hz": float(bpf_hz), "used_harmonic_orders": [],
                "harmonics": harmonics, "fused_azimuth_deg": None,
                "fused_elevation_deg": None, "fused_confidence": 0.0,
                "status": "NO_TARGET"}
    vectors = az_el_to_unit([h["azimuth_deg"] for h in usable], [h["elevation_deg"] for h in usable])
    weights = np.asarray([h["confidence"]["score"] * max(1.0, h["snr_db"]-minimum_snr_db) for h in usable])
    fused_vector = np.sum(vectors*weights[:, None], axis=0)
    fused_vector /= np.linalg.norm(fused_vector)
    az, el = xyz_to_az_el(fused_vector)
    angular = np.rad2deg(np.arccos(np.clip(vectors @ fused_vector, -1, 1)))
    consistency = float(np.clip(1-np.average(angular, weights=weights)/30.0, 0, 1))
    fused_conf = float(np.clip(np.average([h["confidence"]["score"] for h in usable], weights=weights)*consistency, 0, 1))
    return {"estimated_bpf_hz": float(bpf_hz),
            "used_harmonic_orders": [h["order"] for h in usable], "harmonics": harmonics,
            "fused_azimuth_deg": float(az), "fused_elevation_deg": float(el),
            "spatial_consistency": consistency, "fused_confidence": fused_conf,
            "status": "VALID" if fused_conf >= 0.58 and len(usable) >= 2 else "LOW_CONFIDENCE"}
