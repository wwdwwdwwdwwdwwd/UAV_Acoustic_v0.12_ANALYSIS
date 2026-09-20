from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .coordinates import CONVENTION
from .doa.das import (narrowband_steered_power, map_diagnostics,
                      vendor_nearfield_das)
from .doa.srp_phat import estimate_srp_phat
from .doa.confidence import confidence_and_status
from .doa.harmonic import harmonic_multiband_doa
from .features.harmonics import harmonic_comb_score, mean_spectrum
from .features.periodicity import periodicity_score
from .types import AcousticFrame


def _json_floats(values: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(values).ravel()]


def _local_peak_snr(freq, magnitude, requested_hz: float) -> tuple[int, float]:
    radius = max(8.0, requested_hz*0.015)
    local = np.flatnonzero(np.abs(freq-requested_hz) <= radius)
    if not local.size:
        raise ValueError(f"requested tone {requested_hz:g} Hz is outside spectrum")
    k = int(local[np.argmax(magnitude[local])])
    noise = (np.abs(freq-freq[k]) >= max(15.0, radius)) & (np.abs(freq-freq[k]) <= max(250.0, 3*radius))
    floor = float(np.median(magnitude[noise])) if np.any(noise) else 1e-15
    return k, float(20*np.log10((magnitude[k]+1e-15)/(floor+1e-15)))


def _narrowband_result(audio, fs, xyz, frequency, snr, thresholds) -> dict:
    az_grid = np.arange(-90.0, 90.1, 2.0)
    el_grid = np.arange(-60.0, 60.1, 2.0)
    power = narrowband_steered_power(audio, fs, xyz, frequency, az_grid, el_grid)
    diagnostics = map_diagnostics(power, az_grid, el_grid)
    best = diagnostics["top_candidates"][0]
    frame_angles = []
    if audio.shape[1] >= 4096*3:
        for chunk in np.array_split(audio, 3, axis=1):
            small = narrowband_steered_power(chunk, fs, xyz, frequency,
                                             np.arange(-90, 91, 6), np.arange(-60, 61, 6))
            d = map_diagnostics(small, np.arange(-90, 91, 6), np.arange(-60, 61, 6), top_k=2)
            frame_angles.append((d["top_candidates"][0]["azimuth_deg"], d["top_candidates"][0]["elevation_deg"]))
    confidence = confidence_and_status(diagnostics, snr, frame_angles,
                                       no_target_snr_db=thresholds["no_target_snr_db"],
                                       valid_snr_db=thresholds["valid_snr_db"],
                                       valid_confidence=thresholds["valid_confidence"],
                                       max_stability_std_deg=thresholds["max_stability_std_deg"])
    return {"azimuth_deg": best["azimuth_deg"], "elevation_deg": best["elevation_deg"],
            "frequency_hz": float(frequency), "confidence": confidence,
            "diagnostics": diagnostics, "convention": CONVENTION,
            "planar_array_back_front_ambiguity": True}


def analyze_frame(
    frame: AcousticFrame, *, known_tone_hz: float | None = None,
    doa_band_hz=(1000.0, 5000.0), broadband_band_hz=(1000.0, 5000.0),
    clipping_threshold=0.999, dead_channel_rms_ratio=0.02,
    healthy_channel_minimum=120, confidence_thresholds: dict | None = None,
    run_vendor=True, run_broadband=True, run_harmonic=True,
    vendor_focal_distance_m: float = 2.0,
) -> dict:
    frame.validate()
    audio = np.asarray(frame.audio, float)
    channels, samples = audio.shape
    rms = np.sqrt(np.mean(audio*audio, axis=1)); peak = np.max(np.abs(audio), axis=1)
    dc = np.mean(audio, axis=1); clipping_fraction = np.mean(np.abs(audio) >= clipping_threshold, axis=1)
    reference_rms = float(np.median(rms)); dead_threshold = max(1e-8, reference_rms*dead_channel_rms_ratio)
    dead = np.flatnonzero(rms < dead_threshold); healthy = int(channels-dead.size)
    nfft = min(32768, max(2048, 1 << int(np.ceil(np.log2(max(2, min(samples, 32768)))))))
    freq, magnitude = mean_spectrum(audio, frame.fs, nfft=nfft)
    db = 20*np.log10(magnitude+1e-15)
    mask = (freq >= 20) & (freq <= min(frame.fs/2, 10000)); indices = np.flatnonzero(mask)
    peaks, _ = find_peaks(db[mask], distance=max(1, int(20/(freq[1]-freq[0]))))
    peak_indices = indices[peaks]; peak_indices = peak_indices[np.argsort(db[peak_indices])[-10:]][::-1]
    spectral_peaks = [{"frequency_hz": float(freq[k]), "magnitude_db": float(db[k])} for k in peak_indices]
    thresholds = {"no_target_snr_db": 6.0, "valid_snr_db": 12.0,
                  "valid_confidence": 0.58, "max_stability_std_deg": 8.0}
    thresholds.update(confidence_thresholds or {})

    if known_tone_hz is not None:
        doa_index, doa_snr = _local_peak_snr(freq, magnitude, known_tone_hz)
        requested = float(known_tone_hz)
    else:
        band = (freq >= doa_band_hz[0]) & (freq <= min(doa_band_hz[1], frame.fs/2))
        if not np.any(band): raise ValueError(f"No FFT bins in DOA band {doa_band_hz}")
        doa_index = int(np.flatnonzero(band)[np.argmax(magnitude[band])])
        _, doa_snr = _local_peak_snr(freq, magnitude, float(freq[doa_index])); requested = None
    doa_frequency = float(freq[doa_index])
    narrow = _narrowband_result(audio, frame.fs, frame.mic_xyz, doa_frequency, doa_snr, thresholds)
    tone_result = None if requested is None else {
        "requested_hz": requested, "estimated_hz": doa_frequency, "snr_db": doa_snr,
        "doa_frequency_hz": doa_frequency,
        "proof_requested_tone_drives_doa": abs(doa_frequency-requested) <= max(8.0, requested*0.015),
    }
    harmonic_features = harmonic_comb_score(audio, frame.fs)
    periodicity = periodicity_score(np.mean(audio, axis=0), frame.fs)
    vendor = None
    if run_vendor:
        band = (max(20.0, doa_frequency-20), doa_frequency+20) if requested is not None else broadband_band_hz
        vendor = vendor_nearfield_das(audio, frame.fs, frame.mic_xyz, freq_range=band,
                                      focal_z_m=float(vendor_focal_distance_m))
    srp_band = ((max(20.0, doa_frequency-50.0), doa_frequency+50.0)
                if requested is not None else broadband_band_hz)
    broadband = estimate_srp_phat(audio, frame.fs, frame.mic_xyz, freq_range=srp_band,
                                  nfft=2048 if requested is not None else 1024) if run_broadband else None
    if vendor is not None:
        vendor["confidence"] = confidence_and_status(vendor["diagnostics"], doa_snr, None, **thresholds)
    if broadband is not None:
        broadband["confidence"] = confidence_and_status(broadband["diagnostics"], doa_snr, None, **thresholds)
    harmonic = None
    if run_harmonic and known_tone_hz is None and float(harmonic_features["bpf_hz"]) > 0:
        harmonic = harmonic_multiband_doa(audio, frame.fs, frame.mic_xyz,
                                          float(harmonic_features["bpf_hz"]),
                                          max_frequency_hz=float(broadband_band_hz[1]))
    overall = narrow["confidence"]["status"]
    if known_tone_hz is None and doa_snr < thresholds["no_target_snr_db"]:
        overall = "NO_TARGET"
    low_note = None
    if known_tone_hz is not None and known_tone_hz < 1000 and narrow["diagnostics"]["ambiguous"]:
        low_note = "low frequency is detectable but DOA ambiguous"
    return {
        "schema_version": 2, "coordinate_convention": CONVENTION,
        "audio_shape": [int(channels), int(samples)], "fs_hz": int(frame.fs), "duration_s": samples/frame.fs,
        "channel_metrics": {"rms": _json_floats(rms), "peak_abs": _json_floats(peak), "dc": _json_floats(dc),
            "clipping_fraction": _json_floats(clipping_fraction), "dead_or_very_low_zero_based": [int(v) for v in dead],
            "healthy_count": healthy, "healthy_minimum_configured": int(healthy_channel_minimum),
            "healthy_minimum_pass": healthy >= healthy_channel_minimum, "dead_threshold_rms": dead_threshold},
        "spectrum": {"frequency_hz": _json_floats(freq), "mean_magnitude": _json_floats(magnitude), "top_peaks": spectral_peaks},
        "known_tone": tone_result,
        "uav_features": {"bpf_hz": float(harmonic_features["bpf_hz"]),
            "harmonic_comb_score": float(harmonic_features["score"]), "periodicity_score": float(periodicity["score"]),
            "periodicity_hz": periodicity["period_hz"]},
        "narrowband_das": narrow, "das": narrow, "vendor_compatible_das": vendor,
        "broadband_srp_phat": broadband, "harmonic_multiband_doa": harmonic,
        "status": overall, "interpretation": low_note, "confidence_thresholds": thresholds,
        "metadata": frame.metadata,
    }
