"""Independent GCC-PHAT and planar-wave physics diagnostics."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, correlate, sosfiltfilt

from ..coordinates import xyz_to_az_el


def gcc_phat_delay(
    x: np.ndarray, y: np.ndarray, fs: int, *, max_tau_s: float, interp: int = 8,
    freq_range: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Return delay of ``x`` relative to ``y`` and normalized peak strength."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = 1 << int(np.ceil(np.log2(x.size + y.size)))
    X = np.fft.rfft(x, n=n)
    Y = np.fft.rfft(y, n=n)
    cross = X * np.conj(Y)
    magnitude = np.abs(cross)
    # A time-domain bandpass leaves tiny numerical energy in the stop band.
    # Blind PHAT normalization would promote those near-zero bins to unit
    # magnitude and make their arbitrary phase dominate the true pass band,
    # especially when the inverse FFT is oversampled.  Retain only bins with
    # meaningful cross power before phase normalization.
    floor = max(float(np.max(magnitude)) * 1e-8, 1e-15)
    supported = magnitude >= floor
    if freq_range is not None:
        frequency = np.fft.rfftfreq(n, 1.0 / fs)
        supported &= (frequency >= float(freq_range[0])) & (frequency <= float(freq_range[1]))
    cross = np.where(supported, cross / np.maximum(magnitude, floor), 0.0)
    cc = np.fft.irfft(cross, n=interp * n)
    limit = min(int(round(interp * fs * max_tau_s)), cc.size // 2)
    window = np.concatenate((cc[-limit:], cc[: limit + 1]))
    k = int(np.argmax(np.abs(window)))
    lag = k - limit
    strength = float(abs(window[k]) / max(np.sqrt(np.mean(window * window)), 1e-15))
    return lag / float(interp * fs), strength


def _bandpass(audio: np.ndarray, fs: int, freq_range: tuple[float, float] | None) -> np.ndarray:
    if freq_range is None:
        return np.asarray(audio, float)
    lo = max(20.0, float(freq_range[0]))
    hi = min(float(freq_range[1]), fs * 0.49)
    if not lo < hi:
        raise ValueError("invalid wavefront-fit frequency range")
    return sosfiltfilt(butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos"), audio, axis=1)


def wavefront_fit(
    audio: np.ndarray,
    fs: int,
    mic_xyz: np.ndarray,
    *,
    freq_range: tuple[float, float] | None = None,
    c: float = 343.0,
    reference_channel: int | None = None,
) -> dict:
    """Robust plane-wave fit using independently measured channel delays.

    Since PosMic is planar, only X/Y slowness is observed; +Z is selected as
    the declared front hemisphere and the mirror ambiguity is reported.
    """
    a = _bandpass(np.asarray(audio, float), fs, freq_range)
    xyz = np.asarray(mic_xyz, float)
    if a.ndim != 2 or a.shape[0] != xyz.shape[0]:
        raise ValueError("audio/microphone layout mismatch")
    if a.shape[1] > 65536:
        start = (a.shape[1] - 65536) // 2
        a = a[:, start : start + 65536]
    center = np.argmin(np.linalg.norm(xyz - xyz.mean(axis=0), axis=1))
    ref = int(center if reference_channel is None else reference_channel)
    aperture = np.linalg.norm(xyz - xyz[ref], axis=1)
    tau = np.zeros(len(xyz), float)
    quality = np.zeros(len(xyz), float)
    for m in range(len(xyz)):
        tau[m], quality[m] = gcc_phat_delay(
            a[m], a[ref], fs, max_tau_s=float(aperture[m] / c + 1.5 / fs),
            freq_range=freq_range,
        )
    design = np.column_stack([np.ones(len(xyz)), xyz[:, 0], xyz[:, 1]])
    inlier = np.ones(len(xyz), dtype=bool)
    for _ in range(4):
        coef = np.linalg.lstsq(design[inlier], tau[inlier], rcond=None)[0]
        residual = tau - design @ coef
        scale = 1.4826 * np.median(np.abs(residual[inlier] - np.median(residual[inlier])))
        threshold = max(2.0 / fs, 3.5 * scale)
        new_inlier = np.abs(residual) <= threshold
        if new_inlier.sum() < 16 or np.array_equal(new_inlier, inlier):
            break
        inlier = new_inlier
    coef = np.linalg.lstsq(design[inlier], tau[inlier], rcond=None)[0]
    residual = tau - design @ coef
    # Arrival time is t = constant - dot(position, source_unit)/c.
    ux, uy = -c * coef[1], -c * coef[2]
    transverse_sq = ux * ux + uy * uy
    physical = transverse_sq <= 1.05
    scale = max(1.0, np.sqrt(transverse_sq))
    ux, uy = ux / scale, uy / scale
    uz = np.sqrt(max(0.0, 1.0 - ux * ux - uy * uy))
    az, el = xyz_to_az_el(np.array([ux, uy, uz]))
    abs_us = np.abs(residual[inlier]) * 1e6
    return {
        "reference_channel": ref,
        "relative_tdoa_s": tau.tolist(),
        "gcc_peak_strength": quality.tolist(),
        "azimuth_deg": float(az),
        "elevation_deg": float(el),
        "rms_residual_us": float(np.sqrt(np.mean(abs_us**2))),
        "median_residual_us": float(np.median(abs_us)),
        "p90_residual_us": float(np.percentile(abs_us, 90)),
        "max_residual_us": float(np.max(abs_us)),
        "inlier_mic_count": int(inlier.sum()),
        "physically_bounded_direction": bool(physical),
        "front_back_ambiguity": True,
    }


def local_pair_wavefront_fit(
    audio: np.ndarray,
    fs: int,
    mic_xyz: np.ndarray,
    *,
    freq_range: tuple[float, float] | None = (500.0, 5000.0),
    neighbours_per_mic: int = 3,
    c: float = 343.0,
) -> dict:
    """Fit a plane wave from local physical-neighbour TDOAs.

    Local pairs make mapping errors directly observable: a wrong channel map
    assigns unrelated signals to short physical baselines. Normalized broadband
    correlation is deliberately independent of SRP/DAS steering code.
    """
    a = _bandpass(np.asarray(audio, float), fs, freq_range)
    xyz = np.asarray(mic_xyz, float)
    if a.ndim != 2 or a.shape[0] != len(xyz):
        raise ValueError("audio/microphone layout mismatch")
    if a.shape[1] > 65536:
        start = (a.shape[1] - 65536) // 2
        a = a[:, start : start + 65536]
    a -= np.mean(a, axis=1, keepdims=True)
    distance = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    np.fill_diagonal(distance, np.inf)
    pairs: set[tuple[int, int]] = set()
    for i in range(len(xyz)):
        for j in np.argsort(distance[i])[:neighbours_per_mic]:
            pairs.add(tuple(sorted((i, int(j)))))
    delays = []
    strengths = []
    deltas = []
    pair_rows = []
    for i, j in sorted(pairs):
        limit = int(np.ceil(distance[i, j] / c * fs)) + 2
        corr = correlate(a[i], a[j], mode="full", method="fft")
        center = a.shape[1] - 1
        window = corr[center - limit : center + limit + 1]
        k = int(np.argmax(np.abs(window)))
        lag_samples = k - limit
        delay = lag_samples / fs
        strength = float(abs(window[k]) / max(np.sqrt(np.dot(a[i], a[i]) * np.dot(a[j], a[j])), 1e-15))
        delays.append(delay)
        strengths.append(strength)
        deltas.append(xyz[i, :2] - xyz[j, :2])
        pair_rows.append((i, j, distance[i, j], delay, strength))
    design = np.asarray(deltas)
    observed = np.asarray(delays)
    inlier = np.ones(len(observed), dtype=bool)
    for _ in range(6):
        coef = np.linalg.lstsq(design[inlier], observed[inlier], rcond=None)[0]
        residual = observed - design @ coef
        center = np.median(residual[inlier])
        scale = 1.4826 * np.median(np.abs(residual[inlier] - center))
        new_inlier = np.abs(residual - center) <= max(2.0 / fs, 3.5 * scale)
        if new_inlier.sum() < 32 or np.array_equal(new_inlier, inlier):
            break
        inlier = new_inlier
    coef = np.linalg.lstsq(design[inlier], observed[inlier], rcond=None)[0]
    residual = observed - design @ coef
    ux, uy = -c * coef
    transverse_sq = ux * ux + uy * uy
    physical = transverse_sq <= 1.05
    scale = max(1.0, np.sqrt(transverse_sq))
    ux, uy = ux / scale, uy / scale
    uz = np.sqrt(max(0.0, 1.0 - ux * ux - uy * uy))
    az, el = xyz_to_az_el(np.array([ux, uy, uz]))
    abs_us = np.abs(residual[inlier]) * 1e6
    return {
        "pair_count": len(pair_rows), "inlier_pair_count": int(inlier.sum()),
        "median_pair_correlation": float(np.median(np.asarray(strengths)[inlier])),
        "azimuth_deg": float(az), "elevation_deg": float(el),
        "rms_residual_us": float(np.sqrt(np.mean(abs_us**2))),
        "median_residual_us": float(np.median(abs_us)),
        "p90_residual_us": float(np.percentile(abs_us, 90)),
        "max_residual_us": float(np.max(abs_us)),
        "physically_bounded_direction": bool(physical), "front_back_ambiguity": True,
        "pairs": [{"mic_i": i, "mic_j": j, "baseline_m": float(d),
                   "tdoa_us": float(t * 1e6), "correlation": float(q)}
                  for i, j, d, t, q in pair_rows],
    }


def robust_pair_wavefront_fit(
    audio: np.ndarray,
    fs: int,
    mic_xyz: np.ndarray,
    *,
    freq_range: tuple[float, float] | None = (1000.0, 2000.0),
    neighbours_per_mic: int = 5,
    c: float = 343.0,
    interpolation: int = 16,
) -> dict:
    """Sub-sample, robust plane-wave inversion from local microphone pairs.

    ``local_pair_wavefront_fit`` is intentionally a literal sample-lag audit.
    At 80 kHz its 12.5 us quantisation is too coarse for short baselines, so it
    must not be used as the final wavefront scale estimator.  This companion
    uses oversampled GCC-PHAT delays, correlation-quality weights and Huber
    reweighting while retaining only physically local pairs.
    """
    a = _bandpass(np.asarray(audio, float), fs, freq_range)
    xyz = np.asarray(mic_xyz, float)
    if a.ndim != 2 or a.shape[0] != len(xyz):
        raise ValueError("audio/microphone layout mismatch")
    if a.shape[1] > 65536:
        start = (a.shape[1] - 65536) // 2
        a = a[:, start : start + 65536]
    a -= np.mean(a, axis=1, keepdims=True)

    distance = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    np.fill_diagonal(distance, np.inf)
    pairs: set[tuple[int, int]] = set()
    for i in range(len(xyz)):
        for j in np.argsort(distance[i])[:neighbours_per_mic]:
            pairs.add(tuple(sorted((i, int(j)))))

    observed, strengths, deltas, pair_rows = [], [], [], []
    for i, j in sorted(pairs):
        delay, strength = gcc_phat_delay(
            a[i], a[j], fs,
            max_tau_s=float(distance[i, j] / c + 1.5 / fs),
            interp=int(interpolation),
            freq_range=freq_range,
        )
        observed.append(delay)
        strengths.append(strength)
        deltas.append(xyz[i, :2] - xyz[j, :2])
        pair_rows.append((i, j, distance[i, j], delay, strength))

    design = np.asarray(deltas, float)
    delays = np.asarray(observed, float)
    quality = np.asarray(strengths, float)
    # GCC peak-to-RMS is positive but not bounded.  Its square root prevents a
    # handful of very sharp channel pairs from dominating the physical fit.
    base_weight = np.sqrt(np.maximum(quality, 1e-6))
    weight = base_weight.copy()
    coef = np.zeros(2, float)
    for _ in range(12):
        lhs = design * np.sqrt(weight[:, None])
        rhs = delays * np.sqrt(weight)
        new_coef = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        residual = delays - design @ new_coef
        center = float(np.median(residual))
        mad = 1.4826 * float(np.median(np.abs(residual - center)))
        huber = max(0.35 / fs, 1.5 * mad)
        robust = np.ones_like(residual)
        outside = np.abs(residual - center) > huber
        robust[outside] = huber / np.abs(residual[outside] - center)
        new_weight = base_weight * robust
        if np.allclose(new_coef, coef, atol=1e-10, rtol=1e-5):
            coef, weight = new_coef, new_weight
            break
        coef, weight = new_coef, new_weight

    residual = delays - design @ coef
    inlier = weight >= 0.5 * base_weight
    if inlier.sum() < 32:
        inlier = weight >= np.percentile(weight, 25)
    ux, uy = -c * coef
    transverse_norm = float(np.hypot(ux, uy))
    physical = transverse_norm <= 1.05
    normalization = max(1.0, transverse_norm)
    ux, uy = ux / normalization, uy / normalization
    uz = np.sqrt(max(0.0, 1.0 - ux * ux - uy * uy))
    az, el = xyz_to_az_el(np.array([ux, uy, uz]))
    abs_us = np.abs(residual[inlier]) * 1e6
    return {
        "pair_count": len(pair_rows),
        "inlier_pair_count": int(inlier.sum()),
        "median_gcc_peak_strength": float(np.median(quality[inlier])),
        "azimuth_deg": float(az),
        "elevation_deg": float(el),
        "direction_xy": [float(ux), float(uy)],
        "transverse_norm_before_clamp": transverse_norm,
        "rms_residual_us": float(np.sqrt(np.mean(abs_us**2))),
        "median_residual_us": float(np.median(abs_us)),
        "p90_residual_us": float(np.percentile(abs_us, 90)),
        "max_residual_us": float(np.max(abs_us)),
        "physically_bounded_direction": bool(physical),
        "front_back_ambiguity": True,
        "delay_resolution_us": float(1e6 / (fs * interpolation)),
        "implementation": "local_pair_gcc_phat_huber_subsample_v1",
        "pairs": [
            {"mic_i": i, "mic_j": j, "baseline_m": float(d),
             "tdoa_us": float(t * 1e6), "gcc_peak_strength": float(q),
             "fit_residual_us": float(r * 1e6), "inlier": bool(ok)}
            for (i, j, d, t, q), r, ok in zip(pair_rows, residual, inlier)
        ],
    }
