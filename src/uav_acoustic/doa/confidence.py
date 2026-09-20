from __future__ import annotations

import numpy as np


def confidence_and_status(
    diagnostics: dict,
    frequency_snr_db: float,
    frame_angles: list[tuple[float, float]] | None = None,
    *,
    no_target_snr_db: float = 6.0,
    valid_snr_db: float = 12.0,
    valid_confidence: float = 0.58,
    max_stability_std_deg: float = 8.0,
) -> dict:
    """Turn explainable metrics into VALID/LOW_CONFIDENCE/NO_TARGET."""
    psr = np.clip((float(diagnostics["peak_to_sidelobe_db"])-1.0)/8.0, 0, 1)
    sharp = np.clip(float(diagnostics["normalized_peak_sharpness"]), 0, 1)
    separation = np.clip(float(diagnostics["top1_top2_difference"])/0.35, 0, 1)
    snr_score = np.clip((frequency_snr_db-no_target_snr_db)/18.0, 0, 1)
    stability_std = None
    stability = 0.5
    if frame_angles and len(frame_angles) >= 2:
        angles = np.asarray(frame_angles, float)
        stability_std = float(np.sqrt(np.var(angles[:, 0]) + np.var(angles[:, 1])))
        stability = float(np.clip(1-stability_std/max_stability_std_deg, 0, 1))
    confidence = float(0.22*psr + 0.20*sharp + 0.20*separation + 0.25*snr_score + 0.13*stability)
    if frequency_snr_db < no_target_snr_db:
        status = "NO_TARGET"
    elif frequency_snr_db >= valid_snr_db and confidence >= valid_confidence and not diagnostics["ambiguous"]:
        status = "VALID"
    else:
        status = "LOW_CONFIDENCE"
    return {
        "score": confidence, "status": status, "frequency_snr_db": float(frequency_snr_db),
        "cross_frame_stability_std_deg": stability_std,
        "components": {"psr": float(psr), "sharpness": float(sharp),
                       "top_peak_separation": float(separation), "snr": float(snr_score),
                       "cross_frame_stability": float(stability)},
    }
