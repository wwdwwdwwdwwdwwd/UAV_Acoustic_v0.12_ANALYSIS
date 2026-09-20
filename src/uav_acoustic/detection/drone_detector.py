from __future__ import annotations
import numpy as np
from ..features.harmonics import harmonic_comb_score
from ..features.periodicity import periodicity_score


def heuristic_drone_score(audio: np.ndarray, fs: int) -> dict:
    """Interpretable baseline detector, not a trained classifier.

    The output is useful for simulation/integration. Thresholds must be fitted on
    real truck + UAV data before any performance claim.
    """
    h = harmonic_comb_score(audio, fs)
    p = periodicity_score(np.mean(audio, axis=0), fs)
    h_norm = np.clip((h["score"] - 1.0) / 15.0, 0.0, 1.0)
    p_norm = np.clip((p["score"] - 0.05) / 0.7, 0.0, 1.0)
    score = float(0.7*h_norm + 0.3*p_norm)
    return {"score": score, "bpf_hz": h["bpf_hz"], "harmonic_score": h["score"], "periodicity": p}
