from __future__ import annotations

import numpy as np

from ..doa.srp_phat import estimate_srp_phat
from .tracker import AngleEMA


def track_broadband_doa(audio: np.ndarray, fs: int, mic_xyz: np.ndarray, *,
                        freq_range=(1000.0, 5000.0), window_s=0.6, hop_s=0.3) -> dict:
    """Continuous coarse SRP-PHAT trajectory with transparent raw/smoothed angles."""
    nwin = int(round(window_s*fs)); hop = int(round(hop_s*fs))
    if audio.shape[1] < nwin: raise ValueError("audio is shorter than tracking window")
    smoother = AngleEMA(alpha=0.35); rows=[]
    for start in range(0, audio.shape[1]-nwin+1, hop):
        result = estimate_srp_phat(audio[:, start:start+nwin], fs, mic_xyz,
            freq_range=freq_range, nfft=512, azimuth_grid_deg=np.arange(-90, 91, 6),
            elevation_grid_deg=np.arange(-60, 61, 6), max_pairs=256)
        az=float(result["azimuth_deg"]); el=float(result["elevation_deg"])
        smooth_az, smooth_el=smoother.update(az, el)
        d=result["diagnostics"]
        confidence=float(np.clip(0.5*d["normalized_peak_sharpness"]+1.5*d["top1_top2_difference"], 0, 1))
        rows.append({"time_s":(start+nwin/2)/fs, "raw_azimuth_deg":az,
                     "raw_elevation_deg":el, "azimuth_deg":smooth_az,
                     "elevation_deg":smooth_el, "confidence":confidence,
                     "ambiguous":bool(d["ambiguous"])})
    jumps=np.abs(np.diff([r["azimuth_deg"] for r in rows]))
    return {"frames":rows, "max_azimuth_jump_deg":float(jumps.max()) if jumps.size else 0.0,
            "negative_to_positive_crossing":bool(rows and rows[0]["azimuth_deg"] < 0 < rows[-1]["azimuth_deg"]),
            "method":"sliding-window core SRP-PHAT + EMA"}
