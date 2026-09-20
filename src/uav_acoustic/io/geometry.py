from __future__ import annotations
from pathlib import Path
import csv
import numpy as np

from .channel_mapping import vendor_permutation


def load_wavefrag_csv(path: str | Path) -> np.ndarray:
    """Load WaveFrag 128-mic CSV and return (M, 3) coordinates in metres.

    Current vendor CSV contains X/Y only; Z is set to zero. The parser is
    intentionally tolerant of BOM, quotes, tabs and surrounding whitespace.
    """
    rows: list[list[float]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            vals: list[float] = []
            for cell in row:
                text = cell.strip().strip('"').strip()
                if not text:
                    continue
                vals.append(float(text))
            if len(vals) >= 2:
                rows.append([vals[0], vals[1], vals[2] if len(vals) >= 3 else 0.0])
    xyz = np.asarray(rows, dtype=float)
    if xyz.shape[0] != 128:
        raise ValueError(f"Expected 128 microphones, got {xyz.shape[0]}")
    if not np.all(np.isfinite(xyz)):
        raise ValueError("Microphone coordinates contain NaN or infinity")
    # The supplied CSV and MAT both store metres. A millimetre interpretation
    # would make the active aperture about 171 m, which is physically invalid.
    aperture = float(np.max(np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)))
    if not 0.05 <= aperture <= 1.0:
        raise ValueError(f"Implausible geometry aperture {aperture:g} m; expected coordinates in metres")
    return xyz


def vendor_channel_order(n_channels: int = 128) -> np.ndarray:
    """Return zero-based channel reorder from vendor MATLAB demos.

    MATLAB code groups k:8:(120+k), k=1..8. This creates 8 groups of 16.
    """
    return vendor_permutation(n_channels)


def validate_channel_order(order: np.ndarray, n_channels: int = 128) -> None:
    order = np.asarray(order)
    if order.shape != (n_channels,):
        raise ValueError(f"channel order must contain {n_channels} entries")
    if not np.issubdtype(order.dtype, np.integer):
        raise ValueError("channel order must contain integers")
    if not np.array_equal(np.sort(order), np.arange(n_channels)):
        raise ValueError("channel order must be a permutation of every channel exactly once")


def transform_pcb_to_engineering(xyz: np.ndarray, *, x_sign: int, y_sign: int) -> np.ndarray:
    """Apply the fixed PCB-axis to engineering azimuth/elevation transform."""
    if x_sign not in {-1, 1} or y_sign not in {-1, 1}:
        raise ValueError("PCB coordinate signs must each be +1 or -1")
    value = np.asarray(xyz, float).copy()
    value[:, 0] *= x_sign
    value[:, 1] *= y_sign
    return value
