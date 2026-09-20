"""Single coordinate convention for the WaveFrag planar array.

Vendor evidence: PosMic lies in PCB XY and DAS_20cm.m scans [X,Y,Z=+2 m].
Engineering coordinates are produced only after the configured PCB axis-sign
transform. In engineering coordinates +Z is front, +X is positive azimuth and
+Y is positive elevation. A planar array cannot resolve the Z mirror alone.
"""
from __future__ import annotations

import numpy as np

CONVENTION = (
    "right-handed metres: array plane XY; +Z front; azimuth 0 at +Z and "
    "positive toward +X; elevation positive toward +Y"
)


def az_el_to_unit(azimuth_deg, elevation_deg) -> np.ndarray:
    az = np.deg2rad(np.asarray(azimuth_deg, dtype=float))
    el = np.deg2rad(np.asarray(elevation_deg, dtype=float))
    az, el = np.broadcast_arrays(az, el)
    return np.stack(
        [np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)], axis=-1
    )


def xyz_to_az_el(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(xyz, dtype=float)
    norm = np.linalg.norm(value, axis=-1)
    if np.any(norm <= 0):
        raise ValueError("direction vector must be non-zero")
    unit = value / norm[..., None]
    az = np.rad2deg(np.arctan2(unit[..., 0], unit[..., 2]))
    el = np.rad2deg(np.arcsin(np.clip(unit[..., 1], -1.0, 1.0)))
    return az, el
