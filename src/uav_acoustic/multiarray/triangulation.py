from __future__ import annotations
import numpy as np


def direction_from_az_el(az_deg: float, el_deg: float) -> np.ndarray:
    az, el = np.deg2rad([az_deg, el_deg])
    d = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    return d / np.linalg.norm(d)


def triangulate_bearings(origins_xyz: np.ndarray, azimuth_deg: np.ndarray, elevation_deg: np.ndarray) -> dict:
    """Least-squares intersection of 3D bearing lines.

    Does not require sample-level synchronization; it assumes bearings refer to
    approximately the same target state/time.
    """
    origins = np.asarray(origins_xyz, float)
    dirs = np.stack([direction_from_az_el(a,e) for a,e in zip(azimuth_deg, elevation_deg)])
    A = np.zeros((3,3)); b = np.zeros(3)
    I = np.eye(3)
    for o, d in zip(origins, dirs):
        P = I - np.outer(d,d)
        A += P
        b += P @ o
    xyz = np.linalg.lstsq(A, b, rcond=None)[0]
    residuals = [np.linalg.norm(np.cross(xyz-o, d)) for o,d in zip(origins,dirs)]
    return {"xyz_m": xyz, "range_from_each_m": np.linalg.norm(xyz[None,:]-origins, axis=1), "line_residual_m": np.asarray(residuals)}
