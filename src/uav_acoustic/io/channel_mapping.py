"""WaveFrag wire-channel to physical-microphone data contract.

All DOA algorithms consume ``AcousticFrame`` objects in PosMic row order.
Only acquisition/import code may call :func:`to_vendor_physical_order`.
"""
from __future__ import annotations

import hashlib
import numpy as np


RAW_CHANNEL_ORDER = "WaveFragRawOrder"
PHYSICAL_CHANNEL_ORDER = "VendorPosMicRowOrder"
MAPPING_VERSION = "wavefrag-row-indices-v1"


def vendor_permutation(n_channels: int = 128) -> np.ndarray:
    """Zero-based equivalent of MATLAB ``[1:8:121, ..., 8:8:128]``."""
    if n_channels != 128:
        raise ValueError("WaveFrag vendor mapping is defined for 128 channels")
    return np.concatenate([np.arange(k, n_channels, 8) for k in range(8)]).astype(int)


def vendor_inverse_permutation(n_channels: int = 128) -> np.ndarray:
    return np.argsort(vendor_permutation(n_channels))


def mapping_hash(n_channels: int = 128) -> str:
    order = vendor_permutation(n_channels).astype("<i2", copy=False)
    return hashlib.sha256(order.tobytes()).hexdigest()


def physical_order_metadata(*, applied: bool) -> dict:
    return {
        "raw_channel_order": RAW_CHANNEL_ORDER,
        "raw_int16_channel_order": RAW_CHANNEL_ORDER,
        "physical_channel_order": PHYSICAL_CHANNEL_ORDER if applied else RAW_CHANNEL_ORDER,
        "audio_channel_order": PHYSICAL_CHANNEL_ORDER if applied else RAW_CHANNEL_ORDER,
        "mapping_version": MAPPING_VERSION,
        "mapping_applied": bool(applied),
        "mapping_hash": mapping_hash(),
        # Compatibility field. Its value describes data content, not a config toggle.
        "channel_reordered": bool(applied),
    }


def is_vendor_physical_order(metadata: dict) -> bool:
    return bool(
        metadata.get("mapping_applied") is True
        and metadata.get("physical_channel_order") == PHYSICAL_CHANNEL_ORDER
        and metadata.get("mapping_version") == MAPPING_VERSION
        and metadata.get("mapping_hash") == mapping_hash()
    )


def to_vendor_physical_order(
    audio: np.ndarray, metadata: dict | None = None, *, channel_axis: int = 0
) -> tuple[np.ndarray, dict]:
    """Apply H1 exactly once and return data plus content-truth metadata.

    A capture carrying a complete matching contract is returned unchanged. A
    partial/contradictory claim is rejected instead of risking a second reorder.
    """
    value = np.asarray(audio)
    meta = dict(metadata or {})
    if value.shape[channel_axis] != 128:
        raise ValueError("channel mapping requires exactly 128 channels")
    claims_applied = meta.get("mapping_applied") is True or meta.get("channel_reordered") is True
    if claims_applied:
        if not is_vendor_physical_order(meta):
            raise ValueError("capture claims reordered channels but lacks a matching mapping contract")
        return value, meta
    mapped = np.take(value, vendor_permutation(), axis=channel_axis)
    meta.update(physical_order_metadata(applied=True))
    return mapped, meta
