from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class AcousticFrame:
    """Unified data object used by simulation, file playback, and WaveFrag UDP.

    audio: shape (channels, samples), float32/float64 preferred.
    mic_xyz: shape (channels, 3), metres.
    """
    audio: np.ndarray
    fs: int
    mic_xyz: np.ndarray
    timestamp: float | None = None
    metadata: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.audio.ndim != 2:
            raise ValueError("audio must have shape (channels, samples)")
        if self.mic_xyz.ndim != 2 or self.mic_xyz.shape[1] != 3:
            raise ValueError("mic_xyz must have shape (channels, 3)")
        if self.audio.shape[0] != self.mic_xyz.shape[0]:
            raise ValueError("audio channel count must match mic_xyz rows")
        if self.fs <= 0:
            raise ValueError("fs must be positive")
        if not np.all(np.isfinite(self.audio)):
            raise ValueError("audio contains NaN or infinity")
        if not np.all(np.isfinite(self.mic_xyz)):
            raise ValueError("mic_xyz contains NaN or infinity")
