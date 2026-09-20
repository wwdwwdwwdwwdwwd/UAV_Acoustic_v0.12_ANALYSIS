from __future__ import annotations
from pathlib import Path
import numpy as np
import soundfile as sf
from ..types import AcousticFrame


def load_multichannel_wav(path: str | Path, mic_xyz: np.ndarray, expected_fs: int | None = None) -> AcousticFrame:
    data, fs = sf.read(path, always_2d=True, dtype="float32")  # samples, channels
    if expected_fs is not None and fs != expected_fs:
        raise ValueError(f"Expected fs={expected_fs}, got {fs}")
    frame = AcousticFrame(audio=data.T, fs=int(fs), mic_xyz=np.asarray(mic_xyz, float))
    frame.validate()
    return frame


def save_multichannel_wav(path: str | Path, frame: AcousticFrame) -> None:
    frame.validate()
    sf.write(path, frame.audio.T, frame.fs, subtype="PCM_16")
