from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from ..types import AcousticFrame


FORMAT_VERSION = 2


def _metadata_json(frame: AcousticFrame, config_snapshot: dict | None) -> str:
    data = dict(frame.metadata)
    data.update({
        "format_version": FORMAT_VERSION,
        "capture_time_unix": frame.timestamp,
        "audio_layout": "channels_samples",
        "config_snapshot": config_snapshot or {},
    })
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def save_capture(
    path: str | Path,
    frame: AcousticFrame,
    raw_int16: np.ndarray | None = None,
    config_snapshot: dict | None = None,
) -> Path:
    frame.validate()
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        np.asarray(raw_int16, dtype=np.int16)
        if raw_int16 is not None
        else np.clip(np.rint(frame.audio * 32768.0), -32768, 32767).astype(np.int16)
    )
    if raw.shape != frame.audio.shape:
        raise ValueError(f"raw_int16 shape {raw.shape} != audio shape {frame.audio.shape}")
    metadata = _metadata_json(frame, config_snapshot)
    suffix = output.suffix.lower()
    if suffix == ".npz":
        np.savez_compressed(
            output,
            raw_int16=raw,
            audio=frame.audio.astype(np.float32),
            fs=np.int64(frame.fs),
            mic_xyz=frame.mic_xyz.astype(np.float64),
            capture_time_unix=np.float64(frame.timestamp if frame.timestamp is not None else np.nan),
            metadata_json=np.asarray(metadata),
        )
    elif suffix in {".h5", ".hdf5"}:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("HDF5 output requires h5py from core requirements") from exc
        with h5py.File(output, "w") as h5:
            h5.create_dataset("raw_int16", data=raw, compression="gzip", shuffle=True)
            h5.create_dataset("audio", data=frame.audio.astype(np.float32), compression="gzip", shuffle=True)
            h5.create_dataset("mic_xyz", data=frame.mic_xyz.astype(np.float64))
            h5.attrs["fs"] = frame.fs
            h5.attrs["capture_time_unix"] = frame.timestamp if frame.timestamp is not None else np.nan
            h5.attrs["metadata_json"] = metadata
            h5.attrs["format_version"] = FORMAT_VERSION
    else:
        raise ValueError("Capture output must end in .npz, .h5, or .hdf5")
    return output


def load_capture(path: str | Path) -> tuple[AcousticFrame, np.ndarray, dict]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Capture not found: {source}")
    suffix = source.suffix.lower()
    if suffix == ".npz":
        with np.load(source, allow_pickle=False) as data:
            required = {"raw_int16", "audio", "fs", "mic_xyz", "capture_time_unix", "metadata_json"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"Capture missing fields: {sorted(missing)}")
            raw = data["raw_int16"].astype(np.int16, copy=True)
            audio = data["audio"].astype(np.float32, copy=True)
            fs = int(data["fs"])
            mic_xyz = data["mic_xyz"].astype(float, copy=True)
            timestamp_value = float(data["capture_time_unix"])
            metadata = json.loads(str(data["metadata_json"]))
    elif suffix in {".h5", ".hdf5"}:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("HDF5 input requires h5py from core requirements") from exc
        with h5py.File(source, "r") as h5:
            raw = h5["raw_int16"][:].astype(np.int16)
            audio = h5["audio"][:].astype(np.float32)
            mic_xyz = h5["mic_xyz"][:].astype(float)
            fs = int(h5.attrs["fs"])
            timestamp_value = float(h5.attrs["capture_time_unix"])
            metadata = json.loads(str(h5.attrs["metadata_json"]))
    else:
        raise ValueError("Capture input must end in .npz, .h5, or .hdf5")
    timestamp = None if np.isnan(timestamp_value) else timestamp_value
    frame = AcousticFrame(audio=audio, fs=fs, mic_xyz=mic_xyz, timestamp=timestamp, metadata=metadata)
    frame.validate()
    if raw.shape != audio.shape or raw.dtype != np.int16:
        raise ValueError("Capture raw/audio channel layout mismatch")
    return frame, raw, metadata
