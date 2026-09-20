from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml


class ConfigError(ValueError):
    pass


REQUIRED = {
    "hardware": (
        "fs", "channels", "udp_host", "udp_port", "udp_timeout_s",
        "socket_buffer_bytes", "receive_buffer_bytes", "wire_format",
        "reorder_vendor_channels", "geometry_csv",
    ),
    "processing": (
        "doa_band_hz", "feature_band_hz", "nfft", "sound_speed_mps",
        "healthy_channel_minimum", "dead_channel_rms_ratio", "clipping_threshold",
    ),
    "coordinates": (
        "pcb_x_to_engineering_x_sign", "pcb_y_to_engineering_y_sign",
        "pcb_y_physical_sign_evidence",
    ),
    "locks": (
        "packet_header_sequence_timestamp", "low_frequency_raw_below_1khz",
        "cross_device_sample_sync", "real_anc_reference_mic",
        "range_100_150m_in_80_90db", "array_diameter_requirement_30cm",
    ),
}


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"Configuration file not found: {config_path}")
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot read configuration {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("Configuration root must be a mapping")
    missing: list[str] = []
    for section, keys in REQUIRED.items():
        values = data.get(section)
        if not isinstance(values, dict):
            missing.append(section)
            continue
        missing.extend(f"{section}.{key}" for key in keys if key not in values)
    if missing:
        raise ConfigError("Missing configuration keys: " + ", ".join(missing))
    hw = data["hardware"]
    if int(hw["channels"]) != 128:
        raise ConfigError("WaveFrag release is locked to hardware.channels == 128")
    if int(hw["fs"]) <= 0:
        raise ConfigError("hardware.fs must be positive")
    if not 1 <= int(hw["udp_port"]) <= 65535:
        raise ConfigError("hardware.udp_port must be in 1..65535")
    if hw["wire_format"] != "payload_only_int16_le":
        raise ConfigError(
            "Unsupported wire_format. Packet header/sequence/timestamp are UNKNOWN; "
            "only the vendor-demo payload_only_int16_le assumption is implemented."
        )
    coordinates = data["coordinates"]
    for key in ("pcb_x_to_engineering_x_sign", "pcb_y_to_engineering_y_sign"):
        if int(coordinates[key]) not in {-1, 1}:
            raise ConfigError(f"coordinates.{key} must be +1 or -1")
    return data, config_path


def resolve_geometry_path(config: dict[str, Any], config_path: Path) -> Path:
    path = Path(str(config["hardware"]["geometry_csv"]))
    return (config_path.parent / path).resolve() if not path.is_absolute() else path.resolve()
