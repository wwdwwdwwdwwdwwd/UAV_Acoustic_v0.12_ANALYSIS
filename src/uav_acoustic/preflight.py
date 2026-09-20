from __future__ import annotations

from importlib import metadata
from pathlib import Path
import platform
import socket
import sys

import numpy as np

from .config import load_config, resolve_geometry_path
from .io.geometry import load_wavefrag_csv, validate_channel_order, vendor_channel_order


def _check(name: str, ok: bool, detail: str, *, required: bool = True) -> dict:
    return {"name": name, "status": "PASS" if ok else ("FAIL" if required else "WARN"), "detail": detail}


def run_preflight(config_path: str | Path, *, bind_udp: bool = True) -> dict:
    checks: list[dict] = []
    checks.append(_check("python_version", sys.version_info >= (3, 10), platform.python_version()))
    for package in ("numpy", "scipy", "yaml", "h5py"):
        distribution = "PyYAML" if package == "yaml" else package
        try:
            version = metadata.version(distribution)
            checks.append(_check(f"dependency_{package}", True, version))
        except metadata.PackageNotFoundError:
            checks.append(_check(f"dependency_{package}", False, "not installed"))
    try:
        config, resolved_config = load_config(config_path)
        checks.append(_check("config_complete", True, str(resolved_config)))
    except Exception as exc:
        checks.append(_check("config_complete", False, str(exc)))
        return {"status": "FAIL", "checks": checks}
    hw = config["hardware"]
    geometry_path = resolve_geometry_path(config, resolved_config)
    checks.append(_check("geometry_file", geometry_path.is_file(), str(geometry_path)))
    try:
        xyz = load_wavefrag_csv(geometry_path)
        checks.append(_check("geometry_shape", xyz.shape == (128, 3), str(xyz.shape)))
        aperture = float(np.max(np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)))
        checks.append(_check("geometry_aperture", True, f"{aperture:.6f} m measured from supplied coordinates"))
    except Exception as exc:
        checks.append(_check("geometry_shape", False, str(exc)))
    try:
        order = vendor_channel_order(int(hw["channels"]))
        validate_channel_order(order, int(hw["channels"]))
        configured = bool(hw["reorder_vendor_channels"])
        detail = "vendor k:8:(120+k) enabled" if configured else "DISABLED (invalid for verified R2.1 mapping)"
        checks.append(_check("channel_reorder", configured, f"128-index permutation valid; {detail}"))
    except Exception as exc:
        checks.append(_check("channel_reorder", False, str(exc)))
    checks.append(_check("sample_rate", int(hw["fs"]) == 80000, f"configured={hw['fs']} nominal_vendor=80000"))
    coord = config["coordinates"]
    checks.append(_check("pcb_x_physical_sign", int(coord["pcb_x_to_engineering_x_sign"]) == -1,
                         "Round-2 labelled broadband: engineering +azimuth = PCB -X"))
    y_verified = str(coord["pcb_y_physical_sign_evidence"]).upper().startswith("VERIFIED")
    checks.append(_check("pcb_y_physical_sign", y_verified,
                         "physical up/down sign still requires labelled elevation truth", required=False))
    if bind_udp:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind((str(hw["udp_host"]), int(hw["udp_port"])))
            checks.append(_check("udp_bind", True, f"{hw['udp_host']}:{hw['udp_port']}"))
        except OSError as exc:
            checks.append(_check("udp_bind", False, f"{hw['udp_host']}:{hw['udp_port']}: {exc}"))
        finally:
            sock.close()
    else:
        checks.append({"name": "udp_bind", "status": "SKIP", "detail": "--no-bind requested"})
    try:
        import pyroomacoustics  # noqa: F401
        checks.append(_check("optional_pyroomacoustics", True, "available", required=False))
    except Exception as exc:
        checks.append(_check("optional_pyroomacoustics", False, f"optional and unavailable: {exc}", required=False))
    status = "FAIL" if any(item["status"] == "FAIL" for item in checks) else "PASS"
    return {"status": status, "checks": checks, "locks": config["locks"]}
