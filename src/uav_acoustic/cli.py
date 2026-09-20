from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from .analysis import analyze_frame
from .config import load_config, resolve_geometry_path
from .io.capture_file import load_capture, save_capture
from .io.geometry import load_wavefrag_csv, transform_pcb_to_engineering
from .io.wavefrag_udp import WaveFragUDPConfig, WaveFragUDPError, WaveFragUDPSource
from .preflight import run_preflight
from .round2 import run_round2
from .r22_validation import run_r22_validation
from .az_measure import run_console
from .steered_realtime import run_realtime
from .guarded_runtime import run_guarded


def _write_json(path: str | Path, data: dict) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _default_config() -> Path:
    return Path(__file__).resolve().parents[2] / "03_configs" / "default.yaml"


def preflight_command(args: argparse.Namespace) -> int:
    result = run_preflight(args.config, bind_udp=not args.no_bind)
    _write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 2


def capture_command(args: argparse.Namespace) -> int:
    config, config_path = load_config(args.config)
    hw = config["hardware"]
    mic_xyz = load_wavefrag_csv(resolve_geometry_path(config, config_path))
    coord = config["coordinates"]
    mic_xyz = transform_pcb_to_engineering(mic_xyz,
        x_sign=int(coord["pcb_x_to_engineering_x_sign"]),
        y_sign=int(coord["pcb_y_to_engineering_y_sign"]))
    duration_s = float(args.duration)
    n_samples = int(round(duration_s * int(hw["fs"])))
    udp_config = WaveFragUDPConfig(
        host=str(hw["udp_host"]),
        port=int(hw["udp_port"]),
        channels=int(hw["channels"]),
        fs=int(hw["fs"]),
        reorder_vendor_channels=bool(hw["reorder_vendor_channels"]),
        recv_bytes=int(hw["receive_buffer_bytes"]),
        timeout_s=float(hw["udp_timeout_s"]),
        socket_buffer_bytes=int(hw["socket_buffer_bytes"]),
        expected_source_ip=args.source_ip or hw.get("expected_source_ip"),
        expected_source_port=int(hw["expected_source_port"]) if hw.get("expected_source_port") else None,
        expected_datagram_bytes=int(hw["expected_datagram_bytes"]) if hw.get("expected_datagram_bytes") else None,
    )
    try:
        with WaveFragUDPSource(mic_xyz, udp_config) as source:
            frame = source.read_frame(n_samples)
            frame.metadata.update({
                "capture_tool": "uav_acoustic capture",
                "capture_time_utc": datetime.now(timezone.utc).isoformat(),
                "bind_ip": udp_config.host,
                "bind_port": udp_config.port,
                "expected_source_ip": udp_config.expected_source_ip,
                "coordinate_transform": dict(coord),
            })
            output = save_capture(
                args.output,
                frame,
                raw_int16=source.last_raw_int16,
                config_snapshot=config,
            )
    except WaveFragUDPError as exc:
        print(f"CAPTURE ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Saved {frame.audio.shape[0]} x {frame.audio.shape[1]} capture to {output}")
    return 0


def analyze_command(args: argparse.Namespace) -> int:
    config, _ = load_config(args.config)
    processing = config["processing"]
    frame, _, _ = load_capture(args.input)
    result = analyze_frame(
        frame,
        known_tone_hz=args.known_tone,
        doa_band_hz=tuple(float(v) for v in processing["doa_band_hz"]),
        broadband_band_hz=tuple(float(v) for v in processing.get("broadband_band_hz", [1000, 5000])),
        clipping_threshold=float(processing["clipping_threshold"]),
        dead_channel_rms_ratio=float(processing["dead_channel_rms_ratio"]),
        healthy_channel_minimum=int(processing["healthy_channel_minimum"]),
        confidence_thresholds=dict(processing.get("confidence", {})),
    )
    result["input_capture"] = Path(args.input).name
    output = _write_json(args.output, result)
    print(f"Saved analysis to {output}")
    return 0 if result["channel_metrics"]["healthy_minimum_pass"] else 3


def round2_command(args: argparse.Namespace) -> int:
    if args.evidence_rich:
        session = run_r22_validation(args.session_root, config_path=args.config, dry_run=args.dry_run)
    else:
        session = run_round2(args.session_root, config_path=args.config, dry_run=args.dry_run,
                             calibration=args.calibration, calibration_repeats=args.calibration_repeats)
    print(f"Round-2 session: {session.resolve()}")
    return 0


def az_measure_command(args: argparse.Namespace) -> int:
    return run_console(args.config, args.results_root, dry_run=args.dry_run, once=args.once)


def realtime_command(args: argparse.Namespace) -> int:
    duration=args.duration
    if duration is None and args.dry_run:duration=max(5.0,args.frames*.1)
    return run_guarded(args.config,args.results_root,mode=args.mode,duration_s=duration,synthetic=args.synthetic or args.dry_run,replay_path=args.input)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uav_acoustic", description="WaveFrag 128-channel on-machine tools")
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight", help="validate configuration and machine readiness")
    preflight.add_argument("--config", type=Path, default=_default_config())
    preflight.add_argument("--output", type=Path, default=Path("preflight_report.json"))
    preflight.add_argument("--no-bind", action="store_true", help="skip the UDP bind test")
    preflight.set_defaults(func=preflight_command)

    capture = sub.add_parser("capture", help="capture WaveFrag UDP to NPZ/HDF5")
    capture.add_argument("--config", type=Path, default=_default_config())
    capture.add_argument("--duration", type=float, required=True, help="capture duration in seconds")
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--source-ip", help="optional expected device source IP filter")
    capture.set_defaults(func=capture_command)

    analyze = sub.add_parser("analyze", help="analyze a saved capture to JSON")
    analyze.add_argument("input", type=Path)
    analyze.add_argument("--config", type=Path, default=_default_config())
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--known-tone", type=float, help="expected tone frequency in Hz")
    analyze.set_defaults(func=analyze_command)
    round2 = sub.add_parser("round2", help="guided NO_DRONE_MODE Round-2 test matrix")
    round2.add_argument("--session-root", type=Path, default=Path("results"))
    round2.add_argument("--config", type=Path, default=_default_config())
    round2.add_argument("--dry-run", action="store_true")
    round2.add_argument("--calibration", action="store_true", help="run the minimal R2.1 calibration matrix")
    round2.add_argument("--evidence-rich", action="store_true", help="run the R2.2 evidence-rich validation matrix")
    round2.add_argument("--calibration-repeats", type=int, choices=(2, 3), default=2)
    round2.set_defaults(func=round2_command)
    az = sub.add_parser("az-measure", help="blind horizontal azimuth measurement")
    az.add_argument("--config", type=Path, default=_default_config())
    az.add_argument("--results-root", type=Path, default=Path("results"))
    az.add_argument("--dry-run", action="store_true", help="use synthetic data; never opens UDP")
    az.add_argument("--once", action="store_true", help="perform one measurement and exit")
    az.set_defaults(func=az_measure_command)
    realtime = sub.add_parser("realtime", help="v0.12 asynchronous TRACK/SAMID interactive runtime")
    realtime.add_argument("--config", type=Path, default=_default_config())
    realtime.add_argument("--results-root", type=Path, default=Path("results"))
    realtime.add_argument("--dry-run", action="store_true")
    realtime.add_argument("--frames", type=int, default=20)
    realtime.add_argument("--mode", choices=("interactive","receive-only","replay"), default="interactive")
    realtime.add_argument("--duration", type=float, help="optional bounded run duration; omitted runs until Q/Esc")
    realtime.add_argument("--synthetic", action="store_true", help="use the isolated synthetic acquisition producer")
    realtime.add_argument("--input", type=Path, help="external WAV used only with --mode replay")
    realtime.set_defaults(func=realtime_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
