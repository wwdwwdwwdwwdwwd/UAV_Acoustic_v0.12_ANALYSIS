"""Blind, horizontal-only WaveFrag azimuth measurement built on R2.2 algorithms."""
from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

from .config import load_config, resolve_geometry_path
from .coordinates import az_el_to_unit
from .doa.das import farfield_broadband_das
from .doa.srp_phat import estimate_srp_phat
from .doa.wavefront import robust_pair_wavefront_fit
from .io.capture_file import save_capture
from .io.channel_mapping import physical_order_metadata, vendor_inverse_permutation
from .io.geometry import load_wavefrag_csv, transform_pcb_to_engineering
from .io.wavefrag_udp import (CaptureTimeoutError, PortBindError, WaveFragUDPConfig,
                              WaveFragUDPError, WaveFragUDPSource)
from .preflight import run_preflight
from .simulation.propagation import simulate_free_field
from .types import AcousticFrame


SOFTWARE_VERSION = "AZ-Only v1.1 (R2.2 algorithms; distance-independent production)"


def _grids(config: dict, centre: float | None = None) -> np.ndarray:
    search = config["processing"]["azimuth_search"]
    if centre is None:
        lo, hi = map(float, search["range_deg"])
        step = float(search["coarse_step_deg"])
    else:
        radius = float(search["fine_radius_deg"])
        lo, hi = centre - radius, centre + radius
        step = float(search["fine_step_deg"])
        hard_lo, hard_hi = map(float, search["range_deg"])
        lo, hi = max(lo, hard_lo), min(hi, hard_hi)
    return np.round(np.arange(lo, hi + step * 0.25, step), 10)


def _srp(audio: np.ndarray, fs: int, xyz: np.ndarray, band: tuple[float, float], config: dict) -> dict:
    common = dict(freq_range=band, nfft=int(config["processing"]["nfft"]),
                  elevation_grid_deg=np.asarray([0.0]), max_pairs=512,
                  c=float(config["processing"]["sound_speed_mps"]))
    coarse = estimate_srp_phat(audio, fs, xyz, azimuth_grid_deg=_grids(config), **common)
    fine = estimate_srp_phat(audio, fs, xyz,
                             azimuth_grid_deg=_grids(config, coarse["azimuth_deg"]), **common)
    fine["search"] = {"coarse_peak_deg": coarse["azimuth_deg"],
                      "elevation_fixed_deg": 0.0, "method": "coarse_plus_0.1deg_fine_grid"}
    return fine


def _tdoa(audio: np.ndarray, fs: int, xyz: np.ndarray, band: tuple[float, float], config: dict) -> dict:
    # R2.2 GCC-PHAT oversamples every pair by 16x. A bounded central segment
    # preserves its sub-sample/Huber physics while keeping field latency sane.
    if audio.shape[1] > 8192:
        start = (audio.shape[1] - 8192) // 2
        audio = audio[:, start:start+8192]
    result = robust_pair_wavefront_fit(
        audio, fs, xyz, freq_range=band,
        neighbours_per_mic=3, c=float(config["processing"]["sound_speed_mps"]), interpolation=16)
    ux = float(result["direction_xy"][0])
    az = float(np.rad2deg(np.arcsin(np.clip(ux, -1.0, 1.0))))
    return {**result, "unconstrained_azimuth_deg": result["azimuth_deg"],
            "azimuth_deg": az, "elevation_deg": 0.0,
            "constraint": "horizontal projection; elevation fixed at 0 deg"}


def _farfield_das(audio: np.ndarray, fs: int, xyz: np.ndarray, band: tuple[float, float], config: dict) -> dict:
    common = dict(elevation_deg=0.0, freq_range=band,
                  c=float(config["processing"]["sound_speed_mps"]), nfft=4096)
    coarse = farfield_broadband_das(audio, fs, xyz, azimuth_grid_deg=_grids(config), **common)
    fine = farfield_broadband_das(audio, fs, xyz,
                                  azimuth_grid_deg=_grids(config, coarse["azimuth_deg"]), **common)
    fine["search"] = {"coarse_peak_deg": coarse["azimuth_deg"],
                      "elevation_fixed_deg": 0.0, "method": "farfield_DAS_coarse_plus_fine"}
    return fine


def _band_snr_db(audio: np.ndarray, fs: int, band: tuple[float, float]) -> tuple[float, float]:
    n = min(audio.shape[1], 32768)
    start = (audio.shape[1] - n) // 2
    segment = audio[:, start:start+n] * np.hanning(n)[None, :]
    spectrum = np.mean(np.abs(np.fft.rfft(segment, axis=1))**2, axis=0)
    freq = np.fft.rfftfreq(n, 1/fs)
    inside = (freq >= band[0]) & (freq <= band[1])
    reference = ((freq >= 500) & (freq < 900)) | ((freq > 5200) & (freq <= 7000))
    band_power = float(np.mean(spectrum[inside])) if np.any(inside) else 0.0
    noise_power = float(np.median(spectrum[reference])) if np.any(reference) else 0.0
    return float(10*np.log10((band_power+1e-30)/(noise_power+1e-30))), band_power


def _health(audio: np.ndarray, config: dict) -> dict:
    proc = config["processing"]
    rms = np.sqrt(np.mean(np.asarray(audio, float)**2, axis=1))
    threshold = max(1e-8, float(np.median(rms))*float(proc["dead_channel_rms_ratio"]))
    healthy = int(np.sum(rms >= threshold))
    clipping = float(np.max(np.mean(np.abs(audio) >= float(proc["clipping_threshold"]), axis=1)))
    return {"healthy_channels": healthy, "total_channels": int(audio.shape[0]),
            "dead_or_low_channels": np.flatnonzero(rms < threshold).astype(int).tolist(),
            "maximum_clipping_fraction": clipping}


def analyze_azimuth(frame: AcousticFrame, config: dict) -> dict:
    """Analyze one blind capture. This API intentionally has no ground-truth parameter."""
    frame.validate()
    primary = tuple(map(float, config["processing"]["primary_band_hz"]))
    srp = _srp(frame.audio, frame.fs, frame.mic_xyz, primary, config)
    tdoa = _tdoa(frame.audio, frame.fs, frame.mic_xyz, primary, config)
    farfield = _farfield_das(frame.audio, frame.fs, frame.mic_xyz, primary, config)

    measurement = config["measurement"]
    win = int(round(float(measurement["window_duration_s"])*frame.fs))
    hop = int(round(win*(1.0-float(measurement["window_overlap_fraction"]))))
    starts = list(range(0, max(1, frame.audio.shape[1]-win+1), max(1, hop)))
    if not starts:
        starts = [0]
    temporal = []
    for start in starts:
        chunk = frame.audio[:, start:min(start+win, frame.audio.shape[1])]
        if chunk.shape[1] < max(4096, win//2):
            continue
        try:
            s = _srp(chunk, frame.fs, frame.mic_xyz, primary, config)
            value = float(s["azimuth_deg"])
            temporal.append({"start_sample": start, "azimuth_deg": value,
                             "srp_azimuth_deg": s["azimuth_deg"]})
        except (ValueError, np.linalg.LinAlgError):
            continue
    temporal_values = np.asarray([x["azimuth_deg"] for x in temporal], float)
    if temporal_values.size == 0:
        temporal_values = np.asarray([float(np.median(algorithm_values))])
    temporal_median = float(np.median(temporal_values))
    # Exactly one representative vote per independent algorithm family.
    srp_representative = temporal_median
    algorithm_values = np.asarray([srp_representative, tdoa["azimuth_deg"],
                                   farfield["azimuth_deg"]], float)
    final_az = float(np.median(algorithm_values))

    bands = {}
    for lo, hi in config["processing"]["report_bands_hz"]:
        key = f"{int(lo)}-{int(hi)}"
        item = srp if (float(lo), float(hi)) == primary else _srp(
            frame.audio, frame.fs, frame.mic_xyz, (float(lo), float(hi)), config)
        bands[key] = {"azimuth_deg": float(item["azimuth_deg"]),
                      "peak_to_sidelobe_db": float(item["diagnostics"]["peak_to_sidelobe_db"]),
                      "primary": (float(lo), float(hi)) == primary}

    snr_db, band_power = _band_snr_db(frame.audio, frame.fs, primary)
    health = _health(frame.audio, config)
    spread = float(np.ptp(algorithm_values))
    temporal_std = float(np.std(temporal_values))
    required_windows = max(1, int(np.ceil(len(starts)*0.6)))
    quality_cfg = config["processing"]["quality"]
    if (health["healthy_channels"] < int(config["processing"]["healthy_channel_minimum"])
            or spread > float(quality_cfg["algorithm_spread_low_confidence_deg"])
            or temporal_std > float(quality_cfg["temporal_std_low_confidence_deg"])):
        quality = "LOW_CONFIDENCE"
    elif (len(temporal) < required_windows or snr_db < float(quality_cfg["band_snr_proxy_warn_db"])
          or spread > float(quality_cfg["algorithm_spread_warn_deg"])
          or temporal_std > float(quality_cfg["temporal_std_warn_deg"])):
        quality = "WARN"
    else:
        quality = "PASS"
    packet = {"datagram_count": int(frame.metadata.get("datagram_count", 0)),
              "datagram_sizes": frame.metadata.get("datagram_sizes", []),
              "expected_datagram_bytes": int(config["hardware"]["expected_datagram_bytes"]),
              "packet_size_status": "PASS" if frame.metadata.get("datagram_sizes") == [int(config["hardware"]["expected_datagram_bytes"])] else ("SIMULATED" if frame.metadata.get("simulated") else "WARN")}
    return {
        "schema_version": 1, "software_version": SOFTWARE_VERSION,
        "timestamp": datetime.now().astimezone().isoformat(),
        "sample_rate": int(frame.fs), "sample_count": int(frame.audio.shape[1]),
        "capture_duration_s": float(frame.audio.shape[1]/frame.fs),
        "coordinate_convention": "0 deg is array front; viewed from behind: left negative, right positive; engineering +az = PCB -X",
        "mode": "azimuth_only", "elevation_fixed_deg": 0.0,
        "primary_band_hz": list(primary), "all_band_results": bands,
        "algorithms": {"srp_phat": {**srp, "global_azimuth_deg": srp["azimuth_deg"],
                                      "representative_azimuth_deg": srp_representative,
                                      "representative_source": "temporal_median"},
                       "robust_tdoa": tdoa, "farfield_das": farfield},
        "algorithm_estimates_deg": {"srp_phat": float(algorithm_values[0]),
                                      "robust_tdoa": float(algorithm_values[1]),
                                      "farfield_das": float(algorithm_values[2])},
        "algorithm_spread_deg": spread, "final_azimuth_deg": final_az,
        "fusion": {"method": "median_of_three_independent_algorithm_families",
                   "votes": ["SRP temporal median", "Robust TDOA", "Far-field DAS"],
                   "srp_double_weighting": False},
        "temporal": {"median_deg": temporal_median, "mean_deg": float(np.mean(temporal_values)),
                     "std_deg": temporal_std, "min_deg": float(np.min(temporal_values)),
                     "max_deg": float(np.max(temporal_values)),
                     "p10_deg": float(np.percentile(temporal_values, 10)),
                     "p90_deg": float(np.percentile(temporal_values, 90)),
                     "valid_windows": len(temporal), "total_windows": len(starts), "windows": temporal},
        "band_snr_proxy_db": snr_db, "band_energy": band_power,
        "peak_to_sidelobe_db": float(srp["diagnostics"]["peak_to_sidelobe_db"]),
        "channel_health": health, "packet_statistics": packet, "quality": quality,
        "confidence_note": "Quality is a diagnostic flag, not a target-presence probability or an absolute-accuracy claim.",
    }


def _context(config_path: Path) -> tuple[dict, Path, np.ndarray]:
    config, resolved = load_config(config_path)
    xyz = load_wavefrag_csv(resolve_geometry_path(config, resolved))
    coord = config["coordinates"]
    xyz = transform_pcb_to_engineering(xyz, x_sign=int(coord["pcb_x_to_engineering_x_sign"]),
                                       y_sign=int(coord["pcb_y_to_engineering_y_sign"]))
    return config, resolved, xyz


def capture_frame(config: dict, xyz: np.ndarray) -> tuple[AcousticFrame, np.ndarray]:
    """Capture from WaveFrag; there is deliberately no truth/expected-angle input."""
    hw = config["hardware"]
    samples = int(round(float(config["measurement"]["capture_duration_s"])*int(hw["fs"])))
    udp = WaveFragUDPConfig(host=str(hw["udp_host"]), port=int(hw["udp_port"]),
        channels=int(hw["channels"]), fs=int(hw["fs"]),
        reorder_vendor_channels=bool(hw["reorder_vendor_channels"]),
        recv_bytes=int(hw["receive_buffer_bytes"]), timeout_s=float(hw["udp_timeout_s"]),
        socket_buffer_bytes=int(hw["socket_buffer_bytes"]), expected_source_ip=hw.get("expected_source_ip"),
        expected_source_port=int(hw["expected_source_port"]) if hw.get("expected_source_port") else None,
        expected_datagram_bytes=int(hw["expected_datagram_bytes"]))
    with WaveFragUDPSource(xyz, udp) as source:
        frame = source.read_frame(samples)
        return frame, np.asarray(source.last_raw_int16, dtype=np.int16)


def simulated_frame(config: dict, xyz: np.ndarray, azimuth_deg: float = 1.0) -> tuple[AcousticFrame, np.ndarray]:
    """Synthetic signal used only by explicit dry-run/self-test paths."""
    fs = int(config["hardware"]["fs"])
    n = int(round(float(config["measurement"]["capture_duration_s"])*fs))
    rng = np.random.default_rng(2201)
    signal = rng.normal(size=n)
    f = np.fft.rfftfreq(n, 1/fs); X = np.fft.rfft(signal)
    X[(f < 1000) | (f > 5000)] = 0
    signal = np.fft.irfft(X, n).astype(np.float32)
    frame = simulate_free_field(signal, fs, xyz, 3.0*az_el_to_unit(azimuth_deg, 0.0), snr_db=25, seed=2202)
    frame.metadata.update(physical_order_metadata(applied=True))
    frame.metadata.update({"simulated": True, "dry_run_only": True})
    physical = np.clip(np.rint(frame.audio*32768), -32768, 32767).astype(np.int16)
    raw_wire = physical[vendor_inverse_permutation()]
    return frame, raw_wire


def _compact(result: dict, directory: Path) -> str:
    a = result["algorithm_estimates_deg"]; t = result["temporal"]
    measurement_id = result.get("measurement_id", "MEASUREMENT")
    lines = ["="*58, str(measurement_id).upper(), "="*58,
             "PRIMARY BAND          1200-2000 Hz", "",
             f"FINAL AZ              {result['final_azimuth_deg']:+.2f} deg", "",
             f"SRP-PHAT              {a['srp_phat']:+.2f} deg",
             f"Robust TDOA           {a['robust_tdoa']:+.2f} deg",
             f"Far-field DAS         {a['farfield_das']:+.2f} deg", "",
             f"Temporal median       {t['median_deg']:+.2f} deg",
             f"Temporal mean         {t['mean_deg']:+.2f} deg",
             f"Temporal STD          {t['std_deg']:.2f} deg",
             f"Temporal P10-P90      {t['p10_deg']:+.2f} ~ {t['p90_deg']:+.2f} deg",
             f"Algorithm spread      {result['algorithm_spread_deg']:.2f} deg",
             f"Band SNR proxy         {result['band_snr_proxy_db']:.1f} dB",
             f"Peak ratio             {result['peak_to_sidelobe_db']:.1f} dB",
             f"Valid windows          {t['valid_windows']} / {t['total_windows']}",
             f"Healthy channels       {result['channel_health']['healthy_channels']} / {result['channel_health']['total_channels']}",
             "", "Other bands:"]
    for key, item in result["all_band_results"].items():
        lines.append(f"{key + ' Hz':<23}{item['azimuth_deg']:+.1f} deg" + ("  PRIMARY" if item["primary"] else ""))
    lines += ["", f"QUALITY                {result['quality']}", "", "Result saved to:", str(directory), "="*58]
    return "\n".join(lines)


SESSION_FIELDS = ["measurement_id", "timestamp", "final_az_deg", "srp_az_deg",
                  "tdoa_az_deg", "farfield_das_az_deg", "algorithm_spread_deg",
                  "temporal_median_deg", "temporal_mean_deg", "temporal_std_deg",
                  "temporal_p10_deg", "temporal_p90_deg", "band_snr_proxy_db",
                  "peak_ratio_db", "valid_windows", "healthy_channels", "quality"]


def create_session(root: Path, config: dict, preflight: dict, *, dry_run: bool) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = Path(root).resolve()/f"session_{stamp}"
    suffix = 1
    while session.exists():
        session = Path(root).resolve()/f"session_{stamp}_{suffix:02d}"; suffix += 1
    session.mkdir(parents=True)
    info = {"schema_version": 1, "software_version": SOFTWARE_VERSION,
            "created": datetime.now().astimezone().isoformat(), "mode": "distance_independent_az",
            "dry_run": bool(dry_run), "config": config, "preflight": preflight,
            "production_inputs": ["128-channel waveform", "microphone geometry", "sample rate",
                                  "speed of sound", "fixed processing configuration"]}
    (session/"session_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    with (session/"measurements.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        csv.DictWriter(stream, fieldnames=SESSION_FIELDS).writeheader()
    return session


def save_result(session: Path, measurement_number: int, frame: AcousticFrame, raw_wire: np.ndarray,
                result: dict, config: dict) -> Path:
    measurement_id = f"measurement_{int(measurement_number):03d}"
    directory = Path(session).resolve()/measurement_id
    directory.mkdir(parents=True, exist_ok=False)
    save_capture(directory/"raw_capture.npz", frame, raw_int16=raw_wire, config_snapshot=config)
    result = {**result, "measurement_id": measurement_id, "config": config}
    (directory/"result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    estimates = result["algorithm_estimates_deg"]
    row = {"measurement_id": measurement_id, "timestamp": result["timestamp"],
           "final_az_deg": result["final_azimuth_deg"], "srp_az_deg": estimates["srp_phat"],
           "tdoa_az_deg": estimates["robust_tdoa"], "farfield_das_az_deg": estimates["farfield_das"],
           "algorithm_spread_deg": result["algorithm_spread_deg"], "temporal_median_deg": result["temporal"]["median_deg"],
           "temporal_mean_deg": result["temporal"]["mean_deg"], "temporal_std_deg": result["temporal"]["std_deg"],
           "temporal_p10_deg": result["temporal"]["p10_deg"], "temporal_p90_deg": result["temporal"]["p90_deg"],
           "band_snr_proxy_db": result["band_snr_proxy_db"], "healthy_channels": result["channel_health"]["healthy_channels"],
           "peak_ratio_db": result["peak_to_sidelobe_db"], "valid_windows": result["temporal"]["valid_windows"],
           "quality": result["quality"]}
    with (directory/"result.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=SESSION_FIELDS); writer.writeheader(); writer.writerow(row)
    with (Path(session)/"measurements.csv").open("a", newline="", encoding="utf-8-sig") as stream:
        csv.DictWriter(stream, fieldnames=SESSION_FIELDS).writerow(row)
    (directory/"summary.txt").write_text(_compact(result, directory)+"\n", encoding="utf-8")
    return directory


def run_measurement(config_path: Path, results_root: Path, *, dry_run: bool = False) -> tuple[dict, Path]:
    config, resolved, xyz = _context(config_path)
    preflight = run_preflight(resolved, bind_udp=not dry_run)
    if preflight["status"] != "PASS":
        failed = [f"{x['name']}: {x['detail']}" for x in preflight["checks"] if x["status"] == "FAIL"]
        raise RuntimeError("Hardware/config check failed: " + "; ".join(failed))
    frame, raw = simulated_frame(config, xyz) if dry_run else capture_frame(config, xyz)
    result = analyze_azimuth(frame, config)
    result["preflight"] = preflight
    session = create_session(results_root, config, preflight, dry_run=dry_run)
    directory = save_result(session, 1, frame, raw, result, config)
    return result, directory


def run_console(config_path: Path, results_root: Path, *, dry_run: bool = False,
                once: bool = False, input_fn=input) -> int:
    config, resolved, xyz = _context(config_path)
    preflight = run_preflight(resolved, bind_udp=not dry_run)
    if preflight["status"] != "PASS":
        failed = [item for item in preflight["checks"] if item["status"] == "FAIL"]
        if any(item["name"] == "udp_bind" for item in failed):
            print(f"ERROR: UDP port {config['hardware']['udp_port']} is already occupied.")
            print("Close AcousticCamera/WaveFragStudio and check Windows firewall.")
        else:
            print("ERROR: Hardware/config check failed.")
            for item in failed: print(f"- {item['name']}: {item['detail']}")
        return 2
    session = create_session(results_root, config, preflight, dry_run=dry_run)
    print("="*58); print("WaveFrag Horizontal Azimuth Measurement"); print("="*58)
    print(f"128 channels          OK\nSample rate           {config['hardware']['fs']} Hz")
    print("Primary band          1200-2000 Hz\nMode                  Distance-independent AZ")
    print("Beamforming           SRP-PHAT + Far-field DAS\n")
    print("Please confirm:\n- Array is fixed\n- Source is approximately level with array center")
    print("- External phone/speaker is looping broadband_1k_5k_60s.wav")
    measurement_number = 0
    while True:
        if not once:
            command = input_fn("\nPress Enter to measure, or Q to quit: ").strip().upper()
            if command == "Q":
                print(f"\nSession saved to:\n{session}\n\nSummary:\n{session/'measurements.csv'}")
                return 0
        started = time.monotonic()
        try:
            frame, raw = simulated_frame(config, xyz) if dry_run else capture_frame(config, xyz)
            result = analyze_azimuth(frame, config)
            result["preflight"] = preflight
            measurement_number += 1
            directory = save_result(session, measurement_number, frame, raw, result, config)
            result["measurement_id"] = f"measurement_{measurement_number:03d}"
            print("\n"+_compact(result, directory))
            print(f"Processing time: {time.monotonic()-started:.1f} s")
        except (CaptureTimeoutError, PortBindError, WaveFragUDPError) as exc:
            if isinstance(exc, CaptureTimeoutError):
                print(f"\nERROR: No WaveFrag UDP data received on port {config['hardware']['udp_port']}.")
            elif isinstance(exc, PortBindError):
                print(f"\nERROR: UDP port {config['hardware']['udp_port']} is already occupied.")
            else:
                print(f"\nERROR: WaveFrag capture failed: {exc}")
            print("Check:\n- WaveFrag network configuration\n- Target PC IP\n- Ethernet connection")
            print("- AcousticCamera/WaveFragStudio is closed\n- Windows firewall")
            if not once: input_fn("Press Enter to exit.")
            return 2
        except Exception as exc:
            print(f"\nERROR: Measurement processing failed: {exc}")
            return 2
        if once: return 0
