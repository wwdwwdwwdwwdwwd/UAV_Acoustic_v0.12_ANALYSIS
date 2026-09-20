"""v0.12: v0.10 acquisition/search with asynchronous TRACK semantics."""
from __future__ import annotations

from collections import deque
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import yaml

from .async_artifacts import AsyncArtifactWriter
from .async_logger import AsyncCSVLogger
from .async_semantic import SimpleTargetLoss, build_semantic_task, semantic_state
from .controller_state import ControllerState
from .das_tracking import FarFieldDAS, SpatialTracker, TrackerConfig
from .io.channel_mapping import to_vendor_physical_order, mapping_hash, MAPPING_VERSION
from .io.geometry import load_wavefrag_csv, transform_pcb_to_engineering
from .noise_gate import BaselineBuilder, NoveltyGate, band_features, representative_channels
from .process_acquisition import AcquisitionConfig, AcquisitionProcess
from .samid_process import SamidProcess, MODEL_SHA
from .semantic_acquire import SemanticAcquirer, THRESHOLD
from .terminal_renderer import TerminalRenderer


VERSION = "UAV Acoustic v0.12 ASYNC TRACK SEMANTIC"
PRODUCTION_V010_SHA = "179DA45D059E741263E1EACDCC6A5D822F15FA699D0324A8A11D309E3B04FED7"

ACQ_FIELDS = "timestamp elapsed_s received_datagrams_total received_datagrams_per_s expected_datagrams_per_s observed_expected_ratio_1s observed_expected_ratio_5s received_sample_frames_total expected_sample_frames_total audio_clock_ratio received_bytes_total malformed_datagrams unexpected_source_datagrams receive_gap_mean_ms receive_gap_p95_ms receive_gap_max_ms shared_ring_fill_s shared_ring_overwrite_count snapshot_stale_count acquisition_process_alive acquisition_process_cpu_pct controller_cpu_pct samid_worker_cpu_pct ram_bytes window_valid health_state warning_reason".split()
CAL_FIELDS = "timestamp calibration_id frame_index acquisition_valid band_powers_db background_building_status remaining_s invalid_reason".split()
NOV_FIELDS = "timestamp state acquisition_valid band_delta_db band_novelty_z aggregate_novelty_score triggered_band_count persistence_count triggered trigger_reason search_running cooldown_remaining_s".split()
SEARCH_FIELDS = "timestamp search_id candidate_rank window_id az_deg el_deg spatial_score coherence_factor directional_contrast_db off_axis_contrast_db input_rms_median beam_output_rms beam_output_peak beam_output_clip_fraction samid_raw_score samid_raw_present samid_batch_id samid_inference_ms selected_as_uav selection_reason window_acquisition_valid window_audio_clock_ratio window_receive_rate_ratio window_wall_collection_duration_s window_nominal_audio_duration_s window_max_receive_gap_ms novelty_trigger_id".split()
TRACK_FIELDS = "timestamp elapsed_s state tracked_az tracked_el tracked_az_deg tracked_el_deg predicted_az_deg predicted_el_deg az_velocity_dps el_velocity_dps spatial_score coherence activity_delta_db tracker_hz latest_semantic_task_id latest_samid_score latest_samid_age_ms semantic_worker_busy spatial_valid consecutive_spatial_misses search_radius_deg local_scan_ms beamforming_ms total_spatial_ms frame_age_ms window_acquisition_valid audio_clock_ratio".split()
SEM_FIELDS = "semantic_task_id source_sample_start source_sample_end source_sample_count nominal_duration_s window_valid steering_az steering_el samid_raw_score samid_raw_present task_submit_time inference_start_time inference_end_time result_age_ms input_valid error".split()
RUN_FIELDS = "timestamp elapsed_s state acquisition_process_cpu_pct controller_cpu_pct samid_worker_cpu_pct process_affinity acquisition_rate_ratio_1s acquisition_rate_ratio_5s audio_clock_ratio spatial_update_hz semantic_update_hz novelty_trigger search_running pipeline_lag_ms logger_queue_depth logger_dropped_rows".split()
TRANSITION_FIELDS = "timestamp elapsed_s previous_state new_state reason".split()


class SlidingRate:
    def __init__(self, window_s=5.0):
        self.window_s = float(window_s)
        self.timestamps = deque()

    def tick(self, now):
        self.timestamps.append(float(now))
        cutoff = float(now) - self.window_s
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()

    def hz(self, now):
        cutoff = float(now) - self.window_s
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()
        if len(self.timestamps) < 2:
            return 0.0
        return (len(self.timestamps) - 1) / max(self.timestamps[-1] - self.timestamps[0], 1e-6)


class Health:
    """The v0.10 counter-based acquisition guard."""
    def __init__(self, acq, expected, fs, min_rate, min_clock):
        self.acq = acq
        self.expected = expected
        self.fs = fs
        self.min_rate = min_rate
        self.min_clock = min_clock
        self.points = deque(maxlen=16)
        self.snapshot_stale = 0

    def sample(self, now):
        counters = self.acq.counters
        datagrams = int(counters.datagrams.value)
        samples = int(counters.write_samples.value)
        self.points.append((now, datagrams, samples))
        one = self.points[-2] if len(self.points) > 1 else self.points[0]
        five = next((point for point in self.points if now - point[0] <= 5), self.points[0])
        ratio1 = (datagrams - one[1]) / max((now - one[0]) * self.expected, 1)
        ratio5 = (datagrams - five[1]) / max((now - five[0]) * self.expected, 1)
        started = counters.started_ns.value / 1e9
        expected_samples = max(0, (now - started) * self.fs)
        clock = samples / max(expected_samples, 1)
        healthy = len(self.points) > 1 and ratio1 >= self.min_rate and clock >= self.min_clock and bool(counters.alive.value)
        cutoff = samples - self.fs
        mask = (self.acq.ring.packet_indices >= cutoff) & np.isfinite(self.acq.ring.packet_times)
        packet_times = np.sort(self.acq.ring.packet_times[mask])
        gaps = np.diff(packet_times) * 1000
        return {
            "received_datagrams_total": datagrams, "received_datagrams_per_s": ratio1 * self.expected,
            "expected_datagrams_per_s": self.expected, "observed_expected_ratio_1s": ratio1, "observed_expected_ratio_5s": ratio5,
            "received_sample_frames_total": samples, "expected_sample_frames_total": expected_samples, "audio_clock_ratio": clock,
            "received_bytes_total": int(counters.bytes.value), "malformed_datagrams": int(counters.malformed.value),
            "unexpected_source_datagrams": int(counters.unexpected_source.value),
            "receive_gap_mean_ms": float(np.mean(gaps)) if len(gaps) else math.nan,
            "receive_gap_p95_ms": float(np.quantile(gaps, .95)) if len(gaps) else math.nan,
            "receive_gap_max_ms": float(np.max(gaps)) if len(gaps) else math.nan,
            "shared_ring_fill_s": min(samples, self.acq.ring.capacity) / self.fs,
            "shared_ring_overwrite_count": int(counters.ring_overwrites.value), "snapshot_stale_count": self.snapshot_stale,
            "acquisition_process_alive": bool(counters.alive.value), "window_valid": healthy,
            "health_state": "HEALTHY" if healthy else "ACQ_UNHEALTHY",
            "warning_reason": "" if healthy else "RATE_OR_AUDIO_CLOCK_BELOW_GUARD",
        }


def _key():
    if sys.platform != "win32":
        return None
    import msvcrt
    if not msvcrt.kbhit():
        return None
    char = msvcrt.getwch()
    return "ENTER" if char in ("\r", "\n") else ("ESC" if char == "\x1b" else char.upper())


def _resources(config):
    count = os.cpu_count() or 4
    acquisition_core = max(0, count - 1)
    system = {0}
    candidates = [index for index in range(count) if index not in system | {acquisition_core}]
    threads = int(config["performance"]["samid_intraop_threads"])
    return count, [acquisition_core], candidates[:max(1, min(len(candidates), threads))]


def _physical(snapshot):
    mapped, metadata = to_vendor_physical_order(snapshot.raw_int16, channel_axis=1)
    return mapped.T.astype(np.float32) / 32768.0, metadata


def _activity_delta_db(block, fs, channels, baseline):
    if baseline is None:
        return math.nan
    current = np.asarray(band_features(block, fs, channels, baseline.bands)["band_db"], dtype=float)
    background = np.asarray(baseline.band_median, dtype=float)
    return float(np.max(current - background))


def run_guarded(config_path: Path, results_root: Path, mode="interactive", duration_s=None, synthetic=False, replay_path=None):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    hw = config["hardware"]
    guard = config["acquisition_guard"]
    cal_cfg = config["noise_calibration"]
    gate_cfg = config["noise_gate"]
    pipe = config["semantic_acquire_pipeline"]
    semantic_cfg = config["semantic"]
    loss_cfg = config["target_loss"]
    fs = int(hw["fs"])
    samples_per_datagram = int(hw["expected_datagram_bytes"]) // (int(hw["channels"]) * 2)
    expected = fs / samples_per_datagram
    cpu_count, acq_affinity, samid_affinity = _resources(config)
    geometry_path = (Path(config_path).resolve().parent / Path(hw["geometry_csv"])).resolve()
    xyz = transform_pcb_to_engineering(load_wavefrag_csv(geometry_path), x_sign=int(config["coordinates"]["pcb_x_to_engineering_x_sign"]), y_sign=int(config["coordinates"]["pcb_y_to_engineering_y_sign"]))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = Path(results_root).resolve() / f"realtime_{stamp}"
    session.mkdir(parents=True)
    (session / "diagnostics").mkdir()
    schemas = {"acquisition_health.csv": ACQ_FIELDS, "noise_calibration.csv": CAL_FIELDS, "novelty_gate.csv": NOV_FIELDS, "search_candidates.csv": SEARCH_FIELDS, "track.csv": TRACK_FIELDS, "semantic.csv": SEM_FIELDS, "runtime.csv": RUN_FIELDS, "state_transition.csv": TRANSITION_FIELDS}
    logger = AsyncCSVLogger(session, schemas, queue_size=int(pipe["logger_queue_size"]))
    logger.start()
    artifacts = AsyncArtifactWriter(session / "diagnostics")
    artifacts.start()
    terminal = TerminalRenderer(float(config["terminal"]["refresh_hz"]))
    terminal.start()
    if mode == "replay" and replay_path is None:
        raise ValueError("--mode replay requires --input WAV")
    acq_config = AcquisitionConfig(host=str(hw["udp_host"]), port=int(hw["udp_port"]), source_ip=hw.get("expected_source_ip"), source_port=int(hw["expected_source_port"]), channels=128, fs=fs, datagram_bytes=int(hw["expected_datagram_bytes"]), socket_buffer_bytes=int(hw["socket_buffer_bytes"]), ring_s=float(guard["shared_ring_s"]), affinity=acq_affinity, high_priority=bool(config["performance"]["acquisition_high_priority"]), synthetic=synthetic, replay_path=None if replay_path is None else str(replay_path))
    acquisition = AcquisitionProcess(acq_config)
    acquisition.start()
    health = Health(acquisition, expected, fs, float(guard["minimum_rate_ratio_1s"]), float(guard["minimum_audio_clock_ratio"]))
    controller = ControllerState(float(guard["recovery_duration_s"]))
    reps = representative_channels(xyz, int(cal_cfg["representative_channels"]))
    model_dir = Path(__file__).resolve().parents[2] / "third_party" / "samid_ast_model"
    samid = das = acquirer = tracker = None
    if mode != "receive-only":
        das_config = {"mic_xyz": xyz, "sample_rate": fs, "speed_of_sound_mps": float(pipe["speed_of_sound_mps"]), "spatial_band_hz": tuple(pipe["spatial_band_hz"]), "max_frequency_bins": int(pipe["max_frequency_bins"])}
        samid = SamidProcess(model_dir, int(config["performance"]["samid_intraop_threads"]), int(config["performance"]["samid_interop_threads"]), samid_affinity, das_config=das_config)
        samid.start()
        das = FarFieldDAS(xyz, fs, float(pipe["speed_of_sound_mps"]), tuple(pipe["spatial_band_hz"]), int(pipe["max_frequency_bins"]))
        acquirer = SemanticAcquirer(das, samid, pipe, session / "diagnostics", logger, artifacts)
        tracker = SpatialTracker(TrackerConfig(**pipe["tracker"]))
    acquisition_status = []
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and not acquisition_status:
        acquisition_status = acquisition.status_messages()
        time.sleep(.02)
    info = {
        "software_version": VERSION, "mode": mode, "session_start": datetime.now().astimezone().isoformat(),
        "process_architecture": "Acquisition Process + realtime Controller/Tracker + async Terminal thread + latest-only SAMID Process",
        "controller_pid": os.getpid(), "acquisition_status": acquisition_status, "samid_status": None if samid is None else samid.status,
        "cpu_count": cpu_count, "acquisition_affinity": acq_affinity, "samid_affinity": samid_affinity,
        "torch_intraop_threads": config["performance"]["samid_intraop_threads"], "torch_interop_threads": config["performance"]["samid_interop_threads"],
        "socket_requested_receive_buffer": hw["socket_buffer_bytes"], "socket_actual_receive_buffer": int(acquisition.counters.actual_rcvbuf.value),
        "packet_contract": {"channels": 128, "sample_type": "int16_le", "sample_rate": fs, "datagram_bytes": hw["expected_datagram_bytes"], "samples_per_datagram": samples_per_datagram, "expected_datagrams_per_s": expected, "expected_bytes_per_s": expected * int(hw["expected_datagram_bytes"]), "packet_sequence_gap": "unavailable", "kernel_udp_drop": "unavailable"},
        "acquisition_guard": guard, "noise_calibration": cal_cfg, "noise_gate": gate_cfg, "semantic": semantic_cfg, "target_loss": loss_cfg,
        "production_v010_zip_sha256": PRODUCTION_V010_SHA, "git_commit": "unavailable",
        "model": {"revision": "3a12f618dd8aebf180945bf04ebfeef262d65795", "sha256": MODEL_SHA, "threshold": THRESHOLD},
        "geometry_sha256": hashlib.sha256(geometry_path.read_bytes()).hexdigest().upper(),
        "mapping": {"version": MAPPING_VERSION, "hash": mapping_hash(), "execution_stage": "controller local frames and worker semantic snapshot", "input_channels": 128, "output_channels": 128},
    }
    (session / "session_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    started = time.monotonic()
    last_health = last_monitor = last_track = last_semantic_submit = started
    cal_started = builder = baseline = gate = None
    cal_id = cal_frame = invalid_cal = 0
    cooldown_until = 0.0
    search_running = False
    saved_searches = 0
    latest_health = {"window_valid": False, "observed_expected_ratio_1s": 0.0, "observed_expected_ratio_5s": 0.0, "audio_clock_ratio": 0.0}
    latest_novelty = {}
    latest_spatial = None
    latest_semantic = None
    latest_activity_delta = math.nan
    last_azel_time = None
    pipeline_lag_ms = math.nan
    semantic_task_id = 0
    tracker_rate = SlidingRate()
    semantic_rate = SlidingRate()
    loss = SimpleTargetLoss(float(loss_cfg["low_samid_threshold"]), int(loss_cfg["required_consecutive_low"]), float(loss_cfg["max_activity_delta_db"]))
    mapping_execution_count = 0

    def physical(snapshot):
        nonlocal mapping_execution_count
        mapping_execution_count += 1
        return _physical(snapshot)

    def take_snapshot(duration):
        snapshot = acquisition.snapshot(duration, float(guard["minimum_rate_ratio_1s"]), float(guard["minimum_audio_clock_ratio"]))
        if snapshot is None:
            health.snapshot_stale += 1
        return snapshot

    def transition(new_state, reason):
        event = controller.set_state(new_state, reason)
        if event:
            logger.log("state_transition.csv", {"timestamp": datetime.now().astimezone().isoformat(), "elapsed_s": time.monotonic() - started, **event})
        return event

    def ui_snapshot(now, state_override=None, search=None):
        semantic_status, semantic_age = semantic_state(latest_semantic, now, float(semantic_cfg["stale_after_s"]), bool(samid and samid.busy))
        snapshot = {
            "state": state_override or controller.state, "baseline_ready": baseline is not None, "detection_started": controller.detection_started,
            "az": None if latest_spatial is None else latest_spatial.get("tracked_az_deg"), "el": None if latest_spatial is None else latest_spatial.get("tracked_el_deg"),
            "azel_age_ms": None if last_azel_time is None else (now - last_azel_time) * 1000,
            "spatial_score": None if latest_spatial is None else latest_spatial.get("spatial_score"),
            "coherence": None if latest_spatial is None else latest_spatial.get("coherence_factor"), "activity_delta_db": latest_activity_delta,
            "samid_score": None if latest_semantic is None else latest_semantic.get("score"), "samid_state": semantic_status, "samid_age_s": semantic_age,
            "low_count": loss.low_count, "required_low": loss.required_consecutive_low,
            "acq_rate": latest_health.get("observed_expected_ratio_1s", 0), "audio_clock": latest_health.get("audio_clock_ratio", 0),
            "input_valid": latest_health.get("window_valid", False), "tracker_hz": tracker_rate.hz(now), "samid_hz": semantic_rate.hz(now),
            "samid_worker_busy": bool(samid and samid.busy), "pipeline_lag_ms": pipeline_lag_ms,
            "novelty": latest_novelty.get("aggregate_novelty_score"), "trigger": latest_novelty.get("triggered", False),
            "cal_remaining_s": max(0.0, float(cal_cfg["duration_s"]) - (now - cal_started)) if cal_started else 0.0,
        }
        if search is not None:
            snapshot.update(search)
        terminal.publish(snapshot)
        return snapshot

    try:
        import psutil
        controller_proc = psutil.Process(os.getpid())
        acq_proc = psutil.Process(acquisition.process.pid)
        samid_proc = None if samid is None else psutil.Process(samid.process.pid)
        controller_proc.cpu_percent(); acq_proc.cpu_percent()
        if samid_proc:
            samid_proc.cpu_percent()
    except Exception:
        controller_proc = acq_proc = samid_proc = None

    try:
        while True:
            now = time.monotonic()
            elapsed = now - started
            if duration_s is not None and elapsed >= duration_s:
                break
            key = _key()
            if key in ("Q", "ESC"):
                break
            if mode == "interactive":
                if key == "ENTER":
                    previous = controller.state
                    action = controller.enter()
                    if controller.state != previous:
                        logger.log("state_transition.csv", {"timestamp": datetime.now().astimezone().isoformat(), "elapsed_s": elapsed, "previous_state": previous, "new_state": controller.state, "reason": "operator_enter"})
                    if action == "START_CALIBRATION":
                        cal_id += 1; cal_started = now; builder = BaselineBuilder(reps, cal_cfg["bands_hz"]); cal_frame = invalid_cal = 0
                elif key == "R":
                    previous = controller.state; controller.recalibrate(); baseline = gate = cal_started = None; loss.reset(); latest_semantic = latest_spatial = None
                    logger.log("state_transition.csv", {"timestamp": datetime.now().astimezone().isoformat(), "elapsed_s": elapsed, "previous_state": previous, "new_state": controller.state, "reason": "operator_recalibrate"})
                elif key == "P":
                    previous = controller.state; controller.pause()
                    if previous != controller.state:
                        logger.log("state_transition.csv", {"timestamp": datetime.now().astimezone().isoformat(), "elapsed_s": elapsed, "previous_state": previous, "new_state": controller.state, "reason": "operator_pause_toggle"})

            completed = None if samid is None else samid.poll_track_results()
            if completed is not None:
                if completed.get("input_valid", False):
                    latest_semantic = completed
                    semantic_rate.tick(now)
                    decision = loss.observe(completed["score"], latest_activity_delta)
                    logger.log("semantic.csv", {
                        "semantic_task_id": completed["semantic_task_id"], "source_sample_start": completed["source_sample_start"], "source_sample_end": completed["source_sample_end"],
                        "source_sample_count": completed["source_sample_count"], "nominal_duration_s": 1.0, "window_valid": completed["window_valid"],
                        "steering_az": completed["steering_az_deg"], "steering_el": completed["steering_el_deg"], "samid_raw_score": completed["score"],
                        "samid_raw_present": completed["present"], "task_submit_time": completed["created_time"], "inference_start_time": completed["inference_start_time"],
                        "inference_end_time": completed["finished_time"], "result_age_ms": max(0.0, (now - completed["finished_time"]) * 1000), "input_valid": True, "error": "",
                    })
                    if decision["should_reacquire"] and controller.state in ("TRACK", "COAST"):
                        transition("REACQUIRE", "consecutive_low_samid_and_activity_near_background")
                else:
                    logger.log("semantic.csv", {"semantic_task_id": completed.get("semantic_task_id"), "window_valid": False, "input_valid": False, "error": completed.get("error", "worker_error"), "inference_end_time": completed.get("finished_time")})

            if controller.state == "REACQUIRE":
                # Preserve v0.10's automatic REACQUIRE -> SEARCH_IDLE behavior.
                # A returning source is picked up by the unchanged novelty gate;
                # no third Enter is introduced.
                transition("SEARCH_IDLE", "automatic_reacquire_wait_for_novelty")

            if now - last_health >= 1.0:
                latest_health = health.sample(now)
                previous = controller.state
                controller.acquisition_health(bool(latest_health["window_valid"]), now)
                if controller.state != previous:
                    logger.log("state_transition.csv", {"timestamp": datetime.now().astimezone().isoformat(), "elapsed_s": elapsed, "previous_state": previous, "new_state": controller.state, "reason": "acquisition_health"})
                acpu = acq_proc.cpu_percent() if acq_proc else "unavailable"
                ccpu = controller_proc.cpu_percent() if controller_proc else "unavailable"
                scpu = samid_proc.cpu_percent() if samid_proc else (0 if mode == "receive-only" else "unavailable")
                ram = (controller_proc.memory_info().rss + (acq_proc.memory_info().rss if acq_proc else 0) + (samid_proc.memory_info().rss if samid_proc else 0)) if controller_proc else "unavailable"
                stamp_iso = datetime.now().astimezone().isoformat()
                logger.log("acquisition_health.csv", {"timestamp": stamp_iso, "elapsed_s": elapsed, "acquisition_process_cpu_pct": acpu, "controller_cpu_pct": ccpu, "samid_worker_cpu_pct": scpu, "ram_bytes": ram, **latest_health})
                logger.log("runtime.csv", {"timestamp": stamp_iso, "elapsed_s": elapsed, "state": controller.state, "acquisition_process_cpu_pct": acpu, "controller_cpu_pct": ccpu, "samid_worker_cpu_pct": scpu, "process_affinity": json.dumps({"acq": acq_affinity, "samid": samid_affinity}), "acquisition_rate_ratio_1s": latest_health.get("observed_expected_ratio_1s"), "acquisition_rate_ratio_5s": latest_health.get("observed_expected_ratio_5s"), "audio_clock_ratio": latest_health.get("audio_clock_ratio"), "spatial_update_hz": tracker_rate.hz(now), "semantic_update_hz": semantic_rate.hz(now), "novelty_trigger": latest_novelty.get("triggered", False), "search_running": search_running, "pipeline_lag_ms": pipeline_lag_ms, "logger_queue_depth": logger.queue.qsize(), "logger_dropped_rows": logger.dropped})
                last_health = now

            if mode == "receive-only":
                ui_snapshot(now)
                time.sleep(.01)
                continue

            if controller.state == "NOISE_CALIBRATION" and now - last_monitor >= gate_cfg["hop_ms"] / 1000:
                snapshot = take_snapshot(gate_cfg["frame_ms"] / 1000)
                valid = snapshot is not None and snapshot.valid and latest_health.get("window_valid", False)
                cal_frame += 1
                if valid:
                    audio, _ = physical(snapshot); features = band_features(audio, fs, reps, cal_cfg["bands_hz"]); builder.add(features); powers = json.dumps(features["band_db"].tolist())
                else:
                    invalid_cal += 1; powers = ""
                remaining = max(0, float(cal_cfg["duration_s"]) - (now - cal_started))
                logger.log("noise_calibration.csv", {"timestamp": datetime.now().astimezone().isoformat(), "calibration_id": cal_id, "frame_index": cal_frame, "acquisition_valid": valid, "band_powers_db": powers, "background_building_status": "BUILDING" if valid else "INVALID", "remaining_s": remaining, "invalid_reason": "" if valid else "ACQUISITION_DISCONTINUITY"})
                last_monitor = now
                if remaining <= 0:
                    success = invalid_cal == 0 and cal_frame >= int(float(cal_cfg["duration_s"]) * 1000 / gate_cfg["hop_ms"] * .9)
                    if success:
                        baseline = builder.build(); data = baseline.as_dict(); data["calibration_acquisition_health"] = {"valid_frames": cal_frame - invalid_cal, "invalid_frames": invalid_cal, "final_rate_ratio_1s": latest_health.get("observed_expected_ratio_1s"), "final_rate_ratio_5s": latest_health.get("observed_expected_ratio_5s"), "final_audio_clock_ratio": latest_health.get("audio_clock_ratio")}; (session / "noise_baseline.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
                        gate = NoveltyGate(baseline, float(gate_cfg["trigger_z"]), float(gate_cfg["trigger_delta_db"]), int(gate_cfg["minimum_triggered_bands"]), int(gate_cfg["persistence_required_frames"]), int(gate_cfg["persistence_window_frames"]))
                    previous = controller.state; controller.calibration_complete(success)
                    logger.log("state_transition.csv", {"timestamp": datetime.now().astimezone().isoformat(), "elapsed_s": elapsed, "previous_state": previous, "new_state": controller.state, "reason": "calibration_complete" if success else "calibration_failed"})
                    cal_started = None

            elif controller.state == "SEARCH_IDLE" and gate and now - last_monitor >= gate_cfg["hop_ms"] / 1000:
                snapshot = take_snapshot(gate_cfg["frame_ms"] / 1000)
                valid = snapshot is not None and snapshot.valid and latest_health.get("window_valid", False)
                result = {"triggered": False, "aggregate_novelty_score": math.nan, "triggered_band_count": 0, "persistence_count": 0, "trigger_reason": "", "band_delta_db": [], "band_novelty_z": []}
                if valid:
                    audio, _ = physical(snapshot); result = gate.update(band_features(audio, fs, reps, cal_cfg["bands_hz"])); latest_novelty = result
                logger.log("novelty_gate.csv", {"timestamp": datetime.now().astimezone().isoformat(), "state": controller.state, "acquisition_valid": valid, "band_delta_db": json.dumps(result["band_delta_db"]), "band_novelty_z": json.dumps(result["band_novelty_z"]), **result, "search_running": search_running, "cooldown_remaining_s": max(0, cooldown_until - now)})
                last_monitor = now
                if result["triggered"] and now >= cooldown_until and not search_running:
                    transition("SEARCH_TRIGGERED", "novelty_trigger")
                    search_running = True
                    full = take_snapshot(1.0)
                    window_id = f"{full.start_sample}-{full.end_sample}" if full else "unavailable"
                    if full is None or not full.valid:
                        logger.log("semantic.csv", {"semantic_task_id": None, "window_valid": False, "input_valid": False, "error": "ACQUISITION_DISCONTINUITY"})
                        transition("ACQ_UNHEALTHY", "invalid_search_window")
                    else:
                        block, _ = physical(full)
                        context = {"window_id": window_id, "window_acquisition_valid": True, "window_audio_clock_ratio": full.audio_clock_ratio, "window_receive_rate_ratio": full.receive_rate_ratio, "window_wall_collection_duration_s": full.wall_collection_duration_s, "window_nominal_audio_duration_s": full.nominal_audio_duration_s, "window_max_receive_gap_ms": full.max_receive_gap_ms, "novelty_trigger_id": result.get("novelty_trigger_id"), "save_diagnostics": saved_searches < int(config["diagnostics"]["max_saved_searches"])}
                        if context["save_diagnostics"]:
                            search_id = acquirer.search_id + 1; rms = np.sqrt(np.mean(block * block, axis=1)); median_index = int(np.argmin(np.abs(rms - np.median(rms)))); max_index = int(np.argmax(rms)); chosen = list(dict.fromkeys([0, median_index, max_index, *reps[:4]]))
                            for channel in chosen:
                                artifacts.wav(f"search_{search_id:04d}_physical_ch_{channel:03d}.wav", fs, block[channel])
                            artifacts.json(f"search_{search_id:04d}_metadata.json", {**context, "physical_channels_saved": chosen, "mapping_version": MAPPING_VERSION, "mapping_hash": mapping_hash(), "mapping_execution_count_for_snapshot": 1})
                        output = acquirer.run(block, "SEARCH", context)
                        saved_searches += int(context["save_diagnostics"])
                        selected = output["selected"]
                        event_now = time.monotonic()
                        search_snapshot = ui_snapshot(event_now, "SEARCH_RESULT", {"search_id": output["search_id"], "candidates": output["candidates"], "selected": selected})
                        terminal.publish_event(search_snapshot)
                        if selected is None:
                            transition("SEARCH_IDLE", "search_no_samid_candidate"); cooldown_until = time.monotonic() + float(gate_cfg["search_cooldown_s"])
                        else:
                            tracker.acquire(time.monotonic(), selected["az_deg"], selected["el_deg"])
                            latest_spatial = {**selected, "tracked_az_deg": selected["az_deg"], "tracked_el_deg": selected["el_deg"], "predicted_az_deg": selected["az_deg"], "predicted_el_deg": selected["el_deg"], "az_velocity_dps": 0.0, "el_velocity_dps": 0.0, "spatial_valid": True}
                            last_azel_time = time.monotonic(); latest_semantic = None; loss.reset(); transition("TRACK", "samid_acquired")
                    search_running = False

            elif controller.state in ("TRACK", "COAST") and now - last_track >= .08:
                frame_start = time.perf_counter()
                snapshot = take_snapshot(.1)
                valid = snapshot is not None and snapshot.valid
                radius = scan_ms = beam_ms = 0.0
                if not valid:
                    update = tracker.update(now, None, 0.0, math.nan)
                    latest_spatial = {**update, "spatial_score": math.nan, "coherence_factor": math.nan}
                    audio_clock = None if snapshot is None else snapshot.audio_clock_ratio
                else:
                    block, _ = physical(snapshot)
                    predicted_az, predicted_el, dt = tracker.predict(now)
                    predicted_az = float(np.clip(predicted_az, -90, 90)); predicted_el = float(np.clip(predicted_el, -45, 45))
                    radius = tracker.radius(dt)
                    azimuths = np.arange(max(-90, predicted_az - radius), min(90, predicted_az + radius) + .1, float(pipe["local_step_deg"]))
                    elevations = np.arange(max(-45, predicted_el - radius), min(45, predicted_el + radius) + .1, float(pipe["local_step_deg"]))
                    scan = das.scan(block, azimuths, elevations)
                    update = tracker.update(now, (scan["best_az"], scan["best_el"]), scan["best_score"], scan["second_score"])
                    beam_started = time.perf_counter(); _, beam_quality = das.beamform(block, update["tracked_az_deg"], update["tracked_el_deg"]); beam_ms = (time.perf_counter() - beam_started) * 1000
                    latest_activity_delta = _activity_delta_db(block, fs, reps, baseline)
                    latest_spatial = {**update, "spatial_score": scan["best_score"], "coherence_factor": scan["best_score"], **beam_quality}
                    scan_ms = scan["duration_ms"]; audio_clock = snapshot.audio_clock_ratio; pipeline_lag_ms = max(0, (time.perf_counter() - snapshot.wall_end_s) * 1000); last_azel_time = now
                tracker_rate.tick(now)
                if update["new_state"] != controller.state:
                    transition(update["new_state"], update["transition_reason"])
                if controller.state == "REACQUIRE":
                    transition("SEARCH_IDLE", "automatic_reacquire_wait_for_novelty")
                semantic_status, semantic_age = semantic_state(latest_semantic, now, float(semantic_cfg["stale_after_s"]), samid.busy)
                logger.log("track.csv", {"timestamp": datetime.now().astimezone().isoformat(), "elapsed_s": elapsed, "state": controller.state, "tracked_az": latest_spatial.get("tracked_az_deg"), "tracked_el": latest_spatial.get("tracked_el_deg"), **latest_spatial, "coherence": latest_spatial.get("coherence_factor"), "activity_delta_db": latest_activity_delta, "tracker_hz": tracker_rate.hz(now), "latest_semantic_task_id": None if latest_semantic is None else latest_semantic.get("semantic_task_id"), "latest_samid_score": None if latest_semantic is None else latest_semantic.get("score"), "latest_samid_age_ms": None if semantic_age is None else semantic_age * 1000, "semantic_worker_busy": samid.busy, "search_radius_deg": radius, "local_scan_ms": scan_ms, "beamforming_ms": beam_ms, "total_spatial_ms": (time.perf_counter() - frame_start) * 1000, "frame_age_ms": pipeline_lag_ms, "window_acquisition_valid": valid, "audio_clock_ratio": audio_clock})
                last_track = now

                if controller.state in ("TRACK", "COAST") and now - last_semantic_submit >= float(semantic_cfg["track_interval_s"]):
                    semantic_task_id += 1
                    semantic_snapshot = take_snapshot(1.0)
                    if semantic_snapshot is None or not semantic_snapshot.valid or not latest_health.get("window_valid", False):
                        logger.log("semantic.csv", {"semantic_task_id": semantic_task_id, "source_sample_start": None if semantic_snapshot is None else semantic_snapshot.start_sample, "source_sample_end": None if semantic_snapshot is None else semantic_snapshot.end_sample, "source_sample_count": None if semantic_snapshot is None else semantic_snapshot.end_sample - semantic_snapshot.start_sample, "nominal_duration_s": 1.0, "window_valid": False, "steering_az": latest_spatial.get("tracked_az_deg"), "steering_el": latest_spatial.get("tracked_el_deg"), "task_submit_time": now, "input_valid": False, "error": "ACQUISITION_DISCONTINUITY"})
                    else:
                        task = build_semantic_task(semantic_task_id, semantic_snapshot.raw_int16, semantic_snapshot.start_sample, semantic_snapshot.end_sample, latest_spatial["tracked_az_deg"], latest_spatial["tracked_el_deg"], True, semantic_snapshot.audio_clock_ratio, semantic_snapshot.receive_rate_ratio, now)
                        samid.submit_track(task)
                    last_semantic_submit = now

            ui_snapshot(now)
            time.sleep(.002)
        return 0
    finally:
        info["session_end"] = datetime.now().astimezone().isoformat()
        info["mapping"]["execution_count"] = mapping_execution_count
        info["final_logger_dropped_rows"] = logger.dropped
        if samid is not None:
            info["semantic_submitted_tasks"] = samid.submitted_track_tasks
            info["semantic_dropped_pending_tasks"] = samid.dropped_pending_tasks
        (session / "session_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        terminal.close()
        if samid:
            samid.close()
        acquisition.close()
        artifacts.close()
        logger.close()
        print(f"Session saved to: {session}")
