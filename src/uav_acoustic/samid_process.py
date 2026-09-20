"""Frozen SAMID process with synchronous SEARCH and latest-only TRACK APIs."""
from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import queue
import time
import uuid
from pathlib import Path

import numpy as np

from .async_semantic import validate_semantic_task

MODEL_SHA = "10455420F5AF15A7287BE1D32D37AD6B681F74E3C152FE3BF386FDB0A0427489"


def _replace_nowait(mailbox, value) -> bool:
    """Place a value without waiting, replacing one queued stale value."""
    try:
        mailbox.put_nowait(value)
        return True
    except queue.Full:
        try:
            mailbox.get_nowait()
        except queue.Empty:
            return False
        try:
            mailbox.put_nowait(value)
            return True
        except queue.Full:
            return False


def _worker(model_dir, threads, interop, affinity, das_config, requests, responses, ready, stop, busy, mock_delay_s, mock_score):
    status = {"pid": os.getpid(), "intraop_threads": threads, "interop_threads": interop, "affinity_requested": affinity, "affinity_actual": None, "errors": []}
    try:
        import psutil
        process = psutil.Process()
        if affinity:
            process.cpu_affinity(affinity)
        status["affinity_actual"] = process.cpu_affinity()
    except Exception as exc:
        status["errors"].append(f"affinity: {type(exc).__name__}: {exc}")
    try:
        das = None
        if das_config is not None:
            from .das_tracking import FarFieldDAS
            das = FarFieldDAS(np.asarray(das_config["mic_xyz"], dtype=float), int(das_config["sample_rate"]), float(das_config["speed_of_sound_mps"]), tuple(das_config["spatial_band_hz"]), int(das_config["max_frequency_bins"]))
        if mock_delay_s is None:
            import torch
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
            torch.set_num_threads(int(threads))
            torch.set_num_interop_threads(int(interop))
            root = Path(model_dir)
            actual = hashlib.sha256((root / "model.safetensors").read_bytes()).hexdigest().upper()
            if actual != MODEL_SHA:
                raise RuntimeError(f"model SHA mismatch: {actual}")
            extractor = AutoFeatureExtractor.from_pretrained(str(root), local_files_only=True)
            model = AutoModelForAudioClassification.from_pretrained(str(root), local_files_only=True).eval()

            def infer(waves):
                t0 = time.perf_counter()
                inputs = extractor([x for x in np.asarray(waves, np.float32)], sampling_rate=16000, return_tensors="pt")
                feature_ms = (time.perf_counter() - t0) * 1000
                t0 = time.perf_counter()
                with torch.inference_mode():
                    scores = torch.softmax(model(**inputs).logits, dim=-1)[:, 1].cpu().numpy().astype(float)
                return scores, feature_ms, (time.perf_counter() - t0) * 1000
        else:
            status["mock_delay_s"] = float(mock_delay_s)

            def infer(waves):
                time.sleep(float(mock_delay_s))
                return np.full(len(waves), float(mock_score), dtype=float), 0.0, float(mock_delay_s) * 1000

        ready.put(status)
        while not stop.is_set():
            try:
                request = requests.get(timeout=0.1)
            except queue.Empty:
                continue
            if request is None:
                break
            busy.value = 1
            kind = request[0]
            try:
                if kind == "batch":
                    _, request_id, waves = request
                    scores, feature_ms, inference_ms = infer(waves)
                    responses.put(("batch", request_id, scores, feature_ms, inference_ms, None))
                elif kind == "track":
                    task = request[1]
                    validate_semantic_task(task)
                    if das is None:
                        raise RuntimeError("TRACK semantic DAS is not configured")
                    inference_start = time.monotonic()
                    from scipy.signal import resample_poly
                    from .io.channel_mapping import to_vendor_physical_order
                    raw = np.asarray(task["raw_128ch_1s"], dtype=np.int16)
                    mapped, _ = to_vendor_physical_order(raw, channel_axis=0)
                    physical = mapped.astype(np.float32) / 32768.0
                    mono80, quality = das.beamform(physical, task["steering_az_deg"], task["steering_el_deg"])
                    wave16 = resample_poly(mono80, 16000, 80000).astype(np.float32)[-16000:]
                    scores, feature_ms, inference_ms = infer(wave16[None, :])
                    finished = time.monotonic()
                    result = {
                        "semantic_task_id": int(task["semantic_task_id"]), "score": float(scores[0]), "present": bool(scores[0] >= 0.5),
                        "steering_az_deg": float(task["steering_az_deg"]), "steering_el_deg": float(task["steering_el_deg"]),
                        "source_sample_start": int(task["source_sample_start"]), "source_sample_end": int(task["source_sample_end"]), "source_sample_count": int(task["source_sample_count"]),
                        "created_time": float(task["created_monotonic_s"]), "inference_start_time": inference_start, "finished_time": finished,
                        "age_ms": max(0.0, (finished - float(task["created_monotonic_s"])) * 1000), "feature_ms": float(feature_ms), "inference_ms": float(inference_ms),
                        "input_valid": True, "window_valid": True, "audio_clock_ratio": float(task["audio_clock_ratio"]), "receive_rate_ratio": float(task["receive_rate_ratio"]),
                        "beam_output_rms": quality["output_rms"], "beam_output_peak": quality["output_peak"], "beam_output_clip_fraction": quality["output_clip_fraction"],
                    }
                    _replace_nowait(responses, ("track", result))
                else:
                    raise ValueError(f"unknown SAMID request kind: {kind}")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if kind == "batch":
                    responses.put(("batch", request[1], None, 0.0, 0.0, error))
                else:
                    task = request[1]
                    _replace_nowait(responses, ("track", {"semantic_task_id": task.get("semantic_task_id"), "input_valid": False, "error": error, "finished_time": time.monotonic()}))
            finally:
                busy.value = 0
    except Exception as exc:
        ready.put({**status, "fatal": f"{type(exc).__name__}: {exc}"})


class SamidProcess:
    """One inference process; TRACK submission and polling are always nonblocking."""
    def __init__(self, model_dir: Path, threads=4, interop=1, affinity=None, das_config=None, mock_delay_s=None, mock_score=0.7):
        self.requests = mp.Queue(maxsize=1)
        self.responses = mp.Queue(maxsize=2)
        self.ready_queue = mp.Queue()
        self.stop = mp.Event()
        self.busy_flag = mp.Value("b", 0, lock=False)
        self.process = mp.Process(target=_worker, args=(str(model_dir), threads, interop, affinity, das_config, self.requests, self.responses, self.ready_queue, self.stop, self.busy_flag, mock_delay_s, mock_score), name="frozen-samid", daemon=True)
        self.status = None
        self.latest_track_result = None
        self.submitted_track_tasks = 0
        self.dropped_pending_tasks = 0

    @property
    def busy(self) -> bool:
        return bool(self.busy_flag.value)

    def start(self, timeout=60):
        self.process.start()
        self.status = self.ready_queue.get(timeout=timeout)
        if "fatal" in self.status:
            raise RuntimeError(self.status["fatal"])

    def submit_track(self, task) -> bool:
        validate_semantic_task(task)
        envelope = ("track", task)
        try:
            self.requests.put_nowait(envelope)
            self.submitted_track_tasks += 1
            return True
        except queue.Full:
            try:
                old = self.requests.get_nowait()
            except queue.Empty:
                return False
            if old is not None:
                self.dropped_pending_tasks += 1
            try:
                self.requests.put_nowait(envelope)
                self.submitted_track_tasks += 1
                return True
            except queue.Full:
                return False

    def poll_track_results(self):
        latest = None
        while True:
            try:
                item = self.responses.get_nowait()
            except queue.Empty:
                break
            if item[0] == "track":
                latest = item[1]
            else:
                _replace_nowait(self.responses, item)
                break
        if latest is not None:
            self.latest_track_result = latest
        return latest

    def infer_batch_16k(self, waveforms, timeout=30):
        """Validated v0.10 blocking batch API, used only by global SEARCH."""
        request_id = uuid.uuid4().hex
        deadline = time.monotonic() + float(timeout)
        while True:
            try:
                pending = self.requests.get_nowait()
                if pending is not None and pending[0] == "track":
                    self.dropped_pending_tasks += 1
            except queue.Empty:
                break
        self.requests.put(("batch", request_id, np.asarray(waveforms, np.float32)), timeout=max(0.1, deadline - time.monotonic()))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("SAMID batch inference timed out")
            item = self.responses.get(timeout=remaining)
            if item[0] == "track":
                self.latest_track_result = item[1]
                continue
            _, got, scores, feature, inference, error = item
            if got != request_id:
                continue
            if error:
                raise RuntimeError(error)
            return np.asarray(scores, float), feature, inference

    def close(self):
        self.stop.set()
        try:
            self.requests.put_nowait(None)
        except Exception:
            pass
        self.process.join(5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(1)
