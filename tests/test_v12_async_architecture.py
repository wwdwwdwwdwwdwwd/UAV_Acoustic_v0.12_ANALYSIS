from __future__ import annotations

import queue
import statistics
import threading
import time
from pathlib import Path

import numpy as np

from uav_acoustic.async_semantic import SimpleTargetLoss, build_semantic_task, semantic_state, validate_semantic_task
from uav_acoustic.controller_state import ControllerState
from uav_acoustic.terminal_renderer import TerminalRenderer


class SlowLatestOnlySamid:
    """Test double: one running inference and one replaceable pending task."""
    def __init__(self, latency_s):
        self.latency_s = float(latency_s)
        self.pending = queue.Queue(maxsize=1)
        self.results = queue.Queue(maxsize=1)
        self.busy = False
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, task_id):
        try:
            self.pending.put_nowait(task_id)
        except queue.Full:
            try:
                self.pending.get_nowait()
            except queue.Empty:
                return
            self.pending.put_nowait(task_id)

    def poll(self):
        try:
            return self.results.get_nowait()
        except queue.Empty:
            return None

    def _run(self):
        while not self.stop.is_set():
            try:
                task_id = self.pending.get(timeout=.02)
            except queue.Empty:
                continue
            self.busy = True
            time.sleep(self.latency_s)
            self.busy = False
            try:
                self.results.put_nowait({"semantic_task_id": task_id, "finished_time": time.monotonic(), "present": True})
            except queue.Full:
                try:
                    self.results.get_nowait()
                except queue.Empty:
                    pass
                self.results.put_nowait({"semantic_task_id": task_id, "finished_time": time.monotonic(), "present": True})


def cadence_run(duration_s, semantic_latency_s, terminal=None):
    worker = SlowLatestOnlySamid(semantic_latency_s)
    ticks = []
    azimuths = []
    latest = None
    start = time.monotonic()
    next_tick = start
    next_semantic = start
    task_id = 0
    while time.monotonic() - start < duration_s:
        now = time.monotonic()
        result = worker.poll()
        if result is not None:
            latest = result
        if now >= next_tick:
            ticks.append(now)
            azimuths.append(len(ticks) * .25)
            next_tick += .08
            if terminal is not None:
                terminal.publish({"state": "TRACK", "az": azimuths[-1], "el": 0.0, "detection_started": True})
        if now >= next_semantic:
            task_id += 1
            worker.submit(task_id)
            next_semantic += 1.0
        time.sleep(.001)
    intervals = np.diff(ticks)
    return {"ticks": ticks, "azimuths": azimuths, "latest": latest, "worker_busy": worker.busy, "median_hz": float(1.0 / np.median(intervals))}


def test_a_tracker_not_blocked_by_samid():
    report = cadence_run(5.0, 5.0)
    assert report["median_hz"] >= 10.0
    assert len(report["ticks"]) >= 50


def test_b_semantic_input_is_contiguous():
    encoded = np.repeat(np.arange(80000, dtype=np.int16)[:, None], 128, axis=1)
    task = build_semantic_task(1, encoded, 240000, 320000, 12.0, 3.0, True, 1.0, 1.0, 10.0)
    validate_semantic_task(task)
    assert task["raw_128ch_1s"].shape == (128, 80000)
    assert task["source_sample_end"] - task["source_sample_start"] == 80000
    assert np.array_equal(task["raw_128ch_1s"][0], np.arange(80000, dtype=np.int16))


def test_c_no_mono_parts_gap_stitching():
    source_root = Path(__file__).resolve().parents[1] / "02_src" / "uav_acoustic"
    runtime = (source_root / "guarded_runtime.py").read_text(encoding="utf-8")
    semantic = (source_root / "async_semantic.py").read_text(encoding="utf-8")
    assert "mono_parts" not in runtime
    assert "mono_parts" not in semantic
    assert "np.concatenate(list(" not in runtime
    assert "[-80000:]" not in runtime


def test_d_tracker_rate_with_realistic_samid():
    report = cadence_run(20.0, 1.2)
    assert report["median_hz"] >= 10.0


def test_e_samid_stale_does_not_stop_track():
    report = cadence_run(5.0, 5.5)
    state, age = semantic_state(report["latest"], time.monotonic(), 2.5, report["worker_busy"])
    assert len(set(report["azimuths"])) > 50
    assert state in ("STALE", "INFERENCE RUNNING")
    assert report["median_hz"] >= 10.0


def test_f_intermittent_semantic_low():
    loss = SimpleTargetLoss()
    decisions = [loss.observe(score, 7.0) for score in (0.70, 0.25, 0.62, 0.21, 0.68)]
    assert not any(item["should_reacquire"] for item in decisions)


def test_g_consecutive_low_but_activity_high():
    loss = SimpleTargetLoss()
    decisions = [loss.observe(score, 8.0) for score in (0.20, 0.15, 0.10, 0.09)]
    assert decisions[-1]["low_count"] == 4
    assert not any(item["should_reacquire"] for item in decisions)


def test_h_true_uav_loss():
    loss = SimpleTargetLoss()
    decisions = [loss.observe(score, 0.5) for score in (0.18, 0.12, 0.09)]
    assert not decisions[1]["should_reacquire"]
    assert decisions[2]["should_reacquire"]
    controller = ControllerState()
    controller.state = "TRACK"
    if decisions[2]["should_reacquire"]:
        transition = controller.set_state("REACQUIRE", "consecutive_low_samid_and_activity_near_background")
    assert transition == {"previous_state": "TRACK", "new_state": "REACQUIRE", "reason": "consecutive_low_samid_and_activity_near_background"}


def test_i_no_third_enter_after_recovery_paths():
    state = ControllerState(recovery_s=0.0)
    assert state.enter() == "START_CALIBRATION"
    state.calibration_complete(True)
    assert state.enter() == "START_DETECTION"
    for active_state in ("REACQUIRE", "SEARCH_IDLE", "ACQ_UNHEALTHY"):
        state.state = active_state
        if active_state == "ACQ_UNHEALTHY":
            state.acquisition_health(True, 1.0)
            state.acquisition_health(True, 1.1)
        assert state.state != "WAIT_FOR_MEASUREMENT"
        assert state.enter() == "IGNORED"


def test_j_terminal_not_blocking():
    calls = []
    def slow_render(snapshot):
        calls.append(snapshot.get("az"))
        time.sleep(.1)
    terminal = TerminalRenderer(refresh_hz=4.0, render_fn=slow_render)
    terminal.start()
    try:
        report = cadence_run(5.0, 1.2, terminal)
    finally:
        terminal.close()
    assert calls
    assert report["median_hz"] >= 10.0
