"""Snapshot-only terminal UI isolated from the realtime tracker loop."""
from __future__ import annotations

import math
import sys
import threading
import time
from typing import Callable


def _number(value, fmt: str, fallback: str = "--") -> str:
    try:
        number = float(value)
        return format(number, fmt) if math.isfinite(number) else fallback
    except (TypeError, ValueError):
        return fallback


def format_terminal(snapshot: dict) -> str:
    state = snapshot.get("state", "STARTING")
    lines = ["=" * 60, "UAV ACOUSTIC v0.12", "", f"STATE             : {state}"]
    if state in ("TRACK", "COAST"):
        lines.extend([
            f"UAV TARGET        : {snapshot.get('uav_target', 'CONFIRMED')}",
            "",
            f"AZ / EL           : {_number(snapshot.get('az'), '+.1f')} / {_number(snapshot.get('el'), '+.1f')} deg",
            f"AZ/EL AGE         : {_number(snapshot.get('azel_age_ms'), '.0f')} ms",
            "",
            f"SPATIAL SCORE     : {_number(snapshot.get('spatial_score'), '.3f')}",
            f"COHERENCE         : {_number(snapshot.get('coherence'), '.3f')}",
            f"ACTIVITY delta    : {_number(snapshot.get('activity_delta_db'), '+.1f')} dB",
            "",
            f"SAMID RAW         : {_number(snapshot.get('samid_score'), '.3f')}",
            f"SAMID STATE       : {snapshot.get('samid_state', 'STALE')}",
            f"SAMID AGE         : {_number(snapshot.get('samid_age_s'), '.2f')} s",
            f"LOW COUNT         : {snapshot.get('low_count', 0)} / {snapshot.get('required_low', 3)}",
        ])
    elif state == "SEARCH_IDLE":
        lines.extend([
            "UAV TARGET        : NONE", "",
            f"BASELINE          : {'READY' if snapshot.get('baseline_ready') else 'NOT READY'}",
            f"NOVELTY           : {_number(snapshot.get('novelty'), '.2f')}",
            f"TRIGGER           : {'YES' if snapshot.get('trigger') else 'NO'}",
        ])
    elif state == "SEARCH_RESULT":
        lines.extend(["", f"SEARCH #{snapshot.get('search_id', '--')}", "", "RANK   AZ      EL      SPATIAL   COHERENCE   SAMID"])
        for candidate in snapshot.get("candidates", []):
            selected = " <- SELECTED" if candidate.get("selected_as_uav") else ""
            lines.append(f"{candidate.get('candidate_rank', 0):<6} {candidate.get('az_deg', 0):+6.1f}  {candidate.get('el_deg', 0):+6.1f}  {candidate.get('spatial_score', 0):8.3f}  {candidate.get('coherence_factor', 0):9.3f}  {candidate.get('samid_raw_score', 0):5.3f}{selected}")
        selected = snapshot.get("selected")
        if selected:
            lines.extend(["", "UAV ACQUIRED", f"AZ / EL           : {selected['az_deg']:+.1f} / {selected['el_deg']:+.1f} deg", f"SAMID RAW         : {selected['samid_raw_score']:.3f}"])
    else:
        lines.extend(["", f"BASELINE          : {'READY' if snapshot.get('baseline_ready') else 'NOT READY'}", f"CAL REMAINING     : {_number(snapshot.get('cal_remaining_s'), '.1f')} s"])
    lines.extend([
        "",
        f"ACQ 1s            : {_number(100.0 * snapshot.get('acq_rate', 0.0), '.1f')} %",
        f"AUDIO CLOCK       : {_number(100.0 * snapshot.get('audio_clock', 0.0), '.2f')} %",
        f"INPUT             : {'VALID' if snapshot.get('input_valid') else 'INVALID'}",
        "",
        f"TRACK Hz          : {_number(snapshot.get('tracker_hz'), '.1f')}",
        f"SAMID Hz          : {_number(snapshot.get('samid_hz'), '.1f')}",
        f"SAMID WORKER      : {'BUSY' if snapshot.get('samid_worker_busy') else 'IDLE'}",
        f"PIPELINE LAG      : {_number(snapshot.get('pipeline_lag_ms'), '.1f')} ms",
        "",
        "ENTER             : no action" if snapshot.get("detection_started") else "ENTER             : next setup step",
        "R                 : recalibrate",
        "P                 : pause",
        "Q                 : quit",
        "=" * 60,
    ])
    return "\n".join(lines)


class TerminalRenderer:
    """An atomic latest-snapshot publisher plus a throttled render thread."""

    def __init__(self, refresh_hz: float = 4.0, render_fn: Callable[[dict], None] | None = None):
        self.refresh_hz = float(refresh_hz)
        self._latest: dict | None = None
        self._event: dict | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._render_fn = render_fn or self._console_render

    @staticmethod
    def _console_render(snapshot: dict) -> None:
        if sys.stdout.isatty():
            print("\x1b[2J\x1b[H", end="")
        print(format_terminal(snapshot), flush=True)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="terminal-renderer", daemon=True)
        self._thread.start()

    def publish(self, snapshot: dict) -> None:
        # Assignment is constant-time and the renderer never mutates snapshots.
        self._latest = dict(snapshot)

    def publish_event(self, snapshot: dict) -> None:
        """Retain the newest one-shot event until the renderer displays it."""
        self._event = dict(snapshot)

    def _run(self) -> None:
        period = 1.0 / max(self.refresh_hz, 0.1)
        deadline = time.monotonic()
        while not self._stop.is_set():
            deadline += period
            snapshot = self._event or self._latest
            if snapshot is not None:
                self._render_fn(snapshot)
                if snapshot is self._event:
                    self._event = None
            self._stop.wait(max(0.0, deadline - time.monotonic()))

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
