"""Small, deterministic primitives for asynchronous TRACK semantics."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np


SEMANTIC_SAMPLE_RATE = 80_000
SEMANTIC_SAMPLE_COUNT = 80_000


def build_semantic_task(
    task_id: int,
    raw_sample_major: np.ndarray,
    source_sample_start: int,
    source_sample_end: int,
    steering_az_deg: float,
    steering_el_deg: float,
    window_valid: bool,
    audio_clock_ratio: float,
    receive_rate_ratio: float,
    created_monotonic_s: float | None = None,
) -> dict[str, Any]:
    """Freeze exactly one contiguous second into a worker payload.

    ``raw_sample_major`` is the acquisition snapshot in ``(80000, 128)`` raw
    channel order. The task owns a channel-major C-contiguous copy so later
    shared-ring writes cannot change semantic input.
    """
    raw = np.asarray(raw_sample_major)
    if raw.shape != (SEMANTIC_SAMPLE_COUNT, 128):
        raise ValueError(f"semantic source must be (80000, 128), got {raw.shape}")
    if raw.dtype != np.dtype("<i2") and raw.dtype != np.dtype(np.int16):
        raise ValueError(f"semantic source must be int16, got {raw.dtype}")
    start = int(source_sample_start)
    end = int(source_sample_end)
    if end - start != SEMANTIC_SAMPLE_COUNT:
        raise ValueError(f"semantic source range must span 80000 samples, got {start}:{end}")
    if not bool(window_valid):
        raise ValueError("invalid acquisition window cannot become a semantic task")
    return {
        "semantic_task_id": int(task_id),
        "created_monotonic_s": float(time.monotonic() if created_monotonic_s is None else created_monotonic_s),
        "raw_128ch_1s": np.ascontiguousarray(raw.T),
        "sample_rate": SEMANTIC_SAMPLE_RATE,
        "steering_az_deg": float(steering_az_deg),
        "steering_el_deg": float(steering_el_deg),
        "source_sample_start": start,
        "source_sample_end": end,
        "source_sample_count": end - start,
        "window_valid": True,
        "audio_clock_ratio": float(audio_clock_ratio),
        "receive_rate_ratio": float(receive_rate_ratio),
    }


def validate_semantic_task(task: dict[str, Any]) -> None:
    raw = np.asarray(task["raw_128ch_1s"])
    if raw.shape != (128, SEMANTIC_SAMPLE_COUNT):
        raise ValueError(f"semantic task must contain (128, 80000), got {raw.shape}")
    if int(task["sample_rate"]) != SEMANTIC_SAMPLE_RATE:
        raise ValueError("semantic task sample rate must be 80000 Hz")
    start = int(task["source_sample_start"])
    end = int(task["source_sample_end"])
    if end - start != SEMANTIC_SAMPLE_COUNT or int(task.get("source_sample_count", -1)) != SEMANTIC_SAMPLE_COUNT:
        raise ValueError("semantic task source range is not one contiguous 80000-frame interval")
    if not bool(task.get("window_valid", False)):
        raise ValueError("semantic task window is invalid")


@dataclass
class SimpleTargetLoss:
    """Lose a target only when semantic-low count and low activity coincide."""

    low_samid_threshold: float = 0.30
    required_consecutive_low: int = 3
    max_activity_delta_db: float = 1.5
    low_count: int = 0

    def reset(self) -> None:
        self.low_count = 0

    def observe(self, samid_score: float, activity_delta_db: float) -> dict[str, Any]:
        score = float(samid_score)
        activity = float(activity_delta_db)
        self.low_count = self.low_count + 1 if score < self.low_samid_threshold else 0
        semantic_low = self.low_count >= self.required_consecutive_low
        activity_near_background = activity <= self.max_activity_delta_db
        return {
            "low_count": self.low_count,
            "semantic_low": semantic_low,
            "activity_near_background": activity_near_background,
            "should_reacquire": semantic_low and activity_near_background,
        }


def semantic_state(latest_result: dict[str, Any] | None, now: float, stale_after_s: float, worker_busy: bool) -> tuple[str, float | None]:
    if latest_result is None:
        return ("INFERENCE RUNNING" if worker_busy else "STALE"), None
    finished = float(latest_result["finished_time"])
    age_s = max(0.0, float(now) - finished)
    if age_s > float(stale_after_s):
        return "STALE", age_s
    return ("PRESENT" if bool(latest_result["present"]) else "LOW"), age_s
