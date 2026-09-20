from __future__ import annotations

from collections import deque
from pathlib import Path
import time

import numpy as np
from scipy.signal import resample_poly


INTERNAL_ROOT = Path(__file__).resolve().parents[3]


class CadencedMonoAdapter:
    """Transport-only 80 kHz physical-channel-0 ring and sample-and-hold."""
    adapter_id = "unknown"
    upstream_name = "unknown"
    model_sample_rate = 16000
    window_samples = 16000
    hop_samples = 8000
    threshold = 0.5

    def __init__(self, fs: int, config: dict):
        if int(fs) != 80000:
            raise ValueError("the frozen WaveFrag transport must run at 80000 Hz")
        cfg = config["open_source_detector_adapter"]
        if int(cfg.get("physical_channel_index", -1)) != 0:
            raise ValueError("semantic input is locked to physical-order channel index 0")
        self.cfg = cfg
        self.raw_ring = deque(maxlen=int(round(self.window_samples * fs / self.model_sample_rate)))
        self.raw_until_next = int(round(self.window_samples * fs / self.model_sample_rate))
        self.sequence = 0
        self.last_update_monotonic = None
        self.last = self._empty_result()

    def _empty_result(self) -> dict:
        return {
            "semantic_adapter_id": self.adapter_id,
            "semantic_upstream_name": self.upstream_name,
            "semantic_channel_index": 0,
            "semantic_model_sample_rate_hz": self.model_sample_rate,
            "semantic_window_samples": self.window_samples,
            "semantic_hop_samples": self.hop_samples,
            "semantic_score": None,
            "semantic_decision": False,
            "semantic_result_fresh": False,
            "semantic_sequence": 0,
            "semantic_preprocess_ms": None,
            "semantic_inference_ms": None,
            "semantic_score_age_ms": None,
        }

    def accept_realtime_frame(self, audio: np.ndarray, transport_hop_samples: int) -> dict:
        value = np.asarray(audio, np.float32)
        mono = value[0, -int(transport_hop_samples):]
        needed = int(round(self.window_samples * 80000 / self.model_sample_rate))
        hop_raw = int(round(self.hop_samples * 80000 / self.model_sample_rate))
        offset=0; updated=False
        while offset < len(mono):
            take=min(self.raw_until_next,len(mono)-offset)
            self.raw_ring.extend(mono[offset:offset+take].tolist()); offset+=take; self.raw_until_next-=take
            if self.raw_until_next>0: continue
            if len(self.raw_ring)!=needed: raise RuntimeError("semantic cadence ring invariant failed")
            raw = np.asarray(self.raw_ring, np.float32)
            t0 = time.perf_counter(); model_audio = self.prepare_transport(raw)
            prep_ms = (time.perf_counter() - t0) * 1000.0
            score, model_prep_ms, infer_ms = self.infer_model_chunk(model_audio)
            self.sequence += 1; self.last_update_monotonic = time.monotonic(); updated=True
            self.last = {
                "semantic_adapter_id": self.adapter_id, "semantic_upstream_name": self.upstream_name,
                "semantic_channel_index": 0, "semantic_model_sample_rate_hz": self.model_sample_rate,
                "semantic_window_samples": self.window_samples, "semantic_hop_samples": self.hop_samples,
                "semantic_score": float(score), "semantic_decision": bool(score >= self.threshold),
                "semantic_result_fresh": True, "semantic_sequence": self.sequence,
                "semantic_preprocess_ms": float(prep_ms + model_prep_ms),
                "semantic_inference_ms": float(infer_ms), "semantic_score_age_ms": 0.0,
            }
            self.raw_until_next=hop_raw
        if not updated:
            held = dict(self.last); held["semantic_result_fresh"] = False
            if self.last_update_monotonic is not None:
                held["semantic_score_age_ms"] = (time.monotonic()-self.last_update_monotonic)*1000.0
            return held
        return dict(self.last)

    def infer_model_chunk(self, audio_16k: np.ndarray) -> tuple[float, float, float]:
        raise NotImplementedError

    def prepare_transport(self, raw_80k: np.ndarray) -> np.ndarray:
        value = resample_poly(raw_80k, self.model_sample_rate, 80000).astype(np.float32)
        return value[-self.window_samples:]
