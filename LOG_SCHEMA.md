# v0.12 Log Schema

All session logs are written asynchronously under `results/realtime_YYYYMMDD_HHMMSS/`.

## track.csv

Core proof fields are `timestamp`, `tracked_az`, `tracked_el`, `spatial_score`, `coherence`, `activity_delta_db`, `tracker_hz`, `latest_semantic_task_id`, `latest_samid_score`, `latest_samid_age_ms`, and `semantic_worker_busy`. Existing tracker diagnostic fields are retained, including prediction, velocity, validity, misses, search radius, timing, window validity, and audio-clock ratio.

## semantic.csv

Fields are `semantic_task_id`, `source_sample_start`, `source_sample_end`, `source_sample_count`, `nominal_duration_s`, `window_valid`, `steering_az`, `steering_el`, `samid_raw_score`, `samid_raw_present`, `task_submit_time`, `inference_start_time`, `inference_end_time`, `result_age_ms`, `input_valid`, and `error`.

For every valid row:

```text
source_sample_end - source_sample_start = source_sample_count = 80000
nominal_duration_s = 1.0
window_valid = True
```

## state_transition.csv

Fields are only `timestamp`, `elapsed_s`, `previous_state`, `new_state`, and `reason`. There is no target-evidence state machine.

## Other unchanged session logs

`acquisition_health.csv`, `noise_calibration.csv`, `novelty_gate.csv`, `search_candidates.csv`, and `runtime.csv` retain the v0.10 responsibilities. `runtime.csv` adds directly observable spatial/semantic rates and pipeline lag.
