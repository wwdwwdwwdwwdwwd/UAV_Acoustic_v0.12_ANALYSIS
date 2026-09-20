# UAV Acoustic v0.12 Architecture Change Plan

## Baseline and scope

The sole implementation baseline is `UAV_Acoustic_Realtime_AZEL_v0.10_ACQ_GUARDED_NOISE_BASELINE`. v0.11 is not a code source. This release changes only TRACK/SAMID scheduling, the semantic worker interface and continuous semantic waveform construction, lightweight terminal rendering, the simple conjunctive loss decision, and the logs/tests needed to prove those properties.

## Files to modify

- `_internal/02_src/uav_acoustic/guarded_runtime.py`: preserve acquisition, calibration, novelty, global search, and tracker mathematics while wiring independent tracker, semantic, and terminal loops.
- `_internal/02_src/uav_acoustic/samid_process.py`: add a non-blocking latest-only TRACK task/result interface; retain the frozen SAMID adapter and 4/1 thread settings.
- `_internal/02_src/uav_acoustic/async_semantic.py`: define and validate continuous TRACK task payloads and the simple conjunctive loss rule.
- `_internal/02_src/uav_acoustic/controller_state.py`: make post-start recovery automatic and record simple state transitions.
- `_internal/02_src/uav_acoustic/terminal_renderer.py`: new snapshot-based, independently throttled terminal renderer.
- `_internal/03_configs/acq_guarded_noise.yaml`: add only semantic interval/staleness, terminal refresh, and simple target-loss settings; keep 4 intra-op and 1 inter-op thread.
- `_internal/05_tests/test_v12_async_architecture.py`: add release-blocker architecture tests A-J.
- `_internal/dev/benchmark_v12_architecture.py`: generate repeatable tracker/SAMID/terminal timing evidence.
- `README.md`, `LOG_SCHEMA.md`, `KNOWN_LIMITATIONS.md`, `IMPLEMENTATION_REPORT.md`: v0.12 operator and implementation documentation.
- `ARCHITECTURE_AUDIT.md`: generated after implementation and source audit.

## Files prohibited from modification

- WaveFrag decoder: `_internal/02_src/uav_acoustic/io/wavefrag_udp.py`
- H1 mapping: `_internal/02_src/uav_acoustic/io/channel_mapping.py`
- Geometry and coordinate convention: `_internal/02_src/uav_acoustic/io/geometry.py`, `_internal/02_src/uav_acoustic/coordinates.py`, vendor geometry CSV
- Global DAS/search math: `_internal/02_src/uav_acoustic/das_tracking.py`, `_internal/02_src/uav_acoustic/semantic_acquire.py`
- NMS and Top-K candidate selection in `semantic_acquire.py`
- Fractional DAS waveform and 1/128 normalization in `das_tracking.py`
- SAMID preprocessing/model adapter: `_internal/02_src/uav_acoustic/detection_adapters/samid_ast.py`
- Frozen SAMID model files and revision under `_internal/third_party/samid_ast_model/`
- Noise calibration and novelty core: `_internal/02_src/uav_acoustic/noise_gate.py`
- Acquisition process/shared-memory architecture: `_internal/02_src/uav_acoustic/process_acquisition.py`
- Tracker mathematics/configuration in `_internal/02_src/uav_acoustic/das_tracking.py`

## Realtime loops

```text
Acquisition Process -> Shared Ring -> Tracker/Controller loop
                                      |-> latest terminal snapshot -> Terminal thread
                                      `-> latest-only task -> SAMID Process
```

- Acquisition loop: existing independent v0.10 process; continuously receives into the shared ring.
- Tracker loop: controller process, target cadence 12.5 Hz (80 ms); takes the newest valid 100 ms snapshot, runs the unchanged local scan/predict/update/beam operations, and publishes state.
- SAMID loop: independent process; approximately 1 Hz TRACK submissions. Each task owns an exact continuous `(128, 80000)` acquisition snapshot and fixed tracker direction, then performs full-second DAS, 80 kHz to 16 kHz resampling, and frozen SAMID inference.
- Terminal loop: independent thread, default 4 Hz; reads immutable/latest display snapshots only.

## Blocking calls by loop

- Acquisition loop: existing blocking `recvfrom_into` behavior only, isolated in the acquisition process.
- Tracker loop: short shared-ring snapshot copy and spatial computation. **SAMID blocking calls = NONE.** No `queue.get`, `response_queue.get`, `future.result`, worker/process join, or event wait.
- SAMID loop: worker-side request wait and inference are allowed because they cannot gate tracker progress.
- Terminal loop: condition wait and console output are isolated from the tracker; tracker only performs a bounded latest-snapshot assignment.
- Shutdown path: bounded joins are allowed only after the realtime loop has ended.

## Queue semantics

TRACK semantic scheduling is latest-only:

```text
one task currently in inference
+ at most one latest pending task
```

The pending capacity is 1. A new task replaces an older not-yet-started pending task. Completed results are drained non-blockingly and only the newest completed task is retained by the controller. Acquisition/global-search SAMID remains the already-validated v0.10 synchronous batch path.

## Semantic waveform source

Every TRACK semantic task uses the latest continuous 128-channel, 1.0-second acquisition snapshot, exactly 80000 sample frames with `source_sample_end - source_sample_start == 80000`, after acquisition-valid and audio-clock guards pass. The worker applies fixed-direction DAS using the tracker direction captured with that snapshot. It never constructs semantic audio from concatenated local tracker chunks.

## Release gates

Production and analysis ZIPs will be created only if tests A-J, source audit, protected-file hash comparison against v0.10, and the full relevant test suite pass. Hardware-only acquisition throughput remains a documented field acceptance item when no WaveFrag device is available.
