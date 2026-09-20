# v0.12 Implementation Report

Baseline: `UAV_Acoustic_Realtime_AZEL_v0.10_ACQ_GUARDED_NOISE_BASELINE`. v0.11 code was not used as the implementation base.

1. **Tracker loop location:** the controller process main realtime loop in `_internal/02_src/uav_acoustic/guarded_runtime.py`.
2. **Tracker target frequency:** 12.5 Hz (80 ms cadence), within the requested 10-20 Hz range.
3. **SAMID worker location:** independent `frozen-samid` multiprocessing process. That process performs TRACK fixed-direction DAS, resampling, and frozen SAMID inference.
4. **Blocking SAMID wait in Tracker:** none. TRACK calls only `submit_track` and `poll_track_results`; both use nonblocking queue operations.
5. **Maximum SAMID queue length:** one pending task, in addition to at most one task already executing.
6. **Old task discard behavior:** if the pending slot is occupied, a new TRACK task removes and replaces that pending task. Completed TRACK results are drained and only the newest is retained.
7. **Source of 80000 samples:** one `AcquisitionProcess.snapshot(1.0)` from the v0.10 shared raw ring; its absolute interval is logged as `[source_sample_start, source_sample_end)`.
8. **Strict continuity:** yes. Task construction requires shape `(80000, 128)` before transposition, exact range difference 80000, valid acquisition, and 80000 Hz sample rate.
9. **Noncontinuous `mono_parts` stitching:** absent from the v0.12 runtime and semantic modules. TRACK semantics never concatenate local tracker chunks.
10. **Tracker rate with mock SAMID sleep 5 s:** median 12.8205 Hz, 63 updates in 5 seconds. PASS.
11. **Terminal effect on tracker rate:** renderer runs in an independent thread and receives constant-time latest-snapshot assignments. With a simulated 100 ms render cost, the cadence test remains at least 10 Hz. PASS.
12. **v0.10 core files completely unchanged:** `process_acquisition.py`, `io/wavefrag_udp.py`, `io/channel_mapping.py`, `io/geometry.py`, `coordinates.py`, `das_tracking.py`, `semantic_acquire.py`, `noise_gate.py`, `detection_adapters/samid_ast.py`, vendor geometry, model config/preprocessor, and `model.safetensors`; SHA-256 equality was verified.

Additional measured result: with 1.2-second mock inference over 20 seconds, the median tracker cadence was 12.8205 Hz with 250 updates.

The only target-loss rule is consecutive low SAMID **and** activity near background. There is no semantic-only timeout, 8-second fallback, complex evidence state machine, or v0.11 local-invalid recovery behavior. One invalid local snapshot invokes the unchanged tracker prediction/COAST path and does not itself force REACQUIRE.
