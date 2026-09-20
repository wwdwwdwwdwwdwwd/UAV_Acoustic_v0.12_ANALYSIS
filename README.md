# UAV Acoustic Realtime AZ/EL v0.12

`v0.12_ASYNC_TRACK_SEMANTIC` is derived exclusively from the on-machine-validated v0.10 acquisition-guarded noise-baseline release. It keeps acquisition, calibration, novelty, global Top-K/NMS, DAS, SAMID acquire, geometry, mapping, and tracker mathematics unchanged.

The TRACK loop targets 12.5 Hz and never waits for SAMID. TRACK semantic confirmation is submitted at approximately 1 Hz to a latest-only worker. Each task freezes the newest valid, continuous acquisition interval `[source_sample_start, source_sample_end)` with exactly 80000 frames and performs fixed-direction full-second DAS inside the SAMID process. The terminal renders snapshots in its own 4 Hz thread.

## Start

Run `START.bat` from the package root. The interaction is fixed:

1. First Enter starts 10-second noise calibration.
2. Second Enter starts automatic detection.
3. No further Enter is needed after loss, reacquisition, search failure, or acquisition recovery.
4. `R` recalibrates, `P` pauses/resumes, and `Q` or Esc exits.

Configuration is `_internal/03_configs/acq_guarded_noise.yaml`. Frozen SAMID CPU settings remain 4 intra-op threads and 1 inter-op thread.

## Loss rule

TRACK transitions to automatic REACQUIRE only when both conditions are true on a newly completed semantic result:

- three consecutive SAMID scores are below 0.30; and
- current acoustic activity is no more than +1.5 dB above the frozen calibration background.

SAMID staleness or low SAMID alone never drops the target. Spatial COAST/REACQUIRE behavior remains the v0.10 tracker behavior.

## Validation

Run from `_internal` with its Python environment on `PYTHONPATH=02_src`:

```powershell
python -m pytest 05_tests -q
python dev/benchmark_v12_architecture.py
```

The release-blocker suite covers tests A-J. See `ACCEPTANCE_RESULTS.md`, `ARCHITECTURE_AUDIT.md`, and `IMPLEMENTATION_REPORT.md`. Hardware throughput and acoustic behavior still require the field procedure in `KNOWN_LIMITATIONS.md` because synthetic tests cannot replace a WaveFrag/U2 run.
