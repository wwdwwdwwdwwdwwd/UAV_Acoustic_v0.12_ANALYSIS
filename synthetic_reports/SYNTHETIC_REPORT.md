# v0.12 Synthetic Validation

Passed tests:

- startup remains `WAIT_FOR_CALIBRATION`;
- first Enter starts calibration only;
- successful calibration waits for second Enter;
- second Enter starts `SEARCH_IDLE`;
- unhealthy calibration generates no baseline;
- stable background does not trigger;
- persistent new source triggers once and does not queue searches;
- simulated 10/30/50% losses suppress SAMID and return `INVALID_INPUT`;
- baseline feature extraction does not modify waveform samples;
- isolated acquisition remains alive under sustained global DAS and frozen SAMID load.
- tracker cadence remains 12.8205 Hz with a 5-second mock SAMID;
- semantic input is an exact continuous 80000-frame acquisition interval;
- stale SAMID and a 100 ms terminal renderer do not stop TRACK;
- conjunctive target-loss cases F-H behave as specified;
- active recovery never requests a third Enter.

The original v0.10 guards remain passing. v0.12 architecture evidence is `_internal/dev_results/v12_architecture_tests.json`.
