# Inherited v0.10 Synthetic Test Report

| Test | Result |
|---|---|
| A — stronger non-UAV candidates | semantic rank 3 selected, PASS |
| B — no UAV | none selected; remains SEARCH, PASS |
| C — broad-lobe NMS | one direction retained from 20° lobe, PASS |
| D — enhancement | correct/wrong coherence 0.999998/0.185376; 7.319 dB, PASS |
| E — batch equivalence | maximum probability error 8.94e-8, PASS |
| Steering | three cases each <=1.0°, PASS |
| Moving tracker | 20/60/120°/s max errors 0.819/2.457/4.914°, no reacquire, PASS |

Batch inference took about 1.80 s versus 1.85 s total for five individual forwards on this CPU. Exact U2 semantic acquisition (34 s window, injected at `AZ=24°, EL=8°`) selected `AZ=24°, EL=9°`, score 0.833297, error 1.0°.

Machine evidence: `_internal/dev_results/v09_validation.json`, `exact_u2_acquire.json`, and `sustained_120s_summary.json`.

These frozen acquisition/search results are inherited from the sole v0.10 baseline. The v0.12-specific A-J results are in `ACCEPTANCE_RESULTS.md` and `_internal/dev_results/v12_architecture_tests.json`.
