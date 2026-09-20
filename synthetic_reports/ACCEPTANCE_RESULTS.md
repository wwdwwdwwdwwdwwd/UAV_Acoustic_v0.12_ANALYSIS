# v0.12 Acceptance Results

Software release blockers A-J: **PASS**.

| Test | Result |
|---|---|
| A — tracker not blocked by 5 s SAMID | PASS — median 12.8205 Hz; 63 updates / 5 s |
| B — semantic input contiguous | PASS — exact `(128, 80000)` and one absolute 80000-frame range |
| C — no `mono_parts` gap stitching | PASS |
| D — 1.2 s inference over 20 s | PASS — median 12.8205 Hz; 250 updates |
| E — SAMID stale | PASS — changing AZ/EL and tracker continues |
| F — intermittent semantic low | PASS — target retained |
| G — consecutive low/activity +8 dB | PASS — target retained |
| H — three low/activity +0.5 dB | PASS — TRACK to REACQUIRE decision |
| I — no third Enter | PASS |
| J — 100 ms terminal renderer | PASS — tracker remains >=10 Hz |

Final full suite: 16 passed in 35.31 seconds. Dedicated A-J benchmark run: 10 passed in 35.13 seconds. Actual mock-process payload smoke test also returned task 7 with score 0.7 and a continuous source count of 80000.

Hardware field acceptance remains pending because this environment has no WaveFrag/U2 input. Consequently acquisition throughput, actual real-scene local-DAS rate, real U2 reacquisition, and real acoustic loss behavior are not claimed as physically verified by this build.
