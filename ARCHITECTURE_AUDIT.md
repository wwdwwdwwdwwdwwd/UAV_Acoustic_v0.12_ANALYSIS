# v0.12 Architecture Audit

Audit baseline: v0.10. Audit date: 2026-09-20.

| Check | Result | Evidence |
|---|---|---|
| `response_queue.get` in tracker path | PASS | No response queue exists in the controller; TRACK uses `poll_track_results()` -> `get_nowait()` only. |
| `future.result` in tracker path | PASS | No futures are used. |
| `worker.join` / `process.join` in tracker path | PASS | No joins in the realtime loop. Bounded process/thread joins occur only in shutdown `close()` methods. |
| Event wait in tracker path | PASS | No event wait in controller TRACK. Terminal waits only in its own thread. |
| Old `mono_parts` semantic stitching | PASS | No `mono_parts`, local-chunk concatenate, or `[-80000:]` semantic construction in runtime/semantic source. |
| Continuous semantic source | PASS | Builder requires one valid `(80000, 128)` acquisition snapshot and exact absolute range difference 80000. |
| SAMID latest-only | PASS | Request mailbox `maxsize=1`; new TRACK submission replaces pending task; worker can execute only one task. |
| Old semantic result treated as new | PASS | Result carries task ID/timestamps; UI computes age and marks results stale after 2.5 s. |
| Terminal blocking tracker | PASS | Rendering and console output run in `terminal-renderer`; tracker only assigns an immutable latest snapshot. |
| Semantic-only target loss | PASS | No timeout loss. `SimpleTargetLoss` requires low-count threshold and activity <=1.5 dB together. |
| Third Enter after active start | PASS | `detection_started` recovery returns to `SEARCH_IDLE`; subsequent Enter is ignored. |
| Protected v0.10 core files | PASS | SHA-256 equality confirmed for acquisition, decoder, mapping, geometry, DAS/tracker math, NMS/Top-K, noise/novelty, SAMID adapter/model, and vendor geometry. |
| SAMID threads | PASS | Configuration is 4 intra-op / 1 inter-op. |
| Architecture tests A-J | PASS | 10/10 passed; full relevant suite 16/16 passed. |

No critical audit failures were found. Software packaging is permitted; hardware field acceptance remains explicitly pending.
