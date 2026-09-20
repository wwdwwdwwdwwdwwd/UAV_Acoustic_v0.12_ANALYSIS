# Known Limitations

- v0.12 deliberately uses one fixed tracker direction for each continuous one-second semantic block. It does not implement time-varying steering for a moving target.
- The activity guard is the maximum calibrated band-power delta across the frozen representative physical channels. It is intentionally transparent and conservative, but it is not a direction-isolated classifier.
- A one-second, 128-channel int16 snapshot is about 20.5 MB. Snapshot copy and multiprocessing serialization are bounded and nonblocking with respect to inference, but memory bandwidth should be observed on the deployment PC.
- Actual tracker rate depends on local grid size and deployment CPU. Synthetic scheduling tests demonstrate decoupling, not the cost of a particular acoustic scene's local DAS grid.
- SAMID task processing can take longer than the 1.0-second submission interval. Latest-only replacement prevents backlog; skipped task IDs are expected.
- Semantic staleness never causes loss. If SAMID stalls indefinitely, spatial tracking/coasting remains authoritative.
- Hardware acquisition (~16000 datagrams/s and audio clock near 1.0), real U2 acquisition, direction accuracy, and physical loss/reacquisition require the stated field run. No WaveFrag device was available for this software-only build.
- Production packaging includes frozen model weights. The analysis ZIP intentionally excludes weights, caches, virtual environments, recordings, and historical ZIPs.
