# X5-TriBEV-Flow PC Acceptance

Status: **PC_ACCEPTED_BOARD_PENDING**

## Frozen Boundary

- Validated entry remains `bash ~/tools/finals_lift_nav_demo.sh`.
- F407 build remains `2026071907`.
- Validated distance remains `0.50 m`.
- Candidate publishers are restricted to `/x5_triflow_shadow/*`.
- Candidate errors become `MONITOR_OFFLINE`; the validated demo is not blocked.

## Laptop Results

- Focused tests: 55 passed, exit code 0.
- Baseline verification: `true`.
- TinyOccFlow synthetic occupancy mIoU: 0.972968.
- Gain over occupancy persistence: 0.078740.
- TinyOccFlow synthetic flow mean EPE: 0.080766 m.
- Flow mean-EPE reduction versus zero-flow: 77.42%.
- Conformal empirical coverage: 91.67%
  at 90.00% nominal coverage.
- CamSemLite procedural semantic mIoU: 0.939642.
- CamSemLite real-camera accuracy validation: **not performed**.

## Bayes-e Artifacts

- TinyOccFlow `.bin`: `90e01859991c2eabaf71147de299123c656569ef1115df049d03e25f1471fdf9`.
- CamSemLite `.bin`: `cb582808a90ae93c46dbffdce2ddd676ceacfeab3d865a67f755c322968c6f7c`.
- Compiler estimates are toolchain estimates, not board measurements:
  Tiny 126.1 us;
  Cam 167.0 us.

## Remaining Gate

The embodied X5 was off during this phase. Board runtime latency, BPU load,
ION/CMA, RSS, thermal state, live topic rates, and non-interference are still
`PENDING_X5_POWER_ON`. No board-runtime or real-navigation accuracy claim is
made by this report.
