# X5-TriBEV-Flow Finals Successor

This directory contains the independent finals-upgrade candidate for the
embodied-brain stack.

Status: **RDK X5 fixed-input BPU contract accepted for manual shadow use;
real-session sensor validation remains pending.**

The 2026-08-04 board receipt is
`docs/X5_BOARD_ACCEPTANCE_20260804.md`. TinyOccFlow and CamSemLite executed on
the actual Bayes-e BPU through `hbm_runtime`; 200-run latency, fixed
FP32/INT8 differential and 30 load/unload recovery gates passed. No candidate
service, control authority or automatic activation was added.

## Candidate stack

1. A read-only ROS 2 collector stages synchronized LiDAR, depth-derived scan,
   odometry, Vision-BEV, and Lab-FSD reference evidence. Incomplete future
   labels stay in raw staging and are never promoted as complete episodes.
2. `TriBEVFrontend` builds five temporal frames with eight sensor channels per
   frame, for a fixed `1 x 40 x 64 x 64` tensor.
3. The selected 32,712-parameter TinyOccFlow v5r1 Bayes-e model predicts
   occupancy at `+0.4/+0.8/+1.2 s`, dynamic probability, flow, uncertainty,
   and an auxiliary nine-token distribution.
4. X5 CPU samples nine arcs against the predicted occupancy using the measured
   `0.50 x 0.40 m` chassis, `0.08 m` margin, and current odometry speed.
5. ShadowGuard records model, footprint, optional Lab-FSD/Nav2 reference, OOD,
   cross-modal agreement, sensor health, and bounded evidence files.

The selected TinyOccFlow artifact is:

```text
bpu/artifacts/tiny_occ_flow/90e01859991c2eab/tiny_occ_flow.bin
SHA-256 90e01859991c2eabaf71147de299123c656569ef1115df049d03e25f1471fdf9
```

CamSemLite remains an optional procedural-pretrained BPU probe. It is disabled
by default and has no real-camera accuracy claim.

## Non-interference contract

- The validated entry remains `bash ~/tools/finals_lift_nav_demo.sh`.
- F407 build `2026071907` remains the only validated firmware contract.
- Candidate code must not publish `/cmd_vel`, `/cmd_vel_safe`, authoritative
  TF, `map -> odom`, or any F407 command/service.
- Candidate failures must only mark the shadow stack offline. They must never
  block the validated 0.50 m demonstration.
- The AI-brain 4K camera is optional input. Missing, stale, cached, or fixture
  data must be represented explicitly and must not block LiDAR/depth replay.

## Laptop verification

```bash
python tools/verify_finals_baseline.py --json
python tools/build_pc_acceptance_report.py
python tools/package_successor.py --kind tooling --dry-run --json
```

The machine-readable model decision is `evidence/model_selection_v5.json`.
The current acceptance receipt is `evidence/pc_acceptance_report.v1.json`.
Synthetic/procedural metrics and compiler estimates are not board or
real-navigation measurements.

## Board phase

Do not enable this candidate as an automatic service. The pure BPU/fixed-input
part of `docs/X5_BOARD_VALIDATION_RUNBOOK.md` is complete. Collector and live
shadow promotion remain pending because the observed session did not provide
usable `/imu` and `/odom` messages. Candidate failure is fail-open with respect
to the frozen demo: it only marks the monitor offline.
