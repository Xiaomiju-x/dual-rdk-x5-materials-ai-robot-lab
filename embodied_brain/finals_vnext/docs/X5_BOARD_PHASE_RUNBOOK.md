# X5-TriBEV-Flow v2 Board Phase Runbook

Status: **do not execute until the user explicitly powers the embodied X5.**

This runbook validates an independent shadow candidate. It never overwrites
`~/tools`, `~/ros2_ws`, F407 firmware, Nav2/SLAM configuration, or the frozen
`bash ~/tools/finals_lift_nav_demo.sh` entry.

## 1. Preconditions

- User confirms PC and embodied X5 are on `xrd-lab_5G`.
- Use only `rdk@192.0.2.85`.
- Verify the recorded ED25519 host key, hostname `embodied-x5`, and device
  identity before any copy.
- Do not change PC Wi-Fi, TUN, VPN, proxy, route, or ARP.
- Vehicle and lift stay idle; no physical run is needed for model profiling.
- Candidate remains manual and is never registered as a boot service.

## 2. Revalidate the Laptop Candidate

```powershell
C:\Users\YOUR_USER\miniconda3\envs\xrd\python.exe `
  embodied_brain\finals_vnext\tools\build_pc_acceptance.py
C:\Users\YOUR_USER\miniconda3\envs\xrd\python.exe `
  embodied_brain\finals_vnext\tools\package_candidate.py
```

Required: `PC_ACCEPTED_BOARD_PENDING`, 100 tests, valid held-out replay, valid
Bayes-e conversion audit, and unchanged frozen hashes.

## 3. Read-Only Board Inventory

Before transfer, record without starting or stopping services:

- hostname, OS image, kernel, TROS/ROS 2 version;
- `hobot_dnn`, `hrt_model_exec`, BPU runtime and package versions;
- total/available RAM, process RSS/PSS, CMA/ION state;
- BPU status, temperatures and throttling state;
- topic names/types/rates for LiDAR, depth, odometry and existing diagnostics;
- existing frozen service/process identities and hashes.

If identity or runtime compatibility is uncertain, stop the candidate work.
Do not upgrade the board OS or runtime to make the model fit.

## 4. Immutable Transfer

Verify the release ZIP and copy it to a new content-addressed directory:

```text
~/xrd_candidates/finals_vnext/<content_sha256>/
```

Do not extract over an existing directory. Do not write into `~/tools` or the
installed ROS workspace. Record archive SHA-256 before and after transfer.

## 5. Pure BPU Gate

With motion idle:

1. load only the reviewed `.bin`;
2. run fixed synthetic tensors and compare every output with the PC reference;
3. record actual p50/p95/p99 latency, BPU utilization, RSS/PSS, CMA/ION and
   temperature before/during/after;
4. unload and confirm resource recovery.

Acceptance targets:

- output parity within the documented INT8 tolerance;
- p95 below the `10 ms` design gate at the intended `5 Hz`;
- added RSS below `300 MiB`;
- incremental CMA below `96 MiB` target and `160 MiB` hard limit;
- no thermal throttling or unrecovered allocation.

The mapper estimate (`179.3 us`, `5577.43 FPS`) is not substituted for this
gate and must not be presented as X5 performance.

## 6. Read-Only Sensor Replay

Build an adapter node in the candidate namespace only:

```text
/x5_finals_vnext/*
```

Allowed inputs are LiDAR, depth, odometry and optional Vision-BEV diagnostics.
Allowed outputs are diagnostics/evidence under that namespace. Static and
runtime audits must prove zero publishers to motion topics, zero TF, zero
service/action authority, and zero serial/F407 access.

Validate:

- exact frame IDs, timestamps, rates and stale-data behavior;
- calibrated depth intrinsics/extrinsics and the common BEV geometry;
- Vision-BEV provenance distinguishes live/cached/fixture/missing;
- missing 4K semantics never blocks LiDAR/depth processing;
- candidate errors become `MONITOR_OFFLINE` only.

## 7. Shadow Observation

Run a bounded idle observation first, then a read-only replay/rolling
observation. Capture occupancy/flow, sensor reliability, CPU footprint scores,
ShadowGuard state, latency, resources and thermals. No result is sent to the
chassis.

Only after all previous gates pass may the frozen demonstration be run under a
separate explicit physical safety confirmation. The candidate remains an
observer and must not change the demonstrated motion.

## 8. Non-Interference Regression and Rollback

Stop the candidate process, verify no orphan remains, and rehash all frozen
artifacts. Re-run the validated demonstration only if the user requests it.
Rollback means leaving the new content-addressed directory stopped or removing
that directory; no frozen service needs restart and no firmware is changed.

Promotion is limited to **optional manual shadow observation**. It does not
promote the model to autonomous navigation or a safety-enforcement controller.
