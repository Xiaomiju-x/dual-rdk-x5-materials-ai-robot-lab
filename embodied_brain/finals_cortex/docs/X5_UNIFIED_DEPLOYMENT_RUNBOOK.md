# X5-Embodied Cortex Board Runbook

Status: **2026-08-04 inventory and shared v5r1 BPU gate observed; integrated
Cortex gate remains `MONITOR_OFFLINE`.**

The boards were powered and the v5r1 BPU primitive passed its isolated fixed
contract, but no Cortex candidate ROS graph or real synchronized session was
started. `/imu` and `/odom` were unavailable in the observation window, and
the kernel exposed no readable ION allocation node. These missing facts must
not be substituted with CMA or compiler estimates.

This is a board-validation runbook for an optional manual shadow candidate.
It does not modify the frozen finals demonstration, F407 firmware, ROS 2
navigation configuration, Wi-Fi, VPN, TUN, routes, ARP or existing services.

## 1. Preconditions

- User confirms both X5 boards are powered and the PC is on the same current
  LAN defined by the repository root `AGENTS.md`.
- Use only fixed identities after verifying host key and hostname:
  embodied X5 `rdk@192.0.2.85`, hostname `embodied-x5`; AI X5
  `rdk@192.0.2.103`, hostname `xrd-ai`.
- Do not discover or scan the network. Do not modify PC networking.
- Vehicle and lift stay idle. Initial inventory and BPU profiling require no
  physical motion.
- Never register this candidate as a boot service.

## 2. Revalidate The Laptop Artifact

Run the commands in the root README. Required:

- all candidate tests pass;
- Ruff and compileall pass;
- frozen manifest remains unchanged;
- PC receipt is valid;
- archive manifest verifies;
- package kind remains `PC_TOOLING_NOT_X5_DEPLOY`.

Any failure stops promotion.

## 3. Read-Only Identity And Runtime Inventory

Collect separately from both boards:

- hostname, OS image, kernel and architecture;
- TROS/ROS 2, `hobot_dnn`, `hbm_runtime`, `hrt_model_exec` and BPU package
  versions;
- total/available RAM, process PSS, CMA/ION, BPU status, temperature and
  throttling state;
- existing process and service identities;
- topic names, types, frame IDs and rates, without publishing or restarting.

Feed the collected JSON into `tools/board_receipt.py`. Missing, fixture or
non-board values must produce `NO_GO` and `MONITOR_OFFLINE`.

## 4. Content-addressed Transfer

Only after compatibility is known, construct a separate deploy archive and
copy it into a new immutable directory:

```text
~/xrd_candidates/finals_cortex/<content_sha256>/
```

Do not write into `~/tools`, `~/ros2_ws`, system Python, F407 files or any
existing candidate directory. Verify archive SHA-256 before and after transfer.

## 5. Embodied X5 BPU Gate

Validate the already accepted v5r1 model first; the richer vNext or Cortex
student does not replace it automatically.

With the chassis idle:

1. load one reviewed Bayes-e model on one BPU core;
2. compare fixed tensor outputs with PC references;
3. collect at least 200 actual latency samples;
4. record p50/p95/p99, BPU backend, PSS, MemAvailable, CMA/ION, temperature and
   throttling;
5. run 30 load/unload cycles and verify recovery within the budget;
6. stop and unload the candidate.

Design gates are in `contracts/runtime_budget.v1.json`; they are not current
measurements. Any gate failure leaves the runtime `MONITOR_OFFLINE`.

## 6. Real Sensor Recording

After exact topics and frames are known, add a candidate adapter with
subscriptions only. Record synchronized LiDAR, depth, odometry and Vision-BEV
metadata with:

- sequence and source timestamps;
- clock domain and measured clock offset;
- calibration/intrinsics/extrinsics identity;
- live/cached/replay/missing provenance;
- payload and session SHA-256.

Raw 4K video is not streamed to the embodied X5. The AI X5 emits compact
Vision-BEV diagnostics using newest-only queue depth one. A stale or missing
4K branch never blocks LiDAR/depth observation.

## 7. Offline Training And Whole-session Split

- Split by complete capture session, never by individual frame.
- Train CrossBEV and NavTeacher candidates on the PC/RTX 4050 first.
- Keep human/instrument ground truth, teacher pseudo-labels and synthetic
  oracles distinct.
- Require improvement over persistence, zero-flow and always-straight
  shortcuts.
- Freeze calibration and test sets before model selection.
- Export ONNX, perform quantization differential tests, then repeat the real
  board gate for any new `.bin`.

No on-board or online weight update is permitted.

## 8. Bounded Shadow Observation

The only allowed outputs are diagnostics under a dedicated candidate
namespace. Runtime audit must show:

- zero `/cmd_vel` or `/cmd_vel_safe`;
- zero TF or map-to-odom publication;
- zero service/action authority;
- zero serial/F407 access;
- no frozen process restart or configuration write.

Candidate errors only stop the observer. They never inhibit or alter the
validated demonstration.

## 9. Camera Handoff

The physical IMX415 belongs to AI X5. Cortex may consume compact results only.
Before the AI-brain demonstration, stop the optional Cortex observation,
verify its camera lease is released and confirm Dashboard camera mode is idle.
No force-kill or lock-file deletion is used as a normal handoff.

## 10. Final Non-interference Regression

After stopping the candidate:

1. verify no candidate process remains;
2. rehash the frozen 12-file manifest;
3. compare frozen topic rates and resource state;
4. only under a new explicit physical safety confirmation, run the existing
   `bash ~/tools/finals_lift_nav_demo.sh`;
5. record the 0.50 m demonstration result without changing its code.

Promotion is limited to **manual optional shadow observation**. Rollback means
stopping the candidate and leaving or removing only its content-addressed
directory; no frozen service or firmware needs modification.
