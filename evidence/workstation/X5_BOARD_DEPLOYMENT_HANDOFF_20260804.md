# DualArm ShadowVLA X5 board handoff

Date: 2026-08-04

Final status:
`X5_PASSIVE_FIXTURE_REPLAY_ACCEPTED_NOT_REAL_POLICY`.

## Scope

This is an isolated, one-shot replay on the AI brain X5.  It does not replace,
gate, wrap, or command the frozen `workstation/dual_arm/` demonstration.  The
arm Pi boards, cameras, serial ports, GPIO, robot SDK, and actuator endpoints
are absent from the candidate package.

All data remains `COMMAND_DERIVED_DIGITAL_TWIN`.  It is not measured robot
telemetry, synchronized camera/action data, a real policy, physical
generalization evidence, or a robot success-rate result.

## Immutable release

- Release: `dual-arm-shadow-x5-ebc63ae66127e6f5`
- Local archive:
  `workstation/dual_arm_successor/releases/dual-arm-shadow-x5-ebc63ae66127e6f5.tar.gz`
- Archive SHA-256:
  `68ff0bfc47038ce1770386bf8a51a103fd767e6d75f54d565f2e607ff704747e`
- Manifest SHA-256:
  `b2451f9e9f4fe138ed41540c6477e516c39ed1f5e79df8d435a48bc74b26335d`
- Board install root:
  `/home/rdk/dual_arm_shadow_finals/releases/dual-arm-shadow-x5-ebc63ae66127e6f5`
- Automatic start: false
- Service registration: false
- Production overwrite: false

The package contains two accepted RTX5090 teacher ONNX files, one 266,282-byte
Bayes-e student bin, two fixed replay fixtures, receipts, and the one-shot
runtime.  It contains no checkpoint training stack or robot endpoint.

## Actual X5 results

Board identity was `xrd-ai`, user `sunrise`, architecture `aarch64`, Python
3.10.12.  All fixed package hashes passed before execution.

### CPU teachers

| Model | Actual backend | Batch | Mean latency | PC reference max diff |
|---|---|---:|---:|---:|
| Tiny-ACT | `CPUExecutionProvider` | 24 | 72.9885 ms | `3.4571e-6` |
| Residual world model | `CPUExecutionProvider` | 24 | 16.7918 ms | `3.8147e-6` |

These are 24-sample batch timings, not per-sample real-time control latency.

### Bayes-e student

- Actual backend: `hrt_model_exec` on Bayes-e BPU.
- Validated output: `stage_logits` top-1 only.
- Fixed samples: 16.
- FP32/BPU top-1 agreement: `16/16`.
- Reported inference-only latency: mean `1.7955 ms`, median `1.516 ms`, minimum
  `1.367 ms`, maximum `5.297 ms`.
- OpenExplorer placed the temporal convolutions, global average pool, stage
  head, and other compiled heads on BPU.

`next_skill_logits`, sync, success, OOD, and action outputs are compiled but do
not have board semantic promotion.  The action head is explicitly rejected
because its fixture joint MAE (`1.6382 deg`) was worse than persistence
(`1.0812 deg`), and runtime does not consume it.

The earlier `pyeasy_dnn` NCHW adapter run achieved only `37.5%` agreement and
is retained under the candidate evidence as rejected.  A direct
`hrt_model_exec` probe on the same input matched FP32 top-1 and became the
authoritative adapter.

## Non-interference

Before and after the accepted run:

- `dashboard.py` SHA-256 remained
  `3c7ed0178e05a306f956e0d0ad0c5d903b6684a8e18ef24b134201613d05a262`.
- `start_x5.sh` SHA-256 remained
  `9b71d33ce92b22c5ec0d982d7532c301efef55815d4ab38a3de8753d2fa76a88`.
- `8888`, `8080`, `8081`, `5000`, and `5001` health GETs all returned 200.
- Production Dashboard, vision, and four llama-server process IDs remained
  unchanged.
- Candidate processes exited; no `x5_passive_replay`, `hrt_model_exec`, or
  `x5_biskill` process remained.
- Measured `CmaFree` was identical before and after the candidate worker, and
  BPU utilization returned to zero.

## Evidence

- Accepted receipt:
  `evidence/x5_board_live_20260804_164529/x5_passive_replay_receipt.json`
  SHA-256 `6aec17dd4a0f737b9a9fa1550d0166ea1b098f1834dcc4ebc3a367b7de1b3393`.
- BPU worker result:
  `evidence/x5_board_live_20260804_164529/bpu_worker_result.json`
  SHA-256 `ca7177aa51fc26740f7074e927cff914ef7700ae91451f3a7204a3215855778b`.
- Board manifest:
  `evidence/x5_board_live_20260804_164529/board_manifest.json`
  SHA-256 `b2451f9e9f4fe138ed41540c6477e516c39ed1f5e79df8d435a48bc74b26335d`.
- Rejected adapter receipt:
  `evidence/x5_board_candidate_v3_stage_skill_20260804/board_runs/run_20260804_164111/x5_passive_replay_receipt_rejected.json`.

## Manual replay

The candidate has no launcher and is not part of boot.  A deliberate one-shot
replay on AI X5 is:

```bash
python3 /home/rdk/dual_arm_shadow_finals/releases/dual-arm-shadow-x5-ebc63ae66127e6f5/x5_passive_replay.py \
  --root /home/rdk/dual_arm_shadow_finals/releases/dual-arm-shadow-x5-ebc63ae66127e6f5 \
  --out /home/rdk/dual_arm_shadow_finals/releases/dual-arm-shadow-x5-ebc63ae66127e6f5/x5_passive_replay_receipt.json
```

This command replays fixed tensors only.  It does not require the mechanical
arms or Pi boards and cannot move them.
