# DualArm-ShadowVLA

`DualArm-ShadowVLA` is the isolated finals research candidate for the XRD
dual-arm workstation.  The semifinal/finals demonstration under
`workstation/dual_arm/` remains the only motion authority and is not imported,
wrapped, patched, or replaced by this candidate.

The candidate adds three read-only layers:

1. A deterministic dual-arm skill graph and temporal consistency monitor.
2. Tiny-ACT and action-conditioned world-model experiments for offline replay.
3. A VLA teacher path for RTX 5090, followed by a small X5 CPU/BPU student.

Every runtime receipt must retain:

```json
{
  "motion_authority": false,
  "execution_allowed": false,
  "actuator_commands_issued": 0
}
```

The initial maturity level is `OFFLINE_REPLAY`.  No process in this directory
may open a camera, serial port, GPIO, SSH connection, or robot SDK.

## Current evidence boundary

- The authoritative physical run is
  `workstation/dual_arm/evidence/finals_part3_execute_20260720_052630_4956`.
- It proves the fixed skill chain, AprilTag id 2, CPU bag-presence gate,
  auxiliary BPU forwards, four grinding cycles, and both arms returning START.
- It does not contain a continuous synchronized 13-dimensional joint/action
  trajectory and therefore cannot support a real learned-control claim.
- The separate command-derived fixture contains 60 digital-twin episodes and
  30,771 continuous 13-dimensional rows for the same two frozen actions.  It is
  permanently marked `FIXTURE_REPLAY_NOT_REAL_POLICY` and does not reinterpret
  command interpolation as measured telemetry.
- Synthetic training proves software plumbing only.  It is not a real robot
  policy and must carry `SYNTHETIC_SMOKE_NOT_REAL_POLICY`.

## Compute plan

Local development is CPU-only because the laptop RTX 4050 is reserved by
another project.  The cloud RTX 5090 bundle performs all CUDA training,
including compact baselines and optional SmolVLA fine-tuning.  Full VLA models
remain offline teachers; only distilled fixed-shape students are candidates for
Bayes-e conversion after separate X5 board validation.

## Current acceptance

PC status: `PC_FOUNDATION_ACCEPTED`.

Cloud status:
`RTX5090_FIXTURE_REPLAY_ACCEPTED_NOT_REAL_POLICY`.

X5 status:
`X5_PASSIVE_FIXTURE_REPLAY_ACCEPTED_NOT_REAL_POLICY`.

- Frozen v3 baseline: 14/14 files match the recorded July 22 hotfix hashes.
- Candidate tests: 30 passed, 1 skipped.  The skipped forward-shape test needs
  PyTorch, which is intentionally absent from the CPU-only local environment.
- JSON contracts: Draft 2020-12 validation passed.
- Source audit: no camera, serial, GPIO, SSH, robot SDK, or motion authority.
- Historical replay: 10 authoritative stages, `STAGE_ONLY`.
- Local CUDA use: none.  The RTX 4050 was not queried or used.
- RTX 5090 synthetic smoke: passed on a real RTX 5090 with
  `PyTorch 2.8.0+cu128`.
- RTX 5090 fixture training: Tiny-ACT and a residual temporal world model each
  passed three whole-episode split seeds and ONNX checker validation.
- Tiny-ACT mean joint MAE was `0.859215 deg`, a `22.01%` improvement over its
  persistence baseline on command-derived holdouts.
- The residual world model mean joint MAE was `0.204520 deg`, a `30.69%`
  improvement over persistence; mean stage accuracy was `77.60%` versus a
  `9.01%` majority baseline.
- Real-episode Tiny-ACT/world-model/SmolVLA training: not started because the
  existing `STAGE_ONLY` evidence correctly fails the continuous-data gate.
- The two accepted RTX5090 teachers ran on the real AI X5 CPU with reference
  maximum absolute differences below `3.82e-6`.
- The compact `X5BiSkillTCN` stage head was compiled with OpenExplorer 1.2.8
  and ran on the real Bayes-e BPU through `hrt_model_exec`.  Its 16 fixed
  board samples matched FP32 top-1 `16/16`; reported inference-only latency was
  `1.7955 ms` mean and `1.516 ms` median.
- The student's next-skill/sync/other heads are compiled but not semantically
  validated on board.  Its action head was rejected because it did not beat
  the persistence baseline and is not consumed by runtime.
- The first `pyeasy_dnn` NCHW feature-map attempt is retained as rejected
  evidence; the authoritative adapter for this candidate is
  `hrt_model_exec`.

The detailed implementation and truthfulness ledger is
[`docs/IMPLEMENTATION_LEDGER.md`](docs/IMPLEMENTATION_LEDGER.md).

## Reproduce the PC acceptance

From any PowerShell directory:

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
& 'C:\Users\YOUR_USER\Desktop\xrd_backup\workstation\dual_arm_successor\.venv-cpu\Scripts\python.exe' `
  'C:\Users\YOUR_USER\Desktop\xrd_backup\workstation\dual_arm_successor\tools\run_pc_acceptance.py'
```

Expected final status:

```text
PC_FOUNDATION_ACCEPTED
```

## RTX 5090 handoff

The first synthetic smoke remains under `evidence/cloud5090/`.  The rejected
absolute-state fixture experiment remains under `evidence/cloud5090_fixture_v1/`.
The accepted residual experiment and all six model receipts are under
`evidence/cloud5090_fixture_v2_residual/`.

The source package remains training source only, not an X5 deployment bundle.
The current passive algorithm milestone is complete and the RTX 5090 can be
shut down.  Real robot data is required only before making a real-policy or
physical-generalization claim.  The authoritative handoff is
[`docs/FIXTURE_REPLAY_5090_HANDOFF_20260730.md`](docs/FIXTURE_REPLAY_5090_HANDOFF_20260730.md).

The X5 board handoff, immutable release hashes, runtime boundary, and board
receipt are in
[`docs/X5_BOARD_DEPLOYMENT_HANDOFF_20260804.md`](docs/X5_BOARD_DEPLOYMENT_HANDOFF_20260804.md).
