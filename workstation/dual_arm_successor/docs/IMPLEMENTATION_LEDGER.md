# DualArm-ShadowVLA implementation ledger

Date: 2026-07-30

Overall status:
`X5_PASSIVE_FIXTURE_REPLAY_ACCEPTED_NOT_REAL_POLICY`.

## Decision

The national-finals algorithm upgrade is implemented as an isolated,
zero-authority research candidate.  The proven semifinal/finals v3 program
under `workstation/dual_arm/` remains the sole motion authority and the visible
demonstration remains unchanged.

The laptop RTX 4050 is reserved by another project.  All local work for this
candidate is CPU-only.  The first CUDA stage was executed on a rented RTX 5090.

## Implemented

1. Frozen baseline protection
   - Recorded and verified 14 current finals files, including the July 22
     gripper hotfix state.
   - Created a payload backup and ZIP archive.
   - Added a repeatable hash verifier.

2. Evidence and replay
   - Adapted the authoritative `CLOSED_LOOP_DONE` result and two visual receipts
     into a 10-stage deterministic episode.
   - Indexed 20 historical composed results for stage anomaly and hard-case
     analysis.
   - Preserved the limitation that no continuous synchronized 13-dimensional
     state/action trajectory is available.
   - Added a separate command-derived digital-twin fixture for the exact two
     frozen demonstrations: 60 episodes, 30,771 rows, 13-dimensional
     state/action vectors, and permanent
     `FIXTURE_REPLAY_NOT_REAL_POLICY` provenance.
   - Kept the real-data gate unchanged; it rejects the fixture as a physical
     training dataset.

3. Candidate algorithm stack
   - Deterministic dual-arm skill graph and temporal consistency monitor.
   - Tiny-ACT action chunking experiment.
   - Residual action-conditioned temporal world-model experiment.
   - `X5-BiSkill TCN` static-shape compact student skeleton with stage,
     next-skill, synchronization, success, OOD, and `8 x 13` action-chunk
     outputs.
   - Optional SmolVLA offline teacher stage.
   - OpenVLA-OFT, XR-0, XR-1, and XR-U0 are reference-only and are never
     downloaded automatically.

4. RTX 5090 training chain
   - Ubuntu bootstrap without `sudo`, driver changes, or system CUDA changes.
   - RTX 5090 identity and CUDA preflight.
   - Synthetic CUDA/model/ONNX/package smoke with permanent
     `SYNTHETIC_SMOKE_NOT_REAL_POLICY` labeling.
   - Real-data gate requiring at least 30 whole physical episodes, 720 rows,
     synchronized state/action vectors, and a signed readiness report.
   - Whole-episode splits, three seeds, OOM retry reporting, immutable receipts,
     and result packaging.
   - Independent fixture preflight, persistence shortcuts, multi-seed
     aggregation, ONNX checker validation, and downloaded-result revalidation.
   - SmolVLA remains explicit opt-in with local checkpoint and local LeRobot
     dataset paths.

5. Safety and truthfulness contracts
   - Draft 2020-12 JSON schemas for episodes, predictions, and model receipts.
   - Source audit for camera, serial, GPIO, SSH, robot SDK, and actuator paths.
   - Required tuple on all candidate receipts:

```json
{
  "motion_authority": false,
  "execution_allowed": false,
  "actuator_commands_issued": 0
}
```

## Verified on the PC

- Acceptance state: `PC_FOUNDATION_ACCEPTED`.
- Tests: 31 collected, 30 passed, 1 skipped.
- The skip is the PyTorch forward-shape test; PyTorch is intentionally not
  installed in the isolated CPU environment.
- Contract validation: pass.
- Frozen baseline verification: 14/14 match.
- No-motion source audit: pass, zero findings.
- Python compileall: pass.
- Bash syntax for bootstrap and runner: pass.
- Local mode: `CUDA_VISIBLE_DEVICES=-1`.
- Local RTX 4050 used: false.

Authoritative receipts:

- `evidence/pc_acceptance.v1.json`
- `evidence/frozen_baseline_receipt_v1.json`
- `evidence/no_motion_authority_audit_v1.json`
- `evidence/authoritative_stage_dataset_v2/manifest.json`

Frozen backup archive SHA-256:

```text
2dc10a80765d5fa35bc6153f7d3b55ea42d541a931474392f8bb692d942d0019
```

## Cloud handoff

The immutable training-source archive, manifest, and receipt are generated
under `releases/`.  The newest receipt is authoritative; it records the exact
archive SHA-256 and the manifest SHA-256.  The archive embeds the same manifest
and must be verified before execution.

The first paid-GPU action was completed on 2026-07-30:

```bash
bash /absolute/path/dual_arm_successor/training/cloud5090/bootstrap_ubuntu.sh
bash /absolute/path/dual_arm_successor/training/cloud5090/run_all.sh \
  --synthetic-smoke \
  --train-jsonl /tmp/not-used.jsonl \
  --readiness-report /tmp/not-used.json
```

Expected marker:

```text
SYNTHETIC_SMOKE_DONE_NOT_REAL_POLICY
```

Actual verified cloud facts:

- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB, compute capability 12.0.
- Driver: `595.71.05`.
- Runtime: provider PyTorch `2.8.0+cu128`, inherited by an isolated venv.
- Source archive SHA-256:
  `8df82c352481fb86fc1f14d186293503fd6ac785faebf61aa6b9f15b3d9df4d6`.
- Synthetic run: `run_20260730T133904Z_2789`.
- Loss: `5.4914302826 -> 5.1146554947` over 12 steps, batch 32.
- Checkpoint SHA-256:
  `924aa77fd6818cd2cfefe88b20cb951a9f4faf084719a1ae853b88b40693484e`.
- ONNX SHA-256:
  `789e1d1a3db8a2a53395b0b9b8336cb925ffb1abb2c4fa1f6d823c8ea968bb77`.
- ONNX checker passed with opset 11, static input `[1,48,16,1]`, and
  action-chunk output `[1,8,13]`.
- Returned archive SHA-256:
  `5d392f819a68f92f9a64c4179afda96583f3184b7650aadebf1de65f23fb82ea`.
- Six returned artifacts and the normalized embedded manifest passed local
  revalidation with zero mismatches.
- The provider's preinstalled CUDA PyTorch avoided wasting rental time on a
  slow duplicate wheel download.  No driver or system CUDA was changed.

Cloud receipts are stored under `evidence/cloud5090/`; the consolidated record
is `evidence/cloud5090/verification_v1.json`.

The existing 10-stage physical record was also submitted to the real-data gate.
It was correctly rejected because it has no continuous synchronized
observation/action rows, too few parent episodes, and no signed readiness
record.  Therefore no real Tiny-ACT, world-model, or SmolVLA training started.

## Accepted fixture-replay training

The user selected an offline algorithm-stack upgrade that leaves the two
physical demonstrations unchanged and does not require a new robot dataset.
The frozen command sources were parsed without importing robot code and expanded
into 30 episodes per task at 20 Hz.

The first absolute-state world model did not beat persistence.  Its complete
result remains in `evidence/cloud5090_fixture_v1/` as rejected evidence.  A
residual world model was then trained from a new immutable source package and
all three seeds beat persistence.

Accepted v2 results:

- Dataset SHA-256:
  `b5f91630ef924d6421176a5ffb924854857ec6f67b3fadea6e010a7d95480fa5`.
- Tiny-ACT: 1,455,848 parameters; mean joint MAE `0.859215 deg`; mean
  persistence improvement `22.01%`; best seed `20260732`.
- Residual world model: 395,179 parameters; mean joint MAE `0.204520 deg`;
  mean persistence improvement `30.69%`; mean stage accuracy `77.60%` versus
  `9.01%` majority baseline; best seed `20260732`.
- Source archive SHA-256:
  `a91ab0d17225f00e2fab6676a4d14a0ba512bfab390f79dbc232c61a9e3077aa`.
- Result archive SHA-256:
  `623bb5b510e2676ba35e0d5d8d9b3e6ff4fda5c062885dcaf6b183068ec27b14`.
- Local verification status:
  `RTX5090_FIXTURE_REPLAY_ACCEPTED_NOT_REAL_POLICY`.

All metrics are command-derived holdout metrics, not physical accuracy or
success rates.  The full handoff is
`docs/FIXTURE_REPLAY_5090_HANDOFF_20260730.md`.

## X5 passive board deployment

On 2026-08-04 the accepted fixture teachers and a compact stage/skill student
were packaged without modifying the frozen demonstration.  The AI brain X5
was the only execution target; neither arm Pi, robot endpoint, camera, serial,
GPIO, nor motion SDK was used.

- Board identity: `xrd-ai`, user `sunrise`, `aarch64`, Ubuntu 22.04.
- Tiny-ACT actual backend: X5 CPU `CPUExecutionProvider`; 24-sample batch mean
  `72.9885 ms`; PC-reference max absolute difference `3.46e-6`.
- Residual world model actual backend: X5 CPU `CPUExecutionProvider`;
  24-sample batch mean `16.7918 ms`; max difference `3.82e-6`.
- `X5BiSkillTCN` stage head actual backend: Bayes-e BPU via
  `hrt_model_exec`; fixed board fixture top-1 agreement `16/16`; reported
  inference-only mean `1.7955 ms`, median `1.516 ms`.
- PC fixture metrics for the student were stage `97.85%`, next-skill `95.94%`,
  and sync `99.87%` over 4,705 holdout windows.  These are not real-robot
  metrics.
- The action head was rejected: `1.6382 deg` versus persistence
  `1.0812 deg`.  Runtime does not consume it.
- The first `pyeasy_dnn` adapter attempt produced only `37.5%` FP32 top-1
  agreement and remains rejected evidence.  Direct `hrt_model_exec` was then
  verified against the same immutable inputs.
- Candidate exit restored `CmaFree` exactly for the measured run; BPU ratio
  returned to zero.  Five production endpoints remained HTTP 200, production
  process IDs remained unchanged, and `dashboard.py`/`start_x5.sh` hashes were
  unchanged.

Authoritative board handoff:
`docs/X5_BOARD_DEPLOYMENT_HANDOFF_20260804.md`.

The RTX 5090 is no longer required.  AI X5 power is required only to rerun the
passive board receipt.  Mechanical arms and both Pi boards are not required
for fixture replay.

## Optional future gates

1. Collect a separately authorized continuous read-only robot dataset only if a
   future real-policy claim is desired.
2. Pass the signed real-data readiness gate before any real Tiny-ACT/world-model or
   SmolVLA training.
3. Compare seeds and baselines on whole-episode holdouts.
4. Export only a selected compact fixed-shape student.
5. Collect a larger preregistered board fixture only if next-skill/sync heads
   need separate actual-BPU semantic promotion.
6. Keep runtime at `POST_RUN_REPLAY` unless a later explicit authorization and
   full acceptance promote it.  Even promotion does not grant motion authority.

## Claims that are not currently allowed

- The VLA controls either arm.
- SmolVLA, XR-0, XR-1, XR-U0, or OpenVLA-OFT runs on an X5.
- A real policy was trained from the existing stage-only evidence.
- A real robot policy has been trained on an RTX 5090.  Only the explicitly
  labeled synthetic and fixture-replay candidates have been trained there.
- The BPU student controls either arm or has real-robot task accuracy.
- The compiled next-skill, sync, success, OOD, or action outputs passed actual
  BPU semantic validation; only stage top-1 did.
- Any candidate process can change, block, or replace the frozen v3 motion.
