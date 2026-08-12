# DualArm fixture-replay RTX 5090 handoff

Date: 2026-07-30

Final status:
`RTX5090_FIXTURE_REPLAY_ACCEPTED_NOT_REAL_POLICY`.

## Decision

The finals demonstration remains exactly the two already verified physical
flows:

1. arm01 `START -> OBSERVE -> START`;
2. bag transfer plus overlapping arm01 return and arm02 four-cycle grinding,
   ending with both arms at `START`.

No new robot capture is required for the current algorithm-showcase milestone.
Instead, the frozen command sources are parsed without importing robot code and
expanded into a bounded digital-twin fixture. This permits an honest offline
Tiny-ACT and temporal-world-model result while the frozen v3 controller remains
the sole motion authority.

Real synchronized robot data is still mandatory before claiming a learned
real-robot policy, physical generalization, or learned control authority.

## Dataset

- Status: `FIXTURE_REPLAY_NOT_REAL_POLICY`
- Provenance: `COMMAND_DERIVED_DIGITAL_TWIN`
- Episodes: 60 total, 30 per frozen task
- Rows: 30,771 at 20 Hz
- State/action dimension: 13
- Dataset SHA-256:
  `b5f91630ef924d6421176a5ffb924854857ec6f67b3fadea6e010a7d95480fa5`
- Measured robot telemetry: false
- Synchronized camera/action samples: false
- Motion authority: false
- Actuator commands issued: 0

The real-data gate was not weakened. The same fixture data is rejected by the
existing real-episode preflight.

## RTX 5090 selection

Training used a real NVIDIA GeForce RTX 5090 with provider PyTorch
`2.8.0+cu128`. Each model family used three whole-episode split seeds.

### Tiny-ACT

- Parameters: 1,455,848
- Mean joint MAE: `0.859215 deg`
- Persistence mean joint MAE: `1.100007 deg`
- Mean improvement over persistence: `22.01%`
- Best seed: `20260732`
- Best checkpoint SHA-256:
  `d4190f1cdf5e3e8c1b09c49145816128c21398b87c9a4f8f18937d0cb0e321fe`
- Best ONNX SHA-256:
  `8c6599ac9ca5ea352a6f29ae82a214c7f620c3f363aeeb5359597e201c785c62`

### Residual temporal world model

- Parameters: 395,179
- Mean joint MAE: `0.204520 deg`
- Persistence mean joint MAE: `0.294735 deg`
- Mean improvement over persistence: `30.69%`
- Mean frozen-stage accuracy: `77.60%`
- Majority-stage baseline: `9.01%`
- Best seed: `20260732`
- Best checkpoint SHA-256:
  `c1f857ce018c59cf2c831b319f0beb2c6d2742c6c6eab0200f9ca135363915e1`
- Best ONNX SHA-256:
  `99aa6ca4b134606e0b4c16fddb8957bd05860922df7be0119a7815936ebebe66`

These are fixture-replay holdout metrics, not physical accuracy or task success
rates.

## Model-selection audit

The first absolute-state world model did not beat persistence and is retained
under `evidence/cloud5090_fixture_v1/` as rejected evidence. It was not hidden
or promoted. The residual v2 starts from persistence and learns a correction;
all three v2 seeds beat their persistence baselines.

## Immutable artifacts

- Accepted training source:
  `releases/dualarm-shadowvla-rtx5090-80da952901de62d0.zip`
- Source archive SHA-256:
  `a91ab0d17225f00e2fab6676a4d14a0ba512bfab390f79dbc232c61a9e3077aa`
- Source manifest SHA-256:
  `80da952901de62d0981a44f0cec325998867b8d2d760ff8159d4f244abedbe90`
- Accepted result archive:
  `evidence/cloud5090_fixture_v2_residual/xrd-cloud5090-results-03f909ed69ed336c.tar.gz`
- Result archive SHA-256:
  `623bb5b510e2676ba35e0d5d8d9b3e6ff4fda5c062885dcaf6b183068ec27b14`
- Aggregate SHA-256:
  `f3fb541bc0d30587108ca855200f7e1037d1fe4d8b23554ad0fde72bc2e6f72c`
- Local verification SHA-256:
  `d4424a109fb7b96952528ef7dbb9a71127985ea8c0ada33909836dbff2319b87`

The local verifier checked 22 packaged files, six checkpoints/ONNX receipts,
all ONNX checker states, the cloud GPU identity, and every recorded hash.

## Hardware and next action

- RTX 5090: no longer required for this milestone and may be shut down.
- Mechanical arms: no power-up or new motion is required.
- AI/embodied X5 boards: no power-up is required for this offline milestone.
- Bayes-e conversion and real X5 forward validation remain optional future
  work. They are required only before saying the candidate runs on X5.
- Real robot capture remains optional research work and is required only before
  saying the learned model controls or generalizes on the physical arms.

The frozen demonstration and its launchers remain unchanged.
