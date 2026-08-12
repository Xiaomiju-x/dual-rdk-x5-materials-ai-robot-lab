# RTX 5090 cloud runbook

## Scope

Upload the complete `workstation/dual_arm_successor/` directory.  Do not upload
the frozen robot credentials, SSH keys, API keys, or any private network
configuration.  The cloud job is training-only and has no device endpoints.

Recommended host:

- Ubuntu 22.04
- RTX 5090 32 GB
- Python 3.10 or 3.11
- NVIDIA driver compatible with CUDA 12.8 PyTorch wheels
- at least 100 GB system disk and 200 GB data/work disk

## 1. Bootstrap

From any directory:

```bash
bash /absolute/path/dual_arm_successor/training/cloud5090/bootstrap_ubuntu.sh
```

This creates `training/cloud5090/.venv`.  It does not use `sudo`, `apt`, change
the NVIDIA driver, or modify system CUDA.

Install SmolVLA dependencies only after the real dataset gate is ready:

```bash
bash /absolute/path/dual_arm_successor/training/cloud5090/bootstrap_ubuntu.sh --with-smolvla
```

## 2. First paid-GPU check

Run the synthetic smoke first:

```bash
bash /absolute/path/dual_arm_successor/training/cloud5090/run_all.sh \
  --synthetic-smoke \
  --train-jsonl /tmp/not-used.jsonl \
  --readiness-report /tmp/not-used.json
```

Expected terminal marker:

```text
SYNTHETIC_SMOKE_DONE_NOT_REAL_POLICY
```

The generated checkpoint and ONNX file prove only the CUDA, model, export, hash
and packaging path.  Their receipt is permanently labelled
`SYNTHETIC_SMOKE_NOT_REAL_POLICY`.

## 3. Real Tiny-ACT and temporal world model

Do not run this stage until a real continuous dataset contains at least 30
whole physical episodes, 720 rows, synchronized state/action vectors and a
signed readiness report:

```bash
bash /absolute/path/dual_arm_successor/training/cloud5090/run_all.sh \
  --train-jsonl /data/xrd/train.jsonl \
  --readiness-report /data/xrd/readiness.json
```

The gate splits by parent episode, runs three seeds, records CUDA OOM retries,
and does not hide failures.  Training alone never grants deployment or motion
authority.

## 4. SmolVLA

SmolVLA is disabled by default.  Supply a previously downloaded local base
checkpoint and a local LeRobot dataset:

```bash
bash /absolute/path/dual_arm_successor/training/cloud5090/run_all.sh \
  --train-jsonl /data/xrd/train.jsonl \
  --readiness-report /data/xrd/readiness.json \
  --enable-smolvla \
  --smolvla-model-path /models/smolvla_base \
  --lerobot-dataset-root /data/xrd/lerobot_v3
```

The runner forces Hugging Face offline mode during training.  XR-0,
OpenVLA-OFT, XR-1 and Xiaomi-Robotics-U0 remain reference-only and are never
automatically downloaded.

## 5. Return artifacts

Copy the generated package from:

```text
training/cloud5090/outputs/packages/
```

Back to `workstation/dual_arm_successor/evidence/cloud5090/`.  Before any X5
conversion, verify:

1. archive and file SHA-256;
2. receipt status and dataset provenance;
3. whole-episode split with zero parent overlap;
4. no `motion_authority=true`;
5. no claim that synthetic smoke is a robot policy;
6. ONNX static shapes and opset 11;
7. Bayes-e checker/makertbin and actual X5 forward remain separately pending.
