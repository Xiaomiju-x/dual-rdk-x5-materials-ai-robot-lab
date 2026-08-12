# X5-TriBEV-Flow v2 + ShadowGuard v2

Status: **PC accepted; RDK X5 board validation pending.**

This is an independent, shadow-only research candidate that extends the
accepted `embodied_brain/finals_successor/` baseline. It does not replace or
modify the validated finals demonstration:

```text
bash ~/tools/finals_lift_nav_demo.sh
F407 build 2026071907
0.50 m odometry closed-loop drive
```

## What was added

### 1. Metric LiDAR expert

The LiDAR path keeps metric occupancy and visibility as the geometric anchor.
It adds innovation gates for odometry/IMU health, scan-overlap and degeneracy
diagnostics, pose-graph/loop-closure contracts, and read-only A/B metrics for
SLAM Toolbox, localization, and MPPI references.

### 2. Depth4D expert

Depth points are projected into the same `64 x 64`, `0.1 m/cell` robot-centric
BEV as LiDAR. Separate low/mid/high hit planes, ray-cleared free space,
explicit unknown space, temporal decay, connected-component tracks, closing
rate, and TTC preserve information that a single pseudo-laser scan discards.

### 3. Vision-FSD semantic expert

The AI-brain 4K bridge has an explicit provenance contract: live, cached,
fixture, stale, missing, and invalid inputs cannot be confused. Semantic risk
and visibility are encoded in a compact canonical payload, accumulated in
short-term static/dynamic memory, and checked for ghost risk. This borrows the
BEV and temporal-memory ideas used in modern autonomous-driving research; it
is not Tesla source code, a Tesla FSD replica, or an end-to-end controller.

### 4. BPU world model

Five temporal frames, each with 12 channels, form a fixed
`1 x 60 x 64 x 64` tensor. The 57,042-parameter TinyOccFlow v2 predicts:

- occupancy at `+0.4`, `+0.8`, and `+1.2 s`;
- 2-D occupancy flow;
- dynamic probability and model uncertainty;
- auxiliary risk for 15 fixed trajectory candidates;
- reliability for LiDAR geometry, depth geometry, vision semantics, and
  odometry alignment.

The ONNX graph uses only `Add`, `Concat`, `Constant`, `Conv`, `Relu`, and
`Resize`. OpenExplorer compiled 34 nodes to Bayes-e BPU and left zero compiled
nodes on CPU. The compiler estimate is not a board measurement.

### 5. CPU ShadowGuard

X5 CPU keeps deterministic responsibilities: sensor contracts, temporal
alignment, geometric footprint scoring, conformal/OOD calibration, reference
comparison, provenance, and evidence. The model has no motion authority.

## PC evidence

The formal PC run uses 576 synthetic episodes with whole-session
train/validation/calibration/test separation (`360/72/72/72`) and rejects
cross-session duplicate inputs. On the frozen 72-episode synthetic test split:

| Metric | Candidate | Baseline |
|---|---:|---:|
| Future occupancy IoU | 0.960557 | persistence 0.893269 |
| Flow mean EPE | 0.095469 m | zero-flow 0.291625 m |
| Flow p95 EPE | 0.309695 m | zero-flow 0.899910 m |
| Joint conformal coverage | 91.67% | 90% nominal |
| Sensor-reliability accuracy | 98.96% | synthetic labels |

These are synthetic-only development metrics. They do not establish real
corridor navigation accuracy, actual BPU latency, sustained TOPS utilization,
or autonomous driving.

Current immutable identities:

```text
checkpoint 652068b73698fadc1b32b22071296b8cc1839106a242b2ec397758c8d3a08e07
ONNX       eef470db65e00bc8ba1887d3ab80470bd7df01ce17175413c1e0a84f300a935e
Bayes-e    7c131aad52cec62c2e526d41f407bdeaf103ad1fd683abcf532deadeac87b29c
```

PC ONNX Runtime held-out replay processed all 72 episodes. Mean/p95 latency
was `1.787/2.673 ms` on the laptop CPU reference backend. This is not X5 BPU
performance.

## Non-interference

- no `/cmd_vel` or `/cmd_vel_safe`;
- no TF or `map -> odom`;
- no service or action authority;
- no serial or F407 access;
- no automatic service registration;
- a candidate failure only reports monitor offline;
- all 12 frozen demonstration artifacts are rehashed during acceptance.

## Reproduce PC acceptance

From the repository root:

```powershell
C:\Users\YOUR_USER\miniconda3\envs\xrd\python.exe `
  embodied_brain\finals_vnext\tools\replay_runtime_pc.py
C:\Users\YOUR_USER\miniconda3\envs\xrd\python.exe `
  embodied_brain\finals_vnext\tools\build_pc_acceptance.py
C:\Users\YOUR_USER\miniconda3\envs\xrd\python.exe `
  embodied_brain\finals_vnext\tools\package_candidate.py
```

Follow `docs/X5_BOARD_PHASE_RUNBOOK.md` only after the user explicitly powers
the embodied X5. Do not register the candidate as a service.

## Research provenance

The design adapts public ideas rather than claiming source-code equivalence:

- [Nav2 MPPI controller](https://docs.nav2.org/configuration/packages/configuring-mppic.html)
- [SLAM Toolbox lifelong mapping](https://docs.ros.org/en/humble/p/slam_toolbox/)
- [robot_localization state estimation](https://docs.ros.org/en/rolling/p/robot_localization/)
- [BEVFormer](https://arxiv.org/abs/2203.17270)
- [BEVFusion](https://arxiv.org/abs/2205.13542)
- [UniAD](https://openaccess.thecvf.com/content/CVPR2023/html/Hu_Planning-Oriented_Autonomous_Driving_CVPR_2023_paper.html)
- [Waymo Occupancy Flow Fields](https://waymo.com/research/occupancy-flow-fields-for-motion-forecasting-in-autonomous-driving/)

Tesla markets its current product as supervised rather than fully autonomous.
Our candidate is more limited: an occupancy/flow shadow monitor around an
existing LiDAR/SLAM/Nav2 stack.
