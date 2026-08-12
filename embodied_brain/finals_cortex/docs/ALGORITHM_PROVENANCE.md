# Algorithm Provenance And Claim Boundary

This document records which public research ideas shaped `X5-Embodied Cortex`
and what the local implementation actually does. References are architectural
provenance, not claims of equivalent scale, training data or performance.

## Adapted Public Ideas

| Local component | Public idea | Local adaptation |
|---|---|---|
| CrossBEV | BEVFormer temporal BEV and BEVFusion common-space fusion | A compact temporal monocular student distills seven separately auditable indoor BEV maps; LiDAR/depth remain independent experts |
| Tiny occupancy/flow line | Waymo Occupancy Flow Fields and indoor occupancy world models | Short-horizon occupancy, dynamic and flow evidence at laboratory scale; no driving-dataset equivalence |
| NavTeacher-15 | UniAD planning-oriented task coupling and Nav2 MPPI trajectory critics | Fifteen fixed proposals including stop/hold, scored by seven explicit costs; outputs never reach the chassis |
| Skill graph | Skill-centric hierarchical planning and scene-graph verification | Fixed finals workflow with preconditions, effects, deadlines and evidence domains |
| Episodic memory | Sparse spatial/episodic/semantic embodied memory | SQLite graph with provenance, TTL, hard-case selection and immutable hash chaining |
| TrustLab | Calibration, conformal prediction, OOD and abstention | Frozen plus adaptive residual tracks, risk-coverage, robust Mahalanobis, CUSUM and cross-modal disagreement; diagnostic states only |

Primary references:

- [BEVFormer, ECCV 2022](https://arxiv.org/abs/2203.17270)
- [BEVFusion, ICRA 2023](https://arxiv.org/abs/2205.13542)
- [UniAD: Planning-oriented Autonomous Driving, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Hu_Planning-Oriented_Autonomous_Driving_CVPR_2023_paper.html)
- [Occupancy Flow Fields, RA-L](https://waymo.com/research/occupancy-flow-fields-for-motion-forecasting-in-autonomous-driving/)
- [Nav2 MPPI controller documentation](https://docs.nav2.org/configuration/packages/configuring-mppic.html)
- [RoboMatrix skill-centric hierarchy](https://arxiv.org/abs/2412.00171)
- [VeriGraph execution-verifiable scene graphs](https://arxiv.org/abs/2411.10446)
- [Conformal Prediction for Semantically-Aware Autonomous Perception, CoRL 2024 proceedings](https://proceedings.mlr.press/v270/doula25a.html)

## Local Innovations

### 1. Seven-map CrossBEV instead of one fused heatmap

Obstacle, traversability, semantic restriction, dynamic occupancy, visibility,
unknown space and confidence remain separate outputs. This prevents a visually
plausible semantic layer from silently overwriting metric free-space evidence.
Every temporal input carries capture source, timestamp, calibration identity
and freshness state.

### 2. Explicit trajectory teaching without control takeover

`NavTeacher-15` turns the existing navigation stack into an offline teacher.
It evaluates fifteen kinematically fixed alternatives using obstacle,
traversability, semantic, dynamic, unknown, progress and smoothness costs.
Rank, regret, top-k agreement and the always-straight shortcut are measured.
The learned line is rejected if it cannot beat that shortcut.

### 3. Control-state and physical-state separation

The skill graph can prove that command/effect evidence followed the expected
order. It refuses to call pickup or placement physically successful without
independent object, load or limit evidence. This directly prevents serial
acknowledgements from being presented as proof of the real mechanical outcome.

### 4. Hard-case memory with provenance

The memory layer stores sparse area, workstation, object and task-event nodes,
not raw unbounded video. OOD, cross-modal disagreement, guard transitions and
staleness create deduplicated hard cases with a SHA-256 chain. They are review
and future offline-training candidates only; no live weight update exists.

### 5. Dual-track trust evidence

A frozen conformal track preserves the accepted calibration reference while an
adaptive window diagnoses recent drift. Calibration, risk-coverage, OOD,
time/calibration drift and cross-modal disagreement are composed into only
three states: `PASSIVE_OK`, `REVIEW`, or `MONITOR_OFFLINE`. None blocks or
changes the frozen finals motion.

## Current Claims

Allowed now:

- the PC implementation and integrated synthetic replay exist;
- source/provenance, integrity, non-interference and fail-closed tests pass;
- CrossBEV, NavTeacher, skill graph, memory and TrustLab contracts are
  implemented;
- the frozen finals files remain byte-for-byte unchanged.

Pending real session and X5 receipts:

- actual BPU backend, output parity, latency, RAM/CMA, temperature and recovery;
- measured LiDAR/depth/odometry/4K synchronization and calibration;
- real-data model quality, risk-coverage and hard-case utility;
- bounded concurrent shadow observation and frozen-demo non-interference.

Prohibited:

- claiming Tesla FSD was reproduced or proprietary Tesla assets were used;
- claiming the candidate controls the chassis or is an end-to-end autonomous
  driver;
- presenting synthetic fixture metrics or compiler estimates as X5 results;
- presenting a serial command acknowledgement as physical pickup success;
- presenting a conformal diagnostic as functional-safety certification.
