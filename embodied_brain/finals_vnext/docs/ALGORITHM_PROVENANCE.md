# Algorithm Provenance and Claim Boundary

## Public ideas adapted

| Candidate component | Public reference idea | Local adaptation |
|---|---|---|
| Metric navigation expert | EKF/UKF sensor fusion, pose-graph SLAM, MPPI trajectory scoring | Health/innovation gates, scan-overlap diagnostics and read-only reference metrics |
| Depth4D expert | Ray-based free/unknown space and temporal occupancy | Three height bands, temporal decay, connected tracks, closing rate and TTC in a 2-D edge BEV |
| Vision-FSD expert | BEVFormer temporal BEV and BEVFusion common-space fusion | Single 4K semantic bridge with strict provenance and compact semantic BEV; no multi-camera transformer |
| TinyOccFlow v2 | Occupancy Flow Fields joint future occupancy and motion | Three short horizons and 2-D flow in a 64 x 64 lab-scale BEV |
| ShadowGuard v2 | Planning-oriented prediction and uncertainty gating | CPU geometric footprint risk, conformal residual, OOD and cross-modal review states |

Primary references:

- Nav2 MPPI:
  https://docs.nav2.org/configuration/packages/configuring-mppic.html
- SLAM Toolbox:
  https://docs.ros.org/en/humble/p/slam_toolbox/
- robot_localization:
  https://docs.ros.org/en/rolling/p/robot_localization/
- BEVFormer:
  https://arxiv.org/abs/2203.17270
- BEVFusion:
  https://arxiv.org/abs/2205.13542
- UniAD:
  https://openaccess.thecvf.com/content/CVPR2023/html/Hu_Planning-Oriented_Autonomous_Driving_CVPR_2023_paper.html
- Occupancy Flow Fields:
  https://waymo.com/research/occupancy-flow-fields-for-motion-forecasting-in-autonomous-driving/

## Claims allowed now

- A three-expert LiDAR/depth/vision architecture exists on the laptop.
- Five-frame, 60-channel TinyOccFlow v2 was trained on RTX 4050.
- ONNX Runtime parity and held-out synthetic replay pass.
- OpenExplorer generated a Bayes-e artifact with all 34 compiled nodes on BPU.
- Frozen finals code and firmware hashes remain unchanged.
- The candidate is shadow-only and designed for X5 CPU/BPU heterogeneous use.

## Claims requiring an X5 board receipt

- the `.bin` actually loads and executes on the embodied X5;
- actual latency, BPU utilization, RAM/CMA, temperature and recovery;
- real LiDAR/depth/odometry/4K synchronization and output usefulness;
- non-interference while the frozen demonstration is running.

## Claims prohibited

- “Tesla FSD was reproduced” or Tesla proprietary code was used;
- end-to-end autonomous control or bottom-chassis takeover;
- compiler estimates are real X5 throughput;
- synthetic metrics prove real-world navigation accuracy;
- the v2 research model universally outperforms the accepted v5r1 baseline.

The accurate description is: **a Tesla/BEV/occupancy-flow-inspired,
multi-sensor shadow world model that augments a conventional metric navigation
stack without obtaining motion authority.**
