# Material Synthesis AI Prediction and Multi-Agent Embodied Lab Assistant

Dual RDK X5 embedded robot system for the D-Robotics embedded contest.

![System architecture](report_source/generated_figures/fig_xrd_architecture_html.png)

## Overview

This project builds an embedded laboratory assistant for near-infrared phosphor material research. The system connects material formula screening, XRD/PL evidence analysis, embodied execution, workstation manipulation, and public evidence display into one traceable engineering loop.

The submitted public repository is a reviewable boundary of the real project. It shows the system design, frontend evidence platform, interface schema, report assets, non-core workstation logic, edge-facing adapters, public screenshots, public crystal cache files, and offline demo sensor frames. It does not publish private data, model weights, credentials, deployment scripts, or the core material prediction engine.

## System Design

The complete system is organized as four cooperating layers:

| Layer | Main hardware | Responsibility |
| --- | --- | --- |
| AI brain | RDK X5, 4K camera, audio module | Material prediction, XRD/PL analysis, local LLM reasoning, evidence logging |
| Embodied brain | RDK X5, LD14 LiDAR, Astra depth camera, odometry | Verified pickup/lift/0.50 m odometry loop/release/reset, SLAM, and Lab-FSD shadow/assist |
| Workstation | Dual myCobot 280-Pi stations, independent cameras, custom gripper | arm01 visual redundancy and bag drop, arm02 concurrent four-cycle grinding |
| Execution layer | STM32F407, servo, electric push rod, electromagnet, stepper axis | Low-level timing, bottle pickup/release, safety-degraded actuator sequence |

As of July 20, 2026, the first three finals segments have been rehearsed on hardware and frozen. The embodied brain completed bottle pickup, lift, a 0.50 m odometry-closed straight drive, bottle release, and reset. The existing AI Dashboard, XRD vision, and material-synthesis prediction flow completed a tablet rehearsal. The dual-arm station completed arm01 visual redundancy and bag drop together with arm02 concurrent four-cycle grinding. Lab-FSD remains shadow/assist and has no chassis authority. X5 CPU/OpenCV is authoritative for bag state; BPU output is assist/evidence only. Neither the public portal nor this repository exposes hardware actuation.

## Key Capabilities

- Dual RDK X5 heterogeneous cooperation: one AI brain for research reasoning, one embodied brain for mobile perception and planning.
- Four AI analysis lines: vision, XRD numerical analysis, PL vision, and PL numerical analysis.
- Local embedded inference boundary: BPU lightweight models, CPU local LLM processes, and offline fallback paths.
- Verified embodied hardware loop: pickup, lift, 0.50 m odometry-closed drive, release, and reset.
- Lab-FSD shadow planner: FSD-style BEV occupancy reasoning for risk, trajectory, and safety-gate output without direct chassis takeover.
- Finals dual-arm collaboration: arm01 visual redundancy and bag drop with arm02 concurrent four-cycle grinding; CPU/OpenCV is authoritative and BPU remains assist-only.
- STM32F407 actuator sequence: servo, electric push rod, electromagnet, and stepper axis timing under safety-degraded demonstration rules.
- Public evidence platform: static frontend, OpenAPI-style schemas, report figures, rendered pages, screenshots, and reproducible review assets.

## Repository Map

| Path | Purpose |
| --- | --- |
| `public_site_static/` | Static frontend of the public evidence site, including PWA and 3D crystal display assets |
| `public_site_reports/` | Read-only public status report samples |
| `public_site_tools/` | Public 3D crystal asset generation scripts |
| `workstation_public/` | Non-core workstation interlock, mock telemetry, skill replay, and icon tooling |
| `workstation_frontend_public/` | Public workstation UI components, charts, 3D arm scenes, stores, and pages |
| `edge_public/` | Public interface stubs for Lab-FSD shadow planning, Fly-MB decision output, and F407 timing |
| `schemas/` | Read-only status API schema and example response |
| `report_source/` | TeX report source, HTML figure source, MATLAB/Python figure scripts, generated figures |
| `public_evidence_data/` | Public screenshots, rendered report pages, public crystal cache files, and offline demo sensor frames |
| `docs/` | Public project map and open boundary notes |

## Public Evidence

![Evidence site](report_source/generated_figures/fig_xrd_site_home.png)

The repository includes evidence assets that can be inspected without private credentials:

- 23 rendered report pages in `public_evidence_data/report_rendered_pages/`.
- Public evidence site screenshots in `public_evidence_data/site_screenshots/`.
- Public crystal structure cache files in `public_evidence_data/crystal_public_cache/`.
- Offline replay-style sensor frames in `public_evidence_data/demo_sensor_frames/`.
- Static report figures generated from HTML, MATLAB, and Python sources.

## How To Inspect

This public repository is designed for review rather than direct robot deployment.

1. Open `public_site_static/index.html` for the archived offline evidence-site snapshot; the current read-only portal is `https://xiaomiju.xyz`.
2. Read `report_source/main.tex` and `report_source/sections/` for the complete design report source.
3. Inspect `schemas/openapi_status_schema.json` and `schemas/status_snapshot_example.json` for the public status API shape.
4. Review `workstation_public/` and `edge_public/` for non-core logic and interface boundaries.
5. Use the rendered pages and screenshots under `public_evidence_data/` as offline evidence.

## NodeHub Information Draft

- Project name: Material Synthesis AI Prediction and Multi-Agent Embodied Lab Assistant
- Chinese project name: 基于双 RDK X5 异构协同的材料合成 AI 预测与多机具身实验助理机器人
- Repository: `https://github.com/Xiaomiju-x/xrd`
- Suggested tags: RDK X5, Embedded AI, Materials AI, Laboratory Robot, SLAM, BPU, STM32F407
- Suggested platform: RDK X5
- Suggested category: Robot application, Embodied AI, AI vision, Research automation
- Video: submit the Bilibili embed code after the contest video is published.

## Open Boundary And License

The public repository uses an Apache-2.0 open-source boundary for non-core review materials. Private laboratory assets remain excluded:

- API keys, cookies, SSH/WiFi/SSO configuration, account credentials, and private network addresses.
- GGUF, LoRA, BPU bin, tokenizer, tensor packages, and other model weights.
- Unpublished XRD/PL raw data, private experiment logs, failure-pattern libraries, and complete training sets.
- Core material prediction engine, private rules, deployment scripts, and executable real-hardware control scripts.

See `docs/PUBLIC_BOUNDARY.md` for the detailed publication boundary.
