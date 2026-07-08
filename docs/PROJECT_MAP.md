# Project Map

## AI Brain

The full local system uses a dashboard service plus four analysis lines: XRD vision, XRD numerical, PL vision, and PL numerical. The public bundle does not include the core prediction engine or model files. It includes only public-facing UI and report assets.

## Embodied Brain

The full system runs SLAM, LiDAR, depth camera ingestion, odometry, and Lab-FSD shadow planning on the vehicle-side RDK X5. The public bundle includes interface-level shadow planner code, not the full deployment scripts or live robot control endpoints.

## Workstation

The full workstation contains dual myCobot fixed stations and STM32F407-controlled end effectors. The public bundle includes non-core visualization, collision interlock, mock telemetry, and skill replay code; real motion scripts and private network configuration are excluded.

## Evidence Site

The public bundle includes static frontend assets and generated report figures so the review side can inspect UI complexity and evidence-chain design without receiving server credentials or private data.
