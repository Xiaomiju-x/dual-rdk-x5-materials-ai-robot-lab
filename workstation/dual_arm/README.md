# Dual-Arm Finals Fixed-Dish Environment

Current finals handoff: `FINALS_PART3_HANDOFF_20260720.md`. Parts 1 and 2 are
complete and frozen; this directory is now the active Part 3 work line.

This directory contains the finals-facing fixed-dish bag-drop and grinding
workflow. The answer-profile v3 overlap sequence has already passed physical
acceptance; RB-VoE treats that sequence as a frozen, versioned artifact and
does not change poses, speeds, G23 pulses, grinding range or overlap timing.

## Frozen physical mapping

| Node | Physical side | Current role | Identity |
| --- | --- | --- | --- |
| arm01 | left | fixed-dish support; existing bag workflow remains intact | `mycobot-arm-01`, `.64`, MAC `e4:5f:01:bf:de:a7`, serial `1000000092fb92d3` |
| arm02 | right | grinding-rod motion | `192.0.2.136`, MAC `98:fe:54:0c:94:07`, serial `10000000f08c41fc` |

The mapping combines immutable identity evidence with the accepted physical
layout and tool identities. Every new boot still requires a fixed-address,
read-only identity check; no new pose teaching is part of the RB-VoE path.

## Camera topology

There are exactly two cameras:

1. arm01 keeps its wrist camera for the proven bag gate and close-up evidence.
2. The former arm02 wrist camera is mounted rigidly above the grinding dish.
   Its USB cable remains connected to arm02. The camera-only service exposes a
   snapshot on `http://192.0.2.136:8892/snapshot.jpg` on the unified
   `xrd-lab_5G` LAN. The camera and AI-X5 replay have already distinguished
   empty dish and bag-present states. RB-VoE adds a run/challenge-bound A0
   capture rather than a new vision model. Historical K70 `10.*` addresses
   are evidence-only and must not be used as current endpoints.

The camera-only service is intentionally separate from the legacy arm02 HTTP
service because the legacy service opens the robot serial bus and powers the
arm when its poller starts.

## Safety state

`station_config.json` deliberately grants RB-VoE no motion, teaching, serial,
workspace, coordinate-correction or collision-model authority. Passing the
software validator never grants motion authority. The finals wrapper is also
PlanOnly with no arguments; physical execution remains a separate explicit
operator-authorized path.

Run the local static gate:

```powershell
python workstation/dual_arm/validate_environment.py
python -m unittest workstation/dual_arm/test_environment_static.py
```

Preview the RB-VoE and finals wrappers without remote contact or motion:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/deploy_rb_voe_edge_collectors.ps1 -PlanOnly
powershell -NoProfile -ExecutionPolicy Bypass -File workstation/dual_arm/run_dual_arm_bag_grind.ps1 -PlanOnly
```

The execution switches are intentionally separate. The preflight reads only
identity/sysfs/service/dependency state. Deployment writes edge-device files
with timestamped backups but leaves the camera service stopped and disabled.
On 2026-07-13 the deployment also backed up arm02's crontab, removed the
factory `automatic-ager` reboot entry, and stopped its serial-owning aging
process. That program contains autonomous joint and Cartesian motion commands
and must not be restored for dual-arm commissioning.

The overhead camera service remains disabled at boot. During the separately
authorized A0 step it may be started only as the exact camera-only owner; the
runner verifies PID, socket and exclusive `/dev/video0` ownership before and
after acquisition. It must be stopped before any separately authorized arm
motion. The camera service contains no robot SDK or actuator entrypoint.

## Next-boot RB-VoE gates

1. Fresh read-only identity, boot, service, serial-owner and artifact check on
   both Raspberry Pis.
2. Confirm the accepted dish/camera/tool geometry has not moved.
3. Capture five empty and five bag-present A0 frames under one run-bound
   challenge; reject stale, cross-frame, cross-sample or replayed evidence.
4. Stop the camera service and prove the serial owners are unchanged.
5. Run the central zero-authority shadow capture. A valid negative visual
   result is `HOLD`, not an integrity failure.
6. Any physical v3 demonstration remains outside RB-VoE and requires the
   existing explicit operator safety confirmation.

Do not use `workstation/web/arm02_service.py` for A0 because it opens the robot
serial bus. Do not restore the factory `automatic-ager` boot task. Do not use
historical K70 endpoints or old teaching poses as current authority.
