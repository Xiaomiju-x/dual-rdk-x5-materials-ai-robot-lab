# Dual-Arm Environment Handoff - 2026-07-13

## Decision

- Physical left is arm01 (`.64`, blue G23 bag gripper, wrist camera retained).
- Physical right is arm02 (`.136` / K70 `.145`, red grinding-rod gripper).
- The grinding dish is mechanically fixed.
- Exactly two cameras are used. The former arm02 wrist camera becomes the fixed
  overhead camera while its USB cable remains connected to arm02.
- Fixed-point commissioning uses only the two Raspberry Pis. AI brain X5 and
  its AprilTag gate are optional later additions and are not current blockers.

## Local environment completed

- `station_config.json` freezes immutable identities, roles, camera topology,
  marker id 7 / 40 mm, named-pose names, and fail-closed authority flags.
- `overhead_camera_service.py` is camera-only and has no robot SDK, serial, GPIO,
  PWM, or actuator route.
- `overhead_station_gate_x5.py` detects the station marker but cannot calculate
  robot coordinates or authorize motion.
- `preflight_no_motion.ps1` reads identity, sysfs, dependencies, service state,
  and serial-owner PIDs without opening the serial bus or camera.
- `deploy_no_motion_environment.ps1` backs up and uploads edge files, installs
  the camera unit, and deliberately leaves it stopped and disabled.
- Local validation passed, including frozen arm01 SHA-256 checks and five static
  tests. Evidence is under `workstation/dual_arm/evidence/`.

## Current field status

- arm01 is verified at K70 `198.51.100.136`: hostname `mycobot-arm-01`, overlay
  `.64`, MAC `e4:5f:01:bf:de:a7`, CPU serial `1000000092fb92d3`.
- arm02 is verified at K70 `198.51.100.145`: hostname `er`, overlay `.136`, MAC
  `98:fe:54:0c:94:07`, CPU serial `10000000f08c41fc`.
- The arm02 factory `automatic-ager` was found holding `/dev/ttyAMA0`; its code
  contains autonomous motion. Its crontab was backed up, only that reboot entry
  was removed, the process was stopped, and the serial port is now unowned.
- The camera-only service and config are installed on arm02 with matching
  hashes. The service remains disabled and inactive, so no camera was opened.
- arm01 `xrd-workcockpit.service` remains active to preserve the frozen bag demo.
  The future gated teach-preparation script will stop it only after a new
  operator confirmation.
- X5 is powered off and is not required for fixed-point dual-arm commissioning.

Evidence: `preflight_postdeploy_20260713.json`, `deploy_live_20260713.json`, and
`environment_validation_20260713.json` under `workstation/dual_arm/evidence/`.

The arm02 camera has not yet been physically moved, the dish fixture has not
received explicit field confirmation, the two robot-base transform has not
been measured, and no named pose values exist.

## Hard boundary

`motion_ready=false`, `motion_authorized=false`, and `teach_authorized=false`.
The next session may finish read-only SSH/dependency verification and deploy the
stopped camera service. Servo release, pose teaching, gripper commands, and arm
motion require a new explicit operator confirmation after the environment is
fully verified.
