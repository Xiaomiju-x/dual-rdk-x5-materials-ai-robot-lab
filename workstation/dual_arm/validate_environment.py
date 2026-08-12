#!/usr/bin/env python3
"""Validate the dual-arm commissioning environment without touching hardware."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("station_config.json")
FORBIDDEN_CAMERA_SERVICE_MARKERS = (
    "pymycobot",
    "/dev/ttyama0",
    "power_on(",
    "release_all_servos(",
    "send_angles(",
    "send_coords(",
    "set_pwm_",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: Any) -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def validate(config_path: Path) -> dict[str, Any]:
    data = json.loads(config_path.read_text(encoding="utf-8-sig"))
    checks: list[dict[str, Any]] = []

    add_check(
        checks,
        "schema",
        data.get("schema_version") == "xrd-dual-arm-station-environment-v1",
        data.get("schema_version"),
    )
    arms = data.get("arms", {})
    add_check(checks, "exact_arm_ids", set(arms) == {"arm01", "arm02"}, sorted(arms))
    add_check(
        checks,
        "physical_side_mapping",
        arms.get("arm01", {}).get("physical_side") == "left"
        and arms.get("arm02", {}).get("physical_side") == "right",
        {arm: row.get("physical_side") for arm, row in arms.items()},
    )
    macs = [row.get("wlan0_mac") for row in arms.values()]
    serials = [row.get("cpu_serial") for row in arms.values()]
    add_check(checks, "unique_arm_macs", len(macs) == 2 and len(set(macs)) == 2, macs)
    add_check(checks, "unique_arm_cpu_serials", len(serials) == 2 and len(set(serials)) == 2, serials)

    cameras = data.get("cameras", {})
    camera_count = sum(int(row.get("count", 0)) for row in cameras.values())
    add_check(checks, "exactly_two_cameras", camera_count == 2, camera_count)
    overhead = cameras.get("grinding_overhead", {})
    add_check(
        checks,
        "overhead_camera_is_camera_only",
        overhead.get("service") == "xrd-overhead-camera.service"
        and overhead.get("service_start_authorized") is False,
        overhead,
    )

    safety = data.get("safety_authority", {})
    expected_closed = {
        "motion_authorized": False,
        "teach_authorized": False,
        "serial_access_allowed_during_environment_preflight": False,
        "camera_frame_capture_allowed_during_identity_preflight": False,
        "shared_workspace_lease_enabled": False,
        "collision_model_physically_calibrated": False,
    }
    add_check(
        checks,
        "motion_authority_fail_closed",
        all(safety.get(key) is expected for key, expected in expected_closed.items())
        and safety.get("fail_closed_on_missing_peer_state") is True,
        {key: safety.get(key) for key in (*expected_closed, "fail_closed_on_missing_peer_state")},
    )

    pose_contract = data.get("named_pose_contract", {})
    add_check(
        checks,
        "poses_intentionally_unrecorded",
        pose_contract.get("pose_values_recorded") is False,
        pose_contract.get("pose_values_recorded"),
    )

    frozen = data.get("frozen_arm01_baseline", {}).get("files", {})
    frozen_results: dict[str, Any] = {}
    frozen_ok = bool(frozen)
    for relative, expected_hash in frozen.items():
        path = ROOT / relative
        actual_hash = sha256_file(path) if path.is_file() else None
        row_ok = actual_hash == expected_hash
        frozen_ok = frozen_ok and row_ok
        frozen_results[relative] = {
            "exists": path.is_file(),
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "ok": row_ok,
        }
    add_check(checks, "arm01_frozen_baseline_unchanged", frozen_ok, frozen_results)

    camera_service_path = Path(__file__).with_name("overhead_camera_service.py")
    source = camera_service_path.read_text(encoding="utf-8").lower()
    found = [marker for marker in FORBIDDEN_CAMERA_SERVICE_MARKERS if marker in source]
    add_check(
        checks,
        "camera_service_has_no_robot_control_surface",
        camera_service_path.is_file() and not found,
        {"path": str(camera_service_path), "forbidden_found": found},
    )

    ok = all(row["ok"] for row in checks)
    blockers = [
        "live_readonly_identity_recheck",
        "operator_relocates_arm02_camera_to_fixed_overhead_mount",
        "operator_confirms_grinding_dish_fixture",
        "physical_base_transform_measurement",
        "named_pose_teaching",
        "explicit_motion_confirmation",
    ]
    return {
        "schema_version": "xrd-dual-arm-environment-validation-v1",
        "ok": ok,
        "software_environment_ready": ok,
        "motion_ready": False,
        "hardware_touched": False,
        "network_touched": False,
        "serial_opened": False,
        "camera_opened": False,
        "checks": checks,
        "motion_blockers": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    report = validate(args.config.resolve())
    text = json.dumps(report, ensure_ascii=False, indent=None if args.compact else 2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
