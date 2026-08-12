#!/usr/bin/env python3
"""Validate F407 post-flash estop-interlock evidence.

This tool only reads JSON files. It never opens a serial device or writes to
the ROS graph. The physical serial test remains owned by f407_link_test.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


EXPECTED_FIRMWARE = {
    "protocol_version": 2,
    "required_capabilities": 0x003F,
    "build_id": 2026071907,
    "test_mode": 0,
    "hw_variant": 1,
}

EXPECTED_ACKS = {
    "EMERGENCY_STOP": (0, 0),
    "SET_LIFT_HEIGHT blocked": (3, 3),
    "SET_ELECTROMAGNET ON blocked": (3, 3),
    "SET_ELECTROMAGNET OFF allowed": (0, 0),
    "LIFT_HOME blocked": (3, 3),
}


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": "PASS" if ok else "FAIL",
        "detail": detail,
    }


def validate_interlock_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    checks.append(
        _check(
            "schema",
            report.get("schema_version") == "xrd-f407-interlock-evidence-v2",
            str(report.get("schema_version")),
        )
    )
    checks.append(_check("overall", report.get("overall") == "PASS", str(report.get("overall"))))
    checks.append(
        _check(
            "interlock_mode",
            report.get("verify_estop_interlock") is True,
            f"verify_estop_interlock={report.get('verify_estop_interlock')!r}",
        )
    )
    checks.append(
        _check(
            "clear_estop_forbidden",
            report.get("clear_estop_requested") is False,
            f"clear_estop_requested={report.get('clear_estop_requested')!r}",
        )
    )

    firmware = report.get("firmware_identity") if isinstance(report.get("firmware_identity"), dict) else {}
    expected = firmware.get("expected") if isinstance(firmware.get("expected"), dict) else {}
    firmware_ok = (
        firmware.get("valid") is True
        and firmware.get("protocol_version") == EXPECTED_FIRMWARE["protocol_version"]
        and (int(firmware.get("capabilities") or 0) & EXPECTED_FIRMWARE["required_capabilities"])
        == EXPECTED_FIRMWARE["required_capabilities"]
        and firmware.get("build_id") == EXPECTED_FIRMWARE["build_id"]
        and firmware.get("test_mode") == EXPECTED_FIRMWARE["test_mode"]
        and firmware.get("hw_variant") == EXPECTED_FIRMWARE["hw_variant"]
        and expected == EXPECTED_FIRMWARE
    )
    checks.append(
        _check(
            "firmware_identity_exact",
            firmware_ok,
            json.dumps(firmware, ensure_ascii=True, sort_keys=True),
        )
    )

    safety = report.get("safety") if isinstance(report.get("safety"), dict) else {}
    safety_ok = (
        safety.get("motion_cmd_sent") is False
        and safety.get("interlock_test_sends_no_nonzero_cmd_vel") is True
        and safety.get("default_leaves_estop_latched") is True
        and safety.get("serial_exclusive_open") is True
        and safety.get("identity_verified_before_commands") is True
        and safety.get("commands_started") is True
    )
    checks.append(_check("safety_contract", safety_ok, json.dumps(safety, sort_keys=True)))

    last_safety = report.get("last_safety") if isinstance(report.get("last_safety"), dict) else {}
    checks.append(
        _check(
            "estop_left_latched",
            last_safety.get("estop_latched") is True,
            json.dumps(last_safety, sort_keys=True),
        )
    )

    ack_checks = report.get("ack_checks") if isinstance(report.get("ack_checks"), list) else []
    by_label = {
        item.get("label"): item
        for item in ack_checks
        if isinstance(item, dict) and isinstance(item.get("label"), str)
    }
    exact_ack_set = set(by_label) == set(EXPECTED_ACKS) and len(ack_checks) == len(EXPECTED_ACKS)
    checks.append(
        _check(
            "ack_set_exact",
            exact_ack_set,
            f"actual={sorted(by_label)} expected={sorted(EXPECTED_ACKS)} count={len(ack_checks)}",
        )
    )
    for label, (status, expected_status) in EXPECTED_ACKS.items():
        item = by_label.get(label, {})
        ok = (
            item.get("status") == status
            and item.get("expected_status") == expected_status
            and item.get("ok") is True
        )
        checks.append(_check(f"ack:{label}", ok, json.dumps(item, sort_keys=True)))

    ack_failures = report.get("ack_failures")
    checks.append(
        _check(
            "ack_failures_empty",
            ack_failures == [],
            json.dumps(ack_failures, sort_keys=True),
        )
    )
    return checks


def build_validation(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("interlock report root must be an object")
    checks = validate_interlock_report(report)
    failed = sum(item["status"] == "FAIL" for item in checks)
    return {
        "schema_version": "xrd-f407-postflash-interlock-validation-v1",
        "generated_at_unix": time.time(),
        "source_report": str(report_path),
        "overall": "PASS" if failed == 0 else "FAIL",
        "counts": {"PASS": len(checks) - failed, "FAIL": failed},
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report_path = Path(args.report).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    try:
        validation = build_validation(report_path)
    except Exception as exc:
        validation = {
            "schema_version": "xrd-f407-postflash-interlock-validation-v1",
            "generated_at_unix": time.time(),
            "source_report": str(report_path),
            "overall": "FAIL",
            "counts": {"PASS": 0, "FAIL": 1},
            "checks": [
                _check("report_readable", False, f"{type(exc).__name__}: {exc}"),
            ],
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"overall": validation["overall"], "out": str(out_path)}, sort_keys=True))
    return 0 if validation["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
