#!/usr/bin/env python3
"""Audit embodied_brain v3 acceptance evidence.

Inputs are intentionally plain files produced by:

- embodied_v3_acceptance_check.sh
- data_loop_stop.sh
- data_loop_to_lerobot.py

The audit is dependency-free and can run on the RDK X5 or on the PC after
copying evidence back. It does not start ROS and never publishes commands.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TARGET_F407_BUILD_ID = 2026071907

REQUIRED_ACCEPTANCE_CHECKS = {
    "acceptance_start": "Acceptance run start marker was written before evidence collection",
    "tiny_occ_risk_bin": "BPU tiny occ-risk bin is present and hash-matched",
    "hobot_dnn_import": "hobot_dnn runtime import succeeded",
    "ros_nodes": "ROS graph node list captured",
    "ros_topics": "ROS graph topic list captured",
    "ros_services": "ROS graph service list captured",
    "physical_evidence_config": "Pickup physical-evidence mode and ROS endpoints were captured",
    "topic_scan": "LiDAR /scan has a sample",
    "topic_scan_depth": "Depth-derived /scan_depth has a sample",
    "topic_odom": "Odometry /odom has a sample",
    "topic_map": "SLAM /map has a sample",
    "topic_lab_fsd_fsd_v3_status": "Lab-FSD v3 status has a sample",
    "topic_lab_fsd_future_risk": "Lab-FSD future risk has a sample",
    "topic_lab_fsd_input_status": "Lab-FSD input status has a sample",
    "topic_lab_fsd_vision_bev": "Raw Vision-BEV occupancy grid has a sample",
    "topic_lab_fsd_vision_risk": "Raw Vision-BEV risk scalar has a sample",
    "topic_lab_fsd_vision_objects": "Vision-BEV raw provenance metadata has a sample",
    "topic_lab_fsd_safety_gate": "Lab-FSD safety gate has a sample",
    "topic_lab_fsd_shadow_path": "Lab-FSD shadow path has a sample",
    "topic_lab_fsd_trajectory_scores": "Lab-FSD trajectory scores have a sample",
    "topic_lab_fsd_bev": "Lab-FSD current BEV occupancy grid has a sample",
    "topic_lab_fsd_future_bev": "Lab-FSD future BEV occupancy grid has a sample",
    "topic_lab_fsd_policy_tokens": "Lab-FSD policy-token prior has a sample",
    "topic_diagnostics": "F407 /diagnostics has a sample",
    "topic_lift_status": "F407 /lift_status has a sample",
    "topic_f407_estop_latched": "F407 estop latch topic has a sample",
    "topic_f407_cmd_vel_expired": "F407 cmd_vel watchdog topic has a sample",
    "topic_f407_firmware_identity_valid": "F407 firmware identity-valid topic has a sample",
    "topic_f407_firmware_info": "F407 firmware identity JSON has a sample",
    "f407_interlock_report": "F407 firmware estop interlock report captured after flashing",
    "dispatch_stub_integration": "Isolated DispatchTask stub integration passed without physical output",
    "dispatch_fixture_integration": "Stationary pickup fixture passed in an isolated virtual-F407 domain",
    "cmd_vel_publishers": "/cmd_vel publisher list captured",
    "lab_fsd_not_cmd_vel_publisher": "Lab-FSD is not a /cmd_vel publisher",
    "mppi_not_cmd_vel_publisher": "MPPI is not a direct /cmd_vel publisher",
    "mppi_cost_bin": "BPU MPPI cost model is present and hash-matched",
}

WARN_ACCEPTANCE_CHECKS = {
    "lab_anomaly_autoencoder_bin": "BPU anomaly autoencoder bin is present when enabled",
    "data_loop_status": "data-loop status command ran",
    "bpu_status": "BPU status command ran",
    "cockpit_blackbox_recent": "WorkCockpit blackbox recent evidence captured",
    "topic_mppi_cmd_vel_proposed": "MPPI proposed velocity topic has a sample when MPPI is enabled",
    "topic_mppi_stats": "MPPI stats topic has a sample when MPPI is enabled",
}

REQUIRED_DATA_TOPICS = {
    "/cmd_vel",
    "/odom",
    "/scan",
    "/scan_depth",
    "/map",
    "/lab_fsd/fsd_v3_status",
    "/lab_fsd/future_risk",
    "/lab_fsd/input_status",
    "/lab_fsd/vision_bev",
    "/lab_fsd/vision_risk",
    "/lab_fsd/vision_objects",
    "/lab_fsd/safety_gate",
    "/lab_fsd/shadow_path",
    "/lab_fsd/trajectory_scores",
    "/lab_fsd/bev",
    "/lab_fsd/future_bev",
    "/lab_fsd/policy_tokens",
    "/diagnostics",
    "/lift_status",
    "/f407/estop_latched",
    "/f407/cmd_vel_expired",
    "/f407/firmware_identity_valid",
    "/f407/firmware_info",
}

WARN_DATA_TOPICS = {
    "/lab_fsd/anomaly_score",
    "/mppi/cmd_vel_proposed",
    "/mppi/stats",
}

REQUIRED_MODEL_ARTIFACTS = {
    "lab_fsd_tiny_occ_risk": "BPU TinyOccRisk model fingerprint is persisted in the data-loop manifest",
}

WARN_MODEL_ARTIFACTS = {
    "lab_anomaly_autoencoder": "BPU anomaly autoencoder fingerprint is persisted when enabled",
    "mppi_cost": "BPU MPPI cost-model fingerprint is persisted when enabled",
}

LEDGER_SCHEMA_VERSION = "xrd-data-loop-ledger-v1"
AUDIT_SCHEMA_VERSION = "xrd-embodied-v3-audit-v4"
AUDIT_POLICY_VERSION = "xrd-embodied-v3-acceptance-policy-v4"
FIXTURE_SCHEMA_VERSION = "xrd-dispatch-fixture-integration-v2"
FIXTURE_POLICY_VERSION = "xrd-dispatch-fixture-policy-v2"
FIXTURE_MONITOR_POLICY_VERSION = "procfs-process-tree-fd-monitor-v1"
FIXTURE_EXPECTED_STAGES = [1, 5, 6, 5, 8]
FIXTURE_EXPECTED_GOAL_IDS = [
    "fixture-invalid-identity",
    "fixture-estop-latched",
    "fixture-stationary-complete",
    "fixture-one-shot-consumed",
]
FIXTURE_EXPECTED_CHECK_NAMES = {
    "dispatch_action_server_ready",
    "dispatch_process_tree_no_serial_device_fd",
    "literal_cmd_vel_has_no_dispatch_publisher",
    "invalid_firmware_identity_rejected",
    "estop_latched_rejected",
    "stationary_goal_accepted",
    "stationary_f407_reported_completion",
    "stationary_structured_result",
    "stationary_stage_sequence",
    "stationary_progress_monotonic",
    "virtual_f407_service_sequence",
    "stationary_no_cmd_vel_messages",
    "stationary_final_pose_unclaimed",
    "stationary_fixture_one_shot_rejected",
    "dds_callbacks_drained",
    "dispatch_monitor_coverage_complete",
    "dispatch_process_clean_exit",
    "dispatch_log_clean",
    "dispatch_artifacts_recorded",
}
FIXTURE_EXPECTED_PARAMETERS: dict[str, bool | float] = {
    "stub_mode": False,
    "use_nav2": False,
    "execute_pickup_actuators": True,
    "allow_stationary_pickup_fixture": True,
    "stationary_pickup_fixture_only": True,
    "stationary_pickup_fixture_one_shot": True,
    "use_lab_fsd_guard": True,
    "safe_stop_on_failure": True,
    "pickup_height_m": 0.02,
    "transport_height_m": 0.04,
    "actuator_service_wait_s": 2.0,
    "actuator_timeout_s": 4.0,
}
FIXTURE_FORBIDDEN_DEVICE_PREFIXES = [
    "/dev/F407",
    "/dev/ttyUSB",
    "/dev/ttyACM",
    "/dev/serial",
]
FIXTURE_EXPECTED_LOG_ERROR_MARKERS = [
    "rejecting real dispatch goal while F407 firmware identity is invalid",
    "rejecting dispatch goal while F407 estop is latched",
    "rejecting stationary pickup fixture: one-shot already consumed",
]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
POSTFLASH_MANIFEST_SCHEMA_V1 = "xrd-f407-postflash-interlock-orchestration-v1"
POSTFLASH_MANIFEST_SCHEMA_V2 = "xrd-f407-postflash-interlock-orchestration-v2"
POSTFLASH_RECOVERY_SCHEMA_V1 = "xrd-f407-postflash-readonly-recovery-v1"
POSTFLASH_RECOVERABLE_SOURCE_FAILURE = "phase=post_restore_readonly_topics: unexpected exit rc=1"
POSTFLASH_BUNDLE_SCHEMA = "xrd-f407-postflash-bundle-index-v1"
POSTFLASH_VALIDATION_SCHEMA = "xrd-f407-postflash-interlock-validation-v1"
POSTFLASH_CONFIRMATION_SHA256 = hashlib.sha256(
    b"NO_LOAD_PATH_CLEAR_BASE_FIXED_HANDS_CLEAR_OPERATOR_PRESENT"
).hexdigest()
POSTFLASH_ARTIFACT_FILES = {
    "interlock_report": "f407_interlock_report.json",
    "interlock_log": "f407_postflash_interlock.log",
    "validation_report": "f407_postflash_validation.json",
    "post_firmware_topic": "f407_postflash_firmware_info.txt",
    "post_identity_topic": "f407_postflash_firmware_identity_valid.txt",
    "post_estop_topic": "f407_postflash_estop_latched.txt",
}
POSTFLASH_RECOVERY_ARTIFACT_FILES = {
    "source_failed_manifest": "f407_postflash_source_failed_manifest.json",
    "recovery_revalidation": "f407_postflash_recovery_revalidation.json",
}
POSTFLASH_V1_TOOL_FILES = {
    "orchestrator": ("f407_postflash_orchestrator.sh", "f407_postflash_interlock_acceptance.sh"),
    "link_test": ("f407_postflash_link_test.py", "f407_link_test.py"),
    "validator": ("f407_postflash_validator.py", "f407_postflash_report.py"),
}
POSTFLASH_V2_PHYSICAL_TOOL_FILES = {
    "orchestrator": (
        "f407_postflash_physical_orchestrator.sh",
        "physical_tool_orchestrator.sh",
    ),
    "link_test": (
        "f407_postflash_physical_link_test.py",
        "physical_tool_link_test.py",
    ),
    "validator": (
        "f407_postflash_physical_validator.py",
        "physical_tool_validator.py",
    ),
}
POSTFLASH_V2_RECOVERY_TOOL_FILE = (
    "f407_postflash_recovery.py",
    "f407_postflash_recover_readonly.py",
)

REQUIRED_RUNTIME_SERVICES = {
    "/estop": "F407 estop trigger service",
    "/clear_estop": "F407 local clear-estop service",
    "/set_lift_height": "F407 lift target service with ACK/arrival semantics",
    "/set_electromagnet": "F407 electromagnet service",
    "/lift_home": "F407 lift-home service",
}

REQUIRED_DIAGNOSTIC_KEYS = {
    "serial_f407_node: safety_bridge": "F407 safety diagnostic status",
    "estop_latched": "F407 estop latch key",
    "cmd_vel_expired": "F407 cmd_vel watchdog key",
    "cmd_vel_timeouts": "F407 cmd_vel timeout counter",
    "cmd_vel_blocked_by_estop": "F407 cmd_vel estop-block counter",
    "actuator_commands_blocked_by_estop": "F407 actuator estop-block counter",
    "hardware_estop_latched": "F407 firmware-level estop latch state",
    "f407_estop_blocked_commands": "F407 firmware-level blocked command counter",
    "f407_safety_blocked_commands": "F407 firmware safety-interlock blocked command counter",
    "last_safety_state_age_s": "F407 safety-state telemetry freshness",
    "firmware_identity_valid": "F407 exact firmware identity gate",
    "require_firmware_identity": "F407 identity enforcement is enabled",
    "firmware_protocol_version": "F407 protocol version",
    "firmware_capabilities": "F407 advertised safety capability mask",
    "firmware_build_id": "F407 immutable target build ID",
    "firmware_test_mode": "F407 TEST_MODE identity",
    "firmware_info_age_s": "F407 firmware identity freshness",
    "cmd_vel_blocked_by_firmware_identity": "ROS nonzero cmd_vel identity-gate counter",
    "actuator_commands_blocked_by_firmware_identity": "ROS actuator identity-gate counter",
    "lift_arrival_tolerance_m": "F407 lift arrival tolerance key",
    "requested_lift_target_m": "F407 requested lift target key",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(data: dict[str, Any]) -> bytes:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def physical_evidence_digest(record: dict[str, Any]) -> str:
    payload = {
        "observed_at_ns": int(record.get("observed_at_ns") or 0),
        "frame_id": str(record.get("frame_id") or ""),
        "evidence_id": str(record.get("evidence_id") or ""),
        "request_id": str(record.get("request_id") or ""),
        "sensor_id": str(record.get("sensor_id") or ""),
        "source_type": str(record.get("source_type") or ""),
        "observation": str(record.get("observation") or ""),
        "task_id": str(record.get("task_id") or ""),
        "bottle_id": str(record.get("bottle_id") or ""),
        "location_id": str(record.get("location_id") or ""),
        "confirmed": record.get("confirmed") is True,
        "hardware_observed": record.get("hardware_observed") is True,
        "confidence": struct.unpack(
            "!f", struct.pack("!f", float(record.get("confidence") or 0.0))
        )[0],
        "measured_value": float(record.get("measured_value") or 0.0),
        "unit": str(record.get("unit") or ""),
        "detail": str(record.get("detail") or ""),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_physical_confirmation(raw: Any, task_id: str) -> tuple[bool, str]:
    if not isinstance(raw, str) or not raw.strip():
        return False, "physical_confirmation missing"
    try:
        confirmation = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        return False, f"physical_confirmation invalid JSON: {exc}"
    if not isinstance(confirmation, dict):
        return False, "physical_confirmation root is not an object"
    records = confirmation.get("records") if isinstance(confirmation.get("records"), list) else []
    evidence_ids = [str(item.get("evidence_id") or "") for item in records if isinstance(item, dict)]
    observations = [str(item.get("observation") or "") for item in records if isinstance(item, dict)]
    source_compatibility = {
        "lift_position_confirmed": {"encoder", "limit_switch", "vision_depth"},
        "object_attached": {"load_cell", "photoelectric", "vision_depth", "vision_rgb"},
        "object_released": {"load_cell", "photoelectric", "vision_depth", "vision_rgb"},
    }
    records_ok = len(records) == 3
    if records_ok:
        for item in records:
            if not isinstance(item, dict):
                records_ok = False
                break
            observation = str(item.get("observation") or "")
            source_type = str(item.get("source_type") or "")
            expected_sha = str(item.get("payload_sha256") or "")
            try:
                digest_ok = SHA256_RE.fullmatch(expected_sha) is not None and (
                    physical_evidence_digest(item) == expected_sha
                )
            except (TypeError, ValueError, OverflowError):
                digest_ok = False
            records_ok = bool(
                records_ok
                and item.get("confirmed") is True
                and item.get("hardware_observed") is True
                and str(item.get("task_id") or "") == task_id
                and source_type in source_compatibility.get(observation, set())
                and digest_ok
            )
            if not records_ok:
                break
    ordered = bool(
        observations[:1] == ["lift_position_confirmed"]
        and len(observations) == 3
        and observations[1] in {"object_attached", "object_released"}
        and observations[2] == "lift_position_confirmed"
    )
    valid = bool(
        confirmation.get("schema_version") == "xrd-pickup-physical-confirmation-v1"
        and confirmation.get("task_id") == task_id
        and confirmation.get("confirmed") is True
        and confirmation.get("evidence_count") == 3
        and confirmation.get("evidence_ids") == evidence_ids
        and confirmation.get("observations") == observations
        and confirmation.get("independent_lift_evidence") is True
        and confirmation.get("independent_object_evidence") is True
        and confirmation.get("replay_free") is True
        and len(set(evidence_ids)) == 3
        and ordered
        and records_ok
    )
    return valid, f"records={len(records)} ordered={ordered} replay_free={len(set(evidence_ids)) == len(evidence_ids)}"


def ledger_entry_digest(entry: dict[str, Any]) -> str:
    payload = {key: value for key, value in entry.items() if key != "entry_sha256"}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def verify_ledger(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "available": path.exists(),
        "checked": 0,
        "ok": 0,
        "errors": [],
        "entries": [],
    }
    if not path.exists():
        return result
    previous = ""
    for line_number, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if not raw.strip():
            continue
        result["checked"] += 1
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError as exc:
            result["errors"].append(f"line {line_number}: invalid JSON: {exc}")
            continue
        if not isinstance(entry, dict):
            result["errors"].append(f"line {line_number}: entry is not an object")
            continue
        expected_sequence = len(result["entries"])
        actual_digest = ledger_entry_digest(entry)
        errors: list[str] = []
        if entry.get("schema_version") != LEDGER_SCHEMA_VERSION:
            errors.append("schema")
        if entry.get("sequence") != expected_sequence:
            errors.append("sequence")
        if entry.get("previous_entry_sha256", "") != previous:
            errors.append("previous_hash")
        if entry.get("entry_sha256") != actual_digest:
            errors.append("entry_hash")
        if errors:
            result["errors"].append(f"line {line_number}: {','.join(errors)}")
        else:
            result["ok"] += 1
        result["entries"].append(entry)
        previous = str(entry.get("entry_sha256") or actual_digest)
    return result


def load_checks(path: Path) -> dict[str, dict[str, str]]:
    checks: dict[str, dict[str, str]] = {}
    if not path.exists():
        return checks
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.strip():
            continue
        parts = raw.split("\t", 2)
        if len(parts) < 2:
            continue
        name = parts[0].strip()
        checks[name] = {
            "status": parts[1].strip(),
            "detail": parts[2].strip() if len(parts) > 2 else "",
        }
    return checks


def add_result(results: list[dict[str, Any]], name: str, status: str, detail: str, evidence: str = "") -> None:
    results.append(
        {
            "name": name,
            "status": status,
            "detail": detail,
            "evidence": evidence,
        }
    )


def check_file_contains(path: Path, needle: str) -> bool:
    if not path.exists():
        return False
    return needle in path.read_text(encoding="utf-8", errors="replace")


def load_jsonl_objects_from_text(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def load_json_objects_from_ros_text(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    objects: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        candidates: list[str] = []
        if line.startswith("{"):
            candidates.append(line)
        if line.startswith("data:"):
            value = line.split(":", 1)[1].strip()
            if value:
                candidates.append(value)
                try:
                    literal = ast.literal_eval(value)
                    if isinstance(literal, str):
                        candidates.append(literal)
                except Exception:
                    candidates.append(value.strip("'\""))
        for candidate in candidates:
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                objects.append(obj)
                break
    return objects


def audit_postflash_orchestration(
    results: list[dict[str, Any]],
    accept_dir: Path,
    checks: dict[str, dict[str, str]],
    acceptance_start_unix: float,
) -> None:
    gate = checks.get("f407_postflash_manifest")
    manifest_path = accept_dir / "f407_postflash_manifest.json"
    index_path = accept_dir / "f407_postflash_bundle_index.json"
    if gate is None:
        add_result(
            results,
            "f407_postflash_orchestration_content",
            "WARN",
            "exact post-flash orchestration manifest was not requested by this legacy acceptance bundle",
            str(manifest_path),
        )
        return
    if gate.get("status") != "OK":
        status = "WARN" if gate.get("status") == "WARN" else "FAIL"
        add_result(
            results,
            "f407_postflash_orchestration_content",
            status,
            f"collector gate status={gate.get('status')}: {gate.get('detail')}",
            str(manifest_path),
        )
        return

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        index = json.loads(index_path.read_text(encoding="utf-8"))
        validation_path = accept_dir / POSTFLASH_ARTIFACT_FILES["validation_report"]
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        manifest_schema = str(manifest.get("schema_version") or "")
        recovered = manifest_schema == POSTFLASH_MANIFEST_SCHEMA_V2
        source_failed_manifest: dict[str, Any] = {}
        recovery_validation: dict[str, Any] = {}
        if recovered:
            source_failed_path = accept_dir / POSTFLASH_RECOVERY_ARTIFACT_FILES["source_failed_manifest"]
            recovery_validation_path = accept_dir / POSTFLASH_RECOVERY_ARTIFACT_FILES["recovery_revalidation"]
            source_failed_manifest = json.loads(source_failed_path.read_text(encoding="utf-8"))
            recovery_validation = json.loads(recovery_validation_path.read_text(encoding="utf-8"))
    except Exception as exc:
        add_result(
            results,
            "f407_postflash_orchestration_content",
            "FAIL",
            f"post-flash bundle missing or invalid JSON: {type(exc).__name__}: {exc}",
            str(manifest_path),
        )
        return

    now = time.time()
    started_at = float(manifest.get("started_at_unix") or 0.0)
    finished_at = float(manifest.get("finished_at_unix") or 0.0)
    age_s = now - finished_at if finished_at > 0.0 else float("inf")
    source_dir = Path(str(index.get("source_directory") or ""))
    source_manifest = Path(str(index.get("source_manifest") or ""))

    evaluations: dict[str, tuple[bool, str]] = {}
    evaluations["f407_postflash_manifest_contract"] = (
        bool(
            manifest_schema in {POSTFLASH_MANIFEST_SCHEMA_V1, POSTFLASH_MANIFEST_SCHEMA_V2}
            and manifest.get("overall") == "PASS"
            and manifest.get("failure_reason") == ""
            and 0.0 < started_at <= finished_at <= now + 5.0
            and 0.0 <= age_s <= 86400.0
            and acceptance_start_unix > 0.0
            and finished_at <= acceptance_start_unix + 5.0
        ),
        f"schema={manifest_schema} overall={manifest.get('overall')} age_s={age_s:.1f}",
    )

    hostname = manifest.get("hostname") if isinstance(manifest.get("hostname"), dict) else {}
    evaluations["f407_postflash_hostname_contract"] = (
        bool(
            hostname.get("expected") == "embodied-x5"
            and hostname.get("actual") == "embodied-x5"
            and hostname.get("matched") is True
        ),
        json.dumps(hostname, sort_keys=True),
    )

    confirmation = (
        manifest.get("operator_confirmation")
        if isinstance(manifest.get("operator_confirmation"), dict)
        else {}
    )
    evaluations["f407_postflash_operator_confirmation"] = (
        bool(
            confirmation.get("safe_field_state_confirmed") is True
            and confirmation.get("confirmation_token_sha256") == POSTFLASH_CONFIRMATION_SHA256
            and confirmation.get("magnet_off_drop_hazard_acknowledged") is True
            and confirmation.get("raw_confirmation_token_stored") is False
        ),
        json.dumps(confirmation, sort_keys=True),
    )

    tooling = manifest.get("tooling") if isinstance(manifest.get("tooling"), dict) else {}
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    command = manifest.get("command_contract") if isinstance(manifest.get("command_contract"), dict) else {}
    link_test = tooling.get("link_test") if isinstance(tooling.get("link_test"), dict) else {}
    interlock_artifact = (
        artifacts.get("interlock_report")
        if isinstance(artifacts.get("interlock_report"), dict)
        else {}
    )
    expected_report_path = str(interlock_artifact.get("path") or "")
    if recovered:
        physical = (
            manifest.get("physical_interlock")
            if isinstance(manifest.get("physical_interlock"), dict)
            else {}
        )
        physical_source_manifest = str(physical.get("source_manifest") or "")
        if "/" in physical_source_manifest:
            expected_report_path = (
                physical_source_manifest.rsplit("/", 1)[0] + "/f407_interlock_report.json"
            )
        else:
            expected_report_path = str(
                Path(physical_source_manifest).parent / "f407_interlock_report.json"
            )
    expected_argv = [
        "python3",
        str(link_test.get("path") or ""),
        "--port",
        "/dev/F407",
        "--verify-estop-interlock",
        "--require-ack",
        "--report",
        expected_report_path,
    ]
    argv = command.get("argv") if isinstance(command.get("argv"), list) else []
    evaluations["f407_postflash_command_contract"] = (
        bool(
            argv == expected_argv
            and "--clear-estop" not in argv
            and "--v" not in argv
            and "--move-sec" not in argv
            and command.get("clear_estop_requested") is False
            and command.get("nonzero_cmd_vel_requested") is False
            and command.get("electromagnet_off_is_sent") is True
            and command.get("physical_completion_claimed") is False
        ),
        json.dumps(command, sort_keys=True),
    )

    serial = (
        manifest.get("serial_exclusivity")
        if isinstance(manifest.get("serial_exclusivity"), dict)
        else {}
    )
    evaluations["f407_postflash_serial_exclusivity"] = (
        bool(
            serial.get("device") == "/dev/F407"
            and isinstance(serial.get("owners_before"), list)
            and serial.get("owners_after_stop") == []
            and serial.get("unowned_before_test") is True
        ),
        json.dumps(serial, sort_keys=True),
    )

    restore = manifest.get("service_restore") if isinstance(manifest.get("service_restore"), dict) else {}
    pre = restore.get("pre") if isinstance(restore.get("pre"), dict) else {}
    post = restore.get("post") if isinstance(restore.get("post"), dict) else {}
    system_pre = pre.get("system") if isinstance(pre.get("system"), dict) else {}
    user_pre = pre.get("user") if isinstance(pre.get("user"), dict) else {}
    system_post = post.get("system") if isinstance(post.get("system"), dict) else {}
    user_post = post.get("user") if isinstance(post.get("user"), dict) else {}
    service_state_equal = system_pre == system_post and user_pre == user_post
    embodied_active_count = sum(
        item.get("embodied_brain.service") == "active" for item in (system_pre, user_pre)
    )
    evaluations["f407_postflash_service_restore"] = (
        bool(
            restore.get("services_quiesced") is True
            and restore.get("attempted") is True
            and restore.get("success") is True
            and service_state_equal
            and embodied_active_count == 1
        ),
        f"pre={pre} post={post}",
    )

    index_copied = index.get("copied") if isinstance(index.get("copied"), dict) else {}
    expected_source_manifest_name = (
        "postflash_interlock_recovered_manifest.json"
        if recovered
        else "postflash_interlock_manifest.json"
    )
    expected_copy_count = 13 if recovered else 10
    index_ok = bool(
        index.get("schema_version") == POSTFLASH_BUNDLE_SCHEMA
        and index.get("overall") == "PASS"
        and index.get("read_only_source") is True
        and index.get("physical_hardware_touched") is False
        and source_manifest.name == expected_source_manifest_name
        and source_dir.name.startswith("postflash_")
        and source_manifest.parent == source_dir
        and source_dir.parent.name == "f407_postflash_acceptance"
        and index.get("source_manifest_sha256") == sha256_file(manifest_path)
        and isinstance(index_copied.get("manifest"), dict)
        and index_copied["manifest"].get("sha256") == sha256_file(manifest_path)
        and len(index_copied) == expected_copy_count
    )

    artifact_hash_ok = True
    artifact_details: list[str] = []
    expected_source_artifact_names = {
        "interlock_report": "f407_interlock_report.json",
        "interlock_log": "f407_interlock.log",
        "validation_report": "f407_interlock_validation.json",
        "post_firmware_topic": "post_f407_firmware_info.txt",
        "post_identity_topic": "post_f407_firmware_identity_valid.txt",
        "post_estop_topic": "post_f407_estop_latched.txt",
    }
    artifact_files = dict(POSTFLASH_ARTIFACT_FILES)
    if recovered:
        artifact_files.update(POSTFLASH_RECOVERY_ARTIFACT_FILES)
        expected_source_artifact_names.update(
            {
                "source_failed_manifest": "source_failed_manifest.json",
                "recovery_revalidation": "recovery_interlock_revalidation.json",
            }
        )
    for key, bundle_name in artifact_files.items():
        entry = artifacts.get(key) if isinstance(artifacts.get(key), dict) else {}
        copied_entry = index_copied.get(f"artifact:{key}") if isinstance(index_copied.get(f"artifact:{key}"), dict) else {}
        path = accept_dir / bundle_name
        source_path = Path(str(entry.get("path") or ""))
        ok = bool(
            path.is_file()
            and source_path.parent == source_dir
            and source_path.name == expected_source_artifact_names[key]
            and entry.get("exists") is True
            and entry.get("sha256") == sha256_file(path)
            and int(entry.get("size_bytes") or -1) == path.stat().st_size
            and copied_entry.get("sha256") == sha256_file(path)
            and Path(str(copied_entry.get("path") or "")).name == bundle_name
        )
        artifact_hash_ok = artifact_hash_ok and ok
        artifact_details.append(f"{key}={ok}")

    tool_hash_ok = True
    tool_details: list[str] = []
    expected_tools_parent = source_dir.parent.parent / "tools"
    if recovered:
        physical = (
            manifest.get("physical_interlock")
            if isinstance(manifest.get("physical_interlock"), dict)
            else {}
        )
        snapshots = (
            physical.get("tool_snapshots")
            if isinstance(physical.get("tool_snapshots"), dict)
            else {}
        )
        for key, (bundle_name, source_name) in POSTFLASH_V2_PHYSICAL_TOOL_FILES.items():
            entry = snapshots.get(key) if isinstance(snapshots.get(key), dict) else {}
            top_entry = tooling.get(key) if isinstance(tooling.get(key), dict) else {}
            copied_entry = (
                index_copied.get(f"tool:physical_{key}")
                if isinstance(index_copied.get(f"tool:physical_{key}"), dict)
                else {}
            )
            path = accept_dir / bundle_name
            source_path = Path(str(entry.get("path") or ""))
            digest = sha256_file(path) if path.is_file() else ""
            ok = bool(
                path.is_file()
                and source_path.parent == source_dir
                and source_path.name == source_name
                and entry.get("exists") is True
                and entry.get("sha256") == digest
                and int(entry.get("size_bytes") or -1) == path.stat().st_size
                and top_entry.get("sha256") == digest
                and copied_entry.get("sha256") == digest
                and Path(str(copied_entry.get("path") or "")).name == bundle_name
            )
            tool_hash_ok = tool_hash_ok and ok
            tool_details.append(f"physical_{key}={ok}")

        recovery = manifest.get("recovery") if isinstance(manifest.get("recovery"), dict) else {}
        recovery_entry = recovery.get("tool") if isinstance(recovery.get("tool"), dict) else {}
        top_recovery_entry = tooling.get("recovery") if isinstance(tooling.get("recovery"), dict) else {}
        bundle_name, source_name = POSTFLASH_V2_RECOVERY_TOOL_FILE
        copied_entry = (
            index_copied.get("tool:recovery")
            if isinstance(index_copied.get("tool:recovery"), dict)
            else {}
        )
        path = accept_dir / bundle_name
        source_path = Path(str(recovery_entry.get("path") or ""))
        digest = sha256_file(path) if path.is_file() else ""
        ok = bool(
            path.is_file()
            and source_path.parent == expected_tools_parent
            and source_path.name == source_name
            and recovery_entry == top_recovery_entry
            and recovery_entry.get("sha256") == digest
            and copied_entry.get("sha256") == digest
            and Path(str(copied_entry.get("path") or "")).name == bundle_name
        )
        tool_hash_ok = tool_hash_ok and ok
        tool_details.append(f"recovery={ok}")
    else:
        for key, (bundle_name, source_name) in POSTFLASH_V1_TOOL_FILES.items():
            entry = tooling.get(key) if isinstance(tooling.get(key), dict) else {}
            copied_entry = index_copied.get(f"tool:{key}") if isinstance(index_copied.get(f"tool:{key}"), dict) else {}
            path = accept_dir / bundle_name
            source_path = Path(str(entry.get("path") or ""))
            ok = bool(
                path.is_file()
                and source_path.parent == expected_tools_parent
                and source_path.name == source_name
                and entry.get("sha256") == sha256_file(path)
                and copied_entry.get("sha256") == sha256_file(path)
                and Path(str(copied_entry.get("path") or "")).name == bundle_name
            )
            tool_hash_ok = tool_hash_ok and ok
            tool_details.append(f"{key}={ok}")
    evaluations["f407_postflash_bundle_hashes"] = (
        index_ok and artifact_hash_ok and tool_hash_ok,
        f"index={index_ok} artifacts={artifact_details} tools={tool_details}",
    )

    validation_checks = validation.get("checks") if isinstance(validation.get("checks"), list) else []
    evaluations["f407_postflash_validation_content"] = (
        bool(
            validation.get("schema_version") == POSTFLASH_VALIDATION_SCHEMA
            and validation.get("overall") == "PASS"
            and isinstance(validation.get("counts"), dict)
            and validation["counts"].get("FAIL") == 0
            and len(validation_checks) >= 14
            and all(isinstance(item, dict) and item.get("status") == "PASS" for item in validation_checks)
        ),
        f"schema={validation.get('schema_version')} overall={validation.get('overall')} counts={validation.get('counts')}",
    )

    if recovered:
        physical = (
            manifest.get("physical_interlock")
            if isinstance(manifest.get("physical_interlock"), dict)
            else {}
        )
        recovery = manifest.get("recovery") if isinstance(manifest.get("recovery"), dict) else {}
        source_artifacts = (
            source_failed_manifest.get("artifacts")
            if isinstance(source_failed_manifest.get("artifacts"), dict)
            else {}
        )
        source_tooling = (
            source_failed_manifest.get("tooling")
            if isinstance(source_failed_manifest.get("tooling"), dict)
            else {}
        )
        physical_source_path = Path(str(physical.get("source_manifest") or ""))
        physical_source_dir = physical_source_path.parent
        source_copy_entry = (
            artifacts.get("source_failed_manifest")
            if isinstance(artifacts.get("source_failed_manifest"), dict)
            else {}
        )
        source_copy_path = accept_dir / POSTFLASH_RECOVERY_ARTIFACT_FILES["source_failed_manifest"]
        source_copy_sha = sha256_file(source_copy_path) if source_copy_path.is_file() else ""
        source_started = float(source_failed_manifest.get("started_at_unix") or 0.0)
        source_finished = float(source_failed_manifest.get("finished_at_unix") or 0.0)
        recovery_started = float(recovery.get("started_at_unix") or 0.0)
        recovery_finished = float(recovery.get("finished_at_unix") or 0.0)
        source_fields_preserved = all(
            source_failed_manifest.get(key) == manifest.get(key)
            for key in (
                "hostname",
                "operator_confirmation",
                "command_contract",
                "serial_exclusivity",
                "service_restore",
            )
        )
        source_raw_chain_ok = True
        source_raw_details: list[str] = []
        for key in ("interlock_report", "interlock_log", "validation_report"):
            source_entry = source_artifacts.get(key) if isinstance(source_artifacts.get(key), dict) else {}
            recovered_entry = artifacts.get(key) if isinstance(artifacts.get(key), dict) else {}
            source_artifact_path = Path(str(source_entry.get("path") or ""))
            expected_name = expected_source_artifact_names[key]
            ok = bool(
                source_artifact_path.parent == physical_source_dir
                and source_artifact_path.name == expected_name
                and source_entry.get("exists") is True
                and source_entry.get("sha256") == recovered_entry.get("sha256")
                and int(source_entry.get("size_bytes") or -1)
                == int(recovered_entry.get("size_bytes") or -2)
            )
            source_raw_chain_ok = source_raw_chain_ok and ok
            source_raw_details.append(f"{key}={ok}")
        source_topics_absent = all(
            isinstance(source_artifacts.get(key), dict)
            and source_artifacts[key].get("exists") is False
            for key in ("post_firmware_topic", "post_identity_topic", "post_estop_topic")
        )
        source_tool_chain_ok = all(
            isinstance(source_tooling.get(key), dict)
            and isinstance(tooling.get(key), dict)
            and source_tooling[key] == tooling[key]
            and isinstance(physical.get("tool_snapshots"), dict)
            and isinstance(physical["tool_snapshots"].get(key), dict)
            and physical["tool_snapshots"][key].get("sha256") == source_tooling[key].get("sha256")
            for key in ("orchestrator", "link_test", "validator")
        )
        evaluations["f407_postflash_recovered_source_chain"] = (
            bool(
                source_failed_manifest.get("schema_version") == POSTFLASH_MANIFEST_SCHEMA_V1
                and source_failed_manifest.get("overall") == "FAIL"
                and source_failed_manifest.get("failure_reason")
                == POSTFLASH_RECOVERABLE_SOURCE_FAILURE
                and physical.get("source_schema_version") == POSTFLASH_MANIFEST_SCHEMA_V1
                and physical.get("source_overall") == "FAIL"
                and physical.get("source_failure_reason") == POSTFLASH_RECOVERABLE_SOURCE_FAILURE
                and physical.get("raw_interlock_validation") == "PASS"
                and physical_source_path.name == "postflash_interlock_manifest.json"
                and physical_source_dir.name.startswith("postflash_")
                and physical_source_dir.parent.name == "f407_postflash_acceptance"
                and Path(str(physical.get("source_manifest_copy") or "")).parent == source_dir
                and Path(str(physical.get("source_manifest_copy") or "")).name
                == "source_failed_manifest.json"
                and source_copy_entry.get("sha256") == source_copy_sha
                and physical.get("source_manifest_sha256") == source_copy_sha
                and physical.get("source_manifest_copy_sha256") == source_copy_sha
                and source_fields_preserved
                and source_started == started_at
                and 0.0 < source_finished <= recovery_started <= recovery_finished == finished_at
                and source_raw_chain_ok
                and source_topics_absent
                and source_tool_chain_ok
            ),
            (
                f"source_sha={source_copy_sha} source_fields_preserved={source_fields_preserved} "
                f"raw={source_raw_details} source_topics_absent={source_topics_absent} "
                f"source_tools={source_tool_chain_ok}"
            ),
        )

        recovery_pre = (
            recovery.get("service_state_pre")
            if isinstance(recovery.get("service_state_pre"), dict)
            else {}
        )
        recovery_post = (
            recovery.get("service_state_post")
            if isinstance(recovery.get("service_state_post"), dict)
            else {}
        )
        recovery_tool = recovery.get("tool") if isinstance(recovery.get("tool"), dict) else {}
        evaluations["f407_postflash_readonly_recovery_contract"] = (
            bool(
                recovery.get("schema_version") == POSTFLASH_RECOVERY_SCHEMA_V1
                and recovery.get("mode") == "post_restore_readonly_only"
                and recovery.get("serial_device_opened") is False
                and recovery.get("ros_graph_writes") is False
                and recovery.get("services_stopped") is False
                and recovery.get("services_started_or_restarted") is False
                and recovery.get("physical_commands_sent") is False
                and recovery_pre == recovery_post == post
                and recovery_tool == tooling.get("recovery")
            ),
            json.dumps(recovery, sort_keys=True),
        )

        recovery_checks = (
            recovery_validation.get("checks")
            if isinstance(recovery_validation.get("checks"), list)
            else []
        )
        evaluations["f407_postflash_recovery_revalidation_content"] = (
            bool(
                recovery_validation.get("schema_version") == POSTFLASH_VALIDATION_SCHEMA
                and recovery_validation.get("overall") == "PASS"
                and isinstance(recovery_validation.get("counts"), dict)
                and recovery_validation["counts"].get("FAIL") == 0
                and len(recovery_checks) >= 14
                and all(
                    isinstance(item, dict) and item.get("status") == "PASS"
                    for item in recovery_checks
                )
            ),
            (
                f"schema={recovery_validation.get('schema_version')} "
                f"overall={recovery_validation.get('overall')} "
                f"counts={recovery_validation.get('counts')}"
            ),
        )

    firmware_path = accept_dir / POSTFLASH_ARTIFACT_FILES["post_firmware_topic"]
    identity_path = accept_dir / POSTFLASH_ARTIFACT_FILES["post_identity_topic"]
    estop_path = accept_dir / POSTFLASH_ARTIFACT_FILES["post_estop_topic"]
    firmware_objects = load_json_objects_from_ros_text(firmware_path)
    post_firmware = firmware_objects[-1] if firmware_objects else {}
    identity_text = identity_path.read_text(encoding="utf-8", errors="replace") if identity_path.exists() else ""
    estop_text = estop_path.read_text(encoding="utf-8", errors="replace") if estop_path.exists() else ""
    post_identity_true = re.search(r"(?m)^\s*data:\s*true\s*$", identity_text) is not None
    post_estop_true = re.search(r"(?m)^\s*data:\s*true\s*$", estop_text) is not None
    evaluations["f407_postflash_post_restore_topics"] = (
        bool(
            post_firmware.get("protocol_version") == 2
            and (int(post_firmware.get("capabilities") or 0) & 0x003F) == 0x003F
            and post_firmware.get("build_id") == TARGET_F407_BUILD_ID
            and post_firmware.get("test_mode") == 0
            and post_firmware.get("hw_variant") == 1
            and post_firmware.get("identity_valid") is True
            and post_identity_true
            and post_estop_true
        ),
        f"firmware={post_firmware} identity_true={post_identity_true} estop_true={post_estop_true}",
    )

    for name, (ok, detail) in evaluations.items():
        add_result(results, name, "PASS" if ok else "FAIL", detail, str(manifest_path))
    all_ok = all(ok for ok, _detail in evaluations.values())
    add_result(
        results,
        "f407_postflash_orchestration_content",
        "PASS" if all_ok else "FAIL",
        f"collector gate OK; {sum(ok for ok, _detail in evaluations.values())}/{len(evaluations)} strict contracts passed",
        str(manifest_path),
    )


def is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def float_list_close(actual: Any, expected: list[float], tolerance: float = 1e-6) -> bool:
    return bool(
        isinstance(actual, list)
        and len(actual) == len(expected)
        and all(
            is_finite_number(value) and abs(float(value) - target) <= tolerance
            for value, target in zip(actual, expected)
        )
    )


def compact_stages(stages: list[int]) -> list[int]:
    compact: list[int] = []
    for stage in stages:
        if not compact or compact[-1] != stage:
            compact.append(stage)
    return compact


def fixture_expected_remaps(scope: str) -> dict[str, str]:
    return {
        "__ns": scope,
        "/cmd_vel": f"{scope}/cmd_vel_sink",
        "/set_lift_height": f"{scope}/set_lift_height",
        "/set_electromagnet": f"{scope}/set_electromagnet",
        "/lift_status": f"{scope}/lift_status",
        "/f407/estop_latched": f"{scope}/f407/estop_latched",
        "/f407/firmware_identity_valid": f"{scope}/f407/firmware_identity_valid",
        "/lab_fsd/safety_gate": f"{scope}/lab_fsd/safety_gate",
        "/lab_fsd/future_risk": f"{scope}/lab_fsd/future_risk",
        "/lab_fsd/input_status": f"{scope}/lab_fsd/input_status",
    }


def fixture_ros_parameter_value(value: bool | float) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def fixture_expected_command(executable: str, scope: str) -> list[str]:
    command = [executable, "--ros-args"]
    for name, value in FIXTURE_EXPECTED_PARAMETERS.items():
        command.extend(["-p", f"{name}:={fixture_ros_parameter_value(value)}"])
    for source, target in fixture_expected_remaps(scope).items():
        command.extend(["-r", f"{source}:={target}"])
    return command


def audit_dispatch_fixture_report(
    report: dict[str, Any],
    report_path: Path,
    accept_dir: Path,
    acceptance_start_unix: float,
) -> dict[str, tuple[bool, str]]:
    now = time.time()
    checks = report.get("checks") if isinstance(report.get("checks"), list) else []
    check_names = [str(item.get("name") or "") for item in checks if isinstance(item, dict)]
    check_contract_ok = bool(
        len(checks) == len(FIXTURE_EXPECTED_CHECK_NAMES)
        and len(check_names) == len(checks)
        and len(set(check_names)) == len(check_names)
        and set(check_names) == FIXTURE_EXPECTED_CHECK_NAMES
        and all(
            item.get("status") == "PASS"
            and isinstance(item.get("detail"), str)
            and bool(item.get("detail"))
            for item in checks
            if isinstance(item, dict)
        )
        and report.get("expected_check_names") == sorted(FIXTURE_EXPECTED_CHECK_NAMES)
        and report.get("check_contract_ok") is True
    )

    goals = report.get("goals") if isinstance(report.get("goals"), list) else []
    goal_ids = [str(goal.get("task_id") or "") for goal in goals if isinstance(goal, dict)]
    goals_shape_ok = bool(
        len(goals) == len(FIXTURE_EXPECTED_GOAL_IDS)
        and len(goal_ids) == len(goals)
        and goal_ids == FIXTURE_EXPECTED_GOAL_IDS
        and all(
            goal.get("task_type") == "pickup_fixture_stationary"
            and isinstance(goal.get("feedback"), list)
            and is_finite_number(goal.get("elapsed_s"))
            and float(goal.get("elapsed_s")) >= 0.0
            for goal in goals
            if isinstance(goal, dict)
        )
    )
    rejected_goals_ok = goals_shape_ok and all(
        goals[index].get("accepted") is False
        and goals[index].get("feedback") == []
        and "success" not in goals[index]
        for index in (0, 1, 3)
    )
    completed = goals[2] if goals_shape_ok else {}
    completed_message = str(completed.get("message") or "")
    structured_result_ok = bool(
        completed.get("accepted") is True
        and completed.get("status") == 4
        and completed.get("success") is True
        and completed.get("completion_class") == "f407_reported"
        and completed.get("actuator_sequence_completed") is True
        and completed.get("physical_completed") is False
        and completed.get("physical_confirmation") == ""
        and completed.get("base_motion_requested") is False
        and completed_message.startswith("F407_REPORTED_COMPLETED:")
        and "stationary_fixture=true" in completed_message
        and "dispatch_issued_base_motion=false" in completed_message
        and "physical_completed=false" in completed_message
        and is_finite_number(completed.get("server_elapsed_s"))
        and float(completed.get("server_elapsed_s")) >= 0.0
    )
    feedback = completed.get("feedback") if isinstance(completed.get("feedback"), list) else []
    feedback_shape_ok = bool(
        feedback
        and all(
            isinstance(item, dict)
            and set(item) == {"stage", "progress_pct", "stage_message"}
            and isinstance(item.get("stage"), int)
            and not isinstance(item.get("stage"), bool)
            and is_finite_number(item.get("progress_pct"))
            and isinstance(item.get("stage_message"), str)
            and bool(item.get("stage_message"))
            for item in feedback
        )
    )
    feedback_stages = [int(item["stage"]) for item in feedback] if feedback_shape_ok else []
    feedback_progress = [float(item["progress_pct"]) for item in feedback] if feedback_shape_ok else []
    feedback_ok = bool(
        feedback_shape_ok
        and compact_stages(feedback_stages) == FIXTURE_EXPECTED_STAGES
        and all(0.0 <= value <= 100.0 for value in feedback_progress)
        and all(a <= b for a, b in zip(feedback_progress, feedback_progress[1:]))
        and feedback_progress[-1] == 100.0
    )
    expected_zero_pose = {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "orientation_x": 0.0,
        "orientation_y": 0.0,
        "orientation_z": 0.0,
        "orientation_w": 1.0,
    }
    final_pose_ok = completed.get("final_pose") == expected_zero_pose

    safety = report.get("safety") if isinstance(report.get("safety"), dict) else {}
    safety_ok = bool(
        report.get("simulation_only") is True
        and report.get("real_hardware_touched") is False
        and report.get("physical_completed") is False
        and safety.get("dev_f407_opened") is False
        and safety.get("forbidden_device_opened") is False
        and safety.get("nav2_enabled") is False
        and safety.get("literal_cmd_vel_messages") == []
        and safety.get("remapped_cmd_vel_messages") == []
        and float_list_close(safety.get("lift_targets"), [0.02, 0.04])
        and safety.get("magnet_commands") == [True]
    )

    scope = str(report.get("scope") or "")
    executable_raw = str(report.get("dispatch_executable") or "")
    dds_environment = report.get("dds_environment") if isinstance(report.get("dds_environment"), dict) else {}
    domain_id = report.get("domain_id")
    private_domain_ok = bool(
        isinstance(domain_id, int)
        and not isinstance(domain_id, bool)
        and 120 <= domain_id <= 232
        and report.get("ros_localhost_only") is True
        and dds_environment
        == {
            "ROS_DOMAIN_ID": str(domain_id),
            "ROS_LOCALHOST_ONLY": "1",
            "CYCLONEDDS_URI": None,
        }
    )
    launch_contract_ok = bool(
        re.fullmatch(r"/dispatch_fixture_[0-9]+_[0-9]+", scope)
        and report.get("parameters") == FIXTURE_EXPECTED_PARAMETERS
        and report.get("remaps") == fixture_expected_remaps(scope)
        and report.get("command") == fixture_expected_command(executable_raw, scope)
    )

    artifacts = report.get("dispatch_artifacts") if isinstance(report.get("dispatch_artifacts"), dict) else {}
    executable_artifact = artifacts.get("executable") if isinstance(artifacts.get("executable"), dict) else {}
    module_artifact = artifacts.get("module") if isinstance(artifacts.get("module"), dict) else {}
    executable_sha = str(report.get("dispatch_executable_sha256") or "")
    module_raw = str(report.get("dispatch_module") or "")
    module_sha = str(report.get("dispatch_module_sha256") or "")
    executable_input_path = Path(executable_raw).expanduser() if executable_raw else Path()
    module_input_path = Path(module_raw).expanduser() if module_raw else Path()
    executable_path = executable_input_path.resolve() if executable_raw else Path()
    module_path = module_input_path.resolve() if module_raw else Path()
    artifacts_ok = bool(
        set(artifacts) == {"executable", "module"}
        and executable_artifact == {"path": executable_raw, "sha256": executable_sha}
        and module_artifact == {"path": module_raw, "sha256": module_sha}
        and executable_input_path.is_absolute()
        and module_input_path.is_absolute()
        and executable_path.is_file()
        and module_path.is_file()
        and executable_path.name == "dispatch_server"
        and module_path.name == "dispatch_server.py"
        and SHA256_RE.fullmatch(executable_sha)
        and SHA256_RE.fullmatch(module_sha)
        and sha256_file(executable_path) == executable_sha
        and sha256_file(module_path) == module_sha
        and executable_path != module_path
    )

    process = report.get("process") if isinstance(report.get("process"), dict) else {}
    process_pid = process.get("pid")
    process_started = float(process.get("started_at_unix") or 0.0)
    process_stop_requested = float(process.get("stop_requested_at_unix") or 0.0)
    process_exited = float(process.get("exited_at_unix") or 0.0)
    process_ok = bool(
        isinstance(process_pid, int)
        and not isinstance(process_pid, bool)
        and process_pid > 1
        and process.get("shutdown_requested") is True
        and process.get("signals_sent") == ["SIGINT"]
        and process.get("forced_kill") is False
        and process.get("unexpected_early_exit") is False
        and process.get("returncode") in (0, -2, -15)
        and process_started > 0.0
        and process_started <= process_stop_requested <= process_exited
        and report.get("error") == ""
    )

    monitor = report.get("monitor") if isinstance(report.get("monitor"), dict) else {}
    monitor_started = float(monitor.get("started_at_unix") or 0.0)
    first_sample = float(monitor.get("first_sample_at_unix") or 0.0)
    last_sample = float(monitor.get("last_sample_at_unix") or 0.0)
    monitor_stopped = float(monitor.get("stopped_at_unix") or 0.0)
    sample_count = int(monitor.get("sample_count") or 0)
    monitor_ok = bool(
        process_ok
        and monitor.get("policy_version") == FIXTURE_MONITOR_POLICY_VERSION
        and monitor.get("include_descendants") is True
        and monitor.get("forbidden_prefixes") == FIXTURE_FORBIDDEN_DEVICE_PREFIXES
        and monitor.get("root_pid") == process_pid
        and monitor.get("process_started_at_unix") == process_started
        and monitor.get("thread_error") == ""
        and monitor.get("observations") == []
        and monitor.get("observation_count") == 0
        and isinstance(monitor.get("pids_seen"), list)
        and process_pid in monitor.get("pids_seen", [])
        and isinstance(monitor.get("descendant_pids_seen"), list)
        and process_started <= monitor_started <= process_started + 0.2
        and monitor_started <= first_sample <= process_started + 0.2
        and last_sample >= process_exited
        and monitor_stopped >= last_sample
        and sample_count >= 2
        and monitor.get("proc_tree_scan_count") == sample_count
        and int(monitor.get("root_alive_sample_count") or 0) >= 1
        and int(monitor.get("fd_scan_count") or 0) >= 1
        and is_finite_number(monitor.get("interval_s"))
        and 0.0 < float(monitor.get("interval_s")) <= 0.05
    )

    dds_drain = report.get("dds_drain") if isinstance(report.get("dds_drain"), dict) else {}
    drain_before = dds_drain.get("callback_counts_before") if isinstance(dds_drain.get("callback_counts_before"), dict) else {}
    drain_after = dds_drain.get("callback_counts_after") if isinstance(dds_drain.get("callback_counts_after"), dict) else {}
    dds_drain_ok = bool(
        dds_drain.get("requested_s") == 0.75
        and is_finite_number(dds_drain.get("elapsed_s"))
        and float(dds_drain.get("elapsed_s")) >= 0.675
        and dds_drain.get("completed") is True
        and set(drain_before) == {"remapped_cmd_vel", "literal_cmd_vel", "feedback"}
        and set(drain_after) == set(drain_before)
        and drain_before.get("remapped_cmd_vel") == 0
        and drain_before.get("literal_cmd_vel") == 0
        and drain_after.get("remapped_cmd_vel") == 0
        and drain_after.get("literal_cmd_vel") == 0
        and isinstance(drain_before.get("feedback"), int)
        and isinstance(drain_after.get("feedback"), int)
        and drain_after.get("feedback", -1) >= drain_before.get("feedback", 0)
    )

    log_path = accept_dir / "dispatch_fixture_integration.log"
    log_info = report.get("log") if isinstance(report.get("log"), dict) else {}
    actual_log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    severity_lines = [
        line for line in actual_log_text.splitlines()
        if re.search(
            r"\[(?:ERROR|FATAL)\]|(?:^|\s)(?:ERROR|FATAL)(?::|\s)",
            line,
            re.IGNORECASE,
        )
    ]
    unexpected_severity_lines = [
        line for line in severity_lines
        if not any(marker in line for marker in FIXTURE_EXPECTED_LOG_ERROR_MARKERS)
    ]
    fatal_text_markers = [
        "Traceback (most recent call last)",
        "RCLError",
        "ExternalShutdownException",
        "Exception in thread",
        "Exception ignored in",
        "uncaught exception",
        "Segmentation fault",
    ]
    log_content_ok = bool(
        log_path.is_file()
        and not unexpected_severity_lines
        and not any(marker.lower() in actual_log_text.lower() for marker in fatal_text_markers)
        and all(
            sum(marker in line for line in actual_log_text.splitlines()) == 1
            for marker in FIXTURE_EXPECTED_LOG_ERROR_MARKERS
        )
    )
    log_hash = sha256_file(log_path) if log_path.is_file() else ""
    log_evidence_ok = bool(
        log_content_ok
        and Path(str(log_info.get("path") or "")).expanduser().resolve() == log_path.resolve()
        and Path(str(report.get("log_path") or "")).expanduser().resolve() == log_path.resolve()
        and log_info.get("sha256") == log_hash
        and report.get("log_sha256") == log_hash
        and log_info.get("size_bytes") == log_path.stat().st_size
        and log_info.get("clean") is True
        and log_info.get("unexpected_errors") == []
        and log_info.get("expected_error_markers") == FIXTURE_EXPECTED_LOG_ERROR_MARKERS
        and log_info.get("expected_error_markers_found") == FIXTURE_EXPECTED_LOG_ERROR_MARKERS
        and log_info.get("expected_error_marker_counts")
        == {marker: 1 for marker in FIXTURE_EXPECTED_LOG_ERROR_MARKERS}
        and isinstance(log_info.get("expected_errors"), list)
        and len(log_info.get("expected_errors")) == len(FIXTURE_EXPECTED_LOG_ERROR_MARKERS)
    )

    report_started = float(report.get("started_at_unix") or 0.0)
    report_generated = float(report.get("generated_at_unix") or 0.0)
    freshness_ok = bool(
        acceptance_start_unix > 0.0
        and report_path.resolve() == (accept_dir / "dispatch_fixture_integration.json").resolve()
        and report_path.is_file()
        and report_path.stat().st_mtime >= acceptance_start_unix - 1.0
        and report_started >= acceptance_start_unix - 1.0
        and report_started <= process_started <= report_generated
        and report_started <= report_generated <= now + 5.0
        and 0.0 <= now - report_generated <= 86400.0
        and log_path.is_file()
        and log_path.parent.resolve() == accept_dir.resolve()
        and log_path.stat().st_mtime >= acceptance_start_unix - 1.0
    )

    schema_policy_ok = bool(
        report.get("schema_version") == FIXTURE_SCHEMA_VERSION
        and report.get("policy_version") == FIXTURE_POLICY_VERSION
        and report.get("overall") == "PASS"
    )
    return {
        "dispatch_fixture_schema_policy": (
            schema_policy_ok,
            f"schema={report.get('schema_version')} policy={report.get('policy_version')} overall={report.get('overall')}",
        ),
        "dispatch_fixture_freshness": (
            freshness_ok,
            f"acceptance_start={acceptance_start_unix} report_started={report_started} generated={report_generated}",
        ),
        "dispatch_fixture_check_contract": (
            check_contract_ok,
            f"actual={check_names} expected={sorted(FIXTURE_EXPECTED_CHECK_NAMES)}",
        ),
        "dispatch_fixture_goal_contract": (
            goals_shape_ok and rejected_goals_ok and structured_result_ok,
            f"goal_ids={goal_ids} rejected={rejected_goals_ok} structured={structured_result_ok}",
        ),
        "dispatch_fixture_feedback_contract": (
            feedback_ok,
            f"raw_stages={feedback_stages} compact={compact_stages(feedback_stages)} progress={feedback_progress}",
        ),
        "dispatch_fixture_actuator_safety": (
            safety_ok and final_pose_ok,
            f"safety={safety} final_pose={completed.get('final_pose')}",
        ),
        "dispatch_fixture_private_domain": (
            private_domain_ok,
            f"domain_id={domain_id} dds_environment={dds_environment}",
        ),
        "dispatch_fixture_launch_contract": (
            launch_contract_ok,
            f"parameters={report.get('parameters')} remaps={report.get('remaps')}",
        ),
        "dispatch_fixture_artifact_hashes": (
            artifacts_ok,
            f"executable={executable_raw} module={module_raw}",
        ),
        "dispatch_fixture_monitor_evidence": (
            monitor_ok,
            f"samples={sample_count} fd_scans={monitor.get('fd_scan_count')} observations={monitor.get('observations')}",
        ),
        "dispatch_fixture_process_log_evidence": (
            process_ok and dds_drain_ok and log_evidence_ok,
            f"process={process} dds_drain={dds_drain} unexpected_log_lines={unexpected_severity_lines}",
        ),
    }


def audit_acceptance_dir(accept_dir: Path, data_run_dir: Path | None = None) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    checks_path = accept_dir / "checks.tsv"
    checks = load_checks(checks_path)
    if not checks:
        add_result(results, "acceptance_checks_file", "FAIL", "checks.tsv missing or empty", str(checks_path))
        return {"path": accept_dir.as_posix(), "results": results}

    acceptance_start_path = accept_dir / "acceptance_start.json"
    acceptance_start_unix = 0.0
    try:
        acceptance_start = json.loads(acceptance_start_path.read_text(encoding="utf-8"))
    except Exception as exc:
        add_result(
            results,
            "acceptance_start_content",
            "FAIL",
            f"missing or invalid acceptance start marker: {exc}",
            str(acceptance_start_path),
        )
    else:
        acceptance_start_unix = float(acceptance_start.get("started_at_unix") or 0.0)
        acceptance_age_s = time.time() - acceptance_start_unix if acceptance_start_unix > 0.0 else float("inf")
        marker_out_dir = Path(str(acceptance_start.get("out_dir") or "")).expanduser().resolve()
        marker_ok = bool(
            acceptance_start.get("schema_version") == "xrd-embodied-v3-acceptance-start-v1"
            and marker_out_dir == accept_dir.resolve()
            and acceptance_start_unix > 0.0
            and -5.0 <= acceptance_age_s <= 86400.0
            and acceptance_start_path.stat().st_mtime >= acceptance_start_unix - 1.0
            and isinstance(acceptance_start.get("pid"), int)
            and not isinstance(acceptance_start.get("pid"), bool)
            and acceptance_start.get("pid") > 1
        )
        add_result(
            results,
            "acceptance_start_content",
            "PASS" if marker_ok else "FAIL",
            f"schema={acceptance_start.get('schema_version')} age_s={acceptance_age_s:.1f} out_dir={marker_out_dir}",
            str(acceptance_start_path),
        )

    for name, detail in REQUIRED_ACCEPTANCE_CHECKS.items():
        item = checks.get(name)
        if item is None:
            add_result(results, name, "FAIL", f"missing required check: {detail}", str(checks_path))
        elif item["status"] == "OK":
            add_result(results, name, "PASS", detail, item.get("detail", ""))
        else:
            add_result(results, name, "FAIL", f"{detail}; status={item['status']}", item.get("detail", ""))

    for name, detail in WARN_ACCEPTANCE_CHECKS.items():
        item = checks.get(name)
        if item is None:
            add_result(results, name, "WARN", f"missing optional check: {detail}", str(checks_path))
        elif item["status"] == "OK":
            add_result(results, name, "PASS", detail, item.get("detail", ""))
        else:
            add_result(results, name, "WARN", f"{detail}; status={item['status']}", item.get("detail", ""))

    physical_config_path = accept_dir / "physical_evidence_config.txt"
    physical_config_text = (
        physical_config_path.read_text(encoding="utf-8", errors="replace")
        if physical_config_path.exists()
        else ""
    )
    mode_match = re.search(
        r"String value is:\s*(disabled|report_only|required)\b",
        physical_config_text,
    )
    physical_mode = mode_match.group(1) if mode_match else ""
    gate_present = re.search(r"(?m)^/physical_evidence_gate\s*$", physical_config_text) is not None
    sensor_bridge_present = re.search(
        r"(?m)^/physical_sensor_evidence_bridge(?:[_.-][A-Za-z0-9_.-]+)?\s*$",
        physical_config_text,
    ) is not None
    service_present = "my_robot_msgs/srv/VerifyPhysicalEvidence" in physical_config_text
    raw_sample_topic_present = re.search(
        r"(?m)^/pickup/hardware_sensor_sample\s+\[my_robot_msgs/msg/HardwareSensorSample\]\s*$",
        physical_config_text,
    ) is not None
    evidence_topic_present = re.search(
        r"(?m)^/pickup/physical_evidence\s+\[my_robot_msgs/msg/PhysicalEvidence\]\s*$",
        physical_config_text,
    ) is not None
    request_topic_present = re.search(
        r"(?m)^/pickup/physical_evidence_request\s+\[my_robot_msgs/msg/PhysicalEvidenceRequest\]\s*$",
        physical_config_text,
    ) is not None
    status_topic_present = re.search(
        r"(?m)^/pickup/physical_evidence_bridge_status\s+\[std_msgs/msg/String\]\s*$",
        physical_config_text,
    ) is not None
    physical_topics_present = any(
        (
            raw_sample_topic_present,
            evidence_topic_present,
            request_topic_present,
            status_topic_present,
        )
    )
    if (
        physical_mode == "disabled"
        and not gate_present
        and not sensor_bridge_present
        and not service_present
        and not physical_topics_present
    ):
        add_result(
            results,
            "physical_evidence_default_safe",
            "PASS",
            "production mode disabled; no evidence gate, sensor bridge, service, or evidence topic can create a physical completion claim",
            str(physical_config_path),
        )
    elif (
        physical_mode in {"report_only", "required"}
        and gate_present
        and sensor_bridge_present
        and service_present
        and raw_sample_topic_present
        and evidence_topic_present
        and request_topic_present
        and status_topic_present
    ):
        add_result(
            results,
            "physical_evidence_default_safe",
            "PASS",
            f"mode={physical_mode}; gate, calibrated sensor bridge, typed service, and evidence topics are online",
            str(physical_config_path),
        )
    else:
        add_result(
            results,
            "physical_evidence_default_safe",
            "FAIL",
            "inconsistent physical evidence configuration: "
            f"mode={physical_mode or 'missing'} gate={gate_present} "
            f"sensor_bridge={sensor_bridge_present} service={service_present} "
            f"raw={raw_sample_topic_present} evidence={evidence_topic_present} "
            f"request={request_topic_present} status={status_topic_present}",
            str(physical_config_path),
        )

    if check_file_contains(accept_dir / "hobot_dnn_import.txt", "hobot_dnn import OK"):
        add_result(results, "hobot_dnn_import_content", "PASS", "hobot_dnn import output confirms runtime", "hobot_dnn_import.txt")
    else:
        add_result(results, "hobot_dnn_import_content", "FAIL", "hobot_dnn import output did not contain OK marker", "hobot_dnn_import.txt")

    fsd_status_file = accept_dir / "topic_lab_fsd_fsd_v3_status.txt"
    fsd_status_text = fsd_status_file.read_text(encoding="utf-8", errors="replace").lower() if fsd_status_file.exists() else ""
    fsd_status_objects = load_json_objects_from_ros_text(fsd_status_file)
    if fsd_status_objects:
        fsd_status = fsd_status_objects[-1]
        bpu = fsd_status.get("bpu") if isinstance(fsd_status.get("bpu"), dict) else {}
        tiny = bpu.get("tiny_occ_risk") if isinstance(bpu.get("tiny_occ_risk"), dict) else {}
        tiny_state = tiny.get("state")
        tiny_ok = tiny_state in {"runtime_ready", "forward_ok"}
        if tiny_ok:
            add_result(results, "fsd_v3_status_tiny_occ_risk_content", "PASS", f"tiny_occ_risk state={tiny_state}", str(fsd_status_file))
        else:
            add_result(results, "fsd_v3_status_tiny_occ_risk_content", "FAIL", f"tiny_occ_risk runtime state invalid: {tiny_state}", str(fsd_status_file))
        shadow_ok = fsd_status.get("shadow_only") is True and fsd_status.get("cmd_vel_authority") is False
        if shadow_ok:
            add_result(results, "fsd_v3_status_shadow_only_content", "PASS", "parsed JSON confirms shadow_only=true and cmd_vel_authority=false", str(fsd_status_file))
        else:
            add_result(results, "fsd_v3_status_shadow_only_content", "FAIL", "parsed JSON does not enforce shadow-only no-authority state", str(fsd_status_file))
        policy_prior = bpu.get("policy_prior") if isinstance(bpu.get("policy_prior"), dict) else {}
        if policy_prior.get("source"):
            add_result(results, "fsd_v3_status_policy_prior_content", "PASS", f"policy_prior source={policy_prior.get('source')}", str(fsd_status_file))
        else:
            add_result(results, "fsd_v3_status_policy_prior_content", "WARN", "parsed status has no BPU policy_prior source", str(fsd_status_file))
    else:
        add_result(results, "fsd_v3_status_tiny_occ_risk_content", "FAIL", "fsd_v3_status JSON payload could not be parsed", str(fsd_status_file))
        add_result(results, "fsd_v3_status_shadow_only_content", "FAIL", "fsd_v3_status JSON payload could not be parsed", str(fsd_status_file))
        add_result(results, "fsd_v3_status_policy_prior_content", "WARN", "fsd_v3_status JSON payload could not be parsed", str(fsd_status_file))

    input_status_file = accept_dir / "topic_lab_fsd_input_status.txt"
    input_status_objects = load_json_objects_from_ros_text(input_status_file)
    if input_status_objects:
        input_status = input_status_objects[-1]
        sources = input_status.get("sources") if isinstance(input_status.get("sources"), dict) else {}
        geometry_live = all(
            isinstance(sources.get(name), dict)
            and sources[name].get("state") == "live"
            and sources[name].get("usable") is True
            for name in ("scan", "scan_depth")
        )
        if input_status.get("overall") == "live" and geometry_live:
            add_result(results, "lab_fsd_geometry_input_truth", "PASS", "LiDAR and depth are explicitly live/usable", str(input_status_file))
        else:
            add_result(results, "lab_fsd_geometry_input_truth", "FAIL", f"geometry input state is not live: {sources}", str(input_status_file))

        vision = sources.get("vision_bev") if isinstance(sources.get("vision_bev"), dict) else {}
        provenance = vision.get("provenance") if isinstance(vision.get("provenance"), dict) else {}
        vision_state = vision.get("state")
        provenance_state = provenance.get("state")
        truthful = False
        if vision_state == "live":
            truthful = provenance_state == "live_camera" and provenance.get("image_supplied") is True
        elif vision_state == "cached":
            truthful = provenance_state == "cached_camera" and vision.get("fresh") is False
        elif vision_state == "fallback":
            truthful = provenance_state == "fixture_prior" and vision.get("fresh") is False
        elif vision_state == "unverified":
            truthful = provenance_state == "unknown" and vision.get("usable") is False
        elif vision_state in {"stale", "offline", "disabled"}:
            truthful = vision.get("fresh") is False
        if truthful:
            add_result(results, "vision_bev_provenance_truthful", "PASS", f"vision state={vision_state} provenance={provenance_state}", str(input_status_file))
        else:
            add_result(results, "vision_bev_provenance_truthful", "FAIL", f"vision state/provenance mismatch: state={vision_state} provenance={provenance}", str(input_status_file))
    else:
        add_result(results, "lab_fsd_geometry_input_truth", "FAIL", "input_status JSON payload could not be parsed", str(input_status_file))
        add_result(results, "vision_bev_provenance_truthful", "FAIL", "input_status JSON payload could not be parsed", str(input_status_file))

    services_file = accept_dir / "ros_services.txt"
    services_text = services_file.read_text(encoding="utf-8", errors="replace") if services_file.exists() else ""
    for service_name, detail in sorted(REQUIRED_RUNTIME_SERVICES.items()):
        if service_name in services_text:
            add_result(results, f"runtime_service:{service_name}", "PASS", detail, str(services_file))
        else:
            add_result(results, f"runtime_service:{service_name}", "FAIL", f"missing service: {detail}", str(services_file))

    diagnostics_file = accept_dir / "topic_diagnostics.txt"
    diagnostics_text = diagnostics_file.read_text(encoding="utf-8", errors="replace") if diagnostics_file.exists() else ""
    for key, detail in sorted(REQUIRED_DIAGNOSTIC_KEYS.items()):
        if key in diagnostics_text:
            add_result(results, f"diagnostics_key:{key}", "PASS", detail, str(diagnostics_file))
        else:
            add_result(results, f"diagnostics_key:{key}", "FAIL", f"missing diagnostics content: {detail}", str(diagnostics_file))

    firmware_info_path = accept_dir / "topic_f407_firmware_info.txt"
    firmware_info_objects = load_json_objects_from_ros_text(firmware_info_path)
    firmware_info = firmware_info_objects[-1] if firmware_info_objects else {}
    firmware_topic_ok = bool(
        firmware_info.get("protocol_version") == 2
        and (int(firmware_info.get("capabilities") or 0) & 0x003F) == 0x003F
        and firmware_info.get("build_id") == TARGET_F407_BUILD_ID
        and firmware_info.get("test_mode") == 0
        and firmware_info.get("hw_variant") == 1
        and firmware_info.get("identity_valid") is True
        and firmware_info.get("required") is True
        and firmware_info.get("identity_enforcement_enabled") is True
        and firmware_info.get("cmd_vel_authority_when_invalid") is False
    )
    if firmware_topic_ok:
        add_result(results, "f407_firmware_identity_topic_content", "PASS", "firmware identity topic exactly matches target build and safety capabilities", str(firmware_info_path))
    else:
        add_result(results, "f407_firmware_identity_topic_content", "FAIL", f"invalid or missing firmware identity JSON: {firmware_info}", str(firmware_info_path))

    firmware_valid_path = accept_dir / "topic_f407_firmware_identity_valid.txt"
    firmware_valid_text = firmware_valid_path.read_text(encoding="utf-8", errors="replace").lower() if firmware_valid_path.exists() else ""
    if re.search(r"(?m)^\s*data:\s*true\s*$", firmware_valid_text):
        add_result(results, "f407_firmware_identity_valid_content", "PASS", "driver confirms fresh exact firmware identity", str(firmware_valid_path))
    else:
        add_result(results, "f407_firmware_identity_valid_content", "FAIL", "firmware identity-valid topic is absent or false", str(firmware_valid_path))

    interlock_path = accept_dir / "f407_interlock_report.json"
    if interlock_path.exists():
        try:
            interlock = load_json(interlock_path)
        except Exception as exc:
            add_result(results, "f407_interlock_report_content", "FAIL", f"invalid JSON: {exc}", str(interlock_path))
        else:
            ack_checks = interlock.get("ack_checks") if isinstance(interlock.get("ack_checks"), list) else []
            by_label = {
                str(item.get("label")): item
                for item in ack_checks
                if isinstance(item, dict) and item.get("label")
            }
            expected = {
                "EMERGENCY_STOP": 0,
                "SET_LIFT_HEIGHT blocked": 3,
                "SET_ELECTROMAGNET ON blocked": 3,
                "SET_ELECTROMAGNET OFF allowed": 0,
                "LIFT_HOME blocked": 3,
            }
            ack_ok = all(
                label in by_label
                and by_label[label].get("status") == status
                and by_label[label].get("expected_status") == status
                and by_label[label].get("ok") is True
                for label, status in expected.items()
            ) and set(by_label) == set(expected) and len(ack_checks) == len(expected)
            safety = interlock.get("safety") if isinstance(interlock.get("safety"), dict) else {}
            last_safety = interlock.get("last_safety") if isinstance(interlock.get("last_safety"), dict) else {}
            stats = interlock.get("stats") if isinstance(interlock.get("stats"), dict) else {}
            firmware = interlock.get("firmware_identity") if isinstance(interlock.get("firmware_identity"), dict) else {}
            generated_at = float(interlock.get("generated_at_unix") or 0.0)
            age_s = time.time() - generated_at if generated_at > 0.0 else float("inf")
            report_ok = bool(
                interlock.get("schema_version") == "xrd-f407-interlock-evidence-v2"
                and interlock.get("overall") == "PASS"
                and interlock.get("verify_estop_interlock") is True
                and interlock.get("clear_estop_requested") is False
                and interlock.get("ack_failures") == []
                and safety.get("motion_cmd_sent") is False
                and safety.get("interlock_test_sends_no_nonzero_cmd_vel") is True
                and safety.get("default_leaves_estop_latched") is True
                and safety.get("serial_exclusive_open") is True
                and safety.get("identity_verified_before_commands") is True
                and safety.get("commands_started") is True
                and firmware.get("valid") is True
                and firmware.get("protocol_version") == 2
                and (int(firmware.get("capabilities") or 0) & 0x003F) == 0x003F
                and firmware.get("build_id") == TARGET_F407_BUILD_ID
                and firmware.get("test_mode") == 0
                and firmware.get("hw_variant") == 1
                and ack_ok
                and last_safety.get("estop_latched") is True
                and int(last_safety.get("blocked_command_count") or 0) >= 3
                and int(stats.get("SAFETY") or 0) > 0
                and 0.0 <= age_s <= 86400.0
            )
            if report_ok:
                add_result(results, "f407_interlock_report_content", "PASS", f"firmware blocked lift/magnet-on/home with ACK=3 and remained latched; age_s={age_s:.1f}", str(interlock_path))
            else:
                add_result(results, "f407_interlock_report_content", "FAIL", f"interlock evidence incomplete or unsafe: ack_ok={ack_ok}", str(interlock_path))
    else:
        add_result(results, "f407_interlock_report_content", "FAIL", "no post-flash firmware interlock evidence captured", str(interlock_path))

    audit_postflash_orchestration(results, accept_dir, checks, acceptance_start_unix)

    runtime_prepare_check = checks.get("runtime_prepare_report")
    runtime_prepare_path = accept_dir / "runtime_prepare_report.json"
    if runtime_prepare_path.exists():
        try:
            runtime_prepare = load_json(runtime_prepare_path)
        except Exception as exc:
            add_result(results, "runtime_prepare_report_content", "FAIL", f"invalid JSON: {exc}", str(runtime_prepare_path))
        else:
            reference_path = accept_dir / "data_run_reference.txt"
            referenced_run = reference_path.read_text(encoding="utf-8", errors="replace").strip() if reference_path.exists() else ""
            manifest_path = (data_run_dir / "manifest.json") if data_run_dir is not None else (Path(referenced_run) / "manifest.json" if referenced_run else Path())
            manifest_sha = sha256_file(manifest_path) if manifest_path.is_file() else ""
            interlock_sha = sha256_file(interlock_path) if interlock_path.is_file() else ""
            cmd_report_path = (data_run_dir / "logs" / "cmd_vel_evidence.json") if data_run_dir is not None else Path()
            try:
                runtime_cmd_report = load_json(cmd_report_path) if cmd_report_path.is_file() else {}
            except Exception:
                runtime_cmd_report = {}
            runtime_cmd_counts = runtime_cmd_report.get("counts") if isinstance(runtime_cmd_report.get("counts"), dict) else {}
            runtime_cmd_zero_verified = bool(
                runtime_cmd_report.get("schema_version") == "xrd-cmd-vel-bag-evidence-v1"
                and runtime_cmd_report.get("status") == "PASS"
                and runtime_cmd_report.get("expectation") == "zero"
                and runtime_cmd_report.get("topic") == "/cmd_vel"
                and runtime_cmd_report.get("topic_type") == "geometry_msgs/msg/Twist"
                and integer(runtime_cmd_counts.get("message_count")) > 0
                and integer(runtime_cmd_counts.get("decoded_count")) == integer(runtime_cmd_counts.get("message_count"))
                and integer(runtime_cmd_counts.get("nonzero_count")) == 0
            )
            generated_at = float(runtime_prepare.get("generated_at_unix") or 0.0)
            age_s = time.time() - generated_at if generated_at > 0.0 else float("inf")
            runtime_schema = runtime_prepare.get("schema_version")
            runtime_postflash_manifest = accept_dir / "runtime_postflash_manifest.json"
            runtime_postflash_index = accept_dir / "runtime_postflash_bundle_index.json"
            acceptance_postflash_manifest = accept_dir / "f407_postflash_manifest.json"
            interlock_source_mode = str(runtime_prepare.get("interlock_source_mode") or "")
            if runtime_schema == "xrd-embodied-v3-runtime-prepare-v3":
                interlock_source_ok = True
            elif runtime_schema == "xrd-embodied-v3-runtime-prepare-v4" and interlock_source_mode == "direct_interlock":
                interlock_source_ok = bool(
                    not runtime_prepare.get("postflash_manifest")
                    and not runtime_prepare.get("postflash_manifest_sha256")
                    and not runtime_prepare.get("postflash_bundle_index")
                    and not runtime_prepare.get("postflash_bundle_index_sha256")
                )
            elif runtime_schema == "xrd-embodied-v3-runtime-prepare-v4" and interlock_source_mode == "postflash_manifest_reuse":
                interlock_source_ok = bool(
                    runtime_postflash_manifest.is_file()
                    and runtime_postflash_index.is_file()
                    and acceptance_postflash_manifest.is_file()
                    and runtime_prepare.get("postflash_manifest_sha256") == sha256_file(runtime_postflash_manifest)
                    and runtime_prepare.get("postflash_bundle_index_sha256") == sha256_file(runtime_postflash_index)
                    and sha256_file(runtime_postflash_manifest) == sha256_file(acceptance_postflash_manifest)
                    and checks.get("f407_postflash_manifest", {}).get("status") == "OK"
                )
            else:
                interlock_source_ok = False
            content_ok = bool(
                runtime_schema in {
                    "xrd-embodied-v3-runtime-prepare-v3",
                    "xrd-embodied-v3-runtime-prepare-v4",
                }
                and interlock_source_ok
                and runtime_prepare_check is not None
                and runtime_prepare_check.get("status") == "OK"
                and referenced_run
                and runtime_prepare.get("data_run") == referenced_run
                and manifest_sha
                and runtime_prepare.get("data_manifest_sha256") == manifest_sha
                and interlock_sha
                and runtime_prepare.get("interlock_report_sha256") == interlock_sha
                and runtime_prepare.get("runtime_mode") == "shadow_plus_mppi_proposed_only"
                and runtime_prepare.get("cmd_vel_capture") == "zero_twist_only"
                and runtime_prepare.get("cmd_vel_topic") == "/cmd_vel"
                and runtime_prepare.get("cmd_vel_message") == {}
                and runtime_prepare.get("nonzero_cmd_vel_published") is False
                and runtime_cmd_zero_verified
                and runtime_prepare.get("f407_estop_left_latched") is True
                and isinstance(runtime_prepare.get("mppi"), dict)
                and runtime_prepare["mppi"].get("enabled") is True
                and runtime_prepare["mppi"].get("use_bpu_required") is True
                and runtime_prepare["mppi"].get("proposed_only") is True
                and runtime_prepare["mppi"].get("proposed_topic") == "/mppi/cmd_vel_proposed"
                and runtime_prepare["mppi"].get("direct_cmd_vel") is False
                and runtime_prepare["mppi"].get("f407_estop_required") is True
                and 0.0 <= age_s <= 86400.0
            )
            mppi_raw_paths = {
                "stats_sha256": accept_dir / "runtime_mppi_stats.txt",
                "proposed_sha256": accept_dir / "runtime_mppi_cmd_vel_proposed.txt",
                "cmd_vel_publishers_sha256": accept_dir / "runtime_mppi_cmd_vel_publishers.txt",
            }
            mppi_raw_ok = all(
                path.is_file()
                and runtime_prepare["mppi"].get(hash_key) == sha256_file(path)
                for hash_key, path in mppi_raw_paths.items()
            ) if isinstance(runtime_prepare.get("mppi"), dict) else False
            content_ok = content_ok and mppi_raw_ok
            if content_ok:
                source_label = interlock_source_mode or "legacy_v3_direct_interlock"
                add_result(results, "runtime_prepare_report_content", "PASS", f"fresh zero-Twist shadow + BPU MPPI proposed-only run is bound to manifest, interlock ({source_label}), and raw MPPI hashes; age_s={age_s:.1f}", str(runtime_prepare_path))
            else:
                detail = (
                    f"runtime preparation binding invalid: check={runtime_prepare_check} "
                    f"run_match={runtime_prepare.get('data_run') == referenced_run} "
                    f"manifest_match={runtime_prepare.get('data_manifest_sha256') == manifest_sha} "
                    f"interlock_match={runtime_prepare.get('interlock_report_sha256') == interlock_sha} "
                    f"interlock_source_mode={interlock_source_mode or 'legacy_v3'} "
                    f"interlock_source_ok={interlock_source_ok} "
                    f"cmd_vel_zero_verified={runtime_cmd_zero_verified} mppi_raw_ok={mppi_raw_ok}"
                )
                add_result(results, "runtime_prepare_report_content", "FAIL", detail, str(runtime_prepare_path))
    elif runtime_prepare_check is not None and runtime_prepare_check.get("status") == "FAIL":
        add_result(results, "runtime_prepare_report_content", "FAIL", runtime_prepare_check.get("detail", "required runtime preparation report missing"), str(runtime_prepare_path))
    else:
        add_result(results, "runtime_prepare_report_content", "WARN", "runtime preparation report not required for this read-only acceptance check", str(runtime_prepare_path))

    mppi_stats_file = accept_dir / "topic_mppi_stats.txt"
    mppi_stats_text = mppi_stats_file.read_text(encoding="utf-8", errors="replace").lower() if mppi_stats_file.exists() else ""
    mppi_objects = load_json_objects_from_ros_text(mppi_stats_file)
    if mppi_objects:
        mppi = mppi_objects[-1]
        mppi_contract_ok = bool(
            mppi.get("proposed_only") is True
            and mppi.get("proposed_topic") == "/mppi/cmd_vel_proposed"
            and mppi.get("direct_cmd_vel") is False
            and mppi.get("use_bpu") is True
            and mppi.get("estop_latched") is True
            and str(mppi.get("direct_block_reason") or "").startswith("f407_estop_latched")
        )
        if mppi_contract_ok:
            detail = f"BPU proposed-only; eval_ms={mppi.get('eval_ms')} selected=({mppi.get('v_lin')},{mppi.get('v_ang')}) estop-gated"
            add_result(results, "mppi_stats_proposed_only_content", "PASS", detail, str(mppi_stats_file))
        else:
            add_result(results, "mppi_stats_proposed_only_content", "FAIL", f"invalid MPPI safety contract: {mppi}", str(mppi_stats_file))
    elif mppi_stats_text:
        add_result(results, "mppi_stats_proposed_only_content", "WARN", "MPPI stats captured but JSON could not be parsed", str(mppi_stats_file))
    else:
        add_result(results, "mppi_stats_proposed_only_content", "WARN", "MPPI stats not captured; MPPI may be disabled in this run", str(mppi_stats_file))

    mppi_proposed_file = accept_dir / "topic_mppi_cmd_vel_proposed.txt"
    proposed_text = mppi_proposed_file.read_text(encoding="utf-8", errors="replace") if mppi_proposed_file.exists() else ""
    proposed_values = [
        float(value)
        for value in re.findall(r"(?m)^\s+[xyz]:\s*([-+0-9.eE]+)\s*$", proposed_text)
    ]
    if len(proposed_values) >= 6:
        if all(abs(value) <= 1e-9 for value in proposed_values[:6]):
            add_result(results, "mppi_proposed_estop_zero_content", "PASS", "all proposed Twist components are zero while F407 estop is latched", str(mppi_proposed_file))
        else:
            add_result(results, "mppi_proposed_estop_zero_content", "FAIL", f"non-zero proposed Twist while estop-gated: {proposed_values[:6]}", str(mppi_proposed_file))
    elif "mppi node absent" in proposed_text.lower():
        add_result(results, "mppi_proposed_estop_zero_content", "WARN", "MPPI node absent in normal verify; runtime/raw MPPI evidence is audited separately", str(mppi_proposed_file))
    elif proposed_text:
        add_result(results, "mppi_proposed_estop_zero_content", "FAIL", "MPPI proposed Twist could not be parsed", str(mppi_proposed_file))
    else:
        add_result(results, "mppi_proposed_estop_zero_content", "WARN", "MPPI proposed Twist not captured", str(mppi_proposed_file))

    policy_file = accept_dir / "topic_lab_fsd_policy_tokens.txt"
    policy_text = policy_file.read_text(encoding="utf-8", errors="replace").lower() if policy_file.exists() else ""
    if policy_text and "policy_prior" in policy_text and "tiny_waypoint_policy_prior" in policy_text and "cmd_vel_authority" in policy_text and "false" in policy_text:
        add_result(results, "policy_tokens_prior_content", "PASS", "policy_tokens include fused tiny waypoint prior and no cmd_vel authority", str(policy_file))
    elif policy_text:
        add_result(results, "policy_tokens_prior_content", "WARN", "policy_tokens captured but fused policy_prior/no-authority markers were incomplete", str(policy_file))
    else:
        add_result(results, "policy_tokens_prior_content", "WARN", "policy_tokens not captured; planner may be offline", str(policy_file))
    policy_objects = load_json_objects_from_ros_text(policy_file)
    if policy_objects:
        policy_obj = policy_objects[-1]
        prior = policy_obj.get("policy_prior") if isinstance(policy_obj.get("policy_prior"), dict) else {}
        probabilities = prior.get("probabilities") if isinstance(prior, dict) else None
        token_count = prior.get("token_count") if isinstance(prior, dict) else None
        authority_ok = policy_obj.get("cmd_vel_authority") is False and prior.get("cmd_vel_authority") is False
        name_ok = prior.get("name") == "tiny_waypoint_policy_prior"
        token_ok = isinstance(token_count, int) and token_count >= 5
        prob_ok = probabilities is None or (isinstance(probabilities, list) and len(probabilities) == token_count)
        if name_ok and authority_ok and token_ok and prob_ok:
            detail = f"name={prior.get('name')} used_bpu={prior.get('used_bpu')} token_count={token_count}"
            add_result(results, "policy_tokens_prior_json_content", "PASS", detail, str(policy_file))
        else:
            detail = f"name_ok={name_ok} authority_ok={authority_ok} token_ok={token_ok} prob_ok={prob_ok}"
            add_result(results, "policy_tokens_prior_json_content", "FAIL", detail, str(policy_file))
    elif policy_text:
        add_result(results, "policy_tokens_prior_json_content", "WARN", "policy_tokens captured but JSON payload could not be parsed", str(policy_file))
    else:
        add_result(results, "policy_tokens_prior_json_content", "WARN", "policy_tokens not captured; no JSON payload", str(policy_file))

    cmd_vel_file = accept_dir / "cmd_vel_publishers_check.txt"
    if cmd_vel_file.exists():
        text = cmd_vel_file.read_text(encoding="utf-8", errors="replace").lower()
        if "lab_fsd" in text:
            add_result(results, "cmd_vel_no_lab_fsd_content", "FAIL", "lab_fsd appears in /cmd_vel publisher output", str(cmd_vel_file))
        else:
            add_result(results, "cmd_vel_no_lab_fsd_content", "PASS", "lab_fsd absent from /cmd_vel publisher output", str(cmd_vel_file))
        if "mppi" in text:
            add_result(results, "cmd_vel_no_mppi_content", "FAIL", "mppi appears in /cmd_vel publisher output", str(cmd_vel_file))
        else:
            add_result(results, "cmd_vel_no_mppi_content", "PASS", "mppi absent from /cmd_vel publisher output", str(cmd_vel_file))
    else:
        add_result(results, "cmd_vel_no_lab_fsd_content", "WARN", "cmd_vel publisher check file missing", str(cmd_vel_file))
        add_result(results, "cmd_vel_no_mppi_content", "WARN", "cmd_vel publisher check file missing", str(cmd_vel_file))

    cockpit_bb_file = accept_dir / "cockpit_blackbox_recent.txt"
    if cockpit_bb_file.exists():
        bb_events = load_jsonl_objects_from_text(cockpit_bb_file)
        pickup_events = [event for event in bb_events if event.get("k") == "pickup_flow"]
        stage_events = [event for event in bb_events if event.get("k") == "pickup_flow_stage"]
        terminal_states = {"completed", "reported_completed", "simulated", "failed", "timeout", "rejected"}
        terminal = sorted(
            [
                event for event in pickup_events
                if str(event.get("state")) in terminal_states
                and event.get("flow_id")
                and event.get("task_id")
            ],
            key=lambda event: float(event.get("t") or 0.0),
        )
        bound_terminal = terminal[-1] if terminal else None
        bound_flow_id = str(bound_terminal.get("flow_id")) if bound_terminal else ""
        bound_task_id = str(bound_terminal.get("task_id")) if bound_terminal else ""
        bound_flow_events = [
            event for event in pickup_events
            if str(event.get("flow_id") or "") == bound_flow_id
            and str(event.get("task_id") or "") == bound_task_id
        ]
        flow_start_t = min(
            (float(event.get("t") or 0.0) for event in bound_flow_events),
            default=0.0,
        )
        terminal_t = float(bound_terminal.get("t") or 0.0) if bound_terminal else 0.0
        bound_stage_events = sorted(
            [
                event for event in stage_events
                if str(event.get("flow_id") or "") == bound_flow_id
                and str(event.get("task_id") or "") == bound_task_id
                and flow_start_t <= float(event.get("t") or 0.0) <= terminal_t + 1.0
            ],
            key=lambda event: float(event.get("t") or 0.0),
        )
        if pickup_events and bound_terminal and bound_flow_events:
            detail = (
                f"pickup_flow={bound_flow_id}/{bound_task_id} events={len(bound_flow_events)} "
                f"terminal={bound_terminal.get('state')} bound_stages={len(bound_stage_events)}"
            )
            add_result(results, "cockpit_pickup_blackbox_content", "PASS", detail, str(cockpit_bb_file))
        elif pickup_events:
            add_result(results, "cockpit_pickup_blackbox_content", "WARN", "pickup_flow events captured but no terminal state or ids found", str(cockpit_bb_file))
        else:
            add_result(results, "cockpit_pickup_blackbox_content", "WARN", "blackbox captured but no pickup_flow event found", str(cockpit_bb_file))
        if bound_stage_events:
            stages = [int(event.get("stage") or 0) for event in bound_stage_events]
            progress = [float(event.get("progress_pct") or 0.0) for event in bound_stage_events]
            monotonic = all(a <= b for a, b in zip(progress, progress[1:]))
            if monotonic:
                add_result(results, "cockpit_pickup_stage_blackbox_content", "PASS", f"bound stages={stages}; progress monotonic", str(cockpit_bb_file))
            else:
                add_result(results, "cockpit_pickup_stage_blackbox_content", "FAIL", f"bound progress not monotonic: {progress}", str(cockpit_bb_file))
        elif pickup_events:
            add_result(results, "cockpit_pickup_stage_blackbox_content", "WARN", "latest terminal flow has no ID/time-bound DispatchTask stage evidence", str(cockpit_bb_file))
        else:
            add_result(results, "cockpit_pickup_stage_blackbox_content", "WARN", "no pickup_flow stage evidence", str(cockpit_bb_file))
        completion_class = str((bound_terminal or {}).get("completion_class") or "")
        terminal_state = str((bound_terminal or {}).get("state") or "")
        terminal_message = str((bound_terminal or {}).get("message") or "")
        actuator_completed = (bound_terminal or {}).get("actuator_sequence_completed") is True
        physical_completed = (bound_terminal or {}).get("physical_completed") is True
        physical_confirmation_raw = (bound_terminal or {}).get("physical_confirmation")
        physical_confirmation_ok, physical_confirmation_detail = validate_physical_confirmation(
            physical_confirmation_raw,
            str((bound_terminal or {}).get("task_id") or ""),
        )
        terminal_truth = False
        if terminal_state == "simulated":
            terminal_truth = (
                completion_class == "simulated"
                and not actuator_completed
                and not physical_completed
                and not physical_confirmation_raw
                and terminal_message.startswith("SIMULATED_ONLY:")
            )
        elif terminal_state == "reported_completed":
            terminal_truth = (
                completion_class == "f407_reported"
                and actuator_completed
                and not physical_completed
                and not physical_confirmation_raw
                and terminal_message.startswith("F407_REPORTED_COMPLETED:")
            )
        elif terminal_state == "completed":
            terminal_truth = (
                completion_class == "physical"
                and actuator_completed
                and physical_completed
                and physical_confirmation_ok
                and terminal_message.startswith("PHYSICAL_COMPLETED:")
            )
        elif terminal_state in {"failed", "timeout", "rejected"}:
            terminal_truth = (
                not actuator_completed
                and not physical_completed
                and not physical_confirmation_raw
            )
        if bound_terminal and terminal_truth:
            add_result(results, "cockpit_pickup_terminal_truth", "PASS", f"state={terminal_state} completion_class={completion_class}", str(cockpit_bb_file))
        elif bound_terminal:
            add_result(results, "cockpit_pickup_terminal_truth", "FAIL", f"inconsistent terminal state={terminal_state} completion_class={completion_class} actuator={actuator_completed} physical={physical_completed}", str(cockpit_bb_file))
        else:
            add_result(results, "cockpit_pickup_terminal_truth", "WARN", "no bound terminal flow", str(cockpit_bb_file))

        physical_terminal = [bound_terminal] if terminal_state == "completed" and terminal_truth else []
        reported_terminal = [bound_terminal] if terminal_state == "reported_completed" and terminal_truth else []
        stage_messages = [str(event.get("stage_message") or "") for event in bound_stage_events]
        lift_confirmed = any(
            "F407_SERVICE_OK SET_LIFT_HEIGHT" in message
            and ("f407_report_confirmed" in message or "telemetry_confirmed" in message)
            for message in stage_messages
        )
        magnet_confirmed = any(
            "F407_SERVICE_OK SET_ELECTROMAGNET" in message
            and ("f407_report_confirmed" in message or "telemetry_confirmed" in message)
            for message in stage_messages
        )
        if (reported_terminal or physical_terminal) and lift_confirmed and magnet_confirmed:
            completion_kind = "physical" if physical_terminal else "f407_reported_open_loop"
            add_result(results, "cockpit_pickup_actuator_report_evidence", "PASS", f"{completion_kind} terminal plus lift/magnet F407 reports captured", str(cockpit_bb_file))
        elif pickup_events:
            add_result(results, "cockpit_pickup_actuator_report_evidence", "WARN", "pickup flow lacks complete F407 actuator report evidence", str(cockpit_bb_file))
        else:
            add_result(results, "cockpit_pickup_actuator_report_evidence", "WARN", "no pickup flow evidence", str(cockpit_bb_file))
        if physical_terminal and physical_confirmation_ok and lift_confirmed and magnet_confirmed:
            add_result(results, "cockpit_pickup_physical_evidence", "PASS", f"physical terminal includes strict independent evidence plus F407 reports; {physical_confirmation_detail}", str(cockpit_bb_file))
        elif pickup_events:
            add_result(results, "cockpit_pickup_physical_evidence", "WARN", "no independent encoder/limit/object-presence confirmation; F407 open-loop report is not physical completion", str(cockpit_bb_file))
        else:
            add_result(results, "cockpit_pickup_physical_evidence", "WARN", "no pickup flow evidence", str(cockpit_bb_file))
    else:
        add_result(results, "cockpit_pickup_blackbox_content", "WARN", "cockpit blackbox evidence file missing", str(cockpit_bb_file))
        add_result(results, "cockpit_pickup_stage_blackbox_content", "WARN", "cockpit blackbox evidence file missing", str(cockpit_bb_file))
        add_result(results, "cockpit_pickup_actuator_report_evidence", "WARN", "cockpit blackbox evidence file missing", str(cockpit_bb_file))
        add_result(results, "cockpit_pickup_physical_evidence", "WARN", "cockpit blackbox evidence file missing", str(cockpit_bb_file))

    dispatch_stub_file = accept_dir / "dispatch_stub_integration.json"
    if dispatch_stub_file.exists():
        try:
            dispatch_stub = load_json(dispatch_stub_file)
        except Exception as exc:
            add_result(results, "dispatch_stub_integration_content", "FAIL", f"invalid JSON: {exc}", str(dispatch_stub_file))
        else:
            check_items = dispatch_stub.get("checks") if isinstance(dispatch_stub.get("checks"), list) else []
            checks_ok = bool(check_items) and all(
                isinstance(item, dict) and item.get("status") == "PASS"
                for item in check_items
            )
            safety = dispatch_stub.get("safety") if isinstance(dispatch_stub.get("safety"), dict) else {}
            service_calls = safety.get("f407_service_calls") if isinstance(safety.get("f407_service_calls"), dict) else {}
            goals = dispatch_stub.get("goals") if isinstance(dispatch_stub.get("goals"), list) else []
            fetch_goals = [
                goal for goal in goals
                if isinstance(goal, dict) and goal.get("task_id") == "stub-fetch-integration"
            ]
            fetch_ok = bool(fetch_goals) and fetch_goals[-1].get("success") is True and str(
                fetch_goals[-1].get("message") or ""
            ).startswith("SIMULATED_ONLY:")
            truth_ok = (
                dispatch_stub.get("schema_version") == "xrd-dispatch-stub-integration-v1"
                and dispatch_stub.get("overall") == "PASS"
                and dispatch_stub.get("simulation_only") is True
                and dispatch_stub.get("real_hardware_touched") is False
                and dispatch_stub.get("physical_runtime_audit_still_required") is True
                and safety.get("dev_f407_opened") is False
                and safety.get("nav2_enabled") is False
                and safety.get("pickup_actuators_enabled") is False
                and safety.get("cmd_vel_messages") == []
                and safety.get("literal_cmd_vel_messages") == []
                and service_calls
                and all(int(value) == 0 for value in service_calls.values())
                and checks_ok
                and fetch_ok
            )
            if truth_ok:
                add_result(results, "dispatch_stub_integration_content", "PASS", f"checks={len(check_items)}; simulation-only and zero physical output", str(dispatch_stub_file))
            else:
                add_result(
                    results,
                    "dispatch_stub_integration_content",
                    "FAIL",
                    f"schema={dispatch_stub.get('schema_version')} overall={dispatch_stub.get('overall')} checks_ok={checks_ok} fetch_ok={fetch_ok} safety={safety}",
                    str(dispatch_stub_file),
                )
    else:
        add_result(results, "dispatch_stub_integration_content", "FAIL", "isolated DispatchTask evidence missing", str(dispatch_stub_file))

    dispatch_fixture_file = accept_dir / "dispatch_fixture_integration.json"
    if dispatch_fixture_file.exists():
        try:
            dispatch_fixture = load_json(dispatch_fixture_file)
        except Exception as exc:
            add_result(results, "dispatch_fixture_integration_content", "FAIL", f"invalid JSON: {exc}", str(dispatch_fixture_file))
        else:
            fixture_evaluations = audit_dispatch_fixture_report(
                dispatch_fixture,
                dispatch_fixture_file,
                accept_dir,
                acceptance_start_unix,
            )
            for name, (ok, detail) in fixture_evaluations.items():
                add_result(
                    results,
                    name,
                    "PASS" if ok else "FAIL",
                    detail,
                    str(dispatch_fixture_file),
                )
            fixture_content_ok = all(ok for ok, _ in fixture_evaluations.values())
            add_result(
                results,
                "dispatch_fixture_integration_content",
                "PASS" if fixture_content_ok else "FAIL",
                (
                    "fixture policy v2: exact four-goal one-shot transcript, structured result, "
                    "private-domain launch, process-tree monitor, artifact hashes, and clean log"
                    if fixture_content_ok
                    else "one or more strict fixture policy v2 checks failed"
                ),
                str(dispatch_fixture_file),
            )
    else:
        add_result(results, "dispatch_fixture_integration_content", "FAIL", "dispatch fixture integration report missing", str(dispatch_fixture_file))

    status_counts: dict[str, int] = {}
    for item in checks.values():
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1

    return {
        "path": accept_dir.as_posix(),
        "checks_tsv": checks_path.as_posix(),
        "raw_status_counts": status_counts,
        "results": results,
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def verify_sha_file(root: Path, sha_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"available": sha_path.exists(), "checked": 0, "ok": 0, "missing": [], "mismatch": []}
    if not sha_path.exists():
        return out
    for raw in sha_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        expected, name = parts
        name = name.strip().lstrip("*")
        path = root / name
        out["checked"] += 1
        if not path.exists():
            out["missing"].append(name)
            continue
        actual = sha256_file(path)
        if actual == expected:
            out["ok"] += 1
        else:
            out["mismatch"].append({"path": name, "expected": expected, "actual": actual})
    return out


def audit_data_run(run_dir: Path) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        add_result(results, "data_manifest", "FAIL", "manifest.json missing", str(manifest_path))
        return {"path": run_dir.as_posix(), "results": results}

    try:
        manifest = load_json(manifest_path)
        add_result(results, "data_manifest_json", "PASS", "manifest.json parses", str(manifest_path))
    except Exception as exc:
        add_result(results, "data_manifest_json", "FAIL", f"manifest.json invalid: {exc}", str(manifest_path))
        return {"path": run_dir.as_posix(), "results": results}

    hashes = verify_sha_file(run_dir, run_dir / "hashes.sha256")
    if hashes["available"] and hashes["checked"] == hashes["ok"] and not hashes["missing"] and not hashes["mismatch"]:
        add_result(results, "data_hashes", "PASS", f"hashes OK checked={hashes['checked']}", "hashes.sha256")
    else:
        add_result(results, "data_hashes", "FAIL", f"hash verification issue: {hashes}", "hashes.sha256")

    manifest_sha = verify_sha_file(run_dir, run_dir / "manifest.sha256")
    if manifest_sha["available"] and manifest_sha["checked"] == manifest_sha["ok"] and not manifest_sha["missing"] and not manifest_sha["mismatch"]:
        add_result(results, "data_manifest_hash", "PASS", "manifest.sha256 OK", "manifest.sha256")
    else:
        add_result(results, "data_manifest_hash", "FAIL", f"manifest hash issue: {manifest_sha}", "manifest.sha256")
    current_manifest_sha = sha256_file(manifest_path)

    if manifest.get("status") == "stopped":
        add_result(results, "data_manifest_terminal_status", "PASS", "manifest status is immutable terminal state stopped", "manifest.json")
    else:
        add_result(results, "data_manifest_terminal_status", "FAIL", f"manifest status is not stopped: {manifest.get('status')}", "manifest.json")

    ledger_meta = manifest.get("integrity", {}).get("ledger", {}) if isinstance(manifest.get("integrity"), dict) else {}
    ledger_text = ledger_meta.get("path") if isinstance(ledger_meta, dict) else ""
    ledger_path = Path(str(ledger_text)).expanduser() if ledger_text else run_dir.parent / "ledger.jsonl"
    if not ledger_path.is_absolute():
        ledger_path = (run_dir / ledger_path).resolve()
    ledger = verify_ledger(ledger_path)
    ledger_chain_ok = bool(
        ledger["available"]
        and ledger["checked"] > 0
        and ledger["checked"] == ledger["ok"]
        and not ledger["errors"]
    )
    if ledger_chain_ok:
        add_result(results, "data_ledger_chain", "PASS", f"ledger chain OK entries={ledger['checked']}", str(ledger_path))
    else:
        add_result(results, "data_ledger_chain", "FAIL", f"ledger chain issue: {ledger}", str(ledger_path))
    matching_ledger_entries = [
        entry
        for entry in ledger.get("entries", [])
        if entry.get("run_id") == str(manifest.get("run_id") or run_dir.name)
        and entry.get("manifest_sha256") == current_manifest_sha
        and entry.get("status") == "stopped"
    ]
    sequence_ok = bool(
        matching_ledger_entries
        and (
            not isinstance(ledger_meta, dict)
            or ledger_meta.get("sequence") is None
            or matching_ledger_entries[-1].get("sequence") == ledger_meta.get("sequence")
        )
    )
    if sequence_ok:
        entry = matching_ledger_entries[-1]
        add_result(results, "data_ledger_manifest_binding", "PASS", f"ledger sequence={entry.get('sequence')} binds current manifest", str(ledger_path))
    else:
        add_result(results, "data_ledger_manifest_binding", "FAIL", "no valid ledger entry binds current run_id and manifest hash", str(ledger_path))

    model_artifacts = manifest.get("model_artifacts", {})
    artifact_items = model_artifacts.get("artifacts") if isinstance(model_artifacts, dict) else []
    artifact_by_name = {
        str(item.get("name")): item
        for item in artifact_items
        if isinstance(item, dict) and item.get("name")
    }
    for name, detail in sorted(REQUIRED_MODEL_ARTIFACTS.items()):
        item = artifact_by_name.get(name)
        if item and item.get("exists") and item.get("sha256") and item.get("size_bytes", 0) and item.get("sha256_match") is True:
            add_result(results, f"data_model_artifact:{name}", "PASS", detail, "manifest.json")
        elif item:
            add_result(results, f"data_model_artifact:{name}", "FAIL", f"{detail}; artifact missing or expected hash mismatch: {item}", "manifest.json")
        else:
            add_result(results, f"data_model_artifact:{name}", "FAIL", f"missing model artifact: {detail}", "manifest.json")
    for name, detail in sorted(WARN_MODEL_ARTIFACTS.items()):
        item = artifact_by_name.get(name)
        if item and item.get("exists") and item.get("sha256") and item.get("size_bytes", 0) and item.get("sha256_match") is True:
            add_result(results, f"data_model_artifact:{name}", "PASS", detail, "manifest.json")
        elif item:
            add_result(results, f"data_model_artifact:{name}", "WARN", f"{detail}; artifact exists flag/hash incomplete: {item}", "manifest.json")
        else:
            add_result(results, f"data_model_artifact:{name}", "WARN", f"missing optional model artifact: {detail}", "manifest.json")

    ros_info = manifest.get("ros", {})
    bag_metadata = ros_info.get("bag_metadata") or {}
    topics = set(ros_info.get("topics_from_bag") or ros_info.get("topics_recorded") or [])
    topic_counts = {}
    for item in bag_metadata.get("topics") or []:
        name = item.get("name")
        try:
            count = int(item.get("message_count") or 0)
        except (TypeError, ValueError):
            count = 0
        if name:
            topic_counts[name] = count

    try:
        total_messages = int(bag_metadata.get("message_count") or 0)
    except (TypeError, ValueError):
        total_messages = 0
    if total_messages > 0:
        add_result(results, "data_bag_message_count", "PASS", f"rosbag has {total_messages} messages", "manifest.json")
    else:
        add_result(results, "data_bag_message_count", "FAIL", "rosbag message_count is zero or missing", "manifest.json")

    bag_files = ros_info.get("bag_files") or []
    nonempty_bag_files = [
        item
        for item in bag_files
        if isinstance(item, dict)
        and int(item.get("size_bytes") or 0) > 0
        and Path(str(item.get("path") or "")).suffix.lower() in {".mcap", ".db3", ".sqlite3"}
    ]
    if nonempty_bag_files:
        add_result(results, "data_bag_files_nonempty", "PASS", f"bag payload files={len(nonempty_bag_files)}", "manifest.json")
    else:
        add_result(results, "data_bag_files_nonempty", "FAIL", "no non-empty bag payload files were hashed", "manifest.json")

    bag_dir_text = ros_info.get("bag_dir")
    bag_dir = Path(str(bag_dir_text)).expanduser() if bag_dir_text else None
    if bag_dir is not None and not bag_dir.is_absolute():
        bag_dir = (run_dir / bag_dir).resolve()

    cmd_summary = ros_info.get("cmd_vel_evidence") if isinstance(ros_info.get("cmd_vel_evidence"), dict) else {}
    cmd_report_text = cmd_summary.get("path") or "logs/cmd_vel_evidence.json"
    cmd_report_path = Path(str(cmd_report_text)).expanduser()
    if not cmd_report_path.is_absolute():
        cmd_report_path = (run_dir / cmd_report_path).resolve()
    cmd_report: dict[str, Any] = {}
    try:
        cmd_report = load_json(cmd_report_path)
    except Exception as exc:
        add_result(results, "data_cmd_vel_semantic_evidence", "FAIL", f"report missing/invalid: {exc}", str(cmd_report_path))
        add_result(results, "data_cmd_vel_source_binding", "FAIL", "cannot validate source without report", str(cmd_report_path))
    else:
        cmd_counts = cmd_report.get("counts") if isinstance(cmd_report.get("counts"), dict) else {}
        expected_mode = str((manifest.get("safety") or {}).get("cmd_vel_expectation") or "any")
        semantic_ok = bool(
            cmd_report.get("schema_version") == "xrd-cmd-vel-bag-evidence-v1"
            and cmd_report.get("status") == "PASS"
            and cmd_report.get("topic") == "/cmd_vel"
            and cmd_report.get("topic_type") == "geometry_msgs/msg/Twist"
            and cmd_report.get("expectation") == expected_mode
            and integer(cmd_counts.get("message_count")) == topic_counts.get("/cmd_vel", 0)
            and integer(cmd_counts.get("decoded_count")) == integer(cmd_counts.get("message_count"))
            and integer(cmd_counts.get("decode_error_count")) == 0
            and integer(cmd_counts.get("nonfinite_count")) == 0
            and (expected_mode != "zero" or integer(cmd_counts.get("nonzero_count")) == 0)
            and (expected_mode != "nonzero" or integer(cmd_counts.get("nonzero_count")) > 0)
            and cmd_summary.get("status") == cmd_report.get("status")
            and cmd_summary.get("counts") == cmd_report.get("counts")
        )
        if semantic_ok:
            add_result(
                results,
                "data_cmd_vel_semantic_evidence",
                "PASS",
                f"offline decoded Twist messages={cmd_counts.get('message_count')} expectation={expected_mode} nonzero={cmd_counts.get('nonzero_count')}",
                str(cmd_report_path),
            )
        else:
            add_result(
                results,
                "data_cmd_vel_semantic_evidence",
                "FAIL",
                f"semantic report mismatch: status={cmd_report.get('status')} expectation={cmd_report.get('expectation')} counts={cmd_counts}",
                str(cmd_report_path),
            )

        source = cmd_report.get("source") if isinstance(cmd_report.get("source"), dict) else {}
        source_payloads = source.get("payloads") if isinstance(source.get("payloads"), list) else []
        source_errors: list[str] = []
        if bag_dir is None or not bag_dir.is_dir():
            source_errors.append("bag directory missing")
        if not source_payloads:
            source_errors.append("report payload inventory empty")
        if bag_dir is not None:
            for item in source_payloads:
                if not isinstance(item, dict) or not item.get("path"):
                    source_errors.append("malformed payload item")
                    continue
                payload_path = (bag_dir / str(item["path"])).resolve()
                try:
                    payload_path.relative_to(bag_dir.resolve())
                except ValueError:
                    source_errors.append(f"payload path escapes bag dir: {item.get('path')}")
                    continue
                if not payload_path.is_file():
                    source_errors.append(f"payload missing: {item.get('path')}")
                    continue
                if payload_path.stat().st_size != integer(item.get("size_bytes"), -1):
                    source_errors.append(f"payload size mismatch: {item.get('path')}")
                if sha256_file(payload_path) != item.get("sha256"):
                    source_errors.append(f"payload hash mismatch: {item.get('path')}")
            metadata_path = bag_dir / "metadata.yaml"
            if not metadata_path.is_file() or sha256_file(metadata_path) != source.get("metadata_sha256"):
                source_errors.append("metadata.yaml hash mismatch")
        if source_errors:
            add_result(results, "data_cmd_vel_source_binding", "FAIL", "; ".join(source_errors), str(cmd_report_path))
        else:
            add_result(results, "data_cmd_vel_source_binding", "PASS", f"payload hashes bound={len(source_payloads)}", str(cmd_report_path))

    rosbag_info_file = run_dir / "logs" / "rosbag_info.txt"
    if rosbag_info_file.exists() and rosbag_info_file.stat().st_size > 0:
        rosbag_info_text = rosbag_info_file.read_text(encoding="utf-8", errors="replace").lower()
        if any(token in rosbag_info_text for token in ["error", "failed", "traceback", "exception"]):
            add_result(results, "data_rosbag_info_readable", "FAIL", "ros2 bag info output contains an error marker", str(rosbag_info_file))
        else:
            add_result(results, "data_rosbag_info_readable", "PASS", "ros2 bag info output captured without error markers", str(rosbag_info_file))
    else:
        add_result(results, "data_rosbag_info_readable", "FAIL", "logs/rosbag_info.txt missing; data_loop_stop should run ros2 bag info", str(rosbag_info_file))

    ros2_bin = shutil.which("ros2")
    if ros2_bin and bag_dir is not None:
        try:
            live_info = subprocess.run(
                [ros2_bin, "bag", "info", str(bag_dir)],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            add_result(results, "data_rosbag_live_validation", "FAIL", f"ros2 bag info execution failed: {exc}", str(bag_dir))
        else:
            output = (live_info.stdout or "") + "\n" + (live_info.stderr or "")
            match = re.search(r"(?im)^\s*Messages:\s*(\d+)\s*$", output)
            live_messages = int(match.group(1)) if match else 0
            if live_info.returncode == 0 and live_messages == total_messages and live_messages > 0:
                add_result(results, "data_rosbag_live_validation", "PASS", f"ros2 bag info reopened payload; messages={live_messages}", str(bag_dir))
            else:
                detail = f"rc={live_info.returncode} live_messages={live_messages} manifest_messages={total_messages}"
                add_result(results, "data_rosbag_live_validation", "FAIL", detail, str(bag_dir))

    for topic in sorted(REQUIRED_DATA_TOPICS):
        count = topic_counts.get(topic, 0)
        if topic in topics and count > 0:
            add_result(results, f"data_topic:{topic}", "PASS", f"required topic present with {count} messages", "manifest.json")
        elif topic in topics:
            add_result(results, f"data_topic:{topic}", "FAIL", "required topic listed but has no message_count evidence", "manifest.json")
        else:
            add_result(results, f"data_topic:{topic}", "FAIL", "required topic missing", "manifest.json")
    for topic in sorted(WARN_DATA_TOPICS):
        count = topic_counts.get(topic, 0)
        if topic in topics and count > 0:
            add_result(results, f"data_topic:{topic}", "PASS", f"optional topic present with {count} messages", "manifest.json")
        elif topic in topics:
            add_result(results, f"data_topic:{topic}", "WARN", "optional topic listed but has no message_count evidence", "manifest.json")
        else:
            add_result(results, f"data_topic:{topic}", "WARN", "optional topic missing", "manifest.json")

    skeleton_dir = run_dir / "exports" / "training_skeleton"
    skeleton_manifest = skeleton_dir / "dataset_skeleton_manifest.json"
    skeleton_data: dict[str, Any] = {}
    if skeleton_manifest.exists():
        try:
            skeleton_data = load_json(skeleton_manifest)
            expected_status = "skeleton_only_pending_array_extraction"
            if skeleton_data.get("status") == expected_status:
                add_result(results, "training_skeleton_manifest", "PASS", expected_status, str(skeleton_manifest))
            else:
                add_result(results, "training_skeleton_manifest", "FAIL", f"unexpected status={skeleton_data.get('status')}", str(skeleton_manifest))
        except Exception as exc:
            add_result(results, "training_skeleton_manifest", "FAIL", f"invalid JSON: {exc}", str(skeleton_manifest))
    else:
        add_result(results, "training_skeleton_manifest", "FAIL", "training skeleton manifest missing", str(skeleton_manifest))

    skeleton_hashes = verify_sha_file(skeleton_dir, skeleton_dir / "skeleton_hashes.sha256")
    if skeleton_hashes["available"] and skeleton_hashes["checked"] == skeleton_hashes["ok"] and not skeleton_hashes["missing"] and not skeleton_hashes["mismatch"]:
        add_result(results, "training_skeleton_hashes", "PASS", f"skeleton hashes OK checked={skeleton_hashes['checked']}", str(skeleton_dir / "skeleton_hashes.sha256"))
    else:
        add_result(results, "training_skeleton_hashes", "FAIL", f"skeleton hash issue: {skeleton_hashes}", str(skeleton_dir / "skeleton_hashes.sha256"))

    skeleton_manifest_sha = verify_sha_file(skeleton_dir, skeleton_dir / "skeleton_manifest.sha256")
    if skeleton_manifest_sha["available"] and skeleton_manifest_sha["checked"] == skeleton_manifest_sha["ok"] and not skeleton_manifest_sha["missing"] and not skeleton_manifest_sha["mismatch"]:
        add_result(results, "training_skeleton_manifest_hash", "PASS", "skeleton manifest hash OK", str(skeleton_dir / "skeleton_manifest.sha256"))
    else:
        add_result(results, "training_skeleton_manifest_hash", "FAIL", f"skeleton manifest hash issue: {skeleton_manifest_sha}", str(skeleton_dir / "skeleton_manifest.sha256"))

    conversion_status_path = skeleton_dir / "conversion_status.json"
    try:
        conversion_status = load_json(conversion_status_path)
    except Exception as exc:
        conversion_status = {}
        add_result(results, "training_conversion_status", "FAIL", f"conversion status missing/invalid: {exc}", str(conversion_status_path))
    else:
        conversion_ok = (
            conversion_status.get("status") == "complete"
            and conversion_status.get("source_manifest_sha256") == current_manifest_sha
            and conversion_status.get("quality_gate") == "PASS"
        )
        if conversion_ok:
            add_result(results, "training_conversion_status", "PASS", "conversion complete and bound to current manifest", str(conversion_status_path))
        else:
            add_result(results, "training_conversion_status", "FAIL", f"conversion status mismatch: {conversion_status}", str(conversion_status_path))

    quality_path = skeleton_dir / "quality_report.json"
    try:
        quality = load_json(quality_path)
    except Exception as exc:
        quality = {}
        add_result(results, "training_quality_gate", "FAIL", f"quality report missing/invalid: {exc}", str(quality_path))
    else:
        if quality.get("overall") == "PASS" and not quality.get("failed_checks"):
            add_result(results, "training_quality_gate", "PASS", f"checks={len(quality.get('checks') or [])}", str(quality_path))
        else:
            add_result(results, "training_quality_gate", "FAIL", f"failed_checks={quality.get('failed_checks')}", str(quality_path))

    if skeleton_data:
        source_manifest_integrity = skeleton_data.get("source_manifest_integrity") or {}
        source_hashes = skeleton_data.get("source_integrity") or {}
        source_manifest_ok = bool(
            source_manifest_integrity.get("available")
            and source_manifest_integrity.get("ok")
            and source_manifest_integrity.get("actual") == current_manifest_sha
        )
        if source_manifest_ok:
            add_result(results, "training_source_manifest_binding", "PASS", "skeleton references current immutable manifest hash", str(skeleton_manifest))
        else:
            add_result(results, "training_source_manifest_binding", "FAIL", f"source manifest binding mismatch: {source_manifest_integrity}", str(skeleton_manifest))
        source_hashes_ok = bool(
            source_hashes.get("available")
            and int(source_hashes.get("checked") or 0) > 0
            and source_hashes.get("checked") == source_hashes.get("ok")
            and not source_hashes.get("missing")
            and not source_hashes.get("mismatch")
        )
        if source_hashes_ok:
            add_result(results, "training_source_payload_binding", "PASS", f"source hashes checked={source_hashes.get('checked')}", str(skeleton_manifest))
        else:
            add_result(results, "training_source_payload_binding", "FAIL", f"source hash binding issue: {source_hashes}", str(skeleton_manifest))
        if skeleton_data.get("model_artifacts") == manifest.get("model_artifacts"):
            add_result(results, "training_model_provenance_binding", "PASS", "training skeleton inherits exact model artifact provenance", str(skeleton_manifest))
        else:
            add_result(results, "training_model_provenance_binding", "FAIL", "training skeleton model provenance differs from run manifest", str(skeleton_manifest))

    for rel_path in [
        "lerobot_v3_skeleton/meta/info.json",
        "lerobot_v3_skeleton/meta/episodes.jsonl",
        "robomimic_skeleton/dataset_spec.json",
    ]:
        path = skeleton_dir / rel_path
        if path.exists() and path.stat().st_size > 0:
            add_result(results, f"training_skeleton_file:{rel_path}", "PASS", "file present", str(path))
        else:
            add_result(results, f"training_skeleton_file:{rel_path}", "FAIL", "file missing/empty", str(path))

    return {"path": run_dir.as_posix(), "run_id": manifest.get("run_id"), "results": results}


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for item in results:
        status = item.get("status", "WARN")
        counts[status] = counts.get(status, 0) + 1
    if counts.get("FAIL", 0) > 0:
        overall = "FAIL"
    elif counts.get("WARN", 0) > 0:
        overall = "WARN"
    else:
        overall = "PASS"
    return {"overall": overall, "counts": counts}


def write_text_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"EMBODIED_V3_AUDIT {report['summary']['overall']}",
        f"generated_at: {report['generated_at']}",
        f"counts: {report['summary']['counts']}",
        "",
    ]
    for section_name in ["acceptance", "data_run"]:
        section = report.get(section_name)
        if not section:
            continue
        lines.append(f"== {section_name} ==")
        lines.append(f"path: {section.get('path')}")
        for item in section.get("results", []):
            evidence = item.get("evidence") or ""
            lines.append(f"[{item['status']}] {item['name']}: {item['detail']} {evidence}".rstrip())
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-dir", default="", help="Directory generated by embodied_v3_acceptance_check.sh")
    parser.add_argument("--data-run", default="", help="Directory generated by data_loop_start/stop")
    parser.add_argument("--require-data-run", action="store_true", help="Fail when no finalized data-loop run is supplied")
    parser.add_argument("--out", default="", help="JSON report path. Default: ACCEPT_DIR/audit_report.json or ./audit_report.json")
    parser.add_argument("--text-out", default="", help="Text report path. Default: sibling audit_report.txt")
    args = parser.parse_args()

    if not args.accept_dir and not args.data_run:
        raise SystemExit("provide --accept-dir and/or --data-run")

    report: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "policy_version": AUDIT_POLICY_VERSION,
        "generated_at": utc_now(),
    }
    all_results: list[dict[str, Any]] = []

    resolved_data_run = Path(args.data_run).expanduser().resolve() if args.data_run else None
    if args.accept_dir:
        acceptance = audit_acceptance_dir(Path(args.accept_dir).expanduser().resolve(), resolved_data_run)
        report["acceptance"] = acceptance
        all_results.extend(acceptance.get("results", []))
    if resolved_data_run is not None:
        data_run = audit_data_run(resolved_data_run)
        report["data_run"] = data_run
        all_results.extend(data_run.get("results", []))
    elif args.require_data_run:
        data_run = {
            "path": "",
            "results": [
                {
                    "name": "data_run_required",
                    "status": "FAIL",
                    "detail": "a finalized data-loop run is required for embodied v3 acceptance",
                    "evidence": "",
                }
            ],
        }
        report["data_run"] = data_run
        all_results.extend(data_run["results"])

    report["summary"] = summarize(all_results)

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    elif args.accept_dir:
        out_path = Path(args.accept_dir).expanduser().resolve() / "audit_report.json"
    else:
        out_path = Path("audit_report.json").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    text_path = Path(args.text_out).expanduser().resolve() if args.text_out else out_path.with_suffix(".txt")
    write_text_report(text_path, report)

    print(f"EMBODIED_V3_AUDIT {report['summary']['overall']}")
    print(f"json: {out_path}")
    print(f"text: {text_path}")
    print(f"counts: {report['summary']['counts']}")
    return 2 if report["summary"]["overall"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
