#!/usr/bin/env python3
"""Recover a post-flash run that failed only during ROS read-only sampling.

The tool never opens /dev/F407, never writes to the ROS graph, and never
starts/stops services. It validates the immutable failed physical run, copies
its artifacts and exact tool snapshots into a new evidence directory, captures
three read-only ROS topics, and emits a chained orchestration-v2 manifest.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from f407_postflash_report import build_validation


SOURCE_SCHEMA = "xrd-f407-postflash-interlock-orchestration-v1"
RECOVERED_SCHEMA = "xrd-f407-postflash-interlock-orchestration-v2"
RECOVERY_SCHEMA = "xrd-f407-postflash-readonly-recovery-v1"
EXPECTED_FAILURE = "phase=post_restore_readonly_topics: unexpected exit rc=1"
CONFIRMATION_SHA256 = hashlib.sha256(
    b"NO_LOAD_PATH_CLEAR_BASE_FIXED_HANDS_CLEAR_OPERATOR_PRESENT"
).hexdigest()

PHYSICAL_FILES = {
    "interlock_report": "f407_interlock_report.json",
    "interlock_log": "f407_interlock.log",
    "validation_report": "f407_interlock_validation.json",
}
POST_TOPIC_FILES = {
    "post_firmware_topic": "post_f407_firmware_info.txt",
    "post_identity_topic": "post_f407_firmware_identity_valid.txt",
    "post_estop_topic": "post_f407_estop_latched.txt",
}
PHYSICAL_TOOL_SNAPSHOTS = {
    "orchestrator": ("f407_postflash_interlock_acceptance.sh", "physical_tool_orchestrator.sh"),
    "link_test": ("f407_link_test.py", "physical_tool_link_test.py"),
    "validator": ("f407_postflash_report.py", "physical_tool_validator.py"),
}
SYSTEM_SERVICES = ("embodied_brain.service", "cockpit_bridge.service")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    if not path.is_file():
        raise ValueError(f"{label} missing: {path}")
    return path.resolve()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _safe_command_contract(manifest: dict[str, Any]) -> bool:
    tooling = manifest.get("tooling") if isinstance(manifest.get("tooling"), dict) else {}
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    link_test = tooling.get("link_test") if isinstance(tooling.get("link_test"), dict) else {}
    report = artifacts.get("interlock_report") if isinstance(artifacts.get("interlock_report"), dict) else {}
    command = manifest.get("command_contract") if isinstance(manifest.get("command_contract"), dict) else {}
    argv = command.get("argv") if isinstance(command.get("argv"), list) else []
    expected = [
        "python3",
        str(link_test.get("path") or ""),
        "--port",
        "/dev/F407",
        "--verify-estop-interlock",
        "--require-ack",
        "--report",
        str(report.get("path") or ""),
    ]
    return bool(
        argv == expected
        and "--clear-estop" not in argv
        and "--v" not in argv
        and "--move-sec" not in argv
        and command.get("clear_estop_requested") is False
        and command.get("nonzero_cmd_vel_requested") is False
        and command.get("electromagnet_off_is_sent") is True
        and command.get("physical_completion_claimed") is False
    )


def validate_failed_source(source_dir: Path, home_dir: Path) -> dict[str, Any]:
    home = home_dir.expanduser().resolve()
    root = (home / "f407_postflash_acceptance").resolve()
    source = source_dir.expanduser().resolve()
    if source.parent != root or not source.name.startswith("postflash_"):
        raise ValueError("source directory is outside the canonical post-flash root")
    manifest_path = regular_file(source / "postflash_interlock_manifest.json", "failed manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("failed manifest root must be an object")

    finished = float(manifest.get("finished_at_unix") or 0.0)
    now = time.time()
    if not (
        manifest.get("schema_version") == SOURCE_SCHEMA
        and manifest.get("overall") == "FAIL"
        and manifest.get("failure_reason") == EXPECTED_FAILURE
        and 0.0 < finished <= now + 5.0
        and 0.0 <= now - finished <= 86400.0
    ):
        raise ValueError("source is not the narrowly recoverable read-only sampling failure")

    hostname = manifest.get("hostname") if isinstance(manifest.get("hostname"), dict) else {}
    confirmation = (
        manifest.get("operator_confirmation")
        if isinstance(manifest.get("operator_confirmation"), dict)
        else {}
    )
    if not (
        hostname.get("expected") == "embodied-x5"
        and hostname.get("actual") == "embodied-x5"
        and hostname.get("matched") is True
        and confirmation.get("safe_field_state_confirmed") is True
        and confirmation.get("confirmation_token_sha256") == CONFIRMATION_SHA256
        and confirmation.get("magnet_off_drop_hazard_acknowledged") is True
        and confirmation.get("raw_confirmation_token_stored") is False
        and _safe_command_contract(manifest)
    ):
        raise ValueError("source identity, operator confirmation, or command contract failed")

    serial = manifest.get("serial_exclusivity") if isinstance(manifest.get("serial_exclusivity"), dict) else {}
    restore = manifest.get("service_restore") if isinstance(manifest.get("service_restore"), dict) else {}
    if not (
        serial.get("device") == "/dev/F407"
        and serial.get("owners_after_stop") == []
        and serial.get("unowned_before_test") is True
        and restore.get("services_quiesced") is True
        and restore.get("attempted") is True
        and restore.get("success") is True
        and restore.get("pre") == restore.get("post")
    ):
        raise ValueError("source serial exclusivity or service restoration failed")

    source_artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    for key, filename in PHYSICAL_FILES.items():
        path = regular_file(source / filename, f"source artifact {key}")
        entry = source_artifacts.get(key) if isinstance(source_artifacts.get(key), dict) else {}
        if not (
            Path(str(entry.get("path") or "")).resolve() == path
            and entry.get("exists") is True
            and entry.get("sha256") == sha256_file(path)
            and int(entry.get("size_bytes") or -1) == path.stat().st_size
        ):
            raise ValueError(f"source artifact {key} hash/path contract failed")
    for key in POST_TOPIC_FILES:
        entry = source_artifacts.get(key) if isinstance(source_artifacts.get(key), dict) else {}
        if entry.get("exists") is not False:
            raise ValueError(f"source failure unexpectedly contains {key}")

    raw_validation = build_validation(source / PHYSICAL_FILES["interlock_report"])
    stored_validation = json.loads((source / PHYSICAL_FILES["validation_report"]).read_text(encoding="utf-8"))
    if raw_validation.get("overall") != "PASS" or stored_validation.get("overall") != "PASS":
        raise ValueError("source physical interlock validation is not PASS")
    if stored_validation.get("counts", {}).get("FAIL") != 0:
        raise ValueError("stored source validation contains failures")

    tooling = manifest.get("tooling") if isinstance(manifest.get("tooling"), dict) else {}
    tool_paths: dict[str, Path] = {}
    for key, (source_name, _snapshot_name) in PHYSICAL_TOOL_SNAPSHOTS.items():
        path = regular_file(home / "tools" / source_name, f"physical tool {key}")
        entry = tooling.get(key) if isinstance(tooling.get(key), dict) else {}
        if Path(str(entry.get("path") or "")).resolve() != path or entry.get("sha256") != sha256_file(path):
            raise ValueError(f"physical tool {key} no longer matches the failed-run fingerprint")
        tool_paths[key] = path

    return {
        "source_dir": source,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "tool_paths": tool_paths,
    }


def service_states() -> dict[str, dict[str, str]]:
    states = {"system": {}, "user": {}}
    for scope in ("system", "user"):
        for service in SYSTEM_SERVICES:
            command = ["systemctl"]
            if scope == "user":
                command.append("--user")
            command.extend(["is-active", service])
            proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
            states[scope][service] = proc.stdout.strip() or "unknown"
    return states


def capture_topic(topic: str, attempts: int = 4, timeout_s: int = 10) -> str:
    last = ""
    for _attempt in range(attempts):
        try:
            proc = subprocess.run(
                ["ros2", "topic", "echo", "--once", "--full-length", topic],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=False,
                env={**os.environ, "ROS2CLI_NO_DAEMON": "1"},
            )
        except subprocess.TimeoutExpired as exc:
            last = str(exc.stdout or "") + str(exc.stderr or "")
        else:
            last = proc.stdout
            if proc.returncode == 0 and last.strip():
                return last
        time.sleep(2)
    raise RuntimeError(f"read-only topic capture failed: {topic}: {last[-500:]}")


def parse_ros_json(text: str) -> dict[str, Any]:
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        value = line.split(":", 1)[1].strip()
        candidates = [value, value.strip("'\"")]
        try:
            literal = ast.literal_eval(value)
        except Exception:
            literal = None
        if isinstance(literal, str):
            candidates.append(literal)
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
    return {}


def build_recovered_manifest(
    validated: dict[str, Any],
    output_dir: Path,
    service_pre: dict[str, dict[str, str]],
    service_post: dict[str, dict[str, str]],
    recovery_started: float,
    recovery_finished: float,
    recovery_tool: Path,
) -> dict[str, Any]:
    source_manifest = validated["manifest"]
    source_manifest_path = validated["manifest_path"]
    source_copy = output_dir / "source_failed_manifest.json"
    revalidation = output_dir / "recovery_interlock_revalidation.json"
    tooling = dict(source_manifest["tooling"])
    tooling["recovery"] = {
        "path": str(recovery_tool.resolve()),
        "sha256": sha256_file(recovery_tool),
    }
    artifacts = {
        key: artifact(output_dir / filename)
        for key, filename in {**PHYSICAL_FILES, **POST_TOPIC_FILES}.items()
    }
    artifacts["source_failed_manifest"] = artifact(source_copy)
    artifacts["recovery_revalidation"] = artifact(revalidation)
    snapshots = {
        key: artifact(output_dir / snapshot_name)
        for key, (_source_name, snapshot_name) in PHYSICAL_TOOL_SNAPSHOTS.items()
    }
    return {
        "schema_version": RECOVERED_SCHEMA,
        "overall": "PASS",
        "failure_reason": "",
        "started_at_unix": source_manifest["started_at_unix"],
        "started_at_utc": source_manifest["started_at_utc"],
        "finished_at_unix": recovery_finished,
        "finished_at_utc": utc_now(),
        "hostname": source_manifest["hostname"],
        "operator_confirmation": source_manifest["operator_confirmation"],
        "command_contract": source_manifest["command_contract"],
        "serial_exclusivity": source_manifest["serial_exclusivity"],
        "service_restore": source_manifest["service_restore"],
        "tooling": tooling,
        "artifacts": artifacts,
        "physical_interlock": {
            "source_manifest": str(source_manifest_path),
            "source_manifest_sha256": validated["manifest_sha256"],
            "source_manifest_copy": str(source_copy.resolve()),
            "source_manifest_copy_sha256": sha256_file(source_copy),
            "source_schema_version": SOURCE_SCHEMA,
            "source_overall": "FAIL",
            "source_failure_reason": EXPECTED_FAILURE,
            "raw_interlock_validation": "PASS",
            "tool_snapshots": snapshots,
        },
        "recovery": {
            "schema_version": RECOVERY_SCHEMA,
            "mode": "post_restore_readonly_only",
            "started_at_unix": recovery_started,
            "finished_at_unix": recovery_finished,
            "tool": tooling["recovery"],
            "serial_device_opened": False,
            "ros_graph_writes": False,
            "services_stopped": False,
            "services_started_or_restarted": False,
            "physical_commands_sent": False,
            "service_state_pre": service_pre,
            "service_state_post": service_post,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--home", default=str(Path.home()))
    args = parser.parse_args()

    home = Path(args.home).expanduser().resolve()
    validated = validate_failed_source(Path(args.source_dir), home)
    source_dir = validated["source_dir"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else home / "f407_postflash_acceptance" / f"postflash_recovered_{source_dir.name}_{stamp}"
    )
    root = (home / "f407_postflash_acceptance").resolve()
    if output_dir.parent != root or not output_dir.name.startswith("postflash_recovered_"):
        raise SystemExit("recovery output must be a new canonical directory below f407_postflash_acceptance")
    if output_dir.exists():
        raise SystemExit(f"recovery output already exists: {output_dir}")
    output_dir.mkdir(mode=0o700)

    recovery_started = time.time()
    service_pre = service_states()
    expected_services = validated["manifest"]["service_restore"]["post"]
    if service_pre != expected_services:
        raise SystemExit(f"current service state differs from restored physical run: {service_pre}")

    for key, filename in PHYSICAL_FILES.items():
        shutil.copy2(source_dir / filename, output_dir / filename)
    shutil.copy2(validated["manifest_path"], output_dir / "source_failed_manifest.json")
    for key, (_source_name, snapshot_name) in PHYSICAL_TOOL_SNAPSHOTS.items():
        shutil.copy2(validated["tool_paths"][key], output_dir / snapshot_name)

    revalidation = build_validation(output_dir / PHYSICAL_FILES["interlock_report"])
    if revalidation.get("overall") != "PASS":
        raise SystemExit("copied physical interlock failed recovery revalidation")
    (output_dir / "recovery_interlock_revalidation.json").write_text(
        json.dumps(revalidation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    topic_text = {
        "post_firmware_topic": capture_topic("/f407/firmware_info"),
        "post_identity_topic": capture_topic("/f407/firmware_identity_valid"),
        "post_estop_topic": capture_topic("/f407/estop_latched"),
    }
    for key, text in topic_text.items():
        (output_dir / POST_TOPIC_FILES[key]).write_text(text, encoding="utf-8")

    firmware = parse_ros_json(topic_text["post_firmware_topic"])
    true_pattern = re.compile(r"(?m)^\s*data:\s*true\s*$", re.IGNORECASE)
    if not (
        firmware.get("protocol_version") == 2
        and (int(firmware.get("capabilities") or 0) & 0x003F) == 0x003F
        and firmware.get("build_id") == 2026071907
        and firmware.get("test_mode") == 0
        and firmware.get("hw_variant") == 1
        and firmware.get("identity_valid") is True
        and true_pattern.search(topic_text["post_identity_topic"])
        and true_pattern.search(topic_text["post_estop_topic"])
    ):
        raise SystemExit("recovery read-only firmware identity or estop topic validation failed")

    service_post = service_states()
    if service_post != service_pre:
        raise SystemExit(f"service state changed during read-only recovery: pre={service_pre} post={service_post}")
    recovery_finished = time.time()
    recovery_tool = Path(__file__).resolve()
    manifest = build_recovered_manifest(
        validated,
        output_dir,
        service_pre,
        service_post,
        recovery_started,
        recovery_finished,
        recovery_tool,
    )
    manifest_path = output_dir / "postflash_interlock_recovered_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "overall": "PASS",
                "manifest": str(manifest_path),
                "source_failure_manifest_sha256": validated["manifest_sha256"],
                "serial_opened": False,
                "ros_graph_writes": False,
                "services_changed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
