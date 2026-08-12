#!/usr/bin/env python3
"""Stage a fixed, hash-checked post-flash F407 bundle into acceptance evidence.

The source is read-only. Only known regular files below the operator's home
directory are accepted, so a manifest cannot turn the collector into an
arbitrary file-copy primitive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "xrd-f407-postflash-bundle-index-v1"
MANIFEST_SCHEMA_V1 = "xrd-f407-postflash-interlock-orchestration-v1"
MANIFEST_SCHEMA_V2 = "xrd-f407-postflash-interlock-orchestration-v2"
RECOVERY_SCHEMA_V1 = "xrd-f407-postflash-readonly-recovery-v1"
RECOVERABLE_SOURCE_FAILURE = "phase=post_restore_readonly_topics: unexpected exit rc=1"
CONFIRMATION_SHA256 = hashlib.sha256(
    b"NO_LOAD_PATH_CLEAR_BASE_FIXED_HANDS_CLEAR_OPERATOR_PRESENT"
).hexdigest()

BASE_ARTIFACT_SPECS = {
    "interlock_report": ("f407_interlock_report.json", "f407_interlock_report.json"),
    "interlock_log": ("f407_interlock.log", "f407_postflash_interlock.log"),
    "validation_report": ("f407_interlock_validation.json", "f407_postflash_validation.json"),
    "post_firmware_topic": ("post_f407_firmware_info.txt", "f407_postflash_firmware_info.txt"),
    "post_identity_topic": (
        "post_f407_firmware_identity_valid.txt",
        "f407_postflash_firmware_identity_valid.txt",
    ),
    "post_estop_topic": ("post_f407_estop_latched.txt", "f407_postflash_estop_latched.txt"),
}

RECOVERY_ARTIFACT_SPECS = {
    "source_failed_manifest": (
        "source_failed_manifest.json",
        "f407_postflash_source_failed_manifest.json",
    ),
    "recovery_revalidation": (
        "recovery_interlock_revalidation.json",
        "f407_postflash_recovery_revalidation.json",
    ),
}

V1_TOOL_SPECS = {
    "orchestrator": ("f407_postflash_interlock_acceptance.sh", "f407_postflash_orchestrator.sh"),
    "link_test": ("f407_link_test.py", "f407_postflash_link_test.py"),
    "validator": ("f407_postflash_report.py", "f407_postflash_validator.py"),
}

V2_PHYSICAL_TOOL_SPECS = {
    "orchestrator": ("physical_tool_orchestrator.sh", "f407_postflash_physical_orchestrator.sh"),
    "link_test": ("physical_tool_link_test.py", "f407_postflash_physical_link_test.py"),
    "validator": ("physical_tool_validator.py", "f407_postflash_physical_validator.py"),
}
V2_RECOVERY_TOOL_SPEC = (
    "f407_postflash_recover_readonly.py",
    "f407_postflash_recovery.py",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is missing or not a regular file: {path}")
    return path.resolve()


def _artifact_entry(manifest: dict[str, Any], section: str, key: str) -> dict[str, Any]:
    group = manifest.get(section)
    if not isinstance(group, dict) or not isinstance(group.get(key), dict):
        raise ValueError(f"manifest {section}.{key} missing")
    return group[key]


def _validate_digest(entry: dict[str, Any], source: Path, label: str) -> str:
    expected_sha = str(entry.get("sha256") or "")
    actual_sha = sha256_file(source)
    if len(expected_sha) != 64 or expected_sha != actual_sha:
        raise ValueError(f"{label} SHA-256 mismatch")
    expected_size = entry.get("size_bytes")
    if expected_size is not None and int(expected_size) != source.stat().st_size:
        raise ValueError(f"{label} size mismatch")
    return actual_sha


def _validate_hostname_confirmation(manifest: dict[str, Any]) -> None:
    hostname = manifest.get("hostname") if isinstance(manifest.get("hostname"), dict) else {}
    if not (
        hostname.get("expected") == "embodied-x5"
        and hostname.get("actual") == "embodied-x5"
        and hostname.get("matched") is True
    ):
        raise ValueError("post-flash hostname contract failed")

    confirmation = (
        manifest.get("operator_confirmation")
        if isinstance(manifest.get("operator_confirmation"), dict)
        else {}
    )
    if not (
        confirmation.get("safe_field_state_confirmed") is True
        and confirmation.get("confirmation_token_sha256") == CONFIRMATION_SHA256
        and confirmation.get("magnet_off_drop_hazard_acknowledged") is True
        and confirmation.get("raw_confirmation_token_stored") is False
    ):
        raise ValueError("post-flash operator confirmation contract failed")


def _validate_command_contract(manifest: dict[str, Any], expected_report: Path) -> None:
    tooling = manifest.get("tooling") if isinstance(manifest.get("tooling"), dict) else {}
    link_test = tooling.get("link_test") if isinstance(tooling.get("link_test"), dict) else {}
    expected_argv = [
        "python3",
        str(link_test.get("path") or ""),
        "--port",
        "/dev/F407",
        "--verify-estop-interlock",
        "--require-ack",
        "--report",
        str(expected_report),
    ]
    command = manifest.get("command_contract") if isinstance(manifest.get("command_contract"), dict) else {}
    argv = command.get("argv") if isinstance(command.get("argv"), list) else []
    if not (
        argv == expected_argv
        and "--clear-estop" not in argv
        and "--v" not in argv
        and "--move-sec" not in argv
        and command.get("clear_estop_requested") is False
        and command.get("nonzero_cmd_vel_requested") is False
        and command.get("electromagnet_off_is_sent") is True
        and command.get("physical_completion_claimed") is False
    ):
        raise ValueError("post-flash command contract failed")


def _validate_serial_restore(manifest: dict[str, Any]) -> dict[str, Any]:
    serial = manifest.get("serial_exclusivity") if isinstance(manifest.get("serial_exclusivity"), dict) else {}
    if not (
        serial.get("device") == "/dev/F407"
        and isinstance(serial.get("owners_before"), list)
        and serial.get("owners_after_stop") == []
        and serial.get("unowned_before_test") is True
    ):
        raise ValueError("post-flash serial exclusivity contract failed")

    restore = manifest.get("service_restore") if isinstance(manifest.get("service_restore"), dict) else {}
    pre = restore.get("pre") if isinstance(restore.get("pre"), dict) else {}
    post = restore.get("post") if isinstance(restore.get("post"), dict) else {}
    system_pre = pre.get("system") if isinstance(pre.get("system"), dict) else {}
    user_pre = pre.get("user") if isinstance(pre.get("user"), dict) else {}
    embodied_active_count = sum(
        state.get("embodied_brain.service") == "active" for state in (system_pre, user_pre)
    )
    if not (
        restore.get("services_quiesced") is True
        and restore.get("attempted") is True
        and restore.get("success") is True
        and pre == post
        and embodied_active_count == 1
    ):
        raise ValueError("post-flash service restoration contract failed")
    return restore


def _validate_pass_validation(path: Path, label: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    checks = payload.get("checks") if isinstance(payload.get("checks"), list) else []
    if not (
        payload.get("schema_version") == "xrd-f407-postflash-interlock-validation-v1"
        and payload.get("overall") == "PASS"
        and isinstance(payload.get("counts"), dict)
        and payload["counts"].get("FAIL") == 0
        and len(checks) >= 14
        and all(isinstance(item, dict) and item.get("status") == "PASS" for item in checks)
    ):
        raise ValueError(f"{label} is not a strict PASS validation")


def _validate_recovered_chain(
    manifest: dict[str, Any],
    manifest_source: Path,
    evidence_root: Path,
) -> None:
    physical = (
        manifest.get("physical_interlock")
        if isinstance(manifest.get("physical_interlock"), dict)
        else {}
    )
    recovery = manifest.get("recovery") if isinstance(manifest.get("recovery"), dict) else {}
    source_path = _regular_file(
        Path(str(physical.get("source_manifest") or "")).expanduser(),
        "original failed post-flash manifest",
    )
    source_copy = _regular_file(
        manifest_source.parent / RECOVERY_ARTIFACT_SPECS["source_failed_manifest"][0],
        "failed post-flash manifest copy",
    )
    source_copy_entry = _artifact_entry(manifest, "artifacts", "source_failed_manifest")
    if not (
        source_path.name == "postflash_interlock_manifest.json"
        and source_path.parent.parent == evidence_root
        and source_path.parent.name.startswith("postflash_")
        and Path(str(physical.get("source_manifest_copy") or "")).expanduser().resolve() == source_copy
        and Path(str(source_copy_entry.get("path") or "")).expanduser().resolve() == source_copy
    ):
        raise ValueError("recovered post-flash source path contract failed")
    source_sha = sha256_file(source_path)
    source_copy_sha = _validate_digest(source_copy_entry, source_copy, "failed manifest copy")
    if not (
        source_sha == source_copy_sha
        and physical.get("source_manifest_sha256") == source_sha
        and physical.get("source_manifest_copy_sha256") == source_copy_sha
    ):
        raise ValueError("recovered post-flash source manifest hash chain failed")

    source = json.loads(source_copy.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("failed post-flash source manifest root must be an object")
    source_finished = float(source.get("finished_at_unix") or 0.0)
    recovery_started = float(recovery.get("started_at_unix") or 0.0)
    recovery_finished = float(recovery.get("finished_at_unix") or 0.0)
    if not (
        source.get("schema_version") == MANIFEST_SCHEMA_V1
        and source.get("overall") == "FAIL"
        and source.get("failure_reason") == RECOVERABLE_SOURCE_FAILURE
        and physical.get("source_schema_version") == MANIFEST_SCHEMA_V1
        and physical.get("source_overall") == "FAIL"
        and physical.get("source_failure_reason") == RECOVERABLE_SOURCE_FAILURE
        and physical.get("raw_interlock_validation") == "PASS"
        and source.get("started_at_unix") == manifest.get("started_at_unix")
        and 0.0 < source_finished <= recovery_started <= recovery_finished
        and recovery_finished == float(manifest.get("finished_at_unix") or 0.0)
    ):
        raise ValueError("recovered post-flash physical source contract failed")

    for key in (
        "hostname",
        "operator_confirmation",
        "command_contract",
        "serial_exclusivity",
        "service_restore",
    ):
        if source.get(key) != manifest.get(key):
            raise ValueError(f"recovered post-flash source {key} was not preserved")
    _validate_hostname_confirmation(source)
    source_artifacts = source.get("artifacts") if isinstance(source.get("artifacts"), dict) else {}
    source_report = source_artifacts.get("interlock_report") if isinstance(source_artifacts.get("interlock_report"), dict) else {}
    _validate_command_contract(source, Path(str(source_report.get("path") or "")))
    _validate_serial_restore(source)

    for key in ("interlock_report", "interlock_log", "validation_report"):
        source_entry = source_artifacts.get(key) if isinstance(source_artifacts.get(key), dict) else {}
        recovered_entry = _artifact_entry(manifest, "artifacts", key)
        source_file = _regular_file(source_path.parent / BASE_ARTIFACT_SPECS[key][0], f"original {key}")
        if not (
            Path(str(source_entry.get("path") or "")).expanduser().resolve() == source_file
            and _validate_digest(source_entry, source_file, f"original {key}")
            == recovered_entry.get("sha256")
            and int(source_entry.get("size_bytes") or -1)
            == int(recovered_entry.get("size_bytes") or -2)
        ):
            raise ValueError(f"recovered post-flash raw artifact {key} chain failed")
    for key in ("post_firmware_topic", "post_identity_topic", "post_estop_topic"):
        source_entry = source_artifacts.get(key) if isinstance(source_artifacts.get(key), dict) else {}
        if source_entry.get("exists") is not False:
            raise ValueError(f"failed physical source unexpectedly contains {key}")

    source_tooling = source.get("tooling") if isinstance(source.get("tooling"), dict) else {}
    top_tooling = manifest.get("tooling") if isinstance(manifest.get("tooling"), dict) else {}
    snapshots = physical.get("tool_snapshots") if isinstance(physical.get("tool_snapshots"), dict) else {}
    for key, (snapshot_name, _destination_name) in V2_PHYSICAL_TOOL_SPECS.items():
        snapshot_entry = snapshots.get(key) if isinstance(snapshots.get(key), dict) else {}
        snapshot = _regular_file(manifest_source.parent / snapshot_name, f"physical tool snapshot {key}")
        if not (
            Path(str(snapshot_entry.get("path") or "")).expanduser().resolve() == snapshot
            and _validate_digest(snapshot_entry, snapshot, f"physical tool snapshot {key}")
            == source_tooling.get(key, {}).get("sha256")
            == top_tooling.get(key, {}).get("sha256")
            and source_tooling.get(key) == top_tooling.get(key)
        ):
            raise ValueError(f"physical tool snapshot {key} chain failed")

    service_pre = recovery.get("service_state_pre") if isinstance(recovery.get("service_state_pre"), dict) else {}
    service_post = recovery.get("service_state_post") if isinstance(recovery.get("service_state_post"), dict) else {}
    recovery_tool = recovery.get("tool") if isinstance(recovery.get("tool"), dict) else {}
    if not (
        recovery.get("schema_version") == RECOVERY_SCHEMA_V1
        and recovery.get("mode") == "post_restore_readonly_only"
        and recovery.get("serial_device_opened") is False
        and recovery.get("ros_graph_writes") is False
        and recovery.get("services_stopped") is False
        and recovery.get("services_started_or_restarted") is False
        and recovery.get("physical_commands_sent") is False
        and service_pre == service_post == manifest["service_restore"]["post"]
        and recovery_tool == top_tooling.get("recovery")
    ):
        raise ValueError("post-flash read-only recovery contract failed")

    recovery_validation = _regular_file(
        manifest_source.parent / RECOVERY_ARTIFACT_SPECS["recovery_revalidation"][0],
        "recovery interlock revalidation",
    )
    recovery_validation_entry = _artifact_entry(manifest, "artifacts", "recovery_revalidation")
    if Path(str(recovery_validation_entry.get("path") or "")).expanduser().resolve() != recovery_validation:
        raise ValueError("recovery validation path is not canonical")
    _validate_digest(recovery_validation_entry, recovery_validation, "recovery revalidation")
    _validate_pass_validation(recovery_validation, "recovery revalidation")


def _validate_manifest_semantics(
    manifest: dict[str, Any],
    manifest_source: Path,
    evidence_root: Path,
) -> None:
    started = float(manifest.get("started_at_unix") or 0.0)
    finished = float(manifest.get("finished_at_unix") or 0.0)
    now = time.time()
    if not (
        manifest.get("schema_version") in {MANIFEST_SCHEMA_V1, MANIFEST_SCHEMA_V2}
        and manifest.get("overall") == "PASS"
        and manifest.get("failure_reason") == ""
        and 0.0 < started <= finished <= now + 5.0
        and 0.0 <= now - finished <= 86400.0
    ):
        raise ValueError("post-flash manifest status or freshness contract failed")

    _validate_hostname_confirmation(manifest)
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    interlock = artifacts.get("interlock_report") if isinstance(artifacts.get("interlock_report"), dict) else {}
    expected_report = Path(str(interlock.get("path") or ""))
    if manifest.get("schema_version") == MANIFEST_SCHEMA_V2:
        physical = manifest.get("physical_interlock") if isinstance(manifest.get("physical_interlock"), dict) else {}
        source_path = Path(str(physical.get("source_manifest") or ""))
        expected_report = source_path.parent / BASE_ARTIFACT_SPECS["interlock_report"][0]
    _validate_command_contract(manifest, expected_report)
    _validate_serial_restore(manifest)
    if manifest.get("schema_version") == MANIFEST_SCHEMA_V2:
        _validate_recovered_chain(manifest, manifest_source, evidence_root)


def stage_bundle(manifest_path: Path, out_dir: Path, home_dir: Path) -> dict[str, Any]:
    home = home_dir.expanduser().resolve()
    evidence_root = (home / "f407_postflash_acceptance").resolve()
    tools_root = (home / "tools").resolve()
    manifest_source = _regular_file(manifest_path.expanduser(), "post-flash manifest")

    allowed_names = {
        "postflash_interlock_manifest.json",
        "postflash_interlock_recovered_manifest.json",
    }
    if manifest_source.name not in allowed_names:
        raise ValueError("post-flash manifest filename is not canonical")
    if manifest_source.parent.parent != evidence_root or not manifest_source.parent.name.startswith("postflash_"):
        raise ValueError(f"post-flash manifest is outside the canonical evidence root: {manifest_source}")

    destination = out_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("post-flash manifest root must be an object")
    schema = manifest.get("schema_version")
    expected_name = (
        "postflash_interlock_manifest.json"
        if schema == MANIFEST_SCHEMA_V1
        else "postflash_interlock_recovered_manifest.json"
    )
    if schema not in {MANIFEST_SCHEMA_V1, MANIFEST_SCHEMA_V2} or manifest_source.name != expected_name:
        raise ValueError("post-flash manifest schema/filename pairing is invalid")
    _validate_manifest_semantics(manifest, manifest_source, evidence_root)

    sources: dict[str, tuple[Path, str, str]] = {}
    artifact_specs = dict(BASE_ARTIFACT_SPECS)
    if schema == MANIFEST_SCHEMA_V2:
        artifact_specs.update(RECOVERY_ARTIFACT_SPECS)
    for key, (source_name, destination_name) in artifact_specs.items():
        entry = _artifact_entry(manifest, "artifacts", key)
        expected_source = manifest_source.parent / source_name
        source = _regular_file(expected_source, f"artifact {key}")
        entry_path = Path(str(entry.get("path") or "")).expanduser().resolve()
        if source != entry_path:
            raise ValueError(f"artifact {key} path is not canonical")
        digest = _validate_digest(entry, source, f"artifact {key}")
        sources[f"artifact:{key}"] = (source, destination_name, digest)

    if schema == MANIFEST_SCHEMA_V1:
        for key, (source_name, destination_name) in V1_TOOL_SPECS.items():
            entry = _artifact_entry(manifest, "tooling", key)
            expected_source = tools_root / source_name
            source = _regular_file(expected_source, f"tool {key}")
            entry_path = Path(str(entry.get("path") or "")).expanduser().resolve()
            if source != entry_path:
                raise ValueError(f"tool {key} path is not canonical")
            digest = _validate_digest(entry, source, f"tool {key}")
            sources[f"tool:{key}"] = (source, destination_name, digest)
    else:
        physical = manifest.get("physical_interlock") if isinstance(manifest.get("physical_interlock"), dict) else {}
        snapshots = physical.get("tool_snapshots") if isinstance(physical.get("tool_snapshots"), dict) else {}
        for key, (source_name, destination_name) in V2_PHYSICAL_TOOL_SPECS.items():
            entry = snapshots.get(key) if isinstance(snapshots.get(key), dict) else {}
            source = _regular_file(manifest_source.parent / source_name, f"physical tool snapshot {key}")
            entry_path = Path(str(entry.get("path") or "")).expanduser().resolve()
            if source != entry_path:
                raise ValueError(f"physical tool snapshot {key} path is not canonical")
            digest = _validate_digest(entry, source, f"physical tool snapshot {key}")
            sources[f"tool:physical_{key}"] = (source, destination_name, digest)

        recovery = manifest.get("recovery") if isinstance(manifest.get("recovery"), dict) else {}
        entry = recovery.get("tool") if isinstance(recovery.get("tool"), dict) else {}
        source_name, destination_name = V2_RECOVERY_TOOL_SPEC
        source = _regular_file(tools_root / source_name, "read-only recovery tool")
        entry_path = Path(str(entry.get("path") or "")).expanduser().resolve()
        if source != entry_path:
            raise ValueError("read-only recovery tool path is not canonical")
        digest = _validate_digest(entry, source, "read-only recovery tool")
        sources["tool:recovery"] = (source, destination_name, digest)

    copied: dict[str, Any] = {}
    manifest_destination = destination / "f407_postflash_manifest.json"
    shutil.copy2(manifest_source, manifest_destination)
    copied["manifest"] = {
        "path": str(manifest_destination),
        "sha256": sha256_file(manifest_destination),
        "size_bytes": manifest_destination.stat().st_size,
    }
    for label, (source, destination_name, expected_sha) in sources.items():
        target = destination / destination_name
        shutil.copy2(source, target)
        actual_sha = sha256_file(target)
        if actual_sha != expected_sha:
            raise RuntimeError(f"copied {label} failed post-copy SHA-256 verification")
        copied[label] = {
            "path": str(target),
            "sha256": actual_sha,
            "size_bytes": target.stat().st_size,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_unix": time.time(),
        "overall": "PASS",
        "source_manifest": str(manifest_source),
        "source_manifest_sha256": sha256_file(manifest_source),
        "source_directory": str(manifest_source.parent),
        "destination_directory": str(destination),
        "copied": copied,
        "read_only_source": True,
        "physical_hardware_touched": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--home", default=str(Path.home()))
    parser.add_argument("--index", default="")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    index_path = Path(args.index).expanduser().resolve() if args.index else out_dir / "f407_postflash_bundle_index.json"
    try:
        report = stage_bundle(Path(args.manifest), out_dir, Path(args.home))
    except Exception as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_unix": time.time(),
            "overall": "FAIL",
            "source_manifest": str(Path(args.manifest).expanduser()),
            "destination_directory": str(out_dir),
            "read_only_source": True,
            "physical_hardware_touched": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"overall": report["overall"], "index": str(index_path)}, sort_keys=True))
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
