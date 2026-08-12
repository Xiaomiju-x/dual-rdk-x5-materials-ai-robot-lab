"""Side-effect-free runtime identity helpers for AI-brain inference services."""

from __future__ import annotations

import hashlib
import os
import platform
import sys
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from rb_voe.contracts.canonical import canonical_sha256, file_sha256

RUNTIME_IDENTITY_SCHEMA_VERSION: Final[str] = "xrd-rb-voe-runtime-identity-v1"
PROCESS_SESSION_ID: Final[str] = uuid.uuid4().hex


def local_boot_id() -> str:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError:
        value = ""
    return value


def local_device_id() -> str:
    explicit = os.environ.get("RB_VOE_DEVICE_ID", "").strip()
    if explicit:
        return explicit
    candidates = (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id"))
    raw = ""
    for path in candidates:
        try:
            raw = path.read_text(encoding="ascii").strip()
        except OSError:
            continue
        if raw:
            break
    if not raw:
        return ""
    return "machine-sha256:" + hashlib.sha256(raw.encode("ascii")).hexdigest()


def _hash_existing_files(files: Mapping[str, str | Path | None]) -> tuple[dict[str, str], list[str]]:
    digests: dict[str, str] = {}
    missing: list[str] = []
    for logical_name, raw_path in sorted(files.items()):
        if not logical_name or raw_path is None:
            missing.append(str(logical_name or "unnamed"))
            continue
        path = Path(raw_path).expanduser()
        if not path.is_file():
            missing.append(logical_name)
            continue
        digests[logical_name] = file_sha256(path)
    return digests, missing


def build_runtime_identity(
    *,
    line_id: str,
    backend: str,
    model_files: Mapping[str, str | Path | None],
    preprocess_files: Mapping[str, str | Path | None],
    calibration_files: Mapping[str, str | Path | None],
    calibration_payload: Mapping[str, Any],
    last_success_at_ms: int,
    success_count: int,
) -> dict[str, Any]:
    """Build a strict identity snapshot without loading a model or opening hardware."""
    model_sha256, missing_models = _hash_existing_files(model_files)
    preprocess_sha256, missing_preprocess = _hash_existing_files(preprocess_files)
    calibration_sha256, missing_calibration = _hash_existing_files(calibration_files)
    if calibration_payload:
        calibration_sha256["runtime_config"] = canonical_sha256(calibration_payload)
    missing = sorted(missing_models + missing_preprocess + missing_calibration)
    observed_at_ms = time.time_ns() // 1_000_000
    boot_id = local_boot_id()
    device_id = local_device_id()
    valid_success_at = (
        not isinstance(last_success_at_ms, bool)
        and isinstance(last_success_at_ms, int)
        and 0 < last_success_at_ms <= observed_at_ms
    )
    valid_success_count = (
        not isinstance(success_count, bool) and isinstance(success_count, int) and success_count > 0
    )
    ready = bool(
        isinstance(line_id, str)
        and line_id
        and isinstance(backend, str)
        and backend
        and device_id
        and boot_id
        and valid_success_count
        and valid_success_at
        and model_sha256
        and preprocess_sha256
        and calibration_sha256
        and not missing
    )
    payload: dict[str, Any] = {
        "schema_version": RUNTIME_IDENTITY_SCHEMA_VERSION,
        "line_id": line_id,
        "ready": ready,
        "reason_code": "PASS" if ready else "RUNTIME_IDENTITY_INCOMPLETE",
        "device_id": device_id,
        "boot_id": boot_id,
        "session_id": PROCESS_SESSION_ID,
        "backend": backend,
        "model_sha256": model_sha256,
        "preprocess_sha256": preprocess_sha256,
        "calibration_sha256": calibration_sha256,
        "missing_artifacts": missing,
        "last_success_at_ms": last_success_at_ms if valid_success_at else 0,
        "success_count": success_count if valid_success_count else 0,
        "observed_at_ms": observed_at_ms,
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable_name": Path(sys.executable).name,
        },
        "identity_probe": {
            "method": "GET",
            "model_loaded_by_probe": False,
            "inference_triggered_by_probe": False,
            "hardware_touched_by_probe": False,
            "execution_authority": False,
        },
    }
    payload["identity_sha256"] = canonical_sha256(payload)
    return payload


__all__ = [
    "PROCESS_SESSION_ID",
    "RUNTIME_IDENTITY_SCHEMA_VERSION",
    "build_runtime_identity",
    "local_boot_id",
    "local_device_id",
]
