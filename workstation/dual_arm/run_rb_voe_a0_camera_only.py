#!/usr/bin/env python3
"""Fail-closed camera-only A0 producer for the finals arm02 overhead camera.

No arguments is always PlanOnly.  Production sealing is reachable only from
the CLI-owned path that constructs the exact built-in transport, inference,
and local-identity implementations.  Callers that inject dependencies can
exercise the state machine, but their output is permanently marked
SIMULATED_COUNTERFACTUAL and can never create a production A0 record.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import secrets
import shlex
import socket
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from workstation.dual_arm import rb_voe_overhead_acquisition_manifest as acquisition
    from workstation.dual_arm import rb_voe_overhead_actual_record as contract
except ModuleNotFoundError:  # pragma: no cover - direct AI-X5 execution
    import rb_voe_overhead_acquisition_manifest as acquisition  # type: ignore[no-redef]
    import rb_voe_overhead_actual_record as contract  # type: ignore[no-redef]


RUNNER_SCHEMA = "xrd-rb-voe-a0-camera-only-runner-v2"
STATE_SCHEMA = "xrd-rb-voe-a0-camera-only-state-v2"
BINDING_SCHEMA = "xrd-rb-voe-a0-camera-only-binding-v2"
PLAN_SCHEMA = "xrd-rb-voe-a0-camera-only-plan-v2"
REMOTE_ENVELOPE_SCHEMA = "xrd-rb-voe-arm02-camera-envelope-v2"
RESERVATION_SCHEMA = "xrd-rb-voe-a0-challenge-reservation-v1"
PRODUCED_SCHEMA = "xrd-rb-voe-a0-challenge-produced-v1"

ARM02_HOST = "192.0.2.136"
ARM02_USER = "er"
ARM02_HOSTNAME = contract.FROZEN_CAPTURE_HOSTNAME
AI_HOSTNAME = contract.FROZEN_INFERENCE_HOSTNAME
CAMERA_BASE_URL = "http://127.0.0.1:8892"
# The evidence contract records the loopback endpoint actually used for GETs.
# Kernel socket recognition separately freezes the service's IPv4 wildcard bind.
CAMERA_LISTENER_IP = contract.FROZEN_CAMERA_LISTENER_IP
CAMERA_IPV4_BIND_IP = "0.0.0.0"
CAMERA_LISTENER_PORT = 8892
CAMERA_UNIT = "xrd-overhead-camera.service"
CAMERA_UNIT_PATH = "/etc/systemd/system/xrd-overhead-camera.service"
CAMERA_UNIT_SHA256 = "99669577154286055ea449227b86bf8221efeed2df438f5e5f9dc96fadf388e2"
CAMERA_UNIT_SIZE_BYTES = 516
CAMERA_SCRIPT_PATH = "/home/rdk/dual_arm/overhead_camera_service.py"
CAMERA_CMDLINE = ("/usr/bin/python3", CAMERA_SCRIPT_PATH)
CAMERA_PROCESS_USER = "er"
SSH_EXECUTABLE = "/usr/bin/ssh"
AI_PYTHON_EXECUTABLE = "/usr/bin/python3"

FROZEN_KNOWN_HOSTS_SHA256 = "79fc15d37314f1abeae2b07952695f666c993272453fc582b6e571e42dd4212f"
FROZEN_ARM02_KNOWN_HOST = (
    "192.0.2.136 ssh-ed25519 REPLACE_WITH_VERIFIED_HOST_KEY"
)
LIVE_CHALLENGE_SCHEMA = "xrd-rb-voe-live-shadow-challenge-v4"
LIVE_CHALLENGE_PURPOSE = "BIND_A0_ACTUAL_EVIDENCE_THEN_RUN_LIVE_READONLY_SHADOW"
LIVE_ISSUANCE_SCHEMA = "xrd-rb-voe-live-shadow-challenge-issuance-v2"
LIVE_CONSUMPTION_SCHEMA = "xrd-rb-voe-live-shadow-challenge-consumption-v2"

TASK_KIND = "BAG_DROP_IN_GRINDING_DISH"
RESULT_SCHEMA = "xrd-overhead-bag-presence-v2"
SUCCESS_STATE = "BAG_PRESENT"
FRAME_COUNT = 5
DEFAULT_MAX_FRAME_AGE_S = 1.5
MAX_MAX_FRAME_AGE_S = 2.0
DEFAULT_SNAPSHOT_BRACKET_MS = 8_000
MAX_SNAPSHOT_BRACKET_MS = 10_000
DEFAULT_INTERVAL_S = 0.25
MIN_INTERVAL_S = 0.05
MAX_INTERVAL_S = 2.0
MAX_BASELINE_AGE_MS = 5 * 60 * 1000
MAX_REMOTE_CLOCK_SKEW_MS = 5_000
MAX_SSH_TIMEOUT_S = 30.0
MAX_REMOTE_RESPONSE_BYTES = contract.MAX_FRAME_BYTES * 2
MAX_INFERENCE_STDOUT_BYTES = contract.MAX_RESULT_JSON_BYTES
MAX_SUBPROCESS_STDERR_BYTES = 64 * 1024
_SUBPROCESS_READ_CHUNK_BYTES = 64 * 1024

PRODUCTION_SSH_TIMEOUT_S = 15.0
PRODUCTION_INFERENCE_TIMEOUT_S = 120.0
LIVE_COLLECTOR_PREFLIGHT_TIMEOUT_S = (12.0, 8.0, 8.0)
LIVE_CHALLENGE_SEALING_MARGIN_MS = 60_000
A0_OCCUPIED_WORST_CASE_MS = int(
    (
        FRAME_COUNT * 3 * PRODUCTION_SSH_TIMEOUT_S
        + (FRAME_COUNT - 1) * MAX_INTERVAL_S
        + PRODUCTION_INFERENCE_TIMEOUT_S
    )
    * 1000
)
LIVE_COLLECTOR_PREFLIGHT_WORST_CASE_MS = int(sum(LIVE_COLLECTOR_PREFLIGHT_TIMEOUT_S) * 1000)
LIVE_CHALLENGE_MIN_BUDGET_MS = (
    A0_OCCUPIED_WORST_CASE_MS + LIVE_COLLECTOR_PREFLIGHT_WORST_CASE_MS + LIVE_CHALLENGE_SEALING_MARGIN_MS
)
LIVE_CHALLENGE_TTL_MS = 10 * 60 * 1000
if LIVE_CHALLENGE_TTL_MS < LIVE_CHALLENGE_MIN_BUDGET_MS:  # pragma: no cover - import-time invariant
    raise RuntimeError("live challenge TTL is below the frozen worst-case execution budget")

A0_RUN_SCOPE_SENTINEL = "_A0_RUN_SCOPE_"
A0_RUN_SCOPE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}\Z")

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
BOOT_ID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")

PLAN_BLOCKERS = (
    "EMPTY_AND_OCCUPIED_CAPTURE_REQUIRE_OPERATOR_CONFIRMED_PHYSICAL_STATES",
    "REMOTE_FETCH_TIMESTAMP_IS_BOUND_BUT_IS_NOT_A_SENSOR_EXPOSURE_TIMESTAMP",
    "LIVE_CHALLENGE_AND_ARM02_CAMERA_SERVICE_MUST_ALREADY_EXIST",
)

CHALLENGE_KEYS = frozenset(
    {
        "schema_version",
        "challenge_sha256",
        "challenge_content_sha256",
        "config_sha256",
        "bootstrap_manifest_sha256",
        "bootstrap_manifest_file_sha256",
        "release_id",
        "known_hosts_file",
        "known_hosts_sha256",
        "case_id",
        "sample_id",
        "sample_lineage_sha256",
        "parent_evidence_root_sha256",
        "bag_empty_baseline_sha256",
        "task_kind",
        "result_schema",
        "success_state",
        "run_id",
        "run_nonce",
        "run_nonce_sha256",
        "acquisition_id",
        "a0_run_id",
        "issued_at_ms",
        "expires_at_ms",
        "profile_sha256",
        "purpose",
        "remote_contacted",
        "network_touched",
        "execution_authority",
        "transport_commands_issued",
        "read_only_device_observations",
        "actuator_commands_issued",
        "mutating_commands_issued",
        "read_only_transport_operations",
    }
)

STATE_KEYS = frozenset(
    {
        "schema_version",
        "run_dir",
        "authority_mode",
        "status",
        "created_at_ms",
        "updated_at_ms",
        "parameters",
        "capture_identity",
        "inference_identity",
        "capture_service_identity",
        "capture_session_id",
        "inference_session_id",
        "binding_sha256",
        "empty_frames",
        "occupied_frames",
        "empty_result",
        "bag_empty_baseline_sha256",
        "empty_completed_at_ms",
        "challenge_sha256",
        "reservation_sha256",
        "frame_bundle",
        "inference_result",
        "inference_started_at_ms",
        "inference_completed_at_ms",
        "acquisition_manifest",
        "actual_record",
        "replay_receipt",
        "remote_contacted",
        "read_only_transport_operations",
        "last_error",
        "state_sha256",
    }
)

FRAME_ROW_KEYS = frozenset(
    {
        "file_name",
        "sha256",
        "size_bytes",
        "width",
        "height",
        "frame_id",
        "captured_at_ms",
        "health_before_at_ms",
        "health_after_at_ms",
        "service_identity_sha256",
    }
)

SERVICE_KEYS = frozenset(
    {
        "pid",
        "cmdline",
        "script_path",
        "script_sha256",
        "script_size_bytes",
        "unit_name",
        "unit_active_state",
        "unit_sub_state",
        "unit_main_pid",
        "unit_fragment_path",
        "unit_sha256",
        "unit_size_bytes",
        "service_user",
        "video_device",
        "video_owner_pid",
        "video_owner_uid",
        "video_owner_user",
        "usb_id",
        "listener_ip",
        "listener_port",
        "listener_inode",
        "listener_owner_pid",
        "service_identity_sha256",
    }
)


class CameraOnlyError(RuntimeError):
    """Raised whenever camera-only evidence must stop fail-closed."""


@dataclass(frozen=True)
class HostIdentity:
    hostname: str
    device_id: str
    boot_id: str

    def as_dict(self) -> dict[str, str]:
        return {
            "hostname": self.hostname,
            "device_id": self.device_id,
            "boot_id": self.boot_id,
        }


@dataclass(frozen=True)
class CameraServiceIdentity:
    pid: int
    cmdline: tuple[str, ...]
    script_path: str
    script_sha256: str
    script_size_bytes: int
    script_bytes: bytes
    unit_name: str
    unit_active_state: str
    unit_sub_state: str
    unit_main_pid: int
    unit_fragment_path: str
    unit_sha256: str
    unit_size_bytes: int
    service_user: str
    video_device: str
    video_owner_pid: int
    video_owner_uid: int
    video_owner_user: str
    usb_id: str
    listener_ip: str
    listener_port: int
    listener_inode: int
    listener_owner_pid: int

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "cmdline": list(self.cmdline),
            "script_path": self.script_path,
            "script_sha256": self.script_sha256,
            "script_size_bytes": self.script_size_bytes,
            "unit_name": self.unit_name,
            "unit_active_state": self.unit_active_state,
            "unit_sub_state": self.unit_sub_state,
            "unit_main_pid": self.unit_main_pid,
            "unit_fragment_path": self.unit_fragment_path,
            "unit_sha256": self.unit_sha256,
            "unit_size_bytes": self.unit_size_bytes,
            "service_user": self.service_user,
            "video_device": self.video_device,
            "video_owner_pid": self.video_owner_pid,
            "video_owner_uid": self.video_owner_uid,
            "video_owner_user": self.video_owner_user,
            "usb_id": self.usb_id,
            "listener_ip": self.listener_ip,
            "listener_port": self.listener_port,
            "listener_inode": self.listener_inode,
            "listener_owner_pid": self.listener_owner_pid,
        }

    def as_dict(self) -> dict[str, Any]:
        value = self.unsigned_dict()
        value["service_identity_sha256"] = canonical_sha256(value)
        return value


@dataclass(frozen=True)
class RemoteObservation:
    identity: HostIdentity
    fetched_at_ms: int
    received_at_ms: int
    frame_id: str
    payload: bytes
    service: CameraServiceIdentity


@dataclass(frozen=True)
class _ValidatedChallenge:
    path: Path
    value: Mapping[str, Any]
    artifact_sha256: str
    consumed: Mapping[str, Any] | None


@dataclass(frozen=True)
class _TerminalPaths:
    evidence_dir: Path
    replay_dir: Path
    capture_artifact: Path
    inference_artifact: Path
    raw_frame: Path
    frame_bundle: Path
    result_json: Path
    manifest: Path
    record: Path
    receipt: Path
    camera_service_identity: Path


@dataclass(frozen=True)
class _OutputContract:
    config_file: Path
    config_sha256: str
    release_id: str
    root: Path
    record: Path
    manifest: Path
    raw_frame: Path
    frame_bundle: Path
    result_json: Path
    capture_artifact: Path
    inference_artifact: Path
    replay_dir: Path
    camera_service_identity: Path


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _reject_constant(value: str) -> None:
    raise CameraOnlyError(f"non-finite JSON number is forbidden: {value}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CameraOnlyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_bytes(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except CameraOnlyError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CameraOnlyError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise CameraOnlyError(f"{label} must contain one JSON object")
    return value


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise CameraOnlyError(f"{label} must be a lowercase SHA-256")
    return value


def _require_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CameraOnlyError(f"{label} must be non-empty trimmed text")
    return value


def _require_int(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CameraOnlyError(f"{label} must be an integer >= {minimum}")
    return value


def _require_finite(value: object, *, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CameraOnlyError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < minimum or number > maximum:
        raise CameraOnlyError(f"{label} is outside the frozen finite bounds")
    return number


def _absolute_lexical(path: Path | str, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise CameraOnlyError(f"{label} must be an absolute lexical path without '..'")
    return Path(os.path.abspath(os.fspath(candidate)))


def _reject_link_components(path: Path, *, allow_missing_leaf: bool = False) -> None:
    current = Path(path.anchor)
    parts = path.parts[1:]
    for index, part in enumerate(parts):
        current /= part
        if allow_missing_leaf and index == len(parts) - 1 and not current.exists():
            return
        if not current.exists():
            raise CameraOnlyError(f"path component does not exist: {current}")
        metadata = current.lstat()
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if current.is_symlink() or bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag):
            raise CameraOnlyError(f"link path component is forbidden: {current}")


def _regular_file(path: Path | str, *, label: str, max_bytes: int) -> tuple[Path, bytes]:
    lexical = _absolute_lexical(path, label=label)
    _reject_link_components(lexical)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lexical, flags)
    except OSError as exc:
        raise CameraOnlyError(f"cannot open stable {label}: {lexical}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CameraOnlyError(f"{label} must be a regular file")
        if before.st_size <= 0 or before.st_size > max_bytes:
            raise CameraOnlyError(f"{label} size is outside the allowed bound")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise CameraOnlyError(f"{label} changed while being read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise CameraOnlyError(f"{label} grew while being read")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise CameraOnlyError(f"{label} changed while being read")
    try:
        path_after = lexical.stat(follow_symlinks=False)
    except OSError as exc:
        raise CameraOnlyError(f"{label} disappeared after stable read") from exc
    if (path_after.st_dev, path_after.st_ino) != (after.st_dev, after.st_ino):
        raise CameraOnlyError(f"{label} path changed during stable read")
    data = b"".join(chunks)
    return lexical, data


@contextmanager
def _run_execution_lease(run_dir: Path):
    """Hold one cross-process advisory lock for the complete production phase."""

    lock_path = run_dir / ".a0-camera-only.lock"
    _reject_link_components(lock_path, allow_missing_leaf=True)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise CameraOnlyError("cannot open the A0 execution lease safely") from exc
    opened = os.fstat(descriptor)
    try:
        path_state = lock_path.stat(follow_symlinks=False)
    except OSError:
        os.close(descriptor)
        raise CameraOnlyError("A0 execution lease path changed while opening") from None
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if (
        not stat.S_ISREG(opened.st_mode)
        or lock_path.is_symlink()
        or bool(getattr(path_state, "st_file_attributes", 0) & reparse_flag)
        or (path_state.st_dev, path_state.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        os.close(descriptor)
        raise CameraOnlyError("A0 execution lease must be one stable regular non-link file")
    locked = False
    try:
        if os.name == "nt":
            import msvcrt

            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
                os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            try:
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise CameraOnlyError("another A0 camera-only process owns this run_dir") from exc
        else:
            import fcntl

            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise CameraOnlyError("another A0 camera-only process owns this run_dir") from exc
        locked = True
        yield
    finally:
        if locked:
            if os.name == "nt":
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _prepare_run_dir(path: Path | str) -> Path:
    run_dir = _absolute_lexical(path, label="run_dir")
    _reject_link_components(run_dir.parent)
    if run_dir.exists():
        _reject_link_components(run_dir)
        if not run_dir.is_dir():
            raise CameraOnlyError("run_dir exists but is not a directory")
    else:
        run_dir.mkdir(mode=0o700)
        _fsync_directory(run_dir.parent)
    return run_dir


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace_json(path: Path, value: Mapping[str, Any]) -> None:
    data = canonical_bytes(value) + b"\n"
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_bytes_once(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise CameraOnlyError(f"write-once output already exists: {path}")
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise CameraOnlyError(f"write-once output already exists: {path}") from exc
        temporary.unlink()
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    """Create a global sidecar with O_EXCL and durable parent metadata."""

    data = canonical_bytes(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise CameraOnlyError(f"exclusive sidecar already exists: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
    except Exception:
        try:
            path.unlink()
            _fsync_directory(path.parent)
        except OSError:
            pass
        raise


def _ensure_bytes(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        _, current = _regular_file(path, label=f"existing {path.name}", max_bytes=max(1, len(data)))
        if current != data:
            raise CameraOnlyError(f"existing output differs from expected content: {path}")
        return
    _write_bytes_once(path, data)


def _state_unsigned(state: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(state)
    unsigned.pop("state_sha256", None)
    return unsigned


def _save_state(path: Path, state: dict[str, Any], *, now_ms: int) -> None:
    _require_int(now_ms, label="state now_ms", minimum=contract.MIN_TIMESTAMP_MS)
    state["updated_at_ms"] = now_ms
    state["state_sha256"] = canonical_sha256(_state_unsigned(state))
    _atomic_replace_json(path, state)


def _load_state(path: Path, run_dir: Path) -> dict[str, Any]:
    _, raw = _regular_file(path, label="state", max_bytes=2 * 1024 * 1024)
    state = _load_json_bytes(raw, label="state")
    if set(state) != STATE_KEYS or state.get("schema_version") != STATE_SCHEMA:
        raise CameraOnlyError("state schema or keys are unsupported")
    claimed = _require_sha256(state["state_sha256"], label="state_sha256")
    if claimed != canonical_sha256(_state_unsigned(state)):
        raise CameraOnlyError("state_sha256 mismatch")
    if state["run_dir"] != str(run_dir):
        raise CameraOnlyError("state belongs to a different run_dir")
    return state


def _parameters(
    *,
    max_frame_age_s: float,
    snapshot_bracket_ms: int,
    interval_s: float,
    output_contract: _OutputContract | None,
) -> dict[str, Any]:
    age = _require_finite(
        max_frame_age_s,
        label="max_frame_age_s",
        minimum=0.001,
        maximum=MAX_MAX_FRAME_AGE_S,
    )
    bracket = _require_int(snapshot_bracket_ms, label="snapshot_bracket_ms", minimum=1)
    if bracket > MAX_SNAPSHOT_BRACKET_MS:
        raise CameraOnlyError("snapshot_bracket_ms exceeds the frozen ceiling")
    interval = _require_finite(
        interval_s,
        label="interval_s",
        minimum=MIN_INTERVAL_S,
        maximum=MAX_INTERVAL_S,
    )
    return {
        "frame_count": FRAME_COUNT,
        "max_frame_age_s": age,
        "snapshot_bracket_ms": bracket,
        "interval_s": interval,
        "max_baseline_age_ms": MAX_BASELINE_AGE_MS,
        "live_config_sha256": (output_contract.config_sha256 if output_contract is not None else None),
        "production_output_root": (str(output_contract.root) if output_contract is not None else None),
    }


def _new_state(
    run_dir: Path,
    *,
    authority_mode: str,
    parameters: Mapping[str, Any],
    now_ms: int,
) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA,
        "run_dir": str(run_dir),
        "authority_mode": authority_mode,
        "status": "NEW",
        "created_at_ms": now_ms,
        "updated_at_ms": now_ms,
        "parameters": dict(parameters),
        "capture_identity": None,
        "inference_identity": None,
        "capture_service_identity": None,
        "capture_session_id": None,
        "inference_session_id": None,
        "binding_sha256": None,
        "empty_frames": [],
        "occupied_frames": [],
        "empty_result": None,
        "bag_empty_baseline_sha256": None,
        "empty_completed_at_ms": None,
        "challenge_sha256": None,
        "reservation_sha256": None,
        "frame_bundle": None,
        "inference_result": None,
        "inference_started_at_ms": None,
        "inference_completed_at_ms": None,
        "acquisition_manifest": None,
        "actual_record": None,
        "replay_receipt": None,
        "remote_contacted": False,
        "read_only_transport_operations": 0,
        "last_error": None,
        "state_sha256": "",
    }


def _state_store(
    run_dir: Path,
    *,
    authority_mode: str,
    parameters: Mapping[str, Any],
    now_ms: int,
) -> tuple[Path, dict[str, Any]]:
    state_path = run_dir / "state.json"
    if state_path.exists() or state_path.is_symlink():
        state = _load_state(state_path, run_dir)
        if state["authority_mode"] != authority_mode:
            raise CameraOnlyError("a run_dir cannot switch between simulation and production")
        if state["parameters"] != dict(parameters):
            raise CameraOnlyError("capture parameters differ from the frozen run binding")
        return state_path, state
    state = _new_state(
        run_dir,
        authority_mode=authority_mode,
        parameters=parameters,
        now_ms=now_ms,
    )
    _save_state(state_path, state, now_ms=now_ms)
    return state_path, state


def _binding_value(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": BINDING_SCHEMA,
        "authority_mode": state["authority_mode"],
        "parameters": state["parameters"],
        "capture_identity": state["capture_identity"],
        "inference_identity": state["inference_identity"],
        "capture_service_identity": state["capture_service_identity"],
        "capture_session_id": state["capture_session_id"],
        "inference_session_id": state["inference_session_id"],
        "created_at_ms": state["created_at_ms"],
    }


def _write_or_validate_binding(run_dir: Path, state: dict[str, Any]) -> None:
    path = run_dir / "run_binding.json"
    expected = _binding_value(state)
    expected_sha = canonical_sha256(expected)
    if path.exists() or path.is_symlink():
        _, raw = _regular_file(path, label="run binding", max_bytes=128 * 1024)
        if _load_json_bytes(raw, label="run binding") != expected:
            raise CameraOnlyError("run binding differs from persisted identities or parameters")
    else:
        _write_bytes_once(path, canonical_bytes(expected) + b"\n")
    if state["binding_sha256"] not in (None, expected_sha):
        raise CameraOnlyError("state binding_sha256 mismatch")
    state["binding_sha256"] = expected_sha


def _validate_binding(run_dir: Path, state: Mapping[str, Any]) -> None:
    _, raw = _regular_file(run_dir / "run_binding.json", label="run binding", max_bytes=128 * 1024)
    expected = _binding_value(state)
    if _load_json_bytes(raw, label="run binding") != expected:
        raise CameraOnlyError("run binding does not match current state")
    if state["binding_sha256"] != canonical_sha256(expected):
        raise CameraOnlyError("run binding digest mismatch")


def _device_id(machine_id: bytes) -> str:
    stripped = machine_id.strip()
    if not stripped:
        raise CameraOnlyError("machine-id is empty")
    return f"machine-sha256:{hashlib.sha256(stripped).hexdigest()}"


def _validate_identity(identity: HostIdentity, *, expected_hostname: str, label: str) -> None:
    if identity.hostname != expected_hostname:
        raise CameraOnlyError(f"{label} hostname mismatch")
    if not identity.device_id.startswith("machine-sha256:"):
        raise CameraOnlyError(f"{label} device_id is not machine-id-derived")
    _require_sha256(identity.device_id.removeprefix("machine-sha256:"), label=f"{label}.device_id")
    if BOOT_ID_RE.fullmatch(identity.boot_id) is None:
        raise CameraOnlyError(f"{label} boot_id is invalid")


def _identity_from_state(value: object, *, label: str, hostname: str) -> HostIdentity:
    if not isinstance(value, Mapping) or set(value) != {"hostname", "device_id", "boot_id"}:
        raise CameraOnlyError(f"{label} identity is malformed")
    identity = HostIdentity(
        hostname=_require_text(value["hostname"], label=f"{label}.hostname"),
        device_id=_require_text(value["device_id"], label=f"{label}.device_id"),
        boot_id=_require_text(value["boot_id"], label=f"{label}.boot_id"),
    )
    _validate_identity(identity, expected_hostname=hostname, label=label)
    return identity


def _validate_service_identity(service: CameraServiceIdentity) -> None:
    expected = contract.CAPTURE_PIPELINE_CONTRACT
    checks = (
        (service.pid > 1, "camera service PID is invalid"),
        (service.cmdline == CAMERA_CMDLINE, "camera service cmdline drifted"),
        (service.script_path == CAMERA_SCRIPT_PATH, "camera service script path drifted"),
        (service.script_sha256 == expected.sha256, "camera service script hash drifted"),
        (service.script_size_bytes == expected.size_bytes, "camera service script size drifted"),
        (len(service.script_bytes) == expected.size_bytes, "camera service script bytes are incomplete"),
        (_sha256_bytes(service.script_bytes) == expected.sha256, "camera service bytes hash drifted"),
        (service.unit_name == CAMERA_UNIT, "camera service unit identity drifted"),
        (service.unit_active_state == "active", "camera service unit is not active"),
        (service.unit_sub_state == "running", "camera service unit is not running"),
        (service.unit_main_pid == service.pid, "camera service MainPID drifted"),
        (service.unit_fragment_path == CAMERA_UNIT_PATH, "camera service unit path drifted"),
        (service.unit_sha256 == CAMERA_UNIT_SHA256, "camera service unit hash drifted"),
        (service.unit_size_bytes == CAMERA_UNIT_SIZE_BYTES, "camera service unit size drifted"),
        (service.service_user == CAMERA_PROCESS_USER, "camera service process user drifted"),
        (service.video_device == contract.FROZEN_CAMERA_SOURCE, "camera device path drifted"),
        (service.video_owner_pid == service.pid, "camera device owner PID drifted"),
        (service.video_owner_uid >= 0, "camera device owner UID is invalid"),
        (service.video_owner_user == CAMERA_PROCESS_USER, "camera device owner user drifted"),
        (service.usb_id == contract.FROZEN_CAMERA_USB_ID, "camera USB identity drifted"),
        (service.listener_ip == CAMERA_LISTENER_IP, "camera listener IP drifted"),
        (service.listener_port == CAMERA_LISTENER_PORT, "camera listener port drifted"),
        (service.listener_inode > 0, "camera listener inode is invalid"),
        (service.listener_owner_pid == service.pid, "camera listener owner PID drifted"),
    )
    for ok, message in checks:
        if not ok:
            raise CameraOnlyError(message)


def _service_from_value(value: object, *, script_bytes: bytes) -> CameraServiceIdentity:
    if not isinstance(value, Mapping) or set(value) != SERVICE_KEYS - {"service_identity_sha256"}:
        raise CameraOnlyError("remote camera service identity keys are not exact")
    cmdline = value["cmdline"]
    if not isinstance(cmdline, list) or any(not isinstance(item, str) for item in cmdline):
        raise CameraOnlyError("remote camera service cmdline is malformed")
    service = CameraServiceIdentity(
        pid=_require_int(value["pid"], label="service.pid", minimum=2),
        cmdline=tuple(cmdline),
        script_path=_require_text(value["script_path"], label="service.script_path"),
        script_sha256=_require_sha256(value["script_sha256"], label="service.script_sha256"),
        script_size_bytes=_require_int(
            value["script_size_bytes"], label="service.script_size_bytes", minimum=1
        ),
        script_bytes=script_bytes,
        unit_name=_require_text(value["unit_name"], label="service.unit_name"),
        unit_active_state=_require_text(value["unit_active_state"], label="service.unit_active_state"),
        unit_sub_state=_require_text(value["unit_sub_state"], label="service.unit_sub_state"),
        unit_main_pid=_require_int(value["unit_main_pid"], label="service.unit_main_pid", minimum=2),
        unit_fragment_path=_require_text(value["unit_fragment_path"], label="service.unit_fragment_path"),
        unit_sha256=_require_sha256(value["unit_sha256"], label="service.unit_sha256"),
        unit_size_bytes=_require_int(value["unit_size_bytes"], label="service.unit_size_bytes", minimum=1),
        service_user=_require_text(value["service_user"], label="service.service_user"),
        video_device=_require_text(value["video_device"], label="service.video_device"),
        video_owner_pid=_require_int(value["video_owner_pid"], label="service.video_owner_pid", minimum=2),
        video_owner_uid=_require_int(value["video_owner_uid"], label="service.video_owner_uid", minimum=0),
        video_owner_user=_require_text(value["video_owner_user"], label="service.video_owner_user"),
        usb_id=_require_text(value["usb_id"], label="service.usb_id"),
        listener_ip=_require_text(value["listener_ip"], label="service.listener_ip"),
        listener_port=_require_int(value["listener_port"], label="service.listener_port", minimum=1),
        listener_inode=_require_int(value["listener_inode"], label="service.listener_inode", minimum=1),
        listener_owner_pid=_require_int(
            value["listener_owner_pid"], label="service.listener_owner_pid", minimum=2
        ),
    )
    _validate_service_identity(service)
    return service


def _validate_service_state_value(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != SERVICE_KEYS:
        raise CameraOnlyError("persisted camera service identity is malformed")
    unsigned = dict(value)
    claimed = _require_sha256(unsigned.pop("service_identity_sha256"), label="service_identity_sha256")
    if canonical_sha256(unsigned) != claimed:
        raise CameraOnlyError("camera service identity digest mismatch")
    return dict(value)


def _validate_known_hosts(path: Path | str) -> Path:
    known_hosts, raw = _regular_file(path, label="known_hosts", max_bytes=64 * 1024)
    if any(character in str(known_hosts) for character in ("%", "~", "*", "?", "[", "]")) or any(
        character.isspace() for character in str(known_hosts)
    ):
        raise CameraOnlyError("known_hosts path contains OpenSSH expansion syntax")
    if _sha256_bytes(raw) != FROZEN_KNOWN_HOSTS_SHA256:
        raise CameraOnlyError("known_hosts does not match the frozen release digest")
    lines = [
        line.strip() for line in raw.decode("utf-8").splitlines() if line.strip() and not line.startswith("#")
    ]
    if lines.count(FROZEN_ARM02_KNOWN_HOST) != 1:
        raise CameraOnlyError("known_hosts lacks the single frozen arm02 ED25519 key")
    return known_hosts


def _load_output_contract(
    config_file: Path | str,
    *,
    run_dir: Path | str,
    known_hosts: Path,
) -> _OutputContract:
    config_path, raw = _regular_file(
        config_file,
        label="live config",
        max_bytes=2 * 1024 * 1024,
    )
    config = _load_json_bytes(raw, label="live config")
    if set(config) != {
        "schema_version",
        "release_id",
        "known_hosts_file",
        "known_hosts_sha256",
        "bootstrap_manifest_file",
        "targets",
        "expected",
        "overhead",
        "config_sha256",
    }:
        raise CameraOnlyError("live config keys are not exact")
    if config["schema_version"] != "xrd-rb-voe-live-shadow-config-v4":
        raise CameraOnlyError("live config schema is unsupported")
    claimed = _require_sha256(config["config_sha256"], label="live config config_sha256")
    unsigned = dict(config)
    unsigned.pop("config_sha256")
    if canonical_sha256(unsigned) != claimed:
        raise CameraOnlyError("live config digest mismatch")
    if (
        config["known_hosts_file"] != str(known_hosts)
        or config["known_hosts_sha256"] != FROZEN_KNOWN_HOSTS_SHA256
    ):
        raise CameraOnlyError("live config known_hosts binding mismatch")
    overhead = config["overhead"]
    overhead_keys = {
        "record",
        "acquisition_manifest",
        "raw_frame",
        "frame_bundle_artifact",
        "result_json",
        "capture_pipeline_artifact",
        "inference_pipeline_artifact",
        "camera_service_identity_artifact",
        "replay_ledger_dir",
        "task_kind",
        "result_schema",
        "success_state",
    }
    if not isinstance(overhead, Mapping) or set(overhead) != overhead_keys:
        raise CameraOnlyError("live config overhead keys are not exact")
    if (overhead["task_kind"], overhead["result_schema"], overhead["success_state"]) != (
        TASK_KIND,
        RESULT_SCHEMA,
        SUCCESS_STATE,
    ):
        raise CameraOnlyError("live config overhead task/result tuple drifted")
    root = _absolute_lexical(run_dir, label="production output root")
    file_names = {
        "record": "overhead_a0_record.json",
        "acquisition_manifest": "overhead_a0_acquisition.json",
        "raw_frame": "overhead_a0_frame.jpg",
        "frame_bundle_artifact": "overhead_a0_input_frames.json",
        "result_json": "overhead_a0_result.json",
        "capture_pipeline_artifact": contract.CAPTURE_PIPELINE_CONTRACT.name,
        "inference_pipeline_artifact": contract.INFERENCE_PIPELINE_CONTRACTS[RESULT_SCHEMA].name,
        "camera_service_identity_artifact": "camera_service_identity.json",
        "replay_ledger_dir": "replay_ledger",
    }
    configured_paths = {
        field: _absolute_lexical(overhead[field], label=f"overhead.{field}") for field in file_names
    }
    template_roots = {path.parent for path in configured_paths.values()}
    if len(template_roots) != 1:
        raise CameraOnlyError("live config A0 paths do not share one run-scope template root")
    template_root = next(iter(template_roots))
    if template_root.name != A0_RUN_SCOPE_SENTINEL:
        raise CameraOnlyError("live config A0 paths lack the frozen run-scope sentinel")
    if any(configured_paths[field].name != name for field, name in file_names.items()):
        raise CameraOnlyError("live config A0 artifact names differ from the frozen contract")
    if (
        root.parent != template_root.parent
        or root.name == A0_RUN_SCOPE_SENTINEL
        or A0_RUN_SCOPE_RE.fullmatch(root.name) is None
    ):
        raise CameraOnlyError("run_dir must be one safe direct child of the configured A0 run root")
    expected_paths = {field: root / name for field, name in file_names.items()}
    return _OutputContract(
        config_file=config_path,
        config_sha256=claimed,
        release_id=_require_text(config["release_id"], label="live config release_id"),
        root=root,
        record=expected_paths["record"],
        manifest=expected_paths["acquisition_manifest"],
        raw_frame=expected_paths["raw_frame"],
        frame_bundle=expected_paths["frame_bundle_artifact"],
        result_json=expected_paths["result_json"],
        capture_artifact=expected_paths["capture_pipeline_artifact"],
        inference_artifact=expected_paths["inference_pipeline_artifact"],
        replay_dir=expected_paths["replay_ledger_dir"],
        camera_service_identity=expected_paths["camera_service_identity_artifact"],
    )


_REMOTE_FETCH_PROGRAM = r"""
import base64,hashlib,json,os,pathlib,pwd,socket,subprocess,sys,time,urllib.request
UNIT='xrd-overhead-camera.service'
SCRIPT=pathlib.Path('/home/rdk/dual_arm/overhead_camera_service.py')
UNIT_PATH=pathlib.Path('/etc/systemd/system/xrd-overhead-camera.service')
VIDEO='/dev/video0'
def props():
    raw=subprocess.check_output(['systemctl','show',UNIT,'--no-pager','--property=ActiveState','--property=SubState','--property=MainPID','--property=FragmentPath','--property=User'],text=True,timeout=5)
    out={}
    for line in raw.splitlines():
        key,sep,value=line.partition('=')
        if not sep or key in out: raise RuntimeError('invalid systemd property output')
        out[key]=value
    if set(out)!={'ActiveState','SubState','MainPID','FragmentPath','User'}: raise RuntimeError('incomplete systemd properties')
    return out
def usb_id():
    node=pathlib.Path('/sys/class/video4linux/video0/device').resolve(strict=True)
    for candidate in (node,*node.parents):
        vendor=candidate/'idVendor'; product=candidate/'idProduct'
        if vendor.is_file() and product.is_file():
            return vendor.read_text(encoding='ascii').strip().lower()+':'+product.read_text(encoding='ascii').strip().lower()
    raise RuntimeError('USB identity unavailable')
def listener_inode(pid):
    expected='00000000:22BC'
    inodes=[]
    path=pathlib.Path('/proc/net/tcp')
    if not path.is_file(): raise RuntimeError('IPv4 TCP table unavailable')
    for line in path.read_text(encoding='ascii').splitlines()[1:]:
        fields=line.split()
        if len(fields)>9 and fields[1].upper()==expected and fields[3]=='0A':
            inodes.append(int(fields[9]))
    if len(set(inodes))!=1: raise RuntimeError('camera listener is missing or ambiguous')
    inode=inodes[0]; marker=f'socket:[{inode}]'; owned=False
    for fd in pathlib.Path(f'/proc/{pid}/fd').iterdir():
        try:
            if os.readlink(fd)==marker: owned=True; break
        except OSError:
            pass
    if not owned: raise RuntimeError('camera listener is not owned by MainPID')
    return inode
def video_owner_pids():
    owners=set(); unreadable=[]
    for process in pathlib.Path('/proc').iterdir():
        if not process.name.isdigit(): continue
        fd_root=process/'fd'
        try:
            descriptors=list(fd_root.iterdir())
        except FileNotFoundError:
            continue
        except (PermissionError,OSError):
            unreadable.append(process.name); continue
        for fd in descriptors:
            try:
                if os.path.realpath(fd)==VIDEO:
                    owners.add(int(process.name)); break
            except FileNotFoundError:
                break
            except OSError:
                continue
    if unreadable: raise RuntimeError('global /proc fd ownership scan is incomplete')
    return owners
def service_snapshot():
    p=props(); pid=int(p['MainPID'])
    if pid<=1: raise RuntimeError('invalid MainPID')
    cmd=[os.fsdecode(part) for part in pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if part]
    script=SCRIPT.read_bytes(); unit=UNIT_PATH.read_bytes()
    owners=video_owner_pids()
    if owners!={pid}: raise RuntimeError('/dev/video0 owner set is not exactly MainPID')
    uid=None
    for line in pathlib.Path(f'/proc/{pid}/status').read_text(encoding='ascii').splitlines():
        if line.startswith('Uid:'):
            uid=int(line.split()[1]); break
    if uid is None: raise RuntimeError('process UID unavailable')
    listener=listener_inode(pid)
    value={
      'pid':pid,'cmdline':cmd,'script_path':str(SCRIPT),
      'script_sha256':hashlib.sha256(script).hexdigest(),'script_size_bytes':len(script),
      'unit_name':UNIT,'unit_active_state':p['ActiveState'],'unit_sub_state':p['SubState'],
      'unit_main_pid':pid,'unit_fragment_path':p['FragmentPath'],
      'unit_sha256':hashlib.sha256(unit).hexdigest(),'unit_size_bytes':len(unit),
      'service_user':p['User'],'video_device':VIDEO,'video_owner_pid':pid,
      'video_owner_uid':uid,'video_owner_user':pwd.getpwuid(uid).pw_name,'usb_id':usb_id(),
      'listener_ip':'127.0.0.1','listener_port':8892,'listener_inode':listener,'listener_owner_pid':pid,
    }
    return value,script
kind=sys.argv[1]
if kind not in {'health','snapshot'}: raise RuntimeError('unsupported fetch kind')
before,script_before=service_snapshot()
url='http://127.0.0.1:8892/health' if kind=='health' else 'http://127.0.0.1:8892/snapshot.jpg'
with urllib.request.urlopen(url,timeout=5) as response:
    body=response.read(67108865)
    if len(body)>67108864: raise RuntimeError('remote payload exceeds bound')
after,script_after=service_snapshot()
if before!=after or script_before!=script_after: raise RuntimeError('camera service changed during fetch')
digest=hashlib.sha256(body).hexdigest()
payload={
 'schema_version':'xrd-rb-voe-arm02-camera-envelope-v2',
 'hostname':socket.gethostname(),
 'machine_id_sha256':hashlib.sha256(pathlib.Path('/etc/machine-id').read_bytes().strip()).hexdigest(),
 'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text(encoding='ascii').strip(),
 'fetched_at_ms':time.time_ns()//1000000,
 'frame_id':'FRAME-SHA256-'+digest,
 'payload_sha256':digest,
 'payload_b64':base64.b64encode(body).decode('ascii'),
 'capture_script_b64':base64.b64encode(script_after).decode('ascii'),
 'capture_service':after,
}
sys.stdout.write(json.dumps(payload,separators=(',',':'),sort_keys=True)+'\n')
""".strip()


class _SubprocessOutputLimitExceeded(Exception):
    def __init__(
        self,
        *,
        stream_name: str,
        limit_bytes: int,
        stdout: bytes,
        stderr: bytes,
    ) -> None:
        super().__init__(f"subprocess {stream_name} exceeded {limit_bytes} bytes")
        self.stream_name = stream_name
        self.limit_bytes = limit_bytes
        self.stdout = stdout
        self.stderr = stderr


def _run_bounded_subprocess(
    command: Sequence[str],
    *,
    timeout_s: float,
    stdout_limit: int,
    stderr_limit: int,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Consume both output pipes concurrently without retaining past either limit."""

    if not command or any(not isinstance(token, str) or not token for token in command):
        raise ValueError("bounded subprocess command must contain non-empty strings")
    for name, limit in (("stdout", stdout_limit), ("stderr", stderr_limit)):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError(f"{name} limit must be a positive integer")

    process = subprocess.Popen(
        list(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=None if env is None else dict(env),
    )
    if process.stdout is None or process.stderr is None:  # pragma: no cover - Popen invariant
        process.kill()
        process.wait()
        raise RuntimeError("bounded subprocess pipes were not created")

    stdout_buffer = bytearray()
    stderr_buffer = bytearray()
    overflow: list[tuple[str, int]] = []
    reader_errors: list[BaseException] = []
    state_lock = threading.Lock()
    stopping = threading.Event()

    def kill_now() -> None:
        try:
            process.kill()
        except OSError:
            pass

    def read_stream(
        pipe: Any,
        buffer: bytearray,
        *,
        stream_name: str,
        limit: int,
    ) -> None:
        try:
            while True:
                read_size = min(_SUBPROCESS_READ_CHUNK_BYTES, limit - len(buffer) + 1)
                chunk = os.read(pipe.fileno(), read_size)
                if not chunk:
                    return
                remaining = limit - len(buffer)
                if len(chunk) > remaining:
                    if remaining:
                        buffer.extend(chunk[:remaining])
                    with state_lock:
                        if not overflow:
                            overflow.append((stream_name, limit))
                    stopping.set()
                    kill_now()
                    return
                buffer.extend(chunk)
        except OSError as exc:
            if not stopping.is_set():
                with state_lock:
                    if not reader_errors:
                        reader_errors.append(exc)
                stopping.set()
                kill_now()

    readers = (
        threading.Thread(
            target=read_stream,
            args=(process.stdout, stdout_buffer),
            kwargs={"stream_name": "stdout", "limit": stdout_limit},
            daemon=True,
        ),
        threading.Thread(
            target=read_stream,
            args=(process.stderr, stderr_buffer),
            kwargs={"stream_name": "stderr", "limit": stderr_limit},
            daemon=True,
        ),
    )
    started = time.monotonic()
    for reader in readers:
        reader.start()

    timed_out = False
    try:
        process.wait(timeout=timeout_s)
        deadline = started + timeout_s
        for reader in readers:
            reader.join(max(0.0, deadline - time.monotonic()))
        timed_out = any(reader.is_alive() for reader in readers)
    except subprocess.TimeoutExpired:
        timed_out = True

    if timed_out:
        stopping.set()
        kill_now()
        process.wait()
        process.stdout.close()
        process.stderr.close()
        for reader in readers:
            reader.join(1.0)
        raise subprocess.TimeoutExpired(
            list(command),
            timeout_s,
            output=bytes(stdout_buffer),
            stderr=bytes(stderr_buffer),
        )

    process.stdout.close()
    process.stderr.close()
    if reader_errors:
        raise CameraOnlyError(f"failed to read subprocess output: {reader_errors[0]}")
    if overflow:
        stream_name, limit = overflow[0]
        raise _SubprocessOutputLimitExceeded(
            stream_name=stream_name,
            limit_bytes=limit,
            stdout=bytes(stdout_buffer),
            stderr=bytes(stderr_buffer),
        )
    return subprocess.CompletedProcess(
        args=list(command),
        returncode=process.returncode,
        stdout=bytes(stdout_buffer),
        stderr=bytes(stderr_buffer),
    )


class SshCameraTransport:
    """The sole production transport: fixed-host, read-only SSH plus HTTP GET."""

    def __init__(
        self,
        *,
        known_hosts: Path | str,
        timeout_s: float = PRODUCTION_SSH_TIMEOUT_S,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self.known_hosts = _validate_known_hosts(known_hosts)
        self.timeout_s = _require_finite(
            timeout_s,
            label="SSH timeout_s",
            minimum=0.1,
            maximum=MAX_SSH_TIMEOUT_S,
        )
        self.clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)

    def _fetch(self, kind: str) -> RemoteObservation:
        if kind not in {"health", "snapshot"}:
            raise CameraOnlyError("unsupported camera transport operation")
        remote_command = (
            f"{AI_PYTHON_EXECUTABLE} -I -c {shlex.quote(_REMOTE_FETCH_PROGRAM)} {shlex.quote(kind)}"
        )
        command = [
            SSH_EXECUTABLE,
            "-F",
            "/dev/null",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "PasswordAuthentication=no",
            "-o",
            "KbdInteractiveAuthentication=no",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={self.known_hosts}",
            "-o",
            "GlobalKnownHostsFile=/dev/null",
            "-o",
            f"HostKeyAlias={ARM02_HOST}",
            "-o",
            "ProxyCommand=none",
            "-o",
            "ProxyJump=none",
            "-o",
            "ClearAllForwardings=yes",
            "-o",
            "PermitLocalCommand=no",
            "-o",
            "RequestTTY=no",
            f"{ARM02_USER}@{ARM02_HOST}",
            remote_command,
        ]
        started_at_ms = self.clock_ms()
        try:
            completed = _run_bounded_subprocess(
                command,
                timeout_s=self.timeout_s,
                stdout_limit=MAX_REMOTE_RESPONSE_BYTES,
                stderr_limit=MAX_SUBPROCESS_STDERR_BYTES,
            )
        except _SubprocessOutputLimitExceeded as exc:
            if exc.stream_name == "stdout":
                raise CameraOnlyError("arm02 SSH response exceeds the allowed bound") from exc
            raise CameraOnlyError("arm02 SSH stderr exceeds the allowed bound") from exc
        received_at_ms = self.clock_ms()
        if completed.returncode != 0:
            error = completed.stderr.decode("utf-8", errors="replace")[-800:]
            raise CameraOnlyError(f"arm02 read-only SSH fetch failed: {error}")
        envelope = _load_json_bytes(completed.stdout, label="arm02 SSH response")
        expected_keys = {
            "schema_version",
            "hostname",
            "machine_id_sha256",
            "boot_id",
            "fetched_at_ms",
            "frame_id",
            "payload_sha256",
            "payload_b64",
            "capture_script_b64",
            "capture_service",
        }
        if set(envelope) != expected_keys or envelope["schema_version"] != REMOTE_ENVELOPE_SCHEMA:
            raise CameraOnlyError("arm02 SSH response schema or keys are not exact")
        fetched_at_ms = _require_int(
            envelope["fetched_at_ms"],
            label="arm02 fetched_at_ms",
            minimum=contract.MIN_TIMESTAMP_MS,
        )
        if (
            fetched_at_ms < started_at_ms - MAX_REMOTE_CLOCK_SKEW_MS
            or fetched_at_ms > received_at_ms + MAX_REMOTE_CLOCK_SKEW_MS
        ):
            raise CameraOnlyError("arm02 fetched timestamp is outside the bounded clock bracket")
        machine_digest = _require_sha256(envelope["machine_id_sha256"], label="arm02 machine-id")
        identity = HostIdentity(
            hostname=_require_text(envelope["hostname"], label="arm02 hostname"),
            device_id=f"machine-sha256:{machine_digest}",
            boot_id=_require_text(envelope["boot_id"], label="arm02 boot_id"),
        )
        _validate_identity(identity, expected_hostname=ARM02_HOSTNAME, label="arm02")
        try:
            payload = base64.b64decode(envelope["payload_b64"], validate=True)
            script_bytes = base64.b64decode(envelope["capture_script_b64"], validate=True)
        except (TypeError, ValueError) as exc:
            raise CameraOnlyError("arm02 SSH response contains invalid base64") from exc
        payload_sha256 = _require_sha256(envelope["payload_sha256"], label="payload_sha256")
        if _sha256_bytes(payload) != payload_sha256:
            raise CameraOnlyError("arm02 payload hash mismatch")
        frame_id = _require_text(envelope["frame_id"], label="frame_id")
        if frame_id != f"FRAME-SHA256-{payload_sha256}":
            raise CameraOnlyError("arm02 frame_id does not bind the payload")
        service = _service_from_value(envelope["capture_service"], script_bytes=script_bytes)
        observation = RemoteObservation(
            identity=identity,
            fetched_at_ms=fetched_at_ms,
            received_at_ms=received_at_ms,
            frame_id=frame_id,
            payload=payload,
            service=service,
        )
        _validate_observation(observation)
        return observation

    def fetch_health(self) -> RemoteObservation:
        return self._fetch("health")

    def fetch_snapshot(self) -> RemoteObservation:
        return self._fetch("snapshot")


class LocalIdentityProvider:
    """The sole production local identity provider; all paths are fixed."""

    def __call__(self) -> HostIdentity:
        identity = HostIdentity(
            hostname=socket.gethostname().strip(),
            device_id=_device_id(Path("/etc/machine-id").read_bytes()),
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip(),
        )
        _validate_identity(identity, expected_hostname=AI_HOSTNAME, label="AI X5")
        return identity


def _inference_pipeline_path() -> Path:
    frozen = contract.INFERENCE_PIPELINE_CONTRACTS[RESULT_SCHEMA]
    path = Path(__file__).resolve().parent / frozen.name
    _, raw = _regular_file(path, label="inference pipeline", max_bytes=contract.MAX_PIPELINE_ARTIFACT_BYTES)
    if path.name != frozen.name or len(raw) != frozen.size_bytes or _sha256_bytes(raw) != frozen.sha256:
        raise CameraOnlyError("inference pipeline does not match the frozen contract")
    return path


class FixedInferenceExecutor:
    """The sole production inference executor; fixed Python and fixed pipeline."""

    def __init__(self, *, timeout_s: float = PRODUCTION_INFERENCE_TIMEOUT_S) -> None:
        self.timeout_s = _require_finite(
            timeout_s,
            label="inference timeout_s",
            minimum=1.0,
            maximum=180.0,
        )

    def run(
        self,
        *,
        empty_dir: Path,
        occupied_dir: Path | None,
        output_dir: Path,
    ) -> Path:
        output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        attempt_dir = output_dir / "fixed-attempt"
        result = attempt_dir / "result.json"
        if result.exists() or result.is_symlink():
            raise CameraOnlyError("pre-existing fixed inference result is forbidden")
        if attempt_dir.exists() or attempt_dir.is_symlink():
            raise CameraOnlyError("incomplete fixed inference attempt requires operator review")
        attempt_dir.mkdir(mode=0o700)
        pipeline = _inference_pipeline_path()
        frozen = contract.INFERENCE_PIPELINE_CONTRACTS[RESULT_SCHEMA]
        _, pipeline_bytes = _regular_file(
            pipeline,
            label="inference pipeline before execution",
            max_bytes=contract.MAX_PIPELINE_ARTIFACT_BYTES,
        )
        if len(pipeline_bytes) != frozen.size_bytes or _sha256_bytes(pipeline_bytes) != frozen.sha256:
            raise CameraOnlyError("inference pipeline changed before execution")
        executed_pipeline = attempt_dir / frozen.name
        _write_bytes_once(executed_pipeline, pipeline_bytes)
        executed_pipeline.chmod(0o400)
        command = [
            AI_PYTHON_EXECUTABLE,
            "-I",
            str(executed_pipeline),
            "--empty-dir",
            str(empty_dir),
            "--out-dir",
            str(attempt_dir),
        ]
        if occupied_dir is not None:
            command.extend(("--bag-dir", str(occupied_dir)))
        allowed_environment = {
            name: value
            for name, value in os.environ.items()
            if name in {"HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "PATH", "TZ"}
        }
        allowed_environment["PYTHONNOUSERSITE"] = "1"
        try:
            completed = _run_bounded_subprocess(
                command,
                timeout_s=self.timeout_s,
                stdout_limit=MAX_INFERENCE_STDOUT_BYTES,
                stderr_limit=MAX_SUBPROCESS_STDERR_BYTES,
                env=allowed_environment,
            )
        except _SubprocessOutputLimitExceeded as exc:
            raise CameraOnlyError(
                f"pinned AI-X5 inference {exc.stream_name} exceeds the allowed bound"
            ) from exc
        allowed_codes = {0} if occupied_dir is None else {0, 3}
        if completed.returncode not in allowed_codes or not result.is_file():
            error = completed.stderr.decode("utf-8", errors="replace")[-1000:]
            raise CameraOnlyError(f"pinned AI-X5 inference failed: {error}")
        return result


def plan() -> dict[str, Any]:
    return {
        "schema_version": PLAN_SCHEMA,
        "mode": "PlanOnly",
        "remote_contacted": False,
        "network_touched": False,
        "hardware_touched": False,
        "transport_commands_issued": 0,
        "actuator_commands_issued": 0,
        "mutating_device_commands_issued": 0,
        "fixed_target": f"{ARM02_USER}@{ARM02_HOST}",
        "camera_endpoints": [f"{CAMERA_BASE_URL}/health", f"{CAMERA_BASE_URL}/snapshot.jpg"],
        "task_contract": {
            "task_kind": TASK_KIND,
            "result_schema": RESULT_SCHEMA,
            "success_state": SUCCESS_STATE,
        },
        "frame_contract": {"empty": FRAME_COUNT, "occupied": FRAME_COUNT},
        "output_scope_contract": {
            "template_component": A0_RUN_SCOPE_SENTINEL,
            "run_dir_relation": "SAFE_DIRECT_CHILD",
            "terminal_overwrite_allowed": False,
            "single_challenge_binding": True,
        },
        "challenge_ttl_ms": LIVE_CHALLENGE_TTL_MS,
        "challenge_min_budget_ms": LIVE_CHALLENGE_MIN_BUDGET_MS,
        "terminal_artifacts": [
            "overhead_a0_input_frames.json",
            "overhead_a0_acquisition.json",
            "overhead_a0_record.json",
            "replay_ledger/<replay_identity_sha256>.json",
        ],
        "blockers": list(PLAN_BLOCKERS),
    }


def _validate_observation(observation: RemoteObservation) -> None:
    _validate_identity(observation.identity, expected_hostname=ARM02_HOSTNAME, label="arm02")
    _validate_service_identity(observation.service)
    fetched = _require_int(
        observation.fetched_at_ms,
        label="observation.fetched_at_ms",
        minimum=contract.MIN_TIMESTAMP_MS,
    )
    received = _require_int(
        observation.received_at_ms,
        label="observation.received_at_ms",
        minimum=contract.MIN_TIMESTAMP_MS,
    )
    if fetched > received + MAX_REMOTE_CLOCK_SKEW_MS:
        raise CameraOnlyError("remote fetched timestamp exceeds the receive bound")
    digest = _sha256_bytes(observation.payload)
    if observation.frame_id != f"FRAME-SHA256-{digest}":
        raise CameraOnlyError("remote frame_id does not bind payload bytes")


def _health_payload(observation: RemoteObservation, *, max_frame_age_s: float) -> dict[str, Any]:
    _validate_observation(observation)
    health = _load_json_bytes(observation.payload, label="camera health")
    required = {
        "status",
        "service",
        "device",
        "camera_opened",
        "frame_age_s",
        "fps",
        "last_error",
        "robot_control_surface",
        "motion_authority",
    }
    if set(health) != required:
        raise CameraOnlyError("camera health keys are not exact")
    if (
        health["status"] != "ok"
        or health["service"] != "xrd-overhead-camera"
        or health["device"] != contract.FROZEN_CAMERA_SOURCE
        or health["camera_opened"] is not True
        or health["robot_control_surface"] is not False
        or health["motion_authority"] is not False
        or health["last_error"] not in {"", None}
    ):
        raise CameraOnlyError("camera health is not the frozen camera-only ready state")
    _require_finite(
        health["frame_age_s"],
        label="camera frame_age_s",
        minimum=0.0,
        maximum=max_frame_age_s,
    )
    _require_finite(health["fps"], label="camera fps", minimum=0.1, maximum=120.0)
    return health


def _persist_service_artifacts(run_dir: Path, service: CameraServiceIdentity) -> None:
    runtime_dir = run_dir / "runtime"
    runtime_dir.mkdir(mode=0o700, exist_ok=True)
    _ensure_bytes(runtime_dir / contract.CAPTURE_PIPELINE_CONTRACT.name, service.script_bytes)
    _ensure_bytes(
        runtime_dir / "camera_service_runtime.json",
        canonical_bytes(service.as_dict()) + b"\n",
    )


def _camera_service_identity_artifact_value(
    state: Mapping[str, Any],
    *,
    observed_at_ms: int,
) -> dict[str, Any]:
    capture = _identity_from_state(state["capture_identity"], label="stored arm02", hostname=ARM02_HOSTNAME)
    service = _validate_service_state_value(state["capture_service_identity"])
    session_id = _require_text(state["capture_session_id"], label="capture_session_id")
    observed = _require_int(
        observed_at_ms,
        label="camera service observed_at_ms",
        minimum=contract.MIN_TIMESTAMP_MS,
    )
    runtime = {key: service[key] for key in contract.CAMERA_SERVICE_RUNTIME_KEYS}
    runtime_sha256 = canonical_sha256(runtime)
    if runtime_sha256 != service["service_identity_sha256"]:
        raise CameraOnlyError("camera service runtime digest differs from the run binding")
    artifact: dict[str, Any] = {
        "schema_version": contract.CAMERA_SERVICE_IDENTITY_SCHEMA_VERSION,
        "hostname": capture.hostname,
        "device_id": capture.device_id,
        "boot_id": capture.boot_id,
        "session_id": session_id,
        **runtime,
        "observed_at_ms": observed,
        "service_identity_sha256": runtime_sha256,
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return artifact


def _materialize_camera_service_identity(
    path: Path,
    *,
    state: Mapping[str, Any],
    observed_at_ms: int,
    expected: contract.ExpectedAcquisition,
) -> dict[str, Any]:
    artifact = _camera_service_identity_artifact_value(state, observed_at_ms=observed_at_ms)
    raw = canonical_bytes(artifact) + b"\n"
    _ensure_bytes(path, raw)
    try:
        contract._validate_camera_service_identity_value(artifact, expected=expected)
    except contract.ActualRecordError as exc:
        raise CameraOnlyError(str(exc)) from exc
    return artifact


def _bind_or_check_identities(
    state: dict[str, Any],
    *,
    capture: HostIdentity,
    inference: HostIdentity,
    service: CameraServiceIdentity,
    run_dir: Path,
) -> None:
    _validate_identity(capture, expected_hostname=ARM02_HOSTNAME, label="arm02")
    _validate_identity(inference, expected_hostname=AI_HOSTNAME, label="AI X5")
    _validate_service_identity(service)
    _persist_service_artifacts(run_dir, service)
    service_value = service.as_dict()
    if state["capture_identity"] is None:
        state["capture_identity"] = capture.as_dict()
        state["inference_identity"] = inference.as_dict()
        state["capture_service_identity"] = service_value
        state["capture_session_id"] = f"A0-CAP-{secrets.token_hex(16)}"
        state["inference_session_id"] = f"A0-INF-{secrets.token_hex(16)}"
        _write_or_validate_binding(run_dir, state)
        return
    stored_capture = _identity_from_state(
        state["capture_identity"], label="stored arm02", hostname=ARM02_HOSTNAME
    )
    stored_inference = _identity_from_state(
        state["inference_identity"], label="stored AI X5", hostname=AI_HOSTNAME
    )
    stored_service = _validate_service_state_value(state["capture_service_identity"])
    if capture != stored_capture:
        raise CameraOnlyError("arm02 host/device/boot differs from the run binding")
    if inference != stored_inference:
        raise CameraOnlyError("AI X5 host/device/boot differs from the run binding")
    if service_value != stored_service:
        raise CameraOnlyError("arm02 running camera service identity drifted")
    _validate_binding(run_dir, state)


def _transport_operation(state: dict[str, Any], *, production: bool) -> None:
    state["read_only_transport_operations"] += 1
    if production:
        state["remote_contacted"] = True


def _all_frame_hashes(state: Mapping[str, Any]) -> set[str]:
    hashes: set[str] = set()
    for phase in ("empty_frames", "occupied_frames"):
        rows = state[phase]
        if not isinstance(rows, list):
            raise CameraOnlyError(f"state.{phase} must be a list")
        for row in rows:
            if not isinstance(row, Mapping):
                raise CameraOnlyError(f"state.{phase} row is malformed")
            digest = _require_sha256(row.get("sha256"), label=f"state.{phase}.sha256")
            if digest in hashes:
                raise CameraOnlyError("state contains duplicate frame content")
            hashes.add(digest)
    return hashes


def _capture_frames(
    *,
    phase: str,
    run_dir: Path,
    state_path: Path,
    state: dict[str, Any],
    transport: Any,
    identity_provider: Any,
    clock_ms: Callable[[], int],
    sleeper: Callable[[float], None],
    production: bool,
) -> None:
    key = f"{phase}_frames"
    rows = state[key]
    parameters = state["parameters"]
    max_frame_age_s = float(parameters["max_frame_age_s"])
    bracket_ms = int(parameters["snapshot_bracket_ms"])
    interval_s = float(parameters["interval_s"])
    frame_dir = run_dir / "frames" / phase
    frame_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    known_hashes = _all_frame_hashes(state)
    if len(rows) == FRAME_COUNT:
        _transport_operation(state, production=production)
        _save_state(state_path, state, now_ms=clock_ms())
        health = transport.fetch_health()
        _health_payload(health, max_frame_age_s=max_frame_age_s)
        _bind_or_check_identities(
            state,
            capture=health.identity,
            inference=identity_provider(),
            service=health.service,
            run_dir=run_dir,
        )
        _save_state(state_path, state, now_ms=clock_ms())
        return
    if len(rows) > FRAME_COUNT:
        raise CameraOnlyError(f"{phase} has more than the frozen five frames")
    while len(rows) < FRAME_COUNT:
        _transport_operation(state, production=production)
        _save_state(state_path, state, now_ms=clock_ms())
        before = transport.fetch_health()
        _health_payload(before, max_frame_age_s=max_frame_age_s)
        _transport_operation(state, production=production)
        _save_state(state_path, state, now_ms=clock_ms())
        snapshot = transport.fetch_snapshot()
        _validate_observation(snapshot)
        _transport_operation(state, production=production)
        _save_state(state_path, state, now_ms=clock_ms())
        after = transport.fetch_health()
        _health_payload(after, max_frame_age_s=max_frame_age_s)
        if before.identity != snapshot.identity or snapshot.identity != after.identity:
            raise CameraOnlyError("arm02 identity changed across the snapshot health bracket")
        if (
            before.service.as_dict() != snapshot.service.as_dict()
            or snapshot.service.as_dict() != after.service.as_dict()
        ):
            raise CameraOnlyError("camera service changed across the snapshot health bracket")
        if not (
            before.fetched_at_ms <= snapshot.fetched_at_ms <= after.fetched_at_ms
            and after.fetched_at_ms - before.fetched_at_ms <= bracket_ms
        ):
            raise CameraOnlyError("snapshot remote timestamp is outside the frozen health bracket")
        _bind_or_check_identities(
            state,
            capture=snapshot.identity,
            inference=identity_provider(),
            service=snapshot.service,
            run_dir=run_dir,
        )
        _save_state(state_path, state, now_ms=clock_ms())
        if len(snapshot.payload) <= 0 or len(snapshot.payload) > contract.MAX_FRAME_BYTES:
            raise CameraOnlyError("snapshot size is outside the allowed bound")
        try:
            width, height = contract.inspect_jpeg(snapshot.payload)
        except contract.ActualRecordError as exc:
            raise CameraOnlyError(str(exc)) from exc
        digest = _sha256_bytes(snapshot.payload)
        if digest in known_hashes:
            raise CameraOnlyError("duplicate frame content rejected")
        path = frame_dir / f"{digest}.jpg"
        _write_bytes_once(path, snapshot.payload)
        row = {
            "file_name": path.name,
            "sha256": digest,
            "size_bytes": len(snapshot.payload),
            "width": width,
            "height": height,
            "frame_id": snapshot.frame_id,
            "captured_at_ms": snapshot.fetched_at_ms,
            "health_before_at_ms": before.fetched_at_ms,
            "health_after_at_ms": after.fetched_at_ms,
            "service_identity_sha256": snapshot.service.as_dict()["service_identity_sha256"],
        }
        rows.append(row)
        known_hashes.add(digest)
        state["last_error"] = None
        _save_state(state_path, state, now_ms=clock_ms())
        if len(rows) < FRAME_COUNT:
            sleeper(interval_s)


def _frame_path(run_dir: Path, phase: str, row: Mapping[str, Any]) -> Path:
    if set(row) != FRAME_ROW_KEYS:
        raise CameraOnlyError(f"{phase} frame row keys are not exact")
    digest = _require_sha256(row["sha256"], label=f"{phase} frame sha256")
    expected_name = f"{digest}.jpg"
    if row["file_name"] != expected_name:
        raise CameraOnlyError(f"{phase} frame basename is not content-addressed")
    path = run_dir / "frames" / phase / expected_name
    _, raw = _regular_file(path, label=f"{phase} frame", max_bytes=contract.MAX_FRAME_BYTES)
    if _sha256_bytes(raw) != digest or len(raw) != row["size_bytes"]:
        raise CameraOnlyError(f"{phase} frame bytes differ from state")
    width, height = contract.inspect_jpeg(raw)
    if (row["width"], row["height"]) != (width, height):
        raise CameraOnlyError(f"{phase} frame dimensions differ from state")
    if row["frame_id"] != f"FRAME-SHA256-{digest}":
        raise CameraOnlyError(f"{phase} frame_id mismatch")
    before = _require_int(
        row["health_before_at_ms"], label=f"{phase}.health_before_at_ms", minimum=contract.MIN_TIMESTAMP_MS
    )
    captured = _require_int(
        row["captured_at_ms"], label=f"{phase}.captured_at_ms", minimum=contract.MIN_TIMESTAMP_MS
    )
    after = _require_int(
        row["health_after_at_ms"], label=f"{phase}.health_after_at_ms", minimum=contract.MIN_TIMESTAMP_MS
    )
    if not before <= captured <= after:
        raise CameraOnlyError(f"{phase} frame timestamp is outside its health bracket")
    _require_sha256(row["service_identity_sha256"], label=f"{phase}.service_identity_sha256")
    return path


def _validate_frame_rows(run_dir: Path, state: Mapping[str, Any], phase: str) -> list[Path]:
    rows = state[f"{phase}_frames"]
    if not isinstance(rows, list) or len(rows) > FRAME_COUNT:
        raise CameraOnlyError(f"state.{phase}_frames count is invalid")
    paths = [_frame_path(run_dir, phase, row) for row in rows]
    if len({path.name for path in paths}) != len(paths):
        raise CameraOnlyError(f"{phase} frame identities are not unique")
    service_value = state["capture_service_identity"]
    if paths:
        service = _validate_service_state_value(service_value)
        service_sha = service["service_identity_sha256"]
        if any(row["service_identity_sha256"] != service_sha for row in rows):
            raise CameraOnlyError(f"{phase} frame service identity drifted")
    bracket = int(state["parameters"]["snapshot_bracket_ms"])
    if any(row["health_after_at_ms"] - row["health_before_at_ms"] > bracket for row in rows):
        raise CameraOnlyError(f"{phase} frame bracket exceeds the frozen bound")
    return paths


def _result_inventory(payload: Mapping[str, Any], section: str) -> list[tuple[str, str]]:
    value = payload.get(section)
    if not isinstance(value, Mapping) or not isinstance(value.get("files"), list):
        raise CameraOnlyError(f"result.{section}.files is malformed")
    rows: list[tuple[str, str]] = []
    for index, row in enumerate(value["files"]):
        if not isinstance(row, Mapping):
            raise CameraOnlyError(f"result.{section}.files[{index}] is malformed")
        rows.append(
            (
                _require_text(row.get("name"), label=f"result.{section}.files[{index}].name"),
                _require_sha256(row.get("sha256"), label=f"result.{section}.files[{index}].sha256"),
            )
        )
    if len(rows) != len(set(rows)) or len({digest for _, digest in rows}) != len(rows):
        raise CameraOnlyError(f"result.{section} contains duplicate frame evidence")
    return rows


def _state_inventory(state: Mapping[str, Any], phase: str) -> list[tuple[str, str]]:
    rows = state[f"{phase}_frames"]
    return [(_authoritative_name(phase, index), str(row["sha256"])) for index, row in enumerate(rows)]


def _authoritative_name(phase: str, index: int) -> str:
    if phase == "empty":
        return f"empty_{index:02d}.jpg"
    if phase == "occupied":
        return "overhead_a0_frame.jpg" if index == 0 else f"zz_occupied_{index:02d}.jpg"
    raise CameraOnlyError("unsupported authoritative input phase")


def _authoritative_input_dir(run_dir: Path, phase: str) -> Path:
    return run_dir / ".a0_runtime" / "authoritative_inputs" / phase


def _materialize_authoritative_inputs(
    run_dir: Path,
    state: Mapping[str, Any],
    phase: str,
) -> list[Path]:
    rows = state[f"{phase}_frames"]
    if not isinstance(rows, list) or len(rows) != FRAME_COUNT:
        raise CameraOnlyError(f"authoritative {phase} inputs require exactly five captures")
    directory = _authoritative_input_dir(run_dir, phase)
    existed = directory.exists() or directory.is_symlink()
    if existed:
        _reject_link_components(directory)
        if not directory.is_dir():
            raise CameraOnlyError(f"authoritative {phase} input path is not a directory")
    else:
        directory.mkdir(mode=0o700, parents=True)
    expected_names = {_authoritative_name(phase, index) for index in range(FRAME_COUNT)}
    if existed:
        actual_names = {path.name for path in directory.iterdir()}
        if actual_names != expected_names:
            raise CameraOnlyError(f"authoritative {phase} input set is missing, extra, or incomplete")
    paths: list[Path] = []
    for index, row in enumerate(rows):
        source = _frame_path(run_dir, phase, row)
        destination = directory / _authoritative_name(phase, index)
        _ensure_bytes(destination, source.read_bytes())
        paths.append(destination)
    actual_names = {path.name for path in directory.iterdir()}
    if actual_names != expected_names or any(
        not path.is_file() or path.is_symlink() for path in directory.iterdir()
    ):
        raise CameraOnlyError(f"authoritative {phase} input directory contains unexpected entries")
    return paths


def _load_result(path: Path) -> tuple[dict[str, Any], bytes]:
    _, raw = _regular_file(path, label="inference result", max_bytes=contract.MAX_RESULT_JSON_BYTES)
    return _load_json_bytes(raw, label="inference result"), raw


def _validate_empty_result(path: Path, state: Mapping[str, Any]) -> str:
    payload, _ = _load_result(path)
    if payload.get("schema_version") != RESULT_SCHEMA:
        raise CameraOnlyError("empty result schema is not the frozen bag pipeline")
    if payload.get("decision") != "EMPTY_BASELINE_READY" or "occupied" in payload:
        raise CameraOnlyError("empty stage did not produce a baseline-only result")
    if _result_inventory(payload, "empty") != _state_inventory(state, "empty"):
        raise CameraOnlyError("empty result order does not bind every captured frame exactly")
    try:
        return contract.bag_empty_baseline_sha256(payload["empty"])
    except contract.ActualRecordError as exc:
        raise CameraOnlyError(str(exc)) from exc


def _validate_occupied_result(
    path: Path,
    state: Mapping[str, Any],
    *,
    selected_name: str,
    selected_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    payload, raw = _load_result(path)
    if payload.get("schema_version") != RESULT_SCHEMA:
        raise CameraOnlyError("occupied result schema is not the frozen bag pipeline")
    if payload.get("decision") not in {"BAG_PRESENT", "BAG_NOT_DETECTED"}:
        raise CameraOnlyError("occupied result is not a terminal bag decision")
    if _result_inventory(payload, "empty") != _state_inventory(state, "empty"):
        raise CameraOnlyError("occupied result does not bind the frozen empty inputs in order")
    if _result_inventory(payload, "occupied") != _state_inventory(state, "occupied"):
        raise CameraOnlyError("occupied result does not bind every occupied input in order")
    try:
        semantics = contract.inspect_result_json(
            raw,
            raw_frame_sha256=selected_sha256,
            raw_frame_name=selected_name,
        )
    except contract.ActualRecordError as exc:
        raise CameraOnlyError(str(exc)) from exc
    if semantics.schema != RESULT_SCHEMA:
        raise CameraOnlyError("result unexpectedly selected station semantics")
    return payload, raw


def _challenge_content_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("challenge_content_sha256", None)
    return canonical_sha256(unsigned)


def _sidecar(challenge_path: Path, suffix: str) -> Path:
    return challenge_path.with_name(f"{challenge_path.name}.{suffix}.json")


def _read_optional_json(path: Path, *, label: str, max_bytes: int = 256 * 1024) -> dict[str, Any] | None:
    if not path.exists() and not path.is_symlink():
        return None
    _, raw = _regular_file(path, label=label, max_bytes=max_bytes)
    return _load_json_bytes(raw, label=label)


def _validate_consumed(
    challenge_path: Path,
    *,
    challenge: Mapping[str, Any],
    artifact_sha256: str,
) -> dict[str, Any] | None:
    consumed = _read_optional_json(_sidecar(challenge_path, "consumed"), label="challenge consumption")
    if consumed is None:
        return None
    if set(consumed) != {
        "schema_version",
        "challenge_sha256",
        "challenge_artifact_sha256",
        "config_sha256",
        "bootstrap_manifest_sha256",
        "bootstrap_manifest_file_sha256",
        "consumed_at_ms",
    }:
        raise CameraOnlyError("challenge consumption receipt keys are not exact")
    expected = {
        "schema_version": LIVE_CONSUMPTION_SCHEMA,
        "challenge_sha256": challenge["challenge_sha256"],
        "challenge_artifact_sha256": artifact_sha256,
        "config_sha256": challenge["config_sha256"],
        "bootstrap_manifest_sha256": challenge["bootstrap_manifest_sha256"],
        "bootstrap_manifest_file_sha256": challenge["bootstrap_manifest_file_sha256"],
        "consumed_at_ms": consumed["consumed_at_ms"],
    }
    _require_int(consumed["consumed_at_ms"], label="consumed_at_ms", minimum=contract.MIN_TIMESTAMP_MS)
    if consumed != expected:
        raise CameraOnlyError("challenge consumption receipt does not bind this challenge")
    return consumed


def _validate_challenge(
    path: Path | str,
    *,
    known_hosts: Path,
    state: Mapping[str, Any],
    now_ms: int,
    allow_terminal_recovery: bool,
) -> _ValidatedChallenge:
    challenge_path, raw = _regular_file(path, label="challenge", max_bytes=256 * 1024)
    challenge = _load_json_bytes(raw, label="challenge")
    if set(challenge) != CHALLENGE_KEYS:
        raise CameraOnlyError("challenge keys are not exact")
    if challenge["schema_version"] != LIVE_CHALLENGE_SCHEMA:
        raise CameraOnlyError("challenge schema is unsupported")
    if (challenge["task_kind"], challenge["result_schema"], challenge["success_state"]) != (
        TASK_KIND,
        RESULT_SCHEMA,
        SUCCESS_STATE,
    ):
        raise CameraOnlyError("challenge task/result tuple is not the frozen bag-drop contract")
    claimed = _require_sha256(challenge["challenge_sha256"], label="challenge_sha256")
    content = _require_sha256(challenge["challenge_content_sha256"], label="challenge_content_sha256")
    _require_sha256(challenge["bootstrap_manifest_sha256"], label="bootstrap_manifest_sha256")
    _require_sha256(
        challenge["bootstrap_manifest_file_sha256"],
        label="bootstrap_manifest_file_sha256",
    )
    if content != _challenge_content_sha256(challenge):
        raise CameraOnlyError("challenge content digest mismatch")
    if (
        challenge["known_hosts_file"] != str(known_hosts)
        or challenge["known_hosts_sha256"] != FROZEN_KNOWN_HOSTS_SHA256
    ):
        raise CameraOnlyError("challenge known_hosts binding mismatch")
    baseline = _require_sha256(state["bag_empty_baseline_sha256"], label="state.bag_empty_baseline_sha256")
    if challenge["bag_empty_baseline_sha256"] != baseline:
        raise CameraOnlyError("challenge baseline differs from the completed empty stage")
    run_nonce = _require_text(challenge["run_nonce"], label="challenge.run_nonce")
    if challenge["run_nonce_sha256"] != hashlib.sha256(run_nonce.encode("utf-8")).hexdigest():
        raise CameraOnlyError("challenge run nonce digest mismatch")
    try:
        expected_challenge = contract.a0_challenge_sha256(
            acquisition_id=challenge["acquisition_id"],
            a0_run_id=challenge["a0_run_id"],
            r2_run_id=challenge["run_id"],
            r2_run_nonce=run_nonce,
            challenge_issued_at_ms=challenge["issued_at_ms"],
            challenge_expires_at_ms=challenge["expires_at_ms"],
            release_id=challenge["release_id"],
            config_sha256=challenge["config_sha256"],
            case_id=challenge["case_id"],
            sample_id=challenge["sample_id"],
            sample_lineage_sha256=challenge["sample_lineage_sha256"],
            parent_evidence_root_sha256=challenge["parent_evidence_root_sha256"],
            bag_empty_baseline_sha256=challenge["bag_empty_baseline_sha256"],
            task_kind=challenge["task_kind"],
            result_schema=challenge["result_schema"],
            success_state=challenge["success_state"],
        )
    except contract.ActualRecordError as exc:
        raise CameraOnlyError(str(exc)) from exc
    if claimed != expected_challenge:
        raise CameraOnlyError("challenge_sha256 does not bind the A0 challenge")
    issued = _require_int(challenge["issued_at_ms"], label="issued_at_ms", minimum=contract.MIN_TIMESTAMP_MS)
    expires = _require_int(
        challenge["expires_at_ms"], label="expires_at_ms", minimum=contract.MIN_TIMESTAMP_MS
    )
    if expires != issued + LIVE_CHALLENGE_TTL_MS or now_ms < issued:
        raise CameraOnlyError("challenge is future or has the wrong TTL")
    if now_ms >= expires and not allow_terminal_recovery:
        raise CameraOnlyError("challenge expired before production sealing")
    empty_completed = _require_int(
        state["empty_completed_at_ms"],
        label="empty_completed_at_ms",
        minimum=contract.MIN_TIMESTAMP_MS,
    )
    if issued < empty_completed or issued - empty_completed > MAX_BASELINE_AGE_MS:
        raise CameraOnlyError("challenge is not bound to a fresh completed empty baseline")
    if now_ms - empty_completed > MAX_BASELINE_AGE_MS and not allow_terminal_recovery:
        raise CameraOnlyError("empty baseline expired before occupied acquisition")
    if challenge["purpose"] != LIVE_CHALLENGE_PURPOSE:
        raise CameraOnlyError("challenge purpose is invalid")
    for field in ("remote_contacted", "network_touched", "execution_authority"):
        if challenge[field] is not False:
            raise CameraOnlyError(f"challenge.{field} must remain false at issuance")
    for field in (
        "transport_commands_issued",
        "read_only_device_observations",
        "actuator_commands_issued",
        "mutating_commands_issued",
        "read_only_transport_operations",
    ):
        if isinstance(challenge[field], bool) or challenge[field] != 0:
            raise CameraOnlyError(f"challenge.{field} must remain integer zero at issuance")
    profiles = challenge["profile_sha256"]
    if (
        not isinstance(profiles, Mapping)
        or set(profiles) != {"ai_x5", "assay_station", "dual_arm", "embodied_x5"}
        or profiles.get("dual_arm") != contract.DUAL_ARM_SEMANTIC_PROFILE_SHA256
        or any(SHA256_RE.fullmatch(str(value)) is None for value in profiles.values())
    ):
        raise CameraOnlyError("challenge semantic profiles are malformed")
    artifact_sha256 = _sha256_bytes(raw)
    consumed = _validate_consumed(
        challenge_path,
        challenge=challenge,
        artifact_sha256=artifact_sha256,
    )
    issuance = _read_optional_json(_sidecar(challenge_path, "issued"), label="challenge issuance")
    expected_issuance = {
        "schema_version": LIVE_ISSUANCE_SCHEMA,
        "challenge_sha256": claimed,
        "challenge_artifact_sha256": artifact_sha256,
        "config_sha256": challenge["config_sha256"],
        "bootstrap_manifest_sha256": challenge["bootstrap_manifest_sha256"],
        "bootstrap_manifest_file_sha256": challenge["bootstrap_manifest_file_sha256"],
        "issued_at_ms": issued,
    }
    if issuance is None:
        if consumed is None:
            raise CameraOnlyError("challenge issuance receipt is missing")
    elif issuance != expected_issuance:
        raise CameraOnlyError("challenge issuance receipt mismatch")
    return _ValidatedChallenge(
        path=challenge_path,
        value=challenge,
        artifact_sha256=artifact_sha256,
        consumed=consumed,
    )


def _expected_from_challenge(
    challenge: Mapping[str, Any], state: Mapping[str, Any]
) -> contract.ExpectedAcquisition:
    capture = _identity_from_state(state["capture_identity"], label="stored arm02", hostname=ARM02_HOSTNAME)
    inference = _identity_from_state(state["inference_identity"], label="stored AI X5", hostname=AI_HOSTNAME)
    expected = contract.ExpectedAcquisition(
        acquisition_id=challenge["acquisition_id"],
        a0_run_id=challenge["a0_run_id"],
        r2_run_id=challenge["run_id"],
        r2_run_nonce=challenge["run_nonce"],
        challenge_sha256=challenge["challenge_sha256"],
        challenge_issued_at_ms=challenge["issued_at_ms"],
        challenge_expires_at_ms=challenge["expires_at_ms"],
        release_id=challenge["release_id"],
        config_sha256=challenge["config_sha256"],
        case_id=challenge["case_id"],
        sample_id=challenge["sample_id"],
        sample_lineage_sha256=challenge["sample_lineage_sha256"],
        parent_evidence_root_sha256=challenge["parent_evidence_root_sha256"],
        bag_empty_baseline_sha256=challenge["bag_empty_baseline_sha256"],
        task_kind=challenge["task_kind"],
        result_schema=challenge["result_schema"],
        success_state=challenge["success_state"],
        capture_device_id=capture.device_id,
        capture_boot_id=capture.boot_id,
        capture_session_id=_require_text(state["capture_session_id"], label="capture_session_id"),
        inference_device_id=inference.device_id,
        inference_boot_id=inference.boot_id,
        inference_session_id=_require_text(state["inference_session_id"], label="inference_session_id"),
    )
    try:
        expected.validate()
    except contract.ActualRecordError as exc:
        raise CameraOnlyError(str(exc)) from exc
    return expected


def _reservation_value(
    challenge: _ValidatedChallenge,
    *,
    run_dir: Path,
    state: Mapping[str, Any],
    now_ms: int,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": RESERVATION_SCHEMA,
        "challenge_sha256": challenge.value["challenge_sha256"],
        "challenge_artifact_sha256": challenge.artifact_sha256,
        "run_dir": str(run_dir),
        "run_binding_sha256": state["binding_sha256"],
        "reserved_at_ms": now_ms,
        "reservation_nonce": secrets.token_hex(16),
    }
    value["reservation_sha256"] = canonical_sha256(value)
    return value


def _validate_reservation(
    value: Mapping[str, Any],
    challenge: _ValidatedChallenge,
    *,
    run_dir: Path,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    keys = {
        "schema_version",
        "challenge_sha256",
        "challenge_artifact_sha256",
        "run_dir",
        "run_binding_sha256",
        "reserved_at_ms",
        "reservation_nonce",
        "reservation_sha256",
    }
    if set(value) != keys or value["schema_version"] != RESERVATION_SCHEMA:
        raise CameraOnlyError("challenge reservation schema or keys are invalid")
    unsigned = dict(value)
    claimed = _require_sha256(unsigned.pop("reservation_sha256"), label="reservation_sha256")
    if canonical_sha256(unsigned) != claimed:
        raise CameraOnlyError("challenge reservation digest mismatch")
    expected = {
        "challenge_sha256": challenge.value["challenge_sha256"],
        "challenge_artifact_sha256": challenge.artifact_sha256,
        "run_dir": str(run_dir),
        "run_binding_sha256": state["binding_sha256"],
    }
    if any(value[key] != expected_value for key, expected_value in expected.items()):
        raise CameraOnlyError("challenge is reserved by a different run or binding")
    _require_int(value["reserved_at_ms"], label="reserved_at_ms", minimum=contract.MIN_TIMESTAMP_MS)
    if not (challenge.value["issued_at_ms"] <= value["reserved_at_ms"] < challenge.value["expires_at_ms"]):
        raise CameraOnlyError("challenge reservation timestamp is outside the challenge window")
    _require_text(value["reservation_nonce"], label="reservation_nonce")
    return dict(value)


def _acquire_reservation(
    challenge: _ValidatedChallenge,
    *,
    run_dir: Path,
    state: Mapping[str, Any],
    now_ms: int,
) -> dict[str, Any]:
    path = _sidecar(challenge.path, "a0-reserved")
    existing = _read_optional_json(path, label="challenge reservation")
    if existing is not None:
        return _validate_reservation(existing, challenge, run_dir=run_dir, state=state)
    if challenge.consumed is not None:
        raise CameraOnlyError("consumed challenge has no matching A0 production reservation")
    value = _reservation_value(challenge, run_dir=run_dir, state=state, now_ms=now_ms)
    try:
        _write_json_exclusive(path, value)
    except CameraOnlyError:
        raced = _read_optional_json(path, label="raced challenge reservation")
        if raced is None:
            raise
        return _validate_reservation(raced, challenge, run_dir=run_dir, state=state)
    return value


def _produced_value(
    challenge: _ValidatedChallenge,
    *,
    run_dir: Path,
    state: Mapping[str, Any],
    reservation: Mapping[str, Any],
    paths: _TerminalPaths,
    now_ms: int,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": PRODUCED_SCHEMA,
        "challenge_sha256": challenge.value["challenge_sha256"],
        "challenge_artifact_sha256": challenge.artifact_sha256,
        "run_dir": str(run_dir),
        "run_binding_sha256": state["binding_sha256"],
        "reservation_sha256": reservation["reservation_sha256"],
        "frame_bundle_file_sha256": _sha256_bytes(paths.frame_bundle.read_bytes()),
        "acquisition_manifest_sha256": _sha256_bytes(paths.manifest.read_bytes()),
        "record_file_sha256": _sha256_bytes(paths.record.read_bytes()),
        "replay_receipt_file_sha256": _sha256_bytes(paths.receipt.read_bytes()),
        "produced_at_ms": now_ms,
    }
    value["produced_sha256"] = canonical_sha256(value)
    return value


def _validate_produced(
    value: Mapping[str, Any],
    challenge: _ValidatedChallenge,
    *,
    run_dir: Path,
    state: Mapping[str, Any],
    reservation: Mapping[str, Any],
    paths: _TerminalPaths,
) -> dict[str, Any]:
    keys = {
        "schema_version",
        "challenge_sha256",
        "challenge_artifact_sha256",
        "run_dir",
        "run_binding_sha256",
        "reservation_sha256",
        "frame_bundle_file_sha256",
        "acquisition_manifest_sha256",
        "record_file_sha256",
        "replay_receipt_file_sha256",
        "produced_at_ms",
        "produced_sha256",
    }
    if set(value) != keys or value["schema_version"] != PRODUCED_SCHEMA:
        raise CameraOnlyError("challenge produced receipt schema or keys are invalid")
    unsigned = dict(value)
    claimed = _require_sha256(unsigned.pop("produced_sha256"), label="produced_sha256")
    if canonical_sha256(unsigned) != claimed:
        raise CameraOnlyError("challenge produced receipt digest mismatch")
    expected = {
        "challenge_sha256": challenge.value["challenge_sha256"],
        "challenge_artifact_sha256": challenge.artifact_sha256,
        "run_dir": str(run_dir),
        "run_binding_sha256": state["binding_sha256"],
        "reservation_sha256": reservation["reservation_sha256"],
        "frame_bundle_file_sha256": _sha256_bytes(paths.frame_bundle.read_bytes()),
        "acquisition_manifest_sha256": _sha256_bytes(paths.manifest.read_bytes()),
        "record_file_sha256": _sha256_bytes(paths.record.read_bytes()),
        "replay_receipt_file_sha256": _sha256_bytes(paths.receipt.read_bytes()),
    }
    if any(value[key] != expected_value for key, expected_value in expected.items()):
        raise CameraOnlyError("challenge produced receipt differs from complete disk evidence")
    _require_int(value["produced_at_ms"], label="produced_at_ms", minimum=contract.MIN_TIMESTAMP_MS)
    if not (reservation["reserved_at_ms"] <= value["produced_at_ms"] <= challenge.value["expires_at_ms"]):
        raise CameraOnlyError("challenge produced timestamp is outside the reserved challenge window")
    if challenge.consumed is not None and value["produced_at_ms"] > challenge.consumed["consumed_at_ms"]:
        raise CameraOnlyError("challenge was consumed before A0 production completed")
    return dict(value)


def _write_or_validate_produced(
    challenge: _ValidatedChallenge,
    *,
    run_dir: Path,
    state: Mapping[str, Any],
    reservation: Mapping[str, Any],
    paths: _TerminalPaths,
    now_ms: int,
) -> dict[str, Any]:
    path = _sidecar(challenge.path, "a0-produced")
    existing = _read_optional_json(path, label="challenge produced receipt")
    if existing is not None:
        return _validate_produced(
            existing,
            challenge,
            run_dir=run_dir,
            state=state,
            reservation=reservation,
            paths=paths,
        )
    value = _produced_value(
        challenge,
        run_dir=run_dir,
        state=state,
        reservation=reservation,
        paths=paths,
        now_ms=now_ms,
    )
    _write_json_exclusive(path, value)
    return value


def _revalidate_state_disk(run_dir: Path, state: Mapping[str, Any]) -> None:
    empty_paths = _validate_frame_rows(run_dir, state, "empty")
    occupied_paths = _validate_frame_rows(run_dir, state, "occupied")
    if len(empty_paths) + len(occupied_paths) != len(_all_frame_hashes(state)):
        raise CameraOnlyError("empty and occupied frame identities overlap")
    if len(empty_paths) == FRAME_COUNT:
        _materialize_authoritative_inputs(run_dir, state, "empty")
    if len(occupied_paths) == FRAME_COUNT:
        _materialize_authoritative_inputs(run_dir, state, "occupied")
    if state["capture_identity"] is not None:
        _identity_from_state(state["capture_identity"], label="stored arm02", hostname=ARM02_HOSTNAME)
        _identity_from_state(state["inference_identity"], label="stored AI X5", hostname=AI_HOSTNAME)
        service = _validate_service_state_value(state["capture_service_identity"])
        runtime_script, raw = _regular_file(
            run_dir / "runtime" / contract.CAPTURE_PIPELINE_CONTRACT.name,
            label="remote actual capture script",
            max_bytes=contract.MAX_PIPELINE_ARTIFACT_BYTES,
        )
        del runtime_script
        if _sha256_bytes(raw) != service["script_sha256"] or len(raw) != service["script_size_bytes"]:
            raise CameraOnlyError("remote capture script artifact differs from service identity")
        _, identity_raw = _regular_file(
            run_dir / "runtime" / "camera_service_runtime.json",
            label="camera service runtime artifact",
            max_bytes=128 * 1024,
        )
        if _load_json_bytes(identity_raw, label="camera service runtime artifact") != service:
            raise CameraOnlyError("camera service runtime artifact differs from state")
        _validate_binding(run_dir, state)
    elif empty_paths or occupied_paths:
        raise CameraOnlyError("captured frames exist without a bound runtime identity")
    baseline = state["bag_empty_baseline_sha256"]
    if baseline is not None:
        if len(empty_paths) != FRAME_COUNT:
            raise CameraOnlyError("completed empty baseline does not have exactly five frames")
        result_ref = state["empty_result"]
        if not isinstance(result_ref, Mapping) or set(result_ref) != {"path", "sha256"}:
            raise CameraOnlyError("completed empty result reference is malformed")
        result_path, result_raw = _regular_file(
            result_ref["path"], label="persisted empty result", max_bytes=contract.MAX_RESULT_JSON_BYTES
        )
        if _sha256_bytes(result_raw) != result_ref["sha256"]:
            raise CameraOnlyError("persisted empty result hash mismatch")
        if _validate_empty_result(result_path, state) != baseline:
            raise CameraOnlyError("persisted empty baseline digest changed")
        _require_int(
            state["empty_completed_at_ms"],
            label="empty_completed_at_ms",
            minimum=contract.MIN_TIMESTAMP_MS,
        )
    elif any(
        state[field] is not None
        for field in ("empty_result", "empty_completed_at_ms", "challenge_sha256", "reservation_sha256")
    ):
        raise CameraOnlyError("state claims downstream evidence without an empty baseline")


def _ordered_role_paths(run_dir: Path, state: Mapping[str, Any]) -> list[tuple[str, Path]]:
    empty_paths = _materialize_authoritative_inputs(run_dir, state, "empty")
    occupied_paths = _materialize_authoritative_inputs(run_dir, state, "occupied")
    return [
        *(("EMPTY_BASELINE", path) for path in empty_paths),
        *(("OCCUPIED_CANDIDATE", path) for path in occupied_paths),
    ]


def _build_or_validate_frame_bundle(
    path: Path,
    *,
    run_dir: Path,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    role_paths = _ordered_role_paths(run_dir, state)
    try:
        expected = contract.build_input_frame_bundle(role_paths)
    except contract.ActualRecordError as exc:
        raise CameraOnlyError(str(exc)) from exc
    expected_raw = contract.canonical_json_bytes(expected) + b"\n"
    if path.exists() or path.is_symlink():
        _, raw = _regular_file(
            path,
            label="input frame bundle",
            max_bytes=contract.MAX_INPUT_FRAME_BUNDLE_BYTES,
        )
        if raw != expected_raw:
            raise CameraOnlyError("input frame bundle differs from all ten exact JPEGs")
    else:
        try:
            contract.build_input_frame_bundle(role_paths, output=path)
        except contract.ActualRecordError as exc:
            raise CameraOnlyError(str(exc)) from exc
    return expected


def _selected_row(state: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = state["occupied_frames"]
    if len(rows) != FRAME_COUNT:
        raise CameraOnlyError("selected raw frame requires five occupied frames")
    return rows[0]


def _terminal_paths(
    run_dir: Path,
    state: Mapping[str, Any],
    expected: contract.ExpectedAcquisition,
    output_contract: _OutputContract,
    camera_service_identity_sha256: str,
) -> _TerminalPaths:
    del state
    if output_contract.root != run_dir:
        raise CameraOnlyError("production output contract root differs from run_dir")
    replay_identity = contract.replay_identity_sha256_for_expected(
        expected,
        camera_service_identity_sha256=camera_service_identity_sha256,
    )
    return _TerminalPaths(
        evidence_dir=run_dir,
        replay_dir=output_contract.replay_dir,
        capture_artifact=output_contract.capture_artifact,
        inference_artifact=output_contract.inference_artifact,
        raw_frame=output_contract.raw_frame,
        frame_bundle=output_contract.frame_bundle,
        result_json=output_contract.result_json,
        manifest=output_contract.manifest,
        record=output_contract.record,
        receipt=output_contract.replay_dir / f"{replay_identity}.json",
        camera_service_identity=output_contract.camera_service_identity,
    )


def _terminal_presence(paths: _TerminalPaths) -> dict[str, bool]:
    return {
        "manifest": paths.manifest.exists() or paths.manifest.is_symlink(),
        "record": paths.record.exists() or paths.record.is_symlink(),
        "receipt": paths.receipt.exists() or paths.receipt.is_symlink(),
    }


def _validate_terminal_sources(
    paths: _TerminalPaths,
    *,
    run_dir: Path,
    state: Mapping[str, Any],
    expected: contract.ExpectedAcquisition,
    now_ms: int,
) -> dict[str, Any]:
    selected = _selected_row(state)
    _build_or_validate_frame_bundle(paths.frame_bundle, run_dir=run_dir, state=state)
    _frame_path(run_dir, "occupied", selected)
    _, result_raw = _regular_file(
        paths.result_json, label="terminal result", max_bytes=contract.MAX_RESULT_JSON_BYTES
    )
    _validate_occupied_result(
        paths.result_json,
        state,
        selected_name="overhead_a0_frame.jpg",
        selected_sha256=selected["sha256"],
    )
    _, raw_frame = _regular_file(
        paths.raw_frame, label="terminal raw frame", max_bytes=contract.MAX_FRAME_BYTES
    )
    if raw_frame != _frame_path(run_dir, "occupied", selected).read_bytes():
        raise CameraOnlyError("terminal raw frame differs from its selected content-addressed source")
    if _sha256_bytes(result_raw) != state["inference_result"]["sha256"]:
        raise CameraOnlyError("terminal result differs from persisted inference result")
    try:
        record = contract.load_record(paths.record)
        contract._validate_sealed_record_with_clock(
            record,
            replay_ledger_dir=paths.replay_dir,
            acquisition_manifest=paths.manifest,
            raw_frame=paths.raw_frame,
            frame_bundle_artifact=paths.frame_bundle,
            result_json=paths.result_json,
            capture_pipeline_artifact=paths.capture_artifact,
            inference_pipeline_artifact=paths.inference_artifact,
            camera_service_identity_artifact=paths.camera_service_identity,
            expected=expected,
            now_ms=min(now_ms, expected.challenge_expires_at_ms),
        )
    except contract.ActualRecordError as exc:
        raise CameraOnlyError(str(exc)) from exc
    return record


def _set_terminal_state(
    state: dict[str, Any],
    *,
    paths: _TerminalPaths,
    record: Mapping[str, Any],
) -> None:
    state["frame_bundle"] = {
        "path": str(paths.frame_bundle),
        "sha256": _sha256_bytes(paths.frame_bundle.read_bytes()),
    }
    state["acquisition_manifest"] = {
        "path": str(paths.manifest),
        "sha256": _sha256_bytes(paths.manifest.read_bytes()),
    }
    state["actual_record"] = {
        "path": str(paths.record),
        "sha256": _sha256_bytes(paths.record.read_bytes()),
    }
    state["replay_receipt"] = {
        "path": str(paths.receipt),
        "sha256": _sha256_bytes(paths.receipt.read_bytes()),
    }
    if record["schema_version"] != contract.SCHEMA_VERSION:
        raise CameraOnlyError("terminal record is not the current production schema")
    state["status"] = "SEALED"
    state["last_error"] = None


def _recover_complete_terminal(
    *,
    challenge: _ValidatedChallenge,
    reservation: Mapping[str, Any],
    run_dir: Path,
    state: dict[str, Any],
    expected: contract.ExpectedAcquisition,
    paths: _TerminalPaths,
    now_ms: int,
) -> bool:
    presence = _terminal_presence(paths)
    produced_path = _sidecar(challenge.path, "a0-produced")
    produced_present = produced_path.exists() or produced_path.is_symlink()
    if all(presence.values()):
        record = _validate_terminal_sources(
            paths,
            run_dir=run_dir,
            state=state,
            expected=expected,
            now_ms=now_ms,
        )
        if challenge.consumed is not None and not produced_present:
            raise CameraOnlyError("consumed challenge lacks a prior A0 produced receipt")
        _write_or_validate_produced(
            challenge,
            run_dir=run_dir,
            state=state,
            reservation=reservation,
            paths=paths,
            now_ms=now_ms,
        )
        _set_terminal_state(state, paths=paths, record=record)
        return True
    if any(presence.values()) or produced_present or state["status"] == "SEALED":
        raise CameraOnlyError("incomplete terminal evidence cannot be resumed or resealed")
    if challenge.consumed is not None:
        raise CameraOnlyError("challenge was consumed without complete matching A0 evidence")
    return False


def _summary(state: Mapping[str, Any]) -> dict[str, Any]:
    production = state["authority_mode"] == "PRODUCTION"
    root = Path(state["run_dir"])
    production_paths = None
    if production:
        production_paths = {
            "record": str(root / "overhead_a0_record.json"),
            "acquisition_manifest": str(root / "overhead_a0_acquisition.json"),
            "raw_frame": str(root / "overhead_a0_frame.jpg"),
            "camera_service_identity": str(root / "camera_service_identity.json"),
            "frame_bundle_artifact": str(root / "overhead_a0_input_frames.json"),
            "result_json": str(root / "overhead_a0_result.json"),
            "capture_pipeline_artifact": str(root / contract.CAPTURE_PIPELINE_CONTRACT.name),
            "inference_pipeline_artifact": str(
                root / contract.INFERENCE_PIPELINE_CONTRACTS[RESULT_SCHEMA].name
            ),
            "replay_ledger_dir": str(root / "replay_ledger"),
            "replay_receipt": (
                state["replay_receipt"]["path"] if isinstance(state["replay_receipt"], Mapping) else None
            ),
        }
    return {
        "schema_version": RUNNER_SCHEMA,
        "status": state["status"],
        "authority_mode": state["authority_mode"],
        "run_dir": state["run_dir"],
        "empty_frame_count": len(state["empty_frames"]),
        "occupied_frame_count": len(state["occupied_frames"]),
        "bag_empty_baseline_sha256": state["bag_empty_baseline_sha256"],
        "challenge_sha256": state["challenge_sha256"],
        "frame_bundle": state["frame_bundle"],
        "acquisition_manifest": state["acquisition_manifest"],
        "actual_record": state["actual_record"],
        "replay_receipt": state["replay_receipt"],
        "production_output_paths": production_paths,
        "last_error": state["last_error"],
        "remote_contacted": state["remote_contacted"] if production else False,
        "network_touched": state["remote_contacted"] if production else False,
        "hardware_touched": production and bool(state["empty_frames"] or state["occupied_frames"]),
        "read_only_transport_operations": state["read_only_transport_operations"],
        "transport_commands_issued": (state["read_only_transport_operations"] if production else 0),
        "read_only_device_observations": (state["read_only_transport_operations"] if production else 0),
        "execution_authority": False,
        "actuator_commands_issued": 0,
        "mutating_commands_issued": 0,
    }


_PRODUCTION_CAPABILITY = object()


def _execute_with_surfaces(
    *,
    phase: str,
    run_dir: Path | str,
    transport: Any,
    inference: Any,
    identity_provider: Any,
    known_hosts: Path | str,
    live_config_file: Path | str | None,
    challenge_file: Path | str | None,
    frame_count: int,
    max_frame_age_s: float,
    snapshot_bracket_ms: int,
    interval_s: float,
    clock_ms: Callable[[], int] | None,
    sleeper: Callable[[float], None],
    production_capability: object | None,
) -> dict[str, Any]:
    production = production_capability is _PRODUCTION_CAPABILITY
    if production and not (
        type(transport) is SshCameraTransport
        and type(inference) is FixedInferenceExecutor
        and type(identity_provider) is LocalIdentityProvider
    ):
        raise CameraOnlyError("production sealing requires all three exact built-in surfaces")
    if production and (
        getattr(transport, "timeout_s", None) != PRODUCTION_SSH_TIMEOUT_S
        or getattr(inference, "timeout_s", None) != PRODUCTION_INFERENCE_TIMEOUT_S
    ):
        raise CameraOnlyError("production timeout budget differs from the frozen challenge contract")
    if phase not in {"empty", "occupied"}:
        raise CameraOnlyError("phase must be empty or occupied")
    if isinstance(frame_count, bool) or frame_count != FRAME_COUNT:
        raise CameraOnlyError("production and simulation both freeze exactly five frames per state")
    frozen_known_hosts = _validate_known_hosts(known_hosts)
    output_contract: _OutputContract | None = None
    if production:
        if live_config_file is None:
            raise CameraOnlyError("production execution requires --live-config")
        output_contract = _load_output_contract(
            live_config_file,
            run_dir=run_dir,
            known_hosts=frozen_known_hosts,
        )
    _parameters(
        max_frame_age_s=max_frame_age_s,
        snapshot_bracket_ms=snapshot_bracket_ms,
        interval_s=interval_s,
        output_contract=output_contract,
    )
    root = _prepare_run_dir(run_dir)
    arguments = {
        "phase": phase,
        "run_dir": root,
        "prepared_run_dir": root,
        "transport": transport,
        "inference": inference,
        "identity_provider": identity_provider,
        "known_hosts": known_hosts,
        "live_config_file": live_config_file,
        "challenge_file": challenge_file,
        "frame_count": frame_count,
        "max_frame_age_s": max_frame_age_s,
        "snapshot_bracket_ms": snapshot_bracket_ms,
        "interval_s": interval_s,
        "clock_ms": clock_ms,
        "sleeper": sleeper,
        "production_capability": production_capability,
    }
    if production_capability is _PRODUCTION_CAPABILITY:
        with _run_execution_lease(root):
            return _execute_with_surfaces_locked(**arguments)
    return _execute_with_surfaces_locked(**arguments)


def _execute_with_surfaces_locked(
    *,
    phase: str,
    run_dir: Path | str,
    prepared_run_dir: Path,
    transport: Any,
    inference: Any,
    identity_provider: Any,
    known_hosts: Path | str,
    live_config_file: Path | str | None,
    challenge_file: Path | str | None,
    frame_count: int,
    max_frame_age_s: float,
    snapshot_bracket_ms: int,
    interval_s: float,
    clock_ms: Callable[[], int] | None,
    sleeper: Callable[[float], None],
    production_capability: object | None,
) -> dict[str, Any]:
    production = production_capability is _PRODUCTION_CAPABILITY
    if production and not (
        type(transport) is SshCameraTransport
        and type(inference) is FixedInferenceExecutor
        and type(identity_provider) is LocalIdentityProvider
    ):
        raise CameraOnlyError("production sealing requires all three exact built-in surfaces")
    if phase not in {"empty", "occupied"}:
        raise CameraOnlyError("phase must be empty or occupied")
    if isinstance(frame_count, bool) or frame_count != FRAME_COUNT:
        raise CameraOnlyError("production and simulation both freeze exactly five frames per state")
    frozen_known_hosts = _validate_known_hosts(known_hosts)
    output_contract: _OutputContract | None = None
    if production:
        if live_config_file is None:
            raise CameraOnlyError("production execution requires --live-config")
        output_contract = _load_output_contract(
            live_config_file,
            run_dir=run_dir,
            known_hosts=frozen_known_hosts,
        )
    parameters = _parameters(
        max_frame_age_s=max_frame_age_s,
        snapshot_bracket_ms=snapshot_bracket_ms,
        interval_s=interval_s,
        output_contract=output_contract,
    )
    now = clock_ms or (lambda: time.time_ns() // 1_000_000)
    first_now = _require_int(now(), label="clock_ms result", minimum=contract.MIN_TIMESTAMP_MS)
    root = prepared_run_dir
    authority_mode = "PRODUCTION" if production else "SIMULATION_ONLY"
    state_path, state = _state_store(
        root,
        authority_mode=authority_mode,
        parameters=parameters,
        now_ms=first_now,
    )
    try:
        _revalidate_state_disk(root, state)
        state["status"] = f"{phase.upper()}_RUNNING"
        state["last_error"] = None
        _save_state(state_path, state, now_ms=now())
        if phase == "empty":
            _capture_frames(
                phase="empty",
                run_dir=root,
                state_path=state_path,
                state=state,
                transport=transport,
                identity_provider=identity_provider,
                clock_ms=now,
                sleeper=sleeper,
                production=production,
            )
            if state["bag_empty_baseline_sha256"] is not None:
                state["status"] = "EMPTY_COMPLETE"
                _save_state(state_path, state, now_ms=now())
                return _summary(state)
            empty_inputs = _materialize_authoritative_inputs(root, state, "empty")
            result = inference.run(
                empty_dir=empty_inputs[0].parent,
                occupied_dir=None,
                output_dir=root / "inference" / "empty",
            )
            baseline = _validate_empty_result(result, state)
            state["empty_result"] = {
                "path": str(result),
                "sha256": _sha256_bytes(result.read_bytes()),
            }
            state["bag_empty_baseline_sha256"] = baseline
            state["empty_completed_at_ms"] = now()
            state["status"] = "EMPTY_COMPLETE"
            _write_or_validate_binding(root, state)
            _save_state(state_path, state, now_ms=now())
            return _summary(state)

        if state["bag_empty_baseline_sha256"] is None:
            raise CameraOnlyError("occupied phase requires a completed empty baseline")
        if challenge_file is None:
            raise CameraOnlyError("occupied phase requires challenge_file")
        _validate_binding(root, state)
        possible_terminal = (root / "overhead_a0_record.json").exists()
        challenge = _validate_challenge(
            challenge_file,
            known_hosts=frozen_known_hosts,
            state=state,
            now_ms=now(),
            allow_terminal_recovery=possible_terminal,
        )
        if output_contract is not None and (
            challenge.value["config_sha256"] != output_contract.config_sha256
            or challenge.value["release_id"] != output_contract.release_id
        ):
            raise CameraOnlyError("challenge config/release differs from the production output contract")
        if state["challenge_sha256"] not in (None, challenge.value["challenge_sha256"]):
            raise CameraOnlyError("state is already bound to a different challenge")
        state["challenge_sha256"] = challenge.value["challenge_sha256"]
        reservation: Mapping[str, Any] | None = None
        if production:
            reservation = _acquire_reservation(
                challenge,
                run_dir=root,
                state=state,
                now_ms=now(),
            )
            state["reservation_sha256"] = reservation["reservation_sha256"]
            _save_state(state_path, state, now_ms=now())
        _capture_frames(
            phase="occupied",
            run_dir=root,
            state_path=state_path,
            state=state,
            transport=transport,
            identity_provider=identity_provider,
            clock_ms=now,
            sleeper=sleeper,
            production=production,
        )
        expected = _expected_from_challenge(challenge.value, state)
        selected = _selected_row(state)

        if not production:
            simulation_dir = root / "simulation"
            simulation_dir.mkdir(mode=0o700, exist_ok=True)
            bundle_path = simulation_dir / "input_frames.json"
            _build_or_validate_frame_bundle(bundle_path, run_dir=root, state=state)
            empty_inputs = _materialize_authoritative_inputs(root, state, "empty")
            occupied_inputs = _materialize_authoritative_inputs(root, state, "occupied")
            result = inference.run(
                empty_dir=empty_inputs[0].parent,
                occupied_dir=occupied_inputs[0].parent,
                output_dir=root / "inference" / "occupied",
            )
            _validate_occupied_result(
                result,
                state,
                selected_name="overhead_a0_frame.jpg",
                selected_sha256=selected["sha256"],
            )
            state["frame_bundle"] = {
                "path": str(bundle_path),
                "sha256": _sha256_bytes(bundle_path.read_bytes()),
            }
            state["status"] = "SIMULATED_COUNTERFACTUAL"
            state["last_error"] = None
            _save_state(state_path, state, now_ms=now())
            return _summary(state)

        if reservation is None:  # pragma: no cover - guarded by production branch
            raise RuntimeError("production reservation is unavailable")
        if output_contract is None:  # pragma: no cover - production invariant
            raise RuntimeError("production output contract is unavailable")
        service_observed_at_ms = _require_int(
            state["occupied_frames"][0]["health_before_at_ms"],
            label="camera service observed_at_ms",
            minimum=contract.MIN_TIMESTAMP_MS,
        )
        service_artifact = _materialize_camera_service_identity(
            output_contract.camera_service_identity,
            state=state,
            observed_at_ms=service_observed_at_ms,
            expected=expected,
        )
        paths = _terminal_paths(
            root,
            state,
            expected,
            output_contract,
            camera_service_identity_sha256=service_artifact["artifact_sha256"],
        )
        paths.evidence_dir.mkdir(mode=0o700, exist_ok=True)
        paths.replay_dir.mkdir(mode=0o700, exist_ok=True)
        if _recover_complete_terminal(
            challenge=challenge,
            reservation=reservation,
            run_dir=root,
            state=state,
            expected=expected,
            paths=paths,
            now_ms=now(),
        ):
            _save_state(state_path, state, now_ms=now())
            return _summary(state)

        runtime_capture = root / "runtime" / contract.CAPTURE_PIPELINE_CONTRACT.name
        inference_pipeline = _inference_pipeline_path()
        selected_source = _frame_path(root, "occupied", selected)
        _ensure_bytes(paths.capture_artifact, runtime_capture.read_bytes())
        _ensure_bytes(paths.inference_artifact, inference_pipeline.read_bytes())
        _ensure_bytes(paths.raw_frame, selected_source.read_bytes())
        bundle = _build_or_validate_frame_bundle(paths.frame_bundle, run_dir=root, state=state)

        capture_identity = _identity_from_state(
            state["capture_identity"], label="stored arm02", hostname=ARM02_HOSTNAME
        )
        inference_identity = _identity_from_state(
            state["inference_identity"], label="stored AI X5", hostname=AI_HOSTNAME
        )
        recorder = acquisition.A0AcquisitionRecorder(
            expected=expected,
            capture_host=acquisition.ProducerHostIdentity(
                hostname=capture_identity.hostname,
                device_id=capture_identity.device_id,
                boot_id=capture_identity.boot_id,
                session_id=state["capture_session_id"],
            ),
            inference_host=acquisition.ProducerHostIdentity(
                hostname=inference_identity.hostname,
                device_id=inference_identity.device_id,
                boot_id=inference_identity.boot_id,
                session_id=state["inference_session_id"],
            ),
            producer_started_at_ms=max(
                challenge.value["issued_at_ms"],
                state["occupied_frames"][0]["health_before_at_ms"],
            ),
        )
        recorder.record_camera_service_identity(
            camera_service_identity_artifact=paths.camera_service_identity,
        )
        recorder.record_camera_opened(opened_at_ms=state["occupied_frames"][0]["health_before_at_ms"])
        recorder.record_frame_captured(
            raw_frame=paths.raw_frame,
            capture_pipeline_artifact=paths.capture_artifact,
            captured_at_ms=selected["captured_at_ms"],
        )
        existing_bundle_ref = state["frame_bundle"]
        if existing_bundle_ref is None:
            bundle_bound_at_ms = now()
        else:
            if not isinstance(existing_bundle_ref, Mapping) or set(existing_bundle_ref) != {
                "path",
                "sha256",
                "bundle_sha256",
                "bound_at_ms",
            }:
                raise CameraOnlyError("persisted pre-inference frame bundle reference is malformed")
            if (
                existing_bundle_ref["path"] != str(paths.frame_bundle)
                or existing_bundle_ref["sha256"] != _sha256_bytes(paths.frame_bundle.read_bytes())
                or existing_bundle_ref["bundle_sha256"] != bundle["bundle_sha256"]
            ):
                raise CameraOnlyError("persisted pre-inference frame bundle reference drifted")
            bundle_bound_at_ms = _require_int(
                existing_bundle_ref["bound_at_ms"],
                label="frame bundle bound_at_ms",
                minimum=contract.MIN_TIMESTAMP_MS,
            )
        recorder.record_input_frame_bundle(
            frame_bundle_artifact=paths.frame_bundle,
            bound_at_ms=bundle_bound_at_ms,
        )
        state["frame_bundle"] = {
            "path": str(paths.frame_bundle),
            "sha256": _sha256_bytes(paths.frame_bundle.read_bytes()),
            "bundle_sha256": bundle["bundle_sha256"],
            "bound_at_ms": bundle_bound_at_ms,
        }
        _save_state(state_path, state, now_ms=now())

        if state["inference_result"] is None:
            inference_started_at_ms = now()
            state["inference_started_at_ms"] = inference_started_at_ms
            recorder.record_inference_started(started_at_ms=inference_started_at_ms)
            _save_state(state_path, state, now_ms=now())
            empty_inputs = _materialize_authoritative_inputs(root, state, "empty")
            occupied_inputs = _materialize_authoritative_inputs(root, state, "occupied")
            result_source = inference.run(
                empty_dir=empty_inputs[0].parent,
                occupied_dir=occupied_inputs[0].parent,
                output_dir=root / "inference" / "occupied",
            )
            inference_completed_at_ms = now()
            _, result_raw = _validate_occupied_result(
                result_source,
                state,
                selected_name="overhead_a0_frame.jpg",
                selected_sha256=selected["sha256"],
            )
            _ensure_bytes(paths.result_json, result_raw)
            state["inference_result"] = {
                "path": str(paths.result_json),
                "sha256": _sha256_bytes(result_raw),
            }
            state["inference_completed_at_ms"] = inference_completed_at_ms
            _save_state(state_path, state, now_ms=now())
        else:
            result_ref = state["inference_result"]
            if not isinstance(result_ref, Mapping) or set(result_ref) != {"path", "sha256"}:
                raise CameraOnlyError("persisted inference result reference is malformed")
            if result_ref["path"] != str(paths.result_json):
                raise CameraOnlyError("persisted inference result path drifted")
            _, result_raw = _regular_file(
                paths.result_json,
                label="persisted occupied result",
                max_bytes=contract.MAX_RESULT_JSON_BYTES,
            )
            if _sha256_bytes(result_raw) != result_ref["sha256"]:
                raise CameraOnlyError("persisted occupied result hash drifted")
            _validate_occupied_result(
                paths.result_json,
                state,
                selected_name="overhead_a0_frame.jpg",
                selected_sha256=selected["sha256"],
            )
            inference_started_at_ms = _require_int(
                state["inference_started_at_ms"],
                label="inference_started_at_ms",
                minimum=contract.MIN_TIMESTAMP_MS,
            )
            inference_completed_at_ms = _require_int(
                state["inference_completed_at_ms"],
                label="inference_completed_at_ms",
                minimum=inference_started_at_ms,
            )
            recorder.record_inference_started(started_at_ms=inference_started_at_ms)

        recorder.record_inference_completed(
            result_json=paths.result_json,
            inference_pipeline_artifact=paths.inference_artifact,
            completed_at_ms=inference_completed_at_ms,
        )
        manifest_emitted_at_ms = now()
        recorder.emit_once(paths.manifest, manifest_emitted_at_ms=manifest_emitted_at_ms)
        production_attestation = recorder.production_attestation()
        try:
            record = contract._seal_record_to_path_once_with_clock(
                output=paths.record,
                replay_ledger_dir=paths.replay_dir,
                acquisition_manifest=paths.manifest,
                raw_frame=paths.raw_frame,
                frame_bundle_artifact=paths.frame_bundle,
                result_json=paths.result_json,
                capture_pipeline_artifact=paths.capture_artifact,
                inference_pipeline_artifact=paths.inference_artifact,
                camera_service_identity_artifact=paths.camera_service_identity,
                production_attestation=production_attestation,
                expected=expected,
                now_ms=now(),
            )
        except contract.ActualRecordError as exc:
            raise CameraOnlyError(str(exc)) from exc
        _write_or_validate_produced(
            challenge,
            run_dir=root,
            state=state,
            reservation=reservation,
            paths=paths,
            now_ms=now(),
        )
        _set_terminal_state(state, paths=paths, record=record)
        _save_state(state_path, state, now_ms=now())
        return _summary(state)
    except (CameraOnlyError, contract.ActualRecordError, OSError, subprocess.SubprocessError) as exc:
        state["status"] = "BLOCKED"
        state["last_error"] = f"{type(exc).__name__}: {exc}"
        _save_state(state_path, state, now_ms=now())
        if isinstance(exc, CameraOnlyError):
            raise
        raise CameraOnlyError(str(exc)) from exc


def execute_camera_only(
    *,
    phase: str,
    run_dir: Path | str,
    transport: Any,
    inference: Any,
    identity_provider: Any,
    known_hosts: Path | str,
    live_config_file: Path | str | None = None,
    challenge_file: Path | str | None = None,
    frame_count: int = FRAME_COUNT,
    max_frame_age_s: float = DEFAULT_MAX_FRAME_AGE_S,
    snapshot_bracket_ms: int = DEFAULT_SNAPSHOT_BRACKET_MS,
    interval_s: float = DEFAULT_INTERVAL_S,
    clock_ms: Callable[[], int] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run injected surfaces in simulation-only mode; production sealing is impossible."""

    return _execute_with_surfaces(
        phase=phase,
        run_dir=run_dir,
        transport=transport,
        inference=inference,
        identity_provider=identity_provider,
        known_hosts=known_hosts,
        live_config_file=live_config_file,
        challenge_file=challenge_file,
        frame_count=frame_count,
        max_frame_age_s=max_frame_age_s,
        snapshot_bracket_ms=snapshot_bracket_ms,
        interval_s=interval_s,
        clock_ms=clock_ms,
        sleeper=sleeper,
        production_capability=None,
    )


def _execute_production_phase(
    *,
    phase: str,
    run_dir: Path | str,
    known_hosts: Path | str,
    live_config_file: Path | str,
    challenge_file: Path | str | None,
    frame_count: int,
    max_frame_age_s: float,
    snapshot_bracket_ms: int,
    interval_s: float,
) -> dict[str, Any]:
    transport = SshCameraTransport(known_hosts=known_hosts)
    inference = FixedInferenceExecutor()
    identity_provider = LocalIdentityProvider()
    return _execute_with_surfaces(
        phase=phase,
        run_dir=run_dir,
        transport=transport,
        inference=inference,
        identity_provider=identity_provider,
        known_hosts=known_hosts,
        live_config_file=live_config_file,
        challenge_file=challenge_file,
        frame_count=frame_count,
        max_frame_age_s=max_frame_age_s,
        snapshot_bracket_ms=snapshot_bracket_ms,
        interval_s=interval_s,
        clock_ms=None,
        sleeper=time.sleep,
        production_capability=_PRODUCTION_CAPABILITY,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="R2 camera-only A0 producer; default is PlanOnly.")
    parser.add_argument("--execute-camera-only", action="store_true")
    parser.add_argument("--phase", choices=("empty", "occupied"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--known-hosts", type=Path)
    parser.add_argument("--live-config", type=Path)
    parser.add_argument("--challenge-file", type=Path)
    parser.add_argument("--frame-count", type=int, default=FRAME_COUNT)
    parser.add_argument("--max-frame-age-s", type=float, default=DEFAULT_MAX_FRAME_AGE_S)
    parser.add_argument("--snapshot-bracket-ms", type=int, default=DEFAULT_SNAPSHOT_BRACKET_MS)
    parser.add_argument("--interval-s", type=float, default=DEFAULT_INTERVAL_S)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.execute_camera_only:
        sys.stdout.write(canonical_bytes(plan()).decode("utf-8") + "\n")
        return 0
    if args.phase is None or args.run_dir is None or args.known_hosts is None or args.live_config is None:
        sys.stderr.write(
            "ERROR: explicit execution requires --phase, --run-dir, --known-hosts, and --live-config\n"
        )
        return 2
    if args.phase == "occupied" and args.challenge_file is None:
        sys.stderr.write("ERROR: occupied execution requires --challenge-file\n")
        return 2
    try:
        result = _execute_production_phase(
            phase=args.phase,
            run_dir=args.run_dir,
            known_hosts=args.known_hosts,
            live_config_file=args.live_config,
            challenge_file=args.challenge_file,
            frame_count=args.frame_count,
            max_frame_age_s=args.max_frame_age_s,
            snapshot_bracket_ms=args.snapshot_bracket_ms,
            interval_s=args.interval_s,
        )
    except CameraOnlyError as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 2
    sys.stdout.write(canonical_bytes(result).decode("utf-8") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
