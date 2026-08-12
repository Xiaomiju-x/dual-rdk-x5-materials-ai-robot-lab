"""Central, fail-closed R2-PREP live-shadow runner for the AI X5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import secrets
import shlex
import stat
import sys
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType, SimpleNamespace
from typing import Any, Final

import rb_voe.adapters.read_only as read_only_transport
from rb_voe.adapters.ai_x5 import (
    AI_X5_REQUIRED_BACKENDS,
    AI_X5_RUNTIME_SNAPSHOT_PATH,
    AI_X5_RUNTIME_SNAPSHOT_SCHEMA,
    AI_X5_SNAPSHOT_PATHS,
    AiX5CapabilityBinding,
    AiX5ReadOnlyAdapter,
    canonical_runtime_artifact_set_sha256,
)
from rb_voe.adapters.assay_station import AssayStationTargetAdapter
from rb_voe.adapters.dual_arm import (
    DUAL_ARM_MEMBER_PATHS,
    DUAL_ARM_MEMBER_PROBE_SHA256,
    DUAL_ARM_MEMBER_SCHEMA,
    DUAL_ARM_PROFILE_SHA256,
    DUAL_ARM_VISION_RECORD_PATH,
    DualArmCapabilityBinding,
    DualArmReadOnlyAdapter,
    a0_contract,
)
from rb_voe.adapters.embodied_x5 import (
    EMBODIED_X5_ARTIFACT_PATHS,
    EMBODIED_X5_REQUIRED_ARTIFACTS,
    EMBODIED_X5_SNAPSHOT_PATH,
    EMBODIED_X5_SNAPSHOT_SCHEMA,
    EmbodiedX5CapabilityBinding,
    EmbodiedX5ReadOnlyAdapter,
)
from rb_voe.adapters.read_only import (
    FileJsonSnapshotTransport,
    HttpJsonSnapshotTransport,
    JsonSnapshotTransport,
    PrefetchedJsonSnapshotTransport,
    ReadSourceKind,
)
from rb_voe.contracts.canonical import (
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    require_sha256,
)
from rb_voe.runtime_identity import local_boot_id, local_device_id
from rb_voe.semantic_profiles import load_all_profiles
from rb_voe.shadow import ShadowCoordinator, ShadowMode, ShadowRunBinding, ShadowRunReport

LIVE_SHADOW_CONFIG_SCHEMA: Final[str] = "xrd-rb-voe-live-shadow-config-v4"
LIVE_BOOTSTRAP_BINDING_SCHEMA: Final[str] = "xrd-rb-voe-live-bootstrap-binding-v1"
LIVE_BOOTSTRAP_MANIFEST_SCHEMA: Final[str] = "xrd-rb-voe-live-bootstrap-manifest-v1"
LIVE_SHADOW_PLAN_SCHEMA: Final[str] = "xrd-rb-voe-live-shadow-plan-v1"
LIVE_SHADOW_CHALLENGE_SCHEMA: Final[str] = "xrd-rb-voe-live-shadow-challenge-v4"
LIVE_SHADOW_CHALLENGE_ISSUANCE_SCHEMA: Final[str] = "xrd-rb-voe-live-shadow-challenge-issuance-v2"
LIVE_SHADOW_CHALLENGE_CONSUMPTION_SCHEMA: Final[str] = "xrd-rb-voe-live-shadow-challenge-consumption-v2"
A0_CHALLENGE_RESERVATION_SCHEMA: Final[str] = "xrd-rb-voe-a0-challenge-reservation-v1"
A0_CHALLENGE_PRODUCED_SCHEMA: Final[str] = "xrd-rb-voe-a0-challenge-produced-v1"
LIVE_SHADOW_EVIDENCE_SCHEMA: Final[str] = "xrd-rb-voe-live-shadow-evidence-v1"
LIVE_SHADOW_ATTEMPT_EVIDENCE_SCHEMA: Final[str] = "xrd-rb-voe-live-shadow-attempt-evidence-v1"
LIVE_SNAPSHOT_MAX_AGE_MS: Final[int] = 20_000
LIVE_CHALLENGE_TTL_MS: Final[int] = 10 * 60 * 1000
LIVE_A0_OCCUPIED_WORST_CASE_MS: Final[int] = 353_000
LIVE_COLLECTOR_PREFLIGHT_WORST_CASE_MS: Final[int] = 28_000
LIVE_CHALLENGE_SEALING_MARGIN_MS: Final[int] = 60_000
LIVE_CHALLENGE_MIN_BUDGET_MS: Final[int] = (
    LIVE_A0_OCCUPIED_WORST_CASE_MS + LIVE_COLLECTOR_PREFLIGHT_WORST_CASE_MS + LIVE_CHALLENGE_SEALING_MARGIN_MS
)
if LIVE_CHALLENGE_TTL_MS < LIVE_CHALLENGE_MIN_BUDGET_MS:  # pragma: no cover - import-time invariant
    raise RuntimeError("live challenge TTL is below the frozen worst-case execution budget")
LIVE_CHALLENGE_PURPOSE: Final[str] = "BIND_A0_ACTUAL_EVIDENCE_THEN_RUN_LIVE_READONLY_SHADOW"
OVERHEAD_TASK_KIND: Final[str] = "BAG_DROP_IN_GRINDING_DISH"
OVERHEAD_RESULT_SCHEMA: Final[str] = "xrd-overhead-bag-presence-v2"
OVERHEAD_SUCCESS_STATE: Final[str] = "BAG_PRESENT"
LOCAL_SSH_EXECUTABLE: Final[str] = "/usr/bin/ssh"
REMOTE_PYTHON_EXECUTABLE: Final[str] = "/usr/bin/python3"
A0_RUN_SCOPE_SENTINEL: Final[str] = "_A0_RUN_SCOPE_"
_A0_RUN_SCOPE_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}\Z")
LIVE_BOOTSTRAP_SNAPSHOT_FILENAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "ai_x5": "ai_snapshot.json",
        "embodied_x5": "embodied_snapshot.json",
        "arm01": "arm01_snapshot.json",
        "arm02": "arm02_snapshot.json",
    }
)
FROZEN_KNOWN_HOSTS_SHA256: Final[str] = "79fc15d37314f1abeae2b07952695f666c993272453fc582b6e571e42dd4212f"
_FROZEN_ED25519_HOST_KEYS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "192.0.2.85": ("AAAAC3NzaC1lZDI1NTE5AAAAIGLP6Y7m1JwCcp/oD0Fc2g79siTotiqTYV5a3iYUiEmG"),
        "192.0.2.103": ("AAAAC3NzaC1lZDI1NTE5AAAAIGLP6Y7m1JwCcp/oD0Fc2g79siTotiqTYV5a3iYUiEmG"),
        "192.0.2.64": ("AAAAC3NzaC1lZDI1NTE5AAAAIPQ9WXswNkKhmqraYV3zGPry9Rmfuz5VkG449pBrnf1Z"),
        "192.0.2.136": ("AAAAC3NzaC1lZDI1NTE5AAAAIOBUvdLMdBvNFuL27apb/dpvV/fp+jbks7zk7IX8BYoM"),
    }
)

_CONFIG_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "release_id",
        "known_hosts_file",
        "known_hosts_sha256",
        "bootstrap_manifest_file",
        "targets",
        "expected",
        "overhead",
        "config_sha256",
    }
)
_TARGET_KEYS: Final[frozenset[str]] = frozenset({"ai_x5", "embodied_x5", "arm01", "arm02"})
_AI_TARGET_KEYS: Final[frozenset[str]] = frozenset({"base_url"})
_SSH_TARGET_KEYS: Final[frozenset[str]] = frozenset({"host", "user", "host_key_alias", "remote_script"})
_EXPECTED_KEYS: Final[frozenset[str]] = frozenset(
    {
        "ai_device_id",
        "ai_boot_id",
        "ai_runtime_artifact_set_sha256",
        "embodied_device_id",
        "embodied_boot_id",
        "embodied_required_artifact_sha256",
        "arm_machine_id_sha256",
        "arm_boot_id",
        "arm_probe_script_sha256",
        "bootstrap_evidence",
    }
)
_OVERHEAD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "record",
        "acquisition_manifest",
        "raw_frame",
        "frame_bundle_artifact",
        "result_json",
        "camera_service_identity_artifact",
        "capture_pipeline_artifact",
        "inference_pipeline_artifact",
        "replay_ledger_dir",
        "task_kind",
        "result_schema",
        "success_state",
    }
)
_A0_OUTPUT_BASENAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "record": "overhead_a0_record.json",
        "acquisition_manifest": "overhead_a0_acquisition.json",
        "raw_frame": "overhead_a0_frame.jpg",
        "frame_bundle_artifact": "overhead_a0_input_frames.json",
        "result_json": "overhead_a0_result.json",
        "camera_service_identity_artifact": "camera_service_identity.json",
        "capture_pipeline_artifact": "overhead_camera_service.py",
        "inference_pipeline_artifact": "overhead_bag_presence_x5.py",
        "replay_ledger_dir": "replay_ledger",
    }
)
_ARM_IDS: Final[tuple[str, str]] = ("arm01", "arm02")
_BOOTSTRAP_TARGET_IDS: Final[tuple[str, str, str, str]] = (
    "ai_x5",
    "embodied_x5",
    "arm01",
    "arm02",
)
_BOOTSTRAP_EVIDENCE_KEYS: Final[frozenset[str]] = frozenset(
    {"schema_version", "run_id", "run_nonce_sha256", "snapshot_sha256", "observed_at_ms"}
)
_BOOTSTRAP_MANIFEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "status",
        "generated_at_ms",
        "run_id",
        "run_nonce_sha256",
        "release_id",
        "template_sha256",
        "known_hosts_sha256",
        "profile_sha256",
        "fixed_targets",
        "parallel_process_count",
        "capture_deadline_s",
        "command_sha256",
        "snapshot_sha256",
        "file_sha256",
        "config_sha256",
        "ai_runtime_artifact_set_sha256",
        "bootstrap_evidence",
        "ai_secondary_identity",
        "embodied_secondary_identity",
        "x5_host_key_distinguishes_devices",
        "x5_shared_host_key_observed",
        "remote_contacted",
        "network_touched",
        "pc_network_mutated",
        "proxy_or_jump_used",
        "services_mutated",
        "model_or_inference_triggered",
        "hardware_device_opened_by_capture",
        "actuator_commands_issued",
        "execution_authority",
        "config_uploaded",
        "manifest_sha256",
    }
)
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}$")
_AI_DEVICE_ID_RE: Final[re.Pattern[str]] = re.compile(r"^machine-sha256:[0-9a-f]{64}$")
_EMBODIED_DEVICE_ID_RE: Final[re.Pattern[str]] = re.compile(r"^embodied-x5:[0-9a-f]{32}$")
_BOOT_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_PLACEHOLDER_MARKERS: Final[tuple[str, ...]] = (
    "placeholder",
    "replace_with",
    "changeme",
    "change_me",
    "todo",
    "your_",
    "<required",
    "${",
)
_SENSITIVE_CONFIG_KEYS: Final[frozenset[str]] = frozenset(
    {"password", "passwd", "api_key", "secret", "private_key", "identity_file"}
)
_CHALLENGE_KEYS: Final[frozenset[str]] = frozenset(
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
_CHALLENGE_ISSUANCE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "challenge_sha256",
        "challenge_artifact_sha256",
        "config_sha256",
        "bootstrap_manifest_sha256",
        "bootstrap_manifest_file_sha256",
        "issued_at_ms",
    }
)
_CHALLENGE_CONSUMPTION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "challenge_sha256",
        "challenge_artifact_sha256",
        "config_sha256",
        "bootstrap_manifest_sha256",
        "bootstrap_manifest_file_sha256",
        "consumed_at_ms",
    }
)
_A0_RESERVATION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "challenge_sha256",
        "challenge_artifact_sha256",
        "run_dir",
        "run_binding_sha256",
        "reserved_at_ms",
        "reservation_nonce",
        "reservation_sha256",
    }
)
_A0_PRODUCED_KEYS: Final[frozenset[str]] = frozenset(
    {
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
)
_CHALLENGE_PROFILE_KEYS: Final[frozenset[str]] = frozenset(
    {"ai_x5", "assay_station", "dual_arm", "embodied_x5"}
)
_REMOTE_SHA256_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"^([0-9a-f]{64})  (/home/rdk/-]+\.py)\n$"
)
_SSH_REMOTE_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_.:@/+\-]+$")
_VERIFIED_COLLECTOR_BOOTSTRAP: Final[str] = (
    "import hashlib,os,secrets,sys\n"
    "expected,path=sys.argv[1],sys.argv[2]\n"
    "with open(path,'rb') as handle:\n"
    " source=handle.read()\n"
    "if not secrets.compare_digest(hashlib.sha256(source).hexdigest(),expected):\n"
    " raise SystemExit(73)\n"
    "sys.argv=[path,*sys.argv[3:]]\n"
    "sys.path.insert(0,os.path.dirname(path))\n"
    "scope={'__name__':'__main__','__file__':path,'__package__':None,"
    "'__cached__':None,'__spec__':None}\n"
    "exec(compile(source,path,'exec'),scope,scope)\n"
)

_FIXED_TARGETS: Final[Mapping[str, Mapping[str, str]]] = MappingProxyType(
    {
        "ai_x5": MappingProxyType({"base_url": "http://127.0.0.1:8888"}),
        "embodied_x5": MappingProxyType(
            {
                "host": "192.0.2.85",
                "user": "sunrise",
                "host_key_alias": "192.0.2.85",
                "remote_script": "/home/rdk/tools/rb_voe_embodied_snapshot.py",
            }
        ),
        "arm01": MappingProxyType(
            {
                "host": "192.0.2.64",
                "user": "er",
                "host_key_alias": "192.0.2.64",
                "remote_script": "/home/rdk/dual_arm/rb_voe_arm_readonly_probe.py",
            }
        ),
        "arm02": MappingProxyType(
            {
                "host": "192.0.2.136",
                "user": "er",
                "host_key_alias": "192.0.2.136",
                "remote_script": "/home/rdk/dual_arm/rb_voe_arm_readonly_probe.py",
            }
        ),
    }
)


class LiveShadowConfigError(ValueError):
    """Raised when a live-shadow configuration is not exact and content-bound."""


@dataclass(frozen=True, slots=True)
class LiveShadowConfig:
    release_id: str
    source_path: Path
    known_hosts_file: Path
    known_hosts_sha256: str
    bootstrap_manifest_file: Path
    targets: Mapping[str, Mapping[str, str]]
    ai_device_id: str
    ai_boot_id: str
    ai_runtime_artifact_set_sha256: str
    embodied_device_id: str
    embodied_boot_id: str
    embodied_required_artifact_sha256: Mapping[str, str]
    arm_machine_id_sha256: Mapping[str, str]
    arm_boot_id: Mapping[str, str]
    arm_probe_script_sha256: str
    bootstrap_evidence: Mapping[str, Any]
    overhead_record: Path
    overhead_acquisition_manifest: Path
    overhead_raw_frame: Path
    overhead_frame_bundle_artifact: Path
    overhead_result_json: Path
    overhead_camera_service_identity_artifact: Path
    overhead_capture_pipeline_artifact: Path
    overhead_inference_pipeline_artifact: Path
    overhead_replay_ledger_dir: Path
    overhead_task_kind: str
    overhead_result_schema: str
    overhead_success_state: str
    config_sha256: str
    schema_version: str = LIVE_SHADOW_CONFIG_SCHEMA


@dataclass(frozen=True, slots=True)
class LiveShadowRunResult:
    report: ShadowRunReport
    binding: ShadowRunBinding
    output_dir: Path
    evidence_root_sha256: str
    transport_commands_issued: int
    read_only_device_observations: int


@dataclass(frozen=True, slots=True)
class _ValidatedBootstrapBundle:
    manifest: Mapping[str, Any]
    raw_files: Mapping[str, bytes]
    manifest_sha256: str
    manifest_file_sha256: str


@dataclass(frozen=True, slots=True)
class _ValidatedChallenge:
    artifact: Mapping[str, Any]
    artifact_sha256: str
    path: Path
    a0_config: LiveShadowConfig
    a0_record: Mapping[str, Any]
    a0_produced: Mapping[str, Any]

    @property
    def challenge_sha256(self) -> str:
        return str(self.artifact["challenge_sha256"])

    @property
    def challenge_content_sha256(self) -> str:
        return str(self.artifact["challenge_content_sha256"])


@dataclass(frozen=True, slots=True)
class _CollectorPreflightResult:
    target_id: str
    host: str
    remote_script: str
    expected_sha256: str
    observed_sha256: str
    transport_command_issued: bool = True
    read_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "host": self.host,
            "remote_script": self.remote_script,
            "expected_sha256": self.expected_sha256,
            "observed_sha256": self.observed_sha256,
            "matched": secrets.compare_digest(self.expected_sha256, self.observed_sha256),
            "transport_command_issued": self.transport_command_issued,
            "read_only": self.read_only,
            "collector_executed_by_preflight": False,
        }


@dataclass(frozen=True, slots=True)
class _CollectorPreflightAttempt:
    target_id: str
    host: str
    remote_script: str
    expected_sha256: str
    outcome: str
    return_code: int | None = None
    observed_sha256: str | None = None
    stdout_sha256: str | None = None
    stdout_size_bytes: int | None = None
    stderr_sha256: str | None = None
    stderr_size_bytes: int | None = None
    error_type: str | None = None
    transport_command_issued: bool = True
    read_only: bool = True
    collector_executed_by_preflight: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "host": self.host,
            "remote_script": self.remote_script,
            "expected_sha256": self.expected_sha256,
            "observed_sha256": self.observed_sha256,
            "outcome": self.outcome,
            "return_code": self.return_code,
            "stdout_sha256": self.stdout_sha256,
            "stdout_size_bytes": self.stdout_size_bytes,
            "stderr_sha256": self.stderr_sha256,
            "stderr_size_bytes": self.stderr_size_bytes,
            "error_type": self.error_type,
            "transport_command_issued": self.transport_command_issued,
            "read_only": self.read_only,
            "collector_executed_by_preflight": self.collector_executed_by_preflight,
        }


@dataclass(slots=True)
class _LiveFailureState:
    failure_phase: str = "BEFORE_OUTPUT_RESERVATION"
    output: Path | None = None
    challenge: _ValidatedChallenge | None = None
    run_known_hosts_file: Path | None = None
    preflight_attempts: list[_CollectorPreflightAttempt] = field(default_factory=list)
    snapshot_transport_attempts: int = 0


@dataclass(frozen=True, slots=True)
class _LiveSourceSeed:
    """Run identity available before the post-capture evaluation clock is sampled."""

    run_id: str
    run_nonce: str
    release_id: str
    profile_sha256: Mapping[str, str]

    def source_binding_sha256(self, subsystem: str) -> str:
        return canonical_sha256(
            {
                "schema_version": "xrd-rb-voe-live-source-binding-v1",
                "subsystem": subsystem,
                "run_id": self.run_id,
                "run_nonce": self.run_nonce,
                "release_id": self.release_id,
                "profile_sha256": self.profile_sha256[subsystem],
            }
        )


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LiveShadowConfigError(f"duplicate config field: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise LiveShadowConfigError(f"non-finite config value is forbidden: {value}")


def _exact_keys(value: Any, expected: frozenset[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LiveShadowConfigError(f"{label} must be an object")
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise LiveShadowConfigError(
            f"{label} fields differ from the schema; missing={missing}, extra={extra}"
        )
    return value


def _reject_sensitive_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() in _SENSITIVE_CONFIG_KEYS:
                raise LiveShadowConfigError("secret-bearing config fields are forbidden")
            _reject_sensitive_keys(item)
    elif isinstance(value, list):
        for item in value:
            _reject_sensitive_keys(item)


def _is_placeholder(value: str) -> bool:
    lowered = value.casefold()
    if any(marker in lowered for marker in _PLACEHOLDER_MARKERS):
        return True
    candidate = value.removeprefix("machine-sha256:")
    return len(candidate) == 64 and len(set(candidate)) == 1


def _reject_placeholders(value: Any) -> None:
    if isinstance(value, str) and _is_placeholder(value):
        raise LiveShadowConfigError("placeholder config values are forbidden")
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_placeholders(item)
    elif isinstance(value, list):
        for item in value:
            _reject_placeholders(item)


def _string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise LiveShadowConfigError(f"{label} must be a non-empty string")
    return value


def _is_reparse_point(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    return path.is_symlink() or attributes & reparse_flag != 0


def _is_regular_non_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(metadata.st_mode) and not _is_reparse_point(path)


def _regular_absolute_file(
    value: Any,
    *,
    label: str,
    suffixes: tuple[str, ...] = (),
) -> Path:
    raw = _string(value, label=label)
    path = Path(raw)
    if not path.is_absolute() or ".." in path.parts:
        raise LiveShadowConfigError(f"{label} must be an absolute lexical path")
    if suffixes and path.suffix.casefold() not in suffixes:
        raise LiveShadowConfigError(f"{label} has an unregistered file type")
    if not _is_regular_non_reparse(path):
        raise LiveShadowConfigError(f"{label} must be an existing regular non-link file")
    for parent in path.parents:
        if parent.exists() and _is_reparse_point(parent):
            raise LiveShadowConfigError(f"{label} cannot traverse a link")
    return path


def compute_config_sha256(payload: Mapping[str, Any]) -> str:
    """Return the canonical digest over every config field except the digest itself."""

    unsigned = dict(payload)
    unsigned.pop("config_sha256", None)
    return canonical_sha256(unsigned)


def _validate_frozen_known_hosts_bytes(raw: bytes) -> None:
    if hashlib.sha256(raw).hexdigest() != FROZEN_KNOWN_HOSTS_SHA256:
        raise LiveShadowConfigError("known_hosts differs from the compiled frozen trust root")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise LiveShadowConfigError("known_hosts must be ASCII") from exc
    entries: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) != 3 or parts[1] != "ssh-ed25519":
            raise LiveShadowConfigError("known_hosts contains a non-frozen host-key record")
        host, _, key = parts
        if host in entries:
            raise LiveShadowConfigError("known_hosts contains a duplicate host record")
        entries[host] = key
    if entries != dict(_FROZEN_ED25519_HOST_KEYS):
        raise LiveShadowConfigError("known_hosts fixed host entries differ from the release")


def _read_stable_regular_bytes(
    path: Path,
    *,
    label: str,
    max_bytes: int,
) -> bytes:
    if (
        not path.is_absolute()
        or not _is_regular_non_reparse(path)
        or any(parent.exists() and _is_reparse_point(parent) for parent in path.parents)
    ):
        raise LiveShadowConfigError(f"{label} must be an existing absolute regular non-reparse file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise LiveShadowConfigError(f"cannot open {label}: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_size > max_bytes:
            raise LiveShadowConfigError(f"{label} size or file type is invalid")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if not raw or len(raw) > max_bytes:
        raise LiveShadowConfigError(f"{label} size is invalid")
    descriptor_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    if descriptor_identity != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise LiveShadowConfigError(f"{label} changed during descriptor read")
    try:
        current = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise LiveShadowConfigError(f"{label} path changed after descriptor read") from exc
    if descriptor_identity != (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
    ):
        raise LiveShadowConfigError(f"{label} path identity changed during read")
    return raw


def _read_frozen_known_hosts(path: Path) -> bytes:
    raw = _read_stable_regular_bytes(
        path,
        label="known_hosts",
        max_bytes=64 * 1024,
    )
    _validate_frozen_known_hosts_bytes(raw)
    return raw


def _read_config_payload(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    raw = _read_stable_regular_bytes(
        config_path,
        label="config",
        max_bytes=1024 * 1024,
    )
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except LiveShadowConfigError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LiveShadowConfigError("config must be strict UTF-8 JSON") from exc
    root = _exact_keys(payload, _CONFIG_KEYS, label="config")
    _reject_sensitive_keys(root)
    if root["schema_version"] != LIVE_SHADOW_CONFIG_SCHEMA:
        raise LiveShadowConfigError("unsupported live-shadow config schema")
    claimed_digest = _string(root["config_sha256"], label="config_sha256")
    require_sha256("config_sha256", claimed_digest)
    if claimed_digest != compute_config_sha256(root):
        raise LiveShadowConfigError("config_sha256 does not bind the canonical config")
    if root["known_hosts_sha256"] != FROZEN_KNOWN_HOSTS_SHA256:
        raise LiveShadowConfigError("known_hosts_sha256 differs from the compiled frozen trust root")
    return root


def _lexical_absolute_path(value: Any, *, label: str, suffixes: tuple[str, ...] = ()) -> str:
    raw = _string(value, label=label)
    if not PurePosixPath(raw).is_absolute() and not Path(raw).is_absolute():
        raise LiveShadowConfigError(f"{label} must be an absolute lexical path")
    if ".." in PurePosixPath(raw).parts or ".." in Path(raw).parts:
        raise LiveShadowConfigError(f"{label} cannot contain parent traversal")
    suffix = PurePosixPath(raw).suffix.casefold() or Path(raw).suffix.casefold()
    if suffixes and suffix not in suffixes:
        raise LiveShadowConfigError(f"{label} has an unregistered file type")
    return raw


def _regular_absolute_directory(value: Any, *, label: str) -> Path:
    raw = _lexical_absolute_path(value, label=label)
    path = Path(raw)
    if path.is_symlink() or not path.is_dir():
        raise LiveShadowConfigError(f"{label} must be an existing regular non-link directory")
    for parent in path.parents:
        if parent.exists() and parent.is_symlink():
            raise LiveShadowConfigError(f"{label} cannot traverse a link")
    return path


def _placeholder_or_digest(value: Any, *, label: str, unresolved: list[str]) -> None:
    text = _string(value, label=label)
    if _is_placeholder(text):
        unresolved.append(label)
        return
    require_sha256(label, text)


def _placeholder_or_boot_id(value: Any, *, label: str, unresolved: list[str]) -> None:
    text = _string(value, label=label)
    if _is_placeholder(text):
        unresolved.append(label)
        return
    if _BOOT_ID_RE.fullmatch(text) is None:
        raise LiveShadowConfigError(f"{label} must be a canonical boot UUID")


def _inspect_bootstrap_evidence(value: Any, *, unresolved: list[str]) -> None:
    evidence = _exact_keys(value, _BOOTSTRAP_EVIDENCE_KEYS, label="expected.bootstrap_evidence")
    if evidence["schema_version"] != LIVE_BOOTSTRAP_BINDING_SCHEMA:
        raise LiveShadowConfigError("expected.bootstrap_evidence schema is not frozen")
    run_id = _string(evidence["run_id"], label="expected.bootstrap_evidence.run_id")
    if _is_placeholder(run_id):
        unresolved.append("expected.bootstrap_evidence.run_id")
    elif _TOKEN_RE.fullmatch(run_id) is None:
        raise LiveShadowConfigError("expected.bootstrap_evidence.run_id is not a contract token")
    _placeholder_or_digest(
        evidence["run_nonce_sha256"],
        label="expected.bootstrap_evidence.run_nonce_sha256",
        unresolved=unresolved,
    )
    snapshot_sha256 = _exact_keys(
        evidence["snapshot_sha256"],
        frozenset(_BOOTSTRAP_TARGET_IDS),
        label="expected.bootstrap_evidence.snapshot_sha256",
    )
    for target_id, digest in snapshot_sha256.items():
        _placeholder_or_digest(
            digest,
            label=f"expected.bootstrap_evidence.snapshot_sha256.{target_id}",
            unresolved=unresolved,
        )
    observed = _exact_keys(
        evidence["observed_at_ms"],
        frozenset(_BOOTSTRAP_TARGET_IDS),
        label="expected.bootstrap_evidence.observed_at_ms",
    )
    completed_times: list[int] = []
    for target_id, item in observed.items():
        label = f"expected.bootstrap_evidence.observed_at_ms.{target_id}"
        if isinstance(item, str) and _is_placeholder(item):
            unresolved.append(label)
        elif isinstance(item, bool) or not isinstance(item, int) or item < 1_700_000_000_000:
            raise LiveShadowConfigError(f"{label} must be a current Unix timestamp in milliseconds")
        else:
            completed_times.append(item)
    if len(completed_times) == len(_BOOTSTRAP_TARGET_IDS) and (
        max(completed_times) - min(completed_times) > LIVE_SNAPSHOT_MAX_AGE_MS
    ):
        raise LiveShadowConfigError("expected.bootstrap_evidence observation window is not coherent")


def _load_bootstrap_evidence(value: Any) -> Mapping[str, Any]:
    unresolved: list[str] = []
    _inspect_bootstrap_evidence(value, unresolved=unresolved)
    if unresolved:
        raise LiveShadowConfigError("expected.bootstrap_evidence still contains placeholders")
    evidence = value
    return MappingProxyType(
        {
            "schema_version": evidence["schema_version"],
            "run_id": evidence["run_id"],
            "run_nonce_sha256": evidence["run_nonce_sha256"],
            "snapshot_sha256": MappingProxyType(dict(evidence["snapshot_sha256"])),
            "observed_at_ms": MappingProxyType(dict(evidence["observed_at_ms"])),
        }
    )


def _overhead_task_contract(overhead: Mapping[str, Any]) -> tuple[str, str, str]:
    contract = (
        _string(overhead["task_kind"], label="overhead.task_kind"),
        _string(overhead["result_schema"], label="overhead.result_schema"),
        _string(overhead["success_state"], label="overhead.success_state"),
    )
    expected = (OVERHEAD_TASK_KIND, OVERHEAD_RESULT_SCHEMA, OVERHEAD_SUCCESS_STATE)
    if contract != expected:
        raise LiveShadowConfigError("overhead task/result tuple differs from the frozen bag-drop contract")
    return contract


def _portable_config_path(value: str) -> PurePosixPath | Path:
    posix = PurePosixPath(value)
    return posix if posix.is_absolute() else Path(value)


def _validate_a0_run_scope_template(paths: Mapping[str, str]) -> str:
    if set(paths) != set(_A0_OUTPUT_BASENAMES):
        raise LiveShadowConfigError("A0 output path template keys are not exact")
    parsed = {field: _portable_config_path(value) for field, value in paths.items()}
    roots = {str(path.parent) for path in parsed.values()}
    if len(roots) != 1:
        raise LiveShadowConfigError("A0 output paths do not share one run-scope template root")
    root = next(iter(roots))
    if _portable_config_path(root).name != A0_RUN_SCOPE_SENTINEL:
        raise LiveShadowConfigError("A0 output paths lack the frozen run-scope sentinel")
    if any(parsed[field].name != name for field, name in _A0_OUTPUT_BASENAMES.items()):
        raise LiveShadowConfigError("A0 output artifact names differ from the frozen contract")
    return root


def inspect_live_shadow_config(path: str | Path) -> dict[str, Any]:
    """Validate an incomplete deployment template without touching a transport or target."""

    root = _read_config_payload(path)
    release_id = _string(root["release_id"], label="release_id")
    if _TOKEN_RE.fullmatch(release_id) is None:
        raise LiveShadowConfigError("release_id is not a contract token")
    targets = _exact_keys(root["targets"], _TARGET_KEYS, label="targets")
    normalized_targets: dict[str, dict[str, str]] = {}
    for target_id in _TARGET_KEYS:
        expected_keys = _AI_TARGET_KEYS if target_id == "ai_x5" else _SSH_TARGET_KEYS
        target = _exact_keys(targets[target_id], expected_keys, label=f"targets.{target_id}")
        normalized = {key: _string(item, label=f"targets.{target_id}.{key}") for key, item in target.items()}
        if normalized != dict(_FIXED_TARGETS[target_id]):
            raise LiveShadowConfigError(f"targets.{target_id} differs from the fixed topology")
        normalized_targets[target_id] = normalized

    unresolved: list[str] = []
    expected = _exact_keys(root["expected"], _EXPECTED_KEYS, label="expected")
    identity_patterns = {
        "ai_device_id": _AI_DEVICE_ID_RE,
        "embodied_device_id": _EMBODIED_DEVICE_ID_RE,
    }
    for label, pattern in identity_patterns.items():
        value = _string(expected[label], label=f"expected.{label}")
        if _is_placeholder(value):
            unresolved.append(f"expected.{label}")
        elif pattern.fullmatch(value) is None:
            raise LiveShadowConfigError(f"expected.{label} has an invalid frozen identity format")
    _placeholder_or_boot_id(expected["ai_boot_id"], label="expected.ai_boot_id", unresolved=unresolved)
    _placeholder_or_digest(
        expected["ai_runtime_artifact_set_sha256"],
        label="expected.ai_runtime_artifact_set_sha256",
        unresolved=unresolved,
    )
    _placeholder_or_boot_id(
        expected["embodied_boot_id"],
        label="expected.embodied_boot_id",
        unresolved=unresolved,
    )
    artifacts_value = expected["embodied_required_artifact_sha256"]
    if not isinstance(artifacts_value, dict):
        raise LiveShadowConfigError("expected.embodied_required_artifact_sha256 must be an object")
    extra_artifacts = set(artifacts_value) - set(EMBODIED_X5_REQUIRED_ARTIFACTS)
    if extra_artifacts:
        raise LiveShadowConfigError(
            f"expected.embodied_required_artifact_sha256 has unregistered fields: {sorted(extra_artifacts)}"
        )
    for name in sorted(set(EMBODIED_X5_REQUIRED_ARTIFACTS) - set(artifacts_value)):
        unresolved.append(f"expected.embodied_required_artifact_sha256.{name}")
    for name, digest in artifacts_value.items():
        _placeholder_or_digest(
            digest,
            label=f"expected.embodied_required_artifact_sha256.{name}",
            unresolved=unresolved,
        )
    arms = _exact_keys(
        expected["arm_machine_id_sha256"],
        frozenset(_ARM_IDS),
        label="expected.arm_machine_id_sha256",
    )
    for arm_id, digest in arms.items():
        _placeholder_or_digest(
            digest,
            label=f"expected.arm_machine_id_sha256.{arm_id}",
            unresolved=unresolved,
        )
    arm_boot_ids = _exact_keys(
        expected["arm_boot_id"],
        frozenset(_ARM_IDS),
        label="expected.arm_boot_id",
    )
    for arm_id, boot_id in arm_boot_ids.items():
        _placeholder_or_boot_id(
            boot_id,
            label=f"expected.arm_boot_id.{arm_id}",
            unresolved=unresolved,
        )
    _placeholder_or_digest(
        expected["arm_probe_script_sha256"],
        label="expected.arm_probe_script_sha256",
        unresolved=unresolved,
    )
    _inspect_bootstrap_evidence(expected["bootstrap_evidence"], unresolved=unresolved)

    _lexical_absolute_path(root["known_hosts_file"], label="known_hosts_file")
    _lexical_absolute_path(
        root["bootstrap_manifest_file"],
        label="bootstrap_manifest_file",
        suffixes=(".json",),
    )
    _placeholder_or_digest(root["known_hosts_sha256"], label="known_hosts_sha256", unresolved=unresolved)
    overhead = _exact_keys(root["overhead"], _OVERHEAD_KEYS, label="overhead")
    _overhead_task_contract(overhead)
    overhead_paths = {
        "record": _lexical_absolute_path(overhead["record"], label="overhead.record", suffixes=(".json",)),
        "acquisition_manifest": _lexical_absolute_path(
            overhead["acquisition_manifest"],
            label="overhead.acquisition_manifest",
            suffixes=(".json",),
        ),
        "raw_frame": _lexical_absolute_path(
            overhead["raw_frame"],
            label="overhead.raw_frame",
            suffixes=(".jpg", ".jpeg", ".png"),
        ),
        "frame_bundle_artifact": _lexical_absolute_path(
            overhead["frame_bundle_artifact"],
            label="overhead.frame_bundle_artifact",
            suffixes=(".json",),
        ),
        "result_json": _lexical_absolute_path(
            overhead["result_json"], label="overhead.result_json", suffixes=(".json",)
        ),
        "camera_service_identity_artifact": _lexical_absolute_path(
            overhead["camera_service_identity_artifact"],
            label="overhead.camera_service_identity_artifact",
            suffixes=(".json",),
        ),
        "capture_pipeline_artifact": _lexical_absolute_path(
            overhead["capture_pipeline_artifact"],
            label="overhead.capture_pipeline_artifact",
            suffixes=(".py",),
        ),
        "inference_pipeline_artifact": _lexical_absolute_path(
            overhead["inference_pipeline_artifact"],
            label="overhead.inference_pipeline_artifact",
            suffixes=(".py",),
        ),
        "replay_ledger_dir": _lexical_absolute_path(
            overhead["replay_ledger_dir"], label="overhead.replay_ledger_dir"
        ),
    }
    if len(set(overhead_paths.values())) != len(overhead_paths):
        raise LiveShadowConfigError("overhead evidence paths must be distinct")
    _validate_a0_run_scope_template(overhead_paths)
    return {
        "schema_version": LIVE_SHADOW_PLAN_SCHEMA,
        "mode": "PLAN_ONLY",
        "config_sha256": root["config_sha256"],
        "release_id": release_id,
        "live_ready": not unresolved,
        "unresolved_fields": sorted(unresolved),
        "fixed_targets": {
            target_id: normalized_targets[target_id] for target_id in sorted(normalized_targets)
        },
        "semantic_profiles": [
            "ai_x5.v1",
            "assay_station.v1",
            "dual_arm.v1",
            "embodied_x5.v1",
        ],
        "assay_station": "TARGET_ONLY",
        "expected_first_status": "HOLD",
        "expected_hold_reason": "assay_station remains TARGET_ONLY",
        "a0_output_scope": {
            "template_component": A0_RUN_SCOPE_SENTINEL,
            "run_dir_relation": "SAFE_DIRECT_CHILD",
            "single_challenge_binding": True,
        },
        "challenge_ttl_ms": LIVE_CHALLENGE_TTL_MS,
        "challenge_min_budget_ms": LIVE_CHALLENGE_MIN_BUDGET_MS,
        "remote_contacted": False,
        "network_touched": False,
        "execution_authority": False,
        "transport_commands_issued": 0,
        "read_only_device_observations": 0,
        "actuator_commands_issued": 0,
        "mutating_commands_issued": 0,
        "read_only_transport_operations": 0,
        "physical_closure_proven": False,
        "physical_risk_denominator_increment": 0,
    }


def load_live_shadow_config(path: str | Path) -> LiveShadowConfig:
    """Load and strictly validate one local, content-addressed live config."""

    root = _read_config_payload(path)
    claimed_digest = str(root["config_sha256"])
    _reject_placeholders(root)

    release_id = _string(root["release_id"], label="release_id")
    if _TOKEN_RE.fullmatch(release_id) is None:
        raise LiveShadowConfigError("release_id is not a contract token")

    target_root = _exact_keys(root["targets"], _TARGET_KEYS, label="targets")
    frozen_targets: dict[str, Mapping[str, str]] = {}
    for target_id in _TARGET_KEYS:
        expected_keys = _AI_TARGET_KEYS if target_id == "ai_x5" else _SSH_TARGET_KEYS
        target = _exact_keys(target_root[target_id], expected_keys, label=f"targets.{target_id}")
        normalized = {key: _string(item, label=f"targets.{target_id}.{key}") for key, item in target.items()}
        if normalized != dict(_FIXED_TARGETS[target_id]):
            raise LiveShadowConfigError(f"targets.{target_id} differs from the fixed topology")
        frozen_targets[target_id] = MappingProxyType(normalized)

    expected = _exact_keys(root["expected"], _EXPECTED_KEYS, label="expected")
    ai_device_id = _string(expected["ai_device_id"], label="expected.ai_device_id")
    embodied_device_id = _string(expected["embodied_device_id"], label="expected.embodied_device_id")
    if (
        _AI_DEVICE_ID_RE.fullmatch(ai_device_id) is None
        or _EMBODIED_DEVICE_ID_RE.fullmatch(embodied_device_id) is None
    ):
        raise LiveShadowConfigError("expected device IDs differ from their frozen identity formats")
    ai_boot_id = _string(expected["ai_boot_id"], label="expected.ai_boot_id")
    if _BOOT_ID_RE.fullmatch(ai_boot_id) is None:
        raise LiveShadowConfigError("expected.ai_boot_id must be a canonical boot UUID")
    ai_runtime_artifact_set_sha256 = _string(
        expected["ai_runtime_artifact_set_sha256"],
        label="expected.ai_runtime_artifact_set_sha256",
    )
    require_sha256(
        "expected.ai_runtime_artifact_set_sha256",
        ai_runtime_artifact_set_sha256,
    )
    embodied_boot_id = _string(expected["embodied_boot_id"], label="expected.embodied_boot_id")
    if _BOOT_ID_RE.fullmatch(embodied_boot_id) is None:
        raise LiveShadowConfigError("expected.embodied_boot_id must be a canonical boot UUID")

    artifact_hashes = _exact_keys(
        expected["embodied_required_artifact_sha256"],
        frozenset(EMBODIED_X5_REQUIRED_ARTIFACTS),
        label="expected.embodied_required_artifact_sha256",
    )
    for name, digest in artifact_hashes.items():
        require_sha256(f"expected.embodied_required_artifact_sha256.{name}", digest)
    arm_hashes = _exact_keys(
        expected["arm_machine_id_sha256"], frozenset(_ARM_IDS), label="expected.arm_machine_id_sha256"
    )
    for arm_id, digest in arm_hashes.items():
        require_sha256(f"expected.arm_machine_id_sha256.{arm_id}", digest)
    arm_boot_ids = _exact_keys(expected["arm_boot_id"], frozenset(_ARM_IDS), label="expected.arm_boot_id")
    for arm_id, boot_id in arm_boot_ids.items():
        if _BOOT_ID_RE.fullmatch(_string(boot_id, label=f"expected.arm_boot_id.{arm_id}")) is None:
            raise LiveShadowConfigError(f"expected.arm_boot_id.{arm_id} must be a canonical boot UUID")
    if len(set(arm_boot_ids.values())) != len(_ARM_IDS):
        raise LiveShadowConfigError("arm01 and arm02 expected boot identities must be distinct")
    probe_hash = _string(expected["arm_probe_script_sha256"], label="expected.arm_probe_script_sha256")
    require_sha256("expected.arm_probe_script_sha256", probe_hash)
    bootstrap_evidence = _load_bootstrap_evidence(expected["bootstrap_evidence"])

    known_hosts = _regular_absolute_file(root["known_hosts_file"], label="known_hosts_file")
    known_hosts_sha256 = _string(root["known_hosts_sha256"], label="known_hosts_sha256")
    require_sha256("known_hosts_sha256", known_hosts_sha256)
    if known_hosts_sha256 != FROZEN_KNOWN_HOSTS_SHA256:
        raise LiveShadowConfigError("known_hosts digest differs from the compiled trust root")
    _read_frozen_known_hosts(known_hosts)
    bootstrap_manifest_file = _regular_absolute_file(
        root["bootstrap_manifest_file"],
        label="bootstrap_manifest_file",
        suffixes=(".json",),
    )
    overhead = _exact_keys(root["overhead"], _OVERHEAD_KEYS, label="overhead")
    overhead_task_kind, overhead_result_schema, overhead_success_state = _overhead_task_contract(overhead)
    overhead_path_values = {
        "record": _lexical_absolute_path(overhead["record"], label="overhead.record", suffixes=(".json",)),
        "acquisition_manifest": _lexical_absolute_path(
            overhead["acquisition_manifest"],
            label="overhead.acquisition_manifest",
            suffixes=(".json",),
        ),
        "raw_frame": _lexical_absolute_path(
            overhead["raw_frame"],
            label="overhead.raw_frame",
            suffixes=(".jpg", ".jpeg", ".png"),
        ),
        "frame_bundle_artifact": _lexical_absolute_path(
            overhead["frame_bundle_artifact"],
            label="overhead.frame_bundle_artifact",
            suffixes=(".json",),
        ),
        "result_json": _lexical_absolute_path(
            overhead["result_json"], label="overhead.result_json", suffixes=(".json",)
        ),
        "camera_service_identity_artifact": _lexical_absolute_path(
            overhead["camera_service_identity_artifact"],
            label="overhead.camera_service_identity_artifact",
            suffixes=(".json",),
        ),
        "capture_pipeline_artifact": _lexical_absolute_path(
            overhead["capture_pipeline_artifact"],
            label="overhead.capture_pipeline_artifact",
            suffixes=(".py",),
        ),
        "inference_pipeline_artifact": _lexical_absolute_path(
            overhead["inference_pipeline_artifact"],
            label="overhead.inference_pipeline_artifact",
            suffixes=(".py",),
        ),
        "replay_ledger_dir": _lexical_absolute_path(
            overhead["replay_ledger_dir"], label="overhead.replay_ledger_dir"
        ),
    }
    _validate_a0_run_scope_template(overhead_path_values)
    overhead_record = Path(overhead_path_values["record"])
    overhead_manifest = Path(overhead_path_values["acquisition_manifest"])
    overhead_raw = Path(overhead_path_values["raw_frame"])
    overhead_frame_bundle = Path(overhead_path_values["frame_bundle_artifact"])
    overhead_result = Path(overhead_path_values["result_json"])
    overhead_camera_service_identity = Path(overhead_path_values["camera_service_identity_artifact"])
    overhead_capture_pipeline = Path(overhead_path_values["capture_pipeline_artifact"])
    overhead_inference_pipeline = Path(overhead_path_values["inference_pipeline_artifact"])
    overhead_replay_ledger = Path(overhead_path_values["replay_ledger_dir"])
    if (
        len(
            {
                overhead_record,
                overhead_manifest,
                overhead_raw,
                overhead_frame_bundle,
                overhead_result,
                overhead_camera_service_identity,
                overhead_capture_pipeline,
                overhead_inference_pipeline,
                overhead_replay_ledger,
            }
        )
        != 9
    ):
        raise LiveShadowConfigError("overhead evidence paths must be distinct")

    _validate_bootstrap_bundle(root, config_path=Path(path))

    return LiveShadowConfig(
        release_id=release_id,
        source_path=Path(path),
        known_hosts_file=known_hosts,
        known_hosts_sha256=known_hosts_sha256,
        bootstrap_manifest_file=bootstrap_manifest_file,
        targets=MappingProxyType(frozen_targets),
        ai_device_id=ai_device_id,
        ai_boot_id=ai_boot_id,
        ai_runtime_artifact_set_sha256=ai_runtime_artifact_set_sha256,
        embodied_device_id=embodied_device_id,
        embodied_boot_id=embodied_boot_id,
        embodied_required_artifact_sha256=MappingProxyType(dict(artifact_hashes)),
        arm_machine_id_sha256=MappingProxyType(dict(arm_hashes)),
        arm_boot_id=MappingProxyType(dict(arm_boot_ids)),
        arm_probe_script_sha256=probe_hash,
        bootstrap_evidence=bootstrap_evidence,
        overhead_record=overhead_record,
        overhead_acquisition_manifest=overhead_manifest,
        overhead_raw_frame=overhead_raw,
        overhead_frame_bundle_artifact=overhead_frame_bundle,
        overhead_result_json=overhead_result,
        overhead_camera_service_identity_artifact=overhead_camera_service_identity,
        overhead_capture_pipeline_artifact=overhead_capture_pipeline,
        overhead_inference_pipeline_artifact=overhead_inference_pipeline,
        overhead_replay_ledger_dir=overhead_replay_ledger,
        overhead_task_kind=overhead_task_kind,
        overhead_result_schema=overhead_result_schema,
        overhead_success_state=overhead_success_state,
        config_sha256=claimed_digest,
    )


def plan_live_shadow(config: LiveShadowConfig) -> dict[str, Any]:
    """Describe the fixed read-only topology without constructing a transport."""

    return {
        "schema_version": LIVE_SHADOW_PLAN_SCHEMA,
        "mode": "PLAN_ONLY",
        "config_sha256": config.config_sha256,
        "release_id": config.release_id,
        "fixed_targets": {target_id: dict(config.targets[target_id]) for target_id in sorted(config.targets)},
        "semantic_profiles": ["ai_x5.v1", "assay_station.v1", "dual_arm.v1", "embodied_x5.v1"],
        "assay_station": "TARGET_ONLY",
        "a0_output_scope": {
            "template_component": A0_RUN_SCOPE_SENTINEL,
            "run_dir_relation": "SAFE_DIRECT_CHILD",
            "single_challenge_binding": True,
        },
        "challenge_ttl_ms": LIVE_CHALLENGE_TTL_MS,
        "challenge_min_budget_ms": LIVE_CHALLENGE_MIN_BUDGET_MS,
        "remote_contacted": False,
        "network_touched": False,
        "execution_authority": False,
        "transport_commands_issued": 0,
        "read_only_device_observations": 0,
        "actuator_commands_issued": 0,
        "mutating_commands_issued": 0,
        "read_only_transport_operations": 0,
        "physical_closure_proven": False,
        "physical_risk_denominator_increment": 0,
    }


class _AiPrefetchedJsonSnapshotTransport(PrefetchedJsonSnapshotTransport):
    __slots__ = ("_is_loopback", "_request_headers")

    def __init__(self, source: JsonSnapshotTransport, *, path: str) -> None:
        self._is_loopback = bool(getattr(source, "is_loopback", False))
        self._request_headers = MappingProxyType(dict(getattr(source, "request_headers", {})))
        super().__init__(source, path=path)

    @property
    def is_loopback(self) -> bool:
        return self._is_loopback

    @property
    def request_headers(self) -> Mapping[str, str]:
        return self._request_headers


def _run_token(value: str | None, *, label: str, generated_prefix: str) -> str:
    candidate = value if value is not None else f"{generated_prefix}-{secrets.token_hex(16)}"
    if _TOKEN_RE.fullmatch(candidate) is None:
        raise ValueError(f"{label} must be a shell-inert contract token")
    if label == "run_nonce" and len(candidate) < 16:
        raise ValueError("run_nonce must contain at least 16 characters")
    if label == "run_id" and not candidate.startswith("R2-RUN-"):
        raise ValueError("run_id must begin with R2-RUN-")
    return candidate


def _profile_digests() -> Mapping[str, str]:
    profiles = load_all_profiles()
    by_subsystem = {profile.subsystem: profile.profile_sha256 for profile in profiles.values()}
    if set(by_subsystem) != {"ai_x5", "embodied_x5", "dual_arm", "assay_station"}:
        raise RuntimeError("packaged semantic profile inventory is incomplete")
    return MappingProxyType(by_subsystem)


def _remote_arguments(
    *, run_id: str, run_nonce: str, release_id: str, profile_sha256: str
) -> tuple[str, ...]:
    return (
        "--run-id",
        run_id,
        "--run-nonce",
        run_nonce,
        "--release-id",
        release_id,
        "--profile-sha256",
        profile_sha256,
    )


def _strict_ssh_prefix(*, target: Mapping[str, str], known_hosts_file: Path) -> tuple[str, ...]:
    """Mirror the collector transport's fixed OpenSSH trust and authority boundary."""

    known_hosts_text = str(known_hosts_file)
    if any(character in known_hosts_text for character in ("%", "~", "*", "?", "[", "]")) or any(
        character.isspace() for character in known_hosts_text
    ):
        raise LiveShadowConfigError("run-local known_hosts path contains OpenSSH expansion syntax")

    return (
        LOCAL_SSH_EXECUTABLE,
        "-F",
        "/dev/null",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "UpdateHostKeys=no",
        "-o",
        f"UserKnownHostsFile={known_hosts_file}",
        "-o",
        "GlobalKnownHostsFile=/dev/null",
        "-o",
        "KnownHostsCommand=none",
        "-o",
        f"HostKeyAlias={target['host_key_alias']}",
        "-o",
        "HostKeyAlgorithms=ssh-ed25519",
        "-o",
        "CheckHostIP=yes",
        "-o",
        "PasswordAuthentication=no",
        "-o",
        "KbdInteractiveAuthentication=no",
        "-o",
        "NumberOfPasswordPrompts=0",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ForwardAgent=no",
        "-o",
        "ClearAllForwardings=yes",
        "-o",
        "PermitLocalCommand=no",
        "-o",
        "ProxyCommand=none",
        "-o",
        "ProxyJump=none",
        "-o",
        "RequestTTY=no",
        "-o",
        "ConnectionAttempts=1",
        f"{target['user']}@{target['host']}",
    )


def _run_preflight_command(command: tuple[str, ...], *, timeout_s: float) -> tuple[int, bytes, bytes]:
    return read_only_transport._bounded_popen_run(  # noqa: SLF001
        command,
        timeout_s=timeout_s,
    )


class _StrictPinnedSshJsonSnapshotTransport:
    """Collector transport sharing the exact isolated SSH prefix used by preflight."""

    __slots__ = ("_allowed_path", "_command", "_network_touched", "_timeout_s")

    def __init__(
        self,
        *,
        target: Mapping[str, str],
        known_hosts_file: Path,
        expected_script_sha256: str,
        remote_arguments: tuple[str, ...],
        allowed_path: str,
        timeout_s: float,
    ) -> None:
        remote_script = target["remote_script"]
        tokens = (remote_script, *remote_arguments)
        if any(
            not isinstance(token, str) or _SSH_REMOTE_TOKEN_RE.fullmatch(token) is None for token in tokens
        ):
            raise ValueError("SSH collector command contains a non-contract token")
        if not remote_script.startswith("/home/rdk or not remote_script.endswith(".py"):
            raise ValueError("SSH collector script path is outside the frozen topology")
        require_sha256("expected_script_sha256", expected_script_sha256)
        _read_frozen_known_hosts(known_hosts_file)
        self._allowed_path = allowed_path
        self._timeout_s = timeout_s
        self._network_touched = False
        verified_remote_command = " ".join(
            shlex.quote(token)
            for token in (
                REMOTE_PYTHON_EXECUTABLE,
                "-I",
                "-c",
                _VERIFIED_COLLECTOR_BOOTSTRAP,
                expected_script_sha256,
                remote_script,
                *remote_arguments,
            )
        )
        self._command = (
            *_strict_ssh_prefix(target=target, known_hosts_file=known_hosts_file),
            verified_remote_command,
        )

    @property
    def network_touched(self) -> bool:
        return self._network_touched

    @property
    def source_kind(self) -> ReadSourceKind:
        return ReadSourceKind.LIVE_REMOTE_READ

    def get_json(self, path: str) -> Mapping[str, Any]:
        if path != self._allowed_path:
            raise ValueError("snapshot path is not allowlisted")
        self._network_touched = True
        return_code, stdout, _stderr = _run_preflight_command(
            self._command,
            timeout_s=self._timeout_s,
        )
        if return_code != 0:
            raise RuntimeError("verified read-only SSH collector failed")
        try:
            payload = json.loads(
                stdout.decode("utf-8"),
                object_pairs_hook=_strict_object,
                parse_constant=_reject_json_constant,
            )
        except LiveShadowConfigError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("SSH collector response is not strict UTF-8 JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("SSH collector response must be one JSON object")
        return payload


def _expected_collector_sha256(config: LiveShadowConfig, target_id: str) -> str:
    if target_id == "embodied_x5":
        return config.embodied_required_artifact_sha256["collector_script"]
    if target_id in _ARM_IDS:
        return config.arm_probe_script_sha256
    raise ValueError(f"unregistered SSH collector target: {target_id}")


def _preflight_remote_collectors(
    config: LiveShadowConfig,
    *,
    known_hosts_file: Path,
    attempt_log: list[_CollectorPreflightAttempt] | None = None,
) -> Mapping[str, _CollectorPreflightResult]:
    """Read and verify every remote collector digest before any collector executes."""

    _read_frozen_known_hosts(known_hosts_file)
    attempts = [] if attempt_log is None else attempt_log
    results: dict[str, _CollectorPreflightResult] = {}
    for target_id, timeout_s in (
        ("embodied_x5", 12.0),
        ("arm01", 8.0),
        ("arm02", 8.0),
    ):
        target = config.targets[target_id]
        remote_script = target["remote_script"]
        command = (
            *_strict_ssh_prefix(
                target=target,
                known_hosts_file=known_hosts_file,
            ),
            "/usr/bin/sha256sum",
            "--",
            remote_script,
        )
        expected_sha256 = _expected_collector_sha256(config, target_id)
        attempt_index = len(attempts)
        attempt_base = {
            "target_id": target_id,
            "host": target["host"],
            "remote_script": remote_script,
            "expected_sha256": expected_sha256,
        }
        # Cross the transport boundary only after reserving a conservative
        # attempt record. This keeps failures auditable even when an adapter
        # contacts the target and then violates its declared return contract.
        attempts.append(
            _CollectorPreflightAttempt(
                **attempt_base,
                outcome="TRANSPORT_ATTEMPTED_NO_RESULT",
            )
        )
        try:
            transport_result = _run_preflight_command(
                command,
                timeout_s=timeout_s,
            )
            if not isinstance(transport_result, tuple) or len(transport_result) != 3:
                raise TypeError("preflight transport returned an invalid result tuple")
            return_code, stdout, stderr = transport_result
            if (
                isinstance(return_code, bool)
                or not isinstance(return_code, int)
                or not isinstance(stdout, bytes)
                or not isinstance(stderr, bytes)
            ):
                raise TypeError("preflight transport returned invalid result field types")
        except Exception as exc:
            attempts[attempt_index] = _CollectorPreflightAttempt(
                **attempt_base,
                outcome="TRANSPORT_EXCEPTION",
                error_type=type(exc).__name__,
            )
            raise RuntimeError(
                f"remote collector preflight transport failed closed for {target_id}; collector not executed"
            ) from exc
        response_fields = {
            "return_code": return_code,
            "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            "stdout_size_bytes": len(stdout),
            "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
            "stderr_size_bytes": len(stderr),
        }
        if return_code != 0 or stderr:
            attempts[attempt_index] = _CollectorPreflightAttempt(
                **attempt_base,
                outcome="COMMAND_FAILED",
                **response_fields,
            )
            raise RuntimeError(
                f"remote collector preflight failed closed for {target_id}; collector not executed"
            )
        try:
            text = stdout.decode("ascii")
        except UnicodeDecodeError as exc:
            attempts[attempt_index] = _CollectorPreflightAttempt(
                **attempt_base,
                outcome="NON_ASCII_RESPONSE",
                error_type=type(exc).__name__,
                **response_fields,
            )
            raise RuntimeError(
                f"remote collector preflight returned non-ASCII output for {target_id}; "
                "collector not executed"
            ) from exc
        match = _REMOTE_SHA256_LINE_RE.fullmatch(text)
        if match is None or match.group(2) != remote_script:
            attempts[attempt_index] = _CollectorPreflightAttempt(
                **attempt_base,
                outcome="INVALID_DIGEST_RECORD",
                **response_fields,
            )
            raise RuntimeError(
                f"remote collector preflight returned an invalid digest record for {target_id}; "
                "collector not executed"
            )
        result = _CollectorPreflightResult(
            target_id=target_id,
            host=target["host"],
            remote_script=remote_script,
            expected_sha256=expected_sha256,
            observed_sha256=match.group(1),
        )
        if not secrets.compare_digest(result.observed_sha256, expected_sha256):
            attempts[attempt_index] = _CollectorPreflightAttempt(
                **attempt_base,
                observed_sha256=result.observed_sha256,
                outcome="HASH_MISMATCH",
                **response_fields,
            )
            raise RuntimeError(f"remote collector SHA-256 mismatch for {target_id}; collector not executed")
        attempts[attempt_index] = _CollectorPreflightAttempt(
            **attempt_base,
            observed_sha256=result.observed_sha256,
            outcome="VERIFIED",
            **response_fields,
        )
        results[target_id] = result
    return MappingProxyType(results)


def _construct_sources(
    config: LiveShadowConfig,
    binding: ShadowRunBinding | _LiveSourceSeed,
    *,
    known_hosts_file: Path | None = None,
) -> tuple[dict[str, JsonSnapshotTransport], dict[str, str]]:
    pinned_known_hosts = config.known_hosts_file if known_hosts_file is None else known_hosts_file
    headers = {
        "X-RB-VoE-Run-Binding": binding.source_binding_sha256("ai_x5"),
        "X-RB-VoE-Profile-SHA256": binding.profile_sha256["ai_x5"],
    }
    sources: dict[str, JsonSnapshotTransport] = {
        "ai_x5": HttpJsonSnapshotTransport(
            config.targets["ai_x5"]["base_url"],
            allowed_paths=AI_X5_SNAPSHOT_PATHS,
            request_headers=headers,
            timeout_s=3.0,
        )
    }
    paths = {"ai_x5": AI_X5_RUNTIME_SNAPSHOT_PATH}

    embodied = config.targets["embodied_x5"]
    sources["embodied_x5"] = _StrictPinnedSshJsonSnapshotTransport(
        target=embodied,
        known_hosts_file=pinned_known_hosts,
        expected_script_sha256=_expected_collector_sha256(config, "embodied_x5"),
        remote_arguments=_remote_arguments(
            run_id=binding.run_id,
            run_nonce=binding.run_nonce,
            release_id=binding.release_id,
            profile_sha256=binding.profile_sha256["embodied_x5"],
        ),
        allowed_path=EMBODIED_X5_SNAPSHOT_PATH,
        timeout_s=12.0,
    )
    paths["embodied_x5"] = EMBODIED_X5_SNAPSHOT_PATH

    for arm_id in _ARM_IDS:
        target = config.targets[arm_id]
        sources[arm_id] = _StrictPinnedSshJsonSnapshotTransport(
            target=target,
            known_hosts_file=pinned_known_hosts,
            expected_script_sha256=_expected_collector_sha256(config, arm_id),
            remote_arguments=(
                "--arm-id",
                arm_id,
                "--probe-script-sha256",
                config.arm_probe_script_sha256,
                *_remote_arguments(
                    run_id=binding.run_id,
                    run_nonce=binding.run_nonce,
                    release_id=binding.release_id,
                    profile_sha256=binding.profile_sha256["dual_arm"],
                ),
            ),
            allowed_path=DUAL_ARM_MEMBER_PATHS[arm_id],
            timeout_s=8.0,
        )
        paths[arm_id] = DUAL_ARM_MEMBER_PATHS[arm_id]
    return sources, paths


def _prefetch_sources(
    sources: Mapping[str, JsonSnapshotTransport], paths: Mapping[str, str]
) -> dict[str, PrefetchedJsonSnapshotTransport]:
    def capture(name: str) -> PrefetchedJsonSnapshotTransport:
        if name == "ai_x5":
            return _AiPrefetchedJsonSnapshotTransport(sources[name], path=paths[name])
        return PrefetchedJsonSnapshotTransport(sources[name], path=paths[name])

    names = ("ai_x5", "embodied_x5", "arm01", "arm02")
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="rb-voe-read") as executor:
        futures = {name: executor.submit(capture, name) for name in names}
        return {name: futures[name].result() for name in names}


def _available_output_dir(path: str | Path) -> Path:
    output = Path(path)
    if not output.is_absolute() or ".." in output.parts:
        raise ValueError("output_dir must be an absolute lexical path")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"evidence output already exists: {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("evidence output parent must be an existing regular directory")
    return output


def _reserve_output_dir(path: str | Path) -> Path:
    output = _available_output_dir(path)
    output.mkdir(mode=0o700, exist_ok=False)
    _fsync_directory(output.parent)
    return output


def _materialize_run_known_hosts(config: LiveShadowConfig, *, output: Path) -> Path:
    raw = _read_frozen_known_hosts(config.known_hosts_file)
    run_local = output / "transport_known_hosts"
    _write_new(run_local, raw)
    run_local.chmod(0o600)
    if _read_frozen_known_hosts(run_local) != raw:
        raise RuntimeError("run-local known_hosts materialization changed bytes")
    return run_local


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry changes on POSIX deployment hosts."""

    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _document(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _seal_output_inventory(output: Path, *, schema_version: str) -> str:
    entries = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = path.relative_to(output).as_posix()
        entries.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    index_unsigned = {
        "schema_version": schema_version,
        "inventory_scope": "all_files_except_evidence_index.json",
        "file_count": len(entries),
        "files": entries,
    }
    root_sha256 = canonical_sha256(index_unsigned)
    _write_new(
        output / "evidence_index.json",
        _document({**index_unsigned, "root_sha256": root_sha256}),
    )
    return root_sha256


def _write_attempt_started(
    *,
    output: Path,
    config: LiveShadowConfig,
    challenge: _ValidatedChallenge,
    attempt_started_at_ms: int,
) -> None:
    _write_new(
        output / "attempt_started.json",
        _document(
            {
                "schema_version": LIVE_SHADOW_ATTEMPT_EVIDENCE_SCHEMA,
                "attempt_status": "STARTED",
                "next_phase": "REMOTE_COLLECTOR_PREFLIGHT",
                "attempt_started_at_ms": attempt_started_at_ms,
                "config_sha256": config.config_sha256,
                "release_id": config.release_id,
                "known_hosts_sha256": config.known_hosts_sha256,
                "challenge_sha256": challenge.challenge_sha256,
                "challenge_content_sha256": challenge.challenge_content_sha256,
                "challenge_artifact_sha256": challenge.artifact_sha256,
                "run_id": challenge.artifact["run_id"],
                "run_nonce_sha256": challenge.artifact["run_nonce_sha256"],
                "remote_contacted": False,
                "network_touched": False,
                "execution_authority": False,
                "transport_commands_issued": 0,
                "read_only_device_observations": 0,
                "actuator_commands_issued": 0,
                "mutating_commands_issued": 0,
            }
        ),
    )


def _materialize_challenge_consumption_receipt(
    challenge: _ValidatedChallenge,
    *,
    output: Path,
) -> None:
    receipt_path = _challenge_sidecar(challenge.path, "consumed")
    payload, raw = _read_strict_json_artifact(
        receipt_path,
        label="challenge consumption receipt",
    )
    receipt = _exact_keys(
        payload,
        _CHALLENGE_CONSUMPTION_KEYS,
        label="challenge consumption receipt",
    )
    expected = {
        "schema_version": LIVE_SHADOW_CHALLENGE_CONSUMPTION_SCHEMA,
        "challenge_sha256": challenge.challenge_sha256,
        "challenge_artifact_sha256": challenge.artifact_sha256,
        "config_sha256": challenge.artifact["config_sha256"],
        "bootstrap_manifest_sha256": challenge.artifact["bootstrap_manifest_sha256"],
        "bootstrap_manifest_file_sha256": challenge.artifact["bootstrap_manifest_file_sha256"],
        "consumed_at_ms": receipt["consumed_at_ms"],
    }
    _non_negative_int(receipt["consumed_at_ms"], label="consumed_at_ms")
    if receipt != expected:
        raise LiveShadowConfigError("challenge consumption receipt does not bind the attempt")
    _write_new(output / "challenge_consumption_receipt.json", raw)


def _seal_preflight_failure_attempt(
    *,
    output: Path,
    config: LiveShadowConfig,
    challenge: _ValidatedChallenge,
    run_known_hosts_file: Path,
    attempts: list[_CollectorPreflightAttempt],
    error: Exception,
) -> str:
    transport_commands = sum(int(attempt.transport_command_issued) for attempt in attempts)
    if transport_commands <= 0:
        raise RuntimeError("cannot seal a remote preflight failure without a transport attempt")
    if attempts[-1].outcome == "VERIFIED":
        raise RuntimeError("cannot seal a preflight failure whose final attempt is verified")
    if any(not attempt.read_only or attempt.collector_executed_by_preflight for attempt in attempts):
        raise RuntimeError("preflight failure evidence contains a non-read-only attempt")
    verified_collectors = sum(attempt.outcome == "VERIFIED" for attempt in attempts)
    common = {
        "schema_version": LIVE_SHADOW_ATTEMPT_EVIDENCE_SCHEMA,
        "attempt_status": "FAILED_CLOSED",
        "failure_phase": "REMOTE_COLLECTOR_PREFLIGHT",
        "failure_code": "REMOTE_COLLECTOR_PREFLIGHT_FAILED",
        "failure_type": type(error).__name__,
        "failure_message": str(error),
        "failure_outcome": attempts[-1].outcome,
        "challenge_consumed": False,
        "challenge_consumption_receipt_sha256": None,
        "remote_contacted": True,
        "network_touched": True,
        "execution_authority": False,
        "transport_commands_issued": transport_commands,
        "read_only_transport_operations": transport_commands,
        "read_only_device_observations": 0,
        "actuator_commands_issued": 0,
        "mutating_commands_issued": 0,
        "actions_invoked": 0,
        "physical_risk_denominator_increment": 0,
        "collector_execution_attempted": False,
        "collector_snapshot_requests_issued": 0,
        "verified_collectors_before_failure": verified_collectors,
    }
    documents = {
        "challenge_binding.json": dict(challenge.artifact),
        "config_binding.json": {
            "schema_version": LIVE_SHADOW_ATTEMPT_EVIDENCE_SCHEMA,
            "config_schema_version": config.schema_version,
            "config_sha256": config.config_sha256,
            "release_id": config.release_id,
            "known_hosts_sha256": config.known_hosts_sha256,
            "compiled_known_hosts_sha256": FROZEN_KNOWN_HOSTS_SHA256,
            "run_local_known_hosts_sha256": file_sha256(run_known_hosts_file),
            "challenge_sha256": challenge.challenge_sha256,
            "challenge_content_sha256": challenge.challenge_content_sha256,
            "challenge_artifact_sha256": challenge.artifact_sha256,
        },
        "collector_preflight_failure.json": {
            **common,
            "attempts": [attempt.to_dict() for attempt in attempts],
        },
        "metadata.json": {
            **common,
            "case_id": challenge.artifact["case_id"],
            "sample_id": challenge.artifact["sample_id"],
            "sample_lineage_sha256": challenge.artifact["sample_lineage_sha256"],
            "parent_evidence_root_sha256": challenge.artifact["parent_evidence_root_sha256"],
            "failure_evidence_sealed": True,
        },
    }
    for name in sorted(documents):
        _write_new(output / name, _document(documents[name]))
    return _seal_output_inventory(
        output,
        schema_version=LIVE_SHADOW_ATTEMPT_EVIDENCE_SCHEMA,
    )


def _recover_challenge_consumption_receipt(
    *,
    output: Path,
    challenge: _ValidatedChallenge,
) -> tuple[str, str | None, str | None]:
    consumed_path = _challenge_sidecar(challenge.path, "consumed")
    if not consumed_path.exists() and not consumed_path.is_symlink():
        return "ABSENT", None, None
    if consumed_path.is_symlink() or not consumed_path.is_file():
        return "PRESENT_UNVERIFIED_FAIL_CLOSED", None, None
    try:
        payload, raw = _read_strict_json_artifact(
            consumed_path,
            label="challenge consumption receipt",
        )
        receipt = _exact_keys(
            payload,
            _CHALLENGE_CONSUMPTION_KEYS,
            label="challenge consumption receipt",
        )
        expected = {
            "schema_version": LIVE_SHADOW_CHALLENGE_CONSUMPTION_SCHEMA,
            "challenge_sha256": challenge.challenge_sha256,
            "challenge_artifact_sha256": challenge.artifact_sha256,
            "config_sha256": challenge.artifact["config_sha256"],
            "bootstrap_manifest_sha256": challenge.artifact["bootstrap_manifest_sha256"],
            "bootstrap_manifest_file_sha256": challenge.artifact["bootstrap_manifest_file_sha256"],
            "consumed_at_ms": receipt["consumed_at_ms"],
        }
        _non_negative_int(receipt["consumed_at_ms"], label="consumed_at_ms")
        if receipt != expected:
            return "PRESENT_UNVERIFIED_FAIL_CLOSED", None, hashlib.sha256(raw).hexdigest()
    except (LiveShadowConfigError, OSError, TypeError, ValueError):
        return "PRESENT_UNVERIFIED_FAIL_CLOSED", None, None

    target = output / "challenge_consumption_receipt.json"
    if target.exists() or target.is_symlink():
        try:
            if target.is_symlink() or target.read_bytes() != raw:
                target = output / "challenge_consumption_receipt_recovered.json"
        except OSError:
            target = output / "challenge_consumption_receipt_recovered.json"
    if not target.exists() and not target.is_symlink():
        _write_new(target, raw)
    receipt_sha256 = hashlib.sha256(raw).hexdigest()
    if file_sha256(target) != receipt_sha256:
        raise RuntimeError("recovered challenge consumption receipt changed bytes")
    return "VERIFIED_CONSUMED", target.relative_to(output).as_posix(), receipt_sha256


def _seal_live_failure_attempt(
    *,
    config: LiveShadowConfig,
    state: _LiveFailureState,
    error: Exception,
) -> str:
    output = state.output
    challenge = state.challenge
    if output is None or challenge is None:
        raise RuntimeError("cannot seal a live failure before challenge and output reservation")
    evidence_index = output / "evidence_index.json"
    if evidence_index.exists() or evidence_index.is_symlink():
        raise RuntimeError("live attempt already has an evidence index")

    consumption_status, receipt_file, receipt_sha256 = _recover_challenge_consumption_receipt(
        output=output,
        challenge=challenge,
    )
    preflight_commands = sum(int(attempt.transport_command_issued) for attempt in state.preflight_attempts)
    snapshot_commands = state.snapshot_transport_attempts
    transport_commands = preflight_commands + snapshot_commands
    remote_contacted = transport_commands > 0
    run_known_hosts_sha256: str | None = None
    known_hosts = state.run_known_hosts_file
    if known_hosts is not None and known_hosts.is_file() and not known_hosts.is_symlink():
        run_known_hosts_sha256 = file_sha256(known_hosts)
    common = {
        "schema_version": LIVE_SHADOW_ATTEMPT_EVIDENCE_SCHEMA,
        "attempt_status": "FAILED_CLOSED",
        "failure_phase": state.failure_phase,
        "failure_code": "LIVE_READONLY_SHADOW_ATTEMPT_FAILED",
        "failure_type": type(error).__name__,
        "failure_message": str(error),
        "challenge_consumed": consumption_status != "ABSENT",
        "challenge_consumption_status": consumption_status,
        "challenge_consumption_receipt_file": receipt_file,
        "challenge_consumption_receipt_sha256": receipt_sha256,
        "remote_contacted": remote_contacted,
        "network_touched": remote_contacted,
        "execution_authority": False,
        "transport_commands_issued": transport_commands,
        "read_only_transport_operations": transport_commands,
        "preflight_transport_commands_issued": preflight_commands,
        "snapshot_transport_commands_issued": snapshot_commands,
        "read_only_device_observations": 0,
        "snapshot_results_discarded_due_to_failure": snapshot_commands > 0,
        "actuator_commands_issued": 0,
        "mutating_commands_issued": 0,
        "actions_invoked": 0,
        "physical_risk_denominator_increment": 0,
        "collector_execution_attempted": snapshot_commands > 0,
        "collector_snapshot_requests_issued": 3 if snapshot_commands > 0 else 0,
        "loopback_snapshot_requests_issued": 1 if snapshot_commands > 0 else 0,
        "verified_collectors_before_failure": sum(
            attempt.outcome == "VERIFIED" for attempt in state.preflight_attempts
        ),
    }
    documents = {
        "failure_binding.json": {
            "schema_version": LIVE_SHADOW_ATTEMPT_EVIDENCE_SCHEMA,
            "config_schema_version": config.schema_version,
            "config_sha256": config.config_sha256,
            "release_id": config.release_id,
            "known_hosts_sha256": config.known_hosts_sha256,
            "compiled_known_hosts_sha256": FROZEN_KNOWN_HOSTS_SHA256,
            "run_local_known_hosts_sha256": run_known_hosts_sha256,
            "challenge_sha256": challenge.challenge_sha256,
            "challenge_content_sha256": challenge.challenge_content_sha256,
            "challenge_artifact_sha256": challenge.artifact_sha256,
            "run_id": challenge.artifact["run_id"],
            "run_nonce_sha256": challenge.artifact["run_nonce_sha256"],
        },
        "live_failure.json": {
            **common,
            "collector_preflight_attempts": [attempt.to_dict() for attempt in state.preflight_attempts],
        },
        "failure_metadata.json": {
            **common,
            "case_id": challenge.artifact["case_id"],
            "sample_id": challenge.artifact["sample_id"],
            "sample_lineage_sha256": challenge.artifact["sample_lineage_sha256"],
            "parent_evidence_root_sha256": challenge.artifact["parent_evidence_root_sha256"],
            "failure_evidence_sealed": True,
        },
    }
    for name in sorted(documents):
        _write_new(output / name, _document(documents[name]))
    return _seal_output_inventory(
        output,
        schema_version=LIVE_SHADOW_ATTEMPT_EVIDENCE_SCHEMA,
    )


def _seal_evidence(
    *,
    output: Path,
    config: LiveShadowConfig,
    binding: ShadowRunBinding,
    report: ShadowRunReport,
    captures: Mapping[str, PrefetchedJsonSnapshotTransport],
    central_capture_errors: Mapping[str, str],
    challenge: _ValidatedChallenge,
    collector_preflights: Mapping[str, _CollectorPreflightResult],
    run_known_hosts_file: Path,
    bootstrap_bundle: _ValidatedBootstrapBundle,
) -> str:
    capture_errors: dict[str, dict[str, Any]] = {}
    raw_documents: dict[str, bytes] = {}
    raw_names = {
        "ai_x5": "ai_x5.json",
        "embodied_x5": "embodied_x5.json",
        "arm01": "arm01.json",
        "arm02": "arm02.json",
        "overhead_record": "overhead_record.json",
        "overhead_acquisition": "overhead_acquisition.json",
        "overhead_consumption_receipt": "overhead_consumption_receipt.json",
    }
    for name, capture in captures.items():
        error = central_capture_errors.get(name) or capture.capture_error
        payload = capture.payload
        if payload is not None:
            try:
                raw_documents[raw_names[name]] = _document(payload)
            except (TypeError, ValueError):
                error = "CanonicalEncodingError"
        capture_errors[name] = {
            "capture_error": error,
            "payload_available": payload is not None,
            "source_kind": capture.source_kind.value,
            "network_touched": capture.network_touched,
        }

    if raw_documents:
        raw_dir = output / "raw"
        raw_dir.mkdir(exist_ok=False)
        for name in sorted(raw_documents):
            _write_new(raw_dir / name, raw_documents[name])

    source_bindings = {
        subsystem: binding.source_binding_sha256(subsystem)
        for subsystem in ("ai_x5", "embodied_x5", "dual_arm", "assay_station")
    }
    collector_snapshot_observations = sum(
        int(captures[name].payload is not None) for name in ("embodied_x5", "arm01", "arm02")
    )
    loopback_snapshot_observations = int(captures["ai_x5"].payload is not None)
    a0_actual_observations = int(captures["overhead_record"].payload is not None)
    collector_transport_commands = sum(
        int(captures[name].network_touched) for name in ("embodied_x5", "arm01", "arm02")
    )
    preflight_transport_commands = len(collector_preflights)
    coordinator_summary = report.to_dict()
    coordinator_schema_version = coordinator_summary.pop("schema_version")
    coordinator_summary.pop("commands_issued", None)
    coordinator_summary.pop("hardware_touched", None)
    coordinator_summary.update(
        {
            "schema_version": LIVE_SHADOW_EVIDENCE_SCHEMA,
            "coordinator_schema_version": coordinator_schema_version,
            "coordinator_report_sha256": report.content_sha256,
            "transport_commands_issued": (preflight_transport_commands + collector_transport_commands),
            "read_only_device_observations": (
                collector_snapshot_observations + loopback_snapshot_observations + a0_actual_observations
            ),
            "actuator_commands_issued": 0,
            "mutating_commands_issued": 0,
        }
    )
    documents = {
        "capture_errors.json": {
            "schema_version": LIVE_SHADOW_EVIDENCE_SCHEMA,
            "captures": capture_errors,
        },
        "config_binding.json": {
            "schema_version": LIVE_SHADOW_EVIDENCE_SCHEMA,
            "config_schema_version": config.schema_version,
            "config_sha256": config.config_sha256,
            "release_id": config.release_id,
            "known_hosts_sha256": config.known_hosts_sha256,
            "compiled_known_hosts_sha256": FROZEN_KNOWN_HOSTS_SHA256,
            "run_local_known_hosts_sha256": file_sha256(run_known_hosts_file),
            "challenge_sha256": challenge.challenge_sha256,
            "challenge_content_sha256": challenge.challenge_content_sha256,
            "challenge_artifact_sha256": challenge.artifact_sha256,
            "bootstrap_manifest_sha256": bootstrap_bundle.manifest_sha256,
            "bootstrap_manifest_file_sha256": bootstrap_bundle.manifest_file_sha256,
            "bootstrap_trust_scope": "OPERATOR_ENROLLED_CURRENT_BOOT",
            "independent_release_authority": False,
            "bootstrap_evidence": dict(config.bootstrap_evidence),
        },
        "challenge_binding.json": dict(challenge.artifact),
        "collector_preflight.json": {
            "schema_version": LIVE_SHADOW_EVIDENCE_SCHEMA,
            "all_collectors_verified_before_execution": True,
            "transport_commands_issued": preflight_transport_commands,
            "results": {
                target_id: collector_preflights[target_id].to_dict()
                for target_id in sorted(collector_preflights)
            },
        },
        "coordinator_report.json": coordinator_summary,
        "metadata.json": {
            "schema_version": LIVE_SHADOW_EVIDENCE_SCHEMA,
            "remote_contacted": bool(collector_preflights)
            or any(capture.network_touched for capture in captures.values()),
            "network_touched": bool(collector_preflights)
            or any(capture.network_touched for capture in captures.values()),
            "execution_authority": False,
            "actions_invoked": 0,
            "transport_commands_issued": (preflight_transport_commands + collector_transport_commands),
            "collector_hash_preflight_reads": preflight_transport_commands,
            "collector_snapshot_transport_commands": collector_transport_commands,
            "loopback_read_requests_issued": int(captures["ai_x5"].network_touched),
            "read_only_device_observations": (
                collector_snapshot_observations + loopback_snapshot_observations + a0_actual_observations
            ),
            "collector_snapshot_observations": collector_snapshot_observations,
            "loopback_snapshot_observations": loopback_snapshot_observations,
            "bound_a0_actual_observations": a0_actual_observations,
            "actuator_commands_issued": 0,
            "mutating_commands_issued": 0,
            "read_only_transport_operations": (
                preflight_transport_commands
                + sum(int(capture.network_touched) for capture in captures.values())
            ),
            "local_sealed_evidence_reads": sum(
                int(not capture.network_touched) for capture in captures.values()
            ),
            "physical_closure_proven": False,
            "physical_risk_denominator_increment": 0,
            "assay_station": "TARGET_ONLY",
            "collector_code_verified_before_execution": True,
            "bootstrap_bundle_verified": True,
            "bootstrap_files_sealed": len(bootstrap_bundle.raw_files),
            "collector_execution_reverification": ("SHA256_VERIFIED_IN_MEMORY_BYTES_BEFORE_EXEC"),
            "challenge_sha256": challenge.challenge_sha256,
            "challenge_content_sha256": challenge.challenge_content_sha256,
            "case_id": challenge.artifact["case_id"],
            "sample_id": challenge.artifact["sample_id"],
            "sample_lineage_sha256": challenge.artifact["sample_lineage_sha256"],
            "parent_evidence_root_sha256": challenge.artifact["parent_evidence_root_sha256"],
        },
        "run_binding.json": {
            "schema_version": LIVE_SHADOW_EVIDENCE_SCHEMA,
            "run_id": binding.run_id,
            "run_nonce": binding.run_nonce,
            "release_id": binding.release_id,
            "evaluated_at_ms": binding.evaluated_at_ms,
            "mode": binding.mode.value,
            "profile_sha256": dict(binding.profile_sha256),
            "source_binding_sha256": source_bindings,
            "challenge_sha256": challenge.challenge_sha256,
            "challenge_content_sha256": challenge.challenge_content_sha256,
            "case_id": challenge.artifact["case_id"],
            "sample_id": challenge.artifact["sample_id"],
            "sample_lineage_sha256": challenge.artifact["sample_lineage_sha256"],
            "parent_evidence_root_sha256": challenge.artifact["parent_evidence_root_sha256"],
        },
    }
    for name in sorted(documents):
        _write_new(output / name, _document(documents[name]))
    return _seal_output_inventory(output, schema_version=LIVE_SHADOW_EVIDENCE_SCHEMA)


def _clock_ms() -> int:
    return int(time.time() * 1000)


def _non_negative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LiveShadowConfigError(f"{label} must be a non-negative integer")
    return value


def _contract_token(value: Any, *, label: str) -> str:
    text = _string(value, label=label)
    if _TOKEN_RE.fullmatch(text) is None or _is_placeholder(text):
        raise LiveShadowConfigError(f"{label} must be a concrete shell-inert contract token")
    return text


def _challenge_sidecar(path: Path, state: str) -> Path:
    return path.with_name(f"{path.name}.{state}.json")


def _new_challenge_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or path.suffix.casefold() != ".json":
        raise LiveShadowConfigError("challenge_file must be an absolute lexical JSON path")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"challenge artifact already exists: {path}")
    if not path.parent.is_dir() or _is_reparse_point(path.parent):
        raise LiveShadowConfigError(
            "challenge artifact parent must be an existing regular non-link directory"
        )
    for parent in path.parents:
        if parent.exists() and _is_reparse_point(parent):
            raise LiveShadowConfigError("challenge artifact path cannot traverse a link")
    for state in ("issued", "consumed"):
        sidecar = _challenge_sidecar(path, state)
        if sidecar.exists() or sidecar.is_symlink():
            raise FileExistsError(f"challenge state artifact already exists: {sidecar}")
    return path


def _read_strict_json_artifact(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_stable_regular_bytes(
        path,
        label=label,
        max_bytes=1024 * 1024,
    )
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except LiveShadowConfigError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LiveShadowConfigError(f"{label} must be strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise LiveShadowConfigError(f"{label} must contain one JSON object")
    return payload, raw


_DISALLOWED_BOOTSTRAP_PROVENANCE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[^a-z0-9])(fixture|simulated|counterfactual|synthetic|mock)(?:$|[^a-z0-9])",
    re.IGNORECASE,
)
_FROZEN_BOOTSTRAP_LITERAL_ALLOWLIST: Final[frozenset[str]] = frozenset({"material_fixture_executor"})


def _walk_bootstrap_strings(
    value: Any,
    path: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], str]]:
    found: list[tuple[tuple[str, ...], str]] = []
    if isinstance(value, str):
        found.append((path, value))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            found.extend(_walk_bootstrap_strings(item, (*path, str(key))))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_walk_bootstrap_strings(item, (*path, str(index))))
    return found


def _reject_nonlive_bootstrap_provenance(
    snapshot: Mapping[str, Any],
    *,
    target_id: str,
) -> None:
    for path, value in _walk_bootstrap_strings(snapshot):
        if value not in _FROZEN_BOOTSTRAP_LITERAL_ALLOWLIST and _DISALLOWED_BOOTSTRAP_PROVENANCE_RE.search(
            value
        ):
            dotted = ".".join(path) or "<root>"
            raise LiveShadowConfigError(f"bootstrap {target_id}.{dotted} contains non-live provenance")


def _bootstrap_source_binding_sha256(
    *,
    subsystem: str,
    run_id: str,
    run_nonce: str,
    release_id: str,
    profile_sha256: str,
) -> str:
    return canonical_sha256(
        {
            "schema_version": "xrd-rb-voe-live-source-binding-v1",
            "subsystem": subsystem,
            "run_id": run_id,
            "run_nonce": run_nonce,
            "release_id": release_id,
            "profile_sha256": profile_sha256,
        }
    )


def _validate_bootstrap_snapshot(
    *,
    target_id: str,
    snapshot: Mapping[str, Any],
    root: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    run_nonce: str,
    now_ms: int,
) -> None:
    unsigned = dict(snapshot)
    claimed = unsigned.pop("snapshot_sha256", None)
    if claimed != canonical_sha256(unsigned) or claimed != bootstrap["snapshot_sha256"][target_id]:
        raise LiveShadowConfigError(f"bootstrap {target_id} snapshot digest mismatch")
    if snapshot.get("observed_at_ms") != bootstrap["observed_at_ms"][target_id]:
        raise LiveShadowConfigError(f"bootstrap {target_id} observation time mismatch")
    _reject_nonlive_bootstrap_provenance(snapshot, target_id=target_id)

    expected = root["expected"]
    profiles = _profile_digests()
    if snapshot.get("release_id") != root["release_id"]:
        raise LiveShadowConfigError(f"bootstrap {target_id} release mismatch")
    if target_id == "ai_x5":
        if (
            snapshot.get("schema_version") != AI_X5_RUNTIME_SNAPSHOT_SCHEMA
            or snapshot.get("ready") is not True
            or snapshot.get("reason_code") != "PASS"
            or snapshot.get("device_id") != expected["ai_device_id"]
            or snapshot.get("boot_id") != expected["ai_boot_id"]
            or snapshot.get("profile_sha256") != profiles["ai_x5"]
            or canonical_runtime_artifact_set_sha256(snapshot) != expected["ai_runtime_artifact_set_sha256"]
        ):
            raise LiveShadowConfigError("bootstrap AI identity or artifact set mismatch")
        binding = AiX5CapabilityBinding(
            device_id=expected["ai_device_id"],
            release_id=root["release_id"],
            required_backends=AI_X5_REQUIRED_BACKENDS,
            snapshot_max_age_ms=LIVE_SNAPSHOT_MAX_AGE_MS,
            run_binding_sha256=_bootstrap_source_binding_sha256(
                subsystem="ai_x5",
                run_id=bootstrap["run_id"],
                run_nonce=run_nonce,
                release_id=root["release_id"],
                profile_sha256=profiles["ai_x5"],
            ),
            profile_sha256=profiles["ai_x5"],
            expected_runtime_artifact_set_sha256=expected["ai_runtime_artifact_set_sha256"],
            expected_boot_id=expected["ai_boot_id"],
        )
        validator = object.__new__(AiX5ReadOnlyAdapter)
        validator._binding = binding  # type: ignore[attr-defined]
        parsed = validator._validate_snapshot(snapshot, now_ms=now_ms)  # noqa: SLF001
        if isinstance(parsed, str):
            raise LiveShadowConfigError(f"bootstrap AI strict validation failed: {parsed}")
        return
    if target_id == "embodied_x5":
        if (
            snapshot.get("schema_version") != EMBODIED_X5_SNAPSHOT_SCHEMA
            or snapshot.get("ready") is not True
            or snapshot.get("reason_code") != "PASS"
            or snapshot.get("device_id") != expected["embodied_device_id"]
            or snapshot.get("boot_id") != expected["embodied_boot_id"]
            or snapshot.get("profile_sha256") != profiles["embodied_x5"]
            or snapshot.get("run_id") != bootstrap["run_id"]
            or snapshot.get("run_nonce_sha256") != bootstrap["run_nonce_sha256"]
        ):
            raise LiveShadowConfigError("bootstrap embodied identity or run binding mismatch")
        artifacts = snapshot.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise LiveShadowConfigError("bootstrap embodied artifact inventory is missing")
        for name, digest in expected["embodied_required_artifact_sha256"].items():
            record = artifacts.get(name)
            if not isinstance(record, Mapping) or record.get("sha256") != digest:
                raise LiveShadowConfigError(f"bootstrap embodied artifact {name} mismatch")
        if set(artifacts) != set(EMBODIED_X5_ARTIFACT_PATHS):
            raise LiveShadowConfigError("bootstrap embodied artifact inventory is not exact")
        binding = EmbodiedX5CapabilityBinding(
            device_id=expected["embodied_device_id"],
            run_id=bootstrap["run_id"],
            run_nonce=run_nonce,
            release_id=root["release_id"],
            run_binding_sha256=_bootstrap_source_binding_sha256(
                subsystem="embodied_x5",
                run_id=bootstrap["run_id"],
                run_nonce=run_nonce,
                release_id=root["release_id"],
                profile_sha256=profiles["embodied_x5"],
            ),
            profile_sha256=profiles["embodied_x5"],
            required_artifact_sha256=expected["embodied_required_artifact_sha256"],
            expected_boot_id=expected["embodied_boot_id"],
            snapshot_max_age_ms=LIVE_SNAPSHOT_MAX_AGE_MS,
        )
        validator = object.__new__(EmbodiedX5ReadOnlyAdapter)
        validator._binding = binding  # type: ignore[attr-defined]
        reason = validator._validate_envelope(snapshot, now_ms=now_ms)  # noqa: SLF001
        if reason is None:
            reason = validator._validate_ready_semantics(snapshot)  # noqa: SLF001
        if reason is not None:
            raise LiveShadowConfigError(f"bootstrap embodied strict validation failed: {reason}")
        return

    identity = snapshot.get("identity")
    if (
        target_id not in _ARM_IDS
        or snapshot.get("schema_version") != DUAL_ARM_MEMBER_SCHEMA
        or snapshot.get("ready") is not True
        or snapshot.get("reasons") != []
        or not isinstance(identity, Mapping)
        or identity.get("machine_id_sha256") != expected["arm_machine_id_sha256"][target_id]
        or identity.get("boot_id") != expected["arm_boot_id"][target_id]
        or snapshot.get("profile_sha256") != profiles["dual_arm"]
        or snapshot.get("run_id") != bootstrap["run_id"]
        or snapshot.get("run_nonce") != run_nonce
    ):
        raise LiveShadowConfigError(f"bootstrap {target_id} identity or run binding mismatch")
    artifacts = snapshot.get("artifacts")
    probe = artifacts.get("probe_script") if isinstance(artifacts, Mapping) else None
    if not isinstance(probe, Mapping) or probe.get("sha256") != expected["arm_probe_script_sha256"]:
        raise LiveShadowConfigError(f"bootstrap {target_id} probe artifact mismatch")
    validator = object.__new__(DualArmReadOnlyAdapter)
    validator._binding = SimpleNamespace(  # type: ignore[attr-defined]
        run_id=bootstrap["run_id"],
        run_nonce=run_nonce,
        release_id=root["release_id"],
        profile_sha256=DUAL_ARM_PROFILE_SHA256,
        expected_machine_id_sha256=expected["arm_machine_id_sha256"],
        expected_boot_id=expected["arm_boot_id"],
        probe_script_sha256=DUAL_ARM_MEMBER_PROBE_SHA256,
        ttl_ms=LIVE_SNAPSHOT_MAX_AGE_MS,
    )
    reason = validator._validate_member(target_id, snapshot, now_ms=now_ms)  # noqa: SLF001
    if reason is not None:
        raise LiveShadowConfigError(f"bootstrap {target_id} strict validation failed: {reason}")


def _validate_bootstrap_bundle(
    root: Mapping[str, Any],
    *,
    config_path: Path,
) -> _ValidatedBootstrapBundle:
    manifest_path = Path(
        _lexical_absolute_path(
            root["bootstrap_manifest_file"],
            label="bootstrap_manifest_file",
            suffixes=(".json",),
        )
    )
    if manifest_path.parent != config_path.parent:
        raise LiveShadowConfigError("bootstrap manifest and final config must share one bundle directory")
    manifest_payload, manifest_raw = _read_strict_json_artifact(
        manifest_path,
        label="bootstrap manifest",
    )
    manifest = _exact_keys(
        manifest_payload,
        _BOOTSTRAP_MANIFEST_KEYS,
        label="bootstrap manifest",
    )
    unsigned_manifest = dict(manifest)
    manifest_sha256 = _string(
        unsigned_manifest.pop("manifest_sha256"),
        label="bootstrap manifest.manifest_sha256",
    )
    require_sha256("bootstrap manifest.manifest_sha256", manifest_sha256)
    if manifest_sha256 != canonical_sha256(unsigned_manifest):
        raise LiveShadowConfigError("bootstrap manifest digest mismatch")

    bootstrap = root["expected"]["bootstrap_evidence"]
    profiles = dict(_profile_digests())
    expected_fixed_targets = {
        "ai_x5": {
            "host": "192.0.2.103",
            "user": "sunrise",
            "host_key_alias": "192.0.2.103",
        },
        **{
            target_id: {key: root["targets"][target_id][key] for key in ("host", "user", "host_key_alias")}
            for target_id in ("embodied_x5", "arm01", "arm02")
        },
    }
    fixed_contract = {
        "schema_version": LIVE_BOOTSTRAP_MANIFEST_SCHEMA,
        "status": "COMMITTED",
        "release_id": root["release_id"],
        "known_hosts_sha256": FROZEN_KNOWN_HOSTS_SHA256,
        "profile_sha256": profiles,
        "fixed_targets": expected_fixed_targets,
        "parallel_process_count": 4,
        "capture_deadline_s": 18.0,
        "config_sha256": root["config_sha256"],
        "ai_runtime_artifact_set_sha256": root["expected"]["ai_runtime_artifact_set_sha256"],
        "bootstrap_evidence": bootstrap,
        "x5_host_key_distinguishes_devices": False,
        "x5_shared_host_key_observed": True,
        "remote_contacted": True,
        "network_touched": True,
        "pc_network_mutated": False,
        "proxy_or_jump_used": False,
        "services_mutated": False,
        "model_or_inference_triggered": False,
        "hardware_device_opened_by_capture": False,
        "actuator_commands_issued": 0,
        "execution_authority": False,
        "config_uploaded": False,
    }
    for name, value in fixed_contract.items():
        if manifest.get(name) != value:
            raise LiveShadowConfigError(f"bootstrap manifest {name} differs from its frozen contract")
    if (
        manifest.get("run_id") != bootstrap["run_id"]
        or manifest.get("run_nonce_sha256") != bootstrap["run_nonce_sha256"]
        or manifest.get("snapshot_sha256") != bootstrap["snapshot_sha256"]
    ):
        raise LiveShadowConfigError("bootstrap manifest run binding differs from the final config")
    generated_at_ms = manifest.get("generated_at_ms")
    if (
        isinstance(generated_at_ms, bool)
        or not isinstance(generated_at_ms, int)
        or generated_at_ms < max(bootstrap["observed_at_ms"].values())
        or generated_at_ms - max(bootstrap["observed_at_ms"].values()) > 60_000
    ):
        raise LiveShadowConfigError("bootstrap manifest generation time is incoherent")
    expected_template = Path(__file__).with_name("live_shadow_config.template.json")
    if manifest.get("template_sha256") != file_sha256(expected_template):
        raise LiveShadowConfigError("bootstrap manifest template digest differs from deployed release")
    template_payload, _ = _read_strict_json_artifact(
        expected_template.resolve(),
        label="packaged live config template",
    )
    template_expected = _exact_keys(
        template_payload["expected"],
        _EXPECTED_KEYS,
        label="packaged template.expected",
    )
    template_embodied = template_expected["embodied_required_artifact_sha256"]
    configured_embodied = root["expected"]["embodied_required_artifact_sha256"]
    for artifact_name in EMBODIED_X5_REQUIRED_ARTIFACTS - {"body_contour"}:
        if configured_embodied.get(artifact_name) != template_embodied.get(artifact_name):
            raise LiveShadowConfigError(
                f"bootstrap embodied {artifact_name} differs from the packaged release pin"
            )
    if root["expected"]["arm_probe_script_sha256"] != template_expected["arm_probe_script_sha256"]:
        raise LiveShadowConfigError("bootstrap arm probe differs from the packaged release pin")
    commands = manifest.get("command_sha256")
    if not isinstance(commands, Mapping) or set(commands) != set(_BOOTSTRAP_TARGET_IDS):
        raise LiveShadowConfigError("bootstrap command inventory is not exact")
    for target_id, digest in commands.items():
        require_sha256(f"bootstrap command_sha256.{target_id}", digest)

    config_payload, config_raw = _read_strict_json_artifact(config_path, label="final live config")
    if config_payload != root:
        raise LiveShadowConfigError("final config changed before bootstrap validation")
    file_hashes = manifest.get("file_sha256")
    expected_file_names = set(LIVE_BOOTSTRAP_SNAPSHOT_FILENAMES.values()) | {"live-config.final.json"}
    if not isinstance(file_hashes, Mapping) or set(file_hashes) != expected_file_names:
        raise LiveShadowConfigError("bootstrap file inventory is not exact")
    if file_hashes["live-config.final.json"] != hashlib.sha256(config_raw).hexdigest():
        raise LiveShadowConfigError("bootstrap final config file digest mismatch")

    snapshots: dict[str, Mapping[str, Any]] = {}
    raw_files: dict[str, bytes] = {
        "bootstrap_manifest.json": manifest_raw,
        "live-config.final.json": config_raw,
    }
    for target_id, filename in LIVE_BOOTSTRAP_SNAPSHOT_FILENAMES.items():
        snapshot_path = manifest_path.parent / filename
        payload, raw = _read_strict_json_artifact(
            snapshot_path,
            label=f"bootstrap {target_id} snapshot",
        )
        if hashlib.sha256(raw).hexdigest() != file_hashes[filename]:
            raise LiveShadowConfigError(f"bootstrap {target_id} snapshot file digest mismatch")
        snapshots[target_id] = payload
        raw_files[filename] = raw

    arm01_nonce = snapshots["arm01"].get("run_nonce")
    arm02_nonce = snapshots["arm02"].get("run_nonce")
    if (
        not isinstance(arm01_nonce, str)
        or len(arm01_nonce) < 16
        or arm02_nonce != arm01_nonce
        or hashlib.sha256(arm01_nonce.encode("utf-8")).hexdigest() != bootstrap["run_nonce_sha256"]
    ):
        raise LiveShadowConfigError("bootstrap raw run nonce cannot be reconstructed exactly")
    for target_id in _BOOTSTRAP_TARGET_IDS:
        _validate_bootstrap_snapshot(
            target_id=target_id,
            snapshot=snapshots[target_id],
            root=root,
            bootstrap=bootstrap,
            run_nonce=arm01_nonce,
            now_ms=generated_at_ms,
        )

    ai_secondary = manifest.get("ai_secondary_identity")
    embodied_secondary = manifest.get("embodied_secondary_identity")
    if (
        not isinstance(ai_secondary, Mapping)
        or set(ai_secondary) != {"hostname", "device_id", "boot_id", "envelope_sha256"}
        or ai_secondary.get("hostname") != "xrd-ai"
        or ai_secondary.get("device_id") != root["expected"]["ai_device_id"]
        or ai_secondary.get("boot_id") != root["expected"]["ai_boot_id"]
    ):
        raise LiveShadowConfigError("bootstrap AI secondary identity mismatch")
    require_sha256("bootstrap AI envelope_sha256", ai_secondary["envelope_sha256"])
    expected_ai_envelope_sha256 = canonical_sha256(
        {
            "schema_version": "xrd-rb-voe-ai-bootstrap-envelope-v1",
            "hostname": ai_secondary["hostname"],
            "device_id": ai_secondary["device_id"],
            "boot_id": ai_secondary["boot_id"],
            "snapshot": snapshots["ai_x5"],
        }
    )
    if ai_secondary["envelope_sha256"] != expected_ai_envelope_sha256:
        raise LiveShadowConfigError("bootstrap AI secondary envelope digest mismatch")
    if (
        not isinstance(embodied_secondary, Mapping)
        or set(embodied_secondary) != {"hostname", "device_id", "boot_id"}
        or embodied_secondary.get("hostname") != "embodied-x5"
        or embodied_secondary.get("device_id") != root["expected"]["embodied_device_id"]
        or embodied_secondary.get("boot_id") != root["expected"]["embodied_boot_id"]
    ):
        raise LiveShadowConfigError("bootstrap embodied secondary identity mismatch")
    return _ValidatedBootstrapBundle(
        manifest=MappingProxyType(dict(manifest)),
        raw_files=MappingProxyType(raw_files),
        manifest_sha256=manifest_sha256,
        manifest_file_sha256=hashlib.sha256(manifest_raw).hexdigest(),
    )


def _materialize_bootstrap_bundle(bundle: _ValidatedBootstrapBundle, *, output: Path) -> None:
    target = output / "bootstrap"
    target.mkdir(mode=0o700, exist_ok=False)
    for filename in sorted(bundle.raw_files):
        _write_new(target / filename, bundle.raw_files[filename])
    _write_new(
        target / "bootstrap_binding.json",
        _document(
            {
                "schema_version": LIVE_BOOTSTRAP_BINDING_SCHEMA,
                "trust_scope": "OPERATOR_ENROLLED_CURRENT_BOOT",
                "independent_release_authority": False,
                "manifest_sha256": bundle.manifest_sha256,
                "manifest_file_sha256": bundle.manifest_file_sha256,
                "config_sha256": bundle.manifest["config_sha256"],
                "snapshot_sha256": bundle.manifest["snapshot_sha256"],
                "captured_file_count": len(bundle.raw_files),
            }
        ),
    )


def _challenge_content_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("challenge_content_sha256", None)
    return canonical_sha256(unsigned)


def _a0_challenge_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "schema_version": "xrd-rb-voe-overhead-a0-challenge-v2",
            "acquisition_id": payload["acquisition_id"],
            "a0_run_id": payload["a0_run_id"],
            "r2_run_id": payload["run_id"],
            "r2_run_nonce_sha256": payload["run_nonce_sha256"],
            "challenge_issued_at_ms": payload["issued_at_ms"],
            "challenge_expires_at_ms": payload["expires_at_ms"],
            "release_id": payload["release_id"],
            "config_sha256": payload["config_sha256"],
            "case_id": payload["case_id"],
            "sample_id": payload["sample_id"],
            "sample_lineage_sha256": payload["sample_lineage_sha256"],
            "parent_evidence_root_sha256": payload["parent_evidence_root_sha256"],
            "bag_empty_baseline_sha256": payload["bag_empty_baseline_sha256"],
            "task_kind": payload["task_kind"],
            "result_schema": payload["result_schema"],
            "success_state": payload["success_state"],
        }
    )


def _issue_live_shadow_challenge_with_clock(
    config_path: str | Path,
    *,
    challenge_file: str | Path,
    case_id: str,
    sample_id: str,
    sample_lineage_sha256: str,
    parent_evidence_root_sha256: str,
    bag_empty_baseline_sha256: str | None = None,
    clock_ms: Callable[[], int] = _clock_ms,
) -> dict[str, Any]:
    """Test seam for deterministic challenge clocks; production uses the wrapper below."""

    plan = inspect_live_shadow_config(config_path)
    if plan["live_ready"] is not True:
        raise LiveShadowConfigError("live config identities must be frozen before issuing a run challenge")
    root = _read_config_payload(config_path)
    bootstrap_bundle = _validate_bootstrap_bundle(root, config_path=Path(config_path))
    known_hosts = _regular_absolute_file(
        root["known_hosts_file"],
        label="known_hosts_file",
    )
    _read_frozen_known_hosts(known_hosts)
    config_sha256 = _string(root["config_sha256"], label="config_sha256")
    release_id = _contract_token(root["release_id"], label="release_id")
    artifact_path = _new_challenge_path(challenge_file)
    selected_case_id = _contract_token(case_id, label="case_id")
    selected_sample_id = _contract_token(sample_id, label="sample_id")
    require_sha256("sample_lineage_sha256", sample_lineage_sha256)
    require_sha256("parent_evidence_root_sha256", parent_evidence_root_sha256)
    if bag_empty_baseline_sha256 is None:
        raise LiveShadowConfigError("bag-drop challenge requires a frozen empty-baseline digest")
    require_sha256("bag_empty_baseline_sha256", bag_empty_baseline_sha256)
    overhead = _exact_keys(root["overhead"], _OVERHEAD_KEYS, label="overhead")
    task_kind, result_schema, success_state = _overhead_task_contract(overhead)
    issued_at_ms = clock_ms()
    _non_negative_int(issued_at_ms, label="clock_ms result")
    run_id = _run_token(None, label="run_id", generated_prefix="R2-RUN-live")
    run_nonce = _run_token(None, label="run_nonce", generated_prefix="nonce")
    acquisition_id = _run_token(
        None,
        label="acquisition_id",
        generated_prefix="A0-ACQ-overhead",
    )
    a0_run_id = _run_token(None, label="a0_run_id", generated_prefix="A0-RUN-overhead")
    challenge: dict[str, Any] = {
        "schema_version": LIVE_SHADOW_CHALLENGE_SCHEMA,
        "config_sha256": config_sha256,
        "bootstrap_manifest_sha256": bootstrap_bundle.manifest_sha256,
        "bootstrap_manifest_file_sha256": bootstrap_bundle.manifest_file_sha256,
        "release_id": release_id,
        "known_hosts_file": str(known_hosts),
        "known_hosts_sha256": FROZEN_KNOWN_HOSTS_SHA256,
        "case_id": selected_case_id,
        "sample_id": selected_sample_id,
        "sample_lineage_sha256": sample_lineage_sha256,
        "parent_evidence_root_sha256": parent_evidence_root_sha256,
        "bag_empty_baseline_sha256": bag_empty_baseline_sha256,
        "task_kind": task_kind,
        "result_schema": result_schema,
        "success_state": success_state,
        "run_id": run_id,
        "run_nonce": run_nonce,
        "run_nonce_sha256": hashlib.sha256(run_nonce.encode("utf-8")).hexdigest(),
        "acquisition_id": acquisition_id,
        "a0_run_id": a0_run_id,
        "issued_at_ms": issued_at_ms,
        "expires_at_ms": issued_at_ms + LIVE_CHALLENGE_TTL_MS,
        "profile_sha256": dict(sorted(_profile_digests().items())),
        "purpose": LIVE_CHALLENGE_PURPOSE,
        "remote_contacted": False,
        "network_touched": False,
        "execution_authority": False,
        "transport_commands_issued": 0,
        "read_only_device_observations": 0,
        "actuator_commands_issued": 0,
        "mutating_commands_issued": 0,
        "read_only_transport_operations": 0,
    }
    challenge["challenge_sha256"] = _a0_challenge_sha256(challenge)
    challenge["challenge_content_sha256"] = _challenge_content_sha256(challenge)
    try:
        _write_new(artifact_path, _document(challenge))
        artifact_path.chmod(0o600)
        issuance = {
            "schema_version": LIVE_SHADOW_CHALLENGE_ISSUANCE_SCHEMA,
            "challenge_sha256": challenge["challenge_sha256"],
            "challenge_artifact_sha256": file_sha256(artifact_path),
            "config_sha256": config_sha256,
            "bootstrap_manifest_sha256": bootstrap_bundle.manifest_sha256,
            "bootstrap_manifest_file_sha256": bootstrap_bundle.manifest_file_sha256,
            "issued_at_ms": issued_at_ms,
        }
        issuance_path = _challenge_sidecar(artifact_path, "issued")
        _write_new(issuance_path, _document(issuance))
        issuance_path.chmod(0o600)
    except Exception:
        _challenge_sidecar(artifact_path, "issued").unlink(missing_ok=True)
        artifact_path.unlink(missing_ok=True)
        raise
    return challenge


def issue_live_shadow_challenge(
    config_path: str | Path,
    *,
    challenge_file: str | Path,
    case_id: str,
    sample_id: str,
    sample_lineage_sha256: str,
    parent_evidence_root_sha256: str,
    bag_empty_baseline_sha256: str | None = None,
) -> dict[str, Any]:
    """Persist a one-time, case-bound challenge using the host system clock."""

    return _issue_live_shadow_challenge_with_clock(
        config_path,
        challenge_file=challenge_file,
        case_id=case_id,
        sample_id=sample_id,
        sample_lineage_sha256=sample_lineage_sha256,
        parent_evidence_root_sha256=parent_evidence_root_sha256,
        bag_empty_baseline_sha256=bag_empty_baseline_sha256,
        clock_ms=_clock_ms,
    )


def _experiment_context(value: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    fields = {
        "case_id",
        "sample_id",
        "sample_lineage_sha256",
        "parent_evidence_root_sha256",
    }
    candidates: list[Mapping[str, Any]] = []
    if fields <= set(value):
        candidates.append(value)
    for container_name in ("experiment_case", "experiment_case_context"):
        nested = value.get(container_name)
        if isinstance(nested, Mapping) and fields <= set(nested):
            candidates.append(nested)
    if not candidates:
        raise LiveShadowConfigError(f"{label} lacks the explicit ExperimentCase context")
    normalized = {
        "case_id": _contract_token(candidates[0]["case_id"], label=f"{label}.case_id"),
        "sample_id": _contract_token(candidates[0]["sample_id"], label=f"{label}.sample_id"),
        "sample_lineage_sha256": _string(
            candidates[0]["sample_lineage_sha256"],
            label=f"{label}.sample_lineage_sha256",
        ),
        "parent_evidence_root_sha256": _string(
            candidates[0]["parent_evidence_root_sha256"],
            label=f"{label}.parent_evidence_root_sha256",
        ),
    }
    require_sha256(f"{label}.sample_lineage_sha256", normalized["sample_lineage_sha256"])
    require_sha256(
        f"{label}.parent_evidence_root_sha256",
        normalized["parent_evidence_root_sha256"],
    )
    for candidate in candidates[1:]:
        if any(candidate.get(field_name) != normalized[field_name] for field_name in fields):
            raise LiveShadowConfigError(f"{label} carries contradictory ExperimentCase contexts")
    return MappingProxyType(normalized)


def _resolve_a0_run_config(
    config: LiveShadowConfig,
    *,
    challenge: Mapping[str, Any],
    challenge_path: Path,
    challenge_artifact_sha256: str,
) -> LiveShadowConfig:
    reservation_payload, _ = _read_strict_json_artifact(
        _challenge_sidecar(challenge_path, "a0-reserved"),
        label="A0 challenge reservation",
    )
    reservation = _exact_keys(
        reservation_payload,
        _A0_RESERVATION_KEYS,
        label="A0 challenge reservation",
    )
    if reservation["schema_version"] != A0_CHALLENGE_RESERVATION_SCHEMA:
        raise LiveShadowConfigError("unsupported A0 challenge reservation schema")
    reservation_digest = _string(
        reservation["reservation_sha256"],
        label="A0 reservation_sha256",
    )
    require_sha256("A0 reservation_sha256", reservation_digest)
    unsigned = dict(reservation)
    unsigned.pop("reservation_sha256")
    if not secrets.compare_digest(canonical_sha256(unsigned), reservation_digest):
        raise LiveShadowConfigError("A0 challenge reservation digest mismatch")
    if (
        reservation["challenge_sha256"] != challenge["challenge_sha256"]
        or reservation["challenge_artifact_sha256"] != challenge_artifact_sha256
    ):
        raise LiveShadowConfigError("A0 challenge reservation does not bind the issued challenge")

    template_paths = {
        "record": str(config.overhead_record),
        "acquisition_manifest": str(config.overhead_acquisition_manifest),
        "raw_frame": str(config.overhead_raw_frame),
        "frame_bundle_artifact": str(config.overhead_frame_bundle_artifact),
        "result_json": str(config.overhead_result_json),
        "camera_service_identity_artifact": str(config.overhead_camera_service_identity_artifact),
        "capture_pipeline_artifact": str(config.overhead_capture_pipeline_artifact),
        "inference_pipeline_artifact": str(config.overhead_inference_pipeline_artifact),
        "replay_ledger_dir": str(config.overhead_replay_ledger_dir),
    }
    template_root = Path(_validate_a0_run_scope_template(template_paths))
    run_dir_raw = _lexical_absolute_path(reservation["run_dir"], label="A0 reservation.run_dir")
    run_dir = Path(run_dir_raw)
    if (
        run_dir.parent != template_root.parent
        or run_dir.name == A0_RUN_SCOPE_SENTINEL
        or _A0_RUN_SCOPE_RE.fullmatch(run_dir.name) is None
    ):
        raise LiveShadowConfigError("A0 reservation run_dir is outside the configured run-scope root")
    if not run_dir.is_dir() or _is_reparse_point(run_dir):
        raise LiveShadowConfigError("A0 reservation run_dir is not an existing regular directory")
    if any(parent.exists() and _is_reparse_point(parent) for parent in run_dir.parents):
        raise LiveShadowConfigError("A0 reservation run_dir cannot traverse a linked path")
    return replace(
        config,
        overhead_record=run_dir / _A0_OUTPUT_BASENAMES["record"],
        overhead_acquisition_manifest=run_dir / _A0_OUTPUT_BASENAMES["acquisition_manifest"],
        overhead_raw_frame=run_dir / _A0_OUTPUT_BASENAMES["raw_frame"],
        overhead_frame_bundle_artifact=run_dir / _A0_OUTPUT_BASENAMES["frame_bundle_artifact"],
        overhead_result_json=run_dir / _A0_OUTPUT_BASENAMES["result_json"],
        overhead_camera_service_identity_artifact=(
            run_dir / _A0_OUTPUT_BASENAMES["camera_service_identity_artifact"]
        ),
        overhead_capture_pipeline_artifact=run_dir / _A0_OUTPUT_BASENAMES["capture_pipeline_artifact"],
        overhead_inference_pipeline_artifact=(run_dir / _A0_OUTPUT_BASENAMES["inference_pipeline_artifact"]),
        overhead_replay_ledger_dir=run_dir / _A0_OUTPUT_BASENAMES["replay_ledger_dir"],
    )


def _validate_local_a0_challenge_binding(
    config: LiveShadowConfig,
    challenge: Mapping[str, Any],
    *,
    now_ms: int,
) -> Mapping[str, Any]:
    expected = {
        "acquisition_id": challenge["acquisition_id"],
        "a0_run_id": challenge["a0_run_id"],
        "r2_run_id": challenge["run_id"],
        "r2_run_nonce_sha256": challenge["run_nonce_sha256"],
        "challenge_sha256": challenge["challenge_sha256"],
        "challenge_issued_at_ms": challenge["issued_at_ms"],
        "challenge_expires_at_ms": challenge["expires_at_ms"],
        "release_id": challenge["release_id"],
        "config_sha256": challenge["config_sha256"],
        "bag_empty_baseline_sha256": challenge["bag_empty_baseline_sha256"],
        "task_kind": challenge["task_kind"],
        "result_schema": challenge["result_schema"],
        "success_state": challenge["success_state"],
    }
    expected_context = {
        field_name: challenge[field_name]
        for field_name in (
            "case_id",
            "sample_id",
            "sample_lineage_sha256",
            "parent_evidence_root_sha256",
        )
    }
    actual_record: Mapping[str, Any] | None = None
    for label, path in (
        ("A0 actual record", config.overhead_record),
        ("A0 acquisition manifest", config.overhead_acquisition_manifest),
    ):
        payload, _ = _read_strict_json_artifact(path, label=label)
        if label == "A0 actual record":
            actual_record = MappingProxyType(dict(payload))
        for field_name, expected_value in expected.items():
            if payload.get(field_name) != expected_value:
                raise LiveShadowConfigError(f"{label} {field_name} does not match the issued challenge")
        context = _experiment_context(payload, label=label)
        if dict(context) != expected_context:
            raise LiveShadowConfigError(f"{label} ExperimentCase context does not match the issued challenge")
    if actual_record is None:  # pragma: no cover - fixed loop invariant
        raise RuntimeError("A0 actual record was not validated")
    try:
        camera_service_file_name = _string(
            actual_record.get("camera_service_identity_file_name"),
            label="A0 record.camera_service_identity_file_name",
        )
        if Path(camera_service_file_name).name != camera_service_file_name or camera_service_file_name in {
            ".",
            "..",
        }:
            raise LiveShadowConfigError("A0 camera service identity must be a safe run-root basename")
        if (
            config.overhead_camera_service_identity_artifact.parent != config.overhead_record.parent
            or config.overhead_camera_service_identity_artifact.name != camera_service_file_name
        ):
            raise LiveShadowConfigError("A0 camera service identity path does not match the frozen config")
        camera_service_identity_artifact = config.overhead_camera_service_identity_artifact
        local_expected = a0_contract.ExpectedAcquisition(
            acquisition_id=str(challenge["acquisition_id"]),
            a0_run_id=str(challenge["a0_run_id"]),
            r2_run_id=str(challenge["run_id"]),
            r2_run_nonce=str(challenge["run_nonce"]),
            challenge_sha256=str(challenge["challenge_sha256"]),
            challenge_issued_at_ms=int(challenge["issued_at_ms"]),
            challenge_expires_at_ms=int(challenge["expires_at_ms"]),
            release_id=config.release_id,
            config_sha256=config.config_sha256,
            case_id=str(challenge["case_id"]),
            sample_id=str(challenge["sample_id"]),
            sample_lineage_sha256=str(challenge["sample_lineage_sha256"]),
            parent_evidence_root_sha256=str(challenge["parent_evidence_root_sha256"]),
            bag_empty_baseline_sha256=str(challenge["bag_empty_baseline_sha256"]),
            task_kind=config.overhead_task_kind,
            result_schema=config.overhead_result_schema,
            success_state=config.overhead_success_state,
            capture_device_id=str(actual_record["capture_device_id"]),
            capture_boot_id=str(actual_record["capture_boot_id"]),
            capture_session_id=str(actual_record["capture_session_id"]),
            inference_device_id=config.ai_device_id,
            inference_boot_id=config.ai_boot_id,
            inference_session_id=str(actual_record["ai_x5_session_id"]),
        )
        a0_contract._validate_sealed_record_with_clock(
            dict(actual_record),
            replay_ledger_dir=config.overhead_replay_ledger_dir,
            acquisition_manifest=config.overhead_acquisition_manifest,
            raw_frame=config.overhead_raw_frame,
            frame_bundle_artifact=config.overhead_frame_bundle_artifact,
            result_json=config.overhead_result_json,
            capture_pipeline_artifact=config.overhead_capture_pipeline_artifact,
            inference_pipeline_artifact=config.overhead_inference_pipeline_artifact,
            camera_service_identity_artifact=camera_service_identity_artifact,
            expected=local_expected,
            now_ms=now_ms,
            max_age_ms=LIVE_CHALLENGE_TTL_MS,
        )
    except (a0_contract.ActualRecordError, KeyError, OSError, TypeError, ValueError) as exc:
        raise LiveShadowConfigError(f"local A0 evidence reconstruction failed: {exc}") from exc
    return actual_record


def _stable_artifact_sha256(path: Path, *, label: str) -> str:
    raw = _read_stable_regular_bytes(
        path,
        label=label,
        max_bytes=64 * 1024 * 1024,
    )
    return hashlib.sha256(raw).hexdigest()


def _a0_production_root(config: LiveShadowConfig) -> Path:
    roots = {
        config.overhead_record.parent,
        config.overhead_acquisition_manifest.parent,
        config.overhead_raw_frame.parent,
        config.overhead_frame_bundle_artifact.parent,
        config.overhead_result_json.parent,
        config.overhead_camera_service_identity_artifact.parent,
        config.overhead_capture_pipeline_artifact.parent,
        config.overhead_inference_pipeline_artifact.parent,
        config.overhead_replay_ledger_dir.parent,
    }
    if len(roots) != 1:
        raise LiveShadowConfigError("A0 production artifacts do not share one frozen run root")
    root = next(iter(roots))
    if not root.is_absolute() or not root.is_dir() or _is_reparse_point(root):
        raise LiveShadowConfigError("A0 production run root is not a regular absolute directory")
    if any(parent.exists() and _is_reparse_point(parent) for parent in root.parents):
        raise LiveShadowConfigError("A0 production run root cannot traverse a linked path")
    return root


def _validate_a0_produced_sidecar(
    config: LiveShadowConfig,
    *,
    challenge: Mapping[str, Any],
    challenge_path: Path,
    challenge_artifact_sha256: str,
    actual_record: Mapping[str, Any],
) -> Mapping[str, Any]:
    reservation_path = _challenge_sidecar(challenge_path, "a0-reserved")
    reservation_payload, _ = _read_strict_json_artifact(
        reservation_path,
        label="A0 challenge reservation",
    )
    reservation = _exact_keys(
        reservation_payload,
        _A0_RESERVATION_KEYS,
        label="A0 challenge reservation",
    )
    if reservation["schema_version"] != A0_CHALLENGE_RESERVATION_SCHEMA:
        raise LiveShadowConfigError("unsupported A0 challenge reservation schema")
    reservation_digest = _string(
        reservation["reservation_sha256"],
        label="A0 reservation_sha256",
    )
    require_sha256("A0 reservation_sha256", reservation_digest)
    unsigned_reservation = dict(reservation)
    unsigned_reservation.pop("reservation_sha256")
    if not secrets.compare_digest(canonical_sha256(unsigned_reservation), reservation_digest):
        raise LiveShadowConfigError("A0 challenge reservation digest mismatch")

    production_root = _a0_production_root(config)
    run_dir = _string(reservation["run_dir"], label="A0 reservation.run_dir")
    if run_dir != str(production_root):
        raise LiveShadowConfigError("A0 challenge reservation differs from the production run root")
    run_binding_sha256 = _string(
        reservation["run_binding_sha256"],
        label="A0 reservation.run_binding_sha256",
    )
    require_sha256("A0 reservation.run_binding_sha256", run_binding_sha256)
    reservation_nonce = _string(
        reservation["reservation_nonce"],
        label="A0 reservation.reservation_nonce",
    )
    if re.fullmatch(r"[0-9a-f]{32}", reservation_nonce) is None:
        raise LiveShadowConfigError("A0 reservation nonce is not the runner's fixed 128-bit form")
    reserved_at_ms = _non_negative_int(
        reservation["reserved_at_ms"],
        label="A0 reservation.reserved_at_ms",
    )
    if not (int(challenge["issued_at_ms"]) <= reserved_at_ms < int(challenge["expires_at_ms"])):
        raise LiveShadowConfigError("A0 challenge reservation timestamp is outside the challenge window")
    expected_reservation = {
        "challenge_sha256": challenge["challenge_sha256"],
        "challenge_artifact_sha256": challenge_artifact_sha256,
    }
    if any(reservation[name] != value for name, value in expected_reservation.items()):
        raise LiveShadowConfigError("A0 challenge reservation does not bind the issued challenge")

    produced_path = _challenge_sidecar(challenge_path, "a0-produced")
    produced_payload, _ = _read_strict_json_artifact(
        produced_path,
        label="A0 produced receipt",
    )
    produced = _exact_keys(
        produced_payload,
        _A0_PRODUCED_KEYS,
        label="A0 produced receipt",
    )
    if produced["schema_version"] != A0_CHALLENGE_PRODUCED_SCHEMA:
        raise LiveShadowConfigError("unsupported A0 produced receipt schema")
    produced_digest = _string(produced["produced_sha256"], label="A0 produced_sha256")
    require_sha256("A0 produced_sha256", produced_digest)
    unsigned_produced = dict(produced)
    unsigned_produced.pop("produced_sha256")
    if not secrets.compare_digest(canonical_sha256(unsigned_produced), produced_digest):
        raise LiveShadowConfigError("A0 produced receipt digest mismatch")
    for field_name in (
        "run_binding_sha256",
        "reservation_sha256",
        "frame_bundle_file_sha256",
        "acquisition_manifest_sha256",
        "record_file_sha256",
        "replay_receipt_file_sha256",
    ):
        require_sha256(
            f"A0 produced.{field_name}",
            _string(produced[field_name], label=f"A0 produced.{field_name}"),
        )
    expected_produced_binding = {
        "challenge_sha256": challenge["challenge_sha256"],
        "challenge_artifact_sha256": challenge_artifact_sha256,
        "run_dir": run_dir,
        "run_binding_sha256": run_binding_sha256,
        "reservation_sha256": reservation_digest,
    }
    if any(produced[name] != value for name, value in expected_produced_binding.items()):
        raise LiveShadowConfigError("A0 produced receipt does not bind its challenge reservation")

    replay_receipt = a0_contract.replay_receipt_path(
        config.overhead_replay_ledger_dir,
        actual_record,
    )
    expected_disk_hashes = {
        "frame_bundle_file_sha256": _stable_artifact_sha256(
            config.overhead_frame_bundle_artifact,
            label="A0 input frame bundle",
        ),
        "acquisition_manifest_sha256": _stable_artifact_sha256(
            config.overhead_acquisition_manifest,
            label="A0 acquisition manifest",
        ),
        "record_file_sha256": _stable_artifact_sha256(
            config.overhead_record,
            label="A0 actual record",
        ),
        "replay_receipt_file_sha256": _stable_artifact_sha256(
            replay_receipt,
            label="A0 replay receipt",
        ),
    }
    if any(produced[name] != value for name, value in expected_disk_hashes.items()):
        raise LiveShadowConfigError("A0 produced receipt differs from complete A0 disk evidence")
    if (
        actual_record.get("task_kind"),
        actual_record.get("result_schema"),
        actual_record.get("success_state"),
    ) != (
        challenge["task_kind"],
        challenge["result_schema"],
        challenge["success_state"],
    ):
        raise LiveShadowConfigError("A0 produced evidence task tuple differs from the challenge")

    produced_at_ms = _non_negative_int(
        produced["produced_at_ms"],
        label="A0 produced.produced_at_ms",
    )
    if not (reserved_at_ms <= produced_at_ms <= int(challenge["expires_at_ms"])):
        raise LiveShadowConfigError("A0 produced timestamp is outside the reserved challenge window")
    return MappingProxyType(dict(produced))


def _validate_consumed_sidecar(
    challenge: _ValidatedChallenge,
    *,
    required: bool,
) -> Mapping[str, Any] | None:
    consumed_path = _challenge_sidecar(challenge.path, "consumed")
    if not consumed_path.exists() and not consumed_path.is_symlink():
        if required:
            raise LiveShadowConfigError("challenge consumption receipt disappeared during recovery")
        return None
    payload, _ = _read_strict_json_artifact(
        consumed_path,
        label="challenge consumption receipt",
    )
    receipt = _exact_keys(
        payload,
        _CHALLENGE_CONSUMPTION_KEYS,
        label="challenge consumption receipt",
    )
    consumed_at_ms = _non_negative_int(receipt["consumed_at_ms"], label="consumed_at_ms")
    expected = {
        "schema_version": LIVE_SHADOW_CHALLENGE_CONSUMPTION_SCHEMA,
        "challenge_sha256": challenge.challenge_sha256,
        "challenge_artifact_sha256": challenge.artifact_sha256,
        "config_sha256": challenge.artifact["config_sha256"],
        "bootstrap_manifest_sha256": challenge.artifact["bootstrap_manifest_sha256"],
        "bootstrap_manifest_file_sha256": challenge.artifact["bootstrap_manifest_file_sha256"],
        "consumed_at_ms": consumed_at_ms,
    }
    if receipt != expected:
        raise LiveShadowConfigError("challenge consumption receipt does not bind the attempt")
    produced_at_ms = int(challenge.a0_produced["produced_at_ms"])
    if consumed_at_ms < produced_at_ms:
        raise LiveShadowConfigError("challenge was consumed before A0 production completed")
    if consumed_at_ms > int(challenge.artifact["expires_at_ms"]):
        raise LiveShadowConfigError("challenge consumption timestamp is outside the challenge window")
    return MappingProxyType(dict(receipt))


def _recover_concurrently_published_consumed_sidecar(
    challenge: _ValidatedChallenge,
) -> Mapping[str, Any]:
    transient_fragments = (
        "size is invalid",
        "changed during read",
        "disappeared during recovery",
        "must be an existing absolute regular non-link file",
    )
    last_error: LiveShadowConfigError | None = None
    for _ in range(50):
        try:
            receipt = _validate_consumed_sidecar(challenge, required=True)
        except LiveShadowConfigError as exc:
            if not any(fragment in str(exc) for fragment in transient_fragments):
                raise
            last_error = exc
            time.sleep(0.01)
            continue
        if receipt is None:  # pragma: no cover - required invariant
            raise RuntimeError("required challenge consumption receipt was not recovered")
        return receipt
    raise LiveShadowConfigError(
        "concurrent challenge consumption receipt did not become durable"
    ) from last_error


def _validate_challenge(
    config: LiveShadowConfig,
    challenge_file: str | Path,
    *,
    bootstrap_bundle: _ValidatedBootstrapBundle,
    now_ms: int,
) -> _ValidatedChallenge:
    path = _regular_absolute_file(
        str(challenge_file),
        label="challenge_file",
        suffixes=(".json",),
    )
    consumed_path = _challenge_sidecar(path, "consumed")
    consumed_present = consumed_path.exists() or consumed_path.is_symlink()
    payload, raw = _read_strict_json_artifact(path, label="challenge artifact")
    challenge = _exact_keys(payload, _CHALLENGE_KEYS, label="challenge")
    if challenge["schema_version"] != LIVE_SHADOW_CHALLENGE_SCHEMA:
        raise LiveShadowConfigError("unsupported live-shadow challenge schema")
    claimed_digest = _string(challenge["challenge_sha256"], label="challenge_sha256")
    claimed_content_digest = _string(
        challenge["challenge_content_sha256"],
        label="challenge_content_sha256",
    )
    require_sha256("challenge_sha256", claimed_digest)
    require_sha256("challenge_content_sha256", claimed_content_digest)
    if not secrets.compare_digest(claimed_digest, _a0_challenge_sha256(challenge)):
        raise LiveShadowConfigError("challenge_sha256 does not bind the A0 challenge")
    if not secrets.compare_digest(claimed_content_digest, _challenge_content_sha256(challenge)):
        raise LiveShadowConfigError("challenge_content_sha256 does not bind the persisted challenge")
    if challenge["config_sha256"] != config.config_sha256:
        raise LiveShadowConfigError("challenge belongs to a different live config")
    if (
        challenge["bootstrap_manifest_sha256"] != bootstrap_bundle.manifest_sha256
        or challenge["bootstrap_manifest_file_sha256"] != bootstrap_bundle.manifest_file_sha256
    ):
        raise LiveShadowConfigError("challenge belongs to a different bootstrap manifest")
    if challenge["release_id"] != config.release_id:
        raise LiveShadowConfigError("challenge release differs from the live config")
    if (
        challenge["known_hosts_file"] != str(config.known_hosts_file)
        or challenge["known_hosts_sha256"] != config.known_hosts_sha256
    ):
        raise LiveShadowConfigError("challenge known_hosts binding mismatch")
    _read_frozen_known_hosts(config.known_hosts_file)
    for field_name in ("case_id", "sample_id"):
        _contract_token(challenge[field_name], label=f"challenge.{field_name}")
    for field_name in ("sample_lineage_sha256", "parent_evidence_root_sha256"):
        require_sha256(f"challenge.{field_name}", challenge[field_name])
    require_sha256(
        "challenge.bag_empty_baseline_sha256",
        challenge["bag_empty_baseline_sha256"],
    )
    challenge_task = (
        challenge["task_kind"],
        challenge["result_schema"],
        challenge["success_state"],
    )
    config_task = (
        config.overhead_task_kind,
        config.overhead_result_schema,
        config.overhead_success_state,
    )
    if challenge_task != config_task:
        raise LiveShadowConfigError("challenge task/result tuple differs from the live config")
    run_id = _run_token(challenge["run_id"], label="run_id", generated_prefix="unused")
    run_nonce = _run_token(challenge["run_nonce"], label="run_nonce", generated_prefix="unused")
    for field_name in ("acquisition_id", "a0_run_id"):
        _contract_token(challenge[field_name], label=f"challenge.{field_name}")
    if challenge["run_nonce_sha256"] != hashlib.sha256(run_nonce.encode("utf-8")).hexdigest():
        raise LiveShadowConfigError("challenge run nonce digest mismatch")
    del run_id
    issued_at_ms = _non_negative_int(challenge["issued_at_ms"], label="issued_at_ms")
    expires_at_ms = _non_negative_int(challenge["expires_at_ms"], label="expires_at_ms")
    if expires_at_ms != issued_at_ms + LIVE_CHALLENGE_TTL_MS:
        raise LiveShadowConfigError("challenge expiry does not match the fixed TTL")
    if now_ms < issued_at_ms:
        raise LiveShadowConfigError("challenge was issued in the future")
    if now_ms >= expires_at_ms:
        raise LiveShadowConfigError("challenge expired before live execution")
    profiles = _exact_keys(
        challenge["profile_sha256"],
        _CHALLENGE_PROFILE_KEYS,
        label="challenge.profile_sha256",
    )
    for subsystem, digest in profiles.items():
        require_sha256(f"challenge.profile_sha256.{subsystem}", digest)
    if profiles != dict(_profile_digests()):
        raise LiveShadowConfigError("challenge semantic profiles differ from the packaged release")
    if challenge["purpose"] != LIVE_CHALLENGE_PURPOSE:
        raise LiveShadowConfigError("challenge purpose is invalid")
    for field_name in (
        "remote_contacted",
        "network_touched",
        "execution_authority",
    ):
        if challenge[field_name] is not False:
            raise LiveShadowConfigError(f"challenge {field_name} must be false at issuance")
    for field_name in (
        "transport_commands_issued",
        "read_only_device_observations",
        "actuator_commands_issued",
        "mutating_commands_issued",
        "read_only_transport_operations",
    ):
        if isinstance(challenge[field_name], bool) or challenge[field_name] != 0:
            raise LiveShadowConfigError(f"challenge {field_name} must be integer zero at issuance")

    issuance_path = _challenge_sidecar(path, "issued")
    expected_issuance = {
        "schema_version": LIVE_SHADOW_CHALLENGE_ISSUANCE_SCHEMA,
        "challenge_sha256": claimed_digest,
        "challenge_artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "config_sha256": config.config_sha256,
        "bootstrap_manifest_sha256": bootstrap_bundle.manifest_sha256,
        "bootstrap_manifest_file_sha256": bootstrap_bundle.manifest_file_sha256,
        "issued_at_ms": issued_at_ms,
    }
    if issuance_path.exists() or issuance_path.is_symlink():
        issuance_payload, _ = _read_strict_json_artifact(
            issuance_path,
            label="challenge issuance receipt",
        )
        issuance = _exact_keys(
            issuance_payload,
            _CHALLENGE_ISSUANCE_KEYS,
            label="challenge issuance receipt",
        )
        if issuance != expected_issuance:
            raise LiveShadowConfigError("challenge issuance receipt does not bind the artifact")
    elif not consumed_present:
        raise LiveShadowConfigError("challenge issuance receipt is missing before consumption")
    a0_config = _resolve_a0_run_config(
        config,
        challenge=challenge,
        challenge_path=path,
        challenge_artifact_sha256=expected_issuance["challenge_artifact_sha256"],
    )
    a0_record = _validate_local_a0_challenge_binding(
        a0_config,
        challenge,
        now_ms=now_ms,
    )
    a0_produced = _validate_a0_produced_sidecar(
        a0_config,
        challenge=challenge,
        challenge_path=path,
        challenge_artifact_sha256=expected_issuance["challenge_artifact_sha256"],
        actual_record=a0_record,
    )
    validated = _ValidatedChallenge(
        artifact=MappingProxyType(dict(challenge)),
        artifact_sha256=expected_issuance["challenge_artifact_sha256"],
        path=path,
        a0_config=a0_config,
        a0_record=a0_record,
        a0_produced=a0_produced,
    )
    if consumed_present:
        _validate_consumed_sidecar(validated, required=True)
        raise LiveShadowConfigError("challenge replay rejected: already consumed")
    return validated


def _consume_challenge(
    config: LiveShadowConfig,
    challenge: _ValidatedChallenge,
    *,
    consumed_at_ms: int,
) -> None:
    fresh_produced = _validate_a0_produced_sidecar(
        config,
        challenge=challenge.artifact,
        challenge_path=challenge.path,
        challenge_artifact_sha256=challenge.artifact_sha256,
        actual_record=challenge.a0_record,
    )
    if dict(fresh_produced) != dict(challenge.a0_produced):
        raise LiveShadowConfigError("A0 produced receipt changed before challenge consumption")
    produced_at_ms = int(fresh_produced["produced_at_ms"])
    if consumed_at_ms < produced_at_ms:
        raise LiveShadowConfigError("challenge cannot be consumed before A0 production completed")
    if consumed_at_ms > int(challenge.artifact["expires_at_ms"]):
        raise LiveShadowConfigError("challenge cannot be consumed after expiry")
    receipt = {
        "schema_version": LIVE_SHADOW_CHALLENGE_CONSUMPTION_SCHEMA,
        "challenge_sha256": challenge.challenge_sha256,
        "challenge_artifact_sha256": challenge.artifact_sha256,
        "config_sha256": challenge.artifact["config_sha256"],
        "bootstrap_manifest_sha256": challenge.artifact["bootstrap_manifest_sha256"],
        "bootstrap_manifest_file_sha256": challenge.artifact["bootstrap_manifest_file_sha256"],
        "consumed_at_ms": consumed_at_ms,
    }
    consumed_path = _challenge_sidecar(challenge.path, "consumed")
    try:
        _write_new(consumed_path, _document(receipt))
        consumed_path.chmod(0o600)
    except FileExistsError as exc:
        _recover_concurrently_published_consumed_sidecar(challenge)
        raise LiveShadowConfigError("challenge replay rejected: already consumed") from exc
    persisted = _validate_consumed_sidecar(challenge, required=True)
    if persisted is None or dict(persisted) != receipt:  # pragma: no cover - required invariant
        raise RuntimeError("persisted challenge consumption receipt changed after exclusive write")
    issuance_path = _challenge_sidecar(challenge.path, "issued")
    try:
        issuance_path.unlink()
        _fsync_directory(issuance_path.parent)
    except FileNotFoundError:
        pass
    except OSError:
        raise


def _require_ai_x5_identity(config: LiveShadowConfig) -> None:
    if platform.node().strip().casefold() != "xrd-ai":
        raise RuntimeError("live shadow can run only on hostname xrd-ai")
    if local_device_id() != config.ai_device_id:
        raise RuntimeError("live shadow local AI X5 device identity mismatch")
    if local_boot_id() != config.ai_boot_id:
        raise RuntimeError("live shadow local AI X5 boot identity mismatch")


def _run_live_shadow_impl(
    config: LiveShadowConfig,
    *,
    output_dir: str | Path,
    challenge_file: str | Path | None = None,
    run_id: str | None = None,
    run_nonce: str | None = None,
    clock_ms: Callable[[], int] = _clock_ms,
    _failure_state: _LiveFailureState,
) -> LiveShadowRunResult:
    """Capture the fixed topology once, evaluate once, and seal truthful evidence."""

    if not isinstance(config, LiveShadowConfig):
        raise TypeError("config must be loaded with load_live_shadow_config")
    if run_id is not None or run_nonce is not None:
        raise LiveShadowConfigError(
            "hand-entered run identifiers are forbidden; consume an issued challenge artifact"
        )
    if challenge_file is None:
        raise LiveShadowConfigError("run_live_shadow requires an issued challenge artifact")
    _require_ai_x5_identity(config)
    _available_output_dir(output_dir)
    bootstrap_bundle = _validate_bootstrap_bundle(
        _read_config_payload(config.source_path),
        config_path=config.source_path,
    )
    challenge_checked_at_ms = clock_ms()
    _non_negative_int(challenge_checked_at_ms, label="clock_ms result")
    challenge = _validate_challenge(
        config,
        challenge_file,
        bootstrap_bundle=bootstrap_bundle,
        now_ms=challenge_checked_at_ms,
    )
    config = challenge.a0_config
    _failure_state.challenge = challenge
    _failure_state.failure_phase = "OUTPUT_RESERVATION"
    output = _reserve_output_dir(output_dir)
    _failure_state.output = output
    _failure_state.failure_phase = "BOOTSTRAP_EVIDENCE_MATERIALIZATION"
    _materialize_bootstrap_bundle(bootstrap_bundle, output=output)
    _failure_state.failure_phase = "RUN_LOCAL_KNOWN_HOSTS"
    run_known_hosts = _materialize_run_known_hosts(config, output=output)
    _failure_state.run_known_hosts_file = run_known_hosts
    _failure_state.failure_phase = "ATTEMPT_START_RECORD"
    _write_attempt_started(
        output=output,
        config=config,
        challenge=challenge,
        attempt_started_at_ms=challenge_checked_at_ms,
    )
    selected_run_id = str(challenge.artifact["run_id"])
    selected_nonce = str(challenge.artifact["run_nonce"])
    profile_sha256 = MappingProxyType(dict(challenge.artifact["profile_sha256"]))

    source_seed = _LiveSourceSeed(
        run_id=selected_run_id,
        run_nonce=selected_nonce,
        release_id=config.release_id,
        profile_sha256=profile_sha256,
    )
    preflight_attempts = _failure_state.preflight_attempts
    _failure_state.failure_phase = "REMOTE_COLLECTOR_PREFLIGHT"
    try:
        collector_preflights = _preflight_remote_collectors(
            config,
            known_hosts_file=run_known_hosts,
            attempt_log=preflight_attempts,
        )
    except Exception as exc:
        if preflight_attempts:
            try:
                _seal_preflight_failure_attempt(
                    output=output,
                    config=config,
                    challenge=challenge,
                    run_known_hosts_file=run_known_hosts,
                    attempts=preflight_attempts,
                    error=exc,
                )
            except Exception as seal_exc:
                raise RuntimeError(
                    "remote collector preflight failed and local failure evidence could not be sealed"
                ) from seal_exc
        raise
    challenge_consumed_at_ms = clock_ms()
    _non_negative_int(challenge_consumed_at_ms, label="clock_ms result")
    _failure_state.failure_phase = "CHALLENGE_CONSUMPTION"
    _consume_challenge(config, challenge, consumed_at_ms=challenge_consumed_at_ms)
    _failure_state.failure_phase = "CHALLENGE_CONSUMPTION_RECEIPT"
    _materialize_challenge_consumption_receipt(challenge, output=output)
    _failure_state.failure_phase = "READ_ONLY_SOURCE_CONSTRUCTION"
    sources, paths = _construct_sources(
        config,
        source_seed,
        known_hosts_file=run_known_hosts,
    )
    _failure_state.failure_phase = "READ_ONLY_SNAPSHOT_CAPTURE"
    _failure_state.snapshot_transport_attempts = 4
    prefetched = _prefetch_sources(sources, paths)

    _failure_state.failure_phase = "READ_ONLY_EVALUATION"
    vision_source = FileJsonSnapshotTransport(
        config.overhead_record,
        allowed_path=DUAL_ARM_VISION_RECORD_PATH,
    )
    vision_capture = PrefetchedJsonSnapshotTransport(
        vision_source,
        path=DUAL_ARM_VISION_RECORD_PATH,
    )
    acquisition_path = "/rb-voe/dual-arm/overhead/a0-acquisition"
    acquisition_capture = PrefetchedJsonSnapshotTransport(
        FileJsonSnapshotTransport(
            config.overhead_acquisition_manifest,
            allowed_path=acquisition_path,
        ),
        path=acquisition_path,
    )
    all_captures: dict[str, PrefetchedJsonSnapshotTransport] = {
        **prefetched,
        "overhead_record": vision_capture,
        "overhead_acquisition": acquisition_capture,
    }
    central_errors: dict[str, str] = {}
    record_payload = vision_capture.payload
    if record_payload is not None:
        try:
            receipt_file = a0_contract.replay_receipt_path(
                config.overhead_replay_ledger_dir,
                record_payload,
            )
            receipt_path = "/rb-voe/dual-arm/overhead/a0-consumption-receipt"
            all_captures["overhead_consumption_receipt"] = PrefetchedJsonSnapshotTransport(
                FileJsonSnapshotTransport(receipt_file, allowed_path=receipt_path),
                path=receipt_path,
            )
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            central_errors["overhead_consumption_receipt"] = "CAPTURE_UNAVAILABLE"

    evaluated_at_ms = clock_ms()
    if isinstance(evaluated_at_ms, bool) or not isinstance(evaluated_at_ms, int) or evaluated_at_ms < 0:
        raise ValueError("clock_ms must return a non-negative integer")
    binding = ShadowRunBinding(
        run_id=selected_run_id,
        run_nonce=selected_nonce,
        release_id=config.release_id,
        evaluated_at_ms=evaluated_at_ms,
        mode=ShadowMode.LIVE_READONLY_SHADOW,
        profile_sha256=profile_sha256,
    )
    if any(
        source_seed.source_binding_sha256(subsystem) != binding.source_binding_sha256(subsystem)
        for subsystem in profile_sha256
    ):
        raise RuntimeError("source binding changed when the evaluation clock was selected")

    ai_adapter = AiX5ReadOnlyAdapter(
        prefetched["ai_x5"],
        AiX5CapabilityBinding(
            device_id=config.ai_device_id,
            release_id=config.release_id,
            required_backends=AI_X5_REQUIRED_BACKENDS,
            snapshot_max_age_ms=LIVE_SNAPSHOT_MAX_AGE_MS,
            run_binding_sha256=binding.source_binding_sha256("ai_x5"),
            profile_sha256=profile_sha256["ai_x5"],
            expected_runtime_artifact_set_sha256=config.ai_runtime_artifact_set_sha256,
            expected_boot_id=config.ai_boot_id,
        ),
    )
    embodied_adapter = EmbodiedX5ReadOnlyAdapter(
        prefetched["embodied_x5"],
        EmbodiedX5CapabilityBinding(
            device_id=config.embodied_device_id,
            run_id=binding.run_id,
            run_nonce=binding.run_nonce,
            release_id=binding.release_id,
            run_binding_sha256=binding.source_binding_sha256("embodied_x5"),
            profile_sha256=profile_sha256["embodied_x5"],
            required_artifact_sha256=config.embodied_required_artifact_sha256,
            expected_boot_id=config.embodied_boot_id,
            snapshot_max_age_ms=LIVE_SNAPSHOT_MAX_AGE_MS,
        ),
    )
    dual_arm_adapter = DualArmReadOnlyAdapter(
        arm01_transport=prefetched["arm01"],
        arm02_transport=prefetched["arm02"],
        vision_transport=vision_capture,
        binding=DualArmCapabilityBinding(
            run_id=binding.run_id,
            run_nonce=binding.run_nonce,
            release_id=binding.release_id,
            run_binding_sha256=binding.source_binding_sha256("dual_arm"),
            profile_sha256=profile_sha256["dual_arm"],
            expected_machine_id_sha256=config.arm_machine_id_sha256,
            expected_boot_id=config.arm_boot_id,
            probe_script_sha256=config.arm_probe_script_sha256,
            ai_x5_device_id=config.ai_device_id,
            ai_x5_boot_id=config.ai_boot_id,
            vision_acquisition_id=str(challenge.artifact["acquisition_id"]),
            vision_a0_run_id=str(challenge.artifact["a0_run_id"]),
            vision_capture_session_id=str(challenge.a0_record["capture_session_id"]),
            vision_inference_session_id=str(challenge.a0_record["ai_x5_session_id"]),
            vision_challenge_sha256=challenge.challenge_sha256,
            vision_challenge_issued_at_ms=int(challenge.artifact["issued_at_ms"]),
            vision_challenge_expires_at_ms=int(challenge.artifact["expires_at_ms"]),
            vision_config_sha256=config.config_sha256,
            vision_case_id=str(challenge.artifact["case_id"]),
            vision_sample_id=str(challenge.artifact["sample_id"]),
            vision_sample_lineage_sha256=str(challenge.artifact["sample_lineage_sha256"]),
            vision_parent_evidence_root_sha256=str(challenge.artifact["parent_evidence_root_sha256"]),
            vision_bag_empty_baseline_sha256=challenge.artifact["bag_empty_baseline_sha256"],
            vision_task_kind=str(challenge.artifact["task_kind"]),
            vision_result_schema=str(challenge.artifact["result_schema"]),
            vision_success_state=str(challenge.artifact["success_state"]),
            vision_acquisition_manifest=config.overhead_acquisition_manifest,
            vision_raw_frame=config.overhead_raw_frame,
            vision_frame_bundle_artifact=config.overhead_frame_bundle_artifact,
            vision_result_json=config.overhead_result_json,
            vision_camera_service_identity_artifact=(config.overhead_camera_service_identity_artifact),
            vision_capture_pipeline_artifact=config.overhead_capture_pipeline_artifact,
            vision_inference_pipeline_artifact=(config.overhead_inference_pipeline_artifact),
            vision_replay_ledger_dir=config.overhead_replay_ledger_dir,
        ),
    )
    report = ShadowCoordinator(
        {
            "ai_x5": ai_adapter,
            "embodied_x5": embodied_adapter,
            "dual_arm": dual_arm_adapter,
            "assay_station": AssayStationTargetAdapter(),
        }
    ).evaluate(binding)
    _failure_state.failure_phase = "EVIDENCE_SEAL"
    root_sha256 = _seal_evidence(
        output=output,
        config=config,
        binding=binding,
        report=report,
        captures=all_captures,
        central_capture_errors=central_errors,
        challenge=challenge,
        collector_preflights=collector_preflights,
        run_known_hosts_file=run_known_hosts,
        bootstrap_bundle=bootstrap_bundle,
    )
    _failure_state.failure_phase = "COMPLETED"
    return LiveShadowRunResult(
        report=report,
        binding=binding,
        output_dir=output,
        evidence_root_sha256=root_sha256,
        transport_commands_issued=(
            len(collector_preflights)
            + sum(int(all_captures[name].network_touched) for name in ("embodied_x5", "arm01", "arm02"))
        ),
        read_only_device_observations=(
            sum(
                int(all_captures[name].payload is not None)
                for name in ("ai_x5", "embodied_x5", "arm01", "arm02")
            )
            + int(all_captures["overhead_record"].payload is not None)
        ),
    )


def _run_live_shadow_with_clock(
    config: LiveShadowConfig,
    *,
    output_dir: str | Path,
    challenge_file: str | Path | None = None,
    run_id: str | None = None,
    run_nonce: str | None = None,
    clock_ms: Callable[[], int] = _clock_ms,
) -> LiveShadowRunResult:
    """Test seam for deterministic clocks; production uses the wrapper below."""

    state = _LiveFailureState()
    try:
        return _run_live_shadow_impl(
            config,
            output_dir=output_dir,
            challenge_file=challenge_file,
            run_id=run_id,
            run_nonce=run_nonce,
            clock_ms=clock_ms,
            _failure_state=state,
        )
    except Exception as exc:
        output = state.output
        evidence_index = None if output is None else output / "evidence_index.json"
        if (
            output is not None
            and state.challenge is not None
            and evidence_index is not None
            and not evidence_index.exists()
            and not evidence_index.is_symlink()
        ):
            try:
                _seal_live_failure_attempt(
                    config=config,
                    state=state,
                    error=exc,
                )
            except Exception as seal_exc:
                raise RuntimeError(
                    "live shadow failed and local failure evidence could not be sealed"
                ) from seal_exc
        raise


def run_live_shadow(
    config: LiveShadowConfig,
    *,
    output_dir: str | Path,
    challenge_file: str | Path | None = None,
    run_id: str | None = None,
    run_nonce: str | None = None,
) -> LiveShadowRunResult:
    """Capture the fixed topology using the host clock and seal fail-closed evidence."""

    return _run_live_shadow_with_clock(
        config,
        output_dir=output_dir,
        challenge_file=challenge_file,
        run_id=run_id,
        run_nonce=run_nonce,
        clock_ms=_clock_ms,
    )


def _emit(value: object, *, stream=None) -> None:  # noqa: ANN001
    print(
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
        file=sys.stdout if stream is None else stream,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-config", action="store_true")
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--issue-challenge", action="store_true")
    mode.add_argument("--run-live", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--challenge-file", type=Path)
    parser.add_argument("--case-id")
    parser.add_argument("--sample-id")
    parser.add_argument("--sample-lineage-sha256")
    parser.add_argument("--parent-evidence-root-sha256")
    parser.add_argument("--bag-empty-baseline-sha256")
    parser.add_argument("--run-id")
    parser.add_argument("--run-nonce")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config_path = args.config.resolve(strict=True)
        case_arguments = (
            args.case_id,
            args.sample_id,
            args.sample_lineage_sha256,
            args.parent_evidence_root_sha256,
        )
        if args.issue_challenge:
            if args.output_dir is not None or args.run_id is not None or args.run_nonce is not None:
                raise LiveShadowConfigError("run-only arguments cannot be supplied while issuing a challenge")
            if args.challenge_file is None or any(value is None for value in case_arguments):
                raise LiveShadowConfigError(
                    "challenge issuance requires --challenge-file and explicit ExperimentCase fields"
                )
            _emit(
                issue_live_shadow_challenge(
                    config_path,
                    challenge_file=args.challenge_file.resolve(),
                    case_id=args.case_id,
                    sample_id=args.sample_id,
                    sample_lineage_sha256=args.sample_lineage_sha256,
                    parent_evidence_root_sha256=args.parent_evidence_root_sha256,
                    bag_empty_baseline_sha256=args.bag_empty_baseline_sha256,
                )
            )
            return 0
        if not args.run_live and not args.validate_config:
            if (
                args.output_dir is not None
                or args.challenge_file is not None
                or args.run_id is not None
                or args.run_nonce is not None
                or any(value is not None for value in case_arguments)
                or args.bag_empty_baseline_sha256 is not None
            ):
                raise LiveShadowConfigError("run-only arguments require --run-live")
            _emit(inspect_live_shadow_config(config_path))
            return 0

        config = load_live_shadow_config(config_path)
        if args.validate_config:
            if (
                args.output_dir is not None
                or args.challenge_file is not None
                or args.run_id is not None
                or args.run_nonce is not None
                or any(value is not None for value in case_arguments)
                or args.bag_empty_baseline_sha256 is not None
            ):
                raise LiveShadowConfigError("run-only arguments require --run-live")
            _emit(
                {
                    "schema_version": LIVE_SHADOW_PLAN_SCHEMA,
                    "mode": "VALIDATE_CONFIG",
                    "valid": True,
                    "config_sha256": config.config_sha256,
                    "remote_contacted": False,
                }
            )
            return 0
        if args.output_dir is None:
            raise LiveShadowConfigError("--run-live requires an explicit --output-dir")
        if args.challenge_file is None:
            raise LiveShadowConfigError("--run-live requires the challenge artifact issued before A0 capture")
        if args.run_id is not None or args.run_nonce is not None:
            raise LiveShadowConfigError("hand-entered run identifiers are forbidden; use --challenge-file")
        if any(value is not None for value in case_arguments) or (args.bag_empty_baseline_sha256 is not None):
            raise LiveShadowConfigError("ExperimentCase fields are accepted only by --issue-challenge")
        result = run_live_shadow(
            config,
            output_dir=args.output_dir,
            challenge_file=args.challenge_file.resolve(strict=True),
        )
        _emit(
            {
                "schema_version": LIVE_SHADOW_PLAN_SCHEMA,
                "mode": "RUN_LIVE",
                "run_id": result.binding.run_id,
                "status": result.report.status.value,
                "report_sha256": result.report.content_sha256,
                "evidence_root_sha256": result.evidence_root_sha256,
                "output_dir": str(result.output_dir),
                "remote_contacted": result.transport_commands_issued > 0,
                "transport_commands_issued": result.transport_commands_issued,
                "read_only_device_observations": result.read_only_device_observations,
                "actuator_commands_issued": 0,
                "mutating_commands_issued": 0,
                "execution_authority": result.report.execution_authority,
                "physical_closure_proven": result.report.physical_closure_proven,
            }
        )
        return 0 if result.report.status.value != "QUARANTINE" else 3
    except (LiveShadowConfigError, OSError, RuntimeError, ValueError) as exc:
        _emit(
            {"ok": False, "reason_code": type(exc).__name__, "detail": str(exc)},
            stream=sys.stderr,
        )
        return 2


__all__ = [
    "FROZEN_KNOWN_HOSTS_SHA256",
    "LIVE_CHALLENGE_TTL_MS",
    "LIVE_SHADOW_CHALLENGE_SCHEMA",
    "LIVE_SHADOW_CONFIG_SCHEMA",
    "LIVE_SHADOW_EVIDENCE_SCHEMA",
    "LIVE_SHADOW_PLAN_SCHEMA",
    "LIVE_SNAPSHOT_MAX_AGE_MS",
    "OVERHEAD_RESULT_SCHEMA",
    "OVERHEAD_SUCCESS_STATE",
    "OVERHEAD_TASK_KIND",
    "LiveShadowConfig",
    "LiveShadowConfigError",
    "LiveShadowRunResult",
    "compute_config_sha256",
    "inspect_live_shadow_config",
    "issue_live_shadow_challenge",
    "load_live_shadow_config",
    "main",
    "plan_live_shadow",
    "run_live_shadow",
]


if __name__ == "__main__":
    raise SystemExit(main())
