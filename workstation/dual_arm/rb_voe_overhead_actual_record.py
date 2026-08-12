#!/usr/bin/env python3
"""Validate and seal producer-attested A0 overhead-vision evidence.

The sealer never opens a camera, runs inference, contacts a device, or infers
that acquisition events happened from loose files. A live record is accepted
only when an A0 producer manifest binds the run challenge, both hosts, the
camera, event times, source artifacts, input frames, and authoritative result.

The runner attestation in this module is a fail-closed cooperation boundary,
not a privilege or cryptographic boundary. It prevents generic file-only APIs
from sealing production evidence, but it cannot defend against arbitrary code
execution by the same operating-system account as the trusted runner.
"""

from __future__ import annotations

import argparse
import ast
import base64
import binascii
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "xrd-rb-voe-overhead-a0-actual-v5"
ACQUISITION_SCHEMA_VERSION = "xrd-rb-voe-overhead-a0-acquisition-v4"
REPLAY_RECEIPT_SCHEMA_VERSION = "xrd-rb-voe-overhead-a0-consumption-v4"
CHALLENGE_SCHEMA_VERSION = "xrd-rb-voe-overhead-a0-challenge-v2"
REPLAY_IDENTITY_SCHEMA_VERSION = "xrd-rb-voe-overhead-a0-replay-identity-v3"
INPUT_FRAME_BUNDLE_SCHEMA_VERSION = "xrd-rb-voe-overhead-a0-input-frame-bundle-v1"
CAMERA_SERVICE_IDENTITY_SCHEMA_VERSION = "xrd-rb-voe-camera-service-identity-v1"
RUNNER_PRODUCTION_ATTESTATION_SCHEMA_VERSION = "xrd-rb-voe-a0-runner-attestation-v1"
BAG_EMPTY_BASELINE_SCHEMA_VERSION = "xrd-overhead-bag-empty-baseline-v1"
BAG_GATE_DERIVATION_SCHEMA_VERSION = "xrd-overhead-bag-gate-derivation-v1"
STATION_GATE_DERIVATION_SCHEMA_VERSION = "xrd-overhead-station-gate-derivation-v1"

DUAL_ARM_SEMANTIC_PROFILE_SHA256 = "18ec8e10b9cf13bc4075f6873061d338020f39bfbb9ad0b509e6b444b657d538"
FROZEN_BACKEND = "arm02.v4l2.capture_to_ai_x5.cpu_vision.run_bound_actual"
FROZEN_CAPTURE_HOSTNAME = "er"
FROZEN_INFERENCE_HOSTNAME = "xrd-ai"
FROZEN_CAMERA_OWNER = "arm02"
FROZEN_CAMERA_SOURCE = "/dev/video0"
FROZEN_CAMERA_USB_ID = "1bcf:0d1a"
FROZEN_CAMERA_UNIT_NAME = "xrd-overhead-camera.service"
FROZEN_CAMERA_UNIT_PATH = "/etc/systemd/system/xrd-overhead-camera.service"
FROZEN_CAMERA_UNIT_SHA256 = "99669577154286055ea449227b86bf8221efeed2df438f5e5f9dc96fadf388e2"
FROZEN_CAMERA_UNIT_SIZE_BYTES = 516
FROZEN_CAMERA_SCRIPT_PATH = "/home/rdk/dual_arm/overhead_camera_service.py"
FROZEN_CAMERA_CMDLINE = ("/usr/bin/python3", FROZEN_CAMERA_SCRIPT_PATH)
FROZEN_CAMERA_PROCESS_USER = "er"
FROZEN_CAMERA_LISTENER_IP = "127.0.0.1"
FROZEN_CAMERA_LISTENER_PORT = 8892
ACQUISITION_AUTHORITY_DOMAIN = "A0"
CONSUMER_AUTHORITY_DOMAIN = "R2_READONLY_SHADOW"
NO_EXTERNAL_MODEL_CONTRACT = "NO_EXTERNAL_MODEL_CPU_OPENCV"
FROZEN_STATION_MIN_EDGE_PX = 24.0
FROZEN_BAG_COLOR_GATE_FLOOR = 0.012
FROZEN_BAG_COLOR_GATE_SCALE = 5.0
FROZEN_BAG_COLOR_GATE_OFFSET = 0.004
FROZEN_BAG_COMPONENT_GATE_FLOOR = 0.008
FROZEN_BAG_COMPONENT_GATE_SCALE = 5.0
FROZEN_BAG_COMPONENT_GATE_OFFSET = 0.003
FROZEN_BAG_GATE_ROUND_DIGITS = 6
FROZEN_BAG_GATE_LOGIC = "both color-area gates; final decision by frame majority"

MAX_FRAME_BYTES = 64 * 1024 * 1024
MAX_INPUT_FRAME_COUNT = 32
MAX_INPUT_FRAME_TOTAL_BYTES = 128 * 1024 * 1024
MAX_INPUT_FRAME_BUNDLE_BYTES = 192 * 1024 * 1024
MAX_RESULT_JSON_BYTES = 4 * 1024 * 1024
MAX_PIPELINE_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_CAMERA_SERVICE_IDENTITY_BYTES = 256 * 1024
MAX_MANIFEST_BYTES = 256 * 1024
MAX_RECORD_BYTES = 256 * 1024
DEFAULT_MAX_AGE_MS = 5 * 60 * 1000
DEFAULT_MAX_FUTURE_SKEW_MS = 5 * 1000
DEFAULT_MAX_ACQUISITION_SPAN_MS = 10 * 60 * 1000
DEFAULT_MAX_INFERENCE_DURATION_MS = 2 * 60 * 1000
MIN_TIMESTAMP_MS = 1_700_000_000_000
BAG_EMPTY_FRAME_COUNT = 5
BAG_OCCUPIED_FRAME_COUNT = 5
BAG_INPUT_FRAME_COUNT = BAG_EMPTY_FRAME_COUNT + BAG_OCCUPIED_FRAME_COUNT

SUPPORTED_RESULT_SCHEMAS = frozenset(
    {
        "xrd-grinding-overhead-gate-v1",
        "xrd-overhead-bag-presence-v2",
    }
)

TASK_RESULT_CONTRACTS = {
    "BAG_DROP_IN_GRINDING_DISH": (
        "xrd-overhead-bag-presence-v2",
        "BAG_PRESENT",
    ),
    "GRINDING_STATION_PRESENCE_GATE": (
        "xrd-grinding-overhead-gate-v1",
        "STATION_OK",
    ),
}
INPUT_FRAME_ROLES = frozenset(
    {
        "EMPTY_BASELINE",
        "OCCUPIED_CANDIDATE",
        "STATION_OBSERVATION",
    }
)


@dataclass(frozen=True)
class _ArtifactContract:
    name: str
    sha256: str
    size_bytes: int
    role: str


CAPTURE_PIPELINE_CONTRACT = _ArtifactContract(
    name="overhead_camera_service.py",
    sha256="7a117d355c7e92013be1cfec472259655a00b8bbb716033281a547a1a43bd4d5",
    size_bytes=6083,
    role="capture_pipeline",
)

INFERENCE_PIPELINE_CONTRACTS = {
    "xrd-overhead-bag-presence-v2": _ArtifactContract(
        name="overhead_bag_presence_x5.py",
        sha256="00a29c24b306e4a093f83f55e568057bdffdc7d9de9581fb7ae41bd12cf02027",
        size_bytes=16842,
        role="authoritative_inference_pipeline",
    ),
    "xrd-grinding-overhead-gate-v1": _ArtifactContract(
        name="overhead_station_gate_x5.py",
        sha256="02ec923d1d357eb1ef5815c1db5fbbd4d02fd121702c953a2deb9b1e4ceec97d",
        size_bytes=4892,
        role="authoritative_inference_pipeline",
    ),
}

HOST_KEYS = frozenset({"hostname", "device_id", "boot_id", "session_id"})
CAMERA_KEYS = frozenset({"owner", "source", "usb_id", "backend"})
CAMERA_SERVICE_IDENTITY_KEYS = frozenset(
    {
        "schema_version",
        "hostname",
        "device_id",
        "boot_id",
        "session_id",
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
        "observed_at_ms",
        "service_identity_sha256",
        "artifact_sha256",
    }
)
CAMERA_SERVICE_RUNTIME_KEYS = frozenset(
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
    }
)
CAMERA_SERVICE_REFERENCE_KEYS = frozenset(
    {
        "file_name",
        "file_sha256",
        "size_bytes",
        "schema",
        "service_identity_sha256",
        "artifact_sha256",
        "observed_at_ms",
        "main_pid",
    }
)
AUTHORITY_KEYS = frozenset(
    {
        "domain",
        "hardware_touched",
        "camera_opened",
        "inference_triggered",
        "motion_authority",
        "robot_sdk_opened",
        "serial_opened",
        "gpio_opened",
        "actuator_commands_issued",
    }
)
EVENT_KEYS = frozenset(
    {
        "producer_started_at_ms",
        "camera_service_identity_observed_at_ms",
        "camera_opened_at_ms",
        "frame_captured_at_ms",
        "input_frame_bundle_bound_at_ms",
        "inference_started_at_ms",
        "inference_completed_at_ms",
        "manifest_emitted_at_ms",
    }
)
FRAME_KEYS = frozenset({"file_name", "sha256", "size_bytes", "media_type", "width", "height"})
FRAME_BUNDLE_KEYS = frozenset(
    {
        "file_name",
        "file_sha256",
        "size_bytes",
        "schema",
        "bundle_sha256",
        "entry_count",
        "total_bytes",
    }
)
INPUT_FRAME_BUNDLE_KEYS = frozenset(
    {"schema_version", "entries", "entry_count", "total_bytes", "bundle_sha256"}
)
INPUT_FRAME_BUNDLE_ENTRY_KEYS = frozenset(
    {"role", "file_name", "sha256", "size_bytes", "width", "height", "jpeg_base64"}
)
RESULT_KEYS = frozenset(
    {
        "file_name",
        "sha256",
        "size_bytes",
        "schema",
        "state",
        "success",
        "input_frame_sha256",
        "baseline_sha256",
        "derivation_sha256",
    }
)
ARTIFACT_KEYS = frozenset({"capture_pipeline", "inference_pipeline", "model_contract", "models"})
PIPELINE_KEYS = frozenset({"role", "name", "sha256", "size_bytes"})
ACQUISITION_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "acquisition_id",
        "a0_run_id",
        "r2_run_id",
        "r2_run_nonce_sha256",
        "challenge_sha256",
        "challenge_issued_at_ms",
        "challenge_expires_at_ms",
        "replay_identity_sha256",
        "release_id",
        "config_sha256",
        "case_id",
        "sample_id",
        "sample_lineage_sha256",
        "parent_evidence_root_sha256",
        "bag_empty_baseline_sha256",
        "task_kind",
        "result_schema",
        "success_state",
        "dual_arm_semantic_profile_sha256",
        "capture_host",
        "inference_host",
        "camera",
        "camera_service_identity",
        "authority",
        "events",
        "frame",
        "frame_bundle",
        "result",
        "artifacts",
        "manifest_sha256",
    }
)

TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "acquisition_manifest_schema",
        "acquisition_manifest_sha256",
        "acquisition_manifest_size_bytes",
        "acquisition_id",
        "a0_run_id",
        "r2_run_id",
        "r2_run_nonce_sha256",
        "challenge_sha256",
        "challenge_issued_at_ms",
        "challenge_expires_at_ms",
        "replay_identity_sha256",
        "config_sha256",
        "case_id",
        "sample_id",
        "sample_lineage_sha256",
        "parent_evidence_root_sha256",
        "bag_empty_baseline_sha256",
        "task_kind",
        "success_state",
        "capture_hostname",
        "capture_device_id",
        "capture_boot_id",
        "capture_session_id",
        "inference_hostname",
        "ai_x5_device_id",
        "ai_x5_boot_id",
        "ai_x5_session_id",
        "release_id",
        "dual_arm_semantic_profile_sha256",
        "camera_owner",
        "camera_source",
        "camera_usb_id",
        "backend",
        "camera_service_identity_schema",
        "camera_service_identity_file_name",
        "camera_service_identity_file_sha256",
        "camera_service_identity_size_bytes",
        "camera_service_identity_sha256",
        "camera_service_runtime_sha256",
        "camera_service_observed_at_ms",
        "camera_service_main_pid",
        "camera_service_unit_name",
        "camera_service_unit_path",
        "camera_service_unit_sha256",
        "camera_service_script_path",
        "camera_service_script_sha256",
        "camera_service_cmdline_sha256",
        "camera_service_video_owner_pid",
        "camera_service_video_owner_user",
        "camera_service_usb_id",
        "camera_service_listener_ip",
        "camera_service_listener_port",
        "camera_service_listener_inode",
        "camera_service_listener_owner_pid",
        "captured_at_ms",
        "observed_at_ms",
        "manifest_emitted_at_ms",
        "raw_frame_sha256",
        "raw_frame_size_bytes",
        "raw_frame_media_type",
        "raw_frame_width",
        "raw_frame_height",
        "frame_bundle_file_name",
        "frame_bundle_file_sha256",
        "frame_bundle_size_bytes",
        "frame_bundle_schema",
        "frame_bundle_sha256",
        "frame_bundle_entry_count",
        "frame_bundle_total_bytes",
        "result_json_sha256",
        "result_json_size_bytes",
        "result_schema",
        "result_state",
        "result_success",
        "result_input_frame_sha256",
        "result_baseline_sha256",
        "result_derivation_sha256",
        "capture_pipeline_name",
        "capture_pipeline_sha256",
        "capture_pipeline_size_bytes",
        "inference_pipeline_name",
        "inference_pipeline_sha256",
        "inference_pipeline_size_bytes",
        "model_contract",
        "model_artifacts",
        "acquisition_authority_domain",
        "acquisition_hardware_touched",
        "acquisition_camera_opened",
        "acquisition_inference_triggered",
        "acquisition_motion_authority",
        "acquisition_robot_sdk_opened",
        "acquisition_serial_opened",
        "acquisition_gpio_opened",
        "acquisition_actuator_commands_issued",
        "consumer_authority_domain",
        "sealer_hardware_touched",
        "sealer_execution_authority",
        "sealer_actuator_commands_issued",
        "sealer_camera_opened",
        "sealer_inference_triggered",
        "record_sha256",
    }
)

REPLAY_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "replay_key",
        "replay_identity_sha256",
        "challenge_sha256",
        "challenge_issued_at_ms",
        "challenge_expires_at_ms",
        "acquisition_id",
        "a0_run_id",
        "r2_run_id",
        "r2_run_nonce_sha256",
        "release_id",
        "config_sha256",
        "case_id",
        "sample_id",
        "sample_lineage_sha256",
        "parent_evidence_root_sha256",
        "bag_empty_baseline_sha256",
        "task_kind",
        "result_schema",
        "success_state",
        "camera_service_identity_file_sha256",
        "camera_service_identity_sha256",
        "frame_bundle_file_sha256",
        "frame_bundle_sha256",
        "acquisition_manifest_sha256",
        "record_sha256",
        "receipt_sha256",
    }
)

_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@+-]{0,159}\Z")
_SAFE_BASENAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_USB_ID_RE = re.compile(r"[0-9a-f]{4}:[0-9a-f]{4}\Z")


class ActualRecordError(ValueError):
    """Raised when A0 evidence cannot be validated fail-closed."""


@dataclass(frozen=True)
class ExpectedAcquisition:
    """Run challenge and live identities supplied by the R2 coordinator."""

    acquisition_id: str
    a0_run_id: str
    r2_run_id: str
    r2_run_nonce: str
    challenge_sha256: str
    challenge_issued_at_ms: int
    challenge_expires_at_ms: int
    release_id: str
    config_sha256: str
    case_id: str
    sample_id: str
    sample_lineage_sha256: str
    parent_evidence_root_sha256: str
    bag_empty_baseline_sha256: str | None
    task_kind: str
    result_schema: str
    success_state: str
    capture_device_id: str
    capture_boot_id: str
    capture_session_id: str
    inference_device_id: str
    inference_boot_id: str
    inference_session_id: str

    def validate(self) -> None:
        for field in (
            "acquisition_id",
            "a0_run_id",
            "r2_run_id",
            "r2_run_nonce",
            "release_id",
            "case_id",
            "sample_id",
            "task_kind",
            "result_schema",
            "success_state",
            "capture_device_id",
            "capture_boot_id",
            "capture_session_id",
            "inference_device_id",
            "inference_boot_id",
            "inference_session_id",
        ):
            _require_text(getattr(self, field), label=f"expected.{field}")
        for field in (
            "challenge_sha256",
            "config_sha256",
            "sample_lineage_sha256",
            "parent_evidence_root_sha256",
        ):
            _require_sha256(getattr(self, field), label=f"expected.{field}")
        _require_optional_sha256(
            self.bag_empty_baseline_sha256,
            label="expected.bag_empty_baseline_sha256",
        )
        expected_result_contract = TASK_RESULT_CONTRACTS.get(self.task_kind)
        if expected_result_contract != (self.result_schema, self.success_state):
            raise ActualRecordError(
                "expected task_kind/result_schema/success_state is not a fixed valid tuple"
            )
        if self.task_kind == "BAG_DROP_IN_GRINDING_DISH":
            if self.bag_empty_baseline_sha256 is None:
                raise ActualRecordError("bag task requires expected.bag_empty_baseline_sha256")
        elif self.bag_empty_baseline_sha256 is not None:
            raise ActualRecordError("station task forbids expected.bag_empty_baseline_sha256")
        issued_at_ms = _require_int(
            self.challenge_issued_at_ms,
            label="expected.challenge_issued_at_ms",
            minimum=MIN_TIMESTAMP_MS,
        )
        expires_at_ms = _require_int(
            self.challenge_expires_at_ms,
            label="expected.challenge_expires_at_ms",
            minimum=MIN_TIMESTAMP_MS,
        )
        if expires_at_ms <= issued_at_ms:
            raise ActualRecordError("expected challenge expiry must follow issuance")
        if not self.acquisition_id.startswith("A0-ACQ-"):
            raise ActualRecordError("expected.acquisition_id must begin with A0-ACQ-")
        if not self.a0_run_id.startswith("A0-RUN-"):
            raise ActualRecordError("expected.a0_run_id must begin with A0-RUN-")
        if not self.r2_run_id.startswith("R2-RUN-"):
            raise ActualRecordError("expected.r2_run_id must begin with R2-RUN-")
        if len(self.r2_run_nonce) < 16:
            raise ActualRecordError("expected.r2_run_nonce is too short")
        if len({self.acquisition_id, self.a0_run_id, self.r2_run_id}) != 3:
            raise ActualRecordError("expected run identifiers must be distinct")
        if self.challenge_sha256 != challenge_sha256_for_expected(self):
            raise ActualRecordError("expected.challenge_sha256 does not bind the supplied challenge")


@dataclass(frozen=True)
class _FileEvidence:
    path: Path
    size_bytes: int
    sha256: str
    data: bytes
    device: int
    inode: int


@dataclass(frozen=True)
class _ResultSemantics:
    schema: str
    state: str
    success: bool
    event_at_ms: int
    width: int
    height: int
    baseline_sha256: str | None
    derivation_sha256: str
    input_frame_identities: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class _InputFrameBundle:
    value: Mapping[str, Any]
    identities: tuple[tuple[str, str, str], ...]
    bundle_sha256: str
    entry_count: int
    total_bytes: int


@dataclass(frozen=True)
class _CameraServiceIdentity:
    value: Mapping[str, Any]
    service_identity_sha256: str
    artifact_sha256: str
    observed_at_ms: int
    main_pid: int


@dataclass(frozen=True)
class _ValidatedAcquisition:
    manifest: Mapping[str, Any]
    manifest_file: _FileEvidence | None
    frame: _FileEvidence
    frame_bundle: _FileEvidence
    parsed_frame_bundle: _InputFrameBundle
    result: _FileEvidence
    capture_pipeline: _FileEvidence
    inference_pipeline: _FileEvidence
    camera_service_identity_file: _FileEvidence
    camera_service_identity: _CameraServiceIdentity
    semantics: _ResultSemantics


@dataclass(frozen=True)
class ValidatedSealedRecord:
    """Strictly reconstructed record plus its stable one-time receipt."""

    receipt_path: Path
    receipt_sha256: str
    replay_identity_sha256: str


_RUNNER_PRODUCTION_CAPABILITY = object()


@dataclass(frozen=True)
class RunnerProductionAttestation:
    """Opaque in-process proof that the recorder completed its production path.

    This is deliberately not a same-account security primitive. The private
    capability only prevents accidental or generic file-only sealing APIs from
    bypassing the recorder's ordered production checks.
    """

    schema_version: str
    acquisition_manifest_file_sha256: str
    manifest_sha256: str
    camera_service_identity_file_sha256: str
    camera_service_identity_sha256: str
    frame_bundle_file_sha256: str
    frame_bundle_sha256: str
    replay_identity_sha256: str
    _capability: object


def canonical_json_bytes(value: object) -> bytes:
    """Return the sole canonical encoding used by manifest and record hashes."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ActualRecordError(f"value is not canonical JSON: {exc}") from exc


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ActualRecordError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_json_constant(value: str) -> None:
    raise ActualRecordError(f"non-finite JSON number is forbidden: {value}")


def _load_json_bytes(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ActualRecordError(f"{label} is not UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except ActualRecordError:
        raise
    except json.JSONDecodeError as exc:
        raise ActualRecordError(f"{label} is not valid JSON: {exc.msg}") from exc
    if type(value) is not dict:
        raise ActualRecordError(f"{label} must contain one JSON object")
    return value


def _absolute_lexical_path(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    if ".." in candidate.parts:
        raise ActualRecordError(f"parent traversal is forbidden: {candidate}")
    return Path(os.path.abspath(os.fspath(candidate)))


def _is_reparse_point(metadata: os.stat_result) -> bool:
    attributes = getattr(metadata, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _reject_link_components(path: Path) -> None:
    parts = path.parts
    current = Path(parts[0])
    for part in parts[1:]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise ActualRecordError(f"path component does not exist: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode) or _is_reparse_point(metadata):
            raise ActualRecordError(f"link or reparse path component is forbidden: {current}")


def _read_regular_file(
    path: Path | str,
    *,
    label: str,
    max_bytes: int,
) -> _FileEvidence:
    lexical = _absolute_lexical_path(path)
    _reject_link_components(lexical)
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise ActualRecordError(f"{label} does not exist: {lexical}") from exc
    if resolved != lexical:
        raise ActualRecordError(f"{label} path does not resolve literally: {lexical}")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise ActualRecordError(f"cannot open {label}: {resolved}") from exc
    try:
        before = os.fstat(descriptor)
        if _is_reparse_point(before):
            raise ActualRecordError(f"{label} is a reparse point: {resolved}")
        if not stat.S_ISREG(before.st_mode):
            raise ActualRecordError(f"{label} is not a regular file: {resolved}")
        if before.st_size <= 0:
            raise ActualRecordError(f"{label} is empty: {resolved}")
        if before.st_size > max_bytes:
            raise ActualRecordError(f"{label} exceeds {max_bytes} bytes: {before.st_size}")

        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ActualRecordError(f"{label} changed while being read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ActualRecordError(f"{label} grew while being read")
        after = os.fstat(descriptor)
        stable = (
            before.st_dev == after.st_dev
            and before.st_ino == after.st_ino
            and before.st_size == after.st_size
            and before.st_mtime_ns == after.st_mtime_ns
        )
        if not stable:
            raise ActualRecordError(f"{label} changed while being read")
    finally:
        os.close(descriptor)

    data = b"".join(chunks)
    return _FileEvidence(
        path=resolved,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        data=data,
        device=before.st_dev,
        inode=before.st_ino,
    )


def _exact_keys(value: object, expected: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise ActualRecordError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ActualRecordError(f"{label} keys differ; missing={missing}, extra={extra}")
    return value


def _require_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise ActualRecordError(f"{label} must be a boolean")
    return value


def _require_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ActualRecordError(f"{label} must be an integer >= {minimum}")
    return value


def _require_number(value: object, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActualRecordError(f"{label} must be a finite number")
    output = float(value)
    if not math.isfinite(output) or (positive and output <= 0):
        raise ActualRecordError(f"{label} must be a finite positive number")
    return output


def _require_text(value: object, *, label: str) -> str:
    if type(value) is not str or not _IDENTIFIER_RE.fullmatch(value):
        raise ActualRecordError(f"{label} is not a valid nonempty identifier")
    return value


def _require_safe_basename(value: object, *, label: str) -> str:
    if type(value) is not str or not _SAFE_BASENAME_RE.fullmatch(value):
        raise ActualRecordError(f"{label} is not a safe basename")
    if Path(value).name != value or "/" in value or "\\" in value or value in {".", ".."}:
        raise ActualRecordError(f"{label} is not a safe basename")
    if Path(value).suffix.lower() not in {".jpg", ".jpeg"}:
        raise ActualRecordError(f"{label} must have a JPEG basename")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or not _SHA256_RE.fullmatch(value):
        raise ActualRecordError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_optional_sha256(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    return _require_sha256(value, label=label)


def a0_challenge_sha256(
    *,
    acquisition_id: str,
    a0_run_id: str,
    r2_run_id: str,
    r2_run_nonce: str,
    challenge_issued_at_ms: int,
    challenge_expires_at_ms: int,
    release_id: str,
    config_sha256: str,
    case_id: str,
    sample_id: str,
    sample_lineage_sha256: str,
    parent_evidence_root_sha256: str,
    bag_empty_baseline_sha256: str | None,
    task_kind: str,
    result_schema: str,
    success_state: str,
) -> str:
    """Hash the coordinator challenge without manufacturing any field."""

    for name, value in (
        ("acquisition_id", acquisition_id),
        ("a0_run_id", a0_run_id),
        ("r2_run_id", r2_run_id),
        ("r2_run_nonce", r2_run_nonce),
        ("release_id", release_id),
        ("case_id", case_id),
        ("sample_id", sample_id),
        ("task_kind", task_kind),
        ("result_schema", result_schema),
        ("success_state", success_state),
    ):
        _require_text(value, label=f"challenge.{name}")
    for name, value in (
        ("config_sha256", config_sha256),
        ("sample_lineage_sha256", sample_lineage_sha256),
        ("parent_evidence_root_sha256", parent_evidence_root_sha256),
    ):
        _require_sha256(value, label=f"challenge.{name}")
    _require_optional_sha256(
        bag_empty_baseline_sha256,
        label="challenge.bag_empty_baseline_sha256",
    )
    if TASK_RESULT_CONTRACTS.get(task_kind) != (result_schema, success_state):
        raise ActualRecordError("challenge task/result tuple is unsupported")
    if (task_kind == "BAG_DROP_IN_GRINDING_DISH") != (bag_empty_baseline_sha256 is not None):
        raise ActualRecordError("challenge baseline does not match task kind")
    issued = _require_int(
        challenge_issued_at_ms,
        label="challenge.challenge_issued_at_ms",
        minimum=MIN_TIMESTAMP_MS,
    )
    expires = _require_int(
        challenge_expires_at_ms,
        label="challenge.challenge_expires_at_ms",
        minimum=MIN_TIMESTAMP_MS,
    )
    if expires <= issued:
        raise ActualRecordError("challenge expiry must follow issuance")
    return canonical_sha256(
        {
            "schema_version": CHALLENGE_SCHEMA_VERSION,
            "acquisition_id": acquisition_id,
            "a0_run_id": a0_run_id,
            "r2_run_id": r2_run_id,
            "r2_run_nonce_sha256": hashlib.sha256(r2_run_nonce.encode("utf-8")).hexdigest(),
            "challenge_issued_at_ms": issued,
            "challenge_expires_at_ms": expires,
            "release_id": release_id,
            "config_sha256": config_sha256,
            "case_id": case_id,
            "sample_id": sample_id,
            "sample_lineage_sha256": sample_lineage_sha256,
            "parent_evidence_root_sha256": parent_evidence_root_sha256,
            "bag_empty_baseline_sha256": bag_empty_baseline_sha256,
            "task_kind": task_kind,
            "result_schema": result_schema,
            "success_state": success_state,
        }
    )


def challenge_sha256_for_expected(expected: ExpectedAcquisition) -> str:
    return a0_challenge_sha256(
        acquisition_id=expected.acquisition_id,
        a0_run_id=expected.a0_run_id,
        r2_run_id=expected.r2_run_id,
        r2_run_nonce=expected.r2_run_nonce,
        challenge_issued_at_ms=expected.challenge_issued_at_ms,
        challenge_expires_at_ms=expected.challenge_expires_at_ms,
        release_id=expected.release_id,
        config_sha256=expected.config_sha256,
        case_id=expected.case_id,
        sample_id=expected.sample_id,
        sample_lineage_sha256=expected.sample_lineage_sha256,
        parent_evidence_root_sha256=expected.parent_evidence_root_sha256,
        bag_empty_baseline_sha256=expected.bag_empty_baseline_sha256,
        task_kind=expected.task_kind,
        result_schema=expected.result_schema,
        success_state=expected.success_state,
    )


def replay_identity_sha256(
    *,
    challenge_sha256: str,
    acquisition_id: str,
    a0_run_id: str,
    r2_run_id: str,
    r2_run_nonce_sha256: str,
    task_kind: str,
    result_schema: str,
    success_state: str,
    camera_service_identity_sha256: str,
) -> str:
    """Return the immutable challenge and camera-runtime consumption identity."""

    _require_sha256(challenge_sha256, label="replay.challenge_sha256")
    _require_sha256(r2_run_nonce_sha256, label="replay.r2_run_nonce_sha256")
    _require_sha256(
        camera_service_identity_sha256,
        label="replay.camera_service_identity_sha256",
    )
    for name, value in (
        ("acquisition_id", acquisition_id),
        ("a0_run_id", a0_run_id),
        ("r2_run_id", r2_run_id),
        ("task_kind", task_kind),
        ("result_schema", result_schema),
        ("success_state", success_state),
    ):
        _require_text(value, label=f"replay.{name}")
    if TASK_RESULT_CONTRACTS.get(task_kind) != (result_schema, success_state):
        raise ActualRecordError("replay task/result tuple is unsupported")
    return canonical_sha256(
        {
            "schema_version": REPLAY_IDENTITY_SCHEMA_VERSION,
            "challenge_sha256": challenge_sha256,
            "acquisition_id": acquisition_id,
            "a0_run_id": a0_run_id,
            "r2_run_id": r2_run_id,
            "r2_run_nonce_sha256": r2_run_nonce_sha256,
            "task_kind": task_kind,
            "result_schema": result_schema,
            "success_state": success_state,
            "camera_service_identity_sha256": camera_service_identity_sha256,
        }
    )


def replay_identity_sha256_for_expected(
    expected: ExpectedAcquisition,
    *,
    camera_service_identity_sha256: str,
) -> str:
    expected.validate()
    return replay_identity_sha256(
        challenge_sha256=expected.challenge_sha256,
        acquisition_id=expected.acquisition_id,
        a0_run_id=expected.a0_run_id,
        r2_run_id=expected.r2_run_id,
        r2_run_nonce_sha256=hashlib.sha256(expected.r2_run_nonce.encode("utf-8")).hexdigest(),
        task_kind=expected.task_kind,
        result_schema=expected.result_schema,
        success_state=expected.success_state,
        camera_service_identity_sha256=camera_service_identity_sha256,
    )


def replay_identity_sha256_from_record(record: Mapping[str, Any]) -> str:
    return replay_identity_sha256(
        challenge_sha256=record["challenge_sha256"],
        acquisition_id=record["acquisition_id"],
        a0_run_id=record["a0_run_id"],
        r2_run_id=record["r2_run_id"],
        r2_run_nonce_sha256=record["r2_run_nonce_sha256"],
        task_kind=record["task_kind"],
        result_schema=record["result_schema"],
        success_state=record["success_state"],
        camera_service_identity_sha256=record["camera_service_identity_sha256"],
    )


def _require_sequence(value: object, *, label: str) -> list[Any]:
    if type(value) is not list:
        raise ActualRecordError(f"{label} must be an array")
    return value


def _validate_camera_service_identity_value(
    value: object,
    *,
    expected: ExpectedAcquisition,
) -> _CameraServiceIdentity:
    identity = _exact_keys(
        value,
        CAMERA_SERVICE_IDENTITY_KEYS,
        label="camera service identity artifact",
    )
    if identity["schema_version"] != CAMERA_SERVICE_IDENTITY_SCHEMA_VERSION:
        raise ActualRecordError("camera service identity schema is unsupported")
    host_expected = {
        "hostname": FROZEN_CAPTURE_HOSTNAME,
        "device_id": expected.capture_device_id,
        "boot_id": expected.capture_boot_id,
        "session_id": expected.capture_session_id,
    }
    for field, expected_value in host_expected.items():
        actual_value = _require_text(identity[field], label=f"camera service identity.{field}")
        if actual_value != expected_value:
            raise ActualRecordError(f"camera service identity {field} mismatch")

    pid = _require_int(identity["pid"], label="camera service identity.pid", minimum=2)
    cmdline = _require_sequence(identity["cmdline"], label="camera service identity.cmdline")
    if any(type(item) is not str or not item for item in cmdline):
        raise ActualRecordError("camera service identity cmdline is malformed")
    if tuple(cmdline) != FROZEN_CAMERA_CMDLINE:
        raise ActualRecordError("camera service identity cmdline drifted")

    fixed_text = {
        "script_path": FROZEN_CAMERA_SCRIPT_PATH,
        "unit_name": FROZEN_CAMERA_UNIT_NAME,
        "unit_active_state": "active",
        "unit_sub_state": "running",
        "unit_fragment_path": FROZEN_CAMERA_UNIT_PATH,
        "service_user": FROZEN_CAMERA_PROCESS_USER,
        "video_device": FROZEN_CAMERA_SOURCE,
        "video_owner_user": FROZEN_CAMERA_PROCESS_USER,
        "usb_id": FROZEN_CAMERA_USB_ID,
        "listener_ip": FROZEN_CAMERA_LISTENER_IP,
    }
    for field, expected_value in fixed_text.items():
        actual_value = identity[field]
        if type(actual_value) is not str or not actual_value:
            raise ActualRecordError(f"camera service identity.{field} must be a nonempty string")
        if actual_value != expected_value:
            raise ActualRecordError(f"camera service identity {field} drifted")
    if _USB_ID_RE.fullmatch(str(identity["usb_id"])) is None:
        raise ActualRecordError("camera service identity usb_id is invalid")

    digest_checks = {
        "script_sha256": CAPTURE_PIPELINE_CONTRACT.sha256,
        "unit_sha256": FROZEN_CAMERA_UNIT_SHA256,
    }
    for field, expected_value in digest_checks.items():
        actual_value = _require_sha256(identity[field], label=f"camera service identity.{field}")
        if actual_value != expected_value:
            raise ActualRecordError(f"camera service identity {field} drifted")
    size_checks = {
        "script_size_bytes": CAPTURE_PIPELINE_CONTRACT.size_bytes,
        "unit_size_bytes": FROZEN_CAMERA_UNIT_SIZE_BYTES,
    }
    for field, expected_value in size_checks.items():
        actual_value = _require_int(
            identity[field],
            label=f"camera service identity.{field}",
            minimum=1,
        )
        if actual_value != expected_value:
            raise ActualRecordError(f"camera service identity {field} drifted")

    unit_main_pid = _require_int(
        identity["unit_main_pid"],
        label="camera service identity.unit_main_pid",
        minimum=2,
    )
    video_owner_pid = _require_int(
        identity["video_owner_pid"],
        label="camera service identity.video_owner_pid",
        minimum=2,
    )
    listener_port = _require_int(
        identity["listener_port"],
        label="camera service identity.listener_port",
        minimum=1,
    )
    if listener_port != FROZEN_CAMERA_LISTENER_PORT:
        raise ActualRecordError("camera service identity listener_port drifted")
    _require_int(
        identity["listener_inode"],
        label="camera service identity.listener_inode",
        minimum=1,
    )
    listener_owner_pid = _require_int(
        identity["listener_owner_pid"],
        label="camera service identity.listener_owner_pid",
        minimum=2,
    )
    _require_int(
        identity["video_owner_uid"],
        label="camera service identity.video_owner_uid",
        minimum=0,
    )
    if unit_main_pid != pid or video_owner_pid != pid or listener_owner_pid != pid:
        raise ActualRecordError(
            "camera service MainPID/video/listener owner does not match the service process"
        )

    observed_at_ms = _require_int(
        identity["observed_at_ms"],
        label="camera service identity.observed_at_ms",
        minimum=MIN_TIMESTAMP_MS,
    )
    runtime_unsigned = {key: identity[key] for key in CAMERA_SERVICE_RUNTIME_KEYS}
    service_identity_sha256 = _require_sha256(
        identity["service_identity_sha256"],
        label="camera service identity.service_identity_sha256",
    )
    if canonical_sha256(runtime_unsigned) != service_identity_sha256:
        raise ActualRecordError("camera service runtime identity digest mismatch")
    unsigned = dict(identity)
    artifact_sha256 = _require_sha256(
        unsigned.pop("artifact_sha256"),
        label="camera service identity.artifact_sha256",
    )
    if canonical_sha256(unsigned) != artifact_sha256:
        raise ActualRecordError("camera service identity artifact digest mismatch")
    return _CameraServiceIdentity(
        value=identity,
        service_identity_sha256=service_identity_sha256,
        artifact_sha256=artifact_sha256,
        observed_at_ms=observed_at_ms,
        main_pid=pid,
    )


def _camera_service_identity_reference(
    evidence: _FileEvidence,
    identity: _CameraServiceIdentity,
) -> dict[str, Any]:
    return {
        "file_name": evidence.path.name,
        "file_sha256": evidence.sha256,
        "size_bytes": evidence.size_bytes,
        "schema": CAMERA_SERVICE_IDENTITY_SCHEMA_VERSION,
        "service_identity_sha256": identity.service_identity_sha256,
        "artifact_sha256": identity.artifact_sha256,
        "observed_at_ms": identity.observed_at_ms,
        "main_pid": identity.main_pid,
    }


def _validate_pair(value: object, *, label: str, positive: bool = False) -> None:
    items = _require_sequence(value, label=label)
    if len(items) != 2:
        raise ActualRecordError(f"{label} must have exactly two values")
    for index, item in enumerate(items):
        _require_number(item, label=f"{label}[{index}]", positive=positive)


def _validate_quad(value: object, *, label: str) -> None:
    items = _require_sequence(value, label=label)
    if len(items) != 4:
        raise ActualRecordError(f"{label} must have exactly four values")
    for index, item in enumerate(items):
        _require_number(item, label=f"{label}[{index}]")


def _validate_roi(value: object, *, label: str) -> None:
    roi = _exact_keys(
        value,
        frozenset(
            {
                "kind",
                "center_px",
                "axes_px",
                "dish_component_bbox_px",
                "dish_component_area_px",
                "pixel_count",
            }
        ),
        label=label,
    )
    if roi["kind"] != "dynamic_pink_dish_ellipse":
        raise ActualRecordError(f"{label}.kind is not the frozen dish ROI")
    _validate_pair(roi["center_px"], label=f"{label}.center_px")
    _validate_pair(roi["axes_px"], label=f"{label}.axes_px", positive=True)
    _validate_quad(roi["dish_component_bbox_px"], label=f"{label}.bbox")
    _require_int(roi["dish_component_area_px"], label=f"{label}.area", minimum=1)
    _require_int(roi["pixel_count"], label=f"{label}.pixel_count", minimum=1)


def _validate_metrics(value: object, *, label: str) -> Mapping[str, Any]:
    metrics = _exact_keys(
        value,
        frozenset({"bag_color_ratio", "largest_bag_color_component_ratio"}),
        label=label,
    )
    for name in metrics:
        number = _require_number(metrics[name], label=f"{label}.{name}")
        if not 0.0 <= number <= 1.0:
            raise ActualRecordError(f"{label}.{name} is outside [0, 1]")
    return metrics


def _validated_bag_empty_baseline(
    value: object,
) -> tuple[Mapping[str, Any], list[Any], float, float, str]:
    empty = _exact_keys(
        value,
        frozenset(
            {
                "count",
                "files",
                "max_bag_color_ratio",
                "max_largest_bag_color_component_ratio",
            }
        ),
        label="bag result empty",
    )
    empty_files = _require_sequence(empty["files"], label="empty.files")
    empty_count = _require_int(empty["count"], label="empty.count", minimum=1)
    if empty_count != len(empty_files):
        raise ActualRecordError("empty.count does not match empty.files")
    if empty_count != BAG_EMPTY_FRAME_COUNT:
        raise ActualRecordError(
            f"production bag baseline requires exactly {BAG_EMPTY_FRAME_COUNT} EMPTY_BASELINE frames"
        )

    names: set[str] = set()
    digests: set[str] = set()
    color_values: list[float] = []
    component_values: list[float] = []
    for index, row_value in enumerate(empty_files):
        row = _exact_keys(
            row_value,
            frozenset({"name", "sha256", "dish_roi", "metrics"}),
            label=f"empty.files[{index}]",
        )
        name = _require_safe_basename(row["name"], label=f"empty.files[{index}].name")
        digest = _require_sha256(row["sha256"], label=f"empty.files[{index}].sha256")
        if name in names or digest in digests:
            raise ActualRecordError("empty baseline contains duplicate file identity")
        names.add(name)
        digests.add(digest)
        _validate_roi(row["dish_roi"], label=f"empty.files[{index}].dish_roi")
        metrics = _validate_metrics(row["metrics"], label=f"empty.files[{index}].metrics")
        color_values.append(
            _require_number(metrics["bag_color_ratio"], label=f"empty.files[{index}].bag_color_ratio")
        )
        component_values.append(
            _require_number(
                metrics["largest_bag_color_component_ratio"],
                label=f"empty.files[{index}].largest_bag_color_component_ratio",
            )
        )

    computed_color_max = max(color_values)
    computed_component_max = max(component_values)
    claimed_color_max = _require_number(
        empty["max_bag_color_ratio"],
        label="empty.max_bag_color_ratio",
    )
    claimed_component_max = _require_number(
        empty["max_largest_bag_color_component_ratio"],
        label="empty.max_largest_bag_color_component_ratio",
    )
    if claimed_color_max != computed_color_max:
        raise ActualRecordError("empty max_bag_color_ratio contradicts the exact baseline rows")
    if claimed_component_max != computed_component_max:
        raise ActualRecordError(
            "empty max_largest_bag_color_component_ratio contradicts the exact baseline rows"
        )
    baseline_sha256 = canonical_sha256(
        {
            "schema_version": BAG_EMPTY_BASELINE_SCHEMA_VERSION,
            "count": empty_count,
            "files": empty_files,
        }
    )
    return empty, empty_files, computed_color_max, computed_component_max, baseline_sha256


def bag_empty_baseline_sha256(value: object) -> str:
    """Commit the exact ordered empty-frame identities, ROIs, and metrics."""

    return _validated_bag_empty_baseline(value)[4]


def derive_bag_gate_contract(value: object) -> dict[str, Any]:
    """Reproduce the frozen production gate derivation from an exact baseline."""

    _, _, color_max, component_max, baseline_sha256 = _validated_bag_empty_baseline(value)
    decision_color_gate = max(
        FROZEN_BAG_COLOR_GATE_FLOOR,
        color_max * FROZEN_BAG_COLOR_GATE_SCALE + FROZEN_BAG_COLOR_GATE_OFFSET,
    )
    decision_component_gate = max(
        FROZEN_BAG_COMPONENT_GATE_FLOOR,
        component_max * FROZEN_BAG_COMPONENT_GATE_SCALE + FROZEN_BAG_COMPONENT_GATE_OFFSET,
    )
    color_gate = round(decision_color_gate, FROZEN_BAG_GATE_ROUND_DIGITS)
    component_gate = round(decision_component_gate, FROZEN_BAG_GATE_ROUND_DIGITS)
    derivation = {
        "schema_version": BAG_GATE_DERIVATION_SCHEMA_VERSION,
        "baseline_sha256": baseline_sha256,
        "empty_max_bag_color_ratio": color_max,
        "empty_max_largest_bag_color_component_ratio": component_max,
        "bag_color_ratio_formula": {
            "floor": FROZEN_BAG_COLOR_GATE_FLOOR,
            "scale": FROZEN_BAG_COLOR_GATE_SCALE,
            "offset": FROZEN_BAG_COLOR_GATE_OFFSET,
            "round_digits": FROZEN_BAG_GATE_ROUND_DIGITS,
        },
        "largest_bag_color_component_ratio_formula": {
            "floor": FROZEN_BAG_COMPONENT_GATE_FLOOR,
            "scale": FROZEN_BAG_COMPONENT_GATE_SCALE,
            "offset": FROZEN_BAG_COMPONENT_GATE_OFFSET,
            "round_digits": FROZEN_BAG_GATE_ROUND_DIGITS,
        },
        "derived_gates": {
            "decision_bag_color_ratio": decision_color_gate,
            "decision_largest_bag_color_component_ratio": decision_component_gate,
            "bag_color_ratio": color_gate,
            "largest_bag_color_component_ratio": component_gate,
            "logic": FROZEN_BAG_GATE_LOGIC,
        },
    }
    return {
        "baseline_sha256": baseline_sha256,
        "decision_bag_color_ratio": decision_color_gate,
        "decision_largest_bag_color_component_ratio": decision_component_gate,
        "bag_color_ratio": color_gate,
        "largest_bag_color_component_ratio": component_gate,
        "logic": FROZEN_BAG_GATE_LOGIC,
        "derivation_sha256": canonical_sha256(derivation),
    }


def inspect_jpeg(data: bytes) -> tuple[int, int]:
    """Validate a JPEG container enough to reject arbitrary byte strings."""

    if len(data) < 12 or data[:2] != b"\xff\xd8" or data[-2:] != b"\xff\xd9":
        raise ActualRecordError("raw frame is not a complete JPEG")
    offset = 2
    width = height = 0
    sof_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    while offset < len(data) - 2:
        if data[offset] != 0xFF:
            raise ActualRecordError("raw frame has an invalid JPEG marker boundary")
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker == 0xD9:
            break
        if marker == 0xDA:
            break
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(data):
            raise ActualRecordError("raw frame has a truncated JPEG segment")
        segment_length = int.from_bytes(data[offset : offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(data):
            raise ActualRecordError("raw frame has an invalid JPEG segment length")
        if marker in sof_markers:
            if segment_length < 8:
                raise ActualRecordError("raw frame has a truncated JPEG SOF segment")
            height = int.from_bytes(data[offset + 3 : offset + 5], "big")
            width = int.from_bytes(data[offset + 5 : offset + 7], "big")
        offset += segment_length
    if width <= 0 or height <= 0:
        raise ActualRecordError("raw frame has no valid JPEG dimensions")
    return width, height


def _validate_input_frame_bundle_value(value: object) -> _InputFrameBundle:
    bundle = _exact_keys(value, INPUT_FRAME_BUNDLE_KEYS, label="input frame bundle")
    if bundle["schema_version"] != INPUT_FRAME_BUNDLE_SCHEMA_VERSION:
        raise ActualRecordError("input frame bundle schema is unsupported")
    entries = _require_sequence(bundle["entries"], label="input frame bundle.entries")
    entry_count = _require_int(
        bundle["entry_count"],
        label="input frame bundle.entry_count",
        minimum=1,
    )
    if entry_count != len(entries):
        raise ActualRecordError("input frame bundle entry_count mismatch")
    if entry_count > MAX_INPUT_FRAME_COUNT:
        raise ActualRecordError("input frame bundle has too many entries")

    names: set[str] = set()
    digests: set[str] = set()
    identities: list[tuple[str, str, str]] = []
    computed_total = 0
    for index, row_value in enumerate(entries):
        row = _exact_keys(
            row_value,
            INPUT_FRAME_BUNDLE_ENTRY_KEYS,
            label=f"input frame bundle.entries[{index}]",
        )
        role = _require_text(row["role"], label=f"bundle.entries[{index}].role")
        if role not in INPUT_FRAME_ROLES:
            raise ActualRecordError(f"bundle.entries[{index}].role is unsupported")
        file_name = _require_safe_basename(
            row["file_name"],
            label=f"bundle.entries[{index}].file_name",
        )
        digest = _require_sha256(
            row["sha256"],
            label=f"bundle.entries[{index}].sha256",
        )
        if file_name in names:
            raise ActualRecordError("input frame bundle contains duplicate file names")
        if digest in digests:
            raise ActualRecordError("input frame bundle contains duplicate content hashes")
        names.add(file_name)
        digests.add(digest)

        size_bytes = _require_int(
            row["size_bytes"],
            label=f"bundle.entries[{index}].size_bytes",
            minimum=1,
        )
        if size_bytes > MAX_FRAME_BYTES:
            raise ActualRecordError(f"bundle.entries[{index}] exceeds the per-frame byte cap")
        encoded = row["jpeg_base64"]
        if type(encoded) is not str or not encoded:
            raise ActualRecordError(f"bundle.entries[{index}].jpeg_base64 must be strict base64 text")
        try:
            encoded_bytes = encoded.encode("ascii")
            jpeg = base64.b64decode(encoded_bytes, validate=True)
        except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
            raise ActualRecordError(f"bundle.entries[{index}].jpeg_base64 is malformed") from exc
        if base64.b64encode(jpeg) != encoded_bytes:
            raise ActualRecordError(f"bundle.entries[{index}].jpeg_base64 is not canonical")
        if len(jpeg) != size_bytes:
            raise ActualRecordError(f"bundle.entries[{index}] size does not match decoded JPEG")
        if hashlib.sha256(jpeg).hexdigest() != digest:
            raise ActualRecordError(f"bundle.entries[{index}] digest does not match decoded JPEG")
        width, height = inspect_jpeg(jpeg)
        claimed_width = _require_int(
            row["width"],
            label=f"bundle.entries[{index}].width",
            minimum=1,
        )
        claimed_height = _require_int(
            row["height"],
            label=f"bundle.entries[{index}].height",
            minimum=1,
        )
        if (claimed_width, claimed_height) != (width, height):
            raise ActualRecordError(f"bundle.entries[{index}] JPEG dimensions mismatch")
        computed_total += size_bytes
        if computed_total > MAX_INPUT_FRAME_TOTAL_BYTES:
            raise ActualRecordError("input frame bundle exceeds the total decoded byte cap")
        identities.append((role, file_name, digest))

    total_bytes = _require_int(
        bundle["total_bytes"],
        label="input frame bundle.total_bytes",
        minimum=1,
    )
    if total_bytes != computed_total:
        raise ActualRecordError("input frame bundle total_bytes mismatch")
    claimed_bundle_sha256 = _require_sha256(
        bundle["bundle_sha256"],
        label="input frame bundle.bundle_sha256",
    )
    unsigned = dict(bundle)
    unsigned.pop("bundle_sha256")
    if canonical_sha256(unsigned) != claimed_bundle_sha256:
        raise ActualRecordError("input frame bundle bundle_sha256 mismatch")
    return _InputFrameBundle(
        value=bundle,
        identities=tuple(identities),
        bundle_sha256=claimed_bundle_sha256,
        entry_count=entry_count,
        total_bytes=total_bytes,
    )


def build_input_frame_bundle(
    role_paths: Sequence[tuple[str, Path | str]],
    *,
    output: Path | str | None = None,
) -> dict[str, Any]:
    """Build a deterministic, self-contained bundle of exact inference JPEG bytes."""

    if isinstance(role_paths, (str, bytes)):
        raise ActualRecordError("role_paths must be an ordered sequence of role/path pairs")
    pairs = list(role_paths)
    if not pairs or len(pairs) > MAX_INPUT_FRAME_COUNT:
        raise ActualRecordError(f"role_paths must contain between 1 and {MAX_INPUT_FRAME_COUNT} frames")
    entries: list[dict[str, Any]] = []
    names: set[str] = set()
    digests: set[str] = set()
    total_bytes = 0
    for index, pair in enumerate(pairs):
        if type(pair) not in {tuple, list} or len(pair) != 2:
            raise ActualRecordError(f"role_paths[{index}] must be one role/path pair")
        role, path = pair
        if type(role) is not str or role not in INPUT_FRAME_ROLES:
            raise ActualRecordError(f"role_paths[{index}] has an unsupported role")
        evidence = _read_regular_file(
            path,
            label=f"input frame {index}",
            max_bytes=MAX_FRAME_BYTES,
        )
        file_name = _require_safe_basename(
            evidence.path.name,
            label=f"input frame {index} basename",
        )
        if file_name in names:
            raise ActualRecordError("input frames contain duplicate file names")
        if evidence.sha256 in digests:
            raise ActualRecordError("input frames contain duplicate content hashes")
        names.add(file_name)
        digests.add(evidence.sha256)
        width, height = inspect_jpeg(evidence.data)
        total_bytes += evidence.size_bytes
        if total_bytes > MAX_INPUT_FRAME_TOTAL_BYTES:
            raise ActualRecordError("input frames exceed the total decoded byte cap")
        entries.append(
            {
                "role": role,
                "file_name": file_name,
                "sha256": evidence.sha256,
                "size_bytes": evidence.size_bytes,
                "width": width,
                "height": height,
                "jpeg_base64": base64.b64encode(evidence.data).decode("ascii"),
            }
        )
    bundle: dict[str, Any] = {
        "schema_version": INPUT_FRAME_BUNDLE_SCHEMA_VERSION,
        "entries": entries,
        "entry_count": len(entries),
        "total_bytes": total_bytes,
    }
    bundle["bundle_sha256"] = canonical_sha256(bundle)
    _validate_input_frame_bundle_value(bundle)
    if output is not None:
        write_json_once(output, bundle)
    return bundle


def _validate_frame_bundle_binding(
    bundle: _InputFrameBundle,
    semantics: _ResultSemantics,
    *,
    raw_frame: _FileEvidence,
) -> None:
    if semantics.schema == "xrd-overhead-bag-presence-v2":
        expected_identities = semantics.input_frame_identities
    else:
        expected_identities = (("STATION_OBSERVATION", raw_frame.path.name, raw_frame.sha256),)
    if bundle.identities != expected_identities:
        raise ActualRecordError(
            "input frame bundle order or identities do not match the authoritative result"
        )
    if semantics.schema == "xrd-overhead-bag-presence-v2":
        required_roles = (
            *("EMPTY_BASELINE" for _ in range(BAG_EMPTY_FRAME_COUNT)),
            *("OCCUPIED_CANDIDATE" for _ in range(BAG_OCCUPIED_FRAME_COUNT)),
        )
        observed_roles = tuple(role for role, _, _ in bundle.identities)
        if bundle.entry_count != BAG_INPUT_FRAME_COUNT or observed_roles != required_roles:
            raise ActualRecordError(
                "production bag bundle requires exactly 5 EMPTY_BASELINE followed by "
                "5 OCCUPIED_CANDIDATE frames"
            )
    expected_role = (
        "OCCUPIED_CANDIDATE" if semantics.schema == "xrd-overhead-bag-presence-v2" else "STATION_OBSERVATION"
    )
    raw_identity = (expected_role, raw_frame.path.name, raw_frame.sha256)
    if bundle.identities.count(raw_identity) != 1:
        raise ActualRecordError("raw frame must appear exactly once in its required bundle role")


def _validate_bag_result(
    payload: Mapping[str, Any],
    *,
    raw_frame_sha256: str,
    raw_frame_name: str | None,
) -> _ResultSemantics:
    result = _exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "generated_at_unix",
                "processor",
                "motion_authority",
                "camera_orientation",
                "dynamic_dish_relocalization",
                "frame",
                "dish_roi",
                "empty",
                "decision",
                "occupied",
            }
        ),
        label="bag result",
    )
    if result["processor"] != "AI brain X5 CPU / OpenCV":
        raise ActualRecordError("bag result processor is not the frozen AI-X5 pipeline")
    if _require_bool(result["motion_authority"], label="motion_authority"):
        raise ActualRecordError("bag result must not have motion authority")
    if result["camera_orientation"] != "upright":
        raise ActualRecordError("bag result camera orientation is not upright")
    if not _require_bool(result["dynamic_dish_relocalization"], label="dynamic_dish_relocalization"):
        raise ActualRecordError("bag result did not dynamically locate the dish")
    decision = result["decision"]
    if decision not in {"BAG_PRESENT", "BAG_NOT_DETECTED"}:
        raise ActualRecordError("bag result decision is not a normal terminal outcome")

    frame = _exact_keys(result["frame"], frozenset({"width", "height"}), label="bag result frame")
    width = _require_int(frame["width"], label="frame.width", minimum=1)
    height = _require_int(frame["height"], label="frame.height", minimum=1)
    _validate_roi(result["dish_roi"], label="bag result dish_roi")

    _, empty_files, _, _, _ = _validated_bag_empty_baseline(result["empty"])
    gate_contract = derive_bag_gate_contract(result["empty"])
    empty_names = {str(row["name"]) for row in empty_files}
    empty_digests = {str(row["sha256"]) for row in empty_files}

    occupied = _exact_keys(
        result["occupied"],
        frozenset({"count", "positive_count", "majority_pass", "gates", "files"}),
        label="bag result occupied",
    )
    occupied_files = _require_sequence(occupied["files"], label="occupied.files")
    occupied_count = _require_int(occupied["count"], label="occupied.count", minimum=1)
    positive_count = _require_int(occupied["positive_count"], label="occupied.positive_count")
    if occupied_count != len(occupied_files) or positive_count > occupied_count:
        raise ActualRecordError("occupied counts are inconsistent")
    if occupied_count != BAG_OCCUPIED_FRAME_COUNT:
        raise ActualRecordError(
            f"production bag inference requires exactly {BAG_OCCUPIED_FRAME_COUNT} OCCUPIED_CANDIDATE frames"
        )
    majority_pass = _require_bool(occupied["majority_pass"], label="occupied.majority_pass")

    gates = _exact_keys(
        occupied["gates"],
        frozenset({"bag_color_ratio", "largest_bag_color_component_ratio", "logic"}),
        label="occupied.gates",
    )
    color_gate = _require_number(gates["bag_color_ratio"], label="gates.bag_color_ratio", positive=True)
    component_gate = _require_number(
        gates["largest_bag_color_component_ratio"],
        label="gates.largest_bag_color_component_ratio",
        positive=True,
    )
    if (
        color_gate != gate_contract["bag_color_ratio"]
        or component_gate != gate_contract["largest_bag_color_component_ratio"]
        or gates["logic"] != gate_contract["logic"]
    ):
        raise ActualRecordError("occupied gates do not match the frozen empty-baseline derivation")
    decision_color_gate = float(gate_contract["decision_bag_color_ratio"])
    decision_component_gate = float(gate_contract["decision_largest_bag_color_component_ratio"])

    computed_positives = 0
    raw_frame_states: list[bool] = []
    occupied_names: set[str] = set()
    occupied_digests: set[str] = set()
    for index, row_value in enumerate(occupied_files):
        row = _exact_keys(
            row_value,
            frozenset({"name", "sha256", "dish_roi", "metrics", "bag_present", "annotated"}),
            label=f"occupied.files[{index}]",
        )
        row_name = _require_safe_basename(
            row["name"],
            label=f"occupied.files[{index}].name",
        )
        row_sha256 = _require_sha256(row["sha256"], label=f"occupied.files[{index}].sha256")
        if row_name in occupied_names or row_sha256 in occupied_digests:
            raise ActualRecordError("occupied result contains duplicate file identity")
        if row_name in empty_names or row_sha256 in empty_digests:
            raise ActualRecordError("empty and occupied result identities overlap")
        occupied_names.add(row_name)
        occupied_digests.add(row_sha256)
        _validate_roi(row["dish_roi"], label=f"occupied.files[{index}].dish_roi")
        metrics = _validate_metrics(row["metrics"], label=f"occupied.files[{index}].metrics")
        present = _require_bool(row["bag_present"], label=f"occupied.files[{index}].bag_present")
        expected_present = (
            _require_number(
                metrics["bag_color_ratio"],
                label=f"occupied.files[{index}].metrics.bag_color_ratio",
            )
            >= decision_color_gate
            and _require_number(
                metrics["largest_bag_color_component_ratio"],
                label=(f"occupied.files[{index}].metrics.largest_bag_color_component_ratio"),
            )
            >= decision_component_gate
        )
        if present is not expected_present:
            raise ActualRecordError(f"occupied.files[{index}] bag_present contradicts the frozen gates")
        _require_safe_basename(row["annotated"], label=f"occupied.files[{index}].annotated")
        computed_positives += int(present)
        if row_sha256 == raw_frame_sha256 and (raw_frame_name is None or row_name == raw_frame_name):
            raw_frame_states.append(present)
    if computed_positives != positive_count:
        raise ActualRecordError("occupied positive_count contradicts per-frame decisions")
    computed_majority = computed_positives > occupied_count / 2
    if majority_pass is not computed_majority:
        raise ActualRecordError("occupied majority_pass contradicts positive_count")
    expected_decision = "BAG_PRESENT" if computed_majority else "BAG_NOT_DETECTED"
    if decision != expected_decision:
        raise ActualRecordError("bag decision contradicts the computed majority")
    if len(raw_frame_states) != 1:
        raise ActualRecordError("raw frame is absent from the current occupied-frame rows")
    if any(state is not computed_majority for state in raw_frame_states):
        raise ActualRecordError("raw frame row is ambiguous or contradicts the terminal decision")

    generated_at = _require_number(result["generated_at_unix"], label="generated_at_unix", positive=True)
    return _ResultSemantics(
        schema="xrd-overhead-bag-presence-v2",
        state=expected_decision,
        success=computed_majority,
        event_at_ms=round(generated_at * 1000),
        width=width,
        height=height,
        baseline_sha256=str(gate_contract["baseline_sha256"]),
        derivation_sha256=str(gate_contract["derivation_sha256"]),
        input_frame_identities=tuple(
            ("EMPTY_BASELINE", str(row["name"]), str(row["sha256"])) for row in empty_files
        )
        + tuple(("OCCUPIED_CANDIDATE", str(row["name"]), str(row["sha256"])) for row in occupied_files),
    )


def _validate_marker_row(value: object, *, label: str) -> Mapping[str, Any]:
    row = _exact_keys(
        value,
        frozenset({"id", "center_px", "mean_edge_px", "corners_px"}),
        label=label,
    )
    _require_int(row["id"], label=f"{label}.id")
    _validate_pair(row["center_px"], label=f"{label}.center_px")
    _require_number(row["mean_edge_px"], label=f"{label}.mean_edge_px", positive=True)
    corners = _require_sequence(row["corners_px"], label=f"{label}.corners_px")
    if len(corners) != 4:
        raise ActualRecordError(f"{label}.corners_px must contain four corners")
    for index, corner in enumerate(corners):
        _validate_pair(corner, label=f"{label}.corners_px[{index}]")
    return row


def _validate_station_result(
    payload: Mapping[str, Any],
    *,
    raw_frame_sha256: str,
) -> _ResultSemantics:
    result = _exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "captured_at_unix",
                "raw_frame_sha256",
                "station_ok",
                "dictionary",
                "expected_marker_id",
                "printed_marker_size_mm",
                "selected",
                "detected",
                "frame",
                "phase",
                "coordinate_correction_enabled",
                "motion_authority",
            }
        ),
        label="station result",
    )
    bound_frame_sha256 = _require_sha256(
        result["raw_frame_sha256"],
        label="station raw_frame_sha256",
    )
    if bound_frame_sha256 != raw_frame_sha256:
        raise ActualRecordError("station result raw_frame_sha256 does not match the supplied frame")
    station_ok = _require_bool(result["station_ok"], label="station_ok")
    if result["dictionary"] != "DICT_APRILTAG_36h11":
        raise ActualRecordError("station result uses an unsupported marker dictionary")
    expected_id = _require_int(result["expected_marker_id"], label="expected_marker_id")
    _require_number(result["printed_marker_size_mm"], label="printed_marker_size_mm", positive=True)
    detected = _require_sequence(result["detected"], label="station detected")
    detected_rows = [
        _validate_marker_row(row, label=f"station detected[{index}]") for index, row in enumerate(detected)
    ]
    selected_value = result["selected"]
    selected: Mapping[str, Any] | None
    if selected_value is None:
        selected = None
        if any(row["id"] == expected_id for row in detected_rows):
            raise ActualRecordError("station selected marker is missing despite a matching detection")
    else:
        selected = _validate_marker_row(selected_value, label="station selected")
        if selected["id"] != expected_id:
            raise ActualRecordError("selected marker does not match expected_marker_id")
        if not any(row == selected for row in detected_rows):
            raise ActualRecordError("selected marker is absent from detected markers")
    computed_station_ok = (
        selected is not None
        and _require_number(selected["mean_edge_px"], label="station selected.mean_edge_px")
        >= FROZEN_STATION_MIN_EDGE_PX
    )
    if station_ok is not computed_station_ok:
        raise ActualRecordError("station_ok contradicts the frozen marker gate")
    frame = _exact_keys(result["frame"], frozenset({"width", "height"}), label="station frame")
    width = _require_int(frame["width"], label="frame.width", minimum=1)
    height = _require_int(frame["height"], label="frame.height", minimum=1)
    if result["phase"] != "presence_gate_only":
        raise ActualRecordError("station result is not the frozen presence-only gate")
    if _require_bool(result["coordinate_correction_enabled"], label="coordinate_correction_enabled"):
        raise ActualRecordError("station result unexpectedly enables coordinate correction")
    if _require_bool(result["motion_authority"], label="motion_authority"):
        raise ActualRecordError("station result unexpectedly has motion authority")
    captured_at = _require_number(result["captured_at_unix"], label="captured_at_unix", positive=True)
    derivation_sha256 = canonical_sha256(
        {
            "schema_version": STATION_GATE_DERIVATION_SCHEMA_VERSION,
            "raw_frame_sha256": bound_frame_sha256,
            "dictionary": result["dictionary"],
            "expected_marker_id": expected_id,
            "minimum_mean_edge_px": FROZEN_STATION_MIN_EDGE_PX,
            "phase": result["phase"],
            "coordinate_correction_enabled": False,
            "motion_authority": False,
        }
    )
    return _ResultSemantics(
        schema="xrd-grinding-overhead-gate-v1",
        state="STATION_OK" if station_ok else "STATION_NOT_OK",
        success=station_ok,
        event_at_ms=round(captured_at * 1000),
        width=width,
        height=height,
        baseline_sha256=None,
        derivation_sha256=derivation_sha256,
        input_frame_identities=(),
    )


def inspect_result_json(
    data: bytes,
    *,
    raw_frame_sha256: str,
    raw_frame_name: str | None = None,
) -> _ResultSemantics:
    payload = _load_json_bytes(data, label="result JSON")
    schema = payload.get("schema_version")
    if schema not in SUPPORTED_RESULT_SCHEMAS:
        raise ActualRecordError(f"unsupported overhead result schema: {schema!r}")
    if schema == "xrd-overhead-bag-presence-v2":
        return _validate_bag_result(
            payload,
            raw_frame_sha256=raw_frame_sha256,
            raw_frame_name=raw_frame_name,
        )
    return _validate_station_result(payload, raw_frame_sha256=raw_frame_sha256)


def _validate_task_result_semantics(
    semantics: _ResultSemantics,
    expected: ExpectedAcquisition,
) -> None:
    if semantics.schema != expected.result_schema:
        raise ActualRecordError("result schema does not match the challenged task contract")
    expected_success = semantics.state == expected.success_state
    if semantics.success is not expected_success:
        raise ActualRecordError("result success is not equivalent to state == success_state")


def _artifact_value(evidence: _FileEvidence, contract: _ArtifactContract) -> dict[str, Any]:
    return {
        "role": contract.role,
        "name": contract.name,
        "sha256": evidence.sha256,
        "size_bytes": evidence.size_bytes,
    }


def _validate_pipeline_artifact(evidence: _FileEvidence, contract: _ArtifactContract, *, label: str) -> None:
    if evidence.path.name != contract.name:
        raise ActualRecordError(f"{label} file name is not frozen: {evidence.path.name}")
    if evidence.sha256 != contract.sha256 or evidence.size_bytes != contract.size_bytes:
        raise ActualRecordError(f"{label} does not match the frozen artifact digest")
    try:
        source = evidence.data.decode("utf-8")
        tree = ast.parse(source, filename=contract.name)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ActualRecordError(f"{label} is not valid frozen Python source") from exc
    if not any(isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in ast.walk(tree)):
        raise ActualRecordError(f"{label} has no executable pipeline definition")


def _validate_host(
    value: object,
    *,
    label: str,
    frozen_hostname: str,
    expected_device_id: str,
    expected_boot_id: str,
    expected_session_id: str,
) -> Mapping[str, Any]:
    host = _exact_keys(value, HOST_KEYS, label=label)
    if host["hostname"] != frozen_hostname:
        raise ActualRecordError(f"{label}.hostname mismatch")
    comparisons = {
        "device_id": expected_device_id,
        "boot_id": expected_boot_id,
        "session_id": expected_session_id,
    }
    for field, expected in comparisons.items():
        actual_value = _require_text(host[field], label=f"{label}.{field}")
        if actual_value != expected:
            raise ActualRecordError(f"{label}.{field} mismatch")
    return host


def _validate_manifest_hash(manifest: Mapping[str, Any]) -> None:
    claimed = _require_sha256(manifest["manifest_sha256"], label="manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256")
    if canonical_sha256(unsigned) != claimed:
        raise ActualRecordError("manifest_sha256 mismatch")


def _validate_event_times(
    value: object,
    *,
    now_ms: int,
    max_age_ms: int,
    max_future_skew_ms: int,
    max_acquisition_span_ms: int,
    max_inference_duration_ms: int,
) -> Mapping[str, int]:
    events = _exact_keys(value, EVENT_KEYS, label="manifest.events")
    parsed = {
        name: _require_int(events[name], label=f"events.{name}", minimum=MIN_TIMESTAMP_MS)
        for name in EVENT_KEYS
    }
    ordered_names = (
        "producer_started_at_ms",
        "camera_service_identity_observed_at_ms",
        "camera_opened_at_ms",
        "frame_captured_at_ms",
        "input_frame_bundle_bound_at_ms",
        "inference_started_at_ms",
        "inference_completed_at_ms",
        "manifest_emitted_at_ms",
    )
    ordered = [parsed[name] for name in ordered_names]
    if ordered != sorted(ordered):
        raise ActualRecordError("acquisition event timestamps are out of order")
    if parsed["manifest_emitted_at_ms"] - parsed["producer_started_at_ms"] > max_acquisition_span_ms:
        raise ActualRecordError("acquisition span exceeds the allowed bound")
    if parsed["inference_completed_at_ms"] - parsed["inference_started_at_ms"] > max_inference_duration_ms:
        raise ActualRecordError("inference duration exceeds the allowed bound")
    if parsed["manifest_emitted_at_ms"] > now_ms + max_future_skew_ms:
        raise ActualRecordError("acquisition manifest timestamp is in the future")
    if now_ms - parsed["manifest_emitted_at_ms"] > max_age_ms:
        raise ActualRecordError("acquisition manifest is stale")
    return parsed


def _require_distinct_files(*files: _FileEvidence) -> None:
    paths = {item.path for item in files}
    identities = {(item.device, item.inode) for item in files}
    if len(paths) != len(files) or len(identities) != len(files):
        raise ActualRecordError("all manifest, bundle, frame, result, and pipeline files must be distinct")


def validate_acquisition_manifest_value(
    manifest: Mapping[str, Any],
    *,
    raw_frame: Path | str,
    frame_bundle_artifact: Path | str,
    result_json: Path | str,
    capture_pipeline_artifact: Path | str,
    inference_pipeline_artifact: Path | str,
    camera_service_identity_artifact: Path | str,
    expected: ExpectedAcquisition,
    now_ms: int | None = None,
    max_age_ms: int = DEFAULT_MAX_AGE_MS,
    max_future_skew_ms: int = DEFAULT_MAX_FUTURE_SKEW_MS,
    max_acquisition_span_ms: int = DEFAULT_MAX_ACQUISITION_SPAN_MS,
    max_inference_duration_ms: int = DEFAULT_MAX_INFERENCE_DURATION_MS,
) -> _ValidatedAcquisition:
    """Validate a parsed producer manifest against live challenge and files."""

    expected.validate()
    exact = _exact_keys(manifest, ACQUISITION_MANIFEST_KEYS, label="acquisition manifest")
    if exact["schema_version"] != ACQUISITION_SCHEMA_VERSION:
        raise ActualRecordError("acquisition manifest schema is unsupported")
    if exact["dual_arm_semantic_profile_sha256"] != DUAL_ARM_SEMANTIC_PROFILE_SHA256:
        raise ActualRecordError("acquisition manifest semantic profile mismatch")
    _validate_manifest_hash(exact)

    challenge = {
        "acquisition_id": expected.acquisition_id,
        "a0_run_id": expected.a0_run_id,
        "r2_run_id": expected.r2_run_id,
        "release_id": expected.release_id,
        "config_sha256": expected.config_sha256,
        "case_id": expected.case_id,
        "sample_id": expected.sample_id,
        "sample_lineage_sha256": expected.sample_lineage_sha256,
        "parent_evidence_root_sha256": expected.parent_evidence_root_sha256,
        "task_kind": expected.task_kind,
        "result_schema": expected.result_schema,
        "success_state": expected.success_state,
    }
    for field, expected_value in challenge.items():
        actual_value = exact[field]
        if field.endswith("_sha256"):
            _require_sha256(actual_value, label=f"manifest.{field}")
        else:
            _require_text(actual_value, label=f"manifest.{field}")
        if actual_value != expected_value:
            raise ActualRecordError(f"manifest.{field} challenge mismatch or replay")
    baseline_sha256 = _require_optional_sha256(
        exact["bag_empty_baseline_sha256"],
        label="manifest.bag_empty_baseline_sha256",
    )
    if baseline_sha256 != expected.bag_empty_baseline_sha256:
        raise ActualRecordError("manifest.bag_empty_baseline_sha256 challenge mismatch or replay")
    nonce_hash = _require_sha256(exact["r2_run_nonce_sha256"], label="manifest.r2_run_nonce_sha256")
    expected_nonce_hash = hashlib.sha256(expected.r2_run_nonce.encode("utf-8")).hexdigest()
    if nonce_hash != expected_nonce_hash:
        raise ActualRecordError("manifest.r2_run_nonce_sha256 challenge mismatch or replay")
    challenge_sha256 = _require_sha256(exact["challenge_sha256"], label="manifest.challenge_sha256")
    if challenge_sha256 != expected.challenge_sha256:
        raise ActualRecordError("manifest.challenge_sha256 challenge mismatch or replay")
    issued_at_ms = _require_int(
        exact["challenge_issued_at_ms"],
        label="manifest.challenge_issued_at_ms",
        minimum=MIN_TIMESTAMP_MS,
    )
    expires_at_ms = _require_int(
        exact["challenge_expires_at_ms"],
        label="manifest.challenge_expires_at_ms",
        minimum=MIN_TIMESTAMP_MS,
    )
    if issued_at_ms != expected.challenge_issued_at_ms:
        raise ActualRecordError("manifest.challenge_issued_at_ms challenge mismatch or replay")
    if expires_at_ms != expected.challenge_expires_at_ms:
        raise ActualRecordError("manifest.challenge_expires_at_ms challenge mismatch or replay")
    camera_service_identity_evidence = _read_regular_file(
        camera_service_identity_artifact,
        label="camera service identity artifact",
        max_bytes=MAX_CAMERA_SERVICE_IDENTITY_BYTES,
    )
    camera_service_identity = _validate_camera_service_identity_value(
        _load_json_bytes(
            camera_service_identity_evidence.data,
            label="camera service identity artifact",
        ),
        expected=expected,
    )
    expected_replay_identity = replay_identity_sha256_for_expected(
        expected,
        camera_service_identity_sha256=camera_service_identity.artifact_sha256,
    )
    if (
        _require_sha256(
            exact["replay_identity_sha256"],
            label="manifest.replay_identity_sha256",
        )
        != expected_replay_identity
    ):
        raise ActualRecordError("manifest.replay_identity_sha256 challenge mismatch or replay")

    _validate_host(
        exact["capture_host"],
        label="manifest.capture_host",
        frozen_hostname=FROZEN_CAPTURE_HOSTNAME,
        expected_device_id=expected.capture_device_id,
        expected_boot_id=expected.capture_boot_id,
        expected_session_id=expected.capture_session_id,
    )
    _validate_host(
        exact["inference_host"],
        label="manifest.inference_host",
        frozen_hostname=FROZEN_INFERENCE_HOSTNAME,
        expected_device_id=expected.inference_device_id,
        expected_boot_id=expected.inference_boot_id,
        expected_session_id=expected.inference_session_id,
    )

    camera = _exact_keys(exact["camera"], CAMERA_KEYS, label="manifest.camera")
    frozen_camera = {
        "owner": FROZEN_CAMERA_OWNER,
        "source": FROZEN_CAMERA_SOURCE,
        "usb_id": FROZEN_CAMERA_USB_ID,
        "backend": FROZEN_BACKEND,
    }
    for field, expected_value in frozen_camera.items():
        if camera[field] != expected_value:
            raise ActualRecordError(f"manifest.camera.{field} mismatch")
    if not _USB_ID_RE.fullmatch(str(camera["usb_id"])):
        raise ActualRecordError("manifest.camera.usb_id is invalid")
    camera_service_reference = _exact_keys(
        exact["camera_service_identity"],
        CAMERA_SERVICE_REFERENCE_KEYS,
        label="manifest.camera_service_identity",
    )
    if dict(camera_service_reference) != _camera_service_identity_reference(
        camera_service_identity_evidence,
        camera_service_identity,
    ):
        raise ActualRecordError(
            "manifest.camera_service_identity does not bind the supplied runtime artifact"
        )

    authority = _exact_keys(exact["authority"], AUTHORITY_KEYS, label="manifest.authority")
    if authority["domain"] != ACQUISITION_AUTHORITY_DOMAIN:
        raise ActualRecordError("manifest acquisition authority domain mismatch")
    for field in ("hardware_touched", "camera_opened", "inference_triggered"):
        if not _require_bool(authority[field], label=f"authority.{field}"):
            raise ActualRecordError(f"producer did not attest authority.{field}")
    for field in ("motion_authority", "robot_sdk_opened", "serial_opened", "gpio_opened"):
        if _require_bool(authority[field], label=f"authority.{field}"):
            raise ActualRecordError(f"forbidden producer authority asserted: {field}")
    if (
        _require_int(
            authority["actuator_commands_issued"],
            label="authority.actuator_commands_issued",
        )
        != 0
    ):
        raise ActualRecordError("producer issued actuator commands")

    current_ms = int(time.time() * 1000) if now_ms is None else now_ms
    for label, value in (
        ("now_ms", current_ms),
        ("max_age_ms", max_age_ms),
        ("max_future_skew_ms", max_future_skew_ms),
        ("max_acquisition_span_ms", max_acquisition_span_ms),
        ("max_inference_duration_ms", max_inference_duration_ms),
    ):
        _require_int(value, label=label, minimum=0)
    events = _validate_event_times(
        exact["events"],
        now_ms=current_ms,
        max_age_ms=max_age_ms,
        max_future_skew_ms=max_future_skew_ms,
        max_acquisition_span_ms=max_acquisition_span_ms,
        max_inference_duration_ms=max_inference_duration_ms,
    )
    if events["camera_service_identity_observed_at_ms"] != camera_service_identity.observed_at_ms:
        raise ActualRecordError("camera service identity observed_at does not match manifest events")
    if current_ms > expires_at_ms:
        raise ActualRecordError("A0 acquisition challenge expired")
    if issued_at_ms > current_ms + max_future_skew_ms:
        raise ActualRecordError("A0 acquisition challenge issuance is in the future")
    if issued_at_ms > events["producer_started_at_ms"]:
        raise ActualRecordError("producer started before the A0 challenge was issued")
    if events["manifest_emitted_at_ms"] > expires_at_ms:
        raise ActualRecordError("acquisition manifest was emitted after challenge expiry")

    frame_evidence = _read_regular_file(raw_frame, label="raw frame", max_bytes=MAX_FRAME_BYTES)
    frame_bundle_evidence = _read_regular_file(
        frame_bundle_artifact,
        label="input frame bundle artifact",
        max_bytes=MAX_INPUT_FRAME_BUNDLE_BYTES,
    )
    result_evidence = _read_regular_file(result_json, label="result JSON", max_bytes=MAX_RESULT_JSON_BYTES)
    capture_evidence = _read_regular_file(
        capture_pipeline_artifact,
        label="capture pipeline artifact",
        max_bytes=MAX_PIPELINE_ARTIFACT_BYTES,
    )
    inference_evidence = _read_regular_file(
        inference_pipeline_artifact,
        label="inference pipeline artifact",
        max_bytes=MAX_PIPELINE_ARTIFACT_BYTES,
    )
    _require_distinct_files(
        frame_evidence,
        frame_bundle_evidence,
        result_evidence,
        capture_evidence,
        inference_evidence,
        camera_service_identity_evidence,
    )

    width, height = inspect_jpeg(frame_evidence.data)
    parsed_frame_bundle = _validate_input_frame_bundle_value(
        _load_json_bytes(frame_bundle_evidence.data, label="input frame bundle artifact")
    )
    semantics = inspect_result_json(
        result_evidence.data,
        raw_frame_sha256=frame_evidence.sha256,
        raw_frame_name=frame_evidence.path.name,
    )
    _validate_task_result_semantics(semantics, expected)
    if semantics.baseline_sha256 != expected.bag_empty_baseline_sha256:
        raise ActualRecordError("result baseline does not match the coordinator challenge")
    _validate_frame_bundle_binding(
        parsed_frame_bundle,
        semantics,
        raw_frame=frame_evidence,
    )
    if (width, height) != (semantics.width, semantics.height):
        raise ActualRecordError("raw frame dimensions do not match result JSON")
    if abs(events["inference_completed_at_ms"] - semantics.event_at_ms) > 1_000:
        raise ActualRecordError("result timestamp does not match inference completion")

    frame = _exact_keys(exact["frame"], FRAME_KEYS, label="manifest.frame")
    frame_expected: dict[str, object] = {
        "file_name": frame_evidence.path.name,
        "sha256": frame_evidence.sha256,
        "size_bytes": frame_evidence.size_bytes,
        "media_type": "image/jpeg",
        "width": width,
        "height": height,
    }
    if dict(frame) != frame_expected:
        raise ActualRecordError("manifest.frame does not bind the supplied JPEG exactly")

    frame_bundle = _exact_keys(
        exact["frame_bundle"],
        FRAME_BUNDLE_KEYS,
        label="manifest.frame_bundle",
    )
    frame_bundle_expected: dict[str, object] = {
        "file_name": frame_bundle_evidence.path.name,
        "file_sha256": frame_bundle_evidence.sha256,
        "size_bytes": frame_bundle_evidence.size_bytes,
        "schema": INPUT_FRAME_BUNDLE_SCHEMA_VERSION,
        "bundle_sha256": parsed_frame_bundle.bundle_sha256,
        "entry_count": parsed_frame_bundle.entry_count,
        "total_bytes": parsed_frame_bundle.total_bytes,
    }
    if dict(frame_bundle) != frame_bundle_expected:
        raise ActualRecordError("manifest.frame_bundle does not bind the supplied bundle exactly")

    result = _exact_keys(exact["result"], RESULT_KEYS, label="manifest.result")
    result_expected: dict[str, object] = {
        "file_name": result_evidence.path.name,
        "sha256": result_evidence.sha256,
        "size_bytes": result_evidence.size_bytes,
        "schema": semantics.schema,
        "state": semantics.state,
        "success": semantics.success,
        "input_frame_sha256": frame_evidence.sha256,
        "baseline_sha256": semantics.baseline_sha256,
        "derivation_sha256": semantics.derivation_sha256,
    }
    if dict(result) != result_expected:
        raise ActualRecordError("manifest.result does not bind result and input frame exactly")

    _validate_pipeline_artifact(
        capture_evidence, CAPTURE_PIPELINE_CONTRACT, label="capture pipeline artifact"
    )
    inference_contract = INFERENCE_PIPELINE_CONTRACTS[semantics.schema]
    _validate_pipeline_artifact(inference_evidence, inference_contract, label="inference pipeline artifact")
    artifacts = _exact_keys(exact["artifacts"], ARTIFACT_KEYS, label="manifest.artifacts")
    capture_manifest = _exact_keys(
        artifacts["capture_pipeline"], PIPELINE_KEYS, label="artifacts.capture_pipeline"
    )
    inference_manifest = _exact_keys(
        artifacts["inference_pipeline"],
        PIPELINE_KEYS,
        label="artifacts.inference_pipeline",
    )
    if dict(capture_manifest) != _artifact_value(capture_evidence, CAPTURE_PIPELINE_CONTRACT):
        raise ActualRecordError("manifest capture pipeline digest mismatch")
    if dict(inference_manifest) != _artifact_value(inference_evidence, inference_contract):
        raise ActualRecordError("manifest inference pipeline digest mismatch")
    if artifacts["model_contract"] != NO_EXTERNAL_MODEL_CONTRACT:
        raise ActualRecordError("unsupported authoritative model contract")
    models = _require_sequence(artifacts["models"], label="artifacts.models")
    if models:
        raise ActualRecordError("current authoritative CPU/OpenCV pipelines must have no model artifacts")

    return _ValidatedAcquisition(
        manifest=exact,
        manifest_file=None,
        frame=frame_evidence,
        frame_bundle=frame_bundle_evidence,
        parsed_frame_bundle=parsed_frame_bundle,
        result=result_evidence,
        capture_pipeline=capture_evidence,
        inference_pipeline=inference_evidence,
        camera_service_identity_file=camera_service_identity_evidence,
        camera_service_identity=camera_service_identity,
        semantics=semantics,
    )


def validate_acquisition_manifest(
    acquisition_manifest: Path | str,
    *,
    raw_frame: Path | str,
    frame_bundle_artifact: Path | str,
    result_json: Path | str,
    capture_pipeline_artifact: Path | str,
    inference_pipeline_artifact: Path | str,
    camera_service_identity_artifact: Path | str,
    expected: ExpectedAcquisition,
    now_ms: int | None = None,
    max_age_ms: int = DEFAULT_MAX_AGE_MS,
    max_future_skew_ms: int = DEFAULT_MAX_FUTURE_SKEW_MS,
    max_acquisition_span_ms: int = DEFAULT_MAX_ACQUISITION_SPAN_MS,
    max_inference_duration_ms: int = DEFAULT_MAX_INFERENCE_DURATION_MS,
) -> _ValidatedAcquisition:
    manifest_file = _read_regular_file(
        acquisition_manifest,
        label="acquisition manifest",
        max_bytes=MAX_MANIFEST_BYTES,
    )
    manifest = _load_json_bytes(manifest_file.data, label="acquisition manifest")
    validated = validate_acquisition_manifest_value(
        manifest,
        raw_frame=raw_frame,
        frame_bundle_artifact=frame_bundle_artifact,
        result_json=result_json,
        capture_pipeline_artifact=capture_pipeline_artifact,
        inference_pipeline_artifact=inference_pipeline_artifact,
        camera_service_identity_artifact=camera_service_identity_artifact,
        expected=expected,
        now_ms=now_ms,
        max_age_ms=max_age_ms,
        max_future_skew_ms=max_future_skew_ms,
        max_acquisition_span_ms=max_acquisition_span_ms,
        max_inference_duration_ms=max_inference_duration_ms,
    )
    _require_distinct_files(
        manifest_file,
        validated.frame,
        validated.frame_bundle,
        validated.result,
        validated.capture_pipeline,
        validated.inference_pipeline,
        validated.camera_service_identity_file,
    )
    return _ValidatedAcquisition(
        manifest=validated.manifest,
        manifest_file=manifest_file,
        frame=validated.frame,
        frame_bundle=validated.frame_bundle,
        parsed_frame_bundle=validated.parsed_frame_bundle,
        result=validated.result,
        capture_pipeline=validated.capture_pipeline,
        inference_pipeline=validated.inference_pipeline,
        camera_service_identity_file=validated.camera_service_identity_file,
        camera_service_identity=validated.camera_service_identity,
        semantics=validated.semantics,
    )


def _build_record(validated: _ValidatedAcquisition) -> dict[str, Any]:
    if validated.manifest_file is None:
        raise ActualRecordError("record sealing requires a manifest file, not an in-memory claim")
    manifest = validated.manifest
    capture_host = manifest["capture_host"]
    inference_host = manifest["inference_host"]
    camera = manifest["camera"]
    camera_service_reference = manifest["camera_service_identity"]
    camera_service = validated.camera_service_identity.value
    authority = manifest["authority"]
    events = manifest["events"]
    frame = manifest["frame"]
    frame_bundle = manifest["frame_bundle"]
    result = manifest["result"]
    artifacts = manifest["artifacts"]
    capture_pipeline = artifacts["capture_pipeline"]
    inference_pipeline = artifacts["inference_pipeline"]

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "acquisition_manifest_schema": manifest["schema_version"],
        "acquisition_manifest_sha256": validated.manifest_file.sha256,
        "acquisition_manifest_size_bytes": validated.manifest_file.size_bytes,
        "acquisition_id": manifest["acquisition_id"],
        "a0_run_id": manifest["a0_run_id"],
        "r2_run_id": manifest["r2_run_id"],
        "r2_run_nonce_sha256": manifest["r2_run_nonce_sha256"],
        "challenge_sha256": manifest["challenge_sha256"],
        "challenge_issued_at_ms": manifest["challenge_issued_at_ms"],
        "challenge_expires_at_ms": manifest["challenge_expires_at_ms"],
        "replay_identity_sha256": manifest["replay_identity_sha256"],
        "config_sha256": manifest["config_sha256"],
        "case_id": manifest["case_id"],
        "sample_id": manifest["sample_id"],
        "sample_lineage_sha256": manifest["sample_lineage_sha256"],
        "parent_evidence_root_sha256": manifest["parent_evidence_root_sha256"],
        "bag_empty_baseline_sha256": manifest["bag_empty_baseline_sha256"],
        "task_kind": manifest["task_kind"],
        "success_state": manifest["success_state"],
        "capture_hostname": capture_host["hostname"],
        "capture_device_id": capture_host["device_id"],
        "capture_boot_id": capture_host["boot_id"],
        "capture_session_id": capture_host["session_id"],
        "inference_hostname": inference_host["hostname"],
        "ai_x5_device_id": inference_host["device_id"],
        "ai_x5_boot_id": inference_host["boot_id"],
        "ai_x5_session_id": inference_host["session_id"],
        "release_id": manifest["release_id"],
        "dual_arm_semantic_profile_sha256": manifest["dual_arm_semantic_profile_sha256"],
        "camera_owner": camera["owner"],
        "camera_source": camera["source"],
        "camera_usb_id": camera["usb_id"],
        "backend": camera["backend"],
        "camera_service_identity_schema": camera_service_reference["schema"],
        "camera_service_identity_file_name": camera_service_reference["file_name"],
        "camera_service_identity_file_sha256": camera_service_reference["file_sha256"],
        "camera_service_identity_size_bytes": camera_service_reference["size_bytes"],
        "camera_service_identity_sha256": camera_service_reference["artifact_sha256"],
        "camera_service_runtime_sha256": camera_service_reference["service_identity_sha256"],
        "camera_service_observed_at_ms": camera_service_reference["observed_at_ms"],
        "camera_service_main_pid": camera_service_reference["main_pid"],
        "camera_service_unit_name": camera_service["unit_name"],
        "camera_service_unit_path": camera_service["unit_fragment_path"],
        "camera_service_unit_sha256": camera_service["unit_sha256"],
        "camera_service_script_path": camera_service["script_path"],
        "camera_service_script_sha256": camera_service["script_sha256"],
        "camera_service_cmdline_sha256": canonical_sha256(camera_service["cmdline"]),
        "camera_service_video_owner_pid": camera_service["video_owner_pid"],
        "camera_service_video_owner_user": camera_service["video_owner_user"],
        "camera_service_usb_id": camera_service["usb_id"],
        "camera_service_listener_ip": camera_service["listener_ip"],
        "camera_service_listener_port": camera_service["listener_port"],
        "camera_service_listener_inode": camera_service["listener_inode"],
        "camera_service_listener_owner_pid": camera_service["listener_owner_pid"],
        "captured_at_ms": events["frame_captured_at_ms"],
        "observed_at_ms": events["inference_completed_at_ms"],
        "manifest_emitted_at_ms": events["manifest_emitted_at_ms"],
        "raw_frame_sha256": frame["sha256"],
        "raw_frame_size_bytes": frame["size_bytes"],
        "raw_frame_media_type": frame["media_type"],
        "raw_frame_width": frame["width"],
        "raw_frame_height": frame["height"],
        "frame_bundle_file_name": frame_bundle["file_name"],
        "frame_bundle_file_sha256": frame_bundle["file_sha256"],
        "frame_bundle_size_bytes": frame_bundle["size_bytes"],
        "frame_bundle_schema": frame_bundle["schema"],
        "frame_bundle_sha256": frame_bundle["bundle_sha256"],
        "frame_bundle_entry_count": frame_bundle["entry_count"],
        "frame_bundle_total_bytes": frame_bundle["total_bytes"],
        "result_json_sha256": result["sha256"],
        "result_json_size_bytes": result["size_bytes"],
        "result_schema": result["schema"],
        "result_state": result["state"],
        "result_success": result["success"],
        "result_input_frame_sha256": result["input_frame_sha256"],
        "result_baseline_sha256": result["baseline_sha256"],
        "result_derivation_sha256": result["derivation_sha256"],
        "capture_pipeline_name": capture_pipeline["name"],
        "capture_pipeline_sha256": capture_pipeline["sha256"],
        "capture_pipeline_size_bytes": capture_pipeline["size_bytes"],
        "inference_pipeline_name": inference_pipeline["name"],
        "inference_pipeline_sha256": inference_pipeline["sha256"],
        "inference_pipeline_size_bytes": inference_pipeline["size_bytes"],
        "model_contract": artifacts["model_contract"],
        "model_artifacts": list(artifacts["models"]),
        "acquisition_authority_domain": authority["domain"],
        "acquisition_hardware_touched": authority["hardware_touched"],
        "acquisition_camera_opened": authority["camera_opened"],
        "acquisition_inference_triggered": authority["inference_triggered"],
        "acquisition_motion_authority": authority["motion_authority"],
        "acquisition_robot_sdk_opened": authority["robot_sdk_opened"],
        "acquisition_serial_opened": authority["serial_opened"],
        "acquisition_gpio_opened": authority["gpio_opened"],
        "acquisition_actuator_commands_issued": authority["actuator_commands_issued"],
        "consumer_authority_domain": CONSUMER_AUTHORITY_DOMAIN,
        "sealer_hardware_touched": False,
        "sealer_execution_authority": False,
        "sealer_actuator_commands_issued": 0,
        "sealer_camera_opened": False,
        "sealer_inference_triggered": False,
    }
    record["record_sha256"] = canonical_sha256(record)
    return record


def _mint_runner_production_attestation(
    *,
    acquisition_manifest: Path | str,
    camera_service_identity_artifact: Path | str,
    frame_bundle_artifact: Path | str,
    expected: ExpectedAcquisition,
) -> RunnerProductionAttestation:
    """Mint the recorder-only in-process sealing capability after emission."""

    expected.validate()
    manifest_file = _read_regular_file(
        acquisition_manifest,
        label="acquisition manifest",
        max_bytes=MAX_MANIFEST_BYTES,
    )
    service_file = _read_regular_file(
        camera_service_identity_artifact,
        label="camera service identity artifact",
        max_bytes=MAX_CAMERA_SERVICE_IDENTITY_BYTES,
    )
    bundle_file = _read_regular_file(
        frame_bundle_artifact,
        label="input frame bundle artifact",
        max_bytes=MAX_INPUT_FRAME_BUNDLE_BYTES,
    )
    _require_distinct_files(manifest_file, service_file, bundle_file)
    manifest = _exact_keys(
        _load_json_bytes(manifest_file.data, label="acquisition manifest"),
        ACQUISITION_MANIFEST_KEYS,
        label="acquisition manifest",
    )
    if manifest["schema_version"] != ACQUISITION_SCHEMA_VERSION:
        raise ActualRecordError("runner attestation requires the current acquisition schema")
    _validate_manifest_hash(manifest)
    service_identity = _validate_camera_service_identity_value(
        _load_json_bytes(service_file.data, label="camera service identity artifact"),
        expected=expected,
    )
    service_reference = _camera_service_identity_reference(service_file, service_identity)
    if dict(manifest["camera_service_identity"]) != service_reference:
        raise ActualRecordError("runner attestation service identity binding mismatch")
    bundle = _validate_input_frame_bundle_value(
        _load_json_bytes(bundle_file.data, label="input frame bundle artifact")
    )
    frame_bundle_reference = manifest["frame_bundle"]
    if (
        frame_bundle_reference["file_sha256"] != bundle_file.sha256
        or frame_bundle_reference["bundle_sha256"] != bundle.bundle_sha256
    ):
        raise ActualRecordError("runner attestation frame bundle binding mismatch")
    replay_identity = replay_identity_sha256_for_expected(
        expected,
        camera_service_identity_sha256=service_identity.artifact_sha256,
    )
    if manifest["replay_identity_sha256"] != replay_identity:
        raise ActualRecordError("runner attestation replay identity mismatch")
    return RunnerProductionAttestation(
        schema_version=RUNNER_PRODUCTION_ATTESTATION_SCHEMA_VERSION,
        acquisition_manifest_file_sha256=manifest_file.sha256,
        manifest_sha256=str(manifest["manifest_sha256"]),
        camera_service_identity_file_sha256=service_file.sha256,
        camera_service_identity_sha256=service_identity.artifact_sha256,
        frame_bundle_file_sha256=bundle_file.sha256,
        frame_bundle_sha256=bundle.bundle_sha256,
        replay_identity_sha256=replay_identity,
        _capability=_RUNNER_PRODUCTION_CAPABILITY,
    )


def _validate_runner_production_attestation(
    attestation: RunnerProductionAttestation | None,
    validated: _ValidatedAcquisition,
) -> None:
    if type(attestation) is not RunnerProductionAttestation:
        raise ActualRecordError("production sealing requires a recorder-issued runner production attestation")
    if attestation._capability is not _RUNNER_PRODUCTION_CAPABILITY:
        raise ActualRecordError("runner production attestation capability is invalid")
    if attestation.schema_version != RUNNER_PRODUCTION_ATTESTATION_SCHEMA_VERSION:
        raise ActualRecordError("runner production attestation schema is unsupported")
    if validated.manifest_file is None:
        raise ActualRecordError("runner production attestation requires a manifest file")
    manifest = validated.manifest
    expected = {
        "acquisition_manifest_file_sha256": validated.manifest_file.sha256,
        "manifest_sha256": manifest["manifest_sha256"],
        "camera_service_identity_file_sha256": validated.camera_service_identity_file.sha256,
        "camera_service_identity_sha256": validated.camera_service_identity.artifact_sha256,
        "frame_bundle_file_sha256": validated.frame_bundle.sha256,
        "frame_bundle_sha256": validated.parsed_frame_bundle.bundle_sha256,
        "replay_identity_sha256": manifest["replay_identity_sha256"],
    }
    for field, expected_value in expected.items():
        if getattr(attestation, field) != expected_value:
            raise ActualRecordError(f"runner production attestation {field} mismatch")


def _seal_record_with_clock(
    *,
    acquisition_manifest: Path | str,
    raw_frame: Path | str,
    frame_bundle_artifact: Path | str,
    result_json: Path | str,
    capture_pipeline_artifact: Path | str,
    inference_pipeline_artifact: Path | str,
    camera_service_identity_artifact: Path | str,
    expected: ExpectedAcquisition,
    production_attestation: RunnerProductionAttestation | None = None,
    now_ms: int | None = None,
    max_age_ms: int = DEFAULT_MAX_AGE_MS,
    max_future_skew_ms: int = DEFAULT_MAX_FUTURE_SKEW_MS,
    max_acquisition_span_ms: int = DEFAULT_MAX_ACQUISITION_SPAN_MS,
    max_inference_duration_ms: int = DEFAULT_MAX_INFERENCE_DURATION_MS,
) -> dict[str, Any]:
    """Seal only facts already asserted and bound by the producer manifest."""

    validated = validate_acquisition_manifest(
        acquisition_manifest,
        raw_frame=raw_frame,
        frame_bundle_artifact=frame_bundle_artifact,
        result_json=result_json,
        capture_pipeline_artifact=capture_pipeline_artifact,
        inference_pipeline_artifact=inference_pipeline_artifact,
        camera_service_identity_artifact=camera_service_identity_artifact,
        expected=expected,
        now_ms=now_ms,
        max_age_ms=max_age_ms,
        max_future_skew_ms=max_future_skew_ms,
        max_acquisition_span_ms=max_acquisition_span_ms,
        max_inference_duration_ms=max_inference_duration_ms,
    )
    _validate_runner_production_attestation(production_attestation, validated)
    return _build_record(validated)


def seal_record(
    *,
    acquisition_manifest: Path | str,
    raw_frame: Path | str,
    frame_bundle_artifact: Path | str,
    result_json: Path | str,
    capture_pipeline_artifact: Path | str,
    inference_pipeline_artifact: Path | str,
    camera_service_identity_artifact: Path | str,
    expected: ExpectedAcquisition,
    production_attestation: RunnerProductionAttestation | None = None,
    max_age_ms: int = DEFAULT_MAX_AGE_MS,
    max_future_skew_ms: int = DEFAULT_MAX_FUTURE_SKEW_MS,
    max_acquisition_span_ms: int = DEFAULT_MAX_ACQUISITION_SPAN_MS,
    max_inference_duration_ms: int = DEFAULT_MAX_INFERENCE_DURATION_MS,
) -> dict[str, Any]:
    """Seal a producer record using the host clock for freshness checks."""

    return _seal_record_with_clock(
        acquisition_manifest=acquisition_manifest,
        raw_frame=raw_frame,
        frame_bundle_artifact=frame_bundle_artifact,
        result_json=result_json,
        capture_pipeline_artifact=capture_pipeline_artifact,
        inference_pipeline_artifact=inference_pipeline_artifact,
        camera_service_identity_artifact=camera_service_identity_artifact,
        expected=expected,
        production_attestation=production_attestation,
        now_ms=None,
        max_age_ms=max_age_ms,
        max_future_skew_ms=max_future_skew_ms,
        max_acquisition_span_ms=max_acquisition_span_ms,
        max_inference_duration_ms=max_inference_duration_ms,
    )


def _validate_record_with_clock(
    record: Mapping[str, Any],
    *,
    acquisition_manifest: Path | str,
    raw_frame: Path | str,
    frame_bundle_artifact: Path | str,
    result_json: Path | str,
    capture_pipeline_artifact: Path | str,
    inference_pipeline_artifact: Path | str,
    camera_service_identity_artifact: Path | str,
    expected: ExpectedAcquisition,
    now_ms: int | None = None,
    max_age_ms: int = DEFAULT_MAX_AGE_MS,
    max_future_skew_ms: int = DEFAULT_MAX_FUTURE_SKEW_MS,
    max_acquisition_span_ms: int = DEFAULT_MAX_ACQUISITION_SPAN_MS,
    max_inference_duration_ms: int = DEFAULT_MAX_INFERENCE_DURATION_MS,
) -> None:
    exact = _exact_keys(record, TOP_LEVEL_KEYS, label="actual record")
    if exact["schema_version"] != SCHEMA_VERSION:
        raise ActualRecordError("actual record schema is unsupported")
    claimed = _require_sha256(exact["record_sha256"], label="record_sha256")
    unsigned = dict(exact)
    unsigned.pop("record_sha256")
    if canonical_sha256(unsigned) != claimed:
        raise ActualRecordError("record_sha256 mismatch")
    validated = validate_acquisition_manifest(
        acquisition_manifest,
        raw_frame=raw_frame,
        frame_bundle_artifact=frame_bundle_artifact,
        result_json=result_json,
        capture_pipeline_artifact=capture_pipeline_artifact,
        inference_pipeline_artifact=inference_pipeline_artifact,
        camera_service_identity_artifact=camera_service_identity_artifact,
        expected=expected,
        now_ms=now_ms,
        max_age_ms=max_age_ms,
        max_future_skew_ms=max_future_skew_ms,
        max_acquisition_span_ms=max_acquisition_span_ms,
        max_inference_duration_ms=max_inference_duration_ms,
    )
    rebuilt = _build_record(validated)
    if dict(exact) != rebuilt:
        raise ActualRecordError("record differs from the producer manifest or source files")


def validate_record(
    record: Mapping[str, Any],
    *,
    acquisition_manifest: Path | str,
    raw_frame: Path | str,
    frame_bundle_artifact: Path | str,
    result_json: Path | str,
    capture_pipeline_artifact: Path | str,
    inference_pipeline_artifact: Path | str,
    camera_service_identity_artifact: Path | str,
    expected: ExpectedAcquisition,
    max_age_ms: int = DEFAULT_MAX_AGE_MS,
    max_future_skew_ms: int = DEFAULT_MAX_FUTURE_SKEW_MS,
    max_acquisition_span_ms: int = DEFAULT_MAX_ACQUISITION_SPAN_MS,
    max_inference_duration_ms: int = DEFAULT_MAX_INFERENCE_DURATION_MS,
) -> None:
    """Validate a producer record using the host clock for freshness checks."""

    _validate_record_with_clock(
        record,
        acquisition_manifest=acquisition_manifest,
        raw_frame=raw_frame,
        frame_bundle_artifact=frame_bundle_artifact,
        result_json=result_json,
        capture_pipeline_artifact=capture_pipeline_artifact,
        inference_pipeline_artifact=inference_pipeline_artifact,
        camera_service_identity_artifact=camera_service_identity_artifact,
        expected=expected,
        now_ms=None,
        max_age_ms=max_age_ms,
        max_future_skew_ms=max_future_skew_ms,
        max_acquisition_span_ms=max_acquisition_span_ms,
        max_inference_duration_ms=max_inference_duration_ms,
    )


def load_record(path: Path | str) -> dict[str, Any]:
    evidence = _read_regular_file(path, label="actual record", max_bytes=MAX_RECORD_BYTES)
    return _load_json_bytes(evidence.data, label="actual record")


def _safe_directory(path: Path | str, *, label: str) -> Path:
    lexical = _absolute_lexical_path(path)
    _reject_link_components(lexical)
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise ActualRecordError(f"{label} does not exist: {lexical}") from exc
    if resolved != lexical or not resolved.is_dir():
        raise ActualRecordError(f"{label} must be a literal existing directory")
    return resolved


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_once(path: Path | str, data: bytes) -> None:
    lexical = _absolute_lexical_path(path)
    parent = _safe_directory(lexical.parent, label="output parent")
    output = parent / lexical.name
    if output.exists() or output.is_symlink():
        raise ActualRecordError(f"output already exists: {output}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise ActualRecordError(f"output already exists: {output}") from exc
        temporary.unlink()
        _fsync_directory(parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_once(path: Path | str, value: Mapping[str, Any]) -> None:
    _atomic_write_once(path, canonical_json_bytes(value) + b"\n")


def _build_replay_receipt(record: Mapping[str, Any]) -> dict[str, Any]:
    replay_key = replay_identity_sha256_from_record(record)
    if record.get("replay_identity_sha256") != replay_key:
        raise ActualRecordError("record replay_identity_sha256 mismatch")
    receipt: dict[str, Any] = {
        "schema_version": REPLAY_RECEIPT_SCHEMA_VERSION,
        "replay_key": replay_key,
        "replay_identity_sha256": replay_key,
        "challenge_sha256": record["challenge_sha256"],
        "challenge_issued_at_ms": record["challenge_issued_at_ms"],
        "challenge_expires_at_ms": record["challenge_expires_at_ms"],
        "acquisition_id": record["acquisition_id"],
        "a0_run_id": record["a0_run_id"],
        "r2_run_id": record["r2_run_id"],
        "r2_run_nonce_sha256": record["r2_run_nonce_sha256"],
        "release_id": record["release_id"],
        "config_sha256": record["config_sha256"],
        "case_id": record["case_id"],
        "sample_id": record["sample_id"],
        "sample_lineage_sha256": record["sample_lineage_sha256"],
        "parent_evidence_root_sha256": record["parent_evidence_root_sha256"],
        "bag_empty_baseline_sha256": record["bag_empty_baseline_sha256"],
        "task_kind": record["task_kind"],
        "result_schema": record["result_schema"],
        "success_state": record["success_state"],
        "camera_service_identity_file_sha256": record["camera_service_identity_file_sha256"],
        "camera_service_identity_sha256": record["camera_service_identity_sha256"],
        "frame_bundle_file_sha256": record["frame_bundle_file_sha256"],
        "frame_bundle_sha256": record["frame_bundle_sha256"],
        "acquisition_manifest_sha256": record["acquisition_manifest_sha256"],
        "record_sha256": record["record_sha256"],
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def replay_receipt_path(
    replay_ledger_dir: Path | str,
    record: Mapping[str, Any],
) -> Path:
    ledger = _safe_directory(replay_ledger_dir, label="replay ledger directory")
    return ledger / f"{replay_identity_sha256_from_record(record)}.json"


def validate_replay_receipt(
    record: Mapping[str, Any],
    *,
    replay_ledger_dir: Path | str,
) -> ValidatedSealedRecord:
    """Validate the stable, write-once receipt for a reconstructed record."""

    receipt_path = replay_receipt_path(replay_ledger_dir, record)
    receipt_file = _read_regular_file(
        receipt_path,
        label="replay receipt",
        max_bytes=MAX_RECORD_BYTES,
    )
    receipt = _exact_keys(
        _load_json_bytes(receipt_file.data, label="replay receipt"),
        REPLAY_RECEIPT_KEYS,
        label="replay receipt",
    )
    if receipt["schema_version"] != REPLAY_RECEIPT_SCHEMA_VERSION:
        raise ActualRecordError("replay receipt schema is unsupported")
    claimed = _require_sha256(receipt["receipt_sha256"], label="receipt.receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256")
    if canonical_sha256(unsigned) != claimed:
        raise ActualRecordError("replay receipt digest mismatch")
    expected_receipt = _build_replay_receipt(record)
    if dict(receipt) != expected_receipt:
        raise ActualRecordError("replay receipt differs from the reconstructed record")
    return ValidatedSealedRecord(
        receipt_path=receipt_path,
        receipt_sha256=receipt_file.sha256,
        replay_identity_sha256=str(receipt["replay_identity_sha256"]),
    )


def _validate_sealed_record_with_clock(
    record: Mapping[str, Any],
    *,
    replay_ledger_dir: Path | str,
    acquisition_manifest: Path | str,
    raw_frame: Path | str,
    frame_bundle_artifact: Path | str,
    result_json: Path | str,
    capture_pipeline_artifact: Path | str,
    inference_pipeline_artifact: Path | str,
    camera_service_identity_artifact: Path | str,
    expected: ExpectedAcquisition,
    now_ms: int | None = None,
    max_age_ms: int = DEFAULT_MAX_AGE_MS,
    max_future_skew_ms: int = DEFAULT_MAX_FUTURE_SKEW_MS,
    max_acquisition_span_ms: int = DEFAULT_MAX_ACQUISITION_SPAN_MS,
    max_inference_duration_ms: int = DEFAULT_MAX_INFERENCE_DURATION_MS,
) -> ValidatedSealedRecord:
    """Reconstruct every A0 field from sources, then validate one-time use."""

    _validate_record_with_clock(
        record,
        acquisition_manifest=acquisition_manifest,
        raw_frame=raw_frame,
        frame_bundle_artifact=frame_bundle_artifact,
        result_json=result_json,
        capture_pipeline_artifact=capture_pipeline_artifact,
        inference_pipeline_artifact=inference_pipeline_artifact,
        camera_service_identity_artifact=camera_service_identity_artifact,
        expected=expected,
        now_ms=now_ms,
        max_age_ms=max_age_ms,
        max_future_skew_ms=max_future_skew_ms,
        max_acquisition_span_ms=max_acquisition_span_ms,
        max_inference_duration_ms=max_inference_duration_ms,
    )
    return validate_replay_receipt(record, replay_ledger_dir=replay_ledger_dir)


def validate_sealed_record(
    record: Mapping[str, Any],
    *,
    replay_ledger_dir: Path | str,
    acquisition_manifest: Path | str,
    raw_frame: Path | str,
    frame_bundle_artifact: Path | str,
    result_json: Path | str,
    capture_pipeline_artifact: Path | str,
    inference_pipeline_artifact: Path | str,
    camera_service_identity_artifact: Path | str,
    expected: ExpectedAcquisition,
    max_age_ms: int = DEFAULT_MAX_AGE_MS,
    max_future_skew_ms: int = DEFAULT_MAX_FUTURE_SKEW_MS,
    max_acquisition_span_ms: int = DEFAULT_MAX_ACQUISITION_SPAN_MS,
    max_inference_duration_ms: int = DEFAULT_MAX_INFERENCE_DURATION_MS,
) -> ValidatedSealedRecord:
    """Validate and consume-check a record using the host clock."""

    return _validate_sealed_record_with_clock(
        record,
        replay_ledger_dir=replay_ledger_dir,
        acquisition_manifest=acquisition_manifest,
        raw_frame=raw_frame,
        frame_bundle_artifact=frame_bundle_artifact,
        result_json=result_json,
        capture_pipeline_artifact=capture_pipeline_artifact,
        inference_pipeline_artifact=inference_pipeline_artifact,
        camera_service_identity_artifact=camera_service_identity_artifact,
        expected=expected,
        now_ms=None,
        max_age_ms=max_age_ms,
        max_future_skew_ms=max_future_skew_ms,
        max_acquisition_span_ms=max_acquisition_span_ms,
        max_inference_duration_ms=max_inference_duration_ms,
    )


def _seal_record_to_path_once_with_clock(
    *,
    output: Path | str,
    replay_ledger_dir: Path | str,
    acquisition_manifest: Path | str,
    raw_frame: Path | str,
    frame_bundle_artifact: Path | str,
    result_json: Path | str,
    capture_pipeline_artifact: Path | str,
    inference_pipeline_artifact: Path | str,
    camera_service_identity_artifact: Path | str,
    expected: ExpectedAcquisition,
    production_attestation: RunnerProductionAttestation | None = None,
    now_ms: int | None = None,
) -> dict[str, Any]:
    """Seal once and atomically consume the acquisition/run challenge."""

    record = _seal_record_with_clock(
        acquisition_manifest=acquisition_manifest,
        raw_frame=raw_frame,
        frame_bundle_artifact=frame_bundle_artifact,
        result_json=result_json,
        capture_pipeline_artifact=capture_pipeline_artifact,
        inference_pipeline_artifact=inference_pipeline_artifact,
        camera_service_identity_artifact=camera_service_identity_artifact,
        expected=expected,
        production_attestation=production_attestation,
        now_ms=now_ms,
    )
    output_path = _absolute_lexical_path(output)
    if output_path.exists() or output_path.is_symlink():
        raise ActualRecordError(f"output already exists: {output_path}")
    receipt_path = replay_receipt_path(replay_ledger_dir, record)
    receipt = _build_replay_receipt(record)
    try:
        write_json_once(receipt_path, receipt)
    except ActualRecordError as exc:
        if "already exists" in str(exc):
            raise ActualRecordError("acquisition challenge was already consumed; replay rejected") from exc
        raise
    try:
        write_json_once(output_path, record)
    except Exception:
        try:
            receipt_path.unlink()
            _fsync_directory(receipt_path.parent)
        except OSError:
            pass
        raise
    return record


def seal_record_to_path_once(
    *,
    output: Path | str,
    replay_ledger_dir: Path | str,
    acquisition_manifest: Path | str,
    raw_frame: Path | str,
    frame_bundle_artifact: Path | str,
    result_json: Path | str,
    capture_pipeline_artifact: Path | str,
    inference_pipeline_artifact: Path | str,
    camera_service_identity_artifact: Path | str,
    expected: ExpectedAcquisition,
    production_attestation: RunnerProductionAttestation | None = None,
) -> dict[str, Any]:
    """Seal once and consume the challenge using the host clock."""

    return _seal_record_to_path_once_with_clock(
        output=output,
        replay_ledger_dir=replay_ledger_dir,
        acquisition_manifest=acquisition_manifest,
        raw_frame=raw_frame,
        frame_bundle_artifact=frame_bundle_artifact,
        result_json=result_json,
        capture_pipeline_artifact=capture_pipeline_artifact,
        inference_pipeline_artifact=inference_pipeline_artifact,
        camera_service_identity_artifact=camera_service_identity_artifact,
        expected=expected,
        production_attestation=production_attestation,
        now_ms=None,
    )


def _add_expected_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-acquisition-id", required=True)
    parser.add_argument("--expected-a0-run-id", required=True)
    parser.add_argument("--expected-r2-run-id", required=True)
    parser.add_argument("--expected-r2-run-nonce", required=True)
    parser.add_argument("--expected-challenge-sha256", required=True)
    parser.add_argument("--expected-challenge-issued-at-ms", type=int, required=True)
    parser.add_argument("--expected-challenge-expires-at-ms", type=int, required=True)
    parser.add_argument("--expected-release-id", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-case-id", required=True)
    parser.add_argument("--expected-sample-id", required=True)
    parser.add_argument("--expected-sample-lineage-sha256", required=True)
    parser.add_argument("--expected-parent-evidence-root-sha256", required=True)
    parser.add_argument("--expected-task-kind", required=True, choices=sorted(TASK_RESULT_CONTRACTS))
    parser.add_argument("--expected-result-schema", required=True, choices=sorted(SUPPORTED_RESULT_SCHEMAS))
    parser.add_argument("--expected-success-state", required=True)
    parser.add_argument(
        "--expected-bag-empty-baseline-sha256",
        required=True,
        help="lowercase SHA-256 for bag acquisitions; literal NONE for station acquisitions",
    )
    parser.add_argument("--expected-capture-device-id", required=True)
    parser.add_argument("--expected-capture-boot-id", required=True)
    parser.add_argument("--expected-capture-session-id", required=True)
    parser.add_argument("--expected-inference-device-id", required=True)
    parser.add_argument("--expected-inference-boot-id", required=True)
    parser.add_argument("--expected-inference-session-id", required=True)


def _expected_from_args(args: argparse.Namespace) -> ExpectedAcquisition:
    return ExpectedAcquisition(
        acquisition_id=args.expected_acquisition_id,
        a0_run_id=args.expected_a0_run_id,
        r2_run_id=args.expected_r2_run_id,
        r2_run_nonce=args.expected_r2_run_nonce,
        challenge_sha256=args.expected_challenge_sha256,
        challenge_issued_at_ms=args.expected_challenge_issued_at_ms,
        challenge_expires_at_ms=args.expected_challenge_expires_at_ms,
        release_id=args.expected_release_id,
        config_sha256=args.expected_config_sha256,
        case_id=args.expected_case_id,
        sample_id=args.expected_sample_id,
        sample_lineage_sha256=args.expected_sample_lineage_sha256,
        parent_evidence_root_sha256=args.expected_parent_evidence_root_sha256,
        task_kind=args.expected_task_kind,
        result_schema=args.expected_result_schema,
        success_state=args.expected_success_state,
        bag_empty_baseline_sha256=(
            None
            if args.expected_bag_empty_baseline_sha256 == "NONE"
            else args.expected_bag_empty_baseline_sha256
        ),
        capture_device_id=args.expected_capture_device_id,
        capture_boot_id=args.expected_capture_boot_id,
        capture_session_id=args.expected_capture_session_id,
        inference_device_id=args.expected_inference_device_id,
        inference_boot_id=args.expected_inference_boot_id,
        inference_session_id=args.expected_inference_session_id,
    )


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--acquisition-manifest", type=Path, required=True)
    parser.add_argument("--raw-frame", type=Path, required=True)
    parser.add_argument("--frame-bundle-artifact", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--capture-pipeline-artifact", type=Path, required=True)
    parser.add_argument("--inference-pipeline-artifact", type=Path, required=True)
    parser.add_argument("--camera-service-identity-artifact", type=Path, required=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an already sealed, producer-attested A0 overhead record."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    _add_source_arguments(validate)
    _add_expected_arguments(validate)
    validate.add_argument("--record", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        source_kwargs = {
            "acquisition_manifest": args.acquisition_manifest,
            "raw_frame": args.raw_frame,
            "frame_bundle_artifact": args.frame_bundle_artifact,
            "result_json": args.result_json,
            "capture_pipeline_artifact": args.capture_pipeline_artifact,
            "inference_pipeline_artifact": args.inference_pipeline_artifact,
            "camera_service_identity_artifact": args.camera_service_identity_artifact,
            "expected": _expected_from_args(args),
        }
        record = load_record(args.record)
        validate_record(record, **source_kwargs)
        sys.stdout.write('{"status":"VALID"}\n')
        return 0
    except ActualRecordError as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
