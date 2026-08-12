"""Offline pre-board binding for a future native-v8 GGUF replay.

The legacy ``x5_gguf_replay`` runner validates the v5 student-answer schema and
accepts a caller-supplied model hash.  It is therefore not a valid runtime for
the strict v8 pointer model.  This module verifies a content-addressed v8
release and emits only a disabled launch plan for a later native-v8 runner.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from icmat_foundry.llm import gguf_release_v8

VERSION = "icmat-x5-gguf-replay-v8-preboard.0.0"
RELEASE_SCHEMA = "icmat_llm_gguf_release_receipt.v8"
RELEASE_VERSION = "icmat-gguf-release-v8.0.0"
RELEASE_STATUS = "PASS_PC_CPU_GGUF_RELEASE_NOT_ACTIVATED_X5_BOARD_PENDING"
PARITY_SCHEMA = "icmat_hf_gguf_pointer_parity_receipt.v8"
PARITY_STATUS = "PASS_NATIVE_V8_HF_GGUF_POINTER_AND_COMPILER_PARITY"
PLAN_SCHEMA = "icmat_x5_gguf_replay_plan.v8"
PLAN_STATUS = "PASS_V8_X5_REPLAY_INPUTS_BOUND_BOARD_EXECUTION_NOT_AUTHORIZED"


class X5GgufReplayV8Error(RuntimeError):
    """Raised when an offline v8 board replay input is not authoritative."""


@dataclass(frozen=True)
class ReplayPlanInputsV8:
    release_receipt: Path
    release_receipt_sha256: str
    gguf_model: Path
    llama_cli: Path
    llama_cli_sha256: str


def _exact(value: Any, keys: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise X5GgufReplayV8Error(f"{label} has an invalid exact field set")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _verify_digest(value: dict[str, Any], *, label: str) -> None:
    claimed = gguf_release_v8._require_sha256(
        value.get("canonical_digest_sha256"),
        label=f"{label}.canonical_digest_sha256",
    )
    body = dict(value)
    del body["canonical_digest_sha256"]
    if _digest(body) != claimed:
        raise X5GgufReplayV8Error(f"{label} canonical digest mismatch")


def prepare_replay_plan_v8(inputs: ReplayPlanInputsV8) -> dict[str, Any]:
    """Bind a future board replay without contacting or executing on an X5."""

    try:
        release_snapshot, release = gguf_release_v8._load_json(
            inputs.release_receipt,
            label="v8 GGUF release receipt",
            expected_sha256=inputs.release_receipt_sha256,
        )
    except gguf_release_v8.GgufReleaseV8Error as exc:
        raise X5GgufReplayV8Error("v8 GGUF release receipt rejected") from exc
    schema = release.get("schema")
    if (
        isinstance(schema, str)
        and schema in gguf_release_v8.LEGACY_SCHEMA_ROLES
    ):
        raise X5GgufReplayV8Error(
            "legacy GGUF release cannot authorize a native-v8 replay"
        )
    _exact(
        release,
        {
            "schema",
            "version",
            "status",
            "preflight",
            "chain_binding",
            "model",
            "parity",
            "runtime_policy",
            "canonical_digest_sha256",
        },
        label="v8 GGUF release receipt",
    )
    _verify_digest(release, label="v8 GGUF release receipt")
    if (
        release["schema"] != RELEASE_SCHEMA
        or release["version"] != RELEASE_VERSION
        or release["status"] != RELEASE_STATUS
    ):
        raise X5GgufReplayV8Error("GGUF release is not the final disabled v8 candidate")

    preflight = _exact(
        release["preflight"],
        {
            "schema",
            "status",
            "sha256",
            "authorization_digest_sha256",
        },
        label="v8 release preflight binding",
    )
    if (
        preflight["schema"] != gguf_release_v8.PREFLIGHT_SCHEMA
        or preflight["status"] != gguf_release_v8.PREFLIGHT_PASS_STATUS
    ):
        raise X5GgufReplayV8Error("v8 release does not bind a passed preflight")
    gguf_release_v8._require_sha256(
        preflight["sha256"],
        label="v8 release preflight SHA-256",
    )
    gguf_release_v8._require_sha256(
        preflight["authorization_digest_sha256"],
        label="v8 release authorization digest",
    )

    model = _exact(
        release["model"],
        {
            "filename",
            "bytes",
            "sha256",
            "format",
            "architecture",
            "quantization",
        },
        label="v8 release model",
    )
    if (
        model["format"] != "GGUF"
        or model["architecture"] != "qwen2"
        or model["quantization"] != "Q4_K_M"
        or not isinstance(model["filename"], str)
        or Path(model["filename"]).name != model["filename"]
        or not isinstance(model["bytes"], int)
        or model["bytes"] <= 0
    ):
        raise X5GgufReplayV8Error("v8 release model metadata is invalid")
    expected_model_sha = gguf_release_v8._require_sha256(
        model["sha256"],
        label="v8 release model SHA-256",
    )
    try:
        model_snapshot = gguf_release_v8._snapshot_binary_file(
            inputs.gguf_model,
            label="v8 GGUF model",
            expected_sha256=expected_model_sha,
            maximum_bytes=16 * 1024 * 1024 * 1024,
        )
        llama_snapshot = gguf_release_v8._snapshot_binary_file(
            inputs.llama_cli,
            label="native-v8 llama-cli",
            expected_sha256=inputs.llama_cli_sha256,
            maximum_bytes=1024 * 1024 * 1024,
        )
    except gguf_release_v8.GgufReleaseV8Error as exc:
        raise X5GgufReplayV8Error("v8 replay artifact binding failed") from exc
    if model_snapshot.bytes != model["bytes"]:
        raise X5GgufReplayV8Error("v8 GGUF model byte count differs from release")

    parity = _exact(
        release["parity"],
        {
            "schema",
            "status",
            "receipt_sha256",
            "strict_pointer_and_compiler_parity",
            "legacy_v5_comparator_used",
        },
        label="v8 release parity",
    )
    if (
        parity["schema"] != PARITY_SCHEMA
        or parity["status"] != PARITY_STATUS
        or parity["strict_pointer_and_compiler_parity"] is not True
        or parity["legacy_v5_comparator_used"] is not False
    ):
        raise X5GgufReplayV8Error("v8 release parity is absent or legacy")
    gguf_release_v8._require_sha256(
        parity["receipt_sha256"],
        label="v8 parity receipt SHA-256",
    )

    runtime = _exact(
        release["runtime_policy"],
        {
            "default_enabled",
            "autostart",
            "service_registered",
            "production_dependency",
            "production_files_modified",
            "x5_runtime_verified",
            "board_validation_pending",
        },
        label="v8 release runtime policy",
    )
    if runtime != {
        "default_enabled": False,
        "autostart": False,
        "service_registered": False,
        "production_dependency": False,
        "production_files_modified": False,
        "x5_runtime_verified": False,
        "board_validation_pending": True,
    }:
        raise X5GgufReplayV8Error("v8 release runtime policy is unsafe")

    plan = {
        "schema": PLAN_SCHEMA,
        "version": VERSION,
        "status": PLAN_STATUS,
        "release_receipt": {
            **release_snapshot.descriptor(),
            "expected_sha256": inputs.release_receipt_sha256,
        },
        "chain_binding": release["chain_binding"],
        "model": {
            **model_snapshot.descriptor(),
            "release_filename": model["filename"],
            "format": "GGUF",
            "architecture": "qwen2",
            "quantization": "Q4_K_M",
        },
        "llama_cli": {
            **llama_snapshot.descriptor(),
            "expected_sha256": inputs.llama_cli_sha256,
        },
        "parity": parity,
        "runtime_policy": {
            "native_pointer_v8_runner_required": True,
            "legacy_x5_gguf_replay_allowed": False,
            "legacy_reason": (
                "the frozen runner validates icmat_student_answer.v5 rather "
                "than the strict v8 task/decision/span_id pointer contract"
            ),
            "board_execution_authorized": False,
            "default_enabled": False,
            "autostart": False,
            "service_registration_authorized": False,
            "production_integration_authorized": False,
        },
        "x5_contacted": False,
        "process_started": False,
        "network_used": False,
        "claim_boundary": (
            "This plan binds files for a future native-v8 one-shot CPU replay. "
            "It does not contact X5, start llama-cli, authorize execution, "
            "register a service, or establish board performance."
        ),
    }
    plan["canonical_digest_sha256"] = _digest(plan)
    return plan


def write_replay_plan_v8(path: Path, plan: dict[str, Any]) -> Path:
    _verify_digest(plan, label="v8 X5 replay plan")
    output = Path(path).expanduser().absolute()
    parent = output.parent.resolve(strict=True)
    if os.path.lexists(output):
        raise X5GgufReplayV8Error("v8 replay plan output already exists")
    payload = (
        json.dumps(
            plan,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        parent / output.name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if os.path.lexists(output):
            os.unlink(output)
        raise
    return output.resolve(strict=True)


__all__ = [
    "PARITY_SCHEMA",
    "PARITY_STATUS",
    "PLAN_SCHEMA",
    "PLAN_STATUS",
    "RELEASE_SCHEMA",
    "RELEASE_STATUS",
    "RELEASE_VERSION",
    "ReplayPlanInputsV8",
    "VERSION",
    "X5GgufReplayV8Error",
    "prepare_replay_plan_v8",
    "write_replay_plan_v8",
]
