"""Native-v8 HF/GGUF pointer parity verification.

The verifier accepts only observations derived from the controlled v8 runtime
producer. It revalidates the producer receipt, raw process artifacts, executable
and source hashes, model trees, fixed target-free requests, and raw results
before scoring. It does not run a model itself, read a reserved blind split,
contact an X5, or use the legacy v5 seven-field answer comparator.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from icmat_foundry.llm import (
    evidence_pointer_v6,
    gguf_export_v5,
    gguf_release_v8,
    hf_gguf_observation_producer_v8,
    qlora_full_v6,
    selection_freeze_v8,
)

VERSION = "icmat-hf-gguf-pointer-parity-v8.1.0"
OBSERVATIONS_SCHEMA = "icmat_pointer_runtime_observations.v8"
OBSERVATIONS_STATUS = "COMPLETE_NONBLIND_V8_VALIDATION_OBSERVATIONS"
OBSERVATION_SCHEMA = "icmat_pointer_runtime_observation.v8"
PARITY_SCHEMA = "icmat_hf_gguf_pointer_parity_receipt.v8"
PARITY_PASS_STATUS = "PASS_NATIVE_V8_HF_GGUF_POINTER_AND_COMPILER_PARITY"
PARITY_FAIL_STATUS = "FAIL_NATIVE_V8_HF_GGUF_POINTER_OR_COMPILER_PARITY"
EXPECTED_ROWS = 150
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_POINTER_CHARS = 4096
AGREEMENT_FLOOR = 0.98

NON_DEGRADATION_LIMITS: dict[str, tuple[str, float]] = {
    "strict_expected_rate": ("drop", 0.02),
    "structure_valid_rate": ("drop", 0.02),
    "compiler_valid_rate": ("drop", 0.02),
    "expected_pointer_exact_rate": ("drop", 0.02),
    "compiler_expected_exact_rate": ("drop", 0.02),
    "unsupported_wrong_answer_rate": ("increase", 0.01),
}

_ROW_KEYS = {
    "schema",
    "dataset_schema",
    "example_id",
    "source_id",
    "family_id",
    "domain",
    "task",
    "decision",
    "target_span_id",
    "requested_claim",
    "doi",
    "license_id",
    "split",
    "messages",
    "metadata",
    "compiler_prompt",
    "compiler_evidence",
}
_PREFLIGHT_KEYS = {
    "schema",
    "version",
    "status",
    "read_only",
    "export_performed",
    "reserved_blind_dataset_read_by_this_preflight",
    "postfreeze_evidence_artifacts_hashed",
    "network_used",
    "x5_contacted",
    "chain_binding",
    "authority_digest_sha256",
    "authority_receipts",
    "postfreeze_artifacts",
    "tool_binding",
    "llama_server",
    "low_level_export_preflight",
    "required_followup_protocols",
    "authorization",
    "authorization_digest_sha256",
    "claim_boundary",
    "canonical_digest_sha256",
}
_EXPORT_KEYS = {
    "schema",
    "exporter_version",
    "created_at",
    "status",
    "run_id",
    "atomic_publish",
    "network_used",
    "x5_touched",
    "services_touched",
    "autostart_created",
    "input_snapshot",
    "source_inventory",
    "merge",
    "commands",
    "artifacts",
    "software",
    "wall_seconds",
    "claim_boundary",
}
_OBSERVATION_DOCUMENT_KEYS = {
    "schema",
    "version",
    "status",
    "backend",
    "preflight",
    "dataset",
    "generation_policy",
    "samples",
    "runtime_authority",
    "execution_boundary",
    "canonical_digest_sha256",
}
_AUTHORITY_REFERENCE_KEYS = {
    "schema",
    "status",
    "path",
    "bytes",
    "sha256",
    "canonical_digest_sha256",
    "provenance_kind",
    "execution_role",
    "raw_results_sha256",
}
_OBSERVATION_KEYS = {
    "schema",
    "example_id",
    "prompt_sha256",
    "expected_pointer_sha256",
    "raw_pointer",
    "finish_reason",
    "generation_error",
    "truncated",
    "latency_ms",
    "peak_rss_bytes",
}
_CHAIN_KEYS = {
    "selection_freeze_sha256",
    "selection_binding_digest_sha256",
    "manifest_sha256",
    "preblind_commitment_file_sha256",
    "preblind_commitment_sha256",
    "base_model_tree_sha256",
    "checkpoint_id",
    "checkpoint_tree_sha256",
    "adapter_tree_sha256",
}


class HfGgufParityV8Error(RuntimeError):
    """Raised when an integrity or authority input fails closed."""


@dataclass(frozen=True)
class ParityInputsV8:
    preflight_receipt: Path
    preflight_receipt_sha256: str
    dataset_dir: Path
    export_receipt: Path
    export_receipt_sha256: str
    gguf_model: Path
    gguf_model_sha256: str
    hf_observations: Path
    hf_observations_sha256: str
    gguf_observations: Path
    gguf_observations_sha256: str


@dataclass(frozen=True)
class ValidationRecordV8:
    example_id: str
    messages: tuple[Mapping[str, str], Mapping[str, str]]
    prompt: Mapping[str, Any]
    evidence: tuple[Mapping[str, Any], ...]
    prompt_sha256: str
    expected_pointer: Mapping[str, Any]
    expected_pointer_sha256: str
    expected_compilation: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _exact(value: Any, keys: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise HfGgufParityV8Error(
            f"{label} field set differs: expected {sorted(keys)}, got {actual}"
        )
    return value


def _string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise HfGgufParityV8Error(f"{label} must be a non-empty string")
    return value


def _sha(value: Any, *, label: str) -> str:
    try:
        return gguf_release_v8._require_sha256(value, label=label)
    except gguf_release_v8.GgufReleaseV8Error as exc:
        raise HfGgufParityV8Error(str(exc)) from exc


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        gguf_release_v8.canonical_json(value).encode("utf-8")
    ).hexdigest()


def _pointer_json(value: Mapping[str, Any]) -> str:
    ordered = {
        "task": value["task"],
        "decision": value["decision"],
        "span_id": value["span_id"],
    }
    return json.dumps(
        ordered,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _verify_canonical_digest(value: Mapping[str, Any], *, label: str) -> None:
    claimed = _sha(
        value.get("canonical_digest_sha256"),
        label=f"{label}.canonical_digest_sha256",
    )
    body = dict(value)
    del body["canonical_digest_sha256"]
    if _canonical_sha(body) != claimed:
        raise HfGgufParityV8Error(f"{label} canonical digest mismatch")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HfGgufParityV8Error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise HfGgufParityV8Error(f"non-finite JSON constant: {value}")


def _reserved_blind_path(path: Path) -> bool:
    for part in path.parts:
        token = part.casefold().replace("-", "_")
        if token == "blind" or token.startswith("blind_") or "reserved_blind" in token:
            return True
    return False


def _load_preflight(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[gguf_release_v8.FileSnapshotV8, Mapping[str, Any]]:
    try:
        snapshot, receipt = gguf_release_v8._load_json(
            path,
            label="native-v8 GGUF preflight",
            expected_sha256=expected_sha256,
        )
    except gguf_release_v8.GgufReleaseV8Error as exc:
        raise HfGgufParityV8Error("native-v8 GGUF preflight rejected") from exc
    _exact(receipt, _PREFLIGHT_KEYS, label="native-v8 GGUF preflight")
    _verify_canonical_digest(receipt, label="native-v8 GGUF preflight")
    if (
        receipt["schema"] != gguf_release_v8.PREFLIGHT_SCHEMA
        or receipt["status"] != gguf_release_v8.PREFLIGHT_PASS_STATUS
        or receipt["read_only"] is not True
        or receipt["export_performed"] is not False
        or receipt["reserved_blind_dataset_read_by_this_preflight"] is not False
        or receipt["network_used"] is not False
        or receipt["x5_contacted"] is not False
    ):
        raise HfGgufParityV8Error("native-v8 GGUF preflight is not authoritative")
    authorization = _exact(
        receipt["authorization"],
        {"gguf_export_authorized", *gguf_release_v8.FALSE_AUTHORIZATION},
        label="native-v8 preflight authorization",
    )
    if authorization != {
        "gguf_export_authorized": True,
        **gguf_release_v8.FALSE_AUTHORIZATION,
    }:
        raise HfGgufParityV8Error("native-v8 preflight authorization is unsafe")
    protocols = _exact(
        receipt["required_followup_protocols"],
        {
            "hf_gguf_parity",
            "x5_replay",
            "legacy_hf_gguf_parity_v5_allowed",
            "legacy_x5_gguf_replay_allowed",
        },
        label="native-v8 preflight follow-up protocols",
    )
    if protocols != {
        "hf_gguf_parity": gguf_release_v8.PARITY_PROTOCOL,
        "x5_replay": gguf_release_v8.REPLAY_PROTOCOL,
        "legacy_hf_gguf_parity_v5_allowed": False,
        "legacy_x5_gguf_replay_allowed": False,
    }:
        raise HfGgufParityV8Error("native-v8 preflight permits a legacy protocol")
    chain = _exact(
        receipt["chain_binding"],
        _CHAIN_KEYS,
        label="native-v8 preflight chain binding",
    )
    for key in _CHAIN_KEYS - {"checkpoint_id"}:
        _sha(chain[key], label=f"native-v8 chain binding {key}")
    _string(chain["checkpoint_id"], label="native-v8 checkpoint_id")
    _sha(
        receipt["authorization_digest_sha256"],
        label="native-v8 preflight authorization digest",
    )
    return snapshot, receipt


def _validation_records(
    dataset_dir: Path,
    *,
    preflight: Mapping[str, Any],
) -> tuple[list[ValidationRecordV8], dict[str, Any]]:
    raw_root = Path(dataset_dir).expanduser().absolute()
    if _reserved_blind_path(raw_root):
        raise HfGgufParityV8Error("dataset path is reserved-blind labelled")
    try:
        root = raw_root.resolve(strict=True)
    except OSError as exc:
        raise HfGgufParityV8Error("nonblind v8 dataset directory is unavailable") from exc
    if not root.is_dir() or root.is_symlink():
        raise HfGgufParityV8Error("nonblind v8 dataset must be a real directory")

    manifest_path = root / selection_freeze_v8.MANIFEST_NAME
    try:
        manifest_snapshot, manifest = gguf_release_v8._load_json(
            manifest_path,
            label="strict nonblind v8 manifest",
            expected_sha256=str(preflight["chain_binding"]["manifest_sha256"]),
        )
    except gguf_release_v8.GgufReleaseV8Error as exc:
        raise HfGgufParityV8Error("strict nonblind v8 manifest rejected") from exc
    if (
        manifest.get("schema") != selection_freeze_v8.MANIFEST_SCHEMA
        or manifest.get("dataset_schema") != selection_freeze_v8.DATASET_SCHEMA
    ):
        raise HfGgufParityV8Error("dataset is not the strict nonblind v8 dataset")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise HfGgufParityV8Error("strict nonblind v8 manifest has no split table")
    descriptor = _exact(
        splits.get("validation"),
        {"path", "count", "bytes", "sha256"},
        label="strict nonblind v8 validation descriptor",
    )
    if descriptor["count"] != EXPECTED_ROWS:
        raise HfGgufParityV8Error("strict nonblind v8 validation count is not 150")
    relative = Path(_string(descriptor["path"], label="validation path"))
    if relative.is_absolute() or ".." in relative.parts or _reserved_blind_path(relative):
        raise HfGgufParityV8Error("validation path escapes the nonblind dataset")
    split_path = (root / relative).absolute()
    try:
        split_snapshot = gguf_release_v8._snapshot_file(
            split_path,
            label="strict nonblind v8 validation split",
            expected_sha256=str(descriptor["sha256"]),
            maximum_bytes=MAX_JSON_BYTES,
        )
    except gguf_release_v8.GgufReleaseV8Error as exc:
        raise HfGgufParityV8Error("strict nonblind v8 validation split rejected") from exc
    if split_snapshot.path.parent != root or descriptor["bytes"] != split_snapshot.bytes:
        raise HfGgufParityV8Error("validation split path or byte count differs")

    lines = split_snapshot.payload.decode("utf-8").splitlines()
    if len(lines) != EXPECTED_ROWS or any(not line.strip() for line in lines):
        raise HfGgufParityV8Error("validation split must contain 150 non-empty rows")
    records: list[ValidationRecordV8] = []
    seen: set[str] = set()
    for index, line in enumerate(lines):
        try:
            row = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=_reject_nonfinite,
            )
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise HfGgufParityV8Error(
                f"validation row {index} is not strict JSON"
            ) from exc
        _exact(row, _ROW_KEYS, label=f"validation row {index}")
        if (
            row["schema"] != qlora_full_v6.EXAMPLE_SCHEMA
            or row["dataset_schema"] != selection_freeze_v8.DATASET_SCHEMA
            or row["split"] != "validation"
        ):
            raise HfGgufParityV8Error(f"validation row {index} contract mismatch")
        example_id = _string(row["example_id"], label=f"validation row {index} id")
        if example_id in seen:
            raise HfGgufParityV8Error(f"duplicate validation example_id: {example_id}")
        seen.add(example_id)
        task = _string(row["task"], label=f"{example_id}.task")
        decision = row["decision"]
        span_id = row["target_span_id"]
        if decision not in {"ANSWER", "REFUSE"}:
            raise HfGgufParityV8Error(f"{example_id}.decision is invalid")
        if (decision == "ANSWER" and (not isinstance(span_id, str) or not span_id)) or (
            decision == "REFUSE" and span_id is not None
        ):
            raise HfGgufParityV8Error(f"{example_id}.target_span_id is invalid")
        expected = {"task": task, "decision": decision, "span_id": span_id}
        messages = row["messages"]
        if (
            not isinstance(messages, Sequence)
            or isinstance(messages, (str, bytes))
            or len(messages) != 3
            or not isinstance(messages[2], Mapping)
            or messages[2].get("role") != "assistant"
            or messages[2].get("content") != _pointer_json(expected)
        ):
            raise HfGgufParityV8Error(
                f"{example_id} assistant target differs from explicit pointer fields"
            )
        target_free_messages: list[Mapping[str, str]] = []
        for message_index, expected_role in enumerate(("system", "user")):
            message = _exact(
                messages[message_index],
                {"role", "content"},
                label=f"{example_id}.messages[{message_index}]",
            )
            if (
                message["role"] != expected_role
                or not isinstance(message["content"], str)
                or not message["content"]
            ):
                raise HfGgufParityV8Error(
                    f"{example_id} target-free request messages are invalid"
                )
            target_free_messages.append(
                {"role": expected_role, "content": message["content"]}
            )
        prompt = row["compiler_prompt"]
        evidence = row["compiler_evidence"]
        if not isinstance(prompt, Mapping) or not isinstance(evidence, Sequence):
            raise HfGgufParityV8Error(f"{example_id} compiler inputs are invalid")
        if prompt.get("task") != task:
            raise HfGgufParityV8Error(f"{example_id} prompt task mismatch")
        compilation = evidence_pointer_v6.compile_pointer(
            prompt=prompt,
            evidence=evidence,
            raw_pointer=expected,
            finish_reason="eos_token",
        )
        if (
            compilation.get("status") != "COMPILED"
            or compilation.get("fail_closed") is not False
        ):
            raise HfGgufParityV8Error(
                f"{example_id} frozen expected pointer does not compile"
            )
        normalized_prompt = json.loads(gguf_release_v8.canonical_json(prompt))
        normalized_evidence = tuple(
            json.loads(gguf_release_v8.canonical_json(item)) for item in evidence
        )
        records.append(
            ValidationRecordV8(
                example_id=example_id,
                messages=(target_free_messages[0], target_free_messages[1]),
                prompt=normalized_prompt,
                evidence=normalized_evidence,
                prompt_sha256=_canonical_sha(normalized_prompt),
                expected_pointer=expected,
                expected_pointer_sha256=_canonical_sha(expected),
                expected_compilation=compilation,
            )
        )
    return records, {
        "manifest": manifest_snapshot.descriptor(),
        "split": "validation",
        "validation": split_snapshot.descriptor(),
        "samples": EXPECTED_ROWS,
        "example_id_order_sha256": _canonical_sha(
            [record.example_id for record in records]
        ),
    }


def _validate_low_level_export(
    *,
    receipt_path: Path,
    receipt_sha256: str,
    gguf_model: Path,
    gguf_model_sha256: str,
    preflight: Mapping[str, Any],
) -> tuple[
    gguf_release_v8.FileSnapshotV8,
    gguf_release_v8.BinaryFileSnapshotV8,
    Mapping[str, Any],
]:
    try:
        snapshot, receipt = gguf_release_v8._load_json(
            receipt_path,
            label="low-level GGUF export receipt",
            expected_sha256=receipt_sha256,
        )
    except gguf_release_v8.GgufReleaseV8Error as exc:
        raise HfGgufParityV8Error("low-level GGUF export receipt rejected") from exc
    _exact(receipt, _EXPORT_KEYS, label="low-level GGUF export receipt")
    if (
        receipt["schema"] != gguf_export_v5.EXPORT_RECEIPT_SCHEMA
        or receipt["exporter_version"] != gguf_export_v5.EXPORTER_VERSION
        or receipt["status"] != "PASS_GGUF_EXPORT_COMPLETED_NOT_DEPLOYED"
        or receipt["atomic_publish"] is not True
        or receipt["network_used"] is not False
        or receipt["x5_touched"] is not False
        or receipt["services_touched"] is not False
        or receipt["autostart_created"] is not False
    ):
        raise HfGgufParityV8Error("low-level export receipt is unsafe or incomplete")
    low_preflight = preflight["low_level_export_preflight"]
    exported_inputs = receipt["input_snapshot"]
    if not isinstance(exported_inputs, Mapping) or not isinstance(
        low_preflight, Mapping
    ):
        raise HfGgufParityV8Error("low-level export input snapshot is invalid")
    if (
        exported_inputs.get("input_fingerprint_sha256")
        != low_preflight.get("input_fingerprint_sha256")
        or exported_inputs.get("base_model", {}).get("tree_sha256")
        != low_preflight.get("base_model", {}).get("tree_sha256")
        or exported_inputs.get("adapter", {}).get("tree_sha256")
        != low_preflight.get("adapter", {}).get("tree_sha256")
        or exported_inputs.get("base_model_type") != "qwen2"
    ):
        raise HfGgufParityV8Error(
            "low-level export inputs differ from the native-v8 preflight"
        )
    for role in ("converter", "quantizer"):
        exported_tool = exported_inputs.get("tools", {}).get(role, {})
        authorized_tool = low_preflight.get("tools", {}).get(role, {})
        if (
            exported_tool.get("sha256") != authorized_tool.get("sha256")
            or exported_tool.get("runtime_tree", {}).get("tree_sha256")
            != authorized_tool.get("runtime_tree", {}).get("tree_sha256")
        ):
            raise HfGgufParityV8Error(f"low-level {role} binding changed")
    artifacts = receipt["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise HfGgufParityV8Error("low-level export artifact table is invalid")
    q4 = _exact(
        artifacts.get("gguf_q4_k_m"),
        {"path", "bytes", "sha256", "format", "quantization"},
        label="low-level Q4_K_M artifact",
    )
    expected_model_sha = _sha(
        gguf_model_sha256,
        label="expected Q4_K_M GGUF SHA-256",
    )
    if (
        q4["format"] != "GGUF"
        or q4["quantization"] != "Q4_K_M"
        or q4["sha256"] != expected_model_sha
        or not isinstance(q4["bytes"], int)
        or q4["bytes"] <= 0
        or q4["path"] != gguf_export_v5.DEFAULT_Q4_NAME
    ):
        raise HfGgufParityV8Error("low-level Q4_K_M artifact metadata is invalid")
    raw_model = Path(gguf_model).expanduser().absolute()
    if (
        raw_model.name != q4["path"]
        or raw_model.parent.resolve(strict=True) != snapshot.path.parent
    ):
        raise HfGgufParityV8Error(
            "Q4_K_M model is not co-located with its low-level export receipt"
        )
    try:
        model = gguf_release_v8._snapshot_binary_file(
            raw_model,
            label="Q4_K_M GGUF model",
            expected_sha256=expected_model_sha,
            maximum_bytes=16 * 1024 * 1024 * 1024,
        )
    except gguf_release_v8.GgufReleaseV8Error as exc:
        raise HfGgufParityV8Error("Q4_K_M GGUF model rejected") from exc
    if model.bytes != q4["bytes"]:
        raise HfGgufParityV8Error("Q4_K_M GGUF byte count differs from receipt")
    quantizer = receipt.get("commands", {}).get("quantizer", {})
    if not isinstance(quantizer, Mapping) or quantizer.get("returncode") != 0:
        raise HfGgufParityV8Error("low-level quantizer did not complete successfully")
    return snapshot, model, receipt


def _load_observation_envelope(
    path: Path,
    *,
    expected_sha256: str,
    kind: str,
) -> tuple[
    gguf_release_v8.FileSnapshotV8,
    Mapping[str, Any],
    Mapping[str, Any],
]:
    label = f"{kind} v8 observations"
    try:
        snapshot, document = gguf_release_v8._load_json(
            path,
            label=label,
            expected_sha256=expected_sha256,
        )
    except gguf_release_v8.GgufReleaseV8Error as exc:
        raise HfGgufParityV8Error(f"{label} rejected") from exc
    if not isinstance(document, Mapping) or "runtime_authority" not in document:
        raise HfGgufParityV8Error(
            f"{label} lacks controlled runtime authority; legacy caller-authored "
            "observations cannot authorize release"
        )
    _exact(document, _OBSERVATION_DOCUMENT_KEYS, label=label)
    _verify_canonical_digest(document, label=label)
    if (
        document["schema"] != OBSERVATIONS_SCHEMA
        or document["version"] != VERSION
        or document["status"] != OBSERVATIONS_STATUS
    ):
        raise HfGgufParityV8Error(f"{label} contract mismatch")
    authority = _exact(
        document["runtime_authority"],
        _AUTHORITY_REFERENCE_KEYS,
        label=f"{label} runtime authority",
    )
    if (
        authority["schema"]
        != hf_gguf_observation_producer_v8.AUTHORITY_SCHEMA
        or authority["status"]
        != hf_gguf_observation_producer_v8.AUTHORITY_STATUS
        or authority["provenance_kind"]
        != hf_gguf_observation_producer_v8.RUNTIME_PROVENANCE
        or authority["execution_role"] != kind
    ):
        raise HfGgufParityV8Error(
            f"{label} fixture or legacy authority cannot authorize release"
        )
    try:
        authority_path = Path(str(authority["path"])).resolve(strict=True)
    except OSError as exc:
        raise HfGgufParityV8Error(f"{label} authority path is unavailable") from exc
    if (
        not authority_path.is_file()
        or _reserved_blind_path(authority_path)
        or not isinstance(authority["bytes"], int)
        or isinstance(authority["bytes"], bool)
        or authority["bytes"] <= 0
    ):
        raise HfGgufParityV8Error(f"{label} authority reference is invalid")
    for key in ("sha256", "canonical_digest_sha256", "raw_results_sha256"):
        _sha(authority[key], label=f"{label} authority {key}")
    return snapshot, document, authority


def _load_observations(
    *,
    snapshot: gguf_release_v8.FileSnapshotV8,
    document: Mapping[str, Any],
    authority_reference: Mapping[str, Any],
    verified_authority: hf_gguf_observation_producer_v8.VerifiedRuntimeAuthorityV8,
    kind: str,
    preflight_snapshot: gguf_release_v8.FileSnapshotV8,
    preflight: Mapping[str, Any],
    dataset: Mapping[str, Any],
    model: gguf_release_v8.BinaryFileSnapshotV8,
) -> tuple[
    gguf_release_v8.FileSnapshotV8,
    Mapping[str, Any],
    Mapping[str, Mapping[str, Any]],
]:
    label = f"{kind} v8 observations"
    _exact(document, _OBSERVATION_DOCUMENT_KEYS, label=label)
    _verify_canonical_digest(document, label=label)
    if (
        document["schema"] != OBSERVATIONS_SCHEMA
        or document["version"] != VERSION
        or document["status"] != OBSERVATIONS_STATUS
    ):
        raise HfGgufParityV8Error(f"{label} contract mismatch")
    expected_authority_reference = {
        "schema": hf_gguf_observation_producer_v8.AUTHORITY_SCHEMA,
        "status": hf_gguf_observation_producer_v8.AUTHORITY_STATUS,
        **verified_authority.snapshot.descriptor(),
        "canonical_digest_sha256": verified_authority.receipt[
            "canonical_digest_sha256"
        ],
        "provenance_kind": hf_gguf_observation_producer_v8.RUNTIME_PROVENANCE,
        "execution_role": kind,
        "raw_results_sha256": verified_authority.receipt["executions"][
            "hf_selected_adapter" if kind == "HF_SELECTED_ADAPTER" else "gguf_q4_k_m"
        ]["raw_results"]["sha256"],
    }
    if authority_reference != expected_authority_reference:
        raise HfGgufParityV8Error(
            f"{label} does not match the revalidated runtime authority"
        )
    preflight_binding = _exact(
        document["preflight"],
        {"sha256", "authorization_digest_sha256"},
        label=f"{label} preflight binding",
    )
    if preflight_binding != {
        "sha256": preflight_snapshot.sha256,
        "authorization_digest_sha256": preflight["authorization_digest_sha256"],
    }:
        raise HfGgufParityV8Error(f"{label} uses a different preflight")
    dataset_binding = _exact(
        document["dataset"],
        {
            "manifest_sha256",
            "split",
            "split_sha256",
            "samples",
            "example_id_order_sha256",
        },
        label=f"{label} dataset binding",
    )
    expected_dataset = {
        "manifest_sha256": dataset["manifest"]["sha256"],
        "split": "validation",
        "split_sha256": dataset["validation"]["sha256"],
        "samples": EXPECTED_ROWS,
        "example_id_order_sha256": dataset["example_id_order_sha256"],
    }
    if dataset_binding != expected_dataset:
        raise HfGgufParityV8Error(f"{label} dataset binding mismatch")
    backend = _exact(
        document["backend"],
        {
            "kind",
            "engine",
            "engine_version",
            "device",
            "model",
            "runtime_artifact_sha256",
        },
        label=f"{label} backend",
    )
    if (
        backend["kind"] != kind
        or backend["device"] != "LOCAL_PC_CPU"
        or not isinstance(backend["engine_version"], str)
        or not backend["engine_version"]
    ):
        raise HfGgufParityV8Error(f"{label} backend identity mismatch")
    chain = preflight["chain_binding"]
    if kind == "HF_SELECTED_ADAPTER":
        hf_model = _exact(
            backend["model"],
            {
                "base_model_tree_sha256",
                "checkpoint_tree_sha256",
                "adapter_tree_sha256",
            },
            label=f"{label} selected model",
        )
        if hf_model != {
            "base_model_tree_sha256": chain["base_model_tree_sha256"],
            "checkpoint_tree_sha256": chain["checkpoint_tree_sha256"],
            "adapter_tree_sha256": chain["adapter_tree_sha256"],
        }:
            raise HfGgufParityV8Error(f"{label} selected model binding mismatch")
        if (
            backend["engine"] != "transformers_peft"
            or backend["runtime_artifact_sha256"] is not None
        ):
            raise HfGgufParityV8Error(f"{label} must use transformers_peft")
    elif kind == "GGUF_Q4_K_M":
        gguf_model = _exact(
            backend["model"],
            {"filename", "bytes", "sha256", "format", "architecture", "quantization"},
            label=f"{label} GGUF model",
        )
        if gguf_model != {
            "filename": model.path.name,
            "bytes": model.bytes,
            "sha256": model.sha256,
            "format": "GGUF",
            "architecture": "qwen2",
            "quantization": "Q4_K_M",
        }:
            raise HfGgufParityV8Error(f"{label} GGUF model binding mismatch")
        if (
            backend["engine"] != "llama.cpp-server"
            or backend["runtime_artifact_sha256"]
            != preflight["tool_binding"]["llama_server_sha256"]
        ):
            raise HfGgufParityV8Error(f"{label} must use the pinned llama-server")
    else:
        raise HfGgufParityV8Error(f"unsupported observation backend: {kind}")
    policy = _exact(
        document["generation_policy"],
        {
            "do_sample",
            "temperature",
            "max_new_tokens",
            "seed",
            "stop_on_eos",
            "batch_size",
            "device",
        },
        label=f"{label} generation policy",
    )
    if policy != hf_gguf_observation_producer_v8.generation_policy_v8():
        raise HfGgufParityV8Error(f"{label} generation policy is not deterministic")
    boundary = _exact(
        document["execution_boundary"],
        {
            "model_invoked",
            "network_used",
            "reserved_blind_read",
            "x5_contacted",
            "production_services_touched",
        },
        label=f"{label} execution boundary",
    )
    if boundary != {
        "model_invoked": True,
        "network_used": False,
        "reserved_blind_read": False,
        "x5_contacted": False,
        "production_services_touched": False,
    }:
        raise HfGgufParityV8Error(f"{label} crossed the offline validation boundary")
    samples = document["samples"]
    if not isinstance(samples, list) or len(samples) != EXPECTED_ROWS:
        raise HfGgufParityV8Error(f"{label} must contain 150 samples")
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, sample in enumerate(samples):
        _exact(sample, _OBSERVATION_KEYS, label=f"{label} sample {index}")
        if sample["schema"] != OBSERVATION_SCHEMA:
            raise HfGgufParityV8Error(f"{label} sample {index} schema mismatch")
        example_id = _string(
            sample["example_id"],
            label=f"{label} sample {index} example_id",
        )
        if example_id in by_id:
            raise HfGgufParityV8Error(f"{label} duplicates {example_id}")
        raw_pointer = sample["raw_pointer"]
        if not isinstance(raw_pointer, str) or len(raw_pointer) > MAX_POINTER_CHARS:
            raise HfGgufParityV8Error(f"{label} sample {index} pointer is invalid")
        finish_reason = sample["finish_reason"]
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise HfGgufParityV8Error(
                f"{label} sample {index} finish_reason is invalid"
            )
        error = sample["generation_error"]
        if error is not None and not isinstance(error, str):
            raise HfGgufParityV8Error(
                f"{label} sample {index} generation_error is invalid"
            )
        if not isinstance(sample["truncated"], bool):
            raise HfGgufParityV8Error(f"{label} sample {index} truncated is invalid")
        latency = sample["latency_ms"]
        if (
            not isinstance(latency, (int, float))
            or isinstance(latency, bool)
            or not math.isfinite(float(latency))
            or latency < 0
        ):
            raise HfGgufParityV8Error(f"{label} sample {index} latency is invalid")
        rss = sample["peak_rss_bytes"]
        if not isinstance(rss, int) or isinstance(rss, bool) or rss < 0:
            raise HfGgufParityV8Error(f"{label} sample {index} RSS is invalid")
        _sha(sample["prompt_sha256"], label=f"{label} sample {index} prompt hash")
        _sha(
            sample["expected_pointer_sha256"],
            label=f"{label} sample {index} expected pointer hash",
        )
        by_id[example_id] = sample
    raw_samples = verified_authority.results.get(kind)
    if not isinstance(raw_samples, Mapping) or list(raw_samples) != list(by_id):
        raise HfGgufParityV8Error(f"{label} raw authority membership differs")
    for example_id, sample in by_id.items():
        raw = raw_samples[example_id]
        expected_runtime_fields = {
            "raw_pointer": raw["raw_pointer"],
            "finish_reason": raw["finish_reason"],
            "generation_error": raw["generation_error"],
            "truncated": raw["finish_category"] == "LENGTH_LIMIT",
            "latency_ms": raw["latency_ms"],
            "peak_rss_bytes": raw["peak_rss_bytes"],
        }
        if any(sample[key] != value for key, value in expected_runtime_fields.items()):
            raise HfGgufParityV8Error(
                f"{label} sample {example_id} differs from raw runtime authority"
            )
    return snapshot, document, by_id


def _parse_pointer(value: str) -> tuple[Mapping[str, Any] | None, str | None]:
    try:
        parsed = json.loads(
            value,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (json.JSONDecodeError, HfGgufParityV8Error) as exc:
        return None, str(exc)
    if not isinstance(parsed, Mapping):
        return None, "pointer must be one JSON object"
    if list(parsed) != ["task", "decision", "span_id"]:
        return None, "pointer keys or literal order differ"
    task = parsed.get("task")
    decision = parsed.get("decision")
    span_id = parsed.get("span_id")
    if not isinstance(task, str) or not task:
        return None, "pointer.task must be non-empty"
    if decision not in {"ANSWER", "REFUSE"}:
        return None, "pointer.decision must be ANSWER or REFUSE"
    if decision == "ANSWER" and (not isinstance(span_id, str) or not span_id):
        return None, "ANSWER pointer requires a span_id"
    if decision == "REFUSE" and span_id is not None:
        return None, "REFUSE pointer requires null span_id"
    return dict(parsed), None


def _score_backend(
    records: Sequence[ValidationRecordV8],
    observations: Mapping[str, Mapping[str, Any]],
    *,
    label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_order = [record.example_id for record in records]
    if list(observations) != expected_order:
        raise HfGgufParityV8Error(
            f"{label} observation order differs from frozen validation order"
        )
    for record in records:
        observed = observations[record.example_id]
        if observed["prompt_sha256"] != record.prompt_sha256:
            raise HfGgufParityV8Error(
                f"{label} prompt hash mismatch for {record.example_id}"
            )
        if observed["expected_pointer_sha256"] != record.expected_pointer_sha256:
            raise HfGgufParityV8Error(
                f"{label} expected pointer hash mismatch for {record.example_id}"
            )
        parsed, structure_error = _parse_pointer(observed["raw_pointer"])
        compilation = evidence_pointer_v6.compile_pointer(
            prompt=record.prompt,
            evidence=record.evidence,
            raw_pointer=observed["raw_pointer"],
            finish_reason=observed["finish_reason"],
        )
        compiler_valid = (
            compilation.get("status") == "COMPILED"
            and compilation.get("fail_closed") is False
        )
        expected_exact = parsed == record.expected_pointer
        compiler_expected = (
            compiler_valid
            and compilation.get("compiled_answer")
            == record.expected_compilation.get("compiled_answer")
        )
        trusted_finish = (
            observed["finish_reason"]
            in evidence_pointer_v6.TRUSTED_FINISH_REASONS
        )
        no_trace_error = (
            observed["generation_error"] is None
            and observed["truncated"] is False
        )
        strict_expected = all(
            (
                structure_error is None,
                compiler_valid,
                expected_exact,
                compiler_expected,
                trusted_finish,
                no_trace_error,
            )
        )
        unsupported_wrong = (
            record.expected_pointer["decision"] == "REFUSE"
            and parsed is not None
            and parsed.get("decision") == "ANSWER"
        )
        rows.append(
            {
                "example_id": record.example_id,
                "parsed_pointer": parsed,
                "structure_error": structure_error,
                "compilation": compilation,
                "structure_valid": structure_error is None,
                "compiler_valid": compiler_valid,
                "expected_pointer_exact": expected_exact,
                "compiler_expected_exact": compiler_expected,
                "trusted_finish": trusted_finish,
                "no_trace_error": no_trace_error,
                "strict_expected": strict_expected,
                "unsupported_wrong_answer": unsupported_wrong,
                "raw_pointer_sha256": hashlib.sha256(
                    observed["raw_pointer"].encode("utf-8")
                ).hexdigest(),
            }
        )
    count = len(rows)

    def rate(field: str) -> float:
        return sum(bool(row[field]) for row in rows) / count

    metrics = {
        "samples": count,
        "structure_valid_rate": rate("structure_valid"),
        "compiler_valid_rate": rate("compiler_valid"),
        "expected_pointer_exact_rate": rate("expected_pointer_exact"),
        "compiler_expected_exact_rate": rate("compiler_expected_exact"),
        "strict_expected_rate": rate("strict_expected"),
        "unsupported_wrong_answer_rate": rate("unsupported_wrong_answer"),
        "trusted_finish_rate": rate("trusted_finish"),
        "no_trace_error_rate": rate("no_trace_error"),
        "latency_ms_mean": sum(
            float(observations[row["example_id"]]["latency_ms"]) for row in rows
        )
        / count,
        "latency_ms_max": max(
            float(observations[row["example_id"]]["latency_ms"]) for row in rows
        ),
        "peak_rss_bytes": max(
            int(observations[row["example_id"]]["peak_rss_bytes"]) for row in rows
        ),
    }
    return rows, metrics


def _gates(
    hf_rows: Sequence[Mapping[str, Any]],
    gguf_rows: Sequence[Mapping[str, Any]],
    *,
    hf_metrics: Mapping[str, Any],
    gguf_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    non_degradation: dict[str, Any] = {}
    for metric, (direction, tolerance) in NON_DEGRADATION_LIMITS.items():
        hf_value = float(hf_metrics[metric])
        gguf_value = float(gguf_metrics[metric])
        degradation = (
            hf_value - gguf_value
            if direction == "drop"
            else gguf_value - hf_value
        )
        non_degradation[metric] = {
            "direction": direction,
            "tolerance": tolerance,
            "observed_degradation": degradation,
            "passed": degradation <= tolerance + 1e-12,
        }
    pointer_exact = sum(
        left["structure_valid"]
        and right["structure_valid"]
        and left["parsed_pointer"] == right["parsed_pointer"]
        for left, right in zip(hf_rows, gguf_rows, strict=True)
    ) / len(hf_rows)
    compiler_exact = sum(
        left["compiler_valid"]
        and right["compiler_valid"]
        and left["compilation"].get("compiler_decision")
        == right["compilation"].get("compiler_decision")
        and left["compilation"].get("selected_span_id")
        == right["compilation"].get("selected_span_id")
        and left["compilation"].get("compiled_answer")
        == right["compilation"].get("compiled_answer")
        for left, right in zip(hf_rows, gguf_rows, strict=True)
    ) / len(hf_rows)
    safety = {
        "hf_no_generation_error_or_truncation": (
            hf_metrics["no_trace_error_rate"] == 1.0
        ),
        "gguf_no_generation_error_or_truncation": (
            gguf_metrics["no_trace_error_rate"] == 1.0
        ),
        "hf_trusted_finish": hf_metrics["trusted_finish_rate"] == 1.0,
        "gguf_trusted_finish": gguf_metrics["trusted_finish_rate"] == 1.0,
    }
    agreement = {
        "task_decision_span_backend_exact_rate": pointer_exact,
        "compiler_output_backend_exact_rate": compiler_exact,
        "required_floor": AGREEMENT_FLOOR,
        "task_decision_span_backend_exact_passed": (
            pointer_exact + 1e-12 >= AGREEMENT_FLOOR
        ),
        "compiler_output_backend_exact_passed": (
            compiler_exact + 1e-12 >= AGREEMENT_FLOOR
        ),
    }
    all_passed = (
        all(check["passed"] for check in non_degradation.values())
        and all(safety.values())
        and agreement["task_decision_span_backend_exact_passed"]
        and agreement["compiler_output_backend_exact_passed"]
    )
    return {
        "non_degradation": non_degradation,
        "safety": safety,
        "agreement": agreement,
        "all_passed": all_passed,
    }


def verify_hf_gguf_parity_v8(inputs: ParityInputsV8) -> dict[str, Any]:
    """Recompute native-v8 nonblind validation parity from bound observations."""

    preflight_snapshot, preflight = _load_preflight(
        inputs.preflight_receipt,
        expected_sha256=inputs.preflight_receipt_sha256,
    )
    records, dataset = _validation_records(
        inputs.dataset_dir,
        preflight=preflight,
    )
    export_snapshot, model_snapshot, _ = _validate_low_level_export(
        receipt_path=inputs.export_receipt,
        receipt_sha256=inputs.export_receipt_sha256,
        gguf_model=inputs.gguf_model,
        gguf_model_sha256=inputs.gguf_model_sha256,
        preflight=preflight,
    )
    hf_snapshot, hf_document, hf_authority_reference = _load_observation_envelope(
        inputs.hf_observations,
        expected_sha256=inputs.hf_observations_sha256,
        kind="HF_SELECTED_ADAPTER",
    )
    gguf_snapshot, gguf_document, gguf_authority_reference = (
        _load_observation_envelope(
            inputs.gguf_observations,
            expected_sha256=inputs.gguf_observations_sha256,
            kind="GGUF_Q4_K_M",
        )
    )
    shared_authority_keys = {
        "schema",
        "status",
        "path",
        "bytes",
        "sha256",
        "canonical_digest_sha256",
        "provenance_kind",
    }
    if any(
        hf_authority_reference[key] != gguf_authority_reference[key]
        for key in shared_authority_keys
    ):
        raise HfGgufParityV8Error(
            "HF and GGUF observations do not share one runtime authority"
        )
    try:
        verified_authority = (
            hf_gguf_observation_producer_v8.verify_runtime_authority_v8(
                authority_path=Path(str(hf_authority_reference["path"])),
                authority_sha256=str(hf_authority_reference["sha256"]),
                preflight_snapshot=preflight_snapshot,
                preflight=preflight,
                export_snapshot=export_snapshot,
                gguf_model=model_snapshot,
                expected_request_payload=(
                    hf_gguf_observation_producer_v8.request_payload_v8(records)
                ),
            )
        )
    except hf_gguf_observation_producer_v8.ObservationProducerV8Error as exc:
        raise HfGgufParityV8Error(
            "controlled runtime observation authority rejected"
        ) from exc
    hf_snapshot, hf_document, hf_observations = _load_observations(
        snapshot=hf_snapshot,
        document=hf_document,
        authority_reference=hf_authority_reference,
        verified_authority=verified_authority,
        kind="HF_SELECTED_ADAPTER",
        preflight_snapshot=preflight_snapshot,
        preflight=preflight,
        dataset=dataset,
        model=model_snapshot,
    )
    gguf_snapshot, gguf_document, gguf_observations = _load_observations(
        snapshot=gguf_snapshot,
        document=gguf_document,
        authority_reference=gguf_authority_reference,
        verified_authority=verified_authority,
        kind="GGUF_Q4_K_M",
        preflight_snapshot=preflight_snapshot,
        preflight=preflight,
        dataset=dataset,
        model=model_snapshot,
    )
    if hf_document["generation_policy"] != gguf_document["generation_policy"]:
        raise HfGgufParityV8Error("HF and GGUF generation policies differ")
    hf_rows, hf_metrics = _score_backend(
        records,
        hf_observations,
        label="HF",
    )
    gguf_rows, gguf_metrics = _score_backend(
        records,
        gguf_observations,
        label="GGUF",
    )
    gates = _gates(
        hf_rows,
        gguf_rows,
        hf_metrics=hf_metrics,
        gguf_metrics=gguf_metrics,
    )
    passed = bool(gates["all_passed"])
    failing_ids = [
        record.example_id
        for record, hf_row, gguf_row in zip(
            records,
            hf_rows,
            gguf_rows,
            strict=True,
        )
        if not (
            hf_row["strict_expected"]
            and gguf_row["strict_expected"]
            and hf_row["parsed_pointer"] == gguf_row["parsed_pointer"]
            and hf_row["compilation"].get("compiled_answer")
            == gguf_row["compilation"].get("compiled_answer")
        )
    ]
    authority_receipt = verified_authority.receipt
    authority_executions = authority_receipt["executions"]
    authority_implementation = authority_receipt["implementation"]
    observation_authority = {
        "schema": hf_gguf_observation_producer_v8.AUTHORITY_SCHEMA,
        "version": hf_gguf_observation_producer_v8.VERSION,
        "status": hf_gguf_observation_producer_v8.AUTHORITY_STATUS,
        **verified_authority.snapshot.descriptor(),
        "canonical_digest_sha256": authority_receipt["canonical_digest_sha256"],
        "provenance_kind": hf_gguf_observation_producer_v8.RUNTIME_PROVENANCE,
        "request_set_sha256": authority_receipt["request_set"]["sha256"],
        "release_authority_digest_sha256": authority_receipt["release_authority"][
            "authority_digest_sha256"
        ],
        "producer_source_sha256": authority_implementation["producer_module"][
            "sha256"
        ],
        "producer_cli_sha256": authority_implementation["producer_cli"]["sha256"],
        "hf_runner_source_sha256": authority_implementation["hf_runner_source"][
            "sha256"
        ],
        "llama_runner_source_sha256": authority_implementation[
            "llama_runner_source"
        ]["sha256"],
        "hf_program_sha256": authority_executions["hf_selected_adapter"][
            "program"
        ]["sha256"],
        "hf_stdout_sha256": authority_executions["hf_selected_adapter"]["stdout"][
            "sha256"
        ],
        "hf_stderr_sha256": authority_executions["hf_selected_adapter"]["stderr"][
            "sha256"
        ],
        "hf_exit_status": authority_executions["hf_selected_adapter"]["returncode"],
        "gguf_program_sha256": authority_executions["gguf_q4_k_m"]["program"][
            "sha256"
        ],
        "gguf_stdout_sha256": authority_executions["gguf_q4_k_m"]["stdout"][
            "sha256"
        ],
        "gguf_stderr_sha256": authority_executions["gguf_q4_k_m"]["stderr"][
            "sha256"
        ],
        "gguf_exit_status": authority_executions["gguf_q4_k_m"]["returncode"],
        "raw_authority_inputs_revalidated": True,
        "raw_process_artifacts_revalidated": True,
        "fixture_observations_used": False,
    }
    receipt = {
        "schema": PARITY_SCHEMA,
        "version": VERSION,
        "created_at_utc": _utc_now(),
        "status": PARITY_PASS_STATUS if passed else PARITY_FAIL_STATUS,
        "preflight": {
            **preflight_snapshot.descriptor(),
            "authorization_digest_sha256": preflight[
                "authorization_digest_sha256"
            ],
        },
        "chain_binding": dict(preflight["chain_binding"]),
        "low_level_export": export_snapshot.descriptor(),
        "model": {
            **model_snapshot.descriptor(),
            "filename": model_snapshot.path.name,
            "format": "GGUF",
            "architecture": "qwen2",
            "quantization": "Q4_K_M",
        },
        "dataset": dataset,
        "observations": {
            "hf_selected_adapter": hf_snapshot.descriptor(),
            "gguf_q4_k_m": gguf_snapshot.descriptor(),
        },
        "observation_authority": observation_authority,
        "metrics": {
            "hf_selected_adapter": hf_metrics,
            "gguf_q4_k_m": gguf_metrics,
        },
        "gates": gates,
        "failing_example_ids": failing_ids,
        "integrity": {
            "complete_validation_membership_recomputed": True,
            "prompt_hashes_recomputed": True,
            "expected_pointers_derived_from_frozen_rows": True,
            "compiler_outputs_recomputed": True,
            "self_reported_scores_ignored": True,
            "controlled_runtime_authority_revalidated": True,
            "raw_stdout_stderr_and_exit_status_revalidated": True,
            "producer_runner_and_program_hashes_revalidated": True,
            "base_adapter_and_gguf_bindings_revalidated": True,
            "target_free_request_set_recomputed": True,
            "fixture_observations_used": False,
            "legacy_v5_comparator_used": False,
            "compiler": {
                "version": evidence_pointer_v6.COMPILER_VERSION,
                "sha256": gguf_release_v8.sha256_file(
                    Path(evidence_pointer_v6.__file__).resolve()
                ),
            },
        },
        "strict_pointer_and_compiler_parity": passed,
        "legacy_v5_comparator_used": False,
        "authorization": {
            "pc_gguf_release_receipt_authorized": passed,
            "x5_execution_authorized": False,
            "deployment_authorized": False,
            "production_integration_authorized": False,
        },
        "execution_boundary": {
            "model_invoked_by_this_validator": False,
            "observation_files_consumed": True,
            "reserved_blind_read": False,
            "network_used": False,
            "x5_contacted": False,
            "production_services_touched": False,
        },
        "claim_boundary": (
            "A PASS establishes task-level non-degradation and at least 98% "
            "task/decision/span_id plus compiled-answer agreement on the "
            "complete frozen nonblind v8 validation split. It is not token-"
            "level equivalence, reserved-blind evidence, X5 performance, "
            "deployment authorization, or a production claim."
        ),
    }
    receipt["canonical_digest_sha256"] = _canonical_sha(receipt)
    return receipt


def write_parity_receipt_v8(path: Path, receipt: Mapping[str, Any]) -> Path:
    """Write one immutable v8 parity receipt."""

    _verify_canonical_digest(receipt, label="native-v8 parity receipt")
    output = Path(path).expanduser().absolute()
    try:
        parent = output.parent.resolve(strict=True)
    except OSError as exc:
        raise HfGgufParityV8Error("parity output parent must exist") from exc
    if output.name in {"", ".", ".."} or os.path.lexists(output):
        raise HfGgufParityV8Error("parity output must be a new file")
    payload = (
        json.dumps(
            receipt,
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
    "EXPECTED_ROWS",
    "HfGgufParityV8Error",
    "OBSERVATIONS_SCHEMA",
    "OBSERVATIONS_STATUS",
    "OBSERVATION_SCHEMA",
    "PARITY_FAIL_STATUS",
    "PARITY_PASS_STATUS",
    "PARITY_SCHEMA",
    "ParityInputsV8",
    "VERSION",
    "verify_hf_gguf_parity_v8",
    "write_parity_receipt_v8",
]
