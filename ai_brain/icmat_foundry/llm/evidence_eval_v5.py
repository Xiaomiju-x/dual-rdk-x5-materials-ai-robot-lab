"""Free-generation evaluation for the ICMat v5 evidence-bounded student.

The evaluator deliberately separates generation from scoring. Model backends
receive only the system and user messages; the assistant target is retained
only inside the scoring process. Deterministic and fixture backends exist for
evaluator tests and are always labelled as non-model evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import sys
import time
import traceback
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .selection_freeze_v5 import (
    SelectionFreezeV5Error,
    verify_selection_freeze,
)

MANIFEST_SCHEMA = "icmat_evidence_sft_manifest.v5"
EXAMPLE_SCHEMA = "icmat_student_sft_example.v5"
ANSWER_SCHEMA = "icmat_student_answer.v5"
AUTHORIZATION_SCHEMA = "icmat_blind_test_evaluation_authorization.v5"
AUTHORIZATION_VERSION = "icmat-blind-test-authorization-v5.2.0"
CONSUMPTION_SCHEMA = "icmat_blind_test_consumption.v5"
BLIND_AUTHORIZATION_SCOPE = "icmat_v5_blind_hf_evaluation_once"
SAMPLE_RESULT_SCHEMA = "icmat_evidence_eval_sample.v5"
SUMMARY_SCHEMA = "icmat_evidence_eval_summary.v5"
RUN_RECEIPT_SCHEMA = "icmat_evidence_eval_run_receipt.v5"

ALLOWED_SPLITS = ("validation", "calibration", "blind_test")
ALLOWED_DECISIONS = ("ANSWER", "REFUSE")
ALLOWED_VERDICTS = ("SUPPORTED", "REFUSED")
ALLOWED_ABLATIONS = ("none", "evidence_removed", "evidence_swapped", "no_rag")
ALLOWED_BACKENDS = ("hf_model", "deterministic_baseline", "fixture")
TARGET_KEYS = {
    "schema",
    "decision",
    "task",
    "claim",
    "verdict",
    "evidence_ids",
    "provenance",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EVIDENCE_SECTION_RE = re.compile(
    r"\[(?P<tag>EVIDENCE|RAG_EVIDENCE|EVIDENCE_BUNDLE|"
    r"RAG_CONTEXT|SOURCE_CONTEXT)(?:[^\]]*)\]"
    r".*?\[/(?P=tag)\]",
    flags=re.IGNORECASE | re.DOTALL,
)
MAX_AUTHORIZATION_BYTES = 1024 * 1024
EXPECTED_BLIND_EXAMPLES = 150
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{7,127}$")
OUTPUT_BASENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")


class EvidenceEvalV5Error(ValueError):
    """Raised when the v5 dataset or evaluation contract is invalid."""


class BlindTestAuthorizationError(PermissionError):
    """Raised when blind-test labels are requested without a valid receipt."""


@dataclass(frozen=True)
class BlindConsumptionV5:
    """An exclusive blind-test claim created before the JSONL is opened."""

    authorization: dict[str, Any]
    authorization_path: Path
    authorization_sha256: str
    marker_path: Path
    pending_receipt: dict[str, Any]


@dataclass(frozen=True)
class EvidenceSampleV5:
    """One validated example with its target separated from model inputs."""

    example_id: str
    split: str
    domain: str
    task: str
    decision: str
    model_messages: tuple[dict[str, str], dict[str, str]]
    expected: dict[str, Any]
    raw: dict[str, Any]


@dataclass(frozen=True)
class DatasetSelectionV5:
    """Integrity-bound selection from one v5 evaluation split."""

    dataset_dir: Path
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    split: str
    split_path: Path
    split_sha256: str
    samples: tuple[EvidenceSampleV5, ...]
    blind_test_authorization: dict[str, Any] | None


@dataclass(frozen=True)
class GenerationRequestV5:
    """A model-visible prompt after an optional deterministic ablation."""

    example_id: str
    split: str
    domain: str
    task: str
    ablation: str
    messages: tuple[dict[str, str], dict[str, str]]
    prompt_sha256: str


def canonical_json(value: Any) -> str:
    """Return the canonical JSON representation used for identity hashes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _jsonl_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join((canonical_json(dict(record)) + "\n").encode("utf-8") for record in records)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceEvalV5Error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_single_json_object(text: str) -> dict[str, Any]:
    """Parse exactly one JSON object, rejecting prose, fences, and duplicates."""

    if not isinstance(text, str) or not text.strip():
        raise EvidenceEvalV5Error("generation is empty or not a string")
    decoder = json.JSONDecoder(object_pairs_hook=_reject_duplicate_pairs)
    start = len(text) - len(text.lstrip())
    try:
        value, end = decoder.raw_decode(text, idx=start)
    except json.JSONDecodeError as exc:
        raise EvidenceEvalV5Error(f"generation is not one JSON object: {exc}") from exc
    if text[end:].strip():
        raise EvidenceEvalV5Error("generation has trailing content after the JSON object")
    if not isinstance(value, dict):
        raise EvidenceEvalV5Error("generation JSON root must be an object")
    return value


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvidenceEvalV5Error(f"{field} must be a non-empty string")
    return value


def _validate_json_tree(value: Any, field: str) -> None:
    try:
        canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceEvalV5Error(f"{field} is not finite JSON data") from exc


def validate_student_answer(value: Mapping[str, Any]) -> list[str]:
    """Return schema violations for one generated or gold answer."""

    errors: list[str] = []
    keys = set(value)
    if keys != TARGET_KEYS:
        missing = sorted(TARGET_KEYS - keys)
        extra = sorted(keys - TARGET_KEYS)
        errors.append(f"target keys mismatch; missing={missing}, extra={extra}")
    if value.get("schema") != ANSWER_SCHEMA:
        errors.append(f"schema must equal {ANSWER_SCHEMA}")
    decision = value.get("decision")
    if decision not in ALLOWED_DECISIONS:
        errors.append("decision must be ANSWER or REFUSE")
    task = value.get("task")
    if not isinstance(task, str) or not task:
        errors.append("task must be a non-empty string")
    if not isinstance(value.get("claim"), str):
        errors.append("claim must be a string")
    verdict = value.get("verdict")
    if verdict not in ALLOWED_VERDICTS:
        errors.append("verdict must be SUPPORTED or REFUSED")
    if decision == "ANSWER" and verdict != "SUPPORTED":
        errors.append("ANSWER requires verdict=SUPPORTED")
    if decision == "REFUSE" and verdict != "REFUSED":
        errors.append("REFUSE requires verdict=REFUSED")
    evidence_ids = value.get("evidence_ids")
    if not isinstance(evidence_ids, list):
        errors.append("evidence_ids must be a list")
    elif any(not isinstance(item, str) or not item for item in evidence_ids) or len(set(evidence_ids)) != len(
        evidence_ids
    ):
        errors.append("evidence_ids must contain unique non-empty strings")
    if decision == "ANSWER":
        if value.get("claim") == "":
            errors.append("ANSWER requires a non-empty claim")
        if evidence_ids == []:
            errors.append("ANSWER requires at least one evidence ID")
    if decision == "REFUSE":
        if value.get("claim") != "":
            errors.append("REFUSE requires an empty claim")
        if evidence_ids != []:
            errors.append("REFUSE requires an empty evidence_ids list")
    provenance = value.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("provenance must be an object")
    else:
        try:
            _validate_json_tree(provenance, "provenance")
        except EvidenceEvalV5Error as exc:
            errors.append(str(exc))
    return errors


def _safe_split_path(dataset_dir: Path, relative: str) -> Path:
    pure = PurePosixPath(relative.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise EvidenceEvalV5Error(f"unsafe dataset path: {relative}")
    if ":" in pure.parts[0] or pure.suffix.lower() != ".jsonl":
        raise EvidenceEvalV5Error(f"dataset path must be a relative JSONL file: {relative}")
    root = dataset_dir.resolve()
    unresolved = root / Path(*pure.parts)
    if unresolved.is_symlink():
        raise EvidenceEvalV5Error("dataset split must not be a symlink")
    candidate = unresolved.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise EvidenceEvalV5Error("dataset file escapes dataset_dir") from exc
    if not candidate.is_file():
        raise EvidenceEvalV5Error(f"dataset file is missing: {candidate}")
    return candidate


def _descriptor_candidates(
    manifest: Mapping[str, Any],
    split: str,
) -> list[Mapping[str, Any]]:
    candidates: list[Mapping[str, Any]] = []
    direct = manifest.get(split)
    if isinstance(direct, Mapping):
        candidates.append(direct)
    for container_name in ("files", "splits", "data_files", "artifacts"):
        container = manifest.get(container_name)
        if isinstance(container, Mapping):
            item = container.get(split)
            if isinstance(item, Mapping):
                candidates.append(item)
            elif isinstance(item, list):
                candidates.extend(entry for entry in item if isinstance(entry, Mapping))
        elif isinstance(container, list):
            candidates.extend(
                entry for entry in container if isinstance(entry, Mapping) and entry.get("split") == split
            )
    return candidates


def _select_descriptor(
    manifest: Mapping[str, Any],
    split: str,
) -> dict[str, Any]:
    parsed: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in _descriptor_candidates(manifest, split):
        if raw.get("split", split) != split:
            continue
        path = raw.get("path")
        digest = raw.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            continue
        item = {
            "path": path.replace("\\", "/"),
            "sha256": digest.lower(),
            "bytes": raw.get("bytes"),
            "examples": raw.get("examples", raw.get("records", raw.get("count"))),
        }
        key = (item["path"], item["sha256"])
        parsed[key] = item
    if len(parsed) != 1:
        raise EvidenceEvalV5Error(f"manifest must bind exactly one integrity descriptor for {split}")
    descriptor = next(iter(parsed.values()))
    if not SHA256_RE.fullmatch(descriptor["sha256"]):
        raise EvidenceEvalV5Error(f"manifest {split} SHA-256 is invalid")
    if descriptor["bytes"] is not None and (
        not isinstance(descriptor["bytes"], int)
        or isinstance(descriptor["bytes"], bool)
        or descriptor["bytes"] < 0
    ):
        raise EvidenceEvalV5Error(f"manifest {split} byte count is invalid")
    if descriptor["examples"] is not None and (
        not isinstance(descriptor["examples"], int)
        or isinstance(descriptor["examples"], bool)
        or descriptor["examples"] < 0
    ):
        raise EvidenceEvalV5Error(f"manifest {split} example count is invalid")
    return descriptor


def _validate_messages(
    messages: Any,
    example_id: str,
) -> tuple[tuple[dict[str, str], dict[str, str]], dict[str, Any]]:
    if not isinstance(messages, list) or len(messages) != 3:
        raise EvidenceEvalV5Error(f"{example_id} messages must be system/user/assistant")
    normalized: list[dict[str, str]] = []
    expected_roles = ("system", "user", "assistant")
    for index, expected_role in enumerate(expected_roles):
        message = messages[index]
        if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
            raise EvidenceEvalV5Error(f"{example_id} message {index} is invalid")
        if message.get("role") != expected_role:
            raise EvidenceEvalV5Error(f"{example_id} message {index} must have role={expected_role}")
        content = _require_string(
            message.get("content"),
            f"{example_id}.messages[{index}].content",
        )
        normalized.append({"role": expected_role, "content": content})
    expected = parse_single_json_object(normalized[2]["content"])
    target_errors = validate_student_answer(expected)
    if target_errors:
        raise EvidenceEvalV5Error(f"{example_id} assistant target violates v5 schema: {target_errors}")
    prompt_text = "\n".join(item["content"] for item in normalized[:2])
    if normalized[2]["content"] in prompt_text:
        raise EvidenceEvalV5Error(f"{example_id} exact assistant target leaks into prompt")
    return (normalized[0], normalized[1]), expected


def _validate_example(
    raw: Any,
    *,
    split: str,
    line_number: int,
) -> EvidenceSampleV5:
    if not isinstance(raw, Mapping):
        raise EvidenceEvalV5Error(f"line {line_number} is not a JSON object")
    required = {
        "schema",
        "example_id",
        "split",
        "domain",
        "task",
        "decision",
        "messages",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise EvidenceEvalV5Error(f"line {line_number} missing fields: {missing}")
    if raw.get("schema") != EXAMPLE_SCHEMA:
        raise EvidenceEvalV5Error(f"line {line_number} has wrong example schema")
    example_id = _require_string(raw.get("example_id"), "example_id")
    if raw.get("split") != split:
        raise EvidenceEvalV5Error(f"{example_id} split {raw.get('split')!r} does not match {split!r}")
    domain = _require_string(raw.get("domain"), f"{example_id}.domain")
    task = _require_string(raw.get("task"), f"{example_id}.task")
    decision = raw.get("decision")
    if decision not in ALLOWED_DECISIONS:
        raise EvidenceEvalV5Error(f"{example_id}.decision is invalid")
    model_messages, expected = _validate_messages(raw.get("messages"), example_id)
    if expected["decision"] != decision:
        raise EvidenceEvalV5Error(f"{example_id} decision disagrees with target")
    if expected["task"] != task:
        raise EvidenceEvalV5Error(f"{example_id} task disagrees with target")
    return EvidenceSampleV5(
        example_id=example_id,
        split=split,
        domain=domain,
        task=task,
        decision=decision,
        model_messages=model_messages,
        expected=expected,
        raw=dict(raw),
    )


def load_dataset_selection(
    dataset_dir: Path,
    *,
    split: str,
    max_samples: int | None = None,
    blind_authorization_path: Path | None = None,
    blind_authorization_sha256: str | None = None,
    blind_consumption: BlindConsumptionV5 | None = None,
) -> DatasetSelectionV5:
    """Load and integrity-check one v5 split.

    The blind split can only be opened after ``run_evaluation`` has atomically
    consumed a model-bound authorization. Passing the old path/hash pair alone
    is intentionally insufficient.
    """

    if split not in ALLOWED_SPLITS:
        raise EvidenceEvalV5Error(f"unsupported split: {split}")
    if max_samples is not None and (
        not isinstance(max_samples, int) or isinstance(max_samples, bool) or max_samples <= 0
    ):
        raise EvidenceEvalV5Error("max_samples must be a positive integer")
    if split == "blind_test" and max_samples is not None:
        raise BlindTestAuthorizationError("blind_test must be evaluated in full; max_samples is forbidden")
    if split == "blind_test":
        if blind_consumption is None:
            raise BlindTestAuthorizationError(
                "blind_test requires an atomically consumed model-bound authorization"
            )
        if blind_authorization_path is not None or blind_authorization_sha256 is not None:
            raise BlindTestAuthorizationError(
                "raw blind authorization arguments cannot replace a consumption claim"
            )
    elif (
        blind_authorization_path is not None
        or blind_authorization_sha256 is not None
        or blind_consumption is not None
    ):
        raise BlindTestAuthorizationError("blind authorization arguments are invalid for non-blind splits")
    root = dataset_dir.resolve()
    manifest_path = root / "manifest.v5.json"
    if manifest_path.is_symlink():
        raise EvidenceEvalV5Error("manifest.v5.json must not be a symlink")
    if not manifest_path.is_file():
        raise EvidenceEvalV5Error(f"missing v5 manifest: {manifest_path}")
    manifest_payload = manifest_path.read_bytes()
    try:
        manifest = json.loads(
            manifest_payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceEvalV5Error("manifest.v5.json is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise EvidenceEvalV5Error(f"manifest schema must equal {MANIFEST_SCHEMA}")
    descriptor = _select_descriptor(manifest, split)
    split_path = _safe_split_path(root, descriptor["path"])
    authorization: dict[str, Any] | None = None
    if split == "blind_test":
        assert blind_consumption is not None
        if blind_consumption.marker_path.is_symlink():
            raise BlindTestAuthorizationError("blind consumption marker must not be a symlink")
        try:
            marker_payload = blind_consumption.marker_path.read_bytes()
        except OSError as exc:
            raise BlindTestAuthorizationError("blind consumption marker is unavailable") from exc
        try:
            marker = json.loads(
                marker_payload.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_pairs,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BlindTestAuthorizationError("blind consumption marker is invalid") from exc
        expected_marker_status = blind_consumption.pending_receipt.get("status")
        if expected_marker_status not in {"CONSUMED_PENDING", "COMPLETED"}:
            raise BlindTestAuthorizationError("blind consumption claim status is invalid")
        if (
            not isinstance(marker, dict)
            or marker.get("schema") != CONSUMPTION_SCHEMA
            or marker.get("status") != expected_marker_status
            or marker.get("authorization_sha256") != blind_consumption.authorization_sha256
            or marker.get("run_id") != blind_consumption.authorization["evaluation"]["run_id"]
        ):
            raise BlindTestAuthorizationError("blind consumption marker does not match the pending claim")
        authorization = blind_consumption.authorization
    payload = split_path.read_bytes()
    actual_sha256 = sha256_bytes(payload)
    if actual_sha256 != descriptor["sha256"]:
        raise EvidenceEvalV5Error(f"{split} JSONL hash does not match manifest")
    if descriptor["bytes"] is not None and len(payload) != descriptor["bytes"]:
        raise EvidenceEvalV5Error(f"{split} JSONL byte count does not match manifest")

    samples: list[EvidenceSampleV5] = []
    seen_ids: set[str] = set()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceEvalV5Error(f"{split} JSONL is not UTF-8") from exc
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise EvidenceEvalV5Error(f"{split} JSONL has blank line {line_number}")
        try:
            raw = json.loads(line, object_pairs_hook=_reject_duplicate_pairs)
        except json.JSONDecodeError as exc:
            raise EvidenceEvalV5Error(f"{split} JSONL line {line_number} is invalid JSON") from exc
        sample = _validate_example(raw, split=split, line_number=line_number)
        if sample.example_id in seen_ids:
            raise EvidenceEvalV5Error(f"duplicate example_id in {split}: {sample.example_id}")
        seen_ids.add(sample.example_id)
        samples.append(sample)
    if descriptor["examples"] is not None and len(samples) != descriptor["examples"]:
        raise EvidenceEvalV5Error(f"{split} example count does not match manifest")
    if not samples:
        raise EvidenceEvalV5Error(f"{split} contains no examples")
    selected = samples[:max_samples] if max_samples is not None else samples
    return DatasetSelectionV5(
        dataset_dir=root,
        manifest_path=manifest_path,
        manifest_sha256=sha256_bytes(manifest_payload),
        manifest=manifest,
        split=split,
        split_path=split_path,
        split_sha256=actual_sha256,
        samples=tuple(selected),
        blind_test_authorization=authorization,
    )


def load_completed_blind_selection(
    dataset_dir: Path,
    *,
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
) -> DatasetSelectionV5:
    """Reload a completed blind split for post-run integrity verification.

    This path cannot authorize generation. It accepts only the immutable
    terminal artifacts emitted by a prior one-shot evaluation and verifies
    them before the blind JSONL is opened.
    """

    root = Path(dataset_dir).resolve(strict=True)
    if (
        authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("version") != AUTHORIZATION_VERSION
        or authorization.get("status") != "AUTHORIZED"
        or authorization.get("sealed") is not True
        or authorization.get("revoked") is not False
        or authorization.get("scope") != [BLIND_AUTHORIZATION_SCOPE]
    ):
        raise BlindTestAuthorizationError("completed blind authorization contract is invalid")
    if (
        consumption.get("schema") != CONSUMPTION_SCHEMA
        or consumption.get("status") != "COMPLETED"
        or consumption.get("authorization_sha256") != authorization.get("sha256")
        or consumption.get("run_id") != authorization.get("evaluation", {}).get("run_id")
        or consumption.get("failure_is_non_reusable") is not True
    ):
        raise BlindTestAuthorizationError("completed blind consumption contract is invalid")

    def bound_artifact(relative: Any, field: str) -> Path:
        if not isinstance(relative, str) or not relative:
            raise BlindTestAuthorizationError(f"completed blind {field} path is invalid")
        candidate = root / Path(*PurePosixPath(relative).parts)
        if candidate.is_symlink():
            raise BlindTestAuthorizationError(f"completed blind {field} must not be a symlink")
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (FileNotFoundError, ValueError) as exc:
            raise BlindTestAuthorizationError(f"completed blind {field} is unavailable") from exc
        if not resolved.is_file():
            raise BlindTestAuthorizationError(f"completed blind {field} is not a regular file")
        return resolved

    authorization_path = bound_artifact(
        authorization.get("path"),
        "authorization",
    )
    marker_path = bound_artifact(consumption.get("path"), "consumption")
    if sha256_file(authorization_path) != authorization.get("sha256") or sha256_file(
        marker_path
    ) != consumption.get("sha256"):
        raise BlindTestAuthorizationError("completed blind artifact hash mismatch")
    try:
        stored_authorization = json.loads(
            authorization_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
        stored_consumption = json.loads(
            marker_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BlindTestAuthorizationError("completed blind artifact is invalid JSON") from exc
    normalized_authorization = dict(authorization)
    normalized_authorization.pop("path", None)
    normalized_authorization.pop("sha256", None)
    if stored_authorization != normalized_authorization:
        raise BlindTestAuthorizationError("completed blind authorization payload mismatch")
    if (
        stored_consumption.get("status") != "COMPLETED"
        or stored_consumption.get("authorization_sha256") != authorization.get("sha256")
        or stored_consumption.get("run_id") != consumption.get("run_id")
    ):
        raise BlindTestAuthorizationError("completed blind marker payload mismatch")
    claim = BlindConsumptionV5(
        authorization=dict(authorization),
        authorization_path=authorization_path,
        authorization_sha256=str(authorization["sha256"]),
        marker_path=marker_path,
        pending_receipt=dict(stored_consumption),
    )
    return load_dataset_selection(
        root,
        split="blind_test",
        blind_consumption=claim,
    )


def _read_stable_authorization(
    path: Path,
    *,
    dataset_dir: Path,
) -> tuple[dict[str, Any], bytes, Path]:
    root = dataset_dir.resolve()
    candidate = path if path.is_absolute() else root / path
    if candidate.is_symlink():
        raise BlindTestAuthorizationError("authorization receipt cannot be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise BlindTestAuthorizationError(
            "authorization receipt must be a regular file inside dataset_dir"
        ) from exc
    if resolved.parent != root:
        raise BlindTestAuthorizationError("authorization receipt must be a direct child of dataset_dir")
    if not resolved.is_file() or resolved.stat().st_size > MAX_AUTHORIZATION_BYTES:
        raise BlindTestAuthorizationError("authorization receipt is missing, non-regular, or oversized")
    before = resolved.stat()
    first = resolved.read_bytes()
    middle = resolved.stat()
    second = resolved.read_bytes()
    after = resolved.stat()
    identities = {
        (before.st_size, before.st_mtime_ns),
        (middle.st_size, middle.st_mtime_ns),
        (after.st_size, after.st_mtime_ns),
    }
    if len(identities) != 1 or first != second:
        raise BlindTestAuthorizationError("authorization receipt changed during read")
    try:
        receipt = json.loads(
            first.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BlindTestAuthorizationError("authorization receipt is not valid UTF-8 JSON") from exc
    if not isinstance(receipt, dict):
        raise BlindTestAuthorizationError("authorization receipt root must be an object")
    return receipt, first, resolved


def verify_blind_test_authorization(
    selection: DatasetSelectionV5,
    *,
    receipt_path: Path | None,
    expected_sha256: str | None,
    expected_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Verify a sealed, model-bound receipt without opening blind JSONL."""

    if selection.split != "blind_test":
        if receipt_path is not None or expected_sha256 is not None or expected_binding is not None:
            raise BlindTestAuthorizationError(
                "blind-test authorization arguments are invalid for non-blind splits"
            )
        return None
    if receipt_path is None or expected_sha256 is None or expected_binding is None:
        raise BlindTestAuthorizationError("blind_test requires a receipt hash and exact execution binding")
    normalized_expected = expected_sha256.lower()
    if not SHA256_RE.fullmatch(normalized_expected):
        raise BlindTestAuthorizationError("authorization SHA-256 is invalid")
    receipt, payload, resolved = _read_stable_authorization(
        receipt_path,
        dataset_dir=selection.dataset_dir,
    )
    actual_sha256 = sha256_bytes(payload)
    if actual_sha256 != normalized_expected:
        raise BlindTestAuthorizationError("authorization receipt SHA-256 mismatch")
    exact_keys = {
        "schema",
        "version",
        "authorization_id",
        "created_at_utc",
        "status",
        "sealed",
        "revoked",
        "scope",
        "dataset",
        "model",
        "selection_freeze",
        "training_receipt",
        "code",
        "evaluation",
        "consumption",
        "claim_boundary",
    }
    if set(receipt) != exact_keys:
        raise BlindTestAuthorizationError("authorization receipt fields do not match the v5.2 contract")
    if receipt.get("schema") != AUTHORIZATION_SCHEMA:
        raise BlindTestAuthorizationError("authorization receipt schema is invalid")
    if receipt.get("version") != AUTHORIZATION_VERSION:
        raise BlindTestAuthorizationError("authorization receipt version is invalid")
    authorization_id = receipt.get("authorization_id")
    if not isinstance(authorization_id, str) or not RUN_ID_RE.fullmatch(authorization_id):
        raise BlindTestAuthorizationError("authorization_id is invalid")
    if receipt.get("status") != "AUTHORIZED":
        raise BlindTestAuthorizationError("authorization receipt is not approved")
    if receipt.get("sealed") is not True or receipt.get("revoked") is not False:
        raise BlindTestAuthorizationError("authorization receipt must be sealed and not revoked")
    if receipt.get("scope") != [BLIND_AUTHORIZATION_SCOPE]:
        raise BlindTestAuthorizationError("authorization scope does not allow one-shot HF blind evaluation")
    for field in (
        "dataset",
        "model",
        "selection_freeze",
        "training_receipt",
        "code",
        "evaluation",
    ):
        if receipt.get(field) != expected_binding.get(field):
            raise BlindTestAuthorizationError(f"authorization {field} binding mismatch")
    if receipt["dataset"].get("manifest_sha256") != selection.manifest_sha256:
        raise BlindTestAuthorizationError("authorization is bound to a different dataset manifest")
    if receipt["dataset"].get("blind_test_sha256") != selection.split_sha256:
        raise BlindTestAuthorizationError("authorization is bound to a different blind_test JSONL")
    consumption = receipt.get("consumption")
    expected_marker = f".blind_consumptions/{authorization_id}.consumption.v5.json"
    if consumption != {
        "once": True,
        "marker": expected_marker,
        "failure_is_non_reusable": True,
    }:
        raise BlindTestAuthorizationError("authorization consumption contract is invalid")
    normalized = dict(receipt)
    normalized["path"] = resolved.relative_to(selection.dataset_dir).as_posix()
    normalized["sha256"] = actual_sha256
    return normalized


def _blind_provisional_selection(dataset_dir: Path) -> DatasetSelectionV5:
    """Read only manifest metadata and stat the blind file."""

    root = Path(dataset_dir).resolve(strict=True)
    manifest_path = root / "manifest.v5.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise BlindTestAuthorizationError("blind manifest must be a regular non-symlink file")
    payload = manifest_path.read_bytes()
    try:
        manifest = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BlindTestAuthorizationError("blind manifest is invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise BlindTestAuthorizationError("blind manifest schema is invalid")
    descriptor = _select_descriptor(manifest, "blind_test")
    blind_path = _safe_split_path(root, descriptor["path"])
    if descriptor["examples"] != EXPECTED_BLIND_EXAMPLES:
        raise BlindTestAuthorizationError(
            f"blind_test must declare exactly {EXPECTED_BLIND_EXAMPLES} examples"
        )
    if descriptor["bytes"] is None or blind_path.stat().st_size != descriptor["bytes"]:
        raise BlindTestAuthorizationError("blind_test stat size does not match its manifest descriptor")
    return DatasetSelectionV5(
        dataset_dir=root,
        manifest_path=manifest_path,
        manifest_sha256=sha256_bytes(payload),
        manifest=manifest,
        split="blind_test",
        split_path=blind_path,
        split_sha256=descriptor["sha256"],
        samples=(),
        blind_test_authorization=None,
    )


def _exclusive_create(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # A partial file is intentionally retained as a non-reusable claim.
        raise


def _consume_blind_authorization(
    selection: DatasetSelectionV5,
    authorization: Mapping[str, Any],
) -> BlindConsumptionV5:
    relative = PurePosixPath(str(authorization["consumption"]["marker"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise BlindTestAuthorizationError("unsafe blind consumption marker path")
    marker = selection.dataset_dir / Path(*relative.parts)
    resolved_parent = marker.parent.resolve()
    try:
        resolved_parent.relative_to(selection.dataset_dir)
    except ValueError as exc:
        raise BlindTestAuthorizationError("blind consumption marker escapes dataset_dir") from exc
    pending = {
        "schema": CONSUMPTION_SCHEMA,
        "status": "CONSUMED_PENDING",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "authorization_id": authorization["authorization_id"],
        "authorization_sha256": authorization["sha256"],
        "run_id": authorization["evaluation"]["run_id"],
        "output_basename": authorization["evaluation"]["output_basename"],
        "failure_is_non_reusable": True,
    }
    try:
        _exclusive_create(marker, _json_bytes(pending))
    except FileExistsError as exc:
        raise BlindTestAuthorizationError("blind authorization has already been consumed") from exc
    return BlindConsumptionV5(
        authorization=dict(authorization),
        authorization_path=selection.dataset_dir / authorization["path"],
        authorization_sha256=str(authorization["sha256"]),
        marker_path=marker,
        pending_receipt=pending,
    )


def _finalize_blind_consumption(
    consumption: BlindConsumptionV5,
    *,
    status: str,
    error: BaseException | None = None,
) -> dict[str, Any]:
    if status not in {"COMPLETED", "FAILED_NON_REUSABLE"}:
        raise ValueError("invalid blind consumption terminal status")
    receipt = {
        **consumption.pending_receipt,
        "status": status,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "error": (
            None
            if error is None
            else {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": "".join(
                    traceback.format_exception(
                        type(error),
                        error,
                        error.__traceback__,
                    )
                ),
            }
        ),
    }
    _atomic_write(consumption.marker_path, _json_bytes(receipt))
    return {
        "schema": CONSUMPTION_SCHEMA,
        "status": status,
        "path": consumption.marker_path.relative_to(consumption.authorization_path.parent).as_posix(),
        "sha256": sha256_file(consumption.marker_path),
        "authorization_sha256": consumption.authorization_sha256,
        "run_id": receipt["run_id"],
        "output_basename": receipt["output_basename"],
        "failure_is_non_reusable": True,
    }


def _replace_sections(
    content: str,
    *,
    replacement: str,
) -> tuple[str, int]:
    return EVIDENCE_SECTION_RE.subn(replacement, content)


def _evidence_sections(content: str) -> list[str]:
    return [match.group(0) for match in EVIDENCE_SECTION_RE.finditer(content)]


def _ablate_messages(
    sample: EvidenceSampleV5,
    *,
    ablation: str,
    donor: EvidenceSampleV5 | None,
) -> tuple[dict[str, str], dict[str, str]]:
    system = dict(sample.model_messages[0])
    user = dict(sample.model_messages[1])
    original = user["content"]
    if ablation == "none":
        return system, user
    if ablation == "evidence_removed":
        changed, count = _replace_sections(
            original,
            replacement="[EVIDENCE_REMOVED_BY_EVALUATOR]",
        )
    elif ablation == "no_rag":
        changed, count = _replace_sections(original, replacement="")
        if count:
            changed = changed.rstrip() + "\n\n[ABLATION]\nNO_RAG_CONTEXT\n[/ABLATION]"
    elif ablation == "evidence_swapped":
        if donor is None:
            raise EvidenceEvalV5Error("evidence_swapped requires a donor")
        donor_sections = _evidence_sections(donor.model_messages[1]["content"])
        if not donor_sections:
            raise EvidenceEvalV5Error(f"evidence-swapped donor {donor.example_id} has no evidence section")
        replacement = "\n\n".join(donor_sections)
        changed, count = _replace_sections(original, replacement=replacement)
    else:
        raise EvidenceEvalV5Error(f"unsupported ablation: {ablation}")
    if count == 0 or changed == original:
        raise EvidenceEvalV5Error(f"{sample.example_id} has no recognizable evidence section for {ablation}")
    user["content"] = changed
    return system, user


def build_generation_requests(
    samples: Sequence[EvidenceSampleV5],
    *,
    ablation: str,
) -> tuple[GenerationRequestV5, ...]:
    """Build target-free prompts for one ablation."""

    if ablation not in ALLOWED_ABLATIONS:
        raise EvidenceEvalV5Error(f"unsupported ablation: {ablation}")
    if not samples:
        raise EvidenceEvalV5Error("cannot build requests for an empty selection")
    ordered = sorted(samples, key=lambda item: item.example_id)
    donors: dict[str, EvidenceSampleV5] = {}
    if ablation == "evidence_swapped":
        by_domain: dict[str, list[EvidenceSampleV5]] = defaultdict(list)
        for sample in ordered:
            by_domain[sample.domain].append(sample)
        for domain_samples in by_domain.values():
            if len(domain_samples) > 1:
                for index, sample in enumerate(domain_samples):
                    donors[sample.example_id] = domain_samples[(index + 1) % len(domain_samples)]
        for index, sample in enumerate(ordered):
            donors.setdefault(sample.example_id, ordered[(index + 1) % len(ordered)])
            if donors[sample.example_id].example_id == sample.example_id:
                raise EvidenceEvalV5Error("evidence_swapped requires at least two examples")
    requests: list[GenerationRequestV5] = []
    for sample in samples:
        messages = _ablate_messages(
            sample,
            ablation=ablation,
            donor=donors.get(sample.example_id),
        )
        prompt_sha256 = sha256_bytes(canonical_json(messages).encode("utf-8"))
        requests.append(
            GenerationRequestV5(
                example_id=sample.example_id,
                split=sample.split,
                domain=sample.domain,
                task=sample.task,
                ablation=ablation,
                messages=messages,
                prompt_sha256=prompt_sha256,
            )
        )
    return tuple(requests)


def _backend_metadata(mode: str, **details: Any) -> dict[str, Any]:
    if mode not in ALLOWED_BACKENDS:
        raise EvidenceEvalV5Error(f"unsupported backend mode: {mode}")
    is_model = mode == "hf_model"
    metadata = {
        "mode": mode,
        "is_model": is_model,
        "free_generation_executed": is_model,
        "assistant_target_visible_to_backend": False,
        "model_quality_evidence": is_model,
        "claim_boundary": ("LOCAL_HF_FREE_GENERATION" if is_model else "NON_MODEL_EVALUATOR_TEST_ONLY"),
    }
    metadata.update(details)
    return metadata


def deterministic_baseline_generations(
    requests: Sequence[GenerationRequestV5],
) -> tuple[dict[tuple[str, str], str], dict[str, Any]]:
    """Generate an always-refuse rule baseline without inspecting gold targets."""

    generations: dict[tuple[str, str], str] = {}
    for request in requests:
        value = {
            "schema": ANSWER_SCHEMA,
            "decision": "REFUSE",
            "task": request.task,
            "claim": "",
            "verdict": "REFUSED",
            "evidence_ids": [],
            "provenance": {
                "backend": "deterministic_baseline",
                "evidence_used": False,
            },
        }
        generations[(request.ablation, request.example_id)] = canonical_json(value)
    return generations, _backend_metadata(
        "deterministic_baseline",
        algorithm="always_refuse_v1",
    )


def load_fixture_generations(
    path: Path,
    requests: Sequence[GenerationRequestV5],
) -> tuple[dict[tuple[str, str], str], dict[str, Any]]:
    """Load explicit evaluator fixtures keyed by ablation and example_id."""

    if not path.is_file():
        raise EvidenceEvalV5Error(f"fixture generations file is missing: {path}")
    generations: dict[tuple[str, str], str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise EvidenceEvalV5Error(f"fixture JSONL has blank line {line_number}")
        try:
            item = json.loads(line, object_pairs_hook=_reject_duplicate_pairs)
        except json.JSONDecodeError as exc:
            raise EvidenceEvalV5Error(f"fixture JSONL line {line_number} is invalid") from exc
        if not isinstance(item, Mapping):
            raise EvidenceEvalV5Error(f"fixture line {line_number} is not an object")
        example_id = _require_string(item.get("example_id"), "fixture.example_id")
        ablation = item.get("ablation", "none")
        if ablation not in ALLOWED_ABLATIONS:
            raise EvidenceEvalV5Error(f"fixture line {line_number} ablation is invalid")
        generation = item.get("generation")
        if not isinstance(generation, str):
            raise EvidenceEvalV5Error(f"fixture line {line_number} generation must be a string")
        key = (str(ablation), example_id)
        if key in generations:
            raise EvidenceEvalV5Error(f"duplicate fixture generation: {key}")
        generations[key] = generation
    expected = {(request.ablation, request.example_id) for request in requests}
    if set(generations) != expected:
        missing = sorted(expected - set(generations))
        extra = sorted(set(generations) - expected)
        raise EvidenceEvalV5Error(f"fixture generation membership mismatch; missing={missing}, extra={extra}")
    return generations, _backend_metadata(
        "fixture",
        fixture_path=str(path.resolve()),
        fixture_sha256=sha256_file(path),
    )


def _model_inventory(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise EvidenceEvalV5Error(f"model artifact root must not be a symlink: {path}")
    root = path.resolve()
    if not root.is_dir():
        raise EvidenceEvalV5Error(f"model artifact directory is missing: {path}")
    records: list[dict[str, Any]] = []
    # Match the inventory order committed by qlora_full_v5 and verified by the
    # pre-blind selection freeze. On Windows, pathlib path ordering differs
    # from a case-sensitive POSIX-string sort for names such as LICENSE.
    for candidate in sorted(root.rglob("*")):
        if candidate.is_symlink():
            raise EvidenceEvalV5Error(f"model artifact tree contains a symlink: {candidate}")
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        records.append(
            {
                "path": candidate.relative_to(root).as_posix(),
                "bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    if not records:
        raise EvidenceEvalV5Error(f"model artifact directory is empty: {path}")
    return {
        "path": str(root),
        "files": records,
        "content_sha256": sha256_bytes(canonical_json(records).encode("utf-8")),
    }


def hf_decoding_contract(
    *,
    device: str,
    seed: int,
    max_input_tokens: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Canonical decoding contract shared by authorization and execution."""

    if device not in {"cpu", "cuda"}:
        raise EvidenceEvalV5Error("blind HF evaluation requires an explicit cpu or cuda device")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise EvidenceEvalV5Error("generation seed must be an integer")
    if (
        isinstance(max_input_tokens, bool)
        or not isinstance(max_input_tokens, int)
        or max_input_tokens <= 0
        or isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
    ):
        raise EvidenceEvalV5Error("token limits must be positive integers")
    return {
        "device": device,
        "seed": seed,
        "do_sample": False,
        "num_beams": 1,
        "temperature": 0,
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": max_new_tokens,
        "chat_template": "base_model_tokenizer.apply_chat_template",
        "add_generation_prompt": True,
        "tokenizer_add_special_tokens": False,
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
    }


def _validate_run_identity(run_id: str, output_basename: str) -> None:
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        raise BlindTestAuthorizationError("blind run_id must be 8-128 safe filename characters")
    if (
        not isinstance(output_basename, str)
        or not OUTPUT_BASENAME_RE.fullmatch(output_basename)
        or output_basename in {".", ".."}
    ):
        raise BlindTestAuthorizationError("blind output basename is invalid")


def _selection_evidence_binding(
    *,
    subject: str,
    dataset_dir: Path,
    base_model_dir: Path,
    adapter_dir: Path | None,
    base_inventory: Mapping[str, Any],
    adapter_inventory: Mapping[str, Any] | None,
    selection_freeze_path: Path | None,
    selection_freeze_sha256: str | None,
    training_receipt_path: Path | None,
) -> dict[str, Any]:
    supplied = (
        selection_freeze_path,
        selection_freeze_sha256,
        training_receipt_path,
    )
    if subject == "base":
        if any(value is not None for value in supplied):
            raise BlindTestAuthorizationError(
                "base blind authorization rejects selection-freeze and training-receipt arguments"
            )
        return {
            "selection_freeze": None,
            "training_receipt": None,
        }
    if subject != "candidate":
        raise BlindTestAuthorizationError("blind subject must be base or candidate")
    if adapter_dir is None or adapter_inventory is None:
        raise BlindTestAuthorizationError("candidate blind authorization requires an adapter")
    if any(value is None for value in supplied):
        raise BlindTestAuthorizationError(
            "candidate blind authorization requires selection-freeze path/hash and training receipt"
        )
    assert selection_freeze_path is not None
    assert selection_freeze_sha256 is not None
    assert training_receipt_path is not None
    normalized_freeze_sha = selection_freeze_sha256.lower()
    if not SHA256_RE.fullmatch(normalized_freeze_sha):
        raise BlindTestAuthorizationError("selection-freeze SHA-256 is invalid")
    freeze_raw = Path(selection_freeze_path)
    training_raw = Path(training_receipt_path)
    if freeze_raw.is_symlink() or training_raw.is_symlink():
        raise BlindTestAuthorizationError("selection-freeze and training receipt must not be symlinks")
    try:
        freeze_resolved = freeze_raw.resolve(strict=True)
        training_resolved = training_raw.resolve(strict=True)
    except FileNotFoundError as exc:
        raise BlindTestAuthorizationError("selection-freeze or training receipt is unavailable") from exc
    actual_freeze_sha = sha256_file(freeze_resolved)
    if actual_freeze_sha != normalized_freeze_sha:
        raise BlindTestAuthorizationError("selection-freeze SHA-256 mismatch")
    try:
        verified = verify_selection_freeze(
            freeze_receipt_path=freeze_resolved,
            training_receipt_path=training_resolved,
            dataset_dir=dataset_dir,
            base_model_dir=base_model_dir,
            selected_adapter_dir=adapter_dir,
        )
    except SelectionFreezeV5Error as exc:
        raise BlindTestAuthorizationError("candidate selection freeze verification failed") from exc
    if verified.get("sha256") != actual_freeze_sha:
        raise BlindTestAuthorizationError("selection-freeze verifier returned a different file hash")
    if (
        verified.get("dataset_manifest_sha256") != sha256_file(Path(dataset_dir) / "manifest.v5.json")
        or verified.get("base_model_tree_sha256") != base_inventory["content_sha256"]
        or verified.get("selected_adapter_tree_sha256") != adapter_inventory["content_sha256"]
    ):
        raise BlindTestAuthorizationError(
            "selection freeze does not bind the evaluated dataset/model/adapter"
        )
    training_sha = sha256_file(training_resolved)
    if verified.get("training_receipt_sha256") != training_sha:
        raise BlindTestAuthorizationError("selection freeze does not bind the supplied training receipt")
    return {
        "selection_freeze": {
            "path": str(freeze_resolved),
            "sha256": actual_freeze_sha,
            "canonical_digest_sha256": verified["canonical_digest_sha256"],
            "verification_status": verified["status"],
            "selected_adapter_tree_sha256": verified["selected_adapter_tree_sha256"],
        },
        "training_receipt": {
            "path": str(training_resolved),
            "sha256": training_sha,
        },
    }


def _blind_binding(
    *,
    selection: DatasetSelectionV5,
    base_inventory: Mapping[str, Any],
    adapter_inventory: Mapping[str, Any] | None,
    subject: str,
    code: Mapping[str, Any],
    decoding: Mapping[str, Any],
    selection_evidence: Mapping[str, Any],
    run_id: str,
    output_basename: str,
) -> dict[str, Any]:
    _validate_run_identity(run_id, output_basename)
    if subject not in {"base", "candidate"}:
        raise BlindTestAuthorizationError("blind subject must be base or candidate")
    if subject == "base" and adapter_inventory is not None:
        raise BlindTestAuthorizationError("base blind run requires adapter=null")
    if subject == "candidate" and adapter_inventory is None:
        raise BlindTestAuthorizationError("candidate blind run requires an adapter")
    return {
        "dataset": {
            "manifest_sha256": selection.manifest_sha256,
            "blind_test_sha256": selection.split_sha256,
            "blind_test_bytes": selection.split_path.stat().st_size,
            "expected_examples": EXPECTED_BLIND_EXAMPLES,
        },
        "model": {
            "subject": subject,
            "base_model_tree_sha256": base_inventory["content_sha256"],
            "adapter_tree_sha256": (
                None if adapter_inventory is None else adapter_inventory["content_sha256"]
            ),
        },
        "selection_freeze": selection_evidence["selection_freeze"],
        "training_receipt": selection_evidence["training_receipt"],
        "code": {
            "evaluator_sha256": code["evaluator"]["sha256"],
            "runner_sha256": code["runner"]["sha256"],
        },
        "evaluation": {
            "backend_mode": "hf_model",
            "split": "blind_test",
            "ablations": ["none"],
            "max_samples": None,
            "expected_examples": EXPECTED_BLIND_EXAMPLES,
            "run_id": run_id,
            "output_basename": output_basename,
            "decoding": dict(decoding),
        },
    }


def build_blind_test_authorization(
    *,
    dataset_dir: Path,
    base_model_dir: Path,
    adapter_dir: Path | None,
    subject: str,
    runner_path: Path,
    run_id: str,
    output_basename: str,
    device: str,
    seed: int,
    max_input_tokens: int,
    max_new_tokens: int,
    selection_freeze_path: Path | None = None,
    selection_freeze_sha256: str | None = None,
    training_receipt_path: Path | None = None,
    authorization_path: Path,
) -> dict[str, Any]:
    """Create one immutable blind authorization without opening blind JSONL."""

    selection = _blind_provisional_selection(dataset_dir)
    base_inventory = _model_inventory(base_model_dir)
    adapter_inventory = None if adapter_dir is None else _model_inventory(adapter_dir)
    code = _code_hashes(runner_path)
    if set(code) != {"evaluator", "runner"}:
        raise BlindTestAuthorizationError("blind authorization requires evaluator and runner hashes")
    decoding = hf_decoding_contract(
        device=device,
        seed=seed,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
    )
    selection_evidence = _selection_evidence_binding(
        subject=subject,
        dataset_dir=selection.dataset_dir,
        base_model_dir=base_model_dir,
        adapter_dir=adapter_dir,
        base_inventory=base_inventory,
        adapter_inventory=adapter_inventory,
        selection_freeze_path=selection_freeze_path,
        selection_freeze_sha256=selection_freeze_sha256,
        training_receipt_path=training_receipt_path,
    )
    binding = _blind_binding(
        selection=selection,
        base_inventory=base_inventory,
        adapter_inventory=adapter_inventory,
        subject=subject,
        code=code,
        decoding=decoding,
        selection_evidence=selection_evidence,
        run_id=run_id,
        output_basename=output_basename,
    )
    authorization_id = "icmat-blind-" + sha256_bytes(canonical_json(binding).encode("utf-8"))[:24]
    receipt = {
        "schema": AUTHORIZATION_SCHEMA,
        "version": AUTHORIZATION_VERSION,
        "authorization_id": authorization_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "AUTHORIZED",
        "sealed": True,
        "revoked": False,
        "scope": [BLIND_AUTHORIZATION_SCOPE],
        **binding,
        "consumption": {
            "once": True,
            "marker": (f".blind_consumptions/{authorization_id}.consumption.v5.json"),
            "failure_is_non_reusable": True,
        },
        "claim_boundary": (
            "This receipt authorizes exactly one full, none-only local HF blind "
            "evaluation for the bound artifacts. It does not authorize model "
            "selection after results, GGUF comparison, X5, or production use."
        ),
    }
    output = Path(authorization_path)
    if output.is_absolute():
        resolved_parent = output.parent.resolve(strict=True)
        final = resolved_parent / output.name
    else:
        final = selection.dataset_dir / output
    try:
        final.parent.resolve(strict=True).relative_to(selection.dataset_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise BlindTestAuthorizationError("authorization output must be inside dataset_dir") from exc
    if final.parent.resolve(strict=True) != selection.dataset_dir:
        raise BlindTestAuthorizationError("authorization output must be a direct child of dataset_dir")
    if final.suffix.lower() != ".json":
        raise BlindTestAuthorizationError("authorization output must be JSON")
    _exclusive_create(final, _json_bytes(receipt))
    return {
        "path": str(final.resolve(strict=True)),
        "sha256": sha256_file(final),
        "authorization": receipt,
    }


def generate_hf_model(
    requests: Sequence[GenerationRequestV5],
    *,
    base_model_dir: Path,
    adapter_dir: Path | None = None,
    device: str = "auto",
    seed: int = 20260729,
    max_input_tokens: int = 2048,
    max_new_tokens: int = 384,
    expected_base_model_tree_sha256: str | None = None,
    expected_adapter_tree_sha256: str | None = None,
) -> tuple[
    dict[tuple[str, str], str],
    dict[str, Any],
    dict[tuple[str, str], dict[str, Any]],
]:
    """Run deterministic local HF generation with no network access."""

    if device not in {"auto", "cpu", "cuda"}:
        raise EvidenceEvalV5Error(f"unsupported device: {device}")
    if max_input_tokens <= 0 or max_new_tokens <= 0:
        raise EvidenceEvalV5Error("token limits must be positive")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise EvidenceEvalV5Error("hf_model mode requires local torch and transformers") from exc
    if adapter_dir is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise EvidenceEvalV5Error("adapter evaluation requires local peft") from exc
    else:
        PeftModel = None

    base_inventory = _model_inventory(base_model_dir)
    adapter_inventory = _model_inventory(adapter_dir) if adapter_dir is not None else None
    if (
        expected_base_model_tree_sha256 is not None
        and base_inventory["content_sha256"] != expected_base_model_tree_sha256
    ):
        raise BlindTestAuthorizationError("actual base model inventory does not match blind authorization")
    actual_adapter_sha = None if adapter_inventory is None else adapter_inventory["content_sha256"]
    if expected_base_model_tree_sha256 is not None and actual_adapter_sha != expected_adapter_tree_sha256:
        raise BlindTestAuthorizationError("actual adapter inventory does not match blind authorization")
    selected_device = device
    if device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    if selected_device == "cuda" and not torch.cuda.is_available():
        raise EvidenceEvalV5Error("CUDA was requested but is unavailable")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_dir.resolve()),
        local_files_only=True,
        trust_remote_code=False,
    )
    dtype = torch.float16 if selected_device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_dir.resolve()),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=dtype,
    )
    if adapter_dir is not None:
        assert PeftModel is not None
        model = PeftModel.from_pretrained(
            model,
            str(adapter_dir.resolve()),
            local_files_only=True,
            is_trainable=False,
        )
    model.to(selected_device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    generations: dict[tuple[str, str], str] = {}
    traces: dict[tuple[str, str], dict[str, Any]] = {}
    started = time.perf_counter()
    with torch.inference_mode():
        for request in requests:
            prompt = tokenizer.apply_chat_template(
                [dict(message) for message in request.messages],
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            input_tokens = int(encoded["input_ids"].shape[-1])
            if input_tokens > max_input_tokens:
                raise EvidenceEvalV5Error(
                    f"{request.example_id} prompt has {input_tokens} tokens, "
                    f"exceeding max_input_tokens={max_input_tokens}"
                )
            encoded = {key: value.to(selected_device) for key, value in encoded.items()}
            sample_started = time.perf_counter()
            output = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            latency_ms = (time.perf_counter() - sample_started) * 1000.0
            generated_ids = output[0, input_tokens:]
            generation = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            key = (request.ablation, request.example_id)
            generations[key] = generation
            traces[key] = {
                "input_tokens": input_tokens,
                "output_tokens": int(generated_ids.shape[-1]),
                "latency_ms": latency_ms,
            }
    elapsed_seconds = time.perf_counter() - started
    base_after = _model_inventory(base_model_dir)
    adapter_after = _model_inventory(adapter_dir) if adapter_dir is not None else None
    if base_after["content_sha256"] != base_inventory["content_sha256"]:
        raise EvidenceEvalV5Error("base model changed during generation")
    if (None if adapter_after is None else adapter_after["content_sha256"]) != actual_adapter_sha:
        raise EvidenceEvalV5Error("adapter changed during generation")
    decoding = hf_decoding_contract(
        device=selected_device,
        seed=seed,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
    )
    metadata = _backend_metadata(
        "hf_model",
        subject="adapter" if adapter_dir is not None else "base",
        device=selected_device,
        seed=seed,
        decoding=decoding,
        base_model=base_inventory,
        adapter=adapter_inventory,
        inventories_unchanged_after_generation=True,
        elapsed_seconds=elapsed_seconds,
        samples_generated=len(requests),
        network_allowed=False,
        local_files_only=True,
    )
    return generations, metadata, traces


def _metric(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _score_one(
    sample: EvidenceSampleV5,
    *,
    ablation: str,
    generation: str,
    backend: Mapping[str, Any],
    request: GenerationRequestV5,
    trace: Mapping[str, Any] | None,
) -> dict[str, Any]:
    prediction: dict[str, Any] | None = None
    parse_error: str | None = None
    schema_errors: list[str] = []
    try:
        prediction = parse_single_json_object(generation)
        schema_errors = validate_student_answer(prediction)
    except EvidenceEvalV5Error as exc:
        parse_error = str(exc)
    expected = sample.expected

    def exact(field: str) -> bool:
        return prediction is not None and prediction.get(field) == expected.get(field)

    json_valid = prediction is not None
    schema_valid = json_valid and not schema_errors
    flags = {
        "json_valid": json_valid,
        "schema_valid": schema_valid,
        "schema_exact": exact("schema"),
        "citation_exact": exact("evidence_ids"),
        "decision_exact": exact("decision"),
        "task_exact": exact("task"),
        "claim_exact": exact("claim"),
        "verdict_exact": exact("verdict"),
        "provenance_exact": exact("provenance"),
        "strict_exact": prediction == expected,
    }
    predicted_decision = (
        prediction.get("decision")
        if prediction is not None and prediction.get("decision") in ALLOWED_DECISIONS
        else None
    )
    flags["unsupported_wrong_answer"] = sample.decision == "REFUSE" and predicted_decision == "ANSWER"
    return {
        "schema": SAMPLE_RESULT_SCHEMA,
        "example_id": sample.example_id,
        "split": sample.split,
        "domain": sample.domain,
        "task": sample.task,
        "gold_decision": sample.decision,
        "ablation": ablation,
        "backend_mode": backend["mode"],
        "prompt_sha256": request.prompt_sha256,
        "assistant_target_visible_to_backend": False,
        "generation": generation,
        "generation_sha256": sha256_bytes(generation.encode("utf-8")),
        "prediction": prediction,
        "expected": expected,
        "parse_error": parse_error,
        "schema_errors": schema_errors,
        "predicted_decision": predicted_decision,
        "metrics": flags,
        "trace": dict(trace) if trace is not None else None,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    metric_names = (
        "json_valid",
        "schema_valid",
        "schema_exact",
        "citation_exact",
        "decision_exact",
        "task_exact",
        "claim_exact",
        "verdict_exact",
        "provenance_exact",
        "strict_exact",
    )
    metrics = {
        name: _metric(
            sum(bool(row["metrics"][name]) for row in rows),
            count,
        )
        for name in metric_names
    }
    answer_rows = [row for row in rows if row["gold_decision"] == "ANSWER"]
    refuse_rows = [row for row in rows if row["gold_decision"] == "REFUSE"]
    metrics["answer_accuracy"] = _metric(
        sum(bool(row["metrics"]["strict_exact"]) for row in answer_rows),
        len(answer_rows),
    )
    wrong_answers = sum(bool(row["metrics"]["unsupported_wrong_answer"]) for row in refuse_rows)
    metrics["unsupported_wrong_answer_rate"] = _metric(
        wrong_answers,
        len(refuse_rows),
    )
    tp = sum(row["gold_decision"] == "REFUSE" and row["predicted_decision"] == "REFUSE" for row in rows)
    fp = sum(row["gold_decision"] != "REFUSE" and row["predicted_decision"] == "REFUSE" for row in rows)
    fn = sum(row["gold_decision"] == "REFUSE" and row["predicted_decision"] != "REFUSE" for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "examples": count,
        "metrics": metrics,
        "refuse": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
    }


def _stratify(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    source_field = "gold_decision" if field == "decision" else field
    for row in rows:
        groups[str(row[source_field])].append(row)
    return {key: _aggregate(groups[key]) for key in sorted(groups)}


def score_generations(
    *,
    selection: DatasetSelectionV5,
    requests: Sequence[GenerationRequestV5],
    generations: Mapping[tuple[str, str], str],
    backend: Mapping[str, Any],
    traces: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score generated strings against held-out targets."""

    sample_by_id = {sample.example_id: sample for sample in selection.samples}
    rows: list[dict[str, Any]] = []
    expected_keys = {(request.ablation, request.example_id) for request in requests}
    if set(generations) != expected_keys:
        raise EvidenceEvalV5Error("generation membership does not match requests")
    for request in requests:
        sample = sample_by_id[request.example_id]
        key = (request.ablation, request.example_id)
        rows.append(
            _score_one(
                sample,
                ablation=request.ablation,
                generation=generations[key],
                backend=backend,
                request=request,
                trace=(traces or {}).get(key),
            )
        )
    ablation_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        ablation_groups[row["ablation"]].append(row)
    summaries: dict[str, Any] = {}
    for ablation in sorted(ablation_groups):
        ablation_rows = ablation_groups[ablation]
        summaries[ablation] = {
            **_aggregate(ablation_rows),
            "stratified": {
                "domain": _stratify(ablation_rows, "domain"),
                "task": _stratify(ablation_rows, "task"),
                "decision": _stratify(ablation_rows, "decision"),
            },
        }
    return rows, summaries


def rescore_persisted_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    backend: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recompute persisted evaluation rows without trusting stored metrics."""

    if not rows:
        raise EvidenceEvalV5Error("cannot rescore an empty evaluation")
    recomputed: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        required = {
            "example_id",
            "split",
            "domain",
            "task",
            "gold_decision",
            "ablation",
            "prompt_sha256",
            "generation",
            "expected",
        }
        missing = sorted(required - set(row))
        if missing:
            raise EvidenceEvalV5Error(f"persisted row {index} is missing fields: {missing}")
        expected = row["expected"]
        if not isinstance(expected, Mapping):
            raise EvidenceEvalV5Error(f"persisted row {index} expected target is invalid")
        expected_dict = dict(expected)
        target_errors = validate_student_answer(expected_dict)
        if target_errors:
            raise EvidenceEvalV5Error(
                f"persisted row {index} expected target violates schema: {target_errors}"
            )
        example_id = _require_string(row["example_id"], "example_id")
        split = _require_string(row["split"], "split")
        domain = _require_string(row["domain"], "domain")
        task = _require_string(row["task"], "task")
        decision = row["gold_decision"]
        if decision not in ALLOWED_DECISIONS:
            raise EvidenceEvalV5Error(f"persisted row {index} gold decision is invalid")
        if expected_dict.get("decision") != decision:
            raise EvidenceEvalV5Error(f"persisted row {index} target decision mismatch")
        if expected_dict.get("task") != task:
            raise EvidenceEvalV5Error(f"persisted row {index} target task mismatch")
        ablation = _require_string(row["ablation"], "ablation")
        if ablation not in ALLOWED_ABLATIONS:
            raise EvidenceEvalV5Error(f"persisted row {index} ablation is invalid")
        prompt_sha256 = _require_string(row["prompt_sha256"], "prompt_sha256")
        if not SHA256_RE.fullmatch(prompt_sha256):
            raise EvidenceEvalV5Error(f"persisted row {index} prompt SHA-256 is invalid")
        generation = row["generation"]
        if not isinstance(generation, str):
            raise EvidenceEvalV5Error(f"persisted row {index} generation is not a string")
        sample = EvidenceSampleV5(
            example_id=example_id,
            split=split,
            domain=domain,
            task=task,
            decision=decision,
            model_messages=(
                {"role": "system", "content": "[PERSISTED_PROMPT]"},
                {"role": "user", "content": "[PERSISTED_PROMPT]"},
            ),
            expected=expected_dict,
            raw={},
        )
        request = GenerationRequestV5(
            example_id=example_id,
            split=split,
            domain=domain,
            task=task,
            ablation=ablation,
            messages=sample.model_messages,
            prompt_sha256=prompt_sha256,
        )
        trace = row.get("trace")
        if trace is not None and not isinstance(trace, Mapping):
            raise EvidenceEvalV5Error(f"persisted row {index} trace must be an object or null")
        recomputed.append(
            _score_one(
                sample,
                ablation=ablation,
                generation=generation,
                backend=backend,
                request=request,
                trace=trace,
            )
        )
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in recomputed:
        groups[row["ablation"]].append(row)
    summaries = {
        ablation: {
            **_aggregate(group),
            "stratified": {
                "domain": _stratify(group, "domain"),
                "task": _stratify(group, "task"),
                "decision": _stratify(group, "decision"),
            },
        }
        for ablation, group in sorted(groups.items())
    }
    return recomputed, summaries


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _code_hashes(runner_path: Path | None) -> dict[str, Any]:
    paths = {"evaluator": Path(__file__).resolve()}
    if runner_path is not None:
        paths["runner"] = runner_path.resolve()
    return {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in paths.items()
        if path.is_file()
    }


def _validate_ablation_list(ablations: Sequence[str]) -> tuple[str, ...]:
    if not ablations:
        return ("none",)
    unique: list[str] = []
    for ablation in ablations:
        if ablation not in ALLOWED_ABLATIONS:
            raise EvidenceEvalV5Error(f"unsupported ablation: {ablation}")
        if ablation not in unique:
            unique.append(ablation)
    return tuple(unique)


def run_evaluation(
    *,
    dataset_dir: Path,
    split: str,
    output_dir: Path,
    backend_mode: str,
    ablations: Sequence[str] = ("none",),
    max_samples: int | None = None,
    fixture_generations_path: Path | None = None,
    base_model_dir: Path | None = None,
    adapter_dir: Path | None = None,
    device: str = "auto",
    seed: int = 20260729,
    max_input_tokens: int = 2048,
    max_new_tokens: int = 384,
    blind_authorization_path: Path | None = None,
    blind_authorization_sha256: str | None = None,
    blind_run_id: str | None = None,
    selection_freeze_path: Path | None = None,
    selection_freeze_sha256: str | None = None,
    training_receipt_path: Path | None = None,
    runner_path: Path | None = None,
) -> dict[str, Any]:
    """Execute one integrity-bound v5 evaluation and write immutable evidence."""

    if backend_mode not in ALLOWED_BACKENDS:
        raise EvidenceEvalV5Error(f"unsupported backend mode: {backend_mode}")
    selected_ablations = _validate_ablation_list(ablations)
    output = Path(output_dir).resolve()
    protected_names = (
        "per_sample.v5.jsonl",
        "summary.v5.json",
        "run_receipt.v5.json",
    )
    if any((output / name).exists() for name in protected_names):
        raise EvidenceEvalV5Error("evaluation output already exists; use a new content-addressed directory")
    if split != "blind_test" and (
        blind_run_id is not None
        or blind_authorization_path is not None
        or blind_authorization_sha256 is not None
        or selection_freeze_path is not None
        or selection_freeze_sha256 is not None
        or training_receipt_path is not None
    ):
        raise BlindTestAuthorizationError(
            "blind authorization arguments are invalid for a non-blind evaluation"
        )

    code_before = _code_hashes(runner_path)
    consumption: BlindConsumptionV5 | None = None
    consumption_record: dict[str, Any] | None = None
    authorization: dict[str, Any] | None = None
    expected_base_sha: str | None = None
    expected_adapter_sha: str | None = None
    blind_decoding: dict[str, Any] | None = None

    if split == "blind_test":
        if backend_mode != "hf_model":
            raise BlindTestAuthorizationError("blind_test permits only the real hf_model backend")
        if selected_ablations != ("none",):
            raise BlindTestAuthorizationError("blind_test permits exactly ablations=[none]")
        if max_samples is not None:
            raise BlindTestAuthorizationError("blind_test max_samples must be null")
        if fixture_generations_path is not None:
            raise BlindTestAuthorizationError("blind_test cannot use fixture generations")
        if base_model_dir is None:
            raise BlindTestAuthorizationError("blind_test requires a concrete base model")
        if runner_path is None or set(code_before) != {"evaluator", "runner"}:
            raise BlindTestAuthorizationError("blind_test requires evaluator and runner source hashes")
        if blind_run_id is None:
            raise BlindTestAuthorizationError("blind_test requires --blind-run-id")
        if output.exists():
            raise BlindTestAuthorizationError("blind output directory must not already exist")
        blind_decoding = hf_decoding_contract(
            device=device,
            seed=seed,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
        )
        provisional = _blind_provisional_selection(dataset_dir)
        base_inventory = _model_inventory(base_model_dir)
        adapter_inventory = None if adapter_dir is None else _model_inventory(adapter_dir)
        subject = "base" if adapter_dir is None else "candidate"
        selection_evidence = _selection_evidence_binding(
            subject=subject,
            dataset_dir=provisional.dataset_dir,
            base_model_dir=base_model_dir,
            adapter_dir=adapter_dir,
            base_inventory=base_inventory,
            adapter_inventory=adapter_inventory,
            selection_freeze_path=selection_freeze_path,
            selection_freeze_sha256=selection_freeze_sha256,
            training_receipt_path=training_receipt_path,
        )
        binding = _blind_binding(
            selection=provisional,
            base_inventory=base_inventory,
            adapter_inventory=adapter_inventory,
            subject=subject,
            code=code_before,
            decoding=blind_decoding,
            selection_evidence=selection_evidence,
            run_id=blind_run_id,
            output_basename=output.name,
        )
        authorization = verify_blind_test_authorization(
            provisional,
            receipt_path=blind_authorization_path,
            expected_sha256=blind_authorization_sha256,
            expected_binding=binding,
        )
        assert authorization is not None
        consumption = _consume_blind_authorization(provisional, authorization)
        expected_base_sha = binding["model"]["base_model_tree_sha256"]
        expected_adapter_sha = binding["model"]["adapter_tree_sha256"]

    try:
        selection = load_dataset_selection(
            dataset_dir,
            split=split,
            max_samples=max_samples,
            blind_consumption=consumption,
        )
        if split == "blind_test" and len(selection.samples) != EXPECTED_BLIND_EXAMPLES:
            raise BlindTestAuthorizationError(
                f"blind_test must contain exactly {EXPECTED_BLIND_EXAMPLES} samples"
            )
        requests = tuple(
            request
            for ablation in selected_ablations
            for request in build_generation_requests(
                selection.samples,
                ablation=ablation,
            )
        )

        if backend_mode == "hf_model":
            if base_model_dir is None:
                raise EvidenceEvalV5Error("hf_model mode requires base_model_dir")
            if fixture_generations_path is not None:
                raise EvidenceEvalV5Error("hf_model mode cannot use fixture generations")
            generations, backend, traces = generate_hf_model(
                requests,
                base_model_dir=base_model_dir,
                adapter_dir=adapter_dir,
                device=device,
                seed=seed,
                max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens,
                expected_base_model_tree_sha256=expected_base_sha,
                expected_adapter_tree_sha256=expected_adapter_sha,
            )
        elif backend_mode == "deterministic_baseline":
            if any(value is not None for value in (fixture_generations_path, base_model_dir, adapter_dir)):
                raise EvidenceEvalV5Error("deterministic_baseline cannot use model or fixture artifacts")
            generations, backend = deterministic_baseline_generations(requests)
            traces = {}
        else:
            if fixture_generations_path is None:
                raise EvidenceEvalV5Error("fixture mode requires fixture_generations_path")
            if base_model_dir is not None or adapter_dir is not None:
                raise EvidenceEvalV5Error("fixture mode cannot use model artifacts")
            generations, backend = load_fixture_generations(
                fixture_generations_path,
                requests,
            )
            traces = {}

        rows, ablation_summaries = score_generations(
            selection=selection,
            requests=requests,
            generations=generations,
            backend=backend,
            traces=traces,
        )
        code_after = _code_hashes(runner_path)
        if code_after != code_before:
            raise EvidenceEvalV5Error("evaluation source changed during generation")
        if split == "blind_test":
            assert authorization is not None
            assert blind_decoding is not None
            if backend.get("decoding") != blind_decoding:
                raise BlindTestAuthorizationError(
                    "actual decoding metadata does not match blind authorization"
                )
            if backend.get("base_model", {}).get("content_sha256") != expected_base_sha:
                raise BlindTestAuthorizationError("actual base model does not match blind authorization")
            actual_adapter = backend.get("adapter")
            actual_adapter_sha = None if actual_adapter is None else actual_adapter.get("content_sha256")
            if actual_adapter_sha != expected_adapter_sha:
                raise BlindTestAuthorizationError("actual adapter does not match blind authorization")
            assert consumption is not None
            consumption_record = _finalize_blind_consumption(
                consumption,
                status="COMPLETED",
            )

        per_sample_payload = _jsonl_bytes(rows)
        per_sample_sha256 = sha256_bytes(per_sample_payload)
        dataset_record = {
            "manifest_path": str(selection.manifest_path),
            "manifest_sha256": selection.manifest_sha256,
            "split_path": str(selection.split_path),
            "split_sha256": selection.split_sha256,
        }
        generation_contract = {
            "assistant_target_visible_to_backend": False,
            "free_generation_executed": backend["free_generation_executed"],
            "ablations": list(selected_ablations),
            "seed": seed,
            "max_samples": max_samples,
            "decoding": backend.get("decoding"),
        }
        run_id = blind_run_id if blind_run_id is not None else "diagnostic-" + uuid.uuid4().hex
        run_contract = {
            "run_id": run_id,
            "output_basename": output.name,
            "split": split,
            "examples": len(selection.samples),
            "dataset": dataset_record,
            "blind_test_authorization": authorization,
            "blind_test_consumption": consumption_record,
            "backend": backend,
            "generation_contract": generation_contract,
            "code": code_before,
            "per_sample_sha256": per_sample_sha256,
        }
        run_contract_sha256 = sha256_bytes(canonical_json(run_contract).encode("utf-8"))
        summary = {
            "schema": SUMMARY_SCHEMA,
            "split": split,
            "examples": len(selection.samples),
            "ablations": list(selected_ablations),
            "dataset": dataset_record,
            "blind_test_authorization": authorization,
            "blind_test_consumption": consumption_record,
            "backend": backend,
            "model_quality_claim_allowed": backend["is_model"],
            "non_model_test_only": not backend["is_model"],
            "assistant_target_visible_to_backend": False,
            "summaries": ablation_summaries,
            "per_sample_sha256": per_sample_sha256,
            "run_contract_sha256": run_contract_sha256,
        }
        summary_payload = _json_bytes(summary)
        summary_sha256 = sha256_bytes(summary_payload)
        receipt_body = {
            "schema": RUN_RECEIPT_SCHEMA,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "status": "COMPLETED",
            "run_id": run_id,
            "output_basename": output.name,
            "split": split,
            "dataset": dataset_record,
            "blind_test_authorization": authorization,
            "blind_test_consumption": consumption_record,
            "backend": backend,
            "generation_contract": generation_contract,
            "run_contract": run_contract,
            "run_contract_sha256": run_contract_sha256,
            "artifacts": {
                protected_names[0]: {
                    "bytes": len(per_sample_payload),
                    "sha256": per_sample_sha256,
                    "records": len(rows),
                },
                protected_names[1]: {
                    "bytes": len(summary_payload),
                    "sha256": summary_sha256,
                },
            },
            "code": code_before,
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "network_used_by_evaluator": False,
            },
            "claim_boundary": backend["claim_boundary"],
        }
        receipt_body["receipt_payload_sha256"] = sha256_bytes(canonical_json(receipt_body).encode("utf-8"))
        receipt_payload = _json_bytes(receipt_body)
        receipt_sha256 = sha256_bytes(receipt_payload)

        output.mkdir(parents=True, exist_ok=False)
        per_sample_path = output / protected_names[0]
        summary_path = output / protected_names[1]
        receipt_path = output / protected_names[2]
        _atomic_write(per_sample_path, per_sample_payload)
        _atomic_write(summary_path, summary_payload)
        _atomic_write(receipt_path, receipt_payload)
        return {
            "output_dir": str(output),
            "paths": {
                "per_sample": str(per_sample_path),
                "summary": str(summary_path),
                "run_receipt": str(receipt_path),
            },
            "hashes": {
                "per_sample": per_sample_sha256,
                "summary": summary_sha256,
                "run_receipt": receipt_sha256,
            },
            "summary": summary,
            "receipt": receipt_body,
        }
    except BaseException as exc:
        if consumption is not None:
            try:
                _finalize_blind_consumption(
                    consumption,
                    status="FAILED_NON_REUSABLE",
                    error=exc,
                )
            except BaseException:
                # The exclusive pending marker still makes the authorization
                # non-reusable even if terminal receipt replacement fails.
                pass
        raise


__all__ = [
    "ALLOWED_ABLATIONS",
    "ALLOWED_BACKENDS",
    "ALLOWED_SPLITS",
    "ANSWER_SCHEMA",
    "AUTHORIZATION_SCHEMA",
    "AUTHORIZATION_VERSION",
    "BLIND_AUTHORIZATION_SCOPE",
    "CONSUMPTION_SCHEMA",
    "EXPECTED_BLIND_EXAMPLES",
    "BlindTestAuthorizationError",
    "DatasetSelectionV5",
    "EXAMPLE_SCHEMA",
    "EvidenceEvalV5Error",
    "EvidenceSampleV5",
    "GenerationRequestV5",
    "MANIFEST_SCHEMA",
    "build_blind_test_authorization",
    "build_generation_requests",
    "canonical_json",
    "deterministic_baseline_generations",
    "generate_hf_model",
    "hf_decoding_contract",
    "load_completed_blind_selection",
    "load_dataset_selection",
    "load_fixture_generations",
    "parse_single_json_object",
    "run_evaluation",
    "rescore_persisted_rows",
    "score_generations",
    "sha256_bytes",
    "sha256_file",
    "validate_student_answer",
    "verify_blind_test_authorization",
]
