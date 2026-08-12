"""Immutable non-blind canary gate for ICMat evidence SFT v5."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .evidence_eval_v5 import (
    AUTHORIZATION_SCHEMA,
    CONSUMPTION_SCHEMA,
    MANIFEST_SCHEMA,
    RUN_RECEIPT_SCHEMA,
    SAMPLE_RESULT_SCHEMA,
    SUMMARY_SCHEMA,
    EvidenceEvalV5Error,
    canonical_json,
    rescore_persisted_rows,
)

SCHEMA = "icmat_llm_nonblind_canary_gate_receipt.v5"
VERSION = "icmat-llm-nonblind-canary-gate-v5.1.0"
PASS_STATUS = "PASS_CANARY_GATES_BLIND_STILL_UNAUTHORIZED"
STOP_STATUS = "STOP_BEFORE_BLIND"
ERROR_STATUS = "CANARY_GATE_NOT_RECORDED"

EXPECTED_RUN_FILES = frozenset(
    {
        "per_sample.v5.jsonl",
        "run_receipt.v5.json",
        "summary.v5.json",
    }
)
THRESHOLDS = {
    "schema_valid_rate": 1.0,
    "answer_accuracy_min": 0.95,
    "unsupported_wrong_answer_count_max": 0,
    "citation_exact_min": 0.95,
    "provenance_exact_min": 0.95,
    "refusal_f1_min": 0.95,
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

CLAIM_BOUNDARY = (
    "This receipt independently verifies one completed non-blind validation "
    "canary and records whether its absolute gates pass. It never authorizes, "
    "opens, reads, or evaluates blind_test JSONL content. A passing receipt "
    "still requires a separate immutable blind authorization; a failing "
    "receipt requires STOP_BEFORE_BLIND."
)


class CanaryGateV5Error(RuntimeError):
    """Raised when a trustworthy canary receipt cannot be recorded."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanaryGateV5Error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise CanaryGateV5Error(f"non-finite JSON constant is forbidden: {value}")


def _stable_bytes(path: Path, *, label: str) -> tuple[Path, bytes]:
    raw = Path(path)
    if raw.is_symlink():
        raise CanaryGateV5Error(f"{label} must not be a symlink: {raw}")
    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError as exc:
        raise CanaryGateV5Error(f"{label} does not exist: {raw}") from exc
    if not resolved.is_file():
        raise CanaryGateV5Error(f"{label} must be a regular file: {resolved}")
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
        raise CanaryGateV5Error(f"{label} changed while it was read")
    return resolved, first


def _load_json_payload(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CanaryGateV5Error(f"{label} must contain valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise CanaryGateV5Error(f"{label} JSON root must be an object")
    return value


def _load_json_file(path: Path, *, label: str) -> tuple[Path, bytes, dict[str, Any]]:
    resolved, payload = _stable_bytes(path, label=label)
    return resolved, payload, _load_json_payload(payload, label=label)


def _read_rows(payload: bytes, *, label: str) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CanaryGateV5Error(f"{label} must be UTF-8 JSONL") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise CanaryGateV5Error(f"{label} contains blank line {line_number}")
        try:
            value = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=_reject_nonfinite_constant,
            )
        except json.JSONDecodeError as exc:
            raise CanaryGateV5Error(
                f"{label} line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise CanaryGateV5Error(f"{label} line {line_number} is not an object")
        if value.get("schema") != SAMPLE_RESULT_SCHEMA:
            raise CanaryGateV5Error(
                f"{label} line {line_number} has unsupported schema"
            )
        rows.append(value)
    if not rows:
        raise CanaryGateV5Error(f"{label} is empty")
    return rows


def _resolve_directory(path: Path, *, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise CanaryGateV5Error(f"{label} must not be a symlink: {raw}")
    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError as exc:
        raise CanaryGateV5Error(f"{label} does not exist: {raw}") from exc
    if not resolved.is_dir():
        raise CanaryGateV5Error(f"{label} must be a directory: {resolved}")
    return resolved


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CanaryGateV5Error(f"{label} must be an object")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CanaryGateV5Error(f"{label} must be a non-empty string")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CanaryGateV5Error(f"{label} must be a non-negative integer")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise CanaryGateV5Error(f"{label} must be a lowercase SHA-256")
    return value


def _safe_relative_path(value: Any, label: str) -> PurePosixPath:
    text = _require_string(value, label)
    if "\\" in text:
        raise CanaryGateV5Error(f"{label} must use POSIX separators")
    relative = PurePosixPath(text)
    if (
        relative.is_absolute()
        or text in {".", ".."}
        or ".." in relative.parts
        or relative.as_posix() != text
    ):
        raise CanaryGateV5Error(f"{label} is not a safe relative path")
    return relative


def _resolve_bound_child(root: Path, relative: PurePosixPath, *, label: str) -> Path:
    candidate = root.joinpath(*relative.parts)
    if candidate.is_symlink():
        raise CanaryGateV5Error(f"{label} must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise CanaryGateV5Error(f"{label} is missing or escapes the dataset") from exc
    if not resolved.is_file():
        raise CanaryGateV5Error(f"{label} must be a regular file: {resolved}")
    return resolved


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
                candidates.extend(
                    entry for entry in item if isinstance(entry, Mapping)
                )
        elif isinstance(container, list):
            candidates.extend(
                entry
                for entry in container
                if isinstance(entry, Mapping) and entry.get("split") == split
            )
    return candidates


def _validation_descriptor(manifest: Mapping[str, Any]) -> dict[str, Any]:
    parsed: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in _descriptor_candidates(manifest, "validation"):
        if raw.get("split", "validation") != "validation":
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
        parsed[(item["path"], item["sha256"])] = item
    if len(parsed) != 1:
        raise CanaryGateV5Error(
            "manifest must bind exactly one validation integrity descriptor"
        )
    descriptor = next(iter(parsed.values()))
    _safe_relative_path(descriptor["path"], "manifest.validation.path")
    _require_sha256(descriptor["sha256"], "manifest.validation.sha256")
    for field in ("bytes", "examples"):
        if descriptor[field] is not None:
            _require_nonnegative_int(
                descriptor[field],
                f"manifest.validation.{field}",
            )
    return descriptor


def _verify_receipt_self_hash(receipt: Mapping[str, Any]) -> None:
    expected = _require_sha256(
        receipt.get("receipt_payload_sha256"),
        "run_receipt.receipt_payload_sha256",
    )
    body = dict(receipt)
    body.pop("receipt_payload_sha256", None)
    actual = _sha256_bytes(canonical_json(body).encode("utf-8"))
    if actual != expected:
        raise CanaryGateV5Error("run receipt self hash is invalid")


def _verify_artifact_binding(
    record: Any,
    *,
    name: str,
    payload: bytes,
    records: int | None = None,
) -> None:
    binding = _require_mapping(record, f"run_receipt.artifacts.{name}")
    expected_sha = _require_sha256(
        binding.get("sha256"),
        f"run_receipt.artifacts.{name}.sha256",
    )
    if expected_sha != _sha256_bytes(payload):
        raise CanaryGateV5Error(f"{name} SHA-256 does not match run receipt")
    if binding.get("bytes") != len(payload):
        raise CanaryGateV5Error(f"{name} byte count does not match run receipt")
    if records is not None and binding.get("records") != records:
        raise CanaryGateV5Error(f"{name} record count does not match run receipt")


def _verify_run_contract(
    summary: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> str:
    contract = _require_mapping(receipt.get("run_contract"), "run_receipt.run_contract")
    actual = _sha256_bytes(canonical_json(contract).encode("utf-8"))
    expected = _require_sha256(
        receipt.get("run_contract_sha256"),
        "run_receipt.run_contract_sha256",
    )
    if actual != expected or summary.get("run_contract_sha256") != actual:
        raise CanaryGateV5Error("run contract hash binding is invalid")
    mirrored = {
        "run_id": receipt.get("run_id"),
        "output_basename": receipt.get("output_basename"),
        "split": receipt.get("split"),
        "examples": summary.get("examples"),
        "dataset": receipt.get("dataset"),
        "blind_test_authorization": receipt.get("blind_test_authorization"),
        "blind_test_consumption": receipt.get("blind_test_consumption"),
        "backend": receipt.get("backend"),
        "generation_contract": receipt.get("generation_contract"),
        "code": receipt.get("code"),
        "per_sample_sha256": summary.get("per_sample_sha256"),
    }
    if dict(contract) != mirrored:
        raise CanaryGateV5Error("run contract does not mirror summary and receipt")
    return actual


def _verify_code_inventory(value: Any) -> dict[str, dict[str, Any]]:
    code = _require_mapping(value, "run_receipt.code")
    if set(code) != {"evaluator", "runner"}:
        raise CanaryGateV5Error(
            "validation receipt must bind evaluator and runner source"
        )
    result: dict[str, dict[str, Any]] = {}
    for role in ("evaluator", "runner"):
        record = _require_mapping(code[role], f"run_receipt.code.{role}")
        source, payload = _stable_bytes(
            Path(_require_string(record.get("path"), f"run_receipt.code.{role}.path")),
            label=f"{role} source",
        )
        expected = _require_sha256(
            record.get("sha256"),
            f"run_receipt.code.{role}.sha256",
        )
        actual = _sha256_bytes(payload)
        if actual != expected:
            raise CanaryGateV5Error(
                f"{role} source hash no longer matches validation receipt"
            )
        result[role] = {
            "path": str(source),
            "bytes": len(payload),
            "sha256": actual,
        }
    return result


def _find_dataset_blind_state(dataset_dir: Path) -> dict[str, Any]:
    authorization_paths: set[str] = set()
    consumption_paths: set[str] = set()
    json_metadata_checked = 0
    before = sorted(
        path.relative_to(dataset_dir).as_posix()
        for path in dataset_dir.rglob("*")
    )
    for relative_text in before:
        path = dataset_dir.joinpath(*PurePosixPath(relative_text).parts)
        if path.is_symlink():
            raise CanaryGateV5Error(
                f"dataset contains a forbidden symlink: {relative_text}"
            )
        name = path.name.lower()
        if (
            "authorization" in name
            or name.endswith(".authorization.v5.json")
        ):
            authorization_paths.add(relative_text)
        if (
            ".blind_consumptions" in PurePosixPath(relative_text).parts
            or "consumption" in name
        ):
            consumption_paths.add(relative_text)
        if not path.is_file() or path.suffix.lower() != ".json":
            continue
        _, payload = _stable_bytes(
            path,
            label=f"dataset JSON metadata {relative_text}",
        )
        json_metadata_checked += 1
        try:
            value = _load_json_payload(
                payload,
                label=f"dataset JSON metadata {relative_text}",
            )
        except CanaryGateV5Error:
            if "authorization" in name or "consumption" in name:
                continue
            raise
        if value.get("schema") == AUTHORIZATION_SCHEMA:
            authorization_paths.add(relative_text)
        if value.get("schema") == CONSUMPTION_SCHEMA:
            consumption_paths.add(relative_text)
    after = sorted(
        path.relative_to(dataset_dir).as_posix()
        for path in dataset_dir.rglob("*")
    )
    if before != after:
        raise CanaryGateV5Error("dataset membership changed during blind-state scan")
    if authorization_paths or consumption_paths:
        details = {
            "authorization_paths": sorted(authorization_paths),
            "consumption_paths": sorted(consumption_paths),
        }
        raise CanaryGateV5Error(
            "dataset already contains blind authorization or consumption state: "
            + canonical_json(details)
        )
    return {
        "authorization_paths": [],
        "consumption_paths": [],
        "json_metadata_files_checked": json_metadata_checked,
        "blind_jsonl_content_read": False,
        "verified_no_authorization_or_consumption": True,
    }


def _verify_dataset(
    dataset_record: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path, manifest_payload, manifest = _load_json_file(
        Path(
            _require_string(
                dataset_record.get("manifest_path"),
                "dataset.manifest_path",
            )
        ),
        label="dataset manifest",
    )
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise CanaryGateV5Error("dataset manifest has unsupported schema")
    dataset_dir = _resolve_directory(manifest_path.parent, label="dataset directory")
    if manifest_path.parent != dataset_dir:
        raise CanaryGateV5Error("dataset manifest parent is not canonical")
    if _sha256_bytes(manifest_payload) != _require_sha256(
        dataset_record.get("manifest_sha256"),
        "dataset.manifest_sha256",
    ):
        raise CanaryGateV5Error("dataset manifest SHA-256 binding is invalid")

    blind_state = _find_dataset_blind_state(dataset_dir)
    descriptor = _validation_descriptor(manifest)
    relative = _safe_relative_path(
        descriptor["path"],
        "manifest.validation.path",
    )
    if any("blind" in part.lower() for part in relative.parts):
        raise CanaryGateV5Error(
            "validation descriptor must not reference a blind-named path"
        )
    validation_path = _resolve_bound_child(
        dataset_dir,
        relative,
        label="validation split",
    )
    declared_split_path = Path(
        _require_string(dataset_record.get("split_path"), "dataset.split_path")
    ).resolve(strict=True)
    if declared_split_path != validation_path:
        raise CanaryGateV5Error(
            "evaluation split path does not match manifest validation descriptor"
        )
    _, validation_payload = _stable_bytes(
        validation_path,
        label="validation split",
    )
    validation_sha = _sha256_bytes(validation_payload)
    expected_split_sha = _require_sha256(
        dataset_record.get("split_sha256"),
        "dataset.split_sha256",
    )
    if validation_sha != descriptor["sha256"] or validation_sha != expected_split_sha:
        raise CanaryGateV5Error("validation split SHA-256 binding is invalid")
    if descriptor["bytes"] is not None and descriptor["bytes"] != len(
        validation_payload
    ):
        raise CanaryGateV5Error("validation split byte count is invalid")
    return {
        "directory": str(dataset_dir),
        "manifest": {
            "path": str(manifest_path),
            "bytes": len(manifest_payload),
            "sha256": _sha256_bytes(manifest_payload),
        },
        "validation_split": {
            "path": str(validation_path),
            "bytes": len(validation_payload),
            "sha256": validation_sha,
            "manifest_examples": descriptor["examples"],
        },
        "blind_state": blind_state,
    }


def _metric_rate(
    metrics: Mapping[str, Any],
    name: str,
) -> tuple[float | None, int, int]:
    record = _require_mapping(metrics.get(name), f"metrics.{name}")
    numerator = _require_nonnegative_int(
        record.get("numerator"),
        f"metrics.{name}.numerator",
    )
    denominator = _require_nonnegative_int(
        record.get("denominator"),
        f"metrics.{name}.denominator",
    )
    rate = record.get("rate")
    expected_rate = numerator / denominator if denominator else None
    if rate is None:
        if expected_rate is not None:
            raise CanaryGateV5Error(f"metrics.{name}.rate is unexpectedly null")
        return None, numerator, denominator
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise CanaryGateV5Error(f"metrics.{name}.rate must be numeric or null")
    result = float(rate)
    if not math.isfinite(result) or expected_rate is None:
        raise CanaryGateV5Error(f"metrics.{name}.rate is invalid")
    if not math.isclose(result, expected_rate, rel_tol=0.0, abs_tol=1e-15):
        raise CanaryGateV5Error(f"metrics.{name}.rate does not match its counts")
    return result, numerator, denominator


def _absolute_gate(
    summaries: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    none = _require_mapping(summaries.get("none"), "summaries.none")
    metrics = _require_mapping(none.get("metrics"), "summaries.none.metrics")
    schema_rate, schema_num, schema_den = _metric_rate(metrics, "schema_valid")
    answer_rate, answer_num, answer_den = _metric_rate(
        metrics,
        "answer_accuracy",
    )
    wrong_rate, wrong_num, wrong_den = _metric_rate(
        metrics,
        "unsupported_wrong_answer_rate",
    )
    citation_rate, citation_num, citation_den = _metric_rate(
        metrics,
        "citation_exact",
    )
    provenance_rate, provenance_num, provenance_den = _metric_rate(
        metrics,
        "provenance_exact",
    )
    refuse = _require_mapping(none.get("refuse"), "summaries.none.refuse")
    refusal_f1 = refuse.get("f1")
    if (
        isinstance(refusal_f1, bool)
        or not isinstance(refusal_f1, (int, float))
        or not math.isfinite(float(refusal_f1))
    ):
        raise CanaryGateV5Error("summaries.none.refuse.f1 must be finite")
    refusal_f1_value = float(refusal_f1)

    values = {
        "schema_valid": {
            "numerator": schema_num,
            "denominator": schema_den,
            "rate": schema_rate,
        },
        "answer_accuracy": {
            "numerator": answer_num,
            "denominator": answer_den,
            "rate": answer_rate,
        },
        "unsupported_wrong_answer": {
            "count": wrong_num,
            "refuse_denominator": wrong_den,
            "rate": wrong_rate,
        },
        "citation_exact": {
            "numerator": citation_num,
            "denominator": citation_den,
            "rate": citation_rate,
        },
        "provenance_exact": {
            "numerator": provenance_num,
            "denominator": provenance_den,
            "rate": provenance_rate,
        },
        "refusal_f1": {
            "value": refusal_f1_value,
            "true_positive": refuse.get("true_positive"),
            "false_positive": refuse.get("false_positive"),
            "false_negative": refuse.get("false_negative"),
        },
    }
    checks = {
        "schema_valid_rate_equals_1_0": {
            "passed": schema_rate == THRESHOLDS["schema_valid_rate"],
            "actual": schema_rate,
            "required": "== 1.0",
        },
        "answer_accuracy_at_least_0_95": {
            "passed": answer_rate is not None
            and answer_rate >= THRESHOLDS["answer_accuracy_min"],
            "actual": answer_rate,
            "required": ">= 0.95",
        },
        "unsupported_wrong_answer_count_equals_0": {
            "passed": wrong_num
            <= THRESHOLDS["unsupported_wrong_answer_count_max"],
            "actual": wrong_num,
            "required": "== 0",
        },
        "citation_exact_at_least_0_95": {
            "passed": citation_rate is not None
            and citation_rate >= THRESHOLDS["citation_exact_min"],
            "actual": citation_rate,
            "required": ">= 0.95",
        },
        "provenance_exact_at_least_0_95": {
            "passed": provenance_rate is not None
            and provenance_rate >= THRESHOLDS["provenance_exact_min"],
            "actual": provenance_rate,
            "required": ">= 0.95",
        },
        "refusal_f1_at_least_0_95": {
            "passed": refusal_f1_value >= THRESHOLDS["refusal_f1_min"],
            "actual": refusal_f1_value,
            "required": ">= 0.95",
        },
    }
    all_pass = all(bool(check["passed"]) for check in checks.values())
    return values, checks, all_pass


def _verify_validation_run(
    validation_run_dir: Path,
) -> dict[str, Any]:
    run_dir = _resolve_directory(
        validation_run_dir,
        label="validation run directory",
    )
    entries = list(run_dir.iterdir())
    if any(entry.is_symlink() for entry in entries):
        raise CanaryGateV5Error("validation run directory contains a symlink")
    actual_names = {entry.name for entry in entries if entry.is_file()}
    non_files = sorted(entry.name for entry in entries if not entry.is_file())
    if non_files or actual_names != EXPECTED_RUN_FILES:
        raise CanaryGateV5Error(
            "validation run directory must contain exactly "
            f"{sorted(EXPECTED_RUN_FILES)}; files={sorted(actual_names)} "
            f"non_files={non_files}"
        )

    summary_path, summary_payload, summary = _load_json_file(
        run_dir / "summary.v5.json",
        label="validation summary",
    )
    receipt_path, receipt_payload, receipt = _load_json_file(
        run_dir / "run_receipt.v5.json",
        label="validation run receipt",
    )

    # The split is verified before per_sample.v5.jsonl is opened. This makes a
    # mistakenly supplied blind run fail without reading blind sample content.
    if summary.get("schema") != SUMMARY_SCHEMA:
        raise CanaryGateV5Error("validation summary has unsupported schema")
    if receipt.get("schema") != RUN_RECEIPT_SCHEMA:
        raise CanaryGateV5Error("validation receipt has unsupported schema")
    if summary.get("split") != "validation" or receipt.get("split") != "validation":
        raise CanaryGateV5Error(
            "canary gate accepts only non-blind split=validation"
        )
    if receipt.get("status") != "COMPLETED":
        raise CanaryGateV5Error("validation receipt is not completed")
    if receipt.get("output_basename") != run_dir.name:
        raise CanaryGateV5Error("validation output basename binding is invalid")
    for owner, value in (("summary", summary), ("receipt", receipt)):
        if (
            value.get("blind_test_authorization") is not None
            or value.get("blind_test_consumption") is not None
        ):
            raise CanaryGateV5Error(
                f"{owner} unexpectedly carries blind authorization or consumption"
            )
    if summary.get("ablations") != ["none"]:
        raise CanaryGateV5Error(
            "canary gate requires one none-only validation run"
        )

    _verify_receipt_self_hash(receipt)
    run_contract_sha = _verify_run_contract(summary, receipt)
    code_inventory = _verify_code_inventory(receipt.get("code"))
    dataset_record = _require_mapping(summary.get("dataset"), "summary.dataset")
    if dict(dataset_record) != receipt.get("dataset"):
        raise CanaryGateV5Error("summary and receipt dataset bindings differ")
    dataset = _verify_dataset(dataset_record)

    output_candidate = run_dir / "per_sample.v5.jsonl"
    samples_path, samples_payload = _stable_bytes(
        output_candidate,
        label="validation per-sample artifact",
    )
    rows = _read_rows(samples_payload, label="validation per-sample artifact")
    if any(row.get("split") != "validation" for row in rows):
        raise CanaryGateV5Error("per-sample rows are not validation rows")
    if any(row.get("ablation") != "none" for row in rows):
        raise CanaryGateV5Error("per-sample rows are not none-only")
    keys = {
        (str(row.get("example_id")), str(row.get("ablation")))
        for row in rows
    }
    if len(keys) != len(rows):
        raise CanaryGateV5Error("validation run contains duplicate sample keys")
    example_ids = {str(row.get("example_id")) for row in rows}
    if summary.get("examples") != len(example_ids):
        raise CanaryGateV5Error("validation example count is inconsistent")

    artifacts = _require_mapping(receipt.get("artifacts"), "run_receipt.artifacts")
    if set(artifacts) != {"per_sample.v5.jsonl", "summary.v5.json"}:
        raise CanaryGateV5Error("validation receipt artifact membership is invalid")
    _verify_artifact_binding(
        artifacts.get("per_sample.v5.jsonl"),
        name="per_sample.v5.jsonl",
        payload=samples_payload,
        records=len(rows),
    )
    _verify_artifact_binding(
        artifacts.get("summary.v5.json"),
        name="summary.v5.json",
        payload=summary_payload,
    )
    samples_sha = _sha256_bytes(samples_payload)
    if summary.get("per_sample_sha256") != samples_sha:
        raise CanaryGateV5Error("summary does not bind per-sample SHA-256")

    backend = _require_mapping(summary.get("backend"), "summary.backend")
    if dict(backend) != receipt.get("backend"):
        raise CanaryGateV5Error("summary and receipt backend bindings differ")
    generation_contract = _require_mapping(
        receipt.get("generation_contract"),
        "run_receipt.generation_contract",
    )
    if (
        backend.get("mode") != "hf_model"
        or backend.get("is_model") is not True
        or backend.get("free_generation_executed") is not True
        or summary.get("model_quality_claim_allowed") is not True
        or summary.get("assistant_target_visible_to_backend") is not False
        or generation_contract.get("assistant_target_visible_to_backend")
        is not False
    ):
        raise CanaryGateV5Error(
            "canary gate requires target-free HF model generation"
        )

    try:
        rescored_rows, rescored_summaries = rescore_persisted_rows(
            rows,
            backend=backend,
        )
    except EvidenceEvalV5Error as exc:
        raise CanaryGateV5Error(
            "validation rows cannot be independently rescored"
        ) from exc
    if [canonical_json(row) for row in rows] != [
        canonical_json(row) for row in rescored_rows
    ]:
        raise CanaryGateV5Error(
            "persisted row metrics differ from independent rescoring"
        )
    if summary.get("summaries") != rescored_summaries:
        raise CanaryGateV5Error(
            "persisted summary metrics differ from independent rescoring"
        )

    metric_values, checks, all_pass = _absolute_gate(rescored_summaries)
    run_files = {
        path.name: {
            "path": str(path),
            "bytes": len(payload),
            "sha256": _sha256_bytes(payload),
        }
        for path, payload in (
            (samples_path, samples_payload),
            (summary_path, summary_payload),
            (receipt_path, receipt_payload),
        )
    }
    after_names = {entry.name for entry in run_dir.iterdir() if entry.is_file()}
    if after_names != actual_names:
        raise CanaryGateV5Error(
            "validation run membership changed during verification"
        )
    return {
        "directory": str(run_dir),
        "run_id": receipt.get("run_id"),
        "output_basename": run_dir.name,
        "split": "validation",
        "examples": len(example_ids),
        "ablations": ["none"],
        "run_contract_sha256": run_contract_sha,
        "files": {name: run_files[name] for name in sorted(run_files)},
        "code": code_inventory,
        "dataset": dataset,
        "metrics": metric_values,
        "checks": checks,
        "all_absolute_gates_pass": all_pass,
    }


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _exclusive_write(path: Path, payload: bytes) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(output):
        raise CanaryGateV5Error(f"output already exists: {output}")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
    )
    try:
        descriptor = os.open(output, flags, 0o600)
    except FileExistsError as exc:
        raise CanaryGateV5Error(f"output already exists: {output}") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # Retain a partial exclusive file so the failed path cannot be reused.
        raise
    return output.resolve(strict=True)


def _implementation_inventory(runner_path: Path | None) -> dict[str, Any]:
    paths = {"gate_module": Path(__file__).resolve()}
    if runner_path is not None:
        paths["runner"] = Path(runner_path).resolve(strict=True)
    result: dict[str, Any] = {}
    for role, path in paths.items():
        resolved, payload = _stable_bytes(path, label=f"canary {role}")
        result[role] = {
            "path": str(resolved),
            "bytes": len(payload),
            "sha256": _sha256_bytes(payload),
        }
    return result


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def record_canary_gate(
    *,
    validation_run_dir: Path,
    output_path: Path,
    runner_path: Path | None = None,
) -> dict[str, Any]:
    """Verify one validation canary and write an immutable gate receipt."""

    output = Path(output_path).resolve(strict=False)
    if os.path.lexists(output):
        raise CanaryGateV5Error(f"output already exists: {output}")
    verification = _verify_validation_run(validation_run_dir)
    run_dir = Path(verification["directory"])
    dataset_dir = Path(verification["dataset"]["directory"])
    if _path_is_within(output, run_dir) or _path_is_within(output, dataset_dir):
        raise CanaryGateV5Error(
            "output must be outside the immutable validation run and dataset"
        )
    blind_state_recheck = _find_dataset_blind_state(dataset_dir)
    if blind_state_recheck != verification["dataset"]["blind_state"]:
        raise CanaryGateV5Error(
            "dataset blind state changed before exclusive receipt creation"
        )

    passed = bool(verification["all_absolute_gates_pass"])
    status = PASS_STATUS if passed else STOP_STATUS
    body = {
        "schema": SCHEMA,
        "version": VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "gate_passed": passed,
        "blind_test_permitted_by_this_receipt": False,
        "next_action": (
            "SEPARATE_IMMUTABLE_BLIND_AUTHORIZATION_REVIEW_REQUIRED"
            if passed
            else STOP_STATUS
        ),
        "thresholds": THRESHOLDS,
        "claim_boundary": CLAIM_BOUNDARY,
        "implementation": _implementation_inventory(runner_path),
        "validation": verification,
    }
    receipt = {
        **body,
        "canonical_digest_sha256": _sha256_bytes(
            canonical_json(body).encode("utf-8")
        ),
    }
    receipt["receipt_payload_sha256"] = _sha256_bytes(
        canonical_json(receipt).encode("utf-8")
    )
    payload = _json_bytes(receipt)
    written = _exclusive_write(output, payload)
    persisted = written.read_bytes()
    if persisted != payload:
        raise CanaryGateV5Error("persisted canary receipt bytes differ")
    return {
        "status": status,
        "gate_passed": passed,
        "path": str(written),
        "sha256": _sha256_bytes(payload),
        "canonical_digest_sha256": receipt["canonical_digest_sha256"],
        "receipt": receipt,
    }


__all__ = [
    "CLAIM_BOUNDARY",
    "ERROR_STATUS",
    "PASS_STATUS",
    "SCHEMA",
    "STOP_STATUS",
    "THRESHOLDS",
    "VERSION",
    "CanaryGateV5Error",
    "record_canary_gate",
]
