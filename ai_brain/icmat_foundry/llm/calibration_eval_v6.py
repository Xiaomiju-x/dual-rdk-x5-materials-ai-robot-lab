"""Post-freeze, non-blind calibration and split-conformal audit for v6.

This module deliberately composes the existing v6 pointer evaluator instead
of implementing a second inference path.  A valid selection-freeze receipt is
checked before ``calibration.jsonl`` is opened.  Exactly 150 calibration rows
are then generated, compiled, and scored under the frozen singleton-greedy
contract.

The conformal score is an integer count of ten auditable inconsistency flags.
It is not a probability, confidence score, or model logit.  The reported
threshold is the finite-sample corrected order statistic
``ceil((n + 1) * (1 - alpha))`` at fixed ``alpha=0.10``.  Its future-sample
interpretation requires exchangeability; calibration inclusion coverage is
reported separately and is not presented as blind or production coverage.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from icmat_foundry.llm import (
    evidence_pointer_v6,
    pointer_checkpoint_eval_v6,
    pointer_hf_eval_v6,
    selection_freeze_v6,
)

CALIBRATION_EVALUATOR_VERSION = "icmat-calibration-evaluator-v6.1.0"
SELECTION_FREEZE_SCHEMA = selection_freeze_v6.SCHEMA
SELECTION_FREEZE_STATUS = selection_freeze_v6.STATUS
PER_SAMPLE_SCHEMA = "icmat_llm_calibration_sample.v6"
SUMMARY_SCHEMA = "icmat_llm_calibration_summary.v6"
RECEIPT_SCHEMA = "icmat_llm_calibration_receipt.v6"

EXPECTED_CALIBRATION_ROWS = 150
FIXED_ALPHA = Decimal("0.10")
FIXED_NOMINAL_COVERAGE = Decimal("0.90")
FIXED_INFERENCE_CONTRACT = {
    "max_input_tokens": pointer_hf_eval_v6.MAX_INPUT_TOKENS,
    "max_new_tokens": pointer_hf_eval_v6.MAX_NEW_TOKENS,
    "decoding": "greedy",
    "do_sample": False,
    "singleton": True,
    "batch_size": 1,
    "seed": 20260729,
    "messages_visible_to_generation": ["system", "user"],
    "compiler_after_generation": True,
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NONCONFORMITY_COMPONENTS = (
    "untrusted_finish",
    "pointer_parse_invalid",
    "pointer_task_mismatch",
    "pointer_decision_mismatch",
    "pointer_span_mismatch",
    "compiler_or_schema_failure",
    "compiled_citation_mismatch",
    "compiled_provenance_mismatch",
    "compiled_strict_mismatch",
    "unsupported_wrong_answer",
)
_POINTER_METRICS = (
    "parse_valid",
    "task_exact",
    "decision_exact",
    "span_exact",
    "value_exact",
    "strict_exact",
    "compiler_accepted",
)
_COMPILED_METRICS = (
    "json_available",
    "schema_valid",
    "schema_exact",
    "decision_exact",
    "task_exact",
    "claim_exact",
    "verdict_exact",
    "citation_exact",
    "provenance_exact",
    "strict_exact",
)


class CalibrationEvalV6Error(ValueError):
    """Raised when calibration cannot produce trustworthy immutable evidence."""


def canonical_json(value: Any) -> str:
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


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (canonical_json(dict(record)) + "\n").encode("utf-8")
        for record in records
    )


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CalibrationEvalV6Error(f"duplicate JSON key rejected: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(token: str) -> None:
    raise CalibrationEvalV6Error(f"non-finite JSON value rejected: {token}")


def _reject_blind_label(path: Path, *, field: str) -> None:
    if any("blind" in part.casefold() for part in Path(path).parts):
        raise CalibrationEvalV6Error(
            f"{field} must not reference a blind-labelled path"
        )


def _stable_bytes(path: Path, *, field: str) -> tuple[Path, bytes]:
    raw = Path(path)
    _reject_blind_label(raw, field=field)
    if raw.is_symlink():
        raise CalibrationEvalV6Error(f"{field} must not be a symlink")
    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError as exc:
        raise CalibrationEvalV6Error(f"{field} is unavailable: {raw}") from exc
    if not resolved.is_file():
        raise CalibrationEvalV6Error(f"{field} must be a regular file")
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
        raise CalibrationEvalV6Error(f"{field} changed while it was read")
    return resolved, first


def _load_json(path: Path, *, field: str) -> tuple[Path, bytes, dict[str, Any]]:
    resolved, payload = _stable_bytes(path, field=field)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalibrationEvalV6Error(
            f"{field} must be strict UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise CalibrationEvalV6Error(f"{field} root must be an object")
    return resolved, payload, value


def _load_jsonl(path: Path, *, field: str) -> tuple[Path, bytes, list[dict[str, Any]]]:
    resolved, payload = _stable_bytes(path, field=field)
    rows: list[dict[str, Any]] = []
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CalibrationEvalV6Error(f"{field} must be UTF-8") from exc
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line:
            raise CalibrationEvalV6Error(
                f"{field} contains blank line {line_number}"
            )
        try:
            value = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=_reject_nonfinite_constant,
            )
        except json.JSONDecodeError as exc:
            raise CalibrationEvalV6Error(
                f"{field} line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise CalibrationEvalV6Error(
                f"{field} line {line_number} must be an object"
            )
        rows.append(value)
    if not rows:
        raise CalibrationEvalV6Error(f"{field} contains no rows")
    return resolved, payload, rows


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CalibrationEvalV6Error(f"{field} must be an object")
    return value


def _require_bool(value: Any, expected: bool, *, field: str) -> None:
    if value is not expected:
        raise CalibrationEvalV6Error(f"{field} must be {expected}")


def _require_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise CalibrationEvalV6Error(f"{field} must be a non-empty trimmed string")
    return value


def _require_int(
    value: Any,
    *,
    field: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CalibrationEvalV6Error(f"{field} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise CalibrationEvalV6Error(f"{field} is outside the allowed range")
    return value


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CalibrationEvalV6Error(f"{field} must be a lowercase SHA-256")
    return value


def _tree_inventory(
    path: Path,
    *,
    field: str,
    selected_names: frozenset[str] | None = None,
    casefold_order: bool = False,
) -> dict[str, Any]:
    raw = Path(path)
    _reject_blind_label(raw, field=field)
    if raw.is_symlink():
        raise CalibrationEvalV6Error(f"{field} must not be a symlink")
    try:
        root = raw.resolve(strict=True)
    except FileNotFoundError as exc:
        raise CalibrationEvalV6Error(f"{field} is unavailable") from exc
    if not root.is_dir():
        raise CalibrationEvalV6Error(f"{field} must be a directory")
    files: list[dict[str, Any]] = []
    candidates = list(root.rglob("*"))
    candidates.sort(
        key=(
            (lambda item: (
                item.relative_to(root).as_posix().casefold(),
                item.relative_to(root).as_posix(),
            ))
            if casefold_order
            else (lambda item: item.relative_to(root).as_posix())
        )
    )
    casefold_paths: set[str] = set()
    for candidate in candidates:
        if candidate.is_symlink():
            raise CalibrationEvalV6Error(
                f"{field} contains a forbidden symlink: {candidate}"
            )
        if candidate.is_file():
            if selected_names is not None and candidate.name not in selected_names:
                continue
            relative = candidate.relative_to(root).as_posix()
            folded = relative.casefold()
            if folded in casefold_paths:
                raise CalibrationEvalV6Error(
                    f"{field} contains Windows-ambiguous paths"
                )
            casefold_paths.add(folded)
            before = candidate.stat()
            digest = sha256_file(candidate)
            after = candidate.stat()
            if (
                before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
            ):
                raise CalibrationEvalV6Error(
                    f"{field} changed while hashing: {candidate}"
                )
            files.append(
                {
                    "path": relative,
                    "bytes": after.st_size,
                    "sha256": digest,
                }
            )
    if not files:
        raise CalibrationEvalV6Error(f"{field} tree is empty")
    return {
        "path": str(root),
        "files": files,
        "files_count": len(files),
        "bytes": sum(item["bytes"] for item in files),
        "tree_sha256": sha256_bytes(canonical_json(files).encode("utf-8")),
        "ordering": (
            "windows_casefold_then_posix"
            if casefold_order
            else "canonical_posix_case_sensitive"
        ),
    }


def _require_adapter_inventory(value: Any, *, field: str) -> dict[str, Any]:
    inventory = _require_mapping(value, field=field)
    files = inventory.get("files")
    if not isinstance(files, list):
        raise CalibrationEvalV6Error(f"{field}.files must be an array")
    names = [Path(str(item.get("path"))).name for item in files if isinstance(item, Mapping)]
    if (
        len(names) != 2
        or "adapter_config.json" not in names
        or sum(
            name in {"adapter_model.safetensors", "adapter_model.bin"}
            for name in names
        )
        != 1
    ):
        raise CalibrationEvalV6Error(
            f"{field} must contain adapter_config.json and exactly one adapter model"
        )
    return {
        "path": _require_text(inventory.get("path"), field=f"{field}.path"),
        "files": [dict(item) for item in files],
        "file_count": _require_int(
            inventory.get("file_count"),
            field=f"{field}.file_count",
            minimum=2,
            maximum=2,
        ),
        "bytes": _require_int(
            inventory.get("bytes"),
            field=f"{field}.bytes",
            minimum=1,
        ),
        "tree_sha256": _require_sha256(
            inventory.get("tree_sha256"),
            field=f"{field}.tree_sha256",
        ),
    }


def _selection_verification_inputs(
    value: Any,
    *,
    dataset_dir: Path,
) -> dict[str, Path]:
    """Extract only authoritative paths needed for the producer's verifier."""

    root = _require_mapping(value, field="selection freeze")
    if root.get("schema") != SELECTION_FREEZE_SCHEMA:
        raise CalibrationEvalV6Error("selection freeze schema is invalid")
    if root.get("status") != SELECTION_FREEZE_STATUS:
        raise CalibrationEvalV6Error("selection freeze status is not calibration-ready")
    if (
        root.get("selection_locked") is not True
        or root.get("calibration_authorized") is not True
        or root.get("blind_test_authorized") is not False
        or root.get("deployment_authorized") is not False
        or root.get("post_freeze_policy") != selection_freeze_v6.POST_FREEZE_POLICY
    ):
        raise CalibrationEvalV6Error(
            "selection freeze top-level calibration authorization is invalid"
        )
    evaluation = _require_mapping(
        root.get("evaluation_index"),
        field="selection freeze.evaluation_index",
    )
    training = _require_mapping(
        root.get("training_receipt"),
        field="selection freeze.training_receipt",
    )
    dataset = _require_mapping(
        root.get("dataset"),
        field="selection freeze.dataset",
    )
    base_model = _require_mapping(
        root.get("base_model"),
        field="selection freeze.base_model",
    )
    explicit_dataset = Path(dataset_dir).resolve(strict=True)
    recorded_dataset = Path(
        _require_text(dataset.get("path"), field="selection freeze.dataset.path")
    ).resolve(strict=True)
    if explicit_dataset != recorded_dataset:
        raise CalibrationEvalV6Error(
            "explicit dataset directory differs from the selection freeze"
        )
    return {
        "evaluation_index_path": Path(
            _require_text(
                evaluation.get("path"),
                field="selection freeze.evaluation_index.path",
            )
        ),
        "training_receipt_path": Path(
            _require_text(
                training.get("path"),
                field="selection freeze.training_receipt.path",
            )
        ),
        "dataset_dir": recorded_dataset,
        "base_model_dir": Path(
            _require_text(
                base_model.get("path"),
                field="selection freeze.base_model.path",
            )
        ),
    }


def validate_selection_freeze(
    value: Any,
    *,
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize only a producer receipt already independently recomputed."""

    root = _require_mapping(value, field="selection freeze")
    if root.get("schema") != SELECTION_FREEZE_SCHEMA:
        raise CalibrationEvalV6Error("selection freeze schema is invalid")
    if root.get("status") != SELECTION_FREEZE_STATUS:
        raise CalibrationEvalV6Error("selection freeze status is not calibration-ready")

    if verification.get("status") != selection_freeze_v6.VERIFIED_STATUS:
        raise CalibrationEvalV6Error(
            "selection freeze lacks authoritative recomputation verification"
        )
    if (
        verification.get("calibration_authorized") is not True
        or verification.get("blind_test_authorized") is not False
        or verification.get("deployment_authorized") is not False
        or verification.get("selection_locked") is not True
    ):
        raise CalibrationEvalV6Error(
            "selection freeze verification authorization is invalid"
        )

    selection = _require_mapping(root.get("selection"), field="selection")
    checkpoint = _require_mapping(
        selection.get("checkpoint"),
        field="selection.checkpoint",
    )
    adapter = _require_adapter_inventory(
        selection.get("adapter"),
        field="selection.adapter",
    )
    selection_normalized = {
        "checkpoint_id": _require_text(
            selection.get("checkpoint_id"),
            field="selection.checkpoint_id",
        ),
        "seed": _require_int(
            selection.get("seed"),
            field="selection.seed",
            minimum=1,
        ),
        "epoch": _require_int(
            selection.get("epoch"),
            field="selection.epoch",
            minimum=1,
            maximum=6,
        ),
        "checkpoint": {
            "path": _require_text(
                checkpoint.get("path"),
                field="selection.checkpoint.path",
            ),
            "tree_sha256": _require_sha256(
                checkpoint.get("tree_sha256"),
                field="selection.checkpoint.tree_sha256",
            ),
        },
        "adapter": adapter,
        "evaluator_checkpoint_tree_sha256": _require_sha256(
            selection.get("evaluator_adapter_tree_sha256"),
            field="selection.evaluator_adapter_tree_sha256",
        ),
    }
    if (
        Path(selection_normalized["checkpoint"]["path"]).resolve(strict=True)
        != Path(selection_normalized["adapter"]["path"]).resolve(strict=True)
    ):
        raise CalibrationEvalV6Error(
            "frozen adapter inventory is not rooted at the selected checkpoint"
        )
    if (
        verification.get("selected_checkpoint_id")
        != selection_normalized["checkpoint_id"]
        or verification.get("selected_seed") != selection_normalized["seed"]
        or verification.get("selected_epoch") != selection_normalized["epoch"]
        or verification.get("selected_checkpoint_tree_sha256")
        != selection_normalized["checkpoint"]["tree_sha256"]
        or Path(str(verification.get("selected_checkpoint_path"))).resolve(
            strict=True
        )
        != Path(selection_normalized["checkpoint"]["path"]).resolve(strict=True)
        or verification.get("selected_adapter_tree_sha256")
        != selection_normalized["adapter"]["tree_sha256"]
        or Path(str(verification.get("selected_adapter_path"))).resolve(strict=True)
        != Path(selection_normalized["adapter"]["path"]).resolve(strict=True)
        or verification.get("selected_evaluator_checkpoint_tree_sha256")
        != selection_normalized["evaluator_checkpoint_tree_sha256"]
    ):
        raise CalibrationEvalV6Error(
            "selection freeze normalized model differs from verifier output"
        )

    dataset = _require_mapping(root.get("dataset"), field="dataset")
    manifest = _require_mapping(dataset.get("manifest"), field="dataset.manifest")
    declarations = _require_mapping(
        dataset.get("declaration_only_splits"),
        field="dataset.declaration_only_splits",
    )
    calibration = _require_mapping(
        declarations.get("calibration"),
        field="dataset.declaration_only_splits.calibration",
    )
    if calibration.get("path") != "calibration.jsonl":
        raise CalibrationEvalV6Error(
            "dataset.calibration.path must be calibration.jsonl"
        )
    dataset_normalized = {
        "path": _require_text(dataset.get("path"), field="dataset.path"),
        "manifest_sha256": _require_sha256(
            manifest.get("sha256"),
            field="dataset.manifest.sha256",
        ),
        "calibration": {
            "path": "calibration.jsonl",
            "sha256": _require_sha256(
                calibration.get("manifest_declared_sha256"),
                field="dataset.declaration_only_splits.calibration."
                "manifest_declared_sha256",
            ),
            "count": _require_int(
                calibration.get("examples"),
                field="dataset.declaration_only_splits.calibration.examples",
                minimum=EXPECTED_CALIBRATION_ROWS,
                maximum=EXPECTED_CALIBRATION_ROWS,
            ),
        },
    }
    for field in (
        "calibration_content_read",
        "calibration_content_hashed",
        "blind_test_content_read",
        "blind_test_content_hashed",
    ):
        _require_bool(dataset.get(field), False, field=f"dataset.{field}")
    _require_bool(
        calibration.get("content_read_by_freeze"),
        False,
        field="dataset.declaration_only_splits.calibration.content_read_by_freeze",
    )
    _require_bool(
        calibration.get("content_hashed_by_freeze"),
        False,
        field="dataset.declaration_only_splits.calibration.content_hashed_by_freeze",
    )
    _require_bool(
        calibration.get("authorized"),
        True,
        field="dataset.declaration_only_splits.calibration.authorized",
    )

    runtime = _require_mapping(root.get("runtime_contract"), field="runtime_contract")
    decoding = _require_mapping(
        runtime.get("decoding"),
        field="runtime_contract.decoding",
    )
    expected_decoding = {
        "algorithm": "greedy",
        "do_sample": False,
        "num_beams": 1,
        "singleton": True,
        "batch_size": 1,
        "seed": FIXED_INFERENCE_CONTRACT["seed"],
        "max_input_tokens": FIXED_INFERENCE_CONTRACT["max_input_tokens"],
        "max_new_tokens": FIXED_INFERENCE_CONTRACT["max_new_tokens"],
    }
    for field, expected in expected_decoding.items():
        if decoding.get(field) != expected:
            raise CalibrationEvalV6Error(
                "selection freeze runtime_contract.decoding is not fixed v6"
            )
    if (
        runtime.get("model_role") != "evidence_pointer_model"
        or runtime.get("hidden_router") is not False
        or runtime.get("researcher_selects_model_and_task_explicitly") is not True
    ):
        raise CalibrationEvalV6Error(
            "selection freeze runtime contract changes the explicit model role"
        )

    base_model = _require_mapping(root.get("base_model"), field="base_model")
    base_normalized = {
        "path": _require_text(base_model.get("path"), field="base_model.path"),
        "training_tree_sha256": _require_sha256(
            base_model.get("training_tree_sha256"),
            field="base_model.training_tree_sha256",
        ),
        "evaluator_tree_sha256": _require_sha256(
            base_model.get("evaluator_tree_sha256"),
            field="base_model.evaluator_tree_sha256",
        ),
    }
    if (
        verification.get("dataset_manifest_sha256")
        != dataset_normalized["manifest_sha256"]
        or verification.get("base_model_tree_sha256")
        != base_normalized["training_tree_sha256"]
    ):
        raise CalibrationEvalV6Error(
            "selection freeze normalized data/model differs from verifier output"
        )
    authorization = _require_mapping(
        root.get("authorization"),
        field="authorization",
    )
    _require_bool(
        authorization.get("calibration_authorized"),
        True,
        field="authorization.calibration_authorized",
    )
    _require_bool(
        authorization.get("model_authorized_for_calibration"),
        True,
        field="authorization.model_authorized_for_calibration",
    )
    _require_bool(
        authorization.get("calibration_complete_split_only"),
        True,
        field="authorization.calibration_complete_split_only",
    )
    if authorization.get("calibration_expected_rows") != EXPECTED_CALIBRATION_ROWS:
        raise CalibrationEvalV6Error(
            "authorization.calibration_expected_rows must be 150"
        )
    _require_bool(
        authorization.get("blind_test_authorized"),
        False,
        field="authorization.blind_test_authorized",
    )
    _require_bool(
        authorization.get("checkpoint_reselection_allowed"),
        False,
        field="authorization.checkpoint_reselection_allowed",
    )
    return {
        "schema": SELECTION_FREEZE_SCHEMA,
        "status": SELECTION_FREEZE_STATUS,
        "selection": selection_normalized,
        "dataset": dataset_normalized,
        "base_model": base_normalized,
        "inference_contract": dict(FIXED_INFERENCE_CONTRACT),
        "access": {
            "calibration_content_read_before_freeze": False,
            "blind_content_read": False,
        },
        "authorization": {
            "calibration_authorized": True,
            "blind_test_authorized": False,
            "checkpoint_reselection_allowed": False,
        },
        "verification": dict(verification),
    }


def _metric(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _refusal_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    answer_rows = [
        row
        for row in rows
        if row["expected"]["answer"]["decision"] == "ANSWER"
    ]
    refuse_rows = [
        row
        for row in rows
        if row["expected"]["answer"]["decision"] == "REFUSE"
    ]

    def predicted(row: Mapping[str, Any]) -> str | None:
        if not row["pointer_metrics"]["compiler_accepted"]:
            return None
        parsed = row["compilation"].get("parsed_pointer")
        return parsed.get("decision") if isinstance(parsed, Mapping) else None

    true_positive = sum(predicted(row) == "REFUSE" for row in refuse_rows)
    false_positive = sum(predicted(row) == "REFUSE" for row in answer_rows)
    false_negative = len(refuse_rows) - true_positive
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    f1_denominator = 2 * true_positive + false_positive + false_negative
    return {
        "population": {
            "answer": len(answer_rows),
            "refuse": len(refuse_rows),
        },
        "confusion": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        },
        "precision": _metric(true_positive, precision_denominator),
        "recall": _metric(true_positive, recall_denominator),
        "f1": _metric(2 * true_positive, f1_denominator),
    }


def _truthfulness_record(row: Mapping[str, Any]) -> dict[str, Any]:
    compilation = _require_mapping(row.get("compilation"), field="compilation")
    compiled = compilation.get("compiled_answer")
    compiled_mapping = compiled if isinstance(compiled, Mapping) else None
    trace = compilation.get("contract_trace")
    trace_mapping = trace if isinstance(trace, Mapping) else {}
    accepted = compilation.get("status") == "COMPILED"
    decision = (
        compiled_mapping.get("decision")
        if compiled_mapping is not None
        else None
    )
    answer_source_verified = bool(
        accepted
        and decision == "ANSWER"
        and isinstance(compilation.get("selected_span_id"), str)
        and isinstance(compilation.get("selected_evidence_id"), str)
        and trace_mapping.get("claim_source")
        == (
            f"evidence[{compilation.get('selected_span_id')}].text_verbatim"
        )
        and trace_mapping.get("provenance_source")
        == (
            "evidence["
            f"{compilation.get('selected_evidence_id')}"
            "].provenance_validated"
        )
    )
    refusal_contract_verified = bool(
        accepted
        and decision == "REFUSE"
        and compiled_mapping is not None
        and compiled_mapping.get("claim") == ""
        and compiled_mapping.get("evidence_ids") == []
        and compiled_mapping.get("verdict") == "REFUSED"
    )
    fail_closed_safe_refusal = bool(
        compilation.get("fail_closed") is True
        and (
            compiled_mapping is None
            or (
                compiled_mapping.get("decision") == "REFUSE"
                and compiled_mapping.get("claim") == ""
                and compiled_mapping.get("evidence_ids") == []
            )
        )
    )
    unsupported = bool(
        row["compiled_metrics"]["unsupported_wrong_answer"]
    )
    if unsupported:
        source_state = "UNSUPPORTED_FOR_EXPECTED_REFUSAL"
    elif answer_source_verified:
        source_state = "VALIDATED_EVIDENCE_SPAN_VERBATIM"
    elif refusal_contract_verified:
        source_state = "EXPLICIT_REFUSAL_NO_CLAIM"
    elif fail_closed_safe_refusal:
        source_state = "FAIL_CLOSED_NO_SUPPORTED_CLAIM"
    else:
        source_state = "UNVERIFIED_OUTPUT_STATE"
    return {
        "source_state": source_state,
        "answer_claim_source_verified": answer_source_verified,
        "refusal_contract_verified": refusal_contract_verified,
        "fail_closed_safe_refusal": fail_closed_safe_refusal,
        "unsupported_wrong_answer": unsupported,
        "truthfulness_contract_satisfied": bool(
            not unsupported
            and (
                answer_source_verified
                or refusal_contract_verified
                or fail_closed_safe_refusal
            )
        ),
        "scope": (
            "PROVENANCE_AND_COMPILATION_TRUTHFULNESS_NOT_EXTERNAL_FACT_PROOF"
        ),
    }


def nonconformity_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return a unit-weighted, auditable integer score for one sample."""

    pointer = _require_mapping(row.get("pointer_metrics"), field="pointer_metrics")
    compiled = _require_mapping(
        row.get("compiled_metrics"),
        field="compiled_metrics",
    )
    generation = _require_mapping(row.get("generation"), field="generation")
    components = {
        "untrusted_finish": int(
            generation.get("trusted_finish_reason") is not True
        ),
        "pointer_parse_invalid": int(pointer.get("parse_valid") is not True),
        "pointer_task_mismatch": int(pointer.get("task_exact") is not True),
        "pointer_decision_mismatch": int(
            pointer.get("decision_exact") is not True
        ),
        "pointer_span_mismatch": int(pointer.get("span_exact") is not True),
        "compiler_or_schema_failure": int(
            pointer.get("compiler_accepted") is not True
            or compiled.get("schema_valid") is not True
        ),
        "compiled_citation_mismatch": int(
            compiled.get("citation_exact") is not True
        ),
        "compiled_provenance_mismatch": int(
            compiled.get("provenance_exact") is not True
        ),
        "compiled_strict_mismatch": int(
            compiled.get("strict_exact") is not True
        ),
        "unsupported_wrong_answer": int(
            compiled.get("unsupported_wrong_answer") is True
        ),
    }
    if tuple(components) != _NONCONFORMITY_COMPONENTS:
        raise CalibrationEvalV6Error(
            "nonconformity component order changed unexpectedly"
        )
    score = sum(components.values())
    return {
        "schema": "icmat_auditable_nonconformity.v6",
        "score": score,
        "minimum": 0,
        "maximum": len(_NONCONFORMITY_COMPONENTS),
        "components": components,
        "weights": "UNIT_WEIGHT_PER_BINARY_COMPONENT",
        "probability_interpretation_allowed": False,
    }


def _parse_alpha(alpha: Decimal | str | float) -> Decimal:
    if isinstance(alpha, bool):
        raise CalibrationEvalV6Error("alpha must be a decimal in (0,1)")
    try:
        value = alpha if isinstance(alpha, Decimal) else Decimal(str(alpha))
    except InvalidOperation as exc:
        raise CalibrationEvalV6Error("alpha must be a decimal in (0,1)") from exc
    if not value.is_finite() or value <= 0 or value >= 1:
        raise CalibrationEvalV6Error("alpha must be a decimal in (0,1)")
    return value


def split_conformal_summary(
    scores: Sequence[int],
    *,
    alpha: Decimal | str | float = FIXED_ALPHA,
) -> dict[str, Any]:
    """Compute a finite-sample corrected order-statistic threshold."""

    alpha_value = _parse_alpha(alpha)
    if not scores:
        raise CalibrationEvalV6Error("conformal scores must not be empty")
    normalized: list[int] = []
    for index, score in enumerate(scores):
        if (
            isinstance(score, bool)
            or not isinstance(score, int)
            or score < 0
            or score > len(_NONCONFORMITY_COMPONENTS)
        ):
            raise CalibrationEvalV6Error(
                f"conformal score {index} is outside the auditable score range"
            )
        normalized.append(score)
    sample_count = len(normalized)
    rank = int(
        (
            Decimal(sample_count + 1) * (Decimal(1) - alpha_value)
        ).to_integral_value(rounding=ROUND_CEILING)
    )
    if rank < 1 or rank > sample_count:
        raise CalibrationEvalV6Error(
            "alpha is too small for a finite split-conformal threshold"
        )
    ordered = sorted(normalized)
    threshold = ordered[rank - 1]
    included = sum(score <= threshold for score in normalized)

    loo_included = 0
    loo_available = sample_count > 1
    if loo_available:
        loo_rank = int(
            (
                Decimal(sample_count) * (Decimal(1) - alpha_value)
            ).to_integral_value(rounding=ROUND_CEILING)
        )
        if loo_rank > sample_count - 1:
            loo_available = False
        else:
            for held_index, score in enumerate(normalized):
                reference = sorted(
                    item
                    for index, item in enumerate(normalized)
                    if index != held_index
                )
                loo_threshold = reference[loo_rank - 1]
                loo_included += score <= loo_threshold

    nominal = Decimal(1) - alpha_value
    return {
        "method": "SPLIT_CONFORMAL_FINITE_SAMPLE_HIGHER_ORDER_STATISTIC",
        "alpha": format(alpha_value, "f"),
        "nominal_coverage": format(nominal, "f"),
        "sample_count": sample_count,
        "rank_formula": "ceil((n+1)*(1-alpha))",
        "rank": rank,
        "threshold": threshold,
        "score_definition": {
            "type": "UNIT_WEIGHTED_AUDITABLE_BINARY_ERROR_COUNT",
            "components": list(_NONCONFORMITY_COMPONENTS),
            "range": [0, len(_NONCONFORMITY_COMPONENTS)],
            "probability_interpretation_allowed": False,
        },
        "calibration_inclusion_coverage": _metric(included, sample_count),
        "leave_one_out_diagnostic": {
            "available": loo_available,
            "coverage": (
                _metric(loo_included, sample_count)
                if loo_available
                else None
            ),
            "claim_boundary": "INTERNAL_DIAGNOSTIC_NOT_AN_INDEPENDENT_TEST_SET",
        },
        "future_sample_interpretation": (
            "MARGINAL_COVERAGE_REQUIRES_EXCHANGEABILITY; "
            "THIS IS NOT A MODEL PROBABILITY OR BLIND COVERAGE RESULT"
        ),
    }


def _validate_child_sample(row: Any, *, index: int) -> Mapping[str, Any]:
    sample = _require_mapping(row, field=f"sample[{index}]")
    if sample.get("schema") != pointer_hf_eval_v6.SAMPLE_SCHEMA:
        raise CalibrationEvalV6Error(
            f"sample[{index}] pointer evaluator schema is invalid"
        )
    if sample.get("split") != "calibration":
        raise CalibrationEvalV6Error(f"sample[{index}] is not calibration")
    example_id = _require_text(
        sample.get("example_id"),
        field=f"sample[{index}].example_id",
    )
    for field, expected in (
        ("expected_passed_to_model", False),
        ("expected_passed_to_candidate_compiler", False),
        ("gold_repair_applied", False),
        ("blind_data_accessed", False),
    ):
        data_flow = _require_mapping(
            sample.get("data_flow"),
            field=f"sample[{index}].data_flow",
        )
        if data_flow.get(field) is not expected:
            raise CalibrationEvalV6Error(
                f"{example_id} violates data_flow.{field}"
            )
    pointer = _require_mapping(
        sample.get("pointer_metrics"),
        field=f"{example_id}.pointer_metrics",
    )
    compiled = _require_mapping(
        sample.get("compiled_metrics"),
        field=f"{example_id}.compiled_metrics",
    )
    for field in _POINTER_METRICS:
        if not isinstance(pointer.get(field), bool):
            raise CalibrationEvalV6Error(
                f"{example_id}.pointer_metrics.{field} must be boolean"
            )
    for field in (*_COMPILED_METRICS, "unsupported_wrong_answer"):
        if not isinstance(compiled.get(field), bool):
            raise CalibrationEvalV6Error(
                f"{example_id}.compiled_metrics.{field} must be boolean"
            )
    _require_mapping(sample.get("expected"), field=f"{example_id}.expected")
    _require_mapping(sample.get("generation"), field=f"{example_id}.generation")
    _require_mapping(sample.get("compilation"), field=f"{example_id}.compilation")
    return sample


def _aggregate_flags(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    names: Sequence[str],
) -> dict[str, dict[str, Any]]:
    return {
        name: _metric(
            sum(row[field][name] is True for row in rows),
            len(rows),
        )
        for name in names
    }


def _ratio_at_least(metric: Mapping[str, Any], numerator: int, denominator: int) -> bool:
    return (
        int(metric["numerator"]) * denominator
        >= numerator * int(metric["denominator"])
    )


def _quality_gates(
    *,
    pointer: Mapping[str, Mapping[str, Any]],
    compiled: Mapping[str, Mapping[str, Any]],
    pointer_diagnostics: Mapping[str, Any],
    refusal: Mapping[str, Any],
    unsupported_wrong_answer: Mapping[str, Any],
    conformal: Mapping[str, Any],
) -> list[dict[str, Any]]:
    f1 = _require_mapping(refusal.get("f1"), field="refusal.f1")
    populations = _require_mapping(
        refusal.get("population"),
        field="refusal.population",
    )
    gates = [
        (
            "CALIBRATION_HAS_ANSWER_AND_REFUSE_POPULATIONS",
            populations.get("answer", 0) > 0
            and populations.get("refuse", 0) > 0,
        ),
        (
            "POINTER_PARSE_VALID_100_PERCENT",
            pointer["parse_valid"]["numerator"]
            == pointer["parse_valid"]["denominator"],
        ),
        (
            "POINTER_INVALID_ZERO",
            pointer_diagnostics["invalid_count"] == 0,
        ),
        (
            "POINTER_AMBIGUOUS_ZERO",
            pointer_diagnostics["ambiguous_count"] == 0,
        ),
        (
            "POINTER_OUT_OF_RANGE_ZERO",
            pointer_diagnostics["out_of_range_count"] == 0,
        ),
        (
            "COMPILED_SCHEMA_VALID_100_PERCENT",
            compiled["schema_valid"]["numerator"]
            == compiled["schema_valid"]["denominator"],
        ),
        (
            "COMPILED_CITATION_EXACT_100_PERCENT",
            compiled["citation_exact"]["numerator"]
            == compiled["citation_exact"]["denominator"],
        ),
        (
            "COMPILED_PROVENANCE_EXACT_100_PERCENT",
            compiled["provenance_exact"]["numerator"]
            == compiled["provenance_exact"]["denominator"],
        ),
        (
            "ANSWER_SPAN_EXACT_AT_LEAST_95_PERCENT",
            _ratio_at_least(pointer["answer_span_exact"], 95, 100),
        ),
        (
            "REFUSAL_F1_AT_LEAST_95_PERCENT",
            _ratio_at_least(f1, 95, 100),
        ),
        (
            "UNSUPPORTED_WRONG_ANSWER_ZERO",
            unsupported_wrong_answer["numerator"] == 0,
        ),
        (
            "CONFORMAL_CALIBRATION_INCLUSION_AT_LEAST_NOMINAL",
            _ratio_at_least(
                conformal["calibration_inclusion_coverage"],
                90,
                100,
            ),
        ),
        (
            "CONFORMAL_THRESHOLD_ZERO_INCONSISTENCY",
            conformal["threshold"] == 0,
        ),
    ]
    return [
        {"gate": name, "passed": passed}
        for name, passed in gates
    ]


def _recompute_summary(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    backend_mode: str,
    model_bound: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(source_rows) != EXPECTED_CALIBRATION_ROWS:
        raise CalibrationEvalV6Error(
            f"calibration must contain exactly {EXPECTED_CALIBRATION_ROWS} samples"
        )
    validated: list[Mapping[str, Any]] = []
    example_ids: set[str] = set()
    for index, row in enumerate(source_rows):
        sample = _validate_child_sample(row, index=index)
        example_id = str(sample["example_id"])
        if example_id in example_ids:
            raise CalibrationEvalV6Error(
                f"duplicate calibration example_id: {example_id}"
            )
        example_ids.add(example_id)
        validated.append(sample)

    enriched: list[dict[str, Any]] = []
    scores: list[int] = []
    for source in validated:
        truthfulness = _truthfulness_record(source)
        nonconformity = nonconformity_record(source)
        scores.append(nonconformity["score"])
        enriched.append(
            {
                "schema": PER_SAMPLE_SCHEMA,
                "calibration_evaluator_version": CALIBRATION_EVALUATOR_VERSION,
                "example_id": source["example_id"],
                "split": "calibration",
                "backend": backend_mode,
                "raw_pointer": source["generation"]["raw_pointer"],
                "generation": source["generation"],
                "compiled_answer": source["compilation"].get("compiled_answer"),
                "compilation": source["compilation"],
                "expected": source["expected"],
                "pointer_metrics": source["pointer_metrics"],
                "compiled_metrics": source["compiled_metrics"],
                "truthfulness": truthfulness,
                "nonconformity": nonconformity,
                "source_sample_sha256": sha256_bytes(
                    canonical_json(source).encode("utf-8")
                ),
                "claim_boundary": (
                    "NONBLIND_CALIBRATION_SAMPLE_NOT_BLIND_OR_PRODUCTION_EVIDENCE"
                ),
            }
        )

    conformal = split_conformal_summary(scores, alpha=FIXED_ALPHA)
    threshold = int(conformal["threshold"])
    for row in enriched:
        row["nonconformity"]["within_frozen_risk_threshold"] = (
            row["nonconformity"]["score"] <= threshold
        )

    pointer = _aggregate_flags(
        validated,
        field="pointer_metrics",
        names=_POINTER_METRICS,
    )
    compiled = _aggregate_flags(
        validated,
        field="compiled_metrics",
        names=_COMPILED_METRICS,
    )
    answer_rows = [
        row
        for row in validated
        if row["expected"]["answer"]["decision"] == "ANSWER"
    ]
    pointer["answer_span_exact"] = _metric(
        sum(row["pointer_metrics"]["span_exact"] is True for row in answer_rows),
        len(answer_rows),
    )
    refusal = _refusal_metrics(validated)
    unsupported_wrong_answer = _metric(
        sum(
            row["compiled_metrics"]["unsupported_wrong_answer"] is True
            for row in validated
        ),
        len(validated),
    )
    parse_reason_counts = Counter(
        str(row["compilation"]["parse_reason"]["code"])
        for row in validated
    )
    pointer_diagnostics = {
        "schema_valid": pointer["parse_valid"],
        "strict_exact": pointer["strict_exact"],
        "span_exact": pointer["span_exact"],
        "invalid_count": sum(
            row["pointer_metrics"]["parse_valid"] is not True
            for row in validated
        ),
        "ambiguous_count": parse_reason_counts.get("AMBIGUOUS_SPAN_ID", 0),
        "out_of_range_count": parse_reason_counts.get("SPAN_NOT_FOUND", 0),
        "compiler_reason_counts": dict(sorted(parse_reason_counts.items())),
    }
    truthfulness = {
        "contract_satisfied": _metric(
            sum(
                row["truthfulness"]["truthfulness_contract_satisfied"] is True
                for row in enriched
            ),
            len(enriched),
        ),
        "source_state_counts": dict(
            sorted(
                Counter(
                    row["truthfulness"]["source_state"] for row in enriched
                ).items()
            )
        ),
        "scope": (
            "PROVENANCE_AND_COMPILATION_TRUTHFULNESS_NOT_EXTERNAL_FACT_PROOF"
        ),
    }
    gates = _quality_gates(
        pointer=pointer,
        compiled=compiled,
        pointer_diagnostics=pointer_diagnostics,
        refusal=refusal,
        unsupported_wrong_answer=unsupported_wrong_answer,
        conformal=conformal,
    )
    gates_passed = all(gate["passed"] for gate in gates)
    if backend_mode == "fixture":
        status = (
            "PASS_FIXTURE_CALIBRATION_PIPELINE_VERIFIED_NOT_MODEL_EVIDENCE"
            if gates_passed
            else "HOLD_FIXTURE_CALIBRATION_PIPELINE_RISK"
        )
    else:
        status = (
            "PASS_NONBLIND_CALIBRATION_MODEL_BOUND"
            if gates_passed and model_bound
            else "HOLD_NONBLIND_CALIBRATION_RISK"
        )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "calibration_evaluator_version": CALIBRATION_EVALUATOR_VERSION,
        "status": status,
        "split": "calibration",
        "rows": EXPECTED_CALIBRATION_ROWS,
        "complete_split": True,
        "backend": backend_mode,
        "model_bound": model_bound,
        "pointer_metrics": pointer,
        "pointer_diagnostics": pointer_diagnostics,
        "compiled_metrics": compiled,
        "refusal": refusal,
        "unsupported_wrong_answer": unsupported_wrong_answer,
        "truthfulness": truthfulness,
        "nonconformity_score_counts": {
            str(score): count
            for score, count in sorted(Counter(scores).items())
        },
        "conformal": conformal,
        "quality_gates": gates,
        "quality_gate_passed": gates_passed,
        "authorization": {
            "checkpoint_reselection_allowed": False,
            "blind_test_authorized": False,
            "gguf_export_authorized": False,
            "deployment_authorized": False,
            "production_integration_authorized": False,
        },
        "claim_boundary": (
            "NONBLIND_POST_FREEZE_CALIBRATION_ONLY; CONFORMAL SCORE IS AN "
            "AUDITABLE ERROR COUNT, NOT A PROBABILITY; NO BLIND, X5, BPU, "
            "DEPLOYMENT, OR PRODUCTION CLAIM IS AUTHORIZED"
        ),
    }
    return enriched, summary


def _source_bindings(runner_path: Path | None) -> dict[str, Any]:
    paths: dict[str, Path | None] = {
        "calibration_evaluator": Path(__file__).resolve(),
        "selection_freeze_verifier": Path(selection_freeze_v6.__file__).resolve(),
        "pointer_evaluator": Path(pointer_hf_eval_v6.__file__).resolve(),
        "pointer_compiler": Path(evidence_pointer_v6.__file__).resolve(),
        "runner": None if runner_path is None else Path(runner_path).resolve(),
    }
    bindings: dict[str, Any] = {}
    for name, path in paths.items():
        if path is None:
            bindings[name] = None
            continue
        if not path.is_file():
            raise CalibrationEvalV6Error(f"{name} source is unavailable")
        bindings[name] = {"path": str(path), "sha256": sha256_file(path)}
    return bindings


def run_calibration_evaluation(
    *,
    selection_freeze_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    backend_mode: str,
    fixture_path: Path | None = None,
    base_model_dir: Path | None = None,
    adapter_dir: Path | None = None,
    device: str | None = None,
    seed: int = 20260729,
    runner_path: Path | None = None,
) -> dict[str, Any]:
    """Run the complete post-freeze calibration chain and publish atomically."""

    # Reject blind-labelled inputs before resolving or opening any caller path.
    for field, path in (
        ("selection freeze", selection_freeze_path),
        ("dataset directory", dataset_dir),
        ("output directory", output_dir),
        ("fixture path", fixture_path),
        ("base model directory", base_model_dir),
        ("adapter directory", adapter_dir),
    ):
        if path is not None:
            _reject_blind_label(Path(path), field=field)
    if backend_mode not in pointer_hf_eval_v6.SUPPORTED_BACKENDS:
        raise CalibrationEvalV6Error(
            f"backend must be one of {sorted(pointer_hf_eval_v6.SUPPORTED_BACKENDS)}"
        )
    if seed != FIXED_INFERENCE_CONTRACT["seed"]:
        raise CalibrationEvalV6Error("seed must equal the frozen v6 seed 20260729")

    output = Path(output_dir).resolve()
    if output.exists():
        raise CalibrationEvalV6Error(
            f"output directory already exists; overwrite refused: {output}"
        )

    # The freeze is the first content-bearing input opened.
    freeze_resolved, freeze_payload, freeze_raw = _load_json(
        selection_freeze_path,
        field="selection freeze",
    )
    freeze_sha256 = sha256_bytes(freeze_payload)
    verification_inputs = _selection_verification_inputs(
        freeze_raw,
        dataset_dir=dataset_dir,
    )
    try:
        freeze_verification = selection_freeze_v6.verify_selection_freeze(
            freeze_receipt_path=freeze_resolved,
            **verification_inputs,
        )
    except (
        selection_freeze_v6.SelectionFreezeV6Error,
        OSError,
        ValueError,
    ) as exc:
        raise CalibrationEvalV6Error(
            "selection freeze failed authoritative recomputation"
        ) from exc
    if freeze_verification.get("sha256") != freeze_sha256:
        raise CalibrationEvalV6Error(
            "selection freeze verification returned a different receipt hash"
        )
    freeze = validate_selection_freeze(
        freeze_raw,
        verification=freeze_verification,
    )
    code_before = _source_bindings(runner_path)

    base_inventory: dict[str, Any] | None = None
    evaluator_base_inventory: dict[str, Any] | None = None
    checkpoint_inventory: dict[str, Any] | None = None
    adapter_inventory: dict[str, Any] | None = None
    evaluator_checkpoint_inventory: dict[str, Any] | None = None
    if backend_mode == "fixture":
        if fixture_path is None:
            raise CalibrationEvalV6Error("fixture backend requires fixture_path")
        if base_model_dir is not None or adapter_dir is not None or device is not None:
            raise CalibrationEvalV6Error(
                "fixture backend rejects model, adapter, and device arguments"
            )
        model_bound = False
    else:
        if fixture_path is not None:
            raise CalibrationEvalV6Error("hf_model backend rejects fixture_path")
        if base_model_dir is None or adapter_dir is None or device is None:
            raise CalibrationEvalV6Error(
                "hf_model requires base_model_dir, adapter_dir, and explicit device"
            )
        if (
            Path(base_model_dir).resolve(strict=True)
            != Path(freeze["base_model"]["path"]).resolve(strict=True)
        ):
            raise CalibrationEvalV6Error(
                "base model directory differs from the frozen selection"
            )
        if (
            Path(adapter_dir).resolve(strict=True)
            != Path(freeze["selection"]["checkpoint"]["path"]).resolve(strict=True)
            or Path(adapter_dir).resolve(strict=True)
            != Path(freeze["selection"]["adapter"]["path"]).resolve(strict=True)
        ):
            raise CalibrationEvalV6Error(
                "adapter directory differs from the frozen checkpoint"
            )
        base_inventory = _tree_inventory(
            base_model_dir,
            field="base model directory",
            casefold_order=True,
        )
        evaluator_base_inventory = _tree_inventory(
            base_model_dir,
            field="runtime base model directory",
        )
        checkpoint_inventory = _tree_inventory(
            adapter_dir,
            field="selected checkpoint directory",
            casefold_order=True,
        )
        adapter_inventory = _tree_inventory(
            adapter_dir,
            field="selected adapter files",
            selected_names=pointer_checkpoint_eval_v6.ADAPTER_FILENAMES,
            casefold_order=True,
        )
        evaluator_checkpoint_inventory = _tree_inventory(
            adapter_dir,
            field="runtime checkpoint directory",
        )
        if (
            base_inventory["tree_sha256"]
            != freeze["base_model"]["training_tree_sha256"]
        ):
            raise CalibrationEvalV6Error(
                "base model tree does not match the frozen selection"
            )
        if (
            evaluator_base_inventory["tree_sha256"]
            != freeze["base_model"]["evaluator_tree_sha256"]
        ):
            raise CalibrationEvalV6Error(
                "runtime base model tree does not match the frozen evaluator binding"
            )
        if (
            checkpoint_inventory["tree_sha256"]
            != freeze["selection"]["checkpoint"]["tree_sha256"]
        ):
            raise CalibrationEvalV6Error(
                "checkpoint tree does not match the frozen selection"
            )
        if (
            adapter_inventory["tree_sha256"]
            != freeze["selection"]["adapter"]["tree_sha256"]
        ):
            raise CalibrationEvalV6Error(
                "adapter tree does not match the frozen selection"
            )
        if (
            evaluator_checkpoint_inventory["tree_sha256"]
            != freeze["selection"]["evaluator_checkpoint_tree_sha256"]
        ):
            raise CalibrationEvalV6Error(
                "runtime checkpoint tree does not match the frozen evaluator binding"
            )
        model_bound = True

    output.parent.mkdir(parents=True, exist_ok=True)
    inference_output = output.parent / (
        f".{output.name}.pointer-calibration-{uuid.uuid4().hex}"
    )
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    try:
        pointer_result = pointer_hf_eval_v6.run_evaluation(
            dataset_dir=dataset_dir,
            split="calibration",
            output_dir=inference_output,
            backend_mode=backend_mode,
            fixture_path=fixture_path,
            base_model_dir=base_model_dir,
            adapter_dir=adapter_dir,
            device=device,
            seed=seed,
            max_samples=None,
            runner_path=runner_path,
        )
        if pointer_result.get("examples") != EXPECTED_CALIBRATION_ROWS:
            raise CalibrationEvalV6Error(
                "pointer evaluator did not complete exactly 150 calibration rows"
            )
        _, child_payload, child_rows = _load_jsonl(
            inference_output / "sample_results.v6.jsonl",
            field="pointer calibration samples",
        )
        _, child_summary_payload, child_summary = _load_json(
            inference_output / "summary.v6.json",
            field="pointer calibration summary",
        )
        _, child_receipt_payload, child_receipt = _load_json(
            inference_output / "run_receipt.v6.json",
            field="pointer calibration receipt",
        )
        if child_summary.get("status") != "CALIBRATION_EVALUATION_COMPLETE":
            raise CalibrationEvalV6Error("pointer calibration summary is incomplete")
        selection = _require_mapping(
            child_summary.get("selection"),
            field="pointer summary selection",
        )
        if (
            selection.get("complete_split") is not True
            or selection.get("rows_in_file") != EXPECTED_CALIBRATION_ROWS
            or selection.get("rows_evaluated") != EXPECTED_CALIBRATION_ROWS
            or selection.get("max_samples") is not None
        ):
            raise CalibrationEvalV6Error(
                "pointer evaluator did not prove complete 150-row calibration"
            )
        child_dataset = _require_mapping(
            child_receipt.get("dataset"),
            field="pointer receipt dataset",
        )
        opened_sha256 = _require_sha256(
            child_dataset.get("opened_split_sha256"),
            field="pointer receipt opened_split_sha256",
        )
        if opened_sha256 != freeze["dataset"]["calibration"]["sha256"]:
            raise CalibrationEvalV6Error(
                "opened calibration split does not match selection freeze"
            )
        if (
            child_dataset.get("rows_in_file") != EXPECTED_CALIBRATION_ROWS
            or child_dataset.get("rows_evaluated") != EXPECTED_CALIBRATION_ROWS
            or child_dataset.get("blind_data_accessed") is not False
        ):
            raise CalibrationEvalV6Error(
                "pointer receipt calibration boundary is invalid"
            )

        child_bindings = _require_mapping(
            child_receipt.get("bindings"),
            field="pointer receipt bindings",
        )
        if model_bound:
            assert (
                base_inventory is not None
                and evaluator_base_inventory is not None
                and adapter_inventory is not None
                and evaluator_checkpoint_inventory is not None
            )
            if (
                child_bindings.get("base_model_tree_sha256")
                != evaluator_base_inventory["tree_sha256"]
                or child_bindings.get("adapter_tree_sha256")
                != evaluator_checkpoint_inventory["tree_sha256"]
            ):
                raise CalibrationEvalV6Error(
                    "pointer evaluator runtime model binding changed"
                )
        elif (
            child_bindings.get("base_model_tree_sha256") is not None
            or child_bindings.get("adapter_tree_sha256") is not None
        ):
            raise CalibrationEvalV6Error(
                "fixture backend unexpectedly claimed a model binding"
            )

        enriched, summary = _recompute_summary(
            child_rows,
            backend_mode=backend_mode,
            model_bound=model_bound,
        )
        code_after = _source_bindings(runner_path)
        if code_after != code_before:
            raise CalibrationEvalV6Error(
                "calibration implementation changed during evaluation"
            )
        if sha256_file(freeze_resolved) != freeze_sha256:
            raise CalibrationEvalV6Error(
                "selection freeze changed during calibration"
            )
        calibration_path = Path(
            str(child_dataset["opened_split_path"])
        ).resolve(strict=True)
        if sha256_file(calibration_path) != opened_sha256:
            raise CalibrationEvalV6Error(
                "calibration split changed during evaluation"
            )

        staging.mkdir()
        sample_path = staging / "per_sample.v6.jsonl"
        summary_path = staging / "summary.v6.json"
        receipt_path = staging / "receipt.v6.json"
        sample_path.write_bytes(_jsonl_bytes(enriched))
        summary_path.write_bytes(_json_bytes(summary))
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "calibration_evaluator_version": CALIBRATION_EVALUATOR_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "status": summary["status"],
            "selection_freeze": {
                "path": str(freeze_resolved),
                "sha256": freeze_sha256,
                "validated_schema": freeze["schema"],
                "validated_status": freeze["status"],
                "checkpoint_id": freeze["selection"]["checkpoint_id"],
                "base_model_tree_sha256": freeze["base_model"][
                    "training_tree_sha256"
                ],
                "base_model_evaluator_tree_sha256": freeze["base_model"][
                    "evaluator_tree_sha256"
                ],
                "checkpoint_tree_sha256": freeze["selection"]["checkpoint"][
                    "tree_sha256"
                ],
                "adapter_tree_sha256": freeze["selection"]["adapter"][
                    "tree_sha256"
                ],
                "verification_status": freeze["verification"]["status"],
            },
            "dataset": {
                "opened_split": "calibration",
                "path": str(calibration_path),
                "sha256": opened_sha256,
                "rows": EXPECTED_CALIBRATION_ROWS,
                "complete_split": True,
                "calibration_opened_after_freeze_validation": True,
                "blind_data_accessed": False,
            },
            "execution": {
                "backend": backend_mode,
                "model_bound": model_bound,
                "fixed_inference_contract": dict(FIXED_INFERENCE_CONTRACT),
                "expected_passed_to_model": False,
                "expected_passed_to_candidate_compiler": False,
                "gold_repair_applied": False,
                "checkpoint_reselection_performed": False,
                "blind_supported": False,
                "blind_data_accessed": False,
            },
            "model": {
                "base": base_inventory,
                "runtime_base": evaluator_base_inventory,
                "checkpoint": checkpoint_inventory,
                "adapter": adapter_inventory,
                "runtime_checkpoint": evaluator_checkpoint_inventory,
                "fixture_not_model_evidence": backend_mode == "fixture",
            },
            "implementation": code_before,
            "upstream_pointer_evidence": {
                "sample_results_sha256": sha256_bytes(child_payload),
                "summary_sha256": sha256_bytes(child_summary_payload),
                "receipt_sha256": sha256_bytes(child_receipt_payload),
                "summary_metrics_trusted": False,
                "per_sample_metrics_recomputed": True,
            },
            "artifacts": {
                "per_sample.v6.jsonl": {
                    "sha256": sha256_file(sample_path),
                    "bytes": sample_path.stat().st_size,
                    "rows": EXPECTED_CALIBRATION_ROWS,
                },
                "summary.v6.json": {
                    "sha256": sha256_file(summary_path),
                    "bytes": summary_path.stat().st_size,
                },
            },
            "authorization": summary["authorization"],
            "claim_boundary": summary["claim_boundary"],
        }
        receipt_path.write_bytes(_json_bytes(receipt))
        if output.exists():
            raise CalibrationEvalV6Error(
                "output appeared during evaluation; overwrite refused"
            )
        staging.replace(output)
    except pointer_hf_eval_v6.PointerHFEvalV6Error as exc:
        raise CalibrationEvalV6Error(str(exc)) from exc
    finally:
        shutil.rmtree(inference_output, ignore_errors=True)
        shutil.rmtree(staging, ignore_errors=True)

    return {
        "status": summary["status"],
        "output_dir": str(output),
        "examples": EXPECTED_CALIBRATION_ROWS,
        "backend": backend_mode,
        "model_bound": model_bound,
        "quality_gate_passed": summary["quality_gate_passed"],
        "conformal_threshold": summary["conformal"]["threshold"],
        "blind_data_accessed": False,
        "hashes": {
            "per_sample.v6.jsonl": sha256_file(
                output / "per_sample.v6.jsonl"
            ),
            "summary.v6.json": sha256_file(output / "summary.v6.json"),
            "receipt.v6.json": sha256_file(output / "receipt.v6.json"),
        },
    }


__all__ = [
    "CALIBRATION_EVALUATOR_VERSION",
    "CalibrationEvalV6Error",
    "EXPECTED_CALIBRATION_ROWS",
    "FIXED_ALPHA",
    "FIXED_INFERENCE_CONTRACT",
    "PER_SAMPLE_SCHEMA",
    "RECEIPT_SCHEMA",
    "SELECTION_FREEZE_SCHEMA",
    "SELECTION_FREEZE_STATUS",
    "SUMMARY_SCHEMA",
    "nonconformity_record",
    "run_calibration_evaluation",
    "split_conformal_summary",
    "validate_selection_freeze",
]
