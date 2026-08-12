"""Immutable non-blind canary acceptance gate for ICMat Pointer v6.

The gate consumes one completed canary ``evaluation_index.v6.json`` and only
the ``sample_results.v6.jsonl`` and ``summary.v6.json`` files named by each
checkpoint. It independently recomputes the registered metrics, checks the
index and summaries against those rows, and deterministically identifies one
checkpoint solely as evidence for whether final three-seed training may start.

It never reads calibration or blind content and never authorizes a final
model, calibration, blind evaluation, export, deployment, or production use.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from functools import cmp_to_key
from pathlib import Path
from typing import Any

from . import evidence_pointer_v6, pointer_checkpoint_eval_v6, pointer_hf_eval_v6

SCHEMA = "icmat_llm_canary_acceptance_receipt.v6"
VERSION = "icmat-llm-canary-acceptance-v6.1.0"
PASS_STATUS = "PASS_CANARY_ACCEPTED_FOR_THREE_SEED_TRAINING_ONLY"
STOP_STATUS = "STOP_CANARY_NOT_ACCEPTED"
ERROR_STATUS = "CANARY_ACCEPTANCE_NOT_RECORDED"

EXPECTED_INDEX_STATUS = "PASS_CANARY_1X6_VALIDATION_EVALUATED_NO_SELECTION"
EXPECTED_CHECKPOINTS = 6
EXPECTED_SAMPLES = 18
EXPECTED_EPOCHS = frozenset(range(1, 7))
RATE_GATE_NUMERATOR = 90
RATE_GATE_DENOMINATOR = 100
AMBIGUOUS_CODES = frozenset(
    {"AMBIGUOUS_EVIDENCE_ID", "AMBIGUOUS_SPAN_ID"}
)
OUT_OF_RANGE_CODES = frozenset({"SPAN_NOT_FOUND"})
READABLE_CHILD_ARTIFACTS = frozenset(
    {"sample_results.v6.jsonl", "summary.v6.json"}
)

CLAIM_BOUNDARY = (
    "This receipt independently recomputes one non-blind 1x6 canary and may "
    "authorize only the start of final three-seed training. The reference "
    "checkpoint is not a final model selection. Calibration, blind "
    "evaluation, GGUF export, X5 deployment, and production integration "
    "remain unauthorized."
)


class CanaryAcceptanceV6Error(RuntimeError):
    """Raised when a trustworthy v6 canary decision cannot be recorded."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise CanaryAcceptanceV6Error(
            "value cannot be represented as finite canonical JSON"
        ) from exc


def _pretty_bytes(value: Mapping[str, Any]) -> bytes:
    try:
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
    except (TypeError, ValueError) as exc:
        raise CanaryAcceptanceV6Error(
            "receipt cannot be represented as finite JSON"
        ) from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanaryAcceptanceV6Error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise CanaryAcceptanceV6Error(
        f"non-finite JSON constant is forbidden: {value}"
    )


def _stable_bytes(path: Path, *, field: str) -> tuple[Path, bytes]:
    raw = Path(path)
    if raw.is_symlink():
        raise CanaryAcceptanceV6Error(f"{field} must not be a symlink: {raw}")
    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError as exc:
        raise CanaryAcceptanceV6Error(f"{field} does not exist: {raw}") from exc
    if not resolved.is_file():
        raise CanaryAcceptanceV6Error(
            f"{field} must be a regular file: {resolved}"
        )
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
        raise CanaryAcceptanceV6Error(f"{field} changed while it was read")
    return resolved, first


def _load_json_bytes(payload: bytes, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CanaryAcceptanceV6Error(
            f"{field} must contain valid UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise CanaryAcceptanceV6Error(f"{field} JSON root must be an object")
    return value


def _load_json(path: Path, *, field: str) -> tuple[Path, bytes, dict[str, Any]]:
    resolved, payload = _stable_bytes(path, field=field)
    return resolved, payload, _load_json_bytes(payload, field=field)


def _load_jsonl_bytes(payload: bytes, *, field: str) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CanaryAcceptanceV6Error(f"{field} must be UTF-8 JSONL") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise CanaryAcceptanceV6Error(
                f"{field} contains blank line {line_number}"
            )
        try:
            value = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=_reject_nonfinite_constant,
            )
        except json.JSONDecodeError as exc:
            raise CanaryAcceptanceV6Error(
                f"{field} line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise CanaryAcceptanceV6Error(
                f"{field} line {line_number} must be an object"
            )
        rows.append(value)
    if not rows:
        raise CanaryAcceptanceV6Error(f"{field} is empty")
    return rows


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CanaryAcceptanceV6Error(f"{field} must be an object")
    return value


def _require_sequence(value: Any, *, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise CanaryAcceptanceV6Error(f"{field} must be an array")
    return value


def _require_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise CanaryAcceptanceV6Error(f"{field} must be boolean")
    return value


def _require_int(
    value: Any,
    *,
    field: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CanaryAcceptanceV6Error(f"{field} must be an integer")
    if value < minimum:
        raise CanaryAcceptanceV6Error(f"{field} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise CanaryAcceptanceV6Error(f"{field} must be <= {maximum}")
    return value


def _require_text(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(ord(character) < 32 for character in value)
    ):
        raise CanaryAcceptanceV6Error(
            f"{field} must be a non-empty trimmed string"
        )
    return value


def _require_sha256(value: Any, *, field: str) -> str:
    text = _require_text(value, field=field)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise CanaryAcceptanceV6Error(f"{field} must be a lowercase SHA-256")
    return text


def _parse_loss(value: Any, *, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, str, Decimal)
    ):
        raise CanaryAcceptanceV6Error(
            f"{field} must be a finite non-negative decimal"
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise CanaryAcceptanceV6Error(f"{field} must be finite")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise CanaryAcceptanceV6Error(f"{field} is not decimal") from exc
    if not result.is_finite() or result < 0:
        raise CanaryAcceptanceV6Error(
            f"{field} must be a finite non-negative decimal"
        )
    return result


def _ratio(numerator: int, denominator: int) -> dict[str, int]:
    return {"numerator": numerator, "denominator": denominator}


def _ratio_at_least(
    ratio: Mapping[str, Any], numerator: int, denominator: int
) -> bool:
    return (
        int(ratio["numerator"]) * denominator
        >= numerator * int(ratio["denominator"])
    )


def _compare_ratio(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    left_product = int(left["numerator"]) * int(right["denominator"])
    right_product = int(right["numerator"]) * int(left["denominator"])
    return (left_product > right_product) - (left_product < right_product)


def _verify_compilation_hash(compilation: Mapping[str, Any], *, field: str) -> None:
    expected = _require_sha256(
        compilation.get("compilation_sha256"),
        field=f"{field}.compilation_sha256",
    )
    body = dict(compilation)
    body.pop("compilation_sha256", None)
    actual = _sha256_bytes(_canonical_json(body).encode("utf-8"))
    if actual != expected:
        raise CanaryAcceptanceV6Error(
            f"{field}.compilation_sha256 is invalid"
        )


def _expected_pointer(value: Any, *, field: str) -> Mapping[str, Any]:
    pointer = _require_mapping(value, field=field)
    if set(pointer) != evidence_pointer_v6.POINTER_KEYS:
        raise CanaryAcceptanceV6Error(f"{field} keys mismatch")
    task = _require_text(pointer.get("task"), field=f"{field}.task")
    decision = pointer.get("decision")
    if decision not in evidence_pointer_v6.ALLOWED_DECISIONS:
        raise CanaryAcceptanceV6Error(f"{field}.decision is invalid")
    span_id = pointer.get("span_id")
    if decision == "ANSWER":
        if (
            not isinstance(span_id, str)
            or evidence_pointer_v6.SPAN_ID_RE.fullmatch(span_id) is None
        ):
            raise CanaryAcceptanceV6Error(f"{field}.span_id is invalid")
    elif span_id is not None:
        raise CanaryAcceptanceV6Error(
            f"{field}.span_id must be null for REFUSE"
        )
    return {"task": task, "decision": decision, "span_id": span_id}


def _parsed_pointer(value: Any, *, field: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return _expected_pointer(value, field=field)


def _answer(value: Any, *, field: str) -> Mapping[str, Any]:
    answer = _require_mapping(value, field=field)
    errors = evidence_pointer_v6.validate_student_answer(answer)
    if errors:
        raise CanaryAcceptanceV6Error(
            f"{field} violates answer schema: {errors}"
        )
    return answer


def _claimed_bool(
    mapping: Mapping[str, Any],
    key: str,
    expected: bool,
    *,
    field: str,
) -> None:
    claimed = _require_bool(mapping.get(key), field=f"{field}.{key}")
    if claimed is not expected:
        raise CanaryAcceptanceV6Error(
            f"{field}.{key} differs from independent recomputation"
        )


def _recompute_checkpoint(
    rows: Sequence[Mapping[str, Any]],
    *,
    checkpoint_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(rows) != EXPECTED_SAMPLES:
        raise CanaryAcceptanceV6Error(
            f"{checkpoint_id} must contain exactly {EXPECTED_SAMPLES} samples"
        )
    observed_ids: set[str] = set()
    pointer_schema = 0
    pointer_invalid = 0
    pointer_ambiguous = 0
    pointer_out_of_range = 0
    unsupported_wrong = 0
    compiled_schema = 0
    compiled_citation = 0
    compiled_provenance = 0
    compiled_strict = 0
    answer_span_exact = 0
    answer_pointer_strict = 0
    answer_compiled_strict = 0
    answer_examples = 0
    refuse_examples = 0
    refuse_true_positive = 0
    refuse_false_positive = 0
    reason_counts: Counter[str] = Counter()
    compiler_accepted_count = 0
    fail_closed_count = 0
    strata: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for index, raw_row in enumerate(rows):
        field = f"{checkpoint_id}.samples[{index}]"
        row = _require_mapping(raw_row, field=field)
        if row.get("schema") != pointer_hf_eval_v6.SAMPLE_SCHEMA:
            raise CanaryAcceptanceV6Error(f"{field}.schema is unsupported")
        example_id = _require_text(
            row.get("example_id"), field=f"{field}.example_id"
        )
        if example_id in observed_ids:
            raise CanaryAcceptanceV6Error(
                f"{checkpoint_id} contains duplicate example_id {example_id}"
            )
        observed_ids.add(example_id)
        if row.get("split") != "validation" or row.get("backend") != "hf_model":
            raise CanaryAcceptanceV6Error(
                f"{field} must be validation/hf_model evidence"
            )

        data_flow = _require_mapping(
            row.get("data_flow"), field=f"{field}.data_flow"
        )
        for name in (
            "expected_passed_to_model",
            "expected_passed_to_candidate_compiler",
            "gold_repair_applied",
            "assistant_target_visible",
            "blind_data_accessed",
        ):
            if _require_bool(
                data_flow.get(name), field=f"{field}.data_flow.{name}"
            ):
                raise CanaryAcceptanceV6Error(
                    f"{field}.data_flow.{name} must remain false"
                )

        generation = _require_mapping(
            row.get("generation"), field=f"{field}.generation"
        )
        raw_pointer = generation.get("raw_pointer")
        if not isinstance(raw_pointer, str):
            raise CanaryAcceptanceV6Error(
                f"{field}.generation.raw_pointer must be a string"
            )
        if generation.get("raw_pointer_sha256") != _sha256_bytes(
            raw_pointer.encode("utf-8")
        ):
            raise CanaryAcceptanceV6Error(
                f"{field}.generation.raw_pointer_sha256 is invalid"
            )

        compilation = _require_mapping(
            row.get("compilation"), field=f"{field}.compilation"
        )
        if compilation.get("schema") != evidence_pointer_v6.COMPILATION_SCHEMA:
            raise CanaryAcceptanceV6Error(
                f"{field}.compilation.schema is unsupported"
            )
        _verify_compilation_hash(compilation, field=f"{field}.compilation")
        if (
            compilation.get("raw_pointer") != raw_pointer
            or compilation.get("raw_pointer_sha256")
            != generation.get("raw_pointer_sha256")
        ):
            raise CanaryAcceptanceV6Error(
                f"{field} generation and compilation pointer binding differs"
            )
        reason = _require_mapping(
            compilation.get("parse_reason"),
            field=f"{field}.compilation.parse_reason",
        )
        reason_code = _require_text(
            reason.get("code"),
            field=f"{field}.compilation.parse_reason.code",
        )
        reason_counts[reason_code] += 1
        parsed = _parsed_pointer(
            compilation.get("parsed_pointer"),
            field=f"{field}.compilation.parsed_pointer",
        )
        accepted = compilation.get("status") == "COMPILED"
        fail_closed = _require_bool(
            compilation.get("fail_closed"),
            field=f"{field}.compilation.fail_closed",
        )
        if accepted:
            if fail_closed or reason_code != "OK" or parsed is None:
                raise CanaryAcceptanceV6Error(
                    f"{field} has incoherent COMPILED state"
                )
        elif compilation.get("status") != "FAIL_CLOSED" or not fail_closed:
            raise CanaryAcceptanceV6Error(
                f"{field} has incoherent FAIL_CLOSED state"
            )
        compiler_accepted_count += int(accepted)
        fail_closed_count += int(fail_closed)

        expected = _require_mapping(
            row.get("expected"), field=f"{field}.expected"
        )
        expected_pointer = _expected_pointer(
            expected.get("pointer"), field=f"{field}.expected.pointer"
        )
        expected_answer = _answer(
            expected.get("answer"), field=f"{field}.expected.answer"
        )
        if (
            expected_answer.get("task") != expected_pointer["task"]
            or expected_answer.get("decision") != expected_pointer["decision"]
        ):
            raise CanaryAcceptanceV6Error(
                f"{field} expected pointer and answer disagree"
            )

        candidate_raw = compilation.get("compiled_answer")
        candidate = (
            _answer(candidate_raw, field=f"{field}.compilation.compiled_answer")
            if isinstance(candidate_raw, Mapping)
            else None
        )
        parse_valid = parsed is not None
        span_exact = (
            parsed is not None
            and parsed.get("span_id") == expected_pointer.get("span_id")
        )
        pointer_strict = bool(accepted and parsed == expected_pointer)
        schema_valid = bool(accepted and candidate is not None)
        citation_exact = bool(
            accepted
            and candidate is not None
            and candidate.get("evidence_ids")
            == expected_answer.get("evidence_ids")
        )
        provenance_exact = bool(
            accepted
            and candidate is not None
            and candidate.get("provenance")
            == expected_answer.get("provenance")
        )
        strict_exact = bool(
            accepted and candidate is not None and candidate == expected_answer
        )
        unsupported = bool(
            accepted
            and expected_answer.get("decision") == "REFUSE"
            and candidate is not None
            and candidate.get("decision") == "ANSWER"
        )

        pointer_metrics = _require_mapping(
            row.get("pointer_metrics"), field=f"{field}.pointer_metrics"
        )
        for key, actual in (
            ("parse_valid", parse_valid),
            ("span_exact", span_exact),
            ("strict_exact", pointer_strict),
            ("compiler_accepted", accepted),
        ):
            _claimed_bool(
                pointer_metrics,
                key,
                actual,
                field=f"{field}.pointer_metrics",
            )
        compiled_metrics = _require_mapping(
            row.get("compiled_metrics"), field=f"{field}.compiled_metrics"
        )
        for key, actual in (
            ("schema_valid", schema_valid),
            ("citation_exact", citation_exact),
            ("provenance_exact", provenance_exact),
            ("strict_exact", strict_exact),
            ("unsupported_wrong_answer", unsupported),
        ):
            _claimed_bool(
                compiled_metrics,
                key,
                actual,
                field=f"{field}.compiled_metrics",
            )

        metadata = _require_mapping(
            row.get("metadata"), field=f"{field}.metadata"
        )
        domain = _require_text(
            metadata.get("domain"), field=f"{field}.metadata.domain"
        )
        task = _require_text(
            metadata.get("task"), field=f"{field}.metadata.task"
        )
        decision = str(expected_answer["decision"])
        if task != expected_answer.get("task"):
            raise CanaryAcceptanceV6Error(
                f"{field}.metadata.task differs from expected task"
            )

        pointer_schema += int(parse_valid)
        pointer_invalid += int(not parse_valid)
        pointer_ambiguous += int(reason_code in AMBIGUOUS_CODES)
        pointer_out_of_range += int(reason_code in OUT_OF_RANGE_CODES)
        unsupported_wrong += int(unsupported)
        compiled_schema += int(schema_valid)
        compiled_citation += int(citation_exact)
        compiled_provenance += int(provenance_exact)
        compiled_strict += int(strict_exact)
        if decision == "ANSWER":
            answer_examples += 1
            answer_span_exact += int(span_exact)
            answer_pointer_strict += int(pointer_strict)
            answer_compiled_strict += int(strict_exact)
        else:
            refuse_examples += 1

        predicted_decision = (
            parsed.get("decision") if accepted and parsed is not None else None
        )
        if decision == "REFUSE" and predicted_decision == "REFUSE":
            refuse_true_positive += 1
        if decision == "ANSWER" and predicted_decision == "REFUSE":
            refuse_false_positive += 1
        for kind, value in (
            ("domain", domain),
            ("task", task),
            ("decision", decision),
        ):
            stratum = strata[f"{kind}={value}"]
            stratum[0] += int(strict_exact)
            stratum[1] += 1

    if answer_examples == 0 or refuse_examples == 0:
        raise CanaryAcceptanceV6Error(
            f"{checkpoint_id} lacks ANSWER or REFUSE examples"
        )
    refuse_false_negative = refuse_examples - refuse_true_positive
    metrics = {
        "completed_samples": len(rows),
        "pointer_schema_valid": _ratio(pointer_schema, len(rows)),
        "pointer_invalid_count": pointer_invalid,
        "pointer_ambiguous_count": pointer_ambiguous,
        "pointer_out_of_range_count": pointer_out_of_range,
        "unsupported_wrong_answer_count": unsupported_wrong,
        "compiled_schema_valid": _ratio(compiled_schema, len(rows)),
        "compiled_citation_exact": _ratio(compiled_citation, len(rows)),
        "compiled_provenance_exact": _ratio(
            compiled_provenance, len(rows)
        ),
        "answer_span_exact": _ratio(answer_span_exact, answer_examples),
        "refuse_confusion": {
            "true_positive": refuse_true_positive,
            "false_positive": refuse_false_positive,
            "false_negative": refuse_false_negative,
        },
        "compiled_strict_exact": _ratio(compiled_strict, len(rows)),
        "stratified_compiled_strict": [
            {
                "stratum": name,
                "numerator": values[0],
                "denominator": values[1],
            }
            for name, values in sorted(strata.items())
        ],
    }
    audit = {
        "example_ids": sorted(observed_ids),
        "answer_examples": answer_examples,
        "refuse_examples": refuse_examples,
        "answer_pointer_strict": _ratio(
            answer_pointer_strict, answer_examples
        ),
        "answer_compiled_strict": _ratio(
            answer_compiled_strict, answer_examples
        ),
        "reason_counts": dict(sorted(reason_counts.items())),
        "compiler_accepted": compiler_accepted_count,
        "fail_closed": fail_closed_count,
    }
    return metrics, audit


def _summary_ratio(
    value: Any,
    expected: Mapping[str, int],
    *,
    field: str,
) -> None:
    record = _require_mapping(value, field=field)
    numerator = _require_int(record.get("numerator"), field=f"{field}.numerator")
    denominator = _require_int(
        record.get("denominator"), field=f"{field}.denominator", minimum=1
    )
    if (numerator, denominator) != (
        expected["numerator"],
        expected["denominator"],
    ):
        raise CanaryAcceptanceV6Error(
            f"{field} differs from independent recomputation"
        )
    rate = record.get("rate")
    if (
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isfinite(float(rate))
        or float(rate) != numerator / denominator
    ):
        raise CanaryAcceptanceV6Error(f"{field}.rate is inconsistent")


def _validate_summary(
    summary: Mapping[str, Any],
    *,
    metrics: Mapping[str, Any],
    audit: Mapping[str, Any],
    checkpoint_id: str,
) -> None:
    field = f"{checkpoint_id}.summary"
    if (
        summary.get("schema") != pointer_hf_eval_v6.SUMMARY_SCHEMA
        or summary.get("status") != "VALIDATION_EVALUATION_COMPLETE"
        or summary.get("split") != "validation"
        or summary.get("backend") != "hf_model"
    ):
        raise CanaryAcceptanceV6Error(f"{field} status/schema is invalid")
    selection = _require_mapping(
        summary.get("selection"), field=f"{field}.selection"
    )
    if (
        selection.get("rows_in_file") != EXPECTED_SAMPLES
        or selection.get("rows_evaluated") != EXPECTED_SAMPLES
        or selection.get("max_samples") is not None
        or selection.get("complete_split") is not True
    ):
        raise CanaryAcceptanceV6Error(f"{field}.selection is invalid")

    pointer = _require_mapping(
        summary.get("pointer_metrics"), field=f"{field}.pointer_metrics"
    )
    _summary_ratio(
        pointer.get("parse_valid"),
        metrics["pointer_schema_valid"],
        field=f"{field}.pointer_metrics.parse_valid",
    )
    compiled = _require_mapping(
        summary.get("compiled_metrics"), field=f"{field}.compiled_metrics"
    )
    for summary_name, metric_name in (
        ("schema_valid", "compiled_schema_valid"),
        ("citation_exact", "compiled_citation_exact"),
        ("provenance_exact", "compiled_provenance_exact"),
        ("strict_exact", "compiled_strict_exact"),
    ):
        _summary_ratio(
            compiled.get(summary_name),
            metrics[metric_name],
            field=f"{field}.compiled_metrics.{summary_name}",
        )
    _summary_ratio(
        summary.get("answer_pointer_strict_accuracy"),
        audit["answer_pointer_strict"],
        field=f"{field}.answer_pointer_strict_accuracy",
    )
    _summary_ratio(
        summary.get("answer_compiled_strict_accuracy"),
        audit["answer_compiled_strict"],
        field=f"{field}.answer_compiled_strict_accuracy",
    )
    refuse = _require_mapping(summary.get("refuse"), field=f"{field}.refuse")
    confusion = metrics["refuse_confusion"]
    if (
        refuse.get("examples") != audit["refuse_examples"]
        or refuse.get("true_positive") != confusion["true_positive"]
        or refuse.get("false_positive") != confusion["false_positive"]
        or refuse.get("false_negative") != confusion["false_negative"]
    ):
        raise CanaryAcceptanceV6Error(
            f"{field}.refuse differs from independent recomputation"
        )
    unsupported = _ratio(
        metrics["unsupported_wrong_answer_count"],
        audit["refuse_examples"],
    )
    _summary_ratio(
        summary.get("unsupported_wrong_answer_rate"),
        unsupported,
        field=f"{field}.unsupported_wrong_answer_rate",
    )
    compiler = _require_mapping(
        summary.get("compiler"), field=f"{field}.compiler"
    )
    if (
        compiler.get("accepted") != audit["compiler_accepted"]
        or compiler.get("fail_closed") != audit["fail_closed"]
        or compiler.get("reason_counts") != audit["reason_counts"]
    ):
        raise CanaryAcceptanceV6Error(
            f"{field}.compiler differs from independent recomputation"
        )
    boundaries = _require_mapping(
        summary.get("execution_boundaries"),
        field=f"{field}.execution_boundaries",
    )
    for name in (
        "blind_split_supported",
        "blind_data_accessed",
        "gold_repair_applied",
        "promotion_authorized",
    ):
        if _require_bool(
            boundaries.get(name), field=f"{field}.execution_boundaries.{name}"
        ):
            raise CanaryAcceptanceV6Error(
                f"{field}.execution_boundaries.{name} must remain false"
            )


def _refuse_f1(metrics: Mapping[str, Any]) -> dict[str, int]:
    confusion = _require_mapping(
        metrics.get("refuse_confusion"), field="metrics.refuse_confusion"
    )
    tp = _require_int(confusion.get("true_positive"), field="refuse.tp")
    fp = _require_int(confusion.get("false_positive"), field="refuse.fp")
    fn = _require_int(confusion.get("false_negative"), field="refuse.fn")
    numerator = 2 * tp
    denominator = numerator + fp + fn
    if denominator == 0:
        raise CanaryAcceptanceV6Error("refusal F1 is undefined")
    return _ratio(numerator, denominator)


def _failed_gates(metrics: Mapping[str, Any]) -> list[str]:
    checks = (
        (
            "COMPLETE_18",
            metrics["completed_samples"] == EXPECTED_SAMPLES,
        ),
        (
            "POINTER_SCHEMA_100_PERCENT",
            metrics["pointer_schema_valid"]["numerator"]
            == metrics["pointer_schema_valid"]["denominator"],
        ),
        ("POINTER_INVALID_ZERO", metrics["pointer_invalid_count"] == 0),
        (
            "POINTER_AMBIGUOUS_ZERO",
            metrics["pointer_ambiguous_count"] == 0,
        ),
        (
            "POINTER_OUT_OF_RANGE_ZERO",
            metrics["pointer_out_of_range_count"] == 0,
        ),
        (
            "UNSUPPORTED_WRONG_ANSWER_ZERO",
            metrics["unsupported_wrong_answer_count"] == 0,
        ),
        (
            "COMPILED_SCHEMA_100_PERCENT",
            metrics["compiled_schema_valid"]["numerator"]
            == metrics["compiled_schema_valid"]["denominator"],
        ),
        (
            "COMPILED_CITATION_100_PERCENT",
            metrics["compiled_citation_exact"]["numerator"]
            == metrics["compiled_citation_exact"]["denominator"],
        ),
        (
            "COMPILED_PROVENANCE_100_PERCENT",
            metrics["compiled_provenance_exact"]["numerator"]
            == metrics["compiled_provenance_exact"]["denominator"],
        ),
        (
            "ANSWER_SPAN_EXACT_AT_LEAST_90_PERCENT",
            _ratio_at_least(
                metrics["answer_span_exact"],
                RATE_GATE_NUMERATOR,
                RATE_GATE_DENOMINATOR,
            ),
        ),
        (
            "REFUSE_F1_AT_LEAST_90_PERCENT",
            _ratio_at_least(
                _refuse_f1(metrics),
                RATE_GATE_NUMERATOR,
                RATE_GATE_DENOMINATOR,
            ),
        ),
    )
    return [name for name, passed in checks if not passed]


def _minimum_stratum(metrics: Mapping[str, Any]) -> dict[str, int]:
    strata = _require_sequence(
        metrics.get("stratified_compiled_strict"),
        field="metrics.stratified_compiled_strict",
    )
    if not strata:
        raise CanaryAcceptanceV6Error(
            "metrics.stratified_compiled_strict is empty"
        )
    ratios = [
        _ratio(
            _require_int(
                _require_mapping(item, field="stratum").get("numerator"),
                field="stratum.numerator",
            ),
            _require_int(
                _require_mapping(item, field="stratum").get("denominator"),
                field="stratum.denominator",
                minimum=1,
            ),
        )
        for item in strata
    ]
    minimum = ratios[0]
    for ratio in ratios[1:]:
        if _compare_ratio(ratio, minimum) < 0:
            minimum = ratio
    return minimum


def _ranking_metrics(candidate: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _require_mapping(candidate.get("metrics"), field="candidate.metrics")
    return {
        "minimum_stratified_strict": _minimum_stratum(metrics),
        "compiled_strict_exact": metrics["compiled_strict_exact"],
        "answer_span_exact": metrics["answer_span_exact"],
        "refuse_f1": _refuse_f1(metrics),
        "validation_loss": str(candidate["validation_loss"]),
        "epoch": candidate["epoch"],
        "seed": candidate["seed"],
    }


def _compare_candidates(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> int:
    left_rank = _ranking_metrics(left)
    right_rank = _ranking_metrics(right)
    for name in (
        "minimum_stratified_strict",
        "compiled_strict_exact",
        "answer_span_exact",
        "refuse_f1",
    ):
        comparison = _compare_ratio(left_rank[name], right_rank[name])
        if comparison:
            return -comparison
    left_loss = _parse_loss(
        left["validation_loss"], field="left.validation_loss"
    )
    right_loss = _parse_loss(
        right["validation_loss"], field="right.validation_loss"
    )
    if left_loss != right_loss:
        return -1 if left_loss < right_loss else 1
    for name in ("epoch", "seed"):
        if left[name] != right[name]:
            return -1 if left[name] < right[name] else 1
    return (
        (str(left["checkpoint_id"]) > str(right["checkpoint_id"]))
        - (str(left["checkpoint_id"]) < str(right["checkpoint_id"]))
    )


def _validate_index_boundary(index: Mapping[str, Any]) -> None:
    if (
        index.get("schema") != pointer_checkpoint_eval_v6.INDEX_SCHEMA
        or index.get("status") != EXPECTED_INDEX_STATUS
        or index.get("stage") != "canary"
    ):
        raise CanaryAcceptanceV6Error(
            "evaluation index is not a completed canary 1x6 index"
        )
    training = _require_mapping(index.get("training"), field="index.training")
    if training.get("checkpoint_count") != EXPECTED_CHECKPOINTS:
        raise CanaryAcceptanceV6Error(
            "index.training.checkpoint_count must equal 6"
        )
    dataset = _require_mapping(index.get("dataset"), field="index.dataset")
    if dataset.get("evaluated_rows_per_checkpoint") != EXPECTED_SAMPLES:
        raise CanaryAcceptanceV6Error(
            "index dataset must bind 18 rows per checkpoint"
        )
    for name in (
        "calibration_content_read",
        "calibration_content_hashed",
        "blind_test_content_read",
        "blind_test_content_hashed",
    ):
        if _require_bool(dataset.get(name), field=f"index.dataset.{name}"):
            raise CanaryAcceptanceV6Error(
                f"index.dataset.{name} must remain false"
            )
    execution = _require_mapping(
        index.get("execution"), field="index.execution"
    )
    if (
        execution.get("split") != "validation"
        or execution.get("summary_metrics_trusted") is not False
        or execution.get("selection_policy_invoked") is not False
        or execution.get("checkpoint_selected") is not False
        or execution.get("freeze_created") is not False
    ):
        raise CanaryAcceptanceV6Error("index execution boundary is invalid")
    selection = _require_mapping(
        index.get("selection"), field="index.selection"
    )
    if (
        selection.get("performed") is not False
        or selection.get("selected_checkpoint_id") is not None
    ):
        raise CanaryAcceptanceV6Error(
            "canary index must not contain a prior selection"
        )
    authorization = _require_mapping(
        index.get("authorization"), field="index.authorization"
    )
    for name, value in authorization.items():
        if not isinstance(value, bool) or value:
            raise CanaryAcceptanceV6Error(
                f"index.authorization.{name} must remain false"
            )


def _resolve_child_directory(
    root: Path, value: Any, *, checkpoint_id: str
) -> Path:
    text = _require_text(value, field=f"{checkpoint_id}.evaluation_directory")
    raw = Path(text)
    if raw.is_symlink():
        raise CanaryAcceptanceV6Error(
            f"{checkpoint_id}.evaluation_directory must not be a symlink"
        )
    try:
        resolved = raw.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise CanaryAcceptanceV6Error(
            f"{checkpoint_id}.evaluation_directory must stay under index root"
        ) from exc
    if not resolved.is_dir():
        raise CanaryAcceptanceV6Error(
            f"{checkpoint_id}.evaluation_directory must be a directory"
        )
    return resolved


def _revalidate_inputs(
    index_path: Path,
    index_payload: bytes,
    artifacts: Sequence[Mapping[str, Any]],
) -> None:
    _, current_index = _stable_bytes(index_path, field="evaluation index")
    if current_index != index_payload:
        raise CanaryAcceptanceV6Error(
            "evaluation index changed during acceptance"
        )
    for item in artifacts:
        path = Path(str(item["path"]))
        _, payload = _stable_bytes(path, field=str(item["role"]))
        if _sha256_bytes(payload) != item["sha256"]:
            raise CanaryAcceptanceV6Error(
                f"{item['role']} changed during acceptance"
            )


def _write_exclusive(path: Path, payload: bytes) -> Path:
    output = Path(path)
    if output.exists() or output.is_symlink():
        raise CanaryAcceptanceV6Error(f"output already exists: {output}")
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink():
        raise CanaryAcceptanceV6Error(
            f"output parent must not be a symlink: {parent}"
        )
    resolved_parent = parent.resolve(strict=True)
    target = resolved_parent / output.name
    created = False
    try:
        with target.open("xb") as handle:
            created = True
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise CanaryAcceptanceV6Error(
            f"output already exists: {target}"
        ) from exc
    except Exception:
        if created:
            try:
                target.unlink()
            except OSError:
                pass
        raise
    return target


def record_canary_acceptance(
    *,
    evaluation_index_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Record one immutable v6 canary acceptance or stop receipt."""

    index_path, index_payload, index = _load_json(
        Path(evaluation_index_path), field="evaluation index"
    )
    if index_path.name != "evaluation_index.v6.json":
        raise CanaryAcceptanceV6Error(
            "evaluation index filename must be evaluation_index.v6.json"
        )
    _validate_index_boundary(index)
    root = index_path.parent.resolve(strict=True)
    checkpoint_items = _require_sequence(
        index.get("checkpoints"), field="index.checkpoints"
    )
    record_items = _require_sequence(
        index.get("records"), field="index.records"
    )
    if (
        len(checkpoint_items) != EXPECTED_CHECKPOINTS
        or len(record_items) != EXPECTED_CHECKPOINTS
    ):
        raise CanaryAcceptanceV6Error(
            "canary index must contain exactly six checkpoints and six records"
        )

    checkpoint_by_id: dict[str, Mapping[str, Any]] = {}
    for item in checkpoint_items:
        checkpoint = _require_mapping(item, field="index.checkpoints[]")
        checkpoint_id = _require_text(
            checkpoint.get("checkpoint_id"), field="checkpoint.checkpoint_id"
        )
        if checkpoint_id in checkpoint_by_id:
            raise CanaryAcceptanceV6Error(
                f"duplicate checkpoint_id: {checkpoint_id}"
            )
        checkpoint_by_id[checkpoint_id] = checkpoint
    record_by_id: dict[str, Mapping[str, Any]] = {}
    for item in record_items:
        record = _require_mapping(item, field="index.records[]")
        checkpoint_id = _require_text(
            record.get("checkpoint_id"), field="record.checkpoint_id"
        )
        if checkpoint_id in record_by_id:
            raise CanaryAcceptanceV6Error(
                f"duplicate record checkpoint_id: {checkpoint_id}"
            )
        record_by_id[checkpoint_id] = record
    if set(checkpoint_by_id) != set(record_by_id):
        raise CanaryAcceptanceV6Error(
            "checkpoint evidence and metric record IDs differ"
        )

    artifacts_read: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    reference_ids: list[str] | None = None
    observed_directories: set[Path] = set()
    observed_seed_epochs: set[tuple[int, int]] = set()
    observed_seeds: set[int] = set()
    for checkpoint_id in sorted(checkpoint_by_id):
        checkpoint = checkpoint_by_id[checkpoint_id]
        record = record_by_id[checkpoint_id]
        seed = _require_int(checkpoint.get("seed"), field=f"{checkpoint_id}.seed", minimum=1)
        epoch = _require_int(
            checkpoint.get("epoch"),
            field=f"{checkpoint_id}.epoch",
            minimum=1,
            maximum=6,
        )
        if (
            record.get("seed") != seed
            or record.get("epoch") != epoch
            or record.get("checkpoint_id") != checkpoint_id
        ):
            raise CanaryAcceptanceV6Error(
                f"{checkpoint_id} record identity differs from checkpoint evidence"
            )
        pair = (seed, epoch)
        if pair in observed_seed_epochs:
            raise CanaryAcceptanceV6Error(
                f"duplicate canary seed/epoch pair: {seed}/{epoch}"
            )
        observed_seed_epochs.add(pair)
        observed_seeds.add(seed)
        validation_loss = _parse_loss(
            checkpoint.get("validation_loss"),
            field=f"{checkpoint_id}.validation_loss",
        )
        if _parse_loss(
            record.get("validation_loss"),
            field=f"{checkpoint_id}.record.validation_loss",
        ) != validation_loss:
            raise CanaryAcceptanceV6Error(
                f"{checkpoint_id} validation loss binding differs"
            )

        directory = _resolve_child_directory(
            root,
            checkpoint.get("evaluation_directory"),
            checkpoint_id=checkpoint_id,
        )
        if directory in observed_directories:
            raise CanaryAcceptanceV6Error(
                f"multiple checkpoints share evaluation directory {directory}"
            )
        observed_directories.add(directory)
        hashes = _require_mapping(
            checkpoint.get("evaluation_artifacts"),
            field=f"{checkpoint_id}.evaluation_artifacts",
        )
        payloads: dict[str, bytes] = {}
        for name in sorted(READABLE_CHILD_ARTIFACTS):
            expected_hash = _require_sha256(
                hashes.get(name),
                field=f"{checkpoint_id}.evaluation_artifacts.{name}",
            )
            path, payload = _stable_bytes(
                directory / name,
                field=f"{checkpoint_id} {name}",
            )
            actual_hash = _sha256_bytes(payload)
            if actual_hash != expected_hash:
                raise CanaryAcceptanceV6Error(
                    f"{checkpoint_id} {name} SHA-256 differs from index"
                )
            payloads[name] = payload
            artifacts_read.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "role": f"{checkpoint_id}:{name}",
                    "path": str(path),
                    "bytes": len(payload),
                    "sha256": actual_hash,
                }
            )

        rows = _load_jsonl_bytes(
            payloads["sample_results.v6.jsonl"],
            field=f"{checkpoint_id} sample_results.v6.jsonl",
        )
        metrics, audit = _recompute_checkpoint(
            rows, checkpoint_id=checkpoint_id
        )
        current_ids = list(audit["example_ids"])
        if reference_ids is None:
            reference_ids = current_ids
        elif current_ids != reference_ids:
            raise CanaryAcceptanceV6Error(
                "all checkpoints must evaluate the identical 18 example IDs"
            )
        index_metrics = _require_mapping(
            record.get("metrics"), field=f"{checkpoint_id}.record.metrics"
        )
        if dict(index_metrics) != metrics:
            raise CanaryAcceptanceV6Error(
                f"{checkpoint_id} index metrics differ from independent recomputation"
            )
        summary = _load_json_bytes(
            payloads["summary.v6.json"],
            field=f"{checkpoint_id} summary.v6.json",
        )
        _validate_summary(
            summary,
            metrics=metrics,
            audit=audit,
            checkpoint_id=checkpoint_id,
        )
        failed = _failed_gates(metrics)
        candidates.append(
            {
                "checkpoint_id": checkpoint_id,
                "seed": seed,
                "epoch": epoch,
                "validation_loss": str(validation_loss),
                "qualified": not failed,
                "failed_gates": failed,
                "metrics": metrics,
                "ranking_metrics": {
                    "minimum_stratified_strict": _minimum_stratum(metrics),
                    "compiled_strict_exact": metrics["compiled_strict_exact"],
                    "answer_span_exact": metrics["answer_span_exact"],
                    "refuse_f1": _refuse_f1(metrics),
                    "validation_loss": str(validation_loss),
                    "epoch": epoch,
                    "seed": seed,
                },
            }
        )

    if len(observed_seeds) != 1 or {
        epoch for _, epoch in observed_seed_epochs
    } != EXPECTED_EPOCHS:
        raise CanaryAcceptanceV6Error(
            "canary population must be one seed with epochs 1 through 6"
        )
    _revalidate_inputs(index_path, index_payload, artifacts_read)

    qualified = [candidate for candidate in candidates if candidate["qualified"]]
    ordered = sorted(qualified, key=cmp_to_key(_compare_candidates))
    reference = ordered[0] if ordered else None
    gate_passed = reference is not None
    status = PASS_STATUS if gate_passed else STOP_STATUS
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "gate_version": VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "gate_passed": gate_passed,
        "next_action": (
            "START_FINAL_THREE_SEED_TRAINING"
            if gate_passed
            else "STOP_AND_REVIEW_NONBLIND_CANARY"
        ),
        "input": {
            "evaluation_index": {
                "path": str(index_path),
                "bytes": len(index_payload),
                "sha256": _sha256_bytes(index_payload),
            },
            "checkpoint_artifacts_read": artifacts_read,
            "checkpoint_run_receipts_read": False,
            "training_receipt_read": False,
            "calibration_content_read": False,
            "calibration_content_hashed": False,
            "blind_test_content_read": False,
            "blind_test_content_hashed": False,
        },
        "thresholds": {
            "completed_samples": EXPECTED_SAMPLES,
            "pointer_schema_rate": "1/1",
            "pointer_invalid_count_max": 0,
            "pointer_ambiguous_count_max": 0,
            "pointer_out_of_range_count_max": 0,
            "unsupported_wrong_answer_count_max": 0,
            "compiled_schema_rate": "1/1",
            "compiled_citation_exact_rate": "1/1",
            "compiled_provenance_exact_rate": "1/1",
            "answer_span_exact_min": "9/10",
            "refuse_f1_min": "9/10",
        },
        "independent_recomputation": {
            "checkpoint_count": len(candidates),
            "samples_per_checkpoint": EXPECTED_SAMPLES,
            "summary_metrics_trusted": False,
            "index_metrics_trusted": False,
            "all_index_and_summary_metrics_reconciled": True,
            "checkpoints": candidates,
        },
        "deterministic_advancement_reference": (
            None
            if reference is None
            else {
                "checkpoint_id": reference["checkpoint_id"],
                "seed": reference["seed"],
                "epoch": reference["epoch"],
                "ranking_metrics": reference["ranking_metrics"],
                "purpose": "THREE_SEED_TRAINING_ADVANCEMENT_EVIDENCE_ONLY",
                "is_final_model_selection": False,
            }
        ),
        "authorization": {
            "three_seed_training_authorized": gate_passed,
            "checkpoint_selected_as_final_model": False,
            "model_authorized": False,
            "calibration_authorized": False,
            "blind_test_authorized": False,
            "gguf_export_authorized": False,
            "x5_deployment_authorized": False,
            "production_integration_authorized": False,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    receipt["receipt_payload_sha256"] = _sha256_bytes(
        _canonical_json(receipt).encode("utf-8")
    )
    payload = _pretty_bytes(receipt)
    written = _write_exclusive(Path(output_path), payload)
    return {
        "status": status,
        "gate_passed": gate_passed,
        "path": str(written),
        "sha256": _sha256_bytes(payload),
        "canonical_digest_sha256": receipt["receipt_payload_sha256"],
        "advancement_reference_checkpoint_id": (
            None if reference is None else reference["checkpoint_id"]
        ),
        "three_seed_training_authorized": gate_passed,
        "final_model_selected": False,
        "deployment_authorized": False,
    }


__all__ = [
    "ERROR_STATUS",
    "PASS_STATUS",
    "SCHEMA",
    "STOP_STATUS",
    "VERSION",
    "CanaryAcceptanceV6Error",
    "record_canary_acceptance",
]
