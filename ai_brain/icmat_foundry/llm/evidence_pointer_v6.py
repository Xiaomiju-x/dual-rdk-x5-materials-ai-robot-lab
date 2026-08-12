"""Deterministic Evidence Pointer Compiler and fixture-only CPU evaluation.

The v6 model contract is intentionally narrow: a model may emit only
``task``, ``decision`` and ``span_id``. This module validates that pointer
against target-free prompt metadata and an explicit sentence index, then
compiles the existing seven-field ICMat answer contract.

Gold answers are not accepted by :func:`compile_pointer`. Evaluation reads
expected values only after compilation and uses them exclusively for scoring.
This module has no model runtime, GPU, dataset, or blind-test dependency.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

POINTER_SCHEMA = "icmat_evidence_pointer.v6"
PROMPT_SCHEMA = "icmat_evidence_pointer_prompt.v6"
FIXTURE_SCHEMA = "icmat_evidence_pointer_fixture.v6"
ANSWER_SCHEMA = "icmat_student_answer.v6"
COMPILATION_SCHEMA = "icmat_evidence_pointer_compilation.v6"
SAMPLE_RESULT_SCHEMA = "icmat_evidence_pointer_eval_sample.v6"
SUMMARY_SCHEMA = "icmat_evidence_pointer_eval_summary.v6"
RUN_RECEIPT_SCHEMA = "icmat_evidence_pointer_eval_run_receipt.v6"
COMPILER_VERSION = "icmat-evidence-pointer-compiler-v6.1.0"

POINTER_KEYS = frozenset({"task", "decision", "span_id"})
PROMPT_KEYS = frozenset({"schema", "task", "messages", "response_provenance"})
EVIDENCE_KEYS = frozenset({"evidence_id", "sentences", "provenance"})
SENTENCE_KEYS = frozenset({"span_id", "text"})
MODEL_OUTPUT_KEYS = frozenset({"raw_pointer", "finish_reason"})
FIXTURE_KEYS = frozenset(
    {
        "schema",
        "example_id",
        "prompt",
        "evidence",
        "model_output",
        "expected_pointer",
        "expected_answer",
    }
)
ANSWER_KEYS = frozenset(
    {
        "schema",
        "decision",
        "task",
        "claim",
        "verdict",
        "evidence_ids",
        "provenance",
    }
)
PROVENANCE_KEYS = frozenset(
    {
        "source_id",
        "doi",
        "source_title",
        "license_id",
        "measurement_status",
    }
)

ALLOWED_DECISIONS = frozenset({"ANSWER", "REFUSE"})
ALLOWED_VERDICTS = frozenset({"SUPPORTED", "REFUSED"})
TRUSTED_FINISH_REASONS = frozenset({"stop", "eos_token", "end_turn"})
MEASUREMENT_STATUS = "published_literature_not_local_measurement"

EVIDENCE_ID_RE = re.compile(r"^E[1-9][0-9]*$")
SPAN_ID_RE = re.compile(r"^(?P<evidence>E[1-9][0-9]*)\.S[1-9][0-9]*$")

MAX_FIXTURE_BYTES = 16 * 1024 * 1024
MAX_EVIDENCE_ITEMS = 64
MAX_SENTENCES_PER_EVIDENCE = 256
MAX_POINTER_BYTES = 64 * 1024


class EvidencePointerV6Error(ValueError):
    """Raised when a trusted fixture or evaluation contract is invalid."""


class _CompilerViolation(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


def canonical_json(value: Any) -> str:
    """Return the canonical JSON representation used for evidence hashes."""

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


def _jsonl_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (canonical_json(dict(record)) + "\n").encode("utf-8") for record in records
    )


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _CompilerViolation(
                "POINTER_DUPLICATE_KEY",
                f"duplicate JSON key: {key}",
            )
        result[key] = value
    return result


def _parse_single_json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise _CompilerViolation(
            "POINTER_EMPTY",
            "raw pointer must be a non-empty JSON object",
        )
    if len(text.encode("utf-8")) > MAX_POINTER_BYTES:
        raise _CompilerViolation(
            "POINTER_TOO_LARGE",
            f"raw pointer exceeds {MAX_POINTER_BYTES} bytes",
        )
    decoder = json.JSONDecoder(object_pairs_hook=_reject_duplicate_pairs)
    start = len(text) - len(text.lstrip())
    try:
        value, end = decoder.raw_decode(text, idx=start)
    except _CompilerViolation:
        raise
    except json.JSONDecodeError as exc:
        raise _CompilerViolation(
            "POINTER_JSON_INVALID",
            f"raw pointer is not one JSON object: {exc}",
        ) from exc
    if text[end:].strip():
        raise _CompilerViolation(
            "POINTER_TRAILING_CONTENT",
            "raw pointer has trailing content after the JSON object",
        )
    if not isinstance(value, dict):
        raise _CompilerViolation(
            "POINTER_ROOT_NOT_OBJECT",
            "raw pointer JSON root must be an object",
        )
    return value


def _finite_json_clone(value: Any, field: str) -> Any:
    try:
        return json.loads(canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise _CompilerViolation(
            "NON_FINITE_JSON",
            f"{field} must contain only finite JSON data",
        ) from exc


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    field: str,
    code: str,
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise _CompilerViolation(
            code,
            f"{field} keys mismatch; missing={missing}, extra={extra}",
        )


def _require_non_empty_string(value: Any, *, field: str, code: str) -> str:
    if not isinstance(value, str) or not value:
        raise _CompilerViolation(code, f"{field} must be a non-empty string")
    return value


def _validate_provenance(value: Any, *, field: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise _CompilerViolation(
            "PROVENANCE_INVALID",
            f"{field} must be an object",
        )
    _require_exact_keys(
        value,
        PROVENANCE_KEYS,
        field=field,
        code="PROVENANCE_KEYS_MISMATCH",
    )
    normalized: dict[str, str] = {}
    for key in sorted(PROVENANCE_KEYS):
        normalized[key] = _require_non_empty_string(
            value.get(key),
            field=f"{field}.{key}",
            code="PROVENANCE_VALUE_INVALID",
        )
    if normalized["measurement_status"] != MEASUREMENT_STATUS:
        raise _CompilerViolation(
            "PROVENANCE_MEASUREMENT_STATUS_INVALID",
            f"{field}.measurement_status must be {MEASUREMENT_STATUS}",
        )
    return normalized


def _validate_prompt(prompt: Any) -> tuple[dict[str, Any], str, dict[str, str]]:
    if not isinstance(prompt, Mapping):
        raise _CompilerViolation(
            "PROMPT_INVALID",
            "prompt must be an object",
        )
    _require_exact_keys(
        prompt,
        PROMPT_KEYS,
        field="prompt",
        code="PROMPT_KEYS_MISMATCH",
    )
    if prompt.get("schema") != PROMPT_SCHEMA:
        raise _CompilerViolation(
            "PROMPT_SCHEMA_INVALID",
            f"prompt.schema must be {PROMPT_SCHEMA}",
        )
    task = _require_non_empty_string(
        prompt.get("task"),
        field="prompt.task",
        code="PROMPT_TASK_INVALID",
    )
    messages = prompt.get("messages")
    if (
        not isinstance(messages, Sequence)
        or isinstance(messages, (str, bytes))
        or len(messages) != 2
    ):
        raise _CompilerViolation(
            "PROMPT_MESSAGES_INVALID",
            "prompt.messages must contain exactly system and user messages",
        )
    normalized_messages: list[dict[str, str]] = []
    for index, (message, expected_role) in enumerate(
        zip(messages, ("system", "user"), strict=True)
    ):
        if not isinstance(message, Mapping):
            raise _CompilerViolation(
                "PROMPT_MESSAGE_INVALID",
                f"prompt.messages[{index}] must be an object",
            )
        _require_exact_keys(
            message,
            frozenset({"role", "content"}),
            field=f"prompt.messages[{index}]",
            code="PROMPT_MESSAGE_KEYS_MISMATCH",
        )
        if message.get("role") != expected_role:
            raise _CompilerViolation(
                "PROMPT_ROLE_INVALID",
                f"prompt.messages[{index}].role must be {expected_role}",
            )
        content = _require_non_empty_string(
            message.get("content"),
            field=f"prompt.messages[{index}].content",
            code="PROMPT_CONTENT_INVALID",
        )
        normalized_messages.append({"role": expected_role, "content": content})
    provenance = _validate_provenance(
        prompt.get("response_provenance"),
        field="prompt.response_provenance",
    )
    normalized = {
        "schema": PROMPT_SCHEMA,
        "task": task,
        "messages": normalized_messages,
        "response_provenance": provenance,
    }
    return normalized, task, provenance


def _validate_evidence(
    evidence: Any,
    *,
    response_provenance: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    if (
        not isinstance(evidence, Sequence)
        or isinstance(evidence, (str, bytes))
        or not evidence
    ):
        raise _CompilerViolation(
            "EVIDENCE_INVALID",
            "evidence must be a non-empty array",
        )
    if len(evidence) > MAX_EVIDENCE_ITEMS:
        raise _CompilerViolation(
            "EVIDENCE_TOO_LARGE",
            f"evidence contains more than {MAX_EVIDENCE_ITEMS} items",
        )
    normalized: list[dict[str, Any]] = []
    evidence_ids: set[str] = set()
    span_index: dict[str, list[dict[str, Any]]] = {}
    for evidence_index, item in enumerate(evidence):
        field = f"evidence[{evidence_index}]"
        if not isinstance(item, Mapping):
            raise _CompilerViolation(
                "EVIDENCE_ITEM_INVALID",
                f"{field} must be an object",
            )
        _require_exact_keys(
            item,
            EVIDENCE_KEYS,
            field=field,
            code="EVIDENCE_KEYS_MISMATCH",
        )
        evidence_id = _require_non_empty_string(
            item.get("evidence_id"),
            field=f"{field}.evidence_id",
            code="EVIDENCE_ID_INVALID",
        )
        if not EVIDENCE_ID_RE.fullmatch(evidence_id):
            raise _CompilerViolation(
                "EVIDENCE_ID_INVALID",
                f"{field}.evidence_id must match E#",
            )
        if evidence_id in evidence_ids:
            raise _CompilerViolation(
                "AMBIGUOUS_EVIDENCE_ID",
                f"duplicate evidence_id: {evidence_id}",
            )
        evidence_ids.add(evidence_id)
        provenance = _validate_provenance(
            item.get("provenance"),
            field=f"{field}.provenance",
        )
        if provenance != dict(response_provenance):
            raise _CompilerViolation(
                "PROVENANCE_MISMATCH",
                f"{field}.provenance does not match prompt.response_provenance",
            )
        sentences = item.get("sentences")
        if (
            not isinstance(sentences, Sequence)
            or isinstance(sentences, (str, bytes))
            or not sentences
        ):
            raise _CompilerViolation(
                "SENTENCES_INVALID",
                f"{field}.sentences must be a non-empty array",
            )
        if len(sentences) > MAX_SENTENCES_PER_EVIDENCE:
            raise _CompilerViolation(
                "SENTENCES_TOO_LARGE",
                f"{field}.sentences exceeds {MAX_SENTENCES_PER_EVIDENCE} items",
            )
        normalized_sentences: list[dict[str, str]] = []
        for sentence_index, sentence in enumerate(sentences):
            sentence_field = f"{field}.sentences[{sentence_index}]"
            if not isinstance(sentence, Mapping):
                raise _CompilerViolation(
                    "SENTENCE_INVALID",
                    f"{sentence_field} must be an object",
                )
            _require_exact_keys(
                sentence,
                SENTENCE_KEYS,
                field=sentence_field,
                code="SENTENCE_KEYS_MISMATCH",
            )
            span_id = _require_non_empty_string(
                sentence.get("span_id"),
                field=f"{sentence_field}.span_id",
                code="SPAN_ID_INVALID",
            )
            match = SPAN_ID_RE.fullmatch(span_id)
            if match is None:
                raise _CompilerViolation(
                    "SPAN_ID_INVALID",
                    f"{sentence_field}.span_id must match E#.S#",
                )
            if match.group("evidence") != evidence_id:
                raise _CompilerViolation(
                    "SPAN_EVIDENCE_MISMATCH",
                    f"{span_id} does not belong to {evidence_id}",
                )
            text = _require_non_empty_string(
                sentence.get("text"),
                field=f"{sentence_field}.text",
                code="SPAN_TEXT_INVALID",
            )
            if text != text.strip():
                raise _CompilerViolation(
                    "SPAN_TEXT_BOUNDARY_INVALID",
                    f"{span_id} has leading or trailing whitespace",
                )
            normalized_sentence = {"span_id": span_id, "text": text}
            normalized_sentences.append(normalized_sentence)
            span_index.setdefault(span_id, []).append(
                {
                    "span_id": span_id,
                    "text": text,
                    "evidence_id": evidence_id,
                    "provenance": provenance,
                }
            )
        normalized.append(
            {
                "evidence_id": evidence_id,
                "sentences": normalized_sentences,
                "provenance": provenance,
            }
        )
    duplicate_spans = sorted(
        span_id for span_id, entries in span_index.items() if len(entries) != 1
    )
    if duplicate_spans:
        raise _CompilerViolation(
            "AMBIGUOUS_SPAN_ID",
            f"duplicate span IDs: {duplicate_spans}",
        )
    return normalized, span_index


def _normalize_raw_pointer(raw_pointer: Any) -> tuple[Any, str]:
    if isinstance(raw_pointer, str):
        return raw_pointer, raw_pointer
    if isinstance(raw_pointer, Mapping):
        try:
            normalized = _finite_json_clone(dict(raw_pointer), "raw_pointer")
            return normalized, canonical_json(normalized)
        except _CompilerViolation:
            return {
                "unserializable_type": type(raw_pointer).__name__,
            }, f"<unserializable:{type(raw_pointer).__name__}>"
    return {
        "invalid_type": type(raw_pointer).__name__,
    }, f"<invalid-type:{type(raw_pointer).__name__}>"


def _parse_pointer(raw_pointer: Any) -> dict[str, Any]:
    if isinstance(raw_pointer, str):
        value = _parse_single_json_object(raw_pointer)
    elif isinstance(raw_pointer, Mapping):
        value = _finite_json_clone(dict(raw_pointer), "raw_pointer")
    else:
        raise _CompilerViolation(
            "POINTER_TYPE_INVALID",
            "raw pointer must be a JSON string or object",
        )
    _require_exact_keys(
        value,
        POINTER_KEYS,
        field="pointer",
        code="POINTER_KEYS_MISMATCH",
    )
    task = _require_non_empty_string(
        value.get("task"),
        field="pointer.task",
        code="POINTER_TASK_INVALID",
    )
    decision = value.get("decision")
    if decision not in ALLOWED_DECISIONS:
        raise _CompilerViolation(
            "POINTER_DECISION_INVALID",
            "pointer.decision must be ANSWER or REFUSE",
        )
    span_id = value.get("span_id")
    if decision == "ANSWER":
        if not isinstance(span_id, str) or not SPAN_ID_RE.fullmatch(span_id):
            raise _CompilerViolation(
                "POINTER_SPAN_INVALID",
                "ANSWER pointer.span_id must match E#.S#",
            )
    elif span_id is not None:
        raise _CompilerViolation(
            "POINTER_REFUSE_SPAN_NOT_NULL",
            "REFUSE pointer.span_id must be null",
        )
    return {
        "task": task,
        "decision": decision,
        "span_id": span_id,
    }


def _safe_refusal(task: str, provenance: Mapping[str, str]) -> dict[str, Any]:
    return {
        "schema": ANSWER_SCHEMA,
        "decision": "REFUSE",
        "task": task,
        "claim": "",
        "verdict": "REFUSED",
        "evidence_ids": [],
        "provenance": dict(provenance),
    }


def validate_student_answer(value: Mapping[str, Any]) -> list[str]:
    """Return v5-compatible seven-field schema violations."""

    errors: list[str] = []
    keys = set(value)
    if keys != ANSWER_KEYS:
        errors.append(
            "target keys mismatch; "
            f"missing={sorted(ANSWER_KEYS - keys)}, extra={sorted(keys - ANSWER_KEYS)}"
        )
    if value.get("schema") != ANSWER_SCHEMA:
        errors.append(f"schema must equal {ANSWER_SCHEMA}")
    decision = value.get("decision")
    if decision not in ALLOWED_DECISIONS:
        errors.append("decision must be ANSWER or REFUSE")
    if not isinstance(value.get("task"), str) or not value.get("task"):
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
    elif (
        any(not isinstance(item, str) or not item for item in evidence_ids)
        or len(set(evidence_ids)) != len(evidence_ids)
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
    if not isinstance(provenance, Mapping):
        errors.append("provenance must be an object")
    else:
        try:
            canonical_json(provenance)
        except (TypeError, ValueError):
            errors.append("provenance must contain only finite JSON data")
    return errors


def _result_with_hash(result: dict[str, Any]) -> dict[str, Any]:
    result["compilation_sha256"] = sha256_bytes(
        canonical_json(result).encode("utf-8")
    )
    return result


def compile_pointer(
    *,
    prompt: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    raw_pointer: str | Mapping[str, Any],
    finish_reason: str | None,
) -> dict[str, Any]:
    """Compile one model pointer without accepting or inspecting any gold data."""

    raw_evidence, raw_text = _normalize_raw_pointer(raw_pointer)
    result: dict[str, Any] = {
        "schema": COMPILATION_SCHEMA,
        "compiler_version": COMPILER_VERSION,
        "status": "FAIL_CLOSED",
        "fail_closed": True,
        "input_trusted": False,
        "raw_pointer": raw_evidence,
        "raw_pointer_sha256": sha256_bytes(raw_text.encode("utf-8")),
        "finish_reason": finish_reason,
        "parse_reason": {
            "code": "UNPROCESSED",
            "detail": "compiler did not process the pointer",
        },
        "parsed_pointer": None,
        "compiler_decision": "FAIL_CLOSED",
        "selected_span_id": None,
        "selected_evidence_id": None,
        "compiled_answer": None,
        "prompt_sha256": None,
        "evidence_sha256": None,
        "contract_trace": {
            "gold_input_accepted": False,
            "assistant_target_visible": False,
            "claim_source": "none",
            "provenance_source": "none",
        },
    }
    task: str | None = None
    response_provenance: dict[str, str] | None = None
    try:
        normalized_prompt, task, response_provenance = _validate_prompt(prompt)
        result["prompt_sha256"] = sha256_bytes(
            canonical_json(normalized_prompt).encode("utf-8")
        )
    except _CompilerViolation as exc:
        result["parse_reason"] = {"code": exc.code, "detail": exc.detail}
        return _result_with_hash(result)

    safe_refusal = _safe_refusal(task, response_provenance)
    result["compiled_answer"] = safe_refusal
    result["contract_trace"]["provenance_source"] = (
        "validated_prompt.response_provenance"
    )
    try:
        normalized_evidence, span_index = _validate_evidence(
            evidence,
            response_provenance=response_provenance,
        )
        result["evidence_sha256"] = sha256_bytes(
            canonical_json(normalized_evidence).encode("utf-8")
        )
        result["input_trusted"] = True
    except _CompilerViolation as exc:
        result["parse_reason"] = {"code": exc.code, "detail": exc.detail}
        return _result_with_hash(result)

    try:
        pointer = _parse_pointer(raw_pointer)
        result["parsed_pointer"] = pointer
        if pointer["task"] != task:
            raise _CompilerViolation(
                "POINTER_TASK_MISMATCH",
                "pointer.task does not match prompt.task",
            )
        if finish_reason not in TRUSTED_FINISH_REASONS:
            raise _CompilerViolation(
                "FINISH_REASON_UNTRUSTED",
                "finish_reason must prove a normal model stop",
            )
        if pointer["decision"] == "REFUSE":
            result.update(
                {
                    "status": "COMPILED",
                    "fail_closed": False,
                    "parse_reason": {"code": "OK", "detail": "pointer compiled"},
                    "compiler_decision": "REFUSE",
                    "compiled_answer": safe_refusal,
                }
            )
            return _result_with_hash(result)

        span_id = str(pointer["span_id"])
        entries = span_index.get(span_id, [])
        if not entries:
            raise _CompilerViolation(
                "SPAN_NOT_FOUND",
                f"pointer span_id is outside the evidence index: {span_id}",
            )
        if len(entries) != 1:
            raise _CompilerViolation(
                "AMBIGUOUS_SPAN_ID",
                f"pointer span_id resolves to {len(entries)} entries: {span_id}",
            )
        selected = entries[0]
        compiled_answer = {
            "schema": ANSWER_SCHEMA,
            "decision": "ANSWER",
            "task": task,
            "claim": selected["text"],
            "verdict": "SUPPORTED",
            "evidence_ids": [selected["evidence_id"]],
            "provenance": dict(selected["provenance"]),
        }
        schema_errors = validate_student_answer(compiled_answer)
        if schema_errors:
            raise _CompilerViolation(
                "COMPILED_ANSWER_INVALID",
                f"compiled answer violated the external contract: {schema_errors}",
            )
        result.update(
            {
                "status": "COMPILED",
                "fail_closed": False,
                "parse_reason": {"code": "OK", "detail": "pointer compiled"},
                "compiler_decision": "ANSWER",
                "selected_span_id": span_id,
                "selected_evidence_id": selected["evidence_id"],
                "compiled_answer": compiled_answer,
                "contract_trace": {
                    "gold_input_accepted": False,
                    "assistant_target_visible": False,
                    "claim_source": f"evidence[{span_id}].text_verbatim",
                    "provenance_source": (
                        f"evidence[{selected['evidence_id']}].provenance_validated"
                    ),
                },
            }
        )
    except _CompilerViolation as exc:
        result["parse_reason"] = {"code": exc.code, "detail": exc.detail}
    return _result_with_hash(result)


def _validate_fixture_record(record: Any, index: int) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise EvidencePointerV6Error(f"fixture record {index} must be an object")
    if set(record) != FIXTURE_KEYS:
        raise EvidencePointerV6Error(
            f"fixture record {index} keys mismatch; "
            f"missing={sorted(FIXTURE_KEYS - set(record))}, "
            f"extra={sorted(set(record) - FIXTURE_KEYS)}"
        )
    if record.get("schema") != FIXTURE_SCHEMA:
        raise EvidencePointerV6Error(
            f"fixture record {index} schema must be {FIXTURE_SCHEMA}"
        )
    example_id = record.get("example_id")
    if not isinstance(example_id, str) or not example_id:
        raise EvidencePointerV6Error(
            f"fixture record {index} example_id must be a non-empty string"
        )
    model_output = record.get("model_output")
    if not isinstance(model_output, Mapping) or set(model_output) != MODEL_OUTPUT_KEYS:
        raise EvidencePointerV6Error(
            f"fixture record {index} model_output keys must be "
            f"{sorted(MODEL_OUTPUT_KEYS)}"
        )
    return dict(record)


def _expected_from_pointer(
    *,
    prompt: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    expected_pointer: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        normalized_prompt, task, provenance = _validate_prompt(prompt)
        normalized_evidence, span_index = _validate_evidence(
            evidence,
            response_provenance=provenance,
        )
        pointer = _parse_pointer(expected_pointer)
        if pointer["task"] != task:
            raise _CompilerViolation(
                "EXPECTED_POINTER_TASK_MISMATCH",
                "expected pointer task does not match prompt task",
            )
        if pointer["decision"] == "REFUSE":
            answer = _safe_refusal(task, provenance)
        else:
            entries = span_index.get(str(pointer["span_id"]), [])
            if len(entries) != 1:
                raise _CompilerViolation(
                    "EXPECTED_POINTER_SPAN_INVALID",
                    "expected pointer must resolve to exactly one sentence",
                )
            selected = entries[0]
            answer = {
                "schema": ANSWER_SCHEMA,
                "decision": "ANSWER",
                "task": task,
                "claim": selected["text"],
                "verdict": "SUPPORTED",
                "evidence_ids": [selected["evidence_id"]],
                "provenance": dict(selected["provenance"]),
            }
        return pointer, {
            "prompt_sha256": sha256_bytes(
                canonical_json(normalized_prompt).encode("utf-8")
            ),
            "evidence_sha256": sha256_bytes(
                canonical_json(normalized_evidence).encode("utf-8")
            ),
            "deterministic_answer": answer,
        }
    except _CompilerViolation as exc:
        raise EvidencePointerV6Error(
            f"invalid expected pointer fixture: {exc.code}: {exc.detail}"
        ) from exc


def _score_fixture_record(record: Mapping[str, Any], index: int) -> dict[str, Any]:
    checked = _validate_fixture_record(record, index)
    model_output = checked["model_output"]

    # Compilation happens before expected fields are inspected. The compiler
    # signature cannot accept a target, expected answer, or gold pointer.
    compilation = compile_pointer(
        prompt=checked["prompt"],
        evidence=checked["evidence"],
        raw_pointer=model_output["raw_pointer"],
        finish_reason=model_output["finish_reason"],
    )

    expected_pointer, expected_context = _expected_from_pointer(
        prompt=checked["prompt"],
        evidence=checked["evidence"],
        expected_pointer=checked["expected_pointer"],
    )
    expected_answer = checked["expected_answer"]
    if not isinstance(expected_answer, Mapping):
        raise EvidencePointerV6Error(
            f"fixture record {index} expected_answer must be an object"
        )
    try:
        expected_answer = _finite_json_clone(
            dict(expected_answer),
            f"fixture[{index}].expected_answer",
        )
    except _CompilerViolation as exc:
        raise EvidencePointerV6Error(
            f"fixture record {index} expected_answer is invalid: "
            f"{exc.code}: {exc.detail}"
        ) from exc
    expected_errors = validate_student_answer(expected_answer)
    if expected_errors:
        raise EvidencePointerV6Error(
            f"fixture record {index} expected_answer is invalid: {expected_errors}"
        )
    if expected_answer != expected_context["deterministic_answer"]:
        raise EvidencePointerV6Error(
            f"fixture record {index} expected_answer does not match its "
            "expected pointer and evidence"
        )
    if (
        compilation["prompt_sha256"] is not None
        and compilation["prompt_sha256"] != expected_context["prompt_sha256"]
    ):
        raise EvidencePointerV6Error(
            f"fixture record {index} prompt changed between compilation and scoring"
        )
    if (
        compilation["evidence_sha256"] is not None
        and compilation["evidence_sha256"] != expected_context["evidence_sha256"]
    ):
        raise EvidencePointerV6Error(
            f"fixture record {index} evidence changed between compilation and scoring"
        )

    parsed_pointer = compilation["parsed_pointer"]
    pointer_valid = isinstance(parsed_pointer, Mapping)

    def pointer_exact(field: str) -> bool:
        return pointer_valid and parsed_pointer.get(field) == expected_pointer.get(field)

    pointer_metrics = {
        "pointer_parse_valid": pointer_valid,
        "pointer_task_exact": pointer_exact("task"),
        "pointer_decision_exact": pointer_exact("decision"),
        "pointer_span_exact": pointer_exact("span_id"),
        "pointer_strict_exact": parsed_pointer == expected_pointer,
        "compiler_accepted": compilation["status"] == "COMPILED",
    }

    prediction = compilation["compiled_answer"]

    def final_exact(field: str) -> bool:
        return (
            isinstance(prediction, Mapping)
            and prediction.get(field) == expected_answer.get(field)
        )

    schema_errors = (
        validate_student_answer(prediction)
        if isinstance(prediction, Mapping)
        else ["compiled answer is unavailable"]
    )
    final_metrics = {
        "json_valid": isinstance(prediction, Mapping),
        "schema_valid": isinstance(prediction, Mapping) and not schema_errors,
        "schema_exact": final_exact("schema"),
        "citation_exact": final_exact("evidence_ids"),
        "decision_exact": final_exact("decision"),
        "task_exact": final_exact("task"),
        "claim_exact": final_exact("claim"),
        "verdict_exact": final_exact("verdict"),
        "provenance_exact": final_exact("provenance"),
        "strict_exact": prediction == expected_answer,
        "unsupported_wrong_answer": (
            expected_answer["decision"] == "REFUSE"
            and isinstance(prediction, Mapping)
            and prediction.get("decision") == "ANSWER"
        ),
    }
    return {
        "schema": SAMPLE_RESULT_SCHEMA,
        "example_id": checked["example_id"],
        "backend": "fixture_cpu",
        "model_generation_executed": False,
        "gpu_used": False,
        "blind_data_accessed": False,
        "gold_visible_to_compiler": False,
        "compilation": compilation,
        "expected_pointer": expected_pointer,
        "expected_answer": expected_answer,
        "pointer_metrics": pointer_metrics,
        "final_metrics": final_metrics,
        "final_schema_errors": schema_errors,
    }


def _metric(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _aggregate_boolean_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    names: Sequence[str],
) -> dict[str, dict[str, Any]]:
    return {
        name: _metric(
            sum(bool(row[field][name]) for row in rows),
            len(rows),
        )
        for name in names
    }


def _decision_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_field: str,
    strict_name: str,
    predicted_decision,
) -> dict[str, Any]:
    answer_rows = [
        row for row in rows if row["expected_answer"]["decision"] == "ANSWER"
    ]
    refuse_rows = [
        row for row in rows if row["expected_answer"]["decision"] == "REFUSE"
    ]
    answer_accuracy = _metric(
        sum(bool(row[metric_field][strict_name]) for row in answer_rows),
        len(answer_rows),
    )
    tp = sum(
        row["expected_answer"]["decision"] == "REFUSE"
        and predicted_decision(row) == "REFUSE"
        for row in rows
    )
    fp = sum(
        row["expected_answer"]["decision"] != "REFUSE"
        and predicted_decision(row) == "REFUSE"
        for row in rows
    )
    fn = sum(
        row["expected_answer"]["decision"] == "REFUSE"
        and predicted_decision(row) != "REFUSE"
        for row in rows
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "answer_accuracy": answer_accuracy,
        "refuse": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "refuse_examples": len(refuse_rows),
    }


def evaluate_fixture_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compile then score explicit fixtures without model, GPU, or blind access."""

    if not records:
        raise EvidencePointerV6Error("fixture evaluation requires at least one record")
    rows: list[dict[str, Any]] = []
    example_ids: set[str] = set()
    for index, record in enumerate(records, 1):
        row = _score_fixture_record(record, index)
        if row["example_id"] in example_ids:
            raise EvidencePointerV6Error(
                f"duplicate fixture example_id: {row['example_id']}"
            )
        example_ids.add(row["example_id"])
        rows.append(row)

    pointer_names = (
        "pointer_parse_valid",
        "pointer_task_exact",
        "pointer_decision_exact",
        "pointer_span_exact",
        "pointer_strict_exact",
        "compiler_accepted",
    )
    final_names = (
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
    pointer_metrics = _aggregate_boolean_metrics(
        rows,
        field="pointer_metrics",
        names=pointer_names,
    )
    final_metrics = _aggregate_boolean_metrics(
        rows,
        field="final_metrics",
        names=final_names,
    )
    refuse_rows = [
        row for row in rows if row["expected_answer"]["decision"] == "REFUSE"
    ]
    final_metrics["unsupported_wrong_answer_rate"] = _metric(
        sum(bool(row["final_metrics"]["unsupported_wrong_answer"]) for row in refuse_rows),
        len(refuse_rows),
    )
    pointer_decisions = _decision_metrics(
        rows,
        metric_field="pointer_metrics",
        strict_name="pointer_strict_exact",
        predicted_decision=lambda row: (
            row["compilation"]["parsed_pointer"].get("decision")
            if isinstance(row["compilation"]["parsed_pointer"], Mapping)
            else None
        ),
    )
    final_decisions = _decision_metrics(
        rows,
        metric_field="final_metrics",
        strict_name="strict_exact",
        predicted_decision=lambda row: (
            row["compilation"]["compiled_answer"].get("decision")
            if isinstance(row["compilation"]["compiled_answer"], Mapping)
            else None
        ),
    )
    reason_counts = Counter(
        str(row["compilation"]["parse_reason"]["code"]) for row in rows
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "FIXTURE_CPU_EVALUATION_COMPLETE",
        "compiler_version": COMPILER_VERSION,
        "backend": {
            "mode": "fixture_cpu",
            "fixture_only": True,
            "model_generation_executed": False,
            "gpu_used": False,
            "blind_data_accessed": False,
            "model_quality_claim_allowed": False,
        },
        "examples": len(rows),
        "pointer_metrics": pointer_metrics,
        "pointer_answer_accuracy": pointer_decisions["answer_accuracy"],
        "pointer_refuse": pointer_decisions["refuse"],
        "final_metrics": final_metrics,
        "answer_accuracy": final_decisions["answer_accuracy"],
        "refuse": final_decisions["refuse"],
        "compiler": {
            "compiled": sum(row["compilation"]["status"] == "COMPILED" for row in rows),
            "fail_closed": sum(row["compilation"]["fail_closed"] for row in rows),
            "reason_counts": dict(sorted(reason_counts.items())),
        },
        "claim_boundary": (
            "FIXTURE_ONLY_CPU_CONTRACT_TEST_NOT_MODEL_QUALITY_OR_BLIND_EVIDENCE"
        ),
    }
    return rows, summary


def _reject_blind_path(path: Path, *, field: str) -> None:
    if any("blind" in part.casefold() for part in path.parts):
        raise EvidencePointerV6Error(
            f"{field} must not reference a blind-labelled path"
        )


def load_fixture_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load strict fixture JSONL while refusing blind-labelled paths."""

    _reject_blind_path(path, field="fixture path")
    if path.is_symlink():
        raise EvidencePointerV6Error("fixture path must not be a symlink")
    if not path.is_file() or path.suffix.casefold() != ".jsonl":
        raise EvidencePointerV6Error("fixture path must be an existing JSONL file")
    size = path.stat().st_size
    if size <= 0 or size > MAX_FIXTURE_BYTES:
        raise EvidencePointerV6Error(
            f"fixture size must be in 1..{MAX_FIXTURE_BYTES} bytes"
        )
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            raise EvidencePointerV6Error(
                f"fixture JSONL contains blank line {line_number}"
            )
        try:
            record = _parse_single_json_object(line)
        except _CompilerViolation as exc:
            raise EvidencePointerV6Error(
                f"fixture JSONL line {line_number} is invalid: "
                f"{exc.code}: {exc.detail}"
            ) from exc
        records.append(record)
    return records


def run_fixture_evaluation(
    *,
    fixture_path: Path,
    output_dir: Path,
    runner_path: Path | None = None,
) -> dict[str, Any]:
    """Run fixture-only CPU evaluation and atomically publish its evidence."""

    fixture_input = Path(fixture_path)
    records = load_fixture_jsonl(fixture_input)
    fixture_path = fixture_input.resolve()
    output_dir = Path(output_dir).resolve()
    _reject_blind_path(output_dir, field="output directory")
    if output_dir.exists():
        raise EvidencePointerV6Error(f"output directory already exists: {output_dir}")
    rows, summary = evaluate_fixture_records(records)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.tmp-{uuid.uuid4().hex}"
    if staging.exists():
        raise EvidencePointerV6Error(f"staging directory already exists: {staging}")
    staging.mkdir()
    try:
        sample_path = staging / "sample_results.v6.jsonl"
        summary_path = staging / "summary.v6.json"
        receipt_path = staging / "run_receipt.v6.json"
        sample_path.write_bytes(_jsonl_bytes(rows))
        summary_path.write_bytes(_json_bytes(summary))
        runner_sha256 = (
            sha256_file(runner_path.resolve())
            if runner_path is not None and runner_path.is_file()
            else None
        )
        receipt = {
            "schema": RUN_RECEIPT_SCHEMA,
            "status": "FIXTURE_CPU_EVALUATION_COMPLETE",
            "created_at": datetime.now(UTC).isoformat(),
            "compiler_version": COMPILER_VERSION,
            "fixture": {
                "path": str(fixture_path),
                "sha256": sha256_file(fixture_path),
                "records": len(records),
            },
            "execution": {
                "backend": "fixture_cpu",
                "model_generation_executed": False,
                "gpu_used": False,
                "blind_data_accessed": False,
                "gold_visible_to_compiler": False,
            },
            "implementation": {
                "module_path": str(Path(__file__).resolve()),
                "module_sha256": sha256_file(Path(__file__).resolve()),
                "runner_path": str(runner_path.resolve()) if runner_path else None,
                "runner_sha256": runner_sha256,
            },
            "artifacts": {
                "sample_results.v6.jsonl": sha256_file(sample_path),
                "summary.v6.json": sha256_file(summary_path),
            },
            "claim_boundary": (
                "FIXTURE_ONLY_CPU_CONTRACT_TEST_NOT_MODEL_QUALITY_OR_BLIND_EVIDENCE"
            ),
        }
        receipt_path.write_bytes(_json_bytes(receipt))
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "status": "FIXTURE_CPU_EVALUATION_COMPLETE",
        "output_dir": str(output_dir),
        "examples": len(rows),
        "hashes": {
            "sample_results.v6.jsonl": sha256_file(
                output_dir / "sample_results.v6.jsonl"
            ),
            "summary.v6.json": sha256_file(output_dir / "summary.v6.json"),
            "run_receipt.v6.json": sha256_file(
                output_dir / "run_receipt.v6.json"
            ),
        },
        "summary": summary,
    }
