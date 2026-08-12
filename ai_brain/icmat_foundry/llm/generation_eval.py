"""Strict offline generation-level evaluation for self-described ICMat data.

The evaluator accepts integrity-bound JSONL files declared by a dataset
manifest. It evaluates validation or calibration only, never reads final-test
records, never downloads a model, and never contacts an X5 device.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import stat
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from jsonschema import Draft202012Validator

EVALUATOR_VERSION = "icmat-generation-eval-1.3.0"
RESULT_SCHEMA = "icmat_generation_result.v1"
REPORT_SCHEMA = "icmat_generation_report.v1"
RECEIPT_SCHEMA = "icmat_generation_receipt.v1"
AUTHORIZATION_SCHEMA = "icmat_generation_external_authorization.v1"
ALLOWED_SPLITS = ("validation", "calibration")
FORBIDDEN_SPLIT = "test"
DEFAULT_SEED = 20260728
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MAX_AUTHORIZATION_RECEIPT_BYTES = 1024 * 1024
FILE_ATTRIBUTE_REPARSE_POINT = 0x400

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SUBJECT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
SPLIT_TOKEN_PATTERN = re.compile(
    r"(?:^|[._-])(train|validation|calibration|test)(?:[._-]|$)"
)
EVIDENCE_MARKER_PATTERN = re.compile(
    r"(?P<marker>[A-Z][A-Z0-9_]{2,63}_JSON)="
)
V4_PROMPT_FORMAT = "icmat_evidence_response_contract.v4"
V4_ANSWER_SCHEMA_ID = "icmat_teacher_answer.v4"
V4_EVIDENCE_BLOCK_PATTERN = re.compile(
    r"^\[EVIDENCE (?P<header>[^\]\r\n]+)\]\r?\n"
    r"(?P<text>.*?)\r?\n\[/EVIDENCE\]",
    re.MULTILINE | re.DOTALL,
)
V4_RESPONSE_CONTRACT_PATTERN = re.compile(
    r"^\[RESPONSE_CONTRACT\]\r?\n"
    r"(?P<payload>.*?)\r?\n\[/RESPONSE_CONTRACT\]",
    re.MULTILINE | re.DOTALL,
)
V4_IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$"
)
V4_NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?%?"
)
V4_UNIT_AFTER_NUMBER_PATTERN = re.compile(
    r"^\s*(?P<unit>"
    r"(?:[kMGT]?Pa|[munpf]?m|[munpf]?s|[kMGT]?Hz|"
    r"[kM]?eV|[munp]?W|[munp]?A|[munp]?V|K|"
    r"°C|degC|cycles?|samples?|g/cm\^?3|kg/m\^?3)"
    r")\b",
    re.IGNORECASE,
)
V4_SEMANTIC_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,}")
V4_SEMANTIC_STOPWORDS = frozenset(
    {
        "about",
        "also",
        "and",
        "are",
        "because",
        "been",
        "being",
        "but",
        "cited",
        "could",
        "data",
        "evidence",
        "excerpt",
        "excerpts",
        "finding",
        "findings",
        "for",
        "from",
        "has",
        "have",
        "into",
        "its",
        "literature",
        "may",
        "method",
        "methods",
        "one",
        "our",
        "paper",
        "reported",
        "reports",
        "result",
        "results",
        "should",
        "show",
        "shows",
        "study",
        "support",
        "supported",
        "than",
        "that",
        "the",
        "their",
        "these",
        "this",
        "those",
        "using",
        "was",
        "were",
        "which",
        "with",
        "would",
    }
)
V4_LOCAL_SYSTEM_PATTERN = re.compile(
    r"\b(?:"
    r"our\s+(?:rdk(?:\s*x5)?|x5|bpu|board|device|lab(?:oratory)?|fab)|"
    r"local\s+(?:(?:sem|xrd|tem|raman|pl)\s+)?(?:measurement|"
    r"experiment|fabrication|result|execution|deployment|ground[- ]truth|"
    r"lab(?:oratory)?|fab|rdk(?:\s*x5)?|x5|bpu)|"
    r"this\s+(?:board|device|edge|lab(?:oratory)?|fab|production|"
    r"rdk(?:\s*x5)?|x5|bpu)|"
    r"(?:rdk\s*)?x5|rdk|bayes[- ]e|bpu|on[- ]device|edge[- ]device|"
    r"(?:local\s+)?edge\s+accelerator|fab[- ]line|shop[- ]floor|"
    r"production(?:[- ]wafer)?[- ]line"
    r")\b",
    re.IGNORECASE,
)
V4_LOCAL_ACTION_PATTERN = re.compile(
    r"\b(?:"
    r"acquir(?:e|ed|es|ing)|achiev(?:e|ed|es|ing)|"
    r"benchmark(?:ed|s|ing)?|collect(?:ed|s|ing)?|"
    r"complet(?:e|ed|es|ing)|confirm(?:ed|s|ing)?|"
    r"demonstrat(?:e|ed|es|ing)|deploy(?:ed|s|ing)?|"
    r"establish(?:ed|es|ing)?|execut(?:e|ed|es|ing)|"
    r"infer(?:red|s|ring)|measur(?:e|ed|es|ing)|"
    r"occur(?:red|s|ring)?|perform(?:ed|s|ing)?|process(?:ed|es|ing)?|"
    r"produc(?:e|ed|es|ing)|prov(?:e|ed|es|ing|en)|ran|"
    r"report(?:ed|s|ing)?|"
    r"run|running|show(?:ed|n|s|ing)?|test(?:ed|s|ing)?|"
    r"train(?:ed|s|ing)?|us(?:e|ed|es|ing)|"
    r"validat(?:e|ed|es|ing|ion)"
    r")\b",
    re.IGNORECASE,
)
V4_REFUSAL_CUE_PATTERN = re.compile(
    r"\b(?:cannot\s+(?:answer|establish|support)|"
    r"(?:does|do|did)\s+not\s+(?:establish|support)|not\s+(?:established|"
    r"found|present|shown|supported)|refus(?:e|ed|al)|unsupported)\b",
    re.IGNORECASE,
)
V4_AFFIRMATIVE_SUPPORT_PATTERN = re.compile(
    r"\b(?:is|are|was|were|has\s+been|have\s+been)\s+"
    r"(?:confirmed|demonstrated|established|proved|proven|supported)\b",
    re.IGNORECASE,
)
V4_COMPARISON_CUE_PATTERN = re.compile(
    r"\b(?:both|compar(?:e|ed|es|ing|ison)|contrast|differ(?:ent|ence|s)?|"
    r"higher|lower|more|less|relative|similar|superior|than|whereas|while)\b",
    re.IGNORECASE,
)
V4_NEXT_ACTION_CUE_PATTERN = re.compile(
    r"\b(?:follow[- ]up|next|propos(?:e|ed|es|ing|al)|"
    r"recommend(?:ed|s|ing)?|should|could|would|may|might|will|must|"
    r"needs?\s+to)\b",
    re.IGNORECASE,
)
V4_MEASUREMENT_OR_TOOL_PATTERN = re.compile(
    r"\b(?:analysis|assay|characteri[sz](?:ation|e)|diffraction|"
    r"ellipsometr(?:y|ic)|measurement|microscop(?:y|ic)|model|"
    r"photoluminescen(?:ce|t)|raman|sem|simulation|spectroscop(?:y|ic)|"
    r"tem|test|tool|xrd)\b",
    re.IGNORECASE,
)
V4_INFORMATION_VALUE_PATTERN = re.compile(
    r"\b(?:assess|characteri[sz]e|determine|discriminat(?:e|ion)|"
    r"distinguish|identify|quantify|reduce|resolve|uncertain(?:ty)?|"
    r"validat(?:e|ion)|verify)\b",
    re.IGNORECASE,
)
V4_UNRESOLVED_BOUNDARY_PATTERN = re.compile(
    r"\b(?:absent|absence|lack|lacks|missing|no|not|unreported|"
    r"unresolved|without)\b",
    re.IGNORECASE,
)
CITATION_KEYS = ("evidence", "citations", "citation", "provenance")
OUTPUT_SCHEMA_KEYS = (
    "assistant_json_schema",
    "output_json_schema",
    "response_json_schema",
    "generation_json_schema",
)
UNKNOWN_ACTION_KEYS = (
    "value",
    "answer",
    "prediction",
    "result",
    "relation",
    "tool",
    "arguments",
    "action",
    "decision",
)
AUTHORIZATION_CONTAINER_KEYS = (
    "generation_evaluation_authorization",
    "generation_eval_authorization",
    "generation_evaluation",
    "independent_audit",
    "audit",
)
AUTHORIZATION_GO = {"GO", "AUTHORIZED", "APPROVED"}
AUTHORIZATION_HOLD = {
    "HOLD",
    "DENIED",
    "DENY",
    "REJECT",
    "REJECTED",
    "REVOKED",
    "BLOCKED",
    "NOT_AUTHORIZED",
}
BLOCKING_SEVERITIES = {"BLOCKING", "BLOCKER", "CRITICAL"}


class EvaluationContractError(ValueError):
    """Raised when an artifact violates the evaluation contract."""


class FinalTestSplitForbidden(PermissionError):
    """Raised before artifact access when final-test evaluation is requested."""


@dataclass(frozen=True)
class DatasetFile:
    split: str
    relative_path: str
    resolved_path: Path
    sha256: str
    bytes: int
    examples: int


@dataclass(frozen=True)
class EvaluationSample:
    example_id: str
    task: str
    split: str
    record_schema: str
    messages: tuple[dict[str, str], ...]
    prompt_marker: str
    prompt_payload: dict[str, Any]
    expected: dict[str, Any]
    output_schema: dict[str, Any]
    output_schema_source: str
    citation_fields: tuple[str, ...]
    gold_generation: str
    record: dict[str, Any]


@dataclass(frozen=True)
class DatasetSelection:
    manifest_name: str
    manifest_sha256: str
    manifest_schema: str
    dataset_authorized: bool
    authorization: dict[str, Any]
    data_file: DatasetFile
    samples: tuple[EvaluationSample, ...]


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvaluationContractError(f"{label} must be an object")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvaluationContractError(f"{label} must be a non-empty string")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    keys: set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != keys:
        raise EvaluationContractError(
            f"{label} keys mismatch: {sorted(actual ^ keys)}"
        )


def ensure_allowed_split(split: str) -> None:
    if split == FORBIDDEN_SPLIT:
        raise FinalTestSplitForbidden(
            "final-test generation evaluation is forbidden; only validation or "
            "calibration may be evaluated"
        )
    if split not in ALLOWED_SPLITS:
        raise EvaluationContractError(
            f"split must be one of {ALLOWED_SPLITS}; received {split!r}"
        )


def _parse_one_json_object(text: str, label: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise EvaluationContractError(f"{label} must be text")
    stripped = text.lstrip()
    try:
        value, end = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError as exc:
        raise EvaluationContractError(
            f"{label} is not valid JSON: {exc.msg}"
        ) from exc
    if stripped[end:].strip():
        raise EvaluationContractError(
            f"{label} contains trailing non-JSON content"
        )
    if not isinstance(value, dict):
        raise EvaluationContractError(
            f"{label} must contain exactly one JSON object"
        )
    return value


def _v4_citation_enum(
    response_schema: Mapping[str, Any],
) -> tuple[str, ...]:
    try:
        citations = response_schema["properties"]["sentences"]["items"][
            "properties"
        ]["citations"]
        values = citations["items"]["enum"]
    except (KeyError, TypeError) as exc:
        raise EvaluationContractError(
            "V4 response schema must enumerate sentences[*].citations"
        ) from exc
    if (
        not isinstance(values, list)
        or not values
        or any(
            not isinstance(value, str)
            or not V4_IDENTIFIER_PATTERN.fullmatch(value)
            for value in values
        )
        or len(set(values)) != len(values)
    ):
        raise EvaluationContractError(
            "V4 citation enum must contain unique evidence span identifiers"
        )
    return tuple(values)


def _parse_v4_evidence_header(header: str) -> dict[str, str]:
    tokens = header.split()
    if not tokens:
        raise EvaluationContractError("V4 evidence header is empty")
    span_id = tokens[0]
    if not V4_IDENTIFIER_PATTERN.fullmatch(span_id):
        raise EvaluationContractError(
            f"V4 evidence span identifier is invalid: {span_id!r}"
        )
    attributes: dict[str, str] = {"span_id": span_id}
    for token in tokens[1:]:
        key, separator, value = token.partition("=")
        if (
            separator != "="
            or not key
            or not value
            or not re.fullmatch(r"[a-z][a-z0-9_]*", key)
            or key in attributes
        ):
            raise EvaluationContractError(
                f"V4 evidence header token is invalid: {token!r}"
            )
        attributes[key] = value
    return attributes


def _extract_v4_prompt_payload(
    user_text: str,
) -> tuple[str, dict[str, Any]]:
    evidence_matches = list(V4_EVIDENCE_BLOCK_PATTERN.finditer(user_text))
    if (
        not evidence_matches
        or user_text.count("[EVIDENCE ") != len(evidence_matches)
        or user_text.count("[/EVIDENCE]") != len(evidence_matches)
    ):
        raise EvaluationContractError(
            "V4 prompt must contain complete, non-nested evidence blocks"
        )
    contract_matches = list(
        V4_RESPONSE_CONTRACT_PATTERN.finditer(user_text)
    )
    if (
        len(contract_matches) != 1
        or user_text.count("[RESPONSE_CONTRACT]") != 1
        or user_text.count("[/RESPONSE_CONTRACT]") != 1
    ):
        raise EvaluationContractError(
            "V4 prompt must contain exactly one complete response contract"
        )
    contract_match = contract_matches[0]
    if any(
        evidence.start() < contract_match.end()
        and contract_match.start() < evidence.end()
        for evidence in evidence_matches
    ):
        raise EvaluationContractError(
            "V4 response contract must not be nested in evidence"
        )
    if max(match.end() for match in evidence_matches) > contract_match.start():
        raise EvaluationContractError(
            "V4 evidence blocks must precede the response contract"
        )

    spans: list[dict[str, Any]] = []
    seen_span_ids: set[str] = set()
    for match in evidence_matches:
        span = _parse_v4_evidence_header(match.group("header"))
        span_id = span["span_id"]
        if span_id in seen_span_ids:
            raise EvaluationContractError(
                f"V4 prompt repeats evidence span {span_id}"
            )
        seen_span_ids.add(span_id)
        text = match.group("text").strip()
        if not text:
            raise EvaluationContractError(
                f"V4 evidence span {span_id} has empty text"
            )
        span["text"] = text
        spans.append(span)

    contract = _parse_one_json_object(
        contract_match.group("payload"),
        "V4 response contract",
    )
    _require_exact_keys(
        contract,
        {"request_id", "response_schema"},
        "V4 response contract",
    )
    request_id = _require_string(
        contract["request_id"],
        "V4 response contract request_id",
    )
    if not V4_IDENTIFIER_PATTERN.fullmatch(request_id):
        raise EvaluationContractError(
            "V4 response contract request_id is invalid"
        )
    response_schema = _require_mapping(
        contract["response_schema"],
        "V4 response schema",
    )
    try:
        Draft202012Validator.check_schema(response_schema)
    except Exception as exc:
        raise EvaluationContractError(
            f"V4 response schema is invalid: {exc}"
        ) from exc
    properties = response_schema.get("properties")
    if not isinstance(properties, dict):
        raise EvaluationContractError(
            "V4 response schema must define object properties"
        )
    required = response_schema.get("required")
    if (
        response_schema.get("type") != "object"
        or response_schema.get("additionalProperties") is not False
        or not isinstance(required, list)
        or not {
            "schema",
            "request_id",
            "decision",
            "sentences",
        }.issubset(required)
    ):
        raise EvaluationContractError(
            "V4 response schema must strictly require its top-level contract"
        )
    schema_id = properties.get("schema")
    request_contract = properties.get("request_id")
    decision_contract = properties.get("decision")
    if (
        not isinstance(schema_id, dict)
        or schema_id.get("const") != V4_ANSWER_SCHEMA_ID
        or not isinstance(request_contract, dict)
        or request_contract.get("const") != request_id
    ):
        raise EvaluationContractError(
            "V4 response schema must bind schema id and request_id"
        )
    allowed_decisions: set[str] = set()
    if isinstance(decision_contract, dict):
        if isinstance(decision_contract.get("const"), str):
            allowed_decisions.add(decision_contract["const"])
        decision_enum = decision_contract.get("enum")
        if isinstance(decision_enum, list):
            allowed_decisions.update(
                item for item in decision_enum if isinstance(item, str)
            )
    if not allowed_decisions or not allowed_decisions <= {"ANSWER", "REFUSE"}:
        raise EvaluationContractError(
            "V4 response schema decision must allow only ANSWER or REFUSE"
        )
    provenance_contract = properties.get("evidence_provenance")
    if provenance_contract is not None and (
        "evidence_provenance" not in required
        or not isinstance(provenance_contract, dict)
        or "const" not in provenance_contract
    ):
        raise EvaluationContractError(
            "V4 evidence_provenance must be a required fixed contract field"
        )
    allowed_citations = _v4_citation_enum(response_schema)
    if set(allowed_citations) != seen_span_ids:
        raise EvaluationContractError(
            "V4 response schema citation enum must match prompt evidence spans"
        )
    return "[EVIDENCE]/[RESPONSE_CONTRACT]", {
        "__prompt_format__": V4_PROMPT_FORMAT,
        "evidence_spans": spans,
        "response_contract": {
            "request_id": request_id,
            "response_schema": response_schema,
        },
    }


def _extract_prompt_payload(user_text: str) -> tuple[str, dict[str, Any]]:
    has_v4_marker = any(
        marker in user_text
        for marker in (
            "[EVIDENCE ",
            "[/EVIDENCE]",
            "[RESPONSE_CONTRACT]",
            "[/RESPONSE_CONTRACT]",
        )
    )
    if has_v4_marker:
        return _extract_v4_prompt_payload(user_text)
    matches = list(EVIDENCE_MARKER_PATTERN.finditer(user_text))
    if len(matches) != 1:
        raise EvaluationContractError(
            "user prompt must contain exactly one uppercase *_JSON evidence marker"
        )
    match = matches[0]
    marker = match.group("marker") + "="
    payload_text = user_text[match.end() :]
    return marker, _parse_one_json_object(
        payload_text,
        "prompt evidence payload",
    )


def _walk_mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_mappings(child)


def _named_values(value: Any, key: str) -> list[Any]:
    result: list[Any] = []
    for mapping in _walk_mappings(value):
        if key in mapping:
            result.append(mapping[key])
    return result


def _contains_mapping(value: Any, expected: Mapping[str, Any]) -> bool:
    return any(mapping == expected for mapping in _walk_mappings(value))


def _scalar_leaves(value: Any) -> list[tuple[str, Any]]:
    leaves: list[tuple[str, Any]] = []

    def visit(item: Any, key: str) -> None:
        if isinstance(item, dict):
            for child_key, child in item.items():
                visit(child, child_key)
        elif isinstance(item, list):
            for child in item:
                visit(child, key)
        elif item is None or isinstance(item, (str, int, float, bool)):
            leaves.append((key, item))
        else:
            raise EvaluationContractError(
                f"unsupported JSON value type under {key}"
            )

    visit(value, "$")
    return leaves


def _record_binding_spaces(record: Mapping[str, Any]) -> list[Any]:
    spaces: list[Any] = []
    for key in ("source", "host_binding", "provenance", "binding"):
        value = record.get(key)
        if isinstance(value, (dict, list)):
            spaces.append(value)
    return spaces


def _value_is_bound(
    key: str,
    value: Any,
    spaces: Sequence[Any],
) -> bool:
    for space in spaces:
        for candidate in _named_values(space, key):
            if candidate == value:
                return True
    return False


def _citation_fields(target: Mapping[str, Any]) -> tuple[str, ...]:
    fields = tuple(key for key in CITATION_KEYS if key in target)
    if not fields:
        raise EvaluationContractError(
            "assistant JSON must contain evidence/citation/provenance binding"
        )
    for field in fields:
        if not isinstance(target[field], (dict, list)) or not target[field]:
            raise EvaluationContractError(
                f"assistant citation field {field} must be non-empty"
            )
    return fields


def _validate_citations(
    target: Mapping[str, Any],
    citation_fields: Sequence[str],
    prompt_payload: Mapping[str, Any],
    record: Mapping[str, Any],
) -> None:
    spaces = [prompt_payload, *_record_binding_spaces(record)]
    for field in citation_fields:
        citation = target[field]
        if _contains_mapping(prompt_payload, citation):
            continue
        for key, value in _scalar_leaves(citation):
            if value is None:
                continue
            if not _value_is_bound(key, value, spaces):
                raise EvaluationContractError(
                    f"assistant citation leaf {field}.{key} is not prompt-bound"
                )


def _target_requested_field(target: Mapping[str, Any]) -> str | None:
    value = target.get("requested_field")
    if isinstance(value, str) and value:
        return value
    arguments = target.get("arguments")
    if isinstance(arguments, dict):
        field = arguments.get("field")
        if isinstance(field, str) and field:
            return field
    return None


def _prompt_fact_values(
    prompt_payload: Mapping[str, Any],
    field: str,
) -> list[Any]:
    values: list[Any] = []
    record = prompt_payload.get("record")
    if isinstance(record, dict) and field in record:
        values.append(record[field])
    for mapping in _walk_mappings(prompt_payload):
        if mapping.get("field") == field and "value" in mapping:
            values.append(mapping["value"])
        if (
            field == "decision_threshold"
            and mapping.get("kind") == "decision_threshold"
            and "value" in mapping
        ):
            values.append(mapping["value"])
    return values


def _prompt_units(prompt_payload: Mapping[str, Any]) -> list[str]:
    units = [
        value
        for value in _named_values(prompt_payload, "unit")
        if isinstance(value, str) and value
    ]
    units.extend(
        value
        for value in _named_values(prompt_payload, "expected_unit")
        if isinstance(value, str) and value
    )
    return units


def _validate_unknown_target(target: Mapping[str, Any]) -> None:
    if target.get("status") != "UNKNOWN":
        return
    refusal_signal = False
    for key in UNKNOWN_ACTION_KEYS:
        if key not in target:
            continue
        refusal_signal = True
        if target[key] is not None:
            raise EvaluationContractError(
                f"UNKNOWN assistant target must null {key}"
            )
    reason = target.get("reason")
    if reason is not None:
        refusal_signal = True
        if not isinstance(reason, str) or not reason:
            raise EvaluationContractError(
                "UNKNOWN assistant reason must be null or non-empty text"
            )
    if not refusal_signal:
        raise EvaluationContractError(
            "UNKNOWN assistant target has no explicit refusal signal"
        )


def _validate_relation(target: Mapping[str, Any]) -> None:
    relation = target.get("relation")
    value = target.get("value")
    threshold = target.get("threshold")
    if relation is None:
        return
    if not _is_number(value) or not _is_number(threshold):
        raise EvaluationContractError(
            "non-null relation requires finite value and threshold"
        )
    if relation == "ABOVE_OR_EQUAL" and value < threshold:
        raise EvaluationContractError(
            "assistant relation contradicts value/threshold"
        )
    if relation == "BELOW" and value >= threshold:
        raise EvaluationContractError(
            "assistant relation contradicts value/threshold"
        )
    if relation not in {"ABOVE_OR_EQUAL", "BELOW"}:
        raise EvaluationContractError(
            f"unsupported threshold relation: {relation}"
        )


def _validate_supported_values(
    target: Mapping[str, Any],
    prompt_payload: Mapping[str, Any],
) -> None:
    status = target.get("status")
    if status not in {"SUPPORTED", "READY"}:
        return
    requested_field = _target_requested_field(target)
    if requested_field is not None and "value" in target:
        value = target["value"]
        if value is not None and value not in _prompt_fact_values(
            prompt_payload,
            requested_field,
        ):
            raise EvaluationContractError(
                "assistant value is not supplied for requested field"
            )
    if "threshold" in target:
        threshold = target["threshold"]
        threshold_values = _prompt_fact_values(
            prompt_payload,
            "decision_threshold",
        )
        if threshold not in threshold_values:
            raise EvaluationContractError(
                "assistant threshold is not supplied by prompt evidence"
            )

    facts = target.get("facts")
    source_record = prompt_payload.get("record")
    if isinstance(facts, dict) and isinstance(source_record, dict):
        identifiers = {"jid", "record_id", "id"}
        supplied_facts = {
            key: value
            for key, value in source_record.items()
            if key not in identifiers
        }
        if facts != supplied_facts:
            raise EvaluationContractError(
                "assistant facts do not exactly match supplied record facts"
            )

    arguments = target.get("arguments")
    if isinstance(arguments, dict):
        available_fields: list[Any] = []
        for candidate in _named_values(prompt_payload, "available_fields"):
            if isinstance(candidate, list):
                available_fields.extend(candidate)
        for key, value in arguments.items():
            if key == "fields":
                if not isinstance(value, list) or value not in _named_values(
                    prompt_payload,
                    "available_fields",
                ):
                    raise EvaluationContractError(
                        "assistant fields are not copied from available_fields"
                    )
            elif key == "field" and available_fields:
                normalized = {
                    item.get("field") if isinstance(item, dict) else item
                    for item in available_fields
                }
                if value not in normalized:
                    raise EvaluationContractError(
                        "assistant field is not listed as available"
                    )
            elif not _value_is_bound(
                key,
                value,
                [prompt_payload],
            ):
                raise EvaluationContractError(
                    f"assistant argument {key} is not prompt-bound"
                )


def _is_v4_prompt_payload(value: Mapping[str, Any]) -> bool:
    return value.get("__prompt_format__") == V4_PROMPT_FORMAT


def _v4_response_schema(
    prompt_payload: Mapping[str, Any],
) -> dict[str, Any]:
    contract = prompt_payload.get("response_contract")
    if not isinstance(contract, dict):
        raise EvaluationContractError(
            "V4 prompt payload is missing its response contract"
        )
    return _require_mapping(
        contract.get("response_schema"),
        "V4 response schema",
    )


def _v4_uses_deterministic_evidence_operation(
    prompt_payload: Mapping[str, Any],
) -> bool:
    response_schema = _v4_response_schema(prompt_payload)
    properties = response_schema.get("properties")
    if not isinstance(properties, dict):
        return False
    operation_schema = properties.get("evidence_operation_id")
    sentences_schema = properties.get("sentences")
    if not isinstance(operation_schema, dict) or not isinstance(
        operation_schema.get("const"),
        str,
    ):
        return False
    if not isinstance(sentences_schema, dict):
        return False
    sentence_schema = sentences_schema.get("items")
    if not isinstance(sentence_schema, dict):
        return False
    sentence_properties = sentence_schema.get("properties")
    if not isinstance(sentence_properties, dict):
        return False
    text_schema = sentence_properties.get("text")
    citations_schema = sentence_properties.get("citations")
    return (
        isinstance(text_schema, dict)
        and isinstance(text_schema.get("const"), str)
        and isinstance(citations_schema, dict)
        and isinstance(citations_schema.get("const"), list)
    )


def _v4_request_id(prompt_payload: Mapping[str, Any]) -> str:
    contract = prompt_payload.get("response_contract")
    if not isinstance(contract, dict):
        raise EvaluationContractError(
            "V4 prompt payload is missing its response contract"
        )
    return _require_string(
        contract.get("request_id"),
        "V4 response contract request_id",
    )


def _v4_span_by_id(
    prompt_payload: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    spans = prompt_payload.get("evidence_spans")
    if not isinstance(spans, list) or not spans:
        raise EvaluationContractError(
            "V4 prompt payload has no evidence spans"
        )
    result: dict[str, Mapping[str, Any]] = {}
    for span in spans:
        if not isinstance(span, dict):
            raise EvaluationContractError(
                "V4 prompt evidence span must be an object"
            )
        span_id = _require_string(span.get("span_id"), "V4 span_id")
        _require_string(span.get("text"), "V4 evidence text")
        if span_id in result:
            raise EvaluationContractError(
                f"V4 prompt repeats evidence span {span_id}"
            )
        result[span_id] = span
    return result


def _v4_sentences(
    response: Mapping[str, Any],
) -> list[Mapping[str, Any]] | None:
    sentences = response.get("sentences")
    if (
        not isinstance(sentences, list)
        or not sentences
        or any(not isinstance(sentence, dict) for sentence in sentences)
    ):
        return None
    return sentences


def _v4_citations_valid(
    response: Mapping[str, Any],
    prompt_payload: Mapping[str, Any],
) -> bool:
    sentences = _v4_sentences(response)
    if sentences is None:
        return False
    span_by_id = _v4_span_by_id(prompt_payload)
    expected_sentence_ids = [
        f"s{index}" for index in range(1, len(sentences) + 1)
    ]
    if [
        sentence.get("sentence_id")
        for sentence in sentences
    ] != expected_sentence_ids:
        return False
    for sentence in sentences:
        citations = sentence.get("citations")
        if (
            not isinstance(citations, list)
            or not citations
            or any(not isinstance(citation, str) for citation in citations)
            or len(set(citations)) != len(citations)
            or any(citation not in span_by_id for citation in citations)
        ):
            return False
    return True


def _v4_number_signature(token: str) -> tuple[Decimal, bool] | None:
    percent = token.endswith("%")
    normalized = token[:-1] if percent else token
    normalized = normalized.replace(",", "")
    try:
        value = Decimal(normalized)
    except InvalidOperation:
        return None
    return value.normalize(), percent


def _v4_numeric_claims(
    text: str,
) -> list[tuple[tuple[Decimal, bool], str | None]]:
    claims: list[tuple[tuple[Decimal, bool], str | None]] = []
    for match in V4_NUMBER_PATTERN.finditer(text):
        signature = _v4_number_signature(match.group(0))
        if signature is None:
            continue
        unit_match = V4_UNIT_AFTER_NUMBER_PATTERN.match(text[match.end() :])
        unit = (
            unit_match.group("unit").casefold().replace("^", "")
            if unit_match is not None
            else None
        )
        claims.append((signature, unit))
    return claims


def _v4_numeric_grounding_valid(
    response: Mapping[str, Any],
    prompt_payload: Mapping[str, Any],
) -> bool:
    if not _v4_citations_valid(response, prompt_payload):
        return False
    sentences = _v4_sentences(response)
    assert sentences is not None
    span_by_id = _v4_span_by_id(prompt_payload)
    for sentence in sentences:
        cited_text = " ".join(
            str(span_by_id[citation]["text"])
            for citation in sentence["citations"]
        )
        available = _v4_numeric_claims(cited_text)
        available_signatures = {signature for signature, _ in available}
        for signature, unit in _v4_numeric_claims(
            str(sentence.get("text", ""))
        ):
            if signature not in available_signatures:
                return False
            if unit is not None and (signature, unit) not in available:
                return False
    return True


def _v4_semantic_token(token: str) -> str:
    normalized = token.casefold().strip("-")
    for suffix in ("ing", "ied", "ed", "es", "s"):
        if normalized.endswith(suffix) and len(normalized) - len(suffix) >= 4:
            stem = normalized[: -len(suffix)]
            return stem + "y" if suffix == "ied" else stem
    return normalized


def _v4_semantic_tokens(text: str) -> set[str]:
    return {
        normalized
        for token in V4_SEMANTIC_TOKEN_PATTERN.findall(text)
        if (normalized := _v4_semantic_token(token))
        not in V4_SEMANTIC_STOPWORDS
        and len(normalized) >= 3
    }


def _v4_semantic_overlap(text: str, evidence_text: str) -> int:
    return len(_v4_semantic_tokens(text) & _v4_semantic_tokens(evidence_text))


def _v4_semantic_support(
    text: str,
    evidence_text: str,
    *,
    minimum_overlap: int,
    minimum_coverage: float,
) -> bool:
    text_tokens = _v4_semantic_tokens(text)
    if not text_tokens:
        return False
    overlap = len(text_tokens & _v4_semantic_tokens(evidence_text))
    return (
        overlap >= minimum_overlap
        and overlap / len(text_tokens) >= minimum_coverage
    )


def _v4_action_is_negated(clause: str, action: re.Match[str]) -> bool:
    prefix = clause[max(0, action.start() - 90) : action.start()]
    return bool(
        re.search(
            r"(?:\b(?:did|does|do|was|were|is|are|has|have|had|"
            r"can|could|will|would|should|must)\s+not|"
            r"\b(?:never|no|without))"
            r"(?:\s+[A-Za-z0-9_-]+){0,2}\s*$",
            prefix,
            re.IGNORECASE,
        )
    )


def _v4_action_is_proposal(clause: str, action: re.Match[str]) -> bool:
    if V4_NEXT_ACTION_CUE_PATTERN.search(clause) is None:
        return False
    prefix = clause[max(0, action.start() - 120) : action.start()]
    if re.search(
        r"\b(?:should|could|would|may|might|will|must|needs?\s+to)"
        r"(?:\s+be)?\s*$",
        prefix,
        re.IGNORECASE,
    ):
        return True
    return bool(
        re.search(r"\bpropos(?:e|ed|es|ing)\b", prefix, re.IGNORECASE)
        and re.search(r"\b(?:be|to\s+be)\s*$", prefix, re.IGNORECASE)
    )


def _v4_local_term_is_disclaimed(
    clause: str,
    term: re.Match[str],
) -> bool:
    prefix = clause[max(0, term.start() - 100) : term.start()]
    suffix = clause[term.end() : term.end() + 120]
    if re.search(
        r"(?:\b(?:no|without)|"
        r"\b(?:did|does|do|was|were|is|are|has|have|had|can|could)"
        r"\s+not(?:\s+[A-Za-z0-9_-]+){0,2})\s*$",
        prefix,
        re.IGNORECASE,
    ):
        return True
    return bool(
        re.match(
            r"\s+(?:did|does|do|was|were|is|are|has|have|had|can|could)"
            r"\s+not(?:\s+[A-Za-z0-9_-]+){0,2}\s+"
            + V4_LOCAL_ACTION_PATTERN.pattern,
            suffix,
            re.IGNORECASE,
        )
    )


def _v4_local_execution_promotion(
    text: str,
    *,
    allow_explicit_proposal: bool,
) -> str | None:
    clauses = re.split(
        r"(?<=[.!?;])\s+|\n+|"
        r"\s+\b(?:but|however|whereas|while)\b\s+",
        text,
        flags=re.IGNORECASE,
    )
    for clause in clauses:
        local_terms = tuple(V4_LOCAL_SYSTEM_PATTERN.finditer(clause))
        if not local_terms:
            continue
        if all(
            _v4_local_term_is_disclaimed(clause, term)
            for term in local_terms
        ):
            continue
        for action in V4_LOCAL_ACTION_PATTERN.finditer(clause):
            if _v4_action_is_negated(clause, action):
                continue
            if (
                allow_explicit_proposal
                and _v4_action_is_proposal(clause, action)
            ):
                continue
            return f"{local_terms[0].group(0)}:{action.group(0)}"
    return None


def _v4_local_boundary_valid(
    response: Mapping[str, Any],
    *,
    task: str,
) -> bool:
    sentences = _v4_sentences(response)
    if sentences is None:
        return False
    return all(
        _v4_local_execution_promotion(
            str(sentence.get("text", "")),
            allow_explicit_proposal=(task == "next_measurement_or_tool"),
        )
        is None
        for sentence in sentences
    )


def _v4_fixed_provenance_valid(
    response: Mapping[str, Any],
    prompt_payload: Mapping[str, Any],
) -> bool:
    properties = _v4_response_schema(prompt_payload).get("properties")
    if not isinstance(properties, dict):
        return False
    provenance_schema = properties.get("evidence_provenance")
    if provenance_schema is None:
        return "evidence_provenance" not in response
    return (
        isinstance(provenance_schema, dict)
        and "const" in provenance_schema
        and response.get("evidence_provenance")
        == provenance_schema["const"]
    )


def _v4_semantics_valid(
    response: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    prompt_payload: Mapping[str, Any],
    task: str,
) -> bool:
    sentences = _v4_sentences(response)
    if sentences is None or not _v4_citations_valid(
        response,
        prompt_payload,
    ):
        return False
    decision = response.get("decision")
    if decision == "REFUSE":
        first_text = str(sentences[0].get("text", ""))
        return (
            expected.get("decision") == "REFUSE"
            and V4_REFUSAL_CUE_PATTERN.search(first_text) is not None
            and V4_AFFIRMATIVE_SUPPORT_PATTERN.search(first_text) is None
        )
    if decision != "ANSWER":
        return False

    span_by_id = _v4_span_by_id(prompt_payload)
    if task == "computed_experimental_boundary":
        expected_provenance = response.get("evidence_provenance")
        text = str(sentences[0].get("text", ""))
        if (
            not isinstance(expected_provenance, str)
            or expected_provenance.casefold() not in text.casefold()
        ):
            return False
        if (
            expected_provenance == "unresolved"
            and V4_UNRESOLVED_BOUNDARY_PATTERN.search(text) is None
        ):
            return False
        return set(sentences[0]["citations"]) == set(span_by_id)

    if task == "evidence_bounded_comparison":
        first = sentences[0]
        first_text = str(first.get("text", ""))
        if (
            V4_COMPARISON_CUE_PATTERN.search(first_text) is None
            or set(first["citations"]) != set(span_by_id)
        ):
            return False
        if any(
            _v4_semantic_overlap(first_text, str(span["text"])) < 1
            for span in span_by_id.values()
        ):
            return False
        cited_text = " ".join(
            str(span_by_id[citation]["text"])
            for citation in first["citations"]
        )
        if not _v4_semantic_support(
            first_text,
            cited_text,
            minimum_overlap=2,
            minimum_coverage=0.4,
        ):
            return False

    if task == "next_measurement_or_tool":
        first_text = str(sentences[0].get("text", ""))
        if (
            V4_NEXT_ACTION_CUE_PATTERN.search(first_text) is None
            or V4_MEASUREMENT_OR_TOOL_PATTERN.search(first_text) is None
            or V4_INFORMATION_VALUE_PATTERN.search(first_text) is None
        ):
            return False

    for index, sentence in enumerate(sentences):
        text = str(sentence.get("text", ""))
        cited_text = " ".join(
            str(span_by_id[citation]["text"])
            for citation in sentence["citations"]
        )
        minimum_overlap = (
            1
            if task == "next_measurement_or_tool" or index > 0
            else 2
        )
        minimum_coverage = (
            0.2
            if task == "next_measurement_or_tool"
            else (
                0.3
                if index > 0
                else (
                    0.4
                    if task == "evidence_bounded_comparison"
                    else 0.5
                )
            )
        )
        if _v4_semantic_support(
            text,
            cited_text,
            minimum_overlap=minimum_overlap,
            minimum_coverage=minimum_coverage,
        ):
            continue
        if index > 0 and V4_REFUSAL_CUE_PATTERN.search(text):
            continue
        return False
    return True


def _validate_v4_target_grounding(
    target: Mapping[str, Any],
    prompt_payload: Mapping[str, Any],
    record: Mapping[str, Any],
) -> tuple[str, ...]:
    if target.get("schema") != V4_ANSWER_SCHEMA_ID:
        raise EvaluationContractError(
            "V4 assistant target has an unexpected schema id"
        )
    response_schema = _v4_response_schema(prompt_payload)
    schema_errors = _schema_errors(target, response_schema)
    if schema_errors:
        raise EvaluationContractError(
            "V4 assistant target violates its response contract: "
            + "; ".join(schema_errors)
        )
    task = _require_string(record.get("task"), "record.task")
    deterministic_operation = _v4_uses_deterministic_evidence_operation(
        prompt_payload
    )
    checks = {
        "request_id": target.get("request_id")
        == _v4_request_id(prompt_payload),
        "decision": target.get("decision") in {"ANSWER", "REFUSE"},
        "evidence_provenance": _v4_fixed_provenance_valid(
            target,
            prompt_payload,
        ),
        "citations": _v4_citations_valid(target, prompt_payload),
        "numeric_grounding": _v4_numeric_grounding_valid(
            target,
            prompt_payload,
        ),
        "local_execution_boundary": _v4_local_boundary_valid(
            target,
            task=task,
        ),
        "task_semantics": (
            True
            if deterministic_operation
            else _v4_semantics_valid(
                target,
                expected=target,
                prompt_payload=prompt_payload,
                task=task,
            )
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise EvaluationContractError(
            "V4 assistant target grounding failed: " + ", ".join(failed)
        )
    return ("sentences[*].citations",)


def _validate_target_grounding(
    target: Mapping[str, Any],
    prompt_payload: Mapping[str, Any],
    user_text: str,
    record: Mapping[str, Any],
) -> tuple[str, ...]:
    if _is_v4_prompt_payload(prompt_payload):
        return _validate_v4_target_grounding(
            target,
            prompt_payload,
            record,
        )
    if target.get("schema") == V4_ANSWER_SCHEMA_ID:
        raise EvaluationContractError(
            "V4 assistant target requires V4 evidence and response-contract blocks"
        )
    _require_string(target.get("schema"), "assistant JSON schema id")
    status = _require_string(target.get("status"), "assistant JSON status")
    if status not in {"SUPPORTED", "READY", "UNKNOWN"}:
        raise EvaluationContractError(
            f"assistant JSON has unsupported status {status!r}"
        )
    citation_fields = _citation_fields(target)
    _validate_citations(target, citation_fields, prompt_payload, record)

    requested_field = _target_requested_field(target)
    if requested_field is not None:
        prompt_requested = [
            value
            for value in _named_values(prompt_payload, "requested_field")
            if isinstance(value, str)
        ]
        if prompt_requested:
            if any(value != requested_field for value in prompt_requested):
                raise EvaluationContractError(
                    "assistant requested_field contradicts prompt evidence"
                )
        elif not re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(requested_field)}(?![A-Za-z0-9_])",
            user_text,
        ):
            raise EvaluationContractError(
                "assistant requested_field is not bound in the user prompt"
            )

    target_units = [
        value
        for key, value in _scalar_leaves(target)
        if "unit" in key.lower() and isinstance(value, str)
    ]
    supplied_units = _prompt_units(prompt_payload)
    if target_units and any(unit not in supplied_units for unit in target_units):
        raise EvaluationContractError(
            "assistant unit is not bound by prompt evidence"
        )

    _validate_unknown_target(target)
    _validate_relation(target)
    _validate_supported_values(target, prompt_payload)
    return citation_fields


def _shape_schema(value: Any, *, key: str = "$") -> dict[str, Any]:
    if isinstance(value, dict):
        return {
            "type": "object",
            "additionalProperties": False,
            "required": list(value),
            "properties": {
                child_key: _shape_schema(child, key=child_key)
                for child_key, child in value.items()
            },
        }
    if isinstance(value, list):
        return {
            "type": "array",
            "minItems": len(value),
            "maxItems": len(value),
            "prefixItems": [
                _shape_schema(child, key=key)
                for child in value
            ],
            "items": False,
        }
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if _is_number(value):
        return {"type": "number"}
    if isinstance(value, str):
        if key == "schema":
            return {"const": value}
        return {"type": "string"}
    raise EvaluationContractError(
        f"cannot derive JSON schema for {key}: {type(value).__name__}"
    )


def _record_output_schema(
    record: Mapping[str, Any],
    target: Mapping[str, Any],
    prompt_payload: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    candidates: list[tuple[str, dict[str, Any]]] = []
    if _is_v4_prompt_payload(prompt_payload):
        candidates.append(
            (
                "prompt.response_contract.response_schema",
                _v4_response_schema(prompt_payload),
            )
        )
        record_schema = record.get("response_schema")
        if isinstance(record_schema, dict):
            candidates.append(("response_schema", record_schema))
    for key in OUTPUT_SCHEMA_KEYS:
        value = record.get(key)
        if isinstance(value, dict):
            candidates.append((key, value))
    contracts = record.get("contracts")
    if isinstance(contracts, dict):
        for key in OUTPUT_SCHEMA_KEYS:
            value = contracts.get(key)
            if isinstance(value, dict):
                candidates.append((f"contracts.{key}", value))
    if len(candidates) > 1:
        canonical = {canonical_json(value) for _, value in candidates}
        if len(canonical) != 1:
            raise EvaluationContractError(
                "record contains conflicting assistant JSON schemas"
            )
    if candidates:
        source, schema = candidates[0]
    else:
        source = "derived_from_assistant_json"
        schema = _shape_schema(target)
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise EvaluationContractError(
            f"assistant JSON schema is invalid: {exc}"
        ) from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(target),
        key=lambda item: list(item.path),
    )
    if errors:
        raise EvaluationContractError(
            "assistant target violates its own JSON schema: "
            + "; ".join(error.message for error in errors)
        )
    return dict(schema), source


def _schema_errors(
    output: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> list[str]:
    errors = sorted(
        Draft202012Validator(schema).iter_errors(output),
        key=lambda item: list(item.path),
    )
    return [
        f"{'/'.join(str(part) for part in error.path) or '$'}: {error.message}"
        for error in errors
    ]


def _validate_record(
    record: Any,
    split: str,
    line_number: int,
) -> EvaluationSample:
    item = _require_mapping(record, f"record line {line_number}")
    for key in ("schema", "example_id", "task", "split", "messages"):
        if key not in item:
            raise EvaluationContractError(
                f"record line {line_number} missing {key}"
            )
    record_schema = _require_string(item["schema"], "record.schema")
    example_id = _require_string(item["example_id"], "record.example_id")
    task = _require_string(item["task"], "record.task")
    if item["split"] != split:
        raise EvaluationContractError(
            f"record {example_id} split {item['split']!r} does not match {split!r}"
        )
    if str(item["split"]).lower() == FORBIDDEN_SPLIT:
        raise FinalTestSplitForbidden(
            f"record {example_id} attempts to materialize final-test semantics"
        )

    messages = item["messages"]
    if (
        not isinstance(messages, list)
        or len(messages) != 3
        or [
            message.get("role")
            for message in messages
            if isinstance(message, dict)
        ]
        != ["system", "user", "assistant"]
    ):
        raise EvaluationContractError(
            f"record {example_id} messages must be system/user/assistant"
        )
    normalized_messages: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        mapping = _require_mapping(
            message,
            f"record {example_id} message {index}",
        )
        _require_exact_keys(mapping, {"role", "content"}, "message")
        normalized_messages.append(
            {
                "role": _require_string(mapping["role"], "message.role"),
                "content": _require_string(mapping["content"], "message.content"),
            }
        )

    prompt_marker, payload = _extract_prompt_payload(
        normalized_messages[1]["content"]
    )
    gold_text = normalized_messages[2]["content"]
    target = _parse_one_json_object(
        gold_text,
        f"record {example_id} assistant target",
    )
    citation_fields = _validate_target_grounding(
        target,
        payload,
        normalized_messages[1]["content"],
        item,
    )
    output_schema, output_schema_source = _record_output_schema(
        item,
        target,
        payload,
    )
    return EvaluationSample(
        example_id=example_id,
        task=task,
        split=split,
        record_schema=record_schema,
        messages=tuple(normalized_messages),
        prompt_marker=prompt_marker,
        prompt_payload=payload,
        expected=target,
        output_schema=output_schema,
        output_schema_source=output_schema_source,
        citation_fields=citation_fields,
        gold_generation=gold_text,
        record=dict(item),
    )


def _path_split_hint(path: str) -> str | None:
    normalized = path.replace("\\", "/").lower()
    matches = SPLIT_TOKEN_PATTERN.findall(PurePosixPath(normalized).name)
    unique = set(matches)
    if len(unique) > 1:
        raise EvaluationContractError(
            f"ambiguous split tokens in path: {path}"
        )
    return next(iter(unique), None)


def _safe_relative_jsonl(path: Any) -> str:
    value = _require_string(path, "manifest data path").replace("\\", "/")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or ":" in pure.parts[0]:
        raise EvaluationContractError(
            f"manifest data path is unsafe: {value}"
        )
    if pure.suffix.lower() != ".jsonl":
        raise EvaluationContractError(
            f"manifest data path is not JSONL: {value}"
        )
    return pure.as_posix()


def _parse_descriptor(
    raw: Any,
    key_hint: str | None,
) -> dict[str, Any]:
    descriptor = _require_mapping(raw, "manifest file descriptor")
    required = {"path", "sha256", "bytes", "examples"}
    missing = sorted(required - set(descriptor))
    if missing:
        raise EvaluationContractError(
            f"manifest file descriptor missing integrity fields: {missing}"
        )
    path = _safe_relative_jsonl(descriptor["path"])
    digest = _require_string(
        descriptor["sha256"],
        "manifest file sha256",
    )
    if not SHA256_PATTERN.fullmatch(digest):
        raise EvaluationContractError(
            "manifest file sha256 is invalid"
        )
    byte_count = descriptor["bytes"]
    example_count = descriptor["examples"]
    if (
        not isinstance(byte_count, int)
        or isinstance(byte_count, bool)
        or byte_count < 0
    ):
        raise EvaluationContractError(
            "manifest file bytes must be a non-negative integer"
        )
    if (
        not isinstance(example_count, int)
        or isinstance(example_count, bool)
        or example_count < 0
    ):
        raise EvaluationContractError(
            "manifest file examples must be a non-negative integer"
        )
    explicit = descriptor.get("split")
    known_splits = {"train", *ALLOWED_SPLITS, "test"}
    if explicit is not None and explicit not in known_splits:
        raise EvaluationContractError(
            f"unsupported descriptor split: {explicit}"
        )
    hinted = key_hint if key_hint in known_splits else None
    inferred = _path_split_hint(path)
    candidates = {
        value
        for value in (explicit, hinted, inferred)
        if value is not None
    }
    if len(candidates) != 1:
        raise EvaluationContractError(
            "manifest descriptor has missing or conflicting split binding: "
            f"{path}"
        )
    return {
        "split": next(iter(candidates)),
        "path": path,
        "sha256": digest,
        "bytes": byte_count,
        "examples": example_count,
    }


def _collect_descriptors(
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw_descriptors: list[tuple[Any, str | None]] = []
    files = manifest.get("files")
    if isinstance(files, list):
        raw_descriptors.extend((item, None) for item in files)
    elif isinstance(files, dict):
        training = files.get("training")
        if isinstance(training, list):
            raw_descriptors.extend((item, None) for item in training)
        for split in ("train", *ALLOWED_SPLITS, "test"):
            value = files.get(split)
            if isinstance(value, dict):
                raw_descriptors.append((value, split))
            elif isinstance(value, list):
                raw_descriptors.extend((item, split) for item in value)

    for container_name in ("splits", "data_files"):
        container = manifest.get(container_name)
        if not isinstance(container, dict):
            continue
        for split in ("train", *ALLOWED_SPLITS, "test"):
            value = container.get(split)
            if isinstance(value, dict):
                raw_descriptors.append((value, split))
            elif isinstance(value, list):
                raw_descriptors.extend((item, split) for item in value)

    if not raw_descriptors:
        raise EvaluationContractError(
            "manifest has no supported integrity-bound JSONL file layout"
        )
    parsed = [
        _parse_descriptor(raw, hint)
        for raw, hint in raw_descriptors
    ]
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for descriptor in parsed:
        key = (str(descriptor["split"]), str(descriptor["path"]))
        previous = unique.get(key)
        if previous is not None and previous != descriptor:
            raise EvaluationContractError(
                "conflicting duplicate manifest descriptor: "
                f"{descriptor['path']}"
            )
        unique[key] = descriptor
    return list(unique.values())


def _manifest_authorization(
    manifest: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    containers = set(AUTHORIZATION_CONTAINER_KEYS)
    pending: list[tuple[str, Any]] = [("$", manifest)]
    while pending:
        location, current = pending.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                child_location = f"{location}.{key}"
                if key in containers and isinstance(value, dict):
                    status_value = value.get("status", value.get("decision"))
                    status = (
                        status_value.upper()
                        if isinstance(status_value, str)
                        else None
                    )
                    scope_raw = value.get("scope", value.get("scopes"))
                    if isinstance(scope_raw, str):
                        scopes = [scope_raw]
                    elif isinstance(scope_raw, list):
                        scopes = [
                            item
                            for item in scope_raw
                            if isinstance(item, str)
                        ]
                    else:
                        scopes = []
                    observations.append(
                        {
                            "container": key,
                            "location": child_location,
                            "status": status,
                            "scopes": scopes,
                            "revoked": value.get("revoked") is True,
                        }
                    )
                pending.append((child_location, value))
        elif isinstance(current, list):
            pending.extend(
                (f"{location}[{index}]", value)
                for index, value in enumerate(current)
            )

    observations.sort(
        key=lambda item: (
            str(item["location"]),
            str(item.get("status")),
        )
    )
    blocking = [
        observation
        for observation in observations
        if observation.get("status") in AUTHORIZATION_HOLD
        or observation.get("revoked") is True
    ]
    self_go = [
        observation
        for observation in observations
        if observation.get("status") in AUTHORIZATION_GO
    ]
    if blocking:
        reason = "MANIFEST_AUTHORIZATION_BLOCKED"
    elif self_go:
        reason = "MANIFEST_SELF_GO_IGNORED_EXTERNAL_RECEIPT_REQUIRED"
    else:
        reason = "EXTERNAL_AUTHORIZATION_RECEIPT_REQUIRED"
    return False, {
        "authorized": False,
        "reason": reason,
        "blocking": bool(blocking),
        "blocking_observations": blocking,
        "self_go_ignored": bool(self_go),
        "observations": observations,
    }


def _has_reparse_attribute(result: os.stat_result) -> bool:
    attributes = int(getattr(result, "st_file_attributes", 0))
    return bool(attributes & FILE_ATTRIBUTE_REPARSE_POINT)


def _stat_identity(
    result: os.stat_result,
    *,
    include_content_metadata: bool,
) -> dict[str, int]:
    identity = {
        "mode_type": stat.S_IFMT(result.st_mode),
        "device": int(result.st_dev),
        "inode": int(result.st_ino),
        "attributes": int(getattr(result, "st_file_attributes", 0)),
    }
    if include_content_metadata:
        identity.update(
            {
                "bytes": int(result.st_size),
                "mtime_ns": int(result.st_mtime_ns),
                "ctime_ns": int(result.st_ctime_ns),
            }
        )
    return identity


def _is_unc_path(path: Path) -> bool:
    raw = os.fspath(path)
    normalized = raw.replace("/", "\\")
    if normalized.startswith("\\\\"):
        return True
    return PureWindowsPath(raw).drive.startswith("\\\\")


def _safe_regular_file_snapshot(
    path: Path,
    *,
    workspace_root: Path,
) -> tuple[Path, str, dict[str, Any]]:
    if _is_unc_path(path):
        raise EvaluationContractError(
            "authorization receipt path must not be UNC"
        )
    if ".." in path.parts:
        raise EvaluationContractError(
            "authorization receipt path traversal is unsafe"
        )

    root = Path(os.path.abspath(workspace_root))
    candidate = path if path.is_absolute() else root / path
    candidate = Path(os.path.abspath(candidate))
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise EvaluationContractError(
            "authorization receipt must stay inside the workspace"
        ) from exc
    if relative == Path("."):
        raise EvaluationContractError(
            "authorization receipt must be a regular file"
        )
    if any(":" in part for part in relative.parts):
        raise EvaluationContractError(
            "authorization receipt path contains an unsafe stream separator"
        )

    components = [root]
    current = root
    for part in relative.parts:
        current = current / part
        components.append(current)

    component_identities: list[dict[str, Any]] = []
    for index, component in enumerate(components):
        try:
            result = os.lstat(component)
        except OSError as exc:
            raise EvaluationContractError(
                "authorization receipt path is not an accessible regular file"
            ) from exc
        if stat.S_ISLNK(result.st_mode) or _has_reparse_attribute(result):
            raise EvaluationContractError(
                "authorization receipt path must not contain symlink/reparse "
                "components"
            )
        is_final = index == len(components) - 1
        if is_final:
            if not stat.S_ISREG(result.st_mode):
                raise EvaluationContractError(
                    "authorization receipt must be a regular file"
                )
        elif not stat.S_ISDIR(result.st_mode):
            raise EvaluationContractError(
                "authorization receipt parent must be a directory"
            )
        component_identities.append(
            {
                "relative": (
                    "."
                    if component == root
                    else component.relative_to(root).as_posix()
                ),
                "identity": _stat_identity(
                    result,
                    include_content_metadata=is_final,
                ),
            }
        )
    final_identity = component_identities[-1]["identity"]
    if final_identity["bytes"] > MAX_AUTHORIZATION_RECEIPT_BYTES:
        raise EvaluationContractError(
            "authorization receipt exceeds the 1 MiB safety limit"
        )
    return candidate, relative.as_posix(), {
        "components": component_identities,
        "file": final_identity,
    }


def _read_regular_file_once(
    path: Path,
    expected_identity: Mapping[str, int],
) -> tuple[bytes, dict[str, int]]:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EvaluationContractError(
            "authorization receipt could not be opened safely"
        ) from exc
    try:
        before = os.fstat(descriptor)
        before_identity = _stat_identity(
            before,
            include_content_metadata=True,
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or _has_reparse_attribute(before)
            or before_identity != dict(expected_identity)
        ):
            raise EvaluationContractError(
                "authorization receipt metadata changed before read"
            )
        chunks: list[bytes] = []
        remaining = MAX_AUTHORIZATION_RECEIPT_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > MAX_AUTHORIZATION_RECEIPT_BYTES:
            raise EvaluationContractError(
                "authorization receipt exceeds the 1 MiB safety limit"
            )
        after_identity = _stat_identity(
            os.fstat(descriptor),
            include_content_metadata=True,
        )
        if after_identity != before_identity:
            raise EvaluationContractError(
                "authorization receipt metadata changed during read"
            )
        return payload, after_identity
    finally:
        os.close(descriptor)


def _read_stable_workspace_receipt(
    path: Path,
    *,
    workspace_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved, relative, before = _safe_regular_file_snapshot(
        path,
        workspace_root=workspace_root,
    )
    first_payload, first_identity = _read_regular_file_once(
        resolved,
        before["file"],
    )
    _, second_relative, middle = _safe_regular_file_snapshot(
        resolved,
        workspace_root=workspace_root,
    )
    if second_relative != relative or middle != before:
        raise EvaluationContractError(
            "authorization receipt path metadata changed after first read"
        )
    second_payload, second_identity = _read_regular_file_once(
        resolved,
        middle["file"],
    )
    _, third_relative, after = _safe_regular_file_snapshot(
        resolved,
        workspace_root=workspace_root,
    )
    if (
        third_relative != relative
        or after != before
        or second_identity != first_identity
        or second_payload != first_payload
    ):
        raise EvaluationContractError(
            "authorization receipt hash/metadata was not stable"
        )
    try:
        receipt = json.loads(first_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationContractError(
            "authorization receipt must be valid UTF-8 JSON"
        ) from exc
    receipt = _require_mapping(receipt, "authorization receipt")
    return receipt, {
        "path": relative,
        "bytes": len(first_payload),
        "sha256": sha256_bytes(first_payload),
        "metadata": first_identity,
        "stable_read_passes": 2,
    }


def _scope_tokens(value: Any) -> set[str]:
    tokens: set[str] = set()
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_")
        if normalized:
            tokens.add(normalized)
    elif isinstance(value, list):
        for item in value:
            tokens.update(_scope_tokens(item))
    elif isinstance(value, dict):
        for key, item in value.items():
            if item is True:
                tokens.update(_scope_tokens(key))
            else:
                tokens.update(_scope_tokens(item))
    return tokens


def _scope_allows_generation_split(
    receipt: Mapping[str, Any],
    split: str,
) -> tuple[bool, list[str]]:
    raw_scope = receipt.get("scope", receipt.get("scopes"))
    tokens = _scope_tokens(raw_scope)
    split_allowed = split in tokens or any(
        split in token and "generation" in token
        for token in tokens
    )
    return split_allowed, sorted(tokens)


def _blocking_receipt_findings(
    receipt: Mapping[str, Any],
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    blocking_findings = receipt.get("blocking_findings")
    if isinstance(blocking_findings, list) and blocking_findings:
        blockers.extend(
            {
                "location": f"$.blocking_findings[{index}]",
                "reason": "EXPLICIT_BLOCKING_FINDING",
            }
            for index, _ in enumerate(blocking_findings)
        )
    elif blocking_findings not in (None, []):
        blockers.append(
            {
                "location": "$.blocking_findings",
                "reason": "MALFORMED_BLOCKING_FINDINGS",
            }
        )

    pending: list[tuple[str, Any]] = [("$", receipt)]
    while pending:
        location, current = pending.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                child_location = f"{location}.{key}"
                key_lower = key.lower()
                if key_lower in {"status", "decision"} and isinstance(
                    value,
                    str,
                ):
                    if value.upper() in AUTHORIZATION_HOLD:
                        blockers.append(
                            {
                                "location": child_location,
                                "reason": value.upper(),
                            }
                        )
                if key_lower in {
                    "blocking",
                    "blocks_authorization",
                    "revoked",
                } and value is True:
                    blockers.append(
                        {
                            "location": child_location,
                            "reason": key_lower.upper(),
                        }
                    )
                if (
                    key_lower == "severity"
                    and isinstance(value, str)
                    and value.upper() in BLOCKING_SEVERITIES
                ):
                    blockers.append(
                        {
                            "location": child_location,
                            "reason": value.upper(),
                        }
                    )
                pending.append((child_location, value))
        elif isinstance(current, list):
            pending.extend(
                (f"{location}[{index}]", value)
                for index, value in enumerate(current)
            )
    unique = {
        (item["location"], item["reason"]): item
        for item in blockers
    }
    return [
        unique[key]
        for key in sorted(unique)
    ]


def verify_generation_authorization_receipt(
    *,
    selection: DatasetSelection,
    receipt_path: Path,
    expected_sha256: str,
    workspace_root: Path = WORKSPACE_ROOT,
) -> dict[str, Any]:
    if not isinstance(expected_sha256, str) or not SHA256_PATTERN.fullmatch(
        expected_sha256
    ):
        raise EvaluationContractError(
            "authorization SHA-256 must be 64 lowercase hex characters"
        )
    if selection.authorization.get("blocking") is True:
        raise PermissionError(
            "manifest contains a blocking/revoked authorization observation"
        )
    receipt, file_receipt = _read_stable_workspace_receipt(
        receipt_path,
        workspace_root=workspace_root,
    )
    if file_receipt["sha256"] != expected_sha256:
        raise PermissionError(
            "authorization receipt SHA-256 does not match caller expectation"
        )
    if receipt.get("decision") != "GO":
        raise PermissionError(
            "authorization receipt decision must be exactly GO"
        )
    if receipt.get("generation_evaluation_authorized") is not True:
        raise PermissionError(
            "authorization receipt does not authorize generation evaluation"
        )
    subject = receipt.get("subject")
    if not isinstance(subject, dict):
        raise EvaluationContractError(
            "authorization receipt.subject must be an object"
        )
    if subject.get("manifest_sha256") != selection.manifest_sha256:
        raise PermissionError(
            "authorization receipt subject does not bind the current manifest"
        )
    scope_allowed, scope_tokens = _scope_allows_generation_split(
        receipt,
        selection.data_file.split,
    )
    if not scope_allowed:
        raise PermissionError(
            "authorization receipt scope does not allow this generation split"
        )
    blockers = _blocking_receipt_findings(receipt)
    if blockers:
        raise PermissionError(
            "authorization receipt contains blocking/revoked findings"
        )
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "verified": True,
        "authority": "external_independent_audit_receipt",
        "decision": "GO",
        "generation_evaluation_authorized": True,
        "manifest_sha256": selection.manifest_sha256,
        "split": selection.data_file.split,
        "scope_tokens": scope_tokens,
        "receipt": file_receipt,
        "blocking_findings": [],
    }


def _require_verified_generation_authorization(
    selection: DatasetSelection,
    authorization: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(authorization, Mapping):
        raise PermissionError(
            "local-model generation requires a verified external audit receipt"
        )
    required = {
        "schema": AUTHORIZATION_SCHEMA,
        "verified": True,
        "authority": "external_independent_audit_receipt",
        "decision": "GO",
        "generation_evaluation_authorized": True,
        "manifest_sha256": selection.manifest_sha256,
        "split": selection.data_file.split,
        "blocking_findings": [],
    }
    for key, expected in required.items():
        if authorization.get(key) != expected:
            raise PermissionError(
                "local-model generation authorization is invalid or stale"
            )
    if selection.authorization.get("blocking") is True:
        raise PermissionError(
            "manifest contains a blocking/revoked authorization observation"
        )
    return dict(authorization)


def _resolve_dataset_file(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    split: str,
) -> DatasetFile:
    descriptors = [
        descriptor
        for descriptor in _collect_descriptors(manifest)
        if descriptor["split"] == split
    ]
    if len(descriptors) != 1:
        raise EvaluationContractError(
            f"manifest must bind exactly one {split} JSONL file; "
            f"found {len(descriptors)}"
        )
    descriptor = descriptors[0]
    relative_path = str(descriptor["path"])
    for part in PurePosixPath(relative_path).parts:
        lowered = part.lower()
        if "test" in lowered and SPLIT_TOKEN_PATTERN.search(lowered):
            raise EvaluationContractError(
                "allowed split descriptor points at a test artifact"
            )
    root = manifest_path.resolve().parent
    resolved = (
        root / Path(*PurePosixPath(relative_path).parts)
    ).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise EvaluationContractError(
            "dataset file escapes manifest directory"
        ) from exc
    if not resolved.is_file():
        raise EvaluationContractError(
            f"dataset file does not exist: {relative_path}"
        )
    actual_bytes = resolved.stat().st_size
    actual_sha256 = sha256_file(resolved)
    if (
        actual_bytes != descriptor["bytes"]
        or actual_sha256 != descriptor["sha256"]
    ):
        raise EvaluationContractError(
            f"dataset file integrity mismatch: {relative_path}"
        )
    return DatasetFile(
        split=split,
        relative_path=relative_path,
        resolved_path=resolved,
        sha256=actual_sha256,
        bytes=actual_bytes,
        examples=int(descriptor["examples"]),
    )


def _read_jsonl_samples(
    data_file: DatasetFile,
) -> list[EvaluationSample]:
    samples: list[EvaluationSample] = []
    identifiers: set[str] = set()
    with data_file.resolved_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise EvaluationContractError(
                    f"{data_file.relative_path}:{line_number} is blank"
                )
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationContractError(
                    f"{data_file.relative_path}:{line_number} invalid JSON"
                ) from exc
            sample = _validate_record(
                value,
                data_file.split,
                line_number,
            )
            if sample.example_id in identifiers:
                raise EvaluationContractError(
                    f"duplicate example_id: {sample.example_id}"
                )
            identifiers.add(sample.example_id)
            samples.append(sample)
    if len(samples) != data_file.examples:
        raise EvaluationContractError(
            f"manifest examples={data_file.examples}, "
            f"JSONL rows={len(samples)}"
        )
    if not samples:
        raise EvaluationContractError(
            "evaluation split is empty"
        )
    return samples


def _sample_rank(seed: int, example_id: str) -> str:
    return sha256_bytes(f"{seed}|{example_id}".encode())


def select_samples(
    samples: Sequence[EvaluationSample],
    *,
    max_samples: int,
    seed: int,
) -> tuple[EvaluationSample, ...]:
    if max_samples <= 0:
        raise EvaluationContractError(
            "max_samples must be positive"
        )
    groups: dict[tuple[str, str], list[EvaluationSample]] = defaultdict(list)
    for sample in samples:
        groups[
            (sample.task, str(sample.expected.get("status", "MISSING")))
        ].append(sample)
    queues: dict[tuple[str, str], deque[EvaluationSample]] = {}
    for key, group in groups.items():
        queues[key] = deque(
            sorted(
                group,
                key=lambda item: _sample_rank(seed, item.example_id),
            )
        )

    selected: list[EvaluationSample] = []
    keys = sorted(queues)
    target = min(max_samples, len(samples))
    while len(selected) < target:
        progressed = False
        for key in keys:
            if queues[key] and len(selected) < target:
                selected.append(queues[key].popleft())
                progressed = True
        if not progressed:
            break
    if len(selected) != target:
        raise EvaluationContractError(
            "deterministic sample selection was incomplete"
        )
    return tuple(selected)


def load_dataset_selection(
    manifest_path: Path,
    *,
    split: str = "validation",
    max_samples: int = 24,
    seed: int = DEFAULT_SEED,
) -> DatasetSelection:
    ensure_allowed_split(split)
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise EvaluationContractError(
            f"dataset manifest does not exist: {manifest_path}"
        )
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise EvaluationContractError(
            "dataset manifest is invalid JSON"
        ) from exc
    manifest = _require_mapping(manifest, "dataset manifest")
    manifest_schema = _require_string(
        manifest.get("schema"),
        "manifest.schema",
    )
    authorized, authorization = _manifest_authorization(manifest)
    data_file = _resolve_dataset_file(
        manifest_path,
        manifest,
        split,
    )
    samples = _read_jsonl_samples(data_file)
    return DatasetSelection(
        manifest_name=manifest_path.name,
        manifest_sha256=sha256_file(manifest_path),
        manifest_schema=manifest_schema,
        dataset_authorized=authorized,
        authorization=authorization,
        data_file=data_file,
        samples=select_samples(
            samples,
            max_samples=max_samples,
            seed=seed,
        ),
    )


def _without_citations(
    value: Mapping[str, Any],
    citation_fields: Sequence[str],
) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in citation_fields
    }


def _unknown_refusal_valid(
    generated: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    if generated.get("status") != expected.get("status"):
        return False
    if expected.get("status") != "UNKNOWN":
        return True
    for key in UNKNOWN_ACTION_KEYS:
        if key in expected and generated.get(key, object()) is not None:
            return False
    return generated.get("reason") == expected.get("reason")


def _unit_values(value: Mapping[str, Any]) -> list[Any]:
    return [
        item
        for key, item in _scalar_leaves(value)
        if "unit" in key.lower()
    ]


def evaluate_generation(
    sample: EvaluationSample,
    generation: str,
    *,
    subject: str,
    sample_index: int,
) -> dict[str, Any]:
    if not SUBJECT_PATTERN.fullmatch(subject):
        raise EvaluationContractError(
            f"invalid subject label: {subject!r}"
        )
    is_v4 = _is_v4_prompt_payload(sample.prompt_payload)
    deterministic_v4 = (
        is_v4
        and _v4_uses_deterministic_evidence_operation(
            sample.prompt_payload
        )
    )
    checks = {
        "json_object": False,
        "schema_id_bound": False,
        "schema_valid": False,
        "field_semantics": False,
        "citation_binding": False,
        "unknown_refusal": False,
        "unit_validity": False,
        "support_evidence": False,
        "request_id_bound": not is_v4,
        "decision_valid": not is_v4,
        "evidence_provenance_valid": not is_v4,
        "numeric_grounding": not is_v4,
        "local_execution_boundary": not is_v4,
    }
    errors: list[dict[str, Any]] = []
    generated: dict[str, Any] | None = None
    try:
        generated = _parse_one_json_object(
            generation,
            "model generation",
        )
        checks["json_object"] = True
    except EvaluationContractError as exc:
        errors.append(
            {"code": "JSON_OBJECT_INVALID", "detail": str(exc)}
        )

    expected_schema_id = str(sample.expected["schema"])
    if generated is not None:
        checks["schema_id_bound"] = (
            generated.get("schema") == expected_schema_id
        )
        if not checks["schema_id_bound"]:
            errors.append(
                {
                    "code": "SCHEMA_ID_MISMATCH",
                    "detail": (
                        f"expected {expected_schema_id}, got "
                        f"{generated.get('schema')!r}"
                    ),
                }
            )
        schema_messages = _schema_errors(
            generated,
            sample.output_schema,
        )
        checks["schema_valid"] = (
            checks["schema_id_bound"]
            and not schema_messages
        )
        if schema_messages:
            errors.append(
                {
                    "code": "JSON_SCHEMA_INVALID",
                    "detail": schema_messages,
                }
            )
        if is_v4:
            checks["request_id_bound"] = (
                generated.get("request_id")
                == _v4_request_id(sample.prompt_payload)
            )
            checks["decision_valid"] = (
                generated.get("decision")
                == sample.expected.get("decision")
            )
            checks["evidence_provenance_valid"] = (
                _v4_fixed_provenance_valid(
                    generated,
                    sample.prompt_payload,
                )
            )
            checks["citation_binding"] = _v4_citations_valid(
                generated,
                sample.prompt_payload,
            )
            checks["numeric_grounding"] = _v4_numeric_grounding_valid(
                generated,
                sample.prompt_payload,
            )
            checks["local_execution_boundary"] = _v4_local_boundary_valid(
                generated,
                task=sample.task,
            )
            checks["field_semantics"] = (
                generated == sample.expected
                if deterministic_v4
                else _v4_semantics_valid(
                    generated,
                    expected=sample.expected,
                    prompt_payload=sample.prompt_payload,
                    task=sample.task,
                )
            )
            checks["unknown_refusal"] = checks["decision_valid"]
            checks["unit_validity"] = checks["numeric_grounding"]
            checks["support_evidence"] = all(
                checks[name]
                for name in (
                    "field_semantics",
                    "citation_binding",
                    "request_id_bound",
                    "decision_valid",
                    "evidence_provenance_valid",
                    "numeric_grounding",
                    "local_execution_boundary",
                )
            )
        else:
            checks["field_semantics"] = (
                _without_citations(
                    generated,
                    sample.citation_fields,
                )
                == _without_citations(
                    sample.expected,
                    sample.citation_fields,
                )
            )
            checks["citation_binding"] = all(
                generated.get(field) == sample.expected.get(field)
                for field in sample.citation_fields
            )
            checks["unknown_refusal"] = _unknown_refusal_valid(
                generated,
                sample.expected,
            )
            checks["unit_validity"] = (
                _unit_values(generated)
                == _unit_values(sample.expected)
            )
            checks["support_evidence"] = (
                checks["field_semantics"]
                and checks["citation_binding"]
            )
        code_by_check = {
            "field_semantics": "FIELD_SEMANTICS_MISMATCH",
            "citation_binding": "CITATION_BINDING_MISMATCH",
            "unknown_refusal": "UNKNOWN_REFUSAL_INVALID",
            "unit_validity": "UNIT_INVALID",
            "support_evidence": "SUPPORT_EVIDENCE_INVALID",
            "request_id_bound": "REQUEST_ID_MISMATCH",
            "decision_valid": "ANSWER_REFUSE_DECISION_INVALID",
            "evidence_provenance_valid": "EVIDENCE_PROVENANCE_INVALID",
            "numeric_grounding": "UNCITED_NUMERIC_VALUE",
            "local_execution_boundary": "LOCAL_EXECUTION_PROMOTION",
        }
        existing_codes = {
            str(error["code"])
            for error in errors
        }
        for check_name, code in code_by_check.items():
            if (
                not checks[check_name]
                and code not in existing_codes
            ):
                errors.append(
                    {
                        "code": code,
                        "detail": "strict check failed",
                    }
                )

    passed = all(checks.values())
    prompt_messages = list(sample.messages[:-1])
    return {
        "schema": RESULT_SCHEMA,
        "subject": subject,
        "sample_index": sample_index,
        "example_id": sample.example_id,
        "task": sample.task,
        "split": sample.split,
        "record_schema": sample.record_schema,
        "expected_schema": expected_schema_id,
        "expected_status": sample.expected.get(
            "status",
            sample.expected.get("decision"),
        ),
        "generated_schema": (
            generated.get("schema")
            if generated
            else None
        ),
        "generated_status": (
            generated.get("status", generated.get("decision"))
            if generated
            else None
        ),
        "output_schema_source": sample.output_schema_source,
        "prompt_sha256": sha256_bytes(
            canonical_json(prompt_messages).encode("utf-8")
        ),
        "target_sha256": sha256_bytes(
            canonical_json(sample.expected).encode("utf-8")
        ),
        "generation_sha256": sha256_bytes(
            generation.encode("utf-8")
        ),
        "generation": generation,
        "checks": checks,
        "exact_target_match": generated == sample.expected,
        "exact_target_match_role": (
            "required_deterministic_evidence_operation"
            if deterministic_v4
            else (
                "reproducibility_diagnostic_only"
                if is_v4
                else "legacy_strict_semantic_component"
            )
        ),
        "evaluation_semantics": (
            "v4_deterministic_evidence_operation"
            if deterministic_v4
            else (
                "v4_evidence_bounded_open_text"
                if is_v4
                else "legacy_v1_v3_exact_fields"
            )
        ),
        "passed": passed,
        "errors": errors,
    }


def gold_fixture_generations(
    samples: Sequence[EvaluationSample],
    *,
    subject: str = "fixture_gold_echo",
) -> dict[str, dict[str, str]]:
    if not SUBJECT_PATTERN.fullmatch(subject):
        raise EvaluationContractError(
            f"invalid subject label: {subject!r}"
        )
    return {
        subject: {
            sample.example_id: sample.gold_generation
            for sample in samples
        }
    }


def load_fixture_generations(
    path: Path,
    samples: Sequence[EvaluationSample],
) -> dict[str, dict[str, str]]:
    path = path.resolve()
    if not path.is_file():
        raise EvaluationContractError(
            f"fixture generation file does not exist: {path}"
        )
    selected_ids = {
        sample.example_id
        for sample in samples
    }
    result: dict[str, dict[str, str]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise EvaluationContractError(
                    f"fixture generations line {line_number} is blank"
                )
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationContractError(
                    f"fixture generations line {line_number} is invalid JSON"
                ) from exc
            item = _require_mapping(item, "fixture generation")
            _require_exact_keys(
                item,
                {"subject", "example_id", "generation"},
                "fixture generation",
            )
            subject = _require_string(
                item["subject"],
                "fixture subject",
            )
            example_id = _require_string(
                item["example_id"],
                "fixture example_id",
            )
            generation = _require_string(
                item["generation"],
                "fixture generation text",
            )
            if not SUBJECT_PATTERN.fullmatch(subject):
                raise EvaluationContractError(
                    f"invalid fixture subject: {subject!r}"
                )
            if example_id not in selected_ids:
                raise EvaluationContractError(
                    "fixture contains unselected example_id: "
                    f"{example_id}"
                )
            if example_id in result[subject]:
                raise EvaluationContractError(
                    "duplicate fixture generation: "
                    f"{subject}/{example_id}"
                )
            result[subject][example_id] = generation
    if not result:
        raise EvaluationContractError(
            "fixture generation file is empty"
        )
    for subject, generations in result.items():
        missing = sorted(selected_ids - set(generations))
        if missing:
            raise EvaluationContractError(
                f"fixture subject {subject} is missing "
                f"{len(missing)} selected samples"
            )
    return dict(result)


def _artifact_fingerprint(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_dir():
        raise EvaluationContractError(
            f"model artifact is not a local directory: {path}"
        )
    accepted_names = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "adapter_config.json",
    }
    accepted_suffixes = {".safetensors", ".bin"}
    files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and (
            candidate.name in accepted_names
            or candidate.suffix.lower() in accepted_suffixes
        )
    )
    if not files:
        raise EvaluationContractError(
            f"model artifact has no identity files: {path}"
        )
    entries = [
        {
            "path": candidate.relative_to(path).as_posix(),
            "bytes": candidate.stat().st_size,
            "sha256": sha256_file(candidate),
        }
        for candidate in files
    ]
    return {
        "artifact_name": path.name,
        "files": entries,
        "fingerprint_sha256": sha256_bytes(
            canonical_json(entries).encode("utf-8")
        ),
    }


def generate_local_model(
    samples: Sequence[EvaluationSample],
    *,
    base_model_dir: Path,
    subject: str,
    seed: int,
    adapter_dir: Path | None = None,
    device: str = "auto",
    max_input_tokens: int = 1536,
    max_new_tokens: int = 384,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Generate deterministic outputs from local artifacts only."""
    if not SUBJECT_PATTERN.fullmatch(subject):
        raise EvaluationContractError(
            f"invalid subject label: {subject!r}"
        )
    if max_input_tokens <= 0 or max_new_tokens <= 0:
        raise EvaluationContractError(
            "token limits must be positive"
        )
    if device not in {"auto", "cpu", "cuda"}:
        raise EvaluationContractError(
            "device must be auto, cpu, or cuda"
        )

    base_model_dir = base_model_dir.resolve()
    base_fingerprint = _artifact_fingerprint(base_model_dir)
    adapter_fingerprint = (
        _artifact_fingerprint(adapter_dir.resolve())
        if adapter_dir is not None
        else None
    )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise EvaluationContractError(
            "local generation requires installed torch and transformers"
        ) from exc

    selected_device = (
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else device
    )
    if selected_device == "auto":
        selected_device = "cpu"
    if (
        selected_device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise EvaluationContractError(
            "CUDA was requested but is unavailable"
        )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )

    dtype = (
        torch.float16
        if selected_device == "cuda"
        else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_dir),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=dtype,
    )
    if adapter_dir is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise EvaluationContractError(
                "adapter evaluation requires installed peft"
            ) from exc
        model = PeftModel.from_pretrained(
            model,
            str(adapter_dir.resolve()),
            is_trainable=False,
        )
    model.to(selected_device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    generations: dict[str, str] = {}
    try:
        for index, sample in enumerate(samples):
            torch.manual_seed(seed + index)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + index)
            prompt = tokenizer.apply_chat_template(
                list(sample.messages[:-1]),
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_tokens = int(
                encoded["input_ids"].shape[-1]
            )
            if input_tokens > max_input_tokens:
                raise EvaluationContractError(
                    f"prompt {sample.example_id} has {input_tokens} "
                    f"tokens, exceeding limit {max_input_tokens}"
                )
            encoded = {
                key: value.to(selected_device)
                for key, value in encoded.items()
            }
            with torch.inference_mode():
                output = model.generate(
                    **encoded,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            new_tokens = output[0, input_tokens:]
            generations[sample.example_id] = tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metadata: dict[str, Any] = {
        "backend": "local_transformers_free_generation",
        "free_generation_executed": True,
        "network_allowed": False,
        "x5_access_allowed": False,
        "device": selected_device,
        "seed": seed,
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": max_new_tokens,
        "base_model": base_fingerprint,
        "adapter": adapter_fingerprint,
    }
    return generations, metadata


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        text,
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _summarize_subject(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not results:
        raise EvaluationContractError(
            "subject has no evaluation results"
        )
    check_names = list(results[0]["checks"])
    count = len(results)
    passed = sum(
        bool(result["passed"])
        for result in results
    )
    return {
        "samples": count,
        "passed": passed,
        "failed": count - passed,
        "strict_accepted": passed == count,
        "pass_rate": passed / count,
        "check_rates": {
            check: sum(
                bool(result["checks"][check])
                for result in results
            )
            / count
            for check in check_names
        },
        "task_counts": {
            task: sum(
                result["task"] == task
                for result in results
            )
            for task in sorted(
                {str(result["task"]) for result in results}
            )
        },
        "expected_status_counts": {
            status: sum(
                result["expected_status"] == status
                for result in results
            )
            for status in sorted(
                {
                    str(result["expected_status"])
                    for result in results
                }
            )
        },
    }


def write_evaluation(
    *,
    selection: DatasetSelection,
    generations: Mapping[str, Mapping[str, str]],
    subject_metadata: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    mode: str,
    seed: int,
    requested_max_samples: int,
    generation_authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if mode not in {
        "dry_run_gold_fixture",
        "fixture_file",
        "local_model",
    }:
        raise EvaluationContractError(
            f"unsupported evaluation mode: {mode}"
        )
    if not generations:
        raise EvaluationContractError(
            "no generation subjects were supplied"
        )
    verified_authorization: dict[str, Any] | None = None
    if mode == "local_model":
        verified_authorization = _require_verified_generation_authorization(
            selection,
            generation_authorization,
        )
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            "evaluation output is immutable; choose a new directory: "
            f"{output_dir}"
        )

    selected_ids = {
        sample.example_id
        for sample in selection.samples
    }
    results: list[dict[str, Any]] = []
    for subject in sorted(generations):
        if subject not in subject_metadata:
            raise EvaluationContractError(
                f"subject metadata missing for {subject}"
            )
        subject_generations = generations[subject]
        if set(subject_generations) != selected_ids:
            missing = sorted(
                selected_ids - set(subject_generations)
            )
            extra = sorted(
                set(subject_generations) - selected_ids
            )
            raise EvaluationContractError(
                f"subject {subject} generation coverage mismatch: "
                f"missing={missing}, extra={extra}"
            )
        for sample_index, sample in enumerate(selection.samples):
            generation = subject_generations[sample.example_id]
            if not isinstance(generation, str):
                raise EvaluationContractError(
                    f"subject {subject}/{sample.example_id} "
                    "generation is not text"
                )
            results.append(
                evaluate_generation(
                    sample,
                    generation,
                    subject=subject,
                    sample_index=sample_index,
                )
            )

    output_dir.mkdir(parents=True, exist_ok=False)
    results_text = "".join(
        canonical_json(result) + "\n"
        for result in results
    )
    results_path = output_dir / "results.v1.jsonl"
    _write_text_atomic(results_path, results_text)
    results_sha256 = sha256_file(results_path)

    summaries = {
        subject: _summarize_subject(
            [
                result
                for result in results
                if result["subject"] == subject
            ]
        )
        for subject in sorted(generations)
    }
    comparison: dict[str, Any] | None = None
    if "base" in summaries and "adapter" in summaries:
        comparison = {
            "adapter_minus_base_pass_rate": (
                summaries["adapter"]["pass_rate"]
                - summaries["base"]["pass_rate"]
            ),
            "adapter_strictly_better": (
                summaries["adapter"]["pass_rate"]
                > summaries["base"]["pass_rate"]
            ),
        }

    all_strict = all(
        summary["strict_accepted"]
        for summary in summaries.values()
    )
    report_body: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "evaluator_version": EVALUATOR_VERSION,
        "evaluator_implementation": {
            "path": "icmat_foundry/llm/generation_eval.py",
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "mode": mode,
        "determinism": {
            "seed": seed,
            "sampling": (
                "stratified_round_robin_sha256(seed|example_id)"
            ),
            "generation_decoding": (
                "fixture"
                if mode != "local_model"
                else "do_sample=false,num_beams=1"
            ),
        },
        "safety_boundary": {
            "allowed_splits": list(ALLOWED_SPLITS),
            "evaluated_split": selection.data_file.split,
            "test_requested": False,
            "sealed_test_read": False,
            "network_used": False,
            "x5_accessed": False,
            "teacher_forced_loss_used": False,
            "free_generation_executed": mode == "local_model",
        },
        "dataset": {
            "manifest_name": selection.manifest_name,
            "manifest_sha256": selection.manifest_sha256,
            "manifest_schema": selection.manifest_schema,
            "authorized_for_generation_evaluation": (
                verified_authorization is not None
            ),
            "manifest_authorization": selection.authorization,
            "external_authorization": verified_authorization,
            "data_path": selection.data_file.relative_path,
            "data_sha256": selection.data_file.sha256,
            "data_bytes": selection.data_file.bytes,
            "manifest_examples": selection.data_file.examples,
        },
        "selection": {
            "requested_max_samples": requested_max_samples,
            "selected_samples": len(selection.samples),
            "example_ids": [
                sample.example_id
                for sample in selection.samples
            ],
            "record_schemas": sorted(
                {
                    sample.record_schema
                    for sample in selection.samples
                }
            ),
            "assistant_schemas": sorted(
                {
                    str(sample.expected["schema"])
                    for sample in selection.samples
                }
            ),
        },
        "subjects": {
            subject: dict(subject_metadata[subject])
            for subject in sorted(subject_metadata)
            if subject in generations
        },
        "summaries": summaries,
        "base_adapter_comparison": comparison,
        "overall_strict_accepted": all_strict,
        "model_quality_claim_allowed": (
            mode == "local_model"
            and verified_authorization is not None
            and all_strict
        ),
        "results": {
            "path": results_path.name,
            "rows": len(results),
            "bytes": results_path.stat().st_size,
            "sha256": results_sha256,
        },
        "claim_boundary": (
            "Dry-run and fixture modes validate only the evaluator and never "
            "authorize a dataset or model. Local-model mode is permitted only "
            "after a stable, workspace-confined external independent audit "
            "receipt binds the exact manifest and split; manifest self-GO can "
            "never authorize execution, while any manifest HOLD/revocation "
            "still blocks it. "
            "No result is final-test, BPU, X5, or production evidence."
        ),
    }
    report_body["evaluation_payload_sha256"] = sha256_bytes(
        canonical_json(report_body).encode("utf-8")
    )
    report_path = output_dir / "report.v1.json"
    _write_text_atomic(
        report_path,
        json.dumps(
            report_body,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    report_sha256 = sha256_file(report_path)
    receipt_body = {
        "schema": RECEIPT_SCHEMA,
        "evaluator_version": EVALUATOR_VERSION,
        "report": {
            "path": report_path.name,
            "bytes": report_path.stat().st_size,
            "sha256": report_sha256,
        },
        "results": {
            "path": results_path.name,
            "bytes": results_path.stat().st_size,
            "sha256": results_sha256,
        },
    }
    receipt_body["bundle_sha256"] = sha256_bytes(
        canonical_json(receipt_body).encode("utf-8")
    )
    receipt_path = output_dir / "receipt.v1.json"
    _write_text_atomic(
        receipt_path,
        json.dumps(
            receipt_body,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return {
        "report": report_body,
        "receipt": receipt_body,
        "output_dir": str(output_dir),
    }


def build_subject_metadata(
    generations: Mapping[str, Mapping[str, str]],
    *,
    backend: str,
    free_generation_executed: bool,
) -> dict[str, dict[str, Any]]:
    return {
        subject: {
            "backend": backend,
            "free_generation_executed": free_generation_executed,
            "network_allowed": False,
            "x5_access_allowed": False,
        }
        for subject in generations
    }


def map_generation_callable(
    samples: Iterable[EvaluationSample],
    generator: Callable[[EvaluationSample], str],
) -> dict[str, str]:
    """Map a deterministic fixture backend over selected samples."""
    return {
        sample.example_id: generator(sample)
        for sample in samples
    }
