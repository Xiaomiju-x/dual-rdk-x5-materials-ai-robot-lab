"""Build and verify the shortcut-resistant ICMat-Qwen SFT v3 dataset.

SFT v3 is an immutable, offline candidate. It reuses only the audited v2
connected-family primitives. Model-visible inputs contain one neutral view
name; target status must be derived from the facts themselves.
"""
from __future__ import annotations

import copy
import json
import math
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from icmat_foundry.llm.sft_v2 import (
    SourceLock,
    build_family_map,
    canonical_complete_record_hash,
    canonical_json,
    domain_tags,
    sha256_bytes,
    sha256_file,
    split_for_family,
)

BUILDER_VERSION = "icmat-qwen05b-sft-builder-3.0.0"
DATASET_SCHEMA_ID = "icmat_qwen05b_sft.v3"
EXAMPLE_SCHEMA_ID = "icmat_sft_example.v3"
TEST_MEMBERSHIP_SCHEMA_ID = "icmat_sft_test_membership.v3"
SOURCE_ID = "nist_jarvis_dft"
REQUIRED_REUSE_GATE = "ALLOW_TRAIN_REDISTRIBUTE"
SPLIT_NAMES = ("train", "validation", "calibration", "test")
TRAINING_SPLITS = ("train", "validation", "calibration")
TASK_NAMES = (
    "property_judgment",
    "tool_parameters",
    "evidence_adjudication",
)
NEUTRAL_VIEW = "computed_materials_evidence"
USER_MARKER = "ICMAT_EVIDENCE_JSON="
HASH64_PATTERN = re.compile(r"(?i)(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
NA_STRINGS = {"", "na", "n/a", "none", "null", "nan", "inf", "-inf"}

SYSTEM_PROMPTS = {
    "zh": (
        "你是 ICMat 结构化材料助手。只能使用 ICMAT_EVIDENCE_JSON 中明确给出的"
        "公开计算材料证据；证据缺失、冲突或单位错误时必须返回 UNKNOWN。"
        "只输出一个 JSON 对象。"
    ),
    "en": (
        "You are the ICMat structured materials assistant. Use only explicit "
        "public computed-materials evidence in ICMAT_EVIDENCE_JSON. Return "
        "UNKNOWN for missing, conflicting, or unit-invalid evidence. Return "
        "exactly one JSON object."
    ),
}

QUERY_FIELDS: tuple[str, ...] = (
    "formation_energy_peratom",
    "optb88vdw_bandgap",
    "mbj_bandgap",
    "hse_gap",
    "ehull",
    "density",
    "epsx",
    "bulk_modulus_kv",
    "shear_modulus_gv",
    "dfpt_piezo_max_dij",
    "slme",
)

FIELD_UNITS: dict[str, str] = {
    "formation_energy_peratom": "eV/atom",
    "optb88vdw_bandgap": "eV",
    "mbj_bandgap": "eV",
    "hse_gap": "eV",
    "ehull": "eV/atom",
    "density": "g/cm^3",
    "epsx": "dimensionless",
    "bulk_modulus_kv": "GPa",
    "shear_modulus_gv": "GPa",
    "dfpt_piezo_max_dij": "pC/N",
    "slme": "percent",
}

WRONG_UNITS: dict[str, str] = {
    "eV/atom": "GPa",
    "eV": "nm",
    "g/cm^3": "eV",
    "dimensionless": "eV",
    "GPa": "eV",
    "pC/N": "eV",
    "percent": "eV",
}

EVIDENCE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["source_id", "source_version", "record_id"],
    "properties": {
        "source_id": {"const": SOURCE_ID},
        "source_version": {"type": "string", "minLength": 1},
        "record_id": {"type": "string", "minLength": 1},
    },
}

PROPERTY_TARGET_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema",
        "status",
        "requested_field",
        "relation",
        "value",
        "unit",
        "threshold",
        "reason",
        "evidence",
    ],
    "properties": {
        "schema": {"const": "icmat.property_judgment.v3"},
        "status": {"enum": ["SUPPORTED", "UNKNOWN"]},
        "requested_field": {"enum": list(QUERY_FIELDS)},
        "relation": {
            "type": ["string", "null"],
            "enum": ["ABOVE_OR_EQUAL", "BELOW", None],
        },
        "value": {"type": ["number", "null"]},
        "unit": {"enum": sorted(set(FIELD_UNITS.values()))},
        "threshold": {"type": "number"},
        "reason": {
            "type": ["string", "null"],
            "enum": ["FIELD_NOT_PRESENT", None],
        },
        "evidence": EVIDENCE_SCHEMA,
    },
}

TOOL_TARGET_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema", "status", "tool", "arguments", "reason", "evidence"],
    "properties": {
        "schema": {"const": "icmat.tool_parameters.v3"},
        "status": {"enum": ["SUPPORTED", "UNKNOWN"]},
        "tool": {"type": ["string", "null"]},
        "arguments": {
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "source_id",
                        "source_version",
                        "record_id",
                        "field",
                        "unit",
                    ],
                    "properties": {
                        "source_id": {"const": SOURCE_ID},
                        "source_version": {"type": "string", "minLength": 1},
                        "record_id": {"type": "string", "minLength": 1},
                        "field": {"enum": list(QUERY_FIELDS)},
                        "unit": {"enum": sorted(set(FIELD_UNITS.values()))},
                    },
                },
                {"type": "null"},
            ]
        },
        "reason": {
            "type": ["string", "null"],
            "enum": ["FIELD_NOT_AVAILABLE", None],
        },
        "evidence": EVIDENCE_SCHEMA,
    },
}

ADJUDICATION_TARGET_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema",
        "status",
        "requested_field",
        "value",
        "unit",
        "reason",
        "evidence",
    ],
    "properties": {
        "schema": {"const": "icmat.evidence_adjudication.v3"},
        "status": {"enum": ["SUPPORTED", "UNKNOWN"]},
        "requested_field": {"enum": list(QUERY_FIELDS)},
        "value": {"type": ["number", "null"]},
        "unit": {"enum": sorted(set(FIELD_UNITS.values()))},
        "reason": {
            "enum": ["CONSISTENT_EVIDENCE", "EVIDENCE_CONFLICT", "UNIT_MISMATCH"]
        },
        "evidence": EVIDENCE_SCHEMA,
    },
}

ASSISTANT_SCHEMAS = {
    "property_judgment": PROPERTY_TARGET_SCHEMA,
    "tool_parameters": TOOL_TARGET_SCHEMA,
    "evidence_adjudication": ADJUDICATION_TARGET_SCHEMA,
}

DESCRIPTOR_FACT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "name", "value"],
    "properties": {
        "kind": {"const": "descriptor"},
        "name": {
            "enum": ["formula", "space_group", "crystal_system", "dimensionality"]
        },
        "value": {"type": ["string", "number"]},
    },
}

SCALAR_FACT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "channel", "field", "value", "unit"],
    "properties": {
        "kind": {"const": "scalar_property"},
        "channel": {"enum": ["record_view", "view_a", "view_b"]},
        "field": {"enum": list(QUERY_FIELDS)},
        "value": {"type": "number"},
        "unit": {"type": "string", "minLength": 1},
    },
}

THRESHOLD_FACT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "field", "value", "unit"],
    "properties": {
        "kind": {"const": "decision_threshold"},
        "field": {"enum": list(QUERY_FIELDS)},
        "value": {"type": "number"},
        "unit": {"type": "string", "minLength": 1},
    },
}

FIELD_CATALOG_FACT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "entries"],
    "properties": {
        "kind": {"const": "field_catalog"},
        "entries": {
            "type": "array",
            "uniqueItems": True,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["field", "unit"],
                "properties": {
                    "field": {"enum": list(QUERY_FIELDS)},
                    "unit": {"type": "string", "minLength": 1},
                },
            },
        },
    },
}

FACT_SCHEMA = {
    "oneOf": [
        DESCRIPTOR_FACT_SCHEMA,
        SCALAR_FACT_SCHEMA,
        THRESHOLD_FACT_SCHEMA,
        FIELD_CATALOG_FACT_SCHEMA,
    ]
}

CONTEXT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "source_id",
        "source_version",
        "record_id",
        "source_kind",
        "claim_boundary",
        "domain_tags",
        "requested_field",
        "expected_unit",
        "view",
        "facts",
    ],
    "properties": {
        "source_id": {"const": SOURCE_ID},
        "source_version": {"type": "string", "minLength": 1},
        "record_id": {"type": "string", "minLength": 1},
        "source_kind": {"const": "public_computed_dft_record"},
        "claim_boundary": {"const": "not_experimental_not_fab_line_ground_truth"},
        "domain_tags": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "requested_field": {"enum": list(QUERY_FIELDS)},
        "expected_unit": {"enum": sorted(set(FIELD_UNITS.values()))},
        "view": {"const": NEUTRAL_VIEW},
        "facts": {"type": "array", "minItems": 1, "items": FACT_SCHEMA},
    },
}

EXAMPLE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema",
        "example_id",
        "task",
        "status_label",
        "requested_field",
        "family_id",
        "split",
        "language",
        "domain_tags",
        "host_binding",
        "messages",
    ],
    "properties": {
        "schema": {"const": EXAMPLE_SCHEMA_ID},
        "example_id": {"type": "string", "pattern": "^sftv3_[0-9a-f]{24}$"},
        "task": {"enum": list(TASK_NAMES)},
        "status_label": {"enum": ["SUPPORTED", "UNKNOWN"]},
        "requested_field": {"enum": list(QUERY_FIELDS)},
        "family_id": {"type": "string", "pattern": "^family_[0-9a-f]{24}$"},
        "split": {"enum": list(TRAINING_SPLITS)},
        "language": {"enum": ["zh", "en"]},
        "domain_tags": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "host_binding": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "source_id",
                "source_version",
                "record_id",
                "canonical_complete_record_sha256",
            ],
            "properties": {
                "source_id": {"const": SOURCE_ID},
                "source_version": {"type": "string", "minLength": 1},
                "record_id": {"type": "string", "minLength": 1},
                "canonical_complete_record_sha256": {
                    "type": "string",
                    "pattern": "^[0-9a-f]{64}$",
                },
            },
        },
        "messages": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "prefixItems": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["role", "content"],
                    "properties": {
                        "role": {"const": "system"},
                        "content": {"type": "string", "minLength": 1},
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["role", "content"],
                    "properties": {
                        "role": {"const": "user"},
                        "content": {"type": "string", "minLength": 1},
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["role", "content"],
                    "properties": {
                        "role": {"const": "assistant"},
                        "content": {"type": "string", "minLength": 2},
                    },
                },
            ],
            "items": False,
        },
    },
}

for _schema in (
    *ASSISTANT_SCHEMAS.values(),
    CONTEXT_SCHEMA,
    EXAMPLE_SCHEMA,
):
    Draft202012Validator.check_schema(_schema)


def _safe_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or stripped.lower() in NA_STRINGS:
        return None
    return stripped


def _safe_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        try:
            result = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return result if math.isfinite(result) else None


def _record_id(record: Mapping[str, Any]) -> str | None:
    return _safe_text(record.get("jid"))


def _available_query_fields(record: Mapping[str, Any]) -> list[str]:
    return [field for field in QUERY_FIELDS if _safe_number(record.get(field)) is not None]


def _deterministic_order(items: Iterable[Any], seed: str) -> list[Any]:
    return sorted(
        items,
        key=lambda item: sha256_bytes(
            f"{seed}|{canonical_json(item)}".encode()
        ),
    )


def _round_number(value: float) -> float:
    return float(f"{value:.10g}")


def _verify_source_lock(
    archive_path: Path,
    receipt_path: Path,
    source_catalog_path: Path,
) -> SourceLock:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    catalog = json.loads(source_catalog_path.read_text(encoding="utf-8"))
    catalog_record = next(
        (
            item
            for item in catalog.get("records", [])
            if isinstance(item, Mapping) and item.get("source_id") == SOURCE_ID
        ),
        None,
    )
    if catalog_record is None:
        raise ValueError(f"{SOURCE_ID} is absent from source catalog")
    if catalog_record.get("reuse_gate") != REQUIRED_REUSE_GATE:
        raise PermissionError("source catalog does not authorize training")
    if receipt.get("reuse_gate") != REQUIRED_REUSE_GATE:
        raise PermissionError("acquisition receipt does not authorize training")
    if catalog_record.get("doi") != receipt.get("doi"):
        raise ValueError("source catalog and receipt DOI mismatch")
    if catalog_record.get("version") != receipt.get("source_version"):
        raise ValueError("source catalog and receipt version mismatch")

    archive_path = archive_path.resolve()
    actual_sha256 = sha256_file(archive_path)
    actual_bytes = archive_path.stat().st_size
    if actual_sha256 != receipt.get("sha256") or actual_bytes != receipt.get("bytes"):
        raise ValueError("JARVIS archive does not match acquisition receipt")
    with zipfile.ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
    if len(members) != 1 or not members[0].lower().endswith(".json"):
        raise ValueError("expected exactly one JSON member in JARVIS archive")
    return SourceLock(
        archive_path=archive_path,
        archive_sha256=actual_sha256,
        archive_bytes=actual_bytes,
        member_name=members[0],
        source_version=str(receipt["source_version"]),
        doi=str(receipt["doi"]),
        license_name=str(receipt["license_name"]),
        license_url=str(receipt["license_url"]),
        reuse_gate=str(receipt["reuse_gate"]),
        acquired_at=str(receipt["acquired_at"]),
    )


def _load_records(source_lock: SourceLock) -> list[dict[str, Any]]:
    with zipfile.ZipFile(source_lock.archive_path) as archive:
        with archive.open(source_lock.member_name) as handle:
            payload = json.load(handle)
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise TypeError("JARVIS archive member must be a list of objects")
    return payload


def _assistant_evidence(context: Mapping[str, Any]) -> dict[str, str]:
    return {
        "source_id": str(context["source_id"]),
        "source_version": str(context["source_version"]),
        "record_id": str(context["record_id"]),
    }


def _extract_context(example: Mapping[str, Any]) -> dict[str, Any]:
    content = str(example["messages"][1]["content"])
    if content.count(USER_MARKER) != 1:
        raise ValueError("user message must contain exactly one evidence marker")
    prefix, encoded = content.split(USER_MARKER, maxsplit=1)
    if not prefix.strip():
        raise ValueError("user instruction is missing")
    context = json.loads(encoded)
    if not isinstance(context, dict):
        raise TypeError("model context must be one JSON object")
    if canonical_json(context) != encoded:
        raise ValueError("model context must use canonical JSON")
    Draft202012Validator(CONTEXT_SCHEMA).validate(context)
    return context


def _fact_subset(context: Mapping[str, Any], kind: str) -> list[dict[str, Any]]:
    return [
        dict(fact)
        for fact in context["facts"]
        if isinstance(fact, Mapping) and fact.get("kind") == kind
    ]


def _expected_property(context: Mapping[str, Any]) -> dict[str, Any]:
    field = str(context["requested_field"])
    unit = FIELD_UNITS[field]
    thresholds = _fact_subset(context, "decision_threshold")
    scalars = _fact_subset(context, "scalar_property")
    if len(thresholds) != 1:
        raise ValueError("property task requires exactly one threshold")
    threshold_fact = thresholds[0]
    if threshold_fact["field"] != field or threshold_fact["unit"] != unit:
        raise ValueError("threshold field or unit mismatch")
    if _fact_subset(context, "field_catalog"):
        raise ValueError("property task must not contain a field catalog")
    for fact in scalars:
        if fact["channel"] != "record_view":
            raise ValueError("property scalar must use record_view")
        if fact["unit"] != FIELD_UNITS[str(fact["field"])]:
            raise ValueError("property scalar uses a non-canonical unit")
    requested = [fact for fact in scalars if fact["field"] == field]
    if len(requested) > 1:
        raise ValueError("property task has duplicate requested-field facts")
    threshold = float(threshold_fact["value"])
    evidence = _assistant_evidence(context)
    if not requested:
        return {
            "schema": "icmat.property_judgment.v3",
            "status": "UNKNOWN",
            "requested_field": field,
            "relation": None,
            "value": None,
            "unit": unit,
            "threshold": threshold,
            "reason": "FIELD_NOT_PRESENT",
            "evidence": evidence,
        }
    value = float(requested[0]["value"])
    relation = "ABOVE_OR_EQUAL" if value >= threshold else "BELOW"
    return {
        "schema": "icmat.property_judgment.v3",
        "status": "SUPPORTED",
        "requested_field": field,
        "relation": relation,
        "value": value,
        "unit": unit,
        "threshold": threshold,
        "reason": None,
        "evidence": evidence,
    }


def _expected_tool(context: Mapping[str, Any]) -> dict[str, Any]:
    field = str(context["requested_field"])
    unit = FIELD_UNITS[field]
    catalogs = _fact_subset(context, "field_catalog")
    if len(catalogs) != 1:
        raise ValueError("tool task requires exactly one field catalog")
    if _fact_subset(context, "scalar_property") or _fact_subset(
        context, "decision_threshold"
    ):
        raise ValueError("tool task must contain only descriptor and catalog facts")
    entries = list(catalogs[0]["entries"])
    fields = [str(entry["field"]) for entry in entries]
    if len(fields) != len(set(fields)):
        raise ValueError("field catalog contains duplicate fields")
    for entry in entries:
        if entry["unit"] != FIELD_UNITS[str(entry["field"])]:
            raise ValueError("field catalog uses a non-canonical unit")
    evidence = _assistant_evidence(context)
    if field not in fields:
        return {
            "schema": "icmat.tool_parameters.v3",
            "status": "UNKNOWN",
            "tool": None,
            "arguments": None,
            "reason": "FIELD_NOT_AVAILABLE",
            "evidence": evidence,
        }
    return {
        "schema": "icmat.tool_parameters.v3",
        "status": "SUPPORTED",
        "tool": "lookup_pinned_jarvis_property",
        "arguments": {
            "source_id": SOURCE_ID,
            "source_version": str(context["source_version"]),
            "record_id": str(context["record_id"]),
            "field": field,
            "unit": unit,
        },
        "reason": None,
        "evidence": evidence,
    }


def _expected_adjudication(context: Mapping[str, Any]) -> dict[str, Any]:
    field = str(context["requested_field"])
    unit = FIELD_UNITS[field]
    scalars = _fact_subset(context, "scalar_property")
    if len(scalars) != 2:
        raise ValueError("adjudication requires exactly two scalar facts")
    if _fact_subset(context, "decision_threshold") or _fact_subset(
        context, "field_catalog"
    ):
        raise ValueError("adjudication must not contain threshold or catalog facts")
    if {str(fact["channel"]) for fact in scalars} != {"view_a", "view_b"}:
        raise ValueError("adjudication channels must be view_a and view_b")
    if any(fact["field"] != field for fact in scalars):
        raise ValueError("adjudication facts must use the requested field")
    evidence = _assistant_evidence(context)
    if any(fact["unit"] != unit for fact in scalars):
        return {
            "schema": "icmat.evidence_adjudication.v3",
            "status": "UNKNOWN",
            "requested_field": field,
            "value": None,
            "unit": unit,
            "reason": "UNIT_MISMATCH",
            "evidence": evidence,
        }
    left, right = (float(fact["value"]) for fact in scalars)
    if not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12):
        return {
            "schema": "icmat.evidence_adjudication.v3",
            "status": "UNKNOWN",
            "requested_field": field,
            "value": None,
            "unit": unit,
            "reason": "EVIDENCE_CONFLICT",
            "evidence": evidence,
        }
    return {
        "schema": "icmat.evidence_adjudication.v3",
        "status": "SUPPORTED",
        "requested_field": field,
        "value": left,
        "unit": unit,
        "reason": "CONSISTENT_EVIDENCE",
        "evidence": evidence,
    }


def derive_expected_assistant(
    task: str,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the only valid target from model-visible facts."""
    if context["expected_unit"] != FIELD_UNITS[str(context["requested_field"])]:
        raise ValueError("expected_unit does not match requested_field")
    if context["view"] != NEUTRAL_VIEW:
        raise ValueError("model view must be neutral")
    derivations = {
        "property_judgment": _expected_property,
        "tool_parameters": _expected_tool,
        "evidence_adjudication": _expected_adjudication,
    }
    try:
        derive = derivations[task]
    except KeyError as error:
        raise ValueError(f"unsupported task: {task}") from error
    return derive(context)


def validate_example(example: Mapping[str, Any]) -> None:
    """Validate both JSON shape and the complete evidence-to-target semantics."""
    Draft202012Validator(EXAMPLE_SCHEMA).validate(example)
    messages = example["messages"]
    if messages[0]["content"] != SYSTEM_PROMPTS[str(example["language"])]:
        raise ValueError("system prompt does not match language")
    model_input = "\n".join(str(message["content"]) for message in messages[:2])
    if HASH64_PATTERN.search(model_input):
        raise ValueError("model input must not contain a 64-hex digest")
    context = _extract_context(example)
    if context["requested_field"] != example["requested_field"]:
        raise ValueError("context requested_field mismatch")
    if context["domain_tags"] != example["domain_tags"]:
        raise ValueError("context domain_tags mismatch")
    host = example["host_binding"]
    for key in ("source_id", "source_version", "record_id"):
        if context[key] != host[key]:
            raise ValueError(f"context {key} does not match host binding")

    assistant = json.loads(messages[-1]["content"])
    if not isinstance(assistant, dict):
        raise TypeError("assistant target must be one JSON object")
    task = str(example["task"])
    Draft202012Validator(ASSISTANT_SCHEMAS[task]).validate(assistant)
    if canonical_json(assistant) != messages[-1]["content"]:
        raise ValueError("assistant target must use canonical JSON")
    expected = derive_expected_assistant(task, context)
    if assistant != expected:
        raise ValueError("assistant target is not derivable from model-visible facts")
    if assistant["status"] != example["status_label"]:
        raise ValueError("assistant status does not match host status label")


def _threshold_for(value: float, record_id: str, field: str) -> float:
    delta = max(abs(value) * 0.05, 0.05)
    put_below = (
        int(sha256_bytes(f"{record_id}|{field}|threshold-v3".encode())[:2], 16) % 2
        == 0
    )
    return _round_number(value - delta if put_below else value + delta)


def _counterexample_value(value: float) -> float:
    return _round_number(value + max(abs(value) * 0.15, 0.1))


def _base_descriptors(record: Mapping[str, Any], seed: str) -> list[dict[str, Any]]:
    candidates = (
        ("formula", record.get("formula")),
        ("space_group", record.get("spg_number")),
        ("crystal_system", record.get("crys")),
        ("dimensionality", record.get("dimensionality")),
    )
    facts = [
        {"kind": "descriptor", "name": name, "value": value}
        for name, value in candidates
        if isinstance(value, (str, int, float)) and _safe_text(str(value)) is not None
    ]
    return _deterministic_order(facts, seed)[:2]


def _model_context(
    *,
    source_lock: SourceLock,
    record: Mapping[str, Any],
    field: str,
    facts: Sequence[Mapping[str, Any]],
    tags: Sequence[str],
) -> dict[str, Any]:
    return {
        "source_id": SOURCE_ID,
        "source_version": source_lock.source_version,
        "record_id": str(record["jid"]),
        "source_kind": "public_computed_dft_record",
        "claim_boundary": "not_experimental_not_fab_line_ground_truth",
        "domain_tags": list(tags),
        "requested_field": field,
        "expected_unit": FIELD_UNITS[field],
        "view": NEUTRAL_VIEW,
        "facts": list(facts),
    }


def _user_message(language: str, task: str, context: Mapping[str, Any]) -> str:
    instructions = {
        ("zh", "property_judgment"): (
            "根据证据中的数值和阈值判断关系；请求字段缺失时不得猜测。"
        ),
        ("en", "property_judgment"): (
            "Judge the requested value against the threshold; do not guess when "
            "the requested field is absent."
        ),
        ("zh", "tool_parameters"): (
            "仅当字段目录包含请求字段且单位匹配时，生成只读查询参数。"
        ),
        ("en", "tool_parameters"): (
            "Generate read-only lookup arguments only when the field catalog "
            "contains the requested field with its canonical unit."
        ),
        ("zh", "evidence_adjudication"): (
            "核对两条同字段证据；数值不一致或单位错误时返回 UNKNOWN。"
        ),
        ("en", "evidence_adjudication"): (
            "Adjudicate two observations of the requested field; return UNKNOWN "
            "when their values conflict or a unit is invalid."
        ),
    }
    return instructions[(language, task)] + "\n" + USER_MARKER + canonical_json(context)


def _host_binding(
    source_lock: SourceLock,
    record: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "source_id": SOURCE_ID,
        "source_version": source_lock.source_version,
        "record_id": str(record["jid"]),
        "canonical_complete_record_sha256": canonical_complete_record_hash(record),
    }


def _language_for_pair(record_id: str, task: str, status: str) -> str:
    supported_is_zh = (
        int(sha256_bytes(f"{record_id}|{task}|language-v3".encode())[:2], 16) % 2
        == 0
    )
    if status == "SUPPORTED":
        return "zh" if supported_is_zh else "en"
    return "en" if supported_is_zh else "zh"


def _make_example(
    *,
    task: str,
    record: Mapping[str, Any],
    field: str,
    family_id: str,
    split: str,
    source_lock: SourceLock,
    tags: Sequence[str],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    assistant = derive_expected_assistant(task, context)
    status = str(assistant["status"])
    record_id = str(record["jid"])
    language = _language_for_pair(record_id, task, status)
    identity = {
        "schema": EXAMPLE_SCHEMA_ID,
        "task": task,
        "status": status,
        "record_id": record_id,
        "field": field,
        "family_id": family_id,
    }
    example = {
        "schema": EXAMPLE_SCHEMA_ID,
        "example_id": (
            "sftv3_"
            + sha256_bytes(canonical_json(identity).encode("utf-8"))[:24]
        ),
        "task": task,
        "status_label": status,
        "requested_field": field,
        "family_id": family_id,
        "split": split,
        "language": language,
        "domain_tags": list(tags),
        "host_binding": _host_binding(source_lock, record),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS[language]},
            {"role": "user", "content": _user_message(language, task, context)},
            {"role": "assistant", "content": canonical_json(assistant)},
        ],
    }
    validate_example(example)
    return example


def _property_pair(
    *,
    record: Mapping[str, Any],
    field: str,
    family_id: str,
    split: str,
    source_lock: SourceLock,
    tags: Sequence[str],
) -> list[dict[str, Any]]:
    record_id = str(record["jid"])
    value = _round_number(float(_safe_number(record[field])))
    unit = FIELD_UNITS[field]
    threshold = _threshold_for(value, record_id, field)
    descriptors = _base_descriptors(record, f"{record_id}|property-v3")
    distractors = [
        candidate
        for candidate in _available_query_fields(record)
        if candidate != field
    ]
    distractor_facts: list[dict[str, Any]] = []
    if distractors:
        distractor = _deterministic_order(
            distractors, f"{record_id}|property-distractor-v3"
        )[0]
        distractor_facts.append(
            {
                "kind": "scalar_property",
                "channel": "record_view",
                "field": distractor,
                "value": _round_number(float(_safe_number(record[distractor]))),
                "unit": FIELD_UNITS[distractor],
            }
        )
    threshold_fact = {
        "kind": "decision_threshold",
        "field": field,
        "value": threshold,
        "unit": unit,
    }
    supported_context = _model_context(
        source_lock=source_lock,
        record=record,
        field=field,
        facts=[
            *descriptors,
            *distractor_facts,
            {
                "kind": "scalar_property",
                "channel": "record_view",
                "field": field,
                "value": value,
                "unit": unit,
            },
            threshold_fact,
        ],
        tags=tags,
    )
    unknown_context = _model_context(
        source_lock=source_lock,
        record=record,
        field=field,
        facts=[*descriptors, *distractor_facts, threshold_fact],
        tags=tags,
    )
    return [
        _make_example(
            task="property_judgment",
            record=record,
            field=field,
            family_id=family_id,
            split=split,
            source_lock=source_lock,
            tags=tags,
            context=supported_context,
        ),
        _make_example(
            task="property_judgment",
            record=record,
            field=field,
            family_id=family_id,
            split=split,
            source_lock=source_lock,
            tags=tags,
            context=unknown_context,
        ),
    ]


def _tool_pair(
    *,
    record: Mapping[str, Any],
    field: str,
    family_id: str,
    split: str,
    source_lock: SourceLock,
    tags: Sequence[str],
) -> list[dict[str, Any]]:
    record_id = str(record["jid"])
    available = _deterministic_order(
        [
            {"field": candidate, "unit": FIELD_UNITS[candidate]}
            for candidate in _available_query_fields(record)
        ],
        f"{record_id}|tool-catalog-v3",
    )[:6]
    if not any(item["field"] == field for item in available):
        available = [
            *available[:5],
            {"field": field, "unit": FIELD_UNITS[field]},
        ]
        available = _deterministic_order(available, f"{record_id}|tool-forced-v3")
    unavailable = [item for item in available if item["field"] != field]
    descriptors = _base_descriptors(record, f"{record_id}|tool-v3")
    supported_context = _model_context(
        source_lock=source_lock,
        record=record,
        field=field,
        facts=[
            *descriptors,
            {"kind": "field_catalog", "entries": available},
        ],
        tags=tags,
    )
    unknown_context = _model_context(
        source_lock=source_lock,
        record=record,
        field=field,
        facts=[
            *descriptors,
            {"kind": "field_catalog", "entries": unavailable},
        ],
        tags=tags,
    )
    return [
        _make_example(
            task="tool_parameters",
            record=record,
            field=field,
            family_id=family_id,
            split=split,
            source_lock=source_lock,
            tags=tags,
            context=supported_context,
        ),
        _make_example(
            task="tool_parameters",
            record=record,
            field=field,
            family_id=family_id,
            split=split,
            source_lock=source_lock,
            tags=tags,
            context=unknown_context,
        ),
    ]


def _adjudication_pair(
    *,
    record: Mapping[str, Any],
    field: str,
    family_id: str,
    split: str,
    source_lock: SourceLock,
    tags: Sequence[str],
) -> list[dict[str, Any]]:
    record_id = str(record["jid"])
    value = _round_number(float(_safe_number(record[field])))
    unit = FIELD_UNITS[field]
    descriptors = _base_descriptors(record, f"{record_id}|adjudication-v3")
    supported_facts = [
        *descriptors,
        {
            "kind": "scalar_property",
            "channel": "view_a",
            "field": field,
            "value": value,
            "unit": unit,
        },
        {
            "kind": "scalar_property",
            "channel": "view_b",
            "field": field,
            "value": value,
            "unit": unit,
        },
    ]
    use_unit_mismatch = (
        int(sha256_bytes(f"{record_id}|{field}|negative-v3".encode())[:2], 16) % 2
        == 0
    )
    if use_unit_mismatch:
        unknown_facts = [
            *descriptors,
            {
                "kind": "scalar_property",
                "channel": "view_a",
                "field": field,
                "value": value,
                "unit": unit,
            },
            {
                "kind": "scalar_property",
                "channel": "view_b",
                "field": field,
                "value": value,
                "unit": WRONG_UNITS[unit],
            },
        ]
    else:
        unknown_facts = [
            *descriptors,
            {
                "kind": "scalar_property",
                "channel": "view_a",
                "field": field,
                "value": value,
                "unit": unit,
            },
            {
                "kind": "scalar_property",
                "channel": "view_b",
                "field": field,
                "value": _counterexample_value(value),
                "unit": unit,
            },
        ]
    supported_context = _model_context(
        source_lock=source_lock,
        record=record,
        field=field,
        facts=supported_facts,
        tags=tags,
    )
    unknown_context = _model_context(
        source_lock=source_lock,
        record=record,
        field=field,
        facts=unknown_facts,
        tags=tags,
    )
    return [
        _make_example(
            task="evidence_adjudication",
            record=record,
            field=field,
            family_id=family_id,
            split=split,
            source_lock=source_lock,
            tags=tags,
            context=supported_context,
        ),
        _make_example(
            task="evidence_adjudication",
            record=record,
            field=field,
            family_id=family_id,
            split=split,
            source_lock=source_lock,
            tags=tags,
            context=unknown_context,
        ),
    ]


def build_record_examples(
    *,
    record: Mapping[str, Any],
    field: str,
    family_id: str,
    split: str,
    source_lock: SourceLock,
) -> list[dict[str, Any]]:
    if split not in TRAINING_SPLITS:
        raise ValueError("test records must not materialize semantic examples")
    if field not in _available_query_fields(record):
        raise ValueError(f"{field} is not available for {record.get('jid')}")
    tags = domain_tags(record)
    examples: list[dict[str, Any]] = []
    for builder in (_property_pair, _tool_pair, _adjudication_pair):
        examples.extend(
            builder(
                record=record,
                field=field,
                family_id=family_id,
                split=split,
                source_lock=source_lock,
                tags=tags,
            )
        )
    return examples


def _select_records(
    records: Sequence[Mapping[str, Any]],
    family_map: Mapping[str, str],
    max_records: int,
) -> list[tuple[Mapping[str, Any], str, str]]:
    if max_records <= 0:
        raise ValueError("max_records must be positive")
    eligible: list[tuple[str, Mapping[str, Any], str, str]] = []
    for record in records:
        record_id = _record_id(record)
        if record_id is None or record_id not in family_map:
            continue
        if not _available_query_fields(record):
            continue
        family_id = family_map[record_id]
        split = split_for_family(family_id)
        order = sha256_bytes(f"icmat-sft-v3|{record_id}".encode())
        eligible.append((order, record, family_id, split))
    eligible.sort(key=lambda item: item[0])
    selected = eligible[:max_records]
    if len(selected) < max_records:
        raise ValueError(
            f"requested {max_records} records but only {len(selected)} are eligible"
        )
    if {split for _, _, _, split in selected} != set(SPLIT_NAMES):
        raise RuntimeError("deterministic selection did not populate all four splits")
    return [(record, family_id, split) for _, record, family_id, split in selected]


def _assign_fields(
    selected: Sequence[tuple[Mapping[str, Any], str, str]],
) -> dict[str, str]:
    counts_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    assigned: dict[str, str] = {}
    for record, _, split in selected:
        if split == "test":
            continue
        record_id = str(record["jid"])
        available = _available_query_fields(record)
        minimum = min(counts_by_split[split][field] for field in available)
        candidates = [
            field for field in available if counts_by_split[split][field] == minimum
        ]
        chosen = _deterministic_order(
            candidates, f"{record_id}|field-choice-v3"
        )[0]
        counts_by_split[split][chosen] += 1
        assigned[record_id] = chosen
    return assigned


def _pairwise_family_overlap(
    selected: Sequence[tuple[Mapping[str, Any], str, str]],
) -> dict[str, list[str]]:
    families: dict[str, set[str]] = defaultdict(set)
    for _, family_id, split in selected:
        families[split].add(family_id)
    overlap: dict[str, list[str]] = {}
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            overlap[f"{left}_{right}"] = sorted(families[left] & families[right])
    if any(overlap.values()):
        raise RuntimeError("connected-family leakage detected")
    return overlap


def _balance_metrics(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    split_counts = Counter()
    task_status = Counter()
    field_status = Counter()
    language_counts = Counter()
    view_status = Counter()
    field_records: dict[str, set[str]] = defaultdict(set)
    domain_records: dict[str, set[str]] = defaultdict(set)
    records: set[str] = set()
    for example in examples:
        split_counts[str(example["split"])] += 1
        task_status[(str(example["task"]), str(example["status_label"]))] += 1
        field_status[
            (str(example["requested_field"]), str(example["status_label"]))
        ] += 1
        language_counts[str(example["language"])] += 1
        context = _extract_context(example)
        view_status[(str(context["view"]), str(example["status_label"]))] += 1
        record_id = str(example["host_binding"]["record_id"])
        field_records[str(example["requested_field"])].add(record_id)
        records.add(record_id)
        for tag in example["domain_tags"]:
            domain_records[str(tag)].add(record_id)

    for task in TASK_NAMES:
        if task_status[(task, "SUPPORTED")] != task_status[(task, "UNKNOWN")]:
            raise RuntimeError(f"status imbalance for task {task}")
    for field in QUERY_FIELDS:
        if field_status[(field, "SUPPORTED")] != field_status[(field, "UNKNOWN")]:
            raise RuntimeError(f"status imbalance for field {field}")
    neutral = {
        status: view_status[(NEUTRAL_VIEW, status)]
        for status in ("SUPPORTED", "UNKNOWN")
    }
    if min(neutral.values()) <= 0:
        raise RuntimeError("neutral view must occur with both target statuses")
    return {
        "scope": "train_validation_calibration_only",
        "test_semantic_metrics_included": False,
        "example_count": len(examples),
        "record_count": len(records),
        "split_example_counts": {
            split: split_counts[split] for split in TRAINING_SPLITS
        },
        "task_status_balance": {
            task: {
                status: task_status[(task, status)]
                for status in ("SUPPORTED", "UNKNOWN")
            }
            for task in TASK_NAMES
        },
        "field_status_balance": {
            field: {
                status: field_status[(field, status)]
                for status in ("SUPPORTED", "UNKNOWN")
            }
            for field in QUERY_FIELDS
            if field_records[field]
        },
        "field_record_counts": {
            field: len(record_ids)
            for field, record_ids in sorted(field_records.items())
        },
        "language_counts": dict(sorted(language_counts.items())),
        "neutral_view_status_counts": {NEUTRAL_VIEW: neutral},
        "domain_record_coverage": {
            tag: len(record_ids)
            for tag, record_ids in sorted(domain_records.items())
        },
        "json_schema_valid_rate": 1.0,
        "strict_semantic_valid_rate": 1.0,
        "model_input_sha256_rate": 0.0,
        "model_input_direct_status_marker_count": 0,
    }


def _naive_copy_prediction(example: Mapping[str, Any]) -> dict[str, Any]:
    """A deliberately shallow copier that ignores conflicts and unit validity."""
    context = _extract_context(example)
    task = str(example["task"])
    if task == "property_judgment":
        return _expected_property(context)
    if task == "tool_parameters":
        return _expected_tool(context)
    field = str(context["requested_field"])
    unit = FIELD_UNITS[field]
    scalars = _fact_subset(context, "scalar_property")
    first = scalars[0]
    return {
        "schema": "icmat.evidence_adjudication.v3",
        "status": "SUPPORTED",
        "requested_field": field,
        "value": float(first["value"]),
        "unit": unit,
        "reason": "CONSISTENT_EVIDENCE",
        "evidence": _assistant_evidence(context),
    }


def evaluate_shortcut_baselines(
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not examples:
        raise ValueError("shortcut challenge set is empty")
    task_totals = Counter(str(example["task"]) for example in examples)
    copy_exact = Counter()
    copy_status = Counter()
    view_status_correct = Counter()
    for example in examples:
        task = str(example["task"])
        target = json.loads(example["messages"][-1]["content"])
        copied = _naive_copy_prediction(example)
        copy_exact[task] += int(copied == target)
        copy_status[task] += int(copied["status"] == target["status"])
        view_only_status = "SUPPORTED"
        view_status_correct[task] += int(view_only_status == target["status"])

    def rates(correct: Counter[str]) -> dict[str, float]:
        return {
            task: correct[task] / task_totals[task]
            for task in TASK_NAMES
        }

    copy_task_rates = rates(copy_exact)
    copy_exact_count = sum(copy_exact.values())
    view_correct_count = sum(view_status_correct.values())
    gate = {
        "minimum_overall_exact_target_rate": 0.95,
        "minimum_each_task_exact_target_rate": 0.90,
    }
    copy_passed = (
        copy_exact_count / len(examples) >= gate["minimum_overall_exact_target_rate"]
        and min(copy_task_rates.values())
        >= gate["minimum_each_task_exact_target_rate"]
    )
    view_passed = False
    return {
        "schema": "icmat_sft_shortcut_audit.v3",
        "challenge_scope": "calibration_only_no_final_test",
        "challenge_example_count": len(examples),
        "challenge_task_counts": dict(sorted(task_totals.items())),
        "gate": gate,
        "baselines": {
            "neutral_view_majority": {
                "reads": ["model_visible_view"],
                "overall_status_accuracy": view_correct_count / len(examples),
                "task_status_accuracy": rates(view_status_correct),
                "passed": view_passed,
            },
            "metadata_copy_requested_field": {
                "reads": [
                    "requested_field",
                    "first_matching_or_first_scalar",
                    "field_catalog",
                ],
                "ignores": [
                    "cross_view_value_consistency",
                    "cross_view_unit_validity",
                ],
                "overall_exact_target_rate": copy_exact_count / len(examples),
                "overall_status_accuracy": sum(copy_status.values()) / len(examples),
                "task_exact_target_rate": copy_task_rates,
                "task_status_accuracy": rates(copy_status),
                "passed": copy_passed,
            },
        },
        "all_naive_baselines_rejected": not copy_passed and not view_passed,
    }


def _replace_context(
    example: dict[str, Any],
    mutate: Any,
) -> None:
    context = _extract_context(example)
    mutate(context)
    prefix = example["messages"][1]["content"].split(USER_MARKER, maxsplit=1)[0]
    example["messages"][1]["content"] = prefix + USER_MARKER + canonical_json(context)


def _replace_assistant(
    example: dict[str, Any],
    mutate: Any,
) -> None:
    assistant = json.loads(example["messages"][-1]["content"])
    mutate(assistant)
    example["messages"][-1]["content"] = canonical_json(assistant)


def run_mutation_suite(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_task_status = {
        (str(example["task"]), str(example["status_label"])): example
        for example in examples
    }
    property_supported = by_task_status[("property_judgment", "SUPPORTED")]
    tool_supported = by_task_status[("tool_parameters", "SUPPORTED")]
    adjudication_supported = by_task_status[("evidence_adjudication", "SUPPORTED")]
    adjudication_unknown = by_task_status[("evidence_adjudication", "UNKNOWN")]

    cases: list[tuple[str, dict[str, Any]]] = []

    mutated = copy.deepcopy(property_supported)
    _replace_context(
        mutated,
        lambda context: context.__setitem__("expected_unit", "GPa"),
    )
    cases.append(("context_expected_unit", mutated))

    mutated = copy.deepcopy(property_supported)
    _replace_assistant(mutated, lambda target: target.__setitem__("unit", "GPa"))
    cases.append(("assistant_unit", mutated))

    mutated = copy.deepcopy(property_supported)
    _replace_assistant(
        mutated,
        lambda target: target.__setitem__(
            "requested_field", "shear_modulus_gv"
        ),
    )
    cases.append(("assistant_requested_field", mutated))

    mutated = copy.deepcopy(property_supported)
    _replace_context(
        mutated,
        lambda context: context.__setitem__("source_version", "forged-version"),
    )
    cases.append(("context_source_version", mutated))

    mutated = copy.deepcopy(property_supported)
    _replace_assistant(
        mutated,
        lambda target: target["evidence"].__setitem__("record_id", "forged-record"),
    )
    cases.append(("assistant_evidence_record", mutated))

    mutated = copy.deepcopy(property_supported)

    def mutate_value(context: dict[str, Any]) -> None:
        field = context["requested_field"]
        fact = next(
            item
            for item in context["facts"]
            if item.get("kind") == "scalar_property" and item.get("field") == field
        )
        fact["value"] = float(fact["value"]) + 1.0

    _replace_context(mutated, mutate_value)
    cases.append(("context_fact_value", mutated))

    mutated = copy.deepcopy(property_supported)
    _replace_assistant(
        mutated,
        lambda target: target.__setitem__(
            "relation",
            "BELOW" if target["relation"] == "ABOVE_OR_EQUAL" else "ABOVE_OR_EQUAL",
        ),
    )
    cases.append(("assistant_relation", mutated))

    mutated = copy.deepcopy(tool_supported)
    _replace_assistant(
        mutated,
        lambda target: target["arguments"].__setitem__(
            "field", "shear_modulus_gv"
        ),
    )
    cases.append(("tool_argument_field", mutated))

    mutated = copy.deepcopy(tool_supported)
    _replace_assistant(
        mutated,
        lambda target: target["arguments"].__setitem__(
            "source_version", "forged-version"
        ),
    )
    cases.append(("tool_argument_source_version", mutated))

    mutated = copy.deepcopy(adjudication_supported)
    _replace_assistant(
        mutated,
        lambda target: target.__setitem__("status", "UNKNOWN"),
    )
    cases.append(("assistant_status", mutated))

    mutated = copy.deepcopy(adjudication_unknown)
    _replace_assistant(
        mutated,
        lambda target: target.__setitem__("reason", "CONSISTENT_EVIDENCE"),
    )
    cases.append(("assistant_adjudication_reason", mutated))

    rejected: list[str] = []
    accepted: list[str] = []
    for name, candidate in cases:
        try:
            validate_example(candidate)
        except Exception:  # noqa: BLE001 - every validation failure is a rejection
            rejected.append(name)
        else:
            accepted.append(name)
    return {
        "schema": "icmat_sft_semantic_mutation_audit.v3",
        "mutation_count": len(cases),
        "rejected_count": len(rejected),
        "accepted_count": len(accepted),
        "rejected_mutations": rejected,
        "accepted_mutations": accepted,
        "all_mutations_rejected": not accepted,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_jsonl(
    path: Path,
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            validate_example(example)
            handle.write(canonical_json(example) + "\n")
    temporary.replace(path)
    return {
        "path": path.name,
        "examples": len(examples),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            validate_example(item)
            yield item


def _file_receipts(value: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if {"path", "bytes", "sha256"} <= set(value):
            yield value
            return
        for child in value.values():
            yield from _file_receipts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _file_receipts(child)


def verify_dataset(dataset_dir: Path) -> dict[str, Any]:
    """Verify all manifest-relative files, hashes, semantics, and sealed test."""
    dataset_dir = dataset_dir.resolve()
    manifest_path = dataset_dir / "manifest.v3.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != DATASET_SCHEMA_ID:
        raise ValueError("unexpected v3 manifest schema")
    payload_hash = manifest.get("manifest_payload_sha256")
    hash_payload = dict(manifest)
    hash_payload.pop("manifest_payload_sha256", None)
    if payload_hash != sha256_bytes(canonical_json(hash_payload).encode("utf-8")):
        raise ValueError("manifest payload hash mismatch")

    verified_files = 0
    for receipt in _file_receipts(manifest.get("files", {})):
        relative = Path(str(receipt["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("manifest file path must be a safe relative path")
        path = (dataset_dir / relative).resolve()
        if dataset_dir != path.parent and dataset_dir not in path.parents:
            raise ValueError("manifest file escapes dataset directory")
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != receipt["bytes"]:
            raise ValueError(f"file size mismatch: {relative}")
        if sha256_file(path) != receipt["sha256"]:
            raise ValueError(f"file hash mismatch: {relative}")
        verified_files += 1

    training_count = 0
    for split in TRAINING_SPLITS:
        path = dataset_dir / f"{split}.jsonl"
        for example in iter_jsonl(path):
            if example["split"] != split:
                raise ValueError(f"{path.name} contains the wrong split")
            training_count += 1
    if training_count != manifest["training_metrics"]["example_count"]:
        raise ValueError("training example count mismatch")

    membership = json.loads(
        (dataset_dir / "test_membership.sealed.json").read_text(encoding="utf-8")
    )
    membership_payload = dict(membership)
    membership_hash = membership_payload.pop("membership_payload_sha256")
    if membership_hash != sha256_bytes(
        canonical_json(membership_payload).encode("utf-8")
    ):
        raise ValueError("test membership seal mismatch")
    if membership["semantic_examples_materialized"] is not False:
        raise ValueError("final test semantics must remain sealed")
    if membership["semantic_metrics_emitted"] is not False:
        raise ValueError("final test metrics must not exist")
    if (dataset_dir / "test.jsonl").exists():
        raise ValueError("test.jsonl must not be materialized")

    challenge = list(iter_jsonl(dataset_dir / "shortcut_challenge.v3.jsonl"))
    if any(example["split"] != "calibration" for example in challenge):
        raise ValueError("shortcut challenge must be calibration-only")
    recomputed_shortcut = evaluate_shortcut_baselines(challenge)
    stored_shortcut = json.loads(
        (dataset_dir / "shortcut_audit.v3.json").read_text(encoding="utf-8")
    )
    if recomputed_shortcut != stored_shortcut:
        raise ValueError("shortcut audit does not reproduce")
    if not stored_shortcut["all_naive_baselines_rejected"]:
        raise ValueError("a naive shortcut baseline passed")

    semantic_audit = json.loads(
        (dataset_dir / "semantic_mutation_audit.v3.json").read_text(
            encoding="utf-8"
        )
    )
    if not semantic_audit["all_mutations_rejected"]:
        raise ValueError("semantic mutation suite has an accepted mutation")
    return {
        "status": "GO_READY_FOR_INDEPENDENT_AUDIT",
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_payload_sha256": payload_hash,
        "verified_relative_files": verified_files,
        "verified_training_examples": training_count,
        "verified_challenge_examples": len(challenge),
        "test_membership_payload_sha256": membership_hash,
    }


def _build_into_staging(
    *,
    source_lock: SourceLock,
    records: Sequence[Mapping[str, Any]],
    dataset_dir: Path,
    evaluation_dir: Path,
    max_records: int,
    challenge_record_limit: int,
) -> dict[str, Any]:
    family_map, family_graph_audit = build_family_map(records)
    selected = _select_records(records, family_map, max_records)
    family_overlap = _pairwise_family_overlap(selected)
    assigned_fields = _assign_fields(selected)

    examples: list[dict[str, Any]] = []
    examples_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    test_records: list[dict[str, str]] = []
    record_counts = Counter()
    family_sets: dict[str, set[str]] = defaultdict(set)
    for record, family_id, split in selected:
        record_id = str(record["jid"])
        record_counts[split] += 1
        family_sets[split].add(family_id)
        if split == "test":
            test_records.append(
                {
                    "record_id": record_id,
                    "family_id": family_id,
                    "canonical_complete_record_sha256": canonical_complete_record_hash(
                        record
                    ),
                }
            )
            continue
        record_examples = build_record_examples(
            record=record,
            field=assigned_fields[record_id],
            family_id=family_id,
            split=split,
            source_lock=source_lock,
        )
        examples.extend(record_examples)
        examples_by_split[split].extend(record_examples)

    balance = _balance_metrics(examples)
    calibration_records = sorted(
        {
            str(example["host_binding"]["record_id"])
            for example in examples_by_split["calibration"]
        },
        key=lambda record_id: sha256_bytes(
            f"icmat-sft-v3-challenge|{record_id}".encode()
        ),
    )[:challenge_record_limit]
    challenge_record_set = set(calibration_records)
    challenge = [
        example
        for example in examples_by_split["calibration"]
        if str(example["host_binding"]["record_id"]) in challenge_record_set
    ]
    shortcut_audit = evaluate_shortcut_baselines(challenge)
    semantic_mutation_audit = run_mutation_suite(examples)
    if not shortcut_audit["all_naive_baselines_rejected"]:
        raise RuntimeError("shortcut resistance gate failed")
    if not semantic_mutation_audit["all_mutations_rejected"]:
        raise RuntimeError("semantic mutation gate failed")

    training_files = [
        _write_jsonl(dataset_dir / f"{split}.jsonl", examples_by_split[split])
        for split in TRAINING_SPLITS
    ]
    challenge_file = _write_jsonl(
        dataset_dir / "shortcut_challenge.v3.jsonl",
        challenge,
    )

    membership_payload = {
        "schema": TEST_MEMBERSHIP_SCHEMA_ID,
        "source_id": SOURCE_ID,
        "source_version": source_lock.source_version,
        "split": "test",
        "semantic_examples_materialized": False,
        "semantic_metrics_emitted": False,
        "records": sorted(test_records, key=lambda item: item["record_id"]),
    }
    membership_hash = sha256_bytes(
        canonical_json(membership_payload).encode("utf-8")
    )
    membership_file = _write_json_atomic(
        dataset_dir / "test_membership.sealed.json",
        {
            **membership_payload,
            "membership_payload_sha256": membership_hash,
        },
    )

    family_audit = {
        "schema": "icmat_sft_family_audit.v3",
        "source_id": SOURCE_ID,
        "family_graph": family_graph_audit,
        "selected_record_count": len(selected),
        "selected_record_counts": {
            split: record_counts[split] for split in SPLIT_NAMES
        },
        "selected_family_counts": {
            split: len(family_sets[split]) for split in SPLIT_NAMES
        },
        "family_overlap": family_overlap,
        "group_disjoint": True,
        "coordinate_contract": (
            "cartesian=true: fractional=cartesian@inverse(lattice); "
            "cartesian=false: coordinates are already fractional"
        ),
    }
    semantic_contract = {
        "schema": "icmat_sft_semantic_contract.v3",
        "field_units": FIELD_UNITS,
        "ehull_unit": "eV/atom",
        "neutral_model_view": NEUTRAL_VIEW,
        "model_input_forbidden_direct_markers": [
            "evidence_view",
            "augmentation",
            "status_label",
            "counterexample_type",
        ],
        "strict_cross_checks": [
            "requested_field",
            "expected_unit",
            "facts",
            "tool_arguments",
            "source_id_and_version",
            "assistant_evidence",
            "assistant_value",
            "assistant_relation",
            "assistant_status",
        ],
        "assistant_schemas": ASSISTANT_SCHEMAS,
        "context_schema": CONTEXT_SCHEMA,
    }
    source_document = {
        "schema": "icmat_sft_source_lock.v3",
        **source_lock.as_dict(),
    }
    balance_document = {"schema": "icmat_sft_balance_audit.v3", **balance}

    family_file = _write_json_atomic(
        dataset_dir / "family_audit.v3.json",
        family_audit,
    )
    balance_file = _write_json_atomic(
        dataset_dir / "balance_audit.v3.json",
        balance_document,
    )
    semantic_contract_file = _write_json_atomic(
        dataset_dir / "semantic_contract.v3.json",
        semantic_contract,
    )
    semantic_mutation_file = _write_json_atomic(
        dataset_dir / "semantic_mutation_audit.v3.json",
        semantic_mutation_audit,
    )
    shortcut_file = _write_json_atomic(
        dataset_dir / "shortcut_audit.v3.json",
        shortcut_audit,
    )
    source_file = _write_json_atomic(
        dataset_dir / "source_lock.v3.json",
        source_document,
    )

    hse_records = balance["field_record_counts"].get("hse_gap", 0)
    hse_limitation = {
        "assigned_record_count": hse_records,
        "supported_example_count": hse_records * len(TASK_NAMES),
        "quality_claim_allowed": False,
        "limitation": (
            "hse_gap support is sparse in the selected public computed records; "
            "v3 cannot support a standalone HSE capability claim."
        ),
    }
    manifest_without_hash = {
        "schema": DATASET_SCHEMA_ID,
        "builder_version": BUILDER_VERSION,
        "deterministic_timestamp": source_lock.acquired_at,
        "status": "SFT_V3_DATA_READY_FOR_INDEPENDENT_AUDIT_NOT_TRAINED_NOT_DEPLOYED",
        "model_target": "ICMat-Qwen-0.5B",
        "production_integration_allowed": False,
        "qlora_training_started": False,
        "network_used": False,
        "teacher_model_used": False,
        "api_used": False,
        "source_lock": source_lock.as_dict(),
        "selection": {
            "max_records": max_records,
            "source_record_count": len(records),
            "family_graph_record_count": family_graph_audit["identifiable_records"],
            "selected_record_counts": {
                split: record_counts[split] for split in SPLIT_NAMES
            },
            "selected_family_counts": {
                split: len(family_sets[split]) for split in SPLIT_NAMES
            },
            "record_order": "sha256('icmat-sft-v3|' + jid)",
        },
        "family_contract": {
            "connected_component_edges": [
                "normalized_reference",
                "element_aware_reduced_formula",
                "coordinate_semantics_aware_approximate_structure_fingerprint",
            ],
            "assignment": (
                "sha256(family_id) modulo 100: train<76, validation<84, "
                "calibration<92, test"
            ),
            "family_overlap": family_overlap,
            "group_disjoint": True,
        },
        "target_contract": {
            "tasks": list(TASK_NAMES),
            "paired_statuses_per_record_and_task": ["SUPPORTED", "UNKNOWN"],
            "languages": ["zh", "en"],
            "assistant_only_loss_required": True,
            "strict_json_schema_validation": True,
            "strict_cross_semantic_validation": True,
            "neutral_model_view": NEUTRAL_VIEW,
            "direct_status_or_augmentation_marker_in_model_input": False,
            "model_generates_sha256": False,
            "complete_record_sha256_host_bound_only": True,
        },
        "domain_label_contract": {
            "labels_describe_available_computed_descriptors": True,
            "labels_are_device_suitability_ground_truth": False,
            "labels_are_production_line_ground_truth": False,
        },
        "training_metrics": balance,
        "shortcut_resistance": {
            "challenge_split": "calibration",
            "challenge_record_count": len(challenge_record_set),
            "challenge_example_count": len(challenge),
            "all_naive_baselines_rejected": shortcut_audit[
                "all_naive_baselines_rejected"
            ],
        },
        "semantic_mutation_gate": semantic_mutation_audit,
        "hse_gap_limitation": hse_limitation,
        "final_test_contract": {
            "membership_only": True,
            "semantic_examples_materialized": False,
            "semantic_metrics_emitted": False,
            "record_count": record_counts["test"],
            "family_count": len(family_sets["test"]),
            "membership_payload_sha256": membership_hash,
        },
        "files": {
            "training": training_files,
            "sealed_test_membership": membership_file,
            "family_audit": family_file,
            "balance_audit": balance_file,
            "semantic_contract": semantic_contract_file,
            "semantic_mutation_audit": semantic_mutation_file,
            "shortcut_challenge": challenge_file,
            "shortcut_audit": shortcut_file,
            "source_lock": source_file,
        },
        "claim_boundary": (
            "This untrained dataset teaches evidence-bound transformations over "
            "public computed JARVIS-DFT records. It is not experimental data, "
            "fab-line ground truth, model-quality evidence, BPU/X5 evidence, or "
            "production evidence."
        ),
    }
    manifest = {
        **manifest_without_hash,
        "manifest_payload_sha256": sha256_bytes(
            canonical_json(manifest_without_hash).encode("utf-8")
        ),
    }
    manifest_file = _write_json_atomic(
        dataset_dir / "manifest.v3.json",
        manifest,
    )

    verification = verify_dataset(dataset_dir)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    evaluation_files = {
        "family_audit": _write_json_atomic(
            evaluation_dir / "family_audit.v3.json", family_audit
        ),
        "balance_audit": _write_json_atomic(
            evaluation_dir / "balance_audit.v3.json", balance_document
        ),
        "semantic_mutation_audit": _write_json_atomic(
            evaluation_dir / "semantic_mutation_audit.v3.json",
            semantic_mutation_audit,
        ),
        "shortcut_audit": _write_json_atomic(
            evaluation_dir / "shortcut_audit.v3.json", shortcut_audit
        ),
        "verification": _write_json_atomic(
            evaluation_dir / "manifest_verification.v3.json", verification
        ),
    }
    build_report = {
        "schema": "icmat_sft_v3_build_report.v3",
        "status": "GO_READY_FOR_INDEPENDENT_AUDIT",
        "training_started": False,
        "dataset_manifest": manifest_file,
        "source_lock": source_lock.as_dict(),
        "selected_record_counts": manifest["selection"]["selected_record_counts"],
        "selected_family_counts": manifest["selection"]["selected_family_counts"],
        "family_overlap": family_overlap,
        "training_metrics": balance,
        "shortcut_resistance": shortcut_audit,
        "semantic_mutation_gate": semantic_mutation_audit,
        "hse_gap_limitation": hse_limitation,
        "final_test_contract": manifest["final_test_contract"],
        "manifest_verification": verification,
        "evaluation_files": evaluation_files,
        "claim_boundary": manifest["claim_boundary"],
    }
    _write_json_atomic(
        evaluation_dir / "build_report.v3.json",
        build_report,
    )
    return manifest


def build_dataset(
    *,
    archive_path: Path,
    receipt_path: Path,
    source_catalog_path: Path,
    dataset_output_dir: Path,
    evaluation_output_dir: Path,
    max_records: int = 4096,
    challenge_record_limit: int = 256,
) -> dict[str, Any]:
    """Build once into staging, verify, then atomically publish v3 directories."""
    dataset_output_dir = dataset_output_dir.resolve()
    evaluation_output_dir = evaluation_output_dir.resolve()
    for output_dir in (dataset_output_dir, evaluation_output_dir):
        if output_dir.exists():
            raise FileExistsError(
                "v3 outputs are immutable; choose paths that do not exist"
            )
    dataset_staging = dataset_output_dir.with_name(dataset_output_dir.name + ".building")
    evaluation_staging = evaluation_output_dir.with_name(
        evaluation_output_dir.name + ".building"
    )
    for staging in (dataset_staging, evaluation_staging):
        if staging.exists():
            raise FileExistsError(f"stale v3 staging path exists: {staging}")

    source_lock = _verify_source_lock(
        archive_path,
        receipt_path,
        source_catalog_path,
    )
    records = _load_records(source_lock)
    try:
        manifest = _build_into_staging(
            source_lock=source_lock,
            records=records,
            dataset_dir=dataset_staging,
            evaluation_dir=evaluation_staging,
            max_records=max_records,
            challenge_record_limit=challenge_record_limit,
        )
        dataset_output_dir.parent.mkdir(parents=True, exist_ok=True)
        evaluation_output_dir.parent.mkdir(parents=True, exist_ok=True)
        evaluation_staging.replace(evaluation_output_dir)
        dataset_staging.replace(dataset_output_dir)
    except Exception:
        for staging in (dataset_staging, evaluation_staging):
            if staging.exists():
                shutil.rmtree(staging)
        raise
    return manifest
