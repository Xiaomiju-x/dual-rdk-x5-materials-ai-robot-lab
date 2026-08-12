"""Strict, stdlib-only validation for the frozen RB-VoE R0 contracts."""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

from rb_voe.contracts.models import (
    CapabilityManifest,
    ContractError,
    ContractRecord,
    Decision,
    DecisionReceipt,
    EvidenceIntent,
    ExecutionChallenge,
    ExperimentCase,
    JointPermit,
    Maturity,
    PhysicalEvidenceCapsule,
)

SCHEMA_DIRECTORY = Path(__file__).with_name("schemas")
SCHEMA_VERSION_TO_FILE: Mapping[str, str] = MappingProxyType(
    {
        "xrd-rb-voe-experiment-case-v1": "experiment_case.v1.schema.json",
        "xrd-rb-voe-evidence-intent-v1": "evidence_intent.v1.schema.json",
        "xrd-rb-voe-execution-challenge-v1": "execution_challenge.v1.schema.json",
        "xrd-rb-voe-joint-permit-v1": "joint_permit.v1.schema.json",
        "xrd-rb-voe-physical-evidence-v1": "physical_evidence_capsule.v1.schema.json",
        "xrd-rb-voe-decision-receipt-v1": "decision_receipt.v1.schema.json",
        "xrd-rb-voe-ai-capability-v1": "ai_capability_manifest.v1.schema.json",
        "xrd-rb-voe-embodied-capability-v1": "embodied_capability_manifest.v1.schema.json",
        "xrd-rb-voe-dual-arm-capability-v1": "dual_arm_capability_manifest.v1.schema.json",
        "xrd-rb-voe-assay-station-capability-v1": "assay_station_capability_manifest.v1.schema.json",
    }
)
CAPABILITY_SCHEMA_VERSIONS = frozenset(
    {
        "xrd-rb-voe-ai-capability-v1",
        "xrd-rb-voe-embodied-capability-v1",
        "xrd-rb-voe-dual-arm-capability-v1",
        "xrd-rb-voe-assay-station-capability-v1",
    }
)
EXPIRING_SCHEMA_VERSIONS = CAPABILITY_SCHEMA_VERSIONS | {
    "xrd-rb-voe-execution-challenge-v1",
    "xrd-rb-voe-joint-permit-v1",
}


class ContractValidationError(ContractError):
    """Raised when a payload is not an exact, usable R0 contract."""


ValidationError = ContractValidationError


@lru_cache(maxsize=1)
def _schema_index() -> Mapping[str, Mapping[str, Any]]:
    schemas: dict[str, Mapping[str, Any]] = {}
    for expected_version, filename in SCHEMA_VERSION_TO_FILE.items():
        path = SCHEMA_DIRECTORY / filename
        try:
            schema = json.loads(path.read_text(encoding="utf-8"))
            schema_version = schema["properties"]["schema_version"]["const"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError(f"invalid bundled contract schema: {path.name}") from exc
        if not isinstance(schema_version, str) or not schema_version:
            raise RuntimeError(f"schema version is invalid in {path.name}")
        if schema_version != expected_version:
            raise RuntimeError(
                f"schema version mismatch in {path.name}: expected {expected_version}, got {schema_version}"
            )
        schemas[schema_version] = schema
    return MappingProxyType(schemas)


def available_schema_versions() -> tuple[str, ...]:
    """Return the ten frozen R0 schema versions in lexical order."""
    return tuple(sorted(_schema_index()))


def load_schema(schema_version: str) -> dict[str, Any]:
    """Load an isolated copy of one bundled schema without exposing cache state."""
    try:
        schema = _schema_index()[schema_version]
    except KeyError as exc:
        raise ContractValidationError(f"unsupported schema_version: {schema_version!r}") from exc
    return copy.deepcopy(dict(schema))


def validate_contract_structure(payload: Mapping[str, Any] | ContractRecord) -> ContractRecord:
    """Validate schema and model invariants without authorizing an expiring record.

    This function is intended for fixture construction and offline inspection.
    Call :func:`validate_contract` at an execution boundary so freshness is also
    checked against an explicit caller-provided clock.
    """
    raw: Any = payload.to_dict() if isinstance(payload, ContractRecord) else payload
    if not isinstance(raw, Mapping):
        raise ContractValidationError("contract payload must be an object")
    if not all(isinstance(key, str) for key in raw):
        raise ContractValidationError("contract object keys must be strings")
    value = copy.deepcopy(dict(raw))
    schema_version = value.get("schema_version")
    if not isinstance(schema_version, str):
        raise ContractValidationError("$.schema_version must be a string")
    try:
        schema = _schema_index()[schema_version]
    except KeyError as exc:
        raise ContractValidationError(f"unsupported schema_version: {schema_version!r}") from exc
    _validate_node(value, schema, "$")
    return _construct_record(schema_version, value)


def validate_contract(
    payload: Mapping[str, Any] | ContractRecord,
    *,
    now_ms: int | None = None,
    require_ready: bool = False,
) -> ContractRecord:
    """Validate a contract and fail closed on missing or stale time context."""
    record = validate_contract_structure(payload)
    if record.schema_version not in EXPIRING_SCHEMA_VERSIONS:
        if now_ms is not None:
            _require_clock(now_ms)
        return record
    if now_ms is None:
        raise ContractValidationError(f"now_ms is required for expiring contract {record.schema_version}")
    _require_clock(now_ms)
    _require_fresh(record, now_ms)
    if require_ready and isinstance(record, CapabilityManifest) and record.maturity is Maturity.TARGET_ONLY:
        raise ContractValidationError("TARGET_ONLY capability manifest is NOT_READY")
    return record


def validate_capability_manifest(
    payload: Mapping[str, Any] | CapabilityManifest,
    *,
    now_ms: int,
    require_ready: bool = False,
) -> CapabilityManifest:
    """Validate and freshness-check one of the four capability manifests."""
    record = validate_contract(payload, now_ms=now_ms, require_ready=require_ready)
    if not isinstance(record, CapabilityManifest):
        raise ContractValidationError("payload is not a capability manifest")
    return record


def validate_contract_payload(
    payload: Mapping[str, Any] | ContractRecord,
    *,
    now_ms: int | None = None,
    require_ready: bool = False,
) -> dict[str, Any]:
    """Validate a contract and return a detached canonical wire mapping."""
    record = validate_contract(payload, now_ms=now_ms, require_ready=require_ready)
    return copy.deepcopy(record.to_dict())


def _require_clock(now_ms: object) -> None:
    if isinstance(now_ms, bool) or not isinstance(now_ms, int) or now_ms < 0:
        raise ContractValidationError("now_ms must be a non-negative integer")


def _require_fresh(record: ContractRecord, now_ms: int) -> None:
    if isinstance(record, CapabilityManifest):
        if now_ms < record.issued_at_ms:
            raise ContractValidationError("CAPABILITY_MANIFEST_NOT_YET_VALID")
        if now_ms >= record.expires_at_ms:
            raise ContractValidationError("CAPABILITY_MANIFEST_STALE")
        return
    if isinstance(record, ExecutionChallenge):
        if now_ms < record.issued_at_ms:
            raise ContractValidationError("CHALLENGE_NOT_YET_VALID")
        if now_ms >= record.expires_at_ms:
            raise ContractValidationError("CHALLENGE_STALE")
        return
    if isinstance(record, JointPermit):
        if now_ms < record.issued_at_ms:
            raise ContractValidationError("PERMIT_NOT_YET_VALID")
        if now_ms >= record.start_expires_at_ms:
            raise ContractValidationError("PERMIT_START_EXPIRED")


def _construct_record(schema_version: str, value: Mapping[str, Any]) -> ContractRecord:
    data = dict(value)
    try:
        if schema_version == "xrd-rb-voe-experiment-case-v1":
            data["allowed_options"] = tuple(data["allowed_options"])
            data["required_failure_atoms"] = tuple(data["required_failure_atoms"])
            return ExperimentCase(**data)
        if schema_version == "xrd-rb-voe-evidence-intent-v1":
            data["failure_core"] = tuple(data["failure_core"])
            data["candidate_options"] = tuple(data["candidate_options"])
            data["required_capabilities"] = tuple(data["required_capabilities"])
            data["decision"] = Decision(data["decision"])
            return EvidenceIntent(**data)
        if schema_version in CAPABILITY_SCHEMA_VERSIONS:
            data["maturity"] = Maturity(data["maturity"])
            data["capabilities"] = tuple(data["capabilities"])
            data["actual_backends"] = dict(data["actual_backends"])
            data["artifact_sha256"] = dict(data["artifact_sha256"])
            data["calibration_sha256"] = dict(data["calibration_sha256"])
            data["stations"] = tuple(data["stations"])
            return CapabilityManifest(**data)
        if schema_version == "xrd-rb-voe-execution-challenge-v1":
            data["reserved_routes"] = tuple(data["reserved_routes"])
            data["reserved_stations"] = tuple(data["reserved_stations"])
            data["reserved_zones"] = tuple(data["reserved_zones"])
            return ExecutionChallenge(**data)
        if schema_version == "xrd-rb-voe-joint-permit-v1":
            data["roles"] = dict(data["roles"])
            data["zones"] = tuple(data["zones"])
            data["required_capability_hashes"] = tuple(data["required_capability_hashes"])
            data["required_local_gates"] = tuple(data["required_local_gates"])
            return JointPermit(**data)
        if schema_version == "xrd-rb-voe-physical-evidence-v1":
            data["artifact_sha256"] = dict(data["artifact_sha256"])
            data["calibration_sha256"] = dict(data["calibration_sha256"])
            return PhysicalEvidenceCapsule(**data)
        if schema_version == "xrd-rb-voe-decision-receipt-v1":
            data["decision"] = Decision(data["decision"])
            data["terminal_evidence_sha256"] = tuple(data["terminal_evidence_sha256"])
            return DecisionReceipt(**data)
    except (ContractError, TypeError, ValueError) as exc:
        raise ContractValidationError(str(exc)) from exc
    raise ContractValidationError(f"unsupported schema_version: {schema_version!r}")


def _validate_node(value: Any, schema: Mapping[str, Any], path: str) -> None:
    if "const" in schema and not _json_equal(value, schema["const"]):
        raise ContractValidationError(f"{path} must equal {schema['const']!r}")
    if "enum" in schema and not any(_json_equal(value, item) for item in schema["enum"]):
        raise ContractValidationError(f"{path} is not in the frozen registry")

    expected_type = schema.get("type")
    if expected_type is not None and not _has_json_type(value, expected_type):
        raise ContractValidationError(f"{path} must have JSON type {_type_label(expected_type)}")

    for index, child_schema in enumerate(schema.get("allOf", ())):
        _validate_node(value, child_schema, f"{path}.allOf[{index}]")
    if "anyOf" in schema:
        matches = sum(_matches(value, child) for child in schema["anyOf"])
        if matches == 0:
            raise ContractValidationError(f"{path} does not match any allowed schema")
    if "oneOf" in schema:
        matches = sum(_matches(value, child) for child in schema["oneOf"])
        if matches != 1:
            raise ContractValidationError(f"{path} must match exactly one allowed schema")
    if "if" in schema:
        branch = "then" if _matches(value, schema["if"]) else "else"
        if branch in schema:
            _validate_node(value, schema[branch], path)

    if isinstance(value, Mapping):
        _validate_object(value, schema, path)
    elif isinstance(value, list):
        _validate_array(value, schema, path)
    elif isinstance(value, str):
        _validate_string(value, schema, path)
    elif _is_json_number(value):
        _validate_number(value, schema, path)


def _validate_object(value: Mapping[str, Any], schema: Mapping[str, Any], path: str) -> None:
    required = schema.get("required", ())
    missing = [key for key in required if key not in value]
    if missing:
        raise ContractValidationError(f"{path} is missing required fields: {', '.join(missing)}")
    properties = schema.get("properties", {})
    additional = schema.get("additionalProperties", True)
    for key, item in value.items():
        if not isinstance(key, str):
            raise ContractValidationError(f"{path} object keys must be strings")
        child_path = f"{path}.{key}"
        if "propertyNames" in schema:
            _validate_node(key, schema["propertyNames"], f"{child_path}<name>")
        if key in properties:
            _validate_node(item, properties[key], child_path)
        elif additional is False:
            raise ContractValidationError(f"{path} contains unknown field {key!r}")
        elif isinstance(additional, Mapping):
            _validate_node(item, additional, child_path)
    if len(value) < schema.get("minProperties", 0):
        raise ContractValidationError(f"{path} has too few properties")


def _validate_array(value: list[Any], schema: Mapping[str, Any], path: str) -> None:
    if len(value) < schema.get("minItems", 0):
        raise ContractValidationError(f"{path} has too few items")
    if "maxItems" in schema and len(value) > schema["maxItems"]:
        raise ContractValidationError(f"{path} has too many items")
    if "items" in schema:
        for index, item in enumerate(value):
            _validate_node(item, schema["items"], f"{path}[{index}]")
    if schema.get("uniqueItems"):
        try:
            encoded = [
                json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False) for item in value
            ]
        except (TypeError, ValueError) as exc:
            raise ContractValidationError(f"{path} contains a non-JSON item") from exc
        if len(encoded) != len(set(encoded)):
            raise ContractValidationError(f"{path} items must be unique")


def _validate_string(value: str, schema: Mapping[str, Any], path: str) -> None:
    if len(value) < schema.get("minLength", 0):
        raise ContractValidationError(f"{path} is too short")
    if "maxLength" in schema and len(value) > schema["maxLength"]:
        raise ContractValidationError(f"{path} is too long")
    if "pattern" in schema and re.search(schema["pattern"], value) is None:
        raise ContractValidationError(f"{path} has invalid format")


def _validate_number(value: int | float, schema: Mapping[str, Any], path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractValidationError(f"{path} must be finite")
    if "minimum" in schema and value < schema["minimum"]:
        raise ContractValidationError(f"{path} is below its minimum")
    if "maximum" in schema and value > schema["maximum"]:
        raise ContractValidationError(f"{path} is above its maximum")
    if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
        raise ContractValidationError(f"{path} is below its exclusive minimum")


def _matches(value: Any, schema: Mapping[str, Any]) -> bool:
    try:
        _validate_node(value, schema, "$")
    except ContractValidationError:
        return False
    return True


def _has_json_type(value: Any, expected: str | list[str]) -> bool:
    names = [expected] if isinstance(expected, str) else expected
    return any(
        {
            "null": value is None,
            "boolean": isinstance(value, bool),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": _is_json_number(value),
            "string": isinstance(value, str),
            "array": isinstance(value, list),
            "object": isinstance(value, Mapping),
        }.get(name, False)
        for name in names
    )


def _is_json_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _json_equal(left: Any, right: Any) -> bool:
    if _is_json_number(left) and _is_json_number(right):
        return left == right
    return type(left) is type(right) and left == right


def _type_label(expected: str | list[str]) -> str:
    if isinstance(expected, str):
        return expected
    return " or ".join(expected)


__all__ = [
    "CAPABILITY_SCHEMA_VERSIONS",
    "ContractValidationError",
    "EXPIRING_SCHEMA_VERSIONS",
    "SCHEMA_DIRECTORY",
    "SCHEMA_VERSION_TO_FILE",
    "ValidationError",
    "available_schema_versions",
    "load_schema",
    "validate_capability_manifest",
    "validate_contract",
    "validate_contract_payload",
    "validate_contract_structure",
]
