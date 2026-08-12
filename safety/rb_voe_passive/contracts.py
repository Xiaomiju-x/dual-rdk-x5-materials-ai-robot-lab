"""Strict passive bundle/report contracts with no production dependencies."""

from __future__ import annotations

import math
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from rb_voe_passive.canonical import canonical_sha256, is_sha256, seal_mapping
from rb_voe_passive.errors import BundleInvalid

BUNDLE_SCHEMA_VERSION = "passive_bundle.v1"
REPORT_SCHEMA_VERSION = "passive_report.v1"
DIFF_PROFILE_SCHEMA_VERSION = "numeric_diff_profile.v1"

_AUDIT_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")
_PROFILE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,95}$")
_PRODUCER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{2,127}$")
_MAX_VECTOR_LENGTH = 8192
_MAX_NUMERIC_ABS = 1e100

_BINDING_FIELDS = {
    "input_sha256",
    "model_sha256",
    "preprocess_sha256",
    "calibration_sha256",
    "runtime_sha256",
    "output_sha256",
    "threshold_profile_sha256",
}
_ROOT_FIELDS = {
    "schema_version",
    "audit_id",
    "created_at",
    "producer",
    "sealed",
    "bindings",
    "differential",
    "bundle_sha256",
}
_PROFILE_FIELDS = {
    "schema_version",
    "profile_id",
    "absolute_tolerance",
    "relative_tolerance",
    "max_violation_fraction",
    "max_mean_absolute_error",
    "max_root_mean_square_error",
    "profile_sha256",
}
_OBSERVATION_FIELDS = {
    "role",
    "model_sha256",
    "runtime_sha256",
    "values",
    "observation_sha256",
}


@dataclass(frozen=True, slots=True)
class ValidatedBundle:
    payload: dict[str, Any]
    audit_id: str
    bindings: dict[str, str]
    differential: dict[str, Any] | None


def _fail(code: str, message: str) -> None:
    raise BundleInvalid(code, message)


def _require_exact_fields(value: object, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("SCHEMA_TYPE_MISMATCH", f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        _fail(
            "SCHEMA_FIELDS_MISMATCH",
            f"{label} fields mismatch; missing={missing}, extra={extra}",
        )
    return value


def _require_identifier(value: object, pattern: re.Pattern[str], *, field: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        _fail("SCHEMA_IDENTIFIER_INVALID", f"{field} is not a canonical identifier")
    return value


def _require_sha(value: object, *, field: str) -> str:
    if not is_sha256(value):
        _fail("SCHEMA_SHA256_INVALID", f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_utc_timestamp(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        _fail("SCHEMA_TIMESTAMP_INVALID", f"{field} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise BundleInvalid(
            "SCHEMA_TIMESTAMP_INVALID",
            f"{field} must be an RFC3339 UTC timestamp",
        ) from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        _fail("SCHEMA_TIMESTAMP_INVALID", f"{field} must use UTC")
    return value


def _require_number(
    value: object,
    *,
    field: str,
    minimum: float = 0.0,
    maximum: float = _MAX_NUMERIC_ABS,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("SCHEMA_NUMBER_INVALID", f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < minimum or number > maximum:
        _fail("SCHEMA_NUMBER_INVALID", f"{field} is outside its finite range")
    return number


def _validate_profile(profile_value: object) -> dict[str, Any]:
    profile = _require_exact_fields(profile_value, _PROFILE_FIELDS, label="differential.profile")
    if profile["schema_version"] != DIFF_PROFILE_SCHEMA_VERSION:
        _fail("PROFILE_SCHEMA_UNSUPPORTED", "unsupported numeric differential profile schema")
    _require_identifier(profile["profile_id"], _PROFILE_ID, field="profile_id")
    _require_number(profile["absolute_tolerance"], field="absolute_tolerance")
    _require_number(profile["relative_tolerance"], field="relative_tolerance")
    _require_number(
        profile["max_violation_fraction"],
        field="max_violation_fraction",
        maximum=1.0,
    )
    for field in ("max_mean_absolute_error", "max_root_mean_square_error"):
        if profile[field] is not None:
            _require_number(profile[field], field=field)
    claimed = _require_sha(profile["profile_sha256"], field="profile_sha256")
    unsigned = {key: value for key, value in profile.items() if key != "profile_sha256"}
    if not secrets.compare_digest(canonical_sha256(unsigned), claimed):
        _fail("PROFILE_SHA256_MISMATCH", "threshold profile content does not match its digest")
    return profile


def _validate_observation(observation_value: object, *, expected_role: str) -> dict[str, Any]:
    observation = _require_exact_fields(
        observation_value,
        _OBSERVATION_FIELDS,
        label=expected_role.lower(),
    )
    if observation["role"] != expected_role:
        _fail("OBSERVATION_ROLE_INVALID", f"observation role must be {expected_role}")
    _require_sha(observation["model_sha256"], field=f"{expected_role}.model_sha256")
    _require_sha(observation["runtime_sha256"], field=f"{expected_role}.runtime_sha256")
    values = observation["values"]
    if not isinstance(values, list) or not 1 <= len(values) <= _MAX_VECTOR_LENGTH:
        _fail(
            "OBSERVATION_VECTOR_INVALID",
            f"{expected_role}.values must contain 1..{_MAX_VECTOR_LENGTH} numbers",
        )
    for index, value in enumerate(values):
        _require_number(
            value,
            field=f"{expected_role}.values[{index}]",
            minimum=-_MAX_NUMERIC_ABS,
            maximum=_MAX_NUMERIC_ABS,
        )
    claimed = _require_sha(
        observation["observation_sha256"],
        field=f"{expected_role}.observation_sha256",
    )
    unsigned = {
        key: value
        for key, value in observation.items()
        if key != "observation_sha256"
    }
    if not secrets.compare_digest(canonical_sha256(unsigned), claimed):
        _fail(
            "OBSERVATION_SHA256_MISMATCH",
            f"{expected_role} values and identity do not match their digest",
        )
    return observation


def validate_bundle(payload_value: object) -> ValidatedBundle:
    payload = _require_exact_fields(payload_value, _ROOT_FIELDS, label="bundle")
    if payload["schema_version"] != BUNDLE_SCHEMA_VERSION:
        _fail("BUNDLE_SCHEMA_UNSUPPORTED", "unsupported passive bundle schema")
    audit_id = _require_identifier(payload["audit_id"], _AUDIT_ID, field="audit_id")
    _require_utc_timestamp(payload["created_at"], field="created_at")
    _require_identifier(payload["producer"], _PRODUCER_ID, field="producer")
    if payload["sealed"] is not True:
        _fail("BUNDLE_NOT_SEALED", "bundle.sealed must be true")

    bindings_value = _require_exact_fields(payload["bindings"], _BINDING_FIELDS, label="bindings")
    bindings = {
        field: _require_sha(bindings_value[field], field=f"bindings.{field}")
        for field in sorted(_BINDING_FIELDS)
    }
    claimed_bundle_sha = _require_sha(payload["bundle_sha256"], field="bundle_sha256")
    unsigned_bundle = {
        key: value for key, value in payload.items() if key != "bundle_sha256"
    }
    if not secrets.compare_digest(canonical_sha256(unsigned_bundle), claimed_bundle_sha):
        _fail("BUNDLE_SHA256_MISMATCH", "bundle content does not match its seal")

    differential_value = payload["differential"]
    if differential_value is None:
        return ValidatedBundle(
            payload=payload,
            audit_id=audit_id,
            bindings=bindings,
            differential=None,
        )
    differential = _require_exact_fields(
        differential_value,
        {"profile", "reference_cpu", "actual_bpu"},
        label="differential",
    )
    profile = _validate_profile(differential["profile"])
    reference_cpu = _validate_observation(
        differential["reference_cpu"],
        expected_role="REFERENCE_CPU",
    )
    actual_bpu = _validate_observation(
        differential["actual_bpu"],
        expected_role="ACTUAL_BPU",
    )
    if not secrets.compare_digest(
        profile["profile_sha256"],
        bindings["threshold_profile_sha256"],
    ):
        _fail(
            "THRESHOLD_PROFILE_BINDING_MISMATCH",
            "threshold profile digest is not the digest bound by the bundle",
        )
    if not secrets.compare_digest(actual_bpu["model_sha256"], bindings["model_sha256"]):
        _fail("MODEL_BINDING_MISMATCH", "ACTUAL_BPU model is not the bound model")
    if not secrets.compare_digest(actual_bpu["runtime_sha256"], bindings["runtime_sha256"]):
        _fail("RUNTIME_BINDING_MISMATCH", "ACTUAL_BPU runtime is not the bound runtime")
    if not secrets.compare_digest(
        actual_bpu["observation_sha256"],
        bindings["output_sha256"],
    ):
        _fail("OUTPUT_BINDING_MISMATCH", "ACTUAL_BPU observation is not the bound output")
    return ValidatedBundle(
        payload=payload,
        audit_id=audit_id,
        bindings=bindings,
        differential={
            "profile": profile,
            "reference_cpu": reference_cpu,
            "actual_bpu": actual_bpu,
        },
    )


def seal_profile(profile: dict[str, Any]) -> dict[str, Any]:
    candidate = dict(profile)
    candidate.setdefault("schema_version", DIFF_PROFILE_SCHEMA_VERSION)
    return seal_mapping(candidate, "profile_sha256")


def seal_observation(observation: dict[str, Any]) -> dict[str, Any]:
    return seal_mapping(observation, "observation_sha256")


def seal_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    candidate = dict(bundle)
    candidate.setdefault("schema_version", BUNDLE_SCHEMA_VERSION)
    candidate.setdefault("sealed", True)
    return seal_mapping(candidate, "bundle_sha256")


def validate_report_digest(report: dict[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("generated report schema version is invalid")
    if report.get("status") not in {"AUDIT_PASS", "AUDIT_HOLD", "AUDIT_INVALID"}:
        raise ValueError("generated report status is invalid")
    authority = report.get("authority")
    if not isinstance(authority, dict) or any(authority.values()):
        raise ValueError("generated report must have zero execution authority and side effects")
    claimed = report.get("report_sha256")
    if not is_sha256(claimed):
        raise ValueError("generated report digest is invalid")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if not secrets.compare_digest(canonical_sha256(unsigned), claimed):
        raise ValueError("generated report digest does not match its content")
