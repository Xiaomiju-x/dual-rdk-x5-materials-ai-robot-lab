"""Authenticated RB-VoE PASSIVE_ONESHOT v2 data contracts."""

from __future__ import annotations

import base64
import math
import re
import secrets
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from rb_voe_passive.canonical import (
    canonical_json_bytes,
    canonical_sha256,
    is_sha256,
    seal_mapping,
)
from rb_voe_passive.errors import BundleInvalid, TrustPolicyError

TRUST_POLICY_SCHEMA_VERSION = "rb_voe_passive_trust_policy.v2"
BUNDLE_SCHEMA_VERSION_V2 = "passive_bundle.v2"
REPORT_SCHEMA_VERSION_V2 = "passive_report.v2"
TENSOR_SCHEMA_VERSION_V2 = "tensor_contract.v2"
PROFILE_SCHEMA_VERSION_V2 = "numeric_diff_profile.v2"
RECEIPT_SCHEMA_VERSION_V2 = "rdk_x5_execution_receipt.v2"
OBSERVATION_SCHEMA_VERSION_V2 = "passive_observation.v2"

_AUDIT_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{2,127}$")
_NONCE = re.compile(r"^[a-f0-9]{32,128}$")
_MAX_VECTOR_LENGTH = 8192
_MAX_NUMERIC_ABS = 1e100

_POLICY_FIELDS = {
    "schema_version",
    "policy_id",
    "issued_at",
    "expires_at",
    "inbox_root",
    "evidence_root",
    "protected_paths",
    "trusted_producer",
    "allowed_actual_backends",
    "allowed_devices",
    "trusted_threshold_profiles",
    "freshness",
    "policy_sha256",
}
_TRUSTED_PRODUCER_FIELDS = {
    "producer_id",
    "key_id",
    "public_key_ed25519_base64",
    "allowed_collector_sha256",
}
_PROTECTED_PATH_FIELDS = {"label", "path"}
_DEVICE_FIELDS = {"device_identity", "hostname", "board", "soc"}
_TRUSTED_PROFILE_FIELDS = {"profile_id", "profile_sha256"}
_FRESHNESS_FIELDS = {"max_bundle_age_seconds", "max_future_skew_seconds"}

_BUNDLE_FIELDS = {
    "schema_version",
    "audit_id",
    "created_at",
    "producer",
    "sealed",
    "bindings",
    "tensor_contract",
    "threshold_profile",
    "execution_receipt",
    "reference_cpu",
    "actual_bpu",
    "bundle_sha256",
    "signature",
}
_PRODUCER_FIELDS = {"producer_id", "key_id"}
_SIGNATURE_FIELDS = {"algorithm", "key_id", "signature_base64"}
_BINDING_FIELDS_V2 = {
    "input_sha256",
    "preprocess_sha256",
    "calibration_sha256",
    "model_semantics_sha256",
    "reference_model_sha256",
    "actual_model_sha256",
    "reference_runtime_sha256",
    "actual_runtime_sha256",
    "reference_output_sha256",
    "actual_output_sha256",
    "tensor_contract_sha256",
    "threshold_profile_sha256",
    "execution_receipt_sha256",
}
_TENSOR_FIELDS = {
    "schema_version",
    "logical_name",
    "dtype",
    "shape",
    "ordering",
    "dequantization",
    "tensor_contract_sha256",
}
_PROFILE_FIELDS_V2 = {
    "schema_version",
    "profile_id",
    "absolute_tolerance",
    "relative_tolerance",
    "max_violation_fraction",
    "max_mean_absolute_error",
    "max_root_mean_square_error",
    "profile_sha256",
}
_RECEIPT_FIELDS = {
    "schema_version",
    "receipt_id",
    "executed_at",
    "backend",
    "device",
    "collector_sha256",
    "nonce",
    "input_sha256",
    "preprocess_sha256",
    "calibration_sha256",
    "model_semantics_sha256",
    "model_sha256",
    "runtime_sha256",
    "tensor_contract_sha256",
    "output_sha256",
    "receipt_sha256",
}
_OBSERVATION_FIELDS_V2 = {
    "schema_version",
    "role",
    "backend",
    "device_identity",
    "input_sha256",
    "preprocess_sha256",
    "calibration_sha256",
    "model_semantics_sha256",
    "model_sha256",
    "runtime_sha256",
    "tensor_contract_sha256",
    "output_sha256",
    "execution_receipt_sha256",
    "values",
    "observation_sha256",
}


@dataclass(frozen=True, slots=True)
class TrustPolicyV2:
    payload: dict[str, Any]
    policy_id: str
    policy_sha256: str
    inbox_root: Path
    evidence_root: Path
    protected_paths: tuple[Path, ...]
    producer_id: str
    key_id: str
    public_key: Ed25519PublicKey
    allowed_collector_sha256: frozenset[str]
    allowed_devices: dict[str, dict[str, str]]
    trusted_threshold_profiles: dict[str, str]
    max_bundle_age_seconds: int
    max_future_skew_seconds: int


@dataclass(frozen=True, slots=True)
class ValidatedBundleV2:
    payload: dict[str, Any]
    audit_id: str
    bindings: dict[str, str]
    tensor_contract: dict[str, Any]
    threshold_profile: dict[str, Any]
    execution_receipt: dict[str, Any]
    reference_cpu: dict[str, Any]
    actual_bpu: dict[str, Any]


def _policy_fail(code: str, message: str) -> None:
    raise TrustPolicyError(code, message)


def _bundle_fail(code: str, message: str) -> None:
    raise BundleInvalid(code, message)


def _exact(
    value: object,
    expected: set[str],
    *,
    label: str,
    policy: bool = False,
) -> dict[str, Any]:
    fail = _policy_fail if policy else _bundle_fail
    if not isinstance(value, dict):
        fail("SCHEMA_TYPE_MISMATCH", f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        fail(
            "SCHEMA_FIELDS_MISMATCH",
            f"{label} fields mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}",
        )
    return value


def _identifier(value: object, *, field: str, audit: bool = False, policy: bool = False) -> str:
    pattern = _AUDIT_ID if audit else _IDENTIFIER
    fail = _policy_fail if policy else _bundle_fail
    if not isinstance(value, str) or not pattern.fullmatch(value):
        fail("SCHEMA_IDENTIFIER_INVALID", f"{field} is not a canonical identifier")
    return value


def _sha(value: object, *, field: str, policy: bool = False) -> str:
    fail = _policy_fail if policy else _bundle_fail
    if not is_sha256(value):
        fail("SCHEMA_SHA256_INVALID", f"{field} must be a lowercase SHA-256 digest")
    return value


def _timestamp(value: object, *, field: str, policy: bool = False) -> datetime:
    fail = _policy_fail if policy else _bundle_fail
    if not isinstance(value, str) or not value.endswith("Z"):
        fail("SCHEMA_TIMESTAMP_INVALID", f"{field} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        fail("SCHEMA_TIMESTAMP_INVALID", f"{field} must be an RFC3339 UTC timestamp")
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        fail("SCHEMA_TIMESTAMP_INVALID", f"{field} must use UTC")
    return parsed


def _number(
    value: object,
    *,
    field: str,
    minimum: float = 0.0,
    maximum: float = _MAX_NUMERIC_ABS,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _bundle_fail("SCHEMA_NUMBER_INVALID", f"{field} must be a finite number")
    number = float(value)
    if (
        not math.isfinite(number)
        or (isinstance(value, int) and number != value)
        or number < minimum
        or number > maximum
    ):
        _bundle_fail("SCHEMA_NUMBER_INVALID", f"{field} is outside its finite range")
    return number


def _positive_int(value: object, *, field: str, policy: bool = False) -> int:
    fail = _policy_fail if policy else _bundle_fail
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        fail("SCHEMA_INTEGER_INVALID", f"{field} must be a positive integer")
    return value


def _strict_base64(value: object, *, field: str, expected_length: int) -> bytes:
    if not isinstance(value, str):
        _policy_fail("POLICY_KEY_INVALID", f"{field} must be canonical base64")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise TrustPolicyError("POLICY_KEY_INVALID", f"{field} must be canonical base64") from exc
    if len(decoded) != expected_length or base64.b64encode(decoded).decode("ascii") != value:
        _policy_fail("POLICY_KEY_INVALID", f"{field} has an invalid Ed25519 key length")
    return decoded


def validate_trust_policy(
    payload_value: object,
    *,
    now: datetime | None = None,
) -> TrustPolicyV2:
    payload = _exact(payload_value, _POLICY_FIELDS, label="trust_policy", policy=True)
    if payload["schema_version"] != TRUST_POLICY_SCHEMA_VERSION:
        _policy_fail("POLICY_SCHEMA_UNSUPPORTED", "unsupported trust policy schema")
    policy_id = _identifier(payload["policy_id"], field="policy_id", policy=True)
    issued_at = _timestamp(payload["issued_at"], field="issued_at", policy=True)
    expires_at = _timestamp(payload["expires_at"], field="expires_at", policy=True)
    clock = now or datetime.now(timezone.utc)
    if issued_at > clock or expires_at <= clock or expires_at <= issued_at:
        _policy_fail("POLICY_TIME_INVALID", "trust policy is not currently valid")

    claimed = _sha(payload["policy_sha256"], field="policy_sha256", policy=True)
    unsigned = {key: value for key, value in payload.items() if key != "policy_sha256"}
    if not secrets.compare_digest(canonical_sha256(unsigned), claimed):
        _policy_fail("POLICY_SHA256_MISMATCH", "trust policy content does not match its digest")

    for field in ("inbox_root", "evidence_root"):
        if not isinstance(payload[field], str) or not payload[field]:
            _policy_fail("POLICY_PATH_INVALID", f"{field} must be a non-empty absolute path string")

    protected_value = payload["protected_paths"]
    if not isinstance(protected_value, list) or not protected_value:
        _policy_fail("POLICY_PROTECTED_PATHS_INVALID", "protected_paths must be a non-empty list")
    protected_paths: list[Path] = []
    protected_labels: set[str] = set()
    for index, item_value in enumerate(protected_value):
        item = _exact(
            item_value,
            _PROTECTED_PATH_FIELDS,
            label=f"protected_paths[{index}]",
            policy=True,
        )
        label = _identifier(item["label"], field=f"protected_paths[{index}].label", policy=True)
        if label in protected_labels:
            _policy_fail("POLICY_PROTECTED_PATHS_INVALID", "protected path labels must be unique")
        if not isinstance(item["path"], str) or not item["path"]:
            _policy_fail("POLICY_PATH_INVALID", "protected path must be a non-empty path string")
        protected_labels.add(label)
        protected_paths.append(Path(item["path"]))

    producer = _exact(
        payload["trusted_producer"],
        _TRUSTED_PRODUCER_FIELDS,
        label="trusted_producer",
        policy=True,
    )
    producer_id = _identifier(producer["producer_id"], field="producer_id", policy=True)
    key_id = _identifier(producer["key_id"], field="key_id", policy=True)
    public_key_bytes = _strict_base64(
        producer["public_key_ed25519_base64"],
        field="public_key_ed25519_base64",
        expected_length=32,
    )
    collectors_value = producer["allowed_collector_sha256"]
    if not isinstance(collectors_value, list) or not collectors_value:
        _policy_fail("POLICY_COLLECTOR_INVALID", "allowed_collector_sha256 must be non-empty")
    collectors = frozenset(
        _sha(item, field="allowed_collector_sha256", policy=True) for item in collectors_value
    )
    if len(collectors) != len(collectors_value):
        _policy_fail("POLICY_COLLECTOR_INVALID", "collector digests must be unique")

    if payload["allowed_actual_backends"] != ["RDK_X5_BPU"]:
        _policy_fail(
            "POLICY_BACKEND_INVALID",
            "v2 permits exactly one actual backend: RDK_X5_BPU",
        )

    devices_value = payload["allowed_devices"]
    if not isinstance(devices_value, list) or not devices_value:
        _policy_fail("POLICY_DEVICE_INVALID", "allowed_devices must be a non-empty list")
    devices: dict[str, dict[str, str]] = {}
    for index, device_value in enumerate(devices_value):
        device = _exact(
            device_value,
            _DEVICE_FIELDS,
            label=f"allowed_devices[{index}]",
            policy=True,
        )
        normalized = {
            field: _identifier(
                device[field],
                field=f"allowed_devices[{index}].{field}",
                policy=True,
            )
            for field in sorted(_DEVICE_FIELDS)
        }
        identity = normalized["device_identity"]
        if identity in devices:
            _policy_fail("POLICY_DEVICE_INVALID", "device identities must be unique")
        if normalized["board"] != "RDK_X5" or normalized["soc"] != "Bayes-e":
            _policy_fail("POLICY_DEVICE_INVALID", "allowed device must identify RDK_X5/Bayes-e")
        devices[identity] = normalized

    profiles_value = payload["trusted_threshold_profiles"]
    if not isinstance(profiles_value, list) or not profiles_value:
        _policy_fail("POLICY_PROFILE_INVALID", "trusted_threshold_profiles must be non-empty")
    profiles: dict[str, str] = {}
    for index, profile_value in enumerate(profiles_value):
        profile = _exact(
            profile_value,
            _TRUSTED_PROFILE_FIELDS,
            label=f"trusted_threshold_profiles[{index}]",
            policy=True,
        )
        profile_id = _identifier(profile["profile_id"], field="profile_id", policy=True)
        digest = _sha(profile["profile_sha256"], field="profile_sha256", policy=True)
        if profile_id in profiles:
            _policy_fail("POLICY_PROFILE_INVALID", "threshold profile IDs must be unique")
        profiles[profile_id] = digest

    freshness = _exact(
        payload["freshness"],
        _FRESHNESS_FIELDS,
        label="freshness",
        policy=True,
    )
    max_age = _positive_int(
        freshness["max_bundle_age_seconds"],
        field="max_bundle_age_seconds",
        policy=True,
    )
    max_skew = _positive_int(
        freshness["max_future_skew_seconds"],
        field="max_future_skew_seconds",
        policy=True,
    )
    if max_age > 31 * 24 * 3600 or max_skew > 3600:
        _policy_fail("POLICY_FRESHNESS_INVALID", "freshness bounds are too permissive")

    return TrustPolicyV2(
        payload=payload,
        policy_id=policy_id,
        policy_sha256=claimed,
        inbox_root=Path(payload["inbox_root"]),
        evidence_root=Path(payload["evidence_root"]),
        protected_paths=tuple(protected_paths),
        producer_id=producer_id,
        key_id=key_id,
        public_key=Ed25519PublicKey.from_public_bytes(public_key_bytes),
        allowed_collector_sha256=collectors,
        allowed_devices=devices,
        trusted_threshold_profiles=profiles,
        max_bundle_age_seconds=max_age,
        max_future_skew_seconds=max_skew,
    )


def _validate_self_digest(
    payload: dict[str, Any],
    digest_field: str,
    *,
    mismatch_code: str,
) -> str:
    claimed = _sha(payload[digest_field], field=digest_field)
    unsigned = {key: value for key, value in payload.items() if key != digest_field}
    if not secrets.compare_digest(canonical_sha256(unsigned), claimed):
        _bundle_fail(mismatch_code, f"{digest_field} does not match canonical content")
    return claimed


def _validate_tensor(value: object) -> dict[str, Any]:
    tensor = _exact(value, _TENSOR_FIELDS, label="tensor_contract")
    if tensor["schema_version"] != TENSOR_SCHEMA_VERSION_V2:
        _bundle_fail("TENSOR_SCHEMA_UNSUPPORTED", "unsupported tensor contract schema")
    _identifier(tensor["logical_name"], field="logical_name")
    if tensor["dtype"] not in {"float32", "float16", "int8"}:
        _bundle_fail("TENSOR_DTYPE_INVALID", "unsupported tensor dtype")
    shape = tensor["shape"]
    if not isinstance(shape, list) or not 1 <= len(shape) <= 4:
        _bundle_fail("TENSOR_SHAPE_INVALID", "tensor shape must have 1..4 dimensions")
    product = 1
    for index, dimension in enumerate(shape):
        product *= _positive_int(dimension, field=f"shape[{index}]")
    if product > _MAX_VECTOR_LENGTH:
        _bundle_fail("TENSOR_SHAPE_INVALID", "tensor exceeds the vector limit")
    for field in ("ordering", "dequantization"):
        _identifier(tensor[field], field=field)
    _validate_self_digest(
        tensor,
        "tensor_contract_sha256",
        mismatch_code="TENSOR_SHA256_MISMATCH",
    )
    return tensor


def _validate_profile_v2(value: object) -> dict[str, Any]:
    profile = _exact(value, _PROFILE_FIELDS_V2, label="threshold_profile")
    if profile["schema_version"] != PROFILE_SCHEMA_VERSION_V2:
        _bundle_fail("PROFILE_SCHEMA_UNSUPPORTED", "unsupported threshold profile schema")
    _identifier(profile["profile_id"], field="profile_id")
    _number(profile["absolute_tolerance"], field="absolute_tolerance")
    _number(profile["relative_tolerance"], field="relative_tolerance")
    _number(
        profile["max_violation_fraction"],
        field="max_violation_fraction",
        maximum=1.0,
    )
    for field in ("max_mean_absolute_error", "max_root_mean_square_error"):
        if profile[field] is not None:
            _number(profile[field], field=field)
    _validate_self_digest(
        profile,
        "profile_sha256",
        mismatch_code="PROFILE_SHA256_MISMATCH",
    )
    return profile


def _validate_values(
    values: object,
    *,
    label: str,
    expected_length: int,
    dtype: str,
) -> list[int | float]:
    if not isinstance(values, list) or len(values) != expected_length:
        _bundle_fail(
            "OBSERVATION_VECTOR_INVALID",
            f"{label}.values must match the sealed tensor shape",
        )
    for index, value in enumerate(values):
        field = f"{label}.values[{index}]"
        if dtype == "int8":
            if isinstance(value, bool) or not isinstance(value, int) or not -128 <= value <= 127:
                _bundle_fail(
                    "OBSERVATION_VALUE_DOMAIN_INVALID",
                    f"{field} must be a JSON integer in the int8 domain",
                )
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _bundle_fail(
                "OBSERVATION_VALUE_DOMAIN_INVALID",
                f"{field} must be a finite JSON number",
            )
        number = float(value)
        if not math.isfinite(number):
            _bundle_fail(
                "OBSERVATION_VALUE_DOMAIN_INVALID",
                f"{field} must be finite",
            )
        format_code = "e" if dtype == "float16" else "f"
        try:
            quantized = struct.unpack(f">{format_code}", struct.pack(f">{format_code}", number))[0]
        except (OverflowError, struct.error) as exc:
            raise BundleInvalid(
                "OBSERVATION_VALUE_DOMAIN_INVALID",
                f"{field} is outside the {dtype} finite domain",
            ) from exc
        if quantized != value:
            _bundle_fail(
                "OBSERVATION_VALUE_PRECISION_INVALID",
                f"{field} is not exactly representable as {dtype}",
            )
    return values


def tensor_values_sha256(tensor_contract_sha256: str, values: list[int | float]) -> str:
    return canonical_sha256(
        {
            "tensor_contract_sha256": tensor_contract_sha256,
            "values": values,
        }
    )


def _validate_observation_v2(
    value: object,
    *,
    expected_role: str,
    expected_backend: str,
    expected_length: int,
    expected_dtype: str,
) -> dict[str, Any]:
    label = expected_role.lower()
    observation = _exact(value, _OBSERVATION_FIELDS_V2, label=label)
    if observation["schema_version"] != OBSERVATION_SCHEMA_VERSION_V2:
        _bundle_fail("OBSERVATION_SCHEMA_UNSUPPORTED", "unsupported observation schema")
    if observation["role"] != expected_role or observation["backend"] != expected_backend:
        _bundle_fail("OBSERVATION_ROLE_INVALID", f"{label} role/backend is invalid")
    _identifier(observation["device_identity"], field=f"{label}.device_identity")
    for field in (
        "input_sha256",
        "preprocess_sha256",
        "calibration_sha256",
        "model_semantics_sha256",
        "model_sha256",
        "runtime_sha256",
        "tensor_contract_sha256",
        "output_sha256",
    ):
        _sha(observation[field], field=f"{label}.{field}")
    receipt_digest = observation["execution_receipt_sha256"]
    if expected_role == "REFERENCE_CPU":
        if receipt_digest is not None:
            _bundle_fail(
                "REFERENCE_RECEIPT_INVALID",
                "CPU reference cannot claim an RDK X5 execution receipt",
            )
    else:
        _sha(receipt_digest, field=f"{label}.execution_receipt_sha256")
    values = _validate_values(
        observation["values"],
        label=label,
        expected_length=expected_length,
        dtype=expected_dtype,
    )
    expected_output = tensor_values_sha256(observation["tensor_contract_sha256"], values)
    if not secrets.compare_digest(expected_output, observation["output_sha256"]):
        _bundle_fail("OUTPUT_SHA256_MISMATCH", f"{label} values do not match output_sha256")
    _validate_self_digest(
        observation,
        "observation_sha256",
        mismatch_code="OBSERVATION_SHA256_MISMATCH",
    )
    return observation


def _validate_receipt(value: object) -> dict[str, Any]:
    receipt = _exact(value, _RECEIPT_FIELDS, label="execution_receipt")
    if receipt["schema_version"] != RECEIPT_SCHEMA_VERSION_V2:
        _bundle_fail("RECEIPT_SCHEMA_UNSUPPORTED", "unsupported execution receipt schema")
    _identifier(receipt["receipt_id"], field="receipt_id")
    _timestamp(receipt["executed_at"], field="executed_at")
    if receipt["backend"] != "RDK_X5_BPU":
        _bundle_fail("ACTUAL_BACKEND_INVALID", "execution receipt backend must be RDK_X5_BPU")
    device = _exact(receipt["device"], _DEVICE_FIELDS, label="execution_receipt.device")
    for field in sorted(_DEVICE_FIELDS):
        _identifier(device[field], field=f"execution_receipt.device.{field}")
    _sha(receipt["collector_sha256"], field="collector_sha256")
    if not isinstance(receipt["nonce"], str) or not _NONCE.fullmatch(receipt["nonce"]):
        _bundle_fail("RECEIPT_NONCE_INVALID", "execution receipt nonce is invalid")
    for field in (
        "input_sha256",
        "preprocess_sha256",
        "calibration_sha256",
        "model_semantics_sha256",
        "model_sha256",
        "runtime_sha256",
        "tensor_contract_sha256",
        "output_sha256",
    ):
        _sha(receipt[field], field=f"execution_receipt.{field}")
    _validate_self_digest(
        receipt,
        "receipt_sha256",
        mismatch_code="RECEIPT_SHA256_MISMATCH",
    )
    return receipt


def _same_digest(
    left: dict[str, Any],
    left_field: str,
    right: dict[str, Any],
    right_field: str,
    *,
    code: str,
) -> None:
    if not secrets.compare_digest(left[left_field], right[right_field]):
        _bundle_fail(code, f"{left_field} and {right_field} are not identically bound")


def validate_bundle_v2(
    payload_value: object,
    policy: TrustPolicyV2,
    *,
    now: datetime | None = None,
) -> ValidatedBundleV2:
    payload = _exact(payload_value, _BUNDLE_FIELDS, label="bundle")
    if payload["schema_version"] != BUNDLE_SCHEMA_VERSION_V2:
        _bundle_fail("BUNDLE_SCHEMA_UNSUPPORTED", "unsupported passive bundle schema")
    audit_id = _identifier(payload["audit_id"], field="audit_id", audit=True)
    created_at = _timestamp(payload["created_at"], field="created_at")
    clock = now or datetime.now(timezone.utc)
    age = (clock - created_at).total_seconds()
    if age > policy.max_bundle_age_seconds:
        _bundle_fail("BUNDLE_STALE", "bundle exceeds the trust-policy freshness window")
    if age < -policy.max_future_skew_seconds:
        _bundle_fail("BUNDLE_FROM_FUTURE", "bundle exceeds the allowed future clock skew")
    if payload["sealed"] is not True:
        _bundle_fail("BUNDLE_NOT_SEALED", "bundle.sealed must be true")

    producer = _exact(payload["producer"], _PRODUCER_FIELDS, label="producer")
    if producer != {"producer_id": policy.producer_id, "key_id": policy.key_id}:
        _bundle_fail("PRODUCER_NOT_TRUSTED", "bundle producer/key is not trusted by policy")

    claimed_bundle_sha = _sha(payload["bundle_sha256"], field="bundle_sha256")
    unsigned_bundle = {
        key: value
        for key, value in payload.items()
        if key not in {"bundle_sha256", "signature"}
    }
    if not secrets.compare_digest(canonical_sha256(unsigned_bundle), claimed_bundle_sha):
        _bundle_fail("BUNDLE_SHA256_MISMATCH", "bundle content does not match its digest")

    signature = _exact(payload["signature"], _SIGNATURE_FIELDS, label="signature")
    if signature["algorithm"] != "Ed25519" or signature["key_id"] != policy.key_id:
        _bundle_fail("SIGNATURE_IDENTITY_INVALID", "bundle signature identity is invalid")
    if not isinstance(signature["signature_base64"], str):
        _bundle_fail("SIGNATURE_INVALID", "signature must be canonical base64")
    try:
        signature_bytes = base64.b64decode(signature["signature_base64"], validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise BundleInvalid("SIGNATURE_INVALID", "signature must be canonical base64") from exc
    if (
        len(signature_bytes) != 64
        or base64.b64encode(signature_bytes).decode("ascii") != signature["signature_base64"]
    ):
        _bundle_fail("SIGNATURE_INVALID", "Ed25519 signature encoding is invalid")
    signed_payload = {key: value for key, value in payload.items() if key != "signature"}
    try:
        policy.public_key.verify(signature_bytes, canonical_json_bytes(signed_payload))
    except InvalidSignature as exc:
        raise BundleInvalid(
            "SIGNATURE_VERIFICATION_FAILED",
            "bundle producer signature is not valid",
        ) from exc

    bindings_value = _exact(payload["bindings"], _BINDING_FIELDS_V2, label="bindings")
    bindings = {
        field: _sha(bindings_value[field], field=f"bindings.{field}")
        for field in sorted(_BINDING_FIELDS_V2)
    }
    tensor = _validate_tensor(payload["tensor_contract"])
    profile = _validate_profile_v2(payload["threshold_profile"])
    receipt = _validate_receipt(payload["execution_receipt"])
    expected_length = math.prod(tensor["shape"])
    reference = _validate_observation_v2(
        payload["reference_cpu"],
        expected_role="REFERENCE_CPU",
        expected_backend="CPU_REFERENCE",
        expected_length=expected_length,
        expected_dtype=tensor["dtype"],
    )
    actual = _validate_observation_v2(
        payload["actual_bpu"],
        expected_role="ACTUAL_BPU",
        expected_backend="RDK_X5_BPU",
        expected_length=expected_length,
        expected_dtype=tensor["dtype"],
    )

    trusted_profile_digest = policy.trusted_threshold_profiles.get(profile["profile_id"])
    if trusted_profile_digest is None or not secrets.compare_digest(
        trusted_profile_digest,
        profile["profile_sha256"],
    ):
        _bundle_fail(
            "THRESHOLD_PROFILE_NOT_PRECOMMITTED",
            "threshold profile is not precommitted by the trust policy",
        )
    device_identity = receipt["device"]["device_identity"]
    trusted_device = policy.allowed_devices.get(device_identity)
    if trusted_device is None or receipt["device"] != trusted_device:
        _bundle_fail("DEVICE_NOT_TRUSTED", "execution receipt device is not allowed by policy")
    if receipt["collector_sha256"] not in policy.allowed_collector_sha256:
        _bundle_fail("COLLECTOR_NOT_TRUSTED", "execution collector is not allowed by policy")
    executed_at = _timestamp(receipt["executed_at"], field="executed_at")
    if abs((created_at - executed_at).total_seconds()) > policy.max_future_skew_seconds:
        _bundle_fail(
            "RECEIPT_TIME_MISMATCH",
            "execution receipt and bundle timestamps exceed the allowed skew",
        )

    for observation in (reference, actual):
        for field in (
            "input_sha256",
            "preprocess_sha256",
            "calibration_sha256",
            "model_semantics_sha256",
            "tensor_contract_sha256",
        ):
            _same_digest(
                observation,
                field,
                bindings,
                field,
                code="SEMANTIC_BINDING_MISMATCH",
            )
    for field in (
        "input_sha256",
        "preprocess_sha256",
        "calibration_sha256",
        "model_semantics_sha256",
        "tensor_contract_sha256",
    ):
        _same_digest(receipt, field, bindings, field, code="RECEIPT_BINDING_MISMATCH")

    expected_pairs = (
        (tensor, "tensor_contract_sha256", bindings, "tensor_contract_sha256"),
        (profile, "profile_sha256", bindings, "threshold_profile_sha256"),
        (receipt, "receipt_sha256", bindings, "execution_receipt_sha256"),
        (reference, "model_sha256", bindings, "reference_model_sha256"),
        (actual, "model_sha256", bindings, "actual_model_sha256"),
        (reference, "runtime_sha256", bindings, "reference_runtime_sha256"),
        (actual, "runtime_sha256", bindings, "actual_runtime_sha256"),
        (reference, "output_sha256", bindings, "reference_output_sha256"),
        (actual, "output_sha256", bindings, "actual_output_sha256"),
        (receipt, "model_sha256", actual, "model_sha256"),
        (receipt, "runtime_sha256", actual, "runtime_sha256"),
        (receipt, "output_sha256", actual, "output_sha256"),
    )
    for left, left_field, right, right_field in expected_pairs:
        _same_digest(left, left_field, right, right_field, code="ARTIFACT_BINDING_MISMATCH")

    if actual["device_identity"] != device_identity:
        _bundle_fail("DEVICE_BINDING_MISMATCH", "actual observation device differs from receipt")
    if actual["execution_receipt_sha256"] != receipt["receipt_sha256"]:
        _bundle_fail(
            "RECEIPT_BINDING_MISMATCH",
            "actual observation is not bound to the execution receipt",
        )
    if reference["device_identity"] != "offline-reference":
        _bundle_fail("REFERENCE_DEVICE_INVALID", "CPU reference identity must be offline-reference")

    return ValidatedBundleV2(
        payload=payload,
        audit_id=audit_id,
        bindings=bindings,
        tensor_contract=tensor,
        threshold_profile=profile,
        execution_receipt=receipt,
        reference_cpu=reference,
        actual_bpu=actual,
    )


def seal_trust_policy_v2(payload: dict[str, Any]) -> dict[str, Any]:
    candidate = dict(payload)
    candidate.setdefault("schema_version", TRUST_POLICY_SCHEMA_VERSION)
    return seal_mapping(candidate, "policy_sha256")


def seal_tensor_contract_v2(payload: dict[str, Any]) -> dict[str, Any]:
    candidate = dict(payload)
    candidate.setdefault("schema_version", TENSOR_SCHEMA_VERSION_V2)
    return seal_mapping(candidate, "tensor_contract_sha256")


def seal_profile_v2(payload: dict[str, Any]) -> dict[str, Any]:
    candidate = dict(payload)
    candidate.setdefault("schema_version", PROFILE_SCHEMA_VERSION_V2)
    return seal_mapping(candidate, "profile_sha256")


def seal_receipt_v2(payload: dict[str, Any]) -> dict[str, Any]:
    candidate = dict(payload)
    candidate.setdefault("schema_version", RECEIPT_SCHEMA_VERSION_V2)
    return seal_mapping(candidate, "receipt_sha256")


def seal_observation_v2(payload: dict[str, Any]) -> dict[str, Any]:
    candidate = dict(payload)
    candidate.setdefault("schema_version", OBSERVATION_SCHEMA_VERSION_V2)
    return seal_mapping(candidate, "observation_sha256")


def sign_bundle_v2(
    payload: dict[str, Any],
    *,
    private_key: Ed25519PrivateKey,
    key_id: str,
) -> dict[str, Any]:
    candidate = dict(payload)
    candidate.setdefault("schema_version", BUNDLE_SCHEMA_VERSION_V2)
    candidate.setdefault("sealed", True)
    candidate.pop("bundle_sha256", None)
    candidate.pop("signature", None)
    candidate["bundle_sha256"] = canonical_sha256(candidate)
    signature = private_key.sign(canonical_json_bytes(candidate))
    candidate["signature"] = {
        "algorithm": "Ed25519",
        "key_id": key_id,
        "signature_base64": base64.b64encode(signature).decode("ascii"),
    }
    return candidate
