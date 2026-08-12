"""Historical PASSIVE_ONESHOT v1 implementation retained in security HOLD."""

from __future__ import annotations

import math
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rb_voe_passive.canonical import (
    canonical_sha256,
    is_sha256,
    raw_sha256,
    seal_mapping,
    strict_json_object,
    text_sha256,
)
from rb_voe_passive.contracts import (
    REPORT_SCHEMA_VERSION,
    ValidatedBundle,
    validate_bundle,
    validate_report_digest,
)
from rb_voe_passive.errors import BundleInvalid, EvidenceError
from rb_voe_passive.io import (
    create_evidence_directory,
    read_sealed_bundle,
    validate_evidence_root,
    write_report_exclusive,
)

AUDIT_PASS = "AUDIT_PASS"
AUDIT_HOLD = "AUDIT_HOLD"
AUDIT_INVALID = "AUDIT_INVALID"
V1_SECURITY_STATUS = AUDIT_HOLD
V1_DEPLOYMENT_STATE = "CANDIDATE_ONLY_NOT_DEPLOYABLE"

EXIT_CODES = {
    AUDIT_PASS: 0,
    AUDIT_HOLD: 2,
    AUDIT_INVALID: 3,
}

_AUTHORITY_BOUNDARY = {
    "execution_authority": False,
    "network_touched": False,
    "subprocess_used": False,
    "inference_invoked": False,
    "device_accessed": False,
    "hardware_touched": False,
    "business_mutated": False,
    "production_files_opened": False,
}


@dataclass(frozen=True, slots=True)
class OneShotResult:
    status: str
    report_id: str
    report_dir: Path
    report_path: Path
    report: dict[str, Any]
    exit_code: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _invalid_report_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%f")
    return f"invalid_{timestamp}_{secrets.token_hex(4)}"


def _new_invalid_directory(root: Path) -> tuple[str, Path]:
    for _ in range(8):
        report_id = _invalid_report_id()
        try:
            return report_id, create_evidence_directory(root, report_id)
        except EvidenceError as exc:
            if exc.code != "AUDIT_ID_ALREADY_EXISTS":
                raise
    raise EvidenceError(
        "EVIDENCE_ID_EXHAUSTED",
        "could not reserve a unique invalid-audit evidence directory",
    )


def _seal_matches(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    claimed = payload.get("bundle_sha256")
    if not is_sha256(claimed):
        return False
    unsigned = {key: value for key, value in payload.items() if key != "bundle_sha256"}
    try:
        return secrets.compare_digest(canonical_sha256(unsigned), claimed)
    except (TypeError, ValueError):
        return False


def _safe_claimed_digest(payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    claimed = payload.get("bundle_sha256")
    return claimed if is_sha256(claimed) else None


def _source_record(
    bundle_path: Path,
    *,
    raw: bytes | None,
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "bundle_path_sha256": text_sha256(str(bundle_path)),
        "bundle_file_sha256": raw_sha256(raw) if raw is not None else None,
        "claimed_bundle_sha256": _safe_claimed_digest(payload),
        "bundle_seal_verified": _seal_matches(payload),
    }


def _empty_differential(reason: str) -> dict[str, Any]:
    return {
        "performed": False,
        "differential_claim_allowed": False,
        "equivalence_claim_allowed": False,
        "reason": reason,
        "profile_id": None,
        "profile_sha256": None,
        "reference_cpu_observation_sha256": None,
        "actual_bpu_observation_sha256": None,
        "vector_length": None,
        "metrics": None,
    }


def _binding_assessment(
    *,
    bundle_valid: bool,
    differential_content_present: bool,
) -> dict[str, bool]:
    return {
        "artifact_hashes_bound_by_bundle": bundle_valid,
        "artifact_contents_opened": False,
        "threshold_profile_content_verified": bundle_valid and differential_content_present,
        "actual_output_content_verified": bundle_valid and differential_content_present,
    }


def _base_report(
    *,
    report_id: str,
    bundle_audit_id: str | None,
    status: str,
    started_at: str,
    source: dict[str, Any],
    bindings: dict[str, str] | None,
    binding_assessment: dict[str, bool],
    differential: dict[str, Any],
    findings: list[dict[str, str]],
) -> dict[str, Any]:
    unsigned = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_id": report_id,
        "bundle_audit_id": bundle_audit_id,
        "status": status,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "source": source,
        "bindings": bindings,
        "binding_assessment": binding_assessment,
        "differential": differential,
        "authority": dict(_AUTHORITY_BOUNDARY),
        "findings": findings,
    }
    return seal_mapping(unsigned, "report_sha256")


def _evaluate_differential(bundle: ValidatedBundle) -> tuple[str, dict[str, Any], list[dict[str, str]]]:
    differential = bundle.differential
    if differential is None:
        return (
            AUDIT_HOLD,
            _empty_differential("DUAL_OUTPUTS_NOT_SUPPLIED"),
            [
                {
                    "code": "DIFFERENTIAL_NOT_SUPPLIED",
                    "severity": "HOLD",
                    "message": "bundle is sealed, but no CPU/BPU dual outputs were supplied",
                }
            ],
        )

    profile = differential["profile"]
    reference = differential["reference_cpu"]
    actual = differential["actual_bpu"]
    reference_values = reference["values"]
    actual_values = actual["values"]
    common = {
        "profile_id": profile["profile_id"],
        "profile_sha256": profile["profile_sha256"],
        "reference_cpu_observation_sha256": reference["observation_sha256"],
        "actual_bpu_observation_sha256": actual["observation_sha256"],
    }
    if len(reference_values) != len(actual_values):
        return (
            AUDIT_HOLD,
            {
                "performed": False,
                "differential_claim_allowed": False,
                "equivalence_claim_allowed": False,
                "reason": "VECTOR_LENGTH_MISMATCH",
                **common,
                "vector_length": None,
                "metrics": None,
            },
            [
                {
                    "code": "VECTOR_LENGTH_MISMATCH",
                    "severity": "HOLD",
                    "message": "CPU and BPU output vectors have different lengths",
                }
            ],
        )

    absolute_tolerance = float(profile["absolute_tolerance"])
    relative_tolerance = float(profile["relative_tolerance"])
    errors: list[float] = []
    violation_count = 0
    for reference_value, actual_value in zip(reference_values, actual_values, strict=True):
        reference_number = float(reference_value)
        actual_number = float(actual_value)
        error = abs(actual_number - reference_number)
        limit = absolute_tolerance + relative_tolerance * abs(reference_number)
        errors.append(error)
        if error > limit:
            violation_count += 1

    count = len(errors)
    mean_absolute_error = math.fsum(errors) / count
    root_mean_square_error = math.sqrt(math.fsum(error * error for error in errors) / count)
    max_absolute_error = max(errors)
    violation_fraction = violation_count / count
    allowed_violation_fraction = float(profile["max_violation_fraction"])
    elementwise_pass = violation_fraction <= allowed_violation_fraction
    mae_limit = profile["max_mean_absolute_error"]
    rmse_limit = profile["max_root_mean_square_error"]
    mae_pass = mae_limit is None or mean_absolute_error <= float(mae_limit)
    rmse_pass = rmse_limit is None or root_mean_square_error <= float(rmse_limit)
    passed = elementwise_pass and mae_pass and rmse_pass
    metrics = {
        "max_absolute_error": max_absolute_error,
        "mean_absolute_error": mean_absolute_error,
        "root_mean_square_error": root_mean_square_error,
        "violation_count": violation_count,
        "violation_fraction": violation_fraction,
        "allowed_violation_fraction": allowed_violation_fraction,
        "elementwise_pass": elementwise_pass,
        "mean_absolute_error_pass": mae_pass,
        "root_mean_square_error_pass": rmse_pass,
    }
    result = {
        "performed": True,
        "differential_claim_allowed": True,
        "equivalence_claim_allowed": passed,
        "reason": "WITHIN_PROFILE" if passed else "OUTSIDE_PROFILE",
        **common,
        "vector_length": count,
        "metrics": metrics,
    }
    if passed:
        return (
            AUDIT_PASS,
            result,
            [
                {
                    "code": "NUMERIC_DIFFERENTIAL_PASS",
                    "severity": "INFO",
                    "message": "CPU/BPU outputs are within the sealed threshold profile",
                }
            ],
        )
    return (
        AUDIT_HOLD,
        result,
        [
            {
                "code": "NUMERIC_DIFFERENTIAL_OUTSIDE_PROFILE",
                "severity": "HOLD",
                "message": "CPU/BPU outputs exceed the sealed threshold profile",
            }
        ],
    )


def _write_result(
    *,
    report_dir: Path,
    report: dict[str, Any],
) -> OneShotResult:
    validate_report_digest(report)
    report_path = write_report_exclusive(report_dir, report)
    status = report["status"]
    return OneShotResult(
        status=status,
        report_id=report["report_id"],
        report_dir=report_dir,
        report_path=report_path,
        report=report,
        exit_code=EXIT_CODES[status],
    )


def _write_invalid_result(
    *,
    evidence_root: Path,
    bundle_path: Path,
    started_at: str,
    error: BundleInvalid | EvidenceError,
    raw: bytes | None,
    payload: dict[str, Any] | None,
    bundle_audit_id: str | None = None,
) -> OneShotResult:
    report_id, report_dir = _new_invalid_directory(evidence_root)
    report = _base_report(
        report_id=report_id,
        bundle_audit_id=bundle_audit_id,
        status=AUDIT_INVALID,
        started_at=started_at,
        source=_source_record(bundle_path, raw=raw, payload=payload),
        bindings=None,
        binding_assessment=_binding_assessment(
            bundle_valid=False,
            differential_content_present=False,
        ),
        differential=_empty_differential("AUDIT_INVALID"),
        findings=[
            {
                "code": error.code,
                "severity": "INVALID",
                "message": error.message,
            }
        ],
    )
    return _write_result(report_dir=report_dir, report=report)


def run_passive_oneshot(
    bundle_path: str | Path,
    evidence_root: str | Path,
) -> OneShotResult:
    """Run the legacy candidate-only v1 compatibility audit.

    This API exists only for frozen tests and historical evidence readability. It
    has no deployment, service, inference, device, network, or execution
    authority. Authenticated claims require ``run_passive_oneshot_v2``.
    """

    return _run_passive_oneshot_v1_candidate_only(bundle_path, evidence_root)


def _run_passive_oneshot_v1_candidate_only(
    bundle_path: str | Path,
    evidence_root: str | Path,
) -> OneShotResult:
    """Retain bounded, non-link v1 behavior without granting trust or deployment status."""

    # Historical implementation remains below so the independent v1 audit stays readable.
    started_at = _utc_now()
    source_path = Path(bundle_path)
    evidence_path = Path(evidence_root)
    validate_evidence_root(evidence_path)

    raw: bytes | None = None
    payload: dict[str, Any] | None = None
    try:
        raw = read_sealed_bundle(source_path)
        payload = strict_json_object(raw)
        bundle = validate_bundle(payload)
    except BundleInvalid as exc:
        return _write_invalid_result(
            evidence_root=evidence_path,
            bundle_path=source_path,
            started_at=started_at,
            error=exc,
            raw=raw,
            payload=payload,
        )

    try:
        report_dir = create_evidence_directory(evidence_path, bundle.audit_id)
    except EvidenceError as exc:
        if exc.code != "AUDIT_ID_ALREADY_EXISTS":
            raise
        return _write_invalid_result(
            evidence_root=evidence_path,
            bundle_path=source_path,
            started_at=started_at,
            error=exc,
            raw=raw,
            payload=payload,
            bundle_audit_id=bundle.audit_id,
        )

    status, differential, findings = _evaluate_differential(bundle)
    findings.append(
        {
            "code": "LEGACY_CANDIDATE_ONLY",
            "severity": "INFO",
            "message": (
                "v1 result is a candidate-only compatibility artifact; its unkeyed "
                "self-seal is not producer authentication and it is not deployable"
            ),
        }
    )
    report = _base_report(
        report_id=bundle.audit_id,
        bundle_audit_id=bundle.audit_id,
        status=status,
        started_at=started_at,
        source=_source_record(source_path, raw=raw, payload=payload),
        bindings=bundle.bindings,
        binding_assessment=_binding_assessment(
            bundle_valid=True,
            differential_content_present=bundle.differential is not None,
        ),
        differential=differential,
        findings=findings,
    )
    return _write_result(report_dir=report_dir, report=report)
