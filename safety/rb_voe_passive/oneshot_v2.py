"""Authenticated, path-confined, one-shot RB-VoE passive audit v2."""

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
from rb_voe_passive.contracts_v2 import (
    REPORT_SCHEMA_VERSION_V2,
    TrustPolicyV2,
    ValidatedBundleV2,
    validate_bundle_v2,
)
from rb_voe_passive.errors import BundleInvalid, EvidenceError, PathPolicyError
from rb_voe_passive.io_v2 import ConfinedRootsV2, confined_roots_v2, load_trust_policy

AUDIT_PASS = "AUDIT_PASS"
AUDIT_HOLD = "AUDIT_HOLD"
AUDIT_INVALID = "AUDIT_INVALID"

EXIT_CODES_V2 = {
    AUDIT_PASS: 0,
    AUDIT_HOLD: 2,
    AUDIT_INVALID: 3,
}

_REPORT_FIELDS = {
    "schema_version",
    "report_id",
    "bundle_audit_id",
    "status",
    "started_at",
    "completed_at",
    "source",
    "root_policy_proof",
    "assurance",
    "actual_backend",
    "bindings",
    "differential",
    "authority",
    "findings",
    "report_sha256",
}
_AUTHORITY_FIELDS = {
    "execution_authority",
    "bundle_opened",
    "evidence_written",
    "network_touched",
    "subprocess_used",
    "inference_invoked",
    "device_accessed",
    "hardware_touched",
    "business_mutated",
    "production_files_opened",
    "production_exclusion_proven_by_root_policy",
}
_ASSURANCE_FIELDS = {
    "producer_authenticated",
    "bundle_contents_opened",
    "artifact_contents_opened",
    "actual_backend_attested",
    "threshold_profile_precommitted",
    "numeric_differential_performed",
    "self_hash_treated_as_authentication",
}


@dataclass(frozen=True, slots=True)
class OneShotResultV2:
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
    return f"invalid_v2_{timestamp}_{secrets.token_hex(4)}"


def _new_invalid_directory(roots: ConfinedRootsV2) -> tuple[str, Path, int | None]:
    for _ in range(8):
        report_id = _invalid_report_id()
        try:
            directory, descriptor = roots.create_report_directory(report_id)
            return report_id, directory, descriptor
        except EvidenceError as exc:
            if exc.code != "AUDIT_ID_ALREADY_EXISTS":
                raise
    raise EvidenceError(
        "EVIDENCE_ID_EXHAUSTED",
        "could not reserve a unique invalid-audit evidence directory",
    )


def _bundle_integrity_verified(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    claimed = payload.get("bundle_sha256")
    if not is_sha256(claimed):
        return False
    unsigned = {
        key: value
        for key, value in payload.items()
        if key not in {"bundle_sha256", "signature"}
    }
    try:
        return secrets.compare_digest(canonical_sha256(unsigned), claimed)
    except (TypeError, ValueError):
        return False


def _source_record(
    *,
    bundle_path: Path,
    policy: TrustPolicyV2,
    raw: bytes | None,
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    claimed = payload.get("bundle_sha256") if isinstance(payload, dict) else None
    return {
        "bundle_path_sha256": text_sha256(str(bundle_path)),
        "bundle_file_sha256": raw_sha256(raw) if raw is not None else None,
        "claimed_bundle_sha256": claimed if is_sha256(claimed) else None,
        "bundle_content_integrity_verified": _bundle_integrity_verified(payload),
        "trust_policy_id": policy.policy_id,
        "trust_policy_sha256": policy.policy_sha256,
    }


def _assurance(
    *,
    producer_authenticated: bool,
    bundle_opened: bool,
    actual_backend_attested: bool,
    profile_precommitted: bool,
    differential_performed: bool,
) -> dict[str, bool]:
    return {
        "producer_authenticated": producer_authenticated,
        "bundle_contents_opened": bundle_opened,
        "artifact_contents_opened": False,
        "actual_backend_attested": actual_backend_attested,
        "threshold_profile_precommitted": profile_precommitted,
        "numeric_differential_performed": differential_performed,
        "self_hash_treated_as_authentication": False,
    }


def _empty_differential(reason: str) -> dict[str, Any]:
    return {
        "performed": False,
        "equivalence_claim_allowed": False,
        "reason": reason,
        "profile_id": None,
        "profile_sha256": None,
        "reference_cpu_observation_sha256": None,
        "actual_bpu_observation_sha256": None,
        "vector_length": None,
        "metrics": None,
    }


def _evaluate_differential(
    bundle: ValidatedBundleV2,
) -> tuple[str, dict[str, Any], list[dict[str, str]]]:
    profile = bundle.threshold_profile
    reference = bundle.reference_cpu
    actual = bundle.actual_bpu
    reference_values = reference["values"]
    actual_values = actual["values"]
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
    violation_fraction = violation_count / count
    elementwise_pass = violation_fraction <= float(profile["max_violation_fraction"])
    mae_limit = profile["max_mean_absolute_error"]
    rmse_limit = profile["max_root_mean_square_error"]
    mae_pass = mae_limit is None or mean_absolute_error <= float(mae_limit)
    rmse_pass = rmse_limit is None or root_mean_square_error <= float(rmse_limit)
    passed = elementwise_pass and mae_pass and rmse_pass
    result = {
        "performed": True,
        "equivalence_claim_allowed": passed,
        "reason": "WITHIN_PRECOMMITTED_PROFILE" if passed else "OUTSIDE_PRECOMMITTED_PROFILE",
        "profile_id": profile["profile_id"],
        "profile_sha256": profile["profile_sha256"],
        "reference_cpu_observation_sha256": reference["observation_sha256"],
        "actual_bpu_observation_sha256": actual["observation_sha256"],
        "vector_length": count,
        "metrics": {
            "max_absolute_error": max(errors),
            "mean_absolute_error": mean_absolute_error,
            "root_mean_square_error": root_mean_square_error,
            "violation_count": violation_count,
            "violation_fraction": violation_fraction,
            "allowed_violation_fraction": float(profile["max_violation_fraction"]),
            "elementwise_pass": elementwise_pass,
            "mean_absolute_error_pass": mae_pass,
            "root_mean_square_error_pass": rmse_pass,
        },
    }
    if passed:
        return (
            AUDIT_PASS,
            result,
            [
                {
                    "code": "AUTHENTICATED_NUMERIC_DIFFERENTIAL_PASS",
                    "severity": "INFO",
                    "message": (
                        "authenticated RDK X5 BPU actual output is within the "
                        "precommitted numeric profile"
                    ),
                }
            ],
        )
    return (
        AUDIT_HOLD,
        result,
        [
            {
                "code": "AUTHENTICATED_NUMERIC_DIFFERENTIAL_OUTSIDE_PROFILE",
                "severity": "HOLD",
                "message": (
                    "authenticated RDK X5 BPU actual output exceeds the "
                    "precommitted numeric profile"
                ),
            }
        ],
    )


def _authority_for_persisted_report(roots: ConfinedRootsV2) -> dict[str, bool]:
    authority = roots.authority
    authority["evidence_written"] = True
    return authority


def _base_report(
    *,
    report_id: str,
    bundle_audit_id: str | None,
    status: str,
    started_at: str,
    source: dict[str, Any],
    root_policy_proof: dict[str, Any],
    assurance: dict[str, bool],
    actual_backend: dict[str, Any] | None,
    bindings: dict[str, str] | None,
    differential: dict[str, Any],
    authority: dict[str, bool],
    findings: list[dict[str, str]],
) -> dict[str, Any]:
    unsigned = {
        "schema_version": REPORT_SCHEMA_VERSION_V2,
        "report_id": report_id,
        "bundle_audit_id": bundle_audit_id,
        "status": status,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "source": source,
        "root_policy_proof": root_policy_proof,
        "assurance": assurance,
        "actual_backend": actual_backend,
        "bindings": bindings,
        "differential": differential,
        "authority": authority,
        "findings": findings,
    }
    return seal_mapping(unsigned, "report_sha256")


def validate_report_v2_digest(report: dict[str, Any]) -> None:
    if set(report) != _REPORT_FIELDS:
        raise ValueError("generated v2 report fields are invalid")
    if report["schema_version"] != REPORT_SCHEMA_VERSION_V2:
        raise ValueError("generated v2 report schema version is invalid")
    if report["status"] not in EXIT_CODES_V2:
        raise ValueError("generated v2 report status is invalid")
    authority = report.get("authority")
    if not isinstance(authority, dict) or set(authority) != _AUTHORITY_FIELDS:
        raise ValueError("generated v2 authority ledger is invalid")
    expected_false = {
        "execution_authority",
        "network_touched",
        "subprocess_used",
        "inference_invoked",
        "device_accessed",
        "hardware_touched",
        "business_mutated",
        "production_files_opened",
    }
    if any(authority[field] is not False for field in expected_false):
        raise ValueError("generated v2 authority ledger exceeds passive scope")
    if authority["evidence_written"] is not True:
        raise ValueError("persisted v2 report must truthfully record evidence_written=true")
    if authority["production_exclusion_proven_by_root_policy"] is not True:
        raise ValueError("v2 report must bind its production exclusion proof")
    if not isinstance(authority["bundle_opened"], bool):
        raise ValueError("v2 bundle_opened must be a measured boolean")
    assurance = report.get("assurance")
    if not isinstance(assurance, dict) or set(assurance) != _ASSURANCE_FIELDS:
        raise ValueError("generated v2 assurance ledger is invalid")
    if assurance["artifact_contents_opened"] is not False:
        raise ValueError("v2 does not open model/runtime artifact contents")
    if assurance["self_hash_treated_as_authentication"] is not False:
        raise ValueError("v2 cannot treat an unkeyed self-hash as authentication")
    claimed = report.get("report_sha256")
    if not is_sha256(claimed):
        raise ValueError("generated v2 report digest is invalid")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if not secrets.compare_digest(canonical_sha256(unsigned), claimed):
        raise ValueError("generated v2 report digest does not match its content")


def _persist_report(
    *,
    roots: ConfinedRootsV2,
    report_dir: Path,
    directory_fd: int | None,
    report: dict[str, Any],
) -> OneShotResultV2:
    validate_report_v2_digest(report)
    report_path = roots.write_report(report_dir, directory_fd, report)
    status = report["status"]
    return OneShotResultV2(
        status=status,
        report_id=report["report_id"],
        report_dir=report_dir,
        report_path=report_path,
        report=report,
        exit_code=EXIT_CODES_V2[status],
    )


def _write_invalid(
    *,
    roots: ConfinedRootsV2,
    policy: TrustPolicyV2,
    bundle_path: Path,
    started_at: str,
    error: BundleInvalid | PathPolicyError | EvidenceError,
    raw: bytes | None,
    payload: dict[str, Any] | None,
    bundle_audit_id: str | None = None,
) -> OneShotResultV2:
    report_id, report_dir, directory_fd = _new_invalid_directory(roots)
    report = _base_report(
        report_id=report_id,
        bundle_audit_id=bundle_audit_id,
        status=AUDIT_INVALID,
        started_at=started_at,
        source=_source_record(
            bundle_path=bundle_path,
            policy=policy,
            raw=raw,
            payload=payload,
        ),
        root_policy_proof=roots.root_policy_proof(),
        assurance=_assurance(
            producer_authenticated=False,
            bundle_opened=roots.authority["bundle_opened"],
            actual_backend_attested=False,
            profile_precommitted=False,
            differential_performed=False,
        ),
        actual_backend=None,
        bindings=None,
        differential=_empty_differential("AUDIT_INVALID"),
        authority=_authority_for_persisted_report(roots),
        findings=[
            {
                "code": error.code,
                "severity": "INVALID",
                "message": error.message,
            }
        ],
    )
    return _persist_report(
        roots=roots,
        report_dir=report_dir,
        directory_fd=directory_fd,
        report=report,
    )


def run_passive_oneshot_v2(
    *,
    trust_policy_path: str | Path,
    bundle_path: str | Path,
    evidence_root: str | Path,
) -> OneShotResultV2:
    """Authenticate and compare one signed offline bundle, then exit."""

    started_at = _utc_now()
    policy_path = Path(trust_policy_path)
    source_path = Path(bundle_path)
    policy = load_trust_policy(policy_path)
    with confined_roots_v2(
        policy,
        policy_path=policy_path,
        evidence_root_argument=evidence_root,
    ) as roots:
        raw: bytes | None = None
        payload: dict[str, Any] | None = None
        try:
            raw = roots.read_bundle(source_path)
            payload = strict_json_object(raw)
            bundle = validate_bundle_v2(payload, policy)
        except (BundleInvalid, PathPolicyError) as exc:
            return _write_invalid(
                roots=roots,
                policy=policy,
                bundle_path=source_path,
                started_at=started_at,
                error=exc,
                raw=raw,
                payload=payload,
            )

        try:
            report_dir, directory_fd = roots.create_report_directory(bundle.audit_id)
        except EvidenceError as exc:
            if exc.code != "AUDIT_ID_ALREADY_EXISTS":
                raise
            return _write_invalid(
                roots=roots,
                policy=policy,
                bundle_path=source_path,
                started_at=started_at,
                error=exc,
                raw=raw,
                payload=payload,
                bundle_audit_id=bundle.audit_id,
            )

        status, differential, findings = _evaluate_differential(bundle)
        receipt = bundle.execution_receipt
        report = _base_report(
            report_id=bundle.audit_id,
            bundle_audit_id=bundle.audit_id,
            status=status,
            started_at=started_at,
            source=_source_record(
                bundle_path=source_path,
                policy=policy,
                raw=raw,
                payload=payload,
            ),
            root_policy_proof=roots.root_policy_proof(),
            assurance=_assurance(
                producer_authenticated=True,
                bundle_opened=True,
                actual_backend_attested=True,
                profile_precommitted=True,
                differential_performed=True,
            ),
            actual_backend={
                "backend": receipt["backend"],
                "device": receipt["device"],
                "collector_sha256": receipt["collector_sha256"],
                "execution_receipt_sha256": receipt["receipt_sha256"],
                "attestation_kind": "TRUSTED_PRODUCER_SIGNED_EXECUTION_RECEIPT",
                "hardware_root_of_trust_claimed": False,
            },
            bindings=bundle.bindings,
            differential=differential,
            authority=_authority_for_persisted_report(roots),
            findings=findings,
        )
        return _persist_report(
            roots=roots,
            report_dir=report_dir,
            directory_fd=directory_fd,
            report=report,
        )
