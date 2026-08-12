"""Conservative evidence-grade gate for the assay bootstrap."""

from __future__ import annotations

from enum import Enum
from typing import Any

from rb_voe.assay_bootstrap.models import ClaimEvidenceSummary
from rb_voe.contracts.canonical import canonical_sha256

CLAIM_REPORT_SCHEMA = "xrd-rb-voe-assay-bootstrap-claim-report-v1"


class EvidenceGrade(str, Enum):
    CATALOG_ONLY = "CATALOG_ONLY"
    E_OFFLINE_REPLAY = "E_OFFLINE_REPLAY"
    D_NARROW_MACRO = "D_NARROW_MACRO"
    C_ENGINEERING_E2E = "C_ENGINEERING_E2E"
    B_PROSPECTIVE_PILOT = "B_PROSPECTIVE_PILOT"
    A_CONFIRMATORY = "A_CONFIRMATORY"


_PUBLIC_CLAIMS = {
    EvidenceGrade.CATALOG_ONLY: "SOURCE_CATALOG_PREPARED",
    EvidenceGrade.E_OFFLINE_REPLAY: "OFFLINE_REPLAY_VALIDATED",
    EvidenceGrade.D_NARROW_MACRO: "NARROW_MACRO_CASES_ONLY",
    EvidenceGrade.C_ENGINEERING_E2E: "REPRODUCIBLE_ENGINEERING_CLOSED_LOOP",
    EvidenceGrade.B_PROSPECTIVE_PILOT: "PROSPECTIVE_PILOT_WITH_N_K_U95",
    EvidenceGrade.A_CONFIRMATORY: "LOCKED_TARGET_POPULATION_CLAIM_ONLY",
}


def evaluate_claim_gate(summary: ClaimEvidenceSummary) -> dict[str, Any]:
    if not isinstance(summary, ClaimEvidenceSummary):
        raise TypeError("summary must be a ClaimEvidenceSummary")

    # This bootstrap has no qualified physical-artifact verifier. Higher grades
    # must remain unreachable instead of trusting caller-supplied counts or flags.
    checks = {
        "offline_replay": summary.historical_file_count > 0,
        "narrow_macro": False,
        "engineering_e2e": False,
        "prospective_pilot": False,
        "confirmatory": False,
    }
    if checks["confirmatory"]:
        grade = EvidenceGrade.A_CONFIRMATORY
    elif checks["prospective_pilot"]:
        grade = EvidenceGrade.B_PROSPECTIVE_PILOT
    elif checks["engineering_e2e"]:
        grade = EvidenceGrade.C_ENGINEERING_E2E
    elif checks["narrow_macro"]:
        grade = EvidenceGrade.D_NARROW_MACRO
    elif checks["offline_replay"]:
        grade = EvidenceGrade.E_OFFLINE_REPLAY
    else:
        grade = EvidenceGrade.CATALOG_ONLY

    blocked_claims: list[str] = []
    if not checks["engineering_e2e"]:
        blocked_claims.extend(
            [
                "PHYSICAL_CLOSED_LOOP_COMPLETED",
                "FULLY_AUTONOMOUS_ASSAY",
                "REAL_MATERIAL_BENEFIT_PROVEN",
            ]
        )
    if not checks["prospective_pilot"]:
        blocked_claims.append("PROSPECTIVE_RISK_BOUND")
    if not checks["confirmatory"]:
        blocked_claims.extend(["FIVE_PERCENT_RISK_CERTIFICATE", "TWENTY_PERCENT_RMT_FCC_GAIN"])

    payload: dict[str, Any] = {
        "schema_version": CLAIM_REPORT_SCHEMA,
        "evidence_grade": grade.value,
        "max_public_claim": _PUBLIC_CLAIMS[grade],
        "maturity_unchanged": "R1_INTEGRATION_PREPARED_R2_LIVE_NOT_RUN",
        "summary": summary.to_dict(),
        "checks": checks,
        "check_authority": {
            "offline_replay": "RECOMPUTED_WORKSPACE_INVENTORY",
            "narrow_macro": "NO_QUALIFIED_PHYSICAL_ARTIFACT_VERIFIER",
            "engineering_e2e": "NO_QUALIFIED_PHYSICAL_ARTIFACT_VERIFIER",
            "prospective_pilot": "NO_QUALIFIED_PHYSICAL_ARTIFACT_VERIFIER",
            "confirmatory": "NO_QUALIFIED_LOCKED_DATASET_ARTIFACT_VERIFIER",
        },
        "blocked_claims": sorted(set(blocked_claims)),
        "boundary": {
            "d_work_line_is_not_evidence_grade_d": True,
            "historical_or_public_data_count_as_fresh_truth": False,
            "rag_chunks_count_as_independent_physical_evidence": False,
            "bootstrap_can_issue_execution_authority": False,
            "self_reported_counts_can_upgrade_claim": False,
            "higher_grades_require_future_verified_artifact_api": True,
            "physical_denominator_increment": 0,
        },
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


__all__ = ["CLAIM_REPORT_SCHEMA", "EvidenceGrade", "evaluate_claim_gate"]
