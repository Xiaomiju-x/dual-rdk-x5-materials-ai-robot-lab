"""Frozen generic profiles and C0-C7 diagnostic casebook."""

from __future__ import annotations

from typing import Any

from rb_voe.contracts.canonical import canonical_sha256


def build_file_drop_profiles() -> dict[str, Any]:
    common = {
        "load_mode": "HUMAN_IN_LOOP_LOAD",
        "trigger_mode": "HUMAN_IN_LOOP_TRIGGER",
        "watcher_mode": "READ_ONLY_FILE_DROP",
        "autonomous_instrument_control": False,
        "execution_authority": False,
        "required_sample_fields": ["sample_id", "batch_id", "aliquot_id", "parent_block_id"],
        "required_holder_fields": [
            "holder_id",
            "presence_evidence_sha256",
            "orientation_evidence_sha256",
            "load_operator_id",
            "loaded_at_ms",
        ],
        "required_acquisition_fields": [
            "instrument_id",
            "instrument_serial",
            "method_id",
            "method_sha256",
            "calibration_id",
            "calibration_sha256",
            "acquisition_id",
            "trigger_operator_id",
            "started_at_ms",
            "ended_at_ms",
        ],
        "required_raw_fields": [
            "spool_relative_path",
            "byte_count",
            "sha256",
            "exported_at_ms",
            "immutable",
            "overwrite_detected",
            "custody_root_sha256",
        ],
        "required_qualification_fields": [
            "analyzer_id",
            "analyzer_release_sha256",
            "closure_predicate",
            "truth_uncertainty",
            "blind_locked",
            "qualified_at_ms",
        ],
        "fail_codes": [
            "ACQUISITION_ID_REPLAY",
            "CALIBRATION_MISSING_OR_STALE",
            "CUSTODY_ROOT_MISMATCH",
            "HOLDER_EVIDENCE_MISSING",
            "METHOD_HASH_MISMATCH",
            "RAW_HASH_MISMATCH",
            "RAW_OVERWRITE_DETECTED",
            "RAW_STALE_OR_PREEXISTING",
            "SAMPLE_ID_MISMATCH",
            "UNQUALIFIED_ACTUAL",
        ],
    }
    profiles = []
    for profile_id, modality, extensions in (
        ("HITL_FILE_DROP_XRD", "XRD", [".raw"]),
        ("HITL_FILE_DROP_PL", "PL", [".csv"]),
    ):
        payload = dict(common)
        payload.update(
            {
                "schema_version": "xrd-rb-voe-assay-file-drop-profile-v1",
                "profile_id": profile_id,
                "modality": modality,
                "accepted_extensions": extensions,
                "status": "TEMPLATE_ONLY_NOT_QUALIFIED",
            }
        )
        profiles.append(payload)
    result = {
        "schema_version": "xrd-rb-voe-assay-file-drop-profile-set-v1",
        "profiles": profiles,
        "boundary": {
            "instrument_specific": False,
            "mapper_implemented": False,
            "profile_qualified": False,
            "acquire_pl_added_to_r1_registry": False,
            "r1_release_modified": False,
        },
    }
    result["content_sha256"] = canonical_sha256(result)
    return result


def build_golden_casebook() -> dict[str, Any]:
    rows = [
        (
            "C0",
            "CLEAN",
            ["IDENTITY_VALID", "RUNTIME_VALID", "RAW_VALID"],
            ["INDEPENDENT_FULL_EVIDENCE"],
            [],
            ["UNNECESSARY_EVIDENCE_ACTION", "UNJUSTIFIED_HOLD"],
        ),
        (
            "C1",
            "IDENTITY_OR_CUSTODY_GAP",
            ["LABEL_OR_HOLDER_OR_CUSTODY_MISSING"],
            ["INDEPENDENT_SCAN_LEDGER", "SEALED_HANDOFF_LOG"],
            ["E_VERIFY_IDENTITY", "E_QUARANTINE"],
            ["SCIENTIFIC_DIAGNOSIS_BEFORE_IDENTITY"],
        ),
        (
            "C2",
            "XRD_ALIAS",
            ["IDENTITY_VALID", "PREDEFINED_XRD_AMBIGUITY"],
            ["NEW_SAME_HOLDER_XRD", "BLINDED_XRD_REVIEW"],
            ["E_XRD_SAME_HOLDER", "E_BLINDED_ASSAY"],
            ["FILENAME_AS_PHASE_TRUTH"],
        ),
        (
            "C3",
            "PREPARATION_DEFECT",
            ["IDENTITY_VALID", "PREDEFINED_PREPARATION_ANOMALY"],
            ["NEW_PREPARATION_REVISION", "NEW_XRD"],
            ["E_REPREP_XRD"],
            ["REPEATED_PARSE_AS_NEW_PHYSICAL_EVIDENCE"],
        ),
        (
            "C4",
            "XRD_PL_CONFLICT",
            ["TWO_REAL_RAW_MODALITIES_CONFLICT"],
            ["INDEPENDENT_ANALYZER", "FULL_EVIDENCE"],
            ["E_PL_CROSSCHECK", "E_BLINDED_ASSAY"],
            ["ONE_MODALITY_SILENTLY_OVERRIDES_OTHER"],
        ),
        (
            "C5",
            "MIXED_RECOVERABLE",
            ["IDENTITY_VALID", "AT_LEAST_TWO_SCIENTIFIC_CAUSES_REMAIN"],
            ["MATCHED_ALIQUOT_OR_DEV_FULL_EVIDENCE"],
            ["CONDITIONAL_H2_BRANCH"],
            ["SECOND_ACTION_SELECTED_WITH_SEALED_TRUTH"],
        ),
        (
            "C6",
            "RUNTIME_OR_PROVENANCE_GAP",
            ["SAME_SAMPLE", "BACKEND_OR_HASH_OR_FRESHNESS_ANOMALY"],
            ["ACTUAL_RUNTIME_LOG", "FROZEN_RELEASE"],
            ["E_VERIFY_RUNTIME", "E_QUARANTINE"],
            ["FALLBACK_BACKEND_REPORTED_AS_TARGET_BACKEND"],
        ),
        (
            "C7",
            "IRREPARABLE",
            ["SAMPLE_OR_CAPABILITY_OR_EXTERNAL_TRUTH_UNAVAILABLE"],
            ["INDEPENDENT_ADJUDICATOR"],
            ["E_QUARANTINE"],
            ["FORCED_ACTION_TO_AVOID_HOLD"],
        ),
    ]
    cases = [
        {
            "case_id": case_id,
            "case_class": case_class,
            "visible_admission_conditions": visible,
            "sealed_truth_requirements": truth,
            "hypothesis_options_not_prescribed_answers": options,
            "forbidden_shortcuts": forbidden,
            "parent_block_required": True,
            "technical_repeats_count_as_independent_samples": False,
            "empty_class_rule": "MARK_EMPTY_DO_NOT_SYNTHESIZE_DENOMINATOR",
        }
        for case_id, case_class, visible, truth, options, forbidden in rows
    ]
    result = {
        "schema_version": "xrd-rb-voe-assay-golden-casebook-v1",
        "purpose": "DIAGNOSTIC_PILOT_MATRIX_NOT_STATISTICAL_STRATA",
        "cases": cases,
        "r6_mapping_status": "NOT_FROZEN",
        "r6_boundary": {
            "case_count": 8,
            "locked_strata_count": 5,
            "one_case_equals_one_stratum": False,
            "double_counting_allowed": False,
        },
    }
    result["content_sha256"] = canonical_sha256(result)
    return result


__all__ = ["build_file_drop_profiles", "build_golden_casebook"]
