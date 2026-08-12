from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import onnx

from common import (
    ARTIFACT_ROOT,
    CONTRACT_ROOT,
    EVIDENCE_ROOT,
    ROOT,
    RND_ROOT,
    TRIAL_ROOT,
    canonical_json_bytes,
    file_record,
    relative,
    sha256_file,
    utc_now,
    verify_frozen50,
    write_json,
)


EXPECTED_FITS = {
    "RND-PROC-01": 56,
    "RND-SEM-01": 32,
    "RND-XRD-01": 32,
    "RND-MAT-01": 16,
    "RND-MAT-02": 16,
    "RND-MAT-03": 8,
}

FAMILY_RECEIPTS = {
    "RND-PROC-01": EVIDENCE_ROOT / "RND-PROC-01/recovery_family_receipt.v1.json",
    "RND-SEM-01": EVIDENCE_ROOT / "RND-SEM-01/family_receipt.v1.json",
    "RND-XRD-01": EVIDENCE_ROOT / "RND-XRD-01/family_receipt.v1.json",
    "RND-MAT-01": EVIDENCE_ROOT / "RND-MAT-01/family_receipt.v1.json",
    "RND-MAT-02": EVIDENCE_ROOT / "RND-MAT-02/family_receipt.v1.json",
    "RND-MAT-03": EVIDENCE_ROOT / "RND-MAT-03/family_receipt.v1.json",
}

CONDITIONAL_SKIPS = [
    {"family_id": "RND-MAT-04", "name": "DielectricPair", "planned_fits": 8},
    {"family_id": "RND-MAT-05", "name": "Thermoelectric4", "planned_fits": 8},
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_seal(payload: dict[str, Any]) -> bool:
    claimed = payload.get("receipt_sha256")
    if not isinstance(claimed, str):
        return False
    unsigned = dict(payload)
    unsigned.pop("receipt_sha256", None)
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() == claimed


def verify_record(record: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / record["path"]
    inside_rnd = False
    try:
        path.resolve().relative_to(RND_ROOT.resolve())
        inside_rnd = True
    except ValueError:
        pass
    exists = path.is_file()
    actual_bytes = path.stat().st_size if exists else None
    actual_sha = sha256_file(path) if exists else None
    return {
        "path": record["path"],
        "exists": exists,
        "inside_isolated_rnd_root": inside_rnd,
        "bytes_match": exists and actual_bytes == record.get("bytes"),
        "sha256_match": exists and actual_sha == record.get("sha256"),
        "actual_bytes": actual_bytes,
        "actual_sha256": actual_sha,
        "pass": bool(
            exists
            and inside_rnd
            and actual_bytes == record.get("bytes")
            and actual_sha == record.get("sha256")
        ),
    }


def verify_fit_receipts() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    family_checks: list[dict[str, Any]] = []
    receipt_checks: list[dict[str, Any]] = []
    for family_id, expected_count in EXPECTED_FITS.items():
        paths = sorted((TRIAL_ROOT / family_id).glob("*.json"))
        family_checks.append(
            {
                "family_id": family_id,
                "expected": expected_count,
                "actual": len(paths),
                "pass": len(paths) == expected_count,
            }
        )
        for path in paths:
            payload = load_json(path)
            checkpoint = payload.get("checkpoint")
            checkpoint_check = verify_record(checkpoint) if checkpoint else None
            restrictions = {
                "test_observed_false": payload.get("test_observed") is False,
                "network_used_false": payload.get("network_used") is False,
                "x5_contacted_false": payload.get("x5_contacted") is False,
                "official_registry_member_false": payload.get("official_registry_member") is False,
            }
            receipt_checks.append(
                {
                    "path": relative(path),
                    "family_id": payload.get("family_id"),
                    "fit_id": payload.get("fit_id"),
                    "seal_pass": verify_seal(payload),
                    "family_match": payload.get("family_id") == family_id,
                    "checkpoint": checkpoint_check,
                    "restrictions": restrictions,
                    "pass": bool(
                        verify_seal(payload)
                        and payload.get("family_id") == family_id
                        and (checkpoint_check is None or checkpoint_check["pass"])
                        and all(restrictions.values())
                    ),
                }
            )
    return family_checks, receipt_checks


def collect_families() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    families: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    artifact_checks: list[dict[str, Any]] = []
    for family_id, receipt_path in FAMILY_RECEIPTS.items():
        receipt = load_json(receipt_path)
        restrictions = {
            "official_registry_member_false": receipt.get("official_registry_member") is False,
            "release_created_false": receipt.get("release_created") is False,
            "deployed_false": receipt.get("deployed") is False,
            "x5_verified_false": receipt.get("x5_verified") is False,
            "network_used_false": receipt.get("network_used") is False,
            "x5_contacted_false": receipt.get("x5_contacted") is False,
            "passive_minimal_manual": receipt.get("execution_policy") == "PASSIVE_MINIMAL_MANUAL",
        }
        receipt_fit_count = int(receipt.get("total_physical_fits", receipt.get("fit_count", 0)))
        family = {
            "family_id": family_id,
            "model_name": receipt.get("model_name", "HERO-VM17" if family_id == "RND-PROC-01" else None),
            "receipt": file_record(receipt_path),
            "receipt_seal_pass": verify_seal(receipt),
            "fit_count": receipt_fit_count,
            "fit_count_match": receipt_fit_count == EXPECTED_FITS[family_id],
            "candidate_class": receipt.get("candidate_class"),
            "innovation_gate": receipt.get("innovation_gate"),
            "state": "PC_RND_ONLY",
            "restrictions": restrictions,
            "exports": [],
        }
        export_ids = sorted(receipt.get("exports", {}).keys())
        primary_variant = receipt.get("selected_quality_variant")
        if not primary_variant:
            primary_variant = "quality" if "quality" in export_ids else "recovery_quality" if "recovery_quality" in export_ids else (export_ids[0] if export_ids else None)
        family["primary_variant"] = primary_variant
        supplemental_pass = True
        if family_id == "RND-XRD-01":
            supplemental_path = EVIDENCE_ROOT / "RND-XRD-01/aux_head_diagnostic.v1.json"
            supplemental = load_json(supplemental_path)
            supplemental_pass = bool(
                verify_seal(supplemental)
                and supplemental.get("fit_count_added") == 0
                and supplemental.get("test_retuning_performed") is False
                and supplemental.get("candidate_class_changed") is False
            )
            family["supplemental_evidence"] = {
                **file_record(supplemental_path),
                "seal_and_policy_pass": supplemental_pass,
            }
        if family_id == "RND-SEM-01":
            extension_path = EVIDENCE_ROOT / "RND-SEM-01/epoch_extension.v1.json"
            extension = load_json(extension_path)
            extension_pass = bool(
                extension.get("fit_identifiers_added") == 0
                and extension.get("completed_fit_cap_changed") is False
                and extension.get("test_observed_before_extension") is False
                and extension.get("new_fit_batch_size") == 32
            )
            supplemental_pass = supplemental_pass and extension_pass
            family["training_extension_evidence"] = {
                **file_record(extension_path),
                "policy_pass": extension_pass,
            }
        for variant_id, export in sorted(receipt.get("exports", {}).items()):
            records = export.get("artifacts", {})
            verified_records: dict[str, Any] = {}
            for kind, record in records.items():
                check = verify_record(record)
                check["family_id"] = family_id
                check["variant_id"] = variant_id
                check["kind"] = kind
                artifact_checks.append(check)
                verified_records[kind] = record
            onnx_record = records.get("onnx")
            onnx_checker_pass = False
            if onnx_record:
                try:
                    onnx.checker.check_model(onnx.load(ROOT / onnx_record["path"]))
                    onnx_checker_pass = True
                except Exception:
                    onnx_checker_pass = False
            onnx_evidence = export.get("onnx", {})
            onnx_receipt_pass = bool(
                onnx_evidence.get("checker") == "PASS"
                and onnx_evidence.get("ort_fixture") == "PASS"
                and all(item.get("all_finite") and item.get("max_abs", 1.0) <= 1e-4 for item in onnx_evidence.get("parity", []))
            )
            candidate_id = f"{family_id}:{variant_id}"
            family_class = receipt.get("candidate_class")
            if variant_id == primary_variant:
                variant_class = family_class
                variant_role = "PRIMARY_LOCKED_VARIANT"
            else:
                variant_class = "RND_SUPPORTING_EXPERIMENTAL" if family_class == "RND_USABLE_EXPERIMENTAL" else "RND_SUPPORTING_VARIANT"
                variant_role = "SUPPORTING_VARIANT"
            candidate = {
                "candidate_id": candidate_id,
                "family_id": family_id,
                "variant_id": variant_id,
                "candidate_class": variant_class,
                "variant_role": variant_role,
                "states": [
                    "PC_RND_ONLY",
                    "NOT_IN_OFFICIAL_ACCOUNTING",
                    "NOT_RELEASED",
                    "NOT_DEPLOYED",
                    "NOT_X5_VERIFIED",
                    "PASSIVE_MINIMAL_MANUAL",
                ],
                "artifacts": verified_records,
                "onnx_validation": {
                    "receipt_pass": onnx_receipt_pass,
                    "fresh_checker_pass": onnx_checker_pass,
                },
                "official_registry_member": False,
                "release_created": False,
                "deployed": False,
                "x5_verified": False,
            }
            candidates.append(candidate)
            family["exports"].append(candidate_id)
        family["pass"] = bool(
            family["receipt_seal_pass"]
            and family["fit_count_match"]
            and all(restrictions.values())
            and supplemental_pass
            and len(family["exports"]) > 0
        )
        families.append(family)
    for item in CONDITIONAL_SKIPS:
        families.append(
            {
                **item,
                "fit_count": 0,
                "candidate_class": "SKIPPED_BUDGET_REALLOCATED",
                "state": "NOT_TRAINED",
                "exports": [],
                "pass": True,
            }
        )
    return families, candidates, artifact_checks


def build_summary_markdown(acceptance: dict[str, Any], families: list[dict[str, Any]]) -> str:
    lines = [
        "# Post-50 Passive R&D Execution Summary",
        "",
        f"Generated: `{acceptance['created_at']}`",
        "",
        "This is an isolated PC R&D result. The official 50-model registry, acceptance files, release archives, production services, and X5 board were not modified or contacted.",
        "",
        "## Outcome",
        "",
        f"- Execution acceptance: **{acceptance['execution_acceptance']}**",
        f"- Completed receipted physical fits: **{acceptance['fit_accounting']['completed_receipted_fit_count']} / {acceptance['fit_accounting']['physical_fit_cap']}**",
        f"- Registered candidate weights: **{acceptance['candidate_accounting']['registered_candidate_weights']} / {acceptance['candidate_accounting']['cap']}**",
        f"- Frozen official files unchanged: **{acceptance['frozen50']['passed']} / {acceptance['frozen50']['checked']}**",
        f"- X5-verified candidates: **0**",
        "",
        "## Family classification",
        "",
        "| Family | Name | Fits | Gate | Classification | Exports |",
        "|---|---|---:|---|---|---:|",
    ]
    for family in families:
        gate = family.get("innovation_gate")
        gate_text = "PASS" if isinstance(gate, dict) and gate.get("pass") else ("FAIL" if isinstance(gate, dict) else "N/A")
        lines.append(
            f"| {family['family_id']} | {family.get('model_name') or family.get('name') or '-'} | {family.get('fit_count', 0)} | {gate_text} | {family.get('candidate_class')} | {len(family.get('exports', []))} |"
        )
    lines += [
        "",
        "## Non-negotiable boundaries",
        "",
        "- HERO's first 40 fits remain retained as `INVALID_PIPELINE`; the 16-fit recovery did not pass its innovation gate and is experimental only.",
        "- Conditional JARVIS dielectric and thermoelectric families were not trained; their 16-fit budget was reallocated to the HERO recovery, preserving the 160-fit cap.",
        "- Six partial attempts (one XRD runtime optimization and five SEM training/control-flow corrections) were stopped before completion or receipt creation; they produced no candidates and are disclosed separately from the 160 completed receipted fits.",
        "- A passed execution audit proves reproducibility and isolation, not X5 performance or scientific superiority.",
        "- No new release ZIP, OpenExplorer compilation, service, port, dashboard integration, camera access, routing action, or enforcement path was created.",
        "- Any future board work remains blocked until the user separately confirms that the AI-brain X5 is powered and the PC is manually on the same LAN.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    plan = load_json(CONTRACT_ROOT / "training_plan.lock.json")
    amendment = load_json(CONTRACT_ROOT / "training_plan.amendment.hero_recovery.v1.json")
    passive_contract = load_json(CONTRACT_ROOT / "passive_minimal_manual.v1.json")
    family_fit_checks, fit_receipt_checks = verify_fit_receipts()
    families, candidates, artifact_checks = collect_families()

    candidate_checkpoint_hashes = [
        item["artifacts"]["checkpoint"]["sha256"]
        for item in candidates
        if "checkpoint" in item.get("artifacts", {})
    ]
    collisions = len(candidate_checkpoint_hashes) - len(set(candidate_checkpoint_hashes))
    family_receipt_pass = all(item.get("pass") for item in families)
    fit_receipt_pass = all(item.get("pass") for item in fit_receipt_checks)
    fit_count_pass = all(item.get("pass") for item in family_fit_checks)
    artifact_pass = all(item.get("pass") for item in artifact_checks)
    onnx_pass = all(
        item["onnx_validation"]["receipt_pass"] and item["onnx_validation"]["fresh_checker_pass"]
        for item in candidates
    )
    physical_fit_count = sum(item["actual"] for item in family_fit_checks)

    frozen_audit = verify_frozen50(CONTRACT_ROOT / "frozen50_pre_hashes.v1.json")
    frozen_payload = {
        "schema": "x5_icmat_foundry.frozen50_pre_post_hash_audit.v1",
        "created_at": utc_now(),
        "pre_hash_contract": relative(CONTRACT_ROOT / "frozen50_pre_hashes.v1.json"),
        "checked": len(frozen_audit["files"]),
        "passed": sum(1 for item in frozen_audit["files"] if item["pass"]),
        "all_pass": frozen_audit["all_pass"],
        "files": frozen_audit["files"],
        "network_used": False,
        "x5_contacted": False,
    }
    frozen_path = EVIDENCE_ROOT / "final_acceptance/frozen50_pre_post_hash_audit.v1.json"
    write_json(frozen_path, frozen_payload, seal=True)

    registry = {
        "schema": "x5_icmat_foundry.post50_rnd_registry.v1",
        "created_at": utc_now(),
        "plan_id": plan["plan_id"],
        "official_registry_member": False,
        "official_model_count_unchanged": 50,
        "official_registry_path": "icmat_foundry/finals_50model/contracts/model_registry.v3.json",
        "accounting_scope": "ISOLATED_POST50_PC_RND_ONLY",
        "candidate_weight_cap": plan["registered_candidate_weight_cap"],
        "registered_candidate_weight_count": len(candidates),
        "canonical_checkpoint_hash_collisions": collisions,
        "families": families,
        "candidates": candidates,
        "invalidated_evidence": [
            {
                "family_id": "RND-PROC-01",
                "fit_count": 40,
                "disposition": "INVALID_PIPELINE",
                "incident": file_record(EVIDENCE_ROOT / "RND-PROC-01/invalid_pipeline_incident.v1.json"),
                "promotion_allowed": False,
            }
        ],
        "conditional_families": {
            "disposition": amendment["conditional_families_disposition"],
            "budget_reallocated_fits": 16,
        },
        "release_created": False,
        "deployed": False,
        "x5_verified_count": 0,
        "execution_policy": "PASSIVE_MINIMAL_MANUAL",
        "network_used": False,
        "x5_contacted": False,
    }
    registry_path = CONTRACT_ROOT / "rnd_registry.v1.json"
    write_json(registry_path, registry, seal=True)

    artifact_inventory = {
        "schema": "x5_icmat_foundry.post50_artifact_inventory.v1",
        "created_at": utc_now(),
        "candidate_count": len(candidates),
        "artifact_record_count": len(artifact_checks),
        "all_pass": artifact_pass,
        "records": artifact_checks,
    }
    inventory_path = EVIDENCE_ROOT / "final_acceptance/artifact_inventory.v1.json"
    write_json(inventory_path, artifact_inventory, seal=True)

    checks = {
        "fit_count_exact": fit_count_pass and physical_fit_count == plan["total_fit_cap"],
        "fit_receipts_valid": fit_receipt_pass,
        "family_receipts_valid": family_receipt_pass,
        "artifact_integrity": artifact_pass,
        "onnx_static_fixture_validation": onnx_pass,
        "candidate_weight_cap": len(candidates) <= plan["registered_candidate_weight_cap"],
        "candidate_checkpoint_hash_collisions_zero": collisions == 0,
        "frozen50_unchanged": frozen_audit["all_pass"],
        "passive_contract_deployed_off": passive_contract["rb_voe"]["state"] == "DEPLOYED_OFF",
        "no_release": registry["release_created"] is False,
        "no_deployment": registry["deployed"] is False,
        "no_x5_verification_claim": registry["x5_verified_count"] == 0,
        "no_network_or_x5_contact": registry["network_used"] is False and registry["x5_contacted"] is False,
    }
    acceptance = {
        "schema": "x5_icmat_foundry.post50_execution_acceptance.v1",
        "created_at": utc_now(),
        "execution_acceptance": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "fit_accounting": {
            "completed_receipted_fit_count": physical_fit_count,
            "physical_fit_cap": plan["total_fit_cap"],
            "per_family": family_fit_checks,
            "invalid_pipeline_fits_retained": 40,
            "valid_recovery_fits": 16,
            "conditional_fits_skipped": 16,
            "interrupted_partial_attempts_not_counted": 6,
            "interrupted_attempt_evidence": [
                file_record(EVIDENCE_ROOT / "RND-XRD-01/interrupted_attempt.v1.json"),
                file_record(EVIDENCE_ROOT / "RND-SEM-01/epoch_extension.v1.json"),
            ],
        },
        "candidate_accounting": {
            "registered_candidate_weights": len(candidates),
            "cap": plan["registered_candidate_weight_cap"],
            "checkpoint_hash_collisions": collisions,
        },
        "family_outcomes": [
            {
                "family_id": item["family_id"],
                "candidate_class": item.get("candidate_class"),
                "innovation_gate_pass": item.get("innovation_gate", {}).get("pass") if isinstance(item.get("innovation_gate"), dict) else None,
                "fit_count": item.get("fit_count", 0),
                "export_count": len(item.get("exports", [])),
            }
            for item in families
        ],
        "frozen50": {
            "checked": len(frozen_audit["files"]),
            "passed": sum(1 for item in frozen_audit["files"] if item["pass"]),
            "audit": file_record(frozen_path),
        },
        "registry": file_record(registry_path),
        "artifact_inventory": file_record(inventory_path),
        "claim_boundary": [
            "Official 50-model accounting remains unchanged.",
            "PC training and ONNX Runtime fixtures are not X5 board evidence.",
            "Execution acceptance is not a claim that every scientific innovation gate passed.",
            "All candidates are passive, manual, isolated, unreleased, undeployed, and not X5 verified.",
            "RB-VoE remains DEPLOYED_OFF and was not made a dependency or enforcement path.",
        ],
        "release_created": False,
        "openexplorer_run": False,
        "x5_contacted": False,
        "network_used": False,
    }
    acceptance_path = EVIDENCE_ROOT / "final_acceptance/post50_execution_acceptance.v1.json"
    write_json(acceptance_path, acceptance, seal=True)
    summary_path = EVIDENCE_ROOT / "final_acceptance/EXECUTION_SUMMARY.md"
    summary_path.write_text(build_summary_markdown(acceptance, families), encoding="utf-8")

    result = {
        "execution_acceptance": acceptance["execution_acceptance"],
        "completed_receipted_fit_count": physical_fit_count,
        "registered_candidate_weights": len(candidates),
        "family_gate_passes": sum(1 for item in families if isinstance(item.get("innovation_gate"), dict) and item["innovation_gate"].get("pass")),
        "frozen50": f"{frozen_payload['passed']}/{frozen_payload['checked']}",
        "registry": relative(registry_path),
        "acceptance": relative(acceptance_path),
        "summary": relative(summary_path),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if acceptance["execution_acceptance"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
