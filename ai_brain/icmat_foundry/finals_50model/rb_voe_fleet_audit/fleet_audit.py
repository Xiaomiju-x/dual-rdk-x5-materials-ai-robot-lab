#!/usr/bin/env python3
"""Read-only PASSIVE_ONESHOT audit for the 50-model finals release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
FINAL_ROOT = HERE.parent
REPO_ROOT = FINAL_ROOT.parents[1]
CONTRACT = HERE / "contract.v1.json"
REGISTRY = FINAL_ROOT / "contracts" / "model_registry.v3.json"
ACCEPTANCE = FINAL_ROOT / "evidence" / "final_acceptance" / "final_acceptance.v1.json"
EVIDENCE = HERE / "evidence"
FROZEN_RELEASES = [
    {
        "kind": "pc",
        "path": "icmat_foundry/finals_50model/releases/x5-icmat-foundry-50model-pc-c7aff501602bde2f.zip",
        "sha256": "c7aff501602bde2f31881bdbc9d872c9570162359e6b6e8596b463c73add2a81",
    },
    {
        "kind": "x5-staging",
        "path": "icmat_foundry/finals_50model/releases/x5-icmat-foundry-50model-x5-staging-c5fa215a58168c0c.zip",
        "sha256": "c5fa215a58168c0cb7274c2b1cf6d66bcd0f3c1e70d3f4cf13749e9b57dafb52",
    },
]


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def check(name: str, passed: bool, observed: Any, expected: Any) -> dict[str, Any]:
    return {"name": name, "status": "PASS" if passed else "FAIL", "observed": observed, "expected": expected}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def release_records(acceptance: dict[str, Any]) -> list[dict[str, Any]]:
    """Return immutable release records across pre/post-build acceptance snapshots."""
    records = acceptance.get("release", {}).get("artifacts")
    return records if isinstance(records, list) and records else FROZEN_RELEASES


def audit_release(record: dict[str, Any], checks: list[dict[str, Any]]) -> dict[str, Any]:
    path = REPO_ROOT / record["path"]
    actual_sha = sha256(path) if path.is_file() else None
    checks.append(check(f"release.{record['kind']}.sha256", actual_sha == record["sha256"], actual_sha, record["sha256"]))
    if not path.is_file():
        return {"kind": record["kind"], "path": record["path"], "present": False}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        policy = json.loads(archive.read("release_policy.json"))
        manifest = json.loads(archive.read("release_manifest.json"))
    unsafe = [name for name in names if name.startswith(("/", "\\")) or ".." in Path(name).parts]
    checks.extend(
        [
            check(f"release.{record['kind']}.duplicates", len(names) == len(set(names)), len(names) - len(set(names)), 0),
            check(f"release.{record['kind']}.safe_paths", not unsafe, unsafe, []),
            check(f"release.{record['kind']}.manual_only", policy.get("install_mode") == "MANUAL_STAGING_ONLY", policy.get("install_mode"), "MANUAL_STAGING_ONLY"),
            check(f"release.{record['kind']}.no_autostart", policy.get("automatic_start") is False and policy.get("contains_startup_service") is False, policy, "both false"),
            check(f"release.{record['kind']}.no_production_overwrite", policy.get("production_overwrite") is False and policy.get("production_paths_allowed") == [], policy, "false and []"),
            check(f"release.{record['kind']}.rb_voe_off", policy.get("rb_voe_state") == "DEPLOYED_OFF", policy.get("rb_voe_state"), "DEPLOYED_OFF"),
            check(f"release.{record['kind']}.no_x5_contact", policy.get("x5_contacted") is False, policy.get("x5_contacted"), False),
        ]
    )
    llm_assets = {}
    for inventory_id in ("F-LLM-03", "F-LLM-04", "F-LLM-05"):
        row = next((item for item in manifest["models"] if item["inventory_id"] == inventory_id), None)
        files = row["files"] if row else []
        observed = {
            "bin_count": sum(item.endswith(".bin") for item in files),
            "cpu_tensor_count": sum("cpu_tensors" in item for item in files),
        }
        checks.append(check(f"release.{record['kind']}.{inventory_id}.runtime_assets", observed == {"bin_count": 2, "cpu_tensor_count": 3}, observed, {"bin_count": 2, "cpu_tensor_count": 3}))
        llm_assets[inventory_id] = observed
    return {
        "kind": record["kind"],
        "path": record["path"],
        "sha256": actual_sha,
        "entries": len(names),
        "manifest_models": len(manifest["models"]),
        "bpu_llm_runtime_assets": llm_assets,
    }


def audit_board_receipt(
    board_receipt_path: Path,
    board: dict[str, Any],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    accounting = board.get("accounting", {})
    models = board.get("new_candidate_models", [])
    counts = Counter(item.get("final_board_status") for item in models)
    ids = [item.get("inventory_id") for item in models]
    noninterference = board.get("final_noninterference", {})
    endpoints = noninterference.get("endpoint_status", {})
    llms = board.get("bpu_llm_fixed_contracts", [])
    boundaries = board.get("claim_boundaries", {})
    checks.extend(
        [
            check("board.schema", board.get("schema") == "x5_icmat_foundry.board_phase_acceptance.v1", board.get("schema"), "x5_icmat_foundry.board_phase_acceptance.v1"),
            check("board.phase_complete", board.get("board_phase_status") == "COMPLETE_WITH_REJECTIONS_AND_EXPERIMENTALS", board.get("board_phase_status"), "COMPLETE_WITH_REJECTIONS_AND_EXPERIMENTALS"),
            check("board.model_status_complete", len(models) == 38 and len(ids) == len(set(ids)) == 38, {"models": len(models), "unique_ids": len(set(ids))}, {"models": 38, "unique_ids": 38}),
            check("board.status_counts", dict(counts) == {"X5_VALIDATED": 31, "BOARD_REJECTED": 3, "BOARD_EXPERIMENTAL": 4}, dict(counts), {"X5_VALIDATED": 31, "BOARD_REJECTED": 3, "BOARD_EXPERIMENTAL": 4}),
            check("board.x5_local_accounting", accounting.get("x5_local_logical_models") == 49 and accounting.get("frozen_production_baseline_preserved") == 11 and accounting.get("new_finals_candidates_status_complete") == 38, accounting, "49 = 11 frozen + 38 finals"),
            check("board.actual_backends", accounting.get("actual_x5_backend_executed") == 35 and accounting.get("actual_x5_cpu_executed") == 11 and accounting.get("actual_x5_bpu_executed") == 24, accounting, {"total": 35, "CPU": 11, "BPU": 24}),
            check("board.bpu_llm_bins", accounting.get("actual_x5_bpu_segment_bins_executed") == 6, accounting.get("actual_x5_bpu_segment_bins_executed"), 6),
            check("board.bpu_llm_content_binding", len(llms) == 3 and all(item.get("content_bound") is True and item.get("part1_output_tensor_sha256") == item.get("part2_input_tensor_sha256") for item in llms), [{"inventory_id": item.get("inventory_id"), "content_bound": item.get("content_bound"), "part1": item.get("part1_output_tensor_sha256"), "part2": item.get("part2_input_tensor_sha256")} for item in llms], "3 content-bound part1/part2 pairs"),
            check("board.bpu_llm_divergence_retained", len(llms) == 3 and all(item.get("next_token_exact") is False for item in llms), [{"inventory_id": item.get("inventory_id"), "expected": item.get("expected_next_token_id"), "actual": item.get("actual_next_token_id"), "exact": item.get("next_token_exact")} for item in llms], "three BOARD_EXPERIMENTAL fixed-token divergences"),
            check("board.noninterference", noninterference.get("result") == "PASS" and noninterference.get("checks") == {"passed": 33, "failed": 0}, noninterference.get("checks"), {"passed": 33, "failed": 0}),
            check("board.frozen_endpoints", len(endpoints) == 8 and all(code == 200 for code in endpoints.values()), endpoints, "8 HTTP 200"),
            check("board.candidate_exit", noninterference.get("candidate_processes") == "" and noninterference.get("candidate_systemd_units") == "" and noninterference.get("release_service_files") == "", {"processes": noninterference.get("candidate_processes"), "units": noninterference.get("candidate_systemd_units"), "service_files": noninterference.get("release_service_files")}, "all empty"),
            check("board.claim_boundaries", boundaries.get("F_PROC_03") == "QUALITY_LIMITED_NOT_PROMOTED" and boundaries.get("SIM_ONLY") == ["F-PKG-01", "F-PKG-02", "F-PKG-03", "F-PKG-04"] and boundaries.get("board_results_are_separate_overlay") is True, boundaries, "quality/SIM_ONLY/separate-overlay retained"),
            check("board.pc_acceptance_hash_discrepancy_recorded", isinstance(boundaries.get("pc_acceptance_hash_discrepancy"), dict) and boundaries["pc_acceptance_hash_discrepancy"].get("documented_sha256") != boundaries["pc_acceptance_hash_discrepancy"].get("workspace_actual_sha256") and len(set(boundaries["pc_acceptance_hash_discrepancy"].get("embedded_release_acceptance_sha256", {}).values())) == 1, boundaries.get("pc_acceptance_hash_discrepancy"), "documented mismatch retained and both frozen release embeddings agree with workspace acceptance"),
        ]
    )

    evidence_records = list(board.get("source_artifacts", []))
    for model in models:
        evidence_records.extend(model.get("evidence", []))
    evidence_results = []
    for record in evidence_records:
        path = REPO_ROOT / record["path"]
        actual = sha256(path) if path.is_file() else None
        evidence_results.append({"path": record["path"], "expected_sha256": record["sha256"], "actual_sha256": actual})
    checks.append(
        check(
            "board.evidence_hashes",
            bool(evidence_results) and all(item["actual_sha256"] == item["expected_sha256"] for item in evidence_results),
            evidence_results,
            "all copied board evidence hashes match",
        )
    )
    return {
        "path": str(board_receipt_path.resolve()),
        "sha256": sha256(board_receipt_path),
        "status": board.get("board_phase_status"),
        "models": len(models),
        "status_counts": dict(counts),
        "evidence_records_verified": len(evidence_results),
    }


def run_passive_oneshot(board_receipt_path: Path | None = None) -> tuple[dict[str, Any], Path, str]:
    contract = load_json(CONTRACT)
    registry = load_json(REGISTRY)
    acceptance = load_json(ACCEPTANCE)
    expected = contract["expected"]
    models = registry["models"]
    finals = [model for model in models if str(model["inventory_id"]).startswith("F-")]
    backend_counts = Counter(model["primary_backend"] for model in finals)
    ids = [model["inventory_id"] for model in models]
    logical_ids = [model["model_id"] for model in models]
    checks: list[dict[str, Any]] = [
        check("registry.sha256", sha256(REGISTRY) == acceptance["registry"]["sha256"], sha256(REGISTRY), acceptance["registry"]["sha256"]),
        check("registry.total", len(models) == expected["registry_models"], len(models), expected["registry_models"]),
        check("registry.inventory_unique", len(ids) == len(set(ids)), len(set(ids)), len(ids)),
        check("registry.logical_unique", len(logical_ids) == len(set(logical_ids)), len(set(logical_ids)), len(logical_ids)),
        check("registry.finals", len(finals) == expected["finals_models"], len(finals), expected["finals_models"]),
        check("registry.finals_cpu", backend_counts["CPU"] == expected["finals_cpu_primary"], backend_counts["CPU"], expected["finals_cpu_primary"]),
        check("registry.finals_bpu", backend_counts["BPU"] == expected["finals_bpu_primary"], backend_counts["BPU"], expected["finals_bpu_primary"]),
        check("registry.authority_zero", all(model.get("authority") == 0 for model in models), sorted({model.get("authority") for model in models}), [0]),
        check("registry.production_off", registry.get("production_integration_allowed") is False, registry.get("production_integration_allowed"), False),
        check("registry.x5_not_contacted", registry.get("x5_contacted") is False, registry.get("x5_contacted"), False),
        check("acceptance.status", acceptance.get("status") == "ACCEPTED_FOR_CONTENT_ADDRESSED_STAGING", acceptance.get("status"), "ACCEPTED_FOR_CONTENT_ADDRESSED_STAGING"),
        check("acceptance.release_ready", acceptance["counts"]["release_ready_models"] == expected["release_ready_models"], acceptance["counts"]["release_ready_models"], expected["release_ready_models"]),
        check("acceptance.pending", acceptance["counts"]["pending_models"] == expected["pending_models"], acceptance["counts"]["pending_models"], expected["pending_models"]),
        check("acceptance.unique_weights", acceptance["counts"]["canonical_weight_hash_collisions"] == expected["canonical_weight_hash_collisions"], acceptance["counts"]["canonical_weight_hash_collisions"], expected["canonical_weight_hash_collisions"]),
        check("acceptance.bpu_compiled", acceptance["counts"]["bpu_pc_toolchain_compiled"] == expected["finals_bpu_primary"], acceptance["counts"]["bpu_pc_toolchain_compiled"], expected["finals_bpu_primary"]),
        check("acceptance.x5_board_pending", acceptance["counts"]["x5_board_verified"] == expected["x5_board_verified_before_power_on"], acceptance["counts"]["x5_board_verified"], expected["x5_board_verified_before_power_on"]),
        check("acceptance.no_gaps", acceptance.get("gaps") == [], acceptance.get("gaps"), []),
    ]

    observed_status = {model["inventory_id"]: " ".join(model["observed_statuses"]).upper() for model in acceptance["models"]}
    for inventory_id in contract["required_claim_boundaries"]["quality_limited"]:
        checks.append(check(f"boundary.{inventory_id}.quality_limited", "QUALITY_LIMITED" in observed_status[inventory_id], observed_status[inventory_id], "contains QUALITY_LIMITED"))
    for inventory_id in contract["required_claim_boundaries"]["sim_only"]:
        checks.append(check(f"boundary.{inventory_id}.sim_only", "SIM_ONLY" in observed_status[inventory_id], observed_status[inventory_id], "contains SIM_ONLY"))

    release_reports = [audit_release(record, checks) for record in release_records(acceptance)]
    board_report = None
    if board_receipt_path is not None:
        board_receipt_path = board_receipt_path.resolve()
        board_report = audit_board_receipt(board_receipt_path, load_json(board_receipt_path), checks)
    passed = all(item["status"] == "PASS" for item in checks)
    receipt = {
        "schema": "x5_rb_voe.fleet_audit_receipt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "PASSIVE_ONESHOT",
        "result": "PASS" if passed else "FAIL",
        "state_after_exit": "DEPLOYED_OFF",
        "checks": checks,
        "summary": {"passed": sum(item["status"] == "PASS" for item in checks), "failed": sum(item["status"] != "PASS" for item in checks)},
        "releases": release_reports,
        "board_receipt": board_report,
        "effects": {
            "network_access": False,
            "x5_access": False,
            "production_write": False,
            "service_registration": False,
            "model_loading": False,
            "camera_access": False,
            "port_access": False,
            "decision_enforcement": False,
        },
        "claim_boundary": (
            "Read-only audit of the frozen PC release plus a copied X5 board receipt; "
            "FleetAudit itself performs no X5/runtime/model/port/camera access and grants no decision authority."
            if board_report is not None
            else "PC release audit only; not X5 runtime, INT8 semantic parity, latency, memory, TOPS, or production evidence"
        ),
    }
    payload = canonical_bytes(receipt)
    digest = hashlib.sha256(payload).hexdigest()
    output = EVIDENCE / f"fleet_audit-{digest[:16]}.json"
    atomic_write(output, payload)
    return receipt, output, digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("STATUS", "PASSIVE_ONESHOT"), default="STATUS")
    parser.add_argument("--board-receipt", type=Path)
    args = parser.parse_args()
    if args.mode == "STATUS":
        print(json.dumps({"state": "DEPLOYED_OFF", "allowed_mode": "PASSIVE_ONESHOT", "enforce_implemented": False}, indent=2))
        return 0
    receipt, output, digest = run_passive_oneshot(args.board_receipt)
    print(json.dumps({"result": receipt["result"], "state_after_exit": receipt["state_after_exit"], "receipt": str(output), "sha256": digest, "summary": receipt["summary"]}, ensure_ascii=False, indent=2))
    return 0 if receipt["result"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
