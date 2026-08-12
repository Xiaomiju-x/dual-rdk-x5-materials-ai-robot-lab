#!/usr/bin/env python3
"""Reconcile the six SEM candidates after the bounded F-SEM-01/DINOv2 corrections."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SCOPE = ROOT / "icmat_foundry/finals_50model"
EVIDENCE = SCOPE / "evidence/sem_bank"
CONTRACTS = SCOPE / "contracts"
RECEIPTS = {
    "F-SEM-01": EVIDENCE / "F-SEM-01/training_receipt.v1.json",
    "F-SEM-02": EVIDENCE / "F-SEM-02/training_receipt.v1.json",
    "F-SEM-03": EVIDENCE / "F-SEM-03/training_receipt.v1.json",
    "F-SEM-04": EVIDENCE / "F-SEM-04/training_receipt.v1.json",
    "F-SEM-05": EVIDENCE / "F-SEM-05/evaluation_receipt.v1.json",
    "F-SEM-06": EVIDENCE / "F-SEM-06/training_receipt.v1.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii", newline="\n"
    )
    return digest


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def accepted(inventory_id: str, receipt: dict[str, Any]) -> bool:
    metrics = receipt["metrics"]
    if inventory_id in {"F-SEM-01", "F-SEM-02", "F-SEM-03", "F-SEM-04", "F-SEM-06"}:
        return bool(metrics.get("passes_simple_baseline"))
    if inventory_id == "F-SEM-05":
        learned = metrics["cosine_centroid_ood"]["auroc"]
        baseline = metrics["pixel_mean_std_baseline"]["auroc"]
        return (
            receipt.get("model_id") == "DINOv2-SEM-OOD-CPU"
            and receipt.get("backend") == "CPU"
            and learned > baseline
            and receipt.get("ort_evidence", {}).get("load_and_infer") == "PASS"
        )
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    loaded = {model_id: json.loads(path.read_text(encoding="utf-8")) for model_id, path in RECEIPTS.items()}
    failures = [model_id for model_id, receipt in loaded.items() if not accepted(model_id, receipt)]
    amendments = {
        "schema": "x5_icmat_foundry.implementation_amendments.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_registry": "icmat_foundry/finals_50model/contracts/model_registry.v3.json",
        "counting_rule_unchanged": True,
        "logical_model_count_delta": 0,
        "amendments": [
            {
                "inventory_id": "F-SEM-01",
                "registry_architecture_summary": "MobileNet/RepViT tiny / X5 BPU",
                "implemented_architecture": "SpatialTextureCNN / Bayes-e BPU candidate",
                "reason": "The first global-pooling CNN collapsed to the majority class; the single bounded repair retained a 2x2 texture layout.",
                "quality_evidence": relative(RECEIPTS["F-SEM-01"]),
            },
            {
                "inventory_id": "F-SEM-05",
                "registry_model_id": "DINOv2-SEM-OOD-CPU",
                "implemented_model_id": "DINOv2-SEM-OOD-CPU",
                "contract_alignment": "RESTORED",
                "rejected_candidate": "IndependentSEMEmbeddingCNN archived as non-registry evidence",
                "quality_evidence": relative(RECEIPTS["F-SEM-05"]),
            },
            {
                "inventory_id": "F-SEM-06",
                "registry_model_id": "NIST-SEM-ImageQuality-CPU",
                "implemented_model_id": "SEM-ImageQuality-CPU",
                "implemented_data_scope": "Carinthia real SEM with controlled degradation",
                "reason": "The official NIST intensity archive is unavailable locally, so the candidate uses controlled degradation targets on real CC BY 4.0 Carinthia SEM instead of inventing NIST image labels.",
                "task_contract_preserved": "SEM image quality and segmentation reliability proxy",
                "claim_boundary": "Controlled degradation severity is not human MOS, production tool health, or NIST intensity-set accuracy.",
                "quality_evidence": relative(RECEIPTS["F-SEM-06"]),
            },
        ],
        "x5_contacted": False,
        "production_integration_allowed": False,
    }
    bank = {
        "schema": "x5_icmat_foundry.sem_bank.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state": "PASS" if not failures else "FAIL",
        "model_count": 6,
        "bpu_candidate_count": 4,
        "cpu_model_count": 2,
        "hard_failures": failures,
        "x5_contacted": False,
        "production_files_modified": False,
        "models": [
            {
                "inventory_id": model_id,
                "model_id": receipt["model_id"],
                "receipt_path": relative(RECEIPTS[model_id]),
                "receipt_sha256": sha256_file(RECEIPTS[model_id]),
                "accepted": accepted(model_id, receipt),
                "candidate_status": receipt.get("candidate_status", receipt.get("status")),
                "backend": receipt.get("backend_target", receipt.get("backend")),
            }
            for model_id, receipt in loaded.items()
        ],
        "amendment_contract": "icmat_foundry/finals_50model/contracts/implementation_amendments.v1.json",
    }
    preview = {"bank": bank, "amendments": amendments}
    if not args.execute:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0 if not failures else 2
    amendment_sha = write_json(CONTRACTS / "implementation_amendments.v1.json", amendments)
    bank["amendment_contract_sha256"] = amendment_sha
    bank_sha = write_json(EVIDENCE / "sem_bank_receipt.v2.json", bank)
    print(json.dumps({"state": bank["state"], "bank_sha256": bank_sha, "amendment_sha256": amendment_sha}, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
