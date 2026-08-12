"""Verify the generated finals model registry and fast-track invariants."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "icmat_foundry" / "finals_50model" / "contracts" / "model_registry.v3.json"


def main() -> None:
    payload = REGISTRY.read_bytes()
    data = json.loads(payload)
    models = data["models"]
    ids = [item["inventory_id"] for item in models]
    model_ids = [item["model_id"] for item in models]
    new = [item for item in models if item["inventory_id"].startswith("F-")]
    counts = Counter(item["primary_backend"] for item in models)
    new_counts = Counter(item["primary_backend"] for item in new)
    family_counts = Counter(item["family"] for item in new)
    checks = {
        "state": data["state"] == "FAST_TRACK_CONTRACT_FROZEN",
        "total_50": len(models) == 50,
        "inventory_ids_unique": len(ids) == len(set(ids)),
        "logical_model_ids_unique": len(model_ids) == len(set(model_ids)),
        "new_38": len(new) == 38,
        "new_cpu_14": new_counts == Counter({"BPU": 24, "CPU": 14}),
        "all_backends": counts == Counter({"BPU": 34, "CPU": 15, "PC": 1}),
        "families": family_counts
        == Counter({"PROC": 9, "MAT": 8, "SEM": 6, "LLM": 5, "KNW": 4, "PKG": 4, "XRD": 2}),
        "authority_zero": all(item["authority"] == 0 for item in models),
        "no_production_integration": data["production_integration_allowed"] is False,
        "x5_not_contacted": data["x5_contacted"] is False,
    }
    result = {
        "schema": "x5_icmat_foundry.registry_verification.v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "registry_sha256": hashlib.sha256(payload).hexdigest(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
