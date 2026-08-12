"""Fast static validation for the isolated BPU LLM sidecar."""
from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> int:
    models = json.loads((HERE / "contracts/models.v1.json").read_text(encoding="utf-8"))
    swap = json.loads((HERE / "contracts/swap_load.v1.json").read_text(encoding="utf-8"))
    ids = [item["inventory_id"] for item in models["models"]]
    assert ids == ["F-LLM-03", "F-LLM-04", "F-LLM-05"]
    arch = models["architecture_contract"]
    assert arch["num_hidden_layers"] == 24
    assert [(item["layer_start"], item["layer_end_inclusive"]) for item in arch["segments"]] == [(0, 11), (12, 23)]
    assert arch["march"] == "bayes-e"
    assert swap["default_state"] == "DEPLOYED_OFF"
    assert swap["startup"]["autostart"] is False
    assert swap["authority"]["decision_authority"] is False
    assert swap["residency"]["maximum_loaded_domain_models"] == 1
    assert (HERE / "evidence/legacy_24layer_chain_audit.v1.json").is_file()
    print("SIDE_CAR_STATIC_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
