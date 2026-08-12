"""Build the approved 50-model registry from the frozen Markdown plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CANDIDATE = ROOT / "icmat_foundry" / "finals_50model"
PLAN = (
    ROOT
    / "docs"
    / "ai_brain_finals_20260728"
    / "X5_ICMAT_FOUNDRY_50_MODEL_FINALS_EXECUTION_PLAN_20260801.md"
)
OUTPUT = CANDIDATE / "contracts" / "model_registry.v3.json"
SHA_OUTPUT = OUTPUT.with_suffix(OUTPUT.suffix + ".sha256")


EXISTING_MODELS = (
    ("E-MDL-01", "Qwen2-0.5B-generic", "LLM", "BPU"),
    ("E-MDL-02", "Qwen2.5-1.5B-NIR-SFT", "LLM", "CPU"),
    ("E-MDL-03", "Qwen3-1.7B-NIR", "LLM", "BPU"),
    ("E-MDL-04", "R1-Distill-Qwen-1.5B", "LLM", "BPU"),
    ("E-MDL-05", "Qwen2-0.5B-NIR", "LLM", "BPU"),
    ("E-MDL-06", "Qwen2-0.5B-verdict", "LLM", "BPU"),
    ("E-MDL-07", "xrd_mlp_classify", "XRD", "BPU"),
    ("E-MDL-08", "xrd_mlp_fine", "XRD", "BPU"),
    ("E-MDL-09", "pl_mlp_classify", "PL", "BPU"),
    ("E-MDL-10", "yolo_xrd_detect", "XRD", "BPU"),
    ("E-MDL-11", "pl_detect", "PL", "BPU"),
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_new_models(plan_text: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for raw_line in plan_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("| F-"):
            continue
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) != 8:
            raise ValueError(f"unexpected model table row: {raw_line}")
        inventory_id, model_id, task, data, architecture, acceptance = cells[1:7]
        if not inventory_id.rsplit("-", 1)[-1].isdigit():
            continue
        model_id = model_id.strip("`")
        if "X5 BPU" in architecture:
            backend = "BPU"
        elif "X5 CPU" in architecture:
            backend = "CPU"
        else:
            raise ValueError(f"missing X5 backend for {inventory_id}: {architecture}")
        family = inventory_id.split("-")[1]
        records.append(
            {
                "inventory_id": inventory_id,
                "model_id": model_id,
                "family": family,
                "primary_backend": backend,
                "runtime_scope": "X5_ON_DEMAND",
                "task_contract_summary": task,
                "data_and_reuse_summary": data,
                "architecture_summary": architecture,
                "acceptance_summary": acceptance,
                "authority": 0,
                "counted_unique_logical_model": True,
                "status": "PLANNED",
            }
        )
    return records


def validate_registry(registry: dict[str, object]) -> None:
    models = registry["models"]
    if not isinstance(models, list):
        raise TypeError("models must be a list")
    ids = [str(item["inventory_id"]) for item in models]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate inventory_id")
    new_models = [item for item in models if str(item["inventory_id"]).startswith("F-")]
    existing = [item for item in models if str(item["inventory_id"]).startswith("E-MDL-")]
    pc_models = [item for item in models if item["primary_backend"] == "PC"]
    new_cpu = [item for item in new_models if item["primary_backend"] == "CPU"]
    new_bpu = [item for item in new_models if item["primary_backend"] == "BPU"]
    all_cpu = [item for item in models if item["primary_backend"] == "CPU"]
    all_bpu = [item for item in models if item["primary_backend"] == "BPU"]
    expected = {
        "models": 50,
        "existing": 11,
        "new": 38,
        "new_cpu": 14,
        "new_bpu": 24,
        "all_cpu": 15,
        "all_bpu": 34,
        "pc": 1,
    }
    actual = {
        "models": len(models),
        "existing": len(existing),
        "new": len(new_models),
        "new_cpu": len(new_cpu),
        "new_bpu": len(new_bpu),
        "all_cpu": len(all_cpu),
        "all_bpu": len(all_bpu),
        "pc": len(pc_models),
    }
    if actual != expected:
        raise ValueError(f"registry accounting mismatch: {actual} != {expected}")


def main() -> None:
    plan_bytes = PLAN.read_bytes()
    plan_text = plan_bytes.decode("utf-8-sig")
    existing = [
        {
            "inventory_id": inventory_id,
            "model_id": model_id,
            "family": family,
            "primary_backend": backend,
            "runtime_scope": "X5_FROZEN_PRODUCTION",
            "authority": 0,
            "counted_unique_logical_model": True,
            "status": "FROZEN_BASELINE_LINEAGE_AUDIT_PENDING",
        }
        for inventory_id, model_id, family, backend in EXISTING_MODELS
    ]
    new_models = parse_new_models(plan_text)
    mace = {
        "inventory_id": "P-MDL-01",
        "model_id": "MACE-MPA-0",
        "family": "PHYSICS",
        "primary_backend": "PC",
        "runtime_scope": "PC_OFFLINE_X5_HASHED_CACHE",
        "authority": 0,
        "counted_unique_logical_model": True,
        "status": "EXISTING_PC_OFFLINE_MODEL",
    }
    registry: dict[str, object] = {
        "schema": "x5_icmat_foundry.model_registry.v3",
        "state": "FAST_TRACK_CONTRACT_FROZEN",
        "approved_at_local": "2026-08-01",
        "source_plan": str(PLAN.relative_to(ROOT)).replace("\\", "/"),
        "source_plan_sha256": sha256_bytes(plan_bytes),
        "counting_rule": "unique_task_contract_plus_unique_weights; backend/export/seed/prompt variants do not add models",
        "production_integration_allowed": False,
        "x5_contacted": False,
        "models": existing + new_models + [mace],
    }
    validate_registry(registry)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(registry, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    OUTPUT.write_bytes(payload)
    SHA_OUTPUT.write_text(f"{sha256_bytes(payload)}  {OUTPUT.name}\n", encoding="ascii")
    print(
        json.dumps(
            {
                "status": "PASS",
                "models": 50,
                "new_models": 38,
                "new_cpu": 14,
                "new_bpu": 24,
                "registry_sha256": sha256_bytes(payload),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
