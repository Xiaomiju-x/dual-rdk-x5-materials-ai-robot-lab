from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from icmat_foundry.finals_post50_passive_rnd_20260803.training.common import (
    CONTRACT_ROOT,
    EVIDENCE_ROOT,
    SEED_BASE,
    ThermalGuard,
    bootstrap_ci,
    export_static_onnx,
    relative,
    utc_now,
    write_json,
)
from icmat_foundry.finals_post50_passive_rnd_20260803.training.train_hero_vm17 import (
    CONFIGS,
    DOMAINS,
    FAMILY_ID,
    INPUT_WIDTH,
    assemble,
    fit_preprocess,
    fit_ridge,
    load_data,
    metrics,
    predict,
    preprocessing_arrays,
    train_torch,
)


def main() -> int:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    thermal = ThermalGuard()
    data = load_data()
    # Recovery budget is exactly 16 fits: 3 Ridge teachers, 8 full-grid fits on
    # fold 0, 2 confirmation fits on fold 1, and 3 frozen final variants.
    fold_definitions = ((0.60, 0.70), (0.70, 0.80))
    ranked: list[dict[str, Any]] = []
    first_scores: dict[str, float] = {}
    for fold, (train_fraction, val_fraction) in enumerate(fold_definitions):
        train_idx, val_idx = {}, {}
        for domain in DOMAINS:
            length = len(data[domain][0])
            train_idx[domain] = np.arange(0, int(length * train_fraction))
            val_idx[domain] = np.arange(int(length * train_fraction), int(length * val_fraction))
        preprocess = fit_preprocess(data, train_idx)
        train = assemble(data, train_idx, preprocess)
        val = assemble(data, val_idx, preprocess)
        teacher, _, _ = fit_ridge(
            f"repair_teacher_fold{fold}", train[0], train[1], val[0], val[1], val[2]
        )
        teacher_train = teacher.predict(train[0]).astype(np.float32)
        teacher_val = teacher.predict(val[0]).astype(np.float32)
        selected_configs = list(CONFIGS) if fold == 0 else ranked[:2]
        current_scores = {}
        for index, config in enumerate(selected_configs):
            _, selection, _ = train_torch(
                f"repair_fold{fold}_{config['id']}", config,
                SEED_BASE + 3000 + fold * 100 + index, preprocess, train, val,
                teacher_train, teacher_val, device, thermal,
            )
            current_scores[config["id"]] = float(selection["selection_score"])
        if fold == 0:
            first_scores = current_scores
            ranked = sorted(CONFIGS, key=lambda config: current_scores[config["id"]])
        else:
            ranked = sorted(ranked[:2], key=lambda config: 0.5 * first_scores[config["id"]] + 0.5 * current_scores[config["id"]]) + ranked[2:]

    train_idx, val_idx, test_idx = {}, {}, {}
    for domain in DOMAINS:
        length = len(data[domain][0])
        train_idx[domain] = np.arange(0, int(length * 0.75))
        val_idx[domain] = np.arange(int(length * 0.75), int(length * 0.85))
        test_idx[domain] = np.arange(int(length * 0.85), length)
    preprocess = fit_preprocess(data, train_idx)
    train = assemble(data, train_idx, preprocess)
    val = assemble(data, val_idx, preprocess)
    test = assemble(data, test_idx, preprocess)
    teacher, _, _ = fit_ridge("repair_teacher_final", train[0], train[1], val[0], val[1], val[2])
    teacher_train = teacher.predict(train[0]).astype(np.float32)
    teacher_val = teacher.predict(val[0]).astype(np.float32)
    selected = copy.deepcopy(ranked[0])
    variants = [
        ("recovery_quality", selected, SEED_BASE + 3900),
        ("recovery_robust", next(config for config in CONFIGS if config["id"] == "v6_hero_full"), SEED_BASE + 3901),
        ("recovery_lite", next(config for config in CONFIGS if config["id"] == "v7_hero_lite"), SEED_BASE + 3902),
    ]
    models = {}
    for variant, config, seed in variants:
        model, _, _ = train_torch(
            f"{variant}_{config['id']}", config, seed, preprocess, train, val,
            teacher_train, teacher_val, device, thermal,
        )
        models[variant] = model.cpu().eval()
    lock = {
        "schema": "x5_icmat_foundry.post50_selection_lock.v1",
        "created_at": utc_now(),
        "family_id": FAMILY_ID,
        "kind": "PRE_REGISTERED_PIPELINE_RECOVERY",
        "invalid_attempt_receipt": "icmat_foundry/finals_post50_passive_rnd_20260803/evidence/RND-PROC-01/family_receipt.v1.json",
        "invalid_reason": "missing evidenced PVD normalization floor and clip",
        "recovery_fit_count": 16,
        "ranked_configuration_ids": [config["id"] for config in ranked],
        "variants": [{"variant": name, "configuration": config["id"], "seed": seed} for name, config, seed in variants],
        "test_previously_opened_by_invalid_pipeline": True,
        "post_recovery_test_retuning_allowed": False,
    }
    lock_path = CONTRACT_ROOT / "selection_locks/RND-PROC-01.recovery.json"
    write_json(lock_path, lock, seal=True)

    # Confirmation reuses the frozen tail after the invalid pipeline attempt;
    # it is explicitly not described as a pristine unseen test.
    mean_baseline = np.zeros_like(test[1])
    for domain_index, domain in enumerate(DOMAINS):
        mask = test[2] == domain_index
        mean_baseline[mask] = data[domain][1][train_idx[domain]].mean(axis=0)
    baseline_metrics = metrics(test[1], mean_baseline, test[2])
    ridge_metrics = metrics(test[1], teacher.predict(test[0]).astype(np.float32), test[2])
    locked_metrics, exports = {}, {}
    for variant, model in models.items():
        prediction = predict(model.to(device), test[0], device)
        locked_metrics[variant] = metrics(test[1], prediction, test[2])
        exports[variant] = export_static_onnx(
            FAMILY_ID, variant, model.cpu(),
            [torch.from_numpy(test[0][:1].reshape(1, 1, 1, INPUT_WIDTH))],
            ["pvd_features_fp32"], ["thickness17"], preprocessing_arrays(preprocess),
            {"configuration": next(item[1] for item in variants if item[0] == variant), "seed": next(item[2] for item in variants if item[0] == variant)},
        )
    quality = locked_metrics["recovery_quality"]
    gate = {
        "combined_beats_domain_mean": quality["combined"]["mae"] < baseline_metrics["combined"]["mae"],
        "alcu_not_worse_than_domain_mean": quality["per_domain"]["AlCu"]["mae"] <= baseline_metrics["per_domain"]["AlCu"]["mae"],
        "wti_not_worse_than_domain_mean": quality["per_domain"]["WTi"]["mae"] <= baseline_metrics["per_domain"]["WTi"]["mae"],
        "relative_improvement_ge_5pct": quality["combined"]["mae"] <= 0.95 * baseline_metrics["combined"]["mae"],
    }
    gate["pass"] = all(gate.values())
    prediction = predict(models["recovery_quality"].to(device), test[0], device)
    receipt = {
        "schema": "x5_icmat_foundry.post50_family_recovery_receipt.v1",
        "created_at": utc_now(),
        "family_id": FAMILY_ID,
        "state": "PC_FIXED_FIXTURE_VALIDATED",
        "candidate_class": "RND_INNOVATION_ANCHOR" if gate["pass"] else "RND_USABLE_EXPERIMENTAL",
        "original_fit_count": 40,
        "original_fit_disposition": "INVALID_PIPELINE",
        "recovery_fit_count": 16,
        "total_physical_fits": 56,
        "conditional_jarvis_budget_reallocated": 16,
        "selection_lock": {"path": relative(lock_path), "test_previously_opened_by_invalid_pipeline": True},
        "confirmation_set_status": "FROZEN_TAIL_REUSED_AFTER_INVALID_PIPELINE_NOT_PRISTINE_UNSEEN",
        "baseline": baseline_metrics,
        "ridge": ridge_metrics,
        "variants": locked_metrics,
        "quality_per_sample_mae_bootstrap95": bootstrap_ci(np.mean(np.abs(test[1] - prediction), axis=1)),
        "innovation_gate": gate,
        "exports": exports,
        "execution_policy": "PASSIVE_MINIMAL_MANUAL",
        "official_registry_member": False,
        "release_created": False,
        "deployed": False,
        "x5_verified": False,
        "network_used": False,
        "x5_contacted": False,
        "thermal_samples": thermal.samples,
    }
    receipt_path = EVIDENCE_ROOT / FAMILY_ID / "recovery_family_receipt.v1.json"
    write_json(receipt_path, receipt, seal=True)
    print(json.dumps({"family": FAMILY_ID, "recovery_fits": 16, "gate": gate, "receipt": relative(receipt_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
