from __future__ import annotations

import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from icmat_foundry.finals_post50_passive_rnd_20260803.training.common import (
    CONTRACT_ROOT,
    FitResult,
    SEED_BASE,
    ThermalGuard,
    bootstrap_ci,
    export_static_onnx,
    mae,
    parameter_count,
    r2,
    rmse,
    save_fit,
    set_seed,
    state_to_cpu,
    utc_now,
    write_family_receipt,
    write_json,
)


FAMILY_ID = "RND-PROC-01"
SOURCE = ROOT / "research/icmat_foundry/fabyield_replacement_20260728/candidates/zenodo_16881338"
DOMAINS = ("AlCu", "WTi")
DOMAIN_WIDTH = 108
INPUT_WIDTH = 218


@dataclass
class Preprocess:
    x_mean_a: np.ndarray
    x_std_a: np.ndarray
    x_mean_w: np.ndarray
    x_std_w: np.ndarray
    y_mean_a: np.ndarray
    y_std_a: np.ndarray
    y_mean_w: np.ndarray
    y_std_w: np.ndarray
    basis_a: np.ndarray
    basis_w: np.ndarray


class HeroVM17(nn.Module):
    def __init__(self, preprocess: Preprocess, config: dict[str, Any]) -> None:
        super().__init__()
        hidden = int(config["hidden"])
        latent = bool(config["latent"])
        rank = int(config.get("rank", 17 if not latent else 6))
        self.latent = latent
        self.rank = rank
        self.adapter_a = nn.Sequential(nn.Linear(DOMAIN_WIDTH, hidden), nn.ReLU())
        self.adapter_w = nn.Sequential(nn.Linear(DOMAIN_WIDTH, hidden), nn.ReLU())
        self.shared = nn.Sequential(
            nn.Linear(hidden + 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        if latent:
            self.coefficient = nn.Linear(hidden, rank)
            self.register_buffer("basis_a", torch.from_numpy(preprocess.basis_a[:rank].astype(np.float32)))
            self.register_buffer("basis_w", torch.from_numpy(preprocess.basis_w[:rank].astype(np.float32)))
        else:
            self.head_a = nn.Linear(hidden, 17)
            self.head_w = nn.Linear(hidden, 17)
        self.register_buffer("y_mean_a", torch.from_numpy(preprocess.y_mean_a.astype(np.float32)))
        self.register_buffer("y_std_a", torch.from_numpy(preprocess.y_std_a.astype(np.float32)))
        self.register_buffer("y_mean_w", torch.from_numpy(preprocess.y_mean_w.astype(np.float32)))
        self.register_buffer("y_std_w", torch.from_numpy(preprocess.y_std_w.astype(np.float32)))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values.reshape(-1, INPUT_WIDTH)
        left = self.adapter_a(values[:, :DOMAIN_WIDTH])
        right = self.adapter_w(values[:, DOMAIN_WIDTH : 2 * DOMAIN_WIDTH])
        domain = values[:, 2 * DOMAIN_WIDTH :]
        hidden = self.shared(torch.cat((left + right, domain), dim=1))
        if self.latent:
            coefficient = self.coefficient(hidden)
            normalized_a = coefficient @ self.basis_a
            normalized_w = coefficient @ self.basis_w
        else:
            normalized_a = self.head_a(hidden)
            normalized_w = self.head_w(hidden)
        output_a = self.y_mean_a + normalized_a * self.y_std_a
        output_w = self.y_mean_w + normalized_w * self.y_std_w
        return domain[:, :1] * output_a + domain[:, 1:2] * output_w


CONFIGS = (
    {"id": "v0_full_output_erm", "latent": False, "rank": 17, "hidden": 64, "group_dro": False, "distill": False},
    {"id": "v1_lowrank3", "latent": True, "rank": 3, "hidden": 64, "group_dro": False, "distill": False},
    {"id": "v2_lowrank6", "latent": True, "rank": 6, "hidden": 64, "group_dro": False, "distill": False},
    {"id": "v3_lowrank6_wide", "latent": True, "rank": 6, "hidden": 128, "group_dro": False, "distill": False},
    {"id": "v4_lowrank6_dro", "latent": True, "rank": 6, "hidden": 96, "group_dro": True, "distill": False},
    {"id": "v5_lowrank6_distill", "latent": True, "rank": 6, "hidden": 96, "group_dro": False, "distill": True},
    {"id": "v6_hero_full", "latent": True, "rank": 6, "hidden": 96, "group_dro": True, "distill": True},
    {"id": "v7_hero_lite", "latent": True, "rank": 6, "hidden": 48, "group_dro": True, "distill": True},
)


def load_data() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    output = {}
    for domain in DOMAINS:
        x = pd.read_csv(SOURCE / f"X_pvd_{domain}.csv").to_numpy(dtype=np.float32)
        y = pd.read_csv(SOURCE / f"Y_pvd_{domain}.csv").to_numpy(dtype=np.float32)
        if x.ndim != 2 or y.shape[1] != 17 or len(x) != len(y):
            raise ValueError(f"invalid PVD source shape for {domain}: {x.shape}, {y.shape}")
        output[domain] = (x, y)
    return output


def fold_indices(length: int, fold: int) -> tuple[np.ndarray, np.ndarray]:
    boundaries = ((0.45, 0.55), (0.55, 0.65), (0.65, 0.75), (0.70, 0.85))
    train_end, val_end = boundaries[fold]
    return np.arange(0, int(length * train_end)), np.arange(int(length * train_end), int(length * val_end))


def final_indices(length: int) -> tuple[np.ndarray, np.ndarray]:
    boundary = int(length * 0.85)
    return np.arange(0, boundary), np.arange(boundary, length)


def fit_preprocess(
    data: dict[str, tuple[np.ndarray, np.ndarray]],
    train_indices: dict[str, np.ndarray],
) -> Preprocess:
    values: dict[str, np.ndarray] = {}
    for domain in DOMAINS:
        x, y = data[domain]
        idx = train_indices[domain]
        x_mean = x[idx].mean(axis=0).astype(np.float32)
        x_std = x[idx].std(axis=0).astype(np.float32)
        # Match the already-evidenced PVD deployment contract: tiny sensor
        # scales are floored and normalized drift tails are clipped.
        x_std = np.maximum(x_std, 1e-3).astype(np.float32)
        y_mean = y[idx].mean(axis=0).astype(np.float32)
        y_std = y[idx].std(axis=0).astype(np.float32)
        y_std[y_std < 1e-6] = 1.0
        normalized = (y[idx] - y_mean) / y_std
        _, _, vh = np.linalg.svd(normalized, full_matrices=False)
        values[f"x_mean_{domain}"] = x_mean
        values[f"x_std_{domain}"] = x_std
        values[f"y_mean_{domain}"] = y_mean
        values[f"y_std_{domain}"] = y_std
        values[f"basis_{domain}"] = vh.astype(np.float32)
    return Preprocess(
        x_mean_a=values["x_mean_AlCu"], x_std_a=values["x_std_AlCu"],
        x_mean_w=values["x_mean_WTi"], x_std_w=values["x_std_WTi"],
        y_mean_a=values["y_mean_AlCu"], y_std_a=values["y_std_AlCu"],
        y_mean_w=values["y_mean_WTi"], y_std_w=values["y_std_WTi"],
        basis_a=values["basis_AlCu"], basis_w=values["basis_WTi"],
    )


def encode_domain(x: np.ndarray, domain: str, preprocess: Preprocess) -> np.ndarray:
    output = np.zeros((len(x), INPUT_WIDTH), dtype=np.float32)
    if domain == "AlCu":
        normalized = np.clip((x - preprocess.x_mean_a) / preprocess.x_std_a, -8.0, 8.0)
        output[:, : x.shape[1]] = normalized
        output[:, 216] = 1.0
    else:
        normalized = np.clip((x - preprocess.x_mean_w) / preprocess.x_std_w, -8.0, 8.0)
        output[:, 108 : 108 + x.shape[1]] = normalized
        output[:, 217] = 1.0
    return output


def assemble(
    data: dict[str, tuple[np.ndarray, np.ndarray]],
    indices: dict[str, np.ndarray],
    preprocess: Preprocess,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, domains, groups = [], [], [], []
    for domain_index, domain in enumerate(DOMAINS):
        x, y = data[domain]
        idx = indices[domain]
        xs.append(encode_domain(x[idx], domain, preprocess))
        ys.append(y[idx])
        domains.append(np.full(len(idx), domain_index, dtype=np.int64))
        denominator = max(int(idx.max()) + 1 if len(idx) else 1, 1)
        quartile = np.minimum((idx.astype(np.float64) / denominator * 4).astype(np.int64), 3)
        groups.append(domain_index * 4 + quartile)
    return (
        np.concatenate(xs).astype(np.float32),
        np.concatenate(ys).astype(np.float32),
        np.concatenate(domains),
        np.concatenate(groups),
    )


def target_scale(preprocess: Preprocess, domain: torch.Tensor, device: torch.device) -> torch.Tensor:
    left = torch.from_numpy(preprocess.y_std_a).to(device)
    right = torch.from_numpy(preprocess.y_std_w).to(device)
    return torch.where(domain[:, None] == 0, left[None], right[None])


@torch.inference_mode()
def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(x), 2048):
        outputs.append(model(torch.from_numpy(x[start : start + 2048]).to(device)).cpu().numpy())
    return np.concatenate(outputs)


def metrics(y: np.ndarray, prediction: np.ndarray, domains: np.ndarray) -> dict[str, Any]:
    result = {"combined": {"mae": mae(y, prediction), "rmse": rmse(y, prediction), "r2": r2(y, prediction)}}
    per_domain = {}
    for index, domain in enumerate(DOMAINS):
        mask = domains == index
        per_domain[domain] = {
            "mae": mae(y[mask], prediction[mask]),
            "rmse": rmse(y[mask], prediction[mask]),
            "r2": r2(y[mask], prediction[mask]),
            "worst_point_mae": float(np.max(np.mean(np.abs(y[mask] - prediction[mask]), axis=0))),
            "uniformity_mae": mae(np.std(y[mask], axis=1), np.std(prediction[mask], axis=1)),
        }
    result["per_domain"] = per_domain
    result["worst_domain_mae"] = max(value["mae"] for value in per_domain.values())
    result["selection_score"] = result["combined"]["mae"] + 0.5 * result["worst_domain_mae"]
    return result


def fit_ridge(
    fit_id: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    val_domains: np.ndarray,
) -> tuple[Ridge, np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    teacher = Ridge(alpha=100.0)
    teacher.fit(x_train, y_train)
    prediction = teacher.predict(x_val).astype(np.float32)
    result_metrics = metrics(y_val, prediction, val_domains)
    save_fit(
        FitResult(
            family_id=FAMILY_ID,
            fit_id=fit_id,
            configuration={"kind": "ridge_teacher", "alpha": 100.0},
            seed=SEED_BASE,
            selection_metrics=result_metrics,
            state_dict=None,
            fit_seconds=time.perf_counter() - started,
            parameter_count=int(teacher.coef_.size + teacher.intercept_.size),
            extra={"teacher_only": True},
        ),
        keep_checkpoint=False,
    )
    return teacher, prediction, result_metrics


def train_torch(
    fit_id: str,
    config: dict[str, Any],
    seed: int,
    preprocess: Preprocess,
    train: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    teacher_train: np.ndarray,
    teacher_val: np.ndarray,
    device: torch.device,
    thermal: ThermalGuard,
) -> tuple[HeroVM17, dict[str, Any], dict[str, Any]]:
    thermal.check()
    set_seed(seed)
    x_train, y_train, d_train, g_train = train
    x_val, y_val, d_val, _ = val
    model = HeroVM17(preprocess, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    tensors = TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(d_train),
        torch.from_numpy(g_train), torch.from_numpy(teacher_train.astype(np.float32)),
    )
    loader = DataLoader(tensors, batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(seed))
    q = torch.ones(8, device=device) / 8.0
    best_state = None
    best_score = float("inf")
    history = []
    stale = 0
    started = time.perf_counter()
    for epoch in range(24):
        model.train()
        epoch_losses = []
        for xb, yb, db, gb, tb in loader:
            xb, yb, db, gb, tb = xb.to(device), yb.to(device), db.to(device), gb.to(device), tb.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(xb)
            scale = target_scale(preprocess, db, device)
            per_sample = F.smooth_l1_loss(output / scale, yb / scale, reduction="none").mean(dim=1)
            if config["group_dro"]:
                group_loss = []
                observed = []
                for group in range(8):
                    mask = gb == group
                    if bool(mask.any()):
                        group_loss.append(per_sample[mask].mean())
                        observed.append(group)
                stacked = torch.stack(group_loss)
                with torch.no_grad():
                    indices = torch.tensor(observed, device=device)
                    q[indices] *= torch.exp(0.08 * stacked.detach())
                    q /= q.sum()
                loss = torch.sum(q[torch.tensor(observed, device=device)] * stacked)
            else:
                loss = per_sample.mean()
            if config["distill"]:
                loss = loss + 0.15 * F.smooth_l1_loss(output / scale, tb / scale)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        val_prediction = predict(model, x_val, device)
        val_metrics = metrics(y_val, val_prediction, d_val)
        score = float(val_metrics["selection_score"])
        history.append({"epoch": epoch + 1, "loss": float(np.mean(epoch_losses)), "selection_score": score})
        if score < best_score - 1e-6:
            best_score = score
            best_state = state_to_cpu(model)
            stale = 0
        else:
            stale += 1
        if stale >= 5 and epoch >= 7:
            break
    if best_state is None:
        raise RuntimeError(f"no HERO checkpoint for {fit_id}")
    model.load_state_dict(best_state)
    model.to(device)
    final_prediction = predict(model, x_val, device)
    final_metrics = metrics(y_val, final_prediction, d_val)
    result = FitResult(
        family_id=FAMILY_ID,
        fit_id=fit_id,
        configuration=config,
        seed=seed,
        selection_metrics=final_metrics,
        state_dict=best_state,
        fit_seconds=time.perf_counter() - started,
        parameter_count=parameter_count(model),
        extra={"epochs": len(history), "history": history, "teacher_distillation": bool(config["distill"])},
    )
    receipt = save_fit(result)
    return model, final_metrics, receipt


def preprocessing_arrays(preprocess: Preprocess) -> dict[str, np.ndarray]:
    return {name: np.asarray(value) for name, value in vars(preprocess).items()}


def main() -> int:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    thermal = ThermalGuard()
    data = load_data()
    task_contract = {
        "schema": "x5_icmat_foundry.post50_task_contract.v1",
        "family_id": FAMILY_ID,
        "task": "predict 17 PVD thickness measurements from per-domain sensor vectors",
        "domains": {domain: {"rows": len(data[domain][0]), "input_width": data[domain][0].shape[1]} for domain in DOMAINS},
        "source_license": "CC-BY-4.0",
        "source_doi": "10.5281/zenodo.16881338",
        "split": "four ordered-row proxy folds inside first 85%; final 15% confirmation",
        "truth_boundary": [
            "row order is a drift proxy, not recorded equipment time",
            "17 outputs are structured measurements; measurement coordinates are unavailable",
            "no fab production accuracy claim",
        ],
        "model_output": "17 finite thickness values only; no PASS/HOLD/routing/decision output",
    }
    write_json(CONTRACT_ROOT / "tasks/RND-PROC-01.json", task_contract, seal=True)
    fold_receipts: list[dict[str, Any]] = []
    config_scores: dict[str, list[float]] = {config["id"]: [] for config in CONFIGS}
    for fold in range(4):
        train_idx, val_idx = {}, {}
        for domain in DOMAINS:
            train_idx[domain], val_idx[domain] = fold_indices(len(data[domain][0]), fold)
        preprocess = fit_preprocess(data, train_idx)
        train = assemble(data, train_idx, preprocess)
        val = assemble(data, val_idx, preprocess)
        teacher, _, _ = fit_ridge(
            f"teacher_fold{fold}", train[0], train[1], val[0], val[1], val[2]
        )
        teacher_train = teacher.predict(train[0]).astype(np.float32)
        teacher_val = teacher.predict(val[0]).astype(np.float32)
        for config_index, config in enumerate(CONFIGS):
            seed = SEED_BASE + fold * 100 + config_index
            _, selection, receipt = train_torch(
                f"fold{fold}_{config['id']}", config, seed, preprocess, train, val,
                teacher_train, teacher_val, device, thermal,
            )
            config_scores[config["id"]].append(float(selection["selection_score"]))
            fold_receipts.append(receipt)
    ranked = sorted(CONFIGS, key=lambda config: float(np.mean(config_scores[config["id"]])))
    selected = copy.deepcopy(ranked[0])
    final_train_idx, final_val_idx, test_idx = {}, {}, {}
    for domain in DOMAINS:
        length = len(data[domain][0])
        final_train_idx[domain] = np.arange(0, int(length * 0.75))
        final_val_idx[domain] = np.arange(int(length * 0.75), int(length * 0.85))
        test_idx[domain] = np.arange(int(length * 0.85), length)
    final_preprocess = fit_preprocess(data, final_train_idx)
    final_train = assemble(data, final_train_idx, final_preprocess)
    final_val = assemble(data, final_val_idx, final_preprocess)
    final_test = assemble(data, test_idx, final_preprocess)
    final_teacher, _, teacher_val_metrics = fit_ridge(
        "teacher_final",
        final_train[0], final_train[1], final_val[0], final_val[1], final_val[2],
    )
    teacher_train = final_teacher.predict(final_train[0]).astype(np.float32)
    teacher_val = final_teacher.predict(final_val[0]).astype(np.float32)
    variants = [
        ("quality", selected, SEED_BASE + 900),
        ("robust", next(config for config in CONFIGS if config["id"] == "v6_hero_full"), SEED_BASE + 901),
        ("lite", next(config for config in CONFIGS if config["id"] == "v7_hero_lite"), SEED_BASE + 902),
    ]
    final_models: dict[str, HeroVM17] = {}
    final_receipts = []
    for variant, config, seed in variants:
        model, _, receipt = train_torch(
            f"final_{variant}_{config['id']}", config, seed, final_preprocess,
            final_train, final_val, teacher_train, teacher_val, device, thermal,
        )
        final_models[variant] = model.cpu().eval()
        final_receipts.append(receipt)
    selection_lock = {
        "schema": "x5_icmat_foundry.post50_selection_lock.v1",
        "created_at": utc_now(),
        "family_id": FAMILY_ID,
        "fit_count_before_test": 40,
        "ranked_configuration_ids": [config["id"] for config in ranked],
        "mean_fold_scores": {key: float(np.mean(value)) for key, value in config_scores.items()},
        "variants": [{"variant": name, "configuration": config["id"], "seed": seed} for name, config, seed in variants],
        "test_opened": False,
        "post_test_retuning_allowed": False,
    }
    write_json(CONTRACT_ROOT / "selection_locks/RND-PROC-01.json", selection_lock, seal=True)

    # Locked confirmation begins here. No configuration changes are permitted after this point.
    mean_baseline = np.zeros_like(final_test[1])
    for domain_index, domain in enumerate(DOMAINS):
        mask = final_test[2] == domain_index
        source_mean = data[domain][1][final_train_idx[domain]].mean(axis=0)
        mean_baseline[mask] = source_mean
    baseline_metrics = metrics(final_test[1], mean_baseline, final_test[2])
    ridge_prediction = final_teacher.predict(final_test[0]).astype(np.float32)
    ridge_metrics = metrics(final_test[1], ridge_prediction, final_test[2])
    locked_metrics = {}
    exports = {}
    for variant, model in final_models.items():
        prediction = predict(model.to(device), final_test[0], device)
        locked_metrics[variant] = metrics(final_test[1], prediction, final_test[2])
        sample = torch.from_numpy(final_test[0][:1].reshape(1, 1, 1, INPUT_WIDTH))
        exports[variant] = export_static_onnx(
            FAMILY_ID, variant, model.cpu(), [sample], ["pvd_features_fp32"], ["thickness17"],
            preprocessing_arrays(final_preprocess),
            {"configuration": next(item[1] for item in variants if item[0] == variant), "seed": next(item[2] for item in variants if item[0] == variant)},
        )
    quality = locked_metrics["quality"]
    innovation_gate = {
        "combined_beats_domain_mean": quality["combined"]["mae"] < baseline_metrics["combined"]["mae"],
        "alcu_not_worse_than_domain_mean": quality["per_domain"]["AlCu"]["mae"] <= baseline_metrics["per_domain"]["AlCu"]["mae"],
        "wti_not_worse_than_domain_mean": quality["per_domain"]["WTi"]["mae"] <= baseline_metrics["per_domain"]["WTi"]["mae"],
        "relative_improvement_ge_5pct": quality["combined"]["mae"] <= 0.95 * baseline_metrics["combined"]["mae"],
    }
    innovation_gate["pass"] = all(innovation_gate.values())
    per_sample_quality = np.mean(np.abs(final_test[1] - predict(final_models["quality"].to(device), final_test[0], device)), axis=1)
    receipt = write_family_receipt(
        FAMILY_ID,
        {
            "model_name": "HERO-VM17",
            "fit_count": 40,
            "fit_count_contract_pass": True,
            "task_contract": task_contract,
            "selection_lock": selection_lock,
            "locked_test": {
                "rows": len(final_test[0]),
                "domain_rows": {domain: int(np.sum(final_test[2] == index)) for index, domain in enumerate(DOMAINS)},
                "domain_mean_baseline": baseline_metrics,
                "ridge_baseline": ridge_metrics,
                "variants": locked_metrics,
                "quality_per_sample_mae_bootstrap95": bootstrap_ci(per_sample_quality),
            },
            "innovation_gate": innovation_gate,
            "candidate_class": "RND_INNOVATION_ANCHOR" if innovation_gate["pass"] else "RND_USABLE_EXPERIMENTAL",
            "exports": exports,
            "thermal_samples": thermal.samples,
            "claim_boundary": [
                "public PVD virtual-metrology proxy",
                "ordered rows are a drift proxy, not timestamps",
                "PC metrics are not X5 performance",
                "model has no risk gate, audit role, routing, or execution authority",
            ],
        },
    )
    print(json.dumps({"family": FAMILY_ID, "fit_count": 40, "innovation_gate": innovation_gate, "receipt": receipt["path"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
