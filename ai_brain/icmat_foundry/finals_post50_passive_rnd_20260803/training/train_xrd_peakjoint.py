from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from icmat_foundry.finals_50model.training.train_xrd_bank import degrade_spectrum, profile_metrics
from icmat_foundry.finals_post50_passive_rnd_20260803.training.common import (
    CONTRACT_ROOT,
    FitResult,
    ROOT as COMMON_ROOT,
    SEED_BASE,
    TRIAL_ROOT,
    ThermalGuard,
    export_static_onnx,
    parameter_count,
    save_fit,
    set_seed,
    state_to_cpu,
    utc_now,
    write_family_receipt,
    write_json,
)


FAMILY_ID = "RND-XRD-01"
DATASET = ROOT / "icmat_foundry/finals_50model/artifacts/xrd_bank/theoretical_xrd_dataset.v1.npz"


class Residual1D(nn.Module):
    def __init__(self, width: int, dilation: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(width, width, (1, 3), 1, (0, dilation), dilation=(1, dilation)),
            nn.ReLU(),
            nn.Conv2d(width, width, (1, 3), 1, (0, dilation), dilation=(1, dilation)),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.relu(value + self.block(value))


class XRDPeakJoint(nn.Module):
    def __init__(self, width: int = 20) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, width, (1, 7), 1, (0, 3)), nn.ReLU())
        self.encoder = nn.Sequential(Residual1D(width, 1), Residual1D(width, 2), Residual1D(width, 4))
        self.reconstruction = nn.Sequential(
            nn.Conv2d(width, width, (1, 5), 1, (0, 2)), nn.ReLU(), nn.Conv2d(width, 1, 1)
        )
        self.peaks = nn.Sequential(nn.Conv2d(width, width, (1, 3), 1, (0, 1)), nn.ReLU(), nn.Conv2d(width, 1, 1))
        self.pool = nn.AdaptiveAvgPool2d((1, 8))
        self.embedding = nn.Sequential(nn.Flatten(), nn.Linear(width * 8, 64), nn.ReLU(), nn.Linear(64, 32))

    def forward(self, spectrum: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spectrum = spectrum.reshape(-1, 1, 1, 512)
        features = self.encoder(self.stem(spectrum))
        restored = F.relu(spectrum + 0.20 * self.reconstruction(features))
        peak_logits = self.peaks(features)
        embedding = F.normalize(self.embedding(self.pool(features)), p=2, dim=1)
        return restored, peak_logits, embedding


CONFIGS = (
    {"id": "v0_reconstruction", "width": 16, "w_peak": 0.0, "w_diff": 0.0, "w_contrast": 0.0},
    {"id": "v1_peak", "width": 16, "w_peak": 0.15, "w_diff": 0.0, "w_contrast": 0.0},
    {"id": "v2_derivative", "width": 16, "w_peak": 0.0, "w_diff": 0.25, "w_contrast": 0.0},
    {"id": "v3_contrastive", "width": 16, "w_peak": 0.0, "w_diff": 0.0, "w_contrast": 0.10},
    {"id": "v4_peak_contrast", "width": 20, "w_peak": 0.12, "w_diff": 0.0, "w_contrast": 0.08},
    {"id": "v5_full", "width": 20, "w_peak": 0.12, "w_diff": 0.20, "w_contrast": 0.08},
    {"id": "v6_full_wide", "width": 28, "w_peak": 0.12, "w_diff": 0.20, "w_contrast": 0.08},
    {"id": "v7_lite", "width": 10, "w_peak": 0.12, "w_diff": 0.20, "w_contrast": 0.08},
)


def load_bank() -> dict[str, np.ndarray]:
    with np.load(DATASET, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def peak_targets(clean: np.ndarray) -> np.ndarray:
    left = clean[:, :-2]
    center = clean[:, 1:-1]
    right = clean[:, 2:]
    peaks = (center > left) & (center >= right) & (center >= 0.08)
    output = np.zeros_like(clean, dtype=np.float32)
    output[:, 1:-1] = peaks.astype(np.float32)
    # One-bin dilation provides a trainable soft neighborhood while exact F1
    # remains evaluated with the original peak extractor.
    output[:, 1:] = np.maximum(output[:, 1:], 0.45 * output[:, :-1])
    output[:, :-1] = np.maximum(output[:, :-1], 0.45 * output[:, 1:])
    return output


def build_degraded(clean: np.ndarray, tag: str) -> np.ndarray:
    values = np.empty_like(clean, dtype=np.float32)
    for index, spectrum in enumerate(clean):
        seed = int(hashlib.sha256(f"{tag}|{index}".encode()).hexdigest()[:8], 16)
        values[index] = degrade_spectrum(spectrum.astype(np.float32), seed)
    return values


def contrastive(left: torch.Tensor, right: torch.Tensor, temperature: float = 0.08) -> torch.Tensor:
    logits = left @ right.T / temperature
    labels = torch.arange(len(left), device=left.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


@torch.inference_mode()
def infer(model: nn.Module, values: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    restored, peaks, embeddings = [], [], []
    for start in range(0, len(values), 256):
        tensor = torch.from_numpy(values[start : start + 256].reshape(-1, 1, 1, 512)).to(device)
        output = model(tensor)
        restored.append(output[0][:, 0, 0].cpu().numpy())
        peaks.append(torch.sigmoid(output[1][:, 0, 0]).cpu().numpy())
        embeddings.append(output[2].cpu().numpy())
    return np.concatenate(restored), np.concatenate(peaks), np.concatenate(embeddings)


def retrieval_metrics(query: np.ndarray, gallery: np.ndarray) -> dict[str, float]:
    similarity = query @ gallery.T
    order = np.argsort(-similarity, axis=1)
    truth = np.arange(len(query))[:, None]
    ranks = np.argmax(order == truth, axis=1) + 1
    return {
        "top1": float(np.mean(ranks <= 1)),
        "top5": float(np.mean(ranks <= 5)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
    }


def evaluate(
    model: nn.Module,
    degraded: np.ndarray,
    clean: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    restored, _, query_embedding = infer(model, degraded, device)
    _, _, gallery_embedding = infer(model, clean, device)
    profile = profile_metrics(clean, restored)
    retrieval = retrieval_metrics(query_embedding, gallery_embedding)
    score = profile["mae"] + 0.05 * (1.0 - profile["peak_f1"]) + 0.02 * (1.0 - retrieval["top1"])
    return {"profile": profile, "retrieval": retrieval, "selection_score": float(score)}


def train_fit(
    fit_id: str,
    config: dict[str, Any],
    seed: int,
    train_degraded: np.ndarray,
    train_clean: np.ndarray,
    train_peaks: np.ndarray,
    tune_degraded: np.ndarray,
    tune_clean: np.ndarray,
    device: torch.device,
    thermal: ThermalGuard,
) -> tuple[XRDPeakJoint, dict[str, Any], dict[str, Any]]:
    thermal.check()
    set_seed(seed)
    model = XRDPeakJoint(int(config["width"])).to(device)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_degraded.reshape(-1, 1, 1, 512)),
            torch.from_numpy(train_clean.reshape(-1, 1, 1, 512)),
            torch.from_numpy(train_peaks.reshape(-1, 1, 1, 512)),
        ),
        batch_size=512,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    best_state = None
    best_score = float("inf")
    history = []
    stale = 0
    started = time.perf_counter()
    for epoch in range(6):
        model.train()
        losses = []
        for degraded, clean, peaks in loader:
            degraded, clean, peaks = degraded.to(device), clean.to(device), peaks.to(device)
            optimizer.zero_grad(set_to_none=True)
            restored, peak_logits, query_embedding = model(degraded)
            _, _, clean_embedding = model(clean)
            reconstruction = F.smooth_l1_loss(restored, clean)
            derivative = F.l1_loss(restored[..., 1:] - restored[..., :-1], clean[..., 1:] - clean[..., :-1])
            peak_loss = F.binary_cross_entropy_with_logits(peak_logits, peaks, pos_weight=torch.tensor(8.0, device=device))
            contrastive_loss = contrastive(query_embedding, clean_embedding)
            loss = reconstruction + float(config["w_diff"]) * derivative + float(config["w_peak"]) * peak_loss + float(config["w_contrast"]) * contrastive_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        selection = evaluate(model, tune_degraded, tune_clean, device)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "selection_score": selection["selection_score"]})
        if selection["selection_score"] < best_score - 1e-5:
            best_score = selection["selection_score"]
            best_state = state_to_cpu(model)
            stale = 0
        else:
            stale += 1
        if stale >= 2 and epoch >= 3:
            break
    if best_state is None:
        raise RuntimeError(f"no XRD checkpoint for {fit_id}")
    model.load_state_dict(best_state)
    model.to(device)
    selection = evaluate(model, tune_degraded, tune_clean, device)
    receipt = save_fit(
        FitResult(
            family_id=FAMILY_ID,
            fit_id=fit_id,
            configuration=config,
            seed=seed,
            selection_metrics=selection,
            state_dict=best_state,
            fit_seconds=time.perf_counter() - started,
            parameter_count=parameter_count(model),
            extra={"epochs": len(history), "history": history, "test_split_observed": False},
        )
    )
    return model, selection, receipt


def main() -> int:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    thermal = ThermalGuard()
    bank = load_bank()
    clean = bank["spectra"].astype(np.float32)
    train_clean = clean[bank["split_codes"] == 0]
    tune_clean = clean[bank["split_codes"] == 1]
    test_clean = clean[bank["split_codes"] == 2]
    train_views = [build_degraded(train_clean, f"train-view-{index}") for index in range(4)]
    tune_degraded = build_degraded(tune_clean, "locked-tune")
    test_degraded = build_degraded(test_clean, "locked-test")
    train_peaks = peak_targets(train_clean)
    task_contract = {
        "schema": "x5_icmat_foundry.post50_task_contract.v1",
        "family_id": FAMILY_ID,
        "task": "joint XRD profile restoration, peak heatmap, and exact-phase embedding",
        "source": "JARVIS-DFT theoretical XRD plus deterministic synthetic degradation",
        "sample_counts": {"train": len(train_clean), "tune": len(tune_clean), "test": len(test_clean)},
        "split": "formula and approximate structure-family disjoint existing split",
        "model_outputs": ["restored_profile", "peak_logits", "phase_embedding"],
        "forbidden_outputs": ["PASS", "HOLD", "RETAKE", "model_route", "robot_action"],
        "claim_boundary": "not experimental XRD and not instrument accuracy",
    }
    write_json(CONTRACT_ROOT / "tasks/RND-XRD-01.json", task_contract, seal=True)
    config_scores: dict[str, list[float]] = {config["id"]: [] for config in CONFIGS}

    def existing_score(fit_id: str) -> float | None:
        path = TRIAL_ROOT / FAMILY_ID / f"{fit_id}.json"
        if not path.is_file():
            return None
        receipt = json.loads(path.read_text(encoding="utf-8"))
        return float(receipt["selection_metrics"]["selection_score"])

    for config_index, config in enumerate(CONFIGS):
        for repeat in range(3):
            fit_id = f"grid_{config['id']}_r{repeat}"
            resumed = existing_score(fit_id)
            if resumed is not None:
                config_scores[config["id"]].append(resumed)
                continue
            _, selection, _ = train_fit(
                fit_id, config, SEED_BASE + 5000 + config_index * 10 + repeat,
                train_views[repeat], train_clean, train_peaks, tune_degraded, tune_clean,
                device, thermal,
            )
            config_scores[config["id"]].append(float(selection["selection_score"]))
    ranked = sorted(CONFIGS, key=lambda config: float(np.mean(config_scores[config["id"]])))
    for rank, config in enumerate(ranked[:2]):
        for repeat in range(2):
            _, selection, _ = train_fit(
                f"stability_top{rank}_{config['id']}_r{repeat}", config,
                SEED_BASE + 6000 + rank * 10 + repeat,
                train_views[2 + repeat], train_clean, train_peaks, tune_degraded, tune_clean,
                device, thermal,
            )
            config_scores[config["id"]].append(float(selection["selection_score"]))
    ranked = sorted(CONFIGS, key=lambda config: float(np.mean(config_scores[config["id"]])))
    variants = [
        ("quality", copy.deepcopy(ranked[0]), SEED_BASE + 6900),
        ("robust", next(config for config in CONFIGS if config["id"] == "v5_full"), SEED_BASE + 6901),
        ("lite", next(config for config in CONFIGS if config["id"] == "v7_lite"), SEED_BASE + 6902),
        ("control", next(config for config in CONFIGS if config["id"] == "v0_reconstruction"), SEED_BASE + 6903),
    ]
    models = {}
    for index, (variant, config, seed) in enumerate(variants):
        model, _, _ = train_fit(
            f"final_{variant}_{config['id']}", config, seed,
            train_views[index], train_clean, train_peaks, tune_degraded, tune_clean,
            device, thermal,
        )
        models[variant] = model.cpu().eval()
    selection_lock = {
        "schema": "x5_icmat_foundry.post50_selection_lock.v1",
        "created_at": utc_now(),
        "family_id": FAMILY_ID,
        "fit_count_before_test": 32,
        "ranked_configuration_ids": [config["id"] for config in ranked],
        "mean_tune_scores": {key: float(np.mean(values)) for key, values in config_scores.items()},
        "variants": [{"variant": name, "configuration": config["id"], "seed": seed} for name, config, seed in variants],
        "test_opened": False,
        "post_test_retuning_allowed": False,
    }
    write_json(CONTRACT_ROOT / "selection_locks/RND-XRD-01.json", selection_lock, seal=True)

    locked_metrics, exports = {}, {}
    for variant, model in models.items():
        locked_metrics[variant] = evaluate(model.to(device), test_degraded, test_clean, device)
        if variant != "control":
            exports[variant] = export_static_onnx(
                FAMILY_ID, variant, model.cpu(),
                [torch.from_numpy(test_degraded[:1].reshape(1, 1, 1, 512))],
                ["xrd_degraded_fp32"], ["restored_profile", "peak_logits", "phase_embedding"],
                None,
                {"configuration": next(item[1] for item in variants if item[0] == variant), "seed": next(item[2] for item in variants if item[0] == variant)},
            )
    quality = locked_metrics["quality"]
    gate = {
        "peak_f1_gt_legacy": quality["profile"]["peak_f1"] > 0.6733753409197781,
        "mae_within_2pct_legacy": quality["profile"]["mae"] <= 1.02 * 0.022970125079154968,
        "top1_within_one_point_legacy": quality["retrieval"]["top1"] >= 0.938,
    }
    gate["pass"] = all(gate.values())
    receipt = write_family_receipt(
        FAMILY_ID,
        {
            "model_name": "XRD-PeakJoint",
            "fit_count": 32,
            "fit_count_contract_pass": True,
            "task_contract": task_contract,
            "selection_lock": selection_lock,
            "locked_test": locked_metrics,
            "innovation_gate": gate,
            "candidate_class": "RND_INNOVATION_ANCHOR" if gate["pass"] else "RND_USABLE_EXPERIMENTAL",
            "exports": exports,
            "thermal_samples": thermal.samples,
            "claim_boundary": [
                "JARVIS theoretical XRD only",
                "all query degradation is deterministic synthetic physical degradation",
                "PC ONNX is not actual X5 BPU evidence",
                "no quality gate, routing, audit, or execution authority",
            ],
        },
    )
    print(json.dumps({"family": FAMILY_ID, "fit_count": 32, "gate": gate, "receipt": receipt["path"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
