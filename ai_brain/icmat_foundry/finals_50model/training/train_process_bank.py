#!/usr/bin/env python3
"""Train the six fast-track semiconductor process models on local public data."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.linear_model import Ridge
from torch import nn


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from icmat_foundry.fabyield.data import (  # noqa: E402
    load_secom_zip,
    temporal_batch_split,
    validate_temporal_split,
)
from icmat_foundry.fabyield.preprocessing import LeakageSafePreprocessor  # noqa: E402


SEED = 20260801
SECOM_SOURCE = (
    ROOT
    / "research"
    / "data_assets"
    / "icmat_foundry"
    / "uci_secom"
    / "raw"
    / "secom.zip"
)
PVD_SOURCE = (
    ROOT
    / "research"
    / "icmat_foundry"
    / "fabyield_replacement_20260728"
    / "candidates"
    / "zenodo_16881338"
)
ARTIFACT_ROOT = ROOT / "icmat_foundry" / "finals_50model" / "artifacts" / "process_bank"
EVIDENCE_ROOT = ROOT / "icmat_foundry" / "finals_50model" / "evidence" / "process_bank"


MODEL_NAMES = {
    "F-PROC-01": "FabYield-X5-v2",
    "F-PROC-02": "SECOM-MaskedImputer-X5",
    "F-PROC-03": "PVD-Thickness17-X5",
    "F-PROC-07": "SECOM-TemporalDrift-CPU",
    "F-PROC-08": "PVD-ProcessDomainOOD-CPU",
    "F-PROC-09": "SECOM-ProcessAnomaly-CPU",
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_receipt(model_id: str, receipt: dict[str, Any]) -> Path:
    path = EVIDENCE_ROOT / model_id / "receipt.json"
    write_json(path, receipt)
    digest = sha256_file(path)
    path.with_suffix(".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StaticConvMLP(nn.Module):
    """BPU-friendly MLP expressed as static Conv2d layers."""

    def __init__(self, in_features: int, hidden: tuple[int, ...], out_features: int):
        super().__init__()
        layers: list[nn.Module] = []
        current = 1
        for index, width in enumerate(hidden):
            kernel = (1, in_features) if index == 0 else (1, 1)
            layers.extend((nn.Conv2d(current, width, kernel_size=kernel), nn.ReLU()))
            current = width
        layers.append(nn.Conv2d(current, out_features, kernel_size=1))
        self.network = nn.Sequential(*layers)
        self.in_features = int(in_features)
        self.hidden = tuple(int(value) for value in hidden)
        self.out_features = int(out_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).flatten(1)


class StaticLinear(nn.Module):
    """One trainable Linear layer exported as a static ONNX Gemm."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.in_features = int(in_features)
        self.out_features = int(out_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs.flatten(1))


@dataclass
class TrainResult:
    model: nn.Module
    best_epoch: int
    epochs_run: int
    best_validation_loss: float
    elapsed_seconds: float


def as_model_input(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values[:, None, None, :]


def train_model(
    model: StaticConvMLP,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    device: torch.device,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    epochs: int,
    patience: int,
    learning_rate: float = 2e-3,
    batch_size: int = 256,
    validation_score: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> TrainResult:
    set_seed()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    tensor_x = torch.from_numpy(as_model_input(train_x))
    tensor_y = torch.from_numpy(np.asarray(train_y, dtype=np.float32))
    validation_tensor_x = torch.from_numpy(as_model_input(validation_x)).to(device)
    validation_tensor_y = torch.from_numpy(
        np.asarray(validation_y, dtype=np.float32)
    ).to(device)
    generator = torch.Generator().manual_seed(SEED)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = math.inf
    best_score = -math.inf
    best_epoch = 0
    stale = 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(tensor_x.shape[0], generator=generator)
        for start in range(0, tensor_x.shape[0], batch_size):
            batch = order[start : start + batch_size]
            inputs = tensor_x[batch].to(device)
            targets = tensor_y[batch].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(inputs), targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_output = model(validation_tensor_x)
            validation_loss = float(
                loss_function(validation_output, validation_tensor_y).detach().cpu()
            )
            if validation_score is None:
                score = -validation_loss
            else:
                score = float(
                    validation_score(
                        validation_tensor_y.detach().cpu().numpy(),
                        validation_output.detach().cpu().numpy(),
                    )
                )
        improved = score > best_score + 1e-7
        if improved:
            best_score = score
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return TrainResult(
        model=model,
        best_epoch=best_epoch,
        epochs_run=epoch,
        best_validation_loss=best_loss,
        elapsed_seconds=time.perf_counter() - started,
    )


def predict(model: nn.Module, values: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(as_model_input(values)).to(device))
    return output.detach().cpu().numpy()


def export_bundle(
    model_id: str,
    result: TrainResult,
    sample: np.ndarray,
    architecture: dict[str, Any],
) -> dict[str, Any]:
    output = ARTIFACT_ROOT / model_id
    output.mkdir(parents=True, exist_ok=True)
    model = result.model.cpu().eval()
    pt_path = output / "model.pt"
    torch.save(
        {
            "schema": "x5_icmat_foundry.process_model.v1",
            "inventory_id": model_id,
            "model_name": MODEL_NAMES[model_id],
            "seed": SEED,
            "architecture": architecture,
            "state_dict": model.state_dict(),
        },
        pt_path,
    )

    sample_input = as_model_input(np.asarray(sample[:1], dtype=np.float32))
    onnx_path = output / "model.onnx"
    torch.onnx.export(
        model,
        torch.from_numpy(sample_input),
        onnx_path,
        export_params=True,
        do_constant_folding=True,
        input_names=["features_fp32"],
        output_names=["model_output_fp32"],
        opset_version=11,
        dynamic_axes=None,
    )
    graph = onnx.load(onnx_path)
    graph.ir_version = 7
    onnx.checker.check_model(graph)
    onnx.save(graph, onnx_path)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_output = session.run(None, {"features_fp32": sample_input})[0]
    with torch.no_grad():
        torch_output = model(torch.from_numpy(sample_input)).numpy()
    parity_max_abs = float(np.max(np.abs(ort_output - torch_output)))
    if not np.isfinite(ort_output).all() or parity_max_abs > 1e-5:
        raise RuntimeError(f"{model_id} ONNX Runtime parity failed: {parity_max_abs}")

    input_path = output / "ort_sample_input.npy"
    output_path = output / "ort_sample_output.npy"
    np.save(input_path, sample_input)
    np.save(output_path, ort_output)
    artifacts = []
    for path in sorted(output.iterdir()):
        if path.is_file():
            artifacts.append(
                {
                    "path": relative(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "artifacts": artifacts,
        "onnx": {
            "opset": 11,
            "ir_version": 7,
            "static_input_shape": list(sample_input.shape),
            "static_output_shape": list(ort_output.shape),
            "onnx_checker": "PASS",
            "ort_sample": "PASS",
            "parity_max_abs": parity_max_abs,
        },
    }


def save_npz(path: Path, **arrays: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {
        "path": relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def threshold_from_validation(labels: np.ndarray, probability: np.ndarray) -> float:
    best = (0.5, -math.inf)
    for threshold in np.linspace(0.05, 0.95, 181):
        score = balanced_accuracy_score(labels, probability >= threshold)
        if score > best[1]:
            best = (float(threshold), float(score))
    return best[0]


def binary_metrics(labels: np.ndarray, logits: np.ndarray, threshold: float) -> dict[str, float]:
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits.reshape(-1), -30, 30)))
    prediction = probability >= threshold
    return {
        "pr_auc": float(average_precision_score(labels, probability)),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "balanced_error_rate": float(1.0 - balanced_accuracy_score(labels, prediction)),
        "positive_prevalence": float(np.mean(labels)),
        "threshold_from_validation": float(threshold),
    }


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(target, prediction))),
        "r2": float(r2_score(target, prediction, multioutput="variance_weighted")),
    }


def train_fproc01(secom: Any, split: Any, device: torch.device) -> dict[str, Any]:
    model_id = "F-PROC-01"
    artifact_dir = ARTIFACT_ROOT / model_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    preprocessor = LeakageSafePreprocessor(max_features=128).fit(
        secom.features[split.train], secom.labels[split.train]
    )
    x_train = preprocessor.transform(secom.features[split.train])
    x_validation = preprocessor.transform(secom.features[split.validation])
    x_test = preprocessor.transform(secom.features[split.test])
    y_train = secom.labels[split.train].astype(np.float32)[:, None]
    y_validation = secom.labels[split.validation].astype(np.float32)[:, None]
    y_test = secom.labels[split.test].astype(np.int64)
    preprocessor_path = artifact_dir / "preprocessor.npz"
    preprocessor.save_npz(preprocessor_path)

    positive_weight = float((y_train.size - y_train.sum()) / y_train.sum())
    pos_weight = torch.tensor([positive_weight], device=device)
    loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def validation_ap(labels: np.ndarray, logits: np.ndarray) -> float:
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits.reshape(-1), -30, 30)))
        return float(average_precision_score(labels.reshape(-1), probability))

    result = train_model(
        StaticConvMLP(x_train.shape[1], (96, 32), 1),
        x_train,
        y_train,
        x_validation,
        y_validation,
        device,
        loss,
        epochs=260,
        patience=55,
        learning_rate=1.5e-3,
        batch_size=192,
        validation_score=validation_ap,
    )
    validation_logits = predict(result.model, x_validation, device).reshape(-1)
    validation_probability = 1.0 / (
        1.0 + np.exp(-np.clip(validation_logits, -30, 30))
    )
    threshold = threshold_from_validation(
        secom.labels[split.validation], validation_probability
    )
    test_logits = predict(result.model, x_test, device)
    metrics = binary_metrics(y_test, test_logits, threshold)
    baseline = {
        "name": "constant_training_prevalence",
        "pr_auc": float(np.mean(y_test)),
        "balanced_accuracy": 0.5,
    }
    bundle = export_bundle(
        model_id,
        result,
        x_test,
        {"type": "StaticConvMLP", "input": x_train.shape[1], "hidden": [96, 32], "output": 1},
    )
    quality = metrics["pr_auc"] > baseline["pr_auc"]
    return {
        "schema": "x5_icmat_foundry.process_model_receipt.v1",
        "inventory_id": model_id,
        "model_name": MODEL_NAMES[model_id],
        "task": "SECOM pass/fail risk ranking",
        "status": "PC_TRAINED_ONNX_READY_BOARD_PENDING" if quality else "PC_ARTIFACT_COMPLETE_QUALITY_LIMITED_BOARD_PENDING",
        "created_at_utc": utc_now(),
        "seed_runs": [SEED],
        "source": {"path": relative(SECOM_SOURCE), "sha256": sha256_file(SECOM_SOURCE)},
        "split": {
            "kind": "recorded_timestamp_calendar_batch_holdout",
            "train": int(split.train.size),
            "validation": int(split.validation.size),
            "test": int(split.test.size),
            "checks": validate_temporal_split(secom, split),
        },
        "training": {
            "best_epoch": result.best_epoch,
            "epochs_run": result.epochs_run,
            "seconds": result.elapsed_seconds,
            "positive_weight_train_only": positive_weight,
        },
        "metrics": metrics,
        "simple_baseline": baseline,
        "quality_note": "Single-seed public SECOM benchmark; not local-fab production accuracy.",
        **bundle,
        "x5_contacted": False,
    }


@dataclass
class Projection:
    indices: np.ndarray
    medians: np.ndarray
    centers: np.ndarray
    scales: np.ndarray

    def transform(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        selected = np.asarray(values[:, self.indices], dtype=np.float64)
        missing = np.isnan(selected)
        selected = np.where(missing, self.medians, selected)
        normalized = (selected - self.centers) / self.scales
        return normalized.astype(np.float32), missing.astype(np.float32)

    def save(self, path: Path) -> dict[str, Any]:
        return save_npz(
            path,
            indices=self.indices,
            medians=self.medians,
            centers=self.centers,
            scales=self.scales,
        )


def fit_projection(values: np.ndarray, max_features: int, min_observed: float = 0.5) -> Projection:
    values = np.asarray(values, dtype=np.float64)
    observed = np.mean(~np.isnan(values), axis=0)
    valid = observed >= min_observed
    medians_all = np.zeros(values.shape[1], dtype=np.float64)
    medians_all[valid] = np.nanmedian(values[:, valid], axis=0)
    imputed = np.where(np.isnan(values), medians_all, values)
    variance = np.var(imputed, axis=0)
    ranking = np.argsort(-np.where(valid & np.isfinite(variance), variance, -1.0), kind="stable")
    indices = ranking[: min(max_features, int(np.sum(valid)))]
    selected = imputed[:, indices]
    centers = np.mean(selected, axis=0)
    scales = np.std(selected, axis=0)
    scales = np.where(scales > 1e-8, scales, 1.0)
    return Projection(
        indices=indices.astype(np.int64),
        medians=medians_all[indices].astype(np.float64),
        centers=centers.astype(np.float64),
        scales=scales.astype(np.float64),
    )


def masked_examples(
    normalized: np.ndarray,
    original_missing: np.ndarray,
    seed: int,
    repeats: int,
    mask_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    inputs = []
    targets = []
    for _ in range(repeats):
        synthetic = (rng.random(normalized.shape) < mask_fraction) & (original_missing < 0.5)
        empty = np.flatnonzero(np.sum(synthetic, axis=1) == 0)
        for row in empty:
            available = np.flatnonzero(original_missing[row] < 0.5)
            if available.size:
                synthetic[row, available[int(rng.integers(available.size))]] = True
        corrupted = normalized.copy()
        corrupted[synthetic] = 0.0
        inputs.append(
            np.concatenate(
                (corrupted, original_missing, synthetic.astype(np.float32)), axis=1
            )
        )
        targets.append(
            np.concatenate((normalized, synthetic.astype(np.float32)), axis=1)
        )
    return np.concatenate(inputs).astype(np.float32), np.concatenate(targets).astype(np.float32)


def masked_loss(width: int) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def loss(prediction: torch.Tensor, packed_target: torch.Tensor) -> torch.Tensor:
        target = packed_target[:, :width]
        mask = packed_target[:, width:]
        squared = torch.square(prediction - target) * mask
        return torch.sum(squared) / torch.clamp(torch.sum(mask), min=1.0)

    return loss


def masked_metrics(prediction: np.ndarray, packed_target: np.ndarray, width: int) -> dict[str, float]:
    target = packed_target[:, :width]
    mask = packed_target[:, width:] > 0.5
    error = prediction[mask] - target[mask]
    baseline_error = target[mask]
    return {
        "masked_mae_standardized": float(np.mean(np.abs(error))),
        "masked_rmse_standardized": float(np.sqrt(np.mean(np.square(error)))),
        "train_mean_baseline_mae_standardized": float(np.mean(np.abs(baseline_error))),
        "train_mean_baseline_rmse_standardized": float(np.sqrt(np.mean(np.square(baseline_error)))),
        "masked_values_evaluated": int(np.sum(mask)),
    }


def train_fproc02(secom: Any, split: Any, device: torch.device) -> dict[str, Any]:
    model_id = "F-PROC-02"
    artifact_dir = ARTIFACT_ROOT / model_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    projection = fit_projection(secom.features[split.train], 96, min_observed=0.8)
    train_values, train_missing = projection.transform(secom.features[split.train])
    validation_values, validation_missing = projection.transform(secom.features[split.validation])
    test_values, test_missing = projection.transform(secom.features[split.test])
    train_x, train_y = masked_examples(train_values, train_missing, SEED + 2, 3)
    validation_x, validation_y = masked_examples(validation_values, validation_missing, SEED + 3, 1)
    test_x, test_y = masked_examples(test_values, test_missing, SEED + 4, 1)
    width = train_values.shape[1]
    loss = masked_loss(width)
    result = train_model(
        StaticConvMLP(train_x.shape[1], (160, 64), width),
        train_x,
        train_y,
        validation_x,
        validation_y,
        device,
        loss,
        epochs=220,
        patience=35,
        learning_rate=1.8e-3,
        batch_size=256,
    )
    prediction = predict(result.model, test_x, device)
    metrics = masked_metrics(prediction, test_y, width)
    projection_artifact = projection.save(artifact_dir / "preprocessor.npz")
    bundle = export_bundle(
        model_id,
        result,
        test_x,
        {"type": "StaticConvMLP", "input": train_x.shape[1], "hidden": [160, 64], "output": width},
    )
    quality = metrics["masked_mae_standardized"] < metrics["train_mean_baseline_mae_standardized"]
    return {
        "schema": "x5_icmat_foundry.process_model_receipt.v1",
        "inventory_id": model_id,
        "model_name": MODEL_NAMES[model_id],
        "task": "recover synthetically masked observed SECOM sensor values",
        "status": "PC_TRAINED_ONNX_READY_BOARD_PENDING" if quality else "PC_ARTIFACT_COMPLETE_QUALITY_LIMITED_BOARD_PENDING",
        "created_at_utc": utc_now(),
        "seed_runs": [SEED],
        "source": {"path": relative(SECOM_SOURCE), "sha256": sha256_file(SECOM_SOURCE)},
        "split": {
            "kind": "recorded_timestamp_calendar_batch_holdout",
            "train": int(split.train.size),
            "validation": int(split.validation.size),
            "test": int(split.test.size),
            "synthetic_mask_fraction": 0.2,
            "test_mask_seed": SEED + 4,
        },
        "training": {"best_epoch": result.best_epoch, "epochs_run": result.epochs_run, "seconds": result.elapsed_seconds},
        "metrics": metrics,
        "simple_baseline": "training-feature-mean (zero in train-standardized space)",
        "quality_note": "Synthetic missing-value restoration on a public benchmark; original missing cells are never used as targets.",
        **bundle,
        "x5_contacted": False,
    }


def load_pvd() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result = {}
    for domain in ("AlCu", "WTi"):
        x = pd.read_csv(PVD_SOURCE / f"X_pvd_{domain}.csv").to_numpy(dtype=np.float32)
        y = pd.read_csv(PVD_SOURCE / f"Y_pvd_{domain}.csv").to_numpy(dtype=np.float32)
        if x.shape[0] != y.shape[0] or y.shape[1] != 17:
            raise ValueError(f"invalid PVD {domain} shapes: {x.shape}, {y.shape}")
        result[domain] = (x, y)
    return result


def ordered_slices(size: int) -> tuple[slice, slice, slice]:
    train_end = int(math.floor(size * 0.70))
    validation_end = int(math.floor(size * 0.85))
    return slice(0, train_end), slice(train_end, validation_end), slice(validation_end, size)


def pad_pvd(values: np.ndarray, width: int, domain: int) -> np.ndarray:
    padded = np.zeros((values.shape[0], width + 1), dtype=np.float32)
    padded[:, : values.shape[1]] = values
    padded[:, width] = float(domain)
    return padded


def fit_scaler(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(values, axis=0, dtype=np.float64)
    scale = np.std(values, axis=0, dtype=np.float64)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return center.astype(np.float32), scale.astype(np.float32)


def train_fproc03(pvd: dict[str, tuple[np.ndarray, np.ndarray]], device: torch.device) -> dict[str, Any]:
    model_id = "F-PROC-03"
    artifact_dir = ARTIFACT_ROOT / model_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    width = max(values[0].shape[1] for values in pvd.values())
    input_width = 2 * width + 2
    alpha_grid = np.asarray(
        [0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0],
        dtype=np.float64,
    )
    coefficient = np.zeros((17, input_width), dtype=np.float32)
    scalers: dict[str, dict[str, np.ndarray | float]] = {}
    split_manifest: dict[str, Any] = {}
    validation_selection: dict[str, Any] = {}
    test_features: list[np.ndarray] = []
    test_targets: list[np.ndarray] = []
    test_baselines: list[np.ndarray] = []

    for domain_index, (domain, (x, y)) in enumerate(pvd.items()):
        train_slice, validation_slice, test_slice = ordered_slices(x.shape[0])
        x_center, x_scale = fit_scaler(x[train_slice])
        x_scale = np.maximum(x_scale, 1e-3).astype(np.float32)
        y_mean = np.mean(y[train_slice], axis=0, dtype=np.float64).astype(np.float32)

        def features(part: slice) -> np.ndarray:
            normalized = np.clip(
                (x[part] - x_center) / x_scale, -8.0, 8.0
            ).astype(np.float32)
            output = np.zeros((normalized.shape[0], input_width), dtype=np.float32)
            start = domain_index * width
            output[:, start : start + x.shape[1]] = normalized
            output[:, 2 * width + domain_index] = 1.0
            return output

        train_x = features(train_slice)
        validation_x = features(validation_slice)
        domain_test_x = features(test_slice)
        train_residual = (y[train_slice] - y_mean).astype(np.float64)
        best: tuple[float, float, Ridge] | None = None
        domain_columns = slice(domain_index * width, domain_index * width + x.shape[1])
        train_domain_x = train_x[:, domain_columns]
        validation_domain_x = validation_x[:, domain_columns]
        for alpha in alpha_grid:
            candidate = Ridge(alpha=float(alpha), fit_intercept=True)
            candidate.fit(train_domain_x, train_residual)
            validation_prediction = y_mean + candidate.predict(validation_domain_x)
            validation_mae = float(
                mean_absolute_error(y[validation_slice], validation_prediction)
            )
            if best is None or validation_mae < best[0]:
                best = (validation_mae, float(alpha), candidate)
        assert best is not None
        validation_mae, selected_alpha, ridge = best
        coefficient[:, domain_columns] = np.asarray(ridge.coef_, dtype=np.float32)
        coefficient[:, 2 * width + domain_index] = np.asarray(
            ridge.intercept_, dtype=np.float32
        )
        validation_baseline = np.repeat(
            y_mean[None, :], y[validation_slice].shape[0], axis=0
        )
        validation_selection[domain] = {
            "selected_alpha": selected_alpha,
            "ridge_mae_original_units": validation_mae,
            "per_domain_train_mean_mae_original_units": float(
                mean_absolute_error(y[validation_slice], validation_baseline)
            ),
        }
        scalers[domain] = {
            "x_center": x_center,
            "x_scale": x_scale,
            "y_mean": y_mean,
            "selected_alpha": selected_alpha,
        }
        test_features.append(domain_test_x)
        test_targets.append(y[test_slice])
        test_baselines.append(
            np.repeat(y_mean[None, :], y[test_slice].shape[0], axis=0)
        )
        split_manifest[domain] = {
            "rows": int(x.shape[0]),
            "train": [train_slice.start, train_slice.stop],
            "validation": [validation_slice.start, validation_slice.stop],
            "test": [test_slice.start, test_slice.stop],
        }

    x_test = np.concatenate(test_features)
    y_test_raw = np.concatenate(test_targets)
    baseline_prediction = np.concatenate(test_baselines)
    model = StaticLinear(input_width, 17)
    with torch.no_grad():
        model.linear.weight.copy_(torch.from_numpy(coefficient))
        model.linear.bias.zero_()
    result = TrainResult(
        model=model,
        best_epoch=0,
        epochs_run=0,
        best_validation_loss=float(
            np.mean([value["ridge_mae_original_units"] for value in validation_selection.values()])
        ),
        elapsed_seconds=0.0,
    )
    residual_prediction = predict(model, x_test, torch.device("cpu"))
    prediction = baseline_prediction + residual_prediction
    metrics = regression_metrics(y_test_raw, prediction)
    baseline_per_domain: dict[str, Any] = {}
    per_domain: dict[str, Any] = {}
    offset = 0
    for domain, target in zip(pvd, test_targets, strict=True):
        count = target.shape[0]
        per_domain[domain] = regression_metrics(
            target, prediction[offset : offset + count]
        )
        baseline_per_domain[domain] = regression_metrics(
            target, baseline_prediction[offset : offset + count]
        )
        offset += count
    baseline = {
        "name": "per-domain training_target_mean",
        **regression_metrics(y_test_raw, baseline_prediction),
        "per_domain": baseline_per_domain,
    }
    scaler_artifact = save_npz(
        artifact_dir / "preprocessor.npz",
        x_center_alcu=np.asarray(scalers["AlCu"]["x_center"]),
        x_scale_alcu=np.asarray(scalers["AlCu"]["x_scale"]),
        y_mean_alcu=np.asarray(scalers["AlCu"]["y_mean"]),
        alpha_alcu=np.asarray([scalers["AlCu"]["selected_alpha"]], dtype=np.float64),
        x_center_wti=np.asarray(scalers["WTi"]["x_center"]),
        x_scale_wti=np.asarray(scalers["WTi"]["x_scale"]),
        y_mean_wti=np.asarray(scalers["WTi"]["y_mean"]),
        alpha_wti=np.asarray([scalers["WTi"]["selected_alpha"]], dtype=np.float64),
        padded_sensor_width=np.asarray([width], dtype=np.int64),
    )
    bundle = export_bundle(
        model_id,
        result,
        x_test,
        {"type": "single StaticLinear/Gemm residual-to-per-domain-mean", "input": input_width, "hidden": [], "output": 17},
    )
    quality = metrics["mae"] < baseline["mae"]
    pvd_sources = []
    for name in ("X_pvd_AlCu.csv", "Y_pvd_AlCu.csv", "X_pvd_WTi.csv", "Y_pvd_WTi.csv", "zenodo_record_16881338.json"):
        path = PVD_SOURCE / name
        pvd_sources.append({"path": relative(path), "sha256": sha256_file(path)})
    return {
        "schema": "x5_icmat_foundry.process_model_receipt.v1",
        "inventory_id": model_id,
        "model_name": MODEL_NAMES[model_id],
        "task": "predict 17-point PVD film-thickness residual from the per-domain training mean",
        "status": "PC_TRAINED_ONNX_READY_BOARD_PENDING" if quality else "PC_ARTIFACT_COMPLETE_QUALITY_LIMITED_BOARD_PENDING",
        "created_at_utc": utc_now(),
        "seed_runs": [SEED],
        "source": {
            "doi": "10.5281/zenodo.16881338",
            "license": "CC-BY-4.0",
            "files": pvd_sources,
        },
        "split": {
            "kind": "ordered-row proxy holdout within each AlCu/WTi domain",
            "warning": "The public files contain no explicit timestamps; row order is a temporal proxy, not recorded equipment time.",
            "domains": split_manifest,
        },
        "training": {
            "method": "multi-output Ridge residual; coefficients packed into one torch Linear/Gemm",
            "alpha_grid": alpha_grid.tolist(),
            "validation_selection": validation_selection,
            "seed": SEED,
            "cpu_only": True,
        },
        "metrics": {"combined": metrics, "per_domain": per_domain},
        "simple_baseline": baseline,
        "quality_note": "ONNX emits the original-unit residual; add the matching per-domain training mean from preprocessor.npz to recover the 17 source values. Promotion requires strict test MAE improvement over the per-domain train-mean baseline.",
        **bundle,
        "x5_contacted": False,
    }


def make_temporal_pairs(values: np.ndarray, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    near_count = count // 2
    far_count = count - near_count
    near_left = rng.integers(0, max(values.shape[0] - 4, 1), size=near_count)
    near_gap = rng.integers(1, 4, size=near_count)
    near_right = np.minimum(near_left + near_gap, values.shape[0] - 1)
    low = max(values.shape[0] // 3, 1)
    far_left = rng.integers(0, low, size=far_count)
    far_right = rng.integers(max(2 * values.shape[0] // 3, low + 1), values.shape[0], size=far_count)
    near = np.abs(values[near_left] - values[near_right])
    far = np.abs(values[far_left] - values[far_right])
    x = np.concatenate((near, far)).astype(np.float32)
    y = np.concatenate((np.zeros(near_count), np.ones(far_count))).astype(np.float32)[:, None]
    order = rng.permutation(count)
    return x[order], y[order]


def train_fproc07(secom: Any, split: Any, device: torch.device) -> dict[str, Any]:
    model_id = "F-PROC-07"
    artifact_dir = ARTIFACT_ROOT / model_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    projection = fit_projection(secom.features[split.train], 96, min_observed=0.5)
    train_values, _ = projection.transform(secom.features[split.train])
    validation_values, _ = projection.transform(secom.features[split.validation])
    test_values, _ = projection.transform(secom.features[split.test])
    train_x, train_y = make_temporal_pairs(train_values, 2400, SEED + 7)
    validation_x, validation_y = make_temporal_pairs(validation_values, 600, SEED + 8)
    test_x, test_y = make_temporal_pairs(test_values, 600, SEED + 9)
    loss = nn.BCEWithLogitsLoss()

    def validation_auc(labels: np.ndarray, logits: np.ndarray) -> float:
        return float(roc_auc_score(labels.reshape(-1), logits.reshape(-1)))

    result = train_model(
        StaticConvMLP(train_x.shape[1], (96, 32), 1),
        train_x,
        train_y,
        validation_x,
        validation_y,
        device,
        loss,
        epochs=180,
        patience=30,
        learning_rate=1.8e-3,
        batch_size=256,
        validation_score=validation_auc,
    )
    validation_logits = predict(result.model, validation_x, device).reshape(-1)
    validation_probability = 1.0 / (1.0 + np.exp(-np.clip(validation_logits, -30, 30)))
    threshold = threshold_from_validation(validation_y.reshape(-1), validation_probability)
    metrics = binary_metrics(test_y.reshape(-1).astype(np.int64), predict(result.model, test_x, device), threshold)
    baseline = {"name": "constant_equal_probability", "pr_auc": 0.5, "roc_auc": 0.5, "balanced_accuracy": 0.5}
    projection_artifact = projection.save(artifact_dir / "preprocessor.npz")
    bundle = export_bundle(
        model_id,
        result,
        test_x,
        {"type": "StaticConvMLP", "input": train_x.shape[1], "hidden": [96, 32], "output": 1},
    )
    quality = metrics["roc_auc"] > 0.5
    return {
        "schema": "x5_icmat_foundry.process_model_receipt.v1",
        "inventory_id": model_id,
        "model_name": MODEL_NAMES[model_id],
        "task": "learn whether two runs are near or far apart within a chronological window",
        "status": "PC_TRAINED_ONNX_READY_BOARD_PENDING" if quality else "PC_ARTIFACT_COMPLETE_QUALITY_LIMITED_BOARD_PENDING",
        "created_at_utc": utc_now(),
        "seed_runs": [SEED],
        "source": {"path": relative(SECOM_SOURCE), "sha256": sha256_file(SECOM_SOURCE)},
        "split": {
            "kind": "recorded_timestamp calendar-batch train/validation/test, then deterministic within-partition temporal pairs",
            "train_pairs": int(train_x.shape[0]),
            "validation_pairs": int(validation_x.shape[0]),
            "test_pairs": int(test_x.shape[0]),
        },
        "training": {"best_epoch": result.best_epoch, "epochs_run": result.epochs_run, "seconds": result.elapsed_seconds},
        "metrics": metrics,
        "simple_baseline": baseline,
        "quality_note": "This is a learned pairwise temporal-drift signal, not a fabricated process change-point ground truth.",
        **bundle,
        "x5_contacted": False,
    }


def reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.square(prediction - target))


def anomaly_metrics(id_score: np.ndarray, ood_score: np.ndarray, threshold: float) -> dict[str, float]:
    labels = np.concatenate((np.zeros(id_score.size), np.ones(ood_score.size))).astype(np.int64)
    score = np.concatenate((id_score, ood_score))
    return {
        "roc_auc": float(roc_auc_score(labels, score)),
        "pr_auc": float(average_precision_score(labels, score)),
        "unknown_recall": float(np.mean(ood_score > threshold)),
        "id_false_positive_rate": float(np.mean(id_score > threshold)),
        "threshold_from_validation_id_p95": float(threshold),
        "id_count": int(id_score.size),
        "unknown_count": int(ood_score.size),
    }


def train_fproc08(pvd: dict[str, tuple[np.ndarray, np.ndarray]], device: torch.device) -> dict[str, Any]:
    model_id = "F-PROC-08"
    artifact_dir = ARTIFACT_ROOT / model_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    alcu = pvd["AlCu"][0]
    wti = pvd["WTi"][0]
    common_width = min(alcu.shape[1], wti.shape[1])
    train_slice, validation_slice, test_slice = ordered_slices(alcu.shape[0])
    x_train_raw = alcu[train_slice, :common_width]
    x_validation_raw = alcu[validation_slice, :common_width]
    x_test_id_raw = alcu[test_slice, :common_width]
    x_test_ood_raw = wti[:, :common_width]
    center, scale = fit_scaler(x_train_raw)
    normalize = lambda values: ((values - center) / scale).astype(np.float32)
    x_train = normalize(x_train_raw)
    x_validation = normalize(x_validation_raw)
    x_test_id = normalize(x_test_id_raw)
    x_test_ood = normalize(x_test_ood_raw)
    result = train_model(
        StaticConvMLP(common_width, (48, 12, 48), common_width),
        x_train,
        x_train,
        x_validation,
        x_validation,
        device,
        reconstruction_loss,
        epochs=220,
        patience=35,
        learning_rate=1.8e-3,
        batch_size=384,
    )
    validation_score = np.mean(np.square(predict(result.model, x_validation, device) - x_validation), axis=1)
    id_score = np.mean(np.square(predict(result.model, x_test_id, device) - x_test_id), axis=1)
    ood_score = np.mean(np.square(predict(result.model, x_test_ood, device) - x_test_ood), axis=1)
    threshold = float(np.quantile(validation_score, 0.95))
    metrics = anomaly_metrics(id_score, ood_score, threshold)
    baseline_id = np.mean(np.square(x_test_id), axis=1)
    baseline_ood = np.mean(np.square(x_test_ood), axis=1)
    baseline_labels = np.concatenate((np.zeros(baseline_id.size), np.ones(baseline_ood.size)))
    baseline_scores = np.concatenate((baseline_id, baseline_ood))
    baseline = {
        "name": "distance_to_training_mean",
        "roc_auc": float(roc_auc_score(baseline_labels, baseline_scores)),
        "pr_auc": float(average_precision_score(baseline_labels, baseline_scores)),
    }
    scaler_artifact = save_npz(artifact_dir / "preprocessor.npz", center=center, scale=scale, shared_sensor_width=np.asarray([common_width], dtype=np.int64))
    bundle = export_bundle(
        model_id,
        result,
        x_test_id,
        {"type": "StaticConvMLP autoencoder", "input": common_width, "hidden": [48, 12, 48], "output": common_width},
    )
    return {
        "schema": "x5_icmat_foundry.process_model_receipt.v1",
        "inventory_id": model_id,
        "model_name": MODEL_NAMES[model_id],
        "task": "leave-one-PVD-material-domain-out reconstruction OOD",
        "status": "PC_TRAINED_ONNX_READY_BOARD_PENDING" if metrics["roc_auc"] > 0.5 else "PC_ARTIFACT_COMPLETE_QUALITY_LIMITED_BOARD_PENDING",
        "created_at_utc": utc_now(),
        "seed_runs": [SEED],
        "source": {"doi": "10.5281/zenodo.16881338", "license": "CC-BY-4.0"},
        "split": {
            "kind": "leave-WTi-domain-out; ordered-row AlCu 70/15/15 for train/calibration/ID test",
            "warning": "The public files contain no timestamps; AlCu row order is only a temporal proxy.",
            "train_domain": "AlCu",
            "held_out_unknown_domain": "WTi",
            "shared_sensors": common_width,
        },
        "training": {"best_epoch": result.best_epoch, "epochs_run": result.epochs_run, "seconds": result.elapsed_seconds},
        "metrics": metrics,
        "simple_baseline": baseline,
        "quality_note": "OOD means held-out public material/process domain, not arbitrary fab excursions.",
        **bundle,
        "x5_contacted": False,
    }


def synthetic_anomaly_examples(
    normal: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    blocks = [normal]
    labels = [np.zeros((normal.shape[0], 1), dtype=np.float32)]
    columns = np.arange(normal.shape[1])[None, :]
    masked = normal.copy()
    masked[rng.random(normal.shape) < 0.22] = 0.0
    spiked = normal.copy()
    spike_mask = rng.random(normal.shape) < 0.08
    spiked += spike_mask * rng.normal(0.0, 2.5, size=normal.shape)
    shuffled = normal.copy()
    donor_rows = rng.integers(0, normal.shape[0], size=normal.shape)
    replace = rng.random(normal.shape) < 0.25
    shuffled[replace] = normal[donor_rows, columns][replace]
    for synthetic in (masked, spiked, shuffled):
        blocks.append(synthetic.astype(np.float32))
        labels.append(np.ones((normal.shape[0], 1), dtype=np.float32))
    values = np.concatenate(blocks)
    target = np.concatenate(labels)
    order = rng.permutation(values.shape[0])
    return values[order], target[order]


def train_fproc09(secom: Any, split: Any, device: torch.device) -> dict[str, Any]:
    model_id = "F-PROC-09"
    artifact_dir = ARTIFACT_ROOT / model_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    normal_train_indices = split.train[secom.labels[split.train] == 0]
    normal_validation_indices = split.validation[secom.labels[split.validation] == 0]
    projection = fit_projection(secom.features[normal_train_indices], 96, min_observed=0.5)
    normal_train, _ = projection.transform(secom.features[normal_train_indices])
    normal_validation, _ = projection.transform(secom.features[normal_validation_indices])
    x_test, _ = projection.transform(secom.features[split.test])
    x_train, y_train = synthetic_anomaly_examples(normal_train, SEED + 90)
    x_validation, y_validation = synthetic_anomaly_examples(
        normal_validation, SEED + 91
    )

    def validation_auc(labels: np.ndarray, logits: np.ndarray) -> float:
        return float(roc_auc_score(labels.reshape(-1), logits.reshape(-1)))

    result = train_model(
        StaticConvMLP(x_train.shape[1], (64, 24), 1),
        x_train,
        y_train,
        x_validation,
        y_validation,
        device,
        nn.BCEWithLogitsLoss(),
        epochs=220,
        patience=35,
        learning_rate=1.8e-3,
        batch_size=256,
        validation_score=validation_auc,
    )
    validation_logit = predict(result.model, normal_validation, device).reshape(-1)
    validation_score = 1.0 / (
        1.0 + np.exp(-np.clip(validation_logit, -30, 30))
    )
    test_labels = secom.labels[split.test].astype(np.int64)
    normal_test = x_test[test_labels == 0]
    permutation = np.random.default_rng(SEED + 92).permutation(normal_test.shape[1])
    unseen_corruption = normal_test[:, permutation].astype(np.float32)
    primary_x = np.concatenate((normal_test, unseen_corruption))
    primary_labels = np.concatenate(
        (np.zeros(normal_test.shape[0]), np.ones(unseen_corruption.shape[0]))
    ).astype(np.int64)
    primary_logit = predict(result.model, primary_x, device).reshape(-1)
    primary_score = 1.0 / (1.0 + np.exp(-np.clip(primary_logit, -30, 30)))
    threshold = float(np.quantile(validation_score, 0.95))
    primary_metrics = {
        "corruption_type": "held-out deterministic sensor-channel permutation",
        "roc_auc": float(roc_auc_score(primary_labels, primary_score)),
        "pr_auc": float(average_precision_score(primary_labels, primary_score)),
        "corruption_recall_at_normal_validation_p95": float(
            np.mean(primary_score[primary_labels == 1] > threshold)
        ),
        "normal_false_positive_rate": float(
            np.mean(primary_score[primary_labels == 0] > threshold)
        ),
        "threshold_from_normal_validation_p95": threshold,
        "normal_rows": int(normal_test.shape[0]),
    }
    baseline_score = np.mean(np.square(primary_x), axis=1)
    baseline = {
        "name": "distance_to_training_normal_mean",
        "roc_auc": float(roc_auc_score(primary_labels, baseline_score)),
        "pr_auc": float(average_precision_score(primary_labels, baseline_score)),
    }
    actual_failure_logit = predict(result.model, x_test, device).reshape(-1)
    actual_failure_score = 1.0 / (
        1.0 + np.exp(-np.clip(actual_failure_logit, -30, 30))
    )
    secondary = {
        "status": "SECONDARY_OBSERVATION_NOT_PROMOTION_GATE",
        "roc_auc": float(roc_auc_score(test_labels, actual_failure_score)),
        "pr_auc": float(average_precision_score(test_labels, actual_failure_score)),
        "failure_recall_at_normal_validation_p95": float(
            np.mean(actual_failure_score[test_labels == 1] > threshold)
        ),
        "test_failures": int(np.sum(test_labels == 1)),
    }
    projection_artifact = projection.save(artifact_dir / "preprocessor.npz")
    bundle = export_bundle(
        model_id,
        result,
        x_test,
        {"type": "StaticConvMLP synthetic-anomaly discriminator", "input": x_train.shape[1], "hidden": [64, 24], "output": 1},
    )
    quality = (
        primary_metrics["roc_auc"] > baseline["roc_auc"]
        and primary_metrics["pr_auc"] > baseline["pr_auc"]
    )
    return {
        "schema": "x5_icmat_foundry.process_model_receipt.v1",
        "inventory_id": model_id,
        "model_name": MODEL_NAMES[model_id],
        "task": "unsupervised early-normal-window SECOM synthetic-corruption anomaly score",
        "status": "PC_TRAINED_ONNX_READY_BOARD_PENDING" if quality else "PC_ARTIFACT_COMPLETE_QUALITY_LIMITED_BOARD_PENDING",
        "created_at_utc": utc_now(),
        "seed_runs": [SEED],
        "source": {"path": relative(SECOM_SOURCE), "sha256": sha256_file(SECOM_SOURCE)},
        "split": {
            "kind": "recorded timestamp calendar-batch holdout",
            "fit_labels": "early normal rows plus deterministic mask/spike/cross-row shuffle corruptions; failure labels excluded from fitting",
            "threshold": "95th percentile of raw normal-validation anomaly probability",
            "primary_test": "independent time-out normal rows paired with a held-out sensor-channel permutation corruption type",
            "secondary_test": "real SECOM failure labels, observation only",
        },
        "training": {"best_epoch": result.best_epoch, "epochs_run": result.epochs_run, "seconds": result.elapsed_seconds},
        "metrics": {
            "primary_unseen_corruption": primary_metrics,
            "secondary_real_failure_observation": secondary,
        },
        "simple_baseline": baseline,
        "quality_note": "SECOM failure labels are used only for final evaluation, not model fitting or threshold calibration.",
        **bundle,
        "x5_contacted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--models",
        default=",".join(MODEL_NAMES),
        help="comma-separated inventory IDs; omitted IDs are not retrained",
    )
    args = parser.parse_args()
    selected = [value.strip() for value in args.models.split(",") if value.strip()]
    unknown = sorted(set(selected) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"unknown model IDs: {unknown}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    selected_device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    device = torch.device(selected_device)
    set_seed()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    if not SECOM_SOURCE.is_file():
        raise FileNotFoundError(SECOM_SOURCE)
    if not PVD_SOURCE.is_dir():
        raise FileNotFoundError(PVD_SOURCE)

    started = time.perf_counter()
    secom = load_secom_zip(SECOM_SOURCE)
    split = temporal_batch_split(secom.timestamps, secom.labels)
    pvd = load_pvd()
    jobs = {
        "F-PROC-01": lambda: train_fproc01(secom, split, device),
        "F-PROC-02": lambda: train_fproc02(secom, split, device),
        "F-PROC-03": lambda: train_fproc03(pvd, device),
        "F-PROC-07": lambda: train_fproc07(secom, split, device),
        "F-PROC-08": lambda: train_fproc08(pvd, device),
        "F-PROC-09": lambda: train_fproc09(secom, split, device),
    }
    receipts = []
    for inventory_id in MODEL_NAMES:
        path = EVIDENCE_ROOT / inventory_id / "receipt.json"
        if inventory_id in selected:
            receipt = jobs[inventory_id]()
            path = write_receipt(receipt["inventory_id"], receipt)
        else:
            if not path.is_file():
                raise FileNotFoundError(
                    f"unselected model has no existing receipt: {inventory_id}"
                )
            receipt = json.loads(path.read_text(encoding="utf-8"))
        receipts.append(
            {
                "inventory_id": receipt["inventory_id"],
                "model_name": receipt["model_name"],
                "status": receipt["status"],
                "receipt": relative(path),
                "receipt_sha256": sha256_file(path),
                "weight_sha256": next(
                    item["sha256"] for item in receipt["artifacts"] if item["path"].endswith("/model.pt")
                ),
                "onnx_sha256": next(
                    item["sha256"] for item in receipt["artifacts"] if item["path"].endswith("/model.onnx")
                ),
            }
        )
        if inventory_id in selected:
            print(json.dumps(receipts[-1], ensure_ascii=False), flush=True)

    weight_hashes = [item["weight_sha256"] for item in receipts]
    quality_limited = [
        item["inventory_id"]
        for item in receipts
        if "QUALITY_LIMITED" in item["status"]
    ]
    summary = {
        "schema": "x5_icmat_foundry.process_bank_run.v1",
        "created_at_utc": utc_now(),
        "status": (
            "PC_PROCESS_BANK_COMPLETE_BOARD_PENDING"
            if not quality_limited
            else "PC_PROCESS_BANK_COMPLETE_WITH_QUALITY_LIMITATIONS_BOARD_PENDING"
        ),
        "models_requested": list(MODEL_NAMES),
        "models_retrained_this_run": selected,
        "models_completed": len(receipts),
        "independent_weight_hashes": len(set(weight_hashes)),
        "all_weights_unique": len(set(weight_hashes)) == len(receipts),
        "quality_limited_models": quality_limited,
        "all_onnx_static_opset11_ir7_ort_pass": True,
        "single_seed_per_model": SEED,
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "receipts": receipts,
        "scope": {
            "written_roots": [relative(Path(__file__)), relative(ARTIFACT_ROOT), relative(EVIDENCE_ROOT)],
            "registry_written": False,
            "overlay_written": False,
            "agents_written": False,
            "production_written": False,
            "x5_contacted": False,
        },
        "script_sha256": sha256_file(Path(__file__)),
    }
    summary_path = EVIDENCE_ROOT / "process_bank_run.v1.json"
    write_json(summary_path, summary)
    summary_path.with_suffix(".sha256").write_text(
        f"{sha256_file(summary_path)}  {summary_path.name}\n", encoding="ascii"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
