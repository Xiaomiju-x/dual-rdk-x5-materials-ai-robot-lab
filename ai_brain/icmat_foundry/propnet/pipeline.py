"""Train, evaluate, export, and package the isolated PropNet candidate."""
from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import io
import json
import math
import os
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .contracts import (
    ALLOWED_ONNX_OPS,
    CLAIM_BOUNDARY,
    FEATURE_NAMES,
    MODEL_HIDDEN_DIMS,
    MODEL_INPUT_SHAPE,
    MODEL_OUTPUT_SHAPE,
    ONNX_OPSET,
    PRIMARY_TARGETS,
    SCHEMA_VERSION,
    SPLIT_SEED,
    TARGET_SPECS,
)
from .data import PreparedDataset, load_prepared_dataset, split_name
from .model import PropNet, masked_smooth_l1_loss

ARTIFACT_FILENAMES = (
    "artifact_manifest.json",
    "data_manifest.json",
    "feature_contract.json",
    "metrics_manifest.json",
    "model_fp32.onnx",
    "model_fp32.pt",
    "model_manifest.json",
    "preprocessing.npz",
    "split_assignments.csv.gz",
    "test_predictions.csv.gz",
    "training_history.json",
)


@dataclass(frozen=True)
class BuildConfig:
    seed: int = SPLIT_SEED
    hidden_dims: tuple[int, ...] = MODEL_HIDDEN_DIMS
    batch_size: int = 2048
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    patience: int = 15
    feature_clip_abs: float = 8.0
    ridge_alpha: float = 1.0
    device: str = "auto"


@dataclass
class Preprocessing:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    feature_clip_abs: float


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _resolve_output(root: Path, output: Path) -> Path:
    allowed = (root / "evaluation" / "icmat_foundry" / "propnet").resolve()
    resolved = output.resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise ValueError(f"output must stay inside {allowed}, got {resolved}")
    return resolved


def _set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"device must be auto, cpu, or cuda; got {requested!r}")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def fit_preprocessing(dataset: PreparedDataset, feature_clip_abs: float) -> Preprocessing:
    train = dataset.indices("train")
    if not train.size:
        raise ValueError("empty training split")
    feature_mean = np.mean(dataset.features[train], axis=0, dtype=np.float64)
    feature_scale = np.std(dataset.features[train], axis=0, dtype=np.float64)
    feature_scale[feature_scale < 1e-8] = 1.0

    target_mean = np.empty(len(TARGET_SPECS), dtype=np.float64)
    target_scale = np.empty(len(TARGET_SPECS), dtype=np.float64)
    for task_index, spec in enumerate(TARGET_SPECS):
        active = train[dataset.label_mask[train, task_index]]
        if not active.size:
            raise ValueError(f"no training labels for {spec.name}")
        target_mean[task_index] = np.mean(
            dataset.labels[active, task_index], dtype=np.float64
        )
        target_scale[task_index] = np.std(
            dataset.labels[active, task_index], dtype=np.float64
        )
        if target_scale[task_index] < 1e-8:
            raise ValueError(f"zero target variance for {spec.name}")

    return Preprocessing(
        feature_mean=feature_mean.astype(np.float32),
        feature_scale=feature_scale.astype(np.float32),
        target_mean=target_mean.astype(np.float32),
        target_scale=target_scale.astype(np.float32),
        feature_clip_abs=float(feature_clip_abs),
    )


def transform_features(features: np.ndarray, preprocessing: Preprocessing) -> np.ndarray:
    normalized = (features - preprocessing.feature_mean) / preprocessing.feature_scale
    return np.clip(
        normalized,
        -preprocessing.feature_clip_abs,
        preprocessing.feature_clip_abs,
    ).astype(np.float32)


def transform_targets(
    labels: np.ndarray,
    label_mask: np.ndarray,
    preprocessing: Preprocessing,
) -> np.ndarray:
    normalized = np.zeros_like(labels, dtype=np.float32)
    for task_index in range(labels.shape[1]):
        active = label_mask[:, task_index]
        normalized[active, task_index] = (
            labels[active, task_index] - preprocessing.target_mean[task_index]
        ) / preprocessing.target_scale[task_index]
    return normalized


def _validation_score(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    with torch.inference_mode():
        prediction = model(features.to(device)).cpu()
    scores: list[float] = []
    for task_index in range(prediction.shape[1]):
        active = mask[:, task_index].bool()
        if torch.any(active):
            scores.append(
                float(torch.mean(torch.abs(prediction[active, task_index] - targets[active, task_index])))
            )
    if not scores:
        raise ValueError("validation split has no active targets")
    return float(np.mean(scores))


def train_model(
    dataset: PreparedDataset,
    features: np.ndarray,
    normalized_targets: np.ndarray,
    config: BuildConfig,
    device: torch.device,
) -> tuple[PropNet, list[dict[str, Any]], dict[str, Any]]:
    train_indices = dataset.indices("train")
    val_indices = dataset.indices("val")
    train_dataset = TensorDataset(
        torch.from_numpy(features[train_indices]),
        torch.from_numpy(normalized_targets[train_indices]),
        torch.from_numpy(dataset.label_mask[train_indices]),
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
        drop_last=False,
    )

    model = PropNet(
        input_dim=features.shape[1],
        output_dim=normalized_targets.shape[1],
        hidden_dims=config.hidden_dims,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-5,
    )
    val_features = torch.from_numpy(features[val_indices])
    val_targets = torch.from_numpy(normalized_targets[val_indices])
    val_mask = torch.from_numpy(dataset.label_mask[val_indices])

    best_score = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        weighted_loss = 0.0
        seen = 0
        for batch_features, batch_targets, batch_mask in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_features)
            loss = masked_smooth_l1_loss(prediction, batch_targets, batch_mask)
            loss.backward()
            optimizer.step()
            batch_rows = int(batch_features.shape[0])
            weighted_loss += float(loss.detach().cpu()) * batch_rows
            seen += batch_rows

        val_score = _validation_score(
            model,
            val_features,
            val_targets,
            val_mask,
            device,
        )
        scheduler.step(val_score)
        epoch_record = {
            "epoch": epoch,
            "train_masked_smooth_l1": weighted_loss / max(seen, 1),
            "val_macro_normalized_mae": val_score,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_record)
        if val_score < best_score - 1e-6:
            best_score = val_score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break

    if best_state is None:
        raise RuntimeError("training did not produce a best checkpoint")
    model.load_state_dict(best_state)
    model = model.cpu().eval()
    return model, history, {
        "best_epoch": best_epoch,
        "epochs_executed": len(history),
        "best_val_macro_normalized_mae": best_score,
        "early_stopped": len(history) < config.max_epochs,
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "torch_version": torch.__version__,
    }


def _batched_torch_prediction(
    model: PropNet,
    features: np.ndarray,
    batch_size: int = 8192,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            output = model(torch.from_numpy(features[start : start + batch_size]))
            outputs.append(output.numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def _raw_predictions(
    normalized_prediction: np.ndarray,
    preprocessing: Preprocessing,
) -> np.ndarray:
    return (
        normalized_prediction * preprocessing.target_scale
        + preprocessing.target_mean
    ).astype(np.float32)


def baseline_predictions(
    dataset: PreparedDataset,
    features: np.ndarray,
    preprocessing: Preprocessing,
    ridge_alpha: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train = dataset.indices("train")
    mean_prediction = np.tile(
        preprocessing.target_mean,
        (len(dataset.features), 1),
    ).astype(np.float32)
    ridge_prediction = np.zeros_like(dataset.labels, dtype=np.float32)
    fit_counts: dict[str, int] = {}
    for task_index, spec in enumerate(TARGET_SPECS):
        active = train[dataset.label_mask[train, task_index]]
        model = Ridge(alpha=ridge_alpha, solver="lsqr")
        model.fit(features[active], dataset.labels[active, task_index])
        ridge_prediction[:, task_index] = model.predict(features).astype(np.float32)
        fit_counts[spec.name] = int(active.size)
    return mean_prediction, ridge_prediction, {
        "mean_baseline": "train-label mean per task",
        "ridge_baseline": {
            "algorithm": "sklearn.linear_model.Ridge",
            "alpha": ridge_alpha,
            "solver": "lsqr",
            "fit_counts": fit_counts,
            "selection_data": "train only",
        },
    }


def _regression_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    groups: Iterable[str],
) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    residual = target - prediction
    absolute = np.abs(residual)
    denominator = float(np.sum((target - np.mean(target)) ** 2))
    r2 = (
        1.0 - float(np.sum(residual**2)) / denominator
        if denominator > 0.0
        else 0.0
    )
    grouped: dict[str, list[float]] = {}
    for group, error in zip(groups, absolute.tolist(), strict=True):
        grouped.setdefault(group, []).append(error)
    group_mae = [float(np.mean(errors)) for errors in grouped.values()]
    return {
        "n": int(target.size),
        "groups": len(grouped),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "median_absolute_error": float(np.median(absolute)),
        "p90_absolute_error": float(np.quantile(absolute, 0.90)),
        "group_macro_mae": float(np.mean(group_mae)),
        "r2": r2,
    }


def evaluate_models(
    dataset: PreparedDataset,
    predictions: dict[str, np.ndarray],
    preprocessing: Preprocessing,
) -> dict[str, Any]:
    result: dict[str, Any] = {"models": {}, "conformal_90": {}}
    for model_name, prediction in predictions.items():
        result["models"][model_name] = {}
        for split in ("val", "test"):
            indices = dataset.indices(split)
            per_target: dict[str, Any] = {}
            normalized_mae: list[float] = []
            for task_index, spec in enumerate(TARGET_SPECS):
                active = indices[dataset.label_mask[indices, task_index]]
                metrics = _regression_metrics(
                    dataset.labels[active, task_index],
                    prediction[active, task_index],
                    (dataset.formula_groups[index] for index in active),
                )
                metrics["unit"] = spec.unit
                per_target[spec.name] = metrics
                normalized_mae.append(
                    metrics["mae"] / float(preprocessing.target_scale[task_index])
                )
            result["models"][model_name][split] = {
                "per_target": per_target,
                "macro_normalized_mae": float(np.mean(normalized_mae)),
            }

    mlp_prediction = predictions["propnet_mlp"]
    val_indices = dataset.indices("val")
    test_indices = dataset.indices("test")
    for task_index, spec in enumerate(TARGET_SPECS):
        val_active = val_indices[dataset.label_mask[val_indices, task_index]]
        test_active = test_indices[dataset.label_mask[test_indices, task_index]]
        val_residual = np.abs(
            dataset.labels[val_active, task_index]
            - mlp_prediction[val_active, task_index]
        )
        rank = min(
            int(math.ceil((len(val_residual) + 1) * 0.90)),
            len(val_residual),
        )
        half_width = float(np.sort(val_residual)[rank - 1])
        test_residual = np.abs(
            dataset.labels[test_active, task_index]
            - mlp_prediction[test_active, task_index]
        )
        result["conformal_90"][spec.name] = {
            "calibration_split": "val",
            "calibration_n": int(val_active.size),
            "finite_sample_rank": rank,
            "half_width": half_width,
            "unit": spec.unit,
            "test_n": int(test_active.size),
            "test_coverage": float(np.mean(test_residual <= half_width)),
            "boundary": (
                "Marginal split-conformal interval on the version-pinned public "
                "DFT test protocol; not a conditional or experimental guarantee."
            ),
        }
    return result


def export_onnx(model: PropNet) -> tuple[bytes, dict[str, Any]]:
    stream = io.BytesIO()
    dummy = torch.zeros(MODEL_INPUT_SHAPE, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        stream,
        input_names=["features_normalized_fp32"],
        output_names=["properties_normalized"],
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    payload = stream.getvalue()
    graph = onnx.load_model_from_string(payload)
    onnx.checker.check_model(graph)
    operations = sorted({node.op_type for node in graph.graph.node})
    unsupported = sorted(set(operations) - ALLOWED_ONNX_OPS)
    if unsupported:
        raise ValueError(f"ONNX graph contains non-contract operations: {unsupported}")
    return payload, {
        "opset": ONNX_OPSET,
        "operations": operations,
        "allowed_operations": sorted(ALLOWED_ONNX_OPS),
        "unsupported_operations": unsupported,
        "graph_contract_passed": True,
        "input_shape": list(MODEL_INPUT_SHAPE),
        "output_shape": list(MODEL_OUTPUT_SHAPE),
        "dynamic_axes": False,
    }


def onnx_parity(
    onnx_payload: bytes,
    features: np.ndarray,
    torch_normalized_prediction: np.ndarray,
    preprocessing: Preprocessing,
    indices: np.ndarray,
    split_label: str = "test",
) -> dict[str, Any]:
    session = ort.InferenceSession(
        onnx_payload,
        providers=["CPUExecutionProvider"],
    )
    onnx_outputs: list[np.ndarray] = []
    for index in indices:
        input_tensor = features[index].reshape(MODEL_INPUT_SHAPE)
        output = session.run(
            ["properties_normalized"],
            {"features_normalized_fp32": input_tensor},
        )[0]
        onnx_outputs.append(output.reshape(1, -1))
    onnx_normalized = np.concatenate(onnx_outputs, axis=0).astype(np.float32)
    torch_selected = torch_normalized_prediction[indices]
    normalized_delta = np.abs(onnx_normalized - torch_selected)
    raw_delta = normalized_delta * preprocessing.target_scale.reshape(1, -1)
    max_normalized = float(np.max(normalized_delta))
    max_raw_by_target = {
        spec.name: float(np.max(raw_delta[:, task_index]))
        for task_index, spec in enumerate(TARGET_SPECS)
    }
    if max_normalized > 1e-5:
        raise ValueError(f"Torch/ONNX normalized parity failed: {max_normalized}")
    return {
        "split": split_label,
        "rows": int(indices.size),
        "providers": session.get_providers(),
        "max_abs_normalized": max_normalized,
        "mean_abs_normalized": float(np.mean(normalized_delta)),
        "max_abs_raw_by_target": max_raw_by_target,
        "threshold_normalized": 1e-5,
        "passed": True,
    }


def _gzip_csv_bytes(fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> bytes:
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
            writer = csv.DictWriter(text, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    return raw.getvalue()


def _split_assignment_bytes(dataset: PreparedDataset) -> bytes:
    fieldnames = [
        "jid",
        "split",
        "reduced_formula_group",
        "approx_structure_family",
        *[f"has_{spec.name}" for spec in TARGET_SPECS],
    ]
    rows = (
        {
            "jid": dataset.jids[index],
            "split": split_name(int(dataset.split_codes[index])),
            "reduced_formula_group": dataset.formula_groups[index],
            "approx_structure_family": dataset.structure_groups[index],
            **{
                f"has_{spec.name}": int(dataset.label_mask[index, task_index])
                for task_index, spec in enumerate(TARGET_SPECS)
            },
        }
        for index in range(len(dataset.jids))
    )
    return _gzip_csv_bytes(fieldnames, rows)


def _test_prediction_bytes(
    dataset: PreparedDataset,
    predictions: dict[str, np.ndarray],
) -> bytes:
    fieldnames = ["jid", "reduced_formula_group"]
    for spec in TARGET_SPECS:
        fieldnames.extend(
            (
                f"has_{spec.name}",
                f"target_{spec.name}",
                f"propnet_{spec.name}",
                f"ridge_{spec.name}",
                f"train_mean_{spec.name}",
            )
        )
    test_indices = dataset.indices("test")

    def records() -> Iterable[dict[str, Any]]:
        for index in test_indices:
            record: dict[str, Any] = {
                "jid": dataset.jids[index],
                "reduced_formula_group": dataset.formula_groups[index],
            }
            for task_index, spec in enumerate(TARGET_SPECS):
                active = bool(dataset.label_mask[index, task_index])
                record[f"has_{spec.name}"] = int(active)
                record[f"target_{spec.name}"] = (
                    f"{float(dataset.labels[index, task_index]):.8g}" if active else ""
                )
                record[f"propnet_{spec.name}"] = (
                    f"{float(predictions['propnet_mlp'][index, task_index]):.8g}"
                )
                record[f"ridge_{spec.name}"] = (
                    f"{float(predictions['ridge'][index, task_index]):.8g}"
                )
                record[f"train_mean_{spec.name}"] = (
                    f"{float(predictions['train_mean'][index, task_index]):.8g}"
                )
            yield record

    return _gzip_csv_bytes(fieldnames, records())


def _preprocessing_bytes(preprocessing: Preprocessing) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        feature_mean=preprocessing.feature_mean,
        feature_scale=preprocessing.feature_scale,
        target_mean=preprocessing.target_mean,
        target_scale=preprocessing.target_scale,
        feature_clip_abs=np.asarray([preprocessing.feature_clip_abs], dtype=np.float32),
    )
    return stream.getvalue()


def _state_dict_bytes(model: PropNet, config: BuildConfig) -> bytes:
    stream = io.BytesIO()
    torch.save(
        {
            "schema": SCHEMA_VERSION,
            "state_dict": model.state_dict(),
            "input_dim": len(FEATURE_NAMES),
            "output_dim": len(TARGET_SPECS),
            "hidden_dims": config.hidden_dims,
            "feature_names": FEATURE_NAMES,
            "target_names": tuple(spec.name for spec in TARGET_SPECS),
        },
        stream,
    )
    return stream.getvalue()


def _quality_comparison(metrics: dict[str, Any]) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for target in PRIMARY_TARGETS:
        mlp = metrics["models"]["propnet_mlp"]["test"]["per_target"][target]["mae"]
        ridge = metrics["models"]["ridge"]["test"]["per_target"][target]["mae"]
        mean = metrics["models"]["train_mean"]["test"]["per_target"][target]["mae"]
        comparison[target] = {
            "propnet_mae": mlp,
            "ridge_mae": ridge,
            "train_mean_mae": mean,
            "improvement_vs_ridge_percent": 100.0 * (ridge - mlp) / ridge,
            "improvement_vs_train_mean_percent": 100.0 * (mean - mlp) / mean,
            "beats_ridge": mlp < ridge,
            "beats_train_mean": mlp < mean,
        }
    return {
        "primary_targets": comparison,
        "beats_train_mean_on_all_primary": all(
            record["beats_train_mean"] for record in comparison.values()
        ),
        "beats_ridge_primary_count": sum(
            record["beats_ridge"] for record in comparison.values()
        ),
        "selection_boundary": (
            "Model architecture and early stopping use train/val only. Test is evaluated "
            "once after the checkpoint is locked; no same-run test-driven retuning."
        ),
    }


def build_propnet(
    *,
    root: Path,
    source: Path,
    output: Path,
    config: BuildConfig,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    source = source.resolve()
    output = _resolve_output(root, output)
    if output.exists():
        existing = {path.name for path in output.iterdir() if path.is_file()}
        if existing and not overwrite:
            raise FileExistsError(
                f"output already contains files; pass --overwrite to replace known artifacts: {output}"
            )
        unknown = existing - set(ARTIFACT_FILENAMES)
        if unknown:
            raise ValueError(f"refusing to overwrite directory with unknown files: {sorted(unknown)}")

    _set_seed(config.seed)
    dataset = load_prepared_dataset(source)
    preprocessing = fit_preprocessing(dataset, config.feature_clip_abs)
    model_features = transform_features(dataset.features, preprocessing)
    normalized_targets = transform_targets(
        dataset.labels,
        dataset.label_mask,
        preprocessing,
    )
    device = _select_device(config.device)
    model, history, training = train_model(
        dataset,
        model_features,
        normalized_targets,
        config,
        device,
    )
    torch_normalized = _batched_torch_prediction(model, model_features)
    propnet_prediction = _raw_predictions(torch_normalized, preprocessing)
    mean_prediction, ridge_prediction, baseline_contract = baseline_predictions(
        dataset,
        model_features,
        preprocessing,
        config.ridge_alpha,
    )
    predictions = {
        "propnet_mlp": propnet_prediction,
        "ridge": ridge_prediction,
        "train_mean": mean_prediction,
    }
    metrics = evaluate_models(dataset, predictions, preprocessing)
    metrics["quality_comparison"] = _quality_comparison(metrics)

    onnx_payload, onnx_contract = export_onnx(model)
    parity = onnx_parity(
        onnx_payload,
        model_features,
        torch_normalized,
        preprocessing,
        dataset.indices("test"),
    )
    metrics["torch_onnx_parity"] = parity

    created_at = _utc_now()
    feature_contract = {
        "schema": "icmat_propnet_feature_contract.v1",
        "created_at": created_at,
        "input_dtype": "float32",
        "raw_feature_shape": [len(FEATURE_NAMES)],
        "onnx_input_shape": list(MODEL_INPUT_SHAPE),
        "feature_names": list(FEATURE_NAMES),
        "preprocessing": {
            "owner": "CPU",
            "fit_split": "train only",
            "operation": "(raw - train_mean) / train_std, clipped symmetrically",
            "clip_abs": preprocessing.feature_clip_abs,
            "artifact": "preprocessing.npz",
        },
        "composition_source": "atoms.elements counts; formula text is not parsed",
        "structure_features": (
            "nat, density, cell volume per atom, normalized lattice lengths, "
            "angle cosines, space group, and crystal-system one-hot"
        ),
        "label_fields_excluded": True,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    data_manifest = copy.deepcopy(dataset.metadata)
    data_manifest.update(
        {
            "created_at": created_at,
            "raw_source_path_relative": source.relative_to(root).as_posix(),
            "artifacts": {
                "split_assignments": "split_assignments.csv.gz",
                "feature_contract": "feature_contract.json",
                "preprocessing": "preprocessing.npz",
            },
            "network_used": False,
            "x5_contacted": False,
        }
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    state_payload = _state_dict_bytes(model, config)
    model_manifest = {
        "schema": "icmat_propnet_model_manifest.v1",
        "created_at": created_at,
        "model_id": "ICMat-PropNet",
        "status": "PC_FP32_ONNX_CANDIDATE",
        "architecture": {
            "type": "multi_task_mlp",
            "input_dim": len(FEATURE_NAMES),
            "hidden_dims": list(config.hidden_dims),
            "output_dim": len(TARGET_SPECS),
            "activations": "ReLU",
            "parameter_count": parameter_count,
            "targets": [asdict(spec) for spec in TARGET_SPECS],
            "missing_label_policy": "per-task boolean mask in training loss",
        },
        "training": {
            **asdict(config),
            **training,
            "split_selection": "group-disjoint train/val; quarantine excluded",
            "test_used_during_training": False,
            "test_used_for_model_selection": False,
        },
        "baselines": baseline_contract,
        "onnx": onnx_contract,
        "artifacts": {
            "pytorch_state": {
                "path": "model_fp32.pt",
                "bytes": len(state_payload),
                "sha256": _sha256_bytes(state_payload),
            },
            "onnx": {
                "path": "model_fp32.onnx",
                "bytes": len(onnx_payload),
                "sha256": _sha256_bytes(onnx_payload),
            },
        },
        "promotion": {
            "bpu_mapper_executed": False,
            "bpu_binary_present": False,
            "x5_replay_executed": False,
            "x5_ready": False,
            "production_integration_allowed": False,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    metrics_manifest = {
        "schema": "icmat_propnet_metrics_manifest.v1",
        "created_at": created_at,
        "status": "PC_TEST_EVALUATED_ONCE",
        "metrics": metrics,
        "units": {spec.name: spec.unit for spec in TARGET_SPECS},
        "baselines": baseline_contract,
        "test_membership_sha256": dataset.metadata["split_membership_sha256"]["test"],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    history_manifest = {
        "schema": "icmat_propnet_training_history.v1",
        "created_at": created_at,
        "history": history,
        "best_epoch": training["best_epoch"],
    }

    payloads: dict[str, bytes] = {
        "data_manifest.json": _json_bytes(data_manifest),
        "feature_contract.json": _json_bytes(feature_contract),
        "metrics_manifest.json": _json_bytes(metrics_manifest),
        "model_fp32.onnx": onnx_payload,
        "model_fp32.pt": state_payload,
        "model_manifest.json": _json_bytes(model_manifest),
        "preprocessing.npz": _preprocessing_bytes(preprocessing),
        "split_assignments.csv.gz": _split_assignment_bytes(dataset),
        "test_predictions.csv.gz": _test_prediction_bytes(dataset, predictions),
        "training_history.json": _json_bytes(history_manifest),
    }
    artifact_manifest = {
        "schema": "icmat_propnet_artifact_manifest.v1",
        "created_at": created_at,
        "status": "COMPLETE",
        "artifacts": [
            {
                "path": name,
                "bytes": len(payload),
                "sha256": _sha256_bytes(payload),
            }
            for name, payload in sorted(payloads.items())
        ],
        "artifact_count": len(payloads),
        "network_used": False,
        "x5_contacted": False,
        "production_files_modified": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payloads["artifact_manifest.json"] = _json_bytes(artifact_manifest)

    output.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        _atomic_write(output / name, payload)

    return {
        "ok": True,
        "output": output.as_posix(),
        "rows": dataset.metadata["rows"],
        "split_counts": dataset.metadata["split_counts"],
        "feature_dim": len(FEATURE_NAMES),
        "parameter_count": parameter_count,
        "training": training,
        "test_metrics": metrics["models"]["propnet_mlp"]["test"],
        "quality_comparison": metrics["quality_comparison"],
        "onnx_parity": parity,
        "artifact_manifest_sha256": _sha256_bytes(payloads["artifact_manifest.json"]),
    }
