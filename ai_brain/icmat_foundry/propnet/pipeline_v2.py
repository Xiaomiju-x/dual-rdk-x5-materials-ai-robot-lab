"""Leakage-resistant build phase for the locked ICMat-PropNet v2 candidate."""

from __future__ import annotations

import copy
import hashlib
import math
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .contracts import (
    CLAIM_BOUNDARY,
    FEATURE_NAMES,
    MODEL_INPUT_SHAPE,
    PRIMARY_TARGETS,
    TARGET_SPECS,
)
from .data import PreparedDataset, load_prepared_dataset
from .model import PropNet, masked_smooth_l1_loss
from .pipeline import (
    BuildConfig,
    Preprocessing,
    _batched_torch_prediction,
    _json_bytes,
    _preprocessing_bytes,
    _raw_predictions,
    _regression_metrics,
    _resolve_output,
    _select_device,
    _set_seed,
    _sha256_bytes,
    _split_assignment_bytes,
    _state_dict_bytes,
    export_onnx,
    fit_preprocessing,
    onnx_parity,
    transform_features,
    transform_targets,
)

CANDIDATE_ID = "icmat-propnet-task8-v2-20260728"
ARTIFACT_FILENAMES_V2 = (
    "artifact_manifest.json",
    "calibration_contract.json",
    "data_manifest.json",
    "feature_contract.json",
    "locked_metrics.json",
    "model_fp32.onnx",
    "model_fp32.pt",
    "model_manifest.json",
    "preprocessing.npz",
    "split_assignments.csv.gz",
    "training_history.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tune_score(
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
                float(
                    torch.mean(
                        torch.abs(
                            prediction[active, task_index]
                            - targets[active, task_index]
                        )
                    )
                )
            )
    if not scores:
        raise ValueError("tune split has no active targets")
    return float(np.mean(scores))


def train_locked_model(
    dataset: PreparedDataset,
    features: np.ndarray,
    normalized_targets: np.ndarray,
    config: BuildConfig,
    device: torch.device,
) -> tuple[PropNet, list[dict[str, Any]], dict[str, Any]]:
    """Fit on train and select the checkpoint on the disjoint tune split."""

    train_indices = dataset.indices("train")
    tune_indices = dataset.indices("tune")
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
    tune_features = torch.from_numpy(features[tune_indices])
    tune_targets = torch.from_numpy(normalized_targets[tune_indices])
    tune_mask = torch.from_numpy(dataset.label_mask[tune_indices])

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

        tune_score = _tune_score(
            model,
            tune_features,
            tune_targets,
            tune_mask,
            device,
        )
        scheduler.step(tune_score)
        history.append(
            {
                "epoch": epoch,
                "train_masked_smooth_l1": weighted_loss / max(seen, 1),
                "tune_macro_normalized_mae": tune_score,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if tune_score < best_score - 1e-6:
            best_score = tune_score
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
        "best_tune_macro_normalized_mae": best_score,
        "early_stopped": len(history) < config.max_epochs,
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "torch_version": torch.__version__,
    }


def _baseline_predictions_for_indices(
    dataset: PreparedDataset,
    features: np.ndarray,
    preprocessing: Preprocessing,
    indices: np.ndarray,
    ridge_alpha: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train = dataset.indices("train")
    mean_prediction = np.tile(
        preprocessing.target_mean,
        (indices.size, 1),
    ).astype(np.float32)
    ridge_prediction = np.zeros((indices.size, len(TARGET_SPECS)), dtype=np.float32)
    fit_counts: dict[str, int] = {}
    for task_index, spec in enumerate(TARGET_SPECS):
        active = train[dataset.label_mask[train, task_index]]
        model = Ridge(alpha=ridge_alpha, solver="lsqr")
        model.fit(features[active], dataset.labels[active, task_index])
        ridge_prediction[:, task_index] = model.predict(features[indices]).astype(
            np.float32
        )
        fit_counts[spec.name] = int(active.size)
    return mean_prediction, ridge_prediction, {
        "mean_baseline": "train-label mean per task",
        "ridge_baseline": {
            "algorithm": "sklearn.linear_model.Ridge",
            "alpha": ridge_alpha,
            "solver": "lsqr",
            "fit_counts": fit_counts,
            "selection_data": "train only; alpha fixed before sealed test",
        },
    }


def _metrics_for_indices(
    dataset: PreparedDataset,
    indices: np.ndarray,
    prediction: np.ndarray,
    preprocessing: Preprocessing,
) -> dict[str, Any]:
    per_target: dict[str, Any] = {}
    normalized_mae: list[float] = []
    for task_index, spec in enumerate(TARGET_SPECS):
        local_active = np.flatnonzero(dataset.label_mask[indices, task_index])
        active_indices = indices[local_active]
        metrics = _regression_metrics(
            dataset.labels[active_indices, task_index],
            prediction[local_active, task_index],
            (dataset.formula_groups[index] for index in active_indices),
        )
        metrics["unit"] = spec.unit
        per_target[spec.name] = metrics
        normalized_mae.append(
            metrics["mae"] / float(preprocessing.target_scale[task_index])
        )
    return {
        "per_target": per_target,
        "macro_normalized_mae": float(np.mean(normalized_mae)),
    }


def _group_conformal_contract(
    dataset: PreparedDataset,
    calibration_indices: np.ndarray,
    prediction: np.ndarray,
    *,
    coverage: float = 0.90,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task_index, spec in enumerate(TARGET_SPECS):
        local_active = np.flatnonzero(
            dataset.label_mask[calibration_indices, task_index]
        )
        active_indices = calibration_indices[local_active]
        residual = np.abs(
            dataset.labels[active_indices, task_index]
            - prediction[local_active, task_index]
        )
        group_scores: dict[str, float] = {}
        for index, error in zip(active_indices, residual, strict=True):
            group = dataset.formula_groups[index]
            group_scores[group] = max(group_scores.get(group, 0.0), float(error))
        scores = np.asarray(list(group_scores.values()), dtype=np.float64)
        rank = int(math.ceil((len(scores) + 1) * coverage))
        if not scores.size or rank > len(scores):
            raise ValueError(
                f"insufficient calibration formula groups for {spec.name}: "
                f"groups={len(scores)}, rank={rank}"
            )
        half_width = float(np.sort(scores)[rank - 1])
        result[spec.name] = {
            "coverage_target": coverage,
            "calibration_split": "calibration",
            "exchangeability_unit": "reduced_formula_group",
            "group_nonconformity": "maximum absolute row residual within formula group",
            "calibration_rows": int(active_indices.size),
            "calibration_groups": int(scores.size),
            "finite_sample_rank": rank,
            "half_width": half_width,
            "unit": spec.unit,
            "test_labels_accessed": False,
            "boundary": (
                "Group-aware split-conformal interval under exchangeable formula "
                "groups on the pinned public DFT protocol; not a conditional, "
                "experimental, fab-line, or deployment guarantee."
            ),
        }
    return result


def _quality_comparison(
    metrics: dict[str, dict[str, Any]],
    *,
    split: str,
) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for target in PRIMARY_TARGETS:
        mlp = metrics["propnet_mlp"][split]["per_target"][target]["mae"]
        ridge = metrics["ridge"][split]["per_target"][target]["mae"]
        mean = metrics["train_mean"][split]["per_target"][target]["mae"]
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
        "split": split,
        "selection_bias_boundary": (
            "Tune metrics are diagnostic and checkpoint-selected; they are not "
            "reported as unbiased final quality. Final quality requires the "
            "separate one-shot sealed test evaluator."
        ),
        "primary_targets": comparison,
    }


def _locked_metrics(
    dataset: PreparedDataset,
    features: np.ndarray,
    model: PropNet,
    preprocessing: Preprocessing,
    ridge_alpha: float,
) -> tuple[dict[str, Any], np.ndarray]:
    split_indices = {
        split: dataset.indices(split) for split in ("tune", "calibration")
    }
    combined = np.concatenate(tuple(split_indices.values()))
    normalized = _batched_torch_prediction(model, features[combined])
    propnet = _raw_predictions(normalized, preprocessing)
    mean, ridge, baseline_contract = _baseline_predictions_for_indices(
        dataset,
        features,
        preprocessing,
        combined,
        ridge_alpha,
    )
    prediction_by_model = {
        "propnet_mlp": propnet,
        "ridge": ridge,
        "train_mean": mean,
    }
    metrics: dict[str, dict[str, Any]] = {
        name: {} for name in prediction_by_model
    }
    offset = 0
    for split, indices in split_indices.items():
        next_offset = offset + indices.size
        for model_name, prediction in prediction_by_model.items():
            metrics[model_name][split] = _metrics_for_indices(
                dataset,
                indices,
                prediction[offset:next_offset],
                preprocessing,
            )
        offset = next_offset

    calibration_offset = split_indices["tune"].size
    calibration_prediction = propnet[calibration_offset:]
    conformal = _group_conformal_contract(
        dataset,
        split_indices["calibration"],
        calibration_prediction,
    )
    return (
        {
            "models": metrics,
            "diagnostic_quality_comparison": _quality_comparison(
                metrics,
                split="tune",
            ),
            "group_conformal_90": conformal,
            "baselines": baseline_contract,
            "test_metrics": None,
            "test_labels_accessed_for_selection": False,
        },
        calibration_prediction,
    )


def build_propnet_v2(
    *,
    root: Path,
    source: Path,
    output: Path,
    config: BuildConfig,
) -> dict[str, Any]:
    """Build one immutable pre-test candidate; no overwrite path is provided."""

    root = root.resolve()
    source = source.resolve()
    output = _resolve_output(root, output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite locked candidate: {output}")

    _set_seed(config.seed)
    dataset = load_prepared_dataset(source)
    preprocessing = fit_preprocessing(dataset, config.feature_clip_abs)
    model_features = transform_features(dataset.features, preprocessing)
    normalized_targets = transform_targets(
        dataset.labels,
        dataset.label_mask,
        preprocessing,
    )
    test_indices = dataset.indices("test")
    dataset.labels[test_indices] = np.nan
    normalized_targets[test_indices] = np.nan

    device = _select_device(config.device)
    model, history, training = train_locked_model(
        dataset,
        model_features,
        normalized_targets,
        config,
        device,
    )
    metrics, calibration_prediction = _locked_metrics(
        dataset,
        model_features,
        model,
        preprocessing,
        config.ridge_alpha,
    )
    onnx_payload, onnx_contract = export_onnx(model)
    calibration_indices = dataset.indices("calibration")
    calibration_normalized = (
        calibration_prediction - preprocessing.target_mean
    ) / preprocessing.target_scale
    parity_reference = np.full(
        (len(dataset.features), len(TARGET_SPECS)),
        np.nan,
        dtype=np.float32,
    )
    parity_reference[calibration_indices] = calibration_normalized.astype(
        np.float32
    )
    parity = onnx_parity(
        onnx_payload,
        model_features,
        parity_reference,
        preprocessing,
        calibration_indices,
        split_label="calibration",
    )

    from .pipeline import _utc_now

    created_at = _utc_now()
    feature_contract = {
        "schema": "icmat_propnet_feature_contract.v2",
        "created_at": created_at,
        "candidate_id": CANDIDATE_ID,
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
        "coordinate_semantics": (
            "JARVIS atoms.cartesian is mandatory; Cartesian coordinates are "
            "converted with coords @ inv(lattice), fractional coordinates are used directly"
        ),
        "label_fields_excluded": True,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    data_manifest = copy.deepcopy(dataset.metadata)
    data_manifest.update(
        {
            "schema": "icmat_propnet_data_contract.v2",
            "created_at": created_at,
            "candidate_id": CANDIDATE_ID,
            "raw_source_path_relative": source.relative_to(root).as_posix(),
            "test_label_policy": {
                "source_parser_loaded_version_pinned_rows": True,
                "test_label_distributions_emitted": False,
                "test_labels_redacted_before_preprocessing_training_and_evaluation": True,
                "test_metrics_emitted": False,
                "final_test_requires_separate_one_shot_sealer": True,
            },
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
        "schema": "icmat_propnet_model_manifest.v2",
        "created_at": created_at,
        "candidate_id": CANDIDATE_ID,
        "model_id": "ICMat-PropNet",
        "status": "LOCKED_PRE_TEST",
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
            "split_selection": (
                "train fit; tune checkpoint selection; calibration group-conformal; "
                "test excluded until separate one-shot seal"
            ),
            "test_used_during_training": False,
            "test_used_for_model_selection": False,
        },
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
            "test_sealed": False,
            "bpu_mapper_executed": False,
            "bpu_binary_present": False,
            "x5_replay_executed": False,
            "x5_ready": False,
            "production_integration_allowed": False,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    calibration_contract = {
        "schema": "icmat_propnet_group_conformal.v2",
        "created_at": created_at,
        "candidate_id": CANDIDATE_ID,
        "model_sha256": _sha256_bytes(state_payload),
        "calibration_membership_sha256": dataset.metadata[
            "split_membership_sha256"
        ]["calibration"],
        "intervals": metrics["group_conformal_90"],
        "test_labels_accessed": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    locked_metrics = {
        "schema": "icmat_propnet_locked_metrics.v2",
        "created_at": created_at,
        "candidate_id": CANDIDATE_ID,
        "status": "LOCKED_PRE_TEST_NO_TEST_METRICS",
        "metrics": metrics,
        "torch_onnx_parity": parity,
        "units": {spec.name: spec.unit for spec in TARGET_SPECS},
        "test_membership_sha256": dataset.metadata["split_membership_sha256"][
            "test"
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    history_manifest = {
        "schema": "icmat_propnet_training_history.v2",
        "created_at": created_at,
        "candidate_id": CANDIDATE_ID,
        "history": history,
        "best_epoch": training["best_epoch"],
    }
    payloads: dict[str, bytes] = {
        "calibration_contract.json": _json_bytes(calibration_contract),
        "data_manifest.json": _json_bytes(data_manifest),
        "feature_contract.json": _json_bytes(feature_contract),
        "locked_metrics.json": _json_bytes(locked_metrics),
        "model_fp32.onnx": onnx_payload,
        "model_fp32.pt": state_payload,
        "model_manifest.json": _json_bytes(model_manifest),
        "preprocessing.npz": _preprocessing_bytes(preprocessing),
        "split_assignments.csv.gz": _split_assignment_bytes(dataset),
        "training_history.json": _json_bytes(history_manifest),
    }
    artifact_manifest = {
        "schema": "icmat_propnet_artifact_manifest.v2",
        "created_at": created_at,
        "candidate_id": CANDIDATE_ID,
        "status": "LOCKED_PRE_TEST",
        "artifacts": [
            {
                "path": name,
                "bytes": len(payload),
                "sha256": _sha256_bytes(payload),
            }
            for name, payload in sorted(payloads.items())
        ],
        "artifact_count": len(payloads),
        "implementation_sha256": {
            "pipeline_v2.py": _sha256_file(Path(__file__)),
            "data.py": _sha256_file(Path(__file__).with_name("data.py")),
            "model.py": _sha256_file(Path(__file__).with_name("model.py")),
            "contracts.py": _sha256_file(Path(__file__).with_name("contracts.py")),
        },
        "network_used": False,
        "x5_contacted": False,
        "production_files_modified": False,
        "test_evaluated": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payloads["artifact_manifest.json"] = _json_bytes(artifact_manifest)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.building-",
        dir=output.parent,
    ) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        for name, payload in payloads.items():
            (staging / name).write_bytes(payload)
        staging.replace(output)

    return {
        "ok": True,
        "candidate_id": CANDIDATE_ID,
        "status": "LOCKED_PRE_TEST",
        "output": output.as_posix(),
        "rows": dataset.metadata["rows"],
        "split_counts": dataset.metadata["split_counts"],
        "feature_dim": len(FEATURE_NAMES),
        "parameter_count": parameter_count,
        "training": training,
        "tune_metrics_are_selection_biased": True,
        "test_evaluated": False,
        "onnx_parity": parity,
        "artifact_manifest_sha256": _sha256_bytes(
            payloads["artifact_manifest.json"]
        ),
    }
