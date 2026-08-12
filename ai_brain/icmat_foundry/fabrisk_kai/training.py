"""One-pass, non-test FabRisk-KAI model selection and promotion gate."""

from __future__ import annotations

import os
import platform
import random
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import GroupKFold

from .io_utils import read_json, sha256_file, write_json
from .model import (
    FROZEN_SEARCH_SPACE,
    ArchitectureConfig,
    TemporalRiskNet,
    TrainOnlyPreprocessor,
    parameter_count,
)
from .splitting import PARTITION_CODES

SEED = 20260728
TRAINING_SCHEMA = "fabrisk_kai_non_test_training.v1"
REQUIRED_DATA_FILES = (
    "development_temporal_values.npy",
    "development_temporal_observed_mask.npy",
    "development_summary_values.npy",
    "development_summary_observed_mask.npy",
    "development_labels.npy",
    "development_partition_codes.npy",
    "development_membership.v1.json",
    "non_test_gate_contract.v1.json",
)


@dataclass(frozen=True)
class DevelopmentData:
    temporal_values: np.ndarray
    temporal_mask: np.ndarray
    summary_values: np.ndarray
    summary_mask: np.ndarray
    labels: np.ndarray
    partition_codes: np.ndarray
    lots: np.ndarray
    loaded_files: tuple[str, ...]


def _seed_everything(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _load_development_data(dataset_dir: Path) -> DevelopmentData:
    missing = [name for name in REQUIRED_DATA_FILES if not (dataset_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing development artifacts: {missing}")
    arrays = {
        "temporal_values": np.load(
            dataset_dir / "development_temporal_values.npy",
            allow_pickle=False,
        ),
        "temporal_mask": np.load(
            dataset_dir / "development_temporal_observed_mask.npy",
            allow_pickle=False,
        ),
        "summary_values": np.load(
            dataset_dir / "development_summary_values.npy",
            allow_pickle=False,
        ),
        "summary_mask": np.load(
            dataset_dir / "development_summary_observed_mask.npy",
            allow_pickle=False,
        ),
        "labels": np.load(
            dataset_dir / "development_labels.npy",
            allow_pickle=False,
        ),
        "partition_codes": np.load(
            dataset_dir / "development_partition_codes.npy",
            allow_pickle=False,
        ),
    }
    membership = read_json(dataset_dir / "development_membership.v1.json")
    lots = np.asarray([member["lot"] for member in membership["members"]])
    rows = int(arrays["labels"].shape[0])
    if len(lots) != rows:
        raise ValueError("development membership is not row-aligned")
    if np.any(arrays["partition_codes"] >= PARTITION_CODES["test"]):
        raise ValueError("sealed-test partition code present in development cache")
    if set(np.unique(arrays["partition_codes"]).tolist()) != {
        PARTITION_CODES["train"],
        PARTITION_CODES["tune"],
        PARTITION_CODES["calibration"],
    }:
        raise ValueError("unexpected development partition set")
    return DevelopmentData(
        temporal_values=arrays["temporal_values"],
        temporal_mask=arrays["temporal_mask"],
        summary_values=arrays["summary_values"],
        summary_mask=arrays["summary_mask"],
        labels=arrays["labels"].astype(np.int64),
        partition_codes=arrays["partition_codes"],
        lots=lots,
        loaded_files=REQUIRED_DATA_FILES,
    )


def _indices(data: DevelopmentData, partition: str) -> np.ndarray:
    return np.flatnonzero(data.partition_codes == PARTITION_CODES[partition])


def _subset(
    data: DevelopmentData,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        data.temporal_values[indices],
        data.temporal_mask[indices],
        data.summary_values[indices],
        data.summary_mask[indices],
    )


def _tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array)).to(device=device)


def _focal_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    positive_weight: float,
    gamma: float,
) -> torch.Tensor:
    weights = torch.where(
        labels > 0.5,
        torch.full_like(labels, positive_weight),
        torch.ones_like(labels),
    )
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction="none",
    )
    if gamma > 0:
        probabilities = torch.sigmoid(logits)
        target_probability = torch.where(labels > 0.5, probabilities, 1 - probabilities)
        losses = losses * (1 - target_probability).pow(gamma)
    return (losses * weights).mean()


def _train_model(
    temporal_input: np.ndarray,
    summary_input: np.ndarray,
    labels: np.ndarray,
    config: ArchitectureConfig,
    *,
    seed: int,
    device: torch.device,
) -> tuple[TemporalRiskNet, dict[str, Any]]:
    _seed_everything(seed)
    model = TemporalRiskNet(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    x_temporal = _tensor(temporal_input, device)
    x_summary = _tensor(summary_input, device)
    y = _tensor(labels.astype(np.float32), device)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    positive_weight = float(negatives / max(positives, 1))
    loss_history: list[float] = []
    model.train()
    for epoch in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_temporal, x_summary)
        loss = _focal_bce(
            logits,
            y,
            positive_weight=positive_weight,
            gamma=config.focal_gamma,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if epoch in {0, config.epochs - 1} or (epoch + 1) % 12 == 0:
            loss_history.append(float(loss.detach().cpu()))
    model.eval()
    return model, {
        "seed": seed,
        "epochs_fixed": config.epochs,
        "positive_weight_train_only": positive_weight,
        "sampled_loss_history": loss_history,
        "early_stopping": False,
        "tune_metrics_observed_during_training": False,
    }


@torch.inference_mode()
def _predict_logits(
    model: TemporalRiskNet,
    temporal_input: np.ndarray,
    summary_input: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    logits = model(
        _tensor(temporal_input, device),
        _tensor(summary_input, device),
    )
    return logits.detach().cpu().numpy().astype(np.float64)


def _fit_logit_calibrator(logits: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    calibrator = LogisticRegression(
        C=1.0,
        class_weight=None,
        max_iter=2000,
        random_state=SEED,
        solver="lbfgs",
    )
    calibrator.fit(logits.reshape(-1, 1), labels)
    return calibrator


def _calibrated_probabilities(
    calibrator: LogisticRegression,
    logits: np.ndarray,
) -> np.ndarray:
    return calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]


def _select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.linspace(0.05, 0.95, 91)
    scored = [
        (
            f1_score(
                labels,
                probabilities >= threshold,
                average="macro",
                zero_division=0,
            ),
            -abs(float(threshold) - 0.5),
            -float(threshold),
            float(threshold),
        )
        for threshold in candidates
    ]
    return max(scored)[-1]


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    error = 0.0
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        if index == bins - 1:
            selected = (probabilities >= lower) & (probabilities <= upper)
        else:
            selected = (probabilities >= lower) & (probabilities < upper)
        if not selected.any():
            continue
        confidence = float(probabilities[selected].mean())
        accuracy = float(labels[selected].mean())
        error += float(selected.sum() / total) * abs(confidence - accuracy)
    return float(error)


def _metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = probabilities >= threshold
    return {
        "bad_average_precision": float(average_precision_score(labels, probabilities)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "ece_10_bin": _ece(labels, probabilities),
        "threshold_selected_on_tune": float(threshold),
    }


def _fit_logistic_baseline(
    train_summary: np.ndarray,
    train_labels: np.ndarray,
) -> LogisticRegression:
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=4000,
        random_state=SEED,
        solver="liblinear",
    )
    model.fit(train_summary, train_labels)
    return model


def _candidate_search(
    data: DevelopmentData,
    train_indices: np.ndarray,
    tune_indices: np.ndarray,
    device: torch.device,
) -> tuple[
    ArchitectureConfig,
    TemporalRiskNet,
    TrainOnlyPreprocessor,
    list[dict[str, Any]],
]:
    preprocessor = TrainOnlyPreprocessor.fit(*_subset(data, train_indices))
    train_temporal, train_summary = preprocessor.transform(*_subset(data, train_indices))
    tune_temporal, tune_summary = preprocessor.transform(*_subset(data, tune_indices))
    search_records: list[dict[str, Any]] = []
    trained: list[TemporalRiskNet] = []
    for candidate_index, config in enumerate(FROZEN_SEARCH_SPACE):
        model, audit = _train_model(
            train_temporal,
            train_summary,
            data.labels[train_indices],
            config,
            seed=SEED + candidate_index,
            device=device,
        )
        tune_logits = _predict_logits(model, tune_temporal, tune_summary, device)
        tune_ap = float(average_precision_score(data.labels[tune_indices], tune_logits))
        search_records.append(
            {
                "candidate": config.to_dict(),
                "parameter_count": parameter_count(model),
                "training": audit,
                "tune_bad_average_precision": tune_ap,
            }
        )
        trained.append(model)
    selected_index = max(
        range(len(search_records)),
        key=lambda index: (
            search_records[index]["tune_bad_average_precision"],
            -index,
        ),
    )
    return (
        FROZEN_SEARCH_SPACE[selected_index],
        trained[selected_index],
        preprocessor,
        search_records,
    )


def _group_kfold_stability(
    data: DevelopmentData,
    train_indices: np.ndarray,
    config: ArchitectureConfig,
    device: torch.device,
) -> list[dict[str, Any]]:
    splitter = GroupKFold(n_splits=5)
    records: list[dict[str, Any]] = []
    train_lots = data.lots[train_indices]
    train_labels = data.labels[train_indices]
    for fold, (relative_fit, relative_validation) in enumerate(
        splitter.split(train_indices, train_labels, groups=train_lots),
        start=1,
    ):
        fit_indices = train_indices[relative_fit]
        validation_indices = train_indices[relative_validation]
        preprocessor = TrainOnlyPreprocessor.fit(*_subset(data, fit_indices))
        fit_temporal, fit_summary = preprocessor.transform(*_subset(data, fit_indices))
        validation_temporal, validation_summary = preprocessor.transform(
            *_subset(data, validation_indices)
        )
        model, training_audit = _train_model(
            fit_temporal,
            fit_summary,
            data.labels[fit_indices],
            config,
            seed=SEED + 100 + fold,
            device=device,
        )
        cnn_scores = _predict_logits(
            model,
            validation_temporal,
            validation_summary,
            device,
        )
        logistic = _fit_logistic_baseline(
            fit_summary,
            data.labels[fit_indices],
        )
        logistic_scores = logistic.predict_proba(validation_summary)[:, 1]
        cnn_ap = float(
            average_precision_score(data.labels[validation_indices], cnn_scores)
        )
        logistic_ap = float(
            average_precision_score(
                data.labels[validation_indices],
                logistic_scores,
            )
        )
        records.append(
            {
                "fold": fold,
                "fit_rows": int(len(fit_indices)),
                "validation_rows": int(len(validation_indices)),
                "fit_lots": sorted(set(data.lots[fit_indices].tolist())),
                "validation_lots": sorted(
                    set(data.lots[validation_indices].tolist())
                ),
                "cnn_bad_average_precision": cnn_ap,
                "logistic_bad_average_precision": logistic_ap,
                "cnn_margin": cnn_ap - logistic_ap,
                "cnn_win": cnn_ap > logistic_ap,
                "training": training_audit,
            }
        )
    return records


def _gate_decision(
    contract: dict[str, Any],
    *,
    tune_margin: float,
    calibration_margin: float,
    fold_records: list[dict[str, Any]],
    calibration_macro_f1_margin: float,
    calibration_mcc_margin: float,
    calibrated_ece: float,
) -> dict[str, Any]:
    stability = contract["stability_requirements"]
    secondary = contract["secondary_non_inferiority_constraints"]
    fold_mean_margin = float(
        np.mean([record["cnn_margin"] for record in fold_records])
    )
    fold_wins = int(sum(record["cnn_win"] for record in fold_records))
    checks = {
        "tune_ap_margin": {
            "actual": tune_margin,
            "required_minimum": stability["minimum_tune_ap_margin"],
            "passed": tune_margin >= stability["minimum_tune_ap_margin"],
        },
        "calibration_ap_margin": {
            "actual": calibration_margin,
            "required_minimum": stability["minimum_calibration_ap_margin"],
            "passed": calibration_margin
            >= stability["minimum_calibration_ap_margin"],
        },
        "group_kfold_mean_ap_margin": {
            "actual": fold_mean_margin,
            "required_minimum": stability["minimum_mean_group_kfold_ap_margin"],
            "passed": fold_mean_margin
            >= stability["minimum_mean_group_kfold_ap_margin"],
        },
        "group_kfold_cnn_wins": {
            "actual": fold_wins,
            "required_minimum": stability[
                "minimum_cnn_fold_wins_over_logistic"
            ],
            "passed": fold_wins
            >= stability["minimum_cnn_fold_wins_over_logistic"],
        },
        "calibration_macro_f1_margin": {
            "actual": calibration_macro_f1_margin,
            "required_minimum": secondary[
                "minimum_calibration_macro_f1_margin"
            ],
            "passed": calibration_macro_f1_margin
            >= secondary["minimum_calibration_macro_f1_margin"],
        },
        "calibration_mcc_margin": {
            "actual": calibration_mcc_margin,
            "required_minimum": secondary["minimum_calibration_mcc_margin"],
            "passed": calibration_mcc_margin
            >= secondary["minimum_calibration_mcc_margin"],
        },
        "calibrated_ece": {
            "actual": calibrated_ece,
            "required_maximum": secondary["maximum_calibrated_ece"],
            "passed": calibrated_ece <= secondary["maximum_calibrated_ece"],
        },
    }
    passed = all(check["passed"] for check in checks.values())
    return {
        "schema": "fabrisk_kai_non_test_gate_decision.v1",
        "passed": passed,
        "status": contract["pass_status"] if passed else contract["hold_status"],
        "checks": checks,
        "sealed_test_opened": False,
        "test_metrics_computed": False,
        "onnx_exported": False,
        "horizon_mapper_run": False,
        "next_action": (
            "READY_FOR_INDEPENDENT_AUDIT"
            if passed
            else "HOLD; do not open test, export ONNX, or run mapper"
        ),
    }


def _write_search_contract(path: Path, dataset_dir: Path) -> None:
    write_json(
        path,
        {
            "schema": "fabrisk_kai_preregistered_search.v1",
            "seed": SEED,
            "created_before_dataset_labels_loaded": True,
            "single_pass_only": True,
            "search_may_not_expand_after_metrics": True,
            "selection_partition": "tune",
            "candidates": [config.to_dict() for config in FROZEN_SEARCH_SPACE],
            "group_kfold": {
                "splits": 5,
                "partition": "train only",
                "architecture": "single tune-selected candidate",
            },
            "baseline": {
                "logistic": (
                    "class-balanced logistic regression on the same train-only "
                    "normalized 50 summary values plus 50 observed-mask channels"
                ),
                "class_prior": "bad prevalence fitted from the active training rows",
            },
            "dataset_dir": str(dataset_dir.resolve()),
            "prohibited": [
                "read sealed test features or labels",
                "compute test metrics",
                "expand search after tune/calibration metrics",
                "export ONNX or run Horizon mapper before independent audit",
            ],
        },
    )


def run_non_test_training(
    dataset_dir: Path,
    output_dir: Path,
    *,
    device_name: str = "cuda",
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite prior run: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.with_name(f".{output_dir.name}.tmp-{uuid.uuid4().hex}")
    temporary.mkdir()
    started = time.time()
    try:
        _write_search_contract(
            temporary / "preregistered_search_space.v1.json",
            dataset_dir,
        )
        data = _load_development_data(dataset_dir)
        gate_contract = read_json(dataset_dir / "non_test_gate_contract.v1.json")
        if gate_contract.get("test_access_allowed") is not False:
            raise ValueError("non-test gate contract unexpectedly permits test access")
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device(device_name)
        train_indices = _indices(data, "train")
        tune_indices = _indices(data, "tune")
        calibration_indices = _indices(data, "calibration")
        selected_config, selected_model, preprocessor, search_records = (
            _candidate_search(data, train_indices, tune_indices, device)
        )
        train_temporal, train_summary = preprocessor.transform(
            *_subset(data, train_indices)
        )
        tune_temporal, tune_summary = preprocessor.transform(
            *_subset(data, tune_indices)
        )
        calibration_temporal, calibration_summary = preprocessor.transform(
            *_subset(data, calibration_indices)
        )

        tune_logits = _predict_logits(
            selected_model,
            tune_temporal,
            tune_summary,
            device,
        )
        calibration_logits = _predict_logits(
            selected_model,
            calibration_temporal,
            calibration_summary,
            device,
        )
        calibrator = _fit_logit_calibrator(
            tune_logits,
            data.labels[tune_indices],
        )
        tune_probabilities = _calibrated_probabilities(calibrator, tune_logits)
        calibration_probabilities = _calibrated_probabilities(
            calibrator,
            calibration_logits,
        )
        threshold = _select_threshold(
            data.labels[tune_indices],
            tune_probabilities,
        )
        model_tune_metrics = _metrics(
            data.labels[tune_indices],
            tune_probabilities,
            threshold,
        )
        model_calibration_metrics = _metrics(
            data.labels[calibration_indices],
            calibration_probabilities,
            threshold,
        )

        logistic = _fit_logistic_baseline(
            train_summary,
            data.labels[train_indices],
        )
        logistic_tune_probabilities = logistic.predict_proba(tune_summary)[:, 1]
        logistic_calibration_probabilities = logistic.predict_proba(
            calibration_summary
        )[:, 1]
        logistic_threshold = _select_threshold(
            data.labels[tune_indices],
            logistic_tune_probabilities,
        )
        logistic_tune_metrics = _metrics(
            data.labels[tune_indices],
            logistic_tune_probabilities,
            logistic_threshold,
        )
        logistic_calibration_metrics = _metrics(
            data.labels[calibration_indices],
            logistic_calibration_probabilities,
            logistic_threshold,
        )
        prior = float(data.labels[train_indices].mean())
        class_prior_metrics = {
            "fitted_bad_prevalence": prior,
            "tune_bad_average_precision": float(
                average_precision_score(
                    data.labels[tune_indices],
                    np.full(len(tune_indices), prior),
                )
            ),
            "calibration_bad_average_precision": float(
                average_precision_score(
                    data.labels[calibration_indices],
                    np.full(len(calibration_indices), prior),
                )
            ),
        }

        fold_records = _group_kfold_stability(
            data,
            train_indices,
            selected_config,
            device,
        )
        tune_margin = (
            model_tune_metrics["bad_average_precision"]
            - logistic_tune_metrics["bad_average_precision"]
        )
        calibration_margin = (
            model_calibration_metrics["bad_average_precision"]
            - logistic_calibration_metrics["bad_average_precision"]
        )
        gate_decision = _gate_decision(
            gate_contract,
            tune_margin=tune_margin,
            calibration_margin=calibration_margin,
            fold_records=fold_records,
            calibration_macro_f1_margin=(
                model_calibration_metrics["macro_f1"]
                - logistic_calibration_metrics["macro_f1"]
            ),
            calibration_mcc_margin=(
                model_calibration_metrics["mcc"]
                - logistic_calibration_metrics["mcc"]
            ),
            calibrated_ece=model_calibration_metrics["ece_10_bin"],
        )

        np.savez(
            temporary / "train_only_preprocessing.v1.npz",
            **preprocessor.arrays(),
        )
        torch.save(
            {
                "schema": "fabrisk_kai_checkpoint.v1",
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in selected_model.state_dict().items()
                },
                "architecture": selected_config.to_dict(),
                "parameter_count": parameter_count(selected_model),
                "seed": SEED,
                "status": gate_decision["status"],
                "test_opened": False,
            },
            temporary / "selected_model_candidate.pt",
        )
        write_json(
            temporary / "non_test_metrics.v1.json",
            {
                "schema": "fabrisk_kai_non_test_metrics.v1",
                "selected_candidate": selected_config.to_dict(),
                "parameter_count": parameter_count(selected_model),
                "tune": {
                    "model": model_tune_metrics,
                    "logistic": logistic_tune_metrics,
                    "model_ap_margin": tune_margin,
                },
                "calibration": {
                    "model": model_calibration_metrics,
                    "logistic": logistic_calibration_metrics,
                    "model_ap_margin": calibration_margin,
                    "model_macro_f1_margin": (
                        model_calibration_metrics["macro_f1"]
                        - logistic_calibration_metrics["macro_f1"]
                    ),
                    "model_mcc_margin": (
                        model_calibration_metrics["mcc"]
                        - logistic_calibration_metrics["mcc"]
                    ),
                },
                "class_prior": class_prior_metrics,
                "group_kfold_train_only": fold_records,
            },
        )
        write_json(temporary / "gate_decision.v1.json", gate_decision)
        write_json(
            temporary / "training_selection_audit.v1.json",
            {
                "schema": TRAINING_SCHEMA,
                "seed": SEED,
                "device": str(device),
                "torch_version": torch.__version__,
                "sklearn_version": sklearn.__version__,
                "python_version": sys.version,
                "platform": platform.platform(),
                "elapsed_seconds": time.time() - started,
                "loaded_dataset_files": list(data.loaded_files),
                "sealed_test_files_loaded": [],
                "test_metrics_computed": False,
                "partitions": {
                    "train_rows": int(len(train_indices)),
                    "tune_rows": int(len(tune_indices)),
                    "calibration_rows": int(len(calibration_indices)),
                },
                "preprocessing": {
                    "fit_partition": "train only",
                    "missing_value_policy": (
                        "replace missing values with train-only feature median "
                        "after preserving a separate observed-mask channel; "
                        "imputed numeric zero is normalized-median encoding, "
                        "never an observed zero"
                    ),
                    "normalized_clip": 8.0,
                },
                "search_records": search_records,
                "selected_candidate_id": selected_config.candidate_id,
                "selection_executions": 1,
                "calibration_executions": 1,
                "search_expanded_after_metrics": False,
                "probability_calibration": (
                    "one-dimensional logistic calibrator fitted on tune logits"
                ),
                "threshold_selection": (
                    "fixed 0.05..0.95 grid on tune macro-F1"
                ),
                "operator_intent": [
                    "Conv1d",
                    "ReLU",
                    "ReduceMean",
                    "ReduceMax",
                    "Concat",
                    "Gemm",
                ],
                "bpu_binary_compiled": False,
                "x5_executed": False,
            },
        )
        artifact_names = sorted(
            path.name
            for path in temporary.iterdir()
            if path.is_file()
        )
        write_json(
            temporary / "artifact_manifest.v1.json",
            {
                "schema": "fabrisk_kai_non_test_artifacts.v1",
                "status": gate_decision["status"],
                "immutable_candidate": True,
                "dataset_inputs": {
                    name: {
                        "sha256": sha256_file(dataset_dir / name),
                        "bytes": (dataset_dir / name).stat().st_size,
                    }
                    for name in REQUIRED_DATA_FILES
                },
                "artifacts": {
                    name: {
                        "sha256": sha256_file(temporary / name),
                        "bytes": (temporary / name).stat().st_size,
                    }
                    for name in artifact_names
                },
                "sealed_test_opened": False,
                "test_metrics_computed": False,
                "onnx_exported": False,
                "horizon_mapper_run": False,
            },
        )
        os.replace(temporary, output_dir)
        return {
            "status": gate_decision["status"],
            "output_dir": str(output_dir),
            "selected_candidate": selected_config.candidate_id,
            "parameter_count": parameter_count(selected_model),
            "gate_checks": gate_decision["checks"],
            "manifest_sha256": sha256_file(
                output_dir / "artifact_manifest.v1.json"
            ),
        }
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def verify_non_test_run(output_dir: Path) -> dict[str, Any]:
    manifest = read_json(output_dir / "artifact_manifest.v1.json")
    if manifest.get("schema") != "fabrisk_kai_non_test_artifacts.v1":
        raise ValueError("unexpected FabRisk non-test manifest schema")
    for name, expected in manifest["artifacts"].items():
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != expected["sha256"]:
            raise ValueError(f"artifact hash mismatch: {name}")
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"artifact size mismatch: {name}")
    generated_names = [path.name.lower() for path in output_dir.iterdir()]
    forbidden_names = [
        name
        for name in generated_names
        if name.endswith((".onnx", ".bin"))
        or "mapper" in name
        or name.startswith(("sealed_test_", "test_metrics."))
    ]
    if forbidden_names:
        raise ValueError("forbidden post-gate artifact found")
    decision = read_json(output_dir / "gate_decision.v1.json")
    if decision["sealed_test_opened"] or decision["test_metrics_computed"]:
        raise ValueError("sealed-test boundary violated")
    return {
        "schema": "fabrisk_kai_non_test_verification.v1",
        "status": manifest["status"],
        "artifacts_verified": len(manifest["artifacts"]),
        "manifest_sha256": sha256_file(
            output_dir / "artifact_manifest.v1.json"
        ),
        "sealed_test_opened": False,
        "test_metrics_computed": False,
    }
