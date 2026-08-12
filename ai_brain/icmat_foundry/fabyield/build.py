"""End-to-end FabYield-X5 local CPU/ONNX candidate builder."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime
import sklearn
import torch

from .data import (
    SecomDataset,
    load_secom_zip,
    sha256_file,
    temporal_batch_split,
    validation_calibration_policy_split,
)
from .metrics import (
    classification_metrics,
    expected_calibration_error,
    select_decision_threshold,
    select_reject_margin,
    selective_metrics,
    sigmoid,
    stratified_bootstrap_intervals,
    temporal_batch_metrics,
)
from .model import (
    FabYieldMLP,
    PlattCalibrator,
    TrainingConfig,
    export_onnx,
    torch_logits,
    train_balanced_logistic,
    train_mlp,
    verify_onnx_parity,
)
from .preprocessing import LeakageSafePreprocessor

SCHEMA = "fabyield_x5_candidate_manifest.v1"


@dataclass(frozen=True)
class BuildConfig:
    seed: int = 20260728
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    max_features: int = 128
    hidden_dims: tuple[int, ...] = (64, 16)
    epochs: int = 120
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    minimum_selective_coverage: float = 0.80
    bootstrap_repeats: int = 500
    torch_threads: int = 1

    def training_config(self) -> TrainingConfig:
        return TrainingConfig(
            seed=self.seed,
            hidden_dims=self.hidden_dims,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            torch_threads=self.torch_threads,
        )


@dataclass
class FittedCandidate:
    preprocessor: LeakageSafePreprocessor
    mlp: FabYieldMLP
    logistic: Any
    mlp_calibrator: PlattCalibrator
    logistic_calibrator: PlattCalibrator
    mlp_threshold: dict[str, Any]
    mlp_reject: dict[str, Any]
    logistic_threshold: dict[str, Any]
    training_history: list[dict[str, float]]
    imbalance: dict[str, float]
    calibration_indices: np.ndarray
    policy_indices: np.ndarray


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _save_torch_state(model: FabYieldMLP, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(model.state_dict(), temporary)
    temporary.replace(path)


def _state_fingerprint(model: FabYieldMLP) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _rows_digest(indices: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(indices, dtype="<i8").tobytes()
    ).hexdigest()


def fit_candidate(
    dataset: SecomDataset,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    config: BuildConfig,
) -> FittedCandidate:
    """Fit all learned state without receiving test rows or test labels."""
    preprocessor = LeakageSafePreprocessor(max_features=config.max_features)
    preprocessor.fit(
        dataset.features[train_indices],
        dataset.labels[train_indices],
    )
    train_features = preprocessor.transform(dataset.features[train_indices])
    validation_features = preprocessor.transform(dataset.features[validation_indices])
    train_labels = dataset.labels[train_indices]

    logistic = train_balanced_logistic(
        train_features,
        train_labels,
        seed=config.seed,
    )
    mlp, history, imbalance = train_mlp(
        train_features,
        train_labels,
        config.training_config(),
    )

    calibration_indices, policy_indices = validation_calibration_policy_split(
        dataset.timestamps,
        dataset.labels,
        validation_indices,
    )
    validation_position = {
        int(row_index): position
        for position, row_index in enumerate(validation_indices)
    }
    calibration_positions = np.asarray(
        [validation_position[int(index)] for index in calibration_indices],
        dtype=np.int64,
    )
    policy_positions = np.asarray(
        [validation_position[int(index)] for index in policy_indices],
        dtype=np.int64,
    )

    validation_mlp_logits = torch_logits(mlp, validation_features)
    validation_logistic_logits = logistic.decision_function(validation_features)
    calibration_labels = dataset.labels[calibration_indices]
    policy_labels = dataset.labels[policy_indices]

    mlp_calibrator = PlattCalibrator.fit(
        validation_mlp_logits[calibration_positions],
        calibration_labels,
        seed=config.seed,
    )
    logistic_calibrator = PlattCalibrator.fit(
        validation_logistic_logits[calibration_positions],
        calibration_labels,
        seed=config.seed,
    )
    mlp_policy_probabilities = mlp_calibrator.transform(
        validation_mlp_logits[policy_positions]
    )
    logistic_policy_probabilities = logistic_calibrator.transform(
        validation_logistic_logits[policy_positions]
    )
    mlp_threshold = select_decision_threshold(
        policy_labels, mlp_policy_probabilities
    )
    logistic_threshold = select_decision_threshold(
        policy_labels, logistic_policy_probabilities
    )
    mlp_reject = select_reject_margin(
        policy_labels,
        mlp_policy_probabilities,
        threshold=float(mlp_threshold["threshold"]),
        minimum_coverage=config.minimum_selective_coverage,
    )
    return FittedCandidate(
        preprocessor=preprocessor,
        mlp=mlp,
        logistic=logistic,
        mlp_calibrator=mlp_calibrator,
        logistic_calibrator=logistic_calibrator,
        mlp_threshold=mlp_threshold,
        mlp_reject=mlp_reject,
        logistic_threshold=logistic_threshold,
        training_history=history,
        imbalance=imbalance,
        calibration_indices=calibration_indices,
        policy_indices=policy_indices,
    )


def fitted_candidate_fingerprint(candidate: FittedCandidate) -> dict[str, Any]:
    """Fingerprint only learned state and validation-selected policy."""
    return {
        "preprocessor_sha256": candidate.preprocessor.state_sha256(),
        "mlp_state_sha256": _state_fingerprint(candidate.mlp),
        "logistic_coefficients_sha256": hashlib.sha256(
            np.ascontiguousarray(candidate.logistic.coef_, dtype="<f8").tobytes()
            + np.ascontiguousarray(candidate.logistic.intercept_, dtype="<f8").tobytes()
        ).hexdigest(),
        "mlp_calibration": candidate.mlp_calibrator.as_dict(),
        "mlp_threshold": candidate.mlp_threshold,
        "mlp_reject": candidate.mlp_reject,
        "logistic_calibration": candidate.logistic_calibrator.as_dict(),
        "logistic_threshold": candidate.logistic_threshold,
    }


def _model_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    calibrator: PlattCalibrator,
    threshold: dict[str, Any],
    reject: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_probabilities = sigmoid(logits)
    calibrated_probabilities = calibrator.transform(logits)
    decision_threshold = float(threshold["threshold"])
    raw_brier = float(np.mean(np.square(raw_probabilities - labels)))
    calibrated_brier = float(
        np.mean(np.square(calibrated_probabilities - labels))
    )
    raw_ece = expected_calibration_error(labels, raw_probabilities)
    calibrated_ece = expected_calibration_error(labels, calibrated_probabilities)
    result = {
        "raw_probability_at_0_5": classification_metrics(
            labels, raw_probabilities, threshold=0.5
        ),
        "calibrated_probability_selected_threshold": classification_metrics(
            labels,
            calibrated_probabilities,
            threshold=decision_threshold,
        ),
        "calibration_delta": {
            "brier_calibrated_minus_raw": calibrated_brier - raw_brier,
            "ece_calibrated_minus_raw": calibrated_ece - raw_ece,
            "brier_improved": calibrated_brier < raw_brier,
            "ece_improved": calibrated_ece < raw_ece,
            "negative_delta_means_improved": True,
        },
    }
    if reject is not None:
        result["selective"] = selective_metrics(
            labels,
            calibrated_probabilities,
            threshold=decision_threshold,
            reject_margin=float(reject["margin"]),
        )
    return result


def _artifact_inventory(output_dir: Path, names: list[str]) -> list[dict[str, Any]]:
    records = []
    for name in sorted(names):
        path = output_dir / name
        records.append(
            {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def build_fabyield_candidate(
    source_zip: Path,
    output_dir: Path,
    config: BuildConfig | None = None,
) -> dict[str, Any]:
    config = config or BuildConfig()
    source_zip = source_zip.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_secom_zip(source_zip)
    split = temporal_batch_split(
        dataset.timestamps,
        dataset.labels,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )
    candidate = fit_candidate(
        dataset,
        split.train,
        split.validation,
        config,
    )
    test_features = candidate.preprocessor.transform(dataset.features[split.test])
    test_labels = dataset.labels[split.test]
    test_mlp_logits = torch_logits(candidate.mlp, test_features)
    test_logistic_logits = candidate.logistic.decision_function(test_features)

    mlp_metrics = _model_metrics(
        test_labels,
        test_mlp_logits,
        candidate.mlp_calibrator,
        candidate.mlp_threshold,
        candidate.mlp_reject,
    )
    logistic_metrics = _model_metrics(
        test_labels,
        test_logistic_logits,
        candidate.logistic_calibrator,
        candidate.logistic_threshold,
    )
    always_pass_probabilities = np.zeros(test_labels.size, dtype=np.float64)
    baseline_metrics = {
        "always_predict_pass": classification_metrics(
            test_labels,
            always_pass_probabilities,
            threshold=0.5,
        ),
        "balanced_logistic": logistic_metrics,
    }
    calibrated_test_probabilities = candidate.mlp_calibrator.transform(
        test_mlp_logits
    )
    test_batch_ids = dataset.batch_ids[split.test]
    mlp_metrics["calendar_batch_proxy"] = temporal_batch_metrics(
        test_labels,
        calibrated_test_probabilities,
        test_batch_ids,
        threshold=float(candidate.mlp_threshold["threshold"]),
    )
    bootstrap = stratified_bootstrap_intervals(
        test_labels,
        calibrated_test_probabilities,
        threshold=float(candidate.mlp_threshold["threshold"]),
        repeats=config.bootstrap_repeats,
        seed=config.seed,
    )

    onnx_path = output_dir / "fabyield_x5_mlp.onnx"
    export_onnx(
        candidate.mlp,
        onnx_path,
        input_features=candidate.preprocessor.output_features,
    )
    parity = verify_onnx_parity(
        candidate.mlp,
        onnx_path,
        test_features,
    )
    if not parity["passed"]:
        raise RuntimeError(f"ONNX parity failed: {parity}")

    split_manifest = split.as_dict(dataset)
    split_manifest["validation_subsets"] = {
        "calibration": {
            "role": "fit_platt_calibration_only",
            "rows": int(candidate.calibration_indices.size),
            "failures": int(np.sum(dataset.labels[candidate.calibration_indices] == 1)),
            "first_timestamp": min(
                dataset.timestamps[int(index)]
                for index in candidate.calibration_indices
            ).isoformat(),
            "last_timestamp": max(
                dataset.timestamps[int(index)]
                for index in candidate.calibration_indices
            ).isoformat(),
            "source_row_ids_sha256": _rows_digest(
                dataset.source_row_ids[candidate.calibration_indices]
            ),
        },
        "policy": {
            "role": "select_decision_threshold_and_reject_margin_only",
            "rows": int(candidate.policy_indices.size),
            "failures": int(np.sum(dataset.labels[candidate.policy_indices] == 1)),
            "first_timestamp": min(
                dataset.timestamps[int(index)] for index in candidate.policy_indices
            ).isoformat(),
            "last_timestamp": max(
                dataset.timestamps[int(index)] for index in candidate.policy_indices
            ).isoformat(),
            "source_row_ids_sha256": _rows_digest(
                dataset.source_row_ids[candidate.policy_indices]
            ),
        },
    }
    split_manifest["test_usage"] = (
        "single final evaluation and ONNX parity only; no fitting, feature selection, "
        "calibration, threshold selection, rejection policy selection, or early stopping"
    )

    data_audit = {
        "schema": "fabyield_secom_data_audit.v1",
        "source": {
            "path": str(source_zip),
            "sha256": dataset.source_sha256,
            "doi": "10.24432/C54305",
            "license": "CC BY 4.0",
            "rows": int(dataset.features.shape[0]),
            "anonymous_sensor_features": int(dataset.features.shape[1]),
            "failure_rows": int(np.sum(dataset.labels == 1)),
            "pass_rows": int(np.sum(dataset.labels == 0)),
            "failure_prevalence": float(np.mean(dataset.labels == 1)),
            "missing_fraction": float(np.mean(np.isnan(dataset.features))),
            "first_timestamp": min(dataset.timestamps).isoformat(),
            "last_timestamp": max(dataset.timestamps).isoformat(),
            "source_order_monotonic": dataset.source_order_monotonic,
        },
        "risk_flags": [
            "anonymous_features",
            "legacy_2008_unknown_process",
            "severe_class_imbalance",
            "no_true_wafer_or_run_identifier",
            "calendar_date_used_as_batch_proxy",
            "public_benchmark_not_local_or_modern_fab_ground_truth",
        ],
    }
    preprocessor_manifest = candidate.preprocessor.manifest()
    policy = {
        "schema": "fabyield_calibration_and_reject_policy.v1",
        "positive_class": "manufacturing_failure_original_label_plus_1",
        "mlp_calibration": candidate.mlp_calibrator.as_dict(),
        "mlp_decision": candidate.mlp_threshold,
        "mlp_rejection": candidate.mlp_reject,
        "logistic_calibration": candidate.logistic_calibrator.as_dict(),
        "logistic_decision": candidate.logistic_threshold,
        "test_labels_consulted": False,
        "policy_semantics": (
            "REVIEW is an uncertainty output for a researcher-selected benchmark tool. "
            "It never blocks Dashboard, production services, instruments, or robots."
        ),
    }
    mlp_test_summary = mlp_metrics[
        "calibrated_probability_selected_threshold"
    ]
    logistic_test_summary = logistic_metrics[
        "calibrated_probability_selected_threshold"
    ]
    quality_checks = {
        "auprc_above_test_prevalence_no_skill": bool(
            mlp_test_summary["auprc_primary"]
            > mlp_test_summary["failure_prevalence"]
        ),
        "balanced_accuracy_above_no_skill": bool(
            mlp_test_summary["balanced_accuracy_primary"] > 0.5
        ),
        "auprc_beats_balanced_logistic": bool(
            mlp_test_summary["auprc_primary"]
            > logistic_test_summary["auprc_primary"]
        ),
        "onnx_parity_passed": bool(parity["passed"]),
    }
    quality_gate_passed = all(quality_checks.values())
    quality_gate = {
        "passed": quality_gate_passed,
        "checks": quality_checks,
        "policy": (
            "Promotion requires PR-AUC above the locked test prevalence, balanced "
            "accuracy above 0.5, PR-AUC above the balanced logistic baseline, and "
            "ONNX parity. The locked test is not reused to tune this failed candidate."
        ),
        "recommendation": (
            "ELIGIBLE_FOR_LATER_BPU_CONVERSION"
            if quality_gate_passed
            else "FREEZE_AS_REPRODUCIBLE_BASELINE_DO_NOT_BPU_CONVERT"
        ),
    }
    metrics = {
        "schema": "fabyield_metrics.v1",
        "primary_metric_policy": (
            "AUPRC, balanced accuracy, macro F1 and failure recall are primary. "
            "Accuracy is reported only as a supplemental diagnostic."
        ),
        "test_partition_locked": True,
        "test_failure_rows": int(np.sum(test_labels == 1)),
        "mlp": mlp_metrics,
        "baselines": baseline_metrics,
        "bootstrap_intervals": bootstrap,
        "quality_gate": quality_gate,
        "claim_boundary": (
            "These results measure a chronological holdout of the public 2008 anonymous "
            "UCI SECOM benchmark. They do not establish local-fab, modern-fab, production, "
            "BPU, or RDK X5 performance."
        ),
    }

    candidate.preprocessor.save_npz(output_dir / "preprocessor.npz")
    _save_torch_state(candidate.mlp, output_dir / "fabyield_x5_mlp_state.pt")
    _write_json(output_dir / "data_audit.json", data_audit)
    _write_json(output_dir / "split_manifest.json", split_manifest)
    _write_json(output_dir / "preprocessor_manifest.json", preprocessor_manifest)
    _write_json(output_dir / "training_history.json", candidate.training_history)
    _write_json(output_dir / "calibration_policy.json", policy)
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "onnx_parity.json", parity)
    _write_json(
        output_dir / "logistic_baseline.json",
        {
            "schema": "fabyield_logistic_baseline.v1",
            "class_weight": "balanced",
            "coefficient_shape": list(candidate.logistic.coef_.shape),
            "coefficients": candidate.logistic.coef_.reshape(-1).tolist(),
            "intercept": candidate.logistic.intercept_.tolist(),
            "calibration": candidate.logistic_calibrator.as_dict(),
            "threshold": candidate.logistic_threshold,
        },
    )

    readme = """# FabYield-X5 local candidate

This directory is a reproducible CPU/ONNX candidate built from the public UCI
SECOM benchmark. It is not connected to Dashboard, any service, an RDK X5, or a
real fab. The ONNX graph consumes CPU-preprocessed `[1, 1, 1, K]` float32
features and emits a raw failure logit. Platt calibration, decision threshold,
and optional REVIEW rejection remain explicit CPU-side policy artifacts.

Primary metrics are AUPRC, balanced accuracy, macro F1, and failure recall.
Accuracy is supplemental because the benchmark is severely imbalanced.
Read `metrics.json` and `manifest.json` before promotion. A failed quality gate
means the model remains a research baseline and must not be converted for BPU
deployment merely because ONNX parity passed.
"""
    _write_text(output_dir / "README.md", readme)

    artifact_names = [
        "README.md",
        "calibration_policy.json",
        "data_audit.json",
        "fabyield_x5_mlp.onnx",
        "fabyield_x5_mlp_state.pt",
        "logistic_baseline.json",
        "metrics.json",
        "onnx_parity.json",
        "preprocessor.npz",
        "preprocessor_manifest.json",
        "split_manifest.json",
        "training_history.json",
    ]
    manifest = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "LOCAL_CPU_ONNX_CANDIDATE_QUALITY_GATE_PASSED_NOT_DEPLOYED"
            if quality_gate_passed
            else "LOCAL_CPU_ONNX_CANDIDATE_QUALITY_GATE_FAILED_NOT_DEPLOYED"
        ),
        "model_id": "FabYield-X5",
        "task": "public_semiconductor_process_quality_benchmark",
        "production_integration_allowed": False,
        "dashboard_modified": False,
        "service_modified": False,
        "network_used": False,
        "device_contacted": False,
        "x5_contacted": False,
        "bpu_compiled": False,
        "actual_bpu_backend_tested": False,
        "source": data_audit["source"],
        "config": asdict(config),
        "learned_state_fingerprint": fitted_candidate_fingerprint(candidate),
        "split_checks": split_manifest["checks"],
        "preprocessing_fit_scope": "train_partition_only",
        "calibration_fit_scope": "earlier_validation_batches_only",
        "decision_policy_fit_scope": "later_validation_batches_only",
        "test_scope": "final_metrics_and_onnx_parity_only",
        "model": {
            "architecture": "Flatten-Linear-ReLU-Linear-ReLU-Linear",
            "input_shape": [
                1,
                1,
                1,
                candidate.preprocessor.output_features,
            ],
            "hidden_dims": list(config.hidden_dims),
            "output": "raw_failure_logit",
            "parameter_count": int(
                sum(parameter.numel() for parameter in candidate.mlp.parameters())
            ),
            "imbalance_handling": candidate.imbalance,
            "onnx_opset": 13,
        },
        "onnx_parity": parity,
        "quality_gate": quality_gate,
        "test_summary": mlp_test_summary,
        "claim_boundary": metrics["claim_boundary"],
        "reproducibility": {
            "seed": config.seed,
            "device": "cpu",
            "torch_deterministic_algorithms": True,
            "python": sys.version,
            "platform": platform.platform(),
            "versions": {
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
                "torch": torch.__version__,
                "onnx": onnx.__version__,
                "onnxruntime": onnxruntime.__version__,
            },
            "command": (
                "python tools/build_fabyield_x5.py --source "
                "research/data_assets/icmat_foundry/uci_secom/raw/secom.zip "
                "--output evaluation/icmat_foundry/fabyield/fabyield_x5_baseline_v1 "
                "--device cpu"
            ),
        },
        "artifacts": _artifact_inventory(output_dir, artifact_names),
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest
