"""Metrics and validation-only decision policy selection."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
)

EPSILON = 1e-7


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probabilities >= edges[index]) & (
                probabilities <= edges[index + 1]
            )
        else:
            mask = (probabilities >= edges[index]) & (
                probabilities < edges[index + 1]
            )
        if not np.any(mask):
            continue
        error += float(np.mean(mask)) * abs(
            float(np.mean(labels[mask])) - float(np.mean(probabilities[mask]))
        )
    return float(error)


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.clip(
        np.asarray(probabilities, dtype=np.float64), EPSILON, 1.0 - EPSILON
    )
    predictions = (probabilities >= threshold).astype(np.int64)
    true_positive = int(np.sum((labels == 1) & (predictions == 1)))
    true_negative = int(np.sum((labels == 0) & (predictions == 0)))
    false_positive = int(np.sum((labels == 0) & (predictions == 1)))
    false_negative = int(np.sum((labels == 1) & (predictions == 0)))
    failure_recall = true_positive / max(1, true_positive + false_negative)
    failure_precision = true_positive / max(1, true_positive + false_positive)
    specificity = true_negative / max(1, true_negative + false_positive)

    both_classes = np.unique(labels).size == 2
    return {
        "rows": int(labels.size),
        "failure_rows": int(np.sum(labels == 1)),
        "failure_prevalence": float(np.mean(labels == 1)),
        "threshold": float(threshold),
        "auprc_primary": (
            float(average_precision_score(labels, probabilities))
            if both_classes
            else None
        ),
        "auroc_secondary": (
            float(roc_auc_score(labels, probabilities)) if both_classes else None
        ),
        "balanced_accuracy_primary": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "macro_f1_primary": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "failure_recall_primary": float(failure_recall),
        "failure_precision": float(failure_precision),
        "failure_f2": float(
            fbeta_score(labels, predictions, beta=2.0, zero_division=0)
        ),
        "specificity": float(specificity),
        "brier_score": float(np.mean(np.square(probabilities - labels))),
        "log_loss": float(
            -np.mean(
                labels * np.log(probabilities)
                + (1 - labels) * np.log(1.0 - probabilities)
            )
        ),
        "expected_calibration_error_10bin": expected_calibration_error(
            labels, probabilities
        ),
        "accuracy_supplemental_not_primary": float(np.mean(predictions == labels)),
        "confusion": {
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_positive": true_positive,
        },
    }


def select_decision_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    """Select a failure-sensitive threshold on validation policy rows only."""
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    candidates = np.unique(
        np.concatenate(
            (
                np.linspace(0.01, 0.99, 197),
                np.clip(probabilities, 0.01, 0.99),
            )
        )
    )
    ranked: list[tuple[float, float, float, float]] = []
    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(np.int64)
        balanced = balanced_accuracy_score(labels, predictions)
        macro_f1 = f1_score(
            labels,
            predictions,
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )
        recall = float(np.mean(predictions[labels == 1] == 1))
        objective = 0.5 * balanced + 0.5 * macro_f1
        ranked.append((float(objective), recall, -float(threshold), float(threshold)))
    objective, recall, _, threshold = max(ranked)
    return {
        "threshold": threshold,
        "selection_scope": "validation_policy_batches_only",
        "objective": "0.5 * balanced_accuracy + 0.5 * fixed_class_macro_f1",
        "objective_value": objective,
        "failure_recall_at_selection": recall,
        "candidate_count": len(candidates),
    }


def select_reject_margin(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
    minimum_coverage: float = 0.80,
) -> dict[str, Any]:
    """Choose a symmetric uncertainty band without consulting test labels."""
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    distances = np.abs(probabilities - threshold)
    candidates = np.unique(np.concatenate(([0.0], distances)))
    ranked: list[tuple[float, float, float, float]] = []
    for margin in candidates:
        accepted = distances >= margin - 1e-15
        coverage = float(np.mean(accepted))
        if coverage + 1e-12 < minimum_coverage or not np.any(accepted):
            continue
        accepted_labels = labels[accepted]
        if np.unique(accepted_labels).size < 2:
            continue
        predictions = (probabilities[accepted] >= threshold).astype(np.int64)
        balanced = balanced_accuracy_score(accepted_labels, predictions)
        macro_f1 = f1_score(
            accepted_labels, predictions, average="macro", zero_division=0
        )
        objective = 0.5 * balanced + 0.5 * macro_f1
        ranked.append((float(objective), -coverage, float(margin), coverage))
    if not ranked:
        return {
            "margin": 0.0,
            "minimum_coverage": minimum_coverage,
            "selection_scope": "validation_policy_batches_only",
            "coverage_at_selection": 1.0,
            "objective": "fallback_no_rejection",
        }
    objective, _, margin, coverage = max(ranked)
    return {
        "margin": margin,
        "minimum_coverage": minimum_coverage,
        "selection_scope": "validation_policy_batches_only",
        "coverage_at_selection": coverage,
        "objective": "0.5 * selective_balanced_accuracy + 0.5 * selective_macro_f1",
        "objective_value": objective,
    }


def selective_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
    reject_margin: float,
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    accepted = np.abs(probabilities - threshold) >= reject_margin - 1e-15
    accepted_labels = labels[accepted]
    result: dict[str, Any] = {
        "rows": int(labels.size),
        "accepted_rows": int(np.sum(accepted)),
        "rejected_rows": int(np.sum(~accepted)),
        "coverage": float(np.mean(accepted)),
        "reject_rate": float(np.mean(~accepted)),
        "reject_rule": "abs(calibrated_probability - decision_threshold) < margin",
        "reject_margin": float(reject_margin),
    }
    if accepted_labels.size and np.unique(accepted_labels).size == 2:
        result["accepted_metrics"] = classification_metrics(
            accepted_labels,
            probabilities[accepted],
            threshold=threshold,
        )
    else:
        result["accepted_metrics"] = None
        result["warning"] = "accepted subset does not contain both classes"
    return result


def temporal_batch_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    batch_ids: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Report fixed-class metrics across calendar-day proxy batches."""
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    batch_ids = np.asarray(batch_ids)
    records = []
    for batch_id in sorted(set(str(value) for value in batch_ids)):
        mask = batch_ids.astype(str) == batch_id
        batch_labels = labels[mask]
        predictions = (probabilities[mask] >= threshold).astype(np.int64)
        true_positive = int(np.sum((batch_labels == 1) & (predictions == 1)))
        true_negative = int(np.sum((batch_labels == 0) & (predictions == 0)))
        positives = int(np.sum(batch_labels == 1))
        negatives = int(np.sum(batch_labels == 0))
        records.append(
            {
                "batch_id": batch_id,
                "rows": int(np.sum(mask)),
                "failures": positives,
                "fixed_class_macro_f1": float(
                    f1_score(
                        batch_labels,
                        predictions,
                        labels=[0, 1],
                        average="macro",
                        zero_division=0,
                    )
                ),
                "failure_recall": (
                    float(true_positive / positives) if positives else None
                ),
                "specificity": (
                    float(true_negative / negatives) if negatives else None
                ),
            }
        )
    positive_batches = [
        record["failure_recall"]
        for record in records
        if record["failure_recall"] is not None
    ]
    specificities = [
        record["specificity"]
        for record in records
        if record["specificity"] is not None
    ]
    return {
        "group_definition": (
            "calendar-day temporal proxy; SECOM provides no wafer/run identifier"
        ),
        "batch_count": len(records),
        "mean_fixed_class_macro_f1": float(
            np.mean([record["fixed_class_macro_f1"] for record in records])
        ),
        "mean_failure_recall_across_batches_with_failures": (
            float(np.mean(positive_batches)) if positive_batches else None
        ),
        "mean_specificity_across_batches_with_pass_rows": (
            float(np.mean(specificities)) if specificities else None
        ),
        "per_batch": records,
    }


def stratified_bootstrap_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    class_indices = [np.flatnonzero(labels == value) for value in (0, 1)]
    if repeats <= 0 or any(indices.size == 0 for indices in class_indices):
        return {"repeats": 0, "intervals": {}}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "auprc_primary": [],
        "balanced_accuracy_primary": [],
        "macro_f1_primary": [],
        "failure_recall_primary": [],
        "brier_score": [],
    }
    for _ in range(repeats):
        sample = np.concatenate(
            (
                rng.choice(class_indices[0], class_indices[0].size, replace=True),
                rng.choice(class_indices[1], class_indices[1].size, replace=True),
            )
        )
        metrics = classification_metrics(
            labels[sample], probabilities[sample], threshold=threshold
        )
        for name in values:
            values[name].append(float(metrics[name]))
    return {
        "method": "deterministic_stratified_nonparametric_bootstrap",
        "repeats": repeats,
        "seed": seed,
        "interval": "percentile_95",
        "warning": (
            "Intervals reflect this small public test partition only and do not establish "
            "modern-fab or local-fab performance."
        ),
        "intervals": {
            name: {
                "low": float(np.percentile(samples, 2.5)),
                "high": float(np.percentile(samples, 97.5)),
            }
            for name, samples in values.items()
        },
    }
