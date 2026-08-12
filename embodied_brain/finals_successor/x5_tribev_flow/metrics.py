"""Task metrics for offline TriBEV model acceptance."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    shifted = array - np.max(array, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=axis, keepdims=True), 1e-12)


def binary_occupancy_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    from_logits: bool = True,
    threshold: float = 0.5,
    valid_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute IoU, F1, precision and recall for an occupancy tensor."""
    pred = _sigmoid(prediction) if from_logits else np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64) >= 0.5
    chosen = pred >= float(threshold)
    if chosen.shape != truth.shape:
        raise ValueError(f"prediction/target shape mismatch: {chosen.shape} != {truth.shape}")
    mask = np.ones_like(truth, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if mask.shape != truth.shape:
        mask = np.broadcast_to(mask, truth.shape)
    chosen = chosen[mask]
    truth = truth[mask]
    tp = int(np.logical_and(chosen, truth).sum())
    fp = int(np.logical_and(chosen, ~truth).sum())
    fn = int(np.logical_and(~chosen, truth).sum())
    union = tp + fp + fn
    return {
        "iou": float(tp / union) if union else 1.0,
        "f1": float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 1.0,
        "precision": float(tp / (tp + fp)) if (tp + fp) else 1.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 1.0,
        "support": float(truth.sum()),
    }


def occupancy_horizon_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    from_logits: bool = True,
) -> dict[str, Any]:
    """Compute occupancy metrics independently for each future horizon."""
    pred = np.asarray(prediction)
    truth = np.asarray(target)
    if pred.shape != truth.shape or pred.ndim < 3:
        raise ValueError(f"expected equal [H,...] tensors, got {pred.shape} and {truth.shape}")
    rows = [
        {
            "horizon_index": index,
            **binary_occupancy_metrics(pred[index], truth[index], from_logits=from_logits),
        }
        for index in range(pred.shape[0])
    ]
    return {
        "horizons": rows,
        "mean_iou": float(np.mean([row["iou"] for row in rows])),
        "mean_f1": float(np.mean([row["f1"] for row in rows])),
    }


def flow_endpoint_error(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute endpoint error for channel-first 2D flow tensors."""
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim < 3 or pred.shape[-3] % 2:
        raise ValueError("flow tensors must match and have 2*K channels")
    pair_count = pred.shape[-3] // 2
    paired = (pred - truth).reshape(*pred.shape[:-3], pair_count, 2, *pred.shape[-2:])
    epe = np.sqrt(np.square(paired).sum(axis=-3))
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        mask = np.broadcast_to(mask, epe.shape)
        values = epe[mask]
    else:
        values = epe.reshape(-1)
    if not values.size:
        return {"mean_epe": math.nan, "p95_epe": math.nan, "count": 0.0}
    return {
        "mean_epe": float(values.mean()),
        "p95_epe": float(np.quantile(values, 0.95)),
        "count": float(values.size),
    }


def trajectory_distribution_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    prediction_is_logits: bool = True,
) -> dict[str, float | bool | int]:
    """Compare a nine-token student distribution with its teacher target."""
    pred = _softmax(prediction) if prediction_is_logits else np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    pred = pred / max(float(pred.sum()), 1e-12)
    truth = truth / max(float(truth.sum()), 1e-12)
    if pred.shape != truth.shape or pred.ndim != 1:
        raise ValueError("trajectory distributions must be equal 1D arrays")
    eps = 1e-12
    kl = float(np.sum(truth * np.log(np.maximum(truth, eps) / np.maximum(pred, eps))))
    return {
        "predicted_token": int(np.argmax(pred)),
        "target_token": int(np.argmax(truth)),
        "top1_agreement": bool(np.argmax(pred) == np.argmax(truth)),
        "kl_target_to_prediction": kl,
        "max_abs_probability_error": float(np.max(np.abs(pred - truth))),
    }


def expected_calibration_error(
    confidence: np.ndarray,
    correct: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Return standard equal-width expected calibration error."""
    conf = np.clip(np.asarray(confidence, dtype=np.float64).reshape(-1), 0.0, 1.0)
    hit = np.asarray(correct, dtype=np.float64).reshape(-1)
    if conf.shape != hit.shape:
        raise ValueError("confidence and correctness arrays must match")
    if not conf.size:
        return math.nan
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    score = 0.0
    for index in range(int(bins)):
        lower, upper = edges[index], edges[index + 1]
        selected = (conf >= lower) & (conf < upper if index < bins - 1 else conf <= upper)
        if selected.any():
            score += float(selected.mean()) * abs(float(conf[selected].mean() - hit[selected].mean()))
    return float(score)
