"""Validation-only segmentation and quality calibration helpers."""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def dice_score(probability: np.ndarray, target: np.ndarray, threshold: float) -> float:
    prediction = np.asarray(probability) >= threshold
    truth = np.asarray(target, dtype=bool)
    intersection = int(np.count_nonzero(prediction & truth))
    denominator = int(np.count_nonzero(prediction)) + int(np.count_nonzero(truth))
    return 2.0 * intersection / denominator if denominator else 1.0


def select_segmentation_threshold(
    probabilities: list[np.ndarray],
    targets: list[np.ndarray],
    *,
    candidates: tuple[float, ...] = (
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    ),
) -> dict[str, Any]:
    if not probabilities or len(probabilities) != len(targets):
        raise ValueError("calibration probabilities and targets must be paired")
    scores = {
        threshold: float(
            np.mean(
                [
                    dice_score(probability, target, threshold)
                    for probability, target in zip(probabilities, targets, strict=True)
                ]
            )
        )
        for threshold in candidates
    }
    best = max(candidates, key=lambda threshold: (scores[threshold], -abs(threshold - 0.5)))
    return {
        "threshold": best,
        "calibration_macro_dice": scores[best],
        "candidate_scores": {f"{key:.2f}": value for key, value in scores.items()},
    }


def conformal_lower_bounds(
    calibration_prediction: np.ndarray,
    calibration_observed: np.ndarray,
    validation_prediction: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, float]:
    predicted = np.asarray(calibration_prediction, dtype=np.float64)
    observed = np.asarray(calibration_observed, dtype=np.float64)
    validation = np.asarray(validation_prediction, dtype=np.float64)
    if predicted.shape != observed.shape or predicted.ndim != 1:
        raise ValueError("calibration quality arrays must be paired 1D arrays")
    if predicted.size < 10:
        raise ValueError("at least 10 calibration examples are required")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    residual = np.abs(predicted - observed)
    rank = min(
        int(math.ceil((predicted.size + 1) * (1.0 - alpha))),
        predicted.size,
    )
    radius = float(np.partition(residual, rank - 1)[rank - 1])
    return np.clip(validation - radius, 0.0, 1.0), radius


def expected_calibration_error(
    confidence: np.ndarray,
    outcome: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    conf = np.asarray(confidence, dtype=np.float64)
    obs = np.asarray(outcome, dtype=np.float64)
    if conf.shape != obs.shape or conf.ndim != 1:
        raise ValueError("confidence and outcome must be paired 1D arrays")
    if np.any((conf < 0.0) | (conf > 1.0)):
        raise ValueError("confidence must be in [0, 1]")
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        upper_inclusive = index == bins - 1
        selected = (conf >= edges[index]) & (
            (conf <= edges[index + 1])
            if upper_inclusive
            else (conf < edges[index + 1])
        )
        if np.any(selected):
            result += float(np.mean(selected)) * abs(
                float(np.mean(conf[selected])) - float(np.mean(obs[selected]))
            )
    return result
