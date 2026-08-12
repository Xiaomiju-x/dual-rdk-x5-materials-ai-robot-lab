"""Risk/selective-prediction metrics and lightweight binary calibration."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

ArrayLike = Sequence[float] | np.ndarray


def _finite_vector(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    return array


def _binary_labels(labels: ArrayLike) -> np.ndarray:
    values = _finite_vector(labels, "labels")
    if not np.all((values == 0.0) | (values == 1.0)):
        raise ValueError("labels must contain only 0 and 1")
    if np.unique(values).size != 2:
        raise ValueError("labels must contain both classes")
    return values


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def binary_log_loss(probabilities: ArrayLike, labels: ArrayLike) -> float:
    probabilities_array = _finite_vector(probabilities, "probabilities")
    labels_array = _finite_vector(labels, "labels")
    if probabilities_array.shape != labels_array.shape:
        raise ValueError("probabilities and labels must have matching shapes")
    if np.any((probabilities_array < 0.0) | (probabilities_array > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.all((labels_array == 0.0) | (labels_array == 1.0)):
        raise ValueError("labels must contain only 0 and 1")
    clipped = np.clip(probabilities_array, 1e-12, 1.0 - 1e-12)
    loss = -labels_array * np.log(clipped) - (1.0 - labels_array) * np.log1p(
        -clipped
    )
    return float(np.mean(loss))


def expected_calibration_error(
    probabilities: ArrayLike,
    labels: ArrayLike,
    *,
    bins: int = 10,
) -> float:
    """Return equal-width expected calibration error for binary predictions."""

    if isinstance(bins, bool) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    probabilities_array = _finite_vector(probabilities, "probabilities")
    labels_array = _finite_vector(labels, "labels")
    if probabilities_array.shape != labels_array.shape:
        raise ValueError("probabilities and labels must have matching shapes")
    if np.any((probabilities_array < 0.0) | (probabilities_array > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.all((labels_array == 0.0) | (labels_array == 1.0)):
        raise ValueError("labels must contain only 0 and 1")

    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.minimum(np.searchsorted(edges, probabilities_array, side="right") - 1, bins - 1)
    indices = np.maximum(indices, 0)
    error = 0.0
    for index in range(bins):
        selected = indices == index
        count = int(np.count_nonzero(selected))
        if count:
            confidence = float(np.mean(probabilities_array[selected]))
            accuracy = float(np.mean(labels_array[selected]))
            error += (count / probabilities_array.size) * abs(confidence - accuracy)
    return float(error)


def risk_coverage_curve(
    losses: ArrayLike,
    confidence: ArrayLike | None = None,
    *,
    uncertainty: ArrayLike | None = None,
) -> dict[str, np.ndarray | float | int]:
    """Compute selective risk as progressively less certain samples are kept.

    Exactly one of ``confidence`` or ``uncertainty`` is required. Lower losses
    are better. Confidence is sorted high-to-low; uncertainty low-to-high.
    AURC is the discrete mean risk over all attainable non-zero coverages.
    """

    loss = _finite_vector(losses, "losses")
    if np.any(loss < 0.0):
        raise ValueError("losses must be non-negative")
    if (confidence is None) == (uncertainty is None):
        raise ValueError("provide exactly one of confidence or uncertainty")

    if confidence is not None:
        ranking = _finite_vector(confidence, "confidence")
        if ranking.shape != loss.shape:
            raise ValueError("confidence and losses must have matching shapes")
        order = np.argsort(-ranking, kind="stable")
        thresholds = ranking[order]
    else:
        ranking = _finite_vector(uncertainty, "uncertainty")
        if ranking.shape != loss.shape:
            raise ValueError("uncertainty and losses must have matching shapes")
        order = np.argsort(ranking, kind="stable")
        thresholds = ranking[order]

    ordered_loss = loss[order]
    counts = np.arange(1, loss.size + 1, dtype=np.float64)
    coverage = counts / float(loss.size)
    risk = np.cumsum(ordered_loss) / counts
    return {
        "coverage": coverage,
        "risk": risk,
        "thresholds": thresholds,
        "order": order,
        "aurc": float(np.mean(risk)),
        "sample_count": int(loss.size),
    }


def aurc(
    losses: ArrayLike,
    confidence: ArrayLike | None = None,
    *,
    uncertainty: ArrayLike | None = None,
) -> float:
    return float(
        risk_coverage_curve(
            losses,
            confidence,
            uncertainty=uncertainty,
        )["aurc"]
    )


def risk_at_coverage(
    losses: ArrayLike,
    target_coverage: float,
    confidence: ArrayLike | None = None,
    *,
    uncertainty: ArrayLike | None = None,
) -> dict[str, float | int]:
    """Return risk at the smallest attainable coverage >= target coverage."""

    if not math.isfinite(target_coverage) or not 0.0 < target_coverage <= 1.0:
        raise ValueError("target_coverage must lie in (0, 1]")
    curve = risk_coverage_curve(
        losses,
        confidence,
        uncertainty=uncertainty,
    )
    sample_count = int(curve["sample_count"])
    retained = min(sample_count, max(1, int(math.ceil(target_coverage * sample_count))))
    return {
        "target_coverage": float(target_coverage),
        "actual_coverage": float(retained / sample_count),
        "risk": float(np.asarray(curve["risk"])[retained - 1]),
        "retained": retained,
        "sample_count": sample_count,
    }


@dataclass(slots=True)
class TemperatureScaler:
    """One-parameter binary temperature scaling using bounded golden search."""

    temperature: float = 1.0
    fitted: bool = False

    def fit(
        self,
        logits: ArrayLike,
        labels: ArrayLike,
        *,
        minimum: float = 0.05,
        maximum: float = 20.0,
        iterations: int = 96,
    ) -> TemperatureScaler:
        values = _finite_vector(logits, "logits")
        targets = _binary_labels(labels)
        if values.shape != targets.shape:
            raise ValueError("logits and labels must have matching shapes")
        if minimum <= 0.0 or maximum <= minimum:
            raise ValueError("temperature bounds must satisfy 0 < minimum < maximum")
        if isinstance(iterations, bool) or iterations < 8:
            raise ValueError("iterations must be an integer >= 8")

        lower = math.log(minimum)
        upper = math.log(maximum)
        ratio = (math.sqrt(5.0) - 1.0) / 2.0

        def objective(log_temperature: float) -> float:
            temperature = math.exp(log_temperature)
            return binary_log_loss(_sigmoid(values / temperature), targets)

        left = upper - ratio * (upper - lower)
        right = lower + ratio * (upper - lower)
        left_value = objective(left)
        right_value = objective(right)
        for _ in range(iterations):
            if left_value <= right_value:
                upper = right
                right = left
                right_value = left_value
                left = upper - ratio * (upper - lower)
                left_value = objective(left)
            else:
                lower = left
                left = right
                left_value = right_value
                right = lower + ratio * (upper - lower)
                right_value = objective(right)

        candidate = math.exp(0.5 * (lower + upper))
        baseline_loss = objective(0.0)
        candidate_loss = objective(math.log(candidate))
        self.temperature = float(candidate if candidate_loss <= baseline_loss else 1.0)
        self.fitted = True
        return self

    def predict_proba(self, logits: ArrayLike) -> np.ndarray:
        values = _finite_vector(logits, "logits")
        if not self.fitted:
            raise RuntimeError("temperature scaler is not fitted")
        return _sigmoid(values / self.temperature)


@dataclass(slots=True)
class PlattScaler:
    """Binary Platt calibration with damped Newton updates."""

    slope: float = 1.0
    intercept: float = 0.0
    fitted: bool = False

    def fit(
        self,
        scores: ArrayLike,
        labels: ArrayLike,
        *,
        l2: float = 1e-6,
        max_iterations: int = 100,
        tolerance: float = 1e-9,
    ) -> PlattScaler:
        values = _finite_vector(scores, "scores")
        targets = _binary_labels(labels)
        if values.shape != targets.shape:
            raise ValueError("scores and labels must have matching shapes")
        if not math.isfinite(l2) or l2 < 0.0:
            raise ValueError("l2 must be finite and non-negative")

        design = np.column_stack((values, np.ones_like(values)))
        parameters = np.array([1.0, 0.0], dtype=np.float64)

        def objective(candidate: np.ndarray) -> float:
            probabilities = _sigmoid(design @ candidate)
            return binary_log_loss(probabilities, targets) + 0.5 * l2 * float(
                candidate @ candidate
            )

        for _ in range(max_iterations):
            probabilities = _sigmoid(design @ parameters)
            gradient = design.T @ (probabilities - targets) / values.size
            gradient += l2 * parameters
            weights = np.maximum(probabilities * (1.0 - probabilities), 1e-9)
            hessian = (design.T * weights) @ design / values.size
            hessian += (l2 + 1e-9) * np.eye(2)
            step = np.linalg.solve(hessian, gradient)
            if float(np.linalg.norm(step)) <= tolerance:
                break

            current = objective(parameters)
            scale = 1.0
            while scale >= 1e-6:
                candidate = parameters - scale * step
                if objective(candidate) <= current:
                    parameters = candidate
                    break
                scale *= 0.5
            else:
                break

        identity = np.array([1.0, 0.0], dtype=np.float64)
        if objective(parameters) > objective(identity):
            parameters = identity
        self.slope = float(parameters[0])
        self.intercept = float(parameters[1])
        self.fitted = True
        return self

    def predict_proba(self, scores: ArrayLike) -> np.ndarray:
        values = _finite_vector(scores, "scores")
        if not self.fitted:
            raise RuntimeError("Platt scaler is not fitted")
        return _sigmoid(self.slope * values + self.intercept)
