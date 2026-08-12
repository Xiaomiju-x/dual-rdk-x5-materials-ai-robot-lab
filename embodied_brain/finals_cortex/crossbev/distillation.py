"""Pure NumPy teacher/student distillation losses for CrossBEV."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

import numpy as np

from .contracts import CROSSBEV_LAYER_NAMES, CrossBEVMaps

DEFAULT_LAYER_WEIGHTS: Mapping[str, float] = {
    "obstacle": 2.0,
    "traversability": 1.5,
    "semantic": 1.25,
    "dynamic": 1.5,
    "visibility": 0.75,
    "unknown": 1.25,
    "confidence": 0.50,
}


@dataclass(frozen=True, slots=True)
class DistillationLoss:
    """Weighted Bernoulli-KL decomposition; zero means exact agreement."""

    total: float
    components: Mapping[str, float]
    layer_weights: Mapping[str, float]
    effective_weight_sum: float


def _logit(probabilities: np.ndarray, epsilon: float) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), epsilon, 1.0 - epsilon)
    return np.log(values) - np.log1p(-values)


def _temperature_probability(
    probabilities: np.ndarray,
    *,
    temperature: float,
    epsilon: float,
) -> np.ndarray:
    logits = _logit(probabilities, epsilon) / temperature
    logits = np.clip(logits, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _bernoulli_kl(
    teacher: np.ndarray,
    student: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    target = np.clip(teacher, epsilon, 1.0 - epsilon)
    prediction = np.clip(student, epsilon, 1.0 - epsilon)
    complement_target = 1.0 - target
    complement_prediction = 1.0 - prediction
    result = target * np.log(target / prediction)
    result += complement_target * np.log(complement_target / complement_prediction)
    return np.maximum(result, 0.0)


def crossbev_distillation_loss(
    student: CrossBEVMaps,
    teacher: CrossBEVMaps,
    *,
    valid_mask: np.ndarray | None = None,
    layer_weights: Mapping[str, float] | None = None,
    temperature: float = 1.0,
    confidence_floor: float = 0.10,
    epsilon: float = 1e-6,
) -> DistillationLoss:
    """Distill seven separate BEV layers without collapsing their semantics.

    Teacher confidence weights supervision but never converts unknown cells to
    free. ``temperature`` is applied in Bernoulli logit space. The result is a
    proposal-training loss only and has no control-plane side effects.
    """

    if not isinstance(student, CrossBEVMaps) or not isinstance(teacher, CrossBEVMaps):
        raise TypeError("student and teacher must be CrossBEVMaps")
    if student.shape != teacher.shape:
        raise ValueError("student and teacher CrossBEV shapes must match")
    if not isfinite(float(temperature)) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not isfinite(float(confidence_floor)) or not 0.0 <= confidence_floor <= 1.0:
        raise ValueError("confidence_floor must lie in [0, 1]")
    if not isfinite(float(epsilon)) or not 0.0 < epsilon < 0.1:
        raise ValueError("epsilon must lie in (0, 0.1)")

    mask = (
        np.ones(student.shape, dtype=np.float64)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=np.float64)
    )
    if mask.shape != student.shape or not np.isfinite(mask).all():
        raise ValueError("valid_mask must be finite and match the BEV shape")
    if np.any((mask < 0.0) | (mask > 1.0)):
        raise ValueError("valid_mask values must lie in [0, 1]")

    weights = dict(DEFAULT_LAYER_WEIGHTS)
    if layer_weights is not None:
        unknown = set(layer_weights).difference(CROSSBEV_LAYER_NAMES)
        if unknown:
            raise ValueError(f"unknown CrossBEV loss layers: {sorted(unknown)}")
        weights.update({name: float(value) for name, value in layer_weights.items()})
    if any(not isfinite(value) or value < 0.0 for value in weights.values()):
        raise ValueError("layer weights must be finite and non-negative")
    if not any(weights.values()):
        raise ValueError("at least one layer weight must be positive")

    confidence_weight = confidence_floor + (
        1.0 - confidence_floor
    ) * np.asarray(teacher.confidence, dtype=np.float64)
    spatial_weight = mask * confidence_weight
    denominator = float(spatial_weight.sum())
    if denominator <= 0.0:
        raise ValueError("distillation has no valid weighted cells")

    components: dict[str, float] = {}
    total = 0.0
    effective_weight_sum = 0.0
    for name in CROSSBEV_LAYER_NAMES:
        teacher_probability = _temperature_probability(
            getattr(teacher, name),
            temperature=float(temperature),
            epsilon=float(epsilon),
        )
        student_probability = _temperature_probability(
            getattr(student, name),
            temperature=float(temperature),
            epsilon=float(epsilon),
        )
        divergence = _bernoulli_kl(
            teacher_probability,
            student_probability,
            epsilon=float(epsilon),
        )
        component = float(
            np.sum(divergence * spatial_weight)
            / denominator
            * float(temperature) ** 2
        )
        components[name] = component
        layer_weight = float(weights[name])
        total += layer_weight * component
        effective_weight_sum += layer_weight

    return DistillationLoss(
        total=float(total),
        components=components,
        layer_weights=weights,
        effective_weight_sum=float(effective_weight_sum),
    )


__all__ = [
    "DEFAULT_LAYER_WEIGHTS",
    "DistillationLoss",
    "crossbev_distillation_loss",
]
