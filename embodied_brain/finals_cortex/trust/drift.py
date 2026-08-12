"""Robust feature, stream, timestamp, and calibration drift diagnostics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

ArrayLike = Sequence[float] | np.ndarray
VALID_STATES = frozenset({"PASSIVE_OK", "REVIEW", "MONITOR_OFFLINE"})


def _matrix(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite matrix")
    return array


class RobustMahalanobis:
    """Median/MAD initialized, trimmed covariance Mahalanobis detector."""

    def __init__(
        self,
        *,
        threshold_quantile: float = 0.99,
        trim_quantile: float = 0.90,
        regularization: float = 1e-6,
    ) -> None:
        if not 0.5 < threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must lie in (0.5, 1)")
        if not 0.5 < trim_quantile <= 1.0:
            raise ValueError("trim_quantile must lie in (0.5, 1]")
        if not math.isfinite(regularization) or regularization <= 0.0:
            raise ValueError("regularization must be finite and positive")
        self.threshold_quantile = float(threshold_quantile)
        self.trim_quantile = float(trim_quantile)
        self.regularization = float(regularization)
        self.center_: np.ndarray | None = None
        self.inverse_covariance_: np.ndarray | None = None
        self.threshold_: float | None = None

    def fit(self, baseline: ArrayLike) -> RobustMahalanobis:
        values = _matrix(baseline, "baseline")
        if values.shape[0] < max(8, values.shape[1] + 2):
            raise ValueError("baseline has too few rows for robust covariance")

        initial_center = np.median(values, axis=0)
        mad = 1.4826 * np.median(np.abs(values - initial_center), axis=0)
        fallback = np.std(values, axis=0, ddof=1)
        scale = np.where(mad > 1e-9, mad, np.where(fallback > 1e-9, fallback, 1.0))
        robust_radius = np.sum(((values - initial_center) / scale) ** 2, axis=1)
        cutoff = float(np.quantile(robust_radius, self.trim_quantile))
        selected = values[robust_radius <= cutoff]
        if selected.shape[0] < values.shape[1] + 2:
            selected = values

        center = np.median(selected, axis=0)
        centered = selected - center
        covariance = centered.T @ centered / max(1, selected.shape[0] - 1)
        diagonal_scale = max(float(np.trace(covariance) / covariance.shape[0]), 1.0)
        covariance += self.regularization * diagonal_scale * np.eye(
            covariance.shape[0]
        )
        inverse = np.linalg.pinv(covariance, hermitian=True)

        self.center_ = center
        self.inverse_covariance_ = inverse
        training_scores = self.score(values)
        self.threshold_ = float(np.quantile(training_scores, self.threshold_quantile))
        return self

    @property
    def fitted(self) -> bool:
        return self.center_ is not None and self.inverse_covariance_ is not None

    def score(self, samples: ArrayLike) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Mahalanobis detector is not fitted")
        values = _matrix(samples, "samples")
        if values.shape[1] != self.center_.shape[0]:
            raise ValueError("sample feature dimension does not match baseline")
        centered = values - self.center_
        squared = np.einsum(
            "ni,ij,nj->n",
            centered,
            self.inverse_covariance_,
            centered,
        )
        return np.sqrt(np.maximum(squared, 0.0))

    def diagnose(self, samples: ArrayLike) -> dict[str, object]:
        if self.threshold_ is None:
            raise RuntimeError("Mahalanobis detector is not fitted")
        scores = self.score(samples)
        flags = scores > self.threshold_
        return {
            "scores": scores,
            "threshold": self.threshold_,
            "ood": flags,
            "ood_fraction": float(np.mean(flags)),
            "max_score": float(np.max(scores)),
            "read_only": True,
            "control_authority": False,
        }


class CUSUMDrift:
    """Two-sided standardized CUSUM for gradual streaming drift."""

    def __init__(
        self,
        baseline: ArrayLike,
        *,
        allowance: float = 0.5,
        threshold: float = 5.0,
    ) -> None:
        values = np.asarray(baseline, dtype=np.float64).reshape(-1)
        if values.size < 8 or not np.isfinite(values).all():
            raise ValueError("baseline must contain at least eight finite values")
        if allowance < 0.0 or threshold <= 0.0:
            raise ValueError("allowance and threshold must be non-negative/positive")
        self.center = float(np.median(values))
        mad = float(1.4826 * np.median(np.abs(values - self.center)))
        standard = float(np.std(values, ddof=1))
        self.scale = max(mad, standard * 0.25, 1e-9)
        self.allowance = float(allowance)
        self.threshold = float(threshold)
        self.positive = 0.0
        self.negative = 0.0
        self.samples_seen = 0
        self.triggered = False

    def update(self, value: float) -> dict[str, float | int | bool]:
        sample = float(value)
        if not math.isfinite(sample):
            raise ValueError("CUSUM value must be finite")
        standardized = (sample - self.center) / self.scale
        self.positive = max(
            0.0,
            self.positive + standardized - self.allowance,
        )
        self.negative = max(
            0.0,
            self.negative - standardized - self.allowance,
        )
        self.samples_seen += 1
        self.triggered = self.triggered or max(self.positive, self.negative) >= self.threshold
        return {
            "standardized": float(standardized),
            "positive": self.positive,
            "negative": self.negative,
            "triggered": self.triggered,
            "samples_seen": self.samples_seen,
            "read_only": True,
            "control_authority": False,
        }

    def reset(self) -> None:
        self.positive = 0.0
        self.negative = 0.0
        self.samples_seen = 0
        self.triggered = False


@dataclass(frozen=True, slots=True)
class TimeCalibrationThresholds:
    maximum_absolute_offset_s: float = 0.10
    maximum_offset_jitter_s: float = 0.03
    maximum_translation_delta_m: float = 0.10
    maximum_yaw_delta_rad: float = math.radians(5.0)
    minimum_samples: int = 3

    def __post_init__(self) -> None:
        numeric = (
            self.maximum_absolute_offset_s,
            self.maximum_offset_jitter_s,
            self.maximum_translation_delta_m,
            self.maximum_yaw_delta_rad,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in numeric):
            raise ValueError("time/calibration thresholds must be finite and positive")
        if isinstance(self.minimum_samples, bool) or self.minimum_samples < 1:
            raise ValueError("minimum_samples must be a positive integer")


def diagnose_time_calibration_drift(
    timestamp_offsets_s: ArrayLike,
    translation_deltas_m: ArrayLike,
    yaw_deltas_rad: ArrayLike,
    *,
    thresholds: TimeCalibrationThresholds | None = None,
) -> dict[str, object]:
    """Diagnose clock and extrinsic deltas without writing time or transforms."""

    selected = thresholds or TimeCalibrationThresholds()
    try:
        offsets = np.asarray(timestamp_offsets_s, dtype=np.float64).reshape(-1)
        translations = np.asarray(translation_deltas_m, dtype=np.float64)
        yaw = np.asarray(yaw_deltas_rad, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        offsets = np.empty(0)
        translations = np.empty((0, 0))
        yaw = np.empty(0)

    if translations.ndim == 1:
        translations = translations.reshape(-1, 1)
    valid = (
        offsets.size >= selected.minimum_samples
        and yaw.size >= selected.minimum_samples
        and translations.ndim == 2
        and translations.shape[0] >= selected.minimum_samples
        and translations.shape[1] in (2, 3)
        and np.isfinite(offsets).all()
        and np.isfinite(translations).all()
        and np.isfinite(yaw).all()
    )
    if not valid:
        return {
            "state": "MONITOR_OFFLINE",
            "reasons": ["time_calibration_inputs_unavailable"],
            "read_only": True,
            "control_authority": False,
        }

    median_offset = float(np.median(offsets))
    offset_jitter = float(1.4826 * np.median(np.abs(offsets - median_offset)))
    translation_norm = np.linalg.norm(translations, axis=1)
    translation_p95 = float(np.quantile(translation_norm, 0.95))
    yaw_p95 = float(np.quantile(np.abs(yaw), 0.95))
    reasons: list[str] = []
    if abs(median_offset) > selected.maximum_absolute_offset_s:
        reasons.append("timestamp_offset_drift")
    if offset_jitter > selected.maximum_offset_jitter_s:
        reasons.append("timestamp_jitter_drift")
    if translation_p95 > selected.maximum_translation_delta_m:
        reasons.append("translation_calibration_drift")
    if yaw_p95 > selected.maximum_yaw_delta_rad:
        reasons.append("yaw_calibration_drift")

    state = "REVIEW" if reasons else "PASSIVE_OK"
    return {
        "state": state,
        "reasons": reasons,
        "median_timestamp_offset_s": median_offset,
        "robust_timestamp_jitter_s": offset_jitter,
        "translation_delta_p95_m": translation_p95,
        "yaw_delta_p95_rad": yaw_p95,
        "read_only": True,
        "control_authority": False,
    }
