"""Frozen and adaptive-diagnostic split-conformal utilities."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

ArrayLike = Sequence[float] | np.ndarray


def _residuals(values: ArrayLike) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError("residuals must be non-empty, finite, and non-negative")
    return array


def split_conformal_quantile(
    residuals: ArrayLike,
    *,
    alpha: float = 0.10,
) -> float:
    """Finite-sample upper split-conformal quantile."""

    values = _residuals(residuals)
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    rank = int(math.ceil((values.size + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), values.size)
    return float(np.partition(values, rank - 1)[rank - 1])


@dataclass(frozen=True, slots=True)
class ConformalCoverage:
    frozen: float
    adaptive_diagnostic: float | None
    sample_count: int


class DualTrackConformal:
    """Keep an immutable deployment q and a rolling diagnostic q in parallel."""

    def __init__(
        self,
        frozen_q: float,
        *,
        alpha: float = 0.10,
        adaptive_window: int = 128,
        minimum_adaptive_samples: int = 20,
    ) -> None:
        if not math.isfinite(frozen_q) or frozen_q < 0.0:
            raise ValueError("frozen_q must be finite and non-negative")
        if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if isinstance(adaptive_window, bool) or adaptive_window < 2:
            raise ValueError("adaptive_window must be an integer >= 2")
        if (
            isinstance(minimum_adaptive_samples, bool)
            or minimum_adaptive_samples < 2
            or minimum_adaptive_samples > adaptive_window
        ):
            raise ValueError(
                "minimum_adaptive_samples must lie in [2, adaptive_window]"
            )
        self._frozen_q = float(frozen_q)
        self.alpha = float(alpha)
        self.adaptive_window = int(adaptive_window)
        self.minimum_adaptive_samples = int(minimum_adaptive_samples)
        self._adaptive_residuals: deque[float] = deque(maxlen=adaptive_window)

    @classmethod
    def fit(
        cls,
        residuals: ArrayLike,
        *,
        alpha: float = 0.10,
        adaptive_window: int = 128,
        minimum_adaptive_samples: int = 20,
    ) -> DualTrackConformal:
        return cls(
            split_conformal_quantile(residuals, alpha=alpha),
            alpha=alpha,
            adaptive_window=adaptive_window,
            minimum_adaptive_samples=minimum_adaptive_samples,
        )

    @property
    def frozen_q(self) -> float:
        return self._frozen_q

    @property
    def adaptive_diagnostic_q(self) -> float | None:
        if len(self._adaptive_residuals) < self.minimum_adaptive_samples:
            return None
        return split_conformal_quantile(
            np.asarray(self._adaptive_residuals, dtype=np.float64),
            alpha=self.alpha,
        )

    def update_diagnostic(self, residuals: ArrayLike | float) -> dict[str, object]:
        values = _residuals(np.atleast_1d(residuals))
        self._adaptive_residuals.extend(float(value) for value in values)
        return self.snapshot()

    def snapshot(self) -> dict[str, object]:
        return {
            "frozen_q": self.frozen_q,
            "adaptive_diagnostic_q": self.adaptive_diagnostic_q,
            "adaptive_sample_count": len(self._adaptive_residuals),
            "adaptive_ready": self.adaptive_diagnostic_q is not None,
            "adaptive_is_diagnostic_only": True,
            "read_only": True,
            "control_authority": False,
        }

    def upper(
        self,
        predictions: ArrayLike,
        *,
        clip: tuple[float, float] | None = (0.0, 1.0),
    ) -> dict[str, np.ndarray | None]:
        values = np.asarray(predictions, dtype=np.float64)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError("predictions must be non-empty and finite")

        frozen = values + self.frozen_q
        adaptive_q = self.adaptive_diagnostic_q
        adaptive = values + adaptive_q if adaptive_q is not None else None
        if clip is not None:
            lower, upper = map(float, clip)
            if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
                raise ValueError("clip must be an increasing finite pair")
            frozen = np.clip(frozen, lower, upper)
            adaptive = (
                np.clip(adaptive, lower, upper) if adaptive is not None else None
            )
        return {
            "frozen": frozen,
            "adaptive_diagnostic": adaptive,
        }

    def empirical_coverage(
        self,
        predictions: ArrayLike,
        targets: ArrayLike,
    ) -> ConformalCoverage:
        predicted = np.asarray(predictions, dtype=np.float64).reshape(-1)
        observed = np.asarray(targets, dtype=np.float64).reshape(-1)
        if (
            predicted.size == 0
            or predicted.shape != observed.shape
            or not np.isfinite(predicted).all()
            or not np.isfinite(observed).all()
        ):
            raise ValueError("predictions and targets must be matching finite vectors")
        frozen = float(np.mean(observed <= predicted + self.frozen_q))
        adaptive_q = self.adaptive_diagnostic_q
        adaptive = (
            float(np.mean(observed <= predicted + adaptive_q))
            if adaptive_q is not None
            else None
        )
        return ConformalCoverage(frozen, adaptive, int(predicted.size))
