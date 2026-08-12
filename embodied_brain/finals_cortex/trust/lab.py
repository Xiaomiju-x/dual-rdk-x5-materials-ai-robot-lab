"""Read-only Trust Lab composition for finals-cortex shadow diagnostics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .conformal import DualTrackConformal
from .disagreement import cross_modal_disagreement
from .drift import (
    CUSUMDrift,
    RobustMahalanobis,
    TimeCalibrationThresholds,
    diagnose_time_calibration_drift,
)
from .metrics import risk_at_coverage, risk_coverage_curve


class TrustState(str, Enum):
    PASSIVE_OK = "PASSIVE_OK"
    REVIEW = "REVIEW"
    MONITOR_OFFLINE = "MONITOR_OFFLINE"


@dataclass(frozen=True, slots=True)
class TrustThresholds:
    maximum_disagreement: float = 0.45
    maximum_ood_fraction: float = 0.20
    maximum_adaptive_q_ratio: float = 2.0
    risk_coverages: tuple[float, ...] = (0.80, 0.90)

    def __post_init__(self) -> None:
        if not 0.0 <= self.maximum_disagreement <= 1.0:
            raise ValueError("maximum_disagreement must lie in [0, 1]")
        if not 0.0 <= self.maximum_ood_fraction <= 1.0:
            raise ValueError("maximum_ood_fraction must lie in [0, 1]")
        if (
            not math.isfinite(self.maximum_adaptive_q_ratio)
            or self.maximum_adaptive_q_ratio < 1.0
        ):
            raise ValueError("maximum_adaptive_q_ratio must be finite and >= 1")
        if not self.risk_coverages or any(
            not math.isfinite(value) or not 0.0 < value <= 1.0
            for value in self.risk_coverages
        ):
            raise ValueError("risk_coverages must contain values in (0, 1]")


class TrustLab:
    """Aggregate statistical checks into one non-authoritative shadow report."""

    read_only = True
    control_authority = False

    def __init__(
        self,
        *,
        conformal: DualTrackConformal,
        ood_detector: RobustMahalanobis,
        cusum: CUSUMDrift | None = None,
        thresholds: TrustThresholds | None = None,
        time_calibration_thresholds: TimeCalibrationThresholds | None = None,
    ) -> None:
        if not ood_detector.fitted:
            raise ValueError("ood_detector must be fitted")
        self.conformal = conformal
        self.ood_detector = ood_detector
        self.cusum = cusum
        self.thresholds = thresholds or TrustThresholds()
        self.time_calibration_thresholds = (
            time_calibration_thresholds or TimeCalibrationThresholds()
        )

    def _offline(self, reason: str) -> dict[str, object]:
        return {
            "state": TrustState.MONITOR_OFFLINE.value,
            "reasons": [reason],
            "read_only": True,
            "control_authority": False,
            "side_effects": False,
        }

    def evaluate(
        self,
        *,
        losses: Sequence[float] | np.ndarray,
        confidence: Sequence[float] | np.ndarray,
        features: np.ndarray,
        modalities: Mapping[str, np.ndarray],
        timestamp_offsets_s: Sequence[float] | np.ndarray,
        translation_deltas_m: np.ndarray,
        yaw_deltas_rad: Sequence[float] | np.ndarray,
        predictions: Sequence[float] | np.ndarray,
        targets: Sequence[float] | np.ndarray,
        monitor_ready: bool = True,
        update_adaptive_diagnostic: bool = True,
    ) -> dict[str, object]:
        if not monitor_ready:
            return self._offline("monitor_not_ready")

        try:
            curve = risk_coverage_curve(losses, confidence)
            coverage_metrics = {
                str(coverage): risk_at_coverage(
                    losses,
                    coverage,
                    confidence,
                )
                for coverage in self.thresholds.risk_coverages
            }
            ood = self.ood_detector.diagnose(features)
            disagreement = cross_modal_disagreement(modalities)
            time_calibration = diagnose_time_calibration_drift(
                timestamp_offsets_s,
                translation_deltas_m,
                yaw_deltas_rad,
                thresholds=self.time_calibration_thresholds,
            )
            predicted = np.asarray(predictions, dtype=np.float64).reshape(-1)
            observed = np.asarray(targets, dtype=np.float64).reshape(-1)
            if (
                predicted.size == 0
                or predicted.shape != observed.shape
                or not np.isfinite(predicted).all()
                or not np.isfinite(observed).all()
            ):
                raise ValueError("conformal prediction contract invalid")
            residuals = np.maximum(observed - predicted, 0.0)
            if update_adaptive_diagnostic:
                self.conformal.update_diagnostic(residuals)
            conformal_coverage = self.conformal.empirical_coverage(
                predicted,
                observed,
            )
        except (TypeError, ValueError, RuntimeError, np.linalg.LinAlgError):
            return self._offline("diagnostic_input_contract_invalid")

        if not disagreement["valid"]:
            return self._offline("cross_modal_inputs_unavailable")
        if time_calibration["state"] == TrustState.MONITOR_OFFLINE.value:
            return self._offline("time_calibration_inputs_unavailable")

        reasons: list[str] = []
        if float(ood["ood_fraction"]) > self.thresholds.maximum_ood_fraction:
            reasons.append("feature_ood")
        if float(disagreement["max_score"]) > self.thresholds.maximum_disagreement:
            reasons.append("cross_modal_disagreement")
        if time_calibration["state"] == TrustState.REVIEW.value:
            reasons.extend(str(reason) for reason in time_calibration["reasons"])

        adaptive_q = self.conformal.adaptive_diagnostic_q
        if adaptive_q is not None:
            denominator = max(self.conformal.frozen_q, 1e-9)
            if adaptive_q / denominator > self.thresholds.maximum_adaptive_q_ratio:
                reasons.append("adaptive_conformal_shift")

        cusum_result: dict[str, object] | None = None
        if self.cusum is not None:
            cusum_result = self.cusum.update(float(np.mean(ood["scores"])))
            if bool(cusum_result["triggered"]):
                reasons.append("cusum_drift")

        state = TrustState.REVIEW if reasons else TrustState.PASSIVE_OK
        return {
            "state": state.value,
            "reasons": list(dict.fromkeys(reasons)),
            "risk_coverage": {
                "aurc": float(curve["aurc"]),
                "sample_count": int(curve["sample_count"]),
                "risk_at_coverage": coverage_metrics,
            },
            "conformal": {
                **self.conformal.snapshot(),
                "frozen_empirical_coverage": conformal_coverage.frozen,
                "adaptive_diagnostic_empirical_coverage": (
                    conformal_coverage.adaptive_diagnostic
                ),
                "sample_count": conformal_coverage.sample_count,
            },
            "ood": ood,
            "cusum": cusum_result,
            "cross_modal": disagreement,
            "time_calibration": time_calibration,
            "read_only": True,
            "control_authority": False,
            "side_effects": False,
        }
