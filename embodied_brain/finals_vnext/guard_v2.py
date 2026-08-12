"""Online-ready confidence and runtime-assurance logic for shadow outputs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

from embodied_brain.finals_successor.x5_tribev_flow.shadow_guard import (
    energy_ood,
    trajectory_token_js_divergence,
)


class GuardState(str, Enum):
    PASSIVE_OK = "PASSIVE_OK"
    REVIEW = "REVIEW"
    MONITOR_OFFLINE = "MONITOR_OFFLINE"


@dataclass(frozen=True, slots=True)
class GuardThresholdsV2:
    energy_ood_threshold: float
    max_cross_modal_disagreement: float = 0.45
    max_candidate_js: float = 0.40
    max_candidate_risk_gap: float = 0.35
    min_required_health: float = 0.75

    def __post_init__(self) -> None:
        values = (
            self.energy_ood_threshold,
            self.max_cross_modal_disagreement,
            self.max_candidate_js,
            self.max_candidate_risk_gap,
            self.min_required_health,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("guard thresholds must be finite")
        if not 0.0 <= self.max_cross_modal_disagreement <= 1.0:
            raise ValueError("cross-modal threshold must be in [0, 1]")
        if not 0.0 <= self.max_candidate_js <= 1.0:
            raise ValueError("JS threshold must be in [0, 1]")
        if not 0.0 <= self.max_candidate_risk_gap <= 1.0:
            raise ValueError("risk-gap threshold must be in [0, 1]")
        if not 0.0 <= self.min_required_health <= 1.0:
            raise ValueError("health threshold must be in [0, 1]")


def split_conformal_quantile(
    residuals: Sequence[float] | np.ndarray,
    alpha: float = 0.10,
) -> float:
    """Return the finite-sample split-conformal upper residual quantile."""

    values = np.asarray(residuals, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("residuals must be a non-empty finite sequence")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    rank = int(math.ceil((values.size + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), values.size)
    return float(np.partition(values, rank - 1)[rank - 1])


def conformal_risk_upper(
    predicted_risk: Sequence[float] | np.ndarray,
    residual_quantile: float,
) -> np.ndarray:
    """Calibrate fixed-candidate risk probabilities with a bounded upper set."""

    risk = np.asarray(predicted_risk, dtype=np.float64).reshape(-1)
    if risk.size == 0 or not np.isfinite(risk).all():
        raise ValueError("predicted_risk must be finite and non-empty")
    if np.any((risk < 0.0) | (risk > 1.0)):
        raise ValueError("predicted_risk must be in [0, 1]")
    if not math.isfinite(residual_quantile) or residual_quantile < 0.0:
        raise ValueError("residual_quantile must be finite and non-negative")
    return np.clip(risk + residual_quantile, 0.0, 1.0)


def trajectory_error(
    proposed_xy: np.ndarray,
    reference_xy: np.ndarray,
) -> dict[str, Any]:
    """Compute ADE/FDE for equally sampled shadow trajectories."""

    proposed = np.asarray(proposed_xy, dtype=np.float64)
    reference = np.asarray(reference_xy, dtype=np.float64)
    if (
        proposed.ndim != 2
        or proposed.shape[1] != 2
        or proposed.shape != reference.shape
        or proposed.shape[0] == 0
        or not np.isfinite(proposed).all()
        or not np.isfinite(reference).all()
    ):
        return {
            "valid": False,
            "reason": "trajectories_must_be_matching_finite_nx2",
            "shadow_only": True,
            "cmd_vel_authority": False,
        }
    distances = np.linalg.norm(proposed - reference, axis=1)
    return {
        "valid": True,
        "ade_m": float(np.mean(distances)),
        "fde_m": float(distances[-1]),
        "sample_count": int(distances.size),
        "shadow_only": True,
        "cmd_vel_authority": False,
    }


def evaluate_shadow_guard_v2(
    *,
    thresholds: GuardThresholdsV2,
    required_sensor_health: Mapping[str, float],
    candidate_logits: Sequence[float] | np.ndarray,
    cpu_candidate_probabilities: Sequence[float] | np.ndarray,
    cross_modal_disagreement: float | None,
    conformal_residual_quantile: float,
    warm: bool,
    model_ready: bool,
) -> dict[str, Any]:
    """Fuse calibrated checks into a non-authoritative runtime state."""

    reasons: list[str] = []
    if not model_ready:
        return {
            "state": GuardState.MONITOR_OFFLINE.value,
            "trusted": False,
            "reasons": ["model_not_ready"],
            "shadow_only": True,
            "cmd_vel_authority": False,
        }
    if not warm:
        reasons.append("temporal_warmup")

    health_values = {
        str(name): float(value)
        for name, value in required_sensor_health.items()
    }
    if not health_values:
        reasons.append("required_sensor_health_missing")
        minimum_health = 0.0
    elif not all(
        math.isfinite(value) and 0.0 <= value <= 1.0
        for value in health_values.values()
    ):
        reasons.append("required_sensor_health_invalid")
        minimum_health = 0.0
    else:
        minimum_health = min(health_values.values())
        if minimum_health < thresholds.min_required_health:
            reasons.append("required_sensor_degraded")

    logits = np.asarray(candidate_logits, dtype=np.float64).reshape(-1)
    cpu_probabilities = np.asarray(
        cpu_candidate_probabilities,
        dtype=np.float64,
    ).reshape(-1)
    if (
        logits.size == 0
        or logits.shape != cpu_probabilities.shape
        or not np.isfinite(logits).all()
        or not np.isfinite(cpu_probabilities).all()
        or np.any((cpu_probabilities < 0.0) | (cpu_probabilities > 1.0))
    ):
        return {
            "state": GuardState.MONITOR_OFFLINE.value,
            "trusted": False,
            "reasons": ["candidate_contract_invalid"],
            "shadow_only": True,
            "cmd_vel_authority": False,
        }

    predicted = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    calibrated_upper = conformal_risk_upper(
        predicted,
        conformal_residual_quantile,
    )
    energy_result = energy_ood(
        logits,
        threshold=thresholds.energy_ood_threshold,
        higher_is_ood=True,
    )
    if not energy_result.get("valid") or energy_result.get("is_ood") is None:
        reasons.append("energy_ood_unavailable")
    elif energy_result.get("is_ood"):
        reasons.append("energy_ood")

    js_result = trajectory_token_js_divergence(
        1.0 - predicted,
        1.0 - cpu_probabilities,
        inputs_are_logits=False,
    )
    if not js_result.get("valid"):
        reasons.append("candidate_js_unavailable")
    elif float(js_result["js_normalized"]) > thresholds.max_candidate_js:
        reasons.append("candidate_js_high")

    risk_gap = float(np.max(np.abs(predicted - cpu_probabilities)))
    if risk_gap > thresholds.max_candidate_risk_gap:
        reasons.append("candidate_risk_gap_high")

    if cross_modal_disagreement is None or not math.isfinite(
        cross_modal_disagreement
    ):
        reasons.append("cross_modal_disagreement_unavailable")
    elif not 0.0 <= cross_modal_disagreement <= 1.0:
        reasons.append("cross_modal_disagreement_invalid")
    elif (
        cross_modal_disagreement
        > thresholds.max_cross_modal_disagreement
    ):
        reasons.append("cross_modal_disagreement_high")

    state = GuardState.PASSIVE_OK if not reasons else GuardState.REVIEW
    return {
        "state": state.value,
        "trusted": state is GuardState.PASSIVE_OK,
        "reasons": reasons,
        "minimum_required_health": minimum_health,
        "sensor_health": health_values,
        "energy_ood": energy_result,
        "candidate_js": js_result,
        "candidate_risk_gap": risk_gap,
        "candidate_predicted_risk": predicted.tolist(),
        "candidate_conformal_upper": calibrated_upper.tolist(),
        "cpu_candidate_risk": cpu_probabilities.tolist(),
        "cross_modal_disagreement": cross_modal_disagreement,
        "shadow_only": True,
        "cmd_vel_authority": False,
    }
