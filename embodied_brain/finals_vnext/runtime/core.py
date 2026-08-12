"""Read-only temporal world-model and ShadowGuard orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from embodied_brain.finals_successor.x5_tribev_flow.contracts import (
    OdometryDelta,
)
from embodied_brain.finals_vnext.fusion import (
    FusionInputsV2,
    TemporalTriBEVV2,
    build_fusion_frame,
)
from embodied_brain.finals_vnext.guard_v2 import (
    GuardThresholdsV2,
    evaluate_shadow_guard_v2,
)
from embodied_brain.finals_vnext.world_model.trajectories import (
    rectangular_footprint_risk_labels,
)

from .backend import DiagnosticBackend, ModelOutputsV2


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _cross_modal_disagreement(channels: np.ndarray) -> float | None:
    modality = (
        channels[0] >= 0.5,
        np.maximum.reduce((channels[2], channels[3], channels[4])) >= 0.5,
        channels[8] >= 0.5,
    )
    nonempty = [value for value in modality if np.any(value)]
    if len(nonempty) < 2:
        return None
    disagreements = []
    for first_index, first in enumerate(nonempty):
        for second in nonempty[first_index + 1 :]:
            union = np.logical_or(first, second).sum()
            if union:
                intersection = np.logical_and(first, second).sum()
                disagreements.append(1.0 - float(intersection / union))
    return float(np.mean(disagreements)) if disagreements else None


@dataclass(frozen=True, slots=True)
class ShadowDiagnosticsV2:
    timestamp_s: float
    state: str
    warm: bool
    inference_latency_ms: float | None
    model_identity: Mapping[str, object]
    model_outputs: ModelOutputsV2 | None
    cpu_candidate_risk: np.ndarray | None
    guard: Mapping[str, Any]
    history: Mapping[str, object]
    shadow_only: bool = True
    motion_authority: bool = False


class ShadowRuntimeV2:
    def __init__(
        self,
        backend: DiagnosticBackend,
        *,
        thresholds: GuardThresholdsV2,
        conformal_residual_quantile: float,
    ) -> None:
        self.backend = backend
        self.thresholds = thresholds
        self.conformal_residual_quantile = float(
            conformal_residual_quantile
        )
        self.temporal = TemporalTriBEVV2()

    def reset(self) -> None:
        self.temporal.reset()

    def observe(
        self,
        inputs: FusionInputsV2,
        *,
        ego_delta: OdometryDelta | None = None,
        odometry_health: float = 1.0,
    ) -> ShadowDiagnosticsV2:
        if not np.isfinite(odometry_health) or not 0.0 <= odometry_health <= 1.0:
            raise ValueError("odometry_health must be in [0, 1]")
        frame = build_fusion_frame(inputs)
        model_input, history = self.temporal.update(frame, ego_delta)
        warm = bool(history["warm"])
        if not self.backend.ready:
            guard = {
                "state": "MONITOR_OFFLINE",
                "trusted": False,
                "reasons": ["model_not_ready"],
                "shadow_only": True,
                "motion_authority": False,
            }
            return ShadowDiagnosticsV2(
                timestamp_s=inputs.timestamp_s,
                state="MONITOR_OFFLINE",
                warm=warm,
                inference_latency_ms=None,
                model_identity=self.backend.identity,
                model_outputs=None,
                cpu_candidate_risk=None,
                guard=guard,
                history=history,
            )
        if not warm:
            guard = {
                "state": "REVIEW",
                "trusted": False,
                "reasons": ["temporal_warmup"],
                "shadow_only": True,
                "motion_authority": False,
            }
            return ShadowDiagnosticsV2(
                timestamp_s=inputs.timestamp_s,
                state="REVIEW",
                warm=False,
                inference_latency_ms=None,
                model_identity=self.backend.identity,
                model_outputs=None,
                cpu_candidate_risk=None,
                guard=guard,
                history=history,
            )

        started = time.perf_counter()
        outputs = self.backend.infer(model_input)
        latency_ms = (time.perf_counter() - started) * 1000.0
        persistence = np.repeat(frame.channels[11:12], 3, axis=0)
        cpu_risk = rectangular_footprint_risk_labels(persistence)
        health = {
            "lidar_geometry": frame.source_validity[0],
            "depth_geometry": frame.source_validity[1],
            "vision_semantics": frame.source_validity[2],
            "odometry_alignment": float(odometry_health),
        }
        guard = evaluate_shadow_guard_v2(
            thresholds=self.thresholds,
            required_sensor_health=health,
            candidate_logits=outputs.trajectory_risk_logits,
            cpu_candidate_probabilities=cpu_risk,
            cross_modal_disagreement=_cross_modal_disagreement(frame.channels),
            conformal_residual_quantile=self.conformal_residual_quantile,
            warm=True,
            model_ready=True,
        )
        return ShadowDiagnosticsV2(
            timestamp_s=inputs.timestamp_s,
            state=str(guard["state"]),
            warm=True,
            inference_latency_ms=latency_ms,
            model_identity=self.backend.identity,
            model_outputs=outputs,
            cpu_candidate_risk=cpu_risk,
            guard=guard,
            history=history,
        )


__all__ = ["ShadowDiagnosticsV2", "ShadowRuntimeV2"]
