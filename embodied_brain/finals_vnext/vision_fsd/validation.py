"""Freshness and quality gates for supplied semantic BEV observations."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from .contracts import (
    ContractError,
    ProvenanceState,
    SemanticBEVFrame,
)


class FrameRejectedError(ContractError):
    """Raised when a structurally valid frame is not safe to consume."""

    def __init__(self, assessment: "FrameAssessment") -> None:
        self.assessment = assessment
        super().__init__(
            "semantic BEV frame rejected: " + ", ".join(assessment.reasons)
        )


@dataclass(frozen=True, slots=True)
class FreshnessQualityPolicy:
    """Runtime acceptance policy for the read-only cross-X5 bridge."""

    max_age_s: float = 0.75
    max_future_skew_s: float = 0.05
    max_pipeline_latency_s: float = 0.50
    max_packaging_latency_s: float = 0.25
    min_overall_score: float = 0.65
    min_projection_valid_fraction: float = 0.40
    min_visible_fraction: float = 0.02
    min_mean_confidence: float = 0.30
    max_dropped_input_fraction: float = 0.05
    min_source_width: int = 3840
    min_source_height: int = 2160
    expected_target_frame: str = "base_link"
    require_model_sha256: bool = True
    require_input_sha256: bool = True
    accepted_states: tuple[ProvenanceState, ...] = (
        ProvenanceState.LIVE_CAMERA,
        ProvenanceState.CACHED_CAMERA,
    )

    def __post_init__(self) -> None:
        positive = (
            ("max_age_s", self.max_age_s),
            ("max_future_skew_s", self.max_future_skew_s),
            ("max_pipeline_latency_s", self.max_pipeline_latency_s),
            ("max_packaging_latency_s", self.max_packaging_latency_s),
        )
        for name, value in positive:
            if not isfinite(float(value)) or float(value) < 0.0:
                raise ContractError(f"{name} must be finite and non-negative")
        probabilities = (
            ("min_overall_score", self.min_overall_score),
            (
                "min_projection_valid_fraction",
                self.min_projection_valid_fraction,
            ),
            ("min_visible_fraction", self.min_visible_fraction),
            ("min_mean_confidence", self.min_mean_confidence),
            (
                "max_dropped_input_fraction",
                self.max_dropped_input_fraction,
            ),
        )
        for name, value in probabilities:
            if not isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ContractError(f"{name} must be finite and in [0, 1]")
        if (
            not isinstance(self.min_source_width, int)
            or isinstance(self.min_source_width, bool)
            or self.min_source_width <= 0
            or not isinstance(self.min_source_height, int)
            or isinstance(self.min_source_height, bool)
            or self.min_source_height <= 0
        ):
            raise ContractError("minimum source dimensions must be positive integers")
        if not isinstance(self.expected_target_frame, str) or not (
            self.expected_target_frame.strip()
        ):
            raise ContractError("expected_target_frame must not be empty")
        try:
            states = tuple(ProvenanceState(state) for state in self.accepted_states)
        except ValueError as exc:
            raise ContractError("accepted_states contains an invalid state") from exc
        if not states:
            raise ContractError("accepted_states must not be empty")
        object.__setattr__(self, "accepted_states", states)


@dataclass(frozen=True, slots=True)
class FrameAssessment:
    """Complete, non-mutating result of a freshness and quality check."""

    accepted: bool
    reasons: tuple[str, ...]
    age_s: float
    pipeline_latency_s: float
    packaging_latency_s: float
    actual_visible_fraction: float
    actual_mean_confidence: float


def assess_frame(
    frame: SemanticBEVFrame,
    *,
    now_s: float,
    policy: FreshnessQualityPolicy | None = None,
) -> FrameAssessment:
    """Check a frame without mutating it or any downstream memory."""

    if not isinstance(frame, SemanticBEVFrame):
        raise TypeError("frame must be SemanticBEVFrame")
    current_time = float(now_s)
    if not isfinite(current_time):
        raise ContractError("now_s must be finite")
    cfg = policy or FreshnessQualityPolicy()
    if not isinstance(cfg, FreshnessQualityPolicy):
        raise TypeError("policy must be FreshnessQualityPolicy")

    reasons: list[str] = []
    age_s = current_time - frame.timestamp_s
    pipeline_latency_s = (
        frame.provenance.inference_timestamp_s
        - frame.provenance.capture_timestamp_s
    )
    packaging_latency_s = (
        frame.timestamp_s - frame.provenance.inference_timestamp_s
    )
    actual_visible = float(np.mean(frame.visibility, dtype=np.float64))
    actual_confidence = (
        float(
            np.mean(
                frame.confidence[frame.visibility],
                dtype=np.float64,
            )
        )
        if frame.visibility.any()
        else 0.0
    )

    if age_s > cfg.max_age_s:
        reasons.append("STALE_FRAME")
    if age_s < -cfg.max_future_skew_s:
        reasons.append("FUTURE_TIMESTAMP")
    if pipeline_latency_s > cfg.max_pipeline_latency_s:
        reasons.append("PIPELINE_LATENCY")
    if packaging_latency_s > cfg.max_packaging_latency_s:
        reasons.append("PACKAGING_LATENCY")
    if frame.provenance.state not in cfg.accepted_states:
        reasons.append("PROVENANCE_NOT_ACCEPTED")
    if cfg.require_model_sha256 and frame.provenance.model_sha256 is None:
        reasons.append("MODEL_DIGEST_MISSING")
    if cfg.require_input_sha256 and frame.provenance.input_sha256 is None:
        reasons.append("INPUT_DIGEST_MISSING")
    if frame.intrinsics.width < cfg.min_source_width:
        reasons.append("SOURCE_WIDTH_BELOW_4K")
    if frame.intrinsics.height < cfg.min_source_height:
        reasons.append("SOURCE_HEIGHT_BELOW_4K")
    if frame.extrinsics.target_frame != cfg.expected_target_frame:
        reasons.append("EXTRINSICS_TARGET_MISMATCH")
    if frame.quality.overall_score < cfg.min_overall_score:
        reasons.append("OVERALL_QUALITY")
    if (
        frame.quality.projection_valid_fraction
        < cfg.min_projection_valid_fraction
    ):
        reasons.append("PROJECTION_COVERAGE")
    if actual_visible < cfg.min_visible_fraction:
        reasons.append("VISIBILITY_COVERAGE")
    if actual_confidence < cfg.min_mean_confidence:
        reasons.append("MEAN_CONFIDENCE")
    if (
        frame.quality.dropped_input_fraction
        > cfg.max_dropped_input_fraction
    ):
        reasons.append("DROPPED_INPUT")

    return FrameAssessment(
        accepted=not reasons,
        reasons=tuple(reasons),
        age_s=float(age_s),
        pipeline_latency_s=float(pipeline_latency_s),
        packaging_latency_s=float(packaging_latency_s),
        actual_visible_fraction=actual_visible,
        actual_mean_confidence=actual_confidence,
    )


def require_acceptable_frame(
    frame: SemanticBEVFrame,
    *,
    now_s: float,
    policy: FreshnessQualityPolicy | None = None,
) -> FrameAssessment:
    """Return the assessment or raise without updating downstream state."""

    assessment = assess_frame(frame, now_s=now_s, policy=policy)
    if not assessment.accepted:
        raise FrameRejectedError(assessment)
    return assessment


__all__ = [
    "FrameAssessment",
    "FrameRejectedError",
    "FreshnessQualityPolicy",
    "assess_frame",
    "require_acceptable_frame",
]
