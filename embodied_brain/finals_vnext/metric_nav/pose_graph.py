"""Pose-graph and loop-closure event contracts for offline evidence."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .contracts import DiagnosticState, contract_payload, json_ready, worst_state


class PoseGraphEventType(str, Enum):
    LOCAL_SCAN_MATCH = "LOCAL_SCAN_MATCH"
    LOOP_CLOSURE = "LOOP_CLOSURE"
    GRAPH_OPTIMIZATION = "GRAPH_OPTIMIZATION"
    RELOCALIZATION = "RELOCALIZATION"


def _finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


@dataclass(frozen=True, slots=True)
class PoseGraphEvent:
    run_id: str
    event_id: str
    timestamp_s: float
    backend: str
    event_type: PoseGraphEventType
    source_node: int
    target_node: int
    accepted_by_backend: bool
    residual_before: float
    residual_after: float
    chi2: float
    degrees_of_freedom: int
    scan_overlap: float
    inlier_fraction: float
    correction_translation_m: float
    correction_yaw_rad: float
    latency_ms: float = 0.0
    covariance_diagonal: tuple[float, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("run_id", "event_id", "backend"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")
        try:
            event_type = PoseGraphEventType(self.event_type)
        except ValueError as exc:
            raise ValueError("event_type is invalid") from exc
        object.__setattr__(self, "event_type", event_type)
        if isinstance(self.source_node, bool) or self.source_node < 0:
            raise ValueError("source_node must be a non-negative integer")
        if isinstance(self.target_node, bool) or self.target_node < 0:
            raise ValueError("target_node must be a non-negative integer")
        if isinstance(self.degrees_of_freedom, bool) or self.degrees_of_freedom <= 0:
            raise ValueError("degrees_of_freedom must be a positive integer")
        for name in (
            "timestamp_s",
            "residual_before",
            "residual_after",
            "chi2",
            "correction_translation_m",
            "latency_ms",
        ):
            _finite(getattr(self, name), name, minimum=0.0)
        _finite(self.correction_yaw_rad, "correction_yaw_rad")
        for name in ("scan_overlap", "inlier_fraction"):
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        covariance = tuple(
            _finite(value, f"covariance_diagonal[{index}]", minimum=0.0)
            for index, value in enumerate(self.covariance_diagonal)
        )
        object.__setattr__(self, "covariance_diagonal", covariance)
        object.__setattr__(self, "provenance", json_ready(dict(self.provenance)))

    def to_dict(self) -> dict[str, Any]:
        return contract_payload(
            "metric_nav.pose_graph_event",
            run_id=self.run_id,
            event_id=self.event_id,
            timestamp_s=self.timestamp_s,
            backend=self.backend,
            event_type=self.event_type.value,
            source_node=self.source_node,
            target_node=self.target_node,
            accepted_by_backend=self.accepted_by_backend,
            residual_before=self.residual_before,
            residual_after=self.residual_after,
            chi2=self.chi2,
            degrees_of_freedom=self.degrees_of_freedom,
            scan_overlap=self.scan_overlap,
            inlier_fraction=self.inlier_fraction,
            correction_translation_m=self.correction_translation_m,
            correction_yaw_rad=self.correction_yaw_rad,
            latency_ms=self.latency_ms,
            covariance_diagonal=list(self.covariance_diagonal),
            provenance=dict(self.provenance),
        )


@dataclass(frozen=True, slots=True)
class LoopClosurePolicy:
    min_scan_overlap: float = 0.55
    min_inlier_fraction: float = 0.60
    max_normalized_chi2: float = 3.0
    min_residual_reduction_fraction: float = 0.05
    max_translation_correction_m: float = 2.0
    max_yaw_correction_rad: float = math.radians(60.0)

    def __post_init__(self) -> None:
        for name in (
            "min_scan_overlap",
            "min_inlier_fraction",
            "min_residual_reduction_fraction",
        ):
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        for name in (
            "max_normalized_chi2",
            "max_translation_correction_m",
            "max_yaw_correction_rad",
        ):
            if _finite(getattr(self, name), name) <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class PoseGraphEventQuality:
    event_id: str
    state: DiagnosticState
    normalized_chi2: float
    residual_reduction_fraction: float
    trusted_for_analysis: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return contract_payload(
            "metric_nav.pose_graph_event_quality",
            event_id=self.event_id,
            state=self.state.value,
            normalized_chi2=self.normalized_chi2,
            residual_reduction_fraction=self.residual_reduction_fraction,
            trusted_for_analysis=self.trusted_for_analysis,
            reasons=list(self.reasons),
        )


def assess_pose_graph_event(
    event: PoseGraphEvent,
    *,
    policy: LoopClosurePolicy | None = None,
) -> PoseGraphEventQuality:
    """Assess event evidence without accepting or rejecting it in the backend."""

    if not isinstance(event, PoseGraphEvent):
        raise TypeError("event must be a PoseGraphEvent")
    selected = policy or LoopClosurePolicy()
    normalized_chi2 = event.chi2 / event.degrees_of_freedom
    if event.residual_before > 1e-12:
        reduction = (event.residual_before - event.residual_after) / event.residual_before
    else:
        reduction = 0.0 if event.residual_after <= 1e-12 else -1.0
    reasons: list[str] = []
    if event.scan_overlap < selected.min_scan_overlap:
        reasons.append("scan_overlap_below_minimum")
    if event.inlier_fraction < selected.min_inlier_fraction:
        reasons.append("inlier_fraction_below_minimum")
    if normalized_chi2 > selected.max_normalized_chi2:
        reasons.append("normalized_chi2_above_maximum")
    if event.accepted_by_backend and reduction < selected.min_residual_reduction_fraction:
        reasons.append("accepted_without_residual_improvement")
    if event.residual_after > event.residual_before + 1e-12:
        reasons.append("residual_increased")
    if event.correction_translation_m > selected.max_translation_correction_m:
        reasons.append("translation_correction_requires_review")
    if abs(event.correction_yaw_rad) > selected.max_yaw_correction_rad:
        reasons.append("yaw_correction_requires_review")
    if not event.accepted_by_backend:
        reasons.append("backend_rejected_event")

    fatal = {
        "normalized_chi2_above_maximum",
        "accepted_without_residual_improvement",
        "residual_increased",
    }
    if event.accepted_by_backend and any(reason in fatal for reason in reasons):
        state = DiagnosticState.UNHEALTHY
    elif reasons:
        state = DiagnosticState.DEGRADED
    else:
        state = DiagnosticState.HEALTHY
    trusted = event.accepted_by_backend and state is DiagnosticState.HEALTHY
    return PoseGraphEventQuality(
        event_id=event.event_id,
        state=state,
        normalized_chi2=normalized_chi2,
        residual_reduction_fraction=reduction,
        trusted_for_analysis=trusted,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True, slots=True)
class PoseGraphRunSummary:
    run_id: str
    state: DiagnosticState
    event_count: int
    accepted_count: int
    trusted_count: int
    mean_normalized_chi2: float | None
    mean_residual_reduction_fraction: float | None
    assessments: tuple[PoseGraphEventQuality, ...]
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return contract_payload(
            "metric_nav.pose_graph_run_summary",
            run_id=self.run_id,
            state=self.state.value,
            event_count=self.event_count,
            accepted_count=self.accepted_count,
            trusted_count=self.trusted_count,
            mean_normalized_chi2=self.mean_normalized_chi2,
            mean_residual_reduction_fraction=self.mean_residual_reduction_fraction,
            assessments=[assessment.to_dict() for assessment in self.assessments],
            reasons=list(self.reasons),
        )


def summarize_pose_graph_events(
    events: Sequence[PoseGraphEvent],
    *,
    policy: LoopClosurePolicy | None = None,
) -> PoseGraphRunSummary:
    """Build a serializable run summary from one pose-graph event stream."""

    event_list = list(events)
    if not event_list:
        return PoseGraphRunSummary(
            run_id="UNKNOWN",
            state=DiagnosticState.INSUFFICIENT_DATA,
            event_count=0,
            accepted_count=0,
            trusted_count=0,
            mean_normalized_chi2=None,
            mean_residual_reduction_fraction=None,
            assessments=(),
            reasons=("missing_pose_graph_events",),
        )
    if any(not isinstance(event, PoseGraphEvent) for event in event_list):
        raise TypeError("all events must be PoseGraphEvent instances")
    run_ids = {event.run_id for event in event_list}
    if len(run_ids) != 1:
        raise ValueError("all events in a summary must share run_id")
    assessments = tuple(
        assess_pose_graph_event(event, policy=policy) for event in event_list
    )
    accepted_count = sum(event.accepted_by_backend for event in event_list)
    trusted_count = sum(assessment.trusted_for_analysis for assessment in assessments)
    reasons: list[str] = []
    if accepted_count and trusted_count == 0:
        reasons.append("no_trusted_accepted_events")
    states = [assessment.state for assessment in assessments]
    state = worst_state(states)
    return PoseGraphRunSummary(
        run_id=event_list[0].run_id,
        state=state,
        event_count=len(event_list),
        accepted_count=accepted_count,
        trusted_count=trusted_count,
        mean_normalized_chi2=statistics.fmean(
            assessment.normalized_chi2 for assessment in assessments
        ),
        mean_residual_reduction_fraction=statistics.fmean(
            assessment.residual_reduction_fraction for assessment in assessments
        ),
        assessments=assessments,
        reasons=tuple(reasons),
    )
