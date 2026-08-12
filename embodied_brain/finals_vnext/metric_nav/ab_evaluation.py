"""Serializable A/B records for AMCL, slam_toolbox, and MPPI shadow runs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .contracts import RecommendationState, contract_payload, json_ready


class NavigationStack(str, Enum):
    AMCL = "AMCL"
    SLAM_TOOLBOX = "SLAM_TOOLBOX"
    MPPI_SHADOW = "MPPI_SHADOW"


class MetricDirection(str, Enum):
    MINIMIZE = "MINIMIZE"
    MAXIMIZE = "MAXIMIZE"


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
class MetricSpec:
    name: str
    unit: str
    direction: MetricDirection
    required: bool = True
    maximum_regression: float = 0.0
    meaningful_improvement: float = 0.0
    critical: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("metric name must be non-empty")
        if not isinstance(self.unit, str):
            raise TypeError("unit must be a string")
        try:
            direction = MetricDirection(self.direction)
        except ValueError as exc:
            raise ValueError("direction is invalid") from exc
        object.__setattr__(self, "direction", direction)
        _finite(self.maximum_regression, "maximum_regression", minimum=0.0)
        _finite(self.meaningful_improvement, "meaningful_improvement", minimum=0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "direction": self.direction.value,
            "required": self.required,
            "maximum_regression": self.maximum_regression,
            "meaningful_improvement": self.meaningful_improvement,
            "critical": self.critical,
        }


DEFAULT_METRIC_SPECS: dict[NavigationStack, tuple[MetricSpec, ...]] = {
    NavigationStack.AMCL: (
        MetricSpec("pose_rmse_m", "m", MetricDirection.MINIMIZE, True, 0.02, 0.01, True),
        MetricSpec("yaw_rmse_rad", "rad", MetricDirection.MINIMIZE, True, 0.03, 0.01),
        MetricSpec("lost_fraction", "ratio", MetricDirection.MINIMIZE, True, 0.01, 0.01, True),
        MetricSpec(
            "relocalization_success_fraction",
            "ratio",
            MetricDirection.MAXIMIZE,
            True,
            0.02,
            0.02,
        ),
        MetricSpec("latency_p95_ms", "ms", MetricDirection.MINIMIZE, True, 5.0, 2.0),
    ),
    NavigationStack.SLAM_TOOLBOX: (
        MetricSpec("ate_rmse_m", "m", MetricDirection.MINIMIZE, True, 0.03, 0.01, True),
        MetricSpec("endpoint_drift_m", "m", MetricDirection.MINIMIZE, True, 0.04, 0.02, True),
        MetricSpec("map_overlap_iou", "ratio", MetricDirection.MAXIMIZE, True, 0.02, 0.02),
        MetricSpec(
            "loop_closure_precision",
            "ratio",
            MetricDirection.MAXIMIZE,
            True,
            0.02,
            0.02,
            True,
        ),
        MetricSpec("latency_p95_ms", "ms", MetricDirection.MINIMIZE, True, 8.0, 3.0),
    ),
    NavigationStack.MPPI_SHADOW: (
        MetricSpec(
            "collision_fraction",
            "ratio",
            MetricDirection.MINIMIZE,
            True,
            0.0,
            0.01,
            True,
        ),
        MetricSpec(
            "goal_progress_fraction",
            "ratio",
            MetricDirection.MAXIMIZE,
            True,
            0.02,
            0.02,
        ),
        MetricSpec(
            "path_tracking_rmse_m",
            "m",
            MetricDirection.MINIMIZE,
            True,
            0.03,
            0.01,
        ),
        MetricSpec("jerk_rms_m_s3", "m/s^3", MetricDirection.MINIMIZE, True, 0.15, 0.05),
        MetricSpec("latency_p95_ms", "ms", MetricDirection.MINIMIZE, True, 5.0, 2.0),
    ),
}


@dataclass(frozen=True, slots=True)
class ShadowABRun:
    stack: NavigationStack
    run_id: str
    variant_id: str
    dataset_id: str
    configuration_digest: str
    sample_count: int
    duration_s: float
    metrics: Mapping[str, float]
    completed: bool = True
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            stack = NavigationStack(self.stack)
        except ValueError as exc:
            raise ValueError("stack is invalid") from exc
        object.__setattr__(self, "stack", stack)
        for name in ("run_id", "variant_id", "dataset_id", "configuration_digest"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.sample_count, bool) or self.sample_count < 0:
            raise ValueError("sample_count must be a non-negative integer")
        _finite(self.duration_s, "duration_s", minimum=0.0)
        converted_metrics: dict[str, float] = {}
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("metric names must be non-empty strings")
            converted_metrics[name] = _finite(value, f"metrics.{name}")
        object.__setattr__(self, "metrics", converted_metrics)
        object.__setattr__(self, "provenance", json_ready(dict(self.provenance)))

    def to_dict(self) -> dict[str, Any]:
        return contract_payload(
            "metric_nav.shadow_ab_run",
            stack=self.stack.value,
            run_id=self.run_id,
            variant_id=self.variant_id,
            dataset_id=self.dataset_id,
            configuration_digest=self.configuration_digest,
            sample_count=self.sample_count,
            duration_s=self.duration_s,
            metrics=dict(self.metrics),
            completed=self.completed,
            provenance=dict(self.provenance),
        )


@dataclass(frozen=True, slots=True)
class ABPolicy:
    metric_specs: tuple[MetricSpec, ...]
    minimum_samples: int = 30
    require_same_dataset: bool = True

    def __post_init__(self) -> None:
        if not self.metric_specs:
            raise ValueError("metric_specs must not be empty")
        if any(not isinstance(spec, MetricSpec) for spec in self.metric_specs):
            raise TypeError("metric_specs must contain MetricSpec values")
        names = [spec.name for spec in self.metric_specs]
        if len(names) != len(set(names)):
            raise ValueError("metric_specs names must be unique")
        if isinstance(self.minimum_samples, bool) or self.minimum_samples <= 0:
            raise ValueError("minimum_samples must be a positive integer")


def default_ab_policy(stack: NavigationStack | str) -> ABPolicy:
    selected = NavigationStack(stack)
    return ABPolicy(metric_specs=DEFAULT_METRIC_SPECS[selected])


@dataclass(frozen=True, slots=True)
class MetricComparison:
    name: str
    unit: str
    direction: MetricDirection
    baseline: float
    candidate: float
    signed_improvement: float
    regressed: bool
    meaningfully_improved: bool
    critical: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "direction": self.direction.value,
            "baseline": self.baseline,
            "candidate": self.candidate,
            "signed_improvement": self.signed_improvement,
            "regressed": self.regressed,
            "meaningfully_improved": self.meaningfully_improved,
            "critical": self.critical,
        }


@dataclass(frozen=True, slots=True)
class ABRecommendation:
    stack: NavigationStack
    baseline_run_id: str
    candidate_run_id: str
    state: RecommendationState
    comparisons: tuple[MetricComparison, ...]
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return contract_payload(
            "metric_nav.ab_recommendation",
            stack=self.stack.value,
            baseline_run_id=self.baseline_run_id,
            candidate_run_id=self.candidate_run_id,
            state=self.state.value,
            comparisons=[comparison.to_dict() for comparison in self.comparisons],
            reasons=list(self.reasons),
            activation_authorized=False,
        )


def compare_shadow_runs(
    baseline: ShadowABRun,
    candidate: ShadowABRun,
    *,
    policy: ABPolicy | None = None,
) -> ABRecommendation:
    """Compare two passive runs and recommend review, never activation."""

    if not isinstance(baseline, ShadowABRun) or not isinstance(candidate, ShadowABRun):
        raise TypeError("baseline and candidate must be ShadowABRun instances")
    if baseline.stack is not candidate.stack:
        raise ValueError("baseline and candidate stacks differ")
    selected = policy or default_ab_policy(baseline.stack)
    reasons: list[str] = []
    if selected.require_same_dataset and baseline.dataset_id != candidate.dataset_id:
        reasons.append("dataset_mismatch")
    if not baseline.completed:
        reasons.append("baseline_incomplete")
    if not candidate.completed:
        reasons.append("candidate_incomplete")
    if baseline.sample_count < selected.minimum_samples:
        reasons.append("baseline_sample_count_below_minimum")
    if candidate.sample_count < selected.minimum_samples:
        reasons.append("candidate_sample_count_below_minimum")

    missing = [
        spec.name
        for spec in selected.metric_specs
        if spec.required
        and (
            spec.name not in baseline.metrics
            or spec.name not in candidate.metrics
        )
    ]
    if missing:
        reasons.extend(f"missing_required_metric:{name}" for name in missing)

    comparisons: list[MetricComparison] = []
    for spec in selected.metric_specs:
        if spec.name not in baseline.metrics or spec.name not in candidate.metrics:
            continue
        baseline_value = baseline.metrics[spec.name]
        candidate_value = candidate.metrics[spec.name]
        signed_improvement = (
            baseline_value - candidate_value
            if spec.direction is MetricDirection.MINIMIZE
            else candidate_value - baseline_value
        )
        comparisons.append(
            MetricComparison(
                name=spec.name,
                unit=spec.unit,
                direction=spec.direction,
                baseline=baseline_value,
                candidate=candidate_value,
                signed_improvement=signed_improvement,
                regressed=signed_improvement < -spec.maximum_regression,
                meaningfully_improved=(
                    signed_improvement >= spec.meaningful_improvement
                    and spec.meaningful_improvement > 0.0
                ),
                critical=spec.critical,
            )
        )

    insufficient = any(
        reason
        for reason in reasons
        if reason == "dataset_mismatch"
        or reason.endswith("_incomplete")
        or "sample_count_below_minimum" in reason
        or reason.startswith("missing_required_metric:")
    )
    regressions = [comparison for comparison in comparisons if comparison.regressed]
    critical_regressions = [
        comparison for comparison in regressions if comparison.critical
    ]
    improvements = [
        comparison for comparison in comparisons if comparison.meaningfully_improved
    ]
    if insufficient:
        state = RecommendationState.INSUFFICIENT_DATA
    elif critical_regressions:
        state = RecommendationState.REJECT
        reasons.extend(
            f"critical_regression:{comparison.name}"
            for comparison in critical_regressions
        )
    elif regressions:
        state = RecommendationState.REJECT
        reasons.extend(f"regression:{comparison.name}" for comparison in regressions)
    elif improvements:
        state = RecommendationState.RECOMMEND
        reasons.extend(
            f"meaningful_improvement:{comparison.name}"
            for comparison in improvements
        )
    else:
        state = RecommendationState.HOLD
        reasons.append("no_meaningful_improvement")
    return ABRecommendation(
        stack=baseline.stack,
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        state=state,
        comparisons=tuple(comparisons),
        reasons=tuple(dict.fromkeys(reasons)),
    )


@dataclass(frozen=True, slots=True)
class MultiStackRecommendation:
    state: RecommendationState
    recommendations: tuple[ABRecommendation, ...]
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return contract_payload(
            "metric_nav.multi_stack_recommendation",
            state=self.state.value,
            recommendations=[
                recommendation.to_dict() for recommendation in self.recommendations
            ],
            reasons=list(self.reasons),
            activation_authorized=False,
        )


def summarize_stack_recommendations(
    recommendations: Sequence[ABRecommendation],
) -> MultiStackRecommendation:
    """Combine AMCL, slam_toolbox, and MPPI shadow recommendations."""

    values = tuple(recommendations)
    if not values:
        return MultiStackRecommendation(
            RecommendationState.INSUFFICIENT_DATA,
            (),
            ("missing_recommendations",),
        )
    if any(not isinstance(value, ABRecommendation) for value in values):
        raise TypeError("recommendations must contain ABRecommendation values")
    stacks = [value.stack for value in values]
    if len(stacks) != len(set(stacks)):
        raise ValueError("recommendations must have unique stacks")
    states = {value.state for value in values}
    reasons: list[str] = []
    if RecommendationState.REJECT in states:
        state = RecommendationState.REJECT
        reasons.append("at_least_one_stack_rejected")
    elif RecommendationState.INSUFFICIENT_DATA in states:
        state = RecommendationState.INSUFFICIENT_DATA
        reasons.append("at_least_one_stack_has_insufficient_data")
    elif states == {RecommendationState.RECOMMEND}:
        state = RecommendationState.RECOMMEND
        reasons.append("all_evaluated_stacks_recommend_candidate")
    else:
        state = RecommendationState.HOLD
        reasons.append("mixed_or_hold_recommendations")
    return MultiStackRecommendation(state, values, tuple(reasons))
