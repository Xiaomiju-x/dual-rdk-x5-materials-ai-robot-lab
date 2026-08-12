"""Offline LiDAR overlap and planar-geometry degeneracy diagnostics."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .contracts import DiagnosticState, contract_payload


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires data")
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


@dataclass(frozen=True, slots=True)
class ScanQualityPolicy:
    min_range_m: float = 0.08
    max_range_m: float = 12.0
    max_pair_residual_m: float = 0.20
    min_points: int = 30
    min_valid_fraction: float = 0.20
    min_common_fraction: float = 0.35
    min_overlap_fraction: float = 0.55
    sector_count: int = 12
    min_sector_coverage: float = 0.40
    min_spatial_span_m: float = 0.50
    min_minor_eigenvalue_m2: float = 0.005
    max_geometry_condition_number: float = 120.0

    def __post_init__(self) -> None:
        finite_positive = (
            "min_range_m",
            "max_range_m",
            "max_pair_residual_m",
            "min_spatial_span_m",
            "min_minor_eigenvalue_m2",
            "max_geometry_condition_number",
        )
        for name in finite_positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_range_m <= self.min_range_m:
            raise ValueError("max_range_m must exceed min_range_m")
        if isinstance(self.min_points, bool) or self.min_points <= 2:
            raise ValueError("min_points must be an integer > 2")
        if isinstance(self.sector_count, bool) or self.sector_count < 4:
            raise ValueError("sector_count must be an integer >= 4")
        for name in (
            "min_valid_fraction",
            "min_common_fraction",
            "min_overlap_fraction",
            "min_sector_coverage",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class ScanQuality:
    state: DiagnosticState
    beam_count: int
    reference_valid_count: int
    current_valid_count: int
    common_valid_count: int
    valid_fraction: float
    common_fraction: float
    overlap_fraction: float
    median_abs_residual_m: float | None
    p95_abs_residual_m: float | None
    sector_coverage: float
    spatial_span_m: float
    covariance_eigenvalues_m2: tuple[float, float] | tuple[()]
    geometry_condition_number: float | None
    degenerate: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return contract_payload(
            "metric_nav.scan_quality",
            state=self.state.value,
            beam_count=self.beam_count,
            reference_valid_count=self.reference_valid_count,
            current_valid_count=self.current_valid_count,
            common_valid_count=self.common_valid_count,
            valid_fraction=self.valid_fraction,
            common_fraction=self.common_fraction,
            overlap_fraction=self.overlap_fraction,
            median_abs_residual_m=self.median_abs_residual_m,
            p95_abs_residual_m=self.p95_abs_residual_m,
            sector_coverage=self.sector_coverage,
            spatial_span_m=self.spatial_span_m,
            covariance_eigenvalues_m2=list(self.covariance_eigenvalues_m2),
            geometry_condition_number=self.geometry_condition_number,
            degenerate=self.degenerate,
            reasons=list(self.reasons),
        )


def _valid_range(value: Any, policy: ScanQualityPolicy) -> float | None:
    converted = _finite_or_none(value)
    if converted is None:
        return None
    if not policy.min_range_m <= converted <= policy.max_range_m:
        return None
    return converted


def _geometry_metrics(points: Sequence[tuple[float, float]]) -> tuple[
    float,
    float,
    tuple[float, float],
    float | None,
]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    covariance_xx = statistics.fmean((value - mean_x) ** 2 for value in xs)
    covariance_yy = statistics.fmean((value - mean_y) ** 2 for value in ys)
    covariance_xy = statistics.fmean(
        (x - mean_x) * (y - mean_y) for x, y in points
    )
    trace = covariance_xx + covariance_yy
    discriminant = math.sqrt(
        max(
            0.0,
            (covariance_xx - covariance_yy) ** 2 + 4.0 * covariance_xy**2,
        )
    )
    major = max(0.0, 0.5 * (trace + discriminant))
    minor = max(0.0, 0.5 * (trace - discriminant))
    condition = major / minor if minor > 1e-12 else None
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    return x_span, y_span, (minor, major), condition


def assess_scan_quality(
    reference_ranges_m: Sequence[float],
    current_ranges_m: Sequence[float],
    *,
    angle_min_rad: float,
    angle_increment_rad: float,
    policy: ScanQualityPolicy | None = None,
) -> ScanQuality:
    """Assess beam overlap and whether the current scan constrains planar pose.

    The two scans must already share an angular grid.  This function does not
    perform scan matching and therefore cannot alter SLAM state.
    """

    selected = policy or ScanQualityPolicy()
    if isinstance(reference_ranges_m, (str, bytes)) or isinstance(
        current_ranges_m,
        (str, bytes),
    ):
        raise TypeError("range inputs must be numeric sequences")
    reference = list(reference_ranges_m)
    current = list(current_ranges_m)
    if len(reference) != len(current):
        raise ValueError("reference and current scans must have equal beam counts")
    angle_min = float(angle_min_rad)
    angle_increment = float(angle_increment_rad)
    if not math.isfinite(angle_min):
        raise ValueError("angle_min_rad must be finite")
    if not math.isfinite(angle_increment) or angle_increment == 0.0:
        raise ValueError("angle_increment_rad must be finite and non-zero")

    beam_count = len(reference)
    reference_valid = [_valid_range(value, selected) for value in reference]
    current_valid = [_valid_range(value, selected) for value in current]
    reference_count = sum(value is not None for value in reference_valid)
    current_count = sum(value is not None for value in current_valid)
    common_indices = [
        index
        for index, (left, right) in enumerate(
            zip(reference_valid, current_valid, strict=True)
        )
        if left is not None and right is not None
    ]
    common_count = len(common_indices)
    union_count = sum(
        left is not None or right is not None
        for left, right in zip(reference_valid, current_valid, strict=True)
    )
    valid_fraction = current_count / beam_count if beam_count else 0.0
    common_fraction = common_count / union_count if union_count else 0.0
    residuals = [
        abs(reference_valid[index] - current_valid[index])  # type: ignore[operator]
        for index in common_indices
    ]
    matching_count = sum(
        residual <= selected.max_pair_residual_m for residual in residuals
    )
    overlap_fraction = matching_count / common_count if common_count else 0.0
    median_residual = statistics.median(residuals) if residuals else None
    p95_residual = _quantile(residuals, 0.95) if residuals else None

    points: list[tuple[float, float]] = []
    covered_sectors: set[int] = set()
    for index, distance in enumerate(current_valid):
        if distance is None:
            continue
        angle = angle_min + index * angle_increment
        points.append((distance * math.cos(angle), distance * math.sin(angle)))
        normalized = (angle + math.pi) % (2.0 * math.pi)
        sector = min(
            selected.sector_count - 1,
            int(normalized / (2.0 * math.pi) * selected.sector_count),
        )
        covered_sectors.add(sector)
    sector_coverage = len(covered_sectors) / selected.sector_count

    eigenvalues: tuple[float, float] | tuple[()] = ()
    condition: float | None = None
    spatial_span = 0.0
    if len(points) >= 3:
        _, _, eigenvalues, condition = _geometry_metrics(points)
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        spatial_span = math.hypot(max(xs) - min(xs), max(ys) - min(ys))

    reasons: list[str] = []
    if beam_count == 0:
        reasons.append("empty_scan")
    if current_count < selected.min_points:
        reasons.append("too_few_valid_points")
    if valid_fraction < selected.min_valid_fraction:
        reasons.append("valid_fraction_below_minimum")
    if common_fraction < selected.min_common_fraction:
        reasons.append("common_fraction_below_minimum")
    if overlap_fraction < selected.min_overlap_fraction:
        reasons.append("scan_overlap_below_minimum")
    if sector_coverage < selected.min_sector_coverage:
        reasons.append("angular_coverage_below_minimum")
    if spatial_span < selected.min_spatial_span_m:
        reasons.append("spatial_span_below_minimum")
    if eigenvalues:
        minor, _ = eigenvalues
        if minor < selected.min_minor_eigenvalue_m2:
            reasons.append("minor_eigenvalue_below_minimum")
        if condition is None or condition > selected.max_geometry_condition_number:
            reasons.append("geometry_condition_number_above_maximum")

    insufficient_reasons = {
        "empty_scan",
        "too_few_valid_points",
        "valid_fraction_below_minimum",
    }
    degeneracy_reasons = {
        "angular_coverage_below_minimum",
        "spatial_span_below_minimum",
        "minor_eigenvalue_below_minimum",
        "geometry_condition_number_above_maximum",
    }
    degenerate = any(reason in degeneracy_reasons for reason in reasons)
    if any(reason in insufficient_reasons for reason in reasons):
        state = DiagnosticState.INSUFFICIENT_DATA
    elif "scan_overlap_below_minimum" in reasons:
        state = DiagnosticState.UNHEALTHY
    elif reasons:
        state = DiagnosticState.DEGRADED
    else:
        state = DiagnosticState.HEALTHY
    return ScanQuality(
        state=state,
        beam_count=beam_count,
        reference_valid_count=reference_count,
        current_valid_count=current_count,
        common_valid_count=common_count,
        valid_fraction=valid_fraction,
        common_fraction=common_fraction,
        overlap_fraction=overlap_fraction,
        median_abs_residual_m=median_residual,
        p95_abs_residual_m=p95_residual,
        sector_coverage=sector_coverage,
        spatial_span_m=spatial_span,
        covariance_eigenvalues_m2=eigenvalues,
        geometry_condition_number=condition,
        degenerate=degenerate,
        reasons=tuple(reasons),
    )
