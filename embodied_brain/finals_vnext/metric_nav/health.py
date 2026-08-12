"""Passive timing, covariance, innovation, and stationary-drift diagnostics."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .contracts import DiagnosticState, contract_payload, json_ready, worst_state


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_series(values: Sequence[float], name: str) -> list[float]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a numeric sequence")
    return [_finite_float(value, f"{name}[{index}]") for index, value in enumerate(values)]


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _trapezoid_integral(timestamps_s: Sequence[float], values: Sequence[float]) -> float:
    total = 0.0
    for index in range(1, len(timestamps_s)):
        dt = timestamps_s[index] - timestamps_s[index - 1]
        total += 0.5 * (values[index] + values[index - 1]) * dt
    return total


@dataclass(frozen=True, slots=True)
class TimingPolicy:
    freshness_s: float = 0.5
    min_frequency_hz: float = 5.0
    max_frequency_hz: float | None = None
    max_jitter_cv: float = 0.35
    min_samples: int = 3
    future_tolerance_s: float = 0.10

    def __post_init__(self) -> None:
        if _finite_float(self.freshness_s, "freshness_s") <= 0.0:
            raise ValueError("freshness_s must be positive")
        if _finite_float(self.min_frequency_hz, "min_frequency_hz") < 0.0:
            raise ValueError("min_frequency_hz must be non-negative")
        if self.max_frequency_hz is not None:
            maximum = _finite_float(self.max_frequency_hz, "max_frequency_hz")
            if maximum <= self.min_frequency_hz:
                raise ValueError("max_frequency_hz must exceed min_frequency_hz")
        if _finite_float(self.max_jitter_cv, "max_jitter_cv") < 0.0:
            raise ValueError("max_jitter_cv must be non-negative")
        if isinstance(self.min_samples, bool) or self.min_samples < 2:
            raise ValueError("min_samples must be an integer >= 2")
        if _finite_float(self.future_tolerance_s, "future_tolerance_s") < 0.0:
            raise ValueError("future_tolerance_s must be non-negative")


@dataclass(frozen=True, slots=True)
class TimingHealth:
    sensor: str
    state: DiagnosticState
    sample_count: int
    last_timestamp_s: float | None
    age_s: float | None
    frequency_hz: float | None
    median_period_s: float | None
    p95_period_s: float | None
    jitter_cv: float | None
    monotonic: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return contract_payload(
            "metric_nav.timing_health",
            sensor=self.sensor,
            state=self.state.value,
            sample_count=self.sample_count,
            last_timestamp_s=self.last_timestamp_s,
            age_s=self.age_s,
            frequency_hz=self.frequency_hz,
            median_period_s=self.median_period_s,
            p95_period_s=self.p95_period_s,
            jitter_cv=self.jitter_cv,
            monotonic=self.monotonic,
            reasons=list(self.reasons),
        )


def assess_timing_health(
    sensor: str,
    timestamps_s: Sequence[float],
    *,
    now_s: float,
    policy: TimingPolicy | None = None,
) -> TimingHealth:
    """Assess monotonicity, freshness, receive frequency, and interval jitter."""

    selected = policy or TimingPolicy()
    now = _finite_float(now_s, "now_s")
    name = str(sensor).strip()
    if not name:
        raise ValueError("sensor must be non-empty")
    try:
        timestamps = _finite_series(timestamps_s, "timestamps_s")
    except (TypeError, ValueError):
        return TimingHealth(
            name,
            DiagnosticState.UNHEALTHY,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            False,
            ("non_finite_or_invalid_timestamp",),
        )

    if not timestamps:
        return TimingHealth(
            name,
            DiagnosticState.INSUFFICIENT_DATA,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            False,
            ("missing_timestamps",),
        )

    reasons: list[str] = []
    intervals = [
        timestamps[index] - timestamps[index - 1]
        for index in range(1, len(timestamps))
    ]
    monotonic = all(interval > 0.0 for interval in intervals)
    if not monotonic:
        reasons.append("non_monotonic_timestamps")

    last_timestamp = timestamps[-1]
    age = now - last_timestamp
    if age < -selected.future_tolerance_s:
        reasons.append("timestamp_in_future")
    age = max(0.0, age)
    if age > selected.freshness_s:
        reasons.append("stale")

    positive_intervals = [interval for interval in intervals if interval > 0.0]
    median_period: float | None = None
    p95_period: float | None = None
    frequency: float | None = None
    jitter_cv: float | None = None
    if positive_intervals:
        median_period = statistics.median(positive_intervals)
        p95_period = _quantile(positive_intervals, 0.95)
        frequency = 1.0 / median_period
        mean_period = statistics.fmean(positive_intervals)
        jitter_cv = (
            statistics.pstdev(positive_intervals) / mean_period
            if len(positive_intervals) > 1 and mean_period > 0.0
            else 0.0
        )
        if frequency < selected.min_frequency_hz:
            reasons.append("frequency_below_minimum")
        if selected.max_frequency_hz is not None and frequency > selected.max_frequency_hz:
            reasons.append("frequency_above_maximum")
        if jitter_cv > selected.max_jitter_cv:
            reasons.append("jitter_above_maximum")

    if len(timestamps) < selected.min_samples:
        state = DiagnosticState.INSUFFICIENT_DATA
        reasons.append("insufficient_samples")
    elif not monotonic or "timestamp_in_future" in reasons or "stale" in reasons:
        state = DiagnosticState.UNHEALTHY
    elif reasons:
        state = DiagnosticState.DEGRADED
    else:
        state = DiagnosticState.HEALTHY

    return TimingHealth(
        sensor=name,
        state=state,
        sample_count=len(timestamps),
        last_timestamp_s=last_timestamp,
        age_s=age,
        frequency_hz=frequency,
        median_period_s=median_period,
        p95_period_s=p95_period,
        jitter_cv=jitter_cv,
        monotonic=monotonic,
        reasons=tuple(dict.fromkeys(reasons)),
    )


@dataclass(frozen=True, slots=True)
class CovariancePolicy:
    dimension: int | None = None
    symmetry_tolerance: float = 1e-8
    psd_tolerance: float = 1e-10
    max_variance: float = 1e3
    max_condition_number: float = 1e8

    def __post_init__(self) -> None:
        if self.dimension is not None and (
            isinstance(self.dimension, bool) or self.dimension <= 0
        ):
            raise ValueError("dimension must be a positive integer or None")
        if _finite_float(self.symmetry_tolerance, "symmetry_tolerance") < 0.0:
            raise ValueError("symmetry_tolerance must be non-negative")
        if _finite_float(self.psd_tolerance, "psd_tolerance") < 0.0:
            raise ValueError("psd_tolerance must be non-negative")
        if _finite_float(self.max_variance, "max_variance") <= 0.0:
            raise ValueError("max_variance must be positive")
        if _finite_float(self.max_condition_number, "max_condition_number") <= 1.0:
            raise ValueError("max_condition_number must exceed 1")


@dataclass(frozen=True, slots=True)
class CovarianceHealth:
    name: str
    state: DiagnosticState
    dimension: int
    symmetric: bool
    positive_semidefinite: bool
    diagonal: tuple[float, ...]
    eigenvalues: tuple[float, ...]
    condition_number: float | None
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return contract_payload(
            "metric_nav.covariance_health",
            name=self.name,
            state=self.state.value,
            dimension=self.dimension,
            symmetric=self.symmetric,
            positive_semidefinite=self.positive_semidefinite,
            diagonal=list(self.diagonal),
            eigenvalues=list(self.eigenvalues),
            condition_number=self.condition_number,
            reasons=list(self.reasons),
        )


def _matrix(values: Sequence[Sequence[float]], name: str) -> list[list[float]]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a square numeric matrix")
    rows = list(values)
    if not rows:
        raise ValueError(f"{name} must not be empty")
    matrix: list[list[float]] = []
    size = len(rows)
    for row_index, row in enumerate(rows):
        if isinstance(row, (str, bytes)):
            raise TypeError(f"{name}[{row_index}] must be a numeric sequence")
        converted = _finite_series(list(row), f"{name}[{row_index}]")
        if len(converted) != size:
            raise ValueError(f"{name} must be square")
        matrix.append(converted)
    return matrix


def _jacobi_eigenvalues(matrix: Sequence[Sequence[float]]) -> list[float]:
    """Eigenvalues of a small real symmetric matrix using Jacobi rotations."""

    size = len(matrix)
    work = [list(row) for row in matrix]
    if size == 1:
        return [work[0][0]]
    tolerance = 1e-13
    for _ in range(max(32, size * size * 80)):
        row, column = max(
            (
                (i, j)
                for i in range(size)
                for j in range(i + 1, size)
            ),
            key=lambda pair: abs(work[pair[0]][pair[1]]),
        )
        off_diagonal = work[row][column]
        if abs(off_diagonal) <= tolerance:
            break
        angle = 0.5 * math.atan2(
            2.0 * off_diagonal,
            work[column][column] - work[row][row],
        )
        cosine = math.cos(angle)
        sine = math.sin(angle)
        app = work[row][row]
        aqq = work[column][column]
        work[row][row] = (
            cosine * cosine * app
            - 2.0 * sine * cosine * off_diagonal
            + sine * sine * aqq
        )
        work[column][column] = (
            sine * sine * app
            + 2.0 * sine * cosine * off_diagonal
            + cosine * cosine * aqq
        )
        work[row][column] = 0.0
        work[column][row] = 0.0
        for index in range(size):
            if index in (row, column):
                continue
            aip = work[index][row]
            aiq = work[index][column]
            work[index][row] = work[row][index] = cosine * aip - sine * aiq
            work[index][column] = work[column][index] = sine * aip + cosine * aiq
    return sorted(work[index][index] for index in range(size))


def assess_covariance_health(
    name: str,
    covariance: Sequence[Sequence[float]],
    *,
    policy: CovariancePolicy | None = None,
) -> CovarianceHealth:
    """Validate covariance shape, symmetry, PSD, scale, and conditioning."""

    selected = policy or CovariancePolicy()
    label = str(name).strip()
    if not label:
        raise ValueError("name must be non-empty")
    try:
        matrix = _matrix(covariance, "covariance")
    except (TypeError, ValueError) as exc:
        return CovarianceHealth(
            label,
            DiagnosticState.UNHEALTHY,
            0,
            False,
            False,
            (),
            (),
            None,
            (f"invalid_covariance:{exc}",),
        )

    size = len(matrix)
    reasons: list[str] = []
    if selected.dimension is not None and size != selected.dimension:
        reasons.append("dimension_mismatch")
    max_asymmetry = max(
        abs(matrix[row][column] - matrix[column][row])
        for row in range(size)
        for column in range(size)
    )
    symmetric = max_asymmetry <= selected.symmetry_tolerance
    if not symmetric:
        reasons.append("not_symmetric")
    symmetrized = [
        [0.5 * (matrix[row][column] + matrix[column][row]) for column in range(size)]
        for row in range(size)
    ]
    eigenvalues = _jacobi_eigenvalues(symmetrized)
    positive_semidefinite = min(eigenvalues) >= -selected.psd_tolerance
    if not positive_semidefinite:
        reasons.append("not_positive_semidefinite")
    diagonal = tuple(matrix[index][index] for index in range(size))
    if any(value < -selected.psd_tolerance for value in diagonal):
        reasons.append("negative_variance")
    if any(value > selected.max_variance for value in diagonal):
        reasons.append("variance_above_maximum")

    positive = [value for value in eigenvalues if value > selected.psd_tolerance]
    condition_number: float | None
    if not positive:
        condition_number = None
        reasons.append("zero_information")
    elif len(positive) < size:
        condition_number = selected.max_condition_number + 1.0
        reasons.append("rank_deficient")
    else:
        condition_number = max(positive) / min(positive)
        if condition_number > selected.max_condition_number:
            reasons.append("condition_number_above_maximum")

    fatal = {
        "dimension_mismatch",
        "not_symmetric",
        "not_positive_semidefinite",
        "negative_variance",
        "zero_information",
    }
    if any(reason in fatal for reason in reasons):
        state = DiagnosticState.UNHEALTHY
    elif reasons:
        state = DiagnosticState.DEGRADED
    else:
        state = DiagnosticState.HEALTHY
    return CovarianceHealth(
        name=label,
        state=state,
        dimension=size,
        symmetric=symmetric,
        positive_semidefinite=positive_semidefinite,
        diagonal=diagonal,
        eigenvalues=tuple(eigenvalues),
        condition_number=condition_number,
        reasons=tuple(dict.fromkeys(reasons)),
    )


@dataclass(frozen=True, slots=True)
class MahalanobisPolicy:
    threshold_squared: float = 6.634896601
    regularization: float = 1e-9

    def __post_init__(self) -> None:
        if _finite_float(self.threshold_squared, "threshold_squared") <= 0.0:
            raise ValueError("threshold_squared must be positive")
        if _finite_float(self.regularization, "regularization") < 0.0:
            raise ValueError("regularization must be non-negative")


@dataclass(frozen=True, slots=True)
class MahalanobisGate:
    accepted: bool
    state: DiagnosticState
    distance_squared: float | None
    threshold_squared: float
    degrees_of_freedom: int
    innovation: tuple[float, ...]
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return contract_payload(
            "metric_nav.mahalanobis_gate",
            accepted=self.accepted,
            state=self.state.value,
            distance_squared=self.distance_squared,
            threshold_squared=self.threshold_squared,
            degrees_of_freedom=self.degrees_of_freedom,
            innovation=list(self.innovation),
            reasons=list(self.reasons),
        )


def _solve_linear_system(
    matrix: Sequence[Sequence[float]],
    vector: Sequence[float],
    regularization: float,
) -> list[float]:
    size = len(vector)
    augmented = [
        [
            float(matrix[row][column])
            + (regularization if row == column else 0.0)
            for column in range(size)
        ]
        + [float(vector[row])]
        for row in range(size)
    ]
    for pivot_index in range(size):
        pivot_row = max(
            range(pivot_index, size),
            key=lambda row: abs(augmented[row][pivot_index]),
        )
        if abs(augmented[pivot_row][pivot_index]) <= 1e-15:
            raise ValueError("innovation covariance is singular")
        augmented[pivot_index], augmented[pivot_row] = (
            augmented[pivot_row],
            augmented[pivot_index],
        )
        pivot = augmented[pivot_index][pivot_index]
        augmented[pivot_index] = [value / pivot for value in augmented[pivot_index]]
        for row in range(size):
            if row == pivot_index:
                continue
            factor = augmented[row][pivot_index]
            augmented[row] = [
                augmented[row][column] - factor * augmented[pivot_index][column]
                for column in range(size + 1)
            ]
    return [augmented[row][-1] for row in range(size)]


def mahalanobis_gate(
    innovation: Sequence[float],
    innovation_covariance: Sequence[Sequence[float]],
    *,
    policy: MahalanobisPolicy | None = None,
) -> MahalanobisGate:
    """Gate a vector innovation using its supplied covariance."""

    selected = policy or MahalanobisPolicy()
    try:
        vector = _finite_series(innovation, "innovation")
        matrix = _matrix(innovation_covariance, "innovation_covariance")
        if not vector:
            raise ValueError("innovation must not be empty")
        if len(matrix) != len(vector):
            raise ValueError("innovation and covariance dimensions differ")
        covariance_health = assess_covariance_health(
            "innovation_covariance",
            matrix,
            policy=CovariancePolicy(dimension=len(vector)),
        )
        if covariance_health.state is DiagnosticState.UNHEALTHY:
            raise ValueError(",".join(covariance_health.reasons))
        solution = _solve_linear_system(matrix, vector, selected.regularization)
        distance_squared = sum(
            value * solved for value, solved in zip(vector, solution, strict=True)
        )
        distance_squared = max(0.0, distance_squared)
    except (TypeError, ValueError) as exc:
        try:
            degrees_of_freedom = len(innovation)
        except TypeError:
            degrees_of_freedom = 0
        return MahalanobisGate(
            accepted=False,
            state=DiagnosticState.UNHEALTHY,
            distance_squared=None,
            threshold_squared=selected.threshold_squared,
            degrees_of_freedom=degrees_of_freedom,
            innovation=(),
            reasons=(f"invalid_gate_input:{exc}",),
        )

    accepted = distance_squared <= selected.threshold_squared
    return MahalanobisGate(
        accepted=accepted,
        state=DiagnosticState.HEALTHY if accepted else DiagnosticState.DEGRADED,
        distance_squared=distance_squared,
        threshold_squared=selected.threshold_squared,
        degrees_of_freedom=len(vector),
        innovation=tuple(vector),
        reasons=() if accepted else ("innovation_outlier",),
    )


@dataclass(frozen=True, slots=True)
class YawRatePolicy:
    gate: MahalanobisPolicy = field(default_factory=MahalanobisPolicy)
    min_samples: int = 10
    min_acceptance_fraction: float = 0.90
    max_abs_bias_rad_s: float = 0.08
    max_rms_innovation_rad_s: float = 0.12

    def __post_init__(self) -> None:
        if not isinstance(self.gate, MahalanobisPolicy):
            raise TypeError("gate must be a MahalanobisPolicy")
        if isinstance(self.min_samples, bool) or self.min_samples <= 0:
            raise ValueError("min_samples must be positive")
        if not 0.0 <= _finite_float(
            self.min_acceptance_fraction,
            "min_acceptance_fraction",
        ) <= 1.0:
            raise ValueError("min_acceptance_fraction must be in [0, 1]")
        if _finite_float(self.max_abs_bias_rad_s, "max_abs_bias_rad_s") < 0.0:
            raise ValueError("max_abs_bias_rad_s must be non-negative")
        if _finite_float(
            self.max_rms_innovation_rad_s,
            "max_rms_innovation_rad_s",
        ) < 0.0:
            raise ValueError("max_rms_innovation_rad_s must be non-negative")


@dataclass(frozen=True, slots=True)
class YawRateConsistency:
    state: DiagnosticState
    sample_count: int
    accepted_count: int
    acceptance_fraction: float
    mean_innovation_rad_s: float | None
    rms_innovation_rad_s: float | None
    p95_abs_innovation_rad_s: float | None
    mean_nis: float | None
    gates: tuple[MahalanobisGate, ...]
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return contract_payload(
            "metric_nav.yaw_rate_consistency",
            state=self.state.value,
            sample_count=self.sample_count,
            accepted_count=self.accepted_count,
            acceptance_fraction=self.acceptance_fraction,
            mean_innovation_rad_s=self.mean_innovation_rad_s,
            rms_innovation_rad_s=self.rms_innovation_rad_s,
            p95_abs_innovation_rad_s=self.p95_abs_innovation_rad_s,
            mean_nis=self.mean_nis,
            gates=[gate.to_dict() for gate in self.gates],
            reasons=list(self.reasons),
        )


def assess_yaw_rate_consistency(
    wheel_yaw_rates_rad_s: Sequence[float],
    imu_yaw_rates_rad_s: Sequence[float],
    wheel_variances: Sequence[float],
    imu_variances: Sequence[float],
    *,
    policy: YawRatePolicy | None = None,
) -> YawRateConsistency:
    """Compare wheel and IMU yaw rates using per-sample normalized innovation."""

    selected = policy or YawRatePolicy()
    try:
        wheel = _finite_series(wheel_yaw_rates_rad_s, "wheel_yaw_rates_rad_s")
        imu = _finite_series(imu_yaw_rates_rad_s, "imu_yaw_rates_rad_s")
        wheel_var = _finite_series(wheel_variances, "wheel_variances")
        imu_var = _finite_series(imu_variances, "imu_variances")
    except (TypeError, ValueError) as exc:
        return YawRateConsistency(
            DiagnosticState.UNHEALTHY,
            0,
            0,
            0.0,
            None,
            None,
            None,
            None,
            (),
            (f"invalid_series:{exc}",),
        )
    lengths = {len(wheel), len(imu), len(wheel_var), len(imu_var)}
    if len(lengths) != 1:
        return YawRateConsistency(
            DiagnosticState.UNHEALTHY,
            0,
            0,
            0.0,
            None,
            None,
            None,
            None,
            (),
            ("series_length_mismatch",),
        )
    sample_count = len(wheel)
    if sample_count == 0:
        return YawRateConsistency(
            DiagnosticState.INSUFFICIENT_DATA,
            0,
            0,
            0.0,
            None,
            None,
            None,
            None,
            (),
            ("missing_samples",),
        )

    gates: list[MahalanobisGate] = []
    innovations: list[float] = []
    for wheel_value, imu_value, wheel_variance, imu_variance in zip(
        wheel,
        imu,
        wheel_var,
        imu_var,
        strict=True,
    ):
        innovations.append(wheel_value - imu_value)
        combined_variance = wheel_variance + imu_variance
        gates.append(
            mahalanobis_gate(
                [innovations[-1]],
                [[combined_variance]],
                policy=selected.gate,
            )
        )
    accepted_count = sum(gate.accepted for gate in gates)
    acceptance_fraction = accepted_count / sample_count
    mean_innovation = statistics.fmean(innovations)
    rms_innovation = math.sqrt(statistics.fmean(value * value for value in innovations))
    p95_abs = _quantile([abs(value) for value in innovations], 0.95)
    valid_nis = [
        gate.distance_squared
        for gate in gates
        if gate.distance_squared is not None
    ]
    mean_nis = statistics.fmean(valid_nis) if valid_nis else None
    reasons: list[str] = []
    if sample_count < selected.min_samples:
        reasons.append("insufficient_samples")
        state = DiagnosticState.INSUFFICIENT_DATA
    else:
        if acceptance_fraction < selected.min_acceptance_fraction:
            reasons.append("acceptance_fraction_below_minimum")
        if abs(mean_innovation) > selected.max_abs_bias_rad_s:
            reasons.append("yaw_rate_bias_above_maximum")
        if rms_innovation > selected.max_rms_innovation_rad_s:
            reasons.append("yaw_rate_rms_above_maximum")
        invalid_gate_count = sum(
            gate.state is DiagnosticState.UNHEALTHY for gate in gates
        )
        if invalid_gate_count:
            reasons.append("invalid_innovation_covariance")
            state = DiagnosticState.UNHEALTHY
        elif reasons:
            state = DiagnosticState.DEGRADED
        else:
            state = DiagnosticState.HEALTHY
    return YawRateConsistency(
        state=state,
        sample_count=sample_count,
        accepted_count=accepted_count,
        acceptance_fraction=acceptance_fraction,
        mean_innovation_rad_s=mean_innovation,
        rms_innovation_rad_s=rms_innovation,
        p95_abs_innovation_rad_s=p95_abs,
        mean_nis=mean_nis,
        gates=tuple(gates),
        reasons=tuple(reasons),
    )


@dataclass(frozen=True, slots=True)
class StaticDriftPolicy:
    min_duration_s: float = 3.0
    max_p95_linear_speed_m_s: float = 0.02
    max_p95_yaw_rate_rad_s: float = 0.03
    max_linear_drift_m: float = 0.03
    max_yaw_drift_rad: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "min_duration_s",
            "max_p95_linear_speed_m_s",
            "max_p95_yaw_rate_rad_s",
            "max_linear_drift_m",
            "max_yaw_drift_rad",
        ):
            if _finite_float(getattr(self, name), name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.min_duration_s <= 0.0:
            raise ValueError("min_duration_s must be positive")


@dataclass(frozen=True, slots=True)
class StaticDrift:
    state: DiagnosticState
    sample_count: int
    duration_s: float
    p95_abs_linear_speed_m_s: float | None
    p95_abs_wheel_yaw_rate_rad_s: float | None
    p95_abs_imu_yaw_rate_rad_s: float | None
    linear_drift_m: float | None
    wheel_yaw_drift_rad: float | None
    imu_yaw_drift_rad: float | None
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return contract_payload(
            "metric_nav.static_drift",
            state=self.state.value,
            sample_count=self.sample_count,
            duration_s=self.duration_s,
            p95_abs_linear_speed_m_s=self.p95_abs_linear_speed_m_s,
            p95_abs_wheel_yaw_rate_rad_s=self.p95_abs_wheel_yaw_rate_rad_s,
            p95_abs_imu_yaw_rate_rad_s=self.p95_abs_imu_yaw_rate_rad_s,
            linear_drift_m=self.linear_drift_m,
            wheel_yaw_drift_rad=self.wheel_yaw_drift_rad,
            imu_yaw_drift_rad=self.imu_yaw_drift_rad,
            reasons=list(self.reasons),
        )


def assess_static_drift(
    timestamps_s: Sequence[float],
    wheel_linear_speeds_m_s: Sequence[float],
    wheel_yaw_rates_rad_s: Sequence[float],
    imu_yaw_rates_rad_s: Sequence[float],
    *,
    policy: StaticDriftPolicy | None = None,
) -> StaticDrift:
    """Measure apparent motion while the robot is known to be stationary."""

    selected = policy or StaticDriftPolicy()
    try:
        timestamps = _finite_series(timestamps_s, "timestamps_s")
        linear = _finite_series(wheel_linear_speeds_m_s, "wheel_linear_speeds_m_s")
        wheel_yaw = _finite_series(wheel_yaw_rates_rad_s, "wheel_yaw_rates_rad_s")
        imu_yaw = _finite_series(imu_yaw_rates_rad_s, "imu_yaw_rates_rad_s")
    except (TypeError, ValueError) as exc:
        return StaticDrift(
            DiagnosticState.UNHEALTHY,
            0,
            0.0,
            None,
            None,
            None,
            None,
            None,
            None,
            (f"invalid_series:{exc}",),
        )
    if len({len(timestamps), len(linear), len(wheel_yaw), len(imu_yaw)}) != 1:
        return StaticDrift(
            DiagnosticState.UNHEALTHY,
            0,
            0.0,
            None,
            None,
            None,
            None,
            None,
            None,
            ("series_length_mismatch",),
        )
    if len(timestamps) < 2:
        return StaticDrift(
            DiagnosticState.INSUFFICIENT_DATA,
            len(timestamps),
            0.0,
            None,
            None,
            None,
            None,
            None,
            None,
            ("insufficient_samples",),
        )
    intervals = [
        timestamps[index] - timestamps[index - 1]
        for index in range(1, len(timestamps))
    ]
    if any(interval <= 0.0 for interval in intervals):
        return StaticDrift(
            DiagnosticState.UNHEALTHY,
            len(timestamps),
            0.0,
            None,
            None,
            None,
            None,
            None,
            None,
            ("non_monotonic_timestamps",),
        )

    duration = timestamps[-1] - timestamps[0]
    linear_p95 = _quantile([abs(value) for value in linear], 0.95)
    wheel_yaw_p95 = _quantile([abs(value) for value in wheel_yaw], 0.95)
    imu_yaw_p95 = _quantile([abs(value) for value in imu_yaw], 0.95)
    linear_drift = abs(_trapezoid_integral(timestamps, linear))
    wheel_yaw_drift = abs(_trapezoid_integral(timestamps, wheel_yaw))
    imu_yaw_drift = abs(_trapezoid_integral(timestamps, imu_yaw))
    reasons: list[str] = []
    if duration < selected.min_duration_s:
        reasons.append("duration_below_minimum")
        state = DiagnosticState.INSUFFICIENT_DATA
    else:
        if linear_p95 > selected.max_p95_linear_speed_m_s:
            reasons.append("linear_speed_drift")
        if max(wheel_yaw_p95, imu_yaw_p95) > selected.max_p95_yaw_rate_rad_s:
            reasons.append("yaw_rate_drift")
        if linear_drift > selected.max_linear_drift_m:
            reasons.append("integrated_linear_drift")
        if max(wheel_yaw_drift, imu_yaw_drift) > selected.max_yaw_drift_rad:
            reasons.append("integrated_yaw_drift")
        severe = (
            linear_p95 > 2.0 * selected.max_p95_linear_speed_m_s
            or max(wheel_yaw_p95, imu_yaw_p95)
            > 2.0 * selected.max_p95_yaw_rate_rad_s
            or linear_drift > 2.0 * selected.max_linear_drift_m
            or max(wheel_yaw_drift, imu_yaw_drift)
            > 2.0 * selected.max_yaw_drift_rad
        )
        state = (
            DiagnosticState.UNHEALTHY
            if severe
            else DiagnosticState.DEGRADED
            if reasons
            else DiagnosticState.HEALTHY
        )
    return StaticDrift(
        state=state,
        sample_count=len(timestamps),
        duration_s=duration,
        p95_abs_linear_speed_m_s=linear_p95,
        p95_abs_wheel_yaw_rate_rad_s=wheel_yaw_p95,
        p95_abs_imu_yaw_rate_rad_s=imu_yaw_p95,
        linear_drift_m=linear_drift,
        wheel_yaw_drift_rad=wheel_yaw_drift,
        imu_yaw_drift_rad=imu_yaw_drift,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True, slots=True)
class MetricNavHealthReport:
    timing: Mapping[str, TimingHealth]
    covariance: Mapping[str, CovarianceHealth]
    yaw_rate: YawRateConsistency | None = None
    static_drift: StaticDrift | None = None

    @property
    def state(self) -> DiagnosticState:
        states = [result.state for result in self.timing.values()]
        states.extend(result.state for result in self.covariance.values())
        if self.yaw_rate is not None:
            states.append(self.yaw_rate.state)
        if self.static_drift is not None:
            states.append(self.static_drift.state)
        return worst_state(states)

    def to_dict(self) -> dict[str, Any]:
        return contract_payload(
            "metric_nav.health_report",
            state=self.state.value,
            timing={name: result.to_dict() for name, result in self.timing.items()},
            covariance={
                name: result.to_dict() for name, result in self.covariance.items()
            },
            yaw_rate=self.yaw_rate.to_dict() if self.yaw_rate is not None else None,
            static_drift=(
                self.static_drift.to_dict() if self.static_drift is not None else None
            ),
        )


def assert_json_serializable(result: Any) -> None:
    """Compatibility helper used by offline collectors and tests."""

    payload = result.to_dict() if hasattr(result, "to_dict") else result
    json_ready(payload)
