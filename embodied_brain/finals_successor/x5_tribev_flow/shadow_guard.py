"""Pure NumPy confidence monitoring for the X5-TriBEV-Flow shadow stack.

This module is deliberately control-plane free. It evaluates observations and
model outputs, but it cannot publish velocity commands, write to the F407, or
alter the frozen finals demonstration.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np


SHADOW_ONLY = True
CMD_VEL_AUTHORITY = False
DEFAULT_TRAJECTORY_OMEGA_RAD_S = (
    -0.80,
    -0.55,
    -0.30,
    -0.12,
    0.0,
    0.12,
    0.30,
    0.55,
    0.80,
)


def _contract(**values: Any) -> dict[str, Any]:
    return {
        **values,
        "cmd_vel_authority": CMD_VEL_AUTHORITY,
        "shadow_only": SHADOW_ONLY,
    }


class SensorHealthState(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    STALE = "STALE"
    INVALID = "INVALID"
    MISSING = "MISSING"


class ShadowState(str, Enum):
    TRUSTED_SHADOW = "TRUSTED_SHADOW"
    REVIEW = "REVIEW"
    UNTRUSTED_SHADOW = "UNTRUSTED_SHADOW"
    MONITOR_OFFLINE = "MONITOR_OFFLINE"


@dataclass(frozen=True)
class SensorPolicy:
    freshness_s: float
    min_frequency_hz: float
    min_valid_fraction: float
    required: bool
    payload_kind: str = "generic"

    def __post_init__(self) -> None:
        if not math.isfinite(self.freshness_s) or self.freshness_s <= 0.0:
            raise ValueError("freshness_s must be finite and positive")
        if not math.isfinite(self.min_frequency_hz) or self.min_frequency_hz < 0.0:
            raise ValueError("min_frequency_hz must be finite and non-negative")
        if not 0.0 <= self.min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must be within [0, 1]")


DEFAULT_SENSOR_POLICIES: dict[str, SensorPolicy] = {
    "lidar": SensorPolicy(1.0, 5.0, 0.10, True, "range"),
    "depth": SensorPolicy(1.5, 2.0, 0.05, False, "range"),
    "vision": SensorPolicy(4.0, 0.0, 0.01, False, "vision"),
    "odom": SensorPolicy(1.0, 5.0, 1.00, True, "odom"),
}


@dataclass(frozen=True)
class SensorHealth:
    name: str
    state: SensorHealthState
    healthy: bool
    required: bool
    available: bool
    fresh: bool
    frequency_ok: bool | None
    valid: bool
    age_s: float | None
    frequency_hz: float | None
    valid_fraction: float
    sample_count: int
    finite_count: int
    last_timestamp_s: float | None
    reasons: tuple[str, ...] = field(default_factory=tuple)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    cmd_vel_authority: bool = CMD_VEL_AUTHORITY
    shadow_only: bool = SHADOW_ONLY

    def to_dict(self) -> dict[str, Any]:
        return _contract(
            name=self.name,
            state=self.state.value,
            healthy=self.healthy,
            required=self.required,
            available=self.available,
            fresh=self.fresh,
            frequency_ok=self.frequency_ok,
            valid=self.valid,
            age_s=self.age_s,
            frequency_hz=self.frequency_hz,
            valid_fraction=self.valid_fraction,
            sample_count=self.sample_count,
            finite_count=self.finite_count,
            last_timestamp_s=self.last_timestamp_s,
            reasons=list(self.reasons),
            provenance=dict(self.provenance),
        )


def _timestamps_array(timestamps_s: Any) -> np.ndarray:
    if timestamps_s is None:
        return np.empty(0, dtype=np.float64)
    try:
        arr = np.asarray(timestamps_s, dtype=np.float64)
    except (TypeError, ValueError):
        return np.empty(0, dtype=np.float64)
    return arr.reshape(-1)


def _numeric_payload(payload: Any) -> np.ndarray:
    if payload is None:
        return np.empty(0, dtype=np.float64)
    if isinstance(payload, Mapping):
        numeric: list[float] = []
        for value in payload.values():
            if isinstance(value, (int, float, np.integer, np.floating)):
                numeric.append(float(value))
            elif isinstance(value, (list, tuple, np.ndarray)):
                try:
                    numeric.extend(np.asarray(value, dtype=np.float64).reshape(-1).tolist())
                except (TypeError, ValueError):
                    continue
        return np.asarray(numeric, dtype=np.float64)
    try:
        return np.asarray(payload, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return np.empty(0, dtype=np.float64)


def _payload_validity(
    payload: Any,
    payload_kind: str,
    valid_mask: Any = None,
) -> tuple[float, int, int, list[str]]:
    arr = _numeric_payload(payload)
    if arr.size == 0:
        return 0.0, 0, 0, ["empty_payload"]

    finite = np.isfinite(arr)
    reasons: list[str] = []
    if valid_mask is not None:
        try:
            mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
        except (TypeError, ValueError):
            return 0.0, int(arr.size), int(np.count_nonzero(finite)), ["invalid_valid_mask"]
        if mask.size != arr.size:
            return 0.0, int(arr.size), int(np.count_nonzero(finite)), ["valid_mask_shape_mismatch"]
        finite &= mask

    if payload_kind == "range":
        usable = finite & (arr > 0.0)
    elif payload_kind in {"vision", "bev"}:
        usable = finite & (arr >= 0.0)
    elif payload_kind == "odom":
        usable = finite
        if arr.size < 2:
            reasons.append("odom_payload_too_small")
    else:
        usable = finite

    finite_count = int(np.count_nonzero(finite))
    valid_count = int(np.count_nonzero(usable))
    fraction = float(valid_count / arr.size)
    if finite_count < arr.size:
        reasons.append("non_finite_payload_values")
    return fraction, int(arr.size), finite_count, reasons


def assess_sensor_health(
    name: str,
    timestamps_s: Any,
    payload: Any,
    *,
    now_s: float | None = None,
    policy: SensorPolicy | None = None,
    valid_mask: Any = None,
    explicitly_valid: bool | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> SensorHealth:
    """Evaluate freshness, receive rate, and payload validity for one sensor."""
    sensor_name = str(name).strip().lower()
    selected = policy or DEFAULT_SENSOR_POLICIES.get(
        sensor_name,
        SensorPolicy(1.5, 0.0, 0.50, False, "generic"),
    )
    now = float(time.time() if now_s is None else now_s)
    reasons: list[str] = []
    ts_raw = _timestamps_array(timestamps_s)
    ts = ts_raw[np.isfinite(ts_raw)]

    available = bool(ts.size)
    age_s: float | None = None
    last_ts: float | None = None
    fresh = False
    frequency_hz: float | None = None
    frequency_ok: bool | None = None

    if not math.isfinite(now):
        reasons.append("invalid_monitor_clock")
    elif not available:
        reasons.append("missing_timestamp")
    else:
        last_ts = float(ts[-1])
        age_s = float(now - last_ts)
        future_tolerance = max(0.25, selected.freshness_s * 0.5)
        if age_s < -future_tolerance:
            reasons.append("timestamp_in_future")
        else:
            age_s = max(0.0, age_s)
            fresh = age_s <= selected.freshness_s
            if not fresh:
                reasons.append("stale")

        if ts.size >= 2:
            deltas = np.diff(ts)
            positive = deltas[deltas > 0.0]
            if positive.size != deltas.size:
                reasons.append("non_monotonic_timestamps")
            if positive.size:
                frequency_hz = float(1.0 / np.median(positive))
                frequency_ok = frequency_hz >= selected.min_frequency_hz
                if not frequency_ok:
                    reasons.append("frequency_below_minimum")
            else:
                frequency_ok = False
                reasons.append("frequency_unavailable")
        elif selected.min_frequency_hz > 0.0:
            reasons.append("frequency_insufficient_samples")
        else:
            frequency_ok = True

    valid_fraction, _, finite_count, validity_reasons = _payload_validity(
        payload,
        selected.payload_kind,
        valid_mask,
    )
    reasons.extend(validity_reasons)
    valid = valid_fraction >= selected.min_valid_fraction

    if explicitly_valid is False:
        valid = False
        reasons.append("explicit_invalid")
    elif explicitly_valid is True and valid_fraction == 0.0:
        reasons.append("explicit_valid_but_payload_empty")

    prov = dict(provenance or {})
    if selected.payload_kind == "vision" and prov:
        state = str(prov.get("state") or "unknown")
        image_supplied = bool(prov.get("image_supplied"))
        source_ok = state in {"live_camera", "recorded_camera", "replay_camera"}
        if not source_ok or (state == "live_camera" and not image_supplied):
            valid = False
            reasons.append("vision_provenance_not_observed_frame")

    if not available:
        state = SensorHealthState.MISSING
    elif "invalid_monitor_clock" in reasons or "timestamp_in_future" in reasons or not valid:
        state = SensorHealthState.INVALID
    elif not fresh:
        state = SensorHealthState.STALE
    elif frequency_ok is False or (
        frequency_ok is None and selected.min_frequency_hz > 0.0
    ):
        state = SensorHealthState.DEGRADED
    else:
        state = SensorHealthState.HEALTHY

    reasons = list(dict.fromkeys(reasons))
    return SensorHealth(
        name=sensor_name,
        state=state,
        healthy=state is SensorHealthState.HEALTHY,
        required=selected.required,
        available=available,
        fresh=fresh,
        frequency_ok=frequency_ok,
        valid=valid,
        age_s=age_s,
        frequency_hz=frequency_hz,
        valid_fraction=valid_fraction,
        sample_count=int(ts.size),
        finite_count=finite_count,
        last_timestamp_s=last_ts,
        reasons=tuple(reasons),
        provenance=prov,
    )


class SensorHealthMonitor:
    """Stateless window evaluator with explicit per-sensor policies."""

    def __init__(self, policies: Mapping[str, SensorPolicy] | None = None) -> None:
        self.policies = dict(DEFAULT_SENSOR_POLICIES)
        if policies:
            self.policies.update({str(k).lower(): v for k, v in policies.items()})

    def assess(
        self,
        name: str,
        timestamps_s: Any,
        payload: Any,
        **kwargs: Any,
    ) -> SensorHealth:
        policy = kwargs.pop("policy", self.policies.get(str(name).lower()))
        return assess_sensor_health(
            name,
            timestamps_s,
            payload,
            policy=policy,
            **kwargs,
        )

    def assess_all(
        self,
        samples: Mapping[str, Mapping[str, Any]],
        *,
        now_s: float | None = None,
    ) -> dict[str, Any]:
        sensors: dict[str, Any] = {}
        for name, sample in samples.items():
            item = sample if isinstance(sample, Mapping) else {}
            health = self.assess(
                name,
                item.get("timestamps_s"),
                item.get("payload"),
                now_s=now_s,
                valid_mask=item.get("valid_mask"),
                explicitly_valid=item.get("explicitly_valid"),
                provenance=item.get("provenance"),
            )
            sensors[str(name).lower()] = health.to_dict()
        return _contract(sensors=sensors, sensor_count=len(sensors))


def _occupancy_probability(grid: Any) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    try:
        arr = np.asarray(grid, dtype=np.float64)
    except (TypeError, ValueError):
        return None, None, "non_numeric_bev"
    if arr.ndim != 2 or arr.size == 0:
        return None, None, "bev_must_be_nonempty_2d"

    known = np.isfinite(arr) & (arr >= 0.0)
    if not np.any(known):
        return None, known, "bev_has_no_known_cells"

    finite_known = arr[known]
    scale = 100.0 if float(np.max(finite_known)) > 1.0 else 1.0
    prob = np.zeros_like(arr, dtype=np.float64)
    prob[known] = np.clip(arr[known] / scale, 0.0, 1.0)
    return prob, known, None


def compare_bev(
    first: Any,
    second: Any,
    *,
    threshold: float = 0.5,
    valid_mask: Any = None,
) -> dict[str, Any]:
    """Return IoU and disagreement for two occupancy BEVs."""
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        return _contract(valid=False, reason="threshold_out_of_range")

    a, known_a, error_a = _occupancy_probability(first)
    b, known_b, error_b = _occupancy_probability(second)
    if error_a or error_b or a is None or b is None or known_a is None or known_b is None:
        return _contract(valid=False, reason=error_a or error_b or "invalid_bev")
    if a.shape != b.shape:
        return _contract(
            valid=False,
            reason="bev_shape_mismatch",
            first_shape=list(a.shape),
            second_shape=list(b.shape),
        )

    valid = known_a & known_b
    if valid_mask is not None:
        try:
            external = np.asarray(valid_mask, dtype=bool)
        except (TypeError, ValueError):
            return _contract(valid=False, reason="invalid_bev_valid_mask")
        if external.shape != a.shape:
            return _contract(valid=False, reason="bev_valid_mask_shape_mismatch")
        valid &= external

    compared_cells = int(np.count_nonzero(valid))
    if compared_cells == 0:
        return _contract(valid=False, reason="no_jointly_known_cells", compared_cells=0)

    occupied_a = (a >= threshold) & valid
    occupied_b = (b >= threshold) & valid
    intersection = int(np.count_nonzero(occupied_a & occupied_b))
    union = int(np.count_nonzero(occupied_a | occupied_b))
    xor = int(np.count_nonzero(occupied_a ^ occupied_b))
    empty_union = union == 0
    iou = 1.0 if empty_union else float(intersection / union)
    disagreement = float(xor / compared_cells)
    mean_abs = float(np.mean(np.abs(a[valid] - b[valid])))
    return _contract(
        valid=True,
        iou=iou,
        disagreement=disagreement,
        mean_absolute_disagreement=mean_abs,
        intersection_cells=intersection,
        union_cells=union,
        compared_cells=compared_cells,
        empty_union=empty_union,
        threshold=threshold,
    )


def cross_modal_bev_metrics(
    bevs: Mapping[str, Any],
    *,
    threshold: float = 0.5,
    valid_mask: Any = None,
) -> dict[str, Any]:
    pairs: dict[str, Any] = {}
    valid_results: list[dict[str, Any]] = []
    names = sorted(str(name) for name in bevs)
    for first_name, second_name in combinations(names, 2):
        result = compare_bev(
            bevs[first_name],
            bevs[second_name],
            threshold=threshold,
            valid_mask=valid_mask,
        )
        pairs[f"{first_name}__{second_name}"] = result
        if result.get("valid"):
            valid_results.append(result)

    if not valid_results:
        return _contract(
            valid=False,
            reason="no_valid_bev_pairs",
            pairs=pairs,
            pair_count=len(pairs),
            valid_pair_count=0,
        )
    ious = [float(item["iou"]) for item in valid_results]
    disagreements = [float(item["disagreement"]) for item in valid_results]
    return _contract(
        valid=True,
        pairs=pairs,
        pair_count=len(pairs),
        valid_pair_count=len(valid_results),
        mean_iou=float(np.mean(ious)),
        min_iou=float(np.min(ious)),
        mean_disagreement=float(np.mean(disagreements)),
        max_disagreement=float(np.max(disagreements)),
    )


def energy_ood(
    logits: Any,
    *,
    temperature: float = 1.0,
    threshold: float | None = None,
    higher_is_ood: bool = True,
) -> dict[str, Any]:
    """Compute E(x) = -T logsumexp(logits / T), with optional OOD decision."""
    if not math.isfinite(temperature) or temperature <= 0.0:
        return _contract(valid=False, reason="temperature_must_be_positive")
    try:
        values = np.asarray(logits, dtype=np.float64)
    except (TypeError, ValueError):
        return _contract(valid=False, reason="non_numeric_logits")
    if values.size == 0:
        return _contract(valid=False, reason="empty_logits")
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] == 0:
        return _contract(valid=False, reason="logits_must_be_1d_or_2d")
    if not np.all(np.isfinite(values)):
        return _contract(valid=False, reason="non_finite_logits")

    scaled = values / temperature
    maximum = np.max(scaled, axis=1, keepdims=True)
    logsumexp = maximum[:, 0] + np.log(np.sum(np.exp(scaled - maximum), axis=1))
    energies = -temperature * logsumexp
    aggregate = float(np.max(energies) if higher_is_ood else np.min(energies))

    is_ood: bool | None = None
    if threshold is not None:
        if not math.isfinite(threshold):
            return _contract(valid=False, reason="non_finite_ood_threshold")
        is_ood = aggregate > threshold if higher_is_ood else aggregate < threshold
    return _contract(
        valid=True,
        energy=aggregate,
        energies=energies.tolist(),
        threshold=threshold,
        is_ood=is_ood,
        higher_is_ood=bool(higher_is_ood),
        temperature=float(temperature),
    )


def _token_distribution(values: Any, inputs_are_logits: bool) -> np.ndarray | None:
    try:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    if inputs_are_logits:
        shifted = arr - np.max(arr)
        exp = np.exp(shifted)
        total = float(np.sum(exp))
        return exp / total if total > 0.0 else None
    if np.any(arr < 0.0):
        return None
    total = float(np.sum(arr))
    return arr / total if total > 0.0 else None


def trajectory_token_js_divergence(
    first: Any,
    second: Any,
    *,
    inputs_are_logits: bool = True,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Jensen-Shannon divergence for two trajectory-token distributions."""
    p = _token_distribution(first, inputs_are_logits)
    q = _token_distribution(second, inputs_are_logits)
    if p is None or q is None:
        return _contract(valid=False, reason="invalid_token_distribution")
    if p.shape != q.shape:
        return _contract(
            valid=False,
            reason="token_shape_mismatch",
            first_tokens=int(p.size),
            second_tokens=int(q.size),
        )
    m = 0.5 * (p + q)
    p_safe = np.clip(p, epsilon, 1.0)
    q_safe = np.clip(q, epsilon, 1.0)
    m_safe = np.clip(m, epsilon, 1.0)
    js_nats = 0.5 * (
        float(np.sum(p * np.log(p_safe / m_safe)))
        + float(np.sum(q * np.log(q_safe / m_safe)))
    )
    js_nats = max(0.0, js_nats)
    return _contract(
        valid=True,
        js_nats=js_nats,
        js_normalized=float(min(1.0, js_nats / math.log(2.0))),
        first_top_token=int(np.argmax(p)),
        second_top_token=int(np.argmax(q)),
        top_token_agreement=bool(np.argmax(p) == np.argmax(q)),
        token_count=int(p.size),
    )


def occupancy_conditioned_trajectory_tokens(
    future_occupancy: Any,
    *,
    inputs_are_logits: bool = False,
    omega_rad_s: Sequence[float] = DEFAULT_TRAJECTORY_OMEGA_RAD_S,
    speed_m_s: float = 0.65,
    footprint_length_m: float = 0.50,
    footprint_width_m: float = 0.40,
    safety_margin_m: float = 0.08,
    grid_resolution_m: float = 0.10,
    grid_x_min_m: float = -1.20,
    grid_y_min_m: float = -3.20,
    temperature: float = 0.35,
) -> dict[str, Any]:
    """Score fixed trajectory arcs against predicted future occupancy.

    This is CPU shadow post-processing. It never emits a velocity command and
    uses the measured robot footprint rather than a point-mass approximation.
    """

    try:
        occupancy = np.asarray(future_occupancy, dtype=np.float64)
    except (TypeError, ValueError):
        return _contract(valid=False, reason="non_numeric_future_occupancy")
    if occupancy.ndim == 4 and occupancy.shape[0] == 1:
        occupancy = occupancy[0]
    if occupancy.ndim != 3 or occupancy.shape[0] != 3:
        return _contract(
            valid=False,
            reason="future_occupancy_must_be_3xhxw",
        )
    if not np.all(np.isfinite(occupancy)):
        return _contract(valid=False, reason="future_occupancy_non_finite")
    if inputs_are_logits:
        occupancy = 1.0 / (1.0 + np.exp(-np.clip(occupancy, -40.0, 40.0)))
    elif np.any((occupancy < 0.0) | (occupancy > 1.0)):
        return _contract(
            valid=False,
            reason="future_occupancy_probability_out_of_range",
        )
    scalar_values = (
        speed_m_s,
        footprint_length_m,
        footprint_width_m,
        safety_margin_m,
        grid_resolution_m,
        grid_x_min_m,
        grid_y_min_m,
        temperature,
    )
    if not all(math.isfinite(float(value)) for value in scalar_values):
        return _contract(valid=False, reason="non_finite_geometry_parameter")
    if (
        speed_m_s < 0.0
        or footprint_length_m <= 0.0
        or footprint_width_m <= 0.0
        or safety_margin_m < 0.0
        or grid_resolution_m <= 0.0
        or temperature <= 0.0
    ):
        return _contract(valid=False, reason="invalid_geometry_parameter")
    omega = np.asarray(tuple(omega_rad_s), dtype=np.float64)
    if omega.shape != (9,) or not np.all(np.isfinite(omega)):
        return _contract(valid=False, reason="omega_contract_must_have_9_tokens")

    half_length = footprint_length_m / 2.0 + safety_margin_m
    half_width = footprint_width_m / 2.0 + safety_margin_m
    local_x = np.arange(
        -half_length,
        half_length + grid_resolution_m * 0.5,
        grid_resolution_m,
    )
    local_y = np.arange(
        -half_width,
        half_width + grid_resolution_m * 0.5,
        grid_resolution_m,
    )
    offset_x, offset_y = np.meshgrid(local_x, local_y, indexing="ij")
    offset_x = offset_x.reshape(-1)
    offset_y = offset_y.reshape(-1)
    height, width = occupancy.shape[-2:]
    evaluation_times = np.linspace(0.10, 1.20, 12)
    costs = np.empty(omega.size, dtype=np.float64)

    for token_index, angular_velocity in enumerate(omega):
        risks: list[float] = []
        for time_s in evaluation_times:
            if abs(float(angular_velocity)) < 1e-9:
                center_x = speed_m_s * float(time_s)
                center_y = 0.0
            else:
                radius = speed_m_s / float(angular_velocity)
                center_x = radius * math.sin(
                    float(angular_velocity) * float(time_s)
                )
                center_y = radius * (
                    1.0
                    - math.cos(float(angular_velocity) * float(time_s))
                )
            yaw = float(angular_velocity) * float(time_s)
            cosine = math.cos(yaw)
            sine = math.sin(yaw)
            world_x = center_x + cosine * offset_x - sine * offset_y
            world_y = center_y + sine * offset_x + cosine * offset_y
            rows = np.floor(
                (world_x - grid_x_min_m) / grid_resolution_m
            ).astype(np.int64)
            columns = np.floor(
                (world_y - grid_y_min_m) / grid_resolution_m
            ).astype(np.int64)
            inside = (
                (rows >= 0)
                & (rows < height)
                & (columns >= 0)
                & (columns < width)
            )
            horizon = min(2, int(math.ceil(float(time_s) / 0.4)) - 1)
            risk = 1.0
            if np.all(inside):
                risk = float(np.max(occupancy[horizon, rows, columns]))
            risks.append(risk)
        costs[token_index] = (
            5.0 * max(risks)
            + 1.5 * float(np.mean(risks))
            + 0.05 * abs(float(angular_velocity))
        )

    logits = -costs / temperature
    probabilities = _token_distribution(logits, inputs_are_logits=True)
    if probabilities is None:
        return _contract(valid=False, reason="trajectory_probability_failure")
    return _contract(
        valid=True,
        method="future_occupancy_rectangular_footprint_arc_sampling",
        probabilities=probabilities.tolist(),
        costs=costs.tolist(),
        top_token=int(np.argmax(probabilities)),
        omega_rad_s=omega.tolist(),
        speed_m_s=float(speed_m_s),
        footprint_m=[
            float(footprint_length_m),
            float(footprint_width_m),
        ],
        safety_margin_m=float(safety_margin_m),
        future_horizons_s=[0.4, 0.8, 1.2],
    )


def fuse_trajectory_token_evidence(
    model_values: Any,
    occupancy_values: Any,
    *,
    reference_values: Any | None = None,
    model_is_logits: bool = False,
    reference_is_logits: bool = False,
    model_weight: float = 0.20,
    occupancy_weight: float = 0.65,
    reference_weight: float = 0.15,
    epsilon: float = 1e-9,
) -> dict[str, Any]:
    """Fuse independent shadow token evidence with a weighted log opinion pool."""

    model = _token_distribution(model_values, model_is_logits)
    occupancy = _token_distribution(occupancy_values, False)
    reference = (
        _token_distribution(reference_values, reference_is_logits)
        if reference_values is not None
        else None
    )
    if model is None or occupancy is None:
        return _contract(valid=False, reason="invalid_required_token_evidence")
    if model.shape != occupancy.shape:
        return _contract(valid=False, reason="token_shape_mismatch")
    sources = [
        ("model_auxiliary", model, float(model_weight)),
        ("occupancy_footprint", occupancy, float(occupancy_weight)),
    ]
    if reference is not None:
        if reference.shape != model.shape:
            return _contract(valid=False, reason="reference_token_shape_mismatch")
        sources.append(("reference_shadow", reference, float(reference_weight)))
    if any(not math.isfinite(weight) or weight < 0.0 for _, _, weight in sources):
        return _contract(valid=False, reason="invalid_fusion_weight")
    total_weight = sum(weight for _, _, weight in sources)
    if total_weight <= 0.0:
        return _contract(valid=False, reason="zero_fusion_weight")
    log_probability = np.zeros_like(model)
    normalized_weights: dict[str, float] = {}
    for name, distribution, weight in sources:
        normalized = weight / total_weight
        normalized_weights[name] = normalized
        log_probability += normalized * np.log(
            np.clip(distribution, epsilon, 1.0)
        )
    fused = _token_distribution(log_probability, inputs_are_logits=True)
    if fused is None:
        return _contract(valid=False, reason="token_fusion_failure")
    return _contract(
        valid=True,
        method="weighted_log_opinion_pool",
        probabilities=fused.tolist(),
        top_token=int(np.argmax(fused)),
        weights=normalized_weights,
        sources=[name for name, _, _ in sources],
    )


def _path_array(path: Any) -> tuple[np.ndarray | None, int, str | None]:
    try:
        arr = np.asarray(path, dtype=np.float64)
    except (TypeError, ValueError):
        return None, 0, "non_numeric_trajectory"
    if arr.size == 0:
        return None, 0, "empty_trajectory"
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None, 0, "trajectory_must_be_n_by_2"
    arr = arr[:, :2]
    finite_rows = np.all(np.isfinite(arr), axis=1)
    dropped = int(arr.shape[0] - np.count_nonzero(finite_rows))
    arr = arr[finite_rows]
    if arr.shape[0] == 0:
        return None, dropped, "trajectory_has_no_finite_points"
    return arr, dropped, None


def _resample_path(path: np.ndarray, count: int) -> np.ndarray:
    if path.shape[0] == 1:
        return np.repeat(path, count, axis=0)
    segment = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment)))
    total = float(cumulative[-1])
    if total <= 1e-12:
        return np.repeat(path[:1], count, axis=0)
    position = cumulative / total
    target = np.linspace(0.0, 1.0, count)
    x = np.interp(target, position, path[:, 0])
    y = np.interp(target, position, path[:, 1])
    return np.column_stack((x, y))


def trajectory_distance_metrics(
    actual_path: Any,
    shadow_path: Any,
    *,
    sample_count: int = 64,
) -> dict[str, Any]:
    """Compute resampled ADE/FDE and symmetric Hausdorff distance."""
    if sample_count < 2:
        return _contract(valid=False, reason="sample_count_must_be_at_least_2")
    actual, actual_dropped, actual_error = _path_array(actual_path)
    shadow, shadow_dropped, shadow_error = _path_array(shadow_path)
    if actual_error or shadow_error or actual is None or shadow is None:
        return _contract(
            valid=False,
            reason=actual_error or shadow_error or "invalid_trajectory",
            actual_dropped_points=actual_dropped,
            shadow_dropped_points=shadow_dropped,
        )

    count = min(int(sample_count), 512)
    actual_r = _resample_path(actual, count)
    shadow_r = _resample_path(shadow, count)
    aligned = np.linalg.norm(actual_r - shadow_r, axis=1)
    pairwise = np.linalg.norm(actual_r[:, None, :] - shadow_r[None, :, :], axis=2)
    directed_actual = float(np.max(np.min(pairwise, axis=1)))
    directed_shadow = float(np.max(np.min(pairwise, axis=0)))
    return _contract(
        valid=True,
        ade_m=float(np.mean(aligned)),
        fde_m=float(aligned[-1]),
        hausdorff_m=max(directed_actual, directed_shadow),
        directed_actual_to_shadow_m=directed_actual,
        directed_shadow_to_actual_m=directed_shadow,
        sample_count=count,
        actual_input_points=int(actual.shape[0]),
        shadow_input_points=int(shadow.shape[0]),
        actual_dropped_points=actual_dropped,
        shadow_dropped_points=shadow_dropped,
    )


class SplitConformalEpisodeCalibrator:
    """Finite-sample split-conformal calibrator for episode scores.

    Larger nonconformity scores are assumed to be less trustworthy. Calibration
    must be split by independent episodes rather than by temporally adjacent
    frames.
    """

    def __init__(self, alpha: float = 0.10) -> None:
        if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be within (0, 1)")
        self.alpha = float(alpha)
        self._scores = np.empty(0, dtype=np.float64)
        self._q_hat: float | None = None
        self._rank: int | None = None
        self._fit_reason = "not_fit"
        self._dropped_scores = 0

    @property
    def fitted(self) -> bool:
        return self._q_hat is not None and bool(self._scores.size)

    def reset(self) -> dict[str, Any]:
        self._scores = np.empty(0, dtype=np.float64)
        self._q_hat = None
        self._rank = None
        self._fit_reason = "not_fit"
        self._dropped_scores = 0
        return self.summary()

    def fit(self, calibration_scores: Any) -> dict[str, Any]:
        try:
            raw = np.asarray(calibration_scores, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            self.reset()
            self._fit_reason = "non_numeric_calibration_scores"
            return self.summary()
        finite = raw[np.isfinite(raw)]
        self._dropped_scores = int(raw.size - finite.size)
        if finite.size == 0:
            self.reset()
            self._dropped_scores = int(raw.size)
            self._fit_reason = "no_finite_calibration_scores"
            return self.summary()

        self._scores = np.sort(finite)
        nominal_rank = int(math.ceil((self._scores.size + 1) * (1.0 - self.alpha)))
        self._rank = min(max(nominal_rank, 1), int(self._scores.size))
        self._q_hat = float(self._scores[self._rank - 1])
        self._fit_reason = "fit"
        return self.summary()

    def summary(self) -> dict[str, Any]:
        return _contract(
            fitted=self.fitted,
            state="FIT" if self.fitted else "NOT_FIT",
            reason=self._fit_reason,
            alpha=self.alpha,
            calibration_episode_count=int(self._scores.size),
            dropped_score_count=self._dropped_scores,
            quantile_rank=self._rank,
            q_hat=self._q_hat,
            exchangeability_scope="independent_episode_split_only",
        )

    def calibration_p_value(self, score: Any) -> dict[str, Any]:
        if not self.fitted:
            return _contract(
                valid=False,
                fitted=False,
                state="NOT_FIT",
                reason=self._fit_reason,
                p_value=None,
            )
        try:
            value = float(score)
        except (TypeError, ValueError):
            return _contract(
                valid=False,
                fitted=True,
                state="FIT",
                reason="non_numeric_score",
                p_value=None,
            )
        if not math.isfinite(value):
            return _contract(
                valid=False,
                fitted=True,
                state="FIT",
                reason="non_finite_score",
                p_value=None,
            )
        exceedances = int(np.count_nonzero(self._scores >= value))
        p_value = float((exceedances + 1) / (self._scores.size + 1))
        return _contract(
            valid=True,
            fitted=True,
            state="FIT",
            score=value,
            p_value=p_value,
            calibration_episode_count=int(self._scores.size),
        )

    def upper_envelope(
        self,
        prediction: Any,
        *,
        clip_min: float | None = 0.0,
        clip_max: float | None = 1.0,
    ) -> dict[str, Any]:
        if not self.fitted:
            return _contract(
                valid=False,
                fitted=False,
                state="NOT_FIT",
                reason=self._fit_reason,
                upper_envelope=None,
            )
        try:
            values = np.asarray(prediction, dtype=np.float64)
        except (TypeError, ValueError):
            return _contract(
                valid=False,
                fitted=True,
                state="FIT",
                reason="non_numeric_prediction",
                upper_envelope=None,
            )
        if values.size == 0 or not np.all(np.isfinite(values)):
            return _contract(
                valid=False,
                fitted=True,
                state="FIT",
                reason="empty_or_non_finite_prediction",
                upper_envelope=None,
            )

        upper = values + float(self._q_hat)
        if clip_min is not None:
            upper = np.maximum(upper, float(clip_min))
        if clip_max is not None:
            upper = np.minimum(upper, float(clip_max))
        return _contract(
            valid=True,
            fitted=True,
            state="FIT",
            q_hat=self._q_hat,
            upper_envelope=upper,
            shape=list(upper.shape),
        )


@dataclass(frozen=True)
class ShadowGuardConfig:
    required_sensors: tuple[str, ...] = ("lidar", "odom")
    review_p_value: float = 0.10
    untrusted_p_value: float = 0.05
    review_bev_disagreement: float = 0.35
    untrusted_bev_disagreement: float = 0.60
    review_js_divergence: float = 0.25
    untrusted_js_divergence: float = 0.50
    review_ade_m: float = 0.25
    untrusted_ade_m: float = 0.50

    def __post_init__(self) -> None:
        if not 0.0 <= self.untrusted_p_value <= self.review_p_value <= 1.0:
            raise ValueError("p-value thresholds must satisfy 0 <= untrusted <= review <= 1")
        for name, value in (
            ("review_bev_disagreement", self.review_bev_disagreement),
            ("untrusted_bev_disagreement", self.untrusted_bev_disagreement),
            ("review_js_divergence", self.review_js_divergence),
            ("untrusted_js_divergence", self.untrusted_js_divergence),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1]")
        if self.untrusted_bev_disagreement < self.review_bev_disagreement:
            raise ValueError("untrusted BEV threshold must be >= review threshold")
        if self.untrusted_js_divergence < self.review_js_divergence:
            raise ValueError("untrusted JS threshold must be >= review threshold")
        if (
            not math.isfinite(self.review_ade_m)
            or not math.isfinite(self.untrusted_ade_m)
            or self.review_ade_m < 0.0
            or self.untrusted_ade_m < self.review_ade_m
        ):
            raise ValueError("ADE thresholds must be finite and 0 <= review <= untrusted")


def _sensor_dict(value: SensorHealth | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, SensorHealth):
        return value.to_dict()
    if isinstance(value, Mapping):
        out = dict(value)
        out["cmd_vel_authority"] = CMD_VEL_AUTHORITY
        out["shadow_only"] = SHADOW_ONLY
        return out
    return _contract(state=SensorHealthState.INVALID.value, healthy=False, reasons=["invalid_health"])


class ShadowGuard:
    """Fuse confidence evidence into a shadow-only trust state."""

    def __init__(self, config: ShadowGuardConfig = ShadowGuardConfig()) -> None:
        self.config = config

    def assess(
        self,
        sensor_health: Mapping[str, SensorHealth | Mapping[str, Any]],
        *,
        bev_metrics: Mapping[str, Any] | None = None,
        energy_result: Mapping[str, Any] | None = None,
        token_js_result: Mapping[str, Any] | None = None,
        trajectory_result: Mapping[str, Any] | None = None,
        conformal_result: Mapping[str, Any] | None = None,
        monitor_online: bool = True,
        monitor_errors: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        source: Mapping[str, Any] = sensor_health
        nested = sensor_health.get("sensors") if isinstance(sensor_health, Mapping) else None
        if isinstance(nested, Mapping):
            source = nested
        sensors = {str(name).lower(): _sensor_dict(value) for name, value in source.items()}
        errors = [str(item) for item in (monitor_errors or ()) if str(item)]
        untrusted: list[str] = []
        review: list[str] = []

        if not monitor_online:
            errors.append("monitor_marked_offline")
        if not sensors:
            errors.append("no_sensor_health_inputs")
        if errors:
            state = ShadowState.MONITOR_OFFLINE
            return _contract(
                status=state.value,
                trusted=False,
                reasons=list(dict.fromkeys(errors)),
                review_reasons=[],
                sensor_health=sensors,
                evidence={},
                policy_effect="none_ignore_shadow_result",
            )

        required = {str(name).lower() for name in self.config.required_sensors}
        for name in sorted(required):
            health = sensors.get(name)
            if health is None:
                untrusted.append(f"required_sensor_missing:{name}")
                continue
            raw_state = health.get("state")
            state_value = raw_state.value if isinstance(raw_state, Enum) else str(raw_state or "")
            health_valid = bool(health.get("valid", health.get("healthy", False)))
            if state_value in {
                SensorHealthState.MISSING.value,
                SensorHealthState.STALE.value,
                SensorHealthState.INVALID.value,
            } or not health_valid:
                untrusted.append(f"required_sensor_unusable:{name}:{state_value or 'UNKNOWN'}")
            elif state_value == SensorHealthState.DEGRADED.value:
                review.append(f"required_sensor_degraded:{name}")

        for name, health in sorted(sensors.items()):
            if name in required:
                continue
            raw_state = health.get("state")
            state_value = raw_state.value if isinstance(raw_state, Enum) else str(raw_state or "")
            if state_value != SensorHealthState.HEALTHY.value:
                review.append(f"optional_sensor_not_healthy:{name}:{state_value or 'UNKNOWN'}")

        if bev_metrics is not None:
            if not bool(bev_metrics.get("valid")):
                review.append("bev_comparison_invalid")
            else:
                disagreement = bev_metrics.get("max_disagreement")
                if disagreement is None:
                    disagreement = bev_metrics.get("disagreement")
                if disagreement is not None and math.isfinite(float(disagreement)):
                    if float(disagreement) >= self.config.untrusted_bev_disagreement:
                        untrusted.append("cross_modal_bev_disagreement_severe")
                    elif float(disagreement) >= self.config.review_bev_disagreement:
                        review.append("cross_modal_bev_disagreement")

        if energy_result is not None:
            if not bool(energy_result.get("valid")):
                review.append("energy_ood_invalid")
            elif energy_result.get("is_ood") is True:
                untrusted.append("energy_ood_detected")

        if token_js_result is not None:
            if not bool(token_js_result.get("valid")):
                review.append("trajectory_token_js_invalid")
            else:
                js_value = token_js_result.get("js_normalized")
                if js_value is not None and math.isfinite(float(js_value)):
                    if float(js_value) >= self.config.untrusted_js_divergence:
                        untrusted.append("trajectory_token_divergence_severe")
                    elif float(js_value) >= self.config.review_js_divergence:
                        review.append("trajectory_token_divergence")

        if trajectory_result is not None:
            if not bool(trajectory_result.get("valid")):
                review.append("trajectory_comparison_invalid")
            else:
                ade = trajectory_result.get("ade_m")
                if ade is not None and math.isfinite(float(ade)):
                    if float(ade) >= self.config.untrusted_ade_m:
                        untrusted.append("trajectory_ade_severe")
                    elif float(ade) >= self.config.review_ade_m:
                        review.append("trajectory_ade")

        if conformal_result is not None:
            if not bool(conformal_result.get("fitted")):
                review.append("conformal_calibrator_not_fit")
            elif not bool(conformal_result.get("valid")):
                review.append("conformal_result_invalid")
            else:
                p_value = conformal_result.get("p_value")
                if p_value is None or not math.isfinite(float(p_value)):
                    review.append("conformal_p_value_invalid")
                elif float(p_value) <= self.config.untrusted_p_value:
                    untrusted.append("conformal_p_value_untrusted")
                elif float(p_value) <= self.config.review_p_value:
                    review.append("conformal_p_value_review")

        untrusted = list(dict.fromkeys(untrusted))
        review = list(dict.fromkeys(review))
        if untrusted:
            state = ShadowState.UNTRUSTED_SHADOW
        elif review:
            state = ShadowState.REVIEW
        else:
            state = ShadowState.TRUSTED_SHADOW

        evidence = {
            "bev": dict(bev_metrics) if bev_metrics is not None else None,
            "energy_ood": dict(energy_result) if energy_result is not None else None,
            "trajectory_token_js": dict(token_js_result) if token_js_result is not None else None,
            "trajectory_distance": dict(trajectory_result) if trajectory_result is not None else None,
            "conformal": dict(conformal_result) if conformal_result is not None else None,
        }
        return _contract(
            status=state.value,
            trusted=state is ShadowState.TRUSTED_SHADOW,
            reasons=untrusted,
            review_reasons=review,
            sensor_health=sensors,
            evidence=evidence,
            policy_effect="none_observe_and_record_only",
        )


__all__ = [
    "CMD_VEL_AUTHORITY",
    "DEFAULT_SENSOR_POLICIES",
    "SHADOW_ONLY",
    "SensorHealth",
    "SensorHealthMonitor",
    "SensorHealthState",
    "SensorPolicy",
    "ShadowGuard",
    "ShadowGuardConfig",
    "ShadowState",
    "SplitConformalEpisodeCalibrator",
    "assess_sensor_health",
    "compare_bev",
    "cross_modal_bev_metrics",
    "energy_ood",
    "fuse_trajectory_token_evidence",
    "occupancy_conditioned_trajectory_tokens",
    "trajectory_distance_metrics",
    "trajectory_token_js_divergence",
]
