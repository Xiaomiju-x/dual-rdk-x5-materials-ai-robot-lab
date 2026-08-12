"""Odometry-aligned static/dynamic semantic memory and ghost-risk estimation."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, exp, isfinite, log, sin
from threading import RLock

import numpy as np

from ..contracts.core import BEVGeometryV2
from .contracts import (
    ContractError,
    OdometryDelta,
    SemanticBEVFrame,
    SparseVectorToken,
)
from .validation import (
    FrameAssessment,
    FreshnessQualityPolicy,
    require_acceptable_frame,
)


class MemoryUpdateError(ContractError):
    """Raised before state mutation when temporal or odometry data is invalid."""


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    """Update, decay, and occlusion parameters for the passive observer."""

    static_half_life_s: float = 20.0
    dynamic_half_life_s: float = 1.5
    static_observation_gain: float = 0.65
    dynamic_observation_gain: float = 0.85
    static_clear_gain: float = 0.08
    dynamic_clear_gain: float = 0.55
    token_min_confidence: float = 0.05
    max_step_translation_m: float = 2.0
    max_step_yaw_rad: float = 1.6
    max_step_dt_s: float = 2.0
    max_odom_time_error_s: float = 0.10
    ghost_obstacle_threshold: float = 0.30
    ghost_decay_distance_m: float = 1.5
    ghost_unknown_prior: float = 0.08
    ghost_static_memory_weight: float = 0.35
    ghost_dynamic_memory_weight: float = 0.90
    ghost_shadow_half_width_cells: int = 1
    ghost_horizon_s: float = 1.2

    def __post_init__(self) -> None:
        positive = (
            ("static_half_life_s", self.static_half_life_s),
            ("dynamic_half_life_s", self.dynamic_half_life_s),
            ("max_step_translation_m", self.max_step_translation_m),
            ("max_step_yaw_rad", self.max_step_yaw_rad),
            ("max_step_dt_s", self.max_step_dt_s),
            ("ghost_decay_distance_m", self.ghost_decay_distance_m),
            ("ghost_horizon_s", self.ghost_horizon_s),
        )
        for name, value in positive:
            if not isfinite(float(value)) or float(value) <= 0.0:
                raise ContractError(f"{name} must be finite and positive")
        non_negative = (
            ("max_odom_time_error_s", self.max_odom_time_error_s),
            ("ghost_unknown_prior", self.ghost_unknown_prior),
        )
        for name, value in non_negative:
            if not isfinite(float(value)) or float(value) < 0.0:
                raise ContractError(f"{name} must be finite and non-negative")
        probabilities = (
            ("static_observation_gain", self.static_observation_gain),
            ("dynamic_observation_gain", self.dynamic_observation_gain),
            ("static_clear_gain", self.static_clear_gain),
            ("dynamic_clear_gain", self.dynamic_clear_gain),
            ("token_min_confidence", self.token_min_confidence),
            ("ghost_obstacle_threshold", self.ghost_obstacle_threshold),
            (
                "ghost_static_memory_weight",
                self.ghost_static_memory_weight,
            ),
            (
                "ghost_dynamic_memory_weight",
                self.ghost_dynamic_memory_weight,
            ),
        )
        for name, value in probabilities:
            if not isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ContractError(f"{name} must be finite and in [0, 1]")
        if self.ghost_unknown_prior > 1.0:
            raise ContractError("ghost_unknown_prior must not exceed 1")
        if (
            not isinstance(self.ghost_shadow_half_width_cells, int)
            or isinstance(self.ghost_shadow_half_width_cells, bool)
            or not 0 <= self.ghost_shadow_half_width_cells <= 4
        ):
            raise ContractError(
                "ghost_shadow_half_width_cells must be an integer in [0, 4]"
            )


def _immutable_probability_grid(
    value: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ContractError("memory grid has invalid shape or values")
    array = np.ascontiguousarray(np.clip(array, 0.0, 1.0))
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """Immutable output from one accepted memory update."""

    static_probability: np.ndarray
    dynamic_probability: np.ndarray
    ghost_risk: np.ndarray
    visibility: np.ndarray
    vector_tokens: tuple[SparseVectorToken, ...]
    timestamp_s: float | None
    update_count: int
    assessment: FrameAssessment | None
    geometry: BEVGeometryV2

    def __post_init__(self) -> None:
        shape = self.geometry.shape
        object.__setattr__(
            self,
            "static_probability",
            _immutable_probability_grid(self.static_probability, shape),
        )
        object.__setattr__(
            self,
            "dynamic_probability",
            _immutable_probability_grid(self.dynamic_probability, shape),
        )
        object.__setattr__(
            self,
            "ghost_risk",
            _immutable_probability_grid(self.ghost_risk, shape),
        )
        visibility = np.asarray(self.visibility, dtype=np.bool_)
        if visibility.shape != shape:
            raise ContractError("snapshot visibility has invalid shape")
        visibility = np.ascontiguousarray(visibility)
        visibility.setflags(write=False)
        object.__setattr__(self, "visibility", visibility)
        object.__setattr__(self, "vector_tokens", tuple(self.vector_tokens))


def warp_bev_nearest(
    bev: np.ndarray,
    odometry: OdometryDelta,
    geometry: BEVGeometryV2 | None = None,
) -> np.ndarray:
    """Warp an old robot-frame BEV into the current robot frame."""

    geom = geometry or BEVGeometryV2()
    array = np.asarray(bev, dtype=np.float32)
    if array.ndim == 2:
        source = array[np.newaxis, ...]
        squeeze = True
    elif array.ndim == 3:
        source = array
        squeeze = False
    else:
        raise ContractError("bev must have shape (H, W) or (C, H, W)")
    if source.shape[1:] != geom.shape or not np.isfinite(source).all():
        raise ContractError("bev shape or values violate the warp contract")
    if not isinstance(odometry, OdometryDelta):
        raise TypeError("odometry must be OdometryDelta")

    x_centers = (
        geom.x_min_m
        + (np.arange(geom.height, dtype=np.float64) + 0.5)
        * geom.resolution_m
    )
    y_centers = (
        geom.y_min_m
        + (np.arange(geom.width, dtype=np.float64) + 0.5)
        * geom.resolution_m
    )
    destination_x, destination_y = np.meshgrid(
        x_centers,
        y_centers,
        indexing="ij",
    )
    cosine = cos(odometry.dyaw_rad)
    sine = sin(odometry.dyaw_rad)
    source_x = (
        cosine * destination_x
        - sine * destination_y
        + odometry.dx_m
    )
    source_y = (
        sine * destination_x
        + cosine * destination_y
        + odometry.dy_m
    )
    source_rows = np.floor(
        (
            source_x
            - (geom.x_min_m + 0.5 * geom.resolution_m)
        )
        / geom.resolution_m
        + 0.5
    ).astype(np.int64)
    source_cols = np.floor(
        (
            source_y
            - (geom.y_min_m + 0.5 * geom.resolution_m)
        )
        / geom.resolution_m
        + 0.5
    ).astype(np.int64)
    inside = (
        (source_rows >= 0)
        & (source_rows < geom.height)
        & (source_cols >= 0)
        & (source_cols < geom.width)
    )
    output = np.zeros_like(source, dtype=np.float32)
    destination_rows, destination_cols = np.nonzero(inside)
    output[:, destination_rows, destination_cols] = source[
        :,
        source_rows[inside],
        source_cols[inside],
    ]
    return output[0] if squeeze else output


def _warp_token(
    token: SparseVectorToken,
    odometry: OdometryDelta,
    geometry: BEVGeometryV2,
    confidence: float,
) -> SparseVectorToken | None:
    cosine = cos(odometry.dyaw_rad)
    sine = sin(odometry.dyaw_rad)
    points: list[tuple[float, float]] = []
    for x_old, y_old in token.points_xy_m:
        shifted_x = x_old - odometry.dx_m
        shifted_y = y_old - odometry.dy_m
        x_new = cosine * shifted_x + sine * shifted_y
        y_new = -sine * shifted_x + cosine * shifted_y
        if (
            geometry.x_min_m <= x_new < geometry.x_max_m
            and geometry.y_min_m <= y_new < geometry.y_max_m
        ):
            points.append((x_new, y_new))
    if not points:
        return None
    vx_old, vy_old = token.velocity_xy_mps
    velocity = (
        cosine * vx_old + sine * vy_old,
        -sine * vx_old + cosine * vy_old,
    )
    return SparseVectorToken(
        token_id=token.token_id,
        kind=token.kind,
        class_id=token.class_id,
        confidence=confidence,
        points_xy_m=tuple(points),
        velocity_xy_mps=velocity,
        track_id=token.track_id,
    )


def compute_ghost_risk(
    static_probability: np.ndarray,
    dynamic_probability: np.ndarray,
    visibility: np.ndarray,
    *,
    geometry: BEVGeometryV2 | None = None,
    camera_origin_xy_m: tuple[float, float] = (0.0, 0.0),
    ego_speed_mps: float = 0.0,
    config: MemoryConfig | None = None,
) -> np.ndarray:
    """Estimate bounded risk in cells hidden behind visible obstacles."""

    geom = geometry or BEVGeometryV2()
    cfg = config or MemoryConfig()
    static = np.asarray(static_probability, dtype=np.float32)
    dynamic = np.asarray(dynamic_probability, dtype=np.float32)
    visible = np.asarray(visibility, dtype=np.bool_)
    if (
        static.shape != geom.shape
        or dynamic.shape != geom.shape
        or visible.shape != geom.shape
        or not np.isfinite(static).all()
        or not np.isfinite(dynamic).all()
    ):
        raise ContractError("ghost-risk inputs violate the BEV contract")
    camera_x, camera_y = (float(axis) for axis in camera_origin_xy_m)
    speed = float(ego_speed_mps)
    if not all(isfinite(value) for value in (camera_x, camera_y, speed)):
        raise ContractError("ghost-risk pose and speed must be finite")
    if speed < 0.0:
        raise ContractError("ego_speed_mps must be non-negative")

    hidden = ~visible
    memory_prior = np.maximum(
        static * cfg.ghost_static_memory_weight,
        dynamic * cfg.ghost_dynamic_memory_weight,
    )
    ghost = np.where(hidden, memory_prior, 0.0).astype(np.float32)

    camera_row = (
        (camera_x - geom.x_min_m) / geom.resolution_m - 0.5
    )
    camera_col = (
        (camera_y - geom.y_min_m) / geom.resolution_m - 0.5
    )
    obstacle = np.maximum(static, dynamic)
    blockers = np.argwhere(
        visible & (obstacle >= cfg.ghost_obstacle_threshold)
    )
    max_steps = int(np.ceil(np.hypot(geom.height, geom.width))) + 1
    half_width = cfg.ghost_shadow_half_width_cells

    for row, col in blockers:
        delta_row = float(row) - camera_row
        delta_col = float(col) - camera_col
        norm = float(np.hypot(delta_row, delta_col))
        if norm < 0.5:
            continue
        unit_row = delta_row / norm
        unit_col = delta_col / norm
        perpendicular_row = -unit_col
        perpendicular_col = unit_row
        strength = float(obstacle[row, col])
        for step in range(1, max_steps):
            center_row = float(row) + unit_row * step
            center_col = float(col) + unit_col * step
            any_inside = False
            for width in range(-half_width, half_width + 1):
                target_row = int(
                    round(center_row + perpendicular_row * width)
                )
                target_col = int(
                    round(center_col + perpendicular_col * width)
                )
                if not (
                    0 <= target_row < geom.height
                    and 0 <= target_col < geom.width
                ):
                    continue
                any_inside = True
                if visible[target_row, target_col]:
                    continue
                distance_m = step * geom.resolution_m
                candidate = strength * exp(
                    -distance_m / cfg.ghost_decay_distance_m
                )
                if candidate > ghost[target_row, target_col]:
                    ghost[target_row, target_col] = candidate
            if not any_inside:
                break

    visible_neighbor = np.zeros_like(visible)
    visible_neighbor[1:] |= visible[:-1]
    visible_neighbor[:-1] |= visible[1:]
    visible_neighbor[:, 1:] |= visible[:, :-1]
    visible_neighbor[:, :-1] |= visible[:, 1:]
    frontier = hidden & visible_neighbor
    ghost[frontier] = np.maximum(
        ghost[frontier],
        cfg.ghost_unknown_prior,
    )
    speed_scale = 1.0 + min(
        1.0,
        speed * cfg.ghost_horizon_s
        / max(cfg.ghost_decay_distance_m, 1e-6),
    )
    ghost = np.where(hidden, ghost * speed_scale, 0.0)
    return np.clip(ghost, 0.0, 1.0).astype(np.float32)


def _decay_factor(dt_s: float, half_life_s: float) -> float:
    return exp(-log(2.0) * dt_s / half_life_s)


def _fuse_visible_evidence(
    prior: np.ndarray,
    evidence: np.ndarray,
    visibility: np.ndarray,
    *,
    observation_gain: float,
    clear_gain: float,
) -> np.ndarray:
    cleared = prior * (1.0 - clear_gain * (1.0 - evidence))
    fused = 1.0 - (1.0 - cleared) * (
        1.0 - evidence * observation_gain
    )
    return np.where(visibility, fused, prior).astype(np.float32)


class DualBEVMemory:
    """Thread-safe, read-only-consumer semantic memory.

    Rejected frames and invalid odometry leave the previous snapshot untouched.
    """

    def __init__(
        self,
        *,
        geometry: BEVGeometryV2 | None = None,
        config: MemoryConfig | None = None,
    ) -> None:
        self.geometry = geometry or BEVGeometryV2()
        self.config = config or MemoryConfig()
        self._lock = RLock()
        self._static = np.zeros(self.geometry.shape, dtype=np.float32)
        self._dynamic = np.zeros(self.geometry.shape, dtype=np.float32)
        self._ghost = np.zeros(self.geometry.shape, dtype=np.float32)
        self._visibility = np.zeros(self.geometry.shape, dtype=np.bool_)
        self._tokens: dict[str, SparseVectorToken] = {}
        self._timestamp_s: float | None = None
        self._update_count = 0
        self._assessment: FrameAssessment | None = None

    def reset(self) -> None:
        """Clear candidate-only memory without touching any external system."""

        with self._lock:
            self._static.fill(0.0)
            self._dynamic.fill(0.0)
            self._ghost.fill(0.0)
            self._visibility.fill(False)
            self._tokens.clear()
            self._timestamp_s = None
            self._update_count = 0
            self._assessment = None

    def snapshot(self) -> MemorySnapshot:
        with self._lock:
            return MemorySnapshot(
                static_probability=self._static.copy(),
                dynamic_probability=self._dynamic.copy(),
                ghost_risk=self._ghost.copy(),
                visibility=self._visibility.copy(),
                vector_tokens=tuple(
                    self._tokens[key] for key in sorted(self._tokens)
                ),
                timestamp_s=self._timestamp_s,
                update_count=self._update_count,
                assessment=self._assessment,
                geometry=self.geometry,
            )

    def _validate_step(
        self,
        frame: SemanticBEVFrame,
        odometry: OdometryDelta,
    ) -> float:
        if frame.geometry != self.geometry:
            raise MemoryUpdateError("frame and memory geometries differ")
        translation = float(np.hypot(odometry.dx_m, odometry.dy_m))
        if translation > self.config.max_step_translation_m:
            raise MemoryUpdateError("odometry translation exceeds step limit")
        if abs(odometry.dyaw_rad) > self.config.max_step_yaw_rad:
            raise MemoryUpdateError("odometry yaw exceeds step limit")
        if odometry.dt_s > self.config.max_step_dt_s:
            raise MemoryUpdateError("odometry dt exceeds step limit")
        if self._timestamp_s is None:
            if odometry.dt_s > self.config.max_odom_time_error_s:
                raise MemoryUpdateError(
                    "first update must not carry an elapsed odometry interval"
                )
            return 0.0
        frame_dt = frame.timestamp_s - self._timestamp_s
        if frame_dt <= 0.0:
            raise MemoryUpdateError("frame timestamps must increase monotonically")
        if frame_dt > self.config.max_step_dt_s:
            raise MemoryUpdateError("frame time step exceeds the memory limit")
        if (
            abs(frame_dt - odometry.dt_s)
            > self.config.max_odom_time_error_s
        ):
            raise MemoryUpdateError(
                "odometry dt does not match the semantic frame interval"
            )
        return frame_dt

    def update(
        self,
        frame: SemanticBEVFrame,
        odometry: OdometryDelta | None = None,
        *,
        now_s: float,
        ego_speed_mps: float = 0.0,
        policy: FreshnessQualityPolicy | None = None,
    ) -> MemorySnapshot:
        """Validate, warp, decay, and atomically commit one semantic frame."""

        if not isinstance(frame, SemanticBEVFrame):
            raise TypeError("frame must be SemanticBEVFrame")
        delta = odometry or OdometryDelta()
        if not isinstance(delta, OdometryDelta):
            raise TypeError("odometry must be OdometryDelta")
        speed = float(ego_speed_mps)
        if not isfinite(speed) or speed < 0.0:
            raise MemoryUpdateError("ego_speed_mps must be finite and non-negative")
        assessment = require_acceptable_frame(
            frame,
            now_s=now_s,
            policy=policy,
        )

        with self._lock:
            frame_dt = self._validate_step(frame, delta)
            if self._timestamp_s is None:
                warped_static = self._static.copy()
                warped_dynamic = self._dynamic.copy()
                warped_tokens: dict[str, SparseVectorToken] = {}
            else:
                warped_static = warp_bev_nearest(
                    self._static,
                    delta,
                    self.geometry,
                )
                warped_dynamic = warp_bev_nearest(
                    self._dynamic,
                    delta,
                    self.geometry,
                )
                static_decay = _decay_factor(
                    frame_dt,
                    self.config.static_half_life_s,
                )
                dynamic_decay = _decay_factor(
                    frame_dt,
                    self.config.dynamic_half_life_s,
                )
                warped_static *= static_decay
                warped_dynamic *= dynamic_decay
                warped_tokens = {}
                for token in self._tokens.values():
                    decay = dynamic_decay if token.is_dynamic else static_decay
                    confidence = token.confidence * decay
                    if confidence < self.config.token_min_confidence:
                        continue
                    warped = _warp_token(
                        token,
                        delta,
                        self.geometry,
                        confidence,
                    )
                    if warped is not None:
                        warped_tokens[warped.token_id] = warped

            observation_weight = frame.quality.overall_score
            risk = frame.semantic_risk * frame.confidence
            static_evidence = (
                risk
                * (1.0 - frame.dynamic_probability)
                * observation_weight
            )
            dynamic_evidence = (
                risk
                * frame.dynamic_probability
                * observation_weight
            )
            new_static = _fuse_visible_evidence(
                warped_static,
                static_evidence,
                frame.visibility,
                observation_gain=self.config.static_observation_gain,
                clear_gain=self.config.static_clear_gain,
            )
            new_dynamic = _fuse_visible_evidence(
                warped_dynamic,
                dynamic_evidence,
                frame.visibility,
                observation_gain=self.config.dynamic_observation_gain,
                clear_gain=self.config.dynamic_clear_gain,
            )
            for token in frame.vector_tokens:
                warped_tokens[token.token_id] = token

            camera_origin = (
                frame.extrinsics.translation_m[0],
                frame.extrinsics.translation_m[1],
            )
            new_ghost = compute_ghost_risk(
                new_static,
                new_dynamic,
                frame.visibility,
                geometry=self.geometry,
                camera_origin_xy_m=camera_origin,
                ego_speed_mps=speed,
                config=self.config,
            )

            self._static = np.ascontiguousarray(new_static, dtype=np.float32)
            self._dynamic = np.ascontiguousarray(new_dynamic, dtype=np.float32)
            self._ghost = np.ascontiguousarray(new_ghost, dtype=np.float32)
            self._visibility = np.ascontiguousarray(
                frame.visibility,
                dtype=np.bool_,
            )
            self._tokens = warped_tokens
            self._timestamp_s = frame.timestamp_s
            self._update_count += 1
            self._assessment = assessment
            return self.snapshot()


__all__ = [
    "DualBEVMemory",
    "MemoryConfig",
    "MemorySnapshot",
    "MemoryUpdateError",
    "compute_ghost_risk",
    "warp_bev_nearest",
]
