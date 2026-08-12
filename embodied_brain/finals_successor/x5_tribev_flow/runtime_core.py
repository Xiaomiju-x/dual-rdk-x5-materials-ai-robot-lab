"""Pure conversion helpers used by the isolated ROS2 shadow runtime."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .contracts import BEVGeometry, OdometryDelta, SemanticProvenance


@dataclass(frozen=True, slots=True)
class Pose2D:
    x_m: float
    y_m: float
    yaw_rad: float
    timestamp_s: float

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (self.x_m, self.y_m, self.yaw_rad, self.timestamp_s)
        ):
            raise ValueError("Pose2D values must be finite")


@dataclass(frozen=True, slots=True)
class OccupancyGridSpec:
    width: int
    height: int
    resolution_m: float
    origin_x_m: float
    origin_y_m: float
    origin_yaw_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("occupancy grid dimensions must be positive")
        if not math.isfinite(self.resolution_m) or self.resolution_m <= 0.0:
            raise ValueError("occupancy grid resolution must be finite and positive")
        if not all(
            math.isfinite(value)
            for value in (
                self.origin_x_m,
                self.origin_y_m,
                self.origin_yaw_rad,
            )
        ):
            raise ValueError("occupancy grid origin values must be finite")


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    values = np.asarray([x, y, z, w], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("quaternion contains non-finite values")
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


def odometry_delta(previous: Pose2D | None, current: Pose2D) -> OdometryDelta:
    if previous is None:
        return OdometryDelta()
    world_dx = current.x_m - previous.x_m
    world_dy = current.y_m - previous.y_m
    cosine = math.cos(previous.yaw_rad)
    sine = math.sin(previous.yaw_rad)
    dx_robot = cosine * world_dx + sine * world_dy
    dy_robot = -sine * world_dx + cosine * world_dy
    dyaw = math.atan2(
        math.sin(current.yaw_rad - previous.yaw_rad),
        math.cos(current.yaw_rad - previous.yaw_rad),
    )
    return OdometryDelta(
        dx_m=float(dx_robot),
        dy_m=float(dy_robot),
        dyaw_rad=float(dyaw),
        dt_s=max(0.0, current.timestamp_s - previous.timestamp_s),
    )


def laser_scan_to_points(
    ranges: Any,
    *,
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    origin_x_m: float = 0.0,
    origin_y_m: float = 0.0,
) -> np.ndarray:
    values = np.asarray(ranges, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    scalars = (
        angle_min,
        angle_increment,
        range_min,
        range_max,
        origin_x_m,
        origin_y_m,
    )
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError("LaserScan geometry contains non-finite values")
    if angle_increment == 0.0 or range_min < 0.0 or range_max <= range_min:
        raise ValueError("LaserScan range/angle contract is invalid")
    angles = angle_min + np.arange(values.size, dtype=np.float64) * angle_increment
    valid = (
        np.isfinite(values)
        & (values >= range_min)
        & (values <= range_max)
    )
    distances = values[valid]
    angles = angles[valid]
    points = np.column_stack(
        (
            origin_x_m + distances * np.cos(angles),
            origin_y_m + distances * np.sin(angles),
        )
    )
    return np.ascontiguousarray(points, dtype=np.float64)


def depth_scan_to_points(
    ranges: Any,
    *,
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    origin_x_m: float = 0.25,
    origin_y_m: float = 0.0,
) -> np.ndarray:
    points_xy = laser_scan_to_points(
        ranges,
        angle_min=angle_min,
        angle_increment=angle_increment,
        range_min=range_min,
        range_max=range_max,
        origin_x_m=origin_x_m,
        origin_y_m=origin_y_m,
    )
    if not points_xy.size:
        return np.empty((0, 3), dtype=np.float64)
    return np.column_stack(
        (points_xy, np.zeros(points_xy.shape[0], dtype=np.float64))
    )


def occupancy_grid_to_semantic_bev(
    data: Any,
    source: OccupancyGridSpec,
    target: BEVGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a standard ROS OccupancyGrid into the TriBEV metric geometry."""

    flat = np.asarray(data, dtype=np.float32).reshape(-1)
    if flat.size != source.width * source.height:
        raise ValueError(
            f"occupancy data size {flat.size} != {source.width * source.height}"
        )
    source_grid = flat.reshape(source.height, source.width)

    rows = np.arange(target.height, dtype=np.float64)
    cols = np.arange(target.width, dtype=np.float64)
    x = target.x_min_m + (rows + 0.5) * target.resolution_m
    y = target.y_min_m + (cols + 0.5) * target.resolution_m
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")

    dx = x_grid - source.origin_x_m
    dy = y_grid - source.origin_y_m
    cosine = math.cos(source.origin_yaw_rad)
    sine = math.sin(source.origin_yaw_rad)
    source_x = cosine * dx + sine * dy
    source_y = -sine * dx + cosine * dy
    source_cols = np.floor(source_x / source.resolution_m).astype(np.int64)
    source_rows = np.floor(source_y / source.resolution_m).astype(np.int64)

    inside = (
        (source_cols >= 0)
        & (source_cols < source.width)
        & (source_rows >= 0)
        & (source_rows < source.height)
    )
    risk = np.zeros(target.shape, dtype=np.float32)
    known = np.zeros(target.shape, dtype=bool)
    target_rows, target_cols = np.nonzero(inside)
    sampled = source_grid[
        source_rows[inside],
        source_cols[inside],
    ]
    sampled_known = np.isfinite(sampled) & (sampled >= 0.0)
    if np.any(sampled_known):
        selected_rows = target_rows[sampled_known]
        selected_cols = target_cols[sampled_known]
        risk[selected_rows, selected_cols] = np.clip(
            sampled[sampled_known] / 100.0,
            0.0,
            1.0,
        )
        known[selected_rows, selected_cols] = True
    return risk, known


def normalize_semantic_provenance(
    payload: Mapping[str, Any] | None,
) -> tuple[SemanticProvenance, bool]:
    provenance = payload if isinstance(payload, Mapping) else {}
    nested = provenance.get("provenance")
    if isinstance(nested, Mapping):
        provenance = nested
    state = str(provenance.get("state") or "").strip().lower()
    image_supplied = bool(provenance.get("image_supplied"))
    aliases = {
        "live_camera": SemanticProvenance.LIVE_CAMERA,
        "cached": SemanticProvenance.CACHED,
        "cached_camera": SemanticProvenance.CACHED,
        "fixture": SemanticProvenance.FIXTURE_PRIOR,
        "fixture_prior": SemanticProvenance.FIXTURE_PRIOR,
    }
    return aliases.get(state, SemanticProvenance.UNAVAILABLE), image_supplied


def image_message_to_bgr(
    data: bytes | bytearray | memoryview,
    *,
    width: int,
    height: int,
    step: int,
    encoding: str,
) -> np.ndarray:
    if width <= 0 or height <= 0 or step < width * 3:
        raise ValueError("invalid ROS image dimensions")
    normalized = str(encoding).strip().lower()
    if normalized not in {"bgr8", "rgb8"}:
        raise ValueError(f"unsupported image encoding: {encoding}")
    raw = np.frombuffer(data, dtype=np.uint8)
    if raw.size < height * step:
        raise ValueError("ROS image buffer is shorter than height*step")
    rows = raw[: height * step].reshape(height, step)
    image = rows[:, : width * 3].reshape(height, width, 3)
    if normalized == "rgb8":
        image = image[:, :, ::-1]
    return np.ascontiguousarray(image, dtype=np.uint8)


def parse_reference_trajectory_probabilities(
    payload: Mapping[str, Any] | None,
    *,
    token_count: int = 9,
) -> np.ndarray | None:
    if not isinstance(payload, Mapping):
        return None
    arc_tokens = payload.get("arc_tokens")
    source = arc_tokens if isinstance(arc_tokens, Mapping) else payload
    tokens = source.get("tokens") if isinstance(source, Mapping) else None
    if not isinstance(tokens, list):
        return None
    probabilities = np.zeros(token_count, dtype=np.float64)
    seen = np.zeros(token_count, dtype=bool)
    for item in tokens:
        if not isinstance(item, Mapping):
            continue
        try:
            token_id = int(item.get("token_id"))
            probability = float(item.get("probability"))
        except (TypeError, ValueError):
            continue
        if 0 <= token_id < token_count and math.isfinite(probability) and probability >= 0.0:
            probabilities[token_id] = probability
            seen[token_id] = True
    if not np.any(seen) or probabilities.sum() <= 0.0:
        return None
    return probabilities / probabilities.sum()


def occupancy_probability_to_ros_data(probability: Any) -> list[int]:
    grid = np.asarray(probability, dtype=np.float32)
    if grid.ndim != 2:
        raise ValueError("occupancy probability must be 2D")
    values = np.rint(np.clip(grid.T, 0.0, 1.0) * 100.0).astype(np.int8)
    return values.reshape(-1).astype(np.int16).tolist()


__all__ = [
    "OccupancyGridSpec",
    "Pose2D",
    "depth_scan_to_points",
    "image_message_to_bgr",
    "laser_scan_to_points",
    "normalize_semantic_provenance",
    "occupancy_grid_to_semantic_bev",
    "occupancy_probability_to_ros_data",
    "odometry_delta",
    "parse_reference_trajectory_probabilities",
    "quaternion_to_yaw",
]
