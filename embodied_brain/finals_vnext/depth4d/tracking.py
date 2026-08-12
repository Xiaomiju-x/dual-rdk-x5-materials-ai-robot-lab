"""Connected components, nearest-neighbour tracks, and radial TTC."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from .bev import cell_centres
from .contracts import (
    BEVGeometry,
    DepthBEVGrid,
    GRID_SIZE,
    TrackerConfig,
)


@dataclass
class ComponentBatch:
    """Fixed-capacity connected-component output."""

    valid: np.ndarray
    centroid_xy_m: np.ndarray
    cell_count: np.ndarray
    bbox_rc: np.ndarray
    min_height_m: np.ndarray
    max_height_m: np.ndarray

    @classmethod
    def empty(cls, capacity: int) -> "ComponentBatch":
        return cls(
            valid=np.zeros(capacity, dtype=np.bool_),
            centroid_xy_m=np.zeros((capacity, 2), dtype=np.float32),
            cell_count=np.zeros(capacity, dtype=np.int32),
            bbox_rc=np.full((capacity, 4), -1, dtype=np.int32),
            min_height_m=np.zeros(capacity, dtype=np.float32),
            max_height_m=np.zeros(capacity, dtype=np.float32),
        )

    @property
    def capacity(self) -> int:
        return int(self.valid.shape[0])

    @property
    def count(self) -> int:
        return int(np.count_nonzero(self.valid))

    def validate(self) -> None:
        capacity = self.capacity
        if self.valid.dtype != np.bool_:
            raise ValueError("component valid mask must use bool")
        expected = {
            "centroid_xy_m": ((capacity, 2), np.float32),
            "cell_count": ((capacity,), np.int32),
            "bbox_rc": ((capacity, 4), np.int32),
            "min_height_m": ((capacity,), np.float32),
            "max_height_m": ((capacity,), np.float32),
        }
        for name, (shape, dtype) in expected.items():
            array = getattr(self, name)
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")


@dataclass
class TrackBatch:
    """Fixed-capacity track and TTC output."""

    valid: np.ndarray
    observed: np.ndarray
    track_id: np.ndarray
    position_xy_m: np.ndarray
    velocity_xy_mps: np.ndarray
    speed_mps: np.ndarray
    ttc_s: np.ndarray
    age_since_seen_s: np.ndarray
    hit_count: np.ndarray

    @classmethod
    def empty(cls, capacity: int) -> "TrackBatch":
        return cls(
            valid=np.zeros(capacity, dtype=np.bool_),
            observed=np.zeros(capacity, dtype=np.bool_),
            track_id=np.full(capacity, -1, dtype=np.int64),
            position_xy_m=np.zeros((capacity, 2), dtype=np.float32),
            velocity_xy_mps=np.zeros((capacity, 2), dtype=np.float32),
            speed_mps=np.zeros(capacity, dtype=np.float32),
            ttc_s=np.full(capacity, np.inf, dtype=np.float32),
            age_since_seen_s=np.zeros(capacity, dtype=np.float32),
            hit_count=np.zeros(capacity, dtype=np.int32),
        )

    @property
    def capacity(self) -> int:
        return int(self.valid.shape[0])

    @property
    def count(self) -> int:
        return int(np.count_nonzero(self.valid))

    def validate(self) -> None:
        capacity = self.capacity
        expected = {
            "valid": ((capacity,), np.bool_),
            "observed": ((capacity,), np.bool_),
            "track_id": ((capacity,), np.int64),
            "position_xy_m": ((capacity, 2), np.float32),
            "velocity_xy_mps": ((capacity, 2), np.float32),
            "speed_mps": ((capacity,), np.float32),
            "ttc_s": ((capacity,), np.float32),
            "age_since_seen_s": ((capacity,), np.float32),
            "hit_count": ((capacity,), np.int32),
        }
        for name, (shape, dtype) in expected.items():
            array = getattr(self, name)
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")


def extract_components(
    grid: DepthBEVGrid,
    geometry: BEVGeometry | None = None,
    config: TrackerConfig | None = None,
) -> ComponentBatch:
    """Extract deterministic 4/8-connected hit components."""

    grid.validate()
    geom = geometry or BEVGeometry()
    cfg = config or TrackerConfig()
    output = ComponentBatch.empty(cfg.max_components)
    occupied = grid.hit
    visited = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.bool_)
    if cfg.connectivity == 4:
        neighbours = ((-1, 0), (0, -1), (0, 1), (1, 0))
    else:
        neighbours = tuple(
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if dr != 0 or dc != 0
        )

    found: list[tuple[int, np.ndarray]] = []
    for start_row, start_col in np.argwhere(occupied):
        row = int(start_row)
        col = int(start_col)
        if visited[row, col]:
            continue
        stack = [(row, col)]
        visited[row, col] = True
        cells: list[tuple[int, int]] = []
        while stack:
            current_row, current_col = stack.pop()
            cells.append((current_row, current_col))
            for delta_row, delta_col in neighbours:
                next_row = current_row + delta_row
                next_col = current_col + delta_col
                if (
                    0 <= next_row < GRID_SIZE
                    and 0 <= next_col < GRID_SIZE
                    and occupied[next_row, next_col]
                    and not visited[next_row, next_col]
                ):
                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))
        if len(cells) >= cfg.minimum_component_cells:
            cell_array = np.asarray(cells, dtype=np.int32)
            found.append((len(cells), cell_array))

    found.sort(
        key=lambda item: (
            -item[0],
            int(item[1][:, 0].min()),
            int(item[1][:, 1].min()),
        )
    )
    for index, (_, cells) in enumerate(found[: cfg.max_components]):
        rows = cells[:, 0]
        cols = cells[:, 1]
        centres = cell_centres(rows, cols, geom)
        output.valid[index] = True
        output.centroid_xy_m[index] = centres.mean(axis=0)
        output.cell_count[index] = int(cells.shape[0])
        output.bbox_rc[index] = (
            int(rows.min()),
            int(cols.min()),
            int(rows.max()),
            int(cols.max()),
        )
        output.min_height_m[index] = float(
            grid.min_height_m[rows, cols].min()
        )
        output.max_height_m[index] = float(
            grid.max_height_m[rows, cols].max()
        )
    output.validate()
    return output


def radial_ttc_s(
    position_xy_m: np.ndarray,
    velocity_xy_mps: np.ndarray,
    *,
    safety_radius_m: float = 0.0,
    minimum_closing_speed_mps: float = 0.02,
) -> float:
    """Return radial time-to-contact, or infinity when not approaching."""

    position = np.asarray(position_xy_m, dtype=np.float64)
    velocity = np.asarray(velocity_xy_mps, dtype=np.float64)
    if position.shape != (2,) or velocity.shape != (2,):
        raise ValueError("position and velocity must be XY vectors")
    if not np.isfinite(position).all() or not np.isfinite(velocity).all():
        raise ValueError("position and velocity must be finite")
    if safety_radius_m < 0.0 or not isfinite(float(safety_radius_m)):
        raise ValueError("safety_radius_m must be finite and non-negative")
    if (
        minimum_closing_speed_mps <= 0.0
        or not isfinite(float(minimum_closing_speed_mps))
    ):
        raise ValueError("minimum_closing_speed_mps must be finite and positive")
    distance = float(np.linalg.norm(position))
    remaining = distance - float(safety_radius_m)
    if remaining <= 0.0:
        return 0.0
    if distance <= 1e-12:
        return 0.0
    closing_speed = -float(np.dot(position, velocity)) / distance
    if closing_speed < float(minimum_closing_speed_mps):
        return float("inf")
    return remaining / closing_speed


class NearestNeighbourTracker:
    """Greedy global-nearest association with fixed-capacity outputs."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.reset()

    def reset(self) -> None:
        capacity = self.config.max_tracks
        self._active = np.zeros(capacity, dtype=np.bool_)
        self._ids = np.full(capacity, -1, dtype=np.int64)
        self._positions = np.zeros((capacity, 2), dtype=np.float64)
        self._velocities = np.zeros((capacity, 2), dtype=np.float64)
        self._last_seen = np.zeros(capacity, dtype=np.float64)
        self._hits = np.zeros(capacity, dtype=np.int32)
        self._next_id = 1
        self._last_timestamp_s: float | None = None

    def update(
        self,
        detections: ComponentBatch,
        timestamp_s: float,
    ) -> TrackBatch:
        detections.validate()
        timestamp = float(timestamp_s)
        if not isfinite(timestamp):
            raise ValueError("timestamp_s must be finite")
        if (
            self._last_timestamp_s is not None
            and timestamp < self._last_timestamp_s
        ):
            raise ValueError("timestamp_s must be monotonic")
        self._last_timestamp_s = timestamp

        detection_indices = np.flatnonzero(detections.valid)
        track_indices = np.flatnonzero(self._active)
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        candidates: list[tuple[float, int, int]] = []
        for track_index in track_indices:
            deltas = (
                detections.centroid_xy_m[detection_indices]
                - self._positions[track_index]
            )
            distances = np.linalg.norm(deltas, axis=1)
            for local_index, distance in enumerate(distances):
                if distance <= self.config.association_distance_m:
                    candidates.append(
                        (
                            float(distance),
                            int(track_index),
                            int(detection_indices[local_index]),
                        )
                    )
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        for _, track_index, detection_index in candidates:
            if (
                track_index in matched_tracks
                or detection_index in matched_detections
            ):
                continue
            dt_s = timestamp - self._last_seen[track_index]
            new_position = detections.centroid_xy_m[detection_index].astype(
                np.float64
            )
            if dt_s > 1e-9:
                measured_velocity = (
                    new_position - self._positions[track_index]
                ) / dt_s
                alpha = self.config.velocity_alpha
                self._velocities[track_index] = (
                    alpha * measured_velocity
                    + (1.0 - alpha) * self._velocities[track_index]
                )
            self._positions[track_index] = new_position
            self._last_seen[track_index] = timestamp
            self._hits[track_index] += 1
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)

        expired = self._active & (
            (timestamp - self._last_seen) > self.config.maximum_missed_s
        )
        self._active[expired] = False
        self._ids[expired] = -1
        self._velocities[expired] = 0.0
        self._hits[expired] = 0

        for detection_index in detection_indices:
            detection_id = int(detection_index)
            if detection_id in matched_detections:
                continue
            free_slots = np.flatnonzero(~self._active)
            if free_slots.size == 0:
                break
            slot = int(free_slots[0])
            self._active[slot] = True
            self._ids[slot] = self._next_id
            self._next_id += 1
            self._positions[slot] = detections.centroid_xy_m[detection_id]
            self._velocities[slot] = 0.0
            self._last_seen[slot] = timestamp
            self._hits[slot] = 1
            matched_tracks.add(slot)
            matched_detections.add(detection_id)

        output = TrackBatch.empty(self.config.max_tracks)
        active_indices = np.flatnonzero(self._active)
        ordered = sorted(active_indices, key=lambda index: int(self._ids[index]))
        for output_index, slot in enumerate(ordered):
            output.valid[output_index] = True
            output.observed[output_index] = slot in matched_tracks
            output.track_id[output_index] = self._ids[slot]
            output.position_xy_m[output_index] = self._positions[slot]
            output.velocity_xy_mps[output_index] = self._velocities[slot]
            output.speed_mps[output_index] = float(
                np.linalg.norm(self._velocities[slot])
            )
            output.ttc_s[output_index] = radial_ttc_s(
                self._positions[slot],
                self._velocities[slot],
                safety_radius_m=self.config.safety_radius_m,
                minimum_closing_speed_mps=(
                    self.config.minimum_closing_speed_mps
                ),
            )
            output.age_since_seen_s[output_index] = max(
                0.0,
                timestamp - self._last_seen[slot],
            )
            output.hit_count[output_index] = self._hits[slot]
        output.validate()
        return output
