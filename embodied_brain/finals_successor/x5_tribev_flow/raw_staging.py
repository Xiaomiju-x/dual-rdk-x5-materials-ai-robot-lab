#!/usr/bin/env python3
"""Atomic raw staging and delayed pseudo-label promotion for real TriBEV data.

This module is deliberately independent from the validated finals runtime. It
stores raw numeric ROS payloads plus derived sensor grids, never camera images
or control commands. Raw observations are immutable. A strict training episode
is written only after real observations near all three future horizons exist.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import BEVGeometry, OdometryDelta
from .dataset import (
    FUTURE_HORIZON_SECONDS,
    FUTURE_HORIZONS,
    GRID_HEIGHT,
    GRID_WIDTH,
    HISTORY_FRAMES,
    SENSOR_NAMES,
    TRAJECTORY_TOKEN_OMEGA_RAD_S,
    build_episode_metadata,
    save_episode,
)
from .tribev import warp_bev_nearest

RAW_SCHEMA_VERSION = "x5-tribev-raw-frame.v1"
RAW_LABEL_PROVENANCE = "pseudo"
RAW_SOURCE_KIND = "real"
_NS_PER_SECOND = 1_000_000_000
_FUSED_CHANNEL = 7


class RawStagingError(ValueError):
    """Raised when an immutable raw observation violates the staging contract."""


@dataclass(frozen=True, slots=True)
class RawObservation:
    """One reference-time TriBEV history and its real sensor provenance."""

    timestamp_ns: int
    history_timestamps_ns: np.ndarray
    tribev_history: np.ndarray
    sensor_validity: np.ndarray
    sensor_age_s: np.ndarray
    sensor_provenance: np.ndarray
    vision_image_supplied: np.ndarray
    pose_xyyaw: np.ndarray
    ros_topics: tuple[str, ...] = ("/scan", "/scan_depth", "/odom")
    lidar_ranges: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    lidar_scan_geometry: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    depth_ranges: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    depth_scan_geometry: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    semantic_grid: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int16))
    semantic_grid_geometry: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp_ns, int) or self.timestamp_ns <= 0:
            raise RawStagingError("timestamp_ns must be a positive integer")
        expected = {
            "history_timestamps_ns": ((HISTORY_FRAMES,), np.dtype("int64")),
            "tribev_history": (
                (HISTORY_FRAMES, 8, GRID_HEIGHT, GRID_WIDTH),
                np.dtype("float32"),
            ),
            "sensor_validity": (
                (HISTORY_FRAMES, len(SENSOR_NAMES)),
                np.dtype("uint8"),
            ),
            "sensor_age_s": (
                (HISTORY_FRAMES, len(SENSOR_NAMES)),
                np.dtype("float32"),
            ),
            "vision_image_supplied": (
                (HISTORY_FRAMES,),
                np.dtype("uint8"),
            ),
            "pose_xyyaw": ((3,), np.dtype("float64")),
        }
        for name, (shape, dtype) in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape or value.dtype != dtype:
                raise RawStagingError(
                    f"{name} must have shape={shape}, dtype={dtype}; "
                    f"received shape={value.shape}, dtype={value.dtype}"
                )
            if value.dtype.kind in {"f", "c"} and not np.isfinite(value).all():
                raise RawStagingError(f"{name} contains NaN or Inf")
        provenance = np.asarray(self.sensor_provenance)
        if provenance.shape != (HISTORY_FRAMES, len(SENSOR_NAMES)):
            raise RawStagingError("sensor_provenance must have shape (5, 3)")
        if provenance.dtype.kind != "U":
            raise RawStagingError("sensor_provenance must use Unicode dtype")
        if np.any(np.diff(self.history_timestamps_ns) <= 0):
            raise RawStagingError("history timestamps must be strictly increasing")
        if int(self.history_timestamps_ns[-1]) != self.timestamp_ns:
            raise RawStagingError("the last history timestamp must be timestamp_ns")
        if not np.isin(self.sensor_validity, (0, 1)).all():
            raise RawStagingError("sensor_validity must contain only 0/1")
        if not np.isin(self.vision_image_supplied, (0, 1)).all():
            raise RawStagingError("vision_image_supplied must contain only 0/1")
        if np.any((self.sensor_validity == 0) & (self.sensor_age_s != -1.0)):
            raise RawStagingError("invalid sensors must use age -1")
        if float(self.tribev_history.min()) < -1e-6:
            raise RawStagingError("tribev_history must be non-negative")
        if float(self.tribev_history.max()) > 1.0 + 1e-6:
            raise RawStagingError("tribev_history must remain in [0, 1]")
        if not self.ros_topics or any(
            not isinstance(topic, str) or not topic.startswith("/") for topic in self.ros_topics
        ):
            raise RawStagingError("ros_topics must contain absolute ROS topic names")
        for name in ("lidar_ranges", "depth_ranges"):
            value = np.asarray(getattr(self, name))
            if value.ndim != 1 or value.dtype != np.dtype("float32"):
                raise RawStagingError(f"{name} must be a 1D float32 array")
        for ranges_name, geometry_name in (
            ("lidar_ranges", "lidar_scan_geometry"),
            ("depth_ranges", "depth_scan_geometry"),
        ):
            ranges = np.asarray(getattr(self, ranges_name))
            geometry = np.asarray(getattr(self, geometry_name))
            expected_shape = (4,) if ranges.size else (0,)
            if geometry.shape != expected_shape or geometry.dtype != np.dtype("float64"):
                raise RawStagingError(f"{geometry_name} must have shape {expected_shape} and dtype float64")
            if geometry.size and not np.isfinite(geometry).all():
                raise RawStagingError(f"{geometry_name} contains NaN or Inf")
        semantic = np.asarray(self.semantic_grid)
        semantic_geometry = np.asarray(self.semantic_grid_geometry)
        if semantic.ndim != 1 or semantic.dtype != np.dtype("int16"):
            raise RawStagingError("semantic_grid must be a 1D int16 array")
        expected_semantic_geometry = (6,) if semantic.size else (0,)
        if semantic_geometry.shape != expected_semantic_geometry or semantic_geometry.dtype != np.dtype(
            "float64"
        ):
            raise RawStagingError("semantic_grid_geometry must be empty or a 6D float64 array")
        if semantic_geometry.size:
            if not np.isfinite(semantic_geometry).all():
                raise RawStagingError("semantic_grid_geometry contains NaN or Inf")
            width, height = map(int, semantic_geometry[:2])
            if width <= 0 or height <= 0 or width * height != semantic.size:
                raise RawStagingError("semantic grid dimensions disagree with its payload")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(
                dict(payload),
                temporary,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return path


def _raw_metadata(
    observation: RawObservation,
    *,
    session_id: str,
    scenario_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": RAW_SCHEMA_VERSION,
        "session_id": session_id,
        "scenario_id": scenario_id,
        "timestamp_ns": observation.timestamp_ns,
        "source": {
            "kind": RAW_SOURCE_KIND,
            "contains_personal_data": False,
            "stored_payload": "raw_ros_numeric_payloads_plus_derived_bev_and_pose",
            "ros_topics": list(observation.ros_topics),
            "raw_camera_images_stored": False,
            "raw_audio_stored": False,
        },
        "labels": {
            "kind": RAW_LABEL_PROVENANCE,
            "state": "pending_future_observations",
            "required_horizons_s": list(FUTURE_HORIZON_SECONDS),
        },
        "authority": {
            "read_only_sidecar": True,
            "publishers": 0,
            "services": 0,
            "actions": 0,
            "tf_writes": 0,
            "control_writes": 0,
        },
    }


def save_raw_observation(
    path: str | Path,
    observation: RawObservation,
    *,
    session_id: str,
    scenario_id: str,
) -> Path:
    """Atomically write one immutable raw observation."""

    destination = Path(path).expanduser().resolve()
    if destination.suffix.lower() != ".npz":
        raise RawStagingError("raw observation path must end in .npz")
    if destination.exists():
        raise FileExistsError(destination)
    metadata = _raw_metadata(
        observation,
        session_id=session_id,
        scenario_id=scenario_id,
    )
    payload = {
        "raw_schema_version": np.asarray(RAW_SCHEMA_VERSION, dtype=np.str_),
        "timestamp_ns": np.asarray(observation.timestamp_ns, dtype=np.int64),
        "history_timestamps_ns": observation.history_timestamps_ns,
        "tribev_history": observation.tribev_history,
        "sensor_validity": observation.sensor_validity,
        "sensor_age_s": observation.sensor_age_s,
        "sensor_provenance": observation.sensor_provenance,
        "vision_image_supplied": observation.vision_image_supplied,
        "pose_xyyaw": observation.pose_xyyaw,
        "lidar_ranges": observation.lidar_ranges,
        "lidar_scan_geometry": observation.lidar_scan_geometry,
        "depth_ranges": observation.depth_ranges,
        "depth_scan_geometry": observation.depth_scan_geometry,
        "semantic_grid": observation.semantic_grid,
        "semantic_grid_geometry": observation.semantic_grid_geometry,
        "metadata_json": np.asarray(
            json.dumps(
                metadata,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            dtype=np.str_,
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{destination.stem}.",
            suffix=".npz.tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            np.savez_compressed(temporary, **payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


def load_raw_observation(path: str | Path) -> tuple[RawObservation, dict[str, Any]]:
    """Load a raw observation without pickle and revalidate its contract."""

    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "raw_schema_version",
            "timestamp_ns",
            "history_timestamps_ns",
            "tribev_history",
            "sensor_validity",
            "sensor_age_s",
            "sensor_provenance",
            "vision_image_supplied",
            "pose_xyyaw",
            "lidar_ranges",
            "lidar_scan_geometry",
            "depth_ranges",
            "depth_scan_geometry",
            "semantic_grid",
            "semantic_grid_geometry",
            "metadata_json",
        }
        missing = required - set(archive.files)
        extras = set(archive.files) - required
        if missing or extras:
            raise RawStagingError(
                f"raw NPZ member mismatch: missing={sorted(missing)}, extras={sorted(extras)}"
            )
        version = str(np.asarray(archive["raw_schema_version"]).item())
        if version != RAW_SCHEMA_VERSION:
            raise RawStagingError(f"unsupported raw schema: {version}")
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
        observation = RawObservation(
            timestamp_ns=int(np.asarray(archive["timestamp_ns"]).item()),
            history_timestamps_ns=np.array(archive["history_timestamps_ns"], dtype=np.int64, copy=True),
            tribev_history=np.array(archive["tribev_history"], dtype=np.float32, copy=True),
            sensor_validity=np.array(archive["sensor_validity"], dtype=np.uint8, copy=True),
            sensor_age_s=np.array(archive["sensor_age_s"], dtype=np.float32, copy=True),
            sensor_provenance=np.array(archive["sensor_provenance"], dtype=np.str_, copy=True),
            vision_image_supplied=np.array(archive["vision_image_supplied"], dtype=np.uint8, copy=True),
            pose_xyyaw=np.array(archive["pose_xyyaw"], dtype=np.float64, copy=True),
            ros_topics=tuple(metadata["source"]["ros_topics"]),
            lidar_ranges=np.array(archive["lidar_ranges"], dtype=np.float32, copy=True),
            lidar_scan_geometry=np.array(archive["lidar_scan_geometry"], dtype=np.float64, copy=True),
            depth_ranges=np.array(archive["depth_ranges"], dtype=np.float32, copy=True),
            depth_scan_geometry=np.array(archive["depth_scan_geometry"], dtype=np.float64, copy=True),
            semantic_grid=np.array(archive["semantic_grid"], dtype=np.int16, copy=True),
            semantic_grid_geometry=np.array(archive["semantic_grid_geometry"], dtype=np.float64, copy=True),
        )
    if metadata.get("source", {}).get("kind") != RAW_SOURCE_KIND:
        raise RawStagingError("raw source must be real")
    if metadata.get("labels", {}).get("kind") != RAW_LABEL_PROVENANCE:
        raise RawStagingError("raw labels must be explicitly pseudo")
    return observation, metadata


def _relative_odometry(source_pose: np.ndarray, destination_pose: np.ndarray) -> OdometryDelta:
    """Return source-to-destination motion expressed in the source frame."""

    source_x, source_y, source_yaw = map(float, source_pose)
    destination_x, destination_y, destination_yaw = map(float, destination_pose)
    world_dx = destination_x - source_x
    world_dy = destination_y - source_y
    cosine = math.cos(source_yaw)
    sine = math.sin(source_yaw)
    return OdometryDelta(
        dx_m=cosine * world_dx + sine * world_dy,
        dy_m=-sine * world_dx + cosine * world_dy,
        dyaw_rad=math.atan2(
            math.sin(destination_yaw - source_yaw),
            math.cos(destination_yaw - source_yaw),
        ),
        dt_s=0.0,
    )


def _pseudo_dynamic_and_flow(
    reference_occupancy: np.ndarray,
    future_occupancy: np.ndarray,
    *,
    resolution_m: float = 0.1,
    maximum_match_m: float = 1.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Build conservative nearest-cell pseudo motion labels.

    Only fully occupied future cells with no static overlap are considered.
    Labels are intentionally conservative and remain explicitly pseudo.
    """

    dynamic = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    flow = np.zeros((2, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    reference_cells = np.argwhere(reference_occupancy >= 0.999)
    future_cells = np.argwhere(future_occupancy >= 0.999)
    if reference_cells.size == 0 or future_cells.size == 0:
        return dynamic, flow
    maximum_cells = maximum_match_m / resolution_m
    for start in range(0, len(future_cells), 256):
        candidates = future_cells[start : start + 256]
        differences = candidates[:, np.newaxis, :].astype(np.float32) - reference_cells[
            np.newaxis, :, :
        ].astype(np.float32)
        squared = np.sum(differences * differences, axis=2)
        nearest_indices = np.argmin(squared, axis=1)
        nearest_distance = np.sqrt(squared[np.arange(len(candidates)), nearest_indices])
        moving = (nearest_distance >= 1.5) & (nearest_distance <= maximum_cells)
        for candidate, nearest_index, is_moving in zip(
            candidates,
            nearest_indices,
            moving,
            strict=True,
        ):
            if not bool(is_moving):
                continue
            row, column = map(int, candidate)
            reference_row, reference_column = map(int, reference_cells[int(nearest_index)])
            dynamic[row, column] = 1.0
            flow[0, row, column] = (row - reference_row) * resolution_m
            flow[1, row, column] = (column - reference_column) * resolution_m
    return dynamic, flow


def _trajectory_soft_labels(future_occupancy: np.ndarray) -> np.ndarray:
    geometry = BEVGeometry()
    omegas = np.asarray(TRAJECTORY_TOKEN_OMEGA_RAD_S, dtype=np.float64)
    horizons = np.asarray(FUTURE_HORIZON_SECONDS, dtype=np.float64)
    speed_m_s = 0.45
    radius_m = 0.34
    rows = np.arange(GRID_HEIGHT, dtype=np.float64)
    columns = np.arange(GRID_WIDTH, dtype=np.float64)
    x = geometry.x_min_m + (rows + 0.5) * geometry.resolution_m
    y = geometry.y_min_m + (columns + 0.5) * geometry.resolution_m
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    risks = np.zeros(len(omegas), dtype=np.float64)
    for token_index, omega in enumerate(omegas):
        token_risk = 0.0
        for horizon_index, horizon_s in enumerate(horizons):
            if abs(omega) < 1e-8:
                center_x = speed_m_s * horizon_s
                center_y = 0.0
            else:
                center_x = speed_m_s * math.sin(omega * horizon_s) / omega
                center_y = speed_m_s * (1.0 - math.cos(omega * horizon_s)) / omega
            footprint = ((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) <= radius_m**2
            if np.any(footprint):
                token_risk = max(
                    token_risk,
                    float(np.max(future_occupancy[horizon_index][footprint])),
                )
        risks[token_index] = token_risk
    logits = -risks / 0.20
    logits -= float(np.max(logits))
    probabilities = np.exp(logits)
    probabilities /= float(np.sum(probabilities))
    return probabilities.astype(np.float32)


def build_pseudo_episode(
    reference: RawObservation,
    future_observations: Sequence[RawObservation],
    *,
    session_id: str,
    scenario_id: str,
    raw_session_uri: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Create a strict episode from real inputs and delayed pseudo labels."""

    if len(future_observations) != FUTURE_HORIZONS:
        raise RawStagingError("all three future observations are required")
    future_occupancy = np.zeros((FUTURE_HORIZONS, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    future_flow = np.zeros((FUTURE_HORIZONS, 2, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    dynamic_mask = np.zeros_like(future_occupancy)
    uncertainty = np.zeros_like(future_occupancy)
    reference_occupancy = reference.tribev_history[-1, _FUSED_CHANNEL]
    actual_timestamps_ns: list[int] = []
    offset_errors_s: list[float] = []
    for index, (horizon_s, future) in enumerate(
        zip(FUTURE_HORIZON_SECONDS, future_observations, strict=True)
    ):
        transform = _relative_odometry(future.pose_xyyaw, reference.pose_xyyaw)
        occupancy = warp_bev_nearest(
            future.tribev_history[-1, _FUSED_CHANNEL],
            transform,
            BEVGeometry(),
        ).astype(np.float32)
        future_occupancy[index] = np.clip(occupancy, 0.0, 1.0)
        dynamic, flow = _pseudo_dynamic_and_flow(
            reference_occupancy,
            future_occupancy[index],
        )
        dynamic_mask[index] = dynamic
        future_flow[index] = flow
        valid_fraction = float(np.mean(future.sensor_validity[-1]))
        uncertainty[index] = np.clip(
            (1.0 - valid_fraction) + 0.25 * np.abs(future_occupancy[index] - reference_occupancy),
            0.0,
            1.0,
        )
        actual_timestamps_ns.append(future.timestamp_ns)
        actual_horizon_s = (future.timestamp_ns - reference.timestamp_ns) / _NS_PER_SECOND
        offset_errors_s.append(actual_horizon_s - horizon_s)

    episode_id = f"real-pseudo-{session_id}-{reference.timestamp_ns}"
    ideal_future_timestamps = np.asarray(
        [reference.timestamp_ns + int(round(value * _NS_PER_SECOND)) for value in FUTURE_HORIZON_SECONDS],
        dtype=np.int64,
    )
    metadata = build_episode_metadata(
        episode_id=episode_id,
        session_id=session_id,
        scenario_id=scenario_id,
        source={
            "kind": RAW_SOURCE_KIND,
            "license_id": "project-private-real-sensor-data",
            "contains_personal_data": False,
            "consent_status": "not_applicable",
            "raw_session_uri": raw_session_uri,
            "ros_topics": list(reference.ros_topics),
            "privacy_controls": [
                "private_storage",
                "retention_limit",
                "no_public_release",
            ],
        },
        generator={
            "name": "x5-tribev-readonly-collector",
            "version": "1",
            "labeler_version": "future-observation-pseudo-v1",
            "label_provenance": RAW_LABEL_PROVENANCE,
            "actual_future_timestamps_ns": actual_timestamps_ns,
            "future_offset_errors_s": offset_errors_s,
            "promotion_rule": "all_t+0.4_0.8_1.2_observations_required",
        },
        notes=(
            "Input source is real derived sensor data.",
            "All future occupancy, flow, dynamic, uncertainty and trajectory labels "
            "are pseudo labels derived from later observations.",
            "No raw image or audio payload is stored.",
        ),
    )
    arrays = {
        "timestamps_ns": reference.history_timestamps_ns.astype(np.int64),
        "history_offsets_s": (
            (reference.history_timestamps_ns.astype(np.float64) - float(reference.timestamp_ns))
            / _NS_PER_SECOND
        ).astype(np.float32),
        "future_timestamps_ns": ideal_future_timestamps,
        "future_horizons_s": np.asarray(FUTURE_HORIZON_SECONDS, dtype=np.float32),
        "tribev_input": reference.tribev_history.astype(np.float32),
        "future_occupancy": future_occupancy,
        "future_flow_m": future_flow,
        "dynamic_mask": dynamic_mask,
        "uncertainty_target": uncertainty,
        "trajectory_soft_labels": _trajectory_soft_labels(future_occupancy),
        "trajectory_token_omega_rad_s": np.asarray(TRAJECTORY_TOKEN_OMEGA_RAD_S, dtype=np.float32),
        "sensor_validity": reference.sensor_validity.astype(np.uint8),
        "sensor_age_s": reference.sensor_age_s.astype(np.float32),
        "sensor_provenance": reference.sensor_provenance.astype(np.str_),
        "vision_image_supplied": reference.vision_image_supplied.astype(np.uint8),
    }
    return arrays, metadata


class RawStagingStore:
    """Append-only session store with delayed, all-horizons promotion."""

    def __init__(
        self,
        root: str | Path,
        *,
        session_id: str,
        scenario_id: str = "real_unlabeled_navigation",
        anchor_stride_s: float = 1.0,
        future_tolerance_s: float = 0.12,
        memory_window_s: float = 8.0,
    ) -> None:
        if not session_id or any(char.isspace() for char in session_id):
            raise RawStagingError("session_id must be non-empty without whitespace")
        if not scenario_id or any(char.isspace() for char in scenario_id):
            raise RawStagingError("scenario_id must be non-empty without whitespace")
        if anchor_stride_s <= 0.0 or future_tolerance_s <= 0.0:
            raise RawStagingError("stride and tolerance must be positive")
        if memory_window_s < max(FUTURE_HORIZON_SECONDS) + future_tolerance_s:
            raise RawStagingError("memory_window_s is too short")
        self.root = Path(root).expanduser().resolve()
        self.session_id = session_id
        self.scenario_id = scenario_id
        self.anchor_stride_ns = int(round(anchor_stride_s * _NS_PER_SECOND))
        self.future_tolerance_ns = int(round(future_tolerance_s * _NS_PER_SECOND))
        self.memory_window_ns = int(round(memory_window_s * _NS_PER_SECOND))
        self.session_dir = self.root / "sessions" / session_id
        self.raw_dir = self.session_dir / "raw"
        self.receipt_dir = self.session_dir / "receipts"
        self.strict_dir = self.root / "strict_npz" / session_id
        self._observations: list[RawObservation] = []
        self._pending: list[int] = []
        self._last_anchor_ns: int | None = None
        self._promoted: set[int] = set()
        self._raw_only: set[int] = set()
        self._lock = threading.Lock()

    @property
    def manifest_path(self) -> Path:
        return self.session_dir / "session_manifest.json"

    def _write_manifest(self) -> None:
        _atomic_write_json(
            self.manifest_path,
            {
                "schema_version": "x5-tribev-raw-session.v1",
                "session_id": self.session_id,
                "scenario_id": self.scenario_id,
                "source": RAW_SOURCE_KIND,
                "labels": RAW_LABEL_PROVENANCE,
                "raw_frames": len(self._observations),
                "pending_anchors": len(self._pending),
                "promoted_anchors": len(self._promoted),
                "raw_only_anchors": len(self._raw_only),
                "future_horizons_s": list(FUTURE_HORIZON_SECONDS),
                "immutable_raw": True,
            },
        )

    def _nearest_future(
        self,
        target_ns: int,
    ) -> RawObservation | None:
        candidates = [
            item
            for item in self._observations
            if abs(item.timestamp_ns - target_ns) <= self.future_tolerance_ns
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: abs(item.timestamp_ns - target_ns))

    def _reference(self, timestamp_ns: int) -> RawObservation:
        for item in self._observations:
            if item.timestamp_ns == timestamp_ns:
                return item
        raise RawStagingError(f"reference observation {timestamp_ns} is unavailable")

    def _expire_old_memory(self, latest_ns: int) -> None:
        keep_after = latest_ns - self.memory_window_ns
        protected = set(self._pending)
        self._observations = [
            item
            for item in self._observations
            if item.timestamp_ns >= keep_after or item.timestamp_ns in protected
        ]

    def _promote_ready(self, latest_ns: int) -> list[Path]:
        promoted_paths: list[Path] = []
        still_pending: list[int] = []
        maximum_horizon_ns = int(round(max(FUTURE_HORIZON_SECONDS) * _NS_PER_SECOND))
        for anchor_ns in self._pending:
            targets = [anchor_ns + int(round(value * _NS_PER_SECOND)) for value in FUTURE_HORIZON_SECONDS]
            future = [self._nearest_future(target) for target in targets]
            if all(item is not None for item in future):
                reference = self._reference(anchor_ns)
                arrays, metadata = build_pseudo_episode(
                    reference,
                    [item for item in future if item is not None],
                    session_id=self.session_id,
                    scenario_id=self.scenario_id,
                    raw_session_uri=self.session_dir.as_uri(),
                )
                destination = self.strict_dir / (f"real-pseudo-{self.session_id}-{anchor_ns}.npz")
                save_episode(destination, arrays, metadata)
                receipt = {
                    "state": "PROMOTED_STRICT_NPZ",
                    "source": RAW_SOURCE_KIND,
                    "labels": RAW_LABEL_PROVENANCE,
                    "anchor_timestamp_ns": anchor_ns,
                    "future_source_timestamps_ns": [item.timestamp_ns for item in future if item is not None],
                    "strict_npz": str(destination),
                    "strict_npz_sha256": _sha256(destination),
                }
                _atomic_write_json(
                    self.receipt_dir / f"{anchor_ns}.promotion.json",
                    receipt,
                )
                self._promoted.add(anchor_ns)
                promoted_paths.append(destination)
                continue
            expiry_ns = anchor_ns + maximum_horizon_ns + self.future_tolerance_ns
            if latest_ns > expiry_ns:
                _atomic_write_json(
                    self.receipt_dir / f"{anchor_ns}.raw_only.json",
                    {
                        "state": "RAW_ONLY_INCOMPLETE_FUTURE_LABELS",
                        "source": RAW_SOURCE_KIND,
                        "labels": RAW_LABEL_PROVENANCE,
                        "anchor_timestamp_ns": anchor_ns,
                        "required_future_timestamps_ns": targets,
                        "available": [item is not None for item in future],
                        "strict_npz_written": False,
                    },
                )
                self._raw_only.add(anchor_ns)
                continue
            still_pending.append(anchor_ns)
        self._pending = still_pending
        return promoted_paths

    def ingest(self, observation: RawObservation) -> dict[str, Any]:
        """Persist one raw frame and attempt eligible delayed promotions."""

        with self._lock:
            if self._observations and (observation.timestamp_ns <= self._observations[-1].timestamp_ns):
                raise RawStagingError("raw observations must arrive in strict timestamp order")
            raw_path = self.raw_dir / f"{observation.timestamp_ns}.npz"
            save_raw_observation(
                raw_path,
                observation,
                session_id=self.session_id,
                scenario_id=self.scenario_id,
            )
            raw_sha = _sha256(raw_path)
            _atomic_write_json(
                self.receipt_dir / f"{observation.timestamp_ns}.raw.json",
                {
                    "state": "RAW_STAGED",
                    "source": RAW_SOURCE_KIND,
                    "labels": RAW_LABEL_PROVENANCE,
                    "timestamp_ns": observation.timestamp_ns,
                    "raw_path": str(raw_path),
                    "raw_sha256": raw_sha,
                    "strict_npz_written": False,
                },
            )
            self._observations.append(observation)
            if (
                self._last_anchor_ns is None
                or observation.timestamp_ns - self._last_anchor_ns >= self.anchor_stride_ns
            ):
                self._pending.append(observation.timestamp_ns)
                self._last_anchor_ns = observation.timestamp_ns
            promoted = self._promote_ready(observation.timestamp_ns)
            self._expire_old_memory(observation.timestamp_ns)
            self._write_manifest()
            return {
                "raw_path": raw_path,
                "raw_sha256": raw_sha,
                "promoted_paths": promoted,
                "pending_anchors": len(self._pending),
                "raw_only_anchors": len(self._raw_only),
            }


__all__ = [
    "RAW_LABEL_PROVENANCE",
    "RAW_SCHEMA_VERSION",
    "RAW_SOURCE_KIND",
    "RawObservation",
    "RawStagingError",
    "RawStagingStore",
    "build_pseudo_episode",
    "load_raw_observation",
    "save_raw_observation",
]
