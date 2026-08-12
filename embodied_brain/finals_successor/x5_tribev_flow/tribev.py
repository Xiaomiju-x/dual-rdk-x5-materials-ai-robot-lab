"""Pure NumPy LiDAR-depth-semantic BEV construction and history warping."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

import numpy as np

from .contracts import (
    BEVGeometry,
    CHANNELS_PER_FRAME,
    DepthBEV,
    FRAME_CHANNEL_NAMES,
    HISTORY_FRAMES,
    LidarBEV,
    MODEL_INPUT_SHAPE,
    OdometryDelta,
    SemanticBEV,
    SemanticObservation,
    SemanticProvenance,
    TriBEVConfig,
    TriBEVObservation,
    TriBEVOutput,
    history_channel_names,
)


_CHANNEL_INDEX = {name: index for index, name in enumerate(FRAME_CHANNEL_NAMES)}


def _empty_grid(geometry: BEVGeometry) -> np.ndarray:
    return np.zeros(geometry.shape, dtype=np.float32)


def _full_grid(geometry: BEVGeometry) -> np.ndarray:
    return np.ones(geometry.shape, dtype=np.float32)


def _as_points(
    points: np.ndarray | None,
    columns: int,
    name: str,
) -> np.ndarray:
    if points is None:
        return np.empty((0, columns), dtype=np.float64)
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, columns), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"{name} must have shape (N, {columns})")
    return np.ascontiguousarray(array)


def _resolve_valid(points: np.ndarray | None, explicit: bool | None) -> bool:
    return points is not None if explicit is None else bool(explicit)


def _metric_indices(
    x_m: np.ndarray,
    y_m: np.ndarray,
    geometry: BEVGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inside = (
        (x_m >= geometry.x_min_m)
        & (x_m < geometry.x_max_m)
        & (y_m >= geometry.y_min_m)
        & (y_m < geometry.y_max_m)
    )
    rows = np.full(x_m.shape, -1, dtype=np.int64)
    cols = np.full(y_m.shape, -1, dtype=np.int64)
    rows[inside] = np.floor(
        (x_m[inside] - geometry.x_min_m) / geometry.resolution_m
    ).astype(np.int64)
    cols[inside] = np.floor(
        (y_m[inside] - geometry.y_min_m) / geometry.resolution_m
    ).astype(np.int64)
    return rows, cols, inside


def _clip_segment_to_bev(
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    geometry: BEVGeometry,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Clip a metric segment to the half-open BEV rectangle."""

    direction = end_xy - start_xy
    lower = np.array(
        [geometry.x_min_m, geometry.y_min_m],
        dtype=np.float64,
    )
    upper = np.array(
        [
            np.nextafter(geometry.x_max_m, geometry.x_min_m),
            np.nextafter(geometry.y_max_m, geometry.y_min_m),
        ],
        dtype=np.float64,
    )
    enter = 0.0
    leave = 1.0
    for axis in range(2):
        delta = float(direction[axis])
        origin = float(start_xy[axis])
        if abs(delta) < 1e-12:
            if origin < lower[axis] or origin > upper[axis]:
                return None
            continue
        first = (lower[axis] - origin) / delta
        second = (upper[axis] - origin) / delta
        axis_enter = min(first, second)
        axis_leave = max(first, second)
        enter = max(enter, axis_enter)
        leave = min(leave, axis_leave)
        if enter > leave:
            return None
    return start_xy + enter * direction, start_xy + leave * direction


def rasterize_lidar(
    points_xy: np.ndarray | None,
    config: TriBEVConfig | None = None,
    *,
    valid: bool | None = None,
) -> LidarBEV:
    """Rasterize LiDAR endpoints and ray-observed cells into metric BEV grids."""

    cfg = config or TriBEVConfig()
    geometry = cfg.geometry
    occupancy = _empty_grid(geometry)
    visibility = _empty_grid(geometry)
    is_valid = _resolve_valid(points_xy, valid)
    validity = _full_grid(geometry) if is_valid else _empty_grid(geometry)
    if not is_valid:
        return LidarBEV(occupancy, visibility, validity)

    points = _as_points(points_xy, 2, "points_xy")
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] == 0:
        return LidarBEV(occupancy, visibility, validity)

    endpoint_rows, endpoint_cols, endpoints_inside = _metric_indices(
        points[:, 0],
        points[:, 1],
        geometry,
    )
    occupancy[
        endpoint_rows[endpoints_inside],
        endpoint_cols[endpoints_inside],
    ] = 1.0

    origin = np.asarray(cfg.lidar_origin_xy_m, dtype=np.float64)
    ray_step_m = geometry.resolution_m * cfg.lidar_ray_step_fraction
    for point in points:
        clipped = _clip_segment_to_bev(origin, point, geometry)
        if clipped is None:
            continue
        clipped_start, clipped_end = clipped
        ray_length_m = float(np.linalg.norm(clipped_end - clipped_start))
        sample_count = max(1, int(np.ceil(ray_length_m / ray_step_m)) + 1)
        samples = np.linspace(
            clipped_start,
            clipped_end,
            num=sample_count,
            dtype=np.float64,
        )
        rows, cols, inside = _metric_indices(
            samples[:, 0],
            samples[:, 1],
            geometry,
        )
        visibility[rows[inside], cols[inside]] = 1.0

    return LidarBEV(occupancy, visibility, validity)


def rasterize_depth(
    points_xyz: np.ndarray | None,
    config: TriBEVConfig | None = None,
    *,
    valid: bool | None = None,
) -> DepthBEV:
    """Rasterize base-frame depth XYZ points into planar range layers."""

    cfg = config or TriBEVConfig()
    geometry = cfg.geometry
    near = _empty_grid(geometry)
    mid = _empty_grid(geometry)
    far = _empty_grid(geometry)
    is_valid = _resolve_valid(points_xyz, valid)
    validity = _full_grid(geometry) if is_valid else _empty_grid(geometry)
    if not is_valid:
        return DepthBEV(near, mid, far, validity)

    points = _as_points(points_xyz, 3, "points_xyz")
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] == 0:
        return DepthBEV(near, mid, far, validity)

    rows, cols, inside = _metric_indices(
        points[:, 0],
        points[:, 1],
        geometry,
    )
    range_m = np.hypot(points[:, 0], points[:, 1])
    bands = cfg.depth_bands
    near_mask = (
        inside
        & (range_m >= bands.minimum_m)
        & (range_m < bands.near_max_m)
    )
    mid_mask = (
        inside
        & (range_m >= bands.near_max_m)
        & (range_m < bands.mid_max_m)
    )
    far_mask = (
        inside
        & (range_m >= bands.mid_max_m)
        & (range_m <= bands.far_max_m)
    )
    near[rows[near_mask], cols[near_mask]] = 1.0
    mid[rows[mid_mask], cols[mid_mask]] = 1.0
    far[rows[far_mask], cols[far_mask]] = 1.0
    return DepthBEV(near, mid, far, validity)


def prepare_semantic_bev(
    observation: SemanticObservation | None,
    config: TriBEVConfig | None = None,
) -> SemanticBEV:
    """Validate semantic risk, source, and age without inventing missing data."""

    cfg = config or TriBEVConfig()
    geometry = cfg.geometry
    zero = _empty_grid(geometry)
    if observation is None or observation.bev is None:
        return SemanticBEV(
            risk=zero.copy(),
            validity=zero.copy(),
            provenance=SemanticProvenance.UNAVAILABLE,
            age_s=0.0,
            present=False,
            usable=False,
            image_supplied=False,
        )

    semantic = np.asarray(observation.bev, dtype=np.float32)
    if semantic.shape != geometry.shape:
        raise ValueError(
            f"semantic BEV must have shape {geometry.shape}, got {semantic.shape}"
        )
    if not np.isfinite(semantic).all():
        raise ValueError("semantic BEV contains non-finite values")
    if float(semantic.min()) < -1e-6 or float(semantic.max()) > 1.0 + 1e-6:
        raise ValueError("semantic BEV values must be probabilities in [0, 1]")
    semantic = np.clip(semantic, 0.0, 1.0).astype(np.float32, copy=False)

    provenance = SemanticProvenance(observation.provenance)
    source_accepted = provenance in cfg.accepted_semantic_provenance
    source_truthful = (
        provenance != SemanticProvenance.LIVE_CAMERA
        or bool(observation.image_supplied)
    )
    age_valid = observation.age_s <= cfg.semantic_max_age_s
    usable = source_accepted and source_truthful and age_valid
    validity = _full_grid(geometry) if usable else _empty_grid(geometry)
    risk = semantic.copy() if usable else _empty_grid(geometry)

    return SemanticBEV(
        risk=risk,
        validity=validity,
        provenance=provenance,
        age_s=float(observation.age_s),
        present=True,
        usable=usable,
        image_supplied=bool(observation.image_supplied),
    )


def warp_bev_nearest(
    bev: np.ndarray,
    odometry: OdometryDelta,
    geometry: BEVGeometry | None = None,
) -> np.ndarray:
    """Warp an old-frame ``HW`` or ``CHW`` BEV into the current robot frame.

    The function inverse-maps current cell centers into the previous robot
    frame and samples the nearest old cell. Uncovered cells are zero.
    """

    geom = geometry or BEVGeometry()
    array = np.asarray(bev, dtype=np.float32)
    if array.ndim == 2:
        source = array[np.newaxis, ...]
        squeeze = True
    elif array.ndim == 3:
        source = array
        squeeze = False
    else:
        raise ValueError("bev must have shape (H, W) or (C, H, W)")
    if source.shape[1:] != geom.shape:
        raise ValueError(
            f"bev spatial shape must be {geom.shape}, got {source.shape[1:]}"
        )
    if not np.isfinite(source).all():
        raise ValueError("bev contains non-finite values")

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
    source_row_float = (
        source_x - (geom.x_min_m + 0.5 * geom.resolution_m)
    ) / geom.resolution_m
    source_col_float = (
        source_y - (geom.y_min_m + 0.5 * geom.resolution_m)
    ) / geom.resolution_m
    source_rows = np.floor(source_row_float + 0.5).astype(np.int64)
    source_cols = np.floor(source_col_float + 0.5).astype(np.int64)
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


@dataclass(slots=True)
class _HistoryEntry:
    channels: np.ndarray
    coverage: np.ndarray
    lidar_valid: bool
    depth_valid: bool
    semantic_provenance: SemanticProvenance
    semantic_age_s: float
    semantic_present: bool
    semantic_image_supplied: bool
    timestamp_s: float | None


class TriBEVFrontend:
    """Stateful five-frame TriBEV builder with newest-first NCHW output."""

    def __init__(self, config: TriBEVConfig | None = None) -> None:
        self.config = config or TriBEVConfig()
        self._entries: list[_HistoryEntry] = []
        self._channel_names = history_channel_names(self.config.history_frames)
        self._last_odometry = OdometryDelta()

    @property
    def channel_names(self) -> tuple[str, ...]:
        return self._channel_names

    @property
    def output_shape(self) -> tuple[int, int, int, int]:
        return (
            1,
            len(self._channel_names),
            self.config.geometry.height,
            self.config.geometry.width,
        )

    @property
    def populated_history(self) -> int:
        return len(self._entries)

    def reset(self) -> None:
        """Drop candidate history without touching any external state."""

        self._entries.clear()
        self._last_odometry = OdometryDelta()

    def _semantic_is_usable(
        self,
        provenance: SemanticProvenance,
        age_s: float,
        image_supplied: bool,
        present: bool,
    ) -> bool:
        return bool(
            present
            and provenance in self.config.accepted_semantic_provenance
            and (
                provenance != SemanticProvenance.LIVE_CAMERA
                or image_supplied
            )
            and age_s <= self.config.semantic_max_age_s
        )

    @staticmethod
    def _validity_fraction(
        lidar_valid: bool,
        depth_valid: bool,
        semantic_valid: bool,
    ) -> float:
        return (
            float(lidar_valid)
            + float(depth_valid)
            + float(semantic_valid)
        ) / 3.0

    def _build_entry(self, observation: TriBEVObservation) -> _HistoryEntry:
        lidar = rasterize_lidar(
            observation.lidar_points_xy,
            self.config,
            valid=observation.lidar_valid,
        )
        depth = rasterize_depth(
            observation.depth_points_xyz,
            self.config,
            valid=observation.depth_valid,
        )
        semantic = prepare_semantic_bev(observation.semantic, self.config)
        lidar_valid = bool(lidar.validity.any())
        depth_valid = bool(depth.validity.any())
        semantic_valid = semantic.usable
        depth_occupancy = np.maximum.reduce(
            (depth.near, depth.mid, depth.far)
        )
        fused_occupancy = np.maximum.reduce(
            (
                lidar.occupancy if lidar_valid else _empty_grid(self.config.geometry),
                depth_occupancy if depth_valid else _empty_grid(self.config.geometry),
                semantic.risk if semantic_valid else _empty_grid(self.config.geometry),
            )
        ).astype(np.float32, copy=False)
        validity_fraction = self._validity_fraction(
            lidar_valid,
            depth_valid,
            semantic_valid,
        )
        channels = np.stack(
            (
                lidar.occupancy,
                lidar.visibility,
                depth.near,
                depth.mid,
                depth.far,
                semantic.risk,
                _full_grid(self.config.geometry) * validity_fraction,
                fused_occupancy,
            ),
            axis=0,
        ).astype(np.float32, copy=False)
        return _HistoryEntry(
            channels=channels,
            coverage=_full_grid(self.config.geometry),
            lidar_valid=lidar_valid,
            depth_valid=depth_valid,
            semantic_provenance=semantic.provenance,
            semantic_age_s=semantic.age_s,
            semantic_present=semantic.present,
            semantic_image_supplied=semantic.image_supplied,
            timestamp_s=(
                float(observation.timestamp_s)
                if observation.timestamp_s is not None
                else None
            ),
        )

    def _warp_entry(
        self,
        entry: _HistoryEntry,
        odometry: OdometryDelta,
    ) -> _HistoryEntry:
        warped = warp_bev_nearest(
            entry.channels,
            odometry,
            self.config.geometry,
        )
        coverage = warp_bev_nearest(
            entry.coverage,
            odometry,
            self.config.geometry,
        )
        age_s = entry.semantic_age_s + odometry.dt_s
        semantic_valid = self._semantic_is_usable(
            entry.semantic_provenance,
            age_s,
            entry.semantic_image_supplied,
            entry.semantic_present,
        )
        if not semantic_valid:
            warped[_CHANNEL_INDEX["camera_semantic_risk"]].fill(0.0)
        validity_fraction = self._validity_fraction(
            entry.lidar_valid,
            entry.depth_valid,
            semantic_valid,
        )
        warped[_CHANNEL_INDEX["sensor_validity_fraction"]] = (
            coverage * validity_fraction
        )
        depth_occupancy = np.maximum.reduce(
            (
                warped[_CHANNEL_INDEX["depth_near"]],
                warped[_CHANNEL_INDEX["depth_mid"]],
                warped[_CHANNEL_INDEX["depth_far"]],
            )
        )
        warped[_CHANNEL_INDEX["fused_occupancy"]] = np.maximum.reduce(
            (
                warped[_CHANNEL_INDEX["lidar_occupancy"]]
                if entry.lidar_valid
                else _empty_grid(self.config.geometry),
                depth_occupancy
                if entry.depth_valid
                else _empty_grid(self.config.geometry),
                warped[_CHANNEL_INDEX["camera_semantic_risk"]]
                if semantic_valid
                else _empty_grid(self.config.geometry),
            )
        )
        return _HistoryEntry(
            channels=warped,
            coverage=coverage,
            lidar_valid=entry.lidar_valid,
            depth_valid=entry.depth_valid,
            semantic_provenance=entry.semantic_provenance,
            semantic_age_s=age_s,
            semantic_present=entry.semantic_present,
            semantic_image_supplied=entry.semantic_image_supplied,
            timestamp_s=entry.timestamp_s,
        )

    def _entry_metadata(
        self,
        entry: _HistoryEntry | None,
        age_index: int,
    ) -> dict[str, object]:
        if entry is None:
            return {
                "history_index": age_index,
                "frame_present": False,
                "timestamp_s": None,
                "lidar_valid": False,
                "depth_valid": False,
                "camera_semantic_present": False,
                "camera_semantic_valid": False,
                "camera_semantic_provenance": SemanticProvenance.UNAVAILABLE.value,
                "camera_semantic_age_s": None,
                "camera_image_supplied": False,
                "sensor_validity_fraction": 0.0,
            }
        semantic_valid = self._semantic_is_usable(
            entry.semantic_provenance,
            entry.semantic_age_s,
            entry.semantic_image_supplied,
            entry.semantic_present,
        )
        return {
            "history_index": age_index,
            "frame_present": True,
            "timestamp_s": entry.timestamp_s,
            "lidar_valid": entry.lidar_valid,
            "depth_valid": entry.depth_valid,
            "camera_semantic_present": entry.semantic_present,
            "camera_semantic_valid": semantic_valid,
            "camera_semantic_provenance": entry.semantic_provenance.value,
            "camera_semantic_age_s": (
                entry.semantic_age_s if entry.semantic_present else None
            ),
            "camera_image_supplied": entry.semantic_image_supplied,
            "sensor_validity_fraction": self._validity_fraction(
                entry.lidar_valid,
                entry.depth_valid,
                semantic_valid,
            ),
        }

    def _metadata(self) -> dict[str, object]:
        frames = [
            self._entry_metadata(
                self._entries[index] if index < len(self._entries) else None,
                index,
            )
            for index in range(HISTORY_FRAMES)
        ]
        geometry = self.config.geometry
        return {
            "contract_version": "x5-tribev-input.v1",
            "layout": "NCHW",
            "tensor_shape": list(MODEL_INPUT_SHAPE),
            "frame_channel_names": list(FRAME_CHANNEL_NAMES),
            "history_order": ["t0", "t_minus_1", "t_minus_2", "t_minus_3", "t_minus_4"],
            "coordinate_convention": {
                "row_axis": "+x_forward",
                "column_axis": "+y_left",
            },
            "grid": {
                "height": geometry.height,
                "width": geometry.width,
                "resolution_m": float(geometry.resolution_m),
                "x_min_m": float(geometry.x_min_m),
                "x_max_m": float(geometry.x_max_m),
                "y_min_m": float(geometry.y_min_m),
                "y_max_m": float(geometry.y_max_m),
            },
            "depth_channel_semantics": {
                "depth_near": "minimum_m <= hypot(x,y) < near_max_m",
                "depth_mid": "near_max_m <= hypot(x,y) < mid_max_m",
                "depth_far": "mid_max_m <= hypot(x,y) <= far_max_m",
            },
            "frames": frames,
            "last_odometry_delta": {
                "dx_m": float(self._last_odometry.dx_m),
                "dy_m": float(self._last_odometry.dy_m),
                "dyaw_rad": float(self._last_odometry.dyaw_rad),
                "dt_s": float(self._last_odometry.dt_s),
            },
            "sensor_validity_fraction": "(lidar+depth+camera)/3",
            "fusion": {
                "method": "reliability_gated_current_frame_max",
                "uses_future_labels": False,
            },
            "authority": {
                "shadow_only": True,
                "publishes_cmd_vel": False,
                "writes_f407": False,
                "publishes_authoritative_tf": False,
            },
        }

    def output(self) -> TriBEVOutput:
        """Return a zero-padded fixed NCHW tensor without changing history."""

        frame_shape = (
            CHANNELS_PER_FRAME,
            self.config.geometry.height,
            self.config.geometry.width,
        )
        frames = [entry.channels for entry in self._entries]
        frames.extend(
            np.zeros(frame_shape, dtype=np.float32)
            for _ in range(self.config.history_frames - len(frames))
        )
        tensor = np.concatenate(frames, axis=0)[np.newaxis, ...]
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        return TriBEVOutput(
            tensor=tensor,
            channel_names=self._channel_names,
            geometry=self.config.geometry,
            populated_history=len(self._entries),
            metadata=self._metadata(),
        )

    def update(
        self,
        observation: TriBEVObservation,
        odometry: OdometryDelta | None = None,
    ) -> TriBEVOutput:
        """Warp old history, prepend one observation, and return fixed NCHW."""

        if not isinstance(observation, TriBEVObservation):
            raise TypeError("observation must be a TriBEVObservation")
        delta = odometry or OdometryDelta()
        if not isinstance(delta, OdometryDelta):
            raise TypeError("odometry must be an OdometryDelta or None")
        self._last_odometry = delta
        warped_history = [
            self._warp_entry(entry, delta)
            for entry in self._entries
        ]
        self._entries = [
            self._build_entry(observation),
            *warped_history,
        ][: self.config.history_frames]
        return self.output()


__all__ = [
    "TriBEVFrontend",
    "prepare_semantic_bev",
    "rasterize_depth",
    "rasterize_lidar",
    "warp_bev_nearest",
]
