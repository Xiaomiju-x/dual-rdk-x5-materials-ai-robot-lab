"""Typed contracts for the independent X5 TriBEV front end."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isclose, isfinite

import numpy as np


HISTORY_FRAMES = 5
CHANNELS_PER_FRAME = 8
MODEL_INPUT_SHAPE = (1, 40, 64, 64)
FRAME_CHANNEL_NAMES: tuple[str, ...] = (
    "lidar_occupancy",
    "lidar_visibility",
    "depth_near",
    "depth_mid",
    "depth_far",
    "camera_semantic_risk",
    "sensor_validity_fraction",
    "fused_occupancy",
)


class SemanticProvenance(str, Enum):
    """Origin of a semantic BEV observation."""

    UNAVAILABLE = "unavailable"
    LIVE_CAMERA = "live_camera"
    CACHED = "cached_camera"
    FIXTURE_PRIOR = "fixture_prior"


@dataclass(frozen=True, slots=True)
class BEVGeometry:
    """Metric BEV geometry with x forward, y left, and cell-center sampling."""

    height: int = 64
    width: int = 64
    resolution_m: float = 0.1
    x_min_m: float = -1.2
    y_min_m: float = -3.2

    def __post_init__(self) -> None:
        if (
            not isinstance(self.height, int)
            or isinstance(self.height, bool)
            or self.height <= 0
        ):
            raise ValueError("height must be a positive integer")
        if (
            not isinstance(self.width, int)
            or isinstance(self.width, bool)
            or self.width <= 0
        ):
            raise ValueError("width must be a positive integer")
        if not isfinite(self.resolution_m) or self.resolution_m <= 0.0:
            raise ValueError("resolution_m must be finite and positive")
        if not isfinite(self.x_min_m) or not isfinite(self.y_min_m):
            raise ValueError("BEV origins must be finite")
        if not self.x_min_m < 0.0 < self.x_max_m:
            raise ValueError("the BEV must cover both rear and forward x")
        if not self.y_min_m < 0.0 < self.y_max_m:
            raise ValueError("the BEV must cover both right and left y")

    @property
    def x_max_m(self) -> float:
        return self.x_min_m + self.height * self.resolution_m

    @property
    def y_max_m(self) -> float:
        return self.y_min_m + self.width * self.resolution_m

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def forward_extent_m(self) -> float:
        return self.x_max_m

    @property
    def rear_extent_m(self) -> float:
        return -self.x_min_m


@dataclass(frozen=True, slots=True)
class DepthRangeBands:
    """Planar range bands exported as near, mid, and far depth channels.

    Using planar range keeps the contract implementable from either a depth
    point cloud or the existing ``/scan_depth`` projection.
    """

    minimum_m: float = 0.05
    near_max_m: float = 1.50
    mid_max_m: float = 3.00
    far_max_m: float = 6.00

    def __post_init__(self) -> None:
        values = (
            self.minimum_m,
            self.near_max_m,
            self.mid_max_m,
            self.far_max_m,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("depth range band edges must be finite")
        if not (
            self.minimum_m
            < self.near_max_m
            < self.mid_max_m
            < self.far_max_m
        ):
            raise ValueError("depth range band edges must be strictly increasing")


@dataclass(frozen=True, slots=True)
class TriBEVConfig:
    """Static configuration for the fixed ``1x40x64x64`` model contract."""

    geometry: BEVGeometry = field(default_factory=BEVGeometry)
    depth_bands: DepthRangeBands = field(default_factory=DepthRangeBands)
    history_frames: int = HISTORY_FRAMES
    lidar_origin_xy_m: tuple[float, float] = (0.0, 0.0)
    lidar_ray_step_fraction: float = 0.5
    semantic_max_age_s: float = 2.0
    accepted_semantic_provenance: tuple[SemanticProvenance, ...] = (
        SemanticProvenance.LIVE_CAMERA,
        SemanticProvenance.CACHED,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, BEVGeometry):
            raise TypeError("geometry must be a BEVGeometry")
        if not isinstance(self.depth_bands, DepthRangeBands):
            raise TypeError("depth_bands must be a DepthRangeBands")
        if (
            self.geometry.height,
            self.geometry.width,
        ) != MODEL_INPUT_SHAPE[2:]:
            raise ValueError("the v1 model contract requires a 64x64 grid")
        if not isclose(self.geometry.resolution_m, 0.1, abs_tol=1e-12):
            raise ValueError("the v1 model contract requires 0.1 m cells")
        if not isclose(self.geometry.x_min_m, -1.2, abs_tol=1e-12):
            raise ValueError("the v1 model contract requires x_min_m=-1.2")
        if not isclose(self.geometry.y_min_m, -3.2, abs_tol=1e-12):
            raise ValueError("the v1 model contract requires y_min_m=-3.2")
        if self.history_frames != HISTORY_FRAMES:
            raise ValueError("history_frames must remain fixed at 5")
        if len(self.lidar_origin_xy_m) != 2 or not all(
            isfinite(value) for value in self.lidar_origin_xy_m
        ):
            raise ValueError("lidar_origin_xy_m must contain two finite values")
        if (
            not isfinite(self.lidar_ray_step_fraction)
            or not 0.0 < self.lidar_ray_step_fraction <= 1.0
        ):
            raise ValueError(
                "lidar_ray_step_fraction must be finite and in (0, 1]"
            )
        if not isfinite(self.semantic_max_age_s) or self.semantic_max_age_s <= 0.0:
            raise ValueError("semantic_max_age_s must be finite and positive")

        try:
            provenances = tuple(
                SemanticProvenance(value)
                for value in self.accepted_semantic_provenance
            )
        except ValueError as exc:
            raise ValueError("accepted_semantic_provenance is invalid") from exc
        if not provenances:
            raise ValueError("at least one semantic provenance must be accepted")
        if SemanticProvenance.UNAVAILABLE in provenances:
            raise ValueError("unavailable semantic data cannot be accepted")
        object.__setattr__(
            self,
            "accepted_semantic_provenance",
            provenances,
        )


@dataclass(frozen=True, slots=True)
class SemanticObservation:
    """A single metric semantic risk BEV and its truthful source metadata."""

    bev: np.ndarray | None = None
    provenance: SemanticProvenance = SemanticProvenance.UNAVAILABLE
    age_s: float = 0.0
    image_supplied: bool = False

    def __post_init__(self) -> None:
        try:
            provenance = SemanticProvenance(self.provenance)
        except ValueError as exc:
            raise ValueError(f"invalid semantic provenance: {self.provenance}") from exc
        object.__setattr__(self, "provenance", provenance)
        if not isfinite(self.age_s) or self.age_s < 0.0:
            raise ValueError("semantic age_s must be finite and non-negative")
        if not isinstance(self.image_supplied, (bool, np.bool_)):
            raise TypeError("image_supplied must be boolean")


@dataclass(frozen=True, slots=True)
class TriBEVObservation:
    """One synchronized observation in the current robot base frame."""

    lidar_points_xy: np.ndarray | None = None
    depth_points_xyz: np.ndarray | None = None
    semantic: SemanticObservation | None = None
    lidar_valid: bool | None = None
    depth_valid: bool | None = None
    timestamp_s: float | None = None

    def __post_init__(self) -> None:
        if self.semantic is not None and not isinstance(
            self.semantic,
            SemanticObservation,
        ):
            raise TypeError("semantic must be a SemanticObservation or None")
        for name, value in (
            ("lidar_valid", self.lidar_valid),
            ("depth_valid", self.depth_valid),
        ):
            if value is not None and not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be boolean or None")
        if self.timestamp_s is not None and not isfinite(self.timestamp_s):
            raise ValueError("timestamp_s must be finite when provided")


@dataclass(frozen=True, slots=True)
class OdometryDelta:
    """Robot motion from the previous frame to the current frame.

    ``dx_m`` and ``dy_m`` are expressed in the previous robot frame.
    Positive ``dyaw_rad`` is a counter-clockwise (left) turn.
    """

    dx_m: float = 0.0
    dy_m: float = 0.0
    dyaw_rad: float = 0.0
    dt_s: float = 0.0

    def __post_init__(self) -> None:
        if not all(
            isfinite(value)
            for value in (self.dx_m, self.dy_m, self.dyaw_rad, self.dt_s)
        ):
            raise ValueError("odometry delta values must be finite")
        if self.dt_s < 0.0:
            raise ValueError("odometry dt_s must be non-negative")


@dataclass(frozen=True, slots=True)
class LidarBEV:
    occupancy: np.ndarray
    visibility: np.ndarray
    validity: np.ndarray


@dataclass(frozen=True, slots=True)
class DepthBEV:
    near: np.ndarray
    mid: np.ndarray
    far: np.ndarray
    validity: np.ndarray


@dataclass(frozen=True, slots=True)
class SemanticBEV:
    risk: np.ndarray
    validity: np.ndarray
    provenance: SemanticProvenance
    age_s: float
    present: bool
    usable: bool
    image_supplied: bool


def history_channel_names(
    history_frames: int = HISTORY_FRAMES,
) -> tuple[str, ...]:
    """Return stable newest-to-oldest channel names for an NCHW tensor."""

    if history_frames != HISTORY_FRAMES:
        raise ValueError("history_frames must remain fixed at 5")
    names: list[str] = []
    for age in range(history_frames):
        prefix = "t0" if age == 0 else f"t_minus_{age}"
        names.extend(f"{prefix}.{name}" for name in FRAME_CHANNEL_NAMES)
    return tuple(names)


@dataclass(frozen=True, slots=True)
class TriBEVOutput:
    """Fixed NCHW float32 tensor plus non-model scalar metadata."""

    tensor: np.ndarray
    channel_names: tuple[str, ...]
    geometry: BEVGeometry
    populated_history: int
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        if self.tensor.dtype != np.float32:
            raise TypeError("TriBEVOutput.tensor must be float32")
        if self.tensor.shape != MODEL_INPUT_SHAPE:
            raise ValueError(
                f"TriBEVOutput.tensor must have shape {MODEL_INPUT_SHAPE}"
            )
        expected_names = history_channel_names(HISTORY_FRAMES)
        if self.channel_names != expected_names:
            raise ValueError("channel_names do not match the frozen v1 order")
        if not np.isfinite(self.tensor).all():
            raise ValueError("TriBEVOutput.tensor contains non-finite values")
        if (
            not isinstance(self.populated_history, int)
            or isinstance(self.populated_history, bool)
            or not 0 <= self.populated_history <= HISTORY_FRAMES
        ):
            raise ValueError("populated_history must be in [0, 5]")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict")
        required_metadata = {
            "contract_version",
            "layout",
            "tensor_shape",
            "frame_channel_names",
            "frames",
            "authority",
        }
        missing = required_metadata.difference(self.metadata)
        if missing:
            raise ValueError(f"metadata is missing keys: {sorted(missing)}")


__all__ = [
    "BEVGeometry",
    "CHANNELS_PER_FRAME",
    "DepthBEV",
    "DepthRangeBands",
    "FRAME_CHANNEL_NAMES",
    "HISTORY_FRAMES",
    "LidarBEV",
    "MODEL_INPUT_SHAPE",
    "OdometryDelta",
    "SemanticBEV",
    "SemanticObservation",
    "SemanticProvenance",
    "TriBEVConfig",
    "TriBEVObservation",
    "TriBEVOutput",
    "history_channel_names",
]
