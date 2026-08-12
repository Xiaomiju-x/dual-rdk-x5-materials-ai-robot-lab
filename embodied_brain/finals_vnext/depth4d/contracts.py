"""Static data contracts for the NumPy-only Depth-4D observer."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

import numpy as np


GRID_SIZE = 64
DEFAULT_MAX_COMPONENTS = 16
DEFAULT_MAX_TRACKS = 16


def _finite_positive(value: float, name: str) -> float:
    value = float(value)
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _readonly_array(
    value: np.ndarray,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    array = np.array(array, dtype=np.float64, copy=True, order="C")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics and the scale used by uint16 depth images."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale_m: float = 0.001

    def __post_init__(self) -> None:
        if isinstance(self.width, bool) or int(self.width) != self.width:
            raise ValueError("width must be a positive integer")
        if isinstance(self.height, bool) or int(self.height) != self.height:
            raise ValueError("height must be a positive integer")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive integers")
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))
        object.__setattr__(self, "fx", _finite_positive(self.fx, "fx"))
        object.__setattr__(self, "fy", _finite_positive(self.fy, "fy"))
        object.__setattr__(
            self,
            "depth_scale_m",
            _finite_positive(self.depth_scale_m, "depth_scale_m"),
        )
        cx = float(self.cx)
        cy = float(self.cy)
        if not isfinite(cx) or not (0.0 <= cx < self.width):
            raise ValueError("cx must lie inside the calibrated image")
        if not isfinite(cy) or not (0.0 <= cy < self.height):
            raise ValueError("cy must lie inside the calibrated image")
        object.__setattr__(self, "cx", cx)
        object.__setattr__(self, "cy", cy)

    @property
    def image_shape(self) -> tuple[int, int]:
        return (self.height, self.width)


@dataclass(frozen=True)
class CameraToBase:
    """Rigid transform mapping camera-frame XYZ points into ``base``."""

    rotation: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float64)
    )
    translation_m: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        rotation = _readonly_array(self.rotation, (3, 3), "rotation")
        translation = _readonly_array(
            self.translation_m,
            (3,),
            "translation_m",
        )
        if not np.allclose(
            rotation.T @ rotation,
            np.eye(3),
            atol=1e-5,
            rtol=0.0,
        ):
            raise ValueError("rotation must be orthonormal")
        determinant = float(np.linalg.det(rotation))
        if not np.isclose(determinant, 1.0, atol=1e-5, rtol=0.0):
            raise ValueError("rotation must be a proper rotation (det=+1)")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation_m", translation)


@dataclass(frozen=True)
class ProjectionLimits:
    minimum_depth_m: float = 0.10
    maximum_depth_m: float = 8.0

    def __post_init__(self) -> None:
        minimum = _finite_positive(self.minimum_depth_m, "minimum_depth_m")
        maximum = _finite_positive(self.maximum_depth_m, "maximum_depth_m")
        if minimum >= maximum:
            raise ValueError("minimum_depth_m must be below maximum_depth_m")
        object.__setattr__(self, "minimum_depth_m", minimum)
        object.__setattr__(self, "maximum_depth_m", maximum)


@dataclass(frozen=True)
class HeightBands:
    """Obstacle-height bands in the base frame, in metres."""

    minimum_m: float = -0.10
    low_max_m: float = 0.25
    mid_max_m: float = 1.20
    high_max_m: float = 2.20

    def __post_init__(self) -> None:
        values = tuple(
            float(value)
            for value in (
                self.minimum_m,
                self.low_max_m,
                self.mid_max_m,
                self.high_max_m,
            )
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("height band limits must be finite")
        if not (values[0] < values[1] < values[2] < values[3]):
            raise ValueError("height band limits must be strictly increasing")
        object.__setattr__(self, "minimum_m", values[0])
        object.__setattr__(self, "low_max_m", values[1])
        object.__setattr__(self, "mid_max_m", values[2])
        object.__setattr__(self, "high_max_m", values[3])


@dataclass(frozen=True)
class BEVGeometry:
    """Fixed 64x64 metric raster in the robot base frame."""

    x_min_m: float = -1.20
    x_max_m: float = 5.20
    y_min_m: float = -3.20
    y_max_m: float = 3.20
    grid_size: int = GRID_SIZE
    ray_stride: int = 2
    maximum_rays: int = 8192
    ray_step_fraction: float = 0.45

    def __post_init__(self) -> None:
        if self.grid_size != GRID_SIZE:
            raise ValueError(f"grid_size is fixed at {GRID_SIZE}")
        limits = tuple(
            float(value)
            for value in (
                self.x_min_m,
                self.x_max_m,
                self.y_min_m,
                self.y_max_m,
            )
        )
        if not all(isfinite(value) for value in limits):
            raise ValueError("BEV limits must be finite")
        if limits[0] >= limits[1] or limits[2] >= limits[3]:
            raise ValueError("BEV minima must be below maxima")
        x_resolution = (limits[1] - limits[0]) / GRID_SIZE
        y_resolution = (limits[3] - limits[2]) / GRID_SIZE
        if not np.isclose(x_resolution, y_resolution, atol=1e-9, rtol=0.0):
            raise ValueError("BEV cells must be square")
        if isinstance(self.ray_stride, bool) or int(self.ray_stride) != self.ray_stride:
            raise ValueError("ray_stride must be a positive integer")
        if int(self.ray_stride) <= 0:
            raise ValueError("ray_stride must be a positive integer")
        if (
            isinstance(self.maximum_rays, bool)
            or int(self.maximum_rays) != self.maximum_rays
            or int(self.maximum_rays) <= 0
        ):
            raise ValueError("maximum_rays must be a positive integer")
        fraction = float(self.ray_step_fraction)
        if not isfinite(fraction) or not (0.0 < fraction <= 1.0):
            raise ValueError("ray_step_fraction must be in (0, 1]")
        object.__setattr__(self, "x_min_m", limits[0])
        object.__setattr__(self, "x_max_m", limits[1])
        object.__setattr__(self, "y_min_m", limits[2])
        object.__setattr__(self, "y_max_m", limits[3])
        object.__setattr__(self, "ray_stride", int(self.ray_stride))
        object.__setattr__(self, "maximum_rays", int(self.maximum_rays))
        object.__setattr__(self, "ray_step_fraction", fraction)

    @property
    def shape(self) -> tuple[int, int]:
        return (GRID_SIZE, GRID_SIZE)

    @property
    def resolution_m(self) -> float:
        return (self.x_max_m - self.x_min_m) / GRID_SIZE


@dataclass(frozen=True)
class STVLConfig:
    """Bounded temporal-memory policy for the local voxel projection."""

    decay_tau_s: float = 1.50
    unknown_after_s: float = 3.0
    minimum_hit_confidence: float = 0.20

    def __post_init__(self) -> None:
        tau = _finite_positive(self.decay_tau_s, "decay_tau_s")
        unknown_after = _finite_positive(
            self.unknown_after_s,
            "unknown_after_s",
        )
        threshold = float(self.minimum_hit_confidence)
        if not isfinite(threshold) or not (0.0 < threshold <= 1.0):
            raise ValueError("minimum_hit_confidence must be in (0, 1]")
        object.__setattr__(self, "decay_tau_s", tau)
        object.__setattr__(self, "unknown_after_s", unknown_after)
        object.__setattr__(self, "minimum_hit_confidence", threshold)


@dataclass(frozen=True)
class TrackerConfig:
    max_components: int = DEFAULT_MAX_COMPONENTS
    max_tracks: int = DEFAULT_MAX_TRACKS
    minimum_component_cells: int = 1
    connectivity: int = 8
    association_distance_m: float = 0.75
    maximum_missed_s: float = 1.0
    velocity_alpha: float = 1.0
    safety_radius_m: float = 0.25
    minimum_closing_speed_mps: float = 0.02

    def __post_init__(self) -> None:
        for name in (
            "max_components",
            "max_tracks",
            "minimum_component_cells",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if self.connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")
        object.__setattr__(
            self,
            "association_distance_m",
            _finite_positive(
                self.association_distance_m,
                "association_distance_m",
            ),
        )
        object.__setattr__(
            self,
            "maximum_missed_s",
            _finite_positive(self.maximum_missed_s, "maximum_missed_s"),
        )
        alpha = float(self.velocity_alpha)
        if not isfinite(alpha) or not (0.0 < alpha <= 1.0):
            raise ValueError("velocity_alpha must be in (0, 1]")
        object.__setattr__(self, "velocity_alpha", alpha)
        safety = float(self.safety_radius_m)
        if not isfinite(safety) or safety < 0.0:
            raise ValueError("safety_radius_m must be finite and non-negative")
        object.__setattr__(self, "safety_radius_m", safety)
        object.__setattr__(
            self,
            "minimum_closing_speed_mps",
            _finite_positive(
                self.minimum_closing_speed_mps,
                "minimum_closing_speed_mps",
            ),
        )


@dataclass(frozen=True)
class ReadOnlyAuthority:
    """Machine-readable non-interference contract."""

    publishes_cmd_vel: bool = False
    publishes_tf: bool = False
    accesses_f407: bool = False
    writes_nav_costmap: bool = False
    controls_base: bool = False
    ros_dependencies: tuple[str, ...] = ()


READ_ONLY_AUTHORITY = ReadOnlyAuthority()


@dataclass
class PointImage:
    """Dense, calibration-sized XYZ image with a validity mask."""

    points_xyz: np.ndarray
    valid: np.ndarray
    depth_m: np.ndarray
    frame: str

    def validate(self, intrinsics: CameraIntrinsics) -> None:
        expected_points = intrinsics.image_shape + (3,)
        if self.points_xyz.shape != expected_points:
            raise ValueError(
                f"points_xyz must have shape {expected_points}, "
                f"got {self.points_xyz.shape}"
            )
        if self.valid.shape != intrinsics.image_shape:
            raise ValueError("valid mask does not match calibrated image shape")
        if self.depth_m.shape != intrinsics.image_shape:
            raise ValueError("depth_m does not match calibrated image shape")
        if self.points_xyz.dtype != np.float32:
            raise ValueError("points_xyz must use float32")
        if self.valid.dtype != np.bool_:
            raise ValueError("valid must use bool")
        if self.depth_m.dtype != np.float32:
            raise ValueError("depth_m must use float32")
        if self.frame not in ("camera", "base"):
            raise ValueError("frame must be 'camera' or 'base'")


@dataclass
class DepthBEVGrid:
    """Fixed-size tri-state BEV plus height and temporal metadata."""

    hit: np.ndarray
    free: np.ndarray
    unknown: np.ndarray
    low: np.ndarray
    mid: np.ndarray
    high: np.ndarray
    min_height_m: np.ndarray
    max_height_m: np.ndarray
    height_variance_m2: np.ndarray
    age_s: np.ndarray
    hit_count: np.ndarray
    occupancy_confidence: np.ndarray
    source_valid_fraction: float = 0.0
    rays_used: int = 0

    def validate(self) -> None:
        shape = (GRID_SIZE, GRID_SIZE)
        for name in ("hit", "free", "unknown", "low", "mid", "high"):
            value = getattr(self, name)
            if value.shape != shape or value.dtype != np.bool_:
                raise ValueError(f"{name} must be a bool {shape} grid")
        for name in (
            "min_height_m",
            "max_height_m",
            "height_variance_m2",
            "age_s",
            "occupancy_confidence",
        ):
            value = getattr(self, name)
            if value.shape != shape or value.dtype != np.float32:
                raise ValueError(f"{name} must be a float32 {shape} grid")
        if self.hit_count.shape != shape or self.hit_count.dtype != np.uint16:
            raise ValueError(f"hit_count must be a uint16 {shape} grid")
        state_count = (
            self.hit.astype(np.uint8)
            + self.free.astype(np.uint8)
            + self.unknown.astype(np.uint8)
        )
        if not np.all(state_count == 1):
            raise ValueError("hit/free/unknown must be mutually exclusive and exhaustive")
        if np.any((self.low | self.mid | self.high) & ~self.hit):
            raise ValueError("height layers may only be set on hit cells")
        if np.any(self.occupancy_confidence < 0.0) or np.any(
            self.occupancy_confidence > 1.0
        ):
            raise ValueError("occupancy_confidence must lie in [0, 1]")


def empty_depth_bev(unknown_age_s: float) -> DepthBEVGrid:
    """Create an all-unknown grid with deterministic dtypes and shapes."""

    shape = (GRID_SIZE, GRID_SIZE)
    zeros_bool = np.zeros(shape, dtype=np.bool_)
    zeros_float = np.zeros(shape, dtype=np.float32)
    grid = DepthBEVGrid(
        hit=zeros_bool.copy(),
        free=zeros_bool.copy(),
        unknown=np.ones(shape, dtype=np.bool_),
        low=zeros_bool.copy(),
        mid=zeros_bool.copy(),
        high=zeros_bool.copy(),
        min_height_m=zeros_float.copy(),
        max_height_m=zeros_float.copy(),
        height_variance_m2=zeros_float.copy(),
        age_s=np.full(shape, float(unknown_age_s), dtype=np.float32),
        hit_count=np.zeros(shape, dtype=np.uint16),
        occupancy_confidence=zeros_float.copy(),
    )
    grid.validate()
    return grid
