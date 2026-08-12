"""Strict in-memory contracts for the passive 4K semantic BEV bridge.

The module deliberately contains no camera, network, ROS, or actuator code.
It describes data that an external adapter may supply after doing inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite, sqrt
from typing import Iterable

import numpy as np

from ..contracts.core import BEVGeometryV2


SCHEMA_VERSION = "x5-vision-fsd-semantic-bev/1.0"
MAX_VECTOR_TOKENS = 128
MAX_POINTS_PER_TOKEN = 32
MAX_TEXT_CHARS = 160
PROBABILITY_TOLERANCE = 1e-6
METRIC_TOLERANCE = 0.01


class ContractError(ValueError):
    """Raised when a semantic bridge object violates its static contract."""


class ProvenanceState(str, Enum):
    """Truthful origin of the image-derived observation."""

    LIVE_CAMERA = "live_camera"
    CACHED_CAMERA = "cached_camera"
    RECORDED_REPLAY = "recorded_replay"
    SYNTHETIC_FIXTURE = "synthetic_fixture"


class VectorTokenKind(str, Enum):
    """Sparse vector evidence accompanying the dense semantic BEV."""

    STATIC_BOUNDARY = "static_boundary"
    DYNAMIC_OBJECT = "dynamic_object"
    DRIVABLE_EDGE = "drivable_edge"
    OCCLUSION_BOUNDARY = "occlusion_boundary"
    SEMANTIC_REGION = "semantic_region"


def _require_text(name: str, value: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized and not allow_empty:
        raise ContractError(f"{name} must not be empty")
    if len(normalized) > MAX_TEXT_CHARS:
        raise ContractError(f"{name} exceeds {MAX_TEXT_CHARS} characters")
    if any(ord(char) < 0x20 for char in normalized):
        raise ContractError(f"{name} contains control characters")
    return normalized


def _require_probability(name: str, value: float) -> float:
    number = float(value)
    if not isfinite(number) or not 0.0 <= number <= 1.0:
        raise ContractError(f"{name} must be finite and in [0, 1]")
    return number


def _require_sha256(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _require_text(name, value).lower()
    if len(normalized) != 64 or any(
        char not in "0123456789abcdef" for char in normalized
    ):
        raise ContractError(f"{name} must be a lowercase SHA-256 hex digest")
    return normalized


def _immutable_grid(
    name: str,
    value: np.ndarray,
    shape: tuple[int, int],
    *,
    kind: str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ContractError(f"{name} must have shape {shape}, got {array.shape}")

    if kind == "probability":
        array = np.asarray(array, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ContractError(f"{name} contains non-finite values")
        if (
            float(array.min(initial=0.0)) < -PROBABILITY_TOLERANCE
            or float(array.max(initial=0.0)) > 1.0 + PROBABILITY_TOLERANCE
        ):
            raise ContractError(f"{name} values must be in [0, 1]")
        array = np.clip(array, 0.0, 1.0)
    elif kind == "class_id":
        if array.dtype.kind not in "uib":
            raise ContractError(f"{name} must contain integer class identifiers")
        if array.size and (
            int(array.min()) < 0 or int(array.max()) > np.iinfo(np.uint8).max
        ):
            raise ContractError(f"{name} values must fit uint8")
        array = np.asarray(array, dtype=np.uint8)
    elif kind == "visibility":
        if array.dtype.kind not in "bu":
            raise ContractError(f"{name} must be boolean or uint8")
        if array.dtype.kind == "u" and array.size and int(array.max()) > 1:
            raise ContractError(f"{name} uint8 values must be 0 or 1")
        array = np.asarray(array, dtype=np.bool_)
    else:
        raise AssertionError(f"unknown grid kind: {kind}")

    array = np.ascontiguousarray(array)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Calibrated source camera intrinsics before BEV projection."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str = "plumb_bob"
    distortion: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.width, int)
            or isinstance(self.width, bool)
            or not 1 <= self.width <= 16384
        ):
            raise ContractError("intrinsics width must be an integer in [1, 16384]")
        if (
            not isinstance(self.height, int)
            or isinstance(self.height, bool)
            or not 1 <= self.height <= 16384
        ):
            raise ContractError("intrinsics height must be an integer in [1, 16384]")
        for name, value in (
            ("fx", self.fx),
            ("fy", self.fy),
            ("cx", self.cx),
            ("cy", self.cy),
        ):
            if not isfinite(float(value)):
                raise ContractError(f"intrinsics {name} must be finite")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ContractError("intrinsics focal lengths must be positive")
        if not 0.0 <= self.cx <= float(self.width):
            raise ContractError("intrinsics cx must lie within the source image")
        if not 0.0 <= self.cy <= float(self.height):
            raise ContractError("intrinsics cy must lie within the source image")
        model = _require_text("distortion_model", self.distortion_model)
        if model not in {
            "none",
            "plumb_bob",
            "rational_polynomial",
            "equidistant",
        }:
            raise ContractError(f"unsupported distortion model: {model}")
        coefficients = tuple(float(value) for value in self.distortion)
        if len(coefficients) > 14 or not all(isfinite(value) for value in coefficients):
            raise ContractError("distortion must contain at most 14 finite values")
        object.__setattr__(self, "distortion_model", model)
        object.__setattr__(self, "distortion", coefficients)


@dataclass(frozen=True, slots=True)
class CameraExtrinsics:
    """Rigid transform from the camera frame into the robot base frame."""

    source_frame: str
    target_frame: str
    translation_m: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        source = _require_text("source_frame", self.source_frame)
        target = _require_text("target_frame", self.target_frame)
        if source == target:
            raise ContractError("extrinsics source and target frames must differ")
        translation = tuple(float(value) for value in self.translation_m)
        rotation = tuple(float(value) for value in self.rotation_xyzw)
        if len(translation) != 3 or not all(isfinite(value) for value in translation):
            raise ContractError("translation_m must contain three finite values")
        if len(rotation) != 4 or not all(isfinite(value) for value in rotation):
            raise ContractError("rotation_xyzw must contain four finite values")
        norm = sqrt(sum(value * value for value in rotation))
        if abs(norm - 1.0) > 1e-3:
            raise ContractError("rotation_xyzw must be a normalized quaternion")
        object.__setattr__(self, "source_frame", source)
        object.__setattr__(self, "target_frame", target)
        object.__setattr__(self, "translation_m", translation)
        object.__setattr__(self, "rotation_xyzw", rotation)


@dataclass(frozen=True, slots=True)
class FrameProvenance:
    """Auditable identity and timing for one supplied camera result."""

    state: ProvenanceState
    source_host: str
    source_pipeline: str
    model_id: str
    frame_id: str
    calibration_id: str
    image_supplied: bool
    capture_timestamp_s: float
    inference_timestamp_s: float
    model_sha256: str | None = None
    input_sha256: str | None = None

    def __post_init__(self) -> None:
        try:
            state = ProvenanceState(self.state)
        except ValueError as exc:
            raise ContractError(f"unsupported provenance state: {self.state}") from exc
        for name in (
            "source_host",
            "source_pipeline",
            "model_id",
            "frame_id",
            "calibration_id",
        ):
            object.__setattr__(self, name, _require_text(name, getattr(self, name)))
        if not isinstance(self.image_supplied, (bool, np.bool_)):
            raise ContractError("image_supplied must be boolean")
        capture = float(self.capture_timestamp_s)
        inference = float(self.inference_timestamp_s)
        if not isfinite(capture) or not isfinite(inference):
            raise ContractError("provenance timestamps must be finite")
        if inference + 1e-9 < capture:
            raise ContractError("inference timestamp precedes capture timestamp")
        if state in {
            ProvenanceState.LIVE_CAMERA,
            ProvenanceState.CACHED_CAMERA,
            ProvenanceState.RECORDED_REPLAY,
        } and not bool(self.image_supplied):
            raise ContractError(f"{state.value} provenance requires image_supplied=true")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "image_supplied", bool(self.image_supplied))
        object.__setattr__(self, "capture_timestamp_s", capture)
        object.__setattr__(self, "inference_timestamp_s", inference)
        object.__setattr__(
            self,
            "model_sha256",
            _require_sha256("model_sha256", self.model_sha256),
        )
        object.__setattr__(
            self,
            "input_sha256",
            _require_sha256("input_sha256", self.input_sha256),
        )


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    """Declared quality metrics checked against the dense arrays."""

    overall_score: float
    projection_valid_fraction: float
    visible_fraction: float
    mean_confidence: float
    dropped_input_fraction: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "overall_score",
            "projection_valid_fraction",
            "visible_fraction",
            "mean_confidence",
            "dropped_input_fraction",
        ):
            object.__setattr__(
                self,
                name,
                _require_probability(name, getattr(self, name)),
            )
        if self.visible_fraction > self.projection_valid_fraction + METRIC_TOLERANCE:
            raise ContractError(
                "visible_fraction cannot exceed projection_valid_fraction"
            )


@dataclass(frozen=True, slots=True)
class SparseVectorToken:
    """A bounded vector-space token in the robot-centric BEV frame."""

    token_id: str
    kind: VectorTokenKind
    class_id: int
    confidence: float
    points_xy_m: tuple[tuple[float, float], ...]
    velocity_xy_mps: tuple[float, float] = (0.0, 0.0)
    track_id: str | None = None

    def __post_init__(self) -> None:
        token_id = _require_text("token_id", self.token_id)
        try:
            kind = VectorTokenKind(self.kind)
        except ValueError as exc:
            raise ContractError(f"unsupported vector token kind: {self.kind}") from exc
        if (
            not isinstance(self.class_id, int)
            or isinstance(self.class_id, bool)
            or not 0 <= self.class_id <= 255
        ):
            raise ContractError("class_id must be an integer in [0, 255]")
        confidence = _require_probability("token confidence", self.confidence)
        points = tuple(tuple(float(axis) for axis in point) for point in self.points_xy_m)
        if not 1 <= len(points) <= MAX_POINTS_PER_TOKEN:
            raise ContractError(
                f"points_xy_m must contain 1..{MAX_POINTS_PER_TOKEN} points"
            )
        if any(
            len(point) != 2 or not all(isfinite(axis) for axis in point)
            for point in points
        ):
            raise ContractError("each vector point must contain two finite values")
        velocity = tuple(float(axis) for axis in self.velocity_xy_mps)
        if len(velocity) != 2 or not all(isfinite(axis) for axis in velocity):
            raise ContractError("velocity_xy_mps must contain two finite values")
        if max(abs(axis) for axis in velocity) > 100.0:
            raise ContractError("velocity_xy_mps exceeds the defensive limit")
        track_id = (
            _require_text("track_id", self.track_id)
            if self.track_id is not None
            else None
        )
        if kind == VectorTokenKind.DYNAMIC_OBJECT and track_id is None:
            raise ContractError("dynamic object tokens require a track_id")
        object.__setattr__(self, "token_id", token_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "points_xy_m", points)
        object.__setattr__(self, "velocity_xy_mps", velocity)
        object.__setattr__(self, "track_id", track_id)

    @property
    def is_dynamic(self) -> bool:
        return self.kind == VectorTokenKind.DYNAMIC_OBJECT


@dataclass(frozen=True, slots=True)
class SemanticBEVFrame:
    """Dense and sparse semantic evidence ready for the cross-X5 bridge."""

    timestamp_s: float
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    provenance: FrameProvenance
    quality: QualityMetrics
    semantic_risk: np.ndarray
    confidence: np.ndarray
    dynamic_probability: np.ndarray
    class_ids: np.ndarray
    visibility: np.ndarray
    vector_tokens: tuple[SparseVectorToken, ...] = ()
    geometry: BEVGeometryV2 = field(default_factory=BEVGeometryV2)

    def __post_init__(self) -> None:
        timestamp = float(self.timestamp_s)
        if not isfinite(timestamp):
            raise ContractError("timestamp_s must be finite")
        if not isinstance(self.intrinsics, CameraIntrinsics):
            raise ContractError("intrinsics must be CameraIntrinsics")
        if not isinstance(self.extrinsics, CameraExtrinsics):
            raise ContractError("extrinsics must be CameraExtrinsics")
        if not isinstance(self.provenance, FrameProvenance):
            raise ContractError("provenance must be FrameProvenance")
        if not isinstance(self.quality, QualityMetrics):
            raise ContractError("quality must be QualityMetrics")
        if not isinstance(self.geometry, BEVGeometryV2):
            raise ContractError("geometry must be BEVGeometryV2")
        if self.geometry.shape != (64, 64):
            raise ContractError("the finals vNext semantic bridge requires 64x64 BEV")
        if timestamp + 1e-9 < self.provenance.inference_timestamp_s:
            raise ContractError("frame timestamp precedes inference timestamp")

        shape = self.geometry.shape
        risk = _immutable_grid(
            "semantic_risk",
            self.semantic_risk,
            shape,
            kind="probability",
        )
        confidence = _immutable_grid(
            "confidence",
            self.confidence,
            shape,
            kind="probability",
        )
        dynamic = _immutable_grid(
            "dynamic_probability",
            self.dynamic_probability,
            shape,
            kind="probability",
        )
        class_ids = _immutable_grid(
            "class_ids",
            self.class_ids,
            shape,
            kind="class_id",
        )
        visibility = _immutable_grid(
            "visibility",
            self.visibility,
            shape,
            kind="visibility",
        )

        hidden = ~visibility
        if (
            np.any(risk[hidden] > PROBABILITY_TOLERANCE)
            or np.any(confidence[hidden] > PROBABILITY_TOLERANCE)
            or np.any(dynamic[hidden] > PROBABILITY_TOLERANCE)
            or np.any(class_ids[hidden] != 0)
        ):
            raise ContractError(
                "invisible cells must carry zero dense semantic evidence"
            )

        actual_visible = float(np.mean(visibility, dtype=np.float64))
        actual_confidence = (
            float(np.mean(confidence[visibility], dtype=np.float64))
            if visibility.any()
            else 0.0
        )
        if abs(actual_visible - self.quality.visible_fraction) > METRIC_TOLERANCE:
            raise ContractError(
                "declared visible_fraction does not match the visibility mask"
            )
        if abs(actual_confidence - self.quality.mean_confidence) > METRIC_TOLERANCE:
            raise ContractError(
                "declared mean_confidence does not match the confidence grid"
            )

        tokens = tuple(self.vector_tokens)
        if len(tokens) > MAX_VECTOR_TOKENS:
            raise ContractError(
                f"vector_tokens exceeds the {MAX_VECTOR_TOKENS} token limit"
            )
        if any(not isinstance(token, SparseVectorToken) for token in tokens):
            raise ContractError("vector_tokens must contain SparseVectorToken values")
        token_ids = [token.token_id for token in tokens]
        if len(set(token_ids)) != len(token_ids):
            raise ContractError("vector token identifiers must be unique per frame")
        for token in tokens:
            for x_m, y_m in token.points_xy_m:
                if not (
                    self.geometry.x_min_m <= x_m < self.geometry.x_max_m
                    and self.geometry.y_min_m <= y_m < self.geometry.y_max_m
                ):
                    raise ContractError(
                        f"vector token {token.token_id} lies outside the BEV"
                    )

        object.__setattr__(self, "timestamp_s", timestamp)
        object.__setattr__(self, "semantic_risk", risk)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "dynamic_probability", dynamic)
        object.__setattr__(self, "class_ids", class_ids)
        object.__setattr__(self, "visibility", visibility)
        object.__setattr__(self, "vector_tokens", tokens)


@dataclass(frozen=True, slots=True)
class OdometryDelta:
    """Robot motion from the previous base frame into the current frame."""

    dx_m: float = 0.0
    dy_m: float = 0.0
    dyaw_rad: float = 0.0
    dt_s: float = 0.0

    def __post_init__(self) -> None:
        values = tuple(
            float(value)
            for value in (self.dx_m, self.dy_m, self.dyaw_rad, self.dt_s)
        )
        if not all(isfinite(value) for value in values):
            raise ContractError("odometry values must be finite")
        if values[3] < 0.0:
            raise ContractError("odometry dt_s must be non-negative")
        object.__setattr__(self, "dx_m", values[0])
        object.__setattr__(self, "dy_m", values[1])
        object.__setattr__(self, "dyaw_rad", values[2])
        object.__setattr__(self, "dt_s", values[3])


@dataclass(frozen=True, slots=True)
class PayloadLimits:
    """Defensive cross-X5 payload and decompression bounds."""

    max_payload_bytes: int = 256 * 1024
    max_header_bytes: int = 64 * 1024
    max_raw_bytes: int = 64 * 1024
    max_tokens: int = MAX_VECTOR_TOKENS
    max_points_per_token: int = MAX_POINTS_PER_TOKEN

    def __post_init__(self) -> None:
        values = (
            self.max_payload_bytes,
            self.max_header_bytes,
            self.max_raw_bytes,
            self.max_tokens,
            self.max_points_per_token,
        )
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in values
        ):
            raise ContractError("payload limits must be positive integers")
        if self.max_payload_bytes > 1024 * 1024:
            raise ContractError("max_payload_bytes exceeds the 1 MiB hard limit")
        if self.max_header_bytes > 256 * 1024:
            raise ContractError("max_header_bytes exceeds the 256 KiB hard limit")
        if self.max_raw_bytes > 512 * 1024:
            raise ContractError("max_raw_bytes exceeds the 512 KiB hard limit")
        if self.max_tokens > MAX_VECTOR_TOKENS:
            raise ContractError("max_tokens exceeds the contract hard limit")
        if self.max_points_per_token > MAX_POINTS_PER_TOKEN:
            raise ContractError(
                "max_points_per_token exceeds the contract hard limit"
            )


def token_tuple(values: Iterable[SparseVectorToken]) -> tuple[SparseVectorToken, ...]:
    """Materialize a token iterable once for callers constructing a frame."""

    return tuple(values)


__all__ = [
    "CameraExtrinsics",
    "CameraIntrinsics",
    "ContractError",
    "FrameProvenance",
    "MAX_POINTS_PER_TOKEN",
    "MAX_VECTOR_TOKENS",
    "METRIC_TOLERANCE",
    "OdometryDelta",
    "PayloadLimits",
    "ProvenanceState",
    "QualityMetrics",
    "SCHEMA_VERSION",
    "SemanticBEVFrame",
    "SparseVectorToken",
    "VectorTokenKind",
    "token_tuple",
]
