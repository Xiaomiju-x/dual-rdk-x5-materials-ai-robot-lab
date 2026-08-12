"""Strict NumPy contracts for a monocular temporal CrossBEV student.

The contract is deliberately data-plane only. It contains no camera capture,
ROS publisher, network client, transform publisher, serial port, or actuator
interface. Metric BEV use is rejected unless calibration and per-frame
provenance pass an explicit gate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from math import isfinite

import numpy as np

SCHEMA_VERSION = "x5-crossbev-kd/1.0"
CROSSBEV_LAYER_NAMES = (
    "obstacle",
    "traversability",
    "semantic",
    "dynamic",
    "visibility",
    "unknown",
    "confidence",
)
_SHA256_CHARS = frozenset("0123456789abcdef")
_PROBABILITY_TOLERANCE = 1e-6


class ContractError(ValueError):
    """Raised when data cannot truthfully satisfy the CrossBEV contract."""


class ProvenanceState(str, Enum):
    """Origin of one image frame; origin and freshness are separate concepts."""

    LIVE_CAMERA = "live_camera"
    CACHED_CAMERA = "cached_camera"
    RECORDED_REPLAY = "recorded_replay"
    SYNTHETIC_FIXTURE = "synthetic_fixture"


def _text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ContractError(f"{name} must not be empty")
    if len(normalized) > 160:
        raise ContractError(f"{name} exceeds 160 characters")
    if any(ord(character) < 0x20 for character in normalized):
        raise ContractError(f"{name} contains control characters")
    return normalized


def _sha256(name: str, value: str) -> str:
    normalized = _text(name, value).lower()
    if len(normalized) != 64 or any(
        character not in _SHA256_CHARS for character in normalized
    ):
        raise ContractError(f"{name} must be a 64-character SHA-256 hex digest")
    return normalized


def _finite(name: str, value: float) -> float:
    number = float(value)
    if not isfinite(number):
        raise ContractError(f"{name} must be finite")
    return number


def _probability_grid(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or not array.size:
        raise ContractError(f"{name} must be a non-empty HxW grid")
    if not np.isfinite(array).all():
        raise ContractError(f"{name} contains non-finite values")
    if (
        float(array.min()) < -_PROBABILITY_TOLERANCE
        or float(array.max()) > 1.0 + _PROBABILITY_TOLERANCE
    ):
        raise ContractError(f"{name} values must lie in [0, 1]")
    result = np.ascontiguousarray(np.clip(array, 0.0, 1.0))
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class CalibrationRecord:
    """Audited camera calibration required before metric BEV projection."""

    calibration_id: str
    camera_id: str
    calibration_sha256: str
    source_width: int
    source_height: int
    intrinsics_fx_fy_cx_cy: tuple[float, float, float, float]
    camera_to_base: np.ndarray
    reprojection_rmse_px: float
    metric_error_p95_m: float
    approved_for_metric_bev: bool
    target_frame: str = "base_link"
    valid_from_s: float = 0.0
    valid_until_s: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "calibration_id", _text("calibration_id", self.calibration_id))
        object.__setattr__(self, "camera_id", _text("camera_id", self.camera_id))
        object.__setattr__(
            self,
            "calibration_sha256",
            _sha256("calibration_sha256", self.calibration_sha256),
        )
        object.__setattr__(self, "target_frame", _text("target_frame", self.target_frame))
        if self.target_frame != "base_link":
            raise ContractError("metric CrossBEV calibration target_frame must be base_link")
        for name, value in (
            ("source_width", self.source_width),
            ("source_height", self.source_height),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ContractError(f"{name} must be a positive integer")
        intrinsics = tuple(float(value) for value in self.intrinsics_fx_fy_cx_cy)
        if len(intrinsics) != 4 or not all(isfinite(value) for value in intrinsics):
            raise ContractError("intrinsics_fx_fy_cx_cy must contain four finite values")
        fx, fy, cx, cy = intrinsics
        if fx <= 0.0 or fy <= 0.0:
            raise ContractError("focal lengths must be positive")
        if not 0.0 <= cx <= self.source_width or not 0.0 <= cy <= self.source_height:
            raise ContractError("principal point must lie inside the source image")
        object.__setattr__(self, "intrinsics_fx_fy_cx_cy", intrinsics)

        transform = np.asarray(self.camera_to_base, dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ContractError("camera_to_base must be a finite 4x4 matrix")
        if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
            raise ContractError("camera_to_base must be a homogeneous transform")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
            raise ContractError("camera_to_base rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3):
            raise ContractError("camera_to_base rotation determinant must be +1")
        transform = np.ascontiguousarray(transform)
        transform.setflags(write=False)
        object.__setattr__(self, "camera_to_base", transform)

        reprojection = _finite("reprojection_rmse_px", self.reprojection_rmse_px)
        metric_error = _finite("metric_error_p95_m", self.metric_error_p95_m)
        if reprojection < 0.0 or metric_error < 0.0:
            raise ContractError("calibration errors must be non-negative")
        object.__setattr__(self, "reprojection_rmse_px", reprojection)
        object.__setattr__(self, "metric_error_p95_m", metric_error)
        if not isinstance(self.approved_for_metric_bev, (bool, np.bool_)):
            raise ContractError("approved_for_metric_bev must be boolean")
        valid_from = _finite("valid_from_s", self.valid_from_s)
        valid_until = (
            None
            if self.valid_until_s is None
            else _finite("valid_until_s", self.valid_until_s)
        )
        if valid_until is not None and valid_until <= valid_from:
            raise ContractError("valid_until_s must be later than valid_from_s")
        object.__setattr__(self, "valid_from_s", valid_from)
        object.__setattr__(self, "valid_until_s", valid_until)


@dataclass(frozen=True, slots=True)
class TemporalFrameProvenance:
    """Auditable identity and timing for one monocular frame."""

    state: ProvenanceState
    source_host: str
    source_pipeline: str
    camera_id: str
    calibration_id: str
    calibration_sha256: str
    frame_id: str
    sequence: int
    capture_timestamp_s: float
    receive_timestamp_s: float
    image_sha256: str
    image_supplied: bool = True

    def __post_init__(self) -> None:
        try:
            state = ProvenanceState(self.state)
        except ValueError as exc:
            raise ContractError(f"unsupported provenance state: {self.state}") from exc
        object.__setattr__(self, "state", state)
        for name in (
            "source_host",
            "source_pipeline",
            "camera_id",
            "calibration_id",
            "frame_id",
        ):
            object.__setattr__(self, name, _text(name, getattr(self, name)))
        object.__setattr__(
            self,
            "calibration_sha256",
            _sha256("calibration_sha256", self.calibration_sha256),
        )
        object.__setattr__(self, "image_sha256", _sha256("image_sha256", self.image_sha256))
        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool):
            raise ContractError("sequence must be an integer")
        if self.sequence < 0:
            raise ContractError("sequence must be non-negative")
        capture = _finite("capture_timestamp_s", self.capture_timestamp_s)
        received = _finite("receive_timestamp_s", self.receive_timestamp_s)
        if received + 1e-9 < capture:
            raise ContractError("receive_timestamp_s precedes capture_timestamp_s")
        object.__setattr__(self, "capture_timestamp_s", capture)
        object.__setattr__(self, "receive_timestamp_s", received)
        if not isinstance(self.image_supplied, (bool, np.bool_)):
            raise ContractError("image_supplied must be boolean")


@dataclass(frozen=True, slots=True)
class TemporalMonocularInput:
    """Fixed camera, ordered-frame input for a temporal monocular student."""

    images: np.ndarray
    calibration: CalibrationRecord
    provenance: tuple[TemporalFrameProvenance, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.calibration, CalibrationRecord):
            raise ContractError("calibration must be CalibrationRecord")
        frames = np.asarray(self.images, dtype=np.float32)
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ContractError("images must have shape Tx3xHxW")
        if not 2 <= frames.shape[0] <= 8 or min(frames.shape[2:]) < 8:
            raise ContractError("temporal input requires 2..8 frames with H,W >= 8")
        if not np.isfinite(frames).all():
            raise ContractError("images contain non-finite values")
        if float(frames.min()) < 0.0 or float(frames.max()) > 1.0:
            raise ContractError("images must be normalized to [0, 1]")
        frames = np.ascontiguousarray(frames)
        frames.setflags(write=False)
        object.__setattr__(self, "images", frames)

        records = tuple(self.provenance)
        if len(records) != frames.shape[0] or not all(
            isinstance(record, TemporalFrameProvenance) for record in records
        ):
            raise ContractError("one TemporalFrameProvenance is required per frame")
        timestamps = np.asarray(
            [record.capture_timestamp_s for record in records],
            dtype=np.float64,
        )
        sequences = np.asarray([record.sequence for record in records], dtype=np.int64)
        if np.any(np.diff(timestamps) <= 0.0):
            raise ContractError("capture timestamps must be strictly increasing")
        if np.any(np.diff(sequences) <= 0):
            raise ContractError("frame sequences must be strictly increasing")
        if len({record.camera_id for record in records}) != 1:
            raise ContractError("temporal input must come from one physical camera")
        object.__setattr__(self, "provenance", records)

    @property
    def frame_count(self) -> int:
        return int(self.images.shape[0])

    @property
    def capture_timestamps_s(self) -> np.ndarray:
        result = np.asarray(
            [record.capture_timestamp_s for record in self.provenance],
            dtype=np.float64,
        )
        result.setflags(write=False)
        return result


@dataclass(frozen=True, slots=True)
class CrossBEVMaps:
    """Seven independent BEV probability layers.

    Semantic cost is never folded into physical occupancy, and unknown is
    never interpreted as free. Downstream code must choose an explicit policy.
    """

    obstacle: np.ndarray
    traversability: np.ndarray
    semantic: np.ndarray
    dynamic: np.ndarray
    visibility: np.ndarray
    unknown: np.ndarray
    confidence: np.ndarray

    def __post_init__(self) -> None:
        shape: tuple[int, int] | None = None
        for name in CROSSBEV_LAYER_NAMES:
            grid = _probability_grid(name, getattr(self, name))
            if shape is None:
                shape = grid.shape
            elif grid.shape != shape:
                raise ContractError(
                    f"all CrossBEV layers must share shape {shape}, got {name}={grid.shape}"
                )
            object.__setattr__(self, name, grid)

    @property
    def shape(self) -> tuple[int, int]:
        return self.obstacle.shape

    def as_array(self) -> np.ndarray:
        """Return a new CxHxW float32 tensor in the documented layer order."""

        return np.stack(
            [getattr(self, name) for name in CROSSBEV_LAYER_NAMES],
            axis=0,
        ).astype(np.float32, copy=False)

    def as_mapping(self) -> Mapping[str, np.ndarray]:
        return {name: getattr(self, name) for name in CROSSBEV_LAYER_NAMES}


@dataclass(frozen=True, slots=True)
class GatePolicy:
    """Strict policy for allowing metric student inference or distillation."""

    accepted_states: tuple[ProvenanceState, ...] = (
        ProvenanceState.LIVE_CAMERA,
        ProvenanceState.RECORDED_REPLAY,
    )
    max_age_s: float = 0.50
    max_transport_latency_s: float = 0.25
    max_interframe_gap_s: float = 0.30
    min_temporal_span_s: float = 0.05
    max_reprojection_rmse_px: float = 2.0
    max_metric_error_p95_m: float = 0.10
    require_image_digest: bool = True

    def __post_init__(self) -> None:
        try:
            states = tuple(ProvenanceState(state) for state in self.accepted_states)
        except ValueError as exc:
            raise ContractError("accepted_states contains an invalid state") from exc
        if not states:
            raise ContractError("accepted_states must not be empty")
        object.__setattr__(self, "accepted_states", states)
        for name in (
            "max_age_s",
            "max_transport_latency_s",
            "max_interframe_gap_s",
            "min_temporal_span_s",
            "max_reprojection_rmse_px",
            "max_metric_error_p95_m",
        ):
            number = _finite(name, getattr(self, name))
            if number < 0.0:
                raise ContractError(f"{name} must be non-negative")
            object.__setattr__(self, name, number)
        if self.max_interframe_gap_s < self.min_temporal_span_s:
            raise ContractError("max_interframe_gap_s must cover min_temporal_span_s")
        if not isinstance(self.require_image_digest, (bool, np.bool_)):
            raise ContractError("require_image_digest must be boolean")


@dataclass(frozen=True, slots=True)
class GateDecision:
    accepted: bool
    reasons: tuple[str, ...]
    latest_age_s: float
    temporal_span_s: float
    max_transport_latency_s: float


def assess_temporal_input(
    temporal_input: TemporalMonocularInput,
    *,
    now_s: float,
    policy: GatePolicy | None = None,
) -> GateDecision:
    """Evaluate calibration, source identity, timing, and digest consistency."""

    if not isinstance(temporal_input, TemporalMonocularInput):
        raise TypeError("temporal_input must be TemporalMonocularInput")
    current_time = _finite("now_s", now_s)
    cfg = policy or GatePolicy()
    if not isinstance(cfg, GatePolicy):
        raise TypeError("policy must be GatePolicy")

    calibration = temporal_input.calibration
    records = temporal_input.provenance
    timestamps = temporal_input.capture_timestamps_s
    reasons: list[str] = []
    latest_age = current_time - float(timestamps[-1])
    temporal_span = float(timestamps[-1] - timestamps[0])
    transport_latencies = np.asarray(
        [
            record.receive_timestamp_s - record.capture_timestamp_s
            for record in records
        ],
        dtype=np.float64,
    )
    maximum_transport = float(transport_latencies.max(initial=0.0))

    if not bool(calibration.approved_for_metric_bev):
        reasons.append("CALIBRATION_NOT_APPROVED")
    if calibration.reprojection_rmse_px > cfg.max_reprojection_rmse_px:
        reasons.append("CALIBRATION_REPROJECTION_ERROR")
    if calibration.metric_error_p95_m > cfg.max_metric_error_p95_m:
        reasons.append("CALIBRATION_METRIC_ERROR")
    if latest_age > cfg.max_age_s:
        reasons.append("STALE_SEQUENCE")
    if latest_age < -1e-3:
        reasons.append("FUTURE_SEQUENCE")
    if temporal_span < cfg.min_temporal_span_s:
        reasons.append("TEMPORAL_SPAN_TOO_SHORT")
    if np.any(np.diff(timestamps) > cfg.max_interframe_gap_s):
        reasons.append("INTERFRAME_GAP")
    if maximum_transport > cfg.max_transport_latency_s:
        reasons.append("TRANSPORT_LATENCY")

    for record in records:
        if record.state not in cfg.accepted_states:
            reasons.append("PROVENANCE_NOT_ACCEPTED")
        if not bool(record.image_supplied):
            reasons.append("IMAGE_NOT_SUPPLIED")
        if record.camera_id != calibration.camera_id:
            reasons.append("CAMERA_ID_MISMATCH")
        if record.calibration_id != calibration.calibration_id:
            reasons.append("CALIBRATION_ID_MISMATCH")
        if record.calibration_sha256 != calibration.calibration_sha256:
            reasons.append("CALIBRATION_DIGEST_MISMATCH")
        if cfg.require_image_digest and not record.image_sha256:
            reasons.append("IMAGE_DIGEST_MISSING")
        if record.capture_timestamp_s < calibration.valid_from_s:
            reasons.append("CALIBRATION_NOT_YET_VALID")
        if (
            calibration.valid_until_s is not None
            and record.capture_timestamp_s > calibration.valid_until_s
        ):
            reasons.append("CALIBRATION_EXPIRED")

    return GateDecision(
        accepted=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        latest_age_s=float(latest_age),
        temporal_span_s=temporal_span,
        max_transport_latency_s=maximum_transport,
    )


def require_accepted_temporal_input(
    temporal_input: TemporalMonocularInput,
    *,
    now_s: float,
    policy: GatePolicy | None = None,
) -> GateDecision:
    """Raise before metric inference when calibration/provenance is untrusted."""

    decision = assess_temporal_input(temporal_input, now_s=now_s, policy=policy)
    if not decision.accepted:
        raise ContractError(
            "CrossBEV temporal input rejected: " + ", ".join(decision.reasons)
        )
    return decision


__all__ = [
    "CROSSBEV_LAYER_NAMES",
    "SCHEMA_VERSION",
    "CalibrationRecord",
    "ContractError",
    "CrossBEVMaps",
    "GateDecision",
    "GatePolicy",
    "ProvenanceState",
    "TemporalFrameProvenance",
    "TemporalMonocularInput",
    "assess_temporal_input",
    "require_accepted_temporal_input",
]
