"""Bounded binary codec for cross-X5 semantic BEV payloads.

The format uses canonical JSON metadata plus zlib-compressed, quantized dense
arrays. It never uses pickle or executable object deserialization.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import struct
import zlib
from typing import Any, Mapping

import numpy as np

from ..contracts.core import BEVGeometryV2
from .contracts import (
    CameraExtrinsics,
    CameraIntrinsics,
    ContractError,
    FrameProvenance,
    PayloadLimits,
    ProvenanceState,
    QualityMetrics,
    SCHEMA_VERSION,
    SemanticBEVFrame,
    SparseVectorToken,
    VectorTokenKind,
)


MAGIC = b"XFSDv1\x00\x00"
_PREFIX = struct.Struct(">8sIII32s")
_ARRAY_ORDER = (
    "semantic_risk",
    "confidence",
    "dynamic_probability",
    "class_ids",
    "visibility",
)
_TOP_LEVEL_KEYS = {
    "schema_version",
    "timestamp_s",
    "geometry",
    "intrinsics",
    "extrinsics",
    "provenance",
    "quality",
    "vector_tokens",
    "array_layout",
    "compression",
    "authority",
}


class PayloadError(ContractError):
    """Raised when an encoded payload is malformed, unsafe, or corrupted."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise PayloadError("payload metadata is not canonical JSON") from exc
    return text.encode("ascii")


def _quantize_probability(array: np.ndarray) -> np.ndarray:
    return np.rint(np.asarray(array, dtype=np.float32) * 255.0).astype(np.uint8)


def _array_layout(height: int, width: int) -> tuple[list[dict[str, Any]], int]:
    cells = height * width
    visibility_bytes = (cells + 7) // 8
    sizes = (cells, cells, cells, cells, visibility_bytes)
    layout: list[dict[str, Any]] = []
    offset = 0
    for name, size in zip(_ARRAY_ORDER, sizes, strict=True):
        entry: dict[str, Any] = {
            "name": name,
            "dtype": "uint8",
            "shape": [height, width],
            "offset": offset,
            "nbytes": size,
        }
        if name in {"semantic_risk", "confidence", "dynamic_probability"}:
            entry["quantization"] = "round(value*255)"
        elif name == "visibility":
            entry["packing"] = "packbits-little"
        else:
            entry["quantization"] = "identity"
        layout.append(entry)
        offset += size
    return layout, offset


def _token_to_dict(token: SparseVectorToken) -> dict[str, Any]:
    return {
        "token_id": token.token_id,
        "kind": token.kind.value,
        "class_id": token.class_id,
        "confidence": token.confidence,
        "points_xy_m": [list(point) for point in token.points_xy_m],
        "velocity_xy_mps": list(token.velocity_xy_mps),
        "track_id": token.track_id,
    }


def _frame_header(frame: SemanticBEVFrame) -> dict[str, Any]:
    geometry = frame.geometry
    layout, raw_size = _array_layout(geometry.height, geometry.width)
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_s": frame.timestamp_s,
        "geometry": {
            "height": geometry.height,
            "width": geometry.width,
            "resolution_m": geometry.resolution_m,
            "x_min_m": geometry.x_min_m,
            "y_min_m": geometry.y_min_m,
        },
        "intrinsics": {
            "width": frame.intrinsics.width,
            "height": frame.intrinsics.height,
            "fx": frame.intrinsics.fx,
            "fy": frame.intrinsics.fy,
            "cx": frame.intrinsics.cx,
            "cy": frame.intrinsics.cy,
            "distortion_model": frame.intrinsics.distortion_model,
            "distortion": list(frame.intrinsics.distortion),
        },
        "extrinsics": {
            "source_frame": frame.extrinsics.source_frame,
            "target_frame": frame.extrinsics.target_frame,
            "translation_m": list(frame.extrinsics.translation_m),
            "rotation_xyzw": list(frame.extrinsics.rotation_xyzw),
        },
        "provenance": {
            "state": frame.provenance.state.value,
            "source_host": frame.provenance.source_host,
            "source_pipeline": frame.provenance.source_pipeline,
            "model_id": frame.provenance.model_id,
            "frame_id": frame.provenance.frame_id,
            "calibration_id": frame.provenance.calibration_id,
            "image_supplied": frame.provenance.image_supplied,
            "capture_timestamp_s": frame.provenance.capture_timestamp_s,
            "inference_timestamp_s": frame.provenance.inference_timestamp_s,
            "model_sha256": frame.provenance.model_sha256,
            "input_sha256": frame.provenance.input_sha256,
        },
        "quality": {
            "overall_score": frame.quality.overall_score,
            "projection_valid_fraction": (
                frame.quality.projection_valid_fraction
            ),
            "visible_fraction": frame.quality.visible_fraction,
            "mean_confidence": frame.quality.mean_confidence,
            "dropped_input_fraction": frame.quality.dropped_input_fraction,
        },
        "vector_tokens": [_token_to_dict(token) for token in frame.vector_tokens],
        "array_layout": layout,
        "compression": {
            "codec": "zlib",
            "raw_bytes": raw_size,
            "probability_encoding": "uint8-linear",
        },
        "authority": {
            "shadow_only": True,
            "publishes_motion": False,
            "publishes_tf": False,
            "writes_serial": False,
        },
    }


def _raw_arrays(frame: SemanticBEVFrame) -> bytes:
    chunks = [
        _quantize_probability(frame.semantic_risk).tobytes(order="C"),
        _quantize_probability(frame.confidence).tobytes(order="C"),
        _quantize_probability(frame.dynamic_probability).tobytes(order="C"),
        np.asarray(frame.class_ids, dtype=np.uint8).tobytes(order="C"),
        np.packbits(
            np.asarray(frame.visibility, dtype=np.uint8).reshape(-1),
            bitorder="little",
        ).tobytes(order="C"),
    ]
    return b"".join(chunks)


def encode_payload(
    frame: SemanticBEVFrame,
    *,
    limits: PayloadLimits | None = None,
    compression_level: int = 6,
) -> bytes:
    """Encode a structurally validated frame under explicit byte limits."""

    if not isinstance(frame, SemanticBEVFrame):
        raise TypeError("frame must be SemanticBEVFrame")
    cfg = limits or PayloadLimits()
    if not isinstance(cfg, PayloadLimits):
        raise TypeError("limits must be PayloadLimits")
    if (
        not isinstance(compression_level, int)
        or isinstance(compression_level, bool)
        or not 0 <= compression_level <= 9
    ):
        raise PayloadError("compression_level must be an integer in [0, 9]")
    if len(frame.vector_tokens) > cfg.max_tokens:
        raise PayloadError("frame exceeds the configured token limit")
    if any(
        len(token.points_xy_m) > cfg.max_points_per_token
        for token in frame.vector_tokens
    ):
        raise PayloadError("frame exceeds the configured token point limit")

    header = _canonical_json(_frame_header(frame))
    if len(header) > cfg.max_header_bytes:
        raise PayloadError("encoded metadata exceeds max_header_bytes")
    raw = _raw_arrays(frame)
    if len(raw) > cfg.max_raw_bytes:
        raise PayloadError("dense arrays exceed max_raw_bytes")
    compressed = zlib.compress(raw, level=compression_level)
    digest = hashlib.sha256(header + raw).digest()
    prefix = _PREFIX.pack(
        MAGIC,
        len(header),
        len(compressed),
        len(raw),
        digest,
    )
    payload = prefix + header + compressed
    if len(payload) > cfg.max_payload_bytes:
        raise PayloadError("encoded payload exceeds max_payload_bytes")
    return payload


def _bounded_decompress(
    compressed: bytes,
    *,
    declared_raw_bytes: int,
    max_raw_bytes: int,
) -> bytes:
    if declared_raw_bytes <= 0 or declared_raw_bytes > max_raw_bytes:
        raise PayloadError("declared raw payload size is outside allowed bounds")
    decompressor = zlib.decompressobj()
    try:
        raw = decompressor.decompress(compressed, declared_raw_bytes + 1)
        if len(raw) > declared_raw_bytes or decompressor.unconsumed_tail:
            raise PayloadError("compressed body expands beyond its declared size")
        remaining = declared_raw_bytes + 1 - len(raw)
        if remaining > 0:
            raw += decompressor.flush(remaining)
    except zlib.error as exc:
        raise PayloadError("compressed body is not a valid zlib stream") from exc
    if len(raw) != declared_raw_bytes:
        raise PayloadError("decompressed body length does not match metadata")
    if not decompressor.eof or decompressor.unused_data:
        raise PayloadError("compressed body has an incomplete or trailing stream")
    return raw


def _mapping(name: str, value: Any, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PayloadError(f"{name} must be an object")
    if set(value) != keys:
        raise PayloadError(f"{name} has unexpected or missing fields")
    return value


def _number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PayloadError(f"{name} must be numeric")
    number = float(value)
    if not np.isfinite(number):
        raise PayloadError(f"{name} must be finite")
    return number


def _integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PayloadError(f"{name} must be an integer")
    return value


def _string(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise PayloadError(f"{name} must be a string")
    return value


def _sequence(name: str, value: Any, length: int | None = None) -> list[Any]:
    if not isinstance(value, list):
        raise PayloadError(f"{name} must be an array")
    if length is not None and len(value) != length:
        raise PayloadError(f"{name} must contain {length} values")
    return value


def _parse_geometry(value: Any) -> BEVGeometryV2:
    data = _mapping(
        "geometry",
        value,
        {"height", "width", "resolution_m", "x_min_m", "y_min_m"},
    )
    try:
        geometry = BEVGeometryV2(
            height=_integer("geometry.height", data["height"]),
            width=_integer("geometry.width", data["width"]),
            resolution_m=_number(
                "geometry.resolution_m",
                data["resolution_m"],
            ),
            x_min_m=_number("geometry.x_min_m", data["x_min_m"]),
            y_min_m=_number("geometry.y_min_m", data["y_min_m"]),
        )
    except (TypeError, ValueError) as exc:
        raise PayloadError("invalid geometry") from exc
    if geometry.shape != (64, 64):
        raise PayloadError("semantic bridge payload geometry must be 64x64")
    return geometry


def _parse_intrinsics(value: Any) -> CameraIntrinsics:
    data = _mapping(
        "intrinsics",
        value,
        {
            "width",
            "height",
            "fx",
            "fy",
            "cx",
            "cy",
            "distortion_model",
            "distortion",
        },
    )
    return CameraIntrinsics(
        width=_integer("intrinsics.width", data["width"]),
        height=_integer("intrinsics.height", data["height"]),
        fx=_number("intrinsics.fx", data["fx"]),
        fy=_number("intrinsics.fy", data["fy"]),
        cx=_number("intrinsics.cx", data["cx"]),
        cy=_number("intrinsics.cy", data["cy"]),
        distortion_model=_string(
            "intrinsics.distortion_model",
            data["distortion_model"],
        ),
        distortion=tuple(
            _number("intrinsics.distortion[]", item)
            for item in _sequence("intrinsics.distortion", data["distortion"])
        ),
    )


def _parse_extrinsics(value: Any) -> CameraExtrinsics:
    data = _mapping(
        "extrinsics",
        value,
        {"source_frame", "target_frame", "translation_m", "rotation_xyzw"},
    )
    return CameraExtrinsics(
        source_frame=_string("extrinsics.source_frame", data["source_frame"]),
        target_frame=_string("extrinsics.target_frame", data["target_frame"]),
        translation_m=tuple(
            _number("extrinsics.translation_m[]", item)
            for item in _sequence(
                "extrinsics.translation_m",
                data["translation_m"],
                3,
            )
        ),
        rotation_xyzw=tuple(
            _number("extrinsics.rotation_xyzw[]", item)
            for item in _sequence(
                "extrinsics.rotation_xyzw",
                data["rotation_xyzw"],
                4,
            )
        ),
    )


def _parse_provenance(value: Any) -> FrameProvenance:
    data = _mapping(
        "provenance",
        value,
        {
            "state",
            "source_host",
            "source_pipeline",
            "model_id",
            "frame_id",
            "calibration_id",
            "image_supplied",
            "capture_timestamp_s",
            "inference_timestamp_s",
            "model_sha256",
            "input_sha256",
        },
    )
    if not isinstance(data["image_supplied"], bool):
        raise PayloadError("provenance.image_supplied must be boolean")
    for digest_name in ("model_sha256", "input_sha256"):
        if data[digest_name] is not None and not isinstance(
            data[digest_name],
            str,
        ):
            raise PayloadError(f"provenance.{digest_name} must be string or null")
    return FrameProvenance(
        state=ProvenanceState(_string("provenance.state", data["state"])),
        source_host=_string("provenance.source_host", data["source_host"]),
        source_pipeline=_string(
            "provenance.source_pipeline",
            data["source_pipeline"],
        ),
        model_id=_string("provenance.model_id", data["model_id"]),
        frame_id=_string("provenance.frame_id", data["frame_id"]),
        calibration_id=_string(
            "provenance.calibration_id",
            data["calibration_id"],
        ),
        image_supplied=data["image_supplied"],
        capture_timestamp_s=_number(
            "provenance.capture_timestamp_s",
            data["capture_timestamp_s"],
        ),
        inference_timestamp_s=_number(
            "provenance.inference_timestamp_s",
            data["inference_timestamp_s"],
        ),
        model_sha256=data["model_sha256"],
        input_sha256=data["input_sha256"],
    )


def _parse_quality(value: Any) -> QualityMetrics:
    data = _mapping(
        "quality",
        value,
        {
            "overall_score",
            "projection_valid_fraction",
            "visible_fraction",
            "mean_confidence",
            "dropped_input_fraction",
        },
    )
    return QualityMetrics(
        overall_score=_number("quality.overall_score", data["overall_score"]),
        projection_valid_fraction=_number(
            "quality.projection_valid_fraction",
            data["projection_valid_fraction"],
        ),
        visible_fraction=_number(
            "quality.visible_fraction",
            data["visible_fraction"],
        ),
        mean_confidence=_number(
            "quality.mean_confidence",
            data["mean_confidence"],
        ),
        dropped_input_fraction=_number(
            "quality.dropped_input_fraction",
            data["dropped_input_fraction"],
        ),
    )


def _parse_tokens(value: Any, limits: PayloadLimits) -> tuple[SparseVectorToken, ...]:
    values = _sequence("vector_tokens", value)
    if len(values) > limits.max_tokens:
        raise PayloadError("payload exceeds the configured token limit")
    tokens: list[SparseVectorToken] = []
    keys = {
        "token_id",
        "kind",
        "class_id",
        "confidence",
        "points_xy_m",
        "velocity_xy_mps",
        "track_id",
    }
    for index, item in enumerate(values):
        data = _mapping(f"vector_tokens[{index}]", item, keys)
        points_raw = _sequence(
            f"vector_tokens[{index}].points_xy_m",
            data["points_xy_m"],
        )
        if len(points_raw) > limits.max_points_per_token:
            raise PayloadError("payload exceeds the configured token point limit")
        points = tuple(
            tuple(
                _number(f"vector_tokens[{index}].points_xy_m[][]", axis)
                for axis in _sequence(
                    f"vector_tokens[{index}].points_xy_m[]",
                    point,
                    2,
                )
            )
            for point in points_raw
        )
        track_id = data["track_id"]
        if track_id is not None and not isinstance(track_id, str):
            raise PayloadError("vector token track_id must be string or null")
        tokens.append(
            SparseVectorToken(
                token_id=_string(
                    f"vector_tokens[{index}].token_id",
                    data["token_id"],
                ),
                kind=VectorTokenKind(
                    _string(
                        f"vector_tokens[{index}].kind",
                        data["kind"],
                    )
                ),
                class_id=_integer(
                    f"vector_tokens[{index}].class_id",
                    data["class_id"],
                ),
                confidence=_number(
                    f"vector_tokens[{index}].confidence",
                    data["confidence"],
                ),
                points_xy_m=points,
                velocity_xy_mps=tuple(
                    _number(
                        f"vector_tokens[{index}].velocity_xy_mps[]",
                        axis,
                    )
                    for axis in _sequence(
                        f"vector_tokens[{index}].velocity_xy_mps",
                        data["velocity_xy_mps"],
                        2,
                    )
                ),
                track_id=track_id,
            )
        )
    return tuple(tokens)


def _validate_static_header(
    header: Mapping[str, Any],
    geometry: BEVGeometryV2,
    raw_bytes: int,
) -> None:
    if header["schema_version"] != SCHEMA_VERSION:
        raise PayloadError("unsupported semantic BEV schema version")
    expected_layout, expected_raw = _array_layout(
        geometry.height,
        geometry.width,
    )
    if header["array_layout"] != expected_layout:
        raise PayloadError("array_layout does not match the fixed codec contract")
    compression = _mapping(
        "compression",
        header["compression"],
        {"codec", "raw_bytes", "probability_encoding"},
    )
    if compression != {
        "codec": "zlib",
        "raw_bytes": expected_raw,
        "probability_encoding": "uint8-linear",
    }:
        raise PayloadError("compression metadata does not match the codec")
    if raw_bytes != expected_raw:
        raise PayloadError("prefix raw length does not match array_layout")
    authority = _mapping(
        "authority",
        header["authority"],
        {"shadow_only", "publishes_motion", "publishes_tf", "writes_serial"},
    )
    if authority != {
        "shadow_only": True,
        "publishes_motion": False,
        "publishes_tf": False,
        "writes_serial": False,
    }:
        raise PayloadError("payload authority must remain read-only shadow")


def _decode_arrays(
    raw: bytes,
    geometry: BEVGeometryV2,
) -> dict[str, np.ndarray]:
    layout, expected_raw = _array_layout(geometry.height, geometry.width)
    if len(raw) != expected_raw:
        raise PayloadError("raw dense array length is invalid")
    arrays: dict[str, np.ndarray] = {}
    cells = geometry.height * geometry.width
    for entry in layout:
        start = int(entry["offset"])
        end = start + int(entry["nbytes"])
        chunk = raw[start:end]
        name = str(entry["name"])
        if name == "visibility":
            packed = np.frombuffer(chunk, dtype=np.uint8)
            unpacked = np.unpackbits(
                packed,
                bitorder="little",
                count=cells,
            )
            arrays[name] = unpacked.reshape(geometry.shape).astype(np.bool_)
        else:
            values = np.frombuffer(chunk, dtype=np.uint8, count=cells).reshape(
                geometry.shape
            )
            if name in {
                "semantic_risk",
                "confidence",
                "dynamic_probability",
            }:
                arrays[name] = values.astype(np.float32) / 255.0
            else:
                arrays[name] = values.copy()
    return arrays


def decode_payload(
    payload: bytes | bytearray | memoryview,
    *,
    limits: PayloadLimits | None = None,
) -> SemanticBEVFrame:
    """Decode and fully validate a bounded cross-X5 payload."""

    cfg = limits or PayloadLimits()
    if not isinstance(cfg, PayloadLimits):
        raise TypeError("limits must be PayloadLimits")
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("payload must be bytes-like")
    data = bytes(payload)
    if len(data) > cfg.max_payload_bytes:
        raise PayloadError("payload exceeds max_payload_bytes")
    if len(data) < _PREFIX.size:
        raise PayloadError("payload is shorter than its fixed prefix")

    magic, header_size, compressed_size, raw_size, expected_digest = (
        _PREFIX.unpack_from(data)
    )
    if magic != MAGIC:
        raise PayloadError("payload magic or version is invalid")
    if header_size <= 0 or header_size > cfg.max_header_bytes:
        raise PayloadError("header length is outside allowed bounds")
    if compressed_size <= 0:
        raise PayloadError("compressed body must not be empty")
    if raw_size <= 0 or raw_size > cfg.max_raw_bytes:
        raise PayloadError("raw body length is outside allowed bounds")
    expected_total = _PREFIX.size + header_size + compressed_size
    if expected_total != len(data):
        raise PayloadError("payload length does not match its prefix")

    header_bytes = data[_PREFIX.size : _PREFIX.size + header_size]
    compressed = data[_PREFIX.size + header_size :]
    try:
        header = json.loads(header_bytes.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PayloadError("header is not valid canonical ASCII JSON") from exc
    if not isinstance(header, dict) or set(header) != _TOP_LEVEL_KEYS:
        raise PayloadError("payload header has unexpected or missing fields")
    if _canonical_json(header) != header_bytes:
        raise PayloadError("payload header is not in canonical JSON form")

    raw = _bounded_decompress(
        compressed,
        declared_raw_bytes=raw_size,
        max_raw_bytes=cfg.max_raw_bytes,
    )
    actual_digest = hashlib.sha256(header_bytes + raw).digest()
    if not hmac.compare_digest(actual_digest, expected_digest):
        raise PayloadError("payload integrity digest mismatch")

    geometry = _parse_geometry(header["geometry"])
    _validate_static_header(header, geometry, raw_size)
    arrays = _decode_arrays(raw, geometry)
    try:
        return SemanticBEVFrame(
            timestamp_s=_number("timestamp_s", header["timestamp_s"]),
            intrinsics=_parse_intrinsics(header["intrinsics"]),
            extrinsics=_parse_extrinsics(header["extrinsics"]),
            provenance=_parse_provenance(header["provenance"]),
            quality=_parse_quality(header["quality"]),
            semantic_risk=arrays["semantic_risk"],
            confidence=arrays["confidence"],
            dynamic_probability=arrays["dynamic_probability"],
            class_ids=arrays["class_ids"],
            visibility=arrays["visibility"],
            vector_tokens=_parse_tokens(header["vector_tokens"], cfg),
            geometry=geometry,
        )
    except PayloadError:
        raise
    except (ContractError, ValueError) as exc:
        raise PayloadError("decoded frame violates the semantic BEV contract") from exc


__all__ = [
    "MAGIC",
    "PayloadError",
    "decode_payload",
    "encode_payload",
]
