#!/usr/bin/env python3
"""Strict NPZ episode contract for the X5-TriBEV-Flow candidate.

The metadata and validation helpers require NumPy but never require PyTorch.
PyTorch is imported opportunistically so the same module remains usable on
lightweight data-audit hosts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset as _TorchDatasetBase

    TORCH_AVAILABLE = True
    _TORCH_IMPORT_ERROR = ""
except (ImportError, OSError) as exc:
    torch = None
    _TorchDatasetBase = object
    TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = repr(exc)


SCHEMA_VERSION = "x5-tribev-episode.v1"
HISTORY_FRAMES = 5
FUTURE_HORIZONS = 3
TRAJECTORY_TOKENS = 9
GRID_HEIGHT = 64
GRID_WIDTH = 64
GRID_RESOLUTION_M = 0.10
GRID_X_MIN_M = -1.20
GRID_Y_MIN_M = -3.20

TRIBEV_CHANNEL_NAMES = (
    "lidar_occupancy",
    "lidar_visibility",
    "depth_near",
    "depth_mid",
    "depth_far",
    "camera_semantic_risk",
    "sensor_validity_fraction",
    "fused_occupancy",
)
MODEL_INPUT_CHANNELS = HISTORY_FRAMES * len(TRIBEV_CHANNEL_NAMES)
SENSOR_NAMES = ("lidar", "depth", "vision_4k")
PROVENANCE_STATES = (
    "live_sensor",
    "live_camera",
    "synthetic",
    "modality_dropout",
    "unavailable",
    "cached_camera",
    "fixture_prior",
)
SPLIT_NAMES = ("train", "validation", "calibration", "test")
DEFAULT_SPLIT_RATIOS = (0.50, 0.20, 0.15, 0.15)
FUTURE_HORIZON_SECONDS = (0.4, 0.8, 1.2)
TRAJECTORY_TOKEN_OMEGA_RAD_S = (
    -0.80,
    -0.55,
    -0.30,
    -0.12,
    0.0,
    0.12,
    0.30,
    0.55,
    0.80,
)
MODEL_FLATTEN_ORDER = "reverse_chronological_frame_major_then_channel"


class EpisodeValidationError(ValueError):
    """Raised when an episode violates the fixed v1 data contract."""


@dataclass(frozen=True)
class EpisodeRef:
    """Metadata-only episode reference used for leakage-safe splitting."""

    path: Path
    episode_id: str
    session_id: str
    scenario_id: str
    source_kind: str


def expected_array_specs() -> dict[str, dict[str, Any]]:
    """Return the normative NPZ member contract for schema v1."""

    return {
        "schema_version": {"dtype": "unicode", "shape": []},
        "episode_id": {"dtype": "unicode", "shape": []},
        "session_id": {"dtype": "unicode", "shape": []},
        "scenario_id": {"dtype": "unicode", "shape": []},
        "metadata_json": {"dtype": "unicode", "shape": []},
        "timestamps_ns": {"dtype": "int64", "shape": [HISTORY_FRAMES]},
        "history_offsets_s": {"dtype": "float32", "shape": [HISTORY_FRAMES]},
        "future_timestamps_ns": {"dtype": "int64", "shape": [FUTURE_HORIZONS]},
        "future_horizons_s": {"dtype": "float32", "shape": [FUTURE_HORIZONS]},
        "tribev_input": {
            "dtype": "float32",
            "shape": [
                HISTORY_FRAMES,
                len(TRIBEV_CHANNEL_NAMES),
                GRID_HEIGHT,
                GRID_WIDTH,
            ],
        },
        "future_occupancy": {
            "dtype": "float32",
            "shape": [FUTURE_HORIZONS, GRID_HEIGHT, GRID_WIDTH],
        },
        "future_flow_m": {
            "dtype": "float32",
            "shape": [FUTURE_HORIZONS, 2, GRID_HEIGHT, GRID_WIDTH],
        },
        "dynamic_mask": {
            "dtype": "float32",
            "shape": [FUTURE_HORIZONS, GRID_HEIGHT, GRID_WIDTH],
        },
        "uncertainty_target": {
            "dtype": "float32",
            "shape": [FUTURE_HORIZONS, GRID_HEIGHT, GRID_WIDTH],
        },
        "trajectory_soft_labels": {
            "dtype": "float32",
            "shape": [TRAJECTORY_TOKENS],
        },
        "trajectory_token_omega_rad_s": {
            "dtype": "float32",
            "shape": [TRAJECTORY_TOKENS],
        },
        "sensor_validity": {
            "dtype": "uint8",
            "shape": [HISTORY_FRAMES, len(SENSOR_NAMES)],
        },
        "sensor_age_s": {
            "dtype": "float32",
            "shape": [HISTORY_FRAMES, len(SENSOR_NAMES)],
        },
        "sensor_provenance": {
            "dtype": "unicode",
            "shape": [HISTORY_FRAMES, len(SENSOR_NAMES)],
        },
        "vision_image_supplied": {"dtype": "uint8", "shape": [HISTORY_FRAMES]},
    }


def build_episode_metadata(
    *,
    episode_id: str,
    session_id: str,
    scenario_id: str,
    source: Mapping[str, Any],
    generator: Mapping[str, Any],
    seed: int | None = None,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Build metadata that mirrors the fixed NPZ array contract."""

    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "session_id": session_id,
        "scenario_id": scenario_id,
        "source": dict(source),
        "generator": dict(generator),
        "grid": {
            "frame_id": "base_link",
            "height": GRID_HEIGHT,
            "width": GRID_WIDTH,
            "resolution_m": GRID_RESOLUTION_M,
            "x_min_m": GRID_X_MIN_M,
            "x_max_m": GRID_X_MIN_M + GRID_HEIGHT * GRID_RESOLUTION_M,
            "y_min_m": GRID_Y_MIN_M,
            "y_max_m": GRID_Y_MIN_M + GRID_WIDTH * GRID_RESOLUTION_M,
            "row_axis": "+x_forward",
            "column_axis": "+y_left",
        },
        "history": {
            "count": HISTORY_FRAMES,
            "channel_names": list(TRIBEV_CHANNEL_NAMES),
            "reference": "last_history_frame",
            "model_input_shape": [1, MODEL_INPUT_CHANNELS, GRID_HEIGHT, GRID_WIDTH],
            "storage_order": "oldest_to_newest_strict_timestamp_order",
            "model_history_order": "newest_to_oldest_t0_first",
            "model_flatten_order": MODEL_FLATTEN_ORDER,
        },
        "future": {
            "count": FUTURE_HORIZONS,
            "horizons_s": list(FUTURE_HORIZON_SECONDS),
            "flow_definition": (
                "At each future dynamic occupied cell, future_flow_m stores "
                "the x/y displacement in metres from the object's reference-time "
                "position to that horizon; static and invalid cells are zero."
            ),
        },
        "trajectory_tokens": {
            "count": TRAJECTORY_TOKENS,
            "label_semantics": "probability_distribution_over_fixed_arc_tokens",
            "omega_rad_s": list(TRAJECTORY_TOKEN_OMEGA_RAD_S),
        },
        "sensors": {
            "names": list(SENSOR_NAMES),
            "provenance_states": list(PROVENANCE_STATES),
            "live_4k_rule": (
                "vision_4k is real live input only when sensor_validity=1, "
                "sensor_provenance=live_camera, and vision_image_supplied=1"
            ),
        },
        "arrays": expected_array_specs(),
        "notes": list(notes),
    }
    if seed is not None:
        metadata["seed"] = int(seed)
    return metadata


def _scalar_text(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind != "U":
        raise EpisodeValidationError(f"{name} must be a scalar Unicode array")
    return str(array.item())


def _parse_metadata_array(value: np.ndarray) -> dict[str, Any]:
    text = _scalar_text(value, "metadata_json")
    try:
        metadata = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EpisodeValidationError(f"metadata_json is invalid JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise EpisodeValidationError("metadata_json root must be an object")
    return metadata


def _validate_metadata(metadata: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "episode_id",
        "session_id",
        "scenario_id",
        "source",
        "generator",
        "grid",
        "history",
        "future",
        "trajectory_tokens",
        "sensors",
        "arrays",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise EpisodeValidationError(f"metadata_json missing fields: {missing}")
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise EpisodeValidationError(
            f"unsupported schema_version={metadata['schema_version']!r}"
        )
    for name in ("episode_id", "session_id", "scenario_id"):
        value = metadata[name]
        if not isinstance(value, str) or not value.strip():
            raise EpisodeValidationError(f"metadata.{name} must be non-empty text")
        if any(char.isspace() for char in value):
            raise EpisodeValidationError(f"metadata.{name} must not contain whitespace")

    source = metadata["source"]
    if not isinstance(source, dict):
        raise EpisodeValidationError("metadata.source must be an object")
    source_required = {
        "kind",
        "license_id",
        "contains_personal_data",
        "consent_status",
    }
    source_missing = sorted(source_required - set(source))
    if source_missing:
        raise EpisodeValidationError(
            f"metadata.source missing fields: {source_missing}"
        )
    if source["kind"] not in {"real", "synthetic", "replay"}:
        raise EpisodeValidationError("metadata.source.kind must be real/synthetic/replay")
    if not isinstance(source["contains_personal_data"], bool):
        raise EpisodeValidationError(
            "metadata.source.contains_personal_data must be boolean"
        )

    grid = metadata["grid"]
    expected_grid = {
        "frame_id": "base_link",
        "height": GRID_HEIGHT,
        "width": GRID_WIDTH,
        "resolution_m": GRID_RESOLUTION_M,
        "x_min_m": GRID_X_MIN_M,
        "y_min_m": GRID_Y_MIN_M,
    }
    for key, expected in expected_grid.items():
        if grid.get(key) != expected:
            raise EpisodeValidationError(
                f"metadata.grid.{key}={grid.get(key)!r}, expected {expected!r}"
            )

    history = metadata["history"]
    if history.get("count") != HISTORY_FRAMES:
        raise EpisodeValidationError("metadata.history.count must be 5")
    if history.get("channel_names") != list(TRIBEV_CHANNEL_NAMES):
        raise EpisodeValidationError("metadata.history.channel_names mismatch")
    if history.get("model_input_shape") != [
        1,
        MODEL_INPUT_CHANNELS,
        GRID_HEIGHT,
        GRID_WIDTH,
    ]:
        raise EpisodeValidationError(
            "metadata.history.model_input_shape must be [1, 40, 64, 64]"
        )
    if history.get("model_flatten_order") != MODEL_FLATTEN_ORDER:
        raise EpisodeValidationError("metadata.history.model_flatten_order mismatch")
    if history.get("storage_order") != "oldest_to_newest_strict_timestamp_order":
        raise EpisodeValidationError("metadata.history.storage_order mismatch")
    if history.get("model_history_order") != "newest_to_oldest_t0_first":
        raise EpisodeValidationError("metadata.history.model_history_order mismatch")

    future = metadata["future"]
    if future.get("count") != FUTURE_HORIZONS:
        raise EpisodeValidationError("metadata.future.count must be 3")
    if future.get("horizons_s") != list(FUTURE_HORIZON_SECONDS):
        raise EpisodeValidationError("metadata.future.horizons_s mismatch")
    if metadata["trajectory_tokens"].get("count") != TRAJECTORY_TOKENS:
        raise EpisodeValidationError("metadata.trajectory_tokens.count must be 9")
    if metadata["trajectory_tokens"].get("omega_rad_s") != list(
        TRAJECTORY_TOKEN_OMEGA_RAD_S
    ):
        raise EpisodeValidationError("metadata.trajectory_tokens.omega_rad_s mismatch")
    if metadata["sensors"].get("names") != list(SENSOR_NAMES):
        raise EpisodeValidationError("metadata.sensors.names mismatch")
    if metadata["arrays"] != expected_array_specs():
        raise EpisodeValidationError("metadata.arrays does not match schema v1")


def _check_shape_and_dtype(
    name: str, value: np.ndarray, spec: Mapping[str, Any]
) -> None:
    array = np.asarray(value)
    expected_shape = tuple(spec["shape"])
    if array.shape != expected_shape:
        raise EpisodeValidationError(
            f"{name}.shape={array.shape}, expected {expected_shape}"
        )
    expected_dtype = spec["dtype"]
    if expected_dtype == "unicode":
        if array.dtype.kind != "U":
            raise EpisodeValidationError(
                f"{name}.dtype={array.dtype}, expected Unicode"
            )
    elif array.dtype != np.dtype(expected_dtype):
        raise EpisodeValidationError(
            f"{name}.dtype={array.dtype}, expected {expected_dtype}"
        )
    if array.dtype.kind == "O":
        raise EpisodeValidationError(f"{name} must never use object dtype")


def _check_unit_interval(name: str, value: np.ndarray) -> None:
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise EpisodeValidationError(f"{name} contains NaN or Inf")
    if float(array.min()) < -1e-6 or float(array.max()) > 1.0 + 1e-6:
        raise EpisodeValidationError(f"{name} must stay in [0, 1]")


def validate_episode_payload(
    payload: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Validate a fully loaded NPZ payload and return a compact summary."""

    expected_specs = expected_array_specs()
    actual_names = set(payload)
    expected_names = set(expected_specs)
    missing = sorted(expected_names - actual_names)
    extras = sorted(actual_names - expected_names)
    if missing or extras:
        raise EpisodeValidationError(
            f"NPZ member mismatch: missing={missing}, extras={extras}"
        )
    for name, spec in expected_specs.items():
        _check_shape_and_dtype(name, np.asarray(payload[name]), spec)

    metadata = _parse_metadata_array(np.asarray(payload["metadata_json"]))
    _validate_metadata(metadata)
    for name in ("schema_version", "episode_id", "session_id", "scenario_id"):
        array_value = _scalar_text(np.asarray(payload[name]), name)
        if array_value != metadata[name]:
            raise EpisodeValidationError(
                f"{name} array disagrees with metadata_json: "
                f"{array_value!r} != {metadata[name]!r}"
            )

    timestamps_ns = np.asarray(payload["timestamps_ns"])
    future_timestamps_ns = np.asarray(payload["future_timestamps_ns"])
    if np.any(np.diff(timestamps_ns) <= 0):
        raise EpisodeValidationError("timestamps_ns must be strictly increasing")
    if np.any(np.diff(future_timestamps_ns) <= 0):
        raise EpisodeValidationError(
            "future_timestamps_ns must be strictly increasing"
        )
    if int(future_timestamps_ns[0]) <= int(timestamps_ns[-1]):
        raise EpisodeValidationError(
            "all future timestamps must follow the reference history frame"
        )

    history_offsets_s = np.asarray(payload["history_offsets_s"])
    future_horizons_s = np.asarray(payload["future_horizons_s"])
    expected_history_offsets = (
        timestamps_ns.astype(np.float64) - float(timestamps_ns[-1])
    ) / 1e9
    expected_future_horizons = (
        future_timestamps_ns.astype(np.float64) - float(timestamps_ns[-1])
    ) / 1e9
    if not np.allclose(history_offsets_s, expected_history_offsets, atol=1e-6):
        raise EpisodeValidationError(
            "history_offsets_s disagrees with timestamps_ns"
        )
    if not np.allclose(future_horizons_s, expected_future_horizons, atol=1e-6):
        raise EpisodeValidationError(
            "future_horizons_s disagrees with future_timestamps_ns"
        )
    if not np.isclose(float(history_offsets_s[-1]), 0.0, atol=1e-7):
        raise EpisodeValidationError("last history offset must be zero")
    if np.any(future_horizons_s <= 0):
        raise EpisodeValidationError("future_horizons_s must be positive")
    if not np.allclose(
        future_horizons_s,
        np.asarray(FUTURE_HORIZON_SECONDS, dtype=np.float32),
        atol=1e-7,
    ):
        raise EpisodeValidationError(
            "future_horizons_s must be [0.4, 0.8, 1.2]"
        )

    for name in (
        "tribev_input",
        "future_occupancy",
        "dynamic_mask",
        "uncertainty_target",
    ):
        _check_unit_interval(name, np.asarray(payload[name]))
    future_flow = np.asarray(payload["future_flow_m"])
    if not np.isfinite(future_flow).all():
        raise EpisodeValidationError("future_flow_m contains NaN or Inf")
    if float(np.abs(future_flow).max()) > 10.0:
        raise EpisodeValidationError("future_flow_m exceeds the 10 m sanity bound")

    dynamic_mask = np.asarray(payload["dynamic_mask"])
    future_occupancy = np.asarray(payload["future_occupancy"])
    if np.any(dynamic_mask > future_occupancy + 1e-5):
        raise EpisodeValidationError(
            "dynamic_mask must be a subset of future_occupancy"
        )
    flow_outside_dynamic = np.abs(future_flow) * (
        1.0 - dynamic_mask[:, np.newaxis, :, :]
    )
    if float(flow_outside_dynamic.max()) > 1e-5:
        raise EpisodeValidationError(
            "future_flow_m must be zero outside dynamic_mask"
        )

    soft_labels = np.asarray(payload["trajectory_soft_labels"])
    if not np.isfinite(soft_labels).all() or np.any(soft_labels < 0):
        raise EpisodeValidationError(
            "trajectory_soft_labels must be finite and non-negative"
        )
    if not np.isclose(float(soft_labels.sum()), 1.0, atol=1e-5):
        raise EpisodeValidationError("trajectory_soft_labels must sum to 1")
    omega = np.asarray(payload["trajectory_token_omega_rad_s"])
    if not np.isfinite(omega).all() or np.any(np.diff(omega) <= 0):
        raise EpisodeValidationError(
            "trajectory_token_omega_rad_s must be finite and strictly increasing"
        )
    if not np.allclose(
        omega,
        np.asarray(TRAJECTORY_TOKEN_OMEGA_RAD_S, dtype=np.float32),
        atol=1e-7,
    ):
        raise EpisodeValidationError("trajectory_token_omega_rad_s mismatch")

    validity = np.asarray(payload["sensor_validity"])
    image_supplied = np.asarray(payload["vision_image_supplied"])
    if not np.isin(validity, (0, 1)).all():
        raise EpisodeValidationError("sensor_validity must contain only 0/1")
    if not np.isin(image_supplied, (0, 1)).all():
        raise EpisodeValidationError("vision_image_supplied must contain only 0/1")
    sensor_age_s = np.asarray(payload["sensor_age_s"])
    if not np.isfinite(sensor_age_s).all():
        raise EpisodeValidationError("sensor_age_s must be finite")
    if np.any((validity == 1) & (sensor_age_s < 0)):
        raise EpisodeValidationError(
            "valid sensors must have non-negative sensor_age_s"
        )
    if np.any((validity == 0) & (sensor_age_s != -1.0)):
        raise EpisodeValidationError(
            "invalid sensors must use sensor_age_s=-1"
        )

    provenance = np.asarray(payload["sensor_provenance"])
    unknown_provenance = sorted(set(provenance.ravel()) - set(PROVENANCE_STATES))
    if unknown_provenance:
        raise EpisodeValidationError(
            f"unknown sensor provenance values: {unknown_provenance}"
        )

    tribev = np.asarray(payload["tribev_input"])
    sensor_channels = {
        0: (0, 1),
        1: (2, 3, 4),
        2: (5,),
    }
    for history_index in range(HISTORY_FRAMES):
        for sensor_index, channels in sensor_channels.items():
            valid = bool(validity[history_index, sensor_index])
            if not valid and float(np.abs(tribev[history_index, channels]).max()) > 1e-6:
                raise EpisodeValidationError(
                    f"invalid {SENSOR_NAMES[sensor_index]} input must be zero "
                    f"at history index {history_index}"
                )
        validity_fraction_plane = tribev[history_index, 6]
        expected_fraction = float(np.mean(validity[history_index]))
        valid_fraction_cells = np.isclose(
            validity_fraction_plane, expected_fraction, atol=1e-6
        )
        uncovered_cells = np.isclose(validity_fraction_plane, 0.0, atol=1e-6)
        if not np.all(valid_fraction_cells | uncovered_cells):
            raise EpisodeValidationError(
                "sensor_validity_fraction cells must be zero where an "
                "ego-warped history frame has no coverage, or equal that "
                "frame's three-source validity fraction"
            )
        expected_fused = np.maximum.reduce(
            (
                tribev[history_index, 0],
                tribev[history_index, 2],
                tribev[history_index, 3],
                tribev[history_index, 4],
                tribev[history_index, 5],
            )
        )
        if not np.allclose(tribev[history_index, 7], expected_fused, atol=1e-6):
            raise EpisodeValidationError(
                "fused_occupancy must be derived only from source channels in "
                "the same history frame"
            )

        vision_state = str(provenance[history_index, 2])
        vision_valid = bool(validity[history_index, 2])
        supplied = bool(image_supplied[history_index])
        if vision_state == "live_camera" and not (vision_valid and supplied):
            raise EpisodeValidationError(
                "live_camera requires validity=1 and vision_image_supplied=1"
            )
        if supplied and vision_state != "live_camera":
            raise EpisodeValidationError(
                "vision_image_supplied=1 is reserved for live_camera provenance"
            )
        if vision_state in {
            "cached_camera",
            "fixture_prior",
            "modality_dropout",
            "unavailable",
        } and vision_valid:
            raise EpisodeValidationError(
                f"{vision_state} must not be marked as valid 4K input"
            )

    source_kind = metadata["source"]["kind"]
    if source_kind == "real":
        for history_index in range(HISTORY_FRAMES):
            if validity[history_index, 2]:
                if (
                    provenance[history_index, 2] != "live_camera"
                    or not image_supplied[history_index]
                ):
                    raise EpisodeValidationError(
                        "real valid 4K samples require preserved live_camera "
                        "provenance and image_supplied=true"
                    )

    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": metadata["episode_id"],
        "session_id": metadata["session_id"],
        "scenario_id": metadata["scenario_id"],
        "source_kind": source_kind,
        "real_live_4k_frames": int(
            np.sum(
                (validity[:, 2] == 1)
                & (provenance[:, 2] == "live_camera")
                & (image_supplied == 1)
            )
        ),
    }


def flatten_tribev_history(tribev_input: np.ndarray) -> np.ndarray:
    """Convert chronological storage to newest-first model channel order.

    Episode timestamps stay strictly increasing (oldest to newest) for audit
    and synchronization. The model contract matches the live front end:
    ``t0`` channels first, followed by ``t-1`` through ``t-4``.
    """

    array = np.asarray(tribev_input)
    expected_shape = (
        HISTORY_FRAMES,
        len(TRIBEV_CHANNEL_NAMES),
        GRID_HEIGHT,
        GRID_WIDTH,
    )
    if array.shape != expected_shape:
        raise EpisodeValidationError(
            f"tribev_input.shape={array.shape}, expected {expected_shape}"
        )
    return np.ascontiguousarray(
        array[::-1].reshape(MODEL_INPUT_CHANNELS, GRID_HEIGHT, GRID_WIDTH)
    )


def load_episode(
    path: str | Path, *, validate: bool = True
) -> dict[str, Any]:
    """Load an episode without pickle and optionally enforce the full contract."""

    episode_path = Path(path).expanduser().resolve()
    with np.load(episode_path, allow_pickle=False) as archive:
        payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    summary = validate_episode_payload(payload) if validate else None
    metadata = _parse_metadata_array(payload["metadata_json"])
    return {
        "path": episode_path,
        "arrays": payload,
        "metadata": metadata,
        "summary": summary,
    }


def read_episode_metadata(
    path: str | Path, *, validate_metadata: bool = True
) -> dict[str, Any]:
    """Read only embedded metadata; this path never imports or requires Torch."""

    episode_path = Path(path).expanduser().resolve()
    with np.load(episode_path, allow_pickle=False) as archive:
        if "metadata_json" not in archive.files:
            raise EpisodeValidationError(f"{episode_path} has no metadata_json")
        metadata = _parse_metadata_array(np.asarray(archive["metadata_json"]))
    if validate_metadata:
        _validate_metadata(metadata)
    return metadata


def save_episode(
    path: str | Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    *,
    overwrite: bool = False,
    compressed: bool = True,
) -> Path:
    """Validate and atomically save one NPZ episode."""

    destination = Path(path).expanduser().resolve()
    if destination.suffix.lower() != ".npz":
        raise ValueError("episode path must end in .npz")
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)

    metadata_copy = json.loads(json.dumps(dict(metadata), ensure_ascii=False))
    metadata_copy["arrays"] = expected_array_specs()
    payload = {name: np.asarray(value) for name, value in arrays.items()}
    payload.update(
        {
            "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.str_),
            "episode_id": np.asarray(metadata_copy["episode_id"], dtype=np.str_),
            "session_id": np.asarray(metadata_copy["session_id"], dtype=np.str_),
            "scenario_id": np.asarray(metadata_copy["scenario_id"], dtype=np.str_),
            "metadata_json": np.asarray(
                json.dumps(
                    metadata_copy,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                dtype=np.str_,
            ),
        }
    )
    validate_episode_payload(payload)

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
            writer = np.savez_compressed if compressed else np.savez
            writer(temporary, **payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


def discover_episode_paths(root: str | Path) -> list[Path]:
    """Discover NPZ episodes under a root or return a single NPZ path."""

    candidate = Path(root).expanduser().resolve()
    if candidate.is_file():
        if candidate.suffix.lower() != ".npz":
            raise ValueError(f"not an NPZ episode: {candidate}")
        return [candidate]
    if not candidate.is_dir():
        raise FileNotFoundError(candidate)
    return sorted(path for path in candidate.rglob("*.npz") if path.is_file())


def build_episode_refs(
    root_or_paths: str | Path | Iterable[str | Path],
) -> list[EpisodeRef]:
    """Build metadata-only references without loading tensor arrays."""

    if isinstance(root_or_paths, (str, Path)):
        paths = discover_episode_paths(root_or_paths)
    else:
        paths = [Path(path).expanduser().resolve() for path in root_or_paths]
    refs: list[EpisodeRef] = []
    for path in paths:
        metadata = read_episode_metadata(path)
        refs.append(
            EpisodeRef(
                path=path,
                episode_id=metadata["episode_id"],
                session_id=metadata["session_id"],
                scenario_id=metadata["scenario_id"],
                source_kind=metadata["source"]["kind"],
            )
        )
    episode_ids = [ref.episode_id for ref in refs]
    if len(episode_ids) != len(set(episode_ids)):
        raise EpisodeValidationError("episode_id values must be globally unique")
    return refs


def _normalize_split_ratios(
    ratios: Sequence[float],
) -> tuple[float, ...]:
    if len(ratios) != len(SPLIT_NAMES):
        raise ValueError(
            "split ratios must contain train/validation/calibration/test"
        )
    values = tuple(float(value) for value in ratios)
    if any(value <= 0 for value in values):
        raise ValueError("all split ratios must be positive")
    total = sum(values)
    return tuple(value / total for value in values)  # type: ignore[return-value]


def split_name_for_session(
    session_id: str,
    *,
    ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
    seed: int = 20260728,
) -> str:
    """Assign one session to a stable four-way hash bucket."""

    normalized = _normalize_split_ratios(ratios)
    digest = hashlib.sha256(f"{seed}:{session_id}".encode("utf-8")).digest()
    unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
    cumulative = 0.0
    for name, ratio in zip(SPLIT_NAMES, normalized, strict=True):
        cumulative += ratio
        if unit < cumulative:
            return name
    return SPLIT_NAMES[-1]


def _stratified_session_assignments(
    refs: Sequence[EpisodeRef],
    ratios: Sequence[float],
    seed: int,
) -> dict[str, str]:
    normalized = _normalize_split_ratios(ratios)
    session_scenarios: dict[str, str] = {}
    scenarios: dict[str, set[str]] = {}
    for ref in refs:
        previous = session_scenarios.setdefault(ref.session_id, ref.scenario_id)
        if previous != ref.scenario_id:
            raise EpisodeValidationError(
                f"session {ref.session_id} spans scenarios {previous} and "
                f"{ref.scenario_id}"
            )
        scenarios.setdefault(ref.scenario_id, set()).add(ref.session_id)

    assignments: dict[str, str] = {}
    for scenario_id, session_ids in sorted(scenarios.items()):
        ordered = sorted(
            session_ids,
            key=lambda session_id: hashlib.sha256(
                f"{seed}:{scenario_id}:{session_id}".encode("utf-8")
            ).digest(),
        )
        raw_counts = np.asarray(normalized, dtype=np.float64) * len(ordered)
        counts = np.floor(raw_counts).astype(np.int64)
        remainder = len(ordered) - int(counts.sum())
        fractional_order = np.argsort(
            -(raw_counts - counts),
            kind="stable",
        )
        for index in fractional_order[:remainder]:
            counts[index] += 1
        cursor = 0
        for split_name, count in zip(SPLIT_NAMES, counts, strict=True):
            for session_id in ordered[cursor : cursor + int(count)]:
                assignments[session_id] = split_name
            cursor += int(count)
        if cursor != len(ordered):
            raise AssertionError("stratified split did not consume all sessions")
    return assignments


def episode_input_content_sha256(path: str | Path) -> str:
    """Hash only causal model inputs for cross-split duplicate detection."""
    digest = hashlib.sha256()
    with np.load(Path(path), allow_pickle=False) as archive:
        value = np.ascontiguousarray(archive["tribev_input"])
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def split_episode_refs(
    refs: Sequence[EpisodeRef],
    *,
    ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
    seed: int = 20260728,
) -> dict[str, list[EpisodeRef]]:
    """Stratify sessions by scenario and reject cross-split input duplicates."""

    result = {name: [] for name in SPLIT_NAMES}
    session_assignments = _stratified_session_assignments(refs, ratios, seed)
    for ref in refs:
        split_name = session_assignments[ref.session_id]
        result[split_name].append(ref)
    assert_no_session_leakage(result)
    assert_no_input_content_leakage(result)
    return result


def assert_no_session_leakage(
    splits: Mapping[str, Sequence[EpisodeRef]],
) -> None:
    """Fail if any session_id appears in more than one split."""

    owners: dict[str, str] = {}
    for split_name, refs in splits.items():
        for ref in refs:
            previous = owners.setdefault(ref.session_id, split_name)
            if previous != split_name:
                raise EpisodeValidationError(
                    f"session leakage: {ref.session_id} in {previous} and {split_name}"
                )


def assert_no_input_content_leakage(
    splits: Mapping[str, Sequence[EpisodeRef]],
) -> None:
    """Fail when an identical causal input appears across split boundaries."""
    owners: dict[str, tuple[str, str]] = {}
    for split_name, refs in splits.items():
        for ref in refs:
            digest = episode_input_content_sha256(ref.path)
            previous = owners.setdefault(digest, (split_name, ref.episode_id))
            if previous[0] != split_name:
                raise EpisodeValidationError(
                    "input content leakage: "
                    f"{ref.episode_id} in {split_name} duplicates "
                    f"{previous[1]} in {previous[0]} ({digest})"
                )


def summarize_splits(
    splits: Mapping[str, Sequence[EpisodeRef]],
) -> dict[str, Any]:
    """Produce an auditable split summary."""

    assert_no_session_leakage(splits)
    assert_no_input_content_leakage(splits)
    result: dict[str, Any] = {}
    for split_name in SPLIT_NAMES:
        refs = list(splits.get(split_name, ()))
        scenarios: dict[str, int] = {}
        for ref in refs:
            scenarios[ref.scenario_id] = scenarios.get(ref.scenario_id, 0) + 1
        result[split_name] = {
            "episodes": len(refs),
            "sessions": len({ref.session_id for ref in refs}),
            "scenario_episode_counts": dict(sorted(scenarios.items())),
        }
    result["session_leakage"] = False
    result["input_content_leakage"] = False
    return result


class TriBEVEpisodeDataset(_TorchDatasetBase):
    """Optional PyTorch Dataset over leakage-safe NPZ episodes."""

    def __init__(
        self,
        root_or_paths: str | Path | Iterable[str | Path],
        *,
        split: str | None = None,
        split_ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
        split_seed: int = 20260728,
        validate: bool = True,
        transform: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError(
                "TriBEVEpisodeDataset requires PyTorch; metadata helpers remain "
                f"available without it. Import error: {_TORCH_IMPORT_ERROR}"
            )
        if split is not None and split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}")
        refs = build_episode_refs(root_or_paths)
        if split is not None:
            refs = split_episode_refs(
                refs, ratios=split_ratios, seed=split_seed
            )[split]
        self.refs = refs
        self.validate = bool(validate)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> Any:
        record = load_episode(self.refs[index].path, validate=self.validate)
        arrays = record["arrays"]
        model_input = flatten_tribev_history(arrays["tribev_input"])
        item: dict[str, Any] = {
            "tribev_input": torch.from_numpy(arrays["tribev_input"]),
            "model_input": torch.from_numpy(model_input),
            "targets": {
                "future_occupancy": torch.from_numpy(arrays["future_occupancy"]),
                "future_flow_m": torch.from_numpy(arrays["future_flow_m"]),
                "dynamic_mask": torch.from_numpy(arrays["dynamic_mask"]),
                "uncertainty_target": torch.from_numpy(
                    arrays["uncertainty_target"]
                ),
                "trajectory_soft_labels": torch.from_numpy(
                    arrays["trajectory_soft_labels"]
                ),
            },
            "aux": {
                "timestamps_ns": torch.from_numpy(arrays["timestamps_ns"]),
                "history_offsets_s": torch.from_numpy(
                    arrays["history_offsets_s"]
                ),
                "future_timestamps_ns": torch.from_numpy(
                    arrays["future_timestamps_ns"]
                ),
                "future_horizons_s": torch.from_numpy(
                    arrays["future_horizons_s"]
                ),
                "trajectory_token_omega_rad_s": torch.from_numpy(
                    arrays["trajectory_token_omega_rad_s"]
                ),
                "sensor_validity": torch.from_numpy(arrays["sensor_validity"]),
                "sensor_age_s": torch.from_numpy(arrays["sensor_age_s"]),
                "vision_image_supplied": torch.from_numpy(
                    arrays["vision_image_supplied"]
                ),
            },
            "metadata": {
                **record["metadata"],
                "sensor_provenance": arrays["sensor_provenance"].tolist(),
                "episode_path": str(record["path"]),
            },
        }
        return self.transform(item) if self.transform else item


def _validate_command(path: Path) -> dict[str, Any]:
    paths = discover_episode_paths(path)
    summaries = []
    failures = []
    for episode_path in paths:
        try:
            summaries.append(load_episode(episode_path, validate=True)["summary"])
        except Exception as exc:  # CLI must report every bad episode.
            failures.append({"path": str(episode_path), "error": str(exc)})
    return {
        "ok": not failures and bool(paths),
        "episodes": len(paths),
        "valid": len(summaries),
        "failures": failures,
        "source_kinds": sorted({item["source_kind"] for item in summaries}),
        "sessions": len({item["session_id"] for item in summaries}),
        "scenarios": sorted({item["scenario_id"] for item in summaries}),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("path", type=Path)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("episode", type=Path)

    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("path", type=Path)
    split_parser.add_argument("--seed", type=int, default=20260728)
    split_parser.add_argument(
        "--ratios", nargs=4, type=float, default=DEFAULT_SPLIT_RATIOS
    )

    args = parser.parse_args(argv)
    if args.command == "validate":
        result = _validate_command(args.path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["ok"] else 1
    if args.command == "inspect":
        metadata = read_episode_metadata(args.episode)
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0
    if args.command == "split":
        refs = build_episode_refs(args.path)
        splits = split_episode_refs(refs, ratios=args.ratios, seed=args.seed)
        result = {
            "seed": args.seed,
            "ratios": list(_normalize_split_ratios(args.ratios)),
            "summary": summarize_splits(splits),
            "episodes": {
                name: [str(ref.path) for ref in splits[name]]
                for name in SPLIT_NAMES
            },
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
