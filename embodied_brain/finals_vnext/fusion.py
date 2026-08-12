"""Shared-BEV fusion for the passive finals vNext candidate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from embodied_brain.finals_successor.x5_tribev_flow.contracts import OdometryDelta
from embodied_brain.finals_successor.x5_tribev_flow.tribev import warp_bev_nearest

from .contracts import (
    CHANNELS_PER_FRAME,
    FRAME_CHANNEL_NAMES,
    HISTORY_FRAMES,
    MODEL_INPUT_SHAPE,
    BEVGeometryV2,
    FusionFrameV2,
)

_CHANNEL_INDEX = {name: index for index, name in enumerate(FRAME_CHANNEL_NAMES)}


def _probability_grid(
    value: np.ndarray | None,
    geometry: BEVGeometryV2,
    name: str,
) -> np.ndarray:
    if value is None:
        return np.zeros(geometry.shape, dtype=np.float32)
    array = np.asarray(value, dtype=np.float32)
    if array.shape != geometry.shape:
        raise ValueError(f"{name} must have shape {geometry.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must be in [0, 1]")
    return np.ascontiguousarray(array)


def _bounded_rate_grid(
    value: np.ndarray | None,
    geometry: BEVGeometryV2,
) -> np.ndarray:
    if value is None:
        return np.zeros(geometry.shape, dtype=np.float32)
    array = np.asarray(value, dtype=np.float32)
    if array.shape != geometry.shape:
        raise ValueError(f"depth_closing_rate must have shape {geometry.shape}")
    if not np.isfinite(array).all():
        raise ValueError("depth_closing_rate contains non-finite values")
    return np.clip(array, -1.0, 1.0).astype(np.float32, copy=False)


@dataclass(frozen=True, slots=True)
class FusionInputsV2:
    """Normalized modality grids for one robot-centric frame."""

    timestamp_s: float
    lidar_occupancy: np.ndarray | None = None
    lidar_visibility: np.ndarray | None = None
    depth_hit_low: np.ndarray | None = None
    depth_hit_mid: np.ndarray | None = None
    depth_hit_high: np.ndarray | None = None
    depth_free: np.ndarray | None = None
    depth_unknown: np.ndarray | None = None
    depth_closing_rate: np.ndarray | None = None
    camera_semantic_risk: np.ndarray | None = None
    camera_visibility: np.ndarray | None = None
    lidar_validity: float = 0.0
    depth_validity: float = 0.0
    vision_validity: float = 0.0


def build_fusion_frame(
    inputs: FusionInputsV2,
    geometry: BEVGeometryV2 | None = None,
) -> FusionFrameV2:
    """Build a truthful 12-channel frame without treating unknown as free."""

    geom = geometry or BEVGeometryV2()
    validity = (
        float(inputs.lidar_validity),
        float(inputs.depth_validity),
        float(inputs.vision_validity),
    )
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in validity):
        raise ValueError("modality validity values must be in [0, 1]")

    grids: dict[str, np.ndarray] = {
        "lidar_occupancy": _probability_grid(
            inputs.lidar_occupancy, geom, "lidar_occupancy"
        ),
        "lidar_visibility": _probability_grid(
            inputs.lidar_visibility, geom, "lidar_visibility"
        ),
        "depth_hit_low": _probability_grid(
            inputs.depth_hit_low, geom, "depth_hit_low"
        ),
        "depth_hit_mid": _probability_grid(
            inputs.depth_hit_mid, geom, "depth_hit_mid"
        ),
        "depth_hit_high": _probability_grid(
            inputs.depth_hit_high, geom, "depth_hit_high"
        ),
        "depth_free": _probability_grid(inputs.depth_free, geom, "depth_free"),
        "depth_unknown": _probability_grid(
            inputs.depth_unknown, geom, "depth_unknown"
        ),
        "depth_closing_rate": _bounded_rate_grid(
            inputs.depth_closing_rate, geom
        ),
        "camera_semantic_risk": _probability_grid(
            inputs.camera_semantic_risk, geom, "camera_semantic_risk"
        ),
        "camera_visibility": _probability_grid(
            inputs.camera_visibility, geom, "camera_visibility"
        ),
    }
    grids["depth_unknown"] = np.maximum(
        grids["depth_unknown"],
        np.float32(1.0 - validity[1]),
    )

    depth_hit = np.maximum.reduce(
        (
            grids["depth_hit_low"],
            grids["depth_hit_mid"],
            grids["depth_hit_high"],
        )
    )
    validity_fraction = np.full(
        geom.shape,
        sum(validity) / 3.0,
        dtype=np.float32,
    )
    fused_occupancy = np.maximum.reduce(
        (
            grids["lidar_occupancy"] * validity[0],
            depth_hit * validity[1],
            grids["camera_semantic_risk"] * validity[2],
        )
    ).astype(np.float32, copy=False)
    grids["sensor_validity_fraction"] = validity_fraction
    grids["fused_occupancy"] = fused_occupancy

    channels = np.stack(
        tuple(grids[name] for name in FRAME_CHANNEL_NAMES),
        axis=0,
    ).astype(np.float32, copy=False)
    return FusionFrameV2(
        channels=channels,
        timestamp_s=float(inputs.timestamp_s),
        source_validity=validity,
    )


class TemporalTriBEVV2:
    """Five-frame newest-first history with odometry compensation."""

    def __init__(
        self,
        geometry: BEVGeometryV2 | None = None,
        history_frames: int = HISTORY_FRAMES,
    ) -> None:
        self.geometry = geometry or BEVGeometryV2()
        if self.geometry.shape != (64, 64):
            raise ValueError("the v2 model contract requires a 64x64 BEV")
        if history_frames != HISTORY_FRAMES:
            raise ValueError(f"the v2 model requires {HISTORY_FRAMES} frames")
        self.history_frames = history_frames
        self._history: list[FusionFrameV2] = []

    @property
    def populated_history(self) -> int:
        return len(self._history)

    def reset(self) -> None:
        self._history.clear()

    def update(
        self,
        frame: FusionFrameV2,
        ego_delta: OdometryDelta | None = None,
    ) -> tuple[np.ndarray, Mapping[str, object]]:
        """Warp old frames, insert the new frame, and return ``1x60x64x64``."""

        delta = ego_delta or OdometryDelta()
        warped: list[FusionFrameV2] = []
        for old in self._history:
            channels = warp_bev_nearest(
                old.channels,
                delta,
                geometry=self.geometry,
            ).astype(np.float32, copy=False)
            coverage = warp_bev_nearest(
                np.ones((1, *self.geometry.shape), dtype=np.float32),
                delta,
                geometry=self.geometry,
            )[0]
            channels[_CHANNEL_INDEX["depth_unknown"]] = np.maximum(
                channels[_CHANNEL_INDEX["depth_unknown"]],
                1.0 - coverage,
            )
            warped.append(
                FusionFrameV2(
                    channels=channels,
                    timestamp_s=old.timestamp_s,
                    source_validity=old.source_validity,
                )
            )
        self._history = [frame, *warped[: self.history_frames - 1]]

        output = np.zeros(MODEL_INPUT_SHAPE, dtype=np.float32)
        for index, item in enumerate(self._history):
            start = index * CHANNELS_PER_FRAME
            output[0, start : start + CHANNELS_PER_FRAME] = item.channels
        metadata = {
            "shape": list(output.shape),
            "history_populated": len(self._history),
            "history_required": self.history_frames,
            "warm": len(self._history) == self.history_frames,
            "timestamps_s": [item.timestamp_s for item in self._history],
            "source_validity": [
                list(item.source_validity) for item in self._history
            ],
            "shadow_only": True,
            "cmd_vel_authority": False,
        }
        return output, metadata
