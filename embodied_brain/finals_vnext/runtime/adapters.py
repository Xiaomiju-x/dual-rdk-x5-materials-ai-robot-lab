"""Pure adapters from expert outputs into the shared finals-vNext BEV."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from embodied_brain.finals_successor.x5_tribev_flow.contracts import (
    BEVGeometry as LegacyBEVGeometry,
    TriBEVConfig,
)
from embodied_brain.finals_successor.x5_tribev_flow.tribev import (
    rasterize_lidar,
)
from embodied_brain.finals_vnext.contracts import BEVGeometryV2
from embodied_brain.finals_vnext.depth4d.contracts import (
    BEVGeometry as DepthGeometry,
)
from embodied_brain.finals_vnext.depth4d.frontend import Depth4DOutput
from embodied_brain.finals_vnext.fusion import FusionInputsV2
from embodied_brain.finals_vnext.vision_fsd.contracts import SemanticBEVFrame


@dataclass(frozen=True, slots=True)
class LidarPlanes:
    occupancy: np.ndarray
    visibility: np.ndarray
    validity: float


@dataclass(frozen=True, slots=True)
class DepthPlanes:
    hit_low: np.ndarray
    hit_mid: np.ndarray
    hit_high: np.ndarray
    free: np.ndarray
    unknown: np.ndarray
    closing_rate: np.ndarray
    validity: float


@dataclass(frozen=True, slots=True)
class VisionPlanes:
    semantic_risk: np.ndarray
    visibility: np.ndarray
    validity: float


def _assert_common_geometry(
    *,
    x_min_m: float,
    x_max_m: float,
    y_min_m: float,
    y_max_m: float,
    common: BEVGeometryV2,
) -> None:
    expected = (
        common.x_min_m,
        common.x_max_m,
        common.y_min_m,
        common.y_max_m,
    )
    actual = (x_min_m, x_max_m, y_min_m, y_max_m)
    if not np.allclose(actual, expected, atol=1e-9, rtol=0.0):
        raise ValueError(f"BEV geometry mismatch: actual={actual}, expected={expected}")


def lidar_points_to_planes(
    points_xy_m: np.ndarray | None,
    *,
    valid: bool,
    geometry: BEVGeometryV2 | None = None,
) -> LidarPlanes:
    common = geometry or BEVGeometryV2()
    legacy = LegacyBEVGeometry(
        height=common.height,
        width=common.width,
        resolution_m=common.resolution_m,
        x_min_m=common.x_min_m,
        y_min_m=common.y_min_m,
    )
    raster = rasterize_lidar(
        points_xy_m,
        TriBEVConfig(geometry=legacy),
        valid=valid,
    )
    return LidarPlanes(
        occupancy=raster.occupancy.astype(np.float32, copy=False),
        visibility=raster.visibility.astype(np.float32, copy=False),
        validity=float(bool(valid)),
    )


def _closing_rate_grid(
    output: Depth4DOutput,
    geometry: DepthGeometry,
) -> np.ndarray:
    closing = np.zeros((64, 64), dtype=np.float32)
    for index in np.flatnonzero(output.tracks.valid & output.tracks.observed):
        position = output.tracks.position_xy_m[index].astype(np.float64)
        velocity = output.tracks.velocity_xy_mps[index].astype(np.float64)
        distance = float(np.linalg.norm(position))
        if distance <= 1e-9:
            normalized = 1.0
        else:
            radial = -float(np.dot(position, velocity)) / distance
            normalized = float(np.clip(radial / 1.0, 0.0, 1.0))
        row = int(
            np.floor((position[0] - geometry.x_min_m) / geometry.resolution_m)
        )
        column = int(
            np.floor((position[1] - geometry.y_min_m) / geometry.resolution_m)
        )
        if 0 <= row < 64 and 0 <= column < 64:
            row_slice = slice(max(0, row - 1), min(64, row + 2))
            col_slice = slice(max(0, column - 1), min(64, column + 2))
            closing[row_slice, col_slice] = np.maximum(
                closing[row_slice, col_slice],
                normalized,
            )
    return closing


def depth_output_to_planes(
    output: Depth4DOutput,
    *,
    geometry: DepthGeometry | None = None,
    common_geometry: BEVGeometryV2 | None = None,
) -> DepthPlanes:
    depth_geometry = geometry or DepthGeometry()
    common = common_geometry or BEVGeometryV2()
    _assert_common_geometry(
        x_min_m=depth_geometry.x_min_m,
        x_max_m=depth_geometry.x_max_m,
        y_min_m=depth_geometry.y_min_m,
        y_max_m=depth_geometry.y_max_m,
        common=common,
    )
    grid = output.temporal_bev
    grid.validate()
    confidence = grid.occupancy_confidence
    return DepthPlanes(
        hit_low=(grid.low.astype(np.float32) * confidence).astype(np.float32),
        hit_mid=(grid.mid.astype(np.float32) * confidence).astype(np.float32),
        hit_high=(grid.high.astype(np.float32) * confidence).astype(np.float32),
        free=grid.free.astype(np.float32),
        unknown=grid.unknown.astype(np.float32),
        closing_rate=_closing_rate_grid(output, depth_geometry),
        validity=float(np.clip(grid.source_valid_fraction, 0.0, 1.0)),
    )


def semantic_frame_to_planes(
    frame: SemanticBEVFrame,
    *,
    common_geometry: BEVGeometryV2 | None = None,
) -> VisionPlanes:
    common = common_geometry or BEVGeometryV2()
    _assert_common_geometry(
        x_min_m=frame.geometry.x_min_m,
        x_max_m=frame.geometry.x_max_m,
        y_min_m=frame.geometry.y_min_m,
        y_max_m=frame.geometry.y_max_m,
        common=common,
    )
    quality = frame.quality
    validity = float(
        np.clip(
            quality.visible_fraction * quality.mean_confidence,
            0.0,
            1.0,
        )
    )
    return VisionPlanes(
        semantic_risk=frame.semantic_risk.astype(np.float32, copy=False),
        visibility=frame.visibility.astype(np.float32, copy=False),
        validity=validity,
    )


def compose_fusion_inputs(
    *,
    timestamp_s: float,
    lidar: LidarPlanes | None = None,
    depth: DepthPlanes | None = None,
    vision: VisionPlanes | None = None,
) -> FusionInputsV2:
    return FusionInputsV2(
        timestamp_s=timestamp_s,
        lidar_occupancy=None if lidar is None else lidar.occupancy,
        lidar_visibility=None if lidar is None else lidar.visibility,
        depth_hit_low=None if depth is None else depth.hit_low,
        depth_hit_mid=None if depth is None else depth.hit_mid,
        depth_hit_high=None if depth is None else depth.hit_high,
        depth_free=None if depth is None else depth.free,
        depth_unknown=None if depth is None else depth.unknown,
        depth_closing_rate=None if depth is None else depth.closing_rate,
        camera_semantic_risk=None if vision is None else vision.semantic_risk,
        camera_visibility=None if vision is None else vision.visibility,
        lidar_validity=0.0 if lidar is None else lidar.validity,
        depth_validity=0.0 if depth is None else depth.validity,
        vision_validity=0.0 if vision is None else vision.validity,
    )


__all__ = [
    "DepthPlanes",
    "LidarPlanes",
    "VisionPlanes",
    "compose_fusion_inputs",
    "depth_output_to_planes",
    "lidar_points_to_planes",
    "semantic_frame_to_planes",
]
