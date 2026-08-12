"""Pure Python/NumPy Depth-4D read-only perception frontend."""

from __future__ import annotations

from .bev import cell_centres, metric_to_cell, rasterize_depth_bev
from .contracts import (
    BEVGeometry,
    CameraIntrinsics,
    CameraToBase,
    DepthBEVGrid,
    GRID_SIZE,
    HeightBands,
    PointImage,
    ProjectionLimits,
    READ_ONLY_AUTHORITY,
    ReadOnlyAuthority,
    STVLConfig,
    TrackerConfig,
    empty_depth_bev,
)
from .frontend import Depth4DFrontend, Depth4DOutput
from .projection import (
    back_project_depth,
    decode_depth_image,
    project_depth_to_base,
    transform_point_image_to_base,
)
from .temporal import STVLLite, unknown_observation
from .tracking import (
    ComponentBatch,
    NearestNeighbourTracker,
    TrackBatch,
    extract_components,
    radial_ttc_s,
)

__all__ = [
    "BEVGeometry",
    "CameraIntrinsics",
    "CameraToBase",
    "ComponentBatch",
    "Depth4DFrontend",
    "Depth4DOutput",
    "DepthBEVGrid",
    "GRID_SIZE",
    "HeightBands",
    "NearestNeighbourTracker",
    "PointImage",
    "ProjectionLimits",
    "READ_ONLY_AUTHORITY",
    "ReadOnlyAuthority",
    "STVLConfig",
    "STVLLite",
    "TrackBatch",
    "TrackerConfig",
    "back_project_depth",
    "cell_centres",
    "decode_depth_image",
    "empty_depth_bev",
    "extract_components",
    "metric_to_cell",
    "project_depth_to_base",
    "radial_ttc_s",
    "rasterize_depth_bev",
    "transform_point_image_to_base",
    "unknown_observation",
]

__version__ = "0.1.0"
