"""Stateful, read-only Depth-4D frontend composed from pure NumPy modules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bev import rasterize_depth_bev
from .contracts import (
    BEVGeometry,
    CameraIntrinsics,
    CameraToBase,
    DepthBEVGrid,
    HeightBands,
    PointImage,
    ProjectionLimits,
    READ_ONLY_AUTHORITY,
    ReadOnlyAuthority,
    STVLConfig,
    TrackerConfig,
)
from .projection import project_depth_to_base
from .temporal import STVLLite
from .tracking import (
    ComponentBatch,
    NearestNeighbourTracker,
    TrackBatch,
    extract_components,
)


@dataclass
class Depth4DOutput:
    points_base: PointImage
    frame_bev: DepthBEVGrid
    temporal_bev: DepthBEVGrid
    components: ComponentBatch
    tracks: TrackBatch
    timestamp_s: float
    authority: ReadOnlyAuthority = READ_ONLY_AUTHORITY


class Depth4DFrontend:
    """Calibrated depth-to-BEV observer with no control-plane interfaces."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        camera_to_base: CameraToBase,
        *,
        projection_limits: ProjectionLimits | None = None,
        geometry: BEVGeometry | None = None,
        height_bands: HeightBands | None = None,
        stvl_config: STVLConfig | None = None,
        tracker_config: TrackerConfig | None = None,
    ) -> None:
        self.intrinsics = intrinsics
        self.camera_to_base = camera_to_base
        self.projection_limits = projection_limits or ProjectionLimits()
        self.geometry = geometry or BEVGeometry()
        self.height_bands = height_bands or HeightBands()
        self.stvl_config = stvl_config or STVLConfig()
        self.tracker_config = tracker_config or TrackerConfig()
        self._temporal = STVLLite(self.stvl_config)
        self._tracker = NearestNeighbourTracker(self.tracker_config)

    @property
    def authority(self) -> ReadOnlyAuthority:
        return READ_ONLY_AUTHORITY

    def reset(self) -> None:
        self._temporal.reset()
        self._tracker.reset()

    def process(
        self,
        depth: np.ndarray,
        timestamp_s: float,
    ) -> Depth4DOutput:
        points_base = project_depth_to_base(
            depth,
            self.intrinsics,
            self.camera_to_base,
            self.projection_limits,
        )
        frame = rasterize_depth_bev(
            points_base,
            self.camera_to_base.translation_m,
            self.geometry,
            self.height_bands,
            unknown_age_s=self.stvl_config.unknown_after_s,
        )
        temporal = self._temporal.update(frame, timestamp_s)
        components = extract_components(
            temporal,
            self.geometry,
            self.tracker_config,
        )
        tracks = self._tracker.update(components, timestamp_s)
        return Depth4DOutput(
            points_base=points_base,
            frame_bev=frame,
            temporal_bev=temporal,
            components=components,
            tracks=tracks,
            timestamp_s=float(timestamp_s),
        )
