"""Static contracts shared by the passive finals vNext modules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HISTORY_FRAMES = 5
CHANNELS_PER_FRAME = 12
TRAJECTORY_COUNT = 15
MODEL_INPUT_SHAPE = (1, HISTORY_FRAMES * CHANNELS_PER_FRAME, 64, 64)

FRAME_CHANNEL_NAMES = (
    "lidar_occupancy",
    "lidar_visibility",
    "depth_hit_low",
    "depth_hit_mid",
    "depth_hit_high",
    "depth_free",
    "depth_unknown",
    "depth_closing_rate",
    "camera_semantic_risk",
    "camera_visibility",
    "sensor_validity_fraction",
    "fused_occupancy",
)


def history_channel_names(
    history_frames: int = HISTORY_FRAMES,
) -> tuple[str, ...]:
    """Return newest-first static model channel names."""

    if history_frames <= 0:
        raise ValueError("history_frames must be positive")
    return tuple(
        f"t_minus_{age}_{name}"
        for age in range(history_frames)
        for name in FRAME_CHANNEL_NAMES
    )


@dataclass(frozen=True, slots=True)
class BEVGeometryV2:
    """Robot-centric metric BEV contract.

    Axis 0 is forward ``x`` and axis 1 is left-positive ``y``. The default
    64x64 grid covers 6.4 m by 6.4 m at 0.1 m resolution.
    """

    height: int = 64
    width: int = 64
    resolution_m: float = 0.1
    x_min_m: float = -1.2
    y_min_m: float = -3.2

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("BEV dimensions must be positive")
        if not np.isfinite(self.resolution_m) or self.resolution_m <= 0.0:
            raise ValueError("resolution_m must be finite and positive")
        if not np.isfinite(self.x_min_m) or not np.isfinite(self.y_min_m):
            raise ValueError("BEV origin must be finite")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def x_max_m(self) -> float:
        return self.x_min_m + self.height * self.resolution_m

    @property
    def y_max_m(self) -> float:
        return self.y_min_m + self.width * self.resolution_m


@dataclass(frozen=True, slots=True)
class FusionFrameV2:
    """One validated 12-channel frame entering temporal fusion."""

    channels: np.ndarray
    timestamp_s: float
    source_validity: tuple[float, float, float]

    def __post_init__(self) -> None:
        array = np.asarray(self.channels)
        if array.shape != (CHANNELS_PER_FRAME, 64, 64):
            raise ValueError(
                "channels must have shape "
                f"({CHANNELS_PER_FRAME}, 64, 64)"
            )
        if not np.isfinite(array).all():
            raise ValueError("channels must contain only finite values")
        if not np.isfinite(self.timestamp_s):
            raise ValueError("timestamp_s must be finite")
        if len(self.source_validity) != 3:
            raise ValueError("source_validity must contain lidar/depth/vision")
        if any(
            not np.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.source_validity
        ):
            raise ValueError("source validity values must be in [0, 1]")
