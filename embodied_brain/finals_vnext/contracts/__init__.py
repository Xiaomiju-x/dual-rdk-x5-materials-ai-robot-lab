"""Machine-readable and Python contracts for the finals vNext candidate."""

from .core import (
    CHANNELS_PER_FRAME,
    FRAME_CHANNEL_NAMES,
    HISTORY_FRAMES,
    MODEL_INPUT_SHAPE,
    TRAJECTORY_COUNT,
    BEVGeometryV2,
    FusionFrameV2,
    history_channel_names,
)

__all__ = [
    "BEVGeometryV2",
    "CHANNELS_PER_FRAME",
    "FRAME_CHANNEL_NAMES",
    "FusionFrameV2",
    "HISTORY_FRAMES",
    "MODEL_INPUT_SHAPE",
    "TRAJECTORY_COUNT",
    "history_channel_names",
]
