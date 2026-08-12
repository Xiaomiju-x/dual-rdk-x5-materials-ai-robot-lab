"""Pure NumPy trajectory definitions and rectangular-footprint risk labels."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

FUTURE_HORIZONS_S = (0.4, 0.8, 1.2)
EVALUATION_TIMES_S = tuple(np.linspace(0.1, 1.2, 12, dtype=np.float64))

DEFAULT_GRID_RESOLUTION_M = 0.10
DEFAULT_GRID_X_MIN_M = -1.20
DEFAULT_GRID_Y_MIN_M = -3.20
DEFAULT_FOOTPRINT_LENGTH_M = 0.50
DEFAULT_FOOTPRINT_WIDTH_M = 0.40
DEFAULT_SAFETY_MARGIN_M = 0.08


@dataclass(frozen=True)
class TrajectoryCandidate:
    """One fixed low-speed constant-twist trajectory candidate."""

    index: int
    name: str
    speed_m_s: float
    angular_velocity_rad_s: float


_SPEED_PROFILES = (
    ("slow", 0.25),
    ("cruise", 0.45),
    ("fast", 0.65),
)
_TURN_PROFILES = (
    ("right_hard", -0.80),
    ("right_soft", -0.40),
    ("straight", 0.0),
    ("left_soft", 0.40),
    ("left_hard", 0.80),
)

CANDIDATE_TRAJECTORIES = tuple(
    TrajectoryCandidate(
        index=speed_index * len(_TURN_PROFILES) + turn_index,
        name=f"{speed_name}_{turn_name}",
        speed_m_s=speed_m_s,
        angular_velocity_rad_s=angular_velocity_rad_s,
    )
    for speed_index, (speed_name, speed_m_s) in enumerate(_SPEED_PROFILES)
    for turn_index, (turn_name, angular_velocity_rad_s) in enumerate(_TURN_PROFILES)
)


def candidate_definition_array() -> np.ndarray:
    """Return immutable candidate metadata as ``index, speed, omega`` rows."""

    return np.asarray(
        [
            (
                candidate.index,
                candidate.speed_m_s,
                candidate.angular_velocity_rad_s,
            )
            for candidate in CANDIDATE_TRAJECTORIES
        ],
        dtype=np.float32,
    )


def sample_candidate_poses(
    candidate: TrajectoryCandidate,
    times_s: Sequence[float] = EVALUATION_TIMES_S,
) -> np.ndarray:
    """Sample ``x, y, yaw`` poses for a constant-twist candidate."""

    times = np.asarray(tuple(times_s), dtype=np.float64)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times_s must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(times)) or np.any(times < 0.0):
        raise ValueError("times_s must contain finite non-negative values")

    speed = float(candidate.speed_m_s)
    omega = float(candidate.angular_velocity_rad_s)
    if not math.isfinite(speed) or not math.isfinite(omega) or speed < 0.0:
        raise ValueError("candidate speed/omega must be finite and speed non-negative")

    yaw = omega * times
    if abs(omega) < 1e-9:
        x_m = speed * times
        y_m = np.zeros_like(times)
    else:
        radius = speed / omega
        x_m = radius * np.sin(yaw)
        y_m = radius * (1.0 - np.cos(yaw))
    return np.stack((x_m, y_m, yaw), axis=1).astype(np.float32)


def _footprint_offsets(
    *,
    footprint_length_m: float,
    footprint_width_m: float,
    safety_margin_m: float,
    sample_resolution_m: float,
) -> np.ndarray:
    values = (
        footprint_length_m,
        footprint_width_m,
        safety_margin_m,
        sample_resolution_m,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("footprint parameters must be finite")
    if (
        footprint_length_m <= 0.0
        or footprint_width_m <= 0.0
        or safety_margin_m < 0.0
        or sample_resolution_m <= 0.0
    ):
        raise ValueError("footprint dimensions/resolution must be positive")

    half_length = footprint_length_m / 2.0 + safety_margin_m
    half_width = footprint_width_m / 2.0 + safety_margin_m
    x_count = max(2, int(math.ceil(2.0 * half_length / sample_resolution_m)) + 1)
    y_count = max(2, int(math.ceil(2.0 * half_width / sample_resolution_m)) + 1)
    local_x = np.linspace(-half_length, half_length, x_count, dtype=np.float64)
    local_y = np.linspace(-half_width, half_width, y_count, dtype=np.float64)
    offset_x, offset_y = np.meshgrid(local_x, local_y, indexing="ij")
    return np.stack((offset_x.reshape(-1), offset_y.reshape(-1)), axis=1)


def _occupancy_probabilities(
    future_occupancy: np.ndarray,
    *,
    inputs_are_logits: bool,
) -> np.ndarray:
    occupancy = np.asarray(future_occupancy, dtype=np.float64)
    if occupancy.ndim == 4 and occupancy.shape[0] == 1:
        occupancy = occupancy[0]
    if occupancy.ndim != 3 or occupancy.shape[0] != len(FUTURE_HORIZONS_S):
        raise ValueError("future_occupancy must have shape 3xHxW or 1x3xHxW")
    if not np.all(np.isfinite(occupancy)):
        raise ValueError("future_occupancy must contain only finite values")
    if inputs_are_logits:
        clipped = np.clip(occupancy, -40.0, 40.0)
        occupancy = 1.0 / (1.0 + np.exp(-clipped))
    elif np.any((occupancy < 0.0) | (occupancy > 1.0)):
        raise ValueError("future_occupancy probabilities must be in [0, 1]")
    return occupancy


def rectangular_footprint_risk_labels(
    future_occupancy: np.ndarray,
    *,
    candidates: Sequence[TrajectoryCandidate] = CANDIDATE_TRAJECTORIES,
    inputs_are_logits: bool = False,
    grid_resolution_m: float = DEFAULT_GRID_RESOLUTION_M,
    grid_x_min_m: float = DEFAULT_GRID_X_MIN_M,
    grid_y_min_m: float = DEFAULT_GRID_Y_MIN_M,
    footprint_length_m: float = DEFAULT_FOOTPRINT_LENGTH_M,
    footprint_width_m: float = DEFAULT_FOOTPRINT_WIDTH_M,
    safety_margin_m: float = DEFAULT_SAFETY_MARGIN_M,
    evaluation_times_s: Sequence[float] = EVALUATION_TIMES_S,
) -> np.ndarray:
    """Generate collision-risk probabilities for fixed trajectory candidates.

    Each pose sweeps the measured rectangular footprint plus safety margin.
    Samples outside the BEV are conservatively assigned risk 1.0. The result
    is a 15-element probability target suitable for independent BCE-with-logits
    supervision; it is not a normalized trajectory-selection distribution.
    """

    occupancy = _occupancy_probabilities(
        future_occupancy,
        inputs_are_logits=inputs_are_logits,
    )
    candidate_tuple = tuple(candidates)
    if len(candidate_tuple) != 15:
        raise ValueError("the TinyOccFlowV2 contract requires 15 candidates")
    if tuple(candidate.index for candidate in candidate_tuple) != tuple(range(15)):
        raise ValueError("candidate indices must be the ordered range 0..14")
    if (
        not math.isfinite(grid_resolution_m)
        or not math.isfinite(grid_x_min_m)
        or not math.isfinite(grid_y_min_m)
        or grid_resolution_m <= 0.0
    ):
        raise ValueError("grid geometry must be finite with positive resolution")

    times = np.asarray(tuple(evaluation_times_s), dtype=np.float64)
    if (
        times.ndim != 1
        or times.size == 0
        or not np.all(np.isfinite(times))
        or np.any(times <= 0.0)
        or np.any(times > FUTURE_HORIZONS_S[-1])
    ):
        raise ValueError("evaluation times must be in (0, 1.2]")

    offsets = _footprint_offsets(
        footprint_length_m=footprint_length_m,
        footprint_width_m=footprint_width_m,
        safety_margin_m=safety_margin_m,
        sample_resolution_m=grid_resolution_m,
    )
    height, width = occupancy.shape[-2:]
    labels = np.empty(15, dtype=np.float64)

    for candidate in candidate_tuple:
        poses = sample_candidate_poses(candidate, times)
        pose_risks = np.empty(times.size, dtype=np.float64)
        for pose_index, (x_m, y_m, yaw_rad) in enumerate(poses):
            cosine = math.cos(float(yaw_rad))
            sine = math.sin(float(yaw_rad))
            world_x = float(x_m) + cosine * offsets[:, 0] - sine * offsets[:, 1]
            world_y = float(y_m) + sine * offsets[:, 0] + cosine * offsets[:, 1]
            rows = np.floor(
                (world_x - grid_x_min_m) / grid_resolution_m
            ).astype(np.int64)
            columns = np.floor(
                (world_y - grid_y_min_m) / grid_resolution_m
            ).astype(np.int64)
            inside = (
                (rows >= 0)
                & (rows < height)
                & (columns >= 0)
                & (columns < width)
            )
            if not np.all(inside):
                pose_risks[pose_index] = 1.0
                continue
            horizon_index = int(np.searchsorted(FUTURE_HORIZONS_S, times[pose_index]))
            horizon_index = min(len(FUTURE_HORIZONS_S) - 1, horizon_index)
            pose_risks[pose_index] = float(
                np.max(occupancy[horizon_index, rows, columns])
            )

        labels[candidate.index] = np.clip(
            0.65 * float(np.max(pose_risks))
            + 0.25 * float(np.mean(pose_risks))
            + 0.10 * float(pose_risks[-1]),
            0.0,
            1.0,
        )
    return labels.astype(np.float32)


def risk_probabilities_to_logits(
    probabilities: np.ndarray,
    *,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """Convert independent risk probabilities to finite BCE logits."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (15,) or not np.all(np.isfinite(values)):
        raise ValueError("probabilities must be a finite 15-element vector")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    if not math.isfinite(epsilon) or epsilon <= 0.0 or epsilon >= 0.5:
        raise ValueError("epsilon must lie in (0, 0.5)")
    clipped = np.clip(values, epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped)).astype(np.float32)


__all__ = [
    "CANDIDATE_TRAJECTORIES",
    "DEFAULT_FOOTPRINT_LENGTH_M",
    "DEFAULT_FOOTPRINT_WIDTH_M",
    "DEFAULT_GRID_RESOLUTION_M",
    "DEFAULT_GRID_X_MIN_M",
    "DEFAULT_GRID_Y_MIN_M",
    "DEFAULT_SAFETY_MARGIN_M",
    "EVALUATION_TIMES_S",
    "FUTURE_HORIZONS_S",
    "TrajectoryCandidate",
    "candidate_definition_array",
    "rectangular_footprint_risk_labels",
    "risk_probabilities_to_logits",
    "sample_candidate_poses",
]
