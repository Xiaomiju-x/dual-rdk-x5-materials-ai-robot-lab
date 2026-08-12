"""Fixed, symmetric NavTeacher-15 trajectory proposal definitions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np

DEFAULT_EVALUATION_TIMES_S = tuple(np.linspace(0.1, 2.0, 20, dtype=np.float64))


@dataclass(frozen=True, slots=True)
class TrajectoryCandidate:
    """One constant-twist proposal; never a velocity command."""

    index: int
    name: str
    mode: str
    speed_m_s: float
    angular_velocity_rad_s: float

    def __post_init__(self) -> None:
        if self.mode not in {"stop", "hold", "arc"}:
            raise ValueError("mode must be stop, hold, or arc")
        if not isfinite(float(self.speed_m_s)) or self.speed_m_s < 0.0:
            raise ValueError("speed_m_s must be finite and non-negative")
        if not isfinite(float(self.angular_velocity_rad_s)):
            raise ValueError("angular_velocity_rad_s must be finite")
        if self.mode in {"stop", "hold"} and (
            self.speed_m_s != 0.0 or self.angular_velocity_rad_s != 0.0
        ):
            raise ValueError("stop/hold candidates must be stationary")

    @property
    def is_stationary(self) -> bool:
        return self.mode in {"stop", "hold"}

    @property
    def is_straight_motion(self) -> bool:
        return self.mode == "arc" and abs(self.angular_velocity_rad_s) < 1e-9


_DEFINITIONS = (
    ("stop", "stop", 0.00, 0.00),
    ("hold", "hold", 0.00, 0.00),
    ("creep_right_hard", "arc", 0.18, -0.80),
    ("creep_right_soft", "arc", 0.18, -0.40),
    ("creep_straight", "arc", 0.18, 0.00),
    ("creep_left_soft", "arc", 0.18, 0.40),
    ("creep_left_hard", "arc", 0.18, 0.80),
    ("cruise_right_hard", "arc", 0.35, -0.80),
    ("cruise_right_soft", "arc", 0.35, -0.40),
    ("cruise_straight", "arc", 0.35, 0.00),
    ("cruise_left_soft", "arc", 0.35, 0.40),
    ("cruise_left_hard", "arc", 0.35, 0.80),
    ("fast_right_soft", "arc", 0.50, -0.40),
    ("fast_straight", "arc", 0.50, 0.00),
    ("fast_left_soft", "arc", 0.50, 0.40),
)

CANDIDATE_TRAJECTORIES = tuple(
    TrajectoryCandidate(index, name, mode, speed, omega)
    for index, (name, mode, speed, omega) in enumerate(_DEFINITIONS)
)


def candidate_definition_array() -> np.ndarray:
    """Return ordered ``index, speed, omega, stationary`` metadata."""

    return np.asarray(
        [
            (
                candidate.index,
                candidate.speed_m_s,
                candidate.angular_velocity_rad_s,
                float(candidate.is_stationary),
            )
            for candidate in CANDIDATE_TRAJECTORIES
        ],
        dtype=np.float32,
    )


def sample_candidate_poses(
    candidate: TrajectoryCandidate,
    times_s: Sequence[float] = DEFAULT_EVALUATION_TIMES_S,
) -> np.ndarray:
    """Sample ``x, y, yaw`` for one proposal in the base frame."""

    if not isinstance(candidate, TrajectoryCandidate):
        raise TypeError("candidate must be TrajectoryCandidate")
    times = np.asarray(tuple(times_s), dtype=np.float64)
    if (
        times.ndim != 1
        or not times.size
        or not np.isfinite(times).all()
        or np.any(times <= 0.0)
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError("times_s must be finite, positive, and strictly increasing")
    if candidate.is_stationary:
        return np.zeros((times.size, 3), dtype=np.float32)

    speed = float(candidate.speed_m_s)
    omega = float(candidate.angular_velocity_rad_s)
    yaw = omega * times
    if abs(omega) < 1e-9:
        x_m = speed * times
        y_m = np.zeros_like(times)
    else:
        radius = speed / omega
        x_m = radius * np.sin(yaw)
        y_m = radius * (1.0 - np.cos(yaw))
    return np.stack((x_m, y_m, yaw), axis=1).astype(np.float32)


__all__ = [
    "CANDIDATE_TRAJECTORIES",
    "DEFAULT_EVALUATION_TIMES_S",
    "TrajectoryCandidate",
    "candidate_definition_array",
    "sample_candidate_poses",
]
