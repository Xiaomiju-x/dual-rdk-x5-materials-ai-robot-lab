"""Decomposed, proposal-only NavTeacher-15 trajectory scoring."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil, cos, exp, hypot, isfinite, sin

import numpy as np

from .trajectories import (
    CANDIDATE_TRAJECTORIES,
    DEFAULT_EVALUATION_TIMES_S,
    TrajectoryCandidate,
    sample_candidate_poses,
)

SCHEMA_VERSION = "x5-navteacher-15-proposal/1.0"
CONTROL_AUTHORITY = False
COST_COMPONENT_NAMES = (
    "collision",
    "unknown",
    "ttc",
    "semantic_forbidden",
    "clearance",
    "progress",
    "smoothness",
)


def _probability_grid(name: str, value: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or not array.size:
        raise ValueError(f"{name} must be a non-empty HxW grid")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(array).all() or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must contain probabilities in [0, 1]")
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class GridGeometry:
    height: int
    width: int
    resolution_m: float = 0.10
    x_min_m: float = -1.20
    y_min_m: float = -3.20

    def __post_init__(self) -> None:
        if (
            not isinstance(self.height, int)
            or not isinstance(self.width, int)
            or self.height <= 0
            or self.width <= 0
        ):
            raise ValueError("height and width must be positive integers")
        for name in ("resolution_m", "x_min_m", "y_min_m"):
            if not isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.resolution_m <= 0.0:
            raise ValueError("resolution_m must be positive")


@dataclass(frozen=True, slots=True)
class NavScene:
    """Independent physical, unknown, semantic, and dynamic cost layers."""

    geometry: GridGeometry
    obstacle: np.ndarray
    unknown: np.ndarray
    semantic_forbidden: np.ndarray
    dynamic: np.ndarray
    current_speed_m_s: float = 0.0
    current_angular_velocity_rad_s: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, GridGeometry):
            raise TypeError("geometry must be GridGeometry")
        shape = (self.geometry.height, self.geometry.width)
        object.__setattr__(self, "obstacle", _probability_grid("obstacle", self.obstacle, shape))
        object.__setattr__(self, "unknown", _probability_grid("unknown", self.unknown, shape))
        object.__setattr__(
            self,
            "semantic_forbidden",
            _probability_grid("semantic_forbidden", self.semantic_forbidden, shape),
        )
        dynamic = np.asarray(self.dynamic, dtype=np.float32)
        if dynamic.ndim == 2:
            dynamic = dynamic[None]
        if dynamic.ndim != 3 or dynamic.shape[1:] != shape:
            raise ValueError("dynamic must have shape HxW or TxHxW")
        if not np.isfinite(dynamic).all() or np.any((dynamic < 0.0) | (dynamic > 1.0)):
            raise ValueError("dynamic must contain probabilities in [0, 1]")
        dynamic = np.ascontiguousarray(dynamic)
        dynamic.setflags(write=False)
        object.__setattr__(self, "dynamic", dynamic)
        for name in ("current_speed_m_s", "current_angular_velocity_rad_s"):
            if not isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class CostWeights:
    collision: float = 4.0
    unknown: float = 1.2
    ttc: float = 2.0
    semantic_forbidden: float = 3.0
    clearance: float = 1.5
    progress: float = 1.0
    smoothness: float = 0.4

    def __post_init__(self) -> None:
        for name in COST_COMPONENT_NAMES:
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} weight must be finite and non-negative")
            object.__setattr__(self, name, value)
        if not any(getattr(self, name) > 0.0 for name in COST_COMPONENT_NAMES):
            raise ValueError("at least one cost weight must be positive")

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [getattr(self, name) for name in COST_COMPONENT_NAMES],
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class TrajectoryProposal:
    candidate: TrajectoryCandidate
    poses_xyyaw: np.ndarray
    components: Mapping[str, float]
    total_cost: float
    first_contact_time_s: float | None

    def __post_init__(self) -> None:
        poses = np.asarray(self.poses_xyyaw, dtype=np.float32)
        if poses.ndim != 2 or poses.shape[1] != 3 or not np.isfinite(poses).all():
            raise ValueError("poses_xyyaw must be finite Nx3")
        poses = np.ascontiguousarray(poses)
        poses.setflags(write=False)
        object.__setattr__(self, "poses_xyyaw", poses)
        if tuple(self.components) != COST_COMPONENT_NAMES:
            raise ValueError("components must follow COST_COMPONENT_NAMES order")
        if not isfinite(float(self.total_cost)):
            raise ValueError("total_cost must be finite")


@dataclass(frozen=True, slots=True)
class ProposalSet:
    """Ranked evidence proposals with an explicit zero-control contract."""

    proposals: tuple[TrajectoryProposal, ...]
    ranked_indices: tuple[int, ...]
    best_index: int
    proposal_only: bool = True
    control_authority: bool = False
    control_interfaces: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.proposals) != 15:
            raise ValueError("NavTeacher requires exactly 15 proposals")
        if tuple(proposal.candidate.index for proposal in self.proposals) != tuple(range(15)):
            raise ValueError("proposal indices must be ordered 0..14")
        if tuple(sorted(self.ranked_indices)) != tuple(range(15)):
            raise ValueError("ranked_indices must be a permutation of 0..14")
        if self.best_index != self.ranked_indices[0]:
            raise ValueError("best_index must be the first ranked proposal")
        if not self.proposal_only or self.control_authority or self.control_interfaces:
            raise ValueError("NavTeacher outputs must remain proposal-only")

    @property
    def total_costs(self) -> np.ndarray:
        return np.asarray(
            [proposal.total_cost for proposal in self.proposals],
            dtype=np.float32,
        )

    @property
    def component_matrix(self) -> np.ndarray:
        return np.asarray(
            [
                [proposal.components[name] for name in COST_COMPONENT_NAMES]
                for proposal in self.proposals
            ],
            dtype=np.float32,
        )


def _footprint_offsets(
    *,
    length_m: float,
    width_m: float,
    margin_m: float,
    resolution_m: float,
) -> np.ndarray:
    if min(length_m, width_m, resolution_m) <= 0.0 or margin_m < 0.0:
        raise ValueError("footprint dimensions must be positive and margin non-negative")
    half_length = length_m / 2.0 + margin_m
    half_width = width_m / 2.0 + margin_m
    x_count = max(2, int(ceil(2.0 * half_length / resolution_m)) + 1)
    y_count = max(2, int(ceil(2.0 * half_width / resolution_m)) + 1)
    local_x = np.linspace(-half_length, half_length, x_count)
    local_y = np.linspace(-half_width, half_width, y_count)
    grid_x, grid_y = np.meshgrid(local_x, local_y, indexing="ij")
    return np.stack((grid_x.ravel(), grid_y.ravel()), axis=1)


def _clearance_field(obstacle: np.ndarray, resolution_m: float) -> np.ndarray:
    """Approximate Euclidean clearance with an eight-neighbor chamfer pass."""

    blocked = obstacle >= 0.50
    height, width = blocked.shape
    maximum = hypot(height, width) * resolution_m
    distances = np.full((height, width), maximum, dtype=np.float64)
    distances[blocked] = 0.0
    diagonal = resolution_m * np.sqrt(2.0)
    neighbors_forward = ((-1, 0, resolution_m), (0, -1, resolution_m), (-1, -1, diagonal), (-1, 1, diagonal))
    neighbors_backward = ((1, 0, resolution_m), (0, 1, resolution_m), (1, 1, diagonal), (1, -1, diagonal))
    for rows, columns, neighbors in (
        (range(height), range(width), neighbors_forward),
        (range(height - 1, -1, -1), range(width - 1, -1, -1), neighbors_backward),
    ):
        for row in rows:
            for column in columns:
                best = distances[row, column]
                for delta_row, delta_column, step in neighbors:
                    other_row = row + delta_row
                    other_column = column + delta_column
                    if 0 <= other_row < height and 0 <= other_column < width:
                        best = min(best, distances[other_row, other_column] + step)
                distances[row, column] = best
    return distances


def _sample_indices(
    pose: np.ndarray,
    offsets: np.ndarray,
    geometry: GridGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_m, y_m, yaw = (float(value) for value in pose)
    cosine = cos(yaw)
    sine = sin(yaw)
    world_x = x_m + cosine * offsets[:, 0] - sine * offsets[:, 1]
    world_y = y_m + sine * offsets[:, 0] + cosine * offsets[:, 1]
    rows = np.floor((world_x - geometry.x_min_m) / geometry.resolution_m).astype(np.int64)
    columns = np.floor((world_y - geometry.y_min_m) / geometry.resolution_m).astype(np.int64)
    inside = (
        (rows >= 0)
        & (rows < geometry.height)
        & (columns >= 0)
        & (columns < geometry.width)
    )
    return rows, columns, inside


def score_trajectory_proposals(
    scene: NavScene,
    *,
    weights: CostWeights | None = None,
    evaluation_times_s: tuple[float, ...] = DEFAULT_EVALUATION_TIMES_S,
    footprint_length_m: float = 0.50,
    footprint_width_m: float = 0.40,
    safety_margin_m: float = 0.08,
    desired_clearance_m: float = 0.25,
    contact_threshold: float = 0.50,
) -> ProposalSet:
    """Score collision, uncertainty, semantics, TTC, and motion separately.

    The lowest-cost item is a diagnostic proposal. This function cannot emit a
    velocity, transform, serial command, ROS service, or action.
    """

    if not isinstance(scene, NavScene):
        raise TypeError("scene must be NavScene")
    cfg = weights or CostWeights()
    if not isinstance(cfg, CostWeights):
        raise TypeError("weights must be CostWeights")
    times = np.asarray(evaluation_times_s, dtype=np.float64)
    if (
        times.ndim != 1
        or not times.size
        or not np.isfinite(times).all()
        or np.any(times <= 0.0)
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError("evaluation_times_s must be positive and strictly increasing")
    if desired_clearance_m <= 0.0 or not 0.0 < contact_threshold <= 1.0:
        raise ValueError("clearance/contact thresholds are invalid")

    geometry = scene.geometry
    offsets = _footprint_offsets(
        length_m=footprint_length_m,
        width_m=footprint_width_m,
        margin_m=safety_margin_m,
        resolution_m=geometry.resolution_m,
    )
    clearance = _clearance_field(scene.obstacle, geometry.resolution_m)
    maximum_progress = max(
        candidate.speed_m_s * float(times[-1])
        for candidate in CANDIDATE_TRAJECTORIES
    )
    weight_array = cfg.as_array()
    proposals: list[TrajectoryProposal] = []

    for candidate in CANDIDATE_TRAJECTORIES:
        poses = sample_candidate_poses(candidate, times)
        collision_samples: list[float] = []
        unknown_samples: list[float] = []
        semantic_samples: list[float] = []
        clearance_samples: list[float] = []
        contact_time: float | None = None

        for pose_index, pose in enumerate(poses):
            rows, columns, inside = _sample_indices(pose, offsets, geometry)
            dynamic_index = min(
                scene.dynamic.shape[0] - 1,
                int(np.floor(pose_index * scene.dynamic.shape[0] / len(poses))),
            )
            if not np.all(inside):
                collision_value = 1.0
                unknown_value = 1.0
                semantic_value = 0.0
                clearance_value = 0.0
                dynamic_value = 1.0
            else:
                collision_value = float(np.max(scene.obstacle[rows, columns]))
                unknown_value = float(np.mean(scene.unknown[rows, columns]))
                semantic_value = float(np.max(scene.semantic_forbidden[rows, columns]))
                clearance_value = float(np.min(clearance[rows, columns]))
                dynamic_value = float(np.max(scene.dynamic[dynamic_index, rows, columns]))
            collision_samples.append(collision_value)
            unknown_samples.append(unknown_value)
            semantic_samples.append(semantic_value)
            clearance_samples.append(clearance_value)
            if (
                contact_time is None
                and max(collision_value, dynamic_value) >= contact_threshold
            ):
                contact_time = float(times[pose_index])

        collision = 0.70 * max(collision_samples) + 0.30 * float(np.mean(collision_samples))
        unknown = float(np.mean(unknown_samples))
        semantic = 0.70 * max(semantic_samples) + 0.30 * float(np.mean(semantic_samples))
        clearance_cost = float(
            np.mean([exp(-value / desired_clearance_m) for value in clearance_samples])
        )
        ttc = (
            0.0
            if contact_time is None
            else 1.0 - min(contact_time / float(times[-1]), 1.0)
        )
        final_progress = max(0.0, float(poses[-1, 0]))
        progress = 1.0 - min(final_progress / max(maximum_progress, 1e-9), 1.0)
        speed_delta = abs(candidate.speed_m_s - scene.current_speed_m_s) / 0.50
        omega_delta = abs(
            candidate.angular_velocity_rad_s
            - scene.current_angular_velocity_rad_s
        ) / 0.80
        smoothness = min(1.0, 0.5 * speed_delta + 0.5 * omega_delta)
        component_values = (
            collision,
            unknown,
            ttc,
            semantic,
            clearance_cost,
            progress,
            smoothness,
        )
        component_mapping = {
            name: float(value)
            for name, value in zip(COST_COMPONENT_NAMES, component_values, strict=True)
        }
        total = float(np.dot(weight_array, np.asarray(component_values)))
        proposals.append(
            TrajectoryProposal(
                candidate=candidate,
                poses_xyyaw=poses,
                components=component_mapping,
                total_cost=total,
                first_contact_time_s=contact_time,
            )
        )

    ranked = tuple(
        int(index)
        for index in np.argsort(
            np.asarray([proposal.total_cost for proposal in proposals]),
            kind="stable",
        )
    )
    return ProposalSet(
        proposals=tuple(proposals),
        ranked_indices=ranked,
        best_index=ranked[0],
    )


__all__ = [
    "CONTROL_AUTHORITY",
    "COST_COMPONENT_NAMES",
    "SCHEMA_VERSION",
    "CostWeights",
    "GridGeometry",
    "NavScene",
    "ProposalSet",
    "TrajectoryProposal",
    "score_trajectory_proposals",
]
