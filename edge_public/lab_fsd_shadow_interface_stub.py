"""Public shadow-planner interface example.

This file demonstrates the data contract used by a laboratory FSD-style shadow
planner. It does not publish motion commands and cannot control hardware.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class OccupancyCell:
    x_m: float
    y_m: float
    probability: float
    source: str


@dataclass(frozen=True)
class CandidatePath:
    name: str
    points_m: Sequence[tuple[float, float]]
    nominal_speed_mps: float


@dataclass(frozen=True)
class PathRisk:
    name: str
    min_clearance_m: float
    risk_score: float
    accepted: bool


@dataclass(frozen=True)
class ShadowPlannerOutput:
    mode: str
    occupancy: Sequence[OccupancyCell]
    path_risks: Sequence[PathRisk]
    safety_gate: str
    note: str


def build_occupancy_from_scan(
    ranges_m: Iterable[float],
    start_angle_deg: float = -45.0,
    step_angle_deg: float = 15.0,
) -> List[OccupancyCell]:
    """Convert a compact range scan into occupied cells."""

    cells: List[OccupancyCell] = []
    for idx, distance_m in enumerate(ranges_m):
        if distance_m <= 0.0 or distance_m > 8.0:
            continue
        angle = math.radians(start_angle_deg + idx * step_angle_deg)
        cells.append(
            OccupancyCell(
                x_m=round(distance_m * math.cos(angle), 3),
                y_m=round(distance_m * math.sin(angle), 3),
                probability=0.75,
                source="range_scan",
            )
        )
    return cells


def _point_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def score_path(
    path: CandidatePath,
    occupancy: Sequence[OccupancyCell],
    stop_clearance_m: float = 0.35,
) -> PathRisk:
    """Score a path against occupied cells.

    A high score only marks risk for the operator and upper layer. The function
    intentionally returns no chassis command.
    """

    min_clearance = 99.0
    for point in path.points_m:
        for cell in occupancy:
            min_clearance = min(min_clearance, _point_distance(point, (cell.x_m, cell.y_m)))
    if not occupancy:
        min_clearance = 99.0

    clearance_risk = max(0.0, 1.0 - min_clearance / max(stop_clearance_m, 1e-6))
    speed_risk = min(1.0, path.nominal_speed_mps / 0.8)
    risk_score = round(0.75 * clearance_risk + 0.25 * speed_risk, 3)
    return PathRisk(
        name=path.name,
        min_clearance_m=round(min_clearance, 3),
        risk_score=risk_score,
        accepted=min_clearance >= stop_clearance_m and risk_score < 0.55,
    )


def run_shadow_planner(
    ranges_m: Sequence[float],
    candidate_paths: Sequence[CandidatePath],
) -> ShadowPlannerOutput:
    occupancy = build_occupancy_from_scan(ranges_m)
    risks = [score_path(path, occupancy) for path in candidate_paths]
    accepted = [risk for risk in risks if risk.accepted]
    gate = "observe" if not accepted else "candidate_available"
    return ShadowPlannerOutput(
        mode="shadow_only",
        occupancy=occupancy,
        path_risks=risks,
        safety_gate=gate,
        note="Read-only shadow result. A safety operator or a separate certified layer owns execution.",
    )


def main() -> None:
    paths = [
        CandidatePath("center_slow", [(0.2, 0.0), (0.6, 0.0), (1.0, 0.0)], 0.25),
        CandidatePath("left_bypass", [(0.2, 0.0), (0.5, 0.25), (0.9, 0.35)], 0.2),
        CandidatePath("right_bypass", [(0.2, 0.0), (0.5, -0.25), (0.9, -0.35)], 0.2),
    ]
    output = run_shadow_planner([1.4, 1.1, 0.85, 1.2, 2.5, 0.0, 1.8], paths)
    print(json.dumps(asdict(output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
