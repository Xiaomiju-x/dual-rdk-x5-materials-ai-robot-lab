#!/usr/bin/env python3
"""Pure geometry guard for the supervised finals straight-path test."""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def validate_straight_path(
    points_xy: Sequence[tuple[float, float]],
    *,
    start_x: float,
    start_y: float,
    start_yaw: float,
    requested_distance: float,
    max_lateral_m: float = 0.07,
    max_heading_rad: float = 0.15,
    max_backtrack_m: float = 0.02,
    endpoint_tolerance_m: float = 0.10,
    max_start_offset_m: float = 0.10,
) -> dict[str, Any]:
    """Validate that a map-frame path is a short, forward-only straight path."""

    scalar_values = (
        start_x,
        start_y,
        start_yaw,
        requested_distance,
        max_lateral_m,
        max_heading_rad,
        max_backtrack_m,
        endpoint_tolerance_m,
        max_start_offset_m,
    )
    if any(not math.isfinite(value) for value in scalar_values):
        return {"ok": False, "reason_codes": ["NONFINITE_GUARD_INPUT"]}
    if requested_distance <= 0.0 or any(
        value < 0.0
        for value in (
            max_lateral_m,
            max_heading_rad,
            max_backtrack_m,
            endpoint_tolerance_m,
            max_start_offset_m,
        )
    ):
        return {"ok": False, "reason_codes": ["INVALID_GUARD_LIMIT"]}
    if len(points_xy) < 2:
        return {"ok": False, "reason_codes": ["PATH_TOO_SHORT"]}

    cos_yaw = math.cos(start_yaw)
    sin_yaw = math.sin(start_yaw)
    local: list[tuple[float, float]] = []
    for point in points_xy:
        if len(point) != 2:
            return {"ok": False, "reason_codes": ["INVALID_PATH_POINT"]}
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            return {"ok": False, "reason_codes": ["NONFINITE_PATH_POINT"]}
        dx = x - start_x
        dy = y - start_y
        local.append(
            (
                cos_yaw * dx + sin_yaw * dy,
                -sin_yaw * dx + cos_yaw * dy,
            )
        )

    segment_lengths: list[float] = []
    heading_errors: list[float] = []
    largest_backtrack = 0.0
    for previous, current in zip(local, local[1:], strict=False):
        dx = current[0] - previous[0]
        dy = current[1] - previous[1]
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            continue
        segment_lengths.append(length)
        heading_errors.append(abs(math.atan2(dy, dx)))
        largest_backtrack = max(largest_backtrack, -dx)

    reasons: list[str] = []
    if not segment_lengths:
        reasons.append("PATH_HAS_NO_MOTION")

    start_offset = math.hypot(*local[0])
    endpoint_forward, endpoint_lateral = local[-1]
    endpoint_error = math.hypot(
        endpoint_forward - requested_distance,
        endpoint_lateral,
    )
    path_length = sum(segment_lengths)
    max_lateral = max(abs(point[1]) for point in local)
    max_heading = max(heading_errors, default=0.0)
    min_forward = min(point[0] for point in local)

    if start_offset > max_start_offset_m:
        reasons.append("PATH_START_OFFSET")
    if min_forward < -max_backtrack_m or largest_backtrack > max_backtrack_m:
        reasons.append("PATH_BACKTRACK")
    if max_lateral > max_lateral_m:
        reasons.append("PATH_LATERAL_DEVIATION")
    if max_heading > max_heading_rad:
        reasons.append("PATH_HEADING_DEVIATION")
    if endpoint_forward <= 0.0 or endpoint_error > endpoint_tolerance_m:
        reasons.append("PATH_ENDPOINT_MISMATCH")
    if not 0.75 * requested_distance <= path_length <= 1.35 * requested_distance:
        reasons.append("PATH_LENGTH_MISMATCH")

    return {
        "ok": not reasons,
        "reason_codes": reasons,
        "point_count": len(local),
        "path_length_m": round(path_length, 5),
        "start_offset_m": round(start_offset, 5),
        "endpoint_forward_m": round(endpoint_forward, 5),
        "endpoint_lateral_m": round(endpoint_lateral, 5),
        "endpoint_error_m": round(endpoint_error, 5),
        "max_lateral_m": round(max_lateral, 5),
        "max_heading_rad": round(max_heading, 5),
        "largest_backtrack_m": round(largest_backtrack, 5),
    }
