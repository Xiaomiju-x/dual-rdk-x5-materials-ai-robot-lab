#!/usr/bin/env python3
"""Deterministic indoor synthetic episodes for X5-TriBEV-Flow.

These episodes exercise the data and training pipeline. They are not evidence
of real sensor performance and never carry live-camera provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from .dataset import (
        FUTURE_HORIZONS,
        FUTURE_HORIZON_SECONDS,
        GRID_HEIGHT,
        GRID_RESOLUTION_M,
        GRID_WIDTH,
        GRID_X_MIN_M,
        GRID_Y_MIN_M,
        HISTORY_FRAMES,
        SENSOR_NAMES,
        TRAJECTORY_TOKENS,
        TRAJECTORY_TOKEN_OMEGA_RAD_S,
        build_episode_metadata,
        build_episode_refs,
        episode_input_content_sha256,
        save_episode,
        split_episode_refs,
        summarize_splits,
    )
except ImportError:
    from dataset import (  # type: ignore[no-redef]
        FUTURE_HORIZONS,
        FUTURE_HORIZON_SECONDS,
        GRID_HEIGHT,
        GRID_RESOLUTION_M,
        GRID_WIDTH,
        GRID_X_MIN_M,
        GRID_Y_MIN_M,
        HISTORY_FRAMES,
        SENSOR_NAMES,
        TRAJECTORY_TOKENS,
        TRAJECTORY_TOKEN_OMEGA_RAD_S,
        build_episode_metadata,
        build_episode_refs,
        episode_input_content_sha256,
        save_episode,
        split_episode_refs,
        summarize_splits,
    )


HISTORY_OFFSETS_S = np.asarray(
    (-0.8, -0.6, -0.4, -0.2, 0.0), dtype=np.float32
)
FUTURE_HORIZONS_S = np.asarray(FUTURE_HORIZON_SECONDS, dtype=np.float32)
TRAJECTORY_OMEGA_RAD_S = np.asarray(
    TRAJECTORY_TOKEN_OMEGA_RAD_S, dtype=np.float32
)

SCENARIO_IDS = (
    "corridor_straight",
    "corridor_narrow",
    "corridor_static_clutter",
    "dynamic_person_crossing",
    "dynamic_person_leading",
    "modality_dropout",
)


@dataclass(frozen=True)
class ScenarioSpec:
    corridor_width_m: float
    clutter: bool = False
    person_motion: str = "none"
    force_dropout: bool = False


SCENARIOS = {
    "corridor_straight": ScenarioSpec(corridor_width_m=2.40),
    "corridor_narrow": ScenarioSpec(corridor_width_m=1.45),
    "corridor_static_clutter": ScenarioSpec(
        corridor_width_m=2.30, clutter=True
    ),
    "dynamic_person_crossing": ScenarioSpec(
        corridor_width_m=2.40, person_motion="crossing"
    ),
    "dynamic_person_leading": ScenarioSpec(
        corridor_width_m=2.40, person_motion="leading"
    ),
    "modality_dropout": ScenarioSpec(
        corridor_width_m=2.20,
        clutter=True,
        person_motion="crossing",
        force_dropout=True,
    ),
}


def _coordinate_grid() -> tuple[np.ndarray, np.ndarray]:
    x = GRID_X_MIN_M + (
        np.arange(GRID_HEIGHT, dtype=np.float32) + 0.5
    ) * GRID_RESOLUTION_M
    y = GRID_Y_MIN_M + (
        np.arange(GRID_WIDTH, dtype=np.float32) + 0.5
    ) * GRID_RESOLUTION_M
    return np.meshgrid(x, y, indexing="ij")


GRID_X, GRID_Y = _coordinate_grid()


def _circle(cx: float, cy: float, radius_m: float) -> np.ndarray:
    return (GRID_X - cx) ** 2 + (GRID_Y - cy) ** 2 <= radius_m**2


def _rectangle(
    x_min: float, x_max: float, y_min: float, y_max: float
) -> np.ndarray:
    return (
        (GRID_X >= x_min)
        & (GRID_X <= x_max)
        & (GRID_Y >= y_min)
        & (GRID_Y <= y_max)
    )


def _shift_mask(mask: np.ndarray, row_shift: int, column_shift: int) -> np.ndarray:
    shifted = np.zeros_like(mask, dtype=bool)
    source_rows = slice(max(0, -row_shift), min(GRID_HEIGHT, GRID_HEIGHT - row_shift))
    source_columns = slice(
        max(0, -column_shift), min(GRID_WIDTH, GRID_WIDTH - column_shift)
    )
    target_rows = slice(max(0, row_shift), min(GRID_HEIGHT, GRID_HEIGHT + row_shift))
    target_columns = slice(
        max(0, column_shift), min(GRID_WIDTH, GRID_WIDTH + column_shift)
    )
    shifted[target_rows, target_columns] = mask[source_rows, source_columns]
    return shifted


def _dilate(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    for row_shift in range(-radius_cells, radius_cells + 1):
        for column_shift in range(-radius_cells, radius_cells + 1):
            if row_shift**2 + column_shift**2 <= radius_cells**2:
                result |= _shift_mask(mask, row_shift, column_shift)
    return result


def _boundary(mask: np.ndarray) -> np.ndarray:
    expanded = _dilate(mask, 1)
    eroded = np.asarray(mask, dtype=bool).copy()
    for row_shift, column_shift in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        eroded &= _shift_mask(mask, row_shift, column_shift)
    return expanded ^ eroded


def _scenario_geometry(
    spec: ScenarioSpec, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    effective_width = spec.corridor_width_m + float(rng.uniform(-0.18, 0.18))
    half_width = effective_width / 2.0
    wall_thickness = 0.16
    forward = np.clip(GRID_X, 0.0, 5.2)
    centerline = (
        float(rng.uniform(-0.10, 0.10))
        + float(rng.uniform(-0.36, 0.36)) * forward
        + float(rng.uniform(-0.020, 0.020)) * np.square(forward)
    )
    wall_mask = (
        (np.abs(GRID_Y - centerline) >= half_width)
        & (np.abs(GRID_Y - centerline) <= half_width + wall_thickness)
        & (GRID_X >= GRID_X_MIN_M)
    )
    clutter_mask = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)
    # Every synthetic scene receives a randomized far-side fixture. It keeps
    # nominally open corridors distinct across independent sessions without
    # changing the near-field navigation task.
    fixture_side = -1.0 if rng.random() < 0.5 else 1.0
    fixture_x = float(rng.uniform(2.4, 4.8))
    fixture_centerline = float(
        np.mean(centerline[np.abs(GRID_X - fixture_x) < 0.08])
    )
    fixture_y = fixture_centerline + fixture_side * (half_width - 0.12)
    clutter_mask |= _circle(
        fixture_x,
        fixture_y,
        float(rng.uniform(0.08, 0.17)),
    )
    if spec.clutter:
        side = -1.0 if rng.random() < 0.5 else 1.0
        # A near-field object forces the nine trajectory labels to encode
        # spatially different avoidance choices inside the 1.2 s horizon.
        # The farther cart/pillar still exercise long-range occupancy.
        near_x = float(rng.uniform(0.55, 1.00))
        near_y = float(rng.uniform(-0.48, 0.48))
        clutter_mask |= _rectangle(
            near_x - 0.16,
            near_x + 0.16,
            near_y - 0.14,
            near_y + 0.14,
        )
        cart_y = side * (half_width - 0.38)
        cart_x = float(rng.uniform(1.7, 3.2))
        clutter_mask |= _rectangle(
            cart_x - 0.28,
            cart_x + 0.28,
            cart_y - 0.22,
            cart_y + 0.22,
        )
        pillar_y = -side * (half_width - 0.18)
        clutter_mask |= _circle(float(rng.uniform(3.4, 4.5)), pillar_y, 0.16)
    static_occupancy = wall_mask | clutter_mask
    return wall_mask, clutter_mask, static_occupancy, centerline[:, 0].copy()


def _person_state(
    motion: str, time_s: float, rng_values: tuple[float, float, float]
) -> tuple[float, float, float, float] | None:
    phase, speed_scale, lateral_offset = rng_values
    if motion == "crossing":
        x0 = 0.82 + 0.16 * math.sin(phase)
        y0 = -0.72 + lateral_offset
        vx = 0.0
        vy = 0.80 * speed_scale
    elif motion == "leading":
        x0 = 0.62 + 0.14 * math.cos(phase)
        y0 = 0.30 * math.sin(phase) + lateral_offset
        vx = 0.24 * speed_scale
        vy = 0.0
    else:
        return None
    return x0 + vx * time_s, y0 + vy * time_s, vx, vy


def _sensor_coverage() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    range_m = np.sqrt(GRID_X**2 + GRID_Y**2)
    lidar_coverage = range_m <= 5.6
    depth_coverage = (
        (GRID_X >= 0.20)
        & (GRID_X <= 4.6)
        & (np.abs(GRID_Y) <= np.maximum(0.25, GRID_X * math.tan(math.radians(52))))
    )
    vision_coverage = (
        (GRID_X >= 0.30)
        & (GRID_X <= 5.2)
        & (np.abs(GRID_Y) <= np.maximum(0.18, GRID_X * math.tan(math.radians(38))))
    )
    return lidar_coverage, depth_coverage, vision_coverage


LIDAR_COVERAGE, DEPTH_COVERAGE, VISION_COVERAGE = _sensor_coverage()
DEPTH_RANGE_M = np.sqrt(GRID_X**2 + GRID_Y**2)


def _make_validity(
    spec: ScenarioSpec,
    episode_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    validity = np.ones((HISTORY_FRAMES, len(SENSOR_NAMES)), dtype=np.uint8)
    provenance = np.full(
        (HISTORY_FRAMES, len(SENSOR_NAMES)), "synthetic", dtype="<U32"
    )
    sensor_age_s = np.zeros(
        (HISTORY_FRAMES, len(SENSOR_NAMES)), dtype=np.float32
    )
    for frame_index in range(HISTORY_FRAMES):
        sensor_age_s[frame_index] = np.asarray(
            (0.015 + 0.002 * frame_index, 0.030, 0.080), dtype=np.float32
        )
    if spec.force_dropout:
        modality_index = episode_index % len(SENSOR_NAMES)
        first_dropped_frame = 1 + (episode_index % 3)
        validity[first_dropped_frame:, modality_index] = 0
        provenance[first_dropped_frame:, modality_index] = "modality_dropout"
        sensor_age_s[first_dropped_frame:, modality_index] = -1.0
        if episode_index % 4 == 3:
            second_modality = (modality_index + 1) % len(SENSOR_NAMES)
            validity[-1, second_modality] = 0
            provenance[-1, second_modality] = "modality_dropout"
            sensor_age_s[-1, second_modality] = -1.0
    return validity, provenance, sensor_age_s


def _history_tensor(
    *,
    clutter_mask: np.ndarray,
    static_occupancy: np.ndarray,
    person_motion: str,
    person_rng_values: tuple[float, float, float],
    sensor_validity: np.ndarray,
) -> np.ndarray:
    tribev = np.zeros(
        (HISTORY_FRAMES, 8, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32
    )
    for frame_index, time_s in enumerate(HISTORY_OFFSETS_S):
        person_state = _person_state(
            person_motion, float(time_s), person_rng_values
        )
        person_mask = (
            _circle(person_state[0], person_state[1], 0.22)
            if person_state
            else np.zeros_like(static_occupancy)
        )
        occupied = static_occupancy | person_mask

        if sensor_validity[frame_index, 0]:
            tribev[frame_index, 0] = (occupied & LIDAR_COVERAGE).astype(
                np.float32
            )
            tribev[frame_index, 1] = LIDAR_COVERAGE.astype(np.float32)

        if sensor_validity[frame_index, 1]:
            depth_occupied = occupied & DEPTH_COVERAGE
            near = depth_occupied & (DEPTH_RANGE_M < 1.50)
            mid = (
                depth_occupied
                & (DEPTH_RANGE_M >= 1.50)
                & (DEPTH_RANGE_M < 3.00)
            )
            far = depth_occupied & (DEPTH_RANGE_M >= 3.00)
            tribev[frame_index, 2] = near.astype(np.float32)
            tribev[frame_index, 3] = mid.astype(np.float32)
            tribev[frame_index, 4] = far.astype(np.float32)

        if sensor_validity[frame_index, 2]:
            semantic = np.zeros_like(GRID_X, dtype=np.float32)
            semantic[clutter_mask & VISION_COVERAGE] = 0.65
            semantic[_dilate(person_mask, 1) & VISION_COVERAGE] = 1.0
            tribev[frame_index, 5] = semantic

        tribev[frame_index, 6] = float(
            np.mean(sensor_validity[frame_index])
        )
        # Current-frame fusion only. Future occupancy/dynamic labels are never
        # read while constructing this model input channel.
        tribev[frame_index, 7] = np.maximum.reduce(
            (
                tribev[frame_index, 0],
                tribev[frame_index, 2],
                tribev[frame_index, 3],
                tribev[frame_index, 4],
                tribev[frame_index, 5],
            )
        )
    return tribev


def _future_targets(
    *,
    static_occupancy: np.ndarray,
    person_motion: str,
    person_rng_values: tuple[float, float, float],
    sensor_validity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    occupancy = np.zeros(
        (FUTURE_HORIZONS, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32
    )
    flow = np.zeros(
        (FUTURE_HORIZONS, 2, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32
    )
    dynamic = np.zeros_like(occupancy)
    uncertainty = np.zeros_like(occupancy)
    reference_state = _person_state(person_motion, 0.0, person_rng_values)

    coverage = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)
    if sensor_validity[-1, 0]:
        coverage |= LIDAR_COVERAGE
    if sensor_validity[-1, 1]:
        coverage |= DEPTH_COVERAGE
    if sensor_validity[-1, 2]:
        coverage |= VISION_COVERAGE

    for horizon_index, horizon_s in enumerate(FUTURE_HORIZONS_S):
        future_state = _person_state(
            person_motion, float(horizon_s), person_rng_values
        )
        person_mask = (
            _circle(future_state[0], future_state[1], 0.22)
            if future_state
            else np.zeros_like(static_occupancy)
        )
        occupied = static_occupancy | person_mask
        occupancy[horizon_index] = occupied.astype(np.float32)
        dynamic[horizon_index] = person_mask.astype(np.float32)
        if future_state and reference_state:
            flow[horizon_index, 0, person_mask] = (
                future_state[0] - reference_state[0]
            )
            flow[horizon_index, 1, person_mask] = (
                future_state[1] - reference_state[1]
            )
        uncertainty[horizon_index] = np.clip(
            0.10
            + 0.35 * _boundary(occupied).astype(np.float32)
            + 0.45 * (~coverage).astype(np.float32)
            + 0.10 * person_mask.astype(np.float32),
            0.0,
            1.0,
        )
    return occupancy, flow, dynamic, uncertainty


VEHICLE_LENGTH_M = 0.50
VEHICLE_WIDTH_M = 0.40
VEHICLE_SAFETY_MARGIN_M = 0.08


def _vehicle_footprint_offsets() -> np.ndarray:
    half_length = VEHICLE_LENGTH_M / 2.0 + VEHICLE_SAFETY_MARGIN_M
    half_width = VEHICLE_WIDTH_M / 2.0 + VEHICLE_SAFETY_MARGIN_M
    local_x = np.arange(
        -half_length,
        half_length + GRID_RESOLUTION_M * 0.5,
        GRID_RESOLUTION_M,
        dtype=np.float32,
    )
    local_y = np.arange(
        -half_width,
        half_width + GRID_RESOLUTION_M * 0.5,
        GRID_RESOLUTION_M,
        dtype=np.float32,
    )
    xx, yy = np.meshgrid(local_x, local_y, indexing="ij")
    return np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1)


VEHICLE_FOOTPRINT_OFFSETS = _vehicle_footprint_offsets()


def _sample_footprint_risk(
    occupancy: np.ndarray,
    x_m: float,
    y_m: float,
    yaw_rad: float,
) -> float:
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    offsets = VEHICLE_FOOTPRINT_OFFSETS
    world_x = x_m + cosine * offsets[:, 0] - sine * offsets[:, 1]
    world_y = y_m + sine * offsets[:, 0] + cosine * offsets[:, 1]
    rows = np.floor((world_x - GRID_X_MIN_M) / GRID_RESOLUTION_M).astype(
        np.int64
    )
    columns = np.floor((world_y - GRID_Y_MIN_M) / GRID_RESOLUTION_M).astype(
        np.int64
    )
    inside = (
        (rows >= 0)
        & (rows < GRID_HEIGHT)
        & (columns >= 0)
        & (columns < GRID_WIDTH)
    )
    if not np.all(inside):
        return 1.0
    return float(np.max(occupancy[rows, columns]))


def _trajectory_soft_labels(
    future_occupancy: np.ndarray,
    reference_centerline_y_m: np.ndarray,
) -> np.ndarray:
    if reference_centerline_y_m.shape != (GRID_HEIGHT,):
        raise ValueError("reference_centerline_y_m must contain one value per BEV row")
    speed_m_s = 0.65
    evaluation_times = np.linspace(0.10, 1.20, 12, dtype=np.float32)
    scores = np.zeros(TRAJECTORY_TOKENS, dtype=np.float64)
    for token_index, omega in enumerate(TRAJECTORY_OMEGA_RAD_S):
        risks = []
        path_errors = []
        for time_s in evaluation_times:
            if abs(float(omega)) < 1e-6:
                x_m = speed_m_s * float(time_s)
                y_m = 0.0
            else:
                radius = speed_m_s / float(omega)
                x_m = radius * math.sin(float(omega) * float(time_s))
                y_m = radius * (
                    1.0 - math.cos(float(omega) * float(time_s))
                )
            yaw_rad = float(omega) * float(time_s)
            horizon_index = min(
                FUTURE_HORIZONS - 1,
                int(math.ceil(float(time_s) / 0.4)) - 1,
            )
            risks.append(
                _sample_footprint_risk(
                    future_occupancy[horizon_index],
                    x_m,
                    y_m,
                    yaw_rad,
                )
            )
            row = int(
                np.clip(
                    math.floor(
                        (x_m - GRID_X_MIN_M) / GRID_RESOLUTION_M
                    ),
                    0,
                    GRID_HEIGHT - 1,
                )
            )
            reference_y = float(reference_centerline_y_m[row])
            path_errors.append(abs(y_m - reference_y))
        collision_cost = 7.0 * max(risks) + 1.5 * float(np.mean(risks))
        path_alignment_cost = (
            2.8 * float(np.mean(path_errors))
            + 1.2 * float(path_errors[-1])
        )
        turn_cost = 0.08 * abs(float(omega))
        scores[token_index] = -(
            collision_cost + path_alignment_cost + turn_cost
        )
    scores -= scores.max()
    probabilities = np.exp(scores / 0.45)
    probabilities /= probabilities.sum()
    return probabilities.astype(np.float32)


def generate_episode(
    *,
    scenario_id: str,
    session_id: str,
    episode_index: int,
    seed: int,
    base_timestamp_ns: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Generate one deterministic episode from a scenario and seed."""

    if scenario_id not in SCENARIOS:
        raise ValueError(f"unknown scenario_id={scenario_id!r}")
    spec = SCENARIOS[scenario_id]
    episode_seed = (
        int(seed)
        + SCENARIO_IDS.index(scenario_id) * 1_000_003
        + int(episode_index) * 10_007
    )
    rng = np.random.default_rng(episode_seed)
    (
        _wall_mask,
        clutter_mask,
        static_occupancy,
        reference_centerline_y_m,
    ) = _scenario_geometry(spec, rng)
    person_rng_values = (
        float(rng.uniform(-math.pi, math.pi)),
        float(rng.uniform(0.85, 1.15)),
        float(rng.uniform(-0.12, 0.12)),
    )
    validity, provenance, sensor_age_s = _make_validity(spec, episode_index)
    tribev = _history_tensor(
        clutter_mask=clutter_mask,
        static_occupancy=static_occupancy,
        person_motion=spec.person_motion,
        person_rng_values=person_rng_values,
        sensor_validity=validity,
    )
    future_occupancy, future_flow, dynamic, uncertainty = _future_targets(
        static_occupancy=static_occupancy,
        person_motion=spec.person_motion,
        person_rng_values=person_rng_values,
        sensor_validity=validity,
    )
    soft_labels = _trajectory_soft_labels(
        future_occupancy,
        reference_centerline_y_m,
    )

    reference_timestamp_ns = int(base_timestamp_ns)
    timestamps_ns = reference_timestamp_ns + np.rint(
        HISTORY_OFFSETS_S.astype(np.float64) * 1e9
    ).astype(np.int64)
    future_timestamps_ns = reference_timestamp_ns + np.rint(
        FUTURE_HORIZONS_S.astype(np.float64) * 1e9
    ).astype(np.int64)
    episode_id = f"{session_id}-e{episode_index:04d}"
    arrays = {
        "timestamps_ns": timestamps_ns.astype(np.int64),
        "history_offsets_s": HISTORY_OFFSETS_S.copy(),
        "future_timestamps_ns": future_timestamps_ns.astype(np.int64),
        "future_horizons_s": FUTURE_HORIZONS_S.copy(),
        "tribev_input": tribev.astype(np.float32),
        "future_occupancy": future_occupancy.astype(np.float32),
        "future_flow_m": future_flow.astype(np.float32),
        "dynamic_mask": dynamic.astype(np.float32),
        "uncertainty_target": uncertainty.astype(np.float32),
        "trajectory_soft_labels": soft_labels.astype(np.float32),
        "trajectory_token_omega_rad_s": TRAJECTORY_OMEGA_RAD_S.copy(),
        "sensor_validity": validity.astype(np.uint8),
        "sensor_age_s": sensor_age_s.astype(np.float32),
        "sensor_provenance": provenance.astype("<U32"),
        "vision_image_supplied": np.zeros(HISTORY_FRAMES, dtype=np.uint8),
    }
    metadata = build_episode_metadata(
        episode_id=episode_id,
        session_id=session_id,
        scenario_id=scenario_id,
        source={
            "kind": "synthetic",
            "license_id": "project-generated-synthetic-v1",
            "contains_personal_data": False,
            "consent_status": "not_applicable",
        },
        generator={
            "name": "x5_tribev_flow.synthetic",
            "version": "4",
            "scenario_spec": {
                "corridor_width_m": spec.corridor_width_m,
                "clutter": spec.clutter,
                "person_motion": spec.person_motion,
                "force_dropout": spec.force_dropout,
                "vehicle_footprint_m": [
                    VEHICLE_LENGTH_M,
                    VEHICLE_WIDTH_M,
                ],
                "vehicle_safety_margin_m": VEHICLE_SAFETY_MARGIN_M,
                "trajectory_speed_m_s": 0.65,
                "trajectory_teacher": (
                    "footprint_collision_plus_visible_corridor_alignment"
                ),
            },
        },
        seed=episode_seed,
        notes=(
            "Synthetic semantics are pipeline fixtures, not live 4K evidence.",
            "Synthetic episodes must not be used as final real-world accuracy proof.",
        ),
    )
    return arrays, metadata


def generate_dataset(
    output_dir: str | Path,
    *,
    scenarios: Sequence[str] = SCENARIO_IDS,
    sessions_per_scenario: int = 3,
    episodes_per_session: int = 4,
    seed: int = 20260728,
    overwrite: bool = False,
    compressed: bool = True,
) -> dict[str, Any]:
    """Generate a reproducible multi-session synthetic corpus."""

    if sessions_per_scenario <= 0 or episodes_per_session <= 0:
        raise ValueError("sessions_per_scenario and episodes_per_session must be > 0")
    unknown = sorted(set(scenarios) - set(SCENARIO_IDS))
    if unknown:
        raise ValueError(f"unknown scenarios: {unknown}")
    destination = Path(output_dir).expanduser().resolve()
    written: list[Path] = []
    for scenario_index, scenario_id in enumerate(scenarios):
        for session_index in range(sessions_per_scenario):
            session_id = (
                f"syn-{scenario_id.replace('_', '-')}-"
                f"s{session_index:03d}-seed{seed}"
            )
            session_base_timestamp_ns = (
                1_700_000_000_000_000_000
                + scenario_index * 10_000_000_000_000
                + session_index * 100_000_000_000
            )
            for episode_index in range(episodes_per_session):
                reference_timestamp_ns = (
                    session_base_timestamp_ns + episode_index * 5_000_000_000
                )
                arrays, metadata = generate_episode(
                    scenario_id=scenario_id,
                    session_id=session_id,
                    episode_index=episode_index,
                    seed=seed + session_index * 100_003,
                    base_timestamp_ns=reference_timestamp_ns,
                )
                path = (
                    destination
                    / scenario_id
                    / session_id
                    / f"{metadata['episode_id']}.npz"
                )
                written.append(
                    save_episode(
                        path,
                        arrays,
                        metadata,
                        overwrite=overwrite,
                        compressed=compressed,
                    )
                )

    refs = build_episode_refs(written)
    splits = split_episode_refs(refs, seed=seed)
    split_by_episode = {
        ref.episode_id: split_name
        for split_name, split_refs in splits.items()
        for ref in split_refs
    }
    records: list[dict[str, Any]] = []
    input_digests: list[str] = []
    for ref in refs:
        target_digest = hashlib.sha256()
        with np.load(ref.path, allow_pickle=False) as archive:
            for name in (
                "future_occupancy",
                "future_flow_m",
                "dynamic_mask",
                "uncertainty_target",
                "trajectory_soft_labels",
            ):
                value = np.ascontiguousarray(archive[name])
                target_digest.update(name.encode("ascii"))
                target_digest.update(str(value.dtype).encode("ascii"))
                target_digest.update(
                    np.asarray(value.shape, dtype=np.int64).tobytes()
                )
                target_digest.update(value.tobytes())
        input_digest = episode_input_content_sha256(ref.path)
        input_digests.append(input_digest)
        file_digest = hashlib.sha256(ref.path.read_bytes()).hexdigest()
        records.append(
            {
                "path": ref.path.relative_to(destination).as_posix(),
                "episode_id": ref.episode_id,
                "session_id": ref.session_id,
                "scenario_id": ref.scenario_id,
                "split": split_by_episode[ref.episode_id],
                "file_sha256": file_digest,
                "input_content_sha256": input_digest,
                "target_content_sha256": target_digest.hexdigest(),
            }
        )
    duplicate_input_count = len(input_digests) - len(set(input_digests))
    if duplicate_input_count:
        raise RuntimeError(
            f"synthetic corpus contains {duplicate_input_count} duplicate inputs"
        )
    source_path = Path(__file__).resolve()
    manifest = {
        "schema_version": "x5-tribev-synthetic-dataset/1.0",
        "generator": {
            "name": "x5_tribev_flow.synthetic",
            "version": "4",
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "numpy_version": np.__version__,
        },
        "parameters": {
            "seed": seed,
            "scenarios": list(scenarios),
            "sessions_per_scenario": sessions_per_scenario,
            "episodes_per_session": episodes_per_session,
            "compressed": compressed,
        },
        "episode_count": len(records),
        "duplicate_input_count": duplicate_input_count,
        "split_summary": summarize_splits(splits),
        "files": sorted(records, key=lambda row: row["path"]),
        "claim_boundary": (
            "Project-generated synthetic oracle labels validate training and "
            "conversion only; they are not real navigation accuracy evidence."
        ),
    }
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "dataset_manifest.json"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".dataset_manifest.",
        suffix=".tmp",
        dir=destination,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, manifest_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return {
        "output_dir": str(destination),
        "seed": seed,
        "scenarios": list(scenarios),
        "sessions_per_scenario": sessions_per_scenario,
        "episodes_per_session": episodes_per_session,
        "episodes_written": len(written),
        "split_summary": summarize_splits(splits),
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "duplicate_input_count": duplicate_input_count,
        "claim_boundary": (
            "Synthetic episodes validate the contract and training pipeline only; "
            "they are not real sensor or navigation performance evidence."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=SCENARIO_IDS,
        help="Repeat to select scenarios; default generates all six.",
    )
    parser.add_argument("--sessions-per-scenario", type=int, default=3)
    parser.add_argument("--episodes-per-session", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--uncompressed", action="store_true")
    args = parser.parse_args(argv)
    result = generate_dataset(
        args.output,
        scenarios=args.scenario or SCENARIO_IDS,
        sessions_per_scenario=args.sessions_per_scenario,
        episodes_per_session=args.episodes_per_session,
        seed=args.seed,
        overwrite=args.overwrite,
        compressed=not args.uncompressed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
