"""Leakage-safe adapter for the existing synthetic TriBEV episode bank.

The source episodes remain immutable. This module maps their five-frame,
eight-channel synthetic representation onto the finals-vNext twelve-channel
contract. Derived free/unknown/closing-rate and camera-visibility planes are
explicitly synthetic proxies and are not evidence of real sensor accuracy.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from embodied_brain.finals_vnext.contracts.core import (
    CHANNELS_PER_FRAME,
    HISTORY_FRAMES,
    MODEL_INPUT_SHAPE,
)
from embodied_brain.finals_vnext.world_model.trajectories import (
    rectangular_footprint_risk_labels,
)

SOURCE_DATASET_RELATIVE = Path(
    "embodied_brain/finals_successor/data/"
    "syn_v5"
)
SPLIT_NAMES = ("train", "validation", "calibration", "test")
SPLIT_SESSION_COUNTS = (15, 3, 3, 3)
ADAPTER_SCHEMA = "x5-tribev-v2-synthetic-adapter/1.0"


@dataclass(frozen=True, slots=True)
class EpisodeRefV2:
    path: Path
    episode_id: str
    session_id: str
    scenario_id: str
    split: str
    source_sha256: str


def _scalar_text(value: np.ndarray) -> str:
    return str(np.asarray(value).reshape(()).item())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _session_order_token(seed: int, scenario: str, session: str) -> str:
    return hashlib.sha256(
        f"{seed}:{scenario}:{session}".encode("utf-8")
    ).hexdigest()


def discover_and_split(
    workspace_root: str | Path,
    *,
    seed: int = 20260728,
) -> dict[str, tuple[EpisodeRefV2, ...]]:
    """Discover source NPZs and split whole sessions within each scenario."""

    root = Path(workspace_root).resolve()
    source_root = root / SOURCE_DATASET_RELATIVE
    paths = tuple(sorted(source_root.rglob("*.npz")))
    if not paths:
        raise FileNotFoundError(f"no source episodes under {source_root}")

    rows: list[tuple[Path, str, str, str, str]] = []
    sessions_by_scenario: dict[str, set[str]] = {}
    input_digests: dict[str, set[str]] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            episode = _scalar_text(payload["episode_id"])
            session = _scalar_text(payload["session_id"])
            scenario = _scalar_text(payload["scenario_id"])
            tribev = np.asarray(payload["tribev_input"], dtype=np.float32)
        if tribev.shape != (5, 8, 64, 64):
            raise ValueError(f"unexpected source shape {tribev.shape} in {path}")
        content_sha = hashlib.sha256(tribev.tobytes()).hexdigest()
        rows.append((path, episode, session, scenario, content_sha))
        sessions_by_scenario.setdefault(scenario, set()).add(session)
        input_digests.setdefault(content_sha, set()).add(session)

    duplicate_sessions = {
        digest: sorted(sessions)
        for digest, sessions in input_digests.items()
        if len(sessions) > 1
    }
    if duplicate_sessions:
        raise ValueError(
            "cross-session duplicate inputs detected: "
            + json.dumps(duplicate_sessions, sort_keys=True)
        )

    assignment: dict[tuple[str, str], str] = {}
    expected_sessions = sum(SPLIT_SESSION_COUNTS)
    for scenario, sessions in sorted(sessions_by_scenario.items()):
        ordered = sorted(
            sessions,
            key=lambda session: _session_order_token(seed, scenario, session),
        )
        if len(ordered) != expected_sessions:
            raise ValueError(
                f"{scenario} has {len(ordered)} sessions; "
                f"expected {expected_sessions}"
            )
        offset = 0
        for split, count in zip(
            SPLIT_NAMES, SPLIT_SESSION_COUNTS, strict=True
        ):
            for session in ordered[offset : offset + count]:
                assignment[(scenario, session)] = split
            offset += count

    result: dict[str, list[EpisodeRefV2]] = {
        split: [] for split in SPLIT_NAMES
    }
    source_hash_cache: dict[Path, str] = {}
    for path, episode, session, scenario, _ in rows:
        split = assignment[(scenario, session)]
        source_sha = source_hash_cache.setdefault(path, _sha256_file(path))
        result[split].append(
            EpisodeRefV2(
                path=path,
                episode_id=episode,
                session_id=session,
                scenario_id=scenario,
                split=split,
                source_sha256=source_sha,
            )
        )

    return {
        split: tuple(
            sorted(
                refs,
                key=lambda ref: (
                    ref.scenario_id,
                    ref.session_id,
                    ref.episode_id,
                ),
            )
        )
        for split, refs in result.items()
    }


def _max_pool_2x2(values: np.ndarray) -> np.ndarray:
    if values.shape[-2:] != (64, 64):
        raise ValueError("max-pool input must end in 64x64")
    shape = values.shape[:-2] + (32, 2, 32, 2)
    return values.reshape(shape).max(axis=(-3, -1))


def _mean_pool_2x2(values: np.ndarray) -> np.ndarray:
    if values.shape[-2:] != (64, 64):
        raise ValueError("mean-pool input must end in 64x64")
    shape = values.shape[:-2] + (32, 2, 32, 2)
    return values.reshape(shape).mean(axis=(-3, -1))


def _adapt_history(
    old_history: np.ndarray,
    sensor_validity: np.ndarray,
) -> np.ndarray:
    """Return chronological five-frame, twelve-channel synthetic history."""

    old = np.asarray(old_history, dtype=np.float32)
    validity = np.asarray(sensor_validity, dtype=np.float32)
    if old.shape != (5, 8, 64, 64):
        raise ValueError(f"old history must be 5x8x64x64, got {old.shape}")
    if validity.shape != (5, 3):
        raise ValueError("sensor_validity must be 5x3")

    output = np.zeros((HISTORY_FRAMES, CHANNELS_PER_FRAME, 64, 64), np.float32)
    previous_depth = np.zeros((64, 64), dtype=np.float32)
    for frame_index in range(HISTORY_FRAMES):
        lidar_valid, depth_valid, vision_valid = validity[frame_index]
        lidar_occupancy = old[frame_index, 0] * lidar_valid
        lidar_visibility = old[frame_index, 1] * lidar_valid
        depth_low = old[frame_index, 2] * depth_valid
        depth_mid = old[frame_index, 3] * depth_valid
        depth_high = old[frame_index, 4] * depth_valid
        depth_union = np.maximum.reduce((depth_low, depth_mid, depth_high))
        depth_free = (
            np.clip(lidar_visibility - depth_union, 0.0, 1.0) * depth_valid
        )
        depth_unknown = np.where(
            depth_valid > 0.5,
            np.clip(1.0 - np.maximum(depth_free, depth_union), 0.0, 1.0),
            1.0,
        ).astype(np.float32)
        closing_rate = np.clip(depth_union - previous_depth, 0.0, 1.0)
        if frame_index == 0 or depth_valid < 0.5:
            closing_rate.fill(0.0)
        previous_depth = depth_union
        camera_risk = old[frame_index, 5] * vision_valid
        camera_visibility = lidar_visibility * vision_valid
        validity_fraction = np.full(
            (64, 64), float(np.mean(validity[frame_index])), np.float32
        )
        fused = np.maximum.reduce(
            (
                lidar_occupancy,
                depth_low,
                depth_mid,
                depth_high,
                camera_risk,
            )
        )
        output[frame_index] = np.stack(
            (
                lidar_occupancy,
                lidar_visibility,
                depth_low,
                depth_mid,
                depth_high,
                depth_free,
                depth_unknown,
                closing_rate,
                camera_risk,
                camera_visibility,
                validity_fraction,
                fused,
            )
        )
    return np.clip(output, 0.0, 1.0).astype(np.float32)


def adapt_episode(ref: EpisodeRefV2) -> dict[str, np.ndarray | str]:
    """Load and adapt one immutable source episode."""

    with np.load(ref.path, allow_pickle=False) as payload:
        old_history = np.asarray(payload["tribev_input"], dtype=np.float32)
        validity = np.asarray(payload["sensor_validity"], dtype=np.float32)
        occupancy = np.asarray(payload["future_occupancy"], dtype=np.float32)
        flow = np.asarray(payload["future_flow_m"], dtype=np.float32)
        dynamic = np.asarray(payload["dynamic_mask"], dtype=np.float32)
        uncertainty = np.asarray(
            payload["uncertainty_target"], dtype=np.float32
        )

    chronological = _adapt_history(old_history, validity)
    model_input = chronological[::-1].reshape(60, 64, 64).copy()
    if (1,) + model_input.shape != MODEL_INPUT_SHAPE:
        raise AssertionError(f"adapted model shape mismatch: {model_input.shape}")

    flow_half = _mean_pool_2x2(flow).reshape(6, 32, 32)
    dynamic_half = _max_pool_2x2(dynamic)
    risk = rectangular_footprint_risk_labels(occupancy)
    reliability = np.concatenate(
        (validity[-1].astype(np.float32), np.ones(1, dtype=np.float32))
    )
    return {
        "input": model_input,
        "future_occupancy": occupancy,
        "flow": flow_half.astype(np.float32),
        "flow_mask": np.repeat(dynamic_half, 2, axis=0).astype(np.float32),
        "dynamic": dynamic,
        "uncertainty": uncertainty,
        "trajectory_risk": risk,
        "sensor_reliability": reliability,
        "episode_id": ref.episode_id,
        "session_id": ref.session_id,
        "scenario_id": ref.scenario_id,
        "split": ref.split,
    }


def split_manifest(
    splits: Mapping[str, Sequence[EpisodeRefV2]],
    *,
    workspace_root: str | Path,
    seed: int,
) -> dict[str, object]:
    root = Path(workspace_root).resolve()
    session_sets = {
        split: sorted({ref.session_id for ref in refs})
        for split, refs in splits.items()
    }
    for first_index, first in enumerate(SPLIT_NAMES):
        for second in SPLIT_NAMES[first_index + 1 :]:
            if set(session_sets[first]) & set(session_sets[second]):
                raise AssertionError(f"session leakage between {first} and {second}")
    return {
        "schema_version": ADAPTER_SCHEMA,
        "source_dataset": SOURCE_DATASET_RELATIVE.as_posix(),
        "source_kind": "synthetic_only",
        "seed": seed,
        "history_order": "newest_to_oldest_frame_major",
        "model_input_shape": list(MODEL_INPUT_SHAPE),
        "adapter_limitations": [
            "depth height bins reuse existing synthetic near/mid/far proxy planes",
            "depth free and unknown planes are deterministic synthetic derivations",
            "camera visibility is a synthetic proxy gated by vision validity",
            "odometry reliability target is one for this synthetic source bank",
            "no metric is real-sensor, X5-runtime, or navigation-control evidence",
        ],
        "splits": {
            split: {
                "episode_count": len(refs),
                "session_count": len(session_sets[split]),
                "sessions": session_sets[split],
                "scenarios": sorted({ref.scenario_id for ref in refs}),
                "relative_paths": [
                    ref.path.resolve().relative_to(root).as_posix()
                    for ref in refs
                ],
            }
            for split, refs in splits.items()
        },
    }


class TriBEVV2Dataset:
    """Small lazy PyTorch-compatible dataset without importing torch here."""

    def __init__(self, refs: Iterable[EpisodeRefV2]) -> None:
        self.refs = tuple(refs)

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | str]:
        return adapt_episode(self.refs[index])
