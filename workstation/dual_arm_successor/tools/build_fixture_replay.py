#!/usr/bin/env python3
"""Build command-derived dual-arm fixture trajectories without hardware access."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATUS = "FIXTURE_REPLAY_NOT_REAL_POLICY"
ROW_SCHEMA = "xrd-dual-arm-fixture-replay-row-v1"
MANIFEST_SCHEMA = "xrd-dual-arm-fixture-replay-manifest-v1"
PROVENANCE = "COMMAND_DERIVED_DIGITAL_TWIN"
ZERO_AUTHORITY = {
    "motion_authority": False,
    "execution_allowed": False,
    "actuator_commands_issued": 0,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def literal_assignments(path: Path, names: set[str]) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value_node = node.value
        for target in targets:
            if isinstance(target, ast.Name) and target.id in names:
                found[target.id] = ast.literal_eval(value_node)
    missing = sorted(names - set(found))
    if missing:
        raise ValueError(f"{path}: missing literal assignments {missing}")
    return found


def wrap_deg(value: float) -> float:
    wrapped = (value + 180.0) % 360.0 - 180.0
    return 180.0 if math.isclose(wrapped, -180.0) and value > 0 else wrapped


def circular_interpolate(start: float, target: float, fraction: float) -> float:
    delta = (target - start + 180.0) % 360.0 - 180.0
    return wrap_deg(start + delta * fraction)


def smoothstep(fraction: float) -> float:
    value = min(max(fraction, 0.0), 1.0)
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


def composed_state(
    arm01: list[float], gripper: float, arm02: list[float]
) -> list[float]:
    if len(arm01) != 6 or len(arm02) != 6:
        raise ValueError("six joints required for each arm")
    return [*map(float, arm01), float(gripper), *map(float, arm02)]


@dataclass
class Timeline:
    episode_id: str
    parent_episode_id: str
    task: str
    instruction: str
    rate_hz: float
    rng: random.Random
    nominal: list[float]
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.dt = 1.0 / self.rate_hz

    def segment(
        self,
        target: list[float],
        duration_s: float,
        stage: str,
        *,
        jitter_scale: float,
    ) -> None:
        if len(target) != 13:
            raise ValueError(f"{stage}: expected 13-dimensional target")
        duration = max(self.dt * 2, duration_s * (1.0 + self.rng.uniform(-0.04, 0.04)))
        steps = max(2, int(round(duration * self.rate_hz)))
        start = list(self.nominal)
        adjusted = list(target)
        for index in (*range(6), *range(7, 13)):
            adjusted[index] = wrap_deg(
                adjusted[index] + self.rng.gauss(0.0, jitter_scale)
            )
        adjusted[6] = min(max(adjusted[6], 0.0), 1.0)
        for step in range(steps):
            fraction = smoothstep((step + 1) / steps)
            next_nominal: list[float] = []
            for index, (left, right) in enumerate(zip(start, adjusted)):
                if index == 6:
                    next_nominal.append(left + (right - left) * fraction)
                else:
                    next_nominal.append(circular_interpolate(left, right, fraction))
            observed = list(self.nominal)
            for index in (*range(6), *range(7, 13)):
                observed[index] = wrap_deg(
                    observed[index] + self.rng.gauss(0.0, 0.06)
                )
            row = {
                "schema_version": ROW_SCHEMA,
                "status": STATUS,
                "provenance_state": PROVENANCE,
                "episode_id": self.episode_id,
                "parent_episode_id": self.parent_episode_id,
                "task": self.task,
                "language_instruction": self.instruction,
                "timestamp": round(self.timestamp, 6),
                "stage": stage,
                "observation_state": [round(value, 6) for value in observed],
                "action": [round(value, 6) for value in next_nominal],
                "measured_robot_telemetry": False,
                "camera_frame_synchronized": False,
                "real_robot_policy": False,
                **ZERO_AUTHORITY,
            }
            self.rows.append(row)
            self.nominal = next_nominal
            self.timestamp += self.dt


def duration(base: float, rng: random.Random) -> float:
    return base * (1.0 + rng.uniform(-0.03, 0.03))


def single_arm_episode(
    episode_index: int,
    seed: int,
    rate_hz: float,
    points: dict[str, list[float]],
    instruction: str,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    start = composed_state(points["START"], 0.0, points["RIGHT_START"])
    timeline = Timeline(
        episode_id=f"fixture-single-{episode_index:03d}",
        parent_episode_id=f"fixture-single-parent-{episode_index:03d}",
        task="single_arm_visual_redundancy",
        instruction=instruction,
        rate_hz=rate_hz,
        rng=rng,
        nominal=start,
    )
    timeline.segment(start, duration(0.5, rng), "SINGLE_START_HOLD", jitter_scale=0.05)
    observe = composed_state(points["OBSERVE"], 0.0, points["RIGHT_START"])
    timeline.segment(
        observe, duration(2.4, rng), "SINGLE_START_TO_OBSERVE", jitter_scale=0.20
    )
    timeline.segment(
        observe, duration(3.0, rng), "SINGLE_OBSERVE_VISUAL_HOLD", jitter_scale=0.05
    )
    timeline.segment(
        start, duration(2.4, rng), "SINGLE_OBSERVE_TO_START", jitter_scale=0.20
    )
    timeline.segment(start, duration(0.5, rng), "SINGLE_DONE", jitter_scale=0.04)
    return timeline.rows


def dual_arm_episode(
    episode_index: int,
    seed: int,
    rate_hz: float,
    points: dict[str, list[float]],
    instruction: str,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    right_start = points["RIGHT_START"]
    start = composed_state(points["START"], 0.0, right_start)
    timeline = Timeline(
        episode_id=f"fixture-dual-{episode_index:03d}",
        parent_episode_id=f"fixture-dual-parent-{episode_index:03d}",
        task="dual_arm_bag_grind",
        instruction=instruction,
        rate_hz=rate_hz,
        rng=rng,
        nominal=start,
    )
    timeline.segment(start, duration(0.4, rng), "DUAL_START_HOLD", jitter_scale=0.04)
    opened = composed_state(points["START"], 1.0, right_start)
    timeline.segment(opened, duration(0.7, rng), "ARM01_GRIPPER_OPEN", jitter_scale=0.03)
    pick_open = composed_state(points["PICK"], 1.0, right_start)
    timeline.segment(
        pick_open, duration(2.5, rng), "ARM01_START_TO_PICK", jitter_scale=0.20
    )
    pick_closed = composed_state(points["PICK"], 0.0, right_start)
    timeline.segment(
        pick_closed, duration(1.5, rng), "ARM01_BAG_GRIP_HOLD", jitter_scale=0.03
    )
    compact_pick = composed_state(points["COMPACT_PICK_BRANCH"], 0.0, right_start)
    timeline.segment(
        compact_pick, duration(2.0, rng), "ARM01_PICK_TO_COMPACT", jitter_scale=0.18
    )
    compact_dish = composed_state(points["COMPACT_DISH_BRANCH"], 0.0, right_start)
    timeline.segment(
        compact_dish,
        duration(2.7, rng),
        "ARM01_COMPACT_SWITCH_TO_DISH",
        jitter_scale=0.20,
    )
    for step in range(1, 7):
        fraction = step / 6.0
        approach = [
            circular_interpolate(left, right, fraction)
            for left, right in zip(points["COMPACT_DISH_BRANCH"], points["DISH_DROP"])
        ]
        timeline.segment(
            composed_state(approach, 0.0, right_start),
            duration(0.9, rng),
            f"ARM01_SLOW_DISH_APPROACH_{step:02d}",
            jitter_scale=0.10,
        )
    dish_open = composed_state(points["DISH_DROP"], 1.0, right_start)
    timeline.segment(
        dish_open, duration(3.0, rng), "ARM01_BAG_RELEASE", jitter_scale=0.04
    )

    # The verified overlap starts after the left arm clears vertically.
    right_work = points["RIGHT_GRIND_WORK"]
    clear_and_right_work = composed_state(
        points["COMPACT_DISH_BRANCH"], 1.0, right_work
    )
    timeline.segment(
        clear_and_right_work,
        duration(4.0, rng),
        "OVERLAP_LEFT_CLEAR_RIGHT_TO_WORK",
        jitter_scale=0.20,
    )
    left_top = composed_state(points["COMPACT_START_BRANCH"], 1.0, right_work)
    timeline.segment(
        left_top,
        duration(2.6, rng),
        "OVERLAP_LEFT_TOP_RETURN_RIGHT_WORK_HOLD",
        jitter_scale=0.16,
    )

    right_low = list(right_work)
    right_low[5] = points["GRIND_LOW_J6"]
    left_start_right_low = composed_state(points["START"], 1.0, right_low)
    timeline.segment(
        left_start_right_low,
        duration(2.5, rng),
        "OVERLAP_LEFT_START_RIGHT_CYCLE_01_FORWARD",
        jitter_scale=0.16,
    )
    for cycle in range(1, 5):
        high = composed_state(points["START"], 1.0, right_work)
        timeline.segment(
            high,
            duration(1.5, rng),
            f"ARM02_GRIND_CYCLE_{cycle:02d}_RETURN",
            jitter_scale=0.10,
        )
        if cycle < 4:
            low = composed_state(points["START"], 1.0, right_low)
            timeline.segment(
                low,
                duration(1.5, rng),
                f"ARM02_GRIND_CYCLE_{cycle + 1:02d}_FORWARD",
                jitter_scale=0.10,
            )
    both_start = composed_state(points["START"], 1.0, right_start)
    timeline.segment(
        both_start, duration(4.0, rng), "ARM02_RETURN_START", jitter_scale=0.18
    )
    final = composed_state(points["START"], 0.0, right_start)
    timeline.segment(final, duration(0.7, rng), "DUAL_DONE", jitter_scale=0.04)
    return timeline.rows


def load_sources(repo_root: Path, source_config: Path) -> tuple[dict[str, Any], dict[str, list[float]]]:
    config = read_object(source_config)
    resolved: dict[str, Path] = {}
    source_receipts: dict[str, Any] = {}
    for name, spec in config["sources"].items():
        path = repo_root / spec["path"]
        actual = sha256_file(path)
        if actual != spec["sha256"]:
            raise RuntimeError(
                f"frozen source mismatch for {name}: expected {spec['sha256']}, got {actual}"
            )
        resolved[name] = path
        source_receipts[name] = {
            "path": spec["path"],
            "sha256": actual,
            "bytes": path.stat().st_size,
        }
    arm01 = literal_assignments(
        resolved["arm01"],
        {
            "START",
            "PICK",
            "DISH_DROP",
            "COMPACT_PICK_BRANCH",
            "COMPACT_DISH_BRANCH",
            "COMPACT_START_BRANCH",
        },
    )
    arm02 = literal_assignments(
        resolved["arm02"],
        {
            "RIGHT_START",
            "RIGHT_GRIND_WORK",
            "GRIND_LOW_J6",
            "GRIND_HIGH_J6",
            "DEFAULT_GRIND_CYCLES",
        },
    )
    if int(arm02["DEFAULT_GRIND_CYCLES"]) != 4:
        raise RuntimeError("fixture contract requires the frozen four-cycle grind")
    station = read_object(resolved["station"])
    observe = station["finals_motion_profile"]["arm01_named_points_deg"]["OBSERVE"]
    points = {
        **{name: [float(item) for item in value] for name, value in arm01.items()},
        "OBSERVE": [float(item) for item in observe],
        "RIGHT_START": [float(item) for item in arm02["RIGHT_START"]],
        "RIGHT_GRIND_WORK": [float(item) for item in arm02["RIGHT_GRIND_WORK"]],
        "GRIND_LOW_J6": float(arm02["GRIND_LOW_J6"]),
        "GRIND_HIGH_J6": float(arm02["GRIND_HIGH_J6"]),
    }
    return {"config": config, "sources": source_receipts}, points


def build_dataset(
    repo_root: Path,
    source_config: Path,
    output_dir: Path,
    *,
    episodes_per_task: int,
    rate_hz: float,
    seed: int,
) -> dict[str, Any]:
    source_bundle, points = load_sources(repo_root, source_config)
    instructions = {
        name: value["instruction"]
        for name, value in source_bundle["config"]["tasks"].items()
    }
    rows: list[dict[str, Any]] = []
    for index in range(episodes_per_task):
        rows.extend(
            single_arm_episode(
                index,
                seed + index * 17,
                rate_hz,
                points,
                instructions["single_arm_visual_redundancy"],
            )
        )
        rows.extend(
            dual_arm_episode(
                index,
                seed + 100_000 + index * 29,
                rate_hz,
                points,
                instructions["dual_arm_bag_grind"],
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "fixture_replay.jsonl"
    dataset_path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )
    task_counts = Counter(row["task"] for row in rows)
    episode_ids = {row["episode_id"] for row in rows}
    parent_ids = {row["parent_episode_id"] for row in rows}
    stage_counts = Counter(row["stage"] for row in rows)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "created_at": utc_now(),
        "status": STATUS,
        "dataset": {
            "path": dataset_path.name,
            "sha256": sha256_file(dataset_path),
            "bytes": dataset_path.stat().st_size,
            "rows": len(rows),
            "episodes": len(episode_ids),
            "parent_episodes": len(parent_ids),
            "episodes_per_task": episodes_per_task,
            "task_row_counts": dict(sorted(task_counts.items())),
            "stage_row_counts": dict(sorted(stage_counts.items())),
            "rate_hz": rate_hz,
            "state_dimension": 13,
            "action_dimension": 13,
            "action_semantics": "next_commanded_state",
        },
        "generation": {
            "seed": seed,
            "command_derived": True,
            "bounded_waypoint_jitter": True,
            "bounded_timing_jitter": True,
            "circular_joint_interpolation": True,
            "measured_robot_telemetry": False,
            "camera_frames_synchronized": False,
        },
        "sources": source_bundle["sources"],
        "source_config": {
            "path": str(source_config.resolve()),
            "sha256": sha256_file(source_config),
        },
        "real_robot_policy": False,
        "deployment_eligible": False,
        **ZERO_AUTHORITY,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    candidate = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument(
        "--source-config",
        type=Path,
        default=candidate / "config" / "fixture_sources_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=30)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.episodes_per_task < 2:
        raise SystemExit("--episodes-per-task must be at least 2")
    if not 5.0 <= args.rate_hz <= 50.0:
        raise SystemExit("--rate-hz must be between 5 and 50")
    manifest = build_dataset(
        args.repo_root.resolve(),
        args.source_config.resolve(),
        args.output_dir.resolve(),
        episodes_per_task=args.episodes_per_task,
        rate_hz=args.rate_hz,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "rows": manifest["dataset"]["rows"],
                "episodes": manifest["dataset"]["episodes"],
                "dataset_sha256": manifest["dataset"]["sha256"],
                **ZERO_AUTHORITY,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
