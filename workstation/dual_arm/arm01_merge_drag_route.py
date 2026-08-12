#!/usr/bin/env python3
"""Merge two arm01 drag-teach captures into front-side route waypoints."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path


WAYPOINT_DELTA_DEG = 2.0
MAX_COMMAND_STEP_DEG = 5.0


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mark_index(payload, name):
    for mark in payload["marks"]:
        if mark["name"] == name:
            return int(mark["sample_index"])
    raise RuntimeError(f"missing mark: {name}")


def exact_angles(poses, name):
    pose = poses.get(name)
    if not pose or len(pose.get("angles", [])) < 6:
        raise RuntimeError(f"missing exact pose: {name}")
    return [float(value) for value in pose["angles"][:6]]


def shortest_angle_delta(current, previous):
    return (float(current) - float(previous) + 180.0) % 360.0 - 180.0


def normalize_angle(value):
    normalized = (float(value) + 180.0) % 360.0 - 180.0
    return 180.0 if normalized == -180.0 and value > 0 else normalized


def unwrap_samples(samples):
    unwrapped = [[float(value) for value in samples[0]["angles"][:6]]]
    for sample in samples[1:]:
        previous_raw = [float(value) for value in samples[len(unwrapped) - 1]["angles"][:6]]
        current_raw = [float(value) for value in sample["angles"][:6]]
        previous_unwrapped = unwrapped[-1]
        unwrapped.append(
            [
                previous_unwrapped[joint]
                + shortest_angle_delta(current_raw[joint], previous_raw[joint])
                for joint in range(6)
            ]
        )
    return unwrapped


def lift_near(value, reference):
    return float(reference) + shortest_angle_delta(value, reference)


def warp_to_exact_endpoints(samples, start_angles, end_angles):
    if len(samples) < 3:
        raise RuntimeError(f"segment has only {len(samples)} samples")
    unwrapped = unwrap_samples(samples)
    raw_start = unwrapped[0]
    raw_end = unwrapped[-1]
    lifted_start = [
        lift_near(start_angles[joint], raw_start[joint]) for joint in range(6)
    ]
    lifted_end = [
        lift_near(end_angles[joint], raw_end[joint]) for joint in range(6)
    ]
    warped = []
    denominator = len(samples) - 1
    for index, sample in enumerate(samples):
        ratio = index / denominator
        angles = []
        for joint in range(6):
            raw = unwrapped[index][joint]
            correction = (
                (1.0 - ratio) * (lifted_start[joint] - raw_start[joint])
                + ratio * (lifted_end[joint] - raw_end[joint])
            )
            angles.append(round(normalize_angle(raw + correction), 2))
        warped.append(
            {
                "source_sample_offset": index,
                "source_t_s": float(sample["t_s"]),
                "angles": angles,
            }
        )
    warped[0]["angles"] = [round(value, 2) for value in start_angles]
    warped[-1]["angles"] = [round(value, 2) for value in end_angles]
    return warped


def downsample(samples):
    waypoints = [samples[0]]
    for sample in samples[1:-1]:
        delta = max(
            abs(shortest_angle_delta(current, previous))
            for current, previous in zip(sample["angles"], waypoints[-1]["angles"])
        )
        if delta >= WAYPOINT_DELTA_DEG:
            waypoints.append(sample)
    if waypoints[-1]["angles"] != samples[-1]["angles"]:
        waypoints.append(samples[-1])
    return waypoints


def densify(waypoints):
    dense = [waypoints[0]]
    for current in waypoints[1:]:
        previous = dense[-1]
        deltas = [
            shortest_angle_delta(a, b)
            for a, b in zip(current["angles"], previous["angles"])
        ]
        steps = max(1, int(math.ceil(max(abs(delta) for delta in deltas) / MAX_COMMAND_STEP_DEG)))
        for step in range(1, steps + 1):
            ratio = step / steps
            dense.append(
                {
                    "source_sample_offset": round(
                        float(previous["source_sample_offset"])
                        + ratio
                        * (
                            float(current["source_sample_offset"])
                            - float(previous["source_sample_offset"])
                        ),
                        3,
                    ),
                    "source_t_s": round(
                        float(previous["source_t_s"])
                        + ratio
                        * (float(current["source_t_s"]) - float(previous["source_t_s"])),
                        3,
                    ),
                    "interpolated": step < steps,
                    "angles": [
                        round(
                            normalize_angle(float(previous["angles"][joint]) + ratio * deltas[joint]),
                            2,
                        )
                        for joint in range(6)
                    ],
                }
            )
    return dense


def build_segment(name, raw_samples, start_name, end_name, poses):
    start_angles = exact_angles(poses, start_name)
    end_angles = exact_angles(poses, end_name)
    warped = warp_to_exact_endpoints(raw_samples, start_angles, end_angles)
    waypoints = densify(downsample(warped))
    max_step = max(
        max(
            abs(shortest_angle_delta(a, b))
            for a, b in zip(current["angles"], previous["angles"])
        )
        for previous, current in zip(waypoints, waypoints[1:])
    )
    if max_step > MAX_COMMAND_STEP_DEG + 0.05:
        raise RuntimeError(f"{name} waypoint step too large: {max_step:.2f}deg")
    return {
        "name": name,
        "exact_start": start_name,
        "exact_end": end_name,
        "raw_sample_count": len(raw_samples),
        "recorded_start_angles": raw_samples[0]["angles"],
        "recorded_end_angles": raw_samples[-1]["angles"],
        "waypoint_delta_deg": WAYPOINT_DELTA_DEG,
        "max_command_step_deg": MAX_COMMAND_STEP_DEG,
        "waypoint_count": len(waypoints),
        "max_waypoint_step_deg": round(max_step, 3),
        "waypoints": waypoints,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--front", type=Path, required=True)
    parser.add_argument("--continuation", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    front = load_json(args.front)
    continuation = load_json(args.continuation)
    poses = load_json(args.poses)

    front_pick = mark_index(front, "PICK")
    front_dish = mark_index(front, "DISH_DROP")
    continuation_start = mark_index(continuation, "PICK")
    continuation_handle = mark_index(continuation, "DISH_DROP")

    segments = [
        build_segment(
            "PICK_TO_DISH_FRONT",
            front["samples"][front_pick : front_dish + 1],
            "PICK",
            "DISH_DROP",
            poses,
        ),
        build_segment(
            "DISH_TO_START_FRONT",
            continuation["samples"][: continuation_start + 1],
            "DISH_DROP",
            "START",
            poses,
        ),
        build_segment(
            "START_TO_LEFT_HANDLE_FRONT",
            continuation["samples"][continuation_start : continuation_handle + 1],
            "START",
            "LEFT_HANDLE",
            poses,
        ),
    ]

    payload = {
        "schema": "arm01_front_side_route.v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "operator_drag_teach_5hz",
        "motion_commands_sent_during_capture": False,
        "replay_executed": False,
        "endpoint_policy": "exact_20260717_poses_with_linear_capture_offset_correction",
        "joint_wrap_policy": "unwrap_capture_preserve_direction_then_normalize_commands",
        "source_files": {
            "front": {"path": str(args.front), "sha256": sha256(args.front)},
            "continuation": {
                "path": str(args.continuation),
                "sha256": sha256(args.continuation),
            },
            "poses": {"path": str(args.poses), "sha256": sha256(args.poses)},
        },
        "intended_flow": [
            "START_TO_PICK_EXISTING_PROVEN_DIRECT_PATH",
            "CLOSE_BAG_G23_PWM_10",
            "PICK_TO_DISH_FRONT",
            "OPEN_BAG_G23_PWM_17",
            "DISH_TO_START_FRONT",
            "START_TO_LEFT_HANDLE_FRONT",
            "SINGLE_PULSE_HANDLE_G23_PWM_10_0.35S_THEN_OFF",
        ],
        "segments": segments,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "segments": [
                    {
                        "name": segment["name"],
                        "raw_samples": segment["raw_sample_count"],
                        "waypoints": segment["waypoint_count"],
                        "max_step_deg": segment["max_waypoint_step_deg"],
                    }
                    for segment in segments
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
