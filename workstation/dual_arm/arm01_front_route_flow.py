#!/usr/bin/env python3
"""Run the final arm01 bag-to-dish-to-handle flow on a drag-taught front route."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import arm01_bag_recalibrated_20260717 as calibrated
import bag_fixed_pick_g23 as baseline


ROUTE_FILE = Path.home() / "route_teach" / "arm01_front_side_route_20260717.json"
EXPECTED_ROUTE_SHA256 = "44311a4bbda1694105c1609472d35685634d810abeb3944f4741f34d50216d13"
EXPECTED_SEGMENTS = [
    "PICK_TO_DISH_FRONT",
    "DISH_TO_START_FRONT",
    "START_TO_LEFT_HANDLE_FRONT",
]
KNOWN_UNSAFE_REASON = (
    "disabled after two physical collisions: the angle route crosses the J1 "
    "+/-180 degree boundary and arm01 does not replay that boundary in the "
    "drag-taught direction"
)
STREAM_PERIOD_S = 0.22
WRAP_INCREMENT_SETTLE_S = 2.5
SEGMENT_FINAL_SETTLE_S = 2.0
START_SETTLE_S = 1.2
PICK_SETTLE_S = 5.5


def sha256(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def shortest_angle_delta(current, previous):
    return (float(current) - float(previous) + 180.0) % 360.0 - 180.0


def load_and_validate_route():
    if sha256(ROUTE_FILE) != EXPECTED_ROUTE_SHA256:
        raise RuntimeError("front route hash does not match the commissioned candidate")
    route = json.loads(ROUTE_FILE.read_text(encoding="utf-8"))
    if route.get("schema") != "arm01_front_side_route.v1":
        raise RuntimeError("unexpected front route schema")
    segments = route.get("segments") or []
    if [segment.get("name") for segment in segments] != EXPECTED_SEGMENTS:
        raise RuntimeError("front route segment order mismatch")
    for segment in segments:
        if float(segment.get("max_waypoint_step_deg", 999.0)) > 5.05:
            raise RuntimeError(f"unsafe waypoint step in {segment['name']}")
        if len(segment.get("waypoints") or []) < 3:
            raise RuntimeError(f"insufficient waypoints in {segment['name']}")
    return route


def send_pose_fixed(mc, pose, speed, label, settle_s):
    baseline.check_pose(pose)
    target = [float(value) for value in pose["angles"][:6]]
    print(f"[move-fixed] {label}: settle={settle_s:.2f}s")
    mc.send_angles(target, speed)
    time.sleep(settle_s)


def replay_segment(mc, segment, speed, stop_after_index=None):
    waypoints = segment["waypoints"]
    last_index = len(waypoints) - 1
    if stop_after_index is not None:
        last_index = min(int(stop_after_index), last_index)
    print(f"[route] {segment['name']}: waypoints={len(waypoints)} speed={speed}")
    previous = [float(value) for value in waypoints[0]["angles"][:6]]
    for index, waypoint in enumerate(waypoints[1 : last_index + 1], start=1):
        target = [float(value) for value in waypoint["angles"][:6]]
        wrapped_joints = [
            joint
            for joint, (before, after) in enumerate(zip(previous, target), start=1)
            if abs(after - before) > 180.0
        ]
        if wrapped_joints:
            for joint, after in enumerate(target, start=1):
                if joint not in wrapped_joints:
                    mc.send_angle(joint, after, speed)
            for joint in wrapped_joints:
                before = previous[joint - 1]
                after = target[joint - 1]
                increment = shortest_angle_delta(after, before)
                print(
                    f"[wrap] {segment['name']} index={index} J{joint} "
                    f"{before:.2f}->{after:.2f} increment={increment:.2f}"
                )
                mc.jog_increment(joint, increment, speed)
            time.sleep(WRAP_INCREMENT_SETTLE_S)
        else:
            mc.send_angles(target, speed)
            time.sleep(STREAM_PERIOD_S)
        previous = target
    final_target = [float(value) for value in waypoints[last_index]["angles"][:6]]
    mc.send_angles(final_target, speed)
    time.sleep(SEGMENT_FINAL_SETTLE_S)


def run(speed):
    route = load_and_validate_route()
    poses = baseline.load_poses()
    calibrated.assert_recalibrated_poses(
        poses, ("START", "PICK", "DISH_DROP", "LEFT_HANDLE")
    )
    segments = {segment["name"]: segment for segment in route["segments"]}

    mc = baseline.arm()
    baseline.power_on(mc)

    print(
        "[run] START -> PICK -> front route DISH_DROP -> front route START -> "
        "front route LEFT_HANDLE"
    )
    send_pose_fixed(mc, poses["START"], speed, "START", START_SETTLE_S)
    baseline.drive_gripper(mc, calibrated.FULL_CLOSE_PWM, calibrated.CLOSE_SETTLE_S)
    baseline.drive_gripper(mc, calibrated.OPEN_PWM, calibrated.OPEN_HOLD_S)
    send_pose_fixed(mc, poses["PICK"], speed, "PICK", PICK_SETTLE_S)
    time.sleep(0.35)

    baseline.hold_gripper(mc, calibrated.FULL_CLOSE_PWM)
    bag_close_started = time.monotonic()
    time.sleep(calibrated.CLOSE_SETTLE_S)
    replay_segment(
        mc,
        segments["PICK_TO_DISH_FRONT"],
        speed,
    )
    bag_hold_duration = time.monotonic() - bag_close_started
    baseline.drive_gripper(mc, calibrated.OPEN_PWM, calibrated.OPEN_HOLD_S)
    print(f"[run] bag full-close hold duration={bag_hold_duration:.2f}s")

    replay_segment(
        mc,
        segments["DISH_TO_START_FRONT"],
        speed,
    )
    replay_segment(
        mc,
        segments["START_TO_LEFT_HANDLE_FRONT"],
        speed,
    )
    baseline.drive_gripper(
        mc, calibrated.FULL_CLOSE_PWM, calibrated.CLOSE_SETTLE_S
    )
    print("[run] front-route flow done")


def test_first_wrap(speed):
    route = load_and_validate_route()
    poses = baseline.load_poses()
    calibrated.assert_recalibrated_poses(poses, ("START", "PICK"))
    segment = next(
        item for item in route["segments"] if item["name"] == "PICK_TO_DISH_FRONT"
    )

    mc = baseline.arm()
    baseline.power_on(mc)
    print("[test] START -> PICK -> first wrap -> stop at waypoint 28")
    send_pose_fixed(mc, poses["START"], speed, "START", START_SETTLE_S)
    baseline.drive_gripper(mc, calibrated.OPEN_PWM, calibrated.OPEN_HOLD_S)
    send_pose_fixed(mc, poses["PICK"], speed, "PICK", PICK_SETTLE_S)
    replay_segment(mc, segment, speed, stop_after_index=28)
    print("[test] stopped after first wrap; no bag close and no remaining flow")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=int, default=10)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--test-first-wrap", action="store_true")
    args = parser.parse_args()
    if not 5 <= args.speed <= 10:
        raise SystemExit("speed must be between 5 and 10")
    route = load_and_validate_route()
    if args.validate_only:
        print(
            "[validate] "
            + ", ".join(
                f"{segment['name']}={segment['waypoint_count']}"
                for segment in route["segments"]
            )
        )
        return
    raise SystemExit(f"known-unsafe route: {KNOWN_UNSAFE_REASON}")


if __name__ == "__main__":
    main()
