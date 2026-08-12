#!/usr/bin/env python3
"""Stage-wise arm01 transfer flows for the fixed dual-arm workstation."""

from __future__ import annotations

import argparse
import json
import math
import time

from mycobot280_fk import forward_frames, forward_kinematics


PICK = [168.13, -140.88, 31.02, 91.14, 18.63, -59.15]
DISH_DROP = [-148.79, -124.01, 47.72, 73.03, 68.81, -55.98]
START = [142.55, -142.03, 31.72, 138.6, 104.41, -50.97]
HANDLE_APPROACH = [-167.5, -140.62, 29.53, 22.41, 112.5, -146.68]

# J2/J3 fold upward while J4-J6 preserve the taught PICK tool orientation.
COMPACT_PICK_BRANCH = [165.0, -90.0, 90.0, 90.0, 18.63, -59.15]
COMPACT_DISH_BRANCH = [-148.79, -90.0, 90.0, 90.0, 18.63, -59.15]
COMPACT_START_BRANCH = [142.55, -90.0, 90.0, 90.0, 18.63, -59.15]
COMPACT_HANDLE_BRANCH = [-167.5, -90.0, 90.0, 90.0, 18.63, -59.15]

TOOL_ENVELOPE_MM = 160.0
SAMPLES_PER_SEGMENT = 80
GRIP_CLOSE_PWM = 9
GRIP_OPEN_PWM = 17
BAG_PICK_CLOSE_HOLD_S = 1.50


def interpolate(start, end, samples=SAMPLES_PER_SEGMENT):
    for index in range(samples + 1):
        ratio = index / samples
        yield [before + ratio * (after - before) for before, after in zip(start, end)]


def tool_axis_endpoints(angles):
    frame = forward_frames(angles)[-1]
    flange = [frame[index][3] * 1000.0 for index in range(3)]
    axis = [frame[index][2] for index in range(3)]
    return [
        [flange[index] + sign * TOOL_ENVELOPE_MM * axis[index] for index in range(3)]
        for sign in (-1.0, 1.0)
    ]


def segment_metrics(start, end):
    max_link_radius = 0.0
    max_tool_radius = 0.0
    min_link_height = math.inf
    flange_path = []
    for angles in interpolate(start, end):
        result = forward_kinematics(angles)
        points = result["joint_points_mm"][2:]
        max_link_radius = max(
            max_link_radius,
            *(math.hypot(point[0], point[1]) for point in points),
        )
        min_link_height = min(min_link_height, *(point[2] for point in points))
        endpoints = tool_axis_endpoints(angles)
        max_tool_radius = max(
            max_tool_radius,
            *(math.hypot(point[0], point[1]) for point in endpoints),
        )
        flange_path.append(result["flange_xyz_mm"])
    return {
        "max_link_radius_mm": round(max_link_radius, 3),
        "max_tool_axis_radius_mm": round(max_tool_radius, 3),
        "min_link_height_mm": round(min_link_height, 3),
        "flange_start_mm": flange_path[0],
        "flange_end_mm": flange_path[-1],
    }


def validate_route():
    report = {
        "route": "PICK -> COMPACT_PICK -> COMPACT_DISH -> DISH_DROP",
        "rear_sweep_used": True,
        "segments": {
            "retract": segment_metrics(PICK, COMPACT_PICK_BRANCH),
            "compact_branch_switch": segment_metrics(
                COMPACT_PICK_BRANCH, COMPACT_DISH_BRANCH
            ),
            "front_unfold": segment_metrics(COMPACT_DISH_BRANCH, DISH_DROP),
            "return_retract": segment_metrics(DISH_DROP, COMPACT_DISH_BRANCH),
            "return_compact_switch": segment_metrics(
                COMPACT_DISH_BRANCH, COMPACT_START_BRANCH
            ),
            "return_unfold": segment_metrics(COMPACT_START_BRANCH, START),
            "handle_retract": segment_metrics(START, COMPACT_START_BRANCH),
            "handle_compact_switch": segment_metrics(
                COMPACT_START_BRANCH, COMPACT_HANDLE_BRANCH
            ),
            "dish_to_handle_compact_switch": segment_metrics(
                COMPACT_DISH_BRANCH, COMPACT_HANDLE_BRANCH
            ),
            "handle_unfold": segment_metrics(COMPACT_HANDLE_BRANCH, HANDLE_APPROACH),
        },
    }
    switch = report["segments"]["compact_branch_switch"]
    return_switch = report["segments"]["return_compact_switch"]
    handle_switch = report["segments"]["handle_compact_switch"]
    dish_to_handle_switch = report["segments"]["dish_to_handle_compact_switch"]
    checks = {
        "switch_link_radius_lt_129mm": switch["max_link_radius_mm"] < 129.0,
        "switch_tool_axis_radius_lt_110mm": switch["max_tool_axis_radius_mm"] < 110.0,
        "switch_link_height_gt_138mm": switch["min_link_height_mm"] > 138.0,
        "return_switch_link_radius_lt_129mm": (
            return_switch["max_link_radius_mm"] < 129.0
        ),
        "return_switch_tool_axis_radius_lt_110mm": (
            return_switch["max_tool_axis_radius_mm"] < 110.0
        ),
        "return_switch_link_height_gt_138mm": (
            return_switch["min_link_height_mm"] > 138.0
        ),
        "handle_switch_link_radius_lt_129mm": (
            handle_switch["max_link_radius_mm"] < 129.0
        ),
        "handle_switch_tool_axis_radius_lt_110mm": (
            handle_switch["max_tool_axis_radius_mm"] < 110.0
        ),
        "handle_switch_link_height_gt_138mm": (
            handle_switch["min_link_height_mm"] > 138.0
        ),
        "dish_to_handle_j1_delta_lt_20deg": (
            abs(COMPACT_DISH_BRANCH[0] - COMPACT_HANDLE_BRANCH[0]) < 20.0
        ),
        "dish_to_handle_switch_link_radius_lt_129mm": (
            dish_to_handle_switch["max_link_radius_mm"] < 129.0
        ),
        "dish_to_handle_switch_tool_axis_radius_lt_110mm": (
            dish_to_handle_switch["max_tool_axis_radius_mm"] < 110.0
        ),
        "dish_to_handle_switch_link_height_gt_138mm": (
            dish_to_handle_switch["min_link_height_mm"] > 138.0
        ),
    }
    report["checks"] = checks
    report["passed"] = all(checks.values())
    if not report["passed"]:
        raise RuntimeError(json.dumps(report, indent=2))
    return report


def connect_arm():
    import bag_fixed_pick_g23 as baseline

    mc = baseline.arm()
    baseline.power_on(mc)
    return mc


def hold_gripper_pwm(mc, baseline, pin_val, settle_s):
    print(
        f"[gripper] G{baseline.PIN} {baseline.FREQ_HZ}Hz "
        f"pin_val={pin_val} continuous_hold=true",
        flush=True,
    )
    mc.set_pwm_output(baseline.PIN, baseline.FREQ_HZ, int(pin_val))
    time.sleep(settle_s)


def release_gripper_pwm(mc, baseline):
    try:
        mc.set_pwm_output(baseline.PIN, baseline.FREQ_HZ, 0)
        mc.set_pin_mode(baseline.PIN, 0)
        print("[gripper] PWM released", flush=True)
    except Exception as exc:
        print(f"[gripper] PWM release warning: {exc}", flush=True)


def move_connected(mc, target, label, speed, timeout):
    before = {"angles": mc.get_angles(), "coords": mc.get_coords()}
    print(json.dumps({"stage": label, "before": before}, ensure_ascii=False), flush=True)
    started = time.monotonic()
    mc.sync_send_angles(target, speed, timeout=timeout)
    time.sleep(1.0)
    elapsed = time.monotonic() - started
    after = {"angles": mc.get_angles(), "coords": mc.get_coords()}
    print(
        json.dumps(
            {
                "stage": label,
                "elapsed_s": round(elapsed, 3),
                "target": target,
                "after": after,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return {
        "stage": label,
        "elapsed_s": round(elapsed, 3),
        "target": target,
        "before": before,
        "after": after,
    }


def run_stage(target, label, speed, timeout):
    mc = connect_arm()
    move_connected(mc, target, label, speed, timeout)


def run_bag_drop_return(speed, timeout):
    import bag_fixed_pick_g23 as baseline

    mc = connect_arm()
    timeline = []
    timeline.append(move_connected(mc, START, "ensure_start", speed, timeout))
    baseline.drive_gripper(mc, GRIP_CLOSE_PWM, hold_s=0.30)
    baseline.drive_gripper(mc, GRIP_OPEN_PWM, hold_s=0.45)
    timeline.append(move_connected(mc, PICK, "start_to_pick", speed, timeout))
    baseline.drive_gripper(mc, GRIP_CLOSE_PWM, hold_s=0.35)
    timeline.append(
        move_connected(mc, COMPACT_PICK_BRANCH, "pick_to_compact", speed, timeout)
    )
    timeline.append(
        move_connected(
            mc,
            COMPACT_DISH_BRANCH,
            "compact_switch_to_dish_branch",
            speed,
            timeout,
        )
    )
    append_slow_dish_approach(mc, timeline, timeout)
    baseline.drive_gripper(mc, GRIP_OPEN_PWM, hold_s=0.45)
    timeline.append(
        move_connected(mc, COMPACT_DISH_BRANCH, "dish_to_compact", speed, timeout)
    )
    timeline.append(
        move_connected(
            mc,
            COMPACT_START_BRANCH,
            "compact_switch_to_start_branch",
            speed,
            timeout,
        )
    )
    timeline.append(move_connected(mc, START, "unfold_to_start", speed, timeout))
    baseline.drive_gripper(mc, GRIP_CLOSE_PWM, hold_s=0.30)
    print(
        json.dumps(
            {
                "flow": "bag_drop_return",
                "result": "completed",
                "pose_gate_used": False,
                "continuous_gripper_pwm_used": False,
                "timeline": timeline,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def interpolate_angles(start, end, fraction):
    return [
        round(a + (b - a) * fraction, 3)
        for a, b in zip(start, end)
    ]


def append_slow_dish_approach(
    mc, timeline, timeout, approach_speed=3, approach_steps=6
):
    for index in range(1, approach_steps + 1):
        fraction = float(index) / float(approach_steps)
        target = interpolate_angles(COMPACT_DISH_BRANCH, DISH_DROP, fraction)
        timeline.append(
            move_connected(
                mc,
                target,
                "slow_dish_approach_{:02d}_of_{:02d}".format(index, approach_steps),
                approach_speed,
                timeout,
            )
        )


def append_slow_handle_approach(
    mc, timeline, timeout, approach_speed=3, approach_steps=6
):
    for index in range(1, approach_steps + 1):
        fraction = float(index) / float(approach_steps)
        target = interpolate_angles(COMPACT_HANDLE_BRANCH, HANDLE_APPROACH, fraction)
        timeline.append(
            move_connected(
                mc,
                target,
                "slow_handle_approach_{:02d}_of_{:02d}".format(
                    index, approach_steps
                ),
                approach_speed,
                timeout,
            )
    )


def run_bag_drop_top(speed, timeout):
    """Drop the bag, clear the shared area, and hold at the left-side top pose."""
    import bag_fixed_pick_g23 as baseline

    mc = connect_arm()
    timeline = []
    timeline.append(move_connected(mc, START, "ensure_start", speed, timeout))
    baseline.drive_gripper(mc, GRIP_CLOSE_PWM, hold_s=0.30)
    baseline.drive_gripper(mc, GRIP_OPEN_PWM, hold_s=0.45)
    timeline.append(move_connected(mc, PICK, "start_to_pick", speed, timeout))
    baseline.drive_gripper(mc, GRIP_CLOSE_PWM, hold_s=0.35)
    timeline.append(
        move_connected(mc, COMPACT_PICK_BRANCH, "pick_to_compact", speed, timeout)
    )
    timeline.append(
        move_connected(
            mc,
            COMPACT_DISH_BRANCH,
            "compact_switch_to_dish_branch",
            speed,
            timeout,
        )
    )
    append_slow_dish_approach(mc, timeline, timeout)
    baseline.drive_gripper(mc, GRIP_OPEN_PWM, hold_s=0.45)
    timeline.append(
        move_connected(mc, COMPACT_DISH_BRANCH, "dish_to_compact", speed, timeout)
    )
    timeline.append(
        move_connected(
            mc,
            COMPACT_START_BRANCH,
            "compact_switch_to_left_clear_top",
            speed,
            timeout,
        )
    )
    baseline.drive_gripper(mc, GRIP_CLOSE_PWM, hold_s=0.30)
    print(
        json.dumps(
            {
                "flow": "bag_drop_top",
                "result": "completed_left_clear_top",
                "left_clear_pose": "COMPACT_START_BRANCH",
                "returned_to_start": False,
                "bag_close_pwm": GRIP_CLOSE_PWM,
                "pose_gate_used": False,
                "continuous_gripper_pwm_used": False,
                "timeline": timeline,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def run_bag_drop_dish_top(speed, timeout):
    """Drop the bag and stop after the vertical retract above the dish."""
    import bag_fixed_pick_g23 as baseline

    mc = connect_arm()
    timeline = []
    timeline.append(move_connected(mc, START, "ensure_start", speed, timeout))
    baseline.drive_gripper(mc, GRIP_OPEN_PWM, hold_s=2.00)
    time.sleep(0.25)
    timeline.append(move_connected(mc, PICK, "start_to_pick", speed, timeout))
    hold_gripper_pwm(mc, baseline, GRIP_CLOSE_PWM, BAG_PICK_CLOSE_HOLD_S)
    gripper_holding = True
    try:
        timeline.append(
            move_connected(mc, COMPACT_PICK_BRANCH, "pick_to_compact", speed, timeout)
        )
        timeline.append(
            move_connected(
                mc,
                COMPACT_DISH_BRANCH,
                "compact_switch_to_dish_branch",
                speed,
                timeout,
            )
        )
        append_slow_dish_approach(mc, timeline, timeout)
        baseline.drive_gripper(mc, GRIP_OPEN_PWM, hold_s=3.00)
        gripper_holding = False
    finally:
        if gripper_holding:
            release_gripper_pwm(mc, baseline)
    timeline.append(
        move_connected(mc, COMPACT_DISH_BRANCH, "dish_to_clear_top", speed, timeout)
    )
    print(
        json.dumps(
            {
                "flow": "bag_drop_dish_top",
                "result": "completed_dish_clear_top",
                "dish_clear_pose": "COMPACT_DISH_BRANCH",
                "returned_to_start": False,
                "bag_close_pwm": GRIP_CLOSE_PWM,
                "bag_close_hold_s": BAG_PICK_CLOSE_HOLD_S,
                "bag_pwm_held_until_drop": True,
                "gripper_left_open": True,
                "timeline": timeline,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def run_dish_top_return_start(speed, timeout):
    """Continue from the dish-side top pose to START."""
    import bag_fixed_pick_g23 as baseline

    mc = connect_arm()
    timeline = []
    timeline.append(
        move_connected(
            mc,
            COMPACT_START_BRANCH,
            "dish_top_switch_to_start_top",
            speed,
            timeout,
        )
    )
    timeline.append(move_connected(mc, START, "start_top_to_start", speed, timeout))
    baseline.drive_gripper(mc, GRIP_CLOSE_PWM, hold_s=0.30)
    print(
        json.dumps(
            {
                "flow": "dish_top_return_start",
                "result": "completed_left_start",
                "returned_to_start": True,
                "gripper_close_pwm": GRIP_CLOSE_PWM,
                "timeline": timeline,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def run_start_to_dish_slow(speed, timeout, approach_speed=3, approach_steps=6):
    mc = connect_arm()
    timeline = []
    timeline.append(move_connected(mc, START, "ensure_start", speed, timeout))
    timeline.append(
        move_connected(mc, COMPACT_START_BRANCH, "start_to_compact", speed, timeout)
    )
    timeline.append(
        move_connected(
            mc,
            COMPACT_DISH_BRANCH,
            "compact_switch_to_dish_branch",
            speed,
            timeout,
        )
    )
    append_slow_dish_approach(
        mc,
        timeline,
        timeout,
        approach_speed=approach_speed,
        approach_steps=approach_steps,
    )
    print(
        json.dumps(
            {
                "flow": "start_to_dish_slow",
                "result": "completed_at_dish_with_gripper_unchanged",
                "pose_gate_used": False,
                "approach_speed": approach_speed,
                "approach_steps": approach_steps,
                "timeline": timeline,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def run_dish_to_handle(speed, timeout, approach_speed=3, approach_steps=6):
    import bag_fixed_pick_g23 as baseline

    mc = connect_arm()
    timeline = []
    baseline.drive_gripper(mc, GRIP_OPEN_PWM, hold_s=0.45)
    timeline.append(
        move_connected(mc, COMPACT_DISH_BRANCH, "dish_to_compact", speed, timeout)
    )
    timeline.append(
        move_connected(
            mc,
            COMPACT_HANDLE_BRANCH,
            "compact_short_switch_to_handle",
            speed,
            timeout,
        )
    )
    append_slow_handle_approach(
        mc,
        timeline,
        timeout,
        approach_speed=approach_speed,
        approach_steps=approach_steps,
    )
    baseline.drive_gripper(mc, GRIP_CLOSE_PWM, hold_s=0.35)
    print(
        json.dumps(
            {
                "flow": "dish_to_handle",
                "result": "completed_holding_handle",
                "returned_to_start": False,
                "continuous_gripper_pwm_used": False,
                "normal_segment_speed": speed,
                "handle_approach_speed": approach_speed,
                "handle_approach_steps": approach_steps,
                "timeline": timeline,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def run_start_dish_handle_test(speed, timeout, approach_speed=3, approach_steps=6):
    import bag_fixed_pick_g23 as baseline

    mc = connect_arm()
    timeline = []
    timeline.append(move_connected(mc, START, "ensure_start", speed, timeout))
    timeline.append(
        move_connected(mc, COMPACT_START_BRANCH, "start_to_compact", speed, timeout)
    )
    timeline.append(
        move_connected(
            mc,
            COMPACT_DISH_BRANCH,
            "compact_switch_to_dish_branch",
            speed,
            timeout,
        )
    )
    append_slow_dish_approach(mc, timeline, timeout)
    baseline.drive_gripper(mc, GRIP_OPEN_PWM, hold_s=0.45)
    timeline.append(
        move_connected(mc, COMPACT_DISH_BRANCH, "dish_to_compact", speed, timeout)
    )
    timeline.append(
        move_connected(
            mc,
            COMPACT_HANDLE_BRANCH,
            "compact_short_switch_to_handle",
            speed,
            timeout,
        )
    )
    append_slow_handle_approach(
        mc,
        timeline,
        timeout,
        approach_speed=approach_speed,
        approach_steps=approach_steps,
    )
    baseline.drive_gripper(mc, GRIP_CLOSE_PWM, hold_s=0.35)
    print(
        json.dumps(
            {
                "flow": "start_dish_handle_test",
                "result": "completed_holding_handle",
                "single_serial_session": True,
                "returned_to_start": False,
                "continuous_gripper_pwm_used": False,
                "normal_segment_speed": speed,
                "dish_approach_speed": 3,
                "handle_approach_speed": approach_speed,
                "timeline": timeline,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def run_bag_drop_handle(speed, timeout, approach_speed=3, approach_steps=6):
    import bag_fixed_pick_g23 as baseline

    mc = connect_arm()
    timeline = []
    timeline.append(move_connected(mc, START, "ensure_start", speed, timeout))
    baseline.drive_gripper(mc, GRIP_CLOSE_PWM, hold_s=0.30)
    baseline.drive_gripper(mc, GRIP_OPEN_PWM, hold_s=0.45)
    timeline.append(move_connected(mc, PICK, "start_to_pick", speed, timeout))
    baseline.drive_gripper(mc, GRIP_CLOSE_PWM, hold_s=0.35)
    timeline.append(
        move_connected(mc, COMPACT_PICK_BRANCH, "pick_to_compact", speed, timeout)
    )
    timeline.append(
        move_connected(
            mc,
            COMPACT_DISH_BRANCH,
            "compact_switch_to_dish_branch",
            speed,
            timeout,
        )
    )
    append_slow_dish_approach(mc, timeline, timeout)
    baseline.drive_gripper(mc, GRIP_OPEN_PWM, hold_s=0.45)
    timeline.append(
        move_connected(mc, COMPACT_DISH_BRANCH, "dish_to_compact", speed, timeout)
    )
    timeline.append(
        move_connected(
            mc,
            COMPACT_HANDLE_BRANCH,
            "compact_short_switch_to_handle",
            speed,
            timeout,
        )
    )
    append_slow_handle_approach(
        mc,
        timeline,
        timeout,
        approach_speed=approach_speed,
        approach_steps=approach_steps,
    )
    baseline.drive_gripper(mc, GRIP_CLOSE_PWM, hold_s=0.35)
    print(
        json.dumps(
            {
                "flow": "bag_drop_handle",
                "result": "completed_holding_handle",
                "returned_to_start": False,
                "pose_gate_used": False,
                "continuous_gripper_pwm_used": False,
                "normal_segment_speed": speed,
                "dish_approach_speed": 3,
                "dish_approach_steps": 6,
                "handle_approach_speed": approach_speed,
                "handle_approach_steps": approach_steps,
                "timeline": timeline,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def run_start_to_handle(speed, timeout):
    import bag_fixed_pick_g23 as baseline

    mc = connect_arm()
    timeline = []
    timeline.append(move_connected(mc, START, "ensure_start", speed, timeout))
    baseline.drive_gripper(mc, GRIP_OPEN_PWM, hold_s=0.45)
    timeline.append(
        move_connected(mc, COMPACT_START_BRANCH, "start_to_compact", speed, timeout)
    )
    timeline.append(
        move_connected(
            mc,
            COMPACT_HANDLE_BRANCH,
            "compact_switch_to_handle_branch",
            speed,
            timeout,
        )
    )
    timeline.append(
        move_connected(mc, HANDLE_APPROACH, "unfold_to_handle_approach", speed, timeout)
    )
    print(
        json.dumps(
            {
                "flow": "start_to_handle_approach",
                "result": "completed_open_gripper",
                "pose_gate_used": False,
                "timeline": timeline,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def grip_handle_once():
    import bag_fixed_pick_g23 as baseline

    mc = connect_arm()
    baseline.drive_gripper(mc, GRIP_CLOSE_PWM, hold_s=0.35)
    print(
        json.dumps(
            {
                "action": "grip_handle_once",
                "gpio": 23,
                "pwm": GRIP_CLOSE_PWM,
                "continuous_pwm_used": False,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "validate",
            "to-compact",
            "switch",
            "to-dish",
            "dish-to-compact",
            "switch-to-start",
            "to-start",
            "bag-drop-return",
            "bag-drop-top",
            "bag-drop-dish-top",
            "dish-top-return-start",
            "bag-drop-handle",
            "start-to-dish-slow",
            "start-dish-handle-test",
            "dish-to-handle",
            "start-to-handle",
            "grip-handle-once",
        ),
    )
    parser.add_argument("--speed", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    if not 1 <= args.speed <= 30:
        raise SystemExit("speed must be in 1..30")
    if args.command == "validate":
        print(json.dumps(validate_route(), indent=2))
        return
    if args.command == "bag-drop-return":
        run_bag_drop_return(args.speed, args.timeout)
        return
    if args.command == "bag-drop-top":
        run_bag_drop_top(args.speed, args.timeout)
        return
    if args.command == "bag-drop-dish-top":
        run_bag_drop_dish_top(args.speed, args.timeout)
        return
    if args.command == "dish-top-return-start":
        run_dish_top_return_start(args.speed, args.timeout)
        return
    if args.command == "bag-drop-handle":
        run_bag_drop_handle(args.speed, args.timeout)
        return
    if args.command == "start-to-dish-slow":
        run_start_to_dish_slow(args.speed, args.timeout)
        return
    if args.command == "start-dish-handle-test":
        run_start_dish_handle_test(args.speed, args.timeout)
        return
    if args.command == "dish-to-handle":
        run_dish_to_handle(args.speed, args.timeout)
        return
    if args.command == "start-to-handle":
        run_start_to_handle(args.speed, args.timeout)
        return
    if args.command == "grip-handle-once":
        grip_handle_once()
        return
    if args.command == "to-compact":
        run_stage(COMPACT_PICK_BRANCH, "to_compact_pick_branch", args.speed, args.timeout)
    elif args.command == "switch":
        run_stage(COMPACT_DISH_BRANCH, "compact_branch_switch", args.speed, args.timeout)
    elif args.command == "to-dish":
        run_stage(DISH_DROP, "front_unfold_to_dish", args.speed, args.timeout)
    elif args.command == "dish-to-compact":
        run_stage(COMPACT_DISH_BRANCH, "dish_retract_to_compact", args.speed, args.timeout)
    elif args.command == "switch-to-start":
        run_stage(COMPACT_START_BRANCH, "compact_switch_to_start_branch", args.speed, args.timeout)
    else:
        run_stage(START, "unfold_to_start", args.speed, args.timeout)


if __name__ == "__main__":
    main()
