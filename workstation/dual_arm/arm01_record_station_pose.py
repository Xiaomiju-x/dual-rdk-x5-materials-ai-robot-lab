#!/usr/bin/env python3
"""Record final-station arm01 poses without commanding motion or the gripper."""

import argparse
import json

import bag_fixed_pick_g23 as baseline


ALLOWED_POSES = {
    "DISH_DROP": "drop_closed_powder_bag_into_grinding_dish",
    "LEFT_PRE_HANDLE": "approach_grinding_dish_handle",
    "LEFT_HANDLE": "grip_grinding_dish_left_handle",
    "LEFT_RETREAT": "retreat_from_grinding_dish_handle",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=sorted(ALLOWED_POSES))
    parser.add_argument(
        "--gripper-state",
        choices=[
            "full_close_pwm_10",
            "full_close_pwm_10_single_pulse_then_off",
            "open_pwm_17",
            "not_asserted",
        ],
        default="not_asserted",
    )
    args = parser.parse_args()

    mc = baseline.arm()
    pose = baseline.read_pose(mc, args.name)
    baseline.check_pose(pose)
    pose["purpose"] = ALLOWED_POSES[args.name]
    pose["operator_asserted_gripper_state"] = args.gripper_state
    pose["motion_commanded_during_record"] = False
    pose["gripper_commanded_during_record"] = False
    if args.name == "LEFT_HANDLE":
        pose["handle_grip_policy"] = {
            "gpio": 23,
            "frequency_hz": 50,
            "primary_close_pwm": 10,
            "primary_pulse_duration_s": 0.35,
            "stop_pwm_after_primary_pulse": True,
            "continuous_pwm_10_forbidden": True,
            "solid_edge_low_force_fallback_pwm": [12, 13],
            "fallback_requires_observed_solid_edge_contact": True,
        }

    poses = baseline.load_poses()
    poses[args.name] = pose
    baseline.save_poses(poses)
    print(json.dumps(pose, indent=2, ensure_ascii=False))
    print(f"[saved] {args.name} -> {baseline.POSE_FILE}")


if __name__ == "__main__":
    main()
