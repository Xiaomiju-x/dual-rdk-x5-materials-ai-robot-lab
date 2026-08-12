#!/usr/bin/env python3
"""Recalibrated arm01 powder-bag flow for the final dual-arm station."""

import argparse
import time

import bag_fixed_pick_g23 as baseline


OPEN_PWM = 17
FULL_CLOSE_PWM = 10
OPEN_HOLD_S = 2.0
CLOSE_SETTLE_S = 0.35
ARRIVAL_ANGLE_ERROR_DEG = 5.0
ARRIVAL_XYZ_ERROR_MM = 15.0
ARRIVAL_STABLE_SAMPLES = 2

EXPECTED_POSES = {
    "START": {
        "angles": [142.55, -142.03, 31.72, 138.6, 104.41, -50.97],
        "coords": [-77.5, 83.5, 71.5, -120.84, -76.69, -174.53],
    },
    "PICK": {
        "angles": [168.13, -140.88, 31.02, 91.14, 18.63, -59.15],
        "coords": [-210.0, 94.0, 70.1, -117.92, -49.5, 119.68],
    },
    "DISH_DROP": {
        "angles": [-148.79, -124.01, 47.72, 73.03, 68.81, -55.98],
        "coords": [-186.5, -88.5, 166.2, -91.94, -52.93, -168.39],
    },
    "LEFT_HANDLE": {
        "angles": [-170.33, -140.62, 29.53, 22.41, 112.5, -146.68],
        "coords": [-234.4, -18.3, 30.5, -51.9, -51.67, 158.62],
    },
}


def assert_recalibrated_poses(poses, required_pose_names):
    for name in required_pose_names:
        expected = EXPECTED_POSES[name]
        actual = poses.get(name)
        if not actual:
            raise RuntimeError(f"missing recalibrated pose: {name}")
        angle_error = max(
            abs(float(a) - float(b))
            for a, b in zip(actual.get("angles", []), expected["angles"])
        )
        coord_error = max(
            abs(float(a) - float(b))
            for a, b in zip(actual.get("coords", []), expected["coords"])
        )
        if angle_error > 0.05 or coord_error > 0.2:
            raise RuntimeError(
                f"{name} does not match the 2026-07-17 calibration: "
                f"angle_error={angle_error:.2f}, coord_error={coord_error:.2f}"
            )


def send_pose_with_arrival_gate(mc, pose, speed, label, timeout_s):
    baseline.check_pose(pose)
    target_angles = [float(value) for value in pose["angles"][:6]]
    mc.send_angles(target_angles, int(speed))
    started = time.monotonic()
    stable_samples = 0
    last_angle_error = None
    last_xyz_error = None
    last_coords = None

    while time.monotonic() - started < timeout_s:
        current_angles = mc.get_angles()
        current_coords = mc.get_coords()
        if current_angles and len(current_angles) >= 6:
            last_angle_error = max(
                abs(float(actual) - target)
                for actual, target in zip(current_angles[:6], target_angles)
            )
        else:
            last_angle_error = None
        last_coords = current_coords
        last_xyz_error = baseline.xyz_distance(current_coords, pose.get("coords"))

        if (
            last_angle_error is not None
            and last_xyz_error is not None
            and last_angle_error <= ARRIVAL_ANGLE_ERROR_DEG
            and last_xyz_error <= ARRIVAL_XYZ_ERROR_MM
        ):
            stable_samples += 1
            if stable_samples >= ARRIVAL_STABLE_SAMPLES:
                elapsed = time.monotonic() - started
                print(
                    f"[arrival] {label}: elapsed={elapsed:.2f}s "
                    f"angle_error={last_angle_error:.2f}deg "
                    f"xyz_error={last_xyz_error:.1f}mm"
                )
                time.sleep(0.2)
                return
        else:
            stable_samples = 0
        time.sleep(0.15)

    angle_text = "n/a" if last_angle_error is None else f"{last_angle_error:.2f}deg"
    xyz_text = "n/a" if last_xyz_error is None else f"{last_xyz_error:.1f}mm"
    raise RuntimeError(
        f"{label} arrival gate timed out: angle_error={angle_text}, "
        f"xyz_error={xyz_text}, coords={last_coords}"
    )


def run_redundancy(speed):
    poses = baseline.load_poses()
    assert_recalibrated_poses(poses, ("START", "PICK"))

    mc = baseline.arm()
    baseline.power_on(mc)
    holding_full_close = False
    close_started = None

    print(
        "[run] START(close=10) -> open=17/2s -> PICK -> "
        "HOLD close=10 -> START -> open/drop -> close=10"
    )
    try:
        send_pose_with_arrival_gate(
            mc, poses["START"], speed, "START", baseline.MOVE_TIMEOUT_S
        )
        baseline.drive_gripper(mc, FULL_CLOSE_PWM, hold_s=CLOSE_SETTLE_S)
        baseline.drive_gripper(mc, OPEN_PWM, hold_s=OPEN_HOLD_S)
        send_pose_with_arrival_gate(
            mc,
            poses["PICK"],
            speed,
            "PICK",
            baseline.PICK_TIMEOUT_S,
        )
        time.sleep(0.35)
        baseline.hold_gripper(mc, FULL_CLOSE_PWM)
        holding_full_close = True
        close_started = time.monotonic()
        time.sleep(CLOSE_SETTLE_S)
        send_pose_with_arrival_gate(
            mc,
            poses["START"],
            speed,
            "START",
            baseline.RETURN_TIMEOUT_S,
        )
        hold_duration = time.monotonic() - close_started
        baseline.drive_gripper(mc, OPEN_PWM, hold_s=OPEN_HOLD_S)
        holding_full_close = False
        time.sleep(0.2)
        baseline.drive_gripper(mc, FULL_CLOSE_PWM, hold_s=CLOSE_SETTLE_S)
        print(f"[safety] full-close hold duration={hold_duration:.2f}s")
    except Exception:
        if holding_full_close:
            print("[safety] opening gripper before abort")
            try:
                baseline.drive_gripper(mc, OPEN_PWM, hold_s=OPEN_HOLD_S)
            except Exception as exc:
                print(f"[safety] failed to open gripper: {exc}")
        raise
    print("[run] done")


def run_bag_to_dish_handle(speed):
    poses = baseline.load_poses()
    assert_recalibrated_poses(
        poses, ("START", "PICK", "DISH_DROP", "LEFT_HANDLE")
    )

    mc = baseline.arm()
    baseline.power_on(mc)
    holding_bag = False
    close_started = None

    print(
        "[run] operator-positioned START -> open -> PICK -> "
        "HOLD bag -> DISH_DROP -> release bag -> START -> LEFT_HANDLE -> "
        "single-pulse close"
    )
    try:
        # This mode has a manual boundary: the operator places the arm at START.
        send_pose_with_arrival_gate(
            mc,
            poses["START"],
            speed,
            "START(operator-positioned)",
            baseline.RETURN_TIMEOUT_S,
        )

        # Preserve the demonstrated default-closed START state, then open for pickup.
        baseline.drive_gripper(mc, FULL_CLOSE_PWM, hold_s=CLOSE_SETTLE_S)
        baseline.drive_gripper(mc, OPEN_PWM, hold_s=OPEN_HOLD_S)
        send_pose_with_arrival_gate(
            mc, poses["PICK"], speed, "PICK", baseline.PICK_TIMEOUT_S
        )
        time.sleep(0.35)

        # The thin bag needs active close only during the short transfer.
        baseline.hold_gripper(mc, FULL_CLOSE_PWM)
        holding_bag = True
        close_started = time.monotonic()
        time.sleep(CLOSE_SETTLE_S)
        send_pose_with_arrival_gate(
            mc, poses["DISH_DROP"], speed, "DISH_DROP", baseline.MOVE_TIMEOUT_S
        )
        bag_hold_duration = time.monotonic() - close_started
        baseline.drive_gripper(mc, OPEN_PWM, hold_s=OPEN_HOLD_S)
        holding_bag = False
        print(f"[safety] bag full-close hold duration={bag_hold_duration:.2f}s")

        # Return through START to avoid the dish collision path.
        send_pose_with_arrival_gate(
            mc, poses["START"], speed, "START(clear dish)", baseline.RETURN_TIMEOUT_S
        )
        send_pose_with_arrival_gate(
            mc,
            poses["LEFT_HANDLE"],
            speed,
            "LEFT_HANDLE",
            baseline.MOVE_TIMEOUT_S,
        )

        # One short close pulse only. PWM is released immediately to limit heating.
        baseline.drive_gripper(mc, FULL_CLOSE_PWM, hold_s=CLOSE_SETTLE_S)
    except Exception:
        if holding_bag:
            print("[safety] opening gripper before abort")
            try:
                baseline.drive_gripper(mc, OPEN_PWM, hold_s=OPEN_HOLD_S)
            except Exception as exc:
                print(f"[safety] failed to open gripper: {exc}")
        raise
    print("[run] bag-to-dish-handle done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=int, default=10)
    parser.add_argument(
        "--mode",
        choices=["redundancy", "bag-to-dish-handle"],
        default="redundancy",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not 5 <= args.speed <= 10:
        raise SystemExit("speed must be between 5 and 10")
    if args.validate_only:
        required = (
            ("START", "PICK")
            if args.mode == "redundancy"
            else ("START", "PICK", "DISH_DROP", "LEFT_HANDLE")
        )
        assert_recalibrated_poses(baseline.load_poses(), required)
        print(f"[validate] {args.mode}: {', '.join(required)} OK")
        return
    if args.mode == "redundancy":
        run_redundancy(args.speed)
    else:
        run_bag_to_dish_handle(args.speed)


if __name__ == "__main__":
    main()
