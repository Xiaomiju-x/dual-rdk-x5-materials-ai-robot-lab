#!/usr/bin/env python3
"""Run the commissioned arm02 direct grinding loop from RIGHT_START."""

from __future__ import annotations

import argparse
import json
import time

from pymycobot.mycobot280 import MyCobot280


PORT = "/dev/ttyAMA0"
BAUD = 1_000_000

RIGHT_START = [5.09, -111.53, -52.29, -49.57, 45.79, 157.76]
RIGHT_GRIND_WORK = [-78.22, -122.87, -125.94, -48.42, 37.88, 157.76]

TRANSPORT_SPEED = 5
GRIND_SPEED = 8
GRIND_LOW_J6 = 77.76
GRIND_HIGH_J6 = 157.76
DEFAULT_GRIND_CYCLES = 4
READ_TIMEOUT_S = 35.0
READ_RETRY_DELAY_S = 0.35
TELEMETRY_POLL_S = 0.5
COMMAND_SETTLE_S = 0.8


def emit(event: str, **payload: object) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=True), flush=True)


def read_angles(mc: MyCobot280, timeout_s: float = READ_TIMEOUT_S) -> list[float]:
    deadline = time.monotonic() + timeout_s
    attempts = 0
    while time.monotonic() < deadline:
        attempts += 1
        angles = mc.get_angles()
        if isinstance(angles, list) and len(angles) == 6:
            return angles
        time.sleep(READ_RETRY_DELAY_S)
    raise RuntimeError(
        f"arm02 did not return a valid six-joint angle sample in "
        f"{timeout_s:.1f}s ({attempts} attempts)"
    )


def read_power_on(mc: MyCobot280, timeout_s: float = 20.0) -> int:
    deadline = time.monotonic() + timeout_s
    last = -1
    while time.monotonic() < deadline:
        last = mc.is_power_on()
        if last in (0, 1):
            return last
        time.sleep(READ_RETRY_DELAY_S)
    return last


def wait_pose(
    mc: MyCobot280,
    target: list[float],
    timeout_s: float = 55.0,
    tolerance_deg: float = 2.0,
) -> list[float]:
    deadline = time.monotonic() + timeout_s
    while True:
        last = read_angles(mc)
        if max(abs(actual - wanted) for actual, wanted in zip(last, target)) <= tolerance_deg:
            return last
        if time.monotonic() >= deadline:
            break
        time.sleep(TELEMETRY_POLL_S)
    raise RuntimeError(f"pose timeout: target={target}, last={last}")


def wait_joint(
    mc: MyCobot280,
    joint_index: int,
    target_deg: float,
    timeout_s: float = 18.0,
    tolerance_deg: float = 2.0,
) -> list[float]:
    deadline = time.monotonic() + timeout_s
    while True:
        last = read_angles(mc)
        if abs(last[joint_index] - target_deg) <= tolerance_deg:
            return last
        if time.monotonic() >= deadline:
            break
        time.sleep(TELEMETRY_POLL_S)
    raise RuntimeError(f"joint timeout: J{joint_index + 1} target={target_deg}, last={last}")


def run(cycles: int) -> dict[str, object]:
    mc = MyCobot280(PORT, BAUD)
    initial = read_angles(mc)
    if read_power_on(mc) != 1:
        raise RuntimeError("arm02 servo power is off; no automatic power-on was attempted")

    emit(
        "FLOW_START",
        initial=initial,
        work_target=RIGHT_GRIND_WORK,
        cycles=cycles,
        transport_speed=TRANSPORT_SPEED,
        grind_speed=GRIND_SPEED,
    )

    mc.send_angles(RIGHT_GRIND_WORK, TRANSPORT_SPEED)
    time.sleep(COMMAND_SETTLE_S)
    work_actual = wait_pose(mc, RIGHT_GRIND_WORK)
    emit("WORK_REACHED", angles=work_actual)

    grind_steps: list[dict[str, object]] = []
    for cycle in range(1, cycles + 1):
        for phase, target in (("forward", GRIND_LOW_J6), ("return", GRIND_HIGH_J6)):
            mc.send_angle(6, target, GRIND_SPEED)
            time.sleep(COMMAND_SETTLE_S)
            actual = wait_joint(mc, 5, target)
            step = {
                "cycle": cycle,
                "phase": phase,
                "target_j6": target,
                "angles": actual,
            }
            grind_steps.append(step)
            emit("GRIND_STEP", **step)

    mc.send_angles(RIGHT_START, TRANSPORT_SPEED)
    time.sleep(COMMAND_SETTLE_S)
    final = wait_pose(mc, RIGHT_START)
    result = {
        "cycles": cycles,
        "work_actual": work_actual,
        "grind_steps": grind_steps,
        "final": final,
        "start_target": RIGHT_START,
        "is_power_on": read_power_on(mc),
        "servo_status": mc.get_servo_status(),
    }
    emit("CLOSED_LOOP_DONE", **result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=DEFAULT_GRIND_CYCLES)
    args = parser.parse_args()
    if not 1 <= args.cycles <= 20:
        parser.error("--cycles must be between 1 and 20")
    run(args.cycles)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
