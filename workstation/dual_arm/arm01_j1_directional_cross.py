#!/usr/bin/env python3
"""Cross arm01 J1 encoder zero with directional JOG instead of an absolute target."""

import argparse
import time

import bag_fixed_pick_g23 as baseline


PRE_CROSS_OTHER_ANGLES = [-115.80, 41.74, 79.84, 34.04, -44.53]
PREPARE_WAIT_S = 15.0
J1_DIRECTION = 1
J1_JOG_SPEED = 5
J1_PULSE_S = 1.0
J1_PULSE_SETTLE_S = 0.12
J1_POST_WRAP_TARGET = 3905
J1_WRAP_LOW = 128
J1_WRAP_HIGH = 3968
J1_CROSS_TIMEOUT_S = 25.0
J1_DEFAULT_MAX_DEG = 168
J1_TEMP_MAX_DEG = 200
ENCODER_MODULUS = 4096
KNOWN_UNSUPPORTED_REASON = (
    "disabled: arm01 firmware enforces the physical J1 range -168..168 even "
    "when the legacy limit readback incorrectly echoes 200; use the compact "
    "front transfer route instead"
)


def shortest_encoder_delta(current, previous):
    return (
        (int(current) - int(previous) + ENCODER_MODULUS // 2) % ENCODER_MODULUS
        - ENCODER_MODULUS // 2
    )


def prepare_high_pose(mc, speed):
    start = mc.get_angles()
    if not start or len(start) < 6:
        raise RuntimeError(f"failed to read starting angles: {start!r}")
    fixed_j1 = float(start[0])
    print(
        f"[prepare] hold J1={fixed_j1:.2f}; move J2-J6 to high crossing pose",
        flush=True,
    )
    target = [fixed_j1, *PRE_CROSS_OTHER_ANGLES]
    mc.send_angles(target, speed)
    time.sleep(PREPARE_WAIT_S)
    print(
        f"[prepare] angles={mc.get_angles()} encoders={mc.get_encoders()} "
        f"coords={mc.get_coords()}",
        flush=True,
    )


def set_j1_max(mc, degrees):
    mc.set_joint_max(1, int(degrees))
    time.sleep(0.25)
    readings = [mc.get_joint_max_angle(1) for _ in range(3)]
    print(f"[limit] set J1 max={degrees}; readback={readings}", flush=True)
    if readings.count(int(degrees)) < 2:
        raise RuntimeError(
            f"J1 max-angle readback mismatch: wanted={degrees}, got={readings}"
        )


def cross_j1_forward(mc):
    start = mc.get_encoder(1)
    if not isinstance(start, int):
        raise RuntimeError(f"failed to read J1 encoder: {start!r}")
    previous = start
    cumulative = 0
    crossed = False
    reached = False
    print(
        f"[cross] J1 encoder start={start}; direction={J1_DIRECTION} "
        f"speed={J1_JOG_SPEED} pulse={J1_PULSE_S:.2f}s",
        flush=True,
    )
    started = time.monotonic()
    while time.monotonic() - started < J1_CROSS_TIMEOUT_S:
        try:
            mc.jog_angle(1, J1_DIRECTION, J1_JOG_SPEED)
            time.sleep(J1_PULSE_S)
        finally:
            mc.jog_stop()
        time.sleep(J1_PULSE_SETTLE_S)
        current = mc.get_encoder(1)
        if not isinstance(current, int):
            continue
        cumulative += shortest_encoder_delta(current, previous)
        print(f"[cross] encoder {previous}->{current}", flush=True)
        if previous <= J1_WRAP_LOW and current >= J1_WRAP_HIGH:
            crossed = True
            print(
                f"[cross] encoder zero crossed: {previous}->{current}",
                flush=True,
            )
        previous = current
        if crossed and current <= J1_POST_WRAP_TARGET:
            reached = True
            break
    time.sleep(0.6)
    final = mc.get_encoder(1)
    print(
        f"[cross] stopped: crossed={crossed} reached={reached} "
        f"start={start} final={final} cumulative={cumulative} "
        f"angles={mc.get_angles()} coords={mc.get_coords()}",
        flush=True,
    )
    if not reached:
        raise RuntimeError("J1 directional crossing did not reach the post-wrap target")


def run(speed):
    raise RuntimeError(KNOWN_UNSUPPORTED_REASON)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=int, default=5)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.speed <= 10:
        raise SystemExit("speed must be between 1 and 10")
    if args.validate_only:
        print(f"[disabled] {KNOWN_UNSUPPORTED_REASON}")
        return
    run(args.speed)


if __name__ == "__main__":
    main()
