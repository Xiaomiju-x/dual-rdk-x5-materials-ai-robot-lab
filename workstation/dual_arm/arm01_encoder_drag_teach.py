#!/usr/bin/env python3
"""Record an arm01 drag-taught segment in the controller's native encoder format."""

import argparse
import json
import signal
import time
from pathlib import Path

import bag_fixed_pick_g23 as baseline


SAMPLE_HZ = 10.0
MAX_RECORD_S = 180.0
ENCODER_MODULUS = 4096

stop_requested = False


def request_stop(_signum, _frame):
    global stop_requested
    stop_requested = True


def read_vector(reader, name):
    value = reader()
    if not isinstance(value, (list, tuple)) or len(value) < 6:
        raise RuntimeError(f"failed to read {name}: {value!r}")
    return value[:6]


def read_sample(mc, started, previous_t):
    encoders = mc.get_encoders()
    if not isinstance(encoders, (list, tuple)) or len(encoders) < 6:
        return None
    now = time.monotonic()
    elapsed = now - started
    return {
        "t_s": round(elapsed, 4),
        "interval_s": round(0.0 if previous_t is None else elapsed - previous_t, 4),
        "encoders": [int(value) for value in encoders],
    }


def validate_sample(sample):
    if len(sample["encoders"]) != 6:
        raise RuntimeError("encoder drag sample must contain six encoders")
    if any(not 0 <= value <= 4096 for value in sample["encoders"]):
        raise RuntimeError(f"encoder outside 0..4096: {sample['encoders']}")


def shortest_encoder_delta(current, previous):
    return (
        (int(current) - int(previous) + ENCODER_MODULUS // 2) % ENCODER_MODULUS
        - ENCODER_MODULUS // 2
    )


def add_derived_speeds(samples):
    for index, sample in enumerate(samples):
        if index == 0:
            sample["servo_speeds"] = [0] * 6
            continue
        interval_s = float(sample["interval_s"])
        if interval_s <= 0.0:
            raise RuntimeError(f"invalid sample interval at {index}: {interval_s}")
        previous = samples[index - 1]["encoders"]
        sample["servo_speeds"] = [
            int(round(shortest_encoder_delta(current, before) / interval_s))
            for current, before in zip(sample["encoders"], previous)
        ]
    if len(samples) > 1:
        samples[0]["servo_speeds"] = list(samples[1]["servo_speeds"])


def record(label, output):
    global stop_requested
    stop_requested = False
    output.parent.mkdir(parents=True, exist_ok=True)
    mc = baseline.arm()
    initial_pose = baseline.read_pose(mc, f"{label}_INITIAL")
    initial_encoders = read_vector(mc.get_encoders, "initial encoders")
    samples = []
    empty_reads = 0
    release_succeeded = False
    locked_pose = None

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    started = time.monotonic()
    previous_t = None
    period_s = 1.0 / SAMPLE_HZ
    try:
        mc.release_all_servos()
        release_succeeded = True
        print(
            f"[teach] {label}: servos released; support and drag the arm, "
            "then send Ctrl+C at the endpoint",
            flush=True,
        )
        while not stop_requested and time.monotonic() - started < MAX_RECORD_S:
            tick = time.monotonic()
            sample = read_sample(mc, started, previous_t)
            if sample is None:
                empty_reads += 1
                time.sleep(max(0.0, period_s - (time.monotonic() - tick)))
                continue
            validate_sample(sample)
            samples.append(sample)
            previous_t = sample["t_s"]
            time.sleep(max(0.0, period_s - (time.monotonic() - tick)))
    finally:
        if release_succeeded:
            mc.power_on()
            time.sleep(0.8)
            locked_pose = baseline.read_pose(mc, f"{label}_LOCKED_END")
            print("[teach] servos locked at the endpoint", flush=True)

        add_derived_speeds(samples)
        payload = {
            "schema": "arm01_native_encoder_drag.v1",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "label": label,
            "sample_hz_target": SAMPLE_HZ,
            "capture_api": ["get_encoders"],
            "servo_speeds_source": "derived_shortest_mod4096_delta_per_second",
            "encoder_modulus": ENCODER_MODULUS,
            "replay_api": "set_encoders_drag",
            "angle_waypoint_conversion": False,
            "endpoint_warping": False,
            "motion_commands_sent_during_capture": False,
            "release_all_servos_sent": release_succeeded,
            "power_on_sent_at_end": release_succeeded,
            "initial_pose": initial_pose,
            "initial_encoders": [int(value) for value in initial_encoders],
            "locked_final_pose": locked_pose,
            "sample_count": len(samples),
            "empty_encoder_reads_skipped": empty_reads,
            "samples": samples,
        }
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[saved] {output} samples={len(samples)}", flush=True)

    if len(samples) < 3:
        raise RuntimeError(f"insufficient encoder samples: {len(samples)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        print(
            f"[validate] native encoder drag capture: sample_hz={SAMPLE_HZ} "
            f"max_s={MAX_RECORD_S}"
        )
        return
    record(args.label, args.output.expanduser())


if __name__ == "__main__":
    main()
