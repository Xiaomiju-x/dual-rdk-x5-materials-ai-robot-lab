#!/usr/bin/env python3
"""Replay an arm01 native encoder drag segment without angle conversion."""

import argparse
import json
import time
from pathlib import Path

import bag_fixed_pick_g23 as baseline


EXPECTED_SCHEMA = "arm01_native_encoder_drag.v1"
FIRST_POINT_SETTLE_S = 3.0


def load_and_validate(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != EXPECTED_SCHEMA:
        raise RuntimeError(f"unexpected route schema: {payload.get('schema')!r}")
    if payload.get("replay_api") != "set_encoders_drag":
        raise RuntimeError("route was not captured for native encoder drag replay")
    if payload.get("angle_waypoint_conversion") is not False:
        raise RuntimeError("angle-converted routes are not accepted")
    if payload.get("endpoint_warping") is not False:
        raise RuntimeError("endpoint-warped routes are not accepted")
    if payload.get("servo_speeds_source") != "derived_shortest_mod4096_delta_per_second":
        raise RuntimeError("route does not preserve signed encoder-wrap direction")
    samples = payload.get("samples") or []
    if len(samples) < 3:
        raise RuntimeError(f"insufficient route samples: {len(samples)}")
    for index, sample in enumerate(samples):
        encoders = sample.get("encoders") or []
        speeds = sample.get("servo_speeds") or []
        if len(encoders) != 6 or len(speeds) != 6:
            raise RuntimeError(f"sample {index} is not a six-joint encoder frame")
        if any(not 0 <= int(value) <= 4096 for value in encoders):
            raise RuntimeError(f"sample {index} encoder outside 0..4096")
        if any(not -32768 <= int(value) <= 32767 for value in speeds):
            raise RuntimeError(f"sample {index} speed outside signed int16")
        if index and not 0.0 <= float(sample.get("interval_s", -1.0)) <= 2.0:
            raise RuntimeError(f"sample {index} has invalid interval")
    return payload


def replay(path):
    payload = load_and_validate(path)
    samples = payload["samples"]
    mc = baseline.arm()
    baseline.power_on(mc)
    print(
        f"[replay] {payload['label']}: native frames={len(samples)} "
        "angle_conversion=false endpoint_warping=false",
        flush=True,
    )
    first = samples[0]
    mc.set_encoders_drag(first["encoders"], first["servo_speeds"])
    time.sleep(FIRST_POINT_SETTLE_S)
    for sample in samples[1:]:
        mc.set_encoders_drag(sample["encoders"], sample["servo_speeds"])
        time.sleep(float(sample["interval_s"]))
    print("[replay] native encoder segment done", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    route = args.route.expanduser()
    payload = load_and_validate(route)
    if args.validate_only:
        print(
            f"[validate] {payload['label']}: samples={len(payload['samples'])} "
            "replay=set_encoders_drag"
        )
        return
    replay(route)


if __name__ == "__main__":
    main()
