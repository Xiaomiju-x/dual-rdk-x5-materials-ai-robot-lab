#!/usr/bin/env python3
"""Read-only arm01 top/clear-space verification for the dual-arm barrier."""

from __future__ import annotations

import argparse
import json
import sys
import time

sys.path.insert(0, "/home/rdk")

import bag_fixed_pick_g23 as baseline


LEFT_CLEAR_TOP = [142.55, -90.0, 90.0, 90.0, 18.63, -59.15]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerance", type=float, default=6.0)
    args = parser.parse_args()

    mc = baseline.arm()
    samples = []
    for _ in range(3):
        angles = mc.get_angles()
        if not isinstance(angles, list) or len(angles) != 6:
            print(json.dumps({"event": "LEFT_CLEAR", "clear": False, "reason": "invalid_angle_read"}))
            return 2
        samples.append(angles)
        time.sleep(0.3)

    errors = [
        max(abs(actual - target) for actual, target in zip(sample, LEFT_CLEAR_TOP))
        for sample in samples
    ]
    spread = max(
        max(sample[joint] for sample in samples)
        - min(sample[joint] for sample in samples)
        for joint in range(6)
    )
    clear = max(errors) <= args.tolerance and spread <= 1.5
    result = {
        "event": "LEFT_CLEAR_TOP",
        "clear": clear,
        "pose": "COMPACT_START_BRANCH",
        "target": LEFT_CLEAR_TOP,
        "samples": samples,
        "max_error_deg": round(max(errors), 2),
        "max_spread_deg": round(spread, 2),
        "tolerance_deg": args.tolerance,
        "motion_command_sent": False,
    }
    print(json.dumps(result), flush=True)
    return 0 if clear else 3


if __name__ == "__main__":
    raise SystemExit(main())
