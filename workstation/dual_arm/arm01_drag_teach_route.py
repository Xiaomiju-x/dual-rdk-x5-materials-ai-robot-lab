#!/usr/bin/env python3
"""Record an arm01 drag-taught route without commanding joint motion."""

import argparse
import json
import signal
import time
from pathlib import Path

import bag_fixed_pick_g23 as baseline


SAMPLE_HZ = 5.0
MAX_RECORD_S = 240.0
MARK_SEQUENCE = ["PICK", "DISH_DROP", "START_RETURN", "LEFT_HANDLE"]

stop_requested = False
mark_requests = 0


def request_stop(_signum, _frame):
    global stop_requested
    stop_requested = True


def request_mark(_signum, _frame):
    global mark_requests
    mark_requests += 1


def read_sample(mc, started):
    angles = mc.get_angles()
    coords = mc.get_coords()
    if not angles or len(angles) < 6 or not coords or len(coords) < 6:
        return None
    return {
        "t_s": round(time.monotonic() - started, 3),
        "angles": [round(float(value), 2) for value in angles[:6]],
        "coords": [round(float(value), 2) for value in coords[:6]],
    }


def lock_servos(mc):
    mc.power_on()
    time.sleep(0.8)


def record(output):
    global mark_requests

    output.parent.mkdir(parents=True, exist_ok=True)
    mc = baseline.arm()
    initial = baseline.read_pose(mc, "START_OPERATOR_POSITIONED")
    baseline.check_pose(initial)

    samples = []
    marks = []
    started = time.monotonic()
    period_s = 1.0 / SAMPLE_HZ
    locked_pose = None
    release_succeeded = False

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGUSR1, request_mark)

    try:
        mc.release_all_servos()
        release_succeeded = True
        print("[teach] servos released; support the arm while drag teaching", flush=True)

        while not stop_requested and time.monotonic() - started < MAX_RECORD_S:
            tick = time.monotonic()
            sample = read_sample(mc, started)
            if sample is not None:
                samples.append(sample)
                if len(samples) == 1:
                    marks.append({"name": "START", "sample_index": 0, **sample})
                    print("[mark] START", flush=True)

                while mark_requests > 0:
                    mark_requests -= 1
                    mark_index = len(marks) - 1
                    if mark_index >= len(MARK_SEQUENCE):
                        print("[warn] extra mark ignored", flush=True)
                        continue
                    mark_name = MARK_SEQUENCE[mark_index]
                    marks.append(
                        {
                            "name": mark_name,
                            "sample_index": len(samples) - 1,
                            **sample,
                        }
                    )
                    print(f"[mark] {mark_name}", flush=True)

            time.sleep(max(0.0, period_s - (time.monotonic() - tick)))
    finally:
        if release_succeeded:
            try:
                lock_servos(mc)
                locked_pose = baseline.read_pose(mc, "LOCKED_FINAL")
                print("[teach] servos locked", flush=True)
            except Exception as exc:
                print(f"[critical] failed to lock servos: {exc}", flush=True)

        payload = {
            "schema": "arm01_drag_taught_route.v1",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sample_hz": SAMPLE_HZ,
            "motion_commands_sent": False,
            "release_all_servos_sent": release_succeeded,
            "power_on_sent_at_end": release_succeeded,
            "initial_pose": initial,
            "locked_final_pose": locked_pose,
            "expected_mark_sequence": ["START", *MARK_SEQUENCE],
            "marks": marks,
            "sample_count": len(samples),
            "samples": samples,
        }
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[saved] {output} samples={len(samples)} marks={len(marks)}", flush=True)

    if len(samples) < 3:
        raise RuntimeError(f"insufficient samples: {len(samples)}")
    if len(marks) != 1 + len(MARK_SEQUENCE):
        raise RuntimeError(
            f"incomplete marks: got {[mark['name'] for mark in marks]}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        print(
            f"[validate] sample_hz={SAMPLE_HZ} max_s={MAX_RECORD_S} "
            f"marks=START,{','.join(MARK_SEQUENCE)}"
        )
        return
    if args.output is None:
        raise SystemExit("--output is required")
    record(args.output.expanduser())


if __name__ == "__main__":
    main()
