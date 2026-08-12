#!/usr/bin/env python3
"""Capture one map-frame robot pose for lab_locations.yaml without moving it."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True, help="Location id, for example furnace_1")
    parser.add_argument("--description", default="", help="Human-readable description")
    parser.add_argument("--frame", default="map")
    parser.add_argument("--base-frame", default="base_footprint")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--out", default="", help="Optional JSON evidence path")
    args = parser.parse_args()

    rclpy.init()
    node = Node("capture_nav_location")
    buffer = Buffer()
    listener = TransformListener(buffer, node, spin_thread=False)
    deadline = time.monotonic() + max(0.5, args.timeout)
    transform = None
    last_error = ""
    try:
        while time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            try:
                transform = buffer.lookup_transform(
                    args.frame,
                    args.base_frame,
                    Time(),
                    timeout=Duration(seconds=0.2),
                )
                break
            except TransformException as exc:
                last_error = str(exc)
        if transform is None:
            print(f"ERROR no TF {args.frame} -> {args.base_frame}: {last_error}")
            return 2

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        x = float(translation.x)
        y = float(translation.y)
        theta = yaw_from_quaternion(rotation.x, rotation.y, rotation.z, rotation.w)
        payload = {
            "schema_version": "xrd-nav-location-calibration-v1",
            "captured_at": utc_now(),
            "location_id": args.id,
            "description": args.description,
            "frame_id": args.frame,
            "base_frame": args.base_frame,
            "pose": {"x": x, "y": y, "theta": theta},
            "source": "tf2_lookup_read_only",
            "motion_command_published": False,
        }
        print(f"{args.id}:")
        if args.description:
            print(f"  description: {args.description}")
        print(f"  pose: {{ x: {x:.4f}, y: {y:.4f}, theta: {theta:.5f} }}")
        print("  approach_distance: 0.4")
        if args.id != "home" and math.hypot(x, y) < 0.05 and abs(theta) < 0.05:
            print("WARN captured pose is near the origin; verify this is not an uncalibrated placeholder")
        if args.out:
            out = Path(args.out).expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"evidence: {out}")
        return 0
    finally:
        del listener
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
