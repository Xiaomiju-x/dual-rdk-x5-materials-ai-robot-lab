#!/usr/bin/env python3
"""Publish fresh synthetic LaserScan frames for isolated Collision Monitor tests.

The tool refuses production/default ROS domains.  It publishes sensor fixtures
only; it never publishes velocity, actuator, estop, or F407 commands.
"""
from __future__ import annotations

import argparse
import os
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


def require_isolated_fixture_mode(explicit_fixture: bool) -> int:
    if not explicit_fixture:
        raise SystemExit("refusing to run without --fixture-only")
    local_only = os.environ.get("ROS_LOCALHOST_ONLY", "").strip().lower()
    if local_only not in {"1", "true", "yes", "on"}:
        raise SystemExit("ROS_LOCALHOST_ONLY=1 is required")
    try:
        domain_id = int(os.environ.get("ROS_DOMAIN_ID", "0"))
    except ValueError as exc:
        raise SystemExit("ROS_DOMAIN_ID must be an integer") from exc
    if domain_id < 200:
        raise SystemExit("ROS_DOMAIN_ID must be >= 200 for fixture isolation")
    return domain_id


class ScanFixture(Node):
    def __init__(self, *, mode: str, topic: str, frame_id: str, rate_hz: float) -> None:
        super().__init__("nav2_collision_fixture")
        self.publisher = self.create_publisher(LaserScan, topic, 10)
        self.frame_id = frame_id
        self.distance_m = 0.20 if mode == "near" else 5.0
        self.timer = self.create_timer(1.0 / rate_hz, self.publish_scan)

    def publish_scan(self) -> None:
        scan = LaserScan()
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.header.frame_id = self.frame_id
        scan.angle_min = -0.5
        scan.angle_max = 0.5
        scan.angle_increment = 0.1
        scan.time_increment = 0.0
        scan.scan_time = 0.05
        scan.range_min = 0.05
        scan.range_max = 10.0
        scan.ranges = [self.distance_m] * 11
        scan.intensities = []
        self.publisher.publish(scan)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-only", action="store_true")
    parser.add_argument("--mode", choices=("far", "near"), required=True)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--topic", default="/scan")
    parser.add_argument("--frame-id", default="laser_link")
    args = parser.parse_args()

    domain_id = require_isolated_fixture_mode(args.fixture_only)
    if not 0.5 <= args.duration <= 120.0:
        raise SystemExit("--duration must be in [0.5, 120.0] seconds")
    if not 1.0 <= args.rate_hz <= 100.0:
        raise SystemExit("--rate-hz must be in [1.0, 100.0]")

    rclpy.init()
    node = ScanFixture(
        mode=args.mode,
        topic=args.topic,
        frame_id=args.frame_id,
        rate_hz=args.rate_hz,
    )
    node.get_logger().info(
        f"isolated fixture active domain={domain_id} mode={args.mode} "
        f"topic={args.topic} duration={args.duration:.1f}s"
    )
    deadline = time.monotonic() + args.duration
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
