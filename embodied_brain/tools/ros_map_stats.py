#!/usr/bin/env python3
"""Print one-shot OccupancyGrid statistics for SLAM diagnostics."""

from __future__ import annotations

import math

import rclpy
from nav_msgs.msg import OccupancyGrid


def main() -> int:
    rclpy.init()
    node = rclpy.create_node("map_stats_once")
    result: dict[str, OccupancyGrid] = {}

    def callback(msg: OccupancyGrid) -> None:
        result["map"] = msg

    sub = node.create_subscription(OccupancyGrid, "/map", callback, 1)
    deadline = node.get_clock().now().nanoseconds + int(8e9)
    while rclpy.ok() and "map" not in result:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.get_clock().now().nanoseconds > deadline:
            break

    if "map" not in result:
        print("MAP_STATS timeout waiting for /map")
        node.destroy_subscription(sub)
        node.destroy_node()
        rclpy.shutdown()
        return 1

    msg = result["map"]
    data = list(msg.data)
    total = len(data)
    unknown = sum(1 for v in data if v < 0)
    free = sum(1 for v in data if v == 0)
    occupied = sum(1 for v in data if v > 50)
    mid = sum(1 for v in data if 0 < v <= 50)
    w = msg.info.width
    h = msg.info.height
    res = msg.info.resolution
    yaw = 2.0 * math.atan2(msg.info.origin.orientation.z, msg.info.origin.orientation.w)
    print(
        "MAP_STATS "
        f"width={w} height={h} res={res:.3f} "
        f"meters={w * res:.2f}x{h * res:.2f} "
        f"unknown={unknown} free={free} mid={mid} occupied={occupied} total={total} "
        f"origin=({msg.info.origin.position.x:.2f},{msg.info.origin.position.y:.2f}) "
        f"yaw={yaw:.3f}"
    )
    node.destroy_subscription(sub)
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
