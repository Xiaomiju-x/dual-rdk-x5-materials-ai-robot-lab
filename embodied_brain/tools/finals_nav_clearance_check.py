#!/usr/bin/env python3
"""Read-only finals navigation clearance and costmap diagnostic."""
from __future__ import annotations

import json
import math
import time

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener


def rotate(q, x: float, y: float, z: float = 0.0) -> tuple[float, float, float]:
    # Quaternion-vector rotation without an extra geometry dependency.
    tx = 2.0 * (q.y * z - q.z * y)
    ty = 2.0 * (q.z * x - q.x * z)
    tz = 2.0 * (q.x * y - q.y * x)
    return (
        x + q.w * tx + (q.y * tz - q.z * ty),
        y + q.w * ty + (q.z * tx - q.x * tz),
        z + q.w * tz + (q.x * ty - q.y * tx),
    )


class Check(Node):
    def __init__(self) -> None:
        super().__init__("finals_nav_clearance_check")
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        self.scans: dict[str, LaserScan] = {}
        self.costmap: OccupancyGrid | None = None
        self.create_subscription(
            LaserScan, "/scan", lambda msg: self.scans.setdefault("scan", msg), qos_profile_sensor_data
        )
        self.create_subscription(
            LaserScan,
            "/scan_depth",
            lambda msg: self.scans.setdefault("scan_depth", msg),
            qos_profile_sensor_data,
        )
        map_qos = QoSProfile(depth=1)
        map_qos.reliability = ReliabilityPolicy.RELIABLE
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(OccupancyGrid, "/global_costmap/costmap", self._map, map_qos)

    def _map(self, msg: OccupancyGrid) -> None:
        self.costmap = msg

    def scan_report(self, name: str, msg: LaserScan) -> dict:
        transform = self.tf.lookup_transform(
            "base_footprint", msg.header.frame_id, Time(), timeout=Duration(seconds=1.0)
        ).transform
        points = []
        angle = msg.angle_min
        for value in msg.ranges:
            if math.isfinite(value) and msg.range_min <= value <= msg.range_max:
                rx, ry, _ = rotate(transform.rotation, value * math.cos(angle), value * math.sin(angle))
                points.append((rx + transform.translation.x, ry + transform.translation.y, value))
            angle += msg.angle_increment
        front = [(x, y, r) for x, y, r in points if 0.02 <= x <= 0.55 and abs(y) <= 0.32]
        body = [(x, y, r) for x, y, r in points if x * x + y * y <= 0.34 * 0.34]
        slow = [(x, y, r) for x, y, r in points if -0.05 <= x <= 0.95 and abs(y) <= 0.45]
        nearest = sorted(points, key=lambda p: p[0] * p[0] + p[1] * p[1])[:8]
        return {
            "frame": msg.header.frame_id,
            "range_min": round(float(msg.range_min), 3),
            "valid_points": len(points),
            "front_stop_points": len(front),
            "body_stop_points": len(body),
            "front_slow_points": len(slow),
            "nearest_base_xy": [[round(x, 3), round(y, 3), round(r, 3)] for x, y, r in nearest],
        }

    def cost_report(self) -> dict:
        msg = self.costmap
        if msg is None:
            return {"available": False}
        tf = self.tf.lookup_transform("map", "base_footprint", Time(), timeout=Duration(seconds=1.0))
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        ox, oy = msg.info.origin.position.x, msg.info.origin.position.y
        res, width, height = msg.info.resolution, msg.info.width, msg.info.height
        samples = []
        for step in range(0, 21):
            d = step * 0.05
            x = tf.transform.translation.x + d * math.cos(yaw)
            y = tf.transform.translation.y + d * math.sin(yaw)
            ix, iy = int((x - ox) / res), int((y - oy) / res)
            value = None
            if 0 <= ix < width and 0 <= iy < height:
                value = int(msg.data[iy * width + ix])
            samples.append([round(d, 2), value])
        return {
            "available": True,
            "size": [width, height],
            "resolution": round(float(res), 3),
            "robot_map": [round(tf.transform.translation.x, 3), round(tf.transform.translation.y, 3), round(yaw, 3)],
            "forward_centerline_costs": samples,
        }


def main() -> int:
    rclpy.init()
    node = Check()
    deadline = time.monotonic() + 18.0
    transforms_ready = False
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
        if len(node.scans) < 2 or node.costmap is None:
            continue
        scan_frames_ready = all(
            node.tf.can_transform(
                "base_footprint",
                msg.header.frame_id,
                Time(),
                timeout=Duration(seconds=0.05),
            )
            for msg in node.scans.values()
        )
        map_ready = node.tf.can_transform(
            "map",
            "base_footprint",
            Time(),
            timeout=Duration(seconds=0.05),
        )
        transforms_ready = scan_frames_ready and map_ready
        if transforms_ready:
            break
    report = {
        "ok": len(node.scans) == 2 and node.costmap is not None and transforms_ready,
        "transforms_ready": transforms_ready,
    }
    for name, msg in node.scans.items():
        try:
            report[name] = node.scan_report(name, msg)
        except Exception as exc:
            report[name] = {"error": f"{type(exc).__name__}:{exc}"}
    try:
        report["global_costmap"] = node.cost_report()
    except Exception as exc:
        report["global_costmap"] = {"error": f"{type(exc).__name__}:{exc}"}
    print(json.dumps(report, sort_keys=True))
    node.destroy_node()
    rclpy.shutdown()
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
