#!/usr/bin/env python3
"""Bridge AI-brain 4K tower Vision-BEV into ROS2.

The bridge publishes semantic BEV hints only. It never publishes cmd_vel and it
does not alter Nav2 authority.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from lab_fsd_core import classify_vision_bev_provenance


def _post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def main() -> None:
    import rclpy
    from nav_msgs.msg import OccupancyGrid
    from rclpy.node import Node
    from std_msgs.msg import Float32, String

    class VisionBevBridge(Node):
        def __init__(self) -> None:
            super().__init__("lab_fsd_vision_bev_bridge")
            self.declare_parameter("ai_brain_url", "http://192.0.2.103:8888")
            self.declare_parameter("fetch_rate_hz", 0.35)
            self.declare_parameter("timeout_s", 1.6)
            self.declare_parameter("capture", False)
            self.declare_parameter("enabled", True)
            self.declare_parameter("topic_bev", "/lab_fsd/vision_bev")
            self.declare_parameter("topic_objects", "/lab_fsd/vision_objects")
            self.declare_parameter("topic_risk", "/lab_fsd/vision_risk")

            self.pub_bev = self.create_publisher(
                OccupancyGrid, str(self.get_parameter("topic_bev").value), 10
            )
            self.pub_objects = self.create_publisher(
                String, str(self.get_parameter("topic_objects").value), 10
            )
            self.pub_risk = self.create_publisher(
                Float32, str(self.get_parameter("topic_risk").value), 10
            )
            hz = max(0.05, float(self.get_parameter("fetch_rate_hz").value))
            self.create_timer(1.0 / hz, self._tick)
            self.last_ok = 0.0
            self.get_logger().info(
                f"Lab-FSD Vision-BEV bridge online: ai={self.get_parameter('ai_brain_url').value}, "
                f"rate={hz:.2f}Hz"
            )

        def _tick(self) -> None:
            if not bool(self.get_parameter("enabled").value):
                return
            base = str(self.get_parameter("ai_brain_url").value).rstrip("/")
            url = base + "/api/lab_fsd_vision_bev"
            payload = {"capture": bool(self.get_parameter("capture").value), "include_grid": True}
            timeout_s = float(self.get_parameter("timeout_s").value)
            try:
                out = _post_json(url, payload, timeout_s)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
                self.get_logger().warn(f"Vision-BEV fetch failed: {str(exc)[:120]}", throttle_duration_sec=8.0)
                return
            if not out.get("ok") or not out.get("grid"):
                self.get_logger().warn(f"Vision-BEV invalid: {str(out)[:160]}", throttle_duration_sec=8.0)
                return
            try:
                self._publish(out)
                self.last_ok = time.time()
            except Exception as exc:
                self.get_logger().warn(f"Vision-BEV publish failed: {str(exc)[:120]}", throttle_duration_sec=8.0)

        def _publish(self, out: dict[str, Any]) -> None:
            n = int(out.get("grid_size") or 48)
            res = float(out.get("resolution_m") or 0.10)
            vals = [int(max(0, min(100, int(v)))) for v in out.get("grid", [])]
            if len(vals) != n * n:
                raise ValueError(f"grid length {len(vals)} != {n*n}")
            msg = OccupancyGrid()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = str(out.get("frame_id") or "base_footprint")
            msg.info.resolution = res
            msg.info.width = n
            msg.info.height = n
            msg.info.origin.position.x = -n * res / 2.0
            msg.info.origin.position.y = -n * res / 2.0
            msg.info.origin.orientation.w = 1.0
            msg.data = vals
            self.pub_bev.publish(msg)

            obj_payload = {
                "mode": out.get("mode"),
                "ts": out.get("ts"),
                "risk_score": out.get("risk_score"),
                "objects": out.get("objects", []),
                "camera": out.get("camera", {}),
                "calibration": out.get("calibration", {}),
                "stale_after_s": out.get("stale_after_s"),
                "provenance": classify_vision_bev_provenance(out),
            }
            self.pub_objects.publish(String(data=json.dumps(obj_payload, ensure_ascii=False)))
            self.pub_risk.publish(Float32(data=float(out.get("risk_score") or 0.0)))

    rclpy.init()
    node = VisionBevBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
