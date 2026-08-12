#!/usr/bin/env python3
"""One-process read-only status collector for the finals navigation demo."""

from __future__ import annotations

import argparse
import json
import subprocess
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, String


class Collector(Node):
    def __init__(self) -> None:
        super().__init__("finals_nav_status_once")
        self.values: dict[str, object] = {}
        self.create_subscription(String, "/lab_fsd/fsd_v3_status", self._json_cb("fsd"), 10)
        self.create_subscription(String, "/mppi/stats", self._json_cb("mppi"), 10)
        self.create_subscription(String, "/f407/firmware_info", self._json_cb("firmware"), 10)
        self.create_subscription(Bool, "/f407/estop_latched", lambda m: self.values.__setitem__("estop", bool(m.data)), 10)

    def _json_cb(self, key: str):
        def callback(msg: String) -> None:
            try:
                value = json.loads(msg.data)
            except (TypeError, json.JSONDecodeError):
                value = {}
            self.values[key] = value if isinstance(value, dict) else {}
        return callback


def service_state() -> str:
    try:
        out = subprocess.run(
            ["systemctl", "is-active", "embodied_brain.service"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def memory_line() -> str:
    values: dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                key, raw = line.split(":", 1)
                values[key] = int(raw.strip().split()[0]) // 1024
    except (OSError, ValueError, IndexError):
        return "resource.memory unavailable"
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", 0)
    return f"resource.memory used_mb={max(0, total - available)} available_mb={available} total_mb={total}"


def node_count(names: list[str], target: str) -> int:
    return sum(1 for name in names if name == target)


def graph_names(node: Node) -> list[str]:
    try:
        return [name for name, _ in node.get_node_names_and_namespaces()]
    except Exception:
        return []


def endpoint_names(infos) -> list[str]:
    return sorted({info.node_name for info in infos})


def topic_endpoints(node: Node, topic: str, publishers: bool) -> list[str]:
    try:
        infos = (
            node.get_publishers_info_by_topic(topic)
            if publishers
            else node.get_subscriptions_info_by_topic(topic)
        )
        return endpoint_names(infos)
    except Exception:
        return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    rclpy.init()
    node = Collector()
    deadline = time.monotonic() + max(1.0, args.timeout)
    names = graph_names(node)
    publishers = topic_endpoints(node, "/cmd_vel_safe", True)
    subscribers = topic_endpoints(node, "/cmd_vel_safe", False)
    direct_subscribers = topic_endpoints(node, "/cmd_vel", False)
    try:
        while time.monotonic() < deadline:
            try:
                rclpy.spin_once(node, timeout_sec=0.2)
            except ExternalShutdownException:
                break
            names = graph_names(node) or names
            publishers = topic_endpoints(node, "/cmd_vel_safe", True) or publishers
            subscribers = topic_endpoints(node, "/cmd_vel_safe", False) or subscribers
            direct_subscribers = topic_endpoints(node, "/cmd_vel", False) or direct_subscribers
            graph_ready = all(
                node_count(names, target) == 1
                for target in (
                    "bt_navigator",
                    "planner_server",
                    "controller_server",
                    "collision_monitor",
                    "lab_fsd_bev_shadow_planner",
                    "mppi_node",
                )
            )
            topology_ready = (
                publishers == ["collision_monitor"]
                and subscribers == ["serial_f407"]
                and "serial_f407" not in direct_subscribers
            )
            if len(node.values) >= 4 and graph_ready and topology_ready:
                break
        names = graph_names(node) or names
        publishers = topic_endpoints(node, "/cmd_vel_safe", True) or publishers
        subscribers = topic_endpoints(node, "/cmd_vel_safe", False) or subscribers
        direct_subscribers = topic_endpoints(node, "/cmd_vel", False) or direct_subscribers
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass

    fsd = node.values.get("fsd", {}) if isinstance(node.values.get("fsd"), dict) else {}
    mppi = node.values.get("mppi", {}) if isinstance(node.values.get("mppi"), dict) else {}
    fw = node.values.get("firmware", {}) if isinstance(node.values.get("firmware"), dict) else {}
    inputs = fsd.get("input_status", {}) if isinstance(fsd.get("input_status"), dict) else {}
    sources = inputs.get("sources", {}) if isinstance(inputs.get("sources"), dict) else {}
    bpu = fsd.get("bpu", {}) if isinstance(fsd.get("bpu"), dict) else {}
    tiny = bpu.get("tiny_occ_risk", {}) if isinstance(bpu.get("tiny_occ_risk"), dict) else {}

    print(f"FINALS_NAV_STATUS {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    print(
        "stack "
        f"service={service_state()} nav2={node_count(names, 'bt_navigator')} "
        f"collision_monitor={node_count(names, 'collision_monitor')} "
        f"lab_fsd={node_count(names, 'lab_fsd_bev_shadow_planner')} "
        f"mppi={node_count(names, 'mppi_node')}"
    )
    topology_ok = (
        publishers == ["collision_monitor"]
        and subscribers == ["serial_f407"]
        and "serial_f407" not in direct_subscribers
    )
    print(
        f"safety.topology state={'PASS' if topology_ok else 'FAIL'} f407_input=/cmd_vel_safe "
        f"safe_publishers={publishers} safe_subscribers={subscribers} direct_subscribers={direct_subscribers}"
    )
    for key in ("scan", "scan_depth", "odom"):
        src = sources.get(key, {}) if isinstance(sources.get(key), dict) else {}
        print(
            f"sensor.{key} state={src.get('state', 'unavailable')} fresh={src.get('fresh', False)} "
            f"usable={src.get('usable', False)} age_s={src.get('age_s', 'n/a')}"
        )
    vision = sources.get("vision_bev", {}) if isinstance(sources.get("vision_bev"), dict) else {}
    prov = vision.get("provenance", {}) if isinstance(vision.get("provenance"), dict) else {}
    live_4k = vision.get("state") == "live" and prov.get("state") == "live_camera" and prov.get("image_supplied") is True
    print(
        f"sensor.4k state={'live' if live_4k else 'fallback'} provenance={prov.get('state', 'unknown')} "
        f"image_supplied={prov.get('image_supplied', False)} usable={vision.get('usable', False)}"
    )
    tiny_live = tiny.get("runtime") == "hobot_dnn" and tiny.get("state") == "forward_ok" and tiny.get("used") is True
    print(
        f"bpu.tiny_occ_risk state={'live' if tiny_live else 'fallback'} runtime={tiny.get('runtime', 'none')} "
        f"forward={tiny.get('state', 'unknown')} used={tiny.get('used', False)} latency_ms={tiny.get('latency_ms', 'n/a')}"
    )
    print(f"bpu.anomaly loaded={bpu.get('anomaly_autoencoder', False)} authority=shadow_only")
    print(
        f"bpu.mppi state={'live' if mppi.get('use_bpu') is True else 'fallback'} eval_ms={mppi.get('eval_ms', 'n/a')} "
        f"proposed_only={mppi.get('proposed_only', 'unknown')} direct_cmd_vel={mppi.get('direct_cmd_vel', 'unknown')}"
    )
    print(
        f"lab_fsd overall={inputs.get('overall', 'unavailable')} shadow_only={fsd.get('shadow_only', 'unknown')} "
        f"cmd_vel_authority={fsd.get('cmd_vel_authority', 'unknown')}"
    )
    print(
        f"safety.estop={node.values.get('estop', 'unknown')} firmware_identity={fw.get('identity_valid', 'unknown')} "
        f"test_mode={fw.get('test_mode', 'unknown')}"
    )
    print(memory_line())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
