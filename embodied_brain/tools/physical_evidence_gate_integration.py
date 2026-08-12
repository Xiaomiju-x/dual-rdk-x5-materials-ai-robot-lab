#!/usr/bin/env python3
"""Isolated ROS integration test for the pickup physical-evidence gate.

This tool publishes synthetic evidence only inside the selected private ROS
domain. It never opens F407 devices and never publishes velocity commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from my_robot_agents.physical_evidence_contracts import canonical_evidence_sha256
from my_robot_agents.physical_evidence_gate import PhysicalEvidenceGate
from my_robot_msgs.msg import PhysicalEvidence, PhysicalEvidenceRequest
from my_robot_msgs.srv import VerifyPhysicalEvidence


SCHEMA = "xrd-physical-evidence-gate-integration-v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def evidence_dict(msg: PhysicalEvidence) -> dict[str, Any]:
    return {
        "observed_at_ns": stamp_ns(msg.header.stamp),
        "frame_id": str(msg.header.frame_id),
        "evidence_id": str(msg.evidence_id),
        "request_id": str(msg.request_id),
        "sensor_id": str(msg.sensor_id),
        "source_type": str(msg.source_type),
        "observation": str(msg.observation),
        "task_id": str(msg.task_id),
        "bottle_id": str(msg.bottle_id),
        "location_id": str(msg.location_id),
        "confirmed": bool(msg.confirmed),
        "hardware_observed": bool(msg.hardware_observed),
        "confidence": float(msg.confidence),
        "measured_value": float(msg.measured_value),
        "unit": str(msg.unit),
        "detail": str(msg.detail),
    }


class Probe(Node):
    def __init__(self) -> None:
        super().__init__("physical_evidence_integration_probe")
        callback_group = ReentrantCallbackGroup()
        self.publisher = self.create_publisher(
            PhysicalEvidence, "/pickup/physical_evidence", 10
        )
        self.create_subscription(
            PhysicalEvidenceRequest,
            "/pickup/physical_evidence_request",
            self._on_request,
            10,
            callback_group=callback_group,
        )
        self.client = self.create_client(
            VerifyPhysicalEvidence,
            "/verify_physical_evidence",
            callback_group=callback_group,
        )
        self.scenario = ""
        self.scenario_lock = threading.Lock()
        self.replay_id = "evidence-integration-replay-0001"

    def _on_request(self, request: PhysicalEvidenceRequest) -> None:
        with self.scenario_lock:
            scenario = self.scenario
        if not scenario:
            return
        msg = PhysicalEvidence()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "integration_fixture"
        msg.request_id = request.request_id
        msg.task_id = request.task_id
        msg.bottle_id = request.bottle_id
        msg.location_id = request.location_id
        msg.observation = request.expected_observation
        msg.sensor_id = "integration-synthetic-sensor"
        msg.confirmed = True
        msg.hardware_observed = True
        msg.confidence = 0.95
        msg.measured_value = request.expected_value
        msg.unit = request.unit
        msg.detail = "simulation_only integration evidence"
        if scenario in {"valid_lift", "tampered_hash", "replay"}:
            msg.source_type = "encoder"
        elif scenario == "valid_object":
            msg.source_type = "vision_depth"
        else:
            msg.source_type = "firmware_output_state"
        msg.evidence_id = (
            self.replay_id
            if scenario == "replay"
            else f"evidence-integration-{scenario}-0001"
        )
        payload = evidence_dict(msg)
        msg.payload_sha256 = canonical_evidence_sha256(payload)
        if scenario == "tampered_hash":
            msg.payload_sha256 = "0" * 64
        self.publisher.publish(msg)

    def verify(self, scenario: str, observation: str, timeout_s: float = 0.6):
        with self.scenario_lock:
            self.scenario = scenario
        request = VerifyPhysicalEvidence.Request()
        msg = request.request
        now = self.get_clock().now()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "integration_probe"
        msg.request_id = f"request-{scenario}-{time.time_ns()}"
        msg.task_id = "integration-task"
        msg.bottle_id = "integration-bottle"
        msg.location_id = "integration-fixture"
        msg.expected_observation = observation
        msg.not_before = now.to_msg()
        msg.timeout_s = float(timeout_s)
        msg.min_confidence = 0.8
        msg.expected_value = 0.05 if observation == "lift_position_confirmed" else 0.0
        msg.tolerance = 0.01 if observation == "lift_position_confirmed" else 0.0
        msg.unit = "m" if observation == "lift_position_confirmed" else ""
        future = self.client.call_async(request)
        done = threading.Event()
        future.add_done_callback(lambda _future: done.set())
        if not done.wait(timeout=timeout_s + 2.0):
            raise TimeoutError(f"scenario {scenario} service timeout")
        return future.result()


def no_forbidden_fd() -> tuple[bool, list[str]]:
    targets: list[str] = []
    fd_root = Path("/proc/self/fd")
    for fd in fd_root.iterdir():
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if target.startswith(("/dev/F407", "/dev/ttyUSB", "/dev/ttyACM", "/dev/serial")):
            targets.append(target)
    return not targets, targets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    report: dict[str, Any] = {
        "schema_version": SCHEMA,
        "started_at_unix": started,
        "simulation_only": True,
        "real_hardware_touched": False,
        "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", ""),
        "ros_localhost_only": os.environ.get("ROS_LOCALHOST_ONLY", ""),
        "checks": [],
    }

    rclpy.init()
    gate = PhysicalEvidenceGate()
    probe = Probe()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(gate)
    executor.add_node(probe)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        ready = probe.client.wait_for_service(timeout_sec=5.0)
        report["checks"].append({"name": "service_ready", "passed": bool(ready)})
        if not ready:
            raise RuntimeError("/verify_physical_evidence unavailable")

        valid_lift = probe.verify("valid_lift", "lift_position_confirmed")
        report["checks"].append(
            {
                "name": "valid_lift_accepted",
                "passed": bool(valid_lift.confirmed),
                "message": valid_lift.message,
            }
        )
        probe.replay_id = str(valid_lift.evidence.evidence_id)

        valid_object = probe.verify("valid_object", "object_attached")
        report["checks"].append(
            {
                "name": "valid_object_accepted",
                "passed": bool(valid_object.confirmed),
                "message": valid_object.message,
            }
        )

        tampered = probe.verify("tampered_hash", "lift_position_confirmed")
        report["checks"].append(
            {
                "name": "tampered_hash_rejected",
                "passed": not bool(tampered.confirmed),
                "message": tampered.message,
            }
        )

        wrong_source = probe.verify("wrong_source", "object_attached")
        report["checks"].append(
            {
                "name": "wrong_source_rejected",
                "passed": not bool(wrong_source.confirmed),
                "message": wrong_source.message,
            }
        )

        replay = probe.verify("replay", "lift_position_confirmed")
        report["checks"].append(
            {
                "name": "evidence_replay_rejected",
                "passed": not bool(replay.confirmed),
                "message": replay.message,
            }
        )

        fd_ok, forbidden_fds = no_forbidden_fd()
        report["checks"].append(
            {
                "name": "no_physical_device_fd",
                "passed": fd_ok,
                "forbidden_fds": forbidden_fds,
            }
        )
        cmd_vel_publishers = probe.get_publishers_info_by_topic("/cmd_vel")
        report["checks"].append(
            {
                "name": "no_cmd_vel_publisher",
                "passed": len(cmd_vel_publishers) == 0,
                "publishers": [item.node_name for item in cmd_vel_publishers],
            }
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        executor.shutdown(timeout_sec=3.0)
        gate.destroy_node()
        probe.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=3.0)

    report["generated_at_unix"] = time.time()
    report["tool_sha256"] = sha256_file(Path(__file__).resolve())
    passed = all(item.get("passed") is True for item in report["checks"])
    report["overall"] = "PASS" if passed and "error" not in report else "FAIL"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"overall": report["overall"], "out": str(out)}, sort_keys=True))
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
