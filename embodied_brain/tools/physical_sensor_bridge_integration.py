#!/usr/bin/env python3
"""Private-domain ROS integration for the calibrated physical-sensor bridge.

All samples are synthetic and explicitly fixture-only. The tool never opens a
serial/GPIO device and never publishes velocity or actuator commands.
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
from rclpy.parameter import Parameter
from std_msgs.msg import String

from my_robot_agents.physical_sensor_contracts import canonical_sample_sha256
from my_robot_agents.physical_sensor_evidence_bridge import (
    PhysicalSensorEvidenceBridge,
)
from my_robot_agents.physical_evidence_gate import PhysicalEvidenceGate
from my_robot_msgs.msg import (
    HardwareSensorSample,
    PhysicalEvidence,
    PhysicalEvidenceRequest,
)
from my_robot_msgs.srv import VerifyPhysicalEvidence


SCHEMA = "xrd-physical-sensor-bridge-integration-v1"
DRIVER_ID = "fixture-driver-0001"
SENSOR_ID = "fixture-lift-encoder-1"
SAMPLE_TOPIC = "/pickup/hardware_sensor_sample"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def sample_dict(msg: HardwareSensorSample) -> dict[str, Any]:
    return {
        "observed_at_ns": stamp_ns(msg.header.stamp),
        "frame_id": str(msg.header.frame_id),
        "sensor_id": str(msg.sensor_id),
        "driver_instance_id": str(msg.driver_instance_id),
        "sequence": int(msg.sequence),
        "hardware_observed": bool(msg.hardware_observed),
        "digital_state": bool(msg.digital_state),
        "raw_value": float(msg.raw_value),
        "raw_unit": str(msg.raw_unit),
        "quality": float(msg.quality),
        "detail": str(msg.detail),
    }


class FixtureDriver(Node):
    def __init__(self) -> None:
        super().__init__("physical_sensor_fixture_driver")
        callback_group = ReentrantCallbackGroup()
        self.sample_pub = self.create_publisher(
            HardwareSensorSample, SAMPLE_TOPIC, 20
        )
        self.create_subscription(
            PhysicalEvidenceRequest,
            "/pickup/physical_evidence_request",
            self._on_request,
            20,
            callback_group=callback_group,
        )
        self.create_subscription(
            PhysicalEvidence,
            "/pickup/physical_evidence",
            self._on_evidence,
            20,
            callback_group=callback_group,
        )
        self.create_subscription(
            String,
            "/pickup/physical_evidence_bridge_status",
            self._on_status,
            10,
            callback_group=callback_group,
        )
        self.client = self.create_client(
            VerifyPhysicalEvidence,
            "/verify_physical_evidence",
            callback_group=callback_group,
        )
        self._lock = threading.Lock()
        self.scenario = ""
        self.scenario_sequence = 0
        self.evidence: list[PhysicalEvidence] = []
        self.status: dict[str, Any] = {}
        self._fixture_timers: list[threading.Timer] = []

    def _on_evidence(self, msg: PhysicalEvidence) -> None:
        with self._lock:
            self.evidence.append(msg)

    def _on_status(self, msg: String) -> None:
        try:
            status = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if isinstance(status, dict):
            with self._lock:
                self.status = status

    def _on_request(self, _request: PhysicalEvidenceRequest) -> None:
        with self._lock:
            scenario = self.scenario
            sequence = self.scenario_sequence
        if not scenario or scenario == "pre_request_only":
            return
        timer = threading.Timer(0.08, self.publish_sample, args=(scenario, sequence))
        timer.daemon = True
        with self._lock:
            self._fixture_timers.append(timer)
        timer.start()

    def publish_sample(self, scenario: str, sequence: int) -> None:
        msg = HardwareSensorSample()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "lift_fixture"
        msg.sensor_id = SENSOR_ID
        msg.driver_instance_id = DRIVER_ID
        msg.sequence = int(sequence)
        msg.hardware_observed = True
        msg.digital_state = False
        msg.raw_value = 500.0
        msg.raw_unit = "count"
        msg.quality = 0.95
        msg.detail = f"simulation_only scenario={scenario}"
        if scenario == "wrong_driver":
            msg.driver_instance_id = "fixture-driver-wrong"
        payload = sample_dict(msg)
        msg.sample_sha256 = canonical_sample_sha256(payload)
        if scenario == "tampered_hash":
            msg.sample_sha256 = "0" * 64
        self.sample_pub.publish(msg)

    def verify(self, scenario: str, sequence: int, timeout_s: float = 0.55):
        with self._lock:
            self.scenario = scenario
            self.scenario_sequence = sequence
        request = VerifyPhysicalEvidence.Request()
        msg = request.request
        now = self.get_clock().now()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "fixture_probe"
        msg.request_id = f"sensor-request-{scenario}-{time.time_ns()}"
        msg.task_id = "sensor-integration-task"
        msg.bottle_id = "sensor-integration-bottle"
        msg.location_id = "sensor-integration-fixture"
        msg.expected_observation = "lift_position_confirmed"
        msg.not_before = now.to_msg()
        msg.timeout_s = float(timeout_s)
        msg.min_confidence = 0.8
        msg.expected_value = 0.05
        msg.tolerance = 0.005
        msg.unit = "m"
        future = self.client.call_async(request)
        done = threading.Event()
        future.add_done_callback(lambda _future: done.set())
        if not done.wait(timeout=timeout_s + 2.0):
            raise TimeoutError(f"scenario {scenario} service timeout")
        return future.result()


def no_forbidden_fd() -> tuple[bool, list[str]]:
    targets: list[str] = []
    for fd in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if target.startswith(
            ("/dev/F407", "/dev/ttyUSB", "/dev/ttyACM", "/dev/serial", "/dev/gpio")
        ):
            targets.append(target)
    return not targets, targets


def calibration_fixture() -> dict[str, Any]:
    return {
        "schema_version": "xrd-physical-sensor-calibration-v1",
        "calibration_id": "fixture-lift-encoder-calibration-v1",
        "sensor_id": SENSOR_ID,
        "source_type": "encoder",
        "frame_id": "lift_fixture",
        "raw_unit": "count",
        "hardware_required": True,
        "production_authorized": False,
        "confidence_ceiling": 0.97,
        "observations": {
            "lift_position_confirmed": {
                "mode": "position",
                "scale": 0.0001,
                "offset": 0.0,
                "required_state": None,
                "output_unit": "m",
            }
        },
    }


def add_check(report: dict[str, Any], name: str, passed: bool, **detail: Any) -> None:
    report["checks"].append({"name": name, "passed": bool(passed), **detail})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    calibration_path = out.with_name(f"{out.stem}_fixture_calibration.json")
    calibration_path.write_text(
        json.dumps(calibration_fixture(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    calibration_sha = sha256_file(calibration_path)

    report: dict[str, Any] = {
        "schema_version": SCHEMA,
        "started_at_unix": time.time(),
        "simulation_only": True,
        "fixture_calibration": True,
        "production_authorized": False,
        "real_hardware_touched": False,
        "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", ""),
        "ros_localhost_only": os.environ.get("ROS_LOCALHOST_ONLY", ""),
        "calibration_path": str(calibration_path),
        "calibration_sha256": calibration_sha,
        "checks": [],
    }

    rclpy.init()
    gate = PhysicalEvidenceGate()
    bridge = PhysicalSensorEvidenceBridge(
        parameter_overrides=[
            Parameter("enabled", Parameter.Type.BOOL, True),
            Parameter(
                "calibration_manifest",
                Parameter.Type.STRING,
                str(calibration_path),
            ),
            Parameter(
                "expected_calibration_sha256",
                Parameter.Type.STRING,
                calibration_sha,
            ),
            Parameter("allow_unapproved_calibration", Parameter.Type.BOOL, True),
            Parameter(
                "expected_driver_instance_id", Parameter.Type.STRING, DRIVER_ID
            ),
            Parameter(
                "expected_publisher_node",
                Parameter.Type.STRING,
                "physical_sensor_fixture_driver",
            ),
            Parameter("expected_publisher_namespace", Parameter.Type.STRING, "/"),
            Parameter("require_unique_publisher", Parameter.Type.BOOL, True),
            Parameter("max_sample_age_s", Parameter.Type.DOUBLE, 1.0),
            Parameter("minimum_quality", Parameter.Type.DOUBLE, 0.8),
        ]
    )
    driver = FixtureDriver()
    executor = MultiThreadedExecutor(num_threads=8)
    for node in (gate, bridge, driver):
        executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    rogue: Node | None = None
    try:
        ready = driver.client.wait_for_service(timeout_sec=5.0)
        add_check(report, "service_ready", ready)
        if not ready:
            raise RuntimeError("/verify_physical_evidence unavailable")
        time.sleep(0.5)

        driver.publish_sample("pre_request_only", 1)
        time.sleep(0.2)
        pre_request = driver.verify("pre_request_only", 2)
        add_check(
            report,
            "pre_request_sample_rejected",
            not bool(pre_request.confirmed),
            message=pre_request.message,
        )

        valid = driver.verify("valid", 2)
        add_check(
            report,
            "request_bound_encoder_evidence_accepted",
            bool(valid.confirmed)
            and valid.evidence.source_type == "encoder"
            and abs(float(valid.evidence.measured_value) - 0.05) <= 1e-9,
            message=valid.message,
            evidence_id=str(valid.evidence.evidence_id),
            evidence_sha256=str(valid.evidence.payload_sha256),
        )

        tampered = driver.verify("tampered_hash", 3)
        add_check(
            report,
            "tampered_sample_hash_rejected",
            not bool(tampered.confirmed),
            message=tampered.message,
        )

        wrong_driver = driver.verify("wrong_driver", 4)
        add_check(
            report,
            "wrong_driver_instance_rejected",
            not bool(wrong_driver.confirmed),
            message=wrong_driver.message,
        )

        duplicate = driver.verify("duplicate_sequence", 2)
        add_check(
            report,
            "duplicate_sensor_sequence_rejected",
            not bool(duplicate.confirmed),
            message=duplicate.message,
        )

        rogue = Node("physical_sensor_fixture_rogue")
        rogue.create_publisher(HardwareSensorSample, SAMPLE_TOPIC, 10)
        executor.add_node(rogue)
        time.sleep(0.5)
        multi_publisher = driver.verify("multiple_publishers", 5)
        add_check(
            report,
            "multiple_sample_publishers_rejected",
            not bool(multi_publisher.confirmed),
            message=multi_publisher.message,
        )

        time.sleep(1.2)
        with driver._lock:
            status = dict(driver.status)
            evidence_count = len(driver.evidence)
        add_check(
            report,
            "bridge_status_truthful",
            status.get("enabled") is True
            and status.get("production_authorized") is False
            and status.get("commands_published") is False
            and status.get("calibration_sha256") == calibration_sha
            and int(status.get("published_evidence") or 0) == 1,
            status=status,
            observed_evidence_count=evidence_count,
        )

        fd_ok, forbidden_fds = no_forbidden_fd()
        add_check(
            report,
            "no_physical_device_fd",
            fd_ok,
            forbidden_fds=forbidden_fds,
        )
        cmd_vel_publishers = driver.get_publishers_info_by_topic("/cmd_vel")
        add_check(
            report,
            "no_cmd_vel_publisher",
            len(cmd_vel_publishers) == 0,
            publishers=[item.node_name for item in cmd_vel_publishers],
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if rogue is not None:
            executor.remove_node(rogue)
            rogue.destroy_node()
        executor.shutdown(timeout_sec=3.0)
        for timer in driver._fixture_timers:
            timer.join(timeout=1.0)
        for node in (gate, bridge, driver):
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=3.0)

    report["generated_at_unix"] = time.time()
    report["tool_sha256"] = sha256_file(Path(__file__).resolve())
    report["bridge_sha256"] = sha256_file(
        Path(__file__).resolve().parents[1]
        / "ros2_ws"
        / "src"
        / "my_robot_agents"
        / "my_robot_agents"
        / "physical_sensor_evidence_bridge.py"
    )
    passed = all(item.get("passed") is True for item in report["checks"])
    report["overall"] = "PASS" if passed and "error" not in report else "FAIL"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"overall": report["overall"], "out": str(out)}, sort_keys=True))
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
