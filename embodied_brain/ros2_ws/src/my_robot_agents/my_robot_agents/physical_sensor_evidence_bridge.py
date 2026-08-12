"""Fail-closed bridge from calibrated hardware samples to pickup physical evidence."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from my_robot_msgs.msg import (
    HardwareSensorSample,
    PhysicalEvidence,
    PhysicalEvidenceRequest,
)

from .physical_evidence_contracts import (
    canonical_evidence_sha256,
    validate_request,
)
from .physical_sensor_contracts import (
    SHA256_RE,
    canonical_manifest_sha256,
    evaluate_sample,
    validate_calibration,
    validate_sample,
)


def _stamp_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _request_dict(msg: PhysicalEvidenceRequest) -> dict[str, Any]:
    return {
        "request_id": str(msg.request_id),
        "task_id": str(msg.task_id),
        "bottle_id": str(msg.bottle_id),
        "location_id": str(msg.location_id),
        "expected_observation": str(msg.expected_observation),
        "not_before_ns": _stamp_ns(msg.not_before),
        "timeout_s": float(msg.timeout_s),
        "min_confidence": float(msg.min_confidence),
        "expected_value": float(msg.expected_value),
        "tolerance": float(msg.tolerance),
        "unit": str(msg.unit),
    }


def _sample_dict(msg: HardwareSensorSample) -> dict[str, Any]:
    return {
        "observed_at_ns": _stamp_ns(msg.header.stamp),
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
        "sample_sha256": str(msg.sample_sha256),
    }


def _normalized_identity(namespace: str, node_name: str) -> str:
    ns = "/" + str(namespace or "").strip("/")
    if ns == "/":
        return f"/{str(node_name).strip('/')}"
    return f"{ns}/{str(node_name).strip('/')}"


class PhysicalSensorEvidenceBridge(Node):
    def __init__(self, *, parameter_overrides: list[Any] | None = None) -> None:
        super().__init__(
            "physical_sensor_evidence_bridge",
            parameter_overrides=parameter_overrides or [],
        )
        self.declare_parameter("enabled", False)
        self.declare_parameter("sample_topic", "/pickup/hardware_sensor_sample")
        self.declare_parameter("request_topic", "/pickup/physical_evidence_request")
        self.declare_parameter("evidence_topic", "/pickup/physical_evidence")
        self.declare_parameter(
            "status_topic", "/pickup/physical_evidence_bridge_status"
        )
        self.declare_parameter("calibration_manifest", "")
        self.declare_parameter("expected_calibration_sha256", "")
        self.declare_parameter("allow_unapproved_calibration", False)
        self.declare_parameter("expected_driver_instance_id", "")
        self.declare_parameter("expected_publisher_node", "")
        self.declare_parameter("expected_publisher_namespace", "/")
        self.declare_parameter("require_unique_publisher", True)
        self.declare_parameter("max_sample_age_s", 0.75)
        self.declare_parameter("max_future_skew_s", 0.25)
        self.declare_parameter("minimum_quality", 0.80)
        self.declare_parameter("max_active_requests", 16)

        self.enabled = bool(self.get_parameter("enabled").value)
        self.sample_topic = str(self.get_parameter("sample_topic").value)
        self.request_topic = str(self.get_parameter("request_topic").value)
        self.evidence_topic = str(self.get_parameter("evidence_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.expected_driver_instance_id = str(
            self.get_parameter("expected_driver_instance_id").value
        ).strip()
        publisher_node = str(self.get_parameter("expected_publisher_node").value).strip()
        publisher_namespace = str(
            self.get_parameter("expected_publisher_namespace").value
        ).strip()
        self.expected_publisher_identity = _normalized_identity(
            publisher_namespace, publisher_node
        )
        self.require_unique_publisher = bool(
            self.get_parameter("require_unique_publisher").value
        )
        self.max_sample_age_ns = int(
            max(0.05, float(self.get_parameter("max_sample_age_s").value)) * 1e9
        )
        self.max_future_skew_ns = int(
            max(0.0, float(self.get_parameter("max_future_skew_s").value)) * 1e9
        )
        self.minimum_quality = min(
            1.0, max(0.0, float(self.get_parameter("minimum_quality").value))
        )
        self.max_active_requests = max(
            1, min(128, int(self.get_parameter("max_active_requests").value))
        )

        self._lock = threading.RLock()
        self._arrival_sequence = 0
        self._last_sensor_sequence = 0
        self._active_requests: dict[str, dict[str, Any]] = {}
        self._completed_request_ids: set[str] = set()
        self._last_rejection = ""
        self._last_sample_ns = 0
        self._published_evidence = 0
        self._publisher_identities: list[str] = []
        self._manifest: dict[str, Any] = {}
        self._calibration_sha256 = ""
        self._status_pub = self.create_publisher(String, self.status_topic, 10)
        self.create_timer(1.0, self._publish_status)

        if not self.enabled:
            self.get_logger().warn(
                "physical sensor evidence bridge disabled; no sensor/request subscriptions "
                "or evidence publisher were created"
            )
            return

        calibration_path = Path(
            str(self.get_parameter("calibration_manifest").value)
        ).expanduser()
        expected_sha = str(
            self.get_parameter("expected_calibration_sha256").value
        ).strip().lower()
        if not calibration_path.is_file():
            raise RuntimeError(f"calibration manifest missing: {calibration_path}")
        if not SHA256_RE.fullmatch(expected_sha):
            raise RuntimeError("expected_calibration_sha256 must be an exact SHA-256")
        raw_manifest = calibration_path.read_bytes()
        actual_sha = canonical_manifest_sha256(raw_manifest)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"calibration SHA-256 mismatch: expected={expected_sha} actual={actual_sha}"
            )
        try:
            manifest = json.loads(raw_manifest.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid calibration JSON: {exc}") from exc
        if not isinstance(manifest, dict):
            raise RuntimeError("calibration manifest must be a JSON object")
        calibration_ok, calibration_reason = validate_calibration(
            manifest,
            allow_unapproved=bool(
                self.get_parameter("allow_unapproved_calibration").value
            ),
        )
        if not calibration_ok:
            raise RuntimeError(f"calibration rejected: {calibration_reason}")
        if not self.expected_driver_instance_id:
            raise RuntimeError("expected_driver_instance_id is required")
        if not publisher_node:
            raise RuntimeError("expected_publisher_node is required")

        self._manifest = manifest
        self._calibration_sha256 = actual_sha
        callback_group = ReentrantCallbackGroup()
        self._evidence_pub = self.create_publisher(
            PhysicalEvidence, self.evidence_topic, 20
        )
        self.create_subscription(
            PhysicalEvidenceRequest,
            self.request_topic,
            self._on_request,
            20,
            callback_group=callback_group,
        )
        self.create_subscription(
            HardwareSensorSample,
            self.sample_topic,
            self._on_sample,
            20,
            callback_group=callback_group,
        )
        self.get_logger().info(
            "physical sensor evidence bridge enabled: "
            f"sensor_id={manifest['sensor_id']} source_type={manifest['source_type']} "
            f"calibration_sha256={actual_sha} publisher={self.expected_publisher_identity}"
        )

    def _publisher_identity_ok(self) -> tuple[bool, str]:
        infos = self.get_publishers_info_by_topic(self.sample_topic)
        identities = sorted(
            {
                _normalized_identity(info.node_namespace, info.node_name)
                for info in infos
            }
        )
        with self._lock:
            self._publisher_identities = identities
        if self.require_unique_publisher and len(infos) != 1:
            return False, f"sample topic publisher count is {len(infos)}, expected 1"
        if self.expected_publisher_identity not in identities:
            return False, (
                f"expected sample publisher {self.expected_publisher_identity} not present; "
                f"observed={identities}"
            )
        return True, "sample publisher identity passed"

    def _on_request(self, msg: PhysicalEvidenceRequest) -> None:
        if not self.enabled:
            return
        request = _request_dict(msg)
        valid, reason = validate_request(request)
        if not valid:
            with self._lock:
                self._last_rejection = f"request rejected: {reason}"
            return
        observation = str(request["expected_observation"])
        observations = self._manifest.get("observations")
        if not isinstance(observations, dict) or observation not in observations:
            return
        request_id = str(request["request_id"])
        now_ros_ns = self.get_clock().now().nanoseconds
        with self._lock:
            if request_id in self._completed_request_ids or request_id in self._active_requests:
                self._last_rejection = "request_id replayed at sensor bridge"
                return
            if len(self._active_requests) >= self.max_active_requests:
                self._last_rejection = "active request capacity reached"
                return
            self._active_requests[request_id] = {
                "contract": request,
                "arrival_cursor": self._arrival_sequence,
                "received_ros_ns": now_ros_ns,
                "deadline_monotonic": time.monotonic() + float(request["timeout_s"]),
            }

    def _on_sample(self, msg: HardwareSensorSample) -> None:
        if not self.enabled:
            return
        now_ns = self.get_clock().now().nanoseconds
        now_monotonic = time.monotonic()
        sample = _sample_dict(msg)
        with self._lock:
            self._arrival_sequence += 1
            arrival_sequence = self._arrival_sequence
            previous_sequence = self._last_sensor_sequence
        publisher_ok, publisher_reason = self._publisher_identity_ok()
        if not publisher_ok:
            with self._lock:
                self._last_rejection = publisher_reason
            return
        sample_ok, sample_reason = validate_sample(
            sample,
            self._manifest,
            expected_driver_instance_id=self.expected_driver_instance_id,
            now_ns=now_ns,
            max_age_ns=self.max_sample_age_ns,
            max_future_skew_ns=self.max_future_skew_ns,
            minimum_quality=self.minimum_quality,
            previous_sequence=previous_sequence,
        )
        if not sample_ok:
            with self._lock:
                self._last_rejection = sample_reason
            return
        with self._lock:
            self._last_sensor_sequence = int(sample["sequence"])
            self._last_sample_ns = int(sample["observed_at_ns"])
            expired_request_ids = [
                request_id
                for request_id, state in self._active_requests.items()
                if now_monotonic >= float(state["deadline_monotonic"])
            ]
            for request_id in expired_request_ids:
                self._active_requests.pop(request_id, None)
                self._completed_request_ids.add(request_id)
            candidates = list(self._active_requests.items())

        for request_id, state in candidates:
            request = state["contract"]
            if arrival_sequence <= int(state["arrival_cursor"]):
                continue
            if int(sample["observed_at_ns"]) < int(state["received_ros_ns"]):
                with self._lock:
                    self._last_rejection = "sample was observed before request receipt"
                continue
            if int(sample["observed_at_ns"]) < int(request["not_before_ns"]):
                with self._lock:
                    self._last_rejection = "sample predates actuator stage"
                continue
            decision_ok, decision, decision_reason = evaluate_sample(
                sample, request, self._manifest
            )
            if not decision_ok:
                with self._lock:
                    self._last_rejection = decision_reason
                continue
            evidence = self._build_evidence(msg, sample, request, decision)
            self._evidence_pub.publish(evidence)
            with self._lock:
                self._active_requests.pop(request_id, None)
                self._completed_request_ids.add(request_id)
                self._published_evidence += 1
                self._last_rejection = ""
            break

    def _build_evidence(
        self,
        sample_msg: HardwareSensorSample,
        sample: dict[str, Any],
        request: dict[str, Any],
        decision: dict[str, Any],
    ) -> PhysicalEvidence:
        seed = (
            f"{self._calibration_sha256}:{sample['sensor_id']}:"
            f"{sample['driver_instance_id']}:{sample['sequence']}:{request['request_id']}"
        ).encode("utf-8")
        evidence_id = f"pe:{hashlib.sha256(seed).hexdigest()[:40]}"
        detail = json.dumps(
            {
                "bridge": self.get_fully_qualified_name(),
                "calibration_id": self._manifest.get("calibration_id"),
                "calibration_sha256": self._calibration_sha256,
                "driver_instance_id": sample["driver_instance_id"],
                "sample_sequence": sample["sequence"],
                "sample_sha256": sample["sample_sha256"],
                "decision": decision.get("decision"),
                "rule_mode": decision.get("mode"),
                "raw_detail": str(sample.get("detail") or "")[:256],
            },
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        evidence = PhysicalEvidence()
        evidence.header.stamp = sample_msg.header.stamp
        evidence.header.frame_id = str(self._manifest["frame_id"])
        evidence.evidence_id = evidence_id
        evidence.request_id = str(request["request_id"])
        evidence.sensor_id = str(self._manifest["sensor_id"])
        evidence.source_type = str(self._manifest["source_type"])
        evidence.observation = str(request["expected_observation"])
        evidence.task_id = str(request["task_id"])
        evidence.bottle_id = str(request["bottle_id"])
        evidence.location_id = str(request["location_id"])
        evidence.confirmed = True
        evidence.hardware_observed = True
        evidence.confidence = float(decision["confidence"])
        evidence.measured_value = float(decision["measured_value"])
        evidence.unit = str(decision["unit"])
        evidence.detail = detail
        evidence_dict = {
            "observed_at_ns": int(sample["observed_at_ns"]),
            "frame_id": evidence.header.frame_id,
            "evidence_id": evidence.evidence_id,
            "request_id": evidence.request_id,
            "sensor_id": evidence.sensor_id,
            "source_type": evidence.source_type,
            "observation": evidence.observation,
            "task_id": evidence.task_id,
            "bottle_id": evidence.bottle_id,
            "location_id": evidence.location_id,
            "confirmed": True,
            "hardware_observed": True,
            "confidence": float(evidence.confidence),
            "measured_value": evidence.measured_value,
            "unit": evidence.unit,
            "detail": evidence.detail,
        }
        evidence.payload_sha256 = canonical_evidence_sha256(evidence_dict)
        return evidence

    def _publish_status(self) -> None:
        now = time.monotonic()
        with self._lock:
            expired = [
                request_id
                for request_id, state in self._active_requests.items()
                if now >= float(state["deadline_monotonic"])
            ]
            for request_id in expired:
                self._active_requests.pop(request_id, None)
                self._completed_request_ids.add(request_id)
            status = {
                "schema_version": "xrd-physical-sensor-evidence-bridge-v1",
                "enabled": self.enabled,
                "node": self.get_fully_qualified_name(),
                "sample_topic": self.sample_topic,
                "request_topic": self.request_topic,
                "evidence_topic": self.evidence_topic,
                "sensor_id": self._manifest.get("sensor_id", ""),
                "source_type": self._manifest.get("source_type", ""),
                "calibration_sha256": self._calibration_sha256,
                "production_authorized": self._manifest.get("production_authorized") is True,
                "expected_driver_instance_id": self.expected_driver_instance_id,
                "expected_publisher": self.expected_publisher_identity,
                "publisher_identities": list(self._publisher_identities),
                "active_requests": len(self._active_requests),
                "expired_requests": len(expired),
                "last_sensor_sequence": self._last_sensor_sequence,
                "last_sample_ns": self._last_sample_ns,
                "published_evidence": self._published_evidence,
                "last_rejection": self._last_rejection,
                "commands_published": False,
            }
        msg = String()
        msg.data = json.dumps(
            status,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self._status_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = PhysicalSensorEvidenceBridge()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
