"""Request-bound, replay-resistant gate for pickup physical evidence."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from my_robot_msgs.msg import PhysicalEvidence, PhysicalEvidenceRequest
from my_robot_msgs.srv import VerifyPhysicalEvidence

from .physical_evidence_contracts import validate_evidence, validate_request


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


def _evidence_dict(msg: PhysicalEvidence) -> dict[str, Any]:
    return {
        "observed_at_ns": _stamp_ns(msg.header.stamp),
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
        "payload_sha256": str(msg.payload_sha256),
    }


class PhysicalEvidenceGate(Node):
    def __init__(self) -> None:
        super().__init__("physical_evidence_gate")
        self.declare_parameter("evidence_topic", "/pickup/physical_evidence")
        self.declare_parameter("request_topic", "/pickup/physical_evidence_request")
        self.declare_parameter("service_name", "/verify_physical_evidence")
        self.declare_parameter("max_evidence_age_s", 2.0)
        self.declare_parameter("max_future_skew_s", 0.25)
        self.declare_parameter("confidence_floor", 0.80)
        self.declare_parameter("queue_depth", 128)

        self.max_age_ns = int(max(0.1, float(self.get_parameter("max_evidence_age_s").value)) * 1e9)
        self.max_future_skew_ns = int(
            max(0.0, float(self.get_parameter("max_future_skew_s").value)) * 1e9
        )
        self.confidence_floor = min(
            1.0, max(0.0, float(self.get_parameter("confidence_floor").value))
        )
        queue_depth = max(16, min(2048, int(self.get_parameter("queue_depth").value)))
        callback_group = ReentrantCallbackGroup()
        self._condition = threading.Condition()
        self._sequence = 0
        self._queue: deque[tuple[int, PhysicalEvidence, int]] = deque(maxlen=queue_depth)
        self._consumed_evidence_ids: set[str] = set()
        self._completed_request_ids: set[str] = set()
        self._inflight_request_ids: set[str] = set()

        evidence_topic = str(self.get_parameter("evidence_topic").value)
        request_topic = str(self.get_parameter("request_topic").value)
        service_name = str(self.get_parameter("service_name").value)
        self._request_pub = self.create_publisher(PhysicalEvidenceRequest, request_topic, 10)
        self.create_subscription(
            PhysicalEvidence,
            evidence_topic,
            self._on_evidence,
            20,
            callback_group=callback_group,
        )
        self.create_service(
            VerifyPhysicalEvidence,
            service_name,
            self._verify,
            callback_group=callback_group,
        )
        self.get_logger().info(
            "physical evidence gate online; it does not create evidence and accepts only "
            "fresh request-bound hardware observations"
        )

    def _on_evidence(self, msg: PhysicalEvidence) -> None:
        received_at_ns = self.get_clock().now().nanoseconds
        with self._condition:
            self._sequence += 1
            self._queue.append((self._sequence, msg, received_at_ns))
            self._condition.notify_all()

    def _verify(
        self,
        request: VerifyPhysicalEvidence.Request,
        response: VerifyPhysicalEvidence.Response,
    ) -> VerifyPhysicalEvidence.Response:
        request_msg = request.request
        contract = _request_dict(request_msg)
        ok, reason = validate_request(contract)
        if not ok:
            response.confirmed = False
            response.message = f"request rejected: {reason}"
            return response

        request_id = contract["request_id"]
        with self._condition:
            if request_id in self._completed_request_ids or request_id in self._inflight_request_ids:
                response.confirmed = False
                response.message = "request_id replayed or already in flight"
                return response
            self._inflight_request_ids.add(request_id)
            cursor = self._sequence

        request_started_ns = self.get_clock().now().nanoseconds
        deadline = time.monotonic() + float(contract["timeout_s"])
        last_rejection = "no matching evidence received"
        try:
            self._request_pub.publish(request_msg)
            while time.monotonic() < deadline:
                with self._condition:
                    candidates = [item for item in self._queue if item[0] > cursor]
                    cursor = self._sequence
                for _sequence, msg, received_at_ns in candidates:
                    if str(msg.request_id) != request_id:
                        continue
                    evidence = _evidence_dict(msg)
                    valid, validation_reason = validate_evidence(
                        evidence,
                        contract,
                        received_at_ns=received_at_ns,
                        request_started_ns=request_started_ns,
                        now_ns=self.get_clock().now().nanoseconds,
                        max_age_ns=self.max_age_ns,
                        max_future_skew_ns=self.max_future_skew_ns,
                        confidence_floor=self.confidence_floor,
                        consumed_evidence_ids=self._consumed_evidence_ids,
                    )
                    if not valid:
                        last_rejection = validation_reason
                        continue
                    with self._condition:
                        self._consumed_evidence_ids.add(str(msg.evidence_id))
                    response.confirmed = True
                    response.evidence = msg
                    response.message = validation_reason
                    return response
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                with self._condition:
                    self._condition.wait(timeout=min(0.1, remaining))
            response.confirmed = False
            response.message = f"physical evidence timeout: {last_rejection}"
            return response
        finally:
            with self._condition:
                self._inflight_request_ids.discard(request_id)
                self._completed_request_ids.add(request_id)


def main() -> None:
    rclpy.init()
    node = PhysicalEvidenceGate()
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
