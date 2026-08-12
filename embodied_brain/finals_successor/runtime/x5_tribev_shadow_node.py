#!/usr/bin/env python3
"""ROS2 adapter for the independent X5-TriBEV-Flow shadow candidate.

The node subscribes to existing sensor/diagnostic topics and publishes only
under ``/x5_triflow_shadow``. It has no velocity, F407, TF, service, or action
interface. A candidate error publishes ``MONITOR_OFFLINE`` and leaves every
validated finals component untouched.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from x5_tribev_flow.bpu_runtime import (
    CamSemLiteBpuRunner,
    TinyOccFlowBpuRunner,
)
from x5_tribev_flow.contracts import (
    SemanticObservation,
    SemanticProvenance,
    TriBEVConfig,
    TriBEVObservation,
)
from x5_tribev_flow.evidence import EvidenceLedger
from x5_tribev_flow.runtime_core import (
    OccupancyGridSpec,
    Pose2D,
    depth_scan_to_points,
    image_message_to_bgr,
    laser_scan_to_points,
    normalize_semantic_provenance,
    occupancy_grid_to_semantic_bev,
    occupancy_probability_to_ros_data,
    odometry_delta,
    parse_reference_trajectory_probabilities,
    quaternion_to_yaw,
)
from x5_tribev_flow.shadow_guard import (
    SensorHealthMonitor,
    ShadowGuard,
    cross_modal_bev_metrics,
    energy_ood,
    fuse_trajectory_token_evidence,
    occupancy_conditioned_trajectory_tokens,
    trajectory_token_js_divergence,
)
from x5_tribev_flow.tribev import TriBEVFrontend


NODE_NAME = "x5_triflow_shadow_monitor"
NAMESPACE = "/x5_triflow_shadow"
FORBIDDEN_TOPICS = {
    "/cmd_vel",
    "/cmd_vel_safe",
    "/tf",
    "/tf_static",
    "/f407",
}
ROS_INFRASTRUCTURE_TOPICS = {"/parameter_events", "/rosout"}


def _json_message(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _finite_list(value: np.ndarray) -> list[float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return [float(item) for item in array[np.isfinite(array)]]


def _message_stamp_seconds(message: Any, fallback: float) -> float:
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    if stamp is None:
        return fallback
    value = float(getattr(stamp, "sec", 0)) + float(
        getattr(stamp, "nanosec", 0)
    ) * 1e-9
    return value if value > 0.0 and math.isfinite(value) else fallback


def _compact_cam_result(result: Mapping[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {
            "state": "NOT_RUN",
            "real_camera_accuracy_validated": False,
            "shadow_only": True,
            "cmd_vel_authority": False,
        }
    quality = np.asarray(result["quality_probabilities"], dtype=np.float32).reshape(-1)
    return {
        "state": "BPU_RUNTIME_PROBE",
        "latency_ms": float(result["latency_ms"]),
        "quality_class": int(np.argmax(quality)),
        "quality_probability": float(np.max(quality)),
        "semantic_class_fraction": [
            float(value) for value in result["semantic_class_fraction"]
        ],
        "claim_scope": result["claim_scope"],
        "real_camera_accuracy_validated": False,
        "model": dict(result["model"]),
        "shadow_only": True,
        "cmd_vel_authority": False,
    }


def main() -> None:
    import rclpy
    from nav_msgs.msg import OccupancyGrid, Odometry
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, LaserScan
    from std_msgs.msg import String

    class X5TriFlowShadowNode(Node):
        def __init__(self) -> None:
            super().__init__(NODE_NAME)
            self.declare_parameter("tiny_model_bin", "")
            self.declare_parameter("cam_model_bin", "")
            self.declare_parameter("camera_enabled", False)
            self.declare_parameter("camera_topic", "/pt_camera/image_raw")
            self.declare_parameter("camera_period_s", 1.0)
            self.declare_parameter("scan_topic", "/scan")
            self.declare_parameter("depth_scan_topic", "/scan_depth")
            self.declare_parameter("odom_topic", "/odom")
            self.declare_parameter("vision_bev_topic", "/lab_fsd/vision_bev")
            self.declare_parameter("vision_objects_topic", "/lab_fsd/vision_objects")
            self.declare_parameter("reference_tokens_topic", "/lab_fsd/policy_tokens")
            self.declare_parameter("publish_rate_hz", 5.0)
            self.declare_parameter("depth_origin_x_m", 0.25)
            self.declare_parameter("semantic_max_age_s", 4.0)
            self.declare_parameter("evidence_directory", "/tmp/x5_triflow_shadow/evidence")
            self.declare_parameter("evidence_interval_s", 30.0)
            self.declare_parameter("evidence_max_files", 120)
            self.declare_parameter("evidence_max_bytes", 64 * 1024 * 1024)

            tiny_path = str(self.get_parameter("tiny_model_bin").value or "")
            if not tiny_path:
                raise RuntimeError("tiny_model_bin is required")
            self.tiny_runner = TinyOccFlowBpuRunner(tiny_path)
            self.cam_runner: CamSemLiteBpuRunner | None = None
            cam_path = str(self.get_parameter("cam_model_bin").value or "")
            if bool(self.get_parameter("camera_enabled").value) and cam_path:
                self.cam_runner = CamSemLiteBpuRunner(cam_path)

            self.frontend = TriBEVFrontend(
                TriBEVConfig(
                    semantic_max_age_s=float(
                        self.get_parameter("semantic_max_age_s").value
                    )
                )
            )
            self.health_monitor = SensorHealthMonitor()
            self.guard = ShadowGuard()
            self.ledger = EvidenceLedger(
                str(self.get_parameter("evidence_directory").value)
            )
            self.latest_scan = None
            self.latest_scan_sequence = 0
            self.last_processed_scan_sequence = 0
            self.latest_depth_scan = None
            self.latest_odom = None
            self.latest_vision_bev = None
            self.latest_vision_payload: dict[str, Any] = {}
            self.latest_reference_tokens: dict[str, Any] = {}
            self.latest_camera_bgr: np.ndarray | None = None
            self.latest_camera_arrival_s = 0.0
            self.last_camera_infer_s = 0.0
            self.last_camera_result: dict[str, Any] | None = None
            self.last_camera_error = ""
            self.previous_pose: Pose2D | None = None
            self.arrivals: dict[str, deque[float]] = {
                name: deque(maxlen=64)
                for name in ("lidar", "depth", "vision", "odom", "camera")
            }
            self.last_evidence_write_s = 0.0
            self.sequence = 0
            self.inference_failures = 0

            self.create_subscription(
                LaserScan,
                str(self.get_parameter("scan_topic").value),
                self._on_scan,
                qos_profile_sensor_data,
            )
            self.create_subscription(
                LaserScan,
                str(self.get_parameter("depth_scan_topic").value),
                self._on_depth_scan,
                qos_profile_sensor_data,
            )
            self.create_subscription(
                Odometry,
                str(self.get_parameter("odom_topic").value),
                self._on_odom,
                qos_profile_sensor_data,
            )
            self.create_subscription(
                OccupancyGrid,
                str(self.get_parameter("vision_bev_topic").value),
                self._on_vision_bev,
                10,
            )
            self.create_subscription(
                String,
                str(self.get_parameter("vision_objects_topic").value),
                self._on_vision_objects,
                10,
            )
            self.create_subscription(
                String,
                str(self.get_parameter("reference_tokens_topic").value),
                self._on_reference_tokens,
                10,
            )
            if self.cam_runner is not None:
                self.create_subscription(
                    Image,
                    str(self.get_parameter("camera_topic").value),
                    self._on_camera,
                    qos_profile_sensor_data,
                )

            self.pub_future = self.create_publisher(
                OccupancyGrid,
                f"{NAMESPACE}/future_occupancy",
                10,
            )
            self.pub_flow = self.create_publisher(
                String,
                f"{NAMESPACE}/occupancy_flow",
                10,
            )
            self.pub_tokens = self.create_publisher(
                String,
                f"{NAMESPACE}/trajectory_tokens",
                10,
            )
            self.pub_sensor_health = self.create_publisher(
                String,
                f"{NAMESPACE}/sensor_health",
                10,
            )
            self.pub_trust = self.create_publisher(
                String,
                f"{NAMESPACE}/trust_state",
                10,
            )
            self.pub_evidence = self.create_publisher(
                String,
                f"{NAMESPACE}/evidence",
                10,
            )
            self._assert_publisher_boundary()

            requested_rate_hz = float(self.get_parameter("publish_rate_hz").value)
            rate_hz = min(5.0, max(0.2, requested_rate_hz))
            if rate_hz != requested_rate_hz:
                self.get_logger().warning(
                    f"publish_rate_hz clamped from {requested_rate_hz} to {rate_hz}"
                )
            self.create_timer(1.0 / rate_hz, self._tick)
            self.get_logger().info(
                "X5-TriBEV-Flow shadow candidate ready; "
                f"rate={rate_hz:.2f}Hz camera_probe={self.cam_runner is not None} "
                "authority=none"
            )

        def _assert_publisher_boundary(self) -> None:
            published = {
                topic
                for topic, _types in self.get_publisher_names_and_types_by_node(
                    self.get_name(),
                    self.get_namespace(),
                )
            }
            illegal = sorted(
                topic for topic in published
                if topic in FORBIDDEN_TOPICS
                or (
                    topic not in ROS_INFRASTRUCTURE_TOPICS
                    and not topic.startswith(f"{NAMESPACE}/")
                )
            )
            if illegal:
                raise RuntimeError(f"shadow publisher boundary violated: {illegal}")

        def _arrival(self, name: str) -> float:
            now = time.time()
            self.arrivals[name].append(now)
            return now

        def _on_scan(self, message: Any) -> None:
            self.latest_scan = message
            self.latest_scan_sequence += 1
            self._arrival("lidar")

        def _on_depth_scan(self, message: Any) -> None:
            self.latest_depth_scan = message
            self._arrival("depth")

        def _on_odom(self, message: Any) -> None:
            self.latest_odom = message
            self._arrival("odom")

        def _on_vision_bev(self, message: Any) -> None:
            self.latest_vision_bev = message
            self._arrival("vision")

        def _on_vision_objects(self, message: Any) -> None:
            try:
                payload = json.loads(message.data)
            except (TypeError, json.JSONDecodeError):
                payload = {}
            self.latest_vision_payload = payload if isinstance(payload, dict) else {}

        def _on_reference_tokens(self, message: Any) -> None:
            try:
                payload = json.loads(message.data)
            except (TypeError, json.JSONDecodeError):
                payload = {}
            self.latest_reference_tokens = payload if isinstance(payload, dict) else {}

        def _on_camera(self, message: Any) -> None:
            try:
                self.latest_camera_bgr = image_message_to_bgr(
                    message.data,
                    width=int(message.width),
                    height=int(message.height),
                    step=int(message.step),
                    encoding=str(message.encoding),
                )
                self.latest_camera_arrival_s = self._arrival("camera")
            except (TypeError, ValueError) as exc:
                self.last_camera_error = str(exc)[:160]

        def _pose(self, now: float) -> Pose2D:
            message = self.latest_odom
            position = message.pose.pose.position
            orientation = message.pose.pose.orientation
            return Pose2D(
                x_m=float(position.x),
                y_m=float(position.y),
                yaw_rad=quaternion_to_yaw(
                    float(orientation.x),
                    float(orientation.y),
                    float(orientation.z),
                    float(orientation.w),
                ),
                timestamp_s=_message_stamp_seconds(message, now),
            )

        def _odom_speed_m_s(self) -> float:
            twist = self.latest_odom.twist.twist.linear
            speed = math.hypot(float(twist.x), float(twist.y))
            if not math.isfinite(speed):
                return 0.0
            return min(1.5, max(0.0, speed))

        def _semantic_observation(self, now: float) -> SemanticObservation:
            if self.latest_vision_bev is None or not self.arrivals["vision"]:
                return SemanticObservation()
            message = self.latest_vision_bev
            orientation = message.info.origin.orientation
            source = OccupancyGridSpec(
                width=int(message.info.width),
                height=int(message.info.height),
                resolution_m=float(message.info.resolution),
                origin_x_m=float(message.info.origin.position.x),
                origin_y_m=float(message.info.origin.position.y),
                origin_yaw_rad=quaternion_to_yaw(
                    float(orientation.x),
                    float(orientation.y),
                    float(orientation.z),
                    float(orientation.w),
                ),
            )
            risk, known = occupancy_grid_to_semantic_bev(
                message.data,
                source,
                self.frontend.config.geometry,
            )
            provenance, image_supplied = normalize_semantic_provenance(
                self.latest_vision_payload
            )
            age_s = max(0.0, now - self.arrivals["vision"][-1])
            if not np.any(known):
                provenance = SemanticProvenance.UNAVAILABLE
                image_supplied = False
            return SemanticObservation(
                bev=risk,
                provenance=provenance,
                age_s=age_s,
                image_supplied=image_supplied,
            )

        def _sensor_health(
            self,
            *,
            now: float,
            lidar_ranges: np.ndarray,
            depth_ranges: np.ndarray,
            pose: Pose2D | None,
            semantic: SemanticObservation,
        ) -> dict[str, Any]:
            vision_payload = (
                semantic.bev
                if semantic.bev is not None
                else np.empty(0, dtype=np.float32)
            )
            samples = {
                "lidar": {
                    "timestamps_s": list(self.arrivals["lidar"]),
                    "payload": lidar_ranges,
                },
                "depth": {
                    "timestamps_s": list(self.arrivals["depth"]),
                    "payload": depth_ranges,
                },
                "vision": {
                    "timestamps_s": list(self.arrivals["vision"]),
                    "payload": vision_payload,
                    "provenance": {
                        "state": semantic.provenance.value,
                        "image_supplied": semantic.image_supplied,
                    },
                },
                "odom": {
                    "timestamps_s": list(self.arrivals["odom"]),
                    "payload": (
                        [pose.x_m, pose.y_m, pose.yaw_rad]
                        if pose is not None
                        else []
                    ),
                },
            }
            return self.health_monitor.assess_all(samples, now_s=now)

        def _publish_monitor_offline(self, reason: str, now: float) -> None:
            result = self.guard.assess(
                {},
                monitor_online=False,
                monitor_errors=[reason],
            )
            payload = {
                **result,
                "candidate_id": "x5-tribev-flow-shadow-v1",
                "sequence": self.sequence,
                "timestamp_s": now,
                "validated_demo_effect": "none",
            }
            self.pub_trust.publish(String(data=_json_message(payload)))
            self.pub_evidence.publish(String(data=_json_message(payload)))

        def _maybe_run_camera(self, now: float) -> dict[str, Any]:
            if self.cam_runner is None or self.latest_camera_bgr is None:
                return _compact_cam_result(self.last_camera_result)
            period = max(0.2, float(self.get_parameter("camera_period_s").value))
            if now - self.last_camera_infer_s < period:
                return _compact_cam_result(self.last_camera_result)
            self.last_camera_infer_s = now
            try:
                self.last_camera_result = self.cam_runner.infer_bgr(
                    self.latest_camera_bgr
                )
                self.last_camera_error = ""
            except Exception as exc:
                self.last_camera_error = f"{exc.__class__.__name__}:{str(exc)[:140]}"
            compact = _compact_cam_result(self.last_camera_result)
            if self.last_camera_error:
                compact["last_error"] = self.last_camera_error
            return compact

        def _publish_future_grid(
            self,
            probability: np.ndarray,
            timestamp_message: Any,
        ) -> None:
            geometry = self.frontend.config.geometry
            message = OccupancyGrid()
            message.header.stamp = timestamp_message
            message.header.frame_id = "base_footprint"
            message.info.resolution = float(geometry.resolution_m)
            message.info.width = int(geometry.height)
            message.info.height = int(geometry.width)
            message.info.origin.position.x = float(geometry.x_min_m)
            message.info.origin.position.y = float(geometry.y_min_m)
            message.info.origin.orientation.w = 1.0
            message.data = occupancy_probability_to_ros_data(probability)
            self.pub_future.publish(message)

        def _write_evidence_if_due(
            self,
            now: float,
            evidence: Mapping[str, Any],
        ) -> dict[str, Any] | None:
            interval = max(
                5.0,
                float(self.get_parameter("evidence_interval_s").value),
            )
            if now - self.last_evidence_write_s < interval:
                return None
            self.last_evidence_write_s = now
            episode_id = time.strftime(
                "x5-triflow-%Y%m%dT%H%M%S",
                time.localtime(now),
            )
            paths: dict[str, str] = {
                "tiny_occ_flow": str(self.tiny_runner.model_path)
            }
            if self.cam_runner is not None:
                paths["cam_sem_lite"] = str(self.cam_runner.model_path)
            receipt = self.ledger.write_episode(
                episode_id,
                evidence,
                model_paths=paths,
                provenance={
                    "runtime": "board_shadow_candidate",
                    "live_sensor_inputs": True,
                    "validated_demo_modified": False,
                    "shadow_only": True,
                    "cmd_vel_authority": False,
                },
                metadata={
                    "candidate_id": "x5-tribev-flow-shadow-v1",
                    "sequence": self.sequence,
                    "shadow_only": True,
                    "cmd_vel_authority": False,
                },
            )
            receipt["retention"] = self.ledger.prune_records(
                max_files=max(
                    1,
                    int(self.get_parameter("evidence_max_files").value),
                ),
                max_bytes=max(
                    1024 * 1024,
                    int(self.get_parameter("evidence_max_bytes").value),
                ),
            )
            return receipt

        def _tick(self) -> None:
            self.sequence += 1
            now = time.time()
            if self.latest_scan is None or self.latest_odom is None:
                self._publish_monitor_offline("waiting_for_required_inputs", now)
                return
            if self.latest_scan_sequence == self.last_processed_scan_sequence:
                return
            self.last_processed_scan_sequence = self.latest_scan_sequence
            try:
                lidar_ranges = np.asarray(self.latest_scan.ranges, dtype=np.float64)
                depth_ranges = (
                    np.asarray(self.latest_depth_scan.ranges, dtype=np.float64)
                    if self.latest_depth_scan is not None
                    else np.empty(0, dtype=np.float64)
                )
                lidar_points = laser_scan_to_points(
                    lidar_ranges,
                    angle_min=float(self.latest_scan.angle_min),
                    angle_increment=float(self.latest_scan.angle_increment),
                    range_min=float(self.latest_scan.range_min),
                    range_max=float(self.latest_scan.range_max),
                )
                depth_points = (
                    depth_scan_to_points(
                        depth_ranges,
                        angle_min=float(self.latest_depth_scan.angle_min),
                        angle_increment=float(self.latest_depth_scan.angle_increment),
                        range_min=float(self.latest_depth_scan.range_min),
                        range_max=float(self.latest_depth_scan.range_max),
                        origin_x_m=float(
                            self.get_parameter("depth_origin_x_m").value
                        ),
                    )
                    if self.latest_depth_scan is not None
                    else None
                )
                pose = self._pose(now)
                delta = odometry_delta(self.previous_pose, pose)
                self.previous_pose = pose
                semantic = self._semantic_observation(now)
                tribev = self.frontend.update(
                    TriBEVObservation(
                        lidar_points_xy=lidar_points,
                        depth_points_xyz=depth_points,
                        semantic=semantic,
                        lidar_valid=bool(lidar_points.size),
                        depth_valid=bool(
                            depth_points is not None and depth_points.size
                        ),
                        timestamp_s=now,
                    ),
                    delta,
                )
                prediction = self.tiny_runner.infer(tribev.tensor)
                self.inference_failures = 0
            except Exception as exc:
                self.inference_failures += 1
                error = f"{exc.__class__.__name__}:{str(exc)[:180]}"
                self.get_logger().error(
                    f"TriBEV shadow tick failed: {error}",
                    throttle_duration_sec=5.0,
                )
                self._publish_monitor_offline(error, now)
                return

            current = tribev.tensor[0, :8]
            frame_meta = tribev.metadata["frames"][0]
            bevs: dict[str, np.ndarray] = {}
            if bool(frame_meta["lidar_valid"]):
                bevs["lidar"] = current[0]
            if bool(frame_meta["depth_valid"]):
                bevs["depth"] = np.maximum.reduce((current[2], current[3], current[4]))
            if bool(frame_meta["camera_semantic_valid"]):
                bevs["vision"] = current[5]
            bev_metrics = cross_modal_bev_metrics(bevs)
            energy_result = energy_ood(prediction["trajectory_logits"])
            occupancy = np.asarray(
                prediction["future_occupancy"],
                dtype=np.float32,
            )[0]
            model_trajectory_probabilities = np.asarray(
                prediction["trajectory_probabilities"],
                dtype=np.float32,
            ).reshape(-1)
            reference_probabilities = parse_reference_trajectory_probabilities(
                self.latest_reference_tokens
            )

            postprocess_reasons: list[str] = []
            occupancy_tokens = occupancy_conditioned_trajectory_tokens(
                occupancy,
                speed_m_s=self._odom_speed_m_s(),
            )
            occupancy_token_probabilities: np.ndarray | None = None
            fusion_result: dict[str, Any] = {
                "valid": False,
                "reason": "occupancy_token_scoring_unavailable",
            }
            trajectory_probabilities = model_trajectory_probabilities
            trajectory_method = "model_auxiliary_fallback"

            if bool(occupancy_tokens.get("valid", False)):
                occupancy_token_probabilities = np.asarray(
                    occupancy_tokens["probabilities"],
                    dtype=np.float32,
                ).reshape(-1)
                fusion_result = fuse_trajectory_token_evidence(
                    model_trajectory_probabilities,
                    occupancy_token_probabilities,
                    reference_values=reference_probabilities,
                )
                if (
                    not bool(fusion_result.get("valid", False))
                    and reference_probabilities is not None
                ):
                    postprocess_reasons.append(
                        "reference_token_fusion_excluded:"
                        f"{fusion_result.get('reason', 'invalid')}"
                    )
                    fusion_result = fuse_trajectory_token_evidence(
                        model_trajectory_probabilities,
                        occupancy_token_probabilities,
                    )
                if bool(fusion_result.get("valid", False)):
                    trajectory_probabilities = np.asarray(
                        fusion_result["probabilities"],
                        dtype=np.float32,
                    ).reshape(-1)
                    trajectory_method = str(fusion_result["method"])
                else:
                    postprocess_reasons.append(
                        "trajectory_fusion_fallback:"
                        f"{fusion_result.get('reason', 'invalid')}"
                    )
            else:
                postprocess_reasons.append(
                    "occupancy_token_fallback:"
                    f"{occupancy_tokens.get('reason', 'invalid')}"
                )

            token_js_source = (
                occupancy_token_probabilities
                if occupancy_token_probabilities is not None
                else model_trajectory_probabilities
            )
            token_js = (
                trajectory_token_js_divergence(
                    token_js_source,
                    reference_probabilities,
                    inputs_are_logits=False,
                )
                if reference_probabilities is not None
                else None
            )
            health = self._sensor_health(
                now=now,
                lidar_ranges=lidar_ranges,
                depth_ranges=depth_ranges,
                pose=pose,
                semantic=semantic,
            )
            trust = self.guard.assess(
                health,
                bev_metrics=bev_metrics,
                energy_result=energy_result,
                token_js_result=token_js,
            )
            if tribev.populated_history < 5 and trust["status"] == "TRUSTED_SHADOW":
                trust["status"] = "REVIEW"
                trust["trusted"] = False
                trust["review_reasons"] = [
                    *trust.get("review_reasons", []),
                    "temporal_history_warmup",
                ]
            if postprocess_reasons:
                trust["status"] = "REVIEW"
                trust["trusted"] = False
                trust["review_reasons"] = list(
                    dict.fromkeys(
                        [
                            *trust.get("review_reasons", []),
                            *postprocess_reasons,
                        ]
                    )
                )

            flow = np.asarray(prediction["flow"], dtype=np.float32)[0]
            dynamic = np.asarray(
                prediction["dynamic_probability"],
                dtype=np.float32,
            )[0]
            uncertainty = np.asarray(prediction["uncertainty"], dtype=np.float32)[0]
            camera = self._maybe_run_camera(now)
            future_summary = {
                "horizons_s": [0.4, 0.8, 1.2],
                "mean_probability": [
                    float(np.mean(item)) for item in occupancy
                ],
                "peak_probability": [
                    float(np.max(item)) for item in occupancy
                ],
                "dynamic_mean": [
                    float(np.mean(item)) for item in dynamic
                ],
                "uncertainty_mean": [
                    float(np.mean(item)) for item in uncertainty
                ],
                "flow_mean_xy_m": [
                    [
                        float(np.mean(flow[index * 2])),
                        float(np.mean(flow[index * 2 + 1])),
                    ]
                    for index in range(3)
                ],
                "latency_ms": float(prediction["latency_ms"]),
                "model": prediction["model"],
                "output_layouts": prediction["output_layouts"],
                "camera_probe": camera,
                "shadow_only": True,
                "cmd_vel_authority": False,
            }
            token_summary = {
                "vocabulary": "fixed_9_arc_tokens",
                "winner": int(np.argmax(trajectory_probabilities)),
                "probabilities": _finite_list(trajectory_probabilities),
                "method": trajectory_method,
                "raw_model_auxiliary": {
                    "winner": int(np.argmax(model_trajectory_probabilities)),
                    "probabilities": _finite_list(
                        model_trajectory_probabilities
                    ),
                },
                "occupancy_conditioned": occupancy_tokens,
                "fusion": fusion_result,
                "reference_available": reference_probabilities is not None,
                "reference_in_fusion": bool(
                    reference_probabilities is not None
                    and "reference_shadow"
                    in fusion_result.get("sources", [])
                ),
                "js_source": (
                    "occupancy_conditioned"
                    if occupancy_token_probabilities is not None
                    else "model_auxiliary_fallback"
                ),
                "js_to_lab_fsd": token_js,
                "postprocess_reasons": postprocess_reasons,
                "shadow_only": True,
                "cmd_vel_authority": False,
            }
            evidence = {
                "candidate_id": "x5-tribev-flow-shadow-v1",
                "sequence": self.sequence,
                "timestamp_s": now,
                "tribev": {
                    "populated_history": tribev.populated_history,
                    "current_frame": frame_meta,
                    "tensor_shape": list(tribev.tensor.shape),
                },
                "future": future_summary,
                "trajectory": token_summary,
                "trust": trust,
                "validated_demo_modified": False,
                "policy_effect": "none_observe_and_record_only",
                "shadow_only": True,
                "cmd_vel_authority": False,
            }

            self._publish_future_grid(
                np.max(occupancy, axis=0),
                self.get_clock().now().to_msg(),
            )
            self.pub_flow.publish(String(data=_json_message(future_summary)))
            self.pub_tokens.publish(String(data=_json_message(token_summary)))
            self.pub_sensor_health.publish(String(data=_json_message(health)))
            self.pub_trust.publish(String(data=_json_message(trust)))
            disk_receipt = self._write_evidence_if_due(now, evidence)
            if disk_receipt is not None:
                evidence = {**evidence, "disk_receipt": disk_receipt}
            self.pub_evidence.publish(String(data=_json_message(evidence)))

    rclpy.init()
    node = X5TriFlowShadowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
