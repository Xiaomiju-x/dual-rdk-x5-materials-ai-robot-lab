#!/usr/bin/env python3
"""Read-only ROS2 sidecar that stages real TriBEV observations.

The node has subscriptions only. It does not expose an output ROS interface and
has no authority over navigation, the validated finals demo, or the F407.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
SUCCESSOR_ROOT = SCRIPT_PATH.parents[1]
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
for import_root in (REPOSITORY_ROOT, SUCCESSOR_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from embodied_brain.finals_successor.x5_tribev_flow.contracts import (  # noqa: E402
    SemanticObservation,
    SemanticProvenance,
    TriBEVConfig,
    TriBEVObservation,
)
from embodied_brain.finals_successor.x5_tribev_flow.raw_staging import (  # noqa: E402
    RawObservation,
    RawStagingStore,
)
from embodied_brain.finals_successor.x5_tribev_flow.runtime_core import (  # noqa: E402
    OccupancyGridSpec,
    Pose2D,
    depth_scan_to_points,
    laser_scan_to_points,
    normalize_semantic_provenance,
    occupancy_grid_to_semantic_bev,
    odometry_delta,
    quaternion_to_yaw,
)
from embodied_brain.finals_successor.x5_tribev_flow.tribev import (  # noqa: E402
    TriBEVFrontend,
)

NODE_NAME = "x5_tribev_readonly_collector"
_NS_PER_SECOND = 1_000_000_000


@dataclass(slots=True)
class _SensorFrameMeta:
    validity: np.ndarray
    ages_s: np.ndarray
    provenance: np.ndarray
    image_supplied: int

    def aged(self, elapsed_s: float) -> _SensorFrameMeta:
        ages = self.ages_s.copy()
        valid = self.validity.astype(bool)
        ages[valid] += max(0.0, elapsed_s)
        return _SensorFrameMeta(
            validity=self.validity.copy(),
            ages_s=ages,
            provenance=self.provenance.copy(),
            image_supplied=self.image_supplied,
        )


def _message_timestamp_ns(message: Any, fallback_ns: int | None = None) -> int:
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    seconds = int(getattr(stamp, "sec", 0) or 0)
    nanoseconds = int(getattr(stamp, "nanosec", 0) or 0)
    value = seconds * _NS_PER_SECOND + nanoseconds
    if value > 0:
        return value
    return int(fallback_ns) if fallback_ns is not None else time.time_ns()


def _pose_from_odometry(message: Any, fallback_timestamp_ns: int) -> Pose2D:
    position = message.pose.pose.position
    orientation = message.pose.pose.orientation
    timestamp_ns = _message_timestamp_ns(message, fallback_timestamp_ns)
    return Pose2D(
        x_m=float(position.x),
        y_m=float(position.y),
        yaw_rad=quaternion_to_yaw(
            float(orientation.x),
            float(orientation.y),
            float(orientation.z),
            float(orientation.w),
        ),
        timestamp_s=timestamp_ns / _NS_PER_SECOND,
    )


def _semantic_dataset_provenance(payload: dict[str, Any]) -> str:
    nested = payload.get("provenance")
    source = nested if isinstance(nested, dict) else payload
    state = str(source.get("state") or "").strip().lower()
    if state == "live_camera":
        return "live_camera"
    if state in {"cached", "cached_camera"}:
        return "cached_camera"
    if state in {"fixture", "fixture_prior"}:
        return "fixture_prior"
    return "unavailable"


def _sanitize_history(
    tensor: np.ndarray,
    frame_metadata: list[dict[str, Any]],
    sensor_history: list[_SensorFrameMeta],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert newest-first runtime history into strict chronological storage."""

    newest = np.asarray(tensor, dtype=np.float32).reshape(5, 8, 64, 64).copy()
    validity_newest = np.zeros((5, 3), dtype=np.uint8)
    ages_newest = np.full((5, 3), -1.0, dtype=np.float32)
    provenance_newest = np.full((5, 3), "unavailable", dtype="<U32")
    supplied_newest = np.zeros(5, dtype=np.uint8)
    for index, (metadata, sensor_meta) in enumerate(zip(frame_metadata, sensor_history, strict=True)):
        lidar_valid = bool(metadata["lidar_valid"])
        depth_valid = bool(metadata["depth_valid"])
        vision_valid = bool(
            metadata["camera_semantic_valid"]
            and metadata["camera_semantic_provenance"] == "live_camera"
            and metadata["camera_image_supplied"]
        )
        validity_newest[index] = (
            int(lidar_valid),
            int(depth_valid),
            int(vision_valid),
        )
        provenance_newest[index] = sensor_meta.provenance
        provenance_newest[index, 0] = "live_sensor" if lidar_valid else "unavailable"
        provenance_newest[index, 1] = "live_sensor" if depth_valid else "unavailable"
        if vision_valid:
            provenance_newest[index, 2] = "live_camera"
            supplied_newest[index] = 1
        else:
            newest[index, 5].fill(0.0)
            supplied_newest[index] = 0
        for sensor_index, valid in enumerate((lidar_valid, depth_valid, vision_valid)):
            if valid:
                ages_newest[index, sensor_index] = max(0.0, float(sensor_meta.ages_s[sensor_index]))
            else:
                ages_newest[index, sensor_index] = -1.0

        old_fraction = float(metadata["sensor_validity_fraction"])
        if old_fraction > 0.0:
            coverage = newest[index, 6] > 0.0
        else:
            coverage = np.zeros((64, 64), dtype=bool)
        new_fraction = float(np.mean(validity_newest[index]))
        newest[index, 6] = coverage.astype(np.float32) * new_fraction
        newest[index, 7] = np.maximum.reduce(
            (
                newest[index, 0],
                newest[index, 2],
                newest[index, 3],
                newest[index, 4],
                newest[index, 5],
            )
        )

    return (
        np.ascontiguousarray(newest[::-1], dtype=np.float32),
        np.ascontiguousarray(validity_newest[::-1], dtype=np.uint8),
        np.ascontiguousarray(ages_newest[::-1], dtype=np.float32),
        np.ascontiguousarray(provenance_newest[::-1], dtype=np.str_),
        np.ascontiguousarray(supplied_newest[::-1], dtype=np.uint8),
    )


def main() -> None:
    import rclpy
    from nav_msgs.msg import OccupancyGrid, Odometry
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import String

    class X5TriBEVReadonlyCollector(Node):
        def __init__(self) -> None:
            super().__init__(NODE_NAME)
            self.declare_parameter("scan_topic", "/scan")
            self.declare_parameter("depth_scan_topic", "/scan_depth")
            self.declare_parameter("odom_topic", "/odom")
            self.declare_parameter("vision_bev_topic", "/lab_fsd/vision_bev")
            self.declare_parameter("vision_provenance_topic", "/lab_fsd/vision_objects")
            self.declare_parameter(
                "staging_root",
                "/home/rdk/x5_tribev_flow_successor_data",
            )
            self.declare_parameter("scenario_id", "real_navigation_unlabeled")
            self.declare_parameter("minimum_frame_period_s", 0.18)
            self.declare_parameter("depth_max_age_s", 0.30)
            self.declare_parameter("odom_max_age_s", 0.25)
            self.declare_parameter("semantic_max_age_s", 1.50)
            self.declare_parameter("depth_origin_x_m", 0.25)
            self.declare_parameter("anchor_stride_s", 1.0)
            self.declare_parameter("future_tolerance_s", 0.12)
            self.declare_parameter("memory_window_s", 8.0)

            self.frontend = TriBEVFrontend(
                TriBEVConfig(
                    semantic_max_age_s=float(self.get_parameter("semantic_max_age_s").value),
                    accepted_semantic_provenance=(SemanticProvenance.LIVE_CAMERA,),
                )
            )
            self.latest_depth = None
            self.latest_depth_arrival_ns = 0
            self.latest_odom = None
            self.latest_odom_arrival_ns = 0
            self.latest_vision = None
            self.latest_vision_arrival_ns = 0
            self.latest_vision_payload: dict[str, Any] = {}
            self.latest_vision_payload_arrival_ns = 0
            self.previous_pose: Pose2D | None = None
            self.sensor_history: list[_SensorFrameMeta] = []
            self.last_processed_ns = 0
            self.store = self._new_store()

            self.create_subscription(
                LaserScan,
                str(self.get_parameter("scan_topic").value),
                self._on_scan,
                qos_profile_sensor_data,
            )
            self.create_subscription(
                LaserScan,
                str(self.get_parameter("depth_scan_topic").value),
                self._on_depth,
                qos_profile_sensor_data,
            )
            self.create_subscription(
                Odometry,
                str(self.get_parameter("odom_topic").value),
                self._on_odom,
                qos_profile_sensor_data,
            )
            vision_topic = str(self.get_parameter("vision_bev_topic").value)
            if vision_topic:
                self.create_subscription(
                    OccupancyGrid,
                    vision_topic,
                    self._on_vision,
                    qos_profile_sensor_data,
                )
            provenance_topic = str(self.get_parameter("vision_provenance_topic").value)
            if provenance_topic:
                self.create_subscription(
                    String,
                    provenance_topic,
                    self._on_vision_provenance,
                    10,
                )
            self.get_logger().info(
                "X5 TriBEV read-only collector ready; "
                f"session={self.store.session_id} ROS output interfaces=0"
            )

        def _new_store(self) -> RawStagingStore:
            session_id = time.strftime(
                "x5-real-%Y%m%dT%H%M%S",
                time.localtime(),
            )
            return RawStagingStore(
                str(self.get_parameter("staging_root").value),
                session_id=session_id,
                scenario_id=str(self.get_parameter("scenario_id").value),
                anchor_stride_s=float(self.get_parameter("anchor_stride_s").value),
                future_tolerance_s=float(self.get_parameter("future_tolerance_s").value),
                memory_window_s=float(self.get_parameter("memory_window_s").value),
            )

        def _on_depth(self, message: Any) -> None:
            self.latest_depth = message
            self.latest_depth_arrival_ns = time.time_ns()

        def _on_odom(self, message: Any) -> None:
            self.latest_odom = message
            self.latest_odom_arrival_ns = time.time_ns()

        def _on_vision(self, message: Any) -> None:
            self.latest_vision = message
            self.latest_vision_arrival_ns = time.time_ns()

        def _on_vision_provenance(self, message: Any) -> None:
            try:
                payload = json.loads(message.data)
            except (TypeError, json.JSONDecodeError):
                payload = {}
            self.latest_vision_payload = payload if isinstance(payload, dict) else {}
            self.latest_vision_payload_arrival_ns = time.time_ns()

        def _semantic(
            self,
            reference_ns: int,
        ) -> tuple[SemanticObservation, str, bool]:
            message = self.latest_vision
            if message is None:
                return SemanticObservation(), "unavailable", False
            age_s = (
                abs(
                    reference_ns
                    - _message_timestamp_ns(
                        message,
                        self.latest_vision_arrival_ns,
                    )
                )
                / _NS_PER_SECOND
            )
            maximum_age = float(self.get_parameter("semantic_max_age_s").value)
            if age_s > maximum_age:
                return SemanticObservation(), "unavailable", False
            if (
                not self.latest_vision_payload_arrival_ns
                or abs(self.latest_vision_arrival_ns - self.latest_vision_payload_arrival_ns) / _NS_PER_SECOND
                > maximum_age
            ):
                return SemanticObservation(), "unavailable", False
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
            runtime_provenance, image_supplied = normalize_semantic_provenance(self.latest_vision_payload)
            dataset_provenance = _semantic_dataset_provenance(self.latest_vision_payload)
            if not np.any(known):
                runtime_provenance = SemanticProvenance.UNAVAILABLE
                dataset_provenance = "unavailable"
                image_supplied = False
            semantic = SemanticObservation(
                bev=risk,
                provenance=runtime_provenance,
                age_s=age_s,
                image_supplied=image_supplied,
            )
            valid_live = bool(
                runtime_provenance == SemanticProvenance.LIVE_CAMERA and image_supplied and np.any(known)
            )
            return semantic, dataset_provenance, valid_live

        def _raw_semantic_payload(
            self,
            reference_ns: int,
        ) -> tuple[np.ndarray, np.ndarray]:
            message = self.latest_vision
            if message is None:
                return (
                    np.empty(0, dtype=np.int16),
                    np.empty(0, dtype=np.float64),
                )
            age_s = (
                abs(
                    reference_ns
                    - _message_timestamp_ns(
                        message,
                        self.latest_vision_arrival_ns,
                    )
                )
                / _NS_PER_SECOND
            )
            if age_s > float(self.get_parameter("semantic_max_age_s").value):
                return (
                    np.empty(0, dtype=np.int16),
                    np.empty(0, dtype=np.float64),
                )
            orientation = message.info.origin.orientation
            return (
                np.asarray(message.data, dtype=np.int16).reshape(-1),
                np.asarray(
                    (
                        int(message.info.width),
                        int(message.info.height),
                        float(message.info.resolution),
                        float(message.info.origin.position.x),
                        float(message.info.origin.position.y),
                        quaternion_to_yaw(
                            float(orientation.x),
                            float(orientation.y),
                            float(orientation.z),
                            float(orientation.w),
                        ),
                    ),
                    dtype=np.float64,
                ),
            )

        def _on_scan(self, message: Any) -> None:
            reference_ns = _message_timestamp_ns(message)
            minimum_period_ns = int(
                float(self.get_parameter("minimum_frame_period_s").value) * _NS_PER_SECOND
            )
            if reference_ns <= self.last_processed_ns:
                if reference_ns < self.last_processed_ns:
                    self.get_logger().warning("scan time moved backwards; starting a new raw session")
                    self.frontend.reset()
                    self.previous_pose = None
                    self.sensor_history.clear()
                    self.store = self._new_store()
                    self.last_processed_ns = 0
                else:
                    return
            if reference_ns - self.last_processed_ns < minimum_period_ns:
                return
            if self.latest_odom is None:
                return
            odom_age_s = (
                abs(
                    reference_ns
                    - _message_timestamp_ns(
                        self.latest_odom,
                        self.latest_odom_arrival_ns,
                    )
                )
                / _NS_PER_SECOND
            )
            if odom_age_s > float(self.get_parameter("odom_max_age_s").value):
                return
            try:
                pose = _pose_from_odometry(
                    self.latest_odom,
                    self.latest_odom_arrival_ns or reference_ns,
                )
                lidar_points = laser_scan_to_points(
                    message.ranges,
                    angle_min=float(message.angle_min),
                    angle_increment=float(message.angle_increment),
                    range_min=float(message.range_min),
                    range_max=float(message.range_max),
                )
                depth_points = None
                depth_age_s = math.inf
                if self.latest_depth is not None:
                    depth_age_s = (
                        abs(
                            reference_ns
                            - _message_timestamp_ns(
                                self.latest_depth,
                                self.latest_depth_arrival_ns,
                            )
                        )
                        / _NS_PER_SECOND
                    )
                    if depth_age_s <= float(self.get_parameter("depth_max_age_s").value):
                        depth_points = depth_scan_to_points(
                            self.latest_depth.ranges,
                            angle_min=float(self.latest_depth.angle_min),
                            angle_increment=float(self.latest_depth.angle_increment),
                            range_min=float(self.latest_depth.range_min),
                            range_max=float(self.latest_depth.range_max),
                            origin_x_m=float(self.get_parameter("depth_origin_x_m").value),
                        )
                semantic, vision_provenance, vision_valid = self._semantic(reference_ns)
                delta = odometry_delta(self.previous_pose, pose)
                output = self.frontend.update(
                    TriBEVObservation(
                        lidar_points_xy=lidar_points,
                        depth_points_xyz=depth_points,
                        semantic=semantic,
                        lidar_valid=bool(lidar_points.size),
                        depth_valid=bool(depth_points is not None and depth_points.size),
                        timestamp_s=reference_ns / _NS_PER_SECOND,
                    ),
                    delta,
                )
                self.previous_pose = pose
                self.sensor_history = [
                    _SensorFrameMeta(
                        validity=np.asarray(
                            (
                                int(bool(lidar_points.size)),
                                int(depth_points is not None and bool(depth_points.size)),
                                int(vision_valid),
                            ),
                            dtype=np.uint8,
                        ),
                        ages_s=np.asarray(
                            (
                                0.0 if lidar_points.size else -1.0,
                                depth_age_s if depth_points is not None and depth_points.size else -1.0,
                                semantic.age_s if vision_valid else -1.0,
                            ),
                            dtype=np.float32,
                        ),
                        provenance=np.asarray(
                            (
                                "live_sensor" if lidar_points.size else "unavailable",
                                "live_sensor"
                                if depth_points is not None and depth_points.size
                                else "unavailable",
                                vision_provenance,
                            ),
                            dtype="<U32",
                        ),
                        image_supplied=int(vision_valid and semantic.image_supplied),
                    ),
                    *[item.aged(delta.dt_s) for item in self.sensor_history],
                ][:5]
                self.last_processed_ns = reference_ns
                if output.populated_history < 5 or len(self.sensor_history) < 5:
                    return
                frames = output.metadata["frames"]
                history_timestamps_ns = np.asarray(
                    [int(round(float(item["timestamp_s"]) * _NS_PER_SECOND)) for item in reversed(frames)],
                    dtype=np.int64,
                )
                history_timestamps_ns[-1] = reference_ns
                (
                    history,
                    validity,
                    ages,
                    provenance,
                    supplied,
                ) = _sanitize_history(
                    output.tensor,
                    list(frames),
                    self.sensor_history,
                )
                topics = [
                    str(self.get_parameter("scan_topic").value),
                    str(self.get_parameter("depth_scan_topic").value),
                    str(self.get_parameter("odom_topic").value),
                ]
                if str(self.get_parameter("vision_bev_topic").value):
                    topics.append(str(self.get_parameter("vision_bev_topic").value))
                semantic_grid, semantic_geometry = self._raw_semantic_payload(reference_ns)
                lidar_ranges = np.asarray(
                    message.ranges,
                    dtype=np.float32,
                ).reshape(-1)
                lidar_geometry = np.asarray(
                    (
                        float(message.angle_min),
                        float(message.angle_increment),
                        float(message.range_min),
                        float(message.range_max),
                    ),
                    dtype=np.float64,
                )
                if depth_points is not None and self.latest_depth is not None:
                    raw_depth_ranges = np.asarray(
                        self.latest_depth.ranges,
                        dtype=np.float32,
                    ).reshape(-1)
                    raw_depth_geometry = np.asarray(
                        (
                            float(self.latest_depth.angle_min),
                            float(self.latest_depth.angle_increment),
                            float(self.latest_depth.range_min),
                            float(self.latest_depth.range_max),
                        ),
                        dtype=np.float64,
                    )
                else:
                    raw_depth_ranges = np.empty(0, dtype=np.float32)
                    raw_depth_geometry = np.empty(0, dtype=np.float64)
                receipt = self.store.ingest(
                    RawObservation(
                        timestamp_ns=reference_ns,
                        history_timestamps_ns=history_timestamps_ns,
                        tribev_history=history,
                        sensor_validity=validity,
                        sensor_age_s=ages,
                        sensor_provenance=provenance,
                        vision_image_supplied=supplied,
                        pose_xyyaw=np.asarray(
                            (pose.x_m, pose.y_m, pose.yaw_rad),
                            dtype=np.float64,
                        ),
                        ros_topics=tuple(dict.fromkeys(topics)),
                        lidar_ranges=lidar_ranges,
                        lidar_scan_geometry=lidar_geometry,
                        depth_ranges=raw_depth_ranges,
                        depth_scan_geometry=raw_depth_geometry,
                        semantic_grid=semantic_grid,
                        semantic_grid_geometry=semantic_geometry,
                    )
                )
                if receipt["promoted_paths"]:
                    self.get_logger().info(
                        "promoted strict pseudo-label episode(s): "
                        + ",".join(str(path) for path in receipt["promoted_paths"])
                    )
            except Exception as exc:
                self.get_logger().error(
                    f"read-only staging skipped frame: {exc.__class__.__name__}:{str(exc)[:180]}"
                )

    rclpy.init()
    node = X5TriBEVReadonlyCollector()
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
