#!/usr/bin/env python3
"""Fail-closed LD14 self-return filter for the finals sensor chain.

The pure validation and filtering functions intentionally avoid ROS imports so
they can be unit-tested off-board.  ``main`` adds the ROS2 subscription,
timestamped TF lookup, and publisher.  No actuator topic, service, or action is
created by this process.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

BODY_CONTOUR_SCHEMA: Final[str] = "xrd-lidar-body-contour-v2"
BODY_CONTOUR_PATH: Final[str] = "/home/rdk/rb_voe/lidar_body_contour.v1.json"
BODY_CONTOUR_MEASUREMENT_PATH: Final[str] = "/home/rdk/rb_voe/lidar_body_contour_measurement.v1.txt"
COLLISION_MONITOR_CONFIG_PATH: Final[str] = (
    "/home/rdk/ros2_ws/src/my_robot_navigation/config/collision_monitor.yaml"
)
NAV2_PARAMS_CONFIG_PATH: Final[str] = "/home/rdk/ros2_ws/src/my_robot_navigation/config/nav2_params.yaml"
FROZEN_COLLISION_MONITOR_SHA256: Final[str] = (
    "94ba1f6a5e7c543694086cf13d0801955788268822b2dd683e4f7b72d74300c1"
)
FROZEN_NAV2_PARAMS_SHA256: Final[str] = "568fd06dda966c5119c540281db201d0f38548d79ca3d9cfb55f60d081a394d2"
FROZEN_SAFETY_CONFIG_SUMMARY_SHA256: Final[str] = (
    "dfa93a9acf13b69213282143a37acb044778bc2b81859ec994e44230c0e5e15c"
)
FROZEN_BODY_STOP_RADIUS_M: Final[float] = 0.34
FROZEN_NAV2_ROBOT_RADIUS_M: Final[float] = 0.30
EXPECTED_SCAN_FRAME_ID: Final[str] = "laser_link"
MIN_CONTOUR_AREA_RATIO_OF_NAV2_DISK: Final[float] = 0.70
MAX_CONTOUR_BYTES: Final[int] = 65_536
MAX_SUPPORTING_FILE_BYTES: Final[int] = 1_048_576
TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")

CONTOUR_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "verification_status",
        "source",
        "frame_id",
        "measurement_id",
        "measured_at_utc",
        "measurement_uncertainty_m",
        "measurement_attachment_path",
        "measurement_attachment_sha256",
        "collision_monitor_config_path",
        "collision_monitor_config_sha256",
        "body_stop_radius_m",
        "nav2_params_config_path",
        "nav2_params_config_sha256",
        "nav2_robot_radius_m",
        "safety_config_summary_sha256",
        "polygon_xy_m",
    }
)


class ContourValidationError(ValueError):
    """The frozen contour or one of its bound supporting files is invalid."""


class FilterInputError(ValueError):
    """A scan or transform is unsafe to filter."""


@dataclass(frozen=True)
class FrozenBodyContour:
    polygon_xy_m: tuple[tuple[float, float], ...]
    measurement_uncertainty_m: float
    artifact_sha256: str
    measurement_attachment_sha256: str
    frame_id: str = "base_footprint"


@dataclass(frozen=True)
class RigidTransform:
    translation_xyz: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]


def _regular_file_bytes(path: Path, *, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ContourValidationError(f"cannot open frozen evidence: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_size > maximum_bytes:
            raise ContourValidationError(f"frozen evidence is not a bounded regular file: {path}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(65_536, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise ContourValidationError(f"frozen evidence exceeds size limit: {path}")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ContourValidationError(f"frozen evidence changed while reading: {path}")
    data = b"".join(chunks)
    if len(data) != before.st_size:
        raise ContourValidationError(f"frozen evidence read was incomplete: {path}")
    return data


def _strict_json_object(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ContourValidationError("body contour is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ContourValidationError("body contour must be a JSON object")
    return value


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _signed_area_twice(points: Sequence[tuple[float, float]]) -> float:
    return sum(
        first[0] * second[1] - second[0] * first[1]
        for first, second in zip(points, points[1:] + points[:1], strict=True)
    )


def _orientation(
    first: tuple[float, float],
    second: tuple[float, float],
    third: tuple[float, float],
) -> float:
    return (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (third[0] - first[0])


def _point_on_segment(
    point: tuple[float, float],
    first: tuple[float, float],
    second: tuple[float, float],
    *,
    tolerance: float = 1e-9,
) -> bool:
    return abs(_orientation(first, second, point)) <= tolerance and (
        min(first[0], second[0]) - tolerance <= point[0] <= max(first[0], second[0]) + tolerance
        and min(first[1], second[1]) - tolerance <= point[1] <= max(first[1], second[1]) + tolerance
    )


def _segments_intersect(
    first_a: tuple[float, float],
    first_b: tuple[float, float],
    second_a: tuple[float, float],
    second_b: tuple[float, float],
) -> bool:
    orientations = (
        _orientation(first_a, first_b, second_a),
        _orientation(first_a, first_b, second_b),
        _orientation(second_a, second_b, first_a),
        _orientation(second_a, second_b, first_b),
    )
    if (orientations[0] > 0.0) != (orientations[1] > 0.0) and (orientations[2] > 0.0) != (
        orientations[3] > 0.0
    ):
        return True
    return any(
        abs(orientation) <= 1e-12 and _point_on_segment(point, segment_a, segment_b)
        for orientation, point, segment_a, segment_b in (
            (orientations[0], second_a, first_a, first_b),
            (orientations[1], second_b, first_a, first_b),
            (orientations[2], first_a, second_a, second_b),
            (orientations[3], first_b, second_a, second_b),
        )
    )


def _simple_polygon(points: Sequence[tuple[float, float]]) -> bool:
    count = len(points)
    for first_index in range(count):
        first_next = (first_index + 1) % count
        for second_index in range(first_index + 1, count):
            second_next = (second_index + 1) % count
            if first_index in (second_index, second_next) or first_next in (
                second_index,
                second_next,
            ):
                continue
            if _segments_intersect(
                points[first_index],
                points[first_next],
                points[second_index],
                points[second_next],
            ):
                return False
    return True


def point_inside_polygon(
    x: float,
    y: float,
    polygon: Sequence[tuple[float, float]],
) -> bool:
    """Return true for points in the polygon or on its boundary."""

    if not math.isfinite(x) or not math.isfinite(y) or len(polygon) < 3:
        return False
    point = (x, y)
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if _point_on_segment(point, previous, current):
            return True
        if (previous[1] > y) != (current[1] > y):
            crossing_x = previous[0] + (y - previous[1]) * (current[0] - previous[0]) / (
                current[1] - previous[1]
            )
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


def _point_segment_distance(
    x: float,
    y: float,
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    dx = second[0] - first[0]
    dy = second[1] - first[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-24:
        return math.hypot(x - first[0], y - first[1])
    projection = ((x - first[0]) * dx + (y - first[1]) * dy) / length_squared
    projection = min(1.0, max(0.0, projection))
    return math.hypot(x - (first[0] + projection * dx), y - (first[1] + projection * dy))


def point_inside_contour_envelope(
    x: float,
    y: float,
    contour: FrozenBodyContour,
) -> bool:
    """Test the measured polygon plus its explicit uncertainty margin."""

    polygon = contour.polygon_xy_m
    if point_inside_polygon(x, y, polygon):
        return True
    margin = contour.measurement_uncertainty_m
    if margin <= 0.0:
        return False
    return any(
        _point_segment_distance(x, y, polygon[index - 1], polygon[index]) <= margin + 1e-12
        for index in range(len(polygon))
    )


def _validated_polygon(value: Any, uncertainty: float) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, list) or not 3 <= len(value) <= 32:
        raise ContourValidationError("body contour polygon vertex count is invalid")
    points: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise ContourValidationError("body contour vertex is invalid")
        x = _finite_number(item[0])
        y = _finite_number(item[1])
        if x is None or y is None:
            raise ContourValidationError("body contour vertex is non-finite")
        points.append((round(x, 6), round(y, 6)))
    polygon = tuple(points)
    if len(set(polygon)) != len(polygon) or not _simple_polygon(polygon):
        raise ContourValidationError("body contour polygon is duplicate or self-intersecting")
    if not point_inside_polygon(0.0, 0.0, polygon):
        raise ContourValidationError("body contour does not contain base origin")
    area = abs(_signed_area_twice(polygon)) / 2.0
    maximum_radius = max(math.hypot(x, y) for x, y in polygon)
    minimum_area = math.pi * FROZEN_NAV2_ROBOT_RADIUS_M**2 * MIN_CONTOUR_AREA_RATIO_OF_NAV2_DISK
    maximum_area = math.pi * (FROZEN_BODY_STOP_RADIUS_M - uncertainty) ** 2
    if maximum_radius + uncertainty > FROZEN_BODY_STOP_RADIUS_M + 1e-12:
        raise ContourValidationError("body contour uncertainty envelope exceeds BodyStop")
    if not minimum_area <= area <= maximum_area + 1e-12:
        raise ContourValidationError("body contour area is outside frozen safety bounds")
    return polygon


def _extract_safety_radii(collision_data: bytes, nav2_data: bytes) -> tuple[float, list[float]]:
    try:
        collision_text = collision_data.decode("utf-8", errors="strict")
        nav2_text = nav2_data.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ContourValidationError("frozen safety config is not UTF-8") from exc
    body = re.search(
        r"(?ms)^    BodyStop:\s*$\n(?P<body>(?:^      [^\n]*\n?)*)",
        collision_text,
    )
    radius = re.search(r"(?m)^      radius:\s*([0-9]+(?:\.[0-9]+)?)\s*$", body["body"]) if body else None
    nav2_radii = [
        float(item)
        for item in re.findall(
            r"(?m)^\s{6}robot_radius:\s*([0-9]+(?:\.[0-9]+)?)\s*$",
            nav2_text,
        )
    ]
    if radius is None:
        raise ContourValidationError("BodyStop radius is missing")
    return float(radius.group(1)), nav2_radii


def load_frozen_body_contour(
    contour_path: Path = Path(BODY_CONTOUR_PATH),
    *,
    measurement_attachment_path: Path = Path(BODY_CONTOUR_MEASUREMENT_PATH),
    collision_monitor_path: Path = Path(COLLISION_MONITOR_CONFIG_PATH),
    nav2_params_path: Path = Path(NAV2_PARAMS_CONFIG_PATH),
) -> FrozenBodyContour:
    """Load an exact frozen v2 contour and all independently bound evidence."""

    contour_data = _regular_file_bytes(contour_path, maximum_bytes=MAX_CONTOUR_BYTES)
    raw = _strict_json_object(contour_data)
    if set(raw) != CONTOUR_KEYS:
        raise ContourValidationError("body contour keys do not match the frozen v2 contract")
    uncertainty = _finite_number(raw.get("measurement_uncertainty_m"))
    measured_at = raw.get("measured_at_utc")
    measurement_id = raw.get("measurement_id")
    if not (
        raw.get("schema_version") == BODY_CONTOUR_SCHEMA
        and raw.get("verification_status") == "MEASURED_AND_FROZEN"
        and raw.get("source") == "physical_measurement"
        and raw.get("frame_id") == "base_footprint"
        and isinstance(measurement_id, str)
        and TOKEN_RE.fullmatch(measurement_id) is not None
        and isinstance(measured_at, str)
        and len(measured_at) >= 20
        and measured_at.endswith("Z")
        and uncertainty is not None
        and 0.0 <= uncertainty <= 0.05
    ):
        raise ContourValidationError("body contour metadata is not frozen v2")
    polygon = _validated_polygon(raw.get("polygon_xy_m"), uncertainty)

    attachment_data = _regular_file_bytes(
        measurement_attachment_path,
        maximum_bytes=MAX_SUPPORTING_FILE_BYTES,
    )
    attachment_sha256 = hashlib.sha256(attachment_data).hexdigest()
    if not (
        raw.get("measurement_attachment_path") == BODY_CONTOUR_MEASUREMENT_PATH
        and SHA256_RE.fullmatch(str(raw.get("measurement_attachment_sha256") or ""))
        and raw.get("measurement_attachment_sha256") == attachment_sha256
    ):
        raise ContourValidationError("body contour measurement attachment binding is invalid")

    collision_data = _regular_file_bytes(
        collision_monitor_path,
        maximum_bytes=MAX_SUPPORTING_FILE_BYTES,
    )
    nav2_data = _regular_file_bytes(
        nav2_params_path,
        maximum_bytes=MAX_SUPPORTING_FILE_BYTES,
    )
    collision_sha256 = hashlib.sha256(collision_data).hexdigest()
    nav2_sha256 = hashlib.sha256(nav2_data).hexdigest()
    body_stop_radius, nav2_radii = _extract_safety_radii(collision_data, nav2_data)
    if not (
        raw.get("collision_monitor_config_path") == COLLISION_MONITOR_CONFIG_PATH
        and raw.get("collision_monitor_config_sha256") == collision_sha256
        and collision_sha256 == FROZEN_COLLISION_MONITOR_SHA256
        and _finite_number(raw.get("body_stop_radius_m")) == FROZEN_BODY_STOP_RADIUS_M
        and body_stop_radius == FROZEN_BODY_STOP_RADIUS_M
        and raw.get("nav2_params_config_path") == NAV2_PARAMS_CONFIG_PATH
        and raw.get("nav2_params_config_sha256") == nav2_sha256
        and nav2_sha256 == FROZEN_NAV2_PARAMS_SHA256
        and _finite_number(raw.get("nav2_robot_radius_m")) == FROZEN_NAV2_ROBOT_RADIUS_M
        and len(nav2_radii) == 2
        and all(value == FROZEN_NAV2_ROBOT_RADIUS_M for value in nav2_radii)
        and raw.get("safety_config_summary_sha256") == FROZEN_SAFETY_CONFIG_SUMMARY_SHA256
    ):
        raise ContourValidationError("body contour safety-config binding is invalid")
    return FrozenBodyContour(
        polygon_xy_m=polygon,
        measurement_uncertainty_m=uncertainty,
        artifact_sha256=hashlib.sha256(contour_data).hexdigest(),
        measurement_attachment_sha256=attachment_sha256,
    )


def _validated_transform(transform: RigidTransform) -> RigidTransform:
    values = (*transform.translation_xyz, *transform.rotation_xyzw)
    if any(not math.isfinite(value) for value in values):
        raise FilterInputError("scan transform contains non-finite values")
    norm = math.sqrt(sum(value * value for value in transform.rotation_xyzw))
    if abs(norm - 1.0) > 1e-3:
        raise FilterInputError("scan transform quaternion is not normalized")
    return transform


def _rotate_point(
    point: tuple[float, float, float],
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    x, y, z = point
    qx, qy, qz, qw = quaternion
    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)
    return (
        x + qw * tx + (qy * tz - qz * ty),
        y + qw * ty + (qz * tx - qx * tz),
        z + qw * tz + (qx * ty - qy * tx),
    )


def filter_ranges(
    ranges: Iterable[float],
    *,
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    transform: RigidTransform,
    contour: FrozenBodyContour,
) -> tuple[list[float], int]:
    """Replace only valid returns inside the base-frame contour with infinity."""

    source = list(ranges)
    metadata = (angle_min, angle_increment, range_min, range_max)
    if not source or any(not math.isfinite(value) for value in metadata):
        raise FilterInputError("LaserScan geometry is empty or non-finite")
    if angle_increment == 0.0 or range_min < 0.0 or range_max <= range_min:
        raise FilterInputError("LaserScan range or angle bounds are invalid")
    transform = _validated_transform(transform)
    tx, ty, tz = transform.translation_xyz
    output = list(source)
    removed = 0
    for index, item in enumerate(source):
        try:
            distance = float(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise FilterInputError("LaserScan range is not numeric") from exc
        if not math.isfinite(distance) or not range_min <= distance <= range_max:
            continue
        angle = angle_min + index * angle_increment
        rx, ry, _ = _rotate_point(
            (distance * math.cos(angle), distance * math.sin(angle), 0.0),
            transform.rotation_xyzw,
        )
        if point_inside_contour_envelope(rx + tx, ry + ty, contour):
            output[index] = math.inf
            removed += 1
    return output, removed


def _validate_laser_scan_message(message: Any) -> None:
    try:
        ranges = list(message.ranges)
        intensities = list(message.intensities)
        frame_id = str(message.header.frame_id)
        stamp_sec = int(message.header.stamp.sec)
        stamp_nanosec = int(message.header.stamp.nanosec)
        angle_min = float(message.angle_min)
        angle_max = float(message.angle_max)
        angle_increment = float(message.angle_increment)
        time_increment = float(message.time_increment)
        scan_time = float(message.scan_time)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise FilterInputError("LaserScan message fields are invalid") from exc
    geometry = (angle_min, angle_max, angle_increment, time_increment, scan_time)
    if (
        not ranges
        or any(not math.isfinite(value) for value in geometry)
        or frame_id != EXPECTED_SCAN_FRAME_ID
        or stamp_sec < 0
        or not 0 <= stamp_nanosec < 1_000_000_000
        or (stamp_sec == 0 and stamp_nanosec == 0)
        or angle_increment == 0.0
        or time_increment < 0.0
        or scan_time <= 0.0
        or (intensities and len(intensities) != len(ranges))
    ):
        raise FilterInputError("LaserScan metadata is outside the frozen finals contract")
    expected_angle_max = angle_min + (len(ranges) - 1) * angle_increment
    angular_tolerance = max(1e-6, abs(angle_increment) * 1e-3)
    if abs(angle_max - expected_angle_max) > angular_tolerance:
        raise FilterInputError("LaserScan angular geometry is inconsistent")
    if time_increment > 0.0:
        sweep_duration = time_increment * max(0, len(ranges) - 1)
        if scan_time + max(1e-6, scan_time * 1e-3) < sweep_duration:
            raise FilterInputError("LaserScan timing geometry is inconsistent")


def filter_laser_scan_message(
    message: Any,
    *,
    transform: RigidTransform,
    contour: FrozenBodyContour,
) -> tuple[Any, int]:
    """Deep-copy a LaserScan-like object and modify only its ``ranges`` field."""

    try:
        _validate_laser_scan_message(message)
        filtered_ranges, removed = filter_ranges(
            message.ranges,
            angle_min=float(message.angle_min),
            angle_increment=float(message.angle_increment),
            range_min=float(message.range_min),
            range_max=float(message.range_max),
            transform=transform,
            contour=contour,
        )
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, FilterInputError):
            raise
        raise FilterInputError("LaserScan message fields are invalid") from exc
    filtered = copy.deepcopy(message)
    filtered.ranges = filtered_ranges
    return filtered, removed


def _rigid_transform_from_ros(message: Any) -> RigidTransform:
    try:
        translation = message.transform.translation
        rotation = message.transform.rotation
        return RigidTransform(
            translation_xyz=(
                float(translation.x),
                float(translation.y),
                float(translation.z),
            ),
            rotation_xyzw=(
                float(rotation.x),
                float(rotation.y),
                float(rotation.z),
                float(rotation.w),
            ),
        )
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise FilterInputError("TF transform fields are invalid") from exc


def main(args: Sequence[str] | None = None) -> None:
    import rclpy
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
        qos_profile_sensor_data,
    )
    from rclpy.time import Time
    from sensor_msgs.msg import LaserScan
    from tf2_ros import Buffer, TransformException, TransformListener

    class ScanSelfFilterNode(Node):
        def __init__(self) -> None:
            super().__init__("scan_self_filter")
            self.declare_parameter("contour_path", BODY_CONTOUR_PATH)
            self.declare_parameter("input_topic", "/scan_raw")
            self.declare_parameter("output_topic", "/scan")
            self.declare_parameter("target_frame", "base_footprint")
            self.declare_parameter("transform_timeout_s", 0.05)
            contour_path = str(self.get_parameter("contour_path").value)
            input_topic = str(self.get_parameter("input_topic").value)
            output_topic = str(self.get_parameter("output_topic").value)
            self._target_frame = str(self.get_parameter("target_frame").value)
            timeout_s = float(self.get_parameter("transform_timeout_s").value)
            if (
                contour_path != BODY_CONTOUR_PATH
                or input_topic != "/scan_raw"
                or output_topic != "/scan"
                or self._target_frame != "base_footprint"
                or not math.isfinite(timeout_s)
                or not 0.0 < timeout_s <= 0.5
            ):
                raise ContourValidationError("finals self-filter parameters are not frozen")
            self._transform_timeout = Duration(seconds=timeout_s)
            self._contour = load_frozen_body_contour(Path(contour_path))
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)
            # A RELIABLE offer is compatible with both Nav2/SLAM reliable
            # subscriptions and best-effort diagnostic subscribers. The raw
            # hardware input remains sensor-data BEST_EFFORT.
            filtered_scan_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            self._publisher = self.create_publisher(
                LaserScan,
                output_topic,
                filtered_scan_qos,
            )
            self._subscription = self.create_subscription(
                LaserScan,
                input_topic,
                self._on_scan,
                qos_profile_sensor_data,
            )
            self._drop_count = 0
            self.get_logger().info(
                f"scan self-filter ready: contour_sha256={self._contour.artifact_sha256} /scan_raw -> /scan"
            )

        def _drop(self, reason: str) -> None:
            self._drop_count += 1
            if self._drop_count == 1 or self._drop_count % 100 == 0:
                self.get_logger().error(f"fail-closed scan drop ({self._drop_count}): {reason}")

        def _on_scan(self, message: LaserScan) -> None:
            frame_id = str(message.header.frame_id)
            stamp = message.header.stamp
            if frame_id != EXPECTED_SCAN_FRAME_ID:
                self._drop("LaserScan frame is not frozen to laser_link")
                return
            if int(stamp.sec) == 0 and int(stamp.nanosec) == 0:
                self._drop("missing acquisition stamp")
                return
            try:
                transform_message = self._tf_buffer.lookup_transform(
                    self._target_frame,
                    frame_id,
                    Time.from_msg(stamp),
                    timeout=self._transform_timeout,
                )
                filtered, _removed = filter_laser_scan_message(
                    message,
                    transform=_rigid_transform_from_ros(transform_message),
                    contour=self._contour,
                )
            except (TransformException, FilterInputError, ValueError) as exc:
                self._drop(str(exc))
                return
            self._publisher.publish(filtered)

    rclpy.init(args=args)
    node: ScanSelfFilterNode | None = None
    try:
        node = ScanSelfFilterNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
