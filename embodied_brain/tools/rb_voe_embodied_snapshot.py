#!/usr/bin/env python3
"""Strict read-only R2-PREP snapshot collector for the embodied RDK X5.

The pure builders and validators in this module use only the Python standard
library. ROS imports are deliberately confined to ``collect_ros_observation``
so the contract can be tested on a PC with no ROS installation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import time
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, NamedTuple

SCHEMA_VERSION: Final[str] = "xrd-rb-voe-embodied-runtime-snapshot-v1"
SOURCE_BINDING_SCHEMA: Final[str] = "xrd-rb-voe-live-source-binding-v1"
SUBSYSTEM: Final[str] = "embodied_x5"
SERVICE_NAME: Final[str] = "embodied_brain.service"
EXPECTED_HOSTNAME: Final[str] = "embodied-x5"
EXPECTED_WLAN_MAC: Final[str] = "40:55:48:a5:41:92"
FROZEN_PROFILE_SHA256: Final[str] = "fc16687488b548b1e7779a433ef1640777bb91bdb18e39de4f4e723e0a4940cc"

LOCALIZATION_ONLINE_SLAM: Final[str] = "ONLINE_SLAM"
LOCALIZATION_SAVED_MAP_AMCL: Final[str] = "SAVED_MAP_AMCL"
LOCALIZATION_MODES: Final[tuple[str, ...]] = (
    LOCALIZATION_ONLINE_SLAM,
    LOCALIZATION_SAVED_MAP_AMCL,
)
BODY_CONTOUR_SCHEMA: Final[str] = "xrd-lidar-body-contour-v2"
BODY_CONTOUR_ARTIFACT_PATH: Final[str] = "/home/rdk/rb_voe/lidar_body_contour.v1.json"
BODY_CONTOUR_PATH: Final[Path] = Path(BODY_CONTOUR_ARTIFACT_PATH)
BODY_CONTOUR_MEASUREMENT_PATH: Final[str] = "/home/rdk/rb_voe/lidar_body_contour_measurement.v1.txt"
COLLISION_MONITOR_CONFIG_PATH: Final[str] = (
    "/home/rdk/ros2_ws/src/my_robot_navigation/config/collision_monitor.yaml"
)
NAV2_PARAMS_CONFIG_PATH: Final[str] = "/home/rdk/ros2_ws/src/my_robot_navigation/config/nav2_params.yaml"
FROZEN_COLLISION_MONITOR_SHA256: Final[str] = (
    "94ba1f6a5e7c543694086cf13d0801955788268822b2dd683e4f7b72d74300c1"
)
FROZEN_NAV2_PARAMS_SHA256: Final[str] = "568fd06dda966c5119c540281db201d0f38548d79ca3d9cfb55f60d081a394d2"
FROZEN_SCAN_SELF_FILTER_SHA256: Final[str] = (
    "c8f9a8d0f116127a832592bd31e6fc95da7f4600f91883d980f9aa6a30adaea3"
)
FROZEN_SENSORS_LAUNCH_SHA256: Final[str] = "5263e7bef87579e7fa0b548cd9f0a8853793c29f345a56ae6f43a976e070ccf9"
FROZEN_LIDAR_LAUNCH_SHA256: Final[str] = "4b9bf4dbc186a466894206963f4b59362d98043f91cdeacd747042bc7fe25dce"
FROZEN_FULL_LAUNCH_SHA256: Final[str] = "62e1a92378bd3f7f7cff9eefbfcb770feff595692cae2aed7192932848ee7458"
FROZEN_SYSTEMD_UNIT_SHA256: Final[str] = "a2024860da4c3831ae136f23e42a8474e4750570720fc457d8a75c200225fdea"
FROZEN_BODY_STOP_RADIUS_M: Final[float] = 0.34
FROZEN_NAV2_ROBOT_RADIUS_M: Final[float] = 0.30
MIN_CONTOUR_AREA_RATIO_OF_NAV2_DISK: Final[float] = 0.70
EXPECTED_SCAN_RAW_PUBLISHER: Final[str] = "ld14_lidar"
EXPECTED_SCAN_FILTER_NODE: Final[str] = "scan_self_filter"
EXPECTED_SCAN_FRAME_ID: Final[str] = "laser_link"
SCAN_FILTER_SOURCE_PATH: Final[str] = "/home/rdk/ros2_ws/src/my_robot_drivers/scripts/scan_self_filter.py"
SCAN_FILTER_INSTALLED_PATH: Final[str] = (
    "/home/rdk/ros2_ws/install/my_robot_drivers/lib/my_robot_drivers/scan_self_filter"
)
MIN_SCAN_VALID_POINTS: Final[int] = 16
MIN_SCAN_COVERAGE_RAD: Final[float] = 5.5
MAX_SCAN_PAIR_AGE_S: Final[float] = 1.0
MAX_SCAN_PAIR_FUTURE_S: Final[float] = 0.1
MAX_F407_SAFETY_STATE_AGE_S: Final[float] = 1.5
SCAN_PAIR_CACHE_SIZE: Final[int] = 64

CAPABILITIES: Final[tuple[str, ...]] = (
    "embodied_x5.geometry.self_filtered_live",
    "embodied_x5.localization.online_slam_live",
    "embodied_x5.state_estimation.ekf_live",
    "embodied_x5.f407.hardware_safety_readonly",
    "embodied_x5.collision_monitor.veto_chain",
    "embodied_x5.lab_fsd.shadow_risk",
    "embodied_x5.tiny_occ_risk.bpu_actual",
    "embodied_x5.mppi.bpu_proposed_only_actual",
)
CAPABILITY_BACKENDS: Final[Mapping[str, str]] = {
    "embodied_x5.geometry.self_filtered_live": (
        "ros2.ld14.scan_raw_to_self_filter_to_scan.astra_scan_depth.live"
    ),
    "embodied_x5.localization.online_slam_live": ("ros2.slam_toolbox.map_and_map_to_odom.fresh_live"),
    "embodied_x5.state_estimation.ekf_live": (
        "ros2.f407_wheel_odom_plus_imu.robot_localization_ekf.odom_live"
    ),
    "embodied_x5.f407.hardware_safety_readonly": ("f407.0xaa55.safety_state_and_firmware_info.ros2_readonly"),
    "embodied_x5.collision_monitor.veto_chain": (
        "ros2.collision_monitor.scan_plus_scan_depth.cmd_vel_to_cmd_vel_safe_to_serial_f407.veto"
    ),
    "embodied_x5.lab_fsd.shadow_risk": ("ros2.lab_fsd.live_inputs.future_risk.safety_gate.shadow_only"),
    "embodied_x5.tiny_occ_risk.bpu_actual": ("hobot_dnn.Bayes-e.INT8.tiny_occ_risk.forward_actual"),
    "embodied_x5.mppi.bpu_proposed_only_actual": ("hobot_dnn.Bayes-e.INT8.mppi_cost.proposed_only_actual"),
}

CRITICAL_SENSOR_TOPICS: Final[tuple[str, ...]] = (
    "/scan_raw",
    "/scan",
    "/scan_depth",
    "/wheel_odom",
    "/imu",
    "/odom",
)
SUBSCRIBED_TOPICS: Final[tuple[str, ...]] = (
    "/scan_raw",
    "/scan",
    "/scan_depth",
    "/wheel_odom",
    "/imu",
    "/odom",
    "/map",
    "/amcl_pose",
    "/global_costmap/costmap",
    "/collision_monitor/state",
    "/f407/firmware_identity_valid",
    "/f407/estop_latched",
    "/f407/cmd_vel_expired",
    "/f407/firmware_info",
    "/diagnostics",
    "/lab_fsd/fsd_v3_status",
    "/lab_fsd/input_status",
    "/lab_fsd/safety_gate",
    "/mppi/stats",
    "/mppi/cmd_vel_proposed",
)
GRAPH_TOPICS: Final[tuple[str, ...]] = tuple(
    sorted(
        set(SUBSCRIBED_TOPICS)
        | {
            "/cmd_vel",
            "/cmd_vel_safe",
            "/mppi/cmd_vel_proposed",
            "/tf",
            "/tf_static",
        }
    )
)
LIFECYCLE_NODES: Final[tuple[str, ...]] = (
    "slam_toolbox",
    "amcl",
    "map_server",
    "collision_monitor",
)

SNAPSHOT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "subsystem",
        "ready",
        "reason_code",
        "reason_codes",
        "observed_at_ms",
        "run_id",
        "run_nonce_sha256",
        "run_binding_sha256",
        "release_id",
        "profile_sha256",
        "device_id",
        "hostname",
        "machine_id_sha256",
        "boot_id",
        "session_id",
        "service_invocation_id",
        "wlan_mac",
        "artifacts",
        "sensors",
        "localization",
        "f407",
        "command_topology",
        "collision_monitor",
        "lab_fsd",
        "tiny_occ_risk",
        "mppi",
        "physical_navigation",
        "capabilities",
        "probe",
        "snapshot_sha256",
    }
)

TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
BOOT_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
INVOCATION_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{32}$")
MAC_RE: Final[re.Pattern[str]] = re.compile(r"^(?:[0-9a-f]{2}:){5}[0-9a-f]{2}$")


class ArtifactSpec(NamedTuple):
    path: str
    required: bool
    expected_sha256: str | None = None


class InstalledArtifactSpec(NamedTuple):
    source_path: str
    installed_path: str
    expected_sha256: str


ARTIFACT_SPECS: Final[Mapping[str, ArtifactSpec]] = {
    "body_contour": ArtifactSpec(BODY_CONTOUR_ARTIFACT_PATH, True),
    "body_contour_measurement": ArtifactSpec(BODY_CONTOUR_MEASUREMENT_PATH, True),
    "collector_script": ArtifactSpec("/home/rdk/tools/rb_voe_embodied_snapshot.py", True),
    "full_launch": ArtifactSpec(
        "/home/rdk/ros2_ws/src/my_robot_bringup/launch/full.launch.py",
        True,
        FROZEN_FULL_LAUNCH_SHA256,
    ),
    "nav2_params": ArtifactSpec(NAV2_PARAMS_CONFIG_PATH, True, FROZEN_NAV2_PARAMS_SHA256),
    "collision_monitor_config": ArtifactSpec(
        COLLISION_MONITOR_CONFIG_PATH, True, FROZEN_COLLISION_MONITOR_SHA256
    ),
    "lab_fsd_config": ArtifactSpec(
        "/home/rdk/ros2_ws/src/my_robot_navigation/config/lab_fsd_shadow.yaml", True
    ),
    "ekf_config": ArtifactSpec("/home/rdk/ros2_ws/src/my_robot_navigation/config/ekf_odom.yaml", True),
    "saved_map_yaml": ArtifactSpec(
        "/home/rdk/maps/lab_final_20260708_210920.yaml",
        True,
        "73e915619d9c86059a1e99befd5d878875965369a7f8ce0582890d9a20a9df9d",
    ),
    "saved_map_pgm": ArtifactSpec(
        "/home/rdk/maps/lab_final_20260708_210920.pgm",
        True,
        "868451418dbf65c86089e831cf5bd8f5c821b5ed860992d7899273ad93dde5ac",
    ),
    "tiny_occ_risk_bin": ArtifactSpec(
        "/home/rdk/models/lab_fsd/lab_fsd_tiny_occ_risk.bin",
        True,
        "3b1a96483351f72746fdcacfb179b69f4527076046e5dd73d5bcae7688d99c90",
    ),
    "mppi_cost_bin": ArtifactSpec(
        "/home/rdk/bpu_models/cost_mlp.bin",
        True,
        "fe54f08d12285cf66c37ee7168b51a6762bb086b30a681a12f18374d8eea853d",
    ),
    "lab_anomaly_bin": ArtifactSpec(
        "/home/rdk/models/lab_fsd/lab_anomaly_autoencoder.bin",
        False,
        "1045be38ff947ad3c97c365416170970f59735504a1f38663bd8cce8d112ad7f",
    ),
    "f407_expected_hex": ArtifactSpec(
        "/home/rdk/stm32_f407/Objects/a.hex",
        False,
    ),
}

SCAN_FILTER_ARTIFACT_SPECS: Final[Mapping[str, ArtifactSpec]] = {
    "scan_self_filter": ArtifactSpec(
        SCAN_FILTER_SOURCE_PATH,
        True,
        FROZEN_SCAN_SELF_FILTER_SHA256,
    ),
    "sensors_launch": ArtifactSpec(
        "/home/rdk/ros2_ws/src/my_robot_drivers/launch/sensors.launch.py",
        True,
        FROZEN_SENSORS_LAUNCH_SHA256,
    ),
    "lidar_launch": ArtifactSpec(
        "/home/rdk/ros2_ws/src/my_robot_drivers/launch/lidar.launch.py",
        True,
        FROZEN_LIDAR_LAUNCH_SHA256,
    ),
}

INSTALLED_RELEASE_ARTIFACT_SPECS: Final[Mapping[str, InstalledArtifactSpec]] = {
    "sensors_launch": InstalledArtifactSpec(
        "/home/rdk/ros2_ws/src/my_robot_drivers/launch/sensors.launch.py",
        "/home/rdk/ros2_ws/install/my_robot_drivers/share/my_robot_drivers/launch/sensors.launch.py",
        FROZEN_SENSORS_LAUNCH_SHA256,
    ),
    "lidar_launch": InstalledArtifactSpec(
        "/home/rdk/ros2_ws/src/my_robot_drivers/launch/lidar.launch.py",
        "/home/rdk/ros2_ws/install/my_robot_drivers/share/my_robot_drivers/launch/lidar.launch.py",
        FROZEN_LIDAR_LAUNCH_SHA256,
    ),
    "full_launch": InstalledArtifactSpec(
        "/home/rdk/ros2_ws/src/my_robot_bringup/launch/full.launch.py",
        "/home/rdk/ros2_ws/install/my_robot_bringup/share/my_robot_bringup/launch/full.launch.py",
        FROZEN_FULL_LAUNCH_SHA256,
    ),
    "systemd_unit": InstalledArtifactSpec(
        "/home/rdk/ros2_ws/src/my_robot_bringup/config/embodied_brain.service",
        "/etc/systemd/system/embodied_brain.service",
        FROZEN_SYSTEMD_UNIT_SHA256,
    ),
}


def _canonical_primitive(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floats are forbidden in canonical payloads")
        if value == 0.0:
            return 0
        if value.is_integer() and abs(value) <= 2**53 - 1:
            return int(value)
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical mapping keys must be strings")
        return {key: _canonical_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_primitive(item) for item in value]
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return the project-wide canonical JSON representation."""

    return json.dumps(
        _canonical_primitive(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_binding_sha256(*, run_id: str, run_nonce: str, release_id: str, profile_sha256: str) -> str:
    """Match the R2 live-source binding used by the central coordinator."""

    return canonical_sha256(
        {
            "schema_version": SOURCE_BINDING_SCHEMA,
            "subsystem": SUBSYSTEM,
            "run_id": run_id,
            "run_nonce": run_nonce,
            "release_id": release_id,
            "profile_sha256": profile_sha256,
        }
    )


def _validate_binding(run_id: str, run_nonce: str, release_id: str, profile_sha256: str) -> None:
    for name, value in (("run_id", run_id), ("release_id", release_id)):
        if not isinstance(value, str) or TOKEN_RE.fullmatch(value) is None:
            raise ValueError(f"{name} must be a non-empty contract token")
    if not isinstance(run_nonce, str) or not 16 <= len(run_nonce) <= 256:
        raise ValueError("run_nonce must contain 16..256 characters")
    if not isinstance(profile_sha256, str) or SHA256_RE.fullmatch(profile_sha256) is None:
        raise ValueError("profile_sha256 must be a lowercase SHA-256 digest")
    if profile_sha256 != FROZEN_PROFILE_SHA256:
        raise ValueError("profile_sha256 must match the frozen packaged embodied profile")


def _read_text(path: Path, *, maximum_bytes: int = 65536) -> str:
    if str(path).startswith("/dev/"):
        raise ValueError("hardware device paths are outside the read-only evidence surface")
    data = path.read_bytes()
    if len(data) > maximum_bytes:
        raise ValueError(f"read limit exceeded for {path}")
    return data.decode("utf-8", errors="strict").strip()


def _strict_json_object(data: bytes) -> dict[str, Any]:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant: {value}")

    parsed = json.loads(
        data.decode("utf-8", errors="strict"),
        object_pairs_hook=unique_pairs,
        parse_constant=reject_constant,
    )
    if not isinstance(parsed, dict):
        raise ValueError("JSON root must be an object")
    return parsed


def _polygon_signed_area_twice(points: list[list[float]]) -> float:
    return sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )


def _orientation(a: list[float], b: list[float], c: list[float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _point_on_segment(point: list[float], a: list[float], b: list[float]) -> bool:
    epsilon = 1e-12
    return (
        abs(_orientation(a, b, point)) <= epsilon
        and min(a[0], b[0]) - epsilon <= point[0] <= max(a[0], b[0]) + epsilon
        and min(a[1], b[1]) - epsilon <= point[1] <= max(a[1], b[1]) + epsilon
    )


def _segments_intersect(
    first_a: list[float],
    first_b: list[float],
    second_a: list[float],
    second_b: list[float],
) -> bool:
    first_side_a = _orientation(first_a, first_b, second_a)
    first_side_b = _orientation(first_a, first_b, second_b)
    second_side_a = _orientation(second_a, second_b, first_a)
    second_side_b = _orientation(second_a, second_b, first_b)
    epsilon = 1e-12
    if (
        (first_side_a > epsilon and first_side_b < -epsilon)
        or (first_side_a < -epsilon and first_side_b > epsilon)
    ) and (
        (second_side_a > epsilon and second_side_b < -epsilon)
        or (second_side_a < -epsilon and second_side_b > epsilon)
    ):
        return True
    return any(
        (abs(side) <= epsilon and _point_on_segment(point, segment_a, segment_b))
        for side, point, segment_a, segment_b in (
            (first_side_a, second_a, first_a, first_b),
            (first_side_b, second_b, first_a, first_b),
            (second_side_a, first_a, second_a, second_b),
            (second_side_b, first_b, second_a, second_b),
        )
    )


def _polygon_is_simple(points: list[list[float]]) -> bool:
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


def _polygon_contains_origin(points: list[list[float]]) -> bool:
    origin = [0.0, 0.0]
    inside = False
    for index, first in enumerate(points):
        second = points[(index + 1) % len(points)]
        if _point_on_segment(origin, first, second):
            return True
        if (first[1] > 0.0) != (second[1] > 0.0):
            x_intersection = first[0] + (second[0] - first[0]) * (-first[1]) / (second[1] - first[1])
            if x_intersection > 0.0:
                inside = not inside
    return inside


def _normalized_polygon(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or not 3 <= len(value) <= 32:
        return None
    points: list[list[float]] = []
    for raw_point in value:
        if not isinstance(raw_point, list) or len(raw_point) != 2:
            return None
        x = _parse_float(raw_point[0])
        y = _parse_float(raw_point[1])
        if x is None or y is None:
            return None
        points.append([round(x, 6), round(y, 6)])
    if len({(point[0], point[1]) for point in points}) != len(points):
        return None
    if any(points[index] == points[(index + 1) % len(points)] for index in range(len(points))):
        return None
    if not _polygon_is_simple(points) or not _polygon_contains_origin(points):
        return None
    return points


def _body_contour_geometry(
    polygon_value: Any,
    uncertainty_value: Any,
) -> tuple[list[list[float]] | None, dict[str, float | bool | None], list[str]]:
    polygon = _normalized_polygon(polygon_value)
    uncertainty = _parse_float(uncertainty_value)
    reasons: list[str] = []
    metrics: dict[str, float | bool | None] = {
        "area_m2": None,
        "maximum_vertex_radius_m": None,
        "uncertainty_envelope_radius_m": None,
        "contains_origin": False,
        "simple_polygon": False,
    }
    if polygon is None:
        reasons.append("BODY_CONTOUR_POLYGON_INVALID")
        return None, metrics, reasons
    area = abs(_polygon_signed_area_twice(polygon)) / 2.0
    maximum_radius = max(math.hypot(point[0], point[1]) for point in polygon)
    metrics.update(
        {
            "area_m2": round(area, 9),
            "maximum_vertex_radius_m": round(maximum_radius, 9),
            "contains_origin": True,
            "simple_polygon": True,
        }
    )
    if uncertainty is None or not 0.0 <= uncertainty <= 0.05:
        reasons.append("BODY_CONTOUR_MEASUREMENT_UNCERTAINTY_INVALID")
        return polygon, metrics, reasons
    envelope_radius = maximum_radius + uncertainty
    metrics["uncertainty_envelope_radius_m"] = round(envelope_radius, 9)
    minimum_area = math.pi * FROZEN_NAV2_ROBOT_RADIUS_M**2 * MIN_CONTOUR_AREA_RATIO_OF_NAV2_DISK
    maximum_area = math.pi * (FROZEN_BODY_STOP_RADIUS_M - uncertainty) ** 2
    if maximum_radius > FROZEN_BODY_STOP_RADIUS_M + 1e-12:
        reasons.append("BODY_CONTOUR_VERTEX_OUTSIDE_BODY_STOP")
    if envelope_radius > FROZEN_BODY_STOP_RADIUS_M + 1e-12:
        reasons.append("BODY_CONTOUR_UNCERTAINTY_ENVELOPE_EXCEEDS_BODY_STOP")
    if not minimum_area <= area <= maximum_area + 1e-12:
        reasons.append("BODY_CONTOUR_AREA_INVALID")
    return polygon, metrics, sorted(set(reasons))


def _safety_config_summary_sha256(
    collision_monitor_sha256: str,
    nav2_params_sha256: str,
) -> str:
    return canonical_sha256(
        {
            "schema_version": "xrd-lidar-body-contour-safety-config-v1",
            "collision_monitor": {
                "path": COLLISION_MONITOR_CONFIG_PATH,
                "sha256": collision_monitor_sha256,
                "BodyStop.radius_m": FROZEN_BODY_STOP_RADIUS_M,
            },
            "nav2": {
                "path": NAV2_PARAMS_CONFIG_PATH,
                "sha256": nav2_params_sha256,
                "robot_radius_m": FROZEN_NAV2_ROBOT_RADIUS_M,
            },
        }
    )


FROZEN_SAFETY_CONFIG_SUMMARY_SHA256: Final[str] = _safety_config_summary_sha256(
    FROZEN_COLLISION_MONITOR_SHA256,
    FROZEN_NAV2_PARAMS_SHA256,
)


def _read_regular_evidence(path: Path, *, maximum_bytes: int = 1024 * 1024) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("evidence path must be a fixed regular file")
    data = path.read_bytes()
    if not data or len(data) > maximum_bytes:
        raise ValueError("evidence file size invalid")
    return data


def _extract_frozen_safety_parameters(
    collision_monitor_data: bytes,
    nav2_params_data: bytes,
) -> tuple[float | None, list[float]]:
    collision_text = collision_monitor_data.decode("utf-8", errors="strict")
    nav2_text = nav2_params_data.decode("utf-8", errors="strict")
    body_stop_match = re.search(
        r"(?ms)^    BodyStop:\s*$\n(?P<body>(?:^      [^\n]*\n?)*)",
        collision_text,
    )
    radius_match = (
        re.search(r"(?m)^      radius:\s*([0-9]+(?:\.[0-9]+)?)\s*$", body_stop_match["body"])
        if body_stop_match
        else None
    )
    body_stop_radius = _parse_float(radius_match.group(1)) if radius_match else None
    nav2_radii = [
        float(value)
        for value in re.findall(
            r"(?m)^\s{6}robot_radius:\s*([0-9]+(?:\.[0-9]+)?)\s*$",
            nav2_text,
        )
    ]
    return body_stop_radius, nav2_radii


def _unavailable_body_contour(reason: str) -> dict[str, Any]:
    return {
        "present": False,
        "valid": False,
        "path": BODY_CONTOUR_ARTIFACT_PATH,
        "sha256": None,
        "schema_version": None,
        "verification_status": None,
        "source": None,
        "frame_id": None,
        "measurement_id": None,
        "measured_at_utc": None,
        "measurement_uncertainty_m": None,
        "measurement_attachment_path": BODY_CONTOUR_MEASUREMENT_PATH,
        "measurement_attachment_sha256": None,
        "collision_monitor_config_path": COLLISION_MONITOR_CONFIG_PATH,
        "collision_monitor_config_sha256": None,
        "body_stop_radius_m": None,
        "nav2_params_config_path": NAV2_PARAMS_CONFIG_PATH,
        "nav2_params_config_sha256": None,
        "nav2_robot_radius_m": None,
        "safety_config_summary_sha256": None,
        "polygon_xy_m": None,
        "geometry": None,
        "reason_codes": [reason],
    }


def _read_verified_body_contour(
    path: Path = BODY_CONTOUR_PATH,
    *,
    collision_monitor_path: Path = Path(COLLISION_MONITOR_CONFIG_PATH),
    nav2_params_path: Path = Path(NAV2_PARAMS_CONFIG_PATH),
    measurement_attachment_path: Path = Path(BODY_CONTOUR_MEASUREMENT_PATH),
) -> dict[str, Any]:
    """Verify geometry against frozen safety configs and independent measurement bytes."""

    if path.is_symlink() or not path.is_file():
        return _unavailable_body_contour("BODY_CONTOUR_VERIFIED_ARTIFACT_MISSING")
    try:
        data = _read_regular_evidence(path, maximum_bytes=65536)
        raw = _strict_json_object(data)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return _unavailable_body_contour("BODY_CONTOUR_VERIFIED_ARTIFACT_INVALID")

    expected_keys = {
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
    uncertainty = _parse_float(raw.get("measurement_uncertainty_m"))
    polygon, geometry, geometry_reasons = _body_contour_geometry(
        raw.get("polygon_xy_m"),
        uncertainty,
    )
    measurement_id = str(raw.get("measurement_id") or "")
    measured_at_utc = str(raw.get("measured_at_utc") or "")
    reasons = list(geometry_reasons)
    attachment_sha256: str | None = None
    collision_sha256: str | None = None
    nav2_sha256: str | None = None
    body_stop_radius: float | None = None
    nav2_radii: list[float] = []
    try:
        attachment_data = _read_regular_evidence(measurement_attachment_path)
        attachment_sha256 = hashlib.sha256(attachment_data).hexdigest()
    except (OSError, ValueError):
        reasons.append("BODY_CONTOUR_MEASUREMENT_ATTACHMENT_INVALID")
    try:
        collision_data = _read_regular_evidence(collision_monitor_path)
        nav2_data = _read_regular_evidence(nav2_params_path)
        collision_sha256 = hashlib.sha256(collision_data).hexdigest()
        nav2_sha256 = hashlib.sha256(nav2_data).hexdigest()
        body_stop_radius, nav2_radii = _extract_frozen_safety_parameters(
            collision_data,
            nav2_data,
        )
    except (OSError, UnicodeError, ValueError):
        reasons.append("BODY_CONTOUR_SAFETY_CONFIG_INVALID")

    structural_valid = (
        set(raw) == expected_keys
        and raw.get("schema_version") == BODY_CONTOUR_SCHEMA
        and raw.get("verification_status") == "MEASURED_AND_FROZEN"
        and raw.get("source") == "physical_measurement"
        and raw.get("frame_id") == "base_footprint"
        and TOKEN_RE.fullmatch(measurement_id) is not None
        and measured_at_utc.endswith("Z")
        and len(measured_at_utc) >= 20
        and uncertainty is not None
        and 0.0 <= uncertainty <= 0.05
        and polygon is not None
    )
    if not structural_valid:
        reasons.append("BODY_CONTOUR_VERIFIED_ARTIFACT_INVALID")
    if not (
        raw.get("measurement_attachment_path") == BODY_CONTOUR_MEASUREMENT_PATH
        and SHA256_RE.fullmatch(str(raw.get("measurement_attachment_sha256") or "")) is not None
        and raw.get("measurement_attachment_sha256") == attachment_sha256
    ):
        reasons.append("BODY_CONTOUR_MEASUREMENT_ATTACHMENT_INVALID")
    safety_summary = (
        _safety_config_summary_sha256(collision_sha256, nav2_sha256)
        if collision_sha256 is not None and nav2_sha256 is not None
        else None
    )
    if not (
        raw.get("collision_monitor_config_path") == COLLISION_MONITOR_CONFIG_PATH
        and raw.get("collision_monitor_config_sha256") == collision_sha256
        and collision_sha256 == FROZEN_COLLISION_MONITOR_SHA256
        and _parse_float(raw.get("body_stop_radius_m")) == FROZEN_BODY_STOP_RADIUS_M
        and body_stop_radius == FROZEN_BODY_STOP_RADIUS_M
        and raw.get("nav2_params_config_path") == NAV2_PARAMS_CONFIG_PATH
        and raw.get("nav2_params_config_sha256") == nav2_sha256
        and nav2_sha256 == FROZEN_NAV2_PARAMS_SHA256
        and _parse_float(raw.get("nav2_robot_radius_m")) == FROZEN_NAV2_ROBOT_RADIUS_M
        and len(nav2_radii) == 2
        and all(radius == FROZEN_NAV2_ROBOT_RADIUS_M for radius in nav2_radii)
        and raw.get("safety_config_summary_sha256") == safety_summary
        and safety_summary == FROZEN_SAFETY_CONFIG_SUMMARY_SHA256
    ):
        reasons.append("BODY_CONTOUR_SAFETY_CONFIG_INVALID")
    reasons = sorted(set(reasons))
    valid = not reasons
    return {
        "present": True,
        "valid": valid,
        "path": path.as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "schema_version": str(raw.get("schema_version") or ""),
        "verification_status": str(raw.get("verification_status") or ""),
        "source": str(raw.get("source") or ""),
        "frame_id": str(raw.get("frame_id") or ""),
        "measurement_id": measurement_id,
        "measured_at_utc": measured_at_utc,
        "measurement_uncertainty_m": _round_number(uncertainty, 6),
        "measurement_attachment_path": str(raw.get("measurement_attachment_path") or ""),
        "measurement_attachment_sha256": attachment_sha256,
        "collision_monitor_config_path": str(raw.get("collision_monitor_config_path") or ""),
        "collision_monitor_config_sha256": collision_sha256,
        "body_stop_radius_m": _round_number(body_stop_radius, 6),
        "nav2_params_config_path": str(raw.get("nav2_params_config_path") or ""),
        "nav2_params_config_sha256": nav2_sha256,
        "nav2_robot_radius_m": (
            FROZEN_NAV2_ROBOT_RADIUS_M
            if len(nav2_radii) == 2 and all(radius == FROZEN_NAV2_ROBOT_RADIUS_M for radius in nav2_radii)
            else None
        ),
        "safety_config_summary_sha256": safety_summary,
        "polygon_xy_m": polygon,
        "geometry": geometry,
        "reason_codes": reasons,
    }


def _bounded_regular_sha256(path: Path, *, maximum_bytes: int = 8 * 1024 * 1024) -> tuple[str, int]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum_bytes:
            raise ValueError("runtime executable is not a bounded regular file")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, min(65_536, maximum_bytes + 1 - size))
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            if size > maximum_bytes:
                raise ValueError("runtime executable exceeds read limit")
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("runtime executable changed during hashing")
        return digest.hexdigest(), size
    finally:
        os.close(descriptor)


def _process_start_epoch_ns(pid: int) -> int:
    stat_text = _read_text(Path(f"/proc/{pid}/stat"), maximum_bytes=65536)
    closing = stat_text.rfind(")")
    if closing <= 0:
        raise ValueError("process stat is malformed")
    fields_after_comm = stat_text[closing + 2 :].split()
    if len(fields_after_comm) <= 19:
        raise ValueError("process stat is incomplete")
    start_ticks = int(fields_after_comm[19])
    clock_ticks = int(os.sysconf("SC_CLK_TCK"))
    uptime_s = float(_read_text(Path("/proc/uptime"), maximum_bytes=4096).split()[0])
    if start_ticks <= 0 or clock_ticks <= 0 or not math.isfinite(uptime_s):
        raise ValueError("process timing is invalid")
    boot_epoch_ns = time.time_ns() - int(uptime_s * 1_000_000_000)
    return boot_epoch_ns + int(start_ticks * 1_000_000_000 / clock_ticks)


def _read_installed_release_artifacts() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, spec in INSTALLED_RELEASE_ARTIFACT_SPECS.items():
        source = Path(spec.source_path)
        installed = Path(spec.installed_path)
        record: dict[str, Any] = {
            "source_path": spec.source_path,
            "installed_path": spec.installed_path,
            "resolved_path": "",
            "file_kind": "unavailable",
            "present": False,
            "sha256": None,
            "size_bytes": None,
            "expected_sha256": spec.expected_sha256,
            "hash_match": False,
            "source_install_match": False,
        }
        try:
            source_lstat = source.lstat()
            installed_lstat = installed.lstat()
            if not stat.S_ISREG(source_lstat.st_mode):
                raise ValueError("source is not a regular file")
            installed_is_link = stat.S_ISLNK(installed_lstat.st_mode)
            if not (installed_is_link or stat.S_ISREG(installed_lstat.st_mode)):
                raise ValueError("installed artifact is not regular or symlinked")
            resolved = installed.resolve(strict=True)
            if installed_is_link and resolved != source:
                raise ValueError("installed symlink does not resolve to frozen source")
            source_digest, source_size = _bounded_regular_sha256(source)
            installed_digest, installed_size = _bounded_regular_sha256(resolved)
            record.update(
                {
                    "resolved_path": str(resolved),
                    "file_kind": "symlink_to_source" if installed_is_link else "regular_install",
                    "present": True,
                    "sha256": installed_digest,
                    "size_bytes": installed_size,
                    "hash_match": installed_digest == spec.expected_sha256,
                    "source_install_match": (
                        source_digest == installed_digest == spec.expected_sha256
                        and source_size == installed_size
                    ),
                }
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            pass
        result[name] = record
    return result


def _read_scan_filter_runtime(pid_values: set[int]) -> dict[str, Any]:
    """Bind the installed executable and its unique process in the finals unit."""

    installed = Path(SCAN_FILTER_INSTALLED_PATH)
    source = Path(SCAN_FILTER_SOURCE_PATH)
    result: dict[str, Any] = {
        "expected_path": SCAN_FILTER_INSTALLED_PATH,
        "resolved_path": "",
        "file_kind": "unavailable",
        "present": False,
        "sha256": None,
        "size_bytes": None,
        "expected_sha256": FROZEN_SCAN_SELF_FILTER_SHA256,
        "hash_match": False,
        "matching_process_count": 0,
        "matched_pid": None,
        "cmdline_sha256": None,
        "exact_cmdline_match": False,
        "artifact_mtime_epoch_ms": None,
        "process_start_epoch_ms": None,
        "process_started_after_artifact": False,
    }
    try:
        installed_lstat = installed.lstat()
        is_link = stat.S_ISLNK(installed_lstat.st_mode)
        if not (is_link or stat.S_ISREG(installed_lstat.st_mode)):
            return result
        resolved = installed.resolve(strict=True)
        if is_link and resolved != source:
            return result
        digest, size = _bounded_regular_sha256(resolved)
        result.update(
            {
                "resolved_path": str(resolved),
                "file_kind": "symlink_to_source" if is_link else "regular_install",
                "present": True,
                "sha256": digest,
                "size_bytes": size,
                "hash_match": digest == FROZEN_SCAN_SELF_FILTER_SHA256,
            }
        )
    except (OSError, RuntimeError, ValueError):
        return result

    expected_token = SCAN_FILTER_INSTALLED_PATH.encode("utf-8")
    matches: list[tuple[int, bytes]] = []
    for pid in sorted(pid_values):
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            continue
        if not cmdline or len(cmdline) > 1024 * 1024:
            continue
        if expected_token in cmdline.rstrip(b"\0").split(b"\0"):
            matches.append((pid, cmdline))
    result["matching_process_count"] = len(matches)
    if len(matches) == 1:
        pid, cmdline = matches[0]
        try:
            artifact_mtime_ns = source.stat().st_mtime_ns
            process_start_ns = _process_start_epoch_ns(pid)
            process_started_after_artifact = process_start_ns + 1_000_000_000 >= artifact_mtime_ns
        except (OSError, TypeError, ValueError):
            artifact_mtime_ns = 0
            process_start_ns = 0
            process_started_after_artifact = False
        result.update(
            {
                "matched_pid": pid,
                "cmdline_sha256": hashlib.sha256(cmdline).hexdigest(),
                "exact_cmdline_match": True,
                "artifact_mtime_epoch_ms": artifact_mtime_ns // 1_000_000 or None,
                "process_start_epoch_ms": process_start_ns // 1_000_000 or None,
                "process_started_after_artifact": process_started_after_artifact,
            }
        )
    return result


def _read_service_invocation() -> tuple[str, int, dict[str, Any], list[str]]:
    """Read systemd session identity from cgroup/proc files without systemctl."""

    errors: list[str] = []
    pid_values: set[int] = set()
    cgroup_candidates = (
        Path("/sys/fs/cgroup/system.slice/embodied_brain.service/cgroup.procs"),
        Path("/sys/fs/cgroup/systemd/system.slice/embodied_brain.service/cgroup.procs"),
    )
    readable_cgroups: list[Path] = []
    for candidate in cgroup_candidates:
        try:
            content = _read_text(candidate)
            readable_cgroups.append(candidate)
            for token in content.split():
                if token.isdigit() and int(token) > 0:
                    pid_values.add(int(token))
        except (OSError, UnicodeError, ValueError):
            continue

    expected_fragment = "/system.slice/embodied_brain.service"
    pids_in_expected_cgroup = True
    if not pid_values:
        pids_in_expected_cgroup = False
    for pid in sorted(pid_values):
        try:
            membership = _read_text(Path(f"/proc/{pid}/cgroup"), maximum_bytes=65536)
        except (OSError, UnicodeError, ValueError):
            pids_in_expected_cgroup = False
            continue
        if expected_fragment not in membership:
            pids_in_expected_cgroup = False

    invocation_ids: set[str] = set()
    for pid in sorted(pid_values):
        try:
            environ = Path(f"/proc/{pid}/environ").read_bytes()
            if len(environ) > 1024 * 1024:
                continue
            for item in environ.split(b"\0"):
                if item.startswith(b"INVOCATION_ID="):
                    value = item.partition(b"=")[2].decode("ascii").lower()
                    if INVOCATION_ID_RE.fullmatch(value):
                        invocation_ids.add(value)
        except (OSError, UnicodeError):
            continue

    if not invocation_ids:
        units_dir = Path("/run/systemd/units")
        try:
            for candidate in units_dir.glob("invocation:*"):
                try:
                    target = os.readlink(candidate)
                except OSError:
                    continue
                if Path(target).name == SERVICE_NAME:
                    value = candidate.name.partition(":")[2].lower()
                    if INVOCATION_ID_RE.fullmatch(value):
                        invocation_ids.add(value)
        except OSError:
            pass

    if len(readable_cgroups) != 1:
        errors.append("SERVICE_CGROUP_MISSING" if not readable_cgroups else "SERVICE_CGROUP_AMBIGUOUS")
    if not pids_in_expected_cgroup:
        errors.append("SERVICE_CGROUP_MEMBERSHIP_INVALID")
    scan_filter_runtime = _read_scan_filter_runtime(pid_values)
    installed_release_artifacts = _read_installed_release_artifacts()
    if not (
        scan_filter_runtime["present"] is True
        and scan_filter_runtime["hash_match"] is True
        and scan_filter_runtime["matching_process_count"] == 1
        and scan_filter_runtime["exact_cmdline_match"] is True
        and scan_filter_runtime["process_started_after_artifact"] is True
    ):
        errors.append("SCAN_FILTER_RUNTIME_PROCESS_INVALID")
    if not all(
        record["present"] is True and record["hash_match"] is True and record["source_install_match"] is True
        for record in installed_release_artifacts.values()
    ):
        errors.append("INSTALLED_RELEASE_ARTIFACT_BINDING_INVALID")
    owner = {
        "unit": SERVICE_NAME,
        "cgroup_path": str(readable_cgroups[0]) if len(readable_cgroups) == 1 else "",
        "cgroup_files_found": len(readable_cgroups),
        "pid_count": len(pid_values),
        "all_pids_in_unit_cgroup": pids_in_expected_cgroup,
        "scan_filter_runtime": scan_filter_runtime,
        "installed_release_artifacts": installed_release_artifacts,
    }
    if len(invocation_ids) != 1:
        errors.append("SERVICE_INVOCATION_MISSING" if not invocation_ids else "SERVICE_INVOCATION_AMBIGUOUS")
        return "", len(pid_values), owner, errors
    return next(iter(invocation_ids)), len(pid_values), owner, errors


def collect_machine_identity() -> dict[str, Any]:
    """Read immutable host/boot/session identity from regular pseudo-files."""

    errors: list[str] = []

    def read_or_reason(path: str, reason: str) -> str:
        try:
            return _read_text(Path(path))
        except (OSError, UnicodeError, ValueError):
            errors.append(reason)
            return ""

    hostname = read_or_reason("/etc/hostname", "HOSTNAME_UNAVAILABLE").lower()
    machine_id = read_or_reason("/etc/machine-id", "MACHINE_ID_UNAVAILABLE").lower()
    boot_id = read_or_reason("/proc/sys/kernel/random/boot_id", "BOOT_ID_UNAVAILABLE").lower()
    wlan_mac = read_or_reason("/sys/class/net/wlan0/address", "WLAN_MAC_UNAVAILABLE").lower()
    invocation_id, process_count, service_owner, invocation_errors = _read_service_invocation()
    errors.extend(invocation_errors)
    return {
        "hostname": hostname,
        "machine_id": machine_id,
        "boot_id": boot_id,
        "wlan_mac": wlan_mac,
        "service_invocation_id": invocation_id,
        "service_process_count": process_count,
        "service_owner": service_owner,
        "collector_errors": sorted(set(errors)),
    }


def _file_sha256(path: Path) -> tuple[str, int]:
    if str(path).startswith("/dev/"):
        raise ValueError("hardware device paths are outside the artifact inventory")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def hash_deployed_artifacts(
    specs: Mapping[str, ArtifactSpec] | None = None,
) -> dict[str, dict[str, Any]]:
    """Hash only fixed regular-file deployment paths; missing files stay explicit."""

    if specs is None:
        specs = {**ARTIFACT_SPECS, **SCAN_FILTER_ARTIFACT_SPECS}
    result: dict[str, dict[str, Any]] = {}
    for name in sorted(specs):
        spec = specs[name]
        path = Path(spec.path)
        present = path.is_file() and not path.is_symlink()
        digest: str | None = None
        size_bytes: int | None = None
        if present:
            try:
                digest, size_bytes = _file_sha256(path)
            except (OSError, ValueError):
                present = False
        result[name] = {
            "path": spec.path,
            "present": present,
            "sha256": digest,
            "size_bytes": size_bytes,
        }
    return result


def _empty_topic_record() -> dict[str, Any]:
    return {
        "message_count": 0,
        "age_s": None,
        "rate_hz": None,
        "frame_id": None,
        "publishers": [],
        "sample": None,
    }


def _empty_scan_filter_observation(
    body_contour: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "available": False,
        "raw_topic": "/scan_raw",
        "filtered_topic": "/scan",
        "raw_valid_count": None,
        "filtered_valid_count": None,
        "sample_count": None,
        "angular_coverage_rad": None,
        "exact_stamp_match": False,
        "paired_stamp_ns": None,
        "paired_stamp_age_s": None,
        "paired_stamp_fresh": False,
        "geometry_valid": False,
        "geometry_match": False,
        "transform_match": False,
        "removed_count": None,
        "positive_infinity_removed_count": None,
        "unremoved_inside_contour_count": None,
        "modified_or_inserted_count": None,
        "removed_outside_contour_count": None,
        "invalid_ranges_preserved": False,
        "intensities_preserved": False,
        "all_removed_points_inside_contour": False,
        "body_contour": dict(
            body_contour or _unavailable_body_contour("BODY_CONTOUR_VERIFIED_ARTIFACT_MISSING")
        ),
    }


def empty_ros_observation(
    reason: str,
    *,
    localization_mode: str = LOCALIZATION_ONLINE_SLAM,
    body_contour: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "collector_errors": [reason],
        "localization_mode": localization_mode,
        "clock": {
            "use_sim_time": None,
            "ros_time_ns": None,
            "system_time_ns": time.time_ns(),
            "wall_time_delta_s": None,
        },
        "topics": {topic: _empty_topic_record() for topic in SUBSCRIBED_TOPICS},
        "graph": {
            "nodes": [],
            "publishers": {topic: [] for topic in GRAPH_TOPICS},
            "subscribers": {topic: [] for topic in GRAPH_TOPICS},
        },
        "lifecycle": {name: "unavailable" for name in LIFECYCLE_NODES},
        "parameters": {
            "collision_monitor": {},
            "mppi_node": {},
        },
        "transforms": {
            "map_to_odom": {"available": False},
            "odom_to_base_footprint": {"available": False},
            "base_to_scan_raw": {"available": False},
            "base_to_scan": {"available": False},
            "base_to_scan_depth": {"available": False},
        },
        "scan_filter": _empty_scan_filter_observation(body_contour),
        "clearance": {
            "available": False,
            "scan_body_stop_points": None,
            "scan_front_stop_points": None,
            "scan_depth_body_stop_points": None,
            "scan_depth_front_stop_points": None,
            "forward_centerline_max_cost": None,
            "forward_centerline_free": False,
        },
    }


def _round_number(value: Any, digits: int = 3) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _full_node_name(namespace: str, node_name: str) -> str:
    namespace = namespace if namespace.startswith("/") else f"/{namespace}"
    namespace = namespace.rstrip("/")
    return f"{namespace}/{node_name}" if namespace else f"/{node_name}"


def _endpoint_record(info: Any) -> dict[str, str]:
    gid_value = getattr(info, "endpoint_gid", b"")
    try:
        gid = bytes(gid_value).hex()
    except (TypeError, ValueError):
        gid = ""
    return {
        "node": _full_node_name(
            str(getattr(info, "node_namespace", "/")), str(getattr(info, "node_name", ""))
        ),
        "topic_type": str(getattr(info, "topic_type", "")),
        "gid": gid,
    }


def _json_string_sample(value: Any) -> dict[str, Any] | None:
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _diagnostic_sample(message: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for status in getattr(message, "status", []):
        values = {
            str(getattr(item, "key", "")): str(getattr(item, "value", ""))
            for item in getattr(status, "values", [])
            if str(getattr(item, "key", ""))
        }
        rows.append(
            {
                "name": str(getattr(status, "name", "")),
                "hardware_id": str(getattr(status, "hardware_id", "")),
                "level": int(getattr(status, "level", -1)),
                "message": str(getattr(status, "message", "")),
                "values": values,
            }
        )
    return sorted(rows, key=lambda row: (row["name"], row["hardware_id"]))


def _rotate_vector(rotation: Any, x: float, y: float, z: float = 0.0) -> tuple[float, float, float]:
    tx = 2.0 * (rotation.y * z - rotation.z * y)
    ty = 2.0 * (rotation.z * x - rotation.x * z)
    tz = 2.0 * (rotation.x * y - rotation.y * x)
    return (
        x + rotation.w * tx + (rotation.y * tz - rotation.z * ty),
        y + rotation.w * ty + (rotation.z * tx - rotation.x * tz),
        z + rotation.w * tz + (rotation.x * ty - rotation.y * tx),
    )


def _scan_clearance_counts(message: Any, transform: Any) -> tuple[int, int]:
    body_points = 0
    front_points = 0
    angle = float(message.angle_min)
    for raw_range in message.ranges:
        value = float(raw_range)
        if math.isfinite(value) and message.range_min <= value <= message.range_max:
            rx, ry, _ = _rotate_vector(transform.rotation, value * math.cos(angle), value * math.sin(angle))
            x = rx + float(transform.translation.x)
            y = ry + float(transform.translation.y)
            if x * x + y * y <= 0.34 * 0.34:
                body_points += 1
            if 0.02 <= x <= 0.55 and abs(y) <= 0.32:
                front_points += 1
        angle += float(message.angle_increment)
    return body_points, front_points


def _ros_header_stamp_ns(message: Any) -> int | None:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    try:
        sec = int(stamp.sec)
        nanosec = int(stamp.nanosec)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if sec < 0 or not 0 <= nanosec < 1_000_000_000:
        return None
    stamp_ns = sec * 1_000_000_000 + nanosec
    return stamp_ns if stamp_ns > 0 else None


class _ExactScanPair(NamedTuple):
    stamp_ns: int
    raw_message: Any
    filtered_message: Any


class _ExactStampScanPairCache:
    """Bounded, arrival-order cache for exact ``scan_raw``/``scan`` pairs."""

    def __init__(self, max_entries: int = SCAN_PAIR_CACHE_SIZE) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        self._max_entries = max_entries
        self._raw: OrderedDict[int, Any] = OrderedDict()
        self._filtered: OrderedDict[int, Any] = OrderedDict()
        self._latest_pair: _ExactScanPair | None = None

    def add(self, topic: str, message: Any) -> _ExactScanPair | None:
        if topic not in {"/scan_raw", "/scan"}:
            raise ValueError("scan pair cache accepts only /scan_raw and /scan")
        stamp_ns = _ros_header_stamp_ns(message)
        if stamp_ns is None:
            return None
        own = self._raw if topic == "/scan_raw" else self._filtered
        peer = self._filtered if topic == "/scan_raw" else self._raw
        own[stamp_ns] = message
        own.move_to_end(stamp_ns)
        if stamp_ns in peer:
            peer_message = peer.pop(stamp_ns)
            own_message = own.pop(stamp_ns)
            pair = (
                _ExactScanPair(stamp_ns, own_message, peer_message)
                if topic == "/scan_raw"
                else _ExactScanPair(stamp_ns, peer_message, own_message)
            )
            self._latest_pair = pair
        else:
            pair = None
        self._prune(self._raw)
        self._prune(self._filtered)
        return pair

    def latest_pair(self) -> _ExactScanPair | None:
        return self._latest_pair

    def _prune(self, cache: OrderedDict[int, Any]) -> None:
        while len(cache) > self._max_entries:
            cache.popitem(last=False)


def _scan_geometry(message: Any) -> dict[str, Any]:
    ranges = list(getattr(message, "ranges", []))
    intensities = list(getattr(message, "intensities", []))
    angle_min = _parse_float(getattr(message, "angle_min", None))
    angle_max = _parse_float(getattr(message, "angle_max", None))
    angle_increment = _parse_float(getattr(message, "angle_increment", None))
    time_increment = _parse_float(getattr(message, "time_increment", None))
    scan_time = _parse_float(getattr(message, "scan_time", None))
    range_min = _parse_float(getattr(message, "range_min", None))
    range_max = _parse_float(getattr(message, "range_max", None))
    finite_count = sum(_parse_float(value) is not None for value in ranges)
    valid_count = 0
    if range_min is not None and range_max is not None and range_min < range_max:
        valid_count = sum(
            value is not None and range_min <= value <= range_max
            for value in (_parse_float(item) for item in ranges)
        )
    coverage = abs(angle_increment) * max(0, len(ranges) - 1) if angle_increment is not None else None
    header = getattr(message, "header", None)
    stamp_ns = _ros_header_stamp_ns(message) or 0
    angular_consistent = False
    timing_consistent = False
    if (
        ranges
        and angle_min is not None
        and angle_max is not None
        and angle_increment is not None
        and angle_increment != 0.0
    ):
        expected_angle_max = angle_min + (len(ranges) - 1) * angle_increment
        angular_consistent = abs(angle_max - expected_angle_max) <= max(1e-6, abs(angle_increment) * 1e-3)
    if time_increment is not None and scan_time is not None and time_increment >= 0.0 and scan_time > 0.0:
        sweep_duration = time_increment * max(0, len(ranges) - 1)
        timing_consistent = scan_time + max(1e-6, scan_time * 1e-3) >= sweep_duration
    geometry_valid = (
        str(getattr(header, "frame_id", "")) == EXPECTED_SCAN_FRAME_ID
        and stamp_ns > 0
        and range_min is not None
        and range_max is not None
        and 0.0 <= range_min < range_max
        and angular_consistent
        and timing_consistent
        and (not intensities or len(intensities) == len(ranges))
    )
    return {
        "frame_id": str(getattr(header, "frame_id", "")),
        "stamp_ns": stamp_ns,
        "geometry_valid": geometry_valid,
        "sample_count": len(ranges),
        "finite_count": finite_count,
        "valid_count": valid_count,
        "angle_min": _round_number(angle_min, 9),
        "angle_max": _round_number(angle_max, 9),
        "angle_increment": _round_number(angle_increment, 12),
        "time_increment": _round_number(time_increment, 12),
        "scan_time": _round_number(scan_time, 9),
        "range_min": _round_number(range_min, 6),
        "range_max": _round_number(range_max, 6),
        "angular_coverage_rad": _round_number(coverage, 6),
    }


def _transform_close(first: Any, second: Any, *, tolerance: float = 1e-6) -> bool:
    try:
        first_values = (
            float(first.translation.x),
            float(first.translation.y),
            float(first.translation.z),
            float(first.rotation.x),
            float(first.rotation.y),
            float(first.rotation.z),
            float(first.rotation.w),
        )
        second_values = (
            float(second.translation.x),
            float(second.translation.y),
            float(second.translation.z),
            float(second.rotation.x),
            float(second.rotation.y),
            float(second.rotation.z),
            float(second.rotation.w),
        )
    except (AttributeError, TypeError, ValueError):
        return False
    return all(
        math.isfinite(left) and math.isfinite(right) and abs(left - right) <= tolerance
        for left, right in zip(first_values, second_values, strict=True)
    )


def _point_inside_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if abs(cross) <= 1e-9 and min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9:
            if min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9:
                return True
        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


def _point_segment_distance(
    x: float,
    y: float,
    first: list[float],
    second: list[float],
) -> float:
    dx = second[0] - first[0]
    dy = second[1] - first[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-24:
        return math.hypot(x - first[0], y - first[1])
    projection = ((x - first[0]) * dx + (y - first[1]) * dy) / length_squared
    projection = min(1.0, max(0.0, projection))
    return math.hypot(x - (first[0] + projection * dx), y - (first[1] + projection * dy))


def _point_inside_contour_envelope(
    x: float,
    y: float,
    polygon: list[list[float]],
    uncertainty_m: float,
) -> bool:
    if _point_inside_polygon(x, y, polygon):
        return True
    if uncertainty_m <= 0.0:
        return False
    return any(
        _point_segment_distance(x, y, polygon[index - 1], polygon[index]) <= uncertainty_m + 1e-12
        for index in range(len(polygon))
    )


def _range_value_preserved(first: Any, second: Any) -> bool:
    try:
        left = float(first)
        right = float(second)
    except (TypeError, ValueError, OverflowError):
        return False
    if math.isnan(left) or math.isnan(right):
        return math.isnan(left) and math.isnan(right)
    if math.isinf(left) or math.isinf(right):
        return (
            math.isinf(left) and math.isinf(right) and math.copysign(1.0, left) == math.copysign(1.0, right)
        )
    tolerance = max(1e-7, abs(left) * 1e-7)
    return abs(left - right) <= tolerance


def _numeric_sequence_preserved(first: Any, second: Any) -> bool:
    try:
        left = list(first)
        right = list(second)
    except TypeError:
        return False
    return len(left) == len(right) and all(
        _range_value_preserved(left_item, right_item)
        for left_item, right_item in zip(left, right, strict=True)
    )


def _scan_filter_observation(
    raw_message: Any,
    filtered_message: Any,
    raw_transform: Any,
    filtered_transform: Any,
    body_contour: Mapping[str, Any],
    *,
    observed_ros_time_ns: int,
) -> dict[str, Any]:
    result = _empty_scan_filter_observation(body_contour)
    raw_geometry = _scan_geometry(raw_message)
    filtered_geometry = _scan_geometry(filtered_message)
    geometry_keys = (
        "frame_id",
        "stamp_ns",
        "sample_count",
        "angle_min",
        "angle_max",
        "angle_increment",
        "time_increment",
        "scan_time",
        "range_min",
        "range_max",
    )
    geometry_valid = raw_geometry["geometry_valid"] is True and filtered_geometry["geometry_valid"] is True
    geometry_match = geometry_valid and all(
        raw_geometry[key] == filtered_geometry[key] for key in geometry_keys
    )
    exact_stamp_match = (
        raw_geometry["stamp_ns"] > 0 and raw_geometry["stamp_ns"] == filtered_geometry["stamp_ns"]
    )
    transform_match = _transform_close(raw_transform, filtered_transform)
    polygon = _normalized_polygon(body_contour.get("polygon_xy_m"))
    uncertainty_m = _parse_float(body_contour.get("measurement_uncertainty_m"))
    raw_ranges = list(getattr(raw_message, "ranges", []))
    filtered_ranges = list(getattr(filtered_message, "ranges", []))
    paired_stamp_age_s = (
        (observed_ros_time_ns - raw_geometry["stamp_ns"]) / 1_000_000_000
        if exact_stamp_match
        and isinstance(observed_ros_time_ns, int)
        and not isinstance(observed_ros_time_ns, bool)
        and observed_ros_time_ns > 0
        else None
    )
    paired_stamp_fresh = (
        paired_stamp_age_s is not None
        and -MAX_SCAN_PAIR_FUTURE_S <= paired_stamp_age_s <= MAX_SCAN_PAIR_AGE_S
    )
    removed_count = 0
    positive_infinity_removed_count = 0
    unremoved_inside_count = 0
    modified_or_inserted_count = 0
    removed_outside_count = 0
    invalid_ranges_preserved = True
    intensities_preserved = _numeric_sequence_preserved(
        getattr(raw_message, "intensities", []),
        getattr(filtered_message, "intensities", []),
    )

    if geometry_match and transform_match and polygon is not None and uncertainty_m is not None:
        angle = float(getattr(raw_message, "angle_min", 0.0))
        increment = float(getattr(raw_message, "angle_increment", 0.0))
        range_min = float(getattr(raw_message, "range_min", 0.0))
        range_max = float(getattr(raw_message, "range_max", 0.0))
        for raw_item, filtered_item in zip(raw_ranges, filtered_ranges, strict=True):
            raw_value = _parse_float(raw_item)
            filtered_value = _parse_float(filtered_item)
            raw_valid = raw_value is not None and range_min <= raw_value <= range_max
            filtered_valid = filtered_value is not None and range_min <= filtered_value <= range_max
            filtered_positive_infinity = False
            try:
                filtered_number = float(filtered_item)
                filtered_positive_infinity = math.isinf(filtered_number) and filtered_number > 0.0
            except (TypeError, ValueError, OverflowError):
                pass
            inside_envelope = False
            if raw_valid:
                rx, ry, _ = _rotate_vector(
                    raw_transform.rotation,
                    raw_value * math.cos(angle),
                    raw_value * math.sin(angle),
                )
                x = rx + float(raw_transform.translation.x)
                y = ry + float(raw_transform.translation.y)
                inside_envelope = _point_inside_contour_envelope(x, y, polygon, uncertainty_m)
            if raw_valid and filtered_positive_infinity:
                removed_count += 1
                positive_infinity_removed_count += 1
                if not inside_envelope:
                    removed_outside_count += 1
            elif raw_valid and filtered_valid and _range_value_preserved(raw_item, filtered_item):
                if inside_envelope:
                    unremoved_inside_count += 1
            elif raw_valid:
                modified_or_inserted_count += 1
            elif not _range_value_preserved(raw_item, filtered_item):
                invalid_ranges_preserved = False
                modified_or_inserted_count += 1
            angle += increment

    result.update(
        {
            "available": True,
            "raw_valid_count": raw_geometry["valid_count"],
            "filtered_valid_count": filtered_geometry["valid_count"],
            "sample_count": raw_geometry["sample_count"],
            "angular_coverage_rad": raw_geometry["angular_coverage_rad"],
            "exact_stamp_match": exact_stamp_match,
            "paired_stamp_ns": raw_geometry["stamp_ns"] if exact_stamp_match else None,
            "paired_stamp_age_s": _round_number(paired_stamp_age_s, 6),
            "paired_stamp_fresh": paired_stamp_fresh,
            "geometry_valid": geometry_valid,
            "geometry_match": geometry_match,
            "transform_match": transform_match,
            "removed_count": removed_count,
            "positive_infinity_removed_count": positive_infinity_removed_count,
            "unremoved_inside_contour_count": unremoved_inside_count,
            "modified_or_inserted_count": modified_or_inserted_count,
            "removed_outside_contour_count": removed_outside_count,
            "invalid_ranges_preserved": invalid_ranges_preserved,
            "intensities_preserved": intensities_preserved,
            "all_removed_points_inside_contour": removed_outside_count == 0,
        }
    )
    return result


def _costmap_forward_clearance(message: Any, map_to_base: Any) -> tuple[int | None, bool]:
    q = map_to_base.rotation
    yaw = math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )
    resolution = float(message.info.resolution)
    width = int(message.info.width)
    height = int(message.info.height)
    if resolution <= 0.0 or width <= 0 or height <= 0:
        return None, False
    origin_x = float(message.info.origin.position.x)
    origin_y = float(message.info.origin.position.y)
    costs: list[int] = []
    for step in range(11):
        distance = step * 0.05
        x = float(map_to_base.translation.x) + distance * math.cos(yaw)
        y = float(map_to_base.translation.y) + distance * math.sin(yaw)
        ix = int((x - origin_x) / resolution)
        iy = int((y - origin_y) / resolution)
        if not (0 <= ix < width and 0 <= iy < height):
            return None, False
        costs.append(int(message.data[iy * width + ix]))
    maximum = max(costs) if costs else None
    return maximum, maximum is not None and 0 <= maximum <= 20


def _twist_sample(message: Any) -> dict[str, float | None]:
    linear = getattr(message, "linear", None)
    angular = getattr(message, "angular", None)
    return {
        "linear_x": _round_number(getattr(linear, "x", None), 9),
        "linear_y": _round_number(getattr(linear, "y", None), 9),
        "linear_z": _round_number(getattr(linear, "z", None), 9),
        "angular_x": _round_number(getattr(angular, "x", None), 9),
        "angular_y": _round_number(getattr(angular, "y", None), 9),
        "angular_z": _round_number(getattr(angular, "z", None), 9),
    }


def _parameter_value_to_python(value: Any) -> Any:
    parameter_type = int(getattr(value, "type", 0))
    scalar_fields = {
        1: "bool_value",
        2: "integer_value",
        3: "double_value",
        4: "string_value",
    }
    array_fields = {
        5: "byte_array_value",
        6: "bool_array_value",
        7: "integer_array_value",
        8: "double_array_value",
        9: "string_array_value",
    }
    if parameter_type in scalar_fields:
        raw = getattr(value, scalar_fields[parameter_type], None)
        if parameter_type == 3:
            return _round_number(raw, 9)
        return raw
    if parameter_type in array_fields:
        raw_values = list(getattr(value, array_fields[parameter_type], []))
        if parameter_type == 8:
            return [_round_number(item, 9) for item in raw_values]
        return raw_values
    return None


def collect_ros_observation(
    timeout_sec: float,
    localization_mode: str = LOCALIZATION_ONLINE_SLAM,
) -> dict[str, Any]:
    """Collect one local ROS graph snapshot using observation-only interfaces."""

    if localization_mode not in LOCALIZATION_MODES:
        raise ValueError("unsupported localization_mode")
    body_contour = _read_verified_body_contour()

    try:
        import rclpy
        from diagnostic_msgs.msg import DiagnosticArray
        from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
        from lifecycle_msgs.srv import GetState
        from nav2_msgs.msg import CollisionMonitorState
        from nav_msgs.msg import OccupancyGrid, Odometry
        from rcl_interfaces.srv import GetParameters
        from rclpy.duration import Duration
        from rclpy.executors import ExternalShutdownException
        from rclpy.node import Node
        from rclpy.qos import (
            DurabilityPolicy,
            QoSProfile,
            ReliabilityPolicy,
            qos_profile_sensor_data,
        )
        from rclpy.time import Time as RosTime
        from sensor_msgs.msg import Imu, LaserScan
        from std_msgs.msg import Bool, String
        from tf2_ros import Buffer, TransformListener
    except ImportError:
        return empty_ros_observation(
            "ROS_IMPORT_UNAVAILABLE",
            localization_mode=localization_mode,
            body_contour=body_contour,
        )

    trackers: dict[str, dict[str, Any]] = {
        topic: {"count": 0, "first": None, "last": None, "sample": None, "message": None}
        for topic in SUBSCRIBED_TOPICS
    }
    scan_pair_cache = _ExactStampScanPairCache()

    def sample_for(topic: str, message: Any) -> Any:
        if topic in {
            "/scan_raw",
            "/scan",
            "/scan_depth",
            "/wheel_odom",
            "/imu",
            "/odom",
            "/map",
            "/amcl_pose",
            "/global_costmap/costmap",
        }:
            header = getattr(message, "header", None)
            frame_id = str(getattr(header, "frame_id", "")) if header is not None else ""
            if topic in {"/scan_raw", "/scan", "/scan_depth"}:
                return _scan_geometry(message)
            if topic in {"/map", "/global_costmap/costmap"}:
                return {
                    "frame_id": frame_id,
                    "width": int(message.info.width),
                    "height": int(message.info.height),
                    "resolution": _round_number(message.info.resolution),
                }
            if topic in {"/wheel_odom", "/odom"}:
                return {"frame_id": frame_id, "child_frame_id": str(message.child_frame_id)}
            return {"frame_id": frame_id}
        if topic == "/collision_monitor/state":
            polygon_names = getattr(message, "polygons", [])
            if not isinstance(polygon_names, (list, tuple)):
                polygon_names = []
            return {
                "action_type": int(getattr(message, "action_type", -1)),
                "polygon_names": sorted(str(item) for item in polygon_names),
            }
        if topic == "/mppi/cmd_vel_proposed":
            return _twist_sample(message)
        if topic in {
            "/f407/firmware_identity_valid",
            "/f407/estop_latched",
            "/f407/cmd_vel_expired",
        }:
            return bool(message.data)
        if topic in {
            "/f407/firmware_info",
            "/lab_fsd/fsd_v3_status",
            "/lab_fsd/input_status",
            "/lab_fsd/safety_gate",
            "/mppi/stats",
        }:
            return _json_string_sample(message.data)
        if topic == "/diagnostics":
            return _diagnostic_sample(message)
        return None

    class ReadOnlyCollector(Node):
        def __init__(self) -> None:
            super().__init__("rb_voe_embodied_snapshot_once")
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.subscription_handles: list[Any] = []

            def track(topic: str):
                def callback(message: Any) -> None:
                    now_value = time.monotonic()
                    item = trackers[topic]
                    item["count"] += 1
                    if item["first"] is None:
                        item["first"] = now_value
                    item["last"] = now_value
                    item["sample"] = sample_for(topic, message)
                    item["message"] = message
                    if topic in {"/scan_raw", "/scan"}:
                        scan_pair_cache.add(topic, message)

                return callback

            state_qos = QoSProfile(depth=10)
            map_qos = QoSProfile(depth=1)
            map_qos.reliability = ReliabilityPolicy.RELIABLE
            map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
            subscriptions = (
                (LaserScan, "/scan_raw", qos_profile_sensor_data),
                (LaserScan, "/scan", qos_profile_sensor_data),
                (LaserScan, "/scan_depth", qos_profile_sensor_data),
                (Odometry, "/wheel_odom", qos_profile_sensor_data),
                (Imu, "/imu", qos_profile_sensor_data),
                (Odometry, "/odom", qos_profile_sensor_data),
                (OccupancyGrid, "/map", map_qos),
                (PoseWithCovarianceStamped, "/amcl_pose", state_qos),
                (OccupancyGrid, "/global_costmap/costmap", map_qos),
                (CollisionMonitorState, "/collision_monitor/state", state_qos),
                (Bool, "/f407/firmware_identity_valid", state_qos),
                (Bool, "/f407/estop_latched", state_qos),
                (Bool, "/f407/cmd_vel_expired", state_qos),
                (String, "/f407/firmware_info", state_qos),
                (DiagnosticArray, "/diagnostics", state_qos),
                (String, "/lab_fsd/fsd_v3_status", state_qos),
                (String, "/lab_fsd/input_status", state_qos),
                (String, "/lab_fsd/safety_gate", state_qos),
                (String, "/mppi/stats", state_qos),
                (Twist, "/mppi/cmd_vel_proposed", state_qos),
            )
            for message_type, topic, qos in subscriptions:
                self.subscription_handles.append(
                    self.create_subscription(message_type, topic, track(topic), qos)
                )

        def endpoint_records(self, topic: str, *, publishers: bool) -> list[dict[str, str]]:
            try:
                infos = (
                    self.get_publishers_info_by_topic(topic)
                    if publishers
                    else self.get_subscriptions_info_by_topic(topic)
                )
            except Exception:
                return []
            records = [_endpoint_record(info) for info in infos]
            return sorted(records, key=lambda row: (row["node"], row["topic_type"], row["gid"]))

        def graph_nodes(self) -> list[str]:
            try:
                values = self.get_node_names_and_namespaces()
            except Exception:
                return []
            return sorted({_full_node_name(namespace, name) for name, namespace in values})

        def lifecycle_states(self) -> dict[str, str]:
            states: dict[str, str] = {}
            for name in LIFECYCLE_NODES:
                client = self.create_client(GetState, f"/{name}/get_state")
                if not client.service_is_ready():
                    states[name] = "unavailable"
                    continue
                future = client.call_async(GetState.Request())
                try:
                    rclpy.spin_until_future_complete(self, future, timeout_sec=0.5)
                    response = future.result() if future.done() else None
                except Exception:
                    response = None
                states[name] = (
                    str(response.current_state.label).lower() if response is not None else "unavailable"
                )
            return states

        def parameter_values(self, node_name: str, names: tuple[str, ...]) -> dict[str, Any]:
            client = self.create_client(GetParameters, f"/{node_name}/get_parameters")
            if not client.wait_for_service(timeout_sec=0.25):
                return {}
            request = GetParameters.Request()
            request.names = list(names)
            future = client.call_async(request)
            try:
                rclpy.spin_until_future_complete(self, future, timeout_sec=0.5)
                response = future.result() if future.done() else None
            except Exception:
                response = None
            if response is None or len(response.values) != len(names):
                return {}
            return {
                name: _parameter_value_to_python(value)
                for name, value in zip(names, response.values, strict=True)
            }

        def transform_record(
            self,
            parent: str,
            child: str,
            *,
            acquisition_stamp: Any | None = None,
        ) -> tuple[dict[str, Any], Any | None]:
            if not child:
                return {"available": False, "parent": parent, "child": child}, None
            query_stamp_ns = 0
            query_time = RosTime()
            if acquisition_stamp is not None:
                try:
                    query_stamp_ns = int(acquisition_stamp.sec) * 1_000_000_000 + int(
                        acquisition_stamp.nanosec
                    )
                    if query_stamp_ns <= 0:
                        raise ValueError("non-positive acquisition stamp")
                    query_time = RosTime.from_msg(acquisition_stamp)
                except (AttributeError, TypeError, ValueError, OverflowError):
                    return {"available": False, "parent": parent, "child": child}, None
            try:
                stamped = self.tf_buffer.lookup_transform(
                    parent, child, query_time, timeout=Duration(seconds=0.4)
                )
            except Exception:
                return {"available": False, "parent": parent, "child": child}, None
            transform = stamped.transform
            stamp = getattr(stamped, "header", None)
            stamp = getattr(stamp, "stamp", None)
            stamp_ns = int(getattr(stamp, "sec", 0)) * 1_000_000_000 + int(getattr(stamp, "nanosec", 0))
            now_ns = int(self.get_clock().now().nanoseconds)
            age_s = (now_ns - stamp_ns) / 1_000_000_000 if stamp_ns > 0 else None
            return (
                {
                    "available": True,
                    "parent": parent,
                    "child": child,
                    "query_stamp_ns": query_stamp_ns,
                    "age_s": _round_number(age_s, 6),
                    "translation": [
                        _round_number(transform.translation.x, 6),
                        _round_number(transform.translation.y, 6),
                        _round_number(transform.translation.z, 6),
                    ],
                    "rotation": [
                        _round_number(transform.rotation.x, 6),
                        _round_number(transform.rotation.y, 6),
                        _round_number(transform.rotation.z, 6),
                        _round_number(transform.rotation.w, 6),
                    ],
                },
                transform,
            )

    initialized_here = False
    try:
        if not rclpy.ok():
            rclpy.init(args=[])
            initialized_here = True
        node = ReadOnlyCollector()
    except Exception:
        if initialized_here and rclpy.ok():
            rclpy.shutdown()
        return empty_ros_observation(
            "ROS_NODE_INITIALIZATION_FAILED",
            localization_mode=localization_mode,
            body_contour=body_contour,
        )

    deadline = time.monotonic() + max(1.0, min(float(timeout_sec), 60.0))
    collector_errors: list[str] = []
    try:
        while time.monotonic() < deadline:
            try:
                rclpy.spin_once(node, timeout_sec=0.1)
            except ExternalShutdownException:
                collector_errors.append("ROS_EXTERNAL_SHUTDOWN")
                break

        observed_monotonic = time.monotonic()
        observed_ros_time_ns = int(node.get_clock().now().nanoseconds)
        observed_system_time_ns = time.time_ns()
        try:
            use_sim_time = node.get_parameter("use_sim_time").value is True
        except Exception:
            use_sim_time = None
        wall_time_delta_s = abs(observed_ros_time_ns - observed_system_time_ns) / 1_000_000_000
        if use_sim_time is not False or wall_time_delta_s > 2.0:
            collector_errors.append("ROS_SIM_TIME_FORBIDDEN")
        graph = {
            "nodes": node.graph_nodes(),
            "publishers": {topic: node.endpoint_records(topic, publishers=True) for topic in GRAPH_TOPICS},
            "subscribers": {topic: node.endpoint_records(topic, publishers=False) for topic in GRAPH_TOPICS},
        }
        lifecycle = node.lifecycle_states()
        parameters = {
            "collision_monitor": node.parameter_values(
                "collision_monitor",
                (
                    "enabled",
                    "base_frame_id",
                    "odom_frame_id",
                    "cmd_vel_in_topic",
                    "cmd_vel_out_topic",
                    "state_topic",
                    "source_timeout",
                    "observation_sources",
                    "scan_lidar.topic",
                    "scan_lidar.enabled",
                    "scan_depth.topic",
                    "scan_depth.enabled",
                ),
            ),
            "mppi_node": node.parameter_values(
                "mppi_node",
                (
                    "bin_path",
                    "use_bpu",
                    "cmd_vel_topic",
                    "publish_direct_cmd_vel",
                    "max_linear_mps",
                    "max_angular_rps",
                ),
            ),
        }
        topic_records: dict[str, dict[str, Any]] = {}
        for topic, tracker in trackers.items():
            first = tracker["first"]
            last = tracker["last"]
            count = int(tracker["count"])
            rate_hz = None
            if count >= 2 and isinstance(first, float) and isinstance(last, float) and last > first:
                rate_hz = round((count - 1) / (last - first), 3)
            sample = tracker["sample"]
            frame_id = sample.get("frame_id") if isinstance(sample, Mapping) else None
            topic_records[topic] = {
                "message_count": count,
                "age_s": round(observed_monotonic - last, 3) if isinstance(last, float) else None,
                "rate_hz": rate_hz,
                "frame_id": frame_id,
                "publishers": graph["publishers"].get(topic, []),
                "sample": sample,
            }

        scan_pair = scan_pair_cache.latest_pair()
        raw_scan_message = scan_pair.raw_message if scan_pair is not None else None
        scan_message = scan_pair.filtered_message if scan_pair is not None else None
        raw_scan_frame = (
            _scan_geometry(raw_scan_message)["frame_id"]
            if raw_scan_message is not None
            else str(topic_records["/scan_raw"].get("frame_id") or "")
        )
        scan_frame = (
            _scan_geometry(scan_message)["frame_id"]
            if scan_message is not None
            else str(topic_records["/scan"].get("frame_id") or "")
        )
        depth_frame = str(topic_records["/scan_depth"].get("frame_id") or "")
        map_to_odom, _ = node.transform_record("map", "odom")
        odom_to_base, _ = node.transform_record("odom", "base_footprint")
        base_to_scan_raw, base_raw_scan_transform = node.transform_record(
            "base_footprint",
            raw_scan_frame,
            acquisition_stamp=(raw_scan_message.header.stamp if raw_scan_message is not None else None),
        )
        base_to_scan, base_scan_transform = node.transform_record(
            "base_footprint",
            scan_frame,
            acquisition_stamp=(scan_message.header.stamp if scan_message is not None else None),
        )
        base_to_depth, base_depth_transform = node.transform_record("base_footprint", depth_frame)
        _, map_base_transform = node.transform_record("map", "base_footprint")
        transforms = {
            "map_to_odom": map_to_odom,
            "odom_to_base_footprint": odom_to_base,
            "base_to_scan_raw": base_to_scan_raw,
            "base_to_scan": base_to_scan,
            "base_to_scan_depth": base_to_depth,
        }

        depth_message = trackers["/scan_depth"]["message"]
        costmap_message = trackers["/global_costmap/costmap"]["message"]
        clearance = {
            "available": False,
            "scan_body_stop_points": None,
            "scan_front_stop_points": None,
            "scan_depth_body_stop_points": None,
            "scan_depth_front_stop_points": None,
            "forward_centerline_max_cost": None,
            "forward_centerline_free": False,
        }
        scan_filter = _empty_scan_filter_observation(body_contour)
        try:
            if (
                raw_scan_message is not None
                and scan_message is not None
                and base_raw_scan_transform is not None
                and base_scan_transform is not None
            ):
                scan_filter = _scan_filter_observation(
                    raw_scan_message,
                    scan_message,
                    base_raw_scan_transform,
                    base_scan_transform,
                    body_contour,
                    observed_ros_time_ns=observed_ros_time_ns,
                )
            if (
                scan_message is not None
                and depth_message is not None
                and costmap_message is not None
                and base_scan_transform is not None
                and base_depth_transform is not None
                and map_base_transform is not None
            ):
                scan_body, scan_front = _scan_clearance_counts(scan_message, base_scan_transform)
                depth_body, depth_front = _scan_clearance_counts(depth_message, base_depth_transform)
                max_cost, forward_free = _costmap_forward_clearance(costmap_message, map_base_transform)
                clearance = {
                    "available": max_cost is not None,
                    "scan_body_stop_points": scan_body,
                    "scan_front_stop_points": scan_front,
                    "scan_depth_body_stop_points": depth_body,
                    "scan_depth_front_stop_points": depth_front,
                    "forward_centerline_max_cost": max_cost,
                    "forward_centerline_free": forward_free,
                }
        except (AttributeError, IndexError, TypeError, ValueError):
            collector_errors.append("CLEARANCE_COMPUTATION_FAILED")

        return {
            "collector_errors": sorted(set(collector_errors)),
            "localization_mode": localization_mode,
            "clock": {
                "use_sim_time": use_sim_time,
                "ros_time_ns": observed_ros_time_ns,
                "system_time_ns": observed_system_time_ns,
                "wall_time_delta_s": round(wall_time_delta_s, 6),
            },
            "topics": topic_records,
            "graph": graph,
            "lifecycle": lifecycle,
            "parameters": parameters,
            "transforms": transforms,
            "scan_filter": scan_filter,
            "clearance": clearance,
        }
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if initialized_here and rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass


def _node_basename(node_name: str) -> str:
    return str(node_name).rstrip("/").rsplit("/", 1)[-1]


def _endpoint_nodes(records: Any) -> list[str]:
    if not isinstance(records, list):
        return []
    result = {
        str(record.get("node"))
        for record in records
        if isinstance(record, Mapping) and isinstance(record.get("node"), str)
    }
    return sorted(name for name in result if name)


def _has_node(records: Any, basename: str) -> bool:
    return any(_node_basename(name) == basename for name in _endpoint_nodes(records))


def _normalized_endpoints(records: Any) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, Mapping):
                continue
            normalized.append(
                {
                    "node": str(record.get("node") or ""),
                    "topic_type": str(record.get("topic_type") or ""),
                    "gid": str(record.get("gid") or ""),
                }
            )
    return sorted(normalized, key=lambda row: (row["node"], row["topic_type"], row["gid"]))


def _topic_health(
    topics: Mapping[str, Any],
    topic: str,
    *,
    stale_after_s: float,
    minimum_rate_hz: float | None,
    reason_prefix: str,
) -> tuple[dict[str, Any], list[str]]:
    raw = topics.get(topic, {})
    raw = raw if isinstance(raw, Mapping) else {}
    count = raw.get("message_count")
    count = int(count) if isinstance(count, int) and not isinstance(count, bool) and count >= 0 else 0
    age_s = _round_number(raw.get("age_s"))
    rate_hz = _round_number(raw.get("rate_hz"))
    publishers = _normalized_endpoints(raw.get("publishers"))
    reasons: list[str] = []
    if count <= 0:
        reasons.append(f"{reason_prefix}_MISSING")
    if age_s is None or age_s < 0.0 or age_s > stale_after_s:
        reasons.append(f"{reason_prefix}_STALE")
    if minimum_rate_hz is not None and (rate_hz is None or rate_hz < minimum_rate_hz):
        reasons.append(f"{reason_prefix}_RATE_LOW")
    if not publishers:
        reasons.append(f"{reason_prefix}_PUBLISHER_MISSING")
    elif len(publishers) != 1:
        reasons.append(f"{reason_prefix}_PUBLISHER_AMBIGUOUS")
    health = {
        "topic": topic,
        "message_count": count,
        "age_s": age_s,
        "stale_after_s": stale_after_s,
        "rate_hz": rate_hz,
        "minimum_rate_hz": minimum_rate_hz,
        "frame_id": str(raw.get("frame_id") or ""),
        "fresh": count > 0 and age_s is not None and 0.0 <= age_s <= stale_after_s,
        "publishers": publishers,
        "publisher_unique": len(publishers) == 1,
    }
    return health, sorted(set(reasons))


def _topic_sample(topics: Mapping[str, Any], topic: str) -> Any:
    record = topics.get(topic, {})
    return record.get("sample") if isinstance(record, Mapping) else None


def _normalized_artifacts(raw_artifacts: Any) -> tuple[dict[str, Any], list[str]]:
    source = raw_artifacts if isinstance(raw_artifacts, Mapping) else {}
    result: dict[str, Any] = {}
    reasons: list[str] = []
    for name in sorted(ARTIFACT_SPECS):
        spec = ARTIFACT_SPECS[name]
        raw = source.get(name, {})
        raw = raw if isinstance(raw, Mapping) else {}
        present = raw.get("present") is True
        digest = raw.get("sha256") if isinstance(raw.get("sha256"), str) else None
        size = raw.get("size_bytes")
        size = int(size) if isinstance(size, int) and not isinstance(size, bool) and size >= 0 else None
        digest_valid = digest is not None and SHA256_RE.fullmatch(digest) is not None
        expected_match = (
            digest == spec.expected_sha256 if present and spec.expected_sha256 is not None else None
        )
        if spec.required and (not present or not digest_valid or size is None):
            reasons.append(f"ARTIFACT_{name.upper()}_MISSING")
        if present and not digest_valid:
            reasons.append(f"ARTIFACT_{name.upper()}_HASH_INVALID")
        if expected_match is False:
            reasons.append(f"ARTIFACT_{name.upper()}_HASH_MISMATCH")
        result[name] = {
            "path": spec.path,
            "required": spec.required,
            "present": present,
            "sha256": digest if digest_valid else None,
            "size_bytes": size,
            "expected_sha256": spec.expected_sha256,
            "expected_match": expected_match,
        }
    return result, sorted(set(reasons))


def _normalized_scan_filter_artifacts(
    raw_artifacts: Any,
) -> tuple[dict[str, Any], list[str]]:
    """Bind the exact filter implementation without changing the release inventory."""

    source = raw_artifacts if isinstance(raw_artifacts, Mapping) else {}
    result: dict[str, Any] = {}
    reasons: list[str] = []
    for name in sorted(SCAN_FILTER_ARTIFACT_SPECS):
        spec = SCAN_FILTER_ARTIFACT_SPECS[name]
        raw = source.get(name, {})
        raw = raw if isinstance(raw, Mapping) else {}
        present = raw.get("present") is True
        digest = raw.get("sha256") if isinstance(raw.get("sha256"), str) else None
        size = raw.get("size_bytes")
        size = int(size) if isinstance(size, int) and not isinstance(size, bool) and size > 0 else None
        digest_valid = digest is not None and SHA256_RE.fullmatch(digest) is not None
        expected_match = present and digest_valid and digest == spec.expected_sha256
        if not present or not digest_valid or size is None:
            reasons.append(f"SCAN_FILTER_ARTIFACT_{name.upper()}_MISSING")
        if present and not digest_valid:
            reasons.append(f"SCAN_FILTER_ARTIFACT_{name.upper()}_HASH_INVALID")
        if present and digest_valid and not expected_match:
            reasons.append(f"SCAN_FILTER_ARTIFACT_{name.upper()}_HASH_MISMATCH")
        result[name] = {
            "path": spec.path,
            "required": True,
            "present": present,
            "sha256": digest if digest_valid else None,
            "size_bytes": size,
            "expected_sha256": spec.expected_sha256,
            "expected_match": expected_match,
        }
    return result, sorted(set(reasons))


def _machine_snapshot(raw_machine: Any) -> tuple[dict[str, Any], list[str]]:
    raw = raw_machine if isinstance(raw_machine, Mapping) else {}
    hostname = str(raw.get("hostname") or "").lower()
    machine_id = str(raw.get("machine_id") or "").lower()
    boot_id = str(raw.get("boot_id") or "").lower()
    wlan_mac = str(raw.get("wlan_mac") or "").lower()
    invocation_id = str(raw.get("service_invocation_id") or "").lower()
    process_count = raw.get("service_process_count")
    process_count = (
        int(process_count)
        if isinstance(process_count, int) and not isinstance(process_count, bool) and process_count >= 0
        else 0
    )
    raw_owner = raw.get("service_owner") if isinstance(raw.get("service_owner"), Mapping) else {}
    raw_runtime = (
        raw_owner.get("scan_filter_runtime")
        if isinstance(raw_owner.get("scan_filter_runtime"), Mapping)
        else {}
    )
    runtime_sha256 = (
        str(raw_runtime.get("sha256"))
        if isinstance(raw_runtime.get("sha256"), str)
        and SHA256_RE.fullmatch(str(raw_runtime.get("sha256"))) is not None
        else None
    )
    runtime_size = raw_runtime.get("size_bytes")
    runtime_size = (
        int(runtime_size)
        if isinstance(runtime_size, int) and not isinstance(runtime_size, bool) and runtime_size > 0
        else None
    )
    runtime_process_count = raw_runtime.get("matching_process_count")
    runtime_process_count = (
        int(runtime_process_count)
        if isinstance(runtime_process_count, int)
        and not isinstance(runtime_process_count, bool)
        and runtime_process_count >= 0
        else 0
    )
    runtime_pid = raw_runtime.get("matched_pid")
    runtime_pid = (
        int(runtime_pid)
        if isinstance(runtime_pid, int) and not isinstance(runtime_pid, bool) and runtime_pid > 0
        else None
    )
    runtime_cmdline_sha256 = (
        str(raw_runtime.get("cmdline_sha256"))
        if isinstance(raw_runtime.get("cmdline_sha256"), str)
        and SHA256_RE.fullmatch(str(raw_runtime.get("cmdline_sha256"))) is not None
        else None
    )
    artifact_mtime_epoch_ms = raw_runtime.get("artifact_mtime_epoch_ms")
    artifact_mtime_epoch_ms = (
        int(artifact_mtime_epoch_ms)
        if isinstance(artifact_mtime_epoch_ms, int)
        and not isinstance(artifact_mtime_epoch_ms, bool)
        and artifact_mtime_epoch_ms > 0
        else None
    )
    process_start_epoch_ms = raw_runtime.get("process_start_epoch_ms")
    process_start_epoch_ms = (
        int(process_start_epoch_ms)
        if isinstance(process_start_epoch_ms, int)
        and not isinstance(process_start_epoch_ms, bool)
        and process_start_epoch_ms > 0
        else None
    )
    runtime = {
        "expected_path": str(raw_runtime.get("expected_path") or ""),
        "resolved_path": str(raw_runtime.get("resolved_path") or ""),
        "file_kind": str(raw_runtime.get("file_kind") or ""),
        "present": raw_runtime.get("present") is True,
        "sha256": runtime_sha256,
        "size_bytes": runtime_size,
        "expected_sha256": str(raw_runtime.get("expected_sha256") or ""),
        "hash_match": raw_runtime.get("hash_match") is True,
        "matching_process_count": runtime_process_count,
        "matched_pid": runtime_pid,
        "cmdline_sha256": runtime_cmdline_sha256,
        "exact_cmdline_match": raw_runtime.get("exact_cmdline_match") is True,
        "artifact_mtime_epoch_ms": artifact_mtime_epoch_ms,
        "process_start_epoch_ms": process_start_epoch_ms,
        "process_started_after_artifact": raw_runtime.get("process_started_after_artifact") is True,
    }
    raw_installed = (
        raw_owner.get("installed_release_artifacts")
        if isinstance(raw_owner.get("installed_release_artifacts"), Mapping)
        else {}
    )
    installed_release_artifacts: dict[str, dict[str, Any]] = {}
    for name in INSTALLED_RELEASE_ARTIFACT_SPECS:
        raw_record = raw_installed.get(name)
        raw_record = raw_record if isinstance(raw_record, Mapping) else {}
        installed_size = raw_record.get("size_bytes")
        installed_size = (
            int(installed_size)
            if isinstance(installed_size, int) and not isinstance(installed_size, bool) and installed_size > 0
            else None
        )
        installed_sha = str(raw_record.get("sha256") or "")
        installed_release_artifacts[name] = {
            "source_path": str(raw_record.get("source_path") or ""),
            "installed_path": str(raw_record.get("installed_path") or ""),
            "resolved_path": str(raw_record.get("resolved_path") or ""),
            "file_kind": str(raw_record.get("file_kind") or ""),
            "present": raw_record.get("present") is True,
            "sha256": installed_sha if SHA256_RE.fullmatch(installed_sha) else None,
            "size_bytes": installed_size,
            "expected_sha256": str(raw_record.get("expected_sha256") or ""),
            "hash_match": raw_record.get("hash_match") is True,
            "source_install_match": raw_record.get("source_install_match") is True,
        }
    owner = {
        "unit": str(raw_owner.get("unit") or ""),
        "cgroup_path": str(raw_owner.get("cgroup_path") or ""),
        "cgroup_files_found": raw_owner.get("cgroup_files_found"),
        "pid_count": raw_owner.get("pid_count"),
        "all_pids_in_unit_cgroup": raw_owner.get("all_pids_in_unit_cgroup") is True,
        "scan_filter_runtime": runtime,
        "installed_release_artifacts": installed_release_artifacts,
    }
    reasons = [str(item) for item in raw.get("collector_errors", []) if isinstance(item, str) and item]
    if hostname != EXPECTED_HOSTNAME:
        reasons.append("MACHINE_HOSTNAME_MISMATCH")
    if not machine_id:
        reasons.append("MACHINE_ID_MISSING")
    if BOOT_ID_RE.fullmatch(boot_id) is None:
        reasons.append("BOOT_ID_INVALID")
    if wlan_mac != EXPECTED_WLAN_MAC or MAC_RE.fullmatch(wlan_mac) is None:
        reasons.append("WLAN_IDENTITY_MISMATCH")
    if INVOCATION_ID_RE.fullmatch(invocation_id) is None or process_count <= 0:
        reasons.append("SERVICE_SESSION_INVALID")
    if not (
        owner["unit"] == SERVICE_NAME
        and owner["cgroup_path"]
        .replace("\\", "/")
        .endswith("/system.slice/embodied_brain.service/cgroup.procs")
        and owner["cgroup_files_found"] == 1
        and owner["pid_count"] == process_count
        and process_count > 0
        and owner["all_pids_in_unit_cgroup"] is True
    ):
        reasons.append("SERVICE_CGROUP_OWNER_INVALID")
    if not (
        runtime["expected_path"] == SCAN_FILTER_INSTALLED_PATH
        and runtime["file_kind"] in {"regular_install", "symlink_to_source"}
        and (
            (
                runtime["file_kind"] == "symlink_to_source"
                and runtime["resolved_path"] == SCAN_FILTER_SOURCE_PATH
            )
            or (
                runtime["file_kind"] == "regular_install"
                and runtime["resolved_path"] == SCAN_FILTER_INSTALLED_PATH
            )
        )
        and runtime["present"] is True
        and runtime["sha256"] == FROZEN_SCAN_SELF_FILTER_SHA256
        and runtime["expected_sha256"] == FROZEN_SCAN_SELF_FILTER_SHA256
        and runtime["hash_match"] is True
        and runtime["size_bytes"] is not None
        and runtime["matching_process_count"] == 1
        and runtime["matched_pid"] is not None
        and runtime["cmdline_sha256"] is not None
        and runtime["exact_cmdline_match"] is True
        and runtime["artifact_mtime_epoch_ms"] is not None
        and runtime["process_start_epoch_ms"] is not None
        and runtime["process_started_after_artifact"] is True
    ):
        reasons.append("SCAN_FILTER_RUNTIME_PROCESS_INVALID")
    for name, spec in INSTALLED_RELEASE_ARTIFACT_SPECS.items():
        record = installed_release_artifacts[name]
        if not (
            record["source_path"] == spec.source_path
            and record["installed_path"] == spec.installed_path
            and record["file_kind"] in {"regular_install", "symlink_to_source"}
            and (
                (record["file_kind"] == "symlink_to_source" and record["resolved_path"] == spec.source_path)
                or (
                    record["file_kind"] == "regular_install"
                    and record["resolved_path"] == spec.installed_path
                )
            )
            and record["present"] is True
            and record["sha256"] == spec.expected_sha256
            and record["expected_sha256"] == spec.expected_sha256
            and record["hash_match"] is True
            and record["source_install_match"] is True
            and record["size_bytes"] is not None
        ):
            reasons.append(f"INSTALLED_RELEASE_ARTIFACT_{name.upper()}_INVALID")
    machine_id_sha256 = _sha256_text(machine_id) if machine_id else "0" * 64
    identity_payload = {
        "hostname": hostname,
        "machine_id_sha256": machine_id_sha256,
        "wlan_mac": wlan_mac,
    }
    device_id = f"embodied-x5:{canonical_sha256(identity_payload)[:32]}"
    session_id = f"systemd:{canonical_sha256({'boot_id': boot_id, 'invocation_id': invocation_id})[:32]}"
    return (
        {
            "device_id": device_id,
            "hostname": hostname,
            "machine_id_sha256": machine_id_sha256,
            "boot_id": boot_id,
            "service_invocation_id": invocation_id,
            "service_process_count": process_count,
            "service_owner": owner,
            "session_id": session_id,
            "wlan_mac": wlan_mac,
        },
        sorted(set(reasons)),
    )


def _diagnostic_status(sample: Any, suffix: str) -> dict[str, Any] | None:
    if not isinstance(sample, list):
        return None
    matches = [
        row for row in sample if isinstance(row, Mapping) and str(row.get("name") or "").endswith(suffix)
    ]
    if len(matches) != 1:
        return None
    row = matches[0]
    values = row.get("values") if isinstance(row.get("values"), Mapping) else {}
    return {
        "name": str(row.get("name") or ""),
        "hardware_id": str(row.get("hardware_id") or ""),
        "level": int(row.get("level")) if isinstance(row.get("level"), int) else -1,
        "message": str(row.get("message") or ""),
        "values": {str(key): str(value) for key, value in sorted(values.items())},
    }


def _parse_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _probability_vector(value: Any, *, length: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) != length:
        return None
    result: list[float] = []
    for item in value:
        number = _parse_float(item)
        if number is None or not 0.0 <= number <= 1.0:
            return None
        result.append(round(number, 9))
    if abs(sum(result) - 1.0) > 0.02:
        return None
    return result


def _finite_twist(value: Any) -> dict[str, float] | None:
    if not isinstance(value, Mapping):
        return None
    names = (
        "linear_x",
        "linear_y",
        "linear_z",
        "angular_x",
        "angular_y",
        "angular_z",
    )
    result: dict[str, float] = {}
    for name in names:
        number = _parse_float(value.get(name))
        if number is None:
            return None
        result[name] = round(number, 9)
    return result


def _capability(ready: bool, backend: str, reasons: list[str]) -> dict[str, Any]:
    return {"ready": ready, "backend": backend, "reason_codes": sorted(set(reasons))}


def build_snapshot(
    *,
    run_id: str,
    run_nonce: str,
    release_id: str,
    profile_sha256: str,
    observed_at_ms: int,
    machine: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    ros: Mapping[str, Any],
    localization_mode: str = LOCALIZATION_ONLINE_SLAM,
) -> dict[str, Any]:
    """Build and sign one exact-schema snapshot from normalized observations."""

    _validate_binding(run_id, run_nonce, release_id, profile_sha256)
    if localization_mode not in LOCALIZATION_MODES:
        raise ValueError("unsupported localization_mode")
    if isinstance(observed_at_ms, bool) or not isinstance(observed_at_ms, int) or observed_at_ms < 0:
        raise ValueError("observed_at_ms must be a non-negative integer")

    machine_out, machine_reasons = _machine_snapshot(machine)
    artifacts_out, artifact_reasons = _normalized_artifacts(artifacts)
    scan_filter_artifacts, scan_filter_artifact_reasons = _normalized_scan_filter_artifacts(artifacts)
    topics = ros.get("topics", {}) if isinstance(ros.get("topics"), Mapping) else {}
    graph = ros.get("graph", {}) if isinstance(ros.get("graph"), Mapping) else {}
    graph_publishers = graph.get("publishers", {}) if isinstance(graph.get("publishers"), Mapping) else {}
    graph_subscribers = graph.get("subscribers", {}) if isinstance(graph.get("subscribers"), Mapping) else {}
    lifecycle = ros.get("lifecycle", {}) if isinstance(ros.get("lifecycle"), Mapping) else {}
    parameters = ros.get("parameters", {}) if isinstance(ros.get("parameters"), Mapping) else {}
    transforms = ros.get("transforms", {}) if isinstance(ros.get("transforms"), Mapping) else {}

    collector_reasons = [
        str(item) for item in ros.get("collector_errors", []) if isinstance(item, str) and item
    ]
    raw_clock = ros.get("clock") if isinstance(ros.get("clock"), Mapping) else {}
    ros_time_ns = raw_clock.get("ros_time_ns")
    system_time_ns = raw_clock.get("system_time_ns")
    wall_time_delta_s = _parse_float(raw_clock.get("wall_time_delta_s"))
    if not (
        raw_clock.get("use_sim_time") is False
        and isinstance(ros_time_ns, int)
        and not isinstance(ros_time_ns, bool)
        and ros_time_ns > 0
        and isinstance(system_time_ns, int)
        and not isinstance(system_time_ns, bool)
        and system_time_ns > 0
        and wall_time_delta_s is not None
        and 0.0 <= wall_time_delta_s <= 2.0
    ):
        collector_reasons.append("ROS_SIM_TIME_FORBIDDEN")

    scan_raw, scan_raw_reasons = _topic_health(
        topics,
        "/scan_raw",
        stale_after_s=1.0,
        minimum_rate_hz=5.0,
        reason_prefix="SENSOR_SCAN_RAW",
    )
    scan, scan_reasons = _topic_health(
        topics,
        "/scan",
        stale_after_s=1.0,
        minimum_rate_hz=5.0,
        reason_prefix="SENSOR_SCAN",
    )
    scan_depth, scan_depth_reasons = _topic_health(
        topics,
        "/scan_depth",
        stale_after_s=1.0,
        minimum_rate_hz=10.0,
        reason_prefix="SENSOR_SCAN_DEPTH",
    )
    wheel_odom, wheel_odom_reasons = _topic_health(
        topics,
        "/wheel_odom",
        stale_after_s=1.0,
        minimum_rate_hz=10.0,
        reason_prefix="STATE_ESTIMATOR_WHEEL_ODOM",
    )
    imu, imu_reasons = _topic_health(
        topics,
        "/imu",
        stale_after_s=1.0,
        minimum_rate_hz=10.0,
        reason_prefix="STATE_ESTIMATOR_IMU",
    )
    odom, odom_reasons = _topic_health(
        topics,
        "/odom",
        stale_after_s=1.5,
        minimum_rate_hz=5.0,
        reason_prefix="SENSOR_ODOM",
    )
    graph_nodes = [str(item) for item in graph.get("nodes", []) if isinstance(item, str)]
    filter_nodes = sorted(
        name for name in set(graph_nodes) if _node_basename(name) == EXPECTED_SCAN_FILTER_NODE
    )
    raw_filter_subscribers = [
        endpoint
        for endpoint in _normalized_endpoints(graph_subscribers.get("/scan_raw"))
        if _node_basename(endpoint["node"]) == EXPECTED_SCAN_FILTER_NODE
    ]
    scan_filter_reasons: list[str] = list(scan_filter_artifact_reasons)
    if not (
        len(scan_raw["publishers"]) == 1 and _has_node(scan_raw["publishers"], EXPECTED_SCAN_RAW_PUBLISHER)
    ):
        scan_filter_reasons.append("SCAN_RAW_PUBLISHER_IDENTITY_INVALID")
    if not (len(scan["publishers"]) == 1 and _has_node(scan["publishers"], EXPECTED_SCAN_FILTER_NODE)):
        scan_filter_reasons.append("SCAN_FILTER_PUBLISHER_IDENTITY_INVALID")
    if len(filter_nodes) != 1 or len(raw_filter_subscribers) != 1:
        scan_filter_reasons.append("SCAN_FILTER_TOPOLOGY_INVALID")
    if not (scan_raw["frame_id"] == EXPECTED_SCAN_FRAME_ID and scan["frame_id"] == EXPECTED_SCAN_FRAME_ID):
        scan_filter_reasons.append("SCAN_FILTER_FRAME_MISMATCH")
    scan_filter_runtime = machine_out["service_owner"]["scan_filter_runtime"]
    if not (
        scan_filter_runtime["expected_path"] == SCAN_FILTER_INSTALLED_PATH
        and (
            (
                scan_filter_runtime["file_kind"] == "symlink_to_source"
                and scan_filter_runtime["resolved_path"] == SCAN_FILTER_SOURCE_PATH
            )
            or (
                scan_filter_runtime["file_kind"] == "regular_install"
                and scan_filter_runtime["resolved_path"] == SCAN_FILTER_INSTALLED_PATH
            )
        )
        and scan_filter_runtime["present"] is True
        and scan_filter_runtime["sha256"] == FROZEN_SCAN_SELF_FILTER_SHA256
        and scan_filter_runtime["hash_match"] is True
        and scan_filter_runtime["matching_process_count"] == 1
        and scan_filter_runtime["exact_cmdline_match"] is True
        and scan_filter_runtime["process_started_after_artifact"] is True
    ):
        scan_filter_reasons.append("SCAN_FILTER_RUNTIME_PROCESS_INVALID")
    installed_release_artifacts = machine_out["service_owner"]["installed_release_artifacts"]
    for name, spec in INSTALLED_RELEASE_ARTIFACT_SPECS.items():
        record = installed_release_artifacts[name]
        if not (
            record["source_path"] == spec.source_path
            and record["installed_path"] == spec.installed_path
            and record["present"] is True
            and record["sha256"] == spec.expected_sha256
            and record["expected_sha256"] == spec.expected_sha256
            and record["hash_match"] is True
            and record["source_install_match"] is True
        ):
            scan_filter_reasons.append(f"INSTALLED_RELEASE_ARTIFACT_{name.upper()}_INVALID")

    raw_scan_filter = ros.get("scan_filter", {})
    raw_scan_filter = raw_scan_filter if isinstance(raw_scan_filter, Mapping) else {}
    raw_contour = raw_scan_filter.get("body_contour", {})
    raw_contour = raw_contour if isinstance(raw_contour, Mapping) else {}
    contour_sha = str(raw_contour.get("sha256") or "")
    contour_uncertainty = _parse_float(raw_contour.get("measurement_uncertainty_m"))
    contour_polygon, contour_geometry, contour_geometry_reasons = _body_contour_geometry(
        raw_contour.get("polygon_xy_m"),
        contour_uncertainty,
    )
    contour_measured_at = str(raw_contour.get("measured_at_utc") or "")
    contour_attachment_sha = str(raw_contour.get("measurement_attachment_sha256") or "")
    contour_collision_sha = str(raw_contour.get("collision_monitor_config_sha256") or "")
    contour_nav2_sha = str(raw_contour.get("nav2_params_config_sha256") or "")
    contour_summary_sha = str(raw_contour.get("safety_config_summary_sha256") or "")
    expected_summary_sha = _safety_config_summary_sha256(
        FROZEN_COLLISION_MONITOR_SHA256,
        FROZEN_NAV2_PARAMS_SHA256,
    )
    contour_claim_valid = (
        raw_contour.get("present") is True
        and raw_contour.get("valid") is True
        and raw_contour.get("path") == BODY_CONTOUR_ARTIFACT_PATH
        and SHA256_RE.fullmatch(contour_sha) is not None
        and raw_contour.get("schema_version") == BODY_CONTOUR_SCHEMA
        and raw_contour.get("verification_status") == "MEASURED_AND_FROZEN"
        and raw_contour.get("source") == "physical_measurement"
        and raw_contour.get("frame_id") == "base_footprint"
        and TOKEN_RE.fullmatch(str(raw_contour.get("measurement_id") or "")) is not None
        and contour_measured_at.endswith("Z")
        and len(contour_measured_at) >= 20
        and contour_uncertainty is not None
        and 0.0 <= contour_uncertainty <= 0.05
        and contour_polygon is not None
        and not contour_geometry_reasons
        and raw_contour.get("measurement_attachment_path") == BODY_CONTOUR_MEASUREMENT_PATH
        and SHA256_RE.fullmatch(contour_attachment_sha) is not None
        and raw_contour.get("collision_monitor_config_path") == COLLISION_MONITOR_CONFIG_PATH
        and contour_collision_sha == FROZEN_COLLISION_MONITOR_SHA256
        and _parse_float(raw_contour.get("body_stop_radius_m")) == FROZEN_BODY_STOP_RADIUS_M
        and raw_contour.get("nav2_params_config_path") == NAV2_PARAMS_CONFIG_PATH
        and contour_nav2_sha == FROZEN_NAV2_PARAMS_SHA256
        and _parse_float(raw_contour.get("nav2_robot_radius_m")) == FROZEN_NAV2_ROBOT_RADIUS_M
        and contour_summary_sha == expected_summary_sha
        and raw_contour.get("reason_codes") == []
    )
    contour_artifact = artifacts_out["body_contour"]
    contour_release_match = (
        contour_artifact["required"] is True
        and contour_artifact["present"] is True
        and contour_artifact["path"] == BODY_CONTOUR_ARTIFACT_PATH
        and contour_artifact["sha256"] == contour_sha
        and isinstance(contour_artifact["size_bytes"], int)
        and contour_artifact["size_bytes"] > 0
    )
    contour_measurement_artifact = artifacts_out["body_contour_measurement"]
    contour_measurement_match = (
        contour_measurement_artifact["required"] is True
        and contour_measurement_artifact["present"] is True
        and contour_measurement_artifact["path"] == BODY_CONTOUR_MEASUREMENT_PATH
        and contour_measurement_artifact["sha256"] == contour_attachment_sha
        and isinstance(contour_measurement_artifact["size_bytes"], int)
        and contour_measurement_artifact["size_bytes"] > 0
    )
    collision_config_artifact = artifacts_out["collision_monitor_config"]
    nav2_config_artifact = artifacts_out["nav2_params"]
    contour_safety_config_match = (
        collision_config_artifact["present"] is True
        and collision_config_artifact["expected_match"] is True
        and collision_config_artifact["sha256"] == contour_collision_sha
        and nav2_config_artifact["present"] is True
        and nav2_config_artifact["expected_match"] is True
        and nav2_config_artifact["sha256"] == contour_nav2_sha
    )
    contour_valid = (
        contour_claim_valid
        and contour_release_match
        and contour_measurement_match
        and contour_safety_config_match
    )
    if not contour_claim_valid:
        scan_filter_reasons.append("BODY_CONTOUR_VERIFIED_ARTIFACT_INVALID")
    scan_filter_reasons.extend(contour_geometry_reasons)
    if not contour_release_match:
        scan_filter_reasons.append("BODY_CONTOUR_RELEASE_ARTIFACT_MISMATCH")
    if not contour_measurement_match:
        scan_filter_reasons.append("BODY_CONTOUR_MEASUREMENT_ATTACHMENT_MISMATCH")
    if not contour_safety_config_match:
        scan_filter_reasons.append("BODY_CONTOUR_SAFETY_CONFIG_MISMATCH")

    def nonnegative_integer(source: Mapping[str, Any], name: str) -> int | None:
        value = source.get(name)
        return int(value) if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None

    def scan_filter_integer(name: str) -> int | None:
        return nonnegative_integer(raw_scan_filter, name)

    raw_valid_count = scan_filter_integer("raw_valid_count")
    filtered_valid_count = scan_filter_integer("filtered_valid_count")
    scan_sample_count = scan_filter_integer("sample_count")
    removed_count = scan_filter_integer("removed_count")
    positive_infinity_removed_count = scan_filter_integer("positive_infinity_removed_count")
    unremoved_inside_count = scan_filter_integer("unremoved_inside_contour_count")
    modified_count = scan_filter_integer("modified_or_inserted_count")
    removed_outside = scan_filter_integer("removed_outside_contour_count")
    angular_coverage = _parse_float(raw_scan_filter.get("angular_coverage_rad"))
    paired_stamp_ns = scan_filter_integer("paired_stamp_ns")
    paired_stamp_age_s = _parse_float(raw_scan_filter.get("paired_stamp_age_s"))
    if (
        raw_scan_filter.get("available") is not True
        or raw_scan_filter.get("raw_topic") != "/scan_raw"
        or raw_scan_filter.get("filtered_topic") != "/scan"
    ):
        scan_filter_reasons.append("SCAN_FILTER_EVIDENCE_UNAVAILABLE")
    if (
        raw_valid_count is None
        or filtered_valid_count is None
        or raw_valid_count < MIN_SCAN_VALID_POINTS
        or filtered_valid_count < MIN_SCAN_VALID_POINTS
        or scan_sample_count is None
        or scan_sample_count < max(raw_valid_count or 0, filtered_valid_count or 0)
        or angular_coverage is None
        or angular_coverage < MIN_SCAN_COVERAGE_RAD
    ):
        scan_filter_reasons.append("SCAN_FILTER_VALID_POINTS_OR_COVERAGE_INVALID")
    if raw_scan_filter.get("geometry_valid") is not True or raw_scan_filter.get("geometry_match") is not True:
        scan_filter_reasons.append("SCAN_FILTER_GEOMETRY_MISMATCH")
    if raw_scan_filter.get("exact_stamp_match") is not True or paired_stamp_ns is None:
        scan_filter_reasons.append("SCAN_FILTER_EXACT_STAMP_PAIR_UNAVAILABLE")
    if raw_scan_filter.get("transform_match") is not True:
        scan_filter_reasons.append("SCAN_FILTER_TF_MISMATCH")
    if (
        raw_scan_filter.get("paired_stamp_fresh") is not True
        or paired_stamp_age_s is None
        or not -MAX_SCAN_PAIR_FUTURE_S <= paired_stamp_age_s <= MAX_SCAN_PAIR_AGE_S
    ):
        scan_filter_reasons.append("SCAN_FILTER_EXACT_STAMP_PAIR_STALE")
    if (
        removed_count is None
        or positive_infinity_removed_count != removed_count
        or unremoved_inside_count != 0
        or modified_count != 0
        or removed_outside != 0
        or raw_scan_filter.get("invalid_ranges_preserved") is not True
        or raw_scan_filter.get("intensities_preserved") is not True
        or raw_scan_filter.get("all_removed_points_inside_contour") is not True
    ):
        scan_filter_reasons.append("SCAN_FILTER_DIFF_OUTSIDE_FROZEN_CONTOUR")

    transform_out: dict[str, dict[str, Any]] = {}
    for key in (
        "map_to_odom",
        "odom_to_base_footprint",
        "base_to_scan_raw",
        "base_to_scan",
        "base_to_scan_depth",
    ):
        raw_transform = transforms.get(key, {})
        raw_transform = raw_transform if isinstance(raw_transform, Mapping) else {}
        available = raw_transform.get("available") is True
        transform_out[key] = {
            "available": available,
            "parent": str(raw_transform.get("parent") or ""),
            "child": str(raw_transform.get("child") or ""),
            "query_stamp_ns": nonnegative_integer(raw_transform, "query_stamp_ns"),
            "age_s": _round_number(raw_transform.get("age_s"), 6),
        }
    for key in ("base_to_scan_raw", "base_to_scan"):
        transform = transform_out[key]
        if not (
            transform["available"]
            and transform["parent"] == "base_footprint"
            and transform["child"] == EXPECTED_SCAN_FRAME_ID
            and transform["query_stamp_ns"] == paired_stamp_ns
        ):
            scan_filter_reasons.append(f"TF_{key.upper()}_INVALID")
    scan_filter_reasons = sorted(set(scan_filter_reasons))
    scan_filter_out = {
        "ready": not scan_filter_reasons,
        "raw_topic": "/scan_raw",
        "filtered_topic": "/scan",
        "implementation_artifacts": scan_filter_artifacts,
        "installed_release_artifacts": installed_release_artifacts,
        "runtime_process": scan_filter_runtime,
        "raw_publisher": scan_raw["publishers"],
        "filter_nodes": filter_nodes,
        "raw_filter_subscribers": raw_filter_subscribers,
        "raw_valid_count": raw_valid_count,
        "filtered_valid_count": filtered_valid_count,
        "sample_count": scan_sample_count,
        "angular_coverage_rad": _round_number(angular_coverage, 6),
        "exact_stamp_match": raw_scan_filter.get("exact_stamp_match") is True,
        "paired_stamp_ns": paired_stamp_ns,
        "paired_stamp_age_s": _round_number(paired_stamp_age_s, 6),
        "paired_stamp_fresh": raw_scan_filter.get("paired_stamp_fresh") is True,
        "geometry_valid": raw_scan_filter.get("geometry_valid") is True,
        "geometry_match": raw_scan_filter.get("geometry_match") is True,
        "transform_match": raw_scan_filter.get("transform_match") is True,
        "removed_count": removed_count,
        "positive_infinity_removed_count": positive_infinity_removed_count,
        "unremoved_inside_contour_count": unremoved_inside_count,
        "modified_or_inserted_count": modified_count,
        "removed_outside_contour_count": removed_outside,
        "invalid_ranges_preserved": raw_scan_filter.get("invalid_ranges_preserved") is True,
        "intensities_preserved": raw_scan_filter.get("intensities_preserved") is True,
        "all_removed_points_inside_contour": (
            raw_scan_filter.get("all_removed_points_inside_contour") is True
        ),
        "body_contour": {
            "valid": contour_valid,
            "release_artifact_match": contour_release_match,
            "measurement_attachment_match": contour_measurement_match,
            "safety_config_match": contour_safety_config_match,
            "path": str(raw_contour.get("path") or ""),
            "sha256": contour_sha if SHA256_RE.fullmatch(contour_sha) else None,
            "schema_version": str(raw_contour.get("schema_version") or ""),
            "verification_status": str(raw_contour.get("verification_status") or ""),
            "source": str(raw_contour.get("source") or ""),
            "frame_id": str(raw_contour.get("frame_id") or ""),
            "measurement_id": str(raw_contour.get("measurement_id") or ""),
            "measured_at_utc": contour_measured_at,
            "measurement_uncertainty_m": _round_number(contour_uncertainty, 6),
            "measurement_attachment_path": str(raw_contour.get("measurement_attachment_path") or ""),
            "measurement_attachment_sha256": (
                contour_attachment_sha if SHA256_RE.fullmatch(contour_attachment_sha) else None
            ),
            "collision_monitor_config_path": str(raw_contour.get("collision_monitor_config_path") or ""),
            "collision_monitor_config_sha256": (
                contour_collision_sha if SHA256_RE.fullmatch(contour_collision_sha) else None
            ),
            "body_stop_radius_m": _round_number(raw_contour.get("body_stop_radius_m"), 6),
            "nav2_params_config_path": str(raw_contour.get("nav2_params_config_path") or ""),
            "nav2_params_config_sha256": (
                contour_nav2_sha if SHA256_RE.fullmatch(contour_nav2_sha) else None
            ),
            "nav2_robot_radius_m": _round_number(
                raw_contour.get("nav2_robot_radius_m"),
                6,
            ),
            "safety_config_summary_sha256": (
                contour_summary_sha if SHA256_RE.fullmatch(contour_summary_sha) else None
            ),
            "polygon_xy_m": contour_polygon,
            "geometry": contour_geometry,
        },
        "reason_codes": scan_filter_reasons,
    }

    state_estimator_reasons = wheel_odom_reasons + imu_reasons + odom_reasons
    ekf_nodes = sorted(name for name in set(graph_nodes) if _node_basename(name) == "ekf_filter_node")
    fake_odom_nodes = sorted(name for name in set(graph_nodes) if "fake_odom" in _node_basename(name).lower())
    if len(ekf_nodes) != 1:
        state_estimator_reasons.append("STATE_ESTIMATOR_EKF_NODE_MISSING_OR_AMBIGUOUS")
    if fake_odom_nodes:
        state_estimator_reasons.append("STATE_ESTIMATOR_FAKE_ODOM_PRESENT")
    if not (len(wheel_odom["publishers"]) == 1 and _has_node(wheel_odom["publishers"], "serial_f407")):
        state_estimator_reasons.append("STATE_ESTIMATOR_WHEEL_ODOM_SOURCE_INVALID")
    if not (len(imu["publishers"]) == 1 and _has_node(imu["publishers"], "serial_f407")):
        state_estimator_reasons.append("STATE_ESTIMATOR_IMU_SOURCE_INVALID")
    if not (len(odom["publishers"]) == 1 and _has_node(odom["publishers"], "ekf_filter_node")):
        state_estimator_reasons.append("STATE_ESTIMATOR_ODOM_SOURCE_INVALID")
    wheel_sample = _topic_sample(topics, "/wheel_odom")
    wheel_sample = wheel_sample if isinstance(wheel_sample, Mapping) else {}
    odom_sample = _topic_sample(topics, "/odom")
    odom_sample = odom_sample if isinstance(odom_sample, Mapping) else {}
    if not (
        wheel_sample.get("frame_id") == "odom"
        and wheel_sample.get("child_frame_id") == "base_footprint"
        and odom_sample.get("frame_id") == "odom"
        and odom_sample.get("child_frame_id") == "base_footprint"
        and imu["frame_id"]
    ):
        state_estimator_reasons.append("STATE_ESTIMATOR_FRAME_CONTRACT_INVALID")
    for topic in ("/wheel_odom", "/imu"):
        if not _has_node(graph_subscribers.get(topic), "ekf_filter_node"):
            state_estimator_reasons.append(f"STATE_ESTIMATOR_{topic.strip('/').upper()}_EKF_INPUT_MISSING")
    odom_transform = transform_out["odom_to_base_footprint"]
    if not (
        odom_transform["available"]
        and odom_transform["parent"] == "odom"
        and odom_transform["child"] == "base_footprint"
        and odom_transform["age_s"] is not None
        and 0.0 <= odom_transform["age_s"] <= 1.5
    ):
        state_estimator_reasons.append("STATE_ESTIMATOR_ODOM_TF_INVALID")
    state_estimator_reasons = sorted(set(state_estimator_reasons))
    state_estimator_out = {
        "ready": not state_estimator_reasons,
        "nodes": ekf_nodes,
        "fake_odom_nodes": fake_odom_nodes,
        "wheel_odom": wheel_odom,
        "imu": imu,
        "odom": odom,
        "reason_codes": state_estimator_reasons,
    }

    sensor_reasons = sorted(
        set(
            scan_raw_reasons
            + scan_reasons
            + scan_depth_reasons
            + scan_filter_reasons
            + state_estimator_reasons
        )
    )
    sensors_ready = not sensor_reasons
    sensors_out = {
        "ready": sensors_ready,
        "scan_raw": scan_raw,
        "scan": scan,
        "scan_depth": scan_depth,
        "state_estimator": state_estimator_out,
        "scan_filter": scan_filter_out,
        "odom": odom,
        "reason_codes": sensor_reasons,
    }

    map_health, map_reasons = _topic_health(
        topics,
        "/map",
        stale_after_s=3.0,
        minimum_rate_hz=None,
        reason_prefix="LOCALIZATION_MAP",
    )
    observation_mode = str(ros.get("localization_mode") or "")
    localization_reasons = list(map_reasons)
    if observation_mode != localization_mode:
        localization_reasons.append("LOCALIZATION_MODE_OBSERVATION_MISMATCH")
    map_to_odom = transform_out["map_to_odom"]
    if not (
        map_to_odom["available"]
        and map_to_odom["parent"] == "map"
        and map_to_odom["child"] == "odom"
        and map_to_odom["age_s"] is not None
        and 0.0 <= map_to_odom["age_s"] <= 2.0
    ):
        localization_reasons.append("LOCALIZATION_MAP_TO_ODOM_TF_INVALID")
    slam_nodes = sorted(name for name in set(graph_nodes) if _node_basename(name) == "slam_toolbox")
    map_server_nodes = sorted(name for name in set(graph_nodes) if _node_basename(name) == "map_server")
    amcl_nodes = sorted(name for name in set(graph_nodes) if _node_basename(name) == "amcl")
    amcl_health: dict[str, Any] | None = None
    if localization_mode == LOCALIZATION_ONLINE_SLAM:
        if not (
            len(slam_nodes) == 1
            and len(map_health["publishers"]) == 1
            and _has_node(map_health["publishers"], "slam_toolbox")
        ):
            localization_reasons.append("LOCALIZATION_ONLINE_SLAM_SOURCE_INVALID")
        if str(lifecycle.get("slam_toolbox") or "").lower() != "active":
            localization_reasons.append("LOCALIZATION_SLAM_TOOLBOX_NOT_ACTIVE")
        if map_server_nodes or amcl_nodes:
            localization_reasons.append("LOCALIZATION_MODE_MIXED")
    else:
        amcl_health, amcl_reasons = _topic_health(
            topics,
            "/amcl_pose",
            stale_after_s=2.0,
            minimum_rate_hz=None,
            reason_prefix="LOCALIZATION_AMCL",
        )
        localization_reasons.extend(amcl_reasons)
        if not (
            len(map_server_nodes) == 1
            and len(map_health["publishers"]) == 1
            and _has_node(map_health["publishers"], "map_server")
        ):
            localization_reasons.append("LOCALIZATION_SAVED_MAP_SOURCE_INVALID")
        if not (
            len(amcl_nodes) == 1
            and len(amcl_health["publishers"]) == 1
            and _has_node(amcl_health["publishers"], "amcl")
        ):
            localization_reasons.append("LOCALIZATION_AMCL_SOURCE_INVALID")
        for name in ("map_server", "amcl"):
            if str(lifecycle.get(name) or "").lower() != "active":
                localization_reasons.append(f"LOCALIZATION_{name.upper()}_NOT_ACTIVE")
        if slam_nodes:
            localization_reasons.append("LOCALIZATION_MODE_MIXED")
        for artifact_name in ("saved_map_yaml", "saved_map_pgm"):
            record = artifacts_out[artifact_name]
            if not record["present"] or record["expected_match"] is not True:
                localization_reasons.append(f"LOCALIZATION_{artifact_name.upper()}_INVALID")
    localization_reasons = sorted(set(localization_reasons))
    localization_ready = not localization_reasons
    localization_out = {
        "ready": localization_ready,
        "mode": localization_mode,
        "observed_mode": observation_mode,
        "mode_binding_sha256": canonical_sha256(
            {
                "run_binding_sha256": source_binding_sha256(
                    run_id=run_id,
                    run_nonce=run_nonce,
                    release_id=release_id,
                    profile_sha256=profile_sha256,
                ),
                "localization_mode": localization_mode,
            }
        ),
        "map": map_health,
        "amcl_pose": amcl_health,
        "slam_nodes": slam_nodes,
        "map_server_nodes": map_server_nodes,
        "amcl_nodes": amcl_nodes,
        "lifecycle": {
            "slam_toolbox": str(lifecycle.get("slam_toolbox") or "unavailable").lower(),
            "map_server": str(lifecycle.get("map_server") or "unavailable").lower(),
            "amcl": str(lifecycle.get("amcl") or "unavailable").lower(),
        },
        "transforms": transform_out,
        "reason_codes": localization_reasons,
    }

    f407_identity_topic, f407_identity_reasons = _topic_health(
        topics,
        "/f407/firmware_identity_valid",
        stale_after_s=3.0,
        minimum_rate_hz=None,
        reason_prefix="F407_IDENTITY_TOPIC",
    )
    f407_estop_topic, f407_estop_reasons = _topic_health(
        topics,
        "/f407/estop_latched",
        stale_after_s=3.0,
        minimum_rate_hz=None,
        reason_prefix="F407_ESTOP_TOPIC",
    )
    f407_firmware_topic, f407_firmware_reasons = _topic_health(
        topics,
        "/f407/firmware_info",
        stale_after_s=3.0,
        minimum_rate_hz=None,
        reason_prefix="F407_FIRMWARE_TOPIC",
    )
    diagnostics_topic, diagnostics_reasons = _topic_health(
        topics,
        "/diagnostics",
        stale_after_s=2.0,
        minimum_rate_hz=None,
        reason_prefix="F407_DIAGNOSTICS_TOPIC",
    )
    f407_reasons = f407_identity_reasons + f407_estop_reasons + f407_firmware_reasons + diagnostics_reasons
    for health in (f407_identity_topic, f407_estop_topic, f407_firmware_topic):
        if not _has_node(health["publishers"], "serial_f407"):
            f407_reasons.append("F407_PUBLISHER_IDENTITY_INVALID")
    identity_valid = _topic_sample(topics, "/f407/firmware_identity_valid") is True
    estop_latched = _topic_sample(topics, "/f407/estop_latched") is True
    cmd_vel_expired_sample = _topic_sample(topics, "/f407/cmd_vel_expired")
    firmware_sample = _topic_sample(topics, "/f407/firmware_info")
    firmware_sample = firmware_sample if isinstance(firmware_sample, Mapping) else {}
    if not identity_valid:
        f407_reasons.append("F407_FIRMWARE_IDENTITY_INVALID")
    if not estop_latched:
        f407_reasons.append("F407_ESTOP_NOT_LATCHED_FOR_READONLY_PREP")
    firmware_age = _parse_float(firmware_sample.get("age_s"))
    if not (
        firmware_sample.get("protocol_version") == 2
        and firmware_sample.get("test_mode") == 0
        and firmware_sample.get("hw_variant") == 1
        and firmware_sample.get("identity_valid") is True
        and firmware_sample.get("required") is True
        and firmware_sample.get("identity_enforcement_enabled") is True
        and firmware_sample.get("cmd_vel_authority_when_invalid") is False
        and firmware_age is not None
        and 0.0 <= firmware_age <= 3.0
    ):
        f407_reasons.append("F407_FIRMWARE_CONTRACT_INVALID")
    if not _has_node(diagnostics_topic["publishers"], "serial_f407"):
        f407_reasons.append("F407_DIAGNOSTICS_PUBLISHER_IDENTITY_INVALID")
    diagnostics_sample = _topic_sample(topics, "/diagnostics")
    serial_link_diag = _diagnostic_status(diagnostics_sample, "serial_link")
    safety_diag = _diagnostic_status(diagnostics_sample, "safety_bridge")
    hardware_estop_latched = False
    safety_state_age: float | None = None
    if serial_link_diag is None or safety_diag is None:
        f407_reasons.append("F407_DIAGNOSTICS_MISSING_OR_AMBIGUOUS")
    else:
        safety_values = safety_diag["values"]
        serial_values = serial_link_diag["values"]
        rx_age = _parse_float(serial_values.get("rx_age_s"))
        if rx_age is None or rx_age < 0.0 or rx_age > 1.5:
            f407_reasons.append("F407_SERIAL_RX_STALE")
        hardware_estop_latched = safety_values.get("hardware_estop_latched") == "true"
        safety_state_age = _parse_float(safety_values.get("last_safety_state_age_s"))
        if not hardware_estop_latched:
            f407_reasons.append("F407_HARDWARE_ESTOP_NOT_LATCHED")
        if (
            safety_state_age is None
            or safety_state_age < 0.0
            or safety_state_age > MAX_F407_SAFETY_STATE_AGE_S
        ):
            f407_reasons.append("F407_HARDWARE_SAFETY_STATE_STALE")
        if not (
            safety_values.get("firmware_identity_valid") == "true"
            and safety_values.get("require_firmware_identity") == "true"
            and safety_values.get("cmd_vel_topic") == "/cmd_vel_safe"
        ):
            f407_reasons.append("F407_DIAGNOSTIC_CONTRACT_INVALID")
    f407_reasons = sorted(set(f407_reasons))
    f407_ready = not f407_reasons
    f407_out = {
        "ready": f407_ready,
        "identity_valid": identity_valid,
        "estop_latched": estop_latched,
        "hardware_estop_latched": hardware_estop_latched,
        "hardware_safety_state_age_s": _round_number(safety_state_age),
        "cmd_vel_expired": cmd_vel_expired_sample if isinstance(cmd_vel_expired_sample, bool) else None,
        "firmware": {
            "protocol_version": firmware_sample.get("protocol_version"),
            "capabilities": firmware_sample.get("capabilities"),
            "build_id": firmware_sample.get("build_id"),
            "test_mode": firmware_sample.get("test_mode"),
            "hw_variant": firmware_sample.get("hw_variant"),
            "identity_valid": firmware_sample.get("identity_valid"),
            "required": firmware_sample.get("required"),
            "identity_enforcement_enabled": firmware_sample.get("identity_enforcement_enabled"),
            "age_s": _round_number(firmware_age),
            "cmd_vel_authority_when_invalid": firmware_sample.get("cmd_vel_authority_when_invalid"),
        },
        "topics": {
            "identity": f407_identity_topic,
            "estop": f407_estop_topic,
            "firmware": f407_firmware_topic,
        },
        "diagnostics": {
            "topic": diagnostics_topic,
            "serial_link": serial_link_diag,
            "safety_bridge": safety_diag,
        },
        "reason_codes": f407_reasons,
    }

    cmd_publishers = _normalized_endpoints(graph_publishers.get("/cmd_vel"))
    cmd_subscribers = _normalized_endpoints(graph_subscribers.get("/cmd_vel"))
    safe_publishers = _normalized_endpoints(graph_publishers.get("/cmd_vel_safe"))
    safe_subscribers = _normalized_endpoints(graph_subscribers.get("/cmd_vel_safe"))
    proposed_publishers = _normalized_endpoints(graph_publishers.get("/mppi/cmd_vel_proposed"))
    shadow_authority_leaks = sorted(
        {
            endpoint["node"]
            for endpoint in cmd_publishers + safe_publishers
            if "mppi" in _node_basename(endpoint["node"]) or "lab_fsd" in _node_basename(endpoint["node"])
        }
    )
    topology_reasons: list[str] = []
    if not _has_node(cmd_subscribers, "collision_monitor"):
        topology_reasons.append("CMD_VEL_COLLISION_MONITOR_INPUT_MISSING")
    if not (len(safe_publishers) == 1 and _has_node(safe_publishers, "collision_monitor")):
        topology_reasons.append("CMD_VEL_SAFE_PUBLISHER_INVALID")
    if not (len(safe_subscribers) == 1 and _has_node(safe_subscribers, "serial_f407")):
        topology_reasons.append("CMD_VEL_SAFE_SUBSCRIBER_INVALID")
    if _has_node(cmd_subscribers, "serial_f407"):
        topology_reasons.append("CMD_VEL_DIRECT_F407_BYPASS")
    if shadow_authority_leaks:
        topology_reasons.append("CMD_VEL_AUTHORITY_LEAK")
    if not (len(proposed_publishers) == 1 and _has_node(proposed_publishers, "mppi_node")):
        topology_reasons.append("MPPI_PROPOSED_PUBLISHER_INVALID")
    topology_reasons = sorted(set(topology_reasons))
    topology_ready = not topology_reasons
    topology_out = {
        "ready": topology_ready,
        "route": "/cmd_vel -> collision_monitor -> /cmd_vel_safe -> serial_f407",
        "cmd_vel_publishers": cmd_publishers,
        "cmd_vel_subscribers": cmd_subscribers,
        "cmd_vel_safe_publishers": safe_publishers,
        "cmd_vel_safe_subscribers": safe_subscribers,
        "mppi_proposed_publishers": proposed_publishers,
        "shadow_authority_leaks": shadow_authority_leaks,
        "authorized_actuator_owner": "/serial_f407",
        "service_owner": machine_out["service_owner"],
        "reason_codes": topology_reasons,
    }

    collision_nodes = sorted(name for name in set(graph_nodes) if _node_basename(name) == "collision_monitor")
    collision_reasons: list[str] = []
    if len(collision_nodes) != 1:
        collision_reasons.append("COLLISION_MONITOR_NODE_MISSING_OR_AMBIGUOUS")
    if str(lifecycle.get("collision_monitor") or "").lower() != "active":
        collision_reasons.append("COLLISION_MONITOR_NOT_ACTIVE")
    for topic in ("/cmd_vel", "/scan", "/scan_depth"):
        if not _has_node(graph_subscribers.get(topic), "collision_monitor"):
            collision_reasons.append(f"COLLISION_MONITOR_{topic.strip('/').upper()}_INPUT_MISSING")
    if not topology_ready:
        collision_reasons.append("COLLISION_MONITOR_VETO_TOPOLOGY_INVALID")
    if not artifacts_out["collision_monitor_config"]["present"]:
        collision_reasons.append("COLLISION_MONITOR_CONFIG_MISSING")
    collision_state, collision_state_reasons = _topic_health(
        topics,
        "/collision_monitor/state",
        stale_after_s=2.0,
        minimum_rate_hz=None,
        reason_prefix="COLLISION_MONITOR_STATE",
    )
    collision_reasons.extend(collision_state_reasons)
    if not _has_node(collision_state["publishers"], "collision_monitor"):
        collision_reasons.append("COLLISION_MONITOR_STATE_SOURCE_INVALID")
    state_sample = _topic_sample(topics, "/collision_monitor/state")
    state_sample = state_sample if isinstance(state_sample, Mapping) else {}
    action_type = state_sample.get("action_type")
    polygon_names = state_sample.get("polygon_names")
    if not (
        isinstance(action_type, int)
        and not isinstance(action_type, bool)
        and 0 <= action_type <= 3
        and isinstance(polygon_names, list)
        and all(isinstance(item, str) for item in polygon_names)
    ):
        collision_reasons.append("COLLISION_MONITOR_STATE_INVALID")

    collision_parameters = parameters.get("collision_monitor", {})
    collision_parameters = collision_parameters if isinstance(collision_parameters, Mapping) else {}
    expected_collision_parameters = {
        "enabled": True,
        "base_frame_id": "base_footprint",
        "odom_frame_id": "odom",
        "cmd_vel_in_topic": "/cmd_vel",
        "cmd_vel_out_topic": "/cmd_vel_safe",
        "state_topic": "/collision_monitor/state",
        "source_timeout": 1.0,
        "observation_sources": ["scan_lidar", "scan_depth"],
        "scan_lidar.topic": "/scan",
        "scan_lidar.enabled": True,
        "scan_depth.topic": "/scan_depth",
        "scan_depth.enabled": True,
    }
    if dict(collision_parameters) != expected_collision_parameters:
        collision_reasons.append("COLLISION_MONITOR_RUNTIME_PARAMETERS_INVALID")
    collision_reasons = sorted(set(collision_reasons))
    collision_ready = not collision_reasons
    collision_out = {
        "ready": collision_ready,
        "nodes": collision_nodes,
        "lifecycle_state": str(lifecycle.get("collision_monitor") or "unavailable").lower(),
        "scan_input": _has_node(graph_subscribers.get("/scan"), "collision_monitor"),
        "scan_depth_input": _has_node(graph_subscribers.get("/scan_depth"), "collision_monitor"),
        "cmd_vel_input": _has_node(graph_subscribers.get("/cmd_vel"), "collision_monitor"),
        "veto_output": "/cmd_vel_safe",
        "state": {
            "topic": collision_state,
            "action_type": action_type if isinstance(action_type, int) else None,
            "polygon_names": (
                sorted(polygon_names)
                if isinstance(polygon_names, list) and all(isinstance(item, str) for item in polygon_names)
                else []
            ),
        },
        "runtime_parameters": {
            key: collision_parameters.get(key) for key in sorted(expected_collision_parameters)
        },
        "reason_codes": collision_reasons,
    }

    fsd_health, fsd_topic_reasons = _topic_health(
        topics,
        "/lab_fsd/fsd_v3_status",
        stale_after_s=2.0,
        minimum_rate_hz=None,
        reason_prefix="LAB_FSD_STATUS",
    )
    input_health, input_topic_reasons = _topic_health(
        topics,
        "/lab_fsd/input_status",
        stale_after_s=2.0,
        minimum_rate_hz=None,
        reason_prefix="LAB_FSD_INPUT",
    )
    safety_health, safety_topic_reasons = _topic_health(
        topics,
        "/lab_fsd/safety_gate",
        stale_after_s=2.0,
        minimum_rate_hz=None,
        reason_prefix="LAB_FSD_SAFETY",
    )
    lab_reasons = fsd_topic_reasons + input_topic_reasons + safety_topic_reasons
    for health in (fsd_health, input_health, safety_health):
        if not _has_node(health["publishers"], "lab_fsd_bev_shadow_planner"):
            lab_reasons.append("LAB_FSD_PUBLISHER_IDENTITY_INVALID")
    fsd_sample = _topic_sample(topics, "/lab_fsd/fsd_v3_status")
    fsd_sample = fsd_sample if isinstance(fsd_sample, Mapping) else {}
    input_sample = _topic_sample(topics, "/lab_fsd/input_status")
    input_sample = input_sample if isinstance(input_sample, Mapping) else {}
    safety_sample = _topic_sample(topics, "/lab_fsd/safety_gate")
    safety_sample = safety_sample if isinstance(safety_sample, Mapping) else {}
    if not (
        fsd_sample.get("stack") == "Lab-FSD v3"
        and fsd_sample.get("shadow_only") is True
        and fsd_sample.get("cmd_vel_authority") is False
    ):
        lab_reasons.append("LAB_FSD_SHADOW_AUTHORITY_INVALID")
    if not (
        input_sample.get("overall") == "live"
        and input_sample.get("shadow_only") is True
        and input_sample.get("cmd_vel_authority") is False
    ):
        lab_reasons.append("LAB_FSD_INPUT_PROVENANCE_INVALID")
    input_sources = (
        input_sample.get("sources", {}) if isinstance(input_sample.get("sources"), Mapping) else {}
    )
    source_states: dict[str, Any] = {}
    for name in ("scan", "scan_depth", "odom"):
        source = input_sources.get(name, {})
        source = source if isinstance(source, Mapping) else {}
        source_states[name] = {
            "state": str(source.get("state") or "unavailable"),
            "fresh": source.get("fresh") is True,
            "usable": source.get("usable") is True,
            "age_s": _round_number(source.get("age_s")),
        }
        if not (
            source.get("state") == "live" and source.get("fresh") is True and source.get("usable") is True
        ):
            lab_reasons.append(f"LAB_FSD_{name.upper()}_SOURCE_NOT_LIVE")
    if not (
        safety_sample.get("shadow_only") is True
        and safety_sample.get("cmd_vel_authority") is False
        and safety_sample.get("shadow_policy") == "observe_only"
    ):
        lab_reasons.append("LAB_FSD_SAFETY_GATE_AUTHORITY_INVALID")
    if shadow_authority_leaks:
        lab_reasons.append("LAB_FSD_COMMAND_AUTHORITY_LEAK")
    lab_reasons = sorted(set(lab_reasons))
    lab_ready = not lab_reasons
    lab_out = {
        "ready": lab_ready,
        "stack": str(fsd_sample.get("stack") or ""),
        "mode": str(fsd_sample.get("mode") or ""),
        "shadow_only": fsd_sample.get("shadow_only") is True,
        "cmd_vel_authority": fsd_sample.get("cmd_vel_authority") is True,
        "input_overall": str(input_sample.get("overall") or "unavailable"),
        "input_sources": source_states,
        "safety_gate": {
            "assist_allowed": safety_sample.get("assist_allowed") is True,
            "shadow_policy": str(safety_sample.get("shadow_policy") or ""),
            "shadow_only": safety_sample.get("shadow_only") is True,
            "cmd_vel_authority": safety_sample.get("cmd_vel_authority") is True,
            "reasons": sorted(
                str(item) for item in safety_sample.get("reasons", []) if isinstance(item, str)
            ),
        },
        "topics": {"status": fsd_health, "input": input_health, "safety": safety_health},
        "reason_codes": lab_reasons,
    }

    bpu_section = fsd_sample.get("bpu", {}) if isinstance(fsd_sample.get("bpu"), Mapping) else {}
    tiny_sample = (
        bpu_section.get("tiny_occ_risk", {}) if isinstance(bpu_section.get("tiny_occ_risk"), Mapping) else {}
    )
    tiny_probabilities = _probability_vector(tiny_sample.get("probs"), length=9)
    policy_prior = fsd_sample.get("policy_prior", {})
    policy_prior = policy_prior if isinstance(policy_prior, Mapping) else {}
    policy_probabilities = _probability_vector(policy_prior.get("probabilities"), length=9)
    tiny_reasons: list[str] = []
    if not (
        tiny_sample.get("ok") is True
        and tiny_sample.get("used") is True
        and tiny_sample.get("runtime") == "hobot_dnn"
        and tiny_sample.get("state") == "forward_ok"
        and tiny_sample.get("authority") == "shadow_diagnostic_only"
        and tiny_sample.get("bin_path") == ARTIFACT_SPECS["tiny_occ_risk_bin"].path
    ):
        tiny_reasons.append("TINY_OCC_RISK_RUNTIME_INVALID")
    if tiny_probabilities is None:
        tiny_reasons.append("TINY_OCC_RISK_9_TOKEN_OUTPUT_INVALID")
    if not (
        policy_prior.get("name") == "tiny_waypoint_policy_prior"
        and policy_prior.get("used_bpu") is True
        and policy_prior.get("token_count") == 9
        and policy_prior.get("cmd_vel_authority") is False
        and policy_prior.get("shadow_only") is True
        and policy_probabilities is not None
    ):
        tiny_reasons.append("TINY_OCC_RISK_POLICY_PRIOR_NOT_BPU_BOUND")
    tiny_artifact = artifacts_out["tiny_occ_risk_bin"]
    if not tiny_artifact["present"] or tiny_artifact["expected_match"] is not True:
        tiny_reasons.append("TINY_OCC_RISK_ARTIFACT_INVALID")
    if not lab_ready:
        tiny_reasons.append("TINY_OCC_RISK_PROVENANCE_NOT_BOUND")
    tiny_reasons = sorted(set(tiny_reasons))
    tiny_ready = not tiny_reasons
    tiny_out = {
        "ready": tiny_ready,
        "backend": str(tiny_sample.get("runtime") or "none"),
        "state": str(tiny_sample.get("state") or "unavailable"),
        "used": tiny_sample.get("used") is True,
        "authority": str(tiny_sample.get("authority") or ""),
        "latency_ms": _round_number(tiny_sample.get("latency_ms")),
        "model_path": str(tiny_sample.get("bin_path") or tiny_sample.get("model") or ""),
        "model_sha256": tiny_artifact["sha256"],
        "token_probabilities": tiny_probabilities,
        "policy_prior": {
            "name": str(policy_prior.get("name") or ""),
            "used_bpu": policy_prior.get("used_bpu") is True,
            "token_count": policy_prior.get("token_count"),
            "probabilities": policy_probabilities,
            "cmd_vel_authority": policy_prior.get("cmd_vel_authority") is True,
            "shadow_only": policy_prior.get("shadow_only") is True,
        },
        "reason_codes": tiny_reasons,
    }

    mppi_health, mppi_topic_reasons = _topic_health(
        topics,
        "/mppi/stats",
        stale_after_s=3.0,
        minimum_rate_hz=None,
        reason_prefix="MPPI_STATUS",
    )
    mppi_sample = _topic_sample(topics, "/mppi/stats")
    mppi_sample = mppi_sample if isinstance(mppi_sample, Mapping) else {}
    mppi_reasons = list(mppi_topic_reasons)
    if not _has_node(mppi_health["publishers"], "mppi_node"):
        mppi_reasons.append("MPPI_PUBLISHER_IDENTITY_INVALID")
    if not (
        mppi_sample.get("use_bpu") is True
        and mppi_sample.get("proposed_only") is True
        and mppi_sample.get("proposed_topic") == "/mppi/cmd_vel_proposed"
        and mppi_sample.get("direct_cmd_vel") is False
        and mppi_sample.get("estop_latched") is True
        and mppi_sample.get("direct_block_reason") == "f407_estop_latched"
    ):
        mppi_reasons.append("MPPI_PROPOSED_ONLY_CONTRACT_INVALID")
    mppi_frame = mppi_sample.get("frame")
    mppi_eval_ms = _parse_float(mppi_sample.get("eval_ms"))
    if not (
        isinstance(mppi_frame, int)
        and not isinstance(mppi_frame, bool)
        and mppi_frame > 0
        and mppi_eval_ms is not None
        and 0.0 <= mppi_eval_ms <= 1000.0
    ):
        mppi_reasons.append("MPPI_FORWARD_EVIDENCE_INVALID")

    mppi_parameters = parameters.get("mppi_node", {})
    mppi_parameters = mppi_parameters if isinstance(mppi_parameters, Mapping) else {}
    max_linear_mps = _parse_float(mppi_parameters.get("max_linear_mps"))
    max_angular_rps = _parse_float(mppi_parameters.get("max_angular_rps"))
    if not (
        mppi_parameters.get("bin_path") == ARTIFACT_SPECS["mppi_cost_bin"].path
        and mppi_parameters.get("use_bpu") is True
        and mppi_parameters.get("cmd_vel_topic") == "/mppi/cmd_vel_proposed"
        and mppi_parameters.get("publish_direct_cmd_vel") is False
        and max_linear_mps is not None
        and 0.0 < max_linear_mps <= 1.0
        and max_angular_rps is not None
        and 0.0 < max_angular_rps <= 3.0
    ):
        mppi_reasons.append("MPPI_RUNTIME_PARAMETERS_INVALID")
    mppi_artifact = artifacts_out["mppi_cost_bin"]
    if not mppi_artifact["present"] or mppi_artifact["expected_match"] is not True:
        mppi_reasons.append("MPPI_BPU_ARTIFACT_INVALID")
    if shadow_authority_leaks or not (
        len(proposed_publishers) == 1 and _has_node(proposed_publishers, "mppi_node")
    ):
        mppi_reasons.append("MPPI_COMMAND_AUTHORITY_LEAK")
    proposed_health, proposed_topic_reasons = _topic_health(
        topics,
        "/mppi/cmd_vel_proposed",
        stale_after_s=1.0,
        minimum_rate_hz=5.0,
        reason_prefix="MPPI_PROPOSED_TWIST",
    )
    mppi_reasons.extend(proposed_topic_reasons)
    if not (
        len(proposed_health["publishers"]) == 1 and _has_node(proposed_health["publishers"], "mppi_node")
    ):
        mppi_reasons.append("MPPI_PROPOSED_TWIST_SOURCE_INVALID")
    proposed_twist = _finite_twist(_topic_sample(topics, "/mppi/cmd_vel_proposed"))
    if proposed_twist is None:
        mppi_reasons.append("MPPI_PROPOSED_TWIST_NONFINITE")
    elif (
        max_linear_mps is None
        or max_angular_rps is None
        or abs(proposed_twist["linear_x"]) > max_linear_mps + 1e-9
        or abs(proposed_twist["angular_z"]) > max_angular_rps + 1e-9
        or any(
            abs(proposed_twist[name]) > 1e-9 for name in ("linear_y", "linear_z", "angular_x", "angular_y")
        )
    ):
        mppi_reasons.append("MPPI_PROPOSED_TWIST_LIMIT_INVALID")
    if hardware_estop_latched and (
        proposed_twist is None or any(abs(value) > 1e-9 for value in proposed_twist.values())
    ):
        mppi_reasons.append("MPPI_ESTOP_PROPOSED_TWIST_NONZERO")
    mppi_reasons = sorted(set(mppi_reasons))
    mppi_ready = not mppi_reasons
    mppi_out = {
        "ready": mppi_ready,
        "backend": "hobot_dnn" if mppi_sample.get("use_bpu") is True else "none",
        "proposed_only": mppi_sample.get("proposed_only") is True,
        "proposed_topic": str(mppi_sample.get("proposed_topic") or ""),
        "direct_cmd_vel": mppi_sample.get("direct_cmd_vel") is True,
        "forward_count": mppi_frame if isinstance(mppi_frame, int) else None,
        "eval_ms": _round_number(mppi_eval_ms),
        "direct_block_reason": str(mppi_sample.get("direct_block_reason") or ""),
        "model_path": str(mppi_parameters.get("bin_path") or ""),
        "model_sha256": mppi_artifact["sha256"],
        "status_topic": mppi_health,
        "proposed_twist_topic": proposed_health,
        "proposed_twist": proposed_twist,
        "runtime_limits": {
            "max_linear_mps": _round_number(max_linear_mps),
            "max_angular_rps": _round_number(max_angular_rps),
        },
        "reason_codes": mppi_reasons,
    }

    raw_clearance = ros.get("clearance", {}) if isinstance(ros.get("clearance"), Mapping) else {}
    clearance_available = raw_clearance.get("available") is True

    def nonnegative_int(name: str) -> int | None:
        value = raw_clearance.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        return None

    scan_body = nonnegative_int("scan_body_stop_points")
    scan_front = nonnegative_int("scan_front_stop_points")
    depth_body = nonnegative_int("scan_depth_body_stop_points")
    depth_front = nonnegative_int("scan_depth_front_stop_points")
    center_cost = nonnegative_int("forward_centerline_max_cost")
    forward_free = raw_clearance.get("forward_centerline_free") is True
    clearance_reasons: list[str] = []
    if not clearance_available or None in (
        scan_body,
        scan_front,
        depth_body,
        depth_front,
        center_cost,
    ):
        clearance_reasons.append("CLEARANCE_GATE_UNAVAILABLE")
    else:
        if scan_filter_reasons:
            clearance_reasons.append("SCAN_SELF_FILTER_NOT_VERIFIED")
        if scan_body > 0:
            clearance_reasons.append("LIDAR_SELF_RETURN_UNRESOLVED")
        if scan_front > 0 or depth_body > 0 or depth_front > 0 or not forward_free or center_cost > 20:
            clearance_reasons.append("FORWARD_CLEARANCE_BLOCKED")
    clearance_reasons = sorted(set(clearance_reasons))
    clearance_ready = not clearance_reasons
    clearance_out = {
        "available": clearance_available,
        "ready": clearance_ready,
        "scan_body_stop_points": scan_body,
        "scan_front_stop_points": scan_front,
        "scan_depth_body_stop_points": depth_body,
        "scan_depth_front_stop_points": depth_front,
        "forward_centerline_max_cost": center_cost,
        "forward_centerline_free": forward_free,
        "self_filter_verified_by_geometry": (
            clearance_available and not scan_filter_reasons and scan_body == 0
        ),
        "reason_codes": clearance_reasons,
    }

    geometry_reasons = sorted(
        set(scan_raw_reasons + scan_reasons + scan_depth_reasons + scan_filter_reasons + clearance_reasons)
    )
    localization_capability_reasons = list(localization_reasons)
    if localization_mode != LOCALIZATION_ONLINE_SLAM:
        localization_capability_reasons.append("LOCALIZATION_PROFILE_REQUIRES_ONLINE_SLAM")
    localization_capability_reasons = sorted(set(localization_capability_reasons))
    capabilities = {
        CAPABILITIES[0]: _capability(
            not geometry_reasons,
            CAPABILITY_BACKENDS[CAPABILITIES[0]],
            geometry_reasons,
        ),
        CAPABILITIES[1]: _capability(
            not localization_capability_reasons,
            CAPABILITY_BACKENDS[CAPABILITIES[1]],
            localization_capability_reasons,
        ),
        CAPABILITIES[2]: _capability(
            not state_estimator_reasons,
            CAPABILITY_BACKENDS[CAPABILITIES[2]],
            state_estimator_reasons,
        ),
        CAPABILITIES[3]: _capability(
            f407_ready,
            CAPABILITY_BACKENDS[CAPABILITIES[3]],
            f407_reasons,
        ),
        CAPABILITIES[4]: _capability(
            collision_ready,
            CAPABILITY_BACKENDS[CAPABILITIES[4]],
            collision_reasons,
        ),
        CAPABILITIES[5]: _capability(
            lab_ready,
            CAPABILITY_BACKENDS[CAPABILITIES[5]],
            lab_reasons,
        ),
        CAPABILITIES[6]: _capability(
            tiny_ready,
            CAPABILITY_BACKENDS[CAPABILITIES[6]],
            tiny_reasons,
        ),
        CAPABILITIES[7]: _capability(
            mppi_ready,
            CAPABILITY_BACKENDS[CAPABILITIES[7]],
            mppi_reasons,
        ),
    }

    physical_reasons = sorted(
        set(
            sensor_reasons
            + localization_capability_reasons
            + f407_reasons
            + topology_reasons
            + collision_reasons
            + lab_reasons
            + tiny_reasons
            + mppi_reasons
            + clearance_reasons
        )
    )
    physical_ready = not physical_reasons
    physical_navigation = {
        "ready": physical_ready,
        "state": "READONLY_PRECONDITION_READY" if physical_ready else "NOT_READY",
        "claim_level": "READONLY_PRECONDITION_ONLY",
        "clearance": clearance_out,
        "motion_executed": False,
        "physical_closure_proven": False,
        "reason_codes": physical_reasons,
    }

    all_reasons = sorted(set(machine_reasons + artifact_reasons + collector_reasons + physical_reasons))
    ready = not all_reasons and all(value["ready"] for value in capabilities.values())
    probe = {
        "read_only": True,
        "operations": [
            "CGROUP_PROC_READ",
            "FILE_READ_SHA256",
            "ROS_GRAPH_QUERY",
            "ROS_LIFECYCLE_GET_STATE",
            "ROS_PARAMETER_GET",
            "ROS_SUBSCRIBE",
            "TF_LOOKUP",
        ],
        "subscribed_topics": list(SUBSCRIBED_TOPICS),
        "publishers_created": 0,
        "actions_called": 0,
        "mutating_services_called": 0,
        "actuator_commands_issued": 0,
        "hardware_device_opens": 0,
        "network_calls_initiated": 0,
        "hardware_touched": False,
        "execution_authority": False,
        "physical_risk_denominator_increment": 0,
    }
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "subsystem": SUBSYSTEM,
        "ready": ready,
        "reason_code": "PASS" if ready else (all_reasons[0] if all_reasons else "HOLD"),
        "reason_codes": all_reasons,
        "observed_at_ms": observed_at_ms,
        "run_id": run_id,
        "run_nonce_sha256": _sha256_text(run_nonce),
        "run_binding_sha256": source_binding_sha256(
            run_id=run_id,
            run_nonce=run_nonce,
            release_id=release_id,
            profile_sha256=profile_sha256,
        ),
        "release_id": release_id,
        "profile_sha256": profile_sha256,
        "device_id": machine_out["device_id"],
        "hostname": machine_out["hostname"],
        "machine_id_sha256": machine_out["machine_id_sha256"],
        "boot_id": machine_out["boot_id"],
        "session_id": machine_out["session_id"],
        "service_invocation_id": machine_out["service_invocation_id"],
        "wlan_mac": machine_out["wlan_mac"],
        "artifacts": artifacts_out,
        "sensors": sensors_out,
        "localization": localization_out,
        "f407": f407_out,
        "command_topology": topology_out,
        "collision_monitor": collision_out,
        "lab_fsd": lab_out,
        "tiny_occ_risk": tiny_out,
        "mppi": mppi_out,
        "physical_navigation": physical_navigation,
        "capabilities": capabilities,
        "probe": probe,
    }
    return {**unsigned, "snapshot_sha256": canonical_sha256(unsigned)}


def validate_snapshot(
    snapshot: Mapping[str, Any],
    *,
    expected_run_id: str,
    expected_run_nonce: str,
    expected_release_id: str,
    expected_profile_sha256: str,
) -> tuple[str, ...]:
    """Validate exact schema, canonical digest, and current anti-replay binding."""

    _validate_binding(
        expected_run_id,
        expected_run_nonce,
        expected_release_id,
        expected_profile_sha256,
    )
    errors: list[str] = []
    if set(snapshot) != SNAPSHOT_KEYS:
        errors.append("SNAPSHOT_SCHEMA_KEYS_INVALID")
    if snapshot.get("schema_version") != SCHEMA_VERSION or snapshot.get("subsystem") != SUBSYSTEM:
        errors.append("SNAPSHOT_SCHEMA_VERSION_INVALID")
    unsigned = dict(snapshot)
    claimed_digest = unsigned.pop("snapshot_sha256", None)
    try:
        actual_digest = canonical_sha256(unsigned)
    except (TypeError, ValueError):
        actual_digest = None
    if claimed_digest != actual_digest:
        errors.append("SNAPSHOT_HASH_MISMATCH")
    if not (
        snapshot.get("run_id") == expected_run_id
        and snapshot.get("release_id") == expected_release_id
        and snapshot.get("profile_sha256") == expected_profile_sha256
        and snapshot.get("run_nonce_sha256") == _sha256_text(expected_run_nonce)
        and snapshot.get("run_binding_sha256")
        == source_binding_sha256(
            run_id=expected_run_id,
            run_nonce=expected_run_nonce,
            release_id=expected_release_id,
            profile_sha256=expected_profile_sha256,
        )
    ):
        errors.append("RUN_BINDING_MISMATCH")
    probe = snapshot.get("probe") if isinstance(snapshot.get("probe"), Mapping) else {}
    if not (
        probe.get("read_only") is True
        and probe.get("publishers_created") == 0
        and probe.get("actions_called") == 0
        and probe.get("mutating_services_called") == 0
        and probe.get("actuator_commands_issued") == 0
        and probe.get("hardware_device_opens") == 0
        and probe.get("hardware_touched") is False
        and probe.get("execution_authority") is False
        and probe.get("physical_risk_denominator_increment") == 0
    ):
        errors.append("READ_ONLY_PROBE_CONTRACT_INVALID")
    reasons = snapshot.get("reason_codes")
    if not isinstance(reasons, list) or reasons != sorted(set(reasons)):
        errors.append("SNAPSHOT_REASON_SET_INVALID")
    if snapshot.get("ready") is True:
        if reasons != [] or snapshot.get("reason_code") != "PASS":
            errors.append("SNAPSHOT_READY_STATE_INVALID")
    elif not reasons or snapshot.get("reason_code") == "PASS":
        errors.append("SNAPSHOT_HOLD_STATE_INVALID")
    return tuple(sorted(set(errors)))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Emit one strict local read-only embodied X5 R2-PREP snapshot."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-nonce", required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument(
        "--localization-mode",
        choices=LOCALIZATION_MODES,
        default=LOCALIZATION_ONLINE_SLAM,
    )
    parser.add_argument("--timeout-sec", type=float, default=6.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _validate_binding(args.run_id, args.run_nonce, args.release_id, args.profile_sha256)
    except ValueError as exc:
        raise SystemExit(f"invalid binding: {exc}") from exc
    if not math.isfinite(args.timeout_sec) or not 1.0 <= args.timeout_sec <= 60.0:
        raise SystemExit("--timeout-sec must be between 1 and 60 seconds")

    observed_at_ms = int(time.time() * 1000)
    snapshot = build_snapshot(
        run_id=args.run_id,
        run_nonce=args.run_nonce,
        release_id=args.release_id,
        profile_sha256=args.profile_sha256,
        observed_at_ms=observed_at_ms,
        machine=collect_machine_identity(),
        artifacts=hash_deployed_artifacts(),
        ros=collect_ros_observation(args.timeout_sec, args.localization_mode),
        localization_mode=args.localization_mode,
    )
    self_check = validate_snapshot(
        snapshot,
        expected_run_id=args.run_id,
        expected_run_nonce=args.run_nonce,
        expected_release_id=args.release_id,
        expected_profile_sha256=args.profile_sha256,
    )
    if self_check:
        raise SystemExit(f"internal snapshot validation failed: {','.join(self_check)}")
    print(canonical_json(snapshot), flush=True)
    # NOT_READY is a valid observation result. The SSH transport must receive
    # its signed reason codes; only argument/build/schema failures are nonzero.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
