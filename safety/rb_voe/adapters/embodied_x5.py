"""Embodied-brain target adapter and strict read-only live snapshot adapter."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from rb_voe.adapters.base import TargetOnlyAdapter
from rb_voe.adapters.read_only import (
    CapabilityReadResult,
    JsonSnapshotTransport,
    ReadSourceKind,
)
from rb_voe.contracts.canonical import canonical_sha256, require_sha256
from rb_voe.contracts.models import CapabilityManifest, ContractError, Maturity
from rb_voe.semantic_profiles import load_profile

EMBODIED_X5_SNAPSHOT_PATH: Final[str] = "/rb-voe/embodied/snapshot"
EMBODIED_X5_SNAPSHOT_SCHEMA: Final[str] = "xrd-rb-voe-embodied-runtime-snapshot-v1"
EMBODIED_X5_EXPECTED_HOSTNAME: Final[str] = "embodied-x5"
EMBODIED_X5_EXPECTED_WLAN_MAC: Final[str] = "40:55:48:a5:41:92"

_PROFILE = load_profile("embodied_x5")
EMBODIED_X5_PROFILE_SHA256: Final[str] = _PROFILE.profile_sha256
EMBODIED_X5_CAPABILITIES: Final[tuple[str, ...]] = _PROFILE.capabilities
EMBODIED_X5_REQUIRED_BACKENDS: Final[Mapping[str, str]] = MappingProxyType(dict(_PROFILE.required_backends))
EMBODIED_X5_PROFILE_REQUIRED_ARTIFACTS: Final[Mapping[str, str]] = MappingProxyType(
    dict(_PROFILE.required_artifacts)
)

EMBODIED_X5_ARTIFACT_PATHS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "collector_script": "/home/rdk/tools/rb_voe_embodied_snapshot.py",
        "body_contour_measurement": ("/home/rdk/rb_voe/lidar_body_contour_measurement.v1.txt"),
        "full_launch": "/home/rdk/ros2_ws/src/my_robot_bringup/launch/full.launch.py",
        "nav2_params": "/home/rdk/ros2_ws/src/my_robot_navigation/config/nav2_params.yaml",
        "collision_monitor_config": (
            "/home/rdk/ros2_ws/src/my_robot_navigation/config/collision_monitor.yaml"
        ),
        "lab_fsd_config": ("/home/rdk/ros2_ws/src/my_robot_navigation/config/lab_fsd_shadow.yaml"),
        "ekf_config": "/home/rdk/ros2_ws/src/my_robot_navigation/config/ekf_odom.yaml",
        "saved_map_yaml": "/home/rdk/maps/lab_final_20260708_210920.yaml",
        "saved_map_pgm": "/home/rdk/maps/lab_final_20260708_210920.pgm",
        "tiny_occ_risk_bin": ("/home/rdk/models/lab_fsd/lab_fsd_tiny_occ_risk.bin"),
        "mppi_cost_bin": "/home/rdk/bpu_models/cost_mlp.bin",
        "lab_anomaly_bin": ("/home/rdk/models/lab_fsd/lab_anomaly_autoencoder.bin"),
        "f407_expected_hex": "/home/rdk/stm32_f407/Objects/a.hex",
        **dict(EMBODIED_X5_PROFILE_REQUIRED_ARTIFACTS),
    }
)
EMBODIED_X5_REQUIRED_ARTIFACTS: Final[frozenset[str]] = frozenset(
    {
        "collector_script",
        "full_launch",
        "nav2_params",
        "collision_monitor_config",
        "lab_fsd_config",
        "ekf_config",
        "saved_map_yaml",
        "saved_map_pgm",
        "tiny_occ_risk_bin",
        "mppi_cost_bin",
        *EMBODIED_X5_PROFILE_REQUIRED_ARTIFACTS,
    }
)
EMBODIED_X5_SCAN_FILTER_ARTIFACTS: Final[Mapping[str, tuple[str, str]]] = MappingProxyType(
    {
        "scan_self_filter": (
            "/home/rdk/ros2_ws/src/my_robot_drivers/scripts/scan_self_filter.py",
            "c8f9a8d0f116127a832592bd31e6fc95da7f4600f91883d980f9aa6a30adaea3",
        ),
        "sensors_launch": (
            "/home/rdk/ros2_ws/src/my_robot_drivers/launch/sensors.launch.py",
            "5263e7bef87579e7fa0b548cd9f0a8853793c29f345a56ae6f43a976e070ccf9",
        ),
        "lidar_launch": (
            "/home/rdk/ros2_ws/src/my_robot_drivers/launch/lidar.launch.py",
            "4b9bf4dbc186a466894206963f4b59362d98043f91cdeacd747042bc7fe25dce",
        ),
    }
)
EMBODIED_X5_SCAN_FILTER_INSTALLED_PATH: Final[str] = (
    "/home/rdk/ros2_ws/install/my_robot_drivers/lib/my_robot_drivers/scan_self_filter"
)
EMBODIED_X5_INSTALLED_RELEASE_ARTIFACTS: Final[Mapping[str, tuple[str, str, str]]] = MappingProxyType(
    {
        "sensors_launch": (
            "/home/rdk/ros2_ws/src/my_robot_drivers/launch/sensors.launch.py",
            "/home/rdk/ros2_ws/install/my_robot_drivers/share/my_robot_drivers/launch/sensors.launch.py",
            "5263e7bef87579e7fa0b548cd9f0a8853793c29f345a56ae6f43a976e070ccf9",
        ),
        "lidar_launch": (
            "/home/rdk/ros2_ws/src/my_robot_drivers/launch/lidar.launch.py",
            "/home/rdk/ros2_ws/install/my_robot_drivers/share/my_robot_drivers/launch/lidar.launch.py",
            "4b9bf4dbc186a466894206963f4b59362d98043f91cdeacd747042bc7fe25dce",
        ),
        "full_launch": (
            "/home/rdk/ros2_ws/src/my_robot_bringup/launch/full.launch.py",
            "/home/rdk/ros2_ws/install/my_robot_bringup/share/my_robot_bringup/launch/full.launch.py",
            "62e1a92378bd3f7f7cff9eefbfcb770feff595692cae2aed7192932848ee7458",
        ),
        "systemd_unit": (
            "/home/rdk/ros2_ws/src/my_robot_bringup/config/embodied_brain.service",
            "/etc/systemd/system/embodied_brain.service",
            "a2024860da4c3831ae136f23e42a8474e4750570720fc457d8a75c200225fdea",
        ),
    }
)

_SNAPSHOT_KEYS: Final[frozenset[str]] = frozenset(
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
_ARTIFACT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "path",
        "required",
        "present",
        "sha256",
        "size_bytes",
        "expected_sha256",
        "expected_match",
    }
)
_CAPABILITY_KEYS: Final[frozenset[str]] = frozenset({"ready", "backend", "reason_codes"})
_PROBE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "read_only",
        "operations",
        "subscribed_topics",
        "publishers_created",
        "actions_called",
        "mutating_services_called",
        "actuator_commands_issued",
        "hardware_device_opens",
        "network_calls_initiated",
        "hardware_touched",
        "execution_authority",
        "physical_risk_denominator_increment",
    }
)
_BOOT_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_INVOCATION_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{32}$")


def _source_binding_sha256(*, run_id: str, run_nonce: str, release_id: str, profile_sha256: str) -> str:
    return canonical_sha256(
        {
            "schema_version": "xrd-rb-voe-live-source-binding-v1",
            "subsystem": "embodied_x5",
            "run_id": run_id,
            "run_nonce": run_nonce,
            "release_id": release_id,
            "profile_sha256": profile_sha256,
        }
    )


@dataclass(frozen=True, slots=True)
class EmbodiedX5CapabilityBinding:
    """Frozen identity and release inventory expected from the vehicle X5."""

    device_id: str
    run_id: str
    run_nonce: str
    release_id: str
    run_binding_sha256: str
    profile_sha256: str
    required_artifact_sha256: Mapping[str, str]
    expected_boot_id: str | None = None
    maturity: Maturity = _PROFILE.maturity
    ttl_ms: int = 10_000
    snapshot_max_age_ms: int = 5_000

    def __post_init__(self) -> None:
        if not self.device_id or not self.run_id or not self.release_id:
            raise ValueError("embodied X5 identity and run bindings must be non-empty")
        if len(self.run_nonce) < 16:
            raise ValueError("embodied X5 run nonce must contain at least 16 characters")
        if self.maturity is not _PROFILE.maturity:
            raise ValueError("embodied X5 maturity is fixed by its semantic profile")
        if self.profile_sha256 != EMBODIED_X5_PROFILE_SHA256:
            raise ValueError("embodied X5 semantic profile digest is not frozen")
        require_sha256("run_binding_sha256", self.run_binding_sha256)
        if self.run_binding_sha256 != _source_binding_sha256(
            run_id=self.run_id,
            run_nonce=self.run_nonce,
            release_id=self.release_id,
            profile_sha256=self.profile_sha256,
        ):
            raise ValueError("embodied X5 run binding digest is inconsistent")
        if set(self.required_artifact_sha256) != EMBODIED_X5_REQUIRED_ARTIFACTS:
            raise ValueError("embodied X5 release inventory must bind every required artifact")
        for name, digest in self.required_artifact_sha256.items():
            require_sha256(f"required_artifact_sha256.{name}", digest)
        if self.expected_boot_id is not None and (
            not isinstance(self.expected_boot_id, str) or _BOOT_ID_RE.fullmatch(self.expected_boot_id) is None
        ):
            raise ValueError("expected_boot_id must be a canonical lowercase UUID")
        for name, value in (
            ("ttl_ms", self.ttl_ms),
            ("snapshot_max_age_ms", self.snapshot_max_age_ms),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        object.__setattr__(
            self,
            "required_artifact_sha256",
            MappingProxyType(dict(self.required_artifact_sha256)),
        )


class EmbodiedX5Adapter(TargetOnlyAdapter):
    subsystem = "embodied_x5"
    capability_schema_version = "xrd-rb-voe-embodied-capability-v1"


EmbodiedX5TargetAdapter = EmbodiedX5Adapter


class EmbodiedX5ReadOnlyAdapter:
    """Compile one run-bound collector snapshot into a read-only manifest."""

    subsystem = "embodied_x5"
    capability_schema_version = "xrd-rb-voe-embodied-capability-v1"

    __slots__ = ("_binding", "_transport")

    def __init__(
        self,
        transport: JsonSnapshotTransport,
        binding: EmbodiedX5CapabilityBinding,
    ) -> None:
        if transport.source_kind is ReadSourceKind.LIVE_REMOTE_READ and binding.expected_boot_id is None:
            raise ValueError("LIVE_REMOTE_READ requires expected_boot_id")
        self._transport = transport
        self._binding = binding

    def read_state(self, *, now_ms: int) -> CapabilityReadResult:
        return self._probe(now_ms=now_ms, operation="READ_STATE")

    def get_capability_manifest(self, *, now_ms: int) -> CapabilityReadResult:
        return self._probe(now_ms=now_ms, operation="CAPABILITY_MANIFEST")

    def _probe(self, *, now_ms: int, operation: str) -> CapabilityReadResult:
        if isinstance(now_ms, bool) or not isinstance(now_ms, int) or now_ms < 0:
            return self._failure(operation, "NOT_READY", details={"upstream": "CLOCK_INVALID"})
        try:
            snapshot = self._transport.get_json(EMBODIED_X5_SNAPSHOT_PATH)
            snapshot_digest = canonical_sha256(snapshot)
        except Exception:
            return self._failure(
                operation,
                "NOT_READY",
                details={"upstream": "EMBODIED_X5_SNAPSHOT_UNAVAILABLE"},
            )

        integrity_reason = self._validate_envelope(snapshot, now_ms=now_ms)
        if integrity_reason is not None:
            return self._failure(
                operation,
                integrity_reason,
                snapshot_sha256=snapshot_digest,
            )
        if snapshot.get("ready") is not True:
            return self._failure(
                operation,
                "NOT_READY",
                snapshot_sha256=snapshot_digest,
                details={
                    "upstream": str(snapshot.get("reason_code")),
                    "reason_codes": snapshot.get("reason_codes", []),
                    "physical_navigation": snapshot.get("physical_navigation", {}),
                },
            )
        semantic_reason = self._validate_ready_semantics(snapshot)
        if semantic_reason is not None:
            return self._failure(
                operation,
                semantic_reason,
                snapshot_sha256=snapshot_digest,
            )

        artifacts = snapshot["artifacts"]
        artifact_sha256 = {
            name: str(record["sha256"])
            for name, record in artifacts.items()
            if record.get("present") is True and isinstance(record.get("sha256"), str)
        }
        calibration_sha256 = {
            name: artifact_sha256[name]
            for name in (
                "nav2_params",
                "collision_monitor_config",
                "lab_fsd_config",
                "ekf_config",
                "saved_map_yaml",
                "saved_map_pgm",
                "body_contour",
            )
        }
        try:
            manifest = CapabilityManifest(
                schema_version=self.capability_schema_version,
                manifest_id=f"embodied-x5-{snapshot_digest[:20]}",
                subsystem=self.subsystem,
                maturity=self._binding.maturity,
                device_id=str(snapshot["device_id"]),
                boot_id=str(snapshot["boot_id"]),
                session_id=str(snapshot["session_id"]),
                release_id=self._binding.release_id,
                capabilities=EMBODIED_X5_CAPABILITIES,
                actual_backends=dict(EMBODIED_X5_REQUIRED_BACKENDS),
                artifact_sha256=artifact_sha256,
                calibration_sha256=calibration_sha256,
                stations=("STATION_IDENTITY",),
                issued_at_ms=now_ms,
                expires_at_ms=now_ms + self._binding.ttl_ms,
            )
        except (ContractError, TypeError, ValueError, KeyError):
            return self._failure(
                operation,
                "EMBODIED_X5_MANIFEST_BINDING_INVALID",
                snapshot_sha256=snapshot_digest,
            )
        return CapabilityReadResult(
            subsystem=self.subsystem,
            operation=operation,
            maturity=manifest.maturity,
            ready=True,
            reason_code="PASS",
            manifest=manifest,
            snapshot_sha256=snapshot_digest,
            details={
                "read_only": True,
                "physical_navigation_claim": "READONLY_PRECONDITION_ONLY",
                "artifact_count": len(artifact_sha256),
            },
            network_touched=self._transport.network_touched,
            source_kind=self._transport.source_kind,
            run_binding_sha256=self._binding.run_binding_sha256,
            profile_sha256=self._binding.profile_sha256,
        )

    def _failure(
        self,
        operation: str,
        reason_code: str,
        *,
        snapshot_sha256: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> CapabilityReadResult:
        return CapabilityReadResult(
            subsystem=self.subsystem,
            operation=operation,
            maturity=Maturity.TARGET_ONLY,
            ready=False,
            reason_code=reason_code,
            snapshot_sha256=snapshot_sha256,
            details={"read_only": True, **dict(details or {})},
            network_touched=self._transport.network_touched,
            source_kind=self._transport.source_kind,
            run_binding_sha256=self._binding.run_binding_sha256,
            profile_sha256=self._binding.profile_sha256,
        )

    def _validate_envelope(self, snapshot: Mapping[str, Any], *, now_ms: int) -> str | None:
        if set(snapshot) != _SNAPSHOT_KEYS:
            return "EMBODIED_X5_SNAPSHOT_SCHEMA_INVALID"
        unsigned = dict(snapshot)
        claimed_digest = unsigned.pop("snapshot_sha256", None)
        if (
            snapshot.get("schema_version") != EMBODIED_X5_SNAPSHOT_SCHEMA
            or snapshot.get("subsystem") != self.subsystem
            or claimed_digest != canonical_sha256(unsigned)
        ):
            return "EMBODIED_X5_SNAPSHOT_INTEGRITY_INVALID"
        if (
            snapshot.get("run_id") != self._binding.run_id
            or snapshot.get("run_nonce_sha256")
            != hashlib.sha256(self._binding.run_nonce.encode("utf-8")).hexdigest()
            or snapshot.get("run_binding_sha256") != self._binding.run_binding_sha256
            or snapshot.get("release_id") != self._binding.release_id
            or snapshot.get("profile_sha256") != self._binding.profile_sha256
        ):
            return "EMBODIED_X5_RUN_BINDING_MISMATCH"
        observed_at_ms = snapshot.get("observed_at_ms")
        if (
            isinstance(observed_at_ms, bool)
            or not isinstance(observed_at_ms, int)
            or observed_at_ms > now_ms
            or now_ms - observed_at_ms > self._binding.snapshot_max_age_ms
        ):
            return "NOT_READY"
        if (
            snapshot.get("device_id") != self._binding.device_id
            or snapshot.get("hostname") != EMBODIED_X5_EXPECTED_HOSTNAME
            or snapshot.get("wlan_mac") != EMBODIED_X5_EXPECTED_WLAN_MAC
            or _BOOT_ID_RE.fullmatch(str(snapshot.get("boot_id", ""))) is None
            or _INVOCATION_ID_RE.fullmatch(str(snapshot.get("service_invocation_id", ""))) is None
            or not str(snapshot.get("session_id", "")).startswith("systemd:")
        ):
            return "EMBODIED_X5_DEVICE_IDENTITY_MISMATCH"
        if (
            self._binding.expected_boot_id is not None
            and snapshot.get("boot_id") != self._binding.expected_boot_id
        ):
            return "EMBODIED_X5_BOOT_ID_MISMATCH"
        try:
            require_sha256("machine_id_sha256", snapshot.get("machine_id_sha256"))
        except (TypeError, ValueError):
            return "EMBODIED_X5_DEVICE_IDENTITY_MISMATCH"
        probe = snapshot.get("probe")
        if not isinstance(probe, Mapping) or set(probe) != _PROBE_KEYS:
            return "EMBODIED_X5_READONLY_PROBE_INVALID"
        if (
            probe.get("read_only") is not True
            or probe.get("publishers_created") != 0
            or probe.get("actions_called") != 0
            or probe.get("mutating_services_called") != 0
            or probe.get("actuator_commands_issued") != 0
            or probe.get("hardware_device_opens") != 0
            or probe.get("network_calls_initiated") != 0
            or probe.get("hardware_touched") is not False
            or probe.get("execution_authority") is not False
            or probe.get("physical_risk_denominator_increment") != 0
        ):
            return "EMBODIED_X5_AUTHORITY_ESCALATION"
        reasons = snapshot.get("reason_codes")
        if not isinstance(reasons, list) or reasons != sorted(set(reasons)):
            return "EMBODIED_X5_REASON_SET_INVALID"
        if snapshot.get("ready") is True:
            if reasons != [] or snapshot.get("reason_code") != "PASS":
                return "EMBODIED_X5_READY_STATE_INVALID"
        elif not reasons or snapshot.get("reason_code") == "PASS":
            return "EMBODIED_X5_HOLD_STATE_INVALID"
        return self._validate_artifacts(snapshot.get("artifacts"))

    def _validate_artifacts(self, raw: Any) -> str | None:
        if not isinstance(raw, Mapping) or set(raw) != set(EMBODIED_X5_ARTIFACT_PATHS):
            return "EMBODIED_X5_ARTIFACT_INVENTORY_INVALID"
        for name, expected_path in EMBODIED_X5_ARTIFACT_PATHS.items():
            record = raw.get(name)
            if not isinstance(record, Mapping) or set(record) != _ARTIFACT_KEYS:
                return "EMBODIED_X5_ARTIFACT_INVENTORY_INVALID"
            if record.get("path") != expected_path:
                return "EMBODIED_X5_ARTIFACT_PATH_MISMATCH"
            if name in EMBODIED_X5_REQUIRED_ARTIFACTS:
                if (
                    record.get("required") is not True
                    or record.get("present") is not True
                    or record.get("sha256") != self._binding.required_artifact_sha256[name]
                    or isinstance(record.get("size_bytes"), bool)
                    or not isinstance(record.get("size_bytes"), int)
                    or record.get("size_bytes") <= 0
                ):
                    return (
                        "EMBODIED_X5_BODY_CONTOUR_RELEASE_MISMATCH"
                        if name == "body_contour"
                        else "EMBODIED_X5_ARTIFACT_RELEASE_MISMATCH"
                    )
            if record.get("present") is True:
                try:
                    require_sha256(f"artifacts.{name}.sha256", record.get("sha256"))
                except (TypeError, ValueError):
                    return "EMBODIED_X5_ARTIFACT_INVENTORY_INVALID"
        return None

    def _validate_ready_semantics(self, snapshot: Mapping[str, Any]) -> str | None:
        capabilities = snapshot.get("capabilities")
        if not isinstance(capabilities, Mapping) or set(capabilities) != set(EMBODIED_X5_CAPABILITIES):
            return "EMBODIED_X5_CAPABILITY_SET_INVALID"
        for capability, expected_backend in EMBODIED_X5_REQUIRED_BACKENDS.items():
            row = capabilities.get(capability)
            if (
                not isinstance(row, Mapping)
                or set(row) != _CAPABILITY_KEYS
                or row.get("ready") is not True
                or row.get("backend") != expected_backend
                or row.get("reason_codes") != []
            ):
                return "EMBODIED_X5_CAPABILITY_BACKEND_INVALID"
        for section in ("sensors", "localization", "f407", "collision_monitor"):
            value = snapshot.get(section)
            if not isinstance(value, Mapping) or value.get("ready") is not True:
                return "EMBODIED_X5_RUNTIME_PRECONDITION_INVALID"
        sensors = snapshot["sensors"]
        scan_filter = sensors.get("scan_filter")
        contour = scan_filter.get("body_contour") if isinstance(scan_filter, Mapping) else None
        contour_artifact = snapshot["artifacts"].get("body_contour")
        measurement_artifact = snapshot["artifacts"].get("body_contour_measurement")
        expected_contour_sha = self._binding.required_artifact_sha256.get("body_contour")
        if (
            not isinstance(scan_filter, Mapping)
            or scan_filter.get("ready") is not True
            or scan_filter.get("exact_stamp_match") is not True
            or scan_filter.get("paired_stamp_fresh") is not True
            or scan_filter.get("geometry_valid") is not True
            or isinstance(scan_filter.get("paired_stamp_ns"), bool)
            or not isinstance(scan_filter.get("paired_stamp_ns"), int)
            or scan_filter.get("paired_stamp_ns") <= 0
            or not isinstance(contour, Mapping)
            or contour.get("valid") is not True
            or contour.get("release_artifact_match") is not True
            or contour.get("path") != EMBODIED_X5_PROFILE_REQUIRED_ARTIFACTS.get("body_contour")
            or contour.get("sha256") != expected_contour_sha
            or contour.get("schema_version") != "xrd-lidar-body-contour-v2"
            or contour.get("measurement_attachment_path")
            != EMBODIED_X5_ARTIFACT_PATHS["body_contour_measurement"]
            or not isinstance(contour.get("measurement_attachment_sha256"), str)
            or not isinstance(contour_artifact, Mapping)
            or contour_artifact.get("sha256") != expected_contour_sha
            or not isinstance(measurement_artifact, Mapping)
            or measurement_artifact.get("required") is not True
            or measurement_artifact.get("present") is not True
            or measurement_artifact.get("path") != EMBODIED_X5_ARTIFACT_PATHS["body_contour_measurement"]
            or measurement_artifact.get("sha256") != contour.get("measurement_attachment_sha256")
            or isinstance(measurement_artifact.get("size_bytes"), bool)
            or not isinstance(measurement_artifact.get("size_bytes"), int)
            or measurement_artifact.get("size_bytes") <= 0
        ):
            return "EMBODIED_X5_BODY_CONTOUR_RUNTIME_BINDING_INVALID"
        implementation_artifacts = scan_filter.get("implementation_artifacts")
        if not isinstance(implementation_artifacts, Mapping) or set(implementation_artifacts) != set(
            EMBODIED_X5_SCAN_FILTER_ARTIFACTS
        ):
            return "EMBODIED_X5_SCAN_FILTER_RELEASE_BINDING_INVALID"
        for name, (expected_path, expected_sha256) in EMBODIED_X5_SCAN_FILTER_ARTIFACTS.items():
            record = implementation_artifacts.get(name)
            if (
                not isinstance(record, Mapping)
                or set(record)
                != {
                    "path",
                    "required",
                    "present",
                    "sha256",
                    "size_bytes",
                    "expected_sha256",
                    "expected_match",
                }
                or record.get("path") != expected_path
                or record.get("required") is not True
                or record.get("present") is not True
                or record.get("sha256") != expected_sha256
                or record.get("expected_sha256") != expected_sha256
                or record.get("expected_match") is not True
                or isinstance(record.get("size_bytes"), bool)
                or not isinstance(record.get("size_bytes"), int)
                or int(record["size_bytes"]) <= 0
            ):
                return "EMBODIED_X5_SCAN_FILTER_RELEASE_BINDING_INVALID"
        runtime_process = scan_filter.get("runtime_process")
        expected_filter_sha256 = EMBODIED_X5_SCAN_FILTER_ARTIFACTS["scan_self_filter"][1]
        if (
            not isinstance(runtime_process, Mapping)
            or runtime_process.get("expected_path") != EMBODIED_X5_SCAN_FILTER_INSTALLED_PATH
            or runtime_process.get("file_kind") not in {"regular_install", "symlink_to_source"}
            or not (
                (
                    runtime_process.get("file_kind") == "symlink_to_source"
                    and runtime_process.get("resolved_path")
                    == EMBODIED_X5_SCAN_FILTER_ARTIFACTS["scan_self_filter"][0]
                )
                or (
                    runtime_process.get("file_kind") == "regular_install"
                    and runtime_process.get("resolved_path") == EMBODIED_X5_SCAN_FILTER_INSTALLED_PATH
                )
            )
            or runtime_process.get("present") is not True
            or runtime_process.get("sha256") != expected_filter_sha256
            or runtime_process.get("expected_sha256") != expected_filter_sha256
            or runtime_process.get("hash_match") is not True
            or runtime_process.get("matching_process_count") != 1
            or isinstance(runtime_process.get("matched_pid"), bool)
            or not isinstance(runtime_process.get("matched_pid"), int)
            or runtime_process.get("matched_pid") <= 0
            or not isinstance(runtime_process.get("cmdline_sha256"), str)
            or runtime_process.get("exact_cmdline_match") is not True
            or isinstance(runtime_process.get("artifact_mtime_epoch_ms"), bool)
            or not isinstance(runtime_process.get("artifact_mtime_epoch_ms"), int)
            or isinstance(runtime_process.get("process_start_epoch_ms"), bool)
            or not isinstance(runtime_process.get("process_start_epoch_ms"), int)
            or runtime_process.get("process_started_after_artifact") is not True
            or scan_filter.get("positive_infinity_removed_count") != scan_filter.get("removed_count")
            or scan_filter.get("unremoved_inside_contour_count") != 0
            or scan_filter.get("modified_or_inserted_count") != 0
            or scan_filter.get("removed_outside_contour_count") != 0
            or scan_filter.get("invalid_ranges_preserved") is not True
            or scan_filter.get("intensities_preserved") is not True
        ):
            return "EMBODIED_X5_SCAN_FILTER_RUNTIME_BINDING_INVALID"
        installed_release_artifacts = scan_filter.get("installed_release_artifacts")
        if not isinstance(installed_release_artifacts, Mapping) or set(installed_release_artifacts) != set(
            EMBODIED_X5_INSTALLED_RELEASE_ARTIFACTS
        ):
            return "EMBODIED_X5_INSTALLED_RELEASE_BINDING_INVALID"
        for name, (
            source_path,
            installed_path,
            expected_sha256,
        ) in EMBODIED_X5_INSTALLED_RELEASE_ARTIFACTS.items():
            record = installed_release_artifacts.get(name)
            if (
                not isinstance(record, Mapping)
                or record.get("source_path") != source_path
                or record.get("installed_path") != installed_path
                or record.get("file_kind") not in {"regular_install", "symlink_to_source"}
                or not (
                    (
                        record.get("file_kind") == "symlink_to_source"
                        and record.get("resolved_path") == source_path
                    )
                    or (
                        record.get("file_kind") == "regular_install"
                        and record.get("resolved_path") == installed_path
                    )
                )
                or record.get("present") is not True
                or record.get("sha256") != expected_sha256
                or record.get("expected_sha256") != expected_sha256
                or record.get("hash_match") is not True
                or record.get("source_install_match") is not True
                or isinstance(record.get("size_bytes"), bool)
                or not isinstance(record.get("size_bytes"), int)
                or record.get("size_bytes") <= 0
            ):
                return "EMBODIED_X5_INSTALLED_RELEASE_BINDING_INVALID"
        f407 = snapshot["f407"]
        if f407.get("identity_valid") is not True or f407.get("estop_latched") is not True:
            return "EMBODIED_X5_F407_SAFETY_INVALID"
        topology = snapshot.get("command_topology")
        if (
            not isinstance(topology, Mapping)
            or topology.get("ready") is not True
            or topology.get("authorized_actuator_owner") != "/serial_f407"
            or topology.get("shadow_authority_leaks") != []
            or topology.get("reason_codes") != []
        ):
            return "EMBODIED_X5_COMMAND_TOPOLOGY_INVALID"
        lab_fsd = snapshot.get("lab_fsd")
        if (
            not isinstance(lab_fsd, Mapping)
            or lab_fsd.get("ready") is not True
            or lab_fsd.get("shadow_only") is not True
            or lab_fsd.get("cmd_vel_authority") is not False
        ):
            return "EMBODIED_X5_LAB_FSD_AUTHORITY_INVALID"
        tiny = snapshot.get("tiny_occ_risk")
        if (
            not isinstance(tiny, Mapping)
            or tiny.get("ready") is not True
            or tiny.get("backend") != "hobot_dnn"
            or tiny.get("state") != "forward_ok"
            or tiny.get("used") is not True
            or tiny.get("authority") != "shadow_diagnostic_only"
        ):
            return "EMBODIED_X5_TINY_OCC_RUNTIME_INVALID"
        mppi = snapshot.get("mppi")
        if (
            not isinstance(mppi, Mapping)
            or mppi.get("ready") is not True
            or mppi.get("backend") != "hobot_dnn"
            or mppi.get("proposed_only") is not True
            or mppi.get("proposed_topic") != "/mppi/cmd_vel_proposed"
            or mppi.get("direct_cmd_vel") is not False
        ):
            return "EMBODIED_X5_MPPI_AUTHORITY_INVALID"
        physical = snapshot.get("physical_navigation")
        if (
            not isinstance(physical, Mapping)
            or physical.get("ready") is not True
            or physical.get("state") != "READONLY_PRECONDITION_READY"
            or physical.get("claim_level") != "READONLY_PRECONDITION_ONLY"
            or physical.get("motion_executed") is not False
            or physical.get("physical_closure_proven") is not False
            or physical.get("reason_codes") != []
        ):
            return "EMBODIED_X5_PHYSICAL_PRECONDITION_INVALID"
        clearance = physical.get("clearance")
        if (
            not isinstance(clearance, Mapping)
            or clearance.get("ready") is not True
            or clearance.get("scan_body_stop_points") != 0
            or clearance.get("scan_front_stop_points") != 0
            or clearance.get("scan_depth_body_stop_points") != 0
            or clearance.get("scan_depth_front_stop_points") != 0
            or clearance.get("forward_centerline_free") is not True
            or clearance.get("self_filter_verified_by_geometry") is not True
        ):
            return "EMBODIED_X5_CLEARANCE_GATE_INVALID"
        return None


__all__ = [
    "EMBODIED_X5_ARTIFACT_PATHS",
    "EMBODIED_X5_CAPABILITIES",
    "EMBODIED_X5_PROFILE_SHA256",
    "EMBODIED_X5_PROFILE_REQUIRED_ARTIFACTS",
    "EMBODIED_X5_REQUIRED_ARTIFACTS",
    "EMBODIED_X5_REQUIRED_BACKENDS",
    "EMBODIED_X5_SCAN_FILTER_ARTIFACTS",
    "EMBODIED_X5_SNAPSHOT_PATH",
    "EmbodiedX5Adapter",
    "EmbodiedX5CapabilityBinding",
    "EmbodiedX5ReadOnlyAdapter",
    "EmbodiedX5TargetAdapter",
]
