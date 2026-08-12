"""Dual-arm target adapter and strict two-Pi read-only aggregation adapter."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from rb_voe.adapters.base import TargetOnlyAdapter
from rb_voe.adapters.read_only import (
    CapabilityReadResult,
    JsonSnapshotTransport,
    ReadSourceKind,
)
from rb_voe.contracts.canonical import canonical_sha256, file_sha256, require_sha256
from rb_voe.contracts.models import CapabilityManifest, ContractError, Maturity
from rb_voe.semantic_profiles import load_profile

try:
    from workstation.dual_arm import rb_voe_overhead_actual_record as a0_contract
except ModuleNotFoundError:  # pragma: no cover - deployed AI-X5 package layout
    from a0_tools import rb_voe_overhead_actual_record as a0_contract  # type: ignore[no-redef]

DUAL_ARM_MEMBER_SCHEMA: Final[str] = "xrd-rb-voe-dual-arm-member-snapshot-v4"
DUAL_ARM_VISION_SCHEMA: Final[str] = a0_contract.SCHEMA_VERSION
DUAL_ARM_MEMBER_PROBE_SHA256: Final[str] = "75707dc8dd5265965ec2fdddb73a8ee81f1c7615b9cb1fc387c6e5421ed89d8c"
DUAL_ARM_FINALS_ORCHESTRATOR_SHA256: Final[str] = (
    "0c224675ab2b38a64387b84eb790414ddca53ba78e3d8cb32dbd7548d4b05e65"
)
DUAL_ARM_FINALS_STATION_CONFIG_SHA256: Final[str] = (
    "e37dfeccb0d35dda8fd9938317a856072b885dd3a2c4672f39097ed0f2de205d"
)
ARM02_OVERHEAD_CAMERA_USB_ID: Final[str] = "1bcf:0d1a"
DUAL_ARM_MEMBER_PATHS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "arm01": "/rb-voe/dual-arm/arm01/snapshot",
        "arm02": "/rb-voe/dual-arm/arm02/snapshot",
    }
)
DUAL_ARM_VISION_RECORD_PATH: Final[str] = "/rb-voe/dual-arm/overhead/a0-record"

_PROFILE = load_profile("dual_arm")
DUAL_ARM_PROFILE_SHA256: Final[str] = _PROFILE.profile_sha256
DUAL_ARM_CAPABILITIES: Final[tuple[str, ...]] = _PROFILE.capabilities
DUAL_ARM_REQUIRED_BACKENDS: Final[Mapping[str, str]] = MappingProxyType(dict(_PROFILE.required_backends))
_MEMBER_SPECS: Final[Mapping[str, Mapping[str, Any]]] = MappingProxyType(
    {
        "arm01": MappingProxyType(
            {
                "hostname": "mycobot-arm-01",
                "wlan0_mac": "e4:5f:01:bf:de:a7",
                "cpu_serial": "1000000092fb92d3",
                "physical_side": "left",
                "rb_voe_role": "material_fixture_executor",
                "tool_role": "blue_g23_powder_bag_gripper",
                "camera_usb_id": None,
                "systemd_unit": "xrd-workcockpit.service",
                "artifacts": MappingProxyType(
                    {
                        "probe_script": MappingProxyType(
                            {
                                "path": "/home/rdk/dual_arm/rb_voe_arm_readonly_probe.py",
                                "sha256": DUAL_ARM_MEMBER_PROBE_SHA256,
                            }
                        ),
                        "motion_entrypoint": MappingProxyType(
                            {
                                "path": "/home/rdk/arm01_compact_front_transfer.py",
                                "sha256": "1e385eb813a89a484ac59aa69f1c7ec82a86b9ed2164f1efb761bd78891b2993",
                            }
                        ),
                        "fk_dependency": MappingProxyType(
                            {
                                "path": "/home/rdk/mycobot280_fk.py",
                                "sha256": "aa41062074fd5b695818ca057078ae7d6a34a137bdaf63cad70c399e033a9d6f",
                            }
                        ),
                        "bag_pick_dependency": MappingProxyType(
                            {
                                "path": "/home/rdk/bag_fixed_pick_g23.py",
                                "sha256": "415fdfff17b34ae24a65ae68b426cf4a63e1bb5b0092fc4dec1cf090ac5ece4d",
                            }
                        ),
                        "finals_orchestrator": MappingProxyType(
                            {
                                "path": "/home/rdk/dual_arm/run_dual_arm_bag_grind.ps1",
                                "sha256": DUAL_ARM_FINALS_ORCHESTRATOR_SHA256,
                            }
                        ),
                        "station_config": MappingProxyType(
                            {
                                "path": "/home/rdk/dual_arm/station_config.json",
                                "sha256": DUAL_ARM_FINALS_STATION_CONFIG_SHA256,
                            }
                        ),
                        "systemd_unit": MappingProxyType(
                            {
                                "path": "/etc/systemd/system/xrd-workcockpit.service",
                                "sha256": "44c1a0e43ae66dcbcad5cd36eb1aacdedac86591f1cb5b23576dad6b8c795363",
                            }
                        ),
                    }
                ),
                "forbidden_cron_surfaces": (
                    "automatic_ager_runner",
                    "bag_pick_motion_dependency",
                    "finals_motion_entrypoint",
                    "workcockpit_app",
                ),
                "forbidden_process_surfaces": (
                    "automatic_ager_runner",
                    "bag_pick_motion_dependency",
                    "finals_motion_entrypoint",
                    "workcockpit_app",
                ),
            }
        ),
        "arm02": MappingProxyType(
            {
                "hostname": "er",
                "wlan0_mac": "98:fe:54:0c:94:07",
                "cpu_serial": "10000000f08c41fc",
                "physical_side": "right",
                "rb_voe_role": "grind_executor",
                "tool_role": "red_grinding_rod_gripper",
                "camera_usb_id": ARM02_OVERHEAD_CAMERA_USB_ID,
                "systemd_unit": "xrd-overhead-camera.service",
                "artifacts": MappingProxyType(
                    {
                        "probe_script": MappingProxyType(
                            {
                                "path": "/home/rdk/dual_arm/rb_voe_arm_readonly_probe.py",
                                "sha256": DUAL_ARM_MEMBER_PROBE_SHA256,
                            }
                        ),
                        "motion_entrypoint": MappingProxyType(
                            {
                                "path": "/home/rdk/xrd/workstation/dual_arm/arm02_direct_grind_closed_loop.py",
                                "sha256": "4952d62719c4eeb5939e3544cf928cd2266367c9c215d49c44bf5a041b0e486d",
                            }
                        ),
                        "overhead_camera_service": MappingProxyType(
                            {
                                "path": "/home/rdk/dual_arm/overhead_camera_service.py",
                                "sha256": "7a117d355c7e92013be1cfec472259655a00b8bbb716033281a547a1a43bd4d5",
                            }
                        ),
                        "station_config": MappingProxyType(
                            {
                                "path": "/home/rdk/dual_arm/station_config.json",
                                "sha256": DUAL_ARM_FINALS_STATION_CONFIG_SHA256,
                            }
                        ),
                        "finals_orchestrator": MappingProxyType(
                            {
                                "path": "/home/rdk/dual_arm/run_dual_arm_bag_grind.ps1",
                                "sha256": DUAL_ARM_FINALS_ORCHESTRATOR_SHA256,
                            }
                        ),
                        "systemd_unit": MappingProxyType(
                            {
                                "path": "/etc/systemd/system/xrd-overhead-camera.service",
                                "sha256": "99669577154286055ea449227b86bf8221efeed2df438f5e5f9dc96fadf388e2",
                            }
                        ),
                    }
                ),
                "forbidden_process_surfaces": (
                    "automatic_ager_aging",
                    "automatic_ager_runner",
                    "finals_motion_entrypoint",
                    "legacy_apriltag_motion",
                    "legacy_arm02_service",
                    "legacy_gripper_test",
                    "legacy_hello_world",
                    "legacy_home_arm02_service",
                ),
                "forbidden_cron_surfaces": (
                    "automatic_ager_aging",
                    "automatic_ager_runner",
                    "finals_motion_entrypoint",
                    "legacy_arm02_service",
                    "legacy_home_arm02_service",
                ),
            }
        ),
    }
)

_MEMBER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "arm_id",
        "run_id",
        "run_nonce",
        "release_id",
        "profile_sha256",
        "observed_at_ms",
        "ready",
        "reasons",
        "frozen_identity",
        "identity",
        "member",
        "systemd",
        "cron",
        "processes",
        "devices",
        "artifacts",
        "probe",
        "snapshot_sha256",
    }
)
_BOOT_RE: Final[re.Pattern[str]] = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_INVOCATION_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{32}$")
_USB_COMPONENT_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{4}$")


def _is_sysfs_usb_identity_source(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sysfs:/sys/")
        and value.endswith("/idVendor+idProduct")
        and ".." not in value
        and "\\" not in value
    )


def _source_binding_sha256(*, run_id: str, run_nonce: str, release_id: str, profile_sha256: str) -> str:
    return canonical_sha256(
        {
            "schema_version": "xrd-rb-voe-live-source-binding-v1",
            "subsystem": "dual_arm",
            "run_id": run_id,
            "run_nonce": run_nonce,
            "release_id": release_id,
            "profile_sha256": profile_sha256,
        }
    )


@dataclass(frozen=True, slots=True)
class DualArmCapabilityBinding:
    run_id: str
    run_nonce: str
    release_id: str
    run_binding_sha256: str
    profile_sha256: str
    expected_machine_id_sha256: Mapping[str, str]
    probe_script_sha256: str
    ai_x5_device_id: str
    ai_x5_boot_id: str
    vision_acquisition_id: str
    vision_a0_run_id: str
    vision_capture_session_id: str
    vision_inference_session_id: str
    vision_challenge_sha256: str
    vision_challenge_issued_at_ms: int
    vision_challenge_expires_at_ms: int
    vision_config_sha256: str
    vision_case_id: str
    vision_sample_id: str
    vision_sample_lineage_sha256: str
    vision_parent_evidence_root_sha256: str
    vision_bag_empty_baseline_sha256: str | None
    vision_task_kind: str
    vision_result_schema: str
    vision_success_state: str
    vision_acquisition_manifest: Path
    vision_raw_frame: Path
    vision_frame_bundle_artifact: Path
    vision_result_json: Path
    vision_camera_service_identity_artifact: Path
    vision_capture_pipeline_artifact: Path
    vision_inference_pipeline_artifact: Path
    vision_replay_ledger_dir: Path
    expected_boot_id: Mapping[str, str] | None = None
    maturity: Maturity = Maturity.SHADOW_VALIDATED
    ttl_ms: int = 10_000
    vision_max_age_ms: int = 300_000

    def __post_init__(self) -> None:
        if not self.run_id or not self.release_id or not self.ai_x5_device_id:
            raise ValueError("dual-arm run and AI X5 identity bindings must be non-empty")
        for name in (
            "vision_acquisition_id",
            "vision_a0_run_id",
            "vision_capture_session_id",
            "vision_inference_session_id",
            "vision_case_id",
            "vision_sample_id",
            "vision_task_kind",
            "vision_result_schema",
            "vision_success_state",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if len(self.run_nonce) < 16:
            raise ValueError("dual-arm run nonce must contain at least 16 characters")
        if self.profile_sha256 != DUAL_ARM_PROFILE_SHA256:
            raise ValueError("dual-arm semantic profile digest is not frozen")
        if self.maturity is not Maturity.SHADOW_VALIDATED:
            raise ValueError("dual-arm live maturity is fixed to SHADOW_VALIDATED")
        require_sha256("run_binding_sha256", self.run_binding_sha256)
        if self.run_binding_sha256 != _source_binding_sha256(
            run_id=self.run_id,
            run_nonce=self.run_nonce,
            release_id=self.release_id,
            profile_sha256=self.profile_sha256,
        ):
            raise ValueError("dual-arm run binding digest is inconsistent")
        if set(self.expected_machine_id_sha256) != set(_MEMBER_SPECS):
            raise ValueError("dual-arm binding requires both Pi machine identities")
        for arm_id, digest in self.expected_machine_id_sha256.items():
            require_sha256(f"expected_machine_id_sha256.{arm_id}", digest)
        if self.expected_boot_id is not None:
            if not isinstance(self.expected_boot_id, Mapping) or set(self.expected_boot_id) != set(
                _MEMBER_SPECS
            ):
                raise ValueError("expected_boot_id must contain exactly arm01 and arm02")
            for arm_id, boot_id in self.expected_boot_id.items():
                if not isinstance(boot_id, str) or _BOOT_RE.fullmatch(boot_id) is None:
                    raise ValueError(f"expected_boot_id.{arm_id} must be a canonical lowercase UUID")
            if len(set(self.expected_boot_id.values())) != len(_MEMBER_SPECS):
                raise ValueError("arm01 and arm02 expected boot IDs must be different")
        require_sha256("probe_script_sha256", self.probe_script_sha256)
        if self.probe_script_sha256 != DUAL_ARM_MEMBER_PROBE_SHA256:
            raise ValueError("probe_script_sha256 does not match the current v4 producer")
        for name in (
            "vision_challenge_sha256",
            "vision_config_sha256",
            "vision_sample_lineage_sha256",
            "vision_parent_evidence_root_sha256",
        ):
            require_sha256(name, getattr(self, name))
        if self.vision_bag_empty_baseline_sha256 is not None:
            require_sha256(
                "vision_bag_empty_baseline_sha256",
                self.vision_bag_empty_baseline_sha256,
            )
        if a0_contract.TASK_RESULT_CONTRACTS.get(self.vision_task_kind) != (
            self.vision_result_schema,
            self.vision_success_state,
        ):
            raise ValueError("dual-arm A0 task/result tuple is unsupported")
        if self.vision_task_kind != "BAG_DROP_IN_GRINDING_DISH":
            raise ValueError("dual-arm live binding is fixed to the bag-drop task")
        for name in ("vision_challenge_issued_at_ms", "vision_challenge_expires_at_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.vision_challenge_expires_at_ms <= self.vision_challenge_issued_at_ms:
            raise ValueError("vision challenge expiry must follow issuance")
        expected_challenge_sha256 = a0_contract.a0_challenge_sha256(
            acquisition_id=self.vision_acquisition_id,
            a0_run_id=self.vision_a0_run_id,
            r2_run_id=self.run_id,
            r2_run_nonce=self.run_nonce,
            challenge_issued_at_ms=self.vision_challenge_issued_at_ms,
            challenge_expires_at_ms=self.vision_challenge_expires_at_ms,
            release_id=self.release_id,
            config_sha256=self.vision_config_sha256,
            case_id=self.vision_case_id,
            sample_id=self.vision_sample_id,
            sample_lineage_sha256=self.vision_sample_lineage_sha256,
            parent_evidence_root_sha256=self.vision_parent_evidence_root_sha256,
            bag_empty_baseline_sha256=self.vision_bag_empty_baseline_sha256,
            task_kind=self.vision_task_kind,
            result_schema=self.vision_result_schema,
            success_state=self.vision_success_state,
        )
        if self.vision_challenge_sha256 != expected_challenge_sha256:
            raise ValueError("vision_challenge_sha256 is inconsistent")
        if not self.ai_x5_boot_id:
            raise ValueError("dual-arm A0 evidence requires the current AI X5 boot ID")
        for name in (
            "vision_acquisition_manifest",
            "vision_raw_frame",
            "vision_frame_bundle_artifact",
            "vision_result_json",
            "vision_camera_service_identity_artifact",
            "vision_capture_pipeline_artifact",
            "vision_inference_pipeline_artifact",
            "vision_replay_ledger_dir",
        ):
            path = Path(getattr(self, name))
            if not path.is_absolute():
                raise ValueError(f"{name} must be an absolute path")
            object.__setattr__(self, name, path)
        evidence_files = {
            self.vision_acquisition_manifest,
            self.vision_raw_frame,
            self.vision_frame_bundle_artifact,
            self.vision_result_json,
            self.vision_camera_service_identity_artifact,
            self.vision_capture_pipeline_artifact,
            self.vision_inference_pipeline_artifact,
        }
        if len(evidence_files) != 7 or self.vision_replay_ledger_dir in evidence_files:
            raise ValueError("dual-arm A0 evidence paths must be distinct")
        for name, value in (("ttl_ms", self.ttl_ms), ("vision_max_age_ms", self.vision_max_age_ms)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        object.__setattr__(
            self,
            "expected_machine_id_sha256",
            MappingProxyType(dict(self.expected_machine_id_sha256)),
        )
        if self.expected_boot_id is not None:
            object.__setattr__(
                self,
                "expected_boot_id",
                MappingProxyType(dict(self.expected_boot_id)),
            )


class DualArmAdapter(TargetOnlyAdapter):
    subsystem = "dual_arm"
    capability_schema_version = "xrd-rb-voe-dual-arm-capability-v1"


DualArmTargetAdapter = DualArmAdapter


class DualArmReadOnlyAdapter:
    subsystem = "dual_arm"
    capability_schema_version = "xrd-rb-voe-dual-arm-capability-v1"

    __slots__ = ("_arms", "_binding", "_source_kind", "_vision")

    def __init__(
        self,
        *,
        arm01_transport: JsonSnapshotTransport,
        arm02_transport: JsonSnapshotTransport,
        vision_transport: JsonSnapshotTransport,
        binding: DualArmCapabilityBinding,
    ) -> None:
        if arm01_transport.source_kind is not arm02_transport.source_kind:
            raise ValueError("dual-arm member transports cannot mix replay and live sources")
        if vision_transport.source_kind is not ReadSourceKind.CAPTURED_REPLAY:
            raise ValueError("dual-arm A0 vision evidence must be a sealed local record")
        if (
            arm01_transport.source_kind is ReadSourceKind.LIVE_REMOTE_READ
            and binding.expected_boot_id is None
        ):
            raise ValueError("LIVE_REMOTE_READ requires expected_boot_id for both arms")
        self._arms = {"arm01": arm01_transport, "arm02": arm02_transport}
        self._vision = vision_transport
        self._binding = binding
        self._source_kind = arm01_transport.source_kind

    @property
    def _network_touched(self) -> bool:
        return any(transport.network_touched for transport in self._arms.values())

    def read_state(self, *, now_ms: int) -> CapabilityReadResult:
        return self._probe(now_ms=now_ms, operation="READ_STATE")

    def get_capability_manifest(self, *, now_ms: int) -> CapabilityReadResult:
        return self._probe(now_ms=now_ms, operation="CAPABILITY_MANIFEST")

    def _probe(self, *, now_ms: int, operation: str) -> CapabilityReadResult:
        if isinstance(now_ms, bool) or not isinstance(now_ms, int) or now_ms < 0:
            return self._failure(operation, "NOT_READY", details={"upstream": "CLOCK_INVALID"})
        members: dict[str, Mapping[str, Any]] = {}
        member_digests: dict[str, str] = {}
        for arm_id, transport in self._arms.items():
            try:
                snapshot = transport.get_json(DUAL_ARM_MEMBER_PATHS[arm_id])
                member_digests[arm_id] = canonical_sha256(snapshot)
            except Exception:
                return self._failure(
                    operation,
                    "NOT_READY",
                    details={"upstream": f"{arm_id.upper()}_SNAPSHOT_UNAVAILABLE"},
                )
            reason = self._validate_member(arm_id, snapshot, now_ms=now_ms)
            if reason is not None:
                return self._failure(
                    operation,
                    reason,
                    snapshot_sha256=member_digests[arm_id],
                )
            if snapshot.get("ready") is not True:
                return self._failure(
                    operation,
                    "NOT_READY",
                    snapshot_sha256=member_digests[arm_id],
                    details={"upstream": arm_id, "reason_codes": snapshot.get("reasons", [])},
                )
            members[arm_id] = snapshot

        try:
            vision = self._vision.get_json(DUAL_ARM_VISION_RECORD_PATH)
        except Exception:
            return self._failure(
                operation,
                "NOT_READY",
                details={"upstream": "OVERHEAD_A0_RECORD_UNAVAILABLE"},
            )
        vision_reason = self._validate_vision(vision, members=members, now_ms=now_ms)
        if vision_reason is not None:
            details = (
                {
                    "upstream": "OVERHEAD_A0_GATE",
                    "reason_codes": ["OVERHEAD_VISION_GATE_NEGATIVE"],
                }
                if vision_reason == "NOT_READY"
                else None
            )
            return self._failure(operation, vision_reason, details=details)

        aggregate_snapshot_sha256 = canonical_sha256(
            {
                "arm01": member_digests["arm01"],
                "arm02": member_digests["arm02"],
                "vision": vision["record_sha256"],
            }
        )
        stable_members = {
            arm_id: {
                "machine_id_sha256": snapshot["identity"]["machine_id_sha256"],
                "cpu_serial": snapshot["identity"]["cpu_serial"],
            }
            for arm_id, snapshot in members.items()
        }
        device_id = f"dual-pi:{canonical_sha256(stable_members)[:32]}"
        boot_id = (
            "dual-boot:"
            + canonical_sha256(
                {arm_id: snapshot["identity"]["boot_id"] for arm_id, snapshot in members.items()}
            )[:32]
        )
        session_id = (
            "dual-read:"
            + canonical_sha256(
                {
                    "run_id": self._binding.run_id,
                    "run_nonce": self._binding.run_nonce,
                    "members": member_digests,
                    "vision": vision["record_sha256"],
                }
            )[:32]
        )
        artifact_sha256: dict[str, str] = {}
        for arm_id, snapshot in members.items():
            for name, record in snapshot["artifacts"].items():
                artifact_sha256[f"{arm_id}.{name}"] = str(record["sha256"])
        artifact_sha256["arm02.video0_usb_identity"] = canonical_sha256(
            members["arm02"]["devices"]["video0"]["usb_identity"]
        )
        artifact_sha256.update(
            {
                "overhead.acquisition_manifest": str(vision["acquisition_manifest_sha256"]),
                "overhead.raw_frame": str(vision["raw_frame_sha256"]),
                "overhead.frame_bundle_file": str(vision["frame_bundle_file_sha256"]),
                "overhead.frame_bundle": str(vision["frame_bundle_sha256"]),
                "overhead.result_json": str(vision["result_json_sha256"]),
                "overhead.camera_service_identity_file": str(vision["camera_service_identity_file_sha256"]),
                "overhead.camera_service_identity": str(vision["camera_service_identity_sha256"]),
                "overhead.capture_pipeline": str(vision["capture_pipeline_sha256"]),
                "overhead.inference_pipeline": str(vision["inference_pipeline_sha256"]),
                "overhead.consumption_receipt": file_sha256(self._vision_receipt_path(vision)),
            }
        )
        calibration_sha256 = {
            "arm02.station_config": str(members["arm02"]["artifacts"]["station_config"]["sha256"]),
        }
        try:
            manifest = CapabilityManifest(
                schema_version=self.capability_schema_version,
                manifest_id=f"dual-arm-{aggregate_snapshot_sha256[:20]}",
                subsystem=self.subsystem,
                maturity=self._binding.maturity,
                device_id=device_id,
                boot_id=boot_id,
                session_id=session_id,
                release_id=self._binding.release_id,
                capabilities=DUAL_ARM_CAPABILITIES,
                actual_backends=dict(DUAL_ARM_REQUIRED_BACKENDS),
                artifact_sha256=artifact_sha256,
                calibration_sha256=calibration_sha256,
                stations=("STATION_IDENTITY", "STATION_GRIND"),
                issued_at_ms=now_ms,
                expires_at_ms=now_ms + self._binding.ttl_ms,
            )
        except (ContractError, TypeError, ValueError, KeyError):
            return self._failure(operation, "DUAL_ARM_MANIFEST_BINDING_INVALID")
        return CapabilityReadResult(
            subsystem=self.subsystem,
            operation=operation,
            maturity=manifest.maturity,
            ready=True,
            reason_code="PASS",
            manifest=manifest,
            snapshot_sha256=aggregate_snapshot_sha256,
            details={
                "read_only": True,
                "members": {
                    arm_id: {
                        "device_id": snapshot["identity"]["machine_id_sha256"],
                        "boot_id": snapshot["identity"]["boot_id"],
                        "snapshot_sha256": member_digests[arm_id],
                    }
                    for arm_id, snapshot in members.items()
                },
                "a0_acquisition_id": vision["acquisition_id"],
                "physical_closure_proven": False,
                "r3_permit_ready": False,
            },
            network_touched=self._network_touched,
            source_kind=self._source_kind,
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
            network_touched=self._network_touched,
            source_kind=self._source_kind,
            run_binding_sha256=self._binding.run_binding_sha256,
            profile_sha256=self._binding.profile_sha256,
        )

    def _validate_member(
        self,
        arm_id: str,
        snapshot: Mapping[str, Any],
        *,
        now_ms: int,
    ) -> str | None:
        spec = _MEMBER_SPECS[arm_id]
        if set(snapshot) != _MEMBER_KEYS:
            return "DUAL_ARM_MEMBER_SCHEMA_INVALID"
        unsigned = dict(snapshot)
        claimed = unsigned.pop("snapshot_sha256", None)
        if snapshot.get("schema_version") != DUAL_ARM_MEMBER_SCHEMA or claimed != canonical_sha256(unsigned):
            return "DUAL_ARM_MEMBER_INTEGRITY_INVALID"
        if (
            snapshot.get("arm_id") != arm_id
            or snapshot.get("run_id") != self._binding.run_id
            or snapshot.get("run_nonce") != self._binding.run_nonce
            or snapshot.get("release_id") != self._binding.release_id
            or snapshot.get("profile_sha256") != self._binding.profile_sha256
        ):
            return "DUAL_ARM_MEMBER_RUN_BINDING_MISMATCH"
        observed_at_ms = snapshot.get("observed_at_ms")
        if isinstance(observed_at_ms, bool) or not isinstance(observed_at_ms, int):
            return "DUAL_ARM_MEMBER_OBSERVATION_TIME_INVALID"
        if observed_at_ms > now_ms:
            return "DUAL_ARM_MEMBER_SNAPSHOT_FUTURE"
        if now_ms - observed_at_ms > self._binding.ttl_ms:
            return "DUAL_ARM_MEMBER_SNAPSHOT_STALE"
        frozen = snapshot.get("frozen_identity")
        identity = snapshot.get("identity")
        if not isinstance(frozen, Mapping) or not isinstance(identity, Mapping):
            return "DUAL_ARM_MEMBER_IDENTITY_INVALID"
        if frozen != {
            "hostname": spec["hostname"],
            "wlan0_mac": spec["wlan0_mac"],
            "cpu_serial": spec["cpu_serial"],
        }:
            return "DUAL_ARM_MEMBER_FROZEN_IDENTITY_MISMATCH"
        if (
            set(identity) != {"hostname", "boot_id", "machine_id_sha256", "wlan0_mac", "cpu_serial"}
            or identity.get("hostname") != spec["hostname"]
            or identity.get("wlan0_mac") != spec["wlan0_mac"]
            or identity.get("cpu_serial") != spec["cpu_serial"]
            or identity.get("machine_id_sha256") != self._binding.expected_machine_id_sha256[arm_id]
            or _BOOT_RE.fullmatch(str(identity.get("boot_id", ""))) is None
        ):
            return "DUAL_ARM_MEMBER_IDENTITY_INVALID"
        if (
            self._binding.expected_boot_id is not None
            and identity.get("boot_id") != self._binding.expected_boot_id[arm_id]
        ):
            return "DUAL_ARM_MEMBER_BOOT_ID_MISMATCH"
        expected_probe = {
            "actuator_commands_issued": 0,
            "read_only_commands": {
                "systemd_show": 1,
                "crontab_list": 1,
            },
            "read_only_queries": {
                "artifact_sha256": len(spec["artifacts"]),
                "proc_cmdline_scan": 1,
                "proc_fd_owner_scan": 1,
                "video0_usb_identity": 1,
            },
            "hardware_touched": False,
            "execution_authority": False,
            "serial_opened": False,
            "camera_opened": False,
            "physical_closure": False,
        }
        if snapshot.get("probe") != expected_probe:
            return "DUAL_ARM_MEMBER_AUTHORITY_ESCALATION"
        reasons = snapshot.get("reasons")
        if not isinstance(reasons, list) or reasons != sorted(set(reasons)):
            return "DUAL_ARM_MEMBER_REASON_SET_INVALID"
        if snapshot.get("ready") is True:
            if reasons != []:
                return "DUAL_ARM_MEMBER_READY_STATE_INVALID"
        elif not reasons:
            return "DUAL_ARM_MEMBER_HOLD_STATE_INVALID"
        member = snapshot.get("member")
        if (
            not isinstance(member, Mapping)
            or set(member)
            != {
                "physical_side",
                "rb_voe_role",
                "declared_tool_role",
                "code_profile_binding_verified",
                "physical_tool_presence_verified",
                "physical_closure",
                "verification_basis",
            }
            or member.get("physical_side") != spec["physical_side"]
            or member.get("rb_voe_role") != spec["rb_voe_role"]
            or member.get("declared_tool_role") != spec["tool_role"]
            or member.get("physical_tool_presence_verified") is not False
            or member.get("physical_closure") is not False
            or member.get("verification_basis") != "frozen_identity_code_profile_and_finals_semantics"
            or not isinstance(member.get("code_profile_binding_verified"), bool)
            or (snapshot.get("ready") is True and member.get("code_profile_binding_verified") is not True)
        ):
            return "DUAL_ARM_MEMBER_ROLE_INVALID"
        systemd = snapshot.get("systemd")
        if not isinstance(systemd, Mapping) or set(systemd) != {
            "unit",
            "active",
            "enabled",
            "invocation_id",
            "query_ok",
            "matches_frozen_state",
        }:
            return "DUAL_ARM_MEMBER_SYSTEMD_INVALID"
        frozen_inactive = systemd == {
            "unit": spec["systemd_unit"],
            "active": "inactive",
            "enabled": "disabled",
            "invocation_id": "",
            "query_ok": True,
            "matches_frozen_state": True,
        }
        camera_service_active = bool(
            arm_id == "arm02"
            and systemd.get("unit") == spec["systemd_unit"]
            and systemd.get("active") == "active"
            and systemd.get("enabled") == "disabled"
            and _INVOCATION_ID_RE.fullmatch(str(systemd.get("invocation_id", ""))) is not None
            and systemd.get("query_ok") is True
            and systemd.get("matches_frozen_state") is True
        )
        if snapshot.get("ready") is True and not (frozen_inactive or camera_service_active):
            return "DUAL_ARM_MEMBER_MOTION_SURFACE_OPEN"
        cron = snapshot.get("cron")
        expected_cron = {
            "required": True,
            "executed": True,
            "query_ok": True,
            "forbidden_surfaces": list(spec["forbidden_cron_surfaces"]),
            "forbidden_entries": [],
        }
        if not isinstance(cron, Mapping) or set(cron) != set(expected_cron):
            return "DUAL_ARM_MEMBER_CRON_INVALID"
        if (
            cron.get("required") is not True
            or cron.get("executed") is not True
            or not isinstance(cron.get("query_ok"), bool)
            or cron.get("forbidden_surfaces") != list(spec["forbidden_cron_surfaces"])
            or not isinstance(cron.get("forbidden_entries"), list)
        ):
            return "DUAL_ARM_MEMBER_CRON_INVALID"
        for entry in cron.get("forbidden_entries", []):
            if (
                not isinstance(entry, Mapping)
                or set(entry) != {"line_number", "surface", "line_sha256"}
                or isinstance(entry.get("line_number"), bool)
                or not isinstance(entry.get("line_number"), int)
                or int(entry["line_number"]) <= 0
                or entry.get("surface") not in spec["forbidden_cron_surfaces"]
                or re.fullmatch(r"[0-9a-f]{64}", str(entry.get("line_sha256", ""))) is None
            ):
                return "DUAL_ARM_MEMBER_CRON_INVALID"
        if snapshot.get("ready") is True and cron != expected_cron:
            return "DUAL_ARM_MEMBER_MOTION_SURFACE_OPEN"
        processes = snapshot.get("processes")
        if not isinstance(processes, Mapping) or set(processes) != {
            "scan_complete",
            "forbidden_surfaces",
            "dangerous_matches",
        }:
            return "DUAL_ARM_MEMBER_PROCESS_SURFACE_INVALID"
        if processes.get("forbidden_surfaces") != list(spec["forbidden_process_surfaces"]) or not isinstance(
            processes.get("dangerous_matches"), list
        ):
            return "DUAL_ARM_MEMBER_PROCESS_SURFACE_INVALID"
        for match in processes.get("dangerous_matches", []):
            if (
                not isinstance(match, Mapping)
                or set(match) != {"pid", "comm", "surface", "cmdline_sha256"}
                or isinstance(match.get("pid"), bool)
                or not isinstance(match.get("pid"), int)
                or int(match["pid"]) <= 0
                or not isinstance(match.get("comm"), str)
                or not match["comm"]
                or match.get("surface") not in spec["forbidden_process_surfaces"]
                or re.fullmatch(r"[0-9a-f]{64}", str(match.get("cmdline_sha256", ""))) is None
            ):
                return "DUAL_ARM_MEMBER_PROCESS_SURFACE_INVALID"
        if snapshot.get("ready") is True and (
            processes.get("scan_complete") is not True or processes.get("dangerous_matches") != []
        ):
            return "DUAL_ARM_MEMBER_MOTION_SURFACE_OPEN"
        devices = snapshot.get("devices")
        if not isinstance(devices, Mapping) or set(devices) != {
            "owner_scan_complete",
            "ttyAMA0",
            "video0",
        }:
            return "DUAL_ARM_MEMBER_DEVICE_SURFACE_INVALID"
        if not isinstance(devices.get("owner_scan_complete"), bool):
            return "DUAL_ARM_MEMBER_DEVICE_SURFACE_INVALID"
        ttyama0 = devices.get("ttyAMA0")
        video0 = devices.get("video0")
        if not isinstance(ttyama0, Mapping) or set(ttyama0) != {
            "path",
            "present",
            "owners",
        }:
            return "DUAL_ARM_MEMBER_DEVICE_SURFACE_INVALID"
        if not isinstance(video0, Mapping) or set(video0) != {
            "path",
            "present",
            "owners",
            "usb_identity",
        }:
            return "DUAL_ARM_MEMBER_DEVICE_SURFACE_INVALID"
        for name, record in (("ttyAMA0", ttyama0), ("video0", video0)):
            owners = record.get("owners")
            if (
                record.get("path") != f"/dev/{name}"
                or not isinstance(record.get("present"), bool)
                or not isinstance(owners, list)
            ):
                return "DUAL_ARM_MEMBER_DEVICE_SURFACE_INVALID"
            for owner in owners:
                if (
                    not isinstance(owner, Mapping)
                    or set(owner) != {"pid", "comm"}
                    or isinstance(owner.get("pid"), bool)
                    or not isinstance(owner.get("pid"), int)
                    or int(owner["pid"]) <= 0
                    or not isinstance(owner.get("comm"), str)
                ):
                    return "DUAL_ARM_MEMBER_DEVICE_SURFACE_INVALID"
        camera_identity = video0.get("usb_identity")
        expected_camera_usb_id = spec["camera_usb_id"]
        if not isinstance(camera_identity, Mapping) or set(camera_identity) != {
            "query_ok",
            "source",
            "id_vendor",
            "id_product",
            "usb_id",
            "expected_usb_id",
            "matches_expected",
        }:
            return "DUAL_ARM_MEMBER_CAMERA_IDENTITY_INVALID"
        if (
            not isinstance(camera_identity.get("query_ok"), bool)
            or not _is_sysfs_usb_identity_source(camera_identity.get("source"))
            or camera_identity.get("expected_usb_id") != expected_camera_usb_id
        ):
            return "DUAL_ARM_MEMBER_CAMERA_IDENTITY_INVALID"
        camera_query_ok = camera_identity["query_ok"] is True
        camera_vendor = camera_identity.get("id_vendor")
        camera_product = camera_identity.get("id_product")
        camera_usb_id = camera_identity.get("usb_id")
        if camera_query_ok:
            if (
                not isinstance(camera_vendor, str)
                or _USB_COMPONENT_RE.fullmatch(camera_vendor) is None
                or not isinstance(camera_product, str)
                or _USB_COMPONENT_RE.fullmatch(camera_product) is None
                or camera_usb_id != f"{camera_vendor}:{camera_product}"
            ):
                return "DUAL_ARM_MEMBER_CAMERA_IDENTITY_INVALID"
        elif (camera_vendor, camera_product, camera_usb_id) != ("", "", ""):
            return "DUAL_ARM_MEMBER_CAMERA_IDENTITY_INVALID"
        expected_camera_match = (
            camera_query_ok and camera_usb_id == expected_camera_usb_id
            if expected_camera_usb_id is not None
            else None
        )
        if camera_identity.get("matches_expected") is not expected_camera_match:
            return "DUAL_ARM_MEMBER_CAMERA_IDENTITY_INVALID"
        if expected_camera_usb_id is not None:
            if not camera_query_ok:
                return "DUAL_ARM_MEMBER_CAMERA_IDENTITY_UNVERIFIED"
            if expected_camera_match is not True:
                return "DUAL_ARM_MEMBER_CAMERA_IDENTITY_MISMATCH"
        if snapshot.get("ready") is True:
            if devices.get("owner_scan_complete") is not True:
                return "DUAL_ARM_MEMBER_OWNER_SCAN_INCOMPLETE"
            video_owners = video0.get("owners")
            video_owner_state_valid = (
                isinstance(video_owners, list)
                and len(video_owners) == 1
                and camera_service_active
                and isinstance(video_owners[0].get("pid"), int)
                and int(video_owners[0]["pid"]) > 0
                and bool(video_owners[0].get("comm"))
            ) or (video_owners == [] and not camera_service_active)
            if (
                ttyama0.get("present") is not True
                or ttyama0.get("owners") != []
                or video0.get("present") is not True
                or not video_owner_state_valid
            ):
                return "DUAL_ARM_MEMBER_DEVICE_SURFACE_OPEN"
            if not camera_query_ok:
                return "DUAL_ARM_MEMBER_CAMERA_IDENTITY_UNVERIFIED"
        artifacts = snapshot.get("artifacts")
        expected_artifacts = {
            name: (
                artifact["path"],
                self._binding.probe_script_sha256 if name == "probe_script" else artifact["sha256"],
            )
            for name, artifact in spec["artifacts"].items()
        }
        if not isinstance(artifacts, Mapping) or set(artifacts) != set(expected_artifacts):
            return "DUAL_ARM_MEMBER_ARTIFACT_INVENTORY_INVALID"
        for name, (path, expected_hash) in expected_artifacts.items():
            record = artifacts.get(name)
            if not isinstance(record, Mapping) or set(record) != {
                "path",
                "present",
                "sha256",
                "expected_sha256",
                "matches",
            }:
                return "DUAL_ARM_MEMBER_ARTIFACT_INVENTORY_INVALID"
            if (
                record.get("path") != path
                or record.get("expected_sha256") != expected_hash
                or record.get("present") is not True
                or record.get("sha256") != expected_hash
                or record.get("matches") is not True
            ):
                return "DUAL_ARM_MEMBER_ARTIFACT_RELEASE_MISMATCH"
        return None

    def _vision_receipt_path(self, record: Mapping[str, Any]) -> Path:
        return a0_contract.replay_receipt_path(
            self._binding.vision_replay_ledger_dir,
            record,
        )

    def _validate_vision(
        self,
        record: Mapping[str, Any],
        *,
        members: Mapping[str, Mapping[str, Any]],
        now_ms: int,
    ) -> str | None:
        observed_at_ms = record.get("observed_at_ms")
        if isinstance(observed_at_ms, bool) or not isinstance(observed_at_ms, int):
            return "DUAL_ARM_VISION_RECONSTRUCTION_INVALID"
        if any(
            not isinstance(member.get("observed_at_ms"), int)
            or isinstance(member.get("observed_at_ms"), bool)
            or int(member["observed_at_ms"]) < observed_at_ms
            for member in members.values()
        ):
            return "DUAL_ARM_MEMBER_PRECEDES_A0_OBSERVATION"
        try:
            member_video0 = members["arm02"]["devices"]["video0"]
            member_camera_identity = member_video0["usb_identity"]
            if (
                record.get("camera_source") != member_video0["path"]
                or record.get("camera_usb_id") != member_camera_identity["usb_id"]
                or member_camera_identity["query_ok"] is not True
                or member_camera_identity["matches_expected"] is not True
            ):
                return "DUAL_ARM_VISION_CAMERA_IDENTITY_MISMATCH"
        except (KeyError, TypeError):
            return "DUAL_ARM_VISION_CAMERA_IDENTITY_MISMATCH"
        try:
            arm02_systemd = members["arm02"]["systemd"]
            arm02_video_owners = members["arm02"]["devices"]["video0"]["owners"]
            if arm02_systemd["active"] == "active":
                service_pid = int(record["camera_service_main_pid"])
                if (
                    arm02_systemd["unit"] != _MEMBER_SPECS["arm02"]["systemd_unit"]
                    or arm02_systemd["enabled"] != "disabled"
                    or len(arm02_video_owners) != 1
                    or int(arm02_video_owners[0]["pid"]) != service_pid
                    or int(record["camera_service_video_owner_pid"]) != service_pid
                    or int(record["camera_service_listener_owner_pid"]) != service_pid
                    or record["camera_service_unit_name"] != _MEMBER_SPECS["arm02"]["systemd_unit"]
                ):
                    return "DUAL_ARM_VISION_CAMERA_SERVICE_HANDOFF_MISMATCH"
            elif arm02_systemd["active"] == "inactive":
                if arm02_video_owners != []:
                    return "DUAL_ARM_VISION_CAMERA_SERVICE_HANDOFF_MISMATCH"
            else:
                return "DUAL_ARM_VISION_CAMERA_SERVICE_HANDOFF_MISMATCH"
        except (KeyError, TypeError, ValueError):
            return "DUAL_ARM_VISION_CAMERA_SERVICE_HANDOFF_MISMATCH"
        try:
            expected_capture_boot_id = (
                self._binding.expected_boot_id["arm02"]
                if self._binding.expected_boot_id is not None
                else str(members["arm02"]["identity"]["boot_id"])
            )
            if record.get("capture_boot_id") != expected_capture_boot_id:
                return "DUAL_ARM_VISION_CAPTURE_BOOT_ID_MISMATCH"
            expected = a0_contract.ExpectedAcquisition(
                acquisition_id=self._binding.vision_acquisition_id,
                a0_run_id=self._binding.vision_a0_run_id,
                r2_run_id=self._binding.run_id,
                r2_run_nonce=self._binding.run_nonce,
                challenge_sha256=self._binding.vision_challenge_sha256,
                challenge_issued_at_ms=self._binding.vision_challenge_issued_at_ms,
                challenge_expires_at_ms=self._binding.vision_challenge_expires_at_ms,
                release_id=self._binding.release_id,
                config_sha256=self._binding.vision_config_sha256,
                case_id=self._binding.vision_case_id,
                sample_id=self._binding.vision_sample_id,
                sample_lineage_sha256=self._binding.vision_sample_lineage_sha256,
                parent_evidence_root_sha256=(self._binding.vision_parent_evidence_root_sha256),
                bag_empty_baseline_sha256=(self._binding.vision_bag_empty_baseline_sha256),
                task_kind=self._binding.vision_task_kind,
                result_schema=self._binding.vision_result_schema,
                success_state=self._binding.vision_success_state,
                capture_device_id=("machine-sha256:" + self._binding.expected_machine_id_sha256["arm02"]),
                capture_boot_id=expected_capture_boot_id,
                capture_session_id=self._binding.vision_capture_session_id,
                inference_device_id=self._binding.ai_x5_device_id,
                inference_boot_id=self._binding.ai_x5_boot_id,
                inference_session_id=self._binding.vision_inference_session_id,
            )
            a0_contract._validate_sealed_record_with_clock(
                record,
                replay_ledger_dir=self._binding.vision_replay_ledger_dir,
                acquisition_manifest=self._binding.vision_acquisition_manifest,
                raw_frame=self._binding.vision_raw_frame,
                frame_bundle_artifact=self._binding.vision_frame_bundle_artifact,
                result_json=self._binding.vision_result_json,
                capture_pipeline_artifact=(self._binding.vision_capture_pipeline_artifact),
                inference_pipeline_artifact=(self._binding.vision_inference_pipeline_artifact),
                camera_service_identity_artifact=(self._binding.vision_camera_service_identity_artifact),
                expected=expected,
                now_ms=now_ms,
                max_age_ms=self._binding.vision_max_age_ms,
            )
        except (a0_contract.ActualRecordError, KeyError, OSError, TypeError, ValueError) as exc:
            temporal_markers = (
                "stale",
                "in the future",
                "expired",
                "after challenge expiry",
            )
            if any(marker in str(exc).lower() for marker in temporal_markers):
                return "NOT_READY"
            return "DUAL_ARM_VISION_RECONSTRUCTION_INVALID"
        return None if record["result_success"] is True else "NOT_READY"


__all__ = [
    "DUAL_ARM_CAPABILITIES",
    "DUAL_ARM_FINALS_ORCHESTRATOR_SHA256",
    "DUAL_ARM_FINALS_STATION_CONFIG_SHA256",
    "DUAL_ARM_MEMBER_PROBE_SHA256",
    "DUAL_ARM_MEMBER_SCHEMA",
    "DUAL_ARM_MEMBER_PATHS",
    "DUAL_ARM_PROFILE_SHA256",
    "DUAL_ARM_REQUIRED_BACKENDS",
    "DUAL_ARM_VISION_RECORD_PATH",
    "DualArmAdapter",
    "DualArmCapabilityBinding",
    "DualArmReadOnlyAdapter",
    "DualArmTargetAdapter",
]
