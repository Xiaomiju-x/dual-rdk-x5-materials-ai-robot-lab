"""Strict, immutable semantic-profile records implemented with the standard library."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from rb_voe.contracts.canonical import canonical_sha256, is_sha256
from rb_voe.contracts.models import CapabilityManifest, Maturity

SEMANTIC_PROFILE_SCHEMA_VERSION: Final[str] = "xrd-rb-voe-semantic-profile-v1"


class SemanticProfileError(ValueError):
    """Raised when a semantic profile or a caller binding fails closed."""


class SemanticProfileMode(str, Enum):
    LIVE_READONLY = "LIVE_READONLY"
    TARGET_ONLY = "TARGET_ONLY"


@dataclass(frozen=True, slots=True)
class _FrozenProfileSpec:
    subsystem: str
    manifest_schema_version: str
    read_mode: SemanticProfileMode
    maturity: Maturity
    capabilities: tuple[str, ...]
    required_backends: Mapping[str, str]
    required_artifacts: Mapping[str, str]
    execution_authority: bool
    r3_permit_ready: bool
    profile_sha256: str


_AI_X5_CAPABILITIES: Final[tuple[str, ...]] = (
    "ai_x5.xrd_vision.bpu_derived",
    "ai_x5.xrd_numerical.bpu_derived",
    "ai_x5.spectrum_vision.bpu_derived",
    "ai_x5.spectrum_numerical.bpu_derived",
)
_AI_X5_BACKENDS: Final[Mapping[str, str]] = MappingProxyType(
    {capability: "hobot_dnn.Bayes-e.INT8" for capability in _AI_X5_CAPABILITIES}
)
_EMBODIED_X5_CAPABILITIES: Final[tuple[str, ...]] = (
    "embodied_x5.geometry.self_filtered_live",
    "embodied_x5.localization.online_slam_live",
    "embodied_x5.state_estimation.ekf_live",
    "embodied_x5.f407.hardware_safety_readonly",
    "embodied_x5.collision_monitor.veto_chain",
    "embodied_x5.lab_fsd.shadow_risk",
    "embodied_x5.tiny_occ_risk.bpu_actual",
    "embodied_x5.mppi.bpu_proposed_only_actual",
)
_EMBODIED_X5_BACKENDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "embodied_x5.geometry.self_filtered_live": (
            "ros2.ld14.scan_raw_to_self_filter_to_scan.astra_scan_depth.live"
        ),
        "embodied_x5.localization.online_slam_live": ("ros2.slam_toolbox.map_and_map_to_odom.fresh_live"),
        "embodied_x5.state_estimation.ekf_live": (
            "ros2.f407_wheel_odom_plus_imu.robot_localization_ekf.odom_live"
        ),
        "embodied_x5.f407.hardware_safety_readonly": (
            "f407.0xaa55.safety_state_and_firmware_info.ros2_readonly"
        ),
        "embodied_x5.collision_monitor.veto_chain": (
            "ros2.collision_monitor.scan_plus_scan_depth.cmd_vel_to_cmd_vel_safe_to_serial_f407.veto"
        ),
        "embodied_x5.lab_fsd.shadow_risk": ("ros2.lab_fsd.live_inputs.future_risk.safety_gate.shadow_only"),
        "embodied_x5.tiny_occ_risk.bpu_actual": ("hobot_dnn.Bayes-e.INT8.tiny_occ_risk.forward_actual"),
        "embodied_x5.mppi.bpu_proposed_only_actual": (
            "hobot_dnn.Bayes-e.INT8.mppi_cost.proposed_only_actual"
        ),
    }
)
_EMBODIED_X5_REQUIRED_ARTIFACTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "body_contour": "/home/rdk/rb_voe/lidar_body_contour.v1.json",
    }
)
_DUAL_ARM_CAPABILITIES: Final[tuple[str, ...]] = (
    "dual_arm.station_identity.live",
    "dual_arm.motion_surface.closed_readonly",
    "dual_arm.arm01.finals_artifact_bundle.readonly",
    "dual_arm.arm02.finals_artifact_bundle.readonly",
    "dual_arm.overhead.vision_gate.run_bound_actual",
)
_DUAL_ARM_BACKENDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "dual_arm.station_identity.live": "dual_pi.strict_readonly_probe.station_identity",
        "dual_arm.motion_surface.closed_readonly": ("dual_pi.strict_readonly_probe.motion_surface_closed"),
        "dual_arm.arm01.finals_artifact_bundle.readonly": ("dual_pi.finals_artifact_bundle.arm01.readonly"),
        "dual_arm.arm02.finals_artifact_bundle.readonly": ("dual_pi.finals_artifact_bundle.arm02.readonly"),
        "dual_arm.overhead.vision_gate.run_bound_actual": (
            "arm02.v4l2.capture_to_ai_x5.cpu_vision.run_bound_actual"
        ),
    }
)

# The pins are updated only when the corresponding canonical profile is deliberately revised.
_FROZEN_PROFILE_SPECS: Final[Mapping[str, _FrozenProfileSpec]] = MappingProxyType(
    {
        "ai_x5.v1": _FrozenProfileSpec(
            subsystem="ai_x5",
            manifest_schema_version="xrd-rb-voe-ai-capability-v1",
            read_mode=SemanticProfileMode.LIVE_READONLY,
            maturity=Maturity.SHADOW_VALIDATED,
            capabilities=_AI_X5_CAPABILITIES,
            required_backends=_AI_X5_BACKENDS,
            required_artifacts=MappingProxyType({}),
            execution_authority=False,
            r3_permit_ready=False,
            profile_sha256="13b9e406438709106d3d48af6deb3b1ef10558c54a18e602da0420606138bb84",
        ),
        "embodied_x5.v1": _FrozenProfileSpec(
            subsystem="embodied_x5",
            manifest_schema_version="xrd-rb-voe-embodied-capability-v1",
            read_mode=SemanticProfileMode.LIVE_READONLY,
            maturity=Maturity.REPLAY_VALIDATED,
            capabilities=_EMBODIED_X5_CAPABILITIES,
            required_backends=_EMBODIED_X5_BACKENDS,
            required_artifacts=_EMBODIED_X5_REQUIRED_ARTIFACTS,
            execution_authority=False,
            r3_permit_ready=False,
            profile_sha256="c0d5488e7371b9ddab85537b7397b4d6cd492fea1683ccc7ad30351cc1feb793",
        ),
        "dual_arm.v1": _FrozenProfileSpec(
            subsystem="dual_arm",
            manifest_schema_version="xrd-rb-voe-dual-arm-capability-v1",
            read_mode=SemanticProfileMode.LIVE_READONLY,
            maturity=Maturity.SHADOW_VALIDATED,
            capabilities=_DUAL_ARM_CAPABILITIES,
            required_backends=_DUAL_ARM_BACKENDS,
            required_artifacts=MappingProxyType({}),
            execution_authority=False,
            r3_permit_ready=False,
            profile_sha256="18ec8e10b9cf13bc4075f6873061d338020f39bfbb9ad0b509e6b444b657d538",
        ),
        "assay_station.v1": _FrozenProfileSpec(
            subsystem="assay_station",
            manifest_schema_version="xrd-rb-voe-assay-station-capability-v1",
            read_mode=SemanticProfileMode.TARGET_ONLY,
            maturity=Maturity.TARGET_ONLY,
            capabilities=(),
            required_backends=MappingProxyType({}),
            required_artifacts=MappingProxyType({}),
            execution_authority=False,
            r3_permit_ready=False,
            profile_sha256="9d77a8d48ffdeee65efda4f35812a9e78350f083f934622c4529643c387c0f80",
        ),
    }
)

PROFILE_IDS: Final[tuple[str, ...]] = tuple(_FROZEN_PROFILE_SPECS)
PROFILE_SHA256_BY_ID: Final[Mapping[str, str]] = MappingProxyType(
    {profile_id: spec.profile_sha256 for profile_id, spec in _FROZEN_PROFILE_SPECS.items()}
)

_PROFILE_REQUIRED_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "profile_id",
        "subsystem",
        "manifest_schema_version",
        "read_mode",
        "maturity",
        "capabilities",
        "required_backends",
        "execution_authority",
        "r3_permit_ready",
        "profile_sha256",
    }
)
_PROFILE_OPTIONAL_FIELDS: Final[frozenset[str]] = frozenset({"required_artifacts"})


@dataclass(frozen=True, slots=True)
class SemanticProfile:
    """One content-addressed semantic ceiling; loading it does not prove readiness."""

    schema_version: str
    profile_id: str
    subsystem: str
    manifest_schema_version: str
    read_mode: SemanticProfileMode
    maturity: Maturity
    capabilities: tuple[str, ...]
    required_backends: Mapping[str, str]
    required_artifacts: Mapping[str, str]
    execution_authority: bool
    r3_permit_ready: bool
    profile_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != SEMANTIC_PROFILE_SCHEMA_VERSION:
            raise SemanticProfileError("unsupported semantic profile schema")
        try:
            spec = _FROZEN_PROFILE_SPECS[self.profile_id]
        except KeyError as exc:
            raise SemanticProfileError(f"unregistered semantic profile: {self.profile_id!r}") from exc
        if not isinstance(self.read_mode, SemanticProfileMode):
            raise SemanticProfileError("read_mode must be a SemanticProfileMode")
        if not isinstance(self.maturity, Maturity):
            raise SemanticProfileError("maturity must be a Maturity")
        if not isinstance(self.capabilities, tuple) or any(
            not isinstance(capability, str) or not capability for capability in self.capabilities
        ):
            raise SemanticProfileError("capabilities must be a tuple of non-empty strings")
        if len(self.capabilities) != len(set(self.capabilities)):
            raise SemanticProfileError("capabilities must be unique")
        if not isinstance(self.required_backends, Mapping):
            raise SemanticProfileError("required_backends must be an object")
        if any(
            not isinstance(capability, str) or not capability or not isinstance(backend, str) or not backend
            for capability, backend in self.required_backends.items()
        ):
            raise SemanticProfileError("required_backends must bind non-empty strings")
        if set(self.required_backends) != set(self.capabilities):
            raise SemanticProfileError("required_backends must bind every capability exactly once")
        if not isinstance(self.required_artifacts, Mapping):
            raise SemanticProfileError("required_artifacts must be an object")
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(path, str)
            or not path.startswith("/")
            or ".." in Path(path).parts
            for name, path in self.required_artifacts.items()
        ):
            raise SemanticProfileError("required_artifacts must bind non-empty names to absolute safe paths")
        if self.execution_authority is not False or self.r3_permit_ready is not False:
            raise SemanticProfileError("R2-PREP profiles cannot grant execution or R3 permit authority")
        if self.read_mode is SemanticProfileMode.LIVE_READONLY:
            if self.maturity not in {
                Maturity.REPLAY_VALIDATED,
                Maturity.SHADOW_VALIDATED,
            }:
                raise SemanticProfileError(
                    "live readonly profiles are limited to replay or shadow validation"
                )
            if not self.capabilities:
                raise SemanticProfileError("live semantic profiles require capabilities")
        elif (
            self.maturity is not Maturity.TARGET_ONLY
            or self.capabilities
            or self.required_backends
            or self.required_artifacts
        ):
            raise SemanticProfileError("TARGET_ONLY profiles cannot declare live bindings")

        observed = (
            self.subsystem,
            self.manifest_schema_version,
            self.read_mode,
            self.maturity,
            self.capabilities,
            dict(self.required_backends),
            dict(self.required_artifacts),
            self.execution_authority,
            self.r3_permit_ready,
        )
        expected = (
            spec.subsystem,
            spec.manifest_schema_version,
            spec.read_mode,
            spec.maturity,
            spec.capabilities,
            dict(spec.required_backends),
            dict(spec.required_artifacts),
            spec.execution_authority,
            spec.r3_permit_ready,
        )
        if observed != expected:
            if dict(self.required_backends) != dict(spec.required_backends):
                raise SemanticProfileError("required backend binding drift")
            raise SemanticProfileError("semantic profile differs from its frozen registry entry")
        if not is_sha256(self.profile_sha256):
            raise SemanticProfileError("profile_sha256 must be a lowercase SHA-256 digest")
        computed = canonical_sha256(self.unsigned_payload())
        if self.profile_sha256 != computed:
            raise SemanticProfileError("semantic profile canonical digest mismatch")
        if self.profile_sha256 != spec.profile_sha256:
            raise SemanticProfileError("semantic profile digest drifted from its bundled pin")
        object.__setattr__(self, "required_backends", MappingProxyType(dict(self.required_backends)))
        object.__setattr__(
            self,
            "required_artifacts",
            MappingProxyType(dict(self.required_artifacts)),
        )

    def unsigned_payload(self) -> dict[str, Any]:
        """Return the exact canonical payload covered by ``profile_sha256``."""
        payload = {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "subsystem": self.subsystem,
            "manifest_schema_version": self.manifest_schema_version,
            "read_mode": self.read_mode.value,
            "maturity": self.maturity.value,
            "capabilities": list(self.capabilities),
            "required_backends": dict(self.required_backends),
            "execution_authority": self.execution_authority,
            "r3_permit_ready": self.r3_permit_ready,
        }
        if self.required_artifacts:
            payload["required_artifacts"] = dict(self.required_artifacts)
        return payload

    def to_dict(self) -> dict[str, Any]:
        payload = self.unsigned_payload()
        payload["profile_sha256"] = self.profile_sha256
        return payload

    def validate_bindings(
        self,
        *,
        maturity: Maturity | str,
        capabilities: Sequence[str],
        actual_backends: Mapping[str, str],
    ) -> None:
        """Reject caller promotion, capability drift, and backend fallback."""
        requested_maturity = _coerce_maturity(maturity)
        if requested_maturity is not self.maturity:
            raise SemanticProfileError("caller maturity override is forbidden")
        if isinstance(capabilities, (str, bytes, bytearray)) or not isinstance(capabilities, Sequence):
            raise SemanticProfileError("caller capabilities must be a sequence")
        observed_capabilities = tuple(capabilities)
        if observed_capabilities != self.capabilities:
            raise SemanticProfileError("caller capabilities differ from the semantic profile")
        if not isinstance(actual_backends, Mapping):
            raise SemanticProfileError("caller actual_backends must be an object")
        if any(
            not isinstance(key, str) or not key or not isinstance(value, str) or not value
            for key, value in actual_backends.items()
        ):
            raise SemanticProfileError("caller actual_backends must bind non-empty strings")
        expected_keys = (
            set(self.capabilities) if self.read_mode is SemanticProfileMode.LIVE_READONLY else set()
        )
        if set(actual_backends) != expected_keys:
            raise SemanticProfileError(
                "caller actual_backends must bind each profile capability exactly once"
            )
        for capability, required_backend in self.required_backends.items():
            if actual_backends[capability] != required_backend:
                raise SemanticProfileError(f"illegal backend for {capability}")

    def validate_manifest(
        self,
        manifest: CapabilityManifest,
        *,
        caller_maturity: Maturity | str | None = None,
    ) -> None:
        """Validate a typed capability manifest against this semantic ceiling."""
        if not isinstance(manifest, CapabilityManifest):
            raise SemanticProfileError("manifest must be a CapabilityManifest")
        if manifest.subsystem != self.subsystem:
            raise SemanticProfileError("manifest subsystem differs from the semantic profile")
        if manifest.schema_version != self.manifest_schema_version:
            raise SemanticProfileError("manifest schema differs from the semantic profile")
        if caller_maturity is not None and _coerce_maturity(caller_maturity) is not manifest.maturity:
            raise SemanticProfileError("caller maturity differs from the manifest")
        self.validate_bindings(
            maturity=manifest.maturity,
            capabilities=manifest.capabilities,
            actual_backends=manifest.actual_backends,
        )
        missing_artifacts = sorted(set(self.required_artifacts) - set(manifest.artifact_sha256))
        if missing_artifacts:
            raise SemanticProfileError(
                "manifest is missing profile-required artifacts: " + ", ".join(missing_artifacts)
            )


def _coerce_maturity(value: Maturity | str) -> Maturity:
    if isinstance(value, Maturity):
        return value
    if not isinstance(value, str):
        raise SemanticProfileError("maturity must be a registered string")
    try:
        return Maturity(value)
    except ValueError as exc:
        raise SemanticProfileError(f"unregistered maturity: {value!r}") from exc


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SemanticProfileError(f"duplicate JSON field: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise SemanticProfileError(f"non-finite JSON value is forbidden: {value}")


def _decode_profile_document(raw: bytes, *, expected_profile_id: str) -> SemanticProfile:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except SemanticProfileError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SemanticProfileError("bundled semantic profile is not strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise SemanticProfileError("semantic profile root must be an object")
    fields = set(payload)
    missing = sorted(_PROFILE_REQUIRED_FIELDS - fields)
    extra = sorted(fields - _PROFILE_REQUIRED_FIELDS - _PROFILE_OPTIONAL_FIELDS)
    if missing:
        raise SemanticProfileError(f"semantic profile is missing fields: {', '.join(missing)}")
    if extra:
        raise SemanticProfileError(f"semantic profile contains extra fields: {', '.join(extra)}")
    if payload["profile_id"] != expected_profile_id:
        raise SemanticProfileError("semantic profile resource identity mismatch")
    for field in ("schema_version", "profile_id", "subsystem", "manifest_schema_version"):
        if not isinstance(payload[field], str) or not payload[field]:
            raise SemanticProfileError(f"{field} must be a non-empty string")
    if not isinstance(payload["read_mode"], str) or not isinstance(payload["maturity"], str):
        raise SemanticProfileError("read_mode and maturity must be strings")
    if not isinstance(payload["capabilities"], list):
        raise SemanticProfileError("capabilities must be an array")
    if not isinstance(payload["required_backends"], dict):
        raise SemanticProfileError("required_backends must be an object")
    required_artifacts = payload.get("required_artifacts", {})
    if not isinstance(required_artifacts, dict):
        raise SemanticProfileError("required_artifacts must be an object")
    if type(payload["execution_authority"]) is not bool or type(payload["r3_permit_ready"]) is not bool:
        raise SemanticProfileError("authority fields must be booleans")
    if not isinstance(payload["profile_sha256"], str):
        raise SemanticProfileError("profile_sha256 must be a string")
    try:
        read_mode = SemanticProfileMode(payload["read_mode"])
        maturity = Maturity(payload["maturity"])
    except ValueError as exc:
        raise SemanticProfileError("profile contains an unregistered mode or maturity") from exc
    return SemanticProfile(
        schema_version=payload["schema_version"],
        profile_id=payload["profile_id"],
        subsystem=payload["subsystem"],
        manifest_schema_version=payload["manifest_schema_version"],
        read_mode=read_mode,
        maturity=maturity,
        capabilities=tuple(payload["capabilities"]),
        required_backends=payload["required_backends"],
        required_artifacts=required_artifacts,
        execution_authority=payload["execution_authority"],
        r3_permit_ready=payload["r3_permit_ready"],
        profile_sha256=payload["profile_sha256"],
    )


__all__ = [
    "PROFILE_IDS",
    "PROFILE_SHA256_BY_ID",
    "SEMANTIC_PROFILE_SCHEMA_VERSION",
    "SemanticProfile",
    "SemanticProfileError",
    "SemanticProfileMode",
]
