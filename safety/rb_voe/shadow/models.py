"""Immutable models for pre-physical, read-only shadow integration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from rb_voe.contracts.canonical import canonical_sha256, require_sha256, to_primitive

SHADOW_REPORT_SCHEMA_VERSION: Final[str] = "xrd-rb-voe-shadow-readiness-report-v2"
REQUIRED_SUBSYSTEMS: Final[tuple[str, ...]] = (
    "ai_x5",
    "embodied_x5",
    "dual_arm",
    "assay_station",
)


class ShadowMode(str, Enum):
    OFFLINE_REPLAY = "OFFLINE_REPLAY"
    LIVE_READONLY_SHADOW = "LIVE_READONLY_SHADOW"


class ShadowStatus(str, Enum):
    HOLD = "HOLD"
    QUARANTINE = "QUARANTINE"
    SHADOW_READY = "SHADOW_READY"


@dataclass(frozen=True, slots=True)
class ShadowRunBinding:
    run_id: str
    release_id: str
    evaluated_at_ms: int
    mode: ShadowMode
    run_nonce: str = ""
    profile_sha256: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id or not self.release_id:
            raise ValueError("shadow run identity fields must be non-empty")
        if isinstance(self.evaluated_at_ms, bool) or not isinstance(self.evaluated_at_ms, int):
            raise TypeError("evaluated_at_ms must be an integer")
        if self.evaluated_at_ms < 0:
            raise ValueError("evaluated_at_ms must be non-negative")
        if not isinstance(self.mode, ShadowMode):
            raise TypeError("mode must be a ShadowMode")
        if not isinstance(self.profile_sha256, Mapping):
            raise TypeError("profile_sha256 must be a mapping")
        if self.mode is ShadowMode.LIVE_READONLY_SHADOW:
            if len(self.run_nonce) < 16:
                raise ValueError("live shadow requires a run nonce of at least 16 characters")
            if set(self.profile_sha256) != set(REQUIRED_SUBSYSTEMS):
                raise ValueError("live shadow requires one semantic profile per subsystem")
            for subsystem, digest in self.profile_sha256.items():
                require_sha256(f"profile_sha256.{subsystem}", digest)
        elif self.run_nonce or self.profile_sha256:
            raise ValueError("offline replay cannot carry live run bindings")
        object.__setattr__(self, "profile_sha256", MappingProxyType(dict(self.profile_sha256)))

    def source_binding_sha256(self, subsystem: str) -> str:
        if self.mode is not ShadowMode.LIVE_READONLY_SHADOW or subsystem not in self.profile_sha256:
            raise ValueError("source bindings exist only for configured live shadow subsystems")
        return canonical_sha256(
            {
                "schema_version": "xrd-rb-voe-live-source-binding-v1",
                "subsystem": subsystem,
                "run_id": self.run_id,
                "run_nonce": self.run_nonce,
                "release_id": self.release_id,
                "profile_sha256": self.profile_sha256[subsystem],
            }
        )


@dataclass(frozen=True, slots=True)
class ShadowRunReport:
    run_id: str
    mode: ShadowMode
    status: ShadowStatus
    release_id: str
    evaluated_at_ms: int
    ready_subsystems: tuple[str, ...]
    manifest_sha256: Mapping[str, str]
    snapshot_sha256: Mapping[str, str]
    source_kinds: Mapping[str, str]
    profile_sha256: Mapping[str, str]
    reason_codes: Mapping[str, str]
    network_touched: bool
    hardware_touched: bool = False
    commands_issued: int = 0
    execution_authority: bool = False
    physical_closure_proven: bool = False
    physical_risk_denominator_increment: int = 0
    required_subsystems: tuple[str, ...] = REQUIRED_SUBSYSTEMS
    schema_version: str = SHADOW_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SHADOW_REPORT_SCHEMA_VERSION:
            raise ValueError("unsupported shadow report schema")
        if not self.run_id or not self.release_id:
            raise ValueError("shadow report identity fields must be non-empty")
        if not isinstance(self.mode, ShadowMode) or not isinstance(self.status, ShadowStatus):
            raise TypeError("shadow report enums are invalid")
        if tuple(self.required_subsystems) != REQUIRED_SUBSYSTEMS:
            raise ValueError("shadow report subsystem set is frozen")
        if tuple(sorted(self.ready_subsystems)) != self.ready_subsystems:
            raise ValueError("ready_subsystems must be sorted")
        if any(item not in REQUIRED_SUBSYSTEMS for item in self.ready_subsystems):
            raise ValueError("unknown ready subsystem")
        if self.hardware_touched or self.commands_issued != 0 or self.execution_authority:
            raise ValueError("shadow report cannot carry execution authority")
        if self.physical_closure_proven or self.physical_risk_denominator_increment != 0:
            raise ValueError("shadow report cannot increment physical evidence claims")
        if self.status is ShadowStatus.SHADOW_READY and set(self.ready_subsystems) != set(
            REQUIRED_SUBSYSTEMS
        ):
            raise ValueError("SHADOW_READY requires all four subsystems")
        if set(self.reason_codes) != set(REQUIRED_SUBSYSTEMS):
            raise ValueError("reason_codes must cover all four subsystems")
        if set(self.snapshot_sha256) - set(REQUIRED_SUBSYSTEMS):
            raise ValueError("snapshot_sha256 contains an unknown subsystem")
        if set(self.source_kinds) - set(REQUIRED_SUBSYSTEMS):
            raise ValueError("source_kinds contains an unknown subsystem")
        if self.mode is ShadowMode.LIVE_READONLY_SHADOW and set(self.profile_sha256) != set(
            REQUIRED_SUBSYSTEMS
        ):
            raise ValueError("live report must retain all semantic profile digests")
        if self.mode is ShadowMode.OFFLINE_REPLAY and self.profile_sha256:
            raise ValueError("offline report cannot carry live semantic profiles")
        for subsystem, digest in self.snapshot_sha256.items():
            require_sha256(f"snapshot_sha256.{subsystem}", digest)
        for subsystem, digest in self.profile_sha256.items():
            require_sha256(f"profile_sha256.{subsystem}", digest)
        object.__setattr__(self, "manifest_sha256", MappingProxyType(dict(self.manifest_sha256)))
        object.__setattr__(self, "snapshot_sha256", MappingProxyType(dict(self.snapshot_sha256)))
        object.__setattr__(self, "source_kinds", MappingProxyType(dict(self.source_kinds)))
        object.__setattr__(self, "profile_sha256", MappingProxyType(dict(self.profile_sha256)))
        object.__setattr__(self, "reason_codes", MappingProxyType(dict(self.reason_codes)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "mode": self.mode.value,
            "status": self.status.value,
            "release_id": self.release_id,
            "evaluated_at_ms": self.evaluated_at_ms,
            "required_subsystems": list(self.required_subsystems),
            "ready_subsystems": list(self.ready_subsystems),
            "manifest_sha256": to_primitive(self.manifest_sha256),
            "snapshot_sha256": to_primitive(self.snapshot_sha256),
            "source_kinds": to_primitive(self.source_kinds),
            "profile_sha256": to_primitive(self.profile_sha256),
            "reason_codes": to_primitive(self.reason_codes),
            "network_touched": self.network_touched,
            "hardware_touched": self.hardware_touched,
            "commands_issued": self.commands_issued,
            "execution_authority": self.execution_authority,
            "physical_closure_proven": self.physical_closure_proven,
            "physical_risk_denominator_increment": self.physical_risk_denominator_increment,
        }

    @property
    def content_sha256(self) -> str:
        return canonical_sha256(self.to_dict())
