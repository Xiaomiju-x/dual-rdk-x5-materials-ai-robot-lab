"""Fail-closed four-system shadow readiness coordinator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from rb_voe.adapters.base import AdapterResult
from rb_voe.adapters.read_only import CapabilityReadResult, ReadSourceKind
from rb_voe.contracts.models import CapabilityManifest, Maturity
from rb_voe.contracts.validation import ContractValidationError, validate_capability_manifest
from rb_voe.semantic_profiles import SemanticProfileError, load_profile
from rb_voe.shadow.models import (
    REQUIRED_SUBSYSTEMS,
    ShadowMode,
    ShadowRunBinding,
    ShadowRunReport,
    ShadowStatus,
)

SCHEMA_BY_SUBSYSTEM: Final[Mapping[str, str]] = {
    "ai_x5": "xrd-rb-voe-ai-capability-v1",
    "embodied_x5": "xrd-rb-voe-embodied-capability-v1",
    "dual_arm": "xrd-rb-voe-dual-arm-capability-v1",
    "assay_station": "xrd-rb-voe-assay-station-capability-v1",
}
_OFFLINE_MATURITIES: Final[frozenset[Maturity]] = frozenset(
    {
        Maturity.REPLAY_VALIDATED,
        Maturity.SHADOW_VALIDATED,
        Maturity.HARDWARE_PILOT,
        Maturity.LOCKED_VALIDATED,
        Maturity.DEMO_ELIGIBLE,
    }
)
_LIVE_MATURITIES: Final[frozenset[Maturity]] = frozenset({Maturity.SHADOW_VALIDATED})
_HOLD_REASONS: Final[frozenset[str]] = frozenset(
    {
        "CONNECTOR_MISSING",
        "NOT_READY",
        "CAPABILITY_MANIFEST_STALE",
        "CAPABILITY_MANIFEST_NOT_YET_VALID",
        "MATURITY_BELOW_MODE",
        "AI_X5_SNAPSHOT_UNAVAILABLE",
        "AI_X5_SNAPSHOT_STALE",
    }
)


class ShadowCoordinator:
    """Bind capability manifests without exposing any execution surface."""

    __slots__ = ("_connectors",)

    def __init__(self, connectors: Mapping[str, Any]) -> None:
        self._connectors = dict(connectors)

    def evaluate(self, binding: ShadowRunBinding) -> ShadowRunReport:
        reasons: dict[str, str] = {}
        manifest_sha256: dict[str, str] = {}
        snapshot_sha256: dict[str, str] = {}
        source_kinds: dict[str, str] = {}
        ready: list[str] = []
        network_touched = False
        quarantine = False
        claimed_capabilities: set[str] = set()

        for subsystem in REQUIRED_SUBSYSTEMS:
            connector = self._connectors.get(subsystem)
            if connector is None:
                reasons[subsystem] = "CONNECTOR_MISSING"
                continue
            if (
                getattr(connector, "subsystem", None) != subsystem
                or getattr(connector, "capability_schema_version", None) != SCHEMA_BY_SUBSYSTEM[subsystem]
            ):
                reasons[subsystem] = "CONNECTOR_IDENTITY_MISMATCH"
                quarantine = True
                continue
            try:
                result = connector.get_capability_manifest(now_ms=binding.evaluated_at_ms)
            except Exception:
                reasons[subsystem] = "CONNECTOR_READ_FAILED"
                quarantine = True
                continue
            if isinstance(result, AdapterResult):
                reasons[subsystem] = result.reason_code
                network_touched = network_touched or result.network_touched
                continue
            if not isinstance(result, CapabilityReadResult):
                reasons[subsystem] = "CONNECTOR_RESULT_INVALID"
                quarantine = True
                continue
            network_touched = network_touched or result.network_touched
            source_kinds[subsystem] = result.source_kind.value
            if result.snapshot_sha256 is not None:
                snapshot_sha256[subsystem] = result.snapshot_sha256
            if result.hardware_touched or result.execution_authority:
                reasons[subsystem] = "SHADOW_AUTHORITY_ESCALATION"
                quarantine = True
                continue
            if binding.mode is ShadowMode.OFFLINE_REPLAY:
                if result.network_touched or result.source_kind is not ReadSourceKind.CAPTURED_REPLAY:
                    reasons[subsystem] = "OFFLINE_MODE_LIVE_SOURCE"
                    quarantine = True
                    continue
            elif not result.network_touched or result.source_kind is not ReadSourceKind.LIVE_REMOTE_READ:
                reasons[subsystem] = "LIVE_MODE_REPLAY_SOURCE"
                quarantine = True
                continue
            if not result.ready or result.manifest is None:
                reasons[subsystem] = result.reason_code
                if result.reason_code not in _HOLD_REASONS:
                    quarantine = True
                continue
            manifest = self._validated_manifest(result.manifest, binding.evaluated_at_ms)
            if manifest is None:
                reasons[subsystem] = "CAPABILITY_MANIFEST_INVALID"
                quarantine = True
                continue
            maturity_set = (
                _OFFLINE_MATURITIES if binding.mode is ShadowMode.OFFLINE_REPLAY else _LIVE_MATURITIES
            )
            if manifest.maturity not in maturity_set:
                reasons[subsystem] = "MATURITY_BELOW_MODE"
                continue
            if binding.mode is ShadowMode.LIVE_READONLY_SHADOW:
                if result.profile_sha256 != binding.profile_sha256[subsystem]:
                    reasons[subsystem] = "SEMANTIC_PROFILE_MISMATCH"
                    quarantine = True
                    continue
                if result.run_binding_sha256 != binding.source_binding_sha256(subsystem):
                    reasons[subsystem] = "LIVE_RUN_BINDING_MISMATCH"
                    quarantine = True
                    continue
                try:
                    profile = load_profile(
                        subsystem,
                        expected_sha256=binding.profile_sha256[subsystem],
                    )
                except SemanticProfileError:
                    reasons[subsystem] = "SEMANTIC_PROFILE_MISMATCH"
                    quarantine = True
                    continue
                try:
                    profile.validate_manifest(manifest)
                except SemanticProfileError:
                    reasons[subsystem] = "SEMANTIC_PROFILE_MANIFEST_MISMATCH"
                    quarantine = True
                    continue
            if manifest.release_id != binding.release_id:
                reasons[subsystem] = "CAPABILITY_RELEASE_MISMATCH"
                quarantine = True
                continue
            prefix = subsystem + "."
            if any(not capability.startswith(prefix) for capability in manifest.capabilities):
                reasons[subsystem] = "CAPABILITY_OWNER_MISMATCH"
                quarantine = True
                continue
            if set(manifest.actual_backends) != set(manifest.capabilities):
                reasons[subsystem] = "CAPABILITY_BACKEND_BINDING_MISMATCH"
                quarantine = True
                continue
            duplicates = claimed_capabilities.intersection(manifest.capabilities)
            if duplicates:
                reasons[subsystem] = "CAPABILITY_DUPLICATE_CLAIM"
                quarantine = True
                continue
            claimed_capabilities.update(manifest.capabilities)
            reasons[subsystem] = "PASS"
            manifest_sha256[subsystem] = manifest.content_sha256
            ready.append(subsystem)

        if quarantine:
            status = ShadowStatus.QUARANTINE
        elif len(ready) == len(REQUIRED_SUBSYSTEMS):
            status = ShadowStatus.SHADOW_READY
        else:
            status = ShadowStatus.HOLD
        return ShadowRunReport(
            run_id=binding.run_id,
            mode=binding.mode,
            status=status,
            release_id=binding.release_id,
            evaluated_at_ms=binding.evaluated_at_ms,
            ready_subsystems=tuple(sorted(ready)),
            manifest_sha256=dict(sorted(manifest_sha256.items())),
            snapshot_sha256=dict(sorted(snapshot_sha256.items())),
            source_kinds=dict(sorted(source_kinds.items())),
            profile_sha256=dict(binding.profile_sha256),
            reason_codes={key: reasons.get(key, "CONNECTOR_MISSING") for key in REQUIRED_SUBSYSTEMS},
            network_touched=network_touched,
        )

    @staticmethod
    def _validated_manifest(manifest: CapabilityManifest, now_ms: int) -> CapabilityManifest | None:
        try:
            return validate_capability_manifest(manifest, now_ms=now_ms, require_ready=True)
        except ContractValidationError:
            return None


__all__ = ["SCHEMA_BY_SUBSYSTEM", "ShadowCoordinator"]
