"""Narrow connector protocol for manifest-only shadow integration."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, Protocol

from rb_voe.adapters.read_only import CapabilityReadResult, ReadSourceKind
from rb_voe.contracts.canonical import canonical_sha256
from rb_voe.contracts.models import Maturity
from rb_voe.contracts.validation import ContractValidationError, validate_capability_manifest


class ShadowConnector(Protocol):
    """Connector surface intentionally has no prepare, execute, or trigger method."""

    subsystem: str
    capability_schema_version: str
    source_kind: ReadSourceKind

    def get_capability_manifest(self, *, now_ms: int) -> CapabilityReadResult: ...


class ManifestPayloadConnector:
    """Offline connector for a captured capability manifest JSON object."""

    source_kind = ReadSourceKind.CAPTURED_REPLAY

    __slots__ = ("_payload", "capability_schema_version", "subsystem")

    def __init__(
        self,
        *,
        subsystem: str,
        capability_schema_version: str,
        payload: Mapping[str, Any],
    ) -> None:
        self.subsystem = subsystem
        self.capability_schema_version = capability_schema_version
        self._payload = copy.deepcopy(dict(payload))

    def get_capability_manifest(self, *, now_ms: int) -> CapabilityReadResult:
        try:
            snapshot_sha256 = canonical_sha256(self._payload)
        except (TypeError, ValueError):
            return self._failure("MANIFEST_PAYLOAD_NOT_CANONICAL")
        try:
            manifest = validate_capability_manifest(self._payload, now_ms=now_ms, require_ready=True)
        except ContractValidationError as exc:
            detail = str(exc)
            reason = (
                "CAPABILITY_MANIFEST_STALE"
                if "STALE" in detail
                else "CAPABILITY_MANIFEST_NOT_YET_VALID"
                if "NOT_YET_VALID" in detail
                else "CAPABILITY_MANIFEST_INVALID"
            )
            return self._failure(reason, snapshot_sha256=snapshot_sha256)
        if manifest.subsystem != self.subsystem or manifest.schema_version != self.capability_schema_version:
            return self._failure("CAPABILITY_MANIFEST_IDENTITY_MISMATCH", snapshot_sha256=snapshot_sha256)
        return CapabilityReadResult(
            subsystem=self.subsystem,
            operation="CAPABILITY_MANIFEST",
            maturity=manifest.maturity,
            ready=True,
            reason_code="PASS",
            manifest=manifest,
            snapshot_sha256=snapshot_sha256,
            details={"source": "captured_manifest", "read_only": True},
            source_kind=self.source_kind,
        )

    def _failure(self, reason: str, *, snapshot_sha256: str | None = None) -> CapabilityReadResult:
        return CapabilityReadResult(
            subsystem=self.subsystem,
            operation="CAPABILITY_MANIFEST",
            maturity=Maturity.TARGET_ONLY,
            ready=False,
            reason_code=reason,
            snapshot_sha256=snapshot_sha256,
            details={"source": "captured_manifest", "read_only": True},
            source_kind=self.source_kind,
        )


__all__ = ["ManifestPayloadConnector", "ShadowConnector"]
