"""AI-brain X5 target adapter and strict read-only live-status adapter."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from rb_voe.adapters.base import TargetOnlyAdapter
from rb_voe.adapters.read_only import CapabilityReadResult, JsonSnapshotTransport, ReadSourceKind
from rb_voe.contracts.canonical import canonical_sha256, require_sha256
from rb_voe.contracts.models import CapabilityManifest, ContractError, Maturity
from rb_voe.runtime_identity import RUNTIME_IDENTITY_SCHEMA_VERSION

AI_X5_RUNTIME_SNAPSHOT_PATH: Final[str] = "/api/rb_voe/runtime_snapshot"
AI_X5_SNAPSHOT_PATHS: Final[tuple[str, ...]] = (AI_X5_RUNTIME_SNAPSHOT_PATH,)
AI_X5_LINE_IDS: Final[tuple[str, ...]] = (
    "xrd_vision",
    "xrd_numerical",
    "spectrum_vision",
    "spectrum_numerical",
)
AI_X5_CAPABILITY_BY_LINE: Final[Mapping[str, str]] = MappingProxyType(
    {line_id: f"ai_x5.{line_id}.bpu_derived" for line_id in AI_X5_LINE_IDS}
)
AI_X5_CAPABILITIES: Final[tuple[str, ...]] = tuple(AI_X5_CAPABILITY_BY_LINE.values())
AI_X5_REQUIRED_BACKENDS: Final[Mapping[str, str]] = MappingProxyType(
    {line_id: "hobot_dnn.Bayes-e.INT8" for line_id in AI_X5_LINE_IDS}
)
AI_X5_CPU_MODELS: Final[tuple[str, ...]] = ()
AI_X5_BPU_SLOTS: Final[tuple[str, ...]] = ()
_BOOT_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
AI_X5_RUNTIME_SNAPSHOT_SCHEMA: Final[str] = "xrd-rb-voe-ai-runtime-snapshot-v2"
AI_X5_RUNTIME_SNAPSHOT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "ready",
        "reason_code",
        "device_id",
        "boot_id",
        "session_id",
        "release_id",
        "profile_sha256",
        "run_binding_sha256",
        "observed_at_ms",
        "max_inference_age_ms",
        "lines",
        "failures",
        "dashboard_artifact_sha256",
        "strict_no_tcp_fallback",
        "network_scope",
        "hardware_touched_by_snapshot",
        "execution_authority",
        "snapshot_sha256",
    }
)
AI_X5_LINE_IDENTITY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "line_id",
        "ready",
        "reason_code",
        "device_id",
        "boot_id",
        "session_id",
        "backend",
        "model_sha256",
        "preprocess_sha256",
        "calibration_sha256",
        "missing_artifacts",
        "last_success_at_ms",
        "success_count",
        "observed_at_ms",
        "runtime",
        "identity_probe",
        "identity_sha256",
    }
)


def canonical_runtime_artifact_set_sha256(snapshot: Mapping[str, Any]) -> str:
    """Hash the complete Dashboard and four-line runtime artifact set."""

    if not isinstance(snapshot, Mapping):
        raise TypeError("AI X5 runtime snapshot must be a mapping")
    dashboard_sha256 = snapshot.get("dashboard_artifact_sha256")
    require_sha256("dashboard_artifact_sha256", dashboard_sha256)
    lines = snapshot.get("lines")
    if not isinstance(lines, Mapping) or set(lines) != set(AI_X5_LINE_IDS):
        raise ValueError("AI X5 runtime artifact set requires exactly the four packaged lines")

    canonical_lines: dict[str, dict[str, dict[str, str]]] = {}
    for line_id in sorted(AI_X5_LINE_IDS):
        line = lines[line_id]
        if not isinstance(line, Mapping):
            raise TypeError(f"AI X5 runtime artifact line {line_id} must be a mapping")
        canonical_groups: dict[str, dict[str, str]] = {}
        for group_name in (
            "model_sha256",
            "preprocess_sha256",
            "calibration_sha256",
        ):
            group = line.get(group_name)
            if not isinstance(group, Mapping) or not group:
                raise ValueError(f"{line_id}.{group_name} must be a non-empty mapping")
            normalized: dict[str, str] = {}
            for logical_name in sorted(group):
                if not isinstance(logical_name, str) or not logical_name:
                    raise ValueError(f"{line_id}.{group_name} contains an invalid logical name")
                digest = group[logical_name]
                require_sha256(f"{line_id}.{group_name}.{logical_name}", digest)
                normalized[logical_name] = digest
            canonical_groups[group_name] = normalized
        canonical_lines[line_id] = canonical_groups

    runtime_artifact_set = {
        "dashboard_artifact_sha256": dashboard_sha256,
        "lines": canonical_lines,
    }
    return canonical_sha256(runtime_artifact_set)


@dataclass(frozen=True, slots=True)
class AiX5CapabilityBinding:
    """Expected identity, release, and per-line backend policy from deployment."""

    device_id: str
    release_id: str
    required_backends: Mapping[str, str]
    maturity: Maturity = Maturity.SHADOW_VALIDATED
    ttl_ms: int = 10_000
    snapshot_max_age_ms: int = 5_000
    inference_max_age_ms: int = 600_000
    run_binding_sha256: str | None = None
    profile_sha256: str | None = None
    expected_runtime_artifact_set_sha256: str | None = None
    expected_boot_id: str | None = None

    def __post_init__(self) -> None:
        if not self.device_id or not self.release_id:
            raise ValueError("AI X5 identity and release bindings must be non-empty")
        if self.maturity is not Maturity.SHADOW_VALIDATED:
            raise ValueError("AI X5 live capability maturity is fixed to SHADOW_VALIDATED")
        for name, value in (
            ("ttl_ms", self.ttl_ms),
            ("snapshot_max_age_ms", self.snapshot_max_age_ms),
            ("inference_max_age_ms", self.inference_max_age_ms),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if dict(self.required_backends) != dict(AI_X5_REQUIRED_BACKENDS):
            raise ValueError("AI X5 backends must match the packaged Bayes-e semantic profile")
        if (self.run_binding_sha256 is None) != (self.profile_sha256 is None):
            raise ValueError("run and profile bindings must be supplied together")
        if self.run_binding_sha256 is not None:
            require_sha256("run_binding_sha256", self.run_binding_sha256)
            require_sha256("profile_sha256", self.profile_sha256)
        if self.expected_runtime_artifact_set_sha256 is not None:
            require_sha256(
                "expected_runtime_artifact_set_sha256",
                self.expected_runtime_artifact_set_sha256,
            )
        if self.expected_boot_id is not None and (
            not isinstance(self.expected_boot_id, str) or _BOOT_ID_RE.fullmatch(self.expected_boot_id) is None
        ):
            raise ValueError("expected_boot_id must be a canonical lowercase UUID")
        object.__setattr__(self, "required_backends", MappingProxyType(dict(self.required_backends)))


class AiX5Adapter(TargetOnlyAdapter):
    subsystem = "ai_x5"
    capability_schema_version = "xrd-rb-voe-ai-capability-v1"


AiX5TargetAdapter = AiX5Adapter


class AiX5ReadOnlyAdapter:
    """Compile one strict Dashboard runtime snapshot into a capability manifest."""

    subsystem = "ai_x5"
    capability_schema_version = "xrd-rb-voe-ai-capability-v1"

    __slots__ = ("_binding", "_transport")

    def __init__(self, transport: JsonSnapshotTransport, binding: AiX5CapabilityBinding) -> None:
        if transport.source_kind is ReadSourceKind.LIVE_REMOTE_READ and (
            binding.run_binding_sha256 is None or binding.profile_sha256 is None
        ):
            raise ValueError("live AI X5 transport requires run-bound semantic profile configuration")
        if (
            transport.source_kind is ReadSourceKind.LIVE_REMOTE_READ
            and binding.expected_runtime_artifact_set_sha256 is None
        ):
            raise ValueError("live AI X5 transport requires expected runtime artifact set binding")
        if transport.source_kind is ReadSourceKind.LIVE_REMOTE_READ and binding.expected_boot_id is None:
            raise ValueError("live AI X5 transport requires expected_boot_id")
        if transport.source_kind is ReadSourceKind.LIVE_REMOTE_READ and not getattr(
            transport, "is_loopback", False
        ):
            raise ValueError("live AI X5 runtime identity must be read from numeric loopback")
        if transport.source_kind is ReadSourceKind.LIVE_REMOTE_READ:
            expected_headers = {
                "X-RB-VoE-Run-Binding": binding.run_binding_sha256,
                "X-RB-VoE-Profile-SHA256": binding.profile_sha256,
            }
            if dict(getattr(transport, "request_headers", {})) != expected_headers:
                raise ValueError("live AI X5 request must carry exact run and profile bindings")
        self._transport = transport
        self._binding = binding

    def read_state(self, *, now_ms: int) -> CapabilityReadResult:
        return self._probe(now_ms=now_ms, operation="READ_STATE")

    def get_capability_manifest(self, *, now_ms: int) -> CapabilityReadResult:
        return self._probe(now_ms=now_ms, operation="CAPABILITY_MANIFEST")

    def _probe(self, *, now_ms: int, operation: str) -> CapabilityReadResult:
        if isinstance(now_ms, bool) or not isinstance(now_ms, int) or now_ms < 0:
            return self._failure(operation, "AI_X5_CLOCK_INVALID")
        try:
            snapshot = self._transport.get_json(AI_X5_RUNTIME_SNAPSHOT_PATH)
            if not isinstance(snapshot, Mapping):
                raise TypeError("runtime snapshot root")
            snapshot_sha256 = canonical_sha256(snapshot)
        except Exception:
            return self._failure(operation, "AI_X5_SNAPSHOT_UNAVAILABLE")

        parsed = self._validate_snapshot(snapshot, now_ms=now_ms)
        if isinstance(parsed, str):
            return self._failure(operation, parsed, snapshot_sha256=snapshot_sha256)
        actual_backends, artifact_sha256, calibration_sha256, runtime_artifact_set_sha256 = parsed
        artifact_sha256["runtime_artifact_set"] = runtime_artifact_set_sha256
        try:
            manifest = CapabilityManifest(
                schema_version=self.capability_schema_version,
                manifest_id=f"ai-x5-{snapshot_sha256[:20]}",
                subsystem=self.subsystem,
                maturity=self._binding.maturity,
                device_id=str(snapshot["device_id"]),
                boot_id=str(snapshot["boot_id"]),
                session_id=str(snapshot["session_id"]),
                release_id=self._binding.release_id,
                capabilities=AI_X5_CAPABILITIES,
                actual_backends=actual_backends,
                artifact_sha256=artifact_sha256,
                calibration_sha256=calibration_sha256,
                stations=("STATION_IDENTITY", "STATION_XRD", "STATION_PL"),
                issued_at_ms=now_ms,
                expires_at_ms=now_ms + self._binding.ttl_ms,
            )
        except (ContractError, TypeError, ValueError):
            return self._failure(operation, "AI_X5_BINDING_INVALID", snapshot_sha256=snapshot_sha256)
        return CapabilityReadResult(
            subsystem=self.subsystem,
            operation=operation,
            maturity=manifest.maturity,
            ready=True,
            reason_code="PASS",
            manifest=manifest,
            snapshot_sha256=snapshot_sha256,
            details={
                "line_count": len(AI_X5_LINE_IDS),
                "strict_no_tcp_fallback": True,
                "read_only": True,
                "runtime_artifact_set_sha256": runtime_artifact_set_sha256,
                "expected_runtime_artifact_set_sha256": (self._binding.expected_runtime_artifact_set_sha256),
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
    ) -> CapabilityReadResult:
        return CapabilityReadResult(
            subsystem=self.subsystem,
            operation=operation,
            maturity=Maturity.TARGET_ONLY,
            ready=False,
            reason_code=reason_code,
            snapshot_sha256=snapshot_sha256,
            details={
                "read_only": True,
                "expected_runtime_artifact_set_sha256": (self._binding.expected_runtime_artifact_set_sha256),
            },
            network_touched=self._transport.network_touched,
            source_kind=self._transport.source_kind,
            run_binding_sha256=self._binding.run_binding_sha256,
            profile_sha256=self._binding.profile_sha256,
        )

    def _validate_snapshot(
        self,
        snapshot: Mapping[str, Any],
        *,
        now_ms: int,
    ) -> tuple[dict[str, str], dict[str, str], dict[str, str], str] | str:
        unsigned = dict(snapshot)
        claimed_snapshot_sha256 = unsigned.pop("snapshot_sha256", None)
        if (
            set(snapshot) != AI_X5_RUNTIME_SNAPSHOT_KEYS
            or snapshot.get("schema_version") != AI_X5_RUNTIME_SNAPSHOT_SCHEMA
            or snapshot.get("ready") is not True
            or snapshot.get("reason_code") != "PASS"
            or snapshot.get("strict_no_tcp_fallback") is not True
            or snapshot.get("network_scope") != "loopback_get_only"
            or snapshot.get("hardware_touched_by_snapshot") is not False
            or snapshot.get("execution_authority") is not False
            or snapshot.get("failures") != {}
            or claimed_snapshot_sha256 != canonical_sha256(unsigned)
        ):
            return "AI_X5_STRICT_SNAPSHOT_REJECTED"
        if snapshot.get("device_id") != self._binding.device_id:
            return "AI_X5_DEVICE_IDENTITY_MISMATCH"
        if snapshot.get("release_id") != self._binding.release_id:
            return "AI_X5_RELEASE_IDENTITY_MISMATCH"
        if self._binding.profile_sha256 is not None and (
            snapshot.get("profile_sha256") != self._binding.profile_sha256
        ):
            return "AI_X5_SEMANTIC_PROFILE_MISMATCH"
        if self._binding.run_binding_sha256 is not None and (
            snapshot.get("run_binding_sha256") != self._binding.run_binding_sha256
        ):
            return "AI_X5_LIVE_RUN_BINDING_MISMATCH"
        try:
            require_sha256("profile_sha256", snapshot.get("profile_sha256"))
            require_sha256("dashboard_artifact_sha256", snapshot.get("dashboard_artifact_sha256"))
        except (TypeError, ValueError):
            return "AI_X5_RUNTIME_BINDING_INCOMPLETE"
        observed_at_ms = snapshot.get("observed_at_ms")
        if (
            isinstance(observed_at_ms, bool)
            or not isinstance(observed_at_ms, int)
            or observed_at_ms > now_ms
            or now_ms - observed_at_ms > self._binding.snapshot_max_age_ms
        ):
            return "AI_X5_SNAPSHOT_STALE"
        boot_id = snapshot.get("boot_id")
        session_id = snapshot.get("session_id")
        if (
            not isinstance(boot_id, str)
            or _BOOT_ID_RE.fullmatch(boot_id) is None
            or not isinstance(session_id, str)
            or not session_id
        ):
            return "AI_X5_RUNTIME_BINDING_INCOMPLETE"
        if self._binding.expected_boot_id is not None and boot_id != self._binding.expected_boot_id:
            return "AI_X5_BOOT_ID_MISMATCH"
        lines = snapshot.get("lines")
        if not isinstance(lines, Mapping) or set(lines) != set(AI_X5_LINE_IDS):
            return "AI_X5_RUNTIME_LINES_INCOMPLETE"
        try:
            runtime_artifact_set_sha256 = canonical_runtime_artifact_set_sha256(snapshot)
        except (TypeError, ValueError):
            return "AI_X5_RUNTIME_ARTIFACT_SET_INVALID"
        if (
            self._binding.expected_runtime_artifact_set_sha256 is not None
            and runtime_artifact_set_sha256 != self._binding.expected_runtime_artifact_set_sha256
        ):
            return "AI_X5_RUNTIME_ARTIFACT_SET_MISMATCH"
        max_inference_age_ms = snapshot.get("max_inference_age_ms")
        if (
            isinstance(max_inference_age_ms, bool)
            or not isinstance(max_inference_age_ms, int)
            or max_inference_age_ms <= 0
            or max_inference_age_ms > self._binding.inference_max_age_ms
        ):
            return "AI_X5_RUNTIME_BINDING_INCOMPLETE"

        actual_backends: dict[str, str] = {}
        artifacts: dict[str, str] = {"dashboard": str(snapshot.get("dashboard_artifact_sha256", ""))}
        calibrations: dict[str, str] = {}
        for line_id in AI_X5_LINE_IDS:
            payload = lines[line_id]
            if not isinstance(payload, Mapping):
                return "AI_X5_RUNTIME_IDENTITY_INVALID"
            unsigned_line = dict(payload)
            claimed_line_sha256 = unsigned_line.pop("identity_sha256", None)
            probe = payload.get("identity_probe")
            runtime = payload.get("runtime")
            backend = payload.get("backend")
            last_success_at_ms = payload.get("last_success_at_ms")
            line_observed_at_ms = payload.get("observed_at_ms")
            if (
                set(payload) != AI_X5_LINE_IDENTITY_KEYS
                or payload.get("schema_version") != RUNTIME_IDENTITY_SCHEMA_VERSION
                or payload.get("line_id") != line_id
                or payload.get("ready") is not True
                or payload.get("reason_code") != "PASS"
                or payload.get("missing_artifacts") != []
                or isinstance(payload.get("success_count"), bool)
                or not isinstance(payload.get("success_count"), int)
                or payload.get("success_count") <= 0
                or payload.get("device_id") != self._binding.device_id
                or payload.get("boot_id") != boot_id
                or not isinstance(payload.get("session_id"), str)
                or not payload.get("session_id")
                or claimed_line_sha256 != canonical_sha256(unsigned_line)
                or not isinstance(probe, Mapping)
                or set(probe)
                != {
                    "method",
                    "model_loaded_by_probe",
                    "inference_triggered_by_probe",
                    "hardware_touched_by_probe",
                    "execution_authority",
                }
                or probe.get("method") != "GET"
                or probe.get("model_loaded_by_probe") is not False
                or probe.get("inference_triggered_by_probe") is not False
                or probe.get("hardware_touched_by_probe") is not False
                or probe.get("execution_authority") is not False
                or not isinstance(runtime, Mapping)
                or set(runtime) != {"python", "implementation", "executable_name"}
                or any(not isinstance(value, str) or not value for value in runtime.values())
                or backend != self._binding.required_backends[line_id]
                or isinstance(last_success_at_ms, bool)
                or not isinstance(last_success_at_ms, int)
                or last_success_at_ms > now_ms
                or now_ms - last_success_at_ms > max_inference_age_ms
                or isinstance(line_observed_at_ms, bool)
                or not isinstance(line_observed_at_ms, int)
                or line_observed_at_ms > now_ms
                or now_ms - line_observed_at_ms > self._binding.snapshot_max_age_ms
            ):
                return "AI_X5_RUNTIME_IDENTITY_INVALID"
            capability = AI_X5_CAPABILITY_BY_LINE[line_id]
            actual_backends[capability] = str(backend)
            for group_name, target in (
                ("model_sha256", artifacts),
                ("preprocess_sha256", artifacts),
                ("calibration_sha256", calibrations),
            ):
                group = payload.get(group_name)
                if not isinstance(group, Mapping) or not group:
                    return "AI_X5_RUNTIME_IDENTITY_INVALID"
                for logical_name, digest in group.items():
                    if not isinstance(logical_name, str) or not isinstance(digest, str):
                        return "AI_X5_RUNTIME_IDENTITY_INVALID"
                    target[f"{line_id}.{group_name}.{logical_name}"] = digest
        return actual_backends, artifacts, calibrations, runtime_artifact_set_sha256


__all__ = [
    "AI_X5_BPU_SLOTS",
    "AI_X5_CAPABILITIES",
    "AI_X5_CAPABILITY_BY_LINE",
    "AI_X5_CPU_MODELS",
    "AI_X5_LINE_IDS",
    "AI_X5_RUNTIME_SNAPSHOT_PATH",
    "AI_X5_REQUIRED_BACKENDS",
    "AI_X5_SNAPSHOT_PATHS",
    "AiX5Adapter",
    "AiX5CapabilityBinding",
    "AiX5ReadOnlyAdapter",
    "AiX5TargetAdapter",
    "canonical_runtime_artifact_set_sha256",
]
