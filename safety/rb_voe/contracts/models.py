"""Immutable v1 contract records for the R0/R1 device-independent core."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from enum import Enum
from types import MappingProxyType
from typing import Any

from rb_voe.contracts.canonical import canonical_sha256, require_sha256, to_primitive
from rb_voe.contracts.registries import (
    RegistryError,
    require_authority_key_pair,
    require_failure_core_reason,
    require_key_domain,
    require_macro_id,
    require_option_id,
    require_physical_evidence_status,
    require_reason_code,
    require_role_bindings,
    require_route_id,
    require_station_id,
    require_zone_id,
)


class ContractError(ValueError):
    """Raised when a contract fails a fail-closed structural check."""


class Maturity(str, Enum):
    TARGET_ONLY = "TARGET_ONLY"
    SIMULATED = "SIMULATED"
    REPLAY_VALIDATED = "REPLAY_VALIDATED"
    SHADOW_VALIDATED = "SHADOW_VALIDATED"
    HARDWARE_PILOT = "HARDWARE_PILOT"
    LOCKED_VALIDATED = "LOCKED_VALIDATED"
    DEMO_ELIGIBLE = "DEMO_ELIGIBLE"


class Decision(str, Enum):
    GO = "GO"
    REVISE = "REVISE"
    DROP = "DROP"
    NEXT_EVIDENCE = "NEXT_EVIDENCE"
    HOLD = "HOLD"
    QUARANTINE = "QUARANTINE"


class EvidenceSource(str, Enum):
    PHYSICAL_ACQUISITION = "PHYSICAL_ACQUISITION"
    DERIVED_COMPUTE = "DERIVED_COMPUTE"
    PROCESS_OBSERVATION = "PROCESS_OBSERVATION"
    OPERATOR_ATTESTATION = "OPERATOR_ATTESTATION"
    SIMULATED_COUNTERFACTUAL = "SIMULATED_COUNTERFACTUAL"


def _require_nonempty(*values: object) -> None:
    if any(not isinstance(value, str) or not value for value in values):
        raise ContractError("identity and binding strings must be non-empty")


def _require_non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{name} must be a non-negative integer")
    return value


def _require_positive_int(name: str, value: object) -> int:
    _require_non_negative_int(name, value)
    if value == 0:
        raise ContractError(f"{name} must be positive")
    return value


def _require_tuple(name: str, value: object, *, nonempty: bool = False) -> tuple[Any, ...]:
    if not isinstance(value, tuple):
        raise ContractError(f"{name} must be an immutable tuple")
    if nonempty and not value:
        raise ContractError(f"{name} cannot be empty")
    if len(value) != len(set(value)):
        raise ContractError(f"{name} values must be unique")
    return value


def _require_hashes(group_name: str, group: Mapping[str, str], *, nonempty: bool) -> None:
    if not isinstance(group, Mapping) or (nonempty and not group):
        raise ContractError(f"{group_name} must be a non-empty mapping")
    for key, digest in group.items():
        if not isinstance(key, str) or not key:
            raise ContractError(f"{group_name} keys must be non-empty strings")
        try:
            require_sha256(f"{group_name}.{key}", digest)
        except ValueError as exc:
            raise ContractError(str(exc)) from exc


def _registry_call(checker: Any, value: object) -> None:
    try:
        checker(value)
    except RegistryError as exc:
        raise ContractError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class ContractRecord:
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return to_primitive({item.name: getattr(self, item.name) for item in fields(self)})

    @property
    def content_sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class EvidenceRecord(ContractRecord):
    evidence_id: str
    kind: str
    source: EvidenceSource
    source_id: str
    lineage_sha256: str
    payload_sha256: str
    observed_at_ms: int
    acquisition_id: str | None = None
    parent_evidence_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != "xrd-rb-voe-evidence-record-v1":
            raise ContractError("unsupported evidence schema")
        if not self.evidence_id or not self.kind or not self.source_id:
            raise ContractError("evidence identity fields must be non-empty")
        if self.observed_at_ms < 0:
            raise ContractError("observed_at_ms must be non-negative")
        try:
            require_sha256("lineage_sha256", self.lineage_sha256)
            require_sha256("payload_sha256", self.payload_sha256)
        except ValueError as exc:
            raise ContractError(str(exc)) from exc
        if self.source is EvidenceSource.PHYSICAL_ACQUISITION and not self.acquisition_id:
            raise ContractError("physical acquisition evidence requires acquisition_id")
        if len(set(self.parent_evidence_ids)) != len(self.parent_evidence_ids):
            raise ContractError("parent evidence ids must be unique")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class ExperimentCase(ContractRecord):
    case_id: str
    sample_id: str
    lineage_sha256: str
    evidence_root_sha256: str
    allowed_options: tuple[str, ...]
    required_failure_atoms: tuple[str, ...]
    closure_predicate_id: str
    release_id: str
    created_at_ms: int

    def __post_init__(self) -> None:
        if self.schema_version != "xrd-rb-voe-experiment-case-v1":
            raise ContractError("unsupported experiment case schema")
        _require_nonempty(
            self.case_id,
            self.sample_id,
            self.closure_predicate_id,
            self.release_id,
        )
        try:
            require_sha256("lineage_sha256", self.lineage_sha256)
            require_sha256("evidence_root_sha256", self.evidence_root_sha256)
        except ValueError as exc:
            raise ContractError(str(exc)) from exc
        _require_tuple("allowed_options", self.allowed_options, nonempty=True)
        _require_tuple("required_failure_atoms", self.required_failure_atoms, nonempty=True)
        for option_id in self.allowed_options:
            _registry_call(require_option_id, option_id)
        for reason_code in self.required_failure_atoms:
            _registry_call(require_failure_core_reason, reason_code)
        _require_non_negative_int("created_at_ms", self.created_at_ms)


@dataclass(frozen=True, slots=True)
class EvidenceIntent(ContractRecord):
    intent_id: str
    case_id: str
    plan_epoch: int
    failure_core: tuple[str, ...]
    candidate_options: tuple[str, ...]
    selected_option: str | None
    value_lower_bound: float | None
    required_capabilities: tuple[str, ...]
    decision: Decision
    created_at_ms: int

    def __post_init__(self) -> None:
        if self.schema_version != "xrd-rb-voe-evidence-intent-v1":
            raise ContractError("unsupported evidence intent schema")
        _require_nonempty(self.intent_id, self.case_id)
        _require_non_negative_int("plan_epoch", self.plan_epoch)
        _require_non_negative_int("created_at_ms", self.created_at_ms)
        _require_tuple("failure_core", self.failure_core, nonempty=True)
        _require_tuple("candidate_options", self.candidate_options, nonempty=True)
        _require_tuple("required_capabilities", self.required_capabilities)
        for reason_code in self.failure_core:
            _registry_call(require_failure_core_reason, reason_code)
        for option_id in self.candidate_options:
            _registry_call(require_option_id, option_id)
        if not isinstance(self.decision, Decision):
            raise ContractError("decision must be a Decision enum")
        if self.selected_option is not None:
            _registry_call(require_option_id, self.selected_option)
        if self.selected_option is not None and self.selected_option not in self.candidate_options:
            raise ContractError("selected option must appear in candidate_options")
        if self.decision is Decision.NEXT_EVIDENCE and self.selected_option is None:
            raise ContractError("NEXT_EVIDENCE requires a selected option")
        if self.decision in {Decision.HOLD, Decision.QUARANTINE} and self.selected_option is not None:
            raise ContractError("terminal refusal cannot carry a selected option")
        if self.value_lower_bound is not None:
            if (
                isinstance(self.value_lower_bound, bool)
                or not isinstance(self.value_lower_bound, (int, float))
                or not math.isfinite(self.value_lower_bound)
            ):
                raise ContractError("value_lower_bound must be finite")
        if self.selected_option is not None and (
            self.value_lower_bound is None or self.value_lower_bound <= 0
        ):
            raise ContractError("selected option requires a positive value lower bound")
        if self.selected_option is None and self.value_lower_bound is not None:
            raise ContractError("value lower bound cannot exist without a selected option")
        for capability in self.required_capabilities:
            _require_nonempty(capability)


@dataclass(frozen=True, slots=True)
class CapabilityManifest(ContractRecord):
    manifest_id: str
    subsystem: str
    maturity: Maturity
    device_id: str
    boot_id: str
    session_id: str
    release_id: str
    capabilities: tuple[str, ...]
    actual_backends: Mapping[str, str]
    artifact_sha256: Mapping[str, str]
    calibration_sha256: Mapping[str, str]
    stations: tuple[str, ...]
    issued_at_ms: int
    expires_at_ms: int

    def __post_init__(self) -> None:
        subsystem_by_schema = {
            "xrd-rb-voe-ai-capability-v1": "ai_x5",
            "xrd-rb-voe-embodied-capability-v1": "embodied_x5",
            "xrd-rb-voe-dual-arm-capability-v1": "dual_arm",
            "xrd-rb-voe-assay-station-capability-v1": "assay_station",
        }
        if self.schema_version not in subsystem_by_schema:
            raise ContractError("unsupported capability schema")
        if self.subsystem != subsystem_by_schema[self.schema_version]:
            raise ContractError("capability subsystem contradicts schema_version")
        if not isinstance(self.maturity, Maturity):
            raise ContractError("maturity must be a Maturity enum")
        _require_nonempty(
            self.manifest_id,
            self.device_id,
            self.boot_id,
            self.session_id,
            self.release_id,
        )
        _require_non_negative_int("issued_at_ms", self.issued_at_ms)
        _require_non_negative_int("expires_at_ms", self.expires_at_ms)
        if self.expires_at_ms <= self.issued_at_ms:
            raise ContractError("capability expiry must follow issue time")
        _require_tuple("capabilities", self.capabilities)
        _require_tuple("stations", self.stations)
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ContractError("capabilities must be unique")
        for capability in self.capabilities:
            _require_nonempty(capability)
        for station_id in self.stations:
            _registry_call(require_station_id, station_id)
        if not isinstance(self.actual_backends, Mapping):
            raise ContractError("actual_backends must be a mapping")
        for key, backend in self.actual_backends.items():
            _require_nonempty(key, backend)
        if self.maturity is not Maturity.TARGET_ONLY and set(self.actual_backends) != set(self.capabilities):
            raise ContractError("actual_backends must bind exactly one backend per capability")
        _require_hashes("artifact_sha256", self.artifact_sha256, nonempty=True)
        _require_hashes(
            "calibration_sha256",
            self.calibration_sha256,
            nonempty=self.maturity is not Maturity.TARGET_ONLY,
        )
        if self.maturity is not Maturity.TARGET_ONLY and (
            not self.capabilities or not self.actual_backends or not self.stations
        ):
            raise ContractError("ready capability manifest is incomplete")
        object.__setattr__(self, "actual_backends", MappingProxyType(dict(self.actual_backends)))
        object.__setattr__(self, "artifact_sha256", MappingProxyType(dict(self.artifact_sha256)))
        object.__setattr__(
            self,
            "calibration_sha256",
            MappingProxyType(dict(self.calibration_sha256)),
        )

    def is_fresh(self, now_ms: int) -> bool:
        return self.issued_at_ms <= now_ms < self.expires_at_ms


@dataclass(frozen=True, slots=True)
class ExecutionChallenge(ContractRecord):
    challenge_id: str
    intent_id: str
    case_id: str
    plan_epoch: int
    nonce: str
    embodied_manifest_sha256: str
    source_boot_id: str
    source_session_id: str
    route_plan_sha256: str
    reserved_routes: tuple[str, ...]
    reserved_stations: tuple[str, ...]
    reserved_zones: tuple[str, ...]
    live_cost_ms: int
    issued_at_ms: int
    expires_at_ms: int
    key_domain: str
    signature: str

    def __post_init__(self) -> None:
        if self.schema_version != "xrd-rb-voe-execution-challenge-v1":
            raise ContractError("unsupported execution challenge schema")
        _require_nonempty(
            self.challenge_id,
            self.intent_id,
            self.case_id,
            self.nonce,
            self.source_boot_id,
            self.source_session_id,
            self.signature,
        )
        try:
            require_sha256("embodied_manifest_sha256", self.embodied_manifest_sha256)
            require_sha256("route_plan_sha256", self.route_plan_sha256)
        except ValueError as exc:
            raise ContractError(str(exc)) from exc
        _require_non_negative_int("plan_epoch", self.plan_epoch)
        _require_non_negative_int("live_cost_ms", self.live_cost_ms)
        _require_non_negative_int("issued_at_ms", self.issued_at_ms)
        _require_non_negative_int("expires_at_ms", self.expires_at_ms)
        if self.expires_at_ms <= self.issued_at_ms:
            raise ContractError("invalid challenge timing")
        _require_tuple("reserved_routes", self.reserved_routes, nonempty=True)
        _require_tuple("reserved_stations", self.reserved_stations, nonempty=True)
        _require_tuple("reserved_zones", self.reserved_zones, nonempty=True)
        for route_id in self.reserved_routes:
            _registry_call(require_route_id, route_id)
        for station_id in self.reserved_stations:
            _registry_call(require_station_id, station_id)
        for zone_id in self.reserved_zones:
            _registry_call(require_zone_id, zone_id)
        _registry_call(require_key_domain, self.key_domain)
        if self.key_domain != "RB_VOE_EMBODIED_CHALLENGE_ED25519_V1":
            raise ContractError("execution challenge uses the wrong key domain")


@dataclass(frozen=True, slots=True)
class JointPermit(ContractRecord):
    permit_id: str
    attempt_id: str
    case_id: str
    intent_id: str
    plan_epoch: int
    option_id: str
    challenge_id: str
    challenge_nonce: str
    sample_lineage_sha256: str
    macro_id: str
    macro_contract_sha256: str
    roles: Mapping[str, str]
    zones: tuple[str, ...]
    command_envelope_sha256: str
    required_capability_hashes: tuple[str, ...]
    required_local_gates: tuple[str, ...]
    fallback: str
    operator_armed: bool
    release_manifest_sha256: str
    key_domain: str
    authority_domain: str
    issued_at_ms: int
    start_expires_at_ms: int
    execution_timeout_ms: int
    internal_micro_retry_budget: int
    signature: str

    def __post_init__(self) -> None:
        if self.schema_version != "xrd-rb-voe-joint-permit-v1":
            raise ContractError("unsupported joint permit schema")
        _require_nonempty(
            self.permit_id,
            self.attempt_id,
            self.case_id,
            self.intent_id,
            self.challenge_id,
            self.challenge_nonce,
            self.signature,
        )
        for name, digest in (
            ("sample_lineage_sha256", self.sample_lineage_sha256),
            ("macro_contract_sha256", self.macro_contract_sha256),
            ("command_envelope_sha256", self.command_envelope_sha256),
            ("release_manifest_sha256", self.release_manifest_sha256),
        ):
            try:
                require_sha256(name, digest)
            except ValueError as exc:
                raise ContractError(str(exc)) from exc
        _require_non_negative_int("plan_epoch", self.plan_epoch)
        _registry_call(require_option_id, self.option_id)
        _registry_call(require_macro_id, self.macro_id)
        _registry_call(require_role_bindings, self.roles)
        _require_tuple("zones", self.zones, nonempty=True)
        for zone_id in self.zones:
            _registry_call(require_zone_id, zone_id)
        _require_tuple("required_capability_hashes", self.required_capability_hashes, nonempty=True)
        for index, digest in enumerate(self.required_capability_hashes):
            try:
                require_sha256(f"required_capability_hashes[{index}]", digest)
            except ValueError as exc:
                raise ContractError(str(exc)) from exc
        _require_tuple("required_local_gates", self.required_local_gates, nonempty=True)
        for gate in self.required_local_gates:
            _require_nonempty(gate)
        _registry_call(require_reason_code, self.fallback)
        if self.fallback != "SAFE_ABORT_AND_HOLD":
            raise ContractError("permit fallback must be SAFE_ABORT_AND_HOLD")
        if not isinstance(self.operator_armed, bool):
            raise ContractError("operator_armed must be a boolean")
        try:
            require_authority_key_pair(self.authority_domain, self.key_domain)
        except RegistryError as exc:
            raise ContractError(str(exc)) from exc
        if self.authority_domain == "SUPERVISED_TRIAL_AUTH" and not self.operator_armed:
            raise ContractError("supervised trial permit requires explicit operator arming")
        _require_non_negative_int("issued_at_ms", self.issued_at_ms)
        _require_non_negative_int("start_expires_at_ms", self.start_expires_at_ms)
        _require_positive_int("execution_timeout_ms", self.execution_timeout_ms)
        _require_non_negative_int("internal_micro_retry_budget", self.internal_micro_retry_budget)
        if self.start_expires_at_ms <= self.issued_at_ms:
            raise ContractError("invalid permit timing")
        object.__setattr__(self, "roles", MappingProxyType(dict(self.roles)))


@dataclass(frozen=True, slots=True)
class PhysicalEvidenceCapsule(ContractRecord):
    capsule_id: str
    case_id: str
    attempt_id: str
    permit_id: str
    permit_sha256: str
    challenge_id: str
    challenge_nonce: str
    sample_lineage_sha256: str
    macro_id: str
    acquisition_id: str
    status: str
    source_id: str
    source_boot_id: str
    source_session_id: str
    source_manifest_sha256: str
    actual_backend: str
    artifact_sha256: Mapping[str, str]
    calibration_sha256: Mapping[str, str]
    started_at_ms: int
    ended_at_ms: int
    payload_sha256: str
    external_truth: bool
    hardware_touched: bool
    failure_code: str | None
    signature: str

    def __post_init__(self) -> None:
        if self.schema_version != "xrd-rb-voe-physical-evidence-v1":
            raise ContractError("unsupported physical evidence schema")
        _require_nonempty(
            self.capsule_id,
            self.case_id,
            self.attempt_id,
            self.permit_id,
            self.challenge_id,
            self.challenge_nonce,
            self.acquisition_id,
            self.source_id,
            self.source_boot_id,
            self.source_session_id,
            self.actual_backend,
            self.signature,
        )
        for name, digest in (
            ("permit_sha256", self.permit_sha256),
            ("sample_lineage_sha256", self.sample_lineage_sha256),
            ("source_manifest_sha256", self.source_manifest_sha256),
            ("payload_sha256", self.payload_sha256),
        ):
            try:
                require_sha256(name, digest)
            except ValueError as exc:
                raise ContractError(str(exc)) from exc
        _registry_call(require_macro_id, self.macro_id)
        _registry_call(require_physical_evidence_status, self.status)
        _require_hashes("artifact_sha256", self.artifact_sha256, nonempty=True)
        _require_hashes("calibration_sha256", self.calibration_sha256, nonempty=True)
        _require_non_negative_int("started_at_ms", self.started_at_ms)
        _require_non_negative_int("ended_at_ms", self.ended_at_ms)
        if self.ended_at_ms < self.started_at_ms:
            raise ContractError("capsule end time cannot precede start")
        if not isinstance(self.external_truth, bool) or not isinstance(self.hardware_touched, bool):
            raise ContractError("capsule truth and hardware flags must be booleans")
        if self.status == "SUCCEEDED" and self.failure_code is not None:
            raise ContractError("successful capsule cannot carry a failure code")
        if self.status != "SUCCEEDED" and self.failure_code is None:
            raise ContractError("non-success capsule requires a failure code")
        if self.failure_code is not None:
            _registry_call(require_reason_code, self.failure_code)
        if self.external_truth and (self.status != "SUCCEEDED" or not self.hardware_touched):
            raise ContractError("external truth requires successful hardware acquisition")
        object.__setattr__(self, "artifact_sha256", MappingProxyType(dict(self.artifact_sha256)))
        object.__setattr__(
            self,
            "calibration_sha256",
            MappingProxyType(dict(self.calibration_sha256)),
        )


@dataclass(frozen=True, slots=True)
class DecisionReceipt(ContractRecord):
    receipt_id: str
    case_id: str
    plan_epoch: int
    decision: Decision
    failure_core_closed: bool
    selected_option: str | None
    terminal_evidence_sha256: tuple[str, ...]
    release_certificate_sha256: str | None
    previous_record_sha256: str
    created_at_ms: int

    def __post_init__(self) -> None:
        if self.schema_version != "xrd-rb-voe-decision-receipt-v1":
            raise ContractError("unsupported decision receipt schema")
        _require_nonempty(self.receipt_id, self.case_id)
        if not isinstance(self.decision, Decision):
            raise ContractError("decision must be a Decision enum")
        try:
            require_sha256("previous_record_sha256", self.previous_record_sha256)
            if self.release_certificate_sha256 is not None:
                require_sha256("release_certificate_sha256", self.release_certificate_sha256)
            for index, digest in enumerate(self.terminal_evidence_sha256):
                require_sha256(f"terminal_evidence_sha256[{index}]", digest)
        except ValueError as exc:
            raise ContractError(str(exc)) from exc
        _require_non_negative_int("created_at_ms", self.created_at_ms)
        _require_non_negative_int("plan_epoch", self.plan_epoch)
        _require_tuple("terminal_evidence_sha256", self.terminal_evidence_sha256)
        if self.selected_option is not None:
            _registry_call(require_option_id, self.selected_option)
        if self.decision is Decision.NEXT_EVIDENCE and self.selected_option is None:
            raise ContractError("NEXT_EVIDENCE receipt requires a selected option")
        if self.decision in {Decision.HOLD, Decision.QUARANTINE} and self.selected_option is not None:
            raise ContractError("terminal refusal receipt cannot select an option")
        if self.decision in {Decision.GO, Decision.REVISE, Decision.DROP} and not self.failure_core_closed:
            raise ContractError("material terminal decisions require a closed failure core")
        if (
            self.decision in {Decision.GO, Decision.REVISE, Decision.DROP}
            and not self.terminal_evidence_sha256
        ):
            raise ContractError("material terminal decisions require terminal evidence")
