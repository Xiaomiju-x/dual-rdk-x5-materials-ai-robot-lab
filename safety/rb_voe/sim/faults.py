"""Preregistered, deterministic admission-fault injections for R1 simulation."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any

from rb_voe.contracts.canonical import canonical_sha256, is_sha256
from rb_voe.contracts.models import ContractError, EvidenceRecord, EvidenceSource
from rb_voe.core.evidence_dag import EvidenceDAG, EvidenceGraphError
from rb_voe.core.invariants import evaluate_evidence_invariants
from rb_voe.sim.simulator import EpisodeResult


class FaultKind(str, Enum):
    SHARED_SENSOR_CORRELATION = "SHARED_SENSOR_CORRELATION"
    STALE_CAPABILITY = "STALE_CAPABILITY"
    PROVENANCE_DRIFT = "PROVENANCE_DRIFT"
    HASH_TAMPER = "HASH_TAMPER"
    REPLAYED_NONCE = "REPLAYED_NONCE"
    SAMPLE_LINEAGE_SWAP = "SAMPLE_LINEAGE_SWAP"
    LOCAL_VETO = "LOCAL_VETO"
    MISSING_STATION_CAPABILITY = "MISSING_STATION_CAPABILITY"
    MISSING_LOADER_CAPABILITY = "MISSING_LOADER_CAPABILITY"
    NETWORK_DROP = "NETWORK_DROP"
    HEARTBEAT_TIMEOUT = "HEARTBEAT_TIMEOUT"
    BOOT_SESSION_CHANGE = "BOOT_SESSION_CHANGE"
    BPU_PROVENANCE_MISMATCH = "BPU_PROVENANCE_MISMATCH"
    ACTION_TIMEOUT = "ACTION_TIMEOUT"
    POWER_DROP = "POWER_DROP"


EXPECTED_REFUSAL_CODES: Mapping[FaultKind, str] = {
    FaultKind.SHARED_SENSOR_CORRELATION: "CORRELATED_EVIDENCE_DOUBLE_COUNTED",
    FaultKind.STALE_CAPABILITY: "CAPABILITY_STALE",
    FaultKind.PROVENANCE_DRIFT: "PROVENANCE_DRIFT",
    FaultKind.HASH_TAMPER: "CONTENT_HASH_MISMATCH",
    FaultKind.REPLAYED_NONCE: "NONCE_REPLAY",
    FaultKind.SAMPLE_LINEAGE_SWAP: "SAMPLE_LINEAGE_MISMATCH",
    FaultKind.LOCAL_VETO: "LOCAL_SAFETY_VETO",
    FaultKind.MISSING_STATION_CAPABILITY: "REQUIRED_STATION_UNAVAILABLE",
    FaultKind.MISSING_LOADER_CAPABILITY: "REQUIRED_LOADER_UNAVAILABLE",
    FaultKind.NETWORK_DROP: "NETWORK_UNAVAILABLE",
    FaultKind.HEARTBEAT_TIMEOUT: "HEARTBEAT_TIMEOUT",
    FaultKind.BOOT_SESSION_CHANGE: "BOOT_SESSION_CHANGED",
    FaultKind.BPU_PROVENANCE_MISMATCH: "BPU_PROVENANCE_MISMATCH",
    FaultKind.ACTION_TIMEOUT: "EXECUTION_TIMEOUT",
    FaultKind.POWER_DROP: "SAFE_ABORT_AND_HOLD",
}

WAVE2_FAULT_KINDS: tuple[FaultKind, ...] = (
    FaultKind.NETWORK_DROP,
    FaultKind.HEARTBEAT_TIMEOUT,
    FaultKind.BOOT_SESSION_CHANGE,
    FaultKind.BPU_PROVENANCE_MISMATCH,
    FaultKind.ACTION_TIMEOUT,
    FaultKind.POWER_DROP,
)

ADMISSION_SCHEMA_VERSION = "xrd-rb-voe-sim-admission-v2"
EVIDENCE_BUNDLE_SCHEMA_VERSION = "xrd-rb-voe-sim-evidence-dag-v1"
_ACQUISITION_ROOT_KIND = "SIMULATED_ACQUISITION_ROOT"
_DERIVED_OBSERVATION_KIND = "SIMULATED_DERIVED_OBSERVATION"
_EVIDENCE_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "evidence_id",
        "kind",
        "source",
        "source_id",
        "lineage_sha256",
        "payload_sha256",
        "observed_at_ms",
        "acquisition_id",
        "parent_evidence_ids",
        "metadata",
    }
)


@dataclass(frozen=True, slots=True)
class AdmissionReport:
    refusal_codes: tuple[str, ...]
    hardware_touch_authorized: bool = False
    execution_authorized: bool = False

    def __post_init__(self) -> None:
        if self.hardware_touch_authorized or self.execution_authorized:
            raise ValueError("simulation admission cannot grant hardware or execution authority")

    @property
    def admitted(self) -> bool:
        return not self.refusal_codes

    def has_refusal(self, code: str) -> bool:
        return code in self.refusal_codes


@dataclass(frozen=True, slots=True)
class FaultInjection:
    fault_kind: FaultKind
    before_sha256: str
    after_sha256: str
    expected_refusal_code: str
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        if self.before_sha256 == self.after_sha256:
            raise ValueError("fault injection must change the payload digest")


def build_admission_envelope(
    episode: EpisodeResult,
    *,
    root_provenance_sha256: str,
    sample_lineage_sha256: str,
) -> dict[str, Any]:
    """Build a valid simulated admission envelope with no execution authority."""
    if not is_sha256(root_provenance_sha256) or not is_sha256(sample_lineage_sha256):
        raise ValueError("provenance and lineage values must be SHA-256 digests")
    episode_payload = episode.to_dict()
    bpu_model_sha256 = canonical_sha256({"model": "sealed-bpu-fixture-v1"})
    bpu_runtime_sha256 = canonical_sha256({"runtime": "hobot-dnn-fixture-v1"})
    evidence_records = (
        EvidenceRecord(
            schema_version="xrd-rb-voe-evidence-record-v1",
            evidence_id="xrd-acquisition-root",
            kind=_ACQUISITION_ROOT_KIND,
            source=EvidenceSource.SIMULATED_COUNTERFACTUAL,
            source_id="xrd-sensor-a",
            lineage_sha256=sample_lineage_sha256,
            payload_sha256=canonical_sha256({"fixture": "xrd-acquisition-root-v1"}),
            observed_at_ms=4_800,
            acquisition_id="xrd-acquisition-root",
            metadata={
                "failure_domains": [
                    "calibration:xrd-calibration-v1",
                    "holder:xrd-holder-a",
                ],
            },
        ),
        EvidenceRecord(
            schema_version="xrd-rb-voe-evidence-record-v1",
            evidence_id="xrd-derived-observation",
            kind=_DERIVED_OBSERVATION_KIND,
            source=EvidenceSource.DERIVED_COMPUTE,
            source_id="xrd-analysis-pipeline-a",
            lineage_sha256=sample_lineage_sha256,
            payload_sha256=canonical_sha256({"fixture": "xrd-derived-observation-v1"}),
            observed_at_ms=4_900,
            parent_evidence_ids=("xrd-acquisition-root",),
        ),
        EvidenceRecord(
            schema_version="xrd-rb-voe-evidence-record-v1",
            evidence_id="pl-acquisition-root",
            kind=_ACQUISITION_ROOT_KIND,
            source=EvidenceSource.SIMULATED_COUNTERFACTUAL,
            source_id="pl-sensor-b",
            lineage_sha256=sample_lineage_sha256,
            payload_sha256=canonical_sha256({"fixture": "pl-acquisition-root-v1"}),
            observed_at_ms=4_800,
            acquisition_id="pl-acquisition-root",
            metadata={
                "failure_domains": [
                    "calibration:pl-calibration-v1",
                    "holder:pl-holder-b",
                ],
            },
        ),
        EvidenceRecord(
            schema_version="xrd-rb-voe-evidence-record-v1",
            evidence_id="pl-derived-observation",
            kind=_DERIVED_OBSERVATION_KIND,
            source=EvidenceSource.DERIVED_COMPUTE,
            source_id="pl-analysis-pipeline-b",
            lineage_sha256=sample_lineage_sha256,
            payload_sha256=canonical_sha256({"fixture": "pl-derived-observation-v1"}),
            observed_at_ms=4_900,
            parent_evidence_ids=("pl-acquisition-root",),
        ),
    )
    evidence_dag = EvidenceDAG(evidence_records)
    return {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "episode": episode_payload,
        "integrity": {"declared_episode_sha256": episode.episode_sha256},
        "provenance": {
            "expected_sha256": root_provenance_sha256,
            "actual_sha256": root_provenance_sha256,
        },
        "capability": {
            "issued_at_ms": 1_000,
            "expires_at_ms": 10_000,
            "evaluated_at_ms": 5_000,
            "required_stations": ["xrd-station"],
            "available_stations": ["xrd-station"],
            "required_loaders": ["bpu-loader"],
            "available_loaders": ["bpu-loader"],
        },
        "evidence": {
            "schema_version": EVIDENCE_BUNDLE_SCHEMA_VERSION,
            "expected_dag_sha256": evidence_dag.content_sha256,
            "selected_evidence_ids": [
                "xrd-derived-observation",
                "pl-derived-observation",
            ],
            "records": [record.to_dict() for record in evidence_records],
        },
        "security": {"nonce": "sim-nonce-001", "consumed_nonces": []},
        "sample": {
            "expected_lineage_sha256": sample_lineage_sha256,
            "actual_lineage_sha256": sample_lineage_sha256,
        },
        "safety": {"local_veto": False},
        "transport": {"network_required": True, "network_available": True},
        "heartbeat": {
            "observed_at_ms": 4_900,
            "evaluated_at_ms": 5_000,
            "timeout_ms": 500,
        },
        "runtime": {
            "expected_boot_session_id": "boot-session-001",
            "actual_boot_session_id": "boot-session-001",
        },
        "bpu_provenance": {
            "expected_model_sha256": bpu_model_sha256,
            "actual_model_sha256": bpu_model_sha256,
            "expected_runtime_sha256": bpu_runtime_sha256,
            "actual_runtime_sha256": bpu_runtime_sha256,
        },
        "action": {
            "started_at_ms": 5_000,
            "deadline_at_ms": 6_000,
            "observed_at_ms": 5_900,
            "completed": True,
        },
        "power": {"power_good": True, "brownout_detected": False},
        "authority": {
            "evidence_source": EvidenceSource.SIMULATED_COUNTERFACTUAL.value,
            "hardware_touch": False,
            "execution_authority": False,
            "physical_risk_denominator_increment": 0,
        },
    }


def _nested_mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name)
    return value if isinstance(value, Mapping) else {}


def _strict_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _string_set(value: object) -> set[str] | None:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        return None
    return set(value)


def _nonempty_string(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("expected a non-empty string")
    return value


def _record_from_mapping(raw: object) -> EvidenceRecord:
    if not isinstance(raw, Mapping) or set(raw) != _EVIDENCE_RECORD_FIELDS:
        raise ValueError("evidence record fields do not match the frozen contract")
    parent_ids = raw["parent_evidence_ids"]
    if not isinstance(parent_ids, list) or any(
        not isinstance(parent_id, str) or not parent_id for parent_id in parent_ids
    ):
        raise ValueError("parent_evidence_ids must be a list of non-empty strings")
    metadata = raw["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("evidence metadata must be a mapping")
    observed_at_ms = _strict_int(raw["observed_at_ms"])
    if observed_at_ms is None:
        raise ValueError("observed_at_ms must be an integer")
    acquisition_id = raw["acquisition_id"]
    if acquisition_id is not None:
        acquisition_id = _nonempty_string(acquisition_id)
    try:
        source = EvidenceSource(_nonempty_string(raw["source"]))
    except ValueError as exc:
        raise ValueError("evidence source is not registered") from exc
    return EvidenceRecord(
        schema_version=_nonempty_string(raw["schema_version"]),
        evidence_id=_nonempty_string(raw["evidence_id"]),
        kind=_nonempty_string(raw["kind"]),
        source=source,
        source_id=_nonempty_string(raw["source_id"]),
        lineage_sha256=_nonempty_string(raw["lineage_sha256"]),
        payload_sha256=_nonempty_string(raw["payload_sha256"]),
        observed_at_ms=observed_at_ms,
        acquisition_id=acquisition_id,
        parent_evidence_ids=tuple(parent_ids),
        metadata=dict(metadata),
    )


def _reconstruct_evidence_dag(
    bundle: object,
) -> tuple[EvidenceDAG, tuple[str, ...], str]:
    if not isinstance(bundle, Mapping):
        raise ValueError("evidence bundle must be a mapping")
    if set(bundle) != {
        "schema_version",
        "expected_dag_sha256",
        "selected_evidence_ids",
        "records",
    }:
        raise ValueError("evidence bundle fields do not match the frozen contract")
    if bundle["schema_version"] != EVIDENCE_BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported evidence bundle schema")
    expected_sha256 = bundle["expected_dag_sha256"]
    if not is_sha256(expected_sha256):
        raise ValueError("expected evidence DAG digest is invalid")
    selected_raw = bundle["selected_evidence_ids"]
    if not isinstance(selected_raw, list) or any(
        not isinstance(evidence_id, str) or not evidence_id for evidence_id in selected_raw
    ):
        raise ValueError("selected evidence ids must be non-empty strings")
    selected = tuple(selected_raw)
    if len(selected) != len(set(selected)):
        raise ValueError("selected evidence ids must be unique")
    records_raw = bundle["records"]
    if not isinstance(records_raw, list) or not records_raw:
        raise ValueError("evidence records must be a non-empty list")
    records = tuple(_record_from_mapping(record) for record in records_raw)
    dag = EvidenceDAG(records)
    if any(evidence_id not in dag for evidence_id in selected):
        raise ValueError("selected evidence id is absent from the DAG")

    roots = {record.evidence_id for record in records if record.kind == _ACQUISITION_ROOT_KIND}
    if len(roots) < 2:
        raise ValueError("at least two acquisition roots are required")
    for root_id in roots:
        root = dag.record(root_id)
        domains = root.metadata.get("failure_domains")
        if (
            root.source is not EvidenceSource.SIMULATED_COUNTERFACTUAL
            or root.parent_evidence_ids
            or root.acquisition_id != root.evidence_id
            or not isinstance(domains, list)
            or len(domains) < 2
            or any(not isinstance(domain, str) or not domain for domain in domains)
            or not any(domain.startswith("calibration:") for domain in domains)
            or not any(domain.startswith("holder:") for domain in domains)
        ):
            raise ValueError("acquisition roots require complete calibration and holder domains")
    for evidence_id in selected:
        record = dag.record(evidence_id)
        inherited_roots = roots.intersection(ancestor.evidence_id for ancestor in dag.ancestors(evidence_id))
        if (
            record.kind != _DERIVED_OBSERVATION_KIND
            or record.source is not EvidenceSource.DERIVED_COMPUTE
            or len(inherited_roots) != 1
        ):
            raise ValueError("selected evidence must derive from exactly one acquisition root")
    return dag, selected, expected_sha256


def _refresh_evidence_dag_digest(bundle: dict[str, Any]) -> None:
    records = tuple(_record_from_mapping(record) for record in bundle["records"])
    bundle["expected_dag_sha256"] = EvidenceDAG(records).content_sha256


def _validate_evidence_bundle(bundle: object) -> tuple[str, ...]:
    try:
        dag, selected, expected_sha256 = _reconstruct_evidence_dag(bundle)
    except (ContractError, EvidenceGraphError, KeyError, TypeError, ValueError):
        return ("EVIDENCE_DAG_INVALID",)
    report = evaluate_evidence_invariants(
        dag,
        evidence_ids=selected,
        minimum_independent=2,
        expected_sha256=expected_sha256,
    )
    return report.failure_codes


def validate_admission_envelope(payload: Mapping[str, Any]) -> AdmissionReport:
    """Evaluate non-compensable simulation admission checks, fail closed."""
    refusal_codes: set[str] = set()
    if payload.get("schema_version") != ADMISSION_SCHEMA_VERSION:
        refusal_codes.add("ADMISSION_ENVELOPE_INVALID")

    episode = _nested_mapping(payload, "episode")
    integrity = _nested_mapping(payload, "integrity")
    declared = integrity.get("declared_episode_sha256")
    if not is_sha256(declared) or canonical_sha256(episode) != declared:
        refusal_codes.add("CONTENT_HASH_MISMATCH")

    provenance = _nested_mapping(payload, "provenance")
    expected_provenance = provenance.get("expected_sha256")
    actual_provenance = provenance.get("actual_sha256")
    if (
        not is_sha256(expected_provenance)
        or not is_sha256(actual_provenance)
        or expected_provenance != actual_provenance
    ):
        refusal_codes.add("PROVENANCE_DRIFT")

    capability = _nested_mapping(payload, "capability")
    issued_at = capability.get("issued_at_ms")
    expires_at = capability.get("expires_at_ms")
    evaluated_at = capability.get("evaluated_at_ms")
    issued_at_int = _strict_int(issued_at)
    expires_at_int = _strict_int(expires_at)
    evaluated_at_int = _strict_int(evaluated_at)
    if (
        issued_at_int is None
        or expires_at_int is None
        or evaluated_at_int is None
        or not (issued_at_int <= evaluated_at_int < expires_at_int)
    ):
        refusal_codes.add("CAPABILITY_STALE")
    required_stations = _string_set(capability.get("required_stations"))
    available_stations = _string_set(capability.get("available_stations"))
    if (
        required_stations is None
        or available_stations is None
        or not required_stations.issubset(available_stations)
    ):
        refusal_codes.add("REQUIRED_STATION_UNAVAILABLE")
    required_loaders = _string_set(capability.get("required_loaders"))
    available_loaders = _string_set(capability.get("available_loaders"))
    if (
        required_loaders is None
        or available_loaders is None
        or not required_loaders.issubset(available_loaders)
    ):
        refusal_codes.add("REQUIRED_LOADER_UNAVAILABLE")

    refusal_codes.update(_validate_evidence_bundle(payload.get("evidence")))

    security = _nested_mapping(payload, "security")
    nonce = security.get("nonce")
    consumed_nonces = security.get("consumed_nonces")
    if not isinstance(nonce, str) or not nonce:
        refusal_codes.add("NONCE_INVALID")
    elif not isinstance(consumed_nonces, list) or nonce in consumed_nonces:
        refusal_codes.add("NONCE_REPLAY")

    sample = _nested_mapping(payload, "sample")
    expected_lineage = sample.get("expected_lineage_sha256")
    actual_lineage = sample.get("actual_lineage_sha256")
    if not is_sha256(expected_lineage) or not is_sha256(actual_lineage) or expected_lineage != actual_lineage:
        refusal_codes.add("SAMPLE_LINEAGE_MISMATCH")

    safety = _nested_mapping(payload, "safety")
    if safety.get("local_veto") is not False:
        refusal_codes.add("LOCAL_SAFETY_VETO")

    transport = _nested_mapping(payload, "transport")
    if transport.get("network_required") is not True or transport.get("network_available") is not True:
        refusal_codes.add("NETWORK_UNAVAILABLE")

    heartbeat = _nested_mapping(payload, "heartbeat")
    heartbeat_observed = _strict_int(heartbeat.get("observed_at_ms"))
    heartbeat_evaluated = _strict_int(heartbeat.get("evaluated_at_ms"))
    heartbeat_timeout = _strict_int(heartbeat.get("timeout_ms"))
    if (
        heartbeat_observed is None
        or heartbeat_evaluated is None
        or heartbeat_timeout is None
        or heartbeat_timeout <= 0
        or not (heartbeat_observed <= heartbeat_evaluated)
        or heartbeat_evaluated - heartbeat_observed >= heartbeat_timeout
    ):
        refusal_codes.add("HEARTBEAT_TIMEOUT")

    runtime = _nested_mapping(payload, "runtime")
    expected_boot_session = runtime.get("expected_boot_session_id")
    actual_boot_session = runtime.get("actual_boot_session_id")
    if (
        not isinstance(expected_boot_session, str)
        or not expected_boot_session
        or not isinstance(actual_boot_session, str)
        or not actual_boot_session
        or expected_boot_session != actual_boot_session
    ):
        refusal_codes.add("BOOT_SESSION_CHANGED")

    bpu_provenance = _nested_mapping(payload, "bpu_provenance")
    expected_model = bpu_provenance.get("expected_model_sha256")
    actual_model = bpu_provenance.get("actual_model_sha256")
    expected_runtime = bpu_provenance.get("expected_runtime_sha256")
    actual_runtime = bpu_provenance.get("actual_runtime_sha256")
    if (
        not is_sha256(expected_model)
        or not is_sha256(actual_model)
        or not is_sha256(expected_runtime)
        or not is_sha256(actual_runtime)
        or expected_model != actual_model
        or expected_runtime != actual_runtime
    ):
        refusal_codes.add("BPU_PROVENANCE_MISMATCH")

    action = _nested_mapping(payload, "action")
    action_started = _strict_int(action.get("started_at_ms"))
    action_deadline = _strict_int(action.get("deadline_at_ms"))
    action_observed = _strict_int(action.get("observed_at_ms"))
    if (
        action_started is None
        or action_deadline is None
        or action_observed is None
        or action.get("completed") is not True
        or not (action_started <= action_observed < action_deadline)
    ):
        refusal_codes.add("EXECUTION_TIMEOUT")

    power = _nested_mapping(payload, "power")
    if power.get("power_good") is not True or power.get("brownout_detected") is not False:
        refusal_codes.add("SAFE_ABORT_AND_HOLD")

    authority = _nested_mapping(payload, "authority")
    episode_source = episode.get("evidence_source")
    if (
        authority.get("evidence_source") != EvidenceSource.SIMULATED_COUNTERFACTUAL.value
        or episode_source != EvidenceSource.SIMULATED_COUNTERFACTUAL.value
    ):
        refusal_codes.add("SIMULATION_PROVENANCE_INVALID")
    if authority.get("hardware_touch") is not False or episode.get("hardware_touch") is not False:
        refusal_codes.add("SIMULATION_HARDWARE_TOUCH_FORBIDDEN")
    if authority.get("execution_authority") is not False or episode.get("execution_authority") is not False:
        refusal_codes.add("SIMULATION_AUTHORITY_ESCALATION")
    if (
        authority.get("physical_risk_denominator_increment") != 0
        or episode.get("physical_risk_denominator_increment") != 0
    ):
        refusal_codes.add("SIMULATION_PHYSICAL_DENOMINATOR_FORBIDDEN")

    return AdmissionReport(tuple(sorted(refusal_codes)))


def inject_fault(payload: Mapping[str, Any], fault_kind: FaultKind) -> FaultInjection:
    """Deep-copy one envelope and apply exactly one preregistered mutation."""
    if not isinstance(fault_kind, FaultKind):
        fault_kind = FaultKind(fault_kind)
    before_sha256 = canonical_sha256(payload)
    mutated = deepcopy(dict(payload))

    if fault_kind is FaultKind.SHARED_SENSOR_CORRELATION:
        evidence_bundle = mutated["evidence"]
        records_by_id = {record["evidence_id"]: record for record in evidence_bundle["records"]}
        records_by_id["pl-acquisition-root"]["source_id"] = records_by_id["xrd-acquisition-root"]["source_id"]
        _refresh_evidence_dag_digest(evidence_bundle)
    elif fault_kind is FaultKind.STALE_CAPABILITY:
        mutated["capability"]["evaluated_at_ms"] = mutated["capability"]["expires_at_ms"]
    elif fault_kind is FaultKind.PROVENANCE_DRIFT:
        mutated["provenance"]["actual_sha256"] = canonical_sha256({"fault": fault_kind.value})
    elif fault_kind is FaultKind.HASH_TAMPER:
        mutated["episode"]["termination_reason"] = "TAMPERED"
    elif fault_kind is FaultKind.REPLAYED_NONCE:
        mutated["security"]["consumed_nonces"].append(mutated["security"]["nonce"])
    elif fault_kind is FaultKind.SAMPLE_LINEAGE_SWAP:
        mutated["sample"]["actual_lineage_sha256"] = canonical_sha256({"fault": fault_kind.value})
    elif fault_kind is FaultKind.LOCAL_VETO:
        mutated["safety"]["local_veto"] = True
    elif fault_kind is FaultKind.MISSING_STATION_CAPABILITY:
        mutated["capability"]["available_stations"] = []
    elif fault_kind is FaultKind.MISSING_LOADER_CAPABILITY:
        mutated["capability"]["available_loaders"] = []
    elif fault_kind is FaultKind.NETWORK_DROP:
        mutated["transport"]["network_available"] = False
    elif fault_kind is FaultKind.HEARTBEAT_TIMEOUT:
        heartbeat = mutated["heartbeat"]
        heartbeat["evaluated_at_ms"] = heartbeat["observed_at_ms"] + heartbeat["timeout_ms"]
    elif fault_kind is FaultKind.BOOT_SESSION_CHANGE:
        mutated["runtime"]["actual_boot_session_id"] = "boot-session-002"
    elif fault_kind is FaultKind.BPU_PROVENANCE_MISMATCH:
        mutated["bpu_provenance"]["actual_model_sha256"] = canonical_sha256({"fault": fault_kind.value})
    elif fault_kind is FaultKind.ACTION_TIMEOUT:
        mutated["action"]["observed_at_ms"] = mutated["action"]["deadline_at_ms"]
        mutated["action"]["completed"] = False
    elif fault_kind is FaultKind.POWER_DROP:
        mutated["power"]["power_good"] = False
        mutated["power"]["brownout_detected"] = True
    else:  # pragma: no cover - exhaustive enum guard
        raise AssertionError(f"unhandled fault kind: {fault_kind}")

    after_sha256 = canonical_sha256(mutated)
    return FaultInjection(
        fault_kind=fault_kind,
        before_sha256=before_sha256,
        after_sha256=after_sha256,
        expected_refusal_code=EXPECTED_REFUSAL_CODES[fault_kind],
        payload=mutated,
    )
