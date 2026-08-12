"""Strict semantic verification for one simulated RB-VoE transaction.

The byte ledger protects append order and local integrity. This verifier also
reconstructs the frozen request and replays its typed option/sequence semantics
so self-reported observations and exhaustive summaries carry no authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any

from rb_voe.audit.ledger import LedgerVerificationError, _verified_state
from rb_voe.contracts.canonical import canonical_sha256, is_sha256
from rb_voe.contracts.models import EvidenceRecord, EvidenceSource, ExperimentCase
from rb_voe.core.counterexample import (
    DecisionAssessment,
    PatchOperation,
    PerturbationRegistry,
    RegisteredPerturbation,
    StatePatch,
    apply_registered_perturbation,
    default_harm_assessment,
    evaluate_declarative_decision,
)
from rb_voe.core.evidence_dag import EvidenceDAG
from rb_voe.core.invariants import evaluate_evidence_invariants
from rb_voe.core.options import (
    ClosurePredicate,
    EvidenceOption,
    HardGate,
    OptionOutcome,
    SequenceOutcome,
    SequenceOutcomeModel,
)
from rb_voe.core.policy import (
    ContingentBranch,
    PolicyEvidenceAdmissionReport,
    PolicyMode,
    PolicyPlan,
    RejectedOption,
    canonical_policy_search_state,
)
from rb_voe.core.scenarios import JointScenario, JointScenarioSet, ScenarioVariant
from rb_voe.sim.simulator import (
    EpisodeResult,
    ExhaustiveReplayResult,
    SimulatedObservation,
    SimulatedOption,
    SimulationRequest,
)

CASE_EVENT = "EXPERIMENT_CASE"
PLAN_EVENT = "CONTINGENT_POLICY_PLAN"
OBSERVATION_EVENT = "SIMULATED_OBSERVATION"
TERMINAL_EVENT = "SIMULATED_TERMINAL"

CASE_SCHEMA = "xrd-rb-voe-experiment-case-v1"
PLAN_SCHEMA = "xrd-rb-voe-policy-plan-v5"
PLAN_EVENT_SCHEMA = "xrd-rb-voe-policy-plan-event-v4"
REQUEST_SCHEMA = "xrd-rb-voe-simulation-request-v4"
EPISODE_SCHEMA = "xrd-rb-voe-simulated-episode-v3"
OBSERVATION_EVENT_SCHEMA = "xrd-rb-voe-simulated-observation-event-v2"
EXHAUSTIVE_SCHEMA = "xrd-rb-voe-exhaustive-replay-v3"
RECEIPT_SCHEMA = "xrd-rb-voe-decision-receipt-v1"
TERMINAL_EVENT_SCHEMA = "xrd-rb-voe-simulated-terminal-event-v2"

_SEMANTIC_MARKER = object()

_CASE_FIELDS = frozenset(
    {
        "schema_version",
        "case_id",
        "sample_id",
        "lineage_sha256",
        "evidence_root_sha256",
        "allowed_options",
        "required_failure_atoms",
        "closure_predicate_id",
        "release_id",
        "created_at_ms",
    }
)
_PLAN_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "case_id",
        "experiment_case_sha256",
        "previous_record_sha256",
        "evidence_admission",
        "evidence_admission_sha256",
        "plan",
        "plan_sha256",
    }
)
_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "decision",
        "mode",
        "horizon",
        "variant",
        "case_id",
        "case_sha256",
        "closure_predicate_sha256",
        "evidence_root_sha256",
        "scenario_set_sha256",
        "option_set_sha256",
        "sequence_model_sha256",
        "evidence_admission_sha256",
        "evidence_invariant_report_sha256",
        "counterexample_registry_sha256",
        "counterexample_search_sha256",
        "counterexample_baseline_state_sha256",
        "evaluator_search_contract_release_sha256",
        "evidence_failure_domains",
        "scenario_failure_domains",
        "failure_domain_binding_sha256",
        "root_provenance_sha256",
        "root_option_id",
        "branches",
        "hold_risk",
        "plan_risk",
        "conservative_voi",
        "minimum_conservative_voi",
        "terminal_closure_guaranteed",
        "rejected_options",
        "reason",
    }
)
_BRANCH_FIELDS = frozenset({"observation", "conditioned_scenario_ids", "option_id"})
_REJECTED_OPTION_FIELDS = frozenset({"option_id", "failure_codes"})
_ADMISSION_FIELDS = frozenset(
    {
        "schema_version",
        "case_sha256",
        "evidence_dag",
        "evidence_dag_current_sha256",
        "selected_evidence_ids",
        "minimum_independent_evidence",
        "invariant_report",
        "invariant_report_sha256",
        "perturbation_registry",
        "perturbation_registry_sha256",
        "counterexample_search",
        "counterexample_search_sha256",
        "evaluator_search_contract",
        "evaluator_search_contract_release_sha256",
        "admission_report",
        "admission_report_sha256",
    }
)
_EVIDENCE_DAG_FIELDS = frozenset({"schema_version", "content_sha256", "records", "topological_order"})
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
_PERTURBATION_REGISTRY_FIELDS = frozenset({"schema_version", "perturbations"})
_PERTURBATION_FIELDS = frozenset(
    {
        "perturbation_id",
        "family",
        "patches",
        "distance",
        "failure_atoms",
        "affected_evidence_ids",
        "repair_options",
    }
)
_SEARCH_FIELDS = frozenset(
    {
        "schema_version",
        "registry_sha256",
        "baseline_state_sha256",
        "evaluator_search_contract_release_sha256",
        "baseline",
        "registered_count",
        "evaluated_count",
        "budget",
        "exhaustive",
        "best_found_id",
        "candidates",
    }
)
_SEARCH_CANDIDATE_FIELDS = frozenset(
    {
        "rank",
        "counterexample_id",
        "perturbation_id",
        "family",
        "distance",
        "harmful",
        "harm_score",
        "harm_reasons",
        "baseline",
        "perturbed",
        "failure_core",
        "affected_evidence_ids",
        "repair_options",
        "perturbed_state_sha256",
    }
)
_DECISION_ASSESSMENT_FIELDS = frozenset({"label", "permission_rank", "loss", "permissions"})
_ADMISSION_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "case_sha256",
        "evidence_root_sha256",
        "selected_evidence_ids",
        "minimum_independent_evidence",
        "invariant_report_sha256",
        "counterexample_registry_sha256",
        "counterexample_search_sha256",
        "counterexample_baseline_state_sha256",
        "evaluator_search_contract_release_sha256",
        "required_failure_atoms",
        "registered_count",
        "evaluated_count",
    }
)
_OBSERVATION_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "case_id",
        "experiment_case_sha256",
        "plan_sha256",
        "previous_record_sha256",
        "simulation_request",
        "simulation_request_sha256",
        "episode",
        "episode_sha256",
    }
)
_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "episode_id",
        "seed",
        "horizon",
        "scenario_set",
        "scenario_set_sha256",
        "scenario_variant",
        "options",
        "closure_predicate",
        "closure_predicate_sha256",
        "sequence_model",
        "sequence_model_sha256",
        "policy_plan_sha256",
        "pinned_policy_plan_sha256",
        "pinned_sequence_model_sha256",
        "root_provenance_sha256",
        "local_veto_steps",
    }
)
_SCENARIO_SET_FIELDS = frozenset({"schema_version", "scenario_set_id", "scenarios"})
_SCENARIO_FIELDS = frozenset(
    {
        "scenario_id",
        "hold_loss",
        "nominal_probability",
        "robust_probability",
        "failure_domains",
    }
)
_CLOSURE_PREDICATE_FIELDS = frozenset({"predicate_id", "required_failure_atoms"})
_SIMULATED_OPTION_FIELDS = frozenset({"option", "duration_ms_by_scenario"})
_OPTION_FIELDS = frozenset({"option_id", "cost", "outcomes", "hard_gates", "repeatable"})
_OUTCOME_FIELDS = frozenset({"scenario_id", "observation", "residual_loss", "closed_failure_atoms"})
_HARD_GATE_FIELDS = frozenset({"gate_id", "passed", "failure_code"})
_SEQUENCE_MODEL_FIELDS = frozenset({"schema_version", "model_id", "outcomes"})
_SEQUENCE_OUTCOME_FIELDS = frozenset(
    {
        "root_option_id",
        "root_observation",
        "second_option_id",
        "scenario_id",
        "terminal_residual_loss",
        "terminal_closed_failure_atoms",
    }
)
_EPISODE_FIELDS = frozenset(
    {
        "schema_version",
        "episode_id",
        "request_sha256",
        "policy_plan_sha256",
        "sequence_model_sha256",
        "root_provenance_sha256",
        "scenario_set_sha256",
        "root_scenario_id",
        "root_scenario_draw_sha256",
        "root_scenario_selection_count",
        "observations",
        "cumulative_duration_ms",
        "modeled_closed_failure_atoms",
        "modeled_closure_satisfied",
        "closure_semantics",
        "physical_closure_proven",
        "termination_reason",
        "evidence_source",
        "hardware_touch",
        "execution_authority",
        "physical_risk_denominator_increment",
    }
)
_STEP_FIELDS = frozenset(
    {
        "step_index",
        "option_id",
        "root_scenario_id",
        "duration_ms",
        "observation",
        "residual_loss",
        "closed_failure_atoms",
        "cumulative_modeled_closed_failure_atoms",
        "modeled_closure_satisfied",
        "sequence_outcome_sha256",
        "terminal_residual_source",
        "closure_semantics",
        "physical_closure_proven",
        "provenance",
        "hardware_touch",
    }
)
_EXHAUSTIVE_FIELDS = frozenset(
    {
        "schema_version",
        "request_sha256",
        "policy_plan_sha256",
        "sequence_model_sha256",
        "root_provenance_sha256",
        "scenario_set_sha256",
        "expected_scenario_ids",
        "scenario_ids",
        "episode_sha256s",
        "episodes",
        "all_modeled_closure_satisfied",
        "closure_semantics",
        "physical_closure_proven",
    }
)
_TERMINAL_FIELDS = frozenset(
    {
        "schema_version",
        "case_id",
        "experiment_case_sha256",
        "plan_sha256",
        "simulation_request_sha256",
        "episode_sha256",
        "previous_record_sha256",
        "exhaustive_replay",
        "exhaustive_replay_sha256",
        "decision_receipt",
        "decision_receipt_sha256",
        "execution_authority",
        "hardware_touched",
        "network_touched",
        "physical_risk_denominator_increment",
        "simulated_only",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "receipt_id",
        "case_id",
        "plan_epoch",
        "decision",
        "failure_core_closed",
        "selected_option",
        "terminal_evidence_sha256",
        "release_certificate_sha256",
        "previous_record_sha256",
        "created_at_ms",
    }
)


def _fail(code: str, message: str, line_number: int) -> LedgerVerificationError:
    return LedgerVerificationError(code, message, line_number=line_number)


def _require_exact_fields(
    payload: Mapping[str, Any],
    expected: frozenset[str],
    *,
    code: str,
    line_number: int,
) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise _fail(
            code,
            f"semantic payload fields mismatch; missing={missing}, extra={extra}",
            line_number,
        )


def _require_mapping(value: object, *, code: str, field_name: str, line_number: int) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise _fail(code, f"{field_name} must be a JSON object", line_number)
    return value


def _require_string(value: object, *, code: str, field_name: str, line_number: int) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(code, f"{field_name} must be a non-empty string", line_number)
    return value


def _require_hash(value: object, *, code: str, field_name: str, line_number: int) -> str:
    if not is_sha256(value):
        raise _fail(code, f"{field_name} must be a full SHA-256 digest", line_number)
    return value


def _require_int(
    value: object,
    *,
    code: str,
    field_name: str,
    line_number: int,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _fail(code, f"{field_name} must be an integer >= {minimum}", line_number)
    return value


def _require_number(value: object, *, code: str, field_name: str, line_number: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise _fail(code, f"{field_name} must be a finite number", line_number)
    return float(value)


def _require_bool(value: object, *, code: str, field_name: str, line_number: int) -> bool:
    if not isinstance(value, bool):
        raise _fail(code, f"{field_name} must be boolean", line_number)
    return value


def _require_string_list(
    value: object,
    *,
    code: str,
    field_name: str,
    line_number: int,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise _fail(code, f"{field_name} must be a JSON array", line_number)
    if any(not isinstance(item, str) or not item for item in value):
        raise _fail(code, f"{field_name} contains an invalid identifier", line_number)
    if len(set(value)) != len(value):
        raise _fail(code, f"{field_name} contains duplicates", line_number)
    return tuple(value)


def _typed_failure(code: str, message: str, line_number: int, exc: Exception) -> LedgerVerificationError:
    return _fail(code, f"{message}: {exc}", line_number)


@dataclass(frozen=True, slots=True)
class SemanticLedgerReport:
    """Immutable digest report minted by the strict semantic verifier."""

    schema_version: str
    record_count: int
    ledger_file_sha256: str
    terminal_record_sha256: str
    anchor_sha256: str
    case_id: str
    experiment_case_sha256: str
    policy_plan_sha256: str
    simulation_request_sha256: str
    exhaustive_replay_sha256: str
    scenario_set_sha256: str
    option_set_sha256: str
    sequence_model_sha256: str
    evidence_admission_sha256: str
    evidence_invariant_report_sha256: str
    counterexample_registry_sha256: str
    counterexample_search_sha256: str
    counterexample_baseline_state_sha256: str
    evaluator_search_contract_release_sha256: str
    failure_domain_binding_sha256: str
    episode_sha256: tuple[str, ...]
    decision_receipt_sha256: str
    decision: str
    selected_option: str | None
    started_at_ms: int
    ended_at_ms: int
    simulated_only: bool
    hardware_touched: bool
    network_touched: bool
    execution_authority: bool
    physical_risk_denominator_increment: int
    _marker: object = field(repr=False, compare=False)

    @property
    def verified(self) -> bool:
        return self._marker is _SEMANTIC_MARKER

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_count": self.record_count,
            "ledger_file_sha256": self.ledger_file_sha256,
            "terminal_record_sha256": self.terminal_record_sha256,
            "anchor_sha256": self.anchor_sha256,
            "case_id": self.case_id,
            "experiment_case_sha256": self.experiment_case_sha256,
            "policy_plan_sha256": self.policy_plan_sha256,
            "simulation_request_sha256": self.simulation_request_sha256,
            "exhaustive_replay_sha256": self.exhaustive_replay_sha256,
            "scenario_set_sha256": self.scenario_set_sha256,
            "option_set_sha256": self.option_set_sha256,
            "sequence_model_sha256": self.sequence_model_sha256,
            "evidence_admission_sha256": self.evidence_admission_sha256,
            "evidence_invariant_report_sha256": self.evidence_invariant_report_sha256,
            "counterexample_registry_sha256": self.counterexample_registry_sha256,
            "counterexample_search_sha256": self.counterexample_search_sha256,
            "counterexample_baseline_state_sha256": self.counterexample_baseline_state_sha256,
            "evaluator_search_contract_release_sha256": (self.evaluator_search_contract_release_sha256),
            "failure_domain_binding_sha256": self.failure_domain_binding_sha256,
            "episode_sha256": list(self.episode_sha256),
            "decision_receipt_sha256": self.decision_receipt_sha256,
            "decision": self.decision,
            "selected_option": self.selected_option,
            "started_at_ms": self.started_at_ms,
            "ended_at_ms": self.ended_at_ms,
            "simulated_only": self.simulated_only,
            "hardware_touched": self.hardware_touched,
            "network_touched": self.network_touched,
            "execution_authority": self.execution_authority,
            "physical_risk_denominator_increment": self.physical_risk_denominator_increment,
        }

    @property
    def report_sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def require_verified_semantic_report(report: object) -> SemanticLedgerReport:
    if not isinstance(report, SemanticLedgerReport) or not report.verified:
        raise ValueError("terminal artifacts require a verified semantic ledger report")
    return report


def _validate_case(payload: Mapping[str, Any], line_number: int) -> dict[str, Any]:
    _require_exact_fields(payload, _CASE_FIELDS, code="CASE_PAYLOAD_INVALID", line_number=line_number)
    if payload["schema_version"] != CASE_SCHEMA:
        raise _fail("CASE_SCHEMA_INVALID", "unsupported experiment case schema", line_number)
    try:
        case = ExperimentCase(
            schema_version=payload["schema_version"],
            case_id=payload["case_id"],
            sample_id=payload["sample_id"],
            lineage_sha256=payload["lineage_sha256"],
            evidence_root_sha256=payload["evidence_root_sha256"],
            allowed_options=tuple(payload["allowed_options"]),
            required_failure_atoms=tuple(payload["required_failure_atoms"]),
            closure_predicate_id=payload["closure_predicate_id"],
            release_id=payload["release_id"],
            created_at_ms=payload["created_at_ms"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise _typed_failure("CASE_PAYLOAD_INVALID", "invalid ExperimentCase", line_number, exc) from exc
    if case.to_dict() != dict(payload):
        raise _fail("CASE_PAYLOAD_INVALID", "case is not canonical ExperimentCase serialization", line_number)
    closure = ClosurePredicate(case.closure_predicate_id, case.required_failure_atoms)
    return {
        "object": case,
        "case_id": case.case_id,
        "case_sha256": case.content_sha256,
        "allowed_options": frozenset(case.allowed_options),
        "required_failure_atoms": frozenset(case.required_failure_atoms),
        "closure_predicate": closure,
        "created_at_ms": case.created_at_ms,
    }


def _parse_decision_assessment(
    payload: Mapping[str, Any],
    *,
    line_number: int,
) -> DecisionAssessment:
    _require_exact_fields(
        payload,
        _DECISION_ASSESSMENT_FIELDS,
        code="ADMISSION_SEARCH_INVALID",
        line_number=line_number,
    )
    try:
        assessment = DecisionAssessment(
            label=payload["label"],
            permission_rank=payload["permission_rank"],
            loss=payload["loss"],
            permissions=tuple(payload["permissions"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise _typed_failure(
            "ADMISSION_SEARCH_INVALID",
            "invalid decision assessment",
            line_number,
            exc,
        ) from exc
    if assessment.to_dict() != dict(payload):
        raise _fail(
            "ADMISSION_SEARCH_INVALID",
            "decision assessment is not canonical",
            line_number,
        )
    return assessment


def _validate_evidence_admission(
    payload: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    supplied_sha256: object,
    line_number: int,
) -> dict[str, Any]:
    _require_exact_fields(
        payload,
        _ADMISSION_FIELDS,
        code="ADMISSION_PAYLOAD_INVALID",
        line_number=line_number,
    )
    if payload["schema_version"] != "xrd-rb-voe-policy-evidence-admission-v1":
        raise _fail("ADMISSION_SCHEMA_INVALID", "unsupported evidence admission schema", line_number)
    admission_sha256 = canonical_sha256(payload)
    if supplied_sha256 != admission_sha256:
        raise _fail("ADMISSION_HASH_MISMATCH", "evidence admission digest mismatch", line_number)
    if payload["case_sha256"] != case["case_sha256"]:
        raise _fail("ADMISSION_CASE_MISMATCH", "evidence admission does not bind CASE", line_number)

    dag_payload = _require_mapping(
        payload["evidence_dag"],
        code="ADMISSION_DAG_INVALID",
        field_name="evidence_dag",
        line_number=line_number,
    )
    _require_exact_fields(
        dag_payload,
        _EVIDENCE_DAG_FIELDS,
        code="ADMISSION_DAG_INVALID",
        line_number=line_number,
    )
    if dag_payload["schema_version"] != "xrd-rb-voe-evidence-dag-v1":
        raise _fail("ADMISSION_DAG_INVALID", "unsupported EvidenceDAG schema", line_number)
    records_value = dag_payload["records"]
    if not isinstance(records_value, list) or not records_value:
        raise _fail("ADMISSION_DAG_INVALID", "EvidenceDAG records must be non-empty", line_number)
    try:
        records: list[EvidenceRecord] = []
        for value in records_value:
            item = _require_mapping(
                value,
                code="ADMISSION_DAG_INVALID",
                field_name="evidence record",
                line_number=line_number,
            )
            _require_exact_fields(
                item,
                _EVIDENCE_RECORD_FIELDS,
                code="ADMISSION_DAG_INVALID",
                line_number=line_number,
            )
            record = EvidenceRecord(
                schema_version=item["schema_version"],
                evidence_id=item["evidence_id"],
                kind=item["kind"],
                source=EvidenceSource(item["source"]),
                source_id=item["source_id"],
                lineage_sha256=item["lineage_sha256"],
                payload_sha256=item["payload_sha256"],
                observed_at_ms=item["observed_at_ms"],
                acquisition_id=item["acquisition_id"],
                parent_evidence_ids=tuple(item["parent_evidence_ids"]),
                metadata=item["metadata"],
            )
            if record.to_dict() != dict(item):
                raise ValueError("evidence record is not canonical")
            records.append(record)
        dag = EvidenceDAG(tuple(records))
    except (KeyError, TypeError, ValueError) as exc:
        raise _typed_failure("ADMISSION_DAG_INVALID", "invalid EvidenceDAG", line_number, exc) from exc
    if dag.to_dict() != dict(dag_payload):
        raise _fail("ADMISSION_DAG_INVALID", "EvidenceDAG is not canonical", line_number)
    if (
        dag.content_sha256 != case["object"].evidence_root_sha256
        or payload["evidence_dag_current_sha256"] != dag.current_sha256
    ):
        raise _fail("ADMISSION_DAG_BINDING_MISMATCH", "EvidenceDAG digest drifted", line_number)

    selected_evidence_ids = _require_string_list(
        payload["selected_evidence_ids"],
        code="ADMISSION_EVIDENCE_IDS_INVALID",
        field_name="selected_evidence_ids",
        line_number=line_number,
        allow_empty=False,
    )
    if selected_evidence_ids != tuple(sorted(selected_evidence_ids)):
        raise _fail(
            "ADMISSION_EVIDENCE_IDS_INVALID",
            "selected evidence ids are not canonical",
            line_number,
        )
    minimum_independent = _require_int(
        payload["minimum_independent_evidence"],
        code="ADMISSION_INVARIANT_INVALID",
        field_name="minimum_independent_evidence",
        line_number=line_number,
        minimum=1,
    )
    try:
        invariant_report = evaluate_evidence_invariants(
            dag,
            evidence_ids=selected_evidence_ids,
            minimum_independent=minimum_independent,
            expected_sha256=case["object"].evidence_root_sha256,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise _typed_failure(
            "ADMISSION_INVARIANT_INVALID",
            "cannot recompute evidence invariants",
            line_number,
            exc,
        ) from exc
    if not invariant_report.passed:
        raise _fail("ADMISSION_INVARIANT_FAILED", "evidence invariants do not pass", line_number)
    if (
        payload["invariant_report"] != invariant_report.to_dict()
        or payload["invariant_report_sha256"] != invariant_report.content_sha256
    ):
        raise _fail(
            "ADMISSION_INVARIANT_MISMATCH",
            "invariant report differs from recomputed evidence invariants",
            line_number,
        )

    registry_payload = _require_mapping(
        payload["perturbation_registry"],
        code="ADMISSION_REGISTRY_INVALID",
        field_name="perturbation_registry",
        line_number=line_number,
    )
    _require_exact_fields(
        registry_payload,
        _PERTURBATION_REGISTRY_FIELDS,
        code="ADMISSION_REGISTRY_INVALID",
        line_number=line_number,
    )
    if registry_payload["schema_version"] != "xrd-rb-voe-perturbation-registry-v1":
        raise _fail("ADMISSION_REGISTRY_INVALID", "unsupported perturbation registry", line_number)
    perturbations_value = registry_payload["perturbations"]
    if not isinstance(perturbations_value, list) or not perturbations_value:
        raise _fail("ADMISSION_REGISTRY_INVALID", "perturbation registry is empty", line_number)
    try:
        perturbations: list[RegisteredPerturbation] = []
        for value in perturbations_value:
            item = _require_mapping(
                value,
                code="ADMISSION_REGISTRY_INVALID",
                field_name="registered perturbation",
                line_number=line_number,
            )
            _require_exact_fields(
                item,
                _PERTURBATION_FIELDS,
                code="ADMISSION_REGISTRY_INVALID",
                line_number=line_number,
            )
            patches: list[StatePatch] = []
            for raw_patch in item["patches"]:
                patch = _require_mapping(
                    raw_patch,
                    code="ADMISSION_REGISTRY_INVALID",
                    field_name="state patch",
                    line_number=line_number,
                )
                operation = PatchOperation(patch["operation"])
                expected_patch_fields = {"path", "operation", "allow_create"}
                if operation is PatchOperation.SET:
                    expected_patch_fields.add("value")
                if set(patch) != expected_patch_fields:
                    raise ValueError("state patch fields are not canonical")
                patches.append(
                    StatePatch(
                        path=tuple(patch["path"]),
                        value=patch.get("value"),
                        operation=operation,
                        allow_create=patch["allow_create"],
                    )
                )
            perturbations.append(
                RegisteredPerturbation(
                    perturbation_id=item["perturbation_id"],
                    family=item["family"],
                    patches=tuple(patches),
                    distance=item["distance"],
                    failure_atoms=tuple(item["failure_atoms"]),
                    affected_evidence_ids=tuple(item["affected_evidence_ids"]),
                    repair_options=tuple(item["repair_options"]),
                )
            )
        registry = PerturbationRegistry(tuple(perturbations))
    except (KeyError, TypeError, ValueError) as exc:
        raise _typed_failure(
            "ADMISSION_REGISTRY_INVALID",
            "invalid perturbation registry",
            line_number,
            exc,
        ) from exc
    if registry.to_dict() != dict(registry_payload):
        raise _fail("ADMISSION_REGISTRY_INVALID", "perturbation registry is not canonical", line_number)
    if payload["perturbation_registry_sha256"] != registry.content_sha256:
        raise _fail("ADMISSION_REGISTRY_HASH_MISMATCH", "perturbation registry digest mismatch", line_number)

    evaluator_contract = _require_mapping(
        payload["evaluator_search_contract"],
        code="ADMISSION_EVALUATOR_INVALID",
        field_name="evaluator_search_contract",
        line_number=line_number,
    )
    evaluator_release_sha256 = canonical_sha256(evaluator_contract)
    if payload["evaluator_search_contract_release_sha256"] != evaluator_release_sha256:
        raise _fail(
            "ADMISSION_EVALUATOR_HASH_MISMATCH",
            "evaluator/search contract digest mismatch",
            line_number,
        )

    search_payload = _require_mapping(
        payload["counterexample_search"],
        code="ADMISSION_SEARCH_INVALID",
        field_name="counterexample_search",
        line_number=line_number,
    )
    _require_exact_fields(
        search_payload,
        _SEARCH_FIELDS,
        code="ADMISSION_SEARCH_INVALID",
        line_number=line_number,
    )
    if search_payload["schema_version"] != "xrd-rb-voe-counterexample-search-v2":
        raise _fail("ADMISSION_SEARCH_INVALID", "unsupported counterexample search", line_number)
    expected_state = canonical_policy_search_state(
        experiment_case=case["object"],
        evidence_dag=dag,
        selected_evidence_ids=selected_evidence_ids,
    )
    expected_state_sha256 = canonical_sha256(expected_state)
    if (
        search_payload["registry_sha256"] != registry.content_sha256
        or search_payload["baseline_state_sha256"] != expected_state_sha256
        or search_payload["evaluator_search_contract_release_sha256"] != evaluator_release_sha256
        or search_payload["registered_count"] != len(registry)
        or search_payload["evaluated_count"] != len(registry)
        or search_payload["budget"] != len(registry)
        or search_payload["exhaustive"] is not True
    ):
        raise _fail(
            "ADMISSION_SEARCH_BINDING_MISMATCH",
            "counterexample search is not exhaustively bound to the admitted objects",
            line_number,
        )
    baseline_payload = _require_mapping(
        search_payload["baseline"],
        code="ADMISSION_SEARCH_INVALID",
        field_name="search baseline",
        line_number=line_number,
    )
    baseline = _parse_decision_assessment(baseline_payload, line_number=line_number)
    try:
        expected_baseline = evaluate_declarative_decision(expected_state, evaluator_contract)
    except (KeyError, TypeError, ValueError) as exc:
        raise _typed_failure(
            "ADMISSION_EVALUATOR_INVALID",
            "declarative evaluator replay failed",
            line_number,
            exc,
        ) from exc
    if baseline != expected_baseline:
        raise _fail(
            "ADMISSION_EVALUATOR_REPLAY_MISMATCH",
            "search baseline differs from declarative evaluator replay",
            line_number,
        )
    candidates_value = search_payload["candidates"]
    if not isinstance(candidates_value, list) or len(candidates_value) != len(registry):
        raise _fail("ADMISSION_SEARCH_COVERAGE_INVALID", "search candidate roster is incomplete", line_number)
    registry_by_id = {item.perturbation_id: item for item in registry.perturbations}
    seen_ids: set[str] = set()
    candidate_sort_values: list[tuple[tuple[object, ...], str]] = []
    first_harmful_id: str | None = None
    for expected_rank, value in enumerate(candidates_value, 1):
        item = _require_mapping(
            value,
            code="ADMISSION_SEARCH_INVALID",
            field_name="counterexample candidate",
            line_number=line_number,
        )
        _require_exact_fields(
            item,
            _SEARCH_CANDIDATE_FIELDS,
            code="ADMISSION_SEARCH_INVALID",
            line_number=line_number,
        )
        perturbation_id = item["perturbation_id"]
        if perturbation_id in seen_ids or perturbation_id not in registry_by_id:
            raise _fail(
                "ADMISSION_SEARCH_COVERAGE_INVALID",
                "search candidate does not cover the registry exactly once",
                line_number,
            )
        seen_ids.add(perturbation_id)
        perturbation = registry_by_id[perturbation_id]
        perturbed_payload = _require_mapping(
            item["perturbed"],
            code="ADMISSION_SEARCH_INVALID",
            field_name="perturbed assessment",
            line_number=line_number,
        )
        perturbed = _parse_decision_assessment(perturbed_payload, line_number=line_number)
        candidate_baseline_payload = _require_mapping(
            item["baseline"],
            code="ADMISSION_SEARCH_INVALID",
            field_name="candidate baseline",
            line_number=line_number,
        )
        candidate_baseline = _parse_decision_assessment(
            candidate_baseline_payload,
            line_number=line_number,
        )
        harm_score = _require_number(
            item["harm_score"],
            code="ADMISSION_SEARCH_INVALID",
            field_name="harm_score",
            line_number=line_number,
        )
        if harm_score < 0 or item["harmful"] is not (harm_score > 0):
            raise _fail("ADMISSION_SEARCH_INVALID", "candidate harm semantics are invalid", line_number)
        perturbed_state = apply_registered_perturbation(expected_state, perturbation)
        perturbed_state_sha256 = canonical_sha256(perturbed_state)
        try:
            expected_perturbed = evaluate_declarative_decision(perturbed_state, evaluator_contract)
        except (KeyError, TypeError, ValueError) as exc:
            raise _typed_failure(
                "ADMISSION_EVALUATOR_INVALID",
                "declarative evaluator perturbation replay failed",
                line_number,
                exc,
            ) from exc
        expected_harm = default_harm_assessment(expected_baseline, expected_perturbed)
        counterexample_id = canonical_sha256(
            {
                "schema_version": "xrd-rb-voe-counterexample-v2",
                "baseline_state_sha256": expected_state_sha256,
                "evaluator_search_contract_release_sha256": evaluator_release_sha256,
                "perturbation_id": perturbation_id,
                "perturbed_state_sha256": perturbed_state_sha256,
                "baseline": baseline.to_dict(),
                "perturbed": perturbed.to_dict(),
                "harm_score": harm_score,
                "harm_reasons": item["harm_reasons"],
                "failure_core": item["failure_core"],
            }
        )
        if (
            _require_int(
                item["rank"],
                code="ADMISSION_SEARCH_INVALID",
                field_name="rank",
                line_number=line_number,
            )
            != expected_rank
            or candidate_baseline != baseline
            or perturbed != expected_perturbed
            or abs(harm_score - expected_harm.score) > 1e-12
            or item["harm_reasons"] != list(expected_harm.reasons)
            or item["harmful"] is not expected_harm.harmful
            or item["family"] != perturbation.family
            or item["distance"] != perturbation.distance
            or item["failure_core"] != list(perturbation.failure_atoms)
            or item["affected_evidence_ids"] != list(perturbation.affected_evidence_ids)
            or item["repair_options"] != list(perturbation.repair_options)
            or item["perturbed_state_sha256"] != perturbed_state_sha256
            or item["counterexample_id"] != counterexample_id
        ):
            raise _fail(
                "ADMISSION_SEARCH_CANDIDATE_MISMATCH",
                "counterexample candidate differs from its registered perturbation",
                line_number,
            )
        if item["harmful"] and first_harmful_id is None:
            first_harmful_id = perturbation_id
        candidate_sort_values.append(
            (
                (
                    not item["harmful"],
                    -harm_score,
                    item["distance"],
                    item["family"],
                    perturbation_id,
                ),
                perturbation_id,
            )
        )
    if candidate_sort_values != sorted(candidate_sort_values) or set(seen_ids) != set(registry_by_id):
        raise _fail("ADMISSION_SEARCH_ORDER_INVALID", "counterexample ranking is not canonical", line_number)
    if search_payload["best_found_id"] != first_harmful_id:
        raise _fail("ADMISSION_SEARCH_INVALID", "best-found counterexample link is invalid", line_number)
    search_sha256 = canonical_sha256(search_payload)
    if payload["counterexample_search_sha256"] != search_sha256:
        raise _fail("ADMISSION_SEARCH_HASH_MISMATCH", "counterexample search digest mismatch", line_number)

    expected_report = PolicyEvidenceAdmissionReport(
        case_sha256=case["case_sha256"],
        evidence_root_sha256=dag.content_sha256,
        selected_evidence_ids=selected_evidence_ids,
        minimum_independent_evidence=minimum_independent,
        invariant_report_sha256=invariant_report.content_sha256,
        counterexample_registry_sha256=registry.content_sha256,
        counterexample_search_sha256=search_sha256,
        counterexample_baseline_state_sha256=expected_state_sha256,
        evaluator_search_contract_release_sha256=evaluator_release_sha256,
        required_failure_atoms=tuple(sorted(case["object"].required_failure_atoms)),
        registered_count=len(registry),
        evaluated_count=len(registry),
    )
    report_payload = _require_mapping(
        payload["admission_report"],
        code="ADMISSION_REPORT_INVALID",
        field_name="admission_report",
        line_number=line_number,
    )
    _require_exact_fields(
        report_payload,
        _ADMISSION_REPORT_FIELDS,
        code="ADMISSION_REPORT_INVALID",
        line_number=line_number,
    )
    if (
        report_payload != expected_report.to_dict()
        or payload["admission_report_sha256"] != expected_report.content_sha256
    ):
        raise _fail(
            "ADMISSION_REPORT_MISMATCH",
            "admission report differs from recomputed evidence/search objects",
            line_number,
        )
    return {
        "admission_sha256": admission_sha256,
        "invariant_report_sha256": invariant_report.content_sha256,
        "registry_sha256": registry.content_sha256,
        "search_sha256": search_sha256,
        "baseline_state_sha256": expected_state_sha256,
        "evaluator_release_sha256": evaluator_release_sha256,
        "evidence_failure_domains": dag.failure_domains(selected_evidence_ids),
    }


def _parse_plan_object(payload: Mapping[str, Any], line_number: int) -> PolicyPlan:
    _require_exact_fields(payload, _PLAN_FIELDS, code="PLAN_PAYLOAD_INVALID", line_number=line_number)
    if payload["schema_version"] != PLAN_SCHEMA:
        raise _fail("PLAN_SCHEMA_INVALID", "unsupported policy plan schema", line_number)
    try:
        branches_value = payload["branches"]
        rejected_value = payload["rejected_options"]
        if not isinstance(branches_value, list) or not isinstance(rejected_value, list):
            raise TypeError("branches and rejected_options must be arrays")
        branches: list[ContingentBranch] = []
        for branch_value in branches_value:
            branch = _require_mapping(
                branch_value,
                code="PLAN_BRANCH_INVALID",
                field_name="branch",
                line_number=line_number,
            )
            _require_exact_fields(branch, _BRANCH_FIELDS, code="PLAN_BRANCH_INVALID", line_number=line_number)
            branches.append(
                ContingentBranch(
                    observation=branch["observation"],
                    conditioned_scenario_ids=tuple(branch["conditioned_scenario_ids"]),
                    option_id=branch["option_id"],
                )
            )
        rejected: list[RejectedOption] = []
        for rejected_value_item in rejected_value:
            item = _require_mapping(
                rejected_value_item,
                code="PLAN_REJECTIONS_INVALID",
                field_name="rejected option",
                line_number=line_number,
            )
            _require_exact_fields(
                item,
                _REJECTED_OPTION_FIELDS,
                code="PLAN_REJECTIONS_INVALID",
                line_number=line_number,
            )
            rejected.append(RejectedOption(item["option_id"], tuple(item["failure_codes"])))
        plan = PolicyPlan(
            decision=payload["decision"],
            mode=PolicyMode(payload["mode"]),
            horizon=payload["horizon"],
            variant=ScenarioVariant(payload["variant"]),
            case_id=payload["case_id"],
            case_sha256=payload["case_sha256"],
            closure_predicate_sha256=payload["closure_predicate_sha256"],
            evidence_root_sha256=payload["evidence_root_sha256"],
            scenario_set_sha256=payload["scenario_set_sha256"],
            option_set_sha256=payload["option_set_sha256"],
            sequence_model_sha256=payload["sequence_model_sha256"],
            evidence_admission_sha256=payload["evidence_admission_sha256"],
            evidence_invariant_report_sha256=payload["evidence_invariant_report_sha256"],
            counterexample_registry_sha256=payload["counterexample_registry_sha256"],
            counterexample_search_sha256=payload["counterexample_search_sha256"],
            counterexample_baseline_state_sha256=payload["counterexample_baseline_state_sha256"],
            evaluator_search_contract_release_sha256=payload["evaluator_search_contract_release_sha256"],
            evidence_failure_domains=tuple(payload["evidence_failure_domains"]),
            scenario_failure_domains=tuple(payload["scenario_failure_domains"]),
            failure_domain_binding_sha256=payload["failure_domain_binding_sha256"],
            root_provenance_sha256=payload["root_provenance_sha256"],
            root_option_id=payload["root_option_id"],
            branches=tuple(branches),
            hold_risk=payload["hold_risk"],
            plan_risk=payload["plan_risk"],
            conservative_voi=payload["conservative_voi"],
            minimum_conservative_voi=payload["minimum_conservative_voi"],
            terminal_closure_guaranteed=payload["terminal_closure_guaranteed"],
            rejected_options=tuple(rejected),
            reason=payload["reason"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise _typed_failure("PLAN_PAYLOAD_INVALID", "invalid PolicyPlan", line_number, exc) from exc
    if plan.to_dict() != dict(payload):
        raise _fail("PLAN_PAYLOAD_INVALID", "plan is not exact PolicyPlan.to_dict output", line_number)
    return plan


def _validate_plan(
    payload: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
    previous_row: Mapping[str, Any],
    case: Mapping[str, Any],
    line_number: int,
) -> dict[str, Any]:
    _require_exact_fields(
        payload, _PLAN_EVENT_FIELDS, code="PLAN_EVENT_PAYLOAD_INVALID", line_number=line_number
    )
    if payload["schema_version"] != PLAN_EVENT_SCHEMA:
        raise _fail("PLAN_EVENT_SCHEMA_INVALID", "unsupported plan event schema", line_number)
    if payload["case_id"] != case["case_id"]:
        raise _fail("PLAN_CASE_MISMATCH", "plan event case_id does not match CASE", line_number)
    if payload["experiment_case_sha256"] != case["case_sha256"]:
        raise _fail("PLAN_CASE_HASH_MISMATCH", "plan event does not bind CASE", line_number)
    if payload["previous_record_sha256"] != previous_row["record_sha256"]:
        raise _fail("SEMANTIC_PREVIOUS_HASH_MISMATCH", "plan does not bind previous record", line_number)
    admission_payload = _require_mapping(
        payload["evidence_admission"],
        code="ADMISSION_PAYLOAD_INVALID",
        field_name="evidence_admission",
        line_number=line_number,
    )
    admission = _validate_evidence_admission(
        admission_payload,
        case=case,
        supplied_sha256=payload["evidence_admission_sha256"],
        line_number=line_number,
    )
    plan_payload = _require_mapping(
        payload["plan"], code="PLAN_PAYLOAD_INVALID", field_name="plan", line_number=line_number
    )
    plan = _parse_plan_object(plan_payload, line_number)
    if plan.case_id != case["case_id"] or plan.case_sha256 != case["case_sha256"]:
        raise _fail("PLAN_CASE_HASH_MISMATCH", "plan does not bind verified CASE", line_number)
    if plan.decision != "NEXT_EVIDENCE" or plan.root_option_id is None:
        raise _fail("PLAN_DECISION_INVALID", "R1 replay requires a NEXT_EVIDENCE plan", line_number)
    if plan.evidence_root_sha256 != case["object"].evidence_root_sha256:
        raise _fail("PLAN_EVIDENCE_ROOT_MISMATCH", "plan evidence root differs from CASE", line_number)
    admission_bindings = {
        "admission": (plan.evidence_admission_sha256, admission["admission_sha256"]),
        "invariant report": (
            plan.evidence_invariant_report_sha256,
            admission["invariant_report_sha256"],
        ),
        "counterexample registry": (
            plan.counterexample_registry_sha256,
            admission["registry_sha256"],
        ),
        "counterexample search": (
            plan.counterexample_search_sha256,
            admission["search_sha256"],
        ),
        "counterexample baseline state": (
            plan.counterexample_baseline_state_sha256,
            admission["baseline_state_sha256"],
        ),
        "evaluator/search contract": (
            plan.evaluator_search_contract_release_sha256,
            admission["evaluator_release_sha256"],
        ),
    }
    for binding_name, (actual, expected) in admission_bindings.items():
        if actual != expected:
            raise _fail(
                "PLAN_ADMISSION_BINDING_MISMATCH",
                f"plan {binding_name} digest differs from verified evidence admission",
                line_number,
            )
    if plan.evidence_failure_domains != admission["evidence_failure_domains"]:
        raise _fail(
            "PLAN_FAILURE_DOMAIN_MISMATCH",
            "plan evidence failure domains differ from verified evidence admission",
            line_number,
        )
    branch_observations = tuple(branch.observation for branch in plan.branches)
    if len(set(branch_observations)) != len(branch_observations):
        raise _fail("PLAN_BRANCH_DUPLICATE", "plan branch observation is duplicated", line_number)
    if any(
        not branch.conditioned_scenario_ids
        or len(set(branch.conditioned_scenario_ids)) != len(branch.conditioned_scenario_ids)
        for branch in plan.branches
    ):
        raise _fail(
            "PLAN_BRANCH_DUPLICATE",
            "plan branch scenarios must be non-empty and unique",
            line_number,
        )
    if plan.root_option_id not in case["allowed_options"] or any(
        branch.option_id is not None and branch.option_id not in case["allowed_options"]
        for branch in plan.branches
    ):
        raise _fail("PLAN_OPTION_NOT_ALLOWED", "plan references an option outside CASE", line_number)
    plan_sha256 = plan.plan_sha256
    if payload["plan_sha256"] != plan_sha256:
        raise _fail("PLAN_HASH_MISMATCH", "plan_sha256 does not match plan", line_number)
    return {
        "object": plan,
        "payload": dict(plan_payload),
        "plan_sha256": plan_sha256,
        "evidence_admission": admission,
        "record_sha256": row["record_sha256"],
    }


def _parse_sequence_model(payload: Mapping[str, Any], line_number: int) -> SequenceOutcomeModel:
    _require_exact_fields(
        payload, _SEQUENCE_MODEL_FIELDS, code="REQUEST_SEQUENCE_MODEL_INVALID", line_number=line_number
    )
    if payload["schema_version"] != "xrd-rb-voe-sequence-outcome-model-v1":
        raise _fail("REQUEST_SEQUENCE_MODEL_INVALID", "unsupported sequence model schema", line_number)
    outcomes_value = payload["outcomes"]
    if not isinstance(outcomes_value, list):
        raise _fail("REQUEST_SEQUENCE_MODEL_INVALID", "sequence outcomes must be an array", line_number)
    try:
        outcomes: list[SequenceOutcome] = []
        for value in outcomes_value:
            item = _require_mapping(
                value,
                code="REQUEST_SEQUENCE_MODEL_INVALID",
                field_name="sequence outcome",
                line_number=line_number,
            )
            _require_exact_fields(
                item,
                _SEQUENCE_OUTCOME_FIELDS,
                code="REQUEST_SEQUENCE_MODEL_INVALID",
                line_number=line_number,
            )
            outcomes.append(
                SequenceOutcome(
                    root_option_id=item["root_option_id"],
                    root_observation=item["root_observation"],
                    second_option_id=item["second_option_id"],
                    scenario_id=item["scenario_id"],
                    terminal_residual_loss=item["terminal_residual_loss"],
                    terminal_closed_failure_atoms=tuple(item["terminal_closed_failure_atoms"]),
                )
            )
        model = SequenceOutcomeModel(payload["model_id"], tuple(outcomes))
    except (KeyError, TypeError, ValueError) as exc:
        raise _typed_failure(
            "REQUEST_SEQUENCE_MODEL_INVALID", "invalid sequence outcome model", line_number, exc
        ) from exc
    if model.to_dict() != dict(payload):
        raise _fail(
            "REQUEST_SEQUENCE_MODEL_INVALID",
            "sequence model is not canonical serialization",
            line_number,
        )
    return model


def _parse_scenario_set(payload: Mapping[str, Any], line_number: int) -> JointScenarioSet:
    _require_exact_fields(
        payload,
        _SCENARIO_SET_FIELDS,
        code="REQUEST_SCENARIO_SET_INVALID",
        line_number=line_number,
    )
    if payload["schema_version"] != "xrd-rb-voe-joint-scenario-set-v1":
        raise _fail(
            "REQUEST_SCENARIO_SET_INVALID",
            "unsupported joint scenario set schema",
            line_number,
        )
    scenarios_value = payload["scenarios"]
    if not isinstance(scenarios_value, list) or not scenarios_value:
        raise _fail(
            "REQUEST_SCENARIO_SET_INVALID",
            "joint scenario set must contain scenarios",
            line_number,
        )
    try:
        scenarios: list[JointScenario] = []
        for value in scenarios_value:
            item = _require_mapping(
                value,
                code="REQUEST_SCENARIO_SET_INVALID",
                field_name="joint scenario",
                line_number=line_number,
            )
            _require_exact_fields(
                item,
                _SCENARIO_FIELDS,
                code="REQUEST_SCENARIO_SET_INVALID",
                line_number=line_number,
            )
            scenarios.append(
                JointScenario(
                    scenario_id=item["scenario_id"],
                    hold_loss=item["hold_loss"],
                    nominal_probability=item["nominal_probability"],
                    robust_probability=item["robust_probability"],
                    failure_domains=tuple(item["failure_domains"]),
                )
            )
        scenario_set = JointScenarioSet(payload["scenario_set_id"], tuple(scenarios))
    except (KeyError, TypeError, ValueError) as exc:
        raise _typed_failure(
            "REQUEST_SCENARIO_SET_INVALID",
            "invalid joint scenario set",
            line_number,
            exc,
        ) from exc
    if scenario_set.to_dict() != dict(payload):
        raise _fail(
            "REQUEST_SCENARIO_SET_INVALID",
            "joint scenario set is not canonical serialization",
            line_number,
        )
    return scenario_set


def _parse_closure_predicate(payload: Mapping[str, Any], line_number: int) -> ClosurePredicate:
    _require_exact_fields(
        payload,
        _CLOSURE_PREDICATE_FIELDS,
        code="REQUEST_CLOSURE_PREDICATE_INVALID",
        line_number=line_number,
    )
    try:
        predicate = ClosurePredicate(
            predicate_id=payload["predicate_id"],
            required_failure_atoms=tuple(payload["required_failure_atoms"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise _typed_failure(
            "REQUEST_CLOSURE_PREDICATE_INVALID",
            "invalid closure predicate",
            line_number,
            exc,
        ) from exc
    if predicate.to_dict() != dict(payload):
        raise _fail(
            "REQUEST_CLOSURE_PREDICATE_INVALID",
            "closure predicate is not canonical serialization",
            line_number,
        )
    return predicate


def _parse_simulated_option(payload: Mapping[str, Any], line_number: int) -> SimulatedOption:
    _require_exact_fields(
        payload, _SIMULATED_OPTION_FIELDS, code="REQUEST_OPTION_INVALID", line_number=line_number
    )
    option_payload = _require_mapping(
        payload["option"], code="REQUEST_OPTION_INVALID", field_name="option", line_number=line_number
    )
    _require_exact_fields(
        option_payload, _OPTION_FIELDS, code="REQUEST_OPTION_INVALID", line_number=line_number
    )
    outcomes_value = option_payload["outcomes"]
    gates_value = option_payload["hard_gates"]
    durations = _require_mapping(
        payload["duration_ms_by_scenario"],
        code="REQUEST_OPTION_INVALID",
        field_name="duration_ms_by_scenario",
        line_number=line_number,
    )
    if not isinstance(outcomes_value, list) or not isinstance(gates_value, list):
        raise _fail("REQUEST_OPTION_INVALID", "outcomes and hard_gates must be arrays", line_number)
    try:
        outcomes: list[OptionOutcome] = []
        for value in outcomes_value:
            item = _require_mapping(
                value,
                code="REQUEST_OPTION_INVALID",
                field_name="option outcome",
                line_number=line_number,
            )
            _require_exact_fields(
                item, _OUTCOME_FIELDS, code="REQUEST_OPTION_INVALID", line_number=line_number
            )
            outcomes.append(
                OptionOutcome(
                    scenario_id=item["scenario_id"],
                    observation=item["observation"],
                    residual_loss=item["residual_loss"],
                    closed_failure_atoms=tuple(item["closed_failure_atoms"]),
                )
            )
        gates: list[HardGate] = []
        for value in gates_value:
            item = _require_mapping(
                value,
                code="REQUEST_OPTION_INVALID",
                field_name="hard gate",
                line_number=line_number,
            )
            _require_exact_fields(
                item, _HARD_GATE_FIELDS, code="REQUEST_OPTION_INVALID", line_number=line_number
            )
            gates.append(HardGate(item["gate_id"], item["passed"], item["failure_code"]))
        option = EvidenceOption(
            option_id=option_payload["option_id"],
            cost=option_payload["cost"],
            outcomes=tuple(outcomes),
            hard_gates=tuple(gates),
            repeatable=option_payload["repeatable"],
        )
        simulated = SimulatedOption(option, dict(durations))
    except (KeyError, TypeError, ValueError) as exc:
        raise _typed_failure("REQUEST_OPTION_INVALID", "invalid simulated option", line_number, exc) from exc
    if simulated.to_dict() != dict(payload):
        raise _fail("REQUEST_OPTION_INVALID", "option is not canonical serialization", line_number)
    return simulated


def _validate_request(
    payload: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    case: Mapping[str, Any],
    supplied_sha256: object,
    line_number: int,
) -> dict[str, Any]:
    _require_exact_fields(payload, _REQUEST_FIELDS, code="REQUEST_PAYLOAD_INVALID", line_number=line_number)
    if payload["schema_version"] != REQUEST_SCHEMA:
        raise _fail("REQUEST_SCHEMA_INVALID", "unsupported simulation request schema", line_number)
    request_sha256 = canonical_sha256(payload)
    if supplied_sha256 != request_sha256:
        raise _fail("REQUEST_HASH_MISMATCH", "simulation_request_sha256 does not match request", line_number)
    episode_id = _require_string(
        payload["episode_id"],
        code="REQUEST_PAYLOAD_INVALID",
        field_name="episode_id",
        line_number=line_number,
    )
    if isinstance(payload["seed"], bool) or not isinstance(payload["seed"], (int, str)):
        raise _fail("REQUEST_PAYLOAD_INVALID", "seed must be an integer or string", line_number)
    horizon = _require_int(
        payload["horizon"],
        code="REQUEST_PAYLOAD_INVALID",
        field_name="horizon",
        line_number=line_number,
        minimum=1,
    )
    if horizon not in {1, 2}:
        raise _fail("REQUEST_PAYLOAD_INVALID", "horizon must be one or two", line_number)
    try:
        variant = ScenarioVariant(payload["scenario_variant"])
    except (TypeError, ValueError) as exc:
        raise _typed_failure("REQUEST_PAYLOAD_INVALID", "invalid scenario variant", line_number, exc) from exc
    scenario_set = _parse_scenario_set(
        _require_mapping(
            payload["scenario_set"],
            code="REQUEST_SCENARIO_SET_INVALID",
            field_name="scenario_set",
            line_number=line_number,
        ),
        line_number,
    )
    closure_predicate = _parse_closure_predicate(
        _require_mapping(
            payload["closure_predicate"],
            code="REQUEST_CLOSURE_PREDICATE_INVALID",
            field_name="closure_predicate",
            line_number=line_number,
        ),
        line_number,
    )
    options_value = payload["options"]
    if not isinstance(options_value, list) or not options_value:
        raise _fail("REQUEST_OPTIONS_INVALID", "request options must be a non-empty array", line_number)
    options = tuple(
        _parse_simulated_option(
            _require_mapping(
                value,
                code="REQUEST_OPTION_INVALID",
                field_name="simulated option",
                line_number=line_number,
            ),
            line_number,
        )
        for value in options_value
    )
    if tuple(sorted(options, key=lambda item: item.option_id)) != options:
        raise _fail("REQUEST_OPTIONS_INVALID", "request options are not canonically ordered", line_number)
    option_ids = tuple(item.option_id for item in options)
    if len(set(option_ids)) != len(option_ids):
        raise _fail("REQUEST_OPTIONS_INVALID", "request option ids are duplicated", line_number)
    if not set(option_ids).issubset(case["allowed_options"]):
        raise _fail("REQUEST_OPTIONS_INVALID", "request contains an option outside CASE", line_number)
    scenario_ids = scenario_set.scenario_ids
    expected_scenario_set = set(scenario_ids)
    if any(
        {outcome.scenario_id for outcome in item.option.outcomes} != expected_scenario_set for item in options
    ):
        raise _fail("REQUEST_SCENARIO_COVERAGE_INVALID", "options do not cover one scenario set", line_number)
    sequence_payload = _require_mapping(
        payload["sequence_model"],
        code="REQUEST_SEQUENCE_MODEL_INVALID",
        field_name="sequence_model",
        line_number=line_number,
    )
    sequence_model = _parse_sequence_model(sequence_payload, line_number)
    sequence_sha256 = sequence_model.content_sha256
    for name in (
        "scenario_set_sha256",
        "closure_predicate_sha256",
        "sequence_model_sha256",
        "policy_plan_sha256",
        "pinned_policy_plan_sha256",
        "pinned_sequence_model_sha256",
        "root_provenance_sha256",
    ):
        _require_hash(payload[name], code="REQUEST_HASH_INVALID", field_name=name, line_number=line_number)
    plan_object: PolicyPlan = plan["object"]
    option_set_sha256 = canonical_sha256([item.option.to_dict() for item in options])
    expected_closure_sha256 = case["closure_predicate"].content_sha256
    bindings = {
        "horizon": (horizon, plan_object.horizon),
        "scenario variant": (variant, plan_object.variant),
        "scenario set content": (scenario_set.content_sha256, payload["scenario_set_sha256"]),
        "scenario set plan": (scenario_set.content_sha256, plan_object.scenario_set_sha256),
        "closure predicate content": (
            closure_predicate.content_sha256,
            payload["closure_predicate_sha256"],
        ),
        "closure predicate CASE": (closure_predicate.content_sha256, expected_closure_sha256),
        "plan closure predicate": (plan_object.closure_predicate_sha256, expected_closure_sha256),
        "option set": (option_set_sha256, plan_object.option_set_sha256),
        "sequence model": (sequence_sha256, plan_object.sequence_model_sha256),
        "request sequence digest": (payload["sequence_model_sha256"], sequence_sha256),
        "pinned sequence digest": (payload["pinned_sequence_model_sha256"], sequence_sha256),
        "policy plan": (payload["policy_plan_sha256"], plan["plan_sha256"]),
        "pinned policy plan": (payload["pinned_policy_plan_sha256"], plan["plan_sha256"]),
        "root provenance": (payload["root_provenance_sha256"], plan_object.root_provenance_sha256),
    }
    for binding_name, (actual, expected) in bindings.items():
        if actual != expected:
            raise _fail(
                "REQUEST_BINDING_MISMATCH",
                f"simulation request {binding_name} drifted from frozen objects",
                line_number,
            )
    if payload["root_provenance_sha256"] != case["object"].evidence_root_sha256:
        raise _fail("REQUEST_BINDING_MISMATCH", "request provenance differs from CASE", line_number)
    veto_steps_value = payload["local_veto_steps"]
    if not isinstance(veto_steps_value, list):
        raise _fail("REQUEST_VETO_STEPS_INVALID", "local_veto_steps must be an array", line_number)
    veto_steps = tuple(
        _require_int(
            step,
            code="REQUEST_VETO_STEPS_INVALID",
            field_name="local_veto_steps",
            line_number=line_number,
            minimum=1,
        )
        for step in veto_steps_value
    )
    if veto_steps != tuple(sorted(set(veto_steps))) or any(step > horizon for step in veto_steps):
        raise _fail("REQUEST_VETO_STEPS_INVALID", "local veto steps are not canonical", line_number)

    option_by_id = {item.option_id: item for item in options}
    expected_rejected = tuple(
        RejectedOption(item.option_id, item.option.failure_codes)
        for item in options
        if not item.option.feasible
    )
    if plan_object.rejected_options != expected_rejected:
        raise _fail(
            "REQUEST_PLAN_REJECTION_MISMATCH",
            "plan rejected options differ from frozen hard gates",
            line_number,
        )
    for terminal in sequence_model.outcomes:
        root_option = option_by_id.get(terminal.root_option_id)
        if (
            root_option is None
            or terminal.second_option_id not in option_by_id
            or terminal.scenario_id not in expected_scenario_set
        ):
            raise _fail(
                "REQUEST_SEQUENCE_COVERAGE_INVALID",
                "sequence model references an option or scenario outside the request",
                line_number,
            )
        if root_option.option.outcome_for(terminal.scenario_id).observation != terminal.root_observation:
            raise _fail(
                "REQUEST_SEQUENCE_COVERAGE_INVALID",
                "sequence model root observation differs from frozen option outcome",
                line_number,
            )
    if plan_object.root_option_id not in option_by_id or any(
        branch.option_id is not None and branch.option_id not in option_by_id
        for branch in plan_object.branches
    ):
        raise _fail("REQUEST_PLAN_OPTION_MISMATCH", "plan references an absent request option", line_number)
    root = option_by_id[plan_object.root_option_id]
    expected_branches: dict[str, tuple[str, ...]] = {}
    if horizon == 2:
        for scenario_id in scenario_ids:
            outcome = root.option.outcome_for(scenario_id)
            if not closure_predicate.is_closed(outcome.closed_failure_atoms):
                expected_branches.setdefault(outcome.observation, ())
                expected_branches[outcome.observation] += (scenario_id,)
    actual_branches = {branch.observation: branch.conditioned_scenario_ids for branch in plan_object.branches}
    if actual_branches != expected_branches:
        raise _fail(
            "REQUEST_PLAN_BRANCH_BINDING_INVALID",
            "plan branches do not equal outcome-conditioned frozen scenarios",
            line_number,
        )
    for branch in plan_object.branches:
        if branch.option_id is None:
            raise _fail("REQUEST_PLAN_BRANCH_BINDING_INVALID", "active branch has no option", line_number)
        for scenario_id in branch.conditioned_scenario_ids:
            try:
                terminal = sequence_model.outcome_for(
                    root_option_id=root.option_id,
                    root_observation=branch.observation,
                    second_option_id=branch.option_id,
                    scenario_id=scenario_id,
                )
            except KeyError as exc:
                raise _typed_failure(
                    "REQUEST_SEQUENCE_COVERAGE_INVALID",
                    "missing sequence terminal outcome",
                    line_number,
                    exc,
                ) from exc
            if not closure_predicate.is_closed(terminal.terminal_closed_failure_atoms):
                raise _fail(
                    "REQUEST_SEQUENCE_COVERAGE_INVALID",
                    "planned sequence does not close the frozen failure core",
                    line_number,
                )
    try:
        sealed_request = SimulationRequest(
            episode_id=episode_id,
            seed=payload["seed"],
            horizon=horizon,
            scenario_set=scenario_set,
            scenario_variant=variant,
            options=options,
            closure_predicate=closure_predicate,
            sequence_model=sequence_model,
            policy_plan=plan_object,
            pinned_policy_plan_sha256=payload["pinned_policy_plan_sha256"],
            pinned_sequence_model_sha256=payload["pinned_sequence_model_sha256"],
            root_provenance_sha256=payload["root_provenance_sha256"],
            local_veto_steps=veto_steps,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise _typed_failure(
            "REQUEST_POLICY_REPLAY_INVALID",
            "request does not reproduce its bound policy semantics",
            line_number,
            exc,
        ) from exc
    if sealed_request.to_dict() != dict(payload):
        raise _fail(
            "REQUEST_PAYLOAD_INVALID",
            "request is not exact SimulationRequest.to_dict output",
            line_number,
        )
    return {
        "payload": dict(payload),
        "request_sha256": request_sha256,
        "episode_id": episode_id,
        "seed": payload["seed"],
        "horizon": horizon,
        "variant": variant,
        "options": options,
        "option_by_id": option_by_id,
        "option_set_sha256": option_set_sha256,
        "scenario_ids": scenario_ids,
        "scenario_set": scenario_set,
        "scenario_set_sha256": scenario_set.content_sha256,
        "closure_predicate": closure_predicate,
        "sequence_model": sequence_model,
        "sequence_model_sha256": sequence_sha256,
        "root_provenance_sha256": payload["root_provenance_sha256"],
        "local_veto_steps": veto_steps,
    }


def _expected_episode_trace(
    *, request: Mapping[str, Any], plan: PolicyPlan, scenario_id: str
) -> tuple[list[dict[str, object]], str, tuple[str, ...], bool, int]:
    closed: set[str] = set()
    executed: set[str] = set()
    observations: list[dict[str, object]] = []
    previous_observation: str | None = None
    duration_total = 0
    termination_reason = "HORIZON_EXHAUSTED"
    root_option_id = plan.root_option_id
    if root_option_id is None:
        raise ValueError("NEXT_EVIDENCE plan has no root option")
    for step_index in range(1, request["horizon"] + 1):
        if step_index in request["local_veto_steps"]:
            termination_reason = "LOCAL_VETO_REJECTED"
            break
        option_id = plan.next_option(
            observation=previous_observation if step_index > 1 else None,
            provenance_sha256=request["root_provenance_sha256"],
        )
        if option_id is None:
            termination_reason = "PLAN_BRANCH_MISSING"
            break
        selected: SimulatedOption = request["option_by_id"][option_id]
        if not selected.option.feasible:
            termination_reason = "INFEASIBLE_OPTION_REJECTED"
            break
        if option_id in executed and not selected.option.repeatable:
            termination_reason = "NON_REPEATABLE_OPTION_REJECTED"
            break
        outcome = selected.option.outcome_for(scenario_id)
        duration_ms = selected.duration_ms_by_scenario[scenario_id]
        executed.add(option_id)
        sequence_outcome_sha256: str | None = None
        if step_index == 1:
            residual_loss = outcome.residual_loss
            closed.update(outcome.closed_failure_atoms)
            closed_atoms = outcome.closed_failure_atoms
        else:
            if previous_observation is None:
                raise ValueError("step two has no root observation")
            terminal = request["sequence_model"].outcome_for(
                root_option_id=root_option_id,
                root_observation=previous_observation,
                second_option_id=option_id,
                scenario_id=scenario_id,
            )
            residual_loss = terminal.terminal_residual_loss
            closed = set(terminal.terminal_closed_failure_atoms)
            closed_atoms = terminal.terminal_closed_failure_atoms
            sequence_outcome_sha256 = terminal.content_sha256
        duration_total += duration_ms
        modeled_closed = request["closure_predicate"].is_closed(closed)
        observations.append(
            SimulatedObservation(
                step_index=step_index,
                option_id=option_id,
                root_scenario_id=scenario_id,
                duration_ms=duration_ms,
                observation=outcome.observation,
                residual_loss=residual_loss,
                closed_failure_atoms=closed_atoms,
                cumulative_modeled_closed_failure_atoms=tuple(sorted(closed)),
                modeled_closure_satisfied=modeled_closed,
                sequence_outcome_sha256=sequence_outcome_sha256,
            ).to_dict()
        )
        previous_observation = outcome.observation
        if modeled_closed:
            termination_reason = "MODELED_CLOSURE_SATISFIED"
            break
    closed_atoms = tuple(sorted(closed))
    return (
        observations,
        termination_reason,
        closed_atoms,
        request["closure_predicate"].is_closed(closed),
        duration_total,
    )


def _parse_episode_object(payload: Mapping[str, Any], line_number: int) -> EpisodeResult:
    _require_exact_fields(payload, _EPISODE_FIELDS, code="EPISODE_PAYLOAD_INVALID", line_number=line_number)
    if payload["schema_version"] != EPISODE_SCHEMA:
        raise _fail("EPISODE_SCHEMA_INVALID", "unsupported simulated episode schema", line_number)
    observations_value = payload["observations"]
    if not isinstance(observations_value, list):
        raise _fail("EPISODE_OBSERVATIONS_INVALID", "observations must be an array", line_number)
    try:
        observations: list[SimulatedObservation] = []
        for value in observations_value:
            item = _require_mapping(
                value,
                code="EPISODE_STEP_INVALID",
                field_name="observation step",
                line_number=line_number,
            )
            _require_exact_fields(item, _STEP_FIELDS, code="EPISODE_STEP_INVALID", line_number=line_number)
            observations.append(
                SimulatedObservation(
                    step_index=item["step_index"],
                    option_id=item["option_id"],
                    root_scenario_id=item["root_scenario_id"],
                    duration_ms=item["duration_ms"],
                    observation=item["observation"],
                    residual_loss=item["residual_loss"],
                    closed_failure_atoms=tuple(item["closed_failure_atoms"]),
                    cumulative_modeled_closed_failure_atoms=tuple(
                        item["cumulative_modeled_closed_failure_atoms"]
                    ),
                    modeled_closure_satisfied=item["modeled_closure_satisfied"],
                    sequence_outcome_sha256=item["sequence_outcome_sha256"],
                    provenance=EvidenceSource(item["provenance"]),
                    hardware_touch=item["hardware_touch"],
                    physical_closure_proven=item["physical_closure_proven"],
                )
            )
        episode = EpisodeResult(
            episode_id=payload["episode_id"],
            request_sha256=payload["request_sha256"],
            policy_plan_sha256=payload["policy_plan_sha256"],
            sequence_model_sha256=payload["sequence_model_sha256"],
            root_provenance_sha256=payload["root_provenance_sha256"],
            scenario_set_sha256=payload["scenario_set_sha256"],
            root_scenario_id=payload["root_scenario_id"],
            root_scenario_draw_sha256=payload["root_scenario_draw_sha256"],
            root_scenario_selection_count=payload["root_scenario_selection_count"],
            observations=tuple(observations),
            cumulative_duration_ms=payload["cumulative_duration_ms"],
            modeled_closed_failure_atoms=tuple(payload["modeled_closed_failure_atoms"]),
            modeled_closure_satisfied=payload["modeled_closure_satisfied"],
            termination_reason=payload["termination_reason"],
            evidence_source=EvidenceSource(payload["evidence_source"]),
            hardware_touch=payload["hardware_touch"],
            execution_authority=payload["execution_authority"],
            physical_closure_proven=payload["physical_closure_proven"],
            physical_risk_denominator_increment=payload["physical_risk_denominator_increment"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise _typed_failure("EPISODE_PAYLOAD_INVALID", "invalid EpisodeResult", line_number, exc) from exc
    if episode.to_dict() != dict(payload):
        raise _fail(
            "EPISODE_PAYLOAD_INVALID", "episode is not exact EpisodeResult.to_dict output", line_number
        )
    return episode


def _validate_episode(
    payload: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    plan: Mapping[str, Any],
    line_number: int,
    exhaustive: bool,
) -> dict[str, Any]:
    episode = _parse_episode_object(payload, line_number)
    plan_object: PolicyPlan = plan["object"]
    expected_bindings = {
        "episode_id": (episode.episode_id, request["episode_id"]),
        "request_sha256": (episode.request_sha256, request["request_sha256"]),
        "policy_plan_sha256": (episode.policy_plan_sha256, plan["plan_sha256"]),
        "sequence_model_sha256": (
            episode.sequence_model_sha256,
            request["sequence_model_sha256"],
        ),
        "root_provenance_sha256": (
            episode.root_provenance_sha256,
            request["root_provenance_sha256"],
        ),
        "scenario_set_sha256": (episode.scenario_set_sha256, request["scenario_set_sha256"]),
    }
    for binding_name, (actual, expected) in expected_bindings.items():
        if actual != expected:
            raise _fail(
                "EPISODE_BINDING_MISMATCH",
                f"episode {binding_name} does not bind frozen request/plan",
                line_number,
            )
    if episode.root_scenario_id not in request["scenario_ids"]:
        raise _fail("EPISODE_SCENARIO_INVALID", "episode scenario is not frozen", line_number)
    if exhaustive:
        expected_draw = canonical_sha256(
            {
                "domain": "xrd-rb-voe-exhaustive-root-v1",
                "request_sha256": request["request_sha256"],
                "scenario_id": episode.root_scenario_id,
            }
        )
    else:
        expected_draw = canonical_sha256(
            {
                "domain": "xrd-rb-voe-root-scenario-draw-v3-common-random-number",
                "seed": request["seed"],
                "scenario_set_sha256": request["scenario_set_sha256"],
                "variant": request["variant"].value,
            }
        )
    if episode.root_scenario_draw_sha256 != expected_draw:
        raise _fail("EPISODE_DRAW_HASH_MISMATCH", "episode root draw is not reproducible", line_number)
    if not exhaustive:
        draw = int(expected_draw, 16) / (1 << 256)
        cumulative = 0.0
        expected_scenario_id = request["scenario_set"].scenarios[-1].scenario_id
        for scenario in request["scenario_set"].scenarios:
            cumulative += scenario.probability(request["variant"])
            if draw < cumulative:
                expected_scenario_id = scenario.scenario_id
                break
        if episode.root_scenario_id != expected_scenario_id:
            raise _fail(
                "EPISODE_DRAW_SCENARIO_MISMATCH",
                "episode root scenario does not match the frozen probability draw",
                line_number,
            )
    request_with_closure = dict(request)
    request_with_closure["closure_predicate"] = request["closure_predicate"]
    try:
        expected_steps, termination, closed_atoms, closed, duration = _expected_episode_trace(
            request=request_with_closure,
            plan=plan_object,
            scenario_id=episode.root_scenario_id,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise _typed_failure(
            "EPISODE_REPLAY_INVALID", "cannot replay frozen episode", line_number, exc
        ) from exc
    if [item.to_dict() for item in episode.observations] != expected_steps:
        raise _fail(
            "EPISODE_OUTCOME_REPLAY_MISMATCH",
            "episode steps differ from frozen option/sequence outcomes",
            line_number,
        )
    if (
        episode.termination_reason != termination
        or episode.modeled_closed_failure_atoms != closed_atoms
        or episode.modeled_closure_satisfied is not closed
        or episode.cumulative_duration_ms != duration
    ):
        raise _fail(
            "EPISODE_SUMMARY_REPLAY_MISMATCH",
            "episode summary differs from recomputed trace",
            line_number,
        )
    selected_option = episode.observations[-1].option_id if episode.observations else None
    return {
        "object": episode,
        "episode_id": episode.episode_id,
        "episode_sha256": episode.episode_sha256,
        "selected_option": selected_option,
        "modeled_closure_satisfied": episode.modeled_closure_satisfied,
        "hardware_touched": episode.hardware_touch,
        "execution_authority": episode.execution_authority,
    }


def _validate_observation_event(
    payload: Mapping[str, Any],
    *,
    previous_row: Mapping[str, Any],
    case: Mapping[str, Any],
    plan: Mapping[str, Any],
    line_number: int,
) -> dict[str, Any]:
    _require_exact_fields(
        payload,
        _OBSERVATION_EVENT_FIELDS,
        code="OBSERVATION_EVENT_PAYLOAD_INVALID",
        line_number=line_number,
    )
    if payload["schema_version"] != OBSERVATION_EVENT_SCHEMA:
        raise _fail("OBSERVATION_EVENT_SCHEMA_INVALID", "unsupported observation event schema", line_number)
    if payload["case_id"] != case["case_id"] or payload["experiment_case_sha256"] != case["case_sha256"]:
        raise _fail("OBSERVATION_CASE_HASH_MISMATCH", "observation does not bind CASE", line_number)
    if payload["plan_sha256"] != plan["plan_sha256"]:
        raise _fail("OBSERVATION_PLAN_HASH_MISMATCH", "observation does not bind plan", line_number)
    if payload["previous_record_sha256"] != previous_row["record_sha256"]:
        raise _fail(
            "SEMANTIC_PREVIOUS_HASH_MISMATCH", "observation does not bind previous record", line_number
        )
    request_payload = _require_mapping(
        payload["simulation_request"],
        code="REQUEST_PAYLOAD_INVALID",
        field_name="simulation_request",
        line_number=line_number,
    )
    request = _validate_request(
        request_payload,
        plan=plan,
        case=case,
        supplied_sha256=payload["simulation_request_sha256"],
        line_number=line_number,
    )
    episode_payload = _require_mapping(
        payload["episode"], code="EPISODE_PAYLOAD_INVALID", field_name="episode", line_number=line_number
    )
    episode = _validate_episode(
        episode_payload,
        request=request,
        plan=plan,
        line_number=line_number,
        exhaustive=False,
    )
    if payload["episode_sha256"] != episode["episode_sha256"]:
        raise _fail("EPISODE_HASH_MISMATCH", "episode_sha256 does not match episode", line_number)
    return {"request": request, **episode}


def _validate_exhaustive(
    payload: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    plan: Mapping[str, Any],
    supplied_sha256: object,
    line_number: int,
) -> dict[str, Any]:
    _require_exact_fields(
        payload, _EXHAUSTIVE_FIELDS, code="EXHAUSTIVE_PAYLOAD_INVALID", line_number=line_number
    )
    if payload["schema_version"] != EXHAUSTIVE_SCHEMA:
        raise _fail("EXHAUSTIVE_SCHEMA_INVALID", "unsupported exhaustive replay schema", line_number)
    for name, expected in (
        ("request_sha256", request["request_sha256"]),
        ("policy_plan_sha256", plan["plan_sha256"]),
        ("sequence_model_sha256", request["sequence_model_sha256"]),
        ("root_provenance_sha256", request["root_provenance_sha256"]),
        ("scenario_set_sha256", request["scenario_set_sha256"]),
    ):
        if payload[name] != expected:
            raise _fail(
                "EXHAUSTIVE_BINDING_MISMATCH",
                f"exhaustive replay {name} drifted from request/plan",
                line_number,
            )
    expected_ids = _require_string_list(
        payload["expected_scenario_ids"],
        code="EXHAUSTIVE_SCENARIO_COVERAGE_INVALID",
        field_name="expected_scenario_ids",
        line_number=line_number,
        allow_empty=False,
    )
    scenario_ids = _require_string_list(
        payload["scenario_ids"],
        code="EXHAUSTIVE_SCENARIO_COVERAGE_INVALID",
        field_name="scenario_ids",
        line_number=line_number,
        allow_empty=False,
    )
    if expected_ids != request["scenario_ids"] or scenario_ids != expected_ids:
        raise _fail(
            "EXHAUSTIVE_SCENARIO_COVERAGE_INVALID",
            "exhaustive replay must cover each frozen scenario exactly once and in order",
            line_number,
        )
    episodes_value = payload["episodes"]
    if not isinstance(episodes_value, list) or len(episodes_value) != len(expected_ids):
        raise _fail("EXHAUSTIVE_EPISODES_INVALID", "exhaustive episodes are incomplete", line_number)
    episodes: list[EpisodeResult] = []
    episode_hashes: list[str] = []
    for scenario_id, episode_value in zip(expected_ids, episodes_value, strict=True):
        episode_payload = _require_mapping(
            episode_value,
            code="EXHAUSTIVE_EPISODES_INVALID",
            field_name="exhaustive episode",
            line_number=line_number,
        )
        validated = _validate_episode(
            episode_payload,
            request=request,
            plan=plan,
            line_number=line_number,
            exhaustive=True,
        )
        episode: EpisodeResult = validated["object"]
        if episode.root_scenario_id != scenario_id:
            raise _fail(
                "EXHAUSTIVE_SCENARIO_COVERAGE_INVALID",
                "episode scenario does not match frozen roster position",
                line_number,
            )
        episodes.append(episode)
        episode_hashes.append(episode.episode_sha256)
    supplied_hashes = _require_string_list(
        payload["episode_sha256s"],
        code="EXHAUSTIVE_EPISODE_HASH_MISMATCH",
        field_name="episode_sha256s",
        line_number=line_number,
        allow_empty=False,
    )
    if supplied_hashes != tuple(episode_hashes):
        raise _fail("EXHAUSTIVE_EPISODE_HASH_MISMATCH", "exhaustive episode hashes drifted", line_number)
    try:
        replay = ExhaustiveReplayResult(
            request_sha256=request["request_sha256"],
            policy_plan_sha256=plan["plan_sha256"],
            sequence_model_sha256=request["sequence_model_sha256"],
            root_provenance_sha256=request["root_provenance_sha256"],
            scenario_set_sha256=request["scenario_set_sha256"],
            expected_scenario_ids=expected_ids,
            episodes=tuple(episodes),
        )
    except (TypeError, ValueError) as exc:
        raise _typed_failure(
            "EXHAUSTIVE_PAYLOAD_INVALID", "invalid exhaustive replay", line_number, exc
        ) from exc
    if replay.to_dict() != dict(payload):
        raise _fail(
            "EXHAUSTIVE_SUMMARY_REPLAY_MISMATCH",
            "exhaustive summary differs from recomputed episodes",
            line_number,
        )
    replay_sha256 = replay.replay_sha256
    if supplied_sha256 != replay_sha256:
        raise _fail("EXHAUSTIVE_HASH_MISMATCH", "exhaustive replay digest mismatch", line_number)
    return {"object": replay, "replay_sha256": replay_sha256}


def _validate_terminal(
    payload: Mapping[str, Any],
    *,
    previous_row: Mapping[str, Any],
    case: Mapping[str, Any],
    plan: Mapping[str, Any],
    episodes: tuple[Mapping[str, Any], ...],
    request: Mapping[str, Any],
    line_number: int,
) -> dict[str, Any]:
    _require_exact_fields(payload, _TERMINAL_FIELDS, code="TERMINAL_PAYLOAD_INVALID", line_number=line_number)
    if payload["schema_version"] != TERMINAL_EVENT_SCHEMA:
        raise _fail("TERMINAL_SCHEMA_INVALID", "unsupported terminal event schema", line_number)
    if payload["case_id"] != case["case_id"] or payload["experiment_case_sha256"] != case["case_sha256"]:
        raise _fail("TERMINAL_CASE_HASH_MISMATCH", "terminal does not bind CASE", line_number)
    if payload["plan_sha256"] != plan["plan_sha256"]:
        raise _fail("TERMINAL_PLAN_HASH_MISMATCH", "terminal does not bind plan", line_number)
    if payload["simulation_request_sha256"] != request["request_sha256"]:
        raise _fail("TERMINAL_REQUEST_HASH_MISMATCH", "terminal does not bind request", line_number)
    if payload["previous_record_sha256"] != previous_row["record_sha256"]:
        raise _fail("SEMANTIC_PREVIOUS_HASH_MISMATCH", "terminal does not bind previous record", line_number)
    episode_hashes = tuple(item["episode_sha256"] for item in episodes)
    supplied_episode_hashes = _require_string_list(
        payload["episode_sha256"],
        code="TERMINAL_EPISODES_INVALID",
        field_name="episode_sha256",
        line_number=line_number,
        allow_empty=False,
    )
    if supplied_episode_hashes != episode_hashes:
        raise _fail(
            "TERMINAL_EPISODE_LINK_MISMATCH", "terminal observation episode links drifted", line_number
        )
    exhaustive_payload = _require_mapping(
        payload["exhaustive_replay"],
        code="EXHAUSTIVE_PAYLOAD_INVALID",
        field_name="exhaustive_replay",
        line_number=line_number,
    )
    exhaustive = _validate_exhaustive(
        exhaustive_payload,
        request=request,
        plan=plan,
        supplied_sha256=payload["exhaustive_replay_sha256"],
        line_number=line_number,
    )
    receipt = _require_mapping(
        payload["decision_receipt"],
        code="DECISION_RECEIPT_INVALID",
        field_name="decision_receipt",
        line_number=line_number,
    )
    _require_exact_fields(receipt, _RECEIPT_FIELDS, code="DECISION_RECEIPT_INVALID", line_number=line_number)
    if receipt["schema_version"] != RECEIPT_SCHEMA:
        raise _fail("DECISION_RECEIPT_SCHEMA_INVALID", "unsupported receipt schema", line_number)
    if receipt["case_id"] != case["case_id"]:
        raise _fail("RECEIPT_CASE_MISMATCH", "receipt case_id mismatch", line_number)
    if receipt["previous_record_sha256"] != previous_row["record_sha256"]:
        raise _fail("RECEIPT_PREVIOUS_HASH_MISMATCH", "receipt does not bind final observation", line_number)
    _require_string(
        receipt["receipt_id"],
        code="DECISION_RECEIPT_INVALID",
        field_name="receipt_id",
        line_number=line_number,
    )
    _require_int(
        receipt["plan_epoch"],
        code="DECISION_RECEIPT_INVALID",
        field_name="plan_epoch",
        line_number=line_number,
    )
    failure_core_closed = _require_bool(
        receipt["failure_core_closed"],
        code="RECEIPT_CLOSURE_INVALID",
        field_name="failure_core_closed",
        line_number=line_number,
    )
    release_certificate = receipt["release_certificate_sha256"]
    if release_certificate is not None:
        _require_hash(
            release_certificate,
            code="DECISION_RECEIPT_INVALID",
            field_name="release_certificate_sha256",
            line_number=line_number,
        )
    evidence_hashes = _require_string_list(
        receipt["terminal_evidence_sha256"],
        code="RECEIPT_EVIDENCE_LINK_MISMATCH",
        field_name="terminal_evidence_sha256",
        line_number=line_number,
        allow_empty=False,
    )
    expected_evidence_hashes = (*episode_hashes, exhaustive["replay_sha256"])
    if evidence_hashes != expected_evidence_hashes:
        raise _fail(
            "RECEIPT_EVIDENCE_LINK_MISMATCH",
            "receipt must bind observation episodes and exhaustive replay",
            line_number,
        )
    selected_option = episodes[-1]["selected_option"]
    if receipt["selected_option"] != selected_option:
        raise _fail("RECEIPT_OPTION_LINK_MISMATCH", "receipt selected option drifted", line_number)
    exhaustive_object: ExhaustiveReplayResult = exhaustive["object"]
    expected_closure = (
        all(item["modeled_closure_satisfied"] for item in episodes)
        and exhaustive_object.all_modeled_closure_satisfied
    )
    if failure_core_closed is not expected_closure:
        raise _fail("RECEIPT_CLOSURE_INVALID", "receipt closure differs from full replay", line_number)
    decision = _require_string(
        receipt["decision"], code="RECEIPT_DECISION_INVALID", field_name="decision", line_number=line_number
    )
    if decision not in {"GO", "REVISE", "DROP", "HOLD", "QUARANTINE"}:
        raise _fail("RECEIPT_DECISION_INVALID", "receipt decision is not terminal", line_number)
    if decision in {"GO", "REVISE", "DROP"} and not expected_closure:
        raise _fail("RECEIPT_CLOSURE_INVALID", "material decision requires exhaustive closure", line_number)
    receipt_sha256 = canonical_sha256(receipt)
    if payload["decision_receipt_sha256"] != receipt_sha256:
        raise _fail("DECISION_RECEIPT_HASH_MISMATCH", "receipt digest mismatch", line_number)
    exact_authority = {
        "simulated_only": True,
        "hardware_touched": False,
        "network_touched": False,
        "execution_authority": False,
        "physical_risk_denominator_increment": 0,
    }
    for field_name, expected in exact_authority.items():
        actual = payload[field_name]
        type_ok = (
            isinstance(actual, bool)
            if isinstance(expected, bool)
            else isinstance(actual, int) and not isinstance(actual, bool)
        )
        if not type_ok or actual != expected:
            raise _fail(
                "SIMULATION_AUTHORITY_CONTRADICTION",
                f"terminal {field_name} must be {expected!r}",
                line_number,
            )
    ended_at_ms = _require_int(
        receipt["created_at_ms"],
        code="RECEIPT_TIME_INVALID",
        field_name="created_at_ms",
        line_number=line_number,
    )
    return {
        "decision_receipt_sha256": receipt_sha256,
        "decision": decision,
        "selected_option": selected_option,
        "ended_at_ms": ended_at_ms,
        "exhaustive_replay_sha256": exhaustive["replay_sha256"],
        **exact_authority,
    }


def verify_r1_semantic_ledger(ledger_path: str | Path) -> SemanticLedgerReport:
    """Verify CASE -> PLAN -> REQUEST+EPISODE(s) -> TERMINAL+EXHAUSTIVE."""
    path = Path(ledger_path).expanduser().resolve()
    base_report, rows, _ = _verified_state(path)
    record_types = tuple(row["record_type"] for row in rows)
    allowed = {CASE_EVENT, PLAN_EVENT, OBSERVATION_EVENT, TERMINAL_EVENT}
    unknown = [name for name in record_types if name not in allowed]
    if unknown:
        raise _fail(
            "SEMANTIC_RECORD_TYPE_FORBIDDEN",
            f"strict R1 ledger contains unsupported record types: {sorted(set(unknown))}",
            record_types.index(unknown[0]) + 1,
        )
    if len(rows) < 4:
        raise _fail(
            "SEMANTIC_SEQUENCE_INCOMPLETE",
            "R1 ledger requires CASE, PLAN, observation, and terminal records",
            max(len(rows), 1),
        )
    if record_types[:2] != (CASE_EVENT, PLAN_EVENT) or record_types[-1] != TERMINAL_EVENT:
        raise _fail("SEMANTIC_SEQUENCE_INVALID", "strict R1 ledger order is invalid", 1)
    if any(name != OBSERVATION_EVENT for name in record_types[2:-1]):
        raise _fail("SEMANTIC_SEQUENCE_INVALID", "middle records must be observations", 3)
    case = _validate_case(rows[0]["payload"], 1)
    plan = _validate_plan(rows[1]["payload"], row=rows[1], previous_row=rows[0], case=case, line_number=2)
    episodes_list: list[dict[str, Any]] = []
    episode_hashes: set[str] = set()
    request: dict[str, Any] | None = None
    for index, row in enumerate(rows[2:-1], start=3):
        episode = _validate_observation_event(
            row["payload"],
            previous_row=rows[index - 2],
            case=case,
            plan=plan,
            line_number=index,
        )
        if episode["episode_sha256"] in episode_hashes:
            raise _fail("DUPLICATE_EPISODE", "observation episode digest was reused", index)
        episode_hashes.add(episode["episode_sha256"])
        if request is None:
            request = episode["request"]
        elif (
            episode["request"]["request_sha256"] != request["request_sha256"]
            or episode["request"]["payload"] != request["payload"]
        ):
            raise _fail(
                "OBSERVATION_REQUEST_DRIFT",
                "all observation events must bind one identical frozen request",
                index,
            )
        episodes_list.append(episode)
    if request is None:
        raise _fail("SEMANTIC_SEQUENCE_INCOMPLETE", "no observation request was recorded", 3)
    episodes = tuple(episodes_list)
    terminal = _validate_terminal(
        rows[-1]["payload"],
        previous_row=rows[-2],
        case=case,
        plan=plan,
        episodes=episodes,
        request=request,
        line_number=len(rows),
    )
    if terminal["ended_at_ms"] < case["created_at_ms"]:
        raise _fail("SEMANTIC_TIME_REVERSED", "terminal receipt predates CASE", len(rows))
    return SemanticLedgerReport(
        schema_version="xrd-rb-voe-semantic-ledger-report-v3",
        record_count=base_report["record_count"],
        ledger_file_sha256=base_report["ledger_file_sha256"],
        terminal_record_sha256=base_report["terminal_record_sha256"],
        anchor_sha256=base_report["anchor_sha256"],
        case_id=case["case_id"],
        experiment_case_sha256=case["case_sha256"],
        policy_plan_sha256=plan["plan_sha256"],
        simulation_request_sha256=request["request_sha256"],
        exhaustive_replay_sha256=terminal["exhaustive_replay_sha256"],
        scenario_set_sha256=request["scenario_set_sha256"],
        option_set_sha256=request["option_set_sha256"],
        sequence_model_sha256=request["sequence_model_sha256"],
        evidence_admission_sha256=plan["object"].evidence_admission_sha256,
        evidence_invariant_report_sha256=plan["object"].evidence_invariant_report_sha256,
        counterexample_registry_sha256=plan["object"].counterexample_registry_sha256,
        counterexample_search_sha256=plan["object"].counterexample_search_sha256,
        counterexample_baseline_state_sha256=(plan["object"].counterexample_baseline_state_sha256),
        evaluator_search_contract_release_sha256=(plan["object"].evaluator_search_contract_release_sha256),
        failure_domain_binding_sha256=plan["object"].failure_domain_binding_sha256,
        episode_sha256=tuple(item["episode_sha256"] for item in episodes),
        decision_receipt_sha256=terminal["decision_receipt_sha256"],
        decision=terminal["decision"],
        selected_option=terminal["selected_option"],
        started_at_ms=case["created_at_ms"],
        ended_at_ms=terminal["ended_at_ms"],
        simulated_only=True,
        hardware_touched=False,
        network_touched=False,
        execution_authority=False,
        physical_risk_denominator_increment=0,
        _marker=_SEMANTIC_MARKER,
    )
