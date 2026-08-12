"""Deterministic, no-hardware end-to-end demonstration for RB-VoE R1."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from rb_voe.audit import (
    append_record,
    initialize_ledger,
    verify_ledger,
    verify_r1_semantic_ledger,
)
from rb_voe.contracts.canonical import (
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    is_sha256,
    to_primitive,
)
from rb_voe.contracts.models import (
    Decision,
    DecisionReceipt,
    EvidenceRecord,
    EvidenceSource,
    ExperimentCase,
    Maturity,
)
from rb_voe.contracts.registries import (
    AUTHORITY_DOMAINS,
    AUTHORITY_KEY_DOMAINS,
    FAILURE_CORE_REASON_CODES,
    KEY_DOMAINS,
    MACRO_IDS,
    OPTION_IDS,
    PHYSICAL_EVIDENCE_STATUSES,
    REASON_CODES,
    ROLE_BINDINGS,
    ROUTE_IDS,
    STATION_IDS,
    ZONE_IDS,
)
from rb_voe.core.counterexample import (
    DecisionAssessment,
    PerturbationRegistry,
    RegisteredPerturbation,
    StatePatch,
    declarative_flag_evaluator_contract,
    evaluate_declarative_decision,
    search_registered_perturbations,
)
from rb_voe.core.evidence_dag import EvidenceDAG
from rb_voe.core.options import (
    ClosurePredicate,
    EvidenceOption,
    HardGate,
    OptionOutcome,
    SequenceOutcome,
    SequenceOutcomeModel,
)
from rb_voe.core.policy import (
    FixedBranch,
    PolicyConfig,
    PolicyEvidenceAdmission,
    PolicyMode,
    PolicyPlan,
    canonical_policy_search_state,
    plan_policy,
)
from rb_voe.core.scenarios import JointScenario, JointScenarioSet, ScenarioVariant
from rb_voe.release import (
    build_release_manifest,
    build_release_root_pin,
    build_terminal_manifest,
    verify_release_bundle_artifacts,
)
from rb_voe.sim import (
    SimulatedOption,
    SimulationRequest,
    replay_all_scenarios,
    run_episode,
)

DEMO_SCHEMA = "xrd-rb-voe-r1-demo-v2"
COMPARISON_SCHEMA = "xrd-rb-voe-strategy-comparison-v1"
EXPECTED_SCHEMA = "xrd-rb-voe-r1-demo-expected-v1"
_EXPECTED_FIXTURE_PATH = Path(__file__).with_name("fixtures") / "demo_expected_v1.json"
_DEMO_FAILURE_ATOMS = (
    "CORRELATED_EVIDENCE_DOUBLE_COUNTED",
    "XRD_PEAK_ALIASING",
)


def _file_inventory(root: Path, pattern: str) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob(pattern))
        if path.is_file()
    }


def _write_canonical_exclusive(path: Path, payload: Any) -> None:
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(to_primitive(payload)) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


class DemoVerificationError(ValueError):
    """Raised when the sealed demonstration differs from its golden vector."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def _build_problem(
    required_failure_atoms: tuple[str, ...] = _DEMO_FAILURE_ATOMS,
) -> tuple[
    JointScenarioSet,
    tuple[EvidenceOption, ...],
    SequenceOutcomeModel,
    ClosurePredicate,
    str,
    ExperimentCase,
    PolicyEvidenceAdmission,
]:
    """Build a synthetic blind-sample failure core with correlated root states."""
    normalized_failure_atoms = tuple(sorted(set(required_failure_atoms)))
    if (
        not normalized_failure_atoms
        or normalized_failure_atoms != required_failure_atoms
        or not set(normalized_failure_atoms).issubset(_DEMO_FAILURE_ATOMS)
    ):
        raise ValueError("demo failure atoms must be a non-empty sorted subset of the fixture registry")
    core_suffix = (
        ""
        if normalized_failure_atoms == _DEMO_FAILURE_ATOMS
        else f"-core-{canonical_sha256(normalized_failure_atoms)[:12]}"
    )
    scenario_set = JointScenarioSet(
        scenario_set_id="blind-sample-shared-failure-domain-v1",
        scenarios=(
            JointScenario(
                scenario_id="joint_emission_alias",
                hold_loss=12.0,
                nominal_probability=0.30,
                robust_probability=0.20,
                failure_domains=("sample_lineage", "spectrometer_calibration"),
            ),
            JointScenario(
                scenario_id="joint_phase_mixture",
                hold_loss=14.0,
                nominal_probability=0.40,
                robust_probability=0.30,
                failure_domains=("diffractometer_calibration", "sample_lineage"),
            ),
            JointScenario(
                scenario_id="joint_shared_reference_drift",
                hold_loss=18.0,
                nominal_probability=0.30,
                robust_probability=0.50,
                failure_domains=(
                    "diffractometer_calibration",
                    "sample_lineage",
                    "spectrometer_calibration",
                ),
            ),
        ),
    )
    predicate = ClosurePredicate(
        predicate_id=f"blind-sample-phase-and-emission-closed-v1{core_suffix}",
        required_failure_atoms=normalized_failure_atoms,
    )
    scenarios = scenario_set.scenario_ids

    def outcome(
        scenario_id: str,
        observation: str,
        residual_loss: float,
        *closed: str,
    ) -> OptionOutcome:
        return OptionOutcome(scenario_id, observation, residual_loss, tuple(closed))

    options = (
        EvidenceOption(
            option_id="E_VERIFY_IDENTITY",
            cost=1.0,
            outcomes=(
                outcome(
                    "joint_emission_alias",
                    "emission_origin_unresolved",
                    4.0,
                    "XRD_PEAK_ALIASING",
                ),
                outcome(
                    "joint_phase_mixture",
                    "phase_identity_unresolved",
                    4.5,
                    "CORRELATED_EVIDENCE_DOUBLE_COUNTED",
                ),
                outcome(
                    "joint_shared_reference_drift",
                    "shared_reference_drift",
                    7.0,
                ),
            ),
            hard_gates=(HardGate("offline_fixture_only", True),),
        ),
        EvidenceOption(
            option_id="E_REPREP_XRD",
            cost=2.4,
            outcomes=tuple(
                outcome(
                    scenario_id,
                    "paired_reference_complete",
                    0.6 if scenario_id == "joint_shared_reference_drift" else 9.0,
                    *(
                        (
                            "CORRELATED_EVIDENCE_DOUBLE_COUNTED",
                            "XRD_PEAK_ALIASING",
                        )
                        if scenario_id == "joint_shared_reference_drift"
                        else ()
                    ),
                )
                for scenario_id in scenarios
            ),
            hard_gates=(HardGate("offline_fixture_only", True),),
        ),
        EvidenceOption(
            option_id="E_PL_CROSSCHECK",
            cost=1.2,
            outcomes=tuple(
                outcome(
                    scenario_id,
                    "pl_lifetime_complete",
                    0.4 if scenario_id == "joint_emission_alias" else 8.0,
                    *(
                        ("CORRELATED_EVIDENCE_DOUBLE_COUNTED",)
                        if scenario_id == "joint_emission_alias"
                        else ()
                    ),
                )
                for scenario_id in scenarios
            ),
            hard_gates=(HardGate("offline_fixture_only", True),),
        ),
        EvidenceOption(
            option_id="E_XRD_SAME_HOLDER",
            cost=1.1,
            outcomes=tuple(
                outcome(
                    scenario_id,
                    "xrd_reference_complete",
                    0.5 if scenario_id == "joint_phase_mixture" else 8.0,
                    *(("XRD_PEAK_ALIASING",) if scenario_id == "joint_phase_mixture" else ()),
                )
                for scenario_id in scenarios
            ),
            hard_gates=(HardGate("offline_fixture_only", True),),
        ),
        EvidenceOption(
            option_id="E_BLINDED_ASSAY",
            cost=0.0,
            outcomes=tuple(
                outcome(scenario_id, "forbidden", 0.0, *predicate.required_failure_atoms)
                for scenario_id in scenarios
            ),
            hard_gates=(HardGate("execution_authority", False, "EXECUTION_AUTHORITY_ABSENT"),),
        ),
    )
    sequence_model = SequenceOutcomeModel(
        model_id="blind-sample-explicit-sequence-terminal-v1",
        outcomes=tuple(
            SequenceOutcome(
                root_option_id=root.option_id,
                root_observation=root.outcome_for(scenario_id).observation,
                second_option_id=second.option_id,
                scenario_id=scenario_id,
                terminal_residual_loss=second.outcome_for(scenario_id).residual_loss,
                terminal_closed_failure_atoms=tuple(
                    sorted(
                        set(root.outcome_for(scenario_id).closed_failure_atoms)
                        | set(second.outcome_for(scenario_id).closed_failure_atoms)
                    )
                ),
            )
            for root in options
            for second in options
            if root.option_id != second.option_id or root.repeatable
            for scenario_id in scenarios
        ),
    )
    lineage_sha256 = canonical_sha256({"source": "synthetic", "sample_id": "blind-sample-fixture-001"})
    evidence_dag = EvidenceDAG(
        (
            EvidenceRecord(
                schema_version="xrd-rb-voe-evidence-record-v1",
                evidence_id="demo-xrd-visual-fixture",
                kind="XRD_VISUAL_FIXTURE",
                source=EvidenceSource.SIMULATED_COUNTERFACTUAL,
                source_id="sealed-demo-xrd-visual-v1",
                lineage_sha256=lineage_sha256,
                payload_sha256=canonical_sha256({"fixture": "xrd-visual-v1"}),
                observed_at_ms=0,
                metadata={"failure_domains": ["spectrometer_calibration"]},
            ),
            EvidenceRecord(
                schema_version="xrd-rb-voe-evidence-record-v1",
                evidence_id="demo-xrd-numerical-fixture",
                kind="XRD_NUMERICAL_FIXTURE",
                source=EvidenceSource.SIMULATED_COUNTERFACTUAL,
                source_id="sealed-demo-xrd-numerical-v1",
                lineage_sha256=lineage_sha256,
                payload_sha256=canonical_sha256({"fixture": "xrd-numerical-v1"}),
                observed_at_ms=0,
                metadata={"failure_domains": ["diffractometer_calibration"]},
            ),
        )
    )
    root_provenance_sha256 = evidence_dag.content_sha256
    experiment_case = ExperimentCase(
        schema_version="xrd-rb-voe-experiment-case-v1",
        case_id=f"synthetic-blind-sample-001{core_suffix}",
        sample_id="blind-sample-fixture-001",
        lineage_sha256=lineage_sha256,
        evidence_root_sha256=root_provenance_sha256,
        allowed_options=tuple(sorted(option.option_id for option in options)),
        required_failure_atoms=predicate.required_failure_atoms,
        closure_predicate_id=predicate.predicate_id,
        release_id="x5-rb-voe-r1-demo-release-v1",
        created_at_ms=0,
    )
    perturbation_specs = {
        "CORRELATED_EVIDENCE_DOUBLE_COUNTED": (
            "demo-failure-atom-0",
            "failure-atom-0",
            "demo-xrd-numerical-fixture",
            "E_PL_CROSSCHECK",
            1.0,
        ),
        "XRD_PEAK_ALIASING": (
            "demo-failure-atom-1",
            "failure-atom-1",
            "demo-xrd-visual-fixture",
            "E_XRD_SAME_HOLDER",
            2.0,
        ),
    }
    perturbation_registry = PerturbationRegistry(
        tuple(
            RegisteredPerturbation(
                perturbation_id=perturbation_specs[atom][0],
                family=perturbation_specs[atom][1],
                patches=(StatePatch(("failure_flags", atom), True),),
                distance=perturbation_specs[atom][4],
                failure_atoms=(atom,),
                affected_evidence_ids=(perturbation_specs[atom][2],),
                repair_options=(perturbation_specs[atom][3],),
            )
            for atom in predicate.required_failure_atoms
        )
    )
    counterexample_state = canonical_policy_search_state(
        experiment_case=experiment_case,
        evidence_dag=evidence_dag,
        selected_evidence_ids=evidence_dag.evidence_ids,
    )
    evaluator_search_contract = declarative_flag_evaluator_contract(
        release_id=experiment_case.release_id,
        authority="SIMULATED_COUNTERFACTUAL_ONLY",
    )
    evaluator_search_contract_release_sha256 = canonical_sha256(evaluator_search_contract)

    def evaluate_counterexample(candidate: Mapping[str, object]) -> DecisionAssessment:
        return evaluate_declarative_decision(candidate, evaluator_search_contract)

    counterexample_search = search_registered_perturbations(
        counterexample_state,
        perturbation_registry,
        evaluate_counterexample,
        evaluator_search_contract_release_sha256=(evaluator_search_contract_release_sha256),
    )
    evidence_admission = PolicyEvidenceAdmission(
        experiment_case=experiment_case,
        evidence_dag=evidence_dag,
        evidence_ids=evidence_dag.evidence_ids,
        minimum_independent_evidence=2,
        perturbation_registry=perturbation_registry,
        counterexample_search=counterexample_search,
        evaluator_search_contract=evaluator_search_contract,
        evaluator_search_contract_release_sha256=(evaluator_search_contract_release_sha256),
    )
    return (
        scenario_set,
        options,
        sequence_model,
        predicate,
        root_provenance_sha256,
        experiment_case,
        evidence_admission,
    )


def _plan_summary(plan: PolicyPlan) -> dict[str, object]:
    effective_risk = plan.hold_risk if plan.is_hold else plan.plan_risk
    return {
        "decision": plan.decision,
        "reason": plan.reason,
        "root_option_id": plan.root_option_id,
        "branches": [branch.to_dict() for branch in plan.branches],
        "risk": effective_risk,
        "hold_risk": plan.hold_risk,
        "conservative_voi": plan.conservative_voi,
        "terminal_closure_guaranteed": plan.terminal_closure_guaranteed,
        "plan_sha256": plan.plan_sha256,
    }


def _plan_losses(
    *,
    plan: PolicyPlan,
    scenario_set: JointScenarioSet,
    options: tuple[EvidenceOption, ...],
    sequence_model: SequenceOutcomeModel,
    predicate: ClosurePredicate,
) -> dict[str, float]:
    if plan.is_hold or plan.root_option_id is None:
        return {scenario.scenario_id: scenario.hold_loss for scenario in scenario_set.scenarios}
    option_by_id = {option.option_id: option for option in options}
    root = option_by_id[plan.root_option_id]
    branch_by_observation = {branch.observation: branch.option_id for branch in plan.branches}
    losses: dict[str, float] = {}
    for scenario_id in scenario_set.scenario_ids:
        root_outcome = root.outcome_for(scenario_id)
        if plan.horizon == 1 or predicate.is_closed(root_outcome.closed_failure_atoms):
            losses[scenario_id] = root.cost + root_outcome.residual_loss
            continue
        second_id = branch_by_observation[root_outcome.observation]
        if second_id is None:
            raise DemoVerificationError(
                "COMPARISON_BRANCH_MISSING",
                f"no second option for {root_outcome.observation}",
            )
        second = option_by_id[second_id]
        terminal = sequence_model.outcome_for(
            root_option_id=root.option_id,
            root_observation=root_outcome.observation,
            second_option_id=second.option_id,
            scenario_id=scenario_id,
        )
        losses[scenario_id] = root.cost + second.cost + terminal.terminal_residual_loss
    return losses


def _maximum_evidence_cost(plan: PolicyPlan, options: tuple[EvidenceOption, ...]) -> float:
    if plan.is_hold or plan.root_option_id is None:
        return 0.0
    option_by_id = {option.option_id: option for option in options}
    root_cost = option_by_id[plan.root_option_id].cost
    branch_costs = tuple(
        option_by_id[branch.option_id].cost for branch in plan.branches if branch.option_id is not None
    )
    return root_cost + (max(branch_costs) if branch_costs else 0.0)


def _fixed_sequence_reference(
    *,
    experiment_case: ExperimentCase,
    scenario_set: JointScenarioSet,
    options: tuple[EvidenceOption, ...],
    sequence_model: SequenceOutcomeModel,
    predicate: ClosurePredicate,
    evidence_admission: PolicyEvidenceAdmission,
) -> dict[str, object]:
    """Exhaust every non-contingent root/second pair through the real planner."""
    feasible = tuple(
        sorted((option for option in options if option.feasible), key=lambda item: item.option_id)
    )
    attempts: list[dict[str, object]] = []
    complete_plans: list[tuple[PolicyPlan, str | None]] = []
    for root in feasible:
        open_observations = tuple(
            sorted(
                {
                    root.outcome_for(scenario_id).observation
                    for scenario_id in scenario_set.scenario_ids
                    if not predicate.is_closed(root.outcome_for(scenario_id).closed_failure_atoms)
                }
            )
        )
        second_ids: tuple[str | None, ...]
        if open_observations:
            second_ids = tuple(
                option.option_id
                for option in feasible
                if option.option_id != root.option_id or root.repeatable
            )
        else:
            second_ids = (None,)
        for second_id in second_ids:
            fixed_branches = tuple(FixedBranch(observation, second_id) for observation in open_observations)
            fixed_plan = plan_policy(
                experiment_case=experiment_case,
                scenario_set=scenario_set,
                options=options,
                closure_predicate=predicate,
                sequence_model=sequence_model,
                evidence_admission=evidence_admission,
                config=PolicyConfig(
                    horizon=2,
                    variant=ScenarioVariant.ROBUST,
                    mode=PolicyMode.FIXED,
                    fixed_root_option_id=root.option_id,
                    fixed_branches=fixed_branches,
                ),
            )
            attempts.append(
                {
                    "root_option_id": root.option_id,
                    "second_option_id": second_id,
                    "decision": fixed_plan.decision,
                    "reason": fixed_plan.reason,
                    "risk": fixed_plan.hold_risk if fixed_plan.is_hold else fixed_plan.plan_risk,
                    "terminal_closure_guaranteed": fixed_plan.terminal_closure_guaranteed,
                    "plan_sha256": fixed_plan.plan_sha256,
                }
            )
            if not fixed_plan.is_hold:
                complete_plans.append((fixed_plan, second_id))

    if not complete_plans:
        return {
            "decision": "HOLD",
            "reason": "NO_COMPLETE_FIXED_SEQUENCE",
            "risk": scenario_set.expected_hold_loss(ScenarioVariant.ROBUST),
            "terminal_closure_guaranteed": False,
            "enumerated_sequence_count": len(attempts),
            "complete_sequence_count": 0,
            "selected_root_option_id": None,
            "selected_second_option_id": None,
            "attempts": attempts,
        }
    selected, second_id = min(
        complete_plans,
        key=lambda item: (
            item[0].plan_risk if item[0].plan_risk is not None else float("inf"),
            item[0].root_option_id or "",
            item[1] or "",
        ),
    )
    return {
        "decision": selected.decision,
        "reason": selected.reason,
        "risk": selected.plan_risk,
        "terminal_closure_guaranteed": selected.terminal_closure_guaranteed,
        "enumerated_sequence_count": len(attempts),
        "complete_sequence_count": len(complete_plans),
        "selected_root_option_id": selected.root_option_id,
        "selected_second_option_id": second_id,
        "attempts": attempts,
    }


def _full_evidence_reference(
    *,
    scenario_set: JointScenarioSet,
    options: tuple[EvidenceOption, ...],
    predicate: ClosurePredicate,
) -> dict[str, object]:
    """Evaluate a transparent all-feasible-evidence diagnostic, not a deployable policy."""
    feasible = tuple(
        sorted((option for option in options if option.feasible), key=lambda item: item.option_id)
    )
    evidence_cost = sum(option.cost for option in feasible)
    closure_by_scenario: dict[str, bool] = {}
    for scenario_id in scenario_set.scenario_ids:
        closed_atoms = {
            atom for option in feasible for atom in option.outcome_for(scenario_id).closed_failure_atoms
        }
        closure_by_scenario[scenario_id] = predicate.is_closed(closed_atoms)
    return {
        "decision": "DIAGNOSTIC_REFERENCE",
        "reason": "ALL_FEASIBLE_OPTIONS_EXECUTED",
        "risk": None,
        "risk_reason": "UNDEFINED_NO_N_STEP_OUTCOME_MODEL",
        "option_ids": [option.option_id for option in feasible],
        "option_count": len(feasible),
        "evidence_cost": evidence_cost,
        "standalone_atom_union_covers_failure_core": all(closure_by_scenario.values()),
        "closure_by_scenario": closure_by_scenario,
        "coverage_semantics": "UNION_OF_STANDALONE_CLOSED_FAILURE_ATOMS_ONLY",
        "deployable_policy": False,
    }


def _strategy_comparison(
    *,
    experiment_case: ExperimentCase,
    scenario_set: JointScenarioSet,
    options: tuple[EvidenceOption, ...],
    sequence_model: SequenceOutcomeModel,
    predicate: ClosurePredicate,
    evidence_admission: PolicyEvidenceAdmission,
    h2_plan: PolicyPlan,
) -> dict[str, object]:
    h1_plan = plan_policy(
        experiment_case=experiment_case,
        scenario_set=scenario_set,
        options=options,
        closure_predicate=predicate,
        evidence_admission=evidence_admission,
        config=PolicyConfig(horizon=1, variant=ScenarioVariant.ROBUST),
    )
    always_hold_plan = plan_policy(
        experiment_case=experiment_case,
        scenario_set=scenario_set,
        options=options,
        closure_predicate=predicate,
        sequence_model=sequence_model,
        evidence_admission=evidence_admission,
        config=PolicyConfig(
            horizon=2,
            variant=ScenarioVariant.ROBUST,
            mode=PolicyMode.ALWAYS_HOLD,
        ),
    )
    fixed_reference = _fixed_sequence_reference(
        experiment_case=experiment_case,
        scenario_set=scenario_set,
        options=options,
        sequence_model=sequence_model,
        predicate=predicate,
        evidence_admission=evidence_admission,
    )
    full_reference = _full_evidence_reference(
        scenario_set=scenario_set,
        options=options,
        predicate=predicate,
    )
    blocked_options = tuple(
        replace(
            option,
            hard_gates=(
                HardGate(
                    "benchmark_all_options_blocked",
                    False,
                    "BENCHMARK_HARD_GATE",
                ),
            ),
        )
        for option in options
    )
    hard_gate_plan = plan_policy(
        experiment_case=experiment_case,
        scenario_set=scenario_set,
        options=blocked_options,
        closure_predicate=predicate,
        sequence_model=sequence_model,
        evidence_admission=evidence_admission,
        config=PolicyConfig(horizon=2, variant=ScenarioVariant.ROBUST),
    )
    hard_gate_request = SimulationRequest(
        episode_id="r1-demo-hard-gate-probe",
        seed="x5-rb-voe-r1-hard-gate-seed",
        horizon=2,
        scenario_set=scenario_set,
        scenario_variant=ScenarioVariant.ROBUST,
        options=tuple(
            SimulatedOption(
                option=option,
                duration_ms_by_scenario={scenario_id: 100 for scenario_id in scenario_set.scenario_ids},
            )
            for option in blocked_options
        ),
        closure_predicate=predicate,
        sequence_model=sequence_model,
        policy_plan=hard_gate_plan,
        pinned_policy_plan_sha256=hard_gate_plan.plan_sha256,
        pinned_sequence_model_sha256=sequence_model.content_sha256,
        root_provenance_sha256=experiment_case.evidence_root_sha256,
    )
    hard_gate_replay = replay_all_scenarios(hard_gate_request)

    failure_core_probes: list[dict[str, object]] = []
    for failure_atom in predicate.required_failure_atoms:
        (
            probe_scenarios,
            probe_options,
            probe_sequence,
            probe_predicate,
            _,
            probe_case,
            probe_admission,
        ) = _build_problem((failure_atom,))
        probe_plan = plan_policy(
            experiment_case=probe_case,
            scenario_set=probe_scenarios,
            options=probe_options,
            closure_predicate=probe_predicate,
            sequence_model=probe_sequence,
            evidence_admission=probe_admission,
            config=PolicyConfig(horizon=2, variant=ScenarioVariant.ROBUST),
        )
        failure_core_probes.append(
            {
                "required_failure_atoms": [failure_atom],
                "root_option_id": probe_plan.root_option_id,
                "branches": [branch.to_dict() for branch in probe_plan.branches],
                "risk": probe_plan.hold_risk if probe_plan.is_hold else probe_plan.plan_risk,
                "terminal_closure_guaranteed": probe_plan.terminal_closure_guaranteed,
                "plan_sha256": probe_plan.plan_sha256,
            }
        )

    h2_summary = _plan_summary(h2_plan)
    h1_summary = _plan_summary(h1_plan)
    always_hold_summary = _plan_summary(always_hold_plan)
    for summary, plan in ((h2_summary, h2_plan), (h1_summary, h1_plan)):
        losses = _plan_losses(
            plan=plan,
            scenario_set=scenario_set,
            options=options,
            sequence_model=sequence_model,
            predicate=predicate,
        )
        recomputed_risk = scenario_set.risk(losses, ScenarioVariant.ROBUST)
        if recomputed_risk != plan.plan_risk:
            raise DemoVerificationError(
                "COMPARISON_RISK_MISMATCH",
                f"recomputed {recomputed_risk}, plan reports {plan.plan_risk}",
            )
        summary["loss_by_scenario"] = losses
        summary["risk_members"] = list(scenario_set.risk_members(losses, ScenarioVariant.ROBUST))
        summary["maximum_evidence_cost"] = _maximum_evidence_cost(plan, options)
    if h2_plan.plan_risk is None or h1_plan.plan_risk is None:
        raise DemoVerificationError(
            "COMPARISON_PLAN_HELD",
            "H2 and H1 comparison plans must both expose modeled risk",
        )
    body = {
        "schema_version": COMPARISON_SCHEMA,
        "claim_boundary": {
            "evidence_source": EvidenceSource.SIMULATED_COUNTERFACTUAL.value,
            "execution_authority": False,
            "hardware_touched": False,
            "network_touched": False,
            "physical_closure": False,
            "same_sealed_scenario_set_sha256": scenario_set.content_sha256,
            "same_evidence_admission_sha256": evidence_admission.content_sha256,
            "same_closure_predicate_sha256": predicate.content_sha256,
            "same_option_set_sha256": canonical_sha256(
                [option.to_dict() for option in sorted(options, key=lambda item: item.option_id)]
            ),
            "same_sequence_model_sha256": sequence_model.content_sha256,
        },
        "risk_contract": {
            "variant": ScenarioVariant.ROBUST.value,
            "h1_formula": "ROOT_COST_PLUS_STANDALONE_RESIDUAL",
            "h2_formula": "ROOT_COST_PLUS_BRANCH_COST_PLUS_SEQUENCE_TERMINAL_RESIDUAL",
            "aggregate": "MAX_OVER_ROBUST_RISK_MEMBERS",
            "full_evidence_risk": "UNDEFINED_NO_N_STEP_OUTCOME_MODEL",
            "floating_point_tolerance": 0.0,
        },
        "strategies": {
            "rb_voe_h2_adaptive": h2_summary,
            "rb_voe_h1": h1_summary,
            "fixed_two_step": fixed_reference,
            "full_evidence_diagnostic": full_reference,
            "always_hold": always_hold_summary,
        },
        "measured_deltas": {
            "h2_minus_h1_risk": h2_plan.plan_risk - h1_plan.plan_risk,
            "h2_minus_hold_risk": h2_plan.plan_risk - h2_plan.hold_risk,
            "h2_max_evidence_cost_minus_full_evidence_cost": (
                h2_summary["maximum_evidence_cost"] - full_reference["evidence_cost"]
            ),
        },
        "hard_gate_probe": {
            **_plan_summary(hard_gate_plan),
            "rejected_option_count": len(hard_gate_plan.rejected_options),
            "exhaustive_replay": hard_gate_replay.to_dict(),
            "exhaustive_replay_sha256": hard_gate_replay.replay_sha256,
            "all_observation_counts_zero": all(
                len(episode.observations) == 0 for episode in hard_gate_replay.episodes
            ),
            "termination_reasons": sorted(
                {episode.termination_reason for episode in hard_gate_replay.episodes}
            ),
        },
        "failure_core_sensitivity": {
            "probes": failure_core_probes,
            "first_action_changes": len({probe["root_option_id"] for probe in failure_core_probes}) > 1,
            "action_tree_changes": len({canonical_sha256(probe["branches"]) for probe in failure_core_probes})
            > 1,
        },
    }
    return {**body, "comparison_sha256": canonical_sha256(body)}


def verify_demo_strategy_comparison(payload: Mapping[str, object]) -> dict[str, object]:
    """Rebuild the sealed world and reject any detached comparison assertion."""
    actual = to_primitive(payload)
    if not isinstance(actual, dict) or set(actual) != {
        "schema_version",
        "claim_boundary",
        "risk_contract",
        "strategies",
        "measured_deltas",
        "hard_gate_probe",
        "failure_core_sensitivity",
        "comparison_sha256",
    }:
        raise DemoVerificationError(
            "COMPARISON_FIELDS_INVALID",
            "strategy comparison fields do not match the sealed contract",
        )
    claimed_digest = actual.pop("comparison_sha256")
    if claimed_digest != canonical_sha256(actual):
        raise DemoVerificationError(
            "COMPARISON_DIGEST_MISMATCH",
            "strategy comparison body does not match its digest",
        )
    actual["comparison_sha256"] = claimed_digest

    (
        scenario_set,
        options,
        sequence_model,
        predicate,
        _,
        experiment_case,
        evidence_admission,
    ) = _build_problem()
    h2_plan = plan_policy(
        experiment_case=experiment_case,
        scenario_set=scenario_set,
        options=options,
        closure_predicate=predicate,
        sequence_model=sequence_model,
        evidence_admission=evidence_admission,
        config=PolicyConfig(horizon=2, variant=ScenarioVariant.ROBUST),
    )
    expected = _strategy_comparison(
        experiment_case=experiment_case,
        scenario_set=scenario_set,
        options=options,
        sequence_model=sequence_model,
        predicate=predicate,
        evidence_admission=evidence_admission,
        h2_plan=h2_plan,
    )
    if actual != expected:
        raise DemoVerificationError(
            "COMPARISON_REPLAY_MISMATCH",
            "strategy comparison differs from a fresh sealed-world replay",
        )
    return {
        "ok": True,
        "reason_code": "PASS",
        "comparison_sha256": claimed_digest,
    }


def _load_expected(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DemoVerificationError("EXPECTED_FIXTURE_INVALID", str(exc)) from exc
    required = {
        "schema_version",
        "expected_root_option_id",
        "expected_branches",
        "expected_root_scenario_id",
        "expected_observed_options",
        "expected_plan_sha256",
        "expected_episode_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise DemoVerificationError(
            "EXPECTED_FIXTURE_FIELDS_INVALID",
            "golden fixture fields do not match the v1 contract",
        )
    if payload["schema_version"] != EXPECTED_SCHEMA:
        raise DemoVerificationError("EXPECTED_FIXTURE_SCHEMA_INVALID", "unsupported schema")
    if not isinstance(payload["expected_branches"], dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in payload["expected_branches"].items()
    ):
        raise DemoVerificationError("EXPECTED_BRANCHES_INVALID", "branches must be strings")
    if not isinstance(payload["expected_observed_options"], list) or not all(
        isinstance(value, str) for value in payload["expected_observed_options"]
    ):
        raise DemoVerificationError(
            "EXPECTED_OBSERVED_OPTIONS_INVALID",
            "observed options must be a string list",
        )
    for field in ("expected_plan_sha256", "expected_episode_sha256"):
        if not is_sha256(payload[field]):
            raise DemoVerificationError("EXPECTED_DIGEST_INVALID", f"{field} must be a SHA-256 digest")
    return payload


def _assert_expected(
    expected: Mapping[str, Any],
    plan: PolicyPlan,
    episode_sha256: str,
    root_scenario_id: str,
    observed_options: list[str],
) -> None:
    actual_branches = {branch.observation: branch.option_id for branch in plan.branches}
    checks = (
        ("ROOT_OPTION_MISMATCH", plan.root_option_id, expected["expected_root_option_id"]),
        ("BRANCH_MAP_MISMATCH", actual_branches, expected["expected_branches"]),
        ("ROOT_SCENARIO_MISMATCH", root_scenario_id, expected["expected_root_scenario_id"]),
        (
            "OBSERVED_OPTIONS_MISMATCH",
            observed_options,
            expected["expected_observed_options"],
        ),
        ("PLAN_DIGEST_MISMATCH", plan.plan_sha256, expected["expected_plan_sha256"]),
        ("EPISODE_DIGEST_MISMATCH", episode_sha256, expected["expected_episode_sha256"]),
    )
    for code, actual, wanted in checks:
        if actual != wanted:
            raise DemoVerificationError(code, f"expected {wanted!r}, got {actual!r}")


def _release_manifest(
    *,
    expected_path: Path,
    scenario_set: JointScenarioSet,
    options: tuple[EvidenceOption, ...],
    sequence_model: SequenceOutcomeModel,
    predicate: ClosurePredicate,
    config: PolicyConfig,
    plan: PolicyPlan,
    evidence_admission: PolicyEvidenceAdmission,
    strategy_comparison_sha256: str,
    simulation_request_sha256: str,
    exhaustive_replay_sha256: str,
) -> tuple[Any, dict[str, dict[str, str]], Path]:
    package_root = Path(__file__).resolve().parent
    project_root = package_root.parent
    source_inventory = {
        path.relative_to(project_root).as_posix(): file_sha256(path)
        for path in sorted(package_root.rglob("*.py"))
        if path.is_file()
    }
    for project_file in ("pyproject.toml", "requirements-dev.txt"):
        path = project_root / project_file
        if path.is_file():
            source_inventory[project_file] = file_sha256(path)
    schema_inventory = {
        path.relative_to(project_root).as_posix(): file_sha256(path)
        for path in sorted((package_root / "contracts" / "schemas").rglob("*.json"))
        if path.is_file()
    }
    try:
        fixture_key = expected_path.resolve(strict=True).relative_to(project_root).as_posix()
    except (OSError, ValueError) as exc:
        raise DemoVerificationError(
            "FIXTURE_PATH_OUTSIDE_PROJECT_ROOT",
            "the verified golden fixture must be a project file",
        ) from exc
    registry_values = {
        "authority_domains": AUTHORITY_DOMAINS,
        "authority_key_domains": dict(AUTHORITY_KEY_DOMAINS),
        "failure_core_reason_codes": FAILURE_CORE_REASON_CODES,
        "key_domains": KEY_DOMAINS,
        "macro_ids": MACRO_IDS,
        "option_ids": OPTION_IDS,
        "physical_evidence_statuses": PHYSICAL_EVIDENCE_STATUSES,
        "reason_codes": REASON_CODES,
        "role_bindings": {key: list(value) for key, value in ROLE_BINDINGS.items()},
        "route_ids": ROUTE_IDS,
        "station_ids": STATION_IDS,
        "zone_ids": ZONE_IDS,
    }
    logical_inventories = {
        "registry_sha256": {
            **{name: canonical_sha256(value) for name, value in registry_values.items()},
            "evidence_options": canonical_sha256([option.to_dict() for option in options]),
            "sequence_outcome_model": sequence_model.content_sha256,
        },
        "policy_config_sha256": {
            "closure_predicate": canonical_sha256(predicate.to_dict()),
            "evidence_admission": evidence_admission.content_sha256,
            "exhaustive_replay": exhaustive_replay_sha256,
            "joint_scenario_set": scenario_set.content_sha256,
            "policy_config": canonical_sha256(asdict(config)),
            "policy_plan": plan.plan_sha256,
            "simulation_request": simulation_request_sha256,
            "strategy_comparison": strategy_comparison_sha256,
        },
        "environment_sha256": {
            "hardware": canonical_sha256(False),
            "network": canonical_sha256(False),
            "python_minimum": canonical_sha256("3.10"),
            "runtime": canonical_sha256("stdlib-only"),
        },
    }
    manifest = build_release_manifest(
        release_id="x5-rb-voe-r1-demo-release-v1",
        maturity=Maturity.SIMULATED,
        source_sha256=source_inventory,
        schema_sha256=schema_inventory,
        registry_sha256=logical_inventories["registry_sha256"],
        fixture_sha256={fixture_key: file_sha256(expected_path)},
        policy_config_sha256=logical_inventories["policy_config_sha256"],
        environment_sha256=logical_inventories["environment_sha256"],
        created_at_ms=0,
    )
    return manifest, logical_inventories, project_root


def run_demo(
    output_dir: str | Path,
    *,
    expected_fixture_path: str | Path | None = None,
    external_pin_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run and verify the sealed R1 demo in a fresh output directory.

    The function has no hardware, network, signature, permit, or wall-clock path.
    It fails closed before ledger creation when the golden vector does not match.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    expected_path = (
        Path(expected_fixture_path) if expected_fixture_path is not None else _EXPECTED_FIXTURE_PATH
    )
    expected = _load_expected(expected_path)
    (
        scenario_set,
        options,
        sequence_model,
        predicate,
        root_provenance_sha256,
        experiment_case,
        evidence_admission,
    ) = _build_problem()
    config = PolicyConfig(horizon=2, variant=ScenarioVariant.ROBUST)
    plan = plan_policy(
        experiment_case=experiment_case,
        scenario_set=scenario_set,
        options=options,
        closure_predicate=predicate,
        sequence_model=sequence_model,
        evidence_admission=evidence_admission,
        config=config,
    )
    if plan.is_hold or plan.root_option_id is None:
        raise DemoVerificationError("POLICY_HELD", plan.reason)
    strategy_comparison = _strategy_comparison(
        experiment_case=experiment_case,
        scenario_set=scenario_set,
        options=options,
        sequence_model=sequence_model,
        predicate=predicate,
        evidence_admission=evidence_admission,
        h2_plan=plan,
    )
    strategy_comparison_verification = verify_demo_strategy_comparison(strategy_comparison)

    simulated_options = tuple(
        SimulatedOption(
            option=option,
            duration_ms_by_scenario={
                scenario_id: 100 + index * 20 for index, scenario_id in enumerate(scenario_set.scenario_ids)
            },
        )
        for option in options
    )
    request = SimulationRequest(
        episode_id="r1-demo-episode-001",
        seed="x5-rb-voe-r1-golden-seed",
        horizon=2,
        scenario_set=scenario_set,
        scenario_variant=ScenarioVariant.ROBUST,
        options=simulated_options,
        closure_predicate=predicate,
        sequence_model=sequence_model,
        policy_plan=plan,
        pinned_policy_plan_sha256=plan.plan_sha256,
        pinned_sequence_model_sha256=sequence_model.content_sha256,
        root_provenance_sha256=root_provenance_sha256,
    )
    episode = run_episode(request)
    exhaustive_replay = replay_all_scenarios(request)
    observed_options = [item.option_id for item in episode.observations]
    _assert_expected(
        expected,
        plan,
        episode.episode_sha256,
        episode.root_scenario_id,
        observed_options,
    )
    if not episode.modeled_closure_satisfied:
        raise DemoVerificationError("FAILURE_CORE_OPEN", episode.termination_reason)
    if not exhaustive_replay.all_modeled_closure_satisfied:
        raise DemoVerificationError("EXHAUSTIVE_FAILURE_CORE_OPEN", exhaustive_replay.replay_sha256)
    release_manifest, logical_inventories, project_root = _release_manifest(
        expected_path=expected_path,
        scenario_set=scenario_set,
        options=options,
        sequence_model=sequence_model,
        predicate=predicate,
        config=config,
        plan=plan,
        evidence_admission=evidence_admission,
        strategy_comparison_sha256=strategy_comparison["comparison_sha256"],
        simulation_request_sha256=request.content_sha256,
        exhaustive_replay_sha256=exhaustive_replay.replay_sha256,
    )

    ledger_path = initialize_ledger(output / "demo_audit.jsonl")
    case_row = append_record(
        ledger_path,
        record_id="demo.case.001",
        nonce="demo-v1-case",
        record_type="EXPERIMENT_CASE",
        payload=experiment_case.to_dict(),
    )
    plan_row = append_record(
        ledger_path,
        record_id="demo.plan.001",
        nonce="demo-v1-plan",
        record_type="CONTINGENT_POLICY_PLAN",
        payload={
            "schema_version": "xrd-rb-voe-policy-plan-event-v4",
            "case_id": experiment_case.case_id,
            "experiment_case_sha256": experiment_case.content_sha256,
            "previous_record_sha256": case_row["record_sha256"],
            "evidence_admission": evidence_admission.to_dict(),
            "evidence_admission_sha256": evidence_admission.content_sha256,
            "plan": plan.to_dict(),
            "plan_sha256": plan.plan_sha256,
        },
    )
    observation_row = append_record(
        ledger_path,
        record_id="demo.observation.001",
        nonce="demo-v1-observation",
        record_type="SIMULATED_OBSERVATION",
        payload={
            "schema_version": "xrd-rb-voe-simulated-observation-event-v2",
            "case_id": experiment_case.case_id,
            "experiment_case_sha256": experiment_case.content_sha256,
            "plan_sha256": plan.plan_sha256,
            "previous_record_sha256": plan_row["record_sha256"],
            "simulation_request": request.to_dict(),
            "simulation_request_sha256": request.content_sha256,
            "episode": episode.to_dict(),
            "episode_sha256": episode.episode_sha256,
        },
    )
    receipt = DecisionReceipt(
        schema_version="xrd-rb-voe-decision-receipt-v1",
        receipt_id="demo-receipt-001",
        case_id=experiment_case.case_id,
        plan_epoch=1,
        decision=Decision.REVISE,
        failure_core_closed=True,
        selected_option=observed_options[-1],
        terminal_evidence_sha256=(
            episode.episode_sha256,
            exhaustive_replay.replay_sha256,
        ),
        # A SIMULATED release manifest is provenance, never a production policy certificate.
        release_certificate_sha256=None,
        previous_record_sha256=observation_row["record_sha256"],
        created_at_ms=episode.cumulative_duration_ms,
    )
    append_record(
        ledger_path,
        record_id="demo.terminal.001",
        nonce="demo-v1-terminal",
        record_type="SIMULATED_TERMINAL",
        payload={
            "schema_version": "xrd-rb-voe-simulated-terminal-event-v2",
            "case_id": experiment_case.case_id,
            "experiment_case_sha256": experiment_case.content_sha256,
            "plan_sha256": plan.plan_sha256,
            "simulation_request_sha256": request.content_sha256,
            "episode_sha256": [episode.episode_sha256],
            "previous_record_sha256": observation_row["record_sha256"],
            "exhaustive_replay": exhaustive_replay.to_dict(),
            "exhaustive_replay_sha256": exhaustive_replay.replay_sha256,
            "decision_receipt": receipt.to_dict(),
            "decision_receipt_sha256": receipt.content_sha256,
            "execution_authority": False,
            "hardware_touched": False,
            "network_touched": False,
            "physical_risk_denominator_increment": 0,
            "simulated_only": True,
        },
    )
    ledger_report = verify_ledger(ledger_path)
    semantic_report = verify_r1_semantic_ledger(ledger_path)
    terminal_manifest = build_terminal_manifest(
        run_id="x5-rb-voe-r1-demo-run-001",
        release_manifest_sha256=release_manifest.content_sha256,
        ledger_path=ledger_path,
        maturity=Maturity.SIMULATED,
    )
    candidate_release_root = build_release_root_pin(release_manifest, terminal_manifest)
    pin_path = Path(external_pin_path) if external_pin_path is not None else None
    external_pin_verified = False
    external_pin_reason = "EXTERNAL_PIN_MISSING"
    if pin_path is not None and pin_path.is_file():
        try:
            external_pin = json.loads(pin_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DemoVerificationError(
                "EXTERNAL_PIN_READ_ERROR", f"cannot read release root pin: {exc}"
            ) from exc
        if not isinstance(external_pin, Mapping):
            raise DemoVerificationError("EXTERNAL_PIN_INVALID", "release root pin must be a JSON object")
        external_pin_verified, external_pin_reason = verify_release_bundle_artifacts(
            release_payload=to_primitive(release_manifest),
            terminal_payload=to_primitive(terminal_manifest),
            external_pin_payload=external_pin,
            ledger_path=ledger_path,
            project_root=project_root,
            actual_registry_sha256=logical_inventories["registry_sha256"],
            actual_policy_config_sha256=logical_inventories["policy_config_sha256"],
            actual_environment_sha256=logical_inventories["environment_sha256"],
        )
        if not external_pin_verified:
            raise DemoVerificationError(
                external_pin_reason,
                "release bundle does not match the externally retained root",
            )

    result = {
        "ok": external_pin_verified,
        "acceptance_status": ("PASS" if external_pin_verified else "UNPINNED_CANDIDATE"),
        "schema_version": DEMO_SCHEMA,
        "case": experiment_case.to_dict(),
        "policy_plan": plan.to_dict(),
        "policy_plan_sha256": plan.plan_sha256,
        "strategy_comparison": strategy_comparison,
        "strategy_comparison_verification": strategy_comparison_verification,
        "evidence_admission": evidence_admission.to_dict(),
        "evidence_admission_sha256": evidence_admission.content_sha256,
        "simulation_request": request.to_dict(),
        "simulation_request_sha256": request.content_sha256,
        "episode": episode.to_dict(),
        "episode_sha256": episode.episode_sha256,
        "exhaustive_replay": exhaustive_replay.to_dict(),
        "exhaustive_replay_sha256": exhaustive_replay.replay_sha256,
        "decision_receipt": receipt.to_dict(),
        "release_manifest": to_primitive(release_manifest),
        "terminal_manifest": to_primitive(terminal_manifest),
        "candidate_release_root": to_primitive(candidate_release_root),
        "external_pin_verification": {
            "verified": external_pin_verified,
            "reason_code": external_pin_reason,
        },
        "semantic_ledger_verification": semantic_report.to_dict(),
        "release_terminal_summary": {
            "maturity": Maturity.SIMULATED.value,
            "release_manifest_sha256": release_manifest.content_sha256,
            "terminal_manifest_sha256": terminal_manifest.content_sha256,
            "candidate_release_root_sha256": candidate_release_root.root_sha256,
            "execution_authority": False,
            "hardware_touched": False,
            "network_touched": False,
            "physical_risk_denominator_increment": 0,
            "simulated_only": True,
        },
        "ledger_verification": ledger_report,
        "artifacts": {
            "ledger": ledger_path.name,
            "terminal_anchor": f"{ledger_path.name}.anchor.json",
            "release_manifest": "release_manifest.json",
            "terminal_manifest": "terminal_manifest.json",
            "candidate_release_root": "candidate_release_root.json",
            "registry_inventory": "registry_inventory.json",
            "policy_inventory": "policy_inventory.json",
            "strategy_comparison": "strategy_comparison.json",
            "environment_inventory": "environment_inventory.json",
            "result": "demo_result.json",
        },
        "authority": {
            "execution_authority": False,
            "hardware_touched": False,
            "network_touched": False,
            "physical_risk_denominator_increment": 0,
            "simulated_only": True,
        },
        "golden_vector_verified": True,
    }
    primitive_result = to_primitive(result)
    _write_canonical_exclusive(output / "release_manifest.json", release_manifest)
    _write_canonical_exclusive(output / "terminal_manifest.json", terminal_manifest)
    _write_canonical_exclusive(output / "candidate_release_root.json", candidate_release_root)
    _write_canonical_exclusive(output / "registry_inventory.json", logical_inventories["registry_sha256"])
    _write_canonical_exclusive(output / "policy_inventory.json", logical_inventories["policy_config_sha256"])
    _write_canonical_exclusive(output / "strategy_comparison.json", strategy_comparison)
    _write_canonical_exclusive(
        output / "environment_inventory.json", logical_inventories["environment_sha256"]
    )
    _write_canonical_exclusive(output / "demo_result.json", primitive_result)
    return primitive_result


__all__ = ["DemoVerificationError", "run_demo", "verify_demo_strategy_comparison"]
