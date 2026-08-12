"""Fail-closed finite-horizon RB-VoE policy kernel."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import product
from math import isclose, isfinite

from rb_voe.contracts.canonical import canonical_sha256, require_sha256, to_primitive
from rb_voe.contracts.models import ExperimentCase
from rb_voe.contracts.registries import (
    FAILURE_CORE_REASON_CODES,
    require_option_id,
)
from rb_voe.core.counterexample import (
    CounterexampleSearchResult,
    PerturbationRegistry,
    apply_registered_perturbation,
    default_harm_assessment,
    evaluate_declarative_decision,
)
from rb_voe.core.evidence_dag import EvidenceDAG
from rb_voe.core.invariants import InvariantReport, evaluate_evidence_invariants
from rb_voe.core.options import ClosurePredicate, EvidenceOption, SequenceOutcomeModel
from rb_voe.core.scenarios import JointScenarioSet, ScenarioVariant


class PolicyMode(str, Enum):
    OPTIMIZED = "OPTIMIZED"
    ALWAYS_HOLD = "ALWAYS_HOLD"
    ALWAYS_PERMIT = "ALWAYS_PERMIT"
    FIRST_FEASIBLE = "FIRST_FEASIBLE"
    FIXED = "FIXED"


class PolicyEvidenceAdmissionError(ValueError):
    """Raised when policy evidence cannot enter the optimization boundary."""


def _failure_domain_binding_sha256(
    *,
    evidence_root_sha256: str,
    scenario_set_sha256: str,
    evidence_failure_domains: tuple[str, ...],
    scenario_failure_domains: tuple[str, ...],
) -> str:
    return canonical_sha256(
        {
            "schema_version": "xrd-rb-voe-failure-domain-binding-v1",
            "evidence_root_sha256": evidence_root_sha256,
            "scenario_set_sha256": scenario_set_sha256,
            "evidence_failure_domains": list(evidence_failure_domains),
            "scenario_failure_domains": list(scenario_failure_domains),
        }
    )


def canonical_policy_search_state(
    *,
    experiment_case: ExperimentCase,
    evidence_dag: EvidenceDAG,
    selected_evidence_ids: tuple[str, ...],
) -> dict[str, object]:
    """Build the sole case-bound baseline accepted by policy counterexample search."""
    if not isinstance(experiment_case, ExperimentCase):
        raise TypeError("experiment_case must be an ExperimentCase")
    if not isinstance(evidence_dag, EvidenceDAG):
        raise TypeError("evidence_dag must be an EvidenceDAG")
    normalized_ids = tuple(sorted(set(selected_evidence_ids)))
    if not normalized_ids or len(normalized_ids) != len(selected_evidence_ids):
        raise ValueError("policy search evidence ids must be non-empty and unique")
    unknown_ids = sorted(set(normalized_ids) - set(evidence_dag.evidence_ids))
    if unknown_ids:
        raise ValueError(f"policy search references evidence outside the DAG: {unknown_ids}")
    if experiment_case.evidence_root_sha256 != evidence_dag.content_sha256:
        raise ValueError("experiment case evidence root does not match the policy search DAG")
    required_atoms = tuple(sorted(experiment_case.required_failure_atoms))
    return {
        "schema_version": "xrd-rb-voe-policy-search-state-v1",
        "case_sha256": experiment_case.content_sha256,
        "evidence_dag_current_sha256": evidence_dag.current_sha256,
        "evidence_dag_content_sha256": evidence_dag.content_sha256,
        "selected_evidence_ids": list(normalized_ids),
        "required_failure_atoms": list(required_atoms),
        "release_id": experiment_case.release_id,
        "failure_flags": {atom: False for atom in required_atoms},
    }


@dataclass(frozen=True, slots=True)
class PolicyEvidenceAdmissionReport:
    case_sha256: str
    evidence_root_sha256: str
    selected_evidence_ids: tuple[str, ...]
    minimum_independent_evidence: int
    invariant_report_sha256: str
    counterexample_registry_sha256: str
    counterexample_search_sha256: str
    counterexample_baseline_state_sha256: str
    evaluator_search_contract_release_sha256: str
    required_failure_atoms: tuple[str, ...]
    registered_count: int
    evaluated_count: int

    @property
    def content_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "xrd-rb-voe-policy-evidence-admission-report-v2",
            "case_sha256": self.case_sha256,
            "evidence_root_sha256": self.evidence_root_sha256,
            "selected_evidence_ids": list(self.selected_evidence_ids),
            "minimum_independent_evidence": self.minimum_independent_evidence,
            "invariant_report_sha256": self.invariant_report_sha256,
            "counterexample_registry_sha256": self.counterexample_registry_sha256,
            "counterexample_search_sha256": self.counterexample_search_sha256,
            "counterexample_baseline_state_sha256": (self.counterexample_baseline_state_sha256),
            "evaluator_search_contract_release_sha256": (self.evaluator_search_contract_release_sha256),
            "required_failure_atoms": list(self.required_failure_atoms),
            "registered_count": self.registered_count,
            "evaluated_count": self.evaluated_count,
        }


class PolicyEvidenceAdmission:
    """Revalidatable admission over the actual evidence and search objects.

    The object deliberately retains the source DAG, registry, and search result.
    Planning re-evaluates them instead of trusting a detached boolean or digest.
    """

    __slots__ = (
        "_baseline_admission_sha256",
        "_baseline_admission_report_sha256",
        "_baseline_case_sha256",
        "_baseline_dag_sha256",
        "_baseline_invariant_report_sha256",
        "_baseline_registry_sha256",
        "_baseline_search_sha256",
        "_baseline_search_state_sha256",
        "_dag",
        "_evidence_ids",
        "_evaluator_search_contract",
        "_evaluator_search_contract_release_sha256",
        "_experiment_case",
        "_minimum_independent_evidence",
        "_registry",
        "_search_result",
    )

    def __init__(
        self,
        *,
        experiment_case: ExperimentCase,
        evidence_dag: EvidenceDAG,
        evidence_ids: tuple[str, ...],
        minimum_independent_evidence: int,
        perturbation_registry: PerturbationRegistry,
        counterexample_search: CounterexampleSearchResult,
        evaluator_search_contract: Mapping[str, object],
        evaluator_search_contract_release_sha256: str,
    ) -> None:
        if not isinstance(experiment_case, ExperimentCase):
            raise TypeError("experiment_case must be an ExperimentCase")
        if not isinstance(evidence_dag, EvidenceDAG):
            raise TypeError("evidence_dag must be an EvidenceDAG")
        if not isinstance(perturbation_registry, PerturbationRegistry):
            raise TypeError("perturbation_registry must be a PerturbationRegistry")
        if not isinstance(counterexample_search, CounterexampleSearchResult):
            raise TypeError("counterexample_search must be a CounterexampleSearchResult")
        require_sha256(
            "evaluator_search_contract_release_sha256",
            evaluator_search_contract_release_sha256,
        )
        canonical_evaluator_contract = to_primitive(evaluator_search_contract)
        if not isinstance(canonical_evaluator_contract, dict) or not canonical_evaluator_contract:
            raise ValueError("evaluator/search contract must be a non-empty canonical mapping")
        if canonical_sha256(canonical_evaluator_contract) != evaluator_search_contract_release_sha256:
            raise ValueError("evaluator/search contract does not match its release digest")
        normalized_ids = tuple(sorted(set(evidence_ids)))
        if not normalized_ids or len(normalized_ids) != len(evidence_ids):
            raise ValueError("policy admission evidence ids must be non-empty and unique")
        if (
            isinstance(minimum_independent_evidence, bool)
            or not isinstance(minimum_independent_evidence, int)
            or minimum_independent_evidence <= 0
        ):
            raise ValueError("minimum independent evidence must be a positive integer")

        self._dag = evidence_dag
        self._experiment_case = experiment_case
        self._evidence_ids = normalized_ids
        self._minimum_independent_evidence = minimum_independent_evidence
        self._registry = perturbation_registry
        self._search_result = counterexample_search
        self._evaluator_search_contract = canonical_evaluator_contract
        self._evaluator_search_contract_release_sha256 = evaluator_search_contract_release_sha256

        report, invariant_report = self._evaluate(experiment_case)
        self._baseline_case_sha256 = experiment_case.content_sha256
        self._baseline_dag_sha256 = evidence_dag.current_sha256
        self._baseline_invariant_report_sha256 = invariant_report.content_sha256
        self._baseline_registry_sha256 = perturbation_registry.content_sha256
        self._baseline_search_sha256 = counterexample_search.content_sha256
        self._baseline_search_state_sha256 = report.counterexample_baseline_state_sha256
        self._baseline_admission_report_sha256 = report.content_sha256
        self._baseline_admission_sha256 = canonical_sha256(
            self._snapshot(report=report, invariant_report=invariant_report)
        )

    @property
    def content_sha256(self) -> str:
        return self._baseline_admission_sha256

    @property
    def evidence_failure_domains(self) -> tuple[str, ...]:
        self.revalidate(self._experiment_case)
        return self._dag.failure_domains(self._evidence_ids)

    def bind_scenario_failure_domains(
        self, scenario_set: JointScenarioSet
    ) -> tuple[tuple[str, ...], tuple[str, ...], str]:
        """Bind admitted evidence dependencies to the sealed scenario registry."""
        if not isinstance(scenario_set, JointScenarioSet):
            raise TypeError("scenario_set must be a JointScenarioSet")
        self.revalidate(self._experiment_case)
        evidence_domains = self._dag.failure_domains(self._evidence_ids)
        scenario_domains = scenario_set.failure_domains
        uncovered = tuple(sorted(set(evidence_domains) - set(scenario_domains)))
        if uncovered:
            raise PolicyEvidenceAdmissionError(
                "sealed scenarios do not cover admitted evidence failure domains: " + ", ".join(uncovered)
            )
        digest = _failure_domain_binding_sha256(
            evidence_root_sha256=self._dag.content_sha256,
            scenario_set_sha256=scenario_set.content_sha256,
            evidence_failure_domains=evidence_domains,
            scenario_failure_domains=scenario_domains,
        )
        return evidence_domains, scenario_domains, digest

    def _snapshot(
        self,
        *,
        report: PolicyEvidenceAdmissionReport,
        invariant_report: InvariantReport,
    ) -> dict[str, object]:
        return {
            "schema_version": "xrd-rb-voe-policy-evidence-admission-v1",
            "case_sha256": self._experiment_case.content_sha256,
            "evidence_dag": self._dag.to_dict(),
            "evidence_dag_current_sha256": self._dag.current_sha256,
            "selected_evidence_ids": list(self._evidence_ids),
            "minimum_independent_evidence": self._minimum_independent_evidence,
            "invariant_report": invariant_report.to_dict(),
            "invariant_report_sha256": invariant_report.content_sha256,
            "perturbation_registry": self._registry.to_dict(),
            "perturbation_registry_sha256": self._registry.content_sha256,
            "counterexample_search": self._search_result.to_dict(),
            "counterexample_search_sha256": self._search_result.content_sha256,
            "evaluator_search_contract": self._evaluator_search_contract,
            "evaluator_search_contract_release_sha256": (self._evaluator_search_contract_release_sha256),
            "admission_report": report.to_dict(),
            "admission_report_sha256": report.content_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        self.revalidate(self._experiment_case)
        report, invariant_report = self._evaluate(self._experiment_case)
        return self._snapshot(report=report, invariant_report=invariant_report)

    def revalidate(self, experiment_case: ExperimentCase) -> PolicyEvidenceAdmissionReport:
        """Recompute every admission fact and reject all post-admission drift."""
        if not isinstance(experiment_case, ExperimentCase):
            raise TypeError("experiment_case must be an ExperimentCase")
        if experiment_case.content_sha256 != self._baseline_case_sha256:
            raise PolicyEvidenceAdmissionError("policy admission is bound to a different experiment case")
        report, invariant_report = self._evaluate(experiment_case)
        frozen_digests = {
            "evidence DAG": (self._baseline_dag_sha256, self._dag.current_sha256),
            "invariant report": (
                self._baseline_invariant_report_sha256,
                invariant_report.content_sha256,
            ),
            "counterexample registry": (
                self._baseline_registry_sha256,
                self._registry.content_sha256,
            ),
            "counterexample search": (
                self._baseline_search_sha256,
                self._search_result.content_sha256,
            ),
            "counterexample baseline state": (
                self._baseline_search_state_sha256,
                report.counterexample_baseline_state_sha256,
            ),
            "admission report": (
                self._baseline_admission_report_sha256,
                report.content_sha256,
            ),
        }
        drifted = tuple(name for name, (expected, actual) in frozen_digests.items() if expected != actual)
        if drifted:
            raise PolicyEvidenceAdmissionError(
                f"policy evidence admission drifted after construction: {', '.join(drifted)}"
            )
        return report

    def _evaluate(
        self, experiment_case: ExperimentCase
    ) -> tuple[PolicyEvidenceAdmissionReport, InvariantReport]:
        if experiment_case.evidence_root_sha256 != self._dag.content_sha256:
            raise PolicyEvidenceAdmissionError(
                "experiment case evidence root does not match the admitted EvidenceDAG"
            )
        invariant_report = evaluate_evidence_invariants(
            self._dag,
            evidence_ids=self._evidence_ids,
            minimum_independent=self._minimum_independent_evidence,
            expected_sha256=experiment_case.evidence_root_sha256,
        )
        if not invariant_report.passed:
            codes = ", ".join(invariant_report.failure_codes)
            raise PolicyEvidenceAdmissionError(f"hard evidence invariants failed: {codes}")

        self._validate_counterexample_search(experiment_case)
        report = PolicyEvidenceAdmissionReport(
            case_sha256=experiment_case.content_sha256,
            evidence_root_sha256=experiment_case.evidence_root_sha256,
            selected_evidence_ids=self._evidence_ids,
            minimum_independent_evidence=self._minimum_independent_evidence,
            invariant_report_sha256=invariant_report.content_sha256,
            counterexample_registry_sha256=self._registry.content_sha256,
            counterexample_search_sha256=self._search_result.content_sha256,
            counterexample_baseline_state_sha256=(self._search_result.baseline_state_sha256),
            evaluator_search_contract_release_sha256=(self._evaluator_search_contract_release_sha256),
            required_failure_atoms=tuple(sorted(experiment_case.required_failure_atoms)),
            registered_count=self._search_result.registered_count,
            evaluated_count=self._search_result.evaluated_count,
        )
        return report, invariant_report

    def _validate_counterexample_search(self, experiment_case: ExperimentCase) -> None:
        registry_sha256 = self._registry.content_sha256
        result = self._search_result
        registered = self._registry.perturbations
        expected_search_state = canonical_policy_search_state(
            experiment_case=experiment_case,
            evidence_dag=self._dag,
            selected_evidence_ids=self._evidence_ids,
        )
        expected_search_state_sha256 = canonical_sha256(expected_search_state)
        if result.baseline_state_sha256 != expected_search_state_sha256:
            raise PolicyEvidenceAdmissionError(
                "counterexample search baseline state does not match the admitted case and EvidenceDAG"
            )
        if result.evaluator_search_contract_release_sha256 != self._evaluator_search_contract_release_sha256:
            raise PolicyEvidenceAdmissionError(
                "counterexample search evaluator/search-contract release does not match the admitted release"
            )
        if result.registry_sha256 != registry_sha256:
            raise PolicyEvidenceAdmissionError(
                "counterexample search is not bound to the admitted perturbation registry"
            )
        expected_baseline = evaluate_declarative_decision(
            expected_search_state, self._evaluator_search_contract
        )
        if result.baseline != expected_baseline:
            raise PolicyEvidenceAdmissionError(
                "counterexample baseline differs from declarative evaluator replay"
            )
        if (
            not result.exhaustive
            or result.registered_count != len(registered)
            or result.evaluated_count != len(registered)
            or result.budget != len(registered)
        ):
            raise PolicyEvidenceAdmissionError(
                "counterexample search must exhaustively evaluate the frozen registry"
            )
        if len(result.candidates) != len(registered):
            raise PolicyEvidenceAdmissionError(
                "counterexample search candidate coverage does not match the frozen registry"
            )

        registered_by_id = {item.perturbation_id: item for item in registered}
        required_atoms = set(experiment_case.required_failure_atoms)
        registry_atoms = {atom for item in registered for atom in item.failure_atoms}
        if registry_atoms != required_atoms:
            raise PolicyEvidenceAdmissionError(
                "counterexample failure core must exactly match experiment case required failure atoms"
            )
        candidate_ids = tuple(candidate.perturbation_id for candidate in result.candidates)
        if len(set(candidate_ids)) != len(candidate_ids) or set(candidate_ids) != set(registered_by_id):
            raise PolicyEvidenceAdmissionError(
                "counterexample search does not cover every registered perturbation exactly once"
            )
        if set(candidate.rank for candidate in result.candidates) != set(range(1, len(registered) + 1)):
            raise PolicyEvidenceAdmissionError("counterexample search ranks are incomplete or duplicated")

        dag_ids = set(self._dag.evidence_ids)
        for candidate in result.candidates:
            perturbation = registered_by_id[candidate.perturbation_id]
            perturbed_state = apply_registered_perturbation(expected_search_state, perturbation)
            expected_perturbed = evaluate_declarative_decision(
                perturbed_state, self._evaluator_search_contract
            )
            expected_harm = default_harm_assessment(expected_baseline, expected_perturbed)
            if (
                candidate.family != perturbation.family
                or candidate.distance != perturbation.distance
                or candidate.failure_core != perturbation.failure_atoms
                or candidate.affected_evidence_ids != perturbation.affected_evidence_ids
                or candidate.repair_options != perturbation.repair_options
                or candidate.baseline != expected_baseline
                or candidate.perturbed != expected_perturbed
                or not isclose(
                    candidate.harm_score,
                    expected_harm.score,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or candidate.harm_reasons != expected_harm.reasons
                or candidate.harmful is not expected_harm.harmful
            ):
                raise PolicyEvidenceAdmissionError(
                    f"counterexample candidate drifted from registry: {candidate.perturbation_id}"
                )
            if not set(candidate.affected_evidence_ids).issubset(dag_ids):
                raise PolicyEvidenceAdmissionError(
                    f"counterexample references evidence outside the admitted DAG: "
                    f"{candidate.perturbation_id}"
                )

        candidate_atoms = {atom for item in result.candidates for atom in item.failure_core}
        if registry_atoms != required_atoms or candidate_atoms != required_atoms:
            raise PolicyEvidenceAdmissionError(
                "counterexample failure core must exactly match experiment case required failure atoms"
            )


@dataclass(frozen=True, slots=True)
class FixedBranch:
    observation: str
    option_id: str | None

    def __post_init__(self) -> None:
        if not self.observation:
            raise ValueError("fixed branch observation must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {"observation": self.observation, "option_id": self.option_id}


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    horizon: int = 2
    variant: ScenarioVariant = ScenarioVariant.ROBUST
    mode: PolicyMode = PolicyMode.OPTIMIZED
    minimum_conservative_voi: float = 0.0
    fixed_root_option_id: str | None = None
    fixed_branches: tuple[FixedBranch, ...] = ()

    def __post_init__(self) -> None:
        if self.horizon not in {1, 2}:
            raise ValueError("policy horizon must be one or two")
        if not isinstance(self.variant, ScenarioVariant):
            object.__setattr__(self, "variant", ScenarioVariant(self.variant))
        if not isinstance(self.mode, PolicyMode):
            object.__setattr__(self, "mode", PolicyMode(self.mode))
        if not isfinite(self.minimum_conservative_voi) or self.minimum_conservative_voi < 0:
            raise ValueError("minimum_conservative_voi must be finite and non-negative")
        normalized = tuple(sorted(self.fixed_branches, key=lambda item: item.observation))
        observations = tuple(item.observation for item in normalized)
        if len(set(observations)) != len(observations):
            raise ValueError("fixed branch observations must be unique")
        object.__setattr__(self, "fixed_branches", normalized)
        if self.mode is PolicyMode.FIXED and self.fixed_root_option_id is None:
            raise ValueError("fixed policy requires fixed_root_option_id")


@dataclass(frozen=True, slots=True)
class RejectedOption:
    option_id: str
    failure_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"option_id": self.option_id, "failure_codes": list(self.failure_codes)}


@dataclass(frozen=True, slots=True)
class ContingentBranch:
    observation: str
    conditioned_scenario_ids: tuple[str, ...]
    option_id: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "observation": self.observation,
            "conditioned_scenario_ids": list(self.conditioned_scenario_ids),
            "option_id": self.option_id,
        }


@dataclass(frozen=True, slots=True)
class PolicyPlan:
    """A complete root-contingent tree or a fail-closed HOLD decision."""

    decision: str
    mode: PolicyMode
    horizon: int
    variant: ScenarioVariant
    case_id: str
    case_sha256: str
    closure_predicate_sha256: str
    evidence_root_sha256: str
    scenario_set_sha256: str
    option_set_sha256: str
    sequence_model_sha256: str
    evidence_admission_sha256: str
    evidence_invariant_report_sha256: str
    counterexample_registry_sha256: str
    counterexample_search_sha256: str
    counterexample_baseline_state_sha256: str
    evaluator_search_contract_release_sha256: str
    evidence_failure_domains: tuple[str, ...]
    scenario_failure_domains: tuple[str, ...]
    failure_domain_binding_sha256: str
    root_provenance_sha256: str
    root_option_id: str | None
    branches: tuple[ContingentBranch, ...]
    hold_risk: float
    plan_risk: float | None
    conservative_voi: float | None
    minimum_conservative_voi: float
    terminal_closure_guaranteed: bool
    rejected_options: tuple[RejectedOption, ...]
    reason: str

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("policy plan case_id must be non-empty")
        for name, digest in (
            ("case_sha256", self.case_sha256),
            ("closure_predicate_sha256", self.closure_predicate_sha256),
            ("evidence_root_sha256", self.evidence_root_sha256),
            ("scenario_set_sha256", self.scenario_set_sha256),
            ("option_set_sha256", self.option_set_sha256),
            ("sequence_model_sha256", self.sequence_model_sha256),
            ("evidence_admission_sha256", self.evidence_admission_sha256),
            ("evidence_invariant_report_sha256", self.evidence_invariant_report_sha256),
            ("counterexample_registry_sha256", self.counterexample_registry_sha256),
            ("counterexample_search_sha256", self.counterexample_search_sha256),
            ("counterexample_baseline_state_sha256", self.counterexample_baseline_state_sha256),
            (
                "evaluator_search_contract_release_sha256",
                self.evaluator_search_contract_release_sha256,
            ),
            ("failure_domain_binding_sha256", self.failure_domain_binding_sha256),
            ("root_provenance_sha256", self.root_provenance_sha256),
        ):
            require_sha256(name, digest)
        if self.evidence_root_sha256 != self.root_provenance_sha256:
            raise ValueError("policy evidence root and root provenance must match")
        evidence_domains = tuple(sorted(set(self.evidence_failure_domains)))
        scenario_domains = tuple(sorted(set(self.scenario_failure_domains)))
        if evidence_domains != self.evidence_failure_domains:
            raise ValueError("policy evidence failure domains must be sorted and unique")
        if scenario_domains != self.scenario_failure_domains:
            raise ValueError("policy scenario failure domains must be sorted and unique")
        if not set(evidence_domains).issubset(scenario_domains):
            raise ValueError("policy scenarios do not cover evidence failure domains")
        if not isfinite(self.minimum_conservative_voi) or self.minimum_conservative_voi < 0:
            raise ValueError("policy minimum conservative VoE must be finite and non-negative")
        if not isinstance(self.terminal_closure_guaranteed, bool):
            raise TypeError("terminal_closure_guaranteed must be boolean")

    @property
    def plan_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @property
    def is_hold(self) -> bool:
        return self.decision == "HOLD"

    def next_option(
        self,
        *,
        observation: str | None = None,
        provenance_sha256: str,
        local_veto: bool = False,
    ) -> str | None:
        """Resolve the precommitted branch, invalidating drift or a local veto."""
        if local_veto or provenance_sha256 != self.root_provenance_sha256 or self.is_hold:
            return None
        if observation is None:
            return self.root_option_id
        for branch in self.branches:
            if branch.observation == observation:
                return branch.option_id
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "xrd-rb-voe-policy-plan-v5",
            "decision": self.decision,
            "mode": self.mode.value,
            "horizon": self.horizon,
            "variant": self.variant.value,
            "case_id": self.case_id,
            "case_sha256": self.case_sha256,
            "closure_predicate_sha256": self.closure_predicate_sha256,
            "evidence_root_sha256": self.evidence_root_sha256,
            "scenario_set_sha256": self.scenario_set_sha256,
            "option_set_sha256": self.option_set_sha256,
            "sequence_model_sha256": self.sequence_model_sha256,
            "evidence_admission_sha256": self.evidence_admission_sha256,
            "evidence_invariant_report_sha256": self.evidence_invariant_report_sha256,
            "counterexample_registry_sha256": self.counterexample_registry_sha256,
            "counterexample_search_sha256": self.counterexample_search_sha256,
            "counterexample_baseline_state_sha256": (self.counterexample_baseline_state_sha256),
            "evaluator_search_contract_release_sha256": (self.evaluator_search_contract_release_sha256),
            "evidence_failure_domains": list(self.evidence_failure_domains),
            "scenario_failure_domains": list(self.scenario_failure_domains),
            "failure_domain_binding_sha256": self.failure_domain_binding_sha256,
            "root_provenance_sha256": self.root_provenance_sha256,
            "root_option_id": self.root_option_id,
            "branches": [branch.to_dict() for branch in self.branches],
            "hold_risk": self.hold_risk,
            "plan_risk": self.plan_risk,
            "conservative_voi": self.conservative_voi,
            "minimum_conservative_voi": self.minimum_conservative_voi,
            "terminal_closure_guaranteed": self.terminal_closure_guaranteed,
            "rejected_options": [item.to_dict() for item in self.rejected_options],
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class _CandidateTree:
    root_option_id: str
    branches: tuple[ContingentBranch, ...]
    risk: float

    @property
    def tie_key(self) -> tuple[float, str, tuple[tuple[str, str], ...]]:
        return (
            self.risk,
            self.root_option_id,
            tuple((branch.observation, branch.option_id or "") for branch in self.branches),
        )


def _option_map(options: tuple[EvidenceOption, ...]) -> dict[str, EvidenceOption]:
    normalized = tuple(sorted(options, key=lambda item: item.option_id))
    identifiers = tuple(item.option_id for item in normalized)
    if not normalized or len(set(identifiers)) != len(identifiers):
        raise ValueError("policy options must be non-empty with unique ids")
    return {item.option_id: item for item in normalized}


def _validate_coverage(options: tuple[EvidenceOption, ...], scenario_set: JointScenarioSet) -> None:
    expected = set(scenario_set.scenario_ids)
    for option in options:
        actual = {outcome.scenario_id for outcome in option.outcomes}
        if actual != expected:
            raise ValueError(f"option {option.option_id} outcomes must exactly cover sealed root scenarios")


def _risk(
    scenario_set: JointScenarioSet,
    variant: ScenarioVariant,
    scenario_losses: dict[str, float],
) -> float:
    return scenario_set.risk(scenario_losses, variant)


def _option_set_sha256(options: tuple[EvidenceOption, ...]) -> str:
    return canonical_sha256([item.to_dict() for item in sorted(options, key=lambda option: option.option_id)])


def _validate_case_bindings(
    *,
    experiment_case: ExperimentCase,
    options: tuple[EvidenceOption, ...],
    closure_predicate: ClosurePredicate,
) -> None:
    if not isinstance(experiment_case, ExperimentCase):
        raise TypeError("experiment_case must be an ExperimentCase")
    if experiment_case.closure_predicate_id != closure_predicate.predicate_id:
        raise ValueError("closure predicate id does not match experiment case")
    case_atoms = tuple(sorted(set(experiment_case.required_failure_atoms)))
    if len(case_atoms) != len(experiment_case.required_failure_atoms):
        raise ValueError("experiment case failure atoms must be unique")
    if case_atoms != closure_predicate.required_failure_atoms:
        raise ValueError("closure predicate atoms must exactly equal the case failure core")
    registered_atoms = set(FAILURE_CORE_REASON_CODES)
    if any(atom not in registered_atoms for atom in case_atoms):
        raise ValueError("experiment case contains an unregistered failure-core atom")

    allowed = tuple(experiment_case.allowed_options)
    if len(set(allowed)) != len(allowed):
        raise ValueError("experiment case allowed options must be unique")
    for option_id in allowed:
        require_option_id(option_id)
    supplied = {option.option_id for option in options}
    for option_id in supplied:
        require_option_id(option_id)
    if not supplied.issubset(set(allowed)):
        raise ValueError("policy options must be a subset of case allowed options")


def _validate_sequence_model_references(
    *,
    sequence_model: SequenceOutcomeModel,
    option_by_id: dict[str, EvidenceOption],
    scenario_set: JointScenarioSet,
) -> None:
    scenario_ids = set(scenario_set.scenario_ids)
    for sequence_outcome in sequence_model.outcomes:
        root = option_by_id.get(sequence_outcome.root_option_id)
        if root is None or sequence_outcome.second_option_id not in option_by_id:
            raise ValueError("sequence outcome references an option outside the policy option set")
        if sequence_outcome.scenario_id not in scenario_ids:
            raise ValueError("sequence outcome references an unsealed root scenario")
        expected_observation = root.outcome_for(sequence_outcome.scenario_id).observation
        if sequence_outcome.root_observation != expected_observation:
            raise ValueError("sequence outcome observation does not match the root option outcome")


def _build_trees(
    *,
    root: EvidenceOption,
    feasible: tuple[EvidenceOption, ...],
    scenario_set: JointScenarioSet,
    predicate: ClosurePredicate,
    sequence_model: SequenceOutcomeModel,
    config: PolicyConfig,
    fixed_branches: dict[str, str | None] | None = None,
    first_feasible_branches: bool = False,
) -> tuple[_CandidateTree, ...]:
    observations: dict[str, list[str]] = {}
    for scenario_id in scenario_set.scenario_ids:
        outcome = root.outcome_for(scenario_id)
        if not predicate.is_closed(outcome.closed_failure_atoms):
            observations.setdefault(outcome.observation, []).append(scenario_id)

    if config.horizon == 1:
        if fixed_branches:
            return ()
        losses = {
            scenario_id: root.cost + root.outcome_for(scenario_id).residual_loss
            for scenario_id in scenario_set.scenario_ids
        }
        return (
            _CandidateTree(
                root.option_id,
                (),
                _risk(scenario_set, config.variant, losses),
            ),
        )

    branch_groups: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
    for observation in sorted(observations):
        scenario_ids = tuple(sorted(observations[observation]))
        viable: list[str] = []
        for option in feasible:
            if option.option_id == root.option_id and not root.repeatable:
                continue
            try:
                sequence_outcomes = tuple(
                    sequence_model.outcome_for(
                        root_option_id=root.option_id,
                        root_observation=observation,
                        second_option_id=option.option_id,
                        scenario_id=scenario_id,
                    )
                    for scenario_id in scenario_ids
                )
            except KeyError:
                continue
            if all(
                predicate.is_closed(outcome.terminal_closed_failure_atoms) for outcome in sequence_outcomes
            ):
                viable.append(option.option_id)
        if fixed_branches is not None:
            forced = fixed_branches.get(observation)
            if forced is None or forced not in viable:
                return ()
            choices = (forced,)
        elif first_feasible_branches:
            if not viable:
                return ()
            choices = (viable[0],)
        else:
            choices = tuple(viable)
        if not choices:
            return ()
        branch_groups.append((observation, scenario_ids, choices))

    trees: list[_CandidateTree] = []
    combinations = product(*(group[2] for group in branch_groups)) if branch_groups else [()]
    for selected_ids in combinations:
        branches = tuple(
            ContingentBranch(observation, scenario_ids, selected_id)
            for (observation, scenario_ids, _), selected_id in zip(branch_groups, selected_ids, strict=True)
        )
        branch_by_observation = {branch.observation: branch.option_id for branch in branches}
        losses: dict[str, float] = {}
        for scenario_id in scenario_set.scenario_ids:
            root_outcome = root.outcome_for(scenario_id)
            if predicate.is_closed(root_outcome.closed_failure_atoms):
                losses[scenario_id] = root.cost + root_outcome.residual_loss
                continue
            branch_id = branch_by_observation[root_outcome.observation]
            branch = next(option for option in feasible if option.option_id == branch_id)
            sequence_outcome = sequence_model.outcome_for(
                root_option_id=root.option_id,
                root_observation=root_outcome.observation,
                second_option_id=branch.option_id,
                scenario_id=scenario_id,
            )
            losses[scenario_id] = root.cost + branch.cost + sequence_outcome.terminal_residual_loss
        trees.append(
            _CandidateTree(
                root.option_id,
                branches,
                _risk(scenario_set, config.variant, losses),
            )
        )
    return tuple(trees)


def plan_policy(
    *,
    experiment_case: ExperimentCase,
    scenario_set: JointScenarioSet,
    options: tuple[EvidenceOption, ...],
    closure_predicate: ClosurePredicate,
    evidence_admission: PolicyEvidenceAdmission,
    sequence_model: SequenceOutcomeModel | None = None,
    config: PolicyConfig | None = None,
) -> PolicyPlan:
    """Choose one complete contingent tree from a sealed root scenario set."""
    if config is None:
        config = PolicyConfig()
    if not isinstance(evidence_admission, PolicyEvidenceAdmission):
        raise TypeError("evidence_admission must be a PolicyEvidenceAdmission")
    admission_report = evidence_admission.revalidate(experiment_case)
    (
        evidence_failure_domains,
        scenario_failure_domains,
        failure_domain_binding_sha256,
    ) = evidence_admission.bind_scenario_failure_domains(scenario_set)
    root_provenance_sha256 = admission_report.evidence_root_sha256
    if sequence_model is None:
        if config.horizon == 2:
            raise ValueError("H=2 policy planning requires an explicit sequence outcome model")
        sequence_model = SequenceOutcomeModel("h1-no-sequence", ())
    _validate_case_bindings(
        experiment_case=experiment_case,
        options=options,
        closure_predicate=closure_predicate,
    )
    option_by_id = _option_map(options)
    _validate_coverage(options, scenario_set)
    _validate_sequence_model_references(
        sequence_model=sequence_model,
        option_by_id=option_by_id,
        scenario_set=scenario_set,
    )
    option_set_sha256 = _option_set_sha256(options)
    sequence_model_sha256 = sequence_model.content_sha256
    rejected = tuple(
        RejectedOption(option.option_id, option.failure_codes)
        for option in sorted(options, key=lambda item: item.option_id)
        if not option.feasible
    )
    feasible = tuple(option for option in sorted(options, key=lambda item: item.option_id) if option.feasible)
    hold_risk = scenario_set.expected_hold_loss(config.variant)

    def hold(reason: str) -> PolicyPlan:
        return PolicyPlan(
            decision="HOLD",
            mode=config.mode,
            horizon=config.horizon,
            variant=config.variant,
            case_id=experiment_case.case_id,
            case_sha256=experiment_case.content_sha256,
            closure_predicate_sha256=closure_predicate.content_sha256,
            evidence_root_sha256=experiment_case.evidence_root_sha256,
            scenario_set_sha256=scenario_set.content_sha256,
            option_set_sha256=option_set_sha256,
            sequence_model_sha256=sequence_model_sha256,
            evidence_admission_sha256=evidence_admission.content_sha256,
            evidence_invariant_report_sha256=admission_report.invariant_report_sha256,
            counterexample_registry_sha256=admission_report.counterexample_registry_sha256,
            counterexample_search_sha256=admission_report.counterexample_search_sha256,
            counterexample_baseline_state_sha256=(admission_report.counterexample_baseline_state_sha256),
            evaluator_search_contract_release_sha256=(
                admission_report.evaluator_search_contract_release_sha256
            ),
            evidence_failure_domains=evidence_failure_domains,
            scenario_failure_domains=scenario_failure_domains,
            failure_domain_binding_sha256=failure_domain_binding_sha256,
            root_provenance_sha256=root_provenance_sha256,
            root_option_id=None,
            branches=(),
            hold_risk=hold_risk,
            plan_risk=None,
            conservative_voi=None,
            minimum_conservative_voi=config.minimum_conservative_voi,
            terminal_closure_guaranteed=False,
            rejected_options=rejected,
            reason=reason,
        )

    if config.mode is PolicyMode.ALWAYS_HOLD:
        return hold("ALWAYS_HOLD_BASELINE")
    if not feasible:
        return hold("NO_FEASIBLE_OPTION")

    candidates: list[_CandidateTree] = []
    if config.mode is PolicyMode.FIXED:
        root = option_by_id.get(config.fixed_root_option_id or "")
        if root is None or not root.feasible:
            return hold("FIXED_OPTION_UNAVAILABLE_OR_UNSAFE")
        trees = _build_trees(
            root=root,
            feasible=feasible,
            scenario_set=scenario_set,
            predicate=closure_predicate,
            sequence_model=sequence_model,
            config=config,
            fixed_branches={item.observation: item.option_id for item in config.fixed_branches},
        )
        if not trees:
            return hold("FIXED_TREE_INCOMPLETE")
        candidates.extend(trees)
    elif config.mode in {PolicyMode.ALWAYS_PERMIT, PolicyMode.FIRST_FEASIBLE}:
        trees = _build_trees(
            root=feasible[0],
            feasible=feasible,
            scenario_set=scenario_set,
            predicate=closure_predicate,
            sequence_model=sequence_model,
            config=config,
            first_feasible_branches=True,
        )
        if not trees:
            return hold("FIRST_FEASIBLE_TREE_INCOMPLETE")
        candidates.extend(trees)
    else:
        for root in feasible:
            trees = _build_trees(
                root=root,
                feasible=feasible,
                scenario_set=scenario_set,
                predicate=closure_predicate,
                sequence_model=sequence_model,
                config=config,
            )
            candidates.extend(trees)
        if not candidates:
            return hold("NO_COMPLETE_CONTINGENT_TREE")

    selected = min(candidates, key=lambda item: item.tie_key)
    voi = hold_risk - selected.risk
    diagnostic = config.mode in {
        PolicyMode.ALWAYS_PERMIT,
        PolicyMode.FIRST_FEASIBLE,
        PolicyMode.FIXED,
    }
    if not diagnostic and voi <= config.minimum_conservative_voi:
        return hold("NONPOSITIVE_CONSERVATIVE_VOE")
    return PolicyPlan(
        decision="NEXT_EVIDENCE",
        mode=config.mode,
        horizon=config.horizon,
        variant=config.variant,
        case_id=experiment_case.case_id,
        case_sha256=experiment_case.content_sha256,
        closure_predicate_sha256=closure_predicate.content_sha256,
        evidence_root_sha256=experiment_case.evidence_root_sha256,
        scenario_set_sha256=scenario_set.content_sha256,
        option_set_sha256=option_set_sha256,
        sequence_model_sha256=sequence_model_sha256,
        evidence_admission_sha256=evidence_admission.content_sha256,
        evidence_invariant_report_sha256=admission_report.invariant_report_sha256,
        counterexample_registry_sha256=admission_report.counterexample_registry_sha256,
        counterexample_search_sha256=admission_report.counterexample_search_sha256,
        counterexample_baseline_state_sha256=(admission_report.counterexample_baseline_state_sha256),
        evaluator_search_contract_release_sha256=(admission_report.evaluator_search_contract_release_sha256),
        evidence_failure_domains=evidence_failure_domains,
        scenario_failure_domains=scenario_failure_domains,
        failure_domain_binding_sha256=failure_domain_binding_sha256,
        root_provenance_sha256=root_provenance_sha256,
        root_option_id=selected.root_option_id,
        branches=selected.branches,
        hold_risk=hold_risk,
        plan_risk=selected.risk,
        conservative_voi=voi,
        minimum_conservative_voi=config.minimum_conservative_voi,
        terminal_closure_guaranteed=(
            config.horizon == 2
            or all(
                closure_predicate.is_closed(
                    option_by_id[selected.root_option_id].outcome_for(scenario_id).closed_failure_atoms
                )
                for scenario_id in scenario_set.scenario_ids
            )
        ),
        rejected_options=rejected,
        reason="DIAGNOSTIC_BASELINE" if diagnostic else "POSITIVE_CONSERVATIVE_VOE",
    )


def validate_policy_plan_semantics(
    *,
    plan: PolicyPlan,
    scenario_set: JointScenarioSet,
    options: tuple[EvidenceOption, ...],
    closure_predicate: ClosurePredicate,
    sequence_model: SequenceOutcomeModel,
) -> None:
    """Recompute branch coverage, risk, VoE, and optimized-tree selection."""
    option_by_id = _option_map(options)
    _validate_coverage(options, scenario_set)
    _validate_sequence_model_references(
        sequence_model=sequence_model,
        option_by_id=option_by_id,
        scenario_set=scenario_set,
    )
    expected_digests = {
        "scenario set": (plan.scenario_set_sha256, scenario_set.content_sha256),
        "option set": (plan.option_set_sha256, _option_set_sha256(options)),
        "closure predicate": (
            plan.closure_predicate_sha256,
            closure_predicate.content_sha256,
        ),
        "sequence model": (plan.sequence_model_sha256, sequence_model.content_sha256),
    }
    for name, (actual, expected) in expected_digests.items():
        if actual != expected:
            raise ValueError(f"policy {name} digest does not match the sealed problem")
    if plan.scenario_failure_domains != scenario_set.failure_domains:
        raise ValueError("policy scenario failure domains differ from the sealed scenarios")
    expected_failure_domain_binding = _failure_domain_binding_sha256(
        evidence_root_sha256=plan.evidence_root_sha256,
        scenario_set_sha256=scenario_set.content_sha256,
        evidence_failure_domains=plan.evidence_failure_domains,
        scenario_failure_domains=scenario_set.failure_domains,
    )
    if plan.failure_domain_binding_sha256 != expected_failure_domain_binding:
        raise ValueError("policy failure-domain binding is not reproducible")

    rejected = tuple(
        RejectedOption(option.option_id, option.failure_codes)
        for option in sorted(options, key=lambda item: item.option_id)
        if not option.feasible
    )
    if plan.rejected_options != rejected:
        raise ValueError("policy rejected options differ from sealed hard gates")
    hold_risk = scenario_set.expected_hold_loss(plan.variant)
    if not isclose(plan.hold_risk, hold_risk, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("policy hold risk is not reproducible")
    if plan.is_hold:
        if (
            plan.root_option_id is not None
            or plan.branches
            or plan.plan_risk is not None
            or plan.conservative_voi is not None
            or plan.terminal_closure_guaranteed
        ):
            raise ValueError("HOLD policy carries an executable or terminal-closure claim")
        return
    if plan.decision != "NEXT_EVIDENCE" or plan.root_option_id is None:
        raise ValueError("non-HOLD policy must be a NEXT_EVIDENCE tree")
    if plan.mode is PolicyMode.ALWAYS_HOLD:
        raise ValueError("ALWAYS_HOLD mode cannot carry a NEXT_EVIDENCE tree")

    feasible = tuple(option for option in sorted(options, key=lambda item: item.option_id) if option.feasible)
    root = option_by_id.get(plan.root_option_id)
    if root is None or not root.feasible:
        raise ValueError("policy root option is absent or infeasible")
    config = PolicyConfig(
        horizon=plan.horizon,
        variant=plan.variant,
        mode=plan.mode,
        minimum_conservative_voi=plan.minimum_conservative_voi,
        fixed_root_option_id=(plan.root_option_id if plan.mode is PolicyMode.FIXED else None),
        fixed_branches=(
            tuple(FixedBranch(branch.observation, branch.option_id) for branch in plan.branches)
            if plan.mode is PolicyMode.FIXED
            else ()
        ),
    )
    if plan.mode is PolicyMode.OPTIMIZED:
        candidates = tuple(
            tree
            for candidate_root in feasible
            for tree in _build_trees(
                root=candidate_root,
                feasible=feasible,
                scenario_set=scenario_set,
                predicate=closure_predicate,
                sequence_model=sequence_model,
                config=config,
            )
        )
        if not candidates:
            raise ValueError("optimized policy has no complete candidate tree")
        expected_tree = min(candidates, key=lambda item: item.tie_key)
    elif plan.mode in {PolicyMode.ALWAYS_PERMIT, PolicyMode.FIRST_FEASIBLE}:
        if root.option_id != feasible[0].option_id:
            raise ValueError("first-feasible diagnostic policy changed its root option")
        candidates = _build_trees(
            root=root,
            feasible=feasible,
            scenario_set=scenario_set,
            predicate=closure_predicate,
            sequence_model=sequence_model,
            config=config,
            first_feasible_branches=True,
        )
        if not candidates:
            raise ValueError("first-feasible diagnostic policy has no complete tree")
        expected_tree = min(candidates, key=lambda item: item.tie_key)
    elif plan.mode is PolicyMode.FIXED:
        candidates = _build_trees(
            root=root,
            feasible=feasible,
            scenario_set=scenario_set,
            predicate=closure_predicate,
            sequence_model=sequence_model,
            config=config,
            fixed_branches={branch.observation: branch.option_id for branch in plan.branches},
        )
        if not candidates:
            raise ValueError("fixed policy tree is incomplete")
        expected_tree = min(candidates, key=lambda item: item.tie_key)
    else:
        raise ValueError(f"unsupported executable policy mode: {plan.mode.value}")

    if plan.root_option_id != expected_tree.root_option_id or plan.branches != expected_tree.branches:
        raise ValueError("policy tree differs from recomputed branch coverage or optimum")
    if plan.plan_risk is None or not isclose(
        plan.plan_risk,
        expected_tree.risk,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("policy plan risk is not reproducible")
    expected_voi = hold_risk - expected_tree.risk
    if plan.conservative_voi is None or not isclose(
        plan.conservative_voi,
        expected_voi,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("policy conservative VoE is not reproducible")

    terminal_closure = True
    branch_by_observation = {branch.observation: branch.option_id for branch in plan.branches}
    for scenario_id in scenario_set.scenario_ids:
        root_outcome = root.outcome_for(scenario_id)
        if closure_predicate.is_closed(root_outcome.closed_failure_atoms):
            continue
        if plan.horizon == 1:
            terminal_closure = False
            continue
        branch_id = branch_by_observation.get(root_outcome.observation)
        if branch_id is None:
            terminal_closure = False
            continue
        terminal = sequence_model.outcome_for(
            root_option_id=root.option_id,
            root_observation=root_outcome.observation,
            second_option_id=branch_id,
            scenario_id=scenario_id,
        )
        if not closure_predicate.is_closed(terminal.terminal_closed_failure_atoms):
            terminal_closure = False
    if plan.terminal_closure_guaranteed is not terminal_closure:
        raise ValueError("policy terminal-closure flag is not reproducible")
    diagnostic = plan.mode in {
        PolicyMode.ALWAYS_PERMIT,
        PolicyMode.FIRST_FEASIBLE,
        PolicyMode.FIXED,
    }
    expected_reason = "DIAGNOSTIC_BASELINE" if diagnostic else "POSITIVE_CONSERVATIVE_VOE"
    if plan.reason != expected_reason:
        raise ValueError("policy reason does not match its mode")
    if not diagnostic and expected_voi <= plan.minimum_conservative_voi:
        raise ValueError("optimized policy does not clear its conservative VoE threshold")
