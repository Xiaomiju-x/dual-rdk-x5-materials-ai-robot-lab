"""Pure, deterministic execution of sealed RB-VoE policy plans.

Simulation is counterfactual evidence only. It can establish that a modeled
policy tree closes a modeled failure core; it never proves physical closure,
touches hardware, or grants execution authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

from rb_voe.contracts.canonical import canonical_sha256, require_sha256
from rb_voe.contracts.models import EvidenceSource
from rb_voe.core.options import (
    ClosurePredicate,
    EvidenceOption,
    SequenceOutcomeModel,
)
from rb_voe.core.policy import PolicyPlan, validate_policy_plan_semantics
from rb_voe.core.scenarios import JointScenarioSet, ScenarioVariant


@dataclass(frozen=True, slots=True)
class SimulatedOption:
    """An evidence option plus sealed per-scenario execution durations."""

    option: EvidenceOption
    duration_ms_by_scenario: Mapping[str, int]

    def __post_init__(self) -> None:
        expected = {outcome.scenario_id for outcome in self.option.outcomes}
        actual = set(self.duration_ms_by_scenario)
        if actual != expected:
            raise ValueError("duration map must exactly cover option scenarios")
        for scenario_id, duration_ms in self.duration_ms_by_scenario.items():
            if (
                not scenario_id
                or isinstance(duration_ms, bool)
                or not isinstance(duration_ms, int)
                or duration_ms < 0
            ):
                raise ValueError("scenario durations must be non-negative integers")

    @property
    def option_id(self) -> str:
        return self.option.option_id

    @property
    def content_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "option": self.option.to_dict(),
            "duration_ms_by_scenario": {
                key: self.duration_ms_by_scenario[key] for key in sorted(self.duration_ms_by_scenario)
            },
        }


def _option_set_sha256(options: tuple[SimulatedOption, ...]) -> str:
    return canonical_sha256(
        [item.option.to_dict() for item in sorted(options, key=lambda item: item.option_id)]
    )


@dataclass(frozen=True, slots=True)
class FixedOptionSelector:
    """Legacy inert descriptor retained only for import compatibility.

    It is deliberately not callable and ``run_episode`` never accepts it. New
    simulation requests must carry a hashed ``PolicyPlan``.
    """

    root_option_id: str | None
    branches: Mapping[str, str | None]


@dataclass(frozen=True, slots=True)
class SimulationRequest:
    """A sealed request that can execute only its bound ``PolicyPlan``."""

    episode_id: str
    seed: int | str
    horizon: int
    scenario_set: JointScenarioSet
    scenario_variant: ScenarioVariant
    options: tuple[SimulatedOption, ...]
    closure_predicate: ClosurePredicate
    sequence_model: SequenceOutcomeModel
    policy_plan: PolicyPlan
    pinned_policy_plan_sha256: str
    pinned_sequence_model_sha256: str
    root_provenance_sha256: str
    local_veto_steps: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must be non-empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, str)):
            raise TypeError("seed must be an integer or string")
        if self.horizon not in {1, 2}:
            raise ValueError("simulation horizon must be one or two")
        if not isinstance(self.scenario_variant, ScenarioVariant):
            object.__setattr__(self, "scenario_variant", ScenarioVariant(self.scenario_variant))
        require_sha256("root_provenance_sha256", self.root_provenance_sha256)
        require_sha256("pinned_policy_plan_sha256", self.pinned_policy_plan_sha256)
        require_sha256("pinned_sequence_model_sha256", self.pinned_sequence_model_sha256)

        normalized = tuple(sorted(self.options, key=lambda item: item.option_id))
        identifiers = tuple(item.option_id for item in normalized)
        if not normalized or len(set(identifiers)) != len(identifiers):
            raise ValueError("simulation options must be non-empty with unique ids")
        expected_scenarios = set(self.scenario_set.scenario_ids)
        for item in normalized:
            actual_scenarios = {outcome.scenario_id for outcome in item.option.outcomes}
            if actual_scenarios != expected_scenarios:
                raise ValueError("each option must cover the sealed scenario set exactly")
        object.__setattr__(self, "options", normalized)

        veto_steps = tuple(sorted(set(self.local_veto_steps)))
        if len(veto_steps) != len(self.local_veto_steps) or any(
            isinstance(step, bool) or not isinstance(step, int) or step < 1 or step > self.horizon
            for step in veto_steps
        ):
            raise ValueError("local veto steps must be unique valid policy steps")
        object.__setattr__(self, "local_veto_steps", veto_steps)

        if self.policy_plan.horizon != self.horizon:
            raise ValueError("simulation horizon does not match policy plan")
        if self.policy_plan.variant is not self.scenario_variant:
            raise ValueError("simulation variant does not match policy plan")
        if self.policy_plan.scenario_set_sha256 != self.scenario_set.content_sha256:
            raise ValueError("simulation scenario set does not match policy plan")
        if self.policy_plan.closure_predicate_sha256 != self.closure_predicate.content_sha256:
            raise ValueError("simulation closure predicate does not match policy plan")
        if self.policy_plan.option_set_sha256 != _option_set_sha256(normalized):
            raise ValueError("simulation option set does not match policy plan")
        if self.pinned_policy_plan_sha256 != self.policy_plan.plan_sha256:
            raise ValueError("simulation policy plan drifted from its pinned digest")
        if self.pinned_sequence_model_sha256 != self.sequence_model.content_sha256:
            raise ValueError("simulation sequence model drifted from its pinned digest")
        if self.policy_plan.sequence_model_sha256 != self.sequence_model.content_sha256:
            raise ValueError("simulation sequence model does not match policy plan")
        if self.root_provenance_sha256 != self.policy_plan.root_provenance_sha256:
            raise ValueError("simulation provenance drifted from policy plan")
        validate_policy_plan_semantics(
            plan=self.policy_plan,
            scenario_set=self.scenario_set,
            options=tuple(item.option for item in normalized),
            closure_predicate=self.closure_predicate,
            sequence_model=self.sequence_model,
        )

        selected_ids = {
            option_id
            for option_id in (
                self.policy_plan.root_option_id,
                *(branch.option_id for branch in self.policy_plan.branches),
            )
            if option_id is not None
        }
        if not selected_ids.issubset(set(identifiers)):
            raise ValueError("policy plan references an option absent from the simulation")

    @property
    def policy_plan_sha256(self) -> str:
        return self.policy_plan.plan_sha256

    @property
    def content_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "xrd-rb-voe-simulation-request-v4",
            "episode_id": self.episode_id,
            "seed": self.seed,
            "horizon": self.horizon,
            "scenario_set": self.scenario_set.to_dict(),
            "scenario_set_sha256": self.scenario_set.content_sha256,
            "scenario_variant": self.scenario_variant.value,
            "options": [item.to_dict() for item in self.options],
            "closure_predicate": self.closure_predicate.to_dict(),
            "closure_predicate_sha256": self.closure_predicate.content_sha256,
            "sequence_model": self.sequence_model.to_dict(),
            "sequence_model_sha256": self.sequence_model.content_sha256,
            "policy_plan_sha256": self.policy_plan_sha256,
            "pinned_policy_plan_sha256": self.pinned_policy_plan_sha256,
            "pinned_sequence_model_sha256": self.pinned_sequence_model_sha256,
            "root_provenance_sha256": self.root_provenance_sha256,
            "local_veto_steps": list(self.local_veto_steps),
        }


@dataclass(frozen=True, slots=True)
class SimulatedObservation:
    step_index: int
    option_id: str
    root_scenario_id: str
    duration_ms: int
    observation: str
    residual_loss: float
    closed_failure_atoms: tuple[str, ...]
    cumulative_modeled_closed_failure_atoms: tuple[str, ...]
    modeled_closure_satisfied: bool
    sequence_outcome_sha256: str | None = None
    provenance: EvidenceSource = EvidenceSource.SIMULATED_COUNTERFACTUAL
    hardware_touch: bool = False
    physical_closure_proven: bool = False

    def __post_init__(self) -> None:
        if self.step_index not in {1, 2}:
            raise ValueError("simulated observation step must be one or two")
        if not self.option_id or not self.root_scenario_id or not self.observation:
            raise ValueError("simulated observation identity fields must be non-empty")
        if (
            isinstance(self.duration_ms, bool)
            or not isinstance(self.duration_ms, int)
            or self.duration_ms < 0
        ):
            raise ValueError("duration_ms must be non-negative")
        if not isfinite(self.residual_loss) or self.residual_loss < 0:
            raise ValueError("residual_loss must be finite and non-negative")
        if self.step_index == 1 and self.sequence_outcome_sha256 is not None:
            raise ValueError("step one cannot claim a sequence terminal effect")
        if self.step_index == 2:
            if self.sequence_outcome_sha256 is None:
                raise ValueError("step two requires a sequence terminal-effect digest")
            require_sha256("sequence_outcome_sha256", self.sequence_outcome_sha256)
        if self.provenance is not EvidenceSource.SIMULATED_COUNTERFACTUAL:
            raise ValueError("simulation provenance must be SIMULATED_COUNTERFACTUAL")
        if self.hardware_touch or self.physical_closure_proven:
            raise ValueError("simulation cannot claim hardware contact or physical closure")

    @property
    def closure_satisfied(self) -> bool:
        """Compatibility alias; the value remains modeled-only."""
        return self.modeled_closure_satisfied

    def to_dict(self) -> dict[str, object]:
        return {
            "step_index": self.step_index,
            "option_id": self.option_id,
            "root_scenario_id": self.root_scenario_id,
            "duration_ms": self.duration_ms,
            "observation": self.observation,
            "residual_loss": self.residual_loss,
            "closed_failure_atoms": list(self.closed_failure_atoms),
            "cumulative_modeled_closed_failure_atoms": list(self.cumulative_modeled_closed_failure_atoms),
            "modeled_closure_satisfied": self.modeled_closure_satisfied,
            "sequence_outcome_sha256": self.sequence_outcome_sha256,
            "terminal_residual_source": (
                "SEQUENCE_OUTCOME_MODEL" if self.step_index == 2 else "SINGLE_OPTION_OUTCOME"
            ),
            "closure_semantics": "SIMULATED_MODELED_ONLY",
            "physical_closure_proven": self.physical_closure_proven,
            "provenance": self.provenance.value,
            "hardware_touch": self.hardware_touch,
        }


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    episode_id: str
    request_sha256: str
    policy_plan_sha256: str
    sequence_model_sha256: str
    root_provenance_sha256: str
    scenario_set_sha256: str
    root_scenario_id: str
    root_scenario_draw_sha256: str
    root_scenario_selection_count: int
    observations: tuple[SimulatedObservation, ...]
    cumulative_duration_ms: int
    modeled_closed_failure_atoms: tuple[str, ...]
    modeled_closure_satisfied: bool
    termination_reason: str
    evidence_source: EvidenceSource = EvidenceSource.SIMULATED_COUNTERFACTUAL
    hardware_touch: bool = False
    execution_authority: bool = False
    physical_closure_proven: bool = False
    physical_risk_denominator_increment: int = 0

    def __post_init__(self) -> None:
        require_sha256("request_sha256", self.request_sha256)
        require_sha256("policy_plan_sha256", self.policy_plan_sha256)
        require_sha256("sequence_model_sha256", self.sequence_model_sha256)
        require_sha256("root_provenance_sha256", self.root_provenance_sha256)
        require_sha256("scenario_set_sha256", self.scenario_set_sha256)
        require_sha256("root_scenario_draw_sha256", self.root_scenario_draw_sha256)
        if self.root_scenario_selection_count != 1:
            raise ValueError("a simulation must select exactly one root scenario")
        if self.evidence_source is not EvidenceSource.SIMULATED_COUNTERFACTUAL:
            raise ValueError("simulation evidence source is fixed")
        if self.hardware_touch or self.execution_authority or self.physical_closure_proven:
            raise ValueError("simulation cannot create hardware, execution, or physical proof")
        if (
            isinstance(self.physical_risk_denominator_increment, bool)
            or not isinstance(self.physical_risk_denominator_increment, int)
            or self.physical_risk_denominator_increment != 0
        ):
            raise ValueError("simulation cannot count toward the physical risk denominator")
        if any(item.root_scenario_id != self.root_scenario_id for item in self.observations):
            raise ValueError("root scenario changed during the episode")

    @property
    def closed_failure_atoms(self) -> tuple[str, ...]:
        """Compatibility alias; these atoms are modeled, not physical proof."""
        return self.modeled_closed_failure_atoms

    @property
    def closure_satisfied(self) -> bool:
        """Compatibility alias; the value remains modeled-only."""
        return self.modeled_closure_satisfied

    @property
    def episode_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "xrd-rb-voe-simulated-episode-v3",
            "episode_id": self.episode_id,
            "request_sha256": self.request_sha256,
            "policy_plan_sha256": self.policy_plan_sha256,
            "sequence_model_sha256": self.sequence_model_sha256,
            "root_provenance_sha256": self.root_provenance_sha256,
            "scenario_set_sha256": self.scenario_set_sha256,
            "root_scenario_id": self.root_scenario_id,
            "root_scenario_draw_sha256": self.root_scenario_draw_sha256,
            "root_scenario_selection_count": self.root_scenario_selection_count,
            "observations": [item.to_dict() for item in self.observations],
            "cumulative_duration_ms": self.cumulative_duration_ms,
            "modeled_closed_failure_atoms": list(self.modeled_closed_failure_atoms),
            "modeled_closure_satisfied": self.modeled_closure_satisfied,
            "closure_semantics": "SIMULATED_MODELED_ONLY",
            "physical_closure_proven": self.physical_closure_proven,
            "termination_reason": self.termination_reason,
            "evidence_source": self.evidence_source.value,
            "hardware_touch": self.hardware_touch,
            "execution_authority": self.execution_authority,
            "physical_risk_denominator_increment": self.physical_risk_denominator_increment,
        }


@dataclass(frozen=True, slots=True)
class ExhaustiveReplayResult:
    request_sha256: str
    policy_plan_sha256: str
    sequence_model_sha256: str
    root_provenance_sha256: str
    scenario_set_sha256: str
    expected_scenario_ids: tuple[str, ...]
    episodes: tuple[EpisodeResult, ...]

    def __post_init__(self) -> None:
        require_sha256("request_sha256", self.request_sha256)
        require_sha256("policy_plan_sha256", self.policy_plan_sha256)
        require_sha256("sequence_model_sha256", self.sequence_model_sha256)
        require_sha256("root_provenance_sha256", self.root_provenance_sha256)
        require_sha256("scenario_set_sha256", self.scenario_set_sha256)
        if (
            not self.expected_scenario_ids
            or any(not scenario_id for scenario_id in self.expected_scenario_ids)
            or len(set(self.expected_scenario_ids)) != len(self.expected_scenario_ids)
        ):
            raise ValueError("expected scenario ids must be non-empty and unique")
        scenario_ids = tuple(item.root_scenario_id for item in self.episodes)
        if scenario_ids != self.expected_scenario_ids:
            raise ValueError("exhaustive replay must cover every expected scenario exactly once")
        if any(
            item.request_sha256 != self.request_sha256
            or item.policy_plan_sha256 != self.policy_plan_sha256
            or item.sequence_model_sha256 != self.sequence_model_sha256
            or item.root_provenance_sha256 != self.root_provenance_sha256
            or item.scenario_set_sha256 != self.scenario_set_sha256
            for item in self.episodes
        ):
            raise ValueError("exhaustive replay episode binding mismatch")

    @property
    def all_modeled_closure_satisfied(self) -> bool:
        return all(item.modeled_closure_satisfied for item in self.episodes)

    @property
    def physical_closure_proven(self) -> bool:
        return False

    @property
    def replay_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "xrd-rb-voe-exhaustive-replay-v3",
            "request_sha256": self.request_sha256,
            "policy_plan_sha256": self.policy_plan_sha256,
            "sequence_model_sha256": self.sequence_model_sha256,
            "root_provenance_sha256": self.root_provenance_sha256,
            "scenario_set_sha256": self.scenario_set_sha256,
            "expected_scenario_ids": list(self.expected_scenario_ids),
            "scenario_ids": [item.root_scenario_id for item in self.episodes],
            "episode_sha256s": [item.episode_sha256 for item in self.episodes],
            "episodes": [item.to_dict() for item in self.episodes],
            "all_modeled_closure_satisfied": self.all_modeled_closure_satisfied,
            "closure_semantics": "SIMULATED_MODELED_ONLY",
            "physical_closure_proven": self.physical_closure_proven,
        }


def _select_root_scenario(request: SimulationRequest) -> tuple[str, str]:
    draw_sha256 = canonical_sha256(
        {
            "domain": "xrd-rb-voe-root-scenario-draw-v3-common-random-number",
            "seed": request.seed,
            "scenario_set_sha256": request.scenario_set.content_sha256,
            "variant": request.scenario_variant.value,
        }
    )
    draw = int(draw_sha256, 16) / (1 << 256)
    cumulative = 0.0
    selected = request.scenario_set.scenarios[-1].scenario_id
    for scenario in request.scenario_set.scenarios:
        cumulative += scenario.probability(request.scenario_variant)
        if draw < cumulative:
            selected = scenario.scenario_id
            break
    return selected, draw_sha256


def _execute_scenario(
    request: SimulationRequest,
    *,
    root_scenario_id: str,
    draw_sha256: str,
) -> EpisodeResult:
    options = {item.option_id: item for item in request.options}
    closed: set[str] = set()
    executed: set[str] = set()
    observations: list[SimulatedObservation] = []
    previous_observation: str | None = None
    cumulative_duration_ms = 0

    if request.policy_plan.is_hold:
        termination_reason = "POLICY_HOLD"
    elif request.root_provenance_sha256 != request.policy_plan.root_provenance_sha256:
        termination_reason = "PROVENANCE_DRIFT_REJECTED"
    else:
        termination_reason = "HORIZON_EXHAUSTED"
        root_option_id = request.policy_plan.root_option_id
        if root_option_id is None:
            raise ValueError("non-HOLD policy plan must bind a root option")
        for step_index in range(1, request.horizon + 1):
            if step_index in request.local_veto_steps:
                termination_reason = "LOCAL_VETO_REJECTED"
                break
            option_id = request.policy_plan.next_option(
                observation=previous_observation if step_index > 1 else None,
                provenance_sha256=request.root_provenance_sha256,
            )
            if option_id is None:
                termination_reason = "PLAN_BRANCH_MISSING"
                break
            selected_option = options[option_id]
            if not selected_option.option.feasible:
                termination_reason = "INFEASIBLE_OPTION_REJECTED"
                break
            if option_id in executed and not selected_option.option.repeatable:
                termination_reason = "NON_REPEATABLE_OPTION_REJECTED"
                break

            outcome = selected_option.option.outcome_for(root_scenario_id)
            duration_ms = selected_option.duration_ms_by_scenario[root_scenario_id]
            executed.add(option_id)
            sequence_outcome_sha256: str | None = None
            if step_index == 1:
                residual_loss = outcome.residual_loss
                closed.update(outcome.closed_failure_atoms)
                closed_failure_atoms = outcome.closed_failure_atoms
            else:
                if previous_observation is None:
                    raise ValueError("step two requires the bound root observation")
                sequence_outcome = request.sequence_model.outcome_for(
                    root_option_id=root_option_id,
                    root_observation=previous_observation,
                    second_option_id=option_id,
                    scenario_id=root_scenario_id,
                )
                residual_loss = sequence_outcome.terminal_residual_loss
                closed = set(sequence_outcome.terminal_closed_failure_atoms)
                closed_failure_atoms = sequence_outcome.terminal_closed_failure_atoms
                sequence_outcome_sha256 = sequence_outcome.content_sha256
            cumulative_duration_ms += duration_ms
            modeled_closed = request.closure_predicate.is_closed(closed)
            observations.append(
                SimulatedObservation(
                    step_index=step_index,
                    option_id=option_id,
                    root_scenario_id=root_scenario_id,
                    duration_ms=duration_ms,
                    observation=outcome.observation,
                    residual_loss=residual_loss,
                    closed_failure_atoms=closed_failure_atoms,
                    cumulative_modeled_closed_failure_atoms=tuple(sorted(closed)),
                    modeled_closure_satisfied=modeled_closed,
                    sequence_outcome_sha256=sequence_outcome_sha256,
                )
            )
            previous_observation = outcome.observation
            if modeled_closed:
                termination_reason = "MODELED_CLOSURE_SATISFIED"
                break

    return EpisodeResult(
        episode_id=request.episode_id,
        request_sha256=request.content_sha256,
        policy_plan_sha256=request.policy_plan_sha256,
        sequence_model_sha256=request.sequence_model.content_sha256,
        root_provenance_sha256=request.root_provenance_sha256,
        scenario_set_sha256=request.scenario_set.content_sha256,
        root_scenario_id=root_scenario_id,
        root_scenario_draw_sha256=draw_sha256,
        root_scenario_selection_count=1,
        observations=tuple(observations),
        cumulative_duration_ms=cumulative_duration_ms,
        modeled_closed_failure_atoms=tuple(sorted(closed)),
        modeled_closure_satisfied=request.closure_predicate.is_closed(closed),
        termination_reason=termination_reason,
    )


def run_episode(request: SimulationRequest) -> EpisodeResult:
    """Execute exactly the policy plan bound into the request."""
    root_scenario_id, draw_sha256 = _select_root_scenario(request)
    return _execute_scenario(
        request,
        root_scenario_id=root_scenario_id,
        draw_sha256=draw_sha256,
    )


def replay_all_scenarios(request: SimulationRequest) -> ExhaustiveReplayResult:
    """Execute the same bound plan once for every sealed root scenario."""
    episodes = tuple(
        _execute_scenario(
            request,
            root_scenario_id=scenario_id,
            draw_sha256=canonical_sha256(
                {
                    "domain": "xrd-rb-voe-exhaustive-root-v1",
                    "request_sha256": request.content_sha256,
                    "scenario_id": scenario_id,
                }
            ),
        )
        for scenario_id in request.scenario_set.scenario_ids
    )
    return ExhaustiveReplayResult(
        request_sha256=request.content_sha256,
        policy_plan_sha256=request.policy_plan_sha256,
        sequence_model_sha256=request.sequence_model.content_sha256,
        root_provenance_sha256=request.root_provenance_sha256,
        scenario_set_sha256=request.scenario_set.content_sha256,
        expected_scenario_ids=request.scenario_set.scenario_ids,
        episodes=episodes,
    )
