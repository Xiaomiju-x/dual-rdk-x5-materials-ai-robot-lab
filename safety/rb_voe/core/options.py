"""Immutable evidence options and testable failure-core closure contracts."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from rb_voe.contracts.canonical import canonical_sha256


def _require_finite_nonnegative(name: str, value: float) -> None:
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class OptionOutcome:
    """Outcome of one evidence option in one sealed root scenario."""

    scenario_id: str
    observation: str
    residual_loss: float
    closed_failure_atoms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.observation:
            raise ValueError("outcome scenario_id and observation must be non-empty")
        _require_finite_nonnegative("residual_loss", self.residual_loss)
        normalized = tuple(sorted(set(self.closed_failure_atoms)))
        if any(not atom for atom in normalized):
            raise ValueError("closed failure atoms must be non-empty")
        object.__setattr__(self, "closed_failure_atoms", normalized)

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "observation": self.observation,
            "residual_loss": self.residual_loss,
            "closed_failure_atoms": list(self.closed_failure_atoms),
        }


@dataclass(frozen=True, slots=True)
class SequenceOutcome:
    """Terminal effect of one explicit two-step evidence sequence.

    The key includes the root option, its observation, the second option, and
    the sealed root scenario. The residual is therefore a property of the
    complete ordered sequence, not an alias for the second option's standalone
    residual.
    """

    root_option_id: str
    root_observation: str
    second_option_id: str
    scenario_id: str
    terminal_residual_loss: float
    terminal_closed_failure_atoms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not all(
            (
                self.root_option_id,
                self.root_observation,
                self.second_option_id,
                self.scenario_id,
            )
        ):
            raise ValueError("sequence outcome identity fields must be non-empty")
        _require_finite_nonnegative("terminal_residual_loss", self.terminal_residual_loss)
        normalized = tuple(sorted(set(self.terminal_closed_failure_atoms)))
        if any(not atom for atom in normalized):
            raise ValueError("terminal closed failure atoms must be non-empty")
        object.__setattr__(self, "terminal_closed_failure_atoms", normalized)

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (
            self.root_option_id,
            self.root_observation,
            self.second_option_id,
            self.scenario_id,
        )

    @property
    def content_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "root_option_id": self.root_option_id,
            "root_observation": self.root_observation,
            "second_option_id": self.second_option_id,
            "scenario_id": self.scenario_id,
            "terminal_residual_loss": self.terminal_residual_loss,
            "terminal_closed_failure_atoms": list(self.terminal_closed_failure_atoms),
        }


@dataclass(frozen=True, slots=True)
class SequenceOutcomeModel:
    """Frozen, content-addressed terminal-effect table for H=2 planning."""

    model_id: str
    outcomes: tuple[SequenceOutcome, ...]

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("sequence outcome model_id must be non-empty")
        normalized = tuple(sorted(self.outcomes, key=lambda item: item.key))
        keys = tuple(item.key for item in normalized)
        if len(set(keys)) != len(keys):
            raise ValueError("sequence outcome keys must be unique")
        object.__setattr__(self, "outcomes", normalized)

    @property
    def content_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def outcome_for(
        self,
        *,
        root_option_id: str,
        root_observation: str,
        second_option_id: str,
        scenario_id: str,
    ) -> SequenceOutcome:
        key = (root_option_id, root_observation, second_option_id, scenario_id)
        for outcome in self.outcomes:
            if outcome.key == key:
                return outcome
        raise KeyError(
            "sequence outcome model has no terminal effect for "
            f"root={root_option_id}, observation={root_observation}, "
            f"second={second_option_id}, scenario={scenario_id}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "xrd-rb-voe-sequence-outcome-model-v1",
            "model_id": self.model_id,
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
        }


@dataclass(frozen=True, slots=True)
class HardGate:
    """A non-compensable admission gate evaluated before policy value."""

    gate_id: str
    passed: bool
    failure_code: str | None = None

    def __post_init__(self) -> None:
        if not self.gate_id:
            raise ValueError("gate_id must be non-empty")
        if self.passed and self.failure_code is not None:
            raise ValueError("a passing hard gate cannot carry a failure code")
        if not self.passed and not self.failure_code:
            raise ValueError("a failing hard gate requires a failure code")

    def to_dict(self) -> dict[str, object]:
        return {
            "gate_id": self.gate_id,
            "passed": self.passed,
            "failure_code": self.failure_code,
        }


@dataclass(frozen=True, slots=True)
class EvidenceOption:
    """One candidate acquisition with explicit cost and scenario outcomes."""

    option_id: str
    cost: float
    outcomes: tuple[OptionOutcome, ...]
    hard_gates: tuple[HardGate, ...] = ()
    repeatable: bool = False

    def __post_init__(self) -> None:
        if not self.option_id:
            raise ValueError("option_id must be non-empty")
        _require_finite_nonnegative("option cost", self.cost)
        normalized_outcomes = tuple(sorted(self.outcomes, key=lambda item: item.scenario_id))
        scenario_ids = tuple(item.scenario_id for item in normalized_outcomes)
        if not normalized_outcomes or len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("option outcomes must contain unique scenario ids")
        normalized_gates = tuple(sorted(self.hard_gates, key=lambda gate: gate.gate_id))
        gate_ids = tuple(gate.gate_id for gate in normalized_gates)
        if len(set(gate_ids)) != len(gate_ids):
            raise ValueError("hard gate ids must be unique per option")
        object.__setattr__(self, "outcomes", normalized_outcomes)
        object.__setattr__(self, "hard_gates", normalized_gates)

    @property
    def feasible(self) -> bool:
        return all(gate.passed for gate in self.hard_gates)

    @property
    def failure_codes(self) -> tuple[str, ...]:
        return tuple(
            gate.failure_code for gate in self.hard_gates if not gate.passed and gate.failure_code is not None
        )

    @property
    def content_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def outcome_for(self, scenario_id: str) -> OptionOutcome:
        for outcome in self.outcomes:
            if outcome.scenario_id == scenario_id:
                return outcome
        raise KeyError(f"option {self.option_id} has no outcome for scenario {scenario_id}")

    def to_dict(self) -> dict[str, object]:
        return {
            "option_id": self.option_id,
            "cost": self.cost,
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
            "hard_gates": [gate.to_dict() for gate in self.hard_gates],
            "repeatable": self.repeatable,
        }


@dataclass(frozen=True, slots=True)
class ClosurePredicate:
    """Monotone predicate defining whether the selected failure core is closed."""

    predicate_id: str
    required_failure_atoms: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.predicate_id:
            raise ValueError("predicate_id must be non-empty")
        normalized = tuple(sorted(set(self.required_failure_atoms)))
        if not normalized or any(not atom for atom in normalized):
            raise ValueError("closure predicate requires non-empty failure atoms")
        object.__setattr__(self, "required_failure_atoms", normalized)

    def is_closed(self, closed_failure_atoms: tuple[str, ...] | set[str]) -> bool:
        return set(self.required_failure_atoms).issubset(closed_failure_atoms)

    @property
    def content_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "predicate_id": self.predicate_id,
            "required_failure_atoms": list(self.required_failure_atoms),
        }
