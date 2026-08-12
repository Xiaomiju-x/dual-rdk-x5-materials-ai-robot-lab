"""Sealed finite joint scenarios for nominal and minimax policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isclose, isfinite

from rb_voe.contracts.canonical import canonical_sha256


class ScenarioVariant(str, Enum):
    NOMINAL = "NOMINAL"
    ROBUST = "ROBUST"


@dataclass(frozen=True, slots=True)
class JointScenario:
    """One root realization under nominal and sealed stress distributions."""

    scenario_id: str
    hold_loss: float
    nominal_probability: float
    robust_probability: float
    failure_domains: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id must be non-empty")
        for name, value in (
            ("hold_loss", self.hold_loss),
            ("nominal_probability", self.nominal_probability),
            ("robust_probability", self.robust_probability),
        ):
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        normalized = tuple(sorted(set(self.failure_domains)))
        if any(not domain for domain in normalized):
            raise ValueError("failure domains must be non-empty")
        object.__setattr__(self, "failure_domains", normalized)

    def probability(self, variant: ScenarioVariant) -> float:
        return self.nominal_probability if variant is ScenarioVariant.NOMINAL else self.robust_probability

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "hold_loss": self.hold_loss,
            "nominal_probability": self.nominal_probability,
            "robust_probability": self.robust_probability,
            "failure_domains": list(self.failure_domains),
        }


@dataclass(frozen=True, slots=True)
class JointScenarioSet:
    """Immutable root ambiguity set; conditioning never creates new scenarios.

    NOMINAL uses the sealed nominal probability distribution. ROBUST treats each
    sealed root joint scenario as one member of the ambiguity set and minimizes
    the maximum individual scenario loss. The robust probability remains in the
    contract for deterministic counterfactual sampling; it is never used to
    average away a rare catastrophic scenario during robust planning.
    """

    scenario_set_id: str
    scenarios: tuple[JointScenario, ...]

    def __post_init__(self) -> None:
        if not self.scenario_set_id:
            raise ValueError("scenario_set_id must be non-empty")
        normalized = tuple(sorted(self.scenarios, key=lambda item: item.scenario_id))
        identifiers = tuple(item.scenario_id for item in normalized)
        if not normalized or len(set(identifiers)) != len(identifiers):
            raise ValueError("joint scenario ids must be non-empty and unique")
        for variant in ScenarioVariant:
            total = sum(item.probability(variant) for item in normalized)
            if not isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"{variant.value} scenario probabilities must sum to one")
        object.__setattr__(self, "scenarios", normalized)

    @property
    def scenario_ids(self) -> tuple[str, ...]:
        return tuple(item.scenario_id for item in self.scenarios)

    @property
    def failure_domains(self) -> tuple[str, ...]:
        return tuple(sorted({domain for scenario in self.scenarios for domain in scenario.failure_domains}))

    @property
    def content_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def scenario(self, scenario_id: str) -> JointScenario:
        for scenario in self.scenarios:
            if scenario.scenario_id == scenario_id:
                return scenario
        raise KeyError(f"unknown root scenario: {scenario_id}")

    def conditional_weights(
        self,
        scenario_ids: tuple[str, ...],
        variant: ScenarioVariant,
    ) -> tuple[tuple[str, float], ...]:
        """Condition one root distribution on a subset without resampling it."""
        selected = tuple(sorted(set(scenario_ids)))
        if not selected or not set(selected).issubset(self.scenario_ids):
            raise ValueError("conditioning requires a non-empty root-scenario subset")
        mass = sum(self.scenario(identifier).probability(variant) for identifier in selected)
        if mass <= 0:
            raise ValueError("conditioned scenario subset has zero probability mass")
        return tuple(
            (identifier, self.scenario(identifier).probability(variant) / mass) for identifier in selected
        )

    def risk_members(
        self,
        losses: dict[str, float],
        variant: ScenarioVariant,
    ) -> tuple[float, ...]:
        """Return the exact values over which the policy takes a maximum."""
        if set(losses) != set(self.scenario_ids):
            raise ValueError("losses must exactly cover the sealed root scenarios")
        for scenario_id, loss in losses.items():
            if not isfinite(loss) or loss < 0:
                raise ValueError(f"loss for {scenario_id} must be finite and non-negative")
        if variant is ScenarioVariant.NOMINAL:
            return (
                sum(
                    scenario.nominal_probability * losses[scenario.scenario_id] for scenario in self.scenarios
                ),
            )
        return tuple(losses[scenario_id] for scenario_id in self.scenario_ids)

    def risk(self, losses: dict[str, float], variant: ScenarioVariant) -> float:
        return max(self.risk_members(losses, variant))

    def expected_hold_loss(self, variant: ScenarioVariant) -> float:
        losses = {scenario.scenario_id: scenario.hold_loss for scenario in self.scenarios}
        return self.risk(losses, variant)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "xrd-rb-voe-joint-scenario-set-v1",
            "scenario_set_id": self.scenario_set_id,
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
        }
