from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .model import (
    AuthenticityLevel,
    EvidenceDomain,
    EvidenceRequirement,
    SkillDefinition,
    immutable_mapping,
)


@dataclass(frozen=True)
class SkillGraph:
    skills: Mapping[str, SkillDefinition]
    edges: tuple[tuple[str, str], ...]
    initial_facts: frozenset[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "skills", immutable_mapping(self.skills))
        if not self.skills:
            raise ValueError("skill graph must contain at least one skill")
        self._validate_linear_dag()
        available = set(self.initial_facts)
        for skill_id in self.ordered_skill_ids:
            skill = self.skills[skill_id]
            missing = skill.preconditions - available
            if missing:
                raise ValueError(
                    f"{skill_id}: graph cannot establish preconditions {sorted(missing)}"
                )
            available.update(skill.expected_effects)

    def _validate_linear_dag(self) -> None:
        incoming = {skill_id: 0 for skill_id in self.skills}
        outgoing: dict[str, list[str]] = {skill_id: [] for skill_id in self.skills}
        seen_edges: set[tuple[str, str]] = set()
        for source, target in self.edges:
            if source not in self.skills or target not in self.skills:
                raise ValueError(f"edge references unknown skill: {(source, target)}")
            if source == target or (source, target) in seen_edges:
                raise ValueError(f"invalid or duplicate edge: {(source, target)}")
            seen_edges.add((source, target))
            incoming[target] += 1
            outgoing[source].append(target)
        roots = [key for key, count in incoming.items() if count == 0]
        leaves = [key for key, targets in outgoing.items() if not targets]
        if len(roots) != 1 or len(leaves) != 1:
            raise ValueError("task verifier requires one linear root and one leaf")
        if any(count > 1 for count in incoming.values()) or any(
            len(targets) > 1 for targets in outgoing.values()
        ):
            raise ValueError("task verifier requires a linear directed skill graph")
        ordered: list[str] = []
        current = roots[0]
        while True:
            if current in ordered:
                raise ValueError("skill graph contains a cycle")
            ordered.append(current)
            if not outgoing[current]:
                break
            current = outgoing[current][0]
        if len(ordered) != len(self.skills):
            raise ValueError("skill graph contains disconnected skills")
        object.__setattr__(self, "_ordered_skill_ids", tuple(ordered))

    @property
    def ordered_skill_ids(self) -> tuple[str, ...]:
        return self._ordered_skill_ids

    def expected_skill(self, completed_count: int) -> str | None:
        if completed_count >= len(self.ordered_skill_ids):
            return None
        return self.ordered_skill_ids[completed_count]


def _control(
    key: str,
    *sources: str,
) -> EvidenceRequirement:
    return EvidenceRequirement(
        key=key,
        domain=EvidenceDomain.CONTROL,
        allowed_sources=frozenset(sources),
        minimum_authenticity=AuthenticityLevel.CONTROL_TELEMETRY,
    )


def _physical(
    key: str,
    *sources: str,
) -> EvidenceRequirement:
    return EvidenceRequirement(
        key=key,
        domain=EvidenceDomain.PHYSICAL,
        allowed_sources=frozenset(sources),
        minimum_authenticity=AuthenticityLevel.PHYSICAL_SENSOR,
    )


def build_finals_skill_graph() -> SkillGraph:
    lifecycle = frozenset({"finals_demo_orchestrator"})
    skills = {
        "pickup": SkillDefinition(
            skill_id="pickup",
            preconditions=frozenset(
                {"system_ready", "lift_at_bottom_control", "arm_at_start_control"}
            ),
            expected_effects=frozenset({"payload_pickup_control"}),
            timeout_s=12.0,
            required_evidence=(
                _control("pickup_sequence_complete", "f407_bridge"),
                _physical(
                    "payload_attached_observed",
                    "load_sensor",
                    "vision_bottle_monitor",
                ),
            ),
            allowed_lifecycle_sources=lifecycle,
        ),
        "lift_top": SkillDefinition(
            skill_id="lift_top",
            preconditions=frozenset({"payload_pickup_control"}),
            expected_effects=frozenset({"lift_at_top_control"}),
            timeout_s=15.0,
            required_evidence=(
                _control(
                    "lift_top_reached",
                    "f407_bridge",
                    "lift_status_monitor",
                ),
                _physical(
                    "payload_at_top_observed",
                    "load_sensor",
                    "vision_bottle_monitor",
                ),
            ),
            allowed_lifecycle_sources=lifecycle,
        ),
        "transport_0p5m": SkillDefinition(
            skill_id="transport_0p5m",
            preconditions=frozenset({"lift_at_top_control"}),
            expected_effects=frozenset({"transport_0p5m_control"}),
            timeout_s=20.0,
            required_evidence=(
                _control("odometry_distance_0p5m", "odometry_monitor"),
                _physical(
                    "payload_retained_observed",
                    "load_sensor",
                    "vision_bottle_monitor",
                ),
            ),
            allowed_lifecycle_sources=lifecycle,
        ),
        "lower": SkillDefinition(
            skill_id="lower",
            preconditions=frozenset({"transport_0p5m_control"}),
            expected_effects=frozenset({"lift_at_bottom_after_transport_control"}),
            timeout_s=15.0,
            required_evidence=(
                _control(
                    "lift_bottom_reached",
                    "f407_bridge",
                    "lift_status_monitor",
                ),
                _physical(
                    "payload_at_lower_observed",
                    "load_sensor",
                    "vision_bottle_monitor",
                ),
            ),
            allowed_lifecycle_sources=lifecycle,
        ),
        "release": SkillDefinition(
            skill_id="release",
            preconditions=frozenset({"lift_at_bottom_after_transport_control"}),
            expected_effects=frozenset({"payload_release_control"}),
            timeout_s=10.0,
            required_evidence=(
                _control("release_sequence_complete", "f407_bridge"),
                _physical(
                    "payload_released_observed",
                    "load_sensor",
                    "vision_bottle_monitor",
                ),
            ),
            allowed_lifecycle_sources=lifecycle,
        ),
        "reset": SkillDefinition(
            skill_id="reset",
            preconditions=frozenset({"payload_release_control"}),
            expected_effects=frozenset({"mechanism_reset_control"}),
            timeout_s=15.0,
            required_evidence=(
                _control("reset_sequence_complete", "f407_bridge"),
                _physical(
                    "mechanism_home_observed",
                    "position_sensor",
                    "vision_workcell_monitor",
                ),
            ),
            allowed_lifecycle_sources=lifecycle,
        ),
    }
    order = tuple(skills)
    edges = tuple(zip(order, order[1:], strict=False))
    return SkillGraph(
        skills=skills,
        edges=edges,
        initial_facts=frozenset(
            {"system_ready", "lift_at_bottom_control", "arm_at_start_control"}
        ),
    )
