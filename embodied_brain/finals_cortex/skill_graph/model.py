from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from types import MappingProxyType
from typing import Any


class AuthenticityLevel(IntEnum):
    """Strength of the evidence asserted by a trusted adapter."""

    UNVERIFIED = 0
    ASSERTED = 10
    CONTROL_TELEMETRY = 20
    PHYSICAL_SENSOR = 30


class EvidenceDomain(str, Enum):
    CONTROL = "control"
    PHYSICAL = "physical"


class EventKind(str, Enum):
    SKILL_STARTED = "skill_started"
    EVIDENCE_RECORDED = "evidence_recorded"
    SKILL_COMPLETED = "skill_completed"


class TraceCode(str, Enum):
    ACCEPTED_START = "ACCEPTED_START"
    ACCEPTED_EVIDENCE = "ACCEPTED_EVIDENCE"
    ACCEPTED_COMPLETION = "ACCEPTED_COMPLETION"
    ORDER_ERROR = "ORDER_ERROR"
    PRECONDITION_UNSATISFIED = "PRECONDITION_UNSATISFIED"
    NO_ACTIVE_SKILL = "NO_ACTIVE_SKILL"
    ACTIVE_SKILL_MISMATCH = "ACTIVE_SKILL_MISMATCH"
    LIFECYCLE_SOURCE_REJECTED = "LIFECYCLE_SOURCE_REJECTED"
    UNKNOWN_EVIDENCE = "UNKNOWN_EVIDENCE"
    SOURCE_REJECTED = "SOURCE_REJECTED"
    AUTHENTICITY_REJECTED = "AUTHENTICITY_REJECTED"
    EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"
    EFFECT_MISMATCH = "EFFECT_MISMATCH"
    DUPLICATE_EVIDENCE = "DUPLICATE_EVIDENCE"
    REPLAY_DETECTED = "REPLAY_DETECTED"
    NON_MONOTONIC_TIMESTAMP = "NON_MONOTONIC_TIMESTAMP"
    TIMEOUT = "TIMEOUT"
    TASK_TERMINAL = "TASK_TERMINAL"


class ControlState(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    CONTROL_STATE_VERIFIED = "CONTROL_STATE_VERIFIED"
    CONTROL_STATE_UNVERIFIED = "CONTROL_STATE_UNVERIFIED"


class PhysicalState(str, Enum):
    PHYSICAL_SUCCESS_VERIFIED = "PHYSICAL_SUCCESS_VERIFIED"
    PHYSICAL_SUCCESS_UNVERIFIED = "PHYSICAL_SUCCESS_UNVERIFIED"


@dataclass(frozen=True)
class EvidenceRequirement:
    key: str
    domain: EvidenceDomain
    allowed_sources: frozenset[str]
    minimum_authenticity: AuthenticityLevel

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("evidence key must be non-empty")
        if not self.allowed_sources or any(not item for item in self.allowed_sources):
            raise ValueError(f"{self.key}: allowed_sources must be non-empty")
        if (
            self.domain is EvidenceDomain.PHYSICAL
            and self.minimum_authenticity < AuthenticityLevel.PHYSICAL_SENSOR
        ):
            raise ValueError(
                f"{self.key}: physical evidence must require PHYSICAL_SENSOR"
            )


@dataclass(frozen=True)
class SkillDefinition:
    skill_id: str
    preconditions: frozenset[str]
    expected_effects: frozenset[str]
    timeout_s: float
    required_evidence: tuple[EvidenceRequirement, ...]
    allowed_lifecycle_sources: frozenset[str]

    def __post_init__(self) -> None:
        if not self.skill_id:
            raise ValueError("skill_id must be non-empty")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0.0:
            raise ValueError(f"{self.skill_id}: timeout_s must be finite and positive")
        if not self.expected_effects:
            raise ValueError(f"{self.skill_id}: expected_effects must be non-empty")
        if not self.allowed_lifecycle_sources:
            raise ValueError(
                f"{self.skill_id}: allowed_lifecycle_sources must be non-empty"
            )
        keys = [item.key for item in self.required_evidence]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{self.skill_id}: evidence keys must be unique")
        domains = {item.domain for item in self.required_evidence}
        if EvidenceDomain.CONTROL not in domains:
            raise ValueError(f"{self.skill_id}: control evidence is required")
        if EvidenceDomain.PHYSICAL not in domains:
            raise ValueError(f"{self.skill_id}: physical evidence is required")

    def evidence(self, key: str) -> EvidenceRequirement | None:
        return next((item for item in self.required_evidence if item.key == key), None)


@dataclass(frozen=True)
class TaskEvent:
    event_id: str
    kind: EventKind
    skill_id: str
    timestamp_s: float
    source: str
    evidence_key: str | None = None
    authenticity: AuthenticityLevel = AuthenticityLevel.UNVERIFIED
    observed_effects: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.event_id or not self.skill_id or not self.source:
            raise ValueError("event_id, skill_id, and source must be non-empty")
        if not math.isfinite(self.timestamp_s) or self.timestamp_s < 0.0:
            raise ValueError("timestamp_s must be finite and non-negative")
        if self.kind is EventKind.EVIDENCE_RECORDED:
            if not self.evidence_key:
                raise ValueError("evidence events require evidence_key")
            if self.observed_effects:
                raise ValueError("evidence events cannot declare skill effects")
        elif self.evidence_key is not None:
            raise ValueError("only evidence events may set evidence_key")
        if (
            self.kind is not EventKind.SKILL_COMPLETED
            and self.observed_effects
        ):
            raise ValueError("only completion events may declare observed_effects")

    @classmethod
    def started(
        cls,
        event_id: str,
        skill_id: str,
        timestamp_s: float,
        source: str = "finals_demo_orchestrator",
    ) -> TaskEvent:
        return cls(
            event_id=event_id,
            kind=EventKind.SKILL_STARTED,
            skill_id=skill_id,
            timestamp_s=timestamp_s,
            source=source,
        )

    @classmethod
    def evidence(
        cls,
        event_id: str,
        skill_id: str,
        timestamp_s: float,
        source: str,
        evidence_key: str,
        authenticity: AuthenticityLevel,
    ) -> TaskEvent:
        return cls(
            event_id=event_id,
            kind=EventKind.EVIDENCE_RECORDED,
            skill_id=skill_id,
            timestamp_s=timestamp_s,
            source=source,
            evidence_key=evidence_key,
            authenticity=authenticity,
        )

    @classmethod
    def completed(
        cls,
        event_id: str,
        skill_id: str,
        timestamp_s: float,
        observed_effects: frozenset[str],
        source: str = "finals_demo_orchestrator",
    ) -> TaskEvent:
        return cls(
            event_id=event_id,
            kind=EventKind.SKILL_COMPLETED,
            skill_id=skill_id,
            timestamp_s=timestamp_s,
            source=source,
            observed_effects=observed_effects,
        )

    def replay_fingerprint(self) -> str:
        payload = {
            "kind": self.kind.value,
            "skill_id": self.skill_id,
            "timestamp_s": self.timestamp_s,
            "source": self.source,
            "evidence_key": self.evidence_key,
            "authenticity": int(self.authenticity),
            "observed_effects": sorted(self.observed_effects),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class TraceEntry:
    sequence: int
    event_id: str
    timestamp_s: float
    skill_id: str
    code: TraceCode
    accepted: bool
    detail: str
    causes: tuple[str, ...]
    facts_after: tuple[str, ...]


@dataclass(frozen=True)
class VerificationReport:
    control_state: ControlState
    physical_state: PhysicalState
    completed_skills: tuple[str, ...]
    facts: tuple[str, ...]
    missing_physical_evidence: tuple[str, ...]
    violations: tuple[TraceCode, ...]
    trace: tuple[TraceEntry, ...]
    motion_authority: bool = False

    @property
    def boundary(self) -> str:
        if (
            self.control_state is ControlState.CONTROL_STATE_VERIFIED
            and self.physical_state is PhysicalState.PHYSICAL_SUCCESS_UNVERIFIED
        ):
            return (
                "CONTROL_STATE_VERIFIED; PHYSICAL_SUCCESS_UNVERIFIED: "
                "the accepted telemetry proves the commanded state sequence, "
                "not physical payload success."
            )
        return f"{self.control_state.value}; {self.physical_state.value}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_state": self.control_state.value,
            "physical_state": self.physical_state.value,
            "completed_skills": list(self.completed_skills),
            "facts": list(self.facts),
            "missing_physical_evidence": list(self.missing_physical_evidence),
            "violations": [item.value for item in self.violations],
            "motion_authority": self.motion_authority,
            "boundary": self.boundary,
            "trace": [
                {
                    "sequence": item.sequence,
                    "event_id": item.event_id,
                    "timestamp_s": item.timestamp_s,
                    "skill_id": item.skill_id,
                    "code": item.code.value,
                    "accepted": item.accepted,
                    "detail": item.detail,
                    "causes": list(item.causes),
                    "facts_after": list(item.facts_after),
                }
                for item in self.trace
            ],
        }


def immutable_mapping(
    value: Mapping[str, SkillDefinition],
) -> Mapping[str, SkillDefinition]:
    return MappingProxyType(dict(value))
