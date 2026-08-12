from __future__ import annotations

import math
from collections import defaultdict

from .graph import SkillGraph
from .model import (
    ControlState,
    EventKind,
    EvidenceDomain,
    PhysicalState,
    TaskEvent,
    TraceCode,
    TraceEntry,
    VerificationReport,
)

_INTEGRITY_VIOLATIONS = frozenset(
    {
        TraceCode.ORDER_ERROR,
        TraceCode.PRECONDITION_UNSATISFIED,
        TraceCode.NO_ACTIVE_SKILL,
        TraceCode.ACTIVE_SKILL_MISMATCH,
        TraceCode.LIFECYCLE_SOURCE_REJECTED,
        TraceCode.UNKNOWN_EVIDENCE,
        TraceCode.SOURCE_REJECTED,
        TraceCode.AUTHENTICITY_REJECTED,
        TraceCode.EFFECT_MISMATCH,
        TraceCode.DUPLICATE_EVIDENCE,
        TraceCode.REPLAY_DETECTED,
        TraceCode.NON_MONOTONIC_TIMESTAMP,
        TraceCode.TIMEOUT,
        TraceCode.TASK_TERMINAL,
    }
)


class TaskVerifier:
    """Read-only event verifier with no execution or motion interfaces."""

    motion_authority = False

    def __init__(self, graph: SkillGraph) -> None:
        self.graph = graph
        self._facts = set(graph.initial_facts)
        self._completed: list[str] = []
        self._active_skill: str | None = None
        self._active_started_at: float | None = None
        self._active_start_event: str | None = None
        self._accepted_evidence: dict[str, dict[str, str]] = defaultdict(dict)
        self._seen_event_ids: set[str] = set()
        self._seen_fingerprints: set[str] = set()
        self._last_timestamp_s: float | None = None
        self._trace: list[TraceEntry] = []
        self._violations: list[TraceCode] = []
        self._terminal = False

    def process(self, event: TaskEvent) -> TraceEntry:
        replay = (
            event.event_id in self._seen_event_ids
            or event.replay_fingerprint() in self._seen_fingerprints
        )
        if replay:
            return self._record(
                event,
                TraceCode.REPLAY_DETECTED,
                False,
                "event id or canonical event payload was already observed",
            )

        self._seen_event_ids.add(event.event_id)
        self._seen_fingerprints.add(event.replay_fingerprint())

        if (
            self._last_timestamp_s is not None
            and event.timestamp_s < self._last_timestamp_s
        ):
            return self._record(
                event,
                TraceCode.NON_MONOTONIC_TIMESTAMP,
                False,
                "event timestamp is older than the last accepted session timestamp",
            )
        self._last_timestamp_s = event.timestamp_s

        if self._terminal:
            return self._record(
                event,
                TraceCode.TASK_TERMINAL,
                False,
                "task is terminal after an integrity or timeout failure",
            )

        if self._deadline_exceeded(event.timestamp_s):
            self._terminal = True
            return self._record(
                event,
                TraceCode.TIMEOUT,
                False,
                self._timeout_detail(event.timestamp_s),
                causes=self._active_causes(),
            )

        if event.kind is EventKind.SKILL_STARTED:
            return self._start(event)
        if event.kind is EventKind.EVIDENCE_RECORDED:
            return self._accept_evidence(event)
        return self._complete(event)

    def advance_time(self, timestamp_s: float) -> TraceEntry | None:
        if not math.isfinite(timestamp_s) or timestamp_s < 0.0:
            raise ValueError("timestamp_s must be finite and non-negative")
        if self._terminal or self._active_skill is None:
            return None
        if self._last_timestamp_s is not None and timestamp_s < self._last_timestamp_s:
            raise ValueError("clock cannot move backwards")
        self._last_timestamp_s = timestamp_s
        if not self._deadline_exceeded(timestamp_s):
            return None
        self._terminal = True
        event = TaskEvent.started(
            event_id=f"verifier-clock-{len(self._trace) + 1}",
            skill_id=self._active_skill,
            timestamp_s=timestamp_s,
            source="verifier_clock",
        )
        return self._record(
            event,
            TraceCode.TIMEOUT,
            False,
            self._timeout_detail(timestamp_s),
            causes=self._active_causes(),
        )

    def report(self) -> VerificationReport:
        violations = tuple(self._violations)
        disqualifying = any(item in _INTEGRITY_VIOLATIONS for item in violations)
        all_completed = len(self._completed) == len(self.graph.ordered_skill_ids)
        if all_completed and not disqualifying:
            control_state = ControlState.CONTROL_STATE_VERIFIED
        elif self._terminal or disqualifying:
            control_state = ControlState.CONTROL_STATE_UNVERIFIED
        else:
            control_state = ControlState.IN_PROGRESS

        missing_physical: list[str] = []
        for skill_id in self.graph.ordered_skill_ids:
            for requirement in self.graph.skills[skill_id].required_evidence:
                if (
                    requirement.domain is EvidenceDomain.PHYSICAL
                    and requirement.key
                    not in self._accepted_evidence.get(skill_id, {})
                ):
                    missing_physical.append(f"{skill_id}:{requirement.key}")
        if (
            control_state is ControlState.CONTROL_STATE_VERIFIED
            and not missing_physical
        ):
            physical_state = PhysicalState.PHYSICAL_SUCCESS_VERIFIED
        else:
            physical_state = PhysicalState.PHYSICAL_SUCCESS_UNVERIFIED

        return VerificationReport(
            control_state=control_state,
            physical_state=physical_state,
            completed_skills=tuple(self._completed),
            facts=tuple(sorted(self._facts)),
            missing_physical_evidence=tuple(missing_physical),
            violations=violations,
            trace=tuple(self._trace),
            motion_authority=self.motion_authority,
        )

    def _start(self, event: TaskEvent) -> TraceEntry:
        expected = self.graph.expected_skill(len(self._completed))
        if expected is None or event.skill_id != expected or self._active_skill:
            self._terminal = True
            return self._record(
                event,
                TraceCode.ORDER_ERROR,
                False,
                f"expected next skill {expected!r}, got {event.skill_id!r}",
            )
        skill = self.graph.skills[event.skill_id]
        if event.source not in skill.allowed_lifecycle_sources:
            self._terminal = True
            return self._record(
                event,
                TraceCode.LIFECYCLE_SOURCE_REJECTED,
                False,
                f"source {event.source!r} cannot start {event.skill_id!r}",
            )
        missing = skill.preconditions - self._facts
        if missing:
            self._terminal = True
            return self._record(
                event,
                TraceCode.PRECONDITION_UNSATISFIED,
                False,
                f"missing preconditions: {sorted(missing)}",
            )
        self._active_skill = event.skill_id
        self._active_started_at = event.timestamp_s
        self._active_start_event = event.event_id
        return self._record(
            event,
            TraceCode.ACCEPTED_START,
            True,
            f"started {event.skill_id}",
        )

    def _accept_evidence(self, event: TaskEvent) -> TraceEntry:
        mismatch = self._active_mismatch(event)
        if mismatch is not None:
            return mismatch
        skill = self.graph.skills[event.skill_id]
        requirement = skill.evidence(event.evidence_key or "")
        if requirement is None:
            return self._record(
                event,
                TraceCode.UNKNOWN_EVIDENCE,
                False,
                f"{event.evidence_key!r} is not required by {event.skill_id!r}",
                causes=self._active_causes(),
            )
        if event.evidence_key in self._accepted_evidence[event.skill_id]:
            return self._record(
                event,
                TraceCode.DUPLICATE_EVIDENCE,
                False,
                f"{event.evidence_key!r} was already accepted",
                causes=self._active_causes(),
            )
        if event.source not in requirement.allowed_sources:
            return self._record(
                event,
                TraceCode.SOURCE_REJECTED,
                False,
                f"source {event.source!r} is not allowed for {requirement.key!r}",
                causes=self._active_causes(),
            )
        if event.authenticity < requirement.minimum_authenticity:
            return self._record(
                event,
                TraceCode.AUTHENTICITY_REJECTED,
                False,
                (
                    f"{event.authenticity.name} is below required "
                    f"{requirement.minimum_authenticity.name}"
                ),
                causes=self._active_causes(),
            )
        self._accepted_evidence[event.skill_id][requirement.key] = event.event_id
        return self._record(
            event,
            TraceCode.ACCEPTED_EVIDENCE,
            True,
            (
                f"accepted {requirement.domain.value} evidence "
                f"{requirement.key!r}"
            ),
            causes=self._active_causes(),
        )

    def _complete(self, event: TaskEvent) -> TraceEntry:
        mismatch = self._active_mismatch(event)
        if mismatch is not None:
            return mismatch
        skill = self.graph.skills[event.skill_id]
        if event.source not in skill.allowed_lifecycle_sources:
            self._terminal = True
            return self._record(
                event,
                TraceCode.LIFECYCLE_SOURCE_REJECTED,
                False,
                f"source {event.source!r} cannot complete {event.skill_id!r}",
                causes=self._active_causes(),
            )
        if event.observed_effects != skill.expected_effects:
            self._terminal = True
            return self._record(
                event,
                TraceCode.EFFECT_MISMATCH,
                False,
                (
                    f"expected effects {sorted(skill.expected_effects)}, "
                    f"got {sorted(event.observed_effects)}"
                ),
                causes=self._active_causes(),
            )
        missing_control = [
            requirement.key
            for requirement in skill.required_evidence
            if requirement.domain is EvidenceDomain.CONTROL
            and requirement.key not in self._accepted_evidence[event.skill_id]
        ]
        if missing_control:
            return self._record(
                event,
                TraceCode.EVIDENCE_INSUFFICIENT,
                False,
                f"missing required control evidence: {missing_control}",
                causes=self._active_causes(),
            )

        causes = self._active_causes()
        self._facts.update(skill.expected_effects)
        self._completed.append(skill.skill_id)
        self._active_skill = None
        self._active_started_at = None
        self._active_start_event = None
        return self._record(
            event,
            TraceCode.ACCEPTED_COMPLETION,
            True,
            f"completed {event.skill_id}",
            causes=causes,
        )

    def _active_mismatch(self, event: TaskEvent) -> TraceEntry | None:
        if self._active_skill is None:
            return self._record(
                event,
                TraceCode.NO_ACTIVE_SKILL,
                False,
                "no skill is active",
            )
        if event.skill_id != self._active_skill:
            self._terminal = True
            return self._record(
                event,
                TraceCode.ACTIVE_SKILL_MISMATCH,
                False,
                f"active skill is {self._active_skill!r}",
                causes=self._active_causes(),
            )
        return None

    def _deadline_exceeded(self, timestamp_s: float) -> bool:
        if self._active_skill is None or self._active_started_at is None:
            return False
        timeout_s = self.graph.skills[self._active_skill].timeout_s
        return timestamp_s > self._active_started_at + timeout_s

    def _timeout_detail(self, timestamp_s: float) -> str:
        assert self._active_skill is not None
        assert self._active_started_at is not None
        elapsed = timestamp_s - self._active_started_at
        timeout = self.graph.skills[self._active_skill].timeout_s
        return (
            f"{self._active_skill!r} elapsed {elapsed:.3f}s, "
            f"exceeding timeout {timeout:.3f}s"
        )

    def _active_causes(self) -> tuple[str, ...]:
        if self._active_skill is None:
            return ()
        evidence_ids = self._accepted_evidence.get(self._active_skill, {})
        ordered = [
            evidence_ids[requirement.key]
            for requirement in self.graph.skills[
                self._active_skill
            ].required_evidence
            if requirement.key in evidence_ids
        ]
        if self._active_start_event is not None:
            return (self._active_start_event, *ordered)
        return tuple(ordered)

    def _record(
        self,
        event: TaskEvent,
        code: TraceCode,
        accepted: bool,
        detail: str,
        causes: tuple[str, ...] = (),
    ) -> TraceEntry:
        entry = TraceEntry(
            sequence=len(self._trace) + 1,
            event_id=event.event_id,
            timestamp_s=event.timestamp_s,
            skill_id=event.skill_id,
            code=code,
            accepted=accepted,
            detail=detail,
            causes=causes,
            facts_after=tuple(sorted(self._facts)),
        )
        self._trace.append(entry)
        if not accepted:
            self._violations.append(code)
        return entry
