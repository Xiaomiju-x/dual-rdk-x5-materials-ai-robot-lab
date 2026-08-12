"""Deterministic task automaton for the approved RB-VoE v2 workflow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from rb_voe.contracts.canonical import canonical_sha256


class TaskState(str, Enum):
    CASE_BOUND = "CASE_BOUND"
    BELIEF_UPDATED = "BELIEF_UPDATED"
    FAILURE_CORE_READY = "FAILURE_CORE_READY"
    NEEDS_EVIDENCE = "NEEDS_EVIDENCE"
    ACTION_SELECTED = "ACTION_SELECTED"
    PREPARE_PHYSICAL = "PREPARE_PHYSICAL"
    REPAIR_EVIDENCE = "REPAIR_EVIDENCE"
    JOINT_PERMIT = "JOINT_PERMIT"
    EXECUTING = "EXECUTING"
    SAFE_ABORT = "SAFE_ABORT"
    PHYSICAL_VERIFY = "PHYSICAL_VERIFY"
    QUARANTINE = "QUARANTINE"
    GO = "GO"
    REVISE = "REVISE"
    DROP = "DROP"
    HOLD = "HOLD"


LEGAL_TRANSITIONS: Mapping[TaskState, frozenset[TaskState]] = MappingProxyType(
    {
        TaskState.CASE_BOUND: frozenset({TaskState.BELIEF_UPDATED}),
        TaskState.BELIEF_UPDATED: frozenset(
            {
                TaskState.FAILURE_CORE_READY,
                TaskState.GO,
                TaskState.REVISE,
                TaskState.DROP,
            }
        ),
        TaskState.FAILURE_CORE_READY: frozenset({TaskState.NEEDS_EVIDENCE}),
        TaskState.NEEDS_EVIDENCE: frozenset({TaskState.ACTION_SELECTED}),
        TaskState.ACTION_SELECTED: frozenset({TaskState.HOLD, TaskState.PREPARE_PHYSICAL}),
        TaskState.PREPARE_PHYSICAL: frozenset({TaskState.REPAIR_EVIDENCE, TaskState.JOINT_PERMIT}),
        TaskState.JOINT_PERMIT: frozenset({TaskState.EXECUTING}),
        TaskState.EXECUTING: frozenset({TaskState.SAFE_ABORT, TaskState.PHYSICAL_VERIFY}),
        TaskState.PHYSICAL_VERIFY: frozenset({TaskState.QUARANTINE, TaskState.BELIEF_UPDATED}),
        TaskState.REPAIR_EVIDENCE: frozenset(),
        TaskState.SAFE_ABORT: frozenset(),
        TaskState.QUARANTINE: frozenset(),
        TaskState.GO: frozenset(),
        TaskState.REVISE: frozenset(),
        TaskState.DROP: frozenset(),
        TaskState.HOLD: frozenset(),
    }
)

TERMINAL_STATES = frozenset(state for state, allowed in LEGAL_TRANSITIONS.items() if not allowed)


@dataclass(frozen=True, slots=True)
class TaskTransition:
    sequence: int
    source: TaskState
    target: TaskState
    reason: str

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise TypeError("task transition sequence must be an integer")
        if self.sequence < 1:
            raise ValueError("task transition sequence must be positive")
        if not isinstance(self.source, TaskState):
            object.__setattr__(self, "source", TaskState(self.source))
        if not isinstance(self.target, TaskState):
            object.__setattr__(self, "target", TaskState(self.target))
        if not isinstance(self.reason, str):
            raise TypeError("task transition reason must be a string")

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "source": self.source.value,
            "target": self.target.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class TransitionFailure:
    code: str
    source: TaskState
    requested_target: TaskState
    allowed_targets: tuple[TaskState, ...]
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "source": self.source.value,
            "requested_target": self.requested_target.value,
            "allowed_targets": [state.value for state in self.allowed_targets],
            "message": self.message,
        }


class IllegalTransition(ValueError):
    def __init__(self, failure: TransitionFailure):
        self.failure = failure
        super().__init__(failure.message)


@dataclass(frozen=True, slots=True)
class TransitionResult:
    accepted: bool
    automaton: TaskAutomaton | None
    failure: TransitionFailure | None


@dataclass(frozen=True, slots=True)
class TaskAutomaton:
    state: TaskState = TaskState.CASE_BOUND
    history: tuple[TaskTransition, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.state, TaskState):
            object.__setattr__(self, "state", TaskState(self.state))
        history = tuple(self.history)
        if any(not isinstance(transition, TaskTransition) for transition in history):
            raise TypeError("task history accepts TaskTransition instances only")
        object.__setattr__(self, "history", history)
        self._validate_history()

    @classmethod
    def start(cls) -> TaskAutomaton:
        return cls()

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def legal_targets(self) -> tuple[TaskState, ...]:
        return tuple(sorted(LEGAL_TRANSITIONS[self.state], key=lambda state: state.value))

    @property
    def content_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def can_transition(self, target: TaskState | str) -> bool:
        return TaskState(target) in LEGAL_TRANSITIONS[self.state]

    def transition(self, target: TaskState | str, reason: str = "") -> TaskAutomaton:
        if not isinstance(reason, str):
            raise TypeError("task transition reason must be a string")
        normalized = TaskState(target)
        if not self.can_transition(normalized):
            raise IllegalTransition(self._failure(normalized))
        transition = TaskTransition(
            sequence=len(self.history) + 1,
            source=self.state,
            target=normalized,
            reason=reason,
        )
        return TaskAutomaton(normalized, (*self.history, transition))

    def try_transition(self, target: TaskState | str, reason: str = "") -> TransitionResult:
        normalized = TaskState(target)
        if not self.can_transition(normalized):
            return TransitionResult(False, None, self._failure(normalized))
        return TransitionResult(True, self.transition(normalized, reason), None)

    def _failure(self, target: TaskState) -> TransitionFailure:
        allowed = self.legal_targets
        if self.is_terminal:
            message = f"terminal state {self.state.value} cannot transition to {target.value}"
            code = "TERMINAL_STATE_TRANSITION"
        else:
            message = f"illegal task transition {self.state.value} -> {target.value}"
            code = "ILLEGAL_TASK_TRANSITION"
        return TransitionFailure(code, self.state, target, allowed, message)

    def _validate_history(self) -> None:
        expected_source = TaskState.CASE_BOUND
        for index, transition in enumerate(self.history, start=1):
            if transition.sequence != index:
                raise ValueError("task transition history sequence is not contiguous")
            if transition.source is not expected_source:
                raise ValueError("task transition history source does not match prior state")
            if transition.target not in LEGAL_TRANSITIONS[transition.source]:
                raise ValueError(
                    f"task transition history contains illegal edge "
                    f"{transition.source.value} -> {transition.target.value}"
                )
            expected_source = transition.target
        if self.history and self.state is not expected_source:
            raise ValueError("task automaton state does not match transition history")
        if not self.history and self.state is not TaskState.CASE_BOUND:
            raise ValueError("non-initial task state requires transition history")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "xrd-rb-voe-task-automaton-v1",
            "state": self.state.value,
            "terminal": self.is_terminal,
            "history": [transition.to_dict() for transition in self.history],
        }
