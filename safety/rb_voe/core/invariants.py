"""Non-compensable invariant evaluation with structured failures."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, is_dataclass
from enum import Enum
from math import isfinite

from rb_voe.contracts.canonical import canonical_sha256, to_primitive
from rb_voe.contracts.models import EvidenceSource
from rb_voe.core.evidence_dag import EvidenceDAG, EvidenceIssue

PathPart = str | int
_MISSING = object()


class InvariantOperator(str, Enum):
    PRESENT = "PRESENT"
    EQUALS = "EQUALS"
    NOT_EQUALS = "NOT_EQUALS"
    IS_TRUE = "IS_TRUE"
    IS_FALSE = "IS_FALSE"
    IN = "IN"
    NOT_IN = "NOT_IN"
    LESS_THAN_OR_EQUAL = "LESS_THAN_OR_EQUAL"
    GREATER_THAN_OR_EQUAL = "GREATER_THAN_OR_EQUAL"


def _safe_primitive(value: object) -> object:
    if value is _MISSING:
        return "<missing>"
    try:
        return to_primitive(value)
    except (TypeError, ValueError):
        return repr(value)


def _path_text(path: tuple[PathPart, ...]) -> str:
    pieces: list[str] = []
    for part in path:
        if isinstance(part, int):
            pieces.append(f"[{part}]")
        else:
            pieces.append(("." if pieces else "") + part)
    return "".join(pieces) or "$"


def _resolve_path(root: object, path: tuple[PathPart, ...]) -> object:
    current = root
    for part in path:
        if isinstance(part, int):
            if (
                isinstance(current, Sequence)
                and not isinstance(current, (str, bytes, bytearray))
                and -len(current) <= part < len(current)
            ):
                current = current[part]
            else:
                return _MISSING
        elif isinstance(current, Mapping):
            if part not in current:
                return _MISSING
            current = current[part]
        elif is_dataclass(current) or hasattr(current, part):
            if not hasattr(current, part):
                return _MISSING
            current = getattr(current, part)
        else:
            return _MISSING
    return current


@dataclass(frozen=True, slots=True)
class InvariantFailure:
    invariant_id: str
    code: str
    message: str
    path: str
    expected: object
    actual: object
    related_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not all((self.invariant_id, self.code, self.message, self.path)):
            raise ValueError("invariant failure identity fields must be non-empty")
        object.__setattr__(self, "related_ids", tuple(sorted(set(self.related_ids))))

    def to_dict(self) -> dict[str, object]:
        return {
            "invariant_id": self.invariant_id,
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "expected": _safe_primitive(self.expected),
            "actual": _safe_primitive(self.actual),
            "related_ids": list(self.related_ids),
        }


@dataclass(frozen=True, slots=True)
class InvariantReport:
    failures: tuple[InvariantFailure, ...] = ()

    def __post_init__(self) -> None:
        if any(not isinstance(failure, InvariantFailure) for failure in self.failures):
            raise TypeError("invariant reports accept InvariantFailure instances only")
        object.__setattr__(
            self,
            "failures",
            tuple(
                sorted(
                    self.failures,
                    key=lambda failure: (
                        failure.invariant_id,
                        failure.code,
                        failure.path,
                        failure.related_ids,
                    ),
                )
            ),
        )

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def failure_codes(self) -> tuple[str, ...]:
        return tuple(failure.code for failure in self.failures)

    @property
    def content_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def has_failure(self, code: str) -> bool:
        return any(failure.code == code for failure in self.failures)

    def raise_for_failure(self) -> None:
        if not self.passed:
            raise HardInvariantViolation(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "xrd-rb-voe-invariant-report-v1",
            "passed": self.passed,
            "failures": [failure.to_dict() for failure in self.failures],
        }


class HardInvariantViolation(ValueError):
    def __init__(self, report: InvariantReport):
        self.report = report
        codes = ", ".join(report.failure_codes)
        super().__init__(f"hard invariants failed: {codes}")


@dataclass(frozen=True, slots=True)
class InvariantRule:
    invariant_id: str
    failure_code: str
    path: tuple[PathPart, ...] | str
    operator: InvariantOperator
    expected: object = None
    message: str = "hard invariant failed"
    related_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.invariant_id or not self.failure_code:
            raise ValueError("invariant_id and failure_code must be non-empty")
        normalized_path: tuple[PathPart, ...]
        if isinstance(self.path, str):
            normalized_path = tuple(part for part in self.path.split(".") if part)
        else:
            normalized_path = tuple(self.path)
        if not normalized_path:
            raise ValueError("invariant path must be non-empty")
        if any(not isinstance(part, (str, int)) or isinstance(part, bool) for part in normalized_path):
            raise TypeError("invariant path parts must be strings or integer indexes")
        object.__setattr__(self, "path", normalized_path)
        if not isinstance(self.operator, InvariantOperator):
            object.__setattr__(self, "operator", InvariantOperator(self.operator))
        if self.operator is InvariantOperator.PRESENT:
            object.__setattr__(self, "expected", "<present>")
        elif self.operator is InvariantOperator.IS_TRUE:
            object.__setattr__(self, "expected", True)
        elif self.operator is InvariantOperator.IS_FALSE:
            object.__setattr__(self, "expected", False)
        object.__setattr__(self, "related_ids", tuple(sorted(set(self.related_ids))))

    @property
    def path_text(self) -> str:
        return _path_text(self.path)

    def evaluate(self, context: object) -> InvariantFailure | None:
        actual = _resolve_path(context, self.path)
        passed = self._compare(actual)
        if passed:
            return None
        return InvariantFailure(
            invariant_id=self.invariant_id,
            code=self.failure_code,
            message=self.message,
            path=self.path_text,
            expected=_safe_primitive(self.expected),
            actual=_safe_primitive(actual),
            related_ids=self.related_ids,
        )

    def _compare(self, actual: object) -> bool:
        if self.operator is InvariantOperator.PRESENT:
            return actual is not _MISSING and actual is not None
        if self.operator is InvariantOperator.EQUALS:
            return actual is not _MISSING and actual == self.expected
        if self.operator is InvariantOperator.NOT_EQUALS:
            return actual is not _MISSING and actual != self.expected
        if self.operator is InvariantOperator.IS_TRUE:
            return actual is True
        if self.operator is InvariantOperator.IS_FALSE:
            return actual is False
        if self.operator in {InvariantOperator.IN, InvariantOperator.NOT_IN}:
            if actual is _MISSING:
                return False
            try:
                contained = actual in self.expected  # type: ignore[operator]
            except TypeError:
                return False
            return contained if self.operator is InvariantOperator.IN else not contained
        if self.operator in {
            InvariantOperator.LESS_THAN_OR_EQUAL,
            InvariantOperator.GREATER_THAN_OR_EQUAL,
        }:
            if isinstance(actual, bool) or isinstance(self.expected, bool):
                return False
            if not isinstance(actual, (int, float)) or not isinstance(self.expected, (int, float)):
                return False
            if not isfinite(float(actual)) or not isfinite(float(self.expected)):
                return False
            if self.operator is InvariantOperator.LESS_THAN_OR_EQUAL:
                return actual <= self.expected
            return actual >= self.expected
        raise AssertionError(f"unsupported invariant operator: {self.operator}")


@dataclass(frozen=True, slots=True)
class InvariantEvaluator:
    rules: tuple[InvariantRule, ...]

    def __post_init__(self) -> None:
        if any(not isinstance(rule, InvariantRule) for rule in self.rules):
            raise TypeError("invariant evaluators accept InvariantRule instances only")
        normalized = tuple(sorted(self.rules, key=lambda rule: rule.invariant_id))
        identifiers = [rule.invariant_id for rule in normalized]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("invariant ids must be unique")
        object.__setattr__(self, "rules", normalized)

    def evaluate(self, context: object) -> InvariantReport:
        failures = tuple(failure for rule in self.rules if (failure := rule.evaluate(context)) is not None)
        return InvariantReport(failures)


def evaluate_hard_invariants(context: object, rules: Iterable[InvariantRule]) -> InvariantReport:
    return InvariantEvaluator(tuple(rules)).evaluate(context)


def _failure_from_evidence_issue(issue: EvidenceIssue) -> InvariantFailure:
    return InvariantFailure(
        invariant_id=f"evidence.{issue.code.value.lower()}",
        code=issue.code.value,
        message=issue.detail,
        path="evidence_dag",
        expected="structurally valid and untampered evidence",
        actual=issue.code.value,
        related_ids=issue.evidence_ids,
    )


def evaluate_evidence_invariants(
    dag: EvidenceDAG,
    *,
    evidence_ids: Iterable[str] | None = None,
    minimum_independent: int = 0,
    expected_sha256: str | None = None,
) -> InvariantReport:
    """Apply hard graph integrity and independent-evidence requirements.

    Duplicate acquisitions and shared sources are legal provenance, but they cannot
    satisfy more than one independent-evidence slot.
    """
    if minimum_independent < 0:
        raise ValueError("minimum_independent cannot be negative")
    selected = tuple(sorted(set(dag.evidence_ids if evidence_ids is None else evidence_ids)))
    for evidence_id in selected:
        dag.record(evidence_id)

    failures = [_failure_from_evidence_issue(issue) for issue in dag.validate(expected_sha256).fatal_issues]
    independent = dag.independent_count(selected) if selected else 0
    if independent < minimum_independent:
        correlated = len(selected) >= minimum_independent and len(selected) > independent
        code = "CORRELATED_EVIDENCE_DOUBLE_COUNTED" if correlated else "INSUFFICIENT_INDEPENDENT_EVIDENCE"
        failures.append(
            InvariantFailure(
                invariant_id="evidence.minimum_independent",
                code=code,
                message="selected evidence does not meet the independent-source requirement",
                path="evidence_dag",
                expected={"minimum_independent": minimum_independent},
                actual={
                    "selected": len(selected),
                    "independent": independent,
                    "dependence_groups": [list(group) for group in dag.dependence_groups(selected)],
                },
                related_ids=selected,
            )
        )
    return InvariantReport(tuple(sorted(failures, key=lambda failure: (failure.invariant_id, failure.code))))


def evaluate_physical_evidence_invariants(
    dag: EvidenceDAG,
    *,
    evidence_ids: Iterable[str],
    minimum_independent_acquisitions: int,
    now_ms: int,
    maximum_age_ms: int,
    expected_sha256: str | None = None,
) -> InvariantReport:
    """Require fresh, independent physical acquisitions behind selected evidence."""
    if minimum_independent_acquisitions <= 0:
        raise ValueError("minimum_independent_acquisitions must be positive")
    if now_ms < 0 or maximum_age_ms < 0:
        raise ValueError("physical evidence time bounds must be non-negative")
    selected = tuple(sorted(set(evidence_ids)))
    failures = list(
        evaluate_evidence_invariants(
            dag,
            evidence_ids=selected,
            expected_sha256=expected_sha256,
        ).failures
    )
    physical_ids: set[str] = set()
    stale_ids: set[str] = set()
    future_ids: set[str] = set()
    for evidence_id in selected:
        lineage = (dag.record(evidence_id), *dag.ancestors(evidence_id))
        for record in lineage:
            if record.source is not EvidenceSource.PHYSICAL_ACQUISITION:
                continue
            physical_ids.add(record.evidence_id)
            age_ms = now_ms - record.observed_at_ms
            if age_ms < 0:
                future_ids.add(record.evidence_id)
            elif age_ms > maximum_age_ms:
                stale_ids.add(record.evidence_id)
    fresh_ids = tuple(sorted(physical_ids - stale_ids - future_ids))
    independent = dag.independent_count(fresh_ids) if fresh_ids else 0
    if stale_ids or future_ids:
        failures.append(
            InvariantFailure(
                invariant_id="evidence.physical_freshness",
                code="PHYSICAL_EVIDENCE_STALE_OR_FUTURE",
                message="physical acquisition evidence is outside the frozen freshness window",
                path="evidence_dag",
                expected={"maximum_age_ms": maximum_age_ms, "now_ms": now_ms},
                actual={
                    "stale_ids": sorted(stale_ids),
                    "future_ids": sorted(future_ids),
                },
                related_ids=tuple(sorted(stale_ids | future_ids)),
            )
        )
    if independent < minimum_independent_acquisitions:
        failures.append(
            InvariantFailure(
                invariant_id="evidence.minimum_physical_acquisitions",
                code="INSUFFICIENT_INDEPENDENT_PHYSICAL_ACQUISITIONS",
                message="selected evidence lacks enough fresh independent physical roots",
                path="evidence_dag",
                expected={"minimum_independent_acquisitions": minimum_independent_acquisitions},
                actual={"fresh_physical_ids": list(fresh_ids), "independent": independent},
                related_ids=selected,
            )
        )
    return InvariantReport(tuple(sorted(failures, key=lambda failure: (failure.invariant_id, failure.code))))
