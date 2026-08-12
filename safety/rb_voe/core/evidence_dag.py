"""Deterministic typed evidence graph and source-dependence accounting."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from rb_voe.contracts.canonical import canonical_sha256
from rb_voe.contracts.models import EvidenceRecord, EvidenceSource


class EvidenceIssueCode(str, Enum):
    DUPLICATE_EVIDENCE_ID = "DUPLICATE_EVIDENCE_ID"
    EVIDENCE_ID_TAMPER = "EVIDENCE_ID_TAMPER"
    MISSING_PARENT = "MISSING_PARENT"
    CYCLE = "CYCLE"
    ORPHAN_DERIVED_EVIDENCE = "ORPHAN_DERIVED_EVIDENCE"
    LINEAGE_MISMATCH = "LINEAGE_MISMATCH"
    DUPLICATE_ACQUISITION = "DUPLICATE_ACQUISITION"
    SOURCE_DEPENDENCE = "SOURCE_DEPENDENCE"
    RECORD_TAMPERED = "RECORD_TAMPERED"
    GRAPH_DIGEST_MISMATCH = "GRAPH_DIGEST_MISMATCH"


@dataclass(frozen=True, slots=True)
class EvidenceIssue:
    code: EvidenceIssueCode
    evidence_ids: tuple[str, ...]
    detail: str
    fatal: bool
    dependence_tokens: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code.value,
            "evidence_ids": list(self.evidence_ids),
            "detail": self.detail,
            "fatal": self.fatal,
            "dependence_tokens": list(self.dependence_tokens),
        }


@dataclass(frozen=True, slots=True)
class EvidenceValidation:
    issues: tuple[EvidenceIssue, ...] = ()

    @property
    def valid(self) -> bool:
        return not any(issue.fatal for issue in self.issues)

    @property
    def fatal_issues(self) -> tuple[EvidenceIssue, ...]:
        return tuple(issue for issue in self.issues if issue.fatal)

    @property
    def dependence_issues(self) -> tuple[EvidenceIssue, ...]:
        return tuple(
            issue
            for issue in self.issues
            if issue.code in {EvidenceIssueCode.DUPLICATE_ACQUISITION, EvidenceIssueCode.SOURCE_DEPENDENCE}
        )

    def has(self, code: EvidenceIssueCode | str) -> bool:
        normalized = EvidenceIssueCode(code)
        return any(issue.code is normalized for issue in self.issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "issues": [issue.to_dict() for issue in self.issues],
        }


class EvidenceGraphError(ValueError):
    """Raised when an evidence graph contains structural corruption."""

    def __init__(self, validation: EvidenceValidation):
        self.validation = validation
        summary = ", ".join(issue.code.value for issue in validation.fatal_issues)
        super().__init__(f"invalid evidence DAG: {summary or 'unknown structural failure'}")


def _issue_sort_key(issue: EvidenceIssue) -> tuple[object, ...]:
    return (issue.code.value, issue.evidence_ids, issue.dependence_tokens, issue.detail)


def _failure_domains(record: EvidenceRecord) -> tuple[str, ...]:
    domains: set[str] = set()
    for key in ("failure_domain", "failure_domains", "source_dependencies"):
        value = record.metadata.get(key)
        if isinstance(value, str) and value:
            domains.add(value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            domains.update(item for item in value if isinstance(item, str) and item)
    return tuple(sorted(domains))


def _strongly_connected_components(records: Mapping[str, EvidenceRecord]) -> tuple[tuple[str, ...], ...]:
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    components: list[tuple[str, ...]] = []

    def visit(evidence_id: str) -> None:
        nonlocal index
        indices[evidence_id] = index
        lowlinks[evidence_id] = index
        index += 1
        stack.append(evidence_id)
        on_stack.add(evidence_id)

        parents = sorted(
            parent_id for parent_id in records[evidence_id].parent_evidence_ids if parent_id in records
        )
        for parent_id in parents:
            if parent_id not in indices:
                visit(parent_id)
                lowlinks[evidence_id] = min(lowlinks[evidence_id], lowlinks[parent_id])
            elif parent_id in on_stack:
                lowlinks[evidence_id] = min(lowlinks[evidence_id], indices[parent_id])

        if lowlinks[evidence_id] != indices[evidence_id]:
            return
        component: list[str] = []
        while True:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == evidence_id:
                break
        components.append(tuple(sorted(component)))

    for evidence_id in sorted(records):
        if evidence_id not in indices:
            visit(evidence_id)
    return tuple(sorted(components))


def _structural_validation(records: tuple[EvidenceRecord, ...]) -> EvidenceValidation:
    issues: list[EvidenceIssue] = []
    grouped: dict[str, list[EvidenceRecord]] = defaultdict(list)
    for record in records:
        if not isinstance(record, EvidenceRecord):
            raise TypeError("evidence DAG accepts EvidenceRecord instances only")
        grouped[record.evidence_id].append(record)

    for evidence_id, duplicates in sorted(grouped.items()):
        if len(duplicates) < 2:
            continue
        hashes = {record.content_sha256 for record in duplicates}
        if len(hashes) > 1:
            code = EvidenceIssueCode.EVIDENCE_ID_TAMPER
            detail = "one evidence_id resolves to conflicting record contents"
        else:
            code = EvidenceIssueCode.DUPLICATE_EVIDENCE_ID
            detail = "evidence_id is registered more than once"
        issues.append(EvidenceIssue(code, (evidence_id,), detail, True))

    if any(len(group) > 1 for group in grouped.values()):
        return EvidenceValidation(tuple(sorted(issues, key=_issue_sort_key)))

    by_id = {record.evidence_id: record for record in records}
    for evidence_id in sorted(by_id):
        record = by_id[evidence_id]
        for parent_id in sorted(record.parent_evidence_ids):
            if parent_id not in by_id:
                issues.append(
                    EvidenceIssue(
                        EvidenceIssueCode.MISSING_PARENT,
                        (evidence_id, parent_id),
                        f"parent {parent_id!r} is not present in the DAG",
                        True,
                    )
                )
        if record.source is EvidenceSource.DERIVED_COMPUTE and not record.parent_evidence_ids:
            issues.append(
                EvidenceIssue(
                    EvidenceIssueCode.ORPHAN_DERIVED_EVIDENCE,
                    (evidence_id,),
                    "derived compute evidence requires at least one parent",
                    True,
                )
            )

    lineages: dict[str, list[str]] = defaultdict(list)
    for record in records:
        lineages[record.lineage_sha256].append(record.evidence_id)
    if len(lineages) > 1:
        issues.append(
            EvidenceIssue(
                EvidenceIssueCode.LINEAGE_MISMATCH,
                tuple(sorted(record.evidence_id for record in records)),
                "all evidence in one DAG must bind to the same sample lineage",
                True,
            )
        )

    for component in _strongly_connected_components(by_id):
        self_cycle = len(component) == 1 and component[0] in by_id[component[0]].parent_evidence_ids
        if len(component) > 1 or self_cycle:
            issues.append(
                EvidenceIssue(
                    EvidenceIssueCode.CYCLE,
                    component,
                    "parent relationships form a directed cycle",
                    True,
                )
            )

    return EvidenceValidation(tuple(sorted(issues, key=_issue_sort_key)))


def validate_records(records: Iterable[EvidenceRecord]) -> EvidenceValidation:
    """Validate structure and source dependence without raising on bad input."""
    materialized = tuple(records)
    structural = _structural_validation(materialized)
    if not structural.valid:
        return structural
    return EvidenceDAG(materialized).validate()


class EvidenceDAG:
    """Immutable evidence DAG with a construction-time integrity snapshot.

    Correlated observations are valid graph members. They are reported and grouped,
    but never counted as independent acquisitions.
    """

    __slots__ = ("_baseline_digest", "_by_id", "_record_hashes", "_records")

    def __init__(self, records: Iterable[EvidenceRecord] = ()) -> None:
        materialized = tuple(records)
        validation = _structural_validation(materialized)
        if not validation.valid:
            raise EvidenceGraphError(validation)
        self._records = tuple(sorted(materialized, key=lambda record: record.evidence_id))
        self._by_id = {record.evidence_id: record for record in self._records}
        self._record_hashes = tuple((record.evidence_id, record.content_sha256) for record in self._records)
        self._baseline_digest = self._calculate_digest(self._record_hashes)

    @classmethod
    def from_records(cls, records: Iterable[EvidenceRecord]) -> EvidenceDAG:
        return cls(records)

    @staticmethod
    def _calculate_digest(record_hashes: tuple[tuple[str, str], ...]) -> str:
        return canonical_sha256(
            {
                "schema_version": "xrd-rb-voe-evidence-dag-v1",
                "records": [
                    {"evidence_id": evidence_id, "record_sha256": digest}
                    for evidence_id, digest in record_hashes
                ],
            }
        )

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[EvidenceRecord]:
        return iter(self._records)

    def __contains__(self, evidence_id: object) -> bool:
        return evidence_id in self._by_id

    @property
    def records(self) -> tuple[EvidenceRecord, ...]:
        return self._records

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(self._by_id)

    @property
    def content_sha256(self) -> str:
        """Construction-time digest used to detect later mutation of nested metadata."""
        return self._baseline_digest

    @property
    def current_sha256(self) -> str:
        hashes = tuple((record.evidence_id, record.content_sha256) for record in self._records)
        return self._calculate_digest(hashes)

    def record(self, evidence_id: str) -> EvidenceRecord:
        try:
            return self._by_id[evidence_id]
        except KeyError as exc:
            raise KeyError(f"unknown evidence_id: {evidence_id}") from exc

    def parents(self, evidence_id: str) -> tuple[EvidenceRecord, ...]:
        record = self.record(evidence_id)
        return tuple(self._by_id[parent_id] for parent_id in sorted(record.parent_evidence_ids))

    def children(self, evidence_id: str) -> tuple[EvidenceRecord, ...]:
        self.record(evidence_id)
        return tuple(record for record in self._records if evidence_id in record.parent_evidence_ids)

    def ancestors(self, evidence_id: str) -> tuple[EvidenceRecord, ...]:
        self.record(evidence_id)
        pending = list(self._by_id[evidence_id].parent_evidence_ids)
        found: set[str] = set()
        while pending:
            parent_id = pending.pop()
            if parent_id in found:
                continue
            found.add(parent_id)
            pending.extend(self._by_id[parent_id].parent_evidence_ids)
        return tuple(self._by_id[parent_id] for parent_id in sorted(found))

    def topological_order(self) -> tuple[str, ...]:
        indegree = {
            evidence_id: len(record.parent_evidence_ids) for evidence_id, record in self._by_id.items()
        }
        children: dict[str, list[str]] = defaultdict(list)
        for evidence_id, record in self._by_id.items():
            for parent_id in record.parent_evidence_ids:
                children[parent_id].append(evidence_id)
        ready = sorted(evidence_id for evidence_id, degree in indegree.items() if degree == 0)
        ordered: list[str] = []
        while ready:
            evidence_id = ready.pop(0)
            ordered.append(evidence_id)
            for child_id in sorted(children[evidence_id]):
                indegree[child_id] -= 1
                if indegree[child_id] == 0:
                    ready.append(child_id)
                    ready.sort()
        if len(ordered) != len(self._records):
            raise EvidenceGraphError(_structural_validation(self._records))
        return tuple(ordered)

    def append(self, record: EvidenceRecord) -> EvidenceDAG:
        # Do not let appending a new node re-baseline mutated nested metadata.
        self.assert_valid()
        return EvidenceDAG((*self._records, record))

    def dependency_tokens(self, evidence_id: str) -> tuple[str, ...]:
        """Return conservative shared-failure tokens inherited through the DAG."""
        tokens: set[str] = set()
        record_ids = (evidence_id, *(record.evidence_id for record in self.ancestors(evidence_id)))
        for record_id in record_ids:
            record = self._by_id[record_id]
            tokens.add(f"source:{record.source_id}")
            if record.acquisition_id:
                tokens.add(f"acquisition:{record.acquisition_id}")
            tokens.update(f"domain:{domain}" for domain in _failure_domains(record))
        return tuple(sorted(tokens))

    def failure_domains(self, evidence_ids: Iterable[str] | None = None) -> tuple[str, ...]:
        """Return declared failure domains for selected evidence and its ancestors."""
        selected = tuple(sorted(set(self.evidence_ids if evidence_ids is None else evidence_ids)))
        domains: set[str] = set()
        for evidence_id in selected:
            record = self.record(evidence_id)
            domains.update(_failure_domains(record))
            for ancestor in self.ancestors(evidence_id):
                domains.update(_failure_domains(ancestor))
        return tuple(sorted(domains))

    def dependence_groups(self, evidence_ids: Iterable[str] | None = None) -> tuple[tuple[str, ...], ...]:
        selected = tuple(sorted(set(self.evidence_ids if evidence_ids is None else evidence_ids)))
        for evidence_id in selected:
            self.record(evidence_id)
        tokens = {evidence_id: set(self.dependency_tokens(evidence_id)) for evidence_id in selected}
        remaining = set(selected)
        groups: list[tuple[str, ...]] = []
        while remaining:
            seed = min(remaining)
            component = {seed}
            component_tokens = set(tokens[seed])
            changed = True
            while changed:
                changed = False
                for candidate in sorted(remaining - component):
                    if component_tokens.intersection(tokens[candidate]):
                        component.add(candidate)
                        component_tokens.update(tokens[candidate])
                        changed = True
            remaining.difference_update(component)
            groups.append(tuple(sorted(component)))
        return tuple(sorted(groups))

    def independent_count(self, evidence_ids: Iterable[str] | None = None) -> int:
        return len(self.dependence_groups(evidence_ids))

    def are_independent(self, evidence_ids: Iterable[str]) -> bool:
        selected = tuple(sorted(set(evidence_ids)))
        return len(self.dependence_groups(selected)) == len(selected)

    def independent_acquisition_count(self, evidence_ids: Iterable[str] | None = None) -> int:
        selected = self.evidence_ids if evidence_ids is None else tuple(evidence_ids)
        physical = tuple(
            evidence_id
            for evidence_id in selected
            if self.record(evidence_id).source is EvidenceSource.PHYSICAL_ACQUISITION
        )
        return self.independent_count(physical) if physical else 0

    def validate(self, expected_sha256: str | None = None) -> EvidenceValidation:
        issues: list[EvidenceIssue] = list(_structural_validation(self._records).issues)
        baseline_hashes = dict(self._record_hashes)
        current_hashes: list[tuple[str, str]] = []
        for record in self._records:
            try:
                current_hash = record.content_sha256
            except (TypeError, ValueError) as exc:
                issues.append(
                    EvidenceIssue(
                        EvidenceIssueCode.RECORD_TAMPERED,
                        (record.evidence_id,),
                        f"record content is no longer canonical: {type(exc).__name__}",
                        True,
                    )
                )
                continue
            current_hashes.append((record.evidence_id, current_hash))
            if current_hash != baseline_hashes[record.evidence_id]:
                issues.append(
                    EvidenceIssue(
                        EvidenceIssueCode.RECORD_TAMPERED,
                        (record.evidence_id,),
                        "record content changed after DAG construction",
                        True,
                    )
                )
        current_digest = (
            self._calculate_digest(tuple(current_hashes))
            if len(current_hashes) == len(self._records)
            else None
        )
        if expected_sha256 is not None and current_digest != expected_sha256:
            issues.append(
                EvidenceIssue(
                    EvidenceIssueCode.GRAPH_DIGEST_MISMATCH,
                    self.evidence_ids,
                    "current graph digest does not match the expected digest",
                    True,
                )
            )

        acquisitions: dict[str, list[str]] = defaultdict(list)
        for record in self._records:
            if record.acquisition_id:
                acquisitions[record.acquisition_id].append(record.evidence_id)
        for acquisition_id, evidence_ids in sorted(acquisitions.items()):
            if len(evidence_ids) > 1:
                issues.append(
                    EvidenceIssue(
                        EvidenceIssueCode.DUPLICATE_ACQUISITION,
                        tuple(sorted(evidence_ids)),
                        f"acquisition {acquisition_id!r} contributes multiple evidence records",
                        False,
                        (f"acquisition:{acquisition_id}",),
                    )
                )

        for group in self.dependence_groups():
            if len(group) < 2:
                continue
            shared: set[str] = set()
            for index, left_id in enumerate(group):
                left_tokens = set(self.dependency_tokens(left_id))
                for right_id in group[index + 1 :]:
                    shared.update(left_tokens.intersection(self.dependency_tokens(right_id)))
            issues.append(
                EvidenceIssue(
                    EvidenceIssueCode.SOURCE_DEPENDENCE,
                    group,
                    "evidence shares an acquisition, source, ancestor, or registered failure domain",
                    False,
                    tuple(sorted(shared)),
                )
            )
        return EvidenceValidation(tuple(sorted(set(issues), key=_issue_sort_key)))

    def assert_valid(self, expected_sha256: str | None = None) -> None:
        validation = self.validate(expected_sha256)
        if not validation.valid:
            raise EvidenceGraphError(validation)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "xrd-rb-voe-evidence-dag-v1",
            "content_sha256": self.content_sha256,
            "records": [record.to_dict() for record in self._records],
            "topological_order": list(self.topological_order()),
        }
