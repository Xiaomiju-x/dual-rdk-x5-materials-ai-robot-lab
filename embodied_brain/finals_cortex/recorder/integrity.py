"""Input-order and sequence-integrity auditing for recorder samples."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .models import MessageSample

SEVERITY_ORDER = {"info": 0, "warning": 1, "error": 2, "critical": 3}


@dataclass(frozen=True)
class IntegrityIssue:
    code: str
    severity: str
    stream: str | None
    sequence: int | None
    timestamp_ns: int | None
    message: str

    def __post_init__(self) -> None:
        if self.severity not in SEVERITY_ORDER:
            raise ValueError(f"unsupported severity: {self.severity}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "stream": self.stream,
            "sequence": self.sequence,
            "timestamp_ns": self.timestamp_ns,
            "message": self.message,
        }


@dataclass(frozen=True)
class IntegrityReport:
    valid: bool
    issues: tuple[IntegrityIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        counts = Counter(issue.severity for issue in self.issues)
        return {
            "valid": self.valid,
            "issue_counts": {
                severity: counts.get(severity, 0)
                for severity in ("info", "warning", "error", "critical")
            },
            "issues": [issue.to_dict() for issue in self.issues],
        }


class IntegrityDetector:
    """Detect duplicates and arrival-order defects without discarding evidence."""

    def __init__(
        self,
        required_streams: Iterable[str],
        expected_start_sequences: Mapping[str, int] | None = None,
    ) -> None:
        self._required_streams = tuple(dict.fromkeys(required_streams))
        self._expected_start = dict(expected_start_sequences or {})
        self._accepted: list[MessageSample] = []
        self._issues: list[IntegrityIssue] = []
        self._by_key: dict[tuple[str, int], MessageSample] = {}
        self._timestamp_sequences: dict[str, dict[int, int]] = defaultdict(dict)
        self._last_arrival_sequence: dict[str, int] = {}
        self._last_arrival_timestamp: dict[str, int] = {}

    @property
    def accepted_samples(self) -> tuple[MessageSample, ...]:
        return tuple(self._accepted)

    def add(self, sample: MessageSample) -> bool:
        existing = self._by_key.get(sample.sample_key)
        if existing is not None:
            if existing.content_identity == sample.content_identity:
                self._issues.append(
                    IntegrityIssue(
                        "duplicate_sequence",
                        "warning",
                        sample.stream,
                        sample.sequence,
                        sample.timestamp_ns,
                        "identical stream/sequence sample was received more than once",
                    )
                )
            else:
                self._issues.append(
                    IntegrityIssue(
                        "conflicting_duplicate",
                        "critical",
                        sample.stream,
                        sample.sequence,
                        sample.timestamp_ns,
                        "stream/sequence was reused with conflicting sample content",
                    )
                )
            return False

        previous_sequence = self._last_arrival_sequence.get(sample.stream)
        if previous_sequence is not None and sample.sequence < previous_sequence:
            self._issues.append(
                IntegrityIssue(
                    "out_of_order_sequence",
                    "warning",
                    sample.stream,
                    sample.sequence,
                    sample.timestamp_ns,
                    f"sequence arrived after {previous_sequence}",
                )
            )

        previous_timestamp = self._last_arrival_timestamp.get(sample.stream)
        if previous_timestamp is not None and sample.timestamp_ns < previous_timestamp:
            self._issues.append(
                IntegrityIssue(
                    "out_of_order_timestamp",
                    "warning",
                    sample.stream,
                    sample.sequence,
                    sample.timestamp_ns,
                    f"timestamp arrived after {previous_timestamp}",
                )
            )

        prior_sequence = self._timestamp_sequences[sample.stream].get(sample.timestamp_ns)
        if prior_sequence is not None and prior_sequence != sample.sequence:
            self._issues.append(
                IntegrityIssue(
                    "duplicate_timestamp",
                    "warning",
                    sample.stream,
                    sample.sequence,
                    sample.timestamp_ns,
                    f"timestamp is also used by sequence {prior_sequence}",
                )
            )

        self._by_key[sample.sample_key] = sample
        self._timestamp_sequences[sample.stream][sample.timestamp_ns] = sample.sequence
        self._last_arrival_sequence[sample.stream] = sample.sequence
        self._last_arrival_timestamp[sample.stream] = sample.timestamp_ns
        self._accepted.append(sample)
        return True

    def report(self, extra_issues: Iterable[IntegrityIssue] = ()) -> IntegrityReport:
        issues = list(self._issues)
        streams: dict[str, list[int]] = defaultdict(list)
        for sample in self._accepted:
            streams[sample.stream].append(sample.sequence)

        for stream in self._required_streams:
            sequences = sorted(set(streams.get(stream, [])))
            if not sequences:
                issues.append(
                    IntegrityIssue(
                        "missing_stream",
                        "error",
                        stream,
                        None,
                        None,
                        "required stream has no accepted samples",
                    )
                )
                continue

            missing_ranges: list[tuple[int, int]] = []
            expected_start = self._expected_start.get(stream)
            if expected_start is not None and sequences[0] > expected_start:
                missing_ranges.append((expected_start, sequences[0] - 1))
            for left, right in zip(sequences, sequences[1:], strict=False):
                if right > left + 1:
                    missing_ranges.append((left + 1, right - 1))

            if missing_ranges:
                ranges = ", ".join(
                    str(start) if start == end else f"{start}-{end}"
                    for start, end in missing_ranges
                )
                issues.append(
                    IntegrityIssue(
                        "missing_sequence",
                        "error",
                        stream,
                        None,
                        None,
                        f"missing sequence range(s): {ranges}",
                    )
                )

        issues.extend(extra_issues)
        issues.sort(
            key=lambda issue: (
                -SEVERITY_ORDER[issue.severity],
                issue.stream or "",
                issue.sequence if issue.sequence is not None else -1,
                issue.code,
            )
        )
        valid = not any(
            SEVERITY_ORDER[issue.severity] >= SEVERITY_ORDER["error"]
            for issue in issues
        )
        return IntegrityReport(valid=valid, issues=tuple(issues))
