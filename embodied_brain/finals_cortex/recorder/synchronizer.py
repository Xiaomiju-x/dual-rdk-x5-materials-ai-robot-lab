"""Deterministic sequence-first, timestamp-fallback sample synchronization."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .models import MessageSample


def _quantile(values: list[int], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


@dataclass(frozen=True)
class TimeOffsetStats:
    stream: str
    count: int
    min_ns: int | None
    max_ns: int | None
    mean_ns: float | None
    mean_abs_ns: float | None
    p50_abs_ns: float | None
    p95_abs_ns: float | None
    stddev_ns: float | None

    @classmethod
    def from_offsets(cls, stream: str, offsets: Iterable[int]) -> TimeOffsetStats:
        values = list(offsets)
        absolute = [abs(value) for value in values]
        return cls(
            stream=stream,
            count=len(values),
            min_ns=min(values) if values else None,
            max_ns=max(values) if values else None,
            mean_ns=statistics.fmean(values) if values else None,
            mean_abs_ns=statistics.fmean(absolute) if values else None,
            p50_abs_ns=_quantile(absolute, 0.50),
            p95_abs_ns=_quantile(absolute, 0.95),
            stddev_ns=statistics.pstdev(values) if values else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream": self.stream,
            "count": self.count,
            "min_ns": self.min_ns,
            "max_ns": self.max_ns,
            "mean_ns": self.mean_ns,
            "mean_abs_ns": self.mean_abs_ns,
            "p50_abs_ns": self.p50_abs_ns,
            "p95_abs_ns": self.p95_abs_ns,
            "stddev_ns": self.stddev_ns,
        }


@dataclass(frozen=True)
class SyncGroup:
    group_id: str
    anchor_stream: str
    anchor_sequence: int
    anchor_timestamp_ns: int
    samples: dict[str, MessageSample]
    match_modes: dict[str, str]
    offsets_ns: dict[str, int]
    missing_streams: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_streams

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "anchor_stream": self.anchor_stream,
            "anchor_sequence": self.anchor_sequence,
            "anchor_timestamp_ns": self.anchor_timestamp_ns,
            "complete": self.complete,
            "missing_streams": list(self.missing_streams),
            "match_modes": dict(sorted(self.match_modes.items())),
            "offsets_ns": dict(sorted(self.offsets_ns.items())),
            "sample_sequences": {
                stream: sample.sequence for stream, sample in sorted(self.samples.items())
            },
            "sample_timestamps_ns": {
                stream: sample.timestamp_ns
                for stream, sample in sorted(self.samples.items())
            },
        }


@dataclass(frozen=True)
class SynchronizationResult:
    groups: tuple[SyncGroup, ...]
    unmatched_samples: tuple[MessageSample, ...]
    offset_stats: tuple[TimeOffsetStats, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_count": len(self.groups),
            "complete_group_count": sum(group.complete for group in self.groups),
            "incomplete_group_count": sum(not group.complete for group in self.groups),
            "groups": [group.to_dict() for group in self.groups],
            "unmatched_samples": [
                {"stream": sample.stream, "sequence": sample.sequence}
                for sample in self.unmatched_samples
            ],
            "offset_stats": [stats.to_dict() for stats in self.offset_stats],
        }


class SampleSynchronizer:
    """Build one non-reusing group per anchor sample."""

    def __init__(
        self,
        required_streams: Iterable[str],
        anchor_stream: str,
        tolerance_ns: int,
    ) -> None:
        streams = tuple(dict.fromkeys(required_streams))
        if not streams:
            raise ValueError("required_streams must not be empty")
        if anchor_stream not in streams:
            raise ValueError("anchor_stream must be a required stream")
        if isinstance(tolerance_ns, bool) or not isinstance(tolerance_ns, int):
            raise ValueError("tolerance_ns must be an integer")
        if tolerance_ns < 0:
            raise ValueError("tolerance_ns must be non-negative")
        self.required_streams = streams
        self.anchor_stream = anchor_stream
        self.tolerance_ns = tolerance_ns

    def synchronize(self, samples: Iterable[MessageSample]) -> SynchronizationResult:
        by_stream: dict[str, list[MessageSample]] = defaultdict(list)
        all_samples = tuple(samples)
        for sample in all_samples:
            by_stream[sample.stream].append(sample)
        for stream in by_stream:
            by_stream[stream].sort(key=lambda item: (item.timestamp_ns, item.sequence))

        used: set[tuple[str, int]] = set()
        groups: list[SyncGroup] = []
        offsets: dict[str, list[int]] = defaultdict(list)

        for anchor in by_stream.get(self.anchor_stream, []):
            used.add(anchor.sample_key)
            matched = {self.anchor_stream: anchor}
            modes = {self.anchor_stream: "anchor"}
            group_offsets = {self.anchor_stream: 0}
            missing: list[str] = []
            offsets[self.anchor_stream].append(0)

            for stream in self.required_streams:
                if stream == self.anchor_stream:
                    continue
                available = [
                    candidate
                    for candidate in by_stream.get(stream, [])
                    if candidate.sample_key not in used
                ]
                exact = [
                    candidate
                    for candidate in available
                    if candidate.sequence == anchor.sequence
                    and abs(candidate.timestamp_ns - anchor.timestamp_ns)
                    <= self.tolerance_ns
                ]
                if exact:
                    candidate = min(
                        exact,
                        key=lambda item: (
                            abs(item.timestamp_ns - anchor.timestamp_ns),
                            item.timestamp_ns,
                        ),
                    )
                    mode = "sequence"
                else:
                    within_tolerance = [
                        candidate
                        for candidate in available
                        if abs(candidate.timestamp_ns - anchor.timestamp_ns)
                        <= self.tolerance_ns
                    ]
                    if not within_tolerance:
                        missing.append(stream)
                        continue
                    candidate = min(
                        within_tolerance,
                        key=lambda item: (
                            abs(item.timestamp_ns - anchor.timestamp_ns),
                            abs(item.sequence - anchor.sequence),
                            item.sequence,
                        ),
                    )
                    mode = "timestamp"

                used.add(candidate.sample_key)
                offset = candidate.timestamp_ns - anchor.timestamp_ns
                matched[stream] = candidate
                modes[stream] = mode
                group_offsets[stream] = offset
                offsets[stream].append(offset)

            groups.append(
                SyncGroup(
                    group_id=f"{self.anchor_stream}:{anchor.sequence}",
                    anchor_stream=self.anchor_stream,
                    anchor_sequence=anchor.sequence,
                    anchor_timestamp_ns=anchor.timestamp_ns,
                    samples=matched,
                    match_modes=modes,
                    offsets_ns=group_offsets,
                    missing_streams=tuple(missing),
                )
            )

        unmatched = tuple(
            sorted(
                (sample for sample in all_samples if sample.sample_key not in used),
                key=lambda item: (item.stream, item.timestamp_ns, item.sequence),
            )
        )
        stats = tuple(
            TimeOffsetStats.from_offsets(stream, offsets.get(stream, []))
            for stream in self.required_streams
        )
        return SynchronizationResult(tuple(groups), unmatched, stats)
