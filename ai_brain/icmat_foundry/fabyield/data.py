"""SECOM loading and chronological batch-disjoint splitting."""
from __future__ import annotations

import hashlib
import io
import re
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_MEMBERS = ("secom.data", "secom_labels.data")
LABEL_PATTERN = re.compile(
    r'^\s*(-1|1)\s+"(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})"\s*$'
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SecomDataset:
    features: np.ndarray
    labels: np.ndarray
    timestamps: tuple[datetime, ...]
    source_row_ids: np.ndarray
    source_zip: str
    source_sha256: str
    source_order_monotonic: bool

    @property
    def batch_ids(self) -> np.ndarray:
        # SECOM exposes no wafer/run identifier. Calendar date is the strictest
        # reproducible batch proxy supported by the public source.
        return np.asarray([stamp.date().isoformat() for stamp in self.timestamps])


@dataclass(frozen=True)
class TemporalSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    batch_kind: str = "calendar_date_proxy"

    def as_dict(
        self,
        dataset: SecomDataset,
        *,
        include_row_ids: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": "fabyield_temporal_split.v1",
            "policy": "chronological_whole_calendar_day_batches",
            "batch_kind": self.batch_kind,
            "source_has_true_wafer_or_run_ids": False,
            "claim_boundary": (
                "SECOM does not publish wafer/run identifiers. Calendar date is used as a "
                "conservative temporal batch proxy; this prevents same-day rows crossing "
                "partitions but cannot prove wafer-level disjointness."
            ),
            "partitions": {},
        }
        for name, indices in (
            ("train", self.train),
            ("validation", self.validation),
            ("test", self.test),
        ):
            labels = dataset.labels[indices]
            stamps = [dataset.timestamps[int(index)] for index in indices]
            batch_ids = sorted({stamp.date().isoformat() for stamp in stamps})
            record: dict[str, Any] = {
                "rows": int(indices.size),
                "pass_rows": int(np.sum(labels == 0)),
                "failure_rows": int(np.sum(labels == 1)),
                "failure_prevalence": float(np.mean(labels == 1)),
                "first_timestamp": min(stamps).isoformat(),
                "last_timestamp": max(stamps).isoformat(),
                "first_batch": batch_ids[0],
                "last_batch": batch_ids[-1],
                "batch_count": len(batch_ids),
                "batch_ids_sha256": sha256_bytes(
                    ("\n".join(batch_ids) + "\n").encode("utf-8")
                ),
                "source_row_ids_sha256": sha256_bytes(
                    np.asarray(dataset.source_row_ids[indices], dtype="<i8").tobytes()
                ),
            }
            if include_row_ids:
                record["source_row_ids"] = [
                    int(value) for value in dataset.source_row_ids[indices]
                ]
            payload["partitions"][name] = record

        payload["checks"] = validate_temporal_split(dataset, self)
        return payload


def load_secom_zip(path: Path) -> SecomDataset:
    """Load UCI SECOM directly from its immutable source archive."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    with zipfile.ZipFile(path) as archive:
        missing = sorted(set(REQUIRED_MEMBERS) - set(archive.namelist()))
        if missing:
            raise ValueError(f"SECOM archive missing members: {', '.join(missing)}")
        feature_text = archive.read("secom.data").decode("utf-8")
        label_text = archive.read("secom_labels.data").decode("utf-8")

    features = np.genfromtxt(
        io.StringIO(feature_text),
        dtype=np.float64,
        missing_values=("NaN", "nan"),
        filling_values=np.nan,
    )
    if features.ndim != 2:
        raise ValueError(f"expected a 2-D feature matrix, got {features.shape}")

    labels: list[int] = []
    timestamps: list[datetime] = []
    for line_number, line in enumerate(label_text.splitlines(), start=1):
        match = LABEL_PATTERN.match(line)
        if match is None:
            raise ValueError(f"invalid SECOM label row {line_number}: {line!r}")
        labels.append(1 if int(match.group(1)) == 1 else 0)
        timestamps.append(datetime.strptime(match.group(2), "%d/%m/%Y %H:%M:%S"))

    label_array = np.asarray(labels, dtype=np.int64)
    if features.shape[0] != label_array.size:
        raise ValueError(
            f"feature/label row mismatch: {features.shape[0]} != {label_array.size}"
        )
    if not np.isin(label_array, (0, 1)).all():
        raise ValueError("labels must map to binary values 0/1")

    source_order_monotonic = all(
        timestamps[index] <= timestamps[index + 1]
        for index in range(len(timestamps) - 1)
    )
    source_row_ids = np.arange(features.shape[0], dtype=np.int64)
    order = np.asarray(
        sorted(range(len(timestamps)), key=lambda index: (timestamps[index], index)),
        dtype=np.int64,
    )

    return SecomDataset(
        features=np.asarray(features[order], dtype=np.float64),
        labels=label_array[order],
        timestamps=tuple(timestamps[int(index)] for index in order),
        source_row_ids=source_row_ids[order],
        source_zip=str(path),
        source_sha256=sha256_file(path),
        source_order_monotonic=source_order_monotonic,
    )


def _partition_counts(
    ordered_batches: Sequence[str],
    counts_by_batch: dict[str, int],
) -> np.ndarray:
    counts = np.asarray([counts_by_batch[batch] for batch in ordered_batches], dtype=int)
    return np.cumsum(counts)


def temporal_batch_split(
    timestamps: Sequence[datetime],
    labels: np.ndarray,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    minimum_failures_per_partition: int = 1,
) -> TemporalSplit:
    """Split chronologically while assigning each calendar day to one partition."""
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between zero and one")
    if not (0.0 < validation_fraction < 1.0 - train_fraction):
        raise ValueError("validation_fraction must leave a non-empty test fraction")
    labels = np.asarray(labels, dtype=np.int64)
    if len(timestamps) != labels.size:
        raise ValueError("timestamps and labels must have equal length")
    if labels.size < 6:
        raise ValueError("at least six rows are required")

    batch_ids = np.asarray([stamp.date().isoformat() for stamp in timestamps])
    ordered_batches = sorted(set(batch_ids))
    if len(ordered_batches) < 3:
        raise ValueError("at least three calendar-day batches are required")
    counts_by_batch = {
        batch: int(np.sum(batch_ids == batch)) for batch in ordered_batches
    }
    cumulative = _partition_counts(ordered_batches, counts_by_batch)
    total = labels.size
    target_train = train_fraction * total
    target_validation_end = (train_fraction + validation_fraction) * total

    candidates: list[tuple[float, int, int]] = []
    for train_batch_end in range(len(ordered_batches) - 2):
        train_end = cumulative[train_batch_end]
        for validation_batch_end in range(
            train_batch_end + 1, len(ordered_batches) - 1
        ):
            validation_end = cumulative[validation_batch_end]
            masks = (
                batch_ids <= ordered_batches[train_batch_end],
                (batch_ids > ordered_batches[train_batch_end])
                & (batch_ids <= ordered_batches[validation_batch_end]),
                batch_ids > ordered_batches[validation_batch_end],
            )
            if any(
                int(np.sum(labels[mask] == 1)) < minimum_failures_per_partition
                or int(np.sum(labels[mask] == 0)) < 1
                for mask in masks
            ):
                continue
            size_error = abs(train_end - target_train) + abs(
                validation_end - target_validation_end
            )
            candidates.append((float(size_error), train_batch_end, validation_batch_end))

    if not candidates:
        raise ValueError(
            "unable to create three chronological batch-disjoint partitions with "
            "both classes represented"
        )
    _, train_batch_end, validation_batch_end = min(candidates)
    train_last = ordered_batches[train_batch_end]
    validation_last = ordered_batches[validation_batch_end]

    train = np.flatnonzero(batch_ids <= train_last)
    validation = np.flatnonzero(
        (batch_ids > train_last) & (batch_ids <= validation_last)
    )
    test = np.flatnonzero(batch_ids > validation_last)
    split = TemporalSplit(train=train, validation=validation, test=test)
    checks = validate_temporal_split_from_arrays(timestamps, labels, split)
    if not checks["all_passed"]:
        raise AssertionError(f"invalid temporal split: {checks}")
    return split


def validation_calibration_policy_split(
    timestamps: Sequence[datetime],
    labels: np.ndarray,
    validation_indices: np.ndarray,
    *,
    target_calibration_fraction: float = 0.50,
) -> tuple[np.ndarray, np.ndarray]:
    """Divide validation into earlier calibration and later policy batches."""
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    if validation_indices.size < 4:
        raise ValueError("validation partition is too small")
    batch_ids = np.asarray(
        [timestamps[int(index)].date().isoformat() for index in validation_indices]
    )
    ordered_batches = sorted(set(batch_ids))
    candidates: list[tuple[float, int]] = []
    for end in range(len(ordered_batches) - 1):
        calibration_mask = batch_ids <= ordered_batches[end]
        policy_mask = ~calibration_mask
        calibration_labels = labels[validation_indices[calibration_mask]]
        policy_labels = labels[validation_indices[policy_mask]]
        if (
            np.unique(calibration_labels).size < 2
            or np.unique(policy_labels).size < 2
        ):
            continue
        error = abs(
            int(np.sum(calibration_mask))
            - target_calibration_fraction * validation_indices.size
        )
        candidates.append((float(error), end))
    if not candidates:
        raise ValueError(
            "validation cannot be divided into chronological calibration and policy "
            "batches with both classes"
        )
    _, end = min(candidates)
    calibration_mask = batch_ids <= ordered_batches[end]
    calibration = validation_indices[calibration_mask]
    policy = validation_indices[~calibration_mask]
    return calibration, policy


def validate_temporal_split_from_arrays(
    timestamps: Sequence[datetime],
    labels: np.ndarray,
    split: TemporalSplit,
) -> dict[str, Any]:
    partitions = (split.train, split.validation, split.test)
    row_sets = [set(int(index) for index in values) for values in partitions]
    batch_sets = [
        {timestamps[index].date().isoformat() for index in rows} for rows in row_sets
    ]
    all_rows = set().union(*row_sets)
    expected_rows = set(range(len(timestamps)))
    checks = {
        "all_rows_assigned_once": (
            all_rows == expected_rows
            and sum(len(rows) for rows in row_sets) == len(expected_rows)
        ),
        "no_row_overlap": all(
            row_sets[left].isdisjoint(row_sets[right])
            for left in range(3)
            for right in range(left + 1, 3)
        ),
        "no_batch_overlap": all(
            batch_sets[left].isdisjoint(batch_sets[right])
            for left in range(3)
            for right in range(left + 1, 3)
        ),
        "strict_temporal_order": (
            max(timestamps[int(index)] for index in split.train)
            < min(timestamps[int(index)] for index in split.validation)
            and max(timestamps[int(index)] for index in split.validation)
            < min(timestamps[int(index)] for index in split.test)
        ),
        "both_classes_each_partition": all(
            np.unique(labels[indices]).size == 2 for indices in partitions
        ),
    }
    checks["all_passed"] = all(bool(value) for value in checks.values())
    return checks


def validate_temporal_split(
    dataset: SecomDataset,
    split: TemporalSplit,
) -> dict[str, Any]:
    return validate_temporal_split_from_arrays(
        dataset.timestamps,
        dataset.labels,
        split,
    )
