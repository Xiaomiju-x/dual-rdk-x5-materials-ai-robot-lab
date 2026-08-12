"""Immutable lot-disjoint split construction before model access."""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np

from .io_utils import sha256_text
from .parsing import Key

DEFAULT_SEED = 20260728
PARTITION_CODES = {"train": 0, "tune": 1, "calibration": 2, "test": 3}


@dataclass(frozen=True)
class FrozenSplit:
    partition_by_lot: dict[str, str]
    split_id: str
    manifest: dict[str, Any]

    def row_partitions(self, keys: tuple[Key, ...]) -> np.ndarray:
        return np.asarray(
            [PARTITION_CODES[self.partition_by_lot[key[0]]] for key in keys],
            dtype=np.uint8,
        )


def _lot_hash(seed: int, lot: str) -> str:
    return hashlib.sha256(f"{seed}|{lot}".encode()).hexdigest()


def _choose_development_subset(
    candidates: tuple[str, ...],
    *,
    count: int,
    name: str,
    seed: int,
    rows_by_lot: dict[str, int],
    bad_by_lot: dict[str, int],
    target_rows: int,
    total_rows: int,
    total_bad: int,
) -> tuple[str, ...]:
    best_score: tuple[int, int, str] | None = None
    best: tuple[str, ...] | None = None
    for combination in itertools.combinations(candidates, count):
        rows = sum(rows_by_lot[lot] for lot in combination)
        bad = sum(bad_by_lot[lot] for lot in combination)
        if bad == 0 or bad == rows:
            continue
        prevalence_distance_numerator = abs(bad * total_rows - total_bad * rows)
        digest = sha256_text(f"{seed}|{name}|{','.join(combination)}")
        score = (
            abs(rows - target_rows),
            prevalence_distance_numerator,
            digest,
        )
        if best_score is None or score < best_score:
            best_score = score
            best = combination
    if best is None:
        raise ValueError(f"unable to construct deterministic {name} partition")
    return best


def build_frozen_split(
    keys: tuple[Key, ...],
    labels: np.ndarray,
    *,
    seed: int = DEFAULT_SEED,
) -> FrozenSplit:
    labels = np.asarray(labels, dtype=np.uint8)
    if labels.shape != (len(keys),):
        raise ValueError("labels must align one-to-one with keys")
    rows_by_lot: dict[str, int] = {}
    bad_by_lot: dict[str, int] = {}
    for key, label in zip(keys, labels, strict=True):
        lot = key[0]
        rows_by_lot[lot] = rows_by_lot.get(lot, 0) + 1
        bad_by_lot[lot] = bad_by_lot.get(lot, 0) + int(label)
    ordered_lots = tuple(sorted(rows_by_lot, key=lambda lot: _lot_hash(seed, lot)))
    if len(ordered_lots) < 20:
        raise ValueError("at least 20 lots are required for the frozen split")

    test_count = max(1, round(0.20 * len(ordered_lots)))
    test_lots = ordered_lots[:test_count]
    test_rows = sum(rows_by_lot[lot] for lot in test_lots)
    test_bad = sum(bad_by_lot[lot] for lot in test_lots)
    if test_bad == 0 or test_bad == test_rows:
        raise ValueError("seeded SHA test partition must contain both classes")

    remaining = tuple(lot for lot in ordered_lots if lot not in test_lots)
    development_count = len(remaining)
    tune_count = max(2, round(0.15 * len(ordered_lots)))
    calibration_count = max(2, round(0.15 * len(ordered_lots)))
    if tune_count + calibration_count >= development_count:
        raise ValueError("not enough lots for train after tune/calibration allocation")
    target_rows = round(0.15 * len(keys))
    total_bad = int(labels.sum())
    tune_lots = _choose_development_subset(
        remaining,
        count=tune_count,
        name="tune",
        seed=seed,
        rows_by_lot=rows_by_lot,
        bad_by_lot=bad_by_lot,
        target_rows=target_rows,
        total_rows=len(keys),
        total_bad=total_bad,
    )
    after_tune = tuple(lot for lot in remaining if lot not in tune_lots)
    calibration_lots = _choose_development_subset(
        after_tune,
        count=calibration_count,
        name="calibration",
        seed=seed,
        rows_by_lot=rows_by_lot,
        bad_by_lot=bad_by_lot,
        target_rows=target_rows,
        total_rows=len(keys),
        total_bad=total_bad,
    )
    train_lots = tuple(
        lot for lot in after_tune if lot not in calibration_lots
    )
    partitions = {
        "train": train_lots,
        "tune": tune_lots,
        "calibration": calibration_lots,
        "test": test_lots,
    }
    for name, lots in partitions.items():
        rows = sum(rows_by_lot[lot] for lot in lots)
        bad = sum(bad_by_lot[lot] for lot in lots)
        if rows == 0 or bad == 0 or bad == rows:
            raise ValueError(f"{name} partition must contain both classes")
    partition_by_lot = {
        lot: partition for partition, lots in partitions.items() for lot in lots
    }
    if len(partition_by_lot) != len(ordered_lots):
        raise AssertionError("lot partition assignment is not complete")

    split_basis = {
        "schema": "fabrisk_kai_split_basis.v1",
        "seed": seed,
        "hash_expression": "sha256(f'{seed}|{lot}')",
        "ordered_lot_hashes": [
            {"lot": lot, "sha256": _lot_hash(seed, lot)} for lot in ordered_lots
        ],
        "partition_lots": {
            name: list(lots) for name, lots in partitions.items()
        },
    }
    split_id = sha256_text(str(split_basis))

    partition_records: dict[str, Any] = {}
    for name in ("train", "tune", "calibration"):
        lots = partitions[name]
        rows = sum(rows_by_lot[lot] for lot in lots)
        bad = sum(bad_by_lot[lot] for lot in lots)
        partition_records[name] = {
            "lots": list(lots),
            "lot_count": len(lots),
            "rows": rows,
            "class_counts": {"good": rows - bad, "bad": bad},
            "fraction": rows / len(keys),
        }
    partition_records["test"] = {
        "lot_count": len(test_lots),
        "rows": test_rows,
        "fraction": test_rows / len(keys),
        "contains_both_classes_verified_at_preregistration": True,
        "class_counts_withheld": True,
        "membership_only": True,
    }
    return FrozenSplit(
        partition_by_lot=partition_by_lot,
        split_id=split_id,
        manifest={
            "schema": "fabrisk_kai_lot_disjoint_split.v1",
            "split_id": split_id,
            "created_before_model_access": True,
            "seed": seed,
            "method": {
                "test": (
                    "first round(20% of lots) after seeded SHA-256 ordering; "
                    "only both-class eligibility checked"
                ),
                "development": (
                    "deterministic lot-subset selection for row-count and class-"
                    "prevalence proximity; SHA-256 digest breaks exact ties"
                ),
            },
            "partitions": partition_records,
            "lot_overlap_count": 0,
            "test_semantic_metrics_generated": False,
            "test_features_exported_to_training_cache": False,
            "prohibited_features": [
                "lot",
                "wafer",
                "source_row_number",
                "source_file_order",
                "response",
                "class",
            ],
            "split_basis": split_basis,
        },
    )
