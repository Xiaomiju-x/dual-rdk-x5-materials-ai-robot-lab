"""Strict parsing and key alignment for the two KAI Zenodo releases."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io_utils import sha256_file

STEPS = 176
SENSOR_COLUMNS = tuple(f"sensor_{index}" for index in range(1, 57))
SUMMARY_COLUMNS = tuple(f"KN{index}" for index in range(1, 51))
KEY_COLUMNS = ("lot", "wafer")
LOT_PATTERN = re.compile(r"^lot([1-9][0-9]*)$")
TIMESTAMP_PATTERN = re.compile(r"^timestamp_([0-9]+)$")
SOURCE_ENCODING = "cp1252"

Key = tuple[str, int]


@dataclass(frozen=True)
class ParsedEquipment:
    keys: tuple[Key, ...]
    values: np.ndarray
    observed_mask: np.ndarray
    audit: dict[str, Any]


@dataclass(frozen=True)
class ParsedSummary:
    keys: tuple[Key, ...]
    values: np.ndarray
    observed_mask: np.ndarray
    audit: dict[str, Any]


@dataclass(frozen=True)
class ParsedResponse:
    keys: tuple[Key, ...]
    response_by_key: dict[Key, float]
    class_by_key: dict[Key, str]
    audit: dict[str, Any]


@dataclass(frozen=True)
class JoinedKAIData:
    keys: tuple[Key, ...]
    temporal_values: np.ndarray
    temporal_observed_mask: np.ndarray
    summary_values: np.ndarray
    summary_observed_mask: np.ndarray
    labels: np.ndarray
    responses: np.ndarray
    audit: dict[str, Any]


def _natural_key(key: Key) -> tuple[int, int]:
    lot_match = LOT_PATTERN.fullmatch(key[0])
    if lot_match is None:
        raise ValueError(f"invalid lot identifier: {key[0]!r}")
    return int(lot_match.group(1)), int(key[1])


def _read_semicolon(path: Path, expected_columns: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep=";",
        dtype=str,
        encoding=SOURCE_ENCODING,
        keep_default_na=False,
        na_filter=False,
        on_bad_lines="error",
    )
    actual = tuple(str(column) for column in frame.columns)
    if actual != expected_columns:
        raise ValueError(
            f"{path.name}: unexpected columns; expected={expected_columns}, actual={actual}"
        )
    if frame.empty:
        raise ValueError(f"{path.name}: empty table")
    for column in KEY_COLUMNS:
        frame[column] = frame[column].astype(str).str.strip()
    if not frame["lot"].map(lambda value: LOT_PATTERN.fullmatch(value) is not None).all():
        bad = frame.loc[
            ~frame["lot"].map(lambda value: LOT_PATTERN.fullmatch(value) is not None),
            "lot",
        ].head(5)
        raise ValueError(f"{path.name}: invalid lot identifiers: {bad.tolist()}")
    wafer_numbers = pd.to_numeric(frame["wafer"], errors="coerce")
    valid_wafer = (
        wafer_numbers.notna()
        & np.isfinite(wafer_numbers.to_numpy(dtype=np.float64))
        & (wafer_numbers > 0)
        & (wafer_numbers % 1 == 0)
    )
    if not bool(valid_wafer.all()):
        raise ValueError(f"{path.name}: wafer identifiers must be positive integers")
    frame["wafer"] = wafer_numbers.astype(np.int64)
    return frame


def _strict_numeric_matrix(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    values = np.empty((len(frame), len(columns)), dtype=np.float32)
    observed = np.empty((len(frame), len(columns)), dtype=bool)
    column_audit: dict[str, Any] = {}
    for index, column in enumerate(columns):
        raw = frame[column].astype(str).str.strip()
        missing = raw.eq("").to_numpy(dtype=bool)
        converted = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(converted)
        invalid = (~missing) & (~finite)
        valid = (~missing) & finite
        column_values = converted.astype(np.float32)
        column_values[~valid] = np.nan
        values[:, index] = column_values
        observed[:, index] = valid
        invalid_tokens = sorted(set(raw[invalid].tolist()))
        column_audit[column] = {
            "rows": int(len(raw)),
            "valid": int(valid.sum()),
            "missing": int(missing.sum()),
            "invalid": int(invalid.sum()),
            "invalid_examples": invalid_tokens[:8],
            "invalid_stored_as_nan": True,
            "invalid_stored_as_zero": False,
        }
    return (
        values,
        observed,
        {
            "columns": column_audit,
            "valid_total": int(observed.sum()),
            "missing_total": int(
                sum(record["missing"] for record in column_audit.values())
            ),
            "invalid_total": int(
                sum(record["invalid"] for record in column_audit.values())
            ),
            "mask_semantics": "true=finite parsed measurement; false=missing_or_invalid",
        },
    )


def _frame_keys(frame: pd.DataFrame) -> list[Key]:
    return [
        (str(lot), int(wafer))
        for lot, wafer in zip(frame["lot"], frame["wafer"], strict=True)
    ]


def parse_equipment(
    path: Path,
    *,
    first_sensor: int,
    last_sensor: int,
) -> ParsedEquipment:
    sensor_columns = tuple(
        f"sensor_{index}" for index in range(first_sensor, last_sensor + 1)
    )
    expected = ("lot", "wafer", "timestamp", *sensor_columns)
    frame = _read_semicolon(path, expected)
    timestamp_text = frame["timestamp"].astype(str).str.strip()
    timestamp_index = timestamp_text.map(
        lambda value: (
            int(match.group(1))
            if (match := TIMESTAMP_PATTERN.fullmatch(value)) is not None
            else -1
        )
    ).to_numpy(dtype=np.int64)
    if np.any((timestamp_index < 0) | (timestamp_index >= STEPS)):
        bad = timestamp_text[(timestamp_index < 0) | (timestamp_index >= STEPS)]
        raise ValueError(f"{path.name}: invalid timestamps: {bad.head(5).tolist()}")
    frame = frame.assign(_timestamp_index=timestamp_index)
    if bool(frame.duplicated(["lot", "wafer", "_timestamp_index"]).any()):
        raise ValueError(f"{path.name}: duplicate lot/wafer/timestamp rows")
    grouped = frame.groupby(["lot", "wafer"], sort=False)["_timestamp_index"]
    counts = grouped.size()
    unique_counts = grouped.nunique()
    if not bool(((counts == STEPS) & (unique_counts == STEPS)).all()):
        raise ValueError(f"{path.name}: every key must contain timestamps 0..175")

    numeric, observed, numeric_audit = _strict_numeric_matrix(frame, sensor_columns)
    lot_order = frame["lot"].map(
        lambda value: int(LOT_PATTERN.fullmatch(str(value)).group(1))  # type: ignore[union-attr]
    )
    order = np.lexsort(
        (
            frame["_timestamp_index"].to_numpy(dtype=np.int64),
            frame["wafer"].to_numpy(dtype=np.int64),
            lot_order.to_numpy(dtype=np.int64),
        )
    )
    sorted_frame = frame.iloc[order]
    sorted_numeric = numeric[order]
    sorted_observed = observed[order]
    sorted_keys = _frame_keys(sorted_frame.iloc[::STEPS])
    values = sorted_numeric.reshape(-1, STEPS, len(sensor_columns)).transpose(0, 2, 1)
    masks = sorted_observed.reshape(-1, STEPS, len(sensor_columns)).transpose(0, 2, 1)
    return ParsedEquipment(
        keys=tuple(sorted_keys),
        values=np.ascontiguousarray(values, dtype=np.float32),
        observed_mask=np.ascontiguousarray(masks, dtype=bool),
        audit={
            "schema": "fabrisk_kai_equipment_parse_audit.v1",
            "file": path.name,
            "sha256": sha256_file(path),
            "encoding": SOURCE_ENCODING,
            "delimiter": "semicolon",
            "rows": int(len(frame)),
            "wafer_keys": int(len(sorted_keys)),
            "lots": int(frame["lot"].nunique()),
            "timestamps_per_key": STEPS,
            "sensor_range": [first_sensor, last_sensor],
            "numeric": numeric_audit,
            "identity_columns_used_as_features": False,
            "timestamp_used_only_for_alignment": True,
            "source_row_and_file_order_used_as_features": False,
        },
    )


def _validate_duplicate_numeric_rows(
    frame: pd.DataFrame,
    values: np.ndarray,
    observed: np.ndarray,
    *,
    table_name: str,
) -> int:
    duplicate_keys = 0
    positions: dict[Key, list[int]] = {}
    for position, key in enumerate(_frame_keys(frame)):
        positions.setdefault(key, []).append(position)
    for key, indices in positions.items():
        if len(indices) == 1:
            continue
        duplicate_keys += 1
        reference_values = values[indices[0]]
        reference_mask = observed[indices[0]]
        for position in indices[1:]:
            if not np.array_equal(reference_mask, observed[position]):
                raise ValueError(f"{table_name}: inconsistent duplicate mask for {key}")
            if not np.array_equal(
                reference_values,
                values[position],
                equal_nan=True,
            ):
                raise ValueError(f"{table_name}: inconsistent duplicate values for {key}")
    return duplicate_keys


def parse_summary_file(
    path: Path,
    *,
    first_feature: int,
    last_feature: int,
) -> ParsedSummary:
    columns = tuple(f"KN{index}" for index in range(first_feature, last_feature + 1))
    frame = _read_semicolon(path, ("lot", "wafer", *columns))
    values, observed, numeric_audit = _strict_numeric_matrix(frame, columns)
    duplicate_keys = _validate_duplicate_numeric_rows(
        frame,
        values,
        observed,
        table_name=path.name,
    )
    keep = ~frame.duplicated(["lot", "wafer"], keep="first")
    frame = frame.loc[keep].copy()
    values = values[keep.to_numpy(dtype=bool)]
    observed = observed[keep.to_numpy(dtype=bool)]
    keys = _frame_keys(frame)
    order = np.asarray(
        sorted(range(len(keys)), key=lambda index: _natural_key(keys[index])),
        dtype=np.int64,
    )
    return ParsedSummary(
        keys=tuple(keys[index] for index in order),
        values=np.ascontiguousarray(values[order], dtype=np.float32),
        observed_mask=np.ascontiguousarray(observed[order], dtype=bool),
        audit={
            "schema": "fabrisk_kai_summary_parse_audit.v1",
            "file": path.name,
            "sha256": sha256_file(path),
            "encoding": SOURCE_ENCODING,
            "delimiter": "semicolon",
            "rows_before_deduplication": int(len(keep)),
            "rows_after_deduplication": int(keep.sum()),
            "duplicate_keys": duplicate_keys,
            "duplicate_values_identical": True,
            "feature_range": [first_feature, last_feature],
            "numeric": numeric_audit,
            "identity_columns_used_as_features": False,
            "source_row_and_file_order_used_as_features": False,
        },
    )


def parse_response(path: Path) -> ParsedResponse:
    frame = _read_semicolon(path, ("lot", "wafer", "response", "class"))
    response_values, response_mask, response_audit = _strict_numeric_matrix(
        frame,
        ("response",),
    )
    if not bool(response_mask.all()):
        raise ValueError(f"{path.name}: response contains missing or invalid values")
    classes = frame["class"].astype(str).str.strip().str.lower()
    if not bool(classes.isin(["good", "bad"]).all()):
        raise ValueError(f"{path.name}: class must be good or bad")
    keys = _frame_keys(frame)
    positions: dict[Key, list[int]] = {}
    for position, key in enumerate(keys):
        positions.setdefault(key, []).append(position)
    duplicate_keys = 0
    duplicate_rows = 0
    for key, indices in positions.items():
        if len(indices) == 1:
            continue
        duplicate_keys += 1
        duplicate_rows += len(indices) - 1
        first = indices[0]
        for position in indices[1:]:
            if float(response_values[position, 0]) != float(
                response_values[first, 0]
            ) or str(classes.iloc[position]) != str(classes.iloc[first]):
                raise ValueError(f"{path.name}: inconsistent duplicate response for {key}")
    keep = ~frame.duplicated(["lot", "wafer"], keep="first")
    kept_indices = np.flatnonzero(keep.to_numpy(dtype=bool))
    response_by_key = {
        keys[index]: float(response_values[index, 0]) for index in kept_indices
    }
    class_by_key = {
        keys[index]: str(classes.iloc[index]) for index in kept_indices
    }
    ordered_keys = tuple(sorted(response_by_key, key=_natural_key))
    return ParsedResponse(
        keys=ordered_keys,
        response_by_key=response_by_key,
        class_by_key=class_by_key,
        audit={
            "schema": "fabrisk_kai_response_parse_audit.v1",
            "file": path.name,
            "sha256": sha256_file(path),
            "encoding": SOURCE_ENCODING,
            "delimiter": "semicolon",
            "rows_before_deduplication": int(len(frame)),
            "rows_after_deduplication": int(len(response_by_key)),
            "duplicate_keys": duplicate_keys,
            "duplicate_rows_removed": duplicate_rows,
            "duplicate_response_and_class_identical": True,
            "numeric": response_audit,
        },
    )


def _rows_for_keys(
    source_keys: tuple[Key, ...],
    requested_keys: tuple[Key, ...],
) -> np.ndarray:
    lookup = {key: index for index, key in enumerate(source_keys)}
    return np.asarray([lookup[key] for key in requested_keys], dtype=np.int64)


def load_joined_kai_data(
    sensor_root: Path,
    summary_root: Path,
) -> JoinedKAIData:
    equipment1 = parse_equipment(
        sensor_root / "equipment1.csv",
        first_sensor=1,
        last_sensor=24,
    )
    equipment2 = parse_equipment(
        sensor_root / "equipment2.csv",
        first_sensor=25,
        last_sensor=56,
    )
    sensor_response = parse_response(sensor_root / "response.csv")
    process1 = parse_summary_file(
        summary_root / "process1.csv",
        first_feature=1,
        last_feature=36,
    )
    process2 = parse_summary_file(
        summary_root / "process2.csv",
        first_feature=37,
        last_feature=50,
    )
    summary_response = parse_response(summary_root / "response.csv")

    key_sets = [
        set(equipment1.keys),
        set(equipment2.keys),
        set(sensor_response.keys),
        set(process1.keys),
        set(process2.keys),
        set(summary_response.keys),
    ]
    common_keys = tuple(sorted(set.intersection(*key_sets), key=_natural_key))
    if not common_keys:
        raise ValueError("no common lot/wafer keys across the six source tables")

    class_disagreements = []
    response_differences = []
    lexical_equivalent_rows = 0
    for key in common_keys:
        sensor_class = sensor_response.class_by_key[key]
        summary_class = summary_response.class_by_key[key]
        if sensor_class != summary_class:
            class_disagreements.append(key)
        difference = abs(
            sensor_response.response_by_key[key]
            - summary_response.response_by_key[key]
        )
        response_differences.append(difference)
        if difference > 0.0:
            lexical_equivalent_rows += 1
    max_difference = max(response_differences, default=0.0)
    if class_disagreements:
        raise ValueError(
            f"response class disagreement across sources: {class_disagreements[:5]}"
        )
    if max_difference > 1e-12:
        raise ValueError(
            f"response numeric disagreement across sources exceeds tolerance: "
            f"{max_difference}"
        )

    eq1_rows = _rows_for_keys(equipment1.keys, common_keys)
    eq2_rows = _rows_for_keys(equipment2.keys, common_keys)
    p1_rows = _rows_for_keys(process1.keys, common_keys)
    p2_rows = _rows_for_keys(process2.keys, common_keys)
    temporal_values = np.concatenate(
        (equipment1.values[eq1_rows], equipment2.values[eq2_rows]),
        axis=1,
    )
    temporal_mask = np.concatenate(
        (
            equipment1.observed_mask[eq1_rows],
            equipment2.observed_mask[eq2_rows],
        ),
        axis=1,
    )
    summary_values = np.concatenate(
        (process1.values[p1_rows], process2.values[p2_rows]),
        axis=1,
    )
    summary_mask = np.concatenate(
        (
            process1.observed_mask[p1_rows],
            process2.observed_mask[p2_rows],
        ),
        axis=1,
    )
    labels = np.asarray(
        [1 if sensor_response.class_by_key[key] == "bad" else 0 for key in common_keys],
        dtype=np.uint8,
    )
    responses = np.asarray(
        [sensor_response.response_by_key[key] for key in common_keys],
        dtype=np.float64,
    )
    return JoinedKAIData(
        keys=common_keys,
        temporal_values=np.ascontiguousarray(temporal_values, dtype=np.float32),
        temporal_observed_mask=np.ascontiguousarray(temporal_mask, dtype=bool),
        summary_values=np.ascontiguousarray(summary_values, dtype=np.float32),
        summary_observed_mask=np.ascontiguousarray(summary_mask, dtype=bool),
        labels=labels,
        responses=responses,
        audit={
            "schema": "fabrisk_kai_joined_parse_audit.v1",
            "candidate": "FabRisk-KAI-X5",
            "sources": {
                "equipment1": equipment1.audit,
                "equipment2": equipment2.audit,
                "sensor_response": sensor_response.audit,
                "process1": process1.audit,
                "process2": process2.audit,
                "summary_response": summary_response.audit,
            },
            "common_key_contract": {
                "tables": [
                    "equipment1",
                    "equipment2",
                    "sensor_response",
                    "process1",
                    "process2",
                    "summary_response",
                ],
                "common_rows": len(common_keys),
                "common_lots": len({key[0] for key in common_keys}),
                "temporal_shape": list(temporal_values.shape),
                "summary_shape": list(summary_values.shape),
                "class_counts": {
                    "good": int(np.sum(labels == 0)),
                    "bad": int(np.sum(labels == 1)),
                },
            },
            "cross_release_response_check": {
                "class_disagreements": 0,
                "numeric_tolerance": 1e-12,
                "max_abs_numeric_difference": max_difference,
                "rows_with_nonzero_float_representation_difference": (
                    lexical_equivalent_rows
                ),
                "passed": True,
            },
            "feature_prohibitions": [
                "lot",
                "wafer",
                "source_row_number",
                "source_file_order",
                "response",
                "class",
            ],
            "identity_or_label_columns_used_as_features": False,
            "invalid_values_silently_replaced_with_zero": False,
        },
    )
