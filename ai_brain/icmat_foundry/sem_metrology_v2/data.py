"""NIST sets 1-5 audit and deterministic quality-cell split.

No function in this module reads set 6 payload bytes.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
from PIL import Image

from .contracts import (
    ARCHIVES,
    CLAIM_BOUNDARY,
    EXPECTED_CONTRAST_LEVELS,
    EXPECTED_IMAGES_PER_SET,
    EXPECTED_NOISE_LEVELS,
    INTENSITY_REFERENCE_MEMBER,
    MASK_MEMBER,
    METRICS_MEMBER,
    SOURCE_IMAGE_SIZE,
    SPLIT_SEED,
    TRAIN_SETS,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _archive_record(raw_dir: Path, name: str) -> dict[str, Any]:
    path = raw_dir / name
    contract = ARCHIVES[name]
    record: dict[str, Any] = {
        "filename": name,
        "official_url": contract["official_url"],
        "expected_bytes": contract["bytes"],
        "expected_sha256": contract["sha256"],
        "present": path.is_file(),
        "payload_read": False,
    }
    if not path.is_file():
        record.update(
            {
                "bytes": None,
                "sha256": None,
                "size_verified": False,
                "integrity_verified": False,
            }
        )
        return record
    digest = sha256_file(path)
    record.update(
        {
            "bytes": path.stat().st_size,
            "sha256": digest,
            "size_verified": path.stat().st_size == contract["bytes"],
            "integrity_verified": digest == contract["sha256"],
        }
    )
    return record


def _read_train_member(archive: ZipFile, member: str) -> np.ndarray:
    payload = archive.read(member)
    with Image.open(io.BytesIO(payload)) as image:
        array = np.asarray(image)
    if array.shape != (SOURCE_IMAGE_SIZE, SOURCE_IMAGE_SIZE):
        raise ValueError(f"{member}: expected 512x512, got {array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"{member}: expected uint8, got {array.dtype}")
    return array.copy()


def _load_train_metrics(path: Path) -> dict[int, list[dict[str, str]]]:
    records: dict[int, list[dict[str, str]]] = {}
    with ZipFile(path) as archive:
        for set_id in TRAIN_SETS:
            member = METRICS_MEMBER.format(set_id=set_id)
            rows = list(
                csv.DictReader(
                    io.StringIO(archive.read(member).decode("utf-8-sig"))
                )
            )
            records[set_id] = rows
    return records


def _split_for_cell(noise: int, contrast: int) -> str:
    token = f"{SPLIT_SEED}:{noise:03d}:{contrast:03d}".encode("ascii")
    bucket = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % 10
    if bucket <= 5:
        return "train"
    if bucket <= 7:
        return "calibration"
    return "validation"


def build_split_manifest(metrics_path: Path) -> dict[str, Any]:
    """Build grouped train/calibration/validation records from sets 1-5 only."""
    rows_by_set = _load_train_metrics(metrics_path)
    assignments: list[dict[str, Any]] = []
    cell_to_split: dict[tuple[int, int], str] = {}
    per_set_counts: dict[str, dict[str, int]] = {}
    grid_reports: dict[str, Any] = {}

    for set_id, rows in rows_by_set.items():
        names = [row["IMAGE-NAME"] for row in rows]
        noises = {int(row["Noise_level"]) for row in rows}
        contrasts = {int(row["Contrast_level"]) for row in rows}
        cells = {
            (int(row["Noise_level"]), int(row["Contrast_level"]))
            for row in rows
        }
        grid_ok = (
            len(rows) == EXPECTED_IMAGES_PER_SET
            and len(set(names)) == EXPECTED_IMAGES_PER_SET
            and len(noises) == EXPECTED_NOISE_LEVELS
            and len(contrasts) == EXPECTED_CONTRAST_LEVELS
            and len(cells) == EXPECTED_IMAGES_PER_SET
        )
        grid_reports[str(set_id)] = {
            "rows": len(rows),
            "unique_names": len(set(names)),
            "noise_levels": len(noises),
            "contrast_levels": len(contrasts),
            "quality_cells": len(cells),
            "complete_27x21_grid": grid_ok,
        }
        counts: Counter[str] = Counter()
        for row in sorted(rows, key=lambda item: item["IMAGE-NAME"]):
            noise = int(row["Noise_level"])
            contrast = int(row["Contrast_level"])
            split = _split_for_cell(noise, contrast)
            prior = cell_to_split.setdefault((noise, contrast), split)
            if prior != split:
                raise RuntimeError("quality cell crossed split boundaries")
            counts[split] += 1
            assignments.append(
                {
                    "image_name": row["IMAGE-NAME"],
                    "set_id": set_id,
                    "noise_level": noise,
                    "contrast_level": contrast,
                    "split": split,
                }
            )
        per_set_counts[str(set_id)] = dict(sorted(counts.items()))

    split_counts = Counter(item["split"] for item in assignments)
    manifest = {
        "schema": "icmat_sem_v2_split_manifest.v2",
        "source_sets": list(TRAIN_SETS),
        "sealed_set_excluded": 6,
        "grouping_key": ["Noise_level", "Contrast_level"],
        "grouping_rationale": (
            "The same noise/contrast cell cannot occur in more than one split. "
            "This tests quality-regime generalization while retaining all five "
            "official training geometries in each partition."
        ),
        "partitions": {
            "train": "hash buckets 0-5",
            "calibration": "hash buckets 6-7; threshold and quality calibration only",
            "validation": "hash buckets 8-9; immutable non-test gate only",
        },
        "split_seed": SPLIT_SEED,
        "counts": dict(sorted(split_counts.items())),
        "per_set_counts": per_set_counts,
        "grid_reports": grid_reports,
        "all_grids_complete": all(
            report["complete_27x21_grid"] for report in grid_reports.values()
        ),
        "assignments": assignments,
        "set6_payload_read": False,
        "manifest_sha256": None,
    }
    manifest["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    return manifest


def audit_official_train_data(raw_dir: Path) -> dict[str, Any]:
    """Audit official sets 1-5 and fail closed without touching set 6 payloads."""
    raw_dir = raw_dir.resolve()
    archives = {name: _archive_record(raw_dir, name) for name in ARCHIVES}
    failures: list[str] = []

    for name in ("mask_sets.zip", "metrics_sets.zip"):
        record = archives[name]
        if not record["present"]:
            failures.append(f"MISSING_ARCHIVE:{name}")
        elif not record["integrity_verified"] or not record["size_verified"]:
            failures.append(f"ARCHIVE_INTEGRITY:{name}")

    intensity = archives["intensity_sets.zip"]
    if not intensity["present"]:
        failures.append("MISSING_ARCHIVE:intensity_sets.zip")
    elif not intensity["integrity_verified"] or not intensity["size_verified"]:
        failures.append("ARCHIVE_INTEGRITY:intensity_sets.zip")

    split_manifest: dict[str, Any] | None = None
    if archives["metrics_sets.zip"]["integrity_verified"]:
        split_manifest = build_split_manifest(raw_dir / "metrics_sets.zip")
        archives["metrics_sets.zip"]["payload_read"] = True
        if not split_manifest["all_grids_complete"]:
            failures.append("METRICS_GRID_INCOMPLETE")

    set_reports: dict[str, Any] = {}
    if archives["mask_sets.zip"]["integrity_verified"]:
        with ZipFile(raw_dir / "mask_sets.zip") as archive:
            for set_id in TRAIN_SETS:
                image_member = INTENSITY_REFERENCE_MEMBER.format(set_id=set_id)
                mask_member = MASK_MEMBER.format(set_id=set_id)
                image = _read_train_member(archive, image_member)
                mask = _read_train_member(archive, mask_member)
                unique = np.unique(mask)
                binary = set(unique.tolist()) == {0, 255}
                identical = bool(np.array_equal(image, mask))
                usable = binary and not identical
                set_reports[str(set_id)] = {
                    "set_id": set_id,
                    "mask_unique_values": [int(value) for value in unique.tolist()],
                    "mask_is_binary_0_255": binary,
                    "mask_identical_to_reference_intensity": identical,
                    "usable_as_ground_truth": usable,
                    "reference_only": True,
                }
                if not usable:
                    failures.append(f"INVALID_GROUND_TRUTH:set{set_id}")
        archives["mask_sets.zip"]["payload_read"] = True

    image_member_check: dict[str, Any] = {
        "performed": False,
        "expected_train_images": EXPECTED_IMAGES_PER_SET * len(TRAIN_SETS),
        "matched_train_images": 0,
        "set6_payload_read": False,
    }
    if intensity["integrity_verified"] and split_manifest is not None:
        expected = {
            item["image_name"]
            for item in split_manifest["assignments"]
        }
        with ZipFile(raw_dir / "intensity_sets.zip") as archive:
            names = {
                Path(info.filename).name
                for info in archive.infolist()
                if not info.is_dir()
            }
        matched = expected.intersection(names)
        image_member_check.update(
            {
                "performed": True,
                "matched_train_images": len(matched),
                "all_expected_train_images_present": matched == expected,
                "central_directory_only": True,
            }
        )
        if matched != expected:
            failures.append("INTENSITY_MEMBER_CONTRACT")
        archives["intensity_sets.zip"]["payload_read"] = False

    gate_pass = not failures
    return {
        "schema": "icmat_sem_v2_data_audit.v2",
        "decision": "PASS" if gate_pass else "HOLD_DATA",
        "gate_pass": gate_pass,
        "failures": sorted(set(failures)),
        "archives": archives,
        "train_sets": list(TRAIN_SETS),
        "set_reports": set_reports,
        "split_manifest_available": split_manifest is not None,
        "image_member_check": image_member_check,
        "set6": {
            "candidate_specific_seal_intact": True,
            "image_or_label_payload_read_by_v2_pipeline": False,
            "metrics_payload_read_by_v2_pipeline": False,
            "used_for_selection": False,
            "global_blindness_claim_allowed": False,
            "historical_disclosure": (
                "NIST publishes set6 model metrics and the frozen v1 candidate "
                "already evaluated set6. Source-audit inspection also observed "
                "public metric rows. No v2 candidate image, label, prediction, "
                "or score was opened."
            ),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
