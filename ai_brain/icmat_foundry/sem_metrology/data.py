"""Fail-closed NIST archive validation and deterministic subset loading."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .contracts import (
    ARCHIVES,
    CLAIM_BOUNDARY,
    EXPECTED_METRICS_ROWS_PER_SET,
    INPUT_SIZE,
    INTENSITY_MEMBER,
    LOCKED_TEST_SET,
    MASK_MEMBER,
    METRICS_MEMBER,
    SOURCE_IMAGE_SIZE,
    TRAIN_SETS,
    VALID_SUBSET_TRAIN_SETS,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
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


def _read_tiff(archive: ZipFile, member: str) -> np.ndarray:
    try:
        payload = archive.read(member)
    except KeyError as exc:
        raise ValueError(f"required archive member missing: {member}") from exc
    with Image.open(io.BytesIO(payload)) as image:
        array = np.asarray(image)
    if array.shape != (SOURCE_IMAGE_SIZE, SOURCE_IMAGE_SIZE):
        raise ValueError(f"{member} has shape {array.shape}, expected 512x512")
    if array.dtype != np.uint8:
        raise ValueError(f"{member} has dtype {array.dtype}, expected uint8")
    return array.copy()


def load_official_pair(mask_archive: Path, set_id: int) -> tuple[np.ndarray, np.ndarray]:
    if set_id not in (*TRAIN_SETS, LOCKED_TEST_SET):
        raise ValueError(f"set_id must be 1..6, got {set_id}")
    with ZipFile(mask_archive) as archive:
        image = _read_tiff(archive, INTENSITY_MEMBER.format(set_id=set_id))
        mask = _read_tiff(archive, MASK_MEMBER.format(set_id=set_id))
    return image, mask


def _archive_record(raw_dir: Path, name: str) -> dict[str, Any]:
    contract = ARCHIVES[name]
    path = raw_dir / name
    record: dict[str, Any] = {
        "filename": name,
        "official_url": contract["official_url"],
        "expected_sha256": contract["sha256"],
        "present": path.is_file(),
    }
    if path.is_file():
        record.update(
            {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "integrity_verified": sha256_file(path) == contract["sha256"],
            }
        )
        expected_bytes = contract.get("bytes")
        if expected_bytes is not None:
            record["size_verified"] = path.stat().st_size == expected_bytes
    else:
        record["integrity_verified"] = False
    return record


def validate_metrics_archive(path: Path) -> dict[str, Any]:
    counts: dict[str, int] = {}
    with ZipFile(path) as archive:
        for set_id in range(1, 7):
            member = METRICS_MEMBER.format(set_id=set_id)
            try:
                text = archive.read(member).decode("utf-8-sig")
            except KeyError as exc:
                raise ValueError(f"metrics member missing: {member}") from exc
            rows = list(csv.DictReader(io.StringIO(text)))
            if len(rows) != EXPECTED_METRICS_ROWS_PER_SET:
                raise ValueError(
                    f"{member} has {len(rows)} rows, "
                    f"expected {EXPECTED_METRICS_ROWS_PER_SET}"
                )
            names = {row.get("IMAGE-NAME", "") for row in rows}
            if len(names) != len(rows):
                raise ValueError(f"{member} contains duplicate IMAGE-NAME values")
            if {
                int(row.get("Set_index", "-1"))
                for row in rows
                if row.get("Set_index")
            } != {set_id}:
                raise ValueError(f"{member} has an invalid Set_index contract")
            counts[str(set_id)] = len(rows)
    return {
        "ok": True,
        "rows_per_set": counts,
        "total_rows": sum(counts.values()),
    }


def build_data_contract(raw_dir: Path, *, require_full_corpus: bool) -> dict[str, Any]:
    raw_dir = raw_dir.resolve()
    records = {
        name: _archive_record(raw_dir, name)
        for name in ("intensity_sets.zip", "mask_sets.zip", "metrics_sets.zip")
    }
    for name in ("mask_sets.zip", "metrics_sets.zip"):
        if not records[name]["present"]:
            raise FileNotFoundError(f"required official archive missing: {raw_dir / name}")
        if not records[name]["integrity_verified"]:
            raise ValueError(f"official archive SHA-256 mismatch: {name}")
        if records[name].get("size_verified") is False:
            raise ValueError(f"official archive byte size mismatch: {name}")

    intensity = records["intensity_sets.zip"]
    if intensity["present"] and not intensity["integrity_verified"]:
        raise ValueError("official intensity archive SHA-256 mismatch")
    if require_full_corpus and not intensity["present"]:
        raise RuntimeError(
            "FULL_CORPUS_HOLD: intensity_sets.zip is unavailable; "
            "refusing to produce a full-corpus candidate"
        )

    mask_archive = raw_dir / "mask_sets.zip"
    set_reports: dict[str, dict[str, Any]] = {}
    valid_binary_sets: list[int] = []
    for set_id in range(1, 7):
        image, mask = load_official_pair(mask_archive, set_id)
        unique = np.unique(mask)
        binary = set(unique.tolist()) <= {0, 255} and len(unique) == 2
        identical = bool(np.array_equal(image, mask))
        report = {
            "set_id": set_id,
            "image_shape": list(image.shape),
            "image_dtype": str(image.dtype),
            "mask_unique_values": [int(value) for value in unique.tolist()],
            "mask_is_binary_0_255": binary,
            "mask_identical_to_intensity": identical,
            "foreground_fraction": (
                float(np.mean(mask == 255)) if binary else None
            ),
            "usable_as_ground_truth": binary and not identical,
        }
        set_reports[str(set_id)] = report
        if report["usable_as_ground_truth"]:
            valid_binary_sets.append(set_id)

    expected_usable = [*VALID_SUBSET_TRAIN_SETS, LOCKED_TEST_SET]
    if valid_binary_sets != expected_usable:
        raise ValueError(
            f"unexpected usable-mask sets {valid_binary_sets}; expected {expected_usable}"
        )

    metrics = validate_metrics_archive(raw_dir / "metrics_sets.zip")
    set4_failure = set_reports["4"]
    full_split_satisfied = (
        intensity["present"]
        and intensity["integrity_verified"]
        and all(set_reports[str(set_id)]["usable_as_ground_truth"] for set_id in TRAIN_SETS)
    )
    if require_full_corpus and not full_split_satisfied:
        raise RuntimeError(
            "FULL_CORPUS_HOLD: sets 1-5 label contract is not satisfied"
        )

    return {
        "schema": "icmat_sem_data_contract.v1",
        "ok_for_official_subset_baseline": True,
        "ok_for_full_corpus_candidate": full_split_satisfied,
        "candidate_scope": "OFFICIAL_SUBSET_BASELINE",
        "archives": records,
        "official_split_contract": {
            "train_sets": list(TRAIN_SETS),
            "locked_test_set": LOCKED_TEST_SET,
            "satisfied": full_split_satisfied,
        },
        "effective_subset_split": {
            "train_sets": list(VALID_SUBSET_TRAIN_SETS),
            "excluded_train_sets": [4],
            "locked_test_set": LOCKED_TEST_SET,
            "test_used_for_model_selection": False,
        },
        "set_contracts": set_reports,
        "set4_fail_closed_reason": (
            "Official mask_set4 is pixel-identical to the set4 intensity image "
            f"and has {len(set4_failure['mask_unique_values'])} grayscale values; "
            "it is not accepted as segmentation ground truth."
        ),
        "metrics_archive": metrics,
        "claim_boundary": CLAIM_BOUNDARY,
    }


@dataclass(frozen=True)
class PatchRef:
    set_id: int
    top: int
    left: int
    transform: int


def _grid_positions(length: int, patch: int, stride: int) -> list[int]:
    if patch > length:
        raise ValueError("patch cannot exceed image length")
    positions = list(range(0, length - patch + 1, stride))
    if positions[-1] != length - patch:
        positions.append(length - patch)
    return positions


def _apply_transform(array: np.ndarray, transform: int) -> np.ndarray:
    if transform == 0:
        return array
    if transform == 1:
        return np.flip(array, axis=1)
    if transform == 2:
        return np.flip(array, axis=0)
    if transform == 3:
        return np.rot90(array, 2)
    raise ValueError(f"unsupported transform: {transform}")


class OfficialSubsetPatchDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Geometric patches from valid official baseline pairs in sets 1/2/3/5."""

    def __init__(
        self,
        mask_archive: Path,
        *,
        patch_size: int = INPUT_SIZE,
        stride: int = 64,
        transforms: Iterable[int] = (0, 1, 2, 3),
    ) -> None:
        self.images: dict[int, np.ndarray] = {}
        self.masks: dict[int, np.ndarray] = {}
        self.refs: list[PatchRef] = []
        positions = _grid_positions(SOURCE_IMAGE_SIZE, patch_size, stride)
        for set_id in VALID_SUBSET_TRAIN_SETS:
            image, mask = load_official_pair(mask_archive, set_id)
            if set(np.unique(mask).tolist()) != {0, 255}:
                raise ValueError(f"set {set_id} mask is not binary")
            self.images[set_id] = image
            self.masks[set_id] = mask
            for top in positions:
                for left in positions:
                    for transform in transforms:
                        self.refs.append(PatchRef(set_id, top, left, int(transform)))
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        ref = self.refs[index]
        size = self.patch_size
        image = self.images[ref.set_id][
            ref.top : ref.top + size,
            ref.left : ref.left + size,
        ]
        mask = self.masks[ref.set_id][
            ref.top : ref.top + size,
            ref.left : ref.left + size,
        ]
        image = np.ascontiguousarray(_apply_transform(image, ref.transform))
        mask = np.ascontiguousarray(_apply_transform(mask, ref.transform))
        image_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        return image_tensor, mask_tensor


def tile_origins(
    height: int,
    width: int,
    *,
    tile_size: int = INPUT_SIZE,
    stride: int = 64,
) -> list[tuple[int, int]]:
    tops = _grid_positions(height, tile_size, stride)
    lefts = _grid_positions(width, tile_size, stride)
    return [(top, left) for top in tops for left in lefts]


def validate_finite_probability_map(probability: np.ndarray) -> None:
    if probability.shape != (SOURCE_IMAGE_SIZE, SOURCE_IMAGE_SIZE):
        raise ValueError(f"invalid probability shape: {probability.shape}")
    if not np.all(np.isfinite(probability)):
        raise ValueError("probability map contains non-finite values")
    if float(probability.min()) < 0.0 or float(probability.max()) > 1.0:
        raise ValueError("probability map is outside [0, 1]")
