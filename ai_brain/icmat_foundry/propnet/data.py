"""Fail-closed JARVIS ingestion, fixed features, and leakage-resistant splits."""
from __future__ import annotations

import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import (
    ACTIVE_SPLITS,
    CLAIM_BOUNDARY,
    CODE_TO_SPLIT,
    CRYSTAL_SYSTEMS,
    ELEMENT_TO_Z,
    ELEMENTS,
    FEATURE_NAMES,
    PRIMARY_TARGETS,
    SOURCE_DOI,
    SOURCE_EXPECTED_BYTES,
    SOURCE_EXPECTED_MD5,
    SOURCE_EXPECTED_MEMBER,
    SOURCE_EXPECTED_ROWS,
    SOURCE_EXPECTED_SHA256,
    SOURCE_ID,
    SOURCE_LICENSE,
    SOURCE_LICENSE_URL,
    SOURCE_VERSION,
    SPLIT_NAMES,
    SPLIT_SEED,
    SPLIT_TO_CODE,
    STRUCTURE_ANGLE_BIN_DEG,
    STRUCTURE_LATTICE_BIN,
    STRUCTURE_PAIR_DISTANCE_BIN,
    STRUCTURE_PAIR_QUANTILES,
    TARGET_SPECS,
)


@dataclass
class PreparedDataset:
    features: np.ndarray
    labels: np.ndarray
    label_mask: np.ndarray
    jids: list[str]
    formula_groups: list[str]
    structure_groups: list[str]
    split_codes: np.ndarray
    metadata: dict[str, Any]

    def indices(self, split: str) -> np.ndarray:
        if split not in SPLIT_TO_CODE:
            raise KeyError(f"unknown split: {split}")
        return np.flatnonzero(self.split_codes == SPLIT_TO_CODE[split])


def _file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source_archive(path: Path) -> dict[str, Any]:
    """Verify the version-pinned source before parsing any training records."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"JARVIS source archive not found: {path}")
    size = path.stat().st_size
    md5 = _file_digest(path, "md5")
    sha256 = _file_digest(path, "sha256")
    failures: list[str] = []
    if size != SOURCE_EXPECTED_BYTES:
        failures.append(f"bytes={size}, expected={SOURCE_EXPECTED_BYTES}")
    if md5 != SOURCE_EXPECTED_MD5:
        failures.append(f"md5={md5}, expected={SOURCE_EXPECTED_MD5}")
    if sha256 != SOURCE_EXPECTED_SHA256:
        failures.append(f"sha256={sha256}, expected={SOURCE_EXPECTED_SHA256}")
    if failures:
        raise ValueError("source integrity contract failed: " + "; ".join(failures))

    with zipfile.ZipFile(path) as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        if len(members) != 1 or members[0].filename != SOURCE_EXPECTED_MEMBER:
            found = [member.filename for member in members]
            raise ValueError(
                f"unexpected archive members: {found}; expected [{SOURCE_EXPECTED_MEMBER!r}]"
            )
        member = members[0]

    return {
        "source_id": SOURCE_ID,
        "source_version": SOURCE_VERSION,
        "doi": SOURCE_DOI,
        "license": SOURCE_LICENSE,
        "license_url": SOURCE_LICENSE_URL,
        "archive_path": path.as_posix(),
        "archive_bytes": size,
        "archive_md5": md5,
        "archive_sha256": sha256,
        "member": member.filename,
        "member_uncompressed_bytes": member.file_size,
        "integrity_verified": True,
        "network_used": False,
        "x5_contacted": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def load_source_archive(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    integrity = validate_source_archive(path)
    with zipfile.ZipFile(path) as archive, archive.open(SOURCE_EXPECTED_MEMBER) as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise TypeError("JARVIS JSON must be a list of objects")
    if len(rows) != SOURCE_EXPECTED_ROWS:
        raise ValueError(
            f"source row count changed: {len(rows)}; expected {SOURCE_EXPECTED_ROWS}"
        )
    return rows, integrity


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _gcd(values: Iterable[int]) -> int:
    result = 0
    for value in values:
        result = math.gcd(result, int(value))
    return max(result, 1)


def reduced_formula_contract(elements: Sequence[str]) -> tuple[str, tuple[int, ...], dict[str, int]]:
    if not elements:
        raise ValueError("atoms.elements is empty")
    unknown = sorted({element for element in elements if element not in ELEMENT_TO_Z})
    if unknown:
        raise ValueError(f"unknown element symbols: {unknown}")
    counts = Counter(elements)
    divisor = _gcd(counts.values())
    reduced = {element: count // divisor for element, count in counts.items()}
    formula_key = "|".join(f"{element}{reduced[element]}" for element in sorted(reduced))
    anonymous_stoichiometry = tuple(sorted(reduced.values()))
    return formula_key, anonymous_stoichiometry, reduced


def _validated_structure(
    row: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, float, int, bool]:
    atoms = row.get("atoms")
    if not isinstance(atoms, dict):
        raise ValueError("atoms must be an object")
    elements = atoms.get("elements")
    if not isinstance(elements, list) or not all(isinstance(item, str) for item in elements):
        raise ValueError("atoms.elements must be a string list")
    lattice = np.asarray(atoms.get("lattice_mat"), dtype=np.float64)
    coordinates = np.asarray(atoms.get("coords"), dtype=np.float64)
    angles = np.asarray(atoms.get("angles"), dtype=np.float64)
    coordinates_are_cartesian = atoms.get("cartesian")
    if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)):
        raise ValueError(f"invalid lattice shape or values: {lattice.shape}")
    if coordinates.shape != (len(elements), 3) or not np.all(np.isfinite(coordinates)):
        raise ValueError(
            f"invalid coordinates: {coordinates.shape}, expected {(len(elements), 3)}"
        )
    if not isinstance(coordinates_are_cartesian, bool):
        raise ValueError(
            "atoms.cartesian must be a boolean so coordinate semantics are explicit"
        )
    if angles.shape != (3,) or not np.all(np.isfinite(angles)):
        raise ValueError(f"invalid angle vector: {angles}")
    if np.any(angles <= 0.0) or np.any(angles >= 180.0):
        raise ValueError(f"angles outside (0, 180): {angles.tolist()}")
    volume = abs(float(np.linalg.det(lattice)))
    if not math.isfinite(volume) or volume <= 1e-8:
        raise ValueError(f"degenerate lattice volume: {volume}")
    top_level_nat = int(row.get("nat", len(elements)))
    if top_level_nat <= 0:
        raise ValueError(f"invalid top-level nat: {top_level_nat}")
    spg = int(row.get("spg_number", row.get("spg", 0)))
    if not 1 <= spg <= 230:
        raise ValueError(f"space group outside [1, 230]: {spg}")
    return (
        atoms,
        lattice,
        coordinates,
        angles,
        volume,
        spg,
        coordinates_are_cartesian,
    )


def _quantize(value: float, width: float) -> int:
    return int(round(float(value) / width))


def approximate_structure_family(
    row: dict[str, Any],
    anonymous_stoichiometry: tuple[int, ...],
    reduced: dict[str, int],
) -> str:
    """Return a chemistry-agnostic approximate prototype digest."""
    (
        atoms,
        lattice,
        coordinates,
        angles,
        volume,
        spg,
        coordinates_are_cartesian,
    ) = _validated_structure(row)
    nat = len(atoms["elements"])
    formula_units = nat // sum(reduced.values())

    lengths = np.linalg.norm(lattice, axis=1)
    geometric_length = max(float(np.prod(lengths)), 1e-12) ** (1.0 / 3.0)
    lattice_bins = tuple(
        _quantize(value / geometric_length, STRUCTURE_LATTICE_BIN)
        for value in sorted(lengths.tolist())
    )
    angle_bins = tuple(
        _quantize(value, STRUCTURE_ANGLE_BIN_DEG) for value in sorted(angles.tolist())
    )

    if nat == 1:
        pair_bins: tuple[int | str, ...] = ("single",)
    else:
        fractional = (
            coordinates @ np.linalg.inv(lattice)
            if coordinates_are_cartesian
            else coordinates.copy()
        )
        fractional -= np.floor(fractional)
        left, right = np.triu_indices(nat, 1)
        displacement = fractional[left] - fractional[right]
        displacement -= np.round(displacement)
        distances = np.linalg.norm(displacement @ lattice, axis=1)
        distance_scale = max((volume / nat) ** (1.0 / 3.0), 1e-8)
        quantiles = np.quantile(
            distances / distance_scale,
            STRUCTURE_PAIR_QUANTILES,
        )
        pair_bins = tuple(
            _quantize(value, STRUCTURE_PAIR_DISTANCE_BIN) for value in quantiles
        )

    descriptor = json.dumps(
        {
            "spacegroup": spg,
            "anonymous_stoichiometry": anonymous_stoichiometry,
            "formula_units": formula_units,
            "lattice_bins": lattice_bins,
            "angle_bins": angle_bins,
            "pair_distance_quantile_bins": pair_bins,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return "sf_" + hashlib.sha256(descriptor.encode("utf-8")).hexdigest()


def _crystal_system(spg: int) -> str:
    if spg <= 2:
        return "triclinic"
    if spg <= 15:
        return "monoclinic"
    if spg <= 74:
        return "orthorhombic"
    if spg <= 142:
        return "tetragonal"
    if spg <= 167:
        return "trigonal"
    if spg <= 194:
        return "hexagonal"
    return "cubic"


def fixed_features(
    row: dict[str, Any],
    reduced: dict[str, int],
) -> np.ndarray:
    atoms, lattice, _, angles, volume, spg, _ = _validated_structure(row)
    elements = atoms["elements"]
    counts = Counter(elements)
    nat = len(elements)
    fractions = np.zeros(len(ELEMENTS), dtype=np.float64)
    for element, count in counts.items():
        fractions[ELEMENT_TO_Z[element] - 1] = count / nat

    active_z = np.asarray([ELEMENT_TO_Z[element] for element in counts], dtype=np.float64)
    active_fraction = np.asarray([counts[element] / nat for element in counts], dtype=np.float64)
    z_mean = float(np.sum(active_z * active_fraction))
    z_std = float(np.sqrt(np.sum(((active_z - z_mean) ** 2) * active_fraction)))
    entropy = -float(np.sum(active_fraction * np.log(active_fraction)))
    ranked_fraction = sorted(active_fraction.tolist(), reverse=True)[:8]
    ranked_fraction.extend([0.0] * (8 - len(ranked_fraction)))

    density = _finite_float(row.get("density"))
    if density is None or density < 0.0:
        raise ValueError(f"invalid density: {row.get('density')!r}")
    lengths = np.linalg.norm(lattice, axis=1)
    geometric_length = max(float(np.prod(lengths)), 1e-12) ** (1.0 / 3.0)
    length_ratios = np.clip(np.sort(lengths / geometric_length), 0.0, 10.0)
    sorted_angles = np.sort(angles)
    angle_cosines = np.cos(np.deg2rad(sorted_angles))
    crystal = _crystal_system(spg)
    crystal_onehot = [1.0 if crystal == name else 0.0 for name in CRYSTAL_SYSTEMS]

    summaries = [
        len(counts) / len(ELEMENTS),
        entropy / math.log(len(ELEMENTS)),
        z_mean / len(ELEMENTS),
        z_std / len(ELEMENTS),
        float(np.min(active_z)) / len(ELEMENTS),
        float(np.max(active_z)) / len(ELEMENTS),
        *ranked_fraction,
        math.log1p(nat),
        math.log1p(density),
        math.log1p(volume / nat),
        *length_ratios.tolist(),
        *angle_cosines.tolist(),
        spg / 230.0,
        *crystal_onehot,
    ]
    features = np.concatenate((fractions, np.asarray(summaries, dtype=np.float64)))
    if features.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(features)):
        raise ValueError(f"invalid fixed feature vector: {features.shape}")
    if sum(reduced.values()) <= 0:
        raise ValueError("invalid reduced composition")
    return features.astype(np.float32)


def target_vector(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros(len(TARGET_SPECS), dtype=np.float32)
    mask = np.zeros(len(TARGET_SPECS), dtype=bool)
    for index, spec in enumerate(TARGET_SPECS):
        if spec.name == "electronic_dielectric_mean":
            components = [_finite_float(row.get(field)) for field in spec.source_fields]
            value = (
                float(np.mean(components))
                if all(component is not None for component in components)
                else None
            )
        else:
            value = _finite_float(row.get(spec.source_fields[0]))
        if value is None:
            if spec.required:
                raise ValueError(f"required target missing: {spec.name}")
            continue
        if not spec.minimum <= value <= spec.maximum:
            raise ValueError(
                f"target {spec.name} outside [{spec.minimum}, {spec.maximum}]: {value}"
            )
        values[index] = value
        mask[index] = True
    return values, mask


def _formula_split(formula_group: str) -> str:
    digest = hashlib.sha256(
        f"{SPLIT_SEED}|reduced_formula|{formula_group}".encode()
    ).hexdigest()
    bucket = int(digest[:16], 16) % 100
    if bucket < 75:
        return "train"
    if bucket < 85:
        return "tune"
    if bucket < 90:
        return "calibration"
    return "test"


def assign_group_disjoint_splits(
    formula_groups: Sequence[str],
    structure_groups: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Hash formulas, then quarantine approximate families crossing split borders."""
    if len(formula_groups) != len(structure_groups) or not formula_groups:
        raise ValueError("formula and structure groups must be non-empty and aligned")
    formula_assignments = {group: _formula_split(group) for group in set(formula_groups)}
    family_splits: dict[str, set[str]] = defaultdict(set)
    for formula_group, structure_group in zip(formula_groups, structure_groups, strict=True):
        family_splits[structure_group].add(formula_assignments[formula_group])
    crossing_families = {
        family for family, splits in family_splits.items() if len(splits) > 1
    }

    split_names = [
        (
            "quarantine"
            if structure_group in crossing_families
            else formula_assignments[formula_group]
        )
        for formula_group, structure_group in zip(
            formula_groups, structure_groups, strict=True
        )
    ]
    split_codes = np.asarray([SPLIT_TO_CODE[name] for name in split_names], dtype=np.int8)

    formula_sets = {
        split: {
            formula_groups[index]
            for index in np.flatnonzero(split_codes == SPLIT_TO_CODE[split])
        }
        for split in ACTIVE_SPLITS
    }
    structure_sets = {
        split: {
            structure_groups[index]
            for index in np.flatnonzero(split_codes == SPLIT_TO_CODE[split])
        }
        for split in ACTIVE_SPLITS
    }
    pairs = tuple(
        (left, right)
        for left_index, left in enumerate(ACTIVE_SPLITS)
        for right in ACTIVE_SPLITS[left_index + 1 :]
    )
    formula_overlap = {
        f"{left}_{right}": len(formula_sets[left] & formula_sets[right])
        for left, right in pairs
    }
    structure_overlap = {
        f"{left}_{right}": len(structure_sets[left] & structure_sets[right])
        for left, right in pairs
    }
    if any(formula_overlap.values()) or any(structure_overlap.values()):
        raise ValueError(
            f"group leakage after quarantine: formula={formula_overlap}, "
            f"structure={structure_overlap}"
        )

    return split_codes, {
        "seed": SPLIT_SEED,
        "method": (
            "SHA-256 reduced-formula 75/10/5/10 train/tune/calibration/test "
            "assignment; every approximate structure family observed across "
            "assignment borders is excluded as quarantine"
        ),
        "label_blind": True,
        "quarantine_used_for_training": False,
        "quarantine_used_for_tuning": False,
        "quarantine_used_for_calibration": False,
        "quarantine_used_for_test": False,
        "crossing_structure_family_count": len(crossing_families),
        "formula_overlap": formula_overlap,
        "structure_family_overlap": structure_overlap,
        "group_disjoint": True,
    }


def _distribution(values: np.ndarray) -> dict[str, Any]:
    if not values.size:
        return {"n": 0}
    return {
        "n": int(values.size),
        "min": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "median": float(np.median(values)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def prepare_rows(
    rows: Sequence[dict[str, Any]],
    *,
    source_integrity: dict[str, Any] | None = None,
    enforce_full_contract: bool = False,
) -> PreparedDataset:
    if not rows:
        raise ValueError("no JARVIS records")
    if enforce_full_contract and len(rows) != SOURCE_EXPECTED_ROWS:
        raise ValueError(
            f"full contract expects {SOURCE_EXPECTED_ROWS} rows, got {len(rows)}"
        )

    features = np.empty((len(rows), len(FEATURE_NAMES)), dtype=np.float32)
    labels = np.zeros((len(rows), len(TARGET_SPECS)), dtype=np.float32)
    label_mask = np.zeros((len(rows), len(TARGET_SPECS)), dtype=bool)
    jids: list[str] = []
    formula_groups: list[str] = []
    structure_groups: list[str] = []
    seen_jids: set[str] = set()
    top_level_nat_site_count_mismatches = 0
    cartesian_coordinate_rows = 0
    fractional_coordinate_rows = 0

    for index, row in enumerate(rows):
        jid = str(row.get("jid", "")).strip()
        if not jid or jid in seen_jids:
            raise ValueError(f"missing or duplicate jid at row {index}: {jid!r}")
        seen_jids.add(jid)
        atoms = row.get("atoms")
        if not isinstance(atoms, dict):
            raise ValueError(f"{jid}: missing atoms object")
        if atoms.get("cartesian") is True:
            cartesian_coordinate_rows += 1
        elif atoms.get("cartesian") is False:
            fractional_coordinate_rows += 1
        try:
            elements = atoms.get("elements", [])
            formula_group, anonymous, reduced = reduced_formula_contract(elements)
            top_level_nat = int(row.get("nat", len(elements)))
            if top_level_nat != len(elements):
                top_level_nat_site_count_mismatches += 1
            structure_group = approximate_structure_family(row, anonymous, reduced)
            feature_row = fixed_features(row, reduced)
            target_row, target_mask = target_vector(row)
        except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
            raise ValueError(f"{jid}: data contract failed: {exc}") from exc

        jids.append(jid)
        formula_groups.append(formula_group)
        structure_groups.append(structure_group)
        features[index] = feature_row
        labels[index] = target_row
        label_mask[index] = target_mask

    split_codes, split_contract = assign_group_disjoint_splits(
        formula_groups, structure_groups
    )
    split_counts = {
        split: int(np.sum(split_codes == SPLIT_TO_CODE[split])) for split in SPLIT_NAMES
    }
    if enforce_full_contract:
        for split in ACTIVE_SPLITS:
            minimum_rows = 500 if split == "calibration" else 1_000
            if split_counts[split] < minimum_rows:
                raise ValueError(f"{split} has too few rows: {split_counts[split]}")
        target_names = [spec.name for spec in TARGET_SPECS]
        for name in PRIMARY_TARGETS:
            task_index = target_names.index(name)
            if not bool(np.all(label_mask[:, task_index])):
                raise ValueError(f"required primary target is not complete: {name}")
        for task_index, spec in enumerate(TARGET_SPECS):
            for split in ACTIVE_SPLITS:
                indices = np.flatnonzero(split_codes == SPLIT_TO_CODE[split])
                available = int(np.sum(label_mask[indices, task_index]))
                minimum = (
                    500
                    if spec.required and split == "calibration"
                    else 1_000
                    if spec.required
                    else 50
                    if split == "calibration"
                    else 100
                )
                if available < minimum:
                    raise ValueError(
                        f"{spec.name}/{split} has {available} labels, requires {minimum}"
                    )

    target_coverage: dict[str, Any] = {}
    for task_index, spec in enumerate(TARGET_SPECS):
        target_coverage[spec.name] = {
            "unit": spec.unit,
            "required": spec.required,
            "all_rows": _distribution(labels[label_mask[:, task_index], task_index]),
            "by_split": {},
        }
        for split in SPLIT_NAMES:
            indices = np.flatnonzero(split_codes == SPLIT_TO_CODE[split])
            active = indices[label_mask[indices, task_index]]
            target_coverage[spec.name]["by_split"][split] = (
                {
                    "n": int(active.size),
                    "values_withheld_from_training_manifest": True,
                }
                if split == "test"
                else _distribution(labels[active, task_index])
            )

    split_membership_sha256 = {}
    for split in SPLIT_NAMES:
        members = sorted(
            jids[index]
            for index in np.flatnonzero(split_codes == SPLIT_TO_CODE[split])
        )
        split_membership_sha256[split] = hashlib.sha256(
            ("\n".join(members) + "\n").encode("utf-8")
        ).hexdigest()

    metadata = {
        "schema": "icmat_propnet_data_contract.v1",
        "status": "VALIDATED" if enforce_full_contract else "FIXTURE_VALIDATED",
        "source": source_integrity,
        "rows": len(rows),
        "feature_dim": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "targets": target_coverage,
        "split_counts": split_counts,
        "split_membership_sha256": split_membership_sha256,
        "formula_group_counts": {
            split: len(
                {
                    formula_groups[index]
                    for index in np.flatnonzero(
                        split_codes == SPLIT_TO_CODE[split]
                    )
                }
            )
            for split in SPLIT_NAMES
        },
        "structure_family_counts": {
            split: len(
                {
                    structure_groups[index]
                    for index in np.flatnonzero(
                        split_codes == SPLIT_TO_CODE[split]
                    )
                }
            )
            for split in SPLIT_NAMES
        },
        "split_contract": split_contract,
        "structure_family_contract": {
            "chemistry_agnostic": True,
            "spacegroup": True,
            "anonymous_stoichiometry": True,
            "formula_units": True,
            "normalized_lattice_bin": STRUCTURE_LATTICE_BIN,
            "angle_bin_deg": STRUCTURE_ANGLE_BIN_DEG,
            "normalized_pair_distance_bin": STRUCTURE_PAIR_DISTANCE_BIN,
            "pair_distance_quantiles": list(STRUCTURE_PAIR_QUANTILES),
            "boundary": (
                "This is a conservative approximate prototype fingerprint, not a "
                "crystallographic equivalence proof."
            ),
        },
        "field_authority": {
            "structure_site_count": "len(atoms.elements) == len(atoms.coords)",
            "top_level_nat_role": (
                "source metadata that may describe a reduced/primitive atom count; "
                "it is not used as the structure site count"
            ),
            "top_level_nat_vs_structure_site_count_mismatch_rows": (
                top_level_nat_site_count_mismatches
            ),
            "atoms_elements_vs_coords_mismatch_rows": 0,
            "coordinate_semantics": (
                "atoms.cartesian=true means Cartesian coordinates converted with "
                "coords @ inv(lattice); false means fractional coordinates used directly"
            ),
            "cartesian_coordinate_rows": cartesian_coordinate_rows,
            "fractional_coordinate_rows": fractional_coordinate_rows,
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "production_integration_allowed": False,
        "bpu_compiled": False,
        "x5_tested": False,
    }
    return PreparedDataset(
        features=features,
        labels=labels,
        label_mask=label_mask,
        jids=jids,
        formula_groups=formula_groups,
        structure_groups=structure_groups,
        split_codes=split_codes,
        metadata=metadata,
    )


def load_prepared_dataset(path: Path) -> PreparedDataset:
    rows, integrity = load_source_archive(path)
    return prepare_rows(rows, source_integrity=integrity, enforce_full_contract=True)


def split_name(code: int) -> str:
    try:
        return CODE_TO_SPLIT[int(code)]
    except KeyError as exc:
        raise ValueError(f"invalid split code: {code}") from exc
