"""Build the leakage-resistant ICMat-Qwen-0.5B SFT v2 dataset.

The builder is deliberately offline and deterministic. It uses the complete
version-pinned JARVIS record only for host-side provenance and family grouping.
The model-facing messages never contain a SHA-256 digest and never describe
JARVIS-DFT as experimental or production-line ground truth.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator

BUILDER_VERSION = "icmat-qwen05b-sft-builder-2.0.1"
DATASET_SCHEMA_ID = "icmat_qwen05b_sft.v2"
EXAMPLE_SCHEMA_ID = "icmat_sft_example.v2"
TEST_MEMBERSHIP_SCHEMA_ID = "icmat_sft_test_membership.v2"
SOURCE_ID = "nist_jarvis_dft"
REQUIRED_REUSE_GATE = "ALLOW_TRAIN_REDISTRIBUTE"
SPLIT_NAMES = ("train", "validation", "calibration", "test")
TRAINING_SPLITS = ("train", "validation", "calibration")
HASH64_PATTERN = re.compile(r"(?i)(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
NA_STRINGS = {"", "na", "n/a", "none", "null", "nan", "inf", "-inf"}

SYSTEM_PROMPTS = {
    "zh": (
        "你是 ICMat 结构化材料助手。只能使用 ICMAT_EVIDENCE_JSON 中明确给出的"
        "计算材料证据；证据缺失、冲突或单位错误时必须返回 UNKNOWN。只输出一个 JSON 对象。"
    ),
    "en": (
        "You are the ICMat structured materials assistant. Use only explicit computed-"
        "materials evidence in ICMAT_EVIDENCE_JSON. Return UNKNOWN for missing, "
        "conflicting, or unit-invalid evidence. Return exactly one JSON object."
    ),
}

QUERY_FIELDS: tuple[str, ...] = (
    "formation_energy_peratom",
    "optb88vdw_bandgap",
    "mbj_bandgap",
    "hse_gap",
    "ehull",
    "density",
    "epsx",
    "bulk_modulus_kv",
    "shear_modulus_gv",
    "dfpt_piezo_max_dij",
    "slme",
)

FIELD_UNITS: dict[str, str] = {
    "formation_energy_peratom": "eV/atom",
    "optb88vdw_bandgap": "eV",
    "mbj_bandgap": "eV",
    "hse_gap": "eV",
    "ehull": "eV",
    "density": "g/cm^3",
    "epsx": "dimensionless",
    "bulk_modulus_kv": "GPa",
    "shear_modulus_gv": "GPa",
    "dfpt_piezo_max_dij": "pC/N",
    "slme": "percent",
}

WRONG_UNITS: dict[str, str] = {
    "eV/atom": "GPa",
    "eV": "nm",
    "g/cm^3": "eV",
    "dimensionless": "eV",
    "GPa": "eV",
    "pC/N": "eV",
    "percent": "eV",
}

TASK_NAMES = (
    "property_judgment",
    "tool_parameters",
    "evidence_adjudication",
)

EVIDENCE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["source_id", "source_version", "record_id"],
    "properties": {
        "source_id": {"const": SOURCE_ID},
        "source_version": {"type": "string", "minLength": 1},
        "record_id": {"type": "string", "minLength": 1},
    },
}

PROPERTY_TARGET_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema",
        "status",
        "requested_field",
        "relation",
        "value",
        "unit",
        "threshold",
        "reason",
        "evidence",
    ],
    "properties": {
        "schema": {"const": "icmat.property_judgment.v2"},
        "status": {"enum": ["SUPPORTED", "UNKNOWN"]},
        "requested_field": {"enum": list(QUERY_FIELDS)},
        "relation": {
            "type": ["string", "null"],
            "enum": ["ABOVE_OR_EQUAL", "BELOW", None],
        },
        "value": {"type": ["number", "null"]},
        "unit": {"enum": sorted(set(FIELD_UNITS.values()))},
        "threshold": {"type": "number"},
        "reason": {
            "type": ["string", "null"],
            "enum": ["FIELD_NOT_IN_PROVIDED_EVIDENCE_VIEW", None],
        },
        "evidence": EVIDENCE_SCHEMA,
    },
    "allOf": [
        {
            "if": {"properties": {"status": {"const": "SUPPORTED"}}},
            "then": {
                "properties": {
                    "relation": {"enum": ["ABOVE_OR_EQUAL", "BELOW"]},
                    "value": {"type": "number"},
                    "reason": {"type": "null"},
                }
            },
            "else": {
                "properties": {
                    "relation": {"type": "null"},
                    "value": {"type": "null"},
                    "reason": {"const": "FIELD_NOT_IN_PROVIDED_EVIDENCE_VIEW"},
                }
            },
        }
    ],
}

TOOL_TARGET_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema",
        "status",
        "tool",
        "arguments",
        "reason",
        "evidence",
    ],
    "properties": {
        "schema": {"const": "icmat.tool_parameters.v2"},
        "status": {"enum": ["SUPPORTED", "UNKNOWN"]},
        "tool": {"type": ["string", "null"]},
        "arguments": {
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "source_id",
                        "source_version",
                        "record_id",
                        "field",
                    ],
                    "properties": {
                        "source_id": {"const": SOURCE_ID},
                        "source_version": {"type": "string", "minLength": 1},
                        "record_id": {"type": "string", "minLength": 1},
                        "field": {"enum": list(QUERY_FIELDS)},
                    },
                },
                {"type": "null"},
            ]
        },
        "reason": {
            "type": ["string", "null"],
            "enum": ["FIELD_NOT_IN_PROVIDED_EVIDENCE_VIEW", None],
        },
        "evidence": EVIDENCE_SCHEMA,
    },
    "allOf": [
        {
            "if": {"properties": {"status": {"const": "SUPPORTED"}}},
            "then": {
                "properties": {
                    "tool": {"const": "lookup_pinned_jarvis_property"},
                    "arguments": {"type": "object"},
                    "reason": {"type": "null"},
                }
            },
            "else": {
                "properties": {
                    "tool": {"type": "null"},
                    "arguments": {"type": "null"},
                    "reason": {"const": "FIELD_NOT_IN_PROVIDED_EVIDENCE_VIEW"},
                }
            },
        }
    ],
}

ADJUDICATION_TARGET_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema",
        "status",
        "requested_field",
        "value",
        "unit",
        "reason",
        "evidence",
    ],
    "properties": {
        "schema": {"const": "icmat.evidence_adjudication.v2"},
        "status": {"enum": ["SUPPORTED", "UNKNOWN"]},
        "requested_field": {"enum": list(QUERY_FIELDS)},
        "value": {"type": ["number", "null"]},
        "unit": {"enum": sorted(set(FIELD_UNITS.values()))},
        "reason": {
            "enum": ["CONSISTENT_EVIDENCE", "EVIDENCE_CONFLICT", "UNIT_MISMATCH"]
        },
        "evidence": EVIDENCE_SCHEMA,
    },
    "allOf": [
        {
            "if": {"properties": {"status": {"const": "SUPPORTED"}}},
            "then": {
                "properties": {
                    "value": {"type": "number"},
                    "reason": {"const": "CONSISTENT_EVIDENCE"},
                }
            },
            "else": {
                "properties": {
                    "value": {"type": "null"},
                    "reason": {"enum": ["EVIDENCE_CONFLICT", "UNIT_MISMATCH"]},
                }
            },
        }
    ],
}

ASSISTANT_SCHEMAS = {
    "property_judgment": PROPERTY_TARGET_SCHEMA,
    "tool_parameters": TOOL_TARGET_SCHEMA,
    "evidence_adjudication": ADJUDICATION_TARGET_SCHEMA,
}

EXAMPLE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema",
        "example_id",
        "task",
        "status_label",
        "requested_field",
        "family_id",
        "split",
        "language",
        "domain_tags",
        "augmentation",
        "host_binding",
        "messages",
    ],
    "properties": {
        "schema": {"const": EXAMPLE_SCHEMA_ID},
        "example_id": {"type": "string", "pattern": "^sftv2_[0-9a-f]{24}$"},
        "task": {"enum": list(TASK_NAMES)},
        "status_label": {"enum": ["SUPPORTED", "UNKNOWN"]},
        "requested_field": {"enum": list(QUERY_FIELDS)},
        "family_id": {"type": "string", "pattern": "^family_[0-9a-f]{24}$"},
        "split": {"enum": list(TRAINING_SPLITS)},
        "language": {"enum": ["zh", "en"]},
        "domain_tags": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "augmentation": {
            "enum": [
                "none",
                "masked_field",
                "synthetic_conflict",
                "synthetic_unit_error",
            ]
        },
        "host_binding": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "source_id",
                "source_version",
                "record_id",
                "canonical_complete_record_sha256",
            ],
            "properties": {
                "source_id": {"const": SOURCE_ID},
                "source_version": {"type": "string", "minLength": 1},
                "record_id": {"type": "string", "minLength": 1},
                "canonical_complete_record_sha256": {
                    "type": "string",
                    "pattern": "^[0-9a-f]{64}$",
                },
            },
        },
        "messages": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "prefixItems": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["role", "content"],
                    "properties": {
                        "role": {"const": "system"},
                        "content": {"type": "string", "minLength": 1},
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["role", "content"],
                    "properties": {
                        "role": {"const": "user"},
                        "content": {"type": "string", "minLength": 1},
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["role", "content"],
                    "properties": {
                        "role": {"const": "assistant"},
                        "content": {"type": "string", "minLength": 2},
                    },
                },
            ],
            "items": False,
        },
    },
}

for _schema in (*ASSISTANT_SCHEMAS.values(), EXAMPLE_SCHEMA):
    Draft202012Validator.check_schema(_schema)


@dataclass(frozen=True)
class SourceLock:
    archive_path: Path
    archive_sha256: str
    archive_bytes: int
    member_name: str
    source_version: str
    doi: str
    license_name: str
    license_url: str
    reuse_gate: str
    acquired_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": SOURCE_ID,
            "archive_name": self.archive_path.name,
            "archive_sha256": self.archive_sha256,
            "archive_bytes": self.archive_bytes,
            "member_name": self.member_name,
            "source_version": self.source_version,
            "doi": self.doi,
            "license_name": self.license_name,
            "license_url": self.license_url,
            "reuse_gate": self.reuse_gate,
            "acquired_at": self.acquired_at,
        }


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        parent = self.parent[item]
        while parent != self.parent[parent]:
            self.parent[parent] = self.parent[self.parent[parent]]
            parent = self.parent[parent]
        self.parent[item] = parent
        return parent

    def union(self, left: int, right: int) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonicalize_complete_value(value: Any) -> Any:
    """Preserve a complete parsed record while making non-finite values explicit."""
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return {"__nonfinite_float__": "NaN"}
        if value > 0:
            return {"__nonfinite_float__": "+Infinity"}
        return {"__nonfinite_float__": "-Infinity"}
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize_complete_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_canonicalize_complete_value(item) for item in value]
    return value


def canonical_complete_record_hash(record: Mapping[str, Any]) -> str:
    """Bind the complete parsed JARVIS record, not a selected field subset."""
    normalized = _canonicalize_complete_value(record)
    return sha256_bytes(canonical_json(normalized).encode("utf-8"))


def _safe_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or stripped.lower() in NA_STRINGS:
        return None
    return stripped


def _safe_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in NA_STRINGS:
            return None
        try:
            result = float(stripped)
        except ValueError:
            return None
    else:
        return None
    return result if math.isfinite(result) else None


def _gcd_counts(counts: Iterable[int]) -> int:
    divisor = 0
    for count in counts:
        divisor = math.gcd(divisor, count)
    return max(divisor, 1)


def reduced_formula_key(record: Mapping[str, Any]) -> str:
    """Return an element-aware reduced composition key."""
    atoms = record.get("atoms")
    elements = atoms.get("elements") if isinstance(atoms, Mapping) else None
    if isinstance(elements, list) and elements and all(isinstance(x, str) for x in elements):
        counts = Counter(elements)
        divisor = _gcd_counts(counts.values())
        return "|".join(
            f"{element}:{counts[element] // divisor}" for element in sorted(counts)
        )
    formula = _safe_text(record.get("formula"))
    return f"literal:{formula}" if formula else "missing"


def _anonymous_stoichiometry(record: Mapping[str, Any]) -> tuple[int, ...]:
    atoms = record.get("atoms")
    elements = atoms.get("elements") if isinstance(atoms, Mapping) else None
    if not isinstance(elements, list) or not elements:
        return ()
    counts = Counter(str(element) for element in elements)
    divisor = _gcd_counts(counts.values())
    return tuple(sorted(count // divisor for count in counts.values()))


def fractional_coordinates(
    record: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return lattice and fractional coordinates with JARVIS semantics respected."""
    atoms = record.get("atoms")
    if not isinstance(atoms, Mapping):
        return None
    try:
        lattice = np.asarray(atoms.get("lattice_mat"), dtype=np.float64)
        coordinates = np.asarray(atoms.get("coords"), dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if lattice.shape != (3, 3) or coordinates.ndim != 2 or coordinates.shape[1] != 3:
        return None
    if not np.isfinite(lattice).all() or not np.isfinite(coordinates).all():
        return None
    if abs(float(np.linalg.det(lattice))) < 1e-10:
        return None
    cartesian = atoms.get("cartesian")
    if cartesian is True:
        fractional = np.linalg.solve(lattice.T, coordinates.T).T
    elif cartesian is False:
        fractional = coordinates.copy()
    else:
        return None
    fractional = np.mod(fractional, 1.0)
    fractional[np.isclose(fractional, 1.0, atol=1e-10)] = 0.0
    return lattice, fractional


def approximate_structure_fingerprint(record: Mapping[str, Any]) -> str | None:
    """Build a translation/order-invariant approximate structural bucket.

    The fingerprint is intentionally conservative: it combines atom count,
    anonymous stoichiometry, space group, normalized cell shape, and a
    quantized nearest-neighbour distance histogram. Cartesian coordinates are
    converted to fractional coordinates only when ``cartesian`` is true.
    """
    converted = fractional_coordinates(record)
    if converted is None:
        return None
    lattice, fractional = converted
    atom_count = int(fractional.shape[0])
    if atom_count == 0:
        return None

    volume = abs(float(np.linalg.det(lattice)))
    scale = volume ** (1.0 / 3.0)
    lengths = np.linalg.norm(lattice, axis=1)
    if scale <= 0 or np.any(lengths <= 0):
        return None
    normalized_lengths = sorted(int(round(float(item / scale) / 0.025)) for item in lengths)

    angles: list[float] = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        cosine = float(
            np.dot(lattice[left], lattice[right]) / (lengths[left] * lengths[right])
        )
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        angles.append(angle)
    angle_bins = sorted(int(round(angle / 2.0)) for angle in angles)

    order = np.lexsort((fractional[:, 2], fractional[:, 1], fractional[:, 0]))
    sample = fractional[order[:64]]
    if len(sample) == 1:
        neighbour_bins: tuple[int, ...] = ()
    else:
        delta = sample[:, None, :] - sample[None, :, :]
        delta -= np.rint(delta)
        cartesian_delta = delta @ lattice
        distances = np.linalg.norm(cartesian_delta, axis=2) / scale
        np.fill_diagonal(distances, np.inf)
        nearest_count = min(4, len(sample) - 1)
        nearest = np.partition(distances, nearest_count - 1, axis=1)[:, :nearest_count]
        neighbour_bins = tuple(
            sorted(int(round(float(value) / 0.04)) for value in nearest.ravel())
        )

    signature = {
        "nat": atom_count,
        "anonymous_stoichiometry": _anonymous_stoichiometry(record),
        "space_group": str(record.get("spg_number", "unknown")),
        "normalized_length_bins": normalized_lengths,
        "angle_bins": angle_bins,
        "nearest_neighbour_bins": neighbour_bins,
    }
    return "structure_" + sha256_bytes(canonical_json(signature).encode("utf-8"))[:24]


def _normalized_reference(record: Mapping[str, Any]) -> str | None:
    reference = _safe_text(record.get("reference"))
    return reference.casefold() if reference else None


def _record_id(record: Mapping[str, Any]) -> str | None:
    return _safe_text(record.get("jid"))


def build_family_map(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Build connected-component families over all identifiable source records."""
    indexed: list[tuple[str, Mapping[str, Any]]] = []
    seen_ids: set[str] = set()
    for record in records:
        record_id = _record_id(record)
        if record_id is None or _safe_text(record.get("formula")) is None:
            continue
        if record_id in seen_ids:
            raise ValueError(f"duplicate JARVIS record id: {record_id}")
        seen_ids.add(record_id)
        indexed.append((record_id, record))

    union_find = UnionFind(len(indexed))
    first_by_key: dict[tuple[str, str], int] = {}
    edge_candidates = Counter()
    successful_unions = Counter()
    coordinate_semantics = Counter()

    for index, (_, record) in enumerate(indexed):
        atoms = record.get("atoms")
        cartesian = atoms.get("cartesian") if isinstance(atoms, Mapping) else None
        if cartesian is True:
            coordinate_semantics["cartesian_true"] += 1
        elif cartesian is False:
            coordinate_semantics["cartesian_false"] += 1
        else:
            coordinate_semantics["cartesian_invalid_or_missing"] += 1

        keys: list[tuple[str, str]] = [
            ("reduced_formula", reduced_formula_key(record)),
        ]
        reference = _normalized_reference(record)
        if reference is not None:
            keys.append(("reference", reference))
        structure = approximate_structure_fingerprint(record)
        if structure is not None:
            keys.append(("approximate_structure", structure))

        for kind, value in keys:
            key = (kind, value)
            previous = first_by_key.get(key)
            if previous is None:
                first_by_key[key] = index
                continue
            edge_candidates[kind] += 1
            if union_find.union(index, previous):
                successful_unions[kind] += 1

    members_by_root: dict[int, list[str]] = defaultdict(list)
    for index, (record_id, _) in enumerate(indexed):
        members_by_root[union_find.find(index)].append(record_id)

    family_by_root: dict[int, str] = {}
    component_sizes = Counter()
    for root, members in members_by_root.items():
        sorted_members = sorted(members)
        family_hash = sha256_bytes(
            canonical_json(
                {
                    "source_id": SOURCE_ID,
                    "record_ids": sorted_members,
                }
            ).encode("utf-8")
        )
        family_by_root[root] = f"family_{family_hash[:24]}"
        component_sizes[len(sorted_members)] += 1

    family_map = {
        record_id: family_by_root[union_find.find(index)]
        for index, (record_id, _) in enumerate(indexed)
    }
    audit = {
        "identifiable_records": len(indexed),
        "family_count": len(members_by_root),
        "edge_candidates": dict(sorted(edge_candidates.items())),
        "successful_unions": dict(sorted(successful_unions.items())),
        "coordinate_semantics": dict(sorted(coordinate_semantics.items())),
        "component_size_histogram": {
            str(size): count for size, count in sorted(component_sizes.items())
        },
        "max_component_size": max((len(value) for value in members_by_root.values()), default=0),
    }
    return family_map, audit


def split_for_family(family_id: str) -> str:
    bucket = int(hashlib.sha256(family_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 76:
        return "train"
    if bucket < 84:
        return "validation"
    if bucket < 92:
        return "calibration"
    return "test"


def domain_tags(record: Mapping[str, Any]) -> list[str]:
    """Assign explicit descriptor-availability tags, not device-suitability labels."""
    tags = ["electronic_materials_computed_dft"]
    if any(_safe_number(record.get(field)) is not None for field in (
        "optb88vdw_bandgap",
        "mbj_bandgap",
        "hse_gap",
    )):
        tags.append("semiconductor_band_structure_screening")
    if any(_safe_number(record.get(field)) is not None for field in ("epsx", "epsy", "epsz")):
        tags.append("dielectric_materials_screening")
    if _safe_number(record.get("slme")) is not None:
        tags.append("optoelectronic_absorber_screening")
    if any(
        _safe_number(record.get(field)) is not None
        for field in ("bulk_modulus_kv", "shear_modulus_gv")
    ):
        tags.append("advanced_packaging_mechanical_screening")
    if _safe_number(record.get("dfpt_piezo_max_dij")) is not None:
        tags.append("functional_electronic_materials_screening")
    return sorted(tags)


def _available_query_fields(record: Mapping[str, Any]) -> list[str]:
    return [field for field in QUERY_FIELDS if _safe_number(record.get(field)) is not None]


def _deterministic_order(items: Iterable[Any], seed: str) -> list[Any]:
    return sorted(
        items,
        key=lambda item: sha256_bytes(
            f"{seed}|{canonical_json(item)}".encode()
        ),
    )


def _round_number(value: float) -> float:
    return float(f"{value:.10g}")


def _threshold_for(value: float, record_id: str, field: str) -> tuple[float, str]:
    delta = max(abs(value) * 0.05, 0.05)
    above = int(sha256_bytes(f"{record_id}|{field}|threshold".encode())[:2], 16) % 2 == 0
    threshold = value - delta if above else value + delta
    relation = "ABOVE_OR_EQUAL" if above else "BELOW"
    return _round_number(threshold), relation


def _counterexample_value(value: float) -> float:
    delta = max(abs(value) * 0.15, 0.1)
    return _round_number(value + delta)


def _assistant_evidence(source_lock: SourceLock, record_id: str) -> dict[str, str]:
    return {
        "source_id": SOURCE_ID,
        "source_version": source_lock.source_version,
        "record_id": record_id,
    }


def _host_binding(
    source_lock: SourceLock,
    record: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "source_id": SOURCE_ID,
        "source_version": source_lock.source_version,
        "record_id": str(record["jid"]),
        "canonical_complete_record_sha256": canonical_complete_record_hash(record),
    }


def _model_context(
    *,
    source_lock: SourceLock,
    record: Mapping[str, Any],
    field: str,
    facts: Sequence[Mapping[str, Any]],
    tags: Sequence[str],
    evidence_view: str,
) -> dict[str, Any]:
    return {
        "source_id": SOURCE_ID,
        "source_version": source_lock.source_version,
        "record_id": str(record["jid"]),
        "source_kind": "public_computed_dft_record",
        "claim_boundary": "not_experimental_not_fab_line_ground_truth",
        "domain_tags": list(tags),
        "requested_field": field,
        "expected_unit": FIELD_UNITS[field],
        "evidence_view": evidence_view,
        "facts": list(facts),
    }


def _user_message(language: str, task: str, context: Mapping[str, Any]) -> str:
    instructions = {
        ("zh", "property_judgment"): (
            "判断给定计算属性值相对阈值的关系。字段缺失时不得猜测。"
        ),
        ("en", "property_judgment"): (
            "Judge the supplied computed property against the threshold. "
            "Do not guess when the field is absent."
        ),
        ("zh", "tool_parameters"): (
            "仅当请求字段出现在 available_fields 时生成只读查询参数，否则拒绝调用。"
        ),
        ("en", "tool_parameters"): (
            "Generate read-only lookup parameters only when the requested field is "
            "listed in available_fields; otherwise refuse the call."
        ),
        ("zh", "evidence_adjudication"): (
            "核对同一计算属性的证据。数值冲突或单位不合规时返回 UNKNOWN。"
        ),
        ("en", "evidence_adjudication"): (
            "Adjudicate evidence for one computed property. Return UNKNOWN for "
            "conflicting values or an invalid unit."
        ),
    }
    return (
        instructions[(language, task)]
        + "\nICMAT_EVIDENCE_JSON="
        + canonical_json(context)
    )


def _example_id(
    *,
    task: str,
    status: str,
    record_id: str,
    field: str,
    family_id: str,
) -> str:
    identity = {
        "schema": EXAMPLE_SCHEMA_ID,
        "task": task,
        "status": status,
        "record_id": record_id,
        "field": field,
        "family_id": family_id,
    }
    return "sftv2_" + sha256_bytes(canonical_json(identity).encode("utf-8"))[:24]


def _language_for_pair(record_id: str, task: str, status: str) -> str:
    supported_is_zh = (
        int(sha256_bytes(f"{record_id}|{task}|language".encode())[:2], 16) % 2 == 0
    )
    if status == "SUPPORTED":
        return "zh" if supported_is_zh else "en"
    return "en" if supported_is_zh else "zh"


def _make_example(
    *,
    task: str,
    status: str,
    record: Mapping[str, Any],
    field: str,
    family_id: str,
    split: str,
    source_lock: SourceLock,
    tags: Sequence[str],
    augmentation: str,
    context: Mapping[str, Any],
    assistant: Mapping[str, Any],
) -> dict[str, Any]:
    record_id = str(record["jid"])
    language = _language_for_pair(record_id, task, status)
    example = {
        "schema": EXAMPLE_SCHEMA_ID,
        "example_id": _example_id(
            task=task,
            status=status,
            record_id=record_id,
            field=field,
            family_id=family_id,
        ),
        "task": task,
        "status_label": status,
        "requested_field": field,
        "family_id": family_id,
        "split": split,
        "language": language,
        "domain_tags": list(tags),
        "augmentation": augmentation,
        "host_binding": _host_binding(source_lock, record),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS[language]},
            {"role": "user", "content": _user_message(language, task, context)},
            {"role": "assistant", "content": canonical_json(assistant)},
        ],
    }
    validate_example(example)
    return example


def _base_facts(record: Mapping[str, Any], seed: str) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for field in ("formula", "spg_number", "crys", "dimensionality"):
        value = record.get(field)
        if isinstance(value, (str, int, float)) and _safe_text(str(value)) is not None:
            facts.append({"field": field, "value": value})
    return _deterministic_order(facts[:3], seed)


def _property_pair(
    *,
    record: Mapping[str, Any],
    field: str,
    family_id: str,
    split: str,
    source_lock: SourceLock,
    tags: Sequence[str],
) -> list[dict[str, Any]]:
    record_id = str(record["jid"])
    value = _round_number(float(_safe_number(record[field])))
    threshold, relation = _threshold_for(value, record_id, field)
    unit = FIELD_UNITS[field]
    evidence = _assistant_evidence(source_lock, record_id)

    base = _base_facts(record, f"{record_id}|property")
    supported_facts = _deterministic_order(
        [
            *base,
            {"field": field, "value": value, "unit": unit},
            {"field": "decision_threshold", "value": threshold, "unit": unit},
        ],
        f"{record_id}|property|supported",
    )
    unknown_facts = _deterministic_order(
        [
            *base,
            {"field": "decision_threshold", "value": threshold, "unit": unit},
        ],
        f"{record_id}|property|unknown",
    )
    supported_context = _model_context(
        source_lock=source_lock,
        record=record,
        field=field,
        facts=supported_facts,
        tags=tags,
        evidence_view="complete_for_requested_field",
    )
    unknown_context = _model_context(
        source_lock=source_lock,
        record=record,
        field=field,
        facts=unknown_facts,
        tags=tags,
        evidence_view="requested_field_masked_counterexample",
    )
    supported_target = {
        "schema": "icmat.property_judgment.v2",
        "status": "SUPPORTED",
        "requested_field": field,
        "relation": relation,
        "value": value,
        "unit": unit,
        "threshold": threshold,
        "reason": None,
        "evidence": evidence,
    }
    unknown_target = {
        "schema": "icmat.property_judgment.v2",
        "status": "UNKNOWN",
        "requested_field": field,
        "relation": None,
        "value": None,
        "unit": unit,
        "threshold": threshold,
        "reason": "FIELD_NOT_IN_PROVIDED_EVIDENCE_VIEW",
        "evidence": evidence,
    }
    return [
        _make_example(
            task="property_judgment",
            status="SUPPORTED",
            record=record,
            field=field,
            family_id=family_id,
            split=split,
            source_lock=source_lock,
            tags=tags,
            augmentation="none",
            context=supported_context,
            assistant=supported_target,
        ),
        _make_example(
            task="property_judgment",
            status="UNKNOWN",
            record=record,
            field=field,
            family_id=family_id,
            split=split,
            source_lock=source_lock,
            tags=tags,
            augmentation="masked_field",
            context=unknown_context,
            assistant=unknown_target,
        ),
    ]


def _tool_pair(
    *,
    record: Mapping[str, Any],
    field: str,
    family_id: str,
    split: str,
    source_lock: SourceLock,
    tags: Sequence[str],
) -> list[dict[str, Any]]:
    record_id = str(record["jid"])
    evidence = _assistant_evidence(source_lock, record_id)
    available = _deterministic_order(
        [
            {"field": candidate, "unit": FIELD_UNITS[candidate]}
            for candidate in _available_query_fields(record)[:6]
        ],
        f"{record_id}|tool|supported",
    )
    masked = [item for item in available if item["field"] != field]
    if not any(item["field"] == field for item in available):
        available.append({"field": field, "unit": FIELD_UNITS[field]})
        available = _deterministic_order(available, f"{record_id}|tool|supported|forced")
    supported_context = _model_context(
        source_lock=source_lock,
        record=record,
        field=field,
        facts=[{"available_fields": available}],
        tags=tags,
        evidence_view="requested_field_listed",
    )
    unknown_context = _model_context(
        source_lock=source_lock,
        record=record,
        field=field,
        facts=[{"available_fields": masked}],
        tags=tags,
        evidence_view="requested_field_not_listed_counterexample",
    )
    supported_target = {
        "schema": "icmat.tool_parameters.v2",
        "status": "SUPPORTED",
        "tool": "lookup_pinned_jarvis_property",
        "arguments": {
            "source_id": SOURCE_ID,
            "source_version": source_lock.source_version,
            "record_id": record_id,
            "field": field,
        },
        "reason": None,
        "evidence": evidence,
    }
    unknown_target = {
        "schema": "icmat.tool_parameters.v2",
        "status": "UNKNOWN",
        "tool": None,
        "arguments": None,
        "reason": "FIELD_NOT_IN_PROVIDED_EVIDENCE_VIEW",
        "evidence": evidence,
    }
    return [
        _make_example(
            task="tool_parameters",
            status="SUPPORTED",
            record=record,
            field=field,
            family_id=family_id,
            split=split,
            source_lock=source_lock,
            tags=tags,
            augmentation="none",
            context=supported_context,
            assistant=supported_target,
        ),
        _make_example(
            task="tool_parameters",
            status="UNKNOWN",
            record=record,
            field=field,
            family_id=family_id,
            split=split,
            source_lock=source_lock,
            tags=tags,
            augmentation="masked_field",
            context=unknown_context,
            assistant=unknown_target,
        ),
    ]


def _adjudication_pair(
    *,
    record: Mapping[str, Any],
    field: str,
    family_id: str,
    split: str,
    source_lock: SourceLock,
    tags: Sequence[str],
) -> list[dict[str, Any]]:
    record_id = str(record["jid"])
    value = _round_number(float(_safe_number(record[field])))
    unit = FIELD_UNITS[field]
    evidence = _assistant_evidence(source_lock, record_id)
    supported_facts = _deterministic_order(
        [
            {"channel": "record_property", "field": field, "value": value, "unit": unit},
            {"channel": "versioned_view", "field": field, "value": value, "unit": unit},
        ],
        f"{record_id}|adjudication|supported",
    )
    use_unit_error = (
        int(sha256_bytes(f"{record_id}|{field}|negative".encode())[:2], 16) % 2 == 0
    )
    if use_unit_error:
        negative_facts = [
            {
                "channel": "unit_corrupted_counterexample",
                "field": field,
                "value": value,
                "unit": WRONG_UNITS[unit],
            }
        ]
        negative_reason = "UNIT_MISMATCH"
        augmentation = "synthetic_unit_error"
        evidence_view = "synthetic_wrong_unit_counterexample"
    else:
        negative_facts = _deterministic_order(
            [
                {"channel": "view_a", "field": field, "value": value, "unit": unit},
                {
                    "channel": "synthetic_conflicting_view_b",
                    "field": field,
                    "value": _counterexample_value(value),
                    "unit": unit,
                },
            ],
            f"{record_id}|adjudication|conflict",
        )
        negative_reason = "EVIDENCE_CONFLICT"
        augmentation = "synthetic_conflict"
        evidence_view = "synthetic_value_conflict_counterexample"

    supported_context = _model_context(
        source_lock=source_lock,
        record=record,
        field=field,
        facts=supported_facts,
        tags=tags,
        evidence_view="two_consistent_computed_views",
    )
    unknown_context = _model_context(
        source_lock=source_lock,
        record=record,
        field=field,
        facts=negative_facts,
        tags=tags,
        evidence_view=evidence_view,
    )
    supported_target = {
        "schema": "icmat.evidence_adjudication.v2",
        "status": "SUPPORTED",
        "requested_field": field,
        "value": value,
        "unit": unit,
        "reason": "CONSISTENT_EVIDENCE",
        "evidence": evidence,
    }
    unknown_target = {
        "schema": "icmat.evidence_adjudication.v2",
        "status": "UNKNOWN",
        "requested_field": field,
        "value": None,
        "unit": unit,
        "reason": negative_reason,
        "evidence": evidence,
    }
    return [
        _make_example(
            task="evidence_adjudication",
            status="SUPPORTED",
            record=record,
            field=field,
            family_id=family_id,
            split=split,
            source_lock=source_lock,
            tags=tags,
            augmentation="none",
            context=supported_context,
            assistant=supported_target,
        ),
        _make_example(
            task="evidence_adjudication",
            status="UNKNOWN",
            record=record,
            field=field,
            family_id=family_id,
            split=split,
            source_lock=source_lock,
            tags=tags,
            augmentation=augmentation,
            context=unknown_context,
            assistant=unknown_target,
        ),
    ]


def build_record_examples(
    *,
    record: Mapping[str, Any],
    field: str,
    family_id: str,
    split: str,
    source_lock: SourceLock,
) -> list[dict[str, Any]]:
    if split not in TRAINING_SPLITS:
        raise ValueError("test records must not be materialized as semantic examples")
    if field not in _available_query_fields(record):
        raise ValueError(f"{field} is not available for {record.get('jid')}")
    tags = domain_tags(record)
    examples: list[dict[str, Any]] = []
    for builder in (_property_pair, _tool_pair, _adjudication_pair):
        examples.extend(
            builder(
                record=record,
                field=field,
                family_id=family_id,
                split=split,
                source_lock=source_lock,
                tags=tags,
            )
        )
    return examples


def validate_example(example: Mapping[str, Any]) -> None:
    Draft202012Validator(EXAMPLE_SCHEMA).validate(example)
    messages = example["messages"]
    model_facing_text = "\n".join(str(message["content"]) for message in messages)
    if HASH64_PATTERN.search(model_facing_text):
        raise ValueError("model-facing messages must not contain a 64-hex digest")
    assistant = json.loads(messages[-1]["content"])
    if not isinstance(assistant, dict):
        raise TypeError("assistant target must be one JSON object")
    Draft202012Validator(ASSISTANT_SCHEMAS[str(example["task"])]).validate(assistant)
    if canonical_json(assistant) != messages[-1]["content"]:
        raise ValueError("assistant target must use canonical JSON")
    if assistant["status"] != example["status_label"]:
        raise ValueError("assistant status does not match status_label")
    if assistant.get("requested_field", example["requested_field"]) != example["requested_field"]:
        raise ValueError("assistant requested_field mismatch")
    if assistant["evidence"]["record_id"] != example["host_binding"]["record_id"]:
        raise ValueError("assistant evidence record mismatch")
    expected_augmentation = {
        ("property_judgment", "SUPPORTED"): {"none"},
        ("property_judgment", "UNKNOWN"): {"masked_field"},
        ("tool_parameters", "SUPPORTED"): {"none"},
        ("tool_parameters", "UNKNOWN"): {"masked_field"},
        ("evidence_adjudication", "SUPPORTED"): {"none"},
        ("evidence_adjudication", "UNKNOWN"): {
            "synthetic_conflict",
            "synthetic_unit_error",
        },
    }
    allowed = expected_augmentation[(str(example["task"]), str(example["status_label"]))]
    if example["augmentation"] not in allowed:
        raise ValueError("augmentation does not match task and target status")


def _verify_source_lock(
    archive_path: Path,
    receipt_path: Path,
    source_catalog_path: Path,
) -> SourceLock:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    catalog = json.loads(source_catalog_path.read_text(encoding="utf-8"))
    catalog_record = next(
        (
            item
            for item in catalog.get("records", [])
            if isinstance(item, Mapping) and item.get("source_id") == SOURCE_ID
        ),
        None,
    )
    if catalog_record is None:
        raise ValueError(f"{SOURCE_ID} is absent from source catalog")
    if catalog_record.get("reuse_gate") != REQUIRED_REUSE_GATE:
        raise PermissionError("source catalog does not authorize training and redistribution")
    if receipt.get("reuse_gate") != REQUIRED_REUSE_GATE:
        raise PermissionError("acquisition receipt does not authorize training")
    if catalog_record.get("doi") != receipt.get("doi"):
        raise ValueError("source catalog and receipt DOI mismatch")
    if catalog_record.get("version") != receipt.get("source_version"):
        raise ValueError("source catalog and receipt version mismatch")

    archive_path = archive_path.resolve()
    actual_sha256 = sha256_file(archive_path)
    actual_bytes = archive_path.stat().st_size
    if actual_sha256 != receipt.get("sha256") or actual_bytes != receipt.get("bytes"):
        raise ValueError("JARVIS archive does not match acquisition receipt")
    with zipfile.ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
    if len(members) != 1 or not members[0].lower().endswith(".json"):
        raise ValueError("expected exactly one JSON member in JARVIS archive")
    return SourceLock(
        archive_path=archive_path,
        archive_sha256=actual_sha256,
        archive_bytes=actual_bytes,
        member_name=members[0],
        source_version=str(receipt["source_version"]),
        doi=str(receipt["doi"]),
        license_name=str(receipt["license_name"]),
        license_url=str(receipt["license_url"]),
        reuse_gate=str(receipt["reuse_gate"]),
        acquired_at=str(receipt["acquired_at"]),
    )


def _load_records(source_lock: SourceLock) -> list[dict[str, Any]]:
    with zipfile.ZipFile(source_lock.archive_path) as archive:
        with archive.open(source_lock.member_name) as handle:
            payload = json.load(handle)
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise TypeError("JARVIS archive member must be a list of objects")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_jsonl(path: Path, examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            validate_example(example)
            handle.write(canonical_json(example) + "\n")
    temporary.replace(path)
    return {
        "path": path.name,
        "examples": len(examples),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            validate_example(item)
            yield item


def _select_records(
    records: Sequence[Mapping[str, Any]],
    family_map: Mapping[str, str],
    max_records: int,
) -> list[tuple[Mapping[str, Any], str, str]]:
    if max_records <= 0:
        raise ValueError("max_records must be positive")
    eligible: list[tuple[str, Mapping[str, Any], str, str]] = []
    for record in records:
        record_id = _record_id(record)
        if record_id is None or record_id not in family_map:
            continue
        if not _available_query_fields(record):
            continue
        family_id = family_map[record_id]
        split = split_for_family(family_id)
        order = sha256_bytes(f"icmat-sft-v2|{record_id}".encode())
        eligible.append((order, record, family_id, split))
    eligible.sort(key=lambda item: item[0])
    selected = eligible[:max_records]
    if len(selected) < max_records:
        raise ValueError(
            f"requested {max_records} records but only {len(selected)} are eligible"
        )
    if set(split for _, _, _, split in selected) != set(SPLIT_NAMES):
        raise RuntimeError("deterministic selection did not populate all four splits")
    return [(record, family_id, split) for _, record, family_id, split in selected]


def _assign_fields(
    selected: Sequence[tuple[Mapping[str, Any], str, str]],
) -> dict[str, str]:
    counts_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    assigned: dict[str, str] = {}
    for record, _, split in selected:
        if split == "test":
            continue
        record_id = str(record["jid"])
        available = _available_query_fields(record)
        minimum = min(counts_by_split[split][field] for field in available)
        candidates = [
            field for field in available if counts_by_split[split][field] == minimum
        ]
        chosen = _deterministic_order(candidates, f"{record_id}|field-choice")[0]
        counts_by_split[split][chosen] += 1
        assigned[record_id] = chosen
    return assigned


def _pairwise_family_overlap(
    selected: Sequence[tuple[Mapping[str, Any], str, str]],
) -> dict[str, list[str]]:
    families: dict[str, set[str]] = defaultdict(set)
    for _, family_id, split in selected:
        families[split].add(family_id)
    overlap: dict[str, list[str]] = {}
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            overlap[f"{left}_{right}"] = sorted(families[left] & families[right])
    if any(overlap.values()):
        raise RuntimeError("connected-component family leakage detected")
    return overlap


def _balance_metrics(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    split_counts = Counter(str(item["split"]) for item in examples)
    split_family: dict[str, set[str]] = defaultdict(set)
    task_status = Counter()
    field_status = Counter()
    language_counts = Counter()
    augmentation_counts = Counter()
    domain_counts = Counter()
    domain_records: dict[str, set[str]] = defaultdict(set)
    record_ids: set[str] = set()
    for example in examples:
        split_family[str(example["split"])].add(str(example["family_id"]))
        task_status[(str(example["task"]), str(example["status_label"]))] += 1
        field_status[
            (
                str(example["requested_field"]),
                str(example["status_label"]),
            )
        ] += 1
        language_counts[str(example["language"])] += 1
        augmentation_counts[str(example["augmentation"])] += 1
        record_id = str(example["host_binding"]["record_id"])
        for tag in example["domain_tags"]:
            domain_counts[str(tag)] += 1
            domain_records[str(tag)].add(record_id)
        record_ids.add(record_id)

    field_balance: dict[str, dict[str, int]] = {}
    for field in QUERY_FIELDS:
        supported = field_status[(field, "SUPPORTED")]
        unknown = field_status[(field, "UNKNOWN")]
        if supported != unknown:
            raise RuntimeError(f"SUPPORTED/UNKNOWN imbalance for {field}")
        if supported:
            field_balance[field] = {
                "SUPPORTED": supported,
                "UNKNOWN": unknown,
                "difference": supported - unknown,
            }

    task_balance: dict[str, dict[str, int]] = {}
    for task in TASK_NAMES:
        supported = task_status[(task, "SUPPORTED")]
        unknown = task_status[(task, "UNKNOWN")]
        if supported != unknown:
            raise RuntimeError(f"SUPPORTED/UNKNOWN imbalance for {task}")
        task_balance[task] = {
            "SUPPORTED": supported,
            "UNKNOWN": unknown,
            "difference": supported - unknown,
        }

    return {
        "scope": "train_validation_calibration_only",
        "test_semantic_metrics_included": False,
        "example_count": len(examples),
        "record_count": len(record_ids),
        "split_example_counts": {
            split: split_counts[split] for split in TRAINING_SPLITS
        },
        "split_family_counts": {
            split: len(split_family[split]) for split in TRAINING_SPLITS
        },
        "task_status_balance": task_balance,
        "field_status_balance": field_balance,
        "language_counts": dict(sorted(language_counts.items())),
        "augmentation_counts": dict(sorted(augmentation_counts.items())),
        "domain_record_coverage": {
            tag: len(records) for tag, records in sorted(domain_records.items())
        },
        "domain_example_occurrences": dict(sorted(domain_counts.items())),
        "json_schema_valid_rate": 1.0,
        "model_facing_sha256_rate": 0.0,
    }


def build_dataset(
    *,
    archive_path: Path,
    receipt_path: Path,
    source_catalog_path: Path,
    dataset_output_dir: Path,
    evaluation_output_dir: Path,
    max_records: int = 4096,
) -> dict[str, Any]:
    """Generate training splits and a membership-only sealed final test."""
    for output_dir in (dataset_output_dir, evaluation_output_dir):
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                "v2 outputs are immutable; choose unused or empty output directories"
            )

    source_lock = _verify_source_lock(
        archive_path,
        receipt_path,
        source_catalog_path,
    )
    records = _load_records(source_lock)
    family_map, family_graph_audit = build_family_map(records)
    selected = _select_records(records, family_map, max_records)
    family_overlap = _pairwise_family_overlap(selected)
    assigned_fields = _assign_fields(selected)

    examples: list[dict[str, Any]] = []
    examples_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    test_records: list[dict[str, str]] = []
    record_counts = Counter()
    family_sets: dict[str, set[str]] = defaultdict(set)
    for record, family_id, split in selected:
        record_id = str(record["jid"])
        record_counts[split] += 1
        family_sets[split].add(family_id)
        if split == "test":
            test_records.append(
                {
                    "record_id": record_id,
                    "family_id": family_id,
                    "canonical_complete_record_sha256": canonical_complete_record_hash(
                        record
                    ),
                }
            )
            continue
        record_examples = build_record_examples(
            record=record,
            field=assigned_fields[record_id],
            family_id=family_id,
            split=split,
            source_lock=source_lock,
        )
        examples.extend(record_examples)
        examples_by_split[split].extend(record_examples)

    balance = _balance_metrics(examples)
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    training_files = [
        _write_jsonl(dataset_output_dir / f"{split}.jsonl", examples_by_split[split])
        for split in TRAINING_SPLITS
    ]

    membership_payload = {
        "schema": TEST_MEMBERSHIP_SCHEMA_ID,
        "source_id": SOURCE_ID,
        "source_version": source_lock.source_version,
        "split": "test",
        "semantic_examples_materialized": False,
        "semantic_metrics_emitted": False,
        "records": sorted(test_records, key=lambda item: item["record_id"]),
    }
    membership_seal = sha256_bytes(
        canonical_json(membership_payload).encode("utf-8")
    )
    membership_document = {
        **membership_payload,
        "membership_payload_sha256": membership_seal,
    }
    membership_file = _write_json_atomic(
        dataset_output_dir / "test_membership.sealed.json",
        membership_document,
    )

    family_audit = {
        "schema": "icmat_sft_family_audit.v2",
        "source_id": SOURCE_ID,
        "family_graph": family_graph_audit,
        "selected_record_count": len(selected),
        "selected_record_counts": {
            split: record_counts[split] for split in SPLIT_NAMES
        },
        "selected_family_counts": {
            split: len(family_sets[split]) for split in SPLIT_NAMES
        },
        "family_overlap": family_overlap,
        "group_disjoint": True,
        "coordinate_contract": (
            "cartesian=true: fractional=cartesian@inverse(lattice); "
            "cartesian=false: coordinates are already fractional"
        ),
    }
    evaluation_output_dir.mkdir(parents=True, exist_ok=True)
    family_audit_evidence_file = _write_json_atomic(
        evaluation_output_dir / "family_audit.v2.json",
        family_audit,
    )
    balance_document = {
        "schema": "icmat_sft_balance_audit.v2",
        **balance,
    }
    balance_evidence_file = _write_json_atomic(
        evaluation_output_dir / "balance_audit.v2.json",
        balance_document,
    )
    # Training verification is intentionally self-contained: manifest-relative
    # audit bindings live beside the JSONL files, while evaluation keeps copies.
    family_audit_file = _write_json_atomic(
        dataset_output_dir / "family_audit.v2.json",
        family_audit,
    )
    balance_file = _write_json_atomic(
        dataset_output_dir / "balance_audit.v2.json",
        balance_document,
    )

    manifest = {
        "schema": DATASET_SCHEMA_ID,
        "builder_version": BUILDER_VERSION,
        "deterministic_timestamp": source_lock.acquired_at,
        "status": "SFT_V2_DATA_READY_NOT_TRAINED_NOT_DEPLOYED",
        "model_target": "ICMat-Qwen-0.5B",
        "production_integration_allowed": False,
        "qlora_training_started": False,
        "network_used": False,
        "teacher_model_used": False,
        "api_used": False,
        "source_lock": source_lock.as_dict(),
        "selection": {
            "max_records": max_records,
            "source_record_count": len(records),
            "family_graph_record_count": family_graph_audit["identifiable_records"],
            "selected_record_counts": {
                split: record_counts[split] for split in SPLIT_NAMES
            },
            "selected_family_counts": {
                split: len(family_sets[split]) for split in SPLIT_NAMES
            },
            "record_order": "sha256('icmat-sft-v2|' + jid)",
        },
        "family_contract": {
            "connected_component_edges": [
                "normalized_reference",
                "element_aware_reduced_formula",
                "coordinate_semantics_aware_approximate_structure_fingerprint",
            ],
            "assignment": (
                "sha256(family_id) modulo 100: train<76, validation<84, "
                "calibration<92, test"
            ),
            "family_overlap": family_overlap,
            "group_disjoint": True,
        },
        "target_contract": {
            "tasks": list(TASK_NAMES),
            "paired_statuses_per_record_and_task": ["SUPPORTED", "UNKNOWN"],
            "languages": ["zh", "en"],
            "negative_evidence": [
                "masked_field",
                "synthetic_conflict",
                "synthetic_unit_error",
            ],
            "assistant_only_loss_required": True,
            "strict_json_schema_validation": True,
            "model_generates_sha256": False,
            "complete_record_sha256_host_bound_only": True,
        },
        "domain_label_contract": {
            "labels_describe_available_computed_descriptors": True,
            "labels_are_device_suitability_ground_truth": False,
            "labels_are_production_line_ground_truth": False,
        },
        "training_metrics": balance,
        "final_test_contract": {
            "membership_only": True,
            "semantic_examples_materialized": False,
            "semantic_metrics_emitted": False,
            "record_count": record_counts["test"],
            "family_count": len(family_sets["test"]),
            "membership_payload_sha256": membership_seal,
        },
        "files": {
            "training": training_files,
            "sealed_test_membership": membership_file,
            "family_audit": family_audit_file,
            "balance_audit": balance_file,
        },
        "claim_boundary": (
            "This dataset teaches evidence-bound operations over public computed "
            "JARVIS-DFT records. It is not experimental data, fab-line data, a trained "
            "domain model, model-quality evidence, BPU/X5 evidence, or production evidence."
        ),
    }
    manifest_file = _write_json_atomic(
        dataset_output_dir / "manifest.v2.json",
        manifest,
    )
    build_report = {
        "schema": "icmat_sft_v2_build_report.v2",
        "status": manifest["status"],
        "dataset_manifest": manifest_file,
        "source_lock": source_lock.as_dict(),
        "selected_record_counts": manifest["selection"]["selected_record_counts"],
        "selected_family_counts": manifest["selection"]["selected_family_counts"],
        "family_overlap": family_overlap,
        "training_metrics": balance,
        "final_test_contract": manifest["final_test_contract"],
        "evaluation_artifacts": {
            "family_audit": family_audit_evidence_file,
            "balance_audit": balance_evidence_file,
        },
        "claim_boundary": manifest["claim_boundary"],
    }
    _write_json_atomic(
        evaluation_output_dir / "build_report.v2.json",
        build_report,
    )
    return manifest
