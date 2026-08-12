"""Build provenance-bound SFT data for ICMat-Qwen-0.5B.

The builder uses only fields present in the pinned JARVIS record. It does not
call a model, a network service, or a production module.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


BUILDER_VERSION = "icmat-qwen05b-sft-builder-1.1.0"
DATASET_SCHEMA = "icmat_qwen05b_sft.v1"
EXAMPLE_SCHEMA = "icmat_sft_example.v1"
SOURCE_ID = "nist_jarvis_dft"
REQUIRED_REUSE_GATE = "ALLOW_TRAIN_REDISTRIBUTE"

SYSTEM_PROMPT = (
    "ICMat structured materials assistant. Use only SOURCE_RECORD_JSON. "
    "Return one JSON object. Never infer; return UNKNOWN for missing values."
)

CORE_FIELDS: tuple[str, ...] = (
    "formula",
    "spg_number",
    "spg_symbol",
    "crys",
    "dimensionality",
)

PROPERTY_FIELDS: tuple[str, ...] = (
    "formation_energy_peratom",
    "optb88vdw_bandgap",
    "mbj_bandgap",
    "hse_gap",
    "ehull",
    "density",
    "epsx",
    "epsy",
    "epsz",
    "avg_elec_mass",
    "avg_hole_mass",
    "bulk_modulus_kv",
    "shear_modulus_gv",
    "dfpt_piezo_max_dij",
    "slme",
)

UNKNOWN_FIELDS: tuple[str, ...] = (
    "mbj_bandgap",
    "hse_gap",
    "avg_elec_mass",
    "avg_hole_mass",
    "bulk_modulus_kv",
    "shear_modulus_gv",
    "dfpt_piezo_max_dij",
    "slme",
    "Tc_supercon",
)

NA_STRINGS = {"", "na", "n/a", "none", "null", "nan", "inf", "-inf"}
SPLIT_NAMES = ("train", "validation", "test")


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
        }


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


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_scalar(value: Any) -> str | int | float | bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in NA_STRINGS:
            return None
        return stripped
    return None


def _normalized_record(record: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in (*CORE_FIELDS, *PROPERTY_FIELDS):
        value = _safe_scalar(record.get(field))
        if value is not None:
            result[field] = value
    return result


def _anonymized_stoichiometry(record: Mapping[str, Any]) -> str:
    atoms = record.get("atoms")
    elements = atoms.get("elements") if isinstance(atoms, Mapping) else None
    if isinstance(elements, list) and elements and all(isinstance(x, str) for x in elements):
        counts = Counter(elements)
        divisor = 0
        for count in counts.values():
            divisor = math.gcd(divisor, count)
        reduced = sorted(count // max(divisor, 1) for count in counts.values())
        return "_".join(str(item) for item in reduced)

    formula = str(record.get("formula", ""))
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    if tokens and "".join(element + count for element, count in tokens) == formula:
        counts = [int(count or "1") for _, count in tokens]
        divisor = 0
        for count in counts:
            divisor = math.gcd(divisor, count)
        return "_".join(str(item // max(divisor, 1)) for item in sorted(counts))
    return "unknown"


def material_family_id(record: Mapping[str, Any]) -> str:
    """Conservatively group substitution analogues into one split."""
    signature = {
        "crystal_system": _safe_scalar(record.get("crys")) or "unknown",
        "dimensionality": _safe_scalar(record.get("dimensionality")) or "unknown",
        "space_group": str(_safe_scalar(record.get("spg_number")) or "unknown"),
        "anonymous_stoichiometry": _anonymized_stoichiometry(record),
    }
    return "family_" + sha256_bytes(canonical_json(signature).encode("utf-8"))[:20]


def split_for_group(group_id: str) -> str:
    bucket = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _source_evidence(
    source_lock: SourceLock,
    record: Mapping[str, Any],
    record_facts: Mapping[str, Any],
) -> dict[str, Any]:
    jid = str(record["jid"])
    record_hash = sha256_bytes(canonical_json(record_facts).encode("utf-8"))
    return {
        "source_id": SOURCE_ID,
        "source_version": source_lock.source_version,
        "doi": source_lock.doi,
        "record_id": jid,
        "record_hash": record_hash,
        "license": source_lock.license_name,
    }


def _assistant_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: evidence[key]
        for key in ("source_id", "source_version", "record_id", "record_hash")
    }


def _example_id(task: str, evidence: Mapping[str, Any], assistant: Mapping[str, Any]) -> str:
    identity = {
        "schema": EXAMPLE_SCHEMA,
        "task": task,
        "record_id": evidence["record_id"],
        "record_hash": evidence["record_hash"],
        "assistant": assistant,
    }
    return "sft_" + sha256_bytes(canonical_json(identity).encode("utf-8"))[:24]


def _make_example(
    *,
    task: str,
    group_id: str,
    user_content: str,
    assistant: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    assistant_text = canonical_json(assistant)
    example = {
        "schema": EXAMPLE_SCHEMA,
        "example_id": _example_id(task, evidence, assistant),
        "task": task,
        "group_id": group_id,
        "split": split_for_group(group_id),
        "source": dict(evidence),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_text},
        ],
    }
    validate_example(example)
    return example


def _structured_extraction_example(
    record: Mapping[str, Any],
    source_lock: SourceLock,
    group_id: str,
) -> dict[str, Any]:
    available = _normalized_record(record)
    selected: dict[str, Any] = {}
    for field in CORE_FIELDS:
        if field in available:
            selected[field] = available[field]
    for field in PROPERTY_FIELDS:
        if field in available and len(selected) < 8:
            selected[field] = available[field]

    evidence = _source_evidence(source_lock, record, available)
    target_evidence = _assistant_evidence(evidence)
    source_record = {
        "record": {"jid": str(record["jid"]), **selected},
        "evidence": target_evidence,
    }
    assistant = {
        "schema": "icmat.extract.v1",
        "status": "SUPPORTED",
        "jid": str(record["jid"]),
        "facts": selected,
        "evidence": target_evidence,
    }
    user = (
        "Extract fields exactly; do not convert or complete values.\n"
        f"SOURCE_RECORD_JSON={canonical_json(source_record)}"
    )
    return _make_example(
        task="structured_extraction",
        group_id=group_id,
        user_content=user,
        assistant=assistant,
        evidence=evidence,
    )


def _tool_parameter_example(
    record: Mapping[str, Any],
    source_lock: SourceLock,
    group_id: str,
) -> dict[str, Any]:
    available = _normalized_record(record)
    evidence = _source_evidence(source_lock, record, available)
    target_evidence = _assistant_evidence(evidence)
    requested = [
        field
        for field in (
            "formula",
            "spg_number",
            "optb88vdw_bandgap",
            "formation_energy_peratom",
        )
        if field in available
    ]
    assistant = {
        "schema": "icmat.tool.v1",
        "status": "READY",
        "tool": "lookup_versioned_jarvis_record",
        "arguments": {
            "source_id": SOURCE_ID,
            "source_version": source_lock.source_version,
            "record_id": str(record["jid"]),
            "fields": requested,
        },
        "evidence": target_evidence,
    }
    source_record = {
        "record": {
            "jid": str(record["jid"]),
            "available_fields": requested,
        },
        "evidence": target_evidence,
    }
    user = (
        "Build read-only lookup arguments; copy only the supplied identifiers and fields.\n"
        f"SOURCE_RECORD_JSON={canonical_json(source_record)}"
    )
    return _make_example(
        task="tool_parameters",
        group_id=group_id,
        user_content=user,
        assistant=assistant,
        evidence=evidence,
    )


def _unknown_example(
    record: Mapping[str, Any],
    source_lock: SourceLock,
    group_id: str,
) -> dict[str, Any]:
    available = _normalized_record(record)
    missing_field = next((field for field in UNKNOWN_FIELDS if field not in available), None)
    if missing_field is None:
        missing_field = "experimental_fab_yield"
    evidence = _source_evidence(source_lock, record, available)
    target_evidence = _assistant_evidence(evidence)
    visible = {
        key: available[key]
        for key in ("formula", "spg_number")
        if key in available
    }
    visible["jid"] = str(record["jid"])
    assistant = {
        "schema": "icmat.unknown.v1",
        "status": "UNKNOWN",
        "requested_field": missing_field,
        "value": None,
        "reason": "FIELD_NOT_PRESENT_IN_SOURCE_RECORD",
        "evidence": target_evidence,
    }
    source_record = {"record": visible, "evidence": target_evidence}
    user = (
        f"Answer field {missing_field} for {record['jid']}; return UNKNOWN if absent.\n"
        f"SOURCE_RECORD_JSON={canonical_json(source_record)}"
    )
    return _make_example(
        task="grounded_unknown",
        group_id=group_id,
        user_content=user,
        assistant=assistant,
        evidence=evidence,
    )


def build_examples(
    records: Iterable[Mapping[str, Any]],
    source_lock: SourceLock,
    *,
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    for record in records:
        jid = _safe_scalar(record.get("jid"))
        normalized = _normalized_record(record)
        if not isinstance(jid, str) or "formula" not in normalized:
            continue
        if len(normalized) < 5:
            continue
        order_key = hashlib.sha256(jid.encode("utf-8")).hexdigest()
        candidates.append((order_key, record))

    candidates.sort(key=lambda item: item[0])
    if max_records is not None:
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        candidates = candidates[:max_records]
    if not candidates:
        raise ValueError("no eligible JARVIS records")

    examples: list[dict[str, Any]] = []
    for _, record in candidates:
        group_id = material_family_id(record)
        examples.extend(
            (
                _structured_extraction_example(record, source_lock, group_id),
                _tool_parameter_example(record, source_lock, group_id),
                _unknown_example(record, source_lock, group_id),
            )
        )
    return examples


def validate_example(example: Mapping[str, Any]) -> None:
    required = {"schema", "example_id", "task", "group_id", "split", "source", "messages"}
    if set(example) != required:
        raise ValueError(f"unexpected example keys: {sorted(set(example) ^ required)}")
    if example["schema"] != EXAMPLE_SCHEMA:
        raise ValueError("unexpected example schema")
    if example["split"] not in SPLIT_NAMES:
        raise ValueError("unexpected split")
    if split_for_group(str(example["group_id"])) != example["split"]:
        raise ValueError("group split mismatch")

    messages = example["messages"]
    if not isinstance(messages, list) or [item.get("role") for item in messages] != [
        "system",
        "user",
        "assistant",
    ]:
        raise ValueError("messages must be system/user/assistant")
    assistant = json.loads(messages[-1]["content"])
    if not isinstance(assistant, dict):
        raise ValueError("assistant target must be one JSON object")
    assistant_evidence = assistant.get("evidence")
    if not isinstance(assistant_evidence, dict) or not assistant_evidence:
        raise ValueError("assistant evidence must be a non-empty object")
    if any(
        example["source"].get(key) != value
        for key, value in assistant_evidence.items()
    ):
        raise ValueError("assistant evidence must be a subset of example source")
    if assistant.get("status") not in {"SUPPORTED", "READY", "UNKNOWN"}:
        raise ValueError("unexpected target status")
    if example["task"] == "grounded_unknown":
        if assistant.get("status") != "UNKNOWN" or assistant.get("value", "missing") is not None:
            raise ValueError("grounded_unknown must carry a null UNKNOWN target")
    canonical_json(assistant)


def _verify_source_lock(
    archive_path: Path,
    receipt_path: Path,
    source_catalog_path: Path,
) -> SourceLock:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    catalog = json.loads(source_catalog_path.read_text(encoding="utf-8"))
    records = {
        item["source_id"]: item
        for item in catalog.get("records", [])
        if isinstance(item, dict) and "source_id" in item
    }
    source_record = records.get(SOURCE_ID)
    if source_record is None:
        raise ValueError(f"{SOURCE_ID} is absent from source catalog")
    if source_record.get("reuse_gate") != REQUIRED_REUSE_GATE:
        raise PermissionError("JARVIS source is not approved for train and redistribution")
    if receipt.get("reuse_gate") != REQUIRED_REUSE_GATE:
        raise PermissionError("acquisition receipt does not authorize training")
    if source_record.get("doi") != receipt.get("doi"):
        raise ValueError("source catalog and acquisition receipt DOI mismatch")
    if source_record.get("version") != receipt.get("source_version"):
        raise ValueError("source catalog and acquisition receipt version mismatch")

    actual_bytes = archive_path.stat().st_size
    actual_sha256 = sha256_file(archive_path)
    if actual_bytes != receipt.get("bytes") or actual_sha256 != receipt.get("sha256"):
        raise ValueError("JARVIS archive does not match acquisition receipt")
    with zipfile.ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
    if len(members) != 1 or not members[0].lower().endswith(".json"):
        raise ValueError("expected one JSON member in JARVIS archive")

    return SourceLock(
        archive_path=archive_path.resolve(),
        archive_sha256=actual_sha256,
        archive_bytes=actual_bytes,
        member_name=members[0],
        source_version=str(receipt["source_version"]),
        doi=str(receipt["doi"]),
        license_name=str(receipt["license_name"]),
        license_url=str(receipt["license_url"]),
        reuse_gate=str(receipt["reuse_gate"]),
    )


def _load_records(source_lock: SourceLock) -> list[dict[str, Any]]:
    with zipfile.ZipFile(source_lock.archive_path) as archive:
        with archive.open(source_lock.member_name) as handle:
            payload = json.load(handle)
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise TypeError("JARVIS archive member must contain a list of objects")
    return payload


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


def _dataset_metrics(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    task_counts = Counter(str(item["task"]) for item in examples)
    split_counts = Counter(str(item["split"]) for item in examples)
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    unknown_count = 0
    for example in examples:
        groups_by_split[str(example["split"])].add(str(example["group_id"]))
        assistant = json.loads(example["messages"][-1]["content"])
        unknown_count += int(assistant.get("status") == "UNKNOWN")

    overlap = {
        "train_validation": sorted(groups_by_split["train"] & groups_by_split["validation"]),
        "train_test": sorted(groups_by_split["train"] & groups_by_split["test"]),
        "validation_test": sorted(groups_by_split["validation"] & groups_by_split["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError("material-family split leakage detected")
    return {
        "example_count": len(examples),
        "record_count": len(examples) // 3,
        "group_count": len({str(item["group_id"]) for item in examples}),
        "task_counts": dict(sorted(task_counts.items())),
        "split_counts": {name: split_counts[name] for name in SPLIT_NAMES},
        "split_group_counts": {
            name: len(groups_by_split[name])
            for name in SPLIT_NAMES
        },
        "group_overlap": overlap,
        "unknown_count": unknown_count,
        "json_target_valid_rate": 1.0,
        "evidence_bound_rate": 1.0,
    }


def build_dataset(
    *,
    archive_path: Path,
    receipt_path: Path,
    source_catalog_path: Path,
    output_dir: Path,
    max_records: int = 4096,
) -> dict[str, Any]:
    source_lock = _verify_source_lock(archive_path, receipt_path, source_catalog_path)
    records = _load_records(source_lock)
    examples = build_examples(records, source_lock, max_records=max_records)
    metrics = _dataset_metrics(examples)

    split_examples = {
        name: [item for item in examples if item["split"] == name]
        for name in SPLIT_NAMES
    }
    if any(not values for values in split_examples.values()):
        raise RuntimeError("all train/validation/test splits must be non-empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    files = [
        _write_jsonl(output_dir / f"{name}.jsonl", split_examples[name])
        for name in SPLIT_NAMES
    ]
    manifest = {
        "schema": DATASET_SCHEMA,
        "builder_version": BUILDER_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "SFT_CANDIDATE_NOT_TRAINED_NOT_DEPLOYED",
        "model_target": "ICMat-Qwen-0.5B",
        "production_integration_allowed": False,
        "network_used": False,
        "teacher_model_used": False,
        "api_used": False,
        "source_lock": source_lock.as_dict(),
        "selection": {
            "max_records": max_records,
            "eligible_records_in_source": sum(
                1
                for row in records
                if isinstance(_safe_scalar(row.get("jid")), str)
                and "formula" in _normalized_record(row)
                and len(_normalized_record(row)) >= 5
            ),
            "deterministic_order": "sha256(jid)",
        },
        "split_contract": {
            "group_key": (
                "sha256(crystal_system,dimensionality,space_group,"
                "anonymous_reduced_stoichiometry)"
            ),
            "assignment": "sha256(group_id) modulo 100: train<80, validation<90, test",
            "near_family_overlap_allowed": False,
        },
        "target_contract": {
            "tasks": [
                "structured_extraction",
                "tool_parameters",
                "grounded_unknown",
            ],
            "assistant_only_loss_required": True,
            "external_scientific_inference_allowed": False,
            "unknown_when_missing_required": True,
        },
        "metrics": metrics,
        "files": files,
        "claim_boundary": (
            "Targets are deterministic transformations of version-pinned public JARVIS-DFT "
            "records. Values are computed-material fields, not experimental or fab-line "
            "ground truth. This artifact is not model-quality, BPU, X5, or production evidence."
        ),
    }
    write_json_atomic(output_dir / "manifest.v1.json", manifest)
    return manifest


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
