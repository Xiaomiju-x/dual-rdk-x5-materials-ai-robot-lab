from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from icmat_foundry.llm import evidence_sft_v6 as evidence
from icmat_foundry.llm.evidence_sft_v6 import (
    ACCEPTED_INVENTORY_SCHEMA,
    BUILDER_VERSION,
    COMPILER_EVIDENCE_FIELDS,
    COMPILER_PROMPT_FIELDS,
    COMPILER_PROMPT_SCHEMA,
    COMPILER_SENTENCE_FIELDS,
    COMPILER_VERSION,
    DATASET_SCHEMA,
    DECISIONS,
    EXAMPLES_PER_FAMILY,
    EXPECTED_FAMILY_COUNT,
    EXPECTED_FAMILY_SPLIT_COUNTS,
    EXPECTED_SPLIT_COUNTS,
    EXTERNAL_ANSWER_FIELDS,
    EXTERNAL_ANSWER_SCHEMA,
    NONBLIND_SPLITS,
    POINTER_FIELDS,
    SEMANTIC_QUERY_SCHEMA,
    TASKS,
    TRAINING_SPLITS,
    EvidenceSFTV6Error,
    SourceFamily,
    _content_leakage_report,
    _group_commitment,
    _write_json,
    _write_jsonl,
    assign_family_splits,
    build_examples,
    canonical_json,
    load_licensed_families,
    load_semantic_inventory,
    sha256_bytes,
)
from icmat_foundry.rag.contracts import (
    ChunkV1,
    ContractError,
    RegistryManifestV2,
)

NONBLIND_BUILDER_VERSION = "icmat-evidence-nonblind-v7.1.0"
SPLIT_ALGORITHM_VERSION = "icmat-semantic-v7-nonblind-split-v1"
NONBLIND_MANIFEST_SCHEMA = "icmat_evidence_pointer_nonblind_manifest.v7"
NONBLIND_REPORT_SCHEMA = "icmat_evidence_pointer_nonblind_build_report.v7"
NONBLIND_BALANCE_SCHEMA = "icmat_evidence_pointer_nonblind_balance_audit.v7"
NONBLIND_GROUP_SCHEMA = "icmat_evidence_pointer_nonblind_group_audit.v7"
NONBLIND_LEAKAGE_SCHEMA = "icmat_evidence_pointer_nonblind_leakage_audit.v7"
PREBLIND_COMMITMENT_SCHEMA = "icmat_evidence_pointer_preblind_commitment.v7"

EXPECTED_NONBLIND_SPLIT_COUNTS = {
    split: EXPECTED_SPLIT_COUNTS[split]
    for split in NONBLIND_SPLITS
}
EXPECTED_NONBLIND_FAMILY_SPLIT_COUNTS = {
    split: EXPECTED_FAMILY_SPLIT_COUNTS[split]
    for split in NONBLIND_SPLITS
}
EXPECTED_NONBLIND_TOTAL = sum(EXPECTED_NONBLIND_SPLIT_COUNTS.values())
EXPECTED_BLIND_COUNT = (
    sum(EXPECTED_SPLIT_COUNTS.values()) - EXPECTED_NONBLIND_TOTAL
)

OUTPUT_FILENAMES = (
    "train.jsonl",
    "validation.jsonl",
    "calibration.jsonl",
    "balance_audit.nonblind.v7.json",
    "group_isolation_audit.nonblind.v7.json",
    "content_leakage_audit.nonblind.v7.json",
    "semantic_inventory_audit.v7.json",
    "preblind_commitment.v7.json",
    "build_report.nonblind.v7.json",
    "manifest.nonblind.v7.json",
)

_SNAPSHOT_ROLES = (
    "licensed_chunks",
    "rag_manifest",
    "semantic_inventory",
    "semantic_records",
    "nonblind_module",
    "evidence_core",
)


@dataclass(frozen=True)
class StableFileSnapshot:
    role: str
    path: Path
    identity: tuple[int, int, int, int, int]
    payload: bytes
    sha256: str
    byte_count: int


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _capture_stable_file(
    path: Path,
    *,
    role: str,
) -> StableFileSnapshot:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise EvidenceSFTV6Error(
            f"{role}: regular non-symlink file required"
        )
    with resolved.open("rb") as handle:
        before = _stat_identity(os.fstat(handle.fileno()))
        payload = handle.read()
        after = _stat_identity(os.fstat(handle.fileno()))
    current = _stat_identity(resolved.stat())
    if before != after or after != current:
        raise EvidenceSFTV6Error(
            f"{role}: file changed while snapshot was captured"
        )
    if len(payload) != current[2]:
        raise EvidenceSFTV6Error(
            f"{role}: snapshot byte count does not match identity"
        )
    return StableFileSnapshot(
        role=role,
        path=resolved,
        identity=current,
        payload=payload,
        sha256=sha256_bytes(payload),
        byte_count=len(payload),
    )


def _builder_source_paths() -> tuple[Path, Path]:
    return Path(__file__).resolve(), Path(evidence.__file__).resolve()


def _capture_snapshot_set(
    *,
    chunks_path: Path,
    rag_manifest_path: Path,
    semantic_inventory_path: Path,
) -> dict[str, StableFileSnapshot]:
    module_path, core_path = _builder_source_paths()
    records_path = semantic_inventory_path.with_name(
        "records.v7.jsonl"
    )
    paths = {
        "licensed_chunks": chunks_path,
        "rag_manifest": rag_manifest_path,
        "semantic_inventory": semantic_inventory_path,
        "semantic_records": records_path,
        "nonblind_module": module_path,
        "evidence_core": core_path,
    }
    snapshots = {
        role: _capture_stable_file(path, role=role)
        for role, path in paths.items()
    }
    if tuple(snapshots) != _SNAPSHOT_ROLES:
        raise EvidenceSFTV6Error(
            "internal snapshot role ordering mismatch"
        )
    return snapshots


def _verify_snapshot_set(
    snapshots: Mapping[str, StableFileSnapshot],
    *,
    phase: str,
) -> None:
    if tuple(snapshots) != _SNAPSHOT_ROLES:
        raise EvidenceSFTV6Error(
            f"{phase}: snapshot role inventory mismatch"
        )
    for role in _SNAPSHOT_ROLES:
        expected = snapshots[role]
        try:
            observed = _capture_stable_file(
                expected.path,
                role=role,
            )
        except (OSError, EvidenceSFTV6Error) as exc:
            raise EvidenceSFTV6Error(
                f"{phase}: {role} snapshot is no longer stable"
            ) from exc
        if (
            observed.path != expected.path
            or observed.identity != expected.identity
            or observed.byte_count != expected.byte_count
            or observed.sha256 != expected.sha256
            or observed.payload != expected.payload
        ):
            raise EvidenceSFTV6Error(
                f"{phase}: {role} identity or bytes changed"
            )


def _decode_json_snapshot(
    snapshot: StableFileSnapshot,
) -> dict[str, Any]:
    try:
        value = evidence._strict_json_mapping(
            snapshot.payload.decode("utf-8"),
            label=snapshot.role,
        )
    except (UnicodeDecodeError, EvidenceSFTV6Error) as exc:
        raise EvidenceSFTV6Error(
            f"{snapshot.role}: invalid UTF-8 JSON"
        ) from exc
    return value


def _decode_chunk_snapshot(
    snapshot: StableFileSnapshot,
) -> tuple[ChunkV1, ...]:
    chunks: list[ChunkV1] = []
    try:
        text = snapshot.payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceSFTV6Error(
            "licensed_chunks: invalid UTF-8"
        ) from exc
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = evidence._strict_json_mapping(
                line,
                label=f"licensed_chunks:{line_number}",
            )
            chunks.append(ChunkV1.from_dict(value))
        except (EvidenceSFTV6Error, ContractError) as exc:
            raise EvidenceSFTV6Error(
                f"licensed_chunks:{line_number}: contract mismatch"
            ) from exc
    if not chunks:
        raise EvidenceSFTV6Error(
            "licensed_chunks: at least one row is required"
        )
    return tuple(chunks)


def _validate_rag_binding(
    *,
    manifest_snapshot: StableFileSnapshot,
    chunks_snapshot: StableFileSnapshot,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_payload = _decode_json_snapshot(manifest_snapshot)
    try:
        manifest = RegistryManifestV2.from_dict(manifest_payload)
    except ContractError as exc:
        raise EvidenceSFTV6Error(
            "RAG manifest canonical contract mismatch"
        ) from exc
    chunks = _decode_chunk_snapshot(chunks_snapshot)
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise EvidenceSFTV6Error(
            "licensed_chunks contains duplicate chunk_id"
        )

    entries = {
        entry.namespace: entry
        for entry in manifest.namespaces
    }
    namespace_counts = Counter(chunk.namespace for chunk in chunks)
    source_counts = Counter(chunk.source_id for chunk in chunks)
    observed_sources: set[str] = set()
    observed_source_namespaces: dict[str, str] = {}
    findings: list[str] = []
    for namespace in evidence.DOMAINS:
        entry = entries[namespace]
        if (
            entry.source_mode
            != "licensed_metadata_and_fulltext_readonly"
        ):
            findings.append(
                f"{namespace}:SOURCE_MODE_MISMATCH"
            )
        declared_literature_count = int(
            entry.evidence_counts["literature_knowledge"]
        )
        if namespace_counts[namespace] != declared_literature_count:
            findings.append(
                f"{namespace}:LITERATURE_COUNT_MISMATCH"
            )
        fulltext_assets = {
            asset.source_id: asset
            for asset in entry.source_assets
            if (
                asset.access_mode == "licensed_fulltext_readonly"
                and asset.license_id == "CC BY 4.0"
            )
        }
        fulltext_asset_list = [
            asset
            for asset in entry.source_assets
            if (
                asset.access_mode == "licensed_fulltext_readonly"
                and asset.license_id == "CC BY 4.0"
            )
        ]
        if len(fulltext_assets) != len(fulltext_asset_list):
            findings.append(
                f"{namespace}:DUPLICATE_FULLTEXT_SOURCE_ASSET"
            )
        all_asset_ids = [
            asset.source_id
            for asset in entry.source_assets
        ]
        if len(all_asset_ids) != len(set(all_asset_ids)):
            findings.append(
                f"{namespace}:DUPLICATE_SOURCE_ASSET"
            )
        namespace_sources = {
            chunk.source_id
            for chunk in chunks
            if chunk.namespace == namespace
        }
        if namespace_sources != set(fulltext_assets):
            findings.append(
                f"{namespace}:FULLTEXT_SOURCE_SET_MISMATCH"
            )
        observed_sources.update(namespace_sources)
        for source_id in namespace_sources:
            prior_namespace = observed_source_namespaces.setdefault(
                source_id,
                namespace,
            )
            if prior_namespace != namespace:
                findings.append(
                    f"{source_id}:CROSS_NAMESPACE_SOURCE_REUSE"
                )
        for chunk in chunks:
            if chunk.namespace != namespace:
                continue
            metadata = chunk.metadata
            asset = fulltext_assets.get(chunk.source_id)
            if (
                chunk.evidence_kind != "literature_knowledge"
                or chunk.license_id != "CC BY 4.0"
                or metadata.get("access_mode")
                != "licensed_fulltext_readonly"
                or metadata.get("license_verified_from_jats")
                is not True
                or metadata.get("license_url")
                != "https://creativecommons.org/licenses/by/4.0/"
                or metadata.get("measurement_status")
                != "published_literature_not_local_measurement"
                or not 600 <= len(chunk.text) <= 1200
                or asset is None
                or metadata.get("xml_sha256") != asset.sha256
            ):
                findings.append(
                    f"{namespace}:{chunk.source_id}:PROVENANCE_MISMATCH"
                )
    if any(
        chunk.namespace not in evidence.DOMAINS
        for chunk in chunks
    ):
        findings.append("UNEXPECTED_LICENSED_CHUNK_NAMESPACE")
    if sum(namespace_counts.values()) != len(chunks):
        findings.append("LICENSED_CHUNK_NAMESPACE_COUNT_MISMATCH")
    if len(observed_sources) != EXPECTED_FAMILY_COUNT:
        findings.append("LICENSED_SOURCE_COUNT_MISMATCH")
    if findings:
        raise EvidenceSFTV6Error(
            "RAG manifest/licensed_chunks authority mismatch: "
            + ",".join(sorted(set(findings)))
        )

    identity_payload = sorted(
        (
            {
                "chunk_id": chunk.chunk_id,
                "content_sha256": chunk.content_sha256,
            }
            for chunk in chunks
        ),
        key=lambda item: item["chunk_id"],
    )
    binding = {
        "status": "PASS_RAG_MANIFEST_LICENSED_CHUNKS_BOUND",
        "manifest_id": manifest.manifest_id,
        "licensed_chunk_count": len(chunks),
        "licensed_source_count": len(observed_sources),
        "namespace_chunk_counts": {
            namespace: namespace_counts[namespace]
            for namespace in evidence.DOMAINS
        },
        "source_chunk_counts": {
            source_id: source_counts[source_id]
            for source_id in sorted(observed_sources)
        },
        "licensed_chunk_identity_sha256": sha256_bytes(
            canonical_json(identity_payload).encode("utf-8")
        ),
    }
    return manifest_payload, binding


def _publish_lock_path(output_dir: Path) -> Path:
    return output_dir.parent / f".{output_dir.name}.publish.lock"


def _acquire_publish_lock(output_dir: Path) -> tuple[Path, str]:
    lock_path = _publish_lock_path(output_dir)
    token = uuid4().hex
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as exc:
        raise EvidenceSFTV6Error(
            f"concurrent output publication is already active: {output_dir}"
        ) from exc
    try:
        os.write(descriptor, token.encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return lock_path, token


def _release_publish_lock(
    lock_path: Path,
    token: str,
) -> None:
    try:
        if lock_path.read_text(encoding="ascii") == token:
            lock_path.unlink()
    except (FileNotFoundError, OSError, UnicodeError):
        pass


def _new_staging_dir(output_dir: Path) -> Path:
    path = tempfile.mkdtemp(
        prefix=f".{output_dir.name}.staging-",
        dir=output_dir.parent,
    )
    return Path(path).resolve()


def _balance_report(
    examples: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, str],
) -> dict[str, Any]:
    split_counts = Counter(str(item["split"]) for item in examples)
    decision_counts = Counter(
        (str(item["split"]), str(item["decision"]))
        for item in examples
    )
    task_counts = Counter(
        (str(item["split"]), str(item["task"]))
        for item in examples
    )
    family_decisions = Counter(
        (str(item["source_id"]), str(item["decision"]))
        for item in examples
    )
    included_source_ids = {
        source_id
        for source_id, split in assignments.items()
        if split in NONBLIND_SPLITS
    }
    findings: list[str] = []
    observed_counts = {
        split: split_counts[split]
        for split in NONBLIND_SPLITS
    }
    if observed_counts != EXPECTED_NONBLIND_SPLIT_COUNTS:
        findings.append("NONBLIND_SPLIT_COUNTS_MISMATCH")
    if any(split not in NONBLIND_SPLITS for split in split_counts):
        findings.append("NONBLIND_OUTPUT_CONTAINS_FORBIDDEN_SPLIT")
    for split, expected in EXPECTED_NONBLIND_SPLIT_COUNTS.items():
        if decision_counts[(split, "ANSWER")] != expected // 2:
            findings.append(f"{split.upper()}_ANSWER_IMBALANCE")
        if decision_counts[(split, "REFUSE")] != expected // 2:
            findings.append(f"{split.upper()}_REFUSE_IMBALANCE")
        if any(
            task_counts[(split, task)] == 0
            for task in TASKS
        ):
            findings.append(f"{split.upper()}_TASK_MISSING")
    imbalanced_families = sum(
        family_decisions[(source_id, "ANSWER")]
        != family_decisions[(source_id, "REFUSE")]
        for source_id in included_source_ids
    )
    if imbalanced_families:
        findings.append("FAMILY_DECISION_IMBALANCE")
    family_integrity = _family_integrity_report(
        examples,
        assignments,
    )
    if family_integrity["status"] != "PASS":
        findings.append("FAMILY_INTEGRITY_FAILED")
    return {
        "schema": NONBLIND_BALANCE_SCHEMA,
        "status": "PASS" if not findings else "FAIL",
        "findings": findings,
        "split_counts": observed_counts,
        "split_decision_counts": {
            split: {
                decision: decision_counts[(split, decision)]
                for decision in DECISIONS
            }
            for split in NONBLIND_SPLITS
        },
        "split_task_counts": {
            split: {
                task: task_counts[(split, task)]
                for task in TASKS
            }
            for split in NONBLIND_SPLITS
        },
        "included_family_count": len(included_source_ids),
        "imbalanced_family_count": imbalanced_families,
        "family_integrity": family_integrity,
    }


def _family_integrity_report(
    examples: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, str],
) -> dict[str, Any]:
    expected_sources = {
        source_id
        for source_id, split in assignments.items()
        if split in NONBLIND_SPLITS
    }
    row_counts: Counter[str] = Counter()
    decision_counts: Counter[tuple[str, str]] = Counter()
    observed_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    findings: list[str] = []
    for row in examples:
        source_id = str(row.get("source_id", ""))
        split = str(row.get("split", ""))
        example_id = row.get("example_id")
        if source_id not in expected_sources:
            findings.append("UNEXPECTED_INCLUDED_FAMILY")
            continue
        if split != assignments[source_id]:
            findings.append("EMBEDDED_SPLIT_ASSIGNMENT_MISMATCH")
        if not isinstance(example_id, str) or not example_id:
            findings.append("EXAMPLE_ID_INVALID")
        elif example_id in observed_ids:
            duplicate_ids.add(example_id)
        else:
            observed_ids.add(example_id)
        row_counts[source_id] += 1
        decision_counts[(source_id, str(row.get("decision", "")))] += 1
    if duplicate_ids:
        findings.append("DUPLICATE_EXAMPLE_ID")
    if set(row_counts) != expected_sources:
        findings.append("INCLUDED_FAMILY_SET_MISMATCH")
    per_family: list[dict[str, Any]] = []
    for source_id in sorted(expected_sources):
        count = row_counts[source_id]
        answers = decision_counts[(source_id, "ANSWER")]
        refusals = decision_counts[(source_id, "REFUSE")]
        if count != EXAMPLES_PER_FAMILY:
            findings.append("FAMILY_EXAMPLE_COUNT_MISMATCH")
        if answers != 25 or refusals != 25:
            findings.append("FAMILY_DECISION_COUNT_MISMATCH")
        if answers + refusals != count:
            findings.append("FAMILY_UNKNOWN_DECISION")
        per_family.append(
            {
                "source_id": source_id,
                "assigned_split": assignments[source_id],
                "example_count": count,
                "decision_counts": {
                    "ANSWER": answers,
                    "REFUSE": refusals,
                },
            }
        )
    return {
        "status": "PASS" if not findings else "FAIL",
        "findings": sorted(set(findings)),
        "expected_family_count": sum(
            EXPECTED_NONBLIND_FAMILY_SPLIT_COUNTS.values()
        ),
        "observed_family_count": len(row_counts),
        "unique_example_id_count": len(observed_ids),
        "duplicate_example_ids": sorted(duplicate_ids),
        "per_family": per_family,
    }


def _group_isolation_report(
    families: Sequence[SourceFamily],
    assignments: Mapping[str, str],
) -> dict[str, Any]:
    included_families = tuple(
        family
        for family in families
        if assignments[family.source_id] in NONBLIND_SPLITS
    )
    source_sets = {
        split: {
            family.source_id
            for family in included_families
            if assignments[family.source_id] == split
        }
        for split in NONBLIND_SPLITS
    }
    doi_sets = {
        split: {
            family.doi.lower()
            for family in included_families
            if assignments[family.source_id] == split
        }
        for split in NONBLIND_SPLITS
    }
    commitments = {
        split: sorted(
            _group_commitment(family)
            for family in included_families
            if assignments[family.source_id] == split
        )
        for split in NONBLIND_SPLITS
    }
    pairwise: list[dict[str, Any]] = []
    findings: list[str] = []
    for left_index, left in enumerate(NONBLIND_SPLITS):
        for right in NONBLIND_SPLITS[left_index + 1 :]:
            source_overlap = len(source_sets[left] & source_sets[right])
            doi_overlap = len(doi_sets[left] & doi_sets[right])
            commitment_overlap = len(
                set(commitments[left]) & set(commitments[right])
            )
            if source_overlap or doi_overlap or commitment_overlap:
                findings.append("GROUP_OVERLAP")
            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    "source_overlap_count": source_overlap,
                    "doi_overlap_count": doi_overlap,
                    "commitment_overlap_count": commitment_overlap,
                }
            )
    observed_family_counts = {
        split: len(source_sets[split])
        for split in NONBLIND_SPLITS
    }
    if observed_family_counts != EXPECTED_NONBLIND_FAMILY_SPLIT_COUNTS:
        findings.append("NONBLIND_FAMILY_SPLIT_COUNTS_MISMATCH")
    return {
        "schema": NONBLIND_GROUP_SCHEMA,
        "status": "PASS" if not findings else "FAIL",
        "findings": sorted(set(findings)),
        "isolation_unit": "licensed DOI/source family",
        "family_split_counts": observed_family_counts,
        "group_commitments": commitments,
        "pairwise": pairwise,
    }


def _assert_nonblind_shape(
    families: Sequence[SourceFamily],
    assignments: Mapping[str, str],
    examples: Sequence[Mapping[str, Any]],
) -> None:
    if len(families) != EXPECTED_FAMILY_COUNT:
        raise EvidenceSFTV6Error(
            f"nonblind v7 requires exactly {EXPECTED_FAMILY_COUNT} families"
        )
    full_family_counts = Counter(assignments.values())
    if {
        split: full_family_counts[split]
        for split in EXPECTED_FAMILY_SPLIT_COUNTS
    } != EXPECTED_FAMILY_SPLIT_COUNTS:
        raise EvidenceSFTV6Error("source-family split shape mismatch")
    split_counts = Counter(str(item["split"]) for item in examples)
    if any(split not in NONBLIND_SPLITS for split in split_counts):
        raise EvidenceSFTV6Error(
            "nonblind builder constructed a forbidden split"
        )
    if {
        split: split_counts[split]
        for split in NONBLIND_SPLITS
    } != EXPECTED_NONBLIND_SPLIT_COUNTS:
        raise EvidenceSFTV6Error("nonblind example split shape mismatch")
    if len(examples) != EXPECTED_NONBLIND_TOTAL:
        raise EvidenceSFTV6Error(
            f"nonblind dataset must contain exactly {EXPECTED_NONBLIND_TOTAL} examples"
        )
    family_integrity = _family_integrity_report(
        examples,
        assignments,
    )
    if family_integrity["status"] != "PASS":
        raise EvidenceSFTV6Error(
            "nonblind per-family integrity gate failed: "
            + ",".join(family_integrity["findings"])
        )


def _preblind_commitment(
    *,
    seed: str,
    snapshots: Mapping[str, StableFileSnapshot],
    rag_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": PREBLIND_COMMITMENT_SCHEMA,
        "status": "PREBLIND_COMMITTED_NONBLIND_ONLY",
        "builder_version": NONBLIND_BUILDER_VERSION,
        "core_builder_version": BUILDER_VERSION,
        "split_algorithm_version": SPLIT_ALGORITHM_VERSION,
        "seed": seed,
        "seed_sha256": sha256_bytes(seed.encode("utf-8")),
        "expected_blind_count": EXPECTED_BLIND_COUNT,
        "builder_code": {
            "nonblind_module_sha256": snapshots[
                "nonblind_module"
            ].sha256,
            "evidence_core_sha256": snapshots[
                "evidence_core"
            ].sha256,
        },
        "source_inputs": {
            "licensed_chunks_sha256": snapshots[
                "licensed_chunks"
            ].sha256,
            "rag_manifest_sha256": snapshots[
                "rag_manifest"
            ].sha256,
            "rag_manifest_id": rag_manifest.get("manifest_id"),
            "semantic_inventory_sha256": snapshots[
                "semantic_inventory"
            ].sha256,
            "semantic_records_sha256": snapshots[
                "semantic_records"
            ].sha256,
        },
    }
    return {
        **payload,
        "commitment_sha256": sha256_bytes(
            canonical_json(payload).encode("utf-8")
        ),
    }


def _assert_preblind_commitment_sanitized(
    commitment: Mapping[str, Any],
) -> None:
    allowed_top_level = {
        "schema",
        "status",
        "builder_version",
        "core_builder_version",
        "split_algorithm_version",
        "seed",
        "seed_sha256",
        "expected_blind_count",
        "builder_code",
        "source_inputs",
        "commitment_sha256",
    }
    if set(commitment) != allowed_top_level:
        raise EvidenceSFTV6Error(
            "preblind commitment contains an undeclared field"
        )
    serialized = canonical_json(commitment)
    forbidden_fragments = (
        "blind_test",
        "sealed.v",
        "blind_path",
        "blind_sha256",
        "blind_bytes",
        "blind_content",
    )
    if any(fragment in serialized for fragment in forbidden_fragments):
        raise EvidenceSFTV6Error(
            "preblind commitment disclosed a forbidden blind artifact detail"
        )


def _verify_file_receipt(
    root: Path,
    receipt: Mapping[str, Any],
    *,
    expected_name: str,
    expected_count: int | None = None,
) -> bytes:
    if receipt.get("path") != expected_name:
        raise EvidenceSFTV6Error(
            f"staging receipt path mismatch: {expected_name}"
        )
    path = root / expected_name
    if not path.is_file() or path.is_symlink():
        raise EvidenceSFTV6Error(
            f"staging artifact is not a regular file: {expected_name}"
        )
    payload = path.read_bytes()
    if (
        receipt.get("bytes") != len(payload)
        or receipt.get("sha256") != sha256_bytes(payload)
    ):
        raise EvidenceSFTV6Error(
            f"staging artifact receipt mismatch: {expected_name}"
        )
    if expected_count is not None:
        count = sum(
            1
            for line in payload.splitlines()
            if line.strip()
        )
        if receipt.get("count") != expected_count or count != expected_count:
            raise EvidenceSFTV6Error(
                f"staging JSONL count mismatch: {expected_name}"
            )
    return payload


def _parse_json_artifact(
    payload: bytes,
    *,
    label: str,
) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceSFTV6Error(
            f"staging artifact is invalid JSON: {label}"
        ) from exc
    if not isinstance(value, dict):
        raise EvidenceSFTV6Error(
            f"staging artifact must be a JSON object: {label}"
        )
    return value


def _parse_jsonl_artifact(
    payload: bytes,
    *,
    label: str,
) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceSFTV6Error(
            f"staging artifact is invalid UTF-8: {label}"
        ) from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvidenceSFTV6Error(
                f"{label}:{line_number}: invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise EvidenceSFTV6Error(
                f"{label}:{line_number}: object required"
            )
        rows.append(value)
    return rows


def _self_validate_staging(
    root: Path,
    *,
    expected_manifest: Mapping[str, Any],
    expected_report: Mapping[str, Any],
    expected_commitment: Mapping[str, Any],
    assignments: Mapping[str, str],
) -> dict[str, Any]:
    observed: set[str] = set()
    with os.scandir(root) as entries:
        for entry in entries:
            if (
                not entry.is_file(follow_symlinks=False)
                or entry.is_symlink()
            ):
                raise EvidenceSFTV6Error(
                    "staging contains a non-regular artifact"
                )
            observed.add(entry.name)
    if observed != set(OUTPUT_FILENAMES):
        raise EvidenceSFTV6Error(
            "staging fixed ten-file inventory mismatch"
        )

    manifest_path = root / "manifest.nonblind.v7.json"
    manifest_payload = manifest_path.read_bytes()
    manifest = _parse_json_artifact(
        manifest_payload,
        label=manifest_path.name,
    )
    if manifest != expected_manifest:
        raise EvidenceSFTV6Error(
            "staging manifest differs from in-memory manifest"
        )
    if set(manifest.get("splits", {})) != set(NONBLIND_SPLITS):
        raise EvidenceSFTV6Error(
            "staging manifest split whitelist mismatch"
        )
    expected_artifact_roles = {
        "balance_audit",
        "group_isolation_audit",
        "content_leakage_audit",
        "semantic_inventory_audit",
        "preblind_commitment",
        "build_report",
    }
    if set(manifest.get("artifacts", {})) != expected_artifact_roles:
        raise EvidenceSFTV6Error(
            "staging manifest artifact whitelist mismatch"
        )

    rows: list[dict[str, Any]] = []
    for split in NONBLIND_SPLITS:
        name = f"{split}.jsonl"
        payload = _verify_file_receipt(
            root,
            manifest["splits"][split],
            expected_name=name,
            expected_count=EXPECTED_NONBLIND_SPLIT_COUNTS[split],
        )
        split_rows = _parse_jsonl_artifact(
            payload,
            label=name,
        )
        if any(row.get("split") != split for row in split_rows):
            raise EvidenceSFTV6Error(
                f"staging row embedded split mismatch: {split}"
            )
        rows.extend(split_rows)

    artifact_names = {
        "balance_audit": "balance_audit.nonblind.v7.json",
        "group_isolation_audit": (
            "group_isolation_audit.nonblind.v7.json"
        ),
        "content_leakage_audit": (
            "content_leakage_audit.nonblind.v7.json"
        ),
        "semantic_inventory_audit": (
            "semantic_inventory_audit.v7.json"
        ),
        "preblind_commitment": "preblind_commitment.v7.json",
        "build_report": "build_report.nonblind.v7.json",
    }
    artifact_payloads: dict[str, dict[str, Any]] = {}
    for role, name in artifact_names.items():
        payload = _verify_file_receipt(
            root,
            manifest["artifacts"][role],
            expected_name=name,
        )
        artifact_payloads[role] = _parse_json_artifact(
            payload,
            label=name,
        )

    if artifact_payloads["build_report"] != expected_report:
        raise EvidenceSFTV6Error(
            "staging build report differs from in-memory report"
        )
    if artifact_payloads["preblind_commitment"] != expected_commitment:
        raise EvidenceSFTV6Error(
            "staging commitment differs from in-memory commitment"
        )
    _assert_preblind_commitment_sanitized(
        artifact_payloads["preblind_commitment"]
    )
    commitment_copy = dict(
        artifact_payloads["preblind_commitment"]
    )
    observed_commitment_hash = commitment_copy.pop(
        "commitment_sha256",
        None,
    )
    if observed_commitment_hash != sha256_bytes(
        canonical_json(commitment_copy).encode("utf-8")
    ):
        raise EvidenceSFTV6Error(
            "staging preblind commitment hash mismatch"
        )
    if any(
        artifact_payloads[role].get("status") != "PASS"
        for role in (
            "balance_audit",
            "group_isolation_audit",
            "content_leakage_audit",
            "semantic_inventory_audit",
        )
    ):
        raise EvidenceSFTV6Error(
            "staging contains a failed nonblind audit"
        )
    family_integrity = _family_integrity_report(
        rows,
        assignments,
    )
    if family_integrity["status"] != "PASS":
        raise EvidenceSFTV6Error(
            "staging per-family integrity verification failed"
        )
    if (
        len(rows) != EXPECTED_NONBLIND_TOTAL
        or manifest.get("counts", {}).get("splits")
        != EXPECTED_NONBLIND_SPLIT_COUNTS
        or expected_report.get("counts", {}).get("splits")
        != EXPECTED_NONBLIND_SPLIT_COUNTS
    ):
        raise EvidenceSFTV6Error(
            "staging aggregate count verification failed"
        )
    return {
        "status": "PASS_FIXED_TEN_FILE_STAGING_VERIFIED",
        "file_count": len(observed),
        "example_count": len(rows),
        "manifest_sha256": sha256_bytes(manifest_payload),
        "family_integrity": family_integrity["status"],
    }


def _remove_staging(root: Path | None) -> None:
    if root is None or not root.exists():
        return
    shutil.rmtree(root)
    if root.exists():
        raise EvidenceSFTV6Error(
            f"failed to clean staging directory: {root}"
        )


def build_nonblind_dataset_v7(
    *,
    chunks_path: Path,
    rag_manifest_path: Path,
    semantic_inventory_path: Path,
    output_dir: Path,
    seed: str = "icmat-evidence-v6-finals-20260730",
) -> dict[str, Any]:
    output = output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise EvidenceSFTV6Error(
            f"output directory already exists: {output}"
        )
    lock_path, lock_token = _acquire_publish_lock(output)
    staging: Path | None = None
    try:
        if output.exists():
            raise EvidenceSFTV6Error(
                f"output directory was concurrently occupied: {output}"
            )
        snapshots = _capture_snapshot_set(
            chunks_path=chunks_path,
            rag_manifest_path=rag_manifest_path,
            semantic_inventory_path=semantic_inventory_path,
        )
        rag_manifest, rag_binding = _validate_rag_binding(
            manifest_snapshot=snapshots["rag_manifest"],
            chunks_snapshot=snapshots["licensed_chunks"],
        )
        families = load_licensed_families(
            snapshots["licensed_chunks"].path
        )
        semantic_inventory, semantic_inventory_audit = (
            load_semantic_inventory(
                snapshots["semantic_inventory"].path,
                families,
            )
        )
        families = evidence.augment_families_with_semantic_candidates(
            families,
            semantic_inventory,
        )
        _verify_snapshot_set(snapshots, phase="after_parse")
        if (
            semantic_inventory_audit.get(
                "semantic_inventory_sha256"
            )
            != snapshots["semantic_inventory"].sha256
            or semantic_inventory_audit.get(
                "semantic_records_sha256"
            )
            != snapshots["semantic_records"].sha256
        ):
            raise EvidenceSFTV6Error(
                "semantic inventory audit does not match stable snapshots"
            )

        assignments = assign_family_splits(
            families,
            seed=seed,
        )
        examples = build_examples(
            families,
            assignments,
            semantic_inventory,
            seed=seed,
            examples_per_family=EXAMPLES_PER_FAMILY,
            included_splits=NONBLIND_SPLITS,
        )
        _assert_nonblind_shape(
            families,
            assignments,
            examples,
        )
        balance = _balance_report(examples, assignments)
        group_audit = _group_isolation_report(
            families,
            assignments,
        )
        leakage = {
            **_content_leakage_report(
                examples,
                splits=NONBLIND_SPLITS,
            ),
            "schema": NONBLIND_LEAKAGE_SCHEMA,
            "audited_splits": list(NONBLIND_SPLITS),
        }
        audits = (
            balance,
            group_audit,
            leakage,
            semantic_inventory_audit,
        )
        if any(audit["status"] != "PASS" for audit in audits):
            raise EvidenceSFTV6Error(
                "nonblind dataset audit failed before any artifact was written"
            )
        _verify_snapshot_set(snapshots, phase="before_write")

        commitment = _preblind_commitment(
            seed=seed,
            snapshots=snapshots,
            rag_manifest=rag_manifest,
        )
        _assert_preblind_commitment_sanitized(commitment)
        report = {
            "schema": NONBLIND_REPORT_SCHEMA,
            "status": "PASS_NONBLIND_DATASET_PREBLIND_COMMITTED",
            "builder_version": NONBLIND_BUILDER_VERSION,
            "counts": {
                "examples": EXPECTED_NONBLIND_TOTAL,
                "families": sum(
                    EXPECTED_NONBLIND_FAMILY_SPLIT_COUNTS.values()
                ),
                "examples_per_family": EXAMPLES_PER_FAMILY,
                "splits": dict(EXPECTED_NONBLIND_SPLIT_COUNTS),
            },
            "audits": {
                "balance": balance["status"],
                "family_integrity": balance[
                    "family_integrity"
                ]["status"],
                "group_isolation": group_audit["status"],
                "content_leakage": leakage["status"],
                "semantic_inventory": semantic_inventory_audit[
                    "status"
                ],
                "rag_authority_binding": rag_binding["status"],
            },
            "family_integrity": balance["family_integrity"],
            "claims": {
                "nonblind_only": True,
                "training_authorized_splits": list(TRAINING_SPLITS),
                "calibration_for_training": False,
                "production_connected": False,
                "x5_deployed": False,
            },
        }

        staging = _new_staging_dir(output)
        split_receipts: dict[str, dict[str, Any]] = {}
        for split in NONBLIND_SPLITS:
            split_receipts[split] = _write_jsonl(
                staging / f"{split}.jsonl",
                (
                    item
                    for item in examples
                    if item["split"] == split
                ),
            )
        artifact_receipts = {
            "balance_audit": _write_json(
                staging / "balance_audit.nonblind.v7.json",
                balance,
            ),
            "group_isolation_audit": _write_json(
                staging
                / "group_isolation_audit.nonblind.v7.json",
                group_audit,
            ),
            "content_leakage_audit": _write_json(
                staging
                / "content_leakage_audit.nonblind.v7.json",
                leakage,
            ),
            "semantic_inventory_audit": _write_json(
                staging / "semantic_inventory_audit.v7.json",
                semantic_inventory_audit,
            ),
            "preblind_commitment": _write_json(
                staging / "preblind_commitment.v7.json",
                commitment,
            ),
        }
        report_receipt = _write_json(
            staging / "build_report.nonblind.v7.json",
            report,
        )
        manifest = {
            "schema": NONBLIND_MANIFEST_SCHEMA,
            "dataset_schema": DATASET_SCHEMA,
            "builder_version": NONBLIND_BUILDER_VERSION,
            "core_builder_version": BUILDER_VERSION,
            "status": (
                "NONBLIND_DATASET_BUILT_PREBLIND_COMMITTED"
            ),
            "ground_truth_policy": (
                "deterministic pointer labels from licensed evidence; "
                "no API or teacher output is ground truth"
            ),
            "selection_policy": (
                "researcher_explicit_domain_and_task"
            ),
            "source_isolation_unit": "DOI/source_family",
            "splits": split_receipts,
            "artifacts": {
                **artifact_receipts,
                "build_report": report_receipt,
            },
            "source_inputs": {
                "licensed_chunks": {
                    "path": snapshots[
                        "licensed_chunks"
                    ].path.as_posix(),
                    "sha256": snapshots[
                        "licensed_chunks"
                    ].sha256,
                },
                "rag_manifest": {
                    "path": snapshots[
                        "rag_manifest"
                    ].path.as_posix(),
                    "sha256": snapshots[
                        "rag_manifest"
                    ].sha256,
                    "manifest_id": rag_manifest.get(
                        "manifest_id"
                    ),
                },
                "semantic_inventory": {
                    "path": snapshots[
                        "semantic_inventory"
                    ].path.as_posix(),
                    "sha256": snapshots[
                        "semantic_inventory"
                    ].sha256,
                    "schema": ACCEPTED_INVENTORY_SCHEMA,
                    "producer_inventory_sha256": (
                        semantic_inventory_audit[
                            "producer_inventory_sha256"
                        ]
                    ),
                    "records_sha256": snapshots[
                        "semantic_records"
                    ].sha256,
                    "record_schema": SEMANTIC_QUERY_SCHEMA,
                    "record_count": semantic_inventory_audit[
                        "record_count"
                    ],
                    "accepted_count": semantic_inventory_audit[
                        "accepted_count"
                    ],
                },
            },
            "builder": {
                "nonblind_module": {
                    "path": snapshots[
                        "nonblind_module"
                    ].path.as_posix(),
                    "sha256": snapshots[
                        "nonblind_module"
                    ].sha256,
                },
                "evidence_core": {
                    "path": snapshots[
                        "evidence_core"
                    ].path.as_posix(),
                    "sha256": snapshots[
                        "evidence_core"
                    ].sha256,
                },
                "split_algorithm_version": (
                    SPLIT_ALGORITHM_VERSION
                ),
                "seed": seed,
            },
            "counts": {
                "examples": EXPECTED_NONBLIND_TOTAL,
                "families": sum(
                    EXPECTED_NONBLIND_FAMILY_SPLIT_COUNTS.values()
                ),
                "examples_per_family": EXAMPLES_PER_FAMILY,
                "splits": dict(EXPECTED_NONBLIND_SPLIT_COUNTS),
            },
            "pointer_contract": {
                "field_order": list(POINTER_FIELDS),
                "answer_span_pattern": "E#.S#",
                "refusal_span_id": None,
            },
            "compiler_input_contract": {
                "compiler_version": COMPILER_VERSION,
                "prompt_schema": COMPILER_PROMPT_SCHEMA,
                "compiler_prompt_keys": sorted(
                    COMPILER_PROMPT_FIELDS
                ),
                "compiler_evidence_keys": sorted(
                    COMPILER_EVIDENCE_FIELDS
                ),
                "compiler_sentence_keys": sorted(
                    COMPILER_SENTENCE_FIELDS
                ),
                "target_free": True,
                "user_text_reverse_parsing_required": False,
            },
            "external_answer_contract": {
                "schema": EXTERNAL_ANSWER_SCHEMA,
                "field_order": list(EXTERNAL_ANSWER_FIELDS),
                "generated_by": (
                    "later_deterministic_evidence_compiler"
                ),
                "implemented_by_this_builder": False,
            },
            "training_boundary": {
                "allowed_splits": list(TRAINING_SPLITS),
                "calibration_content_for_training": False,
            },
            "claims": dict(report["claims"]),
        }
        manifest_receipt = _write_json(
            staging / "manifest.nonblind.v7.json",
            manifest,
        )
        staging_verification = _self_validate_staging(
            staging,
            expected_manifest=manifest,
            expected_report=report,
            expected_commitment=commitment,
            assignments=assignments,
        )
        if (
            staging_verification["manifest_sha256"]
            != manifest_receipt["sha256"]
        ):
            raise EvidenceSFTV6Error(
                "staging manifest receipt changed after write"
            )
        _verify_snapshot_set(
            snapshots,
            phase="before_publish",
        )
        if output.exists():
            raise EvidenceSFTV6Error(
                f"output directory was concurrently occupied: {output}"
            )
        result = {
            "status": manifest["status"],
            "output_dir": output.as_posix(),
            "manifest_sha256": manifest_receipt["sha256"],
            "split_counts": dict(
                EXPECTED_NONBLIND_SPLIT_COUNTS
            ),
            "example_count": EXPECTED_NONBLIND_TOTAL,
            "preblind_commitment_sha256": commitment[
                "commitment_sha256"
            ],
            "staging_verification": staging_verification,
            "output_files": list(OUTPUT_FILENAMES),
        }
        try:
            os.rename(staging, output)
        except FileExistsError as exc:
            raise EvidenceSFTV6Error(
                f"output directory was concurrently occupied: {output}"
            ) from exc
        except OSError as exc:
            if output.exists():
                raise EvidenceSFTV6Error(
                    f"output directory publication collision: {output}"
                ) from exc
            raise
        staging = None
        return result
    finally:
        try:
            _remove_staging(staging)
        finally:
            _release_publish_lock(lock_path, lock_token)
