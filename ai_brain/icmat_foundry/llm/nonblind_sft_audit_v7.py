from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import stat
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from icmat_foundry.llm import evidence_sft_v6 as evidence
from icmat_foundry.llm import nonblind_sft_v7 as builder
from icmat_foundry.llm import semantic_queries_v7 as semantic_queries
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
    EXTERNAL_ANSWER_FIELDS,
    EXTERNAL_ANSWER_SCHEMA,
    NONBLIND_SPLITS,
    POINTER_FIELDS,
    SEMANTIC_QUERY_SCHEMA,
    TASKS,
    TRAINING_SPLITS,
    EvidenceSFTV6Error,
    canonical_json,
)
from icmat_foundry.llm.nonblind_sft_v7 import (
    EXPECTED_BLIND_COUNT,
    EXPECTED_NONBLIND_FAMILY_SPLIT_COUNTS,
    EXPECTED_NONBLIND_SPLIT_COUNTS,
    EXPECTED_NONBLIND_TOTAL,
    NONBLIND_BALANCE_SCHEMA,
    NONBLIND_BUILDER_VERSION,
    NONBLIND_GROUP_SCHEMA,
    NONBLIND_LEAKAGE_SCHEMA,
    NONBLIND_MANIFEST_SCHEMA,
    NONBLIND_REPORT_SCHEMA,
    PREBLIND_COMMITMENT_SCHEMA,
    SPLIT_ALGORITHM_VERSION,
)
from icmat_foundry.rag.contracts import (
    ChunkV1,
    ContractError,
    RegistryManifestV2,
)

AUDIT_SCHEMA = "icmat_evidence_pointer_nonblind_independent_audit.v7"
AUDIT_VERSION = "icmat-evidence-nonblind-independent-audit-v7.1.0"
AUDIT_FILENAME = "independent_audit.nonblind.v7.json"
AUDIT_PASS_STATUS = "PASS_NONBLIND_V7_INDEPENDENT_AUDIT"
COMPARE_PASS_STATUS = "PASS_NONBLIND_V7_DOUBLE_BUILD_BYTE_IDENTICAL"
ERROR_STATUS = "FAILED_NO_NONBLIND_V7_AUDIT"

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_READ_BLOCK_BYTES = 1024 * 1024
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_JSONL_BYTES = 64 * 1024 * 1024
_NEAR_DUPLICATE_THRESHOLD = 0.90
_PROTECTED_PATH_TOKENS = frozenset({"blind", "calibration", "sealed"})
_TARGET_MARKER_FIELDS = frozenset(
    {
        "assistant_target",
        "decision",
        "expected_answer",
        "expected_pointer",
        "gold",
        "label",
        "raw_pointer",
        "target",
        "target_span_id",
        "verdict",
    }
)
_SEMANTIC_INVENTORY_FIELDS = frozenset(
    {
        "schema",
        "status",
        "request_manifest_sha256",
        "record_count",
        "accepted_count",
        "rejected_or_fixture_count",
        "accepted_records",
        "generator_provenance",
        "nli_provenance",
        "quality_claim_allowed",
        "training_authorized",
        "generated_text_is_ground_truth",
        "licensed_original_is_ground_truth",
        "sealed_blind_access",
        "inventory_sha256",
    }
)
_SEMANTIC_INVENTORY_V17_FIELDS = frozenset(
    {
        *_SEMANTIC_INVENTORY_FIELDS,
        "smoke_gate_sha256",
        "staging_contract_sha256",
        "source_coverage",
        "source_coverage_passed",
    }
)
_SEMANTIC_RECORD_FIELDS = frozenset(
    {
        "schema",
        "request_id",
        "request_sha256",
        "source_id",
        "source_record_sha256",
        "namespace",
        "source_title",
        "source_uri",
        "license_id",
        "chunk_ids",
        "locators",
        "original_sentence",
        "original_sha256",
        "ground_truth_boundary",
        "paraphrase",
        "contradiction",
        "mutation_type",
        "mutation",
        "generator_provenance",
        "nli_provenance",
        "audits",
        "acceptance",
        "record_id",
        "record_sha256",
    }
)
_SEMANTIC_RECORD_V17_FIELDS = frozenset(
    {
        *_SEMANTIC_RECORD_FIELDS,
        "source_manifest_authority",
        "source_asset_sha256",
        "source_asset_uri",
        "generation_response_trace",
        "generation_response_tree_sha256",
    }
)
_SEMANTIC_MUTATION_TYPES = frozenset(
    {"polarity_flip", "numeric_change", "entity_swap"}
)
_SEMANTIC_ACCEPTANCE = {
    "accepted": True,
    "formal_audit_backends": True,
    "structural_and_nli_gate_passed": True,
    "status": "ACCEPTED_INDEPENDENT_LOCAL_NLI_PASS",
    "reasons": [],
    "quality_claim_allowed": True,
    "training_eligible": True,
}

# This literal inventory is intentionally independent of manifest contents.
# The auditor never enumerates a dataset directory.
_ROLE_FILENAMES: dict[str, str] = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "calibration": "calibration.jsonl",
    "balance_audit": "balance_audit.nonblind.v7.json",
    "group_isolation_audit": "group_isolation_audit.nonblind.v7.json",
    "content_leakage_audit": "content_leakage_audit.nonblind.v7.json",
    "semantic_inventory_audit": "semantic_inventory_audit.v7.json",
    "preblind_commitment": "preblind_commitment.v7.json",
    "build_report": "build_report.nonblind.v7.json",
    "manifest": "manifest.nonblind.v7.json",
}
_SPLIT_ROLES = tuple(NONBLIND_SPLITS)
_ARTIFACT_ROLES = (
    "balance_audit",
    "group_isolation_audit",
    "content_leakage_audit",
    "semantic_inventory_audit",
    "preblind_commitment",
    "build_report",
)
_EXPECTED_MANIFEST_KEYS = {
    "schema",
    "dataset_schema",
    "builder_version",
    "core_builder_version",
    "status",
    "ground_truth_policy",
    "selection_policy",
    "source_isolation_unit",
    "splits",
    "artifacts",
    "source_inputs",
    "builder",
    "counts",
    "pointer_contract",
    "compiler_input_contract",
    "external_answer_contract",
    "training_boundary",
    "claims",
}
_EXPECTED_CLAIMS = {
    "nonblind_only": True,
    "training_authorized_splits": list(TRAINING_SPLITS),
    "calibration_for_training": False,
    "production_connected": False,
    "x5_deployed": False,
}


class NonblindSFTAuditV7Error(ValueError):
    pass


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    payload: bytes
    sha256: str
    bytes: int
    identity: tuple[int, int, int, int]


@dataclass(frozen=True)
class SourceState:
    licensed_chunks: FileSnapshot
    rag_manifest: FileSnapshot
    semantic_inventory: FileSnapshot
    semantic_records: FileSnapshot
    nonblind_builder: FileSnapshot
    evidence_core: FileSnapshot
    rag_manifest_id: str
    rag_authority_binding: Mapping[str, Any]
    seed: str


@dataclass(frozen=True)
class AuthorityInputs:
    licensed_chunks: FileSnapshot
    rag_manifest: FileSnapshot
    semantic_inventory: FileSnapshot
    semantic_records: FileSnapshot


@dataclass(frozen=True)
class DatasetState:
    root: Path
    files: Mapping[str, FileSnapshot]
    manifest: Mapping[str, Any]
    artifacts: Mapping[str, Mapping[str, Any]]
    rows: tuple[Mapping[str, Any], ...]
    sources: SourceState


@dataclass(frozen=True)
class OutputPlan:
    final_path: Path
    new_directory: Path | None
    anchor_parent: Path
    anchor_parent_identity: tuple[int, int]
    dataset_roots: tuple[Path, ...]


@dataclass
class StableDirectoryHandle:
    path: Path
    identity: tuple[int, int]
    windows_handle: int | None

    def current_path(self, *, label: str) -> Path:
        if self.windows_handle is None:
            current, identity = _snapshot_directory(self.path, label=label)
        else:
            current = _windows_path_from_handle(
                self.windows_handle,
                label=label,
            )
            current, identity = _snapshot_directory(current, label=label)
            if (
                _windows_identity_from_handle(
                    self.windows_handle,
                    label=label,
                )
                != self.identity
            ):
                raise NonblindSFTAuditV7Error(
                    f"{label} stable handle identity changed"
                )
            confirmed = _windows_path_from_handle(
                self.windows_handle,
                label=label,
            )
            if confirmed != current:
                raise NonblindSFTAuditV7Error(
                    f"{label} moved while its stable path was recovered"
                )
        if identity != self.identity:
            raise NonblindSFTAuditV7Error(
                f"{label} stable directory identity mismatch"
            )
        return current

    def close(self) -> None:
        if self.windows_handle is None:
            return
        _windows_close_handle(self.windows_handle)
        self.windows_handle = None


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
    )


def _directory_identity(value: os.stat_result) -> tuple[int, int]:
    return (int(value.st_dev), int(value.st_ino))


def _is_reparse_point(value: os.stat_result) -> bool:
    attributes = int(getattr(value, "st_file_attributes", 0))
    marker = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return bool(marker and attributes & marker)


def _absolute_lexical(path: Path) -> Path:
    lexical = Path(path)
    if not lexical.is_absolute():
        lexical = Path.cwd() / lexical
    return lexical


def _assert_lexically_unprotected(value: str, *, label: str) -> None:
    if not value or "\x00" in value:
        raise NonblindSFTAuditV7Error(f"{label} path is invalid")
    tokens = frozenset(re.findall(r"[a-z0-9]+", value.casefold()))
    blocked = sorted(tokens & _PROTECTED_PATH_TOKENS)
    if blocked:
        raise NonblindSFTAuditV7Error(
            f"{label} path contains a protected lexical marker: "
            + ",".join(blocked)
        )


def _assert_canonical_lexical_path(path: Path, *, label: str) -> None:
    if any(part in {".", ".."} for part in path.parts):
        raise NonblindSFTAuditV7Error(
            f"{label} path must not contain dot traversal components"
        )


def _snapshot_authority_file(
    path: Path,
    *,
    expected_basename: str,
    label: str,
    maximum_bytes: int,
) -> FileSnapshot:
    raw = os.fspath(path)
    _assert_lexically_unprotected(raw, label=label)
    lexical = Path(raw)
    if lexical.name != expected_basename:
        raise NonblindSFTAuditV7Error(
            f"{label} basename must be {expected_basename}"
        )
    _assert_canonical_lexical_path(lexical, label=label)
    return _snapshot_regular_file(
        lexical,
        label=label,
        maximum_bytes=maximum_bytes,
    )


def _snapshot_authority_inputs(
    *,
    licensed_chunks: Path,
    rag_manifest: Path,
    semantic_inventory: Path,
) -> AuthorityInputs:
    licensed = _snapshot_authority_file(
        licensed_chunks,
        expected_basename="licensed_chunks.v1.jsonl",
        label="authority licensed chunks",
        maximum_bytes=_MAX_JSONL_BYTES,
    )
    rag = _snapshot_authority_file(
        rag_manifest,
        expected_basename="manifest.v2.json",
        label="authority RAG manifest",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    semantic = _snapshot_authority_file(
        semantic_inventory,
        expected_basename="accepted_inventory.v7.json",
        label="authority semantic inventory",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    records_path = semantic.path.with_name("records.v7.jsonl")
    records = _snapshot_authority_file(
        records_path,
        expected_basename="records.v7.jsonl",
        label="authority semantic records",
        maximum_bytes=_MAX_JSONL_BYTES,
    )
    return AuthorityInputs(
        licensed_chunks=licensed,
        rag_manifest=rag,
        semantic_inventory=semantic,
        semantic_records=records,
    )


def _assert_no_link_components(
    path: Path,
    *,
    label: str,
    allow_missing_leaf: bool = False,
) -> None:
    lexical = _absolute_lexical(path)
    parts = lexical.parts
    if not parts:
        raise NonblindSFTAuditV7Error(f"{label} path is empty")
    current = Path(parts[0])
    for index, part in enumerate(parts[1:], 1):
        current /= part
        try:
            current_stat = current.lstat()
        except FileNotFoundError:
            if allow_missing_leaf and index == len(parts) - 1:
                return
            raise NonblindSFTAuditV7Error(f"{label} path is missing") from None
        if stat.S_ISLNK(current_stat.st_mode) or _is_reparse_point(
            current_stat
        ):
            raise NonblindSFTAuditV7Error(
                f"{label} must not contain a symbolic link or reparse point"
            )


def _snapshot_regular_file(
    path: Path,
    *,
    label: str,
    maximum_bytes: int = _MAX_JSON_BYTES,
) -> FileSnapshot:
    lexical = _absolute_lexical(path)
    _assert_no_link_components(lexical, label=label)
    try:
        lexical_stat = lexical.lstat()
    except FileNotFoundError as exc:
        raise NonblindSFTAuditV7Error(f"{label} is missing") from exc
    if (
        not stat.S_ISREG(lexical_stat.st_mode)
        or stat.S_ISLNK(lexical_stat.st_mode)
        or _is_reparse_point(lexical_stat)
    ):
        raise NonblindSFTAuditV7Error(f"{label} must be a regular file")
    if lexical_stat.st_size > maximum_bytes:
        raise NonblindSFTAuditV7Error(f"{label} exceeds the size limit")

    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(os.fspath(lexical), flags)
    except OSError as exc:
        raise NonblindSFTAuditV7Error(f"{label} cannot be opened") from exc
    blocks: list[bytes] = []
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or _is_reparse_point(before)
            or before.st_size > maximum_bytes
        ):
            raise NonblindSFTAuditV7Error(
                f"{label} must remain a bounded regular file"
            )
        while True:
            block = os.read(descriptor, _READ_BLOCK_BYTES)
            if not block:
                break
            blocks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    post = lexical.lstat()
    if (
        _stat_identity(before) != _stat_identity(after)
        or _stat_identity(after) != _stat_identity(post)
        or stat.S_ISLNK(post.st_mode)
        or _is_reparse_point(post)
    ):
        raise NonblindSFTAuditV7Error(
            f"{label} changed while it was inspected"
        )
    payload = b"".join(blocks)
    if len(payload) != int(after.st_size):
        raise NonblindSFTAuditV7Error(f"{label} byte count is unstable")
    return FileSnapshot(
        path=lexical.resolve(strict=True),
        payload=payload,
        sha256=_sha256_bytes(payload),
        bytes=len(payload),
        identity=_stat_identity(after),
    )


def _strict_json(payload: bytes, *, label: str) -> Any:
    def reject_duplicates(
        pairs: Sequence[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise NonblindSFTAuditV7Error(
                    f"{label} contains a duplicate JSON key"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise NonblindSFTAuditV7Error(
            f"{label} contains a non-finite JSON number"
        )

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except UnicodeDecodeError as exc:
        raise NonblindSFTAuditV7Error(
            f"{label} is not UTF-8 JSON"
        ) from exc
    except json.JSONDecodeError as exc:
        raise NonblindSFTAuditV7Error(
            f"{label} is invalid JSON"
        ) from exc


def _strict_json_object(
    payload: bytes,
    *,
    label: str,
) -> dict[str, Any]:
    value = _strict_json(payload, label=label)
    if not isinstance(value, dict):
        raise NonblindSFTAuditV7Error(f"{label} must be a JSON object")
    return value


def _validate_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _HEX_SHA256.fullmatch(value):
        raise NonblindSFTAuditV7Error(
            f"{label} is not a lowercase SHA-256"
        )
    return value


def _validate_receipt(
    receipt: Any,
    *,
    snapshot: FileSnapshot,
    expected_path: str,
    expected_count: int | None,
    label: str,
) -> None:
    if not isinstance(receipt, dict):
        raise NonblindSFTAuditV7Error(f"{label} receipt is missing")
    expected_keys = {"path", "sha256", "bytes"}
    if expected_count is not None:
        expected_keys.add("count")
    if set(receipt) != expected_keys:
        raise NonblindSFTAuditV7Error(
            f"{label} receipt keys do not match the fixed v7 contract"
        )
    if receipt.get("path") != expected_path:
        raise NonblindSFTAuditV7Error(
            f"{label} receipt path is not the fixed whitelist path"
        )
    if (
        _validate_sha256(
            receipt.get("sha256"),
            label=f"{label} receipt SHA-256",
        )
        != snapshot.sha256
    ):
        raise NonblindSFTAuditV7Error(f"{label} receipt hash mismatch")
    if (
        not _is_integer(receipt.get("bytes"))
        or int(receipt["bytes"]) != snapshot.bytes
    ):
        raise NonblindSFTAuditV7Error(f"{label} receipt byte mismatch")
    if expected_count is not None and (
        not _is_integer(receipt.get("count"))
        or int(receipt["count"]) != expected_count
    ):
        raise NonblindSFTAuditV7Error(f"{label} receipt count mismatch")


def _dataset_root(path: Path, *, label: str) -> Path:
    lexical = _absolute_lexical(path)
    _assert_no_link_components(lexical, label=label)
    try:
        root = lexical.resolve(strict=True)
        root_stat = lexical.lstat()
    except OSError as exc:
        raise NonblindSFTAuditV7Error(
            f"{label} directory is missing"
        ) from exc
    if (
        not stat.S_ISDIR(root_stat.st_mode)
        or stat.S_ISLNK(root_stat.st_mode)
        or _is_reparse_point(root_stat)
    ):
        raise NonblindSFTAuditV7Error(
            f"{label} must be a regular directory"
        )
    return root


def _snapshot_fixed_dataset(
    path: Path,
    *,
    label: str,
) -> tuple[Path, dict[str, FileSnapshot], dict[str, Any]]:
    root = _dataset_root(path, label=label)
    if tuple(_ROLE_FILENAMES.values()) != tuple(builder.OUTPUT_FILENAMES):
        raise NonblindSFTAuditV7Error(
            "current builder whitelist differs from the independent auditor"
        )
    files: dict[str, FileSnapshot] = {}
    for role, filename in _ROLE_FILENAMES.items():
        maximum = (
            _MAX_JSONL_BYTES if role in _SPLIT_ROLES else _MAX_JSON_BYTES
        )
        files[role] = _snapshot_regular_file(
            root / filename,
            label=f"{label} {role}",
            maximum_bytes=maximum,
        )
    manifest = _strict_json_object(
        files["manifest"].payload,
        label=f"{label} manifest",
    )
    return root, files, manifest


def _validate_declared_path(
    value: Any,
    *,
    authority: FileSnapshot,
    expected_basename: str,
    label: str,
) -> None:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
    ):
        raise NonblindSFTAuditV7Error(f"{label} path is invalid")
    _assert_lexically_unprotected(value, label=label)
    lexical = Path(value)
    if lexical.name != expected_basename:
        raise NonblindSFTAuditV7Error(
            f"{label} basename must be {expected_basename}"
        )
    if not lexical.is_absolute():
        raise NonblindSFTAuditV7Error(
            f"{label} path must be absolute"
        )
    _assert_canonical_lexical_path(lexical, label=label)
    declared_key = os.path.normcase(os.path.normpath(os.fspath(lexical)))
    authority_key = os.path.normcase(
        os.path.normpath(os.fspath(authority.path))
    )
    if declared_key != authority_key:
        raise NonblindSFTAuditV7Error(
            f"{label} does not match the caller authority path"
        )
    try:
        same = os.path.samefile(authority.path, lexical)
    except OSError as exc:
        raise NonblindSFTAuditV7Error(
            f"{label} authority identity cannot be verified"
        ) from exc
    if not same:
        raise NonblindSFTAuditV7Error(
            f"{label} is not the caller authority file"
        )


def _validate_manifest_header(manifest: Mapping[str, Any]) -> None:
    if set(manifest) != _EXPECTED_MANIFEST_KEYS:
        raise NonblindSFTAuditV7Error(
            "manifest keys do not match the fixed nonblind v7 schema"
        )
    if (
        manifest.get("schema") != NONBLIND_MANIFEST_SCHEMA
        or manifest.get("dataset_schema") != DATASET_SCHEMA
        or manifest.get("builder_version") != NONBLIND_BUILDER_VERSION
        or manifest.get("core_builder_version") != BUILDER_VERSION
        or manifest.get("status")
        != "NONBLIND_DATASET_BUILT_PREBLIND_COMMITTED"
    ):
        raise NonblindSFTAuditV7Error(
            "manifest schema, version, or status is not accepted"
        )


def _validate_manifest_contract(
    manifest: Mapping[str, Any],
    files: Mapping[str, FileSnapshot],
) -> None:
    _validate_manifest_header(manifest)
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or set(splits) != set(_SPLIT_ROLES):
        raise NonblindSFTAuditV7Error(
            "manifest split inventory is not the fixed nonblind inventory"
        )
    for split in _SPLIT_ROLES:
        _validate_receipt(
            splits[split],
            snapshot=files[split],
            expected_path=_ROLE_FILENAMES[split],
            expected_count=EXPECTED_NONBLIND_SPLIT_COUNTS[split],
            label=split,
        )

    artifacts = manifest.get("artifacts")
    if (
        not isinstance(artifacts, dict)
        or set(artifacts) != set(_ARTIFACT_ROLES)
    ):
        raise NonblindSFTAuditV7Error(
            "manifest artifact inventory is not the fixed v7 inventory"
        )
    for role in _ARTIFACT_ROLES:
        _validate_receipt(
            artifacts[role],
            snapshot=files[role],
            expected_path=_ROLE_FILENAMES[role],
            expected_count=None,
            label=role,
        )

    if (
        manifest.get("ground_truth_policy")
        != (
            "deterministic pointer labels from licensed evidence; "
            "no API or teacher output is ground truth"
        )
        or manifest.get("selection_policy")
        != "researcher_explicit_domain_and_task"
        or manifest.get("source_isolation_unit") != "DOI/source_family"
        or manifest.get("counts")
        != {
            "examples": EXPECTED_NONBLIND_TOTAL,
            "families": sum(
                EXPECTED_NONBLIND_FAMILY_SPLIT_COUNTS.values()
            ),
            "examples_per_family": EXAMPLES_PER_FAMILY,
            "splits": dict(EXPECTED_NONBLIND_SPLIT_COUNTS),
        }
        or manifest.get("pointer_contract")
        != {
            "field_order": list(POINTER_FIELDS),
            "answer_span_pattern": "E#.S#",
            "refusal_span_id": None,
        }
        or manifest.get("compiler_input_contract")
        != {
            "compiler_version": COMPILER_VERSION,
            "prompt_schema": COMPILER_PROMPT_SCHEMA,
            "compiler_prompt_keys": sorted(COMPILER_PROMPT_FIELDS),
            "compiler_evidence_keys": sorted(COMPILER_EVIDENCE_FIELDS),
            "compiler_sentence_keys": sorted(COMPILER_SENTENCE_FIELDS),
            "target_free": True,
            "user_text_reverse_parsing_required": False,
        }
        or manifest.get("external_answer_contract")
        != {
            "schema": EXTERNAL_ANSWER_SCHEMA,
            "field_order": list(EXTERNAL_ANSWER_FIELDS),
            "generated_by": "later_deterministic_evidence_compiler",
            "implemented_by_this_builder": False,
        }
        or manifest.get("training_boundary")
        != {
            "allowed_splits": list(TRAINING_SPLITS),
            "calibration_content_for_training": False,
        }
        or manifest.get("claims") != _EXPECTED_CLAIMS
    ):
        raise NonblindSFTAuditV7Error(
            "manifest nonblind contract is not exact"
        )


def _independent_rag_binding(
    *,
    manifest_snapshot: FileSnapshot,
    chunks_snapshot: FileSnapshot,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_payload = _strict_json_object(
        manifest_snapshot.payload,
        label="authority RAG manifest",
    )
    try:
        manifest = RegistryManifestV2.from_dict(manifest_payload)
    except ContractError as exc:
        raise NonblindSFTAuditV7Error(
            "RAG manifest canonical contract mismatch"
        ) from exc

    try:
        text = chunks_snapshot.payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise NonblindSFTAuditV7Error(
            "licensed chunks are not UTF-8"
        ) from exc
    chunks: list[ChunkV1] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        value = _strict_json(
            line.encode("utf-8"),
            label=f"licensed chunks row {line_number}",
        )
        if not isinstance(value, dict):
            raise NonblindSFTAuditV7Error(
                f"licensed chunks row {line_number} must be an object"
            )
        try:
            chunks.append(ChunkV1.from_dict(value))
        except ContractError as exc:
            raise NonblindSFTAuditV7Error(
                f"licensed chunks row {line_number} contract mismatch"
            ) from exc
    if not chunks:
        raise NonblindSFTAuditV7Error(
            "licensed chunks require at least one row"
        )
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise NonblindSFTAuditV7Error(
            "licensed chunks contain duplicate chunk_id values"
        )

    entries = {
        entry.namespace: entry
        for entry in manifest.namespaces
    }
    if any(namespace not in entries for namespace in evidence.DOMAINS):
        raise NonblindSFTAuditV7Error(
            "RAG manifest is missing a required ICMat namespace"
        )
    namespace_counts = Counter(chunk.namespace for chunk in chunks)
    source_counts = Counter(chunk.source_id for chunk in chunks)
    observed_sources: set[str] = set()
    observed_source_namespaces: dict[str, str] = {}
    findings: list[str] = []
    for namespace in evidence.DOMAINS:
        entry = entries[namespace]
        namespace_chunks = tuple(
            chunk for chunk in chunks if chunk.namespace == namespace
        )
        if (
            entry.source_mode
            != "licensed_metadata_and_fulltext_readonly"
        ):
            findings.append(f"{namespace}:SOURCE_MODE_MISMATCH")
        declared_count = int(
            entry.evidence_counts["literature_knowledge"]
        )
        if namespace_counts[namespace] != declared_count:
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
            asset.source_id for asset in entry.source_assets
        ]
        if len(all_asset_ids) != len(set(all_asset_ids)):
            findings.append(f"{namespace}:DUPLICATE_SOURCE_ASSET")
        namespace_sources = {
            chunk.source_id
            for chunk in namespace_chunks
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
        for chunk in namespace_chunks:
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
    expected_family_count = (
        sum(EXPECTED_NONBLIND_FAMILY_SPLIT_COUNTS.values())
        + EXPECTED_BLIND_COUNT // EXAMPLES_PER_FAMILY
    )
    if len(observed_sources) != expected_family_count:
        findings.append("LICENSED_SOURCE_COUNT_MISMATCH")
    if findings:
        raise NonblindSFTAuditV7Error(
            "RAG manifest/licensed chunks authority mismatch: "
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
        "licensed_chunk_identity_sha256": _sha256_bytes(
            canonical_json(identity_payload).encode("utf-8")
        ),
    }
    return manifest_payload, binding


def _snapshot_jsonl_objects(
    snapshot: FileSnapshot,
    *,
    label: str,
) -> tuple[dict[str, Any], ...]:
    try:
        text = snapshot.payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise NonblindSFTAuditV7Error(f"{label} is not UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        value = _strict_json(
            line.encode("utf-8"),
            label=f"{label} row {line_number}",
        )
        if not isinstance(value, dict):
            raise NonblindSFTAuditV7Error(
                f"{label} row {line_number} must be an object"
            )
        rows.append(value)
    return tuple(rows)


def _snapshot_candidate_sentences(
    chunk: Mapping[str, Any],
) -> tuple[evidence.SentenceCandidate, ...]:
    text = str(chunk.get("text", ""))
    lines = text.splitlines()
    if lines and lines[0].lower().startswith("section:"):
        text = "\n".join(lines[1:])
    complete = [
        sentence
        for sentence in evidence.split_scientific_sentences(text)
        if evidence.fragment_reason(sentence) is None
    ]
    output: list[evidence.SentenceCandidate] = []
    for index, sentence in enumerate(complete):
        start = max(0, index - 1)
        end = min(len(complete), index + 2)
        passage_sentences = tuple(complete[start:end])
        if len(" ".join(passage_sentences)) > 760:
            passage_sentences = (sentence,)
        normalized = [
            _normalized_text(item) for item in passage_sentences
        ]
        if len(normalized) != len(set(normalized)):
            continue
        output.append(
            evidence.SentenceCandidate(
                chunk_id=str(chunk["chunk_id"]),
                sentence=sentence,
                sentence_index=index,
                passage_sentences=passage_sentences,
            )
        )
    return tuple(output)


def _load_licensed_families_snapshot(
    snapshot: FileSnapshot,
) -> tuple[evidence.SourceFamily, ...]:
    grouped: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in _snapshot_jsonl_objects(
        snapshot,
        label="frozen licensed chunks",
    ):
        try:
            chunk = ChunkV1.from_dict(row)
        except ContractError as exc:
            raise NonblindSFTAuditV7Error(
                "frozen licensed chunk contract mismatch"
            ) from exc
        if chunk.namespace not in evidence.DOMAINS:
            continue
        metadata = dict(chunk.metadata)
        if (
            chunk.license_id != "CC BY 4.0"
            or metadata.get("access_mode")
            != "licensed_fulltext_readonly"
        ):
            raise NonblindSFTAuditV7Error(
                "frozen licensed family provenance is invalid"
            )
        grouped[(chunk.namespace, chunk.source_id)].append(
            chunk.to_dict()
        )

    families: list[evidence.SourceFamily] = []
    observed_dois: dict[str, str] = {}
    for (namespace, source_id), unsorted_chunks in sorted(
        grouped.items()
    ):
        chunks = sorted(
            unsorted_chunks,
            key=lambda item: str(item.get("chunk_id", "")),
        )
        first = chunks[0]
        metadata = first["metadata"]
        doi = str(metadata.get("doi", "")).strip().lower()
        if not doi:
            raise NonblindSFTAuditV7Error(
                "frozen licensed family DOI is required"
            )
        prior_source = observed_dois.get(doi)
        if prior_source is not None and prior_source != source_id:
            raise NonblindSFTAuditV7Error(
                "one DOI cannot define more than one frozen source family"
            )
        observed_dois[doi] = source_id
        expected = {
            "source_title": str(first.get("source_title", "")),
            "source_uri": str(first.get("source_uri", "")),
            "license_id": str(first.get("license_id", "")),
            "doi": doi,
            "measurement_status": str(
                metadata.get(
                    "measurement_status",
                    "published_literature_not_local_measurement",
                )
            ),
        }
        candidates: dict[str, evidence.SentenceCandidate] = {}
        for chunk in chunks:
            chunk_metadata = chunk.get("metadata")
            if not isinstance(chunk_metadata, dict):
                raise NonblindSFTAuditV7Error(
                    "frozen licensed chunk metadata is invalid"
                )
            observed = {
                "source_title": str(chunk.get("source_title", "")),
                "source_uri": str(chunk.get("source_uri", "")),
                "license_id": str(chunk.get("license_id", "")),
                "doi": str(
                    chunk_metadata.get("doi", "")
                ).strip().lower(),
                "measurement_status": str(
                    chunk_metadata.get(
                        "measurement_status",
                        "published_literature_not_local_measurement",
                    )
                ),
            }
            if observed != expected:
                raise NonblindSFTAuditV7Error(
                    "frozen licensed family provenance is inconsistent"
                )
            for candidate in _snapshot_candidate_sentences(chunk):
                normalized = _normalized_text(candidate.sentence)
                candidates.setdefault(normalized, candidate)
        sentences = tuple(
            sorted(
                candidates.values(),
                key=lambda item: _sha256_bytes(
                    f"{source_id}\0{item.sentence}".encode()
                ),
            )
        )
        if len(sentences) < EXAMPLES_PER_FAMILY:
            raise NonblindSFTAuditV7Error(
                "frozen licensed family has fewer than 50 sentences"
            )
        families.append(
            evidence.SourceFamily(
                source_id=source_id,
                namespace=namespace,
                source_title=expected["source_title"],
                source_uri=expected["source_uri"],
                doi=doi,
                license_id=expected["license_id"],
                measurement_status=expected["measurement_status"],
                chunks=tuple(chunks),
                sentences=sentences,
            )
        )
    source_ids = [family.source_id for family in families]
    if len(source_ids) != len(set(source_ids)):
        raise NonblindSFTAuditV7Error(
            "frozen source family IDs are not globally unique"
        )
    domain_counts = Counter(family.namespace for family in families)
    if any(domain_counts[domain] < 4 for domain in evidence.DOMAINS):
        raise NonblindSFTAuditV7Error(
            "each frozen domain requires at least four source families"
        )
    return tuple(families)


def _semantic_record_sha256(row: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in row.items()
        if key != "record_sha256"
    }
    return _sha256_bytes(canonical_json(payload).encode("utf-8"))


def _semantic_v17_original(
    family: evidence.SourceFamily,
    row: Mapping[str, Any],
) -> str | None:
    original = row.get("original_sentence")
    chunk_ids = row.get("chunk_ids")
    if not isinstance(original, str) or not isinstance(chunk_ids, list):
        return None
    chunks_by_id = {
        str(chunk.get("chunk_id", "")): chunk
        for chunk in family.chunks
    }
    found = False
    for chunk_id in sorted(chunk_ids):
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            return None
        candidates = semantic_queries._protected_sentence_split(  # noqa: SLF001
            str(chunk.get("text", ""))
        )
        if original in candidates and (
            semantic_queries.is_usable_scientific_sentence(original)
        ):
            found = True
    return original if found else None


def _load_semantic_inventory_snapshots(
    inventory_snapshot: FileSnapshot,
    records_snapshot: FileSnapshot,
    families: Sequence[evidence.SourceFamily],
) -> tuple[
    dict[tuple[str, str], evidence.SemanticQueryRecord],
    dict[str, Any],
]:
    inventory = _strict_json_object(
        inventory_snapshot.payload,
        label="frozen semantic accepted inventory",
    )
    inventory_fields = frozenset(inventory)
    if inventory_fields not in {
        _SEMANTIC_INVENTORY_FIELDS,
        _SEMANTIC_INVENTORY_V17_FIELDS,
    }:
        raise NonblindSFTAuditV7Error(
            "frozen semantic inventory keys mismatch"
        )
    extended_inventory = (
        inventory_fields == _SEMANTIC_INVENTORY_V17_FIELDS
    )
    inventory_core = {
        key: value
        for key, value in inventory.items()
        if key != "inventory_sha256"
    }
    if (
        inventory.get("schema") != ACCEPTED_INVENTORY_SCHEMA
        or inventory.get("status")
        != "ACCEPTED_INDEPENDENT_LOCAL_NLI_AUDITED"
        or inventory.get("quality_claim_allowed") is not True
        or inventory.get("training_authorized") is not True
        or inventory.get("generated_text_is_ground_truth") is not False
        or inventory.get("licensed_original_is_ground_truth") is not True
        or inventory.get("sealed_blind_access")
        != {
            "read": False,
            "hashed": False,
            "path_discovered": False,
        }
        or _validate_sha256(
            inventory.get("request_manifest_sha256"),
            label="semantic request manifest SHA-256",
        )
        != inventory.get("request_manifest_sha256")
        or inventory.get("inventory_sha256")
        != _sha256_bytes(
            canonical_json(inventory_core).encode("utf-8")
        )
        or (
            extended_inventory
            and (
                _validate_sha256(
                    inventory.get("smoke_gate_sha256"),
                    label="semantic smoke gate SHA-256",
                )
                != inventory.get("smoke_gate_sha256")
                or _validate_sha256(
                    inventory.get("staging_contract_sha256"),
                    label="semantic staging contract SHA-256",
                )
                != inventory.get("staging_contract_sha256")
                or inventory.get("source_coverage_passed") is not True
                or not isinstance(
                    inventory.get("source_coverage"),
                    Mapping,
                )
            )
        )
    ):
        raise NonblindSFTAuditV7Error(
            "frozen semantic inventory integrity mismatch"
        )
    record_count = inventory.get("record_count")
    accepted_count = inventory.get("accepted_count")
    rejected_count = inventory.get("rejected_or_fixture_count")
    if (
        not _is_integer(record_count)
        or not _is_integer(accepted_count)
        or not _is_integer(rejected_count)
        or int(record_count) <= 0
        or int(accepted_count) <= 0
        or int(rejected_count) < 0
        or int(accepted_count) + int(rejected_count)
        != int(record_count)
    ):
        raise NonblindSFTAuditV7Error(
            "frozen semantic inventory counts mismatch"
        )

    records_by_id: dict[str, dict[str, Any]] = {}
    for row in _snapshot_jsonl_objects(
        records_snapshot,
        label="frozen semantic records",
    ):
        record_id = row.get("record_id")
        record_sha256 = row.get("record_sha256")
        if (
            frozenset(row)
            not in {
                _SEMANTIC_RECORD_FIELDS,
                _SEMANTIC_RECORD_V17_FIELDS,
            }
            or row.get("schema") != SEMANTIC_QUERY_SCHEMA
            or not isinstance(record_id, str)
            or not record_id
            or record_id in records_by_id
            or _validate_sha256(
                record_sha256,
                label="semantic record SHA-256",
            )
            != _semantic_record_sha256(row)
        ):
            raise NonblindSFTAuditV7Error(
                "frozen semantic record identity or hash mismatch"
            )
        records_by_id[record_id] = row
    if len(records_by_id) != record_count:
        raise NonblindSFTAuditV7Error(
            "frozen semantic records count mismatch"
        )

    accepted_entries = inventory.get("accepted_records")
    if (
        not isinstance(accepted_entries, list)
        or len(accepted_entries) != accepted_count
    ):
        raise NonblindSFTAuditV7Error(
            "frozen semantic accepted entries mismatch"
        )
    accepted_rows: list[dict[str, Any]] = []
    accepted_ids: set[str] = set()
    accepted_bindings: set[tuple[str, str]] = set()
    for entry in accepted_entries:
        if (
            not isinstance(entry, Mapping)
            or set(entry)
            != {
                "record_id",
                "record_sha256",
                "source_id",
                "original_sha256",
            }
        ):
            raise NonblindSFTAuditV7Error(
                "frozen semantic accepted entry keys mismatch"
            )
        record_id = entry.get("record_id")
        source_id = entry.get("source_id")
        original_sha256 = entry.get("original_sha256")
        row = records_by_id.get(record_id)
        binding = (str(source_id), str(original_sha256))
        if (
            not isinstance(record_id, str)
            or record_id in accepted_ids
            or not isinstance(source_id, str)
            or not isinstance(original_sha256, str)
            or row is None
            or entry.get("record_sha256")
            != row.get("record_sha256")
            or source_id != row.get("source_id")
            or original_sha256 != row.get("original_sha256")
            or binding in accepted_bindings
        ):
            raise NonblindSFTAuditV7Error(
                "frozen semantic accepted entry binding mismatch"
            )
        accepted_ids.add(record_id)
        accepted_bindings.add(binding)
        accepted_rows.append(row)

    originals: dict[tuple[str, str], str] = {}
    families_by_id = {
        family.source_id: family for family in families
    }
    for family in families:
        for candidate in family.sentences:
            digest = _sha256_bytes(candidate.sentence.encode("utf-8"))
            binding = (family.source_id, digest)
            prior = originals.get(binding)
            if prior is not None and prior != candidate.sentence:
                raise NonblindSFTAuditV7Error(
                    "frozen semantic original hash collision"
                )
            originals[binding] = candidate.sentence

    records: dict[
        tuple[str, str],
        evidence.SemanticQueryRecord,
    ] = {}
    observed_hashes: set[str] = set()
    accepted_generator: Mapping[str, Any] | None = None
    accepted_nli: Mapping[str, Any] | None = None
    for row in accepted_rows:
        source_id = row.get("source_id")
        original_sha256 = row.get("original_sha256")
        if (
            not isinstance(source_id, str)
            or source_id not in families_by_id
            or not isinstance(original_sha256, str)
        ):
            raise NonblindSFTAuditV7Error(
                "frozen semantic record references an unknown family"
            )
        binding = (source_id, original_sha256)
        family = families_by_id[source_id]
        v17_record = (
            frozenset(row) == _SEMANTIC_RECORD_V17_FIELDS
        )
        original = (
            _semantic_v17_original(family, row)
            if v17_record
            else originals.get(binding)
        )
        paraphrase = row.get("paraphrase")
        contradiction = row.get("contradiction")
        mutation_type = row.get("mutation_type")
        request_id = row.get("request_id")
        chunk_ids = row.get("chunk_ids")
        locators = row.get("locators")
        if (
            binding in records
            or original is None
            or row.get("original_sentence") != original
            or _sha256_bytes(original.encode("utf-8"))
            != original_sha256
            or row.get("license_id") != "CC BY 4.0"
            or row.get("namespace") != family.namespace
            or row.get("source_title") != family.source_title
            or row.get("source_uri") != family.source_uri
            or not isinstance(chunk_ids, list)
            or not chunk_ids
            or len(chunk_ids) != len(set(chunk_ids))
            or not set(chunk_ids)
            <= {
                str(chunk.get("chunk_id", ""))
                for chunk in family.chunks
            }
            or not isinstance(locators, list)
            or any(
                not isinstance(locator, str) or not locator
                for locator in locators
            )
            or row.get("acceptance") != _SEMANTIC_ACCEPTANCE
            or row.get("ground_truth_boundary")
            != (
                "The licensed original sentence is ground truth. "
                "Generated text is an audited query transformation "
                "and is not ground truth."
            )
            or not isinstance(paraphrase, str)
            or not isinstance(contradiction, str)
            or (
                v17_record
                and (
                    not paraphrase.strip()
                    or not contradiction.strip()
                    or len(paraphrase) > 700
                    or len(contradiction) > 700
                )
            )
            or (
                not v17_record
                and (
                    evidence.fragment_reason(paraphrase) is not None
                    or evidence.fragment_reason(contradiction) is not None
                )
            )
            or _normalized_text(original)
            in {
                _normalized_text(paraphrase),
                _normalized_text(contradiction),
            }
            or mutation_type not in _SEMANTIC_MUTATION_TYPES
            or not isinstance(row.get("mutation"), Mapping)
            or not isinstance(row.get("audits"), Mapping)
            or not isinstance(request_id, str)
            or not request_id.startswith("icmsq7:")
            or not _HEX_SHA256.fullmatch(
                request_id.removeprefix("icmsq7:")
            )
            or any(
                not isinstance(row.get(key), str)
                or not _HEX_SHA256.fullmatch(str(row.get(key)))
                for key in ("request_sha256", "source_record_sha256")
            )
            or (
                v17_record
                and (
                    row.get("source_manifest_authority")
                    != "rag_v2_licensed_source_catalog"
                    or not isinstance(
                        row.get("source_asset_sha256"),
                        str,
                    )
                    or not _HEX_SHA256.fullmatch(
                        str(row.get("source_asset_sha256"))
                    )
                    or {
                        str(
                            chunk.get("metadata", {}).get(
                                "xml_sha256",
                                "",
                            )
                        )
                        for chunk in family.chunks
                        if isinstance(chunk.get("metadata"), Mapping)
                    }
                    != {row.get("source_asset_sha256")}
                    or not isinstance(
                        row.get("source_asset_uri"),
                        str,
                    )
                    or not str(row.get("source_asset_uri")).startswith(
                        ("https://", "http://")
                    )
                    or not isinstance(
                        row.get("generation_response_trace"),
                        list,
                    )
                    or not row.get("generation_response_trace")
                    or row.get("generation_response_tree_sha256")
                    != _sha256_bytes(
                        canonical_json(
                            row.get("generation_response_trace")
                        ).encode("utf-8")
                    )
                    or row.get("mutation", {}).get(
                        "contradiction_constructed_by_code"
                    )
                    is not True
                    or row.get("mutation", {}).get(
                        "model_generated_contradiction_allowed"
                    )
                    is not False
                )
            )
        ):
            raise NonblindSFTAuditV7Error(
                "frozen semantic record contract mismatch"
            )
        expected_record_id = "icmsqr7:" + _sha256_bytes(
            canonical_json(
                {
                    "request_id": request_id,
                    "paraphrase": paraphrase,
                    "contradiction": contradiction,
                    "mutation_type": mutation_type,
                }
            ).encode("utf-8")
        )
        record_hash = str(row["record_sha256"])
        generator = row.get("generator_provenance")
        nli = row.get("nli_provenance")
        if (
            row.get("record_id") != expected_record_id
            or record_hash in observed_hashes
            or not isinstance(generator, Mapping)
            or not isinstance(nli, Mapping)
        ):
            raise NonblindSFTAuditV7Error(
                "frozen semantic record provenance is invalid"
            )
        normalized_generator = dict(generator)
        normalized_generator.pop("raw_response_sha256", None)
        normalized_generator.pop("deterministic_fallback", None)
        if accepted_generator is None:
            accepted_generator = normalized_generator
            accepted_nli = dict(nli)
        elif (
            normalized_generator != accepted_generator
            or dict(nli) != accepted_nli
        ):
            raise NonblindSFTAuditV7Error(
                "frozen semantic records mix model provenance"
            )
        observed_hashes.add(record_hash)
        records[binding] = evidence.SemanticQueryRecord(
            source_id=source_id,
            original_sha256=original_sha256,
            paraphrase=paraphrase,
            contradiction=contradiction,
            mutation_type=str(mutation_type),
            record_sha256=record_hash,
        )
    per_family = Counter(source_id for source_id, _ in records)
    if (
        not records
        or any(
            per_family[family.source_id] < EXAMPLES_PER_FAMILY
            for family in families
        )
        or inventory.get("generator_provenance")
        != accepted_generator
        or inventory.get("nli_provenance") != accepted_nli
    ):
        raise NonblindSFTAuditV7Error(
            "frozen semantic inventory coverage or provenance mismatch"
        )
    if extended_inventory:
        expected_source_coverage = {
            source_id: {
                "accepted_count": per_family[source_id],
                "minimum_required": EXAMPLES_PER_FAMILY,
                "passed": True,
            }
            for source_id in sorted(families_by_id)
        }
        if inventory.get("source_coverage") != expected_source_coverage:
            raise NonblindSFTAuditV7Error(
                "frozen semantic source coverage mismatch"
            )
    audit = {
        "schema": evidence.SEMANTIC_INVENTORY_AUDIT_SCHEMA,
        "builder_version": BUILDER_VERSION,
        "status": "PASS",
        "findings": [],
        "semantic_inventory_sha256": inventory_snapshot.sha256,
        "producer_inventory_sha256": inventory["inventory_sha256"],
        "semantic_records_sha256": records_snapshot.sha256,
        "record_schema": SEMANTIC_QUERY_SCHEMA,
        "record_count": record_count,
        "accepted_count": len(records),
        "rejected_or_fixture_count": rejected_count,
        "unique_binding_count": len(records),
        "unique_record_hash_count": len(observed_hashes),
        "covered_source_family_count": len(per_family),
        "minimum_records_per_family": min(per_family.values()),
        "contract": {
            "binding": "source_id+original_sha256",
            "accepted_inventory_schema": ACCEPTED_INVENTORY_SCHEMA,
            "accepted_required": True,
            "paraphrase_required": True,
            "controlled_contradiction_required": True,
            "provenance_required": True,
            "audit_required": True,
            "record_hash_required": True,
            "fallback_without_inventory": False,
        },
    }
    return records, audit


def _snapshot_declared_sources(
    manifest: Mapping[str, Any],
    *,
    authority: AuthorityInputs,
) -> tuple[SourceState, tuple[Any, ...], Mapping[Any, Any], Mapping[str, Any]]:
    current_nonblind = _snapshot_regular_file(
        Path(builder.__file__),
        label="current nonblind builder source",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    current_core = _snapshot_regular_file(
        Path(evidence.__file__),
        label="current evidence core source",
        maximum_bytes=_MAX_JSON_BYTES,
    )

    builder_receipt = manifest.get("builder")
    if (
        not isinstance(builder_receipt, dict)
        or set(builder_receipt)
        != {
            "nonblind_module",
            "evidence_core",
            "split_algorithm_version",
            "seed",
        }
        or builder_receipt.get("split_algorithm_version")
        != SPLIT_ALGORITHM_VERSION
        or not isinstance(builder_receipt.get("seed"), str)
        or not builder_receipt["seed"]
    ):
        raise NonblindSFTAuditV7Error(
            "manifest builder and seed contract is invalid"
        )
    for role, current in (
        ("nonblind_module", current_nonblind),
        ("evidence_core", current_core),
    ):
        receipt = builder_receipt.get(role)
        if (
            not isinstance(receipt, dict)
            or set(receipt) != {"path", "sha256"}
        ):
            raise NonblindSFTAuditV7Error(
                f"manifest {role} source receipt is invalid"
            )
        _validate_declared_path(
            receipt.get("path"),
            authority=current,
            expected_basename=current.path.name,
            label=f"declared {role} source",
        )
        if (
            _validate_sha256(
                receipt.get("sha256"),
                label=f"{role} source SHA-256",
            )
            != current.sha256
        ):
            raise NonblindSFTAuditV7Error(
                f"manifest {role} source binding is stale"
            )

    source_inputs = manifest.get("source_inputs")
    if (
        not isinstance(source_inputs, dict)
        or set(source_inputs)
        != {"licensed_chunks", "rag_manifest", "semantic_inventory"}
    ):
        raise NonblindSFTAuditV7Error(
            "manifest source input inventory is invalid"
        )
    licensed_receipt = source_inputs["licensed_chunks"]
    rag_receipt = source_inputs["rag_manifest"]
    semantic_receipt = source_inputs["semantic_inventory"]
    if (
        not isinstance(licensed_receipt, dict)
        or set(licensed_receipt) != {"path", "sha256"}
        or not isinstance(rag_receipt, dict)
        or set(rag_receipt) != {"path", "sha256", "manifest_id"}
        or not isinstance(semantic_receipt, dict)
        or set(semantic_receipt)
        != {
            "path",
            "sha256",
            "schema",
            "producer_inventory_sha256",
            "records_sha256",
            "record_schema",
            "record_count",
            "accepted_count",
        }
    ):
        raise NonblindSFTAuditV7Error(
            "manifest source input receipt keys are invalid"
        )

    _validate_declared_path(
        licensed_receipt.get("path"),
        authority=authority.licensed_chunks,
        expected_basename="licensed_chunks.v1.jsonl",
        label="licensed chunks",
    )
    _validate_declared_path(
        rag_receipt.get("path"),
        authority=authority.rag_manifest,
        expected_basename="manifest.v2.json",
        label="RAG manifest",
    )
    _validate_declared_path(
        semantic_receipt.get("path"),
        authority=authority.semantic_inventory,
        expected_basename="accepted_inventory.v7.json",
        label="semantic inventory",
    )
    licensed = authority.licensed_chunks
    rag = authority.rag_manifest
    semantic = authority.semantic_inventory
    semantic_records = authority.semantic_records
    for receipt, snapshot, label in (
        (licensed_receipt, licensed, "licensed chunks"),
        (rag_receipt, rag, "RAG manifest"),
        (semantic_receipt, semantic, "semantic inventory"),
    ):
        if (
            _validate_sha256(
                receipt.get("sha256"),
                label=f"{label} SHA-256",
            )
            != snapshot.sha256
        ):
            raise NonblindSFTAuditV7Error(
                f"{label} current bytes do not match the manifest"
            )
    if (
        _validate_sha256(
            semantic_receipt.get("records_sha256"),
            label="semantic records SHA-256",
        )
        != semantic_records.sha256
        or semantic_receipt.get("schema") != ACCEPTED_INVENTORY_SCHEMA
        or semantic_receipt.get("record_schema") != SEMANTIC_QUERY_SCHEMA
    ):
        raise NonblindSFTAuditV7Error(
            "semantic inventory or records manifest binding is invalid"
        )

    rag_payload, rag_authority_binding = _independent_rag_binding(
        manifest_snapshot=rag,
        chunks_snapshot=licensed,
    )
    rag_manifest_id = rag_payload.get("manifest_id")
    if (
        rag_payload.get("schema") != "icmat.rag.manifest.v2"
        or not isinstance(rag_manifest_id, str)
        or not rag_manifest_id
        or rag_receipt.get("manifest_id") != rag_manifest_id
    ):
        raise NonblindSFTAuditV7Error(
            "RAG manifest identity is invalid"
    )

    try:
        families = _load_licensed_families_snapshot(licensed)
        semantic_map, semantic_audit = (
            _load_semantic_inventory_snapshots(
                semantic,
                semantic_records,
                families,
            )
        )
    except (EvidenceSFTV6Error, OSError, ValueError) as exc:
        raise NonblindSFTAuditV7Error(
            f"source revalidation failed: {exc}"
        ) from exc
    if (
        semantic_audit.get("semantic_inventory_sha256")
        != semantic.sha256
        or semantic_audit.get("semantic_records_sha256")
        != semantic_records.sha256
        or semantic_receipt.get("producer_inventory_sha256")
        != semantic_audit.get("producer_inventory_sha256")
        or semantic_receipt.get("record_count")
        != semantic_audit.get("record_count")
        or semantic_receipt.get("accepted_count")
        != semantic_audit.get("accepted_count")
    ):
        raise NonblindSFTAuditV7Error(
            "semantic inventory audit binding is invalid"
        )

    state = SourceState(
        licensed_chunks=licensed,
        rag_manifest=rag,
        semantic_inventory=semantic,
        semantic_records=semantic_records,
        nonblind_builder=current_nonblind,
        evidence_core=current_core,
        rag_manifest_id=rag_manifest_id,
        rag_authority_binding=rag_authority_binding,
        seed=str(builder_receipt["seed"]),
    )
    return state, families, semantic_map, semantic_audit


def _parse_and_validate_rows(
    files: Mapping[str, FileSnapshot],
    *,
    families: Sequence[Any],
    semantic_map: Mapping[Any, Any],
    assignments: Mapping[str, str],
) -> tuple[Mapping[str, Any], ...]:
    families_by_id = {family.source_id: family for family in families}
    rows: list[Mapping[str, Any]] = []
    all_ids: set[str] = set()
    split_sources: dict[str, set[str]] = {}
    family_counts: Counter[str] = Counter()

    for split in _SPLIT_ROLES:
        snapshot = files[split]
        if not snapshot.payload.endswith(b"\n"):
            raise NonblindSFTAuditV7Error(
                f"{split} JSONL must end with one newline"
            )
        lines = snapshot.payload.splitlines()
        expected_count = EXPECTED_NONBLIND_SPLIT_COUNTS[split]
        if len(lines) != expected_count or any(not line for line in lines):
            raise NonblindSFTAuditV7Error(
                f"{split} JSONL row count mismatch"
            )
        local_sources: set[str] = set()
        for index, line in enumerate(lines, 1):
            row = _strict_json(
                line,
                label=f"{split} JSONL row {index}",
            )
            if not isinstance(row, dict):
                raise NonblindSFTAuditV7Error(
                    f"{split} JSONL row {index} must be an object"
                )
            try:
                evidence.validate_example(row)
            except (EvidenceSFTV6Error, OSError, ValueError) as exc:
                raise NonblindSFTAuditV7Error(
                    f"{split} JSONL row {index} pointer contract failed: {exc}"
                ) from exc
            if row.get("split") != split:
                raise NonblindSFTAuditV7Error(
                    f"{split} JSONL row {index} split mismatch"
                )
            example_id = row.get("example_id")
            source_id = row.get("source_id")
            if (
                not isinstance(example_id, str)
                or not example_id
                or example_id in all_ids
                or not isinstance(source_id, str)
                or source_id not in families_by_id
                or row.get("family_id") != source_id
                or assignments.get(source_id) != split
            ):
                raise NonblindSFTAuditV7Error(
                    f"{split} JSONL row {index} identity or family binding failed"
                )
            family = families_by_id[source_id]
            metadata = row.get("metadata")
            construction = (
                metadata.get("construction")
                if isinstance(metadata, Mapping)
                else None
            )
            if not isinstance(construction, Mapping):
                raise NonblindSFTAuditV7Error(
                    f"{split} JSONL row {index} lacks semantic construction"
                )
            original_sha256 = construction.get("original_sha256")
            semantic = semantic_map.get((source_id, original_sha256))
            if (
                semantic is None
                or semantic.record_sha256
                != construction.get("semantic_record_sha256")
                or row.get("domain") != family.namespace
                or row.get("doi") != family.doi
                or row.get("license_id") != family.license_id
                or metadata.get("source_title") != family.source_title
                or metadata.get("source_uri") != family.source_uri
                or metadata.get("measurement_status")
                != family.measurement_status
            ):
                raise NonblindSFTAuditV7Error(
                    f"{split} JSONL row {index} semantic provenance binding failed"
                )
            query_kind = construction.get("query_kind")
            expected_claim = (
                semantic.contradiction
                if query_kind == "refuse_controlled_contradiction"
                else semantic.paraphrase
            )
            if (
                query_kind
                not in {
                    "answer_paraphrase",
                    "refuse_controlled_contradiction",
                    "refuse_hidden_same_family_paraphrase",
                }
                or row.get("requested_claim") != expected_claim
            ):
                raise NonblindSFTAuditV7Error(
                    f"{split} JSONL row {index} semantic query binding failed"
                )
            all_ids.add(example_id)
            local_sources.add(source_id)
            family_counts[source_id] += 1
            rows.append(row)
        split_sources[split] = local_sources

    if len(rows) != EXPECTED_NONBLIND_TOTAL:
        raise NonblindSFTAuditV7Error("nonblind row total is not 550")
    for index, left in enumerate(_SPLIT_ROLES):
        if (
            len(split_sources[left])
            != EXPECTED_NONBLIND_FAMILY_SPLIT_COUNTS[left]
        ):
            raise NonblindSFTAuditV7Error(
                f"{left} source-family count mismatch"
            )
        for right in _SPLIT_ROLES[index + 1 :]:
            if split_sources[left] & split_sources[right]:
                raise NonblindSFTAuditV7Error(
                    "source families overlap across nonblind splits"
                )
    if (
        set(family_counts) != set().union(*split_sources.values())
        or any(count != EXAMPLES_PER_FAMILY for count in family_counts.values())
    ):
        raise NonblindSFTAuditV7Error(
            "each included source family must contribute exactly 50 rows"
        )
    return tuple(rows)


def _validate_preblind_commitment(
    payload: Mapping[str, Any],
    *,
    sources: SourceState,
) -> None:
    expected_keys = {
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
    if set(payload) != expected_keys:
        raise NonblindSFTAuditV7Error(
            "preblind commitment keys are not exact"
        )
    expected_builder_code = {
        "nonblind_module_sha256": sources.nonblind_builder.sha256,
        "evidence_core_sha256": sources.evidence_core.sha256,
    }
    expected_inputs = {
        "licensed_chunks_sha256": sources.licensed_chunks.sha256,
        "rag_manifest_sha256": sources.rag_manifest.sha256,
        "rag_manifest_id": sources.rag_manifest_id,
        "semantic_inventory_sha256": sources.semantic_inventory.sha256,
        "semantic_records_sha256": sources.semantic_records.sha256,
    }
    if (
        payload.get("schema") != PREBLIND_COMMITMENT_SCHEMA
        or payload.get("status") != "PREBLIND_COMMITTED_NONBLIND_ONLY"
        or payload.get("builder_version") != NONBLIND_BUILDER_VERSION
        or payload.get("core_builder_version") != BUILDER_VERSION
        or payload.get("split_algorithm_version")
        != SPLIT_ALGORITHM_VERSION
        or payload.get("seed") != sources.seed
        or payload.get("seed_sha256")
        != _sha256_bytes(sources.seed.encode("utf-8"))
        or payload.get("expected_blind_count") != EXPECTED_BLIND_COUNT
        or payload.get("builder_code") != expected_builder_code
        or payload.get("source_inputs") != expected_inputs
    ):
        raise NonblindSFTAuditV7Error(
            "preblind commitment binding is invalid"
        )
    core = {key: value for key, value in payload.items() if key != "commitment_sha256"}
    if (
        _validate_sha256(
            payload.get("commitment_sha256"),
            label="preblind commitment SHA-256",
        )
        != _sha256_bytes(canonical_json(core).encode("utf-8"))
    ):
        raise NonblindSFTAuditV7Error(
            "preblind commitment digest is invalid"
        )
    serialized = canonical_json(payload)
    if any(
        marker in serialized
        for marker in (
            "blind_test",
            "sealed.v",
            "blind_path",
            "blind_sha256",
            "blind_bytes",
            "blind_content",
        )
    ):
        raise NonblindSFTAuditV7Error(
            "preblind commitment discloses a forbidden reserved asset detail"
        )


def _independent_family_integrity(
    rows: Sequence[Mapping[str, Any]],
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
    for row in rows:
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


def _independent_balance_report(
    rows: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, str],
) -> dict[str, Any]:
    split_counts = Counter(str(row["split"]) for row in rows)
    decision_counts = Counter(
        (str(row["split"]), str(row["decision"]))
        for row in rows
    )
    task_counts = Counter(
        (str(row["split"]), str(row["task"]))
        for row in rows
    )
    family_decisions = Counter(
        (str(row["source_id"]), str(row["decision"]))
        for row in rows
    )
    included_sources = {
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
        if any(task_counts[(split, task)] == 0 for task in TASKS):
            findings.append(f"{split.upper()}_TASK_MISSING")
    imbalanced_families = sum(
        family_decisions[(source_id, "ANSWER")]
        != family_decisions[(source_id, "REFUSE")]
        for source_id in included_sources
    )
    if imbalanced_families:
        findings.append("FAMILY_DECISION_IMBALANCE")
    family_integrity = _independent_family_integrity(rows, assignments)
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
        "included_family_count": len(included_sources),
        "imbalanced_family_count": imbalanced_families,
        "family_integrity": family_integrity,
    }


def _independent_group_commitment(family: Any) -> str:
    payload = (
        f"icmat-v6-group\0{family.source_id}\0"
        f"{family.doi.lower()}\0{family.namespace}"
    )
    return _sha256_bytes(payload.encode("utf-8"))


def _independent_group_report(
    families: Sequence[Any],
    assignments: Mapping[str, str],
) -> dict[str, Any]:
    included = tuple(
        family
        for family in families
        if assignments[family.source_id] in NONBLIND_SPLITS
    )
    source_sets = {
        split: {
            family.source_id
            for family in included
            if assignments[family.source_id] == split
        }
        for split in NONBLIND_SPLITS
    }
    doi_sets = {
        split: {
            family.doi.lower()
            for family in included
            if assignments[family.source_id] == split
        }
        for split in NONBLIND_SPLITS
    }
    commitments = {
        split: sorted(
            _independent_group_commitment(family)
            for family in included
            if assignments[family.source_id] == split
        )
        for split in NONBLIND_SPLITS
    }
    findings: list[str] = []
    pairwise: list[dict[str, Any]] = []
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
    family_counts = {
        split: len(source_sets[split])
        for split in NONBLIND_SPLITS
    }
    if family_counts != EXPECTED_NONBLIND_FAMILY_SPLIT_COUNTS:
        findings.append("NONBLIND_FAMILY_SPLIT_COUNTS_MISMATCH")
    return {
        "schema": NONBLIND_GROUP_SCHEMA,
        "status": "PASS" if not findings else "FAIL",
        "findings": sorted(set(findings)),
        "isolation_unit": "licensed DOI/source family",
        "family_split_counts": family_counts,
        "group_commitments": commitments,
        "pairwise": pairwise,
    }


def _normalized_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _word_ngrams(
    value: str,
    *,
    size: int = 5,
) -> set[tuple[str, ...]]:
    tokens = _normalized_text(value).split()
    if len(tokens) < size:
        return {tuple(tokens)} if tokens else set()
    return {
        tuple(tokens[index : index + size])
        for index in range(len(tokens) - size + 1)
    }


def _jaccard(
    left: set[tuple[str, ...]],
    right: set[tuple[str, ...]],
) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _target_marker_count(value: Any) -> int:
    if isinstance(value, Mapping):
        return sum(
            int(str(key) in _TARGET_MARKER_FIELDS)
            + _target_marker_count(nested)
            for key, nested in value.items()
        )
    if isinstance(value, (list, tuple)):
        return sum(_target_marker_count(item) for item in value)
    return 0


def _independent_shortcut_audit(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    exact_match_count = 0
    exact_match_answer_count = 0
    decision_correct_count = 0
    answer_span_recovery_count = 0
    construction_missing_count = 0
    target_original_in_query_count = 0
    refusal_mode_counts: Counter[str] = Counter()
    refusal_modes_by_family: dict[str, Counter[str]] = defaultdict(
        Counter
    )
    for row in rows:
        query = _normalized_text(str(row["requested_claim"]))
        matching_spans: list[str] = []
        for passage in row.get("compiler_evidence", []):
            for sentence in passage.get("sentences", []):
                if (
                    _normalized_text(
                        str(sentence.get("text", ""))
                    )
                    == query
                ):
                    matching_spans.append(
                        str(sentence.get("span_id", ""))
                    )
        predicted = "ANSWER" if len(matching_spans) == 1 else "REFUSE"
        exact_match_count += len(matching_spans)
        exact_match_answer_count += int(predicted == "ANSWER")
        decision_correct_count += int(predicted == row.get("decision"))
        answer_span_recovery_count += int(
            predicted == "ANSWER"
            and matching_spans[0] == row.get("target_span_id")
        )
        metadata = row.get("metadata")
        construction = (
            metadata.get("construction")
            if isinstance(metadata, Mapping)
            else None
        )
        if not isinstance(construction, Mapping):
            construction_missing_count += 1
            continue
        target_original_in_query_count += int(
            construction.get("target_original_in_query") is not False
        )
        query_kind = str(construction.get("query_kind", ""))
        if query_kind.startswith("refuse_"):
            refusal_mode_counts[query_kind] += 1
            refusal_modes_by_family[
                str(row.get("family_id", ""))
            ][query_kind] += 1
    total = len(rows)
    accuracy = decision_correct_count / total if total else 0.0
    contradiction_key = "refuse_controlled_contradiction"
    hidden_key = "refuse_hidden_same_family_paraphrase"
    global_difference = abs(
        refusal_mode_counts[contradiction_key]
        - refusal_mode_counts[hidden_key]
    )
    maximum_family_difference = max(
        (
            abs(
                counts[contradiction_key]
                - counts[hidden_key]
            )
            for counts in refusal_modes_by_family.values()
        ),
        default=0,
    )
    findings: list[str] = []
    if exact_match_count:
        findings.append("NORMALIZED_EXACT_QUERY_EVIDENCE_MATCH")
    if answer_span_recovery_count:
        findings.append("NORMALIZED_EXACT_SPAN_RECOVERY")
    if accuracy > 0.5:
        findings.append("NORMALIZED_EXACT_LABEL_SHORTCUT")
    if construction_missing_count:
        findings.append("SEMANTIC_CONSTRUCTION_MISSING")
    if target_original_in_query_count:
        findings.append("TARGET_ORIGINAL_PRESENT_IN_QUERY")
    if global_difference > 1:
        findings.append("REFUSAL_MODE_GLOBAL_IMBALANCE")
    if maximum_family_difference > 1:
        findings.append("REFUSAL_MODE_FAMILY_IMBALANCE")
    return {
        "status": "PASS" if not findings else "FAIL",
        "findings": findings,
        "baseline": "normalized_exact_query_to_evidence_span",
        "example_count": total,
        "normalized_exact_match_count": exact_match_count,
        "normalized_exact_match_answer_count": exact_match_answer_count,
        "decision_correct_count": decision_correct_count,
        "decision_accuracy": round(accuracy, 6),
        "answer_span_recovery_count": answer_span_recovery_count,
        "can_directly_recover_label_or_span": bool(
            accuracy > 0.5 or answer_span_recovery_count
        ),
        "semantic_construction_missing_count": (
            construction_missing_count
        ),
        "target_original_in_query_count": (
            target_original_in_query_count
        ),
        "refusal_mode_counts": {
            contradiction_key: refusal_mode_counts[contradiction_key],
            hidden_key: refusal_mode_counts[hidden_key],
        },
        "refusal_mode_global_difference": global_difference,
        "refusal_mode_maximum_family_difference": (
            maximum_family_difference
        ),
    }


def _independent_leakage_report(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    claims: dict[
        str,
        list[tuple[str, set[tuple[str, ...]]]],
    ] = defaultdict(list)
    prompt_hashes: dict[str, set[str]] = defaultdict(set)
    evidence_hashes: dict[str, set[str]] = defaultdict(set)
    prompt_markers = 0
    evidence_markers = 0
    assistant_messages = 0
    interface_missing = 0
    for row in rows:
        split = str(row["split"])
        claim = str(row["requested_claim"])
        claim_hash = _sha256_bytes(
            _normalized_text(claim).encode("utf-8")
        )
        claims[split].append((claim_hash, _word_ngrams(claim)))
        prompt = row.get("compiler_prompt")
        compiler_evidence = row.get("compiler_evidence")
        if not isinstance(prompt, Mapping) or (
            not isinstance(compiler_evidence, Sequence)
            or isinstance(compiler_evidence, (str, bytes))
        ):
            interface_missing += 1
        prompt_hashes[split].add(
            _sha256_bytes(canonical_json(prompt).encode("utf-8"))
        )
        evidence_hashes[split].add(
            _sha256_bytes(
                canonical_json(compiler_evidence).encode("utf-8")
            )
        )
        prompt_markers += _target_marker_count(prompt)
        evidence_markers += _target_marker_count(compiler_evidence)
        if isinstance(prompt, Mapping):
            messages = prompt.get("messages", [])
            if (
                isinstance(messages, Sequence)
                and not isinstance(messages, (str, bytes))
            ):
                assistant_messages += sum(
                    isinstance(message, Mapping)
                    and message.get("role") == "assistant"
                    for message in messages
                )
    exact_claims = 0
    exact_prompts = 0
    exact_evidence = 0
    maximum_jaccard = 0.0
    near_duplicates = 0
    for left_index, left in enumerate(NONBLIND_SPLITS):
        for right in NONBLIND_SPLITS[left_index + 1 :]:
            exact_claims += len(
                {item[0] for item in claims[left]}
                & {item[0] for item in claims[right]}
            )
            exact_prompts += len(
                prompt_hashes[left] & prompt_hashes[right]
            )
            exact_evidence += len(
                evidence_hashes[left] & evidence_hashes[right]
            )
            for _, left_grams in claims[left]:
                for _, right_grams in claims[right]:
                    score = _jaccard(left_grams, right_grams)
                    maximum_jaccard = max(maximum_jaccard, score)
                    if score >= _NEAR_DUPLICATE_THRESHOLD:
                        near_duplicates += 1
    findings: list[str] = []
    if exact_claims:
        findings.append("EXACT_CLAIM_OVERLAP")
    if exact_prompts:
        findings.append("EXACT_PROMPT_OVERLAP")
    if exact_evidence:
        findings.append("EXACT_COMPILER_EVIDENCE_OVERLAP")
    if near_duplicates:
        findings.append("NEAR_DUPLICATE_CLAIM_OVERLAP")
    if prompt_markers:
        findings.append("COMPILER_PROMPT_TARGET_LEAKAGE")
    if evidence_markers:
        findings.append("COMPILER_EVIDENCE_TARGET_LEAKAGE")
    if assistant_messages:
        findings.append("COMPILER_PROMPT_ASSISTANT_MESSAGE_LEAKAGE")
    if interface_missing:
        findings.append("COMPILER_INTERFACE_MISSING")
    shortcut = _independent_shortcut_audit(rows)
    if shortcut["status"] != "PASS":
        findings.append("NORMALIZED_EXACT_MATCH_SHORTCUT_AUDIT_FAILED")
    return {
        "schema": NONBLIND_LEAKAGE_SCHEMA,
        "status": "PASS" if not findings else "FAIL",
        "findings": findings,
        "near_duplicate_threshold": _NEAR_DUPLICATE_THRESHOLD,
        "exact_claim_overlap_count": exact_claims,
        "exact_prompt_overlap_count": exact_prompts,
        "exact_compiler_evidence_overlap_count": exact_evidence,
        "near_duplicate_claim_pair_count": near_duplicates,
        "compiler_prompt_target_marker_count": prompt_markers,
        "compiler_evidence_target_marker_count": evidence_markers,
        "compiler_prompt_assistant_message_count": assistant_messages,
        "compiler_interface_missing_count": interface_missing,
        "shortcut_audit_status": shortcut["status"],
        "shortcut_audit": shortcut,
        "maximum_cross_split_claim_jaccard": round(
            maximum_jaccard,
            6,
        ),
        "pointer_target_overlap_policy": (
            "allowed_by_design_compact_contract_not_content_leakage"
        ),
        "audited_splits": list(NONBLIND_SPLITS),
    }


def _validate_stored_audits(
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    rows: Sequence[Mapping[str, Any]],
    families: Sequence[Any],
    assignments: Mapping[str, str],
    semantic_audit: Mapping[str, Any],
    sources: SourceState,
) -> None:
    balance = _independent_balance_report(rows, assignments)
    family_integrity = _independent_family_integrity(
        rows,
        assignments,
    )
    group = _independent_group_report(families, assignments)
    leakage = _independent_leakage_report(rows)
    expected = {
        "balance_audit": balance,
        "group_isolation_audit": group,
        "content_leakage_audit": leakage,
        "semantic_inventory_audit": dict(semantic_audit),
    }
    for role, recomputed in expected.items():
        if artifacts[role] != recomputed:
            raise NonblindSFTAuditV7Error(
                f"stored {role} does not match independent recomputation"
            )
    if (
        balance.get("schema") != NONBLIND_BALANCE_SCHEMA
        or group.get("schema") != NONBLIND_GROUP_SCHEMA
        or leakage.get("schema") != NONBLIND_LEAKAGE_SCHEMA
        or any(item.get("status") != "PASS" for item in expected.values())
    ):
        raise NonblindSFTAuditV7Error(
            "one or more independently recomputed audits did not pass"
        )
    if (
        family_integrity.get("status") != "PASS"
        or balance.get("family_integrity") != family_integrity
    ):
        raise NonblindSFTAuditV7Error(
            "balance family-integrity binding is invalid"
        )

    _validate_preblind_commitment(
        artifacts["preblind_commitment"],
        sources=sources,
    )
    expected_report = {
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
            "balance": "PASS",
            "family_integrity": "PASS",
            "group_isolation": "PASS",
            "content_leakage": "PASS",
            "semantic_inventory": "PASS",
            "rag_authority_binding": (
                "PASS_RAG_MANIFEST_LICENSED_CHUNKS_BOUND"
            ),
        },
        "family_integrity": family_integrity,
        "claims": dict(_EXPECTED_CLAIMS),
    }
    if artifacts["build_report"] != expected_report:
        raise NonblindSFTAuditV7Error(
            "stored build report is not the exact passing v7 report"
        )


def _recheck_snapshot(expected: FileSnapshot, *, label: str) -> None:
    current = _snapshot_regular_file(
        expected.path,
        label=label,
        maximum_bytes=max(expected.bytes, _MAX_JSON_BYTES),
    )
    if (
        current.identity != expected.identity
        or current.bytes != expected.bytes
        or current.sha256 != expected.sha256
        or current.payload != expected.payload
    ):
        raise NonblindSFTAuditV7Error(
            f"{label} changed before receipt creation"
        )


def _load_and_validate_dataset(
    path: Path,
    *,
    label: str,
    authority: AuthorityInputs,
) -> DatasetState:
    root, files, manifest = _snapshot_fixed_dataset(path, label=label)
    _validate_manifest_contract(manifest, files)
    sources, families, semantic_map, semantic_audit = (
        _snapshot_declared_sources(manifest, authority=authority)
    )
    seed = sources.seed
    try:
        assignments = evidence.assign_family_splits(families, seed=seed)
    except (EvidenceSFTV6Error, OSError, ValueError) as exc:
        raise NonblindSFTAuditV7Error(
            f"source-family split reconstruction failed: {exc}"
        ) from exc
    rows = _parse_and_validate_rows(
        files,
        families=families,
        semantic_map=semantic_map,
        assignments=assignments,
    )
    artifacts = {
        role: _strict_json_object(
            files[role].payload,
            label=f"{label} {role}",
        )
        for role in _ARTIFACT_ROLES
    }
    _validate_stored_audits(
        artifacts,
        rows=rows,
        families=families,
        assignments=assignments,
        semantic_audit=semantic_audit,
        sources=sources,
    )
    return DatasetState(
        root=root,
        files=files,
        manifest=manifest,
        artifacts=artifacts,
        rows=rows,
        sources=sources,
    )


def _same_regular_file(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except OSError as exc:
        raise NonblindSFTAuditV7Error(
            "artifact file identity could not be verified"
        ) from exc


def _assert_double_build_identity(
    left: DatasetState,
    right: DatasetState,
) -> None:
    if left.root == right.root or _same_regular_file(left.root, right.root):
        raise NonblindSFTAuditV7Error(
            "the two build directories must be distinct"
        )
    for role in _ROLE_FILENAMES:
        left_file = left.files[role]
        right_file = right.files[role]
        if (
            left_file.identity[:2] == right_file.identity[:2]
            or _same_regular_file(left_file.path, right_file.path)
        ):
            raise NonblindSFTAuditV7Error(
                f"{role} is shared rather than independently materialized"
            )
        if (
            left_file.bytes != right_file.bytes
            or left_file.sha256 != right_file.sha256
            or left_file.payload != right_file.payload
        ):
            raise NonblindSFTAuditV7Error(
                f"{role} is not byte-for-byte reproducible"
            )
    for role in (
        "licensed_chunks",
        "rag_manifest",
        "semantic_inventory",
        "semantic_records",
        "nonblind_builder",
        "evidence_core",
    ):
        left_file = getattr(left.sources, role)
        right_file = getattr(right.sources, role)
        if (
            left_file.path != right_file.path
            or left_file.bytes != right_file.bytes
            or left_file.sha256 != right_file.sha256
            or left_file.payload != right_file.payload
        ):
            raise NonblindSFTAuditV7Error(
                f"the two builds do not bind identical {role} bytes"
            )
    if (
        left.sources.seed != right.sources.seed
        or left.sources.rag_manifest_id != right.sources.rag_manifest_id
    ):
        raise NonblindSFTAuditV7Error(
            "the two builds do not bind the same seed and RAG identity"
        )


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


class _WindowsFileInformation(ctypes.Structure):
    _fields_ = (
        ("attributes", ctypes.c_uint32),
        ("creation_time_low", ctypes.c_uint32),
        ("creation_time_high", ctypes.c_uint32),
        ("last_access_time_low", ctypes.c_uint32),
        ("last_access_time_high", ctypes.c_uint32),
        ("last_write_time_low", ctypes.c_uint32),
        ("last_write_time_high", ctypes.c_uint32),
        ("volume_serial_number", ctypes.c_uint32),
        ("file_size_high", ctypes.c_uint32),
        ("file_size_low", ctypes.c_uint32),
        ("number_of_links", ctypes.c_uint32),
        ("file_index_high", ctypes.c_uint32),
        ("file_index_low", ctypes.c_uint32),
    )


def _windows_kernel32() -> Any:
    if os.name != "nt":
        raise NonblindSFTAuditV7Error(
            "Windows stable directory handles are unavailable"
        )
    return ctypes.WinDLL("kernel32", use_last_error=True)


def _windows_close_handle(handle: int) -> None:
    kernel32 = _windows_kernel32()
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (ctypes.c_void_p,)
    close_handle.restype = ctypes.c_int
    if not close_handle(ctypes.c_void_p(handle)):
        raise NonblindSFTAuditV7Error(
            "stable directory handle could not be closed"
        )


def _windows_identity_from_handle(
    handle: int,
    *,
    label: str,
) -> tuple[int, int]:
    kernel32 = _windows_kernel32()
    get_information = kernel32.GetFileInformationByHandle
    get_information.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(_WindowsFileInformation),
    )
    get_information.restype = ctypes.c_int
    information = _WindowsFileInformation()
    if not get_information(
        ctypes.c_void_p(handle),
        ctypes.byref(information),
    ):
        raise NonblindSFTAuditV7Error(
            f"{label} stable handle identity cannot be read"
        )
    file_attribute_directory = 0x00000010
    file_attribute_reparse_point = 0x00000400
    if (
        not information.attributes & file_attribute_directory
        or information.attributes & file_attribute_reparse_point
    ):
        raise NonblindSFTAuditV7Error(
            f"{label} stable handle is not a regular directory"
        )
    file_index = (
        int(information.file_index_high) << 32
    ) | int(information.file_index_low)
    return int(information.volume_serial_number), file_index


def _normalize_windows_handle_path(value: str) -> Path:
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return Path(value)


def _windows_path_from_handle(handle: int, *, label: str) -> Path:
    kernel32 = _windows_kernel32()
    get_name = kernel32.GetFinalPathNameByHandleW
    get_name.argtypes = (
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
    )
    get_name.restype = ctypes.c_uint32
    size = 512
    for _ in range(3):
        buffer = ctypes.create_unicode_buffer(size)
        written = int(
            get_name(
                ctypes.c_void_p(handle),
                buffer,
                size,
                0,
            )
        )
        if written == 0:
            raise NonblindSFTAuditV7Error(
                f"{label} stable path cannot be recovered"
            )
        if written < size:
            return _normalize_windows_handle_path(buffer.value)
        size = written + 1
    raise NonblindSFTAuditV7Error(
        f"{label} stable path exceeds the recovery limit"
    )


def _open_stable_directory(
    path: Path,
    *,
    expected_identity: tuple[int, int],
    label: str,
) -> StableDirectoryHandle:
    current, identity = _snapshot_directory(path, label=label)
    if identity != expected_identity:
        raise NonblindSFTAuditV7Error(
            f"{label} identity changed before stable open"
        )
    if os.name != "nt":
        return StableDirectoryHandle(
            path=current,
            identity=identity,
            windows_handle=None,
        )

    kernel32 = _windows_kernel32()
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    )
    create_file.restype = ctypes.c_void_p
    file_read_attributes = 0x00000080
    share_read_write_delete = 0x00000001 | 0x00000002 | 0x00000004
    open_existing = 3
    open_reparse_point = 0x00200000
    backup_semantics = 0x02000000
    raw_handle = create_file(
        os.fspath(current),
        file_read_attributes,
        share_read_write_delete,
        None,
        open_existing,
        open_reparse_point | backup_semantics,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if raw_handle in (None, invalid_handle):
        raise NonblindSFTAuditV7Error(
            f"{label} stable handle cannot be opened"
        )
    handle = int(raw_handle)
    try:
        if _windows_identity_from_handle(handle, label=label) != identity:
            raise NonblindSFTAuditV7Error(
                f"{label} identity changed during stable open"
            )
        recovered = _windows_path_from_handle(handle, label=label)
        recovered, recovered_identity = _snapshot_directory(
            recovered,
            label=label,
        )
        if recovered != current or recovered_identity != identity:
            raise NonblindSFTAuditV7Error(
                f"{label} path changed during stable open"
            )
    except BaseException:
        _windows_close_handle(handle)
        raise
    return StableDirectoryHandle(
        path=current,
        identity=identity,
        windows_handle=handle,
    )


def _snapshot_directory(
    path: Path,
    *,
    label: str,
) -> tuple[Path, tuple[int, int]]:
    lexical = _absolute_lexical(path)
    _assert_no_link_components(lexical, label=label)
    try:
        value = lexical.lstat()
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise NonblindSFTAuditV7Error(
            f"{label} is unavailable"
        ) from exc
    if (
        not stat.S_ISDIR(value.st_mode)
        or stat.S_ISLNK(value.st_mode)
        or _is_reparse_point(value)
    ):
        raise NonblindSFTAuditV7Error(
            f"{label} must be a regular directory"
        )
    return resolved, _directory_identity(value)


def _recheck_output_anchor(
    plan: OutputPlan,
    *,
    stable: StableDirectoryHandle | None = None,
) -> None:
    if stable is None:
        current, identity = _snapshot_directory(
            plan.anchor_parent,
            label="audit output parent",
        )
    else:
        current = stable.current_path(label="audit output parent")
        identity = stable.identity
    if (
        current != plan.anchor_parent
        or identity != plan.anchor_parent_identity
    ):
        raise NonblindSFTAuditV7Error(
            "audit output parent identity changed before publication"
        )
    if any(
        _path_is_within(plan.final_path, root)
        for root in plan.dataset_roots
    ):
        raise NonblindSFTAuditV7Error(
            "audit output containment changed before publication"
        )


def _recheck_publication_parent(
    path: Path,
    *,
    expected_identity: tuple[int, int],
    plan: OutputPlan,
    anchor: StableDirectoryHandle | None = None,
    publication: StableDirectoryHandle | None = None,
) -> None:
    _recheck_output_anchor(plan, stable=anchor)
    if publication is None:
        current, identity = _snapshot_directory(
            path,
            label="audit publication directory",
        )
    else:
        current = publication.current_path(
            label="audit publication directory"
        )
        identity = publication.identity
    if current != path or identity != expected_identity:
        raise NonblindSFTAuditV7Error(
            "audit publication directory identity changed"
        )
    if plan.final_path.parent != current:
        raise NonblindSFTAuditV7Error(
            "audit publication containment changed"
        )


def _plan_output(
    requested: Path,
    *,
    dataset_roots: Sequence[Path],
) -> OutputPlan:
    lexical = _absolute_lexical(requested)
    _assert_canonical_lexical_path(lexical, label="audit output")
    dataset_root_tuple = tuple(dataset_roots)
    if lexical.suffix.lower() == ".json":
        if os.path.lexists(lexical.parent):
            parent, parent_identity = _snapshot_directory(
                lexical.parent,
                label="audit output parent",
            )
            final = parent / lexical.name
            new_directory = None
        else:
            parent, parent_identity = _snapshot_directory(
                lexical.parent.parent,
                label="audit output parent anchor",
            )
            new_directory = parent / lexical.parent.name
            _assert_no_link_components(
                new_directory,
                label="audit output parent",
                allow_missing_leaf=True,
            )
            final = new_directory / lexical.name
    else:
        parent, parent_identity = _snapshot_directory(
            lexical.parent,
            label="audit output parent",
        )
        new_directory = parent / lexical.name
        if os.path.lexists(new_directory):
            raise NonblindSFTAuditV7Error(
                "audit output directory already exists"
            )
        final = new_directory / AUDIT_FILENAME
    if new_directory is None:
        _assert_no_link_components(
            final,
            label="audit output",
            allow_missing_leaf=True,
        )
    if os.path.lexists(final):
        raise NonblindSFTAuditV7Error(
            "audit output already exists"
        )
    absolute_final = final.absolute()
    if any(
        _path_is_within(absolute_final, root)
        for root in dataset_root_tuple
    ):
        raise NonblindSFTAuditV7Error(
            "audit output must be outside all dataset directories"
        )
    return OutputPlan(
        final_path=absolute_final,
        new_directory=new_directory.absolute()
        if new_directory is not None
        else None,
        anchor_parent=parent,
        anchor_parent_identity=parent_identity,
        dataset_roots=dataset_root_tuple,
    )


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _unlink_stable_publication_child(
    directory: StableDirectoryHandle,
    *,
    filename: str,
    expected_identity: tuple[int, int],
    label: str,
) -> None:
    for _ in range(4):
        parent = directory.current_path(label=f"{label} parent")
        candidate = parent / filename
        try:
            snapshot = _snapshot_regular_file(
                candidate,
                label=label,
                maximum_bytes=_MAX_JSON_BYTES,
            )
        except NonblindSFTAuditV7Error as exc:
            if not os.path.lexists(candidate):
                return
            raise NonblindSFTAuditV7Error(
                f"{label} cannot be safely identified for cleanup"
            ) from exc
        if snapshot.identity[:2] != expected_identity:
            raise NonblindSFTAuditV7Error(
                f"{label} identity differs from this publication"
            )
        try:
            candidate.unlink()
        except FileNotFoundError:
            continue
        current = directory.current_path(label=f"{label} parent")
        if not os.path.lexists(current / filename):
            return
    raise NonblindSFTAuditV7Error(
        f"{label} could not be removed after bounded stable retries"
    )


def _atomic_exclusive_write(plan: OutputPlan, payload: bytes) -> Path:
    created_directory = False
    published = False
    temporary: Path | None = None
    temporary_identity: tuple[int, int] | None = None
    anchor: StableDirectoryHandle | None = None
    publication: StableDirectoryHandle | None = None
    created_directory_path: Path | None = None
    try:
        anchor = _open_stable_directory(
            plan.anchor_parent,
            expected_identity=plan.anchor_parent_identity,
            label="audit output parent",
        )
        _recheck_output_anchor(plan, stable=anchor)
        if plan.new_directory is not None:
            anchor_path = anchor.current_path(label="audit output parent")
            created_directory_path = anchor_path / plan.new_directory.name
            created_directory_path.mkdir()
            created_directory = True
            parent, parent_identity = _snapshot_directory(
                created_directory_path,
                label="audit publication directory",
            )
            publication = _open_stable_directory(
                parent,
                expected_identity=parent_identity,
                label="audit publication directory",
            )
        else:
            parent = anchor.current_path(label="audit publication directory")
            parent_identity = anchor.identity
            publication = anchor
        _recheck_publication_parent(
            parent,
            expected_identity=parent_identity,
            plan=plan,
            anchor=anchor,
            publication=publication,
        )
        temporary = parent / (
            f".{plan.final_path.name}.{uuid.uuid4().hex}.tmp"
        )
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
        )
        descriptor = os.open(os.fspath(temporary), flags, 0o600)
        temporary_identity = _stat_identity(os.fstat(descriptor))[:2]
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            raise
        _recheck_publication_parent(
            parent,
            expected_identity=parent_identity,
            plan=plan,
            anchor=anchor,
            publication=publication,
        )
        try:
            os.link(
                os.fspath(temporary),
                os.fspath(plan.final_path),
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise NonblindSFTAuditV7Error(
                "audit output already exists"
            ) from exc
        published = True
        _recheck_publication_parent(
            parent,
            expected_identity=parent_identity,
            plan=plan,
            anchor=anchor,
            publication=publication,
        )
        temporary.unlink()
        temporary = None
        persisted = _snapshot_regular_file(
            plan.final_path,
            label="persisted audit receipt",
            maximum_bytes=_MAX_JSON_BYTES,
        )
        if persisted.payload != payload:
            raise NonblindSFTAuditV7Error(
                "persisted audit receipt bytes differ"
            )
        return persisted.path
    except BaseException as exc:
        cleanup_errors: list[str] = []
        if publication is not None and temporary_identity is not None:
            if published:
                try:
                    _unlink_stable_publication_child(
                        publication,
                        filename=plan.final_path.name,
                        expected_identity=temporary_identity,
                        label="published audit receipt",
                    )
                except NonblindSFTAuditV7Error as cleanup_exc:
                    cleanup_errors.append(str(cleanup_exc))
            if temporary is not None:
                try:
                    _unlink_stable_publication_child(
                        publication,
                        filename=temporary.name,
                        expected_identity=temporary_identity,
                        label="temporary audit receipt",
                    )
                except NonblindSFTAuditV7Error as cleanup_exc:
                    cleanup_errors.append(str(cleanup_exc))
        elif temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                cleanup_errors.append(str(cleanup_exc))
        elif published:
            try:
                plan.final_path.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                cleanup_errors.append(str(cleanup_exc))
        if created_directory:
            recovered_directory: Path | None = None
            try:
                if publication is not None:
                    recovered_directory = publication.current_path(
                        label="audit publication directory cleanup"
                    )
                elif created_directory_path is not None:
                    recovered_directory = created_directory_path
                if publication is not None and publication is not anchor:
                    publication.close()
                    publication = None
                if recovered_directory is not None:
                    recovered_directory.rmdir()
            except (OSError, NonblindSFTAuditV7Error) as cleanup_exc:
                cleanup_errors.append(str(cleanup_exc))
        if cleanup_errors:
            raise NonblindSFTAuditV7Error(
                "audit publication failed and stable cleanup was incomplete: "
                + "; ".join(cleanup_errors)
            ) from exc
        raise
    finally:
        if publication is not None and publication is not anchor:
            publication.close()
        if anchor is not None:
            anchor.close()


def _implementation_inventory(
    dataset: DatasetState,
    *,
    runner_path: Path | None,
) -> tuple[dict[str, Any], tuple[FileSnapshot, ...]]:
    auditor = _snapshot_regular_file(
        Path(__file__).resolve(strict=True),
        label="independent auditor source",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    snapshots = [auditor]
    inventory: dict[str, Any] = {
        "nonblind_builder": {
            "filename": dataset.sources.nonblind_builder.path.name,
            "bytes": dataset.sources.nonblind_builder.bytes,
            "sha256": dataset.sources.nonblind_builder.sha256,
        },
        "evidence_core": {
            "filename": dataset.sources.evidence_core.path.name,
            "bytes": dataset.sources.evidence_core.bytes,
            "sha256": dataset.sources.evidence_core.sha256,
        },
        "independent_auditor": {
            "filename": auditor.path.name,
            "bytes": auditor.bytes,
            "sha256": auditor.sha256,
        },
    }
    if runner_path is not None:
        runner = _snapshot_regular_file(
            Path(runner_path).resolve(strict=True),
            label="audit CLI source",
            maximum_bytes=_MAX_JSON_BYTES,
        )
        snapshots.append(runner)
        inventory["audit_cli"] = {
            "filename": runner.path.name,
            "bytes": runner.bytes,
            "sha256": runner.sha256,
        }
    return inventory, tuple(snapshots)


def _artifact_inventory(
    dataset: DatasetState,
    *,
    byte_identical: bool | None,
) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for role, filename in _ROLE_FILENAMES.items():
        snapshot = dataset.files[role]
        item: dict[str, Any] = {
            "path": filename,
            "bytes": snapshot.bytes,
            "sha256": snapshot.sha256,
            "safely_parsed": True,
        }
        if role in _SPLIT_ROLES:
            item["count"] = EXPECTED_NONBLIND_SPLIT_COUNTS[role]
        if byte_identical is not None:
            item["byte_identical"] = byte_identical
        files[role] = item
    return files


def _source_inventory(sources: SourceState) -> dict[str, Any]:
    return {
        "licensed_chunks": {
            "bytes": sources.licensed_chunks.bytes,
            "sha256": sources.licensed_chunks.sha256,
        },
        "rag_manifest": {
            "schema": "icmat.rag.manifest.v2",
            "manifest_id": sources.rag_manifest_id,
            "bytes": sources.rag_manifest.bytes,
            "sha256": sources.rag_manifest.sha256,
            "authority_binding_sha256": _sha256_bytes(
                canonical_json(
                    sources.rag_authority_binding
                ).encode("utf-8")
            ),
        },
        "semantic_inventory": {
            "schema": ACCEPTED_INVENTORY_SCHEMA,
            "bytes": sources.semantic_inventory.bytes,
            "sha256": sources.semantic_inventory.sha256,
        },
        "semantic_records": {
            "schema": SEMANTIC_QUERY_SCHEMA,
            "bytes": sources.semantic_records.bytes,
            "sha256": sources.semantic_records.sha256,
        },
        "seed_sha256": _sha256_bytes(sources.seed.encode("utf-8")),
        "split_algorithm_version": SPLIT_ALGORITHM_VERSION,
    }


def _receipt_body(
    *,
    mode: str,
    status: str,
    dataset: DatasetState,
    implementation: Mapping[str, Any],
    double_build: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema": AUDIT_SCHEMA,
        "version": AUDIT_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "mode": mode,
        "status": status,
        "audit_passed": True,
        "dataset_contract": {
            "manifest_schema": NONBLIND_MANIFEST_SCHEMA,
            "manifest_version": NONBLIND_BUILDER_VERSION,
            "manifest_status": (
                "NONBLIND_DATASET_BUILT_PREBLIND_COMMITTED"
            ),
            "dataset_schema": DATASET_SCHEMA,
            "counts": {
                "total": EXPECTED_NONBLIND_TOTAL,
                "splits": dict(EXPECTED_NONBLIND_SPLIT_COUNTS),
            },
            "pointer_rows_structurally_revalidated": len(dataset.rows),
            "source_family_isolation_recomputed": True,
            "content_leakage_recomputed": True,
            "semantic_inventory_and_records_rebound": True,
            "preblind_commitment_recomputed": True,
        },
        "artifact_inventory": {
            "fixed_whitelist_file_count": len(_ROLE_FILENAMES),
            "directory_enumerated": False,
            "files": _artifact_inventory(
                dataset,
                byte_identical=True if mode == "compare" else None,
            ),
        },
        "source_bindings": _source_inventory(dataset.sources),
        "implementation": dict(implementation),
        "double_build": dict(double_build)
        if double_build is not None
        else None,
        "reserved_asset_boundary": {
            "path_accessed": False,
            "path_discovered": False,
            "read": False,
            "hashed": False,
            "stat_called": False,
            "directory_scanned": False,
            "content_disclosed": False,
            "expected_reserved_count_only": EXPECTED_BLIND_COUNT,
        },
        "claims": {
            "training_or_checkpoint_selection_performed": False,
            "model_quality_authorized": False,
            "x5_contacted_or_verified": False,
        },
        "pre_write_evidence_recheck": True,
    }


def _finalize_receipt(body: Mapping[str, Any]) -> dict[str, Any]:
    receipt = {
        **body,
        "canonical_digest_sha256": _sha256_bytes(
            canonical_json(body).encode("utf-8")
        ),
    }
    receipt["receipt_payload_sha256"] = _sha256_bytes(
        canonical_json(receipt).encode("utf-8")
    )
    return receipt


def _all_input_snapshots(
    *datasets: DatasetState,
    implementation: Sequence[FileSnapshot],
) -> tuple[FileSnapshot, ...]:
    result: list[FileSnapshot] = []
    for dataset in datasets:
        result.extend(dataset.files.values())
        result.extend(
            (
                dataset.sources.licensed_chunks,
                dataset.sources.rag_manifest,
                dataset.sources.semantic_inventory,
                dataset.sources.semantic_records,
                dataset.sources.nonblind_builder,
                dataset.sources.evidence_core,
            )
        )
    result.extend(implementation)
    unique: dict[Path, FileSnapshot] = {}
    for snapshot in result:
        unique.setdefault(snapshot.path, snapshot)
    return tuple(unique.values())


def audit_nonblind_dataset_v7(
    *,
    dataset: Path,
    licensed_chunks: Path,
    rag_manifest: Path,
    semantic_inventory: Path,
    output: Path,
    runner_path: Path | None = None,
) -> dict[str, Any]:
    authority = _snapshot_authority_inputs(
        licensed_chunks=licensed_chunks,
        rag_manifest=rag_manifest,
        semantic_inventory=semantic_inventory,
    )
    state = _load_and_validate_dataset(
        dataset,
        label="dataset",
        authority=authority,
    )
    plan = _plan_output(output, dataset_roots=(state.root,))
    implementation, implementation_snapshots = _implementation_inventory(
        state,
        runner_path=runner_path,
    )
    body = _receipt_body(
        mode="audit",
        status=AUDIT_PASS_STATUS,
        dataset=state,
        implementation=implementation,
        double_build=None,
    )
    receipt = _finalize_receipt(body)
    for snapshot in _all_input_snapshots(
        state,
        implementation=implementation_snapshots,
    ):
        _recheck_snapshot(snapshot, label="pre-write audit evidence")
    payload = _json_bytes(receipt)
    written = _atomic_exclusive_write(plan, payload)
    return {
        "status": AUDIT_PASS_STATUS,
        "audit_passed": True,
        "path": written.as_posix(),
        "sha256": _sha256_bytes(payload),
        "canonical_digest_sha256": receipt[
            "canonical_digest_sha256"
        ],
        "receipt": receipt,
    }


def compare_nonblind_datasets_v7(
    *,
    dataset_a: Path,
    dataset_b: Path,
    licensed_chunks: Path,
    rag_manifest: Path,
    semantic_inventory: Path,
    output: Path,
    runner_path: Path | None = None,
) -> dict[str, Any]:
    authority = _snapshot_authority_inputs(
        licensed_chunks=licensed_chunks,
        rag_manifest=rag_manifest,
        semantic_inventory=semantic_inventory,
    )
    left = _load_and_validate_dataset(
        dataset_a,
        label="dataset_a",
        authority=authority,
    )
    right = _load_and_validate_dataset(
        dataset_b,
        label="dataset_b",
        authority=authority,
    )
    _assert_double_build_identity(left, right)
    plan = _plan_output(
        output,
        dataset_roots=(left.root, right.root),
    )
    implementation, implementation_snapshots = _implementation_inventory(
        left,
        runner_path=runner_path,
    )
    double_build = {
        "directories_distinct": True,
        "artifact_file_identities_distinct": True,
        "fixed_whitelist_file_count": len(_ROLE_FILENAMES),
        "all_whitelist_files_byte_identical": True,
        "dataset_a_root_fingerprint_sha256": _sha256_bytes(
            os.fspath(left.root).encode("utf-8")
        ),
        "dataset_b_root_fingerprint_sha256": _sha256_bytes(
            os.fspath(right.root).encode("utf-8")
        ),
        "manifest_sha256": left.files["manifest"].sha256,
    }
    body = _receipt_body(
        mode="compare",
        status=COMPARE_PASS_STATUS,
        dataset=left,
        implementation=implementation,
        double_build=double_build,
    )
    receipt = _finalize_receipt(body)
    for snapshot in _all_input_snapshots(
        left,
        right,
        implementation=implementation_snapshots,
    ):
        _recheck_snapshot(snapshot, label="pre-write compare evidence")
    payload = _json_bytes(receipt)
    written = _atomic_exclusive_write(plan, payload)
    return {
        "status": COMPARE_PASS_STATUS,
        "audit_passed": True,
        "byte_identical": True,
        "path": written.as_posix(),
        "sha256": _sha256_bytes(payload),
        "canonical_digest_sha256": receipt[
            "canonical_digest_sha256"
        ],
        "receipt": receipt,
    }


__all__ = [
    "AUDIT_FILENAME",
    "AUDIT_PASS_STATUS",
    "AUDIT_SCHEMA",
    "AUDIT_VERSION",
    "COMPARE_PASS_STATUS",
    "ERROR_STATUS",
    "NonblindSFTAuditV7Error",
    "audit_nonblind_dataset_v7",
    "compare_nonblind_datasets_v7",
]
