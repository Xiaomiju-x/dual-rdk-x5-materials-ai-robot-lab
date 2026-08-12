from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from icmat_foundry.llm import evidence_sft_v6 as evidence
from icmat_foundry.llm import nonblind_sft_audit_v7 as v7audit
from icmat_foundry.llm import nonblind_sft_v7 as v7builder
from icmat_foundry.llm import nonblind_sft_v8 as v8builder
from icmat_foundry.llm import semantic_queries_v7 as semantic
from icmat_foundry.llm.evidence_sft_v6 import (
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
    TASKS,
    TRAINING_SPLITS,
    canonical_json,
)

AUDIT_SCHEMA = "icmat_evidence_pointer_nonblind_independent_audit.v8"
AUDIT_VERSION = "icmat-evidence-nonblind-independent-audit-v8.0.0"
AUDIT_FILENAME = "independent_audit.nonblind.v8.json"
AUDIT_PASS_STATUS = "PASS_NONBLIND_V8_INDEPENDENT_AUDIT"
COMPARE_PASS_STATUS = "PASS_NONBLIND_V8_DOUBLE_BUILD_BYTE_IDENTICAL"
ERROR_STATUS = "FAILED_NO_NONBLIND_V8_AUDIT"

_MAX_JSON_BYTES = 32 * 1024 * 1024
_MAX_JSONL_BYTES = 96 * 1024 * 1024
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROTECTED_SOURCE_TOKENS = frozenset({"blind", "reserved", "sealed", "calibration"})
_SEALED_FALSE = {
    "read": False,
    "hashed": False,
    "path_discovered": False,
}
_EXPECTED_SPLIT_COUNTS = {
    "train": 250,
    "validation": 150,
    "calibration": 150,
}
_EXPECTED_FAMILY_SPLIT_COUNTS = {
    "train": 5,
    "validation": 3,
    "calibration": 3,
}
_EXPECTED_ANSWER_SPLIT_COUNTS = {split: count // 2 for split, count in _EXPECTED_SPLIT_COUNTS.items()}
_EXPECTED_TOTAL = 550
_EXPECTED_ANSWERS = 275
_EXPECTED_FAMILIES = 11
_TARGET_ENTAILMENT_MIN = 0.90
_NON_TARGET_ENTAILMENT_MAX = 0.10

ROLE_FILENAMES: dict[str, str] = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "calibration": "calibration.jsonl",
    "balance_audit": "balance_audit.nonblind.v8.json",
    "group_isolation_audit": "group_isolation_audit.nonblind.v8.json",
    "content_leakage_audit": "content_leakage_audit.nonblind.v8.json",
    "semantic_binding_audit": "semantic_binding_audit.v8.json",
    "nli_unique_support_audit": "nli_unique_support_audit.v8.json",
    "repair_manifest": "repair_manifest.v8.json",
    "preblind_commitment": "preblind_commitment.v8.json",
    "build_report": "build_report.nonblind.v8.json",
    "manifest": "manifest.nonblind.v8.json",
}
_SPLIT_ROLES = tuple(NONBLIND_SPLITS)
_ARTIFACT_ROLES = (
    "balance_audit",
    "group_isolation_audit",
    "content_leakage_audit",
    "semantic_binding_audit",
    "nli_unique_support_audit",
    "repair_manifest",
    "preblind_commitment",
    "build_report",
)
_MANIFEST_FIELDS = frozenset(
    {
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
        "input_commitment_sha256",
        "output_content_sha256",
        "builder",
        "nli_unique_support",
        "counts",
        "pointer_contract",
        "compiler_input_contract",
        "external_answer_contract",
        "training_boundary",
        "sealed_blind_access",
        "claims",
    }
)
_SOURCE_FILENAMES = {
    "licensed_chunks": "licensed_chunks.v1.jsonl",
    "rag_manifest": "manifest.v2.json",
    "licensed_source_catalog": "licensed_source_catalog.v2.json",
    "semantic_inventory": "accepted_inventory.v7.json",
    "semantic_records": "records.v7.jsonl",
    "semantic_requests": "requests.v7.jsonl",
    "semantic_request_manifest": "request_manifest.v7.json",
}
_MANIFEST_SOURCE_ROLES = (
    "licensed_chunks",
    "rag_manifest",
    "semantic_inventory",
    "semantic_records",
    "semantic_requests",
    "semantic_request_manifest",
)
_CODE_ROLES = {
    "nonblind_v8_module": Path(v8builder.__file__),
    "nonblind_v7_module": Path(v7builder.__file__),
    "evidence_core": Path(evidence.__file__),
    "semantic_core": Path(semantic.__file__),
}
_NLI_IDENTITY_FIELDS = (
    "repo_id",
    "revision",
    "license_name",
    "model_tree_sha256",
    "model_receipt_sha256",
    "model_file_count",
    "model_total_bytes",
    "local_files_only",
)
_EXPECTED_CLAIMS = {
    "nonblind_only": True,
    "manual_jsonl_editing": False,
    "target_passage_modified": False,
    "training_authorized_splits": list(TRAINING_SPLITS),
    "calibration_for_training": False,
    "production_connected": False,
    "x5_deployed": False,
}
_REQUEST_FIELDS = frozenset(
    {
        "schema",
        "source_id",
        "source_record_sha256",
        "source_manifest_authority",
        "source_asset_sha256",
        "source_asset_uri",
        "namespace",
        "source_title",
        "source_uri",
        "license_id",
        "original_sentence",
        "original_sha256",
        "chunk_ids",
        "locators",
        "constraints",
        "request_id",
        "request_sha256",
    }
)
_REQUEST_CONSTRAINTS = {
    "paraphrase_must_not_normalize_to_source": True,
    "paraphrase_token_jaccard_min": 0.25,
    "paraphrase_token_jaccard_max": 0.88,
    "preserve_all_numbers_units_chemical_formulas": True,
    "contradiction_exactly_one_controlled_mutation": [
        "entity_swap",
        "numeric_change",
        "polarity_flip",
    ],
    "ground_truth_is_licensed_original_only": True,
}

FileSnapshot = v7audit.FileSnapshot
_snapshot_regular_file = v7audit._snapshot_regular_file  # noqa: SLF001
_strict_json = v7audit._strict_json  # noqa: SLF001
_strict_json_object = v7audit._strict_json_object  # noqa: SLF001
_validate_sha256 = v7audit._validate_sha256  # noqa: SLF001
_validate_declared_path = v7audit._validate_declared_path  # noqa: SLF001
_snapshot_authority_file = v7audit._snapshot_authority_file  # noqa: SLF001
_independent_rag_binding = v7audit._independent_rag_binding  # noqa: SLF001
_load_families = v7audit._load_licensed_families_snapshot  # noqa: SLF001
_load_semantics = v7audit._load_semantic_inventory_snapshots  # noqa: SLF001
_validate_rows_v7 = v7audit._parse_and_validate_rows  # noqa: SLF001
_balance_report_v7 = v7audit._independent_balance_report  # noqa: SLF001
_group_report_v7 = v7audit._independent_group_report  # noqa: SLF001
_leakage_report_v7 = v7audit._independent_leakage_report  # noqa: SLF001
_recheck_snapshot_v7 = v7audit._recheck_snapshot  # noqa: SLF001
_assert_no_link_components = v7audit._assert_no_link_components  # noqa: SLF001
_assert_unprotected = v7audit._assert_lexically_unprotected  # noqa: SLF001


class NonblindSFTAuditV8Error(ValueError):
    pass


@dataclass(frozen=True)
class AuthorityState:
    files: Mapping[str, FileSnapshot]
    code: Mapping[str, FileSnapshot]
    nli: Mapping[str, Any]
    rag_manifest_id: str
    rag_binding: Mapping[str, Any]
    families: tuple[Any, ...]
    semantic_map: Mapping[Any, Any]
    semantic_audit: Mapping[str, Any]
    request_binding: Mapping[str, Any]
    seed: str


@dataclass(frozen=True)
class DatasetState:
    root: Path
    root_identity: tuple[int, int]
    files: Mapping[str, FileSnapshot]
    manifest: Mapping[str, Any]
    artifacts: Mapping[str, Mapping[str, Any]]
    rows: tuple[Mapping[str, Any], ...]
    authorities: AuthorityState


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_reparse(value: os.stat_result) -> bool:
    attributes = int(getattr(value, "st_file_attributes", 0))
    marker = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return bool(marker and attributes & marker)


def _directory_identity(value: os.stat_result) -> tuple[int, int]:
    return int(value.st_dev), int(value.st_ino)


def _absolute(path: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


def _ensure_no_hardlink(path: Path, *, label: str) -> None:
    metadata = path.lstat()
    if int(getattr(metadata, "st_nlink", 1)) != 1:
        raise NonblindSFTAuditV8Error(f"{label} must not be a hardlink")


def _snapshot_strict_file(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
) -> FileSnapshot:
    snapshot = _snapshot_regular_file(
        path,
        label=label,
        maximum_bytes=maximum_bytes,
    )
    _ensure_no_hardlink(snapshot.path, label=label)
    return snapshot


def _safe_dataset_root(path: Path, *, label: str) -> tuple[Path, tuple[int, int]]:
    lexical = _absolute(path)
    _assert_no_link_components(lexical, label=label)
    try:
        metadata = lexical.lstat()
        root = lexical.resolve(strict=True)
    except OSError as exc:
        raise NonblindSFTAuditV8Error(f"{label} directory is missing") from exc
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
        raise NonblindSFTAuditV8Error(f"{label} must be a regular directory without link/reparse")
    return root, _directory_identity(metadata)


def _scan_fixed_inventory(
    root: Path,
    *,
    root_identity: tuple[int, int],
    label: str,
) -> None:
    expected = set(ROLE_FILENAMES.values())
    try:
        with os.scandir(root) as entries:
            observed = {entry.name for entry in entries}
    except OSError as exc:
        raise NonblindSFTAuditV8Error(f"{label} inventory cannot be read") from exc
    if observed != expected:
        raise NonblindSFTAuditV8Error(f"{label} does not match the exact 12-file whitelist")
    current = root.lstat()
    if (
        not stat.S_ISDIR(current.st_mode)
        or stat.S_ISLNK(current.st_mode)
        or _is_reparse(current)
        or _directory_identity(current) != root_identity
    ):
        raise NonblindSFTAuditV8Error(f"{label} directory identity changed")


def _snapshot_dataset_files(
    path: Path,
    *,
    label: str,
) -> tuple[
    Path,
    tuple[int, int],
    dict[str, FileSnapshot],
    dict[str, Any],
]:
    root, identity = _safe_dataset_root(path, label=label)
    _scan_fixed_inventory(
        root,
        root_identity=identity,
        label=label,
    )
    files: dict[str, FileSnapshot] = {}
    identities: set[tuple[int, int]] = set()
    for role, filename in ROLE_FILENAMES.items():
        maximum = _MAX_JSONL_BYTES if role in _SPLIT_ROLES else _MAX_JSON_BYTES
        snapshot = _snapshot_strict_file(
            root / filename,
            label=f"{label} {role}",
            maximum_bytes=maximum,
        )
        file_identity = snapshot.identity[:2]
        if file_identity in identities:
            raise NonblindSFTAuditV8Error(f"{label} contains aliased or hardlinked fixed members")
        identities.add(file_identity)
        files[role] = snapshot
    _scan_fixed_inventory(
        root,
        root_identity=identity,
        label=label,
    )
    manifest = _strict_json_object(
        files["manifest"].payload,
        label=f"{label} manifest",
    )
    return root, identity, files, manifest


def _validate_receipt(
    value: Any,
    *,
    snapshot: FileSnapshot,
    expected_path: str,
    expected_count: int | None,
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        raise NonblindSFTAuditV8Error(f"{label} receipt is missing")
    keys = {"path", "sha256", "bytes"}
    if expected_count is not None:
        keys.add("count")
    if set(value) != keys:
        raise NonblindSFTAuditV8Error(f"{label} receipt fields are not exact")
    if value.get("path") != expected_path:
        raise NonblindSFTAuditV8Error(f"{label} receipt path is outside the fixed whitelist")
    if (
        _validate_sha256(
            value.get("sha256"),
            label=f"{label} receipt SHA-256",
        )
        != snapshot.sha256
        or value.get("bytes") != snapshot.bytes
        or not _is_integer(value.get("bytes"))
    ):
        raise NonblindSFTAuditV8Error(f"{label} receipt SHA-256/size mismatch")
    if expected_count is not None and (
        not _is_integer(value.get("count")) or value.get("count") != expected_count
    ):
        raise NonblindSFTAuditV8Error(f"{label} receipt count mismatch")


def _expected_nli_identity() -> dict[str, Any]:
    return {
        "repo_id": semantic.PINNED_NLI_REPO_ID,
        "revision": semantic.PINNED_NLI_REVISION,
        "license_name": semantic.PINNED_NLI_LICENSE,
        "model_tree_sha256": semantic.PINNED_NLI_MODEL_TREE_SHA256,
        "model_receipt_sha256": semantic.PINNED_NLI_RECEIPT_SHA256,
        "model_file_count": semantic.PINNED_NLI_FILE_COUNT,
        "model_total_bytes": semantic.PINNED_NLI_TOTAL_BYTES,
        "local_files_only": True,
    }


def _validate_nli_asset(model_dir: Path) -> dict[str, Any]:
    raw = os.fspath(model_dir)
    _assert_unprotected(raw, label="NLI model source")
    _assert_no_link_components(
        _absolute(model_dir),
        label="NLI model source",
    )
    try:
        observed = semantic.validate_pinned_nli_asset(
            model_dir,
            expected_tree_sha256=(semantic.PINNED_NLI_MODEL_TREE_SHA256),
        )
    except (OSError, ValueError) as exc:
        raise NonblindSFTAuditV8Error(f"NLI model source validation failed: {exc}") from exc
    if observed != _expected_nli_identity():
        raise NonblindSFTAuditV8Error("NLI model source does not match the fixed identity")
    return observed


def _validate_manifest_contract(
    manifest: Mapping[str, Any],
    files: Mapping[str, FileSnapshot],
) -> None:
    if frozenset(manifest) != _MANIFEST_FIELDS or len(manifest) != 22:
        raise NonblindSFTAuditV8Error("manifest does not match the exact 22-field v8 contract")
    if (
        manifest.get("schema") != v8builder.NONBLIND_MANIFEST_SCHEMA
        or manifest.get("dataset_schema") != DATASET_SCHEMA
        or manifest.get("builder_version") != v8builder.NONBLIND_BUILDER_VERSION
        or manifest.get("core_builder_version") != BUILDER_VERSION
        or manifest.get("status") != "NONBLIND_V8_BUILT_NLI_UNIQUE_SUPPORT_PREBLIND_COMMITTED"
    ):
        raise NonblindSFTAuditV8Error("manifest schema/version/status mismatch")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(_SPLIT_ROLES):
        raise NonblindSFTAuditV8Error("manifest split inventory is not exact")
    for split in _SPLIT_ROLES:
        _validate_receipt(
            splits[split],
            snapshot=files[split],
            expected_path=ROLE_FILENAMES[split],
            expected_count=_EXPECTED_SPLIT_COUNTS[split],
            label=split,
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(_ARTIFACT_ROLES):
        raise NonblindSFTAuditV8Error("manifest artifact inventory is not exact")
    for role in _ARTIFACT_ROLES:
        _validate_receipt(
            artifacts[role],
            snapshot=files[role],
            expected_path=ROLE_FILENAMES[role],
            expected_count=None,
            label=role,
        )
    expected_output = _sha256_bytes(
        canonical_json(
            {
                "splits": splits,
                "artifacts": artifacts,
            }
        ).encode("utf-8")
    )
    if manifest.get("output_content_sha256") != expected_output:
        raise NonblindSFTAuditV8Error("manifest output content SHA-256 mismatch")
    if (
        manifest.get("ground_truth_policy")
        != (
            "deterministic pointer labels from licensed evidence; "
            "the fixed local NLI model audits uniqueness but never "
            "creates ground truth"
        )
        or manifest.get("selection_policy") != "researcher_explicit_domain_and_task"
        or manifest.get("source_isolation_unit") != "DOI/source_family"
        or manifest.get("counts")
        != {
            "examples": _EXPECTED_TOTAL,
            "answers": _EXPECTED_ANSWERS,
            "families": _EXPECTED_FAMILIES,
            "examples_per_family": EXAMPLES_PER_FAMILY,
            "splits": dict(_EXPECTED_SPLIT_COUNTS),
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
        or manifest.get("sealed_blind_access") != _SEALED_FALSE
        or manifest.get("claims") != _EXPECTED_CLAIMS
    ):
        raise NonblindSFTAuditV8Error("manifest fixed v8 behavioral contract mismatch")


def _snapshot_authorities(
    *,
    licensed_chunks: Path,
    rag_manifest: Path,
    semantic_inventory: Path,
) -> tuple[dict[str, FileSnapshot], dict[str, FileSnapshot]]:
    files: dict[str, FileSnapshot] = {}
    files["licensed_chunks"] = _snapshot_authority_file(
        licensed_chunks,
        expected_basename=_SOURCE_FILENAMES["licensed_chunks"],
        label="authority licensed chunks",
        maximum_bytes=_MAX_JSONL_BYTES,
    )
    files["rag_manifest"] = _snapshot_authority_file(
        rag_manifest,
        expected_basename=_SOURCE_FILENAMES["rag_manifest"],
        label="authority RAG manifest",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    files["licensed_source_catalog"] = _snapshot_authority_file(
        files["rag_manifest"].path.with_name(_SOURCE_FILENAMES["licensed_source_catalog"]),
        expected_basename=_SOURCE_FILENAMES["licensed_source_catalog"],
        label="authority licensed source catalog",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    files["semantic_inventory"] = _snapshot_authority_file(
        semantic_inventory,
        expected_basename=_SOURCE_FILENAMES["semantic_inventory"],
        label="authority semantic inventory",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    for role in (
        "semantic_records",
        "semantic_requests",
        "semantic_request_manifest",
    ):
        files[role] = _snapshot_authority_file(
            files["semantic_inventory"].path.with_name(_SOURCE_FILENAMES[role]),
            expected_basename=_SOURCE_FILENAMES[role],
            label=f"authority {role.replace('_', ' ')}",
            maximum_bytes=(
                _MAX_JSONL_BYTES if role in {"semantic_records", "semantic_requests"} else _MAX_JSON_BYTES
            ),
        )
    for role, snapshot in files.items():
        _ensure_no_hardlink(snapshot.path, label=f"authority {role}")

    code: dict[str, FileSnapshot] = {}
    for role, path in _CODE_ROLES.items():
        code[role] = _snapshot_strict_file(
            path,
            label=f"current {role} source",
            maximum_bytes=_MAX_JSON_BYTES,
        )
    return files, code


def _validate_declared_sources(
    manifest: Mapping[str, Any],
    *,
    files: Mapping[str, FileSnapshot],
    code: Mapping[str, FileSnapshot],
) -> str:
    source_inputs = manifest.get("source_inputs")
    if not isinstance(source_inputs, Mapping) or set(source_inputs) != set(_MANIFEST_SOURCE_ROLES):
        raise NonblindSFTAuditV8Error("manifest source input whitelist mismatch")
    for role in _MANIFEST_SOURCE_ROLES:
        receipt = source_inputs.get(role)
        if not isinstance(receipt, Mapping) or set(receipt) != {
            "path",
            "sha256",
        }:
            raise NonblindSFTAuditV8Error(f"manifest {role} source receipt fields mismatch")
        _validate_declared_path(
            receipt.get("path"),
            authority=files[role],
            expected_basename=_SOURCE_FILENAMES[role],
            label=f"declared {role}",
        )
        if receipt.get("sha256") != files[role].sha256:
            raise NonblindSFTAuditV8Error(f"manifest {role} source SHA-256 mismatch")
    builder = manifest.get("builder")
    if (
        not isinstance(builder, Mapping)
        or set(builder)
        != {
            "code",
            "split_algorithm_version",
            "repair_policy_version",
            "seed",
        }
        or builder.get("split_algorithm_version") != v8builder.SPLIT_ALGORITHM_VERSION
        or builder.get("repair_policy_version") != v8builder.NLI_REPAIR_POLICY_VERSION
        or not isinstance(builder.get("seed"), str)
        or not builder.get("seed")
    ):
        raise NonblindSFTAuditV8Error("manifest builder contract mismatch")
    code_receipts = builder.get("code")
    if not isinstance(code_receipts, Mapping) or set(code_receipts) != set(_CODE_ROLES):
        raise NonblindSFTAuditV8Error("manifest builder code whitelist mismatch")
    for role in _CODE_ROLES:
        receipt = code_receipts.get(role)
        if not isinstance(receipt, Mapping) or set(receipt) != {
            "path",
            "sha256",
        }:
            raise NonblindSFTAuditV8Error(f"manifest {role} code receipt fields mismatch")
        _validate_declared_path(
            receipt.get("path"),
            authority=code[role],
            expected_basename=code[role].path.name,
            label=f"declared {role}",
        )
        if receipt.get("sha256") != code[role].sha256:
            raise NonblindSFTAuditV8Error(f"manifest {role} code SHA-256 mismatch")
    expected_input = _sha256_bytes(
        canonical_json(
            {
                "files": {role: files[role].sha256 for role in _MANIFEST_SOURCE_ROLES},
                "nli_model_tree_sha256": (semantic.PINNED_NLI_MODEL_TREE_SHA256),
                "seed_sha256": _sha256_bytes(str(builder["seed"]).encode("utf-8")),
            }
        ).encode("utf-8")
    )
    if manifest.get("input_commitment_sha256") != expected_input:
        raise NonblindSFTAuditV8Error("manifest input commitment SHA-256 mismatch")
    return str(builder["seed"])


def _jsonl_objects(
    snapshot: FileSnapshot,
    *,
    label: str,
) -> list[dict[str, Any]]:
    if not snapshot.payload.endswith(b"\n"):
        raise NonblindSFTAuditV8Error(f"{label} must end with exactly one record newline")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(snapshot.payload.splitlines(), 1):
        if not line:
            raise NonblindSFTAuditV8Error(f"{label} contains a blank row")
        value = _strict_json(line, label=f"{label} row {index}")
        if not isinstance(value, dict):
            raise NonblindSFTAuditV8Error(f"{label} row {index} must be an object")
        rows.append(value)
    return rows


def _validate_source_catalog(
    snapshot: FileSnapshot,
    *,
    request_manifest: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    catalog = _strict_json_object(
        snapshot.payload,
        label="licensed source catalog",
    )
    if (
        set(catalog)
        != {
            "schema",
            "status",
            "created_at",
            "source_count",
            "chunk_count",
            "namespace_counts",
            "license_policy",
            "evidence_boundary",
            "records",
        }
        or catalog.get("schema") != "icmat.rag.licensed_source_catalog.v2"
        or catalog.get("status") != "LICENSED_FULLTEXT_CANDIDATE_OFFLINE"
        or catalog.get("source_count") != 14
        or catalog.get("chunk_count") != 519
        or not isinstance(catalog.get("records"), list)
        or len(catalog["records"]) != 14
    ):
        raise NonblindSFTAuditV8Error("licensed source catalog contract mismatch")
    inputs = request_manifest.get("input_artifacts")
    if (
        not isinstance(inputs, Mapping)
        or inputs.get("source_manifest_sha256") != snapshot.sha256
        or inputs.get("source_manifest_schema") != catalog["schema"]
        or inputs.get("source_manifest_authority") != "rag_v2_licensed_source_catalog"
        or inputs.get("source_count") != 14
        or inputs.get("source_asset_count") != 14
        or inputs.get("authorized_source_count") != 14
        or inputs.get("chunk_count") != 519
        or inputs.get("declared_licensed_chunk_count") != 519
        or inputs.get("formal_source_authority") is not True
        or inputs.get("ignored_unauthorized_source_asset_count") != 0
    ):
        raise NonblindSFTAuditV8Error("semantic request source-catalog binding mismatch")
    records: dict[str, Mapping[str, Any]] = {}
    for row in catalog["records"]:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("source_id"), str)
            or row["source_id"] in records
        ):
            raise NonblindSFTAuditV8Error("licensed source catalog records are invalid")
        records[str(row["source_id"])] = row
    return records


def _validate_requests(
    *,
    files: Mapping[str, FileSnapshot],
    families: Sequence[Any],
) -> tuple[
    dict[str, Any],
    dict[str, Mapping[str, Any]],
    Mapping[str, Mapping[str, Any]],
]:
    request_manifest = _strict_json_object(
        files["semantic_request_manifest"].payload,
        label="semantic request manifest",
    )
    if (
        set(request_manifest)
        != {
            "schema",
            "generator_version",
            "input_artifacts",
            "request_count",
            "request_ids",
            "request_file_sha256",
            "sealed_blind_access",
            "manifest_sha256",
        }
        or request_manifest.get("schema") != semantic.REQUEST_MANIFEST_SCHEMA
        or request_manifest.get("generator_version") != semantic.GENERATOR_VERSION
        or request_manifest.get("sealed_blind_access") != _SEALED_FALSE
    ):
        raise NonblindSFTAuditV8Error("semantic request manifest contract mismatch")
    core = {key: value for key, value in request_manifest.items() if key != "manifest_sha256"}
    if request_manifest.get("manifest_sha256") != _sha256_bytes(canonical_json(core).encode("utf-8")):
        raise NonblindSFTAuditV8Error("semantic request manifest self-hash mismatch")
    requests = _jsonl_objects(
        files["semantic_requests"],
        label="semantic requests",
    )
    if (
        request_manifest.get("request_file_sha256") != files["semantic_requests"].sha256
        or request_manifest.get("request_count") != len(requests)
        or request_manifest.get("request_count") != 2057
    ):
        raise NonblindSFTAuditV8Error("semantic request file/count binding mismatch")
    inputs = request_manifest.get("input_artifacts")
    if (
        not isinstance(inputs, Mapping)
        or set(inputs)
        != {
            "licensed_chunks_sha256",
            "source_manifest_sha256",
            "source_count",
            "chunk_count",
            "usable_sentence_count",
            "source_asset_count",
            "authorized_source_count",
            "declared_licensed_chunk_count",
            "ignored_unauthorized_source_asset_count",
            "formal_source_authority",
            "source_manifest_schema",
            "source_manifest_authority",
        }
        or inputs.get("licensed_chunks_sha256") != files["licensed_chunks"].sha256
        or inputs.get("usable_sentence_count") != len(requests)
    ):
        raise NonblindSFTAuditV8Error("semantic request input artifact binding mismatch")
    catalog = _validate_source_catalog(
        files["licensed_source_catalog"],
        request_manifest=request_manifest,
    )
    families_by_id = {family.source_id: family for family in families}
    request_map: dict[str, Mapping[str, Any]] = {}
    request_ids: list[str] = []
    for row in requests:
        if frozenset(row) != _REQUEST_FIELDS:
            raise NonblindSFTAuditV8Error("semantic request row fields mismatch")
        request_sha = row.get("request_sha256")
        request_core = dict(row)
        request_core.pop("request_sha256", None)
        if request_sha != _sha256_bytes(canonical_json(request_core).encode("utf-8")):
            raise NonblindSFTAuditV8Error("semantic request row self-hash mismatch")
        request_id = row.get("request_id")
        identity_core = dict(request_core)
        identity_core.pop("request_id", None)
        expected_id = "icmsq7:" + _sha256_bytes(canonical_json(identity_core).encode("utf-8"))
        if (
            request_id != expected_id
            or request_id in request_map
            or row.get("schema") != semantic.REQUEST_SCHEMA
            or row.get("constraints") != _REQUEST_CONSTRAINTS
        ):
            raise NonblindSFTAuditV8Error("semantic request identity/constraint mismatch")
        source_id = row.get("source_id")
        family = families_by_id.get(source_id)
        source = catalog.get(str(source_id))
        chunk_ids = row.get("chunk_ids")
        original = row.get("original_sentence")
        if (
            family is None
            or source is None
            or not isinstance(chunk_ids, list)
            or not chunk_ids
            or len(chunk_ids) != len(set(chunk_ids))
            or not isinstance(original, str)
            or _sha256_bytes(original.encode("utf-8")) != row.get("original_sha256")
            or row.get("namespace") != family.namespace
            or row.get("source_title") != family.source_title
            or row.get("source_uri") != family.source_uri
            or row.get("license_id") != family.license_id
            or row.get("source_manifest_authority") != "rag_v2_licensed_source_catalog"
            or row.get("source_asset_sha256") != source.get("xml_sha256")
            or row.get("source_asset_uri") != source.get("xml_source_url")
        ):
            raise NonblindSFTAuditV8Error("semantic request source binding mismatch")
        chunks = {str(chunk.get("chunk_id")): chunk for chunk in family.chunks}
        if not set(chunk_ids) <= set(chunks):
            raise NonblindSFTAuditV8Error("semantic request chunk binding mismatch")
        if not any(
            original
            in semantic._protected_sentence_split(  # noqa: SLF001
                str(chunks[chunk_id].get("text", ""))
            )
            for chunk_id in chunk_ids
        ):
            raise NonblindSFTAuditV8Error("semantic request original is absent from bound chunks")
        request_ids.append(str(request_id))
        request_map[str(request_id)] = row
    if request_manifest.get("request_ids") != request_ids:
        raise NonblindSFTAuditV8Error("semantic request ID sequence mismatch")
    return request_manifest, request_map, catalog


def _validate_semantic_request_record_links(
    *,
    files: Mapping[str, FileSnapshot],
    request_map: Mapping[str, Mapping[str, Any]],
) -> None:
    inventory = _strict_json_object(
        files["semantic_inventory"].payload,
        label="semantic inventory",
    )
    if (
        inventory.get("request_manifest_sha256")
        != _strict_json_object(
            files["semantic_request_manifest"].payload,
            label="semantic request manifest",
        ).get("manifest_sha256")
        or inventory.get("sealed_blind_access") != _SEALED_FALSE
    ):
        raise NonblindSFTAuditV8Error("semantic inventory/request manifest binding mismatch")
    records = _jsonl_objects(
        files["semantic_records"],
        label="semantic records",
    )
    accepted_ids = {
        str(row.get("record_id")) for row in inventory.get("accepted_records", []) if isinstance(row, Mapping)
    }
    accepted_seen = 0
    for record in records:
        if str(record.get("record_id")) not in accepted_ids:
            continue
        request = request_map.get(str(record.get("request_id")))
        if (
            request is None
            or record.get("request_sha256") != request.get("request_sha256")
            or record.get("source_id") != request.get("source_id")
            or record.get("source_record_sha256") != request.get("source_record_sha256")
            or record.get("original_sha256") != request.get("original_sha256")
            or record.get("original_sentence") != request.get("original_sentence")
            or record.get("chunk_ids") != request.get("chunk_ids")
        ):
            raise NonblindSFTAuditV8Error("accepted semantic record/request binding mismatch")
        accepted_seen += 1
    if accepted_seen != inventory.get("accepted_count"):
        raise NonblindSFTAuditV8Error("accepted semantic request coverage mismatch")


def _request_binding_payload(
    *,
    files: Mapping[str, FileSnapshot],
    request_manifest: Mapping[str, Any],
    request_count: int,
) -> dict[str, Any]:
    inventory = _strict_json_object(
        files["semantic_inventory"].payload,
        label="semantic inventory",
    )
    nli = inventory.get("nli_provenance")
    if (
        not isinstance(nli, Mapping)
        or nli.get("backend") != "local_transformers_nli"
        or nli.get("local_files_only") is not True
        or nli.get("model_tree_sha256") != semantic.PINNED_NLI_MODEL_TREE_SHA256
    ):
        raise NonblindSFTAuditV8Error("semantic inventory NLI binding mismatch")
    payload = {
        "schema": v8builder.SEMANTIC_BINDING_SCHEMA,
        "status": "PASS_FROZEN_SEMANTIC_R7_INVENTORY_REQUESTS_BOUND",
        "findings": [],
        "semantic_inventory": {
            "file_sha256": files["semantic_inventory"].sha256,
            "producer_inventory_sha256": inventory.get("inventory_sha256"),
            "accepted_count": inventory.get("accepted_count"),
        },
        "semantic_records": {
            "file_sha256": files["semantic_records"].sha256,
        },
        "semantic_requests": {
            "file_sha256": files["semantic_requests"].sha256,
            "request_count": request_count,
        },
        "semantic_request_manifest": {
            "file_sha256": files["semantic_request_manifest"].sha256,
            "producer_manifest_sha256": request_manifest.get("manifest_sha256"),
        },
        "nli_model_tree_sha256": nli.get("model_tree_sha256"),
        "sealed_blind_access": dict(_SEALED_FALSE),
    }
    return {
        **payload,
        "binding_sha256": _sha256_bytes(canonical_json(payload).encode("utf-8")),
    }


def _validate_nli_provenance(
    value: Any,
    *,
    model: Mapping[str, Any],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NonblindSFTAuditV8Error(f"{label} NLI provenance missing")
    expected = {
        "backend": "local_transformers_nli",
        **dict(model),
        "quality_claim_allowed": True,
    }
    if (
        set(value) != {*expected, "device"}
        or any(value.get(key) != expected_value for key, expected_value in expected.items())
        or not isinstance(value.get("device"), str)
        or not value.get("device")
    ):
        raise NonblindSFTAuditV8Error(f"{label} NLI model binding mismatch")
    return value


def _probabilities(value: Any, *, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != {
        "entailment",
        "contradiction",
        "neutral",
    }:
        raise NonblindSFTAuditV8Error(f"{label} probability fields mismatch")
    output: dict[str, float] = {}
    for key in ("entailment", "contradiction", "neutral"):
        number = value.get(key)
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
            or not 0.0 <= float(number) <= 1.0
        ):
            raise NonblindSFTAuditV8Error(f"{label} contains an invalid probability")
        output[key] = float(number)
    if abs(sum(output.values()) - 1.0) > 1e-4:
        raise NonblindSFTAuditV8Error(f"{label} probabilities do not sum to one")
    return output


def _span_rows(
    row: Mapping[str, Any],
) -> tuple[
    list[tuple[str, str]],
    list[tuple[str, str]],
    int,
    int,
]:
    blocks = row.get("compiler_evidence")
    target_span_id = row.get("target_span_id")
    if not isinstance(blocks, list) or len(blocks) != 2 or not isinstance(target_span_id, str):
        raise NonblindSFTAuditV8Error("ANSWER evidence/target contract mismatch")
    target_evidence_id = target_span_id.split(".", 1)[0]
    target_indices = [
        index
        for index, block in enumerate(blocks)
        if isinstance(block, Mapping) and block.get("evidence_id") == target_evidence_id
    ]
    if len(target_indices) != 1:
        raise NonblindSFTAuditV8Error("ANSWER target evidence is not unique")
    target_index = target_indices[0]
    distractor_index = 1 - target_index

    def decode(block: Mapping[str, Any]) -> list[tuple[str, str]]:
        sentences = block.get("sentences")
        if not isinstance(sentences, list) or not sentences:
            raise NonblindSFTAuditV8Error("ANSWER evidence passage is empty")
        output: list[tuple[str, str]] = []
        for sentence in sentences:
            if (
                not isinstance(sentence, Mapping)
                or set(sentence) != COMPILER_SENTENCE_FIELDS
                or not isinstance(sentence.get("span_id"), str)
                or not isinstance(sentence.get("text"), str)
            ):
                raise NonblindSFTAuditV8Error("ANSWER evidence sentence contract mismatch")
            output.append((str(sentence["span_id"]), str(sentence["text"])))
        return output

    return (
        decode(blocks[target_index]),
        decode(blocks[distractor_index]),
        target_index,
        distractor_index,
    )


def _validate_integrated_nli(
    payload: Mapping[str, Any],
    *,
    rows: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    expected_fields = {
        "schema",
        "status",
        "policy_version",
        "score_orientation",
        "non_target_scope",
        "target_passage_neighbor_policy",
        "thresholds",
        "nli_provenance",
        "answer_count",
        "repair_count",
        "split_answer_counts",
        "split_repair_counts",
        "minimum_target_entailment",
        "maximum_target_passage_non_target_entailment",
        "maximum_distractor_entailment",
        "maximum_non_target_entailment",
        "score_cache_pair_count",
        "entries",
        "audit_sha256",
    }
    if set(payload) != expected_fields:
        raise NonblindSFTAuditV8Error("integrated NLI audit fields mismatch")
    core = {key: value for key, value in payload.items() if key != "audit_sha256"}
    if payload.get("audit_sha256") != _sha256_bytes(canonical_json(core).encode("utf-8")):
        raise NonblindSFTAuditV8Error("integrated NLI audit self-hash mismatch")
    if (
        payload.get("schema") != v8builder.NLI_AUDIT_SCHEMA
        or payload.get("status") != "PASS_ALL_ANSWER_EXAMPLES_HAVE_UNIQUE_NLI_SUPPORT"
        or payload.get("policy_version") != v8builder.NLI_REPAIR_POLICY_VERSION
        or payload.get("score_orientation")
        != {
            "premise": "evidence_sentence",
            "hypothesis": "requested_claim",
        }
        or payload.get("non_target_scope")
        != ("every span other than target_span_id across both evidence passages")
        or payload.get("target_passage_neighbor_policy")
        != ("fail_closed_without_rewriting_or_shortening_target_passage")
        or payload.get("thresholds")
        != {
            "target_entailment_min": _TARGET_ENTAILMENT_MIN,
            "distractor_entailment_max": _NON_TARGET_ENTAILMENT_MAX,
        }
    ):
        raise NonblindSFTAuditV8Error("integrated NLI policy/threshold contract mismatch")
    _validate_nli_provenance(
        payload.get("nli_provenance"),
        model=model,
        label="integrated audit",
    )
    answer_rows = {str(row["example_id"]): row for row in rows if row.get("decision") == "ANSWER"}
    entries = payload.get("entries")
    if (
        len(answer_rows) != _EXPECTED_ANSWERS
        or not isinstance(entries, list)
        or len(entries) != _EXPECTED_ANSWERS
    ):
        raise NonblindSFTAuditV8Error("integrated NLI ANSWER coverage mismatch")
    seen: set[str] = set()
    target_values: list[float] = []
    neighbor_values: list[float] = []
    distractor_values: list[float] = []
    repair_counts: Counter[str] = Counter()
    unique_pairs: set[tuple[str, str]] = set()
    entry_by_final: dict[str, Mapping[str, Any]] = {}
    expected_entry_fields = {
        "original_example_id",
        "final_example_id",
        "split",
        "source_id",
        "target_span_id",
        "target_span_sha256",
        "target_probabilities",
        "target_passage_non_target_span_scores",
        "distractor_evidence_id",
        "distractor_chunk_id",
        "distractor_passage_sha256",
        "distractor_span_scores",
        "max_distractor_entailment",
        "repair_applied",
    }
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != expected_entry_fields:
            raise NonblindSFTAuditV8Error("integrated NLI entry fields mismatch")
        final_id = entry.get("final_example_id")
        row = answer_rows.get(str(final_id))
        if (
            row is None
            or final_id in seen
            or entry.get("split") != row.get("split")
            or entry.get("source_id") != row.get("source_id")
            or entry.get("target_span_id") != row.get("target_span_id")
            or not isinstance(entry.get("repair_applied"), bool)
        ):
            raise NonblindSFTAuditV8Error("integrated NLI entry/example binding mismatch")
        seen.add(str(final_id))
        entry_by_final[str(final_id)] = entry
        target_rows, distractor_rows, _, distractor_index = _span_rows(row)
        target_span_id = str(row["target_span_id"])
        target_matches = [text for span_id, text in target_rows if span_id == target_span_id]
        if len(target_matches) != 1:
            raise NonblindSFTAuditV8Error("integrated NLI target span is not unique")
        target_text = target_matches[0]
        if entry.get("target_span_sha256") != _sha256_bytes(target_text.encode("utf-8")):
            raise NonblindSFTAuditV8Error("integrated NLI target span hash mismatch")
        target_probs = _probabilities(
            entry.get("target_probabilities"),
            label="target",
        )
        if target_probs["entailment"] < _TARGET_ENTAILMENT_MIN:
            raise NonblindSFTAuditV8Error("integrated NLI target entailment is below 0.90")
        target_values.append(target_probs["entailment"])
        unique_pairs.add((target_text, str(row["requested_claim"])))

        neighbors = entry.get("target_passage_non_target_span_scores")
        expected_neighbors = [(span_id, text) for span_id, text in target_rows if span_id != target_span_id]
        if not isinstance(neighbors, list) or len(neighbors) != len(expected_neighbors):
            raise NonblindSFTAuditV8Error("integrated NLI target-neighbor coverage mismatch")
        for observed, (span_id, text) in zip(
            neighbors,
            expected_neighbors,
            strict=True,
        ):
            if (
                not isinstance(observed, Mapping)
                or set(observed) != {"span_id", "sentence_sha256", "probabilities"}
                or observed.get("span_id") != span_id
                or observed.get("sentence_sha256") != _sha256_bytes(text.encode("utf-8"))
            ):
                raise NonblindSFTAuditV8Error("integrated NLI target-neighbor span binding mismatch")
            probabilities = _probabilities(
                observed.get("probabilities"),
                label="target-neighbor",
            )
            if probabilities["entailment"] > _NON_TARGET_ENTAILMENT_MAX:
                raise NonblindSFTAuditV8Error("integrated NLI non-target entailment exceeds 0.10")
            neighbor_values.append(probabilities["entailment"])
            unique_pairs.add((text, str(row["requested_claim"])))

        blocks = row["compiler_evidence"]
        distractor_block = blocks[distractor_index]
        distractor_evidence_id = str(distractor_block["evidence_id"])
        metadata = row["metadata"]
        chunk_id = metadata["evidence_chunk_ids"][distractor_index]
        distractor_texts = [text for _, text in distractor_rows]
        if (
            entry.get("distractor_evidence_id") != distractor_evidence_id
            or entry.get("distractor_chunk_id") != chunk_id
            or entry.get("distractor_passage_sha256")
            != _sha256_bytes(canonical_json(distractor_texts).encode("utf-8"))
        ):
            raise NonblindSFTAuditV8Error("integrated NLI distractor passage binding mismatch")
        scores = entry.get("distractor_span_scores")
        if not isinstance(scores, list) or len(scores) != len(distractor_rows):
            raise NonblindSFTAuditV8Error("integrated NLI distractor span coverage mismatch")
        local_entailments: list[float] = []
        for index, (score, (span_id, text)) in enumerate(
            zip(scores, distractor_rows, strict=True),
            1,
        ):
            if (
                not isinstance(score, Mapping)
                or set(score)
                != {
                    "span_id",
                    "sentence_index",
                    "sentence_sha256",
                    "probabilities",
                }
                or score.get("span_id") != span_id
                or score.get("sentence_index") != index
                or score.get("sentence_sha256") != _sha256_bytes(text.encode("utf-8"))
            ):
                raise NonblindSFTAuditV8Error("integrated NLI distractor span binding mismatch")
            probabilities = _probabilities(
                score.get("probabilities"),
                label="distractor",
            )
            if probabilities["entailment"] > _NON_TARGET_ENTAILMENT_MAX:
                raise NonblindSFTAuditV8Error("integrated NLI distractor entailment exceeds 0.10")
            local_entailments.append(probabilities["entailment"])
            unique_pairs.add((text, str(row["requested_claim"])))
        observed_max = max(local_entailments, default=0.0)
        if entry.get("max_distractor_entailment") != round(
            observed_max,
            8,
        ):
            raise NonblindSFTAuditV8Error("integrated NLI distractor maximum mismatch")
        distractor_values.append(observed_max)
        if entry["repair_applied"]:
            repair_counts[str(row["split"])] += 1
        elif entry.get("original_example_id") != final_id:
            raise NonblindSFTAuditV8Error("unrepaired NLI entry changed example identity")
    if seen != set(answer_rows):
        raise NonblindSFTAuditV8Error("integrated NLI entries omit final ANSWER examples")
    split_answer_counts = Counter(str(row["split"]) for row in answer_rows.values())
    expected_repair_counts = {split: repair_counts[split] for split in NONBLIND_SPLITS}
    if (
        payload.get("answer_count") != _EXPECTED_ANSWERS
        or payload.get("split_answer_counts") != dict(sorted(split_answer_counts.items()))
        or payload.get("repair_count") != sum(repair_counts.values())
        or payload.get("split_repair_counts") != expected_repair_counts
        or payload.get("minimum_target_entailment") != round(min(target_values), 8)
        or payload.get("maximum_target_passage_non_target_entailment")
        != round(max(neighbor_values, default=0.0), 8)
        or payload.get("maximum_distractor_entailment") != round(max(distractor_values), 8)
        or payload.get("maximum_non_target_entailment")
        != round(
            max([*neighbor_values, *distractor_values], default=0.0),
            8,
        )
        or not _is_integer(payload.get("score_cache_pair_count"))
        or payload.get("score_cache_pair_count") < len(unique_pairs)
    ):
        raise NonblindSFTAuditV8Error("integrated NLI threshold/count summary mismatch")
    return {
        "entries": entry_by_final,
        "repair_count": sum(repair_counts.values()),
        "split_repair_counts": expected_repair_counts,
        "provenance": dict(payload["nli_provenance"]),
    }


def _validate_repair_manifest(
    payload: Mapping[str, Any],
    *,
    nli_summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    expected_fields = {
        "schema",
        "status",
        "policy_version",
        "target_passage_modified",
        "manual_jsonl_editing",
        "candidate_scope",
        "stable_selection",
        "thresholds",
        "repair_count",
        "split_repair_counts",
        "repairs",
        "repair_manifest_sha256",
    }
    if set(payload) != expected_fields:
        raise NonblindSFTAuditV8Error("repair manifest fields mismatch")
    core = {key: value for key, value in payload.items() if key != "repair_manifest_sha256"}
    if payload.get("repair_manifest_sha256") != _sha256_bytes(canonical_json(core).encode("utf-8")):
        raise NonblindSFTAuditV8Error("repair manifest self-hash mismatch")
    if (
        payload.get("schema") != v8builder.REPAIR_MANIFEST_SCHEMA
        or payload.get("status") != "PASS_DETERMINISTIC_DISTRACTOR_REBUILD_COMPLETE"
        or payload.get("policy_version") != v8builder.NLI_REPAIR_POLICY_VERSION
        or payload.get("target_passage_modified") is not False
        or payload.get("manual_jsonl_editing") is not False
        or payload.get("candidate_scope") != "same_source_family_existing_allowed_candidate_pool"
        or payload.get("stable_selection")
        != ("highest_token_overlap_then_stable_sha_order_among_nli_qualified_nonoverlapping_passages")
        or payload.get("thresholds")
        != {
            "target_entailment_min": _TARGET_ENTAILMENT_MIN,
            "distractor_entailment_max": _NON_TARGET_ENTAILMENT_MAX,
        }
        or payload.get("repair_count") != nli_summary["repair_count"]
        or payload.get("split_repair_counts") != nli_summary["split_repair_counts"]
    ):
        raise NonblindSFTAuditV8Error("repair manifest policy/count mismatch")
    repairs = payload.get("repairs")
    if not isinstance(repairs, list) or len(repairs) != payload["repair_count"]:
        raise NonblindSFTAuditV8Error("repair manifest entry count mismatch")
    rows_by_id = {str(row["example_id"]): row for row in rows}
    observed_final: set[str] = set()
    expected_fields = {
        "original_example_id",
        "rebuilt_example_id",
        "split",
        "source_id",
        "target_span_id",
        "distractor_evidence_id",
        "original_distractor",
        "replacement_distractor",
        "rejected_candidate_count",
        "rejected_candidates",
    }
    for repair in repairs:
        if not isinstance(repair, Mapping) or set(repair) != expected_fields:
            raise NonblindSFTAuditV8Error("repair entry fields mismatch")
        final_id = str(repair.get("rebuilt_example_id"))
        row = rows_by_id.get(final_id)
        nli_entry = nli_summary["entries"].get(final_id)
        original = repair.get("original_distractor")
        replacement = repair.get("replacement_distractor")
        rejected = repair.get("rejected_candidates")
        if (
            row is None
            or nli_entry is None
            or final_id in observed_final
            or nli_entry.get("repair_applied") is not True
            or repair.get("original_example_id") != nli_entry.get("original_example_id")
            or repair.get("split") != row.get("split")
            or repair.get("source_id") != row.get("source_id")
            or repair.get("target_span_id") != row.get("target_span_id")
            or repair.get("distractor_evidence_id") != nli_entry.get("distractor_evidence_id")
            or not isinstance(original, Mapping)
            or set(original) != {"chunk_id", "passage_sha256", "max_entailment"}
            or not isinstance(replacement, Mapping)
            or set(replacement) != {"chunk_id", "passage_sha256", "max_entailment"}
            or replacement.get("chunk_id") != nli_entry.get("distractor_chunk_id")
            or replacement.get("passage_sha256") != nli_entry.get("distractor_passage_sha256")
            or replacement.get("max_entailment") != nli_entry.get("max_distractor_entailment")
            or not isinstance(rejected, list)
            or repair.get("rejected_candidate_count") != len(rejected)
        ):
            raise NonblindSFTAuditV8Error("repair entry/final example binding mismatch")
        if (
            not isinstance(original.get("max_entailment"), (int, float))
            or float(original["max_entailment"]) <= _NON_TARGET_ENTAILMENT_MAX
            or float(replacement["max_entailment"]) > _NON_TARGET_ENTAILMENT_MAX
        ):
            raise NonblindSFTAuditV8Error("repair entailment threshold history mismatch")
        for rejected_row in rejected:
            if (
                not isinstance(rejected_row, Mapping)
                or set(rejected_row) != {"chunk_id", "passage_sha256", "max_entailment"}
                or not _HEX_SHA256.fullmatch(str(rejected_row.get("passage_sha256", "")))
                or not isinstance(
                    rejected_row.get("max_entailment"),
                    (int, float),
                )
                or float(rejected_row["max_entailment"]) <= _NON_TARGET_ENTAILMENT_MAX
            ):
                raise NonblindSFTAuditV8Error("repair rejected-candidate contract mismatch")
        observed_final.add(final_id)
    expected_final = {
        final_id for final_id, entry in nli_summary["entries"].items() if entry.get("repair_applied") is True
    }
    if observed_final != expected_final:
        raise NonblindSFTAuditV8Error("repair manifest does not cover all repaired examples")


def _validate_preblind_commitment(
    payload: Mapping[str, Any],
    *,
    authorities: AuthorityState,
) -> None:
    expected_fields = {
        "schema",
        "status",
        "builder_version",
        "core_builder_version",
        "split_algorithm_version",
        "repair_policy_version",
        "seed",
        "seed_sha256",
        "expected_blind_count",
        "thresholds",
        "nli_model",
        "builder_code",
        "source_inputs",
        "rag_manifest_id",
        "sealed_blind_access",
        "commitment_sha256",
    }
    if set(payload) != expected_fields:
        raise NonblindSFTAuditV8Error("preblind commitment fields mismatch")
    core = {key: value for key, value in payload.items() if key != "commitment_sha256"}
    if payload.get("commitment_sha256") != _sha256_bytes(canonical_json(core).encode("utf-8")):
        raise NonblindSFTAuditV8Error("preblind commitment self-hash mismatch")
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
        raise NonblindSFTAuditV8Error("preblind commitment contains a protected artifact detail")
    if (
        payload.get("schema") != v8builder.PREBLIND_COMMITMENT_SCHEMA
        or payload.get("status") != "PREBLIND_COMMITTED_NONBLIND_ONLY"
        or payload.get("builder_version") != v8builder.NONBLIND_BUILDER_VERSION
        or payload.get("core_builder_version") != BUILDER_VERSION
        or payload.get("split_algorithm_version") != v8builder.SPLIT_ALGORITHM_VERSION
        or payload.get("repair_policy_version") != v8builder.NLI_REPAIR_POLICY_VERSION
        or payload.get("seed") != authorities.seed
        or payload.get("seed_sha256") != _sha256_bytes(authorities.seed.encode("utf-8"))
        or payload.get("expected_blind_count") != v8builder.EXPECTED_BLIND_COUNT
        or payload.get("thresholds")
        != {
            "target_entailment_min": _TARGET_ENTAILMENT_MIN,
            "distractor_entailment_max": _NON_TARGET_ENTAILMENT_MAX,
        }
        or payload.get("builder_code") != {role: authorities.code[role].sha256 for role in _CODE_ROLES}
        or payload.get("source_inputs")
        != {role: authorities.files[role].sha256 for role in _MANIFEST_SOURCE_ROLES}
        or payload.get("rag_manifest_id") != authorities.rag_manifest_id
        or payload.get("sealed_blind_access") != _SEALED_FALSE
    ):
        if payload.get("source_inputs") != {
            role: authorities.files[role].sha256 for role in _MANIFEST_SOURCE_ROLES
        }:
            raise NonblindSFTAuditV8Error("preblind source binding mismatch")
        raise NonblindSFTAuditV8Error("preblind commitment contract mismatch")
    _validate_nli_provenance(
        payload.get("nli_model"),
        model=authorities.nli,
        label="preblind commitment",
    )


def _validate_stored_audits(
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    rows: Sequence[Mapping[str, Any]],
    authorities: AuthorityState,
) -> None:
    assignments = evidence.assign_family_splits(
        authorities.families,
        seed=authorities.seed,
    )
    balance = _balance_report_v7(rows, assignments)
    balance["schema"] = v8builder.NONBLIND_BALANCE_SCHEMA
    group = _group_report_v7(authorities.families, assignments)
    group["schema"] = v8builder.NONBLIND_GROUP_SCHEMA
    leakage = _leakage_report_v7(rows)
    leakage["schema"] = v8builder.NONBLIND_LEAKAGE_SCHEMA
    semantic_binding = {
        "schema": v8builder.SEMANTIC_BINDING_SCHEMA,
        "status": "PASS_SEMANTIC_R7_AND_REQUESTS_BOUND",
        "upstream_semantic_inventory_audit": dict(authorities.semantic_audit),
        "request_binding": dict(authorities.request_binding),
    }
    expected = {
        "balance_audit": balance,
        "group_isolation_audit": group,
        "content_leakage_audit": leakage,
        "semantic_binding_audit": semantic_binding,
    }
    for role, value in expected.items():
        if artifacts[role] != value:
            raise NonblindSFTAuditV8Error(f"stored {role} differs from independent recomputation")
        if not str(value.get("status", "")).startswith("PASS"):
            raise NonblindSFTAuditV8Error(f"independently recomputed {role} did not pass")
    nli_summary = _validate_integrated_nli(
        artifacts["nli_unique_support_audit"],
        rows=rows,
        model=authorities.nli,
    )
    _validate_repair_manifest(
        artifacts["repair_manifest"],
        nli_summary=nli_summary,
        rows=rows,
    )
    _validate_preblind_commitment(
        artifacts["preblind_commitment"],
        authorities=authorities,
    )
    report = artifacts["build_report"]
    expected_report = {
        "schema": v8builder.NONBLIND_REPORT_SCHEMA,
        "status": ("PASS_NONBLIND_V8_NLI_UNIQUE_SUPPORT_PREBLIND_COMMITTED"),
        "builder_version": v8builder.NONBLIND_BUILDER_VERSION,
        "counts": {
            "examples": _EXPECTED_TOTAL,
            "answers": _EXPECTED_ANSWERS,
            "families": _EXPECTED_FAMILIES,
            "examples_per_family": EXAMPLES_PER_FAMILY,
            "splits": dict(_EXPECTED_SPLIT_COUNTS),
            "repairs": nli_summary["repair_count"],
        },
        "audits": {
            "balance": "PASS",
            "group_isolation": "PASS",
            "content_leakage": "PASS",
            "semantic_inventory_requests": ("PASS_SEMANTIC_R7_AND_REQUESTS_BOUND"),
            "rag_authority_binding": ("PASS_RAG_MANIFEST_LICENSED_CHUNKS_BOUND"),
            "nli_unique_support": ("PASS_ALL_ANSWER_EXAMPLES_HAVE_UNIQUE_NLI_SUPPORT"),
            "distractor_repairs": ("PASS_DETERMINISTIC_DISTRACTOR_REBUILD_COMPLETE"),
        },
        "nli_thresholds": {
            "target_entailment_min": _TARGET_ENTAILMENT_MIN,
            "distractor_entailment_max": _NON_TARGET_ENTAILMENT_MAX,
        },
        "claims": dict(_EXPECTED_CLAIMS),
    }
    if report != expected_report:
        raise NonblindSFTAuditV8Error("stored build report is not the exact passing v8 report")


def _validate_manifest_nli_summary(
    manifest: Mapping[str, Any],
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    authorities: AuthorityState,
) -> None:
    nli = artifacts["nli_unique_support_audit"]
    repair = artifacts["repair_manifest"]
    expected = {
        "provenance": dict(nli["nli_provenance"]),
        "score_orientation": dict(nli["score_orientation"]),
        "non_target_scope": nli["non_target_scope"],
        "target_passage_neighbor_policy": nli["target_passage_neighbor_policy"],
        "thresholds": dict(nli["thresholds"]),
        "answer_count": _EXPECTED_ANSWERS,
        "repair_count": repair["repair_count"],
        "target_passage_modified": False,
    }
    if manifest.get("nli_unique_support") != expected:
        raise NonblindSFTAuditV8Error("manifest integrated NLI summary binding mismatch")
    _validate_nli_provenance(
        expected["provenance"],
        model=authorities.nli,
        label="manifest",
    )


def _load_and_validate_dataset(
    primary: Path,
    *,
    label: str,
    licensed_chunks: Path,
    rag_manifest: Path,
    semantic_inventory: Path,
    nli_model_dir: Path,
) -> DatasetState:
    root, identity, dataset_files, manifest = _snapshot_dataset_files(
        primary,
        label=label,
    )
    _validate_manifest_contract(manifest, dataset_files)
    source_files, code = _snapshot_authorities(
        licensed_chunks=licensed_chunks,
        rag_manifest=rag_manifest,
        semantic_inventory=semantic_inventory,
    )
    nli = _validate_nli_asset(nli_model_dir)
    seed = _validate_declared_sources(
        manifest,
        files=source_files,
        code=code,
    )
    rag_payload, rag_binding = _independent_rag_binding(
        manifest_snapshot=source_files["rag_manifest"],
        chunks_snapshot=source_files["licensed_chunks"],
    )
    if rag_binding.get("status") != "PASS_RAG_MANIFEST_LICENSED_CHUNKS_BOUND" or not isinstance(
        rag_payload.get("manifest_id"), str
    ):
        raise NonblindSFTAuditV8Error("RAG authority binding did not pass")
    families = _load_families(source_files["licensed_chunks"])
    semantic_map, semantic_audit = _load_semantics(
        source_files["semantic_inventory"],
        source_files["semantic_records"],
        families,
    )
    request_manifest, request_map, _ = _validate_requests(
        files=source_files,
        families=families,
    )
    _validate_semantic_request_record_links(
        files=source_files,
        request_map=request_map,
    )
    request_binding = _request_binding_payload(
        files=source_files,
        request_manifest=request_manifest,
        request_count=len(request_map),
    )
    assignments = evidence.assign_family_splits(families, seed=seed)
    if Counter(assignments.values()) != Counter(
        {
            "train": 5,
            "validation": 3,
            "calibration": 3,
            "blind_test": 3,
        }
    ):
        raise NonblindSFTAuditV8Error("source-family split reconstruction mismatch")
    rows = _validate_rows_v7(
        dataset_files,
        families=families,
        semantic_map=semantic_map,
        assignments=assignments,
    )
    if (
        len(rows) != _EXPECTED_TOTAL
        or len({str(row["example_id"]) for row in rows}) != _EXPECTED_TOTAL
        or Counter(str(row["split"]) for row in rows) != Counter(_EXPECTED_SPLIT_COUNTS)
        or Counter(str(row["decision"]) for row in rows) != Counter({"ANSWER": 275, "REFUSE": 275})
        or set(str(row["task"]) for row in rows) != set(TASKS)
        or set(str(row["decision"]) for row in rows) != set(DECISIONS)
    ):
        raise NonblindSFTAuditV8Error("v8 example count/identity/contract mismatch")
    artifacts = {
        role: _strict_json_object(
            dataset_files[role].payload,
            label=f"{label} {role}",
        )
        for role in _ARTIFACT_ROLES
    }
    authorities = AuthorityState(
        files=source_files,
        code=code,
        nli=nli,
        rag_manifest_id=str(rag_payload["manifest_id"]),
        rag_binding=rag_binding,
        families=tuple(families),
        semantic_map=semantic_map,
        semantic_audit=semantic_audit,
        request_binding=request_binding,
        seed=seed,
    )
    _validate_stored_audits(
        artifacts,
        rows=rows,
        authorities=authorities,
    )
    _validate_manifest_nli_summary(
        manifest,
        artifacts=artifacts,
        authorities=authorities,
    )
    return DatasetState(
        root=root,
        root_identity=identity,
        files=dataset_files,
        manifest=manifest,
        artifacts=artifacts,
        rows=tuple(rows),
        authorities=authorities,
    )


def _recheck_state(state: DatasetState, *, label: str) -> None:
    _scan_fixed_inventory(
        state.root,
        root_identity=state.root_identity,
        label=label,
    )
    for role, snapshot in state.files.items():
        _recheck_snapshot_v7(snapshot, label=f"{label} {role}")
        _ensure_no_hardlink(
            snapshot.path,
            label=f"{label} {role}",
        )
    for role, snapshot in state.authorities.files.items():
        _recheck_snapshot_v7(
            snapshot,
            label=f"{label} authority {role}",
        )
        _ensure_no_hardlink(
            snapshot.path,
            label=f"{label} authority {role}",
        )
    for role, snapshot in state.authorities.code.items():
        _recheck_snapshot_v7(
            snapshot,
            label=f"{label} code {role}",
        )
        _ensure_no_hardlink(
            snapshot.path,
            label=f"{label} code {role}",
        )


def _assert_independent_builds(
    primary: DatasetState,
    secondary: DatasetState,
) -> None:
    try:
        same_root = os.path.samefile(primary.root, secondary.root)
    except OSError as exc:
        raise NonblindSFTAuditV8Error("independent directory identity cannot be verified") from exc
    if same_root or primary.root_identity == secondary.root_identity:
        raise NonblindSFTAuditV8Error("primary and secondary must be independent directories")
    for role in ROLE_FILENAMES:
        left = primary.files[role]
        right = secondary.files[role]
        try:
            same_file = os.path.samefile(left.path, right.path)
        except OSError as exc:
            raise NonblindSFTAuditV8Error(f"{role} independent identity cannot be verified") from exc
        if same_file or left.identity[:2] == right.identity[:2]:
            raise NonblindSFTAuditV8Error(f"{role} files are not independently materialized")
        if left.payload != right.payload:
            raise NonblindSFTAuditV8Error(f"{ROLE_FILENAMES[role]} byte mismatch")


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _prepare_output_dir(
    output_dir: Path,
    *,
    protected_roots: Sequence[Path],
) -> Path:
    lexical = _absolute(output_dir)
    if os.path.lexists(lexical):
        raise NonblindSFTAuditV8Error(f"output directory already exists: {lexical}")
    _assert_no_link_components(
        lexical,
        label="output directory",
        allow_missing_leaf=True,
    )
    parent = lexical.parent.resolve(strict=True)
    final = parent / lexical.name
    for root in protected_roots:
        resolved = root.resolve(strict=True)
        if _path_within(final, resolved) or _path_within(
            resolved,
            final,
        ):
            raise NonblindSFTAuditV8Error("output directory overlaps an audited input")
    try:
        os.mkdir(final)
    except FileExistsError as exc:
        raise NonblindSFTAuditV8Error(f"output directory already exists: {final}") from exc
    return final


def _receipt_payload(
    *,
    mode: str,
    status: str,
    primary: DatasetState,
    secondary: DatasetState | None,
    runner_path: Path | None,
) -> dict[str, Any]:
    files = {
        ROLE_FILENAMES[role]: {
            "sha256": snapshot.sha256,
            "bytes": snapshot.bytes,
        }
        for role, snapshot in primary.files.items()
    }
    implementation = {
        "auditor": {
            "path": Path(__file__).resolve().as_posix(),
            "sha256": _sha256_bytes(Path(__file__).read_bytes()),
        }
    }
    if runner_path is not None:
        runner = _snapshot_strict_file(
            runner_path,
            label="audit runner",
            maximum_bytes=_MAX_JSON_BYTES,
        )
        implementation["runner"] = {
            "path": runner.path.as_posix(),
            "sha256": runner.sha256,
        }
    body: dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "audit_version": AUDIT_VERSION,
        "status": status,
        "audit_passed": True,
        "mode": mode,
        "created_at": datetime.now(UTC).isoformat(),
        "primary": {
            "root": primary.root.as_posix(),
            "manifest_sha256": primary.files["manifest"].sha256,
            "output_content_sha256": primary.manifest["output_content_sha256"],
        },
        "authorities": {
            role: {
                "path": snapshot.path.as_posix(),
                "sha256": snapshot.sha256,
            }
            for role, snapshot in primary.authorities.files.items()
        },
        "nli_model": dict(primary.authorities.nli),
        "files": files,
        "file_count": len(files),
        "implementation": implementation,
        "reserved_asset_accessed": False,
        "production_connected": False,
        "x5_deployed": False,
    }
    if secondary is not None:
        body["secondary"] = {
            "root": secondary.root.as_posix(),
            "manifest_sha256": secondary.files["manifest"].sha256,
            "output_content_sha256": secondary.manifest["output_content_sha256"],
        }
        body["byte_identical"] = True
        body["independent_file_identity"] = True
    digest = _sha256_bytes(canonical_json(body).encode("utf-8"))
    receipt = {
        **body,
        "canonical_digest_sha256": digest,
    }
    receipt["receipt_sha256"] = _sha256_bytes(canonical_json(receipt).encode("utf-8"))
    return receipt


def _publish_receipt(
    *,
    output_dir: Path,
    receipt: Mapping[str, Any],
    states: Sequence[DatasetState],
    nli_model_dir: Path,
) -> dict[str, Any]:
    protected = [state.root for state in states]
    for state in states:
        protected.extend(snapshot.path for snapshot in state.authorities.files.values())
        protected.extend(snapshot.path for snapshot in state.authorities.code.values())
    protected.append(nli_model_dir.resolve(strict=True))
    directory = _prepare_output_dir(
        output_dir,
        protected_roots=protected,
    )
    path = directory / AUDIT_FILENAME
    payload = (
        json.dumps(
            receipt,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o444)
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise NonblindSFTAuditV8Error("receipt write made no progress")
            written += count
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        for index, state in enumerate(states, 1):
            _recheck_state(state, label=f"dataset {index}")
        final_nli = _validate_nli_asset(nli_model_dir)
        if final_nli != states[0].authorities.nli:
            raise NonblindSFTAuditV8Error("NLI model changed before receipt publication")
        observed = _snapshot_strict_file(
            path,
            label="published v8 audit receipt",
            maximum_bytes=_MAX_JSON_BYTES,
        )
        if observed.payload != payload:
            raise NonblindSFTAuditV8Error("published v8 audit receipt bytes changed")
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        try:
            directory.rmdir()
        except OSError:
            pass
        raise
    return {
        **dict(receipt),
        "path": path.as_posix(),
        "sha256": _sha256_bytes(payload),
    }


def _audit_nonblind_dataset_v8(
    *,
    primary: Path,
    licensed_chunks: Path,
    rag_manifest: Path,
    semantic_inventory: Path,
    nli_model_dir: Path,
    output_dir: Path,
    runner_path: Path | None,
) -> dict[str, Any]:
    state = _load_and_validate_dataset(
        primary,
        label="primary dataset",
        licensed_chunks=licensed_chunks,
        rag_manifest=rag_manifest,
        semantic_inventory=semantic_inventory,
        nli_model_dir=nli_model_dir,
    )
    _recheck_state(state, label="primary dataset")
    receipt = _receipt_payload(
        mode="single",
        status=AUDIT_PASS_STATUS,
        primary=state,
        secondary=None,
        runner_path=runner_path,
    )
    return _publish_receipt(
        output_dir=output_dir,
        receipt=receipt,
        states=(state,),
        nli_model_dir=nli_model_dir,
    )


def _compare_nonblind_datasets_v8(
    *,
    primary: Path,
    secondary: Path,
    licensed_chunks: Path,
    rag_manifest: Path,
    semantic_inventory: Path,
    nli_model_dir: Path,
    output_dir: Path,
    runner_path: Path | None,
) -> dict[str, Any]:
    primary_root, primary_identity = _safe_dataset_root(
        primary,
        label="primary dataset",
    )
    secondary_root, secondary_identity = _safe_dataset_root(
        secondary,
        label="secondary dataset",
    )
    if primary_identity == secondary_identity or os.path.samefile(primary_root, secondary_root):
        raise NonblindSFTAuditV8Error("primary and secondary must be independent directories")
    left = _load_and_validate_dataset(
        primary,
        label="primary dataset",
        licensed_chunks=licensed_chunks,
        rag_manifest=rag_manifest,
        semantic_inventory=semantic_inventory,
        nli_model_dir=nli_model_dir,
    )
    right = _load_and_validate_dataset(
        secondary,
        label="secondary dataset",
        licensed_chunks=licensed_chunks,
        rag_manifest=rag_manifest,
        semantic_inventory=semantic_inventory,
        nli_model_dir=nli_model_dir,
    )
    _assert_independent_builds(left, right)
    _recheck_state(left, label="primary dataset")
    _recheck_state(right, label="secondary dataset")
    receipt = _receipt_payload(
        mode="compare",
        status=COMPARE_PASS_STATUS,
        primary=left,
        secondary=right,
        runner_path=runner_path,
    )
    return _publish_receipt(
        output_dir=output_dir,
        receipt=receipt,
        states=(left, right),
        nli_model_dir=nli_model_dir,
    )


def audit_nonblind_dataset_v8(
    *,
    primary: Path,
    licensed_chunks: Path,
    rag_manifest: Path,
    semantic_inventory: Path,
    nli_model_dir: Path,
    output_dir: Path,
    runner_path: Path | None = None,
) -> dict[str, Any]:
    try:
        return _audit_nonblind_dataset_v8(
            primary=primary,
            licensed_chunks=licensed_chunks,
            rag_manifest=rag_manifest,
            semantic_inventory=semantic_inventory,
            nli_model_dir=nli_model_dir,
            output_dir=output_dir,
            runner_path=runner_path,
        )
    except NonblindSFTAuditV8Error:
        raise
    except v7audit.NonblindSFTAuditV7Error as exc:
        raise NonblindSFTAuditV8Error(str(exc)) from None
    except (OSError, ValueError) as exc:
        raise NonblindSFTAuditV8Error(str(exc)) from exc


def compare_nonblind_datasets_v8(
    *,
    primary: Path,
    secondary: Path,
    licensed_chunks: Path,
    rag_manifest: Path,
    semantic_inventory: Path,
    nli_model_dir: Path,
    output_dir: Path,
    runner_path: Path | None = None,
) -> dict[str, Any]:
    try:
        return _compare_nonblind_datasets_v8(
            primary=primary,
            secondary=secondary,
            licensed_chunks=licensed_chunks,
            rag_manifest=rag_manifest,
            semantic_inventory=semantic_inventory,
            nli_model_dir=nli_model_dir,
            output_dir=output_dir,
            runner_path=runner_path,
        )
    except NonblindSFTAuditV8Error:
        raise
    except v7audit.NonblindSFTAuditV7Error as exc:
        raise NonblindSFTAuditV8Error(str(exc)) from None
    except (OSError, ValueError) as exc:
        raise NonblindSFTAuditV8Error(str(exc)) from exc
