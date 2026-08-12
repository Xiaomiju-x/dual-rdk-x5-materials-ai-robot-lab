from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import icmat_foundry.llm.evidence_sft_v6 as builder_v6
from icmat_foundry.llm.evidence_sft_v6 import (
    BALANCE_AUDIT_SCHEMA,
    BLIND_FILENAME,
    BLIND_SEAL_SCHEMA,
    BUILD_REPORT_SCHEMA,
    BUILDER_VERSION,
    DATASET_SCHEMA,
    EXAMPLE_SCHEMA,
    EXPECTED_FAMILY_SPLIT_COUNTS,
    EXPECTED_SPLIT_COUNTS,
    GROUP_AUDIT_SCHEMA,
    LEAKAGE_AUDIT_SCHEMA,
    MANIFEST_SCHEMA,
    NONBLIND_SPLITS,
    SPLITS,
    TASKS,
    TRAINING_SPLITS,
    canonical_json,
)

AUDIT_SCHEMA = "icmat_evidence_pointer_independent_reproducibility_audit.v6"
AUDIT_VERSION = "icmat-evidence-sft-independent-audit-v6.0.0"
AUDIT_FILENAME = "independent_audit.v6.json"
PASS_STATUS = "PASS_INDEPENDENT_BYTE_REPRODUCIBILITY_VERIFIED"
ERROR_STATUS = "FAILED_NO_INDEPENDENT_REPRODUCIBILITY_AUDIT"

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_READ_BLOCK_BYTES = 1024 * 1024

_ROLE_FILENAMES = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "calibration": "calibration.jsonl",
    "blind_test": BLIND_FILENAME,
    "balance_audit": "balance_audit.v6.json",
    "group_isolation_audit": "group_isolation_audit.v6.json",
    "content_leakage_audit": "content_leakage_audit.v6.json",
    "blind_seal": "blind_test.seal.v6.json",
    "build_report": "build_report.v6.json",
    "manifest": "manifest.v6.json",
}
_ARTIFACT_FILENAMES = {
    "balance_audit": _ROLE_FILENAMES["balance_audit"],
    "group_isolation_audit": _ROLE_FILENAMES["group_isolation_audit"],
    "content_leakage_audit": _ROLE_FILENAMES["content_leakage_audit"],
    "blind_seal": _ROLE_FILENAMES["blind_seal"],
    "build_report": _ROLE_FILENAMES["build_report"],
}
_EXPECTED_CLAIMS = {
    "knowledge_distillation": False,
    "licensed_evidence_sft": True,
    "local_measurement": False,
    "production_connected": False,
    "x5_verified": False,
}
_EXPECTED_TRAINING_BOUNDARY = {
    "allowed_splits": list(TRAINING_SPLITS),
    "calibration_content_for_training": False,
    "forbidden_split": "blind_test",
    "blind_test_requires_explicit_post_freeze_authorization": True,
    "blind_test_content_in_public_reports": False,
}
_PUBLIC_FORBIDDEN_KEYS = frozenset(
    {
        "source_id",
        "doi",
        "source_title",
        "source_uri",
        "example_id",
        "messages",
        "requested_claim",
        "span_id",
        "target_span_id",
        "compiler_prompt",
        "compiler_evidence",
        "response_provenance",
        "expected_pointer",
        "expected_answer",
        "raw_pointer",
        "members",
    }
)


class EvidenceSFTAuditV6Error(ValueError):
    pass


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    payload: bytes
    sha256: str
    bytes: int
    identity: tuple[int, int, int, int]


@dataclass(frozen=True)
class DatasetSnapshot:
    root: Path
    files: Mapping[str, FileSnapshot]
    manifest: Mapping[str, Any]
    public_artifacts: Mapping[str, Mapping[str, Any]]
    nonblind_example_count: int


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


def _snapshot_regular_file(path: Path, *, label: str) -> FileSnapshot:
    lexical = Path(path)
    if lexical.is_symlink():
        raise EvidenceSFTAuditV6Error(f"{label} must not be a symlink")
    try:
        lexical_stat = lexical.lstat()
    except FileNotFoundError as exc:
        raise EvidenceSFTAuditV6Error(f"{label} is missing") from exc
    if not stat.S_ISREG(lexical_stat.st_mode):
        raise EvidenceSFTAuditV6Error(f"{label} must be a regular file")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(os.fspath(lexical), flags)
    except OSError as exc:
        raise EvidenceSFTAuditV6Error(f"{label} cannot be opened") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise EvidenceSFTAuditV6Error(f"{label} must remain a regular file")
        blocks: list[bytes] = []
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
        or lexical.is_symlink()
    ):
        raise EvidenceSFTAuditV6Error(f"{label} changed while it was inspected")
    payload = b"".join(blocks)
    if len(payload) != int(after.st_size):
        raise EvidenceSFTAuditV6Error(f"{label} byte count is unstable")
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
                raise EvidenceSFTAuditV6Error(f"{label} contains a duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise EvidenceSFTAuditV6Error(f"{label} contains a non-finite JSON number")

    try:
        text = payload.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except UnicodeDecodeError as exc:
        raise EvidenceSFTAuditV6Error(f"{label} is not UTF-8 JSON") from exc
    except json.JSONDecodeError as exc:
        raise EvidenceSFTAuditV6Error(f"{label} is invalid JSON") from exc


def _strict_json_object(
    payload: bytes,
    *,
    label: str,
) -> dict[str, Any]:
    value = _strict_json(payload, label=label)
    if not isinstance(value, dict):
        raise EvidenceSFTAuditV6Error(f"{label} must contain one JSON object")
    return value


def _assert_public_payload_sanitized(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> None:
    def inspect(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if str(key) in _PUBLIC_FORBIDDEN_KEYS:
                    raise EvidenceSFTAuditV6Error(f"{label} exposes sealed-content fields")
                inspect(nested)
        elif isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            for nested in value:
                inspect(nested)

    inspect(payload)


def _validate_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _HEX_SHA256.fullmatch(value):
        raise EvidenceSFTAuditV6Error(f"{label} is not a lowercase SHA-256")
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
        raise EvidenceSFTAuditV6Error(f"{label} receipt is missing")
    expected_keys = {"path", "sha256", "bytes"}
    if expected_count is not None:
        expected_keys.add("count")
    if set(receipt) != expected_keys:
        raise EvidenceSFTAuditV6Error(f"{label} receipt keys do not match the v6 contract")
    if receipt.get("path") != expected_path:
        raise EvidenceSFTAuditV6Error(f"{label} receipt path does not match the fixed inventory")
    if (
        _validate_sha256(
            receipt.get("sha256"),
            label=f"{label} receipt SHA-256",
        )
        != snapshot.sha256
    ):
        raise EvidenceSFTAuditV6Error(f"{label} receipt hash mismatch")
    if not _is_integer(receipt.get("bytes")) or (int(receipt["bytes"]) != snapshot.bytes):
        raise EvidenceSFTAuditV6Error(f"{label} receipt byte mismatch")
    if expected_count is not None and (
        not _is_integer(receipt.get("count")) or int(receipt["count"]) != expected_count
    ):
        raise EvidenceSFTAuditV6Error(f"{label} receipt count mismatch")


def _validate_nonblind_jsonl(
    snapshot: FileSnapshot,
    *,
    split: str,
    expected_count: int,
) -> tuple[set[str], set[str]]:
    if not snapshot.payload.endswith(b"\n"):
        raise EvidenceSFTAuditV6Error(f"{split} JSONL must end with one newline")
    lines = snapshot.payload.splitlines()
    if len(lines) != expected_count or any(not line for line in lines):
        raise EvidenceSFTAuditV6Error(f"{split} JSONL row count mismatch")
    example_ids: set[str] = set()
    source_ids: set[str] = set()
    for index, line in enumerate(lines, 1):
        row = _strict_json(
            line,
            label=f"{split} JSONL row {index}",
        )
        if not isinstance(row, dict):
            raise EvidenceSFTAuditV6Error(f"{split} JSONL row {index} must be an object")
        if (
            row.get("schema") != EXAMPLE_SCHEMA
            or row.get("dataset_schema") != DATASET_SCHEMA
            or row.get("split") != split
        ):
            raise EvidenceSFTAuditV6Error(f"{split} JSONL row {index} contract mismatch")
        example_id = row.get("example_id")
        source_id = row.get("source_id")
        if (
            not isinstance(example_id, str)
            or not example_id
            or not isinstance(source_id, str)
            or not source_id
        ):
            raise EvidenceSFTAuditV6Error(f"{split} JSONL row {index} identity is invalid")
        if example_id in example_ids:
            raise EvidenceSFTAuditV6Error(f"{split} JSONL contains duplicate example identities")
        example_ids.add(example_id)
        source_ids.add(source_id)
    return example_ids, source_ids


def _validate_balance_audit(payload: Mapping[str, Any]) -> None:
    if (
        payload.get("schema") != BALANCE_AUDIT_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("findings") != []
        or payload.get("split_counts") != EXPECTED_SPLIT_COUNTS
        or payload.get("imbalanced_family_count") != 0
    ):
        raise EvidenceSFTAuditV6Error("stored balance audit is not a passing v6 audit")
    decisions = payload.get("split_decision_counts")
    tasks = payload.get("split_task_counts")
    if not isinstance(decisions, dict) or not isinstance(tasks, dict):
        raise EvidenceSFTAuditV6Error("stored balance audit counters are missing")
    for split, count in EXPECTED_SPLIT_COUNTS.items():
        if decisions.get(split) != {
            "ANSWER": count // 2,
            "REFUSE": count // 2,
        }:
            raise EvidenceSFTAuditV6Error("stored decision balance is invalid")
        split_tasks = tasks.get(split)
        if (
            not isinstance(split_tasks, dict)
            or set(split_tasks) != set(TASKS)
            or any(not _is_integer(split_tasks[task]) or int(split_tasks[task]) <= 0 for task in TASKS)
            or sum(int(split_tasks[task]) for task in TASKS) != count
        ):
            raise EvidenceSFTAuditV6Error("stored task balance is invalid")


def _validate_group_audit(
    payload: Mapping[str, Any],
) -> Mapping[str, list[str]]:
    if (
        payload.get("schema") != GROUP_AUDIT_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("findings") != []
        or payload.get("isolation_unit") != "licensed DOI/source family"
    ):
        raise EvidenceSFTAuditV6Error("stored group-isolation audit is not PASS")
    commitments = payload.get("group_commitments")
    if not isinstance(commitments, dict) or set(commitments) != set(SPLITS):
        raise EvidenceSFTAuditV6Error("stored group commitments are invalid")
    seen: set[str] = set()
    normalized: dict[str, list[str]] = {}
    for split in SPLITS:
        values = commitments.get(split)
        if (
            not isinstance(values, list)
            or len(values) != EXPECTED_FAMILY_SPLIT_COUNTS[split]
            or any(not isinstance(value, str) for value in values)
            or values != sorted(values)
        ):
            raise EvidenceSFTAuditV6Error("stored group commitment shape is invalid")
        normalized[split] = []
        for value in values:
            commitment = _validate_sha256(
                value,
                label="group commitment",
            )
            if commitment in seen:
                raise EvidenceSFTAuditV6Error("stored group commitments overlap")
            seen.add(commitment)
            normalized[split].append(commitment)

    pairwise = payload.get("pairwise")
    if not isinstance(pairwise, list) or len(pairwise) != 6:
        raise EvidenceSFTAuditV6Error("stored pairwise group audit is incomplete")
    expected_pairs = {(left, right) for index, left in enumerate(SPLITS) for right in SPLITS[index + 1 :]}
    observed_pairs: set[tuple[str, str]] = set()
    for row in pairwise:
        if not isinstance(row, dict):
            raise EvidenceSFTAuditV6Error("stored pairwise group audit row is invalid")
        left = row.get("left")
        right = row.get("right")
        if not isinstance(left, str) or not isinstance(right, str):
            raise EvidenceSFTAuditV6Error("stored pairwise group labels are invalid")
        pair = (left, right)
        if pair not in expected_pairs:
            raise EvidenceSFTAuditV6Error("stored pairwise group labels are invalid")
        observed_pairs.add(pair)
        for key in (
            "source_overlap_count",
            "doi_overlap_count",
            "commitment_overlap_count",
        ):
            if row.get(key) != 0:
                raise EvidenceSFTAuditV6Error("stored pairwise group overlap is nonzero")
    if observed_pairs != expected_pairs:
        raise EvidenceSFTAuditV6Error("stored pairwise group coverage is incomplete")
    return normalized


def _validate_leakage_audit(payload: Mapping[str, Any]) -> None:
    if (
        payload.get("schema") != LEAKAGE_AUDIT_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("findings") != []
    ):
        raise EvidenceSFTAuditV6Error("stored content-leakage audit is not PASS")
    zero_fields = (
        "exact_claim_overlap_count",
        "exact_prompt_overlap_count",
        "exact_compiler_evidence_overlap_count",
        "near_duplicate_claim_pair_count",
        "compiler_prompt_target_marker_count",
        "compiler_evidence_target_marker_count",
        "compiler_prompt_assistant_message_count",
        "compiler_interface_missing_count",
    )
    if any(payload.get(key) != 0 for key in zero_fields):
        raise EvidenceSFTAuditV6Error("stored content-leakage counters are nonzero")


def _validate_blind_seal(
    payload: Mapping[str, Any],
    *,
    blind_receipt: Mapping[str, Any],
    blind_commitments: Sequence[str],
) -> None:
    expected_commitment_hash = _sha256_bytes(canonical_json(list(blind_commitments)).encode("utf-8"))
    if (
        payload.get("schema") != BLIND_SEAL_SCHEMA
        or payload.get("builder_version") != BUILDER_VERSION
        or payload.get("sealed") is not True
        or payload.get("authorization_required") is not True
        or payload.get("authorized_for_training") is not False
        or payload.get("authorized_for_checkpoint_selection") is not False
        or payload.get("content_disclosed") is not False
        or payload.get("blind_test_file") != blind_receipt
        or payload.get("group_commitment_sha256") != expected_commitment_hash
    ):
        raise EvidenceSFTAuditV6Error("stored blind seal is invalid")


def _validate_build_report(
    payload: Mapping[str, Any],
    *,
    blind_receipt: Mapping[str, Any],
) -> None:
    blind_summary = payload.get("blind_test")
    if (
        payload.get("schema") != BUILD_REPORT_SCHEMA
        or payload.get("builder_version") != BUILDER_VERSION
        or payload.get("status") != "PASS_DATASET_BUILT_BLIND_HASH_SEALED"
        or payload.get("audits")
        != {
            "balance": "PASS",
            "group_isolation": "PASS",
            "content_leakage": "PASS",
        }
        or payload.get("claims") != _EXPECTED_CLAIMS
        or not isinstance(blind_summary, dict)
        or blind_summary
        != {
            "sealed": True,
            "content_disclosed": False,
            "count": blind_receipt["count"],
            "sha256": blind_receipt["sha256"],
            "bytes": blind_receipt["bytes"],
        }
    ):
        raise EvidenceSFTAuditV6Error("stored build report is not a passing sealed v6 report")
    counts = payload.get("counts")
    if (
        not isinstance(counts, dict)
        or counts.get("examples") != sum(EXPECTED_SPLIT_COUNTS.values())
        or counts.get("families") != sum(EXPECTED_FAMILY_SPLIT_COUNTS.values())
        or counts.get("splits") != EXPECTED_SPLIT_COUNTS
    ):
        raise EvidenceSFTAuditV6Error("stored build report counts are invalid")


def _resolve_declared_file(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise EvidenceSFTAuditV6Error(f"{label} path is invalid")
    path = Path(value)
    if not path.is_absolute():
        raise EvidenceSFTAuditV6Error(f"{label} path must be absolute")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise EvidenceSFTAuditV6Error(f"{label} path cannot be resolved") from exc


def _validate_manifest(
    dataset: DatasetSnapshot,
    *,
    builder_snapshot: FileSnapshot,
) -> tuple[dict[str, FileSnapshot], str]:
    manifest = dataset.manifest
    counts = manifest.get("counts")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("dataset_schema") != DATASET_SCHEMA
        or manifest.get("builder_version") != BUILDER_VERSION
        or manifest.get("status") != "DATASET_BUILT_BLIND_HASH_SEALED"
        or not isinstance(counts, dict)
        or counts.get("splits") != EXPECTED_SPLIT_COUNTS
        or counts.get("examples") != sum(EXPECTED_SPLIT_COUNTS.values())
        or counts.get("families") != sum(EXPECTED_FAMILY_SPLIT_COUNTS.values())
        or manifest.get("training_boundary") != _EXPECTED_TRAINING_BOUNDARY
        or manifest.get("claims") != _EXPECTED_CLAIMS
    ):
        raise EvidenceSFTAuditV6Error("manifest state does not match the sealed v6 contract")

    split_receipts = manifest.get("splits")
    if not isinstance(split_receipts, dict) or set(split_receipts) != set(SPLITS):
        raise EvidenceSFTAuditV6Error("manifest split inventory is invalid")
    for split in SPLITS:
        _validate_receipt(
            split_receipts[split],
            snapshot=dataset.files[split],
            expected_path=_ROLE_FILENAMES[split],
            expected_count=EXPECTED_SPLIT_COUNTS[split],
            label=split,
        )

    artifact_receipts = manifest.get("artifacts")
    if not isinstance(artifact_receipts, dict) or set(artifact_receipts) != set(_ARTIFACT_FILENAMES):
        raise EvidenceSFTAuditV6Error("manifest public artifact inventory is invalid")
    for role, filename in _ARTIFACT_FILENAMES.items():
        _validate_receipt(
            artifact_receipts[role],
            snapshot=dataset.files[role],
            expected_path=filename,
            expected_count=None,
            label=role,
        )

    builder = manifest.get("builder")
    if not isinstance(builder, dict) or set(builder) != {
        "path",
        "sha256",
    }:
        raise EvidenceSFTAuditV6Error("manifest builder receipt is invalid")
    declared_builder = _resolve_declared_file(
        builder.get("path"),
        label="builder source",
    )
    if declared_builder != builder_snapshot.path:
        raise EvidenceSFTAuditV6Error("manifest builder path does not match current v6 source")
    if (
        _validate_sha256(
            builder.get("sha256"),
            label="builder source SHA-256",
        )
        != builder_snapshot.sha256
    ):
        raise EvidenceSFTAuditV6Error("manifest builder source hash is stale")

    source_inputs = manifest.get("source_inputs")
    if not isinstance(source_inputs, dict) or set(source_inputs) != {"licensed_chunks", "rag_manifest"}:
        raise EvidenceSFTAuditV6Error("manifest RAG input inventory is invalid")
    snapshots: dict[str, FileSnapshot] = {}
    for role in ("licensed_chunks", "rag_manifest"):
        receipt = source_inputs.get(role)
        expected_keys = {"path", "sha256", "manifest_id"} if role == "rag_manifest" else {"path", "sha256"}
        if not isinstance(receipt, dict) or set(receipt) != expected_keys:
            raise EvidenceSFTAuditV6Error(f"{role} source receipt is invalid")
        path = _resolve_declared_file(
            receipt.get("path"),
            label=role,
        )
        snapshot = _snapshot_regular_file(path, label=role)
        if (
            _validate_sha256(
                receipt.get("sha256"),
                label=f"{role} SHA-256",
            )
            != snapshot.sha256
        ):
            raise EvidenceSFTAuditV6Error(f"{role} current hash does not match the manifest")
        snapshots[role] = snapshot

    rag_manifest = _strict_json_object(
        snapshots["rag_manifest"].payload,
        label="RAG manifest",
    )
    manifest_id = rag_manifest.get("manifest_id")
    declared_rag = source_inputs["rag_manifest"]
    if (
        rag_manifest.get("schema") != "icmat.rag.manifest.v2"
        or not isinstance(manifest_id, str)
        or not manifest_id
        or declared_rag.get("manifest_id") != manifest_id
    ):
        raise EvidenceSFTAuditV6Error("current RAG manifest identity is invalid")
    return snapshots, manifest_id


def _dataset_root(path: Path, *, label: str) -> Path:
    lexical = Path(path)
    if lexical.is_symlink():
        raise EvidenceSFTAuditV6Error(f"{label} directory must not be a symlink")
    try:
        root = lexical.resolve(strict=True)
    except OSError as exc:
        raise EvidenceSFTAuditV6Error(f"{label} directory is missing") from exc
    if not root.is_dir():
        raise EvidenceSFTAuditV6Error(f"{label} must be a directory")
    return root


def _snapshot_dataset(path: Path, *, label: str) -> DatasetSnapshot:
    root = _dataset_root(path, label=label)
    entries = list(root.iterdir())
    actual_names = {entry.name for entry in entries}
    expected_names = set(_ROLE_FILENAMES.values())
    if actual_names != expected_names:
        raise EvidenceSFTAuditV6Error(f"{label} does not have the exact v6 artifact inventory")
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise EvidenceSFTAuditV6Error(f"{label} contains a non-regular artifact")

    files = {
        role: _snapshot_regular_file(
            root / filename,
            label=f"{label} {role}",
        )
        for role, filename in _ROLE_FILENAMES.items()
    }
    manifest = _strict_json_object(
        files["manifest"].payload,
        label=f"{label} manifest",
    )

    all_ids: set[str] = set()
    split_sources: dict[str, set[str]] = {}
    nonblind_count = 0
    for split in NONBLIND_SPLITS:
        ids, sources = _validate_nonblind_jsonl(
            files[split],
            split=split,
            expected_count=EXPECTED_SPLIT_COUNTS[split],
        )
        if all_ids & ids:
            raise EvidenceSFTAuditV6Error(f"{label} repeats example identities across splits")
        all_ids.update(ids)
        split_sources[split] = sources
        nonblind_count += len(ids)
    for index, left in enumerate(NONBLIND_SPLITS):
        for right in NONBLIND_SPLITS[index + 1 :]:
            if split_sources[left] & split_sources[right]:
                raise EvidenceSFTAuditV6Error(f"{label} repeats source families across nonblind splits")

    public_artifacts = {
        role: _strict_json_object(
            files[role].payload,
            label=f"{label} {role}",
        )
        for role in _ARTIFACT_FILENAMES
    }
    for role, payload in public_artifacts.items():
        _assert_public_payload_sanitized(
            payload,
            label=f"{label} {role}",
        )
    return DatasetSnapshot(
        root=root,
        files=files,
        manifest=manifest,
        public_artifacts=public_artifacts,
        nonblind_example_count=nonblind_count,
    )


def _validate_dataset_state(
    dataset: DatasetSnapshot,
    *,
    builder_snapshot: FileSnapshot,
) -> tuple[dict[str, FileSnapshot], str]:
    rag_inputs, manifest_id = _validate_manifest(
        dataset,
        builder_snapshot=builder_snapshot,
    )
    balance = dataset.public_artifacts["balance_audit"]
    group = dataset.public_artifacts["group_isolation_audit"]
    leakage = dataset.public_artifacts["content_leakage_audit"]
    blind_seal = dataset.public_artifacts["blind_seal"]
    build_report = dataset.public_artifacts["build_report"]

    _validate_balance_audit(balance)
    commitments = _validate_group_audit(group)
    _validate_leakage_audit(leakage)
    blind_receipt = dataset.manifest["splits"]["blind_test"]
    _validate_blind_seal(
        blind_seal,
        blind_receipt=blind_receipt,
        blind_commitments=commitments["blind_test"],
    )
    _validate_build_report(
        build_report,
        blind_receipt=blind_receipt,
    )
    if dataset.nonblind_example_count != sum(EXPECTED_SPLIT_COUNTS[split] for split in NONBLIND_SPLITS):
        raise EvidenceSFTAuditV6Error("nonblind example count is incomplete")
    return rag_inputs, manifest_id


def _same_regular_file(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except OSError as exc:
        raise EvidenceSFTAuditV6Error("artifact file identity could not be verified") from exc


def _assert_builds_are_independent_and_equal(
    left: DatasetSnapshot,
    right: DatasetSnapshot,
) -> None:
    if left.root == right.root:
        raise EvidenceSFTAuditV6Error("the two dataset directories must be distinct")
    for role in _ROLE_FILENAMES:
        left_file = left.files[role]
        right_file = right.files[role]
        if _same_regular_file(left_file.path, right_file.path):
            raise EvidenceSFTAuditV6Error(f"{role} is shared rather than independently materialized")
        if (
            left_file.bytes != right_file.bytes
            or left_file.sha256 != right_file.sha256
            or left_file.payload != right_file.payload
        ):
            raise EvidenceSFTAuditV6Error(f"{role} is not byte-for-byte reproducible")


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _prepare_output_path(
    output_path: Path,
    *,
    dataset_roots: Sequence[Path],
) -> Path:
    output = Path(output_path).resolve(strict=False)
    if output.name != AUDIT_FILENAME:
        raise EvidenceSFTAuditV6Error(f"output filename must be {AUDIT_FILENAME}")
    if os.path.lexists(output):
        raise EvidenceSFTAuditV6Error("independent audit output already exists")
    if any(_path_is_within(output, root) for root in dataset_roots):
        raise EvidenceSFTAuditV6Error("independent audit output must be outside both datasets")
    return output


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


def _exclusive_write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(os.fspath(path), flags, 0o600)
    except FileExistsError as exc:
        raise EvidenceSFTAuditV6Error("independent audit output already exists") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # Keep a partial exclusive file so a failed receipt path is not reused.
        raise
    return path.resolve(strict=True)


def _implementation_inventory(
    *,
    builder_snapshot: FileSnapshot,
    runner_path: Path | None,
) -> tuple[dict[str, dict[str, Any]], list[FileSnapshot]]:
    paths: dict[str, Path] = {
        "builder": builder_snapshot.path,
        "independent_auditor": Path(__file__).resolve(strict=True),
    }
    if runner_path is not None:
        paths["runner"] = Path(runner_path).resolve(strict=True)
    result: dict[str, dict[str, Any]] = {}
    snapshots: list[FileSnapshot] = []
    for role, path in paths.items():
        snapshot = (
            builder_snapshot if role == "builder" else _snapshot_regular_file(path, label=f"{role} source")
        )
        snapshots.append(snapshot)
        result[role] = {
            "path": snapshot.path.name,
            "bytes": snapshot.bytes,
            "sha256": snapshot.sha256,
        }
    return result, snapshots


def _recheck_snapshots(snapshots: Sequence[FileSnapshot]) -> None:
    seen: set[Path] = set()
    for expected in snapshots:
        if expected.path in seen:
            continue
        seen.add(expected.path)
        current = _snapshot_regular_file(
            expected.path,
            label="pre-write evidence",
        )
        if (
            current.bytes != expected.bytes
            or current.sha256 != expected.sha256
            or current.payload != expected.payload
        ):
            raise EvidenceSFTAuditV6Error("evidence changed before receipt creation")


def audit_dataset_reproducibility_v6(
    *,
    dataset_a: Path,
    dataset_b: Path,
    output_path: Path,
    runner_path: Path | None = None,
) -> dict[str, Any]:
    """Compare two independent v6 builds and write one immutable audit."""

    builder_snapshot = _snapshot_regular_file(
        Path(builder_v6.__file__).resolve(strict=True),
        label="current v6 builder source",
    )
    left = _snapshot_dataset(dataset_a, label="dataset_a")
    right = _snapshot_dataset(dataset_b, label="dataset_b")
    output = _prepare_output_path(
        output_path,
        dataset_roots=(left.root, right.root),
    )
    _assert_builds_are_independent_and_equal(left, right)

    left_rag, left_manifest_id = _validate_dataset_state(
        left,
        builder_snapshot=builder_snapshot,
    )
    right_rag, right_manifest_id = _validate_dataset_state(
        right,
        builder_snapshot=builder_snapshot,
    )
    if left_manifest_id != right_manifest_id:
        raise EvidenceSFTAuditV6Error("the two builds do not bind the same RAG manifest")
    for role in ("licensed_chunks", "rag_manifest"):
        if (
            left_rag[role].path != right_rag[role].path
            or left_rag[role].bytes != right_rag[role].bytes
            or left_rag[role].sha256 != right_rag[role].sha256
            or left_rag[role].payload != right_rag[role].payload
        ):
            raise EvidenceSFTAuditV6Error("the two builds do not bind identical RAG input bytes")

    implementation, implementation_snapshots = _implementation_inventory(
        builder_snapshot=builder_snapshot,
        runner_path=runner_path,
    )
    blind = left.files["blind_test"]
    inventory = {
        role: {
            "path": _ROLE_FILENAMES[role],
            "bytes": left.files[role].bytes,
            "sha256": left.files[role].sha256,
            "byte_identical": True,
            "json_parsed": role != "blind_test",
        }
        for role in _ROLE_FILENAMES
    }
    body = {
        "schema": AUDIT_SCHEMA,
        "version": AUDIT_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": PASS_STATUS,
        "reproducibility_passed": True,
        "dataset_contract": {
            "schema": DATASET_SCHEMA,
            "manifest_schema": MANIFEST_SCHEMA,
            "builder_version": BUILDER_VERSION,
            "manifest_status": "DATASET_BUILT_BLIND_HASH_SEALED",
            "stored_audits": {
                "balance": "PASS",
                "group_isolation": "PASS",
                "content_leakage": "PASS",
            },
            "nonblind_examples_structurally_revalidated": (left.nonblind_example_count),
        },
        "independent_builds": {
            "directories_distinct": True,
            "artifact_file_identities_distinct": True,
            "dataset_a_root_fingerprint_sha256": _sha256_bytes(os.fspath(left.root).encode("utf-8")),
            "dataset_b_root_fingerprint_sha256": _sha256_bytes(os.fspath(right.root).encode("utf-8")),
            "manifest_sha256": left.files["manifest"].sha256,
        },
        "artifact_reproducibility": {
            "expected_file_count": len(_ROLE_FILENAMES),
            "exact_inventory_verified": True,
            "all_files_byte_identical": True,
            "files": inventory,
        },
        "blind_test": {
            "path": BLIND_FILENAME,
            "sealed": True,
            "declared_count": EXPECTED_SPLIT_COUNTS["blind_test"],
            "bytes": blind.bytes,
            "sha256": blind.sha256,
            "hash_only_access": True,
            "decoded": False,
            "json_parsed": False,
            "raw_payload_persisted_in_report": False,
            "content_disclosed": False,
        },
        "rag_inputs": {
            "licensed_chunks": {
                "bytes": left_rag["licensed_chunks"].bytes,
                "sha256": left_rag["licensed_chunks"].sha256,
            },
            "rag_manifest": {
                "schema": "icmat.rag.manifest.v2",
                "manifest_id": left_manifest_id,
                "bytes": left_rag["rag_manifest"].bytes,
                "sha256": left_rag["rag_manifest"].sha256,
            },
            "current_bytes_rehashed": True,
        },
        "implementation": implementation,
        "evidence_boundary": {
            "complete_artifact_byte_identity_verified": True,
            "same_rag_source_bytes_verified": True,
            "seed_value_directly_attested_by_v6_manifest": False,
            "seed_equivalence_evidence": (
                "complete independent artifact byte identity; the current "
                "v6 builder manifest does not persist the seed string"
            ),
            "model_quality_or_x5_claim_authorized": False,
        },
        "pre_write_evidence_recheck": True,
        "report_contains_blind_source_sample_or_span": False,
    }
    _assert_public_payload_sanitized(
        body,
        label="independent audit report",
    )
    receipt = {
        **body,
        "canonical_digest_sha256": _sha256_bytes(canonical_json(body).encode("utf-8")),
    }
    receipt["receipt_payload_sha256"] = _sha256_bytes(canonical_json(receipt).encode("utf-8"))
    _assert_public_payload_sanitized(
        receipt,
        label="independent audit receipt",
    )

    evidence_snapshots = [
        *left.files.values(),
        *right.files.values(),
        *left_rag.values(),
        *right_rag.values(),
        *implementation_snapshots,
    ]
    _recheck_snapshots(evidence_snapshots)
    payload = _json_bytes(receipt)
    written = _exclusive_write(output, payload)
    persisted = written.read_bytes()
    if persisted != payload:
        raise EvidenceSFTAuditV6Error("persisted independent audit bytes differ")
    return {
        "status": PASS_STATUS,
        "reproducibility_passed": True,
        "path": written.as_posix(),
        "sha256": _sha256_bytes(payload),
        "canonical_digest_sha256": receipt["canonical_digest_sha256"],
        "receipt": receipt,
    }


def audit_evidence_sft_v6(
    *,
    dataset_a: Path,
    dataset_b: Path,
    output_path: Path,
    runner_path: Path | None = None,
) -> dict[str, Any]:
    return audit_dataset_reproducibility_v6(
        dataset_a=dataset_a,
        dataset_b=dataset_b,
        output_path=output_path,
        runner_path=runner_path,
    )


__all__ = [
    "AUDIT_FILENAME",
    "AUDIT_SCHEMA",
    "AUDIT_VERSION",
    "ERROR_STATUS",
    "PASS_STATUS",
    "EvidenceSFTAuditV6Error",
    "audit_dataset_reproducibility_v6",
    "audit_evidence_sft_v6",
]
