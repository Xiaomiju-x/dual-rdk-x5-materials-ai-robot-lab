"""Strict one-shot post-freeze evaluation for the ICMat v7 pointer model.

This module enforces an auditable execution protocol:

* selection and contracts are independently recomputed before authorization;
* one global claim is keyed only by the preblind commitment SHA-256;
* the claim is created with O_EXCL and fsync before split assignment, example
  building, or any reserved-test membership derivation;
* exactly 150 rows are deterministically derived from the committed seed,
  builder code, and source snapshots;
* the in-memory rows are evaluated once with the frozen HF adapter on CUDA;
* success, failure, and process abandonment are all non-reusable; and
* a passing run authorizes only an offline GGUF candidate build.

The protocol is an execution-control and evidence mechanism for an honest
local execution environment, not cryptographic secrecy or administrator-
forgery resistance. The committed seed, builder, and source assets are locally
available, so an administrator could derive or forge evidence outside this
implementation. No TPM, external signer, or append-only third-party ledger is
claimed; the receipts state that limitation explicitly.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import traceback
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from icmat_foundry.llm import (
    ablation_eval_v6,
    ablation_eval_v7,
    calibration_eval_v6,
    calibration_eval_v7,
    contracts_v7,
    evidence_pointer_v6,
    evidence_sft_v6,
    lifecycle_bindings_v7,
    nonblind_sft_v7,
    pointer_hf_eval_v6,
    selection_freeze_v7,
)

PROTOCOL_VERSION = "icmat-postfreeze-blind-v7.0.0"
AUTHORIZATION_SCHEMA = "icmat_llm_postfreeze_authorization.v7"
AUTHORIZATION_STATUS = "AUTHORIZED_ONCE_POSTFREEZE_GATES_PASS"
CLAIM_SCHEMA = "icmat_llm_postfreeze_consumption_claim.v7"
TERMINAL_SCHEMA = "icmat_llm_postfreeze_terminal.v7"
SAMPLE_SCHEMA = "icmat_llm_postfreeze_sample.v7"
SUMMARY_SCHEMA = "icmat_llm_postfreeze_summary.v7"
RUN_RECEIPT_SCHEMA = "icmat_llm_postfreeze_run_receipt.v7"
QUALIFICATION_SCHEMA = "icmat_llm_postfreeze_gguf_qualification.v7"

QUALIFICATION_PASS_STATUS = "PASS_GGUF_OFFLINE_CANDIDATE_ONLY"
QUALIFICATION_HOLD_STATUS = "HOLD_POSTFREEZE_THRESHOLDS_NON_REUSABLE"
RUN_COMPLETE_STATUS = "POSTFREEZE_V7_ONE_SHOT_COMPLETE"

EXPECTED_ROWS = 150
EXPECTED_FAMILIES = 3
EXAMPLES_PER_FAMILY = 50
FIXED_SEED = 20260729
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_SOURCE_BYTES = 512 * 1024 * 1024
MAX_ERROR_CHARS = 2000

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_REGISTRY_ROOT = (
    WORKSPACE_ROOT
    / "evaluation"
    / "icmat_foundry"
    / "llm"
    / "postfreeze_blind_v7_registry"
)

DERIVED_FILENAME = "derived_test.v7.jsonl"
SAMPLE_FILENAME = "sample_results.v7.jsonl"
SUMMARY_FILENAME = "summary.v7.json"
RUN_RECEIPT_FILENAME = "run_receipt.v7.json"
QUALIFICATION_FILENAME = "release_qualification.v7.json"
EVIDENCE_FILENAMES = {
    DERIVED_FILENAME,
    SAMPLE_FILENAME,
    SUMMARY_FILENAME,
    RUN_RECEIPT_FILENAME,
    QUALIFICATION_FILENAME,
}

RELEASE_POLICY = {
    "expected_examples": EXPECTED_ROWS,
    "required_backend": "hf_model",
    "required_device": "cuda",
    "model_bound_required": True,
    "pointer_parse_valid": {"numerator": 1, "denominator": 1},
    "pointer_compiler_accepted": {"numerator": 1, "denominator": 1},
    "unsupported_wrong_answer_max_count": 0,
    "compiled_schema_valid": {"numerator": 1, "denominator": 1},
    "compiled_citation_exact": {"numerator": 1, "denominator": 1},
    "compiled_provenance_exact": {"numerator": 1, "denominator": 1},
    "answer_span_exact_minimum": {"numerator": 19, "denominator": 20},
    "refuse_f1_minimum": {"numerator": 19, "denominator": 20},
    "gguf_offline_candidate_allowed": True,
    "activation_authorized": False,
    "deployment_authorized": False,
    "production_integration_authorized": False,
    "model_selection_allowed": False,
    "checkpoint_ranking_allowed": False,
    "threshold_tuning_allowed": False,
    "calibration_allowed": False,
    "retry_after_claim_allowed": False,
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{7,127}$")
_SOURCE_ROLES = (
    "licensed_chunks",
    "rag_manifest",
    "semantic_inventory",
    "semantic_records",
    "nonblind_module",
    "evidence_core",
)


class PostfreezeBlindV7Error(RuntimeError):
    """Raised when the strict v7 one-shot protocol fails closed."""


@dataclass(frozen=True)
class StableSnapshot:
    """A stable regular-file byte snapshot."""

    path: Path
    payload: bytes
    identity: tuple[int, int, int, int, int]
    sha256: str

    @property
    def bytes(self) -> int:
        return len(self.payload)

    def receipt(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "bytes": self.bytes,
            "sha256": self.sha256,
            "stable_identity": {
                "st_dev": self.identity[0],
                "st_ino": self.identity[1],
                "st_size": self.identity[2],
                "st_mtime_ns": self.identity[3],
                "st_ctime_ns": self.identity[4],
            },
        }


@dataclass(frozen=True)
class DirectoryAnchor:
    """Stable identity for a caller-managed real directory."""

    path: Path
    identity: tuple[int, int, int, int, int]

    def receipt(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "stable_identity": {
                "st_dev": self.identity[0],
                "st_ino": self.identity[1],
                "st_size": self.identity[2],
                "st_mtime_ns": self.identity[3],
                "st_ctime_ns": self.identity[4],
            },
        }


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
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


def _jsonl_bytes(values: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (canonical_json(dict(value)) + "\n").encode("utf-8")
        for value in values
    )


def _reject_nonfinite(value: str) -> None:
    raise PostfreezeBlindV7Error(
        f"non-finite JSON constant rejected: {value}"
    )


def _reject_duplicate_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise PostfreezeBlindV7Error(
                f"duplicate JSON key rejected: {key}"
            )
        output[key] = value
    return output


def _parse_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PostfreezeBlindV7Error(
            f"{label}: invalid UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise PostfreezeBlindV7Error(f"{label}: JSON object required")
    return value


def _parse_jsonl_bytes(
    payload: bytes,
    *,
    label: str,
) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PostfreezeBlindV7Error(
            f"{label}: invalid UTF-8 JSONL"
        ) from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise PostfreezeBlindV7Error(
                f"{label}: blank line {line_number}"
            )
        try:
            value = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=_reject_nonfinite,
            )
        except json.JSONDecodeError as exc:
            raise PostfreezeBlindV7Error(
                f"{label}: invalid line {line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise PostfreezeBlindV7Error(
                f"{label}: line {line_number} must be an object"
            )
        rows.append(value)
    return rows


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PostfreezeBlindV7Error(f"{label}: object required")
    return value


def _require_exact_keys(
    value: Any,
    expected: set[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    mapping = _require_mapping(value, label=label)
    if set(mapping) != expected:
        raise PostfreezeBlindV7Error(
            f"{label}: exact field set mismatch"
        )
    return mapping


def _require_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise PostfreezeBlindV7Error(
            f"{label}: lowercase SHA-256 required"
        )
    return value


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _is_reparse(value: os.stat_result) -> bool:
    attributes = int(getattr(value, "st_file_attributes", 0))
    marker = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return bool(attributes & marker)


def _assert_no_reparse_chain(path: Path, *, label: str) -> Path:
    lexical = Path(path).expanduser().absolute()
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as exc:
            raise PostfreezeBlindV7Error(
                f"{label}: path is unavailable"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
            raise PostfreezeBlindV7Error(
                f"{label}: symlink/reparse component rejected"
            )
    return lexical


def _capture_file(
    path: Path,
    *,
    label: str,
    maximum_bytes: int = MAX_JSON_BYTES,
) -> StableSnapshot:
    lexical = _assert_no_reparse_chain(path, label=label)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lexical, flags)
    except OSError as exc:
        raise PostfreezeBlindV7Error(
            f"{label}: file could not be opened"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > maximum_bytes
        ):
            raise PostfreezeBlindV7Error(
                f"{label}: regular file size invalid"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, maximum_bytes + 1),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise PostfreezeBlindV7Error(
                    f"{label}: file exceeds size limit"
                )
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.lstat(lexical)
    payload = b"".join(chunks)
    identity = _identity(before)
    if (
        identity != _identity(after)
        or identity != _identity(current)
        or len(payload) != identity[2]
    ):
        raise PostfreezeBlindV7Error(f"{label}: TOCTOU detected")
    return StableSnapshot(
        path=lexical.resolve(strict=True),
        payload=payload,
        identity=identity,
        sha256=sha256_bytes(payload),
    )


def _verify_snapshot(snapshot: StableSnapshot, *, label: str) -> None:
    observed = _capture_file(
        snapshot.path,
        label=label,
        maximum_bytes=max(snapshot.bytes, 1),
    )
    if observed != snapshot:
        raise PostfreezeBlindV7Error(
            f"{label}: stable snapshot changed"
        )


def _directory_anchor(path: Path, *, label: str) -> DirectoryAnchor:
    lexical = _assert_no_reparse_chain(path, label=label)
    metadata = os.lstat(lexical)
    if not stat.S_ISDIR(metadata.st_mode):
        raise PostfreezeBlindV7Error(f"{label}: real directory required")
    return DirectoryAnchor(
        path=lexical.resolve(strict=True),
        identity=_identity(metadata),
    )


def _verify_directory(anchor: DirectoryAnchor, *, label: str) -> None:
    observed = _directory_anchor(anchor.path, label=label)
    if (
        observed.path != anchor.path
        or observed.identity[:2] != anchor.identity[:2]
    ):
        raise PostfreezeBlindV7Error(f"{label}: identity changed")


def _fsync_parent(path: Path) -> bool:
    if os.name == "nt":
        return False
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


def _exclusive_create(path: Path, payload: bytes) -> dict[str, Any]:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
    )
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # A partial claim or authorization remains intentionally non-reusable.
        raise
    parent_fsynced = _fsync_parent(path.parent)
    return {
        "path": str(path.resolve(strict=True)),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "file_fsync_completed": True,
        "parent_fsync_supported_and_completed": parent_fsynced,
    }


def _universe_id(commitment_sha256: str) -> str:
    digest = _require_sha(
        commitment_sha256,
        label="preblind commitment SHA",
    )
    return sha256_bytes(
        b"icmat-v7-derived-test\0" + digest.encode("ascii")
    )


def _registry_paths(
    registry: DirectoryAnchor,
    *,
    commitment_sha256: str,
) -> dict[str, Path]:
    universe = _universe_id(commitment_sha256)
    paths = {
        "authorization": (
            registry.path / f"{universe}.authorization.v7.json"
        ),
        "claim": registry.path / f"{universe}.claim.v7.json",
        "terminal": registry.path / f"{universe}.terminal.v7.json",
        "evidence": registry.path / f"{universe}.evidence.v7",
    }
    if any(path.parent != registry.path for path in paths.values()):
        raise PostfreezeBlindV7Error("registry path escaped its root")
    return paths


def _production_registry_root(*, create: bool) -> Path:
    """Return the sole workspace registry used by production entrypoints."""

    root = PRODUCTION_REGISTRY_ROOT
    if create:
        root.mkdir(parents=True, exist_ok=True)
    try:
        resolved = root.resolve(strict=True)
        workspace = WORKSPACE_ROOT.resolve(strict=True)
    except OSError as exc:
        raise PostfreezeBlindV7Error(
            "the fixed workspace postfreeze registry is unavailable"
        ) from exc
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise PostfreezeBlindV7Error(
            "the fixed postfreeze registry escaped the workspace"
        ) from exc
    if resolved != PRODUCTION_REGISTRY_ROOT.resolve(strict=True):
        raise PostfreezeBlindV7Error(
            "the fixed postfreeze registry identity changed"
        )
    return resolved


def _snapshot_set_receipt(
    snapshots: Mapping[str, StableSnapshot],
) -> dict[str, Any]:
    if tuple(snapshots) != _SOURCE_ROLES:
        raise PostfreezeBlindV7Error(
            "source snapshot role ordering mismatch"
        )
    files = {
        role: snapshots[role].receipt()
        for role in _SOURCE_ROLES
    }
    digest_payload = {
        role: {
            "bytes": files[role]["bytes"],
            "sha256": files[role]["sha256"],
        }
        for role in _SOURCE_ROLES
    }
    return {
        "roles": list(_SOURCE_ROLES),
        "files": files,
        "content_set_sha256": canonical_sha256(digest_payload),
    }


def _capture_source_set(
    *,
    licensed_chunks_path: Path,
    rag_manifest_path: Path,
    semantic_inventory_path: Path,
) -> dict[str, StableSnapshot]:
    records_path = Path(semantic_inventory_path).with_name(
        "records.v7.jsonl"
    )
    paths = {
        "licensed_chunks": Path(licensed_chunks_path),
        "rag_manifest": Path(rag_manifest_path),
        "semantic_inventory": Path(semantic_inventory_path),
        "semantic_records": records_path,
        "nonblind_module": Path(nonblind_sft_v7.__file__),
        "evidence_core": Path(evidence_sft_v6.__file__),
    }
    return {
        role: _capture_file(
            path,
            label=f"source snapshot {role}",
            maximum_bytes=MAX_SOURCE_BYTES,
        )
        for role, path in paths.items()
    }


def _validate_commitment_and_manifest(
    *,
    dataset_dir: Path,
    preblind_commitment_path: Path,
    snapshots: Mapping[str, StableSnapshot],
) -> dict[str, Any]:
    dataset = _directory_anchor(dataset_dir, label="nonblind dataset")
    commitment_snapshot = _capture_file(
        preblind_commitment_path,
        label="preblind commitment",
    )
    expected_commitment_path = (
        dataset.path / selection_freeze_v7.COMMITMENT_NAME
    )
    if commitment_snapshot.path != expected_commitment_path:
        raise PostfreezeBlindV7Error(
            "preblind commitment is not the frozen dataset artifact"
        )
    commitment = _parse_json_bytes(
        commitment_snapshot.payload,
        label="preblind commitment",
    )
    nonblind_sft_v7._assert_preblind_commitment_sanitized(commitment)
    body = dict(commitment)
    commitment_sha = _require_sha(
        body.pop("commitment_sha256"),
        label="preblind commitment digest",
    )
    if canonical_sha256(body) != commitment_sha:
        raise PostfreezeBlindV7Error(
            "preblind commitment canonical digest mismatch"
        )
    if (
        commitment.get("schema")
        != nonblind_sft_v7.PREBLIND_COMMITMENT_SCHEMA
        or commitment.get("status")
        != "PREBLIND_COMMITTED_NONBLIND_ONLY"
        or commitment.get("builder_version")
        != nonblind_sft_v7.NONBLIND_BUILDER_VERSION
        or commitment.get("core_builder_version")
        != evidence_sft_v6.BUILDER_VERSION
        or commitment.get("split_algorithm_version")
        != nonblind_sft_v7.SPLIT_ALGORITHM_VERSION
        or commitment.get("expected_blind_count") != EXPECTED_ROWS
    ):
        raise PostfreezeBlindV7Error(
            "preblind commitment identity mismatch"
        )
    seed = commitment.get("seed")
    if (
        not isinstance(seed, str)
        or not seed
        or commitment.get("seed_sha256")
        != sha256_bytes(seed.encode("utf-8"))
    ):
        raise PostfreezeBlindV7Error("committed seed is invalid")

    code = _require_mapping(
        commitment.get("builder_code"),
        label="commitment builder_code",
    )
    inputs = _require_mapping(
        commitment.get("source_inputs"),
        label="commitment source_inputs",
    )
    expected_hashes = {
        "licensed_chunks": inputs.get("licensed_chunks_sha256"),
        "rag_manifest": inputs.get("rag_manifest_sha256"),
        "semantic_inventory": inputs.get("semantic_inventory_sha256"),
        "semantic_records": inputs.get("semantic_records_sha256"),
        "nonblind_module": code.get("nonblind_module_sha256"),
        "evidence_core": code.get("evidence_core_sha256"),
    }
    for role, expected in expected_hashes.items():
        if snapshots[role].sha256 != _require_sha(
            expected,
            label=f"commitment {role} SHA",
        ):
            raise PostfreezeBlindV7Error(
                f"committed source snapshot changed: {role}"
            )

    manifest_snapshot = _capture_file(
        dataset.path / selection_freeze_v7.MANIFEST_NAME,
        label="nonblind manifest",
    )
    manifest = _parse_json_bytes(
        manifest_snapshot.payload,
        label="nonblind manifest",
    )
    if (
        manifest.get("schema")
        != nonblind_sft_v7.NONBLIND_MANIFEST_SCHEMA
        or manifest.get("status")
        != "NONBLIND_DATASET_BUILT_PREBLIND_COMMITTED"
        or manifest.get("builder_version")
        != nonblind_sft_v7.NONBLIND_BUILDER_VERSION
        or manifest.get("core_builder_version")
        != evidence_sft_v6.BUILDER_VERSION
    ):
        raise PostfreezeBlindV7Error("nonblind manifest identity mismatch")
    splits = _require_mapping(
        manifest.get("splits"),
        label="nonblind manifest splits",
    )
    if set(splits) != set(evidence_sft_v6.NONBLIND_SPLITS):
        raise PostfreezeBlindV7Error(
            "nonblind manifest exposes an invalid split set"
        )
    for split, expected_count in (
        nonblind_sft_v7.EXPECTED_NONBLIND_SPLIT_COUNTS.items()
    ):
        descriptor = _require_mapping(
            splits.get(split),
            label=f"manifest split {split}",
        )
        if (
            descriptor.get("path") != f"{split}.jsonl"
            or descriptor.get("count") != expected_count
            or isinstance(descriptor.get("bytes"), bool)
            or not isinstance(descriptor.get("bytes"), int)
            or int(descriptor["bytes"]) <= 0
        ):
            raise PostfreezeBlindV7Error(
                f"manifest split descriptor invalid: {split}"
            )
        _require_sha(
            descriptor.get("sha256"),
            label=f"manifest split {split} SHA",
        )

    source_inputs = _require_mapping(
        manifest.get("source_inputs"),
        label="manifest source_inputs",
    )
    expected_paths = {
        "licensed_chunks": snapshots["licensed_chunks"].path,
        "rag_manifest": snapshots["rag_manifest"].path,
        "semantic_inventory": snapshots["semantic_inventory"].path,
    }
    for role, expected_path in expected_paths.items():
        descriptor = _require_mapping(
            source_inputs.get(role),
            label=f"manifest source input {role}",
        )
        try:
            recorded_path = Path(str(descriptor.get("path"))).resolve(
                strict=True
            )
        except OSError as exc:
            raise PostfreezeBlindV7Error(
                f"manifest source path unavailable: {role}"
            ) from exc
        if (
            recorded_path != expected_path
            or descriptor.get("sha256") != snapshots[role].sha256
        ):
            raise PostfreezeBlindV7Error(
                f"manifest source binding mismatch: {role}"
            )
    inventory_descriptor = _require_mapping(
        source_inputs.get("semantic_inventory"),
        label="manifest semantic inventory",
    )
    if (
        inventory_descriptor.get("records_sha256")
        != snapshots["semantic_records"].sha256
    ):
        raise PostfreezeBlindV7Error(
            "manifest semantic-record binding mismatch"
        )
    builder = _require_mapping(
        manifest.get("builder"),
        label="manifest builder",
    )
    for role in ("nonblind_module", "evidence_core"):
        descriptor = _require_mapping(
            builder.get(role),
            label=f"manifest builder {role}",
        )
        if (
            Path(str(descriptor.get("path"))).resolve(strict=True)
            != snapshots[role].path
            or descriptor.get("sha256") != snapshots[role].sha256
        ):
            raise PostfreezeBlindV7Error(
                f"manifest builder binding mismatch: {role}"
            )
    if (
        builder.get("seed") != seed
        or builder.get("split_algorithm_version")
        != commitment["split_algorithm_version"]
    ):
        raise PostfreezeBlindV7Error(
            "manifest seed or split algorithm differs from commitment"
        )
    artifacts = _require_mapping(
        manifest.get("artifacts"),
        label="manifest artifacts",
    )
    committed_artifact = _require_mapping(
        artifacts.get("preblind_commitment"),
        label="manifest commitment artifact",
    )
    if (
        committed_artifact.get("path")
        != selection_freeze_v7.COMMITMENT_NAME
        or committed_artifact.get("sha256")
        != commitment_snapshot.sha256
        or committed_artifact.get("bytes") != commitment_snapshot.bytes
    ):
        raise PostfreezeBlindV7Error(
            "manifest commitment file receipt mismatch"
        )
    rag_payload = _parse_json_bytes(
        snapshots["rag_manifest"].payload,
        label="RAG manifest",
    )
    if rag_payload.get("manifest_id") != inputs.get("rag_manifest_id"):
        raise PostfreezeBlindV7Error(
            "RAG manifest ID differs from commitment"
        )
    return {
        "dataset": dataset,
        "manifest_snapshot": manifest_snapshot,
        "manifest": manifest,
        "commitment_snapshot": commitment_snapshot,
        "commitment": commitment,
        "commitment_sha256": commitment_sha,
        "seed": seed,
    }


def _verify_public_authority(
    *,
    selection_freeze_path: Path,
    evaluation_index_path: Path,
    training_receipt_path: Path,
    dataset_dir: Path,
    base_model_dir: Path,
    preblind_commitment_path: Path,
    contract_dir: Path,
) -> dict[str, Any]:
    try:
        selection = selection_freeze_v7.verify_selection_freeze_v7(
            freeze_receipt_path=selection_freeze_path,
            evaluation_index_path=evaluation_index_path,
            training_receipt_path=training_receipt_path,
            dataset_dir=dataset_dir,
            base_model_dir=base_model_dir,
        )
        contracts = contracts_v7.verify_contracts_v7(
            selection_freeze=selection_freeze_path,
            preblind_commitment=preblind_commitment_path,
            evaluation_index=evaluation_index_path,
            training_receipt=training_receipt_path,
            dataset_dir=dataset_dir,
            base_model_dir=base_model_dir,
            contract_dir=contract_dir,
        )
        lifecycle = lifecycle_bindings_v7.capture_lifecycle_binding_v7(
            selection_freeze_path=selection_freeze_path,
            evaluation_index_path=evaluation_index_path,
            training_receipt_path=training_receipt_path,
            dataset_dir=dataset_dir,
            base_model_dir=base_model_dir,
            preblind_commitment_path=preblind_commitment_path,
            contract_dir=contract_dir,
        )
    except (
        selection_freeze_v7.SelectionFreezeV7Error,
        contracts_v7.ContractsV7Error,
        lifecycle_bindings_v7.LifecycleBindingV7Error,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise PostfreezeBlindV7Error(
            "selection/contracts v7 public verification failed"
        ) from exc
    if (
        selection.get("status") != selection_freeze_v7.VERIFIED_STATUS
        or selection.get("selection_locked") is not True
        or selection.get("blind_test_authorized") is not False
        or selection.get("deployment_authorized") is not False
        or contracts.get("status")
        != "PASS_NONBLIND_V7_CONTRACTS_VERIFIED"
    ):
        raise PostfreezeBlindV7Error(
            "selection/contracts verifier returned an invalid boundary"
        )
    binding = lifecycle.binding
    if (
        binding["selection"]["verification_status"]
        != selection_freeze_v7.VERIFIED_STATUS
        or binding["selection"]["selection_binding_digest_sha256"]
        != selection["selection_binding_digest_sha256"]
        or binding["contracts"]["contract_set_sha256"]
        != contracts["contract_set_sha256"]
    ):
        raise PostfreezeBlindV7Error(
            "lifecycle differs from public verifier results"
        )
    return {
        "selection": dict(selection),
        "contracts": dict(contracts),
        "lifecycle": lifecycle,
    }


def _capture_directory_artifacts(
    directory: Path,
    *,
    expected_names: set[str],
    label: str,
) -> tuple[DirectoryAnchor, dict[str, StableSnapshot]]:
    anchor = _directory_anchor(directory, label=label)
    names = {path.name for path in anchor.path.iterdir()}
    if names != expected_names:
        raise PostfreezeBlindV7Error(
            f"{label}: artifact whitelist mismatch"
        )
    snapshots = {
        name: _capture_file(
            anchor.path / name,
            label=f"{label} {name}",
            maximum_bytes=MAX_SOURCE_BYTES,
        )
        for name in sorted(expected_names)
    }
    _verify_directory(anchor, label=f"{label} final")
    return anchor, snapshots


def _validate_artifact_descriptors(
    receipt: Mapping[str, Any],
    snapshots: Mapping[str, StableSnapshot],
    *,
    receipt_filename: str,
    label: str,
) -> None:
    artifacts = _require_mapping(
        receipt.get("artifacts"),
        label=f"{label} artifacts",
    )
    expected = set(snapshots) - {receipt_filename}
    if set(artifacts) != expected:
        raise PostfreezeBlindV7Error(
            f"{label}: artifact descriptor set mismatch"
        )
    for name in expected:
        descriptor = _require_mapping(
            artifacts[name],
            label=f"{label} artifact {name}",
        )
        snapshot = snapshots[name]
        if (
            descriptor.get("bytes") != snapshot.bytes
            or descriptor.get("sha256") != snapshot.sha256
        ):
            raise PostfreezeBlindV7Error(
                f"{label}: artifact hash mismatch: {name}"
            )


def _generation_from_record(
    value: Any,
    *,
    label: str,
    latency_required: bool,
) -> pointer_hf_eval_v6.GenerationResultV6:
    generation = _require_mapping(value, label=label)
    expected_fields = {
        "raw_pointer",
        "raw_pointer_sha256",
        "finish_reason",
        "finish_category",
        "generation_error",
        "input_tokens",
        "output_tokens",
    }
    if latency_required:
        expected_fields.update({"latency_ms", "trusted_finish_reason"})
    _require_exact_keys(generation, expected_fields, label=label)
    raw_pointer = generation.get("raw_pointer")
    finish_reason = generation.get("finish_reason")
    finish_category = generation.get("finish_category")
    generation_error = generation.get("generation_error")
    input_tokens = generation.get("input_tokens")
    output_tokens = generation.get("output_tokens")
    latency = generation.get("latency_ms", 0.0)
    if (
        not isinstance(raw_pointer, str)
        or generation.get("raw_pointer_sha256")
        != sha256_bytes(raw_pointer.encode("utf-8"))
        or not isinstance(finish_reason, str)
        or not isinstance(finish_category, str)
        or finish_category
        != pointer_hf_eval_v6._finish_category(finish_reason)
        or (
            latency_required
            and generation.get("trusted_finish_reason")
            != (
                finish_reason in pointer_hf_eval_v6.TRUSTED_FINISH_REASONS
            )
        )
        or (
            generation_error is not None
            and not isinstance(generation_error, str)
        )
        or isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or float(latency) < 0.0
    ):
        raise PostfreezeBlindV7Error(f"{label} is invalid")
    for field, item in (
        ("input_tokens", input_tokens),
        ("output_tokens", output_tokens),
    ):
        if item is not None and (
            isinstance(item, bool) or not isinstance(item, int) or item < 0
        ):
            raise PostfreezeBlindV7Error(f"{label}.{field} is invalid")
    return pointer_hf_eval_v6.GenerationResultV6(
        raw_pointer=raw_pointer,
        finish_reason=finish_reason,
        finish_category=finish_category,
        latency_ms=float(latency),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        generation_error=generation_error,
    )


def _model_tree_inventory(
    path: Path,
    *,
    label: str,
) -> dict[str, Any]:
    try:
        return pointer_hf_eval_v6._tree_inventory(path)
    except (
        pointer_hf_eval_v6.PointerHFEvalV6Error,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise PostfreezeBlindV7Error(
            f"{label} model tree cannot be independently inventoried"
        ) from exc


def _casefold_inventory_tree_sha256(
    inventory: Mapping[str, Any],
    *,
    label: str,
) -> str:
    files = inventory.get("files")
    if (
        isinstance(files, (str, bytes))
        or not isinstance(files, Sequence)
        or not files
    ):
        raise PostfreezeBlindV7Error(
            f"{label} model inventory has no complete file list"
        )
    normalized: list[dict[str, Any]] = []
    for position, raw_record in enumerate(files):
        record = _require_mapping(
            raw_record,
            label=f"{label} model inventory file {position}",
        )
        _require_exact_keys(
            record,
            {"path", "bytes", "sha256"},
            label=f"{label} model inventory file {position}",
        )
        path = record.get("path")
        byte_count = record.get("bytes")
        sha256 = record.get("sha256")
        if (
            not isinstance(path, str)
            or not path
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            raise PostfreezeBlindV7Error(
                f"{label} model inventory file {position} is invalid"
            )
        normalized.append(
            {
                "path": path,
                "bytes": byte_count,
                "sha256": _require_sha(
                    sha256,
                    label=f"{label} model inventory file {position} SHA",
                ),
            }
        )
    ordered = sorted(
        normalized,
        key=lambda record: (
            str(record["path"]).casefold(),
            str(record["path"]),
        ),
    )
    return canonical_sha256(ordered)


def _capture_full_model_tree_binding(
    model: Mapping[str, Any],
    *,
    adapter_required: bool,
    label: str,
) -> dict[str, Any]:
    base_inventory = _model_tree_inventory(
        Path(str(model.get("base_model_path"))),
        label=f"{label} base",
    )
    if (
        _casefold_inventory_tree_sha256(
            base_inventory,
            label=f"{label} base",
        )
        != model.get("base_model_tree_sha256")
    ):
        raise PostfreezeBlindV7Error(
            f"{label} base-model frozen tree binding mismatch"
        )
    adapter_inventory: dict[str, Any] | None = None
    if adapter_required:
        adapter_path = Path(
            str(
                model.get("adapter_runtime_path")
                or model.get("checkpoint_path")
            )
        )
        adapter_inventory = _model_tree_inventory(
            adapter_path,
            label=f"{label} adapter",
        )
        if (
            _casefold_inventory_tree_sha256(
                adapter_inventory,
                label=f"{label} adapter",
            )
            != model.get("checkpoint_tree_sha256")
        ):
            raise PostfreezeBlindV7Error(
                f"{label} adapter-model frozen tree binding mismatch"
            )
    return {
        "inventory_algorithm": "pointer_hf_eval_v6._tree_inventory.v1",
        "base": base_inventory,
        "adapter": adapter_inventory,
    }


def _verify_backend_full_model_tree(
    backend: Mapping[str, Any],
    *,
    model: Mapping[str, Any],
    adapter_required: bool,
    label: str,
) -> dict[str, Any]:
    runtime_model = _require_mapping(
        backend.get("model"),
        label=f"{label} model",
    )
    _require_exact_keys(
        runtime_model,
        {"base", "adapter", "inventories_unchanged_after_generation"},
        label=f"{label} model",
    )
    current_binding = _capture_full_model_tree_binding(
        model,
        adapter_required=adapter_required,
        label=label,
    )
    base_inventory = current_binding["base"]
    runtime_base = _require_mapping(
        runtime_model.get("base"),
        label=f"{label} base inventory",
    )
    if (
        dict(runtime_base) != base_inventory
        or runtime_model.get("inventories_unchanged_after_generation")
        is not True
    ):
        raise PostfreezeBlindV7Error(
            f"{label} base-model tree binding mismatch"
        )
    runtime_adapter = runtime_model.get("adapter")
    adapter_inventory = current_binding["adapter"]
    if adapter_required:
        if (
            not isinstance(runtime_adapter, Mapping)
            or dict(runtime_adapter) != adapter_inventory
        ):
            raise PostfreezeBlindV7Error(
                f"{label} adapter-model tree binding mismatch"
            )
    elif runtime_adapter is not None:
        raise PostfreezeBlindV7Error(
            f"{label} unexpectedly binds an adapter tree"
        )
    authorized_binding = model.get("full_model_tree_binding")
    if authorized_binding is not None:
        authorized = _require_mapping(
            authorized_binding,
            label=f"{label} authorized full model tree",
        )
        if dict(authorized) != current_binding:
            raise PostfreezeBlindV7Error(
                f"{label} full model tree changed after authorization"
            )
    return {
        "base": base_inventory,
        "adapter": adapter_inventory,
    }


def _dataset_id_order_receipt(
    rows: Sequence[pointer_hf_eval_v6.DatasetRowV6],
) -> dict[str, Any]:
    example_ids = [row.example_id for row in rows]
    if (
        len(example_ids) != EXPECTED_ROWS
        or len(set(example_ids)) != EXPECTED_ROWS
    ):
        raise PostfreezeBlindV7Error(
            "postselection dataset ID order is incomplete"
        )
    return {
        "rows": EXPECTED_ROWS,
        "example_ids_sha256": canonical_sha256(example_ids),
        "first_example_id": example_ids[0],
        "last_example_id": example_ids[-1],
    }


def _implementation_runner_path(
    implementation: Mapping[str, Any],
    *,
    label: str,
) -> Path | None:
    runner = implementation.get("runner")
    if runner is None:
        return None
    descriptor = _require_mapping(runner, label=f"{label} runner")
    path = descriptor.get("path")
    if not isinstance(path, str) or not path:
        raise PostfreezeBlindV7Error(f"{label} runner path is invalid")
    return Path(path)


def _independently_verify_calibration_gate(
    *,
    receipt: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    lifecycle: lifecycle_bindings_v7.LifecycleSnapshotV7,
) -> dict[str, Any]:
    """Reload calibration membership and deterministically rebuild all metrics."""

    try:
        split = lifecycle_bindings_v7.capture_dataset_split_v7(
            lifecycle,
            split="calibration",
        )
        if len(split.rows) != calibration_eval_v7.EXPECTED_ROWS:
            raise PostfreezeBlindV7Error(
                "calibration split is not the exact 150-row split"
            )
        if len(samples) != calibration_eval_v7.EXPECTED_ROWS:
            raise PostfreezeBlindV7Error(
                "calibration evidence does not contain exactly 150 samples"
            )
        dataset = _require_mapping(
            receipt.get("dataset"),
            label="calibration dataset",
        )
        if dataset.get("file") != split.file.receipt():
            raise PostfreezeBlindV7Error(
                "calibration split SHA or stable identity mismatch"
            )
        order = _dataset_id_order_receipt(split.rows)
        observed_ids = [str(row.get("example_id")) for row in samples]
        expected_ids = [row.example_id for row in split.rows]
        if observed_ids != expected_ids:
            raise PostfreezeBlindV7Error(
                "calibration sample ID order differs from the frozen split"
            )
        implementation = _require_mapping(
            receipt.get("implementation"),
            label="calibration implementation",
        )
        _, current_implementation = calibration_eval_v7._source_snapshots(
            _implementation_runner_path(
                implementation,
                label="calibration implementation",
            )
        )
        if dict(implementation) != current_implementation:
            raise PostfreezeBlindV7Error(
                "calibration implementation sources changed"
            )
        backend = _require_mapping(
            receipt.get("backend"),
            label="calibration backend",
        )
        if (
            backend.get("mode") != "hf_model"
            or backend.get("device") not in {"cpu", "cuda"}
        ):
            raise PostfreezeBlindV7Error(
                "calibration backend is not model-bound HF evidence"
            )
        model = _require_mapping(
            receipt.get("model"),
            label="calibration model",
        )
        _verify_backend_full_model_tree(
            backend,
            model=model,
            adapter_required=True,
            label="calibration backend",
        )
        bindings = calibration_eval_v7._sample_bindings(
            lifecycle,
            implementation=implementation,
            backend=backend,
        )
        source_rows = [
            pointer_hf_eval_v6._score_row(
                row=dataset_row,
                generation=_generation_from_record(
                    recorded.get("generation"),
                    label=f"calibration generation {dataset_row.example_id}",
                    latency_required=True,
                ),
                bindings=bindings,
                backend_mode="hf_model",
            )
            for dataset_row, recorded in zip(
                split.rows,
                samples,
                strict=True,
            )
        ]
        recomputed_samples, recomputed_summary = (
            calibration_eval_v7._v7_results(
                source_rows,
                backend_mode="hf_model",
                model_bound=True,
                lifecycle=lifecycle,
                implementation=implementation,
            )
        )
    except PostfreezeBlindV7Error:
        raise
    except (
        calibration_eval_v6.CalibrationEvalV6Error,
        calibration_eval_v7.CalibrationEvalV7Error,
        lifecycle_bindings_v7.LifecycleBindingV7Error,
        pointer_hf_eval_v6.PointerHFEvalV6Error,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise PostfreezeBlindV7Error(
            "calibration deterministic verification failed"
        ) from exc
    if list(samples) != recomputed_samples:
        raise PostfreezeBlindV7Error(
            "calibration samples differ from independent recompilation"
        )
    if dict(summary) != recomputed_summary:
        raise PostfreezeBlindV7Error(
            "calibration summary differs from recomputed metrics"
        )
    if (
        receipt.get("quality_gate_passed")
        is not recomputed_summary.get("quality_gate_passed")
        or receipt.get("conformal_threshold")
        != _require_mapping(
            recomputed_summary.get("conformal"),
            label="recomputed calibration conformal metrics",
        ).get("threshold")
    ):
        raise PostfreezeBlindV7Error(
            "calibration receipt metrics differ from recomputed summary"
        )
    return {
        "dataset_split_sha256": split.file.sha256,
        "dataset_id_order": order,
        "samples_recompiled": True,
        "summary_recomputed": True,
        "backend_model_tree_reverified": True,
    }


def _strip_ablation_v7_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    stripped = json.loads(canonical_json(dict(row)))
    for field in (
        "ablation_version",
        "lifecycle_binding_sha256",
        "v6_math_implementation_sha256",
    ):
        if field not in stripped:
            raise PostfreezeBlindV7Error(
                f"ablation sample misses v7 binding: {field}"
            )
        stripped.pop(field)
    boundaries = _require_mapping(
        stripped.get("boundaries"),
        label="ablation sample boundaries",
    )
    normalized_boundaries = dict(boundaries)
    for field in ("fixture_not_model_evidence", "reserved_data_accessed"):
        if field not in normalized_boundaries:
            raise PostfreezeBlindV7Error(
                f"ablation sample boundary is absent: {field}"
            )
        normalized_boundaries.pop(field)
    stripped["boundaries"] = normalized_boundaries
    stripped["schema"] = ablation_eval_v6.SAMPLE_SCHEMA
    return stripped


def _independently_verify_ablation_gate(
    *,
    receipt: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    reports: Mapping[str, Mapping[str, Any]],
    lifecycle: lifecycle_bindings_v7.LifecycleSnapshotV7,
) -> dict[str, Any]:
    """Reload validation membership and rebuild the full ablation matrix."""

    try:
        split = lifecycle_bindings_v7.capture_dataset_split_v7(
            lifecycle,
            split="validation",
        )
        if len(split.rows) != ablation_eval_v7.EXPECTED_VALIDATION_ROWS:
            raise PostfreezeBlindV7Error(
                "ablation split is not the exact 150-row validation split"
            )
        if len(samples) != ablation_eval_v7.EXPECTED_SAMPLE_ROWS:
            raise PostfreezeBlindV7Error(
                "ablation evidence does not contain the full sample matrix"
            )
        dataset = _require_mapping(
            receipt.get("dataset"),
            label="ablation dataset",
        )
        if dataset.get("file") != split.file.receipt():
            raise PostfreezeBlindV7Error(
                "ablation split SHA or stable identity mismatch"
            )
        order = _dataset_id_order_receipt(split.rows)
        implementation = _require_mapping(
            receipt.get("implementation"),
            label="ablation implementation",
        )
        _, current_implementation = ablation_eval_v7._source_snapshots(
            _implementation_runner_path(
                implementation,
                label="ablation implementation",
            )
        )
        if dict(implementation) != current_implementation:
            raise PostfreezeBlindV7Error(
                "ablation implementation sources changed"
            )
        stripped = [_strip_ablation_v7_row(row) for row in samples]
        recomputed_v6 = ablation_eval_v6._recompute_sample_rows(
            recorded_rows=stripped,
            dataset_rows=split.rows,
        )
        recomputed_samples = json.loads(canonical_json(recomputed_v6))
        for row in recomputed_samples:
            row["schema"] = ablation_eval_v7.SAMPLE_SCHEMA
            row["ablation_version"] = ablation_eval_v7.VERSION
            row["lifecycle_binding_sha256"] = lifecycle.binding[
                "binding_sha256"
            ]
            row["v6_math_implementation_sha256"] = implementation[
                "ablation_math_v6"
            ]["sha256"]
            row["boundaries"] = {
                **row["boundaries"],
                "fixture_not_model_evidence": False,
                "reserved_data_accessed": False,
            }
        recomputed_samples.sort(
            key=lambda row: (
                ablation_eval_v6.SUBJECTS.index(str(row["subject"])),
                ablation_eval_v6.ALL_VARIANTS.index(str(row["variant"])),
                str(row["example_id"]),
            )
        )
        if list(samples) != recomputed_samples:
            raise PostfreezeBlindV7Error(
                "ablation samples differ from independent recompilation"
            )
        recomputed_reports, invariants_passed = ablation_eval_v7._v7_reports(
            recomputed_samples,
            backend_mode="hf_model",
            lifecycle=lifecycle,
            implementation=implementation,
        )
        if {
            name: dict(report) for name, report in reports.items()
        } != recomputed_reports:
            raise PostfreezeBlindV7Error(
                "ablation reports differ from recomputed metrics"
            )
        if (
            invariants_passed is not True
            or receipt.get("invariants_passed") is not True
        ):
            raise PostfreezeBlindV7Error(
                "ablation invariants did not independently pass"
            )
        cases, _ = ablation_eval_v6._build_cases(split.rows)
        requests = ablation_eval_v6._generation_requests(cases)
        request_digest = lifecycle_bindings_v7.canonical_sha256(
            [
                {
                    "case_id": request.example_id,
                    "messages": list(request.messages),
                }
                for request in requests
            ]
        )
        execution = _require_mapping(
            receipt.get("execution"),
            label="ablation execution",
        )
        if execution.get("request_digest_sha256") != request_digest:
            raise PostfreezeBlindV7Error(
                "ablation request digest differs from the frozen split"
            )
        model = _require_mapping(
            receipt.get("model"),
            label="ablation model",
        )
        backend_bindings = _require_mapping(
            receipt.get("backend_bindings"),
            label="ablation backend bindings",
        )
        if set(backend_bindings) != set(ablation_eval_v6.SUBJECTS):
            raise PostfreezeBlindV7Error(
                "ablation backend subject set is incomplete"
            )
        for subject in ablation_eval_v6.SUBJECTS:
            backend = _require_mapping(
                backend_bindings.get(subject),
                label=f"ablation backend {subject}",
            )
            if (
                backend.get("mode") != "hf_model"
                or backend.get("device") not in {"cpu", "cuda"}
            ):
                raise PostfreezeBlindV7Error(
                    f"ablation backend is not model evidence: {subject}"
                )
            _verify_backend_full_model_tree(
                backend,
                model=model,
                adapter_required=subject == "adapter",
                label=f"ablation backend {subject}",
            )
        artifacts = _require_mapping(
            receipt.get("artifacts"),
            label="ablation artifacts",
        )
        reproducibility = {
            "version": ablation_eval_v7.VERSION,
            "lifecycle_binding_sha256": lifecycle.binding[
                "binding_sha256"
            ],
            "validation_split_sha256": split.file.sha256,
            "validation_rows": ablation_eval_v7.EXPECTED_VALIDATION_ROWS,
            "sample_rows": ablation_eval_v7.EXPECTED_SAMPLE_ROWS,
            "backend_mode": "hf_model",
            "seed": ablation_eval_v7.FIXED_SEED,
            "request_digest_sha256": request_digest,
            "same_requests_for_base_and_adapter": True,
            "implementation": dict(implementation),
            "backend_bindings": {
                subject: dict(
                    _require_mapping(
                        backend_bindings[subject],
                        label=f"ablation backend {subject}",
                    )
                )
                for subject in ablation_eval_v6.SUBJECTS
            },
            "artifacts": dict(artifacts),
        }
        if receipt.get(
            "reproducibility_payload_sha256"
        ) != lifecycle_bindings_v7.canonical_sha256(reproducibility):
            raise PostfreezeBlindV7Error(
                "ablation reproducibility payload mismatch"
            )
    except PostfreezeBlindV7Error:
        raise
    except (
        ablation_eval_v6.AblationEvalV6Error,
        ablation_eval_v7.AblationEvalV7Error,
        lifecycle_bindings_v7.LifecycleBindingV7Error,
        pointer_hf_eval_v6.PointerHFEvalV6Error,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise PostfreezeBlindV7Error(
            "ablation deterministic verification failed"
        ) from exc
    return {
        "dataset_split_sha256": split.file.sha256,
        "dataset_id_order": order,
        "samples_recompiled": True,
        "reports_recomputed": True,
        "invariants_recomputed": True,
        "backend_model_tree_reverified": True,
    }


def _verify_calibration_gate(
    directory: Path,
    *,
    lifecycle: lifecycle_bindings_v7.LifecycleSnapshotV7,
) -> dict[str, Any]:
    binding = lifecycle.binding
    anchor, snapshots = _capture_directory_artifacts(
        directory,
        expected_names=set(calibration_eval_v7.EXPECTED_ARTIFACT_NAMES),
        label="calibration gate",
    )
    receipt_snapshot = snapshots[calibration_eval_v7.RECEIPT_FILENAME]
    receipt = _parse_json_bytes(
        receipt_snapshot.payload,
        label="calibration receipt",
    )
    expected_fields = {
        "schema",
        "version",
        "status",
        "backend",
        "selection",
        "authority",
        "dataset",
        "model",
        "implementation",
        "artifacts",
        "quality_gate_passed",
        "conformal_threshold",
        "selection_locked",
        "checkpoint_reselection_performed",
        "authorization",
        "access_boundary",
        "claim_boundary",
        "canonical_digest_sha256",
    }
    _require_exact_keys(
        receipt,
        expected_fields,
        label="calibration receipt",
    )
    body = dict(receipt)
    digest = _require_sha(
        body.pop("canonical_digest_sha256"),
        label="calibration canonical digest",
    )
    if canonical_sha256(body) != digest:
        raise PostfreezeBlindV7Error(
            "calibration canonical digest mismatch"
        )
    backend = _require_mapping(
        receipt["backend"],
        label="calibration backend",
    )
    dataset = _require_mapping(
        receipt["dataset"],
        label="calibration dataset",
    )
    model = _require_mapping(
        receipt["model"],
        label="calibration model",
    )
    authorization = _require_mapping(
        receipt["authorization"],
        label="calibration authorization",
    )
    access = _require_mapping(
        receipt["access_boundary"],
        label="calibration access boundary",
    )
    expected_authority = {
        "lifecycle_binding_sha256": binding["binding_sha256"],
        "nonblind_manifest_sha256": binding["nonblind_dataset"][
            "manifest"
        ]["sha256"],
        "preblind_commitment_sha256": binding["nonblind_dataset"][
            "preblind_commitment"
        ]["commitment_sha256"],
        "contract_set_sha256": binding["contracts"][
            "contract_set_sha256"
        ],
    }
    if (
        receipt["schema"] != calibration_eval_v7.RECEIPT_SCHEMA
        or receipt["version"] != calibration_eval_v7.VERSION
        or receipt["status"]
        != "PASS_NONBLIND_V7_CALIBRATION_MODEL_BOUND"
        or backend.get("mode") != "hf_model"
        or receipt["selection"] != binding["selection"]
        or receipt["authority"] != expected_authority
        or dataset.get("split") != "calibration"
        or dataset.get("complete_split") is not True
        or dataset.get("rows") != EXPECTED_ROWS
        or dataset.get("max_samples") is not None
        or model.get("model_bound") is not True
        or model.get("fixture_not_model_evidence") is not False
        or receipt["quality_gate_passed"] is not True
        or receipt["selection_locked"] is not True
        or receipt["checkpoint_reselection_performed"] is not False
        or set(authorization.values()) != {False}
        or access.get("reserved_content_accessed") is not False
        or access.get("x5_accessed") is not False
        or access.get("network_accessed") is not False
    ):
        raise PostfreezeBlindV7Error(
            "calibration gate is not complete model-bound nonblind evidence"
        )
    for key, expected in binding["model"].items():
        if model.get(key) != expected:
            raise PostfreezeBlindV7Error(
                f"calibration model binding mismatch: {key}"
            )
    _validate_artifact_descriptors(
        receipt,
        snapshots,
        receipt_filename=calibration_eval_v7.RECEIPT_FILENAME,
        label="calibration gate",
    )
    samples = _parse_jsonl_bytes(
        snapshots[calibration_eval_v7.SAMPLE_FILENAME].payload,
        label="calibration samples",
    )
    sample_ids = [row.get("example_id") for row in samples]
    if (
        len(samples) != EXPECTED_ROWS
        or len(set(sample_ids)) != EXPECTED_ROWS
        or any(not isinstance(item, str) or not item for item in sample_ids)
    ):
        raise PostfreezeBlindV7Error(
            "calibration sample membership is incomplete"
        )
    sample_descriptor = _require_mapping(
        receipt["artifacts"][calibration_eval_v7.SAMPLE_FILENAME],
        label="calibration sample descriptor",
    )
    if sample_descriptor.get("records") != EXPECTED_ROWS:
        raise PostfreezeBlindV7Error(
            "calibration sample descriptor count mismatch"
        )
    summary = _parse_json_bytes(
        snapshots[calibration_eval_v7.SUMMARY_FILENAME].payload,
        label="calibration summary",
    )
    if (
        summary.get("schema") != calibration_eval_v7.SUMMARY_SCHEMA
        or summary.get("status") != receipt["status"]
        or summary.get("rows") != EXPECTED_ROWS
        or summary.get("complete_split") is not True
        or summary.get("model_bound") is not True
        or summary.get("quality_gate_passed") is not True
        or summary.get("checkpoint_reselection_performed") is not False
    ):
        raise PostfreezeBlindV7Error(
            "calibration summary contract mismatch"
        )
    independent = _independently_verify_calibration_gate(
        receipt=receipt,
        samples=samples,
        summary=summary,
        lifecycle=lifecycle,
    )
    for snapshot in snapshots.values():
        _verify_snapshot(
            snapshot,
            label=f"calibration final {snapshot.path.name}",
        )
    _verify_directory(anchor, label="calibration directory final")
    return {
        "directory": str(anchor.path),
        "receipt": receipt_snapshot.receipt(),
        "status": receipt["status"],
        "model_bound": True,
        "rows": EXPECTED_ROWS,
        "quality_gate_passed": True,
        "conformal_threshold": receipt["conformal_threshold"],
        "dataset_split_sha256": independent["dataset_split_sha256"],
        "dataset_id_order": independent["dataset_id_order"],
        "deterministic_reverification": independent,
    }


def _verify_ablation_gate(
    directory: Path,
    *,
    lifecycle: lifecycle_bindings_v7.LifecycleSnapshotV7,
) -> dict[str, Any]:
    binding = lifecycle.binding
    anchor, snapshots = _capture_directory_artifacts(
        directory,
        expected_names=set(ablation_eval_v7.EXPECTED_ARTIFACT_NAMES),
        label="ablation gate",
    )
    receipt_snapshot = snapshots[ablation_eval_v7.RECEIPT_FILENAME]
    receipt = _parse_json_bytes(
        receipt_snapshot.payload,
        label="ablation receipt",
    )
    body = dict(receipt)
    digest = _require_sha(
        body.pop("canonical_digest_sha256", None),
        label="ablation canonical digest",
    )
    if canonical_sha256(body) != digest:
        raise PostfreezeBlindV7Error(
            "ablation canonical digest mismatch"
        )
    dataset = _require_mapping(
        receipt.get("dataset"),
        label="ablation dataset",
    )
    execution = _require_mapping(
        receipt.get("execution"),
        label="ablation execution",
    )
    model = _require_mapping(
        receipt.get("model"),
        label="ablation model",
    )
    authorization = _require_mapping(
        receipt.get("authorization"),
        label="ablation authorization",
    )
    access = _require_mapping(
        receipt.get("access_boundary"),
        label="ablation access boundary",
    )
    expected_authority = {
        "lifecycle_binding_sha256": binding["binding_sha256"],
        "nonblind_manifest_sha256": binding["nonblind_dataset"][
            "manifest"
        ]["sha256"],
        "preblind_commitment_sha256": binding["nonblind_dataset"][
            "preblind_commitment"
        ]["commitment_sha256"],
        "contract_set_sha256": binding["contracts"][
            "contract_set_sha256"
        ],
    }
    if (
        receipt.get("schema") != ablation_eval_v7.RECEIPT_SCHEMA
        or receipt.get("version") != ablation_eval_v7.VERSION
        or receipt.get("status")
        != "PASS_NONBLIND_V7_ABLATIONS_COMPLETE_NO_SELECTION"
        or receipt.get("selection") != binding["selection"]
        or receipt.get("authority") != expected_authority
        or dataset.get("split") != "validation"
        or dataset.get("complete_split") is not True
        or dataset.get("rows") != EXPECTED_ROWS
        or dataset.get("max_samples") is not None
        or execution.get("selection_policy_called") is not False
        or execution.get("automatic_model_selection") is not False
        or execution.get("checkpoint_reselection_performed") is not False
        or model.get("model_bound") is not True
        or model.get("fixture_not_model_evidence") is not False
        or receipt.get("invariants_passed") is not True
        or set(authorization.values()) != {False}
        or access.get("reserved_content_accessed") is not False
        or access.get("x5_accessed") is not False
        or access.get("network_accessed") is not False
    ):
        raise PostfreezeBlindV7Error(
            "ablation gate is not complete no-selection model evidence"
        )
    for key, expected in binding["model"].items():
        if model.get(key) != expected:
            raise PostfreezeBlindV7Error(
                f"ablation model binding mismatch: {key}"
            )
    _validate_artifact_descriptors(
        receipt,
        snapshots,
        receipt_filename=ablation_eval_v7.RECEIPT_FILENAME,
        label="ablation gate",
    )
    samples = _parse_jsonl_bytes(
        snapshots[ablation_eval_v7.SAMPLE_FILENAME].payload,
        label="ablation samples",
    )
    descriptor = _require_mapping(
        receipt["artifacts"][ablation_eval_v7.SAMPLE_FILENAME],
        label="ablation sample descriptor",
    )
    if (
        len(samples) != ablation_eval_v7.EXPECTED_SAMPLE_ROWS
        or descriptor.get("records")
        != ablation_eval_v7.EXPECTED_SAMPLE_ROWS
    ):
        raise PostfreezeBlindV7Error(
            "ablation sample matrix is incomplete"
        )
    reports = {
        name: _parse_json_bytes(
            snapshots[name].payload,
            label=f"ablation report {name}",
        )
        for name in sorted(ablation_eval_v7.REPORT_FILENAMES)
    }
    independent = _independently_verify_ablation_gate(
        receipt=receipt,
        samples=samples,
        reports=reports,
        lifecycle=lifecycle,
    )
    for snapshot in snapshots.values():
        _verify_snapshot(
            snapshot,
            label=f"ablation final {snapshot.path.name}",
        )
    _verify_directory(anchor, label="ablation directory final")
    return {
        "directory": str(anchor.path),
        "receipt": receipt_snapshot.receipt(),
        "status": receipt["status"],
        "model_bound": True,
        "validation_rows": EXPECTED_ROWS,
        "sample_rows": ablation_eval_v7.EXPECTED_SAMPLE_ROWS,
        "selection_performed": False,
        "dataset_split_sha256": independent["dataset_split_sha256"],
        "dataset_id_order": independent["dataset_id_order"],
        "deterministic_reverification": independent,
    }


def _verify_postselection_gates(
    *,
    calibration_dir: Path,
    ablation_dir: Path,
    lifecycle: lifecycle_bindings_v7.LifecycleSnapshotV7,
) -> dict[str, Any]:
    return {
        "calibration": _verify_calibration_gate(
            calibration_dir,
            lifecycle=lifecycle,
        ),
        "ablation": _verify_ablation_gate(
            ablation_dir,
            lifecycle=lifecycle,
        ),
    }


def _capture_code_bindings(execute_runner_path: Path) -> dict[str, Any]:
    paths = {
        "protocol": Path(__file__),
        "execute_runner": Path(execute_runner_path),
        "pointer_evaluator": Path(pointer_hf_eval_v6.__file__),
        "pointer_compiler": Path(evidence_pointer_v6.__file__),
        "nonblind_builder": Path(nonblind_sft_v7.__file__),
        "evidence_core": Path(evidence_sft_v6.__file__),
    }
    snapshots = {
        role: _capture_file(
            path,
            label=f"implementation {role}",
            maximum_bytes=4 * 1024 * 1024,
        )
        for role, path in paths.items()
    }
    return {
        "snapshots": snapshots,
        "receipt": {
            role: snapshot.receipt()
            for role, snapshot in snapshots.items()
        },
    }


def _validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise PostfreezeBlindV7Error("run_id is invalid")
    return run_id


def _prepare_postfreeze_blind_v7_at_registry(
    *,
    dataset_dir: Path,
    licensed_chunks_path: Path,
    rag_manifest_path: Path,
    semantic_inventory_path: Path,
    selection_freeze_path: Path,
    evaluation_index_path: Path,
    training_receipt_path: Path,
    base_model_dir: Path,
    adapter_dir: Path,
    preblind_commitment_path: Path,
    contract_dir: Path,
    calibration_dir: Path,
    ablation_dir: Path,
    registry_root: Path,
    execute_runner_path: Path,
    run_id: str,
) -> dict[str, Any]:
    """Prepare the sole authorization without assigning any dataset split."""

    _validate_run_id(run_id)
    registry = _directory_anchor(
        registry_root,
        label="global one-shot registry",
    )
    dataset_anchor = _directory_anchor(
        dataset_dir,
        label="nonblind dataset",
    )
    try:
        registry.path.relative_to(dataset_anchor.path)
    except ValueError:
        pass
    else:
        raise PostfreezeBlindV7Error(
            "global registry must be outside the dataset directory"
        )

    source_snapshots = _capture_source_set(
        licensed_chunks_path=licensed_chunks_path,
        rag_manifest_path=rag_manifest_path,
        semantic_inventory_path=semantic_inventory_path,
    )
    source_binding = _snapshot_set_receipt(source_snapshots)
    committed = _validate_commitment_and_manifest(
        dataset_dir=dataset_dir,
        preblind_commitment_path=preblind_commitment_path,
        snapshots=source_snapshots,
    )
    authority = _verify_public_authority(
        selection_freeze_path=selection_freeze_path,
        evaluation_index_path=evaluation_index_path,
        training_receipt_path=training_receipt_path,
        dataset_dir=dataset_dir,
        base_model_dir=base_model_dir,
        preblind_commitment_path=preblind_commitment_path,
        contract_dir=contract_dir,
    )
    lifecycle_snapshot = authority["lifecycle"]
    lifecycle = lifecycle_snapshot.binding
    if (
        lifecycle["nonblind_dataset"]["preblind_commitment"][
            "commitment_sha256"
        ]
        != committed["commitment_sha256"]
        or lifecycle["nonblind_dataset"]["manifest"]["sha256"]
        != committed["manifest_snapshot"].sha256
    ):
        raise PostfreezeBlindV7Error(
            "public lifecycle differs from committed dataset"
        )
    model = lifecycle["model"]
    try:
        adapter = Path(adapter_dir).resolve(strict=True)
    except OSError as exc:
        raise PostfreezeBlindV7Error(
            "selected adapter directory is unavailable"
        ) from exc
    if adapter != Path(str(model["checkpoint_path"])).resolve(strict=True):
        raise PostfreezeBlindV7Error(
            "adapter path differs from frozen selection"
        )
    authorization_model = {
        **model,
        "adapter_runtime_path": str(adapter),
        "frozen_model_only": True,
    }
    authorization_model["full_model_tree_binding"] = (
        _capture_full_model_tree_binding(
            authorization_model,
            adapter_required=True,
            label="authorization",
        )
    )
    postselection = _verify_postselection_gates(
        calibration_dir=calibration_dir,
        ablation_dir=ablation_dir,
        lifecycle=lifecycle_snapshot,
    )
    code = _capture_code_bindings(execute_runner_path)

    for snapshot in source_snapshots.values():
        _verify_snapshot(
            snapshot,
            label=f"source final {snapshot.path.name}",
        )
    for snapshot in code["snapshots"].values():
        _verify_snapshot(
            snapshot,
            label=f"implementation final {snapshot.path.name}",
        )
    _verify_directory(registry, label="global registry final")

    commitment_sha = committed["commitment_sha256"]
    paths = _registry_paths(
        registry,
        commitment_sha256=commitment_sha,
    )
    if any(os.path.lexists(path) for path in paths.values()):
        raise PostfreezeBlindV7Error(
            "this preblind commitment already has one-shot state"
        )
    universe_id = _universe_id(commitment_sha)
    policy_sha = canonical_sha256(RELEASE_POLICY)
    binding = {
        "universe_id": universe_id,
        "commitment_sha256": commitment_sha,
        "source_snapshot_set_sha256": source_binding[
            "content_set_sha256"
        ],
        "selection_binding_digest_sha256": lifecycle["selection"][
            "selection_binding_digest_sha256"
        ],
        "selected_checkpoint_tree_sha256": model[
            "checkpoint_tree_sha256"
        ],
        "selected_adapter_tree_sha256": model["adapter_tree_sha256"],
        "contract_set_sha256": lifecycle["contracts"][
            "contract_set_sha256"
        ],
        "calibration_receipt_sha256": postselection["calibration"][
            "receipt"
        ]["sha256"],
        "ablation_receipt_sha256": postselection["ablation"][
            "receipt"
        ]["sha256"],
        "release_policy_sha256": policy_sha,
        "run_id": run_id,
    }
    authorization_id = (
        "icmat-v7-postfreeze-" + canonical_sha256(binding)[:32]
    )
    authorization_body = {
        "schema": AUTHORIZATION_SCHEMA,
        "version": PROTOCOL_VERSION,
        "status": AUTHORIZATION_STATUS,
        "authorization_id": authorization_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "universe": {
            "universe_id": universe_id,
            "claim_key_basis": "preblind_commitment_sha256_only",
            "preblind_commitment_sha256": commitment_sha,
            "expected_rows": EXPECTED_ROWS,
            "expected_families": EXPECTED_FAMILIES,
            "examples_per_family": EXAMPLES_PER_FAMILY,
            "expected_domains": list(evidence_sft_v6.DOMAINS),
        },
        "commitment": {
            "file": committed["commitment_snapshot"].receipt(),
            "schema": committed["commitment"]["schema"],
            "builder_version": committed["commitment"][
                "builder_version"
            ],
            "core_builder_version": committed["commitment"][
                "core_builder_version"
            ],
            "split_algorithm_version": committed["commitment"][
                "split_algorithm_version"
            ],
            "seed": committed["seed"],
            "seed_sha256": committed["commitment"]["seed_sha256"],
            "commitment_sha256": commitment_sha,
        },
        "sources": source_binding,
        "nonblind_dataset": {
            "directory": str(committed["dataset"].path),
            "manifest": committed["manifest_snapshot"].receipt(),
            "splits": {
                split: dict(
                    committed["manifest"]["splits"][split]
                )
                for split in evidence_sft_v6.NONBLIND_SPLITS
            },
        },
        "upstream": {
            "selection_verification": authority["selection"],
            "contract_verification": authority["contracts"],
            "lifecycle_binding": lifecycle,
            "postselection": postselection,
        },
        "model": {
            **authorization_model,
        },
        "implementation": code["receipt"],
        "execution": {
            "run_id": run_id,
            "backend": "hf_model",
            "device": "cuda",
            "seed": FIXED_SEED,
            "max_samples": None,
            "resume_allowed": False,
            "retry_after_claim_allowed": False,
            "single_hf_call_required": True,
            "local_files_only": True,
            "network_allowed": False,
            "x5_access_allowed": False,
        },
        "release_policy": json.loads(canonical_json(RELEASE_POLICY)),
        "registry": {
            **registry.receipt(),
            "authorization_path": str(paths["authorization"]),
            "claim_path": str(paths["claim"]),
            "terminal_path": str(paths["terminal"]),
            "evidence_path": str(paths["evidence"]),
        },
        "authorization": {
            "gguf_offline_candidate_may_be_qualified": True,
            "activation_authorized": False,
            "deployment_authorized": False,
            "production_integration_authorized": False,
            "model_selection_authorized": False,
            "checkpoint_ranking_authorized": False,
            "threshold_tuning_authorized": False,
            "calibration_authorized": False,
        },
        "claim_boundary": {
            "claim_must_precede_split_assignment": True,
            "claim_must_precede_example_build": True,
            "claim_must_precede_test_member_derivation": True,
            "claim_is_non_reusable_after_success_failure_or_crash": True,
            "claim_key_uses_model_or_run_parameters": False,
        },
        "security_boundary": {
            "protocol_execution_constraint": True,
            "cryptographic_secrecy": False,
            "honest_local_execution_environment_required": True,
            "administrator_forgery_resistant": False,
            "tpm_or_external_signature_verified": False,
            "administrator_can_derive_outside_protocol": True,
            "statement": (
                "This is an auditable protocol constraint in an honest local "
                "execution environment, not cryptographic confidentiality, "
                "administrator-forgery resistance, TPM attestation, or an "
                "externally signed ledger."
            ),
        },
    }
    authorization = {
        **authorization_body,
        "canonical_digest_sha256": canonical_sha256(
            authorization_body
        ),
    }
    payload = _json_bytes(authorization)
    try:
        publication = _exclusive_create(paths["authorization"], payload)
    except FileExistsError as exc:
        raise PostfreezeBlindV7Error(
            "this preblind commitment already has an authorization"
        ) from exc
    _verify_directory(registry, label="global registry after authorization")
    return {
        "status": "POSTFREEZE_V7_AUTHORIZATION_PREPARED_NOT_CLAIMED",
        "authorization_id": authorization_id,
        "authorization": publication,
        "universe_id": universe_id,
        "claim_created": False,
        "split_assignment_called": False,
        "test_members_derived": False,
        "retry_policy": "NON_REUSABLE_AFTER_CLAIM",
        "cryptographic_secrecy": False,
        "honest_local_execution_environment_required": True,
        "administrator_forgery_resistant": False,
        "tpm_or_external_signature_verified": False,
    }


def prepare_postfreeze_blind_v7(
    *,
    dataset_dir: Path,
    licensed_chunks_path: Path,
    rag_manifest_path: Path,
    semantic_inventory_path: Path,
    selection_freeze_path: Path,
    evaluation_index_path: Path,
    training_receipt_path: Path,
    base_model_dir: Path,
    adapter_dir: Path,
    preblind_commitment_path: Path,
    contract_dir: Path,
    calibration_dir: Path,
    ablation_dir: Path,
    execute_runner_path: Path,
    run_id: str,
) -> dict[str, Any]:
    """Prepare against the one fixed workspace registry."""

    return _prepare_postfreeze_blind_v7_at_registry(
        dataset_dir=dataset_dir,
        licensed_chunks_path=licensed_chunks_path,
        rag_manifest_path=rag_manifest_path,
        semantic_inventory_path=semantic_inventory_path,
        selection_freeze_path=selection_freeze_path,
        evaluation_index_path=evaluation_index_path,
        training_receipt_path=training_receipt_path,
        base_model_dir=base_model_dir,
        adapter_dir=adapter_dir,
        preblind_commitment_path=preblind_commitment_path,
        contract_dir=contract_dir,
        calibration_dir=calibration_dir,
        ablation_dir=ablation_dir,
        registry_root=_production_registry_root(create=True),
        execute_runner_path=execute_runner_path,
        run_id=run_id,
    )


def _load_authorization(
    authorization_path: Path,
) -> tuple[StableSnapshot, dict[str, Any]]:
    snapshot = _capture_file(
        authorization_path,
        label="postfreeze authorization",
    )
    receipt = _parse_json_bytes(
        snapshot.payload,
        label="postfreeze authorization",
    )
    body = dict(receipt)
    digest = _require_sha(
        body.pop("canonical_digest_sha256", None),
        label="authorization canonical digest",
    )
    if canonical_sha256(body) != digest:
        raise PostfreezeBlindV7Error(
            "authorization canonical digest mismatch"
        )
    if (
        receipt.get("schema") != AUTHORIZATION_SCHEMA
        or receipt.get("version") != PROTOCOL_VERSION
        or receipt.get("status") != AUTHORIZATION_STATUS
    ):
        raise PostfreezeBlindV7Error("authorization identity mismatch")
    security = _require_mapping(
        receipt.get("security_boundary"),
        label="authorization security boundary",
    )
    if (
        security.get("protocol_execution_constraint") is not True
        or security.get("cryptographic_secrecy") is not False
        or security.get("honest_local_execution_environment_required")
        is not True
        or security.get("administrator_forgery_resistant") is not False
        or security.get("tpm_or_external_signature_verified") is not False
        or security.get("administrator_can_derive_outside_protocol")
        is not True
    ):
        raise PostfreezeBlindV7Error(
            "authorization omits the non-cryptographic boundary"
        )
    return snapshot, receipt


def _paths_from_authorization(
    authorization: Mapping[str, Any],
    registry_root: Path,
) -> tuple[DirectoryAnchor, dict[str, Path]]:
    registry = _directory_anchor(
        registry_root,
        label="global one-shot registry",
    )
    recorded = _require_mapping(
        authorization.get("registry"),
        label="authorization registry",
    )
    if recorded.get("path") != str(registry.path):
        raise PostfreezeBlindV7Error(
            "caller registry differs from authorization"
        )
    recorded_identity = _require_mapping(
        recorded.get("stable_identity"),
        label="authorization registry identity",
    )
    expected_identity = {
        "st_dev": registry.identity[0],
        "st_ino": registry.identity[1],
        "st_size": registry.identity[2],
        "st_mtime_ns": registry.identity[3],
        "st_ctime_ns": registry.identity[4],
    }
    # Directory size/time may change when state files are created. Device and
    # inode are the durable anchor; the full initial identity remains evidence.
    if (
        recorded_identity.get("st_dev") != expected_identity["st_dev"]
        or recorded_identity.get("st_ino") != expected_identity["st_ino"]
    ):
        raise PostfreezeBlindV7Error(
            "global registry device/inode changed"
        )
    universe = _require_mapping(
        authorization.get("universe"),
        label="authorization universe",
    )
    commitment_sha = _require_sha(
        universe.get("preblind_commitment_sha256"),
        label="authorization commitment SHA",
    )
    paths = _registry_paths(
        registry,
        commitment_sha256=commitment_sha,
    )
    recorded_paths = {
        "authorization": recorded.get("authorization_path"),
        "claim": recorded.get("claim_path"),
        "terminal": recorded.get("terminal_path"),
        "evidence": recorded.get("evidence_path"),
    }
    if recorded_paths != {
        name: str(path) for name, path in paths.items()
    }:
        raise PostfreezeBlindV7Error(
            "authorization registry paths are not commitment-derived"
        )
    if (
        universe.get("universe_id") != _universe_id(commitment_sha)
        or universe.get("claim_key_basis")
        != "preblind_commitment_sha256_only"
    ):
        raise PostfreezeBlindV7Error(
            "global claim key is not commitment-only"
        )
    return registry, paths


def _source_paths_from_authorization(
    authorization: Mapping[str, Any],
) -> dict[str, Path]:
    sources = _require_mapping(
        authorization.get("sources"),
        label="authorization sources",
    )
    files = _require_mapping(
        sources.get("files"),
        label="authorization source files",
    )
    if tuple(sources.get("roles", [])) != _SOURCE_ROLES:
        raise PostfreezeBlindV7Error(
            "authorization source role set mismatch"
        )
    return {
        role: Path(
            str(
                _require_mapping(
                    files.get(role),
                    label=f"authorization source {role}",
                ).get("path")
            )
        )
        for role in _SOURCE_ROLES
    }


def _recapture_authorized_sources(
    authorization: Mapping[str, Any],
) -> dict[str, StableSnapshot]:
    paths = _source_paths_from_authorization(authorization)
    snapshots = {
        role: _capture_file(
            paths[role],
            label=f"authorized source {role}",
            maximum_bytes=MAX_SOURCE_BYTES,
        )
        for role in _SOURCE_ROLES
    }
    observed = _snapshot_set_receipt(snapshots)
    if observed != authorization.get("sources"):
        raise PostfreezeBlindV7Error(
            "authorized source snapshot set changed"
        )
    return snapshots


def _reverify_authorization_inputs(
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    sources = _recapture_authorized_sources(authorization)
    dataset_record = _require_mapping(
        authorization.get("nonblind_dataset"),
        label="authorization nonblind dataset",
    )
    commitment_record = _require_mapping(
        authorization.get("commitment"),
        label="authorization commitment",
    )
    committed = _validate_commitment_and_manifest(
        dataset_dir=Path(str(dataset_record.get("directory"))),
        preblind_commitment_path=Path(
            str(
                _require_mapping(
                    commitment_record.get("file"),
                    label="authorization commitment file",
                ).get("path")
            )
        ),
        snapshots=sources,
    )
    upstream = _require_mapping(
        authorization.get("upstream"),
        label="authorization upstream",
    )
    lifecycle_record = _require_mapping(
        upstream.get("lifecycle_binding"),
        label="authorization lifecycle",
    )
    selection_record = _require_mapping(
        lifecycle_record.get("selection"),
        label="authorization lifecycle selection",
    )
    selection_file = _require_mapping(
        selection_record.get("receipt"),
        label="authorization selection file",
    )
    nonblind_record = _require_mapping(
        lifecycle_record.get("nonblind_dataset"),
        label="authorization lifecycle dataset",
    )
    lifecycle_manifest = _require_mapping(
        nonblind_record.get("manifest"),
        label="authorization lifecycle manifest",
    )
    model = _require_mapping(
        authorization.get("model"),
        label="authorization model",
    )
    contracts_record = _require_mapping(
        lifecycle_record.get("contracts"),
        label="authorization lifecycle contracts",
    )
    lifecycle_files = _require_mapping(
        contracts_record.get("files"),
        label="authorization contract files",
    )
    build_receipt = _require_mapping(
        lifecycle_files.get(contracts_v7.BUILD_RECEIPT_FILENAME),
        label="authorization contract build receipt",
    )
    contract_dir = Path(str(contracts_record.get("directory")))
    if (
        Path(str(build_receipt.get("path"))).parent.resolve(strict=True)
        != contract_dir.resolve(strict=True)
    ):
        raise PostfreezeBlindV7Error(
            "authorization contract directory mismatch"
        )

    selection_verification = _require_mapping(
        upstream.get("selection_verification"),
        label="authorization selection verification",
    )
    freeze_path = Path(str(selection_file.get("path")))
    # The evaluation and training paths are recorded in the selection receipt
    # nested inside the lifecycle authority files. Use the authorization source
    # records from the public verifier for exact paths.
    selection_receipt_snapshot = _capture_file(
        freeze_path,
        label="authorized selection freeze",
    )
    selection_receipt = _parse_json_bytes(
        selection_receipt_snapshot.payload,
        label="authorized selection freeze",
    )
    evaluation_path = Path(
        str(
            _require_mapping(
                selection_receipt.get("evaluation_receipt"),
                label="selection evaluation receipt",
            ).get("path")
        )
    )
    training_path = Path(
        str(
            _require_mapping(
                selection_receipt.get("training_receipt"),
                label="selection training receipt",
            ).get("path")
        )
    )
    authority = _verify_public_authority(
        selection_freeze_path=freeze_path,
        evaluation_index_path=evaluation_path,
        training_receipt_path=training_path,
        dataset_dir=Path(str(dataset_record.get("directory"))),
        base_model_dir=Path(str(model.get("base_model_path"))),
        preblind_commitment_path=committed[
            "commitment_snapshot"
        ].path,
        contract_dir=contract_dir,
    )
    if (
        authority["selection"] != dict(selection_verification)
        or authority["contracts"]
        != dict(
            _require_mapping(
                upstream.get("contract_verification"),
                label="authorization contract verification",
            )
        )
        or authority["lifecycle"].binding != dict(lifecycle_record)
    ):
        raise PostfreezeBlindV7Error(
            "public authority changed after authorization"
        )
    lifecycle_model = _require_mapping(
        authority["lifecycle"].binding.get("model"),
        label="reverified lifecycle model",
    )
    expected_model = {
        **lifecycle_model,
        "adapter_runtime_path": str(
            Path(str(lifecycle_model.get("checkpoint_path"))).resolve(
                strict=True
            )
        ),
        "frozen_model_only": True,
    }
    expected_model["full_model_tree_binding"] = (
        _capture_full_model_tree_binding(
            expected_model,
            adapter_required=True,
            label="reverified authorization",
        )
    )
    if dict(model) != expected_model:
        raise PostfreezeBlindV7Error(
            "authorization model differs from the reverified frozen model"
        )
    if (
        lifecycle_manifest.get("sha256")
        != committed["manifest_snapshot"].sha256
        or committed["commitment_sha256"]
        != commitment_record.get("commitment_sha256")
    ):
        raise PostfreezeBlindV7Error(
            "authorized manifest or commitment changed"
        )
    postselection_record = _require_mapping(
        upstream.get("postselection"),
        label="authorization postselection",
    )
    calibration_record = _require_mapping(
        postselection_record.get("calibration"),
        label="authorization calibration gate",
    )
    ablation_record = _require_mapping(
        postselection_record.get("ablation"),
        label="authorization ablation gate",
    )
    postselection = _verify_postselection_gates(
        calibration_dir=Path(str(calibration_record.get("directory"))),
        ablation_dir=Path(str(ablation_record.get("directory"))),
        lifecycle=authority["lifecycle"],
    )
    if postselection != dict(postselection_record):
        raise PostfreezeBlindV7Error(
            "postselection gates changed after authorization"
        )
    implementation = _require_mapping(
        authorization.get("implementation"),
        label="authorization implementation",
    )
    execute_runner = _require_mapping(
        implementation.get("execute_runner"),
        label="authorization execute runner",
    )
    code = _capture_code_bindings(
        Path(str(execute_runner.get("path")))
    )
    if code["receipt"] != dict(implementation):
        raise PostfreezeBlindV7Error(
            "authorized implementation changed"
        )
    if canonical_sha256(RELEASE_POLICY) != canonical_sha256(
        authorization.get("release_policy")
    ):
        raise PostfreezeBlindV7Error(
            "release policy changed after authorization"
        )
    return {
        "sources": sources,
        "committed": committed,
        "authority": authority,
        "postselection": postselection,
        "code": code,
    }


def _require_cuda_ready() -> None:
    try:
        import torch
    except ImportError as exc:
        raise PostfreezeBlindV7Error(
            "CUDA HF execution requires local torch"
        ) from exc
    if not torch.cuda.is_available():
        raise PostfreezeBlindV7Error(
            "CUDA is unavailable before the one-shot claim"
        )


def _claim_once(
    *,
    authorization_snapshot: StableSnapshot,
    authorization: Mapping[str, Any],
    registry: DirectoryAnchor,
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    if os.path.lexists(paths["terminal"]):
        raise PostfreezeBlindV7Error(
            "this commitment already has a terminal record"
        )
    claim_body = {
        "schema": CLAIM_SCHEMA,
        "version": PROTOCOL_VERSION,
        "status": "CLAIMED_PENDING_NON_REUSABLE",
        "claimed_at_utc": datetime.now(UTC).isoformat(),
        "authorization_id": authorization["authorization_id"],
        "authorization_path": str(authorization_snapshot.path),
        "authorization_sha256": authorization_snapshot.sha256,
        "universe_id": authorization["universe"]["universe_id"],
        "claim_key_basis": "preblind_commitment_sha256_only",
        "preblind_commitment_sha256": authorization["universe"][
            "preblind_commitment_sha256"
        ],
        "run_id": authorization["execution"]["run_id"],
        "failure_is_non_reusable": True,
        "crash_is_non_reusable": True,
        "overwrite_allowed": False,
        "split_assignment_started": False,
        "example_build_started": False,
        "test_members_derived": False,
    }
    claim = {
        **claim_body,
        "canonical_digest_sha256": canonical_sha256(claim_body),
    }
    payload = _json_bytes(claim)
    _verify_directory(registry, label="registry immediately before claim")
    try:
        publication = _exclusive_create(paths["claim"], payload)
    except FileExistsError as exc:
        raise PostfreezeBlindV7Error(
            "this preblind commitment has already been claimed"
        ) from exc
    observed = _capture_file(
        paths["claim"],
        label="persisted one-shot claim",
    )
    if observed.payload != payload or observed.sha256 != publication["sha256"]:
        raise PostfreezeBlindV7Error(
            "one-shot claim persistence verification failed"
        )
    _verify_directory(registry, label="registry immediately after claim")
    return {
        "path": paths["claim"],
        "snapshot": observed,
        "receipt": claim,
        "publication": publication,
    }


def _assert_claim_persisted(
    claim: Mapping[str, Any],
    *,
    commitment_sha256: str,
) -> None:
    snapshot = _capture_file(
        Path(str(claim["path"])),
        label="claim gate",
    )
    if snapshot != claim["snapshot"]:
        raise PostfreezeBlindV7Error(
            "claim changed before reserved-data derivation"
        )
    receipt = _parse_json_bytes(snapshot.payload, label="claim gate")
    if (
        receipt.get("schema") != CLAIM_SCHEMA
        or receipt.get("status") != "CLAIMED_PENDING_NON_REUSABLE"
        or receipt.get("preblind_commitment_sha256")
        != commitment_sha256
        or receipt.get("claim_key_basis")
        != "preblind_commitment_sha256_only"
    ):
        raise PostfreezeBlindV7Error(
            "claim gate does not authorize derivation"
        )


def _write_snapshot_copy(path: Path, snapshot: StableSnapshot) -> None:
    _exclusive_create(path, snapshot.payload)


def _load_nonblind_rows_after_claim(
    *,
    dataset_dir: Path,
    manifest: Mapping[str, Any],
    claim: Mapping[str, Any],
    commitment_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _assert_claim_persisted(
        claim,
        commitment_sha256=commitment_sha256,
    )
    root = _directory_anchor(dataset_dir, label="nonblind dataset")
    manifest_splits = _require_mapping(
        manifest.get("splits"),
        label="manifest nonblind splits",
    )
    rows: list[dict[str, Any]] = []
    files: dict[str, dict[str, Any]] = {}
    for split in evidence_sft_v6.NONBLIND_SPLITS:
        descriptor = _require_mapping(
            manifest_splits.get(split),
            label=f"manifest split {split}",
        )
        snapshot = _capture_file(
            root.path / f"{split}.jsonl",
            label=f"frozen nonblind split {split}",
            maximum_bytes=MAX_SOURCE_BYTES,
        )
        if (
            snapshot.bytes != descriptor.get("bytes")
            or snapshot.sha256 != descriptor.get("sha256")
        ):
            raise PostfreezeBlindV7Error(
                f"frozen nonblind split changed: {split}"
            )
        split_rows = _parse_jsonl_bytes(
            snapshot.payload,
            label=f"frozen nonblind split {split}",
        )
        if (
            len(split_rows) != descriptor.get("count")
            or any(row.get("split") != split for row in split_rows)
        ):
            raise PostfreezeBlindV7Error(
                f"frozen nonblind split membership mismatch: {split}"
            )
        rows.extend(split_rows)
        files[split] = snapshot.receipt()
    if len(rows) != nonblind_sft_v7.EXPECTED_NONBLIND_TOTAL:
        raise PostfreezeBlindV7Error(
            "frozen nonblind dataset must contain exactly 550 rows"
        )
    return rows, files


def _validate_derived_shape(
    *,
    families: Sequence[Any],
    assignments: Mapping[str, str],
    examples: Sequence[Mapping[str, Any]],
    nonblind_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(families) != evidence_sft_v6.EXPECTED_FAMILY_COUNT:
        raise PostfreezeBlindV7Error(
            "committed sources must contain exactly 14 families"
        )
    family_by_id = {family.source_id: family for family in families}
    if (
        len(family_by_id) != len(families)
        or set(assignments) != set(family_by_id)
    ):
        raise PostfreezeBlindV7Error(
            "family assignment membership mismatch"
        )
    split_counts = Counter(assignments.values())
    if {
        split: split_counts[split]
        for split in evidence_sft_v6.SPLITS
    } != evidence_sft_v6.EXPECTED_FAMILY_SPLIT_COUNTS:
        raise PostfreezeBlindV7Error(
            "family split counts differ from the frozen algorithm"
        )
    reserved_families = {
        source_id
        for source_id, split in assignments.items()
        if split == "blind_test"
    }
    if len(reserved_families) != EXPECTED_FAMILIES:
        raise PostfreezeBlindV7Error(
            "reserved test must contain exactly three source families"
        )
    family_domain_counts = Counter(
        family_by_id[source_id].namespace
        for source_id in reserved_families
    )
    if family_domain_counts != Counter(
        {domain: 1 for domain in evidence_sft_v6.DOMAINS}
    ):
        raise PostfreezeBlindV7Error(
            "reserved test must contain one family per domain"
        )
    if len(examples) != EXPECTED_ROWS:
        raise PostfreezeBlindV7Error(
            "reserved test must contain exactly 150 rows"
        )
    example_ids = [str(row.get("example_id", "")) for row in examples]
    if (
        any(not item for item in example_ids)
        or len(set(example_ids)) != EXPECTED_ROWS
    ):
        raise PostfreezeBlindV7Error(
            "reserved example IDs must be non-empty and unique"
        )
    if any(row.get("split") != "blind_test" for row in examples):
        raise PostfreezeBlindV7Error(
            "derived row has a non-reserved split label"
        )
    source_counts = Counter(str(row.get("source_id")) for row in examples)
    if (
        set(source_counts) != reserved_families
        or set(source_counts.values()) != {EXAMPLES_PER_FAMILY}
    ):
        raise PostfreezeBlindV7Error(
            "reserved test must contain three families with 50 rows each"
        )
    domain_counts = Counter(str(row.get("domain")) for row in examples)
    if domain_counts != Counter(
        {domain: EXAMPLES_PER_FAMILY for domain in evidence_sft_v6.DOMAINS}
    ):
        raise PostfreezeBlindV7Error(
            "reserved test must contain 50 rows per domain"
        )
    nonblind_sources = {
        str(row.get("source_id")) for row in nonblind_rows
    }
    if nonblind_sources & reserved_families:
        raise PostfreezeBlindV7Error(
            "reserved and nonblind source families overlap"
        )
    for example in examples:
        try:
            evidence_sft_v6.validate_example(example)
        except Exception as exc:
            raise PostfreezeBlindV7Error(
                f"derived example contract failed: {example.get('example_id')}"
            ) from exc
    combined = [*nonblind_rows, *examples]
    balance = evidence_sft_v6._balance_report(combined, assignments)
    groups = evidence_sft_v6._group_isolation_report(
        families,
        assignments,
    )
    leakage = evidence_sft_v6._content_leakage_report(
        combined,
        splits=evidence_sft_v6.SPLITS,
    )
    if any(
        report.get("status") != "PASS"
        for report in (balance, groups, leakage)
    ):
        raise PostfreezeBlindV7Error(
            "combined 700-row balance/group/leakage audit failed"
        )
    return {
        "rows": EXPECTED_ROWS,
        "families": EXPECTED_FAMILIES,
        "examples_per_family": EXAMPLES_PER_FAMILY,
        "family_ids": sorted(reserved_families),
        "family_counts": dict(sorted(source_counts.items())),
        "domain_counts": {
            domain: domain_counts[domain]
            for domain in evidence_sft_v6.DOMAINS
        },
        "unique_example_ids": EXPECTED_ROWS,
        "nonblind_source_overlap_count": 0,
        "balance": balance,
        "group_isolation": groups,
        "content_leakage": leakage,
    }


def _derive_rows_after_claim(
    *,
    preflight: Mapping[str, Any],
    claim: Mapping[str, Any],
    staging: Path,
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[pointer_hf_eval_v6.DatasetRowV6, ...],
    bytes,
    dict[str, Any],
]:
    authorization = preflight["authorization"]
    commitment_sha = authorization["universe"][
        "preblind_commitment_sha256"
    ]
    _assert_claim_persisted(
        claim,
        commitment_sha256=commitment_sha,
    )
    sources = preflight["reverified"]["sources"]
    source_work = staging / ".src"
    source_work.mkdir(exist_ok=False)
    copied = {
        "licensed_chunks": source_work / "chunks.jsonl",
        "rag_manifest": source_work / "manifest.json",
        "semantic_inventory": source_work / "inventory.json",
        "semantic_records": source_work / "records.v7.jsonl",
    }
    for role, target in copied.items():
        _write_snapshot_copy(target, sources[role])
    try:
        # The persisted claim above is the sole gate before these calls.
        families = evidence_sft_v6.load_licensed_families(
            copied["licensed_chunks"]
        )
        semantic_inventory, semantic_audit = (
            evidence_sft_v6.load_semantic_inventory(
                copied["semantic_inventory"],
                families,
            )
        )
        assignments = evidence_sft_v6.assign_family_splits(
            families,
            seed=authorization["commitment"]["seed"],
        )
        examples = evidence_sft_v6.build_examples(
            families,
            assignments,
            semantic_inventory,
            seed=authorization["commitment"]["seed"],
            examples_per_family=EXAMPLES_PER_FAMILY,
            included_splits=("blind_test",),
        )
    except Exception as exc:
        raise PostfreezeBlindV7Error(
            "post-claim deterministic reserved-set derivation failed"
        ) from exc
    finally:
        shutil.rmtree(source_work, ignore_errors=True)
    if (
        semantic_audit.get("status") != "PASS"
        or semantic_audit.get("semantic_inventory_sha256")
        != sources["semantic_inventory"].sha256
        or semantic_audit.get("semantic_records_sha256")
        != sources["semantic_records"].sha256
    ):
        raise PostfreezeBlindV7Error(
            "semantic inventory differs from committed source snapshots"
        )
    nonblind_rows, nonblind_files = _load_nonblind_rows_after_claim(
        dataset_dir=Path(
            str(authorization["nonblind_dataset"]["directory"])
        ),
        manifest=preflight["reverified"]["committed"]["manifest"],
        claim=claim,
        commitment_sha256=commitment_sha,
    )
    shape = _validate_derived_shape(
        families=families,
        assignments=assignments,
        examples=examples,
        nonblind_rows=nonblind_rows,
    )
    frozen_examples = tuple(
        json.loads(canonical_json(dict(example)))
        for example in examples
    )
    payload = _jsonl_bytes(frozen_examples)
    dataset_rows: list[pointer_hf_eval_v6.DatasetRowV6] = []
    for index, example in enumerate(frozen_examples, 1):
        try:
            dataset_rows.append(
                pointer_hf_eval_v6._validate_dataset_row(
                    example,
                    split="blind_test",
                    line_number=index,
                )
            )
        except pointer_hf_eval_v6.PointerHFEvalV6Error as exc:
            raise PostfreezeBlindV7Error(
                f"derived evaluator row is invalid: {index}"
            ) from exc
    if len(dataset_rows) != EXPECTED_ROWS:
        raise PostfreezeBlindV7Error(
            "evaluator row conversion changed the reserved-set count"
        )
    derivation = {
        "status": "PASS_POSTCLAIM_DETERMINISTIC_DERIVATION",
        "commitment_sha256": commitment_sha,
        "seed": authorization["commitment"]["seed"],
        "seed_sha256": authorization["commitment"]["seed_sha256"],
        "builder_version": authorization["commitment"][
            "builder_version"
        ],
        "core_builder_version": authorization["commitment"][
            "core_builder_version"
        ],
        "split_algorithm_version": authorization["commitment"][
            "split_algorithm_version"
        ],
        "source_snapshot_set_sha256": authorization["sources"][
            "content_set_sha256"
        ],
        "shape": shape,
        "semantic_inventory_audit_status": semantic_audit["status"],
        "nonblind_files_reopened_after_claim": nonblind_files,
        "materialized_rows": EXPECTED_ROWS,
        "materialization_reopened_for_evaluation": False,
        "same_in_memory_rows_used_for_evaluation": True,
    }
    return frozen_examples, tuple(dataset_rows), payload, derivation


@contextmanager
def _offline_hf_environment() -> Any:
    names = (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
    )
    previous = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ[name] = "1"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _sample_bindings(
    *,
    authorization: Mapping[str, Any],
    claim_sha256: str,
    derived_sha256: str,
) -> dict[str, Any]:
    model = _require_mapping(
        authorization.get("model"),
        label="authorization model",
    )
    return {
        "authorization_sha256": authorization[
            "_authorization_sha256"
        ],
        "claim_sha256": claim_sha256,
        "preblind_commitment_sha256": authorization["universe"][
            "preblind_commitment_sha256"
        ],
        "derived_test_sha256": derived_sha256,
        "selection_binding_digest_sha256": authorization["upstream"][
            "lifecycle_binding"
        ]["selection"]["selection_binding_digest_sha256"],
        "base_model_tree_sha256": model["base_model_tree_sha256"],
        "checkpoint_tree_sha256": model["checkpoint_tree_sha256"],
        "adapter_tree_sha256": model["adapter_tree_sha256"],
        "protocol_source_sha256": authorization["implementation"][
            "protocol"
        ]["sha256"],
        "evaluator_source_sha256": authorization["implementation"][
            "pointer_evaluator"
        ]["sha256"],
        "compiler_source_sha256": authorization["implementation"][
            "pointer_compiler"
        ]["sha256"],
    }


def _validate_sample_set(
    samples: Sequence[Mapping[str, Any]],
    *,
    expected_example_ids: set[str],
    authorization: Mapping[str, Any],
    claim_sha256: str,
    derived_sha256: str,
) -> None:
    if (
        len(samples) != EXPECTED_ROWS
        or len(expected_example_ids) != EXPECTED_ROWS
    ):
        raise PostfreezeBlindV7Error(
            "sample binding requires the complete 150-row set"
        )
    observed_ids = [row.get("example_id") for row in samples]
    if (
        any(not isinstance(item, str) or not item for item in observed_ids)
        or len(set(observed_ids)) != EXPECTED_ROWS
        or set(observed_ids) != expected_example_ids
    ):
        raise PostfreezeBlindV7Error(
            "sample IDs differ from the materialized reserved test"
        )
    bindings = _sample_bindings(
        authorization=authorization,
        claim_sha256=claim_sha256,
        derived_sha256=derived_sha256,
    )
    reserved_policy = {
        "post_generation_scoring_only": True,
        "model_selection_performed": False,
        "checkpoint_ranking_performed": False,
        "threshold_tuning_performed": False,
        "calibration_performed": False,
    }
    for sample in samples:
        data_flow = _require_mapping(
            sample.get("data_flow"),
            label="sample data flow",
        )
        if (
            sample.get("schema") != SAMPLE_SCHEMA
            or sample.get("postfreeze_protocol_version") != PROTOCOL_VERSION
            or sample.get("split") != "blind_test"
            or sample.get("backend") != "hf_model"
            or sample.get("bindings") != bindings
            or sample.get("reserved_use_policy") != reserved_policy
            or data_flow.get("expected_passed_to_model") is not False
            or data_flow.get("expected_passed_to_candidate_compiler")
            is not False
            or data_flow.get("gold_repair_applied") is not False
            or data_flow.get("assistant_target_visible") is not False
            or data_flow.get("blind_data_accessed") is not False
        ):
            raise PostfreezeBlindV7Error(
                f"sample execution binding mismatch: "
                f"{sample.get('example_id')}"
            )


def _recompute_postfreeze_samples(
    *,
    rows: Sequence[pointer_hf_eval_v6.DatasetRowV6],
    recorded_samples: Sequence[Mapping[str, Any]],
    authorization: Mapping[str, Any],
    claim_sha256: str,
    derived_sha256: str,
) -> list[dict[str, Any]]:
    if (
        len(rows) != EXPECTED_ROWS
        or len(recorded_samples) != EXPECTED_ROWS
    ):
        raise PostfreezeBlindV7Error(
            "postfreeze recompilation requires the complete 150-row set"
        )
    bindings = _sample_bindings(
        authorization=authorization,
        claim_sha256=claim_sha256,
        derived_sha256=derived_sha256,
    )
    reserved_policy = {
        "post_generation_scoring_only": True,
        "model_selection_performed": False,
        "checkpoint_ranking_performed": False,
        "threshold_tuning_performed": False,
        "calibration_performed": False,
    }
    recomputed: list[dict[str, Any]] = []
    for row, recorded in zip(rows, recorded_samples, strict=True):
        if recorded.get("example_id") != row.example_id:
            raise PostfreezeBlindV7Error(
                "postfreeze sample ordering differs from derived membership"
            )
        try:
            rebuilt = pointer_hf_eval_v6._score_row(
                row=row,
                generation=_generation_from_record(
                    recorded.get("generation"),
                    label=f"postfreeze generation {row.example_id}",
                    latency_required=True,
                ),
                bindings=bindings,
                backend_mode="hf_model",
            )
        except pointer_hf_eval_v6.PointerHFEvalV6Error as exc:
            raise PostfreezeBlindV7Error(
                f"postfreeze sample recompilation failed: {row.example_id}"
            ) from exc
        rebuilt["schema"] = SAMPLE_SCHEMA
        rebuilt["postfreeze_protocol_version"] = PROTOCOL_VERSION
        rebuilt["reserved_use_policy"] = dict(reserved_policy)
        if rebuilt != dict(recorded):
            raise PostfreezeBlindV7Error(
                "postfreeze sample differs from raw-pointer recompilation: "
                f"{row.example_id}"
            )
        recomputed.append(rebuilt)
    return recomputed


def _rederive_reserved_evidence_for_verifier(
    *,
    authorization: Mapping[str, Any],
    authorization_snapshot: StableSnapshot,
    claim_snapshot: StableSnapshot,
    claim_receipt: Mapping[str, Any],
    reverified: Mapping[str, Any],
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[pointer_hf_eval_v6.DatasetRowV6, ...],
    bytes,
    dict[str, Any],
]:
    claim = {
        "path": claim_snapshot.path,
        "snapshot": claim_snapshot,
        "receipt": dict(claim_receipt),
    }
    preflight = {
        "authorization_snapshot": authorization_snapshot,
        "authorization": authorization,
        "reverified": reverified,
    }
    with tempfile.TemporaryDirectory(
        prefix="icmat-postfreeze-v7-verify-"
    ) as temporary:
        return _derive_rows_after_claim(
            preflight=preflight,
            claim=claim,
            staging=Path(temporary),
        )


def _evaluate_hf_cuda_once(
    rows: Sequence[pointer_hf_eval_v6.DatasetRowV6],
    *,
    authorization: Mapping[str, Any],
    claim: Mapping[str, Any],
    derived_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(rows) != EXPECTED_ROWS:
        raise PostfreezeBlindV7Error(
            "HF execution requires the complete 150-row in-memory set"
        )
    requests = pointer_hf_eval_v6._generation_requests(rows)
    if (
        len(requests) != EXPECTED_ROWS
        or len({request.example_id for request in requests})
        != EXPECTED_ROWS
    ):
        raise PostfreezeBlindV7Error(
            "target-free HF request membership is incomplete"
        )
    model = _require_mapping(
        authorization.get("model"),
        label="authorization model",
    )
    with _offline_hf_environment():
        try:
            generations, backend = pointer_hf_eval_v6.generate_hf_model(
                requests,
                base_model_dir=Path(str(model["base_model_path"])),
                adapter_dir=Path(str(model["adapter_runtime_path"])),
                device="cuda",
                seed=FIXED_SEED,
            )
        except pointer_hf_eval_v6.PointerHFEvalV6Error as exc:
            raise PostfreezeBlindV7Error(
                "the sole HF CUDA evaluation failed"
            ) from exc
    if set(generations) != {row.example_id for row in rows}:
        raise PostfreezeBlindV7Error(
            "HF generation membership differs from the derived set"
        )
    failed = [
        example_id
        for example_id, result in generations.items()
        if result.generation_error is not None
    ]
    if failed:
        raise PostfreezeBlindV7Error(
            "the sole HF run contains generation failures"
        )
    if (
        backend.get("mode") != "hf_model"
        or backend.get("device") != "cuda"
        or backend.get("seed") != FIXED_SEED
        or backend.get("samples_generated") != EXPECTED_ROWS
        or backend.get("local_files_only") is not True
    ):
        raise PostfreezeBlindV7Error(
            "HF backend differs from the frozen CUDA model contract"
        )
    _verify_backend_full_model_tree(
        backend,
        model=model,
        adapter_required=True,
        label="one-shot HF backend",
    )
    bindings = _sample_bindings(
        authorization=authorization,
        claim_sha256=claim["snapshot"].sha256,
        derived_sha256=derived_sha256,
    )
    samples: list[dict[str, Any]] = []
    for row in rows:
        try:
            sample = pointer_hf_eval_v6._score_row(
                row=row,
                generation=generations[row.example_id],
                bindings=bindings,
                backend_mode="hf_model",
            )
        except pointer_hf_eval_v6.PointerHFEvalV6Error as exc:
            raise PostfreezeBlindV7Error(
                f"post-generation scoring failed: {row.example_id}"
            ) from exc
        sample["schema"] = SAMPLE_SCHEMA
        sample["postfreeze_protocol_version"] = PROTOCOL_VERSION
        sample["reserved_use_policy"] = {
            "post_generation_scoring_only": True,
            "model_selection_performed": False,
            "checkpoint_ranking_performed": False,
            "threshold_tuning_performed": False,
            "calibration_performed": False,
        }
        samples.append(sample)
    return samples, dict(backend)


def _ratio_gate(
    *,
    name: str,
    numerator: int,
    denominator: int,
    minimum: Mapping[str, int],
) -> dict[str, Any]:
    required_numerator = int(minimum["numerator"])
    required_denominator = int(minimum["denominator"])
    passed = (
        denominator > 0
        and numerator * required_denominator
        >= required_numerator * denominator
    )
    return {
        "gate": name,
        "actual": {
            "numerator": numerator,
            "denominator": denominator,
        },
        "required_minimum": {
            "numerator": required_numerator,
            "denominator": required_denominator,
        },
        "passed": passed,
    }


def _release_gate_results(
    samples: Sequence[Mapping[str, Any]],
    *,
    backend: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    answer_rows = [
        row
        for row in samples
        if _require_mapping(
            _require_mapping(
                row.get("expected"),
                label="sample expected",
            ).get("answer"),
            label="sample expected answer",
        ).get("decision")
        == "ANSWER"
    ]
    refuse_rows = [
        row
        for row in samples
        if _require_mapping(
            _require_mapping(
                row.get("expected"),
                label="sample expected",
            ).get("answer"),
            label="sample expected answer",
        ).get("decision")
        == "REFUSE"
    ]
    if len(answer_rows) + len(refuse_rows) != len(samples):
        raise PostfreezeBlindV7Error(
            "sample expected decisions are invalid"
        )

    def flag_count(field: str, name: str) -> int:
        count = 0
        for row in samples:
            metrics = _require_mapping(
                row.get(field),
                label=f"sample {field}",
            )
            value = metrics.get(name)
            if not isinstance(value, bool):
                raise PostfreezeBlindV7Error(
                    f"sample {field}.{name} must be boolean"
                )
            count += int(value)
        return count

    def predicted_decision(row: Mapping[str, Any]) -> str | None:
        pointer = _require_mapping(
            row.get("pointer_metrics"),
            label="sample pointer metrics",
        )
        if pointer.get("compiler_accepted") is not True:
            return None
        compilation = _require_mapping(
            row.get("compilation"),
            label="sample compilation",
        )
        parsed = compilation.get("parsed_pointer")
        return (
            parsed.get("decision")
            if isinstance(parsed, Mapping)
            else None
        )

    tp = sum(predicted_decision(row) == "REFUSE" for row in refuse_rows)
    fp = sum(predicted_decision(row) == "REFUSE" for row in answer_rows)
    fn = len(refuse_rows) - tp
    wrong_answers = flag_count(
        "compiled_metrics",
        "unsupported_wrong_answer",
    )
    return [
        {
            "gate": "REAL_HF_CUDA_MODEL_BOUND",
            "actual_backend": backend.get("mode"),
            "actual_device": backend.get("device"),
            "required_backend": policy["required_backend"],
            "required_device": policy["required_device"],
            "passed": (
                backend.get("mode") == policy["required_backend"]
                and backend.get("device") == policy["required_device"]
                and policy["model_bound_required"] is True
            ),
        },
        {
            "gate": "COMPLETE_150",
            "actual": len(samples),
            "required": EXPECTED_ROWS,
            "passed": len(samples) == EXPECTED_ROWS,
        },
        _ratio_gate(
            name="POINTER_PARSE_VALID_100_PERCENT",
            numerator=flag_count("pointer_metrics", "parse_valid"),
            denominator=len(samples),
            minimum=policy["pointer_parse_valid"],
        ),
        _ratio_gate(
            name="POINTER_COMPILER_ACCEPTED_100_PERCENT",
            numerator=flag_count(
                "pointer_metrics",
                "compiler_accepted",
            ),
            denominator=len(samples),
            minimum=policy["pointer_compiler_accepted"],
        ),
        {
            "gate": "UNSUPPORTED_WRONG_ANSWER_ZERO",
            "actual": wrong_answers,
            "required_maximum": policy[
                "unsupported_wrong_answer_max_count"
            ],
            "passed": (
                wrong_answers
                <= int(policy["unsupported_wrong_answer_max_count"])
            ),
        },
        _ratio_gate(
            name="COMPILED_SCHEMA_VALID_100_PERCENT",
            numerator=flag_count("compiled_metrics", "schema_valid"),
            denominator=len(samples),
            minimum=policy["compiled_schema_valid"],
        ),
        _ratio_gate(
            name="COMPILED_CITATION_EXACT_100_PERCENT",
            numerator=flag_count("compiled_metrics", "citation_exact"),
            denominator=len(samples),
            minimum=policy["compiled_citation_exact"],
        ),
        _ratio_gate(
            name="COMPILED_PROVENANCE_EXACT_100_PERCENT",
            numerator=flag_count(
                "compiled_metrics",
                "provenance_exact",
            ),
            denominator=len(samples),
            minimum=policy["compiled_provenance_exact"],
        ),
        _ratio_gate(
            name="ANSWER_SPAN_EXACT_AT_LEAST_95_PERCENT",
            numerator=sum(
                bool(row["pointer_metrics"]["span_exact"])
                for row in answer_rows
            ),
            denominator=len(answer_rows),
            minimum=policy["answer_span_exact_minimum"],
        ),
        _ratio_gate(
            name="REFUSE_F1_AT_LEAST_95_PERCENT",
            numerator=2 * tp,
            denominator=2 * tp + fp + fn,
            minimum=policy["refuse_f1_minimum"],
        ),
    ]


def _metric(
    samples: Sequence[Mapping[str, Any]],
    *,
    field: str,
    name: str,
) -> dict[str, Any]:
    numerator = sum(bool(row[field][name]) for row in samples)
    denominator = len(samples)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else 0.0,
    }


def _build_summary(
    samples: Sequence[Mapping[str, Any]],
    *,
    backend: Mapping[str, Any],
    derivation: Mapping[str, Any],
    gate_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    qualified = all(bool(gate["passed"]) for gate in gate_results)
    return {
        "schema": SUMMARY_SCHEMA,
        "version": PROTOCOL_VERSION,
        "status": (
            QUALIFICATION_PASS_STATUS
            if qualified
            else QUALIFICATION_HOLD_STATUS
        ),
        "rows": len(samples),
        "complete_split": len(samples) == EXPECTED_ROWS,
        "backend": {
            "mode": backend.get("mode"),
            "device": backend.get("device"),
            "seed": backend.get("seed"),
            "samples_generated": backend.get("samples_generated"),
            "local_files_only": backend.get("local_files_only"),
        },
        "pointer_metrics": {
            name: _metric(
                samples,
                field="pointer_metrics",
                name=name,
            )
            for name in (
                "parse_valid",
                "compiler_accepted",
                "span_exact",
                "strict_exact",
            )
        },
        "compiled_metrics": {
            name: _metric(
                samples,
                field="compiled_metrics",
                name=name,
            )
            for name in (
                "schema_valid",
                "citation_exact",
                "provenance_exact",
                "strict_exact",
                "unsupported_wrong_answer",
            )
        },
        "derivation": dict(derivation),
        "gate_results": [dict(gate) for gate in gate_results],
        "authorization": {
            "gguf_offline_candidate_authorized": qualified,
            "activation_authorized": False,
            "deployment_authorized": False,
            "production_integration_authorized": False,
            "model_selection_authorized": False,
            "threshold_tuning_authorized": False,
            "calibration_authorized": False,
        },
        "reserved_use_policy": {
            "model_selection_performed": False,
            "checkpoint_ranking_performed": False,
            "threshold_tuning_performed": False,
            "calibration_performed": False,
            "retry_allowed": False,
        },
        "security_boundary": {
            "protocol_execution_constraint": True,
            "cryptographic_secrecy": False,
            "honest_local_execution_environment_required": True,
            "administrator_forgery_resistant": False,
            "tpm_or_external_signature_verified": False,
        },
    }


def _build_qualification(
    *,
    authorization: Mapping[str, Any],
    authorization_snapshot: StableSnapshot,
    claim: Mapping[str, Any],
    summary: Mapping[str, Any],
    gate_results: Sequence[Mapping[str, Any]],
    derived_artifact: Mapping[str, Any],
    sample_artifact: Mapping[str, Any],
    summary_artifact: Mapping[str, Any],
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    qualified = all(bool(gate["passed"]) for gate in gate_results)
    body = {
        "schema": QUALIFICATION_SCHEMA,
        "version": PROTOCOL_VERSION,
        "created_at_utc": (
            datetime.now(UTC).isoformat()
            if created_at_utc is None
            else created_at_utc
        ),
        "status": (
            QUALIFICATION_PASS_STATUS
            if qualified
            else QUALIFICATION_HOLD_STATUS
        ),
        "qualified": qualified,
        "authorization": {
            "path": str(authorization_snapshot.path),
            "sha256": authorization_snapshot.sha256,
            "authorization_id": authorization["authorization_id"],
            "policy_sha256": canonical_sha256(RELEASE_POLICY),
        },
        "claim": {
            "path": str(claim["path"]),
            "sha256": claim["snapshot"].sha256,
            "preblind_commitment_sha256": authorization["universe"][
                "preblind_commitment_sha256"
            ],
            "failure_is_non_reusable": True,
        },
        "upstream": {
            "selection_binding_digest_sha256": authorization["upstream"][
                "lifecycle_binding"
            ]["selection"]["selection_binding_digest_sha256"],
            "selected_checkpoint_tree_sha256": authorization["model"][
                "checkpoint_tree_sha256"
            ],
            "selected_adapter_tree_sha256": authorization["model"][
                "adapter_tree_sha256"
            ],
            "contract_set_sha256": authorization["upstream"][
                "lifecycle_binding"
            ]["contracts"]["contract_set_sha256"],
            "calibration_receipt_sha256": authorization["upstream"][
                "postselection"
            ]["calibration"]["receipt"]["sha256"],
            "ablation_receipt_sha256": authorization["upstream"][
                "postselection"
            ]["ablation"]["receipt"]["sha256"],
        },
        "artifacts": {
            DERIVED_FILENAME: dict(derived_artifact),
            SAMPLE_FILENAME: dict(sample_artifact),
            SUMMARY_FILENAME: dict(summary_artifact),
        },
        "thresholds": json.loads(canonical_json(RELEASE_POLICY)),
        "gate_results": [dict(gate) for gate in gate_results],
        "summary_status": summary["status"],
        "release_authorization": {
            "gguf_offline_candidate_authorized": qualified,
            "activation_authorized": False,
            "deployment_authorized": False,
            "production_integration_authorized": False,
            "x5_authorized": False,
            "bpu_authorized": False,
        },
        "reserved_use_policy": {
            "model_selection_performed": False,
            "checkpoint_ranking_performed": False,
            "threshold_tuning_performed": False,
            "calibration_performed": False,
            "retry_allowed": False,
        },
        "security_boundary": {
            "protocol_execution_constraint": True,
            "cryptographic_secrecy": False,
            "honest_local_execution_environment_required": True,
            "administrator_forgery_resistant": False,
            "tpm_or_external_signature_verified": False,
            "administrator_can_derive_outside_protocol": True,
        },
    }
    return {
        **body,
        "canonical_digest_sha256": canonical_sha256(body),
    }


def _artifact(payload: bytes, *, records: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }
    if records is not None:
        result["records"] = records
    return result


def _write_terminal(
    *,
    terminal_path: Path,
    authorization_snapshot: StableSnapshot,
    authorization: Mapping[str, Any],
    claim: Mapping[str, Any],
    status: str,
    artifacts: Mapping[str, Any] | None,
    error: BaseException | None,
) -> dict[str, Any]:
    if status not in {
        "COMPLETED_GGUF_OFFLINE_CANDIDATE_ONLY",
        "COMPLETED_HOLD_NON_REUSABLE",
        "FAILED_NON_REUSABLE",
    }:
        raise PostfreezeBlindV7Error("terminal status is invalid")
    error_record = None
    if error is not None:
        error_record = {
            "type": type(error).__name__,
            "message": str(error)[:MAX_ERROR_CHARS],
            "traceback": "".join(
                traceback.format_exception(
                    type(error),
                    error,
                    error.__traceback__,
                )
            )[: MAX_ERROR_CHARS * 4],
        }
    body = {
        "schema": TERMINAL_SCHEMA,
        "version": PROTOCOL_VERSION,
        "status": status,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "authorization_id": authorization["authorization_id"],
        "authorization_sha256": authorization_snapshot.sha256,
        "claim_path": str(claim["path"]),
        "claim_sha256": claim["snapshot"].sha256,
        "preblind_commitment_sha256": authorization["universe"][
            "preblind_commitment_sha256"
        ],
        "run_id": authorization["execution"]["run_id"],
        "artifacts": None if artifacts is None else dict(artifacts),
        "error": error_record,
        "failure_is_non_reusable": True,
        "crash_is_non_reusable": True,
        "overwrite_allowed": False,
    }
    terminal = {
        **body,
        "canonical_digest_sha256": canonical_sha256(body),
    }
    try:
        publication = _exclusive_create(
            terminal_path,
            _json_bytes(terminal),
        )
    except FileExistsError as exc:
        raise PostfreezeBlindV7Error(
            "terminal record already exists and cannot be overwritten"
        ) from exc
    return {
        "status": status,
        "publication": publication,
        "receipt": terminal,
    }


def _execute_postfreeze_blind_v7_at_registry(
    *,
    authorization_path: Path,
    registry_root: Path,
) -> dict[str, Any]:
    """Claim, derive, evaluate once on CUDA, and publish final evidence."""

    authorization_snapshot, authorization_loaded = _load_authorization(
        authorization_path
    )
    authorization = dict(authorization_loaded)
    authorization["_authorization_sha256"] = (
        authorization_snapshot.sha256
    )
    registry, paths = _paths_from_authorization(
        authorization,
        registry_root,
    )
    if authorization_snapshot.path != paths["authorization"]:
        raise PostfreezeBlindV7Error(
            "authorization path differs from the commitment registry"
        )
    if os.path.lexists(paths["claim"]) or os.path.lexists(paths["terminal"]):
        raise PostfreezeBlindV7Error(
            "this preblind commitment is already claimed or terminal"
        )
    if os.path.lexists(paths["evidence"]):
        raise PostfreezeBlindV7Error(
            "one-shot evidence path already exists"
        )

    reverified = _reverify_authorization_inputs(authorization)
    _require_cuda_ready()
    preflight = {
        "authorization_snapshot": authorization_snapshot,
        "authorization": authorization,
        "registry": registry,
        "paths": paths,
        "reverified": reverified,
    }
    claim = _claim_once(
        authorization_snapshot=authorization_snapshot,
        authorization=authorization,
        registry=registry,
        paths=paths,
    )

    staging = registry.path / f".stage-{uuid.uuid4().hex[:16]}"
    published_artifacts: dict[str, Any] | None = None
    try:
        staging.mkdir(exist_ok=False)
        examples, dataset_rows, derived_payload, derivation = (
            _derive_rows_after_claim(
                preflight=preflight,
                claim=claim,
                staging=staging,
            )
        )
        derived_path = staging / DERIVED_FILENAME
        _exclusive_create(derived_path, derived_payload)
        derived_artifact = _artifact(
            derived_payload,
            records=EXPECTED_ROWS,
        )
        if len(examples) != EXPECTED_ROWS:
            raise PostfreezeBlindV7Error(
                "materialization changed the in-memory row count"
            )

        samples, backend = _evaluate_hf_cuda_once(
            dataset_rows,
            authorization=authorization,
            claim=claim,
            derived_sha256=derived_artifact["sha256"],
        )
        _validate_sample_set(
            samples,
            expected_example_ids={
                str(example["example_id"]) for example in examples
            },
            authorization=authorization,
            claim_sha256=claim["snapshot"].sha256,
            derived_sha256=derived_artifact["sha256"],
        )
        final_reverification = _reverify_authorization_inputs(
            authorization
        )
        if (
            final_reverification["authority"]["lifecycle"].binding
            != reverified["authority"]["lifecycle"].binding
            or final_reverification["postselection"]
            != reverified["postselection"]
            or _snapshot_set_receipt(
                final_reverification["sources"]
            )
            != _snapshot_set_receipt(reverified["sources"])
        ):
            raise PostfreezeBlindV7Error(
                "authority changed during one-shot evaluation"
            )

        gate_results = _release_gate_results(
            samples,
            backend=backend,
            policy=RELEASE_POLICY,
        )
        summary = _build_summary(
            samples,
            backend=backend,
            derivation=derivation,
            gate_results=gate_results,
        )
        sample_payload = _jsonl_bytes(samples)
        summary_payload = _json_bytes(summary)
        sample_artifact = _artifact(
            sample_payload,
            records=EXPECTED_ROWS,
        )
        summary_artifact = _artifact(summary_payload)
        qualification = _build_qualification(
            authorization=authorization,
            authorization_snapshot=authorization_snapshot,
            claim=claim,
            summary=summary,
            gate_results=gate_results,
            derived_artifact=derived_artifact,
            sample_artifact=sample_artifact,
            summary_artifact=summary_artifact,
        )
        qualification_payload = _json_bytes(qualification)
        qualification_artifact = _artifact(qualification_payload)
        receipt_body = {
            "schema": RUN_RECEIPT_SCHEMA,
            "version": PROTOCOL_VERSION,
            "status": RUN_COMPLETE_STATUS,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "run_id": authorization["execution"]["run_id"],
            "examples": EXPECTED_ROWS,
            "authorization": {
                "path": str(authorization_snapshot.path),
                "sha256": authorization_snapshot.sha256,
                "authorization_id": authorization["authorization_id"],
            },
            "claim": {
                "path": str(claim["path"]),
                "sha256": claim["snapshot"].sha256,
                "created_before_split_assignment": True,
                "created_before_example_build": True,
                "created_before_test_member_derivation": True,
                "file_fsync_completed": True,
            },
            "derivation": derivation,
            "model": authorization["model"],
            "backend": backend,
            "artifacts": {
                DERIVED_FILENAME: derived_artifact,
                SAMPLE_FILENAME: sample_artifact,
                SUMMARY_FILENAME: summary_artifact,
                QUALIFICATION_FILENAME: qualification_artifact,
            },
            "authorization_boundary": {
                "gguf_offline_candidate_authorized": qualification[
                    "qualified"
                ],
                "activation_authorized": False,
                "deployment_authorized": False,
                "production_integration_authorized": False,
                "model_selection_authorized": False,
                "checkpoint_ranking_authorized": False,
                "threshold_tuning_authorized": False,
                "calibration_authorized": False,
                "retry_allowed": False,
            },
            "security_boundary": authorization["security_boundary"],
        }
        receipt = {
            **receipt_body,
            "canonical_digest_sha256": canonical_sha256(receipt_body),
        }
        receipt_payload = _json_bytes(receipt)
        for name, payload in (
            (SAMPLE_FILENAME, sample_payload),
            (SUMMARY_FILENAME, summary_payload),
            (QUALIFICATION_FILENAME, qualification_payload),
            (RUN_RECEIPT_FILENAME, receipt_payload),
        ):
            _exclusive_create(staging / name, payload)
        if {path.name for path in staging.iterdir()} != EVIDENCE_FILENAMES:
            raise PostfreezeBlindV7Error(
                "one-shot staging artifact whitelist mismatch"
            )
        _verify_directory(registry, label="registry before publication")
        if os.path.lexists(paths["evidence"]):
            raise PostfreezeBlindV7Error(
                "one-shot evidence publication collision"
            )
        os.rename(staging, paths["evidence"])
        _verify_directory(registry, label="registry after publication")
        published_artifacts = {
            name: {
                "path": str(paths["evidence"] / name),
                **descriptor,
            }
            for name, descriptor in {
                DERIVED_FILENAME: derived_artifact,
                SAMPLE_FILENAME: sample_artifact,
                SUMMARY_FILENAME: summary_artifact,
                QUALIFICATION_FILENAME: qualification_artifact,
                RUN_RECEIPT_FILENAME: _artifact(receipt_payload),
            }.items()
        }
        terminal_status = (
            "COMPLETED_GGUF_OFFLINE_CANDIDATE_ONLY"
            if qualification["qualified"]
            else "COMPLETED_HOLD_NON_REUSABLE"
        )
        terminal = _write_terminal(
            terminal_path=paths["terminal"],
            authorization_snapshot=authorization_snapshot,
            authorization=authorization,
            claim=claim,
            status=terminal_status,
            artifacts=published_artifacts,
            error=None,
        )
    except BaseException as exc:
        try:
            if staging.exists() and staging.parent == registry.path:
                shutil.rmtree(staging, ignore_errors=True)
        finally:
            try:
                _write_terminal(
                    terminal_path=paths["terminal"],
                    authorization_snapshot=authorization_snapshot,
                    authorization=authorization,
                    claim=claim,
                    status="FAILED_NON_REUSABLE",
                    artifacts=published_artifacts,
                    error=exc,
                )
            except BaseException as terminal_exc:
                raise PostfreezeBlindV7Error(
                    "one-shot failed after immutable claim; terminal "
                    f"publication also failed: {terminal_exc}"
                ) from exc
        if isinstance(exc, PostfreezeBlindV7Error):
            raise
        raise PostfreezeBlindV7Error(
            "one-shot failed after immutable claim"
        ) from exc

    return {
        "status": RUN_COMPLETE_STATUS,
        "qualified": qualification["qualified"],
        "qualification_status": qualification["status"],
        "evidence_dir": str(paths["evidence"]),
        "authorization_sha256": authorization_snapshot.sha256,
        "claim_sha256": claim["snapshot"].sha256,
        "terminal": terminal,
        "examples": EXPECTED_ROWS,
        "hf_calls": 1,
        "retry_allowed": False,
        "gguf_offline_candidate_authorized": qualification["qualified"],
        "deployment_authorized": False,
        "model_selection_performed": False,
        "threshold_tuning_performed": False,
        "calibration_performed": False,
        "cryptographic_secrecy": False,
        "honest_local_execution_environment_required": True,
        "administrator_forgery_resistant": False,
        "tpm_or_external_signature_verified": False,
    }


def execute_postfreeze_blind_v7(
    *,
    authorization_path: Path,
) -> dict[str, Any]:
    """Consume one authorization from the fixed workspace registry."""

    return _execute_postfreeze_blind_v7_at_registry(
        authorization_path=authorization_path,
        registry_root=_production_registry_root(create=False),
    )


def _load_canonical_receipt(
    snapshot: StableSnapshot,
    *,
    label: str,
) -> dict[str, Any]:
    receipt = _parse_json_bytes(snapshot.payload, label=label)
    body = dict(receipt)
    digest = _require_sha(
        body.pop("canonical_digest_sha256", None),
        label=f"{label} canonical digest",
    )
    if canonical_sha256(body) != digest:
        raise PostfreezeBlindV7Error(
            f"{label}: canonical digest mismatch"
        )
    return receipt


def _verify_release_qualification_v7_at_registry(
    *,
    authorization_path: Path,
    registry_root: Path,
) -> dict[str, Any]:
    """Verify a completed qualification without model execution or selection."""

    authorization_snapshot, authorization_loaded = _load_authorization(
        authorization_path
    )
    authorization = dict(authorization_loaded)
    authorization["_authorization_sha256"] = (
        authorization_snapshot.sha256
    )
    registry, paths = _paths_from_authorization(
        authorization,
        registry_root,
    )
    if authorization_snapshot.path != paths["authorization"]:
        raise PostfreezeBlindV7Error(
            "qualification authorization path mismatch"
        )
    reverified = _reverify_authorization_inputs(authorization)
    claim_snapshot = _capture_file(
        paths["claim"],
        label="qualification claim",
    )
    claim = _load_canonical_receipt(
        claim_snapshot,
        label="qualification claim",
    )
    if (
        claim.get("schema") != CLAIM_SCHEMA
        or claim.get("status") != "CLAIMED_PENDING_NON_REUSABLE"
        or claim.get("authorization_sha256")
        != authorization_snapshot.sha256
    ):
        raise PostfreezeBlindV7Error(
            "qualification claim binding mismatch"
        )
    terminal_snapshot = _capture_file(
        paths["terminal"],
        label="qualification terminal",
    )
    terminal = _load_canonical_receipt(
        terminal_snapshot,
        label="qualification terminal",
    )
    if (
        terminal.get("schema") != TERMINAL_SCHEMA
        or terminal.get("claim_sha256") != claim_snapshot.sha256
        or terminal.get("authorization_sha256")
        != authorization_snapshot.sha256
        or terminal.get("status")
        not in {
            "COMPLETED_GGUF_OFFLINE_CANDIDATE_ONLY",
            "COMPLETED_HOLD_NON_REUSABLE",
        }
    ):
        raise PostfreezeBlindV7Error(
            "qualification terminal binding mismatch"
        )
    evidence_anchor, snapshots = _capture_directory_artifacts(
        paths["evidence"],
        expected_names=EVIDENCE_FILENAMES,
        label="one-shot evidence",
    )
    receipt = _load_canonical_receipt(
        snapshots[RUN_RECEIPT_FILENAME],
        label="one-shot run receipt",
    )
    qualification = _load_canonical_receipt(
        snapshots[QUALIFICATION_FILENAME],
        label="GGUF qualification",
    )
    summary = _parse_json_bytes(
        snapshots[SUMMARY_FILENAME].payload,
        label="one-shot summary",
    )
    samples = _parse_jsonl_bytes(
        snapshots[SAMPLE_FILENAME].payload,
        label="one-shot samples",
    )
    derived_rows = _parse_jsonl_bytes(
        snapshots[DERIVED_FILENAME].payload,
        label="materialized reserved test",
    )
    if (
        receipt.get("schema") != RUN_RECEIPT_SCHEMA
        or receipt.get("version") != PROTOCOL_VERSION
        or receipt.get("status") != RUN_COMPLETE_STATUS
        or receipt.get("examples") != EXPECTED_ROWS
        or qualification.get("schema") != QUALIFICATION_SCHEMA
        or qualification.get("version") != PROTOCOL_VERSION
        or summary.get("schema") != SUMMARY_SCHEMA
        or len(samples) != EXPECTED_ROWS
        or len(derived_rows) != EXPECTED_ROWS
        or len({row.get("example_id") for row in derived_rows})
        != EXPECTED_ROWS
    ):
        raise PostfreezeBlindV7Error(
            "qualification evidence shape mismatch"
        )
    (
        rederived_examples,
        rederived_dataset_rows,
        rederived_payload,
        rederived_derivation,
    ) = _rederive_reserved_evidence_for_verifier(
        authorization=authorization,
        authorization_snapshot=authorization_snapshot,
        claim_snapshot=claim_snapshot,
        claim_receipt=claim,
        reverified=reverified,
    )
    if (
        rederived_payload != snapshots[DERIVED_FILENAME].payload
        or list(rederived_examples) != derived_rows
        or receipt.get("derivation") != rederived_derivation
    ):
        raise PostfreezeBlindV7Error(
            "materialized reserved test differs from committed-source "
            "rederivation"
        )
    _validate_sample_set(
        samples,
        expected_example_ids={
            str(row["example_id"]) for row in derived_rows
        },
        authorization=authorization,
        claim_sha256=claim_snapshot.sha256,
        derived_sha256=snapshots[DERIVED_FILENAME].sha256,
    )
    recomputed_samples = _recompute_postfreeze_samples(
        rows=rederived_dataset_rows,
        recorded_samples=samples,
        authorization=authorization,
        claim_sha256=claim_snapshot.sha256,
        derived_sha256=snapshots[DERIVED_FILENAME].sha256,
    )
    artifacts = _require_mapping(
        receipt.get("artifacts"),
        label="run receipt artifacts",
    )
    expected_artifacts = {
        DERIVED_FILENAME: snapshots[DERIVED_FILENAME],
        SAMPLE_FILENAME: snapshots[SAMPLE_FILENAME],
        SUMMARY_FILENAME: snapshots[SUMMARY_FILENAME],
        QUALIFICATION_FILENAME: snapshots[QUALIFICATION_FILENAME],
    }
    if set(artifacts) != set(expected_artifacts):
        raise PostfreezeBlindV7Error(
            "run receipt artifact membership mismatch"
        )
    for name, snapshot in expected_artifacts.items():
        descriptor = _require_mapping(
            artifacts.get(name),
            label=f"run artifact {name}",
        )
        if (
            descriptor.get("bytes") != snapshot.bytes
            or descriptor.get("sha256") != snapshot.sha256
        ):
            raise PostfreezeBlindV7Error(
                f"run artifact changed: {name}"
            )
    backend = _require_mapping(
        receipt.get("backend"),
        label="run backend",
    )
    model = _require_mapping(
        authorization.get("model"),
        label="authorization model",
    )
    backend_tree = _verify_backend_full_model_tree(
        backend,
        model=model,
        adapter_required=True,
        label="verified one-shot backend",
    )
    if (
        receipt.get("model") != model
        or receipt.get("security_boundary")
        != authorization.get("security_boundary")
    ):
        raise PostfreezeBlindV7Error(
            "run receipt model or security boundary mismatch"
        )
    recomputed_gates = _release_gate_results(
        recomputed_samples,
        backend=backend,
        policy=RELEASE_POLICY,
    )
    recomputed_summary = _build_summary(
        recomputed_samples,
        backend=backend,
        derivation=rederived_derivation,
        gate_results=recomputed_gates,
    )
    if summary != recomputed_summary:
        raise PostfreezeBlindV7Error(
            "one-shot summary differs from deterministic recomputation"
        )
    qualified = all(bool(gate["passed"]) for gate in recomputed_gates)
    expected_status = (
        QUALIFICATION_PASS_STATUS
        if qualified
        else QUALIFICATION_HOLD_STATUS
    )
    created_at_utc = qualification.get("created_at_utc")
    if not isinstance(created_at_utc, str) or not created_at_utc:
        raise PostfreezeBlindV7Error(
            "qualification creation timestamp is invalid"
        )
    claim_context = {
        "path": claim_snapshot.path,
        "snapshot": claim_snapshot,
        "receipt": claim,
    }
    expected_qualification = _build_qualification(
        authorization=authorization,
        authorization_snapshot=authorization_snapshot,
        claim=claim_context,
        summary=recomputed_summary,
        gate_results=recomputed_gates,
        derived_artifact=_artifact(
            snapshots[DERIVED_FILENAME].payload,
            records=EXPECTED_ROWS,
        ),
        sample_artifact=_artifact(
            snapshots[SAMPLE_FILENAME].payload,
            records=EXPECTED_ROWS,
        ),
        summary_artifact=_artifact(
            snapshots[SUMMARY_FILENAME].payload,
        ),
        created_at_utc=created_at_utc,
    )
    if qualification != expected_qualification:
        raise PostfreezeBlindV7Error(
            "GGUF qualification differs from deterministic recomputation"
        )
    release = _require_mapping(
        qualification.get("release_authorization"),
        label="release authorization",
    )
    reserved_use = _require_mapping(
        qualification.get("reserved_use_policy"),
        label="qualification reserved-use policy",
    )
    if (
        qualification.get("qualified") is not qualified
        or qualification.get("status") != expected_status
        or qualification.get("gate_results") != recomputed_gates
        or summary.get("status") != expected_status
        or summary.get("gate_results") != recomputed_gates
        or release.get("gguf_offline_candidate_authorized")
        is not qualified
        or any(
            release.get(key) is not False
            for key in (
                "activation_authorized",
                "deployment_authorized",
                "production_integration_authorized",
                "x5_authorized",
                "bpu_authorized",
            )
        )
        or any(
            reserved_use.get(key) is not False
            for key in (
                "model_selection_performed",
                "checkpoint_ranking_performed",
                "threshold_tuning_performed",
                "calibration_performed",
                "retry_allowed",
            )
        )
    ):
        raise PostfreezeBlindV7Error(
            "GGUF-only qualification recomputation failed"
        )
    expected_terminal = (
        "COMPLETED_GGUF_OFFLINE_CANDIDATE_ONLY"
        if qualified
        else "COMPLETED_HOLD_NON_REUSABLE"
    )
    if terminal["status"] != expected_terminal:
        raise PostfreezeBlindV7Error(
            "terminal status differs from qualification"
        )
    expected_terminal_artifacts = {
        name: {
            "path": str(paths["evidence"] / name),
            **_artifact(
                snapshots[name].payload,
                records=(
                    EXPECTED_ROWS
                    if name in {DERIVED_FILENAME, SAMPLE_FILENAME}
                    else None
                ),
            ),
        }
        for name in EVIDENCE_FILENAMES
    }
    if terminal.get("artifacts") != expected_terminal_artifacts:
        raise PostfreezeBlindV7Error(
            "terminal artifact inventory differs from published evidence"
        )
    for snapshot in snapshots.values():
        _verify_snapshot(
            snapshot,
            label=f"qualification final {snapshot.path.name}",
        )
    _verify_directory(evidence_anchor, label="evidence directory final")
    _verify_directory(registry, label="registry final")
    return {
        "status": "PASS_POSTFREEZE_V7_QUALIFICATION_VERIFIED",
        "qualified": qualified,
        "qualification_status": expected_status,
        "gguf_offline_candidate_authorized": qualified,
        "activation_authorized": False,
        "deployment_authorized": False,
        "production_integration_authorized": False,
        "model_selection_performed": False,
        "threshold_tuning_performed": False,
        "calibration_performed": False,
        "model_executed_by_verifier": False,
        "derived_rows_recomputed": True,
        "samples_recompiled": True,
        "backend_model_tree_reverified": True,
        "summary_recomputed": True,
        "qualification_recomputed": True,
        "calibration_reverified": True,
        "ablation_reverified": True,
        "backend_model_tree": {
            role: (
                None
                if inventory is None
                else {
                    "tree_sha256": inventory["tree_sha256"],
                    "files_count": inventory["files_count"],
                }
            )
            for role, inventory in backend_tree.items()
        },
        "network_accessed": False,
        "x5_accessed": False,
        "cryptographic_secrecy": False,
        "honest_local_execution_environment_required": True,
        "administrator_forgery_resistant": False,
        "tpm_or_external_signature_verified": False,
        "authorization_sha256": authorization_snapshot.sha256,
        "claim_sha256": claim_snapshot.sha256,
        "terminal_sha256": terminal_snapshot.sha256,
        "qualification_sha256": snapshots[
            QUALIFICATION_FILENAME
        ].sha256,
    }


def verify_release_qualification_v7(
    *,
    authorization_path: Path,
) -> dict[str, Any]:
    """Verify one completed run in the fixed workspace registry."""

    return _verify_release_qualification_v7_at_registry(
        authorization_path=authorization_path,
        registry_root=_production_registry_root(create=False),
    )


__all__ = [
    "AUTHORIZATION_SCHEMA",
    "AUTHORIZATION_STATUS",
    "CLAIM_SCHEMA",
    "EXPECTED_ROWS",
    "PostfreezeBlindV7Error",
    "PRODUCTION_REGISTRY_ROOT",
    "PROTOCOL_VERSION",
    "QUALIFICATION_HOLD_STATUS",
    "QUALIFICATION_PASS_STATUS",
    "QUALIFICATION_SCHEMA",
    "RELEASE_POLICY",
    "RUN_RECEIPT_SCHEMA",
    "SUMMARY_SCHEMA",
    "TERMINAL_SCHEMA",
    "execute_postfreeze_blind_v7",
    "prepare_postfreeze_blind_v7",
    "verify_release_qualification_v7",
]
