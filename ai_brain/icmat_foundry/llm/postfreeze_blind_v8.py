"""Native STRICT_NONBLIND_V8 one-shot post-freeze evaluation.

This module deliberately does not accept, translate, or upgrade any v7
selection, calibration, ablation, post-freeze, or qualification receipt.  It
binds the frozen v8 selection and post-selection evidence before creating one
global claim keyed only by the preblind commitment.  Reserved membership is
derived only after that claim is durably published.

The verifier deterministically re-derives all 150 rows, recompiles every
recorded raw pointer, recomputes the summary and qualification, and rechecks
the complete base-model and selected-adapter trees.  This is an auditable
honest-local execution protocol, not cryptographic secrecy, TPM attestation,
or administrator-forgery resistance.
"""

from __future__ import annotations

import hashlib
import json
import math
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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from icmat_foundry.llm import (
    ablation_eval_v6,
    ablation_eval_v8,
    calibration_eval_v8,
    evidence_pointer_v6,
    evidence_sft_v6,
    gguf_release_v8,
    lifecycle_bindings_v7,
    nonblind_sft_v7,
    nonblind_sft_v8,
    pointer_hf_eval_v6,
    selection_freeze_v8,
    semantic_queries_v7,
)

PROTOCOL_VERSION = gguf_release_v8.POSTFREEZE_VERSION
AUTHORIZATION_SCHEMA = "icmat_llm_postfreeze_authorization.v8"
AUTHORIZATION_STATUS = "AUTHORIZED_ONCE_NATIVE_V8_POSTFREEZE_GATES_PASS"
CLAIM_SCHEMA = "icmat_llm_postfreeze_consumption_claim.v8"
TERMINAL_SCHEMA = "icmat_llm_postfreeze_terminal.v8"
SAMPLE_SCHEMA = "icmat_llm_postfreeze_sample.v8"
SUMMARY_SCHEMA = "icmat_llm_postfreeze_summary.v8"
RUN_RECEIPT_SCHEMA = gguf_release_v8.POSTFREEZE_SCHEMA
RUN_COMPLETE_STATUS = gguf_release_v8.POSTFREEZE_STATUS
QUALIFICATION_SCHEMA = gguf_release_v8.QUALIFICATION_SCHEMA
QUALIFICATION_PASS_STATUS = gguf_release_v8.QUALIFICATION_STATUS
QUALIFICATION_HOLD_STATUS = "HOLD_POSTFREEZE_THRESHOLDS_NON_REUSABLE"
VERIFICATION_SCHEMA = gguf_release_v8.VERIFICATION_SCHEMA
VERIFICATION_VERSION = gguf_release_v8.VERIFICATION_VERSION
VERIFICATION_STATUS = gguf_release_v8.VERIFICATION_STATUS

EXPECTED_ROWS = 150
EXPECTED_FAMILIES = 3
EXAMPLES_PER_FAMILY = 50
EXPECTED_ANSWER_ROWS = 75
EXPECTED_REFUSE_ROWS = 75
FIXED_SEED = 20260729
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_SOURCE_BYTES = 512 * 1024 * 1024
MAX_ERROR_CHARS = 2000

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_REGISTRY_ROOT = (
    WORKSPACE_ROOT / "evaluation" / "icmat_foundry" / "llm" / "postfreeze_blind_v8_registry"
)

DERIVED_FILENAME = "derived_test.v8.jsonl"
SAMPLE_FILENAME = "sample_results.v8.jsonl"
SUMMARY_FILENAME = "summary.v8.json"
RUN_RECEIPT_FILENAME = "run_receipt.v8.json"
QUALIFICATION_FILENAME = "release_qualification.v8.json"
VERIFICATION_FILENAME = "independent_verification.v8.json"
EXECUTION_EVIDENCE_FILENAMES = {
    DERIVED_FILENAME,
    SAMPLE_FILENAME,
    SUMMARY_FILENAME,
    RUN_RECEIPT_FILENAME,
    QUALIFICATION_FILENAME,
}
VERIFIED_EVIDENCE_FILENAMES = {
    *EXECUTION_EVIDENCE_FILENAMES,
    VERIFICATION_FILENAME,
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
    "x5_execution_authorized": False,
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
    "semantic_requests",
    "semantic_request_manifest",
    "nonblind_v8_module",
    "nonblind_v7_module",
    "evidence_core",
    "semantic_core",
)
_FALSE_AUTHORIZATION = dict(gguf_release_v8.FALSE_AUTHORIZATION)
_AUTHORIZATION_FIELDS = {
    "schema",
    "version",
    "status",
    "authorization_id",
    "created_at_utc",
    "run_id",
    "chain_binding",
    "upstream_receipts",
    "dataset",
    "sources",
    "nli_model",
    "model",
    "postselection",
    "implementation",
    "execution",
    "release_policy",
    "registry",
    "claim_boundary",
    "security_boundary",
    "authorization",
    "canonical_digest_sha256",
}
_CLAIM_FIELDS = {
    "schema",
    "version",
    "status",
    "created_at_utc",
    "authorization_sha256",
    "authorization_id",
    "preblind_commitment_sha256",
    "nonce_sha256",
    "failure_is_non_reusable",
    "retry_allowed",
    "canonical_digest_sha256",
}
_TERMINAL_FIELDS = {
    "schema",
    "version",
    "status",
    "created_at_utc",
    "authorization_sha256",
    "claim_sha256",
    "failure_is_non_reusable",
    "error",
    "canonical_digest_sha256",
}


class PostfreezeBlindV8Error(RuntimeError):
    """Raised when the native-v8 protocol fails closed."""


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
    return b"".join((canonical_json(dict(value)) + "\n").encode("utf-8") for value in values)


def _mapping(
    value: Any,
    *,
    label: str,
    exact: set[str] | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PostfreezeBlindV8Error(f"{label} must be an object")
    if exact is not None and set(value) != exact:
        raise PostfreezeBlindV8Error(f"{label} exact field set mismatch")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PostfreezeBlindV8Error(f"{label} must be an array")
    return value


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise PostfreezeBlindV8Error(f"{label} is not a SHA-256")
    return value


def _parse_json(snapshot: lifecycle_bindings_v7.StableFileSnapshot, *, label: str) -> dict[str, Any]:
    try:
        return lifecycle_bindings_v7.parse_json_snapshot(snapshot, label=label)
    except (
        lifecycle_bindings_v7.LifecycleBindingV7Error,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        raise PostfreezeBlindV8Error(f"{label} is invalid: {exc}") from exc


def _parse_jsonl(payload: bytes, *, label: str) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PostfreezeBlindV8Error(f"{label} is not UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise PostfreezeBlindV8Error(f"{label} contains blank line {line_number}")
        try:
            value = json.loads(
                line,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant: {value}")
                ),
                object_pairs_hook=_unique_pairs,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PostfreezeBlindV8Error(f"{label} line {line_number} is invalid JSON") from exc
        if not isinstance(value, dict):
            raise PostfreezeBlindV8Error(f"{label} line {line_number} is not an object")
        rows.append(value)
    return rows


def _unique_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _capture(
    path: Path,
    *,
    label: str,
    maximum_bytes: int = MAX_JSON_BYTES,
) -> lifecycle_bindings_v7.StableFileSnapshot:
    try:
        return lifecycle_bindings_v7.capture_file(
            Path(path),
            label=label,
            maximum_bytes=maximum_bytes,
            reject_reserved=False,
        )
    except (
        lifecycle_bindings_v7.LifecycleBindingV7Error,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise PostfreezeBlindV8Error(f"{label} snapshot rejected: {exc}") from exc


def _verify_snapshot(
    snapshot: lifecycle_bindings_v7.StableFileSnapshot,
    *,
    label: str,
) -> None:
    try:
        lifecycle_bindings_v7.verify_file_unchanged(snapshot, label=label)
    except (
        lifecycle_bindings_v7.LifecycleBindingV7Error,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise PostfreezeBlindV8Error(f"{label} changed: {exc}") from exc


def _real_directory(
    path: Path,
    *,
    label: str,
    create: bool = False,
) -> tuple[Path, tuple[int, int, int, int, int]]:
    lexical = Path(path).absolute()
    if create:
        lexical.mkdir(parents=True, exist_ok=True)
    try:
        resolved = lifecycle_bindings_v7._assert_no_reparse_chain(
            lexical,
            label=label,
        ).resolve(strict=True)
        metadata = os.lstat(resolved)
    except (
        lifecycle_bindings_v7.LifecycleBindingV7Error,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise PostfreezeBlindV8Error(f"{label} directory rejected: {exc}") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or lifecycle_bindings_v7._is_reparse(metadata)
    ):
        raise PostfreezeBlindV8Error(f"{label} must be a real directory")
    identity = (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )
    return resolved, identity


def _verify_directory(
    path: Path,
    identity: tuple[int, int, int, int, int],
    *,
    label: str,
) -> None:
    current, observed = _real_directory(path, label=label)
    # Directory mtime/ctime/size legitimately change when this protocol
    # publishes an O_EXCL child.  Device and inode identify the anchored real
    # directory; the no-reparse-chain check above rejects path substitution.
    if current != path or observed[:2] != identity[:2]:
        raise PostfreezeBlindV8Error(f"{label} identity changed")


def _exclusive_create(path: Path, payload: bytes) -> dict[str, Any]:
    parent, identity = _real_directory(path.parent, label=f"{path.name} parent")
    if path.parent.resolve(strict=True) != parent:
        raise PostfreezeBlindV8Error("exclusive output parent changed")
    descriptor = os.open(
        parent / path.name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if os.path.lexists(parent / path.name):
            os.unlink(parent / path.name)
        raise
    _verify_directory(parent, identity, label=f"{path.name} parent after write")
    snapshot = _capture(parent / path.name, label=f"published {path.name}")
    if snapshot.payload != payload:
        raise PostfreezeBlindV8Error(f"published {path.name} bytes changed")
    return snapshot.receipt()


def _canonical_receipt(
    snapshot: lifecycle_bindings_v7.StableFileSnapshot,
    *,
    label: str,
    schema: str | None = None,
    version: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    receipt = _parse_json(snapshot, label=label)
    body = dict(receipt)
    observed = _sha(
        body.pop("canonical_digest_sha256", None),
        label=f"{label} canonical digest",
    )
    if canonical_sha256(body) != observed:
        raise PostfreezeBlindV8Error(f"{label} canonical digest mismatch")
    if schema is not None and receipt.get("schema") != schema:
        raise PostfreezeBlindV8Error(f"{label} schema mismatch")
    if version is not None and receipt.get("version") != version:
        raise PostfreezeBlindV8Error(f"{label} version mismatch")
    if status is not None and receipt.get("status") != status:
        raise PostfreezeBlindV8Error(f"{label} status mismatch")
    return receipt


def _artifact(
    path: Path,
    payload: bytes,
    *,
    records: int | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": str(path.resolve(strict=False)),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }
    if records is not None:
        value["records"] = records
    return value


def _descriptor_without_records(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(value["path"]),
        "bytes": int(value["bytes"]),
        "sha256": str(value["sha256"]),
    }


def _universe_id(commitment_sha256: str) -> str:
    return "icmat-v8-" + _sha(commitment_sha256, label="commitment SHA")[:32]


def _registry_paths(registry: Path, *, commitment_sha256: str) -> dict[str, Path]:
    universe = _universe_id(commitment_sha256)
    paths = {
        "authorization": registry / f"{universe}.authorization.v8.json",
        "claim": registry / f"{universe}.claim.v8.json",
        "terminal": registry / f"{universe}.terminal.v8.json",
        "evidence": registry / f"{universe}.evidence.v8",
    }
    if any(path.parent != registry for path in paths.values()):
        raise PostfreezeBlindV8Error("registry path escaped fixed root")
    return paths


def _production_registry_root(*, create: bool) -> Path:
    root, _ = _real_directory(
        PRODUCTION_REGISTRY_ROOT,
        label="fixed native-v8 registry",
        create=create,
    )
    workspace, _ = _real_directory(WORKSPACE_ROOT, label="workspace")
    try:
        root.relative_to(workspace)
    except ValueError as exc:
        raise PostfreezeBlindV8Error("fixed registry escaped workspace") from exc
    expected = PRODUCTION_REGISTRY_ROOT.resolve(strict=True)
    if root != expected:
        raise PostfreezeBlindV8Error("fixed registry identity changed")
    return root


def _source_paths(
    *,
    licensed_chunks_path: Path,
    rag_manifest_path: Path,
    semantic_inventory_path: Path,
) -> dict[str, Path]:
    inventory = Path(semantic_inventory_path)
    return {
        "licensed_chunks": Path(licensed_chunks_path),
        "rag_manifest": Path(rag_manifest_path),
        "semantic_inventory": inventory,
        "semantic_records": inventory.with_name("records.v7.jsonl"),
        "semantic_requests": inventory.with_name("requests.v7.jsonl"),
        "semantic_request_manifest": inventory.with_name("request_manifest.v7.json"),
        "nonblind_v8_module": Path(nonblind_sft_v8.__file__),
        "nonblind_v7_module": Path(nonblind_sft_v7.__file__),
        "evidence_core": Path(evidence_sft_v6.__file__),
        "semantic_core": Path(semantic_queries_v7.__file__),
    }


def _capture_sources(
    *,
    licensed_chunks_path: Path,
    rag_manifest_path: Path,
    semantic_inventory_path: Path,
) -> dict[str, lifecycle_bindings_v7.StableFileSnapshot]:
    paths = _source_paths(
        licensed_chunks_path=licensed_chunks_path,
        rag_manifest_path=rag_manifest_path,
        semantic_inventory_path=semantic_inventory_path,
    )
    return {
        role: _capture(
            paths[role],
            label=f"native-v8 source {role}",
            maximum_bytes=MAX_SOURCE_BYTES,
        )
        for role in _SOURCE_ROLES
    }


def _source_receipt(
    snapshots: Mapping[str, lifecycle_bindings_v7.StableFileSnapshot],
) -> dict[str, Any]:
    if tuple(snapshots) != _SOURCE_ROLES:
        raise PostfreezeBlindV8Error("source role ordering mismatch")
    files = {role: snapshots[role].receipt() for role in _SOURCE_ROLES}
    return {
        "roles": list(_SOURCE_ROLES),
        "files": files,
        "content_set_sha256": canonical_sha256(
            {
                role: {
                    "bytes": snapshots[role].bytes,
                    "sha256": snapshots[role].sha256,
                }
                for role in _SOURCE_ROLES
            }
        ),
    }


def _validate_commitment_and_manifest_v8(
    *,
    dataset_dir: Path,
    preblind_commitment_path: Path,
    sources: Mapping[str, lifecycle_bindings_v7.StableFileSnapshot],
    nli_model_dir: Path,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    dataset, dataset_identity = _real_directory(dataset_dir, label="strict v8 dataset")
    expected_commitment_path = dataset / selection_freeze_v8.COMMITMENT_NAME
    commitment_snapshot = _capture(
        preblind_commitment_path,
        label="native-v8 preblind commitment",
    )
    if commitment_snapshot.path != expected_commitment_path:
        raise PostfreezeBlindV8Error("preblind commitment is not the frozen v8 dataset artifact")
    commitment = _parse_json(
        commitment_snapshot,
        label="native-v8 preblind commitment",
    )
    _mapping(
        commitment,
        label="native-v8 preblind commitment",
        exact={
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
        },
    )
    commitment_body = dict(commitment)
    commitment_digest = _sha(
        commitment_body.pop("commitment_sha256", None),
        label="native-v8 preblind commitment digest",
    )
    if (
        commitment.get("schema") != nonblind_sft_v8.PREBLIND_COMMITMENT_SCHEMA
        or canonical_sha256(commitment_body) != commitment_digest
    ):
        raise PostfreezeBlindV8Error("native-v8 preblind commitment identity or digest mismatch")
    try:
        nonblind_sft_v8._assert_preblind_commitment_sanitized(commitment)
    except (evidence_sft_v6.EvidenceSFTV6Error, ValueError) as exc:
        raise PostfreezeBlindV8Error("v8 preblind commitment is not sanitized") from exc
    if (
        commitment.get("status") != "PREBLIND_COMMITTED_NONBLIND_ONLY"
        or commitment.get("builder_version") != nonblind_sft_v8.NONBLIND_BUILDER_VERSION
        or commitment.get("core_builder_version") != evidence_sft_v6.BUILDER_VERSION
        or commitment.get("split_algorithm_version") != nonblind_sft_v8.SPLIT_ALGORITHM_VERSION
        or commitment.get("repair_policy_version") != nonblind_sft_v8.NLI_REPAIR_POLICY_VERSION
        or commitment.get("expected_blind_count") != EXPECTED_ROWS
        or commitment.get("thresholds")
        != {
            "target_entailment_min": nonblind_sft_v8.TARGET_ENTAILMENT_MIN,
            "distractor_entailment_max": nonblind_sft_v8.DISTRACTOR_ENTAILMENT_MAX,
        }
    ):
        raise PostfreezeBlindV8Error("v8 preblind commitment identity mismatch")
    seed = commitment.get("seed")
    if (
        not isinstance(seed, str)
        or not seed
        or commitment.get("seed_sha256") != sha256_bytes(seed.encode("utf-8"))
    ):
        raise PostfreezeBlindV8Error("v8 committed seed is invalid")
    expected_sources = _mapping(
        commitment.get("source_inputs"),
        label="v8 commitment source inputs",
        exact={
            "licensed_chunks",
            "rag_manifest",
            "semantic_inventory",
            "semantic_records",
            "semantic_requests",
            "semantic_request_manifest",
        },
    )
    expected_code = _mapping(
        commitment.get("builder_code"),
        label="v8 commitment builder code",
        exact={
            "nonblind_v8_module",
            "nonblind_v7_module",
            "evidence_core",
            "semantic_core",
        },
    )
    for role, expected in {**expected_sources, **expected_code}.items():
        if sources[role].sha256 != _sha(expected, label=f"v8 commitment {role} SHA"):
            raise PostfreezeBlindV8Error(f"v8 committed source changed: {role}")

    nli_declared = _mapping(
        commitment.get("nli_model"),
        label="v8 commitment NLI model",
    )
    expected_tree = _sha(
        nli_declared.get("model_tree_sha256"),
        label="v8 commitment NLI tree",
    )
    try:
        nli_validated = semantic_queries_v7.validate_pinned_nli_asset(
            Path(nli_model_dir),
            expected_tree_sha256=expected_tree,
        )
        nli_provenance = nonblind_sft_v8._validate_nli_provenance(
            nli_declared,
            expected_tree_sha256=expected_tree,
        )
    except Exception as exc:
        raise PostfreezeBlindV8Error(f"fixed local NLI asset rejected: {exc}") from exc
    if nli_validated.get("model_tree_sha256") != expected_tree:
        raise PostfreezeBlindV8Error("fixed local NLI tree changed")

    manifest_snapshot = _capture(
        dataset / selection_freeze_v8.MANIFEST_NAME,
        label="strict v8 manifest",
    )
    manifest = _parse_json(manifest_snapshot, label="strict v8 manifest")
    selection_manifest = _mapping(
        selection.get("manifest"),
        label="selection v8 manifest",
    )
    if (
        manifest.get("schema") != selection_freeze_v8.MANIFEST_SCHEMA
        or manifest.get("builder_version") != selection_freeze_v8.BUILDER_VERSION
        or manifest.get("dataset_schema") != selection_freeze_v8.DATASET_SCHEMA
        or manifest.get("status") != "NONBLIND_V8_BUILT_NLI_UNIQUE_SUPPORT_PREBLIND_COMMITTED"
        or selection_manifest.get("sha256") != manifest_snapshot.sha256
        or selection_manifest.get("bytes") != manifest_snapshot.bytes
    ):
        raise PostfreezeBlindV8Error("strict v8 manifest binding mismatch")
    selection_preblind = _mapping(
        selection.get("preblind_commitment"),
        label="selection v8 preblind commitment",
    )
    if (
        selection_preblind.get("sha256") != commitment_snapshot.sha256
        or selection_preblind.get("bytes") != commitment_snapshot.bytes
        or selection_preblind.get("commitment_sha256") != commitment.get("commitment_sha256")
    ):
        raise PostfreezeBlindV8Error("selection/preblind v8 binding mismatch")
    manifest_sources = _mapping(
        manifest.get("source_inputs"),
        label="v8 manifest source inputs",
        exact=set(expected_sources),
    )
    for role in expected_sources:
        descriptor = _mapping(
            manifest_sources.get(role),
            label=f"v8 manifest source {role}",
            exact={"path", "sha256"},
        )
        try:
            recorded = Path(str(descriptor["path"])).resolve(strict=True)
        except OSError as exc:
            raise PostfreezeBlindV8Error(f"v8 manifest source path unavailable: {role}") from exc
        if recorded != sources[role].path or descriptor["sha256"] != sources[role].sha256:
            raise PostfreezeBlindV8Error(f"v8 manifest source binding mismatch: {role}")
    code = _mapping(
        _mapping(manifest.get("builder"), label="v8 manifest builder").get("code"),
        label="v8 manifest builder code",
        exact=set(expected_code),
    )
    for role in expected_code:
        descriptor = _mapping(
            code.get(role),
            label=f"v8 manifest code {role}",
            exact={"path", "sha256"},
        )
        if (
            Path(str(descriptor["path"])).resolve(strict=True) != sources[role].path
            or descriptor["sha256"] != sources[role].sha256
        ):
            raise PostfreezeBlindV8Error(f"v8 manifest code binding mismatch: {role}")
    artifact = _mapping(
        _mapping(manifest.get("artifacts"), label="v8 manifest artifacts").get("preblind_commitment"),
        label="v8 manifest commitment artifact",
        exact={"path", "bytes", "sha256"},
    )
    if artifact != {
        "path": selection_freeze_v8.COMMITMENT_NAME,
        "bytes": commitment_snapshot.bytes,
        "sha256": commitment_snapshot.sha256,
    }:
        raise PostfreezeBlindV8Error("v8 manifest commitment artifact mismatch")
    rag_manifest = _parse_json(sources["rag_manifest"], label="v8 RAG manifest")
    if rag_manifest.get("manifest_id") != commitment.get("rag_manifest_id"):
        raise PostfreezeBlindV8Error("v8 RAG manifest ID differs from commitment")
    _verify_directory(dataset, dataset_identity, label="strict v8 dataset final")
    return {
        "dataset": dataset,
        "dataset_identity": dataset_identity,
        "manifest_snapshot": manifest_snapshot,
        "manifest": manifest,
        "commitment_snapshot": commitment_snapshot,
        "commitment": commitment,
        "commitment_sha256": str(commitment["commitment_sha256"]),
        "seed": seed,
        "nli_model_dir": str(Path(nli_model_dir).resolve(strict=True)),
        "nli_model": dict(nli_provenance),
    }


def _load_selection_v8(
    *,
    selection_freeze_path: Path,
    evaluation_index_path: Path,
    training_receipt_path: Path,
    dataset_dir: Path,
    base_model_dir: Path,
    adapter_dir: Path,
) -> dict[str, Any]:
    selection_snapshot = _capture(
        selection_freeze_path,
        label="strict v8 selection freeze",
    )
    selection = _canonical_receipt(
        selection_snapshot,
        label="strict v8 selection freeze",
        schema=selection_freeze_v8.SCHEMA,
        version=selection_freeze_v8.VERSION,
        status=selection_freeze_v8.STATUS,
    )
    try:
        verified = selection_freeze_v8.verify_selection_freeze_v8(
            freeze_receipt_path=selection_snapshot.path,
            evaluation_index_path=Path(evaluation_index_path),
            training_receipt_path=Path(training_receipt_path),
            dataset_dir=Path(dataset_dir),
            base_model_dir=Path(base_model_dir),
        )
        model_trees = gguf_release_v8._tree_bindings(
            base_model_dir=Path(base_model_dir),
            selected_adapter_dir=Path(adapter_dir),
        )
    except (
        gguf_release_v8.GgufReleaseV8Error,
        selection_freeze_v8.SelectionFreezeV8Error,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise PostfreezeBlindV8Error(f"strict v8 selection rejected: {exc}") from exc
    if (
        verified.get("status") != selection_freeze_v8.VERIFIED_STATUS
        or verified.get("selection_locked") is not True
        or verified.get("blind_test_authorized") is not False
        or verified.get("gguf_export_authorized") is not False
        or verified.get("deployment_authorized") is not False
    ):
        raise PostfreezeBlindV8Error("strict v8 selection boundary is unsafe")
    selected = _mapping(selection.get("selection"), label="strict v8 selected checkpoint")
    base = _mapping(selection.get("base_model"), label="strict v8 base model")
    try:
        frozen_adapter = Path(str(selected["checkpoint_path"])).resolve(strict=True)
        supplied_adapter = Path(adapter_dir).resolve(strict=True)
        frozen_base = Path(str(base["path"])).resolve(strict=True)
        supplied_base = Path(base_model_dir).resolve(strict=True)
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        raise PostfreezeBlindV8Error("strict v8 model paths are unavailable") from exc
    if (
        supplied_adapter != frozen_adapter
        or supplied_base != frozen_base
        or model_trees["checkpoint_tree_sha256"] != selected.get("checkpoint_tree_sha256")
        or model_trees["adapter_tree_sha256"] != selected.get("adapter_tree_sha256")
        or model_trees["base_model_tree_sha256"] != base.get("tree_sha256")
    ):
        raise PostfreezeBlindV8Error("runtime model trees differ from strict v8 selection")
    chain_binding = {
        "selection_freeze_sha256": selection_snapshot.sha256,
        "selection_binding_digest_sha256": _sha(
            selection.get("selection_binding_digest_sha256"),
            label="selection binding digest",
        ),
        "manifest_sha256": _sha(
            _mapping(selection.get("manifest"), label="selection manifest").get("sha256"),
            label="selection manifest SHA",
        ),
        "preblind_commitment_file_sha256": _sha(
            _mapping(
                selection.get("preblind_commitment"),
                label="selection preblind commitment",
            ).get("sha256"),
            label="selection preblind file SHA",
        ),
        "preblind_commitment_sha256": _sha(
            _mapping(
                selection.get("preblind_commitment"),
                label="selection preblind commitment",
            ).get("commitment_sha256"),
            label="selection preblind commitment digest",
        ),
        "base_model_tree_sha256": model_trees["base_model_tree_sha256"],
        "checkpoint_id": selected.get("checkpoint_id"),
        "checkpoint_tree_sha256": model_trees["checkpoint_tree_sha256"],
        "adapter_tree_sha256": model_trees["adapter_tree_sha256"],
    }
    if not isinstance(chain_binding["checkpoint_id"], str) or not chain_binding["checkpoint_id"]:
        raise PostfreezeBlindV8Error("selected v8 checkpoint ID is invalid")
    return {
        "selection_snapshot": selection_snapshot,
        "selection": selection,
        "verification": dict(verified),
        "chain_binding": chain_binding,
        "model_trees": model_trees,
    }


def _generation_from_record(
    value: Any,
    *,
    label: str,
) -> pointer_hf_eval_v6.GenerationResultV6:
    generation = _mapping(
        value,
        label=label,
        exact={
            "raw_pointer",
            "raw_pointer_sha256",
            "finish_reason",
            "finish_category",
            "trusted_finish_reason",
            "latency_ms",
            "input_tokens",
            "output_tokens",
            "generation_error",
        },
    )
    raw_pointer = generation.get("raw_pointer")
    latency = generation.get("latency_ms")
    if (
        not isinstance(raw_pointer, str)
        or generation.get("raw_pointer_sha256") != sha256_bytes(raw_pointer.encode("utf-8"))
        or not isinstance(generation.get("finish_reason"), str)
        or not isinstance(generation.get("finish_category"), str)
        or not isinstance(generation.get("trusted_finish_reason"), bool)
        or isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or not math.isfinite(float(latency))
        or float(latency) < 0.0
        or generation.get("generation_error") is not None
    ):
        raise PostfreezeBlindV8Error(f"{label} is invalid")
    for field in ("input_tokens", "output_tokens"):
        token_count = generation.get(field)
        if token_count is not None and (
            isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0
        ):
            raise PostfreezeBlindV8Error(f"{label}.{field} is invalid")
    return pointer_hf_eval_v6.GenerationResultV6(
        raw_pointer=raw_pointer,
        finish_reason=str(generation["finish_reason"]),
        finish_category=str(generation["finish_category"]),
        latency_ms=float(latency),
        input_tokens=generation.get("input_tokens"),
        output_tokens=generation.get("output_tokens"),
        generation_error=None,
    )


def _implementation_runner_path(
    implementation: Mapping[str, Any],
    *,
    label: str,
) -> Path | None:
    runner = implementation.get("runner")
    if runner is None:
        return None
    record = _mapping(runner, label=f"{label} runner")
    path = record.get("path")
    if not isinstance(path, str) or not path:
        raise PostfreezeBlindV8Error(f"{label} runner path is invalid")
    return Path(path)


def _capture_artifact_directory(
    directory: Path,
    *,
    expected_names: set[str],
    label: str,
) -> tuple[
    Path,
    tuple[int, int, int, int, int],
    dict[str, lifecycle_bindings_v7.StableFileSnapshot],
]:
    root, identity = _real_directory(directory, label=label)
    names: set[str] = set()
    with os.scandir(root) as entries:
        for entry in entries:
            if not entry.is_file(follow_symlinks=False) or entry.is_symlink():
                raise PostfreezeBlindV8Error(f"{label} contains non-regular artifact")
            names.add(entry.name)
    if names != expected_names:
        raise PostfreezeBlindV8Error(f"{label} exact file inventory mismatch")
    snapshots = {name: _capture(root / name, label=f"{label} {name}") for name in sorted(expected_names)}
    return root, identity, snapshots


def _verify_artifact_descriptors(
    receipt: Mapping[str, Any],
    snapshots: Mapping[str, lifecycle_bindings_v7.StableFileSnapshot],
    *,
    receipt_filename: str,
    label: str,
) -> None:
    artifacts = _mapping(receipt.get("artifacts"), label=f"{label} artifacts")
    expected = set(snapshots) - {receipt_filename}
    if set(artifacts) != expected:
        raise PostfreezeBlindV8Error(f"{label} artifact whitelist mismatch")
    for name in expected:
        record = _mapping(artifacts[name], label=f"{label} artifact {name}")
        if record.get("bytes") != snapshots[name].bytes or record.get("sha256") != snapshots[name].sha256:
            raise PostfreezeBlindV8Error(f"{label} artifact bytes changed: {name}")


def _backend_tree_binding(
    backend: Mapping[str, Any],
    *,
    chain: Mapping[str, Any],
    selected_checkpoint_path: Path,
    base_model_path: Path,
    adapter_required: bool,
    label: str,
) -> None:
    del chain
    if (
        backend.get("mode") != "hf_model"
        or backend.get("device") not in {"cpu", "cuda"}
        or backend.get("local_files_only") is not True
        or backend.get("network_allowed") is not False
    ):
        raise PostfreezeBlindV8Error(f"{label} backend boundary mismatch")
    model = _mapping(backend.get("model"), label=f"{label} model")
    base = _mapping(model.get("base"), label=f"{label} base model")
    adapter_value = model.get("adapter")
    try:
        current_base = pointer_hf_eval_v6._tree_inventory(base_model_path)
    except pointer_hf_eval_v6.PointerHFEvalV6Error as exc:
        raise PostfreezeBlindV8Error(f"{label} base tree could not be rebuilt") from exc
    if dict(base) != current_base:
        raise PostfreezeBlindV8Error(f"{label} complete base tree mismatch")
    if adapter_required:
        adapter = _mapping(adapter_value, label=f"{label} adapter")
        try:
            current_adapter = pointer_hf_eval_v6._tree_inventory(selected_checkpoint_path)
        except pointer_hf_eval_v6.PointerHFEvalV6Error as exc:
            raise PostfreezeBlindV8Error(f"{label} adapter tree could not be rebuilt") from exc
        if dict(adapter) != current_adapter:
            raise PostfreezeBlindV8Error(f"{label} complete adapter tree mismatch")
    elif adapter_value is not None:
        raise PostfreezeBlindV8Error(f"{label} unexpectedly loaded an adapter")


def _verify_calibration_gate_v8(
    directory: Path,
    *,
    selection_args: Mapping[str, Path],
    chain: Mapping[str, Any],
) -> dict[str, Any]:
    root, root_identity, snapshots = _capture_artifact_directory(
        directory,
        expected_names=set(calibration_eval_v8.EXPECTED_ARTIFACT_NAMES),
        label="native-v8 calibration",
    )
    receipt_snapshot = snapshots[calibration_eval_v8.RECEIPT_FILENAME]
    receipt = _canonical_receipt(
        receipt_snapshot,
        label="native-v8 calibration receipt",
        schema=calibration_eval_v8.RECEIPT_SCHEMA,
        version=calibration_eval_v8.VERSION,
        status=gguf_release_v8.CALIBRATION_STATUS,
    )
    try:
        authority = calibration_eval_v8._capture_precalibration_authority_v8(**selection_args)
        declarations = calibration_eval_v8._manifest_split_declarations(authority.manifest)
        training_dataset = calibration_eval_v8._training_dataset(
            authority.training,
            manifest_sha256=authority.manifest_file.sha256,
            gate_sha256=selection_freeze_v8.PINNED_GATE_BUNDLE_R3_SHA256,
        )
        training_splits = calibration_eval_v8._mapping(
            training_dataset.get("splits"),
            label="native-v8 calibration training splits",
            exact={"train", "validation", "calibration"},
        )
        calibration = calibration_eval_v8._capture_split_v8(
            authority.dataset_root,
            split="calibration",
            declaration=declarations["calibration"],
            training_summary=calibration_eval_v8._mapping(
                training_splits["calibration"],
                label="native-v8 calibration split summary",
            ),
        )
    except (
        calibration_eval_v8.CalibrationEvalV8Error,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise PostfreezeBlindV8Error(f"native-v8 calibration authority rejected: {exc}") from exc
    if (
        chain["selection_freeze_sha256"] != authority.selection_file.sha256
        or chain["manifest_sha256"] != authority.manifest_file.sha256
        or receipt.get("quality_gate_passed") is not True
        or receipt.get("selection_locked") is not True
        or receipt.get("checkpoint_reselection_performed") is not False
        or receipt.get("authorization")
        != {
            "blind_test_authorized": False,
            "gguf_export_authorized": False,
            "x5_execution_authorized": False,
            "deployment_authorized": False,
            "production_integration_authorized": False,
        }
    ):
        raise PostfreezeBlindV8Error("native-v8 calibration gate boundary mismatch")
    dataset = _mapping(receipt.get("dataset"), label="native-v8 calibration dataset")
    if (
        dataset.get("split") != "calibration"
        or dataset.get("complete_split") is not True
        or dataset.get("rows") != EXPECTED_ROWS
        or dataset.get("max_samples") is not None
        or dataset.get("file") != calibration.receipt()
        or dataset.get("train") != authority.train.receipt()
        or dataset.get("validation") != authority.validation.receipt()
        or dataset.get("id_sets_pairwise_disjoint") is not True
    ):
        raise PostfreezeBlindV8Error("native-v8 calibration split binding mismatch")
    model = _mapping(receipt.get("model"), label="native-v8 calibration model")
    if model.get("model_bound") is not True or model.get("fixture_not_model_evidence") is not False:
        raise PostfreezeBlindV8Error("native-v8 calibration is not model-bound evidence")
    backend = _mapping(receipt.get("backend"), label="native-v8 calibration backend")
    _backend_tree_binding(
        backend,
        chain=chain,
        selected_checkpoint_path=Path(str(authority.selection["selection"]["checkpoint_path"])),
        base_model_path=Path(str(authority.selection["base_model"]["path"])),
        adapter_required=True,
        label="native-v8 calibration",
    )
    _verify_artifact_descriptors(
        receipt,
        snapshots,
        receipt_filename=calibration_eval_v8.RECEIPT_FILENAME,
        label="native-v8 calibration",
    )
    samples = _parse_jsonl(
        snapshots[calibration_eval_v8.SAMPLE_FILENAME].payload,
        label="native-v8 calibration samples",
    )
    summary = _parse_json(
        snapshots[calibration_eval_v8.SUMMARY_FILENAME],
        label="native-v8 calibration summary",
    )
    implementation = _mapping(
        receipt.get("implementation"),
        label="native-v8 calibration implementation",
    )
    try:
        _, current_implementation = calibration_eval_v8._source_snapshots(
            _implementation_runner_path(
                implementation,
                label="native-v8 calibration implementation",
            )
        )
        bindings = calibration_eval_v8._sample_bindings(
            authority,
            calibration,
            implementation=implementation,
            backend=backend,
        )
        source_rows = [
            pointer_hf_eval_v6._score_row(
                row=row,
                generation=_generation_from_record(
                    recorded.get("generation"),
                    label=f"calibration generation {row.example_id}",
                ),
                bindings=bindings,
                backend_mode="hf_model",
            )
            for row, recorded in zip(calibration.rows, samples, strict=True)
        ]
        recomputed_samples, recomputed_summary = calibration_eval_v8._v8_results(
            source_rows,
            backend_mode="hf_model",
            model_bound=True,
            authority=authority,
            calibration=calibration,
            implementation=implementation,
        )
    except (
        calibration_eval_v8.CalibrationEvalV8Error,
        pointer_hf_eval_v6.PointerHFEvalV6Error,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise PostfreezeBlindV8Error(
            f"native-v8 calibration deterministic recomputation failed: {exc}"
        ) from exc
    if (
        dict(implementation) != current_implementation
        or samples != recomputed_samples
        or summary != recomputed_summary
        or len(samples) != EXPECTED_ROWS
        or receipt.get("conformal_threshold")
        != _mapping(summary.get("conformal"), label="calibration conformal").get("threshold")
    ):
        raise PostfreezeBlindV8Error("native-v8 calibration evidence differs from recomputation")
    for snapshot in snapshots.values():
        _verify_snapshot(snapshot, label=f"native-v8 calibration {snapshot.path.name}")
    _verify_directory(root, root_identity, label="native-v8 calibration final")
    return {
        "directory": str(root),
        "receipt": receipt_snapshot.receipt(),
        "status": receipt["status"],
        "rows": EXPECTED_ROWS,
        "quality_gate_passed": True,
        "conformal_threshold": receipt["conformal_threshold"],
        "samples_recomputed": True,
        "summary_recomputed": True,
        "artifact_snapshots": {name: snapshot.receipt() for name, snapshot in snapshots.items()},
    }


def _strip_ablation_v8_row(row: Mapping[str, Any]) -> dict[str, Any]:
    stripped = json.loads(canonical_json(dict(row)))
    for field in (
        "ablation_version",
        "strict_v8_authority_sha256",
        "v6_math_implementation_sha256",
    ):
        if field not in stripped:
            raise PostfreezeBlindV8Error(f"ablation sample misses v8 field: {field}")
        stripped.pop(field)
    boundaries = dict(_mapping(stripped.get("boundaries"), label="v8 ablation sample boundaries"))
    for field in ("fixture_not_model_evidence", "calibration_accessed", "blind_accessed"):
        if field not in boundaries:
            raise PostfreezeBlindV8Error(f"ablation boundary missing: {field}")
        boundaries.pop(field)
    stripped["boundaries"] = boundaries
    stripped["schema"] = ablation_eval_v6.SAMPLE_SCHEMA
    return stripped


def _verify_ablation_gate_v8(
    directory: Path,
    *,
    selection_args: Mapping[str, Path],
    chain: Mapping[str, Any],
) -> dict[str, Any]:
    root, root_identity, snapshots = _capture_artifact_directory(
        directory,
        expected_names=set(ablation_eval_v8.EXPECTED_ARTIFACT_NAMES),
        label="native-v8 ablation",
    )
    receipt_snapshot = snapshots[ablation_eval_v8.RECEIPT_FILENAME]
    receipt = _canonical_receipt(
        receipt_snapshot,
        label="native-v8 ablation receipt",
        schema=ablation_eval_v8.RECEIPT_SCHEMA,
        version=ablation_eval_v8.VERSION,
        status=gguf_release_v8.ABLATION_STATUS,
    )
    try:
        authority = ablation_eval_v8._capture_authority_v8(**selection_args)
    except (
        ablation_eval_v8.AblationEvalV8Error,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise PostfreezeBlindV8Error(f"native-v8 ablation authority rejected: {exc}") from exc
    binding = authority["binding"]
    if (
        binding["selection_freeze_sha256"] != chain["selection_freeze_sha256"]
        or binding["manifest_sha256"] != chain["manifest_sha256"]
        or binding["selected_checkpoint_id"] != chain["checkpoint_id"]
        or binding["selected_checkpoint_tree_sha256"] != chain["checkpoint_tree_sha256"]
        or binding["selected_adapter_tree_sha256"] != chain["adapter_tree_sha256"]
        or receipt.get("invariants_passed") is not True
        or any(
            value is not False
            for value in _mapping(
                receipt.get("authorization"),
                label="native-v8 ablation authorization",
            ).values()
        )
    ):
        raise PostfreezeBlindV8Error("native-v8 ablation chain or boundary mismatch")
    dataset = _mapping(receipt.get("dataset"), label="native-v8 ablation dataset")
    execution = _mapping(receipt.get("execution"), label="native-v8 ablation execution")
    model = _mapping(receipt.get("model"), label="native-v8 ablation model")
    if (
        dataset.get("split") != "validation"
        or dataset.get("complete_split") is not True
        or dataset.get("rows") != EXPECTED_ROWS
        or dataset.get("max_samples") is not None
        or dataset.get("sha256") != binding["validation_sha256"]
        or execution.get("sample_rows") != ablation_eval_v8.EXPECTED_SAMPLE_ROWS
        or execution.get("selection_policy_called") is not False
        or execution.get("automatic_model_selection") is not False
        or execution.get("checkpoint_reselection_performed") is not False
        or model.get("model_bound") is not True
        or model.get("fixture_not_model_evidence") is not False
    ):
        raise PostfreezeBlindV8Error("native-v8 ablation is not full model-bound validation")
    _verify_artifact_descriptors(
        receipt,
        snapshots,
        receipt_filename=ablation_eval_v8.RECEIPT_FILENAME,
        label="native-v8 ablation",
    )
    samples = _parse_jsonl(
        snapshots[ablation_eval_v8.SAMPLE_FILENAME].payload,
        label="native-v8 ablation samples",
    )
    if len(samples) != ablation_eval_v8.EXPECTED_SAMPLE_ROWS:
        raise PostfreezeBlindV8Error("native-v8 ablation sample matrix is incomplete")
    implementation = _mapping(
        receipt.get("implementation"),
        label="native-v8 ablation implementation",
    )
    try:
        _, current_implementation = ablation_eval_v8._source_snapshots(
            _implementation_runner_path(
                implementation,
                label="native-v8 ablation implementation",
            )
        )
        recomputed_v6 = ablation_eval_v6._recompute_sample_rows(
            recorded_rows=[_strip_ablation_v8_row(row) for row in samples],
            dataset_rows=authority["rows"],
        )
        recomputed_samples = json.loads(canonical_json(recomputed_v6))
        for row in recomputed_samples:
            row["schema"] = ablation_eval_v8.SAMPLE_SCHEMA
            row["ablation_version"] = ablation_eval_v8.VERSION
            row["strict_v8_authority_sha256"] = authority["binding_digest_sha256"]
            row["v6_math_implementation_sha256"] = implementation["ablation_math_v6"]["sha256"]
            row["boundaries"] = {
                **row["boundaries"],
                "fixture_not_model_evidence": False,
                "calibration_accessed": False,
                "blind_accessed": False,
            }
        recomputed_samples.sort(
            key=lambda row: (
                ablation_eval_v6.SUBJECTS.index(str(row["subject"])),
                ablation_eval_v6.ALL_VARIANTS.index(str(row["variant"])),
                str(row["example_id"]),
            )
        )
        reports, invariants = ablation_eval_v8._v8_reports(
            recomputed_samples,
            backend_mode="hf_model",
            authority_digest=authority["binding_digest_sha256"],
            implementation=implementation,
        )
    except (
        ablation_eval_v6.AblationEvalV6Error,
        ablation_eval_v8.AblationEvalV8Error,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise PostfreezeBlindV8Error(f"native-v8 ablation deterministic recomputation failed: {exc}") from exc
    recorded_reports = {
        name: _parse_json(snapshots[name], label=f"native-v8 ablation report {name}")
        for name in sorted(ablation_eval_v8.REPORT_FILENAMES)
    }
    stable_backends = _mapping(
        receipt.get("backend_bindings"),
        label="native-v8 ablation backend bindings",
        exact=set(ablation_eval_v6.SUBJECTS),
    )
    for subject in ablation_eval_v6.SUBJECTS:
        backend = _mapping(stable_backends[subject], label=f"ablation backend {subject}")
        if (
            backend.get("mode") != "hf_model"
            or backend.get("device") not in {"cpu", "cuda"}
            or backend.get("local_files_only") is not True
            or backend.get("network_allowed") is not False
        ):
            raise PostfreezeBlindV8Error(f"ablation backend boundary mismatch: {subject}")
        _backend_tree_binding(
            backend,
            chain=chain,
            selected_checkpoint_path=Path(binding["selected_checkpoint_path"]),
            base_model_path=Path(binding["base_model_path"]),
            adapter_required=subject == "adapter",
            label=f"native-v8 ablation {subject}",
        )
    if (
        dict(implementation) != current_implementation
        or samples != recomputed_samples
        or recorded_reports != reports
        or invariants is not True
    ):
        raise PostfreezeBlindV8Error("native-v8 ablation evidence differs from recomputation")
    cases, _ = ablation_eval_v6._build_cases(authority["rows"])
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
    artifact_records = _mapping(receipt.get("artifacts"), label="ablation artifacts")
    reproducibility = {
        "version": ablation_eval_v8.VERSION,
        "strict_v8_authority_sha256": authority["binding_digest_sha256"],
        "manifest_sha256": binding["manifest_sha256"],
        "train_sha256": binding["train_sha256"],
        "validation_sha256": binding["validation_sha256"],
        "training_gate_bundle_sha256": binding["training_gate_bundle_sha256"],
        "training_receipt_sha256": binding["training_receipt_sha256"],
        "evaluation_receipt_sha256": binding["evaluation_receipt_sha256"],
        "selection_freeze_sha256": binding["selection_freeze_sha256"],
        "selected_checkpoint_id": binding["selected_checkpoint_id"],
        "validation_rows": ablation_eval_v8.EXPECTED_VALIDATION_ROWS,
        "sample_rows": ablation_eval_v8.EXPECTED_SAMPLE_ROWS,
        "backend_mode": "hf_model",
        "seed": ablation_eval_v8.FIXED_SEED,
        "request_digest_sha256": request_digest,
        "same_requests_for_base_and_adapter": True,
        "implementation": dict(implementation),
        "backend_bindings": dict(stable_backends),
        "artifacts": dict(artifact_records),
    }
    if execution.get("request_digest_sha256") != request_digest or receipt.get(
        "reproducibility_payload_sha256"
    ) != lifecycle_bindings_v7.canonical_sha256(reproducibility):
        raise PostfreezeBlindV8Error("native-v8 ablation reproducibility mismatch")
    for snapshot in snapshots.values():
        _verify_snapshot(snapshot, label=f"native-v8 ablation {snapshot.path.name}")
    _verify_directory(root, root_identity, label="native-v8 ablation final")
    return {
        "directory": str(root),
        "receipt": receipt_snapshot.receipt(),
        "status": receipt["status"],
        "validation_rows": EXPECTED_ROWS,
        "sample_rows": ablation_eval_v8.EXPECTED_SAMPLE_ROWS,
        "samples_recomputed": True,
        "reports_recomputed": True,
        "artifact_snapshots": {name: snapshot.receipt() for name, snapshot in snapshots.items()},
    }


def _verify_postselection_gates_v8(
    *,
    calibration_dir: Path,
    ablation_dir: Path,
    selection_args: Mapping[str, Path],
    chain: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "calibration": _verify_calibration_gate_v8(
            calibration_dir,
            selection_args=selection_args,
            chain=chain,
        ),
        "ablation": _verify_ablation_gate_v8(
            ablation_dir,
            selection_args=selection_args,
            chain=chain,
        ),
    }


def _capture_implementation(execute_runner_path: Path) -> dict[str, Any]:
    paths = {
        "protocol_v8": Path(__file__),
        "execute_runner_v8": Path(execute_runner_path),
        "selection_freeze_v8": Path(selection_freeze_v8.__file__),
        "calibration_v8": Path(calibration_eval_v8.__file__),
        "ablation_v8": Path(ablation_eval_v8.__file__),
        "gguf_preflight_v8": Path(gguf_release_v8.__file__),
        "pointer_evaluator_v6": Path(pointer_hf_eval_v6.__file__),
        "pointer_compiler_v6": Path(evidence_pointer_v6.__file__),
        "nonblind_builder_v8": Path(nonblind_sft_v8.__file__),
    }
    snapshots = {
        role: _capture(path, label=f"native-v8 implementation {role}", maximum_bytes=8 * 1024 * 1024)
        for role, path in paths.items()
    }
    return {
        "snapshots": snapshots,
        "receipt": {role: snapshot.receipt() for role, snapshot in snapshots.items()},
    }


def _validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise PostfreezeBlindV8Error("run_id is invalid")
    return run_id


def prepare_postfreeze_blind_v8(
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
    calibration_dir: Path,
    ablation_dir: Path,
    nli_model_dir: Path,
    execute_runner_path: Path,
    run_id: str,
) -> dict[str, Any]:
    """Prepare the sole native-v8 authorization without deriving blind rows."""

    _validate_run_id(run_id)
    registry = _production_registry_root(create=True)
    registry_identity = _real_directory(registry, label="fixed native-v8 registry")[1]
    selection_args = {
        "selection_freeze_path": Path(selection_freeze_path),
        "evaluation_index_path": Path(evaluation_index_path),
        "training_receipt_path": Path(training_receipt_path),
        "dataset_dir": Path(dataset_dir),
        "base_model_dir": Path(base_model_dir),
    }
    upstream = _load_selection_v8(
        **selection_args,
        adapter_dir=Path(adapter_dir),
    )
    sources = _capture_sources(
        licensed_chunks_path=licensed_chunks_path,
        rag_manifest_path=rag_manifest_path,
        semantic_inventory_path=semantic_inventory_path,
    )
    committed = _validate_commitment_and_manifest_v8(
        dataset_dir=dataset_dir,
        preblind_commitment_path=preblind_commitment_path,
        sources=sources,
        nli_model_dir=nli_model_dir,
        selection=upstream["selection"],
    )
    chain = upstream["chain_binding"]
    if (
        chain["preblind_commitment_sha256"] != committed["commitment_sha256"]
        or chain["preblind_commitment_file_sha256"] != committed["commitment_snapshot"].sha256
        or chain["manifest_sha256"] != committed["manifest_snapshot"].sha256
    ):
        raise PostfreezeBlindV8Error("selection chain differs from v8 commitment")
    postselection = _verify_postselection_gates_v8(
        calibration_dir=calibration_dir,
        ablation_dir=ablation_dir,
        selection_args=selection_args,
        chain=chain,
    )
    implementation = _capture_implementation(execute_runner_path)
    for snapshot in (*sources.values(), *implementation["snapshots"].values()):
        _verify_snapshot(snapshot, label=f"prepare final {snapshot.path.name}")
    _verify_directory(registry, registry_identity, label="fixed registry prepare final")

    paths = _registry_paths(registry, commitment_sha256=committed["commitment_sha256"])
    if any(os.path.lexists(path) for path in paths.values()):
        raise PostfreezeBlindV8Error("this v8 commitment already has one-shot state")
    calibration_receipt = postselection["calibration"]["receipt"]
    ablation_receipt = postselection["ablation"]["receipt"]
    authorization_core = {
        "schema": AUTHORIZATION_SCHEMA,
        "version": PROTOCOL_VERSION,
        "status": AUTHORIZATION_STATUS,
        "authorization_id": "icmat-v8-postfreeze-"
        + canonical_sha256(
            {
                "chain": chain,
                "commitment": committed["commitment_sha256"],
                "run_id": run_id,
            }
        )[:32],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "chain_binding": dict(chain),
        "upstream_receipts": {
            "selection_freeze": upstream["selection_snapshot"].receipt(),
            "calibration": dict(calibration_receipt),
            "ablation": dict(ablation_receipt),
        },
        "dataset": {
            "directory": str(committed["dataset"]),
            "manifest": committed["manifest_snapshot"].receipt(),
            "preblind_commitment": committed["commitment_snapshot"].receipt(),
            "preblind_commitment_sha256": committed["commitment_sha256"],
            "seed": committed["seed"],
            "expected_rows": EXPECTED_ROWS,
            "expected_families": EXPECTED_FAMILIES,
            "examples_per_family": EXAMPLES_PER_FAMILY,
        },
        "sources": _source_receipt(sources),
        "nli_model": {
            "path": committed["nli_model_dir"],
            "provenance": committed["nli_model"],
        },
        "model": {
            **upstream["model_trees"],
            "checkpoint_id": chain["checkpoint_id"],
            "adapter_runtime_path": str(Path(adapter_dir).resolve(strict=True)),
            "full_tree_required": True,
        },
        "postselection": postselection,
        "implementation": implementation["receipt"],
        "execution": {
            "backend": "hf_model",
            "device": "cuda",
            "seed": FIXED_SEED,
            "rows": EXPECTED_ROWS,
            "single_hf_call_required": True,
            "resume_allowed": False,
            "retry_after_claim_allowed": False,
            "local_files_only": True,
            "network_allowed": False,
            "x5_access_allowed": False,
        },
        "release_policy": json.loads(canonical_json(RELEASE_POLICY)),
        "registry": {
            "root": str(registry),
            "authorization_path": str(paths["authorization"]),
            "claim_path": str(paths["claim"]),
            "terminal_path": str(paths["terminal"]),
            "evidence_path": str(paths["evidence"]),
        },
        "claim_boundary": {
            "claim_key_basis": "preblind_commitment_sha256_only",
            "claim_must_precede_split_assignment": True,
            "claim_must_precede_example_build": True,
            "claim_must_precede_test_member_derivation": True,
            "failure_or_crash_is_non_reusable": True,
        },
        "security_boundary": {
            "honest_local_execution_environment_required": True,
            "cryptographic_secrecy": False,
            "administrator_forgery_resistant": False,
            "tpm_or_external_signature_verified": False,
        },
        "authorization": {
            **_FALSE_AUTHORIZATION,
            "model_selection_authorized": False,
            "checkpoint_ranking_authorized": False,
            "threshold_tuning_authorized": False,
            "calibration_authorized": False,
            "blind_execution_authorized_once": True,
            "gguf_export_authorized_before_pass": False,
        },
    }
    authorization = {
        **authorization_core,
        "canonical_digest_sha256": canonical_sha256(authorization_core),
    }
    try:
        publication = _exclusive_create(paths["authorization"], _json_bytes(authorization))
    except FileExistsError as exc:
        raise PostfreezeBlindV8Error("v8 authorization already exists") from exc
    return {
        "status": "POSTFREEZE_V8_AUTHORIZATION_PREPARED_NOT_CLAIMED",
        "authorization_id": authorization["authorization_id"],
        "authorization": publication,
        "universe_id": _universe_id(committed["commitment_sha256"]),
        "claim_created": False,
        "split_assignment_called": False,
        "test_members_derived": False,
        "retry_policy": "NON_REUSABLE_AFTER_CLAIM",
    }


def _load_authorization(
    authorization_path: Path,
) -> tuple[lifecycle_bindings_v7.StableFileSnapshot, dict[str, Any], dict[str, Path]]:
    registry = _production_registry_root(create=False)
    snapshot = _capture(authorization_path, label="native-v8 authorization")
    authorization = _canonical_receipt(
        snapshot,
        label="native-v8 authorization",
        schema=AUTHORIZATION_SCHEMA,
        version=PROTOCOL_VERSION,
        status=AUTHORIZATION_STATUS,
    )
    _mapping(
        authorization,
        label="native-v8 authorization",
        exact=_AUTHORIZATION_FIELDS,
    )
    chain = _mapping(
        authorization.get("chain_binding"),
        label="native-v8 authorization chain",
        exact={
            "selection_freeze_sha256",
            "selection_binding_digest_sha256",
            "manifest_sha256",
            "preblind_commitment_file_sha256",
            "preblind_commitment_sha256",
            "base_model_tree_sha256",
            "checkpoint_id",
            "checkpoint_tree_sha256",
            "adapter_tree_sha256",
        },
    )
    for field in (
        "selection_freeze_sha256",
        "selection_binding_digest_sha256",
        "manifest_sha256",
        "preblind_commitment_file_sha256",
        "preblind_commitment_sha256",
        "base_model_tree_sha256",
        "checkpoint_tree_sha256",
        "adapter_tree_sha256",
    ):
        _sha(chain.get(field), label=f"authorization chain {field}")
    if (
        not isinstance(chain.get("checkpoint_id"), str)
        or not chain["checkpoint_id"]
        or authorization.get("release_policy") != RELEASE_POLICY
        or authorization.get("execution")
        != {
            "backend": "hf_model",
            "device": "cuda",
            "seed": FIXED_SEED,
            "rows": EXPECTED_ROWS,
            "single_hf_call_required": True,
            "resume_allowed": False,
            "retry_after_claim_allowed": False,
            "local_files_only": True,
            "network_allowed": False,
            "x5_access_allowed": False,
        }
        or authorization.get("claim_boundary")
        != {
            "claim_key_basis": "preblind_commitment_sha256_only",
            "claim_must_precede_split_assignment": True,
            "claim_must_precede_example_build": True,
            "claim_must_precede_test_member_derivation": True,
            "failure_or_crash_is_non_reusable": True,
        }
        or authorization.get("security_boundary")
        != {
            "honest_local_execution_environment_required": True,
            "cryptographic_secrecy": False,
            "administrator_forgery_resistant": False,
            "tpm_or_external_signature_verified": False,
        }
        or authorization.get("authorization")
        != {
            **_FALSE_AUTHORIZATION,
            "model_selection_authorized": False,
            "checkpoint_ranking_authorized": False,
            "threshold_tuning_authorized": False,
            "calibration_authorized": False,
            "blind_execution_authorized_once": True,
            "gguf_export_authorized_before_pass": False,
        }
    ):
        raise PostfreezeBlindV8Error("native-v8 authorization boundary mismatch")
    _validate_run_id(str(authorization.get("run_id")))
    dataset = _mapping(authorization.get("dataset"), label="authorization dataset")
    commitment_sha = _sha(
        dataset.get("preblind_commitment_sha256"),
        label="authorization commitment SHA",
    )
    paths = _registry_paths(registry, commitment_sha256=commitment_sha)
    if snapshot.path != paths["authorization"]:
        raise PostfreezeBlindV8Error("authorization is not in the fixed registry")
    recorded_registry = _mapping(
        authorization.get("registry"),
        label="authorization registry",
    )
    expected_registry = {
        "root": str(registry),
        "authorization_path": str(paths["authorization"]),
        "claim_path": str(paths["claim"]),
        "terminal_path": str(paths["terminal"]),
        "evidence_path": str(paths["evidence"]),
    }
    if dict(recorded_registry) != expected_registry:
        raise PostfreezeBlindV8Error("authorization registry path binding mismatch")
    authorization["_authorization_sha256"] = snapshot.sha256
    return snapshot, authorization, paths


def _path_from_receipt(value: Mapping[str, Any], *, label: str) -> Path:
    try:
        path = Path(str(value["path"])).resolve(strict=True)
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        raise PostfreezeBlindV8Error(f"{label} path is unavailable") from exc
    return path


def _reverify_authorization_inputs(
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    upstream_receipts = _mapping(
        authorization.get("upstream_receipts"),
        label="authorization upstream receipts",
        exact={"selection_freeze", "calibration", "ablation"},
    )
    dataset_record = _mapping(authorization.get("dataset"), label="authorization dataset")
    sources_record = _mapping(authorization.get("sources"), label="authorization sources")
    source_files = _mapping(
        sources_record.get("files"),
        label="authorization source files",
        exact=set(_SOURCE_ROLES),
    )
    selection_path = _path_from_receipt(
        _mapping(upstream_receipts["selection_freeze"], label="selection receipt"),
        label="selection receipt",
    )
    calibration_path = _path_from_receipt(
        _mapping(upstream_receipts["calibration"], label="calibration receipt"),
        label="calibration receipt",
    )
    ablation_path = _path_from_receipt(
        _mapping(upstream_receipts["ablation"], label="ablation receipt"),
        label="ablation receipt",
    )
    dataset_dir = Path(str(dataset_record["directory"])).resolve(strict=True)
    model = _mapping(authorization.get("model"), label="authorization model")
    selection_args = {
        "selection_freeze_path": selection_path,
        "evaluation_index_path": Path(
            str(
                _mapping(
                    _mapping(
                        _parse_json(
                            _capture(selection_path, label="selection recheck"),
                            label="selection recheck",
                        ).get("evaluation_receipt"),
                        label="selection evaluation receipt",
                    ),
                    label="selection evaluation receipt",
                ).get("path")
            )
        ),
        "training_receipt_path": Path(
            str(
                _mapping(
                    _parse_json(
                        _capture(selection_path, label="selection training recheck"),
                        label="selection training recheck",
                    ).get("training_receipt"),
                    label="selection training receipt",
                ).get("path")
            )
        ),
        "dataset_dir": dataset_dir,
        "base_model_dir": Path(str(model["base_path"])),
    }
    upstream = _load_selection_v8(
        **selection_args,
        adapter_dir=Path(str(model["checkpoint_path"])),
    )
    if upstream["chain_binding"] != authorization.get("chain_binding"):
        raise PostfreezeBlindV8Error("authorization v8 chain changed")
    source_paths = {
        role: _path_from_receipt(
            _mapping(source_files[role], label=f"authorization source {role}"),
            label=f"authorization source {role}",
        )
        for role in _SOURCE_ROLES
    }
    sources = _capture_sources(
        licensed_chunks_path=source_paths["licensed_chunks"],
        rag_manifest_path=source_paths["rag_manifest"],
        semantic_inventory_path=source_paths["semantic_inventory"],
    )
    if _source_receipt(sources) != sources_record:
        raise PostfreezeBlindV8Error("authorization source set changed")
    nli_model = _mapping(authorization.get("nli_model"), label="authorization NLI model")
    committed = _validate_commitment_and_manifest_v8(
        dataset_dir=dataset_dir,
        preblind_commitment_path=Path(str(dataset_record["preblind_commitment"]["path"])),
        sources=sources,
        nli_model_dir=Path(str(nli_model["path"])),
        selection=upstream["selection"],
    )
    calibration_dir = calibration_path.parent
    ablation_dir = ablation_path.parent
    postselection = _verify_postselection_gates_v8(
        calibration_dir=calibration_dir,
        ablation_dir=ablation_dir,
        selection_args=selection_args,
        chain=upstream["chain_binding"],
    )
    if (
        postselection["calibration"]["receipt"] != upstream_receipts["calibration"]
        or postselection["ablation"]["receipt"] != upstream_receipts["ablation"]
        or committed["commitment_sha256"] != dataset_record["preblind_commitment_sha256"]
    ):
        raise PostfreezeBlindV8Error("authorization postselection or commitment changed")
    implementation = _mapping(
        authorization.get("implementation"),
        label="authorization implementation",
    )
    execute_runner = _path_from_receipt(
        _mapping(
            implementation.get("execute_runner_v8"),
            label="authorization execute runner",
        ),
        label="authorization execute runner",
    )
    current = _capture_implementation(execute_runner)
    if current["receipt"] != implementation:
        raise PostfreezeBlindV8Error("native-v8 implementation changed")
    current_trees = gguf_release_v8._tree_bindings(
        base_model_dir=Path(str(model["base_path"])),
        selected_adapter_dir=Path(str(model["checkpoint_path"])),
    )
    for key in (
        "base_model_tree_sha256",
        "checkpoint_tree_sha256",
        "adapter_tree_sha256",
    ):
        if current_trees[key] != model[key]:
            raise PostfreezeBlindV8Error(f"authorization model tree changed: {key}")
    return {
        "upstream": upstream,
        "sources": sources,
        "committed": committed,
        "postselection": postselection,
        "implementation": current,
        "selection_args": selection_args,
    }


def _require_cuda_ready() -> None:
    try:
        import torch
    except ImportError as exc:
        raise PostfreezeBlindV8Error("PyTorch is unavailable for the one-shot CUDA run") from exc
    if not torch.cuda.is_available():
        raise PostfreezeBlindV8Error("CUDA is unavailable for the one-shot v8 run")


def _claim_once(
    *,
    claim_path: Path,
    authorization_snapshot: lifecycle_bindings_v7.StableFileSnapshot,
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    if os.path.lexists(claim_path):
        raise PostfreezeBlindV8Error("native-v8 commitment was already claimed")
    nonce = uuid.uuid4().hex
    body = {
        "schema": CLAIM_SCHEMA,
        "version": PROTOCOL_VERSION,
        "status": "CLAIMED_NON_REUSABLE",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "authorization_sha256": authorization_snapshot.sha256,
        "authorization_id": authorization["authorization_id"],
        "preblind_commitment_sha256": authorization["dataset"]["preblind_commitment_sha256"],
        "nonce_sha256": sha256_bytes(nonce.encode("utf-8")),
        "failure_is_non_reusable": True,
        "retry_allowed": False,
    }
    claim = {**body, "canonical_digest_sha256": canonical_sha256(body)}
    try:
        publication = _exclusive_create(claim_path, _json_bytes(claim))
    except FileExistsError as exc:
        raise PostfreezeBlindV8Error("native-v8 commitment was concurrently claimed") from exc
    snapshot = _capture(claim_path, label="persisted native-v8 claim")
    return {
        "path": claim_path,
        "snapshot": snapshot,
        "receipt": claim,
        "publication": publication,
    }


def _validate_blind_shape(examples: Sequence[Mapping[str, Any]]) -> None:
    if len(examples) != EXPECTED_ROWS:
        raise PostfreezeBlindV8Error("native-v8 blind derivation must contain 150 rows")
    ids = [row.get("example_id") for row in examples]
    families = Counter(str(row.get("family_id")) for row in examples)
    decisions = Counter(str(row.get("decision")) for row in examples)
    if (
        len(set(ids)) != EXPECTED_ROWS
        or any(not isinstance(value, str) or not value for value in ids)
        or set(families.values()) != {EXAMPLES_PER_FAMILY}
        or len(families) != EXPECTED_FAMILIES
        or decisions != {"ANSWER": EXPECTED_ANSWER_ROWS, "REFUSE": EXPECTED_REFUSE_ROWS}
        or any(row.get("split") != "blind_test" for row in examples)
    ):
        raise PostfreezeBlindV8Error("native-v8 blind balance or membership mismatch")
    for example in examples:
        try:
            evidence_sft_v6.validate_example(example)
        except evidence_sft_v6.EvidenceSFTV6Error as exc:
            raise PostfreezeBlindV8Error("native-v8 blind row failed validation") from exc


def _audit_and_rebuild_blind_answers(
    examples: Sequence[Mapping[str, Any]],
    *,
    families: Sequence[evidence_sft_v6.SourceFamily],
    auditor: semantic_queries_v7.NLIAuditor,
    nli_provenance: Mapping[str, Any],
    seed: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    families_by_id = {family.source_id: family for family in families}
    cache: dict[tuple[str, str], semantic_queries_v7.NLIResult] = {}
    rebuilt: list[dict[str, Any]] = []
    target_values: list[float] = []
    non_target_values: list[float] = []
    repair_count = 0
    answer_count = 0
    for source in examples:
        example = json.loads(canonical_json(dict(source)))
        if example.get("decision") != "ANSWER":
            rebuilt.append(example)
            continue
        answer_count += 1
        evidence = _sequence(
            example.get("compiler_evidence"),
            label="blind ANSWER compiler evidence",
        )
        if len(evidence) != 2:
            raise PostfreezeBlindV8Error("blind ANSWER must contain two evidence blocks")
        target_span_id = example.get("target_span_id")
        if not isinstance(target_span_id, str):
            raise PostfreezeBlindV8Error("blind ANSWER target span is invalid")
        target_evidence_id = target_span_id.split(".", 1)[0]
        target_indices = [
            index
            for index, block in enumerate(evidence)
            if isinstance(block, Mapping) and block.get("evidence_id") == target_evidence_id
        ]
        if len(target_indices) != 1:
            raise PostfreezeBlindV8Error("blind ANSWER target passage is not unique")
        target_index = target_indices[0]
        distractor_index = 1 - target_index
        target_rows = nonblind_sft_v8._span_rows(evidence[target_index])
        distractor_rows = nonblind_sft_v8._span_rows(evidence[distractor_index])
        target_matches = [text for span_id, text in target_rows if span_id == target_span_id]
        if len(target_matches) != 1:
            raise PostfreezeBlindV8Error("blind ANSWER target span is unavailable")
        claim = str(example["requested_claim"])
        target_result, _ = nonblind_sft_v8._score_pair(
            auditor,
            cache,
            premise=target_matches[0],
            hypothesis=claim,
        )
        if target_result.entailment < nonblind_sft_v8.TARGET_ENTAILMENT_MIN:
            raise PostfreezeBlindV8Error("blind target entailment is below fixed v8 threshold")
        target_values.append(float(target_result.entailment))
        for span_id, text in target_rows:
            if span_id == target_span_id:
                continue
            result, _ = nonblind_sft_v8._score_pair(
                auditor,
                cache,
                premise=text,
                hypothesis=claim,
            )
            if result.entailment > nonblind_sft_v8.DISTRACTOR_ENTAILMENT_MAX:
                raise PostfreezeBlindV8Error("blind target-passage neighbor exceeds fixed v8 threshold")
            non_target_values.append(float(result.entailment))
        _, original_max = nonblind_sft_v8._passage_scores(
            auditor,
            cache,
            sentences=[text for _, text in distractor_rows],
            claim=claim,
        )
        selected_max = original_max
        if original_max > nonblind_sft_v8.DISTRACTOR_ENTAILMENT_MAX:
            family = families_by_id.get(str(example.get("source_id")))
            if family is None:
                raise PostfreezeBlindV8Error("blind ANSWER source family is unavailable")
            target_passage = tuple(text for _, text in target_rows)
            ranked = nonblind_sft_v8._deduplicated_ranked_candidates(
                family=family,
                target_passage=target_passage,
                target_text=target_matches[0],
                query=claim,
                seed=(f"{seed}:{example['example_id']}:v8-nli-qualified-distractor"),
            )
            replacement = None
            for candidate in ranked:
                _, maximum = nonblind_sft_v8._passage_scores(
                    auditor,
                    cache,
                    sentences=candidate.passage_sentences,
                    claim=claim,
                )
                if maximum <= nonblind_sft_v8.DISTRACTOR_ENTAILMENT_MAX:
                    replacement = candidate
                    selected_max = maximum
                    break
            if replacement is None:
                raise PostfreezeBlindV8Error("no same-family blind distractor satisfies fixed v8 threshold")
            example = nonblind_sft_v8._replace_distractor_passage(
                example,
                distractor_index=distractor_index,
                candidate=replacement,
            )
            repair_count += 1
        if selected_max > nonblind_sft_v8.DISTRACTOR_ENTAILMENT_MAX:
            raise PostfreezeBlindV8Error("blind distractor exceeds fixed v8 threshold")
        non_target_values.append(float(selected_max))
        try:
            evidence_sft_v6.validate_example(example)
        except evidence_sft_v6.EvidenceSFTV6Error as exc:
            raise PostfreezeBlindV8Error("rebuilt blind ANSWER is invalid") from exc
        rebuilt.append(example)
    if answer_count != EXPECTED_ANSWER_ROWS:
        raise PostfreezeBlindV8Error("blind NLI audit did not cover all 75 ANSWER rows")
    rebuilt.sort(key=lambda row: str(row["example_id"]))
    return rebuilt, {
        "schema": "icmat_postfreeze_blind_unique_support_audit.v8",
        "status": "PASS_ALL_75_BLIND_ANSWERS_HAVE_UNIQUE_NLI_SUPPORT",
        "policy_version": nonblind_sft_v8.NLI_REPAIR_POLICY_VERSION,
        "nli_provenance": dict(nli_provenance),
        "answer_count": answer_count,
        "repair_count": repair_count,
        "minimum_target_entailment": min(target_values),
        "maximum_non_target_entailment": max(non_target_values, default=0.0),
        "thresholds": {
            "target_entailment_min": nonblind_sft_v8.TARGET_ENTAILMENT_MIN,
            "distractor_entailment_max": nonblind_sft_v8.DISTRACTOR_ENTAILMENT_MAX,
        },
    }


def _derive_blind_rows_after_claim(
    *,
    authorization: Mapping[str, Any],
    claim: Mapping[str, Any],
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[pointer_hf_eval_v6.DatasetRowV6, ...],
    bytes,
    dict[str, Any],
]:
    del claim
    sources = _mapping(authorization.get("sources"), label="authorization sources")
    source_files = _mapping(sources.get("files"), label="authorization source files")
    paths = {
        role: _path_from_receipt(
            _mapping(source_files[role], label=f"source {role}"),
            label=f"source {role}",
        )
        for role in _SOURCE_ROLES
    }
    dataset = _mapping(authorization.get("dataset"), label="authorization dataset")
    try:
        families = evidence_sft_v6.load_licensed_families(paths["licensed_chunks"])
        semantic_inventory, semantic_audit = evidence_sft_v6.load_semantic_inventory(
            paths["semantic_inventory"],
            families,
        )
        families = evidence_sft_v6.augment_families_with_semantic_candidates(
            families,
            semantic_inventory,
        )
        assignments = evidence_sft_v6.assign_family_splits(
            families,
            seed=str(dataset["seed"]),
        )
        raw_examples = evidence_sft_v6.build_examples(
            families,
            assignments,
            semantic_inventory,
            seed=str(dataset["seed"]),
            examples_per_family=EXAMPLES_PER_FAMILY,
            included_splits=("blind_test",),
        )
        nli = _mapping(authorization.get("nli_model"), label="authorization NLI model")
        provenance = _mapping(nli.get("provenance"), label="authorization NLI provenance")
        auditor = nonblind_sft_v8._create_nli_auditor(
            model_dir=Path(str(nli["path"])),
            expected_tree_sha256=str(provenance["model_tree_sha256"]),
            device=str(provenance["device"]),
        )
        if getattr(auditor, "formal_backend", False) is not True:
            raise PostfreezeBlindV8Error("blind derivation requires fixed formal NLI backend")
        validated_provenance = nonblind_sft_v8._validate_nli_provenance(
            auditor.provenance,
            expected_tree_sha256=str(provenance["model_tree_sha256"]),
        )
        if validated_provenance != provenance:
            raise PostfreezeBlindV8Error("blind NLI runtime provenance changed")
        examples, nli_audit = _audit_and_rebuild_blind_answers(
            raw_examples,
            families=families,
            auditor=auditor,
            nli_provenance=validated_provenance,
            seed=str(dataset["seed"]),
        )
    except PostfreezeBlindV8Error:
        raise
    except (
        evidence_sft_v6.EvidenceSFTV6Error,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise PostfreezeBlindV8Error(f"native-v8 blind derivation failed: {exc}") from exc
    _validate_blind_shape(examples)
    if (
        semantic_audit.get("semantic_inventory_sha256") != source_files["semantic_inventory"]["sha256"]
        or semantic_audit.get("semantic_records_sha256") != source_files["semantic_records"]["sha256"]
    ):
        raise PostfreezeBlindV8Error("blind semantic source audit differs from commitment")
    rows: list[pointer_hf_eval_v6.DatasetRowV6] = []
    for line_number, example in enumerate(examples, 1):
        try:
            rows.append(
                pointer_hf_eval_v6._validate_dataset_row(
                    example,
                    split="blind_test",
                    line_number=line_number,
                )
            )
        except pointer_hf_eval_v6.PointerHFEvalV6Error as exc:
            raise PostfreezeBlindV8Error("blind evaluator row is invalid") from exc
    payload = _jsonl_bytes(examples)
    derivation = {
        "algorithm": nonblind_sft_v8.SPLIT_ALGORITHM_VERSION,
        "builder_version": nonblind_sft_v8.NONBLIND_BUILDER_VERSION,
        "repair_policy_version": nonblind_sft_v8.NLI_REPAIR_POLICY_VERSION,
        "seed_sha256": sha256_bytes(str(dataset["seed"]).encode("utf-8")),
        "rows": EXPECTED_ROWS,
        "families": EXPECTED_FAMILIES,
        "examples_per_family": EXAMPLES_PER_FAMILY,
        "answer_rows": EXPECTED_ANSWER_ROWS,
        "refuse_rows": EXPECTED_REFUSE_ROWS,
        "nli_unique_support": nli_audit,
        "same_in_memory_rows_used_for_evaluation": True,
        "materialization_reopened_for_evaluation": False,
    }
    return tuple(examples), tuple(rows), payload, derivation


@contextmanager
def _offline_hf_environment() -> Any:
    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
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
    chain = _mapping(authorization.get("chain_binding"), label="authorization chain")
    implementation = _mapping(
        authorization.get("implementation"),
        label="authorization implementation",
    )
    return {
        "authorization_sha256": authorization["_authorization_sha256"],
        "claim_sha256": claim_sha256,
        "preblind_commitment_sha256": chain["preblind_commitment_sha256"],
        "derived_test_sha256": derived_sha256,
        "selection_freeze_sha256": chain["selection_freeze_sha256"],
        "selection_binding_digest_sha256": chain["selection_binding_digest_sha256"],
        "base_model_tree_sha256": chain["base_model_tree_sha256"],
        "checkpoint_id": chain["checkpoint_id"],
        "checkpoint_tree_sha256": chain["checkpoint_tree_sha256"],
        "adapter_tree_sha256": chain["adapter_tree_sha256"],
        "protocol_source_sha256": implementation["protocol_v8"]["sha256"],
        "evaluator_source_sha256": implementation["pointer_evaluator_v6"]["sha256"],
        "compiler_source_sha256": implementation["pointer_compiler_v6"]["sha256"],
    }


def _evaluate_hf_cuda_once(
    rows: Sequence[pointer_hf_eval_v6.DatasetRowV6],
    *,
    authorization: Mapping[str, Any],
    claim: Mapping[str, Any],
    derived_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(rows) != EXPECTED_ROWS:
        raise PostfreezeBlindV8Error("HF v8 execution requires all 150 rows")
    requests = pointer_hf_eval_v6._generation_requests(rows)
    if len(requests) != EXPECTED_ROWS or len({item.example_id for item in requests}) != EXPECTED_ROWS:
        raise PostfreezeBlindV8Error("HF v8 request membership is incomplete")
    model = _mapping(authorization.get("model"), label="authorization model")
    with _offline_hf_environment():
        try:
            generations, backend = pointer_hf_eval_v6.generate_hf_model(
                requests,
                base_model_dir=Path(str(model["base_path"])),
                adapter_dir=Path(str(model["checkpoint_path"])),
                device="cuda",
                seed=FIXED_SEED,
            )
        except pointer_hf_eval_v6.PointerHFEvalV6Error as exc:
            raise PostfreezeBlindV8Error("sole native-v8 HF CUDA call failed") from exc
    if (
        set(generations) != {row.example_id for row in rows}
        or any(result.generation_error is not None for result in generations.values())
        or backend.get("mode") != "hf_model"
        or backend.get("device") != "cuda"
        or backend.get("seed") != FIXED_SEED
        or backend.get("samples_generated") != EXPECTED_ROWS
    ):
        raise PostfreezeBlindV8Error("native-v8 HF backend result is incomplete")
    _backend_tree_binding(
        backend,
        chain=authorization["chain_binding"],
        selected_checkpoint_path=Path(str(model["checkpoint_path"])),
        base_model_path=Path(str(model["base_path"])),
        adapter_required=True,
        label="native-v8 one-shot backend",
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
            raise PostfreezeBlindV8Error(
                f"native-v8 post-generation scoring failed: {row.example_id}"
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


def _recompute_samples(
    *,
    rows: Sequence[pointer_hf_eval_v6.DatasetRowV6],
    recorded_samples: Sequence[Mapping[str, Any]],
    authorization: Mapping[str, Any],
    claim_sha256: str,
    derived_sha256: str,
) -> list[dict[str, Any]]:
    if len(rows) != EXPECTED_ROWS or len(recorded_samples) != EXPECTED_ROWS:
        raise PostfreezeBlindV8Error("v8 verifier requires all 150 samples")
    bindings = _sample_bindings(
        authorization=authorization,
        claim_sha256=claim_sha256,
        derived_sha256=derived_sha256,
    )
    policy = {
        "post_generation_scoring_only": True,
        "model_selection_performed": False,
        "checkpoint_ranking_performed": False,
        "threshold_tuning_performed": False,
        "calibration_performed": False,
    }
    recomputed: list[dict[str, Any]] = []
    for row, recorded in zip(rows, recorded_samples, strict=True):
        if recorded.get("example_id") != row.example_id:
            raise PostfreezeBlindV8Error("v8 sample order differs from re-derived rows")
        try:
            sample = pointer_hf_eval_v6._score_row(
                row=row,
                generation=_generation_from_record(
                    recorded.get("generation"),
                    label=f"postfreeze generation {row.example_id}",
                ),
                bindings=bindings,
                backend_mode="hf_model",
            )
        except pointer_hf_eval_v6.PointerHFEvalV6Error as exc:
            raise PostfreezeBlindV8Error(f"v8 sample recompilation failed: {row.example_id}") from exc
        sample["schema"] = SAMPLE_SCHEMA
        sample["postfreeze_protocol_version"] = PROTOCOL_VERSION
        sample["reserved_use_policy"] = dict(policy)
        if sample != dict(recorded):
            raise PostfreezeBlindV8Error(
                f"v8 sample differs from raw-pointer recompilation: {row.example_id}"
            )
        recomputed.append(sample)
    return recomputed


def _ratio_gate(
    *,
    name: str,
    numerator: int,
    denominator: int,
    minimum: Mapping[str, int],
) -> dict[str, Any]:
    required_numerator = int(minimum["numerator"])
    required_denominator = int(minimum["denominator"])
    return {
        "name": name,
        "actual": {"numerator": numerator, "denominator": denominator},
        "required_minimum": {
            "numerator": required_numerator,
            "denominator": required_denominator,
        },
        "passed": (denominator > 0 and numerator * required_denominator >= required_numerator * denominator),
    }


def _release_gates(
    samples: Sequence[Mapping[str, Any]],
    *,
    backend: Mapping[str, Any],
) -> list[dict[str, Any]]:
    answer_rows = [
        row
        for row in samples
        if _mapping(
            _mapping(row.get("expected"), label="sample expected").get("answer"),
            label="sample expected answer",
        ).get("decision")
        == "ANSWER"
    ]
    refuse_rows = [
        row
        for row in samples
        if _mapping(
            _mapping(row.get("expected"), label="sample expected").get("answer"),
            label="sample expected answer",
        ).get("decision")
        == "REFUSE"
    ]
    if len(answer_rows) != EXPECTED_ANSWER_ROWS or len(refuse_rows) != EXPECTED_REFUSE_ROWS:
        raise PostfreezeBlindV8Error("v8 expected decision balance changed")

    def count(field: str, name: str, rows: Sequence[Mapping[str, Any]] = samples) -> int:
        total = 0
        for row in rows:
            value = _mapping(row.get(field), label=f"sample {field}").get(name)
            if not isinstance(value, bool):
                raise PostfreezeBlindV8Error(f"sample {field}.{name} is not boolean")
            total += int(value)
        return total

    def predicted_decision(row: Mapping[str, Any]) -> str | None:
        if _mapping(row.get("pointer_metrics"), label="pointer metrics").get("compiler_accepted") is not True:
            return None
        parsed = _mapping(row.get("compilation"), label="sample compilation").get("parsed_pointer")
        return parsed.get("decision") if isinstance(parsed, Mapping) else None

    tp = sum(predicted_decision(row) == "REFUSE" for row in refuse_rows)
    fp = sum(predicted_decision(row) == "REFUSE" for row in answer_rows)
    fn = len(refuse_rows) - tp
    wrong = count("compiled_metrics", "unsupported_wrong_answer")
    return [
        {
            "name": "REAL_HF_CUDA_MODEL_BOUND",
            "actual_backend": backend.get("mode"),
            "actual_device": backend.get("device"),
            "passed": backend.get("mode") == "hf_model" and backend.get("device") == "cuda",
        },
        {
            "name": "COMPLETE_150",
            "actual": len(samples),
            "required": EXPECTED_ROWS,
            "passed": len(samples) == EXPECTED_ROWS,
        },
        _ratio_gate(
            name="POINTER_PARSE_VALID_100_PERCENT",
            numerator=count("pointer_metrics", "parse_valid"),
            denominator=len(samples),
            minimum=RELEASE_POLICY["pointer_parse_valid"],
        ),
        _ratio_gate(
            name="POINTER_COMPILER_ACCEPTED_100_PERCENT",
            numerator=count("pointer_metrics", "compiler_accepted"),
            denominator=len(samples),
            minimum=RELEASE_POLICY["pointer_compiler_accepted"],
        ),
        {
            "name": "UNSUPPORTED_WRONG_ANSWER_ZERO",
            "actual": wrong,
            "required_maximum": 0,
            "passed": wrong == 0,
        },
        _ratio_gate(
            name="COMPILED_SCHEMA_VALID_100_PERCENT",
            numerator=count("compiled_metrics", "schema_valid"),
            denominator=len(samples),
            minimum=RELEASE_POLICY["compiled_schema_valid"],
        ),
        _ratio_gate(
            name="COMPILED_CITATION_EXACT_100_PERCENT",
            numerator=count("compiled_metrics", "citation_exact"),
            denominator=len(samples),
            minimum=RELEASE_POLICY["compiled_citation_exact"],
        ),
        _ratio_gate(
            name="COMPILED_PROVENANCE_EXACT_100_PERCENT",
            numerator=count("compiled_metrics", "provenance_exact"),
            denominator=len(samples),
            minimum=RELEASE_POLICY["compiled_provenance_exact"],
        ),
        _ratio_gate(
            name="ANSWER_SPAN_EXACT_AT_LEAST_95_PERCENT",
            numerator=count("pointer_metrics", "span_exact", answer_rows),
            denominator=len(answer_rows),
            minimum=RELEASE_POLICY["answer_span_exact_minimum"],
        ),
        _ratio_gate(
            name="REFUSE_F1_AT_LEAST_95_PERCENT",
            numerator=2 * tp,
            denominator=2 * tp + fp + fn,
            minimum=RELEASE_POLICY["refuse_f1_minimum"],
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
    qualified = all(gate.get("passed") is True for gate in gate_results)
    return {
        "schema": SUMMARY_SCHEMA,
        "version": PROTOCOL_VERSION,
        "status": QUALIFICATION_PASS_STATUS if qualified else QUALIFICATION_HOLD_STATUS,
        "rows": len(samples),
        "complete_split": len(samples) == EXPECTED_ROWS,
        "backend": {
            "mode": backend.get("mode"),
            "device": backend.get("device"),
            "seed": backend.get("seed"),
            "samples_generated": backend.get("samples_generated"),
            "local_files_only": backend.get("local_files_only"),
            "network_allowed": backend.get("network_allowed"),
            "model": backend.get("model"),
        },
        "pointer_metrics": {
            name: _metric(samples, field="pointer_metrics", name=name)
            for name in ("parse_valid", "compiler_accepted", "span_exact", "strict_exact")
        },
        "compiled_metrics": {
            name: _metric(samples, field="compiled_metrics", name=name)
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
            "gguf_export_authorized": qualified,
            **_FALSE_AUTHORIZATION,
        },
        "execution_boundary": {
            "model_selection_performed": False,
            "checkpoint_reselection_performed": False,
            "training_performed": False,
            "calibration_performed": False,
            "retry_allowed": False,
        },
    }


def _build_run_receipt(
    *,
    authorization: Mapping[str, Any],
    claim: Mapping[str, Any],
    sample_artifact: Mapping[str, Any],
    summary_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    chain = dict(authorization["chain_binding"])
    body = {
        "schema": RUN_RECEIPT_SCHEMA,
        "version": PROTOCOL_VERSION,
        "status": RUN_COMPLETE_STATUS,
        "chain_binding": chain,
        "upstream_receipts": {
            "selection_freeze_sha256": chain["selection_freeze_sha256"],
            "calibration_sha256": authorization["upstream_receipts"]["calibration"]["sha256"],
            "ablation_sha256": authorization["upstream_receipts"]["ablation"]["sha256"],
        },
        "dataset": {
            "rows_read_once": EXPECTED_ROWS,
            "blind_sha256": authorization["_derived_sha256"],
            "nonblind_manifest_sha256": chain["manifest_sha256"],
            "preblind_commitment_sha256": chain["preblind_commitment_sha256"],
        },
        "execution_boundary": {
            "backend": "hf_model",
            "model_selection_performed": False,
            "checkpoint_reselection_performed": False,
            "training_performed": False,
            "calibration_performed": False,
        },
        "authorization": dict(_FALSE_AUTHORIZATION),
        "consumption_claim": {
            "sha256": claim["snapshot"].sha256,
            "nonce_sha256": claim["receipt"]["nonce_sha256"],
            "failure_is_non_reusable": True,
        },
        "artifacts": {
            "sample_results": _descriptor_without_records(sample_artifact),
            "summary": _descriptor_without_records(summary_artifact),
        },
    }
    return {**body, "canonical_digest_sha256": canonical_sha256(body)}


def _build_qualification(
    *,
    authorization: Mapping[str, Any],
    claim: Mapping[str, Any],
    run_receipt_artifact: Mapping[str, Any],
    gate_results: Sequence[Mapping[str, Any]],
    sample_artifact: Mapping[str, Any],
    summary_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    qualified = all(gate.get("passed") is True for gate in gate_results)
    status = QUALIFICATION_PASS_STATUS if qualified else QUALIFICATION_HOLD_STATUS
    chain = dict(authorization["chain_binding"])
    body = {
        "schema": QUALIFICATION_SCHEMA,
        "version": PROTOCOL_VERSION,
        "status": status,
        "qualified": qualified,
        "chain_binding": chain,
        "upstream_receipts": {
            "selection_freeze_sha256": chain["selection_freeze_sha256"],
            "calibration_sha256": authorization["upstream_receipts"]["calibration"]["sha256"],
            "ablation_sha256": authorization["upstream_receipts"]["ablation"]["sha256"],
            "postfreeze_sha256": run_receipt_artifact["sha256"],
        },
        "blind_run_receipt": {
            "sha256": run_receipt_artifact["sha256"],
            "schema": RUN_RECEIPT_SCHEMA,
            "status": RUN_COMPLETE_STATUS,
        },
        "consumption_claim": {
            "sha256": claim["snapshot"].sha256,
            "nonce_sha256": claim["receipt"]["nonce_sha256"],
            "failure_is_non_reusable": True,
        },
        "gate_results": [
            {"name": str(gate["name"]), "passed": bool(gate["passed"])} for gate in gate_results
        ],
        "release_authorization": {
            "gguf_export_authorized": qualified,
            **_FALSE_AUTHORIZATION,
        },
        "artifacts": {
            "sample_results": _descriptor_without_records(sample_artifact),
            "summary": _descriptor_without_records(summary_artifact),
        },
    }
    return {**body, "canonical_digest_sha256": canonical_sha256(body)}


def _write_terminal(
    *,
    path: Path,
    authorization_snapshot: lifecycle_bindings_v7.StableFileSnapshot,
    claim: Mapping[str, Any],
    status: str,
    error: BaseException | None,
) -> None:
    body = {
        "schema": TERMINAL_SCHEMA,
        "version": PROTOCOL_VERSION,
        "status": status,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "authorization_sha256": authorization_snapshot.sha256,
        "claim_sha256": claim["snapshot"].sha256,
        "failure_is_non_reusable": True,
        "error": (
            None
            if error is None
            else {
                "type": type(error).__name__,
                "message": str(error)[:MAX_ERROR_CHARS],
                "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__))[
                    -MAX_ERROR_CHARS:
                ],
            }
        ),
    }
    receipt = {**body, "canonical_digest_sha256": canonical_sha256(body)}
    _exclusive_create(path, _json_bytes(receipt))


def execute_postfreeze_blind_v8(
    *,
    authorization_path: Path,
) -> dict[str, Any]:
    """Consume the native-v8 claim exactly once and run one HF CUDA pass."""

    authorization_snapshot, authorization, paths = _load_authorization(authorization_path)
    if any(os.path.lexists(paths[name]) for name in ("claim", "terminal", "evidence")):
        raise PostfreezeBlindV8Error("native-v8 commitment is already consumed or in progress")
    _require_cuda_ready()
    _reverify_authorization_inputs(authorization)
    claim = _claim_once(
        claim_path=paths["claim"],
        authorization_snapshot=authorization_snapshot,
        authorization=authorization,
    )
    try:
        examples, rows, derived_payload, derivation = _derive_blind_rows_after_claim(
            authorization=authorization,
            claim=claim,
        )
        if len(examples) != EXPECTED_ROWS or len(rows) != EXPECTED_ROWS:
            raise PostfreezeBlindV8Error("native-v8 derived membership is incomplete")
        derived_sha = sha256_bytes(derived_payload)
        authorization["_derived_sha256"] = derived_sha
        samples, backend = _evaluate_hf_cuda_once(
            rows,
            authorization=authorization,
            claim=claim,
            derived_sha256=derived_sha,
        )
        gates = _release_gates(samples, backend=backend)
        summary = _build_summary(
            samples,
            backend=backend,
            derivation=derivation,
            gate_results=gates,
        )
        sample_payload = _jsonl_bytes(samples)
        summary_payload = _json_bytes(summary)
        registry, registry_identity = _real_directory(
            paths["authorization"].parent,
            label="fixed registry before evidence",
        )
        staging: Path | None = Path(tempfile.mkdtemp(prefix=".native-v8-evidence-", dir=registry))
        try:
            assert staging is not None
            sample_artifact = _artifact(
                paths["evidence"] / SAMPLE_FILENAME,
                sample_payload,
                records=EXPECTED_ROWS,
            )
            summary_artifact = _artifact(
                paths["evidence"] / SUMMARY_FILENAME,
                summary_payload,
            )
            for name, payload in (
                (DERIVED_FILENAME, derived_payload),
                (SAMPLE_FILENAME, sample_payload),
                (SUMMARY_FILENAME, summary_payload),
            ):
                _exclusive_create(staging / name, payload)
            run_receipt = _build_run_receipt(
                authorization=authorization,
                claim=claim,
                sample_artifact=sample_artifact,
                summary_artifact=summary_artifact,
            )
            run_payload = _json_bytes(run_receipt)
            run_artifact = _artifact(staging / RUN_RECEIPT_FILENAME, run_payload)
            _exclusive_create(staging / RUN_RECEIPT_FILENAME, run_payload)
            qualification = _build_qualification(
                authorization=authorization,
                claim=claim,
                run_receipt_artifact=run_artifact,
                gate_results=gates,
                sample_artifact=sample_artifact,
                summary_artifact=summary_artifact,
            )
            qualification_payload = _json_bytes(qualification)
            _exclusive_create(staging / QUALIFICATION_FILENAME, qualification_payload)
            names = {path.name for path in staging.iterdir()}
            if names != EXECUTION_EVIDENCE_FILENAMES:
                raise PostfreezeBlindV8Error("native-v8 evidence inventory mismatch")
            os.replace(staging, paths["evidence"])
            staging = None
        finally:
            if staging is not None and os.path.lexists(staging):
                shutil.rmtree(staging)
        _verify_directory(registry, registry_identity, label="fixed registry after evidence")
        qualified = qualification["qualified"] is True
        _write_terminal(
            path=paths["terminal"],
            authorization_snapshot=authorization_snapshot,
            claim=claim,
            status=("COMPLETED_GGUF_OFFLINE_CANDIDATE_ONLY" if qualified else "COMPLETED_HOLD_NON_REUSABLE"),
            error=None,
        )
        return {
            "status": RUN_COMPLETE_STATUS,
            "qualified": qualified,
            "claim": claim["publication"],
            "evidence_dir": str(paths["evidence"]),
            "run_receipt": _artifact(
                paths["evidence"] / RUN_RECEIPT_FILENAME,
                run_payload,
            ),
            "qualification": _artifact(
                paths["evidence"] / QUALIFICATION_FILENAME,
                qualification_payload,
            ),
            "retry_allowed": False,
        }
    except BaseException as exc:
        if not os.path.lexists(paths["terminal"]):
            try:
                _write_terminal(
                    path=paths["terminal"],
                    authorization_snapshot=authorization_snapshot,
                    claim=claim,
                    status="FAILED_NON_REUSABLE",
                    error=exc,
                )
            except BaseException:
                pass
        if isinstance(exc, PostfreezeBlindV8Error):
            raise
        raise PostfreezeBlindV8Error(f"native-v8 one-shot failed after irreversible claim: {exc}") from exc


def _load_execution_evidence(
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    root, root_identity = _real_directory(paths["evidence"], label="native-v8 evidence")
    names = {path.name for path in root.iterdir()}
    if names not in (EXECUTION_EVIDENCE_FILENAMES, VERIFIED_EVIDENCE_FILENAMES):
        raise PostfreezeBlindV8Error("native-v8 evidence exact inventory mismatch")
    snapshots = {name: _capture(root / name, label=f"native-v8 evidence {name}") for name in sorted(names)}
    return {
        "root": root,
        "identity": root_identity,
        "snapshots": snapshots,
    }


def _verify_run_receipt(
    receipt: Mapping[str, Any],
    *,
    authorization: Mapping[str, Any],
    claim_snapshot: lifecycle_bindings_v7.StableFileSnapshot,
    claim: Mapping[str, Any],
    sample_snapshot: lifecycle_bindings_v7.StableFileSnapshot,
    summary_snapshot: lifecycle_bindings_v7.StableFileSnapshot,
    derived_sha256: str,
) -> None:
    expected_fields = {
        "schema",
        "version",
        "status",
        "chain_binding",
        "upstream_receipts",
        "dataset",
        "execution_boundary",
        "authorization",
        "consumption_claim",
        "artifacts",
        "canonical_digest_sha256",
    }
    _mapping(receipt, label="v8 run receipt", exact=expected_fields)
    if (
        receipt.get("schema") != RUN_RECEIPT_SCHEMA
        or receipt.get("version") != PROTOCOL_VERSION
        or receipt.get("status") != RUN_COMPLETE_STATUS
        or receipt.get("chain_binding") != authorization["chain_binding"]
        or receipt.get("authorization") != _FALSE_AUTHORIZATION
        or receipt.get("dataset")
        != {
            "rows_read_once": EXPECTED_ROWS,
            "blind_sha256": derived_sha256,
            "nonblind_manifest_sha256": authorization["chain_binding"]["manifest_sha256"],
            "preblind_commitment_sha256": authorization["chain_binding"]["preblind_commitment_sha256"],
        }
        or receipt.get("consumption_claim")
        != {
            "sha256": claim_snapshot.sha256,
            "nonce_sha256": claim["nonce_sha256"],
            "failure_is_non_reusable": True,
        }
    ):
        raise PostfreezeBlindV8Error("native-v8 run receipt binding mismatch")
    artifacts = _mapping(
        receipt.get("artifacts"),
        label="v8 run artifacts",
        exact={"sample_results", "summary"},
    )
    expected_artifacts = {
        "sample_results": {
            "path": str(sample_snapshot.path),
            "bytes": sample_snapshot.bytes,
            "sha256": sample_snapshot.sha256,
        },
        "summary": {
            "path": str(summary_snapshot.path),
            "bytes": summary_snapshot.bytes,
            "sha256": summary_snapshot.sha256,
        },
    }
    if artifacts != expected_artifacts:
        raise PostfreezeBlindV8Error("native-v8 run artifact binding mismatch")


def _verify_qualification(
    receipt: Mapping[str, Any],
    *,
    authorization: Mapping[str, Any],
    claim_snapshot: lifecycle_bindings_v7.StableFileSnapshot,
    claim: Mapping[str, Any],
    run_snapshot: lifecycle_bindings_v7.StableFileSnapshot,
    sample_snapshot: lifecycle_bindings_v7.StableFileSnapshot,
    summary_snapshot: lifecycle_bindings_v7.StableFileSnapshot,
    gates: Sequence[Mapping[str, Any]],
) -> None:
    expected = _build_qualification(
        authorization=authorization,
        claim={"snapshot": claim_snapshot, "receipt": claim},
        run_receipt_artifact={
            "path": str(run_snapshot.path),
            "bytes": run_snapshot.bytes,
            "sha256": run_snapshot.sha256,
        },
        gate_results=gates,
        sample_artifact={
            "path": str(sample_snapshot.path),
            "bytes": sample_snapshot.bytes,
            "sha256": sample_snapshot.sha256,
        },
        summary_artifact={
            "path": str(summary_snapshot.path),
            "bytes": summary_snapshot.bytes,
            "sha256": summary_snapshot.sha256,
        },
    )
    # Timestamps are deliberately absent from the GGUF-facing v8 receipt, so
    # deterministic reconstruction must be byte-for-byte identical.
    if dict(receipt) != expected:
        raise PostfreezeBlindV8Error("native-v8 qualification differs from recomputation")


def verify_release_qualification_v8(
    *,
    authorization_path: Path,
) -> dict[str, Any]:
    """Independently re-derive and re-score the consumed native-v8 run."""

    authorization_snapshot, authorization, paths = _load_authorization(authorization_path)
    if not os.path.lexists(paths["claim"]) or not os.path.lexists(paths["terminal"]):
        raise PostfreezeBlindV8Error("native-v8 claim is not terminal")
    claim_snapshot = _capture(paths["claim"], label="native-v8 claim")
    claim = _canonical_receipt(
        claim_snapshot,
        label="native-v8 claim",
        schema=CLAIM_SCHEMA,
        version=PROTOCOL_VERSION,
        status="CLAIMED_NON_REUSABLE",
    )
    _mapping(claim, label="native-v8 claim", exact=_CLAIM_FIELDS)
    if (
        claim.get("authorization_sha256") != authorization_snapshot.sha256
        or claim.get("authorization_id") != authorization["authorization_id"]
        or claim.get("preblind_commitment_sha256")
        != authorization["chain_binding"]["preblind_commitment_sha256"]
        or claim.get("failure_is_non_reusable") is not True
        or claim.get("retry_allowed") is not False
    ):
        raise PostfreezeBlindV8Error("native-v8 claim binding mismatch")
    _sha(claim.get("nonce_sha256"), label="native-v8 claim nonce")
    terminal_snapshot = _capture(paths["terminal"], label="native-v8 terminal")
    terminal = _canonical_receipt(
        terminal_snapshot,
        label="native-v8 terminal",
        schema=TERMINAL_SCHEMA,
        version=PROTOCOL_VERSION,
    )
    _mapping(terminal, label="native-v8 terminal", exact=_TERMINAL_FIELDS)
    if (
        terminal.get("authorization_sha256") != authorization_snapshot.sha256
        or terminal.get("claim_sha256") != claim_snapshot.sha256
        or terminal.get("failure_is_non_reusable") is not True
    ):
        raise PostfreezeBlindV8Error("native-v8 terminal binding mismatch")
    if terminal.get("status") != "COMPLETED_GGUF_OFFLINE_CANDIDATE_ONLY":
        raise PostfreezeBlindV8Error("native-v8 run did not qualify and cannot be promoted")
    reverified = _reverify_authorization_inputs(authorization)
    evidence = _load_execution_evidence(paths)
    snapshots = evidence["snapshots"]
    derived_snapshot = snapshots[DERIVED_FILENAME]
    sample_snapshot = snapshots[SAMPLE_FILENAME]
    summary_snapshot = snapshots[SUMMARY_FILENAME]
    run_snapshot = snapshots[RUN_RECEIPT_FILENAME]
    qualification_snapshot = snapshots[QUALIFICATION_FILENAME]
    run_receipt = _canonical_receipt(
        run_snapshot,
        label="native-v8 run receipt",
        schema=RUN_RECEIPT_SCHEMA,
        version=PROTOCOL_VERSION,
        status=RUN_COMPLETE_STATUS,
    )
    qualification = _canonical_receipt(
        qualification_snapshot,
        label="native-v8 qualification",
        schema=QUALIFICATION_SCHEMA,
        version=PROTOCOL_VERSION,
        status=QUALIFICATION_PASS_STATUS,
    )
    examples, rows, rederived_payload, derivation = _derive_blind_rows_after_claim(
        authorization=authorization,
        claim={"snapshot": claim_snapshot, "receipt": claim},
    )
    if (
        len(examples) != EXPECTED_ROWS
        or rederived_payload != derived_snapshot.payload
        or sha256_bytes(rederived_payload) != derived_snapshot.sha256
    ):
        raise PostfreezeBlindV8Error("native-v8 derived rows differ from independent derivation")
    recorded_samples = _parse_jsonl(
        sample_snapshot.payload,
        label="native-v8 sample results",
    )
    recomputed_samples = _recompute_samples(
        rows=rows,
        recorded_samples=recorded_samples,
        authorization=authorization,
        claim_sha256=claim_snapshot.sha256,
        derived_sha256=derived_snapshot.sha256,
    )
    summary = _parse_json(summary_snapshot, label="native-v8 summary")
    backend_summary = _mapping(summary.get("backend"), label="native-v8 summary backend")
    model = _mapping(authorization.get("model"), label="authorization model")
    synthetic_backend = {
        "mode": backend_summary.get("mode"),
        "device": backend_summary.get("device"),
        "seed": backend_summary.get("seed"),
        "samples_generated": backend_summary.get("samples_generated"),
        "local_files_only": backend_summary.get("local_files_only"),
        "network_allowed": backend_summary.get("network_allowed"),
        "model": backend_summary.get("model"),
    }
    _backend_tree_binding(
        synthetic_backend,
        chain=authorization["chain_binding"],
        selected_checkpoint_path=Path(str(model["checkpoint_path"])),
        base_model_path=Path(str(model["base_path"])),
        adapter_required=True,
        label="native-v8 verifier backend",
    )
    gates = _release_gates(recomputed_samples, backend=synthetic_backend)
    recomputed_summary = _build_summary(
        recomputed_samples,
        backend=synthetic_backend,
        derivation=derivation,
        gate_results=gates,
    )
    if summary != recomputed_summary:
        raise PostfreezeBlindV8Error("native-v8 summary differs from recomputation")
    _verify_run_receipt(
        run_receipt,
        authorization=authorization,
        claim_snapshot=claim_snapshot,
        claim=claim,
        sample_snapshot=sample_snapshot,
        summary_snapshot=summary_snapshot,
        derived_sha256=derived_snapshot.sha256,
    )
    _verify_qualification(
        qualification,
        authorization=authorization,
        claim_snapshot=claim_snapshot,
        claim=claim,
        run_snapshot=run_snapshot,
        sample_snapshot=sample_snapshot,
        summary_snapshot=summary_snapshot,
        gates=gates,
    )
    if qualification.get("qualified") is not True:
        raise PostfreezeBlindV8Error("native-v8 qualification is not passing")
    verification_body = {
        "schema": VERIFICATION_SCHEMA,
        "version": VERIFICATION_VERSION,
        "status": VERIFICATION_STATUS,
        "chain_binding": dict(authorization["chain_binding"]),
        "verified_receipts": {
            "selection_freeze_sha256": authorization["chain_binding"]["selection_freeze_sha256"],
            "calibration_sha256": authorization["upstream_receipts"]["calibration"]["sha256"],
            "ablation_sha256": authorization["upstream_receipts"]["ablation"]["sha256"],
            "postfreeze_sha256": run_snapshot.sha256,
            "qualification_sha256": qualification_snapshot.sha256,
        },
        "independent_recomputation": {
            "selection_reverified": True,
            "calibration_samples_recomputed": reverified["postselection"]["calibration"][
                "samples_recomputed"
            ],
            "calibration_summary_recomputed": reverified["postselection"]["calibration"][
                "summary_recomputed"
            ],
            "ablation_matrix_recomputed": reverified["postselection"]["ablation"]["samples_recomputed"],
            "ablation_reports_recomputed": reverified["postselection"]["ablation"]["reports_recomputed"],
            "blind_samples_recomputed": True,
            "blind_summary_recomputed": True,
            "qualification_recomputed": True,
        },
        "release_authorization": {
            "gguf_export_authorized": True,
            **_FALSE_AUTHORIZATION,
        },
    }
    verification = {
        **verification_body,
        "canonical_digest_sha256": canonical_sha256(verification_body),
    }
    verification_payload = _json_bytes(verification)
    verification_path = evidence["root"] / VERIFICATION_FILENAME
    if os.path.lexists(verification_path):
        existing = _capture(verification_path, label="existing native-v8 verification")
        if existing.payload != verification_payload:
            raise PostfreezeBlindV8Error("existing native-v8 verification differs")
        publication = existing.receipt()
    else:
        publication = _exclusive_create(verification_path, verification_payload)
    final_evidence_identity = _real_directory(
        evidence["root"],
        label="native-v8 evidence directory after verification",
    )[1]
    for snapshot in (
        authorization_snapshot,
        claim_snapshot,
        terminal_snapshot,
        *snapshots.values(),
    ):
        _verify_snapshot(snapshot, label=f"native-v8 verifier final {snapshot.path.name}")
    _verify_directory(
        evidence["root"],
        final_evidence_identity,
        label="native-v8 evidence directory final",
    )
    return {
        "status": VERIFICATION_STATUS,
        "verification": publication,
        "postfreeze_receipt": run_snapshot.receipt(),
        "qualification_receipt": qualification_snapshot.receipt(),
        "rows_rederived": EXPECTED_ROWS,
        "rows_recompiled": EXPECTED_ROWS,
        "gguf_export_authorized": True,
        **_FALSE_AUTHORIZATION,
    }


__all__ = [
    "AUTHORIZATION_SCHEMA",
    "AUTHORIZATION_STATUS",
    "CLAIM_SCHEMA",
    "DERIVED_FILENAME",
    "EXPECTED_ROWS",
    "FIXED_SEED",
    "PRODUCTION_REGISTRY_ROOT",
    "PROTOCOL_VERSION",
    "QUALIFICATION_FILENAME",
    "QUALIFICATION_PASS_STATUS",
    "QUALIFICATION_SCHEMA",
    "RUN_COMPLETE_STATUS",
    "RUN_RECEIPT_FILENAME",
    "RUN_RECEIPT_SCHEMA",
    "SAMPLE_FILENAME",
    "SAMPLE_SCHEMA",
    "SUMMARY_FILENAME",
    "SUMMARY_SCHEMA",
    "TERMINAL_SCHEMA",
    "VERIFICATION_FILENAME",
    "VERIFICATION_SCHEMA",
    "VERIFICATION_STATUS",
    "VERIFICATION_VERSION",
    "PostfreezeBlindV8Error",
    "canonical_json",
    "canonical_sha256",
    "execute_postfreeze_blind_v8",
    "prepare_postfreeze_blind_v8",
    "verify_release_qualification_v8",
]
