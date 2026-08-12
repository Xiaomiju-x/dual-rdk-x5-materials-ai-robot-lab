"""One-shot sealed-blind protocol for the ICMat Evidence Pointer v6 model.

Authorization is metadata-only with respect to the sealed JSONL: it trusts the
manifest declaration and file stat but never reads, hashes, or parses the blind
bytes. Consumption re-verifies every non-blind binding, creates an immutable
global claim with ``O_EXCL``, and only then reads, hashes, and parses the 150
blind rows. A claim is non-reusable even when generation or publication fails.

Blind results remain final-report evidence. A separate immutable receipt may
qualify an offline GGUF artifact build against thresholds frozen before
consumption; it never authorizes model selection, threshold fitting,
activation, deployment, or production integration.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import secrets
import shutil
import traceback
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from icmat_foundry.llm import (
    ablation_eval_v6,
    calibration_eval_v6,
    contracts_v6,
    pointer_checkpoint_eval_v6,
    selection_freeze_v6,
)
from icmat_foundry.llm import pointer_hf_eval_v6 as pointer_eval

AUTHORIZATION_SCHEMA = "icmat_llm_blind_authorization.v6"
AUTHORIZATION_VERSION = "icmat-llm-blind-protocol-v6.2.0"
AUTHORIZATION_STATUS = "AUTHORIZED_ONCE_ALL_PREBLIND_GATES_PASS"
REGISTRY_SCHEMA = "icmat_llm_blind_authorization_registry.v6"
CLAIM_SCHEMA = "icmat_llm_blind_consumption_claim.v6"
TERMINAL_SCHEMA = "icmat_llm_blind_consumption_terminal.v6"
SAMPLE_SCHEMA = "icmat_pointer_blind_sample.v6"
SUMMARY_SCHEMA = "icmat_pointer_blind_summary.v6"
RUN_RECEIPT_SCHEMA = "icmat_pointer_blind_run_receipt.v6"
RELEASE_QUALIFICATION_SCHEMA = "icmat_pointer_blind_release_qualification.v6"
RELEASE_QUALIFICATION_PASS_STATUS = "PASS_BLIND_THRESHOLDS_GGUF_RELEASE_ONLY"
RELEASE_QUALIFICATION_HOLD_STATUS = "HOLD_BLIND_THRESHOLDS_NOT_MET_NON_REUSABLE"
RELEASE_QUALIFICATION_VERSION = "icmat-pointer-blind-release-qualification-v6.0.0"

SELECTION_FREEZE_SCHEMA = selection_freeze_v6.SCHEMA
SELECTION_FREEZE_STATUS = selection_freeze_v6.STATUS
CALIBRATION_SCHEMA = calibration_eval_v6.RECEIPT_SCHEMA
CALIBRATION_STATUS = "PASS_NONBLIND_CALIBRATION_MODEL_BOUND"
CALIBRATION_FIXTURE_STATUS = "PASS_FIXTURE_CALIBRATION_PIPELINE_VERIFIED_NOT_MODEL_EVIDENCE"
ABLATION_SCHEMA = ablation_eval_v6.RECEIPT_SCHEMA
ABLATION_STATUS = "PASS_NONBLIND_ABLATIONS_COMPLETE_NO_SELECTION"

DATASET_MANIFEST_SCHEMA = "icmat_evidence_pointer_manifest.v6"
DATASET_STATUS = "DATASET_BUILT_BLIND_HASH_SEALED"
BLIND_SEAL_SCHEMA = "icmat_evidence_pointer_blind_seal.v6"
EXPECTED_BLIND_EXAMPLES = 150
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_BLIND_BYTES = 128 * 1024 * 1024
MAX_FIXTURE_BYTES = 32 * 1024 * 1024
MAX_ERROR_CHARS = 2000
FIXED_SEED = 20260729
FIXED_MAX_INPUT_TOKENS = 1536
FIXED_MAX_NEW_TOKENS = 64

# These integer thresholds are frozen into the one-shot authorization before
# the sealed rows are parsed. Blind results can qualify an offline GGUF
# artifact build only; they can never authorize activation or deployment.
RELEASE_QUALIFICATION_POLICY = {
    "expected_examples": EXPECTED_BLIND_EXAMPLES,
    "required_backend": "hf_model",
    "model_bound_required": True,
    "pointer_parse_valid": {"numerator": 1, "denominator": 1},
    "pointer_compiler_accepted": {"numerator": 1, "denominator": 1},
    "unsupported_wrong_answer_max_count": 0,
    "compiled_schema_valid": {"numerator": 1, "denominator": 1},
    "compiled_citation_exact": {"numerator": 1, "denominator": 1},
    "compiled_provenance_exact": {"numerator": 1, "denominator": 1},
    "answer_span_exact_minimum": {"numerator": 19, "denominator": 20},
    "refuse_f1_minimum": {"numerator": 19, "denominator": 20},
    "gguf_release_qualification_allowed": True,
    "activation_authorized": False,
    "deployment_authorized": False,
    "production_integration_authorized": False,
    "model_selection_allowed": False,
    "checkpoint_ranking_allowed": False,
    "threshold_tuning_allowed": False,
    "calibration_allowed": False,
    "retry_after_failure_allowed": False,
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{7,127}$")
_OUTPUT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
_SUPPORTED_BACKENDS = frozenset({"fixture", "hf_model"})
_ADAPTER_FILENAMES = pointer_checkpoint_eval_v6.ADAPTER_FILENAMES


class BlindProtocolV6Error(RuntimeError):
    """Raised when a sealed-blind contract fails closed."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join((canonical_json(dict(record)) + "\n").encode("utf-8") for record in records)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise BlindProtocolV6Error(f"duplicate JSON key rejected: {key}")
        value[key] = item
    return value


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise BlindProtocolV6Error(f"{field} must be SHA-256 text")
    normalized = value.lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise BlindProtocolV6Error(f"{field} is not a lowercase SHA-256")
    return normalized


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BlindProtocolV6Error(f"{field} must be an object")
    return value


def _require_bool(value: Any, expected: bool, *, field: str) -> None:
    if value is not expected:
        raise BlindProtocolV6Error(f"{field} must be {expected}")


def _stable_regular_file(path: Path, *, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise BlindProtocolV6Error(f"{label} must not be a symlink")
    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError as exc:
        raise BlindProtocolV6Error(f"{label} is unavailable") from exc
    if not resolved.is_file():
        raise BlindProtocolV6Error(f"{label} must be a regular file")
    return resolved


def _load_json(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[Path, bytes, dict[str, Any]]:
    resolved = _stable_regular_file(path, label=label)
    size = resolved.stat().st_size
    if size <= 0 or size > MAX_JSON_BYTES:
        raise BlindProtocolV6Error(f"{label} bytes must be in 1..{MAX_JSON_BYTES}")
    before = resolved.stat()
    payload = resolved.read_bytes()
    after = resolved.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(payload) != before.st_size
    ):
        raise BlindProtocolV6Error(f"{label} changed while it was read")
    actual_sha = sha256_bytes(payload)
    if expected_sha256 is not None:
        expected = _require_sha256(
            expected_sha256,
            field=f"{label} expected SHA-256",
        )
        if actual_sha != expected:
            raise BlindProtocolV6Error(f"{label} SHA-256 mismatch")
    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                BlindProtocolV6Error(f"{label} contains non-finite JSON constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BlindProtocolV6Error(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(parsed, dict):
        raise BlindProtocolV6Error(f"{label} must contain one JSON object")
    try:
        canonical_json(parsed)
    except (TypeError, ValueError) as exc:
        raise BlindProtocolV6Error(f"{label} contains unsupported JSON values") from exc
    return resolved, payload, parsed


def _tree_inventory(
    path: Path,
    *,
    label: str,
    selected_names: frozenset[str] | None = None,
) -> dict[str, Any]:
    raw = Path(path)
    if raw.is_symlink():
        raise BlindProtocolV6Error(f"{label} root must not be a symlink")
    try:
        root = raw.resolve(strict=True)
    except FileNotFoundError as exc:
        raise BlindProtocolV6Error(f"{label} root is unavailable") from exc
    if not root.is_dir():
        raise BlindProtocolV6Error(f"{label} root must be a directory")
    candidates = list(root.rglob("*"))
    candidates.sort(
        key=lambda item: (
            item.relative_to(root).as_posix().casefold(),
            item.relative_to(root).as_posix(),
        )
    )
    files: list[dict[str, Any]] = []
    casefold_paths: set[str] = set()
    for candidate in candidates:
        if candidate.is_symlink():
            raise BlindProtocolV6Error(f"{label} contains a symlink: {candidate}")
        if candidate.is_file():
            if selected_names is not None and candidate.name not in selected_names:
                continue
            relative = candidate.relative_to(root).as_posix()
            folded = relative.casefold()
            if folded in casefold_paths:
                raise BlindProtocolV6Error(
                    f"{label} contains Windows-ambiguous paths"
                )
            casefold_paths.add(folded)
            before = candidate.stat()
            digest = sha256_file(candidate)
            after = candidate.stat()
            if (
                before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
            ):
                raise BlindProtocolV6Error(
                    f"{label} changed while hashing: {candidate}"
                )
            files.append(
                {
                    "path": relative,
                    "bytes": after.st_size,
                    "sha256": digest,
                }
            )
    if not files:
        raise BlindProtocolV6Error(f"{label} tree is empty")
    return {
        "path": str(root),
        "files": files,
        "file_count": len(files),
        "bytes": sum(int(record["bytes"]) for record in files),
        "tree_sha256": sha256_bytes(canonical_json(files).encode("utf-8")),
        "ordering": "windows_casefold_then_posix",
    }


def _adapter_inventory(path: Path, *, label: str) -> dict[str, Any]:
    inventory = _tree_inventory(
        path,
        label=label,
        selected_names=_ADAPTER_FILENAMES,
    )
    names = [Path(str(item["path"])).name for item in inventory["files"]]
    if (
        len(names) != 2
        or "adapter_config.json" not in names
        or sum(
            name in {"adapter_model.safetensors", "adapter_model.bin"}
            for name in names
        )
        != 1
    ):
        raise BlindProtocolV6Error(
            f"{label} must contain exactly adapter_config.json and one adapter model"
        )
    return inventory


def _safe_child(root: Path, relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str):
        raise BlindProtocolV6Error(f"{label} must be a relative path")
    posix = PurePosixPath(relative)
    if posix.is_absolute() or not posix.parts or ".." in posix.parts:
        raise BlindProtocolV6Error(f"{label} is unsafe")
    candidate = root / Path(*posix.parts)
    if candidate.is_symlink():
        raise BlindProtocolV6Error(f"{label} must not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise BlindProtocolV6Error(f"{label} escapes or is unavailable") from exc
    return resolved


def _dataset_metadata(
    dataset_dir: Path,
    *,
    verify_blind_hash: bool,
) -> dict[str, Any]:
    raw = Path(dataset_dir)
    if raw.is_symlink():
        raise BlindProtocolV6Error("dataset directory must not be a symlink")
    try:
        root = raw.resolve(strict=True)
    except FileNotFoundError as exc:
        raise BlindProtocolV6Error("dataset directory is unavailable") from exc
    if not root.is_dir():
        raise BlindProtocolV6Error("dataset directory must be a directory")

    manifest_path, manifest_payload, manifest = _load_json(
        root / "manifest.v6.json",
        label="v6 dataset manifest",
    )
    if manifest.get("schema") != DATASET_MANIFEST_SCHEMA:
        raise BlindProtocolV6Error("dataset manifest schema is invalid")
    if manifest.get("status") != DATASET_STATUS:
        raise BlindProtocolV6Error("dataset manifest is not sealed")
    training_boundary = _require_mapping(
        manifest.get("training_boundary"),
        field="manifest.training_boundary",
    )
    if (
        training_boundary.get("allowed_splits") != ["train", "validation"]
        or training_boundary.get("calibration_content_for_training") is not False
        or training_boundary.get("forbidden_split") != "blind_test"
        or training_boundary.get("blind_test_requires_explicit_post_freeze_authorization") is not True
    ):
        raise BlindProtocolV6Error("dataset manifest training/blind boundary is invalid")
    splits = _require_mapping(manifest.get("splits"), field="manifest.splits")
    descriptor = _require_mapping(
        splits.get("blind_test"),
        field="manifest.splits.blind_test",
    )
    if descriptor.get("count") != EXPECTED_BLIND_EXAMPLES:
        raise BlindProtocolV6Error(f"blind split must declare {EXPECTED_BLIND_EXAMPLES} rows")
    blind_sha = _require_sha256(
        descriptor.get("sha256"),
        field="manifest blind SHA-256",
    )
    blind_path = _safe_child(
        root,
        descriptor.get("path"),
        label="sealed blind JSONL",
    )
    if blind_path.suffix.casefold() != ".jsonl" or not blind_path.is_file():
        raise BlindProtocolV6Error("sealed blind path must be JSONL")
    blind_bytes = blind_path.stat().st_size
    if (
        isinstance(descriptor.get("bytes"), bool)
        or descriptor.get("bytes") != blind_bytes
        or blind_bytes <= 0
        or blind_bytes > MAX_BLIND_BYTES
    ):
        raise BlindProtocolV6Error("sealed blind byte declaration is invalid")

    if verify_blind_hash:
        # This branch is reserved for post-claim consumption. Authorization
        # and all pre-claim checks must pass verify_blind_hash=False.
        actual_blind_sha = sha256_file(blind_path)
        if actual_blind_sha != blind_sha:
            raise BlindProtocolV6Error("sealed blind JSONL SHA-256 mismatch")

    artifacts = _require_mapping(
        manifest.get("artifacts"),
        field="manifest.artifacts",
    )
    seal_descriptor = _require_mapping(
        artifacts.get("blind_seal"),
        field="manifest.artifacts.blind_seal",
    )
    seal_path = _safe_child(
        root,
        seal_descriptor.get("path"),
        label="blind seal receipt",
    )
    seal_expected_sha = _require_sha256(
        seal_descriptor.get("sha256"),
        field="blind seal SHA-256",
    )
    seal_resolved, seal_payload, seal = _load_json(
        seal_path,
        label="blind seal receipt",
        expected_sha256=seal_expected_sha,
    )
    if (
        seal.get("schema") != BLIND_SEAL_SCHEMA
        or seal.get("sealed") is not True
        or seal.get("authorization_required") is not True
        or seal.get("authorized_for_training") is not False
        or seal.get("authorized_for_checkpoint_selection") is not False
        or seal.get("content_disclosed") is not False
    ):
        raise BlindProtocolV6Error("blind seal policy is invalid")
    sealed_file = _require_mapping(
        seal.get("blind_test_file"),
        field="blind_seal.blind_test_file",
    )
    if dict(sealed_file) != {
        "bytes": blind_bytes,
        "count": EXPECTED_BLIND_EXAMPLES,
        "path": descriptor.get("path"),
        "sha256": blind_sha,
    }:
        raise BlindProtocolV6Error("blind seal and manifest descriptors differ")
    nonblind_descriptors: dict[str, dict[str, Any]] = {}
    for split_name in ("validation", "calibration"):
        split = _require_mapping(
            splits.get(split_name),
            field=f"manifest.splits.{split_name}",
        )
        if split.get("count") != EXPECTED_BLIND_EXAMPLES:
            raise BlindProtocolV6Error(f"{split_name} must declare {EXPECTED_BLIND_EXAMPLES} rows")
        nonblind_descriptors[split_name] = {
            "path": split.get("path"),
            "sha256": _require_sha256(
                split.get("sha256"),
                field=f"manifest {split_name} SHA-256",
            ),
            "bytes": split.get("bytes"),
            "count": split.get("count"),
        }
    return {
        "root": root,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_bytes(manifest_payload),
        "blind_path": blind_path,
        "blind_sha256": blind_sha,
        "blind_bytes": blind_bytes,
        "blind_count": EXPECTED_BLIND_EXAMPLES,
        "seal_path": seal_resolved,
        "seal_sha256": sha256_bytes(seal_payload),
        "nonblind_descriptors": nonblind_descriptors,
    }


def _common_bindings(
    *,
    dataset_manifest_sha256: str,
    base_model_tree_sha256: str,
    adapter_tree_sha256: str,
) -> dict[str, str]:
    return {
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "base_model_tree_sha256": base_model_tree_sha256,
        "adapter_tree_sha256": adapter_tree_sha256,
    }


def _verify_selection_freeze(
    path: Path,
    expected_sha256: str,
    *,
    common: Mapping[str, str],
) -> dict[str, Any]:
    resolved, payload, receipt = _load_json(
        path,
        label="final selection freeze",
        expected_sha256=expected_sha256,
    )
    if receipt.get("schema") != SELECTION_FREEZE_SCHEMA:
        raise BlindProtocolV6Error("final selection freeze schema is invalid")
    if receipt.get("status") != SELECTION_FREEZE_STATUS:
        raise BlindProtocolV6Error("final selection is not frozen")
    for field, expected in (
        ("selection_locked", True),
        ("calibration_authorized", True),
        ("blind_test_authorized", False),
        ("deployment_authorized", False),
    ):
        _require_bool(
            receipt.get(field),
            expected,
            field=f"selection.{field}",
        )
    authorization = _require_mapping(
        receipt.get("authorization"),
        field="selection.authorization",
    )
    for field, expected in (
        ("checkpoint_selected", True),
        ("model_authorized_for_calibration", True),
        ("calibration_authorized", True),
        ("blind_test_authorized", False),
        ("gguf_export_authorized", False),
        ("deployment_authorized", False),
        ("production_integration_authorized", False),
    ):
        _require_bool(
            authorization.get(field),
            expected,
            field=f"selection.authorization.{field}",
        )
    evaluation_index = _require_mapping(
        receipt.get("evaluation_index"),
        field="selection.evaluation_index",
    )
    training_receipt = _require_mapping(
        receipt.get("training_receipt"),
        field="selection.training_receipt",
    )
    dataset = _require_mapping(
        receipt.get("dataset"),
        field="selection.dataset",
    )
    base_model = _require_mapping(
        receipt.get("base_model"),
        field="selection.base_model",
    )
    try:
        verified = selection_freeze_v6.verify_selection_freeze(
            freeze_receipt_path=resolved,
            evaluation_index_path=Path(str(evaluation_index.get("path"))),
            training_receipt_path=Path(str(training_receipt.get("path"))),
            dataset_dir=Path(str(dataset.get("path"))),
            base_model_dir=Path(str(base_model.get("path"))),
        )
    except (selection_freeze_v6.SelectionFreezeV6Error, OSError, ValueError) as exc:
        raise BlindProtocolV6Error("final selection freeze failed independent recomputation") from exc
    if (
        verified.get("status") != selection_freeze_v6.VERIFIED_STATUS
        or verified.get("sha256") != sha256_bytes(payload)
        or verified.get("dataset_manifest_sha256") != common["dataset_manifest_sha256"]
        or verified.get("base_model_tree_sha256") != common["base_model_tree_sha256"]
        or verified.get("selected_adapter_tree_sha256") != common["adapter_tree_sha256"]
        or verified.get("checkpoint_count") != 18
        or verified.get("validation_samples_per_checkpoint") != EXPECTED_BLIND_EXAMPLES
        or int(verified.get("qualified_seed_count", 0)) < 2
        or verified.get("selection_locked") is not True
        or verified.get("calibration_authorized") is not True
        or verified.get("blind_test_authorized") is not False
        or verified.get("deployment_authorized") is not False
    ):
        raise BlindProtocolV6Error("final selection freeze artifact hashes or gates do not match")
    declarations = _require_mapping(
        dataset.get("declaration_only_splits"),
        field="selection.dataset.declaration_only_splits",
    )
    for split_name in ("calibration", "blind_test"):
        declaration = _require_mapping(
            declarations.get(split_name),
            field=f"selection declaration {split_name}",
        )
        if (
            declaration.get("content_read_by_freeze") is not False
            or declaration.get("content_hashed_by_freeze") is not False
        ):
            raise BlindProtocolV6Error(f"selection freeze accessed {split_name}")
        if split_name == "blind_test" and declaration.get("authorized") is not False:
            raise BlindProtocolV6Error("selection freeze authorized blind_test before calibration")
    runtime = _require_mapping(
        receipt.get("runtime_contract"),
        field="selection.runtime_contract",
    )
    decoding = _require_mapping(
        runtime.get("decoding"),
        field="selection.runtime_contract.decoding",
    )
    if (
        decoding.get("algorithm") != "greedy"
        or decoding.get("do_sample") is not False
        or decoding.get("singleton") is not True
        or decoding.get("batch_size") != 1
        or decoding.get("seed") != FIXED_SEED
        or decoding.get("max_input_tokens") != FIXED_MAX_INPUT_TOKENS
        or decoding.get("max_new_tokens") != FIXED_MAX_NEW_TOKENS
    ):
        raise BlindProtocolV6Error("selection freeze runtime contract is not fixed v6 inference")
    return {
        "path": str(resolved),
        "sha256": sha256_bytes(payload),
        "schema": receipt["schema"],
        "status": receipt["status"],
        "verification_status": verified["status"],
        "selection_binding_digest_sha256": _require_sha256(
            receipt.get("selection_binding_digest_sha256"),
            field="selection_binding_digest_sha256",
        ),
        "selected_checkpoint_id": verified["selected_checkpoint_id"],
        "selected_seed": verified["selected_seed"],
        "selected_epoch": verified["selected_epoch"],
        "selected_checkpoint_path": str(
            Path(str(verified["selected_checkpoint_path"])).resolve(strict=True)
        ),
        "selected_checkpoint_tree_sha256": _require_sha256(
            verified.get("selected_checkpoint_tree_sha256"),
            field="verified selected checkpoint tree SHA-256",
        ),
        "selected_adapter_path": str(
            Path(str(verified["selected_adapter_path"])).resolve(strict=True)
        ),
        "selected_adapter_tree_sha256": _require_sha256(
            verified.get("selected_adapter_tree_sha256"),
            field="verified selected adapter tree SHA-256",
        ),
    }


def _run_authoritative_receipt_verifier(
    *,
    module: Any,
    function_name: str,
    label: str,
    receipt_path: Path,
    receipt_sha256: str,
    selection_freeze_path: Path,
    selection_freeze_sha256: str,
    dataset_dir: Path,
    base_model_dir: Path,
    checkpoint_dir: Path,
    adapter_dir: Path,
) -> dict[str, Any]:
    """Call a fixed producer-owned verifier without accepting caller hooks."""

    verifier = getattr(module, function_name, None)
    if not callable(verifier):
        raise BlindProtocolV6Error(
            f"authoritative {label} verifier is unavailable: {function_name}"
        )
    values: dict[str, Any] = {
        "receipt_path": receipt_path,
        f"{label}_receipt_path": receipt_path,
        "path": receipt_path,
        "expected_sha256": receipt_sha256,
        "receipt_sha256": receipt_sha256,
        f"{label}_receipt_sha256": receipt_sha256,
        "selection_freeze_path": selection_freeze_path,
        "selection_freeze_sha256": selection_freeze_sha256,
        "dataset_dir": dataset_dir,
        "base_model_dir": base_model_dir,
        "checkpoint_dir": checkpoint_dir,
        "adapter_dir": adapter_dir,
    }
    signature = inspect.signature(verifier)
    arguments: dict[str, Any] = {}
    for parameter in signature.parameters.values():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise BlindProtocolV6Error(
                f"authoritative {label} verifier exposes positional-only inputs"
            )
        if parameter.name in values:
            arguments[parameter.name] = values[parameter.name]
        elif parameter.default is inspect.Parameter.empty:
            raise BlindProtocolV6Error(
                f"authoritative {label} verifier requires unsupported input "
                f"{parameter.name}"
            )
    try:
        raw_result = verifier(**arguments)
    except Exception as exc:
        raise BlindProtocolV6Error(
            f"authoritative {label} receipt verification failed"
        ) from exc
    if not isinstance(raw_result, Mapping):
        raise BlindProtocolV6Error(
            f"authoritative {label} verifier returned no structured receipt"
        )
    normalized = json.loads(canonical_json(dict(raw_result)))
    for hash_field in ("sha256", "receipt_sha256"):
        if hash_field in normalized and normalized[hash_field] != receipt_sha256:
            raise BlindProtocolV6Error(
                f"authoritative {label} verifier returned a different receipt hash"
            )
    status = normalized.get("status")
    if isinstance(status, str) and any(
        token in status.upper()
        for token in ("FAIL", "BLOCK", "REJECT", "INVALID", "HOLD")
    ):
        raise BlindProtocolV6Error(
            f"authoritative {label} verifier did not pass"
        )
    for field, expected in (
        ("blind_data_accessed", False),
        ("model_bound", True),
        ("complete_split", True),
    ):
        if field in normalized and normalized[field] is not expected:
            raise BlindProtocolV6Error(
                f"authoritative {label} verifier returned invalid {field}"
            )
    source = Path(inspect.getsourcefile(verifier) or module.__file__).resolve(
        strict=True
    )
    return {
        "function": function_name,
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
        },
        "receipt_sha256": receipt_sha256,
        "result": normalized,
    }


def _verify_calibration_receipt_recomputed_v6(
    *,
    receipt_path: Path,
    receipt_sha256: str,
    dataset_dir: Path,
) -> dict[str, Any]:
    """Recompile every calibration generation against the frozen split."""

    try:
        selection = pointer_eval.select_dataset(
            dataset_dir=dataset_dir,
            split="calibration",
            max_samples=None,
        )
        _, sample_payload, observed_rows = calibration_eval_v6._load_jsonl(
            receipt_path.parent / "per_sample.v6.jsonl",
            field="calibration per-sample evidence",
        )
        _, summary_payload, observed_summary = calibration_eval_v6._load_json(
            receipt_path.parent / "summary.v6.json",
            field="calibration summary evidence",
        )
    except (
        calibration_eval_v6.CalibrationEvalV6Error,
        pointer_eval.PointerHFEvalV6Error,
        OSError,
        ValueError,
    ) as exc:
        raise BlindProtocolV6Error(
            "calibration evidence failed independent reload"
        ) from exc
    if (
        len(selection.rows) != EXPECTED_BLIND_EXAMPLES
        or len(observed_rows) != EXPECTED_BLIND_EXAMPLES
    ):
        raise BlindProtocolV6Error(
            "calibration independent verification requires all 150 rows"
        )
    observed_by_id: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(observed_rows):
        row = _require_mapping(item, field=f"calibration.per_sample[{index}]")
        example_id = row.get("example_id")
        if (
            not isinstance(example_id, str)
            or not example_id
            or example_id in observed_by_id
        ):
            raise BlindProtocolV6Error(
                "calibration per-sample membership is invalid"
            )
        observed_by_id[example_id] = row

    scored_rows: list[dict[str, Any]] = []
    for dataset_row in selection.rows:
        observed = observed_by_id.get(dataset_row.example_id)
        if observed is None:
            raise BlindProtocolV6Error(
                "calibration per-sample membership differs from frozen split"
            )
        generation = _require_mapping(
            observed.get("generation"),
            field=f"calibration.{dataset_row.example_id}.generation",
        )
        try:
            result = pointer_eval.GenerationResultV6(
                raw_pointer=str(generation["raw_pointer"]),
                finish_reason=str(generation["finish_reason"]),
                finish_category=str(generation["finish_category"]),
                latency_ms=float(generation["latency_ms"]),
                input_tokens=(
                    None
                    if generation.get("input_tokens") is None
                    else int(generation["input_tokens"])
                ),
                output_tokens=(
                    None
                    if generation.get("output_tokens") is None
                    else int(generation["output_tokens"])
                ),
                generation_error=(
                    None
                    if generation.get("generation_error") is None
                    else str(generation["generation_error"])
                ),
            )
            recomputed = pointer_eval._score_row(
                row=dataset_row,
                generation=result,
                bindings={},
                backend_mode="hf_model",
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            pointer_eval.PointerHFEvalV6Error,
        ) as exc:
            raise BlindProtocolV6Error(
                f"calibration row could not be independently recompiled: "
                f"{dataset_row.example_id}"
            ) from exc
        for field in (
            "expected",
            "generation",
            "compilation",
            "pointer_metrics",
            "compiled_metrics",
        ):
            if observed.get(field) != recomputed.get(field):
                raise BlindProtocolV6Error(
                    f"calibration row differs from independent recompilation: "
                    f"{dataset_row.example_id}.{field}"
                )
        scored_rows.append(recomputed)

    try:
        recomputed_rows, recomputed_summary = calibration_eval_v6._recompute_summary(
            scored_rows,
            backend_mode="hf_model",
            model_bound=True,
        )
    except calibration_eval_v6.CalibrationEvalV6Error as exc:
        raise BlindProtocolV6Error(
            "calibration metrics failed independent recomputation"
        ) from exc
    if recomputed_summary != observed_summary:
        raise BlindProtocolV6Error(
            "calibration summary differs from independently recomputed metrics"
        )
    for observed, recomputed in zip(
        (observed_by_id[row.example_id] for row in selection.rows),
        recomputed_rows,
        strict=True,
    ):
        for field, expected in recomputed.items():
            if field == "source_sample_sha256":
                _require_sha256(
                    observed.get(field),
                    field="calibration source sample SHA-256",
                )
            elif observed.get(field) != expected:
                raise BlindProtocolV6Error(
                    "calibration enriched row differs from independent recomputation"
                )
    return {
        "status": "PASS_CALIBRATION_RECEIPT_V6_INDEPENDENTLY_RECOMPUTED",
        "receipt_sha256": receipt_sha256,
        "dataset_calibration_sha256": selection.split_sha256,
        "samples_recompiled": len(scored_rows),
        "per_sample_sha256": sha256_bytes(sample_payload),
        "summary_sha256": sha256_bytes(summary_payload),
        "model_bound": True,
        "complete_split": True,
        "blind_data_accessed": False,
    }


def _verify_calibration(
    path: Path,
    expected_sha256: str,
    *,
    common: Mapping[str, str],
    selection_freeze_sha256: str,
    dataset: Mapping[str, Any],
    backend_mode: str,
    selection_freeze_path: Path | None = None,
    base_model_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    resolved, payload, receipt = _load_json(
        path,
        label="complete calibration receipt",
        expected_sha256=expected_sha256,
    )
    if receipt.get("schema") != CALIBRATION_SCHEMA:
        raise BlindProtocolV6Error("calibration receipt schema is invalid")
    allowed_status = (
        {CALIBRATION_FIXTURE_STATUS, CALIBRATION_STATUS}
        if backend_mode == "fixture"
        else {CALIBRATION_STATUS}
    )
    if receipt.get("status") not in allowed_status:
        raise BlindProtocolV6Error("calibration did not pass")
    selection = _require_mapping(
        receipt.get("selection_freeze"),
        field="calibration.selection_freeze",
    )
    if (
        selection.get("sha256") != selection_freeze_sha256
        or selection.get("validated_schema") != SELECTION_FREEZE_SCHEMA
        or selection.get("validated_status") != SELECTION_FREEZE_STATUS
        or selection.get("base_model_tree_sha256") != common["base_model_tree_sha256"]
        or selection.get("adapter_tree_sha256") != common["adapter_tree_sha256"]
    ):
        raise BlindProtocolV6Error("calibration selection-freeze binding mismatch")
    dataset_record = _require_mapping(
        receipt.get("dataset"),
        field="calibration.dataset",
    )
    calibration_descriptor = dataset["nonblind_descriptors"]["calibration"]
    if (
        dataset_record.get("opened_split") != "calibration"
        or dataset_record.get("sha256") != calibration_descriptor["sha256"]
        or dataset_record.get("rows") != EXPECTED_BLIND_EXAMPLES
        or dataset_record.get("complete_split") is not True
        or dataset_record.get("calibration_opened_after_freeze_validation") is not True
        or dataset_record.get("blind_data_accessed") is not False
    ):
        raise BlindProtocolV6Error("calibration is not a complete 150-row run")
    execution = _require_mapping(
        receipt.get("execution"),
        field="calibration.execution",
    )
    if (
        execution.get("fixed_inference_contract") != calibration_eval_v6.FIXED_INFERENCE_CONTRACT
        or execution.get("expected_passed_to_model") is not False
        or execution.get("expected_passed_to_candidate_compiler") is not False
        or execution.get("gold_repair_applied") is not False
        or execution.get("checkpoint_reselection_performed") is not False
        or execution.get("blind_supported") is not False
        or execution.get("blind_data_accessed") is not False
    ):
        raise BlindProtocolV6Error("calibration fixed inference/access contract is invalid")
    model = _require_mapping(
        receipt.get("model"),
        field="calibration.model",
    )
    model_bound = execution.get("model_bound") is True
    if backend_mode == "hf_model":
        base = _require_mapping(model.get("base"), field="calibration.model.base")
        adapter = _require_mapping(
            model.get("adapter"),
            field="calibration.model.adapter",
        )
        if (
            execution.get("backend") != "hf_model"
            or not model_bound
            or base.get("tree_sha256") != common["base_model_tree_sha256"]
            or adapter.get("tree_sha256") != common["adapter_tree_sha256"]
            or model.get("fixture_not_model_evidence") is not False
        ):
            raise BlindProtocolV6Error("calibration is not bound to the frozen model")
    elif execution.get("backend") != "fixture":
        raise BlindProtocolV6Error("fixture calibration backend is invalid")

    output_dir = resolved.parent
    if {item.name for item in output_dir.iterdir()} != {
        "per_sample.v6.jsonl",
        "summary.v6.json",
        "receipt.v6.json",
    }:
        raise BlindProtocolV6Error("calibration output must contain exactly three artifacts")
    _, summary_payload, summary = _load_json(
        output_dir / "summary.v6.json",
        label="calibration summary",
    )
    if (
        summary.get("schema") != calibration_eval_v6.SUMMARY_SCHEMA
        or summary.get("status") != receipt.get("status")
        or summary.get("rows") != EXPECTED_BLIND_EXAMPLES
        or summary.get("complete_split") is not True
        or summary.get("quality_gate_passed") is not True
        or any(
            gate.get("passed") is not True
            for gate in summary.get("quality_gates", [])
            if isinstance(gate, Mapping)
        )
        or not summary.get("quality_gates")
    ):
        raise BlindProtocolV6Error("calibration quality gates are incomplete")
    authorization = _require_mapping(
        summary.get("authorization"),
        field="calibration.summary.authorization",
    )
    if (
        authorization.get("checkpoint_reselection_allowed") is not False
        or authorization.get("blind_test_authorized") is not False
        or authorization.get("deployment_authorized") is not False
    ):
        raise BlindProtocolV6Error("calibration summary changes post-freeze authorization")
    artifacts = _require_mapping(
        receipt.get("artifacts"),
        field="calibration.artifacts",
    )
    sample_record = _require_mapping(
        artifacts.get("per_sample.v6.jsonl"),
        field="calibration per-sample artifact",
    )
    summary_record = _require_mapping(
        artifacts.get("summary.v6.json"),
        field="calibration summary artifact",
    )
    if (
        set(artifacts) != {"per_sample.v6.jsonl", "summary.v6.json"}
        or sample_record.get("rows") != EXPECTED_BLIND_EXAMPLES
        or sample_record.get("sha256") != sha256_file(output_dir / "per_sample.v6.jsonl")
        or summary_record.get("sha256") != sha256_bytes(summary_payload)
    ):
        raise BlindProtocolV6Error("calibration artifact evidence hash mismatch")
    result = {
        "path": str(resolved),
        "sha256": sha256_bytes(payload),
        "schema": receipt["schema"],
        "status": receipt["status"],
        "model_bound": model_bound,
        "conformal_threshold": summary["conformal"]["threshold"],
        "per_sample_sha256": sample_record["sha256"],
        "summary_sha256": summary_record["sha256"],
    }
    if backend_mode == "hf_model":
        result["authoritative_verification"] = (
            _verify_calibration_receipt_recomputed_v6(
            receipt_path=resolved,
            receipt_sha256=result["sha256"],
            dataset_dir=Path(dataset["root"]),
        )
        )
    return result


def _verify_ablation(
    path: Path,
    expected_sha256: str,
    *,
    common: Mapping[str, str],
    dataset: Mapping[str, Any],
    selection_freeze_path: Path,
    selection_freeze_sha256: str,
    base_model_dir: Path,
    checkpoint_dir: Path,
    backend_mode: str,
) -> dict[str, Any]:
    resolved, payload, receipt = _load_json(
        path,
        label="ablation closure receipt",
        expected_sha256=expected_sha256,
    )
    if receipt.get("schema") != ABLATION_SCHEMA:
        raise BlindProtocolV6Error("ablation closure schema is invalid")
    if receipt.get("status") != ABLATION_STATUS:
        raise BlindProtocolV6Error("ablations are not closed")
    dataset_record = _require_mapping(
        receipt.get("dataset"),
        field="ablation.dataset",
    )
    split = dataset_record.get("split")
    if split != "validation":
        raise BlindProtocolV6Error("ablation must use the complete validation split")
    descriptor = dataset["nonblind_descriptors"]["validation"]
    if (
        Path(str(dataset_record.get("directory"))).resolve() != dataset["root"]
        or dataset_record.get("opened_split_sha256") != descriptor["sha256"]
        or dataset_record.get("rows_in_file") != EXPECTED_BLIND_EXAMPLES
        or dataset_record.get("rows_evaluated") != EXPECTED_BLIND_EXAMPLES
        or dataset_record.get("max_samples") is not None
        or dataset_record.get("validation_complete_only") is not True
        or dataset_record.get("calibration_opened") is not False
        or dataset_record.get("sealed_blind_opened") is not False
    ):
        raise BlindProtocolV6Error(
            "ablation did not close the complete 150-row validation split"
        )
    execution = _require_mapping(
        receipt.get("execution"),
        field="ablation.execution",
    )
    if (
        execution.get("seed") != FIXED_SEED
        or execution.get("subjects") != ["base", "adapter"]
        or execution.get("generation_variants") != list(ablation_eval_v6.GENERATION_VARIANTS)
        or execution.get("compiler_only_variant") != ablation_eval_v6.COMPILER_ONLY_VARIANT
        or execution.get("same_requests_for_base_and_adapter") is not True
        or execution.get("expected_passed_to_model") is not False
        or execution.get("expected_passed_to_candidate_compiler") is not False
        or execution.get("synthetic_evidence_added") is not False
        or execution.get("selection_policy_called") is not False
        or execution.get("automatic_model_selection") is not False
        or execution.get("promotion_authorized") is not False
        or execution.get("production_state_modified") is not False
    ):
        raise BlindProtocolV6Error("ablation execution boundary is invalid")
    if execution.get("backend_mode") != backend_mode:
        raise BlindProtocolV6Error("ablation backend differs from blind backend")
    backend_bindings = _require_mapping(
        receipt.get("backend_bindings"),
        field="ablation.backend_bindings",
    )
    if backend_mode == "hf_model":
        adapter_backend = _require_mapping(
            backend_bindings.get("adapter"),
            field="ablation adapter backend",
        )
        model = _require_mapping(
            adapter_backend.get("model"),
            field="ablation adapter model",
        )
        base = _require_mapping(model.get("base"), field="ablation base model")
        adapter = _require_mapping(
            model.get("adapter"),
            field="ablation selected adapter",
        )
        if (
            base.get("tree_sha256") != common["base_model_tree_sha256"]
            or adapter.get("tree_sha256") != common["adapter_tree_sha256"]
        ):
            raise BlindProtocolV6Error("ablation model hashes differ from frozen selection")

    output_dir = resolved.parent
    report_names = {
        "sample_results.v6.jsonl",
        "raw_vs_compiler.v6.json",
        "evidence_order_sensitivity.v6.json",
        "decoy_sensitivity.v6.json",
        "provenance_removal.v6.json",
        "stratified_metrics.v6.json",
        "base_vs_adapter.v6.json",
    }
    if {item.name for item in output_dir.iterdir()} != {
        *report_names,
        "run_receipt.v6.json",
    }:
        raise BlindProtocolV6Error("ablation output artifact membership is incomplete")
    artifacts = _require_mapping(
        receipt.get("artifacts"),
        field="ablation.artifacts",
    )
    if set(artifacts) != report_names:
        raise BlindProtocolV6Error("ablation artifact inventory is invalid")
    for name in report_names:
        record = _require_mapping(
            artifacts.get(name),
            field=f"ablation artifact {name}",
        )
        if record.get("sha256") != sha256_file(output_dir / name):
            raise BlindProtocolV6Error(f"ablation artifact hash mismatch: {name}")
    provenance = _load_json(
        output_dir / "provenance_removal.v6.json",
        label="ablation provenance report",
    )[2]
    base_adapter = _load_json(
        output_dir / "base_vs_adapter.v6.json",
        label="ablation base-adapter report",
    )[2]
    if (
        provenance.get("status") != "PASS_TRUSTED_PROVENANCE_REMOVAL_FAILS_CLOSED"
        or base_adapter.get("status") != "PASS_IDENTICAL_INPUT_CONTRACT_DIAGNOSTIC_ONLY"
        or base_adapter.get("automatic_model_selection") is not False
    ):
        raise BlindProtocolV6Error("ablation invariant report did not pass")
    result = {
        "path": str(resolved),
        "sha256": sha256_bytes(payload),
        "schema": receipt["schema"],
        "status": receipt["status"],
        "split": split,
        "backend_mode": backend_mode,
        "artifact_hashes": {name: artifacts[name]["sha256"] for name in sorted(report_names)},
    }
    if backend_mode == "hf_model":
        result["authoritative_verification"] = _run_authoritative_receipt_verifier(
            module=ablation_eval_v6,
            function_name="verify_ablation_receipt_v6",
            label="ablation",
            receipt_path=resolved,
            receipt_sha256=result["sha256"],
            selection_freeze_path=selection_freeze_path,
            selection_freeze_sha256=selection_freeze_sha256,
            dataset_dir=Path(dataset["root"]),
            base_model_dir=base_model_dir,
            checkpoint_dir=checkpoint_dir,
            adapter_dir=checkpoint_dir,
        )
    return result


def _verify_contract_set(
    *,
    workspace_root: Path,
    dataset_manifest_path: Path,
    dataset_manifest_sha256: str,
    task_contract_path: Path,
    task_contract_sha256: str,
    preprocessing_contract_path: Path,
    preprocessing_contract_sha256: str,
    contract_build_receipt_path: Path,
    contract_build_receipt_sha256: str,
) -> dict[str, Any]:
    task_path, _, task = _load_json(
        task_contract_path,
        label="task inference contract",
        expected_sha256=task_contract_sha256,
    )
    preprocessing_path, _, preprocessing = _load_json(
        preprocessing_contract_path,
        label="preprocessing inference contract",
        expected_sha256=preprocessing_contract_sha256,
    )
    build_path, _, build = _load_json(
        contract_build_receipt_path,
        label="inference contract build receipt",
        expected_sha256=contract_build_receipt_sha256,
    )
    if task_path.parent != preprocessing_path.parent or task_path.parent != build_path.parent:
        raise BlindProtocolV6Error("the three inference contract files must share one directory")
    if (
        task_path.name != contracts_v6.TASK_CONTRACT_FILENAME
        or preprocessing_path.name != contracts_v6.PREPROCESSING_CONTRACT_FILENAME
        or build_path.name != contracts_v6.BUILD_RECEIPT_FILENAME
    ):
        raise BlindProtocolV6Error("inference contract filenames are invalid")
    try:
        verified = contracts_v6.verify_contracts_v6(
            workspace_root=workspace_root,
            dataset_manifest=dataset_manifest_path,
            contract_dir=task_path.parent,
        )
    except (contracts_v6.ContractsV6Error, OSError, ValueError) as exc:
        raise BlindProtocolV6Error("fixed v6 inference contracts failed verification") from exc
    if (
        verified.get("status") != "PASS_V6_CONTRACTS_VERIFIED"
        or verified.get("dataset_manifest_sha256") != dataset_manifest_sha256
        or build.get("status") != "PASS_V6_CONTRACTS_CREATED_NO_MODEL_EXECUTION"
        or task.get("status") != "FROZEN_BEFORE_CALIBRATION_AND_BLIND"
        or preprocessing.get("status") != "FROZEN_BEFORE_CALIBRATION_AND_BLIND"
    ):
        raise BlindProtocolV6Error("fixed inference contract gate is invalid")
    decoding = _require_mapping(
        preprocessing.get("decoding"),
        field="preprocessing.decoding",
    )
    if dict(decoding) != {
        "algorithm": "greedy",
        "batch_size": 1,
        "do_sample": False,
        "max_input_tokens": FIXED_MAX_INPUT_TOKENS,
        "max_new_tokens": FIXED_MAX_NEW_TOKENS,
        "seed": FIXED_SEED,
        "singleton_batch": True,
    }:
        raise BlindProtocolV6Error("fixed inference decoding contract changed")
    selection_policy = _require_mapping(
        preprocessing.get("selection_policy"),
        field="preprocessing.selection_policy",
    )
    if (
        selection_policy.get("calibration_may_reselect_checkpoint") is not False
        or selection_policy.get("blind_may_reselect_checkpoint") is not False
    ):
        raise BlindProtocolV6Error("inference contract permits post-freeze reselection")
    split_policy = _require_mapping(
        preprocessing.get("split_access_policy"),
        field="preprocessing.split_access_policy",
    )
    blind_policy = _require_mapping(
        split_policy.get("blind_test"),
        field="preprocessing.split_access_policy.blind_test",
    )
    if (
        blind_policy.get("one_time_model_bound_authorization_required") is not True
        or blind_policy.get("authorization_reusable_after_failure") is not False
        or blind_policy.get("eligible_for_parameter_fitting") is not False
        or blind_policy.get("eligible_for_checkpoint_selection") is not False
    ):
        raise BlindProtocolV6Error("inference blind policy is invalid")
    contract_set_sha = _require_sha256(
        build.get("contract_set_sha256"),
        field="contract_set_sha256",
    )
    if verified.get("contract_set_sha256") != contract_set_sha:
        raise BlindProtocolV6Error("contract set digest mismatch")
    return {
        "directory": str(task_path.parent),
        "task_contract": {
            "path": str(task_path),
            "sha256": sha256_file(task_path),
        },
        "preprocessing_contract": {
            "path": str(preprocessing_path),
            "sha256": sha256_file(preprocessing_path),
        },
        "build_receipt": {
            "path": str(build_path),
            "sha256": sha256_file(build_path),
        },
        "contract_id": verified["contract_id"],
        "contract_set_sha256": contract_set_sha,
        "verification_status": verified["status"],
    }


def _source_bindings(runner_path: Path) -> dict[str, Any]:
    protocol = Path(__file__).resolve()
    evaluator = Path(pointer_eval.__file__).resolve()
    compiler = Path(pointer_eval.evidence_pointer_v6.__file__).resolve()
    runner = _stable_regular_file(runner_path, label="blind evaluator runner")
    return {
        "protocol": {"path": str(protocol), "sha256": sha256_file(protocol)},
        "evaluator": {"path": str(evaluator), "sha256": sha256_file(evaluator)},
        "compiler": {"path": str(compiler), "sha256": sha256_file(compiler)},
        "runner": {"path": str(runner), "sha256": sha256_file(runner)},
    }


def _validate_run_identity(run_id: str, output_basename: str) -> None:
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise BlindProtocolV6Error("run_id is invalid")
    if (
        not isinstance(output_basename, str)
        or not _OUTPUT_NAME_RE.fullmatch(output_basename)
        or output_basename in {".", ".."}
    ):
        raise BlindProtocolV6Error("output_basename is invalid")


def _execution_binding(
    *,
    backend_mode: str,
    fixture_path: Path | None,
    device: str | None,
    run_id: str,
    output_basename: str,
) -> dict[str, Any]:
    _validate_run_identity(run_id, output_basename)
    if backend_mode not in _SUPPORTED_BACKENDS:
        raise BlindProtocolV6Error(f"backend must be one of {sorted(_SUPPORTED_BACKENDS)}")
    fixture: dict[str, Any] | None = None
    if backend_mode == "fixture":
        if fixture_path is None or device is not None:
            raise BlindProtocolV6Error("fixture backend requires fixture_path and rejects device")
        resolved = _stable_regular_file(
            fixture_path,
            label="blind fixture generations",
        )
        size = resolved.stat().st_size
        if size <= 0 or size > MAX_FIXTURE_BYTES:
            raise BlindProtocolV6Error("fixture byte size is invalid")
        fixture = {
            "path": str(resolved),
            "bytes": size,
            "sha256": sha256_file(resolved),
        }
    else:
        if fixture_path is not None or device not in {"cpu", "cuda"}:
            raise BlindProtocolV6Error("hf_model requires explicit cpu/cuda and rejects fixture")
    return {
        "backend": backend_mode,
        "fixture": fixture,
        "device": device,
        "run_id": run_id,
        "output_basename": output_basename,
        "split": "blind_test",
        "expected_examples": EXPECTED_BLIND_EXAMPLES,
        "max_samples": None,
        "ablations": ["none"],
        "decoding": {
            "algorithm": "greedy",
            "do_sample": False,
            "singleton_batch": True,
            "batch_size": 1,
            "seed": FIXED_SEED,
            "max_input_tokens": FIXED_MAX_INPUT_TOKENS,
            "max_new_tokens": FIXED_MAX_NEW_TOKENS,
        },
        "blind_use": {
            "model_selection_allowed": False,
            "checkpoint_ranking_allowed": False,
            "threshold_tuning_allowed": False,
            "calibration_allowed": False,
            "gguf_release_qualification_allowed": True,
            "deployment_authorization_allowed": False,
        },
    }


def _exclusive_create(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # Partial claims remain intentionally non-reusable.
        raise


def _authorization_output(
    dataset_root: Path,
    authorization_path: Path,
) -> Path:
    output = Path(authorization_path)
    final = output.resolve() if output.is_absolute() else (dataset_root / output).resolve()
    try:
        final.parent.relative_to(dataset_root)
    except ValueError as exc:
        raise BlindProtocolV6Error("authorization output must be inside dataset directory") from exc
    if final.parent != dataset_root:
        raise BlindProtocolV6Error("authorization output must be a direct dataset child")
    if final.suffix.casefold() != ".json":
        raise BlindProtocolV6Error("authorization output must be JSON")
    if final.exists() or final.is_symlink():
        raise BlindProtocolV6Error("authorization output exists and cannot be overwritten")
    return final


def _global_registry_paths(
    *,
    dataset_root: Path,
    registry_root: Path,
    blind_sha256: str,
) -> dict[str, Path]:
    raw = Path(registry_root)
    if raw.is_symlink():
        raise BlindProtocolV6Error("global blind registry root must not be a symlink")
    try:
        root = raw.resolve(strict=True)
    except FileNotFoundError as exc:
        raise BlindProtocolV6Error(
            "caller-specified global blind registry root is unavailable"
        ) from exc
    if not root.is_dir():
        raise BlindProtocolV6Error("global blind registry root must be a directory")
    dataset = Path(dataset_root).resolve(strict=True)
    try:
        root.relative_to(dataset)
    except ValueError:
        pass
    else:
        raise BlindProtocolV6Error(
            "global blind registry root must be outside the dataset directory"
        )
    prefix = _require_sha256(
        blind_sha256,
        field="global registry blind SHA-256",
    )
    paths = {
        "root": root,
        "reservation": root / f"{prefix}.registry.v6.json",
        "claim": root / f"{prefix}.claim.v6.json",
        "terminal": root / f"{prefix}.terminal.v6.json",
    }
    for label, path in paths.items():
        if label == "root":
            continue
        if path.parent != root or path.is_symlink():
            raise BlindProtocolV6Error(f"global blind registry {label} path is unsafe")
    return paths


def authorize_blind_evaluation(
    *,
    workspace_root: Path,
    dataset_dir: Path,
    base_model_dir: Path,
    adapter_dir: Path,
    selection_freeze_path: Path,
    selection_freeze_sha256: str,
    calibration_receipt_path: Path,
    calibration_receipt_sha256: str,
    ablation_receipt_path: Path,
    ablation_receipt_sha256: str,
    task_contract_path: Path,
    task_contract_sha256: str,
    preprocessing_contract_path: Path,
    preprocessing_contract_sha256: str,
    contract_build_receipt_path: Path,
    contract_build_receipt_sha256: str,
    runner_path: Path,
    backend_mode: str,
    fixture_path: Path | None,
    device: str | None,
    run_id: str,
    output_basename: str,
    authorization_path: Path,
    registry_root: Path,
) -> dict[str, Any]:
    """Create the sole immutable authorization without parsing blind JSONL."""

    dataset = _dataset_metadata(dataset_dir, verify_blind_hash=False)
    base = _tree_inventory(base_model_dir, label="base model")
    checkpoint = _tree_inventory(adapter_dir, label="selected checkpoint")
    adapter = _adapter_inventory(adapter_dir, label="selected adapter")
    common = _common_bindings(
        dataset_manifest_sha256=dataset["manifest_sha256"],
        base_model_tree_sha256=base["tree_sha256"],
        adapter_tree_sha256=adapter["tree_sha256"],
    )
    contracts = _verify_contract_set(
        workspace_root=workspace_root,
        dataset_manifest_path=dataset["manifest_path"],
        dataset_manifest_sha256=dataset["manifest_sha256"],
        task_contract_path=task_contract_path,
        task_contract_sha256=task_contract_sha256,
        preprocessing_contract_path=preprocessing_contract_path,
        preprocessing_contract_sha256=preprocessing_contract_sha256,
        contract_build_receipt_path=contract_build_receipt_path,
        contract_build_receipt_sha256=contract_build_receipt_sha256,
    )
    selection_sha = _require_sha256(
        selection_freeze_sha256,
        field="selection freeze SHA-256",
    )
    selection = _verify_selection_freeze(
        selection_freeze_path,
        selection_sha,
        common=common,
    )
    checkpoint_root = str(Path(adapter_dir).resolve(strict=True))
    if (
        selection["selected_checkpoint_path"] != checkpoint_root
        or selection["selected_adapter_path"] != checkpoint_root
        or selection["selected_checkpoint_tree_sha256"]
        != checkpoint["tree_sha256"]
        or selection["selected_adapter_tree_sha256"] != adapter["tree_sha256"]
    ):
        raise BlindProtocolV6Error(
            "selected checkpoint or two-file adapter inventory differs from selection"
        )
    calibration_sha = _require_sha256(
        calibration_receipt_sha256,
        field="calibration receipt SHA-256",
    )
    calibration = _verify_calibration(
        calibration_receipt_path,
        calibration_sha,
        common=common,
        selection_freeze_sha256=selection_sha,
        selection_freeze_path=selection_freeze_path,
        dataset=dataset,
        base_model_dir=base_model_dir,
        checkpoint_dir=adapter_dir,
        backend_mode=backend_mode,
    )
    ablation = _verify_ablation(
        ablation_receipt_path,
        ablation_receipt_sha256,
        common=common,
        dataset=dataset,
        selection_freeze_path=selection_freeze_path,
        selection_freeze_sha256=selection_sha,
        base_model_dir=base_model_dir,
        checkpoint_dir=adapter_dir,
        backend_mode=backend_mode,
    )
    code = _source_bindings(runner_path)
    execution = _execution_binding(
        backend_mode=backend_mode,
        fixture_path=fixture_path,
        device=device,
        run_id=run_id,
        output_basename=output_basename,
    )
    binding = {
        "dataset": {
            "directory": str(dataset["root"]),
            "manifest_path": str(dataset["manifest_path"]),
            "manifest_sha256": dataset["manifest_sha256"],
            "blind_path": str(dataset["blind_path"]),
            "blind_sha256": dataset["blind_sha256"],
            "blind_bytes": dataset["blind_bytes"],
            "blind_count": dataset["blind_count"],
            "seal_path": str(dataset["seal_path"]),
            "seal_sha256": dataset["seal_sha256"],
            "authorization_stage_jsonl_parsed": False,
        },
        "model": {
            "base_model_path": base["path"],
            "base_model_tree_sha256": base["tree_sha256"],
            "checkpoint_path": checkpoint["path"],
            "checkpoint_tree_sha256": checkpoint["tree_sha256"],
            "adapter_path": checkpoint["path"],
            "adapter_tree_sha256": adapter["tree_sha256"],
            "adapter_file_count": adapter["file_count"],
        },
        "gates": {
            "selection_freeze": selection,
            "calibration": calibration,
            "ablation": ablation,
            "inference_contracts": contracts,
        },
        "code": code,
        "execution": execution,
        "release_qualification_policy": json.loads(canonical_json(RELEASE_QUALIFICATION_POLICY)),
    }
    authorization_id = "icmat-v6-blind-" + sha256_bytes(canonical_json(binding).encode("utf-8"))[:32]
    nonce = secrets.token_hex(32)
    registry = _global_registry_paths(
        dataset_root=dataset["root"],
        registry_root=registry_root,
        blind_sha256=dataset["blind_sha256"],
    )
    receipt = {
        "schema": AUTHORIZATION_SCHEMA,
        "version": AUTHORIZATION_VERSION,
        "authorization_id": authorization_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": AUTHORIZATION_STATUS,
        "sealed": True,
        "revoked": False,
        **binding,
        "consumption": {
            "once": True,
            "nonce": nonce,
            "nonce_sha256": sha256_bytes(nonce.encode("ascii")),
            "registry_root": str(registry["root"]),
            "registry_path": str(registry["reservation"]),
            "claim_path": str(registry["claim"]),
            "terminal_path": str(registry["terminal"]),
            "claim_must_precede_blind_parse": True,
            "claim_must_precede_blind_hash": True,
            "claim_and_terminal_overwrite_allowed": False,
            "failure_is_non_reusable": True,
        },
        "claim_boundary": (
            "Exactly one full 150-row blind_test evaluation is authorized for "
            "the frozen model and fixed inference contract. Blind results may "
            "qualify one offline GGUF artifact build only when every threshold "
            "frozen in this authorization passes. They cannot select a model, "
            "rank checkpoints, fit or tune thresholds, recalibrate, authorize "
            "activation/deployment/production, or be replayed after success "
            "or failure."
        ),
    }
    final = _authorization_output(
        dataset["root"],
        authorization_path,
    )
    payload = _json_bytes(receipt)
    authorization_sha = sha256_bytes(payload)
    registry_payload = _json_bytes(
        {
            "schema": REGISTRY_SCHEMA,
            "status": "AUTHORIZATION_RESERVED_IMMUTABLY",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "authorization_id": authorization_id,
            "authorization_path": str(final),
            "authorization_sha256": authorization_sha,
            "blind_sha256": dataset["blind_sha256"],
            "registry_root": str(registry["root"]),
            "claim_path": str(registry["claim"]),
            "terminal_path": str(registry["terminal"]),
            "nonce_sha256": receipt["consumption"]["nonce_sha256"],
            "one_authorization_per_sealed_blind_hash": True,
            "overwrite_allowed": False,
        }
    )
    try:
        _exclusive_create(registry["reservation"], registry_payload)
    except FileExistsError as exc:
        raise BlindProtocolV6Error("this sealed blind hash already has an authorization reservation") from exc
    try:
        _exclusive_create(final, payload)
    except BaseException:
        # The reservation intentionally remains, preventing a second issuance.
        raise
    return {
        "status": "BLIND_AUTHORIZATION_CREATED_NOT_CONSUMED",
        "path": str(final),
        "sha256": authorization_sha,
        "authorization_id": authorization_id,
        "registry_path": str(registry["reservation"]),
        "registry_sha256": sha256_file(registry["reservation"]),
        "blind_jsonl_parsed": False,
        "authorization": receipt,
    }


def _load_authorization(
    authorization_path: Path,
    expected_sha256: str,
) -> tuple[Path, dict[str, Any], str]:
    resolved, payload, receipt = _load_json(
        authorization_path,
        label="blind authorization",
        expected_sha256=expected_sha256,
    )
    actual_sha = sha256_bytes(payload)
    required = {
        "schema",
        "version",
        "authorization_id",
        "created_at_utc",
        "status",
        "sealed",
        "revoked",
        "dataset",
        "model",
        "gates",
        "code",
        "execution",
        "release_qualification_policy",
        "consumption",
        "claim_boundary",
    }
    if set(receipt) != required:
        raise BlindProtocolV6Error("blind authorization fields do not match the v6 contract")
    if (
        receipt.get("schema") != AUTHORIZATION_SCHEMA
        or receipt.get("version") != AUTHORIZATION_VERSION
        or receipt.get("status") != AUTHORIZATION_STATUS
        or receipt.get("sealed") is not True
        or receipt.get("revoked") is not False
    ):
        raise BlindProtocolV6Error("blind authorization is not active")
    authorization_id = receipt.get("authorization_id")
    if (
        not isinstance(authorization_id, str)
        or not authorization_id.startswith("icmat-v6-blind-")
        or len(authorization_id) != len("icmat-v6-blind-") + 32
    ):
        raise BlindProtocolV6Error("authorization_id is invalid")
    binding = {
        key: receipt[key]
        for key in (
            "dataset",
            "model",
            "gates",
            "code",
            "execution",
            "release_qualification_policy",
        )
    }
    if receipt["release_qualification_policy"] != RELEASE_QUALIFICATION_POLICY:
        raise BlindProtocolV6Error("blind release qualification thresholds changed after authorization")
    expected_id = "icmat-v6-blind-" + sha256_bytes(canonical_json(binding).encode("utf-8"))[:32]
    if authorization_id != expected_id:
        raise BlindProtocolV6Error("authorization_id does not match its immutable bindings")
    return resolved, receipt, actual_sha


def _verify_registry(
    *,
    dataset_root: Path,
    registry_root: Path,
    blind_sha256: str,
    authorization_path: Path,
    authorization_sha256: str,
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    paths = _global_registry_paths(
        dataset_root=dataset_root,
        registry_root=registry_root,
        blind_sha256=blind_sha256,
    )
    path = paths["reservation"]
    resolved, payload, registry = _load_json(
        path,
        label="blind authorization registry",
    )
    consumption = _require_mapping(
        authorization.get("consumption"),
        field="authorization.consumption",
    )
    expected = {
        "schema": REGISTRY_SCHEMA,
        "status": "AUTHORIZATION_RESERVED_IMMUTABLY",
        "created_at_utc": registry.get("created_at_utc"),
        "authorization_id": authorization["authorization_id"],
        "authorization_path": str(authorization_path),
        "authorization_sha256": authorization_sha256,
        "blind_sha256": blind_sha256,
        "registry_root": str(paths["root"]),
        "claim_path": str(paths["claim"]),
        "terminal_path": str(paths["terminal"]),
        "nonce_sha256": consumption.get("nonce_sha256"),
        "one_authorization_per_sealed_blind_hash": True,
        "overwrite_allowed": False,
    }
    if registry != expected or not isinstance(
        registry.get("created_at_utc"),
        str,
    ):
        raise BlindProtocolV6Error("blind authorization registry binding mismatch")
    return {
        "path": str(resolved),
        "sha256": sha256_bytes(payload),
    }


def _verify_preclaim_binding(
    *,
    authorization_path: Path,
    expected_authorization_sha256: str,
    workspace_root: Path,
    dataset_dir: Path,
    base_model_dir: Path,
    adapter_dir: Path,
    runner_path: Path,
    backend_mode: str,
    fixture_path: Path | None,
    device: str | None,
    output_dir: Path,
    registry_root: Path,
) -> dict[str, Any]:
    auth_path, authorization, authorization_sha = _load_authorization(
        authorization_path,
        expected_authorization_sha256,
    )
    # Pre-claim verification checks sealed metadata and stat only. The blind
    # bytes are read exactly once, after the immutable claim, below.
    dataset = _dataset_metadata(dataset_dir, verify_blind_hash=False)
    bound_dataset = _require_mapping(
        authorization.get("dataset"),
        field="authorization.dataset",
    )
    expected_dataset = {
        "directory": str(dataset["root"]),
        "manifest_path": str(dataset["manifest_path"]),
        "manifest_sha256": dataset["manifest_sha256"],
        "blind_path": str(dataset["blind_path"]),
        "blind_sha256": dataset["blind_sha256"],
        "blind_bytes": dataset["blind_bytes"],
        "blind_count": dataset["blind_count"],
        "seal_path": str(dataset["seal_path"]),
        "seal_sha256": dataset["seal_sha256"],
        "authorization_stage_jsonl_parsed": False,
    }
    if dict(bound_dataset) != expected_dataset:
        raise BlindProtocolV6Error("authorization dataset binding mismatch")
    if auth_path.parent != dataset["root"]:
        raise BlindProtocolV6Error("authorization is not a direct child of bound dataset")

    base = _tree_inventory(base_model_dir, label="base model")
    checkpoint = _tree_inventory(adapter_dir, label="selected checkpoint")
    adapter = _adapter_inventory(adapter_dir, label="selected adapter")
    bound_model = _require_mapping(
        authorization.get("model"),
        field="authorization.model",
    )
    if dict(bound_model) != {
        "base_model_path": base["path"],
        "base_model_tree_sha256": base["tree_sha256"],
        "checkpoint_path": checkpoint["path"],
        "checkpoint_tree_sha256": checkpoint["tree_sha256"],
        "adapter_path": checkpoint["path"],
        "adapter_tree_sha256": adapter["tree_sha256"],
        "adapter_file_count": adapter["file_count"],
    }:
        raise BlindProtocolV6Error("authorization model binding mismatch")

    gates = _require_mapping(
        authorization.get("gates"),
        field="authorization.gates",
    )
    contract_binding = _require_mapping(
        gates.get("inference_contracts"),
        field="authorization.gates.inference_contracts",
    )
    task_binding = _require_mapping(
        contract_binding.get("task_contract"),
        field="authorization task contract",
    )
    preprocessing_binding = _require_mapping(
        contract_binding.get("preprocessing_contract"),
        field="authorization preprocessing contract",
    )
    build_binding = _require_mapping(
        contract_binding.get("build_receipt"),
        field="authorization contract build receipt",
    )
    contracts = _verify_contract_set(
        workspace_root=workspace_root,
        dataset_manifest_path=dataset["manifest_path"],
        dataset_manifest_sha256=dataset["manifest_sha256"],
        task_contract_path=Path(str(task_binding.get("path"))),
        task_contract_sha256=str(task_binding.get("sha256")),
        preprocessing_contract_path=Path(str(preprocessing_binding.get("path"))),
        preprocessing_contract_sha256=str(preprocessing_binding.get("sha256")),
        contract_build_receipt_path=Path(str(build_binding.get("path"))),
        contract_build_receipt_sha256=str(build_binding.get("sha256")),
    )
    if contracts != dict(contract_binding):
        raise BlindProtocolV6Error("authorization inference contract binding mismatch")
    common = _common_bindings(
        dataset_manifest_sha256=dataset["manifest_sha256"],
        base_model_tree_sha256=base["tree_sha256"],
        adapter_tree_sha256=adapter["tree_sha256"],
    )
    selection_binding = _require_mapping(
        gates.get("selection_freeze"),
        field="authorization selection freeze",
    )
    selection = _verify_selection_freeze(
        Path(str(selection_binding.get("path"))),
        str(selection_binding.get("sha256")),
        common=common,
    )
    if selection != dict(selection_binding):
        raise BlindProtocolV6Error("authorization selection freeze binding mismatch")
    if (
        selection["selected_checkpoint_path"] != checkpoint["path"]
        or selection["selected_adapter_path"] != checkpoint["path"]
        or selection["selected_checkpoint_tree_sha256"]
        != checkpoint["tree_sha256"]
        or selection["selected_adapter_tree_sha256"] != adapter["tree_sha256"]
    ):
        raise BlindProtocolV6Error(
            "authorization checkpoint or two-file adapter binding mismatch"
        )
    calibration_binding = _require_mapping(
        gates.get("calibration"),
        field="authorization calibration",
    )
    calibration = _verify_calibration(
        Path(str(calibration_binding.get("path"))),
        str(calibration_binding.get("sha256")),
        common=common,
        selection_freeze_sha256=selection["sha256"],
        selection_freeze_path=Path(str(selection_binding.get("path"))),
        dataset=dataset,
        base_model_dir=base_model_dir,
        checkpoint_dir=adapter_dir,
        backend_mode=backend_mode,
    )
    if calibration != dict(calibration_binding):
        raise BlindProtocolV6Error("authorization calibration binding mismatch")
    ablation_binding = _require_mapping(
        gates.get("ablation"),
        field="authorization ablation",
    )
    ablation = _verify_ablation(
        Path(str(ablation_binding.get("path"))),
        str(ablation_binding.get("sha256")),
        common=common,
        dataset=dataset,
        selection_freeze_path=Path(str(selection_binding.get("path"))),
        selection_freeze_sha256=selection["sha256"],
        base_model_dir=base_model_dir,
        checkpoint_dir=adapter_dir,
        backend_mode=backend_mode,
    )
    if ablation != dict(ablation_binding):
        raise BlindProtocolV6Error("authorization ablation binding mismatch")
    code = _source_bindings(runner_path)
    if code != authorization.get("code"):
        raise BlindProtocolV6Error("authorization source binding mismatch")
    bound_execution = _require_mapping(
        authorization.get("execution"),
        field="authorization.execution",
    )
    execution = _execution_binding(
        backend_mode=backend_mode,
        fixture_path=fixture_path,
        device=device,
        run_id=str(bound_execution.get("run_id")),
        output_basename=str(bound_execution.get("output_basename")),
    )
    if execution != dict(bound_execution):
        raise BlindProtocolV6Error("authorization execution binding mismatch")
    output = Path(output_dir).resolve()
    if output.name != execution["output_basename"]:
        raise BlindProtocolV6Error("output directory basename differs from authorization")
    if output.exists() or output.is_symlink():
        raise BlindProtocolV6Error("blind output exists and cannot be replayed or overwritten")
    registry_paths = _global_registry_paths(
        dataset_root=dataset["root"],
        registry_root=registry_root,
        blind_sha256=dataset["blind_sha256"],
    )
    consumption = _require_mapping(
        authorization.get("consumption"),
        field="authorization.consumption",
    )
    if (
        consumption.get("registry_root") != str(registry_paths["root"])
        or consumption.get("registry_path")
        != str(registry_paths["reservation"])
        or consumption.get("claim_path") != str(registry_paths["claim"])
        or consumption.get("terminal_path") != str(registry_paths["terminal"])
    ):
        raise BlindProtocolV6Error(
            "authorization global registry paths differ from the caller binding"
        )
    registry = _verify_registry(
        dataset_root=dataset["root"],
        registry_root=registry_root,
        blind_sha256=dataset["blind_sha256"],
        authorization_path=auth_path,
        authorization_sha256=authorization_sha,
        authorization=authorization,
    )
    return {
        "authorization_path": auth_path,
        "authorization": authorization,
        "authorization_sha256": authorization_sha,
        "dataset": dataset,
        "base": base,
        "checkpoint": checkpoint,
        "adapter": adapter,
        "code": code,
        "execution": execution,
        "registry": registry,
        "registry_paths": registry_paths,
        "output": output,
    }


def _claim_consumption(preflight: Mapping[str, Any]) -> dict[str, Any]:
    authorization = preflight["authorization"]
    consumption = _require_mapping(
        authorization.get("consumption"),
        field="authorization.consumption",
    )
    if (
        consumption.get("once") is not True
        or consumption.get("claim_must_precede_blind_parse") is not True
        or consumption.get("claim_must_precede_blind_hash") is not True
        or consumption.get("claim_and_terminal_overwrite_allowed") is not False
        or consumption.get("failure_is_non_reusable") is not True
    ):
        raise BlindProtocolV6Error("authorization consumption policy is invalid")
    nonce = consumption.get("nonce")
    if (
        not isinstance(nonce, str)
        or len(nonce) != 64
        or not re.fullmatch(r"[0-9a-f]{64}", nonce)
        or sha256_bytes(nonce.encode("ascii")) != consumption.get("nonce_sha256")
    ):
        raise BlindProtocolV6Error("authorization nonce is invalid")
    registry_paths = preflight["registry_paths"]
    claim_path = Path(registry_paths["claim"])
    terminal_path = Path(registry_paths["terminal"])
    if terminal_path.exists() or terminal_path.is_symlink():
        raise BlindProtocolV6Error("blind authorization already has a terminal record")
    claim = {
        "schema": CLAIM_SCHEMA,
        "status": "CONSUMED_PENDING_NON_REUSABLE",
        "claimed_at_utc": datetime.now(UTC).isoformat(),
        "authorization_id": authorization["authorization_id"],
        "authorization_path": str(preflight["authorization_path"]),
        "authorization_sha256": preflight["authorization_sha256"],
        "nonce": nonce,
        "nonce_sha256": consumption["nonce_sha256"],
        "run_id": preflight["execution"]["run_id"],
        "output_basename": preflight["execution"]["output_basename"],
        "blind_sha256": preflight["dataset"]["blind_sha256"],
        "failure_is_non_reusable": True,
        "overwrite_allowed": False,
    }
    try:
        _exclusive_create(claim_path, _json_bytes(claim))
    except FileExistsError as exc:
        raise BlindProtocolV6Error("blind authorization has already been claimed") from exc
    return {
        "path": claim_path,
        "sha256": sha256_file(claim_path),
        "receipt": claim,
        "terminal_path": terminal_path,
    }


def _parse_blind_rows_after_claim(
    *,
    dataset: Mapping[str, Any],
    claim: Mapping[str, Any],
) -> pointer_eval.DatasetSelectionV6:
    claim_path = Path(claim["path"])
    if not claim_path.is_file():
        raise BlindProtocolV6Error("blind claim must exist before JSONL parsing")
    try:
        _, _, claim_receipt = _load_json(
            claim_path,
            label="blind consumption claim",
            expected_sha256=str(claim["sha256"]),
        )
    except BlindProtocolV6Error:
        raise
    if (
        claim_receipt.get("schema") != CLAIM_SCHEMA
        or claim_receipt.get("status") != "CONSUMED_PENDING_NON_REUSABLE"
    ):
        raise BlindProtocolV6Error("blind claim is not pending")

    blind_path = Path(dataset["blind_path"])
    observed_ids: set[str] = set()
    rows: list[pointer_eval.DatasetRowV6] = []
    digest = hashlib.sha256()
    try:
        with blind_path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                digest.update(raw_line)
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise BlindProtocolV6Error(f"blind JSONL line {line_number} is not UTF-8") from exc
                if not line.strip():
                    raise BlindProtocolV6Error(f"blind JSONL contains blank line {line_number}")
                try:
                    parsed = json.loads(
                        line,
                        object_pairs_hook=_reject_duplicate_pairs,
                    )
                    row = pointer_eval._validate_dataset_row(
                        parsed,
                        split="blind_test",
                        line_number=line_number,
                    )
                except (
                    json.JSONDecodeError,
                    pointer_eval.PointerHFEvalV6Error,
                ) as exc:
                    raise BlindProtocolV6Error(
                        f"blind JSONL line {line_number} is invalid or SHA-256 mismatch"
                    ) from exc
                if row.example_id in observed_ids:
                    raise BlindProtocolV6Error(f"duplicate blind example_id: {row.example_id}")
                observed_ids.add(row.example_id)
                rows.append(row)
    except OSError as exc:
        raise BlindProtocolV6Error("sealed blind JSONL could not be read after claim") from exc
    if len(rows) != EXPECTED_BLIND_EXAMPLES:
        raise BlindProtocolV6Error(f"blind JSONL must contain exactly {EXPECTED_BLIND_EXAMPLES} rows")
    if digest.hexdigest() != dataset["blind_sha256"]:
        raise BlindProtocolV6Error("blind JSONL changed during parsing")
    return pointer_eval.DatasetSelectionV6(
        dataset_dir=Path(dataset["root"]),
        split_path=blind_path,
        split_sha256=str(dataset["blind_sha256"]),
        split_bytes=int(dataset["blind_bytes"]),
        rows_total=EXPECTED_BLIND_EXAMPLES,
        rows=tuple(rows),
    )


def _load_fixture_after_claim(
    *,
    fixture_path: Path,
    expected_example_ids: Sequence[str],
    expected_sha256: str,
) -> tuple[dict[str, pointer_eval.GenerationResultV6], dict[str, Any]]:
    path = _stable_regular_file(
        fixture_path,
        label="blind fixture generations",
    )
    if sha256_file(path) != expected_sha256:
        raise BlindProtocolV6Error("blind fixture SHA-256 mismatch")
    generations: dict[str, pointer_eval.GenerationResultV6] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise BlindProtocolV6Error(f"fixture contains blank line {line_number}")
            try:
                value = json.loads(
                    line,
                    object_pairs_hook=_reject_duplicate_pairs,
                )
                example_id, result = pointer_eval._validate_fixture_record(
                    value,
                    line_number=line_number,
                )
            except (
                json.JSONDecodeError,
                pointer_eval.PointerHFEvalV6Error,
            ) as exc:
                raise BlindProtocolV6Error(f"fixture line {line_number} is invalid") from exc
            if example_id in generations:
                raise BlindProtocolV6Error(f"duplicate fixture example_id: {example_id}")
            generations[example_id] = result
    expected = set(expected_example_ids)
    observed = set(generations)
    if observed != expected:
        raise BlindProtocolV6Error("fixture membership differs from the 150 claimed blind rows")
    return generations, {
        "mode": "fixture",
        "fixture": {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": expected_sha256,
        },
        "model": {"base": None, "adapter": None},
        "decoding": {
            "recorded_from_fixture": True,
            "max_input_tokens": FIXED_MAX_INPUT_TOKENS,
            "max_new_tokens": FIXED_MAX_NEW_TOKENS,
        },
        "samples_generated": 0,
        "local_files_only": True,
        "network_allowed": False,
        "assistant_target_visible": False,
        "model_quality_claim_allowed": False,
    }


def _blind_summary(
    *,
    rows: Sequence[Mapping[str, Any]],
    selection: pointer_eval.DatasetSelectionV6,
    backend: Mapping[str, Any],
    preflight: Mapping[str, Any],
    claim: Mapping[str, Any],
) -> dict[str, Any]:
    summary = pointer_eval._summarize(
        rows,
        selection=selection,
        backend=backend,
        max_samples=None,
    )
    summary["schema"] = SUMMARY_SCHEMA
    summary["status"] = "BLIND_EVALUATION_COMPLETE_FINAL_REPORT_ONLY"
    summary["authorization"] = {
        "authorization_id": preflight["authorization"]["authorization_id"],
        "authorization_sha256": preflight["authorization_sha256"],
        "claim_path": str(claim["path"]),
        "claim_sha256": claim["sha256"],
        "nonce_sha256": claim["receipt"]["nonce_sha256"],
    }
    summary["selection"]["complete_split"] = True
    summary["execution_boundaries"].update(
        {
            "blind_split_supported": True,
            "blind_data_accessed": True,
            "promotion_authorized": False,
        }
    )
    summary["blind_use_policy"] = {
        "final_report_only": True,
        "model_selection_performed": False,
        "checkpoint_ranking_performed": False,
        "threshold_tuning_performed": False,
        "calibration_performed": False,
        "deployment_authorized": False,
        "replay_allowed": False,
    }
    summary["claim_boundary"] = (
        "ONE_TIME_150_ROW_SEALED_BLIND_FINAL_REPORT_ONLY_NOT_SELECTION_"
        "THRESHOLD_TUNING_CALIBRATION_DEPLOYMENT_X5_OR_PRODUCTION_EVIDENCE"
    )
    return summary


def _ratio_gate(
    *,
    name: str,
    numerator: int,
    denominator: int,
    minimum: Mapping[str, int],
) -> dict[str, Any]:
    required_numerator = int(minimum["numerator"])
    required_denominator = int(minimum["denominator"])
    passed = denominator > 0 and numerator * required_denominator >= required_numerator * denominator
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
    *,
    rows: Sequence[Mapping[str, Any]],
    backend_mode: str,
    calibration_model_bound: bool,
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    answer_rows = [
        row
        for row in rows
        if _require_mapping(
            _require_mapping(row.get("expected"), field="blind sample expected").get(
                "answer"
            ),
            field="blind sample expected answer",
        ).get("decision")
        == "ANSWER"
    ]
    refuse_rows = [
        row
        for row in rows
        if _require_mapping(
            _require_mapping(row.get("expected"), field="blind sample expected").get(
                "answer"
            ),
            field="blind sample expected answer",
        ).get("decision")
        == "REFUSE"
    ]
    if len(answer_rows) + len(refuse_rows) != len(rows):
        raise BlindProtocolV6Error("blind samples contain invalid expected decisions")

    def flag_count(field: str, name: str) -> int:
        total = 0
        for row in rows:
            metrics = _require_mapping(
                row.get(field),
                field=f"blind sample {field}",
            )
            value = metrics.get(name)
            if not isinstance(value, bool):
                raise BlindProtocolV6Error(
                    f"blind sample {field}.{name} must be boolean"
                )
            total += int(value)
        return total

    def predicted_decision(row: Mapping[str, Any]) -> str | None:
        metrics = _require_mapping(
            row.get("pointer_metrics"),
            field="blind sample pointer_metrics",
        )
        if metrics.get("compiler_accepted") is not True:
            return None
        compilation = _require_mapping(
            row.get("compilation"),
            field="blind sample compilation",
        )
        parsed = compilation.get("parsed_pointer")
        return parsed.get("decision") if isinstance(parsed, Mapping) else None

    tp = sum(predicted_decision(row) == "REFUSE" for row in refuse_rows)
    fp = sum(predicted_decision(row) == "REFUSE" for row in answer_rows)
    fn = len(refuse_rows) - tp
    refuse_f1_numerator = 2 * tp
    refuse_f1_denominator = 2 * tp + fp + fn
    return [
        {
            "gate": "HF_MODEL_BOUND_BLIND_RUN",
            "actual_backend": backend_mode,
            "required_backend": policy["required_backend"],
            "calibration_model_bound": calibration_model_bound,
            "passed": (
                backend_mode == policy["required_backend"]
                and calibration_model_bound is True
                and policy["model_bound_required"] is True
            ),
        },
        {
            "gate": "COMPLETE_150",
            "actual": len(rows),
            "required": EXPECTED_BLIND_EXAMPLES,
            "passed": len(rows) == EXPECTED_BLIND_EXAMPLES,
        },
        _ratio_gate(
            name="POINTER_PARSE_VALID_100_PERCENT",
            numerator=flag_count("pointer_metrics", "parse_valid"),
            denominator=len(rows),
            minimum=policy["pointer_parse_valid"],
        ),
        _ratio_gate(
            name="POINTER_COMPILER_ACCEPTED_100_PERCENT",
            numerator=flag_count("pointer_metrics", "compiler_accepted"),
            denominator=len(rows),
            minimum=policy["pointer_compiler_accepted"],
        ),
        {
            "gate": "UNSUPPORTED_WRONG_ANSWER_ZERO",
            "actual": flag_count(
                "compiled_metrics",
                "unsupported_wrong_answer",
            ),
            "required_maximum": int(policy["unsupported_wrong_answer_max_count"]),
            "passed": flag_count(
                "compiled_metrics",
                "unsupported_wrong_answer",
            )
            <= int(policy["unsupported_wrong_answer_max_count"]),
        },
        _ratio_gate(
            name="COMPILED_SCHEMA_VALID_100_PERCENT",
            numerator=flag_count("compiled_metrics", "schema_valid"),
            denominator=len(rows),
            minimum=policy["compiled_schema_valid"],
        ),
        _ratio_gate(
            name="COMPILED_CITATION_EXACT_100_PERCENT",
            numerator=flag_count("compiled_metrics", "citation_exact"),
            denominator=len(rows),
            minimum=policy["compiled_citation_exact"],
        ),
        _ratio_gate(
            name="COMPILED_PROVENANCE_EXACT_100_PERCENT",
            numerator=flag_count("compiled_metrics", "provenance_exact"),
            denominator=len(rows),
            minimum=policy["compiled_provenance_exact"],
        ),
        _ratio_gate(
            name="ANSWER_SPAN_EXACT_AT_LEAST_95_PERCENT",
            numerator=sum(
                bool(
                    _require_mapping(
                        row.get("pointer_metrics"),
                        field="blind sample pointer_metrics",
                    ).get("span_exact")
                )
                for row in answer_rows
            ),
            denominator=len(answer_rows),
            minimum=policy["answer_span_exact_minimum"],
        ),
        _ratio_gate(
            name="REFUSE_F1_AT_LEAST_95_PERCENT",
            numerator=refuse_f1_numerator,
            denominator=refuse_f1_denominator,
            minimum=policy["refuse_f1_minimum"],
        ),
    ]


def _build_release_qualification(
    *,
    rows: Sequence[Mapping[str, Any]],
    preflight: Mapping[str, Any],
    claim: Mapping[str, Any],
    summary: Mapping[str, Any],
    output: Path,
    run_receipt_sha256: str,
    sample_sha256: str,
    summary_sha256: str,
) -> dict[str, Any]:
    """Recompute the pre-authorized blind gates without changing thresholds."""

    execution = _require_mapping(
        preflight.get("execution"),
        field="preflight.execution",
    )
    if execution.get("backend") != "hf_model":
        raise BlindProtocolV6Error(
            "fixture blind runs can never produce release qualification"
        )
    policy = _require_mapping(
        preflight["authorization"].get("release_qualification_policy"),
        field="authorization.release_qualification_policy",
    )
    if dict(policy) != RELEASE_QUALIFICATION_POLICY:
        raise BlindProtocolV6Error("release qualification policy differs from authorization")
    if (
        len(rows) != EXPECTED_BLIND_EXAMPLES
        or summary.get("status") != "BLIND_EVALUATION_COMPLETE_FINAL_REPORT_ONLY"
    ):
        raise BlindProtocolV6Error("release qualification requires the complete blind run")

    gates = _release_gate_results(
        rows=rows,
        backend_mode=str(preflight["execution"]["backend"]),
        calibration_model_bound=(
            preflight["authorization"]["gates"]["calibration"]["model_bound"]
            is True
        ),
        policy=policy,
    )
    qualified = all(bool(gate["passed"]) for gate in gates)
    status = RELEASE_QUALIFICATION_PASS_STATUS if qualified else RELEASE_QUALIFICATION_HOLD_STATUS
    authorization = preflight["authorization"]
    selection_gate = _require_mapping(
        authorization["gates"].get("selection_freeze"),
        field="authorization.gates.selection_freeze",
    )
    calibration_gate = _require_mapping(
        authorization["gates"].get("calibration"),
        field="authorization.gates.calibration",
    )
    ablation_gate = _require_mapping(
        authorization["gates"].get("ablation"),
        field="authorization.gates.ablation",
    )
    body = {
        "schema": RELEASE_QUALIFICATION_SCHEMA,
        "version": RELEASE_QUALIFICATION_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "qualified": qualified,
        "authorization": {
            "path": str(preflight["authorization_path"]),
            "sha256": preflight["authorization_sha256"],
            "authorization_id": authorization["authorization_id"],
            "policy_sha256": sha256_bytes(canonical_json(dict(policy)).encode("utf-8")),
        },
        "blind_run_receipt": {
            "path": str(output / "run_receipt.v6.json"),
            "sha256": run_receipt_sha256,
            "schema": RUN_RECEIPT_SCHEMA,
            "status": "BLIND_EVALUATION_COMPLETE_FINAL_REPORT_ONLY",
        },
        "consumption_claim": {
            "path": str(claim["path"]),
            "sha256": claim["sha256"],
            "nonce_sha256": claim["receipt"]["nonce_sha256"],
            "failure_is_non_reusable": True,
        },
        "upstream": {
            "selection_freeze_sha256": selection_gate["sha256"],
            "calibration_receipt_sha256": calibration_gate["sha256"],
            "ablation_receipt_sha256": ablation_gate["sha256"],
            "dataset_manifest_sha256": preflight["dataset"]["manifest_sha256"],
            "blind_sha256": preflight["dataset"]["blind_sha256"],
            "base_model_tree_sha256": preflight["base"]["tree_sha256"],
            "checkpoint_tree_sha256": preflight["checkpoint"]["tree_sha256"],
            "adapter_tree_sha256": preflight["adapter"]["tree_sha256"],
        },
        "artifacts": {
            "sample_results.v6.jsonl": {
                "path": str(output / "sample_results.v6.jsonl"),
                "records": EXPECTED_BLIND_EXAMPLES,
                "sha256": sample_sha256,
            },
            "summary.v6.json": {
                "path": str(output / "summary.v6.json"),
                "sha256": summary_sha256,
            },
        },
        "thresholds": json.loads(canonical_json(dict(policy))),
        "gate_results": gates,
        "release_authorization": {
            "gguf_release_authorized": qualified,
            "activation_authorized": False,
            "deployment_authorized": False,
            "production_integration_authorized": False,
        },
        "blind_use_policy": {
            "model_selection_performed": False,
            "checkpoint_ranking_performed": False,
            "threshold_tuning_performed": False,
            "calibration_performed": False,
            "retry_allowed": False,
        },
        "claim_boundary": (
            "GGUF_ARTIFACT_BUILD_ONLY_WHEN_QUALIFIED; NOT_MODEL_SELECTION_"
            "ACTIVATION_DEPLOYMENT_PRODUCTION_X5_OR_BPU_AUTHORIZATION"
        ),
    }
    return {
        **body,
        "canonical_digest_sha256": sha256_bytes(canonical_json(body).encode("utf-8")),
    }


def _load_blind_sample_evidence(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[bytes, list[dict[str, Any]]]:
    resolved = _stable_regular_file(path, label="blind per-sample evidence")
    before = resolved.stat()
    if before.st_size <= 0 or before.st_size > MAX_BLIND_BYTES:
        raise BlindProtocolV6Error("blind per-sample evidence bytes are invalid")
    payload = resolved.read_bytes()
    after = resolved.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(payload) != before.st_size
    ):
        raise BlindProtocolV6Error(
            "blind per-sample evidence changed while it was read"
        )
    if sha256_bytes(payload) != _require_sha256(
        expected_sha256,
        field="blind per-sample evidence SHA-256",
    ):
        raise BlindProtocolV6Error("blind per-sample evidence SHA-256 mismatch")
    rows: list[dict[str, Any]] = []
    example_ids: set[str] = set()
    for line_number, raw_line in enumerate(payload.splitlines(), 1):
        if not raw_line.strip():
            raise BlindProtocolV6Error(
                f"blind per-sample evidence contains blank line {line_number}"
            )
        try:
            value = json.loads(
                raw_line.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_pairs,
            )
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise BlindProtocolV6Error(
                f"blind per-sample evidence line {line_number} is invalid"
            ) from exc
        row = dict(
            _require_mapping(
                value,
                field=f"blind per-sample evidence line {line_number}",
            )
        )
        example_id = row.get("example_id")
        if (
            row.get("schema") != SAMPLE_SCHEMA
            or row.get("split") != "blind_test"
            or not isinstance(example_id, str)
            or not example_id
            or example_id in example_ids
        ):
            raise BlindProtocolV6Error(
                "blind per-sample evidence membership or schema is invalid"
            )
        data_flow = _require_mapping(
            row.get("data_flow"),
            field=f"blind sample {example_id} data_flow",
        )
        blind_use = _require_mapping(
            row.get("blind_use_policy"),
            field=f"blind sample {example_id} use policy",
        )
        if (
            data_flow.get("blind_data_accessed") is not True
            or blind_use.get("post_generation_scoring_only") is not True
            or blind_use.get("selection_or_threshold_feedback") is not False
        ):
            raise BlindProtocolV6Error(
                "blind per-sample evidence violates the one-shot scoring boundary"
            )
        example_ids.add(example_id)
        rows.append(row)
    if len(rows) != EXPECTED_BLIND_EXAMPLES:
        raise BlindProtocolV6Error(
            "blind per-sample evidence must contain exactly 150 rows"
        )
    return payload, rows


def verify_release_qualification_v6(
    *,
    blind_receipt_path: Path,
    blind_receipt_sha256: str,
    qualification_receipt_path: Path,
    qualification_receipt_sha256: str,
) -> dict[str, Any]:
    """Independently recompute one HF blind release qualification."""

    blind_path, blind_payload, blind = _load_json(
        blind_receipt_path,
        label="blind run receipt",
        expected_sha256=blind_receipt_sha256,
    )
    qualification_path, qualification_payload, qualification = _load_json(
        qualification_receipt_path,
        label="blind release qualification",
        expected_sha256=qualification_receipt_sha256,
    )
    if (
        blind_path.name != "run_receipt.v6.json"
        or qualification_path.name != "release_qualification.v6.json"
        or qualification_path.parent != blind_path.parent
    ):
        raise BlindProtocolV6Error(
            "blind run and qualification must be canonical sibling artifacts"
        )
    if (
        blind.get("schema") != RUN_RECEIPT_SCHEMA
        or blind.get("status")
        != "BLIND_EVALUATION_COMPLETE_FINAL_REPORT_ONLY"
        or blind.get("examples") != EXPECTED_BLIND_EXAMPLES
    ):
        raise BlindProtocolV6Error("blind run receipt is not a complete v6 run")
    backend = _require_mapping(blind.get("backend"), field="blind run backend")
    if backend.get("mode") != "hf_model":
        raise BlindProtocolV6Error(
            "fixture blind runs can never produce release qualification"
        )
    if (
        qualification.get("schema") != RELEASE_QUALIFICATION_SCHEMA
        or qualification.get("version") != RELEASE_QUALIFICATION_VERSION
    ):
        raise BlindProtocolV6Error("blind release qualification schema is invalid")
    canonical_digest = _require_sha256(
        qualification.get("canonical_digest_sha256"),
        field="blind qualification canonical digest",
    )
    qualification_body = dict(qualification)
    del qualification_body["canonical_digest_sha256"]
    if (
        sha256_bytes(canonical_json(qualification_body).encode("utf-8"))
        != canonical_digest
    ):
        raise BlindProtocolV6Error(
            "blind release qualification canonical digest mismatch"
        )

    authorization_record = _require_mapping(
        qualification.get("authorization"),
        field="blind qualification authorization",
    )
    authorization_path, authorization_payload, authorization = _load_json(
        Path(str(authorization_record.get("path"))),
        label="blind authorization",
        expected_sha256=str(authorization_record.get("sha256")),
    )
    if (
        authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("status") != AUTHORIZATION_STATUS
        or authorization.get("authorization_id")
        != authorization_record.get("authorization_id")
        or authorization.get("release_qualification_policy")
        != RELEASE_QUALIFICATION_POLICY
    ):
        raise BlindProtocolV6Error(
            "blind qualification does not bind a valid pre-run authorization"
        )
    if authorization_record.get("policy_sha256") != sha256_bytes(
        canonical_json(RELEASE_QUALIFICATION_POLICY).encode("utf-8")
    ):
        raise BlindProtocolV6Error(
            "blind qualification policy was not frozen at authorization"
        )
    blind_authorization = _require_mapping(
        blind.get("authorization"),
        field="blind run authorization",
    )
    if (
        blind_authorization.get("sha256") != sha256_bytes(authorization_payload)
        or blind_authorization.get("authorization_id")
        != authorization.get("authorization_id")
        or Path(str(blind_authorization.get("path"))).resolve(strict=True)
        != authorization_path
    ):
        raise BlindProtocolV6Error(
            "blind run and qualification authorization bindings differ"
        )

    claim_record = _require_mapping(
        qualification.get("consumption_claim"),
        field="blind qualification consumption claim",
    )
    claim_path, claim_payload, claim = _load_json(
        Path(str(claim_record.get("path"))),
        label="blind one-shot claim",
        expected_sha256=str(claim_record.get("sha256")),
    )
    blind_claim = _require_mapping(
        blind.get("consumption_claim"),
        field="blind run consumption claim",
    )
    if (
        claim.get("schema") != CLAIM_SCHEMA
        or claim.get("status") != "CONSUMED_PENDING_NON_REUSABLE"
        or claim.get("authorization_id") != authorization.get("authorization_id")
        or claim.get("blind_sha256")
        != _require_mapping(blind.get("dataset"), field="blind run dataset").get(
            "blind_sha256"
        )
        or claim.get("nonce_sha256") != claim_record.get("nonce_sha256")
        or claim_record.get("failure_is_non_reusable") is not True
        or blind_claim.get("sha256") != sha256_bytes(claim_payload)
        or Path(str(blind_claim.get("path"))).resolve(strict=True) != claim_path
    ):
        raise BlindProtocolV6Error(
            "blind release qualification one-shot claim binding is invalid"
        )

    run_record = _require_mapping(
        qualification.get("blind_run_receipt"),
        field="blind qualification run receipt",
    )
    if (
        run_record.get("sha256") != sha256_bytes(blind_payload)
        or run_record.get("schema") != RUN_RECEIPT_SCHEMA
        or run_record.get("status")
        != "BLIND_EVALUATION_COMPLETE_FINAL_REPORT_ONLY"
        or Path(str(run_record.get("path"))).resolve(strict=True) != blind_path
    ):
        raise BlindProtocolV6Error(
            "blind qualification does not bind the real run receipt"
        )

    artifacts = _require_mapping(
        qualification.get("artifacts"),
        field="blind qualification artifacts",
    )
    run_artifacts = _require_mapping(
        blind.get("artifacts"),
        field="blind run artifacts",
    )
    sample_record = _require_mapping(
        artifacts.get("sample_results.v6.jsonl"),
        field="blind qualification samples",
    )
    summary_record = _require_mapping(
        artifacts.get("summary.v6.json"),
        field="blind qualification summary",
    )
    sample_path = Path(str(sample_record.get("path"))).resolve(strict=True)
    summary_path = Path(str(summary_record.get("path"))).resolve(strict=True)
    if (
        sample_path != blind_path.parent / "sample_results.v6.jsonl"
        or summary_path != blind_path.parent / "summary.v6.json"
        or sample_record.get("sha256")
        != _require_mapping(
            run_artifacts.get("sample_results.v6.jsonl"),
            field="blind run sample artifact",
        ).get("sha256")
        or summary_record.get("sha256")
        != _require_mapping(
            run_artifacts.get("summary.v6.json"),
            field="blind run summary artifact",
        ).get("sha256")
    ):
        raise BlindProtocolV6Error(
            "blind qualification artifact paths or hashes differ from the run"
        )
    sample_payload, rows = _load_blind_sample_evidence(
        sample_path,
        expected_sha256=str(sample_record.get("sha256")),
    )
    _, summary_payload, summary = _load_json(
        summary_path,
        label="blind summary",
        expected_sha256=str(summary_record.get("sha256")),
    )
    selection = _require_mapping(
        summary.get("selection"),
        field="blind summary selection",
    )
    if (
        summary.get("schema") != SUMMARY_SCHEMA
        or summary.get("status")
        != "BLIND_EVALUATION_COMPLETE_FINAL_REPORT_ONLY"
        or selection.get("complete_split") is not True
        or selection.get("rows_evaluated") != EXPECTED_BLIND_EXAMPLES
    ):
        raise BlindProtocolV6Error("blind summary is not a complete final report")

    policy = _require_mapping(
        qualification.get("thresholds"),
        field="blind qualification thresholds",
    )
    if dict(policy) != RELEASE_QUALIFICATION_POLICY:
        raise BlindProtocolV6Error(
            "blind qualification thresholds differ from authorization"
        )
    gates = _release_gate_results(
        rows=rows,
        backend_mode="hf_model",
        calibration_model_bound=(
            _require_mapping(
                _require_mapping(
                    authorization.get("gates"),
                    field="blind authorization gates",
                ).get("calibration"),
                field="blind authorization calibration gate",
            ).get("model_bound")
            is True
        ),
        policy=policy,
    )
    if qualification.get("gate_results") != gates:
        raise BlindProtocolV6Error(
            "blind qualification gates differ from per-sample recomputation"
        )
    qualified = all(gate["passed"] is True for gate in gates)
    expected_status = (
        RELEASE_QUALIFICATION_PASS_STATUS
        if qualified
        else RELEASE_QUALIFICATION_HOLD_STATUS
    )
    release = _require_mapping(
        qualification.get("release_authorization"),
        field="blind qualification release authorization",
    )
    if (
        qualification.get("qualified") is not qualified
        or qualification.get("status") != expected_status
        or release.get("gguf_release_authorized") is not qualified
        or release.get("activation_authorized") is not False
        or release.get("deployment_authorized") is not False
        or release.get("production_integration_authorized") is not False
    ):
        raise BlindProtocolV6Error(
            "blind qualification status or authorization differs from recomputation"
        )
    if not qualified:
        raise BlindProtocolV6Error(
            "blind run completed but did not qualify GGUF release"
        )
    implementation = _require_mapping(
        blind.get("implementation"),
        field="blind run implementation",
    )
    runner = _require_mapping(
        implementation.get("runner"),
        field="blind run runner",
    )
    if _source_bindings(Path(str(runner.get("path")))) != dict(implementation):
        raise BlindProtocolV6Error(
            "blind implementation changed after the qualified run"
        )
    return {
        "status": "PASS_BLIND_RELEASE_QUALIFICATION_INDEPENDENTLY_RECOMPUTED",
        "blind_receipt_sha256": sha256_bytes(blind_payload),
        "qualification_receipt_sha256": sha256_bytes(qualification_payload),
        "authorization_sha256": sha256_bytes(authorization_payload),
        "claim_sha256": sha256_bytes(claim_payload),
        "sample_results_sha256": sha256_bytes(sample_payload),
        "summary_sha256": sha256_bytes(summary_payload),
        "samples_recomputed": len(rows),
        "gates_recomputed": len(gates),
        "qualified": True,
        "fixture_accepted": False,
        "blind_dataset_reopened": False,
    }


def _write_terminal(
    *,
    claim: Mapping[str, Any],
    preflight: Mapping[str, Any],
    status: str,
    artifacts: Mapping[str, Any] | None,
    error: BaseException | None,
) -> dict[str, Any]:
    if status not in {"COMPLETED", "FAILED_NON_REUSABLE"}:
        raise BlindProtocolV6Error("terminal status is invalid")
    error_record = None
    if error is not None:
        message = str(error)
        error_record = {
            "type": type(error).__name__,
            "message": message[:MAX_ERROR_CHARS],
            "traceback": "".join(
                traceback.format_exception(
                    type(error),
                    error,
                    error.__traceback__,
                )
            )[: MAX_ERROR_CHARS * 4],
        }
    terminal = {
        "schema": TERMINAL_SCHEMA,
        "status": status,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "authorization_id": preflight["authorization"]["authorization_id"],
        "authorization_sha256": preflight["authorization_sha256"],
        "claim_path": str(claim["path"]),
        "claim_sha256": claim["sha256"],
        "nonce_sha256": claim["receipt"]["nonce_sha256"],
        "run_id": preflight["execution"]["run_id"],
        "output_basename": preflight["execution"]["output_basename"],
        "blind_sha256": preflight["dataset"]["blind_sha256"],
        "artifacts": None if artifacts is None else dict(artifacts),
        "error": error_record,
        "failure_is_non_reusable": True,
        "overwrite_allowed": False,
    }
    terminal_path = Path(claim["terminal_path"])
    try:
        _exclusive_create(terminal_path, _json_bytes(terminal))
    except FileExistsError as exc:
        raise BlindProtocolV6Error("blind terminal receipt already exists and cannot be overwritten") from exc
    return {
        "path": str(terminal_path),
        "sha256": sha256_file(terminal_path),
        "status": status,
    }


def consume_blind_evaluation(
    *,
    workspace_root: Path,
    dataset_dir: Path,
    authorization_path: Path,
    authorization_sha256: str,
    base_model_dir: Path,
    adapter_dir: Path,
    runner_path: Path,
    output_dir: Path,
    backend_mode: str,
    fixture_path: Path | None,
    device: str | None,
    registry_root: Path,
) -> dict[str, Any]:
    """Claim, read, evaluate, and publish exactly one 150-row blind run."""

    preflight = _verify_preclaim_binding(
        authorization_path=authorization_path,
        expected_authorization_sha256=authorization_sha256,
        workspace_root=workspace_root,
        dataset_dir=dataset_dir,
        base_model_dir=base_model_dir,
        adapter_dir=adapter_dir,
        runner_path=runner_path,
        backend_mode=backend_mode,
        fixture_path=fixture_path,
        device=device,
        output_dir=output_dir,
        registry_root=registry_root,
    )
    claim = _claim_consumption(preflight)
    published_artifacts: dict[str, Any] | None = None
    try:
        # This is the first and only function allowed to parse blind JSONL.
        selection = _parse_blind_rows_after_claim(
            dataset=preflight["dataset"],
            claim=claim,
        )
        requests = pointer_eval._generation_requests(selection.rows)
        if backend_mode == "fixture":
            fixture_binding = _require_mapping(
                preflight["execution"].get("fixture"),
                field="execution.fixture",
            )
            assert fixture_path is not None
            generations, backend = _load_fixture_after_claim(
                fixture_path=fixture_path,
                expected_example_ids=[request.example_id for request in requests],
                expected_sha256=str(fixture_binding.get("sha256")),
            )
        else:
            assert device is not None
            try:
                generations, backend = pointer_eval.generate_hf_model(
                    requests,
                    base_model_dir=base_model_dir,
                    adapter_dir=adapter_dir,
                    device=device,
                    seed=FIXED_SEED,
                )
            except pointer_eval.PointerHFEvalV6Error as exc:
                raise BlindProtocolV6Error("blind HF generation failed") from exc
            model = _require_mapping(
                backend.get("model"),
                field="blind backend model",
            )
            base = _require_mapping(
                model.get("base"),
                field="blind backend base",
            )
            adapter = _require_mapping(
                model.get("adapter"),
                field="blind backend adapter",
            )
            base_after = _tree_inventory(
                base_model_dir,
                label="post-generation base model",
            )
            checkpoint_after = _tree_inventory(
                adapter_dir,
                label="post-generation selected checkpoint",
            )
            adapter_after = _adapter_inventory(
                adapter_dir,
                label="post-generation selected adapter",
            )
            if (
                Path(str(base.get("path"))).resolve(strict=True)
                != Path(preflight["base"]["path"])
                or Path(str(adapter.get("path"))).resolve(strict=True)
                != Path(preflight["checkpoint"]["path"])
                or base_after != preflight["base"]
                or checkpoint_after != preflight["checkpoint"]
                or adapter_after != preflight["adapter"]
            ):
                raise BlindProtocolV6Error("post-generation model inventory differs from authorization")
        expected_ids = {row.example_id for row in selection.rows}
        if set(generations) != expected_ids:
            raise BlindProtocolV6Error("generation membership differs from blind rows")
        sample_bindings = {
            "dataset_manifest_sha256": preflight["dataset"]["manifest_sha256"],
            "blind_sha256": preflight["dataset"]["blind_sha256"],
            "base_model_tree_sha256": preflight["base"]["tree_sha256"],
            "checkpoint_tree_sha256": preflight["checkpoint"]["tree_sha256"],
            "adapter_tree_sha256": preflight["adapter"]["tree_sha256"],
            "authorization_sha256": preflight["authorization_sha256"],
            "claim_sha256": claim["sha256"],
            "protocol_source_sha256": preflight["code"]["protocol"]["sha256"],
            "evaluator_source_sha256": preflight["code"]["evaluator"]["sha256"],
            "compiler_source_sha256": preflight["code"]["compiler"]["sha256"],
            "runner_source_sha256": preflight["code"]["runner"]["sha256"],
        }
        rows: list[dict[str, Any]] = []
        for row in selection.rows:
            try:
                scored = pointer_eval._score_row(
                    row=row,
                    generation=generations[row.example_id],
                    bindings=sample_bindings,
                    backend_mode=backend_mode,
                )
            except pointer_eval.PointerHFEvalV6Error as exc:
                raise BlindProtocolV6Error(f"blind scoring failed for {row.example_id}") from exc
            scored["schema"] = SAMPLE_SCHEMA
            scored["data_flow"]["blind_data_accessed"] = True
            scored["blind_use_policy"] = {
                "post_generation_scoring_only": True,
                "selection_or_threshold_feedback": False,
            }
            rows.append(scored)
        code_after = _source_bindings(runner_path)
        if code_after != preflight["code"]:
            raise BlindProtocolV6Error("protocol, evaluator, compiler, or runner changed during run")
        summary = _blind_summary(
            rows=rows,
            selection=selection,
            backend=backend,
            preflight=preflight,
            claim=claim,
        )

        output = preflight["output"]
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
        if staging.exists():
            raise BlindProtocolV6Error("blind staging path already exists")
        staging.mkdir()
        try:
            sample_path = staging / "sample_results.v6.jsonl"
            summary_path = staging / "summary.v6.json"
            receipt_path = staging / "run_receipt.v6.json"
            sample_payload = _jsonl_bytes(rows)
            summary_payload = _json_bytes(summary)
            sample_path.write_bytes(sample_payload)
            summary_path.write_bytes(summary_payload)
            receipt = {
                "schema": RUN_RECEIPT_SCHEMA,
                "status": summary["status"],
                "created_at_utc": datetime.now(UTC).isoformat(),
                "run_id": preflight["execution"]["run_id"],
                "output_basename": output.name,
                "examples": EXPECTED_BLIND_EXAMPLES,
                "authorization": {
                    "path": str(preflight["authorization_path"]),
                    "sha256": preflight["authorization_sha256"],
                    "authorization_id": preflight["authorization"]["authorization_id"],
                    "registry": preflight["registry"],
                },
                "consumption_claim": {
                    "path": str(claim["path"]),
                    "sha256": claim["sha256"],
                    "nonce_sha256": claim["receipt"]["nonce_sha256"],
                },
                "dataset": {
                    "manifest_sha256": preflight["dataset"]["manifest_sha256"],
                    "blind_sha256": preflight["dataset"]["blind_sha256"],
                    "blind_bytes": preflight["dataset"]["blind_bytes"],
                    "rows_read_once": EXPECTED_BLIND_EXAMPLES,
                },
                "model": {
                    "base_model_tree_sha256": preflight["base"]["tree_sha256"],
                    "checkpoint_tree_sha256": preflight["checkpoint"][
                        "tree_sha256"
                    ],
                    "adapter_tree_sha256": preflight["adapter"]["tree_sha256"],
                },
                "gates": preflight["authorization"]["gates"],
                "implementation": preflight["code"],
                "backend": backend,
                "blind_use_policy": summary["blind_use_policy"],
                "artifacts": {
                    "sample_results.v6.jsonl": {
                        "bytes": len(sample_payload),
                        "records": EXPECTED_BLIND_EXAMPLES,
                        "sha256": sha256_bytes(sample_payload),
                    },
                    "summary.v6.json": {
                        "bytes": len(summary_payload),
                        "sha256": sha256_bytes(summary_payload),
                    },
                },
                "claim_boundary": summary["claim_boundary"],
            }
            receipt_payload = _json_bytes(receipt)
            receipt_path.write_bytes(receipt_payload)
            published_artifacts = {
                "sample_results.v6.jsonl": {
                    **receipt["artifacts"]["sample_results.v6.jsonl"],
                    "path": str(output / "sample_results.v6.jsonl"),
                },
                "summary.v6.json": {
                    **receipt["artifacts"]["summary.v6.json"],
                    "path": str(output / "summary.v6.json"),
                },
                "run_receipt.v6.json": {
                    "path": str(output / "run_receipt.v6.json"),
                    "bytes": receipt_path.stat().st_size,
                    "sha256": sha256_file(receipt_path),
                },
            }
            if backend_mode == "hf_model":
                qualification_path = staging / "release_qualification.v6.json"
                qualification = _build_release_qualification(
                    rows=rows,
                    preflight=preflight,
                    claim=claim,
                    summary=summary,
                    output=output,
                    run_receipt_sha256=sha256_bytes(receipt_payload),
                    sample_sha256=sha256_bytes(sample_payload),
                    summary_sha256=sha256_bytes(summary_payload),
                )
                qualification_path.write_bytes(_json_bytes(qualification))
                published_artifacts["release_qualification.v6.json"] = {
                    "path": str(output / "release_qualification.v6.json"),
                    "bytes": qualification_path.stat().st_size,
                    "sha256": sha256_file(qualification_path),
                    "status": qualification["status"],
                    "qualified": qualification["qualified"],
                }
            staging.replace(output)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        terminal = _write_terminal(
            claim=claim,
            preflight=preflight,
            status="COMPLETED",
            artifacts=published_artifacts,
            error=None,
        )
    except BaseException as exc:
        try:
            _write_terminal(
                claim=claim,
                preflight=preflight,
                status="FAILED_NON_REUSABLE",
                artifacts=published_artifacts,
                error=exc,
            )
        except BaseException as terminal_exc:
            raise BlindProtocolV6Error(
                f"blind run failed after immutable claim; terminal evidence also failed: {terminal_exc}"
            ) from exc
        if isinstance(exc, BlindProtocolV6Error):
            raise
        raise BlindProtocolV6Error("blind run failed after immutable claim") from exc
    result = {
        "status": "BLIND_EVALUATION_COMPLETE_FINAL_REPORT_ONLY",
        "output_dir": str(preflight["output"]),
        "examples": EXPECTED_BLIND_EXAMPLES,
        "blind_rows_read_once": EXPECTED_BLIND_EXAMPLES,
        "authorization_sha256": preflight["authorization_sha256"],
        "claim": {
            "path": str(claim["path"]),
            "sha256": claim["sha256"],
        },
        "terminal": terminal,
        "hashes": {name: item["sha256"] for name, item in published_artifacts.items()},
        "release_qualification_produced": backend_mode == "hf_model",
        "model_selection_performed": False,
        "threshold_tuning_performed": False,
        "replay_allowed": False,
    }
    if backend_mode == "hf_model":
        qualification_artifact = published_artifacts[
            "release_qualification.v6.json"
        ]
        result["release_qualification"] = {
            "path": qualification_artifact["path"],
            "sha256": qualification_artifact["sha256"],
            "status": qualification_artifact["status"],
            "qualified": qualification_artifact["qualified"],
        }
    return result


__all__ = [
    "ABLATION_SCHEMA",
    "ABLATION_STATUS",
    "AUTHORIZATION_SCHEMA",
    "AUTHORIZATION_STATUS",
    "BlindProtocolV6Error",
    "CALIBRATION_SCHEMA",
    "CALIBRATION_STATUS",
    "EXPECTED_BLIND_EXAMPLES",
    "FIXED_MAX_INPUT_TOKENS",
    "FIXED_MAX_NEW_TOKENS",
    "FIXED_SEED",
    "RELEASE_QUALIFICATION_HOLD_STATUS",
    "RELEASE_QUALIFICATION_PASS_STATUS",
    "RELEASE_QUALIFICATION_POLICY",
    "RELEASE_QUALIFICATION_SCHEMA",
    "RUN_RECEIPT_SCHEMA",
    "SELECTION_FREEZE_SCHEMA",
    "SELECTION_FREEZE_STATUS",
    "authorize_blind_evaluation",
    "canonical_json",
    "consume_blind_evaluation",
    "sha256_file",
    "verify_release_qualification_v6",
]
