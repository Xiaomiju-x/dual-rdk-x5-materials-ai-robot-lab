"""GGUF preflight and one-shot export authorization for strict ICMat v8.

Preflight remains read-only.  A separate explicit action irreversibly claims a
digest-keyed registry entry and terminalizes that authorization without running
model merge, GGUF conversion, parity, or board replay.  Legacy v5/v6/v7 receipts
are classified explicitly but can never authorize this path.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from icmat_foundry.llm import (
    ablation_eval_v7,
    ablation_eval_v8,
    calibration_eval_v7,
    calibration_eval_v8,
    gguf_export_v5,
    gguf_release_v6,
    postfreeze_blind_v7,
    selection_freeze_v7,
    selection_freeze_v8,
)

VERSION = "icmat-gguf-release-v8-preflight.0.0"
PREFLIGHT_SCHEMA = "icmat_llm_gguf_release_preflight.v8"
PREFLIGHT_PASS_STATUS = "PASS_V8_GGUF_RELEASE_AUTHORIZATION_READY_NOT_EXPORTED"
PREFLIGHT_FAIL_STATUS = "HOLD_V8_GGUF_RELEASE_NOT_AUTHORIZED_NOT_EXPORTED"
CLAIM_SCHEMA = "icmat_llm_gguf_export_authorization_claim.v8"
CLAIM_STATUS = "CLAIMED_NON_REUSABLE_V8_GGUF_EXPORT_AUTHORIZATION"
TERMINAL_SCHEMA = "icmat_llm_gguf_export_authorization_terminal.v8"
TERMINAL_STATUS = "AUTHORIZED_ONCE_V8_GGUF_EXPORT_NOT_EXPORTED"

CALIBRATION_SCHEMA = calibration_eval_v8.RECEIPT_SCHEMA
CALIBRATION_VERSION = calibration_eval_v8.VERSION
CALIBRATION_STATUS = "PASS_STRICT_NONBLIND_V8_CALIBRATION_MODEL_BOUND"
ABLATION_SCHEMA = ablation_eval_v8.RECEIPT_SCHEMA
ABLATION_VERSION = ablation_eval_v8.VERSION
ABLATION_STATUS = "PASS_STRICT_NONBLIND_V8_POST_SELECTION_ABLATIONS"
POSTFREEZE_SCHEMA = "icmat_llm_postfreeze_run_receipt.v8"
POSTFREEZE_VERSION = "icmat-postfreeze-blind-v8.0.0"
POSTFREEZE_STATUS = "POSTFREEZE_V8_ONE_SHOT_COMPLETE"
QUALIFICATION_SCHEMA = "icmat_llm_postfreeze_gguf_qualification.v8"
QUALIFICATION_VERSION = POSTFREEZE_VERSION
QUALIFICATION_STATUS = "PASS_GGUF_OFFLINE_CANDIDATE_ONLY"
VERIFICATION_SCHEMA = "icmat_llm_postfreeze_verification.v8"
VERIFICATION_VERSION = "icmat-postfreeze-verifier-v8.0.0"
VERIFICATION_STATUS = "PASS_POSTFREEZE_V8_INDEPENDENTLY_RECOMPUTED"

PARITY_PROTOCOL = "NATIVE_POINTER_V8_REQUIRED"
REPLAY_PROTOCOL = "NATIVE_POINTER_V8_REQUIRED"

MAX_JSON_BYTES = 64 * 1024 * 1024
EXPECTED_ROWS = 150
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_REGISTRY_ROOT = (
    WORKSPACE_ROOT
    / "evaluation"
    / "icmat_foundry"
    / "llm"
    / "gguf_release_v8_registry"
)

_PREFLIGHT_FIELDS = {
    "schema",
    "version",
    "status",
    "read_only",
    "export_performed",
    "reserved_blind_dataset_read_by_this_preflight",
    "postfreeze_evidence_artifacts_hashed",
    "network_used",
    "x5_contacted",
    "chain_binding",
    "authority_digest_sha256",
    "authority_receipts",
    "postfreeze_artifacts",
    "tool_binding",
    "llama_server",
    "low_level_export_preflight",
    "required_followup_protocols",
    "authorization",
    "authorization_digest_sha256",
    "claim_boundary",
    "canonical_digest_sha256",
}
_CLAIM_FIELDS = {
    "schema",
    "version",
    "status",
    "registry_key",
    "authorization_digest_sha256",
    "preflight_canonical_digest_sha256",
    "authority_digest_sha256",
    "tool_binding_digest_sha256",
    "failure_is_non_reusable",
    "retry_allowed",
    "export_performed",
    "authorization",
    "canonical_digest_sha256",
}
_TERMINAL_FIELDS = {
    "schema",
    "version",
    "status",
    "registry_key",
    "authorization_digest_sha256",
    "preflight_canonical_digest_sha256",
    "claim_sha256",
    "claim_canonical_digest_sha256",
    "failure_is_non_reusable",
    "retry_allowed",
    "export_performed",
    "authorization",
    "canonical_digest_sha256",
}

LEGACY_SCHEMA_ROLES = {
    selection_freeze_v7.SCHEMA: "selection_freeze_v7",
    calibration_eval_v7.RECEIPT_SCHEMA: "calibration_v7",
    ablation_eval_v7.RECEIPT_SCHEMA: "ablation_v7",
    postfreeze_blind_v7.RUN_RECEIPT_SCHEMA: "postfreeze_blind_v7",
    postfreeze_blind_v7.QUALIFICATION_SCHEMA: "postfreeze_qualification_v7",
    gguf_export_v5.EXPORT_RECEIPT_SCHEMA: "gguf_export_v5",
    gguf_release_v6.RELEASE_RECEIPT_SCHEMA: "gguf_release_v6",
}

FALSE_AUTHORIZATION = {
    "activation_authorized": False,
    "x5_execution_authorized": False,
    "deployment_authorized": False,
    "production_integration_authorized": False,
}


class GgufReleaseV8Error(RuntimeError):
    """Raised when a v8 GGUF authorization input fails closed."""


@dataclass(frozen=True)
class FileSnapshotV8:
    path: Path
    payload: bytes
    sha256: str
    identity: tuple[int, int, int, int, int]

    @property
    def bytes(self) -> int:
        return len(self.payload)

    def descriptor(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class BinaryFileSnapshotV8:
    path: Path
    bytes: int
    sha256: str
    identity: tuple[int, int, int, int, int]

    def descriptor(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ReleaseAuthorityInputsV8:
    selection_freeze: Path
    selection_freeze_sha256: str
    evaluation_index: Path
    training_receipt: Path
    dataset_dir: Path
    base_model_dir: Path
    selected_adapter_dir: Path
    calibration_receipt: Path
    calibration_receipt_sha256: str
    ablation_receipt: Path
    ablation_receipt_sha256: str
    postfreeze_receipt: Path
    postfreeze_receipt_sha256: str
    qualification_receipt: Path
    qualification_receipt_sha256: str
    verification_receipt: Path
    verification_receipt_sha256: str


@dataclass(frozen=True)
class ToolchainInputsV8:
    converter: Path = gguf_release_v6.DEFAULT_CONVERTER
    converter_sha256: str = gguf_release_v6.DEFAULT_CONVERTER_SHA256
    quantizer: Path = gguf_release_v6.DEFAULT_QUANTIZER
    quantizer_sha256: str = gguf_release_v6.DEFAULT_QUANTIZER_SHA256
    llama_server: Path = gguf_release_v6.DEFAULT_LLAMA_SERVER
    llama_server_sha256: str = gguf_release_v6.DEFAULT_LLAMA_SERVER_SHA256


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise GgufReleaseV8Error(f"{label} must be a lowercase SHA-256")
    return value


def _require_exact_mapping(
    value: Any,
    *,
    exact: set[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GgufReleaseV8Error(f"{label} must be an object")
    if set(value) != exact:
        raise GgufReleaseV8Error(
            f"{label} keys differ: expected {sorted(exact)}, got {sorted(value)}"
        )
    return value


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _is_reparse(metadata: os.stat_result) -> bool:
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return bool(reparse and attributes & reparse)


def _real_directory_chain(
    path: Path,
    *,
    label: str,
    create: bool,
) -> tuple[Path, tuple[int, int, int, int, int]]:
    lexical = Path(os.path.abspath(os.fspath(Path(path))))
    parts = lexical.parts
    if not parts or not lexical.is_absolute():
        raise GgufReleaseV8Error(f"{label} must be an absolute directory")
    current = Path(parts[0])
    for part in parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            if not create:
                raise GgufReleaseV8Error(f"{label} is unavailable") from None
            try:
                os.mkdir(current)
            except FileExistsError:
                pass
            metadata = os.lstat(current)
        except OSError as exc:
            raise GgufReleaseV8Error(f"{label} is unavailable") from exc
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or _is_reparse(metadata)
        ):
            raise GgufReleaseV8Error(
                f"{label} contains a symlink/reparse or non-directory component"
            )
    try:
        resolved = lexical.resolve(strict=True)
        metadata = os.lstat(resolved)
    except OSError as exc:
        raise GgufReleaseV8Error(f"{label} is unavailable") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise GgufReleaseV8Error(f"{label} is not a canonical real directory")
    return resolved, _identity(metadata)


def _production_registry_root(
    *,
    create: bool,
) -> tuple[Path, tuple[int, int, int, int, int]]:
    return _real_directory_chain(
        PRODUCTION_REGISTRY_ROOT,
        label="fixed v8 GGUF authorization registry",
        create=create,
    )


def _registry_paths_v8(
    authorization_digest_sha256: str,
    *,
    create: bool,
) -> tuple[dict[str, Path], tuple[int, int, int, int, int]]:
    digest = _require_sha256(
        authorization_digest_sha256,
        label="GGUF export authorization digest",
    )
    root, identity = _production_registry_root(create=create)
    paths = {
        "root": root,
        "claim": root / f"{digest}.claim.v8.json",
        "terminal": root / f"{digest}.terminal.v8.json",
    }
    if any(path.parent != root for path in (paths["claim"], paths["terminal"])):
        raise GgufReleaseV8Error("GGUF authorization registry path escaped")
    return paths, identity


def _verify_registry_identity(
    root: Path,
    expected_identity: tuple[int, int, int, int, int],
) -> None:
    observed, identity = _real_directory_chain(
        root,
        label="fixed v8 GGUF authorization registry",
        create=False,
    )
    if observed != root or identity[:2] != expected_identity[:2]:
        raise GgufReleaseV8Error("GGUF authorization registry identity changed")


def _snapshot_file(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
    maximum_bytes: int = MAX_JSON_BYTES,
) -> FileSnapshotV8:
    raw = Path(path).expanduser().absolute()
    try:
        metadata = os.lstat(raw)
    except OSError as exc:
        raise GgufReleaseV8Error(f"{label} is unavailable: {raw}") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise GgufReleaseV8Error(f"{label} must be a real regular file")
    if metadata.st_size > maximum_bytes:
        raise GgufReleaseV8Error(f"{label} exceeds the fixed byte limit")
    with raw.open("rb") as handle:
        before = _identity(os.fstat(handle.fileno()))
        payload = handle.read()
        after = _identity(os.fstat(handle.fileno()))
    current = _identity(os.lstat(raw))
    if before != after or after != current or len(payload) != current[2]:
        raise GgufReleaseV8Error(f"{label} changed while it was read")
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None:
        expected = _require_sha256(expected_sha256, label=f"{label} expected SHA-256")
        if digest != expected:
            raise GgufReleaseV8Error(f"{label} SHA-256 mismatch")
    return FileSnapshotV8(
        path=raw.resolve(strict=True),
        payload=payload,
        sha256=digest,
        identity=current,
    )


def _snapshot_binary_file(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
    maximum_bytes: int,
) -> BinaryFileSnapshotV8:
    raw = Path(path).expanduser().absolute()
    try:
        metadata = os.lstat(raw)
    except OSError as exc:
        raise GgufReleaseV8Error(f"{label} is unavailable: {raw}") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise GgufReleaseV8Error(f"{label} must be a real regular file")
    if not 0 < metadata.st_size <= maximum_bytes:
        raise GgufReleaseV8Error(f"{label} has an invalid byte count")
    with raw.open("rb") as handle:
        before = _identity(os.fstat(handle.fileno()))
        digest = hashlib.sha256()
        total = 0
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            total += len(block)
            if total > maximum_bytes:
                raise GgufReleaseV8Error(f"{label} exceeds the fixed byte limit")
            digest.update(block)
        after = _identity(os.fstat(handle.fileno()))
    current = _identity(os.lstat(raw))
    if before != after or after != current or total != current[2]:
        raise GgufReleaseV8Error(f"{label} changed while it was hashed")
    observed = digest.hexdigest()
    if expected_sha256 is not None:
        expected = _require_sha256(expected_sha256, label=f"{label} expected SHA-256")
        if observed != expected:
            raise GgufReleaseV8Error(f"{label} SHA-256 mismatch")
    return BinaryFileSnapshotV8(
        path=raw.resolve(strict=True),
        bytes=total,
        sha256=observed,
        identity=current,
    )


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GgufReleaseV8Error(f"duplicate JSON key rejected: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise GgufReleaseV8Error(f"non-finite JSON value rejected: {value}")


def _load_json(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[FileSnapshotV8, dict[str, Any]]:
    snapshot = _snapshot_file(
        path,
        label=label,
        expected_sha256=expected_sha256,
    )
    try:
        value = json.loads(
            snapshot.payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GgufReleaseV8Error(f"{label} must be strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise GgufReleaseV8Error(f"{label} must contain a JSON object")
    return snapshot, value


def _verify_canonical_digest(value: Mapping[str, Any], *, label: str) -> None:
    digest = _require_sha256(
        value.get("canonical_digest_sha256"),
        label=f"{label}.canonical_digest_sha256",
    )
    body = dict(value)
    del body["canonical_digest_sha256"]
    if canonical_sha256(body) != digest:
        raise GgufReleaseV8Error(f"{label} canonical digest mismatch")


def classify_receipt_schema_v8(path: Path) -> dict[str, Any]:
    """Classify one receipt without treating legacy schema as compatible."""

    snapshot, value = _load_json(path, label="receipt compatibility input")
    schema = value.get("schema")
    native = {
        selection_freeze_v8.SCHEMA,
        CALIBRATION_SCHEMA,
        ABLATION_SCHEMA,
        POSTFREEZE_SCHEMA,
        QUALIFICATION_SCHEMA,
        VERIFICATION_SCHEMA,
    }
    if isinstance(schema, str) and schema in native:
        classification = "NATIVE_V8"
        compatible = True
    elif isinstance(schema, str) and schema in LEGACY_SCHEMA_ROLES:
        classification = f"LEGACY_INCOMPATIBLE:{LEGACY_SCHEMA_ROLES[str(schema)]}"
        compatible = False
    else:
        classification = "UNKNOWN_INCOMPATIBLE"
        compatible = False
    return {
        "path": str(snapshot.path),
        "sha256": snapshot.sha256,
        "schema": schema,
        "classification": classification,
        "v8_compatible": compatible,
    }


def _reject_legacy_schema(
    value: Mapping[str, Any],
    *,
    expected_schema: str,
    label: str,
) -> None:
    schema = value.get("schema")
    if schema == expected_schema:
        return
    if isinstance(schema, str) and schema in LEGACY_SCHEMA_ROLES:
        role = LEGACY_SCHEMA_ROLES[str(schema)]
        raise GgufReleaseV8Error(
            f"{label} uses {role}; legacy receipts cannot authorize a v8 selection"
        )
    raise GgufReleaseV8Error(f"{label} schema is not the native v8 contract")


def _tree_bindings(
    *,
    base_model_dir: Path,
    selected_adapter_dir: Path,
) -> dict[str, Any]:
    try:
        base = selection_freeze_v7._stable_tree_snapshot(
            Path(base_model_dir),
            label="v8 GGUF base model",
            reject_reserved_path=False,
        )
        checkpoint = selection_freeze_v7._stable_tree_snapshot(
            Path(selected_adapter_dir),
            label="v8 GGUF selected checkpoint",
            reject_reserved_path=False,
        )
        adapter = selection_freeze_v7._selected_adapter_inventory(checkpoint)
    except (
        OSError,
        RuntimeError,
        ValueError,
        selection_freeze_v7.SelectionFreezeV7Error,
    ) as exc:
        raise GgufReleaseV8Error("selected model tree verification failed") from exc
    return {
        "base_path": str(base.root),
        "base_model_tree_sha256": base.tree_sha256_casefold,
        "checkpoint_path": str(checkpoint.root),
        "checkpoint_tree_sha256": checkpoint.tree_sha256_casefold,
        "adapter_tree_sha256": adapter["tree_sha256"],
        "adapter_file_count": adapter["file_count"],
        "adapter_bytes": adapter["bytes"],
    }


def _selection_binding(
    inputs: ReleaseAuthorityInputsV8,
) -> tuple[dict[str, Any], dict[str, Any], FileSnapshotV8]:
    selection_snapshot, selection = _load_json(
        inputs.selection_freeze,
        label="selection freeze v8",
        expected_sha256=inputs.selection_freeze_sha256,
    )
    _reject_legacy_schema(
        selection,
        expected_schema=selection_freeze_v8.SCHEMA,
        label="selection freeze",
    )
    try:
        verified = selection_freeze_v8.verify_selection_freeze_v8(
            freeze_receipt_path=selection_snapshot.path,
            evaluation_index_path=inputs.evaluation_index,
            training_receipt_path=inputs.training_receipt,
            dataset_dir=inputs.dataset_dir,
            base_model_dir=inputs.base_model_dir,
        )
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        selection_freeze_v8.SelectionFreezeV8Error,
    ) as exc:
        raise GgufReleaseV8Error(
            "selection_freeze_v8 independent verification failed"
        ) from exc
    if (
        verified.get("status") != selection_freeze_v8.VERIFIED_STATUS
        or verified.get("selection_locked") is not True
        or verified.get("gguf_export_authorized") is not False
        or verified.get("x5_execution_authorized") is not False
        or verified.get("deployment_authorized") is not False
    ):
        raise GgufReleaseV8Error("selection verifier returned an unsafe boundary")

    model = _tree_bindings(
        base_model_dir=inputs.base_model_dir,
        selected_adapter_dir=inputs.selected_adapter_dir,
    )
    selected = _require_exact_mapping(
        selection.get("selection"),
        exact={
            "checkpoint_id",
            "seed",
            "epoch",
            "global_step",
            "validation_loss",
            "checkpoint_path",
            "checkpoint_tree_sha256",
            "checkpoint_file_count",
            "checkpoint_bytes",
            "adapter_tree_sha256",
            "stable_tree_digest_sha256",
            "ranking_metrics",
            "qualified_seeds",
            "selection_locked",
        },
        label="selection checkpoint",
    )
    base = _require_exact_mapping(
        selection.get("base_model"),
        exact={
            "path",
            "tree_sha256",
            "evaluator_tree_sha256",
            "file_count",
            "bytes",
            "stable_tree_digest_sha256",
        },
        label="selection base model",
    )
    if (
        Path(str(selected["checkpoint_path"])).resolve(strict=True)
        != Path(model["checkpoint_path"])
        or Path(str(base["path"])).resolve(strict=True) != Path(model["base_path"])
        or selected["checkpoint_tree_sha256"] != model["checkpoint_tree_sha256"]
        or selected["adapter_tree_sha256"] != model["adapter_tree_sha256"]
        or base["tree_sha256"] != model["base_model_tree_sha256"]
    ):
        raise GgufReleaseV8Error("runtime model trees differ from selection_freeze_v8")

    manifest_record = _require_exact_mapping(
        selection.get("manifest"),
        exact={
            "path",
            "bytes",
            "sha256",
            "stable_identity",
            "schema",
            "dataset_schema",
            "builder_version",
        },
        label="selection manifest binding",
    )
    manifest_path = Path(inputs.dataset_dir) / selection_freeze_v8.MANIFEST_NAME
    manifest_snapshot, manifest = _load_json(
        manifest_path,
        label="strict nonblind v8 manifest",
        expected_sha256=str(manifest_record["sha256"]),
    )
    splits = _require_exact_mapping(
        manifest.get("splits"),
        exact={"train", "validation", "calibration"},
        label="strict nonblind v8 split table",
    )
    split_bindings: dict[str, Any] = {}
    for split, rows in (("train", 250), ("validation", 150), ("calibration", 150)):
        record = _require_exact_mapping(
            splits.get(split),
            exact={"path", "count", "bytes", "sha256"},
            label=f"strict nonblind v8 split {split}",
        )
        split_path = Path(inputs.dataset_dir) / str(record["path"])
        split_snapshot = _snapshot_file(
            split_path,
            label=f"strict nonblind v8 split {split}",
            expected_sha256=str(record["sha256"]),
            maximum_bytes=32 * 1024 * 1024,
        )
        if record["count"] != rows or record["bytes"] != split_snapshot.bytes:
            raise GgufReleaseV8Error(f"strict nonblind v8 split {split} is incomplete")
        split_bindings[split] = {
            "rows": rows,
            "bytes": split_snapshot.bytes,
            "sha256": split_snapshot.sha256,
        }

    preblind = _require_exact_mapping(
        selection.get("preblind_commitment"),
        exact={
            "path",
            "bytes",
            "sha256",
            "stable_identity",
            "schema",
            "commitment_sha256",
        },
        label="selection preblind commitment",
    )
    binding = {
        "selection_freeze_sha256": selection_snapshot.sha256,
        "selection_binding_digest_sha256": _require_sha256(
            selection.get("selection_binding_digest_sha256"),
            label="selection binding digest",
        ),
        "manifest_sha256": manifest_snapshot.sha256,
        "preblind_commitment_file_sha256": _require_sha256(
            preblind.get("sha256"),
            label="preblind commitment file SHA-256",
        ),
        "preblind_commitment_sha256": _require_sha256(
            preblind.get("commitment_sha256"),
            label="preblind commitment digest",
        ),
        "base_model_tree_sha256": model["base_model_tree_sha256"],
        "checkpoint_id": selected["checkpoint_id"],
        "checkpoint_tree_sha256": model["checkpoint_tree_sha256"],
        "adapter_tree_sha256": model["adapter_tree_sha256"],
    }
    return (
        binding,
        {
            "selection_verification": dict(verified),
            "selection": selection_snapshot.descriptor(),
            "manifest": manifest_snapshot.descriptor(),
            "splits": split_bindings,
            "model": model,
        },
        selection_snapshot,
    )


def _false_authorization(value: Any, *, label: str) -> Mapping[str, Any]:
    authorization = _require_exact_mapping(
        value,
        exact=set(FALSE_AUTHORIZATION),
        label=label,
    )
    if dict(authorization) != FALSE_AUTHORIZATION:
        raise GgufReleaseV8Error(f"{label} grants an unsafe authorization")
    return authorization


def _common_postselection_receipt(
    *,
    path: Path,
    expected_sha256: str,
    label: str,
    expected_schema: str,
    expected_version: str,
    expected_status: str,
    expected_binding: Mapping[str, Any],
    evidence_keys: set[str],
    execution_keys: set[str],
) -> tuple[FileSnapshotV8, dict[str, Any]]:
    snapshot, receipt = _load_json(
        path,
        label=label,
        expected_sha256=expected_sha256,
    )
    _reject_legacy_schema(receipt, expected_schema=expected_schema, label=label)
    _require_exact_mapping(
        receipt,
        exact={
            "schema",
            "version",
            "status",
            "chain_binding",
            "evidence",
            "execution_boundary",
            "authorization",
            "canonical_digest_sha256",
        },
        label=label,
    )
    _verify_canonical_digest(receipt, label=label)
    if (
        receipt["version"] != expected_version
        or receipt["status"] != expected_status
        or receipt["chain_binding"] != expected_binding
    ):
        raise GgufReleaseV8Error(f"{label} identity or selection binding mismatch")
    _require_exact_mapping(
        receipt["evidence"],
        exact=evidence_keys,
        label=f"{label}.evidence",
    )
    _require_exact_mapping(
        receipt["execution_boundary"],
        exact=execution_keys,
        label=f"{label}.execution_boundary",
    )
    _false_authorization(receipt["authorization"], label=f"{label}.authorization")
    return snapshot, receipt


def _validate_generated_artifacts(
    root: Path,
    value: Any,
    *,
    expected_names: set[str],
    label: str,
) -> dict[str, Any]:
    artifacts = _require_exact_mapping(
        value,
        exact=expected_names,
        label=f"{label} artifacts",
    )
    verified: dict[str, Any] = {}
    for name in sorted(expected_names):
        if Path(name).name != name:
            raise GgufReleaseV8Error(f"{label} artifact name is unsafe: {name}")
        record = artifacts[name]
        if not isinstance(record, Mapping) or set(record) not in (
            {"bytes", "sha256"},
            {"bytes", "sha256", "records"},
        ):
            raise GgufReleaseV8Error(f"{label} artifact record is invalid: {name}")
        snapshot = _snapshot_file(
            root / name,
            label=f"{label} artifact {name}",
            expected_sha256=str(record["sha256"]),
            maximum_bytes=64 * 1024 * 1024,
        )
        if snapshot.path.parent != root.resolve(strict=True):
            raise GgufReleaseV8Error(f"{label} artifact escaped its output: {name}")
        if record["bytes"] != snapshot.bytes:
            raise GgufReleaseV8Error(f"{label} artifact byte mismatch: {name}")
        if "records" in record and (
            not isinstance(record["records"], int) or record["records"] <= 0
        ):
            raise GgufReleaseV8Error(f"{label} artifact record count is invalid: {name}")
        verified[name] = snapshot.descriptor()
    return verified


def _validate_calibration(
    inputs: ReleaseAuthorityInputsV8,
    *,
    binding: Mapping[str, Any],
    calibration_split: Mapping[str, Any],
    validation_split: Mapping[str, Any],
) -> tuple[FileSnapshotV8, dict[str, Any]]:
    snapshot, receipt = _load_json(
        inputs.calibration_receipt,
        label="calibration v8 receipt",
        expected_sha256=inputs.calibration_receipt_sha256,
    )
    _reject_legacy_schema(
        receipt,
        expected_schema=CALIBRATION_SCHEMA,
        label="calibration v8 receipt",
    )
    _require_exact_mapping(
        receipt,
        exact={
            "schema",
            "version",
            "status",
            "backend",
            "selection",
            "strict_v8_authority",
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
        },
        label="calibration v8 receipt",
    )
    _verify_canonical_digest(receipt, label="calibration v8 receipt")
    if (
        receipt["version"] != CALIBRATION_VERSION
        or receipt["status"] != CALIBRATION_STATUS
        or receipt["backend"] != "hf_model"
        or receipt["quality_gate_passed"] is not True
        or receipt["selection_locked"] is not True
        or receipt["checkpoint_reselection_performed"] is not False
    ):
        raise GgufReleaseV8Error("calibration v8 did not pass as bound HF evidence")
    selection = _require_exact_mapping(
        receipt["selection"],
        exact={
            "schema",
            "status",
            "receipt",
            "selection_locked",
            "selection_binding_digest_sha256",
            "checkpoint_id",
            "checkpoint_tree_sha256",
        },
        label="calibration v8 selection",
    )
    selection_file = selection["receipt"]
    if (
        selection["schema"] != selection_freeze_v8.SCHEMA
        or selection["status"] != selection_freeze_v8.STATUS
        or selection["selection_locked"] is not True
        or selection["selection_binding_digest_sha256"]
        != binding["selection_binding_digest_sha256"]
        or selection["checkpoint_id"] != binding["checkpoint_id"]
        or selection["checkpoint_tree_sha256"]
        != binding["checkpoint_tree_sha256"]
        or not isinstance(selection_file, Mapping)
        or selection_file.get("sha256") != binding["selection_freeze_sha256"]
    ):
        raise GgufReleaseV8Error("calibration v8 selection binding mismatch")
    authority = _require_exact_mapping(
        receipt["strict_v8_authority"],
        exact={
            "authority_sha256",
            "manifest_sha256",
            "train_sha256",
            "train_example_id_order_sha256",
            "validation_sha256",
            "validation_example_id_order_sha256",
            "training_gate_bundle_sha256",
            "inspected_input_sha256",
        },
        label="calibration v8 strict authority",
    )
    if (
        authority["manifest_sha256"] != binding["manifest_sha256"]
        or authority["validation_sha256"] != validation_split["sha256"]
    ):
        raise GgufReleaseV8Error("calibration v8 strict authority mismatch")
    for key, value in authority.items():
        _require_sha256(value, label=f"calibration v8 authority {key}")
    dataset = _require_exact_mapping(
        receipt["dataset"],
        exact={
            "split",
            "complete_split",
            "rows",
            "max_samples",
            "file",
            "train",
            "validation",
            "id_sets_pairwise_disjoint",
        },
        label="calibration v8 dataset",
    )
    calibration_file = dataset["file"]
    validation_file = dataset["validation"]
    if (
        dataset["split"] != "calibration"
        or dataset["complete_split"] is not True
        or dataset["rows"] != EXPECTED_ROWS
        or dataset["max_samples"] is not None
        or dataset["id_sets_pairwise_disjoint"] is not True
        or not isinstance(calibration_file, Mapping)
        or calibration_file.get("sha256") != calibration_split["sha256"]
        or calibration_file.get("rows") != EXPECTED_ROWS
        or not isinstance(validation_file, Mapping)
        or validation_file.get("sha256") != validation_split["sha256"]
    ):
        raise GgufReleaseV8Error("calibration v8 dataset binding is incomplete")
    model = receipt["model"]
    selected = model.get("selected_checkpoint") if isinstance(model, Mapping) else None
    if (
        not isinstance(model, Mapping)
        or model.get("tree_sha256") != binding["base_model_tree_sha256"]
        or not isinstance(selected, Mapping)
        or selected.get("checkpoint_id") != binding["checkpoint_id"]
        or selected.get("checkpoint_tree_sha256")
        != binding["checkpoint_tree_sha256"]
        or selected.get("adapter_tree_sha256") != binding["adapter_tree_sha256"]
        or model.get("model_bound") is not True
        or model.get("fixture_not_model_evidence") is not False
    ):
        raise GgufReleaseV8Error("calibration v8 model binding mismatch")
    authorization = _require_exact_mapping(
        receipt["authorization"],
        exact={
            "blind_test_authorized",
            "gguf_export_authorized",
            "x5_execution_authorized",
            "deployment_authorized",
            "production_integration_authorized",
        },
        label="calibration v8 authorization",
    )
    if set(authorization.values()) != {False}:
        raise GgufReleaseV8Error("calibration v8 grants an unsafe authorization")
    access = receipt["access_boundary"]
    if (
        not isinstance(access, Mapping)
        or access.get("selection_verified_before_calibration_path_construction")
        is not True
        or access.get("calibration_rows_accessed") != EXPECTED_ROWS
        or access.get("blind_path_constructed") is not False
        or access.get("blind_filesystem_metadata_accessed") is not False
        or access.get("blind_content_opened") is not False
        or access.get("blind_content_read") is not False
        or access.get("blind_content_hashed") is not False
        or access.get("x5_accessed") is not False
        or access.get("network_accessed") is not False
    ):
        raise GgufReleaseV8Error("calibration v8 crossed the post-selection boundary")
    _validate_generated_artifacts(
        snapshot.path.parent,
        receipt["artifacts"],
        expected_names={
            calibration_eval_v8.SAMPLE_FILENAME,
            calibration_eval_v8.SUMMARY_FILENAME,
        },
        label="calibration v8",
    )
    return snapshot, receipt


def _validate_ablation(
    inputs: ReleaseAuthorityInputsV8,
    *,
    binding: Mapping[str, Any],
    validation_split: Mapping[str, Any],
) -> tuple[FileSnapshotV8, dict[str, Any]]:
    snapshot, receipt = _load_json(
        inputs.ablation_receipt,
        label="ablation v8 receipt",
        expected_sha256=inputs.ablation_receipt_sha256,
    )
    _reject_legacy_schema(
        receipt,
        expected_schema=ABLATION_SCHEMA,
        label="ablation v8 receipt",
    )
    _require_exact_mapping(
        receipt,
        exact={
            "schema",
            "version",
            "status",
            "strict_v8_authority",
            "dataset",
            "execution",
            "model",
            "backend_bindings",
            "implementation",
            "artifacts",
            "invariants_passed",
            "reproducibility_payload_sha256",
            "authorization",
            "access_boundary",
            "methodology",
            "claim_boundary",
            "canonical_digest_sha256",
        },
        label="ablation v8 receipt",
    )
    _verify_canonical_digest(receipt, label="ablation v8 receipt")
    if (
        receipt["version"] != ABLATION_VERSION
        or receipt["status"] != ABLATION_STATUS
        or receipt["invariants_passed"] is not True
    ):
        raise GgufReleaseV8Error("ablation v8 did not pass as model evidence")
    authority = receipt["strict_v8_authority"]
    if (
        not isinstance(authority, Mapping)
        or authority.get("selection_freeze_sha256")
        != binding["selection_freeze_sha256"]
        or authority.get("selection_binding_digest_sha256")
        != binding["selection_binding_digest_sha256"]
        or authority.get("manifest_sha256") != binding["manifest_sha256"]
        or authority.get("validation_sha256") != validation_split["sha256"]
        or authority.get("selected_checkpoint_id") != binding["checkpoint_id"]
        or authority.get("selected_checkpoint_tree_sha256")
        != binding["checkpoint_tree_sha256"]
        or authority.get("selected_adapter_tree_sha256")
        != binding["adapter_tree_sha256"]
        or authority.get("base_model_tree_sha256")
        != binding["base_model_tree_sha256"]
        or authority.get("selection_status") != selection_freeze_v8.STATUS
        or authority.get("selection_verified_status")
        != selection_freeze_v8.VERIFIED_STATUS
    ):
        raise GgufReleaseV8Error("ablation v8 strict authority mismatch")
    dataset = _require_exact_mapping(
        receipt["dataset"],
        exact={"split", "complete_split", "rows", "max_samples", "sha256"},
        label="ablation v8 dataset",
    )
    if dataset != {
        "split": "validation",
        "complete_split": True,
        "rows": EXPECTED_ROWS,
        "max_samples": None,
        "sha256": validation_split["sha256"],
    }:
        raise GgufReleaseV8Error("ablation v8 dataset binding is incomplete")
    execution = receipt["execution"]
    if (
        not isinstance(execution, Mapping)
        or execution.get("sample_rows") != ablation_eval_v8.EXPECTED_SAMPLE_ROWS
        or execution.get("same_requests_for_base_and_adapter") is not True
        or execution.get("expected_passed_to_model") is not False
        or execution.get("expected_passed_to_candidate_compiler") is not False
        or execution.get("selection_locked_before_ablation") is not True
        or execution.get("selection_policy_called") is not False
        or execution.get("automatic_model_selection") is not False
        or execution.get("checkpoint_reselection_performed") is not False
        or execution.get("synthetic_evidence_added") is not False
    ):
        raise GgufReleaseV8Error("ablation v8 execution matrix is unsafe")
    model = _require_exact_mapping(
        receipt["model"],
        exact={
            "base_model_path",
            "base_model_tree_sha256",
            "selected_checkpoint_id",
            "selected_checkpoint_path",
            "selected_checkpoint_tree_sha256",
            "selected_adapter_tree_sha256",
            "model_bound",
            "fixture_not_model_evidence",
        },
        label="ablation v8 model",
    )
    if (
        model["base_model_tree_sha256"] != binding["base_model_tree_sha256"]
        or model["selected_checkpoint_id"] != binding["checkpoint_id"]
        or model["selected_checkpoint_tree_sha256"]
        != binding["checkpoint_tree_sha256"]
        or model["selected_adapter_tree_sha256"] != binding["adapter_tree_sha256"]
        or model["model_bound"] is not True
        or model["fixture_not_model_evidence"] is not False
    ):
        raise GgufReleaseV8Error("ablation v8 model binding mismatch")
    authorization = _require_exact_mapping(
        receipt["authorization"],
        exact={
            "checkpoint_reselection_allowed",
            "calibration_authorized",
            "blind_test_authorized",
            "gguf_export_authorized",
            "x5_execution_authorized",
            "deployment_authorized",
            "production_integration_authorized",
        },
        label="ablation v8 authorization",
    )
    if set(authorization.values()) != {False}:
        raise GgufReleaseV8Error("ablation v8 grants an unsafe authorization")
    access = receipt["access_boundary"]
    if (
        not isinstance(access, Mapping)
        or access.get("validation_content_accessed") is not True
        or access.get("validation_rows_accessed") != EXPECTED_ROWS
        or access.get("train_content_accessed") is not False
        or access.get("calibration_path_constructed") is not False
        or access.get("calibration_filesystem_metadata_accessed") is not False
        or access.get("calibration_content_accessed") is not False
        or access.get("blind_path_constructed") is not False
        or access.get("blind_filesystem_metadata_accessed") is not False
        or access.get("blind_content_accessed") is not False
        or access.get("blind_content_hashed") is not False
        or access.get("x5_accessed") is not False
        or access.get("network_accessed") is not False
    ):
        raise GgufReleaseV8Error("ablation v8 crossed the validation-only boundary")
    methodology = _require_exact_mapping(
        receipt["methodology"],
        exact={
            "calibration_required_for_ablation",
            "reason",
            "independent_generalization_estimate",
            "reason_not_independent",
        },
        label="ablation v8 methodology",
    )
    if (
        methodology["calibration_required_for_ablation"] is not False
        or methodology["independent_generalization_estimate"] is not False
    ):
        raise GgufReleaseV8Error("ablation v8 overstates its evidence")
    _validate_generated_artifacts(
        snapshot.path.parent,
        receipt["artifacts"],
        expected_names={
            ablation_eval_v8.SAMPLE_FILENAME,
            *ablation_eval_v8.REPORT_FILENAMES,
        },
        label="ablation v8",
    )
    return snapshot, receipt


def _artifact_from_record(value: Any, *, label: str) -> dict[str, Any]:
    record = _require_exact_mapping(
        value,
        exact={"path", "bytes", "sha256"},
        label=label,
    )
    snapshot = _snapshot_file(
        Path(str(record["path"])),
        label=label,
        expected_sha256=str(record["sha256"]),
        maximum_bytes=64 * 1024 * 1024,
    )
    if record["bytes"] != snapshot.bytes:
        raise GgufReleaseV8Error(f"{label} byte count mismatch")
    return snapshot.descriptor()


def _validate_postfreeze(
    inputs: ReleaseAuthorityInputsV8,
    *,
    binding: Mapping[str, Any],
    calibration: FileSnapshotV8,
    ablation: FileSnapshotV8,
) -> tuple[FileSnapshotV8, dict[str, Any], dict[str, Any]]:
    snapshot, receipt = _load_json(
        inputs.postfreeze_receipt,
        label="postfreeze v8 receipt",
        expected_sha256=inputs.postfreeze_receipt_sha256,
    )
    _reject_legacy_schema(
        receipt,
        expected_schema=POSTFREEZE_SCHEMA,
        label="postfreeze v8 receipt",
    )
    _require_exact_mapping(
        receipt,
        exact={
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
        },
        label="postfreeze v8 receipt",
    )
    _verify_canonical_digest(receipt, label="postfreeze v8 receipt")
    if (
        receipt["version"] != POSTFREEZE_VERSION
        or receipt["status"] != POSTFREEZE_STATUS
        or receipt["chain_binding"] != binding
    ):
        raise GgufReleaseV8Error("postfreeze v8 identity or selection binding mismatch")
    upstream = _require_exact_mapping(
        receipt["upstream_receipts"],
        exact={"selection_freeze_sha256", "calibration_sha256", "ablation_sha256"},
        label="postfreeze v8 upstream",
    )
    if upstream != {
        "selection_freeze_sha256": binding["selection_freeze_sha256"],
        "calibration_sha256": calibration.sha256,
        "ablation_sha256": ablation.sha256,
    }:
        raise GgufReleaseV8Error("postfreeze v8 upstream receipt binding mismatch")
    dataset = _require_exact_mapping(
        receipt["dataset"],
        exact={
            "rows_read_once",
            "blind_sha256",
            "nonblind_manifest_sha256",
            "preblind_commitment_sha256",
        },
        label="postfreeze v8 dataset",
    )
    if (
        dataset["rows_read_once"] != EXPECTED_ROWS
        or dataset["nonblind_manifest_sha256"] != binding["manifest_sha256"]
        or dataset["preblind_commitment_sha256"]
        != binding["preblind_commitment_sha256"]
    ):
        raise GgufReleaseV8Error("postfreeze v8 dataset binding is incomplete")
    _require_sha256(dataset["blind_sha256"], label="postfreeze blind SHA-256")
    execution = _require_exact_mapping(
        receipt["execution_boundary"],
        exact={
            "backend",
            "model_selection_performed",
            "checkpoint_reselection_performed",
            "training_performed",
            "calibration_performed",
        },
        label="postfreeze v8 execution boundary",
    )
    if execution != {
        "backend": "hf_model",
        "model_selection_performed": False,
        "checkpoint_reselection_performed": False,
        "training_performed": False,
        "calibration_performed": False,
    }:
        raise GgufReleaseV8Error("postfreeze v8 execution boundary is unsafe")
    _false_authorization(
        receipt["authorization"],
        label="postfreeze v8 authorization",
    )
    claim = _require_exact_mapping(
        receipt["consumption_claim"],
        exact={"sha256", "nonce_sha256", "failure_is_non_reusable"},
        label="postfreeze v8 consumption claim",
    )
    _require_sha256(claim["sha256"], label="postfreeze claim SHA-256")
    _require_sha256(claim["nonce_sha256"], label="postfreeze claim nonce SHA-256")
    if claim["failure_is_non_reusable"] is not True:
        raise GgufReleaseV8Error("postfreeze v8 claim is reusable")
    artifacts = _require_exact_mapping(
        receipt["artifacts"],
        exact={"sample_results", "summary"},
        label="postfreeze v8 artifacts",
    )
    verified_artifacts = {
        role: _artifact_from_record(record, label=f"postfreeze v8 {role}")
        for role, record in artifacts.items()
    }
    if (
        verified_artifacts["sample_results"]["path"]
        == verified_artifacts["summary"]["path"]
        or verified_artifacts["sample_results"]["sha256"]
        == verified_artifacts["summary"]["sha256"]
    ):
        raise GgufReleaseV8Error("postfreeze v8 artifacts must be distinct")
    return snapshot, receipt, verified_artifacts


def _validate_qualification(
    inputs: ReleaseAuthorityInputsV8,
    *,
    binding: Mapping[str, Any],
    calibration: FileSnapshotV8,
    ablation: FileSnapshotV8,
    postfreeze: FileSnapshotV8,
    postfreeze_receipt: Mapping[str, Any],
    postfreeze_artifacts: Mapping[str, Any],
) -> tuple[FileSnapshotV8, dict[str, Any]]:
    snapshot, receipt = _load_json(
        inputs.qualification_receipt,
        label="postfreeze v8 qualification",
        expected_sha256=inputs.qualification_receipt_sha256,
    )
    _reject_legacy_schema(
        receipt,
        expected_schema=QUALIFICATION_SCHEMA,
        label="postfreeze v8 qualification",
    )
    _require_exact_mapping(
        receipt,
        exact={
            "schema",
            "version",
            "status",
            "qualified",
            "chain_binding",
            "upstream_receipts",
            "blind_run_receipt",
            "consumption_claim",
            "gate_results",
            "release_authorization",
            "artifacts",
            "canonical_digest_sha256",
        },
        label="postfreeze v8 qualification",
    )
    _verify_canonical_digest(receipt, label="postfreeze v8 qualification")
    if (
        receipt["version"] != QUALIFICATION_VERSION
        or receipt["status"] != QUALIFICATION_STATUS
        or receipt["qualified"] is not True
        or receipt["chain_binding"] != binding
    ):
        raise GgufReleaseV8Error("postfreeze v8 qualification did not pass")
    upstream = _require_exact_mapping(
        receipt["upstream_receipts"],
        exact={
            "selection_freeze_sha256",
            "calibration_sha256",
            "ablation_sha256",
            "postfreeze_sha256",
        },
        label="postfreeze v8 qualification upstream",
    )
    if upstream != {
        "selection_freeze_sha256": binding["selection_freeze_sha256"],
        "calibration_sha256": calibration.sha256,
        "ablation_sha256": ablation.sha256,
        "postfreeze_sha256": postfreeze.sha256,
    }:
        raise GgufReleaseV8Error("postfreeze v8 qualification upstream mismatch")
    run = _require_exact_mapping(
        receipt["blind_run_receipt"],
        exact={"sha256", "schema", "status"},
        label="postfreeze v8 qualification blind run",
    )
    if run != {
        "sha256": postfreeze.sha256,
        "schema": POSTFREEZE_SCHEMA,
        "status": POSTFREEZE_STATUS,
    }:
        raise GgufReleaseV8Error("postfreeze v8 qualification run binding mismatch")
    if receipt["consumption_claim"] != postfreeze_receipt["consumption_claim"]:
        raise GgufReleaseV8Error("postfreeze v8 qualification claim mismatch")
    gates = receipt["gate_results"]
    if (
        not isinstance(gates, Sequence)
        or isinstance(gates, (str, bytes))
        or not gates
        or any(
            not isinstance(gate, Mapping)
            or set(gate) != {"name", "passed"}
            or not isinstance(gate["name"], str)
            or gate["passed"] is not True
            for gate in gates
        )
    ):
        raise GgufReleaseV8Error("postfreeze v8 qualification has a failed gate")
    release = _require_exact_mapping(
        receipt["release_authorization"],
        exact={"gguf_export_authorized", *FALSE_AUTHORIZATION},
        label="postfreeze v8 release authorization",
    )
    if release != {"gguf_export_authorized": True, **FALSE_AUTHORIZATION}:
        raise GgufReleaseV8Error("postfreeze v8 qualification grants an unsafe release")
    if receipt["artifacts"] != postfreeze_artifacts:
        raise GgufReleaseV8Error("postfreeze v8 qualification artifact mismatch")
    return snapshot, receipt


def _validate_independent_verification(
    inputs: ReleaseAuthorityInputsV8,
    *,
    binding: Mapping[str, Any],
    calibration: FileSnapshotV8,
    ablation: FileSnapshotV8,
    postfreeze: FileSnapshotV8,
    qualification: FileSnapshotV8,
) -> tuple[FileSnapshotV8, dict[str, Any]]:
    snapshot, receipt = _load_json(
        inputs.verification_receipt,
        label="postfreeze v8 independent verification",
        expected_sha256=inputs.verification_receipt_sha256,
    )
    _reject_legacy_schema(
        receipt,
        expected_schema=VERIFICATION_SCHEMA,
        label="postfreeze v8 independent verification",
    )
    _require_exact_mapping(
        receipt,
        exact={
            "schema",
            "version",
            "status",
            "chain_binding",
            "verified_receipts",
            "independent_recomputation",
            "release_authorization",
            "canonical_digest_sha256",
        },
        label="postfreeze v8 independent verification",
    )
    _verify_canonical_digest(
        receipt,
        label="postfreeze v8 independent verification",
    )
    if (
        receipt["version"] != VERIFICATION_VERSION
        or receipt["status"] != VERIFICATION_STATUS
        or receipt["chain_binding"] != binding
    ):
        raise GgufReleaseV8Error("postfreeze v8 independent verification failed")
    verified = _require_exact_mapping(
        receipt["verified_receipts"],
        exact={
            "selection_freeze_sha256",
            "calibration_sha256",
            "ablation_sha256",
            "postfreeze_sha256",
            "qualification_sha256",
        },
        label="postfreeze v8 independently verified receipts",
    )
    if verified != {
        "selection_freeze_sha256": binding["selection_freeze_sha256"],
        "calibration_sha256": calibration.sha256,
        "ablation_sha256": ablation.sha256,
        "postfreeze_sha256": postfreeze.sha256,
        "qualification_sha256": qualification.sha256,
    }:
        raise GgufReleaseV8Error("independent verifier receipt binding mismatch")
    recomputed = _require_exact_mapping(
        receipt["independent_recomputation"],
        exact={
            "selection_reverified",
            "calibration_samples_recomputed",
            "calibration_summary_recomputed",
            "ablation_matrix_recomputed",
            "ablation_reports_recomputed",
            "blind_samples_recomputed",
            "blind_summary_recomputed",
            "qualification_recomputed",
        },
        label="postfreeze v8 recomputation",
    )
    if set(recomputed.values()) != {True}:
        raise GgufReleaseV8Error("postfreeze v8 evidence was not independently recomputed")
    release = _require_exact_mapping(
        receipt["release_authorization"],
        exact={"gguf_export_authorized", *FALSE_AUTHORIZATION},
        label="postfreeze v8 verifier release authorization",
    )
    if release != {"gguf_export_authorized": True, **FALSE_AUTHORIZATION}:
        raise GgufReleaseV8Error("postfreeze verifier grants an unsafe release")
    return snapshot, receipt


def validate_authority_chain_v8(
    inputs: ReleaseAuthorityInputsV8,
) -> dict[str, Any]:
    """Validate the complete native-v8 authority chain without exporting."""

    binding, selection_evidence, _ = _selection_binding(inputs)
    splits = selection_evidence["splits"]
    calibration, _ = _validate_calibration(
        inputs,
        binding=binding,
        calibration_split=splits["calibration"],
        validation_split=splits["validation"],
    )
    ablation, _ = _validate_ablation(
        inputs,
        binding=binding,
        validation_split=splits["validation"],
    )
    postfreeze, postfreeze_receipt, postfreeze_artifacts = _validate_postfreeze(
        inputs,
        binding=binding,
        calibration=calibration,
        ablation=ablation,
    )
    qualification, _ = _validate_qualification(
        inputs,
        binding=binding,
        calibration=calibration,
        ablation=ablation,
        postfreeze=postfreeze,
        postfreeze_receipt=postfreeze_receipt,
        postfreeze_artifacts=postfreeze_artifacts,
    )
    verification, _ = _validate_independent_verification(
        inputs,
        binding=binding,
        calibration=calibration,
        ablation=ablation,
        postfreeze=postfreeze,
        qualification=qualification,
    )
    receipts = {
        "calibration": calibration.descriptor(),
        "ablation": ablation.descriptor(),
        "postfreeze": postfreeze.descriptor(),
        "qualification": qualification.descriptor(),
        "independent_verification": verification.descriptor(),
    }
    authority_digest = canonical_sha256(
        {
            "chain_binding": binding,
            "receipts": {
                role: descriptor["sha256"]
                for role, descriptor in sorted(receipts.items())
            },
        }
    )
    return {
        "chain_binding": binding,
        "selection_evidence": selection_evidence,
        "receipts": receipts,
        "postfreeze_artifacts": postfreeze_artifacts,
        "authority_digest_sha256": authority_digest,
        "gguf_export_authorized": True,
        **FALSE_AUTHORIZATION,
    }


def _tool_file(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    snapshot = _snapshot_binary_file(
        path,
        label=label,
        expected_sha256=expected_sha256,
        maximum_bytes=1024 * 1024 * 1024,
    )
    return {
        **snapshot.descriptor(),
        "expected_sha256": expected_sha256,
        "sha256_match": True,
    }


def preflight_release_v8(
    inputs: ReleaseAuthorityInputsV8,
    *,
    toolchain: ToolchainInputsV8 | None = None,
) -> dict[str, Any]:
    """Authorize one future offline export without invoking any ML runtime."""

    authority = validate_authority_chain_v8(inputs)
    tools = toolchain or ToolchainInputsV8()
    low_level = gguf_export_v5.preflight_gguf_export(
        gguf_export_v5.ExportInputs(
            base_model=inputs.base_model_dir,
            adapter=inputs.selected_adapter_dir,
            converter=tools.converter,
            quantizer=tools.quantizer,
            converter_sha256=tools.converter_sha256,
            quantizer_sha256=tools.quantizer_sha256,
        )
    )
    if (
        low_level.get("status")
        != "PASS_READ_ONLY_GGUF_EXPORT_PREFLIGHT_NOT_EXPORTED"
        or low_level.get("read_only") is not True
        or low_level.get("network_used") is not False
        or low_level.get("x5_touched") is not False
    ):
        raise GgufReleaseV8Error("legacy low-level exporter preflight is unsafe")
    model = authority["selection_evidence"]["model"]
    if (
        low_level["base_model"]["tree_sha256"]
        != model["base_model_tree_sha256"]
        or low_level["adapter"]["tree_sha256"] != model["adapter_tree_sha256"]
    ):
        raise GgufReleaseV8Error("low-level exporter inputs differ from v8 authority")
    llama_server = _tool_file(
        tools.llama_server,
        expected_sha256=tools.llama_server_sha256,
        label="llama-server parity runtime",
    )
    tool_binding = {
        "converter_sha256": low_level["tools"]["converter"]["sha256"],
        "converter_runtime_tree_sha256": low_level["tools"]["converter"][
            "runtime_tree"
        ]["tree_sha256"],
        "quantizer_sha256": low_level["tools"]["quantizer"]["sha256"],
        "quantizer_runtime_tree_sha256": low_level["tools"]["quantizer"][
            "runtime_tree"
        ]["tree_sha256"],
        "llama_server_sha256": llama_server["sha256"],
    }
    authorization_digest = canonical_sha256(
        {
            "authority_digest_sha256": authority["authority_digest_sha256"],
            "tool_binding": tool_binding,
            "parity_protocol": PARITY_PROTOCOL,
            "replay_protocol": REPLAY_PROTOCOL,
        }
    )
    report = {
        "schema": PREFLIGHT_SCHEMA,
        "version": VERSION,
        "status": PREFLIGHT_PASS_STATUS,
        "read_only": True,
        "export_performed": False,
        "reserved_blind_dataset_read_by_this_preflight": False,
        "postfreeze_evidence_artifacts_hashed": True,
        "network_used": False,
        "x5_contacted": False,
        "chain_binding": authority["chain_binding"],
        "authority_digest_sha256": authority["authority_digest_sha256"],
        "authority_receipts": authority["receipts"],
        "postfreeze_artifacts": authority["postfreeze_artifacts"],
        "tool_binding": tool_binding,
        "llama_server": llama_server,
        "low_level_export_preflight": low_level,
        "required_followup_protocols": {
            "hf_gguf_parity": PARITY_PROTOCOL,
            "x5_replay": REPLAY_PROTOCOL,
            "legacy_hf_gguf_parity_v5_allowed": False,
            "legacy_x5_gguf_replay_allowed": False,
        },
        "authorization": {
            "gguf_export_authorized": True,
            **FALSE_AUTHORIZATION,
        },
        "authorization_digest_sha256": authorization_digest,
        "claim_boundary": (
            "This preflight authorizes only one future local-PC offline GGUF "
            "export from the bound v8 base and adapter. It performs no export, "
            "parity run, X5 replay, activation, deployment, or production change."
        ),
    }
    report["canonical_digest_sha256"] = canonical_sha256(report)
    return report


def _validate_preflight_report_v8(report: Mapping[str, Any]) -> str:
    _require_exact_mapping(
        report,
        exact=_PREFLIGHT_FIELDS,
        label="v8 GGUF preflight",
    )
    _verify_canonical_digest(report, label="v8 GGUF preflight")
    if (
        report["schema"] != PREFLIGHT_SCHEMA
        or report["version"] != VERSION
        or report["status"] != PREFLIGHT_PASS_STATUS
        or report["read_only"] is not True
        or report["export_performed"] is not False
        or report["reserved_blind_dataset_read_by_this_preflight"] is not False
        or report["postfreeze_evidence_artifacts_hashed"] is not True
        or report["network_used"] is not False
        or report["x5_contacted"] is not False
    ):
        raise GgufReleaseV8Error("v8 GGUF preflight boundary is unsafe")
    authorization = _require_exact_mapping(
        report["authorization"],
        exact={"gguf_export_authorized", *FALSE_AUTHORIZATION},
        label="v8 GGUF preflight authorization",
    )
    if authorization != {
        "gguf_export_authorized": True,
        **FALSE_AUTHORIZATION,
    }:
        raise GgufReleaseV8Error("v8 GGUF preflight authorization is unsafe")
    protocols = _require_exact_mapping(
        report["required_followup_protocols"],
        exact={
            "hf_gguf_parity",
            "x5_replay",
            "legacy_hf_gguf_parity_v5_allowed",
            "legacy_x5_gguf_replay_allowed",
        },
        label="v8 GGUF preflight follow-up protocols",
    )
    if protocols != {
        "hf_gguf_parity": PARITY_PROTOCOL,
        "x5_replay": REPLAY_PROTOCOL,
        "legacy_hf_gguf_parity_v5_allowed": False,
        "legacy_x5_gguf_replay_allowed": False,
    }:
        raise GgufReleaseV8Error("v8 GGUF follow-up protocol boundary is unsafe")
    authority_digest = _require_sha256(
        report["authority_digest_sha256"],
        label="v8 GGUF authority digest",
    )
    tool_binding = _require_exact_mapping(
        report["tool_binding"],
        exact={
            "converter_sha256",
            "converter_runtime_tree_sha256",
            "quantizer_sha256",
            "quantizer_runtime_tree_sha256",
            "llama_server_sha256",
        },
        label="v8 GGUF tool binding",
    )
    for name, value in tool_binding.items():
        _require_sha256(value, label=f"v8 GGUF tool binding {name}")
    digest = _require_sha256(
        report["authorization_digest_sha256"],
        label="v8 GGUF export authorization digest",
    )
    expected = canonical_sha256(
        {
            "authority_digest_sha256": authority_digest,
            "tool_binding": dict(tool_binding),
            "parity_protocol": PARITY_PROTOCOL,
            "replay_protocol": REPLAY_PROTOCOL,
        }
    )
    if digest != expected:
        raise GgufReleaseV8Error("v8 GGUF export authorization digest mismatch")
    return digest


def _registry_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _exclusive_registry_create(
    path: Path,
    payload: bytes,
    *,
    registry_identity: tuple[int, int, int, int, int],
) -> FileSnapshotV8:
    root, observed_identity = _production_registry_root(create=False)
    if path.parent != root or observed_identity[:2] != registry_identity[:2]:
        raise GgufReleaseV8Error("GGUF authorization registry changed before claim")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # A partial claim is deliberate crash evidence and is never reusable.
        raise
    _verify_registry_identity(root, registry_identity)
    snapshot = _snapshot_file(
        path,
        label=f"published {path.name}",
        maximum_bytes=1024 * 1024,
    )
    if snapshot.payload != payload:
        raise GgufReleaseV8Error(f"published {path.name} changed")
    return snapshot


def _claim_body_v8(report: Mapping[str, Any]) -> dict[str, Any]:
    digest = _validate_preflight_report_v8(report)
    body = {
        "schema": CLAIM_SCHEMA,
        "version": VERSION,
        "status": CLAIM_STATUS,
        "registry_key": digest,
        "authorization_digest_sha256": digest,
        "preflight_canonical_digest_sha256": report[
            "canonical_digest_sha256"
        ],
        "authority_digest_sha256": report["authority_digest_sha256"],
        "tool_binding_digest_sha256": canonical_sha256(report["tool_binding"]),
        "failure_is_non_reusable": True,
        "retry_allowed": False,
        "export_performed": False,
        "authorization": {
            "gguf_export_authorized": False,
            **FALSE_AUTHORIZATION,
        },
    }
    return {**body, "canonical_digest_sha256": canonical_sha256(body)}


def _validate_claim_receipt_v8(
    path: Path,
    *,
    report: Mapping[str, Any],
) -> tuple[FileSnapshotV8, dict[str, Any]]:
    snapshot, claim = _load_json(path, label="v8 GGUF export claim")
    _require_exact_mapping(
        claim,
        exact=_CLAIM_FIELDS,
        label="v8 GGUF export claim",
    )
    _verify_canonical_digest(claim, label="v8 GGUF export claim")
    digest = _validate_preflight_report_v8(report)
    expected_authorization = {
        "gguf_export_authorized": False,
        **FALSE_AUTHORIZATION,
    }
    if (
        claim["schema"] != CLAIM_SCHEMA
        or claim["version"] != VERSION
        or claim["status"] != CLAIM_STATUS
        or claim["registry_key"] != digest
        or claim["authorization_digest_sha256"] != digest
        or claim["authority_digest_sha256"]
        != report["authority_digest_sha256"]
        or claim["tool_binding_digest_sha256"]
        != canonical_sha256(report["tool_binding"])
        or claim["failure_is_non_reusable"] is not True
        or claim["retry_allowed"] is not False
        or claim["export_performed"] is not False
        or claim["authorization"] != expected_authorization
    ):
        raise GgufReleaseV8Error("v8 GGUF export claim binding mismatch")
    _require_sha256(
        claim["preflight_canonical_digest_sha256"],
        label="claimed v8 GGUF preflight canonical digest",
    )
    return snapshot, claim


def _terminal_body_v8(
    report: Mapping[str, Any],
    *,
    claim_snapshot: FileSnapshotV8,
    claim: Mapping[str, Any],
) -> dict[str, Any]:
    digest = _validate_preflight_report_v8(report)
    body = {
        "schema": TERMINAL_SCHEMA,
        "version": VERSION,
        "status": TERMINAL_STATUS,
        "registry_key": digest,
        "authorization_digest_sha256": digest,
        "preflight_canonical_digest_sha256": claim[
            "preflight_canonical_digest_sha256"
        ],
        "claim_sha256": claim_snapshot.sha256,
        "claim_canonical_digest_sha256": claim["canonical_digest_sha256"],
        "failure_is_non_reusable": True,
        "retry_allowed": False,
        "export_performed": False,
        "authorization": {
            "gguf_export_authorized": True,
            **FALSE_AUTHORIZATION,
        },
    }
    return {**body, "canonical_digest_sha256": canonical_sha256(body)}


def _validate_terminal_receipt_v8(
    path: Path,
    *,
    report: Mapping[str, Any],
    claim_snapshot: FileSnapshotV8,
    claim: Mapping[str, Any],
) -> tuple[FileSnapshotV8, dict[str, Any]]:
    snapshot, terminal = _load_json(path, label="v8 GGUF export terminal")
    _require_exact_mapping(
        terminal,
        exact=_TERMINAL_FIELDS,
        label="v8 GGUF export terminal",
    )
    _verify_canonical_digest(terminal, label="v8 GGUF export terminal")
    expected = _terminal_body_v8(
        report,
        claim_snapshot=claim_snapshot,
        claim=claim,
    )
    if terminal != expected:
        raise GgufReleaseV8Error("v8 GGUF export terminal binding mismatch")
    return snapshot, terminal


def _reject_existing_registry_state_v8(
    paths: Mapping[str, Path],
    *,
    report: Mapping[str, Any],
) -> None:
    claim_exists = os.path.lexists(paths["claim"])
    terminal_exists = os.path.lexists(paths["terminal"])
    if claim_exists:
        try:
            claim_snapshot, claim = _validate_claim_receipt_v8(
                paths["claim"],
                report=report,
            )
        except (GgufReleaseV8Error, OSError, ValueError) as exc:
            raise GgufReleaseV8Error(
                "existing v8 GGUF export claim is empty or malformed; "
                "authorization is non-reusable"
            ) from exc
        if terminal_exists:
            try:
                _validate_terminal_receipt_v8(
                    paths["terminal"],
                    report=report,
                    claim_snapshot=claim_snapshot,
                    claim=claim,
                )
            except (GgufReleaseV8Error, OSError, ValueError) as exc:
                raise GgufReleaseV8Error(
                    "existing v8 GGUF export terminal is malformed; "
                    "authorization is non-reusable"
                ) from exc
            raise GgufReleaseV8Error(
                "v8 GGUF export authorization was already claimed and terminalized"
            )
        raise GgufReleaseV8Error(
            "v8 GGUF export authorization is already claimed or crash-resident"
        )
    if terminal_exists:
        raise GgufReleaseV8Error(
            "v8 GGUF export terminal exists without its claim; "
            "authorization is non-reusable"
        )


def _claim_validated_preflight_v8(report: Mapping[str, Any]) -> dict[str, Any]:
    digest = _validate_preflight_report_v8(report)
    paths, registry_identity = _registry_paths_v8(digest, create=True)
    _reject_existing_registry_state_v8(paths, report=report)

    claim = _claim_body_v8(report)
    try:
        claim_snapshot = _exclusive_registry_create(
            paths["claim"],
            _registry_json_bytes(claim),
            registry_identity=registry_identity,
        )
    except FileExistsError as exc:
        raise GgufReleaseV8Error(
            "v8 GGUF export authorization was concurrently claimed"
        ) from exc
    persisted_claim_snapshot, persisted_claim = _validate_claim_receipt_v8(
        paths["claim"],
        report=report,
    )
    if persisted_claim_snapshot != claim_snapshot:
        raise GgufReleaseV8Error("v8 GGUF export claim changed after publication")

    terminal = _terminal_body_v8(
        report,
        claim_snapshot=claim_snapshot,
        claim=persisted_claim,
    )
    try:
        terminal_snapshot = _exclusive_registry_create(
            paths["terminal"],
            _registry_json_bytes(terminal),
            registry_identity=registry_identity,
        )
    except FileExistsError as exc:
        raise GgufReleaseV8Error(
            "v8 GGUF export terminal was concurrently occupied; "
            "claim remains non-reusable"
        ) from exc
    persisted_terminal_snapshot, _ = _validate_terminal_receipt_v8(
        paths["terminal"],
        report=report,
        claim_snapshot=claim_snapshot,
        claim=persisted_claim,
    )
    if persisted_terminal_snapshot != terminal_snapshot:
        raise GgufReleaseV8Error(
            "v8 GGUF export terminal changed after publication"
        )
    return {
        "status": TERMINAL_STATUS,
        "authorization_digest_sha256": digest,
        "registry_root": str(paths["root"]),
        "claim": claim_snapshot.descriptor(),
        "terminal": terminal_snapshot.descriptor(),
        "failure_is_non_reusable": True,
        "retry_allowed": False,
        "export_performed": False,
        "authorization": {
            "gguf_export_authorized": True,
            **FALSE_AUTHORIZATION,
        },
    }


def claim_export_authorization_v8(
    inputs: ReleaseAuthorityInputsV8,
    *,
    toolchain: ToolchainInputsV8 | None = None,
    preflight_output: Path | None = None,
) -> dict[str, Any]:
    """Irreversibly issue the sole bound offline-export authorization."""

    report = preflight_release_v8(inputs, toolchain=toolchain)
    written_to: Path | None = None
    if preflight_output is not None:
        written_to = write_preflight_v8(preflight_output, report)
    claim = _claim_validated_preflight_v8(report)
    return {
        **claim,
        "preflight": {
            "schema": report["schema"],
            "status": report["status"],
            "canonical_digest_sha256": report["canonical_digest_sha256"],
            "authorization_digest_sha256": report[
                "authorization_digest_sha256"
            ],
            "written_to": None if written_to is None else str(written_to),
        },
    }


def write_preflight_v8(path: Path, report: Mapping[str, Any]) -> Path:
    """Publish one immutable preflight receipt without overwriting."""

    _validate_preflight_report_v8(report)
    output = Path(path).expanduser().absolute()
    parent = output.parent.resolve(strict=True)
    if not parent.is_dir() or os.path.lexists(output):
        raise GgufReleaseV8Error("preflight output must be a new file")
    payload = (
        json.dumps(
            dict(report),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        parent / output.name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if os.path.lexists(output):
            os.unlink(output)
        raise
    return output.resolve(strict=True)


__all__ = [
    "ABLATION_SCHEMA",
    "ABLATION_STATUS",
    "ABLATION_VERSION",
    "CALIBRATION_SCHEMA",
    "CALIBRATION_STATUS",
    "CALIBRATION_VERSION",
    "CLAIM_SCHEMA",
    "CLAIM_STATUS",
    "FALSE_AUTHORIZATION",
    "GgufReleaseV8Error",
    "PARITY_PROTOCOL",
    "POSTFREEZE_SCHEMA",
    "POSTFREEZE_STATUS",
    "POSTFREEZE_VERSION",
    "PREFLIGHT_FAIL_STATUS",
    "PREFLIGHT_PASS_STATUS",
    "PREFLIGHT_SCHEMA",
    "PRODUCTION_REGISTRY_ROOT",
    "QUALIFICATION_SCHEMA",
    "QUALIFICATION_STATUS",
    "QUALIFICATION_VERSION",
    "REPLAY_PROTOCOL",
    "ReleaseAuthorityInputsV8",
    "ToolchainInputsV8",
    "TERMINAL_SCHEMA",
    "TERMINAL_STATUS",
    "VERIFICATION_SCHEMA",
    "VERIFICATION_STATUS",
    "VERIFICATION_VERSION",
    "VERSION",
    "canonical_sha256",
    "claim_export_authorization_v8",
    "classify_receipt_schema_v8",
    "preflight_release_v8",
    "sha256_file",
    "validate_authority_chain_v8",
    "write_preflight_v8",
]
