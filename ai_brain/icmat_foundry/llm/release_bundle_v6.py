"""Build and verify an immutable PC-offline ICMat Pointer v6 bundle.

The bundle is deliberately not an X5 deployment package.  It contains only a
frozen GGUF candidate, its two runtime contracts, and the immutable evidence
receipts needed to audit how that candidate was produced.  No activation,
service, port, production, or RB-VoE state is included.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

from icmat_foundry.llm import (
    ablation_eval_v6,
    blind_protocol_v6,
    calibration_eval_v6,
    gguf_release_v6,
    selection_freeze_v6,
)

PACKAGE_TYPE = "PC_OFFLINE_CANDIDATE_NOT_X5_DEPLOYED_NOT_ACTIVATED"
PRODUCT_ID = "ICMat-Qwen-Pointer-v6"
RELEASE_SCHEMA = "icmat_llm_offline_release_manifest.v6"
EVIDENCE_INDEX_SCHEMA = "icmat_llm_offline_evidence_index.v6"
BUILD_RESULT_SCHEMA = "icmat_llm_offline_release_build_result.v6"
RELEASE_STATUS = "PASS_PC_OFFLINE_RELEASE_PACKAGE_BUILT"
EVIDENCE_STATUS = "PASS_ALL_UPSTREAM_EVIDENCE_REHASHED_AND_BOUND"

DATASET_AUDIT_SCHEMA = (
    "icmat_evidence_pointer_independent_reproducibility_audit.v6"
)
DATASET_AUDIT_STATUS = "PASS_INDEPENDENT_BYTE_REPRODUCIBILITY_VERIFIED"
TRAINING_SCHEMA = "icmat_qlora_pointer_run_receipt.v6"
TRAINING_STATUS = "PASS_FINAL_THREE_SEED_ALL_EPOCHS_NOT_SELECTED"
CHECKPOINT_EVAL_SCHEMA = "icmat_pointer_checkpoint_evaluation_index.v6"
CHECKPOINT_EVAL_STATUS = "PASS_FINAL_3X6_VALIDATION_EVALUATED_NO_SELECTION"
SELECTION_FREEZE_SCHEMA = selection_freeze_v6.SCHEMA
SELECTION_FREEZE_STATUS = selection_freeze_v6.STATUS
CALIBRATION_SCHEMA = calibration_eval_v6.RECEIPT_SCHEMA
CALIBRATION_STATUS = blind_protocol_v6.CALIBRATION_STATUS
ABLATION_SCHEMA = ablation_eval_v6.RECEIPT_SCHEMA
ABLATION_STATUS = "PASS_NONBLIND_ABLATIONS_COMPLETE_NO_SELECTION"
BLIND_SCHEMA = blind_protocol_v6.RUN_RECEIPT_SCHEMA
BLIND_STATUS = gguf_release_v6.BLIND_PASS_STATUS
BLIND_QUALIFICATION_SCHEMA = blind_protocol_v6.RELEASE_QUALIFICATION_SCHEMA
BLIND_QUALIFICATION_STATUS = (
    blind_protocol_v6.RELEASE_QUALIFICATION_PASS_STATUS
)
GGUF_PARITY_SCHEMA = gguf_release_v6.RELEASE_RECEIPT_SCHEMA
GGUF_PARITY_STATUS = gguf_release_v6.RELEASE_PASS_STATUS
CONTRACTS_SCHEMA = "icmat_qwen_pointer_contract_build_receipt.v6"
CONTRACTS_STATUS = "PASS_V6_CONTRACTS_CREATED_NO_MODEL_EXECUTION"

TASK_CONTRACT_SCHEMA = "icmat_qwen_pointer_task_contract.v6"
PREPROCESSING_CONTRACT_SCHEMA = (
    "icmat_qwen_pointer_preprocessing_contract.v6"
)

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,95}$")
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_ZIP_ROOT = "icmat-qwen-pointer-v6"

FROZEN_SYSTEM_BOUNDARY = {
    "package_type": PACKAGE_TYPE,
    "x5_contacted": False,
    "x5_deployed": False,
    "activated": False,
    "default_enabled": False,
    "autostart_created": False,
    "production_dependency": False,
    "production_code_modified": False,
    "frozen_five_ports_modified": False,
    "frozen_five_ports": [8888, 8080, 8081, 5000, 5001],
    "rb_voe_state": "DEPLOYED_OFF_UNCHANGED",
    "rb_voe_enabled": False,
    "bpu_llm_claimed": False,
}

_ROLE_DESTINATIONS = {
    "dataset_audit": "evidence/01_dataset_audit.v6.json",
    "training_receipt": "evidence/02_training_receipt.v6.json",
    "checkpoint_evaluation": "evidence/03_checkpoint_evaluation.v6.json",
    "selection_freeze": "evidence/04_selection_freeze.v6.json",
    "calibration_receipt": "evidence/05_calibration_receipt.v6.json",
    "ablation_receipt": "evidence/06_ablation_receipt.v6.json",
    "blind_receipt": "evidence/07_blind_receipt.v6.json",
    "blind_qualification_receipt": (
        "evidence/08_blind_release_qualification.v6.json"
    ),
    "gguf_parity_receipt": "evidence/09_gguf_release_receipt.v6.json",
    "contracts_receipt": "evidence/10_contracts_receipt.v6.json",
    "gguf_preflight": "evidence/11_gguf_preflight.v6.json",
    "gguf_golden_set": "evidence/12_validation_golden_set.v6.json",
    "gguf_model": "model/icmat-qwen-pointer-v6-q4_k_m.gguf",
    "task_contract": "contracts/task_contract.v6.json",
    "preprocessing_contract": "contracts/preprocessing_contract.v6.json",
}
_GENERATED_DESTINATIONS = {
    "evidence_index": "manifest/evidence_index.v6.json",
    "release_manifest": "manifest/release_manifest.v6.json",
}
_ALLOWED_PACKAGE_PATHS = frozenset(
    {*_ROLE_DESTINATIONS.values(), *_GENERATED_DESTINATIONS.values()}
)
_RECEIPT_ROLES = (
    "dataset_audit",
    "training_receipt",
    "checkpoint_evaluation",
    "selection_freeze",
    "calibration_receipt",
    "ablation_receipt",
    "blind_receipt",
    "blind_qualification_receipt",
    "gguf_parity_receipt",
    "contracts_receipt",
)
_EXPECTED_RECEIPTS = {
    "dataset_audit": (DATASET_AUDIT_SCHEMA, DATASET_AUDIT_STATUS),
    "training_receipt": (TRAINING_SCHEMA, TRAINING_STATUS),
    "checkpoint_evaluation": (
        CHECKPOINT_EVAL_SCHEMA,
        CHECKPOINT_EVAL_STATUS,
    ),
    "selection_freeze": (
        SELECTION_FREEZE_SCHEMA,
        SELECTION_FREEZE_STATUS,
    ),
    "calibration_receipt": (CALIBRATION_SCHEMA, CALIBRATION_STATUS),
    "ablation_receipt": (ABLATION_SCHEMA, ABLATION_STATUS),
    "blind_receipt": (BLIND_SCHEMA, BLIND_STATUS),
    "blind_qualification_receipt": (
        BLIND_QUALIFICATION_SCHEMA,
        BLIND_QUALIFICATION_STATUS,
    ),
    "gguf_parity_receipt": (
        GGUF_PARITY_SCHEMA,
        GGUF_PARITY_STATUS,
    ),
    "contracts_receipt": (CONTRACTS_SCHEMA, CONTRACTS_STATUS),
}


class ReleaseBundleV6Error(ValueError):
    """Raised when a v6 release input or package violates its contract."""


@dataclass(frozen=True)
class ReleaseBundleInputsV6:
    """Explicit immutable inputs for one final PC-offline release."""

    dataset_dir: Path
    base_model_dir: Path
    selected_checkpoint_dir: Path
    selected_adapter_dir: Path
    dataset_audit: Path
    training_receipt: Path
    checkpoint_evaluation: Path
    selection_freeze: Path
    calibration_receipt: Path
    ablation_receipt: Path
    blind_receipt: Path
    blind_qualification_receipt: Path
    gguf_parity_receipt: Path
    gguf_preflight: Path
    gguf_golden_set: Path
    contracts_receipt: Path
    gguf_model: Path
    task_contract: Path
    preprocessing_contract: Path


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _pretty_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseBundleV6Error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> NoReturn:
    raise ReleaseBundleV6Error(
        f"non-finite JSON constant is forbidden: {value}"
    )


def _assert_finite(value: Any, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ReleaseBundleV6Error(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(item, label=f"{label}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, item in enumerate(value):
            _assert_finite(item, label=f"{label}[{index}]")


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseBundleV6Error(
            f"{label} must be one valid UTF-8 JSON object"
        ) from exc
    if not isinstance(value, dict):
        raise ReleaseBundleV6Error(
            f"{label} must contain one JSON object"
        )
    _assert_finite(value, label=label)
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise ReleaseBundleV6Error(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _contains_scalar(value: Any, expected: Any) -> bool:
    if value == expected:
        return True
    if isinstance(value, Mapping):
        return any(_contains_scalar(item, expected) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return any(_contains_scalar(item, expected) for item in value)
    return False


def _validate_self_digest(receipt: Mapping[str, Any], *, role: str) -> None:
    canonical = receipt.get("canonical_digest_sha256")
    if canonical is not None:
        claimed = _require_sha256(
            canonical,
            label=f"{role}.canonical_digest_sha256",
        )
        body = dict(receipt)
        del body["canonical_digest_sha256"]
        body.pop("receipt_payload_sha256", None)
        if _sha256_bytes(_canonical_bytes(body)) != claimed:
            raise ReleaseBundleV6Error(
                f"{role} canonical digest mismatch"
            )
    payload_digest = receipt.get("receipt_payload_sha256")
    if payload_digest is not None:
        claimed = _require_sha256(
            payload_digest,
            label=f"{role}.receipt_payload_sha256",
        )
        body = dict(receipt)
        del body["receipt_payload_sha256"]
        if _sha256_bytes(_canonical_bytes(body)) != claimed:
            raise ReleaseBundleV6Error(
                f"{role} payload digest mismatch"
            )


def _validate_no_activation_claim(
    value: Any,
    *,
    label: str,
) -> None:
    forbidden_true_keys = {
        "activated",
        "activation_authorized",
        "autostart",
        "autostart_created",
        "bpu_claim_allowed",
        "bpu_llm_claimed",
        "default_enabled",
        "deployed",
        "deployment_authorized",
        "deployment_authorization_allowed",
        "model_reselected",
        "production_dependency",
        "production_integration_allowed",
        "production_integration_authorized",
        "production_state_modified",
        "x5_accessed",
        "x5_contacted",
        "x5_deployed",
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key.casefold() in forbidden_true_keys and item is True:
                raise ReleaseBundleV6Error(
                    f"{label}.{key} must not authorize or claim activation"
                )
            _validate_no_activation_claim(item, label=f"{label}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, item in enumerate(value):
            _validate_no_activation_claim(
                item,
                label=f"{label}[{index}]",
            )


def _validate_package_path(path: str) -> None:
    pure = PurePosixPath(path)
    if (
        path not in _ALLOWED_PACKAGE_PATHS
        or pure.is_absolute()
        or ".." in pure.parts
        or "\\" in path
        or ":" in path
    ):
        raise ReleaseBundleV6Error(
            f"package path is outside the fixed allowlist: {path}"
        )


def _resolve_workspace(root: Path) -> Path:
    lexical = Path(os.path.abspath(os.fspath(root)))
    if not lexical.is_dir() or lexical.is_symlink():
        raise ReleaseBundleV6Error(
            "workspace_root must be a regular non-symlink directory"
        )
    resolved = lexical.resolve(strict=True)
    if resolved != lexical:
        raise ReleaseBundleV6Error(
            "workspace_root must not traverse a symbolic link"
        )
    return resolved


def _reject_symlink_components(root: Path, path: Path) -> None:
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ReleaseBundleV6Error(
                f"symbolic links are forbidden: {current}"
            )


def _resolve_input(root: Path, supplied: Path, *, role: str) -> Path:
    raw = Path(supplied)
    lexical = Path(
        os.path.abspath(os.fspath(raw if raw.is_absolute() else root / raw))
    )
    try:
        lexical.relative_to(root)
    except ValueError as exc:
        raise ReleaseBundleV6Error(
            f"{role} must stay inside workspace_root"
        ) from exc
    _reject_symlink_components(root, lexical)
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise ReleaseBundleV6Error(f"{role} does not exist") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ReleaseBundleV6Error(
            f"{role} resolves outside workspace_root"
        ) from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise ReleaseBundleV6Error(
            f"{role} must be a regular non-symlink file"
        )
    return resolved


def _resolve_directory_input(
    root: Path,
    supplied: Path,
    *,
    role: str,
) -> Path:
    raw = Path(supplied)
    lexical = Path(
        os.path.abspath(os.fspath(raw if raw.is_absolute() else root / raw))
    )
    try:
        lexical.relative_to(root)
    except ValueError as exc:
        raise ReleaseBundleV6Error(
            f"{role} must stay inside workspace_root"
        ) from exc
    _reject_symlink_components(root, lexical)
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise ReleaseBundleV6Error(f"{role} does not exist") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ReleaseBundleV6Error(
            f"{role} resolves outside workspace_root"
        ) from exc
    if not resolved.is_dir() or resolved.is_symlink():
        raise ReleaseBundleV6Error(
            f"{role} must be a regular non-symlink directory"
        )
    return resolved


def _canonical_dataset_inventory(dataset_dir: Path) -> dict[str, Any]:
    root = Path(dataset_dir).resolve(strict=True)
    manifest_path = root / "manifest.v6.json"
    manifest = _load_json(manifest_path, label="frozen dataset manifest")
    if (
        manifest.get("schema")
        != blind_protocol_v6.DATASET_MANIFEST_SCHEMA
        or manifest.get("status") != blind_protocol_v6.DATASET_STATUS
    ):
        raise ReleaseBundleV6Error(
            "dataset manifest is not the frozen sealed v6 contract"
        )
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {
        "train",
        "validation",
        "calibration",
        "blind_test",
    }:
        raise ReleaseBundleV6Error(
            "dataset manifest must contain the complete four-split inventory"
        )

    split_paths: dict[str, str] = {}
    seen_split_paths: set[str] = set()
    for split, value in splits.items():
        if not isinstance(value, Mapping):
            raise ReleaseBundleV6Error(
                f"dataset split descriptor is invalid: {split}"
            )
        relative = value.get("path")
        if not isinstance(relative, str):
            raise ReleaseBundleV6Error(
                f"dataset split path is invalid: {split}"
            )
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or not pure.parts
            or ".." in pure.parts
            or "\\" in relative
        ):
            raise ReleaseBundleV6Error(
                f"dataset split path is unsafe: {split}"
            )
        folded = relative.casefold()
        if folded in seen_split_paths:
            raise ReleaseBundleV6Error(
                "dataset split paths are Windows-case ambiguous"
            )
        seen_split_paths.add(folded)
        split_paths[str(split)] = relative
        _require_sha256(
            value.get("sha256"),
            label=f"dataset split {split} SHA-256",
        )
        if (
            not isinstance(value.get("bytes"), int)
            or int(value["bytes"]) < 1
            or not isinstance(value.get("count"), int)
            or int(value["count"]) < 1
        ):
            raise ReleaseBundleV6Error(
                f"dataset split size/count is invalid: {split}"
            )

    blind_relative = split_paths["blind_test"]
    candidates = list(root.rglob("*"))
    candidates.sort(
        key=lambda path: (
            path.relative_to(root).as_posix().casefold(),
            path.relative_to(root).as_posix(),
        )
    )
    files: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for candidate in candidates:
        if candidate.is_symlink():
            raise ReleaseBundleV6Error(
                f"dataset contains a forbidden symlink: {candidate}"
            )
        if candidate.is_dir():
            continue
        mode = candidate.stat(follow_symlinks=False).st_mode
        if not stat.S_ISREG(mode):
            raise ReleaseBundleV6Error(
                f"dataset contains a non-regular entry: {candidate}"
            )
        relative = candidate.relative_to(root).as_posix()
        folded = relative.casefold()
        if folded in seen_paths:
            raise ReleaseBundleV6Error(
                "dataset contains Windows-case ambiguous paths"
            )
        seen_paths.add(folded)
        before = candidate.stat()
        if relative == blind_relative:
            descriptor = splits["blind_test"]
            digest = str(descriptor["sha256"])
            if before.st_size != int(descriptor["bytes"]):
                raise ReleaseBundleV6Error(
                    "sealed blind split size differs from its manifest"
                )
            hash_source = "SEALED_MANIFEST_AND_CONSUMED_BLIND_RUN"
            content_reopened = False
        else:
            digest = _sha256_file(candidate)
            after = candidate.stat()
            if (
                before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
            ):
                raise ReleaseBundleV6Error(
                    f"dataset changed while hashing: {candidate}"
                )
            hash_source = "REHASHED_BY_RELEASE_BUNDLE"
            content_reopened = True
        files.append(
            {
                "path": relative,
                "bytes": before.st_size,
                "sha256": digest,
                "hash_source": hash_source,
                "content_reopened": content_reopened,
            }
        )

    for split, relative in split_paths.items():
        matches = [row for row in files if row["path"] == relative]
        if len(matches) != 1:
            raise ReleaseBundleV6Error(
                f"dataset split is absent from canonical inventory: {split}"
            )
        row = matches[0]
        descriptor = splits[split]
        if (
            row["bytes"] != descriptor["bytes"]
            or row["sha256"] != descriptor["sha256"]
        ):
            raise ReleaseBundleV6Error(
                f"dataset split differs from manifest: {split}"
            )

    return {
        "ordering": "windows_casefold_then_posix",
        "files": files,
        "file_count": len(files),
        "bytes": sum(int(row["bytes"]) for row in files),
        "tree_sha256": _sha256_bytes(_canonical_bytes(files)),
        "manifest_sha256": _sha256_file(manifest_path),
        "validation_sha256": str(splits["validation"]["sha256"]),
        "blind_sha256": str(splits["blind_test"]["sha256"]),
        "blind_path": blind_relative,
        "blind_content_reopened": False,
    }


def _canonical_model_inventories(
    paths: Mapping[str, Path],
) -> dict[str, dict[str, Any]]:
    try:
        inventories = {
            "base_model": gguf_release_v6.tree_inventory(
                paths["base_model_dir"],
                label="base model",
            ),
            "selected_checkpoint": gguf_release_v6.tree_inventory(
                paths["selected_checkpoint_dir"],
                label="selected checkpoint",
            ),
            "selected_adapter": gguf_release_v6.adapter_inventory(
                paths["selected_adapter_dir"],
                label="selected adapter",
            ),
        }
    except (gguf_release_v6.GgufReleaseV6Error, OSError) as exc:
        raise ReleaseBundleV6Error(
            "model/checkpoint/adapter canonical inventory failed"
        ) from exc
    for role, inventory in inventories.items():
        if inventory.get("ordering") != "windows_casefold_then_posix":
            raise ReleaseBundleV6Error(
                f"{role} inventory is not casefold canonical"
            )
    return inventories


def _prepare_output_root(root: Path, supplied: Path) -> Path:
    raw = Path(supplied)
    lexical = Path(
        os.path.abspath(os.fspath(raw if raw.is_absolute() else root / raw))
    )
    try:
        lexical.relative_to(root)
    except ValueError as exc:
        raise ReleaseBundleV6Error(
            "output_root must stay inside workspace_root"
        ) from exc
    lexical.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(root, lexical.parent)
    if lexical.exists():
        if not lexical.is_dir() or lexical.is_symlink():
            raise ReleaseBundleV6Error(
                "output_root must be a regular directory"
            )
    else:
        lexical.mkdir()
    _reject_symlink_components(root, lexical)
    return lexical.resolve(strict=True)


def _resolve_inputs(
    root: Path,
    inputs: ReleaseBundleInputsV6,
) -> dict[str, Path]:
    directory_fields = {
        "dataset_dir",
        "base_model_dir",
        "selected_checkpoint_dir",
        "selected_adapter_dir",
    }
    resolved: dict[str, Path] = {}
    identities: set[tuple[int, int]] = set()
    for field in fields(inputs):
        if field.name in directory_fields:
            path = _resolve_directory_input(
                root,
                Path(getattr(inputs, field.name)),
                role=field.name,
            )
        else:
            path = _resolve_input(
                root,
                Path(getattr(inputs, field.name)),
                role=field.name,
            )
            info = path.stat()
            identity = (int(info.st_dev), int(info.st_ino))
            if identity in identities:
                raise ReleaseBundleV6Error(
                    f"duplicate physical input file: {field.name}"
                )
            identities.add(identity)
        resolved[field.name] = path
    if (
        resolved["dataset_dir"] == resolved["base_model_dir"]
        or resolved["dataset_dir"] == resolved["selected_checkpoint_dir"]
        or resolved["base_model_dir"] == resolved["selected_checkpoint_dir"]
    ):
        raise ReleaseBundleV6Error(
            "dataset, base model, and selected checkpoint roots must differ"
        )
    if (
        resolved["blind_receipt"].name != "run_receipt.v6.json"
        or resolved["blind_qualification_receipt"].name
        != "release_qualification.v6.json"
        or resolved["blind_receipt"].parent
        != resolved["blind_qualification_receipt"].parent
    ):
        raise ReleaseBundleV6Error(
            "blind run and qualification must be canonical sibling artifacts"
        )
    if (
        resolved["calibration_receipt"].name != "receipt.v6.json"
        or resolved["ablation_receipt"].name != "run_receipt.v6.json"
    ):
        raise ReleaseBundleV6Error(
            "calibration and ablation receipts must use canonical filenames"
        )
    if (
        resolved["gguf_parity_receipt"].name
        != gguf_release_v6.DEFAULT_RECEIPT_NAME
        or resolved["gguf_preflight"].name
        != gguf_release_v6.DEFAULT_PREFLIGHT_NAME
        or resolved["gguf_golden_set"].name
        != gguf_release_v6.DEFAULT_GOLDEN_NAME
        or resolved["gguf_preflight"].parent
        != resolved["gguf_parity_receipt"].parent
        or resolved["gguf_golden_set"].parent
        != resolved["gguf_parity_receipt"].parent
        or resolved["gguf_model"].parent
        != resolved["gguf_parity_receipt"].parent
        or resolved["gguf_model"].name != gguf_release_v6.DEFAULT_Q4_NAME
    ):
        raise ReleaseBundleV6Error(
            "GGUF receipt, preflight, validation golden set, and Q4 model must be canonical siblings"
        )
    with resolved["gguf_model"].open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise ReleaseBundleV6Error(
                "gguf_model does not have the GGUF magic header"
            )
    return resolved


def _receipt_snapshot(path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": record.get("schema"),
        "status": record.get("status"),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _validate_receipt_contracts(
    *,
    paths: Mapping[str, Path],
    records: Mapping[str, dict[str, Any]],
    dataset_inventory: Mapping[str, Any],
    model_inventories: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    snapshots: dict[str, dict[str, Any]] = {}
    for role in _RECEIPT_ROLES:
        expected_schema, expected_status = _EXPECTED_RECEIPTS[role]
        record = records[role]
        if record.get("schema") != expected_schema:
            raise ReleaseBundleV6Error(
                f"{role} schema must be {expected_schema}"
            )
        if record.get("status") != expected_status:
            raise ReleaseBundleV6Error(
                f"{role} status must be {expected_status}"
            )
        _validate_self_digest(record, role=role)
        _validate_no_activation_claim(record, label=role)
        snapshots[role] = _receipt_snapshot(paths[role], record)

    dataset = records["dataset_audit"]
    if (
        dataset.get("reproducibility_passed") is not True
        or _nested(
            dataset,
            "artifact_reproducibility",
            "exact_inventory_verified",
        )
        is not True
        or _nested(
            dataset,
            "artifact_reproducibility",
            "all_files_byte_identical",
        )
        is not True
        or _nested(dataset, "blind_test", "hash_only_access") is not True
        or _nested(dataset, "blind_test", "json_parsed") is not False
        or _nested(dataset, "blind_test", "content_disclosed") is not False
    ):
        raise ReleaseBundleV6Error(
            "dataset_audit does not prove independent sealed reproducibility"
        )
    manifest_sha = _require_sha256(
        _nested(
            dataset,
            "independent_builds",
            "manifest_sha256",
        ),
        label="dataset_audit manifest SHA-256",
    )
    if manifest_sha != dataset_inventory.get("manifest_sha256"):
        raise ReleaseBundleV6Error(
            "dataset_audit manifest differs from the canonical dataset inventory"
        )

    training = records["training_receipt"]
    if (
        training.get("stage") != "final"
        or training.get("checkpoint_count") != 18
        or _nested(training, "authorization", "checkpoint_selected")
        is not False
        or _nested(training, "authorization", "calibration_authorized")
        is not False
        or _nested(training, "authorization", "blind_test_authorized")
        is not False
    ):
        raise ReleaseBundleV6Error(
            "training_receipt is not the unselected final 3x6 run"
        )

    checkpoint = records["checkpoint_evaluation"]
    if (
        checkpoint.get("stage") != "final"
        or _nested(checkpoint, "training", "checkpoint_count") != 18
        or len(checkpoint.get("checkpoints", [])) != 18
        or _nested(checkpoint, "selection", "performed") is not False
        or _nested(checkpoint, "execution", "checkpoint_selected")
        is not False
        or _nested(
            checkpoint,
            "dataset",
            "calibration_content_read",
        )
        is not False
        or _nested(
            checkpoint,
            "dataset",
            "blind_test_content_read",
        )
        is not False
    ):
        raise ReleaseBundleV6Error(
            "checkpoint_evaluation is not a complete unselected final 3x6 audit"
        )
    if not _contains_scalar(
        checkpoint,
        snapshots["training_receipt"]["sha256"],
    ):
        raise ReleaseBundleV6Error(
            "checkpoint_evaluation does not bind training_receipt"
        )

    selection = records["selection_freeze"]
    selection_authorization = selection.get("authorization")
    if (
        selection.get("selection_locked") is not True
        or not isinstance(selection_authorization, Mapping)
        or selection_authorization.get("checkpoint_selected") is not True
        or selection_authorization.get("calibration_authorized") is not True
        or selection_authorization.get("blind_test_authorized") is not False
        or selection_authorization.get("gguf_export_authorized") is not False
        or selection_authorization.get("deployment_authorized") is not False
        or selection_authorization.get("production_integration_authorized")
        is not False
    ):
        raise ReleaseBundleV6Error(
            "selection_freeze does not preserve the canonical post-selection boundary"
        )
    for role in ("training_receipt", "checkpoint_evaluation"):
        if not _contains_scalar(selection, snapshots[role]["sha256"]):
            raise ReleaseBundleV6Error(
                f"selection_freeze does not bind {role}"
            )
    if not _contains_scalar(selection, manifest_sha):
        raise ReleaseBundleV6Error(
            "selection_freeze does not bind the audited dataset manifest"
        )
    selected_checkpoint = _nested(
        selection,
        "selection",
        "checkpoint",
    )
    selected_adapter = _nested(
        selection,
        "selection",
        "adapter",
    )
    selected_base = selection.get("base_model")
    if (
        not isinstance(selected_checkpoint, Mapping)
        or not isinstance(selected_adapter, Mapping)
        or not isinstance(selected_base, Mapping)
    ):
        raise ReleaseBundleV6Error(
            "selection_freeze lacks full base/checkpoint/adapter inventories"
        )
    checkpoint_tree_sha = _require_sha256(
        selected_checkpoint.get("tree_sha256"),
        label="selection_freeze selected checkpoint tree SHA-256",
    )
    adapter_tree_sha = _require_sha256(
        selected_adapter.get("tree_sha256"),
        label="selection_freeze selected adapter tree SHA-256",
    )
    base_tree_sha = _require_sha256(
        selected_base.get("training_tree_sha256"),
        label="selection_freeze base model tree SHA-256",
    )
    if (
        checkpoint_tree_sha
        != model_inventories["selected_checkpoint"].get("tree_sha256")
        or adapter_tree_sha
        != model_inventories["selected_adapter"].get("tree_sha256")
        or base_tree_sha
        != model_inventories["base_model"].get("tree_sha256")
    ):
        raise ReleaseBundleV6Error(
            "selection_freeze differs from canonical model inventories"
        )

    calibration = records["calibration_receipt"]
    if (
        _nested(calibration, "dataset", "opened_split") != "calibration"
        or _nested(calibration, "dataset", "complete_split") is not True
        or _nested(calibration, "dataset", "rows") != 150
        or _nested(calibration, "dataset", "blind_data_accessed") is not False
        or _nested(calibration, "execution", "backend") != "hf_model"
        or _nested(calibration, "execution", "model_bound") is not True
        or _nested(
            calibration,
            "model",
            "fixture_not_model_evidence",
        )
        is not False
        or _nested(
            calibration,
            "execution",
            "checkpoint_reselection_performed",
        )
        is not False
        or _nested(
            calibration,
            "authorization",
            "blind_test_authorized",
        )
        is not False
        or _nested(
            calibration,
            "authorization",
            "deployment_authorized",
        )
        is not False
    ):
        raise ReleaseBundleV6Error(
            "calibration_receipt is not a complete frozen model-bound PASS"
        )
    if not _contains_scalar(
        calibration,
        snapshots["selection_freeze"]["sha256"],
    ):
        raise ReleaseBundleV6Error(
            "calibration_receipt does not bind selection_freeze"
        )
    if (
        _nested(
            calibration,
            "selection_freeze",
            "checkpoint_tree_sha256",
        )
        != checkpoint_tree_sha
        or _nested(
            calibration,
            "selection_freeze",
            "adapter_tree_sha256",
        )
        != adapter_tree_sha
        or _nested(
            calibration,
            "selection_freeze",
            "base_model_tree_sha256",
        )
        != base_tree_sha
    ):
        raise ReleaseBundleV6Error(
            "calibration_receipt does not bind the canonical selected model"
        )

    ablation = records["ablation_receipt"]
    if (
        _nested(ablation, "dataset", "split") != "validation"
        or _nested(ablation, "dataset", "validation_complete_only")
        is not True
        or _nested(ablation, "dataset", "calibration_opened") is not False
        or _nested(ablation, "dataset", "sealed_blind_opened") is not False
        or _nested(ablation, "execution", "backend_mode") != "hf_model"
        or _nested(
            ablation,
            "execution",
            "model_quality_claim_allowed",
        )
        is not True
        or _nested(
            ablation,
            "execution",
            "automatic_model_selection",
        )
        is not False
        or _nested(
            ablation,
            "execution",
            "promotion_authorized",
        )
        is not False
        or _nested(
            ablation,
            "execution",
            "production_state_modified",
        )
        is not False
    ):
        raise ReleaseBundleV6Error(
            "ablation_receipt is not a closed pre-blind PASS"
        )
    if not _contains_scalar(ablation, adapter_tree_sha):
        raise ReleaseBundleV6Error(
            "ablation_receipt does not bind the selected adapter"
        )

    contracts = records["contracts_receipt"]
    if (
        _nested(
            contracts,
            "execution_boundary",
            "model_generation_executed",
        )
        is not False
        or _nested(
            contracts,
            "execution_boundary",
            "x5_accessed",
        )
        is not False
    ):
        raise ReleaseBundleV6Error(
            "contracts_receipt exceeds the offline contract boundary"
        )
    if not _contains_scalar(contracts, manifest_sha):
        raise ReleaseBundleV6Error(
            "contracts_receipt does not bind the audited dataset manifest"
        )
    for role, artifact_name in (
        ("task_contract", "task_contract"),
        ("preprocessing_contract", "preprocessing_contract"),
    ):
        artifact = _nested(contracts, "artifacts", artifact_name)
        if (
            not isinstance(artifact, Mapping)
            or artifact.get("sha256") != _sha256_file(paths[role])
            or artifact.get("bytes") != paths[role].stat().st_size
        ):
            raise ReleaseBundleV6Error(
                f"contracts_receipt does not bind {role}"
            )

    blind = records["blind_receipt"]
    if (
        blind.get("examples") != blind_protocol_v6.EXPECTED_BLIND_EXAMPLES
        or _nested(blind, "backend", "mode") != "hf_model"
        or _nested(blind, "dataset", "rows_read_once")
        != blind_protocol_v6.EXPECTED_BLIND_EXAMPLES
        or _nested(blind, "dataset", "manifest_sha256") != manifest_sha
        or _nested(blind, "dataset", "blind_sha256")
        != dataset_inventory.get("blind_sha256")
        or _nested(blind, "model", "base_model_tree_sha256")
        != base_tree_sha
        or _nested(blind, "model", "checkpoint_tree_sha256")
        != checkpoint_tree_sha
        or _nested(blind, "model", "adapter_tree_sha256")
        != adapter_tree_sha
    ):
        raise ReleaseBundleV6Error(
            "blind_receipt is not the complete model-bound canonical run"
        )
    for role in (
        "selection_freeze",
        "calibration_receipt",
        "ablation_receipt",
        "contracts_receipt",
    ):
        if not _contains_scalar(
            blind.get("gates"),
            snapshots[role]["sha256"],
        ):
            raise ReleaseBundleV6Error(
                f"blind_receipt does not bind {role}"
            )
    if not _contains_scalar(blind, adapter_tree_sha):
        raise ReleaseBundleV6Error(
            "blind_receipt does not bind the selected adapter"
        )

    qualification = records["blind_qualification_receipt"]
    if (
        qualification.get("qualified") is not True
        or qualification.get("thresholds")
        != blind_protocol_v6.RELEASE_QUALIFICATION_POLICY
        or _nested(
            qualification,
            "blind_run_receipt",
            "sha256",
        )
        != snapshots["blind_receipt"]["sha256"]
        or _nested(
            qualification,
            "consumption_claim",
            "failure_is_non_reusable",
        )
        is not True
        or _nested(
            qualification,
            "release_authorization",
            "gguf_release_authorized",
        )
        is not True
        or _nested(
            qualification,
            "release_authorization",
            "activation_authorized",
        )
        is not False
        or _nested(
            qualification,
            "release_authorization",
            "deployment_authorized",
        )
        is not False
        or _nested(
            qualification,
            "release_authorization",
            "production_integration_authorized",
        )
        is not False
    ):
        raise ReleaseBundleV6Error(
            "blind qualification is not canonical GGUF-only authorization"
        )
    expected_qualification_upstream = {
        "selection_freeze_sha256": snapshots["selection_freeze"]["sha256"],
        "calibration_receipt_sha256": snapshots["calibration_receipt"][
            "sha256"
        ],
        "ablation_receipt_sha256": snapshots["ablation_receipt"]["sha256"],
        "dataset_manifest_sha256": manifest_sha,
        "blind_sha256": dataset_inventory["blind_sha256"],
        "base_model_tree_sha256": base_tree_sha,
        "checkpoint_tree_sha256": checkpoint_tree_sha,
        "adapter_tree_sha256": adapter_tree_sha,
    }
    if qualification.get("upstream") != expected_qualification_upstream:
        raise ReleaseBundleV6Error(
            "blind qualification upstream inventory binding is invalid"
        )

    parity = records["gguf_parity_receipt"]
    gguf_sha = _sha256_file(paths["gguf_model"])
    if (
        parity.get("activated") is not False
        or parity.get("deployable_by_this_receipt") is not False
        or parity.get("release_quality_evidence") is not True
        or parity.get("fixture_execution") is not False
        or parity.get("service_registered") is not False
        or parity.get("training_invoked") is not False
        or parity.get("selection_invoked") is not False
        or _nested(parity, "parity", "status")
        != "PASS_STRICT_HF_GGUF_POINTER_AND_COMPILER_PARITY"
        or _nested(parity, "parity", "strict_gate_pass") is not True
        or _nested(
            parity,
            "claim_boundary",
            "rdk_x5_measured",
        )
        is not False
        or _nested(parity, "claim_boundary", "bpu_used") is not False
        or not _contains_scalar(parity, adapter_tree_sha)
        or not _contains_scalar(
            parity,
            snapshots["blind_receipt"]["sha256"],
        )
        or not _contains_scalar(
            parity,
            snapshots["blind_qualification_receipt"]["sha256"],
        )
    ):
        raise ReleaseBundleV6Error(
            "GGUF receipt is not a real qualification-bound release"
        )
    parity_chain = parity.get("receipt_chain")
    if (
        not isinstance(parity_chain, Mapping)
        or parity_chain.get("fixture_chain_only") is not False
        or parity_chain.get("selected_adapter_tree_sha256")
        != adapter_tree_sha
        or parity_chain.get("selected_checkpoint_tree_sha256")
        != checkpoint_tree_sha
        or _nested(parity_chain, "blind_receipt", "sha256")
        != snapshots["blind_receipt"]["sha256"]
        or _nested(
            parity_chain,
            "blind_release_qualification",
            "sha256",
        )
        != snapshots["blind_qualification_receipt"]["sha256"]
    ):
        raise ReleaseBundleV6Error(
            "GGUF receipt chain differs from canonical blind qualification"
        )
    q4_artifact = _nested(parity, "artifacts", "gguf_q4_k_m")
    if (
        not isinstance(q4_artifact, Mapping)
        or q4_artifact.get("sha256") != gguf_sha
        or q4_artifact.get("bytes") != paths["gguf_model"].stat().st_size
    ):
        raise ReleaseBundleV6Error(
            "gguf_parity_receipt does not bind the supplied Q4_K_M GGUF"
        )

    preflight = records["gguf_preflight"]
    golden = records["gguf_golden_set"]
    if (
        preflight.get("schema") != gguf_release_v6.PREFLIGHT_SCHEMA
        or preflight.get("status") != gguf_release_v6.PREFLIGHT_PASS_STATUS
        or preflight.get("execution_ready") is not True
        or preflight.get("receipt_chain") != parity_chain
        or _nested(
            preflight,
            "release_policy",
            "fixture_execution",
        )
        is not False
        or _nested(
            preflight,
            "release_policy",
            "release_quality_evidence",
        )
        is not True
        or _nested(preflight, "selected_checkpoint", "tree_sha256")
        != checkpoint_tree_sha
        or _nested(preflight, "selected_adapter", "tree_sha256")
        != adapter_tree_sha
        or _nested(preflight, "base_model", "tree_sha256")
        != base_tree_sha
    ):
        raise ReleaseBundleV6Error(
            "GGUF preflight is not the real canonical release preflight"
        )
    golden_record = preflight.get("golden_set")
    golden_artifact = _nested(parity, "artifacts", "golden_set")
    golden_sha = _sha256_file(paths["gguf_golden_set"])
    if (
        not isinstance(golden_record, Mapping)
        or not isinstance(golden_artifact, Mapping)
        or golden_record.get("split") != "validation"
        or golden_record.get("blind_data_accessed") is not False
        or golden_record.get("selection_freeze_sha256")
        != snapshots["selection_freeze"]["sha256"]
        or golden_record.get("dataset_manifest_sha256") != manifest_sha
        or golden_record.get("validation_sha256")
        != dataset_inventory.get("validation_sha256")
        or golden_record.get("sha256") != golden_sha
        or golden_artifact.get("sha256") != golden_sha
        or golden_artifact.get("bytes")
        != paths["gguf_golden_set"].stat().st_size
        or golden.get("schema") != gguf_release_v6.GOLDEN_SET_SCHEMA
        or golden.get("split") != "validation"
        or _nested(golden, "sampling", "blind_data_accessed") is not False
        or _nested(golden, "sampling", "used_for_model_selection") is not False
        or golden.get("selection_freeze_sha256")
        != snapshots["selection_freeze"]["sha256"]
        or golden.get("dataset_manifest_sha256") != manifest_sha
        or golden.get("validation_sha256")
        != dataset_inventory.get("validation_sha256")
    ):
        raise ReleaseBundleV6Error(
            "GGUF golden set is not frozen validation-only evidence"
        )
    if parity.get("preflight_sha256") != _sha256_file(
        paths["gguf_preflight"]
    ):
        raise ReleaseBundleV6Error(
            "GGUF release receipt does not bind its real preflight"
        )

    task = records["task_contract"]
    preprocessing = records["preprocessing_contract"]
    if (
        task.get("schema") != TASK_CONTRACT_SCHEMA
        or preprocessing.get("schema") != PREPROCESSING_CONTRACT_SCHEMA
        or task.get("status") != "FROZEN_BEFORE_CALIBRATION_AND_BLIND"
        or preprocessing.get("status")
        != "FROZEN_BEFORE_CALIBRATION_AND_BLIND"
    ):
        raise ReleaseBundleV6Error(
            "runtime contract files are not the frozen v6 pointer contracts"
        )
    _validate_no_activation_claim(task, label="task_contract")
    _validate_no_activation_claim(
        preprocessing,
        label="preprocessing_contract",
    )

    return {
        "manifest_sha256": manifest_sha,
        "dataset_tree_sha256": dataset_inventory["tree_sha256"],
        "base_model_tree_sha256": base_tree_sha,
        "selected_checkpoint_tree_sha256": checkpoint_tree_sha,
        "selected_adapter_tree_sha256": adapter_tree_sha,
        "blind_sha256": dataset_inventory["blind_sha256"],
        "blind_qualification_sha256": snapshots[
            "blind_qualification_receipt"
        ]["sha256"],
        "validation_golden_sha256": golden_sha,
        "gguf_sha256": gguf_sha,
        "receipts": snapshots,
    }


def _authoritative_source_verification(
    *,
    workspace_root: Path,
    paths: Mapping[str, Path],
    records: Mapping[str, dict[str, Any]],
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    qualification = records["blind_qualification_receipt"]
    blind = records["blind_receipt"]

    linked_files: list[tuple[str, Path]] = []
    authorization_record = qualification.get("authorization")
    claim_record = qualification.get("consumption_claim")
    artifacts = qualification.get("artifacts")
    if (
        not isinstance(authorization_record, Mapping)
        or not isinstance(claim_record, Mapping)
        or not isinstance(artifacts, Mapping)
    ):
        raise ReleaseBundleV6Error(
            "blind qualification linked evidence is incomplete"
        )
    linked_files.extend(
        (
            (
                "blind_authorization",
                Path(str(authorization_record.get("path"))),
            ),
            (
                "blind_consumption_claim",
                Path(str(claim_record.get("path"))),
            ),
        )
    )
    for name in ("sample_results.v6.jsonl", "summary.v6.json"):
        record = artifacts.get(name)
        if not isinstance(record, Mapping):
            raise ReleaseBundleV6Error(
                f"blind qualification artifact is absent: {name}"
            )
        linked_files.append(
            (f"blind_qualification_{name}", Path(str(record.get("path"))))
        )
    resolved_linked = {
        role: _resolve_input(workspace_root, path, role=role)
        for role, path in linked_files
    }

    try:
        qualification_result = (
            blind_protocol_v6.verify_release_qualification_v6(
                blind_receipt_path=paths["blind_receipt"],
                blind_receipt_sha256=bindings["receipts"][
                    "blind_receipt"
                ]["sha256"],
                qualification_receipt_path=paths[
                    "blind_qualification_receipt"
                ],
                qualification_receipt_sha256=bindings["receipts"][
                    "blind_qualification_receipt"
                ]["sha256"],
            )
        )
        calibration_result = (
            blind_protocol_v6._verify_calibration_receipt_recomputed_v6(
                receipt_path=paths["calibration_receipt"],
                receipt_sha256=bindings["receipts"][
                    "calibration_receipt"
                ]["sha256"],
                dataset_dir=paths["dataset_dir"],
            )
        )
        ablation_result = ablation_eval_v6.verify_ablation_receipt_v6(
            receipt_path=paths["ablation_receipt"],
            expected_sha256=bindings["receipts"]["ablation_receipt"][
                "sha256"
            ],
            selection_freeze_path=paths["selection_freeze"],
            selection_freeze_sha256=bindings["receipts"][
                "selection_freeze"
            ]["sha256"],
            dataset_dir=paths["dataset_dir"],
            base_model_dir=paths["base_model_dir"],
            checkpoint_dir=paths["selected_checkpoint_dir"],
            adapter_dir=paths["selected_adapter_dir"],
        )
    except (
        blind_protocol_v6.BlindProtocolV6Error,
        ablation_eval_v6.AblationEvalV6Error,
        OSError,
        ValueError,
    ) as exc:
        raise ReleaseBundleV6Error(
            "canonical blind/calibration/ablation verification failed"
        ) from exc

    parity_chain = records["gguf_parity_receipt"].get("receipt_chain")
    if (
        not isinstance(parity_chain, Mapping)
        or parity_chain.get("authoritative_blind_release_qualification")
        != qualification_result
    ):
        raise ReleaseBundleV6Error(
            "GGUF receipt does not embed the independently recomputed qualification"
        )

    authorization_path = resolved_linked["blind_authorization"]
    authorization = _load_json(
        authorization_path,
        label="blind authorization",
    )
    authorization_sha = _sha256_file(authorization_path)
    if (
        authorization_sha != authorization_record.get("sha256")
        or authorization.get("schema")
        != blind_protocol_v6.AUTHORIZATION_SCHEMA
        or authorization.get("status")
        != blind_protocol_v6.AUTHORIZATION_STATUS
        or authorization.get("release_qualification_policy")
        != blind_protocol_v6.RELEASE_QUALIFICATION_POLICY
    ):
        raise ReleaseBundleV6Error(
            "blind authorization or frozen release policy is invalid"
        )
    gates = authorization.get("gates")
    calibration_gate = (
        gates.get("calibration") if isinstance(gates, Mapping) else None
    )
    ablation_gate = (
        gates.get("ablation") if isinstance(gates, Mapping) else None
    )
    if (
        not isinstance(calibration_gate, Mapping)
        or calibration_gate.get("sha256")
        != bindings["receipts"]["calibration_receipt"]["sha256"]
        or calibration_gate.get("authoritative_verification")
        != calibration_result
        or not isinstance(ablation_gate, Mapping)
        or ablation_gate.get("sha256")
        != bindings["receipts"]["ablation_receipt"]["sha256"]
        or _nested(
            ablation_gate,
            "authoritative_verification",
            "function",
        )
        != "verify_ablation_receipt_v6"
        or _nested(
            ablation_gate,
            "authoritative_verification",
            "result",
        )
        != ablation_result
    ):
        raise ReleaseBundleV6Error(
            "authorization does not bind current calibration/ablation verifiers"
        )

    consumption = authorization.get("consumption")
    if not isinstance(consumption, Mapping):
        raise ReleaseBundleV6Error(
            "blind authorization consumption contract is absent"
        )
    registry_root = _resolve_directory_input(
        workspace_root,
        Path(str(consumption.get("registry_root"))),
        role="global_blind_registry_root",
    )
    try:
        registry_result = blind_protocol_v6._verify_registry(
            dataset_root=paths["dataset_dir"],
            registry_root=registry_root,
            blind_sha256=str(bindings["blind_sha256"]),
            authorization_path=authorization_path,
            authorization_sha256=authorization_sha,
            authorization=authorization,
        )
    except (
        blind_protocol_v6.BlindProtocolV6Error,
        OSError,
        ValueError,
    ) as exc:
        raise ReleaseBundleV6Error(
            "global blind authorization registry verification failed"
        ) from exc
    if _nested(blind, "authorization", "registry") != registry_result:
        raise ReleaseBundleV6Error(
            "blind run does not bind the global authorization reservation"
        )

    claim_path = resolved_linked["blind_consumption_claim"]
    claim = _load_json(claim_path, label="blind consumption claim")
    claim_sha = _sha256_file(claim_path)
    nonce = consumption.get("nonce")
    nonce_sha = consumption.get("nonce_sha256")
    expected_claim_path = Path(str(consumption.get("claim_path"))).resolve(
        strict=True
    )
    if (
        claim_path != expected_claim_path
        or claim_sha != claim_record.get("sha256")
        or claim.get("schema") != blind_protocol_v6.CLAIM_SCHEMA
        or claim.get("status") != "CONSUMED_PENDING_NON_REUSABLE"
        or claim.get("authorization_id") != authorization.get("authorization_id")
        or claim.get("authorization_sha256") != authorization_sha
        or claim.get("blind_sha256") != bindings["blind_sha256"]
        or claim.get("nonce") != nonce
        or claim.get("nonce_sha256") != nonce_sha
        or claim_record.get("nonce_sha256") != nonce_sha
        or claim.get("failure_is_non_reusable") is not True
        or claim.get("overwrite_allowed") is not False
        or not isinstance(nonce, str)
        or _sha256_bytes(nonce.encode("ascii")) != nonce_sha
    ):
        raise ReleaseBundleV6Error(
            "global one-shot blind claim or nonce binding is invalid"
        )

    terminal_path = _resolve_input(
        workspace_root,
        Path(str(consumption.get("terminal_path"))),
        role="blind_consumption_terminal",
    )
    terminal = _load_json(
        terminal_path,
        label="blind consumption terminal",
    )
    if (
        terminal.get("schema") != blind_protocol_v6.TERMINAL_SCHEMA
        or terminal.get("status") != "COMPLETED"
        or terminal.get("authorization_sha256") != authorization_sha
        or terminal.get("claim_sha256") != claim_sha
        or terminal.get("nonce_sha256") != nonce_sha
        or terminal.get("blind_sha256") != bindings["blind_sha256"]
        or terminal.get("failure_is_non_reusable") is not True
        or terminal.get("overwrite_allowed") is not False
    ):
        raise ReleaseBundleV6Error(
            "blind global claim terminal is not a completed immutable record"
        )

    def selected(
        result: Mapping[str, Any],
        names: Sequence[str],
    ) -> dict[str, Any]:
        return {name: result.get(name) for name in names}

    return {
        "blind_release_qualification": selected(
            qualification_result,
            (
                "status",
                "blind_receipt_sha256",
                "qualification_receipt_sha256",
                "authorization_sha256",
                "claim_sha256",
                "sample_results_sha256",
                "summary_sha256",
                "samples_recomputed",
                "gates_recomputed",
                "qualified",
                "fixture_accepted",
                "blind_dataset_reopened",
            ),
        ),
        "calibration": selected(
            calibration_result,
            (
                "status",
                "receipt_sha256",
                "dataset_calibration_sha256",
                "samples_recompiled",
                "per_sample_sha256",
                "summary_sha256",
                "model_bound",
                "complete_split",
                "blind_data_accessed",
            ),
        ),
        "ablation": selected(
            ablation_result,
            (
                "status",
                "receipt_sha256",
                "canonical_digest_sha256",
                "selection_freeze_sha256",
                "dataset_validation_sha256",
                "base_model_tree_sha256",
                "selected_checkpoint_id",
                "selected_adapter_tree_sha256",
                "samples_recompiled",
                "reports_recomputed",
                "backend_mode",
                "fixture_accepted",
                "model_bound",
                "complete_split",
                "blind_data_accessed",
                "calibration_opened",
                "blind_opened",
            ),
        ),
        "global_once_registry": {
            "reservation_sha256": registry_result["sha256"],
            "claim_sha256": claim_sha,
            "terminal_sha256": _sha256_file(terminal_path),
            "nonce_sha256": nonce_sha,
            "one_authorization_per_sealed_blind_hash": True,
            "claim_created_with_exclusive_contract": True,
            "failure_is_non_reusable": True,
            "terminal_status": "COMPLETED",
        },
        "linked_artifacts_rehashed": {
            role: _sha256_file(path)
            for role, path in sorted(resolved_linked.items())
        },
        "fixture_accepted": False,
        "runner_injection_used": False,
    }


def _exclusive_write(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise


def _exclusive_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with source.open("rb") as source_handle, os.fdopen(
            descriptor,
            "wb",
            closefd=True,
        ) as destination_handle:
            shutil.copyfileobj(
                source_handle,
                destination_handle,
                length=1024 * 1024,
            )
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
    except BaseException:
        raise


def _artifact_record(role: str, path: str, source: Path) -> dict[str, Any]:
    _validate_package_path(path)
    return {
        "role": role,
        "path": path,
        "bytes": source.stat().st_size,
        "sha256": _sha256_file(source),
    }


def _portable_inventory(value: Mapping[str, Any]) -> dict[str, Any]:
    portable = {
        key: item
        for key, item in value.items()
        if key not in {"path", "root", "directory"}
    }
    return json.loads(_canonical_bytes(portable).decode("utf-8"))


def _evidence_index(
    *,
    candidate_id: str,
    created_at_utc: str,
    bindings: Mapping[str, Any],
    dataset_inventory: Mapping[str, Any],
    model_inventories: Mapping[str, Mapping[str, Any]],
    authoritative_verification: Mapping[str, Any],
) -> dict[str, Any]:
    receipts = [
        {
            "role": role,
            "package_path": _ROLE_DESTINATIONS[role],
            **dict(bindings["receipts"][role]),
        }
        for role in sorted(_RECEIPT_ROLES)
    ]
    body = {
        "schema": EVIDENCE_INDEX_SCHEMA,
        "status": EVIDENCE_STATUS,
        "candidate_id": candidate_id,
        "created_at_utc": created_at_utc,
        "package_type": PACKAGE_TYPE,
        "receipts": receipts,
        "canonical_inventories": {
            "dataset": _portable_inventory(dataset_inventory),
            "base_model": _portable_inventory(
                model_inventories["base_model"]
            ),
            "selected_checkpoint": _portable_inventory(
                model_inventories["selected_checkpoint"]
            ),
            "selected_adapter": _portable_inventory(
                model_inventories["selected_adapter"]
            ),
        },
        "authoritative_verification": dict(authoritative_verification),
        "authorizations": {
            "offline_bundle_publication_authorized": True,
            "gguf_artifact_already_qualified": True,
            "activation_authorized": False,
            "deployment_authorized": False,
            "production_integration_authorized": False,
            "x5_deployment_authorized": False,
            "service_registration_authorized": False,
            "rb_voe_enable_authorized": False,
        },
        "cross_bindings": {
            "dataset_manifest_sha256": bindings["manifest_sha256"],
            "dataset_tree_sha256": bindings["dataset_tree_sha256"],
            "base_model_tree_sha256": bindings["base_model_tree_sha256"],
            "selected_checkpoint_tree_sha256": bindings[
                "selected_checkpoint_tree_sha256"
            ],
            "selected_adapter_tree_sha256": (
                bindings["selected_adapter_tree_sha256"]
            ),
            "blind_sha256": bindings["blind_sha256"],
            "blind_qualification_sha256": bindings[
                "blind_qualification_sha256"
            ],
            "validation_golden_sha256": bindings[
                "validation_golden_sha256"
            ],
            "gguf_sha256": bindings["gguf_sha256"],
            "all_upstream_receipts_passed": True,
            "all_upstream_files_rehashed": True,
            "canonical_blind_qualification_recomputed": True,
            "calibration_recomputed": True,
            "ablation_recomputed": True,
            "global_once_claim_verified": True,
        },
        "system_boundary": FROZEN_SYSTEM_BOUNDARY,
    }
    return {
        **body,
        "canonical_digest_sha256": _sha256_bytes(_canonical_bytes(body)),
    }


def _content_descriptor(
    *,
    candidate_id: str,
    created_at_utc: str,
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": RELEASE_SCHEMA,
        "candidate_id": candidate_id,
        "product_id": PRODUCT_ID,
        "created_at_utc": created_at_utc,
        "package_type": PACKAGE_TYPE,
        "entries": [dict(row) for row in entries],
        "system_boundary": FROZEN_SYSTEM_BOUNDARY,
    }


def _release_manifest(
    *,
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    content_id = _sha256_bytes(_canonical_bytes(descriptor))
    body = {
        **dict(descriptor),
        "status": RELEASE_STATUS,
        "content_id": content_id,
        "entry_count": len(descriptor["entries"]),
        "manifest_order": "path_casefold_then_path_then_role",
        "archive": {
            "format": "ZIP_STORED",
            "root": _ZIP_ROOT,
            "fixed_timestamp": "1980-01-01T00:00:00",
            "bit_for_bit_reproducible": True,
        },
    }
    return {
        **body,
        "canonical_digest_sha256": _sha256_bytes(_canonical_bytes(body)),
    }


def _sorted_entries(entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        entries,
        key=lambda row: (
            str(row["path"]).casefold(),
            str(row["path"]),
            str(row["role"]),
        ),
    )
    paths = [str(row["path"]) for row in ordered]
    if len(paths) != len(set(path.casefold() for path in paths)):
        raise ReleaseBundleV6Error("duplicate package path")
    for path in paths:
        _validate_package_path(path)
    return ordered


def _scan_staging(
    staging: Path,
    *,
    expected_paths: set[str],
) -> None:
    observed: set[str] = set()
    for path in staging.rglob("*"):
        if path.is_symlink():
            raise ReleaseBundleV6Error(
                f"symbolic link appeared in package: {path}"
            )
        if path.is_file():
            logical = path.relative_to(staging).as_posix()
            _validate_package_path(logical)
            observed.add(logical)
    if observed != expected_paths:
        raise ReleaseBundleV6Error(
            "package inventory differs from the fixed path allowlist"
        )


def _write_reproducible_zip(
    package_dir: Path,
    archive_path: Path,
) -> None:
    files = sorted(
        (
            path
            for path in package_dir.rglob("*")
            if path.is_file()
        ),
        key=lambda path: path.relative_to(package_dir).as_posix(),
    )
    with archive_path.open("xb") as archive_handle:
        with zipfile.ZipFile(
            archive_handle,
            mode="w",
            compression=zipfile.ZIP_STORED,
            strict_timestamps=True,
        ) as bundle:
            for path in files:
                if path.is_symlink():
                    raise ReleaseBundleV6Error(
                        "symbolic links are forbidden in archive input"
                    )
                logical = path.relative_to(package_dir).as_posix()
                _validate_package_path(logical)
                info = zipfile.ZipInfo(
                    f"{_ZIP_ROOT}/{logical}",
                    date_time=_ZIP_TIMESTAMP,
                )
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = (
                    stat.S_IFREG | 0o644
                ) << 16
                info.flag_bits = 0
                with path.open("rb") as source, bundle.open(
                    info,
                    mode="w",
                    force_zip64=True,
                ) as target:
                    shutil.copyfileobj(
                        source,
                        target,
                        length=1024 * 1024,
                    )


def _validate_metadata(candidate_id: str, created_at_utc: str) -> None:
    if _CANDIDATE_ID.fullmatch(candidate_id) is None:
        raise ReleaseBundleV6Error(
            "candidate_id must be a stable lowercase identifier"
        )
    if (
        not isinstance(created_at_utc, str)
        or not created_at_utc.endswith("Z")
        or "T" not in created_at_utc
    ):
        raise ReleaseBundleV6Error(
            "created_at_utc must be an explicit UTC RFC3339 string ending in Z"
        )


def build_release_bundle_v6(
    *,
    workspace_root: Path,
    candidate_id: str,
    created_at_utc: str,
    output_root: Path,
    inputs: ReleaseBundleInputsV6,
) -> dict[str, Any]:
    """Build one immutable content-addressed directory and deterministic ZIP."""

    _validate_metadata(candidate_id, created_at_utc)
    root = _resolve_workspace(workspace_root)
    paths = _resolve_inputs(root, inputs)
    dataset_inventory = _canonical_dataset_inventory(paths["dataset_dir"])
    model_inventories = _canonical_model_inventories(paths)
    records = {
        role: _load_json(paths[role], label=role)
        for role in (
            *_RECEIPT_ROLES,
            "gguf_preflight",
            "gguf_golden_set",
            "task_contract",
            "preprocessing_contract",
        )
    }
    bindings = _validate_receipt_contracts(
        paths=paths,
        records=records,
        dataset_inventory=dataset_inventory,
        model_inventories=model_inventories,
    )
    authoritative_verification = _authoritative_source_verification(
        workspace_root=root,
        paths=paths,
        records=records,
        bindings=bindings,
    )
    source_snapshots = {
        role: {
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for role, path in paths.items()
        if role in _ROLE_DESTINATIONS
    }
    destination_root = _prepare_output_root(root, output_root)
    staging = destination_root / (
        f".{candidate_id}.staging-{uuid.uuid4().hex}"
    )
    staging.mkdir(exist_ok=False)
    archive_temp: Path | None = None
    published_archive: Path | None = None
    final_dir: Path | None = None
    try:
        for role in sorted(_ROLE_DESTINATIONS):
            _exclusive_copy(
                paths[role],
                staging / _ROLE_DESTINATIONS[role],
            )
            copied = staging / _ROLE_DESTINATIONS[role]
            if (
                copied.stat().st_size != source_snapshots[role]["bytes"]
                or _sha256_file(copied) != source_snapshots[role]["sha256"]
                or paths[role].stat().st_size
                != source_snapshots[role]["bytes"]
                or _sha256_file(paths[role])
                != source_snapshots[role]["sha256"]
            ):
                raise ReleaseBundleV6Error(
                    f"{role} changed during release construction"
                )

        evidence_index = _evidence_index(
            candidate_id=candidate_id,
            created_at_utc=created_at_utc,
            bindings=bindings,
            dataset_inventory=dataset_inventory,
            model_inventories=model_inventories,
            authoritative_verification=authoritative_verification,
        )
        evidence_path = staging / _GENERATED_DESTINATIONS["evidence_index"]
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        _exclusive_write(evidence_path, _pretty_bytes(evidence_index))

        entries = [
            _artifact_record(
                role,
                _ROLE_DESTINATIONS[role],
                staging / _ROLE_DESTINATIONS[role],
            )
            for role in _ROLE_DESTINATIONS
        ]
        entries.append(
            _artifact_record(
                "evidence_index",
                _GENERATED_DESTINATIONS["evidence_index"],
                evidence_path,
            )
        )
        entries = _sorted_entries(entries)
        descriptor = _content_descriptor(
            candidate_id=candidate_id,
            created_at_utc=created_at_utc,
            entries=entries,
        )
        manifest = _release_manifest(descriptor=descriptor)
        content_id = manifest["content_id"]
        final_dir = destination_root / content_id
        archive_path = destination_root / f"{content_id}.zip"
        if (
            os.path.lexists(final_dir)
            or os.path.lexists(archive_path)
        ):
            raise FileExistsError(
                "content-addressed release or archive already exists; "
                "overwrite refused"
            )
        manifest_path = (
            staging / _GENERATED_DESTINATIONS["release_manifest"]
        )
        _exclusive_write(manifest_path, _pretty_bytes(manifest))
        _scan_staging(
            staging,
            expected_paths=set(_ALLOWED_PACKAGE_PATHS),
        )

        for row in entries:
            packaged = staging / row["path"]
            if (
                packaged.stat().st_size != row["bytes"]
                or _sha256_file(packaged) != row["sha256"]
            ):
                raise ReleaseBundleV6Error(
                    f"packaged artifact changed before publication: {row['role']}"
                )

        staging.replace(final_dir)
        archive_temp = destination_root / (
            f".{content_id}.{uuid.uuid4().hex}.zip.tmp"
        )
        _write_reproducible_zip(final_dir, archive_temp)
        if os.path.lexists(archive_path):
            raise FileExistsError(
                "content-addressed archive appeared; overwrite refused"
            )
        archive_temp.replace(archive_path)
        archive_temp = None
        published_archive = archive_path
        verification = verify_release_bundle_v6(
            package_dir=final_dir,
            archive_path=archive_path,
        )
        return {
            "schema": BUILD_RESULT_SCHEMA,
            "status": RELEASE_STATUS,
            "candidate_id": candidate_id,
            "product_id": PRODUCT_ID,
            "package_type": PACKAGE_TYPE,
            "content_id": content_id,
            "release_directory": str(final_dir),
            "release_manifest": str(
                final_dir / _GENERATED_DESTINATIONS["release_manifest"]
            ),
            "evidence_index": str(
                final_dir / _GENERATED_DESTINATIONS["evidence_index"]
            ),
            "archive": str(archive_path),
            "archive_sha256": _sha256_file(archive_path),
            "system_boundary": FROZEN_SYSTEM_BOUNDARY,
            "verification": verification,
        }
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if archive_temp is not None:
            try:
                archive_temp.unlink()
            except FileNotFoundError:
                pass
        if published_archive is not None:
            try:
                published_archive.unlink()
            except FileNotFoundError:
                pass
        if final_dir is not None and final_dir.exists():
            shutil.rmtree(final_dir, ignore_errors=True)
        raise


def _verify_manifest_digest(manifest: Mapping[str, Any]) -> None:
    claimed = _require_sha256(
        manifest.get("canonical_digest_sha256"),
        label="release manifest canonical digest",
    )
    body = dict(manifest)
    del body["canonical_digest_sha256"]
    if _sha256_bytes(_canonical_bytes(body)) != claimed:
        raise ReleaseBundleV6Error(
            "release manifest canonical digest mismatch"
        )


def _validate_embedded_inventory(
    value: Any,
    *,
    role: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseBundleV6Error(
            f"embedded {role} inventory is absent"
        )
    files = value.get("files")
    if (
        value.get("ordering") != "windows_casefold_then_posix"
        or not isinstance(files, list)
        or not files
        or value.get("file_count") != len(files)
    ):
        raise ReleaseBundleV6Error(
            f"embedded {role} inventory contract is invalid"
        )
    normalized: list[dict[str, Any]] = []
    folded_paths: set[str] = set()
    for index, row in enumerate(files):
        if not isinstance(row, Mapping):
            raise ReleaseBundleV6Error(
                f"embedded {role} inventory row is invalid: {index}"
            )
        path = row.get("path")
        if not isinstance(path, str):
            raise ReleaseBundleV6Error(
                f"embedded {role} inventory path is invalid"
            )
        pure = PurePosixPath(path)
        if (
            pure.is_absolute()
            or not pure.parts
            or ".." in pure.parts
            or "\\" in path
        ):
            raise ReleaseBundleV6Error(
                f"embedded {role} inventory path is unsafe"
            )
        folded = path.casefold()
        if folded in folded_paths:
            raise ReleaseBundleV6Error(
                f"embedded {role} inventory is casefold ambiguous"
            )
        folded_paths.add(folded)
        _require_sha256(
            row.get("sha256"),
            label=f"embedded {role} inventory SHA-256",
        )
        if not isinstance(row.get("bytes"), int) or int(row["bytes"]) < 0:
            raise ReleaseBundleV6Error(
                f"embedded {role} inventory size is invalid"
            )
        normalized.append(dict(row))
    expected_order = sorted(
        normalized,
        key=lambda row: (
            str(row["path"]).casefold(),
            str(row["path"]),
        ),
    )
    if normalized != expected_order:
        raise ReleaseBundleV6Error(
            f"embedded {role} inventory is not canonically ordered"
        )
    if (
        value.get("bytes")
        != sum(int(row["bytes"]) for row in normalized)
        or value.get("tree_sha256")
        != _sha256_bytes(_canonical_bytes(normalized))
    ):
        raise ReleaseBundleV6Error(
            f"embedded {role} inventory digest is invalid"
        )
    if role == "selected_adapter":
        names = [PurePosixPath(str(row["path"])).name for row in normalized]
        if (
            len(names) != 2
            or "adapter_config.json" not in names
            or sum(
                name in {
                    "adapter_model.safetensors",
                    "adapter_model.bin",
                }
                for name in names
            )
            != 1
        ):
            raise ReleaseBundleV6Error(
                "embedded selected adapter is not the canonical two-file inventory"
            )
    return dict(value)


def verify_release_bundle_v6(
    *,
    package_dir: Path,
    archive_path: Path | None = None,
) -> dict[str, Any]:
    """Independently rehash a published v6 directory and optional ZIP."""

    directory = Path(package_dir).resolve(strict=True)
    if not directory.is_dir() or directory.is_symlink():
        raise ReleaseBundleV6Error(
            "package_dir must be a regular non-symlink directory"
        )
    manifest_path = (
        directory / _GENERATED_DESTINATIONS["release_manifest"]
    )
    manifest = _load_json(
        manifest_path,
        label="release manifest",
    )
    if (
        manifest.get("schema") != RELEASE_SCHEMA
        or manifest.get("status") != RELEASE_STATUS
        or manifest.get("package_type") != PACKAGE_TYPE
        or manifest.get("system_boundary") != FROZEN_SYSTEM_BOUNDARY
    ):
        raise ReleaseBundleV6Error(
            "release manifest contract is invalid"
        )
    _verify_manifest_digest(manifest)
    content_id = _require_sha256(
        manifest.get("content_id"),
        label="content_id",
    )
    if directory.name != content_id:
        raise ReleaseBundleV6Error(
            "package directory is not content-addressed"
        )
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ReleaseBundleV6Error("release manifest entries are invalid")
    normalized = _sorted_entries([dict(row) for row in entries])
    if entries != normalized or manifest.get("entry_count") != len(entries):
        raise ReleaseBundleV6Error(
            "release manifest entries are not stably sorted"
        )
    descriptor = _content_descriptor(
        candidate_id=str(manifest.get("candidate_id")),
        created_at_utc=str(manifest.get("created_at_utc")),
        entries=normalized,
    )
    if _sha256_bytes(_canonical_bytes(descriptor)) != content_id:
        raise ReleaseBundleV6Error("content_id mismatch")
    if manifest != _release_manifest(descriptor=descriptor):
        raise ReleaseBundleV6Error(
            "release manifest differs from its deterministic reconstruction"
        )

    expected_paths = {
        str(row["path"]) for row in normalized
    } | {_GENERATED_DESTINATIONS["release_manifest"]}
    _scan_staging(directory, expected_paths=expected_paths)
    for row in normalized:
        path = directory / str(row["path"])
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != row.get("bytes")
            or _sha256_file(path) != row.get("sha256")
        ):
            raise ReleaseBundleV6Error(
                f"packaged artifact hash mismatch: {row.get('role')}"
            )

    evidence = _load_json(
        directory / _GENERATED_DESTINATIONS["evidence_index"],
        label="evidence index",
    )
    _validate_self_digest(evidence, role="evidence_index")
    if (
        evidence.get("schema") != EVIDENCE_INDEX_SCHEMA
        or evidence.get("status") != EVIDENCE_STATUS
        or evidence.get("package_type") != PACKAGE_TYPE
        or evidence.get("system_boundary") != FROZEN_SYSTEM_BOUNDARY
        or _nested(
            evidence,
            "cross_bindings",
            "all_upstream_receipts_passed",
        )
        is not True
        or _nested(
            evidence,
            "cross_bindings",
            "all_upstream_files_rehashed",
        )
        is not True
        or _nested(
            evidence,
            "cross_bindings",
            "canonical_blind_qualification_recomputed",
        )
        is not True
        or _nested(
            evidence,
            "cross_bindings",
            "calibration_recomputed",
        )
        is not True
        or _nested(
            evidence,
            "cross_bindings",
            "ablation_recomputed",
        )
        is not True
        or _nested(
            evidence,
            "cross_bindings",
            "global_once_claim_verified",
        )
        is not True
    ):
        raise ReleaseBundleV6Error("evidence index contract is invalid")
    if evidence.get("authorizations") != {
        "offline_bundle_publication_authorized": True,
        "gguf_artifact_already_qualified": True,
        "activation_authorized": False,
        "deployment_authorized": False,
        "production_integration_authorized": False,
        "x5_deployment_authorized": False,
        "service_registration_authorized": False,
        "rb_voe_enable_authorized": False,
    }:
        raise ReleaseBundleV6Error(
            "evidence index authorization boundary is invalid"
        )
    inventories = evidence.get("canonical_inventories")
    if not isinstance(inventories, Mapping):
        raise ReleaseBundleV6Error(
            "evidence index canonical inventories are absent"
        )
    dataset_inventory = _validate_embedded_inventory(
        inventories.get("dataset"),
        role="dataset",
    )
    model_inventories = {
        role: _validate_embedded_inventory(
            inventories.get(role),
            role=role,
        )
        for role in (
            "base_model",
            "selected_checkpoint",
            "selected_adapter",
        )
    }
    authoritative = evidence.get("authoritative_verification")
    if (
        not isinstance(authoritative, Mapping)
        or authoritative.get("fixture_accepted") is not False
        or authoritative.get("runner_injection_used") is not False
        or _nested(
            authoritative,
            "blind_release_qualification",
            "qualified",
        )
        is not True
        or _nested(
            authoritative,
            "blind_release_qualification",
            "fixture_accepted",
        )
        is not False
        or _nested(
            authoritative,
            "blind_release_qualification",
            "blind_dataset_reopened",
        )
        is not False
        or _nested(authoritative, "calibration", "model_bound") is not True
        or _nested(authoritative, "calibration", "blind_data_accessed")
        is not False
        or _nested(authoritative, "ablation", "fixture_accepted")
        is not False
        or _nested(authoritative, "ablation", "blind_opened") is not False
        or _nested(
            authoritative,
            "global_once_registry",
            "one_authorization_per_sealed_blind_hash",
        )
        is not True
        or _nested(
            authoritative,
            "global_once_registry",
            "terminal_status",
        )
        != "COMPLETED"
    ):
        raise ReleaseBundleV6Error(
            "evidence index authoritative verifier proof is invalid"
        )
    receipt_rows = evidence.get("receipts")
    if not isinstance(receipt_rows, list) or len(receipt_rows) != len(
        _RECEIPT_ROLES
    ):
        raise ReleaseBundleV6Error(
            "evidence index receipt inventory is invalid"
        )
    for row in receipt_rows:
        if not isinstance(row, Mapping):
            raise ReleaseBundleV6Error(
                "evidence index receipt row is invalid"
            )
        role = str(row.get("role"))
        if role not in _RECEIPT_ROLES:
            raise ReleaseBundleV6Error(
                "evidence index contains an unknown receipt role"
            )
        packaged = directory / _ROLE_DESTINATIONS[role]
        if (
            row.get("package_path") != _ROLE_DESTINATIONS[role]
            or row.get("bytes") != packaged.stat().st_size
            or row.get("sha256") != _sha256_file(packaged)
        ):
            raise ReleaseBundleV6Error(
                f"evidence index does not bind {role}"
            )

    packaged_paths = {
        role: directory / logical
        for role, logical in _ROLE_DESTINATIONS.items()
    }
    packaged_records = {
        role: _load_json(packaged_paths[role], label=role)
        for role in (
            *_RECEIPT_ROLES,
            "gguf_preflight",
            "gguf_golden_set",
            "task_contract",
            "preprocessing_contract",
        )
    }
    recomputed_bindings = _validate_receipt_contracts(
        paths=packaged_paths,
        records=packaged_records,
        dataset_inventory=dataset_inventory,
        model_inventories=model_inventories,
    )
    if (
        _nested(
            authoritative,
            "blind_release_qualification",
            "blind_receipt_sha256",
        )
        != recomputed_bindings["receipts"]["blind_receipt"]["sha256"]
        or _nested(
            authoritative,
            "blind_release_qualification",
            "qualification_receipt_sha256",
        )
        != recomputed_bindings["blind_qualification_sha256"]
        or _nested(
            authoritative,
            "calibration",
            "receipt_sha256",
        )
        != recomputed_bindings["receipts"]["calibration_receipt"]["sha256"]
        or _nested(
            authoritative,
            "ablation",
            "receipt_sha256",
        )
        != recomputed_bindings["receipts"]["ablation_receipt"]["sha256"]
        or _nested(
            authoritative,
            "global_once_registry",
            "claim_sha256",
        )
        != _nested(
            packaged_records["blind_qualification_receipt"],
            "consumption_claim",
            "sha256",
        )
    ):
        raise ReleaseBundleV6Error(
            "authoritative verifier proof differs from packaged receipts"
        )
    if evidence.get("cross_bindings") != {
        "dataset_manifest_sha256": recomputed_bindings[
            "manifest_sha256"
        ],
        "dataset_tree_sha256": recomputed_bindings[
            "dataset_tree_sha256"
        ],
        "base_model_tree_sha256": recomputed_bindings[
            "base_model_tree_sha256"
        ],
        "selected_checkpoint_tree_sha256": recomputed_bindings[
            "selected_checkpoint_tree_sha256"
        ],
        "selected_adapter_tree_sha256": recomputed_bindings[
            "selected_adapter_tree_sha256"
        ],
        "blind_sha256": recomputed_bindings["blind_sha256"],
        "blind_qualification_sha256": recomputed_bindings[
            "blind_qualification_sha256"
        ],
        "validation_golden_sha256": recomputed_bindings[
            "validation_golden_sha256"
        ],
        "gguf_sha256": recomputed_bindings["gguf_sha256"],
        "all_upstream_receipts_passed": True,
        "all_upstream_files_rehashed": True,
        "canonical_blind_qualification_recomputed": True,
        "calibration_recomputed": True,
        "ablation_recomputed": True,
        "global_once_claim_verified": True,
    }:
        raise ReleaseBundleV6Error(
            "evidence index cross-bindings differ from packaged evidence"
        )

    archive_sha256: str | None = None
    if archive_path is not None:
        archive = Path(archive_path).resolve(strict=True)
        if not archive.is_file() or archive.is_symlink():
            raise ReleaseBundleV6Error(
                "archive_path must be a regular non-symlink file"
            )
        expected_names = [
            f"{_ZIP_ROOT}/{path}"
            for path in sorted(expected_paths)
        ]
        with zipfile.ZipFile(archive, "r") as bundle:
            infos = bundle.infolist()
            if [info.filename for info in infos] != expected_names:
                raise ReleaseBundleV6Error(
                    "archive inventory or order is invalid"
                )
            for info in infos:
                logical = info.filename.removeprefix(f"{_ZIP_ROOT}/")
                if (
                    info.date_time != _ZIP_TIMESTAMP
                    or info.compress_type != zipfile.ZIP_STORED
                    or logical not in expected_paths
                ):
                    raise ReleaseBundleV6Error(
                        "archive metadata violates reproducibility contract"
                    )
                payload = bundle.read(info)
                source = directory / logical
                if (
                    len(payload) != source.stat().st_size
                    or _sha256_bytes(payload) != _sha256_file(source)
                ):
                    raise ReleaseBundleV6Error(
                        f"archive payload mismatch: {logical}"
                    )
        archive_sha256 = _sha256_file(archive)

    return {
        "status": "PASS_PC_OFFLINE_RELEASE_PACKAGE_VERIFIED",
        "package_type": PACKAGE_TYPE,
        "content_id": content_id,
        "entry_count": len(normalized),
        "archive_sha256": archive_sha256,
        "x5_deployed": False,
        "activated": False,
        "frozen_five_ports_modified": False,
        "rb_voe_state": "DEPLOYED_OFF_UNCHANGED",
    }


build_llm_release_v6 = build_release_bundle_v6
verify_llm_release_v6 = verify_release_bundle_v6
LlmReleaseInputsV6 = ReleaseBundleInputsV6
LlmReleaseV6Error = ReleaseBundleV6Error


__all__ = [
    "ABLATION_SCHEMA",
    "ABLATION_STATUS",
    "BLIND_QUALIFICATION_SCHEMA",
    "BLIND_QUALIFICATION_STATUS",
    "BLIND_SCHEMA",
    "BLIND_STATUS",
    "BUILD_RESULT_SCHEMA",
    "CALIBRATION_SCHEMA",
    "CALIBRATION_STATUS",
    "CHECKPOINT_EVAL_SCHEMA",
    "CHECKPOINT_EVAL_STATUS",
    "CONTRACTS_SCHEMA",
    "CONTRACTS_STATUS",
    "DATASET_AUDIT_SCHEMA",
    "DATASET_AUDIT_STATUS",
    "EVIDENCE_INDEX_SCHEMA",
    "EVIDENCE_STATUS",
    "FROZEN_SYSTEM_BOUNDARY",
    "GGUF_PARITY_SCHEMA",
    "GGUF_PARITY_STATUS",
    "LlmReleaseInputsV6",
    "LlmReleaseV6Error",
    "PACKAGE_TYPE",
    "PREPROCESSING_CONTRACT_SCHEMA",
    "PRODUCT_ID",
    "RELEASE_SCHEMA",
    "RELEASE_STATUS",
    "ReleaseBundleInputsV6",
    "ReleaseBundleV6Error",
    "SELECTION_FREEZE_SCHEMA",
    "SELECTION_FREEZE_STATUS",
    "TASK_CONTRACT_SCHEMA",
    "TRAINING_SCHEMA",
    "TRAINING_STATUS",
    "build_llm_release_v6",
    "build_release_bundle_v6",
    "verify_llm_release_v6",
    "verify_release_bundle_v6",
]
