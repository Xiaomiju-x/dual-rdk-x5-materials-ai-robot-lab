"""Strict nonblind-v8 validation evaluation for retained QLoRA checkpoints.

The v6 evaluator remains the numerical evaluation engine. This module adds a
fail-closed generation boundary for ``STRICT_NONBLIND_V8`` training receipts:
it binds the immutable v8 manifest, the declared training split, the actual
validation bytes, and the complete training-gate bundle before evaluating any
checkpoint. It never opens train, calibration, or blind split content.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from icmat_foundry.llm import (
    evidence_pointer_v6,
    pointer_hf_eval_v6,
    qlora_full_v6,
    selection_policy_v6,
    shortcut_audit_v8,
    unique_support_audit_v8,
)
from icmat_foundry.llm import (
    pointer_checkpoint_eval_v6 as eval_v6,
)

ORCHESTRATOR_VERSION = "icmat-pointer-checkpoint-eval-v8.0.0"
INDEX_SCHEMA = "icmat_pointer_checkpoint_evaluation_index.v8"
FAILURE_SCHEMA = "icmat_pointer_checkpoint_evaluation_failure.v8"
CANARY_STATUS = (
    "PASS_STRICT_NONBLIND_V8_CANARY_1X6_VALIDATION_EVALUATED_NO_SELECTION"
)
FINAL_STATUS = (
    "PASS_STRICT_NONBLIND_V8_FINAL_3X6_VALIDATION_EVALUATED_NO_SELECTION"
)
FIXTURE_CANARY_STATUS = (
    "FIXTURE_STRICT_NONBLIND_V8_CANARY_1X6_VALIDATION_NONQUALIFYING"
)
FIXTURE_FINAL_STATUS = (
    "FIXTURE_STRICT_NONBLIND_V8_FINAL_3X6_VALIDATION_NONQUALIFYING"
)

STRICT_CONTRACT = "STRICT_NONBLIND_V8"
EXPECTED_MANIFEST_STATUS = (
    "NONBLIND_V8_BUILT_NLI_UNIQUE_SUPPORT_PREBLIND_COMMITTED"
)
EXPECTED_SPLITS = frozenset({"train", "validation", "calibration"})
EXPECTED_SPLIT_COUNTS = {
    "train": 250,
    "validation": 150,
    "calibration": 150,
}
EXPECTED_GATE_KEYS = frozenset(
    {
        "contract",
        "nonblind_compare",
        "scoped_lexical",
        "unique_support",
        "nli_model",
        "training_gate_bundle_sha256",
    }
)
EXPECTED_GATE_STATUSES = {
    "nonblind_compare": qlora_full_v6.NONBLIND_V8_COMPARE_STATUS,
    "scoped_lexical": shortcut_audit_v8.PASS_STATUS,
    "unique_support": unique_support_audit_v8.PASS_STATUS,
}
RUNTIME_LOADING_KEYS = frozenset(
    {
        "policy",
        "content_address",
        "tree_sha256",
        "file_count",
        "bytes",
        "verified_before_cuda",
        "loaded_only_from_snapshot",
        "removed_before_publish",
    }
)
RUNTIME_LOADING_POLICY = "CONTENT_ADDRESSED_STABLE_LOCAL_COPY_V7"

_PRODUCTION_RUNNER = pointer_hf_eval_v6.run_evaluation
_PRODUCTION_RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "evaluate_icmat_pointer_checkpoints_v8.py"
)


class PointerCheckpointEvalV8Error(ValueError):
    """Raised when a strict v8 binding or evaluation invariant fails."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PointerCheckpointEvalV8Error(f"{field} must be an object")
    return value


def _require_sequence(value: Any, *, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PointerCheckpointEvalV8Error(f"{field} must be an array")
    return value


def _require_regular_file(path: Path, *, field: str) -> Path:
    lexical = eval_v6._assert_no_reparse_chain(path, field=field)
    metadata = os.lstat(lexical)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or eval_v6._is_reparse(metadata)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise PointerCheckpointEvalV8Error(
            f"{field} must be a real regular file: {path}"
        )
    return lexical


def _reject_protected_path(path: Path, *, field: str) -> None:
    try:
        eval_v6._reject_blind_path(path, field=field)
    except eval_v6.PointerCheckpointEvalV6Error as exc:
        raise PointerCheckpointEvalV8Error(str(exc)) from exc


def _stable_identity(path: Path) -> dict[str, int]:
    metadata = os.lstat(path)
    return {
        "device": int(metadata.st_dev),
        "file_id": int(metadata.st_ino),
        "size": int(metadata.st_size),
        "mtime_ns": int(metadata.st_mtime_ns),
        "ctime_ns": int(metadata.st_ctime_ns),
    }


def _require_hash_record(
    value: Any,
    *,
    field: str,
    expected_status: str,
) -> dict[str, str]:
    mapping = _require_mapping(value, field=field)
    if set(mapping) != {"sha256", "status"}:
        raise PointerCheckpointEvalV8Error(
            f"{field} must contain exactly sha256 and status"
        )
    if not _valid_sha256(mapping.get("sha256")):
        raise PointerCheckpointEvalV8Error(f"{field}.sha256 is invalid")
    if mapping.get("status") != expected_status:
        raise PointerCheckpointEvalV8Error(f"{field}.status is not a PASS gate")
    return {
        "sha256": str(mapping["sha256"]),
        "status": str(mapping["status"]),
    }


def _verify_gate_bundle(dataset: Mapping[str, Any]) -> dict[str, Any]:
    bundle = _require_mapping(
        dataset.get("training_gate_bundle"),
        field="input_snapshot.dataset.training_gate_bundle",
    )
    if set(bundle) != EXPECTED_GATE_KEYS:
        raise PointerCheckpointEvalV8Error(
            "training_gate_bundle keys do not match STRICT_NONBLIND_V8"
        )
    if bundle.get("contract") != STRICT_CONTRACT:
        raise PointerCheckpointEvalV8Error(
            "training_gate_bundle contract is not STRICT_NONBLIND_V8"
        )
    nonblind_compare = _require_hash_record(
        bundle.get("nonblind_compare"),
        field="training_gate_bundle.nonblind_compare",
        expected_status=EXPECTED_GATE_STATUSES["nonblind_compare"],
    )
    scoped_lexical = _require_hash_record(
        bundle.get("scoped_lexical"),
        field="training_gate_bundle.scoped_lexical",
        expected_status=EXPECTED_GATE_STATUSES["scoped_lexical"],
    )
    unique = _require_mapping(
        bundle.get("unique_support"),
        field="training_gate_bundle.unique_support",
    )
    if set(unique) != {"train", "validation"}:
        raise PointerCheckpointEvalV8Error(
            "training_gate_bundle unique_support must bind train and validation"
        )
    unique_train = _require_hash_record(
        unique.get("train"),
        field="training_gate_bundle.unique_support.train",
        expected_status=EXPECTED_GATE_STATUSES["unique_support"],
    )
    unique_validation = _require_hash_record(
        unique.get("validation"),
        field="training_gate_bundle.unique_support.validation",
        expected_status=EXPECTED_GATE_STATUSES["unique_support"],
    )
    nli_model = _require_mapping(
        bundle.get("nli_model"),
        field="training_gate_bundle.nli_model",
    )
    if set(nli_model) != {"tree_sha256", "receipt_sha256", "device"}:
        raise PointerCheckpointEvalV8Error(
            "training_gate_bundle.nli_model keys mismatch"
        )
    if (
        not _valid_sha256(nli_model.get("tree_sha256"))
        or not _valid_sha256(nli_model.get("receipt_sha256"))
        or nli_model.get("device") != "cpu"
    ):
        raise PointerCheckpointEvalV8Error(
            "training_gate_bundle.nli_model identity is invalid"
        )
    digest_payload = {
        "contract": STRICT_CONTRACT,
        "nonblind_compare": nonblind_compare,
        "scoped_lexical": scoped_lexical,
        "unique_support": {
            "train": unique_train,
            "validation": unique_validation,
        },
        "nli_model": {
            "tree_sha256": str(nli_model["tree_sha256"]),
            "receipt_sha256": str(nli_model["receipt_sha256"]),
            "device": "cpu",
        },
    }
    digest = _canonical_sha256(digest_payload)
    if bundle.get("training_gate_bundle_sha256") != digest:
        raise PointerCheckpointEvalV8Error(
            "training_gate_bundle_sha256 does not match its canonical payload"
        )

    audit_gates = _require_mapping(
        dataset.get("strict_audit_gates"),
        field="input_snapshot.dataset.strict_audit_gates",
    )
    if set(audit_gates) != {
        "nonblind_compare",
        "scoped_lexical",
        "unique_support",
    }:
        raise PointerCheckpointEvalV8Error(
            "strict_audit_gates keys mismatch"
        )

    def projection(value: Any, *, field: str) -> dict[str, Any]:
        mapping = _require_mapping(value, field=field)
        return {
            "sha256": mapping.get("sha256"),
            "status": mapping.get("status"),
        }

    audit_unique = _require_mapping(
        audit_gates.get("unique_support"),
        field="strict_audit_gates.unique_support",
    )
    if set(audit_unique) != {"train", "validation"}:
        raise PointerCheckpointEvalV8Error(
            "strict_audit_gates.unique_support splits mismatch"
        )
    if (
        projection(
            audit_gates.get("nonblind_compare"),
            field="strict_audit_gates.nonblind_compare",
        )
        != nonblind_compare
        or projection(
            audit_gates.get("scoped_lexical"),
            field="strict_audit_gates.scoped_lexical",
        )
        != scoped_lexical
        or projection(
            audit_unique.get("train"),
            field="strict_audit_gates.unique_support.train",
        )
        != unique_train
        or projection(
            audit_unique.get("validation"),
            field="strict_audit_gates.unique_support.validation",
        )
        != unique_validation
    ):
        raise PointerCheckpointEvalV8Error(
            "training_gate_bundle is not bound to strict_audit_gates"
        )

    seed_revalidation = _require_mapping(
        dataset.get("seed_revalidation"),
        field="input_snapshot.dataset.seed_revalidation",
    )
    nli_identity = _require_mapping(
        seed_revalidation.get("nli_model"),
        field="input_snapshot.dataset.seed_revalidation.nli_model",
    )
    nli_receipt = _require_mapping(
        nli_identity.get("model_receipt"),
        field="seed_revalidation.nli_model.model_receipt",
    )
    if (
        nli_identity.get("tree_sha256") != nli_model["tree_sha256"]
        or nli_receipt.get("sha256") != nli_model["receipt_sha256"]
    ):
        raise PointerCheckpointEvalV8Error(
            "gate bundle NLI identity is not bound to seed revalidation"
        )
    return {
        **digest_payload,
        "training_gate_bundle_sha256": digest,
    }


def _split_binding(
    *,
    split: str,
    snapshot: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    expected_count = EXPECTED_SPLIT_COUNTS[split]
    expected_path = f"{split}.jsonl"
    if (
        snapshot.get("path") != expected_path
        or snapshot.get("examples") != expected_count
        or not isinstance(snapshot.get("bytes"), int)
        or isinstance(snapshot.get("bytes"), bool)
        or int(snapshot["bytes"]) <= 0
        or not _valid_sha256(snapshot.get("sha256"))
    ):
        raise PointerCheckpointEvalV8Error(
            f"training receipt {split} split declaration is invalid"
        )
    if (
        manifest.get("path") != expected_path
        or manifest.get("count") != expected_count
        or manifest.get("bytes") != snapshot["bytes"]
        or manifest.get("sha256") != snapshot["sha256"]
    ):
        raise PointerCheckpointEvalV8Error(
            f"v8 manifest and training receipt disagree on {split}"
        )
    return {
        "path": expected_path,
        "bytes": int(snapshot["bytes"]),
        "sha256": str(snapshot["sha256"]),
        "examples": expected_count,
    }


def _expected_run_id(
    *,
    receipt: Mapping[str, Any],
    training_root: Path,
    dataset: Mapping[str, Any],
) -> str:
    input_snapshot = _require_mapping(
        receipt.get("input_snapshot"),
        field="input_snapshot",
    )
    base_model = _require_mapping(
        input_snapshot.get("base_model"),
        field="input_snapshot.base_model",
    )
    source_files = input_snapshot.get("source_files")
    if not isinstance(source_files, Mapping) or not source_files:
        raise PointerCheckpointEvalV8Error(
            "input_snapshot.source_files must be a non-empty object"
        )
    configuration = _require_mapping(
        receipt.get("configuration"),
        field="configuration",
    )
    configuration_sha256 = receipt.get("configuration_sha256")
    if (
        not _valid_sha256(configuration_sha256)
        or configuration_sha256 != _canonical_sha256(configuration)
    ):
        raise PointerCheckpointEvalV8Error(
            "configuration_sha256 does not bind the training configuration"
        )
    canary_acceptance = input_snapshot.get("canary_acceptance")
    stage = receipt.get("stage")
    if stage == "canary":
        if canary_acceptance != {
            "required_for_final_training": False,
            "provided": False,
            "validated": False,
        }:
            raise PointerCheckpointEvalV8Error(
                "canary training must carry the fixed no-acceptance receipt"
            )
        canary_acceptance_sha256 = None
    elif stage == "final":
        acceptance = _require_mapping(
            canary_acceptance,
            field="input_snapshot.canary_acceptance",
        )
        canary_acceptance_sha256 = acceptance.get("sha256")
        if not _valid_sha256(canary_acceptance_sha256):
            raise PointerCheckpointEvalV8Error(
                "final training canary acceptance SHA-256 is invalid"
            )
    else:
        raise PointerCheckpointEvalV8Error(
            "training receipt stage must be canary or final"
        )
    inspected = dataset.get("v8_inspected_input_sha256")
    if not _valid_sha256(inspected):
        raise PointerCheckpointEvalV8Error(
            "v8_inspected_input_sha256 is invalid"
        )
    model_tree = base_model.get("tree_sha256")
    if not _valid_sha256(model_tree):
        raise PointerCheckpointEvalV8Error(
            "base-model tree SHA-256 is invalid"
        )
    run_core = {
        "trainer_version": qlora_full_v6.TRAINER_VERSION,
        "stage": stage,
        "dataset_input_sha256": inspected,
        "model_tree_sha256": model_tree,
        "configuration_sha256": configuration_sha256,
        "canary_acceptance_sha256": canary_acceptance_sha256,
        "source_inventory": source_files,
        "output_name": training_root.name,
        "training_gate_bundle_sha256": (
            dataset["training_gate_bundle_sha256"]
        ),
        "v8_inspected_input_sha256": inspected,
    }
    return "icmat-v6-" + _canonical_sha256(run_core)[:20]


def verify_strict_nonblind_v8_binding(
    *,
    receipt: Mapping[str, Any],
    receipt_path: Path,
    dataset_dir: Path,
) -> dict[str, Any]:
    """Verify v8 generation bindings without opening non-validation splits."""

    if (
        receipt.get("schema") != qlora_full_v6.RUN_RECEIPT_SCHEMA
        or receipt.get("trainer_version") != qlora_full_v6.TRAINER_VERSION
        or receipt.get("atomic_publish") is not True
        or receipt.get("network_used") is not False
    ):
        raise PointerCheckpointEvalV8Error(
            "training receipt is not an immutable local QLoRA v8 run"
        )
    root_raw = eval_v6._assert_no_reparse_chain(
        Path(dataset_dir),
        field="dataset directory",
    )
    _reject_protected_path(root_raw, field="dataset directory")
    root = root_raw.resolve(strict=True)
    if not root.is_dir():
        raise PointerCheckpointEvalV8Error(
            "dataset directory must be a real directory"
        )
    input_snapshot = _require_mapping(
        receipt.get("input_snapshot"),
        field="input_snapshot",
    )
    dataset = _require_mapping(
        input_snapshot.get("dataset"),
        field="input_snapshot.dataset",
    )
    if dataset.get("path") != str(root):
        raise PointerCheckpointEvalV8Error(
            "dataset directory does not match the training receipt"
        )
    if dataset.get("contract") != STRICT_CONTRACT:
        raise PointerCheckpointEvalV8Error(
            "training receipt does not declare STRICT_NONBLIND_V8"
        )

    manifest_receipt = _require_mapping(
        dataset.get("manifest"),
        field="input_snapshot.dataset.manifest",
    )
    if (
        manifest_receipt.get("path")
        != qlora_full_v6.NONBLIND_V8_MANIFEST_NAME
        or manifest_receipt.get("schema")
        != qlora_full_v6.NONBLIND_V8_MANIFEST_SCHEMA
        or manifest_receipt.get("dataset_schema")
        != qlora_full_v6.DATASET_SCHEMA
        or manifest_receipt.get("builder_version")
        != qlora_full_v6.NONBLIND_V8_BUILDER_VERSION
        or not _valid_sha256(manifest_receipt.get("sha256"))
    ):
        raise PointerCheckpointEvalV8Error(
            "training receipt v8 manifest identity is invalid"
        )
    manifest_path = _require_regular_file(
        root / qlora_full_v6.NONBLIND_V8_MANIFEST_NAME,
        field="v8 manifest",
    )
    manifest_sha256 = eval_v6._sha256_file(manifest_path)
    manifest_bytes = manifest_path.stat().st_size
    if (
        manifest_receipt.get("sha256") != manifest_sha256
        or manifest_receipt.get("bytes") != manifest_bytes
        or manifest_receipt.get("stable_identity")
        != _stable_identity(manifest_path)
    ):
        raise PointerCheckpointEvalV8Error(
            "v8 manifest bytes or stable identity changed after training"
        )
    manifest = eval_v6._load_json(manifest_path, field="v8 manifest")
    if (
        manifest.get("schema")
        != qlora_full_v6.NONBLIND_V8_MANIFEST_SCHEMA
        or manifest.get("dataset_schema") != qlora_full_v6.DATASET_SCHEMA
        or manifest.get("builder_version")
        != qlora_full_v6.NONBLIND_V8_BUILDER_VERSION
        or manifest.get("status") != EXPECTED_MANIFEST_STATUS
    ):
        raise PointerCheckpointEvalV8Error(
            "v8 manifest schema, builder, or status mismatch"
        )
    manifest_splits = _require_mapping(
        manifest.get("splits"),
        field="v8 manifest splits",
    )
    receipt_splits = _require_mapping(
        dataset.get("splits"),
        field="input_snapshot.dataset.splits",
    )
    if (
        set(manifest_splits) != EXPECTED_SPLITS
        or set(receipt_splits) != EXPECTED_SPLITS
    ):
        raise PointerCheckpointEvalV8Error(
            "v8 split inventory must be exactly train, validation, calibration"
        )
    split_bindings = {
        split: _split_binding(
            split=split,
            snapshot=_require_mapping(
                receipt_splits.get(split),
                field=f"input_snapshot.dataset.splits.{split}",
            ),
            manifest=_require_mapping(
                manifest_splits.get(split),
                field=f"v8 manifest splits.{split}",
            ),
        )
        for split in sorted(EXPECTED_SPLITS)
    }

    sealed = _require_mapping(
        manifest.get("sealed_blind_access"),
        field="v8 manifest sealed_blind_access",
    )
    if sealed != {
        "hashed": False,
        "path_discovered": False,
        "read": False,
    }:
        raise PointerCheckpointEvalV8Error(
            "v8 manifest does not preserve the sealed blind boundary"
        )
    training_boundary = _require_mapping(
        manifest.get("training_boundary"),
        field="v8 manifest training_boundary",
    )
    if (
        training_boundary.get("allowed_splits")
        != ["train", "validation"]
        or training_boundary.get("calibration_content_for_training")
        is not False
    ):
        raise PointerCheckpointEvalV8Error(
            "v8 manifest training boundary is invalid"
        )

    gate_bundle = _verify_gate_bundle(dataset)
    gate_digest = gate_bundle["training_gate_bundle_sha256"]
    inspected = dataset.get("v8_inspected_input_sha256")
    if (
        dataset.get("training_gate_bundle_sha256") != gate_digest
        or receipt.get("training_gate_bundle_sha256") != gate_digest
        or not _valid_sha256(inspected)
        or dataset.get("inspected_input_sha256") != inspected
        or receipt.get("v8_inspected_input_sha256") != inspected
    ):
        raise PointerCheckpointEvalV8Error(
            "top-level v8 gate or inspected-input binding mismatch"
        )

    data_access = _require_mapping(
        receipt.get("data_access"),
        field="training receipt data_access",
    )
    if (
        data_access.get("train_content_read") is not True
        or data_access.get("validation_content_read") is not True
        or data_access.get("calibration_content_read") is not False
        or data_access.get("calibration_content_hashed") is not False
        or data_access.get("blind_test_content_read") is not False
        or data_access.get("blind_test_content_hashed") is not False
    ):
        raise PointerCheckpointEvalV8Error(
            "training receipt data-access boundary is invalid"
        )
    training_access = _require_mapping(
        dataset.get("training_data_access"),
        field="input_snapshot.dataset.training_data_access",
    )
    if (
        training_access.get("opened_splits") != ["train", "validation"]
        or training_access.get("integrity_only_splits") != ["calibration"]
        or training_access.get("calibration_content_loaded_for_training")
        is not False
        or training_access.get("calibration_used_for_checkpoint_selection")
        is not False
    ):
        raise PointerCheckpointEvalV8Error(
            "dataset training-data access receipt is invalid"
        )
    for field in (
        "blind_materialized",
        "blind_discovered",
        "blind_path_constructed",
        "blind_filesystem_metadata_accessed",
        "blind_content_opened",
        "blind_content_read",
        "blind_content_hashed",
    ):
        if training_access.get(field) is not False:
            raise PointerCheckpointEvalV8Error(
                f"dataset training_data_access.{field} must remain false"
            )

    training_root = Path(receipt_path).resolve(strict=True).parent
    expected_run_id = _expected_run_id(
        receipt=receipt,
        training_root=training_root,
        dataset=dataset,
    )
    if receipt.get("run_id") != expected_run_id:
        raise PointerCheckpointEvalV8Error(
            "training run_id does not bind the v8 gate bundle and inputs"
        )

    validation_path = _require_regular_file(
        root / "validation.jsonl",
        field="validation split",
    )
    validation_actual = {
        "path": "validation.jsonl",
        "bytes": validation_path.stat().st_size,
        "sha256": eval_v6._sha256_file(validation_path),
        "examples": EXPECTED_SPLIT_COUNTS["validation"],
    }
    if validation_actual != split_bindings["validation"]:
        raise PointerCheckpointEvalV8Error(
            "actual validation bytes do not match the v8 generation"
        )
    return {
        "contract": STRICT_CONTRACT,
        "manifest": {
            "path": str(manifest_path),
            "bytes": manifest_bytes,
            "sha256": manifest_sha256,
            "schema": qlora_full_v6.NONBLIND_V8_MANIFEST_SCHEMA,
            "builder_version": qlora_full_v6.NONBLIND_V8_BUILDER_VERSION,
        },
        "train": {
            **split_bindings["train"],
            "content_opened_by_evaluator": False,
            "content_hashed_by_evaluator": False,
            "binding_source": "training_receipt_plus_hashed_v8_manifest",
        },
        "validation": {
            **validation_actual,
            "content_opened_by_evaluator": True,
            "content_hashed_by_evaluator": True,
        },
        "gate_bundle": gate_bundle,
        "training_gate_bundle_sha256": gate_digest,
        "v8_inspected_input_sha256": str(inspected),
        "calibration_content_opened_by_evaluator": False,
        "calibration_content_hashed_by_evaluator": False,
        "blind_content_opened_by_evaluator": False,
        "blind_content_hashed_by_evaluator": False,
    }


def _source_bindings(runner_path: Path) -> dict[str, Any]:
    paths = {
        "orchestrator_v8": Path(__file__).resolve(),
        "evaluation_engine_v6": Path(eval_v6.__file__).resolve(),
        "pointer_evaluator": Path(pointer_hf_eval_v6.__file__).resolve(),
        "pointer_compiler": Path(evidence_pointer_v6.__file__).resolve(),
        "selection_policy": Path(selection_policy_v6.__file__).resolve(),
        "runner_v8": Path(runner_path).resolve(strict=True),
    }
    result: dict[str, Any] = {}
    for name, path in paths.items():
        _require_regular_file(path, field=f"{name} source")
        result[name] = {
            "path": str(path),
            "sha256": eval_v6._sha256_file(path),
        }
    return result


def verify_v8_base_model_binding(
    *,
    receipt: Mapping[str, Any],
    base_model_dir: Path,
) -> dict[str, Any]:
    """Verify the real model snapshot plus the exact runtime-copy receipt."""

    raw = eval_v6._assert_no_reparse_chain(
        Path(base_model_dir),
        field="base model directory",
    )
    _reject_protected_path(raw, field="base model directory")
    root = raw.resolve(strict=True)
    if not root.is_dir():
        raise PointerCheckpointEvalV8Error(
            "base model directory must be a real directory"
        )
    input_snapshot = _require_mapping(
        receipt.get("input_snapshot"),
        field="input_snapshot",
    )
    recorded = _require_mapping(
        input_snapshot.get("base_model"),
        field="input_snapshot.base_model",
    )
    try:
        actual = qlora_full_v6._model_snapshot(root)
    except (OSError, qlora_full_v6.QLoRAV6Error) as exc:
        raise PointerCheckpointEvalV8Error(
            f"base model snapshot validation failed: {exc}"
        ) from exc
    expected_keys = set(actual) | {"runtime_loading"}
    if set(recorded) != expected_keys:
        raise PointerCheckpointEvalV8Error(
            "input_snapshot.base_model must contain the exact model "
            "snapshot plus runtime_loading"
        )
    recorded_core = {
        key: recorded[key]
        for key in actual
    }
    if recorded_core != actual:
        raise PointerCheckpointEvalV8Error(
            "input_snapshot.base_model core does not match the exact local "
            "model path/tree/bytes/count/files/config identity"
        )
    runtime = _require_mapping(
        recorded.get("runtime_loading"),
        field="input_snapshot.base_model.runtime_loading",
    )
    if set(runtime) != RUNTIME_LOADING_KEYS:
        raise PointerCheckpointEvalV8Error(
            "runtime_loading must contain exactly the allowed eight fields"
        )
    expected_runtime = {
        "policy": RUNTIME_LOADING_POLICY,
        "content_address": f"sha256:{actual['tree_sha256']}",
        "tree_sha256": actual["tree_sha256"],
        "file_count": actual["file_count"],
        "bytes": actual["bytes"],
        "verified_before_cuda": True,
        "loaded_only_from_snapshot": True,
        "removed_before_publish": True,
    }
    if dict(runtime) != expected_runtime:
        raise PointerCheckpointEvalV8Error(
            "runtime_loading does not exactly bind the model tree, bytes, "
            "file count, verified local copy, and removal-before-publish"
        )

    projected_input = dict(input_snapshot)
    projected_input["base_model"] = actual
    projected_receipt = dict(receipt)
    projected_receipt["input_snapshot"] = projected_input
    try:
        engine_binding = eval_v6._verify_base_binding(
            receipt=projected_receipt,
            base_model_dir=root,
        )
    except eval_v6.PointerCheckpointEvalV6Error as exc:
        raise PointerCheckpointEvalV8Error(str(exc)) from exc
    return {
        **engine_binding,
        "receipt_core_exact": True,
        "runtime_loading": expected_runtime,
    }


def run_checkpoint_evaluations_v8(
    *,
    training_receipt_path: Path,
    dataset_dir: Path,
    base_model_dir: Path,
    output_dir: Path,
    device: str,
    evaluation_seed: int = 20260729,
    runner_path: Path,
    evaluation_runner: Callable[..., Mapping[str, Any]] | None = None,
    fixture_mode: bool = False,
) -> dict[str, Any]:
    """Evaluate all retained checkpoints under a strict nonblind-v8 gate."""

    if not isinstance(fixture_mode, bool):
        raise PointerCheckpointEvalV8Error("fixture_mode must be boolean")
    if fixture_mode:
        if evaluation_runner is None:
            raise PointerCheckpointEvalV8Error(
                "fixture_mode requires an explicit non-production runner"
            )
        effective_runner = evaluation_runner
        runner_mode = "test_fixture_nonqualifying"
    else:
        if evaluation_runner is not None:
            raise PointerCheckpointEvalV8Error(
                "production evaluation forbids injected runners"
            )
        if Path(runner_path).resolve(strict=True) != (
            _PRODUCTION_RUNNER_PATH.resolve(strict=True)
        ):
            raise PointerCheckpointEvalV8Error(
                "production evaluation requires the fixed v8 repository CLI"
            )
        effective_runner = _PRODUCTION_RUNNER
        runner_mode = "production_fixed_v8"
    if device not in {"cpu", "cuda"}:
        raise PointerCheckpointEvalV8Error(
            "device must be explicitly cpu or cuda"
        )
    if (
        isinstance(evaluation_seed, bool)
        or not isinstance(evaluation_seed, int)
        or not 0 <= evaluation_seed <= 2_147_483_647
    ):
        raise PointerCheckpointEvalV8Error(
            "evaluation_seed must be an integer in [0, 2147483647]"
        )
    for field, path in (
        ("training receipt", training_receipt_path),
        ("dataset directory", dataset_dir),
        ("base model directory", base_model_dir),
        ("runner source", runner_path),
    ):
        _reject_protected_path(Path(path), field=field)

    try:
        output = eval_v6._new_output_root(Path(output_dir))
    except eval_v6.PointerCheckpointEvalV6Error as exc:
        raise PointerCheckpointEvalV8Error(str(exc)) from exc
    current_checkpoint: str | None = None
    completed: list[str] = []
    try:
        receipt_path = _require_regular_file(
            Path(training_receipt_path),
            field="training receipt",
        ).resolve(strict=True)
        if receipt_path.name != "training_receipt.v6.json":
            raise PointerCheckpointEvalV8Error(
                "training receipt filename must be training_receipt.v6.json"
            )
        receipt_sha256 = eval_v6._sha256_file(receipt_path)
        receipt = eval_v6._load_json(
            receipt_path,
            field="training receipt",
        )
        strict_binding = verify_strict_nonblind_v8_binding(
            receipt=receipt,
            receipt_path=receipt_path,
            dataset_dir=Path(dataset_dir),
        )
        sources = _source_bindings(Path(runner_path))
        dataset_binding = eval_v6._verify_dataset_binding(
            receipt=receipt,
            dataset_dir=Path(dataset_dir),
        )
        if (
            dataset_binding["sha256"]
            != strict_binding["validation"]["sha256"]
            or dataset_binding["bytes"]
            != strict_binding["validation"]["bytes"]
        ):
            raise PointerCheckpointEvalV8Error(
                "v6 evaluation engine and v8 validation binding disagree"
            )
        base_binding = verify_v8_base_model_binding(
            receipt=receipt,
            base_model_dir=Path(base_model_dir),
        )
        stage, specs = eval_v6._checkpoint_specs(
            receipt=receipt,
            training_root=receipt_path.parent,
        )

        if stage == "canary":
            evaluation_dataset, canary_report = (
                eval_v6._canary_validation_view(
                    dataset_dir=Path(dataset_dir),
                    output_root=output,
                )
            )
            expected_examples = eval_v6.EXPECTED_CANARY_ROWS
        else:
            evaluation_dataset = Path(dataset_dir).resolve(strict=True)
            expected_examples = eval_v6.EXPECTED_SOURCE_VALIDATION_ROWS
            canary_report = None
        validation_selection = pointer_hf_eval_v6.select_dataset(
            dataset_dir=evaluation_dataset,
            split="validation",
            max_samples=None,
        )
        if validation_selection.rows_total != expected_examples:
            raise PointerCheckpointEvalV8Error(
                f"{stage} validation must contain exactly "
                f"{expected_examples} rows"
            )

        evaluation_root = output / "checkpoint_evaluations"
        os.mkdir(evaluation_root)
        records: list[dict[str, Any]] = []
        checkpoint_evidence: list[dict[str, Any]] = []
        for spec in specs:
            current_checkpoint = str(spec["checkpoint_id"])
            checkpoint_output = (
                evaluation_root
                / f"seed-{spec['seed']}"
                / f"epoch-{int(spec['epoch']):02d}"
            )
            checkpoint_output.parent.mkdir(exist_ok=True)
            effective_runner(
                dataset_dir=evaluation_dataset,
                split="validation",
                output_dir=checkpoint_output,
                backend_mode="hf_model",
                base_model_dir=Path(base_model_dir),
                adapter_dir=Path(spec["path"]),
                device=device,
                seed=evaluation_seed,
                max_samples=None,
                runner_path=Path(runner_path),
            )
            record, artifacts = eval_v6._recompute_record(
                evaluation_dir=checkpoint_output,
                spec=spec,
                expected_examples=expected_examples,
                validation_selection=validation_selection,
                expected_base_tree=base_binding[
                    "evaluator_tree_sha256"
                ],
                evaluator_source_sha256=sources["pointer_evaluator"][
                    "sha256"
                ],
                compiler_source_sha256=sources["pointer_compiler"][
                    "sha256"
                ],
                runner_source_sha256=sources["runner_v8"]["sha256"],
            )
            records.append(record)
            checkpoint_evidence.append(
                {
                    "checkpoint_id": spec["checkpoint_id"],
                    "seed": spec["seed"],
                    "epoch": spec["epoch"],
                    "global_step": spec["global_step"],
                    "validation_loss": spec["validation_loss"],
                    "checkpoint_path": str(spec["path"]),
                    "receipt_relative_path": spec["receipt_path"],
                    "training_checkpoint_tree_sha256": spec[
                        "training_checkpoint_tree_sha256"
                    ],
                    "training_adapter_tree_sha256": spec[
                        "training_adapter_tree_sha256"
                    ],
                    "evaluator_adapter_tree_sha256": spec[
                        "evaluator_adapter_tree_sha256"
                    ],
                    "checkpoint_files": spec["checkpoint_files"],
                    "checkpoint_bytes": spec["checkpoint_bytes"],
                    "evaluation_directory": str(checkpoint_output),
                    "evaluation_artifacts": artifacts,
                }
            )
            completed.append(current_checkpoint)

        if eval_v6._sha256_file(receipt_path) != receipt_sha256:
            raise PointerCheckpointEvalV8Error(
                "training receipt changed during evaluation"
            )
        strict_after = verify_strict_nonblind_v8_binding(
            receipt=receipt,
            receipt_path=receipt_path,
            dataset_dir=Path(dataset_dir),
        )
        if strict_after != strict_binding:
            raise PointerCheckpointEvalV8Error(
                "v8 dataset or gate binding changed during evaluation"
            )
        if _source_bindings(Path(runner_path)) != sources:
            raise PointerCheckpointEvalV8Error(
                "evaluation source changed during evaluation"
            )
        expected_checkpoint_count = 6 if stage == "canary" else 18
        if (
            len(records) != expected_checkpoint_count
            or len(completed) != expected_checkpoint_count
        ):
            raise PointerCheckpointEvalV8Error(
                "not every retained checkpoint produced verified evidence"
            )

        status = (
            (
                CANARY_STATUS
                if stage == "canary"
                else FINAL_STATUS
            )
            if not fixture_mode
            else (
                FIXTURE_CANARY_STATUS
                if stage == "canary"
                else FIXTURE_FINAL_STATUS
            )
        )
        index = {
            "schema": INDEX_SCHEMA,
            "orchestrator_version": ORCHESTRATOR_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "status": status,
            "stage": stage,
            "training": {
                "receipt_path": str(receipt_path),
                "receipt_sha256": receipt_sha256,
                "run_id": receipt.get("run_id"),
                "checkpoint_count": len(records),
                "contract": STRICT_CONTRACT,
                "training_gate_bundle_sha256": strict_binding[
                    "training_gate_bundle_sha256"
                ],
                "v8_inspected_input_sha256": strict_binding[
                    "v8_inspected_input_sha256"
                ],
            },
            "strict_nonblind_v8_binding": strict_binding,
            "dataset": {
                **dataset_binding,
                "evaluation_directory": str(evaluation_dataset),
                "evaluated_rows_per_checkpoint": expected_examples,
                "canary_selection": canary_report,
                "opened_split": "validation",
                "train_content_read": False,
                "train_content_hashed": False,
                "calibration_content_read": False,
                "calibration_content_hashed": False,
                "blind_test_content_read": False,
                "blind_test_content_hashed": False,
            },
            "base_model": base_binding,
            "execution": {
                "backend": "hf_model",
                "runner_mode": runner_mode,
                "device": device,
                "seed": evaluation_seed,
                "split": "validation",
                "max_samples": None,
                "checkpoint_outputs_immutable": True,
                "per_sample_metrics_recomputed": True,
                "summary_metrics_trusted": False,
                "selection_policy_invoked": False,
                "checkpoint_selected": False,
                "freeze_created": False,
            },
            "implementation": sources,
            "checkpoints": checkpoint_evidence,
            "records": records,
            "selection": {
                "performed": False,
                "selected_checkpoint_id": None,
                "required_next_step": (
                    "independent v8 selection-policy evaluation and freeze"
                ),
            },
            "authorization": {
                "checkpoint_selected": False,
                "model_authorized": False,
                "calibration_authorized": False,
                "blind_test_authorized": False,
                "gguf_export_authorized": False,
                "deployment_authorized": False,
                "production_integration_authorized": False,
            },
            "claim_boundary": (
                (
                    "This fixture index is test-only and cannot authorize "
                    "selection, calibration, blind evaluation, export, "
                    "release, or deployment. "
                    if fixture_mode
                    else ""
                )
                + "This index proves immutable validation-only evaluation "
                "for every retained checkpoint from a locally bound "
                "STRICT_NONBLIND_V8 receipt. Train is bound only through "
                "the hashed v8 manifest and training receipt and is not "
                "opened by this evaluator. Calibration and blind content "
                "are neither opened nor hashed. No checkpoint is selected "
                "or authorized."
            ),
        }
        index_path = output / "evaluation_index.v8.json"
        eval_v6._atomic_json(index_path, index)
        return {
            "status": status,
            "stage": stage,
            "output_dir": str(output),
            "evaluation_index": str(index_path),
            "evaluation_index_sha256": eval_v6._sha256_file(index_path),
            "checkpoint_count": len(records),
            "examples_per_checkpoint": expected_examples,
            "selection_performed": False,
            "train_content_read": False,
            "calibration_content_read": False,
            "blind_test_content_read": False,
            "training_gate_bundle_sha256": strict_binding[
                "training_gate_bundle_sha256"
            ],
        }
    except BaseException as exc:
        failure = {
            "schema": FAILURE_SCHEMA,
            "orchestrator_version": ORCHESTRATOR_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "status": "FAILED_STRICT_NONBLIND_V8_EVALUATION_NO_SELECTION",
            "current_checkpoint": current_checkpoint,
            "completed_checkpoints": completed,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "evaluation_index_created": False,
            "selection_performed": False,
            "train_content_read": False,
            "train_content_hashed": False,
            "calibration_content_read": False,
            "calibration_content_hashed": False,
            "blind_test_content_read": False,
            "blind_test_content_hashed": False,
            "claim_boundary": (
                "Failure is retained without selecting or authorizing a "
                "checkpoint. No train, calibration, or blind split content "
                "was opened by the v8 evaluator."
            ),
        }
        failure_path = output / "failure_receipt.v8.json"
        try:
            eval_v6._atomic_json(failure_path, failure)
        except Exception:
            pass
        if isinstance(exc, PointerCheckpointEvalV8Error):
            raise
        raise PointerCheckpointEvalV8Error(str(exc)) from exc


__all__ = [
    "CANARY_STATUS",
    "FINAL_STATUS",
    "INDEX_SCHEMA",
    "ORCHESTRATOR_VERSION",
    "PointerCheckpointEvalV8Error",
    "run_checkpoint_evaluations_v8",
    "verify_strict_nonblind_v8_binding",
    "verify_v8_base_model_binding",
]
