"""Fail-closed v8c3 canary acceptance for the formal STRICT_NONBLIND_V8 asset.

The gate binds the completed canary training receipt to the formal v8
manifest, train split, validation split, training-gate bundle, and all six
checkpoint evaluations. It reuses the v6 evaluation protocol's independent
per-sample recomputation, but a v8c2/v6/v7 training receipt cannot authorize
the v8c3 infrastructure-recovery exact replay.

Only the explicitly named manifest, train, and validation files are opened.
Calibration and blind content are never discovered, opened, read, or hashed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from functools import cmp_to_key
from pathlib import Path
from typing import Any

from . import canary_acceptance_v6 as v6
from . import pointer_checkpoint_eval_v8 as eval_v8
from . import qlora_full_v6 as qlora

SCHEMA = "icmat_llm_canary_acceptance_receipt.v8"
VERSION = "icmat-llm-canary-acceptance-v8.3.0"
PASS_STATUS = "PASS_V8C3_CANARY_ACCEPTED_FOR_THREE_SEED_TRAINING_ONLY"
STOP_STATUS = "STOP_V8C3_CANARY_NOT_ACCEPTED"
ERROR_STATUS = "V8C3_CANARY_ACCEPTANCE_NOT_RECORDED"

FORMAL_MANIFEST_SHA256 = "7e2d9e2ab1bc380e1fb626e960a015b7c22c82b4c4c86d1f0c2c1e54b79c2535"
FORMAL_TRAIN_SHA256 = "674ea8cf77b2d61eac31a76d8b0c6af8178b0da93b0d6af6c7b3bb75d95a821c"
FORMAL_VALIDATION_SHA256 = "1ad3013670f90178e0372f1425b30cd867c38ef11551fc0cfb0f6c4e099becf4"
FORMAL_TRAINING_GATE_BUNDLE_SHA256 = "5d9f8e2b0a30a5a50c8ed484d7445eb34c6229f9bbce8231265fe9f6364c2b0a"
EXPECTED_CANARY_SEED = 20260728
V8C3_PROTOCOL_ID = "ICMAT-Pointer-v8c3-INFRA-RECOVERY-EXACT-REPLAY-r1"
V8C3_PREREGISTRATION_SHA256 = (
    "3c17a761b45fe14e4a5b48cb4eeb223a2d81b6dfad043a8060c0ca1ce7c06076"
)
V8C3_CONFIGURATION_SHA256 = (
    "252c79ba1482e03b8ca59e34adf91b5a80aac60f4976ba971bba489792e2f772"
)
V8C3_TRAINING_PROFILE = "V8C2_CAPACITY_REGULARIZED"
V8C2_CLOSURE_SHA256 = (
    "15619ca564a9a4b5589ed132e6d57ebaed35cec178d065a734b89034708b7067"
)
V8C2_CLOSURE_STATUS = "STOP_V8C2_INFRA_ABORTED_PRETRAIN_ATTEMPT_CONSUMED"
V8C1_STOP_ACCEPTANCE_SHA256 = (
    "ba02b07c95b70e3302dbe6a5431e4800c8ced8a5ddffff7242378da784107c5a"
)
V8C1_STOP_STATUS = "STOP_V8_CANARY_NOT_ACCEPTED"
V8C3_PREREGISTRATION_PATH = (
    qlora.WORKSPACE_ROOT
    / "docs"
    / "ai_brain_finals_20260728"
    / "ICMAT_POINTER_V8C3_INFRA_RECOVERY_PREREGISTRATION.json"
)
V8C2_CLOSURE_PATH = (
    qlora.WORKSPACE_ROOT
    / "docs"
    / "ai_brain_finals_20260728"
    / "ICMAT_POINTER_V8C2_INFRA_FAILURE_CLOSURE.json"
)
V8C1_STOP_ACCEPTANCE_PATH = (
    qlora.WORKSPACE_ROOT / "evaluation" / "icmat_foundry" / "llm" / "v8ca2.json"
)
V8C3_RUNTIME_QUALIFICATION_PATH = (
    qlora.WORKSPACE_ROOT
    / "evaluation"
    / "icmat_foundry"
    / "llm"
    / "v8c3.runtime_qualification.v1.json"
)
V8C3_CANARY_ATTEMPT_PATH = (
    qlora.WORKSPACE_ROOT
    / "evaluation"
    / "icmat_foundry"
    / "llm"
    / "v8c3.canary_attempt.v1.json"
)
V8C3_CANONICAL_TRAINING_RECEIPT_PATH = (
    qlora.WORKSPACE_ROOT
    / "evaluation"
    / "icmat_foundry"
    / "llm"
    / "v8c3"
    / "training_receipt.v6.json"
)
V8C3_CANONICAL_EVALUATION_INDEX_PATH = (
    qlora.WORKSPACE_ROOT
    / "evaluation"
    / "icmat_foundry"
    / "llm"
    / "v8c3e1"
    / "evaluation_index.v8.json"
)
V8C3_CANONICAL_ACCEPTANCE_PATH = (
    qlora.WORKSPACE_ROOT
    / "evaluation"
    / "icmat_foundry"
    / "llm"
    / "v8c3a1.json"
)

EXPECTED_TRAIN_EXAMPLES = 250
EXPECTED_VALIDATION_EXAMPLES = 150
EXPECTED_CHECKPOINTS = 6
EXPECTED_SAMPLES = 18
EXPECTED_EPOCHS = frozenset(range(1, 7))
V8_INDEX_FIELDS = frozenset(
    {
        "schema",
        "orchestrator_version",
        "created_at_utc",
        "status",
        "stage",
        "training",
        "strict_nonblind_v8_binding",
        "dataset",
        "base_model",
        "execution",
        "implementation",
        "checkpoints",
        "records",
        "selection",
        "authorization",
        "claim_boundary",
    }
)
V8_INDEX_DATASET_FIELDS = frozenset(
    {
        "directory",
        "path",
        "bytes",
        "sha256",
        "examples",
        "evaluation_directory",
        "evaluated_rows_per_checkpoint",
        "canary_selection",
        "opened_split",
        "train_content_read",
        "train_content_hashed",
        "calibration_content_read",
        "calibration_content_hashed",
        "blind_test_content_read",
        "blind_test_content_hashed",
    }
)
V8_INDEX_CHECKPOINT_FIELDS = frozenset(
    {
        "checkpoint_id",
        "seed",
        "epoch",
        "global_step",
        "validation_loss",
        "checkpoint_path",
        "receipt_relative_path",
        "training_checkpoint_tree_sha256",
        "training_adapter_tree_sha256",
        "evaluator_adapter_tree_sha256",
        "checkpoint_files",
        "checkpoint_bytes",
        "evaluation_directory",
        "evaluation_artifacts",
    }
)
V8_EVALUATION_ARTIFACTS = frozenset(
    {
        "run_receipt.v6.json",
        "sample_results.v6.jsonl",
        "summary.v6.json",
    }
)
V8_READABLE_METRIC_ARTIFACTS = frozenset({"sample_results.v6.jsonl", "summary.v6.json"})
RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "gate_version",
        "created_at_utc",
        "status",
        "gate_passed",
        "next_action",
        "formal_v8_binding",
        "input",
        "thresholds",
        "independent_recomputation",
        "deterministic_advancement_reference",
        "authorization",
        "claim_boundary",
        "receipt_payload_sha256",
    }
)
NORMALIZED_SNAPSHOT_FIELDS = frozenset(
    {
        "required_for_stage",
        "path",
        "bytes",
        "sha256",
        "stable_identity",
        "schema",
        "gate_version",
        "status",
        "gate_passed",
        "next_action",
        "receipt_payload_sha256",
        "authorization",
        "claim_boundary",
        "formal_v8_binding",
        "evaluation_index",
        "canary_training_receipt",
    }
)

CLAIM_BOUNDARY = (
    "This native-v8 receipt accepts only the preregistered v8c3 "
    "infrastructure-recovery exact replay. v8c2 remains permanently closed "
    "after a pretraining infrastructure abort and provides no model-quality "
    "evidence. This receipt binds the canonical v8c3 preregistration, runtime "
    "qualification, one-shot attempt ledger, training receipt, formal "
    "STRICT_NONBLIND_V8 data, and six independently recomputed validation "
    "checkpoint evaluations. A PASS may authorize only the preregistered "
    "final three-seed training. It does not select a final model or authorize "
    "calibration, blind evaluation, GGUF export, X5/BPU deployment, or "
    "production integration. Calibration and blind content were not "
    "discovered, opened, read, or hashed by this gate."
)


class CanaryAcceptanceV8Error(RuntimeError):
    """Raised when a trustworthy v8 canary decision cannot be recorded."""


def _raise(message: str) -> None:
    raise CanaryAcceptanceV8Error(message)


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise CanaryAcceptanceV8Error("value cannot be represented as finite canonical JSON") from exc


def _pretty_bytes(value: Mapping[str, Any]) -> bytes:
    try:
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
    except (TypeError, ValueError) as exc:
        raise CanaryAcceptanceV8Error("receipt cannot be represented as finite JSON") from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_canonical_acceptance_path(path: Path) -> Path:
    candidate = Path(path).resolve(strict=False)
    expected = V8C2_CANONICAL_ACCEPTANCE_PATH.resolve(strict=False)
    if candidate != expected:
        _raise(
            "v8c2 canary decision must use the canonical protocol receipt path"
        )
    return candidate


def _stable_bytes(path: Path, *, field: str) -> tuple[Path, bytes]:
    try:
        return v6._stable_bytes(Path(path), field=field)
    except v6.CanaryAcceptanceV6Error as exc:
        raise CanaryAcceptanceV8Error(str(exc)) from exc


def _load_json(path: Path, *, field: str) -> tuple[Path, bytes, dict[str, Any]]:
    try:
        return v6._load_json(Path(path), field=field)
    except v6.CanaryAcceptanceV6Error as exc:
        raise CanaryAcceptanceV8Error(str(exc)) from exc


def _stable_identity(path: Path, *, field: str) -> dict[str, int]:
    try:
        metadata = path.stat()
    except OSError as exc:
        raise CanaryAcceptanceV8Error(f"{field} stable identity could not be read: {exc}") from exc
    return {
        "device": int(metadata.st_dev),
        "file_id": int(metadata.st_ino),
        "size": int(metadata.st_size),
        "mtime_ns": int(metadata.st_mtime_ns),
        "ctime_ns": int(metadata.st_ctime_ns),
    }


def _immutable_snapshot(
    path: Path,
    *,
    field: str,
    expected_payload: bytes | None = None,
) -> dict[str, Any]:
    resolved, payload = _stable_bytes(path, field=field)
    identity = _stable_identity(resolved, field=field)
    checked_path, checked_payload = _stable_bytes(
        resolved,
        field=f"{field} final recheck",
    )
    checked_identity = _stable_identity(
        checked_path,
        field=f"{field} final recheck",
    )
    if (
        checked_path != resolved
        or checked_payload != payload
        or checked_identity != identity
        or (expected_payload is not None and payload != expected_payload)
    ):
        _raise(f"{field} changed during immutable snapshot")
    return {
        "path": str(resolved),
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
        "stable_identity": identity,
    }


def _utc_timestamp(value: Any, *, field: str) -> str:
    text = _text(value, field=field)
    try:
        timestamp = datetime.fromisoformat(text)
    except ValueError as exc:
        raise CanaryAcceptanceV8Error(f"{field} must be ISO-8601") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() != UTC.utcoffset(timestamp):
        _raise(f"{field} must be UTC")
    return text


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _raise(f"{field} must be an object")
    return value


def _sequence(value: Any, *, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _raise(f"{field} must be an array")
    return value


def _text(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(ord(character) < 32 for character in value)
    ):
        _raise(f"{field} must be a non-empty trimmed string")
    return value


def _sha256(value: Any, *, field: str) -> str:
    text = _text(value, field=field)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        _raise(f"{field} must be a lowercase SHA-256")
    return text


def _false_flags(
    mapping: Mapping[str, Any],
    names: Sequence[str],
    *,
    field: str,
) -> None:
    for name in names:
        if mapping.get(name) is not False:
            _raise(f"{field}.{name} must remain false")


def _formal_split(
    splits: Mapping[str, Any],
    name: str,
    *,
    expected_sha256: str,
    expected_examples: int,
) -> Mapping[str, Any]:
    split = _mapping(splits.get(name), field=f"dataset.splits.{name}")
    if (
        split.get("path") != f"{name}.jsonl"
        or split.get("sha256") != expected_sha256
        or split.get("examples") != expected_examples
    ):
        _raise(f"dataset.splits.{name} is not the formal v8 split")
    return split


def _validate_formal_manifest(
    manifest: Mapping[str, Any],
    *,
    train_bytes: int,
    validation_bytes: int,
) -> None:
    if (
        manifest.get("schema") != qlora.NONBLIND_V8_MANIFEST_SCHEMA
        or manifest.get("dataset_schema") != qlora.DATASET_SCHEMA
        or manifest.get("builder_version") != qlora.NONBLIND_V8_BUILDER_VERSION
        or manifest.get("status") != "NONBLIND_V8_BUILT_NLI_UNIQUE_SUPPORT_PREBLIND_COMMITTED"
    ):
        _raise("manifest identity is not the formal nonblind-v8 contract")
    splits = _mapping(manifest.get("splits"), field="manifest.splits")
    expected = {
        "train": (
            FORMAL_TRAIN_SHA256,
            EXPECTED_TRAIN_EXAMPLES,
            train_bytes,
        ),
        "validation": (
            FORMAL_VALIDATION_SHA256,
            EXPECTED_VALIDATION_EXAMPLES,
            validation_bytes,
        ),
    }
    for name, (sha256, count, byte_count) in expected.items():
        split = _mapping(splits.get(name), field=f"manifest.splits.{name}")
        if split != {
            "path": f"{name}.jsonl",
            "sha256": sha256,
            "count": count,
            "bytes": byte_count,
        }:
            _raise(f"manifest.splits.{name} binding mismatch")
    if manifest.get("training_boundary") != {
        "allowed_splits": ["train", "validation"],
        "calibration_content_for_training": False,
    }:
        _raise("manifest training boundary mismatch")
    sealed = _mapping(
        manifest.get("sealed_blind_access"),
        field="manifest.sealed_blind_access",
    )
    if sealed != {"hashed": False, "path_discovered": False, "read": False}:
        _raise("manifest sealed-blind boundary mismatch")


def _validate_authorization(receipt: Mapping[str, Any]) -> None:
    authorization = _mapping(receipt.get("authorization"), field="training.authorization")
    required = {
        "checkpoint_selected",
        "model_authorized",
        "calibration_authorized",
        "blind_test_authorized",
        "gguf_export_authorized",
        "deployment_authorized",
        "production_integration_authorized",
    }
    if not required.issubset(authorization):
        _raise("training authorization fields are incomplete")
    if any(not isinstance(value, bool) or value for value in authorization.values()):
        _raise("training receipt contains an authorization")
    selection = _mapping(receipt.get("selection"), field="training.selection")
    if (
        selection.get("automatic_selection_performed") is not False
        or selection.get("selected_seed") is not None
        or selection.get("selected_epoch") is not None
        or selection.get("selected_adapter") is not None
    ):
        _raise("canary training receipt already contains a selection")


def _validate_v8c2_training_protocol(receipt: Mapping[str, Any]) -> None:
    expected_protocol = qlora._v8c2_receipt_fields()
    if any(receipt.get(key) != value for key, value in expected_protocol.items()):
        _raise("training receipt is not bound to the frozen v8c2 protocol")

    requested = qlora.QLoRATrainingConfigV6(stage="canary")
    try:
        expected_configuration = qlora._configuration_payload(
            qlora._effective_training_config_v8c2(requested)
        )
    except qlora.QLoRAV6Error as exc:
        raise CanaryAcceptanceV8Error(
            "frozen v8c2 configuration is internally inconsistent"
        ) from exc
    configuration = _mapping(
        receipt.get("configuration"),
        field="training.configuration",
    )
    if dict(configuration) != expected_configuration:
        _raise("training configuration differs from the frozen v8c2 canary")
    if receipt.get("configuration_sha256") != qlora._canonical_sha256(
        expected_configuration
    ):
        _raise("training configuration digest differs from the frozen v8c2 canary")


def _validate_v8c2_canary_attempt(
    receipt: Mapping[str, Any],
    *,
    input_snapshot: Mapping[str, Any],
    dataset: Mapping[str, Any],
) -> None:
    attempt = _mapping(
        receipt.get("canary_attempt"),
        field="training.canary_attempt",
    )
    source_files = _mapping(
        input_snapshot.get("source_files"),
        field="training.input_snapshot.source_files",
    )
    base_model = _mapping(
        input_snapshot.get("base_model"),
        field="training.input_snapshot.base_model",
    )
    try:
        qlora._validate_v8c2_canary_attempt_receipt(
            attempt,
            run_id=_text(receipt.get("run_id"), field="training.run_id"),
            configuration_sha256=_sha256(
                receipt.get("configuration_sha256"),
                field="training.configuration_sha256",
            ),
            dataset_input_sha256=_sha256(
                dataset.get("inspected_input_sha256"),
                field="training dataset inspected_input_sha256",
            ),
            training_gate_bundle_sha256=_sha256(
                receipt.get("training_gate_bundle_sha256"),
                field="training.training_gate_bundle_sha256",
            ),
            source_inventory_sha256=qlora._canonical_sha256(source_files),
            base_model_tree_sha256=_sha256(
                base_model.get("tree_sha256"),
                field="training base-model tree_sha256",
            ),
        )
    except (qlora.QLoRAV6Error, OSError, ValueError) as exc:
        raise CanaryAcceptanceV8Error(
            f"v8c2 canary attempt binding rejected: {exc}"
        ) from exc


def _validate_data_access(receipt: Mapping[str, Any]) -> None:
    access = _mapping(receipt.get("data_access"), field="training.data_access")
    if access.get("train_content_read") is not True or access.get("validation_content_read") is not True:
        _raise("canary did not record train and validation access")
    _false_flags(
        access,
        (
            "calibration_content_read",
            "calibration_content_hashed",
            "blind_test_content_read",
            "blind_test_content_hashed",
            "calibration_content_loaded_for_training",
            "calibration_used_for_checkpoint_selection",
            "blind_materialized",
            "blind_discovered",
            "blind_path_constructed",
            "blind_filesystem_metadata_accessed",
            "blind_content_opened",
            "blind_content_read",
            "blind_content_hashed",
        ),
        field="training.data_access",
    )


def _training_checkpoint_bindings(
    receipt: Mapping[str, Any],
) -> dict[tuple[int, int], dict[str, Any]]:
    seeds = _sequence(receipt.get("seeds"), field="training.seeds")
    if len(seeds) != 1:
        _raise("v8 canary training receipt must contain exactly one seed")
    seed_receipt = _mapping(seeds[0], field="training.seeds[0]")
    seed = seed_receipt.get("seed")
    if seed != EXPECTED_CANARY_SEED:
        _raise("training.seeds[0].seed is not the preregistered v8c2 canary seed")
    if (
        seed_receipt.get("schema") != qlora.SEED_RECEIPT_SCHEMA
        or seed_receipt.get("trainer_version") != qlora.TRAINER_VERSION
        or seed_receipt.get("stage") != "canary"
        or seed_receipt.get("status") != "PASS_SEED_TRAINED_ALL_EPOCHS_NOT_SELECTED"
    ):
        _raise("v8 canary seed receipt identity mismatch")
    seed_authorization = _mapping(
        seed_receipt.get("authorization"),
        field="training.seeds[0].authorization",
    )
    if any(not isinstance(value, bool) or value for value in seed_authorization.values()):
        _raise("v8 canary seed receipt contains an authorization")
    checkpoints = _sequence(
        seed_receipt.get("epoch_checkpoints"),
        field="training.seeds[0].epoch_checkpoints",
    )
    if len(checkpoints) != EXPECTED_CHECKPOINTS:
        _raise("v8 canary seed must retain exactly six checkpoints")
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for position, raw in enumerate(checkpoints):
        checkpoint = _mapping(raw, field=f"training.seeds[0].epoch_checkpoints[{position}]")
        epoch = checkpoint.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch not in EXPECTED_EPOCHS:
            _raise("v8 canary checkpoint epoch is invalid")
        adapter = _mapping(
            checkpoint.get("adapter"),
            field=f"training checkpoint epoch {epoch}.adapter",
        )
        full = _mapping(
            checkpoint.get("checkpoint"),
            field=f"training checkpoint epoch {epoch}.checkpoint",
        )
        binding = {
            "seed": seed,
            "epoch": epoch,
            "global_step": checkpoint.get("global_step"),
            "relative_path": _text(
                checkpoint.get("path"),
                field=f"training checkpoint epoch {epoch}.path",
            ),
            "validation_loss": v6._parse_loss(
                checkpoint.get("validation_loss"),
                field=f"training checkpoint epoch {epoch}.validation_loss",
            ),
            "adapter_tree_sha256": _sha256(
                adapter.get("tree_sha256"),
                field=f"training checkpoint epoch {epoch}.adapter.tree_sha256",
            ),
            "checkpoint_tree_sha256": _sha256(
                full.get("tree_sha256"),
                field=(f"training checkpoint epoch {epoch}.checkpoint.tree_sha256"),
            ),
            "checkpoint_files": full.get("file_count"),
            "checkpoint_bytes": full.get("bytes"),
        }
        key = (seed, epoch)
        if key in result:
            _raise(f"duplicate v8 canary seed/epoch pair: {seed}/{epoch}")
        result[key] = binding
    if {epoch for _, epoch in result} != EXPECTED_EPOCHS:
        _raise("v8 canary checkpoints must cover epochs 1 through 6")
    return result


def _validate_training_receipt(
    receipt: Mapping[str, Any],
    *,
    receipt_path: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    receipt_path = _require_canonical_training_path(receipt_path)
    if (
        receipt.get("schema") != qlora.RUN_RECEIPT_SCHEMA
        or receipt.get("trainer_version") != qlora.TRAINER_VERSION
        or receipt.get("stage") != "canary"
        or receipt.get("status") != "PASS_CANARY_SINGLE_SEED_ALL_EPOCHS_NOT_SELECTED"
        or receipt.get("checkpoint_count") != EXPECTED_CHECKPOINTS
    ):
        _raise("training receipt is not a completed v8 1x6 canary")
    if receipt.get("training_gate_bundle_sha256") != FORMAL_TRAINING_GATE_BUNDLE_SHA256:
        _raise("training receipt gate bundle is not the formal v8 bundle")
    inspected = _sha256(
        receipt.get("v8_inspected_input_sha256"),
        field="training.v8_inspected_input_sha256",
    )
    _validate_authorization(receipt)
    _validate_data_access(receipt)

    snapshot = _mapping(receipt.get("input_snapshot"), field="training.input_snapshot")
    dataset = _mapping(snapshot.get("dataset"), field="training.input_snapshot.dataset")
    preregistration, runtime_qualification, authority_artifacts = (
        _validate_v8c3_training_protocol(receipt, dataset=dataset)
    )
    attempt_artifact = _validate_v8c3_canary_attempt(
        receipt,
        input_snapshot=snapshot,
        dataset=dataset,
        runtime_qualification=runtime_qualification,
    )
    authority_artifacts.append(attempt_artifact)
    if dataset.get("contract") != "STRICT_NONBLIND_V8":
        _raise("v7 or legacy receipt cannot authorize STRICT_NONBLIND_V8")
    if (
        dataset.get("training_gate_bundle_sha256") != FORMAL_TRAINING_GATE_BUNDLE_SHA256
        or dataset.get("v8_inspected_input_sha256") != inspected
        or dataset.get("inspected_input_sha256") != inspected
    ):
        _raise("training receipt v8 identity binding mismatch")
    bundle = _mapping(
        dataset.get("training_gate_bundle"),
        field="training.input_snapshot.dataset.training_gate_bundle",
    )
    expected_bundle_keys = {
        "contract",
        "nonblind_compare",
        "scoped_lexical",
        "unique_support",
        "nli_model",
        "training_gate_bundle_sha256",
    }
    bundle_core = {key: value for key, value in bundle.items() if key != "training_gate_bundle_sha256"}
    if (
        set(bundle) != expected_bundle_keys
        or bundle.get("contract") != "STRICT_NONBLIND_V8"
        or bundle.get("training_gate_bundle_sha256") != FORMAL_TRAINING_GATE_BUNDLE_SHA256
        or _sha256_bytes(_canonical_json(bundle_core).encode("utf-8")) != FORMAL_TRAINING_GATE_BUNDLE_SHA256
    ):
        _raise("embedded training-gate bundle binding mismatch")
    unique_support = _mapping(
        bundle.get("unique_support"),
        field=("training.input_snapshot.dataset.training_gate_bundle.unique_support"),
    )
    if set(unique_support) != {"train", "validation"}:
        _raise("training-gate bundle unique-support split mismatch")
    training_access = _mapping(
        dataset.get("training_data_access"),
        field="training.input_snapshot.dataset.training_data_access",
    )
    if training_access.get("opened_splits") != ["train", "validation"]:
        _raise("dataset snapshot opened-split boundary mismatch")
    _false_flags(
        training_access,
        (
            "calibration_content_loaded_for_training",
            "calibration_used_for_checkpoint_selection",
            "blind_materialized",
            "blind_discovered",
            "blind_path_constructed",
            "blind_filesystem_metadata_accessed",
            "blind_content_opened",
            "blind_content_read",
            "blind_content_hashed",
        ),
        field="training.input_snapshot.dataset.training_data_access",
    )
    manifest_binding = _mapping(
        dataset.get("manifest"),
        field="training.input_snapshot.dataset.manifest",
    )
    if (
        manifest_binding.get("path") != qlora.NONBLIND_V8_MANIFEST_NAME
        or manifest_binding.get("schema") != qlora.NONBLIND_V8_MANIFEST_SCHEMA
        or manifest_binding.get("dataset_schema") != qlora.DATASET_SCHEMA
        or manifest_binding.get("builder_version") != qlora.NONBLIND_V8_BUILDER_VERSION
        or manifest_binding.get("sha256") != FORMAL_MANIFEST_SHA256
    ):
        _raise("training receipt manifest is not the formal v8 manifest")
    splits = _mapping(
        dataset.get("splits"),
        field="training.input_snapshot.dataset.splits",
    )
    train = _formal_split(
        splits,
        "train",
        expected_sha256=FORMAL_TRAIN_SHA256,
        expected_examples=EXPECTED_TRAIN_EXAMPLES,
    )
    validation = _formal_split(
        splits,
        "validation",
        expected_sha256=FORMAL_VALIDATION_SHA256,
        expected_examples=EXPECTED_VALIDATION_EXAMPLES,
    )
    dataset_root = Path(_text(dataset.get("path"), field="training dataset path"))
    try:
        resolved_root = dataset_root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise CanaryAcceptanceV8Error(f"training dataset directory does not exist: {dataset_root}") from exc
    if not resolved_root.is_dir() or dataset_root.is_symlink():
        _raise("training dataset path must be a real directory")

    artifacts: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    for role, name, expected_hash, expected_bytes in (
        (
            "formal_manifest",
            qlora.NONBLIND_V8_MANIFEST_NAME,
            FORMAL_MANIFEST_SHA256,
            manifest_binding.get("bytes"),
        ),
        (
            "formal_train",
            "train.jsonl",
            FORMAL_TRAIN_SHA256,
            train.get("bytes"),
        ),
        (
            "formal_validation",
            "validation.jsonl",
            FORMAL_VALIDATION_SHA256,
            validation.get("bytes"),
        ),
    ):
        path, payload = _stable_bytes(resolved_root / name, field=f"v8 {role}")
        if _sha256_bytes(payload) != expected_hash or len(payload) != expected_bytes:
            _raise(f"{role} bytes or SHA-256 differ from formal v8 binding")
        payloads[role] = payload
        artifacts.append(
            {
                "role": role,
                "path": str(path),
                "bytes": len(payload),
                "sha256": expected_hash,
            }
        )
    try:
        manifest = v6._load_json_bytes(payloads["formal_manifest"], field="formal v8 manifest")
    except v6.CanaryAcceptanceV6Error as exc:
        raise CanaryAcceptanceV8Error(str(exc)) from exc
    _validate_formal_manifest(
        manifest,
        train_bytes=len(payloads["formal_train"]),
        validation_bytes=len(payloads["formal_validation"]),
    )
    declared_checkpoints = _training_checkpoint_bindings(receipt)
    try:
        strict_binding = eval_v8.verify_strict_nonblind_v8_binding(
            receipt=receipt,
            receipt_path=receipt_path,
            dataset_dir=resolved_root,
        )
        stage, specs = eval_v8.eval_v6._checkpoint_specs(
            receipt=receipt,
            training_root=receipt_path.parent,
        )
    except (
        eval_v8.PointerCheckpointEvalV8Error,
        eval_v8.eval_v6.PointerCheckpointEvalV6Error,
    ) as exc:
        raise CanaryAcceptanceV8Error(
            f"formal pointer_checkpoint_eval_v8 training binding rejected: {exc}"
        ) from exc
    if (
        stage != "canary"
        or len(specs) != EXPECTED_CHECKPOINTS
        or strict_binding.get("contract") != eval_v8.STRICT_CONTRACT
        or strict_binding.get("manifest", {}).get("sha256") != FORMAL_MANIFEST_SHA256
        or strict_binding.get("train", {}).get("sha256") != FORMAL_TRAIN_SHA256
        or strict_binding.get("validation", {}).get("sha256") != FORMAL_VALIDATION_SHA256
        or strict_binding.get("training_gate_bundle_sha256") != FORMAL_TRAINING_GATE_BUNDLE_SHA256
        or strict_binding.get("v8_inspected_input_sha256") != inspected
    ):
        _raise("formal pointer_checkpoint_eval_v8 generation binding mismatch")
    checkpoints: dict[tuple[int, int], dict[str, Any]] = {}
    for spec in specs:
        key = (int(spec["seed"]), int(spec["epoch"]))
        declared = declared_checkpoints.get(key)
        if declared is None:
            _raise("v8 evaluator checkpoint is absent from training receipt")
        normalized = {
            "checkpoint_id": str(spec["checkpoint_id"]),
            "seed": int(spec["seed"]),
            "epoch": int(spec["epoch"]),
            "global_step": spec["global_step"],
            "relative_path": str(spec["receipt_path"]),
            "path": str(spec["path"]),
            "validation_loss": v6._parse_loss(
                spec["validation_loss"],
                field=f"{spec['checkpoint_id']}.validation_loss",
            ),
            "adapter_tree_sha256": str(spec["training_adapter_tree_sha256"]),
            "checkpoint_tree_sha256": str(spec["training_checkpoint_tree_sha256"]),
            "evaluator_adapter_tree_sha256": str(spec["evaluator_adapter_tree_sha256"]),
            "checkpoint_files": spec["checkpoint_files"],
            "checkpoint_bytes": spec["checkpoint_bytes"],
        }
        for name in (
            "seed",
            "epoch",
            "global_step",
            "relative_path",
            "validation_loss",
            "adapter_tree_sha256",
            "checkpoint_tree_sha256",
            "checkpoint_files",
            "checkpoint_bytes",
        ):
            if declared[name] != normalized[name]:
                _raise(f"{spec['checkpoint_id']} evaluator/training {name} mismatch")
        checkpoints[key] = normalized
    if set(checkpoints) != set(declared_checkpoints):
        _raise("v8 evaluator and training checkpoint populations differ")
    return (
        {
            "receipt_path": str(receipt_path),
            "run_id": _text(receipt.get("run_id"), field="training.run_id"),
            "dataset_path": str(resolved_root),
            "manifest_sha256": FORMAL_MANIFEST_SHA256,
            "train_sha256": FORMAL_TRAIN_SHA256,
            "validation_sha256": FORMAL_VALIDATION_SHA256,
            "validation_bytes": len(payloads["formal_validation"]),
            "training_gate_bundle_sha256": (FORMAL_TRAINING_GATE_BUNDLE_SHA256),
            "v8_inspected_input_sha256": inspected,
            "strict_nonblind_v8_binding": strict_binding,
            "checkpoints": checkpoints,
            "protocol_id": V8C3_PROTOCOL_ID,
            "training_profile": V8C3_TRAINING_PROFILE,
            "configuration_sha256": V8C3_CONFIGURATION_SHA256,
            "preregistration_sha256": preregistration["sha256"],
            "runtime_qualification_sha256": runtime_qualification["sha256"],
            "v8c3_attempt_sha256": attempt_artifact["sha256"],
            "v8c2_closure_sha256": V8C2_CLOSURE_SHA256,
            "v8c1_stop_acceptance_sha256": V8C1_STOP_ACCEPTANCE_SHA256,
            "runtime_qualification": runtime_qualification,
        },
        artifacts,
        authority_artifacts,
    )


def _validate_index_training_binding(
    index: Mapping[str, Any],
    *,
    training: Mapping[str, Any],
    training_path: Path,
    training_sha256: str,
) -> None:
    binding = _mapping(index.get("training"), field="index.training")
    if set(binding) != {
        "receipt_path",
        "receipt_sha256",
        "run_id",
        "checkpoint_count",
        "contract",
        "training_gate_bundle_sha256",
        "v8_inspected_input_sha256",
    }:
        _raise("formal v8 index training fields mismatch")
    if (
        Path(str(binding.get("receipt_path"))).resolve(strict=False) != training_path
        or binding.get("receipt_sha256") != training_sha256
        or binding.get("run_id") != training["run_id"]
        or binding.get("checkpoint_count") != EXPECTED_CHECKPOINTS
        or binding.get("contract") != eval_v8.STRICT_CONTRACT
        or binding.get("training_gate_bundle_sha256") != training["training_gate_bundle_sha256"]
        or binding.get("v8_inspected_input_sha256") != training["v8_inspected_input_sha256"]
        or index.get("strict_nonblind_v8_binding") != training["strict_nonblind_v8_binding"]
    ):
        _raise("evaluation index and v8 training receipt binding mismatch")
    dataset = _mapping(index.get("dataset"), field="index.dataset")
    if set(dataset) != V8_INDEX_DATASET_FIELDS:
        _raise("formal v8 index dataset fields mismatch")
    expected_validation_path = str(Path(str(training["dataset_path"])) / "validation.jsonl")
    if (
        dataset.get("directory") != training["dataset_path"]
        or dataset.get("path") != expected_validation_path
        or dataset.get("bytes") != training["validation_bytes"]
        or dataset.get("sha256") != FORMAL_VALIDATION_SHA256
        or dataset.get("examples") != EXPECTED_VALIDATION_EXAMPLES
        or dataset.get("evaluated_rows_per_checkpoint") != EXPECTED_SAMPLES
        or dataset.get("opened_split") != "validation"
        or any(
            dataset.get(field) is not False
            for field in (
                "train_content_read",
                "train_content_hashed",
                "calibration_content_read",
                "calibration_content_hashed",
                "blind_test_content_read",
                "blind_test_content_hashed",
            )
        )
    ):
        _raise("evaluation index is not bound to formal v8 validation")
    selection = _mapping(dataset.get("canary_selection"), field="index.dataset.canary_selection")
    source = _mapping(
        selection.get("source_validation"),
        field="index.dataset.canary_selection.source_validation",
    )
    if (
        source.get("path") != expected_validation_path
        or source.get("rows") != EXPECTED_VALIDATION_EXAMPLES
        or source.get("sha256") != FORMAL_VALIDATION_SHA256
    ):
        _raise("canary selection is not bound to formal v8 validation")


def _validate_v8_implementation(
    index: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = _mapping(index.get("implementation"), field="index.implementation")
    expected_paths = {
        "orchestrator_v8": Path(eval_v8.__file__),
        "evaluation_engine_v6": Path(eval_v8.eval_v6.__file__),
        "pointer_evaluator": Path(eval_v8.pointer_hf_eval_v6.__file__),
        "pointer_compiler": Path(eval_v8.evidence_pointer_v6.__file__),
        "selection_policy": Path(eval_v8.selection_policy_v6.__file__),
        "runner_v8": Path(eval_v8._PRODUCTION_RUNNER_PATH),
    }
    if set(records) != set(expected_paths):
        _raise("formal v8 index implementation roles mismatch")
    artifacts: list[dict[str, Any]] = []
    for role, expected_raw in expected_paths.items():
        record = _mapping(records.get(role), field=f"index.implementation.{role}")
        if set(record) != {"path", "sha256"}:
            _raise(f"formal v8 implementation {role} fields mismatch")
        expected_path, payload = _stable_bytes(
            expected_raw.resolve(strict=True),
            field=f"formal v8 implementation {role}",
        )
        if Path(str(record.get("path"))).resolve(strict=True) != expected_path or record.get(
            "sha256"
        ) != _sha256_bytes(payload):
            _raise(f"formal v8 implementation {role} mismatch")
        artifacts.append(
            {
                "role": f"implementation:{role}",
                "path": str(expected_path),
                "bytes": len(payload),
                "sha256": _sha256_bytes(payload),
            }
        )
    return artifacts


def _validate_index_boundary_v8(
    index: Mapping[str, Any],
    *,
    training: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if set(index) != V8_INDEX_FIELDS:
        _raise("formal v8 evaluation index fields mismatch")
    if (
        index.get("schema") != eval_v8.INDEX_SCHEMA
        or index.get("orchestrator_version") != eval_v8.ORCHESTRATOR_VERSION
        or index.get("status") != eval_v8.CANARY_STATUS
        or index.get("stage") != "canary"
    ):
        _raise("evaluation index is not a formal pointer_checkpoint_eval_v8 canary receipt")
    created = _text(index.get("created_at_utc"), field="index.created_at_utc")
    try:
        timestamp = datetime.fromisoformat(created)
    except ValueError as exc:
        raise CanaryAcceptanceV8Error("index.created_at_utc must be ISO-8601") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() != UTC.utcoffset(timestamp):
        _raise("index.created_at_utc must be UTC")
    if index.get("strict_nonblind_v8_binding") != training["strict_nonblind_v8_binding"]:
        _raise("formal v8 index generation binding differs from training")
    execution = _mapping(index.get("execution"), field="index.execution")
    if set(execution) != {
        "backend",
        "runner_mode",
        "device",
        "seed",
        "split",
        "max_samples",
        "checkpoint_outputs_immutable",
        "per_sample_metrics_recomputed",
        "summary_metrics_trusted",
        "selection_policy_invoked",
        "checkpoint_selected",
        "freeze_created",
    }:
        _raise("formal v8 index execution fields mismatch")
    if (
        execution.get("backend") != "hf_model"
        or execution.get("runner_mode") != "production_fixed_v8"
        or execution.get("device") not in {"cpu", "cuda"}
        or isinstance(execution.get("seed"), bool)
        or not isinstance(execution.get("seed"), int)
        or execution.get("split") != "validation"
        or execution.get("max_samples") is not None
        or execution.get("checkpoint_outputs_immutable") is not True
        or execution.get("per_sample_metrics_recomputed") is not True
        or execution.get("summary_metrics_trusted") is not False
        or execution.get("selection_policy_invoked") is not False
        or execution.get("checkpoint_selected") is not False
        or execution.get("freeze_created") is not False
    ):
        _raise("formal v8 evaluation execution boundary mismatch")
    selection = _mapping(index.get("selection"), field="index.selection")
    if (
        set(selection) != {"performed", "selected_checkpoint_id", "required_next_step"}
        or selection.get("performed") is not False
        or selection.get("selected_checkpoint_id") is not None
    ):
        _raise("formal v8 canary index already contains a selection")
    authorization = _mapping(index.get("authorization"), field="index.authorization")
    if set(authorization) != {
        "checkpoint_selected",
        "model_authorized",
        "calibration_authorized",
        "blind_test_authorized",
        "gguf_export_authorized",
        "deployment_authorized",
        "production_integration_authorized",
    } or any(not isinstance(value, bool) or value for value in authorization.values()):
        _raise("formal v8 canary index contains an authorization")
    base = _mapping(index.get("base_model"), field="index.base_model")
    if set(base) != {
        "directory",
        "training_tree_sha256",
        "evaluator_tree_sha256",
        "file_count",
        "bytes",
        "receipt_core_exact",
        "runtime_loading",
    }:
        _raise("formal v8 index base-model fields mismatch")
    training_tree_sha256 = _sha256(
        base.get("training_tree_sha256"),
        field="index.base_model.training_tree_sha256",
    )
    _sha256(
        base.get("evaluator_tree_sha256"),
        field="index.base_model.evaluator_tree_sha256",
    )
    if (
        isinstance(base.get("file_count"), bool)
        or not isinstance(base.get("file_count"), int)
        or base["file_count"] < 1
        or isinstance(base.get("bytes"), bool)
        or not isinstance(base.get("bytes"), int)
        or base["bytes"] < 1
        or base.get("receipt_core_exact") is not True
    ):
        _raise("formal v8 index base-model size binding is invalid")
    runtime_loading = _mapping(
        base.get("runtime_loading"),
        field="index.base_model.runtime_loading",
    )
    expected_runtime_loading = {
        "policy": eval_v8.RUNTIME_LOADING_POLICY,
        "content_address": f"sha256:{training_tree_sha256}",
        "tree_sha256": training_tree_sha256,
        "file_count": base["file_count"],
        "bytes": base["bytes"],
        "verified_before_cuda": True,
        "loaded_only_from_snapshot": True,
        "removed_before_publish": True,
    }
    if dict(runtime_loading) != expected_runtime_loading:
        _raise("formal v8 index runtime-loading binding mismatch")
    _text(index.get("claim_boundary"), field="index.claim_boundary")
    return _validate_v8_implementation(index)


def _validate_checkpoint_training_binding(
    checkpoint: Mapping[str, Any],
    *,
    training_checkpoints: Mapping[tuple[int, int], Mapping[str, Any]],
    checkpoint_id: str,
) -> None:
    seed = checkpoint.get("seed")
    epoch = checkpoint.get("epoch")
    key = (seed, epoch)
    if key not in training_checkpoints:
        _raise(f"{checkpoint_id} is absent from the v8 training receipt")
    expected = training_checkpoints[key]
    try:
        validation_loss = v6._parse_loss(
            checkpoint.get("validation_loss"),
            field=f"{checkpoint_id}.validation_loss",
        )
    except v6.CanaryAcceptanceV6Error as exc:
        raise CanaryAcceptanceV8Error(str(exc)) from exc
    checks = (
        (checkpoint_id, expected["checkpoint_id"]),
        (checkpoint.get("checkpoint_path"), expected["path"]),
        (
            checkpoint.get("training_adapter_tree_sha256"),
            expected["adapter_tree_sha256"],
        ),
        (
            checkpoint.get("training_checkpoint_tree_sha256"),
            expected["checkpoint_tree_sha256"],
        ),
        (
            checkpoint.get("evaluator_adapter_tree_sha256"),
            expected["evaluator_adapter_tree_sha256"],
        ),
        (checkpoint.get("global_step"), expected["global_step"]),
        (
            checkpoint.get("receipt_relative_path"),
            expected["relative_path"],
        ),
        (checkpoint.get("checkpoint_files"), expected["checkpoint_files"]),
        (checkpoint.get("checkpoint_bytes"), expected["checkpoint_bytes"]),
        (validation_loss, expected["validation_loss"]),
    )
    if any(actual != wanted for actual, wanted in checks):
        _raise(f"{checkpoint_id} differs from its v8 training checkpoint")


def _evaluation_candidates(
    *,
    index_path: Path,
    index_payload: bytes,
    index: Mapping[str, Any],
    training: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    authority_artifacts = _validate_index_boundary_v8(
        index,
        training=training,
    )
    training_checkpoints = training["checkpoints"]
    root = index_path.parent.resolve(strict=True)
    checkpoint_items = _sequence(index.get("checkpoints"), field="index.checkpoints")
    record_items = _sequence(index.get("records"), field="index.records")
    if len(checkpoint_items) != EXPECTED_CHECKPOINTS or len(record_items) != EXPECTED_CHECKPOINTS:
        _raise("index must contain exactly six checkpoints and six records")

    checkpoints: dict[str, Mapping[str, Any]] = {}
    records: dict[str, Mapping[str, Any]] = {}
    for raw in checkpoint_items:
        checkpoint = _mapping(raw, field="index.checkpoints[]")
        if set(checkpoint) != V8_INDEX_CHECKPOINT_FIELDS:
            _raise("formal v8 index checkpoint fields mismatch")
        checkpoint_id = _text(
            checkpoint.get("checkpoint_id"),
            field="checkpoint.checkpoint_id",
        )
        if checkpoint_id in checkpoints:
            _raise(f"duplicate checkpoint_id: {checkpoint_id}")
        checkpoints[checkpoint_id] = checkpoint
    for raw in record_items:
        record = _mapping(raw, field="index.records[]")
        checkpoint_id = _text(record.get("checkpoint_id"), field="record.checkpoint_id")
        if checkpoint_id in records:
            _raise(f"duplicate record checkpoint_id: {checkpoint_id}")
        records[checkpoint_id] = record
    if set(checkpoints) != set(records):
        _raise("checkpoint evidence and metric record IDs differ")

    artifacts: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    reference_ids: list[str] | None = None
    observed_directories: set[Path] = set()
    observed_pairs: set[tuple[int, int]] = set()
    observed_seeds: set[int] = set()
    for checkpoint_id in sorted(checkpoints):
        checkpoint = checkpoints[checkpoint_id]
        record = records[checkpoint_id]
        try:
            seed = v6._require_int(
                checkpoint.get("seed"),
                field=f"{checkpoint_id}.seed",
                minimum=1,
            )
            epoch = v6._require_int(
                checkpoint.get("epoch"),
                field=f"{checkpoint_id}.epoch",
                minimum=1,
                maximum=6,
            )
        except v6.CanaryAcceptanceV6Error as exc:
            raise CanaryAcceptanceV8Error(str(exc)) from exc
        if (
            record.get("seed") != seed
            or record.get("epoch") != epoch
            or record.get("checkpoint_id") != checkpoint_id
        ):
            _raise(f"{checkpoint_id} record identity mismatch")
        pair = (seed, epoch)
        if pair in observed_pairs:
            _raise(f"duplicate canary seed/epoch pair: {seed}/{epoch}")
        observed_pairs.add(pair)
        observed_seeds.add(seed)
        _validate_checkpoint_training_binding(
            checkpoint,
            training_checkpoints=training_checkpoints,
            checkpoint_id=checkpoint_id,
        )
        try:
            validation_loss = v6._parse_loss(
                checkpoint.get("validation_loss"),
                field=f"{checkpoint_id}.validation_loss",
            )
            if (
                v6._parse_loss(
                    record.get("validation_loss"),
                    field=f"{checkpoint_id}.record.validation_loss",
                )
                != validation_loss
            ):
                _raise(f"{checkpoint_id} validation loss binding differs")
            directory = v6._resolve_child_directory(
                root,
                checkpoint.get("evaluation_directory"),
                checkpoint_id=checkpoint_id,
            )
        except v6.CanaryAcceptanceV6Error as exc:
            raise CanaryAcceptanceV8Error(str(exc)) from exc
        if directory in observed_directories:
            _raise(f"multiple checkpoints share {directory}")
        observed_directories.add(directory)
        hashes = _mapping(
            checkpoint.get("evaluation_artifacts"),
            field=f"{checkpoint_id}.evaluation_artifacts",
        )
        if set(hashes) != V8_EVALUATION_ARTIFACTS:
            _raise(f"{checkpoint_id} v8 evaluation artifact inventory mismatch")
        for name in V8_EVALUATION_ARTIFACTS:
            _sha256(
                hashes.get(name),
                field=f"{checkpoint_id}.evaluation_artifacts.{name}",
            )
        payloads: dict[str, bytes] = {}
        for name in sorted(V8_READABLE_METRIC_ARTIFACTS):
            expected_hash = _sha256(
                hashes.get(name),
                field=f"{checkpoint_id}.evaluation_artifacts.{name}",
            )
            path, payload = _stable_bytes(directory / name, field=f"{checkpoint_id} {name}")
            actual_hash = _sha256_bytes(payload)
            if actual_hash != expected_hash:
                _raise(f"{checkpoint_id} {name} SHA-256 mismatch")
            payloads[name] = payload
            artifacts.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "role": f"{checkpoint_id}:{name}",
                    "path": str(path),
                    "bytes": len(payload),
                    "sha256": actual_hash,
                }
            )
        try:
            rows = v6._load_jsonl_bytes(
                payloads["sample_results.v6.jsonl"],
                field=f"{checkpoint_id} sample results",
            )
            metrics, audit = v6._recompute_checkpoint(rows, checkpoint_id=checkpoint_id)
            summary = v6._load_json_bytes(
                payloads["summary.v6.json"],
                field=f"{checkpoint_id} summary",
            )
            v6._validate_summary(
                summary,
                metrics=metrics,
                audit=audit,
                checkpoint_id=checkpoint_id,
            )
            failed = v6._failed_gates(metrics)
        except v6.CanaryAcceptanceV6Error as exc:
            raise CanaryAcceptanceV8Error(str(exc)) from exc
        current_ids = list(audit["example_ids"])
        if reference_ids is None:
            reference_ids = current_ids
        elif current_ids != reference_ids:
            _raise("all checkpoints must evaluate identical 18 example IDs")
        index_metrics = _mapping(record.get("metrics"), field=f"{checkpoint_id}.record.metrics")
        if dict(index_metrics) != metrics:
            _raise(f"{checkpoint_id} index metrics recomputation mismatch")
        candidates.append(
            {
                "checkpoint_id": checkpoint_id,
                "seed": seed,
                "epoch": epoch,
                "validation_loss": str(validation_loss),
                "qualified": not failed,
                "failed_gates": failed,
                "metrics": metrics,
                "ranking_metrics": {
                    "minimum_stratified_strict": v6._minimum_stratum(metrics),
                    "compiled_strict_exact": metrics["compiled_strict_exact"],
                    "answer_span_exact": metrics["answer_span_exact"],
                    "refuse_f1": v6._refuse_f1(metrics),
                    "validation_loss": str(validation_loss),
                    "epoch": epoch,
                    "seed": seed,
                },
            }
        )
    if (
        len(observed_seeds) != 1
        or {epoch for _, epoch in observed_pairs} != EXPECTED_EPOCHS
        or observed_pairs != set(training_checkpoints)
    ):
        _raise("evaluation population differs from the v8 1x6 canary")
    _revalidate(
        [
            {
                "path": str(index_path),
                "sha256": _sha256_bytes(index_payload),
                "role": "evaluation_index",
            },
            *artifacts,
            *authority_artifacts,
        ]
    )
    return candidates, artifacts, authority_artifacts


def _revalidate(artifacts: Sequence[Mapping[str, Any]]) -> None:
    for artifact in artifacts:
        path = Path(str(artifact["path"]))
        _, payload = _stable_bytes(path, field=str(artifact["role"]))
        if _sha256_bytes(payload) != artifact["sha256"]:
            _raise(f"{artifact['role']} changed during v8 acceptance")


def _build_canary_acceptance_v8(
    *,
    evaluation_index_path: Path,
    canary_training_receipt_path: Path,
    created_at_utc: str,
) -> dict[str, Any]:
    created_at_utc = _utc_timestamp(
        created_at_utc,
        field="acceptance.created_at_utc",
    )
    canonical_index_path = _require_canonical_evaluation_path(
        evaluation_index_path
    )
    index_path, index_payload, index = _load_json(
        canonical_index_path,
        field="evaluation index",
    )
    if index_path.name != "evaluation_index.v8.json":
        _raise("evaluation index filename must be evaluation_index.v8.json")
    if (
        index.get("schema") != eval_v8.INDEX_SCHEMA
        or index.get("orchestrator_version") != eval_v8.ORCHESTRATOR_VERSION
        or index.get("status") != eval_v8.CANARY_STATUS
        or index.get("stage") != "canary"
    ):
        _raise("evaluation index is not a formal pointer_checkpoint_eval_v8 canary receipt")
    canonical_training_path = _require_canonical_training_path(
        canary_training_receipt_path
    )
    training_path, training_payload, training_receipt = _load_json(
        canonical_training_path,
        field="canary training receipt",
    )
    if training_path.name != "training_receipt.v6.json":
        _raise("training receipt filename must be training_receipt.v6.json")
    training, dataset_artifacts, training_authority_artifacts = (
        _validate_training_receipt(
            training_receipt,
            receipt_path=training_path,
        )
    )
    training_sha256 = _sha256_bytes(training_payload)
    _validate_index_training_binding(
        index,
        training=training,
        training_path=training_path,
        training_sha256=training_sha256,
    )
    (
        candidates,
        checkpoint_artifacts,
        evaluation_authority_artifacts,
    ) = _evaluation_candidates(
        index_path=index_path,
        index_payload=index_payload,
        index=index,
        training=training,
    )
    acceptance_source_path, acceptance_source_payload = _stable_bytes(
        Path(__file__),
        field="canary_acceptance_v8 source",
    )
    acceptance_source_artifact = {
        "role": "canary_acceptance_v8_source",
        "path": str(acceptance_source_path),
        "sha256": _sha256_bytes(acceptance_source_payload),
    }
    all_inputs = [
        {
            "role": "canary_training_receipt",
            "path": str(training_path),
            "sha256": training_sha256,
        },
        *dataset_artifacts,
        *training_authority_artifacts,
        *checkpoint_artifacts,
        *evaluation_authority_artifacts,
        {
            "role": "evaluation_index",
            "path": str(index_path),
            "sha256": _sha256_bytes(index_payload),
        },
        acceptance_source_artifact,
    ]
    _revalidate(all_inputs)

    qualified = [candidate for candidate in candidates if candidate["qualified"]]
    ordered = sorted(qualified, key=cmp_to_key(v6._compare_candidates))
    reference = ordered[0] if ordered else None
    gate_passed = reference is not None
    status = PASS_STATUS if gate_passed else STOP_STATUS
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "gate_version": VERSION,
        "created_at_utc": created_at_utc,
        "status": status,
        "gate_passed": gate_passed,
        "next_action": (
            "START_FINAL_THREE_SEED_TRAINING" if gate_passed else "STOP_AND_REVIEW_V8_NONBLIND_CANARY"
        ),
        "formal_v8_binding": {
            "contract": "STRICT_NONBLIND_V8",
            "recovery_protocol_id": V8C3_PROTOCOL_ID,
            "training_profile": V8C3_TRAINING_PROFILE,
            "configuration_sha256": V8C3_CONFIGURATION_SHA256,
            "v8c3_preregistration_sha256": training[
                "preregistration_sha256"
            ],
            "runtime_qualification_sha256": training[
                "runtime_qualification_sha256"
            ],
            "v8c3_attempt_sha256": training["v8c3_attempt_sha256"],
            "v8c2_closure_sha256": training["v8c2_closure_sha256"],
            "v8c1_stop_acceptance_sha256": training[
                "v8c1_stop_acceptance_sha256"
            ],
            "manifest_sha256": FORMAL_MANIFEST_SHA256,
            "train_sha256": FORMAL_TRAIN_SHA256,
            "validation_sha256": FORMAL_VALIDATION_SHA256,
            "training_gate_bundle_sha256": (FORMAL_TRAINING_GATE_BUNDLE_SHA256),
            "v8_inspected_input_sha256": training["v8_inspected_input_sha256"],
        },
        "input": {
            "evaluation_index": {
                "path": str(index_path),
                "bytes": len(index_payload),
                "sha256": _sha256_bytes(index_payload),
            },
            "canary_training_receipt": {
                "path": str(training_path),
                "bytes": len(training_payload),
                "sha256": training_sha256,
                "run_id": training["run_id"],
            },
            "dataset_artifacts_read": dataset_artifacts,
            "protocol_authority_artifacts_read": (
                training_authority_artifacts
            ),
            "checkpoint_artifacts_read": checkpoint_artifacts,
            "checkpoint_run_receipts_read": False,
            "calibration_content_discovered": False,
            "calibration_content_read": False,
            "calibration_content_hashed": False,
            "blind_test_content_discovered": False,
            "blind_test_content_read": False,
            "blind_test_content_hashed": False,
        },
        "thresholds": {
            "completed_samples": EXPECTED_SAMPLES,
            "pointer_schema_rate": "1/1",
            "pointer_invalid_count_max": 0,
            "pointer_ambiguous_count_max": 0,
            "pointer_out_of_range_count_max": 0,
            "unsupported_wrong_answer_count_max": 0,
            "compiled_schema_rate": "1/1",
            "compiled_citation_exact_rate": "1/1",
            "compiled_provenance_exact_rate": "1/1",
            "answer_span_exact_min": "9/10",
            "refuse_f1_min": "9/10",
        },
        "independent_recomputation": {
            "checkpoint_count": len(candidates),
            "samples_per_checkpoint": EXPECTED_SAMPLES,
            "summary_metrics_trusted": False,
            "index_metrics_trusted": False,
            "all_index_and_summary_metrics_reconciled": True,
            "checkpoints": candidates,
        },
        "deterministic_advancement_reference": (
            None
            if reference is None
            else {
                "checkpoint_id": reference["checkpoint_id"],
                "seed": reference["seed"],
                "epoch": reference["epoch"],
                "ranking_metrics": reference["ranking_metrics"],
                "purpose": ("V8_THREE_SEED_TRAINING_ADVANCEMENT_EVIDENCE_ONLY"),
                "is_final_model_selection": False,
            }
        ),
        "authorization": {
            "three_seed_training_authorized": gate_passed,
            "checkpoint_selected_as_final_model": False,
            "model_authorized": False,
            "calibration_authorized": False,
            "blind_test_authorized": False,
            "gguf_export_authorized": False,
            "x5_deployment_authorized": False,
            "production_integration_authorized": False,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    receipt["receipt_payload_sha256"] = _sha256_bytes(_canonical_json(receipt).encode("utf-8"))
    return {
        "receipt": receipt,
        "index_path": index_path,
        "index_payload": index_payload,
        "training_path": training_path,
        "training_payload": training_payload,
        "all_inputs": all_inputs,
        "acceptance_source_path": acceptance_source_path,
        "acceptance_source_payload": acceptance_source_payload,
        "reference_checkpoint_id": (None if reference is None else reference["checkpoint_id"]),
    }


def _record_canary_acceptance_v8(
    *,
    evaluation_index_path: Path,
    canary_training_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    canonical_output = _require_canonical_acceptance_path(output_path)
    built = _build_canary_acceptance_v8(
        evaluation_index_path=evaluation_index_path,
        canary_training_receipt_path=canary_training_receipt_path,
        created_at_utc=datetime.now(UTC).isoformat(),
    )
    receipt = built["receipt"]
    payload = _pretty_bytes(receipt)
    try:
        written = v6._write_exclusive(canonical_output, payload)
    except v6.CanaryAcceptanceV6Error as exc:
        raise CanaryAcceptanceV8Error(str(exc)) from exc
    _revalidate(built["all_inputs"])
    return {
        "status": receipt["status"],
        "gate_passed": receipt["gate_passed"],
        "path": str(written),
        "sha256": _sha256_bytes(payload),
        "canonical_digest_sha256": receipt["receipt_payload_sha256"],
        "advancement_reference_checkpoint_id": built["reference_checkpoint_id"],
        "three_seed_training_authorized": receipt["authorization"]["three_seed_training_authorized"],
        "final_model_selected": False,
        "deployment_authorized": False,
    }


def _verify_canary_acceptance_v8(
    *,
    acceptance_receipt_path: Path,
    evaluation_index_path: Path,
    training_receipt_path: Path,
) -> dict[str, Any]:
    _require_canonical_acceptance_path(acceptance_receipt_path)
    acceptance_path, acceptance_payload, acceptance = _load_json(
        acceptance_receipt_path,
        field="v8 canary acceptance receipt",
    )
    if set(acceptance) != RECEIPT_FIELDS:
        _raise("v8 canary acceptance receipt fields mismatch")
    if acceptance.get("schema") != SCHEMA or acceptance.get("gate_version") != VERSION:
        _raise("v6/legacy canary acceptance cannot authorize strict v8 final")
    created_at_utc = _utc_timestamp(
        acceptance.get("created_at_utc"),
        field="acceptance.created_at_utc",
    )
    receipt_body = dict(acceptance)
    receipt_digest = _sha256(
        receipt_body.pop("receipt_payload_sha256", None),
        field="acceptance.receipt_payload_sha256",
    )
    if receipt_digest != _sha256_bytes(_canonical_json(receipt_body).encode("utf-8")):
        _raise("v8 canary acceptance canonical digest mismatch")
    if acceptance_payload != _pretty_bytes(acceptance):
        _raise("v8 canary acceptance is not in canonical record encoding")

    acceptance_before = _immutable_snapshot(
        acceptance_path,
        field="v8 canary acceptance receipt",
        expected_payload=acceptance_payload,
    )
    index_before = _immutable_snapshot(
        evaluation_index_path,
        field="formal v8 canary evaluation index",
    )
    training_before = _immutable_snapshot(
        training_receipt_path,
        field="formal v8 canary training receipt",
    )
    source_before = _immutable_snapshot(
        Path(__file__),
        field="canary_acceptance_v8 source",
    )

    built = _build_canary_acceptance_v8(
        evaluation_index_path=evaluation_index_path,
        canary_training_receipt_path=training_receipt_path,
        created_at_utc=created_at_utc,
    )
    if acceptance != built["receipt"]:
        _raise("v8 canary acceptance differs from independent full replay")

    acceptance_after = _immutable_snapshot(
        acceptance_path,
        field="v8 canary acceptance receipt final recheck",
        expected_payload=acceptance_payload,
    )
    index_after = _immutable_snapshot(
        built["index_path"],
        field="formal v8 canary evaluation index final recheck",
        expected_payload=built["index_payload"],
    )
    training_after = _immutable_snapshot(
        built["training_path"],
        field="formal v8 canary training receipt final recheck",
        expected_payload=built["training_payload"],
    )
    source_after = _immutable_snapshot(
        built["acceptance_source_path"],
        field="canary_acceptance_v8 source final recheck",
        expected_payload=built["acceptance_source_payload"],
    )
    if (
        acceptance_after != acceptance_before
        or index_after != index_before
        or training_after != training_before
        or source_after != source_before
    ):
        _raise("v8 canary authority changed during independent verification")
    _revalidate(built["all_inputs"])

    if (
        acceptance.get("status") != PASS_STATUS
        or acceptance.get("gate_passed") is not True
        or acceptance.get("next_action") != "START_FINAL_THREE_SEED_TRAINING"
    ):
        _raise("STOP v8 canary acceptance cannot authorize final training")
    expected_authorization = {
        "three_seed_training_authorized": True,
        "checkpoint_selected_as_final_model": False,
        "model_authorized": False,
        "calibration_authorized": False,
        "blind_test_authorized": False,
        "gguf_export_authorized": False,
        "x5_deployment_authorized": False,
        "production_integration_authorized": False,
    }
    if acceptance.get("authorization") != expected_authorization:
        _raise("v8 canary acceptance final-training authorization mismatch")
    if acceptance_after["stable_identity"]["mtime_ns"] < source_after["stable_identity"]["mtime_ns"]:
        _raise("v8 canary acceptance predates its verifier source; record a new immutable receipt")

    acceptance_input = _mapping(
        acceptance.get("input"),
        field="acceptance.input",
    )
    training_input = _mapping(
        acceptance_input.get("canary_training_receipt"),
        field="acceptance.input.canary_training_receipt",
    )
    normalized = {
        "required_for_stage": "final",
        **acceptance_after,
        "schema": SCHEMA,
        "gate_version": VERSION,
        "status": PASS_STATUS,
        "gate_passed": True,
        "next_action": "START_FINAL_THREE_SEED_TRAINING",
        "receipt_payload_sha256": receipt_digest,
        "authorization": dict(expected_authorization),
        "claim_boundary": CLAIM_BOUNDARY,
        "formal_v8_binding": dict(
            _mapping(
                acceptance.get("formal_v8_binding"),
                field="acceptance.formal_v8_binding",
            )
        ),
        "evaluation_index": index_after,
        "canary_training_receipt": {
            **training_after,
            "run_id": _text(
                training_input.get("run_id"),
                field="acceptance.input.canary_training_receipt.run_id",
            ),
        },
    }
    if set(normalized) != NORMALIZED_SNAPSHOT_FIELDS:
        _raise("normalized v8 canary snapshot fields mismatch")
    return normalized


def verify_canary_acceptance_v8(
    acceptance_receipt_path: Path,
    evaluation_index_path: Path,
    training_receipt_path: Path,
) -> dict[str, Any]:
    """Verify one immutable PASS receipt and normalize it for v8 final selection."""

    try:
        return _verify_canary_acceptance_v8(
            acceptance_receipt_path=Path(acceptance_receipt_path),
            evaluation_index_path=Path(evaluation_index_path),
            training_receipt_path=Path(training_receipt_path),
        )
    except CanaryAcceptanceV8Error:
        raise
    except v6.CanaryAcceptanceV6Error as exc:
        raise CanaryAcceptanceV8Error(str(exc)) from exc


def record_canary_acceptance_v8(
    *,
    evaluation_index_path: Path,
    canary_training_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Record one immutable v8 canary acceptance or stop receipt."""

    try:
        return _record_canary_acceptance_v8(
            evaluation_index_path=Path(evaluation_index_path),
            canary_training_receipt_path=Path(canary_training_receipt_path),
            output_path=Path(output_path),
        )
    except CanaryAcceptanceV8Error:
        raise
    except v6.CanaryAcceptanceV6Error as exc:
        raise CanaryAcceptanceV8Error(str(exc)) from exc


__all__ = [
    "ERROR_STATUS",
    "FORMAL_MANIFEST_SHA256",
    "FORMAL_TRAINING_GATE_BUNDLE_SHA256",
    "FORMAL_TRAIN_SHA256",
    "FORMAL_VALIDATION_SHA256",
    "PASS_STATUS",
    "SCHEMA",
    "STOP_STATUS",
    "VERSION",
    "CanaryAcceptanceV8Error",
    "record_canary_acceptance_v8",
    "verify_canary_acceptance_v8",
]
