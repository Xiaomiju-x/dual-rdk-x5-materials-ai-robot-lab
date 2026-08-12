"""Strict nonblind-v8 validation-only checkpoint selection freeze.

This module is a version-isolated adapter around the already audited v6
selection policy and v7 stable-snapshot primitives.  It accepts only the
current STRICT_NONBLIND_V8 QLoRA receipt, independently revalidates all 18
validation evaluations, and freezes one checkpoint without opening
calibration or discovering any reserved blind asset.

The adapter is intentionally centralized in
``_adapt_final_training_receipt_v8``.  A future QLoRA receipt-field change must
be added there explicitly; it must never fall through to a v6/v7 or permissive
legacy path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from icmat_foundry.llm import (
    canary_acceptance_v8,
    pointer_checkpoint_eval_v6,
    pointer_checkpoint_eval_v8,
    qlora_full_v6,
    selection_freeze_v7,
    selection_policy_v6,
    semantic_queries_v7,
    shortcut_audit_v8,
    unique_support_audit_v8,
)

SCHEMA = "icmat_llm_selection_freeze.v8"
VERSION = "icmat-selection-freeze-v8c2.0.0"
STATUS = "PASS_STRICT_NONBLIND_V8_SELECTION_FROZEN"
VERIFIED_STATUS = "PASS_STRICT_NONBLIND_V8_SELECTION_FREEZE_VERIFIED"
FAILED_STATUS = "FAILED_NO_STRICT_NONBLIND_V8_SELECTION_FREEZE"

MANIFEST_NAME = qlora_full_v6.NONBLIND_V8_MANIFEST_NAME
MANIFEST_SCHEMA = qlora_full_v6.NONBLIND_V8_MANIFEST_SCHEMA
DATASET_SCHEMA = qlora_full_v6.DATASET_SCHEMA
BUILDER_VERSION = qlora_full_v6.NONBLIND_V8_BUILDER_VERSION
COMMITMENT_NAME = "preblind_commitment.v8.json"

# r3 and the independent r4 build are byte-identical.  The selection authority
# is pinned to the r3 bytes, not merely to a self-declared v8 schema.
PINNED_MANIFEST_R3_SHA256 = canary_acceptance_v8.FORMAL_MANIFEST_SHA256
PINNED_NLI_TREE_SHA256 = semantic_queries_v7.PINNED_NLI_MODEL_TREE_SHA256
PINNED_COMPARE_R3_SHA256 = "b3e24cf9797d2c3ad304e28116231231ebe451b2947731c1380d00960ff5ccf7"
PINNED_LEXICAL_R3_SHA256 = "0f092de91ae9ab70536c95e6ff4865a0d590bd831cb9164045fd54410a046fe5"
PINNED_TRAIN_UNIQUE_R3_SHA256 = "f1305e1a5b2af79468fca5cf0a6bd44686847d002d4c570b93018de740d3cb88"
PINNED_VALIDATION_UNIQUE_R3_SHA256 = "3d5bb644d345fa0a1028be565f0bc61bcb638a2fd4f6a197eac06beb0dca8299"
PINNED_GATE_BUNDLE_R3_SHA256 = canary_acceptance_v8.FORMAL_TRAINING_GATE_BUNDLE_SHA256
PINNED_TRAIN_R3_SHA256 = canary_acceptance_v8.FORMAL_TRAIN_SHA256
PINNED_VALIDATION_R3_SHA256 = canary_acceptance_v8.FORMAL_VALIDATION_SHA256

RUN_RECEIPT_SCHEMA = qlora_full_v6.RUN_RECEIPT_SCHEMA
TRAINER_VERSION = qlora_full_v6.TRAINER_VERSION
TRAINING_PASS_STATUS = "PASS_FINAL_THREE_SEED_ALL_EPOCHS_NOT_SELECTED"
EVALUATION_PASS_STATUS = pointer_checkpoint_eval_v8.FINAL_STATUS
COMPARE_PASS_STATUS = qlora_full_v6.NONBLIND_V8_COMPARE_STATUS
LEXICAL_PASS_STATUS = shortcut_audit_v8.PASS_STATUS
UNIQUE_SUPPORT_PASS_STATUS = unique_support_audit_v8.PASS_STATUS
CANARY_SCHEMA = canary_acceptance_v8.SCHEMA
CANARY_VERSION = canary_acceptance_v8.VERSION
CANARY_PASS_STATUS = canary_acceptance_v8.PASS_STATUS

EXPECTED_CHECKPOINTS = selection_policy_v6.EXPECTED_CHECKPOINT_COUNT
EXPECTED_VALIDATION_ROWS = selection_policy_v6.EXPECTED_VALIDATION_SAMPLES
EXPECTED_SEEDS = 3
EXPECTED_EPOCHS = tuple(range(1, qlora_full_v6.FIXED_EPOCHS + 1))
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION_PATH_V8C2 = (
    WORKSPACE_ROOT / "docs" / "ai_brain_finals_20260728" / "ICMAT_POINTER_V8C2_PREREGISTRATION.json"
)
PREREGISTRATION_SCHEMA_V8C2 = "icmat_pointer_v8c2_preregistration.v1"
PREREGISTRATION_PROTOCOL_V8C2 = "ICMAT-Pointer-v8c2-PREREG-r1"
PREREGISTRATION_STATUS_V8C2 = "FROZEN_BEFORE_IMPLEMENTATION_AND_TRAINING"
PINNED_PREREGISTRATION_SHA256_V8C2 = "955165d8e9766300e621fe6a1291e4a2ff1dd96a85692738cfe66f28a1b03c24"
TRAINING_PROFILE_V8C2 = "V8C2_CAPACITY_REGULARIZED"
CANARY_SEEDS_V8C2 = (20260728,)
FINAL_SEEDS_V8C2 = (20260729, 20260730, 20260731)

_TRAINING_FIELDS_V8 = {
    "schema",
    "trainer_version",
    "created_at",
    "status",
    "stage",
    "run_id",
    "atomic_publish",
    "network_used",
    "input_snapshot",
    "configuration",
    "configuration_sha256",
    "software",
    "cuda",
    "seeds",
    "checkpoint_count",
    "selection",
    "authorization",
    "data_access",
    "wall_seconds",
    "claim_boundary",
    "training_gate_bundle_sha256",
    "v8_inspected_input_sha256",
    "training_profile",
    "preregistration_protocol_id",
    "preregistration_sha256",
}
_CANARY_TRAINING_FIELDS_V8 = _TRAINING_FIELDS_V8 | {"canary_attempt"}
_DATASET_FIELDS_V8 = {
    "path",
    "contract",
    "manifest",
    "splits",
    "source_input_binding",
    "strict_artifact_receipts",
    "double_build_evidence",
    "strict_audit_gates",
    "training_gate_bundle",
    "training_gate_bundle_sha256",
    "implementation_receipts",
    "seed_revalidation",
    "training_data_access",
    "inspected_input_sha256",
    "v8_inspected_input_sha256",
    "v8c2_preregistration",
}
_FALSE_TRAINING_AUTHORIZATION = {
    "checkpoint_selected": False,
    "model_authorized": False,
    "calibration_authorized": False,
    "blind_test_authorized": False,
    "gguf_export_authorized": False,
    "deployment_authorized": False,
    "production_integration_authorized": False,
}
_RUN_DATA_ACCESS_FIELDS_V8 = {
    "train_content_read",
    "validation_content_read",
    "calibration_content_read",
    "calibration_content_hashed",
    "blind_test_content_read",
    "blind_test_content_hashed",
    "calibration_integrity_snapshot_opened",
    "calibration_integrity_content_read",
    "calibration_integrity_content_hashed",
    "calibration_content_loaded_for_training",
    "calibration_used_for_checkpoint_selection",
    "nonblind_compare_audit_verified",
    "scoped_lexical_audit_verified",
    "scoped_lexical_audit_locally_recomputed",
    "train_unique_support_audit_verified",
    "validation_unique_support_audit_verified",
    "unique_support_fixed_cpu_nli_load_count",
    "unique_support_nli_repeated_per_seed",
    "second_build_fixed_files_recomputed",
    "declared_nonblind_audit_artifacts_opened",
    "declared_nonblind_audit_artifacts_hashed",
    "blind_materialized",
    "blind_discovered",
    "blind_path_constructed",
    "blind_filesystem_metadata_accessed",
    "blind_content_opened",
    "blind_content_read",
    "blind_content_hashed",
}
_DATASET_ACCESS_FIELDS_V8 = {
    "opened_splits",
    "integrity_only_splits",
    "primary_fixed_files_stably_opened",
    "second_fixed_files_stably_opened",
    "second_build_bytes_compared_directly",
    "second_build_file_identities_compared_directly",
    "nonblind_compare_audit_verified",
    "scoped_lexical_audit_locally_recomputed",
    "train_unique_support_locally_recomputed",
    "validation_unique_support_locally_recomputed",
    "unique_support_nli_load_count",
    "unique_support_nli_device",
    "calibration_integrity_snapshot_opened",
    "calibration_integrity_content_read",
    "calibration_integrity_content_parsed",
    "calibration_integrity_content_hashed",
    "calibration_content_loaded_for_training",
    "calibration_used_for_checkpoint_selection",
    "blind_materialized",
    "blind_discovered",
    "blind_path_constructed",
    "blind_filesystem_metadata_accessed",
    "blind_content_opened",
    "blind_content_read",
    "blind_content_hashed",
}
_COMPARE_GATE_FIELDS = {
    "path",
    "bytes",
    "sha256",
    "stable_identity",
    "schema",
    "audit_version",
    "status",
    "audit_passed",
    "fixed_files_verified",
    "direct_byte_comparison_is_authoritative",
}
_LEXICAL_GATE_FIELDS = {
    "path",
    "bytes",
    "sha256",
    "stable_identity",
    "schema",
    "audit_version",
    "status",
    "audit_id",
    "full_report_locally_recomputed",
    "train_per_sample_locally_recomputed",
    "validation_per_sample_locally_recomputed",
}
_UNIQUE_GATE_FIELDS = {
    "path",
    "bytes",
    "sha256",
    "stable_identity",
    "schema",
    "audit_version",
    "status",
    "audit_id",
    "split",
    "answer_examples_audited",
    "all_spans_locally_recomputed",
    "nli_device",
}
_CANARY_FORMAL_BINDING_FIELDS_V8 = {
    "contract",
    "manifest_sha256",
    "train_sha256",
    "validation_sha256",
    "training_gate_bundle_sha256",
    "v8_inspected_input_sha256",
}
# The QLoRA final receipt must normalize exactly these fields. Any upstream
# v8 contract change is handled here explicitly; v6/v7 snapshots fail closed.
_CANARY_SNAPSHOT_FIELDS_V8 = {
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
_CANARY_RECEIPT_FIELDS_V8 = {
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
_V8_INDEX_FIELDS = {
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
_V8_INDEX_DATASET_FIELDS = {
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
_V8_INDEX_CHECKPOINT_FIELDS = {
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


class SelectionFreezeV8Error(RuntimeError):
    """Raised when a strict nonblind-v8 selection cannot be frozen."""


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


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_mapping(
    value: Any,
    *,
    label: str,
    exact: set[str] | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SelectionFreezeV8Error(f"{label}: object required")
    if exact is not None and set(value) != exact:
        raise SelectionFreezeV8Error(f"{label}: exact fields mismatch")
    return value


def _require_sequence(value: Any, *, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise SelectionFreezeV8Error(f"{label}: array required")
    return value


def _identity_receipt(
    identity: tuple[int, int, int, int, int],
) -> dict[str, int]:
    return selection_freeze_v7._identity_receipt(identity)


def _receipt_binding(
    path: Path,
    payload: bytes,
    value: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "schema": value.get("schema"),
        "status": value.get("status"),
    }


def _selection_authorization_v8() -> dict[str, Any]:
    return {
        "calibration_authorized": True,
        "calibration_complete_split_only": True,
        "calibration_expected_rows": 150,
        "calibration_may_reselect_checkpoint": False,
        "ablation_authorized_on_validation_only": True,
        "blind_test_authorized": False,
        "gguf_export_authorized": False,
        "x5_execution_authorized": False,
        "deployment_authorized": False,
        "production_integration_authorized": False,
    }


def _load_preregistration_authority_v8c2() -> dict[str, Any]:
    """Load the one immutable v8c2 preregistration without reserved-data I/O."""

    try:
        path, payload, value = selection_freeze_v7._load_json(
            PREREGISTRATION_PATH_V8C2,
            label="v8c2 preregistration",
        )
        snapshot = selection_freeze_v7._stable_file_snapshot(
            path,
            label="v8c2 preregistration stable snapshot",
        )
    except (OSError, ValueError, selection_freeze_v7.SelectionFreezeV7Error) as exc:
        raise SelectionFreezeV8Error(f"v8c2 preregistration unavailable: {exc}") from exc
    preregistration = _require_mapping(
        value,
        label="v8c2 preregistration",
    )
    frozen_data = _require_mapping(
        preregistration.get("frozen_data"),
        label="v8c2 preregistration frozen data",
    )
    algorithm = _require_mapping(
        preregistration.get("single_atomic_algorithm_change"),
        label="v8c2 preregistration algorithm",
    )
    fixed_runs = _require_mapping(
        preregistration.get("fixed_runs"),
        label="v8c2 preregistration fixed runs",
    )
    expected_algorithm = {
        "profile": TRAINING_PROFILE_V8C2,
        "assistant_only_cross_entropy": True,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_alpha_over_rank": 2.0,
        "lora_dropout": 0.1,
        "learning_rate": 0.0002,
        "num_train_epochs": 6,
        "max_seq_length": 1152,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "warmup_ratio": 0.05,
        "weight_decay": 0.0,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "data_resampling": False,
        "data_augmentation": False,
        "class_or_task_weighting": False,
        "layer_freezing": False,
        "checkpoint_interpolation": False,
        "checkpoint_voting": False,
        "nli_answer_override": False,
        "inference_contract_changed": False,
    }
    if (
        payload != snapshot.payload
        or snapshot.sha256 != PINNED_PREREGISTRATION_SHA256_V8C2
        or qlora_full_v6.V8C2_TRAINING_PROFILE != TRAINING_PROFILE_V8C2
        or qlora_full_v6.V8C2_PREREGISTRATION_PROTOCOL_ID
        != PREREGISTRATION_PROTOCOL_V8C2
        or qlora_full_v6.V8C2_PREREGISTRATION_SHA256
        != PINNED_PREREGISTRATION_SHA256_V8C2
        or Path(qlora_full_v6.V8C2_PREREGISTRATION_PATH).resolve(strict=True)
        != snapshot.path
        or preregistration.get("schema") != PREREGISTRATION_SCHEMA_V8C2
        or preregistration.get("protocol_id") != PREREGISTRATION_PROTOCOL_V8C2
        or preregistration.get("status") != PREREGISTRATION_STATUS_V8C2
        or frozen_data.get("dataset_contract") != "STRICT_NONBLIND_V8"
        or frozen_data.get("dataset_manifest_sha256") != PINNED_MANIFEST_R3_SHA256
        or frozen_data.get("train_sha256") != PINNED_TRAIN_R3_SHA256
        or frozen_data.get("validation_sha256") != PINNED_VALIDATION_R3_SHA256
        or frozen_data.get("training_gate_bundle_sha256") != PINNED_GATE_BUNDLE_R3_SHA256
        or dict(algorithm) != expected_algorithm
        or tuple(fixed_runs.get("canary_seeds", ())) != CANARY_SEEDS_V8C2
        or tuple(fixed_runs.get("final_seeds", ())) != FINAL_SEEDS_V8C2
        or fixed_runs.get("canary_runs_allowed") != 1
        or fixed_runs.get("final_runs_allowed_after_canary_pass") != 3
        or fixed_runs.get("seed_substitution_allowed") is not False
        or fixed_runs.get("additional_v8c2_variants_allowed") is not False
    ):
        raise SelectionFreezeV8Error("v8c2 preregistration bytes or frozen contract mismatch")
    try:
        qlora_snapshot = qlora_full_v6._stable_snapshot_v7(
            PREREGISTRATION_PATH_V8C2,
            label="v8c2 selection preregistration",
            maximum_bytes=qlora_full_v6._STRICT_MAX_JSON_BYTES,
        )
        authority = qlora_full_v6._validate_v8c2_preregistration(
            qlora_snapshot
        )
    except (OSError, ValueError, qlora_full_v6.QLoRAV6Error) as exc:
        raise SelectionFreezeV8Error(
            f"v8c2 preregistration authority rejected: {exc}"
        ) from exc
    if (
        authority["path"] != str(snapshot.path)
        or authority["bytes"] != len(snapshot.payload)
        or authority["sha256"] != snapshot.sha256
        or authority["stable_identity"]
        != _identity_receipt(snapshot.identity)
    ):
        raise SelectionFreezeV8Error(
            "v8c2 preregistration authority snapshot mismatch"
        )
    return authority


def _expected_qlora_protocol_fields_v8c2() -> dict[str, str]:
    expected = {
        "training_profile": TRAINING_PROFILE_V8C2,
        "preregistration_protocol_id": PREREGISTRATION_PROTOCOL_V8C2,
        "preregistration_sha256": PINNED_PREREGISTRATION_SHA256_V8C2,
    }
    if qlora_full_v6._v8c2_receipt_fields() != expected:
        raise SelectionFreezeV8Error("QLoRA v8c2 receipt-field contract mismatch")
    return expected


def _validate_qlora_protocol_fields_v8c2(
    receipt: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
    label: str,
) -> dict[str, str]:
    preregistration = _require_mapping(
        authority.get("preregistration"),
        label="v8c2 preregistration authority",
    )
    expected = _expected_qlora_protocol_fields_v8c2()
    if (
        any(receipt.get(field) != value for field, value in expected.items())
        or preregistration.get("sha256") != expected["preregistration_sha256"]
        or preregistration.get("protocol_id")
        != expected["preregistration_protocol_id"]
        or preregistration.get("profile") != expected["training_profile"]
    ):
        raise SelectionFreezeV8Error(f"{label}: QLoRA v8c2 protocol binding mismatch")
    return expected


def _validate_dataset_preregistration_v8c2(
    value: Any,
    *,
    authority: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    expected = _require_mapping(
        authority.get("preregistration"),
        label="v8c2 preregistration authority",
    )
    declared = _require_mapping(
        value,
        label=label,
        exact=set(expected),
    )
    if dict(declared) != dict(expected):
        raise SelectionFreezeV8Error(f"{label}: immutable preregistration snapshot mismatch")
    return dict(declared)


def _expected_configuration_fields_v8() -> set[str]:
    return set(
        qlora_full_v6._configuration_payload(
            qlora_full_v6.QLoRATrainingConfigV6()
        )
    )


def _validate_configuration_v8(
    value: Any,
    *,
    expected_stage: str = "final",
) -> tuple[Mapping[str, Any], tuple[int, ...]]:
    if expected_stage not in {"canary", "final"}:
        raise SelectionFreezeV8Error("v8c2 configuration stage must be canary or final")
    configuration = _require_mapping(
        value,
        label=f"v8c2 {expected_stage} training configuration",
        exact=_expected_configuration_fields_v8(),
    )
    seeds_raw = _require_sequence(
        configuration.get("seeds"),
        label=f"v8c2 {expected_stage} training seeds",
    )
    expected_seeds = CANARY_SEEDS_V8C2 if expected_stage == "canary" else FINAL_SEEDS_V8C2
    if len(seeds_raw) != len(expected_seeds) or any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed > 2_147_483_647
        for seed in seeds_raw
    ):
        raise SelectionFreezeV8Error(f"v8c2 {expected_stage} training seed population mismatch")
    seeds = tuple(int(seed) for seed in seeds_raw)
    if seeds != expected_seeds:
        raise SelectionFreezeV8Error(f"v8c2 {expected_stage} training seeds differ from preregistration")
    required = {
        "stage": expected_stage,
        "num_train_epochs": 6,
        "max_seq_length": 1152,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 0.0002,
        "warmup_ratio": 0.05,
        "weight_decay": 0.0,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "model_family": "Qwen2.5-0.5B-Instruct",
        "quantization": "NF4",
        "double_quantization": True,
        "compute_dtype": "bfloat16",
        "optimizer": "paged_adamw_8bit",
        "gradient_checkpointing": True,
        "assistant_only_loss": True,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "save_every_epoch": True,
        "save_total_limit": None,
        "load_best_model_at_end": False,
        "early_stopping": False,
        "automatic_checkpoint_selection": False,
        "calibration_access": "FORBIDDEN",
        "blind_test_access": "FORBIDDEN",
    }
    if any(configuration.get(key) != expected for key, expected in required.items()) or configuration.get(
        "target_modules"
    ) != [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]:
        raise SelectionFreezeV8Error(
            f"v8c2 {expected_stage} training algorithm/configuration contract mismatch"
        )
    return configuration, seeds


def _validate_gate_bundle_v8(
    dataset: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
) -> tuple[str, str]:
    gates = _require_mapping(
        dataset.get("strict_audit_gates"),
        label="v8 strict audit gates",
        exact={"nonblind_compare", "scoped_lexical", "unique_support"},
    )
    unique = _require_mapping(
        gates.get("unique_support"),
        label="v8 unique-support gates",
        exact={"train", "validation"},
    )
    compare = _require_mapping(
        gates.get("nonblind_compare"),
        label="v8 compare gate",
        exact=_COMPARE_GATE_FIELDS,
    )
    lexical = _require_mapping(
        gates.get("scoped_lexical"),
        label="v8 lexical gate",
        exact=_LEXICAL_GATE_FIELDS,
    )
    train_unique = _require_mapping(
        unique.get("train"),
        label="v8 train unique-support gate",
        exact=_UNIQUE_GATE_FIELDS,
    )
    validation_unique = _require_mapping(
        unique.get("validation"),
        label="v8 validation unique-support gate",
        exact=_UNIQUE_GATE_FIELDS,
    )
    if (
        compare.get("status") != COMPARE_PASS_STATUS
        or compare.get("audit_passed") is not True
        or compare.get("fixed_files_verified") != 12
        or compare.get("direct_byte_comparison_is_authoritative") is not True
        or lexical.get("status") != LEXICAL_PASS_STATUS
        or lexical.get("full_report_locally_recomputed") is not True
        or lexical.get("train_per_sample_locally_recomputed") is not True
        or lexical.get("validation_per_sample_locally_recomputed") is not True
        or train_unique.get("status") != UNIQUE_SUPPORT_PASS_STATUS
        or train_unique.get("split") != "train"
        or train_unique.get("all_spans_locally_recomputed") is not True
        or train_unique.get("nli_device") != "cpu"
        or validation_unique.get("status") != UNIQUE_SUPPORT_PASS_STATUS
        or validation_unique.get("split") != "validation"
        or validation_unique.get("all_spans_locally_recomputed") is not True
        or validation_unique.get("nli_device") != "cpu"
        or compare.get("sha256") != PINNED_COMPARE_R3_SHA256
        or lexical.get("sha256") != PINNED_LEXICAL_R3_SHA256
        or train_unique.get("sha256") != PINNED_TRAIN_UNIQUE_R3_SHA256
        or validation_unique.get("sha256") != PINNED_VALIDATION_UNIQUE_R3_SHA256
    ):
        raise SelectionFreezeV8Error("v8 audit gate status/boundary mismatch")
    expected_gates = authority.get("strict_audit_gates")
    if gates != expected_gates:
        raise SelectionFreezeV8Error("v8 gate receipts differ from external authority snapshots")

    bundle = _require_mapping(
        dataset.get("training_gate_bundle"),
        label="v8 training gate bundle",
        exact={
            "contract",
            "nonblind_compare",
            "scoped_lexical",
            "unique_support",
            "nli_model",
            "training_gate_bundle_sha256",
        },
    )
    bundle_unique = _require_mapping(
        bundle.get("unique_support"),
        label="v8 gate bundle unique-support",
        exact={"train", "validation"},
    )
    nli_model = _require_mapping(
        bundle.get("nli_model"),
        label="v8 gate bundle NLI model",
        exact={"tree_sha256", "receipt_sha256", "device"},
    )
    expected_bundle_body = {
        "contract": "STRICT_NONBLIND_V8",
        "nonblind_compare": {
            "sha256": compare.get("sha256"),
            "status": compare.get("status"),
        },
        "scoped_lexical": {
            "sha256": lexical.get("sha256"),
            "status": lexical.get("status"),
        },
        "unique_support": {
            "train": {
                "sha256": train_unique.get("sha256"),
                "status": train_unique.get("status"),
            },
            "validation": {
                "sha256": validation_unique.get("sha256"),
                "status": validation_unique.get("status"),
            },
        },
        "nli_model": {
            "tree_sha256": PINNED_NLI_TREE_SHA256,
            "receipt_sha256": nli_model.get("receipt_sha256"),
            "device": "cpu",
        },
    }
    bundle_digest = canonical_sha256(expected_bundle_body)
    if (
        bundle.get("contract") != "STRICT_NONBLIND_V8"
        or bundle.get("nonblind_compare") != expected_bundle_body["nonblind_compare"]
        or bundle.get("scoped_lexical") != expected_bundle_body["scoped_lexical"]
        or bundle_unique != expected_bundle_body["unique_support"]
        or nli_model != expected_bundle_body["nli_model"]
        or bundle.get("training_gate_bundle_sha256") != bundle_digest
        or dataset.get("training_gate_bundle_sha256") != bundle_digest
        or authority.get("training_gate_bundle_sha256") != bundle_digest
        or authority.get("training_gate_bundle") != bundle
        or bundle_digest != PINNED_GATE_BUNDLE_R3_SHA256
    ):
        raise SelectionFreezeV8Error("v8 training gate bundle or NLI authority mismatch")
    return bundle_digest, str(nli_model["tree_sha256"])


def _validate_training_data_access_v8(
    dataset: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    dataset_access = _require_mapping(
        dataset.get("training_data_access"),
        label="v8 dataset access",
        exact=_DATASET_ACCESS_FIELDS_V8,
    )
    if (
        dataset_access.get("opened_splits") != ["train", "validation"]
        or dataset_access.get("integrity_only_splits") != ["calibration"]
        or dataset_access.get("primary_fixed_files_stably_opened") != 12
        or dataset_access.get("second_fixed_files_stably_opened") != 12
        or dataset_access.get("unique_support_nli_load_count") != 1
        or dataset_access.get("unique_support_nli_device") != "cpu"
    ):
        raise SelectionFreezeV8Error("v8 dataset access population/device mismatch")
    required_dataset_true = {
        "second_build_bytes_compared_directly",
        "second_build_file_identities_compared_directly",
        "nonblind_compare_audit_verified",
        "scoped_lexical_audit_locally_recomputed",
        "train_unique_support_locally_recomputed",
        "validation_unique_support_locally_recomputed",
        "calibration_integrity_snapshot_opened",
        "calibration_integrity_content_read",
        "calibration_integrity_content_parsed",
        "calibration_integrity_content_hashed",
    }
    required_dataset_false = {
        "calibration_content_loaded_for_training",
        "calibration_used_for_checkpoint_selection",
        "blind_materialized",
        "blind_discovered",
        "blind_path_constructed",
        "blind_filesystem_metadata_accessed",
        "blind_content_opened",
        "blind_content_read",
        "blind_content_hashed",
    }
    if any(dataset_access.get(field) is not True for field in required_dataset_true):
        raise SelectionFreezeV8Error("v8 dataset audit access proof is incomplete")
    if any(dataset_access.get(field) is not False for field in required_dataset_false):
        raise SelectionFreezeV8Error("v8 dataset accessed calibration for training or reserved blind data")

    run_access = _require_mapping(
        receipt.get("data_access"),
        label="v8 final run data access",
        exact=_RUN_DATA_ACCESS_FIELDS_V8,
    )
    required_run_true = {
        "train_content_read",
        "validation_content_read",
        "calibration_integrity_snapshot_opened",
        "calibration_integrity_content_read",
        "calibration_integrity_content_hashed",
        "nonblind_compare_audit_verified",
        "scoped_lexical_audit_verified",
        "scoped_lexical_audit_locally_recomputed",
        "train_unique_support_audit_verified",
        "validation_unique_support_audit_verified",
        "second_build_fixed_files_recomputed",
    }
    required_run_false = {
        "calibration_content_read",
        "calibration_content_hashed",
        "blind_test_content_read",
        "blind_test_content_hashed",
        "calibration_content_loaded_for_training",
        "calibration_used_for_checkpoint_selection",
        "unique_support_nli_repeated_per_seed",
        "blind_materialized",
        "blind_discovered",
        "blind_path_constructed",
        "blind_filesystem_metadata_accessed",
        "blind_content_opened",
        "blind_content_read",
        "blind_content_hashed",
    }
    if any(run_access.get(field) is not True for field in required_run_true):
        raise SelectionFreezeV8Error("v8 final run gate verification is incomplete")
    if any(run_access.get(field) is not False for field in required_run_false):
        raise SelectionFreezeV8Error("v8 final run crossed calibration/blind access boundary")
    if (
        run_access.get("unique_support_fixed_cpu_nli_load_count") != 1
        or run_access.get("declared_nonblind_audit_artifacts_opened") != 8
        or run_access.get("declared_nonblind_audit_artifacts_hashed") != 8
    ):
        raise SelectionFreezeV8Error("v8 final run audit-count contract mismatch")


def _validate_checkpoint_population_v8(
    specs: Sequence[Mapping[str, Any]],
    *,
    seeds: tuple[int, ...],
) -> None:
    if len(specs) != EXPECTED_CHECKPOINTS:
        raise SelectionFreezeV8Error("v8 final checkpoint population is not complete 3x6=18")
    expected_pairs = {(seed, epoch) for seed in seeds for epoch in EXPECTED_EPOCHS}
    pairs: set[tuple[int, int]] = set()
    checkpoint_ids: set[str] = set()
    paths: set[str] = set()
    checkpoint_trees: set[str] = set()
    adapter_trees: set[str] = set()
    for position, spec in enumerate(specs):
        checkpoint_id = spec.get("checkpoint_id")
        seed = spec.get("seed")
        epoch = spec.get("epoch")
        path = str(spec.get("path"))
        checkpoint_tree = spec.get("training_checkpoint_tree_sha256")
        adapter_tree = spec.get("training_adapter_tree_sha256")
        if (
            not isinstance(checkpoint_id, str)
            or not checkpoint_id
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or not _valid_sha256(checkpoint_tree)
            or not _valid_sha256(adapter_tree)
        ):
            raise SelectionFreezeV8Error(f"v8 checkpoint spec {position} is malformed")
        pair = (seed, epoch)
        if (
            pair in pairs
            or checkpoint_id in checkpoint_ids
            or path in paths
            or checkpoint_tree in checkpoint_trees
            or adapter_tree in adapter_trees
        ):
            raise SelectionFreezeV8Error("v8 checkpoint population contains a duplicate identity")
        pairs.add(pair)
        checkpoint_ids.add(checkpoint_id)
        paths.add(path)
        checkpoint_trees.add(str(checkpoint_tree))
        adapter_trees.add(str(adapter_tree))
    if pairs != expected_pairs:
        raise SelectionFreezeV8Error("v8 checkpoint population must uniquely cover every seed x epoch")


def _canary_contract_from_authority_v8(
    canary: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> Mapping[str, Any]:
    expected_canary = authority.get("canary_acceptance")
    if expected_canary is not None and canary != expected_canary:
        raise SelectionFreezeV8Error("v8 final canary acceptance differs from external authority")
    contract = authority.get("canary_contract")
    if contract is None:
        binding = _require_mapping(
            canary.get("formal_v8_binding"),
            label="v8 canary formal generation binding",
            exact=_CANARY_FORMAL_BINDING_FIELDS_V8,
        )
        contract = {
            "contract": binding.get("contract"),
            "manifest_sha256": binding.get("manifest_sha256"),
            "train_sha256": binding.get("train_sha256"),
            "validation_sha256": binding.get("validation_sha256"),
            "training_gate_bundle_sha256": binding.get("training_gate_bundle_sha256"),
            "v8_inspected_input_sha256": binding.get("v8_inspected_input_sha256"),
        }
    return _require_mapping(
        contract,
        label="v8 canary contract adapter",
        exact=_CANARY_FORMAL_BINDING_FIELDS_V8,
    )


def _validate_formal_split_bindings_v8(
    dataset: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
) -> None:
    splits = _require_mapping(
        dataset.get("splits"),
        label="v8 final dataset split inventory",
        exact={"train", "validation", "calibration"},
    )
    expected = {
        "train": (
            PINNED_TRAIN_R3_SHA256,
            250,
            authority.get("train_sha256"),
        ),
        "validation": (
            PINNED_VALIDATION_R3_SHA256,
            EXPECTED_VALIDATION_ROWS,
            authority.get("validation_sha256"),
        ),
    }
    for split, (sha256, examples, authority_sha256) in expected.items():
        binding = _require_mapping(
            splits.get(split),
            label=f"v8 final {split} split binding",
        )
        if (
            binding.get("path") != f"{split}.jsonl"
            or binding.get("sha256") != sha256
            or binding.get("examples") != examples
            or authority_sha256 != sha256
        ):
            raise SelectionFreezeV8Error(f"v8 formal {split} split binding mismatch")


def _adapt_final_training_receipt_v8(
    receipt: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
    checkpoint_specs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Adapt exactly the current QLoRA v8 final receipt or fail closed."""

    receipt = _require_mapping(
        receipt,
        label="v8 final training receipt",
        exact=_TRAINING_FIELDS_V8,
    )
    if (
        receipt.get("schema") != RUN_RECEIPT_SCHEMA
        or receipt.get("trainer_version") != TRAINER_VERSION
        or receipt.get("status") != TRAINING_PASS_STATUS
        or receipt.get("stage") != "final"
        or receipt.get("checkpoint_count") != EXPECTED_CHECKPOINTS
        or receipt.get("atomic_publish") is not True
        or receipt.get("network_used") is not False
        or not isinstance(receipt.get("run_id"), str)
        or not receipt.get("run_id")
    ):
        raise SelectionFreezeV8Error("v8 final training receipt is not a completed strict 3x6 run")
    wall_seconds = receipt.get("wall_seconds")
    if (
        isinstance(wall_seconds, bool)
        or not isinstance(wall_seconds, (int, float))
        or not math.isfinite(float(wall_seconds))
        or float(wall_seconds) < 0.0
    ):
        raise SelectionFreezeV8Error("v8 final training wall time is invalid")
    if receipt.get("authorization") != _FALSE_TRAINING_AUTHORIZATION:
        raise SelectionFreezeV8Error("v8 final training receipt already grants authorization")
    protocol = _validate_qlora_protocol_fields_v8c2(
        receipt,
        authority=authority,
        label="v8c2 final training",
    )
    selection = _require_mapping(
        receipt.get("selection"),
        label="v8 final training selection",
        exact={
            "automatic_selection_performed",
            "selected_seed",
            "selected_epoch",
            "selected_adapter",
            "selection_metric",
            "required_next_step",
        },
    )
    if (
        selection.get("automatic_selection_performed") is not False
        or selection.get("selected_seed") is not None
        or selection.get("selected_epoch") is not None
        or selection.get("selected_adapter") is not None
        or selection.get("selection_metric") is not None
    ):
        raise SelectionFreezeV8Error("v8 final training receipt already contains a selection")
    configuration, seeds = _validate_configuration_v8(receipt.get("configuration"))
    if receipt.get("configuration_sha256") != canonical_sha256(configuration):
        raise SelectionFreezeV8Error("v8 final training configuration digest mismatch")
    input_snapshot = _require_mapping(
        receipt.get("input_snapshot"),
        label="v8 final input snapshot",
        exact={"dataset", "base_model", "canary_acceptance", "source_files"},
    )
    dataset = _require_mapping(
        input_snapshot.get("dataset"),
        label="v8 final dataset snapshot",
        exact=_DATASET_FIELDS_V8,
    )
    _validate_dataset_preregistration_v8c2(
        dataset.get("v8c2_preregistration"),
        authority=authority,
        label="v8c2 final training dataset preregistration",
    )
    manifest = _require_mapping(
        dataset.get("manifest"),
        label="v8 final manifest binding",
        exact={
            "path",
            "bytes",
            "sha256",
            "stable_identity",
            "schema",
            "dataset_schema",
            "builder_version",
        },
    )
    if (
        dataset.get("contract") != "STRICT_NONBLIND_V8"
        or str(dataset.get("path")) != str(authority.get("dataset_root"))
        or manifest.get("path") != MANIFEST_NAME
        or manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("dataset_schema") != DATASET_SCHEMA
        or manifest.get("builder_version") != BUILDER_VERSION
        or manifest.get("sha256") != PINNED_MANIFEST_R3_SHA256
        or manifest != authority.get("manifest")
    ):
        raise SelectionFreezeV8Error("v8 final dataset/manifest r3 authority mismatch")
    _validate_formal_split_bindings_v8(dataset, authority=authority)
    inspected = dataset.get("inspected_input_sha256")
    if (
        not _valid_sha256(inspected)
        or dataset.get("v8_inspected_input_sha256") != inspected
        or receipt.get("v8_inspected_input_sha256") != inspected
        or authority.get("inspected_input_sha256") != inspected
    ):
        raise SelectionFreezeV8Error("v8 inspected_input_sha256 binding mismatch")
    gate_digest, nli_tree = _validate_gate_bundle_v8(
        dataset,
        authority=authority,
    )
    if receipt.get("training_gate_bundle_sha256") != gate_digest:
        raise SelectionFreezeV8Error("v8 final training gate digest mismatch")
    nli_identity = _require_mapping(
        _require_mapping(
            dataset.get("seed_revalidation"),
            label="v8 seed revalidation",
            exact={"files", "nli_model"},
        ).get("nli_model"),
        label="v8 seed NLI identity",
    )
    nli_receipt = _require_mapping(
        nli_identity.get("model_receipt"),
        label="v8 seed NLI model receipt",
    )
    authority_nli = _require_mapping(
        authority.get("nli_model_identity"),
        label="v8 external NLI identity",
    )
    authority_nli_receipt = _require_mapping(
        authority_nli.get("model_receipt"),
        label="v8 external NLI model receipt",
    )
    if (
        nli_identity.get("tree_sha256") != PINNED_NLI_TREE_SHA256
        or nli_tree != PINNED_NLI_TREE_SHA256
        or authority_nli.get("tree_sha256") != PINNED_NLI_TREE_SHA256
        or nli_receipt.get("sha256") != authority_nli_receipt.get("sha256")
        or nli_receipt.get("sha256") != dataset["training_gate_bundle"]["nli_model"]["receipt_sha256"]
    ):
        raise SelectionFreezeV8Error("v8 fixed NLI tree/receipt mismatch")

    canary = _require_mapping(
        input_snapshot.get("canary_acceptance"),
        label="v8 final canary acceptance",
    )
    canary_contract = _canary_contract_from_authority_v8(
        canary,
        authority,
    )
    if (
        canary.get("required_for_stage") != "final"
        or canary.get("schema") != CANARY_SCHEMA
        or canary.get("gate_version") != CANARY_VERSION
        or canary.get("status") != CANARY_PASS_STATUS
        or canary.get("gate_passed") is not True
        or canary_contract
        != {
            "contract": "STRICT_NONBLIND_V8",
            "manifest_sha256": PINNED_MANIFEST_R3_SHA256,
            "train_sha256": PINNED_TRAIN_R3_SHA256,
            "validation_sha256": PINNED_VALIDATION_R3_SHA256,
            "training_gate_bundle_sha256": gate_digest,
            "v8_inspected_input_sha256": inspected,
        }
    ):
        raise SelectionFreezeV8Error("v7/legacy canary cannot authorize a strict v8 final run")
    canary_authorization = _require_mapping(
        canary.get("authorization"),
        label="v8 canary authorization",
    )
    if canary_authorization.get("three_seed_training_authorized") is not True or any(
        canary_authorization.get(field) is not False
        for field in (
            "checkpoint_selected_as_final_model",
            "model_authorized",
            "calibration_authorized",
            "blind_test_authorized",
            "gguf_export_authorized",
            "x5_deployment_authorized",
            "production_integration_authorized",
        )
    ):
        raise SelectionFreezeV8Error("v8 canary authorization boundary mismatch")
    _validate_training_data_access_v8(dataset, receipt)

    seed_receipts = _require_sequence(
        receipt.get("seeds"),
        label="v8 final seed receipts",
    )
    if len(seed_receipts) != EXPECTED_SEEDS:
        raise SelectionFreezeV8Error("v8 final training must contain exactly three seed receipts")
    observed_seed_receipts: set[int] = set()
    for raw in seed_receipts:
        seed_receipt = _require_mapping(
            raw,
            label="v8 final seed receipt",
        )
        _validate_qlora_protocol_fields_v8c2(
            seed_receipt,
            authority=authority,
            label="v8c2 final seed receipt",
        )
        seed = seed_receipt.get("seed")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed not in seeds
            or seed in observed_seed_receipts
            or seed_receipt.get("schema") != qlora_full_v6.SEED_RECEIPT_SCHEMA
            or seed_receipt.get("trainer_version") != TRAINER_VERSION
            or seed_receipt.get("status") != "PASS_SEED_TRAINED_ALL_EPOCHS_NOT_SELECTED"
            or seed_receipt.get("stage") != "final"
            or seed_receipt.get("configuration") != configuration
            or len(
                _require_sequence(
                    seed_receipt.get("epoch_checkpoints"),
                    label=f"v8 seed {seed} checkpoints",
                )
            )
            != 6
        ):
            raise SelectionFreezeV8Error("v8 final seed receipt population/contract mismatch")
        observed_seed_receipts.add(seed)
    if observed_seed_receipts != set(seeds):
        raise SelectionFreezeV8Error("v8 final seed receipts do not match configuration")
    _validate_checkpoint_population_v8(checkpoint_specs, seeds=seeds)
    return {
        "contract": "STRICT_NONBLIND_V8",
        "profile": protocol["training_profile"],
        "preregistration_protocol_id": protocol["preregistration_protocol_id"],
        "preregistration_sha256": protocol["preregistration_sha256"],
        "configuration_sha256": str(receipt["configuration_sha256"]),
        "manifest_sha256": PINNED_MANIFEST_R3_SHA256,
        "train_sha256": PINNED_TRAIN_R3_SHA256,
        "validation_sha256": PINNED_VALIDATION_R3_SHA256,
        "inspected_input_sha256": inspected,
        "training_gate_bundle_sha256": gate_digest,
        "nli_model_tree_sha256": PINNED_NLI_TREE_SHA256,
        "checkpoint_count": EXPECTED_CHECKPOINTS,
        "seed_count": EXPECTED_SEEDS,
        "seeds": list(seeds),
        "epoch_count_per_seed": len(EXPECTED_EPOCHS),
        "canary_contract": dict(canary_contract),
        "gate_receipts": {
            "nonblind_compare": dataset["strict_audit_gates"]["nonblind_compare"],
            "scoped_lexical": dataset["strict_audit_gates"]["scoped_lexical"],
            "train_unique_support": dataset["strict_audit_gates"]["unique_support"]["train"],
            "validation_unique_support": dataset["strict_audit_gates"]["unique_support"]["validation"],
        },
    }


def _select_recomputed_records_v8(
    *,
    declared_records: Sequence[Mapping[str, Any]],
    recomputed_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reject summaries/index records unless they equal raw recomputation."""

    if list(declared_records) != list(recomputed_records):
        raise SelectionFreezeV8Error("declared summary/records differ from independently recomputed evidence")
    try:
        decision = selection_policy_v6.select_checkpoint(list(recomputed_records))
    except selection_policy_v6.SelectionPolicyV6Error as exc:
        raise SelectionFreezeV8Error(
            f"deterministic v6 policy rejected recomputed v8 records: {exc}"
        ) from exc
    if (
        decision.get("status") != selection_policy_v6.SELECTED_STATUS
        or decision.get("selection_allowed") is not True
        or not isinstance(decision.get("selection"), Mapping)
    ):
        raise SelectionFreezeV8Error("deterministic selection policy returned HOLD")
    return decision


def _snapshot_declared_gate_v8(
    value: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    path = selection_freeze_v7._assert_unreserved_path(
        Path(str(value.get("path"))),
        label=label,
    )
    snapshot = selection_freeze_v7._stable_file_snapshot(path, label=label)
    expected_identity = _identity_receipt(snapshot.identity)
    if (
        value.get("bytes") != len(snapshot.payload)
        or value.get("sha256") != snapshot.sha256
        or value.get("stable_identity") != expected_identity
    ):
        raise SelectionFreezeV8Error(f"{label}: stable receipt mismatch")
    return dict(value)


def _validate_evaluation_index_identity_v8(
    index: Mapping[str, Any],
    *,
    expected_stage: str = "final",
) -> None:
    if expected_stage not in {"canary", "final"}:
        raise SelectionFreezeV8Error("v8 evaluation expected_stage must be canary or final")
    expected_status = (
        pointer_checkpoint_eval_v8.CANARY_STATUS
        if expected_stage == "canary"
        else pointer_checkpoint_eval_v8.FINAL_STATUS
    )
    if (
        index.get("schema") != pointer_checkpoint_eval_v8.INDEX_SCHEMA
        or index.get("orchestrator_version") != pointer_checkpoint_eval_v8.ORCHESTRATOR_VERSION
        or index.get("status") != expected_status
        or index.get("stage") != expected_stage
    ):
        raise SelectionFreezeV8Error(f"evaluation index is not a formal {expected_stage} v8 receipt")


def _load_canary_authority_v8(
    canary: Mapping[str, Any],
    *,
    dataset_root: Path,
    inspected_input_sha256: str,
    training_gate_bundle_sha256: str,
    preregistration_authority: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    canary = _require_mapping(
        canary,
        label="current v8 canary acceptance adapter",
        exact=_CANARY_SNAPSHOT_FIELDS_V8,
    )
    if (
        canary.get("required_for_stage") != "final"
        or canary.get("schema") != CANARY_SCHEMA
        or canary.get("gate_version") != CANARY_VERSION
        or canary.get("status") != CANARY_PASS_STATUS
        or canary.get("gate_passed") is not True
        or canary.get("next_action") != "START_FINAL_THREE_SEED_TRAINING"
    ):
        raise SelectionFreezeV8Error("v6/legacy canary acceptance cannot authorize strict v8")
    acceptance_path, acceptance_payload, acceptance = selection_freeze_v7._load_json(
        Path(str(canary["path"])),
        label="formal v8 canary acceptance receipt",
    )
    acceptance_snapshot = selection_freeze_v7._stable_file_snapshot(
        acceptance_path,
        label="v8 canary acceptance receipt",
    )
    if (
        acceptance_payload != acceptance_snapshot.payload
        or canary.get("bytes") != len(acceptance_snapshot.payload)
        or canary.get("sha256") != acceptance_snapshot.sha256
        or canary.get("stable_identity") != _identity_receipt(acceptance_snapshot.identity)
    ):
        raise SelectionFreezeV8Error("v8 canary acceptance stable binding mismatch")
    acceptance = _require_mapping(
        acceptance,
        label="formal v8 canary acceptance receipt",
        exact=_CANARY_RECEIPT_FIELDS_V8,
    )
    receipt_body = dict(acceptance)
    receipt_digest = receipt_body.pop("receipt_payload_sha256", None)
    formal = _require_mapping(
        acceptance.get("formal_v8_binding"),
        label="formal v8 canary generation binding",
        exact=_CANARY_FORMAL_BINDING_FIELDS_V8,
    )
    expected_formal = {
        "contract": "STRICT_NONBLIND_V8",
        "manifest_sha256": PINNED_MANIFEST_R3_SHA256,
        "train_sha256": PINNED_TRAIN_R3_SHA256,
        "validation_sha256": PINNED_VALIDATION_R3_SHA256,
        "training_gate_bundle_sha256": training_gate_bundle_sha256,
        "v8_inspected_input_sha256": inspected_input_sha256,
    }
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
    if (
        not _valid_sha256(receipt_digest)
        or receipt_digest != canonical_sha256(receipt_body)
        or acceptance.get("schema") != CANARY_SCHEMA
        or acceptance.get("gate_version") != CANARY_VERSION
        or acceptance.get("status") != CANARY_PASS_STATUS
        or acceptance.get("gate_passed") is not True
        or acceptance.get("next_action") != "START_FINAL_THREE_SEED_TRAINING"
        or acceptance.get("claim_boundary") != canary_acceptance_v8.CLAIM_BOUNDARY
        or formal != expected_formal
        or acceptance.get("authorization") != expected_authorization
    ):
        raise SelectionFreezeV8Error("formal v8 canary acceptance identity or generation binding mismatch")
    for field in (
        "schema",
        "gate_version",
        "status",
        "gate_passed",
        "next_action",
        "receipt_payload_sha256",
        "authorization",
        "claim_boundary",
        "formal_v8_binding",
    ):
        if canary.get(field) != acceptance.get(field):
            raise SelectionFreezeV8Error(f"normalized v8 canary field differs from receipt: {field}")
    if canary.get("required_for_stage") != "final":
        raise SelectionFreezeV8Error("formal v8 canary acceptance is not bound to final training")

    acceptance_input = _require_mapping(
        acceptance.get("input"),
        label="formal v8 canary acceptance input",
        exact={
            "evaluation_index",
            "canary_training_receipt",
            "dataset_artifacts_read",
            "checkpoint_artifacts_read",
            "checkpoint_run_receipts_read",
            "calibration_content_discovered",
            "calibration_content_read",
            "calibration_content_hashed",
            "blind_test_content_discovered",
            "blind_test_content_read",
            "blind_test_content_hashed",
        },
    )
    if acceptance_input.get("checkpoint_run_receipts_read") is not False or any(
        acceptance_input.get(field) is not False
        for field in (
            "calibration_content_discovered",
            "calibration_content_read",
            "calibration_content_hashed",
            "blind_test_content_discovered",
            "blind_test_content_read",
            "blind_test_content_hashed",
        )
    ):
        raise SelectionFreezeV8Error("formal v8 canary acceptance crossed calibration/blind boundary")

    evaluation_binding = _require_mapping(
        canary.get("evaluation_index"),
        label="v8 canary evaluation binding",
        exact={"path", "bytes", "sha256", "stable_identity"},
    )
    index_path, index_payload, index = selection_freeze_v7._load_json(
        Path(str(evaluation_binding["path"])),
        label="formal v8 canary evaluation index",
    )
    index_snapshot = selection_freeze_v7._stable_file_snapshot(
        index_path,
        label="formal v8 canary evaluation index stable snapshot",
    )
    if (
        index_payload != index_snapshot.payload
        or evaluation_binding.get("bytes") != len(index_payload)
        or evaluation_binding.get("sha256") != index_snapshot.sha256
        or evaluation_binding.get("stable_identity") != _identity_receipt(index_snapshot.identity)
        or acceptance_input.get("evaluation_index")
        != {
            "path": str(index_path),
            "bytes": len(index_payload),
            "sha256": index_snapshot.sha256,
        }
    ):
        raise SelectionFreezeV8Error("formal v8 canary evaluation index stable binding mismatch")
    _validate_evaluation_index_identity_v8(index, expected_stage="canary")

    canary_training_binding = _require_mapping(
        canary.get("canary_training_receipt"),
        label="v8 canary training binding",
        exact={
            "path",
            "bytes",
            "sha256",
            "stable_identity",
            "run_id",
        },
    )
    canary_path, canary_payload, canary_training = selection_freeze_v7._load_json(
        Path(str(canary_training_binding["path"])),
        label="v8 canary training receipt",
    )
    canary_snapshot = selection_freeze_v7._stable_file_snapshot(
        canary_path,
        label="v8 canary training receipt stable snapshot",
    )
    if (
        canary_payload != canary_snapshot.payload
        or canary_training_binding.get("bytes") != len(canary_payload)
        or canary_training_binding.get("sha256") != hashlib.sha256(canary_payload).hexdigest()
        or canary_training_binding.get("stable_identity") != _identity_receipt(canary_snapshot.identity)
        or canary_training_binding.get("run_id") != canary_training.get("run_id")
    ):
        raise SelectionFreezeV8Error("v8 canary training stable binding mismatch")
    if acceptance_input.get("canary_training_receipt") != {
        "path": str(canary_path),
        "bytes": len(canary_payload),
        "sha256": hashlib.sha256(canary_payload).hexdigest(),
        "run_id": canary_training.get("run_id"),
    }:
        raise SelectionFreezeV8Error("formal v8 acceptance/canary training binding mismatch")
    if (
        set(canary_training) != _CANARY_TRAINING_FIELDS_V8
        or canary_training.get("schema") != RUN_RECEIPT_SCHEMA
        or canary_training.get("trainer_version") != TRAINER_VERSION
        or canary_training.get("stage") != "canary"
        or canary_training.get("status") != "PASS_CANARY_SINGLE_SEED_ALL_EPOCHS_NOT_SELECTED"
        or canary_training.get("checkpoint_count") != 6
        or canary_training.get("v8_inspected_input_sha256") != inspected_input_sha256
        or canary_training.get("training_gate_bundle_sha256") != training_gate_bundle_sha256
    ):
        raise SelectionFreezeV8Error("v7/legacy canary training cannot authorize v8 final training")
    _validate_qlora_protocol_fields_v8c2(
        canary_training,
        authority={"preregistration": preregistration_authority},
        label="v8c2 canary training",
    )
    canary_configuration, canary_seeds = _validate_configuration_v8(
        canary_training.get("configuration"),
        expected_stage="canary",
    )
    if canary_seeds != CANARY_SEEDS_V8C2 or canary_training.get("configuration_sha256") != canonical_sha256(
        canary_configuration
    ):
        raise SelectionFreezeV8Error("v8c2 canary configuration digest mismatch")
    canary_seed_receipts = _require_sequence(
        canary_training.get("seeds"),
        label="v8c2 canary seed receipts",
    )
    if len(canary_seed_receipts) != 1:
        raise SelectionFreezeV8Error("v8c2 canary must contain one seed receipt")
    _validate_qlora_protocol_fields_v8c2(
        _require_mapping(
            canary_seed_receipts[0],
            label="v8c2 canary seed receipt",
        ),
        authority={"preregistration": preregistration_authority},
        label="v8c2 canary seed receipt",
    )
    canary_input = _require_mapping(
        canary_training.get("input_snapshot"),
        label="v8 canary training input",
    )
    canary_dataset = _require_mapping(
        canary_input.get("dataset"),
        label="v8 canary training dataset",
    )
    canary_source_files = _require_mapping(
        canary_input.get("source_files"),
        label="v8 canary training source inventory",
    )
    canary_base_model = _require_mapping(
        canary_input.get("base_model"),
        label="v8 canary training base model",
    )
    canary_attempt = _require_mapping(
        canary_training.get("canary_attempt"),
        label="v8c2 canary attempt receipt",
    )
    try:
        qlora_full_v6._validate_v8c2_canary_attempt_receipt(
            canary_attempt,
            run_id=str(canary_training["run_id"]),
            configuration_sha256=str(
                canary_training["configuration_sha256"]
            ),
            dataset_input_sha256=str(
                canary_dataset["inspected_input_sha256"]
            ),
            training_gate_bundle_sha256=str(
                canary_training["training_gate_bundle_sha256"]
            ),
            source_inventory_sha256=canonical_sha256(
                canary_source_files
            ),
            base_model_tree_sha256=str(
                canary_base_model["tree_sha256"]
            ),
        )
    except (qlora_full_v6.QLoRAV6Error, OSError, ValueError) as exc:
        raise SelectionFreezeV8Error(
            f"v8c2 canary attempt binding rejected: {exc}"
        ) from exc
    _validate_dataset_preregistration_v8c2(
        canary_dataset.get("v8c2_preregistration"),
        authority={"preregistration": preregistration_authority},
        label="v8c2 canary dataset preregistration",
    )
    canary_manifest = _require_mapping(
        canary_dataset.get("manifest"),
        label="v8 canary training manifest",
    )
    if (
        canary_dataset.get("contract") != "STRICT_NONBLIND_V8"
        or canary_dataset.get("inspected_input_sha256") != inspected_input_sha256
        or canary_dataset.get("training_gate_bundle_sha256") != training_gate_bundle_sha256
        or canary_manifest.get("schema") != MANIFEST_SCHEMA
        or canary_manifest.get("sha256") != PINNED_MANIFEST_R3_SHA256
    ):
        raise SelectionFreezeV8Error("v8 canary training authority does not match v8 final authority")
    try:
        strict_binding = pointer_checkpoint_eval_v8.verify_strict_nonblind_v8_binding(
            receipt=canary_training,
            receipt_path=canary_path,
            dataset_dir=dataset_root,
        )
    except pointer_checkpoint_eval_v8.PointerCheckpointEvalV8Error as exc:
        raise SelectionFreezeV8Error(f"formal v8 canary training generation binding rejected: {exc}") from exc
    if (
        strict_binding.get("manifest", {}).get("sha256") != PINNED_MANIFEST_R3_SHA256
        or strict_binding.get("train", {}).get("sha256") != PINNED_TRAIN_R3_SHA256
        or strict_binding.get("validation", {}).get("sha256") != PINNED_VALIDATION_R3_SHA256
        or strict_binding.get("training_gate_bundle_sha256") != training_gate_bundle_sha256
        or strict_binding.get("v8_inspected_input_sha256") != inspected_input_sha256
        or index.get("strict_nonblind_v8_binding") != strict_binding
    ):
        raise SelectionFreezeV8Error("formal v8 canary index/training generation binding mismatch")
    index_training = _require_mapping(
        index.get("training"),
        label="formal v8 canary index training binding",
    )
    if (
        Path(str(index_training.get("receipt_path"))).resolve(strict=True) != canary_path
        or index_training.get("receipt_sha256") != hashlib.sha256(canary_payload).hexdigest()
        or index_training.get("run_id") != canary_training.get("run_id")
        or index_training.get("checkpoint_count") != 6
        or index_training.get("contract") != "STRICT_NONBLIND_V8"
        or index_training.get("training_gate_bundle_sha256") != training_gate_bundle_sha256
        or index_training.get("v8_inspected_input_sha256") != inspected_input_sha256
    ):
        raise SelectionFreezeV8Error("formal v8 canary index is not bound to its training receipt")
    try:
        verified_canary = canary_acceptance_v8.verify_canary_acceptance_v8(
            acceptance_receipt_path=acceptance_path,
            evaluation_index_path=index_path,
            training_receipt_path=canary_path,
        )
    except Exception as exc:
        raise SelectionFreezeV8Error(f"formal v8 canary acceptance verification failed: {exc}") from exc
    if verified_canary != dict(canary):
        raise SelectionFreezeV8Error("formal v8 canary acceptance differs from verified normalized receipt")
    for snapshot, label in (
        (acceptance_snapshot, "formal v8 canary acceptance"),
        (index_snapshot, "formal v8 canary evaluation index"),
        (canary_snapshot, "formal v8 canary training receipt"),
    ):
        if (
            selection_freeze_v7._stable_file_snapshot(
                snapshot.path,
                label=f"{label} final recheck",
            )
            != snapshot
        ):
            raise SelectionFreezeV8Error(f"{label} changed during independent replay")
    return dict(canary), {
        **expected_formal,
        "profile": TRAINING_PROFILE_V8C2,
        "preregistration_protocol_id": PREREGISTRATION_PROTOCOL_V8C2,
        "preregistration_sha256": PINNED_PREREGISTRATION_SHA256_V8C2,
    }


def _load_dataset_authority_v8(
    dataset_dir: Path,
    dataset: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        root = selection_freeze_v7._assert_unreserved_path(
            dataset_dir,
            label="strict nonblind-v8 dataset",
        )
        root = root.resolve(strict=True)
    except (OSError, selection_freeze_v7.SelectionFreezeV7Error) as exc:
        raise SelectionFreezeV8Error(f"strict nonblind-v8 dataset path rejected: {exc}") from exc
    if not root.is_dir():
        raise SelectionFreezeV8Error("strict nonblind-v8 dataset must be a directory")
    manifest_path, manifest_payload, manifest_value = selection_freeze_v7._load_json(
        root / MANIFEST_NAME,
        label="strict nonblind-v8 manifest r3",
    )
    manifest_snapshot = selection_freeze_v7._stable_file_snapshot(
        manifest_path,
        label="strict nonblind-v8 manifest r3 stable snapshot",
    )
    if (
        manifest_payload != manifest_snapshot.payload
        or manifest_snapshot.sha256 != PINNED_MANIFEST_R3_SHA256
        or manifest_value.get("schema") != MANIFEST_SCHEMA
        or manifest_value.get("builder_version") != BUILDER_VERSION
        or manifest_value.get("dataset_schema") != DATASET_SCHEMA
        or manifest_value.get("status") != "NONBLIND_V8_BUILT_NLI_UNIQUE_SUPPORT_PREBLIND_COMMITTED"
    ):
        raise SelectionFreezeV8Error("dataset is not the pinned strict nonblind-v8 manifest r3 authority")
    declared_manifest = _require_mapping(
        dataset.get("manifest"),
        label="training dataset manifest authority",
    )
    expected_manifest = {
        "path": MANIFEST_NAME,
        "bytes": len(manifest_payload),
        "sha256": manifest_snapshot.sha256,
        "stable_identity": _identity_receipt(manifest_snapshot.identity),
        "schema": MANIFEST_SCHEMA,
        "dataset_schema": DATASET_SCHEMA,
        "builder_version": BUILDER_VERSION,
    }
    if declared_manifest != expected_manifest:
        raise SelectionFreezeV8Error("training receipt manifest binding differs from pinned r3 bytes")

    manifest_splits = _require_mapping(
        manifest_value.get("splits"),
        label="v8 manifest split inventory",
        exact={"train", "validation", "calibration"},
    )
    formal_split_receipts: dict[str, dict[str, Any]] = {}
    for split, expected_sha256, expected_count in (
        ("train", PINNED_TRAIN_R3_SHA256, 250),
        (
            "validation",
            PINNED_VALIDATION_R3_SHA256,
            EXPECTED_VALIDATION_ROWS,
        ),
    ):
        split_receipt = _require_mapping(
            manifest_splits.get(split),
            label=f"v8 manifest {split} split",
            exact={"path", "sha256", "bytes", "count"},
        )
        if (
            split_receipt.get("path") != f"{split}.jsonl"
            or split_receipt.get("sha256") != expected_sha256
            or split_receipt.get("count") != expected_count
            or isinstance(split_receipt.get("bytes"), bool)
            or not isinstance(split_receipt.get("bytes"), int)
            or int(split_receipt["bytes"]) < 1
        ):
            raise SelectionFreezeV8Error(f"v8 manifest formal {split} split mismatch")
        formal_split_receipts[split] = dict(split_receipt)

    commitment_decl = _require_mapping(
        _require_mapping(
            manifest_value.get("artifacts"),
            label="v8 manifest artifacts",
        ).get("preblind_commitment"),
        label="v8 manifest preblind commitment",
        exact={"path", "bytes", "sha256"},
    )
    if commitment_decl.get("path") != COMMITMENT_NAME:
        raise SelectionFreezeV8Error("v8 preblind commitment filename mismatch")
    commitment_path, commitment_payload, commitment_value = selection_freeze_v7._load_json(
        root / COMMITMENT_NAME,
        label="v8 preblind commitment",
    )
    commitment_snapshot = selection_freeze_v7._stable_file_snapshot(
        commitment_path,
        label="v8 preblind commitment stable snapshot",
    )
    if (
        commitment_payload != commitment_snapshot.payload
        or commitment_decl.get("bytes") != len(commitment_payload)
        or commitment_decl.get("sha256") != commitment_snapshot.sha256
    ):
        raise SelectionFreezeV8Error("v8 preblind commitment binding mismatch")

    gates = _require_mapping(
        dataset.get("strict_audit_gates"),
        label="v8 strict audit gates",
        exact={"nonblind_compare", "scoped_lexical", "unique_support"},
    )
    unique = _require_mapping(
        gates.get("unique_support"),
        label="v8 strict unique gates",
        exact={"train", "validation"},
    )
    stable_gates = {
        "nonblind_compare": _snapshot_declared_gate_v8(
            _require_mapping(
                gates["nonblind_compare"],
                label="v8 compare gate",
                exact=_COMPARE_GATE_FIELDS,
            ),
            label="v8 compare gate",
        ),
        "scoped_lexical": _snapshot_declared_gate_v8(
            _require_mapping(
                gates["scoped_lexical"],
                label="v8 lexical gate",
                exact=_LEXICAL_GATE_FIELDS,
            ),
            label="v8 lexical gate",
        ),
        "unique_support": {
            "train": _snapshot_declared_gate_v8(
                _require_mapping(
                    unique["train"],
                    label="v8 train unique gate",
                    exact=_UNIQUE_GATE_FIELDS,
                ),
                label="v8 train unique gate",
            ),
            "validation": _snapshot_declared_gate_v8(
                _require_mapping(
                    unique["validation"],
                    label="v8 validation unique gate",
                    exact=_UNIQUE_GATE_FIELDS,
                ),
                label="v8 validation unique gate",
            ),
        },
    }
    seed_revalidation = _require_mapping(
        dataset.get("seed_revalidation"),
        label="v8 seed revalidation",
        exact={"files", "nli_model"},
    )
    nli = _require_mapping(
        seed_revalidation.get("nli_model"),
        label="v8 NLI identity",
    )
    nli_root = selection_freeze_v7._assert_unreserved_path(
        Path(str(nli.get("root"))),
        label="v8 pinned NLI root",
    )
    try:
        validated_nli = semantic_queries_v7.validate_pinned_nli_asset(
            nli_root,
            expected_tree_sha256=PINNED_NLI_TREE_SHA256,
        )
    except Exception as exc:
        raise SelectionFreezeV8Error(f"v8 pinned NLI asset validation failed: {exc}") from exc
    if validated_nli.get("model_tree_sha256") != PINNED_NLI_TREE_SHA256:
        raise SelectionFreezeV8Error("v8 pinned NLI tree changed")
    nli_model_receipt = _require_mapping(
        nli.get("model_receipt"),
        label="v8 NLI model receipt",
    )
    nli_receipt_snapshot = selection_freeze_v7._stable_file_snapshot(
        selection_freeze_v7._assert_unreserved_path(
            Path(str(nli_model_receipt.get("path"))),
            label="v8 NLI model receipt",
        ),
        label="v8 NLI model receipt",
    )
    if (
        nli_model_receipt.get("bytes") != len(nli_receipt_snapshot.payload)
        or nli_model_receipt.get("sha256") != nli_receipt_snapshot.sha256
        or nli_model_receipt.get("stable_identity") != _identity_receipt(nli_receipt_snapshot.identity)
    ):
        raise SelectionFreezeV8Error("v8 NLI model receipt changed")
    implementation_receipts = dataset.get("implementation_receipts")
    current_implementations = qlora_full_v6._strict_implementation_snapshots_v8()
    expected_implementations = {
        role: {
            "path": str(snapshot.path),
            "bytes": snapshot.byte_count,
            "sha256": snapshot.sha256,
            "stable_identity": snapshot.identity_receipt(),
        }
        for role, snapshot in sorted(current_implementations.items())
    }
    if implementation_receipts != expected_implementations:
        raise SelectionFreezeV8Error("v8 dataset implementation tree changed after training")
    bundle = _require_mapping(
        dataset.get("training_gate_bundle"),
        label="v8 training gate bundle",
    )
    inspected = dataset.get("inspected_input_sha256")
    if not _valid_sha256(inspected):
        raise SelectionFreezeV8Error("v8 inspected input digest is invalid")
    authority = {
        "dataset_root": str(root),
        "manifest": expected_manifest,
        "train_sha256": formal_split_receipts["train"]["sha256"],
        "validation_sha256": (formal_split_receipts["validation"]["sha256"]),
        "formal_splits": formal_split_receipts,
        "inspected_input_sha256": inspected,
        "training_gate_bundle": dict(bundle),
        "training_gate_bundle_sha256": dataset.get("training_gate_bundle_sha256"),
        "strict_audit_gates": stable_gates,
        "nli_model_identity": {
            "root": str(nli_root.resolve(strict=True)),
            "tree_sha256": PINNED_NLI_TREE_SHA256,
            "model_receipt": dict(nli_model_receipt),
        },
        "implementation_receipts": expected_implementations,
    }
    return authority, {
        "manifest": expected_manifest,
        "preblind_commitment": {
            "path": COMMITMENT_NAME,
            "bytes": len(commitment_payload),
            "sha256": commitment_snapshot.sha256,
            "stable_identity": _identity_receipt(commitment_snapshot.identity),
            "schema": commitment_value.get("schema"),
            "commitment_sha256": commitment_value.get("commitment_sha256"),
        },
    }


def _implementation_bindings_v8() -> dict[str, Any]:
    paths = {
        "selection_freeze_v8": Path(__file__),
        "selection_freeze_v8_cli": (WORKSPACE_ROOT / "tools" / "freeze_icmat_llm_selection_v8.py"),
        "qlora_trainer": Path(qlora_full_v6.__file__),
        "checkpoint_orchestrator_v8": Path(pointer_checkpoint_eval_v8.__file__),
        "checkpoint_engine_v6": Path(pointer_checkpoint_eval_v6.__file__),
        "canary_acceptance_v8": Path(canary_acceptance_v8.__file__),
        "pointer_evaluator": Path(pointer_checkpoint_eval_v6.pointer_hf_eval_v6.__file__),
        "pointer_compiler": Path(pointer_checkpoint_eval_v6.evidence_pointer_v6.__file__),
        "selection_policy": Path(selection_policy_v6.__file__),
    }
    records: dict[str, Any] = {}
    for role, path in paths.items():
        snapshot = selection_freeze_v7._stable_file_snapshot(
            path,
            label=f"v8 selection implementation {role}",
        )
        records[role] = {
            "path": str(snapshot.path),
            "bytes": len(snapshot.payload),
            "sha256": snapshot.sha256,
            "stable_identity": _identity_receipt(snapshot.identity),
        }
    return records


def _evaluation_implementation_bindings_v8(
    value: Any,
) -> dict[str, dict[str, Any]]:
    records = _require_mapping(
        value,
        label="formal v8 evaluation implementation",
        exact={
            "orchestrator_v8",
            "evaluation_engine_v6",
            "pointer_evaluator",
            "pointer_compiler",
            "selection_policy",
            "runner_v8",
        },
    )
    expected_paths = {
        "orchestrator_v8": Path(pointer_checkpoint_eval_v8.__file__),
        "evaluation_engine_v6": Path(pointer_checkpoint_eval_v6.__file__),
        "pointer_evaluator": Path(pointer_checkpoint_eval_v8.pointer_hf_eval_v6.__file__),
        "pointer_compiler": Path(pointer_checkpoint_eval_v8.evidence_pointer_v6.__file__),
        "selection_policy": Path(selection_policy_v6.__file__),
        "runner_v8": Path(pointer_checkpoint_eval_v8._PRODUCTION_RUNNER_PATH),
    }
    verified: dict[str, dict[str, Any]] = {}
    for role, expected_path in expected_paths.items():
        record = _require_mapping(
            records.get(role),
            label=f"formal v8 evaluation implementation {role}",
            exact={"path", "sha256"},
        )
        declared = selection_freeze_v7._assert_unreserved_path(
            Path(str(record["path"])),
            label=f"formal v8 evaluation implementation {role}",
        )
        expected = expected_path.resolve(strict=True)
        snapshot = selection_freeze_v7._stable_file_snapshot(
            expected,
            label=f"formal v8 evaluation implementation {role}",
        )
        if declared.resolve(strict=True) != expected or record.get("sha256") != snapshot.sha256:
            raise SelectionFreezeV8Error(f"formal v8 evaluation implementation {role} mismatch")
        verified[role] = {
            "path": str(snapshot.path),
            "bytes": len(snapshot.payload),
            "sha256": snapshot.sha256,
            "stable_identity": _identity_receipt(snapshot.identity),
        }
    return verified


def _validate_evaluation_v8(
    path: Path,
    *,
    training_path: Path,
    training_payload: bytes,
    training: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
) -> tuple[Path, bytes, dict[str, Any], dict[str, Any]]:
    index_path, payload, index = selection_freeze_v7._load_json(
        path,
        label="formal v8 final evaluation index",
    )
    if index_path.name != "evaluation_index.v8.json":
        raise SelectionFreezeV8Error("formal v8 final evaluation filename must be evaluation_index.v8.json")
    index_snapshot = selection_freeze_v7._stable_file_snapshot(
        index_path,
        label="formal v8 final evaluation index stable snapshot",
    )
    if payload != index_snapshot.payload:
        raise SelectionFreezeV8Error("formal v8 final evaluation index changed while loading")
    index = _require_mapping(
        index,
        label="formal v8 final evaluation index",
        exact=_V8_INDEX_FIELDS,
    )
    _validate_evaluation_index_identity_v8(index)

    training_binding = _require_mapping(
        index.get("training"),
        label="formal v8 evaluation training binding",
        exact={
            "receipt_path",
            "receipt_sha256",
            "run_id",
            "checkpoint_count",
            "contract",
            "training_gate_bundle_sha256",
            "v8_inspected_input_sha256",
        },
    )
    dataset_root = Path(str(training["input_snapshot"]["dataset"]["path"])).resolve(strict=True)
    try:
        strict_binding = pointer_checkpoint_eval_v8.verify_strict_nonblind_v8_binding(
            receipt=training,
            receipt_path=training_path,
            dataset_dir=dataset_root,
        )
    except pointer_checkpoint_eval_v8.PointerCheckpointEvalV8Error as exc:
        raise SelectionFreezeV8Error(
            f"formal v8 final evaluation generation binding rejected: {exc}"
        ) from exc
    if (
        index.get("strict_nonblind_v8_binding") != strict_binding
        or strict_binding.get("manifest", {}).get("sha256") != PINNED_MANIFEST_R3_SHA256
        or strict_binding.get("train", {}).get("sha256") != PINNED_TRAIN_R3_SHA256
        or strict_binding.get("validation", {}).get("sha256") != PINNED_VALIDATION_R3_SHA256
        or strict_binding.get("training_gate_bundle_sha256") != PINNED_GATE_BUNDLE_R3_SHA256
        or training_binding.get("contract") != "STRICT_NONBLIND_V8"
        or training_binding.get("training_gate_bundle_sha256")
        != strict_binding["training_gate_bundle_sha256"]
        or training_binding.get("v8_inspected_input_sha256") != strict_binding["v8_inspected_input_sha256"]
    ):
        raise SelectionFreezeV8Error("formal v8 evaluation manifest/train/validation/gate binding mismatch")
    if (
        Path(str(training_binding.get("receipt_path"))).resolve(strict=True) != training_path
        or training_binding.get("receipt_sha256") != hashlib.sha256(training_payload).hexdigest()
        or training_binding.get("run_id") != training.get("run_id")
        or training_binding.get("checkpoint_count") != EXPECTED_CHECKPOINTS
    ):
        raise SelectionFreezeV8Error("formal v8 evaluation/training receipt binding mismatch")

    execution = _require_mapping(
        index.get("execution"),
        label="formal v8 evaluation execution",
        exact={
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
        },
    )
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
        raise SelectionFreezeV8Error("formal v8 evaluation execution boundary mismatch")

    dataset = _require_mapping(
        index.get("dataset"),
        label="formal v8 evaluation dataset",
        exact=_V8_INDEX_DATASET_FIELDS,
    )
    if (
        dataset.get("evaluated_rows_per_checkpoint") != EXPECTED_VALIDATION_ROWS
        or dataset.get("canary_selection") is not None
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
        raise SelectionFreezeV8Error("formal v8 evaluation dataset boundary mismatch")
    validation = strict_binding["validation"]
    expected_validation_path = str(dataset_root / "validation.jsonl")
    validation_snapshot = selection_freeze_v7._stable_file_snapshot(
        Path(expected_validation_path),
        label="formal v8 frozen validation",
    )
    if (
        validation_snapshot.sha256 != PINNED_VALIDATION_R3_SHA256
        or validation_snapshot.sha256 != validation["sha256"]
        or len(validation_snapshot.payload) != validation["bytes"]
        or dataset.get("path") != expected_validation_path
        or dataset.get("directory") != str(dataset_root)
        or dataset.get("evaluation_directory") != str(dataset_root)
        or dataset.get("sha256") != validation["sha256"]
        or dataset.get("bytes") != validation["bytes"]
        or dataset.get("examples") != EXPECTED_VALIDATION_ROWS
    ):
        raise SelectionFreezeV8Error("formal v8 evaluation validation bytes mismatch")
    try:
        validation_selection = pointer_checkpoint_eval_v8.pointer_hf_eval_v6.select_dataset(
            dataset_dir=dataset_root,
            split="validation",
            max_samples=None,
        )
    except Exception as exc:
        raise SelectionFreezeV8Error(f"formal v8 validation parsing failed: {exc}") from exc
    if (
        validation_selection.rows_total != EXPECTED_VALIDATION_ROWS
        or len(validation_selection.rows) != EXPECTED_VALIDATION_ROWS
        or validation_selection.split_sha256 != validation_snapshot.sha256
        or validation_selection.split_bytes != len(validation_snapshot.payload)
    ):
        raise SelectionFreezeV8Error("formal v8 validation selection contract mismatch")

    evaluation_base = _require_mapping(
        index.get("base_model"),
        label="formal v8 evaluation base model",
        exact={
            "directory",
            "training_tree_sha256",
            "evaluator_tree_sha256",
            "file_count",
            "bytes",
        },
    )
    if not _valid_sha256(evaluation_base.get("training_tree_sha256")) or not _valid_sha256(
        evaluation_base.get("evaluator_tree_sha256")
    ):
        raise SelectionFreezeV8Error("formal v8 evaluation base model digest is invalid")
    authorization = _require_mapping(
        index.get("authorization"),
        label="formal v8 evaluation authorization",
        exact=set(_FALSE_TRAINING_AUTHORIZATION),
    )
    if authorization != _FALSE_TRAINING_AUTHORIZATION:
        raise SelectionFreezeV8Error("formal v8 evaluation contains an authorization")
    selection = _require_mapping(
        index.get("selection"),
        label="formal v8 evaluation selection",
        exact={"performed", "selected_checkpoint_id", "required_next_step"},
    )
    if selection.get("performed") is not False or selection.get("selected_checkpoint_id") is not None:
        raise SelectionFreezeV8Error("formal v8 evaluation already selected a checkpoint")

    implementations = _evaluation_implementation_bindings_v8(index.get("implementation"))
    recomputation_implementations = {
        "pointer_evaluator": implementations["pointer_evaluator"],
        "pointer_compiler": implementations["pointer_compiler"],
        "runner": implementations["runner_v8"],
    }
    checkpoint_specs = {str(spec["checkpoint_id"]): spec for spec in specs}
    if len(checkpoint_specs) != EXPECTED_CHECKPOINTS:
        raise SelectionFreezeV8Error("formal v8 training checkpoint IDs are not unique")
    index_checkpoints = _require_sequence(
        index.get("checkpoints"),
        label="formal v8 evaluation checkpoints",
    )
    if len(index_checkpoints) != EXPECTED_CHECKPOINTS:
        raise SelectionFreezeV8Error("formal v8 evaluation checkpoint inventory is not 3x6")
    recomputed_records: list[dict[str, Any]] = []
    evaluation_evidence: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    for position, raw_item in enumerate(index_checkpoints):
        item = _require_mapping(
            raw_item,
            label=f"formal v8 evaluation checkpoints[{position}]",
            exact=_V8_INDEX_CHECKPOINT_FIELDS,
        )
        checkpoint_id = str(item.get("checkpoint_id"))
        if checkpoint_id in observed_ids or checkpoint_id not in checkpoint_specs:
            raise SelectionFreezeV8Error("formal v8 evaluation checkpoint population mismatch")
        observed_ids.add(checkpoint_id)
        spec = checkpoint_specs[checkpoint_id]
        expected_values = {
            "checkpoint_id": spec["checkpoint_id"],
            "seed": spec["seed"],
            "epoch": spec["epoch"],
            "global_step": spec["global_step"],
            "validation_loss": spec["validation_loss"],
            "checkpoint_path": str(spec["path"]),
            "receipt_relative_path": spec["receipt_path"],
            "training_checkpoint_tree_sha256": spec["training_checkpoint_tree_sha256"],
            "training_adapter_tree_sha256": spec["training_adapter_tree_sha256"],
            "evaluator_adapter_tree_sha256": spec["evaluator_adapter_tree_sha256"],
            "checkpoint_files": spec["checkpoint_files"],
            "checkpoint_bytes": spec["checkpoint_bytes"],
        }
        if any(item.get(field) != expected for field, expected in expected_values.items()):
            raise SelectionFreezeV8Error(f"{checkpoint_id} v8 evaluation/training binding mismatch")
        try:
            record, _, evidence = selection_freeze_v7._recompute_evaluation_record(
                checkpoint=item,
                spec=spec,
                expected_examples=EXPECTED_VALIDATION_ROWS,
                validation_selection=validation_selection,
                expected_base_tree=evaluation_base["evaluator_tree_sha256"],
                implementations=recomputation_implementations,
            )
        except selection_freeze_v7.SelectionFreezeV7Error as exc:
            raise SelectionFreezeV8Error(f"{checkpoint_id} v8 evidence recomputation failed: {exc}") from exc
        recomputed_records.append(record)
        evaluation_evidence.append(evidence)
    if observed_ids != set(checkpoint_specs):
        raise SelectionFreezeV8Error("formal v8 evaluation and training populations differ")
    declared_records = _require_sequence(
        index.get("records"),
        label="formal v8 evaluation records",
    )
    decision = _select_recomputed_records_v8(
        declared_records=declared_records,
        recomputed_records=recomputed_records,
    )
    selected_id = str(decision["selection"]["checkpoint_id"])
    if selected_id not in checkpoint_specs:
        raise SelectionFreezeV8Error("v8 selected checkpoint is absent from training inventory")
    selected_spec = dict(checkpoint_specs[selected_id])
    selected_spec.pop("stable_tree", None)

    strict_after = pointer_checkpoint_eval_v8.verify_strict_nonblind_v8_binding(
        receipt=training,
        receipt_path=training_path,
        dataset_dir=dataset_root,
    )
    if strict_after != strict_binding:
        raise SelectionFreezeV8Error("formal v8 generation authority changed during recomputation")
    if (
        _evaluation_implementation_bindings_v8(index.get("implementation")) != implementations
        or selection_freeze_v7._stable_file_snapshot(
            index_path,
            label="formal v8 final evaluation index final snapshot",
        )
        != index_snapshot
    ):
        raise SelectionFreezeV8Error("formal v8 evaluation authority changed during selection")
    return (
        index_path,
        payload,
        dict(index),
        {
            "decision": decision,
            "spec": selected_spec,
            "strict_nonblind_v8_binding": strict_binding,
            "evaluation_evidence": {
                "implementation": implementations,
                "checkpoints": evaluation_evidence,
                "recomputed_records_sha256": canonical_sha256(recomputed_records),
                "evidence_digest_sha256": canonical_sha256(
                    {
                        "implementation": implementations,
                        "checkpoints": evaluation_evidence,
                        "records": recomputed_records,
                    }
                ),
            },
        },
    )


def _snapshot_v8(
    *,
    evaluation_index_path: Path,
    training_receipt_path: Path,
    dataset_dir: Path,
    base_model_dir: Path,
) -> dict[str, Any]:
    try:
        training_path, training_payload, training = selection_freeze_v7._load_json(
            training_receipt_path,
            label="strict v8 final training receipt",
        )
        input_snapshot = _require_mapping(
            training.get("input_snapshot"),
            label="strict v8 final input snapshot",
        )
        dataset = _require_mapping(
            input_snapshot.get("dataset"),
            label="strict v8 final dataset",
        )
        authority, dataset_bindings = _load_dataset_authority_v8(
            dataset_dir,
            dataset,
        )
        preregistration = _load_preregistration_authority_v8c2()
        authority["preregistration"] = preregistration
        canary, canary_contract = _load_canary_authority_v8(
            _require_mapping(
                input_snapshot.get("canary_acceptance"),
                label="strict v8 final canary",
            ),
            dataset_root=Path(authority["dataset_root"]),
            inspected_input_sha256=str(authority["inspected_input_sha256"]),
            training_gate_bundle_sha256=str(authority["training_gate_bundle_sha256"]),
            preregistration_authority=preregistration,
        )
        authority["canary_acceptance"] = canary
        authority["canary_contract"] = canary_contract
        stage, specs = pointer_checkpoint_eval_v8.eval_v6._checkpoint_specs(
            receipt=training,
            training_root=training_path.parent,
        )
        if stage != "final":
            raise SelectionFreezeV8Error("strict v8 selection requires final training stage")
        adapted = _adapt_final_training_receipt_v8(
            training,
            authority=authority,
            checkpoint_specs=specs,
        )
        index_path, index_payload, index, selected = _validate_evaluation_v8(
            evaluation_index_path,
            training_path=training_path,
            training_payload=training_payload,
            training=training,
            specs=specs,
        )
        if (
            index.get("status") != EVALUATION_PASS_STATUS
            or len(index.get("records", ())) != EXPECTED_CHECKPOINTS
            or selected["strict_nonblind_v8_binding"].get("manifest", {}).get("sha256")
            != adapted["manifest_sha256"]
            or selected["strict_nonblind_v8_binding"].get("train", {}).get("sha256")
            != adapted["train_sha256"]
            or selected["strict_nonblind_v8_binding"].get("validation", {}).get("sha256")
            != adapted["validation_sha256"]
            or selected["strict_nonblind_v8_binding"].get("training_gate_bundle_sha256")
            != adapted["training_gate_bundle_sha256"]
        ):
            raise SelectionFreezeV8Error("strict v8 evaluation is not a complete 18-checkpoint run")
        model = selection_freeze_v7._model_binding(
            base_model_dir,
            training=training,
            index=index,
        )
        spec = selected["spec"]
        checkpoint_tree = selection_freeze_v7._stable_tree_snapshot(
            Path(spec["path"]),
            label="strict v8 selected checkpoint",
        )
        adapter_tree = selection_freeze_v7._selected_adapter_inventory(checkpoint_tree)
        if (
            checkpoint_tree.tree_sha256_casefold != spec["training_checkpoint_tree_sha256"]
            or adapter_tree["tree_sha256"] != spec["training_adapter_tree_sha256"]
        ):
            raise SelectionFreezeV8Error("strict v8 selected checkpoint changed after evaluation")
    except SelectionFreezeV8Error:
        raise
    except (
        OSError,
        ValueError,
        selection_freeze_v7.SelectionFreezeV7Error,
        pointer_checkpoint_eval_v6.PointerCheckpointEvalV6Error,
        pointer_checkpoint_eval_v8.PointerCheckpointEvalV8Error,
    ) as exc:
        raise SelectionFreezeV8Error(f"strict v8 authority validation failed: {exc}") from exc

    implementations = _implementation_bindings_v8()
    training_binding = _receipt_binding(
        training_path,
        training_payload,
        training,
    )
    evaluation_binding = _receipt_binding(
        index_path,
        index_payload,
        index,
    )
    selection = {
        "checkpoint_id": spec["checkpoint_id"],
        "seed": spec["seed"],
        "epoch": spec["epoch"],
        "global_step": spec["global_step"],
        "validation_loss": spec["validation_loss"],
        "checkpoint_path": str(checkpoint_tree.root),
        "checkpoint_tree_sha256": checkpoint_tree.tree_sha256_casefold,
        "checkpoint_file_count": checkpoint_tree.file_count,
        "checkpoint_bytes": checkpoint_tree.bytes,
        "adapter_tree_sha256": adapter_tree["tree_sha256"],
        "stable_tree_digest_sha256": canonical_sha256(
            {
                "directories": checkpoint_tree.directory_receipts,
                "records": checkpoint_tree.records_casefold,
            }
        ),
        "ranking_metrics": dict(selected["decision"]["selection"]["ranking_metrics"]),
        "qualified_seeds": list(selected["decision"]["qualified_seeds"]),
        "selection_locked": True,
    }
    authority_tree = {
        "profile": TRAINING_PROFILE_V8C2,
        "preregistration_protocol_id": PREREGISTRATION_PROTOCOL_V8C2,
        "preregistration_sha256": preregistration["sha256"],
        "configuration_sha256": adapted["configuration_sha256"],
        "manifest_sha256": PINNED_MANIFEST_R3_SHA256,
        "train_sha256": PINNED_TRAIN_R3_SHA256,
        "validation_sha256": PINNED_VALIDATION_R3_SHA256,
        "preblind_commitment_sha256": dataset_bindings["preblind_commitment"]["sha256"],
        "inspected_input_sha256": adapted["inspected_input_sha256"],
        "training_gate_bundle_sha256": adapted["training_gate_bundle_sha256"],
        "nli_model_tree_sha256": PINNED_NLI_TREE_SHA256,
        "gate_receipts": {
            role: {
                "sha256": value["sha256"],
                "status": value["status"],
            }
            for role, value in sorted(adapted["gate_receipts"].items())
        },
        "training_receipt_sha256": training_binding["sha256"],
        "evaluation_receipt_sha256": evaluation_binding["sha256"],
        "canary_acceptance_sha256": canary.get("sha256"),
        "base_model_tree_sha256": model["tree_sha256"],
        "selected_checkpoint_tree_sha256": selection["checkpoint_tree_sha256"],
        "training_implementations": {
            role: value["sha256"] for role, value in sorted(authority["implementation_receipts"].items())
        },
        "implementations": {role: value["sha256"] for role, value in sorted(implementations.items())},
    }
    return {
        **dataset_bindings,
        "preregistration": preregistration,
        "training_receipt": training_binding,
        "evaluation_receipt": evaluation_binding,
        "strict_v8_authority": {
            **adapted,
            "canary_acceptance_sha256": canary.get("sha256"),
            "implementation_receipts": authority["implementation_receipts"],
            "authority_tree": authority_tree,
            "authority_tree_sha256": canonical_sha256(authority_tree),
        },
        "evaluation_evidence": selected["evaluation_evidence"],
        "base_model": model,
        "selection_policy": {
            "schema": selection_policy_v6.SCHEMA,
            "version": selection_policy_v6.POLICY_VERSION,
            "decision": selected["decision"],
        },
        "selection": selection,
        "implementation": implementations,
        "authorization": _selection_authorization_v8(),
        "access_boundary": {
            "preregistration_opened": True,
            "manifest_opened": True,
            "preblind_commitment_opened": True,
            "training_receipt_opened": True,
            "evaluation_receipt_opened": True,
            "formal_v8_canary_acceptance_replayed": True,
            "all_6_canary_checkpoint_evaluations_recomputed": True,
            "formal_train_integrity_read_by_canary_replay": True,
            "all_18_checkpoint_evaluations_recomputed": True,
            "summary_metrics_trusted": False,
            "index_metrics_trusted": False,
            "calibration_path_constructed": False,
            "calibration_filesystem_metadata_accessed": False,
            "calibration_content_opened": False,
            "calibration_content_read": False,
            "calibration_content_hashed": False,
            "blind_path_constructed": False,
            "blind_filesystem_metadata_accessed": False,
            "blind_content_opened": False,
            "blind_content_read": False,
            "blind_content_hashed": False,
        },
    }


def _binding_payload_v8(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "profile": snapshot["strict_v8_authority"]["profile"],
        "preregistration_protocol_id": snapshot["strict_v8_authority"]["preregistration_protocol_id"],
        "preregistration_sha256": snapshot["preregistration"]["sha256"],
        "configuration_sha256": snapshot["strict_v8_authority"]["configuration_sha256"],
        "manifest_sha256": snapshot["manifest"]["sha256"],
        "train_sha256": snapshot["strict_v8_authority"]["train_sha256"],
        "validation_sha256": snapshot["strict_v8_authority"]["validation_sha256"],
        "preblind_commitment_sha256": snapshot["preblind_commitment"]["sha256"],
        "training_receipt_sha256": snapshot["training_receipt"]["sha256"],
        "evaluation_receipt_sha256": snapshot["evaluation_receipt"]["sha256"],
        "inspected_input_sha256": snapshot["strict_v8_authority"]["inspected_input_sha256"],
        "training_gate_bundle_sha256": snapshot["strict_v8_authority"]["training_gate_bundle_sha256"],
        "nli_model_tree_sha256": snapshot["strict_v8_authority"]["nli_model_tree_sha256"],
        "authority_tree_sha256": snapshot["strict_v8_authority"]["authority_tree_sha256"],
        "evaluation_evidence_sha256": snapshot["evaluation_evidence"]["evidence_digest_sha256"],
        "base_model_tree_sha256": snapshot["base_model"]["tree_sha256"],
        "selected_checkpoint_id": snapshot["selection"]["checkpoint_id"],
        "selected_checkpoint_tree_sha256": snapshot["selection"]["checkpoint_tree_sha256"],
        "selected_adapter_tree_sha256": snapshot["selection"]["adapter_tree_sha256"],
        "selection_policy_version": selection_policy_v6.POLICY_VERSION,
        "authorization": _selection_authorization_v8(),
    }


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


def _exclusive_write_v8(path: Path, payload: bytes) -> Path:
    try:
        lexical = selection_freeze_v7._assert_no_reparse_chain(
            path,
            label="strict v8 selection output",
        )
        lexical.parent.mkdir(parents=True, exist_ok=True)
        selection_freeze_v7._assert_no_reparse_chain(
            lexical.parent,
            label="strict v8 selection output parent",
        )
    except (OSError, selection_freeze_v7.SelectionFreezeV7Error) as exc:
        raise SelectionFreezeV8Error(f"strict v8 selection output path rejected: {exc}") from exc
    if os.path.lexists(lexical):
        raise SelectionFreezeV8Error(f"output already exists: {lexical}")
    descriptor = os.open(
        lexical,
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
    return lexical.resolve(strict=True)


def _cleanup_owned_v8(path: Path, *, expected_sha256: str) -> None:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return
    if (
        stat.S_ISLNK(metadata.st_mode)
        or selection_freeze_v7._is_reparse(metadata)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise SelectionFreezeV8Error(f"refusing to clean non-regular output: {path}")
    snapshot = selection_freeze_v7._stable_file_snapshot(
        path,
        label="strict v8 failed output cleanup",
    )
    if snapshot.sha256 != expected_sha256:
        raise SelectionFreezeV8Error("refusing to clean output whose bytes changed")
    path.unlink()


def create_selection_freeze_v8(
    *,
    evaluation_index_path: Path,
    training_receipt_path: Path,
    dataset_dir: Path,
    base_model_dir: Path,
    output_path: Path,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Create a validation-selected v8 freeze without calibration/blind I/O."""

    if os.path.lexists(output_path):
        raise SelectionFreezeV8Error(f"output already exists: {output_path}")
    if Path(output_path).name in {"", ".", ".."}:
        raise SelectionFreezeV8Error("output must name a new regular file")
    snapshot = _snapshot_v8(
        evaluation_index_path=evaluation_index_path,
        training_receipt_path=training_receipt_path,
        dataset_dir=dataset_dir,
        base_model_dir=base_model_dir,
    )
    if (
        _snapshot_v8(
            evaluation_index_path=evaluation_index_path,
            training_receipt_path=training_receipt_path,
            dataset_dir=dataset_dir,
            base_model_dir=base_model_dir,
        )
        != snapshot
    ):
        raise SelectionFreezeV8Error("strict v8 authority changed during selection")
    created = created_at_utc or datetime.now(UTC).isoformat()
    try:
        parsed = datetime.fromisoformat(created)
    except ValueError as exc:
        raise SelectionFreezeV8Error("created_at_utc must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise SelectionFreezeV8Error("created_at_utc must be UTC")
    body = {
        "schema": SCHEMA,
        "version": VERSION,
        "created_at_utc": created,
        "status": STATUS,
        "selection_locked": True,
        "selection_binding_digest_sha256": canonical_sha256(_binding_payload_v8(snapshot)),
        **snapshot,
        "claim_boundary": (
            "This receipt freezes one checkpoint selected only from 18 "
            "independently recomputed nonblind validation evaluations under "
            "the existing deterministic v6 policy. It authorizes only "
            "post-freeze full-split calibration and validation-only ablation. "
            "It does not authorize reserved blind evaluation, GGUF export, "
            "X5/BPU execution, deployment, or production integration."
        ),
    }
    receipt = {
        **body,
        "canonical_digest_sha256": canonical_sha256(body),
    }
    payload = _json_bytes(receipt)
    payload_sha = hashlib.sha256(payload).hexdigest()
    final = Path(output_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_parent, parent_identity = selection_freeze_v7._directory_identity(
            final.parent,
            label="strict v8 selection output parent",
        )
    except (OSError, selection_freeze_v7.SelectionFreezeV7Error) as exc:
        raise SelectionFreezeV8Error(f"strict v8 selection output parent rejected: {exc}") from exc
    final = output_parent / final.name
    staging = final.with_name(f".{final.name}.staging-{uuid4().hex}")
    published = False
    try:
        staged = _exclusive_write_v8(staging, payload)
        verify_selection_freeze_v8(
            freeze_receipt_path=staged,
            evaluation_index_path=evaluation_index_path,
            training_receipt_path=training_receipt_path,
            dataset_dir=dataset_dir,
            base_model_dir=base_model_dir,
        )
        if os.path.lexists(final):
            raise SelectionFreezeV8Error(f"output already exists: {final}")
        selection_freeze_v7._recheck_directory_identity(
            output_parent,
            parent_identity,
            label="strict v8 selection output parent before publication",
        )
        try:
            os.link(staged, final, follow_symlinks=False)
        except FileExistsError as exc:
            raise SelectionFreezeV8Error(f"output already exists: {final}") from exc
        published = True
        os.unlink(staged)
        final = final.resolve(strict=True)
        selection_freeze_v7._recheck_directory_identity(
            output_parent,
            parent_identity,
            label="strict v8 selection output parent after publication",
        )
        verification = verify_selection_freeze_v8(
            freeze_receipt_path=final,
            evaluation_index_path=evaluation_index_path,
            training_receipt_path=training_receipt_path,
            dataset_dir=dataset_dir,
            base_model_dir=base_model_dir,
        )
    except BaseException:
        if os.path.lexists(staging):
            _cleanup_owned_v8(staging, expected_sha256=payload_sha)
        if published and os.path.lexists(final):
            _cleanup_owned_v8(final, expected_sha256=payload_sha)
        raise
    return {
        "status": STATUS,
        "path": str(final),
        "sha256": payload_sha,
        "profile": TRAINING_PROFILE_V8C2,
        "preregistration_protocol_id": PREREGISTRATION_PROTOCOL_V8C2,
        "preregistration_sha256": PINNED_PREREGISTRATION_SHA256_V8C2,
        "selection_binding_digest_sha256": receipt["selection_binding_digest_sha256"],
        "selected_checkpoint_id": receipt["selection"]["checkpoint_id"],
        "selected_seed": receipt["selection"]["seed"],
        "selected_epoch": receipt["selection"]["epoch"],
        "verification": verification,
        "receipt": receipt,
    }


def verify_selection_freeze_v8(
    *,
    freeze_receipt_path: Path,
    evaluation_index_path: Path,
    training_receipt_path: Path,
    dataset_dir: Path,
    base_model_dir: Path,
) -> dict[str, Any]:
    """Recompute all v8 authorities without calibration or blind access."""

    try:
        freeze_path, _, receipt = selection_freeze_v7._load_json(
            freeze_receipt_path,
            label="strict v8 selection freeze",
        )
    except (
        OSError,
        ValueError,
        selection_freeze_v7.SelectionFreezeV7Error,
    ) as exc:
        raise SelectionFreezeV8Error(f"strict v8 selection receipt rejected: {exc}") from exc
    expected_fields = {
        "schema",
        "version",
        "created_at_utc",
        "status",
        "selection_locked",
        "selection_binding_digest_sha256",
        "preregistration",
        "manifest",
        "preblind_commitment",
        "training_receipt",
        "evaluation_receipt",
        "strict_v8_authority",
        "evaluation_evidence",
        "base_model",
        "selection_policy",
        "selection",
        "implementation",
        "authorization",
        "access_boundary",
        "claim_boundary",
        "canonical_digest_sha256",
    }
    _require_mapping(
        receipt,
        label="strict v8 selection freeze",
        exact=expected_fields,
    )
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("version") != VERSION
        or receipt.get("status") != STATUS
        or receipt.get("selection_locked") is not True
        or receipt.get("authorization") != _selection_authorization_v8()
    ):
        raise SelectionFreezeV8Error("strict v8 selection identity mismatch")
    body = dict(receipt)
    observed_digest = body.pop("canonical_digest_sha256", None)
    if not _valid_sha256(observed_digest) or observed_digest != canonical_sha256(body):
        raise SelectionFreezeV8Error("strict v8 selection canonical digest mismatch")
    expected = _snapshot_v8(
        evaluation_index_path=evaluation_index_path,
        training_receipt_path=training_receipt_path,
        dataset_dir=dataset_dir,
        base_model_dir=base_model_dir,
    )
    for field in (
        "preregistration",
        "manifest",
        "preblind_commitment",
        "training_receipt",
        "evaluation_receipt",
        "strict_v8_authority",
        "evaluation_evidence",
        "base_model",
        "selection_policy",
        "selection",
        "implementation",
        "authorization",
        "access_boundary",
    ):
        if receipt.get(field) != expected.get(field):
            raise SelectionFreezeV8Error(f"strict v8 selection verification failed: {field} changed")
    binding = canonical_sha256(_binding_payload_v8(expected))
    if receipt.get("selection_binding_digest_sha256") != binding:
        raise SelectionFreezeV8Error("strict v8 selection binding digest mismatch")
    return {
        "status": VERIFIED_STATUS,
        "freeze_path": str(freeze_path),
        "selection_locked": True,
        "selected_checkpoint_id": receipt["selection"]["checkpoint_id"],
        "selected_seed": receipt["selection"]["seed"],
        "selected_epoch": receipt["selection"]["epoch"],
        "selection_binding_digest_sha256": binding,
        "manifest_r3_sha256": PINNED_MANIFEST_R3_SHA256,
        "profile": TRAINING_PROFILE_V8C2,
        "preregistration_protocol_id": PREREGISTRATION_PROTOCOL_V8C2,
        "preregistration_sha256": PINNED_PREREGISTRATION_SHA256_V8C2,
        "configuration_sha256": receipt["strict_v8_authority"]["configuration_sha256"],
        "training_gate_bundle_sha256": receipt["strict_v8_authority"]["training_gate_bundle_sha256"],
        "nli_model_tree_sha256": PINNED_NLI_TREE_SHA256,
        "calibration_authorized": True,
        "ablation_authorized": True,
        "blind_test_authorized": False,
        "gguf_export_authorized": False,
        "x5_execution_authorized": False,
        "deployment_authorized": False,
        "production_integration_authorized": False,
        "calibration_filesystem_metadata_accessed": False,
        "blind_filesystem_metadata_accessed": False,
    }


__all__ = [
    "BUILDER_VERSION",
    "CANARY_SEEDS_V8C2",
    "CANARY_PASS_STATUS",
    "CANARY_SCHEMA",
    "CANARY_VERSION",
    "COMPARE_PASS_STATUS",
    "DATASET_SCHEMA",
    "EXPECTED_CHECKPOINTS",
    "EXPECTED_VALIDATION_ROWS",
    "FINAL_SEEDS_V8C2",
    "LEXICAL_PASS_STATUS",
    "MANIFEST_NAME",
    "MANIFEST_SCHEMA",
    "PINNED_COMPARE_R3_SHA256",
    "PINNED_GATE_BUNDLE_R3_SHA256",
    "PINNED_LEXICAL_R3_SHA256",
    "PINNED_MANIFEST_R3_SHA256",
    "PINNED_NLI_TREE_SHA256",
    "PINNED_PREREGISTRATION_SHA256_V8C2",
    "PINNED_TRAIN_R3_SHA256",
    "PINNED_TRAIN_UNIQUE_R3_SHA256",
    "PINNED_VALIDATION_R3_SHA256",
    "PINNED_VALIDATION_UNIQUE_R3_SHA256",
    "PREREGISTRATION_PATH_V8C2",
    "PREREGISTRATION_PROTOCOL_V8C2",
    "PREREGISTRATION_SCHEMA_V8C2",
    "PREREGISTRATION_STATUS_V8C2",
    "RUN_RECEIPT_SCHEMA",
    "SCHEMA",
    "STATUS",
    "TRAINER_VERSION",
    "TRAINING_PROFILE_V8C2",
    "TRAINING_PASS_STATUS",
    "UNIQUE_SUPPORT_PASS_STATUS",
    "VERIFIED_STATUS",
    "VERSION",
    "SelectionFreezeV8Error",
    "canonical_json",
    "canonical_sha256",
    "create_selection_freeze_v8",
    "_load_preregistration_authority_v8c2",
    "verify_selection_freeze_v8",
]
