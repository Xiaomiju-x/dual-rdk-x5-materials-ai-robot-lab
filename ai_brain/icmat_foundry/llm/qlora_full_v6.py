"""Isolated NF4 QLoRA trainer for the ICMat pointer contracts.

This module deliberately has a narrower data boundary than the dataset
builder. Legacy v6 opens and hashes ``manifest.v6.json``, ``train.jsonl`` and
``validation.jsonl``. Semantic-v7 datasets additionally require their local
semantic inventory audit artifact to be hash-verified. Legacy calibration and
sealed blind declarations are copied from the manifest without touching their
files.

The strict nonblind-v7 contract opens stable snapshots of its fixed ten-file
inventory and validates its original split-specific shortcut gates. The
parallel strict nonblind-v8 contract opens its fixed twelve-file inventory,
directly compares an independent second build, locally recomputes the scoped
train/validation lexical gate, and locally recomputes the two split-specific
unique-support NLI gates once on the fixed CPU backend. Calibration bytes are
used only for double-build integrity validation and never become training or
selection input. No blind artifact name or path is discovered or constructed.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import re
import stat
import subprocess
import sys
import time
import traceback
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from functools import cmp_to_key
from pathlib import Path
from statistics import mean
from typing import Any

from icmat_foundry.llm import (
    evidence_sft_v6 as evidence_contract,
)
from icmat_foundry.llm import semantic_queries_v7 as semantic_contract_v7
from icmat_foundry.llm import (
    shortcut_audit_v7 as shortcut_contract,
)
from icmat_foundry.llm import (
    shortcut_audit_v8 as shortcut_contract_v8,
)
from icmat_foundry.llm import (
    unique_support_audit_v8 as unique_support_contract_v8,
)

TRAINER_VERSION = "icmat-qwen05b-nf4-qlora-pointer-v6.0.0"
PREFLIGHT_SCHEMA = "icmat_qlora_pointer_preflight.v6"
SEED_RECEIPT_SCHEMA = "icmat_qlora_pointer_seed_receipt.v6"
RUN_RECEIPT_SCHEMA = "icmat_qlora_pointer_run_receipt.v6"
FAILURE_RECEIPT_SCHEMA = "icmat_qlora_pointer_failure_receipt.v6"

MANIFEST_NAME = "manifest.v6.json"
MANIFEST_SCHEMA = "icmat_evidence_pointer_manifest.v6"
NONBLIND_MANIFEST_NAME = "manifest.nonblind.v7.json"
NONBLIND_MANIFEST_SCHEMA = "icmat_evidence_pointer_nonblind_manifest.v7"
NONBLIND_V8_MANIFEST_NAME = "manifest.nonblind.v8.json"
NONBLIND_V8_MANIFEST_SCHEMA = (
    "icmat_evidence_pointer_nonblind_manifest.v8"
)
DATASET_SCHEMA = "icmat_qwen05b_evidence_pointer_sft.v6"
EXAMPLE_SCHEMA = "icmat_evidence_pointer_example.v6"
EXPECTED_BUILDER_VERSION = "icmat-evidence-sft-v6.0.0"
SEMANTIC_BUILDER_VERSION = "icmat-evidence-sft-v6.1.0-semantic-v7"
NONBLIND_BUILDER_VERSION = "icmat-evidence-nonblind-v7.1.0"
NONBLIND_SPLIT_ALGORITHM_VERSION = (
    "icmat-semantic-v7-nonblind-split-v1"
)
NONBLIND_PRECOMMIT_SCHEMA = (
    "icmat_evidence_pointer_preblind_commitment.v7"
)
NONBLIND_BALANCE_SCHEMA = (
    "icmat_evidence_pointer_nonblind_balance_audit.v7"
)
NONBLIND_GROUP_SCHEMA = (
    "icmat_evidence_pointer_nonblind_group_audit.v7"
)
NONBLIND_LEAKAGE_SCHEMA = (
    "icmat_evidence_pointer_nonblind_leakage_audit.v7"
)
NONBLIND_BUILD_REPORT_SCHEMA = (
    "icmat_evidence_pointer_nonblind_build_report.v7"
)
NONBLIND_AUDIT_SCHEMA = (
    "icmat_evidence_pointer_nonblind_independent_audit.v7"
)
NONBLIND_AUDIT_VERSION = (
    "icmat-evidence-nonblind-independent-audit-v7.1.0"
)
NONBLIND_COMPARE_STATUS = (
    "PASS_NONBLIND_V7_DOUBLE_BUILD_BYTE_IDENTICAL"
)
NONBLIND_V8_BUILDER_VERSION = "icmat-evidence-nonblind-v8.0.0"
NONBLIND_V8_COMPARE_STATUS = (
    "PASS_NONBLIND_V8_DOUBLE_BUILD_BYTE_IDENTICAL"
)
NONBLIND_V8_SPLIT_ALGORITHM_VERSION = (
    "icmat-semantic-v7-nonblind-split-v1"
)
NONBLIND_V8_REPAIR_POLICY_VERSION = (
    "icmat-answer-unique-support-nli-v8.0.0"
)
SHORTCUT_AUDIT_SCHEMA = "icmat_semantic_shortcut_audit.v7"
SHORTCUT_AUDIT_VERSION = "icmat-semantic-shortcut-audit-v7.0.0"
SHORTCUT_PASS_STATUS = "PASS_NO_USABLE_LEXICAL_SHORTCUT_FOUND"
SHORTCUT_SAMPLE_SCHEMA = "icmat_semantic_shortcut_sample.v7"
CANARY_ACCEPTANCE_SCHEMA = "icmat_llm_canary_acceptance_receipt.v6"
CANARY_ACCEPTANCE_VERSION = "icmat-llm-canary-acceptance-v6.1.0"
CANARY_ACCEPTANCE_STATUS = (
    "PASS_CANARY_ACCEPTED_FOR_THREE_SEED_TRAINING_ONLY"
)
CANARY_EVALUATION_INDEX_SCHEMA = (
    "icmat_pointer_checkpoint_evaluation_index.v6"
)
CANARY_EVALUATION_INDEX_STATUS = (
    "PASS_CANARY_1X6_VALIDATION_EVALUATED_NO_SELECTION"
)
CANARY_ACCEPTANCE_CLAIM_BOUNDARY = (
    "This receipt independently recomputes one non-blind 1x6 canary and may "
    "authorize only the start of final three-seed training. The reference "
    "checkpoint is not a final model selection. Calibration, blind "
    "evaluation, GGUF export, X5 deployment, and production integration "
    "remain unauthorized."
)
SUPPORTED_BUILDER_VERSIONS = frozenset(
    {EXPECTED_BUILDER_VERSION, SEMANTIC_BUILDER_VERSION}
)
SEMANTIC_RECORD_SCHEMA = "icmat_semantic_query_record.v7"
SEMANTIC_AUDIT_SCHEMA = "icmat_semantic_inventory_audit.v7"
SEMANTIC_ACCEPTED_INVENTORY_SCHEMA = (
    "icmat_semantic_query_accepted_inventory.v7"
)
SEMANTIC_SPLIT_COUNTS = {
    "train": 250,
    "validation": 150,
    "calibration": 150,
    "blind_test": 150,
}
NONBLIND_SPLIT_COUNTS = {
    "train": 250,
    "validation": 150,
    "calibration": 150,
}
NONBLIND_TOTAL_EXAMPLES = 550
NONBLIND_FAMILY_COUNT = 11
NONBLIND_EXAMPLES_PER_FAMILY = 50
EXPECTED_FUTURE_BLIND_COUNT = 150
STRICT_ARTIFACT_FILES = {
    "balance_audit": "balance_audit.nonblind.v7.json",
    "group_isolation_audit": "group_isolation_audit.nonblind.v7.json",
    "content_leakage_audit": "content_leakage_audit.nonblind.v7.json",
    "semantic_inventory_audit": "semantic_inventory_audit.v7.json",
    "preblind_commitment": "preblind_commitment.v7.json",
    "build_report": "build_report.nonblind.v7.json",
}
STRICT_V8_ARTIFACT_FILES = {
    "balance_audit": "balance_audit.nonblind.v8.json",
    "group_isolation_audit": "group_isolation_audit.nonblind.v8.json",
    "content_leakage_audit": "content_leakage_audit.nonblind.v8.json",
    "semantic_binding_audit": "semantic_binding_audit.v8.json",
    "nli_unique_support_audit": "nli_unique_support_audit.v8.json",
    "repair_manifest": "repair_manifest.v8.json",
    "preblind_commitment": "preblind_commitment.v8.json",
    "build_report": "build_report.nonblind.v8.json",
}

SPLIT_FILES = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "calibration": "calibration.jsonl",
    "blind_test": "blind_test.sealed.v6.jsonl",
}
NONBLIND_SPLIT_FILES = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "calibration": "calibration.jsonl",
}
NONBLIND_COMPARE_INVENTORY_FILES = {
    **NONBLIND_SPLIT_FILES,
    **STRICT_ARTIFACT_FILES,
    "manifest": NONBLIND_MANIFEST_NAME,
}
NONBLIND_V8_COMPARE_INVENTORY_FILES = {
    **NONBLIND_SPLIT_FILES,
    **STRICT_V8_ARTIFACT_FILES,
    "manifest": NONBLIND_V8_MANIFEST_NAME,
}
READABLE_SPLITS = ("train", "validation")
DECLARATION_ONLY_SPLITS = ("calibration", "blind_test")
POINTER_FIELDS = ("task", "decision", "span_id")
ALLOWED_TASKS = frozenset(
    {"claim_verification", "evidence_selection", "claim_extraction"}
)
ALLOWED_DECISIONS = frozenset({"ANSWER", "REFUSE"})
SPAN_ID_RE = re.compile(r"^E[1-9][0-9]*\.S[1-9][0-9]*$")
_STRICT_MAX_JSON_BYTES = 32 * 1024 * 1024
_STRICT_MAX_JSONL_BYTES = 128 * 1024 * 1024
_STRICT_READ_BLOCK_BYTES = 1024 * 1024

CANARY_DEFAULT_SEEDS = (20260728,)
FINAL_DEFAULT_SEEDS = (20260729, 20260730, 20260731)
FIXED_EPOCHS = 6
DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
EXPECTED_MODEL_CONFIG = {
    "model_type": "qwen2",
    "hidden_size": 896,
    "intermediate_size": 4864,
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "vocab_size": 151936,
}
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
V8C2_TRAINING_PROFILE = "V8C2_CAPACITY_REGULARIZED"
V8C2_PREREGISTRATION_PROTOCOL_ID = "ICMAT-Pointer-v8c2-PREREG-r1"
V8C2_PREREGISTRATION_SHA256 = (
    "955165d8e9766300e621fe6a1291e4a2ff1dd96a85692738cfe66f28a1b03c24"
)
V8C2_PREREGISTRATION_PATH = (
    WORKSPACE_ROOT
    / "docs"
    / "ai_brain_finals_20260728"
    / "ICMAT_POINTER_V8C2_PREREGISTRATION.json"
)
V8C2_CANONICAL_CANARY_OUTPUT_PATH = (
    WORKSPACE_ROOT / "evaluation" / "icmat_foundry" / "llm" / "v8c2"
)
V8C2_CANARY_ATTEMPT_PATH = (
    WORKSPACE_ROOT
    / "evaluation"
    / "icmat_foundry"
    / "llm"
    / "v8c2.canary_attempt.v1.json"
)
V8C2_CANARY_ATTEMPT_SCHEMA = "icmat_v8c2_canary_attempt.v1"
V8C2_CANARY_ATTEMPT_STATUS = "V8C2_CANARY_ATTEMPT_RESERVED"
V8C2_PREDECESSOR_ACCEPTANCE_PATH = (
    WORKSPACE_ROOT / "evaluation" / "icmat_foundry" / "llm" / "v8ca2.json"
)
V8C2_PREDECESSOR_ACCEPTANCE_SHA256 = (
    "ba02b07c95b70e3302dbe6a5431e4800c8ced8a5ddffff7242378da784107c5a"
)
V8C2_PREDECESSOR_STOP_STATUS = "STOP_V8_CANARY_NOT_ACCEPTED"
_LEGACY_PROFILE_INPUT = {
    "learning_rate": 2.0e-4,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
}
_V8C2_PROFILE_VALUES = {
    "learning_rate": 2.0e-4,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.10,
}
_V8C2_FIXED_ALGORITHM = {
    "profile": V8C2_TRAINING_PROFILE,
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
    "target_modules": list(DEFAULT_TARGET_MODULES),
    "data_resampling": False,
    "data_augmentation": False,
    "class_or_task_weighting": False,
    "layer_freezing": False,
    "checkpoint_interpolation": False,
    "checkpoint_voting": False,
    "nli_answer_override": False,
    "inference_contract_changed": False,
}
_V8C2_REQUIRED_HARDENING = {
    "v8_final_acceptance_dispatch": True,
    "v8_source_inventory_isolation": True,
    "v6_or_v7_acceptance_fallback_for_v8": False,
    "legacy_and_v7_behavior_must_remain_unchanged": True,
    "source_tree_must_be_frozen_before_canary": True,
    "all_six_epoch_checkpoints_must_be_retained": True,
}
_V8C2_FIXED_RUNS = {
    "canary_seeds": list(CANARY_DEFAULT_SEEDS),
    "final_seeds": list(FINAL_DEFAULT_SEEDS),
    "canary_runs_allowed": 1,
    "final_runs_allowed_after_canary_pass": 3,
    "seed_substitution_allowed": False,
    "additional_v8c2_variants_allowed": False,
}
_V8C2_FROZEN_DATA = {
    "dataset_contract": "STRICT_NONBLIND_V8",
    "dataset_manifest_sha256": (
        "7e2d9e2ab1bc380e1fb626e960a015b7c22c82b4c4c86d1f0c2c1e54b79c2535"
    ),
    "train_sha256": (
        "674ea8cf77b2d61eac31a76d8b0c6af8178b0da93b0d6af6c7b3bb75d95a821c"
    ),
    "validation_sha256": (
        "1ad3013670f90178e0372f1425b30cd867c38ef11551fc0cfb0f6c4e099becf4"
    ),
    "calibration_sha256": (
        "1320eb15e19f6795ff46492c4c004c5720f31d89c7f91ae6ae3334758ab8d4ab"
    ),
    "training_gate_bundle_sha256": (
        "5d9f8e2b0a30a5a50c8ed484d7445eb34c6229f9bbce8231265fe9f6364c2b0a"
    ),
    "train_rows": 250,
    "validation_rows": 150,
    "calibration_rows": 150,
    "training_data_changes_allowed": False,
    "validation_feedback_into_training_allowed": False,
    "calibration_for_training_or_selection_allowed": False,
    "sealed_blind_access_allowed_before_postfreeze": False,
}
_V8C2_CANARY_GATE = {
    "completed_samples": 18,
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
    "pass_rule": "AT_LEAST_ONE_OF_SIX_CHECKPOINTS_PASSES_ALL_GATES",
}
_V8C2_STOP_RULES = [
    (
        "If the one registered v8c2 canary does not pass, preserve STOP and "
        "do not create v8c2b or v8c2c."
    ),
    (
        "Do not lower thresholds, replace seeds, alter validation membership, "
        "or select an unregistered artifact."
    ),
    (
        "Do not start final three-seed training without an immutable PASS v8 "
        "canary acceptance receipt."
    ),
    (
        "Do not access the sealed blind before selection, calibration, "
        "ablation, native-v8 postfreeze verification, and GGUF preflight all "
        "pass."
    ),
    (
        "Any code, data, configuration, threshold, seed, or generation-policy "
        "change after canary start invalidates this protocol."
    ),
]


class QLoRAV6Error(ValueError):
    """Raised when the immutable v6 training contract is violated."""


@dataclass(frozen=True)
class StableFileSnapshotV7:
    path: Path
    payload: bytes
    sha256: str
    byte_count: int
    identity: tuple[int, int, int, int, int]

    def identity_receipt(self) -> dict[str, int]:
        return {
            "device": self.identity[0],
            "file_id": self.identity[1],
            "size": self.identity[2],
            "mtime_ns": self.identity[3],
            "ctime_ns": self.identity[4],
        }


@dataclass(frozen=True)
class StableModelFileV7:
    path: Path
    relative_path: str
    sha256: str
    byte_count: int
    identity: tuple[int, int, int, int, int]
    config_payload: bytes | None = None


@dataclass(frozen=True)
class StableModelDirectoryV7:
    path: Path
    relative_path: str
    identity: tuple[int, int, int, int, int]
    entries: tuple[str, ...]


@dataclass(frozen=True)
class StableModelTreeV7:
    root: Path
    files: tuple[StableModelFileV7, ...]
    directories: tuple[StableModelDirectoryV7, ...]
    tree_sha256: str
    stable_identity_sha256: str
    byte_count: int

    def inventory_receipt(self) -> dict[str, Any]:
        records = [
            {
                "path": item.relative_path,
                "bytes": item.byte_count,
                "sha256": item.sha256,
            }
            for item in self.files
        ]
        return {
            "files": records,
            "tree_sha256": self.tree_sha256,
            "file_count": len(records),
            "bytes": self.byte_count,
        }


@dataclass(frozen=True)
class QLoRATrainingConfigV6:
    """Frozen algorithm and resource configuration for one v6 run."""

    stage: str = "final"
    seeds: tuple[int, ...] = ()
    num_train_epochs: int = FIXED_EPOCHS
    max_seq_length: int = 1152
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2.0e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    minimum_free_vram_mib: int = 3600

    @property
    def resolved_seeds(self) -> tuple[int, ...]:
        if self.seeds:
            return self.seeds
        if self.stage == "canary":
            return CANARY_DEFAULT_SEEDS
        return FINAL_DEFAULT_SEEDS

    def validate(self) -> None:
        if self.stage not in {"canary", "final"}:
            raise QLoRAV6Error("stage must be canary or final")
        if self.num_train_epochs != FIXED_EPOCHS:
            raise QLoRAV6Error("v6 training is fixed to exactly 6 epochs")
        if self.lora_rank != 16:
            raise QLoRAV6Error("v6 training is fixed to LoRA rank 16")
        if not isinstance(self.seeds, tuple):
            raise TypeError("seeds must be a tuple")
        seeds = self.resolved_seeds
        expected_seed_count = 1 if self.stage == "canary" else 3
        if len(seeds) != expected_seed_count:
            raise QLoRAV6Error(
                f"{self.stage} stage requires exactly {expected_seed_count} seed(s)"
            )
        if len(set(seeds)) != len(seeds):
            raise QLoRAV6Error("training seeds must be unique")
        if any(
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed <= 2_147_483_647
            for seed in seeds
        ):
            raise QLoRAV6Error(
                "every seed must be an integer in [0, 2147483647]"
            )

        integer_fields = {
            "max_seq_length": self.max_seq_length,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "lora_alpha": self.lora_alpha,
            "minimum_free_vram_mib": self.minimum_free_vram_mib,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in integer_fields.values()
        ):
            raise TypeError("integer QLoRA configuration fields must be integers")
        if not 128 <= self.max_seq_length <= 32768:
            raise QLoRAV6Error("max_seq_length must be in [128, 32768]")
        if not 1 <= self.per_device_train_batch_size <= 8:
            raise QLoRAV6Error("train batch size must be in [1, 8]")
        if not 1 <= self.per_device_eval_batch_size <= 8:
            raise QLoRAV6Error("evaluation batch size must be in [1, 8]")
        if not 1 <= self.gradient_accumulation_steps <= 256:
            raise QLoRAV6Error("gradient accumulation must be in [1, 256]")
        if not 1 <= self.lora_alpha <= 512:
            raise QLoRAV6Error("lora_alpha must be in [1, 512]")
        if not 1024 <= self.minimum_free_vram_mib <= 131072:
            raise QLoRAV6Error(
                "minimum_free_vram_mib must be in [1024, 131072]"
            )

        float_fields = {
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "lora_dropout": self.lora_dropout,
        }
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in float_fields.values()
        ):
            raise QLoRAV6Error(
                "floating-point QLoRA configuration fields must be finite"
            )
        if not 0.0 < self.learning_rate <= 0.01:
            raise QLoRAV6Error("learning_rate must be in (0, 0.01]")
        if not 0.0 <= self.warmup_ratio <= 0.5:
            raise QLoRAV6Error("warmup_ratio must be in [0, 0.5]")
        if not 0.0 <= self.weight_decay <= 1.0:
            raise QLoRAV6Error("weight_decay must be in [0, 1]")
        if not 0.0 <= self.lora_dropout < 0.5:
            raise QLoRAV6Error("lora_dropout must be in [0, 0.5)")


def _effective_training_config_v8c2(
    config: QLoRATrainingConfigV6,
) -> QLoRATrainingConfigV6:
    requested = {
        "num_train_epochs": config.num_train_epochs,
        "max_seq_length": config.max_seq_length,
        "per_device_train_batch_size": (
            config.per_device_train_batch_size
        ),
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
    }
    expected = {
        "num_train_epochs": 6,
        "max_seq_length": 1152,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "warmup_ratio": 0.05,
        "weight_decay": 0.0,
        **_LEGACY_PROFILE_INPUT,
    }
    expected_seeds = (
        CANARY_DEFAULT_SEEDS
        if config.stage == "canary"
        else FINAL_DEFAULT_SEEDS
    )
    if requested != expected or config.resolved_seeds != expected_seeds:
        raise QLoRAV6Error(
            "v8c2 training profile cannot be overridden"
        )
    return replace(config, **_V8C2_PROFILE_VALUES)


def _v8c2_receipt_fields() -> dict[str, str]:
    return {
        "training_profile": V8C2_TRAINING_PROFILE,
        "preregistration_protocol_id": V8C2_PREREGISTRATION_PROTOCOL_ID,
        "preregistration_sha256": V8C2_PREREGISTRATION_SHA256,
    }


def _validate_v8c2_predecessor(
    declaration: Mapping[str, Any],
) -> dict[str, Any]:
    expected_declaration = {
        "canary_id": "v8c1",
        "acceptance_receipt": "evaluation/icmat_foundry/llm/v8ca2.json",
        "acceptance_receipt_sha256": (
            V8C2_PREDECESSOR_ACCEPTANCE_SHA256
        ),
        "status": V8C2_PREDECESSOR_STOP_STATUS,
        "best_compiled_strict_exact": "17/18",
        "best_answer_span_exact": "8/9",
        "permanently_superseded": False,
        "may_be_reissued_as_pass": False,
    }
    if dict(declaration) != expected_declaration:
        raise QLoRAV6Error("v8c2 predecessor declaration mismatch")
    snapshot = _stable_snapshot_v7(
        V8C2_PREDECESSOR_ACCEPTANCE_PATH,
        label="v8c2 predecessor STOP receipt",
        maximum_bytes=_STRICT_MAX_JSON_BYTES,
    )
    if snapshot.sha256 != V8C2_PREDECESSOR_ACCEPTANCE_SHA256:
        raise QLoRAV6Error("v8c2 predecessor receipt SHA-256 mismatch")
    receipt = _strict_json_object_v7(
        snapshot,
        label="v8c2 predecessor STOP receipt",
    )
    expected_authorization = {
        "blind_test_authorized": False,
        "calibration_authorized": False,
        "checkpoint_selected_as_final_model": False,
        "gguf_export_authorized": False,
        "model_authorized": False,
        "production_integration_authorized": False,
        "three_seed_training_authorized": False,
        "x5_deployment_authorized": False,
    }
    if (
        receipt.get("schema")
        != "icmat_llm_canary_acceptance_receipt.v8"
        or receipt.get("status") != V8C2_PREDECESSOR_STOP_STATUS
        or receipt.get("gate_passed") is not False
        or receipt.get("next_action")
        != "STOP_AND_REVIEW_V8_NONBLIND_CANARY"
        or receipt.get("authorization") != expected_authorization
    ):
        raise QLoRAV6Error("v8c2 predecessor is not the fixed STOP receipt")
    return {
        "path": str(snapshot.path),
        "bytes": snapshot.byte_count,
        "sha256": snapshot.sha256,
        "stable_identity": snapshot.identity_receipt(),
        "schema": receipt["schema"],
        "status": receipt["status"],
        "gate_passed": False,
        "next_action": receipt["next_action"],
        "authorization": expected_authorization,
    }


def _validate_v8c2_preregistration(
    snapshot: StableFileSnapshotV7,
) -> dict[str, Any]:
    if snapshot.sha256 != V8C2_PREREGISTRATION_SHA256:
        raise QLoRAV6Error(
            "v8c2 preregistration SHA-256 mismatch"
        )
    payload = _strict_json_object_v7(
        snapshot,
        label="v8c2 preregistration",
    )
    expected_top_level = {
        "schema",
        "protocol_id",
        "created_on",
        "status",
        "predecessor",
        "diagnostic_basis",
        "frozen_data",
        "single_atomic_algorithm_change",
        "required_protocol_hardening",
        "fixed_runs",
        "canary_gate",
        "stop_rules",
        "claim_boundary",
    }
    predecessor = payload.get("predecessor")
    if (
        set(payload) != expected_top_level
        or payload.get("schema")
        != "icmat_pointer_v8c2_preregistration.v1"
        or payload.get("protocol_id")
        != V8C2_PREREGISTRATION_PROTOCOL_ID
        or payload.get("created_on") != "2026-07-31"
        or payload.get("status")
        != "FROZEN_BEFORE_IMPLEMENTATION_AND_TRAINING"
        or not isinstance(predecessor, Mapping)
        or payload.get("frozen_data") != _V8C2_FROZEN_DATA
        or payload.get("single_atomic_algorithm_change")
        != _V8C2_FIXED_ALGORITHM
        or payload.get("required_protocol_hardening")
        != _V8C2_REQUIRED_HARDENING
        or payload.get("fixed_runs") != _V8C2_FIXED_RUNS
        or payload.get("canary_gate") != _V8C2_CANARY_GATE
        or payload.get("stop_rules") != _V8C2_STOP_RULES
        or not isinstance(payload.get("diagnostic_basis"), Mapping)
        or not isinstance(payload.get("claim_boundary"), str)
        or not payload["claim_boundary"]
    ):
        raise QLoRAV6Error(
            "v8c2 preregistration contract mismatch"
        )
    predecessor_receipt = _validate_v8c2_predecessor(predecessor)
    return {
        "path": str(snapshot.path),
        "bytes": snapshot.byte_count,
        "sha256": snapshot.sha256,
        "stable_identity": snapshot.identity_receipt(),
        "schema": payload["schema"],
        "protocol_id": V8C2_PREREGISTRATION_PROTOCOL_ID,
        "status": payload["status"],
        "profile": V8C2_TRAINING_PROFILE,
        "predecessor": predecessor_receipt,
        "frozen_data": dict(_V8C2_FROZEN_DATA),
        "algorithm": dict(_V8C2_FIXED_ALGORITHM),
        "required_protocol_hardening": dict(_V8C2_REQUIRED_HARDENING),
        "fixed_runs": dict(_V8C2_FIXED_RUNS),
        "canary_gate": dict(_V8C2_CANARY_GATE),
        "stop_rules": list(_V8C2_STOP_RULES),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_dataset_kind(dataset: Mapping[str, Any]) -> str:
    manifest = dataset.get("manifest")
    if not isinstance(manifest, Mapping):
        raise QLoRAV6Error("dataset manifest receipt is missing")
    schema = manifest.get("schema")
    if schema == MANIFEST_SCHEMA:
        return "legacy"
    if schema == NONBLIND_MANIFEST_SCHEMA:
        return "v7"
    if schema == NONBLIND_V8_MANIFEST_SCHEMA:
        return "v8"
    raise QLoRAV6Error(f"unsupported dataset schema: {schema!r}")


def _v8_training_gate_bundle(
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "nonblind_compare",
        "scoped_lexical",
        "unique_support",
        "nli_model",
    }
    if set(gates) != required:
        raise QLoRAV6Error("v8 training gate bundle keys mismatch")
    unique = gates.get("unique_support")
    if not isinstance(unique, Mapping) or set(unique) != {
        "train",
        "validation",
    }:
        raise QLoRAV6Error(
            "v8 training gate bundle unique-support splits mismatch"
        )
    digest_payload = {
        "contract": "STRICT_NONBLIND_V8",
        "nonblind_compare": gates["nonblind_compare"],
        "scoped_lexical": gates["scoped_lexical"],
        "unique_support": {
            "train": unique["train"],
            "validation": unique["validation"],
        },
        "nli_model": gates["nli_model"],
    }
    return {
        **digest_payload,
        "training_gate_bundle_sha256": _canonical_sha256(digest_payload),
    }


def _validate_v8c2_frozen_data_binding(
    *,
    preregistration: Mapping[str, Any],
    manifest: Mapping[str, Any],
    splits: Mapping[str, Any],
    training_gate_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    frozen = preregistration.get("frozen_data")
    if not isinstance(frozen, Mapping) or dict(frozen) != _V8C2_FROZEN_DATA:
        raise QLoRAV6Error("v8c2 frozen-data authority is missing")
    observed_splits: dict[str, Mapping[str, Any]] = {}
    for split in NONBLIND_SPLIT_FILES:
        value = splits.get(split)
        if not isinstance(value, Mapping):
            raise QLoRAV6Error(f"v8c2 formal {split} receipt is missing")
        observed_splits[split] = value
    observed = {
        "dataset_contract": "STRICT_NONBLIND_V8",
        "dataset_manifest_sha256": manifest.get("sha256"),
        "train_sha256": observed_splits["train"].get("sha256"),
        "validation_sha256": observed_splits["validation"].get("sha256"),
        "calibration_sha256": observed_splits["calibration"].get("sha256"),
        "training_gate_bundle_sha256": training_gate_bundle.get(
            "training_gate_bundle_sha256"
        ),
        "train_rows": observed_splits["train"].get("examples"),
        "validation_rows": observed_splits["validation"].get("examples"),
        "calibration_rows": observed_splits["calibration"].get("examples"),
        "training_data_changes_allowed": False,
        "validation_feedback_into_training_allowed": False,
        "calibration_for_training_or_selection_allowed": False,
        "sealed_blind_access_allowed_before_postfreeze": False,
    }
    if observed != _V8C2_FROZEN_DATA:
        mismatches = sorted(
            key
            for key, expected in _V8C2_FROZEN_DATA.items()
            if observed.get(key) != expected
        )
        raise QLoRAV6Error(
            "v8c2 formal frozen-data binding mismatch: "
            + ", ".join(mismatches)
        )
    return observed


def _stable_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _is_reparse_point(value: os.stat_result) -> bool:
    attributes = int(getattr(value, "st_file_attributes", 0))
    marker = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return bool(marker and attributes & marker)


def _absolute_lexical_v7(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path))))


def _assert_no_link_components_v7(
    path: Path,
    *,
    label: str,
    allow_missing_leaf: bool = False,
) -> None:
    lexical = _absolute_lexical_v7(path)
    parts = lexical.parts
    if not parts:
        raise QLoRAV6Error(f"{label} path is empty")
    current = Path(parts[0])
    for index, part in enumerate(parts[1:], start=1):
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            if allow_missing_leaf and index == len(parts) - 1:
                return
            raise QLoRAV6Error(f"{label} path is missing") from None
        if stat.S_ISLNK(metadata.st_mode) or _is_reparse_point(metadata):
            raise QLoRAV6Error(
                f"{label} must contain only non-reparse path components "
                "(no symbolic link, junction, or reparse point)"
            )


def _strict_directory_identity_v7(
    path: Path,
    *,
    label: str,
) -> tuple[Path, tuple[int, int]]:
    lexical = _absolute_lexical_v7(path)
    _assert_no_link_components_v7(lexical, label=label)
    try:
        metadata = os.lstat(lexical)
    except FileNotFoundError as exc:
        raise QLoRAV6Error(f"{label} is missing") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse_point(metadata)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise QLoRAV6Error(
            f"{label} must be a regular non-reparse directory"
        )
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise QLoRAV6Error(f"{label} cannot be resolved safely") from exc
    return resolved, (int(metadata.st_dev), int(metadata.st_ino))


def _recheck_strict_directory_identity_v7(
    path: Path,
    *,
    expected: tuple[int, int],
    label: str,
) -> None:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError as exc:
        raise QLoRAV6Error(f"{label} disappeared") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse_point(metadata)
        or not stat.S_ISDIR(metadata.st_mode)
        or (int(metadata.st_dev), int(metadata.st_ino)) != expected
    ):
        raise QLoRAV6Error(f"{label} identity changed")


def _stable_snapshot_v7(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
) -> StableFileSnapshotV7:
    lexical = _absolute_lexical_v7(path)
    _assert_no_link_components_v7(lexical, label=label)
    try:
        before = os.lstat(lexical)
    except FileNotFoundError as exc:
        raise QLoRAV6Error(f"{label} is missing") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or _is_reparse_point(before)
        or not stat.S_ISREG(before.st_mode)
    ):
        raise QLoRAV6Error(
            f"{label} must be a regular non-reparse file"
        )
    if before.st_size < 1 or before.st_size > maximum_bytes:
        raise QLoRAV6Error(f"{label} byte count is outside the fixed limit")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(os.fspath(lexical), flags)
    except OSError as exc:
        raise QLoRAV6Error(f"{label} cannot be opened safely") from exc
    blocks: list[bytes] = []
    try:
        descriptor_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(descriptor_before.st_mode)
            or _stable_identity(descriptor_before)
            != _stable_identity(before)
        ):
            raise QLoRAV6Error(f"{label} identity changed before read")
        total = 0
        while True:
            block = os.read(descriptor, _STRICT_READ_BLOCK_BYTES)
            if not block:
                break
            total += len(block)
            if total > maximum_bytes:
                raise QLoRAV6Error(
                    f"{label} exceeded the fixed read limit"
                )
            blocks.append(block)
        descriptor_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after = os.lstat(lexical)
    except FileNotFoundError as exc:
        raise QLoRAV6Error(f"{label} disappeared after read") from exc
    identities = {
        _stable_identity(before),
        _stable_identity(descriptor_before),
        _stable_identity(descriptor_after),
        _stable_identity(after),
    }
    if (
        len(identities) != 1
        or stat.S_ISLNK(after.st_mode)
        or _is_reparse_point(after)
        or not stat.S_ISREG(after.st_mode)
    ):
        raise QLoRAV6Error(f"{label} changed while it was read")
    payload = b"".join(blocks)
    if len(payload) != int(after.st_size):
        raise QLoRAV6Error(f"{label} byte count changed while it was read")
    return StableFileSnapshotV7(
        path=lexical,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_count=len(payload),
        identity=_stable_identity(after),
    )


def _strict_json_object_v7(
    snapshot: StableFileSnapshotV7,
    *,
    label: str,
) -> dict[str, Any]:
    def pairs_hook(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise QLoRAV6Error(f"{label} contains duplicate JSON keys")
            output[key] = value
        return output

    try:
        value = json.loads(
            snapshot.payload.decode("utf-8"),
            object_pairs_hook=pairs_hook,
        )
    except UnicodeDecodeError as exc:
        raise QLoRAV6Error(f"{label} is not UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise QLoRAV6Error(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise QLoRAV6Error(f"{label} must contain one JSON object")
    return value


def _strict_jsonl_rows_v7(
    snapshot: StableFileSnapshotV7,
    *,
    label: str,
) -> list[dict[str, Any]]:
    if not snapshot.payload.endswith(b"\n"):
        raise QLoRAV6Error(f"{label} must end with one JSONL newline")
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        snapshot.payload.splitlines(),
        start=1,
    ):
        if not raw:
            raise QLoRAV6Error(f"{label}:{line_number} is blank")
        line_snapshot = StableFileSnapshotV7(
            path=snapshot.path,
            payload=raw,
            sha256=hashlib.sha256(raw).hexdigest(),
            byte_count=len(raw),
            identity=snapshot.identity,
        )
        rows.append(
            _strict_json_object_v7(
                line_snapshot,
                label=f"{label}:{line_number}",
            )
        )
    if not rows:
        raise QLoRAV6Error(f"{label} is empty")
    return rows


def _stable_model_file_v7(
    path: Path,
    *,
    root: Path,
    label: str,
) -> StableModelFileV7:
    lexical = _absolute_lexical_v7(path)
    try:
        before = os.lstat(lexical)
    except FileNotFoundError as exc:
        raise QLoRAV6Error(f"{label} disappeared") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or _is_reparse_point(before)
        or not stat.S_ISREG(before.st_mode)
    ):
        raise QLoRAV6Error(
            f"{label} must be a regular non-reparse file"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(os.fspath(lexical), flags)
    except OSError as exc:
        raise QLoRAV6Error(f"{label} cannot be opened safely") from exc
    digest = hashlib.sha256()
    total = 0
    config_blocks: list[bytes] | None = (
        [] if lexical == root / "config.json" else None
    )
    try:
        descriptor_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(descriptor_before.st_mode)
            or _stable_identity(descriptor_before) != _stable_identity(before)
        ):
            raise QLoRAV6Error(f"{label} identity changed before read")
        while True:
            block = os.read(descriptor, _STRICT_READ_BLOCK_BYTES)
            if not block:
                break
            total += len(block)
            digest.update(block)
            if config_blocks is not None:
                if total > _STRICT_MAX_JSON_BYTES:
                    raise QLoRAV6Error(
                        "base model config.json exceeds the fixed limit"
                    )
                config_blocks.append(block)
        descriptor_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after = os.lstat(lexical)
    except FileNotFoundError as exc:
        raise QLoRAV6Error(f"{label} disappeared after read") from exc
    identities = {
        _stable_identity(before),
        _stable_identity(descriptor_before),
        _stable_identity(descriptor_after),
        _stable_identity(after),
    }
    if (
        len(identities) != 1
        or stat.S_ISLNK(after.st_mode)
        or _is_reparse_point(after)
        or not stat.S_ISREG(after.st_mode)
        or total != int(after.st_size)
    ):
        raise QLoRAV6Error(f"{label} changed while it was read")
    return StableModelFileV7(
        path=lexical,
        relative_path=lexical.relative_to(root).as_posix(),
        sha256=digest.hexdigest(),
        byte_count=total,
        identity=_stable_identity(after),
        config_payload=(
            None if config_blocks is None else b"".join(config_blocks)
        ),
    )


def _stable_model_tree_v7(
    root: Path,
    *,
    label: str = "input tree",
) -> StableModelTreeV7:
    lexical, _ = _strict_directory_identity_v7(root, label=label)
    files: list[StableModelFileV7] = []
    directories: list[StableModelDirectoryV7] = []

    def visit(directory: Path) -> None:
        try:
            metadata = os.lstat(directory)
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise QLoRAV6Error(f"{label} cannot be enumerated safely") from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or _is_reparse_point(metadata)
            or not stat.S_ISDIR(metadata.st_mode)
        ):
            raise QLoRAV6Error(
                f"{label} contains a non-regular directory"
            )
        names = tuple(
            sorted((entry.name for entry in entries), key=lambda value: (
                value.casefold(),
                value,
            ))
        )
        if len({name.casefold() for name in names}) != len(names):
            raise QLoRAV6Error(
                f"{label} contains case-colliding entries"
            )
        directories.append(
            StableModelDirectoryV7(
                path=directory,
                relative_path=(
                    "."
                    if directory == lexical
                    else directory.relative_to(lexical).as_posix()
                ),
                identity=_stable_identity(metadata),
                entries=names,
            )
        )
        for entry in sorted(
            entries,
            key=lambda item: (item.name.casefold(), item.name),
        ):
            child = directory / entry.name
            try:
                child_metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise QLoRAV6Error(
                    f"{label} entry changed during enumeration"
                ) from exc
            if (
                entry.is_symlink()
                or stat.S_ISLNK(child_metadata.st_mode)
                or _is_reparse_point(child_metadata)
            ):
                raise QLoRAV6Error(
                    f"{label} must not contain symbolic links, junctions, "
                    "or reparse points"
                )
            if stat.S_ISDIR(child_metadata.st_mode):
                visit(child)
            elif stat.S_ISREG(child_metadata.st_mode):
                files.append(
                    _stable_model_file_v7(
                        child,
                        root=lexical,
                        label=f"{label} file",
                    )
                )
            else:
                raise QLoRAV6Error(
                    f"{label} contains a non-regular filesystem entry"
                )

    visit(lexical)
    files.sort(
        key=lambda item: (
            item.relative_path.casefold(),
            item.relative_path,
        )
    )
    if not files:
        raise QLoRAV6Error(f"{label} is empty")
    relative_paths = [item.relative_path for item in files]
    if len({path.casefold() for path in relative_paths}) != len(
        relative_paths
    ):
        raise QLoRAV6Error(f"{label} contains case-colliding file paths")

    for directory in reversed(directories):
        try:
            current = os.lstat(directory.path)
            current_entries = tuple(
                sorted(
                    (entry.name for entry in os.scandir(directory.path)),
                    key=lambda value: (value.casefold(), value),
                )
            )
        except OSError as exc:
            raise QLoRAV6Error(
                f"{label} changed during stable inspection"
            ) from exc
        if (
            _stable_identity(current) != directory.identity
            or stat.S_ISLNK(current.st_mode)
            or _is_reparse_point(current)
            or not stat.S_ISDIR(current.st_mode)
            or current_entries != directory.entries
        ):
            raise QLoRAV6Error(
                f"{label} changed during stable inspection"
            )

    records = [
        {
            "path": item.relative_path,
            "bytes": item.byte_count,
            "sha256": item.sha256,
        }
        for item in files
    ]
    identity_records = {
        "directories": [
            {
                "path": item.relative_path,
                "identity": list(item.identity),
                "entries": list(item.entries),
            }
            for item in directories
        ],
        "files": [
            {
                "path": item.relative_path,
                "identity": list(item.identity),
            }
            for item in files
        ],
    }
    return StableModelTreeV7(
        root=lexical,
        files=tuple(files),
        directories=tuple(directories),
        tree_sha256=_canonical_sha256(records),
        stable_identity_sha256=_canonical_sha256(identity_records),
        byte_count=sum(item.byte_count for item in files),
    )


def _verify_stable_model_tree_v7(
    expected: StableModelTreeV7,
    *,
    label: str,
) -> StableModelTreeV7:
    current = _stable_model_tree_v7(expected.root, label=label)
    if (
        current.inventory_receipt() != expected.inventory_receipt()
        or current.stable_identity_sha256
        != expected.stable_identity_sha256
    ):
        raise PermissionError(f"{label} identity or content changed")
    return current


def _tree_inventory(root: Path) -> dict[str, Any]:
    return _stable_model_tree_v7(root).inventory_receipt()


def _selected_inventory(
    root: Path,
    *,
    filenames: frozenset[str],
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    records: list[dict[str, Any]] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise PermissionError(f"checkpoint symlinks are forbidden: {path}")
        if not path.is_file() or path.name not in filenames:
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    records.sort(
        key=lambda record: (
            str(record["path"]).casefold(),
            str(record["path"]),
        )
    )
    if not records:
        raise QLoRAV6Error(f"adapter files are missing from checkpoint: {root}")
    model_files = [
        record
        for record in records
        if Path(str(record["path"])).name
        in {"adapter_model.safetensors", "adapter_model.bin"}
    ]
    config_files = [
        record
        for record in records
        if Path(str(record["path"])).name == "adapter_config.json"
    ]
    if len(model_files) != 1 or len(config_files) != 1:
        raise QLoRAV6Error(
            "each checkpoint must contain exactly one adapter model and config"
        )
    return {
        "files": records,
        "tree_sha256": _canonical_sha256(records),
        "file_count": len(records),
        "bytes": sum(int(record["bytes"]) for record in records),
    }


def _model_snapshot(model_dir: Path) -> dict[str, Any]:
    tree = _stable_model_tree_v7(
        model_dir,
        label="base model directory",
    )
    root = tree.root
    config_files = [
        item for item in tree.files if item.relative_path == "config.json"
    ]
    if len(config_files) != 1 or config_files[0].config_payload is None:
        raise QLoRAV6Error("base model config.json must be a regular file")
    config_snapshot = StableFileSnapshotV7(
        path=config_files[0].path,
        payload=config_files[0].config_payload,
        sha256=config_files[0].sha256,
        byte_count=config_files[0].byte_count,
        identity=config_files[0].identity,
    )
    config = _strict_json_object_v7(
        config_snapshot,
        label="base model config.json",
    )
    if not isinstance(config, Mapping):
        raise QLoRAV6Error("base model config.json must contain an object")
    for key, expected in EXPECTED_MODEL_CONFIG.items():
        if config.get(key) != expected:
            raise QLoRAV6Error(
                f"base model is not the frozen Qwen2.5-0.5B shape: {key}"
            )
    architectures = config.get("architectures")
    if (
        not isinstance(architectures, list)
        or "Qwen2ForCausalLM" not in architectures
    ):
        raise QLoRAV6Error("base model architecture must be Qwen2ForCausalLM")
    inventory = tree.inventory_receipt()
    return {
        "provided": True,
        "path": str(root),
        "model_family": "Qwen2.5-0.5B-Instruct",
        "no_reparse_components": True,
        "stable_identity_sha256": tree.stable_identity_sha256,
        "config_fingerprint": {
            **EXPECTED_MODEL_CONFIG,
            "architecture": "Qwen2ForCausalLM",
        },
        **inventory,
    }


def _model_receipt_matches_tree_v7(
    receipt: Mapping[str, Any],
    tree: StableModelTreeV7,
) -> bool:
    inventory = tree.inventory_receipt()
    return (
        receipt.get("path") == str(tree.root)
        and receipt.get("tree_sha256") == inventory["tree_sha256"]
        and receipt.get("file_count") == inventory["file_count"]
        and receipt.get("bytes") == inventory["bytes"]
        and receipt.get("files") == inventory["files"]
        and receipt.get("stable_identity_sha256")
        == tree.stable_identity_sha256
        and receipt.get("no_reparse_components") is True
    )


def _copy_model_file_v7(
    source: StableModelFileV7,
    destination: Path,
) -> None:
    try:
        before = os.lstat(source.path)
    except FileNotFoundError as exc:
        raise PermissionError(
            "base model changed before stable snapshot construction"
        ) from exc
    if (
        _stable_identity(before) != source.identity
        or stat.S_ISLNK(before.st_mode)
        or _is_reparse_point(before)
        or not stat.S_ISREG(before.st_mode)
    ):
        raise PermissionError(
            "base model changed before stable snapshot construction"
        )
    source_flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    destination_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
    )
    try:
        source_descriptor = os.open(os.fspath(source.path), source_flags)
    except OSError as exc:
        raise QLoRAV6Error(
            "base-model source cannot be opened for stable copy"
        ) from exc
    try:
        destination_descriptor = os.open(
            os.fspath(destination),
            destination_flags,
            0o600,
        )
    except OSError as exc:
        os.close(source_descriptor)
        raise QLoRAV6Error(
            "content-addressed base-model snapshot cannot be created"
        ) from exc
    digest = hashlib.sha256()
    total = 0
    try:
        descriptor_before = os.fstat(source_descriptor)
        if _stable_identity(descriptor_before) != source.identity:
            raise PermissionError(
                "base model changed before stable snapshot copy"
            )
        while True:
            block = os.read(source_descriptor, _STRICT_READ_BLOCK_BYTES)
            if not block:
                break
            digest.update(block)
            total += len(block)
            offset = 0
            while offset < len(block):
                written = os.write(
                    destination_descriptor,
                    block[offset:],
                )
                if written < 1:
                    raise OSError(
                        "zero-byte write while copying base model"
                    )
                offset += written
        os.fsync(destination_descriptor)
        descriptor_after = os.fstat(source_descriptor)
    finally:
        os.close(source_descriptor)
        os.close(destination_descriptor)
    try:
        after = os.lstat(source.path)
    except FileNotFoundError as exc:
        raise PermissionError(
            "base model disappeared during stable snapshot copy"
        ) from exc
    if (
        _stable_identity(descriptor_after) != source.identity
        or _stable_identity(after) != source.identity
        or total != source.byte_count
        or digest.hexdigest() != source.sha256
    ):
        raise PermissionError(
            "base model changed during stable snapshot copy"
        )


def _copy_content_addressed_model_v7(
    source: StableModelTreeV7,
    *,
    staging: Path,
) -> StableModelTreeV7:
    _verify_stable_model_tree_v7(
        source,
        label="base model source before snapshot copy",
    )
    # The full digest remains in the verified receipt; a short directory
    # component avoids legacy Windows MAX_PATH failures under pytest/tmp roots.
    destination = staging / f"m-{source.tree_sha256[:16]}"
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    if not os.path.lexists(staging):
        staging_parent, staging_parent_identity = (
            _strict_directory_identity_v7(
                staging.parent,
                label="training staging parent",
            )
        )
        _assert_no_link_components_v7(
            staging,
            label="training staging directory",
            allow_missing_leaf=True,
        )
        os.mkdir(staging)
        _recheck_strict_directory_identity_v7(
            staging_parent,
            expected=staging_parent_identity,
            label="training staging parent",
        )
    else:
        _strict_directory_identity_v7(
            staging,
            label="training staging directory",
        )
    os.mkdir(destination)
    for directory in sorted(
        (
            item
            for item in source.directories
            if item.relative_path != "."
        ),
        key=lambda item: (
            len(Path(item.relative_path).parts),
            item.relative_path.casefold(),
            item.relative_path,
        ),
    ):
        os.mkdir(destination / Path(directory.relative_path))
    for item in source.files:
        target = destination / Path(item.relative_path)
        _copy_model_file_v7(item, target)
    copied = _stable_model_tree_v7(
        destination,
        label="content-addressed base model snapshot",
    )
    if copied.inventory_receipt() != source.inventory_receipt():
        raise PermissionError(
            "content-addressed base model snapshot differs from source"
        )
    _verify_stable_model_tree_v7(
        source,
        label="base model source after snapshot copy",
    )
    return copied


def _remove_content_addressed_model_v7(
    snapshot: StableModelTreeV7,
) -> None:
    current = _verify_stable_model_tree_v7(
        snapshot,
        label="content-addressed base model snapshot before cleanup",
    )
    for item in reversed(current.files):
        item.path.unlink()
    for directory in sorted(
        (
            item
            for item in current.directories
            if item.relative_path != "."
        ),
        key=lambda item: len(Path(item.relative_path).parts),
        reverse=True,
    ):
        directory.path.rmdir()
    current.root.rmdir()


def _manifest_split_records(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise QLoRAV6Error("manifest.v6.json must contain split declarations")
    records: dict[str, dict[str, Any]] = {}
    for split, expected_name in SPLIT_FILES.items():
        record = splits.get(split)
        if not isinstance(record, Mapping):
            raise QLoRAV6Error(f"manifest is missing {split} declaration")
        if record.get("path") != expected_name:
            raise QLoRAV6Error(
                f"manifest {split} path must be {expected_name}"
            )
        byte_count = record.get("bytes")
        count = record.get("count")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            raise QLoRAV6Error(f"manifest {split} bytes are invalid")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise QLoRAV6Error(f"manifest {split} count is invalid")
        if not _valid_sha256(record.get("sha256")):
            raise QLoRAV6Error(f"manifest {split} SHA-256 is invalid")
        records[split] = {
            "path": expected_name,
            "bytes": byte_count,
            "sha256": record["sha256"],
            "examples": count,
        }
    declared_counts = manifest.get("counts", {}).get("splits")
    if not isinstance(declared_counts, Mapping):
        raise QLoRAV6Error("manifest split counts are missing")
    if {
        split: records[split]["examples"] for split in SPLIT_FILES
    } != dict(declared_counts):
        raise QLoRAV6Error("manifest split declarations and counts disagree")
    total = manifest.get("counts", {}).get("examples")
    if total != sum(record["examples"] for record in records.values()):
        raise QLoRAV6Error("manifest total example count is inconsistent")
    return records


def _parse_pointer_target(
    raw: str,
    *,
    source: str,
) -> dict[str, str | None]:
    try:
        pairs = json.loads(raw, object_pairs_hook=lambda values: values)
    except json.JSONDecodeError as exc:
        raise QLoRAV6Error(f"{source}: assistant target is not JSON") from exc
    if not isinstance(pairs, list) or any(
        not isinstance(item, tuple) or len(item) != 2 for item in pairs
    ):
        raise QLoRAV6Error(f"{source}: assistant target must be an object")
    keys = tuple(str(item[0]) for item in pairs)
    if keys != POINTER_FIELDS:
        raise QLoRAV6Error(f"{source}: pointer field order mismatch")
    if len(set(keys)) != len(keys):
        raise QLoRAV6Error(f"{source}: duplicate pointer keys")
    target = {str(key): value for key, value in pairs}
    if (
        not isinstance(target["task"], str)
        or not isinstance(target["decision"], str)
        or (
            target["span_id"] is not None
            and not isinstance(target["span_id"], str)
        )
    ):
        raise QLoRAV6Error(
            f"{source}: task/decision must be strings and span_id must be a string or null"
        )
    canonical = json.dumps(
        target,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if raw != canonical:
        raise QLoRAV6Error(f"{source}: pointer target must be compact")
    return target


def _validate_messages(
    messages: Any,
    *,
    source: str,
) -> dict[str, str | None]:
    if not isinstance(messages, list) or len(messages) != 3:
        raise QLoRAV6Error(
            f"{source}: messages must contain system/user/assistant"
        )
    roles = [
        message.get("role") if isinstance(message, Mapping) else None
        for message in messages
    ]
    if roles != ["system", "user", "assistant"]:
        raise QLoRAV6Error(
            f"{source}: message roles must be system/user/assistant"
        )
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise QLoRAV6Error(
                f"{source}: every message content must be non-empty"
            )
    return _parse_pointer_target(messages[-1]["content"], source=source)


def _validate_example(
    item: Any,
    *,
    split: str,
    source: str,
    seen_example_ids: set[str],
) -> tuple[str, str, str, str]:
    if not isinstance(item, Mapping):
        raise QLoRAV6Error(f"{source}: JSONL row must be an object")
    if item.get("schema") != EXAMPLE_SCHEMA:
        raise QLoRAV6Error(f"{source}: unexpected example schema")
    if item.get("dataset_schema") != DATASET_SCHEMA:
        raise QLoRAV6Error(f"{source}: unexpected dataset schema")
    if item.get("split") != split:
        raise QLoRAV6Error(f"{source}: embedded split mismatch")
    example_id = item.get("example_id")
    if not isinstance(example_id, str) or not example_id:
        raise QLoRAV6Error(f"{source}: example_id must be non-empty")
    if example_id in seen_example_ids:
        raise QLoRAV6Error(f"{source}: duplicate example_id {example_id}")
    seen_example_ids.add(example_id)
    domain = item.get("domain")
    task = item.get("task")
    decision = item.get("decision")
    source_id = item.get("source_id")
    if not isinstance(domain, str) or not domain:
        raise QLoRAV6Error(f"{source}: domain must be non-empty")
    if task not in ALLOWED_TASKS:
        raise QLoRAV6Error(f"{source}: unsupported task")
    if decision not in ALLOWED_DECISIONS:
        raise QLoRAV6Error(f"{source}: unsupported decision")
    if not isinstance(source_id, str) or not source_id:
        raise QLoRAV6Error(f"{source}: source_id must be non-empty")
    if item.get("family_id") != source_id:
        raise QLoRAV6Error(f"{source}: family_id/source_id mismatch")
    target = _validate_messages(item.get("messages"), source=source)
    if target["task"] != task or target["decision"] != decision:
        raise QLoRAV6Error(f"{source}: pointer target metadata mismatch")
    span_id = target["span_id"]
    if decision == "ANSWER":
        if not isinstance(span_id, str) or not SPAN_ID_RE.fullmatch(span_id):
            raise QLoRAV6Error(f"{source}: invalid ANSWER span_id")
    elif span_id is not None:
        raise QLoRAV6Error(f"{source}: REFUSE span_id must be null")
    if item.get("target_span_id") != span_id:
        raise QLoRAV6Error(f"{source}: target_span_id mismatch")
    return domain, str(task), str(decision), source_id


def _scan_visible_jsonl(
    path: Path,
    *,
    split: str,
    expected: Mapping[str, Any],
    seen_example_ids: set[str],
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise QLoRAV6Error(f"{split} must be a regular file")
    digest = hashlib.sha256()
    byte_count = 0
    row_count = 0
    domains: Counter[str] = Counter()
    tasks: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    sources: set[str] = set()
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            byte_count += len(raw_line)
            if not raw_line.strip():
                raise QLoRAV6Error(
                    f"{path.name}:{line_number}: blank JSONL row"
                )
            try:
                item = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise QLoRAV6Error(
                    f"{path.name}:{line_number}: invalid UTF-8 JSON"
                ) from exc
            domain, task, decision, source_id = _validate_example(
                item,
                split=split,
                source=f"{path.name}:{line_number}",
                seen_example_ids=seen_example_ids,
            )
            domains[domain] += 1
            tasks[task] += 1
            decisions[decision] += 1
            sources.add(source_id)
            row_count += 1
    observed_sha256 = digest.hexdigest()
    if byte_count != expected["bytes"]:
        raise QLoRAV6Error(f"{path.name}: byte count does not match manifest")
    if observed_sha256 != expected["sha256"]:
        raise QLoRAV6Error(f"{path.name}: SHA-256 does not match manifest")
    if row_count != expected["examples"]:
        raise QLoRAV6Error(f"{path.name}: count does not match manifest")
    return {
        "path": path.name,
        "bytes": byte_count,
        "sha256": observed_sha256,
        "examples": row_count,
        "domains": dict(sorted(domains.items())),
        "tasks": dict(sorted(tasks.items())),
        "decisions": dict(sorted(decisions.items())),
        "source_ids": sorted(sources),
        "content_read": True,
        "content_parsed": True,
        "content_hashed": True,
    }


def _scan_strict_visible_snapshot_v7(
    snapshot: StableFileSnapshotV7,
    *,
    split: str,
    expected: Mapping[str, Any],
    seen_example_ids: set[str],
) -> dict[str, Any]:
    if (
        snapshot.byte_count != expected["bytes"]
        or snapshot.sha256 != expected["sha256"]
    ):
        raise QLoRAV6Error(
            f"nonblind-v7 {split} snapshot does not match manifest"
        )
    rows = _strict_jsonl_rows_v7(snapshot, label=f"{split}.jsonl")
    if len(rows) != expected["examples"]:
        raise QLoRAV6Error(
            f"nonblind-v7 {split} count does not match manifest"
        )
    domains: Counter[str] = Counter()
    tasks: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    sources: set[str] = set()
    for line_number, item in enumerate(rows, start=1):
        try:
            evidence_contract.validate_example(item)
        except evidence_contract.EvidenceSFTV6Error as exc:
            raise QLoRAV6Error(
                f"{split}.jsonl:{line_number}: strict Evidence v6 "
                f"contract rejected the row: {exc}"
            ) from exc
        if item.get("split") != split:
            raise QLoRAV6Error(
                f"{split}.jsonl:{line_number}: embedded split mismatch"
            )
        example_id = item.get("example_id")
        if (
            not isinstance(example_id, str)
            or not example_id
            or example_id in seen_example_ids
        ):
            raise QLoRAV6Error(
                f"{split}.jsonl:{line_number}: duplicate example_id"
            )
        seen_example_ids.add(example_id)
        domains[str(item["domain"])] += 1
        tasks[str(item["task"])] += 1
        decisions[str(item["decision"])] += 1
        sources.add(str(item["source_id"]))
    return {
        "path": snapshot.path.name,
        "bytes": snapshot.byte_count,
        "sha256": snapshot.sha256,
        "examples": len(rows),
        "domains": dict(sorted(domains.items())),
        "tasks": dict(sorted(tasks.items())),
        "decisions": dict(sorted(decisions.items())),
        "source_ids": sorted(sources),
        "stable_identity": snapshot.identity_receipt(),
        "content_read": True,
        "content_parsed": True,
        "content_hashed": True,
        "stable_snapshot": True,
    }


def _declaration_only_record(
    split: str,
    record: Mapping[str, Any],
    *,
    declaration_source: str = MANIFEST_NAME,
) -> dict[str, Any]:
    return {
        "path": record["path"],
        "bytes": record["bytes"],
        "sha256": record["sha256"],
        "examples": record["examples"],
        "declaration_source": declaration_source,
        "content_read": False,
        "content_parsed": False,
        "content_hashed": False,
        "filesystem_metadata_accessed": False,
        "used_for_training": False,
        "used_for_checkpoint_selection": False,
        "policy": (
            "MANIFEST_DECLARATION_ONLY"
            if split == "calibration"
            else "SEALED_MANIFEST_DECLARATION_ONLY"
        ),
    }


def _semantic_query_contract_v7() -> dict[str, Any]:
    return {
        "record_schema": SEMANTIC_RECORD_SCHEMA,
        "required": True,
        "fallback_without_inventory": False,
        "binding": "source_id+original_sha256",
        "answer_query": "accepted_paraphrase",
        "refusal_mix": {
            "controlled_contradiction": 175,
            "hidden_same_family_paraphrase": 175,
        },
        "hard_negative_policy": (
            "highest_token_overlap_nonoverlapping_passage"
        ),
        "normalized_exact_match_shortcut_forbidden": True,
    }


def _semantic_audit_contract_v7() -> dict[str, Any]:
    return {
        "binding": "source_id+original_sha256",
        "accepted_inventory_schema": SEMANTIC_ACCEPTED_INVENTORY_SCHEMA,
        "accepted_required": True,
        "paraphrase_required": True,
        "controlled_contradiction_required": True,
        "provenance_required": True,
        "audit_required": True,
        "record_hash_required": True,
        "fallback_without_inventory": False,
    }


def _read_semantic_audit_artifact_v7(
    root: Path,
    receipt: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(receipt, Mapping):
        raise QLoRAV6Error(
            "semantic-v7 manifest is missing semantic inventory audit receipt"
        )
    relative = receipt.get("path")
    byte_count = receipt.get("bytes")
    declared_sha256 = receipt.get("sha256")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 1
        or not _valid_sha256(declared_sha256)
    ):
        raise QLoRAV6Error(
            "semantic-v7 semantic inventory audit receipt is invalid"
        )
    path = (root / relative).resolve()
    if (
        path.parent != root
        or path.is_symlink()
        or not path.is_file()
    ):
        raise QLoRAV6Error(
            "semantic-v7 semantic inventory audit must be a dataset-root file"
        )
    payload_bytes = path.read_bytes()
    observed_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    if len(payload_bytes) != byte_count or observed_sha256 != declared_sha256:
        raise QLoRAV6Error(
            "semantic-v7 semantic inventory audit receipt mismatch"
        )
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QLoRAV6Error(
            "semantic-v7 semantic inventory audit is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise QLoRAV6Error(
            "semantic-v7 semantic inventory audit must contain an object"
        )
    return dict(payload), {
        "path": relative,
        "bytes": byte_count,
        "sha256": observed_sha256,
        "content_read": True,
        "content_parsed": True,
        "content_hashed": True,
    }


def _semantic_manifest_binding_v7(
    root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    source_inputs = manifest.get("source_inputs")
    source = (
        source_inputs.get("semantic_inventory")
        if isinstance(source_inputs, Mapping)
        else None
    )
    if (
        not isinstance(source, Mapping)
        or set(source)
        != {
            "path",
            "sha256",
            "schema",
            "producer_inventory_sha256",
            "records_path",
            "records_sha256",
            "record_schema",
            "record_count",
            "accepted_count",
        }
        or not isinstance(source.get("path"), str)
        or not source.get("path")
        or not isinstance(source.get("records_path"), str)
        or not source.get("records_path")
        or not _valid_sha256(source.get("sha256"))
        or not _valid_sha256(source.get("producer_inventory_sha256"))
        or not _valid_sha256(source.get("records_sha256"))
        or source.get("schema") != SEMANTIC_ACCEPTED_INVENTORY_SCHEMA
        or source.get("record_schema") != SEMANTIC_RECORD_SCHEMA
        or isinstance(source.get("record_count"), bool)
        or not isinstance(source.get("record_count"), int)
        or isinstance(source.get("accepted_count"), bool)
        or not isinstance(source.get("accepted_count"), int)
        or source.get("record_count", 0)
        < source.get("accepted_count", 0)
        or source.get("accepted_count", 0) < 700
    ):
        raise QLoRAV6Error(
            "semantic-v7 semantic inventory source binding mismatch"
        )

    query_contract = manifest.get("semantic_query_contract")
    expected_query_contract = _semantic_query_contract_v7()
    if query_contract != expected_query_contract:
        raise QLoRAV6Error("semantic-v7 semantic query contract mismatch")

    artifacts = manifest.get("artifacts")
    audit_receipt = (
        artifacts.get("semantic_inventory_audit")
        if isinstance(artifacts, Mapping)
        else None
    )
    audit, verified_receipt = _read_semantic_audit_artifact_v7(
        root,
        audit_receipt,
    )
    record_count = int(source["record_count"])
    accepted_count = int(source["accepted_count"])
    if (
        audit.get("schema") != SEMANTIC_AUDIT_SCHEMA
        or audit.get("builder_version") != SEMANTIC_BUILDER_VERSION
        or audit.get("status") != "PASS"
        or audit.get("findings") != []
        or audit.get("semantic_inventory_sha256")
        != source.get("sha256")
        or audit.get("producer_inventory_sha256")
        != source.get("producer_inventory_sha256")
        or audit.get("semantic_records_sha256")
        != source.get("records_sha256")
        or audit.get("record_schema") != SEMANTIC_RECORD_SCHEMA
        or audit.get("record_count") != record_count
        or audit.get("accepted_count") != accepted_count
        or audit.get("unique_binding_count") != accepted_count
        or audit.get("unique_record_hash_count") != accepted_count
        or audit.get("contract") != _semantic_audit_contract_v7()
    ):
        raise QLoRAV6Error(
            "semantic-v7 semantic inventory audit binding mismatch"
        )

    source_binding = dict(source)
    source_binding.update(
        {
            "declaration_source": MANIFEST_NAME,
            "content_read": False,
            "content_hashed": False,
        }
    )
    return {
        "required": True,
        "source_inventory": source_binding,
        "source_binding_sha256": _canonical_sha256(dict(source)),
        "audit_artifact": verified_receipt,
        "audit_payload_sha256": _canonical_sha256(audit),
        "query_contract": dict(query_contract),
        "query_contract_sha256": _canonical_sha256(query_contract),
    }


def _nonblind_receipt_declaration(
    value: Any,
    *,
    expected_path: str,
    label: str,
) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "sha256", "bytes"}
        or value.get("path") != expected_path
        or not _valid_sha256(value.get("sha256"))
        or isinstance(value.get("bytes"), bool)
        or not isinstance(value.get("bytes"), int)
        or value.get("bytes", 0) < 1
    ):
        raise QLoRAV6Error(f"nonblind-v7 {label} receipt mismatch")
    return {
        "path": expected_path,
        "bytes": int(value["bytes"]),
        "sha256": str(value["sha256"]),
    }


def _verify_strict_receipt_snapshot_v7(
    value: Any,
    snapshot: StableFileSnapshotV7,
    *,
    expected_path: str,
    label: str,
) -> dict[str, Any]:
    receipt = _nonblind_receipt_declaration(
        value,
        expected_path=expected_path,
        label=label,
    )
    if (
        receipt["bytes"] != snapshot.byte_count
        or receipt["sha256"] != snapshot.sha256
    ):
        raise QLoRAV6Error(
            f"nonblind-v7 {label} receipt does not match stable bytes"
        )
    return {
        **receipt,
        "stable_identity": snapshot.identity_receipt(),
        "content_opened": True,
        "content_read": True,
        "content_hashed": True,
        "stable_snapshot": True,
    }


def _nonblind_split_records(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    splits = manifest.get("splits")
    if (
        not isinstance(splits, Mapping)
        or set(splits) != set(NONBLIND_SPLIT_FILES)
    ):
        raise QLoRAV6Error(
            "nonblind-v7 manifest must declare exactly three nonblind splits"
        )
    records: dict[str, dict[str, Any]] = {}
    for split, expected_path in NONBLIND_SPLIT_FILES.items():
        value = splits.get(split)
        if (
            not isinstance(value, Mapping)
            or set(value) != {"path", "sha256", "bytes", "count"}
            or value.get("path") != expected_path
            or not _valid_sha256(value.get("sha256"))
            or isinstance(value.get("bytes"), bool)
            or not isinstance(value.get("bytes"), int)
            or value.get("bytes", 0) < 1
            or isinstance(value.get("count"), bool)
            or not isinstance(value.get("count"), int)
            or value.get("count") != NONBLIND_SPLIT_COUNTS[split]
        ):
            raise QLoRAV6Error(
                f"nonblind-v7 {split} split receipt mismatch"
            )
        records[split] = {
            "path": expected_path,
            "bytes": int(value["bytes"]),
            "sha256": str(value["sha256"]),
            "examples": int(value["count"]),
        }
    return records


def _nonblind_source_and_builder_declarations(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_inputs = manifest.get("source_inputs")
    if (
        not isinstance(source_inputs, Mapping)
        or set(source_inputs)
        != {"licensed_chunks", "rag_manifest", "semantic_inventory"}
    ):
        raise QLoRAV6Error("nonblind-v7 source input declarations mismatch")
    licensed = source_inputs.get("licensed_chunks")
    rag = source_inputs.get("rag_manifest")
    semantic = source_inputs.get("semantic_inventory")
    if (
        not isinstance(licensed, Mapping)
        or set(licensed) != {"path", "sha256"}
        or not isinstance(licensed.get("path"), str)
        or not licensed.get("path")
        or not _valid_sha256(licensed.get("sha256"))
    ):
        raise QLoRAV6Error(
            "nonblind-v7 licensed chunks declaration mismatch"
        )
    if (
        not isinstance(rag, Mapping)
        or set(rag) != {"path", "sha256", "manifest_id"}
        or not isinstance(rag.get("path"), str)
        or not rag.get("path")
        or not _valid_sha256(rag.get("sha256"))
        or not isinstance(rag.get("manifest_id"), str)
        or not rag.get("manifest_id")
    ):
        raise QLoRAV6Error("nonblind-v7 RAG declaration mismatch")
    semantic_keys = {
        "path",
        "sha256",
        "schema",
        "producer_inventory_sha256",
        "records_sha256",
        "record_schema",
        "record_count",
        "accepted_count",
    }
    if (
        not isinstance(semantic, Mapping)
        or set(semantic) != semantic_keys
        or not isinstance(semantic.get("path"), str)
        or not semantic.get("path")
        or not _valid_sha256(semantic.get("sha256"))
        or semantic.get("schema") != SEMANTIC_ACCEPTED_INVENTORY_SCHEMA
        or not _valid_sha256(semantic.get("producer_inventory_sha256"))
        or not _valid_sha256(semantic.get("records_sha256"))
        or semantic.get("record_schema") != SEMANTIC_RECORD_SCHEMA
        or isinstance(semantic.get("record_count"), bool)
        or not isinstance(semantic.get("record_count"), int)
        or isinstance(semantic.get("accepted_count"), bool)
        or not isinstance(semantic.get("accepted_count"), int)
        or semantic.get("record_count", 0)
        < semantic.get("accepted_count", 0)
        or semantic.get("accepted_count", 0) < 700
    ):
        raise QLoRAV6Error(
            "nonblind-v7 semantic inventory declaration mismatch"
        )

    builder = manifest.get("builder")
    if (
        not isinstance(builder, Mapping)
        or set(builder)
        != {
            "nonblind_module",
            "evidence_core",
            "split_algorithm_version",
            "seed",
        }
    ):
        raise QLoRAV6Error("nonblind-v7 builder declaration mismatch")
    module = builder.get("nonblind_module")
    core = builder.get("evidence_core")
    for value, suffix, label in (
        (
            module,
            "icmat_foundry/llm/nonblind_sft_v7.py",
            "nonblind module",
        ),
        (
            core,
            "icmat_foundry/llm/evidence_sft_v6.py",
            "evidence core",
        ),
    ):
        if (
            not isinstance(value, Mapping)
            or set(value) != {"path", "sha256"}
            or not isinstance(value.get("path"), str)
            or not value.get("path", "").replace("\\", "/").endswith(suffix)
            or not _valid_sha256(value.get("sha256"))
        ):
            raise QLoRAV6Error(
                f"nonblind-v7 {label} declaration mismatch"
            )
    if (
        builder.get("split_algorithm_version")
        != NONBLIND_SPLIT_ALGORITHM_VERSION
        or not isinstance(builder.get("seed"), str)
        or not builder.get("seed")
    ):
        raise QLoRAV6Error(
            "nonblind-v7 split algorithm or seed declaration mismatch"
        )
    return dict(source_inputs), dict(builder)


def _nonblind_semantic_binding_v7(
    manifest: Mapping[str, Any],
    *,
    source_inputs: Mapping[str, Any],
    snapshot: StableFileSnapshotV7,
) -> dict[str, Any]:
    artifacts = manifest.get("artifacts")
    audit_receipt = (
        artifacts.get("semantic_inventory_audit")
        if isinstance(artifacts, Mapping)
        else None
    )
    verified_receipt = _verify_strict_receipt_snapshot_v7(
        audit_receipt,
        snapshot,
        expected_path="semantic_inventory_audit.v7.json",
        label="semantic inventory audit",
    )
    audit = _strict_json_object_v7(
        snapshot,
        label="semantic_inventory_audit.v7.json",
    )
    expected_keys = {
        "schema",
        "builder_version",
        "status",
        "findings",
        "semantic_inventory_sha256",
        "producer_inventory_sha256",
        "semantic_records_sha256",
        "record_schema",
        "record_count",
        "accepted_count",
        "rejected_or_fixture_count",
        "unique_binding_count",
        "unique_record_hash_count",
        "covered_source_family_count",
        "minimum_records_per_family",
        "contract",
    }
    if set(audit) != expected_keys:
        raise QLoRAV6Error(
            "nonblind-v7 semantic audit top-level keys mismatch"
        )
    semantic = source_inputs["semantic_inventory"]
    record_count = int(semantic["record_count"])
    accepted_count = int(semantic["accepted_count"])
    if (
        audit.get("schema") != SEMANTIC_AUDIT_SCHEMA
        or audit.get("builder_version") != SEMANTIC_BUILDER_VERSION
        or audit.get("status") != "PASS"
        or audit.get("findings") != []
        or audit.get("semantic_inventory_sha256")
        != semantic.get("sha256")
        or audit.get("producer_inventory_sha256")
        != semantic.get("producer_inventory_sha256")
        or audit.get("semantic_records_sha256")
        != semantic.get("records_sha256")
        or audit.get("record_schema") != SEMANTIC_RECORD_SCHEMA
        or audit.get("record_count") != record_count
        or audit.get("accepted_count") != accepted_count
        or audit.get("unique_binding_count") != accepted_count
        or audit.get("unique_record_hash_count") != accepted_count
        or audit.get("rejected_or_fixture_count")
        != record_count - accepted_count
        or audit.get("covered_source_family_count") != 14
        or isinstance(audit.get("minimum_records_per_family"), bool)
        or not isinstance(audit.get("minimum_records_per_family"), int)
        or audit.get("minimum_records_per_family", 0) < 50
        or audit.get("contract") != _semantic_audit_contract_v7()
    ):
        raise QLoRAV6Error(
            "nonblind-v7 semantic inventory audit binding mismatch"
        )
    source_binding = dict(semantic)
    source_binding.update(
        {
            "declaration_source": NONBLIND_MANIFEST_NAME,
            "content_read": False,
            "content_hashed": False,
            "filesystem_metadata_accessed": False,
        }
    )
    return {
        "required": True,
        "source_inventory": source_binding,
        "source_binding_sha256": _canonical_sha256(dict(semantic)),
        "audit_artifact": verified_receipt,
        "audit_payload_sha256": _canonical_sha256(audit),
        "query_contract": _semantic_query_contract_v7(),
        "query_contract_sha256": _canonical_sha256(
            _semantic_query_contract_v7()
        ),
    }


def _nonblind_precommit_binding_v7(
    manifest: Mapping[str, Any],
    *,
    source_inputs: Mapping[str, Any],
    builder: Mapping[str, Any],
    snapshot: StableFileSnapshotV7,
) -> dict[str, Any]:
    artifacts = manifest.get("artifacts")
    receipt = _verify_strict_receipt_snapshot_v7(
        (
            artifacts.get("preblind_commitment")
            if isinstance(artifacts, Mapping)
            else None
        ),
        snapshot,
        expected_path="preblind_commitment.v7.json",
        label="preblind commitment",
    )
    observed = _strict_json_object_v7(
        snapshot,
        label="preblind_commitment.v7.json",
    )
    observed_sources = observed.get("source_inputs")
    if (
        set(observed)
        != {
            "schema",
            "status",
            "builder_version",
            "core_builder_version",
            "split_algorithm_version",
            "seed",
            "seed_sha256",
            "expected_blind_count",
            "builder_code",
            "source_inputs",
            "commitment_sha256",
        }
        or not isinstance(observed_sources, Mapping)
        or set(observed_sources)
        != {
            "licensed_chunks_sha256",
            "rag_manifest_sha256",
            "rag_manifest_id",
            "semantic_inventory_sha256",
            "semantic_records_sha256",
        }
    ):
        raise QLoRAV6Error(
            "nonblind-v7 preblind commitment keys mismatch"
        )
    semantic = source_inputs["semantic_inventory"]
    payload = {
        "schema": NONBLIND_PRECOMMIT_SCHEMA,
        "status": "PREBLIND_COMMITTED_NONBLIND_ONLY",
        "builder_version": NONBLIND_BUILDER_VERSION,
        "core_builder_version": SEMANTIC_BUILDER_VERSION,
        "split_algorithm_version": NONBLIND_SPLIT_ALGORITHM_VERSION,
        "seed": builder["seed"],
        "seed_sha256": hashlib.sha256(
            str(builder["seed"]).encode("utf-8")
        ).hexdigest(),
        "expected_blind_count": EXPECTED_FUTURE_BLIND_COUNT,
        "builder_code": {
            "nonblind_module_sha256": builder["nonblind_module"]["sha256"],
            "evidence_core_sha256": builder["evidence_core"]["sha256"],
        },
        "source_inputs": {
            "licensed_chunks_sha256": source_inputs["licensed_chunks"][
                "sha256"
            ],
            "rag_manifest_sha256": source_inputs["rag_manifest"]["sha256"],
            "rag_manifest_id": source_inputs["rag_manifest"]["manifest_id"],
            "semantic_inventory_sha256": semantic["sha256"],
            "semantic_records_sha256": semantic["records_sha256"],
        },
    }
    commitment = {
        **payload,
        "commitment_sha256": _canonical_sha256(payload),
    }
    if observed != commitment:
        raise QLoRAV6Error(
            "nonblind-v7 preblind commitment code/input binding mismatch"
        )
    return {
        "receipt": {
            **receipt,
            "declaration_source": NONBLIND_MANIFEST_NAME,
        },
        "commitment_sha256": commitment["commitment_sha256"],
        "code_input_binding_sha256": _canonical_sha256(
            {
                "builder_code": payload["builder_code"],
                "source_inputs": payload["source_inputs"],
                "seed_sha256": payload["seed_sha256"],
                "split_algorithm_version": payload[
                    "split_algorithm_version"
                ],
            }
        ),
        "expected_future_count": EXPECTED_FUTURE_BLIND_COUNT,
        "future_blind_boundary": {
            "blind_materialized": False,
            "blind_discovered": False,
            "blind_path_constructed": False,
            "blind_filesystem_metadata_accessed": False,
            "blind_content_opened": False,
            "blind_content_read": False,
            "blind_content_hashed": False,
        },
    }


def _validate_nonblind_aux_artifacts_v7(
    manifest: Mapping[str, Any],
    *,
    snapshots: Mapping[str, StableFileSnapshotV7],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise QLoRAV6Error("nonblind-v7 artifacts are missing")
    payloads: dict[str, dict[str, Any]] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for role, filename in STRICT_ARTIFACT_FILES.items():
        snapshot = snapshots[role]
        receipts[role] = _verify_strict_receipt_snapshot_v7(
            artifacts.get(role),
            snapshot,
            expected_path=filename,
            label=role.replace("_", " "),
        )
        payloads[role] = _strict_json_object_v7(
            snapshot,
            label=filename,
        )

    balance = payloads["balance_audit"]
    balance_keys = {
        "schema",
        "status",
        "findings",
        "split_counts",
        "split_decision_counts",
        "split_task_counts",
        "included_family_count",
        "imbalanced_family_count",
        "family_integrity",
    }
    if (
        set(balance) != balance_keys
        or balance.get("schema") != NONBLIND_BALANCE_SCHEMA
        or balance.get("status") != "PASS"
        or balance.get("findings") != []
        or balance.get("split_counts") != NONBLIND_SPLIT_COUNTS
        or balance.get("included_family_count") != NONBLIND_FAMILY_COUNT
        or balance.get("imbalanced_family_count") != 0
    ):
        raise QLoRAV6Error("nonblind-v7 balance audit contract mismatch")
    family_integrity = balance.get("family_integrity")
    if (
        not isinstance(family_integrity, Mapping)
        or set(family_integrity)
        != {
            "status",
            "findings",
            "expected_family_count",
            "observed_family_count",
            "unique_example_id_count",
            "duplicate_example_ids",
            "per_family",
        }
        or family_integrity.get("status") != "PASS"
        or family_integrity.get("findings") != []
        or family_integrity.get("expected_family_count")
        != NONBLIND_FAMILY_COUNT
        or family_integrity.get("observed_family_count")
        != NONBLIND_FAMILY_COUNT
        or family_integrity.get("unique_example_id_count")
        != NONBLIND_TOTAL_EXAMPLES
        or family_integrity.get("duplicate_example_ids") != []
        or not isinstance(family_integrity.get("per_family"), list)
        or len(family_integrity["per_family"]) != NONBLIND_FAMILY_COUNT
    ):
        raise QLoRAV6Error(
            "nonblind-v7 family integrity audit contract mismatch"
        )

    group = payloads["group_isolation_audit"]
    if (
        set(group)
        != {
            "schema",
            "status",
            "findings",
            "isolation_unit",
            "family_split_counts",
            "group_commitments",
            "pairwise",
        }
        or group.get("schema") != NONBLIND_GROUP_SCHEMA
        or group.get("status") != "PASS"
        or group.get("findings") != []
        or group.get("isolation_unit") != "licensed DOI/source family"
        or group.get("family_split_counts")
        != {"train": 5, "validation": 3, "calibration": 3}
    ):
        raise QLoRAV6Error(
            "nonblind-v7 group isolation audit contract mismatch"
        )
    pairwise = group.get("pairwise")
    if (
        not isinstance(pairwise, list)
        or len(pairwise) != 3
        or any(
            not isinstance(item, Mapping)
            or set(item)
            != {
                "left",
                "right",
                "source_overlap_count",
                "doi_overlap_count",
                "commitment_overlap_count",
            }
            or any(
                item.get(field) != 0
                for field in (
                    "source_overlap_count",
                    "doi_overlap_count",
                    "commitment_overlap_count",
                )
            )
            for item in pairwise
        )
    ):
        raise QLoRAV6Error(
            "nonblind-v7 group isolation pairwise audit mismatch"
        )

    leakage = payloads["content_leakage_audit"]
    leakage_keys = {
        "schema",
        "status",
        "findings",
        "near_duplicate_threshold",
        "exact_claim_overlap_count",
        "exact_prompt_overlap_count",
        "exact_compiler_evidence_overlap_count",
        "near_duplicate_claim_pair_count",
        "compiler_prompt_target_marker_count",
        "compiler_evidence_target_marker_count",
        "compiler_prompt_assistant_message_count",
        "compiler_interface_missing_count",
        "shortcut_audit_status",
        "shortcut_audit",
        "maximum_cross_split_claim_jaccard",
        "pointer_target_overlap_policy",
        "audited_splits",
    }
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
    if (
        set(leakage) != leakage_keys
        or leakage.get("schema") != NONBLIND_LEAKAGE_SCHEMA
        or leakage.get("status") != "PASS"
        or leakage.get("findings") != []
        or leakage.get("audited_splits")
        != ["train", "validation", "calibration"]
        or leakage.get("shortcut_audit_status") != "PASS"
        or any(leakage.get(field) != 0 for field in zero_fields)
        or not isinstance(leakage.get("shortcut_audit"), Mapping)
        or leakage["shortcut_audit"].get("status") != "PASS"
        or leakage["shortcut_audit"].get("findings") != []
    ):
        raise QLoRAV6Error(
            "nonblind-v7 content leakage audit contract mismatch"
        )

    report = payloads["build_report"]
    if (
        set(report)
        != {
            "schema",
            "status",
            "builder_version",
            "counts",
            "audits",
            "family_integrity",
            "claims",
        }
        or report.get("schema") != NONBLIND_BUILD_REPORT_SCHEMA
        or report.get("status")
        != "PASS_NONBLIND_DATASET_PREBLIND_COMMITTED"
        or report.get("builder_version") != NONBLIND_BUILDER_VERSION
        or report.get("counts")
        != {
            "examples": NONBLIND_TOTAL_EXAMPLES,
            "families": NONBLIND_FAMILY_COUNT,
            "examples_per_family": NONBLIND_EXAMPLES_PER_FAMILY,
            "splits": NONBLIND_SPLIT_COUNTS,
        }
        or report.get("family_integrity") != family_integrity
        or report.get("claims") != manifest.get("claims")
        or report.get("audits")
        != {
            "balance": "PASS",
            "family_integrity": "PASS",
            "group_isolation": "PASS",
            "content_leakage": "PASS",
            "semantic_inventory": "PASS",
            "rag_authority_binding": (
                "PASS_RAG_MANIFEST_LICENSED_CHUNKS_BOUND"
            ),
        }
    ):
        raise QLoRAV6Error(
            "nonblind-v7 build report contract mismatch"
        )
    return payloads, receipts


def _strict_implementation_snapshots_v7(
) -> dict[str, StableFileSnapshotV7]:
    paths = {
        "nonblind_builder": (
            WORKSPACE_ROOT / "icmat_foundry" / "llm" / "nonblind_sft_v7.py"
        ),
        "evidence_core": (
            WORKSPACE_ROOT / "icmat_foundry" / "llm" / "evidence_sft_v6.py"
        ),
        "nonblind_auditor": (
            WORKSPACE_ROOT
            / "icmat_foundry"
            / "llm"
            / "nonblind_sft_audit_v7.py"
        ),
        "nonblind_audit_cli": (
            WORKSPACE_ROOT / "tools" / "audit_icmat_nonblind_sft_v7.py"
        ),
        "shortcut_module": (
            WORKSPACE_ROOT / "icmat_foundry" / "llm" / "shortcut_audit_v7.py"
        ),
        "shortcut_cli": (
            WORKSPACE_ROOT / "tools" / "audit_icmat_semantic_shortcuts_v7.py"
        ),
        "trainer_module": Path(__file__),
        "trainer_cli": (
            WORKSPACE_ROOT / "tools" / "train_icmat_qlora_full_v6.py"
        ),
        "canary_acceptance": (
            WORKSPACE_ROOT / "icmat_foundry" / "llm"
            / "canary_acceptance_v6.py"
        ),
        "checkpoint_orchestrator": (
            WORKSPACE_ROOT / "icmat_foundry" / "llm"
            / "pointer_checkpoint_eval_v6.py"
        ),
        "pointer_evaluator": (
            WORKSPACE_ROOT / "icmat_foundry" / "llm"
            / "pointer_hf_eval_v6.py"
        ),
        "pointer_compiler": (
            WORKSPACE_ROOT / "icmat_foundry" / "llm"
            / "evidence_pointer_v6.py"
        ),
        "selection_policy": (
            WORKSPACE_ROOT / "icmat_foundry" / "llm"
            / "selection_policy_v6.py"
        ),
        "checkpoint_runner": (
            WORKSPACE_ROOT / "tools"
            / "evaluate_icmat_pointer_checkpoints_v6.py"
        ),
    }
    return {
        role: _stable_snapshot_v7(
            path,
            label=f"{role} implementation",
            maximum_bytes=_STRICT_MAX_JSON_BYTES,
        )
        for role, path in paths.items()
    }


def _validate_current_nonblind_builder_v7(
    builder: Mapping[str, Any],
    *,
    implementation: Mapping[str, StableFileSnapshotV7],
) -> None:
    if (
        builder["nonblind_module"]["sha256"]
        != implementation["nonblind_builder"].sha256
        or builder["evidence_core"]["sha256"]
        != implementation["evidence_core"].sha256
    ):
        raise QLoRAV6Error(
            "nonblind manifest builder hashes are not current"
        )


def _assert_nonblind_split_isolation_v7(
    summaries: Mapping[str, Mapping[str, Any]],
) -> None:
    splits = tuple(NONBLIND_SPLIT_FILES)
    for index, left in enumerate(splits):
        left_sources = set(summaries[left]["source_ids"])
        for right in splits[index + 1 :]:
            if left_sources & set(summaries[right]["source_ids"]):
                raise QLoRAV6Error(
                    f"{left}/{right} source-family leakage"
                )


def _assert_exact_nonblind_inventory_v7(
    root: Path,
    *,
    root_identity: tuple[int, int],
    label: str,
) -> None:
    expected_names = frozenset(NONBLIND_COMPARE_INVENTORY_FILES.values())
    try:
        entries = list(os.scandir(root))
    except OSError as exc:
        raise QLoRAV6Error(
            f"{label} cannot be enumerated for the fixed whitelist"
        ) from exc
    observed_names = [entry.name for entry in entries]
    if (
        len(observed_names) != len(expected_names)
        or frozenset(observed_names) != expected_names
        or len({name.casefold() for name in observed_names})
        != len(observed_names)
    ):
        raise QLoRAV6Error(
            f"{label} must contain exactly the fixed ten-file whitelist"
        )
    for entry in entries:
        try:
            metadata = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise QLoRAV6Error(
                f"{label} whitelist entry changed during enumeration"
            ) from exc
        if (
            entry.is_symlink()
            or stat.S_ISLNK(metadata.st_mode)
            or _is_reparse_point(metadata)
            or not stat.S_ISREG(metadata.st_mode)
        ):
            raise QLoRAV6Error(
                f"{label} whitelist entries must be regular non-reparse files"
            )
    _recheck_strict_directory_identity_v7(
        root,
        expected=root_identity,
        label=label,
    )
    try:
        names_after = frozenset(entry.name for entry in os.scandir(root))
    except OSError as exc:
        raise QLoRAV6Error(
            f"{label} changed during whitelist verification"
        ) from exc
    if names_after != expected_names:
        raise QLoRAV6Error(
            f"{label} changed during whitelist verification"
        )


def _fixed_nonblind_file_snapshots_v7(
    root: Path,
    *,
    label: str,
    manifest_snapshot: StableFileSnapshotV7 | None = None,
) -> dict[str, StableFileSnapshotV7]:
    snapshots: dict[str, StableFileSnapshotV7] = {}
    for role, filename in NONBLIND_COMPARE_INVENTORY_FILES.items():
        if role == "manifest" and manifest_snapshot is not None:
            snapshots[role] = manifest_snapshot
            continue
        maximum_bytes = (
            _STRICT_MAX_JSONL_BYTES
            if role in NONBLIND_SPLIT_FILES
            else _STRICT_MAX_JSON_BYTES
        )
        snapshots[role] = _stable_snapshot_v7(
            root / filename,
            label=f"{label} {role}",
            maximum_bytes=maximum_bytes,
        )
    return snapshots


def _validate_second_nonblind_build_v7(
    path: Path,
    *,
    primary_root: Path,
    primary_manifest: Mapping[str, Any],
    primary_source_inputs: Mapping[str, Any],
    primary_builder: Mapping[str, Any],
    implementation: Mapping[str, StableFileSnapshotV7],
) -> dict[str, Any]:
    root, root_identity = _strict_directory_identity_v7(
        path,
        label="nonblind second build directory",
    )
    if root == primary_root:
        raise QLoRAV6Error(
            "nonblind second build directory must be distinct"
        )
    _assert_exact_nonblind_inventory_v7(
        root,
        root_identity=root_identity,
        label="nonblind second build directory",
    )
    try:
        os.lstat(root / MANIFEST_NAME)
    except FileNotFoundError:
        pass
    else:
        raise QLoRAV6Error(
            "nonblind second build contains both manifest.v6.json and "
            "manifest.nonblind.v7.json"
        )

    files = _fixed_nonblind_file_snapshots_v7(
        root,
        label="nonblind second build",
    )
    manifest = _strict_json_object_v7(
        files["manifest"],
        label="nonblind second build manifest",
    )
    if manifest != primary_manifest:
        raise QLoRAV6Error(
            "nonblind second build manifest differs from primary"
        )
    records = _nonblind_split_records(manifest)
    source_inputs, builder = _nonblind_source_and_builder_declarations(
        manifest
    )
    if (
        source_inputs != primary_source_inputs
        or builder != primary_builder
    ):
        raise QLoRAV6Error(
            "nonblind second build source or implementation "
            "declarations differ"
        )
    _validate_current_nonblind_builder_v7(
        builder,
        implementation=implementation,
    )

    split_snapshots = {
        split: files[split] for split in NONBLIND_SPLIT_FILES
    }
    artifact_snapshots = {
        role: files[role] for role in STRICT_ARTIFACT_FILES
    }
    artifact_payloads, artifact_receipts = (
        _validate_nonblind_aux_artifacts_v7(
            manifest,
            snapshots=artifact_snapshots,
        )
    )
    seen_example_ids: set[str] = set()
    summaries = {
        split: _scan_strict_visible_snapshot_v7(
            split_snapshots[split],
            split=split,
            expected=records[split],
            seen_example_ids=seen_example_ids,
        )
        for split in NONBLIND_SPLIT_FILES
    }
    _assert_nonblind_split_isolation_v7(summaries)
    semantic_binding = _nonblind_semantic_binding_v7(
        manifest,
        source_inputs=source_inputs,
        snapshot=artifact_snapshots["semantic_inventory_audit"],
    )
    preblind_binding = _nonblind_precommit_binding_v7(
        manifest,
        source_inputs=source_inputs,
        builder=builder,
        snapshot=artifact_snapshots["preblind_commitment"],
    )
    _recheck_strict_directory_identity_v7(
        root,
        expected=root_identity,
        label="nonblind second build directory",
    )
    _assert_exact_nonblind_inventory_v7(
        root,
        root_identity=root_identity,
        label="nonblind second build directory",
    )
    return {
        "root": root,
        "root_identity": root_identity,
        "files": files,
        "manifest": manifest,
        "records": records,
        "source_inputs": source_inputs,
        "builder": builder,
        "summaries": summaries,
        "artifact_payloads": artifact_payloads,
        "artifact_receipts": artifact_receipts,
        "semantic_binding": semantic_binding,
        "preblind_binding": preblind_binding,
    }


def _compare_nonblind_build_snapshots_v7(
    *,
    primary_root: Path,
    primary_root_identity: tuple[int, int],
    primary_files: Mapping[str, StableFileSnapshotV7],
    second: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    second_root = second["root"]
    second_root_identity = second["root_identity"]
    if (
        primary_root == second_root
        or primary_root_identity == second_root_identity
    ):
        raise QLoRAV6Error(
            "the two nonblind build directories are not independent"
        )
    second_files = second["files"]
    file_receipts: dict[str, Any] = {}
    for role in NONBLIND_COMPARE_INVENTORY_FILES:
        primary = primary_files[role]
        alternate = second_files[role]
        if primary.identity[:2] == alternate.identity[:2]:
            raise QLoRAV6Error(
                f"nonblind second build {role} shares file identity"
            )
        if (
            primary.byte_count != alternate.byte_count
            or primary.sha256 != alternate.sha256
            or primary.payload != alternate.payload
        ):
            raise QLoRAV6Error(
                f"nonblind second build {role} is not byte-identical"
            )
        file_receipts[role] = {
            "path": NONBLIND_COMPARE_INVENTORY_FILES[role],
            "bytes": alternate.byte_count,
            "sha256": alternate.sha256,
            "primary_stable_identity": primary.identity_receipt(),
            "second_stable_identity": alternate.identity_receipt(),
            "byte_identical_recomputed": True,
            "file_identity_distinct_recomputed": True,
        }
    actual_double_build = {
        "directories_distinct": True,
        "artifact_file_identities_distinct": True,
        "fixed_whitelist_file_count": len(
            NONBLIND_COMPARE_INVENTORY_FILES
        ),
        "all_whitelist_files_byte_identical": True,
        "dataset_a_root_fingerprint_sha256": hashlib.sha256(
            os.fspath(primary_root).encode("utf-8")
        ).hexdigest(),
        "dataset_b_root_fingerprint_sha256": hashlib.sha256(
            os.fspath(second_root).encode("utf-8")
        ).hexdigest(),
        "manifest_sha256": primary_files["manifest"].sha256,
    }
    evidence = {
        "second_build_dir": str(second_root),
        "primary_root_identity": {
            "device": primary_root_identity[0],
            "file_id": primary_root_identity[1],
        },
        "second_root_identity": {
            "device": second_root_identity[0],
            "file_id": second_root_identity[1],
        },
        "fixed_whitelist_file_count": len(file_receipts),
        "fixed_files": file_receipts,
        "actual_byte_comparison": True,
        "actual_file_identity_comparison": True,
        "receipt_used_as_corroboration_only": True,
    }
    return actual_double_build, evidence


def _validate_compare_audit_gate_v7(
    gate_path: Path,
    *,
    manifest: Mapping[str, Any],
    artifact_payloads: Mapping[str, Mapping[str, Any]],
    implementation: Mapping[str, StableFileSnapshotV7],
    fixed_file_snapshots: Mapping[str, StableFileSnapshotV7],
    actual_double_build: Mapping[str, Any],
) -> dict[str, Any]:
    gate_snapshot = _stable_snapshot_v7(
        gate_path,
        label="nonblind compare audit receipt",
        maximum_bytes=_STRICT_MAX_JSON_BYTES,
    )
    receipt = _strict_json_object_v7(
        gate_snapshot,
        label="nonblind compare audit receipt",
    )
    expected_keys = {
        "schema",
        "version",
        "created_at_utc",
        "mode",
        "status",
        "audit_passed",
        "dataset_contract",
        "artifact_inventory",
        "source_bindings",
        "implementation",
        "double_build",
        "reserved_asset_boundary",
        "claims",
        "pre_write_evidence_recheck",
        "canonical_digest_sha256",
        "receipt_payload_sha256",
    }
    if set(receipt) != expected_keys:
        raise QLoRAV6Error("nonblind compare audit receipt keys mismatch")
    payload_digest = receipt.get("receipt_payload_sha256")
    payload_core = {
        key: value
        for key, value in receipt.items()
        if key != "receipt_payload_sha256"
    }
    if (
        not _valid_sha256(payload_digest)
        or payload_digest != _canonical_sha256(payload_core)
    ):
        raise QLoRAV6Error(
            "nonblind compare audit payload digest mismatch"
        )
    canonical_digest = payload_core.pop("canonical_digest_sha256", None)
    if (
        not _valid_sha256(canonical_digest)
        or canonical_digest != _canonical_sha256(payload_core)
    ):
        raise QLoRAV6Error(
            "nonblind compare audit canonical digest mismatch"
        )
    if (
        receipt.get("schema") != NONBLIND_AUDIT_SCHEMA
        or receipt.get("version") != NONBLIND_AUDIT_VERSION
        or receipt.get("mode") != "compare"
        or receipt.get("status") != NONBLIND_COMPARE_STATUS
        or receipt.get("audit_passed") is not True
        or receipt.get("pre_write_evidence_recheck") is not True
    ):
        raise QLoRAV6Error(
            "nonblind compare audit did not grant the strict gate"
        )
    created_at = receipt.get("created_at_utc")
    if not isinstance(created_at, str) or not created_at:
        raise QLoRAV6Error(
            "nonblind compare audit timestamp is missing"
        )
    try:
        datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise QLoRAV6Error(
            "nonblind compare audit timestamp is invalid"
        ) from exc
    dataset_contract = receipt.get("dataset_contract")
    if (
        not isinstance(dataset_contract, Mapping)
        or set(dataset_contract)
        != {
            "manifest_schema",
            "manifest_version",
            "manifest_status",
            "dataset_schema",
            "counts",
            "pointer_rows_structurally_revalidated",
            "source_family_isolation_recomputed",
            "content_leakage_recomputed",
            "semantic_inventory_and_records_rebound",
            "preblind_commitment_recomputed",
        }
        or dataset_contract.get("manifest_schema")
        != NONBLIND_MANIFEST_SCHEMA
        or dataset_contract.get("manifest_version")
        != NONBLIND_BUILDER_VERSION
        or dataset_contract.get("manifest_status")
        != "NONBLIND_DATASET_BUILT_PREBLIND_COMMITTED"
        or dataset_contract.get("dataset_schema") != DATASET_SCHEMA
        or dataset_contract.get("counts")
        != {
            "total": NONBLIND_TOTAL_EXAMPLES,
            "splits": NONBLIND_SPLIT_COUNTS,
        }
        or dataset_contract.get("pointer_rows_structurally_revalidated")
        != NONBLIND_TOTAL_EXAMPLES
        or any(
            dataset_contract.get(field) is not True
            for field in (
                "source_family_isolation_recomputed",
                "content_leakage_recomputed",
                "semantic_inventory_and_records_rebound",
                "preblind_commitment_recomputed",
            )
        )
    ):
        raise QLoRAV6Error(
            "nonblind compare audit dataset contract mismatch"
        )

    inventory = receipt.get("artifact_inventory")
    files = (
        inventory.get("files")
        if isinstance(inventory, Mapping)
        else None
    )
    if (
        not isinstance(inventory, Mapping)
        or set(inventory)
        != {
            "fixed_whitelist_file_count",
            "directory_enumerated",
            "files",
        }
        or inventory.get("fixed_whitelist_file_count") != 10
        or inventory.get("directory_enumerated") is not False
        or not isinstance(files, Mapping)
        or set(files) != set(NONBLIND_COMPARE_INVENTORY_FILES)
    ):
        raise QLoRAV6Error(
            "nonblind compare audit artifact inventory mismatch"
        )
    for role, filename in NONBLIND_COMPARE_INVENTORY_FILES.items():
        item = files.get(role)
        expected_keys_for_item = {
            "path",
            "bytes",
            "sha256",
            "safely_parsed",
            "byte_identical",
        }
        if role in NONBLIND_SPLIT_FILES:
            expected_keys_for_item.add("count")
        if (
            not isinstance(item, Mapping)
            or set(item) != expected_keys_for_item
            or item.get("path") != filename
            or item.get("safely_parsed") is not True
            or item.get("byte_identical") is not True
        ):
            raise QLoRAV6Error(
                f"nonblind compare audit {role} inventory mismatch"
            )
        fixed_snapshot = fixed_file_snapshots[role]
        expected_bytes = fixed_snapshot.byte_count
        expected_sha256 = fixed_snapshot.sha256
        if (
            item.get("bytes") != expected_bytes
            or item.get("sha256") != expected_sha256
            or (
                role in NONBLIND_SPLIT_FILES
                and item.get("count") != NONBLIND_SPLIT_COUNTS[role]
            )
        ):
            raise QLoRAV6Error(
                f"nonblind compare audit {role} bytes/hash/count mismatch"
            )

    double_build = receipt.get("double_build")
    if (
        not isinstance(double_build, Mapping)
        or dict(double_build) != dict(actual_double_build)
    ):
        raise QLoRAV6Error(
            "nonblind compare audit double-build binding mismatch"
        )

    source_bindings = receipt.get("source_bindings")
    source_inputs = manifest["source_inputs"]
    semantic = source_inputs["semantic_inventory"]
    commitment = artifact_payloads["preblind_commitment"]
    commitment_sources = commitment["source_inputs"]
    if (
        not isinstance(source_bindings, Mapping)
        or set(source_bindings)
        != {
            "licensed_chunks",
            "rag_manifest",
            "semantic_inventory",
            "semantic_records",
            "seed_sha256",
            "split_algorithm_version",
        }
        or source_bindings.get("seed_sha256")
        != commitment.get("seed_sha256")
        or source_bindings.get("split_algorithm_version")
        != NONBLIND_SPLIT_ALGORITHM_VERSION
    ):
        raise QLoRAV6Error(
            "nonblind compare audit source bindings mismatch"
        )
    licensed_binding = source_bindings.get("licensed_chunks")
    rag_binding = source_bindings.get("rag_manifest")
    semantic_binding = source_bindings.get("semantic_inventory")
    records_binding = source_bindings.get("semantic_records")
    if (
        not isinstance(licensed_binding, Mapping)
        or set(licensed_binding) != {"bytes", "sha256"}
        or licensed_binding.get("sha256")
        != source_inputs["licensed_chunks"]["sha256"]
        or licensed_binding.get("sha256")
        != commitment_sources["licensed_chunks_sha256"]
        or isinstance(licensed_binding.get("bytes"), bool)
        or not isinstance(licensed_binding.get("bytes"), int)
        or licensed_binding.get("bytes", 0) < 1
        or not isinstance(rag_binding, Mapping)
        or set(rag_binding)
        != {
            "schema",
            "manifest_id",
            "bytes",
            "sha256",
            "authority_binding_sha256",
        }
        or rag_binding.get("schema") != "icmat.rag.manifest.v2"
        or rag_binding.get("manifest_id")
        != source_inputs["rag_manifest"]["manifest_id"]
        or rag_binding.get("sha256")
        != source_inputs["rag_manifest"]["sha256"]
        or rag_binding.get("sha256")
        != commitment_sources["rag_manifest_sha256"]
        or not _valid_sha256(
            rag_binding.get("authority_binding_sha256")
        )
        or isinstance(rag_binding.get("bytes"), bool)
        or not isinstance(rag_binding.get("bytes"), int)
        or rag_binding.get("bytes", 0) < 1
        or not isinstance(semantic_binding, Mapping)
        or set(semantic_binding) != {"schema", "bytes", "sha256"}
        or semantic_binding.get("schema")
        != SEMANTIC_ACCEPTED_INVENTORY_SCHEMA
        or semantic_binding.get("sha256") != semantic["sha256"]
        or semantic_binding.get("sha256")
        != commitment_sources["semantic_inventory_sha256"]
        or isinstance(semantic_binding.get("bytes"), bool)
        or not isinstance(semantic_binding.get("bytes"), int)
        or semantic_binding.get("bytes", 0) < 1
        or not isinstance(records_binding, Mapping)
        or set(records_binding) != {"schema", "bytes", "sha256"}
        or records_binding.get("schema") != SEMANTIC_RECORD_SCHEMA
        or records_binding.get("sha256") != semantic["records_sha256"]
        or records_binding.get("sha256")
        != commitment_sources["semantic_records_sha256"]
        or isinstance(records_binding.get("bytes"), bool)
        or not isinstance(records_binding.get("bytes"), int)
        or records_binding.get("bytes", 0) < 1
    ):
        raise QLoRAV6Error(
            "nonblind compare audit source hash binding mismatch"
        )

    builder = manifest["builder"]
    _validate_current_nonblind_builder_v7(
        builder,
        implementation=implementation,
    )
    declared_implementation = receipt.get("implementation")
    implementation_roles = {
        "nonblind_builder": "nonblind_builder",
        "evidence_core": "evidence_core",
        "independent_auditor": "nonblind_auditor",
        "audit_cli": "nonblind_audit_cli",
    }
    if (
        not isinstance(declared_implementation, Mapping)
        or set(declared_implementation) != set(implementation_roles)
    ):
        raise QLoRAV6Error(
            "nonblind compare audit implementation inventory mismatch"
        )
    for declared_role, snapshot_role in implementation_roles.items():
        item = declared_implementation.get(declared_role)
        current = implementation[snapshot_role]
        if (
            not isinstance(item, Mapping)
            or set(item) != {"filename", "bytes", "sha256"}
            or item.get("filename") != current.path.name
            or item.get("bytes") != current.byte_count
            or item.get("sha256") != current.sha256
        ):
            raise QLoRAV6Error(
                "nonblind compare audit implementation hash mismatch"
            )
    if receipt.get("reserved_asset_boundary") != {
        "path_accessed": False,
        "path_discovered": False,
        "read": False,
        "hashed": False,
        "stat_called": False,
        "directory_scanned": False,
        "content_disclosed": False,
        "expected_reserved_count_only": EXPECTED_FUTURE_BLIND_COUNT,
    }:
        raise QLoRAV6Error(
            "nonblind compare audit reserved boundary mismatch"
        )
    if receipt.get("claims") != {
        "training_or_checkpoint_selection_performed": False,
        "model_quality_authorized": False,
        "x5_contacted_or_verified": False,
    }:
        raise QLoRAV6Error(
            "nonblind compare audit claims mismatch"
        )
    return {
        "path": str(gate_snapshot.path),
        "bytes": gate_snapshot.byte_count,
        "sha256": gate_snapshot.sha256,
        "stable_identity": gate_snapshot.identity_receipt(),
        "schema": NONBLIND_AUDIT_SCHEMA,
        "version": NONBLIND_AUDIT_VERSION,
        "status": NONBLIND_COMPARE_STATUS,
        "mode": "compare",
        "canonical_digest_sha256": canonical_digest,
        "receipt_payload_sha256": payload_digest,
        "audit_passed": True,
        "actual_double_build_recomputed": True,
        "receipt_used_as_corroboration_only": True,
    }


def _recompute_shortcut_gate_v7(
    *,
    split: str,
    split_snapshot: StableFileSnapshotV7,
    expected_count: int,
    implementation: Mapping[str, StableFileSnapshotV7],
) -> dict[str, Any]:
    if (
        shortcut_contract.AUDIT_SCHEMA != SHORTCUT_AUDIT_SCHEMA
        or shortcut_contract.AUDIT_VERSION != SHORTCUT_AUDIT_VERSION
        or shortcut_contract.PASS_STATUS != SHORTCUT_PASS_STATUS
        or shortcut_contract.SAMPLE_SCHEMA != SHORTCUT_SAMPLE_SCHEMA
    ):
        raise QLoRAV6Error(
            "shortcut audit v7 interface changed; adapter update required"
        )
    shortcut_snapshot = shortcut_contract.FileSnapshot(
        path=split_snapshot.path,
        payload=split_snapshot.payload,
        sha256=split_snapshot.sha256,
        size_bytes=split_snapshot.byte_count,
        identity=split_snapshot.identity[:4],
    )
    try:
        rows = shortcut_contract.load_training_jsonl(shortcut_snapshot)
        samples, analysis = shortcut_contract.analyze_rows(rows)
        per_sample_payload = shortcut_contract._jsonl_bytes(samples)
    except shortcut_contract.ShortcutAuditV7Error as exc:
        raise QLoRAV6Error(
            f"{split} shortcut local recomputation failed: {exc}"
        ) from exc
    if len(rows) != expected_count or len(samples) != expected_count:
        raise QLoRAV6Error(
            f"{split} shortcut local recomputation count mismatch"
        )
    runner = implementation["shortcut_cli"]
    module = implementation["shortcut_module"]
    report: dict[str, Any] = {
        "schema": shortcut_contract.AUDIT_SCHEMA,
        "audit_version": shortcut_contract.AUDIT_VERSION,
        **analysis,
        "scope": {
            "allowed_splits": sorted(shortcut_contract.ALLOWED_SPLITS),
            "calibration_read": False,
            "blind_read": False,
            "model_loaded": False,
            "training_performed": False,
            "selection_authorized": False,
            "deployment_authorized": False,
            "production_activation_authorized": False,
        },
        "input": {
            "path": split_snapshot.path.as_posix(),
            "sha256": split_snapshot.sha256,
            "bytes": split_snapshot.byte_count,
            "example_count": len(rows),
        },
        "artifacts": {
            "per_sample": {
                "path": "per_sample.v7.jsonl",
                "sha256": hashlib.sha256(
                    per_sample_payload
                ).hexdigest(),
                "bytes": len(per_sample_payload),
                "count": len(samples),
            },
            "runner": {
                "path": runner.path.as_posix(),
                "sha256": runner.sha256,
                "bytes": runner.byte_count,
            },
            "module": {
                "path": module.path.as_posix(),
                "sha256": module.sha256,
            },
        },
    }
    canonical_digest = hashlib.sha256(
        shortcut_contract.canonical_json(report).encode("utf-8")
    ).hexdigest()
    report["audit_id"] = f"icm-shortcut-v7:{canonical_digest}"
    report["canonical_digest_sha256"] = canonical_digest
    if (
        report.get("status") != SHORTCUT_PASS_STATUS
        or any(
            item.get("passed") is not True
            for item in report.get("hard_gates", ())
            if isinstance(item, Mapping)
        )
        or len(report.get("hard_gates", ())) != 5
    ):
        raise QLoRAV6Error(
            f"{split} shortcut local recomputation did not pass"
        )
    return {
        "report": report,
        "per_sample_payload": per_sample_payload,
        "samples": samples,
        "canonical_digest_sha256": canonical_digest,
    }


def _validate_shortcut_gate_v7(
    gate_path: Path,
    *,
    split: str,
    split_snapshot: StableFileSnapshotV7,
    expected_count: int,
    implementation: Mapping[str, StableFileSnapshotV7],
) -> dict[str, Any]:
    report_snapshot = _stable_snapshot_v7(
        gate_path,
        label=f"{split} shortcut audit",
        maximum_bytes=_STRICT_MAX_JSON_BYTES,
    )
    if report_snapshot.path.name != "audit.v7.json":
        raise QLoRAV6Error(
            f"{split} shortcut audit filename must be audit.v7.json"
        )
    report = _strict_json_object_v7(
        report_snapshot,
        label=f"{split} shortcut audit",
    )
    recomputed = _recompute_shortcut_gate_v7(
        split=split,
        split_snapshot=split_snapshot,
        expected_count=expected_count,
        implementation=implementation,
    )
    if report != recomputed["report"]:
        raise QLoRAV6Error(
            f"{split} shortcut audit does not match local recomputation"
        )
    expected_keys = {
        "schema",
        "audit_version",
        "status",
        "counts",
        "thresholds",
        "hard_gates",
        "baselines",
        "stratified",
        "hard_cases",
        "lexical_overlap",
        "duplicates",
        "scope",
        "input",
        "artifacts",
        "audit_id",
        "canonical_digest_sha256",
    }
    if set(report) != expected_keys:
        raise QLoRAV6Error(f"{split} shortcut audit keys mismatch")
    canonical_digest = report.get("canonical_digest_sha256")
    canonical_body = {
        key: value
        for key, value in report.items()
        if key not in {"audit_id", "canonical_digest_sha256"}
    }
    if (
        not _valid_sha256(canonical_digest)
        or canonical_digest != _canonical_sha256(canonical_body)
        or report.get("audit_id")
        != f"icm-shortcut-v7:{canonical_digest}"
        or report.get("schema") != SHORTCUT_AUDIT_SCHEMA
        or report.get("audit_version") != SHORTCUT_AUDIT_VERSION
        or report.get("status") != SHORTCUT_PASS_STATUS
    ):
        raise QLoRAV6Error(
            f"{split} shortcut audit digest or status mismatch"
        )
    split_rows = _strict_jsonl_rows_v7(
        split_snapshot,
        label=f"{split} shortcut-bound input",
    )
    expected_example_ids = {
        str(row["example_id"]) for row in split_rows
    }
    expected_counts = {
        "examples": expected_count,
        "splits": {split: expected_count},
        "domains": dict(
            sorted(Counter(str(row["domain"]) for row in split_rows).items())
        ),
        "tasks": dict(
            sorted(Counter(str(row["task"]) for row in split_rows).items())
        ),
        "decisions": dict(
            sorted(
                Counter(str(row["decision"]) for row in split_rows).items()
            )
        ),
        "normalized_exact_copy": 0,
    }
    if report.get("counts") != expected_counts:
        raise QLoRAV6Error(f"{split} shortcut count binding mismatch")
    hard_gates = report.get("hard_gates")
    expected_gate_names = {
        "normalized_exact_copy_count_is_zero",
        "normalized_presence_decision_below_usable_accuracy",
        "bm25_presence_nearest_strict_below_model_floor",
        "answer_high_overlap_hard_cases_present",
        "refuse_high_overlap_hard_cases_present",
    }
    if (
        not isinstance(hard_gates, list)
        or len(hard_gates) != len(expected_gate_names)
        or {
            item.get("gate")
            for item in hard_gates
            if isinstance(item, Mapping)
        }
        != expected_gate_names
        or any(
            not isinstance(item, Mapping)
            or set(item)
            != (
                {"gate", "threshold", "observed", "passed"}
                if item.get("gate")
                == "normalized_exact_copy_count_is_zero"
                else {
                    "gate",
                    "operator",
                    "threshold",
                    "observed",
                    "passed",
                }
            )
            or item.get("passed") is not True
            for item in hard_gates
        )
    ):
        raise QLoRAV6Error(f"{split} shortcut hard gates did not all pass")
    input_binding = report.get("input")
    expected_input_path = split_snapshot.path.as_posix()
    if (
        not isinstance(input_binding, Mapping)
        or set(input_binding)
        != {"path", "sha256", "bytes", "example_count"}
        or input_binding.get("path") != expected_input_path
        or input_binding.get("sha256") != split_snapshot.sha256
        or input_binding.get("bytes") != split_snapshot.byte_count
        or input_binding.get("example_count") != expected_count
    ):
        raise QLoRAV6Error(
            f"{split} shortcut input binding mismatch"
        )
    scope = report.get("scope")
    if (
        not isinstance(scope, Mapping)
        or scope
        != {
            "allowed_splits": ["train", "validation"],
            "calibration_read": False,
            "blind_read": False,
            "model_loaded": False,
            "training_performed": False,
            "selection_authorized": False,
            "deployment_authorized": False,
            "production_activation_authorized": False,
        }
    ):
        raise QLoRAV6Error(f"{split} shortcut scope boundary mismatch")
    artifacts = report.get("artifacts")
    if (
        not isinstance(artifacts, Mapping)
        or set(artifacts) != {"per_sample", "runner", "module"}
    ):
        raise QLoRAV6Error(f"{split} shortcut artifact receipts mismatch")
    per_sample_receipt = artifacts["per_sample"]
    if (
        not isinstance(per_sample_receipt, Mapping)
        or set(per_sample_receipt)
        != {"path", "sha256", "bytes", "count"}
        or per_sample_receipt.get("path") != "per_sample.v7.jsonl"
    ):
        raise QLoRAV6Error(
            f"{split} shortcut per-sample receipt mismatch"
        )
    per_sample_path = report_snapshot.path.parent / "per_sample.v7.jsonl"
    per_sample_snapshot = _stable_snapshot_v7(
        per_sample_path,
        label=f"{split} shortcut per-sample artifact",
        maximum_bytes=_STRICT_MAX_JSONL_BYTES,
    )
    if per_sample_snapshot.payload != recomputed["per_sample_payload"]:
        raise QLoRAV6Error(
            f"{split} shortcut per-sample does not match local "
            "recomputation"
        )
    if (
        per_sample_receipt.get("sha256") != per_sample_snapshot.sha256
        or per_sample_receipt.get("bytes")
        != per_sample_snapshot.byte_count
        or per_sample_receipt.get("count") != expected_count
    ):
        raise QLoRAV6Error(
            f"{split} shortcut per-sample bytes/hash/count mismatch"
        )
    samples = _strict_jsonl_rows_v7(
        per_sample_snapshot,
        label=f"{split} shortcut per-sample artifact",
    )
    sample_keys = {
        "schema",
        "example_id",
        "split",
        "domain",
        "task",
        "gold",
        "requested_claim_sha256",
        "lexical",
        "target_jaccard",
        "normalized_exact_copy",
        "high_overlap_hard_case",
        "predictions",
        "scores",
    }
    if (
        len(samples) != expected_count
        or {
            sample.get("example_id")
            for sample in samples
            if isinstance(sample.get("example_id"), str)
        }
        != expected_example_ids
        or any(
            set(sample) != sample_keys
            or sample.get("schema") != SHORTCUT_SAMPLE_SCHEMA
            or sample.get("split") != split
            for sample in samples
        )
    ):
        raise QLoRAV6Error(
            f"{split} shortcut per-sample contract mismatch"
        )
    runner = artifacts["runner"]
    current_runner = implementation["shortcut_cli"]
    if (
        not isinstance(runner, Mapping)
        or set(runner) != {"path", "sha256", "bytes"}
        or runner.get("path") != current_runner.path.as_posix()
        or runner.get("sha256") != current_runner.sha256
        or runner.get("bytes") != current_runner.byte_count
    ):
        raise QLoRAV6Error(
            f"{split} shortcut runner implementation hash mismatch"
        )
    module = artifacts["module"]
    current_module = implementation["shortcut_module"]
    if (
        not isinstance(module, Mapping)
        or set(module) != {"path", "sha256"}
        or module.get("path") != current_module.path.as_posix()
        or module.get("sha256") != current_module.sha256
    ):
        raise QLoRAV6Error(
            f"{split} shortcut module implementation hash mismatch"
        )
    return {
        "path": str(report_snapshot.path),
        "bytes": report_snapshot.byte_count,
        "sha256": report_snapshot.sha256,
        "stable_identity": report_snapshot.identity_receipt(),
        "per_sample": {
            "path": str(per_sample_snapshot.path),
            "bytes": per_sample_snapshot.byte_count,
            "sha256": per_sample_snapshot.sha256,
            "count": len(samples),
            "stable_identity": per_sample_snapshot.identity_receipt(),
        },
        "schema": SHORTCUT_AUDIT_SCHEMA,
        "version": SHORTCUT_AUDIT_VERSION,
        "status": SHORTCUT_PASS_STATUS,
        "canonical_digest_sha256": canonical_digest,
        "hard_gates_passed": True,
        "full_report_locally_recomputed": True,
        "per_sample_bytes_locally_recomputed": True,
    }


def _nonblind_dataset_snapshot_v7(
    root: Path,
    *,
    nonblind_second_build_dir: Path | None,
    nonblind_audit_receipt: Path | None,
    train_shortcut_audit: Path | None,
    validation_shortcut_audit: Path | None,
) -> dict[str, Any]:
    gate_paths = {
        "nonblind_second_build_dir": nonblind_second_build_dir,
        "nonblind_audit_receipt": nonblind_audit_receipt,
        "train_shortcut_audit": train_shortcut_audit,
        "validation_shortcut_audit": validation_shortcut_audit,
    }
    missing_gates = [
        name for name, path in gate_paths.items() if path is None
    ]
    if missing_gates:
        raise QLoRAV6Error(
            "strict nonblind-v7 requires explicit audit gates: "
            + ", ".join(missing_gates)
        )

    root, primary_root_identity = _strict_directory_identity_v7(
        root,
        label="nonblind primary build directory",
    )
    _assert_exact_nonblind_inventory_v7(
        root,
        root_identity=primary_root_identity,
        label="nonblind primary build directory",
    )
    manifest_path = root / NONBLIND_MANIFEST_NAME
    manifest_snapshot = _stable_snapshot_v7(
        manifest_path,
        label=NONBLIND_MANIFEST_NAME,
        maximum_bytes=_STRICT_MAX_JSON_BYTES,
    )
    manifest = _strict_json_object_v7(
        manifest_snapshot,
        label=NONBLIND_MANIFEST_NAME,
    )
    expected_top_level = {
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
        "builder",
        "counts",
        "pointer_contract",
        "compiler_input_contract",
        "external_answer_contract",
        "training_boundary",
        "claims",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != expected_top_level:
        raise QLoRAV6Error("nonblind-v7 manifest keys mismatch")
    serialized_manifest = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if any(
        fragment in serialized_manifest
        for fragment in (
            "blind_test",
            "sealed.v",
            "blind_path",
            "blind_sha256",
            "blind_bytes",
            "blind_content",
        )
    ):
        raise QLoRAV6Error(
            "nonblind-v7 manifest discloses a forbidden blind artifact"
        )
    if (
        manifest.get("schema") != NONBLIND_MANIFEST_SCHEMA
        or manifest.get("dataset_schema") != DATASET_SCHEMA
        or manifest.get("builder_version") != NONBLIND_BUILDER_VERSION
        or manifest.get("core_builder_version")
        != SEMANTIC_BUILDER_VERSION
        or manifest.get("status")
        != "NONBLIND_DATASET_BUILT_PREBLIND_COMMITTED"
    ):
        raise QLoRAV6Error("nonblind-v7 manifest identity mismatch")
    if (
        manifest.get("ground_truth_policy")
        != (
            "deterministic pointer labels from licensed evidence; "
            "no API or teacher output is ground truth"
        )
        or manifest.get("selection_policy")
        != "researcher_explicit_domain_and_task"
        or manifest.get("source_isolation_unit") != "DOI/source_family"
    ):
        raise QLoRAV6Error("nonblind-v7 manifest policy mismatch")
    if manifest.get("counts") != {
        "examples": NONBLIND_TOTAL_EXAMPLES,
        "families": NONBLIND_FAMILY_COUNT,
        "examples_per_family": NONBLIND_EXAMPLES_PER_FAMILY,
        "splits": NONBLIND_SPLIT_COUNTS,
    }:
        raise QLoRAV6Error("nonblind-v7 fixed counts mismatch")
    if manifest.get("pointer_contract") != {
        "field_order": list(POINTER_FIELDS),
        "answer_span_pattern": "E#.S#",
        "refusal_span_id": None,
    }:
        raise QLoRAV6Error("nonblind-v7 pointer contract mismatch")
    if manifest.get("compiler_input_contract") != {
        "compiler_version": evidence_contract.COMPILER_VERSION,
        "prompt_schema": evidence_contract.COMPILER_PROMPT_SCHEMA,
        "compiler_prompt_keys": sorted(
            evidence_contract.COMPILER_PROMPT_FIELDS
        ),
        "compiler_evidence_keys": sorted(
            evidence_contract.COMPILER_EVIDENCE_FIELDS
        ),
        "compiler_sentence_keys": sorted(
            evidence_contract.COMPILER_SENTENCE_FIELDS
        ),
        "target_free": True,
        "user_text_reverse_parsing_required": False,
    }:
        raise QLoRAV6Error("nonblind-v7 compiler input contract mismatch")
    if manifest.get("external_answer_contract") != {
        "schema": evidence_contract.EXTERNAL_ANSWER_SCHEMA,
        "field_order": list(evidence_contract.EXTERNAL_ANSWER_FIELDS),
        "generated_by": "later_deterministic_evidence_compiler",
        "implemented_by_this_builder": False,
    }:
        raise QLoRAV6Error(
            "nonblind-v7 external answer contract mismatch"
        )
    if manifest.get("training_boundary") != {
        "allowed_splits": list(READABLE_SPLITS),
        "calibration_content_for_training": False,
    }:
        raise QLoRAV6Error("nonblind-v7 training boundary mismatch")
    if manifest.get("claims") != {
        "nonblind_only": True,
        "training_authorized_splits": list(READABLE_SPLITS),
        "calibration_for_training": False,
        "production_connected": False,
        "x5_deployed": False,
    }:
        raise QLoRAV6Error("nonblind-v7 claims boundary mismatch")

    artifacts = manifest.get("artifacts")
    if (
        not isinstance(artifacts, Mapping)
        or set(artifacts) != set(STRICT_ARTIFACT_FILES)
    ):
        raise QLoRAV6Error("nonblind-v7 artifact receipts mismatch")
    for key, expected_path in STRICT_ARTIFACT_FILES.items():
        _nonblind_receipt_declaration(
            artifacts[key],
            expected_path=expected_path,
            label=key.replace("_", " "),
        )

    records = _nonblind_split_records(manifest)
    source_inputs, builder = _nonblind_source_and_builder_declarations(
        manifest
    )
    fixed_file_snapshots = _fixed_nonblind_file_snapshots_v7(
        root,
        label="nonblind primary build",
        manifest_snapshot=manifest_snapshot,
    )
    split_snapshots = {
        split: fixed_file_snapshots[split]
        for split in NONBLIND_SPLIT_FILES
    }
    artifact_snapshots = {
        role: fixed_file_snapshots[role]
        for role in STRICT_ARTIFACT_FILES
    }
    artifact_payloads, artifact_receipts = (
        _validate_nonblind_aux_artifacts_v7(
            manifest,
            snapshots=artifact_snapshots,
        )
    )
    seen_example_ids: set[str] = set()
    summaries: dict[str, Any] = {}
    for split in NONBLIND_SPLIT_FILES:
        summaries[split] = _scan_strict_visible_snapshot_v7(
            split_snapshots[split],
            split=split,
            expected=records[split],
            seen_example_ids=seen_example_ids,
        )
    _assert_nonblind_split_isolation_v7(summaries)
    semantic_binding = _nonblind_semantic_binding_v7(
        manifest,
        source_inputs=source_inputs,
        snapshot=artifact_snapshots["semantic_inventory_audit"],
    )
    preblind_binding = _nonblind_precommit_binding_v7(
        manifest,
        source_inputs=source_inputs,
        builder=builder,
        snapshot=artifact_snapshots["preblind_commitment"],
    )
    implementation = _strict_implementation_snapshots_v7()
    _validate_current_nonblind_builder_v7(
        builder,
        implementation=implementation,
    )
    second_build = _validate_second_nonblind_build_v7(
        Path(nonblind_second_build_dir),
        primary_root=root,
        primary_manifest=manifest,
        primary_source_inputs=source_inputs,
        primary_builder=builder,
        implementation=implementation,
    )
    actual_double_build, double_build_evidence = (
        _compare_nonblind_build_snapshots_v7(
            primary_root=root,
            primary_root_identity=primary_root_identity,
            primary_files=fixed_file_snapshots,
            second=second_build,
        )
    )
    compare_gate = _validate_compare_audit_gate_v7(
        Path(nonblind_audit_receipt),
        manifest=manifest,
        artifact_payloads=artifact_payloads,
        implementation=implementation,
        fixed_file_snapshots=fixed_file_snapshots,
        actual_double_build=actual_double_build,
    )
    shortcut_gates = {
        "train": _validate_shortcut_gate_v7(
            Path(train_shortcut_audit),
            split="train",
            split_snapshot=split_snapshots["train"],
            expected_count=NONBLIND_SPLIT_COUNTS["train"],
            implementation=implementation,
        ),
        "validation": _validate_shortcut_gate_v7(
            Path(validation_shortcut_audit),
            split="validation",
            split_snapshot=split_snapshots["validation"],
            expected_count=NONBLIND_SPLIT_COUNTS["validation"],
            implementation=implementation,
        ),
    }
    _recheck_strict_directory_identity_v7(
        root,
        expected=primary_root_identity,
        label="nonblind primary build directory",
    )
    _assert_exact_nonblind_inventory_v7(
        root,
        root_identity=primary_root_identity,
        label="nonblind primary build directory",
    )
    implementation_receipts = {
        role: {
            "path": snapshot.path.as_posix(),
            "bytes": snapshot.byte_count,
            "sha256": snapshot.sha256,
            "stable_identity": snapshot.identity_receipt(),
        }
        for role, snapshot in sorted(implementation.items())
    }
    inspected_core = {
        "manifest": {
            "sha256": manifest_snapshot.sha256,
            "bytes": manifest_snapshot.byte_count,
            "stable_identity": manifest_snapshot.identity_receipt(),
        },
        "builder_version": NONBLIND_BUILDER_VERSION,
        "fixed_split_integrity_snapshots": [
            {
                "path": summaries[split]["path"],
                "bytes": summaries[split]["bytes"],
                "sha256": summaries[split]["sha256"],
                "examples": summaries[split]["examples"],
                "stable_identity": summaries[split]["stable_identity"],
            }
            for split in NONBLIND_SPLIT_FILES
        ],
        "semantic_binding": semantic_binding,
        "preblind_binding": preblind_binding,
        "strict_artifact_receipts": artifact_receipts,
        "double_build_evidence": double_build_evidence,
        "strict_audit_gates": {
            "nonblind_compare": compare_gate,
            "shortcut": shortcut_gates,
        },
        "implementation_receipts": implementation_receipts,
    }
    return {
        "path": str(root),
        "contract": "STRICT_NONBLIND_V7",
        "manifest": {
            "path": NONBLIND_MANIFEST_NAME,
            "bytes": manifest_snapshot.byte_count,
            "sha256": manifest_snapshot.sha256,
            "stable_identity": manifest_snapshot.identity_receipt(),
            "schema": NONBLIND_MANIFEST_SCHEMA,
            "dataset_schema": DATASET_SCHEMA,
            "builder_version": NONBLIND_BUILDER_VERSION,
        },
        "splits": summaries,
        "semantic_binding": semantic_binding,
        "preblind_commitment": preblind_binding,
        "strict_artifact_receipts": artifact_receipts,
        "double_build_evidence": double_build_evidence,
        "strict_audit_gates": {
            "nonblind_compare": compare_gate,
            "shortcut": shortcut_gates,
        },
        "implementation_receipts": implementation_receipts,
        "training_data_access": {
            "opened_splits": list(READABLE_SPLITS),
            "integrity_only_splits": ["calibration"],
            "opened_nonblind_audit_artifacts": sorted(
                STRICT_ARTIFACT_FILES
            ),
            "primary_fixed_files_stably_opened": 10,
            "second_fixed_files_stably_opened": 10,
            "primary_exact_whitelist_enumerated": True,
            "second_exact_whitelist_enumerated": True,
            "non_whitelist_entries_observed": False,
            "second_build_bytes_compared_directly": True,
            "second_build_file_identities_compared_directly": True,
            "nonblind_compare_audit_verified": True,
            "train_shortcut_audit_verified": True,
            "validation_shortcut_audit_verified": True,
            "shortcut_reports_locally_recomputed": True,
            "shortcut_per_sample_bytes_locally_recomputed": True,
            "calibration_content_read": False,
            "calibration_content_hashed": False,
            "calibration_legacy_fields_mean_training_access_only": True,
            "calibration_integrity_snapshot_opened": True,
            "calibration_integrity_content_read": True,
            "calibration_integrity_content_parsed": True,
            "calibration_integrity_content_hashed": True,
            "calibration_content_loaded_for_training": False,
            "calibration_used_for_checkpoint_selection": False,
            "blind_materialized": False,
            "blind_discovered": False,
            "blind_path_constructed": False,
            "blind_filesystem_metadata_accessed": False,
            "blind_content_opened": False,
            "blind_content_read": False,
            "blind_content_hashed": False,
        },
        "inspected_input_sha256": _canonical_sha256(inspected_core),
    }


def _assert_exact_nonblind_inventory_v8(
    root: Path,
    *,
    root_identity: tuple[int, int],
    label: str,
) -> None:
    expected_names = frozenset(
        NONBLIND_V8_COMPARE_INVENTORY_FILES.values()
    )
    try:
        entries = list(os.scandir(root))
    except OSError as exc:
        raise QLoRAV6Error(
            f"{label} cannot be enumerated for the fixed whitelist"
        ) from exc
    observed_names = [entry.name for entry in entries]
    if (
        len(observed_names) != len(expected_names)
        or frozenset(observed_names) != expected_names
        or len({name.casefold() for name in observed_names})
        != len(observed_names)
    ):
        raise QLoRAV6Error(
            f"{label} must contain exactly the fixed twelve-file whitelist"
        )
    for entry in entries:
        try:
            metadata = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise QLoRAV6Error(
                f"{label} whitelist entry changed during enumeration"
            ) from exc
        if (
            entry.is_symlink()
            or stat.S_ISLNK(metadata.st_mode)
            or _is_reparse_point(metadata)
            or not stat.S_ISREG(metadata.st_mode)
        ):
            raise QLoRAV6Error(
                f"{label} whitelist entries must be regular non-reparse files"
            )
    _recheck_strict_directory_identity_v7(
        root,
        expected=root_identity,
        label=label,
    )
    try:
        names_after = frozenset(entry.name for entry in os.scandir(root))
    except OSError as exc:
        raise QLoRAV6Error(
            f"{label} changed during whitelist verification"
        ) from exc
    if names_after != expected_names:
        raise QLoRAV6Error(
            f"{label} changed during whitelist verification"
        )


def _fixed_nonblind_file_snapshots_v8(
    root: Path,
    *,
    label: str,
    manifest_snapshot: StableFileSnapshotV7 | None = None,
) -> dict[str, StableFileSnapshotV7]:
    snapshots: dict[str, StableFileSnapshotV7] = {}
    for role, filename in NONBLIND_V8_COMPARE_INVENTORY_FILES.items():
        if role == "manifest" and manifest_snapshot is not None:
            snapshots[role] = manifest_snapshot
            continue
        maximum_bytes = (
            _STRICT_MAX_JSONL_BYTES
            if role in NONBLIND_SPLIT_FILES
            else _STRICT_MAX_JSON_BYTES
        )
        snapshots[role] = _stable_snapshot_v7(
            root / filename,
            label=f"{label} {role}",
            maximum_bytes=maximum_bytes,
        )
    return snapshots


def _strict_implementation_snapshots_v8(
) -> dict[str, StableFileSnapshotV7]:
    paths = {
        "nonblind_v8_builder": (
            WORKSPACE_ROOT / "icmat_foundry" / "llm" / "nonblind_sft_v8.py"
        ),
        "nonblind_v7_builder": (
            WORKSPACE_ROOT / "icmat_foundry" / "llm" / "nonblind_sft_v7.py"
        ),
        "evidence_core": (
            WORKSPACE_ROOT / "icmat_foundry" / "llm" / "evidence_sft_v6.py"
        ),
        "semantic_core": (
            WORKSPACE_ROOT
            / "icmat_foundry"
            / "llm"
            / "semantic_queries_v7.py"
        ),
        "nonblind_v8_auditor": (
            WORKSPACE_ROOT
            / "icmat_foundry"
            / "llm"
            / "nonblind_sft_audit_v8.py"
        ),
        "nonblind_v8_audit_cli": (
            WORKSPACE_ROOT / "tools" / "audit_icmat_nonblind_sft_v8.py"
        ),
        "shortcut_v8_module": (
            WORKSPACE_ROOT
            / "icmat_foundry"
            / "llm"
            / "shortcut_audit_v8.py"
        ),
        "shortcut_v8_cli": (
            WORKSPACE_ROOT
            / "tools"
            / "audit_icmat_semantic_shortcuts_v8.py"
        ),
        "unique_support_v8_module": (
            WORKSPACE_ROOT
            / "icmat_foundry"
            / "llm"
            / "unique_support_audit_v8.py"
        ),
        "unique_support_v8_cli": (
            WORKSPACE_ROOT
            / "tools"
            / "audit_icmat_unique_support_v8.py"
        ),
        "trainer_module": Path(__file__),
        "trainer_cli": (
            WORKSPACE_ROOT / "tools" / "train_icmat_qlora_full_v6.py"
        ),
        "canary_acceptance_v8": (
            WORKSPACE_ROOT
            / "icmat_foundry"
            / "llm"
            / "canary_acceptance_v8.py"
        ),
        "pointer_evaluator_v8": (
            WORKSPACE_ROOT
            / "icmat_foundry"
            / "llm"
            / "pointer_checkpoint_eval_v8.py"
        ),
        "pointer_runner_v8": (
            WORKSPACE_ROOT
            / "tools"
            / "evaluate_icmat_pointer_checkpoints_v8.py"
        ),
        "checkpoint_core_v6": (
            WORKSPACE_ROOT
            / "icmat_foundry"
            / "llm"
            / "pointer_checkpoint_eval_v6.py"
        ),
        "pointer_numeric_evaluator_v6": (
            WORKSPACE_ROOT
            / "icmat_foundry"
            / "llm"
            / "pointer_hf_eval_v6.py"
        ),
        "pointer_compiler_v6": (
            WORKSPACE_ROOT
            / "icmat_foundry"
            / "llm"
            / "evidence_pointer_v6.py"
        ),
        "selection_policy_v6": (
            WORKSPACE_ROOT
            / "icmat_foundry"
            / "llm"
            / "selection_policy_v6.py"
        ),
        "canary_numeric_core_v6": (
            WORKSPACE_ROOT
            / "icmat_foundry"
            / "llm"
            / "canary_acceptance_v6.py"
        ),
        "v8c2_preregistration": V8C2_PREREGISTRATION_PATH,
    }
    return {
        role: _stable_snapshot_v7(
            path,
            label=f"{role} implementation",
            maximum_bytes=_STRICT_MAX_JSON_BYTES,
        )
        for role, path in paths.items()
    }


def _expected_nli_provenance_v8() -> dict[str, Any]:
    return {
        "backend": "local_transformers_nli",
        "repo_id": semantic_contract_v7.PINNED_NLI_REPO_ID,
        "revision": semantic_contract_v7.PINNED_NLI_REVISION,
        "license_name": semantic_contract_v7.PINNED_NLI_LICENSE,
        "model_tree_sha256": (
            semantic_contract_v7.PINNED_NLI_MODEL_TREE_SHA256
        ),
        "model_receipt_sha256": (
            semantic_contract_v7.PINNED_NLI_RECEIPT_SHA256
        ),
        "model_file_count": semantic_contract_v7.PINNED_NLI_FILE_COUNT,
        "model_total_bytes": semantic_contract_v7.PINNED_NLI_TOTAL_BYTES,
        "local_files_only": True,
        "device": "cpu",
        "quality_claim_allowed": True,
    }


def _nonblind_split_records_v8(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    splits = manifest.get("splits")
    if (
        not isinstance(splits, Mapping)
        or set(splits) != set(NONBLIND_SPLIT_FILES)
    ):
        raise QLoRAV6Error(
            "nonblind-v8 manifest must declare exactly three nonblind splits"
        )
    records: dict[str, dict[str, Any]] = {}
    for split, expected_path in NONBLIND_SPLIT_FILES.items():
        value = splits.get(split)
        if (
            not isinstance(value, Mapping)
            or set(value) != {"path", "sha256", "bytes", "count"}
            or value.get("path") != expected_path
            or not _valid_sha256(value.get("sha256"))
            or isinstance(value.get("bytes"), bool)
            or not isinstance(value.get("bytes"), int)
            or value.get("bytes", 0) < 1
            or isinstance(value.get("count"), bool)
            or not isinstance(value.get("count"), int)
            or value.get("count") != NONBLIND_SPLIT_COUNTS[split]
        ):
            raise QLoRAV6Error(
                f"nonblind-v8 {split} split receipt mismatch"
            )
        records[split] = {
            "path": expected_path,
            "bytes": int(value["bytes"]),
            "sha256": str(value["sha256"]),
            "examples": int(value["count"]),
        }
    return records


def _validate_receipt_declaration_v8(
    value: Any,
    *,
    expected_path: str,
    label: str,
) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "sha256", "bytes"}
        or value.get("path") != expected_path
        or not _valid_sha256(value.get("sha256"))
        or isinstance(value.get("bytes"), bool)
        or not isinstance(value.get("bytes"), int)
        or value.get("bytes", 0) < 1
    ):
        raise QLoRAV6Error(f"nonblind-v8 {label} receipt mismatch")
    return {
        "path": expected_path,
        "bytes": int(value["bytes"]),
        "sha256": str(value["sha256"]),
    }


def _verify_receipt_snapshot_v8(
    value: Any,
    snapshot: StableFileSnapshotV7,
    *,
    expected_path: str,
    label: str,
) -> dict[str, Any]:
    receipt = _validate_receipt_declaration_v8(
        value,
        expected_path=expected_path,
        label=label,
    )
    if (
        receipt["bytes"] != snapshot.byte_count
        or receipt["sha256"] != snapshot.sha256
    ):
        raise QLoRAV6Error(
            f"nonblind-v8 {label} receipt does not match stable bytes"
        )
    return {
        **receipt,
        "stable_identity": snapshot.identity_receipt(),
        "content_opened": True,
        "content_read": True,
        "content_hashed": True,
        "stable_snapshot": True,
    }


def _validate_source_and_builder_v8(
    manifest: Mapping[str, Any],
    *,
    implementation: Mapping[str, StableFileSnapshotV7],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_inputs = manifest.get("source_inputs")
    expected_source_roles = {
        "licensed_chunks",
        "rag_manifest",
        "semantic_inventory",
        "semantic_records",
        "semantic_requests",
        "semantic_request_manifest",
    }
    if (
        not isinstance(source_inputs, Mapping)
        or set(source_inputs) != expected_source_roles
    ):
        raise QLoRAV6Error("nonblind-v8 source input declarations mismatch")
    normalized_sources: dict[str, Any] = {}
    for role in sorted(expected_source_roles):
        value = source_inputs[role]
        if (
            not isinstance(value, Mapping)
            or set(value) != {"path", "sha256"}
            or not isinstance(value.get("path"), str)
            or not value.get("path")
            or not _valid_sha256(value.get("sha256"))
        ):
            raise QLoRAV6Error(
                f"nonblind-v8 {role} source declaration mismatch"
            )
        path_tokens = {
            token
            for component in Path(str(value["path"])).parts
            for token in re.findall(r"[a-z0-9]+", component.casefold())
        }
        if path_tokens & {"blind", "reserved", "sealed", "calibration"}:
            raise QLoRAV6Error(
                f"nonblind-v8 {role} source path crosses protected scope"
            )
        normalized_sources[role] = {
            "path": str(value["path"]),
            "sha256": str(value["sha256"]),
        }

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
        or builder.get("split_algorithm_version")
        != NONBLIND_V8_SPLIT_ALGORITHM_VERSION
        or builder.get("repair_policy_version")
        != NONBLIND_V8_REPAIR_POLICY_VERSION
        or not isinstance(builder.get("seed"), str)
        or not builder.get("seed")
    ):
        raise QLoRAV6Error("nonblind-v8 builder declaration mismatch")
    code = builder.get("code")
    implementation_roles = {
        "nonblind_v8_module": "nonblind_v8_builder",
        "nonblind_v7_module": "nonblind_v7_builder",
        "evidence_core": "evidence_core",
        "semantic_core": "semantic_core",
    }
    if not isinstance(code, Mapping) or set(code) != set(
        implementation_roles
    ):
        raise QLoRAV6Error("nonblind-v8 builder code declaration mismatch")
    normalized_code: dict[str, Any] = {}
    for role, implementation_role in implementation_roles.items():
        value = code[role]
        current = implementation[implementation_role]
        if (
            not isinstance(value, Mapping)
            or set(value) != {"path", "sha256"}
            or not isinstance(value.get("path"), str)
            or not _same_regular_path_v7(
                value.get("path"),
                current.path,
                label=f"nonblind-v8 {role} code",
            )
            or value.get("sha256") != current.sha256
        ):
            raise QLoRAV6Error(
                f"nonblind-v8 {role} implementation binding mismatch"
            )
        normalized_code[role] = {
            "path": str(value["path"]),
            "sha256": str(value["sha256"]),
        }
    return normalized_sources, {
        "code": normalized_code,
        "split_algorithm_version": NONBLIND_V8_SPLIT_ALGORITHM_VERSION,
        "repair_policy_version": NONBLIND_V8_REPAIR_POLICY_VERSION,
        "seed": str(builder["seed"]),
    }


def _validate_manifest_contract_v8(
    manifest: Mapping[str, Any],
    *,
    implementation: Mapping[str, StableFileSnapshotV7],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    expected_top_level = {
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
        "claims",
        "sealed_blind_access",
    }
    if set(manifest) != expected_top_level:
        raise QLoRAV6Error("nonblind-v8 manifest keys mismatch")
    if (
        manifest.get("schema") != NONBLIND_V8_MANIFEST_SCHEMA
        or manifest.get("dataset_schema") != DATASET_SCHEMA
        or manifest.get("builder_version") != NONBLIND_V8_BUILDER_VERSION
        or manifest.get("core_builder_version")
        != SEMANTIC_BUILDER_VERSION
        or manifest.get("status")
        != "NONBLIND_V8_BUILT_NLI_UNIQUE_SUPPORT_PREBLIND_COMMITTED"
    ):
        raise QLoRAV6Error("nonblind-v8 manifest identity mismatch")
    if (
        manifest.get("ground_truth_policy")
        != (
            "deterministic pointer labels from licensed evidence; "
            "the fixed local NLI model audits uniqueness but never "
            "creates ground truth"
        )
        or manifest.get("selection_policy")
        != "researcher_explicit_domain_and_task"
        or manifest.get("source_isolation_unit") != "DOI/source_family"
    ):
        raise QLoRAV6Error("nonblind-v8 manifest policy mismatch")
    if manifest.get("counts") != {
        "examples": NONBLIND_TOTAL_EXAMPLES,
        "answers": NONBLIND_TOTAL_EXAMPLES // 2,
        "families": NONBLIND_FAMILY_COUNT,
        "examples_per_family": NONBLIND_EXAMPLES_PER_FAMILY,
        "splits": NONBLIND_SPLIT_COUNTS,
    }:
        raise QLoRAV6Error("nonblind-v8 fixed counts mismatch")
    if manifest.get("pointer_contract") != {
        "field_order": list(POINTER_FIELDS),
        "answer_span_pattern": "E#.S#",
        "refusal_span_id": None,
    }:
        raise QLoRAV6Error("nonblind-v8 pointer contract mismatch")
    if manifest.get("compiler_input_contract") != {
        "compiler_version": evidence_contract.COMPILER_VERSION,
        "prompt_schema": evidence_contract.COMPILER_PROMPT_SCHEMA,
        "compiler_prompt_keys": sorted(
            evidence_contract.COMPILER_PROMPT_FIELDS
        ),
        "compiler_evidence_keys": sorted(
            evidence_contract.COMPILER_EVIDENCE_FIELDS
        ),
        "compiler_sentence_keys": sorted(
            evidence_contract.COMPILER_SENTENCE_FIELDS
        ),
        "target_free": True,
        "user_text_reverse_parsing_required": False,
    }:
        raise QLoRAV6Error("nonblind-v8 compiler input contract mismatch")
    if manifest.get("external_answer_contract") != {
        "schema": evidence_contract.EXTERNAL_ANSWER_SCHEMA,
        "field_order": list(evidence_contract.EXTERNAL_ANSWER_FIELDS),
        "generated_by": "later_deterministic_evidence_compiler",
        "implemented_by_this_builder": False,
    }:
        raise QLoRAV6Error(
            "nonblind-v8 external answer contract mismatch"
        )
    if manifest.get("training_boundary") != {
        "allowed_splits": list(READABLE_SPLITS),
        "calibration_content_for_training": False,
    }:
        raise QLoRAV6Error("nonblind-v8 training boundary mismatch")
    expected_claims = {
        "nonblind_only": True,
        "training_authorized_splits": list(READABLE_SPLITS),
        "calibration_for_training": False,
        "manual_jsonl_editing": False,
        "target_passage_modified": False,
        "production_connected": False,
        "x5_deployed": False,
    }
    if manifest.get("claims") != expected_claims:
        raise QLoRAV6Error("nonblind-v8 claims boundary mismatch")
    if manifest.get("sealed_blind_access") != {
        "read": False,
        "hashed": False,
        "path_discovered": False,
    }:
        raise QLoRAV6Error("nonblind-v8 sealed blind boundary mismatch")

    nli_contract = manifest.get("nli_unique_support")
    if (
        not isinstance(nli_contract, Mapping)
        or set(nli_contract)
        != {
            "provenance",
            "score_orientation",
            "non_target_scope",
            "target_passage_neighbor_policy",
            "thresholds",
            "answer_count",
            "repair_count",
            "target_passage_modified",
        }
        or nli_contract.get("provenance")
        != _expected_nli_provenance_v8()
        or nli_contract.get("score_orientation")
        != {
            "premise": "evidence_sentence",
            "hypothesis": "requested_claim",
        }
        or nli_contract.get("non_target_scope")
        != (
            "every span other than target_span_id across both "
            "evidence passages"
        )
        or nli_contract.get("target_passage_neighbor_policy")
        != "fail_closed_without_rewriting_or_shortening_target_passage"
        or nli_contract.get("thresholds")
        != {
            "target_entailment_min": 0.9,
            "distractor_entailment_max": 0.1,
        }
        or nli_contract.get("answer_count") != NONBLIND_TOTAL_EXAMPLES // 2
        or isinstance(nli_contract.get("repair_count"), bool)
        or not isinstance(nli_contract.get("repair_count"), int)
        or nli_contract.get("repair_count", -1) < 0
        or nli_contract.get("target_passage_modified") is not False
    ):
        raise QLoRAV6Error(
            "nonblind-v8 integrated NLI declaration mismatch"
        )

    artifacts = manifest.get("artifacts")
    if (
        not isinstance(artifacts, Mapping)
        or set(artifacts) != set(STRICT_V8_ARTIFACT_FILES)
    ):
        raise QLoRAV6Error("nonblind-v8 artifact receipts mismatch")
    for role, expected_path in STRICT_V8_ARTIFACT_FILES.items():
        _validate_receipt_declaration_v8(
            artifacts[role],
            expected_path=expected_path,
            label=role.replace("_", " "),
        )
    records = _nonblind_split_records_v8(manifest)
    source_inputs, builder = _validate_source_and_builder_v8(
        manifest,
        implementation=implementation,
    )
    expected_input_commitment = _canonical_sha256(
        {
            "files": {
                role: value["sha256"]
                for role, value in source_inputs.items()
            },
            "nli_model_tree_sha256": (
                semantic_contract_v7.PINNED_NLI_MODEL_TREE_SHA256
            ),
            "seed_sha256": hashlib.sha256(
                builder["seed"].encode("utf-8")
            ).hexdigest(),
        }
    )
    if manifest.get("input_commitment_sha256") != expected_input_commitment:
        raise QLoRAV6Error(
            "nonblind-v8 input commitment binding mismatch"
        )
    expected_output_content = _canonical_sha256(
        {
            "splits": manifest["splits"],
            "artifacts": manifest["artifacts"],
        }
    )
    if manifest.get("output_content_sha256") != expected_output_content:
        raise QLoRAV6Error(
            "nonblind-v8 output content binding mismatch"
        )
    return records, source_inputs, builder


def _lexical_snapshot_v8(
    snapshot: StableFileSnapshotV7,
) -> shortcut_contract.FileSnapshot:
    return shortcut_contract.FileSnapshot(
        path=snapshot.path,
        payload=snapshot.payload,
        sha256=snapshot.sha256,
        size_bytes=snapshot.byte_count,
        identity=(
            snapshot.identity[0],
            snapshot.identity[1],
            snapshot.identity[2],
            snapshot.identity[3],
        ),
    )


def _lexical_rows_v8(
    snapshot: StableFileSnapshotV7,
    *,
    split: str,
) -> list[dict[str, Any]]:
    try:
        return shortcut_contract_v8._load_split(
            _lexical_snapshot_v8(snapshot),
            expected_split=split,
        )
    except (
        shortcut_contract.ShortcutAuditV7Error,
        shortcut_contract_v8.ShortcutAuditV8Error,
    ) as exc:
        raise QLoRAV6Error(
            f"nonblind-v8 {split} lexical parsing failed: {exc}"
        ) from exc


def _validate_primary_with_independent_v8_parser(
    *,
    root: Path,
    files: Mapping[str, StableFileSnapshotV7],
    manifest: Mapping[str, Any],
    source_inputs: Mapping[str, Any],
    nli_model_dir: Path,
) -> Any:
    from icmat_foundry.llm import nonblind_sft_audit_v8

    try:
        state = nonblind_sft_audit_v8._load_and_validate_dataset(
            root,
            label="QLoRA strict nonblind-v8 primary",
            licensed_chunks=Path(
                str(source_inputs["licensed_chunks"]["path"])
            ),
            rag_manifest=Path(
                str(source_inputs["rag_manifest"]["path"])
            ),
            semantic_inventory=Path(
                str(source_inputs["semantic_inventory"]["path"])
            ),
            nli_model_dir=nli_model_dir,
        )
    except (
        OSError,
        RuntimeError,
        ValueError,
        nonblind_sft_audit_v8.NonblindSFTAuditV8Error,
    ) as exc:
        raise QLoRAV6Error(
            f"independent nonblind-v8 parser rejected the dataset: {exc}"
        ) from exc
    if state.manifest != manifest or set(state.files) != set(files):
        raise QLoRAV6Error(
            "independent nonblind-v8 parser snapshot mismatch"
        )
    for role, snapshot in files.items():
        observed = state.files[role]
        if (
            observed.payload != snapshot.payload
            or observed.sha256 != snapshot.sha256
            or observed.bytes != snapshot.byte_count
            or tuple(observed.identity) != tuple(snapshot.identity[:4])
        ):
            raise QLoRAV6Error(
                f"independent nonblind-v8 parser {role} snapshot mismatch"
            )
    return state


def _validate_second_nonblind_build_v8(
    path: Path,
    *,
    primary_root: Path,
    primary_files: Mapping[str, StableFileSnapshotV7],
    primary_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    root, root_identity = _strict_directory_identity_v7(
        path,
        label="nonblind-v8 second build directory",
    )
    if root == primary_root or os.path.samefile(root, primary_root):
        raise QLoRAV6Error(
            "nonblind-v8 second build directory must be distinct"
        )
    _assert_exact_nonblind_inventory_v8(
        root,
        root_identity=root_identity,
        label="nonblind-v8 second build directory",
    )
    files = _fixed_nonblind_file_snapshots_v8(
        root,
        label="nonblind-v8 second build",
    )
    manifest = _strict_json_object_v7(
        files["manifest"],
        label=NONBLIND_V8_MANIFEST_NAME,
    )
    if manifest != primary_manifest:
        raise QLoRAV6Error(
            "nonblind-v8 second build manifest differs from primary"
        )
    comparisons: list[dict[str, Any]] = []
    for role, filename in NONBLIND_V8_COMPARE_INVENTORY_FILES.items():
        left = primary_files[role]
        right = files[role]
        try:
            same_file = os.path.samefile(left.path, right.path)
        except OSError as exc:
            raise QLoRAV6Error(
                f"nonblind-v8 {filename} identity cannot be compared"
            ) from exc
        if (
            same_file
            or left.identity[:2] == right.identity[:2]
            or left.payload != right.payload
            or left.sha256 != right.sha256
            or left.byte_count != right.byte_count
        ):
            raise QLoRAV6Error(
                f"nonblind-v8 {filename} is not an independent "
                "byte-identical build"
            )
        comparisons.append(
            {
                "role": role,
                "path": filename,
                "sha256": left.sha256,
                "bytes": left.byte_count,
                "primary_stable_identity": left.identity_receipt(),
                "secondary_stable_identity": right.identity_receipt(),
                "byte_identical": True,
                "independent_file_identity": True,
            }
        )
    _recheck_strict_directory_identity_v7(
        root,
        expected=root_identity,
        label="nonblind-v8 second build directory",
    )
    _assert_exact_nonblind_inventory_v8(
        root,
        root_identity=root_identity,
        label="nonblind-v8 second build directory",
    )
    return {
        "root": root,
        "root_identity": root_identity,
        "files": files,
        "manifest": manifest,
        "comparisons": comparisons,
    }


def _validate_compare_audit_gate_v8(
    path: Path,
    *,
    primary_root: Path,
    secondary_root: Path,
    files: Mapping[str, StableFileSnapshotV7],
    manifest: Mapping[str, Any],
    authority_state: Any,
    implementation: Mapping[str, StableFileSnapshotV7],
) -> dict[str, Any]:
    from icmat_foundry.llm import nonblind_sft_audit_v8

    snapshot = _stable_snapshot_v7(
        path,
        label="nonblind-v8 independent compare receipt",
        maximum_bytes=_STRICT_MAX_JSON_BYTES,
    )
    receipt = _strict_json_object_v7(
        snapshot,
        label="nonblind-v8 independent compare receipt",
    )
    expected_keys = {
        "schema",
        "audit_version",
        "status",
        "audit_passed",
        "mode",
        "created_at",
        "primary",
        "secondary",
        "authorities",
        "nli_model",
        "files",
        "file_count",
        "implementation",
        "reserved_asset_accessed",
        "production_connected",
        "x5_deployed",
        "byte_identical",
        "independent_file_identity",
        "canonical_digest_sha256",
        "receipt_sha256",
    }
    if set(receipt) != expected_keys:
        raise QLoRAV6Error(
            "nonblind-v8 independent compare receipt keys mismatch"
        )
    receipt_sha = receipt.get("receipt_sha256")
    receipt_without_sha = {
        key: value
        for key, value in receipt.items()
        if key != "receipt_sha256"
    }
    canonical_digest = receipt.get("canonical_digest_sha256")
    body = {
        key: value
        for key, value in receipt_without_sha.items()
        if key != "canonical_digest_sha256"
    }
    if (
        receipt.get("schema") != nonblind_sft_audit_v8.AUDIT_SCHEMA
        or receipt.get("audit_version")
        != nonblind_sft_audit_v8.AUDIT_VERSION
        or receipt.get("status") != NONBLIND_V8_COMPARE_STATUS
        or receipt.get("audit_passed") is not True
        or receipt.get("mode") != "compare"
        or receipt.get("byte_identical") is not True
        or receipt.get("independent_file_identity") is not True
        or receipt.get("reserved_asset_accessed") is not False
        or receipt.get("production_connected") is not False
        or receipt.get("x5_deployed") is not False
        or canonical_digest != _canonical_sha256(body)
        or receipt_sha != _canonical_sha256(receipt_without_sha)
    ):
        raise QLoRAV6Error(
            "nonblind-v8 independent compare receipt did not grant PASS"
        )
    _require_timestamp_v7(
        receipt.get("created_at"),
        label="nonblind-v8 independent compare timestamp",
    )
    primary = receipt.get("primary")
    secondary = receipt.get("secondary")
    if (
        not isinstance(primary, Mapping)
        or set(primary)
        != {"root", "manifest_sha256", "output_content_sha256"}
        or not _same_regular_path_v7(
            primary.get("root"),
            primary_root,
            label="nonblind-v8 compare primary root",
        )
        or primary.get("manifest_sha256") != files["manifest"].sha256
        or primary.get("output_content_sha256")
        != manifest.get("output_content_sha256")
        or not isinstance(secondary, Mapping)
        or set(secondary)
        != {"root", "manifest_sha256", "output_content_sha256"}
        or not _same_regular_path_v7(
            secondary.get("root"),
            secondary_root,
            label="nonblind-v8 compare secondary root",
        )
        or secondary.get("manifest_sha256") != files["manifest"].sha256
        or secondary.get("output_content_sha256")
        != manifest.get("output_content_sha256")
    ):
        raise QLoRAV6Error(
            "nonblind-v8 independent compare dataset binding mismatch"
        )
    expected_files = {
        filename: {
            "sha256": files[role].sha256,
            "bytes": files[role].byte_count,
        }
        for role, filename in NONBLIND_V8_COMPARE_INVENTORY_FILES.items()
    }
    if (
        receipt.get("files") != expected_files
        or receipt.get("file_count") != 12
    ):
        raise QLoRAV6Error(
            "nonblind-v8 independent compare fixed files mismatch"
        )
    expected_authorities = {
        role: {
            "path": authority.path.as_posix(),
            "sha256": authority.sha256,
        }
        for role, authority in authority_state.authorities.files.items()
    }
    if (
        receipt.get("authorities") != expected_authorities
        or receipt.get("nli_model") != authority_state.authorities.nli
    ):
        raise QLoRAV6Error(
            "nonblind-v8 independent compare authority binding mismatch"
        )
    receipt_implementation = receipt.get("implementation")
    if (
        not isinstance(receipt_implementation, Mapping)
        or set(receipt_implementation) != {"auditor", "runner"}
    ):
        raise QLoRAV6Error(
            "nonblind-v8 independent compare implementation mismatch"
        )
    for key, implementation_role in (
        ("auditor", "nonblind_v8_auditor"),
        ("runner", "nonblind_v8_audit_cli"),
    ):
        current = implementation[implementation_role]
        value = receipt_implementation[key]
        if (
            not isinstance(value, Mapping)
            or set(value) != {"path", "sha256"}
            or not _same_regular_path_v7(
                value.get("path"),
                current.path,
                label=f"nonblind-v8 compare {key}",
            )
            or value.get("sha256") != current.sha256
        ):
            raise QLoRAV6Error(
                f"nonblind-v8 compare {key} implementation mismatch"
            )
    return {
        "path": str(snapshot.path),
        "bytes": snapshot.byte_count,
        "sha256": snapshot.sha256,
        "stable_identity": snapshot.identity_receipt(),
        "schema": nonblind_sft_audit_v8.AUDIT_SCHEMA,
        "audit_version": nonblind_sft_audit_v8.AUDIT_VERSION,
        "status": NONBLIND_V8_COMPARE_STATUS,
        "audit_passed": True,
        "fixed_files_verified": 12,
        "direct_byte_comparison_is_authoritative": True,
    }


def _strict_explicit_audit_inventory_v8(
    report_path: Path,
    *,
    expected_names: frozenset[str],
    label: str,
) -> tuple[Path, tuple[int, int]]:
    report = _absolute_lexical_v7(report_path)
    root, identity = _strict_directory_identity_v7(
        report.parent,
        label=f"{label} directory",
    )
    try:
        observed = [entry.name for entry in os.scandir(root)]
    except OSError as exc:
        raise QLoRAV6Error(f"{label} directory cannot be enumerated") from exc
    if (
        len(observed) != len(expected_names)
        or frozenset(observed) != expected_names
        or len({name.casefold() for name in observed}) != len(observed)
    ):
        raise QLoRAV6Error(f"{label} fixed artifact inventory mismatch")
    for name in observed:
        _stable_snapshot_v7(
            root / name,
            label=f"{label} {name}",
            maximum_bytes=_STRICT_MAX_JSON_BYTES,
        )
    return root, identity


def _v8_recomputed_json_equal(observed: Any, expected: Any) -> bool:
    if isinstance(observed, bool) or isinstance(expected, bool):
        return type(observed) is type(expected) and observed == expected
    if isinstance(observed, float) or isinstance(expected, float):
        return (
            isinstance(observed, float)
            and isinstance(expected, float)
            and math.isfinite(observed)
            and math.isfinite(expected)
            and math.isclose(observed, expected, rel_tol=0.0, abs_tol=1.0e-12)
        )
    if type(observed) is not type(expected):
        return False
    if isinstance(observed, Mapping):
        return set(observed) == set(expected) and all(
            _v8_recomputed_json_equal(observed[key], expected[key])
            for key in observed
        )
    if isinstance(observed, list):
        return len(observed) == len(expected) and all(
            _v8_recomputed_json_equal(left, right)
            for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


def _validate_scoped_lexical_gate_v8(
    path: Path,
    *,
    train_snapshot: StableFileSnapshotV7,
    validation_snapshot: StableFileSnapshotV7,
    implementation: Mapping[str, StableFileSnapshotV7],
) -> tuple[dict[str, Any], list[StableFileSnapshotV7]]:
    if path.name != shortcut_contract_v8.REPORT_NAME:
        raise QLoRAV6Error(
            "nonblind-v8 scoped lexical audit filename mismatch"
        )
    root, identity = _strict_explicit_audit_inventory_v8(
        path,
        expected_names=frozenset(shortcut_contract_v8.OUTPUT_NAMES),
        label="nonblind-v8 scoped lexical audit",
    )
    report_snapshot = _stable_snapshot_v7(
        root / shortcut_contract_v8.REPORT_NAME,
        label="nonblind-v8 scoped lexical report",
        maximum_bytes=_STRICT_MAX_JSON_BYTES,
    )
    train_sample = _stable_snapshot_v7(
        root / shortcut_contract_v8.TRAIN_SAMPLE_NAME,
        label="nonblind-v8 scoped lexical train samples",
        maximum_bytes=_STRICT_MAX_JSON_BYTES,
    )
    validation_sample = _stable_snapshot_v7(
        root / shortcut_contract_v8.VALIDATION_SAMPLE_NAME,
        label="nonblind-v8 scoped lexical validation samples",
        maximum_bytes=_STRICT_MAX_JSON_BYTES,
    )
    train_rows = _lexical_rows_v8(train_snapshot, split="train")
    validation_rows = _lexical_rows_v8(
        validation_snapshot,
        split="validation",
    )
    try:
        samples, analysis = shortcut_contract_v8.analyze_train_validation(
            train_rows,
            validation_rows,
        )
    except shortcut_contract_v8.ShortcutAuditV8Error as exc:
        raise QLoRAV6Error(
            f"nonblind-v8 scoped lexical recomputation failed: {exc}"
        ) from exc
    module = implementation["shortcut_v8_module"]
    runner = implementation["shortcut_v8_cli"]
    observed_report = _strict_json_object_v7(
        report_snapshot,
        label="nonblind-v8 scoped lexical report",
    )
    expected_artifacts = {
        "train_per_sample": {
            "path": shortcut_contract_v8.TRAIN_SAMPLE_NAME,
            "bytes": train_sample.byte_count,
            "sha256": train_sample.sha256,
        },
        "validation_per_sample": {
            "path": shortcut_contract_v8.VALIDATION_SAMPLE_NAME,
            "bytes": validation_sample.byte_count,
            "sha256": validation_sample.sha256,
        },
    }
    report_core = {
        "schema": shortcut_contract_v8.AUDIT_SCHEMA,
        "audit_version": shortcut_contract_v8.AUDIT_VERSION,
        **analysis,
        "scope": {
            "opened_splits": ["train", "validation"],
            "calibration_opened": False,
            "blind_discovered": False,
            "blind_opened": False,
            "blind_hashed": False,
        },
        "inputs": {
            "train": {
                "path": train_snapshot.path.as_posix(),
                "bytes": train_snapshot.byte_count,
                "sha256": train_snapshot.sha256,
                "examples": len(train_rows),
            },
            "validation": {
                "path": validation_snapshot.path.as_posix(),
                "bytes": validation_snapshot.byte_count,
                "sha256": validation_snapshot.sha256,
                "examples": len(validation_rows),
            },
        },
        "implementation": {
            "module": {
                "path": module.path.as_posix(),
                "bytes": module.byte_count,
                "sha256": module.sha256,
            },
            "runner": {
                "path": runner.path.as_posix(),
                "bytes": runner.byte_count,
                "sha256": runner.sha256,
            },
        },
        "artifacts": expected_artifacts,
    }
    observed_core = {
        key: value
        for key, value in observed_report.items()
        if key not in {"canonical_digest_sha256", "audit_id"}
    }
    observed_digest = observed_report.get("canonical_digest_sha256")
    observed_audit_id = observed_report.get("audit_id")
    observed_train_rows = _strict_jsonl_rows_v7(
        train_sample,
        label="nonblind-v8 scoped lexical train samples",
    )
    observed_validation_rows = _strict_jsonl_rows_v7(
        validation_sample,
        label="nonblind-v8 scoped lexical validation samples",
    )
    if (
        set(observed_report)
        != set(report_core) | {"canonical_digest_sha256", "audit_id"}
        or observed_report.get("artifacts") != expected_artifacts
        or not _v8_recomputed_json_equal(observed_core, report_core)
        or not _v8_recomputed_json_equal(
            observed_train_rows,
            samples["train"],
        )
        or not _v8_recomputed_json_equal(
            observed_validation_rows,
            samples["validation"],
        )
        or not _valid_sha256(observed_digest)
        or observed_digest != _canonical_sha256(observed_core)
        or observed_audit_id
        != f"icm-scoped-lexical-v8:{observed_digest}"
        or observed_report.get("status")
        != shortcut_contract_v8.PASS_STATUS
        or not all(
            bool(gate["passed"])
            for gate in observed_report.get("gates", ())
        )
    ):
        raise QLoRAV6Error(
            "nonblind-v8 scoped lexical audit does not match "
            "local recomputation"
        )
    _recheck_strict_directory_identity_v7(
        root,
        expected=identity,
        label="nonblind-v8 scoped lexical audit directory",
    )
    return (
        {
            "path": str(report_snapshot.path),
            "bytes": report_snapshot.byte_count,
            "sha256": report_snapshot.sha256,
            "stable_identity": report_snapshot.identity_receipt(),
            "schema": shortcut_contract_v8.AUDIT_SCHEMA,
            "audit_version": shortcut_contract_v8.AUDIT_VERSION,
            "status": shortcut_contract_v8.PASS_STATUS,
            "audit_id": observed_audit_id,
            "full_report_locally_recomputed": True,
            "train_per_sample_locally_recomputed": True,
            "validation_per_sample_locally_recomputed": True,
        },
        [report_snapshot, train_sample, validation_sample],
    )


def _validate_unique_support_gate_v8(
    path: Path,
    *,
    split: str,
    split_snapshot: StableFileSnapshotV7,
    auditor: Any,
    provenance: Mapping[str, Any],
    implementation: Mapping[str, StableFileSnapshotV7],
) -> tuple[dict[str, Any], list[StableFileSnapshotV7]]:
    if split not in READABLE_SPLITS:
        raise QLoRAV6Error("nonblind-v8 unique-support split mismatch")
    if path.name != unique_support_contract_v8.AUDIT_FILENAME:
        raise QLoRAV6Error(
            f"nonblind-v8 {split} unique-support audit filename mismatch"
        )
    expected_names = frozenset(
        {
            unique_support_contract_v8.AUDIT_FILENAME,
            unique_support_contract_v8.SAMPLE_FILENAME,
            unique_support_contract_v8.SUMMARY_FILENAME,
        }
    )
    root, identity = _strict_explicit_audit_inventory_v8(
        path,
        expected_names=expected_names,
        label=f"nonblind-v8 {split} unique-support audit",
    )
    report_snapshot = _stable_snapshot_v7(
        root / unique_support_contract_v8.AUDIT_FILENAME,
        label=f"nonblind-v8 {split} unique-support report",
        maximum_bytes=_STRICT_MAX_JSON_BYTES,
    )
    sample_snapshot = _stable_snapshot_v7(
        root / unique_support_contract_v8.SAMPLE_FILENAME,
        label=f"nonblind-v8 {split} unique-support samples",
        maximum_bytes=_STRICT_MAX_JSON_BYTES,
    )
    summary_snapshot = _stable_snapshot_v7(
        root / unique_support_contract_v8.SUMMARY_FILENAME,
        label=f"nonblind-v8 {split} unique-support summary",
        maximum_bytes=_STRICT_MAX_JSON_BYTES,
    )
    rows = _lexical_rows_v8(split_snapshot, split=split)
    try:
        samples, analysis = (
            unique_support_contract_v8.analyze_unique_support_rows(
                rows,
                auditor=auditor,
            )
        )
    except unique_support_contract_v8.UniqueSupportAuditV8Error as exc:
        raise QLoRAV6Error(
            f"nonblind-v8 {split} unique-support recomputation failed: {exc}"
        ) from exc
    samples_payload = unique_support_contract_v8._jsonl_bytes(samples)
    summary_payload = unique_support_contract_v8._text_summary(
        analysis=analysis,
        input_sha256=split_snapshot.sha256,
        model_tree_sha256=str(provenance["model_tree_sha256"]),
    )
    module = implementation["unique_support_v8_module"]
    runner = implementation["unique_support_v8_cli"]
    report = {
        "schema": unique_support_contract_v8.AUDIT_SCHEMA,
        "audit_version": unique_support_contract_v8.AUDIT_VERSION,
        **analysis,
        "scope": {
            "allowed_splits": ["train", "validation"],
            "answer_examples_only": True,
            "refuse_examples_scored": False,
            "calibration_read": False,
            "reserved_blind_read": False,
            "reserved_blind_discovered": False,
            "network_used": False,
            "training_performed": False,
            "selection_authorized": False,
            "deployment_authorized": False,
            "production_activation_authorized": False,
        },
        "input": {
            "path": split_snapshot.path.as_posix(),
            "sha256": split_snapshot.sha256,
            "bytes": split_snapshot.byte_count,
            "example_count": len(rows),
        },
        "nli_model": dict(provenance),
        "artifacts": {
            "per_sample": {
                "path": unique_support_contract_v8.SAMPLE_FILENAME,
                "sha256": hashlib.sha256(samples_payload).hexdigest(),
                "bytes": len(samples_payload),
                "count": len(samples),
            },
            "text_summary": {
                "path": unique_support_contract_v8.SUMMARY_FILENAME,
                "sha256": hashlib.sha256(summary_payload).hexdigest(),
                "bytes": len(summary_payload),
            },
            "runner": {
                "path": runner.path.as_posix(),
                "sha256": runner.sha256,
                "bytes": runner.byte_count,
            },
            "module": {
                "path": module.path.as_posix(),
                "sha256": module.sha256,
                "bytes": module.byte_count,
            },
        },
    }
    digest = _canonical_sha256(report)
    report["audit_id"] = f"icmat-unique-support-v8:{digest}"
    report["canonical_digest_sha256"] = digest
    observed_report = _strict_json_object_v7(
        report_snapshot,
        label=f"nonblind-v8 {split} unique-support report",
    )
    if (
        observed_report != report
        or sample_snapshot.payload != samples_payload
        or summary_snapshot.payload != summary_payload
        or report.get("status") != unique_support_contract_v8.PASS_STATUS
        or report.get("failed_example_ids") != []
        or not all(bool(gate["passed"]) for gate in report["hard_gates"])
    ):
        raise QLoRAV6Error(
            f"nonblind-v8 {split} unique-support audit does not match "
            "fixed CPU local recomputation"
        )
    _recheck_strict_directory_identity_v7(
        root,
        expected=identity,
        label=f"nonblind-v8 {split} unique-support audit directory",
    )
    return (
        {
            "path": str(report_snapshot.path),
            "bytes": report_snapshot.byte_count,
            "sha256": report_snapshot.sha256,
            "stable_identity": report_snapshot.identity_receipt(),
            "schema": unique_support_contract_v8.AUDIT_SCHEMA,
            "audit_version": unique_support_contract_v8.AUDIT_VERSION,
            "status": unique_support_contract_v8.PASS_STATUS,
            "audit_id": report["audit_id"],
            "split": split,
            "answer_examples_audited": analysis["counts"][
                "answer_examples_audited"
            ],
            "all_spans_locally_recomputed": True,
            "nli_device": "cpu",
        },
        [report_snapshot, sample_snapshot, summary_snapshot],
    )


def _nli_model_identity_receipts_v8(
    root: Path,
) -> dict[str, Any]:
    model_root, root_identity = _strict_directory_identity_v7(
        root,
        label="nonblind-v8 fixed NLI model directory",
    )
    receipt_snapshot = _stable_snapshot_v7(
        model_root.parent / "model_receipt.v1.json",
        label="nonblind-v8 fixed NLI model receipt",
        maximum_bytes=_STRICT_MAX_JSON_BYTES,
    )
    receipt_payload = _strict_json_object_v7(
        receipt_snapshot,
        label="nonblind-v8 fixed NLI model receipt",
    )
    declared_files = receipt_payload.get("files")
    if (
        not isinstance(declared_files, list)
        or len(declared_files)
        != semantic_contract_v7.PINNED_NLI_FILE_COUNT
    ):
        raise QLoRAV6Error(
            "nonblind-v8 fixed NLI model receipt inventory mismatch"
        )
    files: list[dict[str, Any]] = []
    for declaration in declared_files:
        if (
            not isinstance(declaration, Mapping)
            or set(declaration) != {"path", "bytes", "sha256"}
            or not isinstance(declaration.get("path"), str)
            or Path(str(declaration["path"])).is_absolute()
            or ".." in Path(str(declaration["path"])).parts
        ):
            raise QLoRAV6Error(
                "nonblind-v8 fixed NLI model receipt path mismatch"
            )
        path = model_root / str(declaration["path"])
        metadata = os.lstat(path)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or _is_reparse_point(metadata)
        ):
            raise QLoRAV6Error(
                "nonblind-v8 fixed NLI model contains a link/reparse point"
            )
        if not stat.S_ISREG(metadata.st_mode):
            raise QLoRAV6Error(
                "nonblind-v8 fixed NLI model contains a non-regular file"
            )
        files.append(
            {
                "path": str(declaration["path"]),
                "stable_identity": {
                    "device": int(metadata.st_dev),
                    "file_id": int(metadata.st_ino),
                    "size": int(metadata.st_size),
                    "mtime_ns": int(metadata.st_mtime_ns),
                    "ctime_ns": int(metadata.st_ctime_ns),
                },
            }
        )
    if len(files) != semantic_contract_v7.PINNED_NLI_FILE_COUNT:
        raise QLoRAV6Error(
            "nonblind-v8 fixed NLI identity inventory count mismatch"
        )
    return {
        "root": str(model_root),
        "root_identity": {
            "device": root_identity[0],
            "file_id": root_identity[1],
        },
        "tree_sha256": semantic_contract_v7.PINNED_NLI_MODEL_TREE_SHA256,
        "model_receipt": _stable_receipt_v8(receipt_snapshot),
        "files": files,
    }


def _stable_receipt_v8(
    snapshot: StableFileSnapshotV7,
) -> dict[str, Any]:
    return {
        "path": str(snapshot.path),
        "bytes": snapshot.byte_count,
        "sha256": snapshot.sha256,
        "stable_identity": snapshot.identity_receipt(),
    }


def _revalidate_v8_identity_bundle(dataset: Mapping[str, Any]) -> None:
    revalidation = dataset.get("seed_revalidation")
    if (
        not isinstance(revalidation, Mapping)
        or set(revalidation) != {"files", "nli_model"}
        or not isinstance(revalidation.get("files"), list)
    ):
        raise PermissionError(
            "nonblind-v8 seed revalidation bundle is missing"
        )
    for receipt in revalidation["files"]:
        if (
            not isinstance(receipt, Mapping)
            or set(receipt)
            != {"path", "bytes", "sha256", "stable_identity"}
        ):
            raise PermissionError(
                "nonblind-v8 seed file receipt contract mismatch"
            )
        path = _absolute_lexical_v7(Path(str(receipt["path"])))
        _assert_no_link_components_v7(
            path,
            label="nonblind-v8 seed-revalidated file",
        )
        metadata = os.lstat(path)
        observed_identity = {
            "device": int(metadata.st_dev),
            "file_id": int(metadata.st_ino),
            "size": int(metadata.st_size),
            "mtime_ns": int(metadata.st_mtime_ns),
            "ctime_ns": int(metadata.st_ctime_ns),
        }
        if (
            not stat.S_ISREG(metadata.st_mode)
            or _is_reparse_point(metadata)
            or observed_identity != receipt["stable_identity"]
            or int(metadata.st_size) != receipt["bytes"]
        ):
            raise PermissionError(
                "nonblind-v8 training authority changed after preflight"
            )
    nli = revalidation.get("nli_model")
    if (
        not isinstance(nli, Mapping)
        or set(nli)
        != {
            "root",
            "root_identity",
            "tree_sha256",
            "model_receipt",
            "files",
        }
        or nli.get("tree_sha256")
        != semantic_contract_v7.PINNED_NLI_MODEL_TREE_SHA256
        or not isinstance(nli.get("files"), list)
    ):
        raise PermissionError(
            "nonblind-v8 NLI identity receipt contract mismatch"
        )
    model_root = _absolute_lexical_v7(Path(str(nli["root"])))
    root_metadata = os.lstat(model_root)
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or _is_reparse_point(root_metadata)
        or {
            "device": int(root_metadata.st_dev),
            "file_id": int(root_metadata.st_ino),
        }
        != nli["root_identity"]
    ):
        raise PermissionError(
            "nonblind-v8 NLI model root identity changed"
        )
    model_receipt = nli.get("model_receipt")
    if not isinstance(model_receipt, Mapping):
        raise PermissionError(
            "nonblind-v8 NLI model receipt identity is missing"
        )
    receipt_path = _absolute_lexical_v7(
        Path(str(model_receipt["path"]))
    )
    receipt_metadata = os.lstat(receipt_path)
    if (
        not stat.S_ISREG(receipt_metadata.st_mode)
        or _is_reparse_point(receipt_metadata)
        or {
            "device": int(receipt_metadata.st_dev),
            "file_id": int(receipt_metadata.st_ino),
            "size": int(receipt_metadata.st_size),
            "mtime_ns": int(receipt_metadata.st_mtime_ns),
            "ctime_ns": int(receipt_metadata.st_ctime_ns),
        }
        != model_receipt["stable_identity"]
    ):
        raise PermissionError(
            "nonblind-v8 NLI model receipt changed after preflight"
        )
    for receipt in nli["files"]:
        path = model_root / str(receipt["path"])
        metadata = os.lstat(path)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or _is_reparse_point(metadata)
            or {
                "device": int(metadata.st_dev),
                "file_id": int(metadata.st_ino),
                "size": int(metadata.st_size),
                "mtime_ns": int(metadata.st_mtime_ns),
                "ctime_ns": int(metadata.st_ctime_ns),
            }
            != receipt["stable_identity"]
        ):
            raise PermissionError(
                "nonblind-v8 NLI model identity changed after preflight"
            )


def _revalidate_v8_canary_acceptance(
    acceptance: Mapping[str, Any],
) -> None:
    if acceptance.get("validated") is False:
        return
    candidates = [
        ("canary acceptance", acceptance),
        ("canary evaluation index", acceptance.get("evaluation_index")),
        (
            "canary training receipt",
            acceptance.get("canary_training_receipt"),
        ),
    ]
    for label, value in candidates:
        if not isinstance(value, Mapping):
            raise PermissionError(f"nonblind-v8 {label} binding is missing")
        required = {"path", "bytes", "sha256", "stable_identity"}
        if not required.issubset(value):
            raise PermissionError(
                f"nonblind-v8 {label} stable receipt is incomplete"
            )
        path = _absolute_lexical_v7(Path(str(value["path"])))
        _assert_no_link_components_v7(
            path,
            label=f"nonblind-v8 {label}",
        )
        metadata = os.lstat(path)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or _is_reparse_point(metadata)
            or {
                "device": int(metadata.st_dev),
                "file_id": int(metadata.st_ino),
                "size": int(metadata.st_size),
                "mtime_ns": int(metadata.st_mtime_ns),
                "ctime_ns": int(metadata.st_ctime_ns),
            }
            != value["stable_identity"]
            or int(metadata.st_size) != value["bytes"]
        ):
            raise PermissionError(
                f"nonblind-v8 {label} changed after preflight"
            )


def _nonblind_dataset_snapshot_v8(
    root: Path,
    *,
    nonblind_second_build_dir: Path | None,
    nonblind_audit_receipt: Path | None,
    scoped_lexical_audit_v8: Path | None,
    train_unique_support_audit_v8: Path | None,
    validation_unique_support_audit_v8: Path | None,
    unique_support_nli_model_dir: Path | None,
) -> dict[str, Any]:
    gate_paths = {
        "nonblind_second_build_dir": nonblind_second_build_dir,
        "nonblind_audit_receipt": nonblind_audit_receipt,
        "scoped_lexical_audit_v8": scoped_lexical_audit_v8,
        "train_unique_support_audit_v8": (
            train_unique_support_audit_v8
        ),
        "validation_unique_support_audit_v8": (
            validation_unique_support_audit_v8
        ),
        "unique_support_nli_model_dir": unique_support_nli_model_dir,
    }
    missing = [name for name, path in gate_paths.items() if path is None]
    if missing:
        raise QLoRAV6Error(
            "strict nonblind-v8 requires explicit audit gates: "
            + ", ".join(missing)
        )
    root, primary_identity = _strict_directory_identity_v7(
        root,
        label="nonblind-v8 primary build directory",
    )
    _assert_exact_nonblind_inventory_v8(
        root,
        root_identity=primary_identity,
        label="nonblind-v8 primary build directory",
    )
    manifest_snapshot = _stable_snapshot_v7(
        root / NONBLIND_V8_MANIFEST_NAME,
        label=NONBLIND_V8_MANIFEST_NAME,
        maximum_bytes=_STRICT_MAX_JSON_BYTES,
    )
    manifest = _strict_json_object_v7(
        manifest_snapshot,
        label=NONBLIND_V8_MANIFEST_NAME,
    )
    implementation = _strict_implementation_snapshots_v8()
    preregistration = _validate_v8c2_preregistration(
        implementation["v8c2_preregistration"]
    )
    records, source_inputs, builder = _validate_manifest_contract_v8(
        manifest,
        implementation=implementation,
    )
    files = _fixed_nonblind_file_snapshots_v8(
        root,
        label="nonblind-v8 primary build",
        manifest_snapshot=manifest_snapshot,
    )
    split_snapshots = {
        split: files[split] for split in NONBLIND_SPLIT_FILES
    }
    artifact_receipts = {
        role: _verify_receipt_snapshot_v8(
            manifest["artifacts"][role],
            files[role],
            expected_path=filename,
            label=role.replace("_", " "),
        )
        for role, filename in STRICT_V8_ARTIFACT_FILES.items()
    }
    seen_ids: set[str] = set()
    summaries = {
        split: _scan_strict_visible_snapshot_v7(
            split_snapshots[split],
            split=split,
            expected=records[split],
            seen_example_ids=seen_ids,
        )
        for split in NONBLIND_SPLIT_FILES
    }
    _assert_nonblind_split_isolation_v7(summaries)
    nli_model_dir = _absolute_lexical_v7(
        Path(unique_support_nli_model_dir)
    )
    authority_state = _validate_primary_with_independent_v8_parser(
        root=root,
        files=files,
        manifest=manifest,
        source_inputs=source_inputs,
        nli_model_dir=nli_model_dir,
    )
    second = _validate_second_nonblind_build_v8(
        Path(nonblind_second_build_dir),
        primary_root=root,
        primary_files=files,
        primary_manifest=manifest,
    )
    compare_gate = _validate_compare_audit_gate_v8(
        Path(nonblind_audit_receipt),
        primary_root=root,
        secondary_root=second["root"],
        files=files,
        manifest=manifest,
        authority_state=authority_state,
        implementation=implementation,
    )
    lexical_gate, lexical_snapshots = _validate_scoped_lexical_gate_v8(
        Path(scoped_lexical_audit_v8),
        train_snapshot=split_snapshots["train"],
        validation_snapshot=split_snapshots["validation"],
        implementation=implementation,
    )
    try:
        auditor = semantic_contract_v7.LocalTransformersNLIAuditor(
            model_dir=nli_model_dir,
            expected_tree_sha256=(
                semantic_contract_v7.PINNED_NLI_MODEL_TREE_SHA256
            ),
            device="cpu",
        )
    except (
        OSError,
        RuntimeError,
        ValueError,
        semantic_contract_v7.SemanticQueryV7Error,
    ) as exc:
        raise QLoRAV6Error(
            f"nonblind-v8 fixed CPU NLI initialization failed: {exc}"
        ) from exc
    provenance = dict(auditor.provenance)
    if provenance != _expected_nli_provenance_v8():
        raise QLoRAV6Error(
            "nonblind-v8 fixed CPU NLI provenance mismatch"
        )
    train_unique, train_unique_snapshots = (
        _validate_unique_support_gate_v8(
            Path(train_unique_support_audit_v8),
            split="train",
            split_snapshot=split_snapshots["train"],
            auditor=auditor,
            provenance=provenance,
            implementation=implementation,
        )
    )
    validation_unique, validation_unique_snapshots = (
        _validate_unique_support_gate_v8(
            Path(validation_unique_support_audit_v8),
            split="validation",
            split_snapshot=split_snapshots["validation"],
            auditor=auditor,
            provenance=provenance,
            implementation=implementation,
        )
    )
    try:
        final_nli = semantic_contract_v7.validate_pinned_nli_asset(
            nli_model_dir,
            expected_tree_sha256=(
                semantic_contract_v7.PINNED_NLI_MODEL_TREE_SHA256
            ),
        )
    except (
        OSError,
        semantic_contract_v7.SemanticQueryV7Error,
    ) as exc:
        raise QLoRAV6Error(
            "nonblind-v8 fixed CPU NLI changed during recomputation"
        ) from exc
    for key, value in _expected_nli_provenance_v8().items():
        if key in {"backend", "device", "quality_claim_allowed"}:
            continue
        if final_nli.get(key) != value:
            raise QLoRAV6Error(
                "nonblind-v8 fixed CPU NLI final binding mismatch"
            )
    nli_identity = _nli_model_identity_receipts_v8(nli_model_dir)
    gate_bundle = _v8_training_gate_bundle(
        {
            "nonblind_compare": {
                "sha256": compare_gate["sha256"],
                "status": compare_gate["status"],
            },
            "scoped_lexical": {
                "sha256": lexical_gate["sha256"],
                "status": lexical_gate["status"],
            },
            "unique_support": {
                "train": {
                    "sha256": train_unique["sha256"],
                    "status": train_unique["status"],
                },
                "validation": {
                    "sha256": validation_unique["sha256"],
                    "status": validation_unique["status"],
                },
            },
            "nli_model": {
                "tree_sha256": provenance["model_tree_sha256"],
                "receipt_sha256": provenance["model_receipt_sha256"],
                "device": "cpu",
            },
        }
    )
    formal_data_binding = _validate_v8c2_frozen_data_binding(
        preregistration=preregistration,
        manifest={"sha256": manifest_snapshot.sha256},
        splits=summaries,
        training_gate_bundle=gate_bundle,
    )
    implementation_receipts = {
        role: _stable_receipt_v8(snapshot)
        for role, snapshot in sorted(implementation.items())
    }
    seed_files = [
        *files.values(),
        *second["files"].values(),
        *lexical_snapshots,
        *train_unique_snapshots,
        *validation_unique_snapshots,
        *implementation.values(),
        _stable_snapshot_v7(
            V8C2_PREDECESSOR_ACCEPTANCE_PATH,
            label="v8c2 predecessor STOP receipt revalidation",
            maximum_bytes=_STRICT_MAX_JSON_BYTES,
        ),
        _stable_snapshot_v7(
            Path(nonblind_audit_receipt),
            label="nonblind-v8 compare receipt revalidation",
            maximum_bytes=_STRICT_MAX_JSON_BYTES,
        ),
    ]
    deduplicated_seed_files = {
        str(snapshot.path): snapshot for snapshot in seed_files
    }
    inspected_core = {
        "manifest": _stable_receipt_v8(manifest_snapshot),
        "builder_version": NONBLIND_V8_BUILDER_VERSION,
        "source_inputs": source_inputs,
        "builder": builder,
        "fixed_files": {
            role: _stable_receipt_v8(snapshot)
            for role, snapshot in sorted(files.items())
        },
        "strict_artifact_receipts": artifact_receipts,
        "double_build_evidence": second["comparisons"],
        "strict_audit_gates": {
            "nonblind_compare": compare_gate,
            "scoped_lexical": lexical_gate,
            "unique_support": {
                "train": train_unique,
                "validation": validation_unique,
            },
        },
        "training_gate_bundle": gate_bundle,
        "v8c2_formal_data_binding": formal_data_binding,
        "v8c2_preregistration": preregistration,
        "implementation_receipts": implementation_receipts,
        "nli_model_identity": nli_identity,
    }
    inspected_input_sha256 = _canonical_sha256(inspected_core)
    _recheck_strict_directory_identity_v7(
        root,
        expected=primary_identity,
        label="nonblind-v8 primary build directory",
    )
    _assert_exact_nonblind_inventory_v8(
        root,
        root_identity=primary_identity,
        label="nonblind-v8 primary build directory",
    )
    return {
        "path": str(root),
        "contract": "STRICT_NONBLIND_V8",
        "manifest": {
            "path": NONBLIND_V8_MANIFEST_NAME,
            "bytes": manifest_snapshot.byte_count,
            "sha256": manifest_snapshot.sha256,
            "stable_identity": manifest_snapshot.identity_receipt(),
            "schema": NONBLIND_V8_MANIFEST_SCHEMA,
            "dataset_schema": DATASET_SCHEMA,
            "builder_version": NONBLIND_V8_BUILDER_VERSION,
        },
        "splits": summaries,
        "source_input_binding": source_inputs,
        "strict_artifact_receipts": artifact_receipts,
        "double_build_evidence": second["comparisons"],
        "strict_audit_gates": {
            "nonblind_compare": compare_gate,
            "scoped_lexical": lexical_gate,
            "unique_support": {
                "train": train_unique,
                "validation": validation_unique,
            },
        },
        "training_gate_bundle": gate_bundle,
        "training_gate_bundle_sha256": gate_bundle[
            "training_gate_bundle_sha256"
        ],
        "v8c2_formal_data_binding": formal_data_binding,
        "v8c2_preregistration": preregistration,
        "implementation_receipts": implementation_receipts,
        "seed_revalidation": {
            "files": [
                _stable_receipt_v8(snapshot)
                for _, snapshot in sorted(
                    deduplicated_seed_files.items()
                )
            ],
            "nli_model": nli_identity,
        },
        "training_data_access": {
            "opened_splits": list(READABLE_SPLITS),
            "integrity_only_splits": ["calibration"],
            "primary_fixed_files_stably_opened": 12,
            "second_fixed_files_stably_opened": 12,
            "second_build_bytes_compared_directly": True,
            "second_build_file_identities_compared_directly": True,
            "nonblind_compare_audit_verified": True,
            "scoped_lexical_audit_locally_recomputed": True,
            "train_unique_support_locally_recomputed": True,
            "validation_unique_support_locally_recomputed": True,
            "unique_support_nli_load_count": 1,
            "unique_support_nli_device": "cpu",
            "calibration_integrity_snapshot_opened": True,
            "calibration_integrity_content_read": True,
            "calibration_integrity_content_parsed": True,
            "calibration_integrity_content_hashed": True,
            "calibration_content_loaded_for_training": False,
            "calibration_used_for_checkpoint_selection": False,
            "blind_materialized": False,
            "blind_discovered": False,
            "blind_path_constructed": False,
            "blind_filesystem_metadata_accessed": False,
            "blind_content_opened": False,
            "blind_content_read": False,
            "blind_content_hashed": False,
        },
        "inspected_input_sha256": inspected_input_sha256,
        "v8_inspected_input_sha256": inspected_input_sha256,
    }


def _dataset_snapshot(
    dataset_dir: Path,
    *,
    nonblind_second_build_dir: Path | None = None,
    nonblind_audit_receipt: Path | None = None,
    train_shortcut_audit: Path | None = None,
    validation_shortcut_audit: Path | None = None,
    scoped_lexical_audit_v8: Path | None = None,
    train_unique_support_audit_v8: Path | None = None,
    validation_unique_support_audit_v8: Path | None = None,
    unique_support_nli_model_dir: Path | None = None,
) -> dict[str, Any]:
    lexical_root = Path(os.path.abspath(os.fspath(Path(dataset_dir))))
    root = lexical_root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    manifest_path = root / MANIFEST_NAME
    nonblind_manifest_path = root / NONBLIND_MANIFEST_NAME
    nonblind_v8_manifest_path = root / NONBLIND_V8_MANIFEST_NAME

    def manifest_present(path: Path, *, label: str) -> bool:
        try:
            metadata = os.lstat(path)
        except FileNotFoundError:
            return False
        if (
            stat.S_ISLNK(metadata.st_mode)
            or _is_reparse_point(metadata)
            or not stat.S_ISREG(metadata.st_mode)
        ):
            raise QLoRAV6Error(f"{label} must be a regular non-reparse file")
        return True

    legacy_present = manifest_present(
        manifest_path,
        label=MANIFEST_NAME,
    )
    nonblind_present = manifest_present(
        nonblind_manifest_path,
        label=NONBLIND_MANIFEST_NAME,
    )
    if legacy_present and nonblind_present:
        raise QLoRAV6Error(
            "dataset contains both or multiple supported manifests"
        )
    v8_only_paths = {
        "scoped_lexical_audit_v8": scoped_lexical_audit_v8,
        "train_unique_support_audit_v8": (
            train_unique_support_audit_v8
        ),
        "validation_unique_support_audit_v8": (
            validation_unique_support_audit_v8
        ),
        "unique_support_nli_model_dir": unique_support_nli_model_dir,
    }
    if nonblind_present:
        if any(path is not None for path in v8_only_paths.values()):
            raise QLoRAV6Error(
                "strict dataset gates cannot mix v7 and v8 parameters"
            )
        return _nonblind_dataset_snapshot_v7(
            lexical_root,
            nonblind_second_build_dir=nonblind_second_build_dir,
            nonblind_audit_receipt=nonblind_audit_receipt,
            train_shortcut_audit=train_shortcut_audit,
            validation_shortcut_audit=validation_shortcut_audit,
        )
    nonblind_v8_present = False
    if not legacy_present:
        nonblind_v8_present = manifest_present(
            nonblind_v8_manifest_path,
            label=NONBLIND_V8_MANIFEST_NAME,
        )
    if nonblind_v8_present:
        if (
            train_shortcut_audit is not None
            or validation_shortcut_audit is not None
        ):
            raise QLoRAV6Error(
                "strict dataset gates cannot mix v7 and v8 parameters"
            )
        return _nonblind_dataset_snapshot_v8(
            lexical_root,
            nonblind_second_build_dir=nonblind_second_build_dir,
            nonblind_audit_receipt=nonblind_audit_receipt,
            scoped_lexical_audit_v8=scoped_lexical_audit_v8,
            train_unique_support_audit_v8=(
                train_unique_support_audit_v8
            ),
            validation_unique_support_audit_v8=(
                validation_unique_support_audit_v8
            ),
            unique_support_nli_model_dir=unique_support_nli_model_dir,
        )
    if not legacy_present:
        raise QLoRAV6Error(
            "dataset must contain exactly one supported manifest"
        )
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QLoRAV6Error("manifest.v6.json is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, Mapping):
        raise QLoRAV6Error("manifest.v6.json must contain an object")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise QLoRAV6Error("manifest schema mismatch")
    if manifest.get("dataset_schema") != DATASET_SCHEMA:
        raise QLoRAV6Error("manifest dataset schema mismatch")
    builder_version = manifest.get("builder_version")
    if builder_version not in SUPPORTED_BUILDER_VERSIONS:
        raise QLoRAV6Error("manifest builder version mismatch")
    if manifest.get("status") != "DATASET_BUILT_BLIND_HASH_SEALED":
        raise QLoRAV6Error("manifest status is not training-eligible")
    pointer = manifest.get("pointer_contract")
    if not isinstance(pointer, Mapping) or pointer.get("field_order") != list(
        POINTER_FIELDS
    ):
        raise QLoRAV6Error("manifest pointer contract mismatch")
    if (
        pointer.get("answer_span_pattern") != "E#.S#"
        or pointer.get("refusal_span_id") is not None
    ):
        raise QLoRAV6Error("manifest pointer span contract mismatch")
    if manifest.get("training_boundary") != {
        "allowed_splits": ["train", "validation"],
        "calibration_content_for_training": False,
        "forbidden_split": "blind_test",
        "blind_test_requires_explicit_post_freeze_authorization": True,
        "blind_test_content_in_public_reports": False,
    }:
        raise QLoRAV6Error("manifest training boundary mismatch")
    records = _manifest_split_records(manifest)
    if (
        builder_version == SEMANTIC_BUILDER_VERSION
        and {
            split: records[split]["examples"]
            for split in SPLIT_FILES
        }
        != SEMANTIC_SPLIT_COUNTS
    ):
        raise QLoRAV6Error("semantic-v7 split counts mismatch")
    seen_example_ids: set[str] = set()
    summaries: dict[str, Any] = {}
    for split in READABLE_SPLITS:
        summaries[split] = _scan_visible_jsonl(
            root / SPLIT_FILES[split],
            split=split,
            expected=records[split],
            seen_example_ids=seen_example_ids,
        )
    train_sources = set(summaries["train"]["source_ids"])
    validation_sources = set(summaries["validation"]["source_ids"])
    if train_sources & validation_sources:
        raise QLoRAV6Error("train/validation source-family leakage")
    for split in DECLARATION_ONLY_SPLITS:
        summaries[split] = _declaration_only_record(split, records[split])

    semantic_binding = (
        _semantic_manifest_binding_v7(root, manifest)
        if builder_version == SEMANTIC_BUILDER_VERSION
        else {
            "required": False,
            "legacy_builder_version": EXPECTED_BUILDER_VERSION,
        }
    )
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    inspected_core = {
        "manifest_sha256": manifest_sha256,
        "builder_version": builder_version,
        "read_files": [
            {
                "path": summaries[split]["path"],
                "bytes": summaries[split]["bytes"],
                "sha256": summaries[split]["sha256"],
                "examples": summaries[split]["examples"],
            }
            for split in READABLE_SPLITS
        ],
        "declaration_only": [
            {
                "split": split,
                "path": summaries[split]["path"],
                "bytes": summaries[split]["bytes"],
                "sha256": summaries[split]["sha256"],
                "examples": summaries[split]["examples"],
                "content_read": False,
                "content_hashed": False,
            }
            for split in DECLARATION_ONLY_SPLITS
        ],
        "semantic_binding": semantic_binding,
    }
    return {
        "path": str(root),
        "manifest": {
            "path": MANIFEST_NAME,
            "bytes": len(manifest_bytes),
            "sha256": manifest_sha256,
            "schema": MANIFEST_SCHEMA,
            "dataset_schema": DATASET_SCHEMA,
            "builder_version": builder_version,
        },
        "splits": summaries,
        "semantic_binding": semantic_binding,
        "training_data_access": {
            "opened_splits": list(READABLE_SPLITS),
            "declaration_only_splits": list(DECLARATION_ONLY_SPLITS),
            "calibration_content_read": False,
            "calibration_content_hashed": False,
            "blind_test_content_read": False,
            "blind_test_content_hashed": False,
        },
        "inspected_input_sha256": _canonical_sha256(inspected_core),
    }


def _require_exact_keys_v7(
    value: Any,
    expected: set[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise QLoRAV6Error(f"{label} keys mismatch")
    return value


def _require_timestamp_v7(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise QLoRAV6Error(f"{label} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise QLoRAV6Error(f"{label} is invalid") from exc
    if parsed.tzinfo is None:
        raise QLoRAV6Error(f"{label} must include a timezone")
    return value


def _same_regular_path_v7(
    declared: Any,
    actual: Path,
    *,
    label: str,
) -> bool:
    if not isinstance(declared, str) or not declared:
        return False
    declared_path = _absolute_lexical_v7(Path(declared))
    actual_path = _absolute_lexical_v7(actual)
    try:
        _assert_no_link_components_v7(declared_path, label=label)
        _assert_no_link_components_v7(actual_path, label=label)
        return os.path.samefile(declared_path, actual_path)
    except (OSError, QLoRAV6Error):
        return False


def _strict_child_directory_v7(
    path: Any,
    *,
    root: Path,
    label: str,
) -> Path:
    if not isinstance(path, str) or not path:
        raise QLoRAV6Error(f"{label} path is invalid")
    resolved, _ = _strict_directory_identity_v7(
        Path(path),
        label=label,
    )
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise QLoRAV6Error(f"{label} must stay under the evaluation root") from exc
    return resolved


def _expected_canary_thresholds_v7() -> dict[str, Any]:
    return {
        "completed_samples": 18,
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
    }


def _expected_run_authorization_v7() -> dict[str, bool]:
    return {
        "checkpoint_selected": False,
        "model_authorized": False,
        "calibration_authorized": False,
        "blind_test_authorized": False,
        "gguf_export_authorized": False,
        "deployment_authorized": False,
        "production_integration_authorized": False,
    }


def _expected_seed_authorization_v7() -> dict[str, bool]:
    return {
        "checkpoint_selected": False,
        "model_authorized": False,
        "calibration_authorized": False,
        "blind_test_authorized": False,
        "deployment_authorized": False,
    }


def _validate_canary_training_generation_binding_v8(
    *,
    training: Mapping[str, Any],
    dataset: Mapping[str, Any],
) -> str:
    dataset_kind = _strict_dataset_kind(dataset)
    input_snapshot = training.get("input_snapshot")
    canary_dataset = (
        input_snapshot.get("dataset")
        if isinstance(input_snapshot, Mapping)
        else None
    )
    if not isinstance(canary_dataset, Mapping):
        raise QLoRAV6Error("canary training dataset generation is missing")
    canary_kind = _strict_dataset_kind(canary_dataset)
    if dataset_kind == "v8" and canary_kind == "v7":
        raise QLoRAV6Error("v7 canary cannot authorize v8 final")
    if canary_kind != dataset_kind:
        raise QLoRAV6Error(
            "canary and final dataset generations do not match"
        )
    if dataset_kind != "v8":
        return dataset_kind
    protocol = dataset.get("v8c2_preregistration")
    expected_protocol = {
        "protocol_id": V8C2_PREREGISTRATION_PROTOCOL_ID,
        "profile": V8C2_TRAINING_PROFILE,
        "sha256": V8C2_PREREGISTRATION_SHA256,
    }
    if (
        not isinstance(protocol, Mapping)
        or any(
            protocol.get(key) != value
            for key, value in expected_protocol.items()
        )
        or any(
            training.get(key) != value
            for key, value in _v8c2_receipt_fields().items()
        )
    ):
        raise QLoRAV6Error(
            "v8c2 canary protocol binding mismatch"
        )
    if (
        dataset.get("contract") != "STRICT_NONBLIND_V8"
        or canary_dataset.get("contract") != "STRICT_NONBLIND_V8"
        or not _valid_sha256(dataset.get("training_gate_bundle_sha256"))
        or not _valid_sha256(dataset.get("v8_inspected_input_sha256"))
        or training.get("training_gate_bundle_sha256")
        != dataset.get("training_gate_bundle_sha256")
        or training.get("v8_inspected_input_sha256")
        != dataset.get("v8_inspected_input_sha256")
        or canary_dataset.get("training_gate_bundle_sha256")
        != dataset.get("training_gate_bundle_sha256")
        or canary_dataset.get("v8_inspected_input_sha256")
        != dataset.get("v8_inspected_input_sha256")
    ):
        raise QLoRAV6Error("v8 canary generation binding mismatch")
    return dataset_kind


def _validate_canary_training_contract_v7(
    *,
    training: Mapping[str, Any],
    training_snapshot: StableFileSnapshotV7,
    dataset: Mapping[str, Any],
    model: Mapping[str, Any],
    final_configuration: Mapping[str, Any],
) -> tuple[int, list[dict[str, Any]]]:
    from icmat_foundry.llm import pointer_checkpoint_eval_v6

    dataset_kind = _strict_dataset_kind(dataset)
    expected_training_keys = {
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
    }
    if dataset_kind == "v8":
        expected_training_keys.update(
            {
                "training_gate_bundle_sha256",
                "v8_inspected_input_sha256",
                "training_profile",
                "preregistration_protocol_id",
                "preregistration_sha256",
                "canary_attempt",
            }
        )
    _require_exact_keys_v7(
        training,
        expected_training_keys,
        label="canary training receipt",
    )
    _validate_canary_training_generation_binding_v8(
        training=training,
        dataset=dataset,
    )
    _require_timestamp_v7(
        training.get("created_at"),
        label="canary training receipt timestamp",
    )
    if (
        training.get("schema") != RUN_RECEIPT_SCHEMA
        or training.get("trainer_version") != TRAINER_VERSION
        or training.get("stage") != "canary"
        or training.get("status")
        != "PASS_CANARY_SINGLE_SEED_ALL_EPOCHS_NOT_SELECTED"
        or training.get("checkpoint_count") != FIXED_EPOCHS
        or training.get("atomic_publish") is not True
        or training.get("network_used") is not False
        or not isinstance(training.get("run_id"), str)
        or not training["run_id"]
    ):
        raise QLoRAV6Error(
            "canary training receipt is not a completed immutable 1x6 run"
        )
    wall_seconds = training.get("wall_seconds")
    if (
        isinstance(wall_seconds, bool)
        or not isinstance(wall_seconds, (int, float))
        or not math.isfinite(float(wall_seconds))
        or float(wall_seconds) < 0.0
    ):
        raise QLoRAV6Error("canary training wall time is invalid")

    configuration = _require_exact_keys_v7(
        training.get("configuration"),
        set(final_configuration),
        label="canary training configuration",
    )
    seeds_value = configuration.get("seeds")
    if (
        not isinstance(seeds_value, list)
        or len(seeds_value) != 1
        or isinstance(seeds_value[0], bool)
        or not isinstance(seeds_value[0], int)
        or seeds_value != list(CANARY_DEFAULT_SEEDS)
    ):
        raise QLoRAV6Error("canary training seed contract mismatch")
    seed = int(seeds_value[0])
    expected_configuration = dict(final_configuration)
    expected_configuration["stage"] = "canary"
    expected_configuration["seeds"] = [seed]
    if (
        dict(configuration) != expected_configuration
        or training.get("configuration_sha256")
        != _canonical_sha256(configuration)
    ):
        raise QLoRAV6Error(
            "canary and final QLoRA algorithm configurations differ"
        )

    input_snapshot = _require_exact_keys_v7(
        training.get("input_snapshot"),
        {
            "dataset",
            "base_model",
            "canary_acceptance",
            "source_files",
        },
        label="canary training input snapshot",
    )
    canary_dataset = _require_exact_keys_v7(
        input_snapshot.get("dataset"),
        set(dataset),
        label="canary training dataset snapshot",
    )
    if (
        not _same_regular_path_v7(
            canary_dataset.get("path"),
            Path(str(dataset["path"])),
            label="canary training dataset",
        )
        or canary_dataset.get("inspected_input_sha256")
        != dataset.get("inspected_input_sha256")
        or dict(canary_dataset) != dict(dataset)
    ):
        raise QLoRAV6Error(
            "canary training dataset binding mismatch"
        )
    canary_model_raw = input_snapshot.get("base_model")
    if (
        not isinstance(canary_model_raw, Mapping)
        or set(canary_model_raw)
        not in (set(model), set(model) | {"runtime_loading"})
    ):
        raise QLoRAV6Error(
            "canary training base-model snapshot keys mismatch"
        )
    canary_model = canary_model_raw
    model_core = {
        key: value
        for key, value in canary_model.items()
        if key != "runtime_loading"
    }
    if (
        not _same_regular_path_v7(
            model_core.get("path"),
            Path(str(model["path"])),
            label="canary training base model",
        )
        or model_core != dict(model)
    ):
        raise QLoRAV6Error(
            "canary training base-model binding mismatch"
        )
    if "runtime_loading" in canary_model:
        runtime_loading = _require_exact_keys_v7(
            canary_model.get("runtime_loading"),
            {
                "policy",
                "content_address",
                "tree_sha256",
                "file_count",
                "bytes",
                "verified_before_cuda",
                "loaded_only_from_snapshot",
                "removed_before_publish",
            },
            label="canary runtime model snapshot",
        )
        if runtime_loading != {
            "policy": "CONTENT_ADDRESSED_STABLE_LOCAL_COPY_V7",
            "content_address": f"sha256:{model['tree_sha256']}",
            "tree_sha256": model["tree_sha256"],
            "file_count": model["file_count"],
            "bytes": model["bytes"],
            "verified_before_cuda": True,
            "loaded_only_from_snapshot": True,
            "removed_before_publish": True,
        }:
            raise QLoRAV6Error(
                "canary runtime model snapshot contract mismatch"
            )
    if input_snapshot.get("canary_acceptance") != {
        "required_for_final_training": False,
        "provided": False,
        "validated": False,
    }:
        raise QLoRAV6Error(
            "canary training must not consume a prior acceptance gate"
        )
    strict_v7 = dataset_kind == "v7"
    strict_v8 = dataset_kind == "v8"
    strict_nonblind = dataset_kind in {"v7", "v8"}
    if input_snapshot.get("source_files") != _source_inventory(
        strict_nonblind=strict_nonblind,
        strict_v8=strict_v8,
    ):
        raise QLoRAV6Error(
            "canary training implementation identity mismatch"
        )
    if strict_v8:
        attempt = training.get("canary_attempt")
        if not isinstance(attempt, Mapping):
            raise QLoRAV6Error("v8c2 canary attempt receipt is missing")
        _validate_v8c2_canary_attempt_receipt(
            attempt,
            run_id=str(training["run_id"]),
            configuration_sha256=str(training["configuration_sha256"]),
            dataset_input_sha256=str(
                canary_dataset["inspected_input_sha256"]
            ),
            training_gate_bundle_sha256=str(
                training["training_gate_bundle_sha256"]
            ),
            source_inventory_sha256=_canonical_sha256(
                input_snapshot["source_files"]
            ),
            base_model_tree_sha256=str(model_core["tree_sha256"]),
        )

    software = _require_exact_keys_v7(
        training.get("software"),
        {"python", "platform", "dependencies"},
        label="canary software receipt",
    )
    if (
        not isinstance(software.get("python"), str)
        or not isinstance(software.get("platform"), str)
        or not isinstance(software.get("dependencies"), Mapping)
    ):
        raise QLoRAV6Error("canary software receipt is invalid")
    cuda = _require_exact_keys_v7(
        training.get("cuda"),
        {"torch_cuda", "cudnn", "nvidia_driver"},
        label="canary CUDA receipt",
    )
    if not all(
        value is None or isinstance(value, (str, int))
        for value in cuda.values()
    ):
        raise QLoRAV6Error("canary CUDA receipt is invalid")
    if training.get("authorization") != _expected_run_authorization_v7():
        raise QLoRAV6Error(
            "canary training authorization boundary mismatch"
        )
    selection = _require_exact_keys_v7(
        training.get("selection"),
        {
            "automatic_selection_performed",
            "selected_seed",
            "selected_epoch",
            "selected_adapter",
            "selection_metric",
            "required_next_step",
        },
        label="canary training selection",
    )
    if selection != {
        "automatic_selection_performed": False,
        "selected_seed": None,
        "selected_epoch": None,
        "selected_adapter": None,
        "selection_metric": None,
        "required_next_step": (
            "independent full validation pointer evaluation"
        ),
    }:
        raise QLoRAV6Error(
            "canary training must remain completely unselected"
        )
    data_access = training.get("data_access")
    required_access = {
        "train_content_read": True,
        "validation_content_read": True,
        "calibration_content_read": False,
        "calibration_content_hashed": False,
        "blind_test_content_read": False,
        "blind_test_content_hashed": False,
    }
    if not isinstance(data_access, Mapping):
        raise QLoRAV6Error("canary training data-access receipt is invalid")
    for key, value in required_access.items():
        if data_access.get(key) is not value:
            raise QLoRAV6Error(
                "canary training data-access boundary mismatch"
            )
    if strict_v8:
        expected_extra = {
            "calibration_integrity_snapshot_opened": True,
            "calibration_integrity_content_read": True,
            "calibration_integrity_content_hashed": True,
            "calibration_content_loaded_for_training": False,
            "calibration_used_for_checkpoint_selection": False,
            "nonblind_compare_audit_verified": True,
            "scoped_lexical_audit_verified": True,
            "scoped_lexical_audit_locally_recomputed": True,
            "train_unique_support_audit_verified": True,
            "validation_unique_support_audit_verified": True,
            "unique_support_fixed_cpu_nli_load_count": 1,
            "unique_support_nli_repeated_per_seed": False,
            "second_build_fixed_files_recomputed": True,
            "declared_nonblind_audit_artifacts_opened": 8,
            "declared_nonblind_audit_artifacts_hashed": 8,
            "blind_materialized": False,
            "blind_discovered": False,
            "blind_path_constructed": False,
            "blind_filesystem_metadata_accessed": False,
            "blind_content_opened": False,
            "blind_content_read": False,
            "blind_content_hashed": False,
        }
        if set(data_access) != set(required_access) | set(expected_extra):
            raise QLoRAV6Error(
                "strict v8 canary training data-access keys mismatch"
            )
        if any(
            data_access.get(key) != value
            for key, value in expected_extra.items()
        ):
            raise QLoRAV6Error(
                "strict v8 canary training data-access boundary mismatch"
            )
    elif strict_v7:
        expected_extra = {
            "calibration_legacy_fields_mean_training_access_only": True,
            "calibration_integrity_snapshot_opened": True,
            "calibration_integrity_content_read": True,
            "calibration_integrity_content_hashed": True,
            "calibration_content_loaded_for_training": False,
            "calibration_used_for_checkpoint_selection": False,
            "nonblind_compare_audit_verified": True,
            "train_shortcut_audit_verified": True,
            "validation_shortcut_audit_verified": True,
            "second_build_fixed_files_recomputed": True,
            "shortcut_audits_locally_recomputed": True,
            "declared_nonblind_audit_artifacts_opened": 6,
            "declared_nonblind_audit_artifacts_hashed": 6,
            "blind_materialized": False,
            "blind_discovered": False,
            "blind_path_constructed": False,
            "blind_filesystem_metadata_accessed": False,
            "blind_content_opened": False,
            "blind_content_read": False,
            "blind_content_hashed": False,
        }
        if set(data_access) != set(required_access) | set(expected_extra):
            raise QLoRAV6Error(
                "strict canary training data-access keys mismatch"
            )
        if any(
            data_access.get(key) != value
            for key, value in expected_extra.items()
        ):
            raise QLoRAV6Error(
                "strict canary training data-access boundary mismatch"
            )
    elif set(data_access) != set(required_access):
        raise QLoRAV6Error(
            "legacy canary training data-access keys mismatch"
        )

    seeds = training.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 1:
        raise QLoRAV6Error(
            "canary training must contain exactly one seed receipt"
        )
    seed_receipt = _require_exact_keys_v7(
        seeds[0],
        {
            "schema",
            "trainer_version",
            "created_at",
            "status",
            "stage",
            "seed",
            "configuration",
            "dataset",
            "model_parameters",
            "per_epoch_metrics",
            "epoch_checkpoints",
            "metrics",
            "hardware",
            "authorization",
        }
        | (set(_v8c2_receipt_fields()) if strict_v8 else set()),
        label="canary seed receipt",
    )
    if (
        seed_receipt.get("schema") != SEED_RECEIPT_SCHEMA
        or seed_receipt.get("trainer_version") != TRAINER_VERSION
        or seed_receipt.get("status")
        != "PASS_SEED_TRAINED_ALL_EPOCHS_NOT_SELECTED"
        or seed_receipt.get("stage") != "canary"
        or seed_receipt.get("seed") != seed
        or seed_receipt.get("configuration") != configuration
        or seed_receipt.get("authorization")
        != _expected_seed_authorization_v7()
        or (
            strict_v8
            and any(
                seed_receipt.get(key) != value
                for key, value in _v8c2_receipt_fields().items()
            )
        )
    ):
        raise QLoRAV6Error("canary seed receipt contract mismatch")
    if seed_receipt.get("created_at") != "fixture":
        _require_timestamp_v7(
            seed_receipt.get("created_at"),
            label="canary seed receipt timestamp",
        )
    seed_dataset = _require_exact_keys_v7(
        seed_receipt.get("dataset"),
        {
            "train_examples",
            "validation_examples",
            "calibration_content_read",
            "calibration_content_hashed",
            "blind_test_content_read",
            "blind_test_content_hashed",
            "train_tokenization",
            "validation_tokenization",
        },
        label="canary seed dataset receipt",
    )
    if (
        seed_dataset.get("train_examples")
        != dataset["splits"]["train"]["examples"]
        or seed_dataset.get("validation_examples")
        != dataset["splits"]["validation"]["examples"]
        or any(
            seed_dataset.get(field) is not False
            for field in (
                "calibration_content_read",
                "calibration_content_hashed",
                "blind_test_content_read",
                "blind_test_content_hashed",
            )
        )
        or not isinstance(seed_dataset.get("train_tokenization"), Mapping)
        or not isinstance(
            seed_dataset.get("validation_tokenization"),
            Mapping,
        )
    ):
        raise QLoRAV6Error("canary seed dataset contract mismatch")
    history = seed_receipt.get("per_epoch_metrics")
    checkpoints = seed_receipt.get("epoch_checkpoints")
    if (
        not isinstance(history, list)
        or len(history) != FIXED_EPOCHS
        or not isinstance(checkpoints, list)
        or len(checkpoints) != FIXED_EPOCHS
    ):
        raise QLoRAV6Error(
            "canary seed must retain all six epochs and checkpoints"
        )
    for epoch, record in enumerate(history, start=1):
        record = _require_exact_keys_v7(
            record,
            {
                "epoch",
                "global_step",
                "train_loss",
                "validation_loss",
                "learning_rate",
                "validation_runtime_seconds",
            },
            label=f"canary epoch {epoch} metrics",
        )
        if (
            record.get("epoch") != epoch
            or isinstance(record.get("global_step"), bool)
            or not isinstance(record.get("global_step"), int)
            or record["global_step"] < 1
        ):
            raise QLoRAV6Error(
                f"canary epoch {epoch} metrics contract mismatch"
            )
    try:
        stage, specs = pointer_checkpoint_eval_v6._checkpoint_specs(
            receipt=training,
            training_root=training_snapshot.path.parent,
        )
    except Exception as exc:
        raise QLoRAV6Error(
            "canary training checkpoint artifacts failed independent "
            "validation"
        ) from exc
    if stage != "canary" or len(specs) != FIXED_EPOCHS:
        raise QLoRAV6Error(
            "canary training checkpoint population mismatch"
        )
    return seed, specs


def _validate_canary_eval_run_receipt_v7(
    *,
    snapshot: StableFileSnapshotV7,
    sample_snapshot: StableFileSnapshotV7,
    summary_snapshot: StableFileSnapshotV7,
    validation_view: StableFileSnapshotV7,
    checkpoint: Mapping[str, Any],
    implementation: Mapping[str, StableFileSnapshotV7],
    model: Mapping[str, Any],
) -> None:
    from icmat_foundry.llm import pointer_hf_eval_v6

    receipt = _strict_json_object_v7(
        snapshot,
        label="canary checkpoint run receipt",
    )
    _require_exact_keys_v7(
        receipt,
        {
            "schema",
            "status",
            "created_at_utc",
            "evaluator_version",
            "dataset",
            "execution",
            "implementation",
            "bindings",
            "artifacts",
            "claim_boundary",
        },
        label="canary checkpoint run receipt",
    )
    _require_timestamp_v7(
        receipt.get("created_at_utc"),
        label="canary checkpoint run timestamp",
    )
    if (
        receipt.get("schema") != pointer_hf_eval_v6.RUN_RECEIPT_SCHEMA
        or receipt.get("status") != "VALIDATION_EVALUATION_COMPLETE"
        or receipt.get("evaluator_version")
        != pointer_hf_eval_v6.EVALUATOR_VERSION
    ):
        raise QLoRAV6Error(
            "canary checkpoint run receipt identity mismatch"
        )
    dataset_receipt = _require_exact_keys_v7(
        receipt.get("dataset"),
        {
            "directory",
            "opened_split_path",
            "opened_split_sha256",
            "opened_split_bytes",
            "rows_in_file",
            "rows_evaluated",
            "max_samples",
            "files_opened_by_dataset_loader",
            "blind_data_accessed",
        },
        label="canary checkpoint evaluator dataset",
    )
    if (
        not _same_regular_path_v7(
            dataset_receipt.get("opened_split_path"),
            validation_view.path,
            label="canary validation view",
        )
        or dataset_receipt.get("opened_split_sha256")
        != validation_view.sha256
        or dataset_receipt.get("opened_split_bytes")
        != validation_view.byte_count
        or dataset_receipt.get("rows_in_file") != 18
        or dataset_receipt.get("rows_evaluated") != 18
        or dataset_receipt.get("max_samples") is not None
        or dataset_receipt.get("files_opened_by_dataset_loader")
        != [str(validation_view.path)]
        or dataset_receipt.get("blind_data_accessed") is not False
    ):
        raise QLoRAV6Error(
            "canary checkpoint evaluator dataset binding mismatch"
        )
    execution = _require_exact_keys_v7(
        receipt.get("execution"),
        {
            "backend",
            "model_request_type",
            "model_input_roles",
            "expected_passed_to_model",
            "expected_passed_to_candidate_compiler",
            "gold_repair_applied",
            "blind_supported",
            "blind_data_accessed",
        },
        label="canary checkpoint evaluator execution",
    )
    backend = execution.get("backend")
    backend_model = (
        backend.get("model") if isinstance(backend, Mapping) else None
    )
    base = (
        backend_model.get("base")
        if isinstance(backend_model, Mapping)
        else None
    )
    adapter = (
        backend_model.get("adapter")
        if isinstance(backend_model, Mapping)
        else None
    )
    if (
        execution.get("model_request_type")
        != "GenerationRequestV6_target_free"
        or execution.get("model_input_roles") != ["system", "user"]
        or any(
            execution.get(field) is not False
            for field in (
                "expected_passed_to_model",
                "expected_passed_to_candidate_compiler",
                "gold_repair_applied",
                "blind_supported",
                "blind_data_accessed",
            )
        )
        or not isinstance(backend, Mapping)
        or backend.get("mode") != "hf_model"
        or backend.get("subject") != "adapter"
        or backend.get("samples_generated") != 18
        or backend.get("local_files_only") is not True
        or backend.get("network_allowed") is not False
        or backend.get("assistant_target_visible") is not False
        or not isinstance(backend_model, Mapping)
        or backend_model.get("inventories_unchanged_after_generation")
        is not True
        or not isinstance(base, Mapping)
        or base.get("tree_sha256") != model["tree_sha256"]
        or not isinstance(adapter, Mapping)
        or adapter.get("tree_sha256")
        != checkpoint["evaluator_adapter_tree_sha256"]
    ):
        raise QLoRAV6Error(
            "canary checkpoint evaluator execution/model binding mismatch"
        )
    expected_implementation = {
        "evaluator": {
            "path": str(implementation["pointer_evaluator"].path),
            "sha256": implementation["pointer_evaluator"].sha256,
        },
        "compiler": {
            "path": str(implementation["pointer_compiler"].path),
            "sha256": implementation["pointer_compiler"].sha256,
        },
        "runner": {
            "path": str(implementation["checkpoint_runner"].path),
            "sha256": implementation["checkpoint_runner"].sha256,
        },
    }
    if receipt.get("implementation") != expected_implementation:
        raise QLoRAV6Error(
            "canary checkpoint evaluator implementation mismatch"
        )
    expected_bindings = {
        "base_model_tree_sha256": model["tree_sha256"],
        "adapter_tree_sha256": checkpoint[
            "evaluator_adapter_tree_sha256"
        ],
        "evaluator_source_sha256": implementation[
            "pointer_evaluator"
        ].sha256,
        "compiler_source_sha256": implementation[
            "pointer_compiler"
        ].sha256,
        "runner_source_sha256": implementation[
            "checkpoint_runner"
        ].sha256,
    }
    if receipt.get("bindings") != expected_bindings:
        raise QLoRAV6Error(
            "canary checkpoint evaluator binding mismatch"
        )
    if receipt.get("artifacts") != {
        "sample_results.v6.jsonl": sample_snapshot.sha256,
        "summary.v6.json": summary_snapshot.sha256,
    }:
        raise QLoRAV6Error(
            "canary checkpoint evaluator artifact binding mismatch"
        )
    summary = _strict_json_object_v7(
        summary_snapshot,
        label="canary checkpoint summary",
    )
    if receipt.get("claim_boundary") != summary.get("claim_boundary"):
        raise QLoRAV6Error(
            "canary checkpoint evaluator claim boundary mismatch"
        )


def _validate_canary_evaluation_contract_v7(
    *,
    acceptance: Mapping[str, Any],
    index: Mapping[str, Any],
    index_snapshot: StableFileSnapshotV7,
    training: Mapping[str, Any],
    training_snapshot: StableFileSnapshotV7,
    dataset: Mapping[str, Any],
    model: Mapping[str, Any],
    final_configuration: Mapping[str, Any],
) -> dict[str, Any]:
    from icmat_foundry.llm import (
        canary_acceptance_v6,
        pointer_checkpoint_eval_v6,
    )

    implementation = _strict_implementation_snapshots_v7()
    seed, training_specs = _validate_canary_training_contract_v7(
        training=training,
        training_snapshot=training_snapshot,
        dataset=dataset,
        model=model,
        final_configuration=final_configuration,
    )
    _require_exact_keys_v7(
        index,
        {
            "schema",
            "orchestrator_version",
            "created_at_utc",
            "status",
            "stage",
            "training",
            "dataset",
            "base_model",
            "execution",
            "implementation",
            "checkpoints",
            "records",
            "selection",
            "authorization",
            "claim_boundary",
        },
        label="canary evaluation index",
    )
    _require_timestamp_v7(
        index.get("created_at_utc"),
        label="canary evaluation index timestamp",
    )
    if (
        index.get("schema") != CANARY_EVALUATION_INDEX_SCHEMA
        or index.get("orchestrator_version")
        != pointer_checkpoint_eval_v6.ORCHESTRATOR_VERSION
        or index.get("status") != CANARY_EVALUATION_INDEX_STATUS
        or index.get("stage") != "canary"
    ):
        raise QLoRAV6Error(
            "canary evaluation index identity mismatch"
        )
    index_training = _require_exact_keys_v7(
        index.get("training"),
        {
            "receipt_path",
            "receipt_sha256",
            "run_id",
            "checkpoint_count",
        },
        label="canary evaluation training binding",
    )
    if (
        not _same_regular_path_v7(
            index_training.get("receipt_path"),
            training_snapshot.path,
            label="canary training receipt",
        )
        or index_training.get("receipt_sha256")
        != training_snapshot.sha256
        or index_training.get("run_id") != training.get("run_id")
        or index_training.get("checkpoint_count") != FIXED_EPOCHS
    ):
        raise QLoRAV6Error(
            "canary evaluation/training binding mismatch"
        )

    validation = dataset["splits"]["validation"]
    if validation.get("examples") != 150:
        raise QLoRAV6Error(
            "canary source validation must contain exactly 150 rows"
        )
    index_dataset = _require_exact_keys_v7(
        index.get("dataset"),
        {
            "directory",
            "path",
            "bytes",
            "sha256",
            "examples",
            "evaluation_directory",
            "evaluated_rows_per_checkpoint",
            "canary_selection",
            "calibration_content_read",
            "calibration_content_hashed",
            "blind_test_content_read",
            "blind_test_content_hashed",
        },
        label="canary evaluation dataset binding",
    )
    validation_path = Path(str(dataset["path"])) / "validation.jsonl"
    if (
        not _same_regular_path_v7(
            index_dataset.get("path"),
            validation_path,
            label="canary source validation",
        )
        or not _same_regular_path_v7(
            index_dataset.get("directory"),
            Path(str(dataset["path"])),
            label="canary source dataset",
        )
        or index_dataset.get("bytes") != validation.get("bytes")
        or index_dataset.get("sha256") != validation.get("sha256")
        or index_dataset.get("examples") != 150
        or index_dataset.get("evaluated_rows_per_checkpoint") != 18
        or any(
            index_dataset.get(field) is not False
            for field in (
                "calibration_content_read",
                "calibration_content_hashed",
                "blind_test_content_read",
                "blind_test_content_hashed",
            )
        )
    ):
        raise QLoRAV6Error(
            "canary evaluation dataset binding mismatch"
        )

    evaluation_root = index_snapshot.path.parent
    validation_view_dir = _strict_child_directory_v7(
        index_dataset.get("evaluation_directory"),
        root=evaluation_root,
        label="canary validation view directory",
    )
    try:
        view_names = frozenset(
            entry.name for entry in os.scandir(validation_view_dir)
        )
    except OSError as exc:
        raise QLoRAV6Error(
            "canary validation view cannot be enumerated"
        ) from exc
    if view_names != {
        "validation.jsonl",
        "canary_selection.v6.json",
    }:
        raise QLoRAV6Error(
            "canary validation view artifact inventory mismatch"
        )
    validation_view = _stable_snapshot_v7(
        validation_view_dir / "validation.jsonl",
        label="canary validation view",
        maximum_bytes=_STRICT_MAX_JSONL_BYTES,
    )
    selection_snapshot = _stable_snapshot_v7(
        validation_view_dir / "canary_selection.v6.json",
        label="canary selection receipt",
        maximum_bytes=_STRICT_MAX_JSON_BYTES,
    )
    selection_receipt = _strict_json_object_v7(
        selection_snapshot,
        label="canary selection receipt",
    )
    if selection_receipt != index_dataset.get("canary_selection"):
        raise QLoRAV6Error(
            "canary selection receipt/index binding mismatch"
        )
    _require_exact_keys_v7(
        selection_receipt,
        {
            "schema",
            "status",
            "algorithm",
            "source_validation",
            "view_validation",
            "selected",
            "calibration_content_read",
            "blind_test_content_read",
        },
        label="canary selection receipt",
    )
    source_selection = _require_exact_keys_v7(
        selection_receipt.get("source_validation"),
        {"path", "sha256", "rows"},
        label="canary source selection binding",
    )
    view_selection = _require_exact_keys_v7(
        selection_receipt.get("view_validation"),
        {"path", "sha256", "bytes", "rows"},
        label="canary view selection binding",
    )
    if (
        selection_receipt.get("schema")
        != pointer_checkpoint_eval_v6.CANARY_SELECTION_SCHEMA
        or selection_receipt.get("status")
        != "PASS_FIXED_STRATIFIED_18_SELECTED"
        or selection_receipt.get("algorithm")
        != (
            "lexicographically smallest example_id per fixed "
            "domain/task/decision stratum"
        )
        or not _same_regular_path_v7(
            source_selection.get("path"),
            validation_path,
            label="canary selected source validation",
        )
        or source_selection.get("sha256") != validation["sha256"]
        or source_selection.get("rows") != 150
        or not _same_regular_path_v7(
            view_selection.get("path"),
            validation_view.path,
            label="canary selected validation view",
        )
        or view_selection.get("sha256") != validation_view.sha256
        or view_selection.get("bytes") != validation_view.byte_count
        or view_selection.get("rows") != 18
        or selection_receipt.get("calibration_content_read") is not False
        or selection_receipt.get("blind_test_content_read") is not False
    ):
        raise QLoRAV6Error(
            "canary stratified selection contract mismatch"
        )
    view_rows = _strict_jsonl_rows_v7(
        validation_view,
        label="canary validation view",
    )
    selected = selection_receipt.get("selected")
    if (
        not isinstance(selected, list)
        or len(selected) != 18
        or len(view_rows) != 18
    ):
        raise QLoRAV6Error(
            "canary validation view must contain exactly 18 rows"
        )
    expected_ids: list[str] = []
    strata: set[tuple[str, str, str]] = set()
    for selected_item, row in zip(selected, view_rows, strict=True):
        selected_item = _require_exact_keys_v7(
            selected_item,
            {"domain", "task", "decision", "example_id"},
            label="canary selected row",
        )
        observed = (
            row.get("domain"),
            row.get("task"),
            row.get("decision"),
            row.get("example_id"),
        )
        expected = (
            selected_item.get("domain"),
            selected_item.get("task"),
            selected_item.get("decision"),
            selected_item.get("example_id"),
        )
        if observed != expected:
            raise QLoRAV6Error(
                "canary selected row differs from validation view"
            )
        stratum = (
            str(expected[0]),
            str(expected[1]),
            str(expected[2]),
        )
        if stratum in strata:
            raise QLoRAV6Error("canary validation strata are duplicated")
        strata.add(stratum)
        expected_ids.append(str(expected[3]))
    expected_strata = {
        (domain, task, decision)
        for domain in evidence_contract.DOMAINS
        for task in evidence_contract.TASKS
        for decision in evidence_contract.DECISIONS
    }
    if strata != expected_strata or len(set(expected_ids)) != 18:
        raise QLoRAV6Error(
            "canary validation view does not cover fixed 3x3x2 strata"
        )
    expected_ids.sort()

    index_model = _require_exact_keys_v7(
        index.get("base_model"),
        {
            "directory",
            "training_tree_sha256",
            "evaluator_tree_sha256",
            "file_count",
            "bytes",
        },
        label="canary evaluation base model",
    )
    if (
        not _same_regular_path_v7(
            index_model.get("directory"),
            Path(str(model["path"])),
            label="canary evaluation base model",
        )
        or index_model.get("training_tree_sha256")
        != model["tree_sha256"]
        or index_model.get("evaluator_tree_sha256")
        != model["tree_sha256"]
        or index_model.get("file_count") != model["file_count"]
        or index_model.get("bytes") != model["bytes"]
    ):
        raise QLoRAV6Error(
            "canary evaluation base-model binding mismatch"
        )
    expected_index_implementation = {
        "orchestrator": {
            "path": str(implementation["checkpoint_orchestrator"].path),
            "sha256": implementation["checkpoint_orchestrator"].sha256,
        },
        "pointer_evaluator": {
            "path": str(implementation["pointer_evaluator"].path),
            "sha256": implementation["pointer_evaluator"].sha256,
        },
        "pointer_compiler": {
            "path": str(implementation["pointer_compiler"].path),
            "sha256": implementation["pointer_compiler"].sha256,
        },
        "selection_policy": {
            "path": str(implementation["selection_policy"].path),
            "sha256": implementation["selection_policy"].sha256,
        },
        "runner": {
            "path": str(implementation["checkpoint_runner"].path),
            "sha256": implementation["checkpoint_runner"].sha256,
        },
    }
    if index.get("implementation") != expected_index_implementation:
        raise QLoRAV6Error(
            "canary evaluation implementation identity mismatch"
        )
    execution = _require_exact_keys_v7(
        index.get("execution"),
        {
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
        label="canary evaluation execution",
    )
    if (
        execution.get("backend") != "hf_model"
        or execution.get("runner_mode") != "production_fixed"
        or not isinstance(execution.get("device"), str)
        or not execution["device"]
        or execution.get("seed") != 20260729
        or execution.get("split") != "validation"
        or execution.get("max_samples") is not None
        or execution.get("checkpoint_outputs_immutable") is not True
        or execution.get("per_sample_metrics_recomputed") is not True
        or execution.get("summary_metrics_trusted") is not False
        or execution.get("selection_policy_invoked") is not False
        or execution.get("checkpoint_selected") is not False
        or execution.get("freeze_created") is not False
    ):
        raise QLoRAV6Error(
            "canary evaluation execution boundary mismatch"
        )
    if index.get("selection") != {
        "performed": False,
        "selected_checkpoint_id": None,
        "required_next_step": (
            "independent selection-policy evaluation and freeze"
        ),
    }:
        raise QLoRAV6Error(
            "canary evaluation must remain unselected"
        )
    if index.get("authorization") != _expected_run_authorization_v7():
        raise QLoRAV6Error(
            "canary evaluation authorization boundary mismatch"
        )
    if index.get("claim_boundary") != (
        "This index proves only immutable non-blind validation generation "
        "and independent per-sample metric recomputation for every retained "
        "v6 checkpoint. It does not select or authorize a model and does "
        "not access calibration or blind content."
    ):
        raise QLoRAV6Error(
            "canary evaluation claim boundary mismatch"
        )

    checkpoints = index.get("checkpoints")
    records = index.get("records")
    if (
        not isinstance(checkpoints, list)
        or len(checkpoints) != FIXED_EPOCHS
        or not isinstance(records, list)
        or len(records) != FIXED_EPOCHS
    ):
        raise QLoRAV6Error(
            "canary evaluation must contain exactly six checkpoints"
        )
    specs = {
        (int(spec["seed"]), int(spec["epoch"])): spec
        for spec in training_specs
    }
    records_by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        record = _require_exact_keys_v7(
            record,
            {
                "checkpoint_id",
                "seed",
                "epoch",
                "validation_loss",
                "metrics",
            },
            label="canary evaluation metric record",
        )
        checkpoint_id = record.get("checkpoint_id")
        if (
            not isinstance(checkpoint_id, str)
            or checkpoint_id in records_by_id
        ):
            raise QLoRAV6Error(
                "canary evaluation record IDs are invalid"
            )
        records_by_id[checkpoint_id] = record

    candidates: list[dict[str, Any]] = []
    artifacts_read: list[dict[str, Any]] = []
    observed_ids: list[str] | None = None
    observed_pairs: set[tuple[int, int]] = set()
    for checkpoint in checkpoints:
        checkpoint = _require_exact_keys_v7(
            checkpoint,
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
            },
            label="canary checkpoint evidence",
        )
        checkpoint_id = checkpoint.get("checkpoint_id")
        pair = (checkpoint.get("seed"), checkpoint.get("epoch"))
        if (
            not isinstance(pair[0], int)
            or isinstance(pair[0], bool)
            or not isinstance(pair[1], int)
            or isinstance(pair[1], bool)
            or pair in observed_pairs
            or pair not in specs
            or checkpoint_id not in records_by_id
        ):
            raise QLoRAV6Error(
                "canary checkpoint seed/epoch population mismatch"
            )
        observed_pairs.add(pair)
        spec = specs[pair]
        record = records_by_id[str(checkpoint_id)]
        if (
            checkpoint_id != spec["checkpoint_id"]
            or checkpoint.get("global_step") != spec["global_step"]
            or checkpoint.get("receipt_relative_path")
            != spec["receipt_path"]
            or checkpoint.get("training_checkpoint_tree_sha256")
            != spec["training_checkpoint_tree_sha256"]
            or checkpoint.get("training_adapter_tree_sha256")
            != spec["training_adapter_tree_sha256"]
            or checkpoint.get("evaluator_adapter_tree_sha256")
            != spec["evaluator_adapter_tree_sha256"]
            or checkpoint.get("checkpoint_files")
            != spec["checkpoint_files"]
            or checkpoint.get("checkpoint_bytes")
            != spec["checkpoint_bytes"]
            or not _same_regular_path_v7(
                checkpoint.get("checkpoint_path"),
                Path(spec["path"]),
                label=f"{checkpoint_id} training checkpoint",
            )
            or canary_acceptance_v6._parse_loss(
                checkpoint.get("validation_loss"),
                field=f"{checkpoint_id}.validation_loss",
            )
            != canary_acceptance_v6._parse_loss(
                spec["validation_loss"],
                field=f"{checkpoint_id}.training_validation_loss",
            )
            or record.get("seed") != pair[0]
            or record.get("epoch") != pair[1]
            or canary_acceptance_v6._parse_loss(
                record.get("validation_loss"),
                field=f"{checkpoint_id}.record_validation_loss",
            )
            != canary_acceptance_v6._parse_loss(
                spec["validation_loss"],
                field=f"{checkpoint_id}.training_validation_loss",
            )
        ):
            raise QLoRAV6Error(
                f"{checkpoint_id} training/evaluation binding mismatch"
            )
        directory = _strict_child_directory_v7(
            checkpoint.get("evaluation_directory"),
            root=evaluation_root,
            label=f"{checkpoint_id} evaluation directory",
        )
        try:
            artifact_names = frozenset(
                entry.name for entry in os.scandir(directory)
            )
        except OSError as exc:
            raise QLoRAV6Error(
                f"{checkpoint_id} artifacts cannot be enumerated"
            ) from exc
        if artifact_names != {
            "sample_results.v6.jsonl",
            "summary.v6.json",
            "run_receipt.v6.json",
        }:
            raise QLoRAV6Error(
                f"{checkpoint_id} artifact inventory mismatch"
            )
        sample_snapshot = _stable_snapshot_v7(
            directory / "sample_results.v6.jsonl",
            label=f"{checkpoint_id} sample results",
            maximum_bytes=_STRICT_MAX_JSONL_BYTES,
        )
        summary_snapshot = _stable_snapshot_v7(
            directory / "summary.v6.json",
            label=f"{checkpoint_id} summary",
            maximum_bytes=_STRICT_MAX_JSON_BYTES,
        )
        run_snapshot = _stable_snapshot_v7(
            directory / "run_receipt.v6.json",
            label=f"{checkpoint_id} run receipt",
            maximum_bytes=_STRICT_MAX_JSON_BYTES,
        )
        expected_artifacts = {
            "sample_results.v6.jsonl": sample_snapshot.sha256,
            "summary.v6.json": summary_snapshot.sha256,
            "run_receipt.v6.json": run_snapshot.sha256,
        }
        if checkpoint.get("evaluation_artifacts") != expected_artifacts:
            raise QLoRAV6Error(
                f"{checkpoint_id} evaluation artifact hashes mismatch"
            )
        _validate_canary_eval_run_receipt_v7(
            snapshot=run_snapshot,
            sample_snapshot=sample_snapshot,
            summary_snapshot=summary_snapshot,
            validation_view=validation_view,
            checkpoint=checkpoint,
            implementation=implementation,
            model=model,
        )
        rows = canary_acceptance_v6._load_jsonl_bytes(
            sample_snapshot.payload,
            field=f"{checkpoint_id} sample results",
        )
        metrics, audit = canary_acceptance_v6._recompute_checkpoint(
            rows,
            checkpoint_id=str(checkpoint_id),
        )
        current_ids = list(audit["example_ids"])
        if current_ids != expected_ids:
            raise QLoRAV6Error(
                f"{checkpoint_id} did not evaluate the frozen 18-row view"
            )
        if observed_ids is None:
            observed_ids = current_ids
        elif current_ids != observed_ids:
            raise QLoRAV6Error(
                "canary checkpoints evaluated different example IDs"
            )
        if record.get("metrics") != metrics:
            raise QLoRAV6Error(
                f"{checkpoint_id} index metrics differ from recomputation"
            )
        summary = _strict_json_object_v7(
            summary_snapshot,
            label=f"{checkpoint_id} summary",
        )
        try:
            canary_acceptance_v6._validate_summary(
                summary,
                metrics=metrics,
                audit=audit,
                checkpoint_id=str(checkpoint_id),
            )
        except Exception as exc:
            raise QLoRAV6Error(
                f"{checkpoint_id} summary differs from recomputation"
            ) from exc
        failed = canary_acceptance_v6._failed_gates(metrics)
        validation_loss = str(
            canary_acceptance_v6._parse_loss(
                spec["validation_loss"],
                field=f"{checkpoint_id}.validation_loss",
            )
        )
        candidates.append(
            {
                "checkpoint_id": checkpoint_id,
                "seed": pair[0],
                "epoch": pair[1],
                "validation_loss": validation_loss,
                "qualified": not failed,
                "failed_gates": failed,
                "metrics": metrics,
                "ranking_metrics": {
                    "minimum_stratified_strict": (
                        canary_acceptance_v6._minimum_stratum(metrics)
                    ),
                    "compiled_strict_exact": metrics[
                        "compiled_strict_exact"
                    ],
                    "answer_span_exact": metrics["answer_span_exact"],
                    "refuse_f1": canary_acceptance_v6._refuse_f1(metrics),
                    "validation_loss": validation_loss,
                    "epoch": pair[1],
                    "seed": pair[0],
                },
            }
        )
        for role, artifact_snapshot in (
            ("sample_results.v6.jsonl", sample_snapshot),
            ("summary.v6.json", summary_snapshot),
        ):
            artifacts_read.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "role": f"{checkpoint_id}:{role}",
                    "path": str(artifact_snapshot.path),
                    "bytes": artifact_snapshot.byte_count,
                    "sha256": artifact_snapshot.sha256,
                }
            )
    if observed_pairs != {
        (seed, epoch) for epoch in range(1, FIXED_EPOCHS + 1)
    }:
        raise QLoRAV6Error(
            "canary checkpoint population is not one seed across epochs 1..6"
        )
    candidates.sort(key=lambda item: str(item["checkpoint_id"]))
    artifacts_read.sort(
        key=lambda item: (
            str(item["checkpoint_id"]),
            str(item["role"]),
        )
    )
    qualified = [item for item in candidates if item["qualified"]]
    ordered = sorted(
        qualified,
        key=cmp_to_key(canary_acceptance_v6._compare_candidates),
    )
    if not ordered:
        raise QLoRAV6Error(
            "canary acceptance claims PASS without a qualified checkpoint"
        )
    reference_candidate = ordered[0]
    expected_reference = {
        "checkpoint_id": reference_candidate["checkpoint_id"],
        "seed": reference_candidate["seed"],
        "epoch": reference_candidate["epoch"],
        "ranking_metrics": reference_candidate["ranking_metrics"],
        "purpose": "THREE_SEED_TRAINING_ADVANCEMENT_EVIDENCE_ONLY",
        "is_final_model_selection": False,
    }
    expected_recomputation = {
        "checkpoint_count": FIXED_EPOCHS,
        "samples_per_checkpoint": 18,
        "summary_metrics_trusted": False,
        "index_metrics_trusted": False,
        "all_index_and_summary_metrics_reconciled": True,
        "checkpoints": candidates,
    }
    acceptance_input = acceptance.get("input")
    if (
        acceptance.get("thresholds") != _expected_canary_thresholds_v7()
        or acceptance.get("independent_recomputation")
        != expected_recomputation
        or acceptance.get("deterministic_advancement_reference")
        != expected_reference
        or not isinstance(acceptance_input, Mapping)
        or acceptance_input.get("checkpoint_artifacts_read")
        != artifacts_read
    ):
        raise QLoRAV6Error(
            "canary acceptance thresholds, ranking, or artifact "
            "recomputation mismatch"
        )
    return {
        "canary_seed": seed,
        "epoch_population": list(range(1, FIXED_EPOCHS + 1)),
        "checkpoint_artifacts_independently_verified": len(artifacts_read),
        "checkpoint_run_receipts_independently_verified": FIXED_EPOCHS,
        "thresholds_independently_verified": True,
        "ranking_independently_recomputed": True,
        "advancement_reference": expected_reference,
        "implementation_receipts": {
            role: {
                "path": str(snapshot.path),
                "bytes": snapshot.byte_count,
                "sha256": snapshot.sha256,
                "stable_identity": snapshot.identity_receipt(),
            }
            for role, snapshot in sorted(implementation.items())
        },
    }


def _validate_canary_acceptance_gate_v6(
    *,
    acceptance_receipt_path: Path,
    evaluation_index_path: Path,
    canary_training_receipt_path: Path,
    dataset: Mapping[str, Any],
    model: Mapping[str, Any],
    final_configuration: Mapping[str, Any],
) -> dict[str, Any]:
    acceptance_snapshot = _stable_snapshot_v7(
        acceptance_receipt_path,
        label="canary acceptance receipt",
        maximum_bytes=_STRICT_MAX_JSON_BYTES,
    )
    acceptance = _strict_json_object_v7(
        acceptance_snapshot,
        label="canary acceptance receipt",
    )
    expected_acceptance_keys = {
        "schema",
        "gate_version",
        "created_at_utc",
        "status",
        "gate_passed",
        "next_action",
        "input",
        "thresholds",
        "independent_recomputation",
        "deterministic_advancement_reference",
        "authorization",
        "claim_boundary",
        "receipt_payload_sha256",
    }
    if set(acceptance) != expected_acceptance_keys:
        raise QLoRAV6Error("canary acceptance receipt keys mismatch")
    receipt_digest = acceptance.get("receipt_payload_sha256")
    acceptance_core = {
        key: value
        for key, value in acceptance.items()
        if key != "receipt_payload_sha256"
    }
    if (
        not _valid_sha256(receipt_digest)
        or receipt_digest != _canonical_sha256(acceptance_core)
        or acceptance.get("schema") != CANARY_ACCEPTANCE_SCHEMA
        or acceptance.get("gate_version") != CANARY_ACCEPTANCE_VERSION
        or acceptance.get("status") != CANARY_ACCEPTANCE_STATUS
        or acceptance.get("gate_passed") is not True
        or acceptance.get("next_action")
        != "START_FINAL_THREE_SEED_TRAINING"
        or acceptance.get("claim_boundary")
        != CANARY_ACCEPTANCE_CLAIM_BOUNDARY
    ):
        raise QLoRAV6Error(
            "canary acceptance receipt did not authorize final training"
        )
    _require_timestamp_v7(
        acceptance.get("created_at_utc"),
        label="canary acceptance timestamp",
    )
    authorization = acceptance.get("authorization")
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
    if authorization != expected_authorization:
        raise QLoRAV6Error(
            "canary acceptance authorization boundary mismatch"
        )
    reference = acceptance.get("deterministic_advancement_reference")
    if (
        not isinstance(reference, Mapping)
        or set(reference)
        != {
            "checkpoint_id",
            "seed",
            "epoch",
            "ranking_metrics",
            "purpose",
            "is_final_model_selection",
        }
        or not isinstance(reference.get("checkpoint_id"), str)
        or not reference.get("checkpoint_id")
        or reference.get("purpose")
        != "THREE_SEED_TRAINING_ADVANCEMENT_EVIDENCE_ONLY"
        or reference.get("is_final_model_selection") is not False
    ):
        raise QLoRAV6Error(
            "canary advancement reference contract mismatch"
        )
    recomputation = acceptance.get("independent_recomputation")
    candidates = (
        recomputation.get("checkpoints")
        if isinstance(recomputation, Mapping)
        else None
    )
    if (
        not isinstance(recomputation, Mapping)
        or set(recomputation)
        != {
            "checkpoint_count",
            "samples_per_checkpoint",
            "summary_metrics_trusted",
            "index_metrics_trusted",
            "all_index_and_summary_metrics_reconciled",
            "checkpoints",
        }
        or recomputation.get("checkpoint_count") != 6
        or recomputation.get("samples_per_checkpoint") != 18
        or recomputation.get("summary_metrics_trusted") is not False
        or recomputation.get("index_metrics_trusted") is not False
        or recomputation.get(
            "all_index_and_summary_metrics_reconciled"
        )
        is not True
        or not isinstance(candidates, list)
        or len(candidates) != 6
        or not any(
            isinstance(candidate, Mapping)
            and candidate.get("checkpoint_id")
            == reference["checkpoint_id"]
            and candidate.get("qualified") is True
            for candidate in candidates
        )
    ):
        raise QLoRAV6Error(
            "canary acceptance independent recomputation mismatch"
        )

    index_snapshot = _stable_snapshot_v7(
        evaluation_index_path,
        label="canary evaluation index",
        maximum_bytes=_STRICT_MAX_JSON_BYTES,
    )
    if index_snapshot.path.name != "evaluation_index.v6.json":
        raise QLoRAV6Error(
            "canary evaluation index filename mismatch"
        )
    acceptance_input = acceptance.get("input")
    index_binding = (
        acceptance_input.get("evaluation_index")
        if isinstance(acceptance_input, Mapping)
        else None
    )
    artifacts_read = (
        acceptance_input.get("checkpoint_artifacts_read")
        if isinstance(acceptance_input, Mapping)
        else None
    )
    if (
        not isinstance(acceptance_input, Mapping)
        or set(acceptance_input)
        != {
            "evaluation_index",
            "checkpoint_artifacts_read",
            "checkpoint_run_receipts_read",
            "training_receipt_read",
            "calibration_content_read",
            "calibration_content_hashed",
            "blind_test_content_read",
            "blind_test_content_hashed",
        }
        or index_binding
        != {
            "path": str(index_snapshot.path),
            "bytes": index_snapshot.byte_count,
            "sha256": index_snapshot.sha256,
        }
        or not isinstance(artifacts_read, list)
        or len(artifacts_read) != 12
        or acceptance_input.get("checkpoint_run_receipts_read") is not False
        or acceptance_input.get("training_receipt_read") is not False
        or any(
            acceptance_input.get(field) is not False
            for field in (
                "calibration_content_read",
                "calibration_content_hashed",
                "blind_test_content_read",
                "blind_test_content_hashed",
            )
        )
    ):
        raise QLoRAV6Error("canary acceptance input binding mismatch")
    index = _strict_json_object_v7(
        index_snapshot,
        label="canary evaluation index",
    )
    if (
        index.get("schema") != CANARY_EVALUATION_INDEX_SCHEMA
        or index.get("status") != CANARY_EVALUATION_INDEX_STATUS
        or index.get("stage") != "canary"
    ):
        raise QLoRAV6Error(
            "canary evaluation index is not an accepted 1x6 run"
        )

    training_snapshot = _stable_snapshot_v7(
        canary_training_receipt_path,
        label="canary training receipt",
        maximum_bytes=_STRICT_MAX_JSON_BYTES,
    )
    if training_snapshot.path.name != "training_receipt.v6.json":
        raise QLoRAV6Error("canary training receipt filename mismatch")
    canary_training = _strict_json_object_v7(
        training_snapshot,
        label="canary training receipt",
    )
    if (
        canary_training.get("schema") != RUN_RECEIPT_SCHEMA
        or canary_training.get("trainer_version") != TRAINER_VERSION
        or canary_training.get("stage") != "canary"
        or canary_training.get("status")
        != "PASS_CANARY_SINGLE_SEED_ALL_EPOCHS_NOT_SELECTED"
        or canary_training.get("checkpoint_count") != 6
    ):
        raise QLoRAV6Error(
            "canary training receipt is not a completed 1x6 run"
        )
    index_training = index.get("training")
    if (
        not isinstance(index_training, Mapping)
        or index_training.get("receipt_path")
        != str(training_snapshot.path)
        or index_training.get("receipt_sha256")
        != training_snapshot.sha256
        or index_training.get("run_id") != canary_training.get("run_id")
        or index_training.get("checkpoint_count") != 6
    ):
        raise QLoRAV6Error(
            "canary evaluation/training receipt binding mismatch"
        )

    input_snapshot = canary_training.get("input_snapshot")
    canary_dataset = (
        input_snapshot.get("dataset")
        if isinstance(input_snapshot, Mapping)
        else None
    )
    canary_model = (
        input_snapshot.get("base_model")
        if isinstance(input_snapshot, Mapping)
        else None
    )
    if (
        not isinstance(canary_dataset, Mapping)
        or canary_dataset.get("path") != dataset.get("path")
        or canary_dataset.get("inspected_input_sha256")
        != dataset.get("inspected_input_sha256")
        or not isinstance(canary_model, Mapping)
        or canary_model.get("path") != model.get("path")
        or canary_model.get("tree_sha256") != model.get("tree_sha256")
    ):
        raise QLoRAV6Error(
            "canary training dataset/base binding mismatch"
        )
    validation = dataset["splits"]["validation"]
    index_dataset = index.get("dataset")
    expected_validation_path = str(
        Path(str(dataset["path"])) / "validation.jsonl"
    )
    if (
        not isinstance(index_dataset, Mapping)
        or index_dataset.get("directory") != dataset.get("path")
        or index_dataset.get("path") != expected_validation_path
        or index_dataset.get("bytes") != validation.get("bytes")
        or index_dataset.get("sha256") != validation.get("sha256")
        or index_dataset.get("examples") != validation.get("examples")
        or index_dataset.get("evaluated_rows_per_checkpoint") != 18
    ):
        raise QLoRAV6Error(
            "canary evaluation dataset binding mismatch"
        )
    index_model = index.get("base_model")
    if (
        not isinstance(index_model, Mapping)
        or index_model.get("directory") != model.get("path")
        or index_model.get("training_tree_sha256")
        != model.get("tree_sha256")
        or index_model.get("file_count") != model.get("file_count")
        or index_model.get("bytes") != model.get("bytes")
    ):
        raise QLoRAV6Error(
            "canary evaluation base-model binding mismatch"
        )
    independent_evidence = _validate_canary_evaluation_contract_v7(
        acceptance=acceptance,
        index=index,
        index_snapshot=index_snapshot,
        training=canary_training,
        training_snapshot=training_snapshot,
        dataset=dataset,
        model=model,
        final_configuration=final_configuration,
    )
    return {
        "required_for_stage": "final",
        "path": str(acceptance_snapshot.path),
        "bytes": acceptance_snapshot.byte_count,
        "sha256": acceptance_snapshot.sha256,
        "stable_identity": acceptance_snapshot.identity_receipt(),
        "schema": CANARY_ACCEPTANCE_SCHEMA,
        "gate_version": CANARY_ACCEPTANCE_VERSION,
        "status": CANARY_ACCEPTANCE_STATUS,
        "gate_passed": True,
        "next_action": "START_FINAL_THREE_SEED_TRAINING",
        "receipt_payload_sha256": receipt_digest,
        "authorization": expected_authorization,
        "claim_boundary": CANARY_ACCEPTANCE_CLAIM_BOUNDARY,
        "evaluation_index": {
            "path": str(index_snapshot.path),
            "bytes": index_snapshot.byte_count,
            "sha256": index_snapshot.sha256,
            "stable_identity": index_snapshot.identity_receipt(),
        },
        "canary_training_receipt": {
            "path": str(training_snapshot.path),
            "bytes": training_snapshot.byte_count,
            "sha256": training_snapshot.sha256,
            "stable_identity": training_snapshot.identity_receipt(),
            "run_id": canary_training.get("run_id"),
        },
        "dataset_binding": {
            "path": dataset["path"],
            "inspected_input_sha256": dataset["inspected_input_sha256"],
        },
        "base_model_binding": {
            "path": model["path"],
            "tree_sha256": model["tree_sha256"],
            "file_count": model["file_count"],
            "bytes": model["bytes"],
        },
        "independent_contract_validation": independent_evidence,
    }


def _configuration_payload(
    config: QLoRATrainingConfigV6,
) -> dict[str, Any]:
    payload = asdict(config)
    payload["seeds"] = list(config.resolved_seeds)
    payload.update(
        {
            "num_train_epochs": FIXED_EPOCHS,
            "model_family": "Qwen2.5-0.5B-Instruct",
            "quantization": "NF4",
            "double_quantization": True,
            "compute_dtype": "bfloat16",
            "optimizer": "paged_adamw_8bit",
            "gradient_checkpointing": True,
            "assistant_only_loss": True,
            "target_modules": list(DEFAULT_TARGET_MODULES),
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
    )
    return payload


def _configuration_payload_v8c2(
    config: QLoRATrainingConfigV6,
) -> dict[str, Any]:
    payload = _configuration_payload(config)
    payload.update(
        {
            "training_profile": V8C2_TRAINING_PROFILE,
            "assistant_only_cross_entropy": True,
            "lora_alpha_over_rank": 2.0,
            "data_resampling": False,
            "data_augmentation": False,
            "class_or_task_weighting": False,
            "layer_freezing": False,
            "checkpoint_interpolation": False,
            "checkpoint_voting": False,
            "nli_answer_override": False,
            "inference_contract_changed": False,
        }
    )
    observed_algorithm = {
        "profile": payload.get("training_profile"),
        "assistant_only_cross_entropy": payload.get(
            "assistant_only_cross_entropy"
        ),
        "lora_rank": payload.get("lora_rank"),
        "lora_alpha": payload.get("lora_alpha"),
        "lora_alpha_over_rank": payload.get("lora_alpha_over_rank"),
        "lora_dropout": payload.get("lora_dropout"),
        "learning_rate": payload.get("learning_rate"),
        "num_train_epochs": payload.get("num_train_epochs"),
        "max_seq_length": payload.get("max_seq_length"),
        "per_device_train_batch_size": payload.get(
            "per_device_train_batch_size"
        ),
        "per_device_eval_batch_size": payload.get(
            "per_device_eval_batch_size"
        ),
        "gradient_accumulation_steps": payload.get(
            "gradient_accumulation_steps"
        ),
        "warmup_ratio": payload.get("warmup_ratio"),
        "weight_decay": payload.get("weight_decay"),
        "target_modules": payload.get("target_modules"),
        "data_resampling": payload.get("data_resampling"),
        "data_augmentation": payload.get("data_augmentation"),
        "class_or_task_weighting": payload.get("class_or_task_weighting"),
        "layer_freezing": payload.get("layer_freezing"),
        "checkpoint_interpolation": payload.get(
            "checkpoint_interpolation"
        ),
        "checkpoint_voting": payload.get("checkpoint_voting"),
        "nli_answer_override": payload.get("nli_answer_override"),
        "inference_contract_changed": payload.get(
            "inference_contract_changed"
        ),
    }
    expected_seeds = (
        list(CANARY_DEFAULT_SEEDS)
        if config.stage == "canary"
        else list(FINAL_DEFAULT_SEEDS)
    )
    if (
        observed_algorithm != _V8C2_FIXED_ALGORITHM
        or payload.get("stage") != config.stage
        or payload.get("seeds") != expected_seeds
        or payload.get("save_every_epoch") is not True
        or payload.get("save_total_limit") is not None
        or payload.get("load_best_model_at_end") is not False
        or payload.get("early_stopping") is not False
        or payload.get("automatic_checkpoint_selection") is not False
    ):
        raise QLoRAV6Error("v8c2 effective configuration contract mismatch")
    return payload


def _validate_canary_acceptance_gate_v8(
    *,
    acceptance_receipt_path: Path,
    evaluation_index_path: Path,
    canary_training_receipt_path: Path,
    dataset: Mapping[str, Any],
    model: Mapping[str, Any],
    final_configuration: Mapping[str, Any],
) -> dict[str, Any]:
    from icmat_foundry.llm import canary_acceptance_v8

    try:
        normalized = canary_acceptance_v8.verify_canary_acceptance_v8(
            Path(acceptance_receipt_path),
            Path(evaluation_index_path),
            Path(canary_training_receipt_path),
        )
    except canary_acceptance_v8.CanaryAcceptanceV8Error as exc:
        raise QLoRAV6Error(
            f"native v8 canary acceptance rejected: {exc}"
        ) from exc
    training_snapshot = _stable_snapshot_v7(
        Path(canary_training_receipt_path),
        label="v8 canary training receipt",
        maximum_bytes=_STRICT_MAX_JSON_BYTES,
    )
    training = _strict_json_object_v7(
        training_snapshot,
        label="v8 canary training receipt",
    )
    _validate_canary_training_generation_binding_v8(
        training=training,
        dataset=dataset,
    )
    configuration = training.get("configuration")
    if not isinstance(configuration, Mapping):
        raise QLoRAV6Error(
            "v8c2 canary training configuration is missing"
        )
    seeds = configuration.get("seeds")
    if (
        not isinstance(seeds, list)
        or seeds != list(CANARY_DEFAULT_SEEDS)
    ):
        raise QLoRAV6Error("v8c2 canary seed contract mismatch")
    expected_configuration = dict(final_configuration)
    expected_configuration["stage"] = "canary"
    expected_configuration["seeds"] = list(seeds)
    if (
        dict(configuration) != expected_configuration
        or training.get("configuration_sha256")
        != _canonical_sha256(configuration)
    ):
        raise QLoRAV6Error(
            "v8c2 canary and final configurations differ"
        )
    input_snapshot = training.get("input_snapshot")
    source_files = (
        input_snapshot.get("source_files")
        if isinstance(input_snapshot, Mapping)
        else None
    )
    if source_files != _source_inventory(
        strict_nonblind=True,
        strict_v8=True,
    ):
        raise QLoRAV6Error(
            "v8c2 canary source identity cannot authorize this final"
        )
    canary_model = (
        input_snapshot.get("base_model")
        if isinstance(input_snapshot, Mapping)
        else None
    )
    canary_dataset = (
        input_snapshot.get("dataset")
        if isinstance(input_snapshot, Mapping)
        else None
    )
    attempt = training.get("canary_attempt")
    if (
        not isinstance(canary_model, Mapping)
        or not isinstance(canary_dataset, Mapping)
        or not isinstance(attempt, Mapping)
        or canary_model.get("tree_sha256") != model.get("tree_sha256")
    ):
        raise QLoRAV6Error("v8c2 canary attempt binding is missing")
    _validate_v8c2_canary_attempt_receipt(
        attempt,
        run_id=str(training.get("run_id")),
        configuration_sha256=str(training.get("configuration_sha256")),
        dataset_input_sha256=str(
            canary_dataset.get("inspected_input_sha256")
        ),
        training_gate_bundle_sha256=str(
            training.get("training_gate_bundle_sha256")
        ),
        source_inventory_sha256=_canonical_sha256(source_files),
        base_model_tree_sha256=str(canary_model.get("tree_sha256")),
    )
    seed_receipts = training.get("seeds")
    if not isinstance(seed_receipts, list) or len(seed_receipts) != 1:
        raise QLoRAV6Error("v8c2 canary seed receipt is missing")
    seed_receipt = seed_receipts[0]
    if (
        not isinstance(seed_receipt, Mapping)
        or any(
            seed_receipt.get(key) != value
            for key, value in _v8c2_receipt_fields().items()
        )
        or seed_receipt.get("configuration") != configuration
    ):
        raise QLoRAV6Error(
            "v8c2 canary seed profile binding mismatch"
        )
    manifest = dataset.get("manifest")
    splits = dataset.get("splits")
    formal = normalized.get("formal_v8_binding")
    if (
        not isinstance(manifest, Mapping)
        or not isinstance(splits, Mapping)
        or not isinstance(splits.get("train"), Mapping)
        or not isinstance(splits.get("validation"), Mapping)
        or not isinstance(formal, Mapping)
        or formal.get("contract") != "STRICT_NONBLIND_V8"
        or formal.get("manifest_sha256") != manifest.get("sha256")
        or formal.get("train_sha256")
        != splits["train"].get("sha256")
        or formal.get("validation_sha256")
        != splits["validation"].get("sha256")
        or formal.get("training_gate_bundle_sha256")
        != dataset.get("training_gate_bundle_sha256")
        or formal.get("v8_inspected_input_sha256")
        != dataset.get("v8_inspected_input_sha256")
    ):
        raise QLoRAV6Error(
            "native v8 canary acceptance generation mismatch"
        )
    del model
    return normalized


def preflight_v6_contract(
    *,
    dataset_dir: Path,
    model_dir: Path | None = None,
    config: QLoRATrainingConfigV6 | None = None,
    nonblind_second_build_dir: Path | None = None,
    nonblind_audit_receipt: Path | None = None,
    train_shortcut_audit: Path | None = None,
    validation_shortcut_audit: Path | None = None,
    scoped_lexical_audit_v8: Path | None = None,
    train_unique_support_audit_v8: Path | None = None,
    validation_unique_support_audit_v8: Path | None = None,
    unique_support_nli_model_dir: Path | None = None,
    canary_acceptance_receipt: Path | None = None,
    canary_evaluation_index: Path | None = None,
    canary_training_receipt: Path | None = None,
) -> dict[str, Any]:
    """Validate the immutable training contract before QLoRA starts."""

    config = QLoRATrainingConfigV6() if config is None else config
    config.validate()
    dataset = _dataset_snapshot(
        Path(dataset_dir),
        nonblind_second_build_dir=nonblind_second_build_dir,
        nonblind_audit_receipt=nonblind_audit_receipt,
        train_shortcut_audit=train_shortcut_audit,
        validation_shortcut_audit=validation_shortcut_audit,
        scoped_lexical_audit_v8=scoped_lexical_audit_v8,
        train_unique_support_audit_v8=(
            train_unique_support_audit_v8
        ),
        validation_unique_support_audit_v8=(
            validation_unique_support_audit_v8
        ),
        unique_support_nli_model_dir=unique_support_nli_model_dir,
    )
    dataset_kind = _strict_dataset_kind(dataset)
    strict_v8 = dataset_kind == "v8"
    effective_config = (
        _effective_training_config_v8c2(config)
        if strict_v8
        else config
    )
    if model_dir is None:
        model: dict[str, Any] = {
            "provided": False,
            "tree_hashed": False,
        }
    else:
        model = _model_snapshot(Path(model_dir))
    configuration = (
        _configuration_payload_v8c2(effective_config)
        if strict_v8
        else _configuration_payload(effective_config)
    )
    canary_paths = {
        "canary_acceptance_receipt": canary_acceptance_receipt,
        "canary_evaluation_index": canary_evaluation_index,
        "canary_training_receipt": canary_training_receipt,
    }
    supplied_canary_paths = {
        name for name, path in canary_paths.items() if path is not None
    }
    if supplied_canary_paths:
        if supplied_canary_paths != set(canary_paths):
            raise QLoRAV6Error(
                "canary acceptance validation requires all explicit paths"
            )
        if config.stage != "final":
            raise QLoRAV6Error(
                "canary acceptance gate applies only to final stage"
            )
        if model_dir is None:
            raise QLoRAV6Error(
                "canary acceptance binding requires the base model"
            )
        if strict_v8:
            canary_acceptance = _validate_canary_acceptance_gate_v8(
                acceptance_receipt_path=Path(canary_acceptance_receipt),
                evaluation_index_path=Path(canary_evaluation_index),
                canary_training_receipt_path=Path(canary_training_receipt),
                dataset=dataset,
                model=model,
                final_configuration=configuration,
            )
        else:
            canary_acceptance = _validate_canary_acceptance_gate_v6(
                acceptance_receipt_path=Path(canary_acceptance_receipt),
                evaluation_index_path=Path(canary_evaluation_index),
                canary_training_receipt_path=Path(canary_training_receipt),
                dataset=dataset,
                model=model,
                final_configuration=configuration,
            )
    else:
        canary_acceptance = {
            "required_for_final_training": config.stage == "final",
            "provided": False,
            "validated": False,
        }
    strict_v7 = dataset_kind == "v7"
    strict_nonblind = dataset_kind in {"v7", "v8"}
    if strict_v8:
        status = "PASS_NONBLIND_V8_READ_ONLY_PREFLIGHT_NOT_TRAINED"
    elif strict_v7:
        status = "PASS_NONBLIND_V7_READ_ONLY_PREFLIGHT_NOT_TRAINED"
    else:
        status = "PASS_V6_READ_ONLY_PREFLIGHT_NOT_TRAINED"
    if strict_v8:
        claim_boundary = (
            "This CPU preflight validates the strict nonblind-v8 fixed "
            "twelve-file primary and independent secondary builds, the "
            "current independent dataset parser, the compare receipt, the "
            "locally recomputed train/validation scoped lexical gate, and "
            "both split-specific unique-support gates using one fixed local "
            "CPU NLI load. Calibration was opened only as a fixed-inventory "
            "integrity artifact and was not loaded for training or selection. "
            "No blind artifact was materialized, discovered, named, "
            "constructed, opened, statted, read or hashed."
        )
    elif strict_v7:
        claim_boundary = (
        "This CPU-only preflight validates the strict nonblind-v7 "
        "manifest, fixed 250/150/150 split declarations, train and "
        "validation pointer targets, all six declared nonblind audit and "
        "commitment artifacts, two independently materialized fixed ten-file "
        "builds, the compare-mode receipt as corroboration, and locally "
        "recomputed split-specific shortcut audits. Calibration was opened, "
        "read, parsed and hashed only for fixed-inventory reproducibility; it "
        "was not loaded for training or selection. No blind artifact was "
        "materialized, discovered, named, constructed, opened, statted, read "
        "or hashed."
        )
    else:
        claim_boundary = (
            "This CPU-only preflight validates the v6 manifest, train and "
            "validation pointer targets, the required semantic-v7 audit "
            "binding when present, plus the optional frozen Qwen2.5-0.5B "
            "model tree. Calibration and sealed blind contents were not "
            "opened, parsed, hashed, statted or used. Their values are "
            "unverified declarations copied from manifest.v6.json."
        )
    canary_attempt_availability: dict[str, Any] | None = None
    if strict_v8:
        canary_attempt_availability = (
            _v8c2_canary_attempt_availability(
                required_for_stage=config.stage == "canary"
            )
        )
        if (
            config.stage == "canary"
            and canary_attempt_availability["available"] is not True
        ):
            raise QLoRAV6Error(
                "v8c2 canary preflight blocked by one-attempt protocol: "
                + str(canary_attempt_availability["state"])
            )
    receipt = {
        "schema": PREFLIGHT_SCHEMA,
        "trainer_version": TRAINER_VERSION,
        "created_at": _utc_now(),
        "status": status,
        "read_only": True,
        "gpu_required": False,
        "ml_runtime_imported": strict_v8,
        "network_used": False,
        "dataset": dataset,
        "base_model": model,
        "canary_acceptance": canary_acceptance,
        "configuration": configuration,
        "configuration_sha256": _canonical_sha256(configuration),
        "authorization": {
            "training_started": False,
            "checkpoint_selected": False,
            "model_authorized": False,
            "calibration_authorized": False,
            "blind_test_authorized": False,
            "deployment_authorized": False,
        },
        **(
            {
                "preblind_boundary": {
                    "blind_materialized": False,
                    "blind_discovered": False,
                    "blind_path_constructed": False,
                    "blind_filesystem_metadata_accessed": False,
                    "blind_content_opened": False,
                    "blind_content_read": False,
                    "blind_content_hashed": False,
                }
            }
            if strict_nonblind
            else {}
        ),
        "claim_boundary": claim_boundary,
    }
    if strict_v8:
        receipt["training_gate_bundle_sha256"] = dataset[
            "training_gate_bundle_sha256"
        ]
        receipt["v8_inspected_input_sha256"] = dataset[
            "v8_inspected_input_sha256"
        ]
        receipt.update(_v8c2_receipt_fields())
        if canary_attempt_availability is None:
            raise RuntimeError("v8c2 attempt availability was not checked")
        receipt["canary_attempt"] = canary_attempt_availability
    return receipt


def build_assistant_only_labels(
    prefix_ids: Sequence[int],
    full_ids: Sequence[int],
) -> list[int]:
    prefix = list(prefix_ids)
    full = list(full_ids)
    if not prefix or len(prefix) >= len(full):
        raise QLoRAV6Error("assistant target must add at least one token")
    if full[: len(prefix)] != prefix:
        raise QLoRAV6Error(
            "chat-template prefix is not a prefix of the full conversation"
        )
    return [-100] * len(prefix) + full[len(prefix) :]


def encode_assistant_only(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    max_seq_length: int,
) -> dict[str, Any]:
    _validate_messages(list(messages), source="tokenization")
    prefix_ids = tokenizer.apply_chat_template(
        list(messages[:-1]),
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=False,
    )
    if len(full_ids) > max_seq_length:
        raise QLoRAV6Error(
            f"conversation has {len(full_ids)} tokens, above "
            f"max_seq_length={max_seq_length}"
        )
    labels = build_assistant_only_labels(prefix_ids, full_ids)
    assistant_tokens = sum(token != -100 for token in labels)
    return {
        "input_ids": list(full_ids),
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "sequence_tokens": len(full_ids),
        "assistant_tokens": assistant_tokens,
    }


def _load_training_rows(
    dataset_dir: Path,
    split: str,
    *,
    expected: Mapping[str, Any],
    strict_nonblind: bool = False,
) -> list[dict[str, Any]]:
    if split not in READABLE_SPLITS:
        raise PermissionError(f"trainer is forbidden from loading split: {split}")
    filename = (
        NONBLIND_SPLIT_FILES[split]
        if strict_nonblind
        else SPLIT_FILES[split]
    )
    path = dataset_dir / filename
    if strict_nonblind:
        snapshot = _stable_snapshot_v7(
            path,
            label=f"{split} training reload",
            maximum_bytes=_STRICT_MAX_JSONL_BYTES,
        )
        if (
            snapshot.byte_count != expected.get("bytes")
            or snapshot.sha256 != expected.get("sha256")
            or snapshot.identity_receipt()
            != expected.get("stable_identity")
        ):
            raise PermissionError(
                f"{split} changed between preflight and training load"
            )
        rows = _strict_jsonl_rows_v7(
            snapshot,
            label=f"{split} training reload",
        )
        if len(rows) != expected.get("examples"):
            raise PermissionError(
                f"{split} changed between preflight and training load"
            )
        observed_ids: set[str] = set()
        for line_number, row in enumerate(rows, start=1):
            try:
                evidence_contract.validate_example(row)
            except evidence_contract.EvidenceSFTV6Error as exc:
                raise QLoRAV6Error(
                    f"{split} training reload:{line_number}: strict "
                    f"Evidence v6 contract rejected the row: {exc}"
                ) from exc
            example_id = row.get("example_id")
            if (
                row.get("split") != split
                or not isinstance(example_id, str)
                or not example_id
                or example_id in observed_ids
            ):
                raise QLoRAV6Error(
                    f"{split} training reload:{line_number}: "
                    "split or example identity mismatch"
                )
            observed_ids.add(example_id)
        return rows

    digest = hashlib.sha256()
    byte_count = 0
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            byte_count += len(raw_line)
            rows.append(json.loads(raw_line.decode("utf-8")))
    if not rows:
        raise QLoRAV6Error(f"{split} split is empty")
    if (
        byte_count != expected["bytes"]
        or digest.hexdigest() != expected["sha256"]
        or len(rows) != expected["examples"]
    ):
        raise PermissionError(
            f"{split} changed between preflight and training load"
        )
    return rows


def _sanitize_metrics(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize_metrics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_metrics(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return _sanitize_metrics(value.item())
    return str(value)


def _epoch_history(
    log_history: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    train_logs: list[dict[str, Any]] = []
    eval_logs: list[dict[str, Any]] = []
    for record in log_history:
        if "loss" in record and "epoch" in record and "eval_loss" not in record:
            train_logs.append(dict(record))
        if "eval_loss" in record and "epoch" in record:
            eval_logs.append(dict(record))
    results: list[dict[str, Any]] = []
    for expected_epoch in range(1, FIXED_EPOCHS + 1):
        evaluations = [
            record
            for record in eval_logs
            if abs(float(record["epoch"]) - expected_epoch) < 1e-4
        ]
        trainings = [
            record
            for record in train_logs
            if abs(float(record["epoch"]) - expected_epoch) < 1e-4
        ]
        if len(evaluations) != 1 or len(trainings) != 1:
            raise RuntimeError(
                f"trainer must emit exactly one train/eval loss for epoch "
                f"{expected_epoch}"
            )
        evaluation = evaluations[0]
        training = trainings[0]
        results.append(
            {
                "epoch": expected_epoch,
                "global_step": int(evaluation.get("step", 0)),
                "train_loss": float(training["loss"]),
                "validation_loss": float(evaluation["eval_loss"]),
                "learning_rate": training.get("learning_rate"),
                "validation_runtime_seconds": evaluation.get("eval_runtime"),
            }
        )
    return results


def _checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-([1-9][0-9]*)", path.name)
    if not match:
        raise QLoRAV6Error(f"invalid checkpoint directory name: {path.name}")
    return int(match.group(1))


def _checkpoint_receipts(
    seed_dir: Path,
    epoch_history: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    trainer_dir = seed_dir / "trainer"
    checkpoint_dirs = sorted(
        (
            path
            for path in trainer_dir.glob("checkpoint-*")
            if path.is_dir()
        ),
        key=_checkpoint_step,
    )
    if len(checkpoint_dirs) != FIXED_EPOCHS:
        raise RuntimeError(
            f"expected {FIXED_EPOCHS} epoch checkpoints, found "
            f"{len(checkpoint_dirs)}"
        )
    by_step = {
        int(record["global_step"]): dict(record) for record in epoch_history
    }
    if len(by_step) != FIXED_EPOCHS or 0 in by_step:
        raise RuntimeError("per-epoch global steps must be unique and positive")
    receipts: list[dict[str, Any]] = []
    observed_epochs: set[int] = set()
    for checkpoint_dir in checkpoint_dirs:
        state_path = checkpoint_dir / "trainer_state.json"
        if state_path.is_symlink() or not state_path.is_file():
            raise RuntimeError(
                f"{checkpoint_dir.name} is missing trainer_state.json"
            )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state_epoch = state.get("epoch")
        global_step = state.get("global_step")
        if (
            isinstance(state_epoch, bool)
            or not isinstance(state_epoch, (int, float))
            or isinstance(global_step, bool)
            or not isinstance(global_step, int)
        ):
            raise RuntimeError("checkpoint trainer state is incomplete")
        epoch = round(float(state_epoch))
        if abs(float(state_epoch) - epoch) >= 1e-4:
            raise RuntimeError("checkpoint was not saved at an epoch boundary")
        if epoch not in range(1, FIXED_EPOCHS + 1):
            raise RuntimeError("checkpoint epoch is outside the fixed schedule")
        if epoch in observed_epochs:
            raise RuntimeError("duplicate epoch checkpoint")
        observed_epochs.add(epoch)
        if _checkpoint_step(checkpoint_dir) != global_step:
            raise RuntimeError("checkpoint name/global_step mismatch")
        metrics = by_step.get(global_step)
        if metrics is None or metrics["epoch"] != epoch:
            raise RuntimeError("checkpoint does not bind one epoch loss record")
        checkpoint_inventory = _tree_inventory(checkpoint_dir)
        adapter_inventory = _selected_inventory(
            checkpoint_dir,
            filenames=frozenset(
                {
                    "adapter_config.json",
                    "adapter_model.safetensors",
                    "adapter_model.bin",
                }
            ),
        )
        receipts.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "path": checkpoint_dir.relative_to(seed_dir).as_posix(),
                "train_loss": metrics["train_loss"],
                "validation_loss": metrics["validation_loss"],
                "checkpoint": checkpoint_inventory,
                "adapter": adapter_inventory,
                "authorization": "EVIDENCE_ONLY_NOT_SELECTED_NOT_AUTHORIZED",
            }
        )
    receipts.sort(key=lambda record: int(record["epoch"]))
    if [record["epoch"] for record in receipts] != list(
        range(1, FIXED_EPOCHS + 1)
    ):
        raise RuntimeError("one or more epoch checkpoints are missing")
    return receipts


def _encode_rows(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    max_seq_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    encoded = [
        encode_assistant_only(
            tokenizer,
            row["messages"],
            max_seq_length=max_seq_length,
        )
        for row in rows
    ]
    return encoded, {
        "examples": len(encoded),
        "sequence_tokens_min": min(row["sequence_tokens"] for row in encoded),
        "sequence_tokens_mean": mean(row["sequence_tokens"] for row in encoded),
        "sequence_tokens_max": max(row["sequence_tokens"] for row in encoded),
        "assistant_tokens_mean": mean(row["assistant_tokens"] for row in encoded),
        "assistant_supervised_fraction": (
            sum(row["assistant_tokens"] for row in encoded)
            / sum(row["sequence_tokens"] for row in encoded)
        ),
    }


def _training_arguments_payload(
    *,
    trainer_dir: Path,
    seed: int,
    config: QLoRATrainingConfigV6,
) -> dict[str, Any]:
    return {
        "output_dir": str(trainer_dir),
        "overwrite_output_dir": False,
        "do_train": True,
        "do_eval": True,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "load_best_model_at_end": False,
        "save_total_limit": None,
        "num_train_epochs": float(FIXED_EPOCHS),
        "max_steps": -1,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "lr_scheduler_type": "cosine",
        "report_to": [],
        "seed": seed,
        "data_seed": seed,
        "bf16": True,
        "fp16": False,
        "tf32": False,
        "optim": "paged_adamw_8bit",
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
        "remove_unused_columns": False,
        "skip_memory_metrics": True,
        "disable_tqdm": False,
        "save_safetensors": True,
    }


def _prepare_runtime() -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    return {
        "torch": torch,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "set_seed": set_seed,
    }


def _run_single_seed(
    *,
    model_dir: Path,
    expected_model_snapshot: StableModelTreeV7,
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    seed_dir: Path,
    seed: int,
    config: QLoRATrainingConfigV6,
    v8c2_protocol: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if model_dir != expected_model_snapshot.root:
        raise PermissionError(
            "runtime model path is not the verified content-addressed snapshot"
        )
    _verify_stable_model_tree_v7(
        expected_model_snapshot,
        label="runtime base model immediately before CUDA",
    )
    runtime = _prepare_runtime()
    torch = runtime["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NF4 QLoRA training")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_mib = free_bytes // (1024 * 1024)
    if free_mib < config.minimum_free_vram_mib:
        raise RuntimeError(
            f"free VRAM {free_mib} MiB is below required "
            f"{config.minimum_free_vram_mib} MiB"
        )

    runtime["set_seed"](seed)
    tokenizer = runtime["AutoTokenizer"].from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    encoded_train, train_tokenization = _encode_rows(
        tokenizer,
        train_rows,
        max_seq_length=config.max_seq_length,
    )
    encoded_validation, validation_tokenization = _encode_rows(
        tokenizer,
        validation_rows,
        max_seq_length=config.max_seq_length,
    )

    class EncodedDataset(torch.utils.data.Dataset):
        def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
            self.rows = list(rows)

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            row = self.rows[index]
            return {
                "input_ids": list(row["input_ids"]),
                "attention_mask": list(row["attention_mask"]),
                "labels": list(row["labels"]),
            }

    class AssistantOnlyCollator:
        def __call__(
            self,
            features: Sequence[Mapping[str, Sequence[int]]],
        ) -> dict[str, Any]:
            maximum = max(len(feature["input_ids"]) for feature in features)
            input_ids: list[list[int]] = []
            attention_mask: list[list[int]] = []
            labels: list[list[int]] = []
            for feature in features:
                padding = maximum - len(feature["input_ids"])
                input_ids.append(
                    list(feature["input_ids"])
                    + [tokenizer.pad_token_id] * padding
                )
                attention_mask.append(
                    list(feature["attention_mask"]) + [0] * padding
                )
                labels.append(list(feature["labels"]) + [-100] * padding)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(
                    attention_mask,
                    dtype=torch.long,
                ),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

    quantization = runtime["BitsAndBytesConfig"](
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    trainer = None
    model = None
    started = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    try:
        model = runtime["AutoModelForCausalLM"].from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=False,
            quantization_config=quantization,
            dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        model.config.use_cache = False
        model = runtime["prepare_model_for_kbit_training"](
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        model = runtime["get_peft_model"](
            model,
            runtime["LoraConfig"](
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=list(DEFAULT_TARGET_MODULES),
            ),
        )
        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        visible_parameters = sum(
            parameter.numel() for parameter in model.parameters()
        )
        trainer_dir = seed_dir / "trainer"
        arguments_payload = _training_arguments_payload(
            trainer_dir=trainer_dir,
            seed=seed,
            config=config,
        )
        arguments = runtime["TrainingArguments"](**arguments_payload)
        trainer = runtime["Trainer"](
            model=model,
            args=arguments,
            train_dataset=EncodedDataset(encoded_train),
            eval_dataset=EncodedDataset(encoded_validation),
            data_collator=AssistantOnlyCollator(),
            processing_class=tokenizer,
            callbacks=[],
        )
        train_result = trainer.train()
        if trainer.state.best_model_checkpoint is not None:
            raise RuntimeError(
                "trainer unexpectedly selected a best model checkpoint"
            )
        epoch_history = _epoch_history(trainer.state.log_history)
        checkpoints = _checkpoint_receipts(seed_dir, epoch_history)
        peak_allocated = int(torch.cuda.max_memory_allocated(0))
        peak_reserved = int(torch.cuda.max_memory_reserved(0))
        receipt = {
            "schema": SEED_RECEIPT_SCHEMA,
            "trainer_version": TRAINER_VERSION,
            "created_at": _utc_now(),
            "status": "PASS_SEED_TRAINED_ALL_EPOCHS_NOT_SELECTED",
            "stage": config.stage,
            "seed": seed,
            "configuration": _configuration_payload(config),
            "dataset": {
                "train_examples": len(train_rows),
                "validation_examples": len(validation_rows),
                "calibration_content_read": False,
                "calibration_content_hashed": False,
                "blind_test_content_read": False,
                "blind_test_content_hashed": False,
                "train_tokenization": train_tokenization,
                "validation_tokenization": validation_tokenization,
            },
            "model_parameters": {
                "trainable": trainable_parameters,
                "visible": visible_parameters,
                "trainable_fraction": trainable_parameters / visible_parameters,
            },
            "per_epoch_metrics": epoch_history,
            "epoch_checkpoints": checkpoints,
            "metrics": {
                "train_result": _sanitize_metrics(train_result.metrics),
                "wall_seconds": time.perf_counter() - started,
                "peak_allocated_vram_bytes": peak_allocated,
                "peak_reserved_vram_bytes": peak_reserved,
            },
            "hardware": {
                "gpu_name": torch.cuda.get_device_name(0),
                "compute_capability": list(
                    torch.cuda.get_device_capability(0)
                ),
                "total_vram_bytes": int(total_bytes),
                "free_vram_before_bytes": int(free_bytes),
            },
            "authorization": {
                "checkpoint_selected": False,
                "model_authorized": False,
                "calibration_authorized": False,
                "blind_test_authorized": False,
                "deployment_authorized": False,
            },
        }
        if v8c2_protocol is not None:
            if dict(v8c2_protocol) != _v8c2_receipt_fields():
                raise QLoRAV6Error(
                    "v8c2 seed protocol binding mismatch"
                )
            receipt.update(v8c2_protocol)
        _write_json_atomic(seed_dir / "seed_receipt.v6.json", receipt)
        return receipt
    finally:
        if trainer is not None:
            del trainer
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


def _package_versions(names: Sequence[str]) -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return versions


def _nvidia_driver_version() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "UNAVAILABLE"
    versions = [
        line.strip() for line in result.stdout.splitlines() if line.strip()
    ]
    return versions[0] if versions else "UNAVAILABLE"


def _runtime_environment_receipt() -> dict[str, Any]:
    import torch

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": _package_versions(
            (
                "torch",
                "transformers",
                "tokenizers",
                "peft",
                "accelerate",
                "bitsandbytes",
                "safetensors",
            )
        ),
        "cuda": {
            "torch_cuda": getattr(torch.version, "cuda", "UNAVAILABLE"),
            "cudnn": torch.backends.cudnn.version(),
            "nvidia_driver": _nvidia_driver_version(),
        },
    }


def _source_inventory(
    *,
    strict_nonblind: bool = False,
    strict_v8: bool = False,
) -> dict[str, Any]:
    paths = {
        "trainer": Path(__file__).resolve(),
        "cli": WORKSPACE_ROOT / "tools" / "train_icmat_qlora_full_v6.py",
    }
    if strict_v8:
        if not strict_nonblind:
            raise QLoRAV6Error(
                "v8 source inventory requires strict nonblind mode"
            )
        paths.update(
            {
                "pointer_evaluator_v8": (
                    WORKSPACE_ROOT
                    / "icmat_foundry"
                    / "llm"
                    / "pointer_checkpoint_eval_v8.py"
                ),
                "pointer_runner_v8": (
                    WORKSPACE_ROOT
                    / "tools"
                    / "evaluate_icmat_pointer_checkpoints_v8.py"
                ),
                "canary_acceptance_v8": (
                    WORKSPACE_ROOT
                    / "icmat_foundry"
                    / "llm"
                    / "canary_acceptance_v8.py"
                ),
                "v8c2_preregistration": V8C2_PREREGISTRATION_PATH,
                "checkpoint_core_v6": (
                    WORKSPACE_ROOT
                    / "icmat_foundry"
                    / "llm"
                    / "pointer_checkpoint_eval_v6.py"
                ),
                "pointer_numeric_evaluator_v6": (
                    WORKSPACE_ROOT
                    / "icmat_foundry"
                    / "llm"
                    / "pointer_hf_eval_v6.py"
                ),
                "pointer_compiler_v6": (
                    WORKSPACE_ROOT
                    / "icmat_foundry"
                    / "llm"
                    / "evidence_pointer_v6.py"
                ),
                "selection_policy_v6": (
                    WORKSPACE_ROOT
                    / "icmat_foundry"
                    / "llm"
                    / "selection_policy_v6.py"
                ),
                "canary_numeric_core_v6": (
                    WORKSPACE_ROOT
                    / "icmat_foundry"
                    / "llm"
                    / "canary_acceptance_v6.py"
                ),
            }
        )
    records: dict[str, Any] = {}
    for role, path in paths.items():
        if strict_nonblind:
            snapshot = _stable_snapshot_v7(
                path,
                label=f"{role} training implementation",
                maximum_bytes=_STRICT_MAX_JSON_BYTES,
            )
            records[role] = {
                "path": snapshot.path.relative_to(
                    WORKSPACE_ROOT
                ).as_posix(),
                "bytes": snapshot.byte_count,
                "sha256": snapshot.sha256,
                "stable_identity": snapshot.identity_receipt(),
            }
        else:
            if not path.is_file():
                raise FileNotFoundError(path)
            records[role] = {
                "path": path.relative_to(WORKSPACE_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    if strict_v8:
        _validate_v8c2_preregistration(
            _stable_snapshot_v7(
                V8C2_PREREGISTRATION_PATH,
                label="v8c2 source-inventory preregistration",
                maximum_bytes=_STRICT_MAX_JSON_BYTES,
            )
        )
    return records


def _safe_new_output(output_dir: Path) -> tuple[Path, Path]:
    raw = Path(output_dir)
    if raw.name in {"", ".", ".."}:
        raise QLoRAV6Error("output must name a new directory")
    parent = raw.parent.resolve(strict=True)
    if not parent.is_dir():
        raise NotADirectoryError(parent)
    final = parent / raw.name
    if os.path.lexists(final):
        raise FileExistsError(final)
    return parent, final


def _validate_v8c2_canary_output_path(output_dir: Path) -> Path:
    observed = _absolute_lexical_v7(Path(output_dir))
    expected = _absolute_lexical_v7(V8C2_CANONICAL_CANARY_OUTPUT_PATH)
    if os.path.normcase(os.fspath(observed)) != os.path.normcase(
        os.fspath(expected)
    ):
        raise QLoRAV6Error(
            "v8c2 canary output must use the canonical registered path"
        )
    return expected


def _v8c2_canary_attempt_availability(
    *,
    required_for_stage: bool,
) -> dict[str, Any]:
    ledger = _absolute_lexical_v7(V8C2_CANARY_ATTEMPT_PATH)
    output = _absolute_lexical_v7(V8C2_CANONICAL_CANARY_OUTPUT_PATH)
    ledger_exists = os.path.lexists(ledger)
    output_exists = os.path.lexists(output)
    parent_available = True
    try:
        _strict_directory_identity_v7(
            ledger.parent,
            label="v8c2 canary attempt parent",
        )
    except (OSError, QLoRAV6Error):
        parent_available = False
    available = not ledger_exists and not output_exists and parent_available
    if ledger_exists:
        state = "BLOCKED_EXISTING_LEDGER"
    elif output_exists:
        state = "BLOCKED_EXISTING_CANONICAL_OUTPUT"
    elif not parent_available:
        state = "BLOCKED_INVALID_PARENT"
    else:
        state = "AVAILABLE"
    return {
        "path": str(ledger),
        "canonical_output_path": str(output),
        "required_for_stage": required_for_stage,
        "read_only": True,
        "created": False,
        "existing_entry_blocks": True,
        "ledger_available": not ledger_exists and parent_available,
        "canonical_output_available": not output_exists,
        "available": available,
        "state": state,
    }


def _v8c2_canary_attempt_payload(
    *,
    run_id: str,
    configuration_sha256: str,
    dataset_input_sha256: str,
    training_gate_bundle_sha256: str,
    source_inventory_sha256: str,
    base_model_tree_sha256: str,
) -> dict[str, Any]:
    hashes = {
        "configuration_sha256": configuration_sha256,
        "dataset_input_sha256": dataset_input_sha256,
        "training_gate_bundle_sha256": training_gate_bundle_sha256,
        "source_inventory_sha256": source_inventory_sha256,
        "base_model_tree_sha256": base_model_tree_sha256,
    }
    if (
        not isinstance(run_id, str)
        or not run_id.startswith("icmat-v6-")
        or any(not _valid_sha256(value) for value in hashes.values())
    ):
        raise QLoRAV6Error("v8c2 canary attempt binding is invalid")
    core = {
        "schema": V8C2_CANARY_ATTEMPT_SCHEMA,
        "protocol_id": V8C2_PREREGISTRATION_PROTOCOL_ID,
        "preregistration_sha256": V8C2_PREREGISTRATION_SHA256,
        "status": V8C2_CANARY_ATTEMPT_STATUS,
        "run_id": run_id,
        "output_path": str(
            _absolute_lexical_v7(V8C2_CANONICAL_CANARY_OUTPUT_PATH)
        ),
        **hashes,
    }
    return {
        **core,
        "attempt_payload_sha256": _canonical_sha256(core),
    }


def _canonical_json_bytes_v8c2(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _validate_v8c2_canary_attempt_receipt(
    attempt: Mapping[str, Any],
    *,
    run_id: str,
    configuration_sha256: str,
    dataset_input_sha256: str,
    training_gate_bundle_sha256: str,
    source_inventory_sha256: str,
    base_model_tree_sha256: str,
) -> dict[str, Any]:
    expected_payload = _v8c2_canary_attempt_payload(
        run_id=run_id,
        configuration_sha256=configuration_sha256,
        dataset_input_sha256=dataset_input_sha256,
        training_gate_bundle_sha256=training_gate_bundle_sha256,
        source_inventory_sha256=source_inventory_sha256,
        base_model_tree_sha256=base_model_tree_sha256,
    )
    expected_keys = {
        "path",
        "bytes",
        "sha256",
        "stable_identity",
        *expected_payload,
    }
    if set(attempt) != expected_keys:
        raise QLoRAV6Error("v8c2 canary attempt receipt keys mismatch")
    expected_path = _absolute_lexical_v7(V8C2_CANARY_ATTEMPT_PATH)
    observed_path = _absolute_lexical_v7(Path(str(attempt.get("path"))))
    if os.path.normcase(os.fspath(observed_path)) != os.path.normcase(
        os.fspath(expected_path)
    ):
        raise QLoRAV6Error("v8c2 canary attempt path mismatch")
    snapshot = _stable_snapshot_v7(
        expected_path,
        label="v8c2 canary attempt ledger",
        maximum_bytes=_STRICT_MAX_JSON_BYTES,
    )
    ledger = _strict_json_object_v7(
        snapshot,
        label="v8c2 canary attempt ledger",
    )
    semantic_receipt = {
        key: attempt.get(key) for key in expected_payload
    }
    if (
        ledger != expected_payload
        or semantic_receipt != expected_payload
        or snapshot.payload != _canonical_json_bytes_v8c2(expected_payload)
        or attempt.get("bytes") != snapshot.byte_count
        or attempt.get("sha256") != snapshot.sha256
        or attempt.get("stable_identity") != snapshot.identity_receipt()
    ):
        raise QLoRAV6Error("v8c2 canary attempt receipt binding mismatch")
    return dict(attempt)


def _create_v8c2_canary_attempt(
    *,
    run_id: str,
    configuration_sha256: str,
    dataset_input_sha256: str,
    training_gate_bundle_sha256: str,
    source_inventory_sha256: str,
    base_model_tree_sha256: str,
) -> dict[str, Any]:
    ledger = _absolute_lexical_v7(V8C2_CANARY_ATTEMPT_PATH)
    output = _validate_v8c2_canary_output_path(
        V8C2_CANONICAL_CANARY_OUTPUT_PATH
    )
    _strict_directory_identity_v7(
        ledger.parent,
        label="v8c2 canary attempt parent",
    )
    if os.path.lexists(ledger):
        raise FileExistsError(
            "v8c2 canary attempt ledger already exists and blocks another "
            "attempt"
        )
    if os.path.lexists(output):
        raise FileExistsError(
            "v8c2 canonical canary output already exists and blocks another "
            "attempt"
        )
    _assert_no_link_components_v7(
        ledger,
        label="v8c2 canary attempt ledger",
        allow_missing_leaf=True,
    )
    payload = _v8c2_canary_attempt_payload(
        run_id=run_id,
        configuration_sha256=configuration_sha256,
        dataset_input_sha256=dataset_input_sha256,
        training_gate_bundle_sha256=training_gate_bundle_sha256,
        source_inventory_sha256=source_inventory_sha256,
        base_model_tree_sha256=base_model_tree_sha256,
    )
    encoded = _canonical_json_bytes_v8c2(payload)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(os.fspath(ledger), flags, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(
            "v8c2 canary attempt ledger already exists and blocks another "
            "attempt"
        ) from exc
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written < 1:
                raise OSError("v8c2 canary attempt ledger write stalled")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    snapshot = _stable_snapshot_v7(
        ledger,
        label="v8c2 canary attempt ledger",
        maximum_bytes=_STRICT_MAX_JSON_BYTES,
    )
    receipt = {
        "path": str(snapshot.path),
        "bytes": snapshot.byte_count,
        "sha256": snapshot.sha256,
        "stable_identity": snapshot.identity_receipt(),
        **payload,
    }
    return _validate_v8c2_canary_attempt_receipt(
        receipt,
        run_id=run_id,
        configuration_sha256=configuration_sha256,
        dataset_input_sha256=dataset_input_sha256,
        training_gate_bundle_sha256=training_gate_bundle_sha256,
        source_inventory_sha256=source_inventory_sha256,
        base_model_tree_sha256=base_model_tree_sha256,
    )


def _failure_directory(parent: Path, final_name: str, run_id: str) -> Path:
    return parent / f".{final_name}.failed-{run_id}-{uuid.uuid4().hex}"


def _pre_cuda_security_state_v7(
    preflight: Mapping[str, Any],
    source_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": preflight["dataset"],
        "base_model": preflight["base_model"],
        "canary_acceptance": preflight["canary_acceptance"],
        "configuration": preflight["configuration"],
        "configuration_sha256": preflight["configuration_sha256"],
        "authorization": preflight["authorization"],
        "source_inventory": source_inventory,
    }


def run_training_v6(
    *,
    model_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
    config: QLoRATrainingConfigV6 | None = None,
    nonblind_second_build_dir: Path | None = None,
    nonblind_audit_receipt: Path | None = None,
    train_shortcut_audit: Path | None = None,
    validation_shortcut_audit: Path | None = None,
    scoped_lexical_audit_v8: Path | None = None,
    train_unique_support_audit_v8: Path | None = None,
    validation_unique_support_audit_v8: Path | None = None,
    unique_support_nli_model_dir: Path | None = None,
    canary_acceptance_receipt: Path | None = None,
    canary_evaluation_index: Path | None = None,
    canary_training_receipt: Path | None = None,
) -> dict[str, Any]:
    """Train all stage seeds and atomically publish an unselected candidate."""

    config = QLoRATrainingConfigV6() if config is None else config
    config.validate()
    canary_gate_paths = {
        "canary_acceptance_receipt": canary_acceptance_receipt,
        "canary_evaluation_index": canary_evaluation_index,
        "canary_training_receipt": canary_training_receipt,
    }
    if config.stage == "final" and any(
        path is None for path in canary_gate_paths.values()
    ):
        raise QLoRAV6Error(
            "final three-seed training requires explicit canary acceptance "
            "receipt, evaluation index, and canary training receipt"
        )
    model_dir = _absolute_lexical_v7(Path(model_dir))
    _assert_no_link_components_v7(
        model_dir,
        label="base model directory",
    )
    dataset_dir = Path(os.path.abspath(os.fspath(Path(dataset_dir))))
    requested_output = Path(output_dir)
    preflight = preflight_v6_contract(
        dataset_dir=dataset_dir,
        model_dir=model_dir,
        config=config,
        nonblind_second_build_dir=nonblind_second_build_dir,
        nonblind_audit_receipt=nonblind_audit_receipt,
        train_shortcut_audit=train_shortcut_audit,
        validation_shortcut_audit=validation_shortcut_audit,
        scoped_lexical_audit_v8=scoped_lexical_audit_v8,
        train_unique_support_audit_v8=(
            train_unique_support_audit_v8
        ),
        validation_unique_support_audit_v8=(
            validation_unique_support_audit_v8
        ),
        unique_support_nli_model_dir=unique_support_nli_model_dir,
        canary_acceptance_receipt=canary_acceptance_receipt,
        canary_evaluation_index=canary_evaluation_index,
        canary_training_receipt=canary_training_receipt,
    )
    dataset_kind = _strict_dataset_kind(preflight["dataset"])
    strict_v7 = dataset_kind == "v7"
    strict_v8 = dataset_kind == "v8"
    strict_nonblind = dataset_kind in {"v7", "v8"}
    if strict_v8 and config.stage == "canary":
        requested_output = _validate_v8c2_canary_output_path(
            requested_output
        )
    parent, final_output = _safe_new_output(requested_output)
    runtime_config = (
        _effective_training_config_v8c2(config)
        if strict_v8
        else config
    )
    source_inventory = _source_inventory(
        strict_nonblind=strict_nonblind,
        strict_v8=strict_v8,
    )
    source_model_tree = _stable_model_tree_v7(
        model_dir,
        label="base model before runtime snapshot construction",
    )
    if not _model_receipt_matches_tree_v7(
        preflight["base_model"],
        source_model_tree,
    ):
        raise PermissionError(
            "base model changed after initial preflight"
        )
    initial_security_state = _pre_cuda_security_state_v7(
        preflight,
        source_inventory,
    )
    run_core = {
        "trainer_version": TRAINER_VERSION,
        "stage": config.stage,
        "dataset_input_sha256": preflight["dataset"][
            "inspected_input_sha256"
        ],
        "model_tree_sha256": preflight["base_model"]["tree_sha256"],
        "configuration_sha256": preflight["configuration_sha256"],
        "canary_acceptance_sha256": (
            preflight["canary_acceptance"].get("sha256")
        ),
        "source_inventory": source_inventory,
        "output_name": final_output.name,
    }
    if strict_v8:
        run_core["training_gate_bundle_sha256"] = preflight[
            "training_gate_bundle_sha256"
        ]
        run_core["v8_inspected_input_sha256"] = preflight[
            "v8_inspected_input_sha256"
        ]
    run_id = "icmat-v6-" + _canonical_sha256(run_core)[:20]
    staging = parent / (
        f".{final_output.name}.tmp-{run_id}-{uuid.uuid4().hex}"
    )
    if os.path.lexists(staging):
        raise FileExistsError(staging)
    canary_attempt: dict[str, Any] | None = None
    if strict_v8 and config.stage == "canary":
        _revalidate_v8_identity_bundle(preflight["dataset"])
        current_source_inventory = _source_inventory(
            strict_nonblind=True,
            strict_v8=True,
        )
        if current_source_inventory != source_inventory:
            raise PermissionError(
                "v8c2 source inventory changed before canary reservation"
            )
        _verify_stable_model_tree_v7(
            source_model_tree,
            label="base model before v8c2 canary reservation",
        )
        canary_attempt = _create_v8c2_canary_attempt(
            run_id=run_id,
            configuration_sha256=preflight["configuration_sha256"],
            dataset_input_sha256=preflight["dataset"][
                "inspected_input_sha256"
            ],
            training_gate_bundle_sha256=preflight[
                "training_gate_bundle_sha256"
            ],
            source_inventory_sha256=_canonical_sha256(source_inventory),
            base_model_tree_sha256=preflight["base_model"][
                "tree_sha256"
            ],
        )
    os.mkdir(staging)
    started = time.perf_counter()
    completed_seeds: list[dict[str, Any]] = []
    active_seed: int | None = None
    runtime_model_tree: StableModelTreeV7 | None = None
    try:
        _write_json_atomic(staging / "preflight.v6.json", preflight)
        train_rows = _load_training_rows(
            dataset_dir,
            "train",
            expected=preflight["dataset"]["splits"]["train"],
            strict_nonblind=strict_nonblind,
        )
        validation_rows = _load_training_rows(
            dataset_dir,
            "validation",
            expected=preflight["dataset"]["splits"]["validation"],
            strict_nonblind=strict_nonblind,
        )
        runtime_model_tree = _copy_content_addressed_model_v7(
            source_model_tree,
            staging=staging,
        )

        def revalidate_before_cuda() -> dict[str, Any]:
            if strict_v8:
                _revalidate_v8_identity_bundle(preflight["dataset"])
                _revalidate_v8_canary_acceptance(
                    preflight["canary_acceptance"]
                )
                current_source_inventory = _source_inventory(
                    strict_nonblind=True,
                    strict_v8=True,
                )
                if (
                    _pre_cuda_security_state_v7(
                        preflight,
                        current_source_inventory,
                    )
                    != initial_security_state
                ):
                    raise PermissionError(
                        "training authority or implementation changed "
                        "before CUDA"
                    )
                _verify_stable_model_tree_v7(
                    runtime_model_tree,
                    label="content-addressed runtime model before CUDA",
                )
                return preflight
            current = preflight_v6_contract(
                dataset_dir=dataset_dir,
                model_dir=model_dir,
                config=config,
                nonblind_second_build_dir=nonblind_second_build_dir,
                nonblind_audit_receipt=nonblind_audit_receipt,
                train_shortcut_audit=train_shortcut_audit,
                validation_shortcut_audit=validation_shortcut_audit,
                scoped_lexical_audit_v8=scoped_lexical_audit_v8,
                train_unique_support_audit_v8=(
                    train_unique_support_audit_v8
                ),
                validation_unique_support_audit_v8=(
                    validation_unique_support_audit_v8
                ),
                unique_support_nli_model_dir=(
                    unique_support_nli_model_dir
                ),
                canary_acceptance_receipt=canary_acceptance_receipt,
                canary_evaluation_index=canary_evaluation_index,
                canary_training_receipt=canary_training_receipt,
            )
            current_source_inventory = _source_inventory(
                strict_nonblind=strict_nonblind,
            )
            if (
                _pre_cuda_security_state_v7(
                    current,
                    current_source_inventory,
                )
                != initial_security_state
            ):
                raise PermissionError(
                    "training authority, implementation, dataset, or "
                    "base-model identity changed before CUDA"
                )
            _verify_stable_model_tree_v7(
                runtime_model_tree,
                label="content-addressed runtime model before CUDA",
            )
            return current

        for seed in runtime_config.resolved_seeds:
            active_seed = seed
            revalidate_before_cuda()
            seed_dir = staging / f"seed-{seed}"
            os.mkdir(seed_dir)
            completed_seeds.append(
                _run_single_seed(
                    model_dir=runtime_model_tree.root,
                    expected_model_snapshot=runtime_model_tree,
                    train_rows=train_rows,
                    validation_rows=validation_rows,
                    seed_dir=seed_dir,
                    seed=seed,
                    config=runtime_config,
                    v8c2_protocol=(
                        _v8c2_receipt_fields() if strict_v8 else None
                    ),
                )
            )

        final_snapshot = revalidate_before_cuda()
        final_source_inventory = _source_inventory(
            strict_nonblind=strict_nonblind,
            strict_v8=strict_v8,
        )
        if (
            final_snapshot["dataset"]["inspected_input_sha256"]
            != preflight["dataset"]["inspected_input_sha256"]
            or final_snapshot["base_model"]["tree_sha256"]
            != preflight["base_model"]["tree_sha256"]
            or final_snapshot["configuration_sha256"]
            != preflight["configuration_sha256"]
            or final_snapshot["canary_acceptance"]
            != preflight["canary_acceptance"]
            or final_source_inventory != source_inventory
        ):
            raise PermissionError("training inputs changed during the run")

        expected_seed_count = 1 if config.stage == "canary" else 3
        if len(completed_seeds) != expected_seed_count:
            raise RuntimeError("not all configured seed runs completed")
        environment = _runtime_environment_receipt()
        status = (
            "PASS_CANARY_SINGLE_SEED_ALL_EPOCHS_NOT_SELECTED"
            if config.stage == "canary"
            else "PASS_FINAL_THREE_SEED_ALL_EPOCHS_NOT_SELECTED"
        )
        _remove_content_addressed_model_v7(runtime_model_tree)
        runtime_model_tree = None
        base_model_receipt = {
            **preflight["base_model"],
            "runtime_loading": {
                "policy": "CONTENT_ADDRESSED_STABLE_LOCAL_COPY_V7",
                "content_address": (
                    f"sha256:{preflight['base_model']['tree_sha256']}"
                ),
                "tree_sha256": preflight["base_model"]["tree_sha256"],
                "file_count": preflight["base_model"]["file_count"],
                "bytes": preflight["base_model"]["bytes"],
                "verified_before_cuda": True,
                "loaded_only_from_snapshot": True,
                "removed_before_publish": True,
            },
        }
        receipt = {
            "schema": RUN_RECEIPT_SCHEMA,
            "trainer_version": TRAINER_VERSION,
            "created_at": _utc_now(),
            "status": status,
            "stage": config.stage,
            "run_id": run_id,
            "atomic_publish": True,
            "network_used": False,
            "input_snapshot": {
                "dataset": preflight["dataset"],
                "base_model": base_model_receipt,
                "canary_acceptance": preflight["canary_acceptance"],
                "source_files": source_inventory,
            },
            "configuration": preflight["configuration"],
            "configuration_sha256": preflight["configuration_sha256"],
            "software": {
                "python": environment["python"],
                "platform": environment["platform"],
                "dependencies": environment["dependencies"],
            },
            "cuda": environment["cuda"],
            "seeds": completed_seeds,
            "checkpoint_count": sum(
                len(receipt["epoch_checkpoints"])
                for receipt in completed_seeds
            ),
            "selection": {
                "automatic_selection_performed": False,
                "selected_seed": None,
                "selected_epoch": None,
                "selected_adapter": None,
                "selection_metric": None,
                "required_next_step": (
                    "independent full validation pointer evaluation"
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
            "data_access": {
                "train_content_read": True,
                "validation_content_read": True,
                "calibration_content_read": False,
                "calibration_content_hashed": False,
                "blind_test_content_read": False,
                "blind_test_content_hashed": False,
                **(
                    {
                        "calibration_integrity_snapshot_opened": True,
                        "calibration_integrity_content_read": True,
                        "calibration_integrity_content_hashed": True,
                        "calibration_content_loaded_for_training": False,
                        "calibration_used_for_checkpoint_selection": False,
                        "nonblind_compare_audit_verified": True,
                        "scoped_lexical_audit_verified": True,
                        "scoped_lexical_audit_locally_recomputed": True,
                        "train_unique_support_audit_verified": True,
                        "validation_unique_support_audit_verified": True,
                        "unique_support_fixed_cpu_nli_load_count": 1,
                        "unique_support_nli_repeated_per_seed": False,
                        "second_build_fixed_files_recomputed": True,
                        "declared_nonblind_audit_artifacts_opened": 8,
                        "declared_nonblind_audit_artifacts_hashed": 8,
                        "blind_materialized": False,
                        "blind_discovered": False,
                        "blind_path_constructed": False,
                        "blind_filesystem_metadata_accessed": False,
                        "blind_content_opened": False,
                        "blind_content_read": False,
                        "blind_content_hashed": False,
                    }
                    if strict_v8
                    else (
                    {
                        "calibration_legacy_fields_mean_training_access_only": (
                            True
                        ),
                        "calibration_integrity_snapshot_opened": True,
                        "calibration_integrity_content_read": True,
                        "calibration_integrity_content_hashed": True,
                        "calibration_content_loaded_for_training": False,
                        "calibration_used_for_checkpoint_selection": False,
                        "nonblind_compare_audit_verified": True,
                        "train_shortcut_audit_verified": True,
                        "validation_shortcut_audit_verified": True,
                        "second_build_fixed_files_recomputed": True,
                        "shortcut_audits_locally_recomputed": True,
                        "declared_nonblind_audit_artifacts_opened": 6,
                        "declared_nonblind_audit_artifacts_hashed": 6,
                        "blind_materialized": False,
                        "blind_discovered": False,
                        "blind_path_constructed": False,
                        "blind_filesystem_metadata_accessed": False,
                        "blind_content_opened": False,
                        "blind_content_read": False,
                        "blind_content_hashed": False,
                    }
                    if strict_v7
                    else {}
                    )
                ),
            },
            "wall_seconds": time.perf_counter() - started,
            "claim_boundary": (
                (
                    "This receipt proves only that the fixed local NF4, "
                    "v8c2 rank-8 QLoRA schedule completed on strict "
                    "nonblind-v8 "
                    "train and validation data after direct twelve-file "
                    "two-build comparison, local scoped lexical "
                    "recomputation, and one fixed-CPU recomputation of both "
                    "unique-support gates. Per-seed reloads performed stable "
                    "identity checks without rerunning NLI. Calibration was "
                    "integrity-only and no blind artifact was discovered or "
                    "opened. No checkpoint, quality, calibration, blind, "
                    "GGUF, X5, deployment or production authorization is "
                    "granted."
                )
                if strict_v8
                else (
                    "This receipt proves only that the fixed local NF4, "
                    "rank-16 QLoRA schedule completed on strict nonblind-v7 "
                    "train and validation data after direct two-build byte "
                    "comparison and local split-specific shortcut "
                    "recomputation, and retained every epoch checkpoint. "
                    "Calibration was read and hashed only for fixed-inventory "
                    "reproducibility and was never loaded as training or "
                    "selection input; no blind artifact was materialized, "
                    "discovered, constructed, opened, statted, read or "
                    "hashed. No checkpoint, quality, calibration, blind, "
                    "GGUF, X5, deployment or production authorization is "
                    "granted."
                    if strict_v7
                    else (
                    "This receipt proves only that the fixed local NF4, "
                    "rank-16 QLoRA schedule completed and retained every "
                    "epoch checkpoint. No checkpoint or seed was selected, "
                    "and no quality, calibration, blind-test, GGUF, X5, "
                    "deployment or production authorization is granted."
                    )
                )
            ),
        }
        if strict_v8:
            receipt["training_gate_bundle_sha256"] = preflight[
                "training_gate_bundle_sha256"
            ]
            receipt["v8_inspected_input_sha256"] = preflight[
                "v8_inspected_input_sha256"
            ]
            receipt.update(_v8c2_receipt_fields())
            if config.stage == "canary":
                if canary_attempt is None:
                    raise RuntimeError(
                        "v8c2 canary attempt receipt was not reserved"
                    )
                receipt["canary_attempt"] = (
                    _validate_v8c2_canary_attempt_receipt(
                        canary_attempt,
                        run_id=run_id,
                        configuration_sha256=preflight[
                            "configuration_sha256"
                        ],
                        dataset_input_sha256=preflight["dataset"][
                            "inspected_input_sha256"
                        ],
                        training_gate_bundle_sha256=preflight[
                            "training_gate_bundle_sha256"
                        ],
                        source_inventory_sha256=_canonical_sha256(
                            source_inventory
                        ),
                        base_model_tree_sha256=preflight["base_model"][
                            "tree_sha256"
                        ],
                    )
                )
        _write_json_atomic(staging / "training_receipt.v6.json", receipt)
        os.replace(staging, final_output)
        return receipt
    except BaseException as exc:
        if runtime_model_tree is not None:
            try:
                _remove_content_addressed_model_v7(runtime_model_tree)
                runtime_model_tree = None
            except BaseException:
                pass
        failure = {
            "schema": FAILURE_RECEIPT_SCHEMA,
            "trainer_version": TRAINER_VERSION,
            "created_at": _utc_now(),
            "status": "FAILED_NO_SUCCESS_RELEASE",
            "stage": config.stage,
            "run_id": run_id,
            "active_seed": active_seed,
            "completed_seeds": [
                receipt.get("seed") for receipt in completed_seeds
            ],
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "final_output_created": False,
            "calibration_content_read": False,
            "calibration_content_hashed": False,
            "blind_test_content_read": False,
            "blind_test_content_hashed": False,
            **(
                {
                    "calibration_legacy_fields_mean_training_access_only": (
                        True
                    ),
                    "calibration_integrity_snapshot_opened": True,
                    "calibration_integrity_content_read": True,
                    "calibration_integrity_content_hashed": True,
                    "calibration_content_loaded_for_training": False,
                    "blind_materialized": False,
                    "blind_discovered": False,
                    "blind_path_constructed": False,
                    "blind_filesystem_metadata_accessed": False,
                    "blind_content_opened": False,
                    "blind_content_read": False,
                    "blind_content_hashed": False,
                }
                if strict_nonblind
                else {}
            ),
            "checkpoint_selected": False,
            "model_authorized": False,
            **(
                {"canary_attempt": canary_attempt}
                if canary_attempt is not None
                else {}
            ),
        }
        try:
            _write_json_atomic(
                staging / "failure_receipt.v6.json",
                failure,
            )
            failed = _failure_directory(
                parent,
                final_output.name,
                run_id,
            )
            os.replace(staging, failed)
        except BaseException:
            pass
        raise
