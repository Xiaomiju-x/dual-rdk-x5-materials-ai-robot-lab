"""Frozen task/runtime contracts for the nonblind-v7 selection lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from icmat_foundry.llm import (
    evidence_pointer_v6,
    evidence_sft_v6,
    selection_policy_v6,
)
from icmat_foundry.llm.selection_freeze_v7 import (
    SCHEMA as SELECTION_SCHEMA,
)
from icmat_foundry.llm.selection_freeze_v7 import (
    STATUS as SELECTION_STATUS,
)
from icmat_foundry.llm.selection_freeze_v7 import (
    VERSION as SELECTION_VERSION,
)
from icmat_foundry.llm.selection_freeze_v7 import (
    SelectionFreezeV7Error,
    _authority_lease_scope,
    _DirectoryAnchor,
    _lease_authority_path,
    _lease_exclusion,
    canonical_sha256,
    verify_selection_freeze_v7,
)

VERSION = "icmat-contracts-v7.1.0"
TASK_SCHEMA = "icmat_llm_task_contract.v7"
PREPROCESSING_SCHEMA = "icmat_llm_preprocessing_contract.v7"
DECISION_POLICY_SCHEMA = "icmat_llm_decision_policy_contract.v7"
BUILD_RECEIPT_SCHEMA = "icmat_llm_contract_build_receipt.v7"
TASK_FILENAME = "task_contract.v7.json"
PREPROCESSING_FILENAME = "preprocessing_contract.v7.json"
DECISION_POLICY_FILENAME = "decision_policy_contract.v7.json"
BUILD_RECEIPT_FILENAME = "contract_build_receipt.v7.json"
CONTRACT_FILENAMES = {
    TASK_FILENAME,
    PREPROCESSING_FILENAME,
    DECISION_POLICY_FILENAME,
    BUILD_RECEIPT_FILENAME,
}


class ContractsV7Error(RuntimeError):
    """Raised when an immutable v7 contract set cannot be built or verified."""


@dataclass(frozen=True)
class StableSnapshot:
    path: Path
    payload: bytes
    sha256: str
    identity: tuple[int, int, int, int, int]


def _reject_nonfinite(value: str) -> None:
    raise ContractsV7Error(f"non-finite JSON constant rejected: {value}")


def _reject_duplicate_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ContractsV7Error(f"duplicate JSON key rejected: {key}")
        output[key] = value
    return output


def _is_reparse(metadata: os.stat_result) -> bool:
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    marker = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return bool(attributes & marker)


def _assert_no_reparse_chain(path: Path, *, label: str) -> Path:
    lexical = path.expanduser().absolute()
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
            raise ContractsV7Error(
                f"{label}: symlink/reparse component rejected: {current}"
            )
    return lexical


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _snapshot(path: Path, *, label: str) -> StableSnapshot:
    lexical = _assert_no_reparse_chain(path, label=label)
    _lease_authority_path(lexical, directory=False)
    metadata = os.lstat(lexical)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise ContractsV7Error(f"{label}: regular file required")
    with lexical.open("rb") as handle:
        before = _identity(os.fstat(handle.fileno()))
        payload = handle.read()
        after = _identity(os.fstat(handle.fileno()))
    current = _identity(os.lstat(lexical))
    if before != after or after != current or len(payload) != current[2]:
        raise ContractsV7Error(f"{label}: TOCTOU detected")
    return StableSnapshot(
        path=lexical.resolve(strict=True),
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        identity=current,
    )


def _load_json(snapshot: StableSnapshot, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            snapshot.payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractsV7Error(f"{label}: invalid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ContractsV7Error(f"{label}: JSON object required")
    return value


def _require_exact_keys(
    value: Any,
    expected: set[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ContractsV7Error(f"{label}: exact field set mismatch")
    return value


def _require_sha(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractsV7Error(f"{label}: lowercase SHA-256 required")
    return value


def _validate_metadata(contract_id: str, created_at: str) -> None:
    if not contract_id or any(character.isspace() for character in contract_id):
        raise ContractsV7Error("contract_id must be non-empty without whitespace")
    try:
        parsed = datetime.fromisoformat(created_at)
    except ValueError as exc:
        raise ContractsV7Error("created_at must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ContractsV7Error("created_at must include a timezone")


def _selection_binding(
    selection_path: Path,
    *,
    evaluation_index: Path,
    training_receipt: Path,
    dataset_dir: Path,
    base_model_dir: Path,
) -> tuple[StableSnapshot, dict[str, Any], dict[str, Any]]:
    snapshot = _snapshot(selection_path, label="selection freeze")
    try:
        verified = verify_selection_freeze_v7(
            freeze_receipt_path=snapshot.path,
            evaluation_index_path=evaluation_index,
            training_receipt_path=training_receipt,
            dataset_dir=dataset_dir,
            base_model_dir=base_model_dir,
        )
    except (SelectionFreezeV7Error, OSError, ValueError) as exc:
        raise ContractsV7Error(
            f"selection authority verification failed: {exc}"
        ) from exc
    if _snapshot(snapshot.path, label="selection freeze recheck") != snapshot:
        raise ContractsV7Error("selection freeze changed during authority verification")
    receipt = _load_json(snapshot, label="selection freeze")
    expected_fields = {
        "schema",
        "version",
        "created_at_utc",
        "status",
        "selection_locked",
        "calibration_authorized",
        "blind_test_authorized",
        "deployment_authorized",
        "selection_binding_digest_sha256",
        "manifest",
        "preblind_commitment",
        "training_receipt",
        "evaluation_receipt",
        "training_authority",
        "evaluation_evidence",
        "base_model",
        "selection_policy",
        "selection",
        "authorization",
        "access_boundary",
        "claim_boundary",
        "canonical_digest_sha256",
    }
    _require_exact_keys(receipt, expected_fields, label="selection freeze")
    if (
        receipt["schema"] != SELECTION_SCHEMA
        or receipt["version"] != SELECTION_VERSION
        or receipt["status"] != SELECTION_STATUS
        or receipt["selection_locked"] is not True
        or receipt["calibration_authorized"] is not True
        or receipt["blind_test_authorized"] is not False
        or receipt["deployment_authorized"] is not False
    ):
        raise ContractsV7Error("selection freeze identity mismatch")
    if (
        verified.get("selected_checkpoint_id")
        != receipt["selection"]["checkpoint_id"]
        or verified.get("manifest_sha256") != receipt["manifest"]["sha256"]
        or verified.get("preblind_commitment_sha256")
        != receipt["preblind_commitment"]["commitment_sha256"]
    ):
        raise ContractsV7Error("selection authority verification result mismatch")
    body = dict(receipt)
    recorded_digest = _require_sha(
        body.pop("canonical_digest_sha256"),
        label="selection canonical digest",
    )
    if canonical_sha256(body) != recorded_digest:
        raise ContractsV7Error("selection freeze canonical digest mismatch")
    for label, value in (
        ("manifest", receipt["manifest"]),
        ("training receipt", receipt["training_receipt"]),
        ("evaluation receipt", receipt["evaluation_receipt"]),
    ):
        expected = (
            {"path", "bytes", "sha256", "schema"}
            if label != "manifest"
            else {"path", "bytes", "sha256", "schema"}
        )
        record = _require_exact_keys(value, expected, label=label)
        _require_sha(record["sha256"], label=f"{label} SHA")
    preblind = _require_exact_keys(
        receipt["preblind_commitment"],
        {
            "path",
            "bytes",
            "sha256",
            "schema",
            "commitment_sha256",
            "expected_future_rows",
        },
        label="selection preblind binding",
    )
    _require_sha(preblind["sha256"], label="preblind file SHA")
    commitment_sha = _require_sha(
        preblind["commitment_sha256"],
        label="selection preblind commitment SHA",
    )
    _require_exact_keys(
        receipt["base_model"],
        {
            "path",
            "tree_sha256",
            "evaluator_tree_sha256",
            "file_count",
            "bytes",
            "stable_tree_digest_sha256",
        },
        label="selection base model",
    )
    _require_sha(
        receipt["base_model"]["tree_sha256"],
        label="selection base-model tree SHA",
    )
    _require_sha(
        receipt["base_model"]["evaluator_tree_sha256"],
        label="selection evaluator base-model tree SHA",
    )
    _require_sha(
        receipt["base_model"]["stable_tree_digest_sha256"],
        label="selection stable base-model tree SHA",
    )
    policy = _require_exact_keys(
        receipt["selection_policy"],
        {"schema", "version", "decision"},
        label="selection policy binding",
    )
    if (
        policy["schema"] != selection_policy_v6.SCHEMA
        or policy["version"] != selection_policy_v6.POLICY_VERSION
    ):
        raise ContractsV7Error("selection policy identity mismatch")
    decision = _require_exact_keys(
        policy["decision"],
        {
            "schema",
            "policy_version",
            "execution_contract",
            "population",
            "thresholds",
            "qualified_checkpoint_count",
            "qualified_seeds",
            "evaluations",
            "status",
            "selection_allowed",
            "selection",
            "rejection",
        },
        label="selection policy decision",
    )
    if (
        decision["status"] != selection_policy_v6.SELECTED_STATUS
        or decision["selection_allowed"] is not True
        or decision["rejection"] is not None
    ):
        raise ContractsV7Error("selection policy did not select a checkpoint")
    selection = _require_exact_keys(
        receipt["selection"],
        {
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
    if (
        selection.get("selection_locked") is not True
        or not isinstance(selection.get("checkpoint_id"), str)
    ):
        raise ContractsV7Error("selection checkpoint binding mismatch")
    if (
        decision["selection"]["checkpoint_id"] != selection["checkpoint_id"]
        or decision["selection"]["seed"] != selection["seed"]
        or decision["selection"]["epoch"] != selection["epoch"]
    ):
        raise ContractsV7Error("selection decision/checkpoint mismatch")
    authorization = _require_exact_keys(
        receipt["authorization"],
        {
            "calibration_authorized",
            "calibration_complete_split_only",
            "calibration_expected_rows",
            "calibration_may_reselect_checkpoint",
            "ablation_authorized_on_validation_only",
            "blind_test_authorized",
            "gguf_export_authorized",
            "deployment_authorized",
            "production_integration_authorized",
        },
        label="selection authorization",
    )
    if authorization != {
        "calibration_authorized": True,
        "calibration_complete_split_only": True,
        "calibration_expected_rows": 150,
        "calibration_may_reselect_checkpoint": False,
        "ablation_authorized_on_validation_only": True,
        "blind_test_authorized": False,
        "gguf_export_authorized": False,
        "deployment_authorized": False,
        "production_integration_authorized": False,
    }:
        raise ContractsV7Error("selection authorization mismatch")
    access = _require_exact_keys(
        receipt["access_boundary"],
        {
            "manifest_opened",
            "preblind_commitment_opened",
            "training_receipt_opened",
            "evaluation_receipt_opened",
            "base_model_hashed",
            "checkpoint_hashed",
            "training_implementations_hashed",
            "evaluation_implementations_hashed",
            "evaluation_artifacts_opened",
            "evaluation_artifacts_recomputed",
            "calibration_path_constructed",
            "calibration_filesystem_metadata_accessed",
            "calibration_content_opened",
            "calibration_content_read",
            "calibration_content_hashed",
            "blind_path_constructed",
            "blind_filesystem_metadata_accessed",
            "blind_content_opened",
            "blind_content_read",
            "blind_content_hashed",
        },
        label="selection access boundary",
    )
    if any(
        access[field] is not False
        for field in (
            "calibration_path_constructed",
            "calibration_filesystem_metadata_accessed",
            "calibration_content_opened",
            "calibration_content_read",
            "calibration_content_hashed",
            "blind_path_constructed",
            "blind_filesystem_metadata_accessed",
            "blind_content_opened",
            "blind_content_read",
            "blind_content_hashed",
        )
    ):
        raise ContractsV7Error("selection reserved-data boundary mismatch")
    if any(
        access[field] is not True
        for field in (
            "manifest_opened",
            "preblind_commitment_opened",
            "training_receipt_opened",
            "evaluation_receipt_opened",
            "base_model_hashed",
            "checkpoint_hashed",
            "training_implementations_hashed",
            "evaluation_implementations_hashed",
            "evaluation_artifacts_opened",
            "evaluation_artifacts_recomputed",
        )
    ):
        raise ContractsV7Error("selection evidence access receipt mismatch")
    evaluation_evidence = _require_exact_keys(
        receipt["evaluation_evidence"],
        {
            "implementation",
            "checkpoints",
            "recomputed_records_sha256",
            "evidence_digest_sha256",
        },
        label="selection evaluation evidence",
    )
    _require_sha(
        evaluation_evidence["recomputed_records_sha256"],
        label="selection recomputed records SHA",
    )
    _require_sha(
        evaluation_evidence["evidence_digest_sha256"],
        label="selection evaluation evidence SHA",
    )
    for label, value in (
        ("selected checkpoint tree SHA", selection["checkpoint_tree_sha256"]),
        ("selected adapter tree SHA", selection["adapter_tree_sha256"]),
        (
            "selected stable checkpoint tree SHA",
            selection["stable_tree_digest_sha256"],
        ),
    ):
        _require_sha(value, label=label)
    expected_selection_binding = canonical_sha256(
        {
            "schema": SELECTION_SCHEMA,
            "version": SELECTION_VERSION,
            "manifest_sha256": receipt["manifest"]["sha256"],
            "preblind_commitment_sha256": commitment_sha,
            "training_receipt_sha256": receipt["training_receipt"]["sha256"],
            "evaluation_receipt_sha256": receipt["evaluation_receipt"]["sha256"],
            "training_authority_sha256": canonical_sha256(
                receipt["training_authority"]
            ),
            "evaluation_evidence_sha256": evaluation_evidence[
                "evidence_digest_sha256"
            ],
            "base_model_tree_sha256": receipt["base_model"]["tree_sha256"],
            "base_model_stable_tree_sha256": receipt["base_model"][
                "stable_tree_digest_sha256"
            ],
            "selected_checkpoint_id": selection["checkpoint_id"],
            "selected_checkpoint_tree_sha256": selection[
                "checkpoint_tree_sha256"
            ],
            "selected_adapter_tree_sha256": selection["adapter_tree_sha256"],
            "selected_checkpoint_stable_tree_sha256": selection[
                "stable_tree_digest_sha256"
            ],
            "selection_policy_version": selection_policy_v6.POLICY_VERSION,
            "calibration_authorized": True,
            "blind_test_authorized": False,
        }
    )
    if (
        receipt["selection_binding_digest_sha256"]
        != expected_selection_binding
    ):
        raise ContractsV7Error("selection binding digest mismatch")
    binding = {
        "selection_freeze_sha256": snapshot.sha256,
        "selection_binding_digest_sha256": _require_sha(
            receipt["selection_binding_digest_sha256"],
            label="selection binding digest",
        ),
        "selected_checkpoint_id": selection["checkpoint_id"],
        "selected_checkpoint_tree_sha256": _require_sha(
            selection.get("checkpoint_tree_sha256"),
            label="selected checkpoint tree SHA",
        ),
        "selected_adapter_tree_sha256": _require_sha(
            selection.get("adapter_tree_sha256"),
            label="selected adapter tree SHA",
        ),
        "selected_checkpoint_stable_tree_sha256": _require_sha(
            selection.get("stable_tree_digest_sha256"),
            label="selected stable checkpoint tree SHA",
        ),
        "base_model_tree_sha256": _require_sha(
            receipt["base_model"]["tree_sha256"],
            label="base model tree SHA",
        ),
        "base_model_stable_tree_sha256": _require_sha(
            receipt["base_model"]["stable_tree_digest_sha256"],
            label="base model stable tree SHA",
        ),
        "training_receipt_sha256": _require_sha(
            receipt["training_receipt"]["sha256"],
            label="training receipt SHA",
        ),
        "evaluation_receipt_sha256": _require_sha(
            receipt["evaluation_receipt"]["sha256"],
            label="evaluation receipt SHA",
        ),
        "evaluation_evidence_sha256": _require_sha(
            evaluation_evidence["evidence_digest_sha256"],
            label="evaluation evidence SHA",
        ),
        "preblind_file_sha256": _require_sha(
            preblind["sha256"],
            label="preblind file SHA",
        ),
        "preblind_file_bytes": preblind["bytes"],
        "preblind_commitment_sha256": commitment_sha,
    }
    return snapshot, receipt, binding


def _preblind_binding(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_file_bytes: int,
    expected_commitment_sha256: str,
) -> StableSnapshot:
    snapshot = _snapshot(path, label="preblind commitment")
    if (
        snapshot.sha256 != expected_file_sha256
        or len(snapshot.payload) != expected_file_bytes
    ):
        raise ContractsV7Error("preblind commitment file binding mismatch")
    receipt = _load_json(snapshot, label="preblind commitment")
    expected_fields = {
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
    _require_exact_keys(receipt, expected_fields, label="preblind commitment")
    body = dict(receipt)
    recorded = _require_sha(
        body.pop("commitment_sha256"),
        label="preblind commitment SHA",
    )
    if recorded != expected_commitment_sha256 or canonical_sha256(body) != recorded:
        raise ContractsV7Error("preblind commitment does not match selection")
    return snapshot


def _authority(binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "selection_freeze_sha256": binding["selection_freeze_sha256"],
        "selection_binding_digest_sha256": binding[
            "selection_binding_digest_sha256"
        ],
        "preblind_file_sha256": binding["preblind_file_sha256"],
        "preblind_commitment_sha256": binding[
            "preblind_commitment_sha256"
        ],
        "training_receipt_sha256": binding["training_receipt_sha256"],
        "evaluation_receipt_sha256": binding["evaluation_receipt_sha256"],
        "evaluation_evidence_sha256": binding["evaluation_evidence_sha256"],
        "base_model_tree_sha256": binding["base_model_tree_sha256"],
        "base_model_stable_tree_sha256": binding[
            "base_model_stable_tree_sha256"
        ],
        "selected_checkpoint_id": binding["selected_checkpoint_id"],
        "selected_checkpoint_tree_sha256": binding[
            "selected_checkpoint_tree_sha256"
        ],
        "selected_adapter_tree_sha256": binding[
            "selected_adapter_tree_sha256"
        ],
        "selected_checkpoint_stable_tree_sha256": binding[
            "selected_checkpoint_stable_tree_sha256"
        ],
    }


def _task_contract(
    *,
    contract_id: str,
    created_at: str,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": TASK_SCHEMA,
        "version": VERSION,
        "contract_id": contract_id,
        "created_at": created_at,
        "status": "FROZEN_NONBLIND_V7_TASK",
        "model_product": "ICMat-Qwen-0.5B",
        "tasks": list(evidence_sft_v6.TASKS),
        "researcher_selection": {
            "model_selected_explicitly": True,
            "task_selected_explicitly": True,
            "hidden_task_router": False,
            "automatic_cross_task_routing": False,
        },
        "pointer_output": {
            "schema": evidence_pointer_v6.POINTER_SCHEMA,
            "ordered_fields": list(evidence_sft_v6.POINTER_FIELDS),
            "decisions": list(evidence_sft_v6.DECISIONS),
            "answer_span_pattern": "E#.S#",
            "refuse_span_id": None,
            "additional_fields_allowed": False,
        },
        "evidence_compiler": {
            "version": evidence_sft_v6.COMPILER_VERSION,
            "deterministic": True,
            "fail_closed": True,
            "target_free_input": True,
            "answer_schema": evidence_sft_v6.EXTERNAL_ANSWER_SCHEMA,
            "answer_fields": list(evidence_sft_v6.EXTERNAL_ANSWER_FIELDS),
        },
        "authority": _authority(binding),
        "claim_boundary": (
            "The task is literature-evidence pointer generation. Published "
            "evidence is not local measurement, and this contract grants no "
            "equipment, production, X5, deployment, or BPU authority."
        ),
    }


def _preprocessing_contract(
    *,
    contract_id: str,
    created_at: str,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": PREPROCESSING_SCHEMA,
        "version": VERSION,
        "contract_id": contract_id,
        "created_at": created_at,
        "status": "FROZEN_NONBLIND_V7_PREPROCESSING",
        "prompt_visibility": {
            "messages_visible_to_generation": ["system", "user"],
            "assistant_target_visible_to_generation": False,
            "structured_evidence_target_free": True,
            "add_generation_prompt": True,
        },
        "decoding": {
            "algorithm": "greedy",
            "do_sample": False,
            "num_beams": 1,
            "batch_size": 1,
            "seed": 20260729,
            "max_input_tokens": 1536,
            "max_new_tokens": 64,
        },
        "split_policy": {
            "training_splits": ["train", "validation"],
            "calibration_stage": "POST_SELECTION_FREEZE_COMPLETE_SPLIT_ONLY",
            "calibration_may_fit_model_parameters": False,
            "calibration_may_reselect_checkpoint": False,
            "reserved_evaluation_requires_separate_one_shot_authorization": True,
            "reserved_evaluation_may_reselect_checkpoint": False,
        },
        "runtime_policy": {
            "training_backend": "local Transformers NF4 QLoRA",
            "intended_release_backend": "local llama.cpp CPU GGUF",
            "compiler_runs_after_pointer_generation": True,
            "bpu_conversion_target": False,
            "bpu_runtime_claim_allowed": False,
        },
        "authority": _authority(binding),
        "claim_boundary": (
            "This freezes deterministic preprocessing and access policy only; "
            "it does not prove quality, parity, X5 execution, BPU execution, "
            "deployment, or production integration."
        ),
    }


def _decision_policy_contract(
    *,
    contract_id: str,
    created_at: str,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": DECISION_POLICY_SCHEMA,
        "version": VERSION,
        "contract_id": contract_id,
        "created_at": created_at,
        "status": "FROZEN_NONBLIND_V7_DECISION_POLICY",
        "selection": {
            "policy_schema": selection_policy_v6.SCHEMA,
            "policy_version": selection_policy_v6.POLICY_VERSION,
            "checkpoint_count": selection_policy_v6.EXPECTED_CHECKPOINT_COUNT,
            "seed_count": selection_policy_v6.EXPECTED_SEED_COUNT,
            "epochs": sorted(selection_policy_v6.EXPECTED_EPOCHS),
            "validation_rows_per_checkpoint": (
                selection_policy_v6.EXPECTED_VALIDATION_SAMPLES
            ),
            "minimum_qualified_seeds": (
                selection_policy_v6.MIN_QUALIFIED_SEEDS
            ),
            "floating_point_weighted_score_used": False,
            "integer_cross_multiplication_for_rates": True,
            "checkpoint_reselection_after_freeze": False,
        },
        "post_selection": {
            "complete_calibration_allowed": True,
            "calibration_can_reselect": False,
            "validation_ablation_allowed": True,
            "reserved_evaluation_authorized": False,
            "gguf_export_authorized": False,
            "deployment_authorized": False,
        },
        "authority": _authority(binding),
        "claim_boundary": (
            "The decision policy freezes validation-only checkpoint selection. "
            "No later calibration, ablation, reserved evaluation, export, or "
            "deployment result may change the selected checkpoint."
        ),
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


def _artifact(filename: str, payload: bytes) -> dict[str, Any]:
    return {
        "path": filename,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _payloads(
    *,
    contract_id: str,
    created_at: str,
    binding: Mapping[str, Any],
) -> dict[str, bytes]:
    task = _json_bytes(
        _task_contract(
            contract_id=contract_id,
            created_at=created_at,
            binding=binding,
        )
    )
    preprocessing = _json_bytes(
        _preprocessing_contract(
            contract_id=contract_id,
            created_at=created_at,
            binding=binding,
        )
    )
    decision = _json_bytes(
        _decision_policy_contract(
            contract_id=contract_id,
            created_at=created_at,
            binding=binding,
        )
    )
    artifacts = {
        "task": _artifact(TASK_FILENAME, task),
        "preprocessing": _artifact(PREPROCESSING_FILENAME, preprocessing),
        "decision_policy": _artifact(DECISION_POLICY_FILENAME, decision),
    }
    receipt = {
        "schema": BUILD_RECEIPT_SCHEMA,
        "version": VERSION,
        "contract_id": contract_id,
        "created_at": created_at,
        "status": "PASS_NONBLIND_V7_CONTRACTS_CREATED",
        "artifacts": artifacts,
        "contract_set_sha256": canonical_sha256(
            {
                role: record["sha256"]
                for role, record in sorted(artifacts.items())
            }
        ),
        "authority": _authority(binding),
        "execution_boundary": {
            "dataset_manifest_opened": True,
            "dataset_split_path_constructed": False,
            "dataset_split_filesystem_metadata_accessed": False,
            "calibration_content_accessed": False,
            "blind_content_accessed": False,
            "model_executed": False,
            "x5_accessed": False,
            "bpu_claim_allowed": False,
        },
    }
    return {
        TASK_FILENAME: task,
        PREPROCESSING_FILENAME: preprocessing,
        DECISION_POLICY_FILENAME: decision,
        BUILD_RECEIPT_FILENAME: _json_bytes(receipt),
    }


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _directory_identity(
    path: Path,
    *,
    label: str,
) -> tuple[Path, tuple[int, int]]:
    lexical = _assert_no_reparse_chain(path, label=label)
    metadata = os.lstat(lexical)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise ContractsV7Error(f"{label}: real directory required")
    return lexical.resolve(strict=True), (
        int(metadata.st_dev),
        int(metadata.st_ino),
    )


def _recheck_directory_identity(
    path: Path,
    expected: tuple[int, int],
    *,
    label: str,
) -> None:
    _, current = _directory_identity(path, label=label)
    if current != expected:
        raise ContractsV7Error(f"{label}: parent replacement detected")


def _cleanup_contract_directory(
    path: Path,
    *,
    expected_payloads: Mapping[str, bytes],
) -> None:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise ContractsV7Error(
            f"refusing to clean non-directory contract output: {path}"
        )
    names = {entry.name for entry in path.iterdir()}
    if names != set(expected_payloads):
        raise ContractsV7Error(
            f"refusing to clean contract output with unexpected files: {path}"
        )
    for filename, payload in expected_payloads.items():
        snapshot = _snapshot(
            path / filename,
            label=f"failed contract cleanup {filename}",
        )
        if snapshot.payload != payload:
            raise ContractsV7Error(
                f"refusing to clean changed contract output: {filename}"
            )
    shutil.rmtree(path)


@_authority_lease_scope()
def build_contracts_v7(
    *,
    selection_freeze: Path,
    preblind_commitment: Path,
    evaluation_index: Path,
    training_receipt: Path,
    dataset_dir: Path,
    base_model_dir: Path,
    output_dir: Path,
    contract_id: str,
    created_at: str,
) -> dict[str, Any]:
    """Atomically create a four-file contract directory."""

    _validate_metadata(contract_id, created_at)
    selection_snapshot, _, binding = _selection_binding(
        selection_freeze,
        evaluation_index=evaluation_index,
        training_receipt=training_receipt,
        dataset_dir=dataset_dir,
        base_model_dir=base_model_dir,
    )
    commitment_snapshot = _preblind_binding(
        preblind_commitment,
        expected_file_sha256=binding["preblind_file_sha256"],
        expected_file_bytes=binding["preblind_file_bytes"],
        expected_commitment_sha256=binding["preblind_commitment_sha256"],
    )
    payloads = _payloads(
        contract_id=contract_id,
        created_at=created_at,
        binding=binding,
    )
    output = _assert_no_reparse_chain(output_dir, label="contract output")
    output.parent.mkdir(parents=True, exist_ok=True)
    output_parent, parent_identity = _directory_identity(
        output.parent,
        label="contract output parent",
    )
    parent_anchor = _DirectoryAnchor(output_parent)
    output = output_parent / output.name
    if os.path.lexists(output):
        raise ContractsV7Error(f"output directory already exists: {output}")
    staging = output.with_name(f".{output.name}.staging-{uuid4().hex}")
    staging.mkdir(exist_ok=False)
    published = False
    try:
        for filename, payload in payloads.items():
            _write_exclusive(staging / filename, payload)
        with _lease_exclusion(staging):
            verify_contracts_v7(
                selection_freeze=selection_freeze,
                preblind_commitment=preblind_commitment,
                evaluation_index=evaluation_index,
                training_receipt=training_receipt,
                dataset_dir=dataset_dir,
                base_model_dir=base_model_dir,
                contract_dir=staging,
            )
        if (
            _snapshot(selection_snapshot.path, label="selection recheck")
            != selection_snapshot
            or _snapshot(
                commitment_snapshot.path,
                label="commitment recheck",
            )
            != commitment_snapshot
        ):
            raise ContractsV7Error("authority input changed before publication")
        _recheck_directory_identity(
            output_parent,
            parent_identity,
            label="contract output parent",
        )
        if os.path.lexists(output):
            raise ContractsV7Error(f"output directory already exists: {output}")
        os.rename(staging, output)
        published = True
        _recheck_directory_identity(
            output_parent,
            parent_identity,
            label="contract output parent after publication",
        )
        with _lease_exclusion(output):
            verification = verify_contracts_v7(
                selection_freeze=selection_freeze,
                preblind_commitment=preblind_commitment,
                evaluation_index=evaluation_index,
                training_receipt=training_receipt,
                dataset_dir=dataset_dir,
                base_model_dir=base_model_dir,
                contract_dir=output,
            )
    except BaseException:
        anchored_staging = parent_anchor.child(staging.name)
        if os.path.lexists(anchored_staging):
            with _lease_exclusion(anchored_staging):
                _cleanup_contract_directory(
                    anchored_staging,
                    expected_payloads=payloads,
                )
        anchored_output = parent_anchor.child(output.name)
        if published and os.path.lexists(anchored_output):
            with _lease_exclusion(anchored_output):
                _cleanup_contract_directory(
                    anchored_output,
                    expected_payloads=payloads,
                )
        raise
    finally:
        try:
            anchored_staging = parent_anchor.child(staging.name)
            if os.path.lexists(anchored_staging):
                with _lease_exclusion(anchored_staging):
                    _cleanup_contract_directory(
                        anchored_staging,
                        expected_payloads=payloads,
                    )
        finally:
            parent_anchor.close()
    return {
        "status": "PASS_NONBLIND_V7_CONTRACTS_CREATED",
        "contract_id": contract_id,
        "output_dir": str(output.resolve(strict=True)),
        "contract_set_sha256": verification["contract_set_sha256"],
        "preblind_commitment_sha256": binding[
            "preblind_commitment_sha256"
        ],
        "verified": True,
        "calibration_content_accessed": False,
        "blind_content_accessed": False,
    }


@_authority_lease_scope()
def verify_contracts_v7(
    *,
    selection_freeze: Path,
    preblind_commitment: Path,
    evaluation_index: Path,
    training_receipt: Path,
    dataset_dir: Path,
    base_model_dir: Path,
    contract_dir: Path,
) -> dict[str, Any]:
    """Recompute all contract bytes without opening any dataset split."""

    selection_snapshot, _, binding = _selection_binding(
        selection_freeze,
        evaluation_index=evaluation_index,
        training_receipt=training_receipt,
        dataset_dir=dataset_dir,
        base_model_dir=base_model_dir,
    )
    commitment_snapshot = _preblind_binding(
        preblind_commitment,
        expected_file_sha256=binding["preblind_file_sha256"],
        expected_file_bytes=binding["preblind_file_bytes"],
        expected_commitment_sha256=binding["preblind_commitment_sha256"],
    )
    directory, contract_directory_identity = _directory_identity(
        contract_dir,
        label="contract directory",
    )
    names = {path.name for path in directory.iterdir()}
    if names != CONTRACT_FILENAMES:
        raise ContractsV7Error("contract directory file whitelist mismatch")
    receipt_snapshot = _snapshot(
        directory / BUILD_RECEIPT_FILENAME,
        label="contract build receipt",
    )
    receipt = _load_json(receipt_snapshot, label="contract build receipt")
    expected_receipt_fields = {
        "schema",
        "version",
        "contract_id",
        "created_at",
        "status",
        "artifacts",
        "contract_set_sha256",
        "authority",
        "execution_boundary",
    }
    _require_exact_keys(
        receipt,
        expected_receipt_fields,
        label="contract build receipt",
    )
    if (
        receipt["schema"] != BUILD_RECEIPT_SCHEMA
        or receipt["version"] != VERSION
        or receipt["status"] != "PASS_NONBLIND_V7_CONTRACTS_CREATED"
    ):
        raise ContractsV7Error("contract build receipt identity mismatch")
    _validate_metadata(receipt["contract_id"], receipt["created_at"])
    expected = _payloads(
        contract_id=receipt["contract_id"],
        created_at=receipt["created_at"],
        binding=binding,
    )
    for filename, payload in expected.items():
        observed = _snapshot(
            directory / filename,
            label=f"contract artifact {filename}",
        )
        if observed.payload != payload:
            raise ContractsV7Error(f"{filename}: frozen bytes changed")
    if (
        _snapshot(selection_snapshot.path, label="selection final recheck")
        != selection_snapshot
        or _snapshot(
            commitment_snapshot.path,
            label="commitment final recheck",
        )
        != commitment_snapshot
    ):
        raise ContractsV7Error("authority input changed during verification")
    _recheck_directory_identity(
        directory,
        contract_directory_identity,
        label="contract directory final",
    )
    if {path.name for path in directory.iterdir()} != CONTRACT_FILENAMES:
        raise ContractsV7Error(
            "contract directory file whitelist changed during verification"
        )
    return {
        "status": "PASS_NONBLIND_V7_CONTRACTS_VERIFIED",
        "contract_id": receipt["contract_id"],
        "contract_dir": str(directory.resolve(strict=True)),
        "contract_set_sha256": receipt["contract_set_sha256"],
        "selection_freeze_sha256": binding["selection_freeze_sha256"],
        "preblind_commitment_sha256": binding[
            "preblind_commitment_sha256"
        ],
        "dataset_manifest_opened": True,
        "dataset_split_filesystem_metadata_accessed": False,
        "calibration_content_accessed": False,
        "blind_content_accessed": False,
        "model_executed": False,
        "x5_accessed": False,
    }


__all__ = [
    "BUILD_RECEIPT_FILENAME",
    "CONTRACT_FILENAMES",
    "DECISION_POLICY_FILENAME",
    "PREPROCESSING_FILENAME",
    "TASK_FILENAME",
    "ContractsV7Error",
    "build_contracts_v7",
    "verify_contracts_v7",
]
