"""Immutable task and preprocessing contracts for ICMat Pointer v6.

The contract builder reads the dataset manifest and hashes the five fixed v6
implementation modules. It deliberately does not open any dataset split. The
two generated contracts therefore bind the algorithm and data declaration
without consuming calibration or blind-test content.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

TASK_CONTRACT_SCHEMA = "icmat_qwen_pointer_task_contract.v6"
PREPROCESSING_CONTRACT_SCHEMA = "icmat_qwen_pointer_preprocessing_contract.v6"
BUILD_RECEIPT_SCHEMA = "icmat_qwen_pointer_contract_build_receipt.v6"
CONTRACT_IMPLEMENTATION_VERSION = "icmat-qwen-pointer-contracts-v6.0.0"

TASK_CONTRACT_FILENAME = "task_contract.v6.json"
PREPROCESSING_CONTRACT_FILENAME = "preprocessing_contract.v6.json"
BUILD_RECEIPT_FILENAME = "build_receipt.v6.json"
CONTRACT_FILENAMES = frozenset(
    {
        TASK_CONTRACT_FILENAME,
        PREPROCESSING_CONTRACT_FILENAME,
        BUILD_RECEIPT_FILENAME,
    }
)

DEFAULT_DATASET_MANIFEST = Path(
    "evaluation/icmat_foundry/llm/icmat_qwen05b_evidence_pointer_sft_v6_20260730_r3/manifest.v6.json"
)

SOURCE_PATHS = {
    "dataset_builder": Path("icmat_foundry/llm/evidence_sft_v6.py"),
    "evidence_pointer_compiler": Path("icmat_foundry/llm/evidence_pointer_v6.py"),
    "qlora_trainer": Path("icmat_foundry/llm/qlora_full_v6.py"),
    "hf_evaluator": Path("icmat_foundry/llm/pointer_hf_eval_v6.py"),
    "selection_policy": Path("icmat_foundry/llm/selection_policy_v6.py"),
}

MODEL_PRODUCT = "Qwen2.5-0.5B-Instruct"
TASKS = (
    "claim_verification",
    "evidence_selection",
    "claim_extraction",
)
POINTER_FIELDS = ("task", "decision", "span_id")
ANSWER_FIELDS = (
    "schema",
    "decision",
    "task",
    "claim",
    "verdict",
    "evidence_ids",
    "provenance",
)
DECISIONS = ("ANSWER", "REFUSE")
EXPECTED_SPLIT_COUNTS = {
    "train": 250,
    "validation": 150,
    "calibration": 150,
    "blind_test": 150,
}

DATASET_MANIFEST_SCHEMA = "icmat_evidence_pointer_manifest.v6"
DATASET_SCHEMA = "icmat_qwen05b_evidence_pointer_sft.v6"
DATASET_BUILDER_VERSION = "icmat-evidence-sft-v6.0.0"
COMPILER_VERSION = "icmat-evidence-pointer-compiler-v6.1.0"
POINTER_SCHEMA = "icmat_evidence_pointer.v6"
PROMPT_SCHEMA = "icmat_evidence_pointer_prompt.v6"
ANSWER_SCHEMA = "icmat_student_answer.v6"

MAX_INPUT_TOKENS = 1536
MAX_NEW_TOKENS = 64
DECODING_SEED = 20260729

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTRACT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{7,127}$")


class ContractsV6Error(ValueError):
    """Raised when a v6 contract or one of its immutable inputs is invalid."""


@dataclass(frozen=True)
class ContractPathsV6:
    """Paths of one exclusively created contract set."""

    output_dir: Path
    task_contract: Path
    preprocessing_contract: Path
    build_receipt: Path


def canonical_json(value: Any) -> str:
    """Return the canonical JSON representation used by contract digests."""

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


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractsV6Error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ContractsV6Error(f"{label} contains non-finite number: {token}")
            ),
        )
    except UnicodeDecodeError as exc:
        raise ContractsV6Error(f"{label} is not UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise ContractsV6Error(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractsV6Error(f"{label} must be a JSON object")
    return value


def _resolve_workspace(workspace_root: Path) -> Path:
    workspace = workspace_root.expanduser().resolve(strict=True)
    if not workspace.is_dir():
        raise ContractsV6Error("workspace_root must be a directory")
    return workspace


def _resolve_under_workspace(
    workspace: Path,
    candidate: Path,
    *,
    label: str,
) -> tuple[Path, str]:
    absolute = candidate.expanduser()
    if not absolute.is_absolute():
        absolute = workspace / absolute
    absolute = absolute.resolve(strict=True)
    try:
        relative = absolute.relative_to(workspace).as_posix()
    except ValueError as exc:
        raise ContractsV6Error(f"{label} must stay inside workspace_root") from exc
    if not absolute.is_file():
        raise ContractsV6Error(f"{label} must be a file")
    return absolute, relative


def _validate_contract_metadata(contract_id: str, created_at: str) -> None:
    if not isinstance(contract_id, str) or not _CONTRACT_ID_RE.fullmatch(contract_id):
        raise ContractsV6Error("contract_id has an invalid format")
    if not isinstance(created_at, str) or created_at.strip() != created_at:
        raise ContractsV6Error("created_at must be an exact ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractsV6Error("created_at must be valid ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ContractsV6Error("created_at must include a timezone")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _require_exact(value: Any, expected: Any, *, label: str) -> None:
    if value != expected:
        raise ContractsV6Error(f"{label} does not match the frozen v6 contract")


def _validate_split_declarations(manifest: Mapping[str, Any]) -> None:
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or set(splits) != set(EXPECTED_SPLIT_COUNTS):
        raise ContractsV6Error("dataset manifest split declarations are invalid")
    expected_names = {
        "train": "train.jsonl",
        "validation": "validation.jsonl",
        "calibration": "calibration.jsonl",
        "blind_test": "blind_test.sealed.v6.jsonl",
    }
    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        receipt = splits.get(split)
        if not isinstance(receipt, dict):
            raise ContractsV6Error(f"{split} split receipt must be an object")
        if receipt.get("path") != expected_names[split]:
            raise ContractsV6Error(f"{split} split path is invalid")
        if receipt.get("count") != expected_count:
            raise ContractsV6Error(f"{split} split count is invalid")
        size = receipt.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ContractsV6Error(f"{split} split byte count is invalid")
        if not _is_sha256(receipt.get("sha256")):
            raise ContractsV6Error(f"{split} split SHA-256 is invalid")


def _validate_dataset_manifest(
    manifest: Mapping[str, Any],
    *,
    dataset_builder_sha256: str,
) -> None:
    _require_exact(
        manifest.get("schema"),
        DATASET_MANIFEST_SCHEMA,
        label="dataset manifest schema",
    )
    _require_exact(
        manifest.get("dataset_schema"),
        DATASET_SCHEMA,
        label="dataset schema",
    )
    _require_exact(
        manifest.get("builder_version"),
        DATASET_BUILDER_VERSION,
        label="dataset builder version",
    )
    _require_exact(
        manifest.get("status"),
        "DATASET_BUILT_BLIND_HASH_SEALED",
        label="dataset status",
    )
    _require_exact(
        manifest.get("selection_policy"),
        "researcher_explicit_domain_and_task",
        label="researcher selection policy",
    )
    _require_exact(
        manifest.get("pointer_contract"),
        {
            "field_order": list(POINTER_FIELDS),
            "answer_span_pattern": "E#.S#",
            "refusal_span_id": None,
        },
        label="pointer contract",
    )
    _require_exact(
        manifest.get("external_answer_contract"),
        {
            "schema": ANSWER_SCHEMA,
            "field_order": list(ANSWER_FIELDS),
            "generated_by": "later_deterministic_evidence_compiler",
            "implemented_by_this_builder": False,
        },
        label="external answer contract",
    )
    compiler_input = manifest.get("compiler_input_contract")
    if not isinstance(compiler_input, dict):
        raise ContractsV6Error("compiler input contract must be an object")
    _require_exact(
        compiler_input.get("compiler_version"),
        COMPILER_VERSION,
        label="compiler version",
    )
    _require_exact(
        compiler_input.get("prompt_schema"),
        PROMPT_SCHEMA,
        label="compiler prompt schema",
    )
    _require_exact(
        compiler_input.get("target_free"),
        True,
        label="target-free compiler input",
    )
    _require_exact(
        compiler_input.get("user_text_reverse_parsing_required"),
        False,
        label="compiler structured-input policy",
    )
    _require_exact(
        manifest.get("training_boundary"),
        {
            "allowed_splits": ["train", "validation"],
            "calibration_content_for_training": False,
            "forbidden_split": "blind_test",
            "blind_test_requires_explicit_post_freeze_authorization": True,
            "blind_test_content_in_public_reports": False,
        },
        label="training data boundary",
    )
    _require_exact(
        manifest.get("counts"),
        {
            "examples": 700,
            "families": 14,
            "examples_per_family": 50,
            "splits": EXPECTED_SPLIT_COUNTS,
        },
        label="dataset counts",
    )
    claims = manifest.get("claims")
    if not isinstance(claims, dict):
        raise ContractsV6Error("dataset claims must be an object")
    for key, expected in {
        "knowledge_distillation": False,
        "licensed_evidence_sft": True,
        "local_measurement": False,
        "production_connected": False,
        "x5_verified": False,
    }.items():
        _require_exact(claims.get(key), expected, label=f"dataset claim {key}")
    builder = manifest.get("builder")
    if not isinstance(builder, dict):
        raise ContractsV6Error("dataset builder binding must be an object")
    _require_exact(
        builder.get("sha256"),
        dataset_builder_sha256,
        label="dataset builder SHA-256",
    )
    builder_path = builder.get("path")
    if not isinstance(builder_path, str) or not builder_path.replace("\\", "/").endswith(
        SOURCE_PATHS["dataset_builder"].as_posix()
    ):
        raise ContractsV6Error("dataset builder path is invalid")
    _validate_split_declarations(manifest)


def _file_binding(path: Path, relative: str) -> dict[str, Any]:
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _collect_bindings(
    workspace: Path,
    dataset_manifest: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    source_bindings: dict[str, dict[str, Any]] = {}
    for name, relative_path in SOURCE_PATHS.items():
        absolute, relative = _resolve_under_workspace(
            workspace,
            relative_path,
            label=name,
        )
        source_bindings[name] = _file_binding(absolute, relative)

    manifest_path, manifest_relative = _resolve_under_workspace(
        workspace,
        dataset_manifest,
        label="dataset_manifest",
    )
    manifest = _load_json_object(manifest_path, label="dataset manifest")
    _validate_dataset_manifest(
        manifest,
        dataset_builder_sha256=source_bindings["dataset_builder"]["sha256"],
    )
    manifest_binding = _file_binding(manifest_path, manifest_relative)
    manifest_binding.update(
        {
            "schema": manifest["schema"],
            "dataset_schema": manifest["dataset_schema"],
            "status": manifest["status"],
            "builder_version": manifest["builder_version"],
            "split_counts": dict(EXPECTED_SPLIT_COUNTS),
            "split_content_read": False,
        }
    )
    return manifest_binding, source_bindings, manifest


def _binding_block(
    manifest_binding: Mapping[str, Any],
    source_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "dataset_manifest": dict(manifest_binding),
        "implementation_sources": {name: dict(source_bindings[name]) for name in SOURCE_PATHS},
    }


def _task_contract(
    *,
    contract_id: str,
    created_at: str,
    manifest_binding: Mapping[str, Any],
    source_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": TASK_CONTRACT_SCHEMA,
        "contract_version": CONTRACT_IMPLEMENTATION_VERSION,
        "contract_id": contract_id,
        "created_at": created_at,
        "status": "FROZEN_BEFORE_CALIBRATION_AND_BLIND",
        "model": {
            "product": MODEL_PRODUCT,
            "role": "evidence_pointer_model",
            "parameter_scale": "0.5B",
            "researcher_selected_explicitly": True,
            "hidden_model_router": False,
        },
        "tasks": list(TASKS),
        "researcher_selection": {
            "model_selected_explicitly": True,
            "task_selected_explicitly": True,
            "hidden_task_router": False,
            "automatic_cross_task_routing": False,
        },
        "model_output_contract": {
            "format": "one ordered JSON object and no Markdown or prose",
            "schema_value": POINTER_SCHEMA,
            "exact_ordered_keys": list(POINTER_FIELDS),
            "additional_keys_allowed": False,
            "decisions": list(DECISIONS),
            "answer_span_id_pattern": "E#.S#",
            "refuse_span_id": None,
            "claim_or_provenance_generated_by_model": False,
        },
        "evidence_pointer_compiler": {
            "name": "Evidence Pointer Compiler",
            "version": COMPILER_VERSION,
            "deterministic": True,
            "fail_closed": True,
            "input_is_target_free": True,
            "output_schema": ANSWER_SCHEMA,
            "exact_ordered_output_keys": list(ANSWER_FIELDS),
            "answer_claim_source": "exact text selected by a valid E#.S# pointer",
            "refuse_output": {
                "claim": "",
                "verdict": "REFUSED",
                "evidence_ids": [],
            },
        },
        "evidence_boundary": {
            "ground_truth": "deterministic labels from licensed literature evidence",
            "teacher_or_api_output_is_ground_truth": False,
            "knowledge_distillation": False,
            "published_literature_is_local_measurement": False,
            "production_or_equipment_action_allowed": False,
        },
        "runtime_claim_boundary": {
            "intended_x5_runtime": "CPU GGUF",
            "bpu_model": False,
            "bpu_execution_claim_allowed": False,
            "x5_execution_verified_by_this_contract": False,
        },
        "bindings": _binding_block(manifest_binding, source_bindings),
        "claim_boundary": (
            "This contract fixes a literature-evidence pointer task. It does "
            "not prove model quality, BPU execution, X5 execution, local "
            "measurement validity, fab production performance, or autonomous "
            "equipment control."
        ),
    }


def _preprocessing_contract(
    *,
    contract_id: str,
    created_at: str,
    manifest_binding: Mapping[str, Any],
    source_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": PREPROCESSING_CONTRACT_SCHEMA,
        "contract_version": CONTRACT_IMPLEMENTATION_VERSION,
        "contract_id": contract_id,
        "created_at": created_at,
        "status": "FROZEN_BEFORE_CALIBRATION_AND_BLIND",
        "base_model": {
            "product": MODEL_PRODUCT,
            "local_files_only": True,
            "network_allowed": False,
        },
        "prompt": {
            "messages_visible_to_generation": ["system", "user"],
            "assistant_target_visible_to_generation": False,
            "structured_evidence_is_target_free": True,
            "chat_template": "base-model tokenizer chat template",
            "add_generation_prompt": True,
        },
        "decoding": {
            "algorithm": "greedy",
            "do_sample": False,
            "singleton_batch": True,
            "batch_size": 1,
            "seed": DECODING_SEED,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "split_access_policy": {
            "training_readable_splits": ["train", "validation"],
            "training_declaration_only_splits": ["calibration", "blind_test"],
            "calibration": {
                "content_read_stage": "post_selection_freeze_only",
                "eligible_for_parameter_fitting": False,
                "eligible_for_checkpoint_selection": False,
            },
            "blind_test": {
                "content_read_stage": "post_selection_freeze_only",
                "one_time_model_bound_authorization_required": True,
                "authorization_reusable_after_failure": False,
                "eligible_for_parameter_fitting": False,
                "eligible_for_checkpoint_selection": False,
            },
        },
        "selection_policy": {
            "researcher_selects_model_and_task": True,
            "hidden_router": False,
            "selection_uses_complete_validation_generation": True,
            "calibration_may_reselect_checkpoint": False,
            "blind_may_reselect_checkpoint": False,
        },
        "runtime_policy": {
            "hf_backend": "local Transformers pointer generation",
            "release_backend": "local llama.cpp CPU GGUF",
            "gguf_device": "CPU",
            "bpu_conversion_target": False,
            "bpu_runtime_claim_allowed": False,
            "compiler_runs_after_pointer_generation": True,
        },
        "bindings": _binding_block(manifest_binding, source_bindings),
        "claim_boundary": (
            "This preprocessing contract fixes input visibility, split access, "
            "and deterministic generation. It does not prove model quality, "
            "HF/GGUF parity, X5 execution, BPU execution, or production "
            "integration."
        ),
    }


def _artifact_receipt(filename: str, payload: bytes) -> dict[str, Any]:
    return {
        "path": filename,
        "sha256": sha256_bytes(payload),
        "bytes": len(payload),
    }


def _build_payloads(
    *,
    contract_id: str,
    created_at: str,
    manifest_binding: Mapping[str, Any],
    source_bindings: Mapping[str, Mapping[str, Any]],
) -> tuple[bytes, bytes, bytes]:
    task = _task_contract(
        contract_id=contract_id,
        created_at=created_at,
        manifest_binding=manifest_binding,
        source_bindings=source_bindings,
    )
    preprocessing = _preprocessing_contract(
        contract_id=contract_id,
        created_at=created_at,
        manifest_binding=manifest_binding,
        source_bindings=source_bindings,
    )
    task_payload = _json_bytes(task)
    preprocessing_payload = _json_bytes(preprocessing)
    artifacts = {
        "task_contract": _artifact_receipt(
            TASK_CONTRACT_FILENAME,
            task_payload,
        ),
        "preprocessing_contract": _artifact_receipt(
            PREPROCESSING_CONTRACT_FILENAME,
            preprocessing_payload,
        ),
    }
    contract_set_sha256 = sha256_bytes(
        canonical_json({name: receipt["sha256"] for name, receipt in artifacts.items()}).encode("utf-8")
    )
    receipt = {
        "schema": BUILD_RECEIPT_SCHEMA,
        "contract_version": CONTRACT_IMPLEMENTATION_VERSION,
        "contract_id": contract_id,
        "created_at": created_at,
        "status": "PASS_V6_CONTRACTS_CREATED_NO_MODEL_EXECUTION",
        "artifacts": artifacts,
        "contract_set_sha256": contract_set_sha256,
        "dataset_manifest": dict(manifest_binding),
        "implementation_sources": {name: dict(source_bindings[name]) for name in SOURCE_PATHS},
        "execution_boundary": {
            "dataset_split_content_read": False,
            "model_generation_executed": False,
            "training_executed": False,
            "calibration_accessed": False,
            "blind_accessed": False,
            "x5_accessed": False,
            "bpu_claim_allowed": False,
        },
    }
    return task_payload, preprocessing_payload, _json_bytes(receipt)


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def build_contracts_v6(
    *,
    workspace_root: Path,
    dataset_manifest: Path,
    output_dir: Path,
    contract_id: str,
    created_at: str,
) -> dict[str, Any]:
    """Create one immutable contract directory without reading split files."""

    _validate_contract_metadata(contract_id, created_at)
    workspace = _resolve_workspace(workspace_root)
    manifest_binding, source_bindings, _ = _collect_bindings(
        workspace,
        dataset_manifest,
    )
    payloads = _build_payloads(
        contract_id=contract_id,
        created_at=created_at,
        manifest_binding=manifest_binding,
        source_bindings=source_bindings,
    )

    destination = output_dir.expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"output_dir already exists; exclusive create required: {destination}") from exc

    try:
        _write_exclusive(destination / TASK_CONTRACT_FILENAME, payloads[0])
        _write_exclusive(
            destination / PREPROCESSING_CONTRACT_FILENAME,
            payloads[1],
        )
        _write_exclusive(destination / BUILD_RECEIPT_FILENAME, payloads[2])
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise

    paths = ContractPathsV6(
        output_dir=destination,
        task_contract=destination / TASK_CONTRACT_FILENAME,
        preprocessing_contract=destination / PREPROCESSING_CONTRACT_FILENAME,
        build_receipt=destination / BUILD_RECEIPT_FILENAME,
    )
    return {
        "status": "PASS_V6_CONTRACTS_CREATED_NO_MODEL_EXECUTION",
        "contract_id": contract_id,
        "output_dir": paths.output_dir.as_posix(),
        "task_contract": _artifact_receipt(
            paths.task_contract.name,
            payloads[0],
        ),
        "preprocessing_contract": _artifact_receipt(
            paths.preprocessing_contract.name,
            payloads[1],
        ),
        "build_receipt": _artifact_receipt(
            paths.build_receipt.name,
            payloads[2],
        ),
        "dataset_split_content_read": False,
        "model_generation_executed": False,
    }


def verify_contracts_v6(
    *,
    workspace_root: Path,
    dataset_manifest: Path,
    contract_dir: Path,
) -> dict[str, Any]:
    """Recompute and verify a contract set without writing or reading splits."""

    workspace = _resolve_workspace(workspace_root)
    directory = contract_dir.expanduser().resolve(strict=True)
    if not directory.is_dir():
        raise ContractsV6Error("contract_dir must be a directory")
    actual_names = {path.name for path in directory.iterdir()}
    if actual_names != CONTRACT_FILENAMES:
        raise ContractsV6Error("contract_dir must contain exactly three contract files")

    receipt_path = directory / BUILD_RECEIPT_FILENAME
    receipt = _load_json_object(receipt_path, label="build receipt")
    _require_exact(
        receipt.get("schema"),
        BUILD_RECEIPT_SCHEMA,
        label="build receipt schema",
    )
    contract_id = receipt.get("contract_id")
    created_at = receipt.get("created_at")
    _validate_contract_metadata(contract_id, created_at)

    manifest_binding, source_bindings, _ = _collect_bindings(
        workspace,
        dataset_manifest,
    )
    expected_payloads = _build_payloads(
        contract_id=contract_id,
        created_at=created_at,
        manifest_binding=manifest_binding,
        source_bindings=source_bindings,
    )
    expected_by_name = {
        TASK_CONTRACT_FILENAME: expected_payloads[0],
        PREPROCESSING_CONTRACT_FILENAME: expected_payloads[1],
        BUILD_RECEIPT_FILENAME: expected_payloads[2],
    }
    for filename, expected in expected_by_name.items():
        actual = (directory / filename).read_bytes()
        if actual != expected:
            raise ContractsV6Error(f"{filename} does not match recomputed frozen contract bytes")

    return {
        "status": "PASS_V6_CONTRACTS_VERIFIED",
        "contract_id": contract_id,
        "contract_dir": directory.as_posix(),
        "dataset_manifest_sha256": manifest_binding["sha256"],
        "contract_set_sha256": receipt["contract_set_sha256"],
        "source_count": len(source_bindings),
        "dataset_split_content_read": False,
        "model_generation_executed": False,
        "bpu_claim_allowed": False,
    }
