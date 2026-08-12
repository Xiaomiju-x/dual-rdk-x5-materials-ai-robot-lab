"""Offline NF4 QLoRA smoke trainer for ICMat-Qwen-0.5B."""
from __future__ import annotations

import gc
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from .local_teacher import WORKSPACE_ROOT, _read_bound_file
from .sft import iter_jsonl, sha256_file, write_json_atomic
from .sft_v4 import (
    DATASET_SCHEMA_ID as DATASET_SCHEMA_V4,
)
from .sft_v4 import verify_materialized_dataset as verify_v4_dataset

TRAINER_VERSION = "icmat-qwen05b-nf4-qlora-1.6.0"
TRAINING_RECEIPT_SCHEMA = "icmat_qlora_smoke_receipt.v1"
TOKEN_PREFLIGHT_SCHEMA = "icmat_qlora_token_preflight.v1"
PILOT_AUDIT_SCHEMA = "icmat_qlora_pilot_audit.v1"
PILOT_SUBJECT_SCHEMA = "icmat_qlora_pilot_subject.v1"
ALLOWED_OUTPUT_ROOT = (
    WORKSPACE_ROOT / "evaluation" / "icmat_foundry" / "llm"
).resolve()
DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class SmokeConfig:
    max_steps: int = 10
    max_seq_length: int = 512
    max_train_examples: int = 32
    max_eval_examples: int = 16
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2.0e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    seed: int = 20260728
    minimum_free_vram_mib: int = 4200
    save_adapter: bool = True

    def validate(self) -> None:
        integer_fields = {
            "max_steps": self.max_steps,
            "max_seq_length": self.max_seq_length,
            "max_train_examples": self.max_train_examples,
            "max_eval_examples": self.max_eval_examples,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "seed": self.seed,
            "minimum_free_vram_mib": self.minimum_free_vram_mib,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in integer_fields.values()
        ):
            raise TypeError("QLoRA integer configuration fields must be integers")
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
            raise ValueError("QLoRA floating-point fields must be finite")
        if not isinstance(self.save_adapter, bool):
            raise TypeError("save_adapter must be boolean")
        if self.max_steps < 1 or self.max_steps > 10:
            raise PermissionError("QLoRA pilot is limited to 1-10 audited steps")
        if self.max_seq_length < 128 or self.max_seq_length > 4096:
            raise ValueError("max_seq_length must be in [128, 4096]")
        if not 1 <= self.max_train_examples <= 32:
            raise ValueError("pilot train sample limit must be in [1, 32]")
        if not 1 <= self.max_eval_examples <= 16:
            raise ValueError("pilot evaluation sample limit must be in [1, 16]")
        if self.per_device_train_batch_size != 1:
            raise ValueError("RTX4050 6GB contract requires train batch size 1")
        if self.per_device_eval_batch_size != 1:
            raise ValueError("RTX4050 6GB contract requires evaluation batch size 1")
        if not 1 <= self.gradient_accumulation_steps <= 8:
            raise ValueError("gradient accumulation must be in [1, 8]")
        if self.lora_rank not in {4, 8, 16, 32}:
            raise ValueError("unsupported LoRA rank")
        if not 1 <= self.lora_alpha <= 256:
            raise ValueError("LoRA alpha must be in [1, 256]")
        if not 0.0 < float(self.learning_rate) <= 0.01:
            raise ValueError("learning_rate must be in (0, 0.01]")
        if not 0.0 <= float(self.warmup_ratio) <= 0.5:
            raise ValueError("warmup_ratio must be in [0, 0.5]")
        if not 0.0 <= float(self.weight_decay) <= 1.0:
            raise ValueError("weight_decay must be in [0, 1]")
        if not 0.0 <= float(self.lora_dropout) < 0.5:
            raise ValueError("lora_dropout must be in [0, 0.5)")
        if not 0 <= self.seed <= 2_147_483_647:
            raise ValueError("seed must be in [0, 2147483647]")
        if not 1024 <= self.minimum_free_vram_mib <= 6144:
            raise ValueError("minimum_free_vram_mib must be in [1024, 6144]")


def build_assistant_only_labels(
    prefix_ids: Sequence[int],
    full_ids: Sequence[int],
) -> list[int]:
    prefix = list(prefix_ids)
    full = list(full_ids)
    if not prefix or len(prefix) >= len(full):
        raise ValueError("assistant target must add at least one token")
    if full[: len(prefix)] != prefix:
        raise ValueError("chat-template prefix is not a prefix of full conversation")
    return [-100] * len(prefix) + full[len(prefix) :]


def encode_assistant_only(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    max_seq_length: int,
) -> dict[str, Any]:
    if [message.get("role") for message in messages] != [
        "system",
        "user",
        "assistant",
    ]:
        raise ValueError("expected one system/user/assistant conversation")
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
        raise ValueError(
            f"conversation has {len(full_ids)} tokens, above max_seq_length={max_seq_length}"
        )
    labels = build_assistant_only_labels(prefix_ids, full_ids)
    supervised = sum(token != -100 for token in labels)
    return {
        "input_ids": list(full_ids),
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "sequence_tokens": len(full_ids),
        "assistant_tokens": supervised,
    }


def _read_manifest(dataset_dir: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = dataset_dir / "manifest.v4.json"
    legacy_manifests = [
        path
        for path in (
            dataset_dir / "manifest.v1.json",
            dataset_dir / "manifest.v2.json",
        )
        if path.exists()
    ]
    if legacy_manifests:
        raise PermissionError(
            "the finals QLoRA entry accepts only independently audited v4 datasets"
        )
    if not manifest_path.is_file():
        raise ValueError(
            "the finals QLoRA entry requires exactly one manifest.v4.json"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != DATASET_SCHEMA_V4:
        raise ValueError("unexpected v4 SFT dataset schema")
    target_contract = manifest.get("target_contract", {})
    if (
        target_contract.get("assistant_only_loss_required") is not True
        or target_contract.get("teacher_const_schema_excluded") is not True
        or target_contract.get("exact_target_excluded_from_prompt") is not True
    ):
        raise ValueError("dataset does not bind the safe student projection")

    files = manifest.get("files")
    verify_v4_dataset(dataset_dir)
    if not isinstance(files, dict):
        raise ValueError("v4 dataset files contract must be an object")
    records = list(files.get("training", []))
    records.extend(
        files[name]
        for name in (
            "sealed_test_membership",
            "audit_challenge_membership",
        )
        if name in files
    )
    final_test = manifest.get("final_test_contract", {})
    authorization = manifest.get("authorization", {})
    if (
        final_test.get("membership_only") is not True
        or final_test.get("semantic_examples_materialized") is not False
        or final_test.get("semantic_metrics_emitted") is not False
        or (dataset_dir / "test.jsonl").exists()
    ):
        raise ValueError("v4 final test is not a membership-only sealed split")
    if (
        authorization.get("dataset_materialization_authorized") is not True
        or authorization.get("qlora_pilot_authorized") is not False
        or authorization.get("full_training_authorized") is not False
        or authorization.get("bpu_authorized") is not False
        or authorization.get("x5_authorized") is not False
        or authorization.get("production_integration_authorized") is not False
    ):
        raise PermissionError("v4 dataset authorization boundary is unsafe")
    training_paths = {
        str(record.get("path")) for record in files.get("training", [])
    }
    if training_paths != {
        "train.jsonl",
        "validation.jsonl",
        "calibration.jsonl",
    }:
        raise ValueError("v4 training file set is incomplete or unexpected")

    for record in records:
        path = dataset_dir / str(record["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != record.get("bytes") or sha256_file(path) != record.get(
            "sha256"
        ):
            raise ValueError(f"dataset file does not match manifest: {path}")
    return manifest, manifest_path


def _read_model_receipt(model_dir: Path, model_receipt_path: Path) -> dict[str, Any]:
    receipt = json.loads(model_receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "icmat_hf_model_receipt.v1":
        raise ValueError("unexpected model receipt schema")
    if receipt.get("repo_id") != "Qwen/Qwen2.5-0.5B-Instruct":
        raise ValueError("unexpected base model")
    revision = str(receipt.get("revision", ""))
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("base-model revision is not a pinned commit hash")

    expected = {str(item["path"]): item for item in receipt.get("files", [])}
    required = {
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "LICENSE",
    }
    if not required <= set(expected):
        raise ValueError("model receipt omits required files")
    for relative_path, record in expected.items():
        path = model_dir / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != record.get("bytes") or sha256_file(path) != record.get(
            "sha256"
        ):
            raise ValueError(f"base-model file does not match receipt: {relative_path}")
    return receipt


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _has_reparse_or_symlink(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
    except OSError:
        return True
    return bool(attributes & 0x400)


def _safe_new_output_path(path: Path, *, kind: str) -> Path:
    root = ALLOWED_OUTPUT_ROOT.resolve(strict=True)
    if _has_reparse_or_symlink(root):
        raise PermissionError("QLoRA output root must not be a reparse point")
    raw = Path(path)
    if raw.name in {"", ".", ".."}:
        raise ValueError("QLoRA output requires a direct child name")
    parent = raw.parent.resolve(strict=True)
    if parent != root:
        raise PermissionError(
            f"QLoRA output must be a direct child of {root}"
        )
    candidate = root / raw.name
    if os.path.lexists(candidate):
        raise FileExistsError(f"QLoRA output already exists: {candidate}")
    if kind == "file" and candidate.suffix.lower() != ".json":
        raise ValueError("QLoRA preflight output must be a JSON file")
    if kind not in {"file", "directory"}:
        raise ValueError("unknown QLoRA output kind")
    return candidate


def _load_offline_tokenizer(model_dir: Path) -> Any:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )


def _tokenizer_identity(
    tokenizer: Any,
    model_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    tokenizer_files = [
        record
        for record in model_receipt["files"]
        if str(record["path"]).startswith("tokenizer")
        or str(record["path"]) in {"merges.txt", "vocab.json"}
    ]
    return {
        "tokenizer_files": tokenizer_files,
        "chat_template_sha256": _canonical_sha256(tokenizer.chat_template),
        "dependencies": _package_versions(("transformers", "tokenizers")),
    }


def _scan_v4_tokenization(
    *,
    dataset_dir: Path,
    manifest: Mapping[str, Any],
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    seen_example_ids: set[str] = set()
    manifest_training = {
        str(record["path"]): record
        for record in manifest["files"]["training"]
    }
    for split in ("train", "validation", "calibration"):
        relative_path = f"{split}.jsonl"
        path = dataset_dir / relative_path
        count = 0
        for item in iter_jsonl(path):
            messages = item.get("messages")
            if not isinstance(messages, list):
                raise ValueError("v4 example lacks messages")
            example_id = str(item.get("example_id", ""))
            if not example_id:
                raise ValueError("v4 example lacks example_id")
            if example_id in seen_example_ids:
                raise ValueError("v4 token preflight found a duplicate example_id")
            seen_example_ids.add(example_id)
            if item.get("split") != split:
                raise ValueError("v4 example split does not match its file")
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
            labels = build_assistant_only_labels(prefix_ids, full_ids)
            assistant_tokens = sum(token != -100 for token in labels)
            records.append(
                {
                    "example_id": example_id,
                    "split": split,
                    "task": str(item["task"]),
                    "record_sha256": _canonical_sha256(item),
                    "prefix_tokens": len(prefix_ids),
                    "full_tokens": len(full_ids),
                    "assistant_tokens": assistant_tokens,
                }
            )
            count += 1
        expected_count = manifest_training[relative_path].get("examples")
        if count == 0 or count != expected_count:
            raise ValueError(
                f"v4 token preflight count mismatch for {split}: "
                f"{count} != {expected_count}"
            )
        split_counts[split] = count
    return (
        sorted(
            records,
            key=lambda record: (record["split"], record["example_id"]),
        ),
        split_counts,
    )


def _pilot_configuration(config: SmokeConfig) -> dict[str, Any]:
    return {
        **asdict(config),
        "quantization": "NF4",
        "double_quantization": True,
        "compute_dtype": "bfloat16",
        "optimizer": "paged_adamw_8bit",
        "target_modules": list(DEFAULT_TARGET_MODULES),
        "assistant_only_loss": True,
    }


def _model_context_tokens(model_dir: Path) -> int:
    payload = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    value = payload.get("max_position_embeddings")
    if isinstance(value, bool) or not isinstance(value, int) or value < 128:
        raise ValueError("base model lacks a valid max_position_embeddings")
    return value


def build_token_preflight(
    *,
    model_dir: Path,
    model_receipt_path: Path,
    dataset_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Tokenize every v4 training example without truncation and seal the result."""

    model_dir = model_dir.resolve(strict=True)
    dataset_dir = dataset_dir.resolve(strict=True)
    output_path = _safe_new_output_path(output_path, kind="file")
    manifest, manifest_path = _read_manifest(dataset_dir)
    model_receipt_path = model_receipt_path.resolve(strict=True)
    model_receipt = _read_model_receipt(model_dir, model_receipt_path)

    tokenizer = _load_offline_tokenizer(model_dir)
    records, split_counts = _scan_v4_tokenization(
        dataset_dir=dataset_dir,
        manifest=manifest,
        tokenizer=tokenizer,
    )

    maximum = max(record["full_tokens"] for record in records)
    required = int(math.ceil(maximum / 128) * 128)
    model_context = _model_context_tokens(model_dir)
    if required > model_context:
        raise ValueError(
            f"required sequence length {required} exceeds model context {model_context}"
        )
    tokenizer_identity = _tokenizer_identity(tokenizer, model_receipt)
    receipt = {
        "schema": TOKEN_PREFLIGHT_SCHEMA,
        "trainer_version": TRAINER_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_ZERO_TRUNCATION_PREFLIGHT_NOT_TRAINING_AUTHORIZATION",
        "dataset": {
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_path": manifest_path.name,
            "split_files": manifest["files"]["training"],
            "split_counts": split_counts,
        },
        "base_model": {
            "repo_id": model_receipt["repo_id"],
            "revision": model_receipt["revision"],
            "model_receipt_sha256": sha256_file(model_receipt_path),
            "tokenizer_files": tokenizer_identity["tokenizer_files"],
            "chat_template_sha256": tokenizer_identity[
                "chat_template_sha256"
            ],
            "model_context_tokens": model_context,
        },
        "tokenization": {
            "all_training_splits_scanned": True,
            "truncation_used": False,
            "record_count": len(records),
            "minimum_full_tokens": min(record["full_tokens"] for record in records),
            "maximum_full_tokens": maximum,
            "required_max_seq_length": required,
            "records": records,
        },
        "dependencies": tokenizer_identity["dependencies"],
        "authorization": {
            "qlora_pilot_authorized": False,
            "full_training_authorized": False,
            "bpu_authorized": False,
            "x5_authorized": False,
            "production_integration_authorized": False,
        },
    }
    write_json_atomic(output_path, receipt)
    return receipt


def _read_v4_token_preflight(
    *,
    model_dir: Path,
    model_receipt: Mapping[str, Any],
    dataset_dir: Path,
    dataset_manifest: Mapping[str, Any],
    dataset_manifest_path: Path,
    model_receipt_path: Path,
    token_preflight_path: Path | None,
    expected_token_preflight_sha256: str | None,
    config: SmokeConfig,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    if token_preflight_path is None or expected_token_preflight_sha256 is None:
        raise PermissionError("v4 QLoRA requires a bound zero-truncation preflight")
    resolved, payload = _read_bound_file(
        token_preflight_path,
        expected_sha256=expected_token_preflight_sha256,
        workspace_root=WORKSPACE_ROOT,
    )
    report = json.loads(payload.decode("utf-8"))
    tokenizer = _load_offline_tokenizer(model_dir)
    actual_records, actual_split_counts = _scan_v4_tokenization(
        dataset_dir=dataset_dir,
        manifest=dataset_manifest,
        tokenizer=tokenizer,
    )
    actual_maximum = max(
        int(record["full_tokens"]) for record in actual_records
    )
    actual_required = int(math.ceil(actual_maximum / 128) * 128)
    actual_identity = _tokenizer_identity(tokenizer, model_receipt)
    tokenization = report.get("tokenization", {})
    authorization = report.get("authorization", {})
    base_model = report.get("base_model", {})
    dataset = report.get("dataset", {})
    if (
        report.get("schema") != TOKEN_PREFLIGHT_SCHEMA
        or report.get("trainer_version") != TRAINER_VERSION
        or report.get("status")
        != "PASS_ZERO_TRUNCATION_PREFLIGHT_NOT_TRAINING_AUTHORIZATION"
        or dataset.get("manifest_sha256")
        != sha256_file(dataset_manifest_path)
        or dataset.get("manifest_path") != dataset_manifest_path.name
        or dataset.get("split_files")
        != dataset_manifest["files"]["training"]
        or dataset.get("split_counts") != actual_split_counts
        or base_model.get("repo_id") != model_receipt["repo_id"]
        or base_model.get("revision") != model_receipt["revision"]
        or base_model.get("model_receipt_sha256")
        != sha256_file(model_receipt_path)
        or base_model.get("tokenizer_files")
        != actual_identity["tokenizer_files"]
        or base_model.get("chat_template_sha256")
        != actual_identity["chat_template_sha256"]
        or base_model.get("model_context_tokens")
        != _model_context_tokens(model_dir)
        or tokenization.get("all_training_splits_scanned") is not True
        or tokenization.get("truncation_used") is not False
        or tokenization.get("record_count") != len(actual_records)
        or tokenization.get("minimum_full_tokens")
        != min(int(record["full_tokens"]) for record in actual_records)
        or tokenization.get("maximum_full_tokens") != actual_maximum
        or tokenization.get("required_max_seq_length") != actual_required
        or tokenization.get("records") != actual_records
        or report.get("dependencies") != actual_identity["dependencies"]
        or config.max_seq_length != actual_required
        or authorization.get("qlora_pilot_authorized") is not False
        or authorization.get("full_training_authorized") is not False
        or authorization.get("bpu_authorized") is not False
        or authorization.get("x5_authorized") is not False
        or authorization.get("production_integration_authorized") is not False
        or any(
            int(record["full_tokens"]) > config.max_seq_length
            for record in actual_records
        )
    ):
        raise PermissionError("v4 token preflight does not bind this pilot")
    return (
        report,
        {
            "path": resolved.relative_to(WORKSPACE_ROOT).as_posix(),
            "sha256": expected_token_preflight_sha256,
        },
        tokenizer,
    )


def _pilot_audit_subject(
    *,
    dataset_manifest_path: Path,
    model_receipt_path: Path,
    token_preflight_sha256: str,
    input_snapshot_sha256: str,
    output_dir: Path,
    config: SmokeConfig,
) -> dict[str, Any]:
    trainer_path = Path(__file__).resolve()
    cli_path = WORKSPACE_ROOT / "tools" / "train_icmat_qlora.py"
    configuration = _pilot_configuration(config)
    subject_core = {
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "model_receipt_sha256": sha256_file(model_receipt_path),
        "token_preflight_sha256": token_preflight_sha256,
        "input_snapshot_sha256": input_snapshot_sha256,
        "output_path": output_dir.relative_to(WORKSPACE_ROOT).as_posix(),
        "trainer_version": TRAINER_VERSION,
        "trainer_source_sha256": sha256_file(trainer_path),
        "trainer_cli_sha256": sha256_file(cli_path),
        "configuration": configuration,
        "configuration_sha256": _canonical_sha256(configuration),
        "runtime_dependencies": _package_versions(
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
    }
    return {
        **subject_core,
        "run_id": "icmqlp1-" + _canonical_sha256(subject_core),
    }


def _input_snapshot(
    *,
    dataset_dir: Path,
    dataset_manifest: Mapping[str, Any],
    dataset_manifest_path: Path,
    model_dir: Path,
    model_receipt: Mapping[str, Any],
    model_receipt_path: Path,
) -> dict[str, Any]:
    dataset_records = list(dataset_manifest["files"]["training"])
    dataset_records.extend(
        dataset_manifest["files"][name]
        for name in (
            "sealed_test_membership",
            "audit_challenge_membership",
        )
        if name in dataset_manifest["files"]
    )
    files: list[dict[str, Any]] = [
        {
            "role": "dataset_manifest",
            "path": dataset_manifest_path.name,
            "bytes": dataset_manifest_path.stat().st_size,
            "sha256": sha256_file(dataset_manifest_path),
        },
        {
            "role": "model_receipt",
            "path": model_receipt_path.name,
            "bytes": model_receipt_path.stat().st_size,
            "sha256": sha256_file(model_receipt_path),
        },
    ]
    for record in dataset_records:
        path = dataset_dir / str(record["path"])
        files.append(
            {
                "role": "dataset",
                "path": str(record["path"]),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    for record in model_receipt["files"]:
        path = model_dir / str(record["path"])
        files.append(
            {
                "role": "base_model",
                "path": str(record["path"]),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    files.sort(key=lambda record: (record["role"], record["path"]))
    core = {
        "schema": "icmat_qlora_input_snapshot.v1",
        "files": files,
    }
    return {
        **core,
        "snapshot_sha256": _canonical_sha256(core),
    }


def _read_v4_pilot_audit(
    *,
    expected_subject: Mapping[str, Any],
    pilot_audit_path: Path | None,
    expected_pilot_audit_sha256: str | None,
) -> dict[str, Any]:
    if pilot_audit_path is None or expected_pilot_audit_sha256 is None:
        raise PermissionError("v4 QLoRA requires a separate bound pilot GO")
    resolved, payload = _read_bound_file(
        pilot_audit_path,
        expected_sha256=expected_pilot_audit_sha256,
        workspace_root=WORKSPACE_ROOT,
    )
    audit = json.loads(payload.decode("utf-8"))
    if (
        audit.get("schema") != PILOT_AUDIT_SCHEMA
        or audit.get("decision") != "GO"
        or audit.get("revoked") is not False
        or audit.get("scope") != "qlora_pilot"
        or audit.get("blocking_findings") != []
        or audit.get("test_semantics_accessed") is not False
        or audit.get("subject") != dict(expected_subject)
        or audit.get("authorization", {}).get("qlora_pilot") is not True
        or audit.get("authorization", {}).get("full_training") is not False
    ):
        raise PermissionError("v4 pilot audit does not authorize this exact run")
    return {
        "path": resolved.relative_to(WORKSPACE_ROOT).as_posix(),
        "sha256": expected_pilot_audit_sha256,
        "schema": audit["schema"],
        "decision": audit["decision"],
        "run_id": expected_subject["run_id"],
        "subject_sha256": _canonical_sha256(expected_subject),
    }


def build_pilot_audit_subject(
    *,
    model_dir: Path,
    model_receipt_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    token_preflight_path: Path,
    expected_token_preflight_sha256: str,
    config: SmokeConfig | None = None,
) -> dict[str, Any]:
    """Build the exact pilot audit subject without authorizing or running training."""

    config = SmokeConfig() if config is None else config
    config.validate()
    model_dir = model_dir.resolve(strict=True)
    dataset_dir = dataset_dir.resolve(strict=True)
    model_receipt_path = model_receipt_path.resolve(strict=True)
    output_dir = _safe_new_output_path(output_dir, kind="directory")
    dataset_manifest, dataset_manifest_path = _read_manifest(dataset_dir)
    model_receipt = _read_model_receipt(model_dir, model_receipt_path)
    input_snapshot = _input_snapshot(
        dataset_dir=dataset_dir,
        dataset_manifest=dataset_manifest,
        dataset_manifest_path=dataset_manifest_path,
        model_dir=model_dir,
        model_receipt=model_receipt,
        model_receipt_path=model_receipt_path,
    )
    _, token_preflight_binding, _ = _read_v4_token_preflight(
        model_dir=model_dir,
        model_receipt=model_receipt,
        dataset_dir=dataset_dir,
        dataset_manifest=dataset_manifest,
        dataset_manifest_path=dataset_manifest_path,
        model_receipt_path=model_receipt_path,
        token_preflight_path=token_preflight_path,
        expected_token_preflight_sha256=expected_token_preflight_sha256,
        config=config,
    )
    subject = _pilot_audit_subject(
        dataset_manifest_path=dataset_manifest_path,
        model_receipt_path=model_receipt_path,
        token_preflight_sha256=token_preflight_binding["sha256"],
        input_snapshot_sha256=input_snapshot["snapshot_sha256"],
        output_dir=output_dir,
        config=config,
    )
    return {
        "schema": PILOT_SUBJECT_SCHEMA,
        "status": "PILOT_SUBJECT_READY_NOT_AUTHORIZED_NOT_TRAINED",
        "subject": subject,
        "subject_sha256": _canonical_sha256(subject),
        "input_snapshot": input_snapshot,
        "token_preflight": token_preflight_binding,
        "authorization": {
            "qlora_pilot_authorized": False,
            "full_training_authorized": False,
            "bpu_authorized": False,
            "x5_authorized": False,
            "production_integration_authorized": False,
        },
    }


def _select_examples(path: Path, limit: int) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for item in iter_jsonl(path):
        examples.append(item)
        if len(examples) >= limit:
            break
    if not examples:
        raise ValueError(f"no examples in {path}")
    return examples


def _sanitize_metrics(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _sanitize_metrics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_metrics(item) for item in value]
    if hasattr(value, "item"):
        return _sanitize_metrics(value.item())
    return str(value)


def _hash_tree(path: Path) -> list[dict[str, Any]]:
    records = []
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        records.append(
            {
                "path": file_path.relative_to(path).as_posix(),
                "bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
        )
    return records


def _package_versions(names: Sequence[str]) -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "NOT_INSTALLED"
    return result


def _prepare_runtime() -> tuple[Any, Any, Any, Any, Any, Any]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_DISABLED", "true")

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    return (
        torch,
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig),
        (Trainer, TrainingArguments),
    )


def run_smoke_training(
    *,
    model_dir: Path,
    model_receipt_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    config: SmokeConfig | None = None,
    token_preflight_path: Path | None = None,
    expected_token_preflight_sha256: str | None = None,
    pilot_audit_path: Path | None = None,
    expected_pilot_audit_sha256: str | None = None,
) -> dict[str, Any]:
    config = SmokeConfig() if config is None else config
    config.validate()
    model_dir = model_dir.resolve(strict=True)
    dataset_dir = dataset_dir.resolve(strict=True)
    model_receipt_path = model_receipt_path.resolve(strict=True)
    output_dir = _safe_new_output_path(output_dir, kind="directory")

    dataset_manifest, dataset_manifest_path = _read_manifest(dataset_dir)
    model_receipt = _read_model_receipt(model_dir, model_receipt_path)
    input_snapshot = _input_snapshot(
        dataset_dir=dataset_dir,
        dataset_manifest=dataset_manifest,
        dataset_manifest_path=dataset_manifest_path,
        model_dir=model_dir,
        model_receipt=model_receipt,
        model_receipt_path=model_receipt_path,
    )
    (
        _,
        token_preflight_binding,
        tokenizer,
    ) = _read_v4_token_preflight(
        model_dir=model_dir,
        model_receipt=model_receipt,
        dataset_dir=dataset_dir,
        dataset_manifest=dataset_manifest,
        dataset_manifest_path=dataset_manifest_path,
        model_receipt_path=model_receipt_path,
        token_preflight_path=token_preflight_path,
        expected_token_preflight_sha256=expected_token_preflight_sha256,
        config=config,
    )
    subject = _pilot_audit_subject(
        dataset_manifest_path=dataset_manifest_path,
        model_receipt_path=model_receipt_path,
        token_preflight_sha256=token_preflight_binding["sha256"],
        input_snapshot_sha256=input_snapshot["snapshot_sha256"],
        output_dir=output_dir,
        config=config,
    )
    pilot_audit_binding = _read_v4_pilot_audit(
        expected_subject=subject,
        pilot_audit_path=pilot_audit_path,
        expected_pilot_audit_sha256=expected_pilot_audit_sha256,
    )
    if (
        _input_snapshot(
            dataset_dir=dataset_dir,
            dataset_manifest=dataset_manifest,
            dataset_manifest_path=dataset_manifest_path,
            model_dir=model_dir,
            model_receipt=model_receipt,
            model_receipt_path=model_receipt_path,
        )
        != input_snapshot
    ):
        raise PermissionError("QLoRA inputs changed after pilot authorization")
    train_examples = _select_examples(
        dataset_dir / "train.jsonl", config.max_train_examples
    )
    eval_examples = _select_examples(
        dataset_dir / "validation.jsonl", config.max_eval_examples
    )

    (
        torch,
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        model_classes,
        trainer_classes,
    ) = _prepare_runtime()
    AutoModelForCausalLM, _, BitsAndBytesConfig = model_classes
    Trainer, TrainingArguments = trainer_classes

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the NF4 QLoRA smoke run")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_mib = free_bytes // (1024 * 1024)
    if free_mib < config.minimum_free_vram_mib:
        raise RuntimeError(
            f"free VRAM {free_mib} MiB is below required {config.minimum_free_vram_mib} MiB"
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    encoded_train: list[dict[str, Any]] = []
    encoded_eval: list[dict[str, Any]] = []
    rejected_overlength = {"train": 0, "validation": 0}
    rejected_by_task: dict[str, dict[str, int]] = {
        "train": {},
        "validation": {},
    }
    for item in train_examples:
        try:
            encoded_train.append(
                encode_assistant_only(
                    tokenizer,
                    item["messages"],
                    max_seq_length=config.max_seq_length,
                )
            )
        except ValueError as exc:
            if "above max_seq_length" not in str(exc):
                raise
            rejected_overlength["train"] += 1
            task = str(item["task"])
            rejected_by_task["train"][task] = (
                rejected_by_task["train"].get(task, 0) + 1
            )
    for item in eval_examples:
        try:
            encoded_eval.append(
                encode_assistant_only(
                    tokenizer,
                    item["messages"],
                    max_seq_length=config.max_seq_length,
                )
            )
        except ValueError as exc:
            if "above max_seq_length" not in str(exc):
                raise
            rejected_overlength["validation"] += 1
            task = str(item["task"])
            rejected_by_task["validation"][task] = (
                rejected_by_task["validation"].get(task, 0) + 1
            )
    if len(encoded_train) < 2 or not encoded_eval:
        raise RuntimeError("insufficient in-contract examples after token-length validation")
    if rejected_overlength["train"] or rejected_overlength["validation"]:
        raise RuntimeError(
            "default smoke contract requires zero overlength examples; "
            f"observed {rejected_overlength}"
        )
    if (
        _input_snapshot(
            dataset_dir=dataset_dir,
            dataset_manifest=dataset_manifest,
            dataset_manifest_path=dataset_manifest_path,
            model_dir=model_dir,
            model_receipt=model_receipt,
            model_receipt_path=model_receipt_path,
        )
        != input_snapshot
    ):
        raise PermissionError("QLoRA inputs changed before model loading")
    os.mkdir(output_dir)

    token_stats = {
        "train_examples": len(encoded_train),
        "eval_examples": len(encoded_eval),
        "selected_task_counts": {
            "train": dict(sorted(Counter(str(item["task"]) for item in train_examples).items())),
            "validation": dict(
                sorted(Counter(str(item["task"]) for item in eval_examples).items())
            ),
        },
        "rejected_overlength": rejected_overlength,
        "rejected_overlength_by_task": rejected_by_task,
        "sequence_tokens_mean": mean(item["sequence_tokens"] for item in encoded_train),
        "sequence_tokens_max": max(item["sequence_tokens"] for item in encoded_train),
        "assistant_tokens_mean": mean(item["assistant_tokens"] for item in encoded_train),
        "assistant_supervised_fraction": (
            sum(item["assistant_tokens"] for item in encoded_train)
            / sum(item["sequence_tokens"] for item in encoded_train)
        ),
    }

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
        def __call__(self, features: Sequence[Mapping[str, Sequence[int]]]) -> dict[str, Any]:
            max_length = max(len(item["input_ids"]) for item in features)
            input_ids: list[list[int]] = []
            attention_mask: list[list[int]] = []
            labels: list[list[int]] = []
            for item in features:
                padding = max_length - len(item["input_ids"])
                input_ids.append(list(item["input_ids"]) + [tokenizer.pad_token_id] * padding)
                attention_mask.append(list(item["attention_mask"]) + [0] * padding)
                labels.append(list(item["labels"]) + [-100] * padding)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(DEFAULT_TARGET_MODULES),
    )
    model = get_peft_model(model, lora_config)
    model_load_seconds = time.perf_counter() - load_started
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())

    trainer_output_dir = output_dir / "trainer_state"
    training_arguments = TrainingArguments(
        output_dir=str(trainer_output_dir),
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        eval_strategy="no",
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_steps=config.max_steps,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        save_strategy="no",
        report_to=[],
        seed=config.seed,
        data_seed=config.seed,
        bf16=True,
        fp16=False,
        tf32=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        skip_memory_metrics=True,
        disable_tqdm=False,
    )
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=EncodedDataset(encoded_train),
        eval_dataset=EncodedDataset(encoded_eval),
        data_collator=AssistantOnlyCollator(),
        processing_class=tokenizer,
    )

    train_started = time.perf_counter()
    train_result = trainer.train()
    train_seconds = time.perf_counter() - train_started
    eval_metrics = trainer.evaluate()
    adapter_dir = output_dir / "adapter"
    if config.save_adapter:
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

    allocated_peak = torch.cuda.max_memory_allocated(0)
    reserved_peak = torch.cuda.max_memory_reserved(0)
    free_before_cleanup, _ = torch.cuda.mem_get_info(0)
    adapter_files = _hash_tree(adapter_dir) if adapter_dir.is_dir() else []
    final_input_snapshot = _input_snapshot(
        dataset_dir=dataset_dir,
        dataset_manifest=dataset_manifest,
        dataset_manifest_path=dataset_manifest_path,
        model_dir=model_dir,
        model_receipt=model_receipt,
        model_receipt_path=model_receipt_path,
    )
    if final_input_snapshot != input_snapshot:
        raise PermissionError("QLoRA inputs changed during the pilot")
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()
    free_after_cleanup, _ = torch.cuda.mem_get_info(0)

    run_status = "PASS_AUDITED_MAX_10_STEP_PILOT_NOT_FULL_TRAINING_NOT_DEPLOYED"
    run_scope = "separately audited 1-10 step pilot"

    receipt = {
        "schema": TRAINING_RECEIPT_SCHEMA,
        "trainer_version": TRAINER_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": run_status,
        "production_integration_allowed": False,
        "network_used": False,
        "teacher_model_used": False,
        "x5_contacted": False,
        "base_model": {
            "repo_id": model_receipt["repo_id"],
            "revision": model_receipt["revision"],
            "receipt_sha256": sha256_file(model_receipt_path.resolve()),
            "model_sha256": next(
                item["sha256"]
                for item in model_receipt["files"]
                if item["path"] == "model.safetensors"
            ),
        },
        "dataset": {
            "schema": dataset_manifest["schema"],
            "manifest_path": dataset_manifest_path.name,
            "manifest_sha256": sha256_file(dataset_manifest_path),
            "source_archive_sha256": dataset_manifest.get("source_lock", {}).get(
                "archive_sha256"
            ),
            "external_audit_sha256": dataset_manifest.get(
                "authorization", {}
            ).get("external_audit_sha256"),
            "token_preflight": token_preflight_binding,
            "pilot_audit": pilot_audit_binding,
            "input_snapshot": input_snapshot,
            "files": dataset_manifest["files"],
            "tokenization": token_stats,
        },
        "configuration": _pilot_configuration(config),
        "model_parameters": {
            "trainable": trainable_parameters,
            "total_visible": total_parameters,
            "trainable_fraction": trainable_parameters / total_parameters,
        },
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_vram_bytes": total_bytes,
            "free_vram_before_bytes": free_bytes,
            "free_vram_before_cleanup_bytes": free_before_cleanup,
            "free_vram_after_cleanup_bytes": free_after_cleanup,
            "peak_allocated_bytes": allocated_peak,
            "peak_reserved_bytes": reserved_peak,
            "memory_note": (
                "PyTorch allocator peak_reserved may exceed physical VRAM under Windows "
                "WDDM/paged optimizers; peak_allocated and post-cleanup free memory are "
                "reported separately."
            ),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "dependencies": _package_versions(
            ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "safetensors")
        ),
        "metrics": {
            "model_load_seconds": model_load_seconds,
            "train_wall_seconds": train_seconds,
            "train": _sanitize_metrics(train_result.metrics),
            "evaluation": _sanitize_metrics(eval_metrics),
        },
        "adapter_files": adapter_files,
        "claim_boundary": (
            f"This receipt proves only that a local {config.max_steps}-step NF4 QLoRA "
            f"{run_scope} completed with assistant-only labels on the recorded RTX 4050 "
            "environment. It does not establish model quality, full convergence, BPU "
            "conversion, X5 runtime, or production integration."
        ),
    }
    write_json_atomic(output_dir / "training_receipt.v1.json", receipt)
    return receipt
