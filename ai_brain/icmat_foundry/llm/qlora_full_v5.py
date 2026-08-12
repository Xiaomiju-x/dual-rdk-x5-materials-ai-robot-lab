"""Isolated, offline full NF4 QLoRA trainer for the ICMat v5 student contract."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

TRAINER_VERSION = "icmat-qwen05b-nf4-qlora-full-v5.1.0"
EXAMPLE_SCHEMA = "icmat_student_sft_example.v5"
PREFLIGHT_SCHEMA = "icmat_qlora_full_preflight.v5"
SEED_RECEIPT_SCHEMA = "icmat_qlora_full_seed_receipt.v5"
RUN_RECEIPT_SCHEMA = "icmat_qlora_full_run_receipt.v5"
FAILURE_RECEIPT_SCHEMA = "icmat_qlora_full_failure_receipt.v5"
MANIFEST_NAME = "manifest.v5.json"
SPLIT_FILES = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "calibration": "calibration.jsonl",
    "blind_test": "blind_test.jsonl",
}
INSPECTED_SPLITS = ("train", "validation", "calibration")
TRAINING_SPLITS = ("train", "validation")
ALLOWED_DECISIONS = frozenset({"ANSWER", "REFUSE"})
DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FullTrainingConfig:
    """All knobs that can affect one multi-seed full training run."""

    num_train_epochs: float = 6.0
    max_steps: int = -1
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
    early_stopping_patience: int = 2
    early_stopping_threshold: float = 0.0
    save_total_limit: int = 2
    minimum_free_vram_mib: int = 3600
    seeds: tuple[int, ...] = (20260729, 20260730, 20260731)

    def validate(self) -> None:
        integer_fields = {
            "max_steps": self.max_steps,
            "max_seq_length": self.max_seq_length,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "early_stopping_patience": self.early_stopping_patience,
            "save_total_limit": self.save_total_limit,
            "minimum_free_vram_mib": self.minimum_free_vram_mib,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in integer_fields.values()
        ):
            raise TypeError("integer QLoRA configuration fields must be integers")
        float_fields = {
            "num_train_epochs": self.num_train_epochs,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "lora_dropout": self.lora_dropout,
            "early_stopping_threshold": self.early_stopping_threshold,
        }
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in float_fields.values()
        ):
            raise ValueError("floating-point QLoRA configuration fields must be finite")
        if self.num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be positive")
        if self.max_steps != -1 and self.max_steps < 1:
            raise ValueError("max_steps must be -1 or a positive integer")
        if not 128 <= self.max_seq_length <= 32768:
            raise ValueError("max_seq_length must be in [128, 32768]")
        if not 1 <= self.per_device_train_batch_size <= 8:
            raise ValueError("train batch size must be in [1, 8]")
        if not 1 <= self.per_device_eval_batch_size <= 8:
            raise ValueError("evaluation batch size must be in [1, 8]")
        if not 1 <= self.gradient_accumulation_steps <= 256:
            raise ValueError("gradient accumulation must be in [1, 256]")
        if not 0.0 < self.learning_rate <= 0.01:
            raise ValueError("learning_rate must be in (0, 0.01]")
        if not 0.0 <= self.warmup_ratio <= 0.5:
            raise ValueError("warmup_ratio must be in [0, 0.5]")
        if not 0.0 <= self.weight_decay <= 1.0:
            raise ValueError("weight_decay must be in [0, 1]")
        if self.lora_rank not in {4, 8, 16, 32, 64}:
            raise ValueError("unsupported LoRA rank")
        if not 1 <= self.lora_alpha <= 512:
            raise ValueError("lora_alpha must be in [1, 512]")
        if not 0.0 <= self.lora_dropout < 0.5:
            raise ValueError("lora_dropout must be in [0, 0.5)")
        if not 1 <= self.early_stopping_patience <= 20:
            raise ValueError("early_stopping_patience must be in [1, 20]")
        if self.early_stopping_threshold < 0.0:
            raise ValueError("early_stopping_threshold must be non-negative")
        if not 1 <= self.save_total_limit <= 20:
            raise ValueError("save_total_limit must be in [1, 20]")
        if not 1024 <= self.minimum_free_vram_mib <= 131072:
            raise ValueError("minimum_free_vram_mib must be in [1024, 131072]")
        if not isinstance(self.seeds, tuple):
            raise TypeError("seeds must be a tuple")
        if len(self.seeds) < 3:
            raise ValueError("full v5 training requires at least three independent seeds")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("training seeds must be unique")
        if any(
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed <= 2_147_483_647
            for seed in self.seeds
        ):
            raise ValueError("every seed must be an integer in [0, 2147483647]")


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


def _package_versions(names: Sequence[str]) -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return versions


def _tree_inventory(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            target = path.resolve(strict=True)
            if not target.is_file():
                raise ValueError(f"model tree symlink does not resolve to a file: {path}")
        elif not path.is_file():
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not records:
        raise ValueError(f"input tree is empty: {root}")
    return {
        "files": records,
        "tree_sha256": _canonical_sha256(records),
        "file_count": len(records),
        "bytes": sum(record["bytes"] for record in records),
    }


def _flatten_manifest_file_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def walk(node: Any, hint: str | None = None) -> None:
        if isinstance(node, Mapping):
            if "path" in node:
                records.append(dict(node))
                return
            for key, child in node.items():
                if isinstance(child, str) and child in SPLIT_FILES.values():
                    records.append({"path": child, "role": str(key)})
                else:
                    walk(child, str(key))
            return
        if isinstance(node, list):
            for child in node:
                walk(child, hint)
            return
        if isinstance(node, str) and node in SPLIT_FILES.values():
            records.append({"path": node, "role": hint})

    walk(value)
    return records


def _manifest_records(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    files = manifest.get("files", manifest.get("splits"))
    if not isinstance(files, (Mapping, list)):
        raise ValueError(
            "manifest.v5.json must contain a files or splits object/list"
        )
    flattened = _flatten_manifest_file_records(files)
    result: dict[str, dict[str, Any]] = {}
    for split, filename in SPLIT_FILES.items():
        matches = [record for record in flattened if record.get("path") == filename]
        if len(matches) != 1:
            raise ValueError(f"manifest must bind exactly one {filename} record")
        record = matches[0]
        sha256 = record.get("sha256")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError(f"manifest file record lacks a lowercase SHA-256: {filename}")
        byte_count = record.get("bytes")
        example_count = record.get("examples", record.get("count"))
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise ValueError(f"manifest file record lacks a valid byte count: {filename}")
        if (
            isinstance(example_count, bool)
            or not isinstance(example_count, int)
            or example_count < 1
        ):
            raise ValueError(f"manifest file record lacks a positive example count: {filename}")
        result[split] = {
            **record,
            "path": filename,
            "sha256": sha256,
            "bytes": byte_count,
            "examples": example_count,
        }
    return result


def _validate_messages(messages: Any, *, source: str) -> None:
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError(f"{source}: messages must contain system/user/assistant")
    roles = [message.get("role") if isinstance(message, Mapping) else None for message in messages]
    if roles != ["system", "user", "assistant"]:
        raise ValueError(f"{source}: messages roles must be system/user/assistant")
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{source}: every message content must be a non-empty string")
    try:
        assistant = json.loads(messages[-1]["content"])
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source}: assistant content is not JSON") from exc
    if not isinstance(assistant, dict) or not assistant:
        raise ValueError(f"{source}: assistant content must be a non-empty JSON object")


def _validate_example(
    item: Any,
    *,
    split: str,
    source: str,
    seen_example_ids: set[str],
) -> tuple[str, str, str]:
    if not isinstance(item, Mapping):
        raise ValueError(f"{source}: JSONL row must be an object")
    if item.get("schema") != EXAMPLE_SCHEMA:
        raise ValueError(f"{source}: unexpected example schema")
    example_id = item.get("example_id")
    if not isinstance(example_id, str) or not example_id.strip():
        raise ValueError(f"{source}: example_id must be a non-empty string")
    if example_id in seen_example_ids:
        raise ValueError(f"{source}: duplicate example_id {example_id}")
    seen_example_ids.add(example_id)
    if item.get("split") != split:
        raise ValueError(f"{source}: row split does not match {split}.jsonl")
    domain = item.get("domain")
    task = item.get("task")
    decision = item.get("decision")
    if not isinstance(domain, str) or not domain.strip():
        raise ValueError(f"{source}: domain must be a non-empty string")
    if not isinstance(task, str) or not task.strip():
        raise ValueError(f"{source}: task must be a non-empty string")
    if decision not in ALLOWED_DECISIONS:
        raise ValueError(f"{source}: decision must be ANSWER or REFUSE")
    _validate_messages(item.get("messages"), source=source)
    return domain, task, decision


def _scan_jsonl(
    path: Path,
    *,
    split: str,
    expected: Mapping[str, Any],
    seen_example_ids: set[str],
) -> dict[str, Any]:
    if path.is_symlink():
        raise PermissionError(f"dataset split must not be a symlink: {path}")
    digest = hashlib.sha256()
    byte_count = 0
    count = 0
    domains: Counter[str] = Counter()
    tasks: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            byte_count += len(raw_line)
            if not raw_line.strip():
                raise ValueError(f"{path.name}:{line_number}: blank JSONL row")
            try:
                item = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path.name}:{line_number}: invalid UTF-8 JSON") from exc
            domain, task, decision = _validate_example(
                item,
                split=split,
                source=f"{path.name}:{line_number}",
                seen_example_ids=seen_example_ids,
            )
            domains[domain] += 1
            tasks[task] += 1
            decisions[decision] += 1
            count += 1
    actual_sha256 = digest.hexdigest()
    if byte_count != expected["bytes"]:
        raise ValueError(f"{path.name}: byte count does not match manifest")
    if actual_sha256 != expected["sha256"]:
        raise ValueError(f"{path.name}: SHA-256 does not match manifest")
    if count != expected["examples"]:
        raise ValueError(f"{path.name}: example count does not match manifest")
    return {
        "path": path.name,
        "bytes": byte_count,
        "sha256": actual_sha256,
        "examples": count,
        "domains": dict(sorted(domains.items())),
        "tasks": dict(sorted(tasks.items())),
        "decisions": dict(sorted(decisions.items())),
        "content_read": True,
    }


def _dataset_snapshot(dataset_dir: Path) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve(strict=True)
    if not dataset_dir.is_dir():
        raise NotADirectoryError(dataset_dir)
    manifest_path = dataset_dir / MANIFEST_NAME
    if manifest_path.is_symlink():
        raise PermissionError("manifest.v5.json must not be a symlink")
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest.v5.json must contain a JSON object")
    manifest_schema = manifest.get("schema")
    if manifest_schema is not None and (
        not isinstance(manifest_schema, str) or not manifest_schema.endswith(".v5")
    ):
        raise ValueError("manifest schema, when present, must be a v5 schema")
    records = _manifest_records(manifest)
    seen_example_ids: set[str] = set()
    summaries: dict[str, Any] = {}
    for split in INSPECTED_SPLITS:
        path = dataset_dir / SPLIT_FILES[split]
        if not path.is_file():
            raise FileNotFoundError(path)
        summaries[split] = _scan_jsonl(
            path,
            split=split,
            expected=records[split],
            seen_example_ids=seen_example_ids,
        )

    blind_path = dataset_dir / SPLIT_FILES["blind_test"]
    if blind_path.is_symlink():
        raise PermissionError("blind_test.jsonl must not be a symlink")
    if not blind_path.is_file():
        raise FileNotFoundError(blind_path)
    blind_stat = blind_path.stat()
    if blind_stat.st_size != records["blind_test"]["bytes"]:
        raise ValueError("blind_test.jsonl byte count does not match manifest")
    summaries["blind_test"] = {
        "path": blind_path.name,
        "bytes": blind_stat.st_size,
        "sha256": records["blind_test"]["sha256"],
        "examples": records["blind_test"]["examples"],
        "sha256_source": "manifest_declared_not_recomputed",
        "content_read": False,
    }
    inspected_files = [summaries[split] for split in INSPECTED_SPLITS]
    core = {
        "manifest": {
            "path": MANIFEST_NAME,
            "bytes": manifest_path.stat().st_size,
            "sha256": _sha256_file(manifest_path),
            "schema": manifest_schema,
        },
        "splits": summaries,
        "blind_test_policy": {
            "opened": False,
            "hashed_by_trainer": False,
            "used_for_training": False,
            "used_for_validation": False,
            "declared_hash_recorded_only": True,
        },
        "inspected_input_sha256": _canonical_sha256(
            {
                "manifest": _sha256_file(manifest_path),
                "files": [
                    {
                        "path": record["path"],
                        "bytes": record["bytes"],
                        "sha256": record["sha256"],
                    }
                    for record in inspected_files
                ],
                "blind_test_declared": {
                    "path": records["blind_test"]["path"],
                    "bytes": records["blind_test"]["bytes"],
                    "sha256": records["blind_test"]["sha256"],
                    "examples": records["blind_test"]["examples"],
                },
            }
        ),
    }
    return core


def _configuration_payload(config: FullTrainingConfig) -> dict[str, Any]:
    return {
        **asdict(config),
        "seeds": list(config.seeds),
        "quantization": "NF4",
        "double_quantization": True,
        "compute_dtype": "bfloat16",
        "optimizer": "paged_adamw_8bit",
        "gradient_checkpointing": True,
        "assistant_only_loss": True,
        "target_modules": list(DEFAULT_TARGET_MODULES),
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "best_model_metric": "eval_loss",
        "greater_is_better": False,
        "blind_test_access": "FORBIDDEN",
    }


def preflight_v5_contract(
    *,
    dataset_dir: Path,
    model_dir: Path | None = None,
    config: FullTrainingConfig | None = None,
) -> dict[str, Any]:
    """Validate and hash the train-side contract without importing any ML runtime.

    The function intentionally never opens or hashes ``blind_test.jsonl``.
    """

    config = FullTrainingConfig() if config is None else config
    config.validate()
    dataset = _dataset_snapshot(Path(dataset_dir))
    model: dict[str, Any]
    if model_dir is None:
        model = {
            "provided": False,
            "tree_hashed": False,
        }
    else:
        resolved_model = Path(model_dir).resolve(strict=True)
        model = {
            "provided": True,
            "path": str(resolved_model),
            **_tree_inventory(resolved_model),
        }
    configuration = _configuration_payload(config)
    return {
        "schema": PREFLIGHT_SCHEMA,
        "trainer_version": TRAINER_VERSION,
        "created_at": _utc_now(),
        "status": "PASS_READ_ONLY_V5_CONTRACT_PREFLIGHT_NOT_TRAINED",
        "read_only": True,
        "gpu_required": False,
        "ml_runtime_imported": False,
        "network_used": False,
        "dataset": dataset,
        "base_model": model,
        "configuration": configuration,
        "configuration_sha256": _canonical_sha256(configuration),
        "claim_boundary": (
            "This preflight validates only the v5 train/validation/calibration contract "
            "and local input hashes. blind_test.jsonl was not opened or hashed, no model "
            "was loaded, and no training or quality claim is authorized."
        ),
    }


def build_assistant_only_labels(
    prefix_ids: Sequence[int],
    full_ids: Sequence[int],
) -> list[int]:
    prefix = list(prefix_ids)
    full = list(full_ids)
    if not prefix or len(prefix) >= len(full):
        raise ValueError("assistant target must add at least one token")
    if full[: len(prefix)] != prefix:
        raise ValueError("chat-template prefix is not a prefix of the full conversation")
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
        raise ValueError(
            f"conversation has {len(full_ids)} tokens, above max_seq_length={max_seq_length}"
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


def _load_split_rows(dataset_dir: Path, split: str) -> list[dict[str, Any]]:
    if split not in TRAINING_SPLITS:
        raise PermissionError(f"trainer is forbidden from loading split: {split}")
    rows: list[dict[str, Any]] = []
    with (dataset_dir / SPLIT_FILES[split]).open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"{split} split is empty")
    return rows


def _sanitize_metrics(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize_metrics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_metrics(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return _sanitize_metrics(value.item())
    return str(value)


def _adapter_inventory(path: Path) -> dict[str, Any]:
    inventory = _tree_inventory(path)
    return {
        "path": path.name,
        "files": inventory["files"],
        "tree_sha256": inventory["tree_sha256"],
        "file_count": inventory["file_count"],
        "bytes": inventory["bytes"],
    }


def _epoch_history(log_history: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    train_logs: list[dict[str, Any]] = []
    eval_logs: list[dict[str, Any]] = []
    for record in log_history:
        if "loss" in record and "epoch" in record and "eval_loss" not in record:
            train_logs.append(dict(record))
        if "eval_loss" in record and "epoch" in record:
            eval_logs.append(dict(record))
    result: list[dict[str, Any]] = []
    for evaluation in eval_logs:
        epoch = float(evaluation["epoch"])
        candidates = [
            record
            for record in train_logs
            if float(record["epoch"]) <= epoch + 1e-9
        ]
        train_loss = float(candidates[-1]["loss"]) if candidates else None
        result.append(
            {
                "epoch": epoch,
                "global_step": int(evaluation.get("step", 0)),
                "train_loss": train_loss,
                "validation_loss": float(evaluation["eval_loss"]),
                "validation_runtime_seconds": evaluation.get("eval_runtime"),
            }
        )
    if not result:
        raise RuntimeError("Trainer did not emit per-epoch validation loss")
    if any(record["train_loss"] is None for record in result):
        raise RuntimeError("Trainer did not emit per-epoch training loss")
    return result


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
    versions = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return versions[0] if versions else "UNAVAILABLE"


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
        EarlyStoppingCallback,
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
        "EarlyStoppingCallback": EarlyStoppingCallback,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "set_seed": set_seed,
    }


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


def _run_single_seed(
    *,
    model_dir: Path,
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    seed_dir: Path,
    seed: int,
    config: FullTrainingConfig,
) -> dict[str, Any]:
    runtime = _prepare_runtime()
    torch = runtime["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full NF4 QLoRA training")
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
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
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
        visible_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainer_dir = seed_dir / "trainer"
        arguments = runtime["TrainingArguments"](
            output_dir=str(trainer_dir),
            overwrite_output_dir=False,
            do_train=True,
            do_eval=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=config.save_total_limit,
            num_train_epochs=config.num_train_epochs,
            max_steps=config.max_steps,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            lr_scheduler_type="cosine",
            report_to=[],
            seed=seed,
            data_seed=seed,
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
        trainer = runtime["Trainer"](
            model=model,
            args=arguments,
            train_dataset=EncodedDataset(encoded_train),
            eval_dataset=EncodedDataset(encoded_validation),
            data_collator=AssistantOnlyCollator(),
            processing_class=tokenizer,
            callbacks=[
                runtime["EarlyStoppingCallback"](
                    early_stopping_patience=config.early_stopping_patience,
                    early_stopping_threshold=config.early_stopping_threshold,
                )
            ],
        )
        train_result = trainer.train()
        epoch_metrics = _epoch_history(trainer.state.log_history)
        best_checkpoint = trainer.state.best_model_checkpoint
        best_metric = trainer.state.best_metric
        if not best_checkpoint or best_metric is None:
            raise RuntimeError("Trainer did not select a best validation checkpoint")
        best_adapter_dir = seed_dir / "best_adapter"
        trainer.save_model(str(best_adapter_dir))
        tokenizer.save_pretrained(str(best_adapter_dir))
        adapter = _adapter_inventory(best_adapter_dir)
        peak_allocated = int(torch.cuda.max_memory_allocated(0))
        peak_reserved = int(torch.cuda.max_memory_reserved(0))
        wall_seconds = time.perf_counter() - started
        receipt = {
            "schema": SEED_RECEIPT_SCHEMA,
            "trainer_version": TRAINER_VERSION,
            "created_at": _utc_now(),
            "status": "PASS_SEED_TRAINING_COMPLETED_NOT_QUALITY_ACCEPTED",
            "seed": seed,
            "configuration": _configuration_payload(config),
            "dataset": {
                "train_examples": len(train_rows),
                "validation_examples": len(validation_rows),
                "calibration_used": False,
                "blind_test_opened": False,
                "train_tokenization": train_tokenization,
                "validation_tokenization": validation_tokenization,
            },
            "model_parameters": {
                "trainable": trainable_parameters,
                "visible": visible_parameters,
                "trainable_fraction": trainable_parameters / visible_parameters,
            },
            "per_epoch_metrics": epoch_metrics,
            "best_checkpoint": {
                "trainer_path": Path(best_checkpoint).name,
                "validation_loss": float(best_metric),
                "best_adapter_path": best_adapter_dir.name,
            },
            "adapter": adapter,
            "metrics": {
                "train_result": _sanitize_metrics(train_result.metrics),
                "wall_seconds": wall_seconds,
                "peak_allocated_vram_bytes": peak_allocated,
                "peak_reserved_vram_bytes": peak_reserved,
            },
            "hardware": {
                "gpu_name": torch.cuda.get_device_name(0),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "total_vram_bytes": int(total_bytes),
                "free_vram_before_bytes": int(free_bytes),
            },
        }
        _write_json_atomic(seed_dir / "seed_receipt.v5.json", receipt)
        return receipt
    finally:
        if trainer is not None:
            del trainer
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


def _source_inventory() -> dict[str, Any]:
    paths = {
        "trainer": Path(__file__).resolve(),
        "cli": WORKSPACE_ROOT / "tools" / "train_icmat_qlora_full_v5.py",
    }
    records = {}
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        records[role] = {
            "path": path.relative_to(WORKSPACE_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    return records


def _safe_new_output(output_dir: Path) -> tuple[Path, Path]:
    raw = Path(output_dir)
    if raw.name in {"", ".", ".."}:
        raise ValueError("output must name a new directory")
    parent = raw.parent.resolve(strict=True)
    if not parent.is_dir():
        raise NotADirectoryError(parent)
    final = parent / raw.name
    if os.path.lexists(final):
        raise FileExistsError(final)
    return parent, final


def _failure_directory(parent: Path, final_name: str, run_id: str) -> Path:
    return parent / f".{final_name}.failed-{run_id}-{uuid.uuid4().hex}"


def run_full_training(
    *,
    model_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
    config: FullTrainingConfig | None = None,
) -> dict[str, Any]:
    """Train all configured seeds and atomically publish only a complete run."""

    config = FullTrainingConfig() if config is None else config
    config.validate()
    model_dir = Path(model_dir).resolve(strict=True)
    dataset_dir = Path(dataset_dir).resolve(strict=True)
    parent, final_output = _safe_new_output(Path(output_dir))

    preflight = preflight_v5_contract(
        dataset_dir=dataset_dir,
        model_dir=model_dir,
        config=config,
    )
    source_inventory = _source_inventory()
    run_core = {
        "trainer_version": TRAINER_VERSION,
        "dataset_input_sha256": preflight["dataset"]["inspected_input_sha256"],
        "model_tree_sha256": preflight["base_model"]["tree_sha256"],
        "configuration_sha256": preflight["configuration_sha256"],
        "source_inventory": source_inventory,
        "output_name": final_output.name,
    }
    run_id = "icmat-v5-" + _canonical_sha256(run_core)[:20]
    stage = parent / f".{final_output.name}.tmp-{run_id}-{uuid.uuid4().hex}"
    if os.path.lexists(stage):
        raise FileExistsError(stage)
    os.mkdir(stage)
    started = time.perf_counter()
    completed_seeds: list[dict[str, Any]] = []
    active_seed: int | None = None
    try:
        _write_json_atomic(stage / "preflight.v5.json", preflight)
        train_rows = _load_split_rows(dataset_dir, "train")
        validation_rows = _load_split_rows(dataset_dir, "validation")
        for seed in config.seeds:
            active_seed = seed
            seed_dir = stage / f"seed-{seed}"
            os.mkdir(seed_dir)
            completed_seeds.append(
                _run_single_seed(
                    model_dir=model_dir,
                    train_rows=train_rows,
                    validation_rows=validation_rows,
                    seed_dir=seed_dir,
                    seed=seed,
                    config=config,
                )
            )

        final_snapshot = preflight_v5_contract(
            dataset_dir=dataset_dir,
            model_dir=model_dir,
            config=config,
        )
        if (
            final_snapshot["dataset"]["inspected_input_sha256"]
            != preflight["dataset"]["inspected_input_sha256"]
            or final_snapshot["base_model"]["tree_sha256"]
            != preflight["base_model"]["tree_sha256"]
            or final_snapshot["configuration_sha256"]
            != preflight["configuration_sha256"]
        ):
            raise PermissionError("training inputs changed during the multi-seed run")

        best_seed_receipt = min(
            completed_seeds,
            key=lambda receipt: receipt["best_checkpoint"]["validation_loss"],
        )
        dependencies = _package_versions(
            (
                "torch",
                "transformers",
                "tokenizers",
                "peft",
                "accelerate",
                "bitsandbytes",
                "safetensors",
            )
        )
        receipt = {
            "schema": RUN_RECEIPT_SCHEMA,
            "trainer_version": TRAINER_VERSION,
            "created_at": _utc_now(),
            "status": "PASS_FULL_MULTI_SEED_TRAINING_COMPLETED_NOT_DEPLOYED",
            "run_id": run_id,
            "atomic_publish": True,
            "network_used": False,
            "blind_test_opened": False,
            "calibration_used_for_training": False,
            "input_snapshot": {
                "dataset": preflight["dataset"],
                "base_model": preflight["base_model"],
                "source_files": source_inventory,
            },
            "configuration": preflight["configuration"],
            "configuration_sha256": preflight["configuration_sha256"],
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "dependencies": dependencies,
            },
            "cuda": {
                "torch_cuda": getattr(
                    __import__("torch").version,
                    "cuda",
                    "UNAVAILABLE",
                ),
                "cudnn": __import__("torch").backends.cudnn.version(),
                "nvidia_driver": _nvidia_driver_version(),
            },
            "seeds": [
                {
                    "seed": seed_receipt["seed"],
                    "status": seed_receipt["status"],
                    "per_epoch_metrics": seed_receipt["per_epoch_metrics"],
                    "best_checkpoint": seed_receipt["best_checkpoint"],
                    "adapter": seed_receipt["adapter"],
                    "metrics": seed_receipt["metrics"],
                    "hardware": seed_receipt["hardware"],
                }
                for seed_receipt in completed_seeds
            ],
            "selected_best_seed": best_seed_receipt["seed"],
            "selected_best_adapter": (
                f"seed-{best_seed_receipt['seed']}/"
                f"{best_seed_receipt['best_checkpoint']['best_adapter_path']}"
            ),
            "selected_best_validation_loss": best_seed_receipt[
                "best_checkpoint"
            ]["validation_loss"],
            "wall_seconds": time.perf_counter() - started,
            "claim_boundary": (
                "This receipt proves that all recorded local NF4 QLoRA seed runs "
                "completed with assistant-only loss and validation-loss checkpoint "
                "selection. blind_test.jsonl was never opened. This is not a blind-test "
                "quality result, BPU conversion, X5 validation, or production release."
            ),
        }
        _write_json_atomic(stage / "training_receipt.v5.json", receipt)
        os.replace(stage, final_output)
        return receipt
    except BaseException as exc:
        failure = {
            "schema": FAILURE_RECEIPT_SCHEMA,
            "trainer_version": TRAINER_VERSION,
            "created_at": _utc_now(),
            "status": "FAILED_NO_SUCCESS_RELEASE",
            "run_id": run_id,
            "active_seed": active_seed,
            "completed_seeds": [receipt.get("seed") for receipt in completed_seeds],
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "final_output_created": False,
            "blind_test_opened": False,
        }
        try:
            _write_json_atomic(stage / "failure_receipt.v5.json", failure)
            failed = _failure_directory(parent, final_output.name, run_id)
            os.replace(stage, failed)
        except BaseException:
            pass
        raise
