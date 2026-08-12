#!/usr/bin/env python3
"""Train and package F-LLM-02 without touching the production model bank.

The script deliberately keeps the validation set blind during training.  It uses
assistant-only loss, writes a real PEFT adapter, evaluates the fixed validation
set, and can optionally merge/export a Q4_K_M GGUF for a CPU llama.cpp smoke.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ROOT = Path(__file__).resolve().parents[3]
MODEL_ID = "F-LLM-02"
MODEL_NAME = "ICMat-Qwen3-1.7B-EvidenceQA-CPU"
BASE_MODEL = ROOT / "research/model_assets/icmat_foundry/qwen3_17b_instruct/snapshot"
DATA_ROOT = (
    ROOT
    / "evaluation/icmat_foundry/llm"
    / "icmat_qwen05b_evidence_pointer_sft_v8_pretrain_20260731_r4"
)
TRAIN_FILE = DATA_ROOT / "train.jsonl"
VALIDATION_FILE = DATA_ROOT / "validation.jsonl"
ARTIFACT_ROOT = ROOT / "icmat_foundry/finals_50model/artifacts/llm/F-LLM-02"
EVIDENCE_ROOT = ROOT / "icmat_foundry/finals_50model/evidence/llm/F-LLM-02"
LLAMA_ROOT = ROOT / "research/toolchains/llama_cpp_b10158_source/llama.cpp-b10158"
LLAMA_RUNTIME = ROOT / "research/toolchains/llama_cpp_b10158_win_cuda13_3/runtime"
BASE_COMMIT = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
SEED = 20260801


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(encoded, encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 3:
                raise ValueError(f"{path}:{line_number}: messages contract missing")
            if messages[-1].get("role") != "assistant":
                raise ValueError(f"{path}:{line_number}: final assistant message missing")
            rows.append(row)
    return rows


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def validate_inputs() -> dict[str, Any]:
    required = [BASE_MODEL / "config.json", TRAIN_FILE, VALIDATION_FILE]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing fixed inputs: {missing}")
    config = json.loads((BASE_MODEL / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3" or config.get("num_hidden_layers") != 28:
        raise ValueError("fixed Qwen3-1.7B base identity mismatch")
    train_rows = load_jsonl(TRAIN_FILE)
    validation_rows = load_jsonl(VALIDATION_FILE)
    train_ids = {row["example_id"] for row in train_rows}
    validation_ids = {row["example_id"] for row in validation_rows}
    if train_ids & validation_ids:
        raise ValueError("train/validation example_id overlap")
    return {
        "base_commit": BASE_COMMIT,
        "base_config_sha256": sha256_file(BASE_MODEL / "config.json"),
        "base_weights": {
            item.name: sha256_file(item)
            for item in sorted(BASE_MODEL.glob("*.safetensors"))
        },
        "license": "Apache-2.0",
        "train": {
            "path": str(TRAIN_FILE.relative_to(ROOT)),
            "rows": len(train_rows),
            "sha256": sha256_file(TRAIN_FILE),
        },
        "validation": {
            "path": str(VALIDATION_FILE.relative_to(ROOT)),
            "rows": len(validation_rows),
            "sha256": sha256_file(VALIDATION_FILE),
        },
        "forbidden_training_corpus": "legacy_25228_chunk_rag",
        "forbidden_training_corpus_used": False,
    }


class AssistantOnlyDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        rows: Iterable[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.samples: list[dict[str, torch.Tensor]] = []
        for row in rows:
            messages = row["messages"]
            full_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            prefix_ids = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if full_ids[: len(prefix_ids)] != prefix_ids:
                raise ValueError(f"assistant prefix mismatch for {row['example_id']}")
            if len(full_ids) > max_length:
                raise ValueError(
                    f"sample {row['example_id']} has {len(full_ids)} tokens > {max_length}"
                )
            labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids) :]
            if not any(token != -100 for token in labels):
                raise ValueError(f"no assistant target tokens for {row['example_id']}")
            self.samples.append(
                {
                    "input_ids": torch.tensor(full_ids, dtype=torch.long),
                    "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


class CausalCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        width = max(int(item["input_ids"].shape[0]) for item in batch)
        result = {
            "input_ids": torch.full((len(batch), width), self.pad_token_id, dtype=torch.long),
            "attention_mask": torch.zeros((len(batch), width), dtype=torch.long),
            "labels": torch.full((len(batch), width), -100, dtype=torch.long),
        }
        for row_index, item in enumerate(batch):
            size = int(item["input_ids"].shape[0])
            for key in result:
                result[key][row_index, :size] = item[key]
        return result


@dataclass(frozen=True)
class TrainConfig:
    seed: int = SEED
    max_length: int = 768
    max_steps: int = 64
    micro_batch_size: int = 1
    gradient_accumulation: int = 4
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_steps: int = 5
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    max_grad_norm: float = 1.0


def load_tokenizer(path: Path = BASE_MODEL) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        local_files_only=True,
        fix_mistral_regex=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def quantization_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_quantized_base(training: bool) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        local_files_only=True,
        quantization_config=quantization_config(),
        device_map={"": 0},
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = not training
    return model


def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("RTX4050 CUDA is required for the fixed NF4 training run")
    identity = validate_inputs()
    config = TrainConfig(max_steps=args.max_steps)
    seed_everything(config.seed)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer()
    train_rows = load_jsonl(TRAIN_FILE)
    dataset = AssistantOnlyDataset(train_rows, tokenizer, config.max_length)
    loader_generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.micro_batch_size,
        shuffle=True,
        generator=loader_generator,
        collate_fn=CausalCollator(tokenizer.pad_token_id),
    )

    started = time.time()
    model = load_quantized_base(training=True)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.print_trainable_parameters()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    def learning_rate(step: int) -> float:
        if step < config.warmup_steps:
            return config.learning_rate * float(step + 1) / float(config.warmup_steps)
        progress = (step - config.warmup_steps) / max(
            1, config.max_steps - config.warmup_steps
        )
        return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    micro_step = 0
    model.train()
    while optimizer_step < config.max_steps:
        for batch in loader:
            batch = {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}
            output = model(**batch)
            loss = output.loss / config.gradient_accumulation
            loss.backward()
            losses.append(float(output.loss.detach().cpu()))
            micro_step += 1
            if micro_step % config.gradient_accumulation != 0:
                continue
            clip_grad_norm_(trainable, config.max_grad_norm)
            lr = learning_rate(optimizer_step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            if optimizer_step == 1 or optimizer_step % 8 == 0:
                recent = losses[-8 * config.gradient_accumulation :]
                print(
                    json.dumps(
                        {
                            "optimizer_step": optimizer_step,
                            "max_steps": config.max_steps,
                            "loss_mean_recent": sum(recent) / len(recent),
                            "learning_rate": lr,
                            "gpu_peak_mib": torch.cuda.max_memory_allocated() / 1024**2,
                        }
                    ),
                    flush=True,
                )
            if optimizer_step >= config.max_steps:
                break

    adapter_dir = ARTIFACT_ROOT / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    duration = time.time() - started
    receipt = {
        "schema": "icmat_evidenceqa_17b_training_receipt.v1",
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "status": "ADAPTER_TRAINED_VALIDATION_PENDING",
        "created_at": utc_now(),
        "identity": identity,
        "training": {
            **asdict(config),
            "optimizer": "torch.optim.AdamW",
            "quantization": "NF4_DOUBLE_QUANT_FP16_COMPUTE",
            "assistant_only_loss": True,
            "train_rows_seen_approximately": min(
                len(dataset),
                config.max_steps * config.gradient_accumulation * config.micro_batch_size,
            ),
            "optimizer_steps": optimizer_step,
            "micro_steps": micro_step,
            "duration_seconds": duration,
            "loss_first_16_mean": sum(losses[:16]) / min(16, len(losses)),
            "loss_last_16_mean": sum(losses[-16:]) / min(16, len(losses)),
            "trainable_parameters": sum(parameter.numel() for parameter in trainable),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
        },
        "adapter": {
            "path": str(adapter_dir.relative_to(ROOT)),
            "tree_sha256": sha256_tree(adapter_dir),
            "files": {
                item.name: sha256_file(item)
                for item in sorted(adapter_dir.iterdir())
                if item.is_file()
            },
        },
        "claims": {
            "x5_accessed": False,
            "x5_verified": False,
            "production_integrated": False,
            "gguf_exported": False,
        },
    }
    write_json(EVIDENCE_ROOT / "training_receipt.v1.json", receipt)
    print(json.dumps(receipt["training"], indent=2), flush=True)


JSON_OBJECT = re.compile(r"\{[^{}]*\}", re.DOTALL)
SPAN = re.compile(r"\[(E\d+\.S\d+)\]")


def parse_pointer(text: str) -> tuple[dict[str, Any] | None, str | None]:
    for match in reversed(list(JSON_OBJECT.finditer(text))):
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload, None
    return None, "no_json_object"


def expected_pointer(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(row["messages"][-1]["content"])


def evaluate(args: argparse.Namespace) -> None:
    identity = validate_inputs()
    adapter_dir = ARTIFACT_ROOT / "adapter"
    if not (adapter_dir / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("trained adapter is missing")
    seed_everything(SEED)
    tokenizer = load_tokenizer(adapter_dir)
    model = PeftModel.from_pretrained(
        load_quantized_base(training=False),
        adapter_dir,
        local_files_only=True,
    ).eval()
    rows = load_jsonl(VALIDATION_FILE)
    predictions_path = EVIDENCE_ROOT / "validation_predictions.jsonl"
    counters = {
        "rows": len(rows),
        "json_valid": 0,
        "schema_valid": 0,
        "span_from_input": 0,
        "decision_span_constraint": 0,
        "exact": 0,
        "answer_rows": 0,
        "answer_exact": 0,
        "refuse_rows": 0,
        "refuse_exact": 0,
    }
    started = time.time()
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8") as output_handle:
        for index, row in enumerate(rows, start=1):
            prompt_messages = row["messages"][:-1]
            prompt = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            text = tokenizer.decode(
                generated[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ).strip()
            parsed, parse_error = parse_pointer(text)
            expected = expected_pointer(row)
            allowed_spans = sorted(set(SPAN.findall(prompt_messages[-1]["content"])))
            json_valid = parsed is not None
            if json_valid:
                counters["json_valid"] += 1
            schema_valid = bool(
                parsed is not None
                and list(parsed.keys()) == ["task", "decision", "span_id"]
                and parsed.get("task") in {"evidence_selection", "claim_verification"}
                and parsed.get("decision") in {"ANSWER", "REFUSE"}
            )
            if schema_valid:
                counters["schema_valid"] += 1
            span_from_input = bool(
                parsed is not None
                and (
                    parsed.get("decision") != "ANSWER"
                    or parsed.get("span_id") in allowed_spans
                )
            )
            if span_from_input:
                counters["span_from_input"] += 1
            decision_span_constraint = bool(
                parsed is not None
                and (
                    (parsed.get("decision") == "REFUSE" and parsed.get("span_id") is None)
                    or (
                        parsed.get("decision") == "ANSWER"
                        and isinstance(parsed.get("span_id"), str)
                    )
                )
            )
            if decision_span_constraint:
                counters["decision_span_constraint"] += 1
            exact = parsed == expected
            if exact:
                counters["exact"] += 1
            expected_decision = expected["decision"].lower()
            counters[f"{expected_decision}_rows"] += 1
            if exact:
                counters[f"{expected_decision}_exact"] += 1
            record = {
                "example_id": row["example_id"],
                "expected": expected,
                "prediction": parsed,
                "raw_generation": text,
                "parse_error": parse_error,
                "allowed_spans": allowed_spans,
                "checks": {
                    "json_valid": json_valid,
                    "schema_valid": schema_valid,
                    "span_from_input": span_from_input,
                    "decision_span_constraint": decision_span_constraint,
                    "exact": exact,
                },
            }
            output_handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            if index == 1 or index % 25 == 0:
                print(f"validation {index}/{len(rows)}", flush=True)

    rates = {
        "json_valid_rate": counters["json_valid"] / counters["rows"],
        "schema_valid_rate": counters["schema_valid"] / counters["rows"],
        "span_from_input_rate": counters["span_from_input"] / counters["rows"],
        "decision_span_constraint_rate": counters["decision_span_constraint"]
        / counters["rows"],
        "exact_rate": counters["exact"] / counters["rows"],
        "answer_exact_rate": counters["answer_exact"] / max(1, counters["answer_rows"]),
        "refuse_exact_rate": counters["refuse_exact"] / max(1, counters["refuse_rows"]),
    }
    hard_gates = {
        "json_valid_ge_0_95": rates["json_valid_rate"] >= 0.95,
        "span_from_input_ge_0_99": rates["span_from_input_rate"] >= 0.99,
        "decision_span_constraint_eq_1": rates["decision_span_constraint_rate"] == 1.0,
        "exact_ge_0_75": rates["exact_rate"] >= 0.75,
    }
    passed = all(hard_gates.values())
    receipt = {
        "schema": "icmat_evidenceqa_17b_validation_receipt.v1",
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "status": "PC_RUNNABLE_X5_PENDING" if passed else "PC_TRAINED_QUALITY_HOLD",
        "created_at": utc_now(),
        "identity": identity,
        "adapter_tree_sha256": sha256_tree(adapter_dir),
        "validation_blind_during_training": True,
        "validation": {
            "counters": counters,
            "rates": rates,
            "hard_gates": hard_gates,
            "duration_seconds": time.time() - started,
            "predictions_path": str(predictions_path.relative_to(ROOT)),
            "predictions_sha256": sha256_file(predictions_path),
        },
        "claims": {
            "x5_accessed": False,
            "x5_verified": False,
            "production_integrated": False,
            "status_ceiling": "PC_RUNNABLE_X5_PENDING",
        },
    }
    write_json(EVIDENCE_ROOT / "validation_receipt.v1.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


def merge(args: argparse.Namespace) -> None:
    validate_inputs()
    adapter_dir = ARTIFACT_ROOT / "adapter"
    merged_dir = ARTIFACT_ROOT / "merged_hf"
    if not (adapter_dir / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("trained adapter is missing")
    started = time.time()
    tokenizer = load_tokenizer(adapter_dir)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        local_files_only=True,
        dtype=torch.float16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, adapter_dir, local_files_only=True)
    merged = model.merge_and_unload(safe_merge=True)
    merged.save_pretrained(
        merged_dir,
        safe_serialization=True,
        max_shard_size="2GB",
    )
    tokenizer.save_pretrained(merged_dir)
    receipt = {
        "schema": "icmat_evidenceqa_17b_merge_receipt.v1",
        "model_id": MODEL_ID,
        "status": "MERGED_HF_PC_EXPORT_READY",
        "created_at": utc_now(),
        "duration_seconds": time.time() - started,
        "adapter_tree_sha256": sha256_tree(adapter_dir),
        "merged_hf": {
            "path": str(merged_dir.relative_to(ROOT)),
            "tree_sha256": sha256_tree(merged_dir),
            "files": {
                item.name: sha256_file(item)
                for item in sorted(merged_dir.iterdir())
                if item.is_file()
            },
        },
        "claims": {"x5_accessed": False, "production_integrated": False},
    }
    write_json(EVIDENCE_ROOT / "merge_receipt.v1.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


def run_command(command: list[str], *, env: dict[str, str] | None = None) -> dict[str, Any]:
    started = time.time()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "duration_seconds": time.time() - started,
        "stdout_tail": completed.stdout[-12000:],
        "stderr_tail": completed.stderr[-12000:],
    }


def export_gguf(args: argparse.Namespace) -> None:
    merged_dir = ARTIFACT_ROOT / "merged_hf"
    if not (merged_dir / "config.json").is_file():
        raise FileNotFoundError("merged HF model is missing")
    gguf_dir = ARTIFACT_ROOT / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    f16_path = gguf_dir / "ICMat-Qwen3-1.7B-EvidenceQA-F16.gguf"
    q4_path = gguf_dir / "ICMat-Qwen3-1.7B-EvidenceQA-Q4_K_M.gguf"
    conversion_env = os.environ.copy()
    gguf_python = str(LLAMA_ROOT / "gguf-py")
    conversion_env["PYTHONPATH"] = (
        gguf_python
        if not conversion_env.get("PYTHONPATH")
        else gguf_python + os.pathsep + conversion_env["PYTHONPATH"]
    )
    convert = run_command(
        [
            sys.executable,
            str(LLAMA_ROOT / "convert_hf_to_gguf.py"),
            str(merged_dir),
            "--outfile",
            str(f16_path),
            "--outtype",
            "f16",
        ],
        env=conversion_env,
    )
    if convert["returncode"] != 0:
        write_json(EVIDENCE_ROOT / "gguf_receipt.v1.json", {"status": "GGUF_CONVERSION_FAILED", "convert": convert})
        raise RuntimeError("GGUF F16 conversion failed")
    quantize = run_command(
        [
            str(LLAMA_RUNTIME / "llama-quantize.exe"),
            str(f16_path),
            str(q4_path),
            "Q4_K_M",
        ]
    )
    if quantize["returncode"] != 0:
        write_json(
            EVIDENCE_ROOT / "gguf_receipt.v1.json",
            {"status": "GGUF_QUANTIZATION_FAILED", "convert": convert, "quantize": quantize},
        )
        raise RuntimeError("Q4_K_M quantization failed")

    tokenizer = load_tokenizer(merged_dir)
    validation_row = load_jsonl(VALIDATION_FILE)[0]
    prompt = tokenizer.apply_chat_template(
        validation_row["messages"][:-1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_path = EVIDENCE_ROOT / "llama_cpp_cpu_smoke_prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    cpu_smoke = run_command(
        [
            str(LLAMA_RUNTIME / "llama-cli.exe"),
            "-m",
            str(q4_path),
            "-f",
            str(prompt_path),
            "-n",
            str(args.max_new_tokens),
            "-ngl",
            "0",
            "--threads",
            str(args.cpu_threads),
            "--temp",
            "0",
            "--seed",
            str(SEED),
            "--no-display-prompt",
        ]
    )
    parsed, parse_error = parse_pointer(cpu_smoke["stdout_tail"])
    passed = cpu_smoke["returncode"] == 0 and parsed is not None
    receipt = {
        "schema": "icmat_evidenceqa_17b_gguf_receipt.v1",
        "model_id": MODEL_ID,
        "status": "PC_RUNNABLE_X5_PENDING" if passed else "GGUF_EXPORTED_CPU_SMOKE_HOLD",
        "created_at": utc_now(),
        "convert": convert,
        "quantize": quantize,
        "gguf": {
            "f16": {
                "path": str(f16_path.relative_to(ROOT)),
                "bytes": f16_path.stat().st_size,
                "sha256": sha256_file(f16_path),
            },
            "q4_k_m": {
                "path": str(q4_path.relative_to(ROOT)),
                "bytes": q4_path.stat().st_size,
                "sha256": sha256_file(q4_path),
            },
        },
        "cpu_smoke": {
            **cpu_smoke,
            "parsed_pointer": parsed,
            "parse_error": parse_error,
            "actual_cpu_only": True,
            "n_gpu_layers": 0,
        },
        "claims": {
            "x5_accessed": False,
            "x5_verified": False,
            "production_integrated": False,
            "status_ceiling": "PC_RUNNABLE_X5_PENDING",
        },
    }
    write_json(EVIDENCE_ROOT / "gguf_receipt.v1.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


def verify(args: argparse.Namespace) -> None:
    identity = validate_inputs()
    required = [
        ARTIFACT_ROOT / "adapter/adapter_model.safetensors",
        EVIDENCE_ROOT / "training_receipt.v1.json",
        EVIDENCE_ROOT / "validation_receipt.v1.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing outputs: {missing}")
    validation = json.loads((EVIDENCE_ROOT / "validation_receipt.v1.json").read_text(encoding="utf-8"))
    status = validation["status"]
    receipt = {
        "schema": "icmat_evidenceqa_17b_final_receipt.v1",
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "status": status,
        "created_at": utc_now(),
        "identity": identity,
        "artifact_tree_sha256": sha256_tree(ARTIFACT_ROOT),
        "evidence_files": {
            item.name: sha256_file(item)
            for item in sorted(EVIDENCE_ROOT.iterdir())
            if item.is_file() and item.name != "final_receipt.v1.json"
        },
        "claims": {
            "real_adapter_trained": True,
            "fixed_validation_executed": True,
            "x5_accessed": False,
            "x5_verified": False,
            "production_integrated": False,
            "status_ceiling": "PC_RUNNABLE_X5_PENDING",
        },
    }
    write_json(EVIDENCE_ROOT / "final_receipt.v1.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["train", "eval", "merge", "gguf", "verify", "all"],
        default="all",
    )
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--cpu-threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stages = ["train", "eval", "merge", "gguf", "verify"] if args.stage == "all" else [args.stage]
    handlers = {
        "train": train,
        "eval": evaluate,
        "merge": merge,
        "gguf": export_gguf,
        "verify": verify,
    }
    for stage in stages:
        print(json.dumps({"stage": stage, "started_at": utc_now()}), flush=True)
        handlers[stage](args)


if __name__ == "__main__":
    with contextlib.suppress(BrokenPipeError):
        main()
