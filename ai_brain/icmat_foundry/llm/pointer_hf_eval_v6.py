"""Non-blind HF and fixture evaluator for the ICMat v6 pointer contract.

The model sees only the two target-free messages embedded in
``compiler_prompt``. Its raw pointer is compiled before any expected value is
inspected. Expected pointers and answers are used only for post-generation
scoring; they are never passed to the model or used to repair model output.

This evaluator intentionally refuses blind data. It supports validation
(optionally sampled from the start of the immutable file) and full calibration
only.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shutil
import time
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from icmat_foundry.llm import evidence_pointer_v6
from icmat_foundry.llm.evidence_pointer_v6 import (
    PROMPT_SCHEMA,
    compile_pointer,
    validate_student_answer,
)

EVALUATOR_VERSION = "icmat-pointer-hf-evaluator-v6.0.0"
SAMPLE_SCHEMA = "icmat_pointer_hf_eval_sample.v6"
SUMMARY_SCHEMA = "icmat_pointer_hf_eval_summary.v6"
RUN_RECEIPT_SCHEMA = "icmat_pointer_hf_eval_run_receipt.v6"
FIXTURE_GENERATION_SCHEMA = "icmat_pointer_generation_fixture.v6"

SUPPORTED_SPLITS = frozenset({"validation", "calibration"})
SUPPORTED_BACKENDS = frozenset({"fixture", "hf_model"})
MAX_INPUT_TOKENS = 1536
MAX_NEW_TOKENS = 64
MAX_DATASET_BYTES = 128 * 1024 * 1024
MAX_FIXTURE_BYTES = 32 * 1024 * 1024
MAX_ERROR_CHARS = 1000
TRUSTED_FINISH_REASONS = frozenset({"eos_token", "stop", "end_turn"})
_PATH_TOKEN = re.compile(r"[a-z0-9]+")
_PROTECTED_PATH_TOKENS = frozenset({"blind", "sealed"})

_REQUIRED_DATASET_KEYS = frozenset(
    {
        "example_id",
        "split",
        "task",
        "compiler_prompt",
        "compiler_evidence",
    }
)
_FIXTURE_REQUIRED_KEYS = frozenset(
    {"schema", "example_id", "raw_pointer", "finish_reason"}
)
_FIXTURE_OPTIONAL_KEYS = frozenset(
    {
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "generation_error",
    }
)
_POINTER_METRICS = (
    "parse_valid",
    "task_exact",
    "decision_exact",
    "span_exact",
    "value_exact",
    "strict_exact",
    "compiler_accepted",
)
_COMPILED_METRICS = (
    "json_available",
    "schema_valid",
    "schema_exact",
    "decision_exact",
    "task_exact",
    "claim_exact",
    "verdict_exact",
    "citation_exact",
    "provenance_exact",
    "strict_exact",
)


class PointerHFEvalV6Error(ValueError):
    """Raised when a v6 evaluation contract or immutable input is invalid."""


@dataclass(frozen=True)
class DatasetRowV6:
    """One structured, non-blind evaluator row."""

    example_id: str
    split: str
    compiler_prompt: dict[str, Any]
    compiler_evidence: list[dict[str, Any]]
    expected_pointer: Any
    expected_answer: Any | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class GenerationRequestV6:
    """Target-free generation request.

    This type deliberately has no expected pointer, expected answer, decision,
    span, or assistant message field.
    """

    example_id: str
    messages: tuple[dict[str, str], dict[str, str]]


@dataclass(frozen=True)
class GenerationResultV6:
    """One free-generation result or one explicit fixture result."""

    raw_pointer: str
    finish_reason: str
    finish_category: str
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    generation_error: str | None


@dataclass(frozen=True)
class DatasetSelectionV6:
    """Selected rows and the exact non-blind file that was opened."""

    dataset_dir: Path
    split_path: Path
    split_sha256: str
    split_bytes: int
    rows_total: int
    rows: tuple[DatasetRowV6, ...]


def canonical_json(value: Any) -> str:
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


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (canonical_json(dict(record)) + "\n").encode("utf-8")
        for record in records
    )


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PointerHFEvalV6Error(f"duplicate JSON key rejected: {key}")
        result[key] = value
    return result


def _parse_json_object(text: str, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(text, object_pairs_hook=_reject_duplicate_pairs)
    except (json.JSONDecodeError, PointerHFEvalV6Error) as exc:
        raise PointerHFEvalV6Error(f"{field} is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise PointerHFEvalV6Error(f"{field} must be a JSON object")
    try:
        canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise PointerHFEvalV6Error(
            f"{field} must contain only finite JSON values"
        ) from exc
    return value


def _reject_blind_label(path: Path, *, field: str) -> None:
    for part in path.parts:
        if set(_PATH_TOKEN.findall(part.casefold())) & _PROTECTED_PATH_TOKENS:
            raise PointerHFEvalV6Error(
                f"{field} must not reference a blind-labelled path"
            )


def _validate_split(split: str, max_samples: int | None) -> None:
    if split not in SUPPORTED_SPLITS:
        raise PointerHFEvalV6Error(
            "blind and unsupported splits are refused; use validation or calibration"
        )
    if max_samples is not None and (
        isinstance(max_samples, bool)
        or not isinstance(max_samples, int)
        or max_samples <= 0
    ):
        raise PointerHFEvalV6Error("max_samples must be a positive integer")
    if split == "calibration" and max_samples is not None:
        raise PointerHFEvalV6Error(
            "calibration must run the complete split; max_samples is forbidden"
        )


def _model_messages(prompt: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    if prompt.get("schema") != PROMPT_SCHEMA:
        raise PointerHFEvalV6Error(
            f"compiler_prompt.schema must be {PROMPT_SCHEMA}"
        )
    messages = prompt.get("messages")
    if (
        not isinstance(messages, Sequence)
        or isinstance(messages, (str, bytes))
        or len(messages) != 2
    ):
        raise PointerHFEvalV6Error(
            "compiler_prompt.messages must contain exactly system and user"
        )
    normalized: list[dict[str, str]] = []
    for index, expected_role in enumerate(("system", "user")):
        message = messages[index]
        if not isinstance(message, Mapping) or set(message) != {
            "role",
            "content",
        }:
            raise PointerHFEvalV6Error(
                f"compiler_prompt.messages[{index}] keys are invalid"
            )
        role = message.get("role")
        content = message.get("content")
        if role != expected_role:
            raise PointerHFEvalV6Error(
                f"compiler_prompt.messages[{index}].role must be {expected_role}"
            )
        if not isinstance(content, str) or not content:
            raise PointerHFEvalV6Error(
                f"compiler_prompt.messages[{index}].content must be non-empty"
            )
        normalized.append({"role": expected_role, "content": content})
    return normalized[0], normalized[1]


def _validate_dataset_row(
    value: Any,
    *,
    split: str,
    line_number: int,
) -> DatasetRowV6:
    if not isinstance(value, Mapping):
        raise PointerHFEvalV6Error(
            f"{split} line {line_number} must be an object"
        )
    missing = _REQUIRED_DATASET_KEYS - set(value)
    if missing:
        raise PointerHFEvalV6Error(
            f"{split} line {line_number} misses structured fields: {sorted(missing)}"
        )
    example_id = value.get("example_id")
    if not isinstance(example_id, str) or not example_id:
        raise PointerHFEvalV6Error(
            f"{split} line {line_number} example_id must be non-empty"
        )
    if value.get("split") != split:
        raise PointerHFEvalV6Error(
            f"{example_id} split does not match selected {split}"
        )
    prompt = value.get("compiler_prompt")
    evidence = value.get("compiler_evidence")
    if not isinstance(prompt, Mapping):
        raise PointerHFEvalV6Error(
            f"{example_id} compiler_prompt must be an object"
        )
    if (
        not isinstance(evidence, Sequence)
        or isinstance(evidence, (str, bytes))
        or not evidence
        or any(not isinstance(item, Mapping) for item in evidence)
    ):
        raise PointerHFEvalV6Error(
            f"{example_id} compiler_evidence must be a non-empty object array"
        )
    _model_messages(prompt)
    try:
        prompt_clone = json.loads(canonical_json(dict(prompt)))
        evidence_clone = json.loads(
            canonical_json([dict(item) for item in evidence])
        )
        task = value["task"]
        if not isinstance(task, str) or not task:
            raise PointerHFEvalV6Error(
                f"{example_id} task must be a non-empty string"
            )
        if prompt.get("task") != task:
            raise PointerHFEvalV6Error(
                f"{example_id} task does not match compiler_prompt.task"
            )
        has_structured_gold = (
            "decision" in value or "target_span_id" in value
        )
        if has_structured_gold:
            if "decision" not in value or "target_span_id" not in value:
                raise PointerHFEvalV6Error(
                    f"{example_id} structured gold requires decision and "
                    "target_span_id"
                )
            decision = value["decision"]
            target_span_id = value["target_span_id"]
            if decision not in {"ANSWER", "REFUSE"}:
                raise PointerHFEvalV6Error(
                    f"{example_id} decision must be ANSWER or REFUSE"
                )
            if decision == "ANSWER":
                if not isinstance(target_span_id, str) or not target_span_id:
                    raise PointerHFEvalV6Error(
                        f"{example_id} ANSWER target_span_id must be non-empty"
                    )
            elif target_span_id is not None:
                raise PointerHFEvalV6Error(
                    f"{example_id} REFUSE target_span_id must be null"
                )
            expected_pointer_source: Any = {
                "task": task,
                "decision": decision,
                "span_id": target_span_id,
            }
        elif "expected_pointer" in value:
            # Compatibility for pre-r3 evaluator fixtures. The model request
            # still receives only compiler_prompt.messages.
            expected_pointer_source = value["expected_pointer"]
        else:
            raise PointerHFEvalV6Error(
                f"{example_id} misses structured gold fields"
            )
        expected_pointer = json.loads(
            canonical_json(expected_pointer_source)
        )
        expected_answer = (
            None
            if "expected_answer" not in value
            else json.loads(canonical_json(value["expected_answer"]))
        )
    except (TypeError, ValueError) as exc:
        raise PointerHFEvalV6Error(
            f"{example_id} contains non-finite structured data"
        ) from exc
    metadata = {
        key: value.get(key)
        for key in ("domain", "task", "source_id", "family_id")
        if key in value
    }
    return DatasetRowV6(
        example_id=example_id,
        split=split,
        compiler_prompt=prompt_clone,
        compiler_evidence=evidence_clone,
        expected_pointer=expected_pointer,
        expected_answer=expected_answer,
        metadata=metadata,
    )


def select_dataset(
    *,
    dataset_dir: Path,
    split: str,
    max_samples: int | None,
) -> DatasetSelectionV6:
    """Open exactly one non-blind split without listing or reading blind files."""

    _validate_split(split, max_samples)
    dataset_raw = Path(dataset_dir)
    _reject_blind_label(dataset_raw, field="dataset directory")
    if dataset_raw.is_symlink():
        raise PointerHFEvalV6Error("dataset directory must not be a symlink")
    dataset_root = dataset_raw.resolve()
    if not dataset_root.is_dir():
        raise PointerHFEvalV6Error(
            f"dataset directory is unavailable: {dataset_raw}"
        )
    split_path = dataset_root / f"{split}.jsonl"
    _reject_blind_label(split_path, field="split path")
    if split_path.is_symlink():
        raise PointerHFEvalV6Error("split file must not be a symlink")
    if not split_path.is_file():
        raise PointerHFEvalV6Error(f"split file is unavailable: {split_path}")
    split_bytes = split_path.stat().st_size
    if split_bytes <= 0 or split_bytes > MAX_DATASET_BYTES:
        raise PointerHFEvalV6Error(
            f"split bytes must be in 1..{MAX_DATASET_BYTES}"
        )

    rows: list[DatasetRowV6] = []
    observed_ids: set[str] = set()
    with split_path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise PointerHFEvalV6Error(
                    f"{split} contains blank line {line_number}"
                )
            parsed = _parse_json_object(
                line,
                field=f"{split} line {line_number}",
            )
            row = _validate_dataset_row(
                parsed,
                split=split,
                line_number=line_number,
            )
            if row.example_id in observed_ids:
                raise PointerHFEvalV6Error(
                    f"duplicate example_id: {row.example_id}"
                )
            observed_ids.add(row.example_id)
            rows.append(row)
    if not rows:
        raise PointerHFEvalV6Error(f"{split} contains no rows")
    rows_total = len(rows)
    selected = rows if max_samples is None else rows[:max_samples]
    if not selected:
        raise PointerHFEvalV6Error("selection contains no rows")
    return DatasetSelectionV6(
        dataset_dir=dataset_root,
        split_path=split_path,
        split_sha256=sha256_file(split_path),
        split_bytes=split_bytes,
        rows_total=rows_total,
        rows=tuple(selected),
    )


def _tree_inventory(path: Path) -> dict[str, Any]:
    root_raw = Path(path)
    if root_raw.is_symlink():
        raise PointerHFEvalV6Error(
            f"model artifact root must not be a symlink: {root_raw}"
        )
    root = root_raw.resolve()
    if not root.is_dir():
        raise PointerHFEvalV6Error(
            f"model artifact directory is unavailable: {root_raw}"
        )
    candidates = sorted(
        root.rglob("*"),
        key=lambda item: item.relative_to(root).as_posix(),
    )
    files: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.is_symlink():
            raise PointerHFEvalV6Error(
                f"model artifact tree contains a symlink: {candidate}"
            )
        if candidate.is_file():
            files.append(
                {
                    "path": candidate.relative_to(root).as_posix(),
                    "bytes": candidate.stat().st_size,
                    "sha256": sha256_file(candidate),
                }
            )
    if not files:
        raise PointerHFEvalV6Error(
            f"model artifact directory is empty: {root}"
        )
    return {
        "path": str(root),
        "files": files,
        "files_count": len(files),
        "tree_sha256": sha256_bytes(canonical_json(files).encode("utf-8")),
    }


def _finish_category(finish_reason: str) -> str:
    if finish_reason == "eos_token":
        return "EOS"
    if finish_reason in {"stop", "end_turn"}:
        return "NORMAL_STOP"
    if finish_reason == "length":
        return "LENGTH"
    return "ABNORMAL"


def detect_hf_finish_reason(
    *,
    generated_token_ids: Sequence[int],
    eos_token_ids: Sequence[int],
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> tuple[str, str]:
    """Classify EOS, length, and abnormal singleton generation endings."""

    ids = [int(item) for item in generated_token_ids]
    eos = {int(item) for item in eos_token_ids}
    if ids and ids[-1] in eos:
        return "eos_token", "EOS"
    if len(ids) >= max_new_tokens:
        return "length", "LENGTH"
    return "abnormal_end", "ABNORMAL"


def _normalize_eos_ids(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(int(item) for item in value)
    return (int(value),)


def _safe_generation_error(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}"
    return message[:MAX_ERROR_CHARS]


def _generation_requests(
    rows: Sequence[DatasetRowV6],
) -> tuple[GenerationRequestV6, ...]:
    requests: list[GenerationRequestV6] = []
    for row in rows:
        system, user = _model_messages(row.compiler_prompt)
        requests.append(
            GenerationRequestV6(
                example_id=row.example_id,
                messages=(system, user),
            )
        )
    return tuple(requests)


def generate_hf_model(
    requests: Sequence[GenerationRequestV6],
    *,
    base_model_dir: Path,
    adapter_dir: Path | None,
    device: str,
    seed: int,
) -> tuple[dict[str, GenerationResultV6], dict[str, Any]]:
    """Run local singleton greedy generation without target-bearing inputs."""

    if device not in {"cpu", "cuda"}:
        raise PointerHFEvalV6Error("device must be explicit cpu or cuda")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise PointerHFEvalV6Error("seed must be an integer")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise PointerHFEvalV6Error(
            "hf_model backend requires local torch and transformers"
        ) from exc
    if adapter_dir is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise PointerHFEvalV6Error(
                "adapter evaluation requires local peft"
            ) from exc
    else:
        PeftModel = None

    base_before = _tree_inventory(base_model_dir)
    adapter_before = (
        None if adapter_dir is None else _tree_inventory(adapter_dir)
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise PointerHFEvalV6Error("CUDA was requested but is unavailable")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(base_model_dir).resolve()),
        local_files_only=True,
        trust_remote_code=False,
    )
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(Path(base_model_dir).resolve()),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=dtype,
    )
    if adapter_dir is not None:
        assert PeftModel is not None
        model = PeftModel.from_pretrained(
            model,
            str(Path(adapter_dir).resolve()),
            local_files_only=True,
            is_trainable=False,
        )
    model.to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    eos_ids = _normalize_eos_ids(tokenizer.eos_token_id)
    if not eos_ids:
        raise PointerHFEvalV6Error("tokenizer must define at least one EOS token")

    results: dict[str, GenerationResultV6] = {}
    started = time.perf_counter()
    with torch.inference_mode():
        for request in requests:
            prompt = tokenizer.apply_chat_template(
                [dict(message) for message in request.messages],
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_tokens = int(encoded["input_ids"].shape[-1])
            if input_tokens > MAX_INPUT_TOKENS:
                raise PointerHFEvalV6Error(
                    f"{request.example_id} prompt has {input_tokens} tokens; "
                    f"limit is {MAX_INPUT_TOKENS}"
                )
            encoded = {
                key: tensor.to(device)
                for key, tensor in encoded.items()
            }
            sample_started = time.perf_counter()
            try:
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    return_dict_in_generate=True,
                )
                sequences = generated.sequences
                if int(sequences.shape[0]) != 1:
                    raise PointerHFEvalV6Error(
                        "singleton generation returned multiple sequences"
                    )
                generated_tensor = sequences[0, input_tokens:]
                token_ids = [int(item) for item in generated_tensor.tolist()]
                finish_reason, finish_category = detect_hf_finish_reason(
                    generated_token_ids=token_ids,
                    eos_token_ids=eos_ids,
                )
                raw_pointer = tokenizer.decode(
                    token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
                generation_error = None
                output_tokens = len(token_ids)
            except Exception as exc:  # fail closed per sample
                finish_reason = "generation_exception"
                finish_category = "ABNORMAL"
                raw_pointer = ""
                generation_error = _safe_generation_error(exc)
                output_tokens = None
            latency_ms = (time.perf_counter() - sample_started) * 1000.0
            results[request.example_id] = GenerationResultV6(
                raw_pointer=raw_pointer,
                finish_reason=finish_reason,
                finish_category=finish_category,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                generation_error=generation_error,
            )
    elapsed_seconds = time.perf_counter() - started

    base_after = _tree_inventory(base_model_dir)
    adapter_after = (
        None if adapter_dir is None else _tree_inventory(adapter_dir)
    )
    if base_after["tree_sha256"] != base_before["tree_sha256"]:
        raise PointerHFEvalV6Error("base model changed during generation")
    if (
        None if adapter_after is None else adapter_after["tree_sha256"]
    ) != (
        None if adapter_before is None else adapter_before["tree_sha256"]
    ):
        raise PointerHFEvalV6Error("adapter changed during generation")

    backend = {
        "mode": "hf_model",
        "subject": "adapter" if adapter_before is not None else "base",
        "device": device,
        "seed": seed,
        "model": {
            "base": base_before,
            "adapter": adapter_before,
            "inventories_unchanged_after_generation": True,
        },
        "decoding": {
            "batch_size": 1,
            "singleton": True,
            "do_sample": False,
            "num_beams": 1,
            "greedy": True,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "use_cache": True,
            "chat_template": "base_model_tokenizer.apply_chat_template",
            "add_generation_prompt": True,
            "tokenizer_add_special_tokens": False,
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
            "dtype": "float16" if device == "cuda" else "float32",
        },
        "samples_generated": len(results),
        "elapsed_seconds": elapsed_seconds,
        "local_files_only": True,
        "network_allowed": False,
        "assistant_target_visible": False,
    }
    return results, backend


def _validate_fixture_record(
    value: Any,
    *,
    line_number: int,
) -> tuple[str, GenerationResultV6]:
    if not isinstance(value, Mapping):
        raise PointerHFEvalV6Error(
            f"fixture line {line_number} must be an object"
        )
    keys = set(value)
    missing = _FIXTURE_REQUIRED_KEYS - keys
    extra = keys - _FIXTURE_REQUIRED_KEYS - _FIXTURE_OPTIONAL_KEYS
    if missing or extra:
        raise PointerHFEvalV6Error(
            f"fixture line {line_number} keys mismatch; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if value.get("schema") != FIXTURE_GENERATION_SCHEMA:
        raise PointerHFEvalV6Error(
            f"fixture line {line_number} schema is invalid"
        )
    example_id = value.get("example_id")
    raw_pointer = value.get("raw_pointer")
    finish_reason = value.get("finish_reason")
    if not isinstance(example_id, str) or not example_id:
        raise PointerHFEvalV6Error(
            f"fixture line {line_number} example_id is invalid"
        )
    if not isinstance(raw_pointer, str):
        raise PointerHFEvalV6Error(
            f"fixture {example_id} raw_pointer must be a string"
        )
    if not isinstance(finish_reason, str) or not finish_reason:
        raise PointerHFEvalV6Error(
            f"fixture {example_id} finish_reason must be non-empty"
        )
    latency = value.get("latency_ms", 0.0)
    if (
        isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or not math.isfinite(float(latency))
        or float(latency) < 0
    ):
        raise PointerHFEvalV6Error(
            f"fixture {example_id} latency_ms is invalid"
        )
    input_tokens = value.get("input_tokens")
    output_tokens = value.get("output_tokens")
    for field, token_count in (
        ("input_tokens", input_tokens),
        ("output_tokens", output_tokens),
    ):
        if token_count is not None and (
            isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count < 0
        ):
            raise PointerHFEvalV6Error(
                f"fixture {example_id} {field} is invalid"
            )
    generation_error = value.get("generation_error")
    if generation_error is not None and not isinstance(generation_error, str):
        raise PointerHFEvalV6Error(
            f"fixture {example_id} generation_error is invalid"
        )
    return example_id, GenerationResultV6(
        raw_pointer=raw_pointer,
        finish_reason=finish_reason,
        finish_category=_finish_category(finish_reason),
        latency_ms=float(latency),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        generation_error=generation_error,
    )


def load_fixture_generations(
    *,
    fixture_path: Path,
    expected_example_ids: Sequence[str],
) -> tuple[dict[str, GenerationResultV6], dict[str, Any]]:
    path_raw = Path(fixture_path)
    _reject_blind_label(path_raw, field="fixture path")
    if path_raw.is_symlink():
        raise PointerHFEvalV6Error("fixture path must not be a symlink")
    path = path_raw.resolve()
    if not path.is_file() or path.suffix.casefold() != ".jsonl":
        raise PointerHFEvalV6Error(
            "fixture path must be an existing JSONL file"
        )
    size = path.stat().st_size
    if size <= 0 or size > MAX_FIXTURE_BYTES:
        raise PointerHFEvalV6Error(
            f"fixture bytes must be in 1..{MAX_FIXTURE_BYTES}"
        )
    generations: dict[str, GenerationResultV6] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise PointerHFEvalV6Error(
                    f"fixture contains blank line {line_number}"
                )
            parsed = _parse_json_object(
                line,
                field=f"fixture line {line_number}",
            )
            example_id, result = _validate_fixture_record(
                parsed,
                line_number=line_number,
            )
            if example_id in generations:
                raise PointerHFEvalV6Error(
                    f"duplicate fixture example_id: {example_id}"
                )
            generations[example_id] = result
    expected = set(expected_example_ids)
    observed = set(generations)
    if observed != expected:
        raise PointerHFEvalV6Error(
            "fixture membership mismatch; "
            f"missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    return generations, {
        "mode": "fixture",
        "fixture": {
            "path": str(path),
            "bytes": size,
            "sha256": sha256_file(path),
        },
        "model": {"base": None, "adapter": None},
        "decoding": {
            "recorded_from_fixture": True,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "samples_generated": 0,
        "local_files_only": True,
        "network_allowed": False,
        "assistant_target_visible": False,
        "model_quality_claim_allowed": False,
    }


def _expected_after_candidate(
    row: DatasetRowV6,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_compilation = compile_pointer(
        prompt=row.compiler_prompt,
        evidence=row.compiler_evidence,
        raw_pointer=row.expected_pointer,
        finish_reason="eos_token",
    )
    if expected_compilation.get("status") != "COMPILED":
        reason = expected_compilation.get("parse_reason")
        raise PointerHFEvalV6Error(
            f"{row.example_id} expected pointer does not compile: {reason}"
        )
    expected_pointer = expected_compilation.get("parsed_pointer")
    expected_answer = expected_compilation.get("compiled_answer")
    if not isinstance(expected_pointer, Mapping) or not isinstance(
        expected_answer,
        Mapping,
    ):
        raise PointerHFEvalV6Error(
            f"{row.example_id} expected compilation is incomplete"
        )
    expected_pointer = dict(expected_pointer)
    expected_answer = dict(expected_answer)
    if row.expected_answer is not None:
        if not isinstance(row.expected_answer, Mapping):
            raise PointerHFEvalV6Error(
                f"{row.example_id} expected_answer must be an object"
            )
        if dict(row.expected_answer) != expected_answer:
            raise PointerHFEvalV6Error(
                f"{row.example_id} expected_answer does not match the "
                "deterministic expected pointer compilation"
            )
    return expected_pointer, expected_answer


def _score_row(
    *,
    row: DatasetRowV6,
    generation: GenerationResultV6,
    bindings: Mapping[str, Any],
    backend_mode: str,
) -> dict[str, Any]:
    # Candidate compilation must happen before expected/gold is inspected.
    candidate = compile_pointer(
        prompt=row.compiler_prompt,
        evidence=row.compiler_evidence,
        raw_pointer=generation.raw_pointer,
        finish_reason=generation.finish_reason,
    )
    expected_pointer, expected_answer = _expected_after_candidate(row)

    parsed = candidate.get("parsed_pointer")
    parsed_mapping = parsed if isinstance(parsed, Mapping) else None
    accepted = candidate.get("status") == "COMPILED"

    def pointer_exact(field: str) -> bool:
        return (
            parsed_mapping is not None
            and parsed_mapping.get(field) == expected_pointer.get(field)
        )

    pointer_value_exact = (
        parsed_mapping is not None and dict(parsed_mapping) == expected_pointer
    )
    pointer_metrics = {
        "parse_valid": parsed_mapping is not None,
        "task_exact": pointer_exact("task"),
        "decision_exact": pointer_exact("decision"),
        "span_exact": pointer_exact("span_id"),
        "value_exact": pointer_value_exact,
        "strict_exact": bool(accepted and pointer_value_exact),
        "compiler_accepted": accepted,
    }

    prediction = candidate.get("compiled_answer")
    prediction_mapping = (
        prediction if isinstance(prediction, Mapping) else None
    )
    schema_errors = (
        validate_student_answer(prediction_mapping)
        if prediction_mapping is not None
        else ["compiled answer is unavailable"]
    )

    def compiled_exact(field: str) -> bool:
        return bool(
            accepted
            and prediction_mapping is not None
            and prediction_mapping.get(field) == expected_answer.get(field)
        )

    compiled_metrics = {
        "json_available": prediction_mapping is not None,
        "schema_valid": bool(
            accepted and prediction_mapping is not None and not schema_errors
        ),
        "schema_exact": compiled_exact("schema"),
        "decision_exact": compiled_exact("decision"),
        "task_exact": compiled_exact("task"),
        "claim_exact": compiled_exact("claim"),
        "verdict_exact": compiled_exact("verdict"),
        "citation_exact": compiled_exact("evidence_ids"),
        "provenance_exact": compiled_exact("provenance"),
        "strict_exact": bool(
            accepted
            and prediction_mapping is not None
            and dict(prediction_mapping) == expected_answer
        ),
        "unsupported_wrong_answer": bool(
            accepted
            and expected_answer.get("decision") == "REFUSE"
            and prediction_mapping is not None
            and prediction_mapping.get("decision") == "ANSWER"
        ),
    }
    return {
        "schema": SAMPLE_SCHEMA,
        "evaluator_version": EVALUATOR_VERSION,
        "example_id": row.example_id,
        "split": row.split,
        "metadata": dict(row.metadata),
        "backend": backend_mode,
        "generation": {
            "raw_pointer": generation.raw_pointer,
            "raw_pointer_sha256": sha256_bytes(
                generation.raw_pointer.encode("utf-8")
            ),
            "finish_reason": generation.finish_reason,
            "finish_category": generation.finish_category,
            "trusted_finish_reason": (
                generation.finish_reason in TRUSTED_FINISH_REASONS
            ),
            "latency_ms": generation.latency_ms,
            "input_tokens": generation.input_tokens,
            "output_tokens": generation.output_tokens,
            "generation_error": generation.generation_error,
        },
        "compilation": candidate,
        "expected": {
            "pointer": expected_pointer,
            "answer": expected_answer,
            "access_phase": "POST_GENERATION_SCORING_ONLY",
        },
        "pointer_metrics": pointer_metrics,
        "compiled_metrics": compiled_metrics,
        "compiled_schema_errors": schema_errors,
        "bindings": json.loads(canonical_json(bindings)),
        "data_flow": {
            "model_input_fields": [
                "compiler_prompt.messages[0]",
                "compiler_prompt.messages[1]",
            ],
            "compiler_input_fields": [
                "compiler_prompt",
                "compiler_evidence",
                "raw_pointer",
                "finish_reason",
            ],
            "expected_passed_to_model": False,
            "expected_passed_to_candidate_compiler": False,
            "gold_repair_applied": False,
            "assistant_target_visible": False,
            "blind_data_accessed": False,
        },
    }


def _metric(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _aggregate_flags(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    names: Sequence[str],
) -> dict[str, dict[str, Any]]:
    return {
        name: _metric(
            sum(bool(row[field][name]) for row in rows),
            len(rows),
        )
        for name in names
    }


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _decision_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    answer_rows = [
        row
        for row in rows
        if row["expected"]["answer"]["decision"] == "ANSWER"
    ]
    refuse_rows = [
        row
        for row in rows
        if row["expected"]["answer"]["decision"] == "REFUSE"
    ]

    def predicted_decision(row: Mapping[str, Any]) -> str | None:
        if not row["pointer_metrics"]["compiler_accepted"]:
            return None
        parsed = row["compilation"].get("parsed_pointer")
        return parsed.get("decision") if isinstance(parsed, Mapping) else None

    tp = sum(predicted_decision(row) == "REFUSE" for row in refuse_rows)
    fp = sum(predicted_decision(row) == "REFUSE" for row in answer_rows)
    fn = len(refuse_rows) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "answer_pointer_strict_accuracy": _metric(
            sum(
                bool(row["pointer_metrics"]["strict_exact"])
                for row in answer_rows
            ),
            len(answer_rows),
        ),
        "answer_compiled_strict_accuracy": _metric(
            sum(
                bool(row["compiled_metrics"]["strict_exact"])
                for row in answer_rows
            ),
            len(answer_rows),
        ),
        "refuse": {
            "examples": len(refuse_rows),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "unsupported_wrong_answer_rate": _metric(
            sum(
                bool(row["compiled_metrics"]["unsupported_wrong_answer"])
                for row in refuse_rows
            ),
            len(refuse_rows),
        ),
    }


def _summarize(
    rows: Sequence[Mapping[str, Any]],
    *,
    selection: DatasetSelectionV6,
    backend: Mapping[str, Any],
    max_samples: int | None,
) -> dict[str, Any]:
    latencies = [float(row["generation"]["latency_ms"]) for row in rows]
    finish_reasons = Counter(
        str(row["generation"]["finish_reason"]) for row in rows
    )
    finish_categories = Counter(
        str(row["generation"]["finish_category"]) for row in rows
    )
    parse_reasons = Counter(
        str(row["compilation"]["parse_reason"]["code"]) for row in rows
    )
    pointer_metrics = _aggregate_flags(
        rows,
        field="pointer_metrics",
        names=_POINTER_METRICS,
    )
    compiled_metrics = _aggregate_flags(
        rows,
        field="compiled_metrics",
        names=_COMPILED_METRICS,
    )
    decision = _decision_summary(rows)
    return {
        "schema": SUMMARY_SCHEMA,
        "status": f"{selection.rows[0].split.upper()}_EVALUATION_COMPLETE",
        "evaluator_version": EVALUATOR_VERSION,
        "split": selection.rows[0].split,
        "backend": backend["mode"],
        "selection": {
            "rows_in_file": selection.rows_total,
            "rows_evaluated": len(rows),
            "max_samples": max_samples,
            "complete_split": len(rows) == selection.rows_total,
            "calibration_full_split_required": (
                selection.rows[0].split == "calibration"
            ),
        },
        "pointer_metrics": pointer_metrics,
        "compiled_metrics": compiled_metrics,
        "answer_pointer_strict_accuracy": decision[
            "answer_pointer_strict_accuracy"
        ],
        "answer_compiled_strict_accuracy": decision[
            "answer_compiled_strict_accuracy"
        ],
        "refuse": decision["refuse"],
        "unsupported_wrong_answer_rate": decision[
            "unsupported_wrong_answer_rate"
        ],
        "finish": {
            "reason_counts": dict(sorted(finish_reasons.items())),
            "category_counts": dict(sorted(finish_categories.items())),
            "eos": finish_categories.get("EOS", 0),
            "length": finish_categories.get("LENGTH", 0),
            "abnormal": finish_categories.get("ABNORMAL", 0),
        },
        "compiler": {
            "accepted": sum(
                bool(row["pointer_metrics"]["compiler_accepted"])
                for row in rows
            ),
            "fail_closed": sum(
                bool(row["compilation"]["fail_closed"]) for row in rows
            ),
            "reason_counts": dict(sorted(parse_reasons.items())),
        },
        "latency_ms": {
            "count": len(latencies),
            "total": sum(latencies),
            "mean": sum(latencies) / len(latencies),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies),
        },
        "execution_boundaries": {
            "max_input_tokens": MAX_INPUT_TOKENS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "greedy": True,
            "singleton": True,
            "model_saw_system_user_only": True,
            "expected_used_for_post_generation_scoring_only": True,
            "gold_repair_applied": False,
            "blind_split_supported": False,
            "blind_data_accessed": False,
            "promotion_authorized": False,
        },
        "claim_boundary": (
            "NONBLIND_POINTER_EVALUATION_ONLY_NOT_BLIND_X5_OR_PRODUCTION_EVIDENCE"
        ),
    }


def _source_bindings(runner_path: Path | None) -> dict[str, Any]:
    evaluator_path = Path(__file__).resolve()
    compiler_path = Path(evidence_pointer_v6.__file__).resolve()
    runner_resolved = (
        None if runner_path is None else Path(runner_path).resolve()
    )
    if runner_resolved is not None and not runner_resolved.is_file():
        raise PointerHFEvalV6Error(
            f"runner source is unavailable: {runner_resolved}"
        )
    return {
        "evaluator": {
            "path": str(evaluator_path),
            "sha256": sha256_file(evaluator_path),
        },
        "compiler": {
            "path": str(compiler_path),
            "sha256": sha256_file(compiler_path),
        },
        "runner": (
            None
            if runner_resolved is None
            else {
                "path": str(runner_resolved),
                "sha256": sha256_file(runner_resolved),
            }
        ),
    }


def _bindings_for_samples(
    *,
    backend: Mapping[str, Any],
    code: Mapping[str, Any],
) -> dict[str, Any]:
    model = backend.get("model")
    base = model.get("base") if isinstance(model, Mapping) else None
    adapter = model.get("adapter") if isinstance(model, Mapping) else None
    return {
        "base_model_tree_sha256": (
            base.get("tree_sha256") if isinstance(base, Mapping) else None
        ),
        "adapter_tree_sha256": (
            adapter.get("tree_sha256")
            if isinstance(adapter, Mapping)
            else None
        ),
        "evaluator_source_sha256": code["evaluator"]["sha256"],
        "compiler_source_sha256": code["compiler"]["sha256"],
        "runner_source_sha256": (
            None if code["runner"] is None else code["runner"]["sha256"]
        ),
    }


def run_evaluation(
    *,
    dataset_dir: Path,
    split: str,
    output_dir: Path,
    backend_mode: str,
    fixture_path: Path | None = None,
    base_model_dir: Path | None = None,
    adapter_dir: Path | None = None,
    device: str | None = None,
    seed: int = 20260729,
    max_samples: int | None = None,
    runner_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one non-blind split and atomically publish all evidence."""

    # Refuse blind before resolving or opening any caller-supplied path.
    _validate_split(split, max_samples)
    if backend_mode not in SUPPORTED_BACKENDS:
        raise PointerHFEvalV6Error(
            f"backend must be one of {sorted(SUPPORTED_BACKENDS)}"
        )
    output_raw = Path(output_dir)
    _reject_blind_label(output_raw, field="output directory")
    output = output_raw.resolve()
    if output.exists():
        raise PointerHFEvalV6Error(
            f"output directory already exists: {output}"
        )

    code_before = _source_bindings(runner_path)
    selection = select_dataset(
        dataset_dir=dataset_dir,
        split=split,
        max_samples=max_samples,
    )
    requests = _generation_requests(selection.rows)
    if backend_mode == "fixture":
        if fixture_path is None:
            raise PointerHFEvalV6Error(
                "fixture backend requires fixture_path"
            )
        if base_model_dir is not None or adapter_dir is not None or device is not None:
            raise PointerHFEvalV6Error(
                "fixture backend rejects model, adapter, and device arguments"
            )
        generations, backend = load_fixture_generations(
            fixture_path=fixture_path,
            expected_example_ids=[request.example_id for request in requests],
        )
    else:
        if fixture_path is not None:
            raise PointerHFEvalV6Error(
                "hf_model backend rejects fixture_path"
            )
        if base_model_dir is None or device is None:
            raise PointerHFEvalV6Error(
                "hf_model backend requires base_model_dir and explicit device"
            )
        generations, backend = generate_hf_model(
            requests,
            base_model_dir=base_model_dir,
            adapter_dir=adapter_dir,
            device=device,
            seed=seed,
        )

    expected_ids = {row.example_id for row in selection.rows}
    if set(generations) != expected_ids:
        raise PointerHFEvalV6Error(
            "generation membership changed after backend execution"
        )
    sample_bindings = _bindings_for_samples(
        backend=backend,
        code=code_before,
    )
    rows = [
        _score_row(
            row=row,
            generation=generations[row.example_id],
            bindings=sample_bindings,
            backend_mode=backend_mode,
        )
        for row in selection.rows
    ]
    code_after = _source_bindings(runner_path)
    if code_after != code_before:
        raise PointerHFEvalV6Error(
            "evaluator, compiler, or runner source changed during evaluation"
        )
    summary = _summarize(
        rows,
        selection=selection,
        backend=backend,
        max_samples=max_samples,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    if staging.exists():
        raise PointerHFEvalV6Error(
            f"staging directory already exists: {staging}"
        )
    staging.mkdir()
    try:
        sample_path = staging / "sample_results.v6.jsonl"
        summary_path = staging / "summary.v6.json"
        receipt_path = staging / "run_receipt.v6.json"
        sample_path.write_bytes(_jsonl_bytes(rows))
        summary_path.write_bytes(_json_bytes(summary))
        receipt = {
            "schema": RUN_RECEIPT_SCHEMA,
            "status": summary["status"],
            "created_at_utc": datetime.now(UTC).isoformat(),
            "evaluator_version": EVALUATOR_VERSION,
            "dataset": {
                "directory": str(selection.dataset_dir),
                "opened_split_path": str(selection.split_path),
                "opened_split_sha256": selection.split_sha256,
                "opened_split_bytes": selection.split_bytes,
                "rows_in_file": selection.rows_total,
                "rows_evaluated": len(rows),
                "max_samples": max_samples,
                "files_opened_by_dataset_loader": [
                    str(selection.split_path)
                ],
                "blind_data_accessed": False,
            },
            "execution": {
                "backend": backend,
                "model_request_type": "GenerationRequestV6_target_free",
                "model_input_roles": ["system", "user"],
                "expected_passed_to_model": False,
                "expected_passed_to_candidate_compiler": False,
                "gold_repair_applied": False,
                "blind_supported": False,
                "blind_data_accessed": False,
            },
            "implementation": code_before,
            "bindings": sample_bindings,
            "artifacts": {
                "sample_results.v6.jsonl": sha256_file(sample_path),
                "summary.v6.json": sha256_file(summary_path),
            },
            "claim_boundary": summary["claim_boundary"],
        }
        receipt_path.write_bytes(_json_bytes(receipt))
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "status": summary["status"],
        "output_dir": str(output),
        "examples": len(rows),
        "split": split,
        "backend": backend_mode,
        "blind_data_accessed": False,
        "hashes": {
            "sample_results.v6.jsonl": sha256_file(
                output / "sample_results.v6.jsonl"
            ),
            "summary.v6.json": sha256_file(output / "summary.v6.json"),
            "run_receipt.v6.json": sha256_file(
                output / "run_receipt.v6.json"
            ),
        },
        "summary": summary,
    }
