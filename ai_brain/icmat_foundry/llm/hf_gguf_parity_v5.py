"""Task-level parity checks for HF adapters and Q4_K_M llama.cpp runs.

The comparator consumes immutable outputs from ``evidence_eval_v5`` and
``llama_cpp_eval_v5``. It deliberately does not claim token-level prompt
identity: the two runtimes render chat messages with different tokenizer
implementations. The defensible claim is narrower: both backends received the
same message membership (bound by ``prompt_sha256``), and are rescored against
the same expected task outputs with the shared v5 scorer.
"""

from __future__ import annotations

import json
import math
import platform
import shutil
import statistics
import sys
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from icmat_foundry.llm.evidence_eval_v5 import (
    DatasetSelectionV5,
    EvidenceSampleV5,
    GenerationRequestV5,
    canonical_json,
    score_generations,
    sha256_bytes,
    sha256_file,
    validate_student_answer,
)

HF_SUMMARY_SCHEMA = "icmat_evidence_eval_summary.v5"
HF_RECEIPT_SCHEMA = "icmat_evidence_eval_run_receipt.v5"
GGUF_SUMMARY_SCHEMA = "icmat_llama_cpp_evidence_eval_summary.v5"
GGUF_RECEIPT_SCHEMA = "icmat_llama_cpp_evidence_eval_run_receipt.v5"
SAMPLE_SCHEMA = "icmat_evidence_eval_sample.v5"
REPORT_SCHEMA = "icmat_hf_gguf_task_parity_report.v5"
RECEIPT_SCHEMA = "icmat_hf_gguf_task_parity_receipt.v5"
REPORT_NAME = "hf_gguf_parity.v5.json"
RECEIPT_NAME = "parity_receipt.v5.json"
Q4_QUANTIZATION = "Q4_K_M"

SHA256_CHARS = frozenset("0123456789abcdef")
ALLOWED_ABLATIONS = frozenset(
    {"none", "evidence_removed", "evidence_swapped", "no_rag"}
)
RATE_METRICS = (
    "json_valid",
    "schema_valid",
    "schema_exact",
    "citation_exact",
    "decision_exact",
    "task_exact",
    "claim_exact",
    "verdict_exact",
    "provenance_exact",
    "strict_exact",
    "answer_accuracy",
    "unsupported_wrong_answer_rate",
)
NON_DEGRADATION_LIMITS = {
    "strict_exact": ("drop", 0.02),
    "schema_valid": ("drop", 0.02),
    "answer_accuracy": ("drop", 0.02),
    "citation_exact": ("drop", 0.02),
    "provenance_exact": ("drop", 0.02),
    "refusal_f1": ("drop", 0.02),
    "unsupported_wrong_answer_rate": ("increase", 0.01),
}
SCORING_FIELDS = (
    "prediction",
    "parse_error",
    "schema_errors",
    "predicted_decision",
    "metrics",
)
TRUNCATION_KEYS = frozenset(
    {
        "input_truncated",
        "prompt_truncated",
        "context_truncated",
        "truncated",
        "truncation",
        "truncated_tokens",
        "n_truncated",
    }
)


class HfGgufParityV5Error(ValueError):
    """Raised when an input run is invalid or the runs are not comparable."""


@dataclass(frozen=True)
class ValidatedRun:
    """One integrity-checked and independently rescored evaluation run."""

    label: str
    root: Path
    summary: dict[str, Any]
    receipt: dict[str, Any]
    rows: tuple[dict[str, Any], ...]
    recomputed_rows: tuple[dict[str, Any], ...]
    recomputed_summaries: dict[str, Any]
    hashes: dict[str, str]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


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
            raise HfGgufParityV5Error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise HfGgufParityV5Error(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HfGgufParityV5Error(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise HfGgufParityV5Error(f"{label} must contain one JSON object")
    return value


def _load_jsonl(path: Path, *, label: str) -> tuple[dict[str, Any], ...]:
    if not path.is_file() or path.is_symlink():
        raise HfGgufParityV5Error(f"{label} must be a regular file: {path}")
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise HfGgufParityV5Error(f"{label} is not valid UTF-8") from exc
    if not lines:
        raise HfGgufParityV5Error(f"{label} is empty")
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise HfGgufParityV5Error(
                f"{label} has a blank line at {line_number}"
            )
        try:
            value = json.loads(line, object_pairs_hook=_reject_duplicate_pairs)
        except json.JSONDecodeError as exc:
            raise HfGgufParityV5Error(
                f"{label} line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise HfGgufParityV5Error(
                f"{label} line {line_number} is not an object"
            )
        rows.append(value)
    return tuple(rows)


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in SHA256_CHARS for char in value)
    ):
        raise HfGgufParityV5Error(f"{label} is not a lowercase SHA-256")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HfGgufParityV5Error(f"{label} must be a non-empty string")
    return value


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HfGgufParityV5Error(f"{label} must be an object")
    return value


def _require_true(value: Any, label: str) -> None:
    if value is not True:
        raise HfGgufParityV5Error(f"{label} must be true")


def _require_false(value: Any, label: str) -> None:
    if value is not False:
        raise HfGgufParityV5Error(f"{label} must be false")


def _validate_inventory(value: Any, *, label: str) -> Mapping[str, Any]:
    inventory = _require_mapping(value, label)
    _require_nonempty_string(inventory.get("path"), f"{label}.path")
    files = inventory.get("files")
    if not isinstance(files, list) or not files:
        raise HfGgufParityV5Error(f"{label}.files must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for index, raw in enumerate(files):
        record = _require_mapping(raw, f"{label}.files[{index}]")
        path = _require_nonempty_string(
            record.get("path"), f"{label}.files[{index}].path"
        )
        if path in seen_paths:
            raise HfGgufParityV5Error(f"{label} contains duplicate path: {path}")
        seen_paths.add(path)
        size = record.get("bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise HfGgufParityV5Error(
                f"{label}.files[{index}].bytes must be a non-negative integer"
            )
        digest = _require_sha256(
            record.get("sha256"), f"{label}.files[{index}].sha256"
        )
        normalized.append({"path": path, "bytes": size, "sha256": digest})
    claimed = _require_sha256(
        inventory.get("content_sha256"), f"{label}.content_sha256"
    )
    actual = sha256_bytes(canonical_json(normalized).encode("utf-8"))
    if claimed != actual:
        raise HfGgufParityV5Error(f"{label} content SHA does not match file records")
    return inventory


def _validate_file_record(value: Any, *, label: str) -> Mapping[str, Any]:
    record = _require_mapping(value, label)
    _require_nonempty_string(record.get("path"), f"{label}.path")
    size = record.get("bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise HfGgufParityV5Error(f"{label}.bytes must be a positive integer")
    digest = _require_sha256(record.get("sha256"), f"{label}.sha256")
    if record.get("expected_sha256") != digest:
        raise HfGgufParityV5Error(f"{label}.expected_sha256 mismatch")
    _require_true(record.get("sha256_match"), f"{label}.sha256_match")
    _require_false(record.get("symlink"), f"{label}.symlink")
    _require_true(record.get("regular_file"), f"{label}.regular_file")
    return record


def _validate_tree_record(value: Any, *, label: str) -> Mapping[str, Any]:
    record = _require_mapping(value, label)
    _require_nonempty_string(record.get("path"), f"{label}.path")
    files = record.get("files")
    if not isinstance(files, list) or not files:
        raise HfGgufParityV5Error(f"{label}.files must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for index, raw in enumerate(files):
        item = _require_mapping(raw, f"{label}.files[{index}]")
        path = _require_nonempty_string(
            item.get("path"), f"{label}.files[{index}].path"
        )
        if path in seen_paths:
            raise HfGgufParityV5Error(f"{label} contains duplicate path: {path}")
        seen_paths.add(path)
        size = item.get("bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise HfGgufParityV5Error(
                f"{label}.files[{index}].bytes must be non-negative"
            )
        digest = _require_sha256(
            item.get("sha256"), f"{label}.files[{index}].sha256"
        )
        normalized.append({"path": path, "bytes": size, "sha256": digest})
    if record.get("file_count") != len(normalized):
        raise HfGgufParityV5Error(f"{label}.file_count mismatch")
    if record.get("bytes") != sum(item["bytes"] for item in normalized):
        raise HfGgufParityV5Error(f"{label}.bytes mismatch")
    claimed_tree = _require_sha256(
        record.get("tree_sha256"), f"{label}.tree_sha256"
    )
    actual_tree = sha256_bytes(canonical_json(normalized).encode("utf-8"))
    if claimed_tree != actual_tree:
        raise HfGgufParityV5Error(f"{label}.tree_sha256 mismatch")
    _require_false(record.get("symlinks_allowed"), f"{label}.symlinks_allowed")
    return record


def _validate_backend(
    summary: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    kind: str,
) -> Mapping[str, Any]:
    backend = _require_mapping(summary.get("backend"), f"{kind}.summary.backend")
    receipt_backend = _require_mapping(
        receipt.get("backend"), f"{kind}.receipt.backend"
    )
    if canonical_json(backend) != canonical_json(receipt_backend):
        raise HfGgufParityV5Error(f"{kind} summary/receipt backend mismatch")
    _require_true(backend.get("is_model"), f"{kind}.backend.is_model")
    _require_true(
        backend.get("free_generation_executed"),
        f"{kind}.backend.free_generation_executed",
    )
    _require_false(
        backend.get("assistant_target_visible_to_backend"),
        f"{kind}.backend.assistant_target_visible_to_backend",
    )
    generation_contract = _require_mapping(
        receipt.get("generation_contract"),
        f"{kind}.receipt.generation_contract",
    )
    _require_false(
        generation_contract.get("assistant_target_visible_to_backend"),
        f"{kind}.generation_contract.assistant_target_visible_to_backend",
    )
    _require_true(
        generation_contract.get("free_generation_executed"),
        f"{kind}.generation_contract.free_generation_executed",
    )
    if kind == "hf":
        if backend.get("mode") != "hf_model":
            raise HfGgufParityV5Error("HF backend must be hf_model")
        if backend.get("subject") != "adapter":
            raise HfGgufParityV5Error("HF run must evaluate a selected adapter")
        _validate_inventory(backend.get("base_model"), label="hf.base_model")
        _validate_inventory(backend.get("adapter"), label="hf.adapter")
        if backend.get("claim_boundary") != "LOCAL_HF_FREE_GENERATION":
            raise HfGgufParityV5Error("HF claim boundary is invalid")
        _require_false(backend.get("network_allowed"), "hf.backend.network_allowed")
        _require_true(
            backend.get("local_files_only"), "hf.backend.local_files_only"
        )
    else:
        if backend.get("mode") != "llama_cpp_gguf":
            raise HfGgufParityV5Error("GGUF backend must be llama_cpp_gguf")
        if backend.get("quantization") != Q4_QUANTIZATION:
            raise HfGgufParityV5Error("GGUF quantization must be Q4_K_M")
        if backend.get("device") != "cpu" or backend.get("gpu_layers") != 0:
            raise HfGgufParityV5Error(
                "GGUF parity input must be a CPU-only llama.cpp run"
            )
        if backend.get("server_bind_host") != "127.0.0.1":
            raise HfGgufParityV5Error("GGUF llama-server must bind loopback only")
        _require_true(
            backend.get("sequential_generation"),
            "gguf.backend.sequential_generation",
        )
        _require_false(
            backend.get("network_allowed"), "gguf.backend.network_allowed"
        )
        if backend.get("claim_boundary") != "LOCAL_LLAMA_CPP_GGUF_FREE_GENERATION":
            raise HfGgufParityV5Error("GGUF claim boundary is invalid")
        _validate_file_record(backend.get("gguf_model"), label="gguf.gguf_model")
        _validate_file_record(backend.get("llama_server"), label="gguf.llama_server")
        runtime = _validate_tree_record(
            backend.get("llama_cpp_runtime"), label="gguf.llama_cpp_runtime"
        )
        _require_true(
            runtime.get("unchanged_after_run"),
            "gguf.llama_cpp_runtime.unchanged_after_run",
        )
        server = _require_mapping(receipt.get("server"), "gguf.receipt.server")
        _require_true(server.get("port_released"), "gguf.server.port_released")
        _require_true(server.get("process_exited"), "gguf.server.process_exited")
        runtime_receipt = _require_mapping(
            receipt.get("runtime"), "gguf.receipt.runtime"
        )
        _require_true(runtime_receipt.get("loopback_only"), "gguf.runtime.loopback_only")
        _require_true(
            runtime_receipt.get("runtime_tree_unchanged"),
            "gguf.runtime.runtime_tree_unchanged",
        )
        _require_false(
            runtime_receipt.get("network_used_by_evaluator"),
            "gguf.runtime.network_used_by_evaluator",
        )
    return backend


def _validate_dataset_binding(
    summary: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    label: str,
) -> Mapping[str, Any]:
    dataset = _require_mapping(summary.get("dataset"), f"{label}.summary.dataset")
    receipt_dataset = _require_mapping(
        receipt.get("dataset"), f"{label}.receipt.dataset"
    )
    for field in ("manifest_sha256", "split_sha256"):
        _require_sha256(dataset.get(field), f"{label}.dataset.{field}")
        if dataset.get(field) != receipt_dataset.get(field):
            raise HfGgufParityV5Error(
                f"{label} summary/receipt dataset {field} mismatch"
            )
    for field in ("manifest_path", "split_path"):
        _require_nonempty_string(dataset.get(field), f"{label}.dataset.{field}")
        _require_nonempty_string(
            receipt_dataset.get(field), f"{label}.receipt.dataset.{field}"
        )
    return dataset


def _validate_receipt_self_hash(receipt: Mapping[str, Any], *, label: str) -> None:
    claimed = _require_sha256(
        receipt.get("receipt_payload_sha256"),
        f"{label}.receipt_payload_sha256",
    )
    payload = dict(receipt)
    del payload["receipt_payload_sha256"]
    actual = sha256_bytes(canonical_json(payload).encode("utf-8"))
    if actual != claimed:
        raise HfGgufParityV5Error(f"{label} receipt self hash mismatch")


def _validate_artifact_entry(
    entry: Any,
    path: Path,
    *,
    label: str,
    records: int | None = None,
) -> None:
    artifact = _require_mapping(entry, label)
    actual_bytes = path.stat().st_size
    actual_sha = sha256_file(path)
    if artifact.get("bytes") != actual_bytes:
        raise HfGgufParityV5Error(f"{label} byte count mismatch")
    if artifact.get("sha256") != actual_sha:
        raise HfGgufParityV5Error(f"{label} artifact SHA mismatch")
    if records is not None and artifact.get("records") != records:
        raise HfGgufParityV5Error(f"{label} record count mismatch")


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    summary: Mapping[str, Any],
    backend_mode: str,
    allowed_ablations: frozenset[str],
    label: str,
) -> tuple[dict[str, Any], ...]:
    split = _require_nonempty_string(summary.get("split"), f"{label}.summary.split")
    ablations = summary.get("ablations")
    if not isinstance(ablations, list) or not ablations:
        raise HfGgufParityV5Error(f"{label}.summary.ablations must be non-empty")
    if len(set(ablations)) != len(ablations):
        raise HfGgufParityV5Error(f"{label}.summary.ablations contains duplicates")
    if any(
        not isinstance(value, str) or value not in ALLOWED_ABLATIONS
        for value in ablations
    ):
        raise HfGgufParityV5Error(f"{label}.summary.ablations is invalid")
    disallowed = set(ablations) - allowed_ablations
    if disallowed:
        raise HfGgufParityV5Error(
            f"{label} contains disallowed ablations: {sorted(disallowed)}"
        )
    expected_rows = summary.get("examples")
    if (
        not isinstance(expected_rows, int)
        or isinstance(expected_rows, bool)
        or expected_rows <= 0
    ):
        raise HfGgufParityV5Error(f"{label}.summary.examples is invalid")
    if len(rows) != expected_rows * len(ablations):
        raise HfGgufParityV5Error(f"{label} row count does not match summary")

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    ablation_membership: dict[str, set[str]] = defaultdict(set)
    example_contract: dict[str, str] = {}
    for index, raw in enumerate(rows):
        row = dict(raw)
        prefix = f"{label}.rows[{index}]"
        if row.get("schema") != SAMPLE_SCHEMA:
            raise HfGgufParityV5Error(f"{prefix} schema is invalid")
        example_id = _require_nonempty_string(
            row.get("example_id"), f"{prefix}.example_id"
        )
        ablation = _require_nonempty_string(
            row.get("ablation"), f"{prefix}.ablation"
        )
        if ablation not in ablations:
            raise HfGgufParityV5Error(f"{prefix} ablation is not declared")
        key = (ablation, example_id)
        if key in seen:
            raise HfGgufParityV5Error(f"{label} duplicate membership: {key}")
        seen.add(key)
        ablation_membership[ablation].add(example_id)
        if row.get("split") != split:
            raise HfGgufParityV5Error(f"{prefix} split mismatch")
        for field in ("domain", "task", "gold_decision"):
            _require_nonempty_string(row.get(field), f"{prefix}.{field}")
        if row.get("gold_decision") not in {"ANSWER", "REFUSE"}:
            raise HfGgufParityV5Error(f"{prefix}.gold_decision is invalid")
        if row.get("backend_mode") != backend_mode:
            raise HfGgufParityV5Error(f"{prefix}.backend_mode is invalid")
        _require_false(
            row.get("assistant_target_visible_to_backend"),
            f"{prefix}.assistant_target_visible_to_backend",
        )
        prompt_sha = _require_sha256(
            row.get("prompt_sha256"), f"{prefix}.prompt_sha256"
        )
        generation = row.get("generation")
        if not isinstance(generation, str):
            raise HfGgufParityV5Error(f"{prefix}.generation must be a string")
        if row.get("generation_sha256") != sha256_bytes(
            generation.encode("utf-8")
        ):
            raise HfGgufParityV5Error(f"{prefix}.generation SHA mismatch")
        expected = _require_mapping(row.get("expected"), f"{prefix}.expected")
        target_errors = validate_student_answer(expected)
        if target_errors:
            raise HfGgufParityV5Error(
                f"{prefix}.expected violates the v5 answer schema: {target_errors}"
            )
        if expected.get("decision") != row.get("gold_decision"):
            raise HfGgufParityV5Error(f"{prefix} expected decision mismatch")
        if expected.get("task") != row.get("task"):
            raise HfGgufParityV5Error(f"{prefix} expected task mismatch")
        trace = row.get("trace")
        if trace is not None and not isinstance(trace, Mapping):
            raise HfGgufParityV5Error(f"{prefix}.trace must be an object or null")
        if isinstance(trace, Mapping) and "prompt_sha256" in trace:
            if trace.get("prompt_sha256") != prompt_sha:
                raise HfGgufParityV5Error(f"{prefix} trace prompt SHA mismatch")

        contract = canonical_json(
            {
                "split": row["split"],
                "domain": row["domain"],
                "task": row["task"],
                "gold_decision": row["gold_decision"],
                "expected": expected,
            }
        )
        previous = example_contract.setdefault(example_id, contract)
        if previous != contract:
            raise HfGgufParityV5Error(
                f"{label} example contract differs across ablations: {example_id}"
            )
        normalized.append(row)

    memberships = list(ablation_membership.values())
    if any(membership != memberships[0] for membership in memberships[1:]):
        raise HfGgufParityV5Error(
            f"{label} does not contain complete example membership per ablation"
        )
    if len(memberships[0]) != expected_rows:
        raise HfGgufParityV5Error(
            f"{label} unique example count does not match summary"
        )
    return tuple(normalized)


def _recompute(
    rows: Sequence[Mapping[str, Any]],
    *,
    summary: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    first_by_example: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        first_by_example.setdefault(str(row["example_id"]), row)
    samples = tuple(
        EvidenceSampleV5(
            example_id=example_id,
            split=str(row["split"]),
            domain=str(row["domain"]),
            task=str(row["task"]),
            decision=str(row["gold_decision"]),
            model_messages=(
                {"role": "system", "content": "hash-bound prompt"},
                {"role": "user", "content": "hash-bound prompt"},
            ),
            expected=dict(row["expected"]),
            raw={},
        )
        for example_id, row in sorted(first_by_example.items())
    )
    requests = tuple(
        GenerationRequestV5(
            example_id=str(row["example_id"]),
            split=str(row["split"]),
            domain=str(row["domain"]),
            task=str(row["task"]),
            ablation=str(row["ablation"]),
            messages=(
                {"role": "system", "content": "hash-bound prompt"},
                {"role": "user", "content": "hash-bound prompt"},
            ),
            prompt_sha256=str(row["prompt_sha256"]),
        )
        for row in rows
    )
    selection = DatasetSelectionV5(
        dataset_dir=Path("."),
        manifest_path=Path(str(summary["dataset"]["manifest_path"])),
        manifest_sha256=str(summary["dataset"]["manifest_sha256"]),
        manifest={},
        split=str(summary["split"]),
        split_path=Path(str(summary["dataset"]["split_path"])),
        split_sha256=str(summary["dataset"]["split_sha256"]),
        samples=samples,
        blind_test_authorization=summary.get("blind_test_authorization"),
    )
    generations = {
        (str(row["ablation"]), str(row["example_id"])): str(row["generation"])
        for row in rows
    }
    traces = {
        (str(row["ablation"]), str(row["example_id"])): dict(row["trace"])
        for row in rows
        if isinstance(row.get("trace"), Mapping)
    }
    rescored, summaries = score_generations(
        selection=selection,
        requests=requests,
        generations=generations,
        backend=dict(summary["backend"]),
        traces=traces,
    )
    return tuple(rescored), summaries


def _validate_self_report(
    original: Sequence[Mapping[str, Any]],
    recomputed: Sequence[Mapping[str, Any]],
    *,
    reported_summaries: Any,
    recomputed_summaries: Mapping[str, Any],
    label: str,
) -> None:
    original_by_key = {
        (str(row["ablation"]), str(row["example_id"])): row for row in original
    }
    recomputed_by_key = {
        (str(row["ablation"]), str(row["example_id"])): row for row in recomputed
    }
    for key in sorted(original_by_key):
        claimed = original_by_key[key]
        actual = recomputed_by_key[key]
        for field in SCORING_FIELDS:
            if canonical_json(claimed.get(field)) != canonical_json(actual.get(field)):
                raise HfGgufParityV5Error(
                    f"{label} self-reported scoring mismatch at {key}: {field}"
                )
    if canonical_json(reported_summaries) != canonical_json(recomputed_summaries):
        raise HfGgufParityV5Error(f"{label} self-reported summary metrics mismatch")


def _validate_gguf_request_receipt(
    receipt: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    contract = _require_mapping(
        receipt.get("generation_contract"), "gguf.receipt.generation_contract"
    )
    requests = contract.get("requests")
    if not isinstance(requests, list) or len(requests) != len(rows):
        raise HfGgufParityV5Error(
            "GGUF receipt request membership does not match per-sample rows"
        )
    request_map: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, raw in enumerate(requests):
        request = _require_mapping(raw, f"gguf.requests[{index}]")
        key = (
            _require_nonempty_string(
                request.get("ablation"), f"gguf.requests[{index}].ablation"
            ),
            _require_nonempty_string(
                request.get("example_id"), f"gguf.requests[{index}].example_id"
            ),
        )
        if key in request_map:
            raise HfGgufParityV5Error(f"GGUF receipt has duplicate request: {key}")
        request_map[key] = request
    for row in rows:
        key = (str(row["ablation"]), str(row["example_id"]))
        request = request_map.get(key)
        if request is None:
            raise HfGgufParityV5Error(
                f"GGUF receipt request membership is missing: {key}"
            )
        if request.get("prompt_sha256") != row.get("prompt_sha256"):
            raise HfGgufParityV5Error(f"GGUF receipt prompt mismatch at {key}")
        trace = row.get("trace")
        if isinstance(trace, Mapping):
            for field in ("request_sha256", "response_sha256", "finish_reason"):
                if field in trace and request.get(field) != trace.get(field):
                    raise HfGgufParityV5Error(
                        f"GGUF receipt trace mismatch at {key}: {field}"
                    )


def _validate_run(
    root: Path,
    *,
    kind: str,
    allowed_ablations: frozenset[str],
) -> ValidatedRun:
    resolved = root.expanduser().resolve()
    if not resolved.is_dir() or resolved.is_symlink():
        raise HfGgufParityV5Error(f"{kind} run must be a regular directory")
    paths = {
        "summary": resolved / "summary.v5.json",
        "run_receipt": resolved / "run_receipt.v5.json",
        "per_sample": resolved / "per_sample.v5.jsonl",
    }
    summary = _load_json(paths["summary"], label=f"{kind} summary")
    receipt = _load_json(paths["run_receipt"], label=f"{kind} run receipt")
    rows = _load_jsonl(paths["per_sample"], label=f"{kind} per-sample")
    expected_summary_schema = (
        HF_SUMMARY_SCHEMA if kind == "hf" else GGUF_SUMMARY_SCHEMA
    )
    expected_receipt_schema = (
        HF_RECEIPT_SCHEMA if kind == "hf" else GGUF_RECEIPT_SCHEMA
    )
    if summary.get("schema") != expected_summary_schema:
        raise HfGgufParityV5Error(f"{kind} summary schema is invalid")
    if receipt.get("schema") != expected_receipt_schema:
        raise HfGgufParityV5Error(f"{kind} run receipt schema is invalid")
    if receipt.get("status") != "COMPLETED":
        raise HfGgufParityV5Error(f"{kind} run receipt status is not COMPLETED")
    _validate_receipt_self_hash(receipt, label=kind)
    _validate_dataset_binding(summary, receipt, label=kind)
    backend = _validate_backend(summary, receipt, kind=kind)

    if receipt.get("split") != summary.get("split"):
        raise HfGgufParityV5Error(f"{kind} summary/receipt split mismatch")
    if canonical_json(summary.get("blind_test_authorization")) != canonical_json(
        receipt.get("blind_test_authorization")
    ):
        raise HfGgufParityV5Error(
            f"{kind} summary/receipt blind authorization mismatch"
        )
    _require_true(
        summary.get("model_quality_claim_allowed"),
        f"{kind}.summary.model_quality_claim_allowed",
    )
    if kind == "hf":
        _require_false(
            summary.get("non_model_test_only"),
            "hf.summary.non_model_test_only",
        )
    if receipt.get("claim_boundary") != backend.get("claim_boundary"):
        raise HfGgufParityV5Error(
            f"{kind} receipt/backend claim boundary mismatch"
        )
    summary_ablations = summary.get("ablations")
    generation_contract = _require_mapping(
        receipt.get("generation_contract"), f"{kind}.receipt.generation_contract"
    )
    if generation_contract.get("ablations") != summary_ablations:
        raise HfGgufParityV5Error(
            f"{kind} summary/receipt ablation membership mismatch"
        )
    _require_false(
        summary.get("assistant_target_visible_to_backend"),
        f"{kind}.summary.assistant_target_visible_to_backend",
    )

    artifacts = _require_mapping(receipt.get("artifacts"), f"{kind}.artifacts")
    _validate_artifact_entry(
        artifacts.get("per_sample.v5.jsonl"),
        paths["per_sample"],
        label=f"{kind}.artifacts.per_sample",
        records=len(rows),
    )
    _validate_artifact_entry(
        artifacts.get("summary.v5.json"),
        paths["summary"],
        label=f"{kind}.artifacts.summary",
    )
    per_sample_sha = sha256_file(paths["per_sample"])
    if summary.get("per_sample_sha256") != per_sample_sha:
        raise HfGgufParityV5Error(f"{kind} summary per-sample SHA mismatch")

    validated_rows = _validate_rows(
        rows,
        summary=summary,
        backend_mode=str(backend["mode"]),
        allowed_ablations=allowed_ablations,
        label=kind,
    )
    recomputed_rows, recomputed_summaries = _recompute(
        validated_rows,
        summary=summary,
    )
    _validate_self_report(
        validated_rows,
        recomputed_rows,
        reported_summaries=summary.get("summaries"),
        recomputed_summaries=recomputed_summaries,
        label=kind,
    )
    if kind == "gguf":
        _validate_gguf_request_receipt(receipt, validated_rows)

    hashes = {name: sha256_file(path) for name, path in paths.items()}
    return ValidatedRun(
        label=kind,
        root=resolved,
        summary=summary,
        receipt=receipt,
        rows=validated_rows,
        recomputed_rows=recomputed_rows,
        recomputed_summaries=recomputed_summaries,
        hashes=hashes,
    )


def _keyed(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (str(row["ablation"]), str(row["example_id"])): row for row in rows
    }


def _validate_comparability(hf: ValidatedRun, gguf: ValidatedRun) -> None:
    if hf.summary["split"] != gguf.summary["split"]:
        raise HfGgufParityV5Error("HF/GGUF split mismatch")
    for field in ("manifest_sha256", "split_sha256"):
        if hf.summary["dataset"][field] != gguf.summary["dataset"][field]:
            raise HfGgufParityV5Error(f"HF/GGUF dataset {field} mismatch")
    if hf.summary["ablations"] != gguf.summary["ablations"]:
        raise HfGgufParityV5Error("HF/GGUF ablation membership mismatch")
    if canonical_json(hf.summary.get("blind_test_authorization")) != canonical_json(
        gguf.summary.get("blind_test_authorization")
    ):
        raise HfGgufParityV5Error("HF/GGUF blind authorization mismatch")
    hf_rows = _keyed(hf.recomputed_rows)
    gguf_rows = _keyed(gguf.recomputed_rows)
    if set(hf_rows) != set(gguf_rows):
        missing = sorted(set(hf_rows) - set(gguf_rows))
        extra = sorted(set(gguf_rows) - set(hf_rows))
        raise HfGgufParityV5Error(
            f"HF/GGUF example+ablation membership mismatch; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    for key in sorted(hf_rows):
        left = hf_rows[key]
        right = gguf_rows[key]
        for field in ("split", "domain", "task", "gold_decision"):
            if left[field] != right[field]:
                raise HfGgufParityV5Error(
                    f"HF/GGUF membership metadata mismatch at {key}: {field}"
                )
        if canonical_json(left["expected"]) != canonical_json(right["expected"]):
            raise HfGgufParityV5Error(f"HF/GGUF expected mismatch at {key}")
        if left["prompt_sha256"] != right["prompt_sha256"]:
            raise HfGgufParityV5Error(f"HF/GGUF prompt SHA mismatch at {key}")


def _metric_rate(
    aggregate: Mapping[str, Any],
    metric: str,
) -> float | None:
    if metric == "refusal_f1":
        value = aggregate.get("refuse", {}).get("f1")
    else:
        value = aggregate.get("metrics", {}).get(metric, {}).get("rate")
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise HfGgufParityV5Error(f"recomputed metric is unavailable: {metric}")
    result = float(value)
    if not math.isfinite(result):
        raise HfGgufParityV5Error(f"recomputed metric is non-finite: {metric}")
    return result


def _metric_deltas(
    hf_aggregate: Mapping[str, Any],
    gguf_aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    names = (*RATE_METRICS, "refusal_f1")
    result: dict[str, Any] = {}
    for name in names:
        hf_rate = _metric_rate(hf_aggregate, name)
        gguf_rate = _metric_rate(gguf_aggregate, name)
        result[name] = {
            "hf_rate": hf_rate,
            "gguf_rate": gguf_rate,
            "delta_gguf_minus_hf": (
                gguf_rate - hf_rate
                if hf_rate is not None and gguf_rate is not None
                else None
            ),
        }
    return result


def _agreement(rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> dict[str, Any]:
    count = len(rows)
    decision_matches = sum(
        left.get("predicted_decision") == right.get("predicted_decision")
        for left, right in rows
    )
    valid_decisions = [
        (left, right)
        for left, right in rows
        if left.get("predicted_decision") is not None
        and right.get("predicted_decision") is not None
    ]
    valid_decision_matches = sum(
        left.get("predicted_decision") == right.get("predicted_decision")
        for left, right in valid_decisions
    )
    prediction_matches = sum(
        canonical_json(left.get("prediction")) == canonical_json(right.get("prediction"))
        for left, right in rows
    )
    parsed_pairs = [
        (left, right)
        for left, right in rows
        if left.get("prediction") is not None and right.get("prediction") is not None
    ]
    parsed_prediction_matches = sum(
        canonical_json(left.get("prediction")) == canonical_json(right.get("prediction"))
        for left, right in parsed_pairs
    )
    strict_score_matches = sum(
        bool(left["metrics"]["strict_exact"])
        == bool(right["metrics"]["strict_exact"])
        for left, right in rows
    )
    return {
        "examples": count,
        "decision_prediction_agreement": {
            "numerator": decision_matches,
            "denominator": count,
            "rate": decision_matches / count,
        },
        "decision_prediction_agreement_when_both_valid": {
            "numerator": valid_decision_matches,
            "denominator": len(valid_decisions),
            "rate": (
                valid_decision_matches / len(valid_decisions)
                if valid_decisions
                else None
            ),
        },
        "strict_prediction_object_agreement": {
            "definition": "parsed prediction JSON objects are exactly equal",
            "numerator": prediction_matches,
            "denominator": count,
            "rate": prediction_matches / count,
        },
        "strict_prediction_object_agreement_when_both_parsed": {
            "numerator": parsed_prediction_matches,
            "denominator": len(parsed_pairs),
            "rate": (
                parsed_prediction_matches / len(parsed_pairs)
                if parsed_pairs
                else None
            ),
        },
        "strict_score_outcome_agreement": {
            "numerator": strict_score_matches,
            "denominator": count,
            "rate": strict_score_matches / count,
        },
    }


def _percentile(values: Sequence[int], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _length_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lengths = [len(str(row["generation"])) for row in rows]
    return {
        "unit": "unicode_codepoints",
        "mean": statistics.fmean(lengths),
        "p50": _percentile(lengths, 0.50),
        "p95": _percentile(lengths, 0.95),
        "max": max(lengths),
    }


def _positive_truncation_value(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value > 0
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "none", "no"}
    return False


def _find_truncation_markers(
    value: Any,
    *,
    prefix: str = "trace",
) -> list[str]:
    issues: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            path = f"{prefix}.{key}"
            if str(key).lower() in TRUNCATION_KEYS and _positive_truncation_value(
                nested
            ):
                issues.append(path)
            issues.extend(_find_truncation_markers(nested, prefix=path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            issues.extend(
                _find_truncation_markers(nested, prefix=f"{prefix}[{index}]")
            )
    return issues


def _trace_safety(
    rows: Sequence[Mapping[str, Any]],
    *,
    backend: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    available = sum(isinstance(row.get("trace"), Mapping) for row in rows)
    if available not in {0, len(rows)}:
        issues.append(
            {
                "kind": "partial_trace_coverage",
                "available": available,
                "total": len(rows),
            }
        )
    decoding = _require_mapping(backend.get("decoding"), f"{label}.backend.decoding")
    for row in rows:
        trace = row.get("trace")
        if not isinstance(trace, Mapping):
            continue
        key = {
            "example_id": row["example_id"],
            "ablation": row["ablation"],
        }
        markers = _find_truncation_markers(trace)
        if markers:
            issues.append({**key, "kind": "explicit_input_truncation", "fields": markers})
        finish_reason = trace.get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason.lower() == "length":
            issues.append({**key, "kind": "finish_reason_length"})
        if label == "hf":
            input_tokens = trace.get("input_tokens")
            limit = decoding.get("max_input_tokens")
            if (
                isinstance(input_tokens, int)
                and not isinstance(input_tokens, bool)
                and isinstance(limit, int)
                and not isinstance(limit, bool)
                and input_tokens > limit
            ):
                issues.append({**key, "kind": "input_tokens_exceed_limit"})
            output_tokens = trace.get("output_tokens")
            output_limit = decoding.get("max_new_tokens")
            if (
                isinstance(output_tokens, int)
                and not isinstance(output_tokens, bool)
                and isinstance(output_limit, int)
                and not isinstance(output_limit, bool)
                and output_tokens >= output_limit
            ):
                issues.append({**key, "kind": "hf_output_token_limit_reached"})
        else:
            usage = trace.get("usage")
            context_size = decoding.get("context_size")
            max_tokens = decoding.get("max_tokens")
            prompt_tokens = (
                usage.get("prompt_tokens") if isinstance(usage, Mapping) else None
            )
            if (
                isinstance(prompt_tokens, int)
                and not isinstance(prompt_tokens, bool)
                and isinstance(context_size, int)
                and not isinstance(context_size, bool)
                and prompt_tokens >= context_size
            ):
                issues.append({**key, "kind": "prompt_tokens_reach_context_limit"})
            if (
                isinstance(prompt_tokens, int)
                and not isinstance(prompt_tokens, bool)
                and isinstance(context_size, int)
                and not isinstance(context_size, bool)
                and isinstance(max_tokens, int)
                and not isinstance(max_tokens, bool)
                and prompt_tokens + max_tokens > context_size
            ):
                issues.append({**key, "kind": "insufficient_context_headroom"})
    return {
        "trace_rows": available,
        "total_rows": len(rows),
        "enforced": available > 0,
        "issues": issues,
        "passed": not issues,
        "boundary": (
            "Checks explicit trace truncation markers and output-length limits. "
            "It does not prove tokenizer-level prompt identity."
        ),
    }


def _degradation_diagnostics(
    hf_rows: Sequence[Mapping[str, Any]],
    gguf_rows: Sequence[Mapping[str, Any]],
    *,
    hf_backend: Mapping[str, Any],
    gguf_backend: Mapping[str, Any],
) -> dict[str, Any]:
    count = len(hf_rows)
    hf_parse_failures = sum(not row["metrics"]["json_valid"] for row in hf_rows)
    gguf_parse_failures = sum(not row["metrics"]["json_valid"] for row in gguf_rows)
    hf_schema_failures = sum(not row["metrics"]["schema_valid"] for row in hf_rows)
    gguf_schema_failures = sum(not row["metrics"]["schema_valid"] for row in gguf_rows)
    return {
        "generation_length": {
            "hf": _length_stats(hf_rows),
            "gguf": _length_stats(gguf_rows),
            "mean_delta_gguf_minus_hf": (
                _length_stats(gguf_rows)["mean"] - _length_stats(hf_rows)["mean"]
            ),
        },
        "parse_failure": {
            "hf_count": hf_parse_failures,
            "gguf_count": gguf_parse_failures,
            "hf_rate": hf_parse_failures / count,
            "gguf_rate": gguf_parse_failures / count,
            "delta_gguf_minus_hf": (
                gguf_parse_failures - hf_parse_failures
            )
            / count,
        },
        "schema_failure": {
            "hf_count": hf_schema_failures,
            "gguf_count": gguf_schema_failures,
            "hf_rate": hf_schema_failures / count,
            "gguf_rate": gguf_schema_failures / count,
            "delta_gguf_minus_hf": (
                gguf_schema_failures - hf_schema_failures
            )
            / count,
        },
        "trace_safety": {
            "hf": _trace_safety(hf_rows, backend=hf_backend, label="hf"),
            "gguf": _trace_safety(gguf_rows, backend=gguf_backend, label="gguf"),
        },
    }


def _non_degradation_gates(
    deltas: Mapping[str, Mapping[str, Any]],
    trace_safety: Mapping[str, Any],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for metric, (direction, tolerance) in NON_DEGRADATION_LIMITS.items():
        values = deltas[metric]
        hf_rate = float(values["hf_rate"])
        gguf_rate = float(values["gguf_rate"])
        degradation = (
            hf_rate - gguf_rate if direction == "drop" else gguf_rate - hf_rate
        )
        checks[metric] = {
            "direction": direction,
            "tolerance": tolerance,
            "observed_degradation": degradation,
            "passed": degradation <= tolerance + 1e-12,
        }
    trace_passed = bool(trace_safety["hf"]["passed"]) and bool(
        trace_safety["gguf"]["passed"]
    )
    checks["trace_no_truncation_or_length_finish"] = {
        "direction": "absolute_safety",
        "tolerance": 0,
        "observed_degradation": (
            len(trace_safety["hf"]["issues"])
            + len(trace_safety["gguf"]["issues"])
        ),
        "passed": trace_passed,
        "enforced_when_trace_available": True,
    }
    return {
        "checks": checks,
        "all_passed": all(bool(check["passed"]) for check in checks.values()),
    }


def _code_hashes(runner_path: Path | None) -> dict[str, Any]:
    paths = {"comparator": Path(__file__).resolve()}
    if runner_path is not None:
        paths["runner"] = runner_path.resolve()
    result: dict[str, Any] = {}
    for label, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise HfGgufParityV5Error(f"{label} source must be a regular file")
        result[label] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    return result


def _publish_output(
    output_dir: Path,
    *,
    report: Mapping[str, Any],
    receipt_body: Mapping[str, Any],
) -> dict[str, Any]:
    output = output_dir.expanduser().resolve()
    if output.exists():
        raise HfGgufParityV5Error(
            "parity output already exists; use a new immutable directory"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.stage-{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        report_payload = _json_bytes(report)
        report_sha = sha256_bytes(report_payload)
        (stage / REPORT_NAME).write_bytes(report_payload)
        receipt = dict(receipt_body)
        receipt["artifacts"] = {
            REPORT_NAME: {
                "bytes": len(report_payload),
                "sha256": report_sha,
            }
        }
        receipt["receipt_payload_sha256"] = sha256_bytes(
            canonical_json(receipt).encode("utf-8")
        )
        receipt_payload = _json_bytes(receipt)
        receipt_sha = sha256_bytes(receipt_payload)
        (stage / RECEIPT_NAME).write_bytes(receipt_payload)
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        "output_dir": str(output),
        "paths": {
            "report": str(output / REPORT_NAME),
            "receipt": str(output / RECEIPT_NAME),
        },
        "hashes": {
            "report": report_sha,
            "receipt": receipt_sha,
        },
        "report": dict(report),
        "receipt": receipt,
    }


def compare_hf_gguf_parity(
    hf_dir: Path,
    gguf_dir: Path,
    output_dir: Path,
    *,
    allowed_ablations: Sequence[str] = ("none",),
    runner_path: Path | None = None,
) -> dict[str, Any]:
    """Validate and compare one HF selected-adapter run with one Q4_K_M run."""

    if not allowed_ablations:
        raise HfGgufParityV5Error("allowed_ablations must not be empty")
    allowed = frozenset(allowed_ablations)
    if len(allowed) != len(tuple(allowed_ablations)):
        raise HfGgufParityV5Error("allowed_ablations contains duplicates")
    if not allowed.issubset(ALLOWED_ABLATIONS):
        raise HfGgufParityV5Error("allowed_ablations contains an invalid value")
    hf = _validate_run(
        hf_dir,
        kind="hf",
        allowed_ablations=allowed,
    )
    gguf = _validate_run(
        gguf_dir,
        kind="gguf",
        allowed_ablations=allowed,
    )
    _validate_comparability(hf, gguf)

    hf_rows = _keyed(hf.recomputed_rows)
    gguf_rows = _keyed(gguf.recomputed_rows)
    comparisons: dict[str, Any] = {}
    for ablation in hf.summary["ablations"]:
        pairs = [
            (hf_rows[key], gguf_rows[key])
            for key in sorted(hf_rows)
            if key[0] == ablation
        ]
        hf_aggregate = hf.recomputed_summaries[ablation]
        gguf_aggregate = gguf.recomputed_summaries[ablation]
        task_deltas: dict[str, Any] = {}
        task_agreements: dict[str, Any] = {}
        tasks = sorted(
            set(hf_aggregate["stratified"]["task"])
            | set(gguf_aggregate["stratified"]["task"])
        )
        for task in tasks:
            if task not in hf_aggregate["stratified"]["task"]:
                raise HfGgufParityV5Error(f"HF task membership missing: {task}")
            if task not in gguf_aggregate["stratified"]["task"]:
                raise HfGgufParityV5Error(f"GGUF task membership missing: {task}")
            task_deltas[task] = _metric_deltas(
                hf_aggregate["stratified"]["task"][task],
                gguf_aggregate["stratified"]["task"][task],
            )
            task_agreements[task] = _agreement(
                [
                    pair
                    for pair in pairs
                    if pair[0]["task"] == task
                ]
            )
        comparisons[ablation] = {
            "examples": len(pairs),
            "overall_metric_deltas": _metric_deltas(
                hf_aggregate,
                gguf_aggregate,
            ),
            "task_metric_deltas": task_deltas,
            "agreement": {
                "overall": _agreement(pairs),
                "by_task": task_agreements,
            },
        }

    none_pairs = [
        (hf_rows[key], gguf_rows[key])
        for key in sorted(hf_rows)
        if key[0] == "none"
    ]
    if not none_pairs:
        raise HfGgufParityV5Error(
            "the non-degradation gate requires ablation=none"
        )
    diagnostics = _degradation_diagnostics(
        [pair[0] for pair in none_pairs],
        [pair[1] for pair in none_pairs],
        hf_backend=hf.summary["backend"],
        gguf_backend=gguf.summary["backend"],
    )
    gates = _non_degradation_gates(
        comparisons["none"]["overall_metric_deltas"],
        diagnostics["trace_safety"],
    )
    status = "HF_GGUF_TASK_PARITY_PASS" if gates["all_passed"] else (
        "HF_GGUF_TASK_PARITY_FAIL"
    )
    code = _code_hashes(runner_path)
    report = {
        "schema": REPORT_SCHEMA,
        "created_at_utc": _utc_now(),
        "status": status,
        "claim_boundary": {
            "proved_if_passed": (
                "Task-level Q4_K_M non-degradation for identical "
                "example, ablation, prompt-hash and expected membership."
            ),
            "token_level_prompt_identity_claimed": False,
            "reason": (
                "HF and llama.cpp may render/tokenize the same messages "
                "differently; this comparator does not inspect token streams."
            ),
            "production_or_x5_runtime_claimed": False,
        },
        "configuration": {
            "allowed_ablations": sorted(allowed),
            "gate_ablation": "none",
            "non_degradation_limits": {
                metric: {"direction": direction, "tolerance": tolerance}
                for metric, (direction, tolerance) in NON_DEGRADATION_LIMITS.items()
            },
        },
        "inputs": {
            "hf": {
                "directory": str(hf.root),
                "hashes": hf.hashes,
                "backend": {
                    "mode": hf.summary["backend"]["mode"],
                    "subject": hf.summary["backend"]["subject"],
                    "base_model_content_sha256": hf.summary["backend"]["base_model"][
                        "content_sha256"
                    ],
                    "adapter_content_sha256": hf.summary["backend"]["adapter"][
                        "content_sha256"
                    ],
                },
            },
            "gguf": {
                "directory": str(gguf.root),
                "hashes": gguf.hashes,
                "backend": {
                    "mode": gguf.summary["backend"]["mode"],
                    "quantization": gguf.summary["backend"]["quantization"],
                    "gguf_sha256": gguf.summary["backend"]["gguf_model"]["sha256"],
                    "llama_server_sha256": gguf.summary["backend"]["llama_server"][
                        "sha256"
                    ],
                },
            },
            "dataset": {
                "split": hf.summary["split"],
                "manifest_sha256": hf.summary["dataset"]["manifest_sha256"],
                "split_sha256": hf.summary["dataset"]["split_sha256"],
                "examples": hf.summary["examples"],
                "ablations": hf.summary["ablations"],
            },
        },
        "integrity": {
            "schemas_status_and_artifact_hashes_verified": True,
            "receipt_self_hashes_verified": True,
            "complete_membership_verified": True,
            "expected_and_prompt_sha256_verified": True,
            "self_reported_scores_recomputed_and_verified": True,
            "shared_scorer": {
                "path": str(Path(score_generations.__code__.co_filename).resolve()),
                "sha256": sha256_file(
                    Path(score_generations.__code__.co_filename).resolve()
                ),
            },
        },
        "comparisons": comparisons,
        "degradation_diagnostics_ablation_none": diagnostics,
        "non_degradation_gate": gates,
        "code": code,
    }
    receipt_body = {
        "schema": RECEIPT_SCHEMA,
        "created_at_utc": _utc_now(),
        "status": status,
        "input_artifacts": {
            "hf": hf.hashes,
            "gguf": gguf.hashes,
        },
        "dataset": report["inputs"]["dataset"],
        "code": code,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "network_used": False,
        },
        "atomic_directory_publish": True,
        "claim_boundary": report["claim_boundary"],
    }
    return _publish_output(
        output_dir,
        report=report,
        receipt_body=receipt_body,
    )


__all__ = [
    "HfGgufParityV5Error",
    "NON_DEGRADATION_LIMITS",
    "RECEIPT_NAME",
    "RECEIPT_SCHEMA",
    "REPORT_NAME",
    "REPORT_SCHEMA",
    "compare_hf_gguf_parity",
]
