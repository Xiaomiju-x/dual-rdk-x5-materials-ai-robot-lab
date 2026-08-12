#!/usr/bin/env python3
"""Finalize existing F-LLM-02 GGUF and validation evidence without rerunning inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
MODEL_ID = "F-LLM-02"
ARTIFACT_ROOT = ROOT / "icmat_foundry/finals_50model/artifacts/llm/F-LLM-02"
EVIDENCE_ROOT = ROOT / "icmat_foundry/finals_50model/evidence/llm/F-LLM-02"
VALIDATION_FILE = (
    ROOT
    / "evaluation/icmat_foundry/llm"
    / "icmat_qwen05b_evidence_pointer_sft_v8_pretrain_20260731_r4"
    / "validation.jsonl"
)
POINTER = re.compile(
    r'\{\s*"task"\s*:\s*"([^"]+)"\s*,\s*"decision"\s*:\s*"(ANSWER|REFUSE)"'
    r'\s*,\s*"span_id"\s*:\s*(null|"[^"]+")\s*\}'
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(encoded, encoding="utf-8", newline="\n")
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii", newline="\n"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def parse_pointer(text: str) -> dict[str, Any] | None:
    matches = list(POINTER.finditer(text))
    if not matches:
        return None
    task, decision, raw_span = matches[-1].groups()
    return {
        "task": task,
        "decision": decision,
        "span_id": None if raw_span == "null" else json.loads(raw_span),
    }


def validation_postaudit() -> dict[str, Any]:
    prediction_path = EVIDENCE_ROOT / "validation_predictions.jsonl"
    original_receipt_path = EVIDENCE_ROOT / "validation_receipt.v1.json"
    expected_rows = read_jsonl(VALIDATION_FILE)
    predictions = read_jsonl(prediction_path)
    if len(expected_rows) != 150 or len(predictions) != 150:
        raise RuntimeError("the frozen 150-row validation contract is incomplete")
    expected_tasks = {
        json.loads(row["messages"][-1]["content"])["task"] for row in expected_rows
    }
    schema_valid = 0
    exact = 0
    task_counts: Counter[str] = Counter()
    for row in predictions:
        prediction = row.get("prediction")
        expected = row.get("expected")
        if not isinstance(prediction, dict) or not isinstance(expected, dict):
            continue
        task_counts[str(expected.get("task"))] += 1
        keys_valid = list(prediction) == ["decision", "span_id", "task"] or set(prediction) == {
            "task",
            "decision",
            "span_id",
        }
        decision_valid = prediction.get("decision") in {"ANSWER", "REFUSE"}
        span_valid = (
            prediction.get("span_id") is None
            if prediction.get("decision") == "REFUSE"
            else isinstance(prediction.get("span_id"), str)
        )
        if keys_valid and prediction.get("task") in expected_tasks and decision_valid and span_valid:
            schema_valid += 1
        exact += int(prediction == expected)
    receipt = {
        "schema": "icmat_evidenceqa_17b_validation_contract_postaudit.v1",
        "created_at": utc_now(),
        "model_id": MODEL_ID,
        "status": "PASS",
        "method": "RECOUNT_EXISTING_FIXED_PREDICTIONS_NO_REGENERATION",
        "source": {
            "validation_path": str(VALIDATION_FILE.relative_to(ROOT)),
            "validation_sha256": sha256_file(VALIDATION_FILE),
            "prediction_path": str(prediction_path.relative_to(ROOT)),
            "prediction_sha256": sha256_file(prediction_path),
            "original_receipt_path": str(original_receipt_path.relative_to(ROOT)),
            "original_receipt_sha256": sha256_file(original_receipt_path),
        },
        "contract_correction": {
            "original_allowed_tasks": ["claim_verification", "evidence_selection"],
            "fixed_validation_tasks": sorted(expected_tasks),
            "reason": "The original evaluator omitted claim_extraction even though it is a fixed task in the validation data.",
        },
        "metrics": {
            "rows": len(predictions),
            "task_counts": dict(sorted(task_counts.items())),
            "schema_valid": schema_valid,
            "schema_valid_rate": schema_valid / len(predictions),
            "exact": exact,
            "exact_rate": exact / len(predictions),
        },
        "claims": {
            "predictions_changed": False,
            "model_rerun": False,
            "x5_accessed": False,
            "production_integrated": False,
        },
    }
    if schema_valid != 150 or exact != 135:
        raise RuntimeError("post-audit metrics do not match the frozen prediction set")
    return receipt


def gguf_receipt() -> dict[str, Any]:
    f16_path = ARTIFACT_ROOT / "gguf/ICMat-Qwen3-1.7B-EvidenceQA-F16.gguf"
    q4_path = ARTIFACT_ROOT / "gguf/ICMat-Qwen3-1.7B-EvidenceQA-Q4_K_M.gguf"
    merge_receipt_path = EVIDENCE_ROOT / "merge_receipt.v1.json"
    smoke_logs = sorted(EVIDENCE_ROOT.glob("cpu_smoke_*.stdout.log"), key=lambda path: path.stat().st_mtime)
    if not f16_path.is_file() or not q4_path.is_file() or not smoke_logs:
        raise FileNotFoundError("existing F16/Q4 GGUF or CPU smoke log is missing")
    stdout_path = smoke_logs[-1]
    stderr_path = stdout_path.with_name(stdout_path.name.replace(".stdout.log", ".stderr.log"))
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    parsed = parse_pointer(stdout)
    if parsed is None:
        raise RuntimeError("CPU smoke did not emit a compact evidence pointer")
    performance = re.search(r"Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s", stdout)
    return {
        "schema": "icmat_evidenceqa_17b_gguf_receipt.v1",
        "created_at": utc_now(),
        "model_id": MODEL_ID,
        "status": "PC_RUNNABLE_X5_PENDING",
        "recovery_context": {
            "original_all_stage_interrupted_after_q4": True,
            "training_rerun": False,
            "evaluation_rerun": False,
            "merge_rerun": False,
            "quantization_rerun": False,
            "cpu_smoke_rerun_only": True,
        },
        "source": {
            "merge_receipt_path": str(merge_receipt_path.relative_to(ROOT)),
            "merge_receipt_sha256": sha256_file(merge_receipt_path),
        },
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
                "quantization": "Q4_K_M",
            },
        },
        "cpu_smoke": {
            "runtime": "llama.cpp b10158 Windows CPU",
            "actual_cpu_only": True,
            "n_gpu_layers": 0,
            "threads": 8,
            "parsed_pointer": parsed,
            "stdout_path": str(stdout_path.relative_to(ROOT)),
            "stdout_sha256": sha256_file(stdout_path),
            "stderr_path": str(stderr_path.relative_to(ROOT)),
            "stderr_sha256": sha256_file(stderr_path),
            "prompt_tokens_per_second": float(performance.group(1)) if performance else None,
            "generation_tokens_per_second": float(performance.group(2)) if performance else None,
        },
        "claims": {
            "x5_accessed": False,
            "x5_verified": False,
            "production_integrated": False,
            "status_ceiling": "PC_RUNNABLE_X5_PENDING",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("refusing to write receipts without --execute")
    gguf = gguf_receipt()
    audit = validation_postaudit()
    write_json(EVIDENCE_ROOT / "gguf_receipt.v1.json", gguf)
    write_json(EVIDENCE_ROOT / "validation_contract_postaudit.v1.json", audit)
    print(json.dumps({"gguf": gguf["status"], "validation": audit["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
