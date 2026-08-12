"""Validate the bounded llama.cpp smoke and write the F-LLM-01 receipt."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CANDIDATE = ROOT / "icmat_foundry/finals_50model"
ARTIFACT = CANDIDATE / "artifacts/llm/F-LLM-01"
EVIDENCE = CANDIDATE / "evidence/llm"
PROMPT = CANDIDATE / "contracts/F-LLM-01-smoke-prompt.txt"
STDOUT = EVIDENCE / "F-LLM-01.llama_stdout.txt"
STDERR = EVIDENCE / "F-LLM-01.llama_stderr.txt"
F16 = ARTIFACT / "ICMat-Qwen3-0.6B-XRDPL-F16.gguf"
Q4 = ARTIFACT / "ICMat-Qwen3-0.6B-XRDPL-Q4_K_M.gguf"
SOURCE = ROOT / "icmat_foundry/handoffs/20260801/gpuB_success/ICMat-Qwen3-0.6B-XRDPL-HF/model.safetensors"
OUTPUT = EVIDENCE / "F-LLM-01.gguf_receipt.v1.json"
OVERLAY = CANDIDATE / "contracts/model_state_overlay.v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_float(pattern: str, text: str) -> float:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"missing runtime metric: {pattern}")
    return float(match.group(1))


def main() -> None:
    for path in (PROMPT, STDOUT, STDERR, F16, Q4, SOURCE):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    stdout = STDOUT.read_text(encoding="utf-8-sig", errors="replace")
    stderr = STDERR.read_text(encoding="utf-8-sig", errors="replace")
    objects = re.findall(r"\{[^{}]+\}", stdout)
    if len(objects) != 1:
        raise ValueError(f"expected exactly one JSON object, found {len(objects)}")
    generated = json.loads(objects[0])
    expected_keys = {
        "xrd_main_peak_2theta_deg",
        "pl_peak_nm",
        "phase",
        "evidence_status",
    }
    if set(generated) != expected_keys:
        raise ValueError(f"schema mismatch: {sorted(generated)}")
    if float(generated["xrd_main_peak_2theta_deg"]) != 28.4:
        raise ValueError("XRD value changed")
    if float(generated["pl_peak_nm"]) != 980.0:
        raise ValueError("PL value changed")
    if generated["phase"] != "UNKNOWN":
        raise ValueError("unsupported phase was invented")
    if generated["evidence_status"] != "MEASURED_VALUES_ONLY":
        raise ValueError("evidence boundary changed")

    receipt = {
        "schema": "x5_icmat_foundry.gguf_runtime_receipt.v1",
        "status": "PC_RUNNABLE_Q4_GGUF_X5_PENDING",
        "inventory_id": "F-LLM-01",
        "model_id": "ICMat-Qwen3-0.6B-XRDPL-CPU",
        "source_hf_weight_sha256": sha256(SOURCE),
        "f16_gguf_path": str(F16.relative_to(ROOT)).replace("\\", "/"),
        "f16_gguf_bytes": F16.stat().st_size,
        "f16_gguf_sha256": sha256(F16),
        "q4_gguf_path": str(Q4.relative_to(ROOT)).replace("\\", "/"),
        "q4_gguf_bytes": Q4.stat().st_size,
        "q4_gguf_sha256": sha256(Q4),
        "quantization": "Q4_K_M",
        "runtime": "llama.cpp b10158 Windows CPU",
        "gpu_layers": 0,
        "threads": 4,
        "load_time_ms": extract_float(r"load time =\s+([0-9.]+) ms", stderr),
        "prompt_tokens_per_second": extract_float(
            r"prompt eval time =[^\n]+?([0-9.]+) tokens per second", stderr
        ),
        "generation_tokens_per_second": extract_float(
            r"(?m)^.*common_perf_print:\s+eval time =[^\n]+?([0-9.]+) tokens per second",
            stderr,
        ),
        "prompt_path": str(PROMPT.relative_to(ROOT)).replace("\\", "/"),
        "prompt_sha256": sha256(PROMPT),
        "generated_object": generated,
        "semantic_gate": "PASS_EXACT_VALUES_UNKNOWN_NO_EXTRA_KEYS",
        "stdout_sha256": sha256(STDOUT),
        "stderr_sha256": sha256(STDERR),
        "authority": 0,
        "network_used": False,
        "x5_contacted": False,
        "production_integrated": False,
        "claim_boundary": "Real local llama.cpp CPU run; X5 CPU execution and production integration remain pending.",
    }
    OUTPUT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    overlay = json.loads(OVERLAY.read_text(encoding="utf-8-sig"))
    for item in overlay["models"]:
        if item["inventory_id"] == "F-LLM-01":
            item.update(
                {
                    "state": receipt["status"],
                    "model_sha256": receipt["q4_gguf_sha256"],
                    "gguf_sha256": receipt["q4_gguf_sha256"],
                    "receipt_path": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
                    "receipt_sha256": sha256(OUTPUT),
                }
            )
            break
    else:
        raise ValueError("F-LLM-01 missing from overlay")
    overlay["status"] = "FAST_TRACK_QWEN06_GGUF_ACCEPTED"
    OVERLAY.write_text(json.dumps(overlay, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
