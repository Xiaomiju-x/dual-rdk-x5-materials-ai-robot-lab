"""Run one bounded local smoke test for the merged 0.6B XRD/PL expert."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "icmat_foundry/handoffs/20260801/gpuB_success/ICMat-Qwen3-0.6B-XRDPL-HF"
OUTPUT = ROOT / "icmat_foundry/finals_50model/evidence/phase1/qwen06_hf_smoke.v1.json"
PROMPT = (
    "Return one compact JSON object only. Evidence: XRD main peaks are 28.4, "
    "47.3 and 56.1 degrees; no reference phase has been supplied. PL peak is "
    "1020 nm with FWHM 86 nm. Separate measured fields from interpretation and "
    "use UNKNOWN for unsupported phase identity."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the fast-track smoke test")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        local_files_only=True,
        fix_mistral_regex=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        local_files_only=True,
        dtype=torch.float16,
        device_map="cuda:0",
    )
    messages = [
        {
            "role": "system",
            "content": "You are an evidence-bound XRD and photoluminescence assistant.",
        },
        {"role": "user", "content": PROMPT},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(rendered, return_tensors="pt").to("cuda:0")
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=96,
            do_sample=False,
            use_cache=True,
        )
    elapsed = time.perf_counter() - started
    continuation = generated[0, encoded.input_ids.shape[1] :]
    text = tokenizer.decode(continuation, skip_special_tokens=True).strip()
    if not text:
        raise RuntimeError("empty generation")
    payload = {
        "schema": "x5_icmat_foundry.qwen06_hf_smoke.v1",
        "status": "PC_RUNNABLE_HF_GGUF_PENDING",
        "inventory_id": "F-LLM-01",
        "model_id": "ICMat-Qwen3-0.6B-XRDPL-CPU",
        "model_path": str(MODEL.relative_to(ROOT)).replace("\\", "/"),
        "model_safetensors_sha256": sha256(MODEL / "model.safetensors"),
        "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
        "output": text,
        "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "input_tokens": int(encoded.input_ids.shape[1]),
        "output_tokens": int(continuation.shape[0]),
        "elapsed_seconds": elapsed,
        "device": torch.cuda.get_device_name(0),
        "dtype": "float16",
        "authority": 0,
        "network_used": False,
        "x5_contacted": False,
        "production_integrated": False,
        "claim_boundary": "One local bounded HF smoke; GGUF and X5 CPU runtime remain pending.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "input_tokens": payload["input_tokens"],
                "output_tokens": payload["output_tokens"],
                "elapsed_seconds": round(elapsed, 3),
                "output_preview": text[:240],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
