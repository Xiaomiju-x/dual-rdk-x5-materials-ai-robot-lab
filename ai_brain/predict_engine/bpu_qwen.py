"""bpu_qwen.py — Qwen2-0.5B 推理 on X5, 2-bin BPU chain + CPU 前/后.

架构:
    CPU pre:   input_ids [1,64,int32] → embed_tokens lookup → hidden [1,64,896,1,fp32]
    BPU p1:    12 layers transformer → intermediate_hidden   ~215 ms
    BPU p2:    12 layers transformer → post_hidden           ~215 ms
    CPU post:  post_hidden → RMSNorm → × embed.T → logits → verdict

总延迟: ~615 ms/forward (含 CPU pre/post + 2×BPU ~435 ms)

依赖 X5 上: tokenizers, numpy, hobot_dnn (无 transformers/torch)
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Any
import numpy as np

_MODELS_DIR = Path("/home/rdk/models")
PART1_BIN = _MODELS_DIR / "qwen2_part1.bin"
PART2_BIN = _MODELS_DIR / "qwen2_part2.bin"
EMBED_NPY = _MODELS_DIR / "embed_tokens_fp16.npy"
NORM_NPY = _MODELS_DIR / "norm_final_fp32.npy"
TOKENIZER_JSON = _MODELS_DIR / "tokenizer.json"
SEQ_LEN = 64
HIDDEN_SIZE = 896

# 单 token 中文 verdict (Qwen2 tokenizer 里都是 1 token)
VERDICT_TOKENS = {
    "GO":     101914,   # 推荐
    "REVISE": 105023,   # 改进
    "DROP":   102256,   # 放弃
}
# 短引导语, 让 verdict token 自然落在下一个位置
VERDICT_PROMPT_SUFFIX = "\n判:"

_CACHE: dict[str, Any] = {"part1": None, "part2": None, "tok": None,
                          "embed": None, "norm_w": None}


def _ensure_loaded():
    if _CACHE["part1"] is not None:
        return
    from hobot_dnn import pyeasy_dnn as dnn
    print(f"[bpu_qwen] loading part1 + part2 bins...")
    _CACHE["part1"] = dnn.load(str(PART1_BIN))[0]
    _CACHE["part2"] = dnn.load(str(PART2_BIN))[0]

    from tokenizers import Tokenizer
    _CACHE["tok"] = Tokenizer.from_file(str(TOKENIZER_JSON))

    # embed fp32 (544 MB): ARM numpy fp16 matmul 没 SIMD 慢 5×, 不值
    _CACHE["embed"] = np.load(str(EMBED_NPY)).astype(np.float32)
    _CACHE["norm_w"] = np.load(str(NORM_NPY))
    print(f"[bpu_qwen] ready. embed {_CACHE['embed'].shape} ({_CACHE['embed'].dtype})")


def _cpu_embed(input_ids: np.ndarray) -> np.ndarray:
    """[1,L,int32] → [1,L,896,1,fp32]."""
    out = _CACHE["embed"][input_ids[0]]  # [L, 896] fp32
    return out.reshape(1, SEQ_LEN, HIDDEN_SIZE, 1).astype(np.float32)


def _cpu_rmsnorm(h: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    w = _CACHE["norm_w"]
    var = (h * h).mean()
    return (h / np.sqrt(var + eps)) * w


def _cpu_lm_head(post_hidden: np.ndarray, last_valid: int) -> np.ndarray:
    """fp32 matmul (ARM 有 SIMD)."""
    embed = _CACHE["embed"]   # fp32
    h = post_hidden.reshape(1, SEQ_LEN, HIDDEN_SIZE)
    last_h = h[0, last_valid, :]
    last_h_normed = _cpu_rmsnorm(last_h)
    return last_h_normed @ embed.T


def bpu_qwen_verdict(prompt: str, auto_append_suffix: bool = True) -> dict:
    """BPU Qwen2 verdict (prefill + 3-way logit probe).

    流程:
    1. prompt + "\\n判:" 做 1 次 24-layer BPU forward (~435 ms)
    2. last_valid_pos 处直接比较 推荐 / 改进 / 放弃 logit
    3. softmax → 概率, 取最高
    """
    _ensure_loaded()
    t0 = time.perf_counter()

    suffix = VERDICT_PROMPT_SUFFIX if auto_append_suffix else ""
    suffix_ids = _CACHE["tok"].encode(suffix).ids if suffix else []
    prompt_ids = _CACHE["tok"].encode(prompt).ids
    room = SEQ_LEN - len(suffix_ids) - 1
    if len(prompt_ids) > room:
        prompt_ids = prompt_ids[:room]
    full_ids = prompt_ids + suffix_ids

    PAD = 151643
    valid = len(full_ids)
    ids = list(full_ids)
    if valid < SEQ_LEN:
        ids = ids + [PAD] * (SEQ_LEN - valid)
    full_prompt = prompt + suffix

    last_pos = valid - 1
    input_ids = np.array([ids], dtype=np.int32)
    t_pre = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    hidden_in = _cpu_embed(input_ids)
    out1 = _CACHE["part1"].forward(hidden_in)[0].buffer
    out2 = _CACHE["part2"].forward(out1)[0].buffer
    t_bpu = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    logits = _cpu_lm_head(out2, last_pos)

    # 3-way verdict token logit probe
    verdict_logits = {k: float(logits[tid]) for k, tid in VERDICT_TOKENS.items()}
    vals = np.array(list(verdict_logits.values()))
    vals_shifted = vals - vals.max()
    probs = np.exp(vals_shifted) / np.exp(vals_shifted).sum()
    verdict_probs = {k: round(float(p), 3) for k, p in zip(verdict_logits.keys(), probs)}
    verdict = max(verdict_probs, key=verdict_probs.get)
    confidence = verdict_probs[verdict]

    # 诊断: 全词表 top5
    topk = np.argsort(-logits)[:5]
    topk_toks = [_CACHE["tok"].id_to_token(int(tid)) or "" for tid in topk]

    t_post = (time.perf_counter() - t2) * 1000
    t_total = (time.perf_counter() - t0) * 1000

    return {
        "verdict": verdict,
        "confidence": confidence,
        "verdict_probs": verdict_probs,
        "verdict_logits": {k: round(v, 2) for k, v in verdict_logits.items()},
        "top5_tokens": topk_toks,
        "method": "bpu_qwen2_chain_int8 + verdict_token_probing",
        "latency_ms": round(t_total, 1),
        "cpu_pre_ms": round(t_pre, 1),
        "bpu_forward_ms": round(t_bpu, 1),
        "cpu_post_ms": round(t_post, 1),
        "last_valid_pos": last_pos,
        "prompt_chars": len(full_prompt),
    }


def healthcheck() -> dict:
    try:
        _ensure_loaded()
        return {"ok": True,
                "part1_bin": str(PART1_BIN),
                "part2_bin": str(PART2_BIN),
                "method": "BPU Bayes-e INT8, Qwen2-0.5B 24-layer chain (12+12) + verdict probe"}
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"ok": False, "error": str(e)[:300]}


if __name__ == "__main__":
    import sys, json
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
             "化学式 Y3Al5O12 + Cr3+ 1% YAG:Cr"
    print(f"prompt: {prompt}")
    print(json.dumps(healthcheck(), ensure_ascii=False, indent=2))
    print(json.dumps(bpu_qwen_verdict(prompt), ensure_ascii=False, indent=2))
