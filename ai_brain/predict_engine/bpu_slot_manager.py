"""bpu_slot_manager.py — X5 端 5-slot BPU 切换管理器 (Round 8 v2).

背景: X5 CMA 391MB 一次只能装一个模型. 为展示"BPU 能跑多种 LLM", 实现:
    5 个 slot (3 × Qwen2-0.5B 变体 + Qwen3-1.7B + R1-Distill-1.5B),
    一次只装一个, 按请求切换 (swap-load).

每 slot 配置 (bins + tokenizer + CPU tensors + arch 差异):
    - 0.5B × 3 (hidden=896, 2 bins, 553ms/forward, 装 180MB+180MB)
    - Qwen3-1.7B (hidden=2048, 10 bins, ~7s/forward, q_norm/k_norm, 装 10×~150MB)
    - R1-Distill-1.5B (hidden=1536, 10 bins, ~5s/forward, qwen2 arch, 装 10×~130MB)

切换开销: unload 旧 bins → 释放 CMA → load 新 bins 依次 (~500ms/bin) → load 新 embed+norm
"""
from __future__ import annotations
import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np

_MODELS_DIR = Path("/home/rdk/models")
SEQ_LEN = 64
PAD_ID = 151643  # Qwen2 / Qwen3 / R1-Distill 都用相同 pad


@dataclass
class SlotCfg:
    name: str
    label: str            # UI 展示用中文
    bins: list[Path]
    tokenizer_file: Path
    embed_file: Path      # fp32 npy [vocab, hidden]
    norm_file: Path       # fp32 npy [hidden]
    hidden_size: int
    arch: str             # "qwen2" / "qwen3"
    verdict_tokens: dict  # {"GO": tid, "REVISE": tid, "DROP": tid} (空 = 运行时 resolve)
    verdict_suffix: str   # 提示语后缀 (e.g. "\n判:")
    tie_word_embeddings: bool = True   # True=lm_head 共享 embed
    lm_head_file: Path | None = None   # tie=False 时用这个独立 lm_head [vocab, hidden]
    lazy_bins: bool = False            # True: switch() 仅校验 bin 路径, verdict() 每段 load→forward→del (防 CMA OOM)
    note: str = ""

    def available(self) -> bool:
        if not self.tokenizer_file.exists() or not self.embed_file.exists() or not self.norm_file.exists():
            return False
        if not self.tie_word_embeddings and (self.lm_head_file is None or not self.lm_head_file.exists()):
            return False
        return all(b.exists() for b in self.bins)


# --- Slot 定义 (5 个) ---
# Qwen2 tokenizer 的 verdict tokens (已验证)
QWEN2_VERDICT = {"GO": 101914, "REVISE": 105023, "DROP": 102256}
# Qwen3 用同一个 Qwen tokenizer.json, verdict tokens 应同. R1-Distill 用 DeepSeek tokenizer, 需单独 resolve.

SLOTS: dict[str, SlotCfg] = {
    "generic_05b": SlotCfg(
        name="generic_05b", label="Qwen2-0.5B generic (Round 8)",
        bins=[_MODELS_DIR / "qwen2_part1.bin", _MODELS_DIR / "qwen2_part2.bin"],
        tokenizer_file=_MODELS_DIR / "tokenizer.json",
        embed_file=_MODELS_DIR / "embed_tokens_fp16.npy",
        norm_file=_MODELS_DIR / "norm_final_fp32.npy",
        hidden_size=896, arch="qwen2",
        verdict_tokens=QWEN2_VERDICT, verdict_suffix="\n判:",
        note="通用专家 (原 Round 8 bin)",
    ),
    "nir_05b": SlotCfg(
        name="nir_05b", label="Qwen2-0.5B NIR 专家 LoRA",
        bins=[_MODELS_DIR / "qwen2_05b_nir_part1.bin", _MODELS_DIR / "qwen2_05b_nir_part2.bin"],
        tokenizer_file=_MODELS_DIR / "tokenizer.json",
        embed_file=_MODELS_DIR / "qwen2_05b_nir_embed_fp32.npy",
        norm_file=_MODELS_DIR / "qwen2_05b_nir_norm_fp32.npy",
        hidden_size=896, arch="qwen2",
        verdict_tokens=QWEN2_VERDICT, verdict_suffix="\n判:",
        note="NIR 荧光粉 584 silver SFT",
    ),
    "verdict_05b": SlotCfg(
        name="verdict_05b", label="Qwen2-0.5B verdict 专家 LoRA",
        bins=[_MODELS_DIR / "qwen2_05b_verdict_part1.bin", _MODELS_DIR / "qwen2_05b_verdict_part2.bin"],
        tokenizer_file=_MODELS_DIR / "tokenizer.json",
        embed_file=_MODELS_DIR / "qwen2_05b_verdict_embed_fp32.npy",
        norm_file=_MODELS_DIR / "qwen2_05b_verdict_norm_fp32.npy",
        hidden_size=896, arch="qwen2",
        verdict_tokens=QWEN2_VERDICT, verdict_suffix="\n判:",
        note="R1 verdict <think>+JSON SFT",
    ),
    "qwen3_17b": SlotCfg(
        name="qwen3_17b", label="Qwen3-1.7B NIR 10-seg (scale)",
        bins=[_MODELS_DIR / f"qwen3_17b_part{i:02d}.bin" for i in range(1, 11)],
        tokenizer_file=_MODELS_DIR / "qwen3_tokenizer.json",
        embed_file=_MODELS_DIR / "qwen3_17b_embed_fp32.npy",
        norm_file=_MODELS_DIR / "qwen3_17b_norm_fp32.npy",
        hidden_size=2048, arch="qwen3",
        verdict_tokens=QWEN2_VERDICT, verdict_suffix="\n判:",
        lazy_bins=True,
        note="1.7B 10-seg swap (slow, demo)",
    ),
    "r1_distill_15b": SlotCfg(
        name="r1_distill_15b", label="R1-Distill-Qwen-1.5B 10-seg (推理)",
        bins=[_MODELS_DIR / f"r1_distill_part{i:02d}.bin" for i in range(1, 11)],
        tokenizer_file=_MODELS_DIR / "r1_distill_tokenizer.json",
        embed_file=_MODELS_DIR / "r1_distill_embed_fp32.npy",
        norm_file=_MODELS_DIR / "r1_distill_norm_fp32.npy",
        hidden_size=1536, arch="qwen2",
        verdict_tokens={},    # R1-Distill vocab IDs 不同, 需运行时 resolve
        verdict_suffix="\n判:",
        tie_word_embeddings=False,
        lm_head_file=_MODELS_DIR / "r1_distill_lm_head_fp32.npy",
        lazy_bins=True,
        note="R1 <think> 推理风格",
    ),
}


# --- 管理器 ---
class BpuSlotManager:
    def __init__(self):
        self._cur_name: str | None = None
        self._bins: list = []      # dnn models list
        self._tok = None
        self._embed: np.ndarray | None = None
        self._lm_head: np.ndarray | None = None    # 独立 lm_head (tie=False 时)
        self._norm_w: np.ndarray | None = None
        self._cfg: SlotCfg | None = None

    def list_available(self) -> list[dict]:
        out = []
        for name, cfg in SLOTS.items():
            out.append({
                "name": name, "label": cfg.label,
                "arch": cfg.arch, "hidden": cfg.hidden_size,
                "n_segs": len(cfg.bins), "available": cfg.available(),
                "note": cfg.note, "is_current": name == self._cur_name,
            })
        return out

    def switch(self, name: str) -> dict:
        """切换到指定 slot. 返回 {ok, t_unload_ms, t_load_ms, total_ms}."""
        if name == self._cur_name:
            return {"ok": True, "skipped": "already loaded", "total_ms": 0}
        if name not in SLOTS:
            return {"ok": False, "error": f"unknown slot {name}"}

        cfg = SLOTS[name]
        if not cfg.available():
            missing = []
            for b in cfg.bins:
                if not b.exists():
                    missing.append(str(b))
            return {"ok": False, "error": f"slot files missing: {missing[:3]}..."}

        t0 = time.perf_counter()
        # Unload 旧: 显式 del 各 bin (pyeasy_dnn Model __del__ 释放 CMA), 强制等 CMA 压缩
        old_bins = self._bins
        self._bins = []
        for m in old_bins:
            try: del m
            except Exception: pass
        self._embed = None
        self._lm_head = None
        self._norm_w = None
        gc.collect()
        gc.collect()
        time.sleep(1.2)   # give kernel CMA compactor time
        t_unload = (time.perf_counter() - t0) * 1000

        # Load 新
        t1 = time.perf_counter()
        from hobot_dnn import pyeasy_dnn as dnn
        bins = []
        if cfg.lazy_bins:
            # lazy: switch 不装 bin, verdict 时每段 load→forward→del (10 bin × 156MB 超 CMA 391MB)
            # 仅校验路径存在
            for p in cfg.bins:
                if not p.exists():
                    raise FileNotFoundError(str(p))
            self._bins = []  # lazy marker
        else:
            load_err: Exception | None = None
            for attempt in range(3):
                try:
                    bins = []
                    for p in cfg.bins:
                        m = dnn.load(str(p))[0]
                        bins.append(m)
                    load_err = None
                    break
                except Exception as e:
                    load_err = e
                    del bins; bins = []
                    gc.collect()
                    time.sleep(2.0 * (attempt + 1))
            if load_err is not None:
                self._cur_name = None
                raise load_err
            self._bins = bins

        from tokenizers import Tokenizer
        self._tok = Tokenizer.from_file(str(cfg.tokenizer_file))

        self._embed = np.load(str(cfg.embed_file)).astype(np.float32)
        self._norm_w = np.load(str(cfg.norm_file)).astype(np.float32)
        if not cfg.tie_word_embeddings and cfg.lm_head_file:
            self._lm_head = np.load(str(cfg.lm_head_file)).astype(np.float32)
        else:
            self._lm_head = None
        t_load = (time.perf_counter() - t1) * 1000

        self._cur_name = name
        self._cfg = cfg

        # 运行时 verdict token resolution (Qwen3 / R1-Distill 强制重查; Qwen2 slot 如果硬编码 ID 缺席也跑)
        # 关键: 必须保证 3 个 verdict 的 token_id 两两不同, 否则 3-way softmax 恒 33.3% 平局
        need_resolve = (not cfg.verdict_tokens) or cfg.arch == "qwen3" or cfg.name == "r1_distill_15b"
        if need_resolve:
            # 每个 verdict 多候选, 按序试, 选第一个 encode 成 ≥1 token 且首 token 与其他 verdict 首 token 不冲突的
            candidates = {
                "GO":     ["GO", " GO", "Go", "推荐", " 推荐", "是", "Yes", "yes", "A", "G"],
                "REVISE": ["REVISE", " REVISE", "Revise", "改进", " 改进", "改", "Revise", "B", "R"],
                "DROP":   ["DROP", " DROP", "Drop", "放弃", " 放弃", "弃", "No", "C", "D"],
            }
            resolved: dict[str, int] = {}
            def _enc_first(w):
                # add_special_tokens=False 避免 R1-Distill 之类 tokenizer 在每次 encode 前插 BOS
                try:
                    enc = self._tok.encode(w, add_special_tokens=False)
                except TypeError:
                    enc = self._tok.encode(w)   # 老版本 tokenizers 可能不支持该 kwarg
                return int(enc.ids[0]) if enc.ids else None
            for k, cands in candidates.items():
                for w in cands:
                    tid = _enc_first(w)
                    if tid is None:
                        continue
                    if tid in resolved.values():
                        continue
                    resolved[k] = tid
                    break
                if k not in resolved:
                    for ascii_c in "ABCDEFGHJKLM":
                        tid = _enc_first(ascii_c)
                        if tid is not None and tid not in resolved.values():
                            resolved[k] = tid
                            break
            if len(set(resolved.values())) < 3:
                # 3 个不能两两不同 — 保留但打 warn flag (dashboard 会识别并显示提示)
                print(f"[bpu_slot] WARN slot {cfg.name}: verdict tokens not fully distinct: {resolved}")
            cfg.verdict_tokens = resolved

        return {
            "ok": True, "name": name, "label": cfg.label,
            "n_bins": len(bins),
            "t_unload_ms": round(t_unload, 1),
            "t_load_ms": round(t_load, 1),
            "total_ms": round(t_unload + t_load, 1),
            "embed_shape": list(self._embed.shape),
        }

    def current(self) -> str | None:
        return self._cur_name

    def _cpu_embed(self, input_ids: np.ndarray) -> np.ndarray:
        """[1,L,int32] → [1,L,D,1,fp32] for BPU input (NCHW featuremap)."""
        out = self._embed[input_ids[0]]   # [L, D]
        return out.reshape(1, SEQ_LEN, self._cfg.hidden_size, 1).astype(np.float32)

    def _cpu_rmsnorm(self, h: np.ndarray, eps=1e-6) -> np.ndarray:
        var = (h * h).mean()
        return (h / np.sqrt(var + eps)) * self._norm_w

    def _cpu_lm_head(self, post_hidden: np.ndarray, last_valid: int) -> np.ndarray:
        h = post_hidden.reshape(1, SEQ_LEN, self._cfg.hidden_size)
        last_h = h[0, last_valid, :]
        last_h_normed = self._cpu_rmsnorm(last_h)
        W = self._lm_head if self._lm_head is not None else self._embed
        return last_h_normed @ W.T

    def verdict(self, prompt: str, auto_suffix: bool = True) -> dict:
        """Prefill + 3-way verdict token logit probe."""
        if self._cur_name is None:
            return {"ok": False, "error": "no slot loaded, call switch() first"}

        cfg = self._cfg
        t0 = time.perf_counter()

        suffix = cfg.verdict_suffix if auto_suffix else ""
        # add_special_tokens=False 避免 R1-Distill 之类 tokenizer 在 prompt+suffix 中间插 BOS
        def _enc(s: str) -> list:
            if not s:
                return []
            try:
                return list(self._tok.encode(s, add_special_tokens=False).ids)
            except TypeError:
                return list(self._tok.encode(s).ids)
        suffix_ids = _enc(suffix)
        prompt_ids = _enc(prompt)
        room = SEQ_LEN - len(suffix_ids) - 1
        if len(prompt_ids) > room:
            prompt_ids = prompt_ids[:room]
        full_ids = prompt_ids + suffix_ids

        valid = len(full_ids)
        ids = list(full_ids)
        if valid < SEQ_LEN:
            ids = ids + [PAD_ID] * (SEQ_LEN - valid)
        last_pos = valid - 1
        input_ids = np.array([ids], dtype=np.int32)
        t_pre = (time.perf_counter() - t0) * 1000

        # BPU chain forward
        t1 = time.perf_counter()
        hidden = self._cpu_embed(input_ids)
        if self._cfg.lazy_bins:
            # lazy: 每 bin spawn 一个 subprocess (进程退出 CMA 必释放, del 不够)
            import subprocess, tempfile, os
            tmp = Path(tempfile.mkdtemp(prefix="bpu_chain_"))
            try:
                in_path = tmp / "hidden.npy"
                np.save(str(in_path), hidden)
                for i, p in enumerate(self._cfg.bins):
                    out_path = tmp / f"hidden_out_{i:02d}.npy"
                    r = subprocess.run(
                        ["python3", "-m", "predict_engine.bpu_bin_forward",
                         str(p), str(in_path), str(out_path)],
                        capture_output=True, timeout=60, cwd="/home/rdk",
                    )
                    if r.returncode != 0:
                        raise RuntimeError(f"bin {i} subprocess failed: {r.stderr.decode()[:200]}")
                    try:
                        in_path.unlink()
                    except OSError:
                        pass
                    # next iter input = this iter output
                    in_path = out_path
                hidden = np.load(str(in_path))
            finally:
                import shutil
                try: shutil.rmtree(tmp)
                except OSError: pass
        else:
            for bin_mod in self._bins:
                out = bin_mod.forward(hidden)
                hidden = out[0].buffer
        t_bpu = (time.perf_counter() - t1) * 1000

        # CPU post
        t2 = time.perf_counter()
        logits = self._cpu_lm_head(hidden, last_pos)
        vtoks = cfg.verdict_tokens
        verdict_logits = {k: float(logits[tid]) for k, tid in vtoks.items()}
        vals = np.array(list(verdict_logits.values()))
        vs = vals - vals.max()
        probs = np.exp(vs) / np.exp(vs).sum()
        verdict_probs = {k: round(float(p), 3) for k, p in zip(verdict_logits.keys(), probs)}
        verdict = max(verdict_probs, key=verdict_probs.get)
        confidence = verdict_probs[verdict]
        topk = np.argsort(-logits)[:5]
        topk_toks = [self._tok.id_to_token(int(tid)) or "" for tid in topk]
        t_post = (time.perf_counter() - t2) * 1000
        t_total = (time.perf_counter() - t0) * 1000

        return {
            "ok": True,
            "slot": self._cur_name, "slot_label": cfg.label,
            "verdict": verdict, "confidence": confidence,
            "verdict_probs": verdict_probs,
            "verdict_logits": {k: round(v, 2) for k, v in verdict_logits.items()},
            "top5_tokens": topk_toks,
            "latency_ms": round(t_total, 1),
            "cpu_pre_ms": round(t_pre, 1),
            "bpu_forward_ms": round(t_bpu, 1),
            "cpu_post_ms": round(t_post, 1),
            "n_bins": len(self._bins),
            "last_valid_pos": last_pos,
            "prompt_chars": len(prompt + suffix),
        }


# --- 全局单例 ---
_MGR: BpuSlotManager | None = None


def get_manager() -> BpuSlotManager:
    global _MGR
    if _MGR is None:
        _MGR = BpuSlotManager()
    return _MGR


# --- Dashboard-facing wrapper (replaces bpu_qwen_verdict) ---
def bpu_slot_verdict(prompt: str, slot: str = "generic_05b",
                     auto_append_suffix: bool = True) -> dict:
    mgr = get_manager()
    switch_info = mgr.switch(slot)
    if not switch_info["ok"]:
        return {"ok": False, "error": switch_info.get("error"), "slot": slot}
    t_switch = switch_info.get("total_ms", 0)

    result = mgr.verdict(prompt, auto_append_suffix)
    result["switch_ms"] = t_switch
    return result


def healthcheck() -> dict:
    slots = []
    for name, cfg in SLOTS.items():
        slots.append({
            "name": name, "label": cfg.label,
            "arch": cfg.arch, "hidden": cfg.hidden_size,
            "n_segs": len(cfg.bins), "available": cfg.available(),
            "note": cfg.note,
        })
    return {"ok": True, "n_slots": len(slots), "slots": slots,
            "method": "5-slot BPU swap-load (Qwen2-0.5B×3 + Qwen3-1.7B + R1-Distill-1.5B)"}


if __name__ == "__main__":
    import sys
    print(json.dumps(healthcheck(), ensure_ascii=False, indent=2))
    if len(sys.argv) > 1:
        slot = sys.argv[1]
        prompt = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else \
                 "化学式 Y3Al5O12 + Cr3+ 1% YAG:Cr"
        print(f"\nTesting slot={slot}, prompt={prompt}")
        print(json.dumps(bpu_slot_verdict(prompt, slot), ensure_ascii=False, indent=2))
