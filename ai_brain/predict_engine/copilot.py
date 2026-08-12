"""文献副驾 (Research Copilot) — RAG 对话 + 逐句引用溯源.

第 2 期 #1 (2026-06-11): 对标 Bohrium 文献副驾.
- 检索: BM25 (离线) + dense query embedding (DashScope, 非 HyDE — 聊天要快) → RRF
- 生成: DeepSeek-R1 (deep) / DeepSeek-chat (fast) 真流式, 强制 [n] 行内引用
- 溯源: 每条引用映射回 25228 chunk 原文段落 + 推测 DOI / Scholar 链接
- 降级: dense 不通 → BM25-only; 云 LLM 不通 → 离线模板 (只列检索结果)

依赖: spectrum_knowledge_shared.{bm25_index, hyde_retriever(私有 embed/corpus), rrf_fusion}
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Iterator, Optional

import numpy as np
import requests

# ---- 复用现有检索栈 ----
from spectrum_knowledge_shared.bm25_index import bm25_search
from spectrum_knowledge_shared.hyde_retriever import _embed_text, _lazy_corpus
from spectrum_knowledge_shared.rrf_fusion import rrf_fusion

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_R1_KEY", "")
MODEL_DEEP = "deepseek-reasoner"
MODEL_FAST = "deepseek-chat"


# ============================================================ DOI 推测
# 出版社 PDF 文件名即 DOI 后缀的常见约定 (title 字段切自文件名).
# 推测结果标 doi_guess=True, 前端注明"按文件名推测"; 无法推测给 Scholar 检索链接.
_DOI_RULES = [
    (re.compile(r"^acs\.[a-z]+\.\w+$"), "10.1021/"),          # ACS: acs.inorgchem.1c01835
    (re.compile(r"^j\.[a-z]+\.\d{4}\.\w+$"), "10.1016/"),     # Elsevier: j.jlumin.2022.119123
    (re.compile(r"^s\d{5}-\d{3}-\w+(-\w+)?$"), "10.1038/"),   # Nature: s41598-021-91838-4
    (re.compile(r"^(adfm|adom|adma|advs|smll|anie|chem|ejic|ange)\.\w+$"), "10.1002/"),  # Wiley
    (re.compile(r"^[cd]\d[a-z]{2}\d{5}[a-z]$"), "10.1039/"),  # RSC: d1tc04332c / c8tc05623d
    (re.compile(r"^PhysRev[A-Z]\.\d+\.\d+$"), "10.1103/"),    # APS
]


def derive_doi(title: str) -> Optional[str]:
    t = (title or "").strip()
    for pat, prefix in _DOI_RULES:
        if pat.match(t):
            return prefix + t
    return None


# ============================================================ 快检索 (BM25 + dense + RRF)
def _dense_search(query: str, top_k: int = 20) -> list[dict]:
    """直接 query embedding 余弦检索 (跳过 HyDE 的 LLM 幻觉步, 聊天延迟 ~1s)."""
    vec = _embed_text(query)                      # DashScope 1024 维
    vectors, chunks = _lazy_corpus()
    v = vec / (np.linalg.norm(vec) + 1e-9)
    sims = vectors @ v                            # 语料向量已归一化 (建库时)
    idx = np.argsort(-sims)[:top_k]
    out = []
    for i in idx:
        i = int(i)
        c = chunks[i]
        # chunk_idx 必须是全局 int 下标 — rrf_fusion 按它跟 bm25 命中去重
        out.append({"chunk_idx": i, "score": float(sims[i]),
                    "text": c.get("text", ""), "source": c.get("source", ""),
                    "title": c.get("title", "")})
    return out


def retrieve(query: str, k: int = 8) -> tuple[list[dict], str]:
    """混合检索 → 编号 sources. 返回 (sources, method)."""
    ranked, method = [], "bm25_only"
    try:
        sparse = bm25_search(query, top_k=20)
    except Exception as e:
        sparse = []
        print(f"[copilot] bm25 失败: {e}", flush=True)
    try:
        dense = _dense_search(query, top_k=20)
        if sparse:
            ranked = rrf_fusion([sparse, dense], k_const=60, top_n=k)
            method = "hybrid_bm25_dense_rrf"
        else:
            ranked, method = dense[:k], "dense_only"
    except Exception as e:
        print(f"[copilot] dense 失败 (离线?): {e}", flush=True)
        ranked = sparse[:k]
    sources = []
    for n, h in enumerate(ranked, 1):
        title = h.get("title") or os.path.splitext(os.path.basename(h.get("source", "")))[0]
        doi = derive_doi(title)
        sources.append({
            "n": n,
            "title": title,
            "text": (h.get("text") or "").strip(),
            "source": h.get("source", ""),
            "doi": doi,
            "doi_url": f"https://doi.org/{doi}" if doi else None,
            "scholar_url": "https://scholar.google.com/scholar?q=" + requests.utils.quote(title),
            "score": round(float(h.get("rrf_score") or h.get("score") or 0), 4),
        })
    return sources, method


# ============================================================ Prompt
_SYS = (
    "你是 NIR 荧光粉实验室的文献副驾 (research copilot)。仅根据下方编号文献片段回答, "
    "每个论断句末必须用 [n] 标注来源编号 (可多个如 [1][3]); 片段没有的信息明确说"
    "\"检索片段未覆盖\", 严禁编造数据、晶体参数或 DOI。用中文, 面向科研人员, "
    "适度使用专业术语; 数值/峰位/浓度等关键参数尽量原样引用。最后用一行"
    "\"📌 要点\"总结。"
)


def _build_messages(query: str, sources: list[dict], history: list[dict]) -> list[dict]:
    src_block = "\n\n".join(
        f"[{s['n']}] ({s['title']}) {s['text'][:900]}" for s in sources)
    msgs = [{"role": "system", "content": _SYS}]
    for h in (history or [])[-6:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            msgs.append({"role": h["role"], "content": str(h["content"])[:1500]})
    msgs.append({"role": "user",
                 "content": f"文献片段:\n{src_block}\n\n问题: {query}"})
    return msgs


# ============================================================ 流式生成
def stream_chat(query: str, sources: list[dict], history: list[dict],
                mode: str = "deep") -> Iterator[dict]:
    """yield {type: thinking|delta|done|error, ...}. 云断时降级离线模板."""
    model = MODEL_DEEP if mode == "deep" else MODEL_FAST
    body = {"model": model, "messages": _build_messages(query, sources, history),
            "stream": True, "max_tokens": 2000}
    t0 = time.time()
    try:
        resp = requests.post(
            DEEPSEEK_URL, json=body, stream=True, timeout=(10, 120),
            headers={"Authorization": f"Bearer {DEEPSEEK_KEY}",
                     "Content-Type": "application/json"})
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[5:].strip()
            if data == "[DONE]":
                break
            try:
                delta = json.loads(data)["choices"][0]["delta"]
            except Exception:
                continue
            rc = delta.get("reasoning_content")
            if rc:
                yield {"type": "thinking", "text": rc}
            ct = delta.get("content")
            if ct:
                yield {"type": "delta", "text": ct}
        yield {"type": "done", "model": model, "latency_ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        print(f"[copilot] 云 LLM 失败, 降级离线模板: {e}", flush=True)
        # 三级降级末位: 不写答案, 仅诚实列出检索结果
        lines = [f"⚠️ 云端 LLM 不可达 ({type(e).__name__}), 以下为离线检索结果摘要:\n"]
        for s in sources[:5]:
            lines.append(f"[{s['n']}] {s['title']}: {s['text'][:200]}…")
        yield {"type": "delta", "text": "\n".join(lines)}
        yield {"type": "done", "model": "offline_template",
               "latency_ms": int((time.time() - t0) * 1000)}
