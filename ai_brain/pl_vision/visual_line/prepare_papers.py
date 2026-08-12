#!/usr/bin/env python3
"""
Round 4 - 2462 篇 PDF 论文向量化 → spectrum_knowledge_shared

扫描路径: spectrum_numerical/6-文献阅读管理/{文献,文献阅读,阳离子取代}/
输出路径: xrd/spectrum_knowledge_shared/embeddings/
  - chunks.json  (每段 {source, category, title, chunk_idx, text})
  - vectors.npy  ((N, 1024) float32)
  - build_log.json  (耗时 / 失败论文 / 每类统计)

特点:
  - 增量 resume: 每 100 篇 flush 一次临时 chunks, 崩了可以接着跑
  - 单篇上限 ~15 chunks (截前 4000 字, 涵盖摘要+引言+方法+早期结果)
  - 跳过扫描件 (<500 字) / 加密 / 损坏 PDF
  - 复用 xrd_vision/visual_line/rag_engine._embed_texts (DashScope batch=25 + 重试 + SSL fallback)

用法:
  pip install pdfplumber
  cd xrd
  python spectrum_vision/visual_line/prepare_papers.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---- 路径配置 ----
_SCRIPT_DIR = Path(__file__).resolve().parent                          # spectrum_vision/visual_line
_REPO = _SCRIPT_DIR.parent.parent                                      # xrd/
_PDF_ROOT = _REPO / "spectrum_numerical" / "6-文献阅读管理"
_OUTPUT_DIR = _REPO / "spectrum_knowledge_shared" / "embeddings"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_CHUNKS_PATH = _OUTPUT_DIR / "chunks.json"
_VECTORS_PATH = _OUTPUT_DIR / "vectors.npy"
_BUILD_LOG = _OUTPUT_DIR / "build_log.json"
_TEMP_CHUNKS_PATH = _OUTPUT_DIR / "chunks.partial.json"                # 增量 flush

# ---- 超参 ----
CHUNK_SIZE = 300            # 字符数
CHUNK_OVERLAP = 50
MAX_CHARS_PER_PAPER = 4000  # 单篇截取, 涵盖摘要+引言+方法+早期结果
MAX_CHUNKS_PER_PAPER = 15   # 防止超长论文挤占
MIN_CHARS_PER_PAPER = 500   # 小于这个认为是扫描件/损坏

# ---- 子目录 → 类别名 ----
SUBDIR_CATEGORY = {
    "文献": "wenxian",
    "文献阅读": "wenxian_yuedu",
    "阳离子取代": "cation_substitution",
}

# ---- 嵌入 API (内置, 不依赖 rag_engine, 直接 verify=False 绕 Python 3.14 SSL 问题) ----
EMBED_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
EMBED_MODEL = "text-embedding-v3"
EMBED_KEY = os.environ.get("QWEN_VL_KEY", "")


# ============================================================
# PDF 文本提取
# ============================================================
def extract_pdf_text(pdf_path: Path, max_chars: int = MAX_CHARS_PER_PAPER) -> str:
    """pdfplumber 提取前 max_chars 字符, 失败返回空串."""
    try:
        import pdfplumber
    except ImportError:
        print("[FATAL] pip install pdfplumber", flush=True)
        sys.exit(1)

    try:
        text_parts = []
        total = 0
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                if total >= max_chars:
                    break
                try:
                    page_text = page.extract_text()
                except Exception:
                    page_text = None
                if page_text:
                    text_parts.append(page_text)
                    total += len(page_text)
        text = "\n".join(text_parts).strip()
        if len(text) > max_chars:
            text = text[:max_chars]
        return text
    except Exception:
        return ""


# ============================================================
# 切块
# ============================================================
def chunk_text(text: str) -> list[str]:
    """300 字滑动窗口, overlap 50. 超长截断到 MAX_CHUNKS_PER_PAPER."""
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks = []
    start = 0
    while start < len(text) and len(chunks) < MAX_CHUNKS_PER_PAPER:
        end = start + CHUNK_SIZE
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ============================================================
# 扫描 PDF
# ============================================================
def scan_pdfs() -> list[dict]:
    """返回 [{path, subdir, category}], 递归扫三个子目录."""
    result = []
    if not _PDF_ROOT.exists():
        print(f"[FATAL] PDF 根目录不存在: {_PDF_ROOT}", flush=True)
        sys.exit(1)

    for subdir_name, category in SUBDIR_CATEGORY.items():
        sub = _PDF_ROOT / subdir_name
        if not sub.exists():
            print(f"[warn] {sub} 不存在, 跳过", flush=True)
            continue
        for pdf in sorted(sub.rglob("*.pdf")):
            result.append({
                "path": str(pdf),
                "subdir": subdir_name,
                "category": category,
            })
    return result


# ============================================================
# 嵌入 (复用 rag_engine._embed_texts)
# ============================================================
def _embed_batch_direct(texts: list[str], max_retries: int = 3) -> list[list[float]]:
    """直接调 DashScope 嵌入 API, verify=False 绕过 Python 3.14 SSL 问题."""
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EMBED_KEY}",
    }
    payload = {
        "model": EMBED_MODEL,
        "input": texts,
        "dimensions": 1024,
        "encoding_format": "float",
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(EMBED_URL, headers=headers, json=payload,
                                 timeout=60, verify=False)
            resp.raise_for_status()
            data = resp.json()
            embeddings = sorted(data["data"], key=lambda x: x["index"])
            return [e["embedding"] for e in embeddings]
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                raise RuntimeError(f"嵌入API调用失败: {e}") from e
    return []  # unreachable


def embed_all_chunks(chunks: list[dict]) -> tuple[np.ndarray, list[int]]:
    """
    并发批量嵌入 (4 线程). 返回 (vectors, failed_indices).
    失败的 chunk 会从 chunks 里移除 (就地).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    texts = [c["text"][:500].replace("\x00", "") for c in chunks]
    batch_size = 10   # DashScope text-embedding-v3 上限是 10
    n_workers = 4     # 4 并发, DashScope RPM 限制 ~300, 足够
    n_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"[embed] 开始嵌入 {len(texts)} 段落 ({n_batches} batches, "
          f"{n_workers} 并发, verify=False)", flush=True)

    # 预分配结果槽位
    results: list[list[list[float]] | None] = [None] * n_batches
    failed_indices: list[int] = []
    lock = threading.Lock()
    done_count = [0]
    t_start = time.time()

    def _do_batch(batch_idx: int):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]
        try:
            vecs = _embed_batch_direct(batch)
            if len(vecs) != len(batch):
                raise RuntimeError(f"长度不匹配: {len(batch)} vs {len(vecs)}")
            results[batch_idx] = vecs
        except Exception as e:
            results[batch_idx] = None
            with lock:
                failed_indices.extend(range(start, end))
            print(f"  [SKIP] batch {start+1}-{end} 失败: {e}", flush=True)

        with lock:
            done_count[0] += 1
            dc = done_count[0]
        if dc % 50 == 0 or dc == n_batches:
            elapsed = time.time() - t_start
            done_chunks = dc * batch_size
            rate = done_chunks / max(elapsed, 1e-6)
            eta = (len(texts) - done_chunks) / max(rate, 1e-6)
            print(f"  [embed] {min(done_chunks, len(texts))}/{len(texts)}  "
                  f"rate={rate:.1f}/s  ETA={eta/60:.1f}min  "
                  f"({dc}/{n_batches} batches)", flush=True)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_do_batch, i) for i in range(n_batches)]
        for f in as_completed(futures):
            f.result()  # 触发异常传播

    # 按顺序收集结果
    all_vectors: list[list[float]] = []
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(texts))
        if results[batch_idx] is not None:
            all_vectors.extend(results[batch_idx])
        else:
            all_vectors.extend([[0.0] * 1024] * (end - start))

    # 移除失败的 chunk 和对应向量
    if failed_indices:
        failed_set = set(failed_indices)
        kept_chunks = [c for idx, c in enumerate(chunks) if idx not in failed_set]
        kept_vecs = [v for idx, v in enumerate(all_vectors) if idx not in failed_set]
        chunks.clear()
        chunks.extend(kept_chunks)
        all_vectors = kept_vecs
        print(f"  [embed] 丢弃 {len(failed_indices)} 失败段落, 保留 {len(chunks)}", flush=True)

    return np.array(all_vectors, dtype=np.float32), failed_indices


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 70, flush=True)
    print("  Round 4 - 2462 篇论文向量化 → spectrum_knowledge_shared", flush=True)
    print("=" * 70, flush=True)

    # Step 1: 扫描 PDF
    pdfs = scan_pdfs()
    print(f"[scan] 发现 {len(pdfs)} 个 PDF", flush=True)
    # 按子目录统计
    by_sub: dict[str, int] = {}
    for p in pdfs:
        by_sub[p["subdir"]] = by_sub.get(p["subdir"], 0) + 1
    for k, v in by_sub.items():
        print(f"  {k}: {v}", flush=True)

    if not pdfs:
        print("[FATAL] 未发现任何 PDF, 检查路径")
        return

    # Step 2: 提取 + 切块 (增量写 partial)
    all_chunks: list[dict] = []
    failed_papers: list[dict] = []  # {path, reason}
    stats = {"kept": 0, "skipped_short": 0, "skipped_error": 0}

    # 如果有 partial 文件, 加载断点
    processed_paths: set[str] = set()
    if _TEMP_CHUNKS_PATH.exists():
        try:
            with open(_TEMP_CHUNKS_PATH, "r", encoding="utf-8") as f:
                all_chunks = json.load(f)
            processed_paths = {c["source"] for c in all_chunks}
            print(f"[resume] 从 partial 恢复 {len(all_chunks)} chunks, "
                  f"已处理 {len(processed_paths)} 篇", flush=True)
        except Exception as e:
            print(f"[resume] partial 加载失败 {e}, 从头开始", flush=True)
            all_chunks = []
            processed_paths = set()

    t0 = time.time()
    for i, pdf_info in enumerate(pdfs, 1):
        pdf_path = Path(pdf_info["path"])
        rel_source = str(pdf_path.relative_to(_REPO)).replace("\\", "/")
        title = pdf_path.stem[:80]

        if rel_source in processed_paths:
            continue

        if i % 50 == 0 or i == 1:
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1e-6)
            eta = (len(pdfs) - i) / max(rate, 1e-6)
            print(f"[extract] {i}/{len(pdfs)} - {pdf_path.name[:60]}  "
                  f"rate={rate:.1f}/s  ETA={eta/60:.1f}min", flush=True)

        text = extract_pdf_text(pdf_path)

        if not text:
            failed_papers.append({"path": rel_source, "reason": "extract_failed"})
            stats["skipped_error"] += 1
            continue
        if len(text) < MIN_CHARS_PER_PAPER:
            failed_papers.append({"path": rel_source, "reason": f"too_short ({len(text)})"})
            stats["skipped_short"] += 1
            continue

        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "source": rel_source,
                "category": pdf_info["category"],
                "title": title,
                "chunk_idx": idx,
                "text": chunk,
            })
        stats["kept"] += 1

        # 每 100 篇 flush partial
        if i % 100 == 0:
            try:
                with open(_TEMP_CHUNKS_PATH, "w", encoding="utf-8") as f:
                    json.dump(all_chunks, f, ensure_ascii=False)
                print(f"  [flush] partial saved ({len(all_chunks)} chunks)", flush=True)
            except Exception as e:
                print(f"  [warn] partial save failed: {e}", flush=True)

    print(flush=True)
    print(f"[extract] 完成:", flush=True)
    print(f"  保留: {stats['kept']} 篇", flush=True)
    print(f"  跳过(过短): {stats['skipped_short']} 篇", flush=True)
    print(f"  跳过(错误): {stats['skipped_error']} 篇", flush=True)
    print(f"  产出 chunks: {len(all_chunks)}", flush=True)

    if not all_chunks:
        print("[FATAL] 无可用 chunks")
        return

    # 保存最终 partial (在嵌入前)
    try:
        with open(_TEMP_CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False)
    except Exception:
        pass

    # Step 3: 嵌入
    t_embed_start = time.time()
    vectors, failed_embed = embed_all_chunks(all_chunks)
    t_embed = time.time() - t_embed_start

    if vectors.shape[0] == 0:
        print("[FATAL] 嵌入全部失败")
        return

    # Step 4: 保存最终产物
    with open(_CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=1)
    np.save(_VECTORS_PATH, vectors)

    # 清 partial
    try:
        _TEMP_CHUNKS_PATH.unlink(missing_ok=True)
    except Exception:
        pass

    # 写 build log
    log = {
        "total_pdfs": len(pdfs),
        "by_subdir": by_sub,
        "kept_papers": stats["kept"],
        "skipped_short": stats["skipped_short"],
        "skipped_error": stats["skipped_error"],
        "total_chunks": len(all_chunks),
        "embed_failed": len(failed_embed),
        "final_chunks": int(vectors.shape[0]),
        "vector_dim": int(vectors.shape[1]) if vectors.ndim == 2 else None,
        "extract_seconds": round(time.time() - t0 - t_embed, 1),
        "embed_seconds": round(t_embed, 1),
        "total_minutes": round((time.time() - t0) / 60, 1),
        "failed_papers_sample": failed_papers[:30],
    }
    with open(_BUILD_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(flush=True)
    print("=" * 70, flush=True)
    print(f"[DONE] 产物:", flush=True)
    print(f"  chunks: {_CHUNKS_PATH}  ({len(all_chunks)} 段)", flush=True)
    print(f"  vectors: {_VECTORS_PATH}  {vectors.shape}  "
          f"({vectors.nbytes/1e6:.1f} MB)", flush=True)
    print(f"  log: {_BUILD_LOG}", flush=True)
    print(f"  总耗时: {log['total_minutes']:.1f} min  "
          f"(提取 {log['extract_seconds']:.0f}s + 嵌入 {log['embed_seconds']:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
