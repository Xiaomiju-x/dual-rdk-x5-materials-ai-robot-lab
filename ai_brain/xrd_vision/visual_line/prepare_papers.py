#!/usr/bin/env python3
"""
XRD RAG 论文预处理脚本 — 在Windows开发机上运行

功能:
  1. 读取已有16篇txt论文 (xrd_knowledge/papers/)
  2. 读取42篇新PDF论文 (xrd_knowledge/papers_new/*/)
  3. 切块 (~300字/块, 50字重叠)
  4. 调DashScope text-embedding-v3 批量嵌入
  5. 输出 chunks.json + vectors.npy

用法:
  pip install pdfplumber   # 首次需安装
  python prepare_papers.py
"""

import os
import sys
import json
import time
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPERS_DIR = os.path.join(_SCRIPT_DIR, "xrd_knowledge", "papers")
PAPERS_NEW_DIR = os.path.join(_SCRIPT_DIR, "xrd_knowledge", "papers_new")
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "xrd_knowledge", "embeddings")

CHUNK_SIZE = 300      # 每块约300字
CHUNK_OVERLAP = 50    # 重叠50字


# ============================================================
# PDF 文本提取
# ============================================================
def extract_pdf_text(pdf_path):
    """用pdfplumber提取PDF文本"""
    try:
        import pdfplumber
    except ImportError:
        print("[ERROR] 请先安装: pip install pdfplumber")
        sys.exit(1)

    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"  [WARN] pdfplumber失败: {e}")
        return ""


# ============================================================
# 文本切块
# ============================================================
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """将长文本切成重叠块"""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap

    return chunks


# ============================================================
# 论文收集
# ============================================================
def collect_papers():
    """收集所有论文文本, 返回 [{source, category, title, text}]"""
    papers = []

    # 1. 已有txt论文
    if os.path.isdir(PAPERS_DIR):
        for fname in sorted(os.listdir(PAPERS_DIR)):
            if fname.endswith(('.txt', '.md', '.json')):
                fpath = os.path.join(PAPERS_DIR, fname)
                with open(fpath, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                if text:
                    papers.append({
                        "source": fname,
                        "category": "existing",
                        "title": fname.replace('.txt', ''),
                        "text": text,
                    })
        print(f"[1/3] 已有txt论文: {len(papers)}篇")

    # 2. 新PDF论文 (按子目录分类)
    pdf_count = 0
    pdf_fail = 0
    category_map = {
        "Fe3_plus": "Fe3+",
        "Cr3_plus": "Cr3+",
        "Ni2_plus": "Ni2+",
        "review_methods": "review",
        "other_systems": "other",
    }

    if os.path.isdir(PAPERS_NEW_DIR):
        for subdir in sorted(os.listdir(PAPERS_NEW_DIR)):
            subdir_path = os.path.join(PAPERS_NEW_DIR, subdir)
            if not os.path.isdir(subdir_path):
                continue
            category = category_map.get(subdir, subdir)

            for fname in sorted(os.listdir(subdir_path)):
                if not fname.endswith('.pdf'):
                    continue
                fpath = os.path.join(subdir_path, fname)
                print(f"  提取: {subdir}/{fname[:50]}...")

                text = extract_pdf_text(fpath)
                if not text or len(text) < 100:
                    print(f"  [FAIL] 文本过短或为空 ({len(text)}字), 跳过")
                    pdf_fail += 1
                    continue

                # 截取前3000字(论文太长的话只取摘要+实验+结果)
                if len(text) > 3000:
                    text = text[:3000]

                papers.append({
                    "source": f"{subdir}/{fname}",
                    "category": category,
                    "title": fname.replace('.pdf', '')[:60],
                    "text": text,
                })
                pdf_count += 1

    print(f"[2/3] 新PDF论文: 成功{pdf_count}篇, 失败{pdf_fail}篇")
    if pdf_fail > 0:
        print(f"  → 失败的PDF可能是扫描件, 请手动提取文本存为txt放入papers/目录")

    return papers


# ============================================================
# 嵌入
# ============================================================
def embed_chunks(chunks_data):
    """批量嵌入所有chunk的文本"""
    # 复用rag_engine的嵌入函数
    sys.path.insert(0, _SCRIPT_DIR)
    from rag_engine import _embed_texts

    # 清理文本: 截断过长文本, 移除空字符
    texts = [c["text"][:500].replace('\x00', '') for c in chunks_data]
    print(f"[3/3] 嵌入 {len(texts)} 个段落...")

    all_vectors = []
    failed_indices = []  # 记录嵌入失败的chunk索引
    batch_size = 10  # 小batch避免超限
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_end = min(i + batch_size, len(texts))
        print(f"  嵌入 {i+1}-{batch_end}/{len(texts)}...")
        try:
            vecs = _embed_texts(batch)
            all_vectors.extend(vecs)
        except Exception as e:
            print(f"  [SKIP] 嵌入失败 ({i+1}-{batch_end}): {e}")
            failed_indices.extend(range(i, batch_end))
            time.sleep(1)

    # 从chunks_data中移除嵌入失败的条目
    if failed_indices:
        failed_set = set(failed_indices)
        kept = [c for idx, c in enumerate(chunks_data) if idx not in failed_set]
        removed = len(chunks_data) - len(kept)
        print(f"  [INFO] 跳过{removed}个嵌入失败的段落, 保留{len(kept)}个")
        chunks_data.clear()
        chunks_data.extend(kept)

    return np.array(all_vectors, dtype=np.float32)


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("  XRD RAG 论文预处理")
    print("=" * 60)

    # 1. 收集论文
    papers = collect_papers()
    print(f"\n总计: {len(papers)}篇论文")

    if not papers:
        print("[ERROR] 未找到任何论文!")
        return

    # 2. 切块
    all_chunks = []
    for paper in papers:
        chunks = chunk_text(paper["text"])
        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "source": paper["source"],
                "category": paper["category"],
                "title": paper["title"],
                "chunk_idx": idx,
                "text": chunk,
            })

    print(f"切块完成: {len(all_chunks)}个段落 (平均{len(all_chunks)/len(papers):.1f}块/篇)")

    # 3. 嵌入
    vectors = embed_chunks(all_chunks)
    print(f"嵌入完成: {vectors.shape}")

    # 4. 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    chunks_path = os.path.join(OUTPUT_DIR, "chunks.json")
    vectors_path = os.path.join(OUTPUT_DIR, "vectors.npy")

    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=1)
    np.save(vectors_path, vectors)

    print(f"\n输出:")
    print(f"  {chunks_path} ({len(all_chunks)}块)")
    print(f"  {vectors_path} ({vectors.shape})")
    print(f"\n完成! 将以下文件部署到X5:")
    print(f"  scp {chunks_path} rdk@x5:~/xrd1/xrd_knowledge/embeddings/")
    print(f"  scp {vectors_path} rdk@x5:~/xrd1/xrd_knowledge/embeddings/")


if __name__ == "__main__":
    main()
