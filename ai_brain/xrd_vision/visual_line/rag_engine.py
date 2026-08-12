#!/usr/bin/env python3
"""
XRD RAG Engine — 语义向量检索增强生成
基于 DashScope text-embedding-v3 嵌入 + numpy cosine similarity

用法:
  from rag_engine import RAGEngine
  rag = RAGEngine()  # 自动加载 xrd_knowledge/embeddings/
  context = rag.retrieve("石榴石XRD分析", top_k=5)
"""

import os
import json
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# DashScope Embedding API (和千问VL同平台)
EMBED_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
EMBED_MODEL = "text-embedding-v3"
EMBED_KEY = os.environ.get("QWEN_VL_KEY", "")


def _embed_texts(texts, api_key=None, max_retries=3):
    """调用DashScope嵌入API, 返回向量列表"""
    import requests
    import time as _time
    key = api_key or EMBED_KEY
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }
    # DashScope支持批量嵌入, 最多25条/请求
    all_vectors = []
    for i in range(0, len(texts), 25):
        batch = texts[i:i+25]
        payload = {
            "model": EMBED_MODEL,
            "input": batch,
            "dimensions": 1024,
            "encoding_format": "float",
        }
        for attempt in range(max_retries):
            try:
                resp = requests.post(EMBED_URL, headers=headers, json=payload,
                                     timeout=60, verify=True)
                resp.raise_for_status()
                data = resp.json()
                embeddings = sorted(data["data"], key=lambda x: x["index"])
                all_vectors.extend([e["embedding"] for e in embeddings])
                break
            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                if attempt < max_retries - 1:
                    wait = 3 * (attempt + 1)
                    print(f"  [SSL/连接错误] 第{attempt+1}次重试, {wait}s后...")
                    _time.sleep(wait)
                else:
                    # 最后一次尝试: 禁用SSL验证
                    print(f"  [SSL] 尝试禁用SSL验证...")
                    try:
                        resp = requests.post(EMBED_URL, headers=headers, json=payload,
                                             timeout=60, verify=False)
                        resp.raise_for_status()
                        data = resp.json()
                        embeddings = sorted(data["data"], key=lambda x: x["index"])
                        all_vectors.extend([e["embedding"] for e in embeddings])
                    except Exception as e2:
                        raise RuntimeError(f"嵌入API调用失败: {e2}") from e
    return all_vectors


class RAGEngine:
    """语义向量RAG检索引擎"""

    def __init__(self, chunks_path=None, vectors_path=None):
        if chunks_path is None:
            chunks_path = os.path.join(_SCRIPT_DIR, "xrd_knowledge", "embeddings", "chunks.json")
        if vectors_path is None:
            vectors_path = os.path.join(_SCRIPT_DIR, "xrd_knowledge", "embeddings", "vectors.npy")

        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)

        self.vectors = np.load(vectors_path).astype(np.float32)
        # L2归一化用于cosine similarity
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors = self.vectors / norms

        print(f"[RAG] 加载知识库: {len(self.chunks)}个段落, 维度={self.vectors.shape[1]}")

    def retrieve(self, query, top_k=5):
        """
        语义检索: query → 嵌入 → cosine similarity → Top-K段落

        返回格式化字符串, 包含 [Ref.N] 标签:
          [Ref.1] (来源, score=0.87): 段落文本...
          [Ref.2] (来源, score=0.82): 段落文本...
        """
        # 嵌入query
        q_vec = np.array(_embed_texts([query])[0], dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        # cosine similarity
        scores = self.vectors @ q_vec
        top_indices = np.argsort(scores)[::-1][:top_k]

        parts = []
        for rank, idx in enumerate(top_indices):
            chunk = self.chunks[idx]
            score = float(scores[idx])
            source = chunk.get("source", "unknown")
            category = chunk.get("category", "")
            # 简短来源标签
            src_label = os.path.splitext(os.path.basename(source))[0]
            if len(src_label) > 30:
                src_label = src_label[:30] + "..."
            ref_tag = f"[Ref.{rank+1}]"
            header = f"{ref_tag} ({src_label}, {category}, score={score:.2f})"
            parts.append(f"{header}\n{chunk['text']}")

        return "\n\n".join(parts)

    def retrieve_chunks(self, query, top_k=5):
        """返回原始chunk列表(带score), 用于更灵活的处理"""
        q_vec = np.array(_embed_texts([query])[0], dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        scores = self.vectors @ q_vec
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(scores[idx])
            results.append(chunk)
        return results


# ============================================================
# 独立测试
# ============================================================
if __name__ == "__main__":
    import sys
    rag = RAGEngine()
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "石榴石XRD立方晶系"
    print(f"\n查询: {query}\n")
    print(rag.retrieve(query, top_k=5))
