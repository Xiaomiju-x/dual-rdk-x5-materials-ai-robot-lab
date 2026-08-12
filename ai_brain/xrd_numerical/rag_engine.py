#!/usr/bin/env python3
"""
XRD RAG Engine — 语义向量检索增强生成
双模式:
  1. DashScope text-embedding-v3 (主, X5手机热点联网)
  2. 本地TF-IDF (备, 断网自动降级)

用法:
  from rag_engine import RAGEngine
  rag = RAGEngine()  # 自动加载 xrd_knowledge/embeddings/
  context = rag.retrieve("石榴石XRD分析", top_k=5)
"""

import os
import re
import json
import math
import numpy as np
from collections import Counter

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# DashScope Embedding API
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
                                     timeout=30, verify=True)
                resp.raise_for_status()
                data = resp.json()
                embeddings = sorted(data["data"], key=lambda x: x["index"])
                all_vectors.extend([e["embedding"] for e in embeddings])
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    _time.sleep(2 * (attempt + 1))
                else:
                    raise RuntimeError(f"嵌入API调用失败: {e}") from e
    return all_vectors


class RAGEngine:
    """语义向量RAG检索引擎 (DashScope主 + TF-IDF备)"""

    def __init__(self, chunks_path=None, vectors_path=None):
        if chunks_path is None:
            chunks_path = os.path.join(_SCRIPT_DIR, "xrd_knowledge", "embeddings", "chunks.json")
        if vectors_path is None:
            vectors_path = os.path.join(_SCRIPT_DIR, "xrd_knowledge", "embeddings", "vectors.npy")

        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)

        # 加载预计算向量
        self.vectors = None
        if os.path.isfile(vectors_path):
            self.vectors = np.load(vectors_path).astype(np.float32)
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.vectors = self.vectors / norms

        # TF-IDF 延迟构建: 稠密矩阵 ~n×vocab×4B, 断网 fallback 才真用
        # 正常联网走 DashScope 向量路径, TF-IDF 矩阵永远不建 → 省 RSS
        self._tfidf_ready = False
        self.vocab = None
        self.idf = None
        self.tfidf_matrix = None

        if self.vectors is None:
            # 无 vectors 兜底: 必须同步建 TF-IDF, 否则完全检索不了
            self._ensure_tfidf_index()
            mode = "TF-IDF (eager, 无向量)"
        else:
            mode = "向量 (TF-IDF lazy, 断网时才建)"
        print(f"[RAG] 加载知识库: {len(self.chunks)}段落, 模式={mode}")

    def _ensure_tfidf_index(self):
        if self._tfidf_ready:
            return
        import time as _t
        t0 = _t.time()
        print(f"[RAG] 首次 TF-IDF 建索引 (n={len(self.chunks)}, 可能 5-15s)...", flush=True)
        self._build_tfidf_index()
        self._tfidf_ready = True
        print(f"[RAG] TF-IDF 建索引完成 {_t.time()-t0:.1f}s", flush=True)

    def _tokenize(self, text):
        tokens = []
        for m in re.finditer(r'[a-zA-Z][a-zA-Z0-9_]*|[0-9]+\.?[0-9]*', text.lower()):
            tokens.append(m.group())
        chinese = re.findall(r'[\u4e00-\u9fff]+', text)
        for seg in chinese:
            for i in range(len(seg)):
                tokens.append(seg[i])
                if i < len(seg) - 1:
                    tokens.append(seg[i:i+2])
        return tokens

    def _build_tfidf_index(self):
        n = len(self.chunks)
        doc_tokens = []
        df_counter = Counter()
        for chunk in self.chunks:
            tokens = self._tokenize(chunk.get('text', ''))
            doc_tokens.append(tokens)
            for t in set(tokens):
                df_counter[t] += 1

        self.vocab = {}
        idx = 0
        for word, df in df_counter.items():
            if df >= 2:
                self.vocab[word] = idx
                idx += 1

        vocab_size = len(self.vocab)
        self.idf = np.zeros(vocab_size, dtype=np.float32)
        for word, widx in self.vocab.items():
            self.idf[widx] = math.log((n + 1) / (df_counter[word] + 1)) + 1

        self.tfidf_matrix = np.zeros((n, vocab_size), dtype=np.float32)
        for i, tokens in enumerate(doc_tokens):
            tf = Counter(tokens)
            for word, count in tf.items():
                if word in self.vocab:
                    widx = self.vocab[word]
                    self.tfidf_matrix[i, widx] = (1 + math.log(count)) * self.idf[widx]

        norms = np.linalg.norm(self.tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.tfidf_matrix = self.tfidf_matrix / norms

    def _tfidf_query(self, query):
        tokens = self._tokenize(query)
        tf = Counter(tokens)
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        for word, count in tf.items():
            if word in self.vocab:
                widx = self.vocab[word]
                vec[widx] = (1 + math.log(count)) * self.idf[widx]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def _vector_retrieve(self, query, top_k):
        """DashScope向量检索"""
        q_vec = np.array(_embed_texts([query])[0], dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm
        scores = self.vectors @ q_vec
        return scores

    def _tfidf_retrieve(self, query):
        """本地TF-IDF检索 (lazy build, 首次耗 5-15s)"""
        self._ensure_tfidf_index()
        q_vec = self._tfidf_query(query)
        return self.tfidf_matrix @ q_vec

    def retrieve(self, query, top_k=5):
        """
        检索: 优先DashScope向量, 失败回退TF-IDF

        返回格式化字符串, 包含 [Ref.N] 标签
        """
        scores = None
        mode = "tfidf"

        # 尝试向量检索
        if self.vectors is not None:
            try:
                scores = self._vector_retrieve(query, top_k)
                mode = "vector"
            except Exception as e:
                print(f"  [RAG] 向量检索失败({e}), 回退TF-IDF")

        # 回退TF-IDF
        if scores is None:
            scores = self._tfidf_retrieve(query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        parts = []
        for rank, idx in enumerate(top_indices):
            chunk = self.chunks[idx]
            score = float(scores[idx])
            if score < 0.01:
                continue
            source = chunk.get("source", "unknown")
            category = chunk.get("category", "")
            src_label = os.path.splitext(os.path.basename(source))[0]
            if len(src_label) > 30:
                src_label = src_label[:30] + "..."
            ref_tag = f"[Ref.{rank+1}]"
            header = f"{ref_tag} ({src_label}, {category}, score={score:.2f}, {mode})"
            parts.append(f"{header}\n{chunk['text']}")

        return "\n\n".join(parts) if parts else ""

    def retrieve_chunks(self, query, top_k=5):
        """返回原始chunk列表(带score)"""
        scores = None
        if self.vectors is not None:
            try:
                scores = self._vector_retrieve(query, top_k)
            except Exception:
                pass
        if scores is None:
            scores = self._tfidf_retrieve(query)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(scores[idx])
            results.append(chunk)
        return results


if __name__ == "__main__":
    import sys
    rag = RAGEngine()
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "石榴石XRD立方晶系Ia-3d"
    print(f"\n查询: {query}\n")
    print(rag.retrieve(query, top_k=5))
