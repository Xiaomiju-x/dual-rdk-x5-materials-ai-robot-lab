# 向量RAG引擎 — 数值线接入参考

> 本文档供数值线(`web_demo.py`)的Claude Code参考，描述视觉线已实现的语义向量RAG系统。

---

## 1. 架构概述

```
论文PDF/TXT → prepare_papers.py切块+嵌入 → chunks.json + vectors.npy
                                              ↓
查询时: query → DashScope text-embedding-v3嵌入 → cosine similarity → Top-5段落[Ref.N] → LLM prompt
```

**核心改进**: 从"全部论文文本塞入prompt"升级为"语义检索Top-5最相关段落"，支持50+篇论文。

---

## 2. 文件位置

| 文件 | 路径 | 说明 |
|------|------|------|
| `rag_engine.py` | `yolo_xrd_detect/rag_engine.py` | RAG引擎类, 可直接import |
| `chunks.json` | `yolo_xrd_detect/xrd_knowledge/embeddings/chunks.json` | 论文切块+元数据 |
| `vectors.npy` | `yolo_xrd_detect/xrd_knowledge/embeddings/vectors.npy` | 嵌入向量矩阵 |
| `prepare_papers.py` | `yolo_xrd_detect/prepare_papers.py` | 预处理脚本(Windows运行) |

**数值线部署时**: 将`rag_engine.py` + `embeddings/`目录复制到数值线工作目录即可。

---

## 3. 接入方法

### 3a. 复制文件到数值线目录
```bash
cp yolo_xrd_detect/rag_engine.py /path/to/numerical_line/
cp -r yolo_xrd_detect/xrd_knowledge/embeddings /path/to/numerical_line/xrd_knowledge/
```

### 3b. 在web_demo.py中导入
```python
# 顶部
HAS_RAG = False
_rag = None
try:
    from rag_engine import RAGEngine
    _rag = RAGEngine("xrd_knowledge/embeddings/chunks.json",
                     "xrd_knowledge/embeddings/vectors.npy")
    HAS_RAG = True
except Exception as e:
    print(f"[RAG] 向量RAG未加载: {e}")
```

### 3c. 在build_llm_prompt()中使用
```python
# 替换原来的全文拼接
if HAS_RAG:
    # 用分类结果+峰位构建查询
    query = f"XRD {pred_label} {' '.join(str(p) for p in peak_positions[:5])}"
    rag_context = _rag.retrieve(query, top_k=5)
else:
    rag_context = load_rag_context()  # 降级
```

---

## 4. RAGEngine API

```python
class RAGEngine:
    def __init__(self, chunks_path, vectors_path):
        """加载知识库"""

    def retrieve(self, query: str, top_k: int = 5) -> str:
        """
        语义检索, 返回格式化字符串:
        [Ref.1] (来源, 类别, score=0.87): 段落文本...
        [Ref.2] (来源, 类别, score=0.82): 段落文本...
        """

    def retrieve_chunks(self, query: str, top_k: int = 5) -> list:
        """返回原始chunk字典列表(含score), 更灵活"""
```

---

## 5. DashScope嵌入API

- **URL**: `https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings`
- **Model**: `text-embedding-v3`
- **API Key**: 与千问VL共用 (环境变量`QWEN_VL_KEY`或内置默认)
- **调用**: `_embed_texts(["文本1", "文本2", ...])` → 返回向量列表
- **延迟**: ~200-500ms/次

---

## 6. 数值线特有的改动建议

### 6a. CoT五步推理 + 引用溯源
数值线的DeepSeek prompt也应升级为五步推理格式:
```
**步骤1 - 分类结果**: MLP判定为{label}, 置信度{conf}
**步骤2 - 峰位验证**: 检测到的峰位与[Ref.N]中的标准峰匹配
**步骤3 - 晶体结构**: 空间群、晶格参数, 引用[Ref.N]
**步骤4 - 性能分析**: 发光性能/应用方向, 引用[Ref.N]
**步骤5 - 可靠性评估**: OOD拒识、细分类一致性
```

### 6b. 前沿文献感知
在DeepSeek的system prompt中加入:
```
"你了解XRD+AI前沿: DiffractGPT, PXRDGen(96%匹配率), XtalNet。"
```

### 6c. 用户反馈
同视觉线，增加正确/需修正按钮, 存`logs/feedback.jsonl`。

---

## 7. 视觉线新增的10项前沿技术清单

数值线可酌情接入:

| # | 技术 | 视觉线状态 | 数值线建议 |
|---|------|-----------|-----------|
| T1 | 向量RAG | ✅已实现 | ✅建议接入(本文档) |
| T2 | SSE流式输出 | ✅已实现 | 可选(DeepSeek也支持stream) |
| T3 | 3D晶体可视化 | ✅已实现 | ✅建议接入(共用CIF文件) |
| T4 | 多模态融合 | ✅已实现 | N/A(数值线无图像) |
| T5 | CoT五步推理 | ✅已实现 | ✅建议接入 |
| T6 | 自一致性投票 | ✅已实现 | 可选(数值线有MLP仲裁) |
| T7 | 语音工具调用 | ✅已实现 | ✅建议接入(参考VOICE_INTERACTION_GUIDE.md) |
| T8 | 响应缓存 | ✅已实现 | ✅建议接入(用.raw文件名作key) |
| T9 | 用户反馈 | ✅已实现 | ✅建议接入 |
| T10 | 前沿文献感知 | ✅已实现 | ✅建议接入(prompt补充) |
