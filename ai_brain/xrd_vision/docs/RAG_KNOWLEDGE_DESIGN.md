# RAG知识库设计规范 — 数值线与视觉线共享

## 给数值线 Claude Code 的指令

数值线（.raw分析流水线）在设计RAG知识库时，必须遵循以下规范，因为**视觉线会复用同一个知识库**。

---

## 知识库路径

```
/home/rdk/xrd1/xrd_knowledge/       # RDK X5 部署路径
├── papers/                              # ★ 论文知识文档 (两条线共用)
│   ├── paper1_sygo.txt                  # 论文1
│   ├── paper2_ycas.txt                  # 论文2
│   └── ...                              # 以后新增的论文
└── standards/                           # 标准卡片信息 (可选扩展)
    └── common_phases.txt
```

Windows开发路径对应:
```
d:\xrd\yolo_xrd_detect\xrd_knowledge\   # 视觉线脚本同级目录
```

---

## 论文文档格式 (必须严格遵守)

每个论文一个txt文件，放在 `papers/` 目录下，格式如下:

```
【论文标题】
<论文中文简称，如 SYGO体系 / YCAS石榴石体系>

材料: <化学式全称>
期刊: <期刊名 卷号 (年份) 页码>
晶系: <晶系>, 空间群<空间群符号>
参考卡: <ICSD#/PDF#编号>

结构:
<晶体结构描述: 多面体构成、离子占位>

发光机制:
<激发/发射波长、能量转移路径、量子效率>

Rietveld精修:
<Rwp, Rp, χ²等关键参数>

应用方向:
<实际应用场景>

XRD图视觉特征:
<图谱外观描述: 峰形对称性、峰数量密度、特征角度范围>
<这一段非常重要！视觉线的千问VL会根据这些描述来匹配图像>
```

### 示例 — paper1_sygo.txt

```
【论文1 - SYGO体系】

材料: Sr₃YGa₂O₇.₅: xBi³⁺, yEu³⁺ (简称SYGO)
期刊: Journal of Luminescence 281 (2025) 121192
晶系: 单斜晶系, 空间群C2
参考卡: ICSD#47510

结构:
层状钙钛矿结构，Bi³⁺占据Sr和Y位点

发光机制:
Bi³⁺发射蓝光(451nm) → Bi³⁺→Eu³⁺能量转移 → 蓝到红可调发光

Rietveld精修:
Rwp=8.22%, Rp=5.73%, χ²=4.392

应用方向:
UV激发白光LED荧光粉, CRI=85.5, CCT=3537K

XRD图视觉特征:
单斜晶系峰形复杂，多峰密集且不对称，2θ=20-60°范围内峰数量多
与立方晶系石榴石的尖锐对称峰形成明显对比
Fig.1含(a)不同Bi³⁺浓度系列 (b)Bi/Eu共掺系列
```

---

## 视觉线如何使用知识库

视觉线的 `deploy_xrd_system.py` 中有 `load_rag_context()` 函数:

```python
def load_rag_context():
    """加载RAG知识库 - 读取 xrd_knowledge/papers/ 下所有文档"""
    rag_dir = os.path.join(script_dir, "xrd_knowledge", "papers")
    if os.path.isdir(rag_dir):
        # 读取所有 .txt/.json/.md 文件，拼接为上下文
        for fname in sorted(os.listdir(rag_dir)):
            parts.append(open(fname).read())
        return "\n\n".join(parts)
    else:
        return _BUILTIN_PAPERS  # fallback到硬编码的两篇论文信息
```

这些文档会被**整体注入**到千问VL的prompt中，千问VL会结合图像和文本来判断属于哪篇论文。

---

## 数值线如何使用同一个知识库

数值线的MLP分类出 `garnet`/`non_garnet` 后，可以:

```python
# 方式1: 根据分类结果精确加载对应论文
def get_paper_context(classification):
    if classification == "garnet":
        return read_file("xrd_knowledge/papers/paper2_ycas.txt")
    else:
        return read_file("xrd_knowledge/papers/paper1_sygo.txt")

# 方式2: 全部加载让LLM结合分类结果选择 (与视觉线一致)
def get_all_context():
    return load_all_files("xrd_knowledge/papers/")
```

---

## 关键约束

1. **文件编码**: UTF-8, 不要用GBK
2. **文件名**: 用英文+数字，如 `paper3_xxx.txt`，按序号排列
3. **"XRD图视觉特征"段必须写**: 这是视觉线区分论文的关键信息
4. **新增论文**: 只需在 `papers/` 目录下新增一个txt文件，两条线自动生效，无需改代码
5. **部署同步**: 数值线建好知识库后，scp整个目录到X5:
   ```bash
   scp -r xrd_knowledge/ rdk@x5:/home/rdk/xrd1/
   ```

---

## 扩展方向 (论文多了之后)

当论文数量超过5-10篇，全部注入prompt会太长，届时可以:
1. 数值线: MLP分类结果直接索引对应论文，不需要RAG
2. 视觉线: 加embedding向量检索，先用CLIP/千问embedding提取图像特征，检索最相关的2-3篇论文再注入prompt
3. 这是后续优化，现阶段全部注入即可
