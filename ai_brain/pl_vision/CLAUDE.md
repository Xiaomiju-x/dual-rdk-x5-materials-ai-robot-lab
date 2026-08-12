# 光谱智慧实验室 — 视觉线（MVP 开发中）

> **状态**: 🟢 **AI 脑冻结于 2026-04-19** — 摄像头 → YOLO → Qwen-VL → R1 Agent + 25228 chunks RAG. 顶层最新状态见 [AI 脑公开导航](../../docs/modules/AI_BRAIN.md).
> **定位**: 对标 [xrd_vision](../xrd_vision/), 摄像头拍论文里的 PL 光谱图 → 识别 → 解读
> **部署**: X5 共用 IMX415 + M260C, 端口 8081 (不能和 xrd_vision 同时跑)
> **顶层总览**: [AI 脑公开导航](../../docs/modules/AI_BRAIN.md)
> **实验室闭环位置**: 研磨 → 烧制 → XRD → **光谱 (数值 + 视觉 两条线)** → 决策

---

## 系统架构 (Round 4 MVP)

```
IMX415 4K 摄像头 (或 PC USB webcam fallback)
    ↓ OpenCV 4K 采集 + MJPEG 推流
    ↓ YOLOv8n (ONNX, CPU) 检测 PL 光谱图区域 → 绿色 bbox 叠加
    ↓ 用户点"📸 冻结 + AI 分析"
    ↓ 裁剪最大 bbox
    ↓ Qwen-VL (qwen-vl-max) 客观描述 (不做物理)
    ↓ DeepSeek-R1 Agent ReAct 循环 (2 轮工具调用)
    ↓ query_rag_knowledge → spectrum_knowledge_shared (2462 篇论文)
    ↓ 输出 5 段结构化分析: 图像识别 / 核心信息 / 物理解释 / 文献对照 / 配方启示
    ↓ 浅色 monospace 思考链面板流式显示 (复用 Round 3 风格)
```

**与 xrd_vision 区别**:
- 共享 IMX415 硬件, 但 Agent 专攻 PL 图像分析
- 不用 BPU (Round 4 纯 ONNX CPU, 留 Round 5 做 BPU)
- 无语音交互 (Round 5)
- UI 精简 (~700 行 Flask, 不是 xrd_vision 的 3600 行)

---

## 目录结构

```
spectrum_vision/
├── CLAUDE.md                        ← 本文件
│
└── visual_line/                     ← 镜像 xrd_vision/visual_line/
    ├── deploy_spectrum_vision.py    ← ★ 主入口 Flask, 端口 8081
    ├── prepare_papers.py            ← 2462 PDFs → spectrum_knowledge_shared 向量化
    ├── rag_engine.py                ← RAG 引擎 (复制自 xrd_numerical, 指向共享池)
    │
    ├── train/                       ← YOLO 训练 pipeline (镜像 xrd_vision/train/)
    │   ├── generate_synthetic_data.py  ← 从 raw_figures 合成 800+200 样本
    │   ├── train_yolo.py               ← YOLOv8n, 50 epoch, 单类 pl_spectrum
    │   ├── dataset.yaml
    │   ├── export_onnx.py              ← best.pt → best.onnx (不导 BPU)
    │   └── runs/pl_detect/weights/best.onnx  ← 训练产物
    │
    ├── dataset/
    │   ├── raw_figures/             ← pick_training_papers.py 输出的候选 PL 图 (需人工审阅)
    │   ├── images/train/ val/       ← generate_synthetic_data.py 产物
    │   ├── labels/train/ val/
    │   └── README.md                ← 怎么审阅 raw_figures
    │
    ├── papers_for_yolo/             ← 被挑中的 2 篇训练源论文 (备份, 便于复现)
    └── bpu_export/                  ← 空, 预留 Round 5
```

**共享资源**:
- `crystal_data_shared/` — 历史晶体 CIF 缓存；公开缓存的来源与再分发边界见[数据来源说明](../../docs/data/PROVENANCE.md)
- `spectrum_knowledge_shared/` — 历史论文向量库；逐文档授权语料未随发布树分发，见[公开边界](../../docs/safety/PUBLICATION_BOUNDARY.md)
- `tools/pick_training_papers.py` — 历史训练论文筛选工具；其输入语料未公开分发，因此本发布树不提供该流水线

---

## 关键约束

1. **摄像头共用**: 和 xrd_vision 共用 IMX415, 一次只能跑一条线 (deploy 端口不同: xrd_vision 8080, spectrum_vision 8081)
2. **无 BPU**: 本轮 ONNX CPU 推理, X5 上也跑 ONNX (不启用 hobot_dnn)
3. **YOLO 单类**: `pl_spectrum`, 检测论文里的光谱图区域 (不细分发射/激发/对比谱)
4. **RAG 路径**: 优先共享 `spectrum_knowledge_shared/`, fallback 本地 `xrd_knowledge/`
5. **Qwen-VL 定位**: 只做客观描述 (图里看到什么), 物理解释交给 R1 Agent
6. **R1 Agent 必须调 RAG**: system prompt 强制要求至少 1 次 `query_rag_knowledge`

---

## 部署命令

```bash
# Step 1: PC 端一次性生成训练数据 + 训练 YOLO (不在 X5)
pip install ultralytics pdfplumber pymupdf

# 1a. 自动挑 2 篇论文 + 抽候选图 (~1 分钟)
cd xrd
python tools/pick_training_papers.py

# 1b. 人工审阅 spectrum_vision/visual_line/dataset/raw_figures/,
#     删掉非 PL 光谱的图 (化学结构 / 流程图 / 表格等), 留 >= 5 张真 PL 图

# 1c. 合成训练数据 (800+200) + 训练 YOLO (~30-60 分钟 CPU)
cd spectrum_vision/visual_line/train
python generate_synthetic_data.py
python train_yolo.py
python export_onnx.py

# Step 2: PC 端 2462 论文向量化 (~15-30 分钟, 和 Step 1 可并行)
cd xrd
python spectrum_vision/visual_line/prepare_papers.py

# Step 3: scp 到 X5 (同 xrd_vision 的模式)
scp spectrum_vision/visual_line/deploy_spectrum_vision.py rdk@x5:~/spec_vision/
scp spectrum_vision/visual_line/rag_engine.py rdk@x5:~/spec_vision/
scp spectrum_vision/visual_line/train/runs/pl_detect/weights/best.onnx \
    rdk@x5:~/spec_vision/pl_detect.onnx
scp -r spectrum_knowledge_shared rdk@x5:~/

# Step 4: X5 启动
ssh rdk@x5
cd ~/spec_vision
python3 deploy_spectrum_vision.py --port 8081
# 浏览器 http://<X5>:8081/
```

---

## 后续轮次

- **Round 5**: BPU 量化 + spectrum_vision / spectrum_numerical 的 Agent 统一升级为"工业级配方顾问"
- **Round 6**: 闭环集成 demo, 4 条线总入口
