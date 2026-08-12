# 基于双 RDK X5 异构协同的材料合成 AI 预测与多机具身实验助理机器人 — 2026全国嵌入式竞赛

> 基于RDK X5 BPU的XRD衍射图谱智能分析系统

## 项目亮点

- **双BPU模型**: YOLOv8n目标检测 + MLP级联分类, 均为INT8量化
- **AI科学家Agent**: 千问VL(视觉感知) + DeepSeek-R1(推理+5个工具链)
- **197篇论文向量RAG**: DashScope text-embedding-v3语义检索
- **COD国际数据库交叉验证**: 在线查询+本地缓存
- **XRD数字孪生**: Bragg方程计算理论峰位 vs 实验对比
- **3D晶体结构可视化**: 3Dmol.js球棍模型(SYGO 88原子 / YCAS 120原子)
- **XRD光谱声化**: WebAudio将衍射峰转化为声音
- **语音交互全链路**: M260C麦克风阵列 + VAD + ASR + TTS + 语音工具调用
- **苏格拉底教学模式**: AI引导式提问
- **SHA-256防篡改哈希链**: 科学数据完整性保障

## 硬件

| 组件 | 型号 | 说明 |
|------|------|------|
| 开发板 | RDK X5 | BPU Bayes-e 10 TOPS, ARM aarch64 |
| 摄像头 | IMX415 | 4K USB, 94.5°视角, 自动对焦 |
| 音箱 | M260C | USB串口, 麦克风阵列+扬声器 |

## 目录结构

```
├── CLAUDE.md                  # 项目全局上下文
├── competition_pdf.pdf        # 竞赛题目
├── web_demo.py                # 数值线 (端口5000)
├── visual_line/               # 视觉线
│   ├── deploy_xrd_system.py   # 主脚本 (端口8080)
│   ├── rag_engine.py          # 向量RAG引擎
│   ├── prepare_papers.py      # 论文预处理
│   ├── crystal_data/          # CIF晶体结构
│   ├── xrd_knowledge/         # 知识库+向量
│   ├── train/                 # 训练脚本
│   ├── bpu_export/            # BPU转换
│   └── dataset/               # 训练数据
├── docs/                      # 文档
│   ├── RAG_REFERENCE.md       # RAG接入指南
│   ├── VISUAL_LINE_FEATURES_GUIDE.md
│   └── VOICE_INTERACTION_GUIDE.md
└── archive/                   # 已废弃项目
```

## 快速部署

```bash
# 视觉线
scp visual_line/deploy_xrd_system.py rdk@x5:~/xrd1/
scp visual_line/rag_engine.py rdk@x5:~/xrd1/
ssh rdk@x5 "cd ~/xrd1 && python3 deploy_xrd_system.py --port 8080"

# 数值线
scp web_demo.py rdk@x5:~/xrd1/
ssh rdk@x5 "cd ~/xrd1 && python3 web_demo.py --port 5000"
```

## 技术栈

| 层级 | 技术 |
|------|------|
| BPU推理 | hobot_dnn (YOLOv8n + MLP INT8) |
| 视觉LLM | 千问VL-Max (DashScope) |
| 推理Agent | DeepSeek-R1 (thinking + tool-calling) |
| RAG | DashScope text-embedding-v3 + numpy cosine |
| 语音 | 百度ASR/TTS + espeak-ng + M260C |
| 3D可视化 | 3Dmol.js + Canvas API |
| Web框架 | Flask + SSE + MJPEG |

## 选手

**周灵轩** — 重庆邮电大学光电学院
2026全国大学生嵌入式芯片与系统设计竞赛 · 地瓜机器人赛道
