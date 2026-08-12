# 基于双 RDK X5 异构协同的材料合成 AI 预测与多机具身实验助理机器人 — 数值线 Claude Code项目指南

> **状态: AI 脑冻结于 2026-04-19**, 不再做大改动. 顶层最新状态见 [AI 脑公开导航](../../docs/modules/AI_BRAIN.md).

## 项目定位

> 首个在嵌入式AI芯片上运行的XRD智能分析系统——用物理定律验证判断，用拓扑数学检测异常，用AI Agent规划实验

**比赛**: 2026全国大学生嵌入式芯片与系统设计竞赛 - 地瓜机器人赛道(选题一: RDK X5)
**硬件**: RDK X5 (BPU Bayes-e 10TOPS, ARM Cortex-A55, 8GB RAM) + M260C智能音箱
**部署**: web_demo.py (Flask, 端口8080)

---

## 系统架构 (10阶段Pipeline)

```
.raw文件 → 解析 → 峰提取 → 190D特征 → BPU级联分类 → 峰匹配(17标准相)
    → RAG检索(2255段落) → AI Agent(R1+Tools) → 语音播报 → 实验建议
```

### 科学分析层
- **Scherrer微晶尺寸**: D=Kλ/(βcosθ), 从FWHM直接估算
- **理论图谱正演**: pymatgen CIF→连续谱, 与实测叠加, 计算Rwp残差
- **TDA持久同调**: 纯numpy H₀ Vietoris-Rips, 首创拓扑XRD分类
- **系统消光验证**: I/F/C/A/P晶格消光规则自动验证空间群
- **Nelson-Riley精修**: 外推法消除系统误差, 精确晶格常数

### AI决策层
- **BPU INT8分类**: 两级级联(garnet/non_garnet + 6类细分), <1ms确定性延迟
- **Conformal Prediction**: 95%覆盖率数学保证的预测集
- **MC Dropout**: Bayesian不确定性, 20次随机前向
- **XAI归因**: 前向差分法, 190D→2θ位置映射
- **OOD拒识**: 置信度+归一化熵双阈值

### 知识增强层
- **向量RAG**: DashScope text-embedding-v3 主 + 本地TF-IDF 备, 2255段落/197篇论文
- **AI Agent**: DeepSeek-R1推理模型, ReAct循环, 3工具(RAG/峰匹配/实验建议)
- **知识图谱**: 49实体+91关系, 8种节点类型, CSS动画
- **3D晶体**: 3Dmol.js球棍模型+晶胞框+自动旋转

### 交互层
- **语音交互**: VAD(RMS>800) + 百度ASR + 百度TTS/espeak-ng + ALSA自动检测
- **Web UI**: 2400行单文件Flask, 卡片流程图风格, Canvas谱图, PWA离线, 暗色模式
- **QR码分享**: 扫码查看独立报告页面

---

## 目录结构

```
xrd_numerical_pipeline/
├── CLAUDE.md                  ← 本文件
├── README.md                  ← 项目说明(NodeHub提交用)
├── config.yaml                ← 全局配置
├── requirements.txt           ← Python依赖
│
├── web_demo.py                ← ★ 主入口: Flask Web应用 (2400行)
├── rag_engine.py              ← RAG检索引擎 (DashScope+TF-IDF双模式)
├── knowledge_base.py          ← 知识库加载接口
├── demo.py                    ← 终端Demo
├── benchmark_bpu.py           ← BPU性能基准测试
│
├── simulated_patterns.json    ← 预计算理论XRD图谱 (YCAS+SYGO)
├── xrd-demo.service           ← systemd服务文件
├── setup_x5.sh                ← X5一键部署(systemd安装)
├── deploy_to_x5.sh            ← ★ PC→X5完整部署脚本
│
├── src/                       ← 核心处理模块
│   ├── parse_raw.py           ← Bruker .raw二进制解析 (v1-v4)
│   ├── extract_peaks.py       ← 峰位提取 (scipy.signal)
│   ├── build_features.py      ← 190D特征构造 (45D峰+140D直方图+5D统计)
│   ├── model.py               ← MLP模型定义 (PyTorch)
│   ├── peak_matcher.py        ← 晶体学PDF卡片匹配 (17标准相)
│   └── utils.py               ← 工具函数
│
├── scripts/                   ← 训练/评估/数据脚本
│   ├── train.py               ← MLP训练
│   ├── evaluate.py            ← 评估+混淆矩阵
│   ├── build_dataset.py       ← 批量构建数据集
│   ├── export_onnx.py         ← ONNX导出
│   ├── calibrate_conformal.py ← Conformal Prediction校准
│   ├── generate_simulated.py  ← pymatgen理论图谱生成 (仅PC)
│   ├── download_mp_data.py    ← Materials Project数据下载
│   ├── explore_raw.py         ← 交互式峰参数调试
│   └── infer.py               ← 单文件推理验证
│
├── bpu/                       ← BPU部署相关
│   ├── infer_with_llm.py      ← ★ 推理引擎 (1800行, 含全部科学分析函数)
│   ├── infer_bpu.py           ← BPU推理封装
│   ├── prepare_calibration.py ← 校准数据生成
│   ├── config_bpu.yaml        ← 粗分类BPU配置
│   ├── config_bpu_fine.yaml   ← 细分类BPU配置
│   ├── conformal_params.json  ← Conformal校准参数
│   ├── convert_workspace/     ← BPU转换产物 (粗分类)
│   │   └── model_output/xrd_mlp_classify.bin
│   ├── convert_workspace_fine/← BPU转换产物 (细分类)
│   │   └── model_output_fine/xrd_mlp_fine.bin
│   └── x5_deploy/
│       └── infer_bpu.py       ← X5独有BPU推理入口
│
├── data/                      ← 数据
│   ├── labels.csv             ← 标注 (材料名/掺杂/类别)
│   ├── reference_peaks.json   ← 17标准晶相峰位
│   ├── raw_files/             ← Bruker .raw样品 (~11个)
│   └── mp_downloads/          ← Materials Project合成数据 (100+)
│
├── outputs/                   ← 训练输出
│   ├── conformal_params.json  ← Conformal校准参数
│   ├── features/              ← 粗分类特征 (X.npy, y.npy, norm_params.json)
│   ├── features_fine/         ← 细分类特征
│   ├── models/                ← ONNX模型 + 训练信息
│   └── plots/                 ← 可视化图片
│
├── rag/                       ← RAG知识库 (16篇论文txt)
│   └── xrd_knowledge/
│       ├── papers/            ← paper01-16 结构化论文摘要
│       └── standards/         ← common_phases.txt
│
├── xrd_knowledge/             ← 向量化知识库
│   └── embeddings/
│       ├── chunks.json        ← 2255段落
│       └── vectors.npy        ← 1024维嵌入向量
│
├── static/                    ← Web静态资源
│   ├── manifest.json          ← PWA清单
│   └── sw.js                  ← Service Worker
│
└── docs/                      ← 项目文档
    ├── RAG_KNOWLEDGE_DESIGN.md
    ├── RAG_REFERENCE.md
    ├── VOICE_INTERACTION_GUIDE.md
    ├── VISUAL_LINE_FEATURES_GUIDE.md
    └── demo_script.md
```

---

## 关键约束

1. **BPU算子**: 只用Linear+ReLU+Dropout, march=bayes-e, calibration_type=kl
2. **ALSA设备**: 启动时自动检测M260C(XFMDPV)麦克风和USB Audio扬声器
3. **三级降级**: Agent(R1) → DeepSeek → 离线模板
4. **RAG双模式**: DashScope在线 → TF-IDF离线(自动切换)
5. **TTS防自触发**: 播放时设tts_playing=True, VAD跳过该期间

---

## 部署命令

```bash
# PC→X5完整部署
bash deploy_to_x5.sh 198.51.100.103

# X5上启动
python3 web_demo.py --port 8080

# systemd自启动
bash setup_x5.sh
```
