# 光谱智慧实验室 — 数值线（MVP 开发中）

> **状态**: 🟢 **AI 脑冻结于 2026-04-19** — 数据格式已确认, parser / Cr/Ni/Cr+Ni 三 MLP / `/api/bpu_infer_80d` / web demo :5001 已实装. 顶层最新状态见 [AI 脑公开导航](../../docs/modules/AI_BRAIN.md).
> **定位**: 对标 [xrd_numerical](../xrd_numerical/)(数值线 XRD), 做法类似 — 原始谱图文件 → 峰提取 → 特征 → 分类 → Agent 解读
> **顶层总览**: [AI 脑公开导航](../../docs/modules/AI_BRAIN.md)
> **实验室闭环位置**: 研磨 → 烧制 → XRD → **光谱 (本线)** → 决策

---

## 已确认的数据格式

**类型**: PL (光致发光) 发射/激发扫描, 近红外 NIR (600-1650 nm)
**仪器**: Fluoromax / Horiba FluorEssence 系列
**文件**: `.csv` / `.txt` (同内容) + `.FS` (二进制) + `.wmf` (图片), 本轮只用 CSV/TXT

**CSV header 示例**:
```
Labels,0,
Type,Emission Scan,
Start,600.00,
Stop,1650.00,
Step,1.00,
Fixed/Offset,455.00,
Xaxis,Wavelength,
Yaxis,Counts,
```

**文件名编码**: `{浓度}{掺杂}-{波长}-{em|ex|pl}.csv`
- `0.002ni-455-em.csv` = 0.2% Ni 掺杂, 455nm 激发, 发射扫描
- `0.03cr-0.05Be-455-pl.csv` = Cr 0.03 + Be 0.05 共掺, 455nm 激发, 宽带 PL
- `狭缝2.5+2.5` 标注在父目录名里 (非文件名)

**目标材料**: `NaY2Ga2InGe2O12` (126 样品), `Y3ZnGa3GeO12` (279 样品), 都是 garnet 基 NIR 荧光粉宿主

**特殊子目录** (本轮跳过, 留给后续轮次):
- `QY/`: 量子产率测量, 双列格式 (样品 + 参考 KONGBAI)
- `TQ/`: 温度依赖测量, header 含 Temperature 字段

---

## 系统架构 (MVP Pipeline)

```
PL CSV 文件 (Fluoromax)
    ↓ src/parse_pl.py        (CSV header 解析 + 波长/强度二维数组)
    ↓ src/extract_peaks_pl.py (scipy.signal.find_peaks, PL 参数调优)
    ↓ src/build_features_pl.py (80D 特征: 30D 峰 + 40D 直方图 + 10D 统计)
    ↓ src/model.py MLP       (Cr / Ni / Cr+Ni 三分类)
    ↓ DeepSeek-R1 Agent      (物理解释 + 工艺建议)
    ↓ Flask web_demo_pl.py   (端口 5001, Canvas 绘谱 + 浅色思考链)
```

**与 xrd_numerical 的区别**:
- ✅ 纯 CPU + scipy + PyTorch, **可以在 PC 上直接跑**, 不依赖 BPU/相机/ALSA
- ❌ 本轮不做 BPU 量化 (留 Round 3.5)
- ❌ 本轮不做 Conformal/MC Dropout/XAI (简化)
- ❌ 本轮不做语音交互 (留 Round 5)
- ❌ 本轮在原工程中直接复用 `xrd_numerical/xrd_knowledge/` RAG（专属 spectrum knowledge 留 Round 4）；逐文档授权知识库未随发布树分发，见[公开边界](../../docs/safety/PUBLICATION_BOUNDARY.md)

---

## 目录结构

```
spectrum_numerical/
├── CLAUDE.md                      ← 本文件
│
├── NaY2Ga2InGe2O12/               ← 老师提供的原始数据 (126 CSV)
├── Y3ZnGa3GeO12/                  ← 老师提供的原始数据 (279 CSV)
├── 6-文献阅读管理/                  ← 老师的 2462 篇论文 (Round 4 向量化)
│
├── src/                           ← 核心模块
│   ├── parse_pl.py                ← Fluoromax CSV parser
│   ├── extract_peaks_pl.py        ← 峰提取 (复用 xrd_numerical extract_peaks)
│   ├── build_features_pl.py       ← 80D 特征构造
│   ├── label_from_path.py         ← 从文件名/路径抽标签
│   └── model.py                   ← MLP (复制自 xrd_numerical, 保持 BPU 算子约束)
│
├── scripts/                       ← 训练/数据脚本
│   ├── build_dataset.py           ← 扫目录 → X.npy/y.npy/labels.csv
│   ├── train.py                   ← MLP 训练
│   └── infer.py                   ← 单文件推理验证
│
├── data/                          ← 训练产物
│   ├── labels.csv
│   ├── X.npy / y.npy
│   └── norm_params.json
│
├── outputs/models/                ← 训练权重
│   └── pl_classifier.pt
│
├── rag_engine.py                  ← 复制自 xrd_numerical/rag_engine.py
└── web_demo_pl.py                 ← ★ 主入口 Flask, 端口 5001
```

---

## 关键约束

1. **BPU 算子**: 保留 Linear + ReLU + Dropout only (model.py 照搬 xrd_numerical), 为 Round 3.5 的量化打基础
2. **Classification 任务**: 掺杂离子三分类 — `cr` / `ni` / `cr_ni`, 从文件名正则自动打标
3. **本轮只处理 emission 扫描**: 跳过 QY/TQ/excitation/fitted/KONGBAI
4. **样本量目标**: 200-300 条 emission CSV 进入训练集
5. **Agent 定位**: 不仅识别掺杂, 还要给物理解释 (ZPL / vibronic sideband / 交叉弛豫) + 浓度工艺建议
6. **端口**: Flask 5001 (避开 xrd_numerical 5000 和 xrd_vision 8080)

---

## 部署命令

```bash
# PC 本地跑 (不需要 X5)
cd spectrum_numerical
python scripts/build_dataset.py    # 一次性构建数据集
python scripts/train.py            # 训练 MLP
python web_demo_pl.py --port 5001  # 启动 web demo
# 浏览器打开 http://localhost:5001
```

---

## 已踩过的坑 / 注意事项

1. **编码混合**: 老师的 CSV 有 UTF-8 和 GBK 两种, parser 必须做 fallback
2. **header 行数不固定**: 有的样品 header 17 行, 有的 21 行, 必须用 "空行 / 首字段可转 float" 作为 header→data 分界
3. **scan_type 识别**: header 里 `Type,Emission Scan,` / `Type,Excitation Scan,` / 没 Type 字段 (纯 PL), 三种都要 handle
4. **跳过 QY 双列**: QY 文件有两列数据 (sample + reference KONGBAI), 本轮直接跳过 (parser 返回 None)
5. **FWHM 单位**: xrd_numerical 里 FWHM 是度, PL 里 FWHM 是 nm, 代码上是一样的 (x 轴单位抽象), 语义上注意

---

## 后续轮次的事项

- **Round 3.5**: pl_classifier 量化 + BPU 部署 X5
- **Round 4**: 光谱视觉线 + 2462 篇论文向量化 (spectrum_knowledge)
- **Round 5**: Agent 工业级配方顾问升级 (跨线 XRD + PL 数据融合)
