# 基于双 RDK X5 异构协同的材料合成 AI 预测与多机具身实验助理机器人｜XRD 视觉线上下文

> **状态: AI 脑冻结于 2026-04-19**, 不再做大改动. 顶层最新状态见 [AI 脑公开导航](../../docs/modules/AI_BRAIN.md). 本文档保留历史架构说明 (v4.0 描述), 部分内容过时, 但视觉线工作流和 R1 Agent 工具链仍准确.

## 项目背景

**比赛**: 2026 全国大学生嵌入式芯片与系统设计竞赛 · 芯片应用赛道 · 地瓜机器人赛题 (选题一: RDK X5)
**选手**: 周灵轩，重庆邮电大学本科生，课题组两篇论文共同作者
**硬件**: RDK X5开发板 (BPU: Bayes-e, 10 TOPS) + IMX415 USB摄像头 (4K, 94.5°, 自动对焦, 无畸变) + M260C智能音箱(串口+麦克风阵列+扬声器)
**比赛现场可联网**
**赛题要求**: 视觉感知(必选) + BPU加速(鼓励) + 完整任务闭环 + 代码开源到NodeHub

---

## 系统架构 v4.0 (双BPU + AI Agent + 197篇论文RAG)

```
┌─────────────────── 视觉线 (deploy_xrd_system.py, 端口8080) ───────────────────┐
│                                                                                 │
│  IMX415 4K → YOLO检测(BPU) → 千问VL(视觉感知) → DeepSeek-R1 Agent(推理+工具链) │
│                                                                                 │
│  工具链: RAG知识库(197篇) + COD数据库验证 + XRD峰位模拟器 + 实验建议            │
│                                                                                 │
│  语音交互: M260C → VAD → 百度ASR → Agent跟进/语音指令 → TTS播报               │
│                                                                                 │
│  前端特色: 3D晶体 + 知识图谱 + 声化 + 材料指纹 + QR分享 + Demo巡览            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────── 数值线 (web_demo.py, 端口5000) ────────────────────────────┐
│                                                                                 │
│  .raw文件 → 峰位提取 → 45维特征 → MLP分类(BPU) → 峰位匹配 → DeepSeek解读      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**评委亮点**: 双BPU模型(YOLO+MLP) + AI Agent(千问VL+DeepSeek-R1+5个工具) + 197篇论文向量RAG + COD数据库验证 + 3D晶体可视化 + XRD声化 + 语音交互闭环

---

## 文件结构

```
d:\xrd\
├── CLAUDE.md                          # 本文件 (项目上下文)
├── competition_pdf.pdf                # 竞赛题目PDF
├── web_demo.py                        # ★ 数值线Web应用 (Flask, 端口5000)
│
├── visual_line\                       # ★ 视觉线项目
│   ├── deploy_xrd_system.py           # ★ 视觉线主脚本 (~3500行)
│   ├── rag_engine.py                  # 向量RAG引擎 (DashScope embedding)
│   ├── prepare_papers.py              # 论文PDF→切块→向量化 (Windows运行)
│   ├── crystal_data\                  # CIF晶体结构文件
│   │   ├── SYGO.cif                   # 单斜C2 (88原子)
│   │   └── YCAS.cif                   # 立方Ia-3d (120原子)
│   ├── xrd_knowledge\                 # RAG知识库
│   │   ├── papers\                    # 16篇已处理txt论文
│   │   ├── papers_new\                # 180+篇PDF原始论文 (Fe3+/Cr3+/Ni2+/综述/其他)
│   │   ├── embeddings\                # 预计算向量
│   │   │   ├── chunks.json            # 2255个文本段落
│   │   │   └── vectors.npy            # 2255×1024嵌入矩阵
│   │   └── standards\                 # 标准相参考数据
│   ├── train\                         # 训练相关脚本
│   │   ├── train_yolo.py
│   │   ├── generate_synthetic_data.py
│   │   ├── export_for_bpu.py
│   │   ├── prepare_calibration_yolo.py
│   │   └── dataset.yaml
│   ├── bpu_export\                    # BPU转换输出
│   ├── dataset\                       # 合成训练数据 (800+200)
│   └── runs\                          # 训练结果
│
├── docs\                              # 文档集
│   ├── RAG_KNOWLEDGE_DESIGN.md        # RAG设计文档
│   ├── RAG_REFERENCE.md               # 数值线RAG接入指南
│   ├── VISUAL_LINE_FEATURES_GUIDE.md  # AI Agent+知识图谱+3D晶体移植指南
│   └── VOICE_INTERACTION_GUIDE.md     # 语音交互移植指南
│
└── archive\                           # 已废弃项目归档
    ├── paper_xrd_project\             # MobileNetV2 (toolchain不兼容, 已废弃)
    ├── bpu_convert\                   # 旧BPU转换目录
    └── capture_calibration_imx415.py  # 旧校准脚本
```

**RDK X5上** (部署位置):
```
/home/rdk/xrd1/
├── yolo_xrd_detect.bin             # YOLO检测BPU模型
├── xrd_mlp_classify.bin            # MLP分类BPU模型
├── deploy_xrd_system.py            # 视觉线脚本 (端口8080)
├── rag_engine.py                   # RAG引擎
├── web_demo.py                     # 数值线脚本 (端口5000)
├── infer_with_llm.py               # 数值线推理模块
├── crystal_data/                   # CIF文件
├── xrd_knowledge/                  # RAG知识库+向量
│   ├── papers/
│   ├── embeddings/
│   └── standards/
└── data/raw_files/                 # .raw样本文件
```

---

## 视觉线已实现功能 (v4.0)

### 核心AI
- **YOLO目标检测**: YOLOv8n on BPU INT8, 640×640输入
- **AI Agent**: 千问VL(视觉感知) + DeepSeek-R1(推理+工具调用, ReAct循环)
- **5个Agent工具**: query_rag_knowledge / match_pdf_card / query_crystal_database(COD) / compute_theoretical_xrd(Digital Twin) / suggest_next_experiment
- **向量RAG**: 197篇论文, 2255段落, DashScope text-embedding-v3语义检索

### 前端展示
- **3D晶体结构**: 3Dmol.js球棍模型, SYGO(88原子)+YCAS(120原子)
- **知识图谱**: 分组卡片式+动画(脉冲/弹跳/旋转/流动线)
- **Pipeline瀑布图**: 6段计时(预处理/BPU/后处理/裁剪/千问VL/DeepSeek-R1)
- **XRD声化**: WebAudio峰位→音高, 强度→音量
- **材料指纹**: Canvas确定性生成(8角蓝色=YCAS, 2角橙色=SYGO)
- **QR码分享**: 评委扫码查看完整报告
- **引导式Demo**: driver.js 9步高亮巡览

### 语音交互
- **M260C麦克风阵列**: XFMDPV0018自动检测, VAD(RMS>800)
- **百度ASR+TTS**: 语音识别+播报(默认关闭语音输入,按钮开启)
- **语音工具调用**: "保存报告"/"对比上次"/"重新分析"/"重置"
- **苏格拉底教学模式**: 按钮切换,AI引导式提问

### 3D 晶体候选 Agent (v4.1, 进行中)
> 解决 v4.0 硬编码 SYGO/YCAS 两个 CIF + 3Dmol.js 对称性解析不准的问题

- **服务端 CIF 预处理**: pymatgen 在 PC 端把 CIF 展开成 P1 对称 + 精确超胞, 写成新 CIF 丢给 3Dmol.js 只做渲染(不再让 3Dmol.js 自己解析 `doAssembly`)
- **统一 CIF 池**: 所有 CIF 集中到 `xrd/crystal_data_shared/{raw,processed}/`, 来源包括 Materials Project (新 key `j8faPrv105Z8d8bpQntzpa5np7lnVS1Q`) + FindIt 2011 + 老师提供
- **候选结构 Agent**: `crystal_agent.py` 抽出主逻辑(不塞进 3600 行主文件)
  - `generate_candidates(classification, top_k=3)`: 按 Qwen-VL 分类结果从 MP 池拉 K 个候选 CIF
  - `rank_candidates(candidates, exp_peaks)`: pymatgen `XRDCalculator` 算理论谱 → 和实测对比 → R1 推理选优
- **前端候选对比视图**: 主 3Dmol viewer 下方 K 个小 viewer 并排, 最优候选高亮, 其他灰掉, R1 推理链流式输出到现有 `streamText`
- **标准 Jmol 调色板**: 抽出 `ELEMENT_COLORS`/`ELEMENT_RADII` 常量, 按原子序数查表, 不再一个个 `setStyle({elem:'Y'},...)`
- **VESTA 作为金标准**: 每个新 CIF 都要用 VESTA 核对渲染效果一致才算过

### 系统工程
- **启动自检**: 7项子系统自动检测(摄像头/BPU/RAG/API/语音)
- **SHA-256哈希链**: 每条分析记录防篡改指纹
- **响应缓存**: MD5 key + LRU 20条
- **分析时间线**: 实时事件流
- **硬件仪表盘**: BPU温度/CPU/RAM实时显示
- **图像变化检测**: bbox偏移>20%自动重分析
- **CV峰检测叠加**: 红色竖线标注视频中的衍射峰

---

## API配置

### 千问VL (视觉线Stage 1)
- API URL: `https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions`
- Model: `qwen-vl-max`

### DeepSeek-R1 (视觉线Stage 2 Agent)
- API URL: `https://api.deepseek.com/v1/chat/completions`
- Model: `deepseek-reasoner`
- Key: `<API_KEY_FROM_ENVIRONMENT>`

### DashScope Embedding (RAG)
- API URL: `https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings`
- Model: `text-embedding-v3`

### 百度语音
- App ID: `7604178`

---

## ALSA设备自动检测

USB枚举顺序每次重启可能变化,系统启动时自动检测:
- **麦克风**: 搜索`XFMDPV`/`XFM-DP`关键词 (M260C麦克风阵列)
- **扬声器**: 搜索`USB Audio`排除`Camera`和`ES8326` (M260C扬声器)
- **日志**: 启动时打印`[ALSA] ★ 最终选择: 扬声器=plughw:X,0, 麦克风=plughw:Y,0`

---

## 已踩过的坑

1. **校准数据格式**: bin存原始像素(0~255 float32),不提前归一化
2. **量化方式**: 用`max`不用`kl`
3. **MobileNetV2废弃**: toolchain v1.2.8与runtime v1.3.6不兼容 → 替换为YOLO
4. **MLP输入是45维特征**: 不是图像!
5. **TTS防自触发**: tts_playing标志 + RLock(可重入锁,防嵌套调用死锁)
6. **DeepSeek-R1不能看图**: 材料判定必须由千问VL做(它能看图),R1只做推理
7. **COD数据库从中国访问慢**: 必须预缓存,在线查询作为备选
8. **ALSA设备号变化**: USB重启后枚举顺序变,必须自动检测
9. **VAD缓冲区堆积**: 禁用时必须继续read丢弃数据,不能sleep跳过
10. **摄像头误识别为麦克风**: capture设备中排除Camera/4K关键词
11. **3Dmol.js 对称性解析不准**: 对 Ia-3d/C2 等复杂空间群, `doAssembly:true` 会漏原子或重复, `replicateUnitCell` 只是平移不是晶体学扩胞 → v4.1 改成 PC 端 pymatgen 预处理为 P1 扩胞后的 CIF, 3Dmol.js 只做纯渲染

---

## 部署命令

```bash
# 视觉线
scp d:/xrd/visual_line/deploy_xrd_system.py rdk@x5:~/xrd1/
scp d:/xrd/visual_line/rag_engine.py rdk@x5:~/xrd1/
scp -r d:/xrd/visual_line/crystal_data rdk@x5:~/xrd1/
scp -r d:/xrd/visual_line/xrd_knowledge/embeddings rdk@x5:~/xrd1/xrd_knowledge/
ssh rdk@x5 "cd ~/xrd1 && python3 deploy_xrd_system.py --port 8080"

# 数值线
scp d:/xrd/web_demo.py rdk@x5:~/xrd1/
ssh rdk@x5 "cd ~/xrd1 && python3 web_demo.py --port 5000"
```
