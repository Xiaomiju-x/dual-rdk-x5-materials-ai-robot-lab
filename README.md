# 基于双 RDK X5 异构协同的材料合成 AI 预测与多机具身实验助理机器人｜2026 全国大学生嵌入式芯片与系统设计竞赛·芯片应用赛道·地瓜机器人赛题｜西南赛区一等奖·全国总决赛二等奖

> **2026 全国大学生嵌入式芯片与系统设计竞赛 · 芯片应用赛道 · 地瓜机器人赛题 · 西南赛区一等奖 · 全国总决赛二等奖**
>
> 参赛队伍：**荧光具身智研**。西南赛区一等奖、全国总决赛二等奖均为队伍确认结果；组委会官方获奖来源待补。

[![项目最新硬件：移动实验助理与双机械臂工作站](assets/media/hero/project-hardware-hero.webp)](docs/gallery.md)

两台 RDK X5 分担材料 AI 与具身感知计算，把材料候选、XRD/PL 分析、移动实验助理、双机械臂、STM32F407 执行层和只读证据门户连成一个可追溯且权限隔离的实验系统。

[![CI](https://github.com/Xiaomiju-x/dual-rdk-x5-materials-ai-robot-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Xiaomiju-x/dual-rdk-x5-materials-ai-robot-lab/actions/workflows/ci.yml)
[![CodeQL](https://github.com/Xiaomiju-x/dual-rdk-x5-materials-ai-robot-lab/actions/workflows/codeql.yml/badge.svg)](https://github.com/Xiaomiju-x/dual-rdk-x5-materials-ai-robot-lab/actions/workflows/codeql.yml)
[![Gitleaks](https://github.com/Xiaomiju-x/dual-rdk-x5-materials-ai-robot-lab/actions/workflows/gitleaks.yml/badge.svg)](https://github.com/Xiaomiju-x/dual-rdk-x5-materials-ai-robot-lab/actions/workflows/gitleaks.yml)
[![Latest release](https://img.shields.io/github/v/release/Xiaomiju-x/dual-rdk-x5-materials-ai-robot-lab?display_name=tag&sort=semver)](https://github.com/Xiaomiju-x/dual-rdk-x5-materials-ai-robot-lab/releases/latest)
[![License](https://img.shields.io/github/license/Xiaomiju-x/dual-rdk-x5-materials-ai-robot-lab?label=license)](LICENSE)
[![Platform: RDK X5](https://img.shields.io/badge/Edge-RDK%20X5-orange.svg)](docs/architecture/SYSTEM_ARCHITECTURE.md)
[![Safety: tiered](https://img.shields.io/badge/Safety-Tier%200--4-red.svg)](docs/safety/PHYSICAL_SAFETY.md)

[English](README_en.md) · [文档中心](docs/README.md) · [安全离线开始](docs/getting-started/QUICKSTART_OFFLINE.md) · [证据索引](docs/evidence/EVIDENCE_INDEX.md) · [已知限制](docs/evaluation/KNOWN_LIMITATIONS.md)

## 三段实机演示

点击封面即可直接播放仓库内 MP4；完整时间码、SHA-256、处理方式和真实性边界见 [媒体画廊](docs/gallery.md) 与 [`MEDIA_PROVENANCE.yml`](assets/media/MEDIA_PROVENANCE.yml)。三段视频均为静音短片，不把固定工装、人工准备或单次界面读数扩展成通用自主能力。

| 材料 AI：XRD 视觉分析 | 具身助理：协同实验流程 | 双机械臂：完整固定工位流程 |
| --- | --- | --- |
| [![播放材料 AI 与 XRD 分析动态预览](assets/media/previews/dashboard-xrd-pipeline.gif)](assets/media/videos/dashboard-xrd-pipeline.mp4) | [![播放具身实验助理动态预览](assets/media/previews/embodied-assisted-workflow.gif)](assets/media/videos/embodied-assisted-workflow.mp4) | [![播放双机械臂完整流程动态预览](assets/media/previews/dual-arm-complete-hardware-demo.gif)](assets/media/videos/dual-arm-complete-hardware-demo.mp4) |
| [播放材料 AI MP4](assets/media/videos/dashboard-xrd-pipeline.mp4)：平板界面和项目硬件的一次演示记录，界面数值只对应本次片段。 | [播放具身助理 MP4](assets/media/videos/embodied-assisted-workflow.mp4)：固定流程中的瓶体工装、升降与协同动作，边界以画廊说明为准。 | [播放双机械臂完整 MP4](assets/media/videos/dual-arm-complete-hardware-demo.mp4)：固定工装上的实体机械臂流程，不等同于学习策略或任意任务泛化。 |

## 竞赛与奖项

项目参加 **2026 全国大学生嵌入式芯片与系统设计竞赛 · 芯片应用赛道 · 地瓜机器人赛题**，队伍“荧光具身智研”。奖项唯一权威数据源是 [`docs/competition/award_status.yaml`](docs/competition/award_status.yaml)；官方来源补齐时只更新这一处，再由脚本生成展示文本。

<!-- AWARD_STATUS:START -->
| 阶段 | 当前状态 | 事实边界 |
| --- | --- | --- |
| 西南赛区 | 一等奖 | `team_confirmed`：队伍确认，官方获奖来源待补 |
| 全国总决赛 | 二等奖 | `team_confirmed`：队伍确认，组委会官方获奖来源待补 |
<!-- AWARD_STATUS:END -->

[竞赛说明与更新规则](docs/competition/AWARDS.md) · [官方与公开来源](docs/competition/OFFICIAL_SOURCES.md)

## 项目全景

这是一个面向半导体光电子器件与先进封装功能材料的边缘材料智能与机器人实验平台。近红外荧光粉是已完成真实验证的材料载体；仓库提供源码、配置示例、固定输入、验收回执、演示媒体与公开边界，而不是只展示比赛 PPT。

项目把“算法演示”拆成五个权责清晰的工程模块：

- **AI 脑**：材料预测、证据约束推理、XRD/PL 视觉与数值分析、按需模型编排。
- **具身脑**：LiDAR/深度/里程计等感知接入、移动实验助理流程，以及无运动权限的 BEV/占据流研究候选。
- **双机械臂工作站**：固定任务下的投袋、状态确认和研磨协作；研究模型与冻结动作链隔离。
- **STM32F407 执行层**：底盘、升降、推杆、舵机和电磁铁的底层通信与安全状态。
- **指挥中心**：脱敏状态、研究证据和项目内容的只读展示，不反向控制设备。

系统不是“一个大模型控制所有硬件”。模型、感知、动作、审计和公网展示拥有不同权限；`shadow` 或审计通过都不会自动授予动作权。

![系统逻辑架构](assets/images/system/fig_xrd_architecture_html.png)

[阅读完整系统架构、数据边界与权限矩阵 →](docs/architecture/SYSTEM_ARCHITECTURE.md)

## 已验证状态

以下数字来自仓库内 2026-08-04 的 PC acceptance、独立 board overlay 和双板收口回执；每项均保留输入与状态边界。

| 范围 | 结论 | 不能扩大解释为 |
| --- | --- | --- |
| 模型注册 | 50/50 唯一逻辑模型达到 PC release-ready | 50 个模型全部在 X5 上通过 |
| X5-local 模型库 | 49 个逻辑模型状态完整：冻结基线 11 + 新增 38 | 49 个模型同时常驻或同时运行 |
| 新增 38 个候选 | 31 `X5_VALIDATED` / 4 `BOARD_EXPERIMENTAL` / 3 `BOARD_REJECTED` | 所有候选都可生产使用 |
| BPU-primary | 24/24 在 actual X5 Bayes-e BPU 执行 | 每个模型都通过语义或质量门 |
| CPU-primary | 14 个中 11 个完成 actual X5 CPU 推理 | 被拒绝的 3 个也已运行 |
| 三套分段 BPU LLM | 六个分段文件 actual BPU 执行；三套均保持实验态 | 通用自由生成或固定 token 一致 |
| 具身脑 v5r1 | 固定输入 actual BPU、200 次延迟、固定张量差分、30 次恢复通过 | 真实相机准确率、导航成功率或运动控制 |
| 集成 Cortex | `MONITOR_OFFLINE` | 完整实时多传感器闭环 |
| 双臂后继候选 | X5 被动 fixture replay；阶段分类头固定样本验收 | 真实学习策略或动作权限 |

模型会计采用“唯一逻辑模型”而非导出格式、量化版本、提示词或随机种子计数。板端 `X5_VALIDATED` 只表示按需单模型固定任务合同通过。

[逐项证据 →](docs/evidence/EVIDENCE_INDEX.md) · [机器可读断言矩阵 →](docs/evidence/CLAIM_MATRIX.csv)

## 三套 BPU LLM 为什么仍是实验态

`F-LLM-03/04/05` 是三套独立领域权重，每套拆为两个 BPU 分段。六个分段均在 actual X5 BPU 执行，且段间内容绑定成立；但三套模型的固定 next-token 都与合同期望不一致。因此项目保留真实执行证据，也保留 `BOARD_EXPERIMENTAL` 结论，不把“能跑”写成“语义正确”，更不宣传通用问答或自由生成。

这类失败保留机制同样适用于资产不完整的 rejected 候选和质量门未通过的模型。

## 复现分层

| Tier | 你能复现什么 | 默认风险与权限 |
| ---: | --- | --- |
| 0 | 文档、图片、报告、静态站点和证据 | 无安装、无设备 |
| 1 | mock、fixture、合同与离线 replay | PC 本地；禁止硬件访问 |
| 2 | 获得许可的数据/模型评估 | 独立下载、固定版本与哈希 |
| 3 | RDK X5 固定输入推理 | 板卡环境；没有运动权限 |
| 4 | 真实传感器、执行器与机器人 | 现场人工授权、急停和物理隔离 |

第一次接触项目请从 [Tier 0/1 安全离线快速开始](docs/getting-started/QUICKSTART_OFFLINE.md) 进入。仓库根目录已验证以下 Tier 1 命令；它们不访问网络、相机、串口、GPIO、机器人 SDK 或执行器：

```bash
python -B tools/publication/audit_release.py --root . --strict
python -B tools/publication/check_markdown_links.py . --format text
python -B tools/publication/render_award_status.py --check
python -B tools/publication/generate_sbom.py --check
python -B tools/publication/verify_media.py --root .
python -B -m unittest discover -s tests_public -p "test_*.py" -v
python -B examples/offline_demo/run_demo.py
```

本次发布已通过公开边界审计、仓内文档链接检查、奖项单一事实源检查、确定性 SPDX SBOM 检查、无硬件单元测试、离线 demo、工作站前端和具身仪表盘前端构建。`v1.0.1` 最终精确复核范围与命令见 [版本验证记录](docs/releases/v1.0.1/VERIFICATION.md)。

> [!CAUTION]
> 仓库包含可能控制真实底盘、升降、机械臂、推杆、舵机和电磁铁的源码。不要把源码存在理解为运行许可。Tier 4 不提供通用一键命令；任何真机操作前必须阅读 [物理安全规范](docs/safety/PHYSICAL_SAFETY.md)。

## 仓库地图

| 路径 | 内容 | 文档入口 |
| --- | --- | --- |
| [`ai_brain/`](ai_brain/) | 材料预测、XRD/PL、ICMat 50 模型候选 | [AI 脑](docs/modules/AI_BRAIN.md) |
| [`embodied_brain/`](embodied_brain/) | ROS 2、移动感知、v5r1/vNext/Cortex 候选 | [具身脑](docs/modules/EMBODIED_BRAIN.md) |
| [`workstation/`](workstation/) | 双臂冻结链与 replay-only 后继候选 | [双机械臂](docs/modules/DUAL_ARM_WORKSTATION.md) |
| [`firmware/stm32f407/`](firmware/stm32f407/) | STM32F407 固件与执行层 | [固件](docs/modules/STM32F407.md) |
| [`web/command_center/`](web/command_center/) | 指挥中心与只读公开站 | [指挥中心](docs/modules/COMMAND_CENTER.md) |
| [`safety/`](safety/) | RB-VoE 等只读安全审计实现 | [系统架构](docs/architecture/SYSTEM_ARCHITECTURE.md) |
| [`evidence/`](evidence/) | 脱敏发布候选回执与 acceptance | [证据索引](docs/evidence/EVIDENCE_INDEX.md) |
| [`schemas/`](schemas/) | 状态接口合同和示例 | [快速开始](docs/getting-started/QUICKSTART_OFFLINE.md) |
| [`assets/`](assets/) | 系统图片与经审核媒体 | 本页下方画廊 |
| [`report_source/`](report_source/) | 竞赛报告源文件和图表源 | [公开边界](docs/safety/PUBLICATION_BOUNDARY.md) |
| [`public_site_static/`](public_site_static/) | 可离线打开的历史只读站点快照 | [指挥中心](docs/modules/COMMAND_CENTER.md) |

旧的 `edge_public/`、`workstation_public/` 和 `public_evidence_data/` 继续作为早期公开快照与安全 mock 边界保存；它们不替代当前完整模块。

## 最新实机图库

[打开完整项目实物与演示画廊 →](docs/gallery.md)。画廊逐项记录时间码、来源哈希和真实性边界；主展示全部使用 2026-08-13 指定的最新 6 张照片与 3 段视频。

| 移动实验助理三分之四视角 | 双机械臂工作站 |
| --- | --- |
| ![移动实验助理最新实物图](assets/media/photos/embodied-platform-three-quarter-full.webp) | ![双机械臂工作站最新实物图](assets/media/photos/dual-arm-workcell-full.webp) |

| 移动实验助理正面 | 传感器与本地显示 |
| --- | --- |
| ![移动实验助理正面](assets/media/photos/embodied-platform-front-full.webp) | ![移动实验助理传感器平台](assets/media/photos/embodied-platform-sensor-deck-full.webp) |

| 项目原始海报（二维码保留） | 双臂联调现场合影 |
| --- | --- |
| ![包含原始二维码的完整项目海报](assets/media/photos/project-overview-poster.webp) | ![包含现场成员的双机械臂联调合影](assets/media/photos/team-dual-arm-integration-full.webp) |

演示媒体必须标明 `live`、`shadow`、`replay` 或 `sim-only`；照片和视频只证明相应时间点、相应工装与相应边界内发生的内容。

## 数据、模型与许可

- 团队有权发布且未携带更具体声明的源码按 [Apache-2.0](LICENSE) 提供。
- 数据集、基座模型、论文、字体、网页库和媒体保留各自许可，不由顶层许可证重新授权。
- 材料资源的 URL、版本、许可、风险与断言边界记录在 [`source_catalog.v1.json`](ai_brain/icmat_foundry/contracts/source_catalog.v1.json)。
- 凭据、个人/设备身份、未授权实验数据、逐文档受限语料以及不可再分发工件不进入公开制品。

详见 [第三方声明](THIRD_PARTY_NOTICES.md)、[NOTICE](NOTICE) 和 [公开发布边界](docs/safety/PUBLICATION_BOUNDARY.md)。

## 已知限制与路线图

项目不会隐藏失败状态。发布工程门已通过，但这不改变技术限制：三套 BPU LLM 语义门未通过、具身 Cortex 未完成真实同步会话、双臂学习候选仍是 replay、Tier 4 复现依赖具体硬件与现场安全条件，且部分数据/模型不能直接再分发。

[完整已知限制](docs/evaluation/KNOWN_LIMITATIONS.md) · [发布工程状态与研究路线](ROADMAP.md) · [变更记录](CHANGELOG.md)

## 贡献、支持与引用

- 贡献前阅读 [CONTRIBUTING.md](CONTRIBUTING.md)；硬件和公开断言修改需要额外审查。
- 普通问题见 [SUPPORT.md](SUPPORT.md)，安全问题按 [SECURITY.md](SECURITY.md) 私下报告。
- 维护与决策原则见 [MAINTAINERS.md](MAINTAINERS.md)，社区规范见 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。
- 学术或工程复用请引用 [`CITATION.cff`](CITATION.cff) 对应的版本；若后续版本获得归档 DOI，再优先使用 DOI。

## 名称与事实原则

项目正式名称是“**基于双 RDK X5 异构协同的材料合成 AI 预测与多机具身实验助理机器人**”。`XRD` 只作为 X 射线衍射技术缩写和兼容性内部标识，不作为仓库或项目名称。项目不使用“全球第一”“完全自主”等不可验证表述，不把荧光粉等同于全部集成电路材料，不把仿真/回放写成真实闭环，也不在官方公布前预测全国奖项。
