# 项目实物、团队与真机演示

本页使用用户在 2026-08-13 明确指定公开的最新 6 张 JPG 与 3 段 MP4。公开文件均由原件重新编码，移除了 EXIF、GPS、拍摄设备、创建时间和音轨等技术元数据；画面内容未打码、未生成、未补绘。该请求记录的是本次发布的内容选择，**不构成对肖像权、商标、场地或二维码目标的法律授权断言**。逐文件 SHA-256 与转换记录见 [`MEDIA_PROVENANCE.yml`](../assets/media/MEDIA_PROVENANCE.yml)。

> [!IMPORTANT]
> 这些媒体记录特定工装、固定任务和一次真机演示。人工准备、安全员入镜、屏幕读数与物理动作均按原画面保留；不能据此推导完全自主、无人值守、学习策略泛化或跨数据集性能。

## 一图看懂项目

![移动具身平台与双机械臂工作站最新实拍拼图](../assets/media/hero/project-hardware-hero.webp)

左侧是双 RDK X5 具身移动平台，右侧是双 myCobot 固定实验工位。拼图只拼接两张最新实拍照片，没有生成式内容或文字覆盖；完整项目名与竞赛奖项以本仓库标题和奖项单一事实源为准。

## 1. 材料 AI / XRD Dashboard

[![XRD 视觉分析动态预览](../assets/media/previews/dashboard-xrd-pipeline.gif)](../assets/media/videos/dashboard-xrd-pipeline.mp4)

[观看 19 秒静音 MP4](../assets/media/videos/dashboard-xrd-pipeline.mp4)

| 原片时间码 | 公开片段时间码 | 观察到的内容 | 状态边界 |
| --- | --- | --- | --- |
| `00:05–00:24` | `00:00–00:19` | 平板端 XRD 图谱、ROI/视觉分析界面与流程状态 | UI 中置信度、延迟与计数是当次演示读数，不是跨数据集基准 |
| `01:43–01:59` | 未另导出；保留在原片章节记录 | 闭环架构与多路线概览 | 架构页不等于所有链路同时在线 |
| `02:00–02:27` | 未另导出；保留在原片章节记录 | 合成预测、批量预筛与优化矩阵 | 页面展示不替代可复现实验或 acceptance 回执 |
| `02:40–03:20` | 未公开 | 模型全景与 RAG/文献界面 | 原画面包含内部文献路径，仓库演示片不分发该段 |

## 2. 具身实验助理：人工准备、升降与携瓶移动

[![具身实验助理动态预览](../assets/media/previews/embodied-assisted-workflow.gif)](../assets/media/videos/embodied-assisted-workflow.mp4)

[观看 50 秒静音 MP4](../assets/media/videos/embodied-assisted-workflow.mp4)

| 原片时间码 | 公开片段时间码 | 观察到的动作 | 状态边界 |
| --- | --- | --- | --- |
| `00:00–00:08` | 未纳入 MP4；人员画面仍由照片素材公开 | 队员在设备旁观察/准备 | 不把准备阶段计作机器人动作 |
| `00:08–00:24` | `00:00–00:16` | 人员稳定瓶体，升降末端接触并固定瓶体 | 明确为安全员辅助上件，不称自主取瓶 |
| `00:24–00:42` | `00:16–00:34` | 瓶体已固定后，平台/升降机构携瓶移动 | 真机硬件动作；不单独证明自主导航成功率 |
| `00:42–00:58` | `00:34–00:50` | 平台靠近粉色接收容器 | 只描述接近过程，不宣称自动放置已经闭环完成 |
| `01:00–01:08` | 未公开 | RViz/地图界面 | 因原画面显示内部本地路径而不进入仓库片段 |

## 3. 双机械臂：完整真机工位动作

[![双机械臂完整真机演示动态预览](../assets/media/previews/dual-arm-complete-hardware-demo.gif)](../assets/media/videos/dual-arm-complete-hardware-demo.mp4)

[观看 155 秒静音 MP4](../assets/media/videos/dual-arm-complete-hardware-demo.mp4)

| 原片时间码 | 公开片段时间码 | 观察到的动作 | 状态边界 |
| --- | --- | --- | --- |
| `00:10–00:30` | `00:00–00:20` | 视觉末端靠近样品袋工位、对准/夹取 | 固定工装真机动作，不是通用视觉抓取成功率 |
| `00:30–00:58` | `00:20–00:48` | 夹持后移向工位中央 | 只陈述视频中可见的物理运动 |
| `00:58–01:28` | `00:48–01:18` | 投袋/转移阶段与双臂协同 | 双臂真机演示，不证明无人值守或学习策略 |
| `01:28–01:58` | `01:18–01:48` | 容器周边并行动作 | 固定动作链，不外推任务级泛化 |
| `01:58–02:20` | `01:48–02:10` | 一臂回撤、另一臂继续工位动作 | 不把串行动作写成持续并发吞吐 |
| `02:20–02:45` | `02:10–02:35` | 末端继续工位动作并收尾 | 视频证明实体硬件执行，不替代安全/验收回执 |

## 最新 6 张原始画面衍生图

| 项目总览海报（二维码按用户要求原样公开） | 双机械臂完整工位 |
| --- | --- |
| ![项目总览海报](../assets/media/photos/project-overview-poster.webp) | ![双机械臂工位](../assets/media/photos/dual-arm-workcell-full.webp) |

| 移动平台正面 | 传感器与本地显示 |
| --- | --- |
| ![移动具身平台正面](../assets/media/photos/embodied-platform-front-full.webp) | ![移动平台传感器层](../assets/media/photos/embodied-platform-sensor-deck-full.webp) |

| 移动平台三分之四视角 | 队员进行双臂集成调试（人物按用户要求原样公开） |
| --- | --- |
| ![移动平台三分之四视角](../assets/media/photos/embodied-platform-three-quarter-full.webp) | ![队员进行双臂集成调试](../assets/media/photos/team-dual-arm-integration-full.webp) |

照片仅执行缩放、色彩空间归一化和 WebP 重新编码，画面范围完整保留。源手机 EXIF/GPS/设备信息不随派生文件分发。

## 旧静态站截图

先前的 Site27 离线归档截图不再作为本画廊或 README 的主展示素材；仓库中的 `public_site_static/` 仍可用于历史软件说明，但不代表当前在线状态。
