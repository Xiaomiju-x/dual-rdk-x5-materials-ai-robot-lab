# XRD Embedded Contest Public Code Boundary

作品名称：基于双 RDK X5 异构协同的材料合成 AI 预测与多机具身实验助理机器人

本目录用于嵌入式竞赛初赛的重要代码提交和 GitHub 公共边界展示。它包含真实项目中的非核心代码与可审查工程资产，不包含核心材料预测引擎、模型权重、私有实验数据、部署凭据或危险硬件控制入口。

## Included

- `public_site_static/`：公网展示站静态前端、PWA、3D 晶体展示资源和离线页面资产。
- `public_site_tools/`：公开展示用 3D 晶体 GLB 生成脚本。
- `public_site_reports/`：只读状态报告样例，不含账号、IP 或密钥。
- `workstation_public/`：固定工位的碰撞互锁、mock 状态、技能记录回放和图标生成等非核心逻辑。
- `workstation_frontend_public/`：双机械臂工位前端中的 3D 机械臂、图表、交互组件、状态 store 和部分页面源码。
- `edge_public/`：Lab-FSD shadow planner、Fly-MB 判决脑和 F407 动作时序的公开接口/伪代码。
- `schemas/`：只读状态接口 schema 与示例响应。
- `report_source/`：作品设计报告 TeX 源码、HTML 图表源码、MATLAB/Python 图表生成脚本和生成图资产。
- `public_evidence_data/`：公网截图、报告渲染页、公开晶体结构缓存和离线演示传感器帧。

## Excluded

- API Key、Cookie、SSH/WiFi/SSO/隧道配置、账号凭据和内网地址。
- GGUF、LoRA、BPU bin、tokenizer、tensor 包等模型权重。
- 私有 XRD/PL 原始数据、未发表实验记录、failure patterns 和完整训练集。
- 核心材料预测引擎、私有规则库、部署脚本和真实硬件危险控制脚本。

GitHub: https://github.com/Xiaomiju-x/xrd
