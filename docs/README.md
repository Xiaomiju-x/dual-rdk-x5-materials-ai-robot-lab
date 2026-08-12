# XRD Smart Lab 文档中心

这里是项目公开文档的主入口。先根据你的目标选择阅读路径，不要从真实硬件脚本开始。

## 新读者

1. [系统架构](architecture/SYSTEM_ARCHITECTURE.md)：模块、数据流、权限与状态词汇。
2. [安全离线快速开始](getting-started/QUICKSTART_OFFLINE.md)：无需设备即可审阅的 Tier 0/1 入口。
3. [证据索引](evidence/EVIDENCE_INDEX.md)：公开断言与机器可读回执的对应关系。
4. [已知限制](evaluation/KNOWN_LIMITATIONS.md)：哪些能力尚未证明。
5. [数据与晶体缓存来源](data/PROVENANCE.md)：公开数据、缓存与再分发边界。

## 模块导航

| 模块 | 文档 | 核心边界 |
| --- | --- | --- |
| AI 脑 | [AI_BRAIN.md](modules/AI_BRAIN.md) | 材料预测、XRD/PL、按需模型库；不能把实验模型写成通用能力 |
| 具身脑 | [EMBODIED_BRAIN.md](modules/EMBODIED_BRAIN.md) | 移动感知与固定任务 BPU；研究候选无运动权限 |
| 双机械臂工作站 | [DUAL_ARM_WORKSTATION.md](modules/DUAL_ARM_WORKSTATION.md) | 冻结动作链与离线学习候选严格分离 |
| STM32F407 | [STM32F407.md](modules/STM32F407.md) | 底层执行与急停；开环机构需要人工和硬件保护 |
| 指挥中心 | [COMMAND_CENTER.md](modules/COMMAND_CENTER.md) | 公开只读展示，不反向控制设备 |

## 真实性与安全

- [证据矩阵](evidence/CLAIM_MATRIX.csv)
- [物理安全](safety/PHYSICAL_SAFETY.md)
- [公开边界](safety/PUBLICATION_BOUNDARY.md)
- [软件物料清单（SBOM）](safety/SBOM.md)
- [竞赛与奖项](competition/AWARDS.md)
- [项目实物与演示画廊](gallery.md)
- [官方来源](competition/OFFICIAL_SOURCES.md)
- [奖项权威状态](competition/award_status.yaml)

## 社区与版本

- [贡献指南](../CONTRIBUTING.md)
- [安全报告](../SECURITY.md)
- [支持范围](../SUPPORT.md)
- [路线图](../ROADMAP.md)
- [变更记录](../CHANGELOG.md)
- [v1.0.0 验证记录](releases/v1.0.0/VERIFICATION.md)
- [引用信息](../CITATION.cff)

历史文件 `PROJECT_MAP.md` 与 `PUBLIC_BOUNDARY.md` 仅保留兼容入口；当前内容分别以本页和 `safety/PUBLICATION_BOUNDARY.md` 为准。
