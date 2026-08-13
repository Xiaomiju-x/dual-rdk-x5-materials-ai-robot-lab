# 证据索引

本索引把 README 中的主要公开断言映射到本仓库回执。证据描述“在何种输入、后端和边界下发生了什么”，不是产品认证或无限泛化保证。

## 证据使用规则

1. 优先使用机器可读 JSON 和其 SHA-256；Markdown 回执用于解释范围和例外。
2. PC acceptance 与 board overlay 是不同时间点的事实，不覆盖彼此。
3. 回执中的 `live`、`shadow`、`replay`、`sim-only`、`experimental`、`rejected` 标记必须保留。
4. 文件存在不代表已经完成公开脱敏；正式 Release 前仍需按 [公开边界](../safety/PUBLICATION_BOUNDARY.md) 扫描全部证据。
5. 证据中引用但未随公共仓库分发的原始工件，不应被描述为第三方可独立复现。

## 系统收口

- [`evidence/system/DUAL_X5_CANDIDATE_BOARD_CLOSEOUT_20260804.md`](../../evidence/system/DUAL_X5_CANDIDATE_BOARD_CLOSEOUT_20260804.md)：AI 脑与具身脑隔离上板总览、非干扰和状态边界。

## AI 脑模型库

| 断言 | 权威证据 | 边界 |
| --- | --- | --- |
| 50 个唯一逻辑模型的 PC 验收 | [`final_acceptance.v1.json`](../../evidence/ai_brain/final_acceptance.v1.json) 与 [SHA-256 文件](../../evidence/ai_brain/final_acceptance.v1.json.sha256) | PC 合同与制品验收；不是全部板端通过 |
| 49 个 X5-local 状态覆盖、新增 31/4/3 | [`x5_board_phase_acceptance.v1.json`](../../evidence/ai_brain/x5_board_phase_acceptance.v1.json) | 按需模型库，不同时常驻 |
| 24/24 BPU-primary、11/14 CPU-primary actual X5 执行 | [AI 脑板端最终回执](../../evidence/ai_brain/X5_BOARD_PHASE_FINAL_RECEIPT_20260804.md) | 单模型固定任务合同 |
| 三套分段 BPU LLM 保持实验态 | 同上 | 六个分段文件执行过，但固定 next-token 不一致 |

PC acceptance 内的板端字段保留上电前值；板端结果只记录在独立 overlay 中。这种不可变历史设计避免为了新结果回写旧证据。

## 具身脑

- [X5-TriBEV-Flow v5r1 板端验收回执](../../evidence/embodied_brain/X5_BOARD_ACCEPTANCE_20260804.md)：actual BPU 固定输入、200 次延迟、固定张量差分、30 次加载/退出恢复和 shadow-only 边界。

该回执不能证明真实相机准确率、真实导航成功率或完整实时融合；Cortex 状态仍为 `MONITOR_OFFLINE`。

## 双机械臂工作站

- [`FINALS_PART3_HANDOFF_20260720.md`](../../evidence/workstation/FINALS_PART3_HANDOFF_20260720.md)：冻结 v3 物理动作链和现场回执索引。
- [`X5_BOARD_DEPLOYMENT_HANDOFF_20260804.md`](../../evidence/workstation/X5_BOARD_DEPLOYMENT_HANDOFF_20260804.md)：后继候选的 X5 被动 replay 交接。

这些历史交接包含现场部署语境，正式公开前必须再做设备身份与个人路径脱敏。前者只支持固定任务链；后者只支持 replay，不支持真实学习策略主张。

## 竞赛成绩

- [`award_status.yaml`](../competition/award_status.yaml) 是唯一权威状态源。
- [官方来源列表](../competition/OFFICIAL_SOURCES.md) 区分“证明竞赛/赛题存在”和“证明具体名次”。
- 西南赛区一等奖、全国总决赛二等奖当前均为队伍确认；组委会官方获奖证据待补。

## 逐项矩阵

机器友好的简表位于 [`CLAIM_MATRIX.csv`](CLAIM_MATRIX.csv)。新增可量化断言时，应同时增加输入、后端、样本数、状态和限制，不能只追加宣传文字。
