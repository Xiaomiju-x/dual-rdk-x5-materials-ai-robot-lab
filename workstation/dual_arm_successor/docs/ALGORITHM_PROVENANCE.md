# 算法来源、许可与真实性台账

审计日期：2026-07-30
适用候选：`DualArm-ShadowVLA` / `X5-BiSkill Shadow`

## 1. 使用规则

1. 本文件记录的是候选参考来源，不代表已下载、训练、部署或真机验证。
2. 引用论文思想不等于复用代码或权重。代码、权重和数据集分别核验许可。
3. 任何外部资产进入训练包前，必须记录精确版本、来源 URL、许可证、
   文件 SHA-256 和再分发条件。
4. 未明确授权的资产只可作为论文/方法参考，不复制进仓库，不打入发布包。
5. 所有模型输出只进入 `POST_RUN_REPLAY`，不获得运动权限。

## 2. 来源矩阵

| 项目 | 类型与真实能力 | 当前候选用途 | 代码许可 | 权重/数据许可与边界 | 来源 |
|---|---|---|---|---|---|
| Xiaomi-Robotics-0 (XR-0) | 约 4.7B/5B VLA，公开推理、评测和后训练资料，输出动作块 | VLA 架构、跨具身预训练和后训练参考；可选离线教师实验 | 官方 GitHub 标注 Apache-2.0 | 公开 HF 权重页面标注 Apache-2.0；使用时仍固定具体 model revision | [项目页](https://robotics.xiaomi.com/xiaomi-robotics-0.html), [GitHub](https://github.com/XiaomiRobotics/Xiaomi-Robotics-0), [HF 权重](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-0-Pretrain) |
| Xiaomi-Robotics-1 (XR-1) | 基于 100K 小时 UMI 和真实机器人数据的机器人基础模型研究 | 规模化数据、UMI 预训练和具身对齐方法参考 | 截至审计日官方仓库仍写明 code/model weights will be released soon，仓库未提供可复用实现许可 | 不假定权重、训练数据或 UMI 数据可获得或可再分发；仅引用公开报告 | [项目页](https://robotics.xiaomi.com/xiaomi-robotics-1.html), [GitHub](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1) |
| Xiaomi-Robotics-U0 (XR-U0) | 38B 自回归世界基础模型，用于场景、迁移、图像及视频生成；不是直接动作控制 VLA | 世界模型、场景迁移和合成数据研究参考；不作为动作策略 | 官方 GitHub 标注 Apache-2.0 | 当前公开 HF 权重标注 Apache-2.0；Video checkpoint 在审计时仍标为 coming soon | [项目页](https://robotics.xiaomi.com/xiaomi-robotics-u0.html), [GitHub](https://github.com/XiaomiRobotics/Xiaomi-Robotics-U0), [HF 权重](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-U0) |
| SmolVLA / LeRobot | 约 0.5B 轻量 VLA，多视角、状态和语言输入，连续动作块输出 | 主要可训练 VLA 候选；只做离线回放和蒸馏 | LeRobot 代码为 Apache-2.0 | `smolvla_base` 模型卡用于特定任务微调；审计页面未显示独立 license 字段，下载或再分发前必须复核模型仓库 metadata 和上游 SmolVLM 条款 | [模型卡](https://huggingface.co/lerobot/smolvla_base), [LeRobot](https://github.com/huggingface/lerobot), [文档](https://huggingface.co/docs/lerobot/main/smolvla), [论文](https://arxiv.org/abs/2506.01844) |
| ACT | Action Chunking with Transformers；通过动作分块学习精细双臂操作 | Compact ACT / Tiny-ACT 离线基线、13D 动作块教师 | MIT | 训练数据由本项目自行采集或从冻结证据转换；不得默认继承 ALOHA 数据许可 | [GitHub](https://github.com/tonyzhaozh/act), [论文](https://arxiv.org/abs/2304.13705), [MIT License](https://github.com/tonyzhaozh/act/blob/main/LICENSE) |
| OpenVLA-OFT | 对 OpenVLA 做 OFT，官方实现覆盖 LIBERO 与 ALOHA 双臂设定 | 重型双臂教师、14D/动作块方法对标；不直接部署 X5 | MIT | 基础 OpenVLA 权重、Prismatic 依赖及训练数据各自受独立条款约束，必须逐项核验 | [项目页](https://openvla-oft.github.io/), [GitHub](https://github.com/moojink/openvla-oft), [MIT License](https://github.com/moojink/openvla-oft/blob/main/LICENSE) |

## 3. 项目自有算法边界

`DualArm-ShadowVLA` / `X5-BiSkill Shadow` 的项目自有部分计划包括：

- 冻结 v3 证据到双相机、双臂 13D episode 的确定性适配合同。
- 双臂阶段图、并发研磨同步标签和实验任务语言模板。
- 教师动作块到 X5 学生阶段/技能/结果头的蒸馏流程。
- X5 CPU 的时间对齐、规则一致性、Conformal/OOD 与证据落盘。
- Bayes-e BPU 轻量学生转换和板端回执。
- 不具有运动权限的 shadow UI 与证据胶囊。

这些是候选设计，不得在未产生训练、转换和板端回执前写成“已部署”。

## 4. 具体边界

### XR-0

可说：XR-0 是已开放代码与权重的 VLA 参考，提供动作块和后训练路径。
不可直接说：XR-0 已适配本项目双臂、已在 X5 运行或已控制机械臂。

### XR-1

可说：候选借鉴其大规模 UMI 预训练与具身后训练方法。
不可说：本项目获得、训练或部署了 XR-1；截至审计日官方仓库仍未提供可复用
代码和权重。

### XR-U0

可说：它是世界基础模型参考，可启发未来状态、场景迁移和合成数据。
不可说：38B XR-U0 是 VLA、已在 X5 上运行，或直接生成了本项目机械臂命令。

### SmolVLA

可说：它是适合定制数据微调的轻量 VLA 候选。
只有训练回执、评测和模型哈希齐全后，才可说“完成离线微调”。没有板端
回执时不得说“部署到 X5”。

### ACT

可说：ACT 为双臂动作分块的强基线，候选使用 13D 自有动作合同。
合成 smoke 或旧数据实验不能替代决赛实采数据验证。

### OpenVLA-OFT

可说：其 ALOHA 双臂方法适合作为重型教师与对标。
不可说：完整 OpenVLA-OFT 已在单张 5090 上完成官方规模训练，除非有实际
训练配置、资源和回执支持。

## 5. 许可回执最低字段

每个真正进入训练、转换或发布的外部资产必须写入
`model_receipt_v1.schema.json` 对应记录：

- `name`
- `artifact_type`
- `source_url`
- `revision`
- `license_id`
- `license_url`
- `sha256`
- `redistribution_allowed`
- `notice_required`
- `verified_at`

若任一字段未知，资产状态必须为 `REFERENCE_ONLY`，不得进入可再分发包。

## 6. 引用不等于实测

论文、项目页和官方 benchmark 数字只能以“上游报告”表述。项目自己的
实测必须来自本候选的 episode、prediction 和 model receipt，且同时保留：

- 运行硬件与软件环境。
- 数据集拆分与 episode 数量。
- checkpoint / ONNX / Bayes-e `.bin` SHA-256。
- PC、编译器和 X5 板端结果的明确区分。
- `motion_authority=false`、`execution_allowed=false`、
  `actuator_commands_issued=0`。
