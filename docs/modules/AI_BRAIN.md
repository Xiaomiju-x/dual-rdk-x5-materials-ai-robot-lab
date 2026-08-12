# AI 脑

AI 脑面向半导体光电子器件与先进封装功能材料的边缘材料智能，以近红外荧光粉作为真实验证载体，并把公开基准上的电子材料、XRD、SEM、制程和封装任务组织为可追溯候选。

## 代码导航

| 路径 | 内容 |
| --- | --- |
| [`ai_brain/dashboard/`](../../ai_brain/dashboard/) | 材料工作流与状态展示入口 |
| [`ai_brain/predict_engine/`](../../ai_brain/predict_engine/) | 材料预测、证据约束和辅助推理 |
| [`ai_brain/xrd_vision/`](../../ai_brain/xrd_vision/) | XRD 图像分析 |
| [`ai_brain/xrd_numerical/`](../../ai_brain/xrd_numerical/) | XRD 数值分析 |
| [`ai_brain/pl_vision/`](../../ai_brain/pl_vision/) | PL 图像分析 |
| [`ai_brain/pl_numerical/`](../../ai_brain/pl_numerical/) | PL 数值分析 |
| [`ai_brain/icmat_foundry/`](../../ai_brain/icmat_foundry/) | 决赛 50 模型注册、合同、证据与工具链 |

## 已验证状态

- PC 侧 50/50 注册合同达到 release-ready；其中 11 个冻结 X5 基线、38 个新增、1 个 PC-only。
- 49 个 X5-local 模型具有完整状态，但采用按需装载，不同时常驻。
- 新增 38 个候选：31 validated、4 experimental、3 rejected。
- 24/24 BPU-primary 在 actual X5 BPU 执行；14 个 CPU-primary 中 11 个在 actual X5 CPU 执行。
- 三套分段 BPU LLM 的六个文件完成实际 BPU 执行，但 next-token 合同失败，保持实验态。

## 解释边界

- `X5_VALIDATED` 只表示单模型固定任务合同通过，不等于生产质量、任意输入泛化或同时常驻。
- PC acceptance 中的上板字段保留上电前快照；板端结果记录在独立 overlay，二者不是矛盾的同一时间点。
- `SIM_ONLY` 必须持续显示；质量受限且未晋级的候选不能成为效果锚点。
- 语言模型输出是证据辅助，不是科学 ground truth、自动实验许可或安全裁决。
- 公开数据集结果不能称为团队产线数据。

证据入口见 [AI 脑回执索引](../evidence/EVIDENCE_INDEX.md#ai-脑模型库)。
