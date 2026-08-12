# 双机械臂工作站

工作站由两台固定机械臂、独立视觉和定制末端机构组成，用于投袋、状态确认与并发研磨。公开仓库同时保留冻结真机链和后继被动学习候选，但二者的权限完全不同。

## 代码导航

| 路径 | 内容 | 权限 |
| --- | --- | --- |
| [`workstation/dual_arm/`](../../workstation/dual_arm/) | 冻结 v3 现场动作、状态门和回执 | 含物理动作源码；Tier 4 |
| [`workstation/dual_arm_successor/`](../../workstation/dual_arm_successor/) | Tiny-ACT、世界模型、阶段分类与一致性监测 | replay-only，无运动权限 |
| [`workstation_public/`](../../workstation_public/) | mock 遥测、碰撞联锁与回放示例 | 面向离线审阅 |
| [`workstation_frontend_public/`](../../workstation_frontend_public/) | 工作站公开 UI 组件 | 类型检查与 Vite 构建已通过；CI 在干净环境复验 |

## 事实边界

- 冻结 v3 动作链有真实执行证据，动作参数不因研究候选而自动改变。
- 后继模型使用历史阶段或指令派生 fixture 进行 replay；不是连续真实关节遥测训练出的真机策略。
- BPU 阶段分类头完成固定样本板端验收；其他头仅编译或未通过晋级门时，不能被运行时消费。
- 学习候选无相机、串口、GPIO、远程连接或机器人 SDK 权限。

源码可读不等于可以安全复现现场动作。机械布局、工具、负载、标定、急停和操作者均属于 Tier 4 前置条件。
