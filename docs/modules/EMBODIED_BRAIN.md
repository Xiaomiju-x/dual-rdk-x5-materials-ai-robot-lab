# 具身脑

具身脑负责移动平台的传感器接入、ROS 2 组织、取放/升降演示链，以及不拥有运动控制权的 BEV、占据流与风险监测研究候选。

## 代码导航

| 路径 | 内容 | 当前边界 |
| --- | --- | --- |
| [`embodied_brain/ros2_ws/`](../../embodied_brain/ros2_ws/) | ROS 2 描述、驱动、导航、感知、消息与 bringup | 含真实接口；仅 Tier 4 现场使用 |
| [`embodied_brain/finals_successor/`](../../embodied_brain/finals_successor/) | TinyOccFlow v5r1 / CamSemLite | 固定任务 BPU accepted，shadow-only |
| [`embodied_brain/finals_vnext/`](../../embodied_brain/finals_vnext/) | 更丰富的研究候选 | 不替换 v5r1 |
| [`embodied_brain/finals_cortex/`](../../embodied_brain/finals_cortex/) | 多传感器、CrossBEV、NavTeacher、TrustLab | `MONITOR_OFFLINE` |
| [`embodied_brain/tools/`](../../embodied_brain/tools/) | 验证和现场工具 | 运行前逐文件确认副作用 |

## 已验证状态

TinyOccFlow v5r1 与 CamSemLite 在 actual X5 Bayes-e BPU 完成固定输入、200 次延迟、FP32/INT8 固定张量差分和 30 次加载/退出恢复门。候选没有注册常驻服务、发布运动命令或获得底盘权限。

这些结果不证明真实相机语义准确率、完整同步传感器融合、真实导航成功率或自动驾驶能力。集成 Cortex 因缺少候选 ROS 图、完整实时传感器会话和相应资源证据，保持 `MONITOR_OFFLINE`。

## 安全边界

- shadow 候选不得发布运动话题、权威 TF 或 F407 指令。
- 候选失败只能让监视器离线，不能阻断冻结演示链。
- 传感器缺失、过期、fixture 或回放必须在 provenance 中明确标注。
- 任何真实传感器或运动测试均属于 Tier 4，遵循 [物理安全规范](../safety/PHYSICAL_SAFETY.md)。
