# my_robot_msgs

> 具身脑自定义 ROS2 接口 (msg / srv / action). **Phase 0 完工 2026-04-26**, 跨 Round 4 BPU Sprint 无新增需求 (BPU 节点用 std_msgs/String JSON + sensor_msgs/Image + geometry_msgs/Vector3 已够).

## 是什么

定义具身脑系统中跨节点 / 跨网传递的结构化数据. 凡是用 std_msgs / sensor_msgs 不够表达的, 都在这里加.

## 接口列表

### Messages

- **FurnaceReading.msg** — 烧结炉 OCR 读数 (PV/SV/MV + 置信度 + 是否需要 Qwen-VL 复核)
- **Alarm.msg** — 异常报警 (4 类来源, 等级 1-3, 通道 enum)
- **PickupTarget.msg** — 取料目标位姿 (来自 AprilTag 或 YOLO)
- **LiftStatus.msg** — 升降台当前高度 + 限位状态
- **SystemTelemetry.msg** — 周期遥测 (battery / cpu / ram / bpu_load / nav_state)

### Services

- **SetElectromagnet.srv** — 控制电磁铁 on/off
- **SetLiftHeight.srv** — 控制升降台到指定高度 (阻塞返回)

### Actions

- **DispatchTask.action** — AI 脑发任务给具身脑 (从取料/送货/巡检三选一)
- **Pickup.action** — 完整取料动作 (Nav2 → 升降台 → 视觉伺服 → 电磁铁 → 抬起)

## 关键约定

- **时间戳**: 所有 msg 带 `std_msgs/Header header`, ROS2 时间统一
- **置信度**: 凡是机器学习/OCR 结果都带 `float32 confidence` (0~1)
- **enum**: ROS2 不支持原生 enum, 用 uint8 + const 定义 (见各 msg 文件)
- **跨网兼容**: 不要塞图像/点云这种大数据 (走 sensor_msgs/Image), 自定义 msg 控制在 < 1KB

## 编译

```bash
cd ~/ros2_ws
colcon build --packages-select my_robot_msgs
source install/setup.bash

# 验证
ros2 interface show my_robot_msgs/msg/FurnaceReading
```
