# embodied_brain — 快速上手

> 公开架构与当前状态见[具身脑模块导航](../docs/modules/EMBODIED_BRAIN.md). 本文保留历史快速上手语境.

## TL;DR

```bash
# PC 端开发
cd /c/Users/xiaomiju2026/Desktop/xrd_backup/embodied_brain
bash deploy_to_car.sh preflight # 纯静态检查；默认 HOLD，不联网

# 仅在用户确认开机且只读 SSH 身份检查成功后，显式传入核验地址
CAR_HOST='rdk@<verified-k70-upstream-ip>' bash deploy_to_car.sh

# 车载脑端 (X5)
ssh 'rdk@<verified-k70-upstream-ip>'
cd ~/ros2_ws && colcon build --symlink-install --merge-install --parallel-workers 1
source install/setup.bash
ros2 launch my_robot_bringup full.launch.py
```

PC 保持连接 `Redmi K70`。不得为了部署切换到 `xrd-lab_5G`，也不得在未经确认时修改 Wi-Fi、VPN、代理、路由或 ARP。具身脑 K70 上游地址当前按未验证处理，不能使用设备侧固定副 IP 代替。

## 工作区布局

```
embodied_brain/
├── CLAUDE.md                顶层架构 (必读)
├── README.md (本文)
├── deploy_to_car.sh         scp + 重启脚本 (Phase 9)
├── ros2_ws/src/             8 个 ROS2 包
│   ├── my_robot_description/
│   ├── my_robot_drivers/
│   ├── my_robot_navigation/
│   ├── my_robot_perception/
│   ├── my_robot_agents/
│   ├── my_robot_bridge/
│   ├── my_robot_msgs/
│   └── my_robot_bringup/
├── stm32_f407/              F407 固件 (后期)
└── docs/                    硬件接线 / 标定 / 演示
```

## Phase 路线图

| Phase | 内容 | 状态 |
|---|---|---|
| **0** | 脚手架 (URDF + 包结构 + 文档) | 完成 |
| **1** | 传感器驱动 (LD14 + Astra Pro + serial_F407) | 部分完成: 雷达/深度在线，新 F407 安全固件待烧录验收 |
| **2** | SLAM + Nav2 + TF 全树 | 部分完成: SLAM 已实测，Nav2/MPPI 保持安全 shadow/guard |
| **3** | 烧结炉 OCR Agent | 已实现，按相机可用性启停 |
| **4** | 跨网通信 (DDS + HTTP bridge) | 完成 |
| **5** | 车载 LLM fallback 接口 | 已接入，实时控制仍以 ROS2/F407 为权威 |
| **6** | 升降台 + 取料 | 开环演示已验证，独立限位/物体在位反馈待补 |
| **7** | 小米云台拉流 | 非当前主链，保留文档 |
| **8** | K3/PT 相机感知 | 可选设备，当前未接 `/dev/PT_CAM` |
| **9** | Bringup + systemd 一键启动 | 完成，Lab-FSD shadow 已纳入主服务 |

## 硬件清单

公开模块边界见[具身脑模块导航](../docs/modules/EMBODIED_BRAIN.md)，执行层限制见[STM32F407 文档](../docs/modules/STM32F407.md)。

## 与早期 v1 demo 的关系

早期 `ros/` 是已冻结的 SLAM 验证 demo，未随本发布树分发。`embodied_brain/` 是 v2 重写，参考 v1 的 0xAA55 串口协议和 LD14 udev 配置等已验证实现，但代码风格和包结构是新的；未分发内容的原则见[公开边界](../docs/safety/PUBLICATION_BOUNDARY.md)。

历史 v1 归档只读，不作为当前公开复现入口。
