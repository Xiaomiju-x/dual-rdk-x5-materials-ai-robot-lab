# my_robot_description

> URDF / xacro 模型 + display launch + rviz config. **Phase 0 完工 2026-04-26 (TF 树解析正确)**. 跨 Round 4 BPU Sprint 无变更 — 物理结构没动, 只是新增了麦阵/相机的逻辑节点.

## 是什么

定义具身脑机器人的物理结构 (link + joint + 传感器/执行器位置), 给:
- **robot_state_publisher**: 发布 TF 树
- **joint_state_publisher**: 关节状态 (升降台 prismatic, 轮子 continuous)
- **rviz2**: 可视化机器人模型 + TF
- **Nav2 / SLAM**: 用 base_link → laser_link 等 TF 定位传感器

## 目录

```
my_robot_description/
├── package.xml + CMakeLists.txt
├── urdf/
│   ├── my_robot.urdf.xacro          顶层入口, include 所有子文件
│   ├── common.urdf.xacro            inertial / material macro
│   ├── base.urdf.xacro              底盘 (purple platform)
│   ├── pillar.urdf.xacro            立柱 + 顶层托盘
│   ├── sensor/
│   │   ├── imu.urdf.xacro           IMU (在 base 内, 暂占位)
│   │   ├── laser.urdf.xacro         LD14 (顶层立柱中段)
│   │   ├── depth_camera.urdf.xacro  Astra Pro (底盘前部)
│   │   ├── lift_camera.urdf.xacro   200W USB (升降台上)
│   │   └── pt_camera.urdf.xacro     小米云台 OR 备选 S20 (立柱顶端)
│   └── actuator/
│       ├── wheel.urdf.xacro         驱动轮 ×2 macro
│       ├── caster.urdf.xacro        万向轮 ×1 macro
│       ├── lift_stage.urdf.xacro    升降台 (prismatic joint)
│       └── electromagnet.urdf.xacro 电磁铁 (吸盘)
├── launch/
│   └── display.launch.py            rviz + RSP + JSP-gui
├── rviz/
│   └── display.rviz                 rviz config
└── meshes/                          (空, mesh 待加)
```

## 关键约定

- **单位**: 长度 m, 角度 rad
- **TF 命名** (REP-105):
  - `map` → `odom` → `base_link` → 各 `xxx_link`
  - `base_footprint` 在 base_link 正下方 (轮子接地高度), z=0
- **传感器 frame_id 规范** (硬编码到 launch / 节点的, **不要改名**):
  - `laser_link` (LD14)
  - `depth_camera_link` + `depth_camera_optical_frame` (Astra)
  - `lift_camera_link` + `lift_camera_optical_frame` (200W USB)
  - `pt_camera_link` (云台)
  - `imu_link`
- **关节命名**:
  - `left_wheel_joint`, `right_wheel_joint` (continuous, axis y)
  - `front_caster_joint` (continuous, fixed if 万向无 yaw 反馈)
  - `lift_joint` (prismatic, axis z, 升降范围 0~0.6m)
  - `electromagnet_joint` (fixed, 挂在 lift_link 末端)

## 维度 (现在是占位, 后续按实物量)

| 部件 | x (前后) | y (左右) | z (高) | 来源 |
|---|---|---|---|---|
| 底盘 base | 0.40 | 0.30 | 0.05 | 占位 |
| 立柱 pillar | 0.04 | 0.04 | 0.90 | 占位 |
| 顶层托盘 tray | 0.30 | 0.30 | 0.02 | 占位 |
| 驱动轮 wheel | r=0.04, l=0.02 | — | — | 占位 |
| 万向轮 caster | r=0.025 | — | — | 占位 |
| LD14 laser | 0.05 | 0.05 | 0.04 | LD Robot 官方 |
| Astra Pro | 0.16485 | 0.04825 | 0.030 | 官方 spec |
| 200W USB cam | 0.0326 | 0.0326 | 0.028 | 官方 spec |
| 小米云台 | 0.078 | 0.078 | 0.119 | 官方 spec |
| 升降台 lift | 0.20 | 0.10 | 0.02 | 占位 |
| 电磁铁 | r=0.03, l=0.04 | — | — | 占位 |

## 验证

```bash
# 检查 URDF 语法
check_urdf $(xacro urdf/my_robot.urdf.xacro)

# rviz 看模型
ros2 launch my_robot_description display.launch.py

# 检查 TF 树
ros2 run tf2_tools view_frames
```

## 已知坑

- 行首多个 `xacro:` 前缀 (像 v1 demo 里那样) 会被 XML 解析为文字, 别犯
- `<inertial>` 必须有合理 mass/inertia, 不然 Gazebo 仿真会飞 (现在不仿真先随便填, mass=0.1)
- mesh 文件大于 10MB 不要直接放, 用 STL 简化或 collada
