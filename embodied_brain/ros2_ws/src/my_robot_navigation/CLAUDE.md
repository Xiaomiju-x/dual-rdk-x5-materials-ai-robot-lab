# my_robot_navigation

> SLAM (slam_toolbox) + Nav2 自主导航配置. **Phase 2 完工 2026-04-26 (X5 `198.51.100.85` 实测建图通)**. Round 4 C2 MPPI BPU cost MLP 接入点放在 my_robot_agents/mppi_node.py, 跟 Nav2 controller 互斥 (`use_nav2` 与 `use_mppi` 二选一).

## 是什么

集成两套现成的 ROS2 stack:
- **slam_toolbox** (online_async 模式): 实时建图 + 定位 + 回环检测
- **Nav2** (humble): 全局规划 + 局部规划 + 多层 costmap + 行为树

本包**只放 launch + 参数 yaml**, 不写代码 (fake_odom 是 Python 节点放在 my_robot_agents 包).

## 目录

```
my_robot_navigation/
├── launch/
│   ├── slam.launch.py                   slam_toolbox 建图 (✅ X5 实测)
│   ├── depth_to_laserscan.launch.py     Astra 深度 → /scan_depth (✅ launch 完工)
│   ├── nav2.launch.py                   Nav2 完整栈 (写好, 等真车实测)
│   └── full_nav.launch.py               slam + nav2 + d2l 一键
├── config/
│   ├── slam_toolbox_online_async.yaml   slam_toolbox 配置
│   └── nav2_params.yaml                 Nav2 主配置 (含 4 costmap layer)
└── (后期加: maps/, behavior_tree.xml, ekf.yaml)
```

## 关键设计

### 数据流

```
LD14 雷达 ─► /scan         ┐
Astra Pro ─► /depth/image  ─► depthimage_to_laserscan ─► /scan_depth   ┐
              + /points    ─► (直接)                                      ├─► Nav2 voxel + obstacle layer
                                                                          │
slam_toolbox 订阅 /scan + /odom + URDF TF                                  │
   └─► 发 /map + map→odom TF                                              │
                                                                          │
Nav2 订阅 /scan + /scan_depth + /points + /map + TF                      ◄┘
   ├─► /cmd_vel ───────────────────────► serial_f407 ───► STM32 步进电机
   └─► (lifecycle managed: planner/controller/bt/behaviors)
```

### TF 树要求

```
map → odom → base_footprint → base_link → laser_link / depth_camera_optical_frame / ...
```

- `map → odom`: **slam_toolbox 发布** (定位)
- `odom → base_footprint`: **fake_odom (临时) 或 serial_f407_node (真车)** 发布
- `base_footprint → 各 link`: my_robot_description URDF + robot_state_publisher 发布

### Nav2 双 /scan 输入 (按 ADR-EB-2)

LD14 主, Astra Pro 辅. Nav2 局部 costmap 配 voxel_layer:
- `scan_lidar`: /scan (LD14 单线)
- `scan_depth`: /scan_depth (Astra 深度图水平条带投影, 看不到的桌脚靠这个)
- `pointcloud`: /depth_camera/depth/points (3D voxel layer, 看到柜子横梁)

### Nav2 关键参数 (按本机器人调整)

- **robot_radius**: 0.30m (含立柱外伸)
- **max_vel_x**: 0.30 m/s (步进电机不快)
- **max_vel_theta**: 1.0 rad/s
- **planner**: NavfnPlanner (默认; SmacPlanner2D 平直但 ARM 上稍慢, 留备选)
- **controller**: RegulatedPurePursuit (轮椅式底盘比 DWB 稳)
- **controller_frequency**: 20 Hz (X5 ARM 4 核扛得住)
- **resolution**: 0.05 m (5cm grid)

## X5 实测结果 (2026-04-26)

| 项目 | 状态 |
|---|---|
| /scan (LD14) | 6.15 Hz |
| /odom (fake_odom 50Hz) | 49.95 Hz |
| /map (slam_toolbox map_update_interval=1.0) | 1.0 Hz |
| TF map → base_footprint | 出现 ([0,0,0] 因车没动, 正确) |
| /map metadata | 86×174 像素 (4.3m×8.7m) @ 5cm |
| slam_toolbox 启动用时 | ~15s (Ceres solver 初始化 + 第一帧建图) |

## 验证步骤 (复现)

```bash
# X5 端
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

# 顺序启 (用 setsid 隔进程组)
setsid ros2 launch my_robot_description display.launch.py use_jsp_gui:=false &  # URDF TF
setsid ros2 run my_robot_agents fake_odom &                                     # 假 /odom (没 STM32 时)
setsid ros2 launch my_robot_drivers lidar.launch.py &                           # LD14 → /scan
setsid ros2 launch my_robot_drivers depth_camera.launch.py &                    # Astra → /depth_camera/*
setsid ros2 launch my_robot_navigation depth_to_laserscan.launch.py &           # Astra → /scan_depth
setsid ros2 launch my_robot_navigation slam.launch.py                           # slam_toolbox → /map
```

要看建图效果, PC 远程 rviz: `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ros2 run rviz2 rviz2 -d ~/ros2_ws/src/my_robot_description/rviz/display.rviz`, 加 Map display 订阅 /map.

## 已知坑

- slam_toolbox 启动后第一帧 /map 要 ~15s (Ceres solver 初始化 + scan match), 不要 timeout 5s 就以为没出来
- "minimum laser range setting (0.0 m) exceeds capabilities" 是 LD14 SDK 的 range_min 没填好的 warning, 不影响建图
- fake_odom 是临时方案, 真车上必须换成 serial_f407_node, 否则 SLAM 跟实际位置对不上
- **Nav2 8+ 个 lifecycle 节点同时启 X5 8GB RAM 用 ~3GB**, 配 swap (本机暂没配), 真测时注意监控
- Nav2 mapping 模式 (slam_toolbox 在跑) 不需要 amcl, 但 nav2_params.yaml 还是带上以备 localization 模式切换

## 下一步 (Phase 6+)

Phase 6 烧 STM32F407 固件后:
1. 关 fake_odom, 启 serial_f407_node 接真 odom
2. ros2 run nav2_smoother / nav2_collision_monitor 加 EKF (robot_localization) 融合 IMU + 轮式 odom
3. RViz "2D Goal Pose" 戳目标, 看 Nav2 真规划
4. 录 rosbag 复现 + 调参
