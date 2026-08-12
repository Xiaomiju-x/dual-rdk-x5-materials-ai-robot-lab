# my_robot_drivers

> 传感器 + 执行器 ROS2 驱动节点. **Phase 1 完工 2026-04-26** + **Phase 8 Part A K3 USB cam 完工 2026-04-26** (替代弃用的小米云台). X5 IP `198.51.100.85`.

## 是什么

把所有"硬件接入 → ROS2 topic"的工作集中在这个包. 上层节点 (Nav2 / SLAM / Agent) 不直接碰 USB / serial / SDK, 全走 ROS2 topic + service + action.

## 目录

```
my_robot_drivers/
├── package.xml + CMakeLists.txt
├── include/my_robot_drivers/
│   └── serial_protocol.hpp       0xAA55 多 type 帧协议定义 (上行/下行 enum + payload struct)
├── src/
│   └── serial_f407_node.cpp      F407 USB-TTL 桥节点 (✅ Phase 1)
├── launch/
│   ├── sensors.launch.py         ★ 一键拉起 4 个驱动
│   ├── lidar.launch.py           LD14 (包装 LD Robot 官方驱动)
│   ├── depth_camera.launch.py    Astra Pro (包装 ros2_astra_camera)
│   ├── lift_camera.launch.py     200W USB (用 usb_cam 包)
│   └── serial_f407.launch.py     串口节点
├── config/
│   └── lift_camera.yaml          usb_cam 参数 (1280×720 @ 30fps MJPG)
├── udev/
│   ├── 99-ld14-lidar.rules       /dev/LD14 SYMLINK
│   ├── 99-stm32-f407.rules       /dev/F407 SYMLINK (4 种 USB-TTL 都覆盖)
│   ├── 99-cameras.rules          /dev/astra_rgb /dev/lift_camera SYMLINK
│   └── install_udev.sh           一键装到 /etc/udev/rules.d/
└── scripts/
    └── install_third_party.sh    LD14 + ros2_astra_camera + usb_cam + slam_toolbox + nav2 一次性装
```

## 节点清单

| 节点 | 入口 | 来源 | 状态 (2026-04-26) |
|---|---|---|---|
| **ld14_lidar** | `ldlidar_stl_ros2_node` | LD Robot GitHub clone | ✅ **X5 实测通**, /scan @ ~6Hz, 666 点/帧 360° 0.02-25m |
| **astra_pro** | `astra_camera_node` | Orbbec GitHub clone | ✅ **X5 实测通**, depth+color+points @ 30Hz, 7 topic |
| **lift_camera** | `usb_cam_node_exe` | apt: `ros-humble-usb-cam` | ✅ **X5 实测通**, image_raw @ 30Hz 1280×720 MJPG (vid:pid 0c45:6368 Microdia OV2710) |
| **serial_f407_node** | 自写 C++ | 本包 src/ | ⏸ 等 STM32F407 烧固件 (Phase 6) |

## 0xAA55 多 type 帧协议

详细定义见 [include/my_robot_drivers/serial_protocol.hpp](include/my_robot_drivers/serial_protocol.hpp). 关键帧:

### 上行 (STM32 → ROS2)

| type | 名称 | payload | 用途 |
|---|---|---|---|
| 0x01 | BASIC_ODOM | x/y/vx/wz/yaw_deg (5 float, 20B) | 兼容 v1 demo, 走 /odom + TF |
| 0x02 | EXT_TELEMETRY | lift + magnet + IMU + voltage (44B) | 走 /imu + /lift_status |
| 0x10 | ACK | (3B) | 对下行命令的应答 |
| 0x1F | ERROR | error_code + msg | STM32 报错 |

### 下行 (ROS2 → STM32)

| type | 名称 | payload | 触发 |
|---|---|---|---|
| 0x01 | CMD_VEL | linear_v + angular_w (8B) | /cmd_vel sub |
| 0x02 | SET_LIFT_HEIGHT | target_height (4B) | /lift/target_height sub |
| 0x03 | SET_ELECTROMAGNET | turn_on (1B) | /set_electromagnet srv |
| 0x04 | LIFT_HOME | (0B) | /lift_home srv |
| 0x10 | EMERGENCY_STOP | (0B) | (Phase 6 加) |
| 0xFF | HEARTBEAT | (0B) | 5Hz 自动 |

**校验**: 简单 sum 8-bit (从 0xAA 累加到 payload 末).
**字节序**: 小端 (STM32 + ARM64 都默认), 不用 ntoh.

## 部署到车载脑 X5

```bash
# 1. 推代码 (PC 端)
cd ~/Desktop/xrd/embodied_brain
bash deploy_to_car.sh nostart    # 只推, 第一次部署不要直接重启

# 2. SSH 上 X5, 装第三方依赖 (只跑一次)
ssh rdk@198.51.100.85
cd ~/ros2_ws/src/my_robot_drivers/scripts
bash install_third_party.sh      # apt + clone LD14 + clone astra + udev

# 3. 编译
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --merge-install

# 4. 拔插 USB 让 udev 生效, 检查
ls -l /dev/LD14 /dev/F407 /dev/astra_rgb /dev/lift_camera

# 5. 启 sensors
source install/setup.bash
ros2 launch my_robot_drivers sensors.launch.py
```

## 验证

```bash
# 雷达
ros2 topic hz /scan                                      # 应 ~10 Hz
ros2 run rviz2 rviz2 -d ~/ros2_ws/src/my_robot_description/rviz/display.rviz

# 深度相机
ros2 topic hz /depth_camera/depth/image_raw              # ~30 Hz
ros2 topic echo /depth_camera/depth/points --once        # 看到点云数据

# 升降台相机
ros2 topic hz /lift_camera/image_raw                     # ~30 Hz

# F407 串口
ros2 topic echo /odom --once
ros2 topic echo /imu --once
ros2 topic echo /lift_status --once
ros2 topic pub /cmd_vel geometry_msgs/Twist '{linear: {x: 0.1}}' -1   # 让小车前进 0.1m/s
ros2 service call /set_electromagnet my_robot_msgs/srv/SetElectromagnet '{turn_on: true}'

# TF 树
ros2 run tf2_tools view_frames
```

## 已知坑 (Phase 1 踩过的)

- LD14 `port_name='/dev/LD14'`, 没装 udev 时 fallback 到 /dev/ttyACM0 但顺序不稳, **必须装 udev 规则**
- **LD14 跟新版 ldlidar_stl_ros2 兼容性**: 该 SDK 只接受 product_name `LDLiDAR_LD06/_LD19/_STL27L`, 但 LD14 跟 LD06 协议同, **配 `LDLiDAR_LD06` + 波特率 230400** 即可正常用 (v1 demo 用 115200 是错的)
- **astra_camera 编译 OOM**: PCL + cv_bridge 单文件吃几 GB, X5 8GB 多 job 并发被 OOM kill. 单独 `colcon build --packages-select astra_camera --parallel-workers 1 --executor sequential`, 4 分钟单线程编完
- **Astra Pro 是双总线相机**: 深度走 OpenNI2 (vid=2bc5:0403), RGB 走 UVC (vid=2bc5:0501). 我们的 .py launch 不能直接调 `astra_camera_node` 传参, 必须 include 官方 `astra_pro.launch.xml` (它把 use_uvc_camera + uvc_vendor_id 全搞定); 否则会报 "Failed to send a USB control request!"
- ros2_astra_camera 在 Ubuntu 22.04 ARM64 (X5 系统) 编译需要 `ros-humble-image-publisher` + `libuvc-dev libuvc0`, install_third_party.sh 已 apt 装上
- usb_cam 默认 YUYV @5fps, **必须** 在 yaml 里改 `pixel_format: mjpeg2rgb`, 否则视觉伺服会卡
- F407 串口 v1 协议 yaw 是 deg, v2 兼容沿用 (代码中转 rad 给 ROS); cmd_vel angular 是 rad/s (跟 ROS 习惯)
- `O_NONBLOCK` 打开串口很重要, 否则 read() 阻塞会卡 ROS2 spin
- F407 端协议必须**严格匹配 struct 字节布局** (用 `#pragma pack(1)`), 改字段顺序前先告诉 ROS 端
- **launch 文件改了要重 colcon build** (即使 --symlink-install): 因为 `install/share/<pkg>/launch/` 是 install 出来的拷贝不是 symlink. 修法: `colcon build --packages-select my_robot_drivers --symlink-install --merge-install`

## 后续 Phase 关联

- **Phase 2**: SLAM 用 /scan + /odom + TF (本包提供)
- **Phase 3**: OCR Agent 不直接用本包 (用云台 RTSP)
- **Phase 4**: bridge 节点会用 /imu + /odom + /lift_status 上报 AI 脑
- **Phase 6**: pickup_agent 会用 /set_electromagnet + /lift/target_height; 协议可能扩 type=0x05/0x06 给细粒度运动控制

## 2026-07-09 F407 ROS2 safety bridge 增强

范围: 只改 `serial_f407_node` 和 X5 侧工具/launch, 不改 STM32 固件和 0xAA55 协议。

- 服务命令 `/set_electromagnet`, `/lift_home`, `/estop` 发送后等待 F407 `ACK(type=0x10)`, 默认 `ack_timeout_ms=300`; 超时/非零 status 会返回 `success=false`。
- 单帧串口写入有 `write_timeout_ms=50` 上限, 避免 USB-TTL 卡住时 ROS 回调无限阻塞。
- `/cmd_vel` 不等待 ACK, 避免破坏 Nav2/teleop 连续运动链路；节点记录最后接收和最后转发时间。
- `cmd_vel_timeout_s` 默认 `0.60`: 超过时间没有新 `/cmd_vel` 时, ROS2 侧主动发送一次零速度, 并在诊断里标记 `cmd_vel_expired=true`。
- 新增 `/estop` service + `/estop` Bool topic: 置位后发送 `EMERGENCY_STOP(0x10)` 并锁存 ROS2 侧 estop; `/clear_estop` 或 `/estop false` 只清 ROS2 本地锁存并发送零速度, 不新增固件 clear 协议。
- 新增 `/f407/estop_latched`, `/f407/cmd_vel_expired` Bool topic 和 `/diagnostics` 中的 `serial_f407_node: serial_link` / `serial_f407_node: safety_bridge` 状态。

车上最小验证:

```bash
cd ~/ros2_ws
colcon build --packages-select my_robot_drivers --symlink-install --merge-install
source install/setup.bash
ros2 launch my_robot_drivers serial_f407.launch.py
ros2 topic echo /diagnostics --once
ros2 service call /estop std_srvs/srv/Trigger {}
ros2 service call /clear_estop std_srvs/srv/Trigger {}
ros2 topic echo /f407/cmd_vel_expired --once
```
