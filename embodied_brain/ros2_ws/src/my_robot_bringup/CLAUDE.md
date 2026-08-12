# my_robot_bringup

> 一键启动整个具身脑 ROS2 软件栈. **Phase 9 完工 2026-04-26** + **Round 4 BPU Sprint 节点全集成 2026-04-30 (X5 `198.51.100.85` 实测全过)**

## 是什么

收尾整合包. 上电 30 秒内拉起所有节点 (驱动 + SLAM/Nav2 + 感知 + Agent + 桥), 一个 launch 文件搞定.

## 文件

```
my_robot_bringup/
├── package.xml + CMakeLists.txt
├── launch/
│   └── full.launch.py             ← 一键启动主入口
├── config/
│   └── embodied_brain.service     ← systemd unit (开机自启)
└── scripts/
    ├── install_systemd.sh         ← 把 service 装到 /etc/systemd/system/
    └── test_full_integration.sh   ← 集成测试脚本
```

## full.launch.py 启动顺序

```
1. URDF + robot_state_publisher + joint_state_publisher  (TF 树)
2. fake_odom (默认开, Phase 6 烧 STM32 后改 use_fake_odom:=false)
3. 4 路传感器驱动 (sensors.launch.py)
   ├─ LD14 雷达
   ├─ Astra Pro 深度相机
   ├─ 200W USB 升降台相机 (默认 false)
   └─ serial_F407 (默认 false, 跟 fake_odom 互斥)
4. SLAM 导航 (默认开)
   ├─ depth_to_laserscan
   ├─ slam_toolbox
   └─ Nav2 (默认 false, Phase 6 真车后开)
5. 烧结炉 Agent (默认 false, Phase 7 云台拉流后开)
   ├─ furnace_ocr_node
   ├─ furnace_monitor_agent
   └─ alert_dispatcher
6. 跨网通信桥 (默认开)
   ├─ dispatch_server (DispatchTask action server)
   ├─ ai_brain_bridge (HTTP 拉 AI 脑 task)
   └─ telemetry_publisher (1Hz)
7. command_interpreter (常驻 service, 默认 backend=rule)
```

## 启动参数

| 参数 | 默认 | 含义 |
|---|---|---|
| `use_fake_odom` | true | 没 STM32 时假 odom; Phase 6 后改 false |
| `use_serial_f407` | false | STM32F407 烧好后改 true |
| `use_lidar` | true | LD14 雷达 |
| `use_depth_camera` | true | Astra Pro |
| `use_lift_camera` | false | 200W USB (Phase 6 取料时开) |
| `use_slam` | true | slam_toolbox 建图 |
| `use_nav2` | false | Nav2 (要真车闭环) |
| `use_depth_scan` | true | depth_to_laserscan |
| `use_furnace` | false | 烧结炉 OCR (要云台拉流) |
| `use_bridge` | true | 跨网 HTTP 桥 + telemetry |
| `ai_brain_url` | env: EB_AI_BRAIN_URL | AI 脑 dashboard URL |
| `use_yolo_world` | true | Round 4 A1 BPU YOLO-World 开放词检测 |
| `use_edgesam` | false | Round 4 A2 BPU EdgeSAM 像素级分割 |
| `use_xfeat` | false | Round 4 D1 BPU XFeat 视觉关键点 (985KB/17ms) |
| `use_mppi` | false | Round 4 C2 BPU MPPI cost MLP (264KB/1.14ms) |
| `use_bottle_ocr` | false | Round 4 A4 BPU PP-OCRv4 det 6ms + PaddleOCR rec |
| `use_audio` | false | Round 4 E1/E2/E3 M260C 麦阵 (RNNoise + ODAS DOA) |
| `alsa_device` | hw:2,0 | M260C ALSA 设备号 (`aplay -l` 找 XFMDPV) |
| `use_voice` | false | Round 4 B1+B2 SenseVoice ASR + Piper TTS |

## 部署到 X5

```bash
# 1. PC: 推代码
cd ~/Desktop/xrd/embodied_brain
bash deploy_to_car.sh nostart

# 2. X5: 编译
ssh rdk@198.51.100.85
cd ~/ros2_ws && colcon build --symlink-install --merge-install

# 3. X5: 装 systemd (一次性)
sudo bash ~/ros2_ws/src/my_robot_bringup/scripts/install_systemd.sh

# 4. 之后开机自启, 手动管理:
sudo systemctl status embodied_brain     # 状态
sudo systemctl start embodied_brain      # 启
sudo systemctl stop embodied_brain       # 停
sudo systemctl restart embodied_brain    # 重启
journalctl -u embodied_brain -f          # 看实时日志
```

## X5 实测结果 (2026-04-26)

✅ **集成测试** `test_full_integration.sh` 通过:

启动后 25 秒内全栈起来, 关键指标:

| 项目 | 结果 |
|---|---|
| 节点数 | 10+ (RSP/JSP/fake_odom/ld14/astra/d2l/slam/dispatch/bridge/telemetry/cmd_interp) |
| 关键 topic | /scan, /scan_depth, /odom, /map, /tf, /tf_static, /system_telemetry, /depth_camera/* 全部 ✓ |
| 关键 service | /interpret_command, /set_electromagnet, /lift_home (待 STM32) |
| 关键 action | /dispatch_task ✓ |
| /odom 频率 | 49.925 Hz (fake_odom 50Hz target) |
| 跨网通信 | mock AI 脑持续收 GET /dispatch_queue + POST /report |
| CPU | 9.4% 平均 |
| RAM | 1.0 GB / 6.9 GB (留余量 5.8 GB) |

## 开机自启 + 健康监控

```bash
# 启了几次?
systemctl show embodied_brain | grep -E 'NRestarts|ActiveState|MainPID'

# 上次启动用了多久?
journalctl -u embodied_brain --since today | grep 'Started\|started\|seconds'

# 实时看 cpu/mem
top -p $(pgrep -d',' -f 'ros2|astra|ldlidar|slam_toolbox')
```

## 已知坑

- `display.launch.py` 含 rviz2, X5 server 没装 rviz2 → full.launch.py 不 include 它, 直接起 RSP+JSP
- `Command(['xacro ', urdf_path])` 中 'xacro ' 后必须有空格 (Python 字符串拼接陷阱)
- systemd `Type=simple` + `KillMode=mixed` 让 ROS2 launch 子进程能被一起 kill (默认 process 模式只 kill 主进程, 子节点变孤儿)
- `MemoryMax=6500M` 防 OOM 把系统拖挂; 没装 swap 时硬限更重要
- 启动 25 秒是 X5 ARM CPU 慢 (大量 launch import + DDS discovery 排队); SSD 冷启动可能更慢, 加大 ExecStartPre sleep

## 下一步

Phase 6 烧 STM32F407 固件后:
1. `use_fake_odom:=false use_serial_f407:=true`
2. dispatch_server `stub_mode:=false`, 接通真 Nav2 + lift + electromagnet
3. RViz "2D Goal Pose" 戳目标, 看真车走

Phase 7 小米云台拉流后:
1. 启 pt_camera 拉流节点
2. `use_furnace:=true` 让烧结炉 OCR 链路接通真画面

Phase 8 备选: 用 USB 1080P + S20 二维舵机 (J 失败时的 fallback).
