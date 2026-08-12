# 系统遥测, 周期 1Hz, 给 AI 脑 dashboard 显示具身脑健康状况
# 来源: my_robot_agents/telemetry_publisher

std_msgs/Header header

# === 资源 ===
float32 cpu_pct       # CPU 占用百分比 (0~100, 4 核平均)
float32 ram_used_gb   # RAM 已用 (GB)
float32 ram_total_gb  # RAM 总量 (GB)
float32 bpu_pct       # BPU 利用率 (0~100, hobot_dnn_status)
float32 cma_used_mb   # CMA 已用 (MB, 上限 391MB)
float32 cpu_temp_c    # CPU 温度 (°C, /sys/class/thermal)
float32 battery_pct   # 电池电量 (0~100), 没电池就填 -1

# === 网络 ===
bool ai_brain_reachable     # ping AI 脑 198.51.100.103 是否通
float32 ai_brain_latency_ms # 上次 ping 延迟
string current_wifi_ssid    # 当前连的 wifi 名

# === ROS2 状态 ===
bool slam_active            # SLAM 是否在跑
bool nav2_active            # Nav2 是否激活
string nav2_state           # 'IDLE' / 'NAVIGATING' / 'PAUSED' / 'ERROR'

# === 当前位姿 (map frame) ===
geometry_msgs/Pose current_pose
float32 distance_traveled_m  # 累计行驶距离 (m)
