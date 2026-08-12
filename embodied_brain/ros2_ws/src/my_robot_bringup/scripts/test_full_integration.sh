#!/bin/bash
# test_full_integration.sh — 一键 full.launch.py 集成测试 (mock AI 脑 + 全栈)
# 在车载脑 X5 上跑

set -e

source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

# 1. 启 mock AI 脑
setsid python3 ~/ros2_ws/src/my_robot_bridge/scripts/mock_ai_brain.py --port 8889 \
    > /tmp/mock.log 2>&1 < /dev/null &
MOCK_PID=$!
sleep 1

echo "=== 启 full.launch.py (传感器 + SLAM + bridge + lift_camera, 不开 furnace/nav2) ==="
setsid ros2 launch my_robot_bringup full.launch.py \
  ai_brain_url:=http://127.0.0.1:8889 \
  use_furnace:=false use_nav2:=false use_lift_camera:=true \
  > /tmp/full.log 2>&1 < /dev/null &
PID=$!

echo "等 25 秒让所有节点起好..."
sleep 25

echo ""
echo "=== 节点列表 ==="
ros2 node list 2>&1 | sort

echo ""
echo "=== 关键 topic ==="
ros2 topic list 2>&1 | grep -e scan -e odom -e map -e telemetry -e alarm -e furnace -e cmd_vel -e depth_camera -e lift_camera -e tf | sort

echo ""
echo "=== 关键 service ==="
ros2 service list 2>&1 | grep -e interpret -e set_electromagnet -e lift_home | head -10

echo ""
echo "=== 关键 action ==="
ros2 action list 2>&1 | head -10

echo ""
echo "=== /scan @ ==="
timeout 5 ros2 topic hz /scan 2>&1 | grep average | head -1

echo "=== /odom @ ==="
timeout 5 ros2 topic hz /odom 2>&1 | grep average | head -1

echo "=== /system_telemetry @ ==="
timeout 5 ros2 topic hz /system_telemetry 2>&1 | grep average | head -1

echo "=== /map @ ==="
timeout 8 ros2 topic hz /map 2>&1 | grep average | head -1

echo "=== /lift_camera/image_raw @ ==="
timeout 5 ros2 topic hz /lift_camera/image_raw 2>&1 | grep average | head -1

echo "=== /depth_camera/depth/image_raw @ ==="
timeout 5 ros2 topic hz /depth_camera/depth/image_raw 2>&1 | grep average | head -1

echo ""
echo "=== /system_telemetry 一帧详细 (验 telemetry_publisher 工作) ==="
timeout 3 ros2 topic echo /system_telemetry --once --no-arr 2>&1 | head -25

echo ""
echo "=== mock AI 脑收到的请求摘要 ==="
grep -E 'GET|POST' /tmp/mock.log | head -10

echo ""
echo "=== 资源占用 ==="
top -bn1 -d 0 2>&1 | head -5
echo ""
free -h | head -2

# 清理
kill -- -$PID 2>/dev/null || true
kill $MOCK_PID 2>/dev/null || true
sleep 3
echo ""
echo "=== 测试结束 ==="
