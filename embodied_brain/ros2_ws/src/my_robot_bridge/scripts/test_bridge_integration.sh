#!/bin/bash
# test_bridge_integration.sh — 跨网通信集成测试 (mock AI 脑 + ai_brain_bridge + dispatch_server + telemetry_publisher).
#
# 运行场景:
#   1. mock AI 脑 起在 localhost:8889 (避开真 AI 脑端口)
#   2. ai_brain_bridge 拉 mock 的 dispatch_queue → DispatchTask action goal
#   3. dispatch_server 接 task, stub 执行 7 个 stage (~10 秒)
#   4. dispatch_server 完成 → bridge POST /api/embodied/report → mock 打印
#   5. telemetry_publisher 5 秒发一次 telemetry → mock 打印 cpu/ram/slam/nav
#
# 在车载脑 X5 上运行
set -e

source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

# 1. 启 mock AI 脑后台
MOCK_LOG=/tmp/mock_ai_brain.log
> "$MOCK_LOG"
setsid python3 ~/ros2_ws/src/my_robot_bridge/scripts/mock_ai_brain.py \
  --port 8889 > "$MOCK_LOG" 2>&1 < /dev/null &
MOCK_PID=$!
sleep 2

echo "=== mock AI 脑启好了 (port 8889 PID $MOCK_PID) ==="
curl -s http://127.0.0.1:8889/api/health
echo ""

# 2. 启 bridge.launch (指向 mock)
BRIDGE_LOG=/tmp/bridge.log
> "$BRIDGE_LOG"
setsid ros2 launch my_robot_bridge bridge.launch.py \
  ai_brain_url:=http://127.0.0.1:8889 \
  poll_interval_s:=2.0 \
  report_interval_s:=3.0 \
  stub_mode:=true > "$BRIDGE_LOG" 2>&1 < /dev/null &
BRIDGE_PID=$!

echo ""
echo "=== bridge launch 启动中, 等 25 秒 (足够完成 fetch_sample 7 stages 大概 13s) ==="
sleep 25

echo ""
echo "=== mock AI 脑收到的请求 ==="
cat "$MOCK_LOG" | grep -E '\[mock\]' | head -30

echo ""
echo "=== bridge launch 关键日志 ==="
grep -E 'task|stage|started|REJECTED|DONE' "$BRIDGE_LOG" | head -25

# 3. 清理
kill -- -$BRIDGE_PID 2>/dev/null || true
kill $MOCK_PID 2>/dev/null || true
sleep 2

echo ""
echo "=== 集成测试结束 ==="
