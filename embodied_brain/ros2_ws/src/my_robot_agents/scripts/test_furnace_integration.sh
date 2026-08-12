#!/bin/bash
# test_furnace_integration.sh — 烧结炉 OCR + alarm 全栈集成测试
# 在车载脑 X5 上运行
set -e

source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

echo "### 生成测试图 ###"
python3 <<'PYEOF'
import sys
sys.path.insert(0, '/home/rdk/ros2_ws/src/my_robot_agents')
from my_robot_agents.furnace_ocr import render_furnace_panel
import cv2

img1 = render_furnace_panel(pv=1350, sv=1350, mv=49.7, power_on=True)
cv2.imwrite('/tmp/furnace_test.jpg', img1)
print('正常图 /tmp/furnace_test.jpg', img1.shape)

img2 = render_furnace_panel(pv=1750, sv=1350, mv=99.9, power_on=True)
cv2.imwrite('/tmp/furnace_alarm.jpg', img2)
print('异常图 /tmp/furnace_alarm.jpg PV=1750')
PYEOF

# ===================== 场景 1: 正常 =====================
echo ""
echo "### 场景 1: PV=1350 SV=1350 不报警 ###"
setsid ros2 launch my_robot_agents furnace_monitor.launch.py \
  test_image_path:=/tmp/furnace_test.jpg \
  enable_email:=false enable_wechat:=false enable_tts:=false \
  enable_log:=true > /tmp/furnace_s1.log 2>&1 < /dev/null &
PID=$!
sleep 8

echo "-- /furnace_reading 一帧 --"
timeout 3 ros2 topic echo /furnace_reading --once --no-arr 2>&1 | head -25 || true

echo "-- 是否有 /alarm? --"
timeout 2 ros2 topic echo /alarm --once 2>&1 | head -5 || echo "(没报警 OK)"

kill -- -$PID 2>/dev/null || true
sleep 3

# ===================== 场景 2: 异常 =====================
echo ""
echo "### 场景 2: PV=1750 超 1600 阈值, I1 CRITICAL ###"
> /tmp/embodied_brain_alarms.log
setsid ros2 launch my_robot_agents furnace_monitor.launch.py \
  test_image_path:=/tmp/furnace_alarm.jpg \
  enable_email:=false enable_wechat:=false enable_tts:=false \
  enable_log:=true > /tmp/furnace_s2.log 2>&1 < /dev/null &
PID=$!
sleep 8

echo "-- /furnace_reading --"
timeout 3 ros2 topic echo /furnace_reading --once --no-arr 2>&1 | head -25 || true

echo ""
echo "-- /alarm --"
timeout 3 ros2 topic echo /alarm --once --no-arr 2>&1 | head -25 || echo "(超时)"

kill -- -$PID 2>/dev/null || true
sleep 3

echo ""
echo "### 报警 log 文件 ###"
cat /tmp/embodied_brain_alarms.log 2>/dev/null || echo "(空)"

echo ""
echo "### 各 launch 节点日志末尾 (找 ALARM 关键字) ###"
echo "-- s1 (正常场景) --"
grep -E 'ALARM|started|error' /tmp/furnace_s1.log | tail -8
echo ""
echo "-- s2 (异常场景) --"
grep -E 'ALARM|started|error|PV' /tmp/furnace_s2.log | tail -8
