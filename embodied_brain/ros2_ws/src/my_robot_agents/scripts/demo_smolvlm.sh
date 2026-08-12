#!/bin/bash
# demo_smolvlm.sh — 答辩 SmolVLM C1 一键 demo (Round 4 Day 15)
#
# 演示 5 个场景:
#   1. /vlm_query 直调 (LCD 359 单行) — VLM 读出 "359"
#   2. /asr/text 触发词 → /tts/say (语音闭环)
#   3. /dispatch_task observe task type — Nav2 + VLM 综合
#   4. CommandInterpreter 中文人话 "看一下 1 号炉" → observe task
#   5. 多 LCD 场景 (单行/多行) — 体现"VLM 不是 OCR 的替代", 但语义正确
#
# 前置:
#   - X5 装好 SmolVLM-256M 在 ~/smolvlm_256m/
#   - 测试图 /tmp/lab_test_*.png (val_00010/50/100/200)
#   - smolvlm_full.launch.py 已起 (cold start 约 60s)
#
# 跑法:
#   bash ~/ros2_ws/src/my_robot_agents/scripts/demo_smolvlm.sh

set -e
LOG=/tmp/c1_demo_$(date +%H%M%S).log
TS=$(date +%H:%M:%S)

source ~/ros2_ws/install/setup.bash

echo "=================================================="
echo "  SmolVLM-256M C1 Demo  ($TS)"
echo "  Round 4 Day 15 — 答辩流程 5 个场景"
echo "=================================================="
echo "log: $LOG"
echo

if ! ros2 service list 2>/dev/null | grep -q '^/vlm_query$'; then
    echo "[ERROR] /vlm_query 服务未就绪. 先跑:"
    echo "    ros2 launch my_robot_agents smolvlm_full.launch.py"
    exit 1
fi

# ------------ Scene 1: 直调 /vlm_query (LCD 359) ------------
echo "[1/5] /vlm_query 直调 (LCD 359 单行)..."
python3 /tmp/test_vlm_query.py /tmp/lab_test_359.png 'What number is shown on the LCD?' 20 2>&1 | tee -a $LOG | grep -E 'answer|elapsed' || true
sleep 1

# ------------ Scene 2: ASR → VLM → TTS ------------
echo
echo "[2/5] /asr/text → vlm_voice_relay → /tts/say"
echo "  (publish '看一下 LCD' to /asr/text, monitor /tts/say up to 90s...)"
( timeout 90 ros2 topic echo /tts/say std_msgs/msg/String --once >> $LOG 2>&1 ) &
TTS_PID=$!
sleep 1
ros2 topic pub --once /asr/text std_msgs/msg/String "{data: '看一下 LCD'}" 2>&1 | tee -a $LOG | tail -2
wait $TTS_PID 2>/dev/null || true
echo "  /tts/say first msg captured in $LOG"
sleep 1

# ------------ Scene 3: DispatchTask observe ------------
echo
echo "[3/5] DispatchTask 'observe' (Nav2 stub + VLM)"
python3 /tmp/test_observe_action.py /tmp/lab_test_359.png 'What number is on the LCD?' 2>&1 | tee -a $LOG | grep -E 'stage|done|message|VLM 答' || true
sleep 1

# ------------ Scene 4: CommandInterpreter 中文 ------------
echo
echo "[4/5] CommandInterpreter '看一下 1 号炉子' → observe + furnace_1"
python3 -c "
from my_robot_agents.command_interpreter import RuleInterpreter
r = RuleInterpreter().parse('看一下 1 号炉子')
print(f'  task_type={r.task_type} to={r.to_location} prompt={r.bottle_id!r}')
" 2>&1 | tee -a $LOG
sleep 1

# ------------ Scene 5: 多 LCD 场景 ------------
echo
echo "[5/5] 多 LCD 场景 batch (单行/多行)..."
python3 /tmp/test_lab_scenes.py 2>&1 | tee -a $LOG | tail -10
sleep 1

echo
echo "=================================================="
echo "Done. 完整 log: $LOG"
echo "=================================================="
