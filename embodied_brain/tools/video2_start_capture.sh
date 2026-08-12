#!/usr/bin/env bash
set -eo pipefail
set +u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export LD_LIBRARY_PATH="/opt/tros/humble/lib:${LD_LIBRARY_PATH:-}"

source /opt/ros/humble/setup.bash
if [ -f /opt/tros/humble/setup.bash ]; then
  source /opt/tros/humble/setup.bash
fi
if [ -f "$HOME/ros2_ws/install/setup.bash" ]; then
  source "$HOME/ros2_ws/install/setup.bash"
fi

base="$HOME/video2_sessions"
mkdir -p "$base"
session="$base/video2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$session"
rm -f "$session/STOP"

echo "$session" > "$base/current_session.txt"
fps="${VIDEO2_FPS:-2}"
echo "$fps" > "$session/fps.txt"

nohup python3 "$HOME/tools/video2_overlay_recorder.py" \
  --out "$session" \
  --fps "$fps" \
  --ai-url "${AI_BRAIN_URL:-http://192.0.2.103:8888}" \
  --stop-file "$session/STOP" \
  > "$session/recorder.log" 2>&1 < /dev/null &

echo "$!" > "$session/recorder.pid"
echo "VIDEO2_CAPTURE_STARTED"
echo "$session"
echo "Stop with: ~/tools/video2_stop_capture.sh"
