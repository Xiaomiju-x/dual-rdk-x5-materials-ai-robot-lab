#!/usr/bin/env bash
set -eo pipefail
set +u

export DISPLAY="${DISPLAY:-:0}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/1000/bus}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export LD_LIBRARY_PATH="/opt/tros/humble/lib:${LD_LIBRARY_PATH:-}"

chmod +x "$HOME/tools/restart_slam_mapping_desktop.sh" \
         "$HOME/tools/start_lab_fsd_shadow.sh" \
         "$HOME/tools/start_slam_wasd_mapper_desktop.sh" \
         "$HOME/tools/start_slam_rviz_desktop.sh" 2>/dev/null || true

"$HOME/tools/restart_slam_mapping_desktop.sh"
"$HOME/tools/start_lab_fsd_shadow.sh" || true

source /opt/ros/humble/setup.bash
if [ -f /opt/tros/humble/setup.bash ]; then
  source /opt/tros/humble/setup.bash
fi
if [ -f "$HOME/ros2_ws/install/setup.bash" ]; then
  source "$HOME/ros2_ws/install/setup.bash"
fi

mkdir -p "$HOME/video2_sessions"
ros2 node list > "$HOME/video2_sessions/prepare_nodes.txt" 2>&1 || true
ros2 topic list > "$HOME/video2_sessions/prepare_topics.txt" 2>&1 || true

echo "VIDEO2_PREPARE_READY"
echo "WASD terminal and RViz should be visible on the embodied X5 desktop."
echo "Next: ~/tools/video2_start_capture.sh"
