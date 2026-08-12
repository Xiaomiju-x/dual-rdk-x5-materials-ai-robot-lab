#!/usr/bin/env bash
set -eo pipefail
set +u

source /opt/ros/humble/setup.bash
if [ -f /opt/tros/humble/setup.bash ]; then
  source /opt/tros/humble/setup.bash
fi
if [ -f "$HOME/ros2_ws/install/setup.bash" ]; then
  source "$HOME/ros2_ws/install/setup.bash"
fi

echo "== nodes =="
ros2 node list | grep -E 'serial_f407|slam_wasd|rviz|slam_toolbox|ld14|depth_camera|lab_fsd|video2|hobot' | sort || true
echo "== topics =="
ros2 topic list | grep -E '^/cmd_vel$|^/map$|^/odom$|^/scan$|^/scan_depth$|^/lab_fsd|^/hobot_yolo_world$' | sort || true
echo "== current session =="
cat "$HOME/video2_sessions/current_session.txt" 2>/dev/null || true
