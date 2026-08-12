#!/usr/bin/env bash
set -eo pipefail
set +u

export DISPLAY="${DISPLAY:-:0}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/1000/bus}"

source /opt/ros/humble/setup.bash
if [ -f "$HOME/ros2_ws/install/setup.bash" ]; then
  source "$HOME/ros2_ws/install/setup.bash"
fi

pkill -x rviz2 2>/dev/null || true
sleep 1

nohup env DISPLAY="$DISPLAY" DBUS_SESSION_BUS_ADDRESS="$DBUS_SESSION_BUS_ADDRESS" \
  nice -n 10 rviz2 -d "$HOME/tools/slam_mapping.rviz" \
  >/tmp/rviz_slam.log 2>&1 < /dev/null &

sleep 3
pgrep -fa rviz2 || true
