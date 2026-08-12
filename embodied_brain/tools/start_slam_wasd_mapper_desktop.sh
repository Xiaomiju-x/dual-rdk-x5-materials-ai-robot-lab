#!/usr/bin/env bash
set -eo pipefail
set +u

export DISPLAY="${DISPLAY:-:0}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/1000/bus}"

pkill -f /home/rdk/tools/slam_wasd_mapper.py 2>/dev/null || true

if command -v xfce4-terminal >/dev/null 2>&1; then
  xfce4-terminal \
    --disable-server \
    --title="SLAM WASD CONTROL" \
    --command="bash -lc '$HOME/tools/start_slam_wasd_mapper.sh; exec bash'" \
    >/tmp/slam_wasd_terminal.log 2>&1 &
else
  nohup "$HOME/tools/start_slam_wasd_mapper.sh" \
    >/tmp/slam_wasd_mapper.log 2>&1 < /dev/null &
fi

sleep 2
pgrep -fa /home/rdk/tools/slam_wasd_mapper.py || true
