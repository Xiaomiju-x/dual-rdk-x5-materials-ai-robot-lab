#!/usr/bin/env bash
set -eo pipefail
set +u

export DISPLAY="${DISPLAY:-:0}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/1000/bus}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export LD_LIBRARY_PATH="/opt/tros/humble/lib:${LD_LIBRARY_PATH:-}"

if [ ! -x "$HOME/tools/restart_slam_mapping_desktop.sh" ]; then
  chmod +x "$HOME/tools/restart_slam_mapping_desktop.sh" 2>/dev/null || true
fi
if [ ! -x "$HOME/tools/start_lab_fsd_shadow.sh" ]; then
  chmod +x "$HOME/tools/start_lab_fsd_shadow.sh" 2>/dev/null || true
fi

"$HOME/tools/restart_slam_mapping_desktop.sh"
"$HOME/tools/start_lab_fsd_shadow.sh" || true

if command -v xfce4-terminal >/dev/null 2>&1; then
  xfce4-terminal \
    --disable-server \
    --title="VIDEO1 FIXTURE CONTROL" \
    --command="bash -lc 'echo Commands: ~/tools/video1_fixture_control.sh pick/place/status; exec bash'" \
    >/tmp/video1_fixture_terminal.log 2>&1 &
fi

echo "VIDEO1_EMBODIED_DEMO_READY"
echo "Use the WASD terminal for safety-operator driving."
echo "Fixture commands: ~/tools/video1_fixture_control.sh pick | place | home | status"
