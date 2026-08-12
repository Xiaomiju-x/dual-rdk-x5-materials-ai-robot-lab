#!/usr/bin/env bash
set -eo pipefail
set +u

# Compatibility launcher for old demo scripts. Production ownership belongs to
# embodied_brain.service with EB_USE_LAB_FSD_SHADOW=true. This script must never
# create a second planner beside the systemd-managed instance.

ACTION="${1:-start}"
STATE_DIR="$HOME/.cache/embodied_brain"
PID_FILE="$STATE_DIR/lab_fsd_shadow.pid"
LOG_FILE="/tmp/lab_fsd_shadow.log"

source /opt/ros/humble/setup.bash
if [ -f /opt/tros/humble/setup.bash ]; then
  source /opt/tros/humble/setup.bash
fi
if [ -f "$HOME/ros2_ws/install/setup.bash" ]; then
  source "$HOME/ros2_ws/install/setup.bash"
fi
export LD_LIBRARY_PATH="/opt/tros/humble/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$STATE_DIR"

service_active() {
  systemctl is-active --quiet embodied_brain.service 2>/dev/null
}

planner_visible() {
  timeout 5 ros2 node list 2>/dev/null \
    | grep -qx '/lab_fsd_bev_shadow_planner'
}

standalone_parent_pids() {
  ps -eo pid=,ppid=,args= \
    | awk '$2 == 1 && $0 ~ /ros2 launch my_robot_navigation lab_fsd_shadow.launch.py/ {print $1}'
}

stop_standalone() {
  local pid=""
  if [ -s "$PID_FILE" ]; then
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      kill -INT "$pid" 2>/dev/null || true
    fi
  fi
  for pid in $(standalone_parent_pids); do
    kill -INT "$pid" 2>/dev/null || true
  done
  sleep 2
  rm -f "$PID_FILE"
}

show_status() {
  if service_active && planner_visible; then
    echo "LAB_FSD_SHADOW_ACTIVE owner=embodied_brain.service"
    return 0
  fi
  if planner_visible; then
    echo "LAB_FSD_SHADOW_ACTIVE owner=standalone"
    return 0
  fi
  echo "LAB_FSD_SHADOW_INACTIVE"
  return 1
}

case "$ACTION" in
  status)
    show_status
    ;;
  stop)
    if service_active && planner_visible; then
      echo "REFUSE_STOP: Lab-FSD is owned by embodied_brain.service"
      echo "Use sudo systemctl stop embodied_brain.service only during an approved maintenance window."
      exit 3
    fi
    stop_standalone
    echo "LAB_FSD_SHADOW_STOPPED owner=standalone"
    ;;
  start)
    if service_active; then
      for _ in $(seq 1 8); do
        if planner_visible; then
          echo "LAB_FSD_SHADOW_ALREADY_MANAGED owner=embodied_brain.service"
          exit 0
        fi
        sleep 1
      done
      echo "ERROR: embodied_brain.service is active but its Lab-FSD planner is absent"
      echo "Check EB_USE_LAB_FSD_SHADOW=true and restart the service; standalone duplication is refused."
      exit 4
    fi

    stop_standalone
    nohup ros2 launch my_robot_navigation lab_fsd_shadow.launch.py \
      > "$LOG_FILE" 2>&1 < /dev/null &
    echo "$!" > "$PID_FILE"
    sleep 5
    if ! planner_visible; then
      echo "ERROR: standalone Lab-FSD failed to become visible; see $LOG_FILE"
      exit 5
    fi
    ros2 node list > /tmp/lab_fsd_nodes.txt 2>&1 || true
    ros2 topic list > /tmp/lab_fsd_topics.txt 2>&1 || true
    echo "LAB_FSD_SHADOW_STARTED owner=standalone pid=$(cat "$PID_FILE")"
    echo "logs: $LOG_FILE /tmp/lab_fsd_nodes.txt /tmp/lab_fsd_topics.txt"
    ;;
  *)
    echo "Usage: $0 [start|stop|status]" >&2
    exit 2
    ;;
esac
