#!/usr/bin/env bash
# Start or inspect the embodied v3 runtime stack on the car X5.
#
# This is the non-desktop counterpart to video2_prepare_demo.sh. It starts the
# ROS stack needed by embodied_v3_acceptance_check.sh and writes plain evidence
# files under /tmp. It does not publish /cmd_vel and does not open RViz.

set -eo pipefail
set +u

ACTION="start"
RESTART=1
START_WASD=0
USE_MPPI_PROPOSED="${EMBODIED_V3_USE_MPPI_PROPOSED:-true}"
SETTLE_S="${EMBODIED_V3_SETTLE_S:-35}"
LOG_FILE="${EMBODIED_V3_STACK_LOG:-/tmp/embodied_v3_stack.log}"
STATUS_FILE="${EMBODIED_V3_STACK_STATUS:-/tmp/embodied_v3_stack_status.txt}"
PID_FILE="${EMBODIED_V3_STACK_PID:-/tmp/embodied_v3_stack.pid}"

usage() {
  cat <<'USAGE'
Usage:
  start_embodied_v3_stack.sh [start|status|stop] [options]

Actions:
  start                 Start full.launch in embodied-v3 acceptance mode.
  status                Print ROS node/topic/topic-sample status only.
  stop                  Stop the stack processes started by this helper.

Options:
  --no-restart          Do not kill existing matching ROS processes before start.
  --with-wasd           Also start the keyboard WASD mapper in the background.
  --without-mppi        Disable the BPU MPPI proposed-only node for this run.
  --settle-s SEC        Wait this many seconds before status sampling. Default: 35.
  -h, --help            Show this help.

Started stack:
  ros2 launch my_robot_bringup full.launch.py
    use_fake_odom:=false use_serial_f407:=true use_slam:=true
    use_nav2:=false stub_mode:=true use_lab_fsd_shadow:=true
    use_mppi:=true mppi_publish_direct_cmd_vel:=false

Safety:
  This helper never publishes /cmd_vel. Only --with-wasd starts the teleop tool.
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    start|status|stop)
      ACTION="$1"
      ;;
    --no-restart)
      RESTART=0
      ;;
    --with-wasd)
      START_WASD=1
      ;;
    --without-mppi)
      USE_MPPI_PROPOSED=false
      ;;
    --settle-s)
      shift || {
        echo "ERR --settle-s needs a value" >&2
        exit 2
      }
      SETTLE_S="${1:-35}"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERR unknown option/action: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

setup_ros() {
  export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
  export LD_LIBRARY_PATH="/opt/tros/humble/lib:${LD_LIBRARY_PATH:-}"
  if [ -f /opt/ros/humble/setup.bash ]; then
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
  fi
  if [ -f /opt/tros/humble/setup.bash ]; then
    # shellcheck disable=SC1091
    source /opt/tros/humble/setup.bash
  fi
  if [ -f "$HOME/ros2_ws/install/setup.bash" ]; then
    # shellcheck disable=SC1091
    source "$HOME/ros2_ws/install/setup.bash"
  fi
  command -v ros2 >/dev/null 2>&1 || {
    echo "ERR ros2 not found after sourcing ROS environments" >&2
    exit 3
  }
}

stop_stack() {
  local patterns=(
    "ros2 launch my_robot_bringup full.launch.py"
    "lab_fsd_bev_shadow_planner"
    "lab_fsd_vision_bev_bridge"
    "serial_f407_node"
    "async_slam_toolbox_node"
    "sync_slam_toolbox_node"
    "depthimage_to_laserscan_node"
    "ldlidar_stl_ros2_node"
    "astra_camera_node"
    "robot_state_publisher"
    "joint_state_publisher"
    "dispatch_server"
    "ai_brain_bridge"
    "telemetry_publisher"
    "command_interpreter"
    "location_visualizer"
    "mppi_node"
  )
  local pattern
  for pattern in "${patterns[@]}"; do
    pkill -f "$pattern" 2>/dev/null || true
  done
  rm -f "$PID_FILE"
}

sample_topic() {
  local topic="$1"
  local safe
  safe="$(echo "$topic" | sed 's#^/##; s#[/ ]#_#g')"
  timeout 8 ros2 topic echo --once "$topic" > "/tmp/embodied_v3_topic_${safe}.txt" 2>&1
}

write_status() {
  : > "$STATUS_FILE"
  {
    echo "EMBODIED_V3_STACK_STATUS"
    date -Is
    echo "log: $LOG_FILE"
    echo "pid_file: $PID_FILE"
    echo
    echo "== process =="
    if [ -f "$PID_FILE" ]; then
      local pid
      pid="$(cat "$PID_FILE" 2>/dev/null || true)"
      echo "pid: $pid"
      if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "process: alive"
      else
        echo "process: not-alive"
      fi
    else
      echo "pid_file: missing"
    fi
    echo
    echo "== nodes =="
    ros2 node list | sort || true
    echo
    echo "== topics =="
    ros2 topic list | sort || true
    echo
    echo "== required topic samples =="
  } >> "$STATUS_FILE"

  local required=(
    /scan
    /scan_depth
    /odom
    /map
    /lab_fsd/fsd_v3_status
    /lab_fsd/future_risk
    /lab_fsd/input_status
    /lab_fsd/safety_gate
    /lab_fsd/shadow_path
    /lab_fsd/trajectory_scores
    /lab_fsd/bev
    /lab_fsd/future_bev
    /lab_fsd/policy_tokens
    /diagnostics
    /lift_status
    /f407/estop_latched
    /f407/cmd_vel_expired
    /f407/firmware_identity_valid
    /f407/firmware_info
  )
  if [ "$USE_MPPI_PROPOSED" = "true" ]; then
    required+=(/mppi/cmd_vel_proposed /mppi/stats)
  fi
  local topic
  local ok_count=0
  local fail_count=0
  for topic in "${required[@]}"; do
    if sample_topic "$topic"; then
      echo "OK $topic" >> "$STATUS_FILE"
      ok_count=$((ok_count + 1))
    else
      echo "MISS $topic" >> "$STATUS_FILE"
      fail_count=$((fail_count + 1))
    fi
  done
  {
    echo
    echo "summary: OK=${ok_count} MISS=${fail_count}"
    echo "done: $STATUS_FILE"
  } >> "$STATUS_FILE"
  cat "$STATUS_FILE"
  [ "$fail_count" -eq 0 ]
}

setup_ros

case "$ACTION" in
  stop)
    stop_stack
    echo "EMBODIED_V3_STACK_STOPPED"
    exit 0
    ;;
  status)
    write_status
    exit $?
    ;;
esac

if [ "$RESTART" = "1" ]; then
  stop_stack
  sleep 3
fi

mkdir -p "$HOME/maps"
: > "$LOG_FILE"
nohup ros2 launch my_robot_bringup full.launch.py \
  use_fake_odom:=false \
  use_serial_f407:=true \
  use_slam:=true \
  use_nav2:=false \
  stub_mode:=true \
  use_lab_fsd_shadow:=true \
  use_lift_camera:=false \
  use_pt_camera:=false \
  use_yolo_world:=false \
  use_edgesam:=false \
  use_xfeat:=false \
  use_mppi:="$USE_MPPI_PROPOSED" \
  mppi_publish_direct_cmd_vel:=false \
  mppi_cmd_vel_topic:=/mppi/cmd_vel_proposed \
  use_bottle_ocr:=false \
  > "$LOG_FILE" 2>&1 < /dev/null &
echo "$!" > "$PID_FILE"

echo "EMBODIED_V3_STACK_STARTING"
echo "pid: $(cat "$PID_FILE")"
echo "log: $LOG_FILE"
echo "settle_s: $SETTLE_S"
sleep "$SETTLE_S"

if [ "$START_WASD" = "1" ]; then
  if [ -x "$HOME/tools/start_slam_wasd_mapper.sh" ] || [ -f "$HOME/tools/start_slam_wasd_mapper.sh" ]; then
    nohup "$HOME/tools/start_slam_wasd_mapper.sh" \
      > /tmp/slam_wasd_mapper.log 2>&1 < /dev/null &
    echo "wasd: started /tmp/slam_wasd_mapper.log"
  else
    echo "wasd: missing $HOME/tools/start_slam_wasd_mapper.sh"
  fi
fi

write_status
