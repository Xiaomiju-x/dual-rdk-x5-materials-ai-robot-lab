#!/usr/bin/env bash
set -eo pipefail
set +u

ACTION="${1:-prepare}"
if [ "$#" -gt 0 ]; then
  shift
fi

STATE_DIR="${FINALS_NAV_STATE_DIR:-$HOME/.cache/finals_nav_demo}"
TOOLS_DIR="${FINALS_NAV_TOOLS_DIR:-$HOME/tools}"
RVIZ_CONFIG="${FINALS_NAV_RVIZ_CONFIG:-$TOOLS_DIR/slam_mapping.rviz}"
STATUS_TOOL="$TOOLS_DIR/finals_nav_status.sh"
STATUS_PY="$TOOLS_DIR/finals_nav_status.py"
TELEOP_TOOL="$TOOLS_DIR/start_slam_wasd_mapper.sh"
SAVE_TOOL="$TOOLS_DIR/save_current_slam_map.sh"
DISPLAY="${DISPLAY:-:0}"
DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/$(id -u)/bus}"

usage() {
  cat <<'USAGE'
Usage: finals_nav_demo.sh [prepare|nav2|status|stop|restore|save]

  prepare  Safely enable the finals stack, then open RViz, WASD and status.
           This supervised mode is the default action.
  nav2     Safely enable the same stack and open RViz/status without WASD.
           Use RViz 2D Goal Pose only after the separate arming procedure.
  status   Print read-only stack, safety, sensor, 4K, and BPU provenance.
  stop     Stop only desktop processes launched by this wrapper.
  restore  Stop wrapper UI, latch estop, remove temporary finals environment,
           and restart the normal systemd stack.
  save     Save the current SLAM map through the existing safe helper.

This wrapper never clears estop and never publishes a velocity command.
USAGE
}

die() {
  echo "ERR $*" >&2
  exit 2
}

warn() {
  echo "WARN $*" >&2
}

setup_ros() {
  export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
  export LD_LIBRARY_PATH="/opt/tros/humble/lib:${LD_LIBRARY_PATH:-}"
  export DISPLAY DBUS_SESSION_BUS_ADDRESS
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
}

acquire_lock() {
  mkdir -p "$STATE_DIR"
  if command -v flock >/dev/null 2>&1; then
    exec 9> "$STATE_DIR/control.lock"
    flock -n 9 || die "another finals_nav_demo operation is in progress"
  else
    if ! mkdir "$STATE_DIR/control.lock.d" 2>/dev/null; then
      die "another finals_nav_demo operation is in progress"
    fi
    trap 'rmdir "$STATE_DIR/control.lock.d" 2>/dev/null || true' EXIT
  fi
}

service_state() {
  local scope="$1"
  if [ "$scope" = "system" ]; then
    systemctl is-active embodied_brain.service 2>/dev/null || true
  else
    systemctl --user is-active embodied_brain.service 2>/dev/null || true
  fi
}

require_single_service_owner() {
  command -v systemctl >/dev/null 2>&1 || die "systemctl is required to preserve service ownership"
  local system_state user_state
  system_state="$(service_state system)"
  user_state="$(service_state user)"
  if [ "$system_state" = "active" ] && [ "$user_state" = "active" ]; then
    die "embodied_brain.service is active in both system and user scopes; duplicate ownership refused"
  fi
  if [ "$system_state" != "active" ] && [ "$user_state" != "active" ]; then
    die "embodied_brain.service is not active; this wrapper will not create an unowned core stack"
  fi
  if [ "$system_state" = "active" ]; then
    echo "STACK_OWNER=system:embodied_brain.service"
  else
    echo "STACK_OWNER=user:embodied_brain.service"
  fi
}

configure_finals_runtime() {
  [ "$(service_state system)" = "active" ] || die "finals runtime requires the system embodied_brain.service owner"
  command -v sudo >/dev/null 2>&1 || die "sudo is required for the systemd finals profile"
  sudo -n true >/dev/null 2>&1 || die "passwordless sudo is required for one-click finals preparation"

  local expected=(
    EB_USE_NAV2=true
    EB_USE_COLLISION_MONITOR=true
    EB_USE_MPPI=true
    EB_MPPI_PUBLISH_DIRECT_CMD_VEL=false
    EB_MPPI_CMD_VEL_TOPIC=/mppi/cmd_vel_proposed
    EB_DISPATCH_STUB_MODE=true
    EB_AI_BRAIN_URL=http://192.0.2.103:8888
  )
  local current need_restart=0 item
  current="$(sudo -n systemctl show-environment 2>/dev/null || true)"
  for item in "${expected[@]}"; do
    grep -Fqx "$item" <<<"$current" || need_restart=1
  done
  if [ "$need_restart" -eq 1 ]; then
    sudo -n systemctl set-environment "${expected[@]}"
    sudo -n systemctl restart embodied_brain.service
    printf '%s\n' "${expected[@]}" > "$STATE_DIR/finals.environment"
    local deadline=$((SECONDS + ${FINALS_NAV_SERVICE_WAIT_S:-65}))
    while [ "$SECONDS" -le "$deadline" ]; do
      [ "$(service_state system)" = "active" ] && sleep 8 && break
      sleep 1
    done
    [ "$(service_state system)" = "active" ] || die "embodied_brain.service did not recover after finals profile restart"
    echo "FINALS_PROFILE=restarted"
  else
    echo "FINALS_PROFILE=already_active"
  fi
  if systemctl is-active --quiet cockpit_bridge.service 2>/dev/null; then
    sudo -n systemctl stop cockpit_bridge.service
    : > "$STATE_DIR/cockpit_bridge.was_active"
    echo "COCKPIT_SIDECAR=stopped_for_unique_cmd_source"
  else
    echo "COCKPIT_SIDECAR=not_active"
  fi
}

node_count() {
  local node="$1"
  grep -Fxc "$node" "$STATE_DIR/nodes.snapshot" 2>/dev/null || true
}

refresh_nodes() {
  timeout 5 ros2 node list > "$STATE_DIR/nodes.snapshot" 2> "$STATE_DIR/nodes.err" || true
}

require_single_core_stack() {
  local wait_s="${FINALS_NAV_WAIT_S:-18}"
  local deadline=$((SECONDS + wait_s))
  local required=(
    /serial_f407 /ld14_lidar /slam_toolbox /lab_fsd_bev_shadow_planner
    /collision_monitor /bt_navigator /mppi_node
  )
  local ready=0 node count
  while [ "$SECONDS" -le "$deadline" ]; do
    refresh_nodes
    ready=1
    for node in "${required[@]}"; do
      count="$(node_count "$node")"
      if [ "${count:-0}" -gt 1 ]; then
        die "duplicate ROS core node refused: $node count=$count"
      fi
      if [ "${count:-0}" -ne 1 ]; then
        ready=0
      fi
    done
    for node in /lab_fsd_vision_bev_bridge /depth_to_laserscan /ekf_filter_node; do
      count="$(node_count "$node")"
      if [ "${count:-0}" -gt 1 ]; then
        die "duplicate ROS node refused: $node count=$count"
      fi
    done
    if [ "$ready" -eq 1 ]; then
      break
    fi
    sleep 1
  done
  if [ "$ready" -ne 1 ]; then
    for node in "${required[@]}"; do
      printf 'core.%s count=%s\n' "${node#/}" "$(node_count "$node")" >&2
    done
    die "systemd core stack is incomplete; no second launch will be created"
  fi

  local full_launch_count
  full_launch_count="$(pgrep -af '[r]os2 launch my_robot_bringup full.launch.py' 2>/dev/null | wc -l | tr -d ' ' || true)"
  if [ "${full_launch_count:-0}" -gt 1 ]; then
    die "multiple full.launch.py parents detected: $full_launch_count"
  fi
  echo "CORE_STACK=single"
}

require_latched_estop() {
  local out="$STATE_DIR/estop.snapshot"
  if [ ! -f "$STATUS_PY" ] || ! timeout 12 python3 "$STATUS_PY" --timeout 6 > "$out" 2>&1; then
    die "cannot confirm /f407/estop_latched; prepare remains fail-closed"
  fi
  if ! grep -Fq 'safety.estop=True' "$out"; then
    die "estop is not confirmed latched; this conservative prepare refuses to open a new teleop"
  fi
  echo "SAFETY_ESTOP=latched"
}

proc_start_ticks() {
  local pid="$1"
  awk '{print $22}' "/proc/$pid/stat" 2>/dev/null || true
}

record_owned() {
  local kind="$1"
  local pid="$2"
  local pgid start
  start="$(proc_start_ticks "$pid")"
  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
  [ -n "$start" ] || die "cannot record $kind process start time for pid=$pid"
  [ -n "$pgid" ] || pgid="$pid"
  printf '%s\n' "$pid" > "$STATE_DIR/$kind.pid"
  printf '%s\n' "$start" > "$STATE_DIR/$kind.start"
  printf '%s\n' "$pgid" > "$STATE_DIR/$kind.pgid"
}

owned_alive() {
  local kind="$1"
  local pid_file="$STATE_DIR/$kind.pid"
  local start_file="$STATE_DIR/$kind.start"
  [ -s "$pid_file" ] && [ -s "$start_file" ] || return 1
  local pid expected actual
  pid="$(cat "$pid_file")"
  expected="$(cat "$start_file")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  actual="$(proc_start_ticks "$pid")"
  [ -n "$actual" ] && [ "$actual" = "$expected" ]
}

clear_owned_record() {
  local kind="$1"
  rm -f "$STATE_DIR/$kind.pid" "$STATE_DIR/$kind.start" "$STATE_DIR/$kind.pgid"
}

launch_owned() {
  local kind="$1"
  shift
  local log="$STATE_DIR/$kind.log"
  # Close the wrapper flock fd in the child so UI processes cannot keep the
  # one-shot control lock held after prepare returns.
  nohup setsid "$@" 9>&- > "$log" 2>&1 < /dev/null &
  local pid=$!
  sleep 1
  if ! kill -0 "$pid" 2>/dev/null; then
    tail -40 "$log" >&2 2>/dev/null || true
    die "$kind failed to remain running"
  fi
  record_owned "$kind" "$pid"
  printf '%s_STARTED pid=%s log=%s\n' "${kind^^}" "$pid" "$log"
}

surface_window() {
  local title="$1"
  if command -v wmctrl >/dev/null 2>&1; then
    wmctrl -a "$title" >/dev/null 2>&1 || true
  elif command -v xdotool >/dev/null 2>&1; then
    local window
    window="$(xdotool search --name "$title" 2>/dev/null | tail -1 || true)"
    [ -z "$window" ] || xdotool windowactivate "$window" >/dev/null 2>&1 || true
  fi
}

surface_or_launch_teleop() {
  if owned_alive teleop; then
    surface_window "FINALS NAV TELEOP"
    echo "TELEOP=already_wrapper_owned"
    return 0
  fi
  clear_owned_record teleop

  local node_total proc_total
  refresh_nodes
  node_total="$(node_count /slam_wasd_mapper)"
  proc_total="$(pgrep -af '[s]lam_wasd_mapper.py' 2>/dev/null | wc -l | tr -d ' ' || true)"
  if [ "${node_total:-0}" -gt 1 ] || [ "${proc_total:-0}" -gt 1 ]; then
    die "duplicate teleop detected: ros_nodes=${node_total:-0} processes=${proc_total:-0}"
  fi
  if [ "${node_total:-0}" -eq 1 ] || [ "${proc_total:-0}" -eq 1 ]; then
    surface_window "SLAM WASD CONTROL"
    surface_window "FINALS NAV TELEOP"
    echo "TELEOP=existing_external_not_owned"
    return 0
  fi

  [ -f "$TELEOP_TOOL" ] || die "teleop helper missing: $TELEOP_TOOL"
  command -v xfce4-terminal >/dev/null 2>&1 || die "xfce4-terminal is required to open the operator teleop"
  local terminal_command
  terminal_command="bash -lc '$TELEOP_TOOL; rc=\$?; echo; echo TELEOP_EXIT=\$rc; exec bash'"
  launch_owned teleop env DISPLAY="$DISPLAY" DBUS_SESSION_BUS_ADDRESS="$DBUS_SESSION_BUS_ADDRESS" \
    xfce4-terminal --disable-server --title="FINALS NAV TELEOP" --command="$terminal_command"

  local deadline=$((SECONDS + ${FINALS_NAV_TELEOP_WAIT_S:-16}))
  while [ "$SECONDS" -le "$deadline" ]; do
    refresh_nodes
    node_total="$(node_count /slam_wasd_mapper)"
    if [ "${node_total:-0}" -eq 1 ]; then
      echo "TELEOP=ready operator_keyboard_authority_only"
      return 0
    fi
    if [ "${node_total:-0}" -gt 1 ]; then
      die "teleop became duplicated during prepare"
    fi
    sleep 1
  done
  warn "teleop terminal opened but mapper node is not ready yet; inspect $STATE_DIR/teleop.log"
}

surface_or_launch_rviz() {
  if owned_alive rviz; then
    surface_window "RViz"
    echo "RVIZ=already_wrapper_owned"
    return 0
  fi
  clear_owned_record rviz
  local count
  count="$(pgrep -xc rviz2 2>/dev/null || true)"
  if [ "${count:-0}" -gt 1 ]; then
    die "multiple RViz processes detected: $count"
  fi
  if [ "${count:-0}" -eq 1 ]; then
    surface_window "RViz"
    echo "RVIZ=existing_external_not_owned"
    return 0
  fi
  [ -f "$RVIZ_CONFIG" ] || die "RViz config missing: $RVIZ_CONFIG"
  command -v rviz2 >/dev/null 2>&1 || die "rviz2 is not installed"
  launch_owned rviz env DISPLAY="$DISPLAY" DBUS_SESSION_BUS_ADDRESS="$DBUS_SESSION_BUS_ADDRESS" \
    nice -n 10 rviz2 -d "$RVIZ_CONFIG"
  surface_window "RViz"
}

surface_or_launch_status() {
  if owned_alive status; then
    surface_window "FINALS NAV STATUS"
    echo "STATUS_VIEW=already_wrapper_owned"
    return 0
  fi
  clear_owned_record status
  local count
  count="$(pgrep -af '[f]inals_nav_status.sh --watch' 2>/dev/null | wc -l | tr -d ' ' || true)"
  if [ "${count:-0}" -gt 1 ]; then
    die "multiple finals status watchers detected: $count"
  fi
  if [ "${count:-0}" -eq 1 ]; then
    surface_window "FINALS NAV STATUS"
    echo "STATUS_VIEW=existing_external_not_owned"
    return 0
  fi
  [ -f "$STATUS_TOOL" ] || die "status helper missing: $STATUS_TOOL"
  command -v xfce4-terminal >/dev/null 2>&1 || die "xfce4-terminal is required to open status"
  local status_command
  status_command="bash -lc '$STATUS_TOOL --watch 8; exec bash'"
  launch_owned status env DISPLAY="$DISPLAY" DBUS_SESSION_BUS_ADDRESS="$DBUS_SESSION_BUS_ADDRESS" \
    xfce4-terminal --disable-server --title="FINALS NAV STATUS" --command="$status_command"
  surface_window "FINALS NAV STATUS"
}

stop_owned() {
  local kind="$1"
  local signal="$2"
  if ! owned_alive "$kind"; then
    clear_owned_record "$kind"
    echo "${kind^^}=not_wrapper_owned_or_already_stopped"
    return 0
  fi
  local pid pgid
  pid="$(cat "$STATE_DIR/$kind.pid")"
  pgid="$(cat "$STATE_DIR/$kind.pgid" 2>/dev/null || echo "$pid")"
  if [[ "$pgid" =~ ^[0-9]+$ ]]; then
    kill "-$signal" -- "-$pgid" 2>/dev/null || kill "-$signal" "$pid" 2>/dev/null || true
  else
    kill "-$signal" "$pid" 2>/dev/null || true
  fi
  for _ in $(seq 1 20); do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 0.1
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM -- "-$pgid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    sleep 1
  fi
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL -- "-$pgid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  fi
  clear_owned_record "$kind"
  echo "${kind^^}=wrapper_owned_stopped"
}

do_prepare() {
  local mode="${1:-teleop}"
  command -v ros2 >/dev/null 2>&1 || die "ros2 CLI is unavailable"
  command -v timeout >/dev/null 2>&1 || die "timeout is unavailable"
  command -v setsid >/dev/null 2>&1 || die "setsid is unavailable"
  require_single_service_owner
  # The existing stack must prove firmware estop before any restart/profile change.
  require_latched_estop
  configure_finals_runtime
  require_single_core_stack
  require_latched_estop
  if [ "$mode" = "teleop" ]; then
    surface_or_launch_teleop
  else
    stop_owned teleop INT
    local teleop_procs
    teleop_procs="$(pgrep -af '[s]lam_wasd_mapper.py' 2>/dev/null | wc -l | tr -d ' ' || true)"
    [ "${teleop_procs:-0}" -eq 0 ] || die "external WASD process is still active; Nav2 goal mode refuses command contention"
    echo "TELEOP=disabled_for_nav2_goal"
  fi
  surface_or_launch_rviz
  surface_or_launch_status
  echo "FINALS_NAV_PREPARED"
  echo "Motion remains inhibited by the latched estop. This wrapper never clears it."
}

do_restore() {
  setup_ros
  require_single_service_owner
  timeout 4 ros2 topic pub --once /estop std_msgs/msg/Bool '{data: true}' >/dev/null 2>&1 || true
  do_stop
  sudo -n systemctl unset-environment \
    EB_USE_NAV2 EB_USE_COLLISION_MONITOR EB_USE_MPPI \
    EB_MPPI_PUBLISH_DIRECT_CMD_VEL EB_MPPI_CMD_VEL_TOPIC \
    EB_DISPATCH_STUB_MODE EB_AI_BRAIN_URL
  sudo -n systemctl restart embodied_brain.service
  if [ -f "$STATE_DIR/cockpit_bridge.was_active" ]; then
    sudo -n systemctl start cockpit_bridge.service || true
    rm -f "$STATE_DIR/cockpit_bridge.was_active"
  fi
  rm -f "$STATE_DIR/finals.environment"
  echo "FINALS_NAV_RESTORED estop_requested=true"
}

do_stop() {
  # SIGINT lets the existing teleop helper run its zero-velocity shutdown path.
  stop_owned teleop INT
  stop_owned status TERM
  stop_owned rviz TERM
  echo "FINALS_NAV_UI_STOPPED core_owner=unchanged"
}

do_save() {
  command -v ros2 >/dev/null 2>&1 || die "ros2 CLI is unavailable"
  require_single_service_owner
  require_single_core_stack
  [ -f "$SAVE_TOOL" ] || die "map save helper missing: $SAVE_TOOL"
  bash "$SAVE_TOOL" "$@"
}

mkdir -p "$STATE_DIR"
acquire_lock

case "$ACTION" in
  prepare)
    [ "$#" -eq 0 ] || die "prepare takes no arguments"
    setup_ros
    do_prepare teleop
    ;;
  nav2)
    [ "$#" -eq 0 ] || die "nav2 takes no arguments"
    setup_ros
    do_prepare nav2
    ;;
  status)
    [ "$#" -eq 0 ] || die "status takes no arguments"
    setup_ros
    [ -f "$STATUS_TOOL" ] || die "status helper missing: $STATUS_TOOL"
    exec bash "$STATUS_TOOL"
    ;;
  stop)
    [ "$#" -eq 0 ] || die "stop takes no arguments"
    do_stop
    ;;
  restore)
    [ "$#" -eq 0 ] || die "restore takes no arguments"
    do_restore
    ;;
  save)
    [ "$#" -eq 0 ] || die "save takes no arguments"
    setup_ros
    do_save
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    die "unknown action: $ACTION"
    ;;
esac
