#!/usr/bin/env bash
# One terminal command for the finals lift + 0.50 m odom-navigation demo.

set -Eeuo pipefail
set +u

TOOLS_DIR="$HOME/tools"
STATE_DIR="$HOME/.cache/finals_lift_nav_demo"
DROPIN_DIR="/etc/systemd/system/embodied_brain.service.d"
DROPIN_PATH="$DROPIN_DIR/finals-demo.conf"
DISPLAY="${DISPLAY:-:0}"
DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/$(id -u)/bus}"

mkdir -p "$STATE_DIR"
exec 9>"$STATE_DIR/run.lock"
flock -n 9 || { echo "Finals demo is already running; refusing a duplicate run." >&2; exit 2; }

export DISPLAY DBUS_SESSION_BUS_ADDRESS ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
source /opt/ros/humble/setup.bash
if [ -f /opt/tros/humble/setup.bash ]; then
  source /opt/tros/humble/setup.bash
fi
source "$HOME/ros2_ws/install/setup.bash"

ros_node_present() {
  local target="$1"
  timeout 5 ros2 node list 2>/dev/null | grep -Fqx "$target"
}

wait_for_f407_services() {
  local timeout_s="$1"
  local deadline=$((SECONDS + timeout_s))
  local nodes services
  while [ "$SECONDS" -le "$deadline" ]; do
    nodes="$(timeout 5 ros2 node list 2>/dev/null || true)"
    services="$(timeout 5 ros2 service list 2>/dev/null || true)"
    if grep -Fqx /serial_f407 <<<"$nodes" \
        && grep -Fqx /set_lift_height <<<"$services" \
        && grep -Fqx /clear_estop <<<"$services"; then
      return 0
    fi
    sleep 1
  done
  return 1
}

slam_map_publisher_ready() {
  local info
  ros_node_present /slam_toolbox || return 1
  info="$(timeout 5 ros2 topic info /map 2>/dev/null || true)"
  grep -Eq 'Publisher count: [1-9][0-9]*' <<<"$info"
}

observe_optional_runtime() {
  local timeout_s="$1"
  local deadline=$((SECONDS + timeout_s))
  local nodes=""
  local map_ready=0
  local fsd_ready=0

  while [ "$SECONDS" -le "$deadline" ]; do
    nodes="$(timeout 5 ros2 node list 2>/dev/null || true)"
    if slam_map_publisher_ready; then
      map_ready=1
    fi
    if grep -Fqx /lab_fsd_bev_shadow_planner <<<"$nodes"; then
      fsd_ready=1
    fi
    if [ "$map_ready" -eq 1 ] && [ "$fsd_ready" -eq 1 ]; then
      break
    fi
    sleep 1
  done

  nodes="$(timeout 5 ros2 node list 2>/dev/null || true)"
  if grep -Fqx /controller_server <<<"$nodes"; then
    echo "Nav2 FollowPath: READY (display/diagnostics; odom loop owns execution)"
    ros2 param set /controller_server controller_frequency 5.0 >/dev/null 2>&1 || true
    ros2 param set /controller_server FollowPath.transform_tolerance 1.0 >/dev/null 2>&1 || true
    ros2 param set /controller_server general_goal_checker.xy_goal_tolerance 0.03 >/dev/null 2>&1 || true
  else
    echo "Nav2 FollowPath: OPTIONAL OFFLINE (odom drive continues)"
  fi

  if [ "$map_ready" -eq 1 ]; then
    echo "SLAM/RViz: READY (/slam_toolbox + /map publisher)"
  elif grep -Fqx /slam_toolbox <<<"$nodes"; then
    echo "SLAM/RViz: STARTING (node online, /map not published yet; drive continues)"
  else
    echo "SLAM/RViz: OPTIONAL OFFLINE (odom drive continues)"
  fi

  if [ "$fsd_ready" -eq 1 ]; then
    echo "Lab-FSD shadow: READY"
  else
    echo "Lab-FSD shadow: OPTIONAL OFFLINE (odom drive continues)"
  fi

  if curl -fsS --max-time 2 http://192.0.2.103:8888/api/health >/dev/null 2>&1; then
    echo "AI-brain Vision-BEV/FSD bridge: ONLINE"
  else
    echo "AI-brain Vision-BEV/FSD bridge: OPTIONAL OFFLINE (odom drive continues)"
  fi
  touch "$STATE_DIR/runtime-observer.done"
}

latch_estop() {
  timeout 8 ros2 service call /estop std_srvs/srv/Trigger >/tmp/finals_demo_estop.log 2>&1 || true
}
trap latch_estop EXIT INT TERM

echo "============================================================"
echo "FINALS DEMO: pick and lift -> 0.50 m odom drive -> lower and place"
echo "Working directory: $PWD (all runtime paths are absolute under HOME)"
echo "============================================================"

sudo -n true >/dev/null 2>&1 || {
  echo "Passwordless sudo is required to apply the finals service profile." >&2
  exit 3
}

dropin_candidate="$STATE_DIR/finals-demo.conf"
cat >"$dropin_candidate" <<'EOF'
[Service]
Environment=EB_USE_NAV2=true
Environment=EB_USE_COLLISION_MONITOR=false
Environment=EB_USE_SERIAL_F407=true
Environment=EB_USE_FAKE_ODOM=false
Environment=EB_USE_STATE_ESTIMATOR=true
Environment=EB_USE_LAB_FSD_SHADOW=true
Environment=EB_USE_MPPI=false
Environment=EB_MPPI_PUBLISH_DIRECT_CMD_VEL=false
Environment=EB_AI_BRAIN_URL=http://192.0.2.103:8888
EOF

need_restart=0
if ! sudo -n test -f "$DROPIN_PATH" \
    || ! sudo -n cmp -s "$dropin_candidate" "$DROPIN_PATH"; then
  sudo -n install -D -m 0644 "$dropin_candidate" "$DROPIN_PATH"
  sudo -n systemctl daemon-reload
  need_restart=1
fi
if ! systemctl is-active --quiet embodied_brain.service; then
  need_restart=1
fi

if [ "$need_restart" -eq 1 ]; then
  echo "[1/4] Applying Nav2/SLAM/F407/Lab-FSD finals profile..."
  sudo -n systemctl restart embodied_brain.service
else
  echo "[1/4] Finals runtime profile is already active."
fi

echo "[2/4] Starting RViz, SLAM and Lab-FSD observation in parallel..."
rm -f "$STATE_DIR/runtime-observer.done" "$STATE_DIR/runtime-observer.log"
(
  trap - EXIT INT TERM
  exec 9>&-
  bash "$TOOLS_DIR/start_slam_rviz_desktop.sh"
) >/tmp/finals_demo_rviz_start.log 2>&1 &
(
  trap - EXIT INT TERM
  exec 9>&-
  observe_optional_runtime 60
) >"$STATE_DIR/runtime-observer.log" 2>&1 &

echo "[3/4] Waiting only for the F407 bridge..."
wait_for_f407_services 85 || {
  echo "F407 control services are unavailable; refusing to start the fixture or chassis." >&2
  exit 4
}

echo "[4/4] F407 ready; starting pickup now while the perception stack continues loading..."
python3 -u "$TOOLS_DIR/finals_lift_nav_demo.py" \
  --distance 0.50 \
  --drive-mode odom \
  --confirm \
  --report "$HOME/finals_demo_logs/latest.json"
