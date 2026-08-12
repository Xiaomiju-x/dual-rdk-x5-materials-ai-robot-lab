#!/usr/bin/env bash
# Read-only commissioning readiness check for the embodied-brain X5.
# It never publishes ROS messages, calls services, opens serial devices, clears
# estop, starts/stops services, or changes network state.

set -uo pipefail

mode="all"
map_yaml="${EB_MAP_YAML:-$HOME/maps/lab_final_20260708_210920.yaml}"
fail_count=0
warn_count=0

usage() {
  cat <<'EOF'
Usage: embodied_hardware_commissioning_preflight.sh [options]
  --mode base|navigation|lift|all
  --map /absolute/path/to/map.yaml
  -h, --help

Read-only guarantee: no ROS publish/service call, no serial open, no systemctl
mutation, and no network mutation.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --mode)
      shift || { usage >&2; exit 64; }
      mode="${1:-}"
      ;;
    --map)
      shift || { usage >&2; exit 64; }
      map_yaml="${1:-}"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 64
      ;;
  esac
  shift
done

case "$mode" in
  base|navigation|lift|all) ;;
  *) printf 'bad --mode: %s\n' "$mode" >&2; exit 64 ;;
esac

pass() { printf '[PASS] %s\n' "$*"; }
warn() { warn_count=$((warn_count + 1)); printf '[WARN] %s\n' "$*"; }
fail() { fail_count=$((fail_count + 1)); printf '[FAIL] %s\n' "$*"; }

source_ros() {
  export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
  export ROS2CLI_DISABLE_DAEMON=1
  export LD_LIBRARY_PATH="/opt/tros/humble/lib:${LD_LIBRARY_PATH:-}"
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash 2>/dev/null || true
  # shellcheck disable=SC1091
  source /opt/tros/humble/setup.bash 2>/dev/null || true
  # shellcheck disable=SC1091
  source "$HOME/ros2_ws/install/setup.bash" 2>/dev/null || true
  set -u
}

topic_info() {
  timeout 8 ros2 topic info "$1" -v 2>&1
}

require_topic_publisher() {
  local topic="$1"
  local info count
  info="$(topic_info "$topic" || true)"
  count="$(printf '%s\n' "$info" | sed -n 's/^Publisher count: //p' | head -1)"
  if [[ "$count" =~ ^[1-9][0-9]*$ ]]; then
    pass "$topic publisher_count=$count"
  else
    fail "$topic has no live publisher"
  fi
}

sample_topic() {
  local topic="$1"
  local out
  out="$(timeout 8 ros2 topic echo --once "$topic" 2>&1 || true)"
  if [ -n "$out" ] && ! printf '%s\n' "$out" | grep -qiE 'does not appear|unknown topic|failed|timeout'; then
    pass "$topic produced a read-only sample"
    printf '%s\n' "$out" | sed -n '1,14p'
  else
    fail "$topic did not produce a sample"
  fi
}

require_service() {
  if printf '%s\n' "$service_list" | grep -Fxq "$1"; then
    pass "service $1"
  else
    fail "missing service $1"
  fi
}

validate_map() {
  if [[ "$map_yaml" != /* ]]; then
    fail "map path must be absolute: $map_yaml"
    return
  fi
  if [ ! -f "$map_yaml" ]; then
    fail "map YAML missing: $map_yaml"
    return
  fi
  local image_ref image_path
  image_ref="$(sed -n 's/^[[:space:]]*image:[[:space:]]*//p' "$map_yaml" | head -1 | tr -d '\"\r')"
  if [ -z "$image_ref" ]; then
    fail "map YAML has no image field: $map_yaml"
    return
  fi
  if [[ "$image_ref" = /* ]]; then
    image_path="$image_ref"
  else
    image_path="$(dirname "$map_yaml")/$image_ref"
  fi
  if [ ! -f "$image_path" ]; then
    fail "map image missing: $image_path"
    return
  fi
  pass "saved map YAML $(sha256sum "$map_yaml" | awk '{print $1}')"
  pass "saved map image $(sha256sum "$image_path" | awk '{print $1}')"
}

echo "EMBODIED_HARDWARE_COMMISSIONING_PREFLIGHT"
echo "time: $(date -Is)"
echo "host: $(hostname 2>/dev/null || echo unknown)"
echo "mode: $mode"
echo "map: $map_yaml"
echo "safety: read-only; no publish/service/serial/systemctl/network mutation"
echo

if [ "$(hostname 2>/dev/null || true)" = "embodied-x5" ]; then
  pass "host identity embodied-x5"
else
  fail "wrong host; expected embodied-x5"
fi

source_ros
if ! command -v ros2 >/dev/null 2>&1; then
  fail "ros2 unavailable"
  echo "summary: FAIL fail=$fail_count warn=$warn_count"
  exit 2
fi

if [ "$mode" = "navigation" ] || [ "$mode" = "all" ]; then
  if [ ! -x "$HOME/tools/x5_runtime_preflight.sh" ]; then
    fail "missing $HOME/tools/x5_runtime_preflight.sh"
  else
  echo "--- static X5 preflight ---"
    if "$HOME/tools/x5_runtime_preflight.sh"; then
      pass "static X5 preflight"
    else
      fail "static X5 preflight"
    fi
    echo "--- live read-only checks ---"
  fi
fi

for unit in embodied_brain.service cockpit_bridge.service eb_navcockpit.service; do
  if systemctl is-active "$unit" >/dev/null 2>&1; then
    pass "systemd $unit active"
  else
    warn "systemd $unit is not active"
  fi
done

service_list="$(timeout 8 ros2 service list 2>/dev/null || true)"

if [ "$mode" = "base" ] || [ "$mode" = "navigation" ] \
    || [ "$mode" = "lift" ] || [ "$mode" = "all" ]; then
  for service in /estop /clear_estop /set_lift_height /set_electromagnet /lift_home; do
    require_service "$service"
  done
  require_topic_publisher /f407/firmware_identity_valid
  require_topic_publisher /f407/estop_latched
  require_topic_publisher /f407/firmware_info
  require_topic_publisher /diagnostics

  identity="$(timeout 8 ros2 topic echo --once /f407/firmware_identity_valid 2>&1 || true)"
  if printf '%s\n' "$identity" | grep -Eq 'data:[[:space:]]*true'; then
    pass "F407 firmware identity is valid"
  else
    fail "F407 firmware identity is not valid/fresh"
  fi
  estop="$(timeout 8 ros2 topic echo --once /f407/estop_latched 2>&1 || true)"
  if printf '%s\n' "$estop" | grep -Eq 'data:[[:space:]]*true'; then
    pass "F407 estop remains latched before commissioning"
  else
    fail "F407 estop is not visibly latched; do not begin motion commissioning"
  fi
fi

if [ "$mode" = "navigation" ] || [ "$mode" = "all" ]; then
  validate_map
  require_topic_publisher /scan
  require_topic_publisher /scan_depth
  require_topic_publisher /odom
  sample_topic /scan
  sample_topic /scan_depth
  sample_topic /odom
  if timeout 8 ros2 run tf2_ros tf2_echo odom base_footprint 2>&1 | grep -q 'Translation:'; then
    pass "TF odom -> base_footprint"
  else
    fail "TF odom -> base_footprint unavailable"
  fi
  installed_collision="$HOME/ros2_ws/install/share/my_robot_navigation/config/collision_monitor.yaml"
  if [ -f "$installed_collision" ] \
      && grep -Fq 'cmd_vel_in_topic: "/cmd_vel"' "$installed_collision" \
      && grep -Fq 'cmd_vel_out_topic: "/cmd_vel_safe"' "$installed_collision"; then
    pass "installed Collision Monitor safe topic contract"
  else
    fail "installed Collision Monitor safe topic contract mismatch"
  fi
fi

if [ "$mode" = "lift" ] || [ "$mode" = "all" ]; then
  require_topic_publisher /lift_status
  sample_topic /lift_status
fi

echo
if [ "$fail_count" -gt 0 ]; then
  echo "summary: FAIL fail=$fail_count warn=$warn_count"
  exit 2
fi
if [ "$warn_count" -gt 0 ]; then
  echo "summary: PASS_WITH_WARN fail=0 warn=$warn_count"
else
  echo "summary: PASS fail=0 warn=0"
fi
exit 0
