#!/usr/bin/env bash
# Read-only runtime preflight for the embodied-brain X5.
# It does not start ROS nodes, open serial ports, or publish /cmd_vel.

set -uo pipefail

MODEL_BIN="${LAB_FSD_TINY_OCC_BIN:-$HOME/models/lab_fsd/lab_fsd_tiny_occ_risk.bin}"
ANOMALY_BIN="${LAB_FSD_ANOMALY_BIN:-$HOME/models/lab_fsd/lab_anomaly_autoencoder.bin}"
MPPI_BIN="${MPPI_COST_BIN:-$HOME/bpu_models/cost_mlp.bin}"
EXPECTED_OCC_SHA256="3b1a96483351f72746fdcacfb179b69f4527076046e5dd73d5bcae7688d99c90"
EXPECTED_ANOMALY_SHA256="1045be38ff947ad3c97c365416170970f59735504a1f38663bd8cce8d112ad7f"
EXPECTED_MPPI_SHA256="fe54f08d12285cf66c37ee7168b51a6762bb086b30a681a12f18374d8eea853d"

fail_count=0
warn_count=0

pass() {
  printf '[PASS] %s\n' "$*"
}

warn() {
  warn_count=$((warn_count + 1))
  printf '[WARN] %s\n' "$*"
}

fail() {
  fail_count=$((fail_count + 1))
  printf '[FAIL] %s\n' "$*"
}

source_ros() {
  export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
  export LD_LIBRARY_PATH="/opt/tros/humble/lib:${LD_LIBRARY_PATH:-}"
  # ROS generated setup scripts are not nounset-safe on all Humble/TROS images.
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash 2>/dev/null || true
  # shellcheck disable=SC1091
  source /opt/tros/humble/setup.bash 2>/dev/null || true
  # shellcheck disable=SC1091
  source "$HOME/ros2_ws/install/setup.bash" 2>/dev/null || true
  set -u
}

check_pkg_required() {
  local pkg="$1"
  if ros2 pkg prefix "$pkg" >/dev/null 2>&1; then
    pass "ros2 pkg $pkg"
  else
    fail "missing required ROS package: $pkg"
  fi
}

check_pkg_optional() {
  local pkg="$1"
  if ros2 pkg prefix "$pkg" >/dev/null 2>&1; then
    pass "ros2 optional pkg $pkg"
  else
    warn "missing optional ROS package: $pkg"
  fi
}

check_device_required() {
  local dev="$1"
  if [ -e "$dev" ]; then
    pass "$dev -> $(readlink -f "$dev" 2>/dev/null || echo unknown)"
  else
    fail "missing device link: $dev"
  fi
}

check_device_optional() {
  local dev="$1"
  if [ -e "$dev" ]; then
    pass "$dev -> $(readlink -f "$dev" 2>/dev/null || echo unknown)"
  else
    warn "missing optional device link: $dev"
  fi
}

check_model() {
  local name="$1"
  local path="$2"
  local expected="$3"
  local required="$4"
  if [ ! -f "$path" ]; then
    if [ "$required" = "1" ]; then
      fail "missing model $name: $path"
    else
      warn "missing optional model $name: $path"
    fi
    return
  fi
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  local size
  size="$(wc -c < "$path" | tr -d ' ')"
  if [ "$actual" = "$expected" ]; then
    pass "model $name size=${size} sha256=${actual}"
  elif [ "$required" = "1" ]; then
    fail "model $name hash mismatch size=${size} sha256=${actual} expected=${expected}"
  else
    warn "optional model $name hash mismatch size=${size} sha256=${actual} expected=${expected}"
  fi
}

echo "X5_RUNTIME_PREFLIGHT"
echo "time: $(date -Is)"
echo "host: $(hostname 2>/dev/null || echo unknown)"
echo "user: $(id)"
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}"
echo ""

source_ros
if command -v ros2 >/dev/null 2>&1; then
  pass "ros2 command available: $(command -v ros2)"
else
  fail "ros2 command not available after sourcing environments"
fi

if command -v colcon >/dev/null 2>&1; then
  pass "colcon command available"
else
  warn "colcon command not available; deploy build will fail on X5"
fi

for pkg in \
  my_robot_bringup \
  my_robot_drivers \
  my_robot_navigation \
  my_robot_msgs \
  slam_toolbox \
  depthimage_to_laserscan \
  nav2_bringup \
  nav2_lifecycle_manager \
  rosbag2_storage_mcap
do
  check_pkg_required "$pkg"
done

for pkg in \
  nav2_collision_monitor \
  ldlidar_stl_ros2 \
  astra_camera \
  usb_cam \
  diagnostic_updater \
  visualization_msgs
do
  check_pkg_optional "$pkg"
done

check_device_required /dev/F407
check_device_required /dev/LD14
check_device_optional /dev/lift_camera
check_device_optional /dev/PT_CAM

check_model "lab_fsd_tiny_occ_risk" "$MODEL_BIN" "$EXPECTED_OCC_SHA256" 1
check_model "lab_anomaly_autoencoder" "$ANOMALY_BIN" "$EXPECTED_ANOMALY_SHA256" 0
check_model "mppi_cost" "$MPPI_BIN" "$EXPECTED_MPPI_SHA256" 1

if python3 - <<'PY' >/tmp/x5_preflight_hobot_dnn.txt 2>&1
try:
    import hobot_dnn.pyeasy_dnn as dnn
    print("hobot_dnn import OK", dnn)
except Exception as exc:
    print("hobot_dnn import FAIL", repr(exc))
    raise SystemExit(1)
PY
then
  pass "hobot_dnn import"
else
  fail "hobot_dnn import failed; see /tmp/x5_preflight_hobot_dnn.txt"
fi

if systemctl list-unit-files --no-legend embodied_brain.service 2>/dev/null | grep -q '^embodied_brain.service'; then
  pass "system service embodied_brain.service present"
  systemctl is-active embodied_brain.service >/dev/null 2>&1 && pass "system service active" || warn "system service not active"
elif systemctl --user list-unit-files --no-legend embodied_brain.service 2>/dev/null | grep -q '^embodied_brain.service'; then
  pass "user service embodied_brain.service present"
  systemctl --user is-active embodied_brain.service >/dev/null 2>&1 && pass "user service active" || warn "user service not active"
else
  warn "embodied_brain.service not installed; manual ros2 launch is required"
fi

if [ "$fail_count" -gt 0 ]; then
  echo ""
  echo "summary: FAIL fail=${fail_count} warn=${warn_count}"
  exit 2
fi
if [ "$warn_count" -gt 0 ]; then
  echo ""
  echo "summary: PASS_WITH_WARN fail=0 warn=${warn_count}"
else
  echo ""
  echo "summary: PASS fail=0 warn=0"
fi
exit 0
