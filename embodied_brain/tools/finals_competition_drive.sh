#!/usr/bin/env bash
# One-command finals straight-motion demo. SLAM, depth, Lab-FSD and BPU shadow
# inference remain live; the F407 actuator consumes /cmd_vel directly.

set -eo pipefail

DISTANCE_M="0.30"
CONFIRMED=0

usage() {
  cat <<'EOF'
Usage: finals_competition_drive.sh --confirm [--distance METERS]

--confirm asserts that the drive motors are powered, the forward path is clear,
the robot carries no hanging load, and an operator is monitoring the chassis.
Distance must be between 0.10 and 0.50 m.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --confirm) CONFIRMED=1 ;;
    --distance)
      [ "$#" -ge 2 ] || { usage >&2; exit 2; }
      DISTANCE_M="$2"
      shift
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[ "$CONFIRMED" -eq 1 ] || {
  echo "refusing motion without --confirm" >&2
  exit 3
}

python3 - "$DISTANCE_M" <<'PY'
import sys

value = float(sys.argv[1])
if not 0.10 <= value <= 0.50:
    raise SystemExit("distance must be in [0.10, 0.50] m")
PY

source /opt/ros/humble/setup.bash
source "$HOME/ros2_ws/install/setup.bash"

latch_estop() {
  timeout 8 ros2 service call /estop std_srvs/srv/Trigger "{}" >/tmp/finals_demo_estop.txt 2>&1 || true
}
trap latch_estop EXIT INT TERM

# Freeze the chassis before any stack ownership change.
latch_estop

current_env="$(sudo -n systemctl show-environment 2>/dev/null || true)"
if ! grep -Fqx 'EB_USE_COLLISION_MONITOR=false' <<<"$current_env"; then
  sudo -n systemctl set-environment EB_USE_COLLISION_MONITOR=false
  sudo -n systemctl restart embodied_brain.service
fi

deadline=$((SECONDS + 75))
while [ "$SECONDS" -le "$deadline" ]; do
  nodes="$(ros2 node list 2>/dev/null || true)"
  if grep -qx /serial_f407 <<<"$nodes" \
      && grep -qx /controller_server <<<"$nodes" \
      && grep -qx /slam_toolbox <<<"$nodes" \
      && grep -qx /lab_fsd_bev_shadow_planner <<<"$nodes"; then
    break
  fi
  sleep 1
done

nodes="$(ros2 node list 2>/dev/null || true)"
grep -qx /serial_f407 <<<"$nodes" || { echo "serial_f407 unavailable" >&2; exit 4; }
grep -qx /controller_server <<<"$nodes" || { echo "controller_server unavailable" >&2; exit 4; }
grep -qx /slam_toolbox <<<"$nodes" || { echo "slam_toolbox unavailable" >&2; exit 4; }
grep -qx /lab_fsd_bev_shadow_planner <<<"$nodes" || { echo "Lab-FSD unavailable" >&2; exit 4; }
if grep -qx /collision_monitor <<<"$nodes"; then
  echo "collision_monitor is still active in competition-direct mode" >&2
  exit 5
fi
ros2 node info /serial_f407 2>/dev/null | grep -Fq '/cmd_vel: geometry_msgs/msg/Twist' || {
  echo "F407 is not bound directly to /cmd_vel" >&2
  exit 6
}

# Runtime-only tuning avoids changing the hash-bound navigation configuration.
ros2 param set /controller_server controller_frequency 5.0 >/dev/null
ros2 param set /controller_server FollowPath.transform_tolerance 1.0 >/dev/null
ros2 param set /controller_server general_goal_checker.xy_goal_tolerance 0.03 >/dev/null

timeout 70 python3 -u "$HOME/tools/finals_nav_straight_test.py" \
  --distance "$DISTANCE_M" \
  --direct-path \
  --path-frame odom \
  --command-topic /cmd_vel \
  --execute \
  --confirmation SAFE_STRAIGHT_ONLY
