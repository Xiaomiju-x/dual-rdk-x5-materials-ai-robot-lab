#!/usr/bin/env bash
set -eo pipefail

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-227}"
export ROS_LOCALHOST_ONLY=1
if [ "$ROS_DOMAIN_ID" -lt 200 ]; then
  echo "ERROR: ROS_DOMAIN_ID must be >=200" >&2
  exit 64
fi
source /opt/ros/humble/setup.bash
source "$HOME/ros2_ws/install/setup.bash"

out="${1:-/tmp/state_estimator_smoke}"
mkdir -p "$out"
ekf_pid=""
fixture_pid=""
cleanup() {
  [ -z "$fixture_pid" ] || kill -INT "$fixture_pid" 2>/dev/null || true
  [ -z "$ekf_pid" ] || kill -INT "$ekf_pid" 2>/dev/null || true
  [ -z "$fixture_pid" ] || wait "$fixture_pid" 2>/dev/null || true
  [ -z "$ekf_pid" ] || wait "$ekf_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ros2 launch my_robot_navigation state_estimator.launch.py >"$out/ekf.log" 2>&1 &
ekf_pid=$!
sleep 3
python3 "$HOME/tools/state_estimator_fixture.py" --duration 8 >"$out/fixture.log" 2>&1 &
fixture_pid=$!
sleep 2

ros2 topic info /odom -v >"$out/odom_graph.txt" 2>&1
timeout 5 ros2 topic echo /odom --once >"$out/odom_sample.txt" 2>&1
timeout 5 ros2 run tf2_ros tf2_echo odom base_footprint >"$out/tf_sample.txt" 2>&1 || true
wait "$fixture_pid"
fixture_pid=""

grep -q "Publisher count: 1" "$out/odom_graph.txt"
grep -q "child_frame_id: base_footprint" "$out/odom_sample.txt"
grep -q "Translation:" "$out/tf_sample.txt"
! grep -Eq "process has died|error while loading shared libraries" "$out/ekf.log"
echo "STATE_ESTIMATOR_SMOKE PASS domain=$ROS_DOMAIN_ID hardware_touched=false"
