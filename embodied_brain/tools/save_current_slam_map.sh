#!/usr/bin/env bash
set -eo pipefail
set +u

source /opt/ros/humble/setup.bash
if [ -f "$HOME/ros2_ws/install/setup.bash" ]; then
  source "$HOME/ros2_ws/install/setup.bash"
fi

mkdir -p "$HOME/maps"
stamp="$(date +%Y%m%d_%H%M%S)"
base="$HOME/maps/lab_final_${stamp}"

ros2 run nav2_map_server map_saver_cli \
  -f "$base" \
  --ros-args -p map_subscribe_transient_local:=true

python3 "$HOME/tools/ros_map_stats.py" || true

echo "SAVED_MAP_BASE=$base"
ls -lh "${base}.yaml" "${base}.pgm"
