#!/usr/bin/env bash
set -eo pipefail
set +u

source /opt/ros/humble/setup.bash
if [ -f "$HOME/ros2_ws/install/setup.bash" ]; then
  source "$HOME/ros2_ws/install/setup.bash"
fi

mkdir -p "$HOME/maps"
echo "Running read-only mapping sensor gate..."
ros2 run my_robot_navigation mapping_sensor_preflight.py \
  --duration 8 --odom-topic /wheel_odom
exec python3 "$HOME/tools/slam_wasd_mapper.py" "$@"
