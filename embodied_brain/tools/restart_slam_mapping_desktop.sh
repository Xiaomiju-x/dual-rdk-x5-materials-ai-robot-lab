#!/usr/bin/env bash
set -eo pipefail
set +u

export DISPLAY="${DISPLAY:-:0}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export LD_LIBRARY_PATH="/opt/tros/humble/lib:${LD_LIBRARY_PATH:-}"

kill_patterns=(
  "/home/rdk/tools/slam_wasd_mapper.py"
  "rviz2"
  "ros2 launch my_robot_bringup full.launch.py"
  "robot_state_publisher"
  "joint_state_publisher"
  "fake_odom"
  "ldlidar_stl_ros2_node"
  "astra_camera_node"
  "async_slam_toolbox_node"
  "sync_slam_toolbox_node"
  "depthimage_to_laserscan_node"
  "hobot_yolo_world"
  "mono_edgesam"
  "serial_f407_node"
  "pt_camera"
  "dispatch_server"
  "ai_brain_bridge"
  "telemetry_publisher"
  "command_interpreter"
  "location_visualizer"
)

for pattern in "${kill_patterns[@]}"; do
  pkill -f "$pattern" 2>/dev/null || true
done

sleep 4

source /opt/ros/humble/setup.bash
if [ -f /opt/tros/humble/setup.bash ]; then
  source /opt/tros/humble/setup.bash
fi
if [ -f "$HOME/ros2_ws/install/setup.bash" ]; then
  source "$HOME/ros2_ws/install/setup.bash"
fi

mkdir -p "$HOME/maps"

nohup ros2 launch my_robot_bringup full.launch.py \
  use_fake_odom:=false \
  use_serial_f407:=true \
  use_state_estimator:=true \
  use_nav2:=true \
  use_collision_monitor:=true \
  > /tmp/full_launch_slam.log 2>&1 < /dev/null &

sleep 12

if command -v xfce4-terminal >/dev/null 2>&1; then
  xfce4-terminal \
    --title="SLAM WASD CONTROL" \
    --command="bash -lc '$HOME/tools/start_slam_wasd_mapper.sh; exec bash'" &
else
  nohup "$HOME/tools/start_slam_wasd_mapper.sh" \
    > /tmp/slam_wasd_mapper.log 2>&1 < /dev/null &
fi

sleep 1

nohup rviz2 -d "$HOME/tools/slam_mapping.rviz" \
  > /tmp/rviz_slam.log 2>&1 < /dev/null &

sleep 3

ros2 node list > /tmp/slam_nodes_after_restart.txt 2>&1 || true
timeout 6 ros2 topic echo /map_metadata --once --qos-durability transient_local \
  > /tmp/slam_map_metadata_after_restart.txt 2>&1 || true

echo "SLAM_MAPPING_DESKTOP_READY"
echo "logs: /tmp/full_launch_slam.log /tmp/rviz_slam.log /tmp/slam_nodes_after_restart.txt /tmp/slam_map_metadata_after_restart.txt"
