#!/usr/bin/env bash
set -eo pipefail
set +u

source /opt/ros/humble/setup.bash
if [ -f "$HOME/ros2_ws/install/setup.bash" ]; then
  source "$HOME/ros2_ws/install/setup.bash"
fi

cmd="${1:-help}"

case "$cmd" in
  pick)
    echo "VIDEO1_PICK: still-state bottle pickup, then hold for keyboard driving"
    ros2 topic pub --once /lift/target_height std_msgs/msg/Float32 "{data: -1.0}"
    ;;
  place|release)
    echo "VIDEO1_PLACE: release-side fallback sequence after keyboard driving"
    ros2 topic pub --once /lift/target_height std_msgs/msg/Float32 "{data: -2.0}"
    ;;
  home)
    echo "VIDEO1_HOME: retract actuator, magnet off, servo left"
    ros2 service call /lift_home std_srvs/srv/Trigger "{}"
    ;;
  mag-on)
    ros2 service call /set_electromagnet my_robot_msgs/srv/SetElectromagnet "{turn_on: true}"
    ;;
  mag-off)
    ros2 service call /set_electromagnet my_robot_msgs/srv/SetElectromagnet "{turn_on: false}"
    ;;
  status)
    ros2 topic echo /lift_status --once
    ;;
  *)
    cat <<'EOF'
Usage: ~/tools/video1_fixture_control.sh pick|place|home|mag-on|mag-off|status

Video-1 order:
  1) pick    - stationary bottle pickup, magnet holds bottle, servo returns left
  2) drive   - use the WASD terminal opened by start_video1_embodied_demo.sh
  3) place   - release-side fallback: lift attempt, actuator extend, magnet off, retract
EOF
    ;;
esac
