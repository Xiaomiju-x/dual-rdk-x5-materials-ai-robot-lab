cmake_minimum_required(VERSION 3.8)
project(my_robot_navigation)

find_package(ament_cmake REQUIRED)

# This package installs launch/config assets and lightweight Python nodes.
install(DIRECTORY launch config maps
  DESTINATION share/${PROJECT_NAME}/
  OPTIONAL
)

install(PROGRAMS
  scripts/bev_shadow_planner.py
  scripts/lab_fsd_core.py
  scripts/lab_anomaly_bpu.py
  scripts/vision_bev_bridge.py
  scripts/mapping_sensor_preflight.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
