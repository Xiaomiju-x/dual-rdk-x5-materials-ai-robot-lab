cmake_minimum_required(VERSION 3.8)
project(my_robot_drivers)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# C++17 是 ROS2 Humble 默认
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# === 依赖 ===
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(my_robot_msgs REQUIRED)

include_directories(include)

# === serial_f407_node ===
add_executable(serial_f407_node src/serial_f407_node.cpp)
ament_target_dependencies(serial_f407_node
  rclcpp
  diagnostic_msgs
  std_msgs
  std_srvs
  geometry_msgs
  nav_msgs
  sensor_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  my_robot_msgs
)

install(TARGETS
  serial_f407_node
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/scan_self_filter.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME scan_self_filter
)

# === Launch / config / udev / scripts ===
install(DIRECTORY launch config udev scripts
  DESTINATION share/${PROJECT_NAME}/
  USE_SOURCE_PERMISSIONS
)

install(DIRECTORY include/
  DESTINATION include/
)

# === Lint ===
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include)
ament_package()
