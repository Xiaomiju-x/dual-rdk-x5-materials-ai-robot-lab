cmake_minimum_required(VERSION 3.8)
project(my_robot_perception)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
# Phase 3+ 才加: rclcpp / sensor_msgs / cv_bridge / hobot_dnn 等

# 占位
install(DIRECTORY launch config models
  DESTINATION share/${PROJECT_NAME}/
  OPTIONAL
)

ament_package()
