cmake_minimum_required(VERSION 3.8)
project(my_robot_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

set(msg_files
  msg/FurnaceReading.msg
  msg/Alarm.msg
  msg/PickupTarget.msg
  msg/LiftStatus.msg
  msg/SystemTelemetry.msg
  msg/HardwareSensorSample.msg
  msg/PhysicalEvidenceRequest.msg
  msg/PhysicalEvidence.msg
)

set(srv_files
  srv/SetElectromagnet.srv
  srv/SetLiftHeight.srv
  srv/InterpretCommand.srv
  srv/VlmQuery.srv
  srv/VerifyPhysicalEvidence.srv
)

set(action_files
  action/DispatchTask.action
  action/Pickup.action
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  ${action_files}
  DEPENDENCIES
    builtin_interfaces
    std_msgs
    geometry_msgs
    sensor_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
