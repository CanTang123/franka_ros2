cmake_minimum_required(VERSION 3.16)
project(franka_pose_control)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(std_srvs REQUIRED)

add_executable(pose_target_node src/pose_target_node.cpp)
ament_target_dependencies(
  pose_target_node
  builtin_interfaces
  geometry_msgs
  moveit_core
  moveit_msgs
  rclcpp
  rclcpp_action
  std_srvs
)

install(TARGETS pose_target_node DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/pose DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(FILES README.md DESTINATION share/${PROJECT_NAME})

ament_package()
