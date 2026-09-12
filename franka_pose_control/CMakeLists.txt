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
find_package(ament_cmake_python REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

rosidl_generate_interfaces(
  ${PROJECT_NAME}
  "msg/Stage3PolicyCommand.msg"
  "msg/Stage3PolicyStatus.msg"
  DEPENDENCIES geometry_msgs std_msgs
)
ament_export_dependencies(rosidl_default_runtime)

ament_python_install_package(franka_pose_control_policy)

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
install(PROGRAMS scripts/pose scripts/stage3_command scripts/stage3_policy_node DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS ../tools/fr3_comparison/online_client.py DESTINATION lib/${PROJECT_NAME} RENAME fr3_comparison_node)
install(FILES ../tools/fr3_comparison/protocol.py DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch models DESTINATION share/${PROJECT_NAME})
install(FILES README.md README_STAGE3.md requirements-stage3.txt DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(stage3_policy_test test/test_stage3_policy.py)
  set_tests_properties(stage3_policy_test PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}")
endif()

ament_package()
