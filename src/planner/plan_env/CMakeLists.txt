cmake_minimum_required(VERSION 2.8.3)
project(plan_env)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

find_package(OpenCV REQUIRED)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
  cv_bridge
  message_filters
  message_generation
)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.7 REQUIRED)

add_message_files(
  FILES
  MultipleMasksWithConfidence.msg
  SemanticPrompts.msg
  SemanticScores.msg
  IGScore.msg
  VLMVerificationRequest.msg
  VLMVerificationResult.msg
  ObjectUpdateInfo.msg
)

add_service_files(
  FILES
  ValidateObject.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES plan_env
  CATKIN_DEPENDS roscpp std_msgs message_runtime sensor_msgs
  #  DEPENDS system_lib
)

include_directories(
  SYSTEM
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})


add_library(plan_env
  src/sdf_map2d.cpp
  src/map_ros.cpp
  src/raycast2d.cpp
  src/perception_utils2d.cpp
  src/frontier_map2d.cpp
  src/object_map2d.cpp
  src/value_map2d.cpp
  src/multi_source_value_map.cpp
  src/ig_value_map.cpp
)
target_link_libraries(plan_env
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

# Install Python scripts
catkin_install_python(PROGRAMS
  scripts/multi_source_itm_node.py
  scripts/test_semantic_prompts.py
  scripts/qwen_vlm_validation_node.py
  scripts/vlm_approach_verifier.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install shell scripts (without Python wrapper)
install(PROGRAMS
  scripts/multi_source_itm_node_wrapper.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
