cmake_minimum_required(VERSION 3.14)
project(gt_sam_sandbox)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

find_package(
  catkin REQUIRED
  COMPONENTS
    roscpp
)

find_package(GTSAM REQUIRED QUIET)

catkin_package(
  DEPENDS
    GTSAM
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
)

link_directories(
  ${GTSAM_LIBRARY_DIRS}
)

## Add executables
add_executable(${PROJECT_NAME}_iot_slam_concept_main
	src/iot_slam_concept_main.cpp
)
target_link_libraries(${PROJECT_NAME}_iot_slam_concept_main
  ${catkin_LIBRARIES}
  gtsam
)