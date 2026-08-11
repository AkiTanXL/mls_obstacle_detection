cmake_minimum_required(VERSION 3.20)
project(mls_obstacle_detection VERSION 1.0.0 LANGUAGES CXX)

option(MLS_BUILD_TESTS "Build tests" ON)
option(MLS_BUILD_VIEWER "Build the PCL visualizer" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PCL 1.15 REQUIRED COMPONENTS common io filters segmentation search kdtree)
if(MLS_BUILD_VIEWER)
  find_package(PCL 1.15 REQUIRED COMPONENTS visualization)
  if(UNIX AND NOT APPLE)
    find_package(X11 REQUIRED)
  endif()
endif()
find_package(CLI11 CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

add_library(mls_obstacle_detection_core STATIC
  src/camera.cpp
  src/config.cpp
  src/frame_source.cpp
  src/pipeline.cpp
  src/serialization.cpp)
add_library(mls::obstacle_detection_core ALIAS mls_obstacle_detection_core)
target_include_directories(mls_obstacle_detection_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS})
target_link_libraries(mls_obstacle_detection_core PUBLIC
  pcl_common pcl_io pcl_filters pcl_segmentation pcl_search pcl_kdtree
  yaml-cpp::yaml-cpp nlohmann_json::nlohmann_json)
target_compile_definitions(mls_obstacle_detection_core PUBLIC ${PCL_DEFINITIONS})
target_compile_options(mls_obstacle_detection_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(mls_obstacle_detector src/main.cpp)
target_link_libraries(mls_obstacle_detector PRIVATE mls_obstacle_detection_core CLI11::CLI11)
if(MLS_BUILD_VIEWER)
  target_sources(mls_obstacle_detector PRIVATE src/viewer.cpp)
  target_link_libraries(mls_obstacle_detector PRIVATE pcl_visualization)
  if(UNIX AND NOT APPLE)
    target_link_libraries(mls_obstacle_detector PRIVATE X11::X11)
  endif()
  target_compile_definitions(mls_obstacle_detector PRIVATE MLS_HAS_VIEWER=1)
else()
  target_sources(mls_obstacle_detector PRIVATE src/viewer_stub.cpp)
endif()
target_compile_options(mls_obstacle_detector PRIVATE -Wall -Wextra -Wpedantic)

include(CTest)
if(BUILD_TESTING AND MLS_BUILD_TESTS)
  find_package(GTest CONFIG REQUIRED)
  add_executable(mls_obstacle_detection_tests
    tests/camera_test.cpp
    tests/config_test.cpp
    tests/frame_source_test.cpp
    tests/pipeline_test.cpp
    tests/serialization_test.cpp
    tests/dataset_integration_test.cpp)
  target_link_libraries(mls_obstacle_detection_tests PRIVATE
    mls_obstacle_detection_core GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(mls_obstacle_detection_tests)
endif()

install(TARGETS mls_obstacle_detection_core mls_obstacle_detector)
install(DIRECTORY include/ DESTINATION include)
install(FILES config/highway.yaml DESTINATION share/mls_obstacle_detection/config)
