cmake_minimum_required(VERSION 3.8)
project(suas24_localization)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set (CMAKE_CXX_STANDARD 17)

# find dependencies
find_package(OpenCV REQUIRED)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(std_srvs REQUIRED)
#find_package(vision_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(image_geometry REQUIRED)
#find_package(suas23_interfaces REQUIRED)
#find_package(suas23_common REQUIRED)
find_package(suas24_interfaces REQUIRED)


add_executable(suas24_localization
  src/localization.cpp)
target_link_libraries(suas24_localization
  ${OpenCV_LIBS})
target_include_directories(suas24_localization
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${OpenCV_INCLUDE_DIRS})

ament_target_dependencies(suas24_localization
  rclcpp
  ament_index_cpp
  std_srvs
  #vision_msgs
  sensor_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  image_geometry
  #suas23_interfaces
  #suas23_common)
  suas24_interfaces
)

install(
  TARGETS
    suas24_localization
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
