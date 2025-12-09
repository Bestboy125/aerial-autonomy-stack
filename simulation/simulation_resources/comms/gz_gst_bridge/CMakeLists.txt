cmake_minimum_required(VERSION 3.10)
project(gz_gst_bridge)

set(CMAKE_CXX_STANDARD 17)

find_package(gz-transport13 REQUIRED)
find_package(gz-msgs10 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-app-1.0)

add_executable(gz_gst_bridge gz_gst_bridge.cpp)

target_include_directories(gz_gst_bridge PUBLIC 
    ${GST_INCLUDE_DIRS}
)

target_link_libraries(gz_gst_bridge
    gz-transport13::gz-transport13
    gz-msgs10::gz-msgs10
    ${GST_LIBRARIES}
)
