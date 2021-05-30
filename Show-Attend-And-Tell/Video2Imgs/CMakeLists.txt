cmake_minimum_required(VERSION 3.19)
project(generateKeyFrame)
find_package(OpenCV 4 REQUIRED)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/)
# 头文件
include_directories(${OpenCV_INCLUDE_DIRS})
# 库文件
link_libraries(${OpenCV_LIBRAR_DIRS})
add_definitions(${OpenCV_DEFINITIONS})
add_executable(generateKeyFrame.exe generateKeyFrame.cpp)
target_link_libraries(generateKeyFrame.exe ${OpenCV_LIBS})
