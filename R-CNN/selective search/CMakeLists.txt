cmake_minimum_required(VERSION 3.19)
project(selected_search)
find_package(OpenCV 4 REQUIRED)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/)
# 头文件
include_directories(${OpenCV_INCLUDE_DIRS})
# 库文件
link_libraries(${OpenCV_LIBRAR_DIRS})
add_definitions(${OpenCV_DEFINITIONS})
# 编译main.cpp
add_executable(selected_search main.cpp getFiles.h getFiles.h)
# 连接selected_search和opencv库文件
target_link_libraries(selected_search ${OpenCV_LIBS})
#add_executable(readCsv readCsv.h)
#target_link_libraries(readCsv ${OpenCV_LIBS})