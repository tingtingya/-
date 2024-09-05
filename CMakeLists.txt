cmake_minimum_required(VERSION 3.10)
project(demo_ByteTrack)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED on)

# 查找 OpenCV 包
find_package(OpenCV REQUIRED)

# 指定包含目录
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/bytetrack/include)  # ByteTrack 的头文件目录

include_directories(/usr/local/include/eigen3)

# 使用 file(GLOB ...) 来查找所有的 .cpp 文件
file(GLOB SOURCE_FILES
        ${CMAKE_SOURCE_DIR}/main.cpp  # 替换为你的源文件名
        ${CMAKE_SOURCE_DIR}/bytetrack/src/*.cpp  # ByteTrack 的源文件目录
)

# 创建可执行文件
add_executable(demo_ByteTrack ${SOURCE_FILES})

# 链接 OpenCV 库
target_link_libraries(demo_ByteTrack ${OpenCV_LIBS})
