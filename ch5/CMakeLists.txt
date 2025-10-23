cmake_minimum_required(VERSION 2.8)


option(ENABLE_CMAKE_DEBUG "Enable detailed debug messages" ON)

if(ENABLE_CMAKE_DEBUG)
    set(CMAKE_BUILD_TYPE "Debug")
    # 添加c++ 11标准支持
else()
    set(CMAKE_BUILD_TYPE "Release")

endif()


set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11 -O2")
set(CMAKE_CXX_FLAGS_DEBUG "-std=c++11 -g3 -g")

# Eigen
include_directories("/usr/include/eigen3")

# 寻找OpenCV库
find_package(OpenCV 4 REQUIRED)
# 添加头文件
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(imageBasics)
add_subdirectory(stereo)
add_subdirectory(rgbd)
