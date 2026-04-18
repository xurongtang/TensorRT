cmake_minimum_required(VERSION 3.14)
project(yolom26_int8_batch)

set(CMAKE_CXX_STANDARD 14)

# Find CUDA package
find_package(CUDA REQUIRED)

# TensorRT include and lib dirs
include_directories(/home/rton/TensorRT/TensorRT-10.11.0.33/include)
link_directories(/home/rton/TensorRT/TensorRT-10.11.0.33/lib)

# CUDA include and lib dirs
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# Project headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# RPATH for runtime library discovery
set(CMAKE_INSTALL_RPATH "/home/rton/TensorRT/TensorRT-10.11.0.33/targets/x86_64-linux-gnu/lib;/usr/local/cuda/lib64")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# OpenCV
set(OpenCV_DIR /home/rton/CppProj/TensorProj/3rdparty/opencv-install-x86/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

# Source files
set(SOURCES
    trt_base/trt_inference_base.cpp
    src/yolom26_int8_batch.cpp
    test_yolo26_int8_batch.cpp
)

set(HEADERS
    trt_base/trt_inference_base.h
    src/yolom26_int8_batch.h
)

# Create executable
add_executable(test_yolo26_int8_batch ${SOURCES})

# Link libraries
target_link_libraries(test_yolo26_int8_batch
    nvinfer
    nvinfer_plugin
    cudart
    ${OpenCV_LIBS}
)

add_definitions(-O2 -pthread)