cmake_minimum_required(VERSION 3.14)
project(tensorrt_inference_base)

set(CMAKE_CXX_STANDARD 14)

# Find CUDA package
find_package(CUDA REQUIRED)

# include and link dirs of tensorrt, you need adapt them if yours are different
# Note: Adjust these paths according to your TensorRT installation
include_directories(/home/rton/TensorRT/TensorRT-10.11.0.33/include)
link_directories(/home/rton/TensorRT/TensorRT-10.11.0.33/lib)

# include and link dirs of cuda for inference
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# Add our own headers to the include path
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# 设置RPATH，使程序在运行时能自动找到TensorRT库
set(CMAKE_INSTALL_RPATH "/home/rton/TensorRT/TensorRT-10.11.0.33/targets/x86_64-linux-gnu/lib;/usr/local/cuda/lib64")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# Define source files
set(SOURCES
    trt_base/trt_inference_base.cpp
    mlp_test/mlp_inference.cpp
    test.cpp
)

# Define header files for IDE purposes (optional)
set(HEADERS
    trt_base/trt_inference_base.h
    mlp_test/mlp_inference.h
)

# Create executable
add_executable(test_trt_inference ${SOURCES})

# Link libraries
target_link_libraries(test_trt_inference
    nvinfer
    nvonnxparser  # Required for ONNX models
    nvinfer_plugin
    cudart
)

add_definitions(-O2 -pthread)