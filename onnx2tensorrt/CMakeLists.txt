cmake_minimum_required(VERSION 3.14)  # change the version, if asked by compiler
project(onnx2tensorrt)

set(CMAKE_CXX_STANDARD 14)

# include and link dirs of tensorrt, you need adapt them if yours are different
include_directories(/home/rton/TensorRT/TensorRT-10.11.0.33/include)
link_directories(/home/rton/TensorRT/TensorRT-10.11.0.33/lib)

# include and link dirs of cuda for inference
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# 设置RPATH，使程序在运行时能自动找到TensorRT库
set(CMAKE_INSTALL_RPATH "/home/rton/TensorRT/TensorRT-10.11.0.33/targets/x86_64-linux-gnu/lib;/usr/local/cuda/lib64")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# create link for executable files
add_executable(onnx2tensorrt onnx2tensorrt.cpp)

target_link_libraries(onnx2tensorrt
    nvinfer
    nvonnxparser
    nvinfer_plugin
    cudart
)
add_definitions(-O2 -pthread)