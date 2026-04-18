cmake_minimum_required(VERSION 3.14)
project(onnx2engine_int8)

set(CMAKE_CXX_STANDARD 14)

# Find OpenCV (required for calibration image loading)
set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/opencv-install-x86/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

# Include and link dirs of TensorRT
include_directories(/home/rton/TensorRT/TensorRT-10.11.0.33/include)
link_directories(/home/rton/TensorRT/TensorRT-10.11.0.33/lib)

# Include and link dirs of CUDA
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# Set RPATH so the executable can find TensorRT libraries at runtime
set(CMAKE_INSTALL_RPATH "/home/rton/TensorRT/TensorRT-10.11.0.33/targets/x86_64-linux-gnu/lib;/usr/local/cuda/lib64")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# Create executable
add_executable(onnx2engine_int8 onnx2engine_int8.cpp)

target_link_libraries(onnx2engine_int8
    nvinfer
    nvonnxparser
    nvinfer_plugin
    cudart
    ${OpenCV_LIBS}
)

add_definitions(-O2 -pthread)