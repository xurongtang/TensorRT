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

# 添加opencv头文件
# set(OPENCV_INCLUDE_DIRS /home/rton/CppProj/TensorProj/3rdparty/opencv-install-x86/include/opencv4)
# include_directories(${OPENCV_INCLUDE_DIRS})
# set(OPENCV_LIBS /home/rton/CppProj/TensorProj/3rdparty/opencv-install-x86/lib)
# link_directories(${OPENCV_LIBS})

set(OpenCV_DIR /home/rton/CppProj/TensorProj/3rdparty/opencv-install-x86/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

# Define source files
set(SOURCES
    trt_base/trt_inference_base.cpp
    yolo26_src/yolo26_main.cpp
    test_yolo26.cpp
)

# Define header files for IDE purposes (optional)
set(HEADERS
    trt_base/trt_inference_base.h
    yolo26_src/yolo26_main.h
)

# Create executable
add_executable(test_yolo26_trt ${SOURCES})

# Link libraries
target_link_libraries(test_yolo26_trt
    nvinfer
    # nvonnxparser  # Required for ONNX models
    nvinfer_plugin
    cudart
    # 指定opencv
     ${OpenCV_LIBS}
)

add_definitions(-O2 -pthread)