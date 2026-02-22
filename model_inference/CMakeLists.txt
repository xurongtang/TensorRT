cmake_minimum_required(VERSION 3.22)
project(yolov11_demo)

find_package(CUDA REQUIRED)

set(TENSORRT_ROOT "/home/rton/TensorRT/TensorRT-10.11.0.33")
set(OPENCV_INSTALL_DIR "/home/rton/CppProj/TensorProj/3rdparty/opencv-install-x86")

# =============== OpenCV ===============
set(OpenCV_DIR ${OPENCV_INSTALL_DIR}/lib/cmake/opencv4)
find_package(OpenCV REQUIRED NO_MODULE)

# =============== TensorRT ===============
set(TENSORRT_LIB_PATHS
    "${TENSORRT_ROOT}/lib"
    "${TENSORRT_ROOT}/lib64"
    "${TENSORRT_ROOT}/targets/x86_64-linux-gnu/lib"
    "${TENSORRT_ROOT}/targets/x86_64-linux-gnu/lib/stubs"
)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT}
    PATH_SUFFIXES include targets/x86_64-linux-gnu/include)

find_library(TENSORRT_LIBRARY_INFER nvinfer HINTS ${TENSORRT_LIB_PATHS})
find_library(TENSORRT_LIBRARY_ONNX_PARSER nvonnxparser HINTS ${TENSORRT_LIB_PATHS})
find_library(TENSORRT_LIBRARY_PLUGIN nvinfer_plugin HINTS ${TENSORRT_LIB_PATHS})

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY_INFER)
    message(FATAL_ERROR "Cannot find TensorRT")
endif()

# =============== Samples Common 路径 ===============
set(TRT_SAMPLES_COMMON_DIR "${TENSORRT_ROOT}/samples/common")
set(TRT_SAMPLES_DIR "${TENSORRT_ROOT}/samples")

# =============== ✅ 最小化源文件（移除 timing cache 依赖）==============
set(SAMPLES_COMMON_SOURCES
    "${TRT_SAMPLES_COMMON_DIR}/logger.cpp"
    "${TRT_SAMPLES_COMMON_DIR}/sampleOptions.cpp"
    "${TRT_SAMPLES_COMMON_DIR}/sampleDevice.cpp"
    "${TRT_SAMPLES_COMMON_DIR}/sampleUtils.cpp"
    "${TRT_SAMPLES_COMMON_DIR}/getOptions.cpp"
    "${TRT_SAMPLES_COMMON_DIR}/bfloat16.cpp"
)

# ✅ 清空 utils 源文件（不需要 timing cache）
set(SAMPLES_UTILS_SOURCES "")

# =============== 项目源文件 ===============
set(PROJECT_SOURCES
    test/yolov11_test.cpp
    src/tensorrt_infer.cpp
    ${SAMPLES_COMMON_SOURCES}
    ${SAMPLES_UTILS_SOURCES}
)

# =============== 编译可执行文件 ===============
add_executable(yolov11_demo ${PROJECT_SOURCES})

target_include_directories(yolov11_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/test
    ${TENSORRT_INCLUDE_DIR}
    ${TRT_SAMPLES_COMMON_DIR}
    ${TRT_SAMPLES_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(yolov11_demo PRIVATE
    ${TENSORRT_LIBRARY_INFER}
    ${TENSORRT_LIBRARY_ONNX_PARSER}
    ${TENSORRT_LIBRARY_PLUGIN}
    ${CUDA_LIBRARIES}
    cudart
    ${OpenCV_LIBS}
)

# =============== RPATH ===============
set(CMAKE_INSTALL_RPATH 
    "${TENSORRT_ROOT}/targets/x86_64-linux-gnu/lib;${OPENCV_INSTALL_DIR}/lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# =============== 编译选项 ===============
target_compile_options(yolov11_demo PRIVATE
    -std=c++17
    -Wall
    -Wno-deprecated-declarations
)

message(STATUS "✅ yolov11_demo configured (minimal dependencies)")