cmake_minimum_required(VERSION 3.22)

project(sample_onnx_mnist)

#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# 替换原来的 include(../CMakeSamplesTemplate.txt)，添加必要的配置
find_package(CUDA REQUIRED)

# 查找 TensorRT 库
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)
    
find_library(TENSORRT_LIBRARY_INFER nvinfer
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_LIBRARY_ONNX_PARSER nvonnxparser
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_LIBRARY_PLUGIN nvinfer_plugin
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY_INFER OR NOT TENSORRT_LIBRARY_ONNX_PARSER)
    message(FATAL_ERROR "Cannot find TensorRT headers or libraries. Please set TENSORRT_ROOT appropriately.")
endif()

# 添加 TensorRT 各个头文件目录
set(TRT_SAMPLES_COMMON_DIR "${TENSORRT_ROOT}/samples/common")
set(TRT_SAMPLES_UTILS_DIR "${TENSORRT_ROOT}/samples/utils")
set(TRT_SAMPLES_DIR "${TENSORRT_ROOT}/samples")

if(NOT EXISTS "${TRT_SAMPLES_COMMON_DIR}/argsParser.h")
    message(FATAL_ERROR "Cannot find TensorRT samples common directory. Expected location: ${TRT_SAMPLES_COMMON_DIR}")
endif()

include_directories(${TENSORRT_INCLUDE_DIR})
include_directories(${TRT_SAMPLES_COMMON_DIR})  # 为了 "argsParser.h" 这样的直接引用
include_directories(${TRT_SAMPLES_DIR})        # 为了 "utils/timingCache.h" 这样的路径引用
include_directories(${TRT_SAMPLES_UTILS_DIR})  # 添加utils目录

# 添加 samples common 目录中的源文件
set(SAMPLES_COMMON_SOURCES
    "${TRT_SAMPLES_COMMON_DIR}/bfloat16.cpp"
    "${TRT_SAMPLES_COMMON_DIR}/getOptions.cpp"
    "${TRT_SAMPLES_COMMON_DIR}/logger.cpp"
    "${TRT_SAMPLES_COMMON_DIR}/sampleDevice.cpp"
    "${TRT_SAMPLES_COMMON_DIR}/sampleEngines.cpp"
    "${TRT_SAMPLES_COMMON_DIR}/sampleInference.cpp"
    "${TRT_SAMPLES_COMMON_DIR}/sampleOptions.cpp"
    "${TRT_SAMPLES_COMMON_DIR}/sampleReporting.cpp"
    "${TRT_SAMPLES_COMMON_DIR}/sampleUtils.cpp"
)

# 添加 samples utils 目录中的源文件
set(SAMPLES_UTILS_SOURCES
    "${TRT_SAMPLES_UTILS_DIR}/timingCache.cpp"
    "${TRT_SAMPLES_UTILS_DIR}/fileLock.cpp"
)

# 确保所有源文件都存在
foreach(src_file ${SAMPLES_COMMON_SOURCES})
    if(NOT EXISTS ${src_file})
        message(WARNING "Source file does not exist, skipping: ${src_file}")
        list(REMOVE_ITEM SAMPLES_COMMON_SOURCES ${src_file})
    endif()
endforeach()

foreach(src_file ${SAMPLES_UTILS_SOURCES})
    if(NOT EXISTS ${src_file})
        message(WARNING "Source file does not exist, skipping: ${src_file}")
        list(REMOVE_ITEM SAMPLES_UTILS_SOURCES ${src_file})
    endif()
endforeach()

# 合并所有源文件
set(ALL_SOURCES sampleOnnxMNIST.cpp ${SAMPLES_COMMON_SOURCES} ${SAMPLES_UTILS_SOURCES})

add_executable(sample_onnx_mnist ${ALL_SOURCES})

# 链接所有必要的库
target_link_libraries(sample_onnx_mnist 
    ${TENSORRT_LIBRARY_INFER}
    ${TENSORRT_LIBRARY_ONNX_PARSER}
    ${TENSORRT_LIBRARY_PLUGIN}
    ${CUDA_LIBRARIES}
    cudart
)