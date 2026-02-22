#!/bin/bash

# 设置TensorRT根目录
export TENSORRT_ROOT="/home/rton/TensorRT/TensorRT-10.11.0.33"
export LD_LIBRARY_PATH="$TENSORRT_ROOT/lib:$LD_LIBRARY_PATH"

cd /home/rton/CppProj/TensorProj/install_check/demo
rm -rf build
mkdir build && cd build

echo "Building sample_onnx_mnist..."
echo "TensorRT Root: $TENSORRT_ROOT"

cmake .. \
   -DTENSORRT_ROOT="$TENSORRT_ROOT" \
   -DCMAKE_BUILD_TYPE=Release

if [ $? -ne 0 ]; then
    echo "CMake configuration failed!"
    exit 1
fi

cmake --build . --parallel 4

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

# 运行示例（需要有MNIST数据）
if [ -f "./sample_onnx_mnist" ]; then
    echo "Build successful! Executable is at $(pwd)/sample_onnx_mnist"
else
    echo "Build completed but executable not found."
fi