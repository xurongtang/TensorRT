#!/bin/bash

# 设置脚本在遇到错误时退出
set -e

echo "开始构建 YOLO26 TensorRT 项目..."

# 获取当前脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 创建构建目录
BUILD_DIR="${SCRIPT_DIR}/build"
rm -rf build
mkdir -p "${BUILD_DIR}"

# 进入构建目录
cd "${BUILD_DIR}"

echo "正在配置项目..."
# 配置项目
cmake .. \
    -DCMAKE_BUILD_TYPE=Release

echo "正在编译项目..."
# 编译项目
make -j$(nproc)

echo "构建完成!"
echo "可执行文件位置: ${BUILD_DIR}/test_yolo26"

# 返回原始目录
cd - > /dev/null

echo ""
echo "使用方法:"
echo "  ${BUILD_DIR}/test_yolo26 <engine_file> <image_path>"
echo ""
echo "例如:"
echo "  ${BUILD_DIR}/test_yolo26 /path/to/model.engine /path/to/image.jpg"