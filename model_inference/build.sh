#!/bin/bash

# 设置构建目录
BUILD_DIR=build

# 清理旧的构建目录
if [ -d "$BUILD_DIR" ]; then
  rm -rf $BUILD_DIR
fi

# 创建新的构建目录
mkdir $BUILD_DIR
cd $BUILD_DIR

# 运行 cmake 配置项目
cmake ..

# 编译项目
make

# 安装（可选）
# make install

# 输出库路径和测试路径
echo "Libraries are built at: $(pwd)/lib"
echo "Executable binaries are at: $(pwd)/bin"