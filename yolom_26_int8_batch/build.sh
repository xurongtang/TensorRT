#!/bin/bash

# Create and enter build directory
mkdir -p build
cd build

# Configure
cmake .. || { echo "CMake failed"; exit 1; }

# Build
make -j$(nproc) || { echo "Build failed"; exit 1; }

echo ""
echo "Build successful! Executable: ./build/test_yolo26_int8_batch"
echo ""
echo "Usage:"
echo "  ./build/test_yolo26_int8_batch <engine_file> <image1> [image2] [image3] [image4]"
echo ""
echo "Example:"
echo "  ./build/test_yolo26_int8_batch ../quant_convert/yolo26m_int8.engine ../asset/000000002261.jpg"