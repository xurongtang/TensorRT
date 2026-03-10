#!/bin/bash

rm -rf build
# Check if build directory exists, create if not
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir build
fi

# Change to build directory
cd build

# Run cmake
echo "Running cmake..."
cmake ..

# Build the project
echo "Building the project..."
make -j$(nproc)

echo "Build process completed!"