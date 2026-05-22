#!/bin/bash
set -e
cd "$(dirname "$0")"

# 删除旧的 debug 构建目录（如果存在），确保干净的环境
if [ -d "build_debug" ]; then
    echo "🗑️  Removing old build_debug directory..."
    rm -rf build_debug
fi

echo "🔨 Creating build_debug directory..."
mkdir build_debug
cd build_debug

echo "⚙️  Running CMake with Debug mode..."
cmake -DCMAKE_BUILD_TYPE=Debug ..

echo "📦 Building project..."
cmake --build . -j$(nproc)   # -j 加速编译

echo "✅ Debug compilation finished!"
echo "📁 Executable is located at: build_debug/bin/share_memory_read_data"