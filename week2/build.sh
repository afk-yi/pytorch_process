#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ ! -d "build" ]; then
    mkdir build
fi
cd build
cmake ..          # 如果没变化，cmake 会很快
cmake --build .   # 增量编译

echo "✅ 编译完成！可执行文件位于 build/bin/ 下"