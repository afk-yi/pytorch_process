#!/bin/bash
# 用法: ./new_module.sh module_name

if [ -z "$1" ]; then
    echo "用法: $0 <模块名>"
    exit 1
fi

MODULE_NAME="$1"
MODULE_DIR="/data/week2/$MODULE_NAME"

if [ -d "$MODULE_DIR" ]; then
    echo "错误: 目录 $MODULE_DIR 已存在"
    exit 1
fi

# 创建目录和文件
mkdir -p "$MODULE_DIR"

# 创建 CMakeLists.txt
cat > "$MODULE_DIR/CMakeLists.txt" << EOF
add_executable(${MODULE_NAME} ${MODULE_NAME}.cu)
EOF

# 创建 .cu 文件模板
cat > "$MODULE_DIR/${MODULE_NAME}.cu" << EOF
#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/utils.h"

int main(int argc, char** argv) {
    printf("Hello from ${MODULE_NAME}!\n");
    return 0;
}
EOF

# 更新根目录 CMakeLists.txt（如果还没添加这个子目录）
ROOT_CMAKE="/data/week2/CMakeLists.txt"
if ! grep -q "add_subdirectory(${MODULE_NAME})" "$ROOT_CMAKE"; then
    # 确保文件以换行符结尾
    tail -c1 "$ROOT_CMAKE" | read -r _ || echo "" >> "$ROOT_CMAKE"
    echo "add_subdirectory(${MODULE_NAME})" >> "$ROOT_CMAKE"
    echo "已添加到根 CMakeLists.txt"
fi

echo "已创建: $MODULE_DIR/"
echo "  - CMakeLists.txt"
echo "  - ${MODULE_NAME}.cu"