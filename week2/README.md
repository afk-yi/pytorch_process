# 编译命令
# 在顶层CMakeLists.txt中添加子目录
cd /data/week2
mkdir build && cd build
cmake ..
cmake --build .      # 一次性编译所有示例

# 简化为:
./build.sh

# 编译单个文件
nvcc hello_world.cu -o hello_world
/usr/local/cuda-12.4/bin/nvcc -O3 -o test_bank test_bank.cu

# 添加新模块
# 会自动更新顶层CMakeLists.txt
# 自动在子目录添加子CMakeLists.txt 会同名的.cu 文件
./new_module.sh module_name
