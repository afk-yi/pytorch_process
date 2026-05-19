cmake_minimum_required(VERSION 3.12)
project(CUDASamples LANGUAGES CXX CUDA)

# 设置所有 CUDA 目标的默认架构（根据你的 GPU 修改）
#set(CMAKE_CUDA_ARCHITECTURES "75" CACHE STRING "CUDA architectures")
set(CMAKE_CUDA_ARCHITECTURES "60;70;75" CACHE STRING "CUDA architectures")

# 设置可执行文件的输出目录（集中到 build/bin/ 下）
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(hello_world)
add_subdirectory(sum_array)
add_subdirectory(check_dimension)
add_subdirectory(grid_block)
add_subdirectory(thread_index)
add_subdirectory(sum_matrix)
add_subdirectory(device_information)
add_subdirectory(sum_arrays_uniform_memory)
