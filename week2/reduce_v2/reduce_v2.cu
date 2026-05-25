#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/utils.h"


#define BLOCK_SIZE 256   // 每个 block 的线程数

// ------------------------------------------------------------
// 每个 block 内部：使用共享内存树形归约，输出该 block 的部分和
// ------------------------------------------------------------
__global__ void reduce_v2(const float* input, float* partial_sums, int n) {
    extern __shared__ float sdata[];        // 动态共享内存
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // 1. 加载数据到共享内存（越界填 0）
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // 2. 树形归约（要求 blockDim.x 是 2 的幂）
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 3. 每个 block 的线程 0 将部分和写入全局数组 partial_sums
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// ------------------------------------------------------------
// 主机端测试
// ------------------------------------------------------------
int main() {
    initDevice(0);

    const int N = 1 << 20;              // 1,048,576 个元素
    const int blockSize = BLOCK_SIZE;   // 256
    const int gridSize = (N + blockSize - 1) / blockSize;   // 4096

    size_t bytes = N * sizeof(float);
    size_t partialBytes = gridSize * sizeof(float);

    // 主机内存
    float *h_input = (float*)malloc(bytes);
    float *h_partial = (float*)malloc(partialBytes);
    initialData(h_input, N);

    // 设备内存
    float *d_input, *d_partial;
    CHECK(cudaMalloc(&d_input, bytes));
    CHECK(cudaMalloc(&d_partial, partialBytes));

    CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // 启动 kernel（动态共享内存大小：blockSize * sizeof(float)）
    double start = cpuSecond();
    reduce_v2<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_input, d_partial, N);
    CHECK(cudaDeviceSynchronize());
    double gpu_time = cpuSecond() - start;

    // 将 partial_sums 拷贝回主机
    CHECK(cudaMemcpy(h_partial, d_partial, partialBytes, cudaMemcpyDeviceToHost));

    // 主机端累加所有部分和得到最终结果
    float gpu_sum = 0.0f;
    for (int i = 0; i < gridSize; ++i) {
        gpu_sum += h_partial[i];
    }

    // CPU 直接计算总和作为参考
    float cpu_sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        cpu_sum += h_input[i];
    }

    printf("N = %d, blockSize = %d, gridSize = %d\n", N, blockSize, gridSize);
    printf("GPU 耗时: %.3f ms\n", gpu_time * 1000);
    printf("CPU 总和: %f, GPU 总和: %f\n", cpu_sum, gpu_sum);
    float rel_err = fabs(cpu_sum - gpu_sum) / (fabs(cpu_sum) + 1e-8);
    printf("结果验证: %s (rel_err=%e)\n", rel_err < 1e-4 ? "✅ 通过" : "❌ 失败", rel_err);

    free(h_input); free(h_partial);
    CHECK(cudaFree(d_input)); CHECK(cudaFree(d_partial));

    return 0;
}