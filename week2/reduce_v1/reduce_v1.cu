#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/utils.h"


// 每个线程块处理的元素数量（通常设为 256）
#define BLOCK_SIZE 256

// ------------------------------------------------------------------
// 每个线程块内部：用共享内存做树形规约，返回该块的部分和
// ------------------------------------------------------------------
__device__ float blockSum(const float* input, int n, int tid, int idx, float* sdata) {
    // 加载数据到共享内存（越界填 0）
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // 树形规约（要求 BLOCK_SIZE 是 2 的幂）
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    return sdata[0];
}

// ------------------------------------------------------------------
// Reduce V1 内核：每个 block 归约自己的数据，然后用原子操作累加到全局变量
// ------------------------------------------------------------------
__global__ void reduce_v1(const float* input, float* output, int n) {
    extern __shared__ float sdata[];     // 动态共享内存
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // 计算当前 block 的部分和
    float sum = blockSum(input, n, tid, idx, sdata);

    // 用原子操作将部分和加到全局 output[0]
    if (tid == 0) {
        atomicAdd(output, sum);
    }
}

// ------------------------------------------------------------------
// 主机端测试
// ------------------------------------------------------------------
int main() {
    initDevice(0);

    const int N = 1 << 20;               // 1M 个浮点数
    const int blockSize = BLOCK_SIZE;    // 256
    const int gridSize = (N + blockSize - 1) / blockSize;

    size_t bytes = N * sizeof(float);
    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(sizeof(float));   // 只存一个最终结果

    // 初始化输入数据（范围 0~65.535）
    initialData(h_input, N);

    // 分配设备内存
    float *d_input, *d_output;
    CHECK(cudaMalloc(&d_input, bytes));
    CHECK(cudaMalloc(&d_output, sizeof(float)));

    // 将输入拷贝到设备，并将设备上的 output 初始化为 0
    CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_output, 0, sizeof(float)));

    // 启动内核（动态共享内存大小：blockSize * sizeof(float)）
    double start = cpuSecond();
    reduce_v1<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, N);
    CHECK(cudaDeviceSynchronize());
    double gpu_time = cpuSecond() - start;

    // 拷贝结果回主机
    CHECK(cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    // CPU 计算总和作为参考
    float cpu_sum = 0.0f;
    for (int i = 0; i < N; ++i) cpu_sum += h_input[i];

    printf("N = %d, blockSize = %d, gridSize = %d\n", N, blockSize, gridSize);
    printf("GPU 耗时: %.3f ms\n", gpu_time * 1000);
    printf("CPU 总和: %f, GPU 总和: %f\n", cpu_sum, h_output[0]);
    float rel_err = fabs(cpu_sum - h_output[0]) / (fabs(cpu_sum) + 1e-8);
    bool ok = rel_err < 1e-4;
    printf("结果验证: %s (rel_err=%e)\n", ok ? "✅ 通过" : "❌ 失败", rel_err);

    free(h_input); free(h_output);
    CHECK(cudaFree(d_input)); CHECK(cudaFree(d_output));

    return 0;
}