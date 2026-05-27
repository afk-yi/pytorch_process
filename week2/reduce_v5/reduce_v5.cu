#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/utils.h"

#define BLOCK_SIZE 256
#define ITEMS_PER_THREAD 4   // 每个线程处理 4 个元素

// Warp 内归约（与 v4 相同）
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Block 内归约（与 v4 相同）
__device__ float blockReduceSum(float val) {
    __shared__ float warpPartialSums[32];
    int lane = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;

    val = warpReduceSum(val);
    if (lane == 0) warpPartialSums[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        float sum = (lane < blockDim.x / 32) ? warpPartialSums[lane] : 0.0f;
        sum = warpReduceSum(sum);
        if (lane == 0) warpPartialSums[0] = sum;
    }
    __syncthreads();

    return warpPartialSums[0];
}

// reduce_v5 内核：每个线程先累加 ITEMS_PER_THREAD 个元素
__global__ void reduce_v5(const float* input, float* partial_sums, int n) {
    int tid = threadIdx.x;
    int block_start = blockIdx.x * blockDim.x * ITEMS_PER_THREAD;
    int idx = block_start + tid * ITEMS_PER_THREAD;

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        int pos = idx + i;
        if (pos < n) sum += input[pos];
    }

    float block_sum = blockReduceSum(sum);
    if (tid == 0) {
        partial_sums[blockIdx.x] = block_sum;
    }
}

int main() {
    initDevice(0);

    const int N = 1 << 20;                      // 1,048,576
    const int blockSize = BLOCK_SIZE;
    const int itemsPerBlock = blockSize * ITEMS_PER_THREAD;
    const int gridSize = (N + itemsPerBlock - 1) / itemsPerBlock;

    size_t bytes = N * sizeof(float);
    size_t partialBytes = gridSize * sizeof(float);

    float *h_input = (float*)malloc(bytes);
    float *h_partial = (float*)malloc(partialBytes);
    initialData(h_input, N);

    float *d_input, *d_partial;
    CHECK(cudaMalloc(&d_input, bytes));
    CHECK(cudaMalloc(&d_partial, partialBytes));
    CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    double start = cpuSecond();
    reduce_v5<<<gridSize, blockSize>>>(d_input, d_partial, N);
    CHECK(cudaDeviceSynchronize());
    double gpu_time = cpuSecond() - start;

    CHECK(cudaMemcpy(h_partial, d_partial, partialBytes, cudaMemcpyDeviceToHost));

    float gpu_sum = 0.0f;
    for (int i = 0; i < gridSize; ++i) gpu_sum += h_partial[i];

    float cpu_sum = 0.0f;
    for (int i = 0; i < N; ++i) cpu_sum += h_input[i];

    printf("N = %d, blockSize = %d, ITEMS_PER_THREAD = %d, gridSize = %d\n",
           N, blockSize, ITEMS_PER_THREAD, gridSize);
    printf("GPU 耗时: %.3f ms\n", gpu_time * 1000);
    printf("CPU 总和: %f, GPU 总和: %f\n", cpu_sum, gpu_sum);
    float rel_err = fabs(cpu_sum - gpu_sum) / (fabs(cpu_sum) + 1e-8);
    printf("结果验证: %s (rel_err=%e)\n", rel_err < 1e-4 ? "✅ 通过" : "❌ 失败", rel_err);

    free(h_input); free(h_partial);
    CHECK(cudaFree(d_input)); CHECK(cudaFree(d_partial));

    return 0;
}