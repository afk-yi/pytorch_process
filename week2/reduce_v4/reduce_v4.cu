#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/utils.h"


#define BLOCK_SIZE 256   // 每个 block 的线程数（必须是 32 的倍数）

// ------------------------------------------------------------
// Warp 内归约：使用 shuffle 指令，不需要共享内存
// ------------------------------------------------------------
__device__ float warpReduceSum(float val) {
    // 一个 warp 有 32 个线程，mask = 0xffffffff 表示 warp 内所有线程都参与
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;   // 只有每个 warp 的 lane 0 得到完整的和
}

// ------------------------------------------------------------
// Block 内归约：先用 warp 归约，再通过共享内存汇总 warp 的部分和
// ------------------------------------------------------------
__device__ float blockReduceSum(float val) {
    __shared__ float warpPartialSums[32];   // 最多 32 个 warp
    int lane = threadIdx.x & 31;                   // 线程在 warp 内的编号 (0-31)
    int warpId = threadIdx.x >> 5;                 // 线程所在的 warp 编号

    // 第1步：每个 warp 内部归约
    val = warpReduceSum(val);

    // 第2步：每个 warp 的 lane 0 将结果写入共享内存
    if (lane == 0) {
        warpPartialSums[warpId] = val;
    }
    __syncthreads();

    // 第3步：使用第一个 warp 汇总所有 warpPartialSums 的值
    if (warpId == 0) {
        // 收集所有 warp 的结果（不足 32 个 warp 时补 0）
        float sum = (lane < blockDim.x / 32) ? warpPartialSums[lane] : 0.0f;
        sum = warpReduceSum(sum);
        if (lane == 0) {
            warpPartialSums[0] = sum;   // 复用共享内存的第一个位置存放最终结果
        }
    }
    __syncthreads();

    return warpPartialSums[0];
}

// ------------------------------------------------------------
// Reduce V3 内核：每个 block 归约自己的数据，输出到全局数组 partial_sums
// （与 v2 接口相同，但块内实现不同）
// ------------------------------------------------------------
__global__ void reduce_v4(const float* input, float* partial_sums, int n) {
    // extern __shared__ float sdata[];   // 动态共享内存（只用于存放输入数据，不是归约用）
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // // 加载数据到共享内存（越界填 0）
    // sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    // __syncthreads();

    // // 使用 warp shuffle + 共享内存汇总 warp 结果 进行 block 级归约
    // float sum = blockReduceSum(sdata[tid]);
    float val = (idx < n) ? input[idx] : 0.0f;
    float sum = blockReduceSum(val);

    // 每个 block 的线程 0 将部分和写入全局数组
    if (tid == 0) {
        partial_sums[blockIdx.x] = sum;
    }
}

// ------------------------------------------------------------
// 主机端测试
// ------------------------------------------------------------
int main() {
    initDevice(0);

    const int N = 1 << 20;              // 1,048,576
    const int blockSize = BLOCK_SIZE;   // 256
    const int gridSize = (N + blockSize - 1) / blockSize;   // 4096

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
    reduce_v4<<<gridSize, blockSize>>>(d_input, d_partial, N);
    CHECK(cudaDeviceSynchronize());
    double gpu_time = cpuSecond() - start;

    CHECK(cudaMemcpy(h_partial, d_partial, partialBytes, cudaMemcpyDeviceToHost));

    // 主机端累加部分和
    float gpu_sum = 0.0f;
    for (int i = 0; i < gridSize; ++i) gpu_sum += h_partial[i];

    // CPU 参考和
    float cpu_sum = 0.0f;
    for (int i = 0; i < N; ++i) cpu_sum += h_input[i];

    printf("N = %d, blockSize = %d, gridSize = %d\n", N, blockSize, gridSize);
    printf("GPU 耗时: %.3f ms\n", gpu_time * 1000);
    printf("CPU 总和: %f, GPU 总和: %f\n", cpu_sum, gpu_sum);
    float rel_err = fabs(cpu_sum - gpu_sum) / (fabs(cpu_sum) + 1e-8);
    printf("结果验证: %s (rel_err=%e)\n", rel_err < 1e-4 ? "✅ 通过" : "❌ 失败", rel_err);

    free(h_input); free(h_partial);
    CHECK(cudaFree(d_input)); CHECK(cudaFree(d_partial));

    return 0;
}