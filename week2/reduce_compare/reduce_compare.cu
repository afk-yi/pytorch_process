#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/utils.h"


#define BLOCK_SIZE 256
#define N (1 << 24)          // 1,048,576 个元素
#define GRID_SIZE ((N + BLOCK_SIZE - 1) / BLOCK_SIZE)  // 4096

// ------------------------------------------------------------
// 公共工具：检查相对误差
// ------------------------------------------------------------
bool check_result(float cpu_sum, float gpu_sum) {
    float rel_err = fabs(cpu_sum - gpu_sum) / (fabs(cpu_sum) + 1e-8);
    printf("  CPU sum = %f, GPU sum = %f, rel_err = %e\n",
           cpu_sum, gpu_sum, rel_err);
    if (rel_err < 1e-3) {
        printf("✅ 通过 (rel_err=%e)\n", rel_err);
        return true;
    } else {
        printf("❌ 失败 (rel_err=%e)\n", rel_err);
        return false;
    }
}

// ------------------------------------------------------------
// reduce_v1：原子操作版本，直接输出一个总和
// ------------------------------------------------------------
__global__ void reduce_v1(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(output, sdata[0]);
}

// ------------------------------------------------------------
// reduce_v2：无原子操作，输出部分和数组
// ------------------------------------------------------------
__global__ void reduce_v2(const float* input, float* partial_sums, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial_sums[blockIdx.x] = sdata[0];
}

// ------------------------------------------------------------
// reduce_v3：warp shuffle 版本
// ------------------------------------------------------------
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

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

__global__ void reduce_v3(const float* input, float* partial_sums, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    float sum = blockReduceSum(sdata[tid]);
    if (tid == 0) partial_sums[blockIdx.x] = sum;
}

// ------------------------------------------------------------
// CPU 参考求和
// ------------------------------------------------------------
float cpu_sum_all(const float* arr, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += arr[i];
    return sum;
}

// ------------------------------------------------------------
// 主机端累加部分和数组
// ------------------------------------------------------------
float reduce_partial_sums(const float* partials, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) sum += partials[i];
    return sum;
}

// ------------------------------------------------------------
// 测试一个 kernel 的函数模板（避免重复代码）
// ------------------------------------------------------------
void test_kernel(const char* name,
                 void (*kernel)(const float*, float*, int),
                 bool use_atomic,      // v1 需要原子累加到一个变量，output 大小为 1
                 float* d_input,
                 float* d_output,
                 int output_size,      // 1 或 GRID_SIZE
                 float cpu_sum,
                 bool warmup) {
    size_t shared_mem = BLOCK_SIZE * sizeof(float);
    dim3 grid(GRID_SIZE, 1), block(BLOCK_SIZE, 1);

    // Warmup
    if (warmup) {
        if (use_atomic) {
            CHECK(cudaMemset(d_output, 0, output_size * sizeof(float)));
            kernel<<<grid, block, shared_mem>>>(d_input, d_output, N);
            CHECK(cudaDeviceSynchronize());
        } else {
            kernel<<<grid, block, shared_mem>>>(d_input, d_output, N);
            CHECK(cudaDeviceSynchronize());
        }
    }

    // 重置输出（原子操作版本需要清零；非原子版本不需要，因为会被后续完全覆盖）
    if (use_atomic) {
        CHECK(cudaMemset(d_output, 0, output_size * sizeof(float)));
    }

    // 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    kernel<<<grid, block, shared_mem>>>(d_input, d_output, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 拷贝结果
    float *h_output = (float*)malloc(output_size * sizeof(float));
    CHECK(cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));

    // 计算 GPU 总和
    float gpu_sum;
    if (use_atomic) {
        gpu_sum = h_output[0];
    } else {
        gpu_sum = reduce_partial_sums(h_output, output_size);
    }

    printf("%-12s GPU 耗时: %.3f ms, 结果: ", name, elapsed_ms);
    printf("GPU sum = %f\n", gpu_sum);
    check_result(cpu_sum, gpu_sum);

    free(h_output);
}

// ------------------------------------------------------------
// 主函数
// ------------------------------------------------------------
int main() {
    initDevice(0);

    // 分配主机内存并初始化随机数据
    size_t bytes = N * sizeof(float);
    float *h_input = (float*)malloc(bytes);
    initialData(h_input, N);

    // CPU 参考和
    float cpu_sum = cpu_sum_all(h_input, N);
    printf("CPU 总和: %f\n\n", cpu_sum);

    // 分配设备内存（输入数据每个 kernel 测试时共享，但每个测试需要重置输出）
    float *d_input, *d_output_v1, *d_output_v2, *d_output_v3;
    CHECK(cudaMalloc(&d_input, bytes));
    CHECK(cudaMalloc(&d_output_v1, sizeof(float)));
    CHECK(cudaMalloc(&d_output_v2, GRID_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&d_output_v3, GRID_SIZE * sizeof(float)));

    // 拷贝输入到设备（只拷贝一次，因为输入不变）
    CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    bool warmup = true;   // 开启 warmup

    // 测试 v1
    test_kernel("reduce_v1", (void(*)(const float*,float*,int))reduce_v1, true,
                d_input, d_output_v1, 1, cpu_sum, warmup);

    // 测试 v2
    test_kernel("reduce_v2", (void(*)(const float*,float*,int))reduce_v2, false,
                d_input, d_output_v2, GRID_SIZE, cpu_sum, warmup);

    // 测试 v3
    test_kernel("reduce_v3", (void(*)(const float*,float*,int))reduce_v3, false,
                d_input, d_output_v3, GRID_SIZE, cpu_sum, warmup);

    // 清理
    free(h_input);
    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output_v1));
    CHECK(cudaFree(d_output_v2));
    CHECK(cudaFree(d_output_v3));

    return 0;
}