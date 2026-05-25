#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/utils.h"
#include <math.h>   // for fabs

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

bool compareResult(float* cpu, float* gpu, int N, float eps=1e-4) {
    for (int i=0; i<N; ++i) {
        float abs_err = fabs(cpu[i] - gpu[i]);
        float rel_err = abs_err / (fabs(cpu[i]) + 1e-8);
        if (abs_err > eps && rel_err > 1e-4) {
            printf("Mismatch at %d: cpu=%f, gpu=%f, abs_err=%e\n", i, cpu[i], gpu[i], abs_err);
            return false;
        }
    }
    printf("Check result success (eps=%e)!\n", eps);
    return true;
}

// ------------------------------------------------------------
// 分块矩阵乘法内核
// ------------------------------------------------------------
__global__ void tiled_matmul(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        int a_col_start = tile * TILE_SIZE;
        int b_row_start = tile * TILE_SIZE;

        // 加载 A 的子块
        if (row < M && a_col_start + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col_start + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 加载 B 的子块
        if (col < N && b_row_start + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(b_row_start + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // 计算子块内的点积
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ------------------------------------------------------------
// CPU 参考实现（用于验证）
// ------------------------------------------------------------
void cpu_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ------------------------------------------------------------
// 主函数
// ------------------------------------------------------------
int main(int argc, char** argv) {
    // 选择设备（默认设备0）
    initDevice(0);

    // 矩阵维度（可以修改）
    int M = 2048;
    int N = 2048;
    int K = 2048;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // 分配主机内存
    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C_gpu = (float*)malloc(sizeC);
    float *h_C_cpu = (float*)malloc(sizeC);

    // 初始化随机数据
    initialData(h_A, M * K);
    initialData(h_B, K * N);

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc(&d_A, sizeA));
    CHECK(cudaMalloc(&d_B, sizeB));
    CHECK(cudaMalloc(&d_C, sizeC));

    // 拷贝数据到设备
    CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // 配置内核启动参数
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

    // 计时并启动内核
    double start = cpuSecond();
    tiled_matmul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CHECK(cudaDeviceSynchronize());      // 等待内核完成并检查错误
    double gpu_time = cpuSecond() - start;

    // 拷贝结果回主机
    CHECK(cudaMemcpy(h_C_gpu, d_C, sizeC, cudaMemcpyDeviceToHost));

    // CPU 计算（作为基准）
    start = cpuSecond();
    cpu_matmul(h_A, h_B, h_C_cpu, M, N, K);
    double cpu_time = cpuSecond() - start;

    // 验证结果
    printf("GPU 耗时: %f ms\n", gpu_time * 1000);
    printf("CPU 耗时: %f ms\n", cpu_time * 1000);
    printf("加速比: %.2fx\n", cpu_time / gpu_time);

    compareResult(h_C_cpu, h_C_gpu, M * N);

    // 清理内存
    free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);
    CHECK(cudaFree(d_A)); CHECK(cudaFree(d_B)); CHECK(cudaFree(d_C));

    return 0;
}