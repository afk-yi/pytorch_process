// 消除共享内存冲突
#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/utils.h"

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

#define REG_TILE 2

__global__ void reg_tiled_matmul_opt(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int rowBase = blockIdx.y * TILE_SIZE + ty * REG_TILE;
    int colBase = blockIdx.x * TILE_SIZE + tx * REG_TILE;

    float regC[REG_TILE][REG_TILE] = {{0.0f}};

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        int a_col_start = tile * TILE_SIZE;
        int b_row_start = tile * TILE_SIZE;

        // 加载 A 的 2x2 块到 As（无转置）
        for (int i = 0; i < REG_TILE; ++i) {
            int loadRow = blockIdx.y * TILE_SIZE + ty * REG_TILE + i;
            for (int j = 0; j < REG_TILE; ++j) {
                int loadCol = a_col_start + tx * REG_TILE + j;
                if (loadRow < M && loadCol < K)
                    As[ty * REG_TILE + i][tx * REG_TILE + j] = A[loadRow * K + loadCol];
                else
                    As[ty * REG_TILE + i][tx * REG_TILE + j] = 0.0f;
            }
        }

        // 加载 B 的 2x2 块到 Bs（无转置，连续列写入，消除 bank conflict）
        for (int i = 0; i < REG_TILE; ++i) {
            int loadRow = b_row_start + ty * REG_TILE + i;
            for (int j = 0; j < REG_TILE; ++j) {
                int loadCol = blockIdx.x * TILE_SIZE + tx * REG_TILE + j;
                if (loadRow < K && loadCol < N)
                    Bs[ty * REG_TILE + i][tx * REG_TILE + j] = B[loadRow * N + loadCol];
                else
                    Bs[ty * REG_TILE + i][tx * REG_TILE + j] = 0.0f;
            }
        }

        __syncthreads();

        // 计算点积
        for (int k = 0; k < TILE_SIZE; ++k) {
            float a_vals[REG_TILE];
            for (int i = 0; i < REG_TILE; ++i) {
                a_vals[i] = As[ty * REG_TILE + i][k];
            }
            float b_vals[REG_TILE];
            for (int j = 0; j < REG_TILE; ++j) {
                // 直接读取 Bs 的连续列，无 bank conflict
                b_vals[j] = Bs[k][tx * REG_TILE + j];
            }
            for (int i = 0; i < REG_TILE; ++i)
                for (int j = 0; j < REG_TILE; ++j)
                    regC[i][j] += a_vals[i] * b_vals[j];
        }
        __syncthreads();
    }

    // 写回结果
    for (int i = 0; i < REG_TILE; ++i) {
        int row = rowBase + i;
        if (row >= M) continue;
        for (int j = 0; j < REG_TILE; ++j) {
            int col = colBase + j;
            if (col >= N) continue;
            C[row * N + col] = regC[i][j];
        }
    }
}

// ------------------------------------------------------------
// 验证：同时检查绝对误差和相对误差
// ------------------------------------------------------------
bool compareResult(const float* cpu, const float* gpu, int N, float eps=1e-4f) {
    for (int i = 0; i < N; ++i) {
        float abs_err = fabs(cpu[i] - gpu[i]);
        float rel_err = abs_err / (fabs(cpu[i]) + 1e-8f);
        if (abs_err > eps && rel_err > eps) {
            printf("  Mismatch at %d: cpu=%f, gpu=%f, abs_err=%e, rel_err=%e\n",
                   i, cpu[i], gpu[i], abs_err, rel_err);
            return false;
        }
    }
    return true;
}

// CPU 参考实现（与之前相同）
void cpu_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

int main() {
    initDevice(0);

    int M = 2048, N = 2048, K = 2048;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C_gpu = (float*)malloc(sizeC);
    float *h_C_cpu = (float*)malloc(sizeC);

    initialData(h_A, M * K);
    initialData(h_B, K * N);

    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc(&d_A, sizeA));
    CHECK(cudaMalloc(&d_B, sizeB));
    CHECK(cudaMalloc(&d_C, sizeC));

    CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    dim3 blockDim(TILE_SIZE / REG_TILE, TILE_SIZE / REG_TILE); // (16,16)
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

    double start = cpuSecond();
    reg_tiled_matmul_opt<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CHECK(cudaDeviceSynchronize());
    double gpu_time = cpuSecond() - start;

    CHECK(cudaMemcpy(h_C_gpu, d_C, sizeC, cudaMemcpyDeviceToHost));

    start = cpuSecond();
    cpu_matmul(h_A, h_B, h_C_cpu, M, N, K);
    double cpu_time = cpuSecond() - start;

    printf("GPU 耗时: %.3f ms\n", gpu_time * 1000);
    printf("CPU 耗时: %.3f ms\n", cpu_time * 1000);
    printf("加速比: %.2fx\n", cpu_time / gpu_time);

    bool ok = compareResult(h_C_cpu, h_C_gpu, M * N);
    printf("结果验证: %s\n", ok ? "通过" : "失败");

    free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);
    CHECK(cudaFree(d_A)); CHECK(cudaFree(d_B)); CHECK(cudaFree(d_C));

    return 0;
}