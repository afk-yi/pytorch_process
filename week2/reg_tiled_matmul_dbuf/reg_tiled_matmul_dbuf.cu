#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include "../include/utils.h"

#define TILE_SIZE 32
#define REG_TILE 4

__global__ void reg_tiled_matmul_dbuf(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int M, int N, int K) {
    __shared__ float As[2][TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int rowBase = blockIdx.y * TILE_SIZE + ty * REG_TILE;
    int colBase = blockIdx.x * TILE_SIZE + tx * REG_TILE;

    float regC[REG_TILE][REG_TILE] = {{0.0f}};

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // Prologue: issue pipeline copies for tile 0 into buffer 0
    for (int i = 0; i < REG_TILE; ++i) {
        int rowA = blockIdx.y * TILE_SIZE + ty * REG_TILE + i;
        __pipeline_memcpy_async(&As[0][ty * REG_TILE + i][tx * REG_TILE],
                                 &A[rowA * K + tx * REG_TILE],
                                 REG_TILE * sizeof(float));

        int rowB = ty * REG_TILE + i;
        __pipeline_memcpy_async(&Bs[0][ty * REG_TILE + i][tx * REG_TILE],
                                 &B[rowB * N + blockIdx.x * TILE_SIZE + tx * REG_TILE],
                                 REG_TILE * sizeof(float));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    // Main loop: compute tile t while tile t+1 loads asynchronously
    for (int tile = 0; tile < numTiles; ++tile) {
        int comp_buf = tile & 1;
        int next_buf = (tile + 1) & 1;

        // Issue async copies for next tile
        if (tile + 1 < numTiles) {
            int a_col = (tile + 1) * TILE_SIZE;
            int b_row = (tile + 1) * TILE_SIZE;
            for (int i = 0; i < REG_TILE; ++i) {
                int rowA = blockIdx.y * TILE_SIZE + ty * REG_TILE + i;
                __pipeline_memcpy_async(&As[next_buf][ty * REG_TILE + i][tx * REG_TILE],
                                         &A[rowA * K + a_col + tx * REG_TILE],
                                         REG_TILE * sizeof(float));

                int rowB = b_row + ty * REG_TILE + i;
                __pipeline_memcpy_async(&Bs[next_buf][ty * REG_TILE + i][tx * REG_TILE],
                                         &B[rowB * N + blockIdx.x * TILE_SIZE + tx * REG_TILE],
                                         REG_TILE * sizeof(float));
            }
            __pipeline_commit();
        }

        // Compute tile t from comp_buf (overlaps with async load of tile t+1)
        for (int k = 0; k < TILE_SIZE; ++k) {
            float a_val[REG_TILE];
            #pragma unroll
            for (int i = 0; i < REG_TILE; ++i)
                a_val[i] = As[comp_buf][ty * REG_TILE + i][k];

            float b_val[REG_TILE];
            #pragma unroll
            for (int j = 0; j < REG_TILE; ++j)
                b_val[j] = Bs[comp_buf][k][tx * REG_TILE + j];

            #pragma unroll
            for (int i = 0; i < REG_TILE; ++i)
                #pragma unroll
                for (int j = 0; j < REG_TILE; ++j)
                    regC[i][j] += a_val[i] * b_val[j];
        }

        // Wait for next tile's async copies to complete
        if (tile + 1 < numTiles) {
            __pipeline_wait_prior(0);
            __syncthreads();
        }
    }

    // Write back 4x4 result
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

// CPU 参考实现
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

    dim3 blockDim(TILE_SIZE / REG_TILE, TILE_SIZE / REG_TILE); // (8,8)
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

    double start = cpuSecond();
    reg_tiled_matmul_dbuf<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
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
