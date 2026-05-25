#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include "../include/utils.h"

// -------- 两个 kernel 各自的 tile 配置 --------
// 基础分块: 每个线程算 1 个元素
#define TILED_TILE_SIZE 32

// 寄存器分块: 每个线程算 2x2 个元素
#define REG_TILE_SIZE 32
#define REG_TILE 2

// ------------------------------------------------------------
// 基础分块版本：每个线程算 1 个 C 元素
// ------------------------------------------------------------
__global__ void tiled_matmul(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int M, int N, int K) {
    __shared__ float As[TILED_TILE_SIZE][TILED_TILE_SIZE];
    __shared__ float Bs[TILED_TILE_SIZE][TILED_TILE_SIZE];

    int row = blockIdx.y * TILED_TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILED_TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int tile = 0; tile < (K + TILED_TILE_SIZE - 1) / TILED_TILE_SIZE; ++tile) {
        int a_col_start = tile * TILED_TILE_SIZE;
        int b_row_start = tile * TILED_TILE_SIZE;

        if (row < M && a_col_start + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col_start + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && b_row_start + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(b_row_start + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILED_TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ------------------------------------------------------------
// 寄存器分块版本：每个线程算 2x2 个 C 元素
// ------------------------------------------------------------
__global__ void reg_tiled_matmul(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int N, int K) {
    __shared__ float As[REG_TILE_SIZE][REG_TILE_SIZE];
    __shared__ float Bs[REG_TILE_SIZE][REG_TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int rowBase = blockIdx.y * REG_TILE_SIZE + ty * REG_TILE;
    int colBase = blockIdx.x * REG_TILE_SIZE + tx * REG_TILE;

    float regC[REG_TILE][REG_TILE] = {{0.0f}};

    for (int tile = 0; tile < (K + REG_TILE_SIZE - 1) / REG_TILE_SIZE; ++tile) {
        int a_col_start = tile * REG_TILE_SIZE;
        int b_row_start = tile * REG_TILE_SIZE;

        // 每个线程加载一个 2x2 块到 As
        for (int i = 0; i < REG_TILE; ++i) {
            int loadRow = blockIdx.y * REG_TILE_SIZE + ty * REG_TILE + i;
            for (int j = 0; j < REG_TILE; ++j) {
                int loadCol = a_col_start + tx * REG_TILE + j;
                if (loadRow < M && loadCol < K)
                    As[ty * REG_TILE + i][tx * REG_TILE + j] = A[loadRow * K + loadCol];
                else
                    As[ty * REG_TILE + i][tx * REG_TILE + j] = 0.0f;
            }
        }

        // 每个线程加载一个 2x2 块到 Bs
        for (int i = 0; i < REG_TILE; ++i) {
            int loadRow = b_row_start + ty * REG_TILE + i;
            for (int j = 0; j < REG_TILE; ++j) {
                int loadCol = blockIdx.x * REG_TILE_SIZE + tx * REG_TILE + j;
                if (loadRow < K && loadCol < N)
                    Bs[ty * REG_TILE + i][tx * REG_TILE + j] = B[loadRow * N + loadCol];
                else
                    Bs[ty * REG_TILE + i][tx * REG_TILE + j] = 0.0f;
            }
        }

        __syncthreads();

        for (int k = 0; k < REG_TILE_SIZE; ++k) {
            float a_val[REG_TILE];
            for (int i = 0; i < REG_TILE; ++i) {
                int rowIdx = ty * REG_TILE + i;
                a_val[i] = As[rowIdx][k];
            }
            float b_val[REG_TILE];
            for (int j = 0; j < REG_TILE; ++j) {
                int colIdx = tx * REG_TILE + j;
                b_val[j] = Bs[k][colIdx];
            }
            for (int i = 0; i < REG_TILE; ++i)
                for (int j = 0; j < REG_TILE; ++j)
                    regC[i][j] += a_val[i] * b_val[j];
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
// CPU 参考实现
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

// ------------------------------------------------------------
// 命令行参数解析
// ------------------------------------------------------------

// ------------------------------------------------------------
// 主函数
// ------------------------------------------------------------
int main(int argc, char** argv) {
    initDevice(0);

    int M = 2048, N = 2048, K = 2048;
    // 参数: 1 = tiled, 2 = reg (默认 reg)
    bool use_reg = (argc < 2 || atoi(argv[1]) != 1);

    printf("矩阵维度: M=%d, N=%d, K=%d\n", M, N, K);
    printf("核函数: %s\n", use_reg ? "reg_tiled_matmul (REG_TILE=2)" : "tiled_matmul");

    size_t sizeA = (size_t)M * K * sizeof(float);
    size_t sizeB = (size_t)K * N * sizeof(float);
    size_t sizeC = (size_t)M * N * sizeof(float);

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

    // 配置内核启动参数
    dim3 blockDim, gridDim;
    if (use_reg) {
        blockDim = dim3(REG_TILE_SIZE / REG_TILE, REG_TILE_SIZE / REG_TILE);
        gridDim  = dim3((N + REG_TILE_SIZE - 1) / REG_TILE_SIZE,
                        (M + REG_TILE_SIZE - 1) / REG_TILE_SIZE);
    } else {
        blockDim = dim3(TILED_TILE_SIZE, TILED_TILE_SIZE);
        gridDim  = dim3((N + TILED_TILE_SIZE - 1) / TILED_TILE_SIZE,
                        (M + TILED_TILE_SIZE - 1) / TILED_TILE_SIZE);
    }

    // GPU 计算
    double start = cpuSecond();
    if (use_reg) {
        reg_tiled_matmul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    } else {
        tiled_matmul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK(cudaDeviceSynchronize());
    double gpu_time = cpuSecond() - start;

    CHECK(cudaMemcpy(h_C_gpu, d_C, sizeC, cudaMemcpyDeviceToHost));

    // CPU 参考计算
    start = cpuSecond();
    cpu_matmul(h_A, h_B, h_C_cpu, M, N, K);
    double cpu_time = cpuSecond() - start;

    // 输出结果
    printf("CPU 耗时: %f ms\n", cpu_time * 1000);
    printf("GPU 耗时: %f ms\n", gpu_time * 1000);
    printf("加速比: %.2fx\n", cpu_time / gpu_time);

    // 验证
    bool ok = compareResult(h_C_cpu, h_C_gpu, M * N);
    printf("结果验证: %s\n", ok ? "通过" : "失败");

    // 清理
    free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);
    CHECK(cudaFree(d_A)); CHECK(cudaFree(d_B)); CHECK(cudaFree(d_C));

    return ok ? 0 : 1;
}
