#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// 可配置参数
#define DEFAULT_REPEATS 1
#define MATRIX_DIM 64            // 矩阵大小 64x64 (较小，适合快速测试)
#define TILE_DIM 32
#define RECT_X 32
#define RECT_Y 16
#define IPAD 1

typedef void (*kernel_t)(const float*, float*);

// 运行内核并返回平均耗时 (ms)
double run_kernel(kernel_t kernel, dim3 grid, dim3 block,
                  size_t shared_mem, const float* d_in, float* d_out,
                  int repeats, bool print_checksum) {
    int size = MATRIX_DIM * MATRIX_DIM;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < repeats; ++i) {
        kernel<<<grid, block, shared_mem>>>(d_in, d_out);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    double avg_ms = ms / repeats;

    if (print_checksum) {
        float* h_out = (float*)malloc(size * sizeof(float));
        cudaMemcpy(h_out, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);
        double sum = 0.0;
        for (int i = 0; i < size; ++i) sum += h_out[i];
        printf("    checksum = %f\n", sum);
        free(h_out);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return avg_ms;
}

// ------------------- 内核定义 -------------------
__global__ void row_read_row(const float* __restrict__ in, float* out) {
    __shared__ float tile[TILE_DIM][TILE_DIM];
    int x = threadIdx.x + blockIdx.x * TILE_DIM;
    int y = threadIdx.y + blockIdx.y * TILE_DIM;
    int idx = y * MATRIX_DIM + x;
    tile[threadIdx.y][threadIdx.x] = in[idx];
    __syncthreads();
    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void col_read_col(const float* __restrict__ in, float* out) {
    __shared__ float tile[TILE_DIM][TILE_DIM];
    int x = threadIdx.x + blockIdx.x * TILE_DIM;
    int y = threadIdx.y + blockIdx.y * TILE_DIM;
    int idx = y * MATRIX_DIM + x;
    tile[threadIdx.x][threadIdx.y] = in[idx];
    __syncthreads();
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void row_read_col(const float* __restrict__ in, float* out) {
    __shared__ float tile[TILE_DIM][TILE_DIM];
    int x = threadIdx.x + blockIdx.x * TILE_DIM;
    int y = threadIdx.y + blockIdx.y * TILE_DIM;
    int idx = y * MATRIX_DIM + x;
    tile[threadIdx.y][threadIdx.x] = in[idx];
    __syncthreads();
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void row_read_col_padded(const float* __restrict__ in, float* out) {
    __shared__ float tile[TILE_DIM][TILE_DIM + IPAD];
    int x = threadIdx.x + blockIdx.x * TILE_DIM;
    int y = threadIdx.y + blockIdx.y * TILE_DIM;
    int idx = y * MATRIX_DIM + x;
    tile[threadIdx.y][threadIdx.x] = in[idx];
    __syncthreads();
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void row_read_col_dyn(const float* __restrict__ in, float* out) {
    extern __shared__ float tile[];
    int x = threadIdx.x + blockIdx.x * TILE_DIM;
    int y = threadIdx.y + blockIdx.y * TILE_DIM;
    int idx = y * MATRIX_DIM + x;
    int row_idx = threadIdx.y * TILE_DIM + threadIdx.x;
    int col_idx = threadIdx.x * TILE_DIM + threadIdx.y;
    tile[row_idx] = in[idx];
    __syncthreads();
    out[idx] = tile[col_idx];
}

__global__ void row_read_col_dyn_padded(const float* __restrict__ in, float* out) {
    extern __shared__ float tile[];
    int x = threadIdx.x + blockIdx.x * TILE_DIM;
    int y = threadIdx.y + blockIdx.y * TILE_DIM;
    int idx = y * MATRIX_DIM + x;
    int row_idx = threadIdx.y * (TILE_DIM + IPAD) + threadIdx.x;
    int col_idx = threadIdx.x * (TILE_DIM + IPAD) + threadIdx.y;
    tile[row_idx] = in[idx];
    __syncthreads();
    out[idx] = tile[col_idx];
}

__global__ void rect_row_read_col(const float* __restrict__ in, float* out) {
    __shared__ float tile[RECT_Y][RECT_X];
    int x = threadIdx.x + blockIdx.x * RECT_X;
    int y = threadIdx.y + blockIdx.y * RECT_Y;
    int idx = y * MATRIX_DIM + x;
    tile[threadIdx.y][threadIdx.x] = in[idx];
    __syncthreads();
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void rect_row_read_col_padded(const float* __restrict__ in, float* out) {
    __shared__ float tile[RECT_Y][RECT_X + IPAD];
    int x = threadIdx.x + blockIdx.x * RECT_X;
    int y = threadIdx.y + blockIdx.y * RECT_Y;
    int idx = y * MATRIX_DIM + x;
    tile[threadIdx.y][threadIdx.x] = in[idx];
    __syncthreads();
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

// ------------------- 主函数 -------------------
int main(int argc, char** argv) {
    // 解析命令行参数
    int kernel_id = -1;  // -1 表示运行所有
    int repeats = DEFAULT_REPEATS;
    if (argc >= 2) kernel_id = atoi(argv[1]);
    if (argc >= 3) repeats = atoi(argv[2]);

    int size = MATRIX_DIM * MATRIX_DIM;
    size_t bytes = size * sizeof(float);

    // 分配并初始化主机内存
    float* h_in = (float*)malloc(bytes);
    for (int i = 0; i < size; ++i) h_in[i] = (float)(rand() % 100);

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // 配置网格和块
    dim3 block_sq(TILE_DIM, TILE_DIM);
    dim3 grid_sq(MATRIX_DIM / TILE_DIM, MATRIX_DIM / TILE_DIM);
    dim3 block_rect(RECT_X, RECT_Y);
    dim3 grid_rect(MATRIX_DIM / RECT_X, MATRIX_DIM / RECT_Y);

    size_t smem_sq = TILE_DIM * TILE_DIM * sizeof(float);
    size_t smem_sq_pad = (TILE_DIM + IPAD) * TILE_DIM * sizeof(float);
    size_t smem_rect = RECT_X * RECT_Y * sizeof(float);
    size_t smem_rect_pad = (RECT_X + IPAD) * RECT_Y * sizeof(float);

    // 预热
    row_read_row<<<grid_sq, block_sq>>>(d_in, d_out);
    cudaDeviceSynchronize();

    // 定义测试条目
    struct {
        const char* name;
        kernel_t kernel;
        dim3 grid;
        dim3 block;
        size_t smem;
        int enabled;  // 是否参与测试
    } tests[] = {
        {"row_read_row (sq)", row_read_row, grid_sq, block_sq, 0, 1},
        {"col_read_col (sq)", col_read_col, grid_sq, block_sq, 0, 1},
        {"row_read_col (sq)", row_read_col, grid_sq, block_sq, 0, 1},
        {"row_read_col_padded (sq)", row_read_col_padded, grid_sq, block_sq, 0, 1},
        {"row_read_col_dyn (sq)", row_read_col_dyn, grid_sq, block_sq, smem_sq, 1},
        {"row_read_col_dyn_padded (sq)", row_read_col_dyn_padded, grid_sq, block_sq, smem_sq_pad, 1},
        {"rect_row_read_col (32x16)", rect_row_read_col, grid_rect, block_rect, 0, 1},
        {"rect_row_read_col_padded (32x16)", rect_row_read_col_padded, grid_rect, block_rect, 0, 1},
    };
    int num_tests = sizeof(tests) / sizeof(tests[0]);

    // 如果指定了内核 ID，只运行该内核
    if (kernel_id >= 0 && kernel_id < num_tests) {
        tests[kernel_id].enabled = 1;
        for (int i = 0; i < num_tests; ++i) {
            if (i != kernel_id) tests[i].enabled = 0;
        }
    } else if (kernel_id == -1) {
        // 运行所有
    } else {
        printf("Invalid kernel ID. Valid: 0..%d\n", num_tests - 1);
        return 1;
    }

    printf("Matrix size: %dx%d, repeats = %d\n", MATRIX_DIM, MATRIX_DIM, repeats);
    for (int i = 0; i < num_tests; ++i) {
        if (!tests[i].enabled) continue;
        double avg = run_kernel(tests[i].kernel, tests[i].grid, tests[i].block,
                                tests[i].smem, d_in, d_out, repeats, true);
        printf("%-32s: avg %.5f ms\n", tests[i].name, avg);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    return 0;
}