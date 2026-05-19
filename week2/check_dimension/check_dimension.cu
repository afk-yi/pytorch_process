#include <cuda_runtime.h>
#include <stdio.h>
__global__ void checkIndex(void)
{
  printf("threadIdx:(%d,%d,%d) blockIdx:(%d,%d,%d) blockDim:(%d,%d,%d)\
  gridDim(%d,%d,%d)\n",threadIdx.x,threadIdx.y,threadIdx.z,
  blockIdx.x,blockIdx.y,blockIdx.z,blockDim.x,blockDim.y,blockDim.z,
  gridDim.x,gridDim.y,gridDim.z);
}
int main(int argc,char **argv)
{
  int nElem=6;
  dim3 block(3);
  dim3 grid((nElem+block.x-1)/block.x); // 实现向上取整
  printf("grid.x %d grid.y %d grid.z %d\n",grid.x,grid.y,grid.z);
  printf("block.x %d block.y %d block.z %d\n",block.x,block.y,block.z);
  checkIndex<<<grid,block>>>();
  cudaDeviceReset();
  return 0;
}
// grid 的尺寸是 (2, 1, 1)：表示一共有 2 个块，排列在 x 方向上，y/z 方向只有 1。
// block 的尺寸是 (3, 1, 1)：表示每个块内有 3 个线程，同样只使用 x 维度。