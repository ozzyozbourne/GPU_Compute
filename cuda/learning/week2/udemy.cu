#include <stdio.h>

__global__ void hello_cuda() { printf("Hello world cuda\n"); }

__global__ void print_thread_id() {
  printf("\nx -> %d, y -> %d, z -> %d\n", threadIdx.x, threadIdx.y,
         threadIdx.z);
}

__global__ void debug_dimensions() {
  printf("block idx values -> \nx : %d y : %d z : %d \n", blockIdx.x,
         blockIdx.y, blockIdx.z);
  printf("grid  idx values -> \nx : %d y : %d z : %d \n", gridDim.x, gridDim.y,
         gridDim.z);
}

__global__ void exercise_01() {
  printf("threadIdx -> %d %d %d\n", threadIdx.x, threadIdx.y, threadIdx.z);
  printf("blockDim  -> %d %d %d\n", blockDim.x, blockDim.y, blockDim.z);
  printf("blockIdx  -> %d %d %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
  printf("gridDim   -> %d %d %d\n", gridDim.x, gridDim.y, gridDim.z);
}

__global__ void uni_idx_cal(const int *const p) {
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  printf("globalIdx is -> [%d] and the value at that index is -> %d\n",
         global_idx, p[global_idx]);
}

__global__ void d2_grid_block_idx() {
  const int x = blockIdx.x * blockDim.x;
  const int y = blockIdx.y * blockDim.y * gridDim.y;
  printf("x -> %d y -> %d\n", x, y);
}

int main(void) {
  const int array_size = 16;
  const int array_byte_size = sizeof(int) * array_size;
  const int h_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

  for (int i = 0; i < array_size; i++) {
    printf("%d ", h_data[i]);
  }
  printf("\n \n");

  int *d_data;
  cudaMalloc(&d_data, array_byte_size);
  cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

  const dim3 block(4);
  const dim3 grid(2, 2);
  d2_grid_block_idx<<<grid, block>>>();

  cudaDeviceSynchronize();
  cudaFree(d_data);
  cudaDeviceReset();
  return 0;
}
