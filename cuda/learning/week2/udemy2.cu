#include <stdio.h>

__global__ void use_of_shared_memory(const int *const p, int *const out) {
  __shared__ int sharred_array[20];

  const int idx = threadIdx.y * blockDim.x + threadIdx.x;
  // copy from device to sharred
  sharred_array[idx] = p[idx];

  __syncthreads();

  // copy from shared to device
  out[idx] = sharred_array[idx];
}

int main(void) {
  // 2d array
  const int row = 4;
  const int col = 5;
  const int size_of_2d_array = row * col * sizeof(int);

  int h_data_2d[row][col];
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      h_data_2d[i][j] = i * col + j;
    }
  }

  int *const h_output = (int *)malloc(size_of_2d_array);

  // allocate the device in and out
  int *d_data_2d, *d_data_2d_out;
  cudaMalloc(&d_data_2d_out, size_of_2d_array);
  cudaMalloc(&d_data_2d, size_of_2d_array);

  cudaMemcpy(d_data_2d, h_data_2d, size_of_2d_array, cudaMemcpyHostToDevice);

  const dim3 block(5, 4);
  const dim3 grid(1);

  use_of_shared_memory<<<grid, block>>>(d_data_2d, d_data_2d_out);

  cudaDeviceSynchronize();

  cudaMemcpy(h_output, d_data_2d_out, size_of_2d_array, cudaMemcpyDeviceToHost);

  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      printf("arr[%d][%d] = %d\n", i, j, h_output[col * i + j]);
    }
  }

  cudaFree(d_data_2d);
  cudaFree(d_data_2d_out);
  free(h_output);

  cudaDeviceReset();
  return 0;
}
