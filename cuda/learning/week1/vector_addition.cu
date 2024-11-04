#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Error macro
#define CHECK_CUDA_ERROR(call) { \ 
    const cudaError_t err = call; \
    if (err != cudaSuccess) { \ 
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

//CUDA kernel for vector addition
__global__ void vectorAdd(const float *vec_a, const float *vec_b, float *vec_res, const int vec_size){
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < vec_size) { vec_res[global_id] = vec_a[global_id] + vec_b[global_id]; }
}

// Cuda setup 
int main(void){
    //Init Cuda
    int device_count; 
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&device_count));
    if (device_count == 0){
        printf("No CUDA devices found!\n");
        return -1;
    }
    
    const int vec_size = 1000000; 
    const size_t memory_size = vec_size * sizeof(float);

    //Allocate memory in cpu to be transfered to GPU
    float *host_vec_a   = (float*)malloc(memory_size);
    float *host_vec_b   = (float*)malloc(memory_size);
    float *host_vec_res = (float*)malloc(memory_size);


    if(host_vec_a == NULL || host_vec_b == NULL || host_vec_res == NULL){
        printf("CPU memory allocation failed!\n");
        return -1;
    }

    //Init Vectors 
    for(int i = 0; i < vec_size; i++){
        host_vec_a[i] = 1.0f;
        host_vec_b[i] = 2.0f;
    }

    //Allocating memory in GPU
    float *gpu_vec_a, *gpu_vec_b, *gpu_vec_res;
    CHECK_CUDA_ERROR(cudaMalloc(&gpu_vec_a, memory_size));
    CHECK_CUDA_ERROR(cudaMalloc(&gpu_vec_b, memory_size));
    CHECK_CUDA_ERROR(cudaMalloc(&gpu_vec_res, memory_size));

    //Copy the array from cpu to gpu
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_vec_a, host_vec_a, memory_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_vec_b, host_vec_b, memory_size, cudaMemcpyHostToDevice)));

    //calculate the block size and the number of blocks needed
    const int block_size = 256;
    const int num_of_blocks = (vec_size + block_size - 1) / block_size;
    //Launch the gpu kernel
    vectorAdd<<num_of_blocks, block_size>>(gpu_vec_a, gpu_vec_b, gpu_vec_res);

    // Check for kernel launch errors 
    CHECK_CUDA_ERROR(cudaGetLastError());

    //Wait for the kernel to finish and check for errors
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    //Copy the result back to the host 
    CHECK_CUDA_ERROR(cudaMemcpy(host_vec_res, gpu_vec_res, memory_size, cudaMemcpyDeviceToHost))


    // Verify results
    bool success = true;
    for (int i = 0; i < vec_size; i++) {
        if (host_vec_res[i] != 3.0f) {  // Expected: 1.0 + 2.0 = 3.0
            printf("Verification failed at index %d: Expected 3.0f, got %f\n", 
                   i, host_vec_res[i]);
            success = false;
            break;
        }
    }
    if (success) printf("Vector addition successful!\n");


    //Cleanup
    CHECK_CUDA_ERROR(cudaFree(gpu_vec_a));
    CHECK_CUDA_ERROR(cudaFree(gpu_vec_b));
    CHECK_CUDA_ERROR(cudaFree(gpu_vec_res));

    free(host_vec_a);
    free(host_vec_b);
    free(host_vec_res);

    //Reset the Gpu
    CHECK_CUDA_ERROR(cudaDeviceReset());
    return success ? 0 : -1;

}

