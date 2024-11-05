#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

//Error macro for cuda calls
#define CHECK_CUDA_ERROR(call) {\ 
    const cudaError_t err = call;\
    if (err != cudaSuccess) {\ 
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__);\
        exit(1);\
    }\
}
#define N 12
//Since we are using flatten 2d arrays
#define IDX2D(i, j) (i * (N+2) + j)  
#define TOLERANCE 0.003f

//Globals
__device__ bool isDone;
__device__ float *gpu_arr_a;
__device__ float *gpu_arr_b;

void initMatrix(float *a){
    for(int i = 0; i <= N+1; i++){
        for(int j = 0; j <= N+1; j++){
            a[IDX2D(i, j)] = (float)(rand() % 200) / 200.0f;
        }
    }
}

void printMatrix(const float *A) {
    for (int i = 0; i <= N+1; i++) {
        printf("Row: %d ->", i);
        for (int j = 0; j <= N+1; j++) { printf("%.3f  ", A[IDX2D(i, j)]); }
        printf("\n");
    }
}

int main(void){
    float *cpu_arr_a, *gpu_arr_temp_a, *gpu_arr_temp_b;
    bool value = false;
    
    const size_t arr_size = (N*2) * (N+2) *sizeof(float);

    //Allocate in cpu
    cpu_arr_a = (float*)malloc(arr_size);
    initMatrix(cpu_arr_a);
    
    //Allocate in gpu
    CHECK_CUDA_ERROR(cudaMalloc(&gpu_arr_temp_a, arr_size));
    CHECK_CUDA_ERROR(cudaMalloc(&gpu_arr_temp_b, arr_size));

    //Copy pointers to global variables 
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(isDone, &value, sizeof(bool)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(gpu_arr_a, &gpu_arr_temp_a, sizeof(float *)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(gpu_arr_b, &gpu_arr_temp_b, sizeof(float *)));


    //Copy to gpu
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_arr_temp_a, cpu_arr_a, arr_size, cudaMemcpyHostToDevice));

    //Copy from gpu to cpu
    CHECK_CUDA_ERROR(cudaMemcpy(cpu_arr_a, gpu_arr_temp_a, arr_size, cudaMemcpyDeviceToHost));

    printMatrix(cpu_arr_a);

    //CleanUp
    free(cpu_arr_a);
    CHECK_CUDA_ERROR(cudaFree(gpu_arr_temp_a));
    CHECK_CUDA_ERROR(cudaFree(gpu_arr_temp_b));

    return 0;
}
