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

//Globals only on the GPU ``
__device__ bool isDone;
__device__ bool arrivalLock;
__device__ bool departureLock;

__device__ int count;

__device__ float *gpu_arr_a;
__device__ float *gpu_arr_b;

__global__ void jacobi_relaxation(){
    // this is foralls i 
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx > 0 && idx < N + 1){
        float change = 0.0f, max_change = 0.0f;
        bool done = false;
        do {
            max_change = 0.0f;
            for(int j = 1; j <= N; j++){
                gpu_arr_b[IDX2D(idx, j)] =  (
                      gpu_arr_a[IDX2D(idx-1, j)] 
                    + gpu_arr_a[IDX2D(idx+1, j)] 
                    + gpu_arr_a[IDX2D(idx, j-1)] 
                    + gpu_arr_a[IDX2D(idx, j+1)]
                    ) / 4.0f ;
                change = fabsf(gpu_arr_b[IDX2D(idx, j)] - gpu_arr_a[IDX2D(idx, j)]);
                if (change > max_change){ max_change = change; }
            }
            __syncthreads();
            for(int j = 1; j <= N; j++){ gpu_arr_a[IDX2D(idx, j)] = gpu_arr_b[IDX2D(idx, j)]; }
            done = aggregate(max_change < change);
        }while(!done);
    }
}

__device__ bool aggregate(bool mydone, int n){
    bool result;

}

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
    bool value = false, value2 = true;
    int zero = 0;
    
    const size_t arr_size = (N*2) * (N+2) *sizeof(float);

    //Allocate in cpu
    cpu_arr_a = (float*)malloc(arr_size);
    initMatrix(cpu_arr_a);
    
    //Allocate in gpu
    CHECK_CUDA_ERROR(cudaMalloc(&gpu_arr_temp_a, arr_size));
    CHECK_CUDA_ERROR(cudaMalloc(&gpu_arr_temp_b, arr_size));

    //Copy pointers to global variables 
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(count, &zero, sizeof(int)));

    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(isDone, &value, sizeof(bool)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(arrivalLock, &value2, sizeof(bool)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(departureLock, &value, sizeof(bool)));

    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(gpu_arr_a, &gpu_arr_temp_a, sizeof(float *)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(gpu_arr_b, &gpu_arr_temp_b, sizeof(float *)));


    //Copy to gpu
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_arr_temp_a, cpu_arr_a, arr_size, cudaMemcpyHostToDevice));

    //Copy from gpu to cpu
    CHECK_CUDA_ERROR(cudaMemcpy(cpu_arr_a, gpu_arr_temp_a, arr_size, cudaMemcpyDeviceToHost));

    printMatrix(cpu_arr_a);

    // Check for kernel launch errors 
    CHECK_CUDA_ERROR(cudaGetLastError());

    //Wait for the kernel to finish and check for errors
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    //CleanUp
    free(cpu_arr_a);
    CHECK_CUDA_ERROR(cudaFree(gpu_arr_temp_a));
    CHECK_CUDA_ERROR(cudaFree(gpu_arr_temp_b));

    //Reset the Gpu
    CHECK_CUDA_ERROR(cudaDeviceReset());
    return success ? 0 : -1;
}
