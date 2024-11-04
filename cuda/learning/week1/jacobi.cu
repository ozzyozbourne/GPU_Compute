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
#define BLOCK_SIZE 16

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
    float *cpu_arr_a, *cpu_arr_b;
    float *gpu_arr_a, *gpu_arr_b;

}