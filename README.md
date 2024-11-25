# Matrix-Vector Multiplication using CUDA
```
Name: Aldrin Lijo J E
Reg-No:212222240007
```
## Aim
To perform matrix-vector multiplication using CUDA for GPU acceleration and verify its correctness by comparing with a CPU implementation.

---

## Algorithm
1. **Matrix and Vector Initialization**:
   - Generate a random matrix `A` of size \( N \times N \) and a random vector `B` of size \( N \).
2. **CUDA Memory Allocation**:
   - Allocate device memory for the matrix, vector, and result vector.
3. **Data Transfer to Device**:
   - Copy the matrix and vector from host memory to device memory.
4. **Kernel Execution**:
   - Launch a CUDA kernel to compute the matrix-vector multiplication \( C = A \times B \).
5. **Data Transfer to Host**:
   - Copy the resulting vector `C` from device memory back to host memory.
6. **CPU Implementation**:
   - Compute the matrix-vector multiplication on the CPU.
7. **Result Verification**:
   - Compare the GPU result with the CPU result to ensure correctness.
8. **Free Memory**:
   - Release allocated device and host memory.

---

## Code

```c
#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

#define N 512 // Size of the matrix and vector

__global__ void matrixVectorMulKernel(float *A, float *B, float *C, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        float sum = 0;
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i];
        }
        C[row] = sum;
    }
}

void cpuMatrixVectorMul(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        float sum = 0;
        for (int j = 0; j < n; j++) {
            sum += A[i * n + j] * B[j];
        }
        C[i] = sum;
    }
}

int main() {
    int sizeMatrix = N * N * sizeof(float);
    int sizeVector = N * sizeof(float);

    float *h_A, *h_B, *h_C, *h_C_CPU;
    float *d_A, *d_B, *d_C;

    // Allocate host memory
    h_A = (float *)malloc(sizeMatrix);
    h_B = (float *)malloc(sizeVector);
    h_C = (float *)malloc(sizeVector);
    h_C_CPU = (float *)malloc(sizeVector);

    // Initialize matrix and vector
    for (int i = 0; i < N * N; i++) h_A[i] = rand() % 100;
    for (int i = 0; i < N; i++) h_B[i] = rand() % 100;

    // Allocate device memory
    cudaMalloc((void **)&d_A, sizeMatrix);
    cudaMalloc((void **)&d_B, sizeVector);
    cudaMalloc((void **)&d_C, sizeVector);

    // Copy data to device
    cudaMemcpy(d_A, h_A, sizeMatrix, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeVector, cudaMemcpyHostToDevice);

    // Kernel launch parameters
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // Launch CUDA kernel
    matrixVectorMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, sizeVector, cudaMemcpyDeviceToHost);

    // Perform CPU matrix-vector multiplication
    cpuMatrixVectorMul(h_A, h_B, h_C_CPU, N);

    // Verify the results
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (abs(h_C[i] - h_C_CPU[i]) > 1e-5) {
            correct = false;
            break;
        }
    }

    printf("Result is %s\n", correct ? "Correct" : "Incorrect");

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_CPU);

    return 0;
}
# Output:
![image](https://github.com/user-attachments/assets/f707b45f-d62f-473d-ac25-2e998b4625e3)
## Results
```
GPU Execution:
Successfully computed matrix-vector multiplication using CUDA.
Efficiently utilized GPU parallelism.
CPU Verification:
CPU computation matched the GPU result, ensuring correctness.
Performance:
CUDA offers significant speedup for large matrices and vectors compared to CPU execution.
```
