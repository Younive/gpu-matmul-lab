#include "../utils/cuda_utils.cuh"
#include "matmul_kernel.cuh"

__global__ void naive_matmul(const float* A, const float* B, float* C, int M, int N, int K){
    // Calculate the row index of the C element and A
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Calculate the column index of C and B
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the row index of the C element and A

    // Perform the matrix multiplication if within bounds
    if (row < M && col < N) {
        float sum = 0.0f;

        for (int k=0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col]; // Calculate the dot product of the row of A and the column of B
        C[row * N + col] = sum; // Store the result in C
    }
}