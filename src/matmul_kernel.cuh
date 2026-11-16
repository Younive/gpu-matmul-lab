#pragma once

#include <cuda_runtime.h>

// Naive matrix multiplication
extern __global__ void naive_matmul(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
);