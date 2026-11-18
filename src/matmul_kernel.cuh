#pragma once

#include <cuda_runtime.h>

// Naive matrix multiplication
extern __global__ void naive_matmul(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
);

// Tiled matrix multiplication using shared memory
extern __global__ void tiled_matmul(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    int TILE
);

// Register tiled matrix multiplication using shared memory
extern __global__ void regtile_matmul(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    int TILE,
    int ROWS_PER_THREAD,
    int COLS_PER_THREAD
);

// Host function declarations for launching the kernels
void tiled_matmul_launcher(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    int TILE
);

// Host function to launch the register tiled matrix multiplication kernel
void regtile_matmul_launcher(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    int TILE,
    int ROWS_PER_THREAD,
    int COLS_PER_THREAD
);