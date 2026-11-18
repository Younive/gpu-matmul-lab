#include "../utils/cuda_utils.cuh"
#include "matmul_kernel.cuh"

template<int TILE>

__global__ void tiled_matmul(const float* A, const float* B, float* C, int M, int N, int K){
    // allocate shared memory for tiles of A and B
    __shared__ float tile_A[TILE][TILE];
    __shared__ float tile_B[TILE][TILE];

    // Calculate the row index of the C element and A
    int row = blockIdx.y * TILE + threadIdx.y; // Calculate the column index
    int col = blockIdx.x * TILE + threadIdx.x; // Calculate the row index of the C element and A

    float sum = 0.0f;

    int num_tiles = (K + TILE - 1) / TILE; // Calculate the number of tiles needed
    // Loop over all tiles
    for (int t = 0; t < num_tiles; t++) {
        // load the tile of A into shared memory
        if (row < M && t * TILE + threadIdx.x < K)
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        // load the tile B into shared memory
        if (t * TILE + threadIdx.y < K && col < N)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads(); // synchronize to make sure the tiles are loaded

        // perform the multiplication for the tile
        for (int k = 0; k < TILE; k++)
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];

        __syncthreads(); // synchronize to make sure the computation is done before loading new tiles
    }

    // write the result to C if within bounds
    if (row < M && col < N)
        C[row * N + col] = sum;
}

// host function to launch the tiled_matmul kernel
void tiled_matmul_launcher(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    int TILE
) {
    if (TILE == 16) {
        dim3 block(16, 16);
        dim3 grid(
            (N + 15) / 16,
            (M + 15) / 16
        );

        tiled_matmul<16>
            <<<grid, block>>>(A, B, C, M, N, K);
    }
    else if (TILE == 32) {
        dim3 block(32, 32);
        dim3 grid(
            (N + 31) / 32,
            (M + 31) / 32
        );

        tiled_matmul<32>
            <<<grid, block>>>(A, B, C, M, N, K);
    }
}