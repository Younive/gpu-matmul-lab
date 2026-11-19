#include "../utils/cuda_utils.cuh"
#include "matmul_kernel.cuh"

template<
    int WARP_SIZE,
    int ROWS_PER_WARP,
    int WARPS_PER_BLOCK
>
__global__ void warp_optimized_matmul(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    static_assert(WARP_SIZE == 32);

    constexpr int ROWS_PER_BLOCK =
        ROWS_PER_WARP * WARPS_PER_BLOCK;

    int lane = threadIdx.x;
    int warp_id = threadIdx.y;

    int row0 = blockIdx.y * ROWS_PER_BLOCK
             + warp_id * ROWS_PER_WARP;

    int col = blockIdx.x * WARP_SIZE + lane;

    // Each thread accumulates ROWS_PER_WARP output elements.
    float acc[ROWS_PER_WARP] = {};

    for (int k = 0; k < K; k++) {

        // One B element per lane, reused across output rows.
        float b = 0.0f;

        if (col < N)
            b = B[(size_t)k * N + col];

        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; r++) {

            int row = row0 + r;

            float a = 0.0f;

            if (row < M)
                a = A[(size_t)row * K + k];

            acc[r] = fmaf(a, b, acc[r]);
        }
    }

    // Write every output row assigned to this warp.
    if (col < N) {
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; r++) {

            int row = row0 + r;

            if (row < M)
                C[(size_t)row * N + col] = acc[r];
        }
    }
}
template<
    int WARP_SIZE, 
    int ROWS_PER_WARP,
    int WARPS_PER_BLOCK >

void launch_warp_optimized(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
){
    int ROWS_PER_BLOCK = ROWS_PER_WARP * WARPS_PER_BLOCK;
    dim3 block(WARP_SIZE, WARPS_PER_BLOCK);
    dim3 grid(
        (N + WARP_SIZE - 1) / WARP_SIZE,
        (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK
    );
    warp_optimized_matmul<WARP_SIZE, ROWS_PER_WARP, WARPS_PER_BLOCK>
        <<<grid, block>>>(A, B, C, M, N, K);
}

void warp_optimized_matmul_launcher(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    int WARP_SIZE, 
    int ROWS_PER_WARP,
    int WARPS_PER_BLOCK
){
    if (WARP_SIZE == 32 && ROWS_PER_WARP == 16 && WARPS_PER_BLOCK == 16)
        launch_warp_optimized<32, 16, 16>(A, B, C, M, N, K);
    if (WARP_SIZE == 32 && ROWS_PER_WARP == 8 && WARPS_PER_BLOCK == 8)
        launch_warp_optimized<32, 8, 8>(A, B, C, M, N, K);
    if (WARP_SIZE == 32 && ROWS_PER_WARP == 4 && WARPS_PER_BLOCK == 4)
        launch_warp_optimized<32, 4, 4>(A, B, C, M, N, K);
    if (WARP_SIZE == 32 && ROWS_PER_WARP == 2 && WARPS_PER_BLOCK == 2)
        launch_warp_optimized<32, 2, 2>(A, B, C, M, N, K);
}