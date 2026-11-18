#include "../utils/cuda_utils.cuh"
#include "matmul_kernel.cuh"
#include <stdexcept>

template<int TILE, int ROWS_PER_THREAD, int COLS_PER_THREAD>

__global__ void regtile_matmul(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    // Ensure that TILE is divisible by ROW_PER_THREAD and COL_PER_THREAD
    static_assert(TILE % ROWS_PER_THREAD == 0);
    static_assert(TILE % COLS_PER_THREAD == 0);

    // Calculate the number of threads in each dimension
    constexpr int THREADS_X = TILE / COLS_PER_THREAD;
    constexpr int THREADS_Y = TILE / ROWS_PER_THREAD;

    static_assert(THREADS_X * THREADS_Y <= 1024);

    // Allocate shared memory for tiles of A and B
    __shared__ float tile_A[TILE][TILE];
    __shared__ float tile_B[TILE][TILE];

    // Expected block dimensions: (4, 4)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // First output row and column owned by this thread
    int row0 = blockIdx.y * TILE + ty * ROWS_PER_THREAD;
    int col0 = blockIdx.x * TILE + tx * COLS_PER_THREAD;

    // Register accumulators: 8x8 outputs per thread
    float acc[ROWS_PER_THREAD][COLS_PER_THREAD] = {};

    int num_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; t++) {
        int k0 = t * TILE;

        // Cooperatively load 8x8 portions of A and B
        #pragma unroll
        for (int i = 0; i < ROWS_PER_THREAD; i++) {
            #pragma unroll
            for (int j = 0; j < COLS_PER_THREAD; j++) {

                // Load A: [row0 + i, k0 + tx*8 + j]
                int a_row = row0 + i;
                int a_col = k0 + tx * COLS_PER_THREAD + j;

                tile_A[ty * ROWS_PER_THREAD + i]
                      [tx * COLS_PER_THREAD + j] =
                    (a_row < M && a_col < K)
                    ? A[a_row * K + a_col]
                    : 0.0f;

                // Load B: [k0 + ty*8 + i, col0 + j]
                int b_row = k0 + ty * ROWS_PER_THREAD + i;
                int b_col = col0 + j;

                tile_B[ty * ROWS_PER_THREAD + i]
                      [tx * COLS_PER_THREAD + j] =
                    (b_row < K && b_col < N)
                    ? B[b_row * N + b_col]
                    : 0.0f;
            }
        }

        __syncthreads();

        // Compute the 8x8 output tile owned by this thread
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            #pragma unroll
            for (int i = 0; i < ROWS_PER_THREAD; i++) {
                float a = tile_A[ty * ROWS_PER_THREAD + i][k];

                #pragma unroll
                for (int j = 0; j < COLS_PER_THREAD; j++) {
                    float b = tile_B[k][tx * COLS_PER_THREAD + j];
                    acc[i][j] += a * b;
                }
            }
        }

        __syncthreads();
    }

    // Write results AFTER processing every K tile
    #pragma unroll
    for (int i = 0; i < ROWS_PER_THREAD; i++) {
        #pragma unroll
        for (int j = 0; j < COLS_PER_THREAD; j++) {
            int row = row0 + i;
            int col = col0 + j;

            if (row < M && col < N) {
                C[row * N + col] = acc[i][j];
            }
        }
    }
}

template<int TILE, int ROWS_PER_THREAD, int COLS_PER_THREAD>
void launch_regtile(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    dim3 block(
        TILE / COLS_PER_THREAD,
        TILE / ROWS_PER_THREAD
    );

    dim3 grid(
        (N + TILE - 1) / TILE,
        (M + TILE - 1) / TILE
    );

    regtile_matmul<TILE, ROWS_PER_THREAD, COLS_PER_THREAD>
        <<<grid, block>>>(A, B, C, M, N, K);
}

void regtile_matmul_launcher(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    int TILE,
    int ROWS_PER_THREAD,
    int COLS_PER_THREAD
) {
    if (TILE == 32 &&
        ROWS_PER_THREAD == 4 &&
        COLS_PER_THREAD == 4) {

        launch_regtile<32, 4, 4>(A, B, C, M, N, K);

    } else if (TILE == 32 &&
               ROWS_PER_THREAD == 2 &&
               COLS_PER_THREAD == 2) {

        launch_regtile<32, 2, 2>(A, B, C, M, N, K);

    } else {
        throw std::invalid_argument("Unsupported register tile configuration");
    }
}