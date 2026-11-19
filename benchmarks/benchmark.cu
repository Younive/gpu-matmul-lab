#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
#include <string>
#include <functional>
#include "../src/matmul_kernel.cuh"
#include "../utils/cuda_utils.cuh"

void benchmark(
    const std::string& name,
    const std::function<void()>& kernel_func,
    int M, int N, int K,
    int repeat = 1000
) {
    // Warmup
    for (int i = 0; i < 10; i++) {
        kernel_func();
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Timing
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repeat; i++) {
        kernel_func();
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(&milliseconds, start, stop)
    );

    float avg_ms = milliseconds / repeat;

    double gflops =
        (2.0 * M * N * K) /
        (avg_ms / 1000.0) / 1e9;

    printf("%s - Average execution time: %.6f ms\n",
           name.c_str(), avg_ms);

    printf("%s - GFLOP/s: %.3f\n",
           name.c_str(), gflops);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main(int argc, char** argv) {

    if (argc != 4) {
        fprintf(stderr, "Usage: %s <M> <K> <N>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = atoi(argv[3]);

    // allocate and initialize matrices
    // allocate cpu memory
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_C = (float*)malloc(bytes_C);

    // initialize matrices
    for (int i = 0; i < M * K; i++)
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; i++)
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // allocate gpu memory
    float *d_A, *d_B, *d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    // copy input from cpu to gpu
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    // benchmark naive matrix multiplication
    benchmark("Naive MatMul", [&](){
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        naive_matmul<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }, M, N, K);

    // benchmark tiled matrix multiplication
    benchmark("Tiled MatMul", [&](){
        int TILE = 32; 
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
        tiled_matmul_launcher(d_A, d_B, d_C, M, N, K, TILE);
    }, M, N, K);

    // benchmark register tiled matrix multiplication
    benchmark("Register Tiled MatMul", [&](){
        int TILE = 32;
        int ROWS_PER_THREAD = 2;
        int COLS_PER_THREAD = 2;
        regtile_matmul_launcher(d_A, d_B, d_C, M, N, K, TILE, ROWS_PER_THREAD, COLS_PER_THREAD);
    }, M, N, K);

    // benchmark warp-optimized matrix multiplication
    benchmark("Warp-opimized MatMul", [&](){
        int WARP_SIZE = 32;
        int ROWS_PER_WARP = 16;
        int WARPS_PER_BLOCK = 16;
        warp_optimized_matmul_launcher(d_A, d_B, d_C, M, N, K, WARP_SIZE, ROWS_PER_WARP, WARPS_PER_BLOCK);
    }, M, N, K);

    // free gpu memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // free cpu memory
    free(h_A);
    free(h_B);
    free(h_C);

    return EXIT_SUCCESS;
}