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
){
    // warm up the GPU
    for (int i = 0; i < 10; i++)
        kernel_func();

    cudaDeviceSynchronize();

    // create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // benchmark the kernel
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++)
        kernel_func();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float gflops = 2.0f * M * N * K / (milliseconds / 1000.0f) / 1e9f;
    printf("%s - Average execution time: %f ms\n", name.c_str(), milliseconds / repeat);
    printf("%s - GFLOPS: %f\n", name.c_str(), gflops);

    // clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
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
        int TILE = 4; 
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