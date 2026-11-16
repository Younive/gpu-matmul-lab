#include <cuda_runtime.h>
#include <cstdio>
#include "../utils/cuda_utils.cuh"
#include "../src/matmul_kernel.cuh"

// allocate memory and initialize matrices
void allocate_and_initialize(int M, int K, int N) {
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

    // set up execution configuration
    dim3 block(16, 16);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );
}

int main(int argc, char** argv) {

    constexpr int REPEAT = 100;

    if (argc != 4) {
        fprintf(stderr, "Usage: %s <M> <K> <N>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = atoi(argv[3]);

    // allocate and initialize matrices
    allocate_and_initialize(M, K, N);

    // create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --------------------------------
    // Benchmark 1: Naive MatMul
    // --------------------------------

    // warm up the GPU
    naive_matmul<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // benchmark naive_matmul kernel
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < REPEAT; i++)
        naive_matmul<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    
    // record the stop event
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    float gflops = 2.0f * M * N * K / (milliseconds / 1000.0f) / 1e9f;
    printf("Naive MatMul - Average execution time: %f ms\n", milliseconds / REPEAT);
    printf("Naive MatMul - GFLOPS: %f\n", gflops);

    // --------------------------------
    // Benchmark 2: shared memory MatMul
    // --------------------------------

    // clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // free gpu memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return EXIT_SUCCESS;
}