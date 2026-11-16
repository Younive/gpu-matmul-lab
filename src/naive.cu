#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                       \
    cudaError_t err = (call);                       \
    if (err != cudaSuccess) {                       \
        fprintf(stderr,                             \
                "CUDA error at %s:%d: %s\n",        \
                __FILE__, __LINE__,                 \
                cudaGetErrorString(err));           \
        exit(EXIT_FAILURE);                         \
    }                                               \
} while (0)

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

int main(int argc, char** argv) {

    // Set matrix dimensions
    int M = 1024; // Number of rows in A and C
    int K = 1024; // Number of columns in A and rows in B
    int N = 1024; // Number of columns in B and C

    // receive arguments from command line
    if (argc == 4) {
        M = atoi(argv[1]);
        K = atoi(argv[2]);
        N = atoi(argv[3]);
    }

    // check matrix dimensions
    if (M <= 0 || K <= 0 || N <= 0) {
        fprintf(stderr, "Matrix dimensions must be positive integers.\n");
        return EXIT_FAILURE;
    }

    // show current matrix dimensions
    printf("Matrix dimensions: M=%d, K=%d, N=%d\n", M, K, N);

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

    naive_matmul <<< grid, block >>> (d_A, d_B, d_C, M, N, K);

    // Check launch errors
    CUDA_CHECK(cudaGetLastError());

    // Check asynchronous execution errors
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // copy output from gpu to cpu
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    // free gpu memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return EXIT_SUCCESS;
}