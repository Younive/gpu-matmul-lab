#include <cuda_runtime.h>
#include <cstdio>

int main() {
    int device = 0;
    cudaDeviceProp prop{};

    cudaError_t err = cudaGetDeviceProperties(&prop, device);

    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n",
           prop.major, prop.minor);

    printf("\n--- Compute Resources ---\n");
    printf("SM Count: %d\n", prop.multiProcessorCount);
    printf("Max Threads per Block: %d\n",
           prop.maxThreadsPerBlock);
    printf("Max Threads per SM: %d\n",
           prop.maxThreadsPerMultiProcessor);
    printf("Registers per SM: %d\n",
           prop.regsPerMultiprocessor);

    printf("\n--- Memory Resources ---\n");
    printf("Global Memory: %.2f GB\n",
           prop.totalGlobalMem / 1e9);
    printf("Shared Memory per Block: %zu KB\n",
           prop.sharedMemPerBlock / 1024);
    printf("Shared Memory per SM: %zu KB\n",
           prop.sharedMemPerMultiprocessor / 1024);
    printf("L2 Cache: %d KB\n",
           prop.l2CacheSize / 1024);
    printf("Memory Bus Width: %d bits\n",
           prop.memoryBusWidth);

    printf("\n--- Clock Information ---\n");
    printf("GPU Clock: %.2f MHz\n",
           prop.clockRate / 1000.0);
    printf("Memory Clock: %.2f MHz\n",
           prop.memoryClockRate / 1000.0);

    return 0;
}