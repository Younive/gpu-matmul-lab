# gpu-matmul-lab

CUDA matrix multiplication optimization experiments on NVIDIA GPUs.

This project explores GPU performance optimization through progressively more advanced CUDA matrix multiplication kernels. It investigates global memory access, shared-memory tiling, register tiling, warp-level programming, and GPU profiling.

The main goal is to understand how GPU architecture and resource utilization influence kernel performance.

## 1. Project Objectives

- Implement a baseline CUDA matrix multiplication kernel.

- Improve memory reuse using shared-memory tiling.

- Explore register tiling and per-thread output accumulation.

- Investigate warp-level data sharing and communication.

- Profile CUDA kernels using NVIDIA Nsight Compute.

- Compare custom implementations against NVIDIA cuBLAS.

## 2. Hardware and Software Environment

### Hardware

| Component |  Specification |
| :--- | ---: |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU |
| Architecture | NVIDIA Ampere |
| Compute Capability | 8.6 |
| Streaming Multiprocessors | 20 |
| Warp Size | 32 threads |
| CUDA |NVIDIA CUDA platform |

### Software

| Component | Version / Environment |
| :--- | ---: |
| Operating System | Ubuntu on WSL2 |
| CUDA Toolkit | cuda_11.5.r11.5 | 
| Host Compiler | GCC 10 |
| Nsight Compute | 2026.4| 
| NVIDIA Driver | CUDA 13.5 |

To Be Update...