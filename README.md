# gpu-grind
Grinding GPU programming for AMD Associate  CUDA -> HIP on AMD ROCm. Learning in public
# Module 4 - HIP Score Booster (Benchmarked)

**Original work for my AMD Associate journey** - I designed this lab myself to practice CUDA -> HIP porting, using the concepts from Module 4, not the lesson's example code.

Part of [gpu-grind](https://github.com/ancestor-s/gpu-grind)

## Problem Statement
Add a +5 bonus to 10,000,000 exam scores on the GPU, capped at 100, then benchmark it against the CPU.

This is a classic data-parallel workload: the same simple operation applied independently to millions of elements - exactly what GPUs are built for.

## What I Learned in Module 4
- **GPU Memory Management:** Allocating device memory with `hipMalloc`, moving data with `hipMemcpy`, and freeing it with `hipFree`
- **Kernel Launch Configuration:** Calculating grid size with ceil division `blocks = (N + threads - 1) / threads` to handle non-divisible workloads
- **Boundary Safety:** Using `if (i < N)` to prevent out-of-bounds writes from extra threads in the last block
- **CUDA -> HIP Porting:** Converting CUDA code to HIP with `hipify-perl`, and manually replacing `cudaMalloc`, `cudaMemcpy`, kernel launch syntax, and headers
- **Performance Measurement:** Using `std::chrono` + `hipDeviceSynchronize()` for accurate GPU timing, and verifying GPU results against a CPU reference implementation
- **Production Habits:** Adding error-checking macros (`CHECK_HIP`) to catch silent GPU failures instead of assuming success

## Results (Example)
N = 10 000 000
CPU Time: 42.5 ms
GPU Time: 1.8 ms
Speedup: 23.6
Correct: Yes
*Results vary by hardware. Run it yourself.*

## Key Implementation Details
- `blocks = (N + threads - 1) / threads` - Ceil division ensures full coverage
- `if (i < N)` - Prevents out-of-bounds writes
- `CHECK_HIP` macro - Catches silent GPU failures
- `hipDeviceSynchronize()` - Required for accurate GPU timing
- CPU verification - GPU output is compared to CPU output to prove correctness

## Conversion: CUDA -> HIP

hipify-perl score_booster.cu > score_booster_hip.cpp
| CUDA | HIP |
| --- | --- |
| cudaMalloc | hipMalloc |
| cudaMemcpy | hipMemcpy |
| cudaFree | hipFree |
| cudaDeviceSynchronize | hipDeviceSynchronize |
| kernel<<<blocks, threads>>> | hipLaunchKernelGGL(...) |
| #include <cuda_runtime.h> | #include <hip/hip_runtime.h> |
## Build & Run
```bash
hipcc score booster hip.cpp -o booster
./booster
```
## Files
- score booster.cu - Original CUDA version + CPU benchmark
- score booster hip.cpp - Ported HIP version + CPU benchmark




