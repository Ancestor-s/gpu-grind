#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

// Error checking macro
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        return 1; \
    } \
}

__global__ void boostScores(float *scores, int N, float bonus) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        scores[i] += bonus;
        if (scores[i] > 100.0f) scores[i] = 100.0f;
    }
}

void boostScoresCPU(float *scores, int N, float bonus) {
    for (int i = 0; i < N; i++) {
        scores[i] += bonus;
        if (scores[i] > 100.0f) scores[i] = 100.0f;
    }
}

int main() {
    int N = 10000000; 
    size_t size = N * sizeof(float);
    float *h_scores = (float*)malloc(size);
    float *h_cpu = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_scores[i] = 50.0f + (i % 50);
        h_cpu[i] = h_scores[i];
    }

    // --- CPU Benchmark ---
    auto cpu_start = std::chrono::high_resolution_clock::now();
    boostScoresCPU(h_cpu, N, 5.0f);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // --- GPU Benchmark ---
    float *d_scores;
    CHECK_CUDA(cudaMalloc(&d_scores, size));
    CHECK_CUDA(cudaMemcpy(d_scores, h_scores, size, cudaMemcpyHostToDevice));

    int threads = 256;
    // Ceil division: ensures we cover all N even if N % 256 != 0
    int blocks = (N + threads - 1) / threads;

    auto gpu_start = std::chrono::high_resolution_clock::now();
    boostScores<<<blocks, threads>>>(d_scores, N, 5.0f);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    CHECK_CUDA(cudaMemcpy(h_scores, d_scores, size, cudaMemcpyDeviceToHost));

    printf("N = %d scores\n", N);
    printf("CPU Time: %.3f ms\n", cpu_ms);
    printf("GPU Time: %.3f ms\n", gpu_ms);
    printf("Speedup: %.2fx\n", cpu_ms / gpu_ms);
    printf("Sample: %.1f, %.1f, %.1f\n", h_scores[0], h_scores[49], h_scores[N-1]);
    printf("Correct: %s\n", (h_scores[0] == h_cpu[0]) ? "YES" : "NO");

    cudaFree(d_scores);
    free(h_scores); free(h_cpu);
    return 0;
}
