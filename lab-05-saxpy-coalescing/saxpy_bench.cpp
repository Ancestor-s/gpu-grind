#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
__global__ void saxpy(float* y, const float* x, float a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}
int main() {
  int N = 1 << 24;
  float A = 2.0f;
  std::vector<float> hx(N, 1.0f), hy(N, 2.0f);
  float *dx, *dy;
  hipMalloc(&dx, N*4);
  hipMalloc(&dy, N*4);
  hipMemcpy(dx, hx.data(), N*4, hipMemcpyHostToDevice);
  hipMemcpy(dy, hy.data(), N*4, hipMemcpyHostToDevice);
  int sizes[] = {64, 128, 256, 512, 1024};
  for (int bs : sizes) {
    int blocks = (N + bs - 1) / bs;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int k=0;k<100;k++) saxpy<<<blocks, bs>>>(dy, dx, A, N);
    hipDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1-t0).count()/100;
    double gbs = (3.0 * N * 4) / (ms * 1e-3) / 1e9;
    std::cout << "block=" << bs << " GB/s=" << gbs << std::endl;
  }
  return 0;
}
