#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
__global__ void saxpy_s(float* y, const float* x, float a, int n, int s) {
  int i = (blockIdx.x * blockDim.x + threadIdx.x) * s;
  if (i < n) y[i] = a * x[i] + y[i];
}
int main() {
  int N = 1 << 24; float A = 2.0f;
  std::vector<float> hx(N, 1.0f), hy(N, 2.0f);
  float *dx, *dy; hipMalloc(&dx, N*4); hipMalloc(&dy, N*4);
  hipMemcpy(dx, hx.data(), N*4, hipMemcpyHostToDevice);
  hipMemcpy(dy, hy.data(), N*4, hipMemcpyHostToDevice);
  for (int stride : {1, 4, 16}) {
    int blocks = (N/stride + 255)/256;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int k=0;k<100;k++) saxpy_s<<<blocks, 256>>>(dy, dx, A, N, stride);
    hipDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1-t0).count()/100;
    double gbs = (3.0 * (N/stride) * 4) / (ms * 1e-3) / 1e9;
    std::cout << "stride=" << stride << " GB/s=" << gbs << std::endl;
  }
}
