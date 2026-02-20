// QUASAR-SUBNET: minimal CUDA stub for repository validation.
// Optional: replace with real kernels for performance (see guide_cu.md).

#include <cuda_runtime.h>

__global__ void quasar_stub_kernel(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}

extern "C" void launch_quasar_stub(const float* x, float* y, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    quasar_stub_kernel<<<blocks, threads, 0, stream>>>(x, y, n);
}
