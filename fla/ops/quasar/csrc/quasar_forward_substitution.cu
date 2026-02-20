/**
 * CUDA kernel: forward substitution for QuasarAttention.
 * Computes A = L^{-1} where L is lower triangular with 1s on the diagonal.
 * L, A are row-major, shape (n_batch, BT, BT).
 *
 * A[i,i] = 1, A[i,j] = -sum(L[i,k]*A[k,j]) for k in [j, i), j < i.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {

template<typename scalar_t>
__global__ void forward_substitution_kernel(
    const scalar_t* __restrict__ L,
    scalar_t* __restrict__ A,
    const int n_batch,
    const int BT,
    const int stride
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_batch) return;

    const scalar_t* L_b = L + idx * stride;
    scalar_t* A_b = A + idx * stride;

    // Initialize A to identity
    for (int i = 0; i < BT; ++i) {
        for (int j = 0; j < BT; ++j) {
            A_b[i * BT + j] = (i == j) ? scalar_t(1) : scalar_t(0);
        }
    }

    // Forward substitution: for i from 1 to BT-1, for j from 0 to i-1
    for (int i = 1; i < BT; ++i) {
        for (int j = 0; j < i; ++j) {
            scalar_t sum_val = 0;
            for (int k = j; k < i; ++k) {
                sum_val += L_b[i * BT + k] * A_b[k * BT + j];
            }
            A_b[i * BT + j] = -sum_val;
        }
    }
}

} // namespace

void launch_forward_substitution_cuda(
    torch::Tensor L,
    torch::Tensor A,
    int BT
) {
    TORCH_CHECK(L.is_cuda() && A.is_cuda());
    TORCH_CHECK(L.dim() == 3 && A.dim() == 3);
    const int n_batch = L.size(0);
    const int stride = L.size(1) * L.size(2);
    TORCH_CHECK(L.size(1) == BT && L.size(2) == BT);
    TORCH_CHECK(A.size(0) == n_batch && A.size(1) == BT && A.size(2) == BT);

    const int block = 256;
    const int grid = (n_batch + block - 1) / block;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(L.scalar_type(), "forward_substitution_cuda", ([&] {
        forward_substitution_kernel<scalar_t><<<grid, block, 0, stream>>>(
            L.data_ptr<scalar_t>(),
            A.data_ptr<scalar_t>(),
            n_batch,
            BT,
            stride
        );
    }));
}
