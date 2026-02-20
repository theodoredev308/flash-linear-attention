/**
 * Python binding for Quasar forward substitution CUDA kernel.
 */

#include "quasar_forward_substitution.h"
#include <torch/extension.h>

namespace py = pybind11;

torch::Tensor forward_substitution_cuda(torch::Tensor L) {
    TORCH_CHECK(L.dim() == 3, "L must be 3D (n_batch, BT, BT)");
    const int n_batch = L.size(0);
    const int BT = L.size(1);
    TORCH_CHECK(L.size(2) == BT);

    auto A = torch::empty_like(L, L.options());
    launch_forward_substitution_cuda(L, A, BT);
    return A;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_substitution", &forward_substitution_cuda,
          "Forward substitution: A = L^{-1} (CUDA)",
          py::arg("L"));
}
