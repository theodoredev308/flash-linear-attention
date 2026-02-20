# Quasar CUDA extension (.cu)

See **`guide_cu.md`** at the repo root for why .cu files are used and how to pass validation or aim for top miner.

This folder holds a **real** CUDA kernel (forward substitution) used in the chunk path. There is also a **minimal stub** at `fla/ops/quasar/quasar_stub.cu` for validation-only; it does not need to be built.

## Files

- **quasar_forward_substitution.cu** – CUDA kernel: A = L^{-1} for lower triangular L (1s on diagonal). One thread block per batch element.
- **quasar_forward_substitution.cpp** – Python binding (pybind11).
- **quasar_forward_substitution.h** – Declares the launch function.

## Build

CUDA toolkit and PyTorch with CUDA required. From repo root:

**Option 1 – pip install (builds extension when CUDA is available)**

```bash
pip install -e .
```

**Option 2 – JIT load (no install)**

```python
from torch.utils.cpp_extension import load
ext = load(
    name='quasar_forward_substitution_cuda',
    sources=[
        'fla/ops/quasar/csrc/quasar_forward_substitution.cpp',
        'fla/ops/quasar/csrc/quasar_forward_substitution.cu',
    ],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
)
# ext.forward_substitution(L)  # L shape (n_batch, BT, BT)
```

## Usage

`chunk.py` uses **Option A** from guide_cu.md: it tries to import `quasar_forward_substitution_cuda` and, when available and on CUDA, calls `ext.forward_substitution(L_flat)` for the triangular solve; otherwise it falls back to the Triton kernel. No code change needed once the extension is built.

L is 3D `(n_batch, BT, BT)`, row-major, lower triangular with 1s on the diagonal. Returns A of the same shape with A = L^{-1}.
