# CUDA (.cu) file guide for QUASAR-SUBNET miners

This guide explains how to add `.cu` (CUDA) files to your flash-linear-attention fork so you pass repository validation and, optionally, how to use them to improve performance as a top miner.

---

## 1. Why .cu files?

- The subnet (or validators) may **require or prefer** at least one `.cu` file in your repo.
- Validation looks for **any** `*.cu` file under the repo root (`rglob("*.cu")`). One file is enough to clear the “No CUDA files (.cu) found” warning.
- For a **top miner**, a real CUDA kernel can sometimes beat Triton on the same ops (e.g. fused chunk recurrence, custom memory layout). Triton is already strong; `.cu` is an extra option for experts.

---

## 2. Where to put the .cu file(s)

Place `.cu` files **anywhere inside your fork**. Examples:

- `fla/ops/quasar/kernels/chunk_kernel.cu`
- `fla/ops/quasar/quasar_chunk.cu`
- `cuda/chunk_fwd.cu`
- Repo root: `chunk.cu` (works but less organized)

Recommended: **`fla/ops/quasar/`** (e.g. `fla/ops/quasar/chunk_kernel.cu`) so it sits next to `chunk.py` and the rest of the quasar ops.

---

## 3. Minimal .cu file (pass validation only)

To **only satisfy** the “has a .cu file” check, add a single minimal CUDA source file. It does not need to be called from Python.

**Path:** e.g. `fla/ops/quasar/quasar_stub.cu`

```cuda
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
```

- Commit this file and push. Validation will see a `.cu` file and the warning will go away.
- You do **not** have to build or call this from `chunk.py` for validation.

---

## 4. What a .cu file should contain to be a top miner

Ranking is by **tokens/sec** (and league). The benchmark runs the **chunk** path of QuasarAttention (`mode="chunk"`). So a competitive `.cu` should implement (or accelerate) the **same math** as the chunk recurrence used in `chunk.py` / `fused_recurrent.py`, in a more efficient way.

### 4.1 Operations that matter

- **Chunk-wise recurrence**: within each chunk, a recurrence over the sequence (e.g. linear attention / scan).
- **Fused ops**: combine multiple passes (e.g. Q/K/V handling, recurrence, gate, output) into fewer kernels to reduce global memory traffic.
- **Forward substitution** (or equivalent): used in the chunk formulation; often a small triangular solve or scan per chunk.
- **Gate**: gating of hidden state (from `gate.py`); can be fused with the recurrence in one kernel.

So a **top-miner .cu** typically contains:

1. **Fused chunk recurrence kernel(s)**  
   - Input: chunk of sequence, previous state.  
   - Output: new state and/or logits.  
   - Optimize for: coalesced reads/writes, shared memory, occupancy, and bfloat16/fp16 if the validator uses it.

2. **Optional: fused forward + gate**  
   - One kernel that does recurrence + gate to cut memory bandwidth.

3. **Good memory layout**  
   - Layout that matches how `chunk.py` (or the validator) uses the tensors (e.g. `[batch, seq, heads, head_dim]`) to avoid extra copies or reshapes.

4. **Correct numerics**  
   - Same recurrence formula and dtypes as the reference Triton path so logit verification still passes (cosine ≥ 0.99, max_abs_diff ≤ 0.1).

### 4.2 High-level content checklist

| Content | Purpose |
|--------|--------|
| Chunk recurrence / linear-attention kernel | Core of the chunk path; biggest speed gain |
| Fused gate + recurrence (optional) | Fewer kernels, less memory traffic |
| Shared memory for tile/block-level state | Reduce global memory and improve occupancy |
| bfloat16/fp16 support | Match validator autocast and throughput |
| Coalesced global memory access | Maximize memory bandwidth |

### 4.3 Integration with Python

To **use** your `.cu` from the repo (instead of only having it for validation):

- **Option A – PyTorch C++ extension**  
  - Build with `torch.utils.cpp_extension.load_inline` or `load()` and `CUDAExtension` in `setup.py`.  
  - From `chunk.py`: call the loaded function when `torch.cuda.is_available()` and your kernel is available; otherwise fall back to Triton.

- **Option B – Standalone .so**  
  - Build a shared library (e.g. `nvcc` + `-shared`), load it via `ctypes` or `cffi`, and call your `extern "C"` kernels. You’ll need to pass pointers to GPU tensors (e.g. `.data_ptr()` and stream).

- **Option C – Keep Triton, .cu for compliance**  
  - Leave the main path in Triton (`chunk.py`), and only add the minimal `.cu` stub above so the repo has a `.cu` file. No integration needed.

---

## 5. Build (if you integrate the kernel)

Example `setup.py` snippet for a single `.cu`:

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fla_quasar_cuda",
    ext_modules=[
        CUDAExtension(
            name="fla.ops.quasar.chunk_cuda",
            sources=["fla/ops/quasar/chunk_kernel.cu"],
            extra_compile_args={"nvcc": ["-O3", "--use_fast_math"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

Then in `chunk.py` you can try `from fla.ops.quasar.chunk_cuda import ...` and use your kernel when available; otherwise keep the Triton path.

---

## 6. Summary

| Goal | What to do |
|------|------------|
| Pass validation only | Add one minimal `.cu` (e.g. `quasar_stub.cu`) under the repo; no build or Python call needed. |
| Aim for top miner | Implement fused chunk recurrence (and optionally gate) in `.cu`, match Triton numerics, integrate via PyTorch C++ extension or .so, and keep Triton as fallback. |

Place at least one `.cu` file anywhere in your fork (e.g. `fla/ops/quasar/quasar_stub.cu`) to satisfy the “.cu file needed” requirement; then optionally replace or extend it with real kernels as in section 4.
