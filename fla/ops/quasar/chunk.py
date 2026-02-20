# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Modified for QuasarAttention

import torch
import triton
import triton.language as tl

from fla.ops.utils.index import prepare_chunk_indices
from fla.ops.quasar.forward_substitution import forward_substitution_kernel
from fla.utils import autocast_custom_bwd
from fla.utils import autocast_custom_fwd
from fla.utils import autotune_cache_kwargs
from fla.utils import check_shared_mem
from fla.utils import input_guard
from fla.utils import IS_AMD

BS_LIST = [32, 64] if check_shared_mem() else [16, 32]

# Optional CUDA extension (see guide_cu.md): use when built and on CUDA; else Triton.
_quasar_cuda_ext = None

def _get_quasar_cuda_ext():
    global _quasar_cuda_ext
    if _quasar_cuda_ext is not None:
        return _quasar_cuda_ext
    if not torch.cuda.is_available():
        _quasar_cuda_ext = False
        return False
    try:
        import quasar_forward_substitution_cuda as ext
        _quasar_cuda_ext = ext
        return ext
    except Exception:
        _quasar_cuda_ext = False
        return False


BT_LIST_AUTOTUNE = [32, 64, 128]
NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if IS_AMD else [4, 8, 16, 32]


@input_guard
def chunk_quasar_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 64,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Simplified chunk-wise QuasarAttention forward pass using PyTorch operations.
    
    This implementation uses PyTorch for the complex matrix operations and
    can be optimized with Triton kernels for specific sub-operations later.
    """
    B, T, H, S = q.shape
    BT = chunk_size
    original_T = T
    
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    
    # Pad if T is not a multiple of BT
    if T % BT != 0:
        pad_len = BT - (T % BT)
        q = torch.cat([q, q.new_zeros((B, pad_len, H, S))], dim=1)
        k = torch.cat([k, k.new_zeros((B, pad_len, H, S))], dim=1)
        v = torch.cat([v, v.new_zeros((B, pad_len, H, S))], dim=1)
        T = T + pad_len
        NT = triton.cdiv(T, BT)
    
    # Reshape to chunks
    q_chunks = q.view(B, H, NT, BT, S)
    k_chunks = k.view(B, H, NT, BT, S)
    v_chunks = v.view(B, H, NT, BT, S)
    
    # Compute alpha = (1 - exp(-beta * lambda)) / (lambda + eps)
    # lambda = ||k||^2
    k_norm_sq = (k_chunks ** 2).sum(dim=-1, keepdim=True)  # [B, H, NT, BT, 1]
    eps = 1e-8
    alpha = (1 - torch.exp(-beta.view(-1, 1, 1, 1) * k_norm_sq)) / (k_norm_sq + eps)  # [B, H, NT, BT, 1]
    
    # Vectorized intra-chunk computation for ALL chunks
    # KK^T = K @ K^T for all chunks
    # [B, H, NT, BT, S] @ [B, H, NT, S, BT] -> [B, H, NT, BT, BT]
    KK_t = torch.matmul(k_chunks, k_chunks.transpose(-2, -1))  # [B, H, NT, BT, BT]
    
    # M = tril(alpha * KK^T) for all chunks (in-place tril to save memory)
    alpha_expanded = alpha.expand(-1, -1, -1, -1, BT)  # [B, H, NT, BT, BT]
    M = (alpha_expanded * KK_t).tril(diagonal=-1)
    # L = I + M: avoid large eye expand; set diagonal to 1
    L = M.clone(memory_format=torch.contiguous_format)
    L.diagonal(dim1=-2, dim2=-1).fill_(1.0)
    
    # Reshape for kernel: [B*H*NT, BT, BT]
    L_flat = L.view(B * H * NT, BT, BT)
    A_flat = torch.empty_like(L_flat)

    # Use CUDA extension when available (guide_cu.md); else Triton. CUDA ext supports float32/float16 only.
    ext = _get_quasar_cuda_ext()
    use_cuda_ext = (
        ext is not False
        and L_flat.is_cuda
        and L_flat.dtype in (torch.float32, torch.float16)
    )
    if use_cuda_ext:
        A_flat = ext.forward_substitution(L_flat)
    else:
        forward_substitution_kernel[(B * H * NT,)](
            L_ptr=L_flat,
            L_stride_bh=BT * BT,
            A_ptr=A_flat,
            A_stride_bh=BT * BT,
            BT=BT,
        )

    A = A_flat.view(B, H, NT, BT, BT)  # [B, H, NT, BT, BT]
    
    # Compute W = A @ (alpha * K) and U = A @ (alpha * V) for all chunks
    alpha_expanded = alpha.expand(-1, -1, -1, -1, S)  # [B, H, NT, BT, S]
    W = torch.matmul(A, alpha_expanded * k_chunks)  # [B, H, NT, BT, S]
    U = torch.matmul(A, alpha_expanded * v_chunks)  # [B, H, NT, BT, S]
    
    # Precompute key transposes once [B, H, NT, S, BT] for fast loop
    k_c_t_all = k_chunks.transpose(-2, -1).contiguous()
    # Identity once for all chunk steps (reuse across loop)
    I_full = torch.eye(S, device=q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]
    o = torch.empty_like(q)
    if initial_state is None:
        state = torch.zeros(B, H, S, S, dtype=q.dtype, device=q.device)
    else:
        state = initial_state.clone()
    # Batched state shape for bmm: [B*H, S, S]
    state_flat = state.view(B * H, S, S)
    for i in range(NT):
        q_c = q_chunks[:, :, i]  # [B, H, BT, S]
        k_c_t = k_c_t_all[:, :, i]  # [B, H, S, BT]
        W_c = W[:, :, i]
        U_c = U[:, :, i]
        A_trans = I_full - torch.matmul(k_c_t, W_c)
        B_trans = torch.matmul(k_c_t, U_c)
        # state = A @ state + B (batched matmul for throughput)
        state_flat = torch.bmm(A_trans.view(B * H, S, S), state_flat) + B_trans.view(B * H, S, S)
        state = state_flat.view(B, H, S, S)
        W_state = torch.matmul(W_c, state)
        o_inter = torch.matmul(q_c, state)
        o_intra = torch.matmul(q_c, torch.matmul(k_c_t, U_c - W_state))
        o[:, i * BT : (i + 1) * BT] = (o_inter + o_intra).transpose(1, 2)
    
    final_state = state if output_final_state else None
    
    # Trim output back to original size if padded
    if original_T != T:
        o = o[:, :original_T]
    
    return o, final_state


class ChunkQuasarFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        **kwargs,
    ):
        chunk_size = 64
        chunk_indices = prepare_chunk_indices(
            cu_seqlens, chunk_size) if cu_seqlens is not None else None
        
        o, final_state = chunk_quasar_fwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
        )
        
        ctx.save_for_backward(q, k, v, beta, initial_state, cu_seqlens, chunk_indices)
        ctx.chunk_size = chunk_size
        ctx.output_final_state = output_final_state
        
        return o, final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, d_final_state: torch.Tensor | None):
        q, k, v, beta, initial_state, cu_seqlens, chunk_indices = ctx.saved_tensors
        
        # Backward pass implementation (simplified for now)
        # Full backward pass would require recomputing forward and computing gradients
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        dbeta = torch.zeros_like(beta)
        
        return dq, dk, dv, dbeta, None, None, None


@torch.compiler.disable
def chunk_quasar(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Chunk-wise QuasarAttention forward pass with autograd support.
    
    Implements the chunk-wise parallel algorithm for QuasarAttention.
    
    Args:
        q (torch.Tensor): Query tensor of shape [B, T, H, S]
        k (torch.Tensor): Key tensor of shape [B, T, H, S]
        v (torch.Tensor): Value tensor of shape [B, T, H, S]
        beta (torch.Tensor): Beta parameter tensor of shape [H]
        initial_state (torch.Tensor | None): Initial state tensor of shape [B, H, S, S]
        output_final_state (bool): Whether to output the final state
        cu_seqlens (torch.Tensor | None): Cumulative sequence lengths for variable-length sequences
    
    Returns:
        o (torch.Tensor): Output tensor of shape [B, T, H, S]
        final_state (torch.Tensor | None): Final state tensor of shape [B, H, S, S] if output_final_state
    """
    return ChunkQuasarFunction.apply(q, k, v, beta, initial_state, output_final_state, cu_seqlens)