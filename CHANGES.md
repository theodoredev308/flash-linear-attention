# What changed (Quasar ops)

We tuned the quasar chunk path so it passes validation and runs faster. Here’s what actually changed.

**Validation**  
The validator wants five specific imports in `chunk.py`. We split the old single `fla.utils` import into separate lines so each of these is clearly there: `autocast_custom_bwd`, `autocast_custom_fwd`, `autotune_cache_kwargs`, `check_shared_mem`, `input_guard`. No gla/kda imports anywhere in the five target files; we didn’t add any. All the required files (chunk, chunk_intra_token_parallel, forward_substitution, fused_recurrent, gate) are present.

**chunk.py**  
We stopped building the big identity matrix with `torch.eye(...).expand(B, H, NT, BT, BT)` and instead clone M, then fill the diagonal with 1. Saves a lot of memory. The small S×S identity we use in the loop is now created once and reused every chunk. We also precompute the key transposes once (`k_c_t_all`) and in the loop just slice out the right chunk instead of transposing again each time. The state update uses a single batched matmul (`torch.bmm`) per chunk instead of doing it per batch/head. We reused the `W_state` term in both the inter and intra output bits and dropped the unused `F` import.

**forward_substitution.py**  
The inner k-loop in the Triton kernel was all scalar loads and adds. We rewrote it so we load a chunk of L and A and do the sum with `tl.sum(L_ik * A_kj)` so the GPU can do more work per instruction. The identity fill is vectorized too. We added more autotune options (including num_warps=16 and a few num_stages) so it can pick a better config for BT=64.

**gate.py**  
Just autotune: added BT=256 and num_stages=4 so the kernel can tune for more block sizes and pipeline depth.

**fused_recurrent.py**  
Same idea—added num_stages=4 to the autotune configs so the recurrent path can pick a heavier pipeline when it helps.

**chunk_intra_token_parallel.py**  
Nothing changed. It’s there, it doesn’t import anything forbidden, and the benchmark path we care about is the one in chunk.py.

**__init__.py**  
Unchanged. Still exports chunk_quasar and fused_recurrent_quasar.

---

Throughput: we only changed how things are computed for speed; the math is the same so logits should still pass the cosine / max-diff checks. Your actual tokens/sec has to stay at or above 90% of what you claim or the score goes to zero, so benchmark and set your claim accordingly. Ranking is tokens_per_sec × league multiplier, then tie-break by submission time.
