"""
Block-tiled Triton SWA kernel using tl.dot for tensor core utilization.

Key optimization: process BLOCK_N queries per block, sharing K/V in SRAM.
For SWA(17): window=35, so K/V per block = BLOCK_N + 34 positions.

Usage: CUDA_VISIBLE_DEVICES=5 python triton_swa_v2.py
"""
import torch
import triton
import triton.language as tl
import time
import numpy as np
import torch.nn.functional as F
import math


@triton.jit
def swa_attn_v2_kernel(
    Q, K, V, Out,
    stride_qb, stride_qn, stride_qh, stride_qd,
    stride_kb, stride_kn, stride_kh, stride_kd,
    stride_vb, stride_vn, stride_vh, stride_vd,
    stride_ob, stride_on, stride_oh, stride_od,
    N, scale,
    WINDOW: tl.constexpr,      # 17
    BLOCK_N: tl.constexpr,     # queries per block (e.g., 64)
    BLOCK_D: tl.constexpr,     # head dim (64)
    BLOCK_KV: tl.constexpr,    # padded K/V window (next power of 2 >= BLOCK_N + 2*WINDOW)
):
    """Block-tiled SWA with online softmax."""
    block_n = tl.program_id(0)  # Which query block
    h = tl.program_id(1)        # Head
    b = tl.program_id(2)        # Batch

    # Query positions for this block
    q_start = block_n * BLOCK_N
    q_offs = tl.arange(0, BLOCK_N)
    q_pos = q_start + q_offs  # (BLOCK_N,)
    q_mask = q_pos < N

    # K/V window: [q_start - WINDOW, q_start + BLOCK_N + WINDOW)
    kv_start = q_start - WINDOW
    kv_offs = tl.arange(0, BLOCK_KV)
    kv_pos = kv_start + kv_offs  # (BLOCK_KV,)
    kv_mask = (kv_pos >= 0) & (kv_pos < N)

    d_offs = tl.arange(0, BLOCK_D)

    # Load Q block: (BLOCK_N, BLOCK_D)
    q_ptrs = Q + b * stride_qb + q_pos[:, None] * stride_qn + h * stride_qh + d_offs[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=q_mask[:, None] & (d_offs[None, :] < BLOCK_D), other=0.0).to(tl.float32)

    # Load K window: (BLOCK_KV, BLOCK_D)
    k_ptrs = K + b * stride_kb + kv_pos[:, None] * stride_kn + h * stride_kh + d_offs[None, :] * stride_kd
    k = tl.load(k_ptrs, mask=kv_mask[:, None] & (d_offs[None, :] < BLOCK_D), other=0.0).to(tl.float32)

    # Compute scores: Q @ K^T = (BLOCK_N, BLOCK_KV)
    scores = tl.dot(q, tl.trans(k)) * scale

    # Apply SWA mask: query at q_pos[i] can only attend to kv_pos[j] where |q_pos[i] - kv_pos[j]| <= WINDOW
    # For each (i, j): check if |q_start + i - (kv_start + j)| <= WINDOW
    # = |i - j + WINDOW| <= WINDOW (since kv_start = q_start - WINDOW)
    # = |i - j + WINDOW| <= WINDOW
    # This simplifies to: j >= i and j < i + 2*WINDOW + 1
    # i.e., j in [i, i + 2*WINDOW + 1)
    rel_pos = q_offs[:, None] + WINDOW - kv_offs[None, :]  # (BLOCK_N, BLOCK_KV)
    swa_mask = (rel_pos >= -WINDOW) & (rel_pos <= WINDOW)

    # Combined mask: SWA + valid positions
    valid_mask = swa_mask & kv_mask[None, :] & q_mask[:, None]
    scores = tl.where(valid_mask, scores, float('-inf'))

    # Row-wise softmax
    row_max = tl.max(scores, axis=1)  # (BLOCK_N,)
    scores = scores - row_max[:, None]
    exp_scores = tl.exp(scores)
    exp_scores = tl.where(valid_mask, exp_scores, 0.0)
    row_sum = tl.sum(exp_scores, axis=1)  # (BLOCK_N,)
    weights = exp_scores / row_sum[:, None]

    # Load V window: (BLOCK_KV, BLOCK_D)
    v_ptrs = V + b * stride_vb + kv_pos[:, None] * stride_vn + h * stride_vh + d_offs[None, :] * stride_vd
    v = tl.load(v_ptrs, mask=kv_mask[:, None] & (d_offs[None, :] < BLOCK_D), other=0.0).to(tl.float32)

    # Output: weights @ V = (BLOCK_N, BLOCK_D)
    out = tl.dot(weights.to(v.dtype), v)

    # Store
    o_ptrs = Out + b * stride_ob + q_pos[:, None] * stride_on + h * stride_oh + d_offs[None, :] * stride_od
    tl.store(o_ptrs, out.to(tl.bfloat16), mask=q_mask[:, None] & (d_offs[None, :] < BLOCK_D))


def triton_swa_attn_v2(q, k, v, window=17):
    """Block-tiled Triton SWA. q,k,v: (B, N, H, D). Returns (B, N, H, D)."""
    B, N, H, D = q.shape
    out = torch.empty_like(q)
    scale = D ** -0.5

    BLOCK_N = 64
    # K/V window per block = BLOCK_N + 2*window = 64 + 34 = 98 → pad to 128
    BLOCK_KV = 128  # Next power of 2 >= 98

    num_blocks_n = (N + BLOCK_N - 1) // BLOCK_N
    grid = (num_blocks_n, H, B)

    swa_attn_v2_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        N, scale,
        WINDOW=window,
        BLOCK_N=BLOCK_N,
        BLOCK_D=D,
        BLOCK_KV=BLOCK_KV,
    )
    return out


if __name__ == "__main__":
    from flash_attn import flash_attn_func

    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Test accuracy
    for B, N, H, D in [(1, 200, 48, 64), (1, 1836, 48, 64), (1, 5000, 48, 64)]:
        q = torch.randn(B, N, H, D, device=device, dtype=torch.bfloat16)
        k = torch.randn(B, N, H, D, device=device, dtype=torch.bfloat16)
        v = torch.randn(B, N, H, D, device=device, dtype=torch.bfloat16)

        ref = flash_attn_func(q, k, v, window_size=(17, 17))
        out = triton_swa_attn_v2(q, k, v, window=17)

        cos = F.cosine_similarity(ref.flatten().float(), out.flatten().float(), dim=0).item()
        print(f"B={B} N={N:5d} H={H}: cos={cos:.6f}")

    # Benchmark
    print("\n--- Benchmark ---")
    for N_tok in [1836, 21964]:
        B, H, D = 1, 48, 64
        q = torch.randn(B, N_tok, H, D, device=device, dtype=torch.bfloat16)
        k = torch.randn(B, N_tok, H, D, device=device, dtype=torch.bfloat16)
        v = torch.randn(B, N_tok, H, D, device=device, dtype=torch.bfloat16)

        for _ in range(5):
            triton_swa_attn_v2(q, k, v)
            flash_attn_func(q, k, v, window_size=(17, 17))
            torch.cuda.synchronize()

        times = []
        for _ in range(20):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            triton_swa_attn_v2(q, k, v)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        ms_t = np.median(times) * 1000

        times = []
        for _ in range(20):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            flash_attn_func(q, k, v, window_size=(17, 17))
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        ms_f = np.median(times) * 1000

        ratio = ms_t / ms_f
        print(f"N={N_tok:6d}: v2={ms_t:.2f}ms  flash={ms_f:.2f}ms  ratio={ratio:.1f}x")
        print(f"  ×12 layers: v2={ms_t*12:.1f}ms  flash={ms_f*12:.1f}ms")
