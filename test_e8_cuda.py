"""
E8 CUDA kernel correctness + speed test.
Mirrors the structure of test_triton_e8d4.py.
"""

import os
import torch
import time

# Force CUDA kernel path (no Triton for E8)
from kernels import e8_forward

DEVICE = "cuda"
DTYPE  = torch.float16
WARMUP = 50
REPS   = 200

E8_CODEBOOK_SIZE = 65535  # valid indices: 0 .. 65534


# ── Reference implementation ──────────────────────────────────────────────────

def e8_reference(Q, codebook, g, h, x, O_vals, O_col, O_row_ptr):
    """Pure-PyTorch reference: dequantize → matvec → correction → scale."""
    M, N = Q.shape
    B    = x.shape[0]

    idx  = (Q.int() + 32768) & 0xFFFF          # (M, N) uint16 in int32
    mask = (idx != 65535)                        # non-critical positions

    # Codebook lookup for non-critical weights: (M, N, 8)
    safe_idx = idx.clamp(0, E8_CODEBOOK_SIZE - 1)
    W = codebook[safe_idx.long()]                # (M, N, 8) fp16
    W = W * mask.unsqueeze(-1).to(W.dtype)       # zero out critical positions

    # Scaled matvec: y[b, m, k] = sum_n W[m,n,k] * h[n] * x[b,n]
    xh = x.float() * h.float().unsqueeze(0)      # (B, N)
    # einsum: y[b,m,k] = sum_n W[m,n,k] * xh[b,n]
    y = torch.einsum("mnk,bn->bmk", W.float(), xh)  # (B, M, 8)

    # Sparse correction
    for m in range(M):
        start = O_row_ptr[m].item()
        end   = O_row_ptr[m + 1].item()
        for i in range(start, end):
            col = O_col[i].item()
            hx  = h[col].float() * x[:, col].float()   # (B,)
            y[:, m, :] += O_vals[i].float().unsqueeze(0) * hx.unsqueeze(-1)

    # Apply row scale g
    g8 = g.float().view(M, 8)
    y  = y * g8.unsqueeze(0)                     # (B, M, 8)
    return y.reshape(B, M * 8).half()


# ── Test data helpers ─────────────────────────────────────────────────────────

def make_e8_args(M, N, B=1, pct_crit=0.01, seed=42):
    torch.manual_seed(seed)
    # Q: random indices or critical (65535 → stored as int16 32767)
    idx_raw = torch.randint(0, E8_CODEBOOK_SIZE, (M, N), dtype=torch.int32)
    crit    = torch.rand(M, N) < pct_crit
    idx_raw[crit] = 65535
    Q = (idx_raw - 32768).to(torch.int16).to(DEVICE)

    codebook = torch.randn(E8_CODEBOOK_SIZE, 8, dtype=DTYPE, device=DEVICE) * 0.1
    g        = torch.randn(M * 8, dtype=DTYPE, device=DEVICE).abs() + 0.5
    h        = torch.randn(N,     dtype=DTYPE, device=DEVICE).abs() + 0.5
    x        = torch.randn(B, N,  dtype=DTYPE, device=DEVICE)
    bias     = torch.empty(0,     dtype=DTYPE, device=DEVICE)

    # Build CSR for critical weights
    crit_mask = (idx_raw == 65535)               # (M, N)
    row_ptr   = [0]
    cols, vals = [], []
    for m in range(M):
        for n in range(N):
            if crit_mask[m, n]:
                cols.append(n)
                vals.append(torch.randn(8, dtype=DTYPE))
        row_ptr.append(len(cols))

    O_col     = torch.tensor(cols, dtype=torch.int32, device=DEVICE)
    O_vals    = torch.stack(vals).to(DEVICE) if vals else \
                torch.empty(0, 8, dtype=DTYPE, device=DEVICE)
    O_row_ptr = torch.tensor(row_ptr, dtype=torch.int32, device=DEVICE)

    return Q, codebook, g, h, x, O_vals, O_col, O_row_ptr, bias


# ── Correctness ───────────────────────────────────────────────────────────────

def test_correctness():
    print("=== Correctness ===")
    shapes = [
        (64  // 4,  256, "M= 16 N= 256"),   # M must be divisible by TILE_M=4
        (512 // 4, 4096, "M=128 N=4096"),
        (128 // 4, 4096, "M= 32 N=4096"),
    ]
    for (M, N, label) in shapes:
        Q, codebook, g, h, x, O_vals, O_col, O_row_ptr, bias = make_e8_args(M, N)

        ref  = e8_reference(Q, codebook, g, h, x, O_vals, O_col, O_row_ptr)
        cuda = e8_forward(Q, codebook, g, h, x, O_vals, O_col, O_row_ptr, bias)

        diff     = (ref.float() - cuda.float()).abs()
        max_diff = diff.max().item()
        mean_diff= diff.mean().item()
        status   = "PASS" if max_diff < 0.05 else "FAIL"
        print(f"  {label}: max_diff={max_diff:.4f}  mean_diff={mean_diff:.5f}  {status}")


# ── Speed ─────────────────────────────────────────────────────────────────────

def bench_us(fn, *args):
    for _ in range(WARMUP):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / REPS * 1e6


# Llama 3.1-8B shapes: M = out_features / 8, N = in_features
LLAMA_SHAPES = [
    (4096 // 8,  4096, "q/o/down 4k→4k "),
    (1024 // 8,  4096, "k/v_proj 4k→1k "),
    (14336 // 8, 4096, "gate/up  4k→14k"),
    (4096 // 8, 14336, "down     14k→4k"),
]


def test_speed():
    print("\n=== Speed (B=1, eager) ===")
    for (M, N, label) in LLAMA_SHAPES:
        args = make_e8_args(M, N, B=1)
        Q, codebook, g, h, x, O_vals, O_col, O_row_ptr, bias = args
        us = bench_us(e8_forward, Q, codebook, g, h, x, O_vals, O_col, O_row_ptr, bias)
        print(f"  {label}: {us:6.1f} μs")


if __name__ == "__main__":
    test_correctness()
    test_speed()
    print("\nDone.")
