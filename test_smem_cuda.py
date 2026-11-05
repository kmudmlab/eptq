"""Benchmark CUDA e8d4_fused_kernel (SMEM codebook) vs Triton."""
import torch
from eptq.kernels import e8d4_forward
from eptq.kernels.triton_e8d4 import e8d4_triton_forward

device = "cuda"

def make_case(M, N, percmask=0.001):
    Q_raw = torch.randint(-32767, 32767, (M, N), device=device).to(torch.int16)
    mask = torch.rand(M, N, device=device) < percmask
    Q_raw[mask] = torch.iinfo(torch.int16).max  # 32767 = 65535u after +32768
    # ensure 65535 sentinel
    sentinel_idx = (torch.randint(0, 65535, (M, N), device=device).to(torch.int16))
    Q = sentinel_idx
    codes = torch.randn(512, 4, device=device, dtype=torch.float16)
    g = torch.randn(M * 8, device=device, dtype=torch.float16).abs() + 0.1
    h = torch.randn(N, device=device, dtype=torch.float16).abs() + 0.1
    x = torch.randn(1, N, device=device, dtype=torch.float16)
    n_crit = max(1, int(M * N * percmask))
    O_vals = torch.randn(n_crit, 8, device=device, dtype=torch.float16)
    O_col  = torch.zeros(n_crit, device=device, dtype=torch.int32)
    O_row_ptr = torch.zeros(M + 1, device=device, dtype=torch.int32)
    bias = torch.empty(0, device=device, dtype=torch.float16)
    return Q, codes, g, h, x, O_vals, O_col, O_row_ptr, bias

def bench(fn, *args, warmup=20, iters=200):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    t.record()
    for _ in range(iters):
        fn(*args)
    e.record()
    torch.cuda.synchronize()
    return t.elapsed_time(e) / iters * 1000  # μs

shapes = [
    ("q_proj 4k→4k",    512,  4096),
    ("k/v_proj 4k→1k",  128,  4096),
    ("gate/up 4k→14k", 1792, 4096),
    ("down 14k→4k",     512, 14336),
]

print("=== CUDA (SMEM codebook) vs Triton ===")
print(f"{'Layer':<22} {'CUDA':>8} {'Triton':>8} {'speedup':>8}")
for name, M, N in shapes:
    args = make_case(M, N)
    Q, codes, g, h, x, O_vals, O_col, O_row_ptr, bias = args

    cuda_us   = bench(e8d4_forward, Q, codes, g, h, x, O_vals, O_col, O_row_ptr, bias)
    triton_us = bench(e8d4_triton_forward, Q, codes, g, h, x, O_vals, O_col, O_row_ptr, bias)
    print(f"  {name:<20} {cuda_us:>7.1f}μs {triton_us:>7.1f}μs {triton_us/cuda_us:>7.2f}×")

print("\nDone.")
