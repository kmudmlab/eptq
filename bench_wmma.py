"""Throughput benchmark: EPTQ kernels vs fp16 nn.Linear (cuBLAS)."""
import torch
import torch.nn as nn
import time

device = "cuda"
torch.cuda.set_device(0)

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from eptq.kernels import e8d4_forward


def bench_eptq(Q, codes, g, h, x, O_vals, O_col, O_row_ptr, bias,
               warmup=20, iters=200):
    for _ in range(warmup):
        e8d4_forward(Q, codes, g, h, x, O_vals, O_col, O_row_ptr, bias)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        e8d4_forward(Q, codes, g, h, x, O_vals, O_col, O_row_ptr, bias)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def bench_fp16(linear, x, warmup=20, iters=200):
    for _ in range(warmup):
        linear(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        linear(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def make_tensors(M, N, B, device):
    Q = torch.randint(-32767, 32767, (M, N), dtype=torch.int16, device=device)
    codes = torch.randn(512, 4, dtype=torch.float16, device=device)
    g = torch.ones(M * 8, dtype=torch.float16, device=device)
    h = torch.ones(N, dtype=torch.float16, device=device)
    x = torch.randn(B, N, dtype=torch.float16, device=device)
    O_vals   = torch.empty(0, 8, dtype=torch.float16, device=device)
    O_col    = torch.empty(0, dtype=torch.int32, device=device)
    O_row_ptr = torch.zeros(M + 1, dtype=torch.int32, device=device)
    bias     = torch.empty(0, dtype=torch.float16, device=device)
    return Q, codes, g, h, x, O_vals, O_col, O_row_ptr, bias


print("=" * 90)
print("E8-D4 WMMA v2 kernel vs fp16 nn.Linear (cuBLAS)")
print(f"WMMA threshold: B >= 32   Output tile: 64 out × 64 batch   K-tile: 32")
print("=" * 90)

# Real OPT layer shapes
configs = [
    # (out, in, label)
    (768,  768,  "opt-125m qkvo  "),
    (3072, 768,  "opt-125m fc1   "),
    (768,  3072, "opt-125m fc2   "),
    (2048, 2048, "opt-1.3b qkvo  "),
    (8192, 2048, "opt-1.3b fc1   "),
    (2048, 8192, "opt-1.3b fc2   "),
    (2560, 2560, "opt-2.7b qkvo  "),
    (10240,2560, "opt-2.7b fc1   "),
]

print(f"\n{'layer':20s} {'B':>6s} | {'EPTQ ms':>10s} {'fp16 ms':>10s} | "
      f"{'EPTQ tok/s':>12s} {'fp16 tok/s':>12s} | {'EPTQ/fp16':>10s}")
print("-" * 90)

for out_f, in_f, cfg_label in configs:
    M = out_f // 8
    N = in_f

    # One nn.Linear per config, reused across B
    linear = nn.Linear(in_f, out_f, bias=False).half().to(device)

    for B in [1, 4, 16, 32, 64, 128, 512, 2048]:
        tensors = make_tensors(M, N, B, device)
        x_fp16 = tensors[4]

        eptq_t = bench_eptq(*tensors)
        fp16_t = bench_fp16(linear, x_fp16)

        eptq_tps = B / eptq_t
        fp16_tps = B / fp16_t
        ratio = eptq_tps / fp16_tps

        mark = "  ⚡" if ratio > 1.0 else ("  ≈" if ratio > 0.7 else "")
        print(f"{cfg_label:20s} {B:>6d} | {eptq_t*1e3:>10.3f} {fp16_t*1e3:>10.3f} | "
              f"{eptq_tps:>12.0f} {fp16_tps:>12.0f} | {ratio:>10.2f}×{mark}")
    print()
