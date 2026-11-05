"""Isolate the cost of sparse (critical weight) correction in the Triton kernel."""
import sys
import time
import torch

sys.path.insert(0, "/home/hmpark/eptq_new")
from eptq.kernels.triton_e8d4 import e8d4_triton_forward

device = "cuda"
torch.manual_seed(42)

def make_case(M, N, percmask, B=1):
    Q = torch.randint(-32768, 32767, (M, N), device=device).to(torch.int16)
    Q[Q == 32767] = 0
    codes    = torch.randn(512, 4, dtype=torch.float16, device=device)
    g        = torch.randn(M * 8, dtype=torch.float16, device=device).abs() + 0.1
    h        = torch.randn(N,    dtype=torch.float16, device=device).abs() + 0.1
    x        = torch.randn(B, N, dtype=torch.float16, device=device)

    n_crit = max(0, int(M * N * percmask))
    if n_crit == 0:
        O_vals   = torch.zeros(0, 8, dtype=torch.float16, device=device)
        O_col    = torch.zeros(0, dtype=torch.int32, device=device)
        O_row_ptr = torch.zeros(M + 1, dtype=torch.int32, device=device)
    else:
        O_vals   = torch.randn(n_crit, 8, dtype=torch.float16, device=device)
        O_col    = torch.randint(0, N, (n_crit,), dtype=torch.int32, device=device)
        counts   = torch.bincount(torch.randint(0, M, (n_crit,)).long(), minlength=M)
        O_row_ptr = torch.cat([
            torch.zeros(1, dtype=torch.int32),
            counts.cumsum(0).to(torch.int32),
        ]).to(device)

    bias = torch.zeros(0, dtype=torch.float16, device=device)
    return Q, codes, g, h, x, O_vals, O_col, O_row_ptr, bias

def bench(fn, *args, warmup=5, repeat=100):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat * 1e6

shapes = [
    (512,  4096,  "q/o_proj   4k→4k"),
    (128,  4096,  "k/v_proj   4k→1k"),
    (1792, 4096,  "gate/up    4k→14k"),
    (512,  14336, "down_proj 14k→4k"),
]

print(f"{'Shape':25s}  {'percmask':>10s}  {'n_crit/row':>12s}  {'μs':>8s}")
print("-" * 65)
for M, N, label in shapes:
    for percmask in [0.0, 0.002, 0.01]:
        args = make_case(M, N, percmask)
        n_crit_total = args[5].shape[0]
        n_per_row = n_crit_total / M
        t = bench(e8d4_triton_forward, *args)
        print(f"  {label:23s}  {percmask:10.3f}  {n_per_row:12.1f}  {t:8.1f}")
    print()

print("Done.")
