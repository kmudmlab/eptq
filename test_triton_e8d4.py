"""Quick correctness + speed test for the Triton e8d4 kernel vs CUDA kernel."""
import os, sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eptq.kernels.triton_e8d4 import e8d4_triton_forward
from eptq.kernels import e8d4_forward

device = "cuda"
torch.manual_seed(42)


def make_test_case(M, N, B=1, percmask=0.001):
    """Build a random but valid e8d4 quantized layer."""
    total = M * N
    n_crit = max(1, int(total * percmask))

    Q = torch.randint(-32768, 32767, (M, N), device=device).to(torch.int16)
    Q[Q == 32767] = 0

    codes = torch.randn(512, 4, dtype=torch.float16, device=device)
    g     = torch.randn(M * 8, dtype=torch.float16, device=device).abs() + 0.1
    h     = torch.randn(N,     dtype=torch.float16, device=device).abs() + 0.1
    x     = torch.randn(B, N,  dtype=torch.float16, device=device)

    O_vals    = torch.randn(n_crit, 8, dtype=torch.float16, device=device)
    O_col     = torch.randint(0, N, (n_crit,), dtype=torch.int32, device=device)
    counts    = torch.bincount(torch.randint(0, M, (n_crit,)).to(torch.long), minlength=M)
    O_row_ptr = torch.cat([
        torch.zeros(1, dtype=torch.int32),
        counts.cumsum(0).to(torch.int32),
    ]).to(device)

    bias = torch.zeros(0, dtype=torch.float16, device=device)
    return Q, codes, g, h, x, O_vals, O_col, O_row_ptr, bias


def triton_args(Q, codes, g, h, x, O_vals, O_col, O_row_ptr, bias):
    """Convert (Q, codes, g, h, x, ...) → (Q, codes, g, x_h, ...) for Triton v5.
    Keep x_h as fp32 to match CUDA's fp32 multiply exactly — no fp16 rounding step."""
    x_h = x.float() * h.float()   # (B, N) fp32
    return Q, codes, g, x_h, O_vals, O_col, O_row_ptr, bias


def bench(fn, *args, warmup=5, repeat=50):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat * 1e6  # μs


# ── correctness tests ────────────────────────────────────────────────────────
print("=== Correctness ===")
for M, N in [(64, 256), (512, 4096), (128, 4096), (1792, 4096), (512, 14336)]:
    args_cuda   = make_test_case(M, N)
    args_triton = triton_args(*args_cuda)
    y_cuda      = e8d4_forward(*args_cuda)
    y_triton    = e8d4_triton_forward(*args_triton)
    diff = (y_cuda.float() - y_triton.float()).abs()
    ok = diff.max().item() < 0.05
    print(f"  M={M:4d} N={N:5d}: max_diff={diff.max():.4f}  mean_diff={diff.mean():.5f}  {'PASS' if ok else 'FAIL'}")

# ── speed tests: eager ───────────────────────────────────────────────────────
print("\n=== Speed (B=1, eager) ===")
shapes = [
    (512,  4096,  "q_proj/o_proj/down 4k→4k"),
    (128,  4096,  "k/v_proj 4k→1k"),
    (1792, 4096,  "gate/up_proj 4k→14k"),
    (512,  14336, "down_proj 14k→4k"),
]
for M, N, label in shapes:
    args_cuda   = make_test_case(M, N, B=1)
    args_triton = triton_args(*args_cuda)
    t_cuda   = bench(e8d4_forward, *args_cuda)
    t_triton = bench(e8d4_triton_forward, *args_triton)
    speedup  = t_cuda / t_triton
    print(f"  {label:30s}: CUDA {t_cuda:6.1f} μs  Triton {t_triton:6.1f} μs  speedup {speedup:.2f}×")

# ── speed tests: torch.compile ───────────────────────────────────────────────
print("\n=== Speed (B=1, torch.compile) ===")
compiled_cuda   = torch.compile(e8d4_forward)
compiled_triton = torch.compile(e8d4_triton_forward)

for M, N, label in shapes:
    args_cuda   = make_test_case(M, N, B=1)
    args_triton = triton_args(*args_cuda)
    t_cuda   = bench(compiled_cuda, *args_cuda)
    t_triton = bench(compiled_triton, *args_triton)
    speedup  = t_cuda / t_triton
    print(f"  {label:30s}: CUDA {t_cuda:6.1f} μs  Triton {t_triton:6.1f} μs  speedup {speedup:.2f}×")

print("\nDone.")
