"""Diagnose which kernel (CUDA or Triton) produces NaN for M=512, N=4096."""
import sys
import torch
import triton
import triton.language as tl

sys.path.insert(0, "/home/hmpark/eptq_new")
from eptq.kernels import e8d4_forward

device = "cuda"
torch.manual_seed(42)

def make_test_case(M, N, B=1, percmask=0.001):
    total = M * N
    n_crit = max(1, int(total * percmask))
    Q = torch.randint(-32768, 32767, (M, N), device=device).to(torch.int16)
    Q[Q == 32767] = 0
    codes    = torch.randn(512, 4, dtype=torch.float16, device=device)
    g        = torch.randn(M * 8, dtype=torch.float16, device=device).abs() + 0.1
    h        = torch.randn(N,     dtype=torch.float16, device=device).abs() + 0.1
    x        = torch.randn(B, N,  dtype=torch.float16, device=device)
    O_vals   = torch.randn(n_crit, 8, dtype=torch.float16, device=device)
    O_col    = torch.randint(0, N, (n_crit,), dtype=torch.int32, device=device)
    counts   = torch.bincount(torch.randint(0, M, (n_crit,)).to(torch.long), minlength=M)
    O_row_ptr = torch.cat([
        torch.zeros(1, dtype=torch.int32),
        counts.cumsum(0).to(torch.int32),
    ]).to(device)
    bias = torch.zeros(0, dtype=torch.float16, device=device)
    return Q, codes, g, h, x, O_vals, O_col, O_row_ptr, bias


# A minimal Triton kernel with forced BLOCK_N (no autotune)
@triton.jit
def _e8d4_single_config(
    Q_ptr, codes_ptr, g_ptr, h_ptr, x_ptr, y_ptr,
    O_vals_ptr, O_col_ptr, O_row_ptr,
    M, N, M8, stride_x, stride_y,
    BLOCK_N: tl.constexpr,
):
    m     = tl.program_id(0)
    b     = tl.program_id(1)
    x_off = b * stride_x
    y_off = b * stride_y
    q_row = m * N
    d4 = tl.arange(0, 4)
    pa_ci = tl.zeros((4, BLOCK_N), dtype=tl.float32)
    pa_cj = tl.zeros((4, BLOCK_N), dtype=tl.float32)
    for k_base in range(0, N, BLOCK_N):
        k_offs = k_base + tl.arange(0, BLOCK_N)
        mask   = k_offs < N
        h_v = tl.load(h_ptr + k_offs, mask=mask, other=0.0).to(tl.float32)
        x_v = tl.load(x_ptr + x_off + k_offs, mask=mask, other=0.0).to(tl.float32)
        hx  = h_v * x_v
        q   = tl.load(Q_ptr + q_row + k_offs, mask=mask, other=-32768).to(tl.int32)
        idx = (q + 32768) & 0xFFFF
        ok  = (idx < 65535) & mask
        hx  = tl.where(ok, hx, 0.0)
        ct  = (idx >> 14) & 3
        idi = (idx >> 7)  & 0x7F
        idj =  idx        & 0x7F
        ci  = (ct * 128 + idi) * 4
        cj  = (ct * 128 + idj) * 4
        ok4  = ok[:, None]
        ci_v = tl.load(codes_ptr + ci[:, None] + d4[None, :], mask=ok4, other=0.0).to(tl.float32)
        cj_v = tl.load(codes_ptr + cj[:, None] + d4[None, :], mask=ok4, other=0.0).to(tl.float32)
        pa_ci += tl.trans(ci_v) * hx[None, :]
        pa_cj += tl.trans(cj_v) * hx[None, :]
    a_ci = tl.sum(pa_ci, axis=1)
    a_cj = tl.sum(pa_cj, axis=1)
    # no sparse correction for simplicity
    gbase  = m * 8
    res_ci = a_ci * tl.load(g_ptr + gbase + d4    ).to(tl.float32)
    res_cj = a_cj * tl.load(g_ptr + gbase + d4 + 4).to(tl.float32)
    out_ci = y_off + gbase + d4
    out_cj = y_off + gbase + d4 + 4
    tl.store(y_ptr + out_ci, res_ci.to(tl.float16))
    tl.store(y_ptr + out_cj, res_cj.to(tl.float16))


def triton_forward_fixed(Q, codes, g, h, x, BLOCK_N):
    M, N = Q.shape
    B    = x.shape[0]
    M8   = M * 8
    y = torch.empty(B, M8, dtype=torch.float16, device=x.device)
    _e8d4_single_config[(M, B)](
        Q, codes, g, h, x, y,
        None, None, None,
        M=M, N=N, M8=M8,
        stride_x=N, stride_y=M8,
        BLOCK_N=BLOCK_N,
    )
    return y


print("=== NaN Diagnosis ===")
for M, N in [(64, 256), (512, 4096), (128, 4096), (1792, 4096), (512, 14336)]:
    args = make_test_case(M, N)
    Q, codes, g, h, x, O_vals, O_col, O_row_ptr, bias = args

    y_cuda = e8d4_forward(*args)
    has_nan_cuda = torch.isnan(y_cuda).any().item()

    # Test with multiple BLOCK_N values
    results = {}
    for bn in [128, 256, 512, 1024]:
        if bn > N:
            continue
        try:
            y_t = triton_forward_fixed(Q, codes, g, h, x, bn)
            has_nan_t = torch.isnan(y_t).any().item()
            results[bn] = f"NaN={has_nan_t}"
        except Exception as e:
            results[bn] = f"ERR={e}"

    print(f"M={M:4d} N={N:5d}: CUDA_NaN={has_nan_cuda}  Triton: {results}")
