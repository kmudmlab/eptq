"""
Triton kernel for E8-D4 decode matvec.

v7 design (vs v5):
  - Parallel sparse correction: replaces serial while-loop with a range loop
    that processes BLOCK_N correction entries at once — same (BLOCK_N,4) gather
    + tl.trans + tl.sum pattern as the main N-loop.
    For ~1% masking and typical N (4096/14336), a single loop iteration covers
    all critical entries per row (e.g. N=4096 → ~41 entries << BLOCK_N=256).

v5 design (vs v4):
  - x_h = x * h precomputed outside the kernel → 1 global load per column
    instead of 2 (h_v + x_v), and 1 fewer multiply per column.
  - Grid: (M, B) — one program per Q-row × batch item.
  - (4, BLOCK_N) accumulators for the two D4 halves.
  - Inner loop: 2 vectorized (BLOCK_N, 4) gathers + tl.trans + element-wise FMA.
    → Halves the number of codebook load instructions vs 8 scalar loads.
  - ONE tl.sum(axis=1) per D4 half AFTER the loop (2 reductions total).
  - OUT_DTYPE constexpr: native fp16/bfloat16 output — no Python-side dtype cast.
"""

import triton
import triton.language as tl
import torch
from torch import Tensor


TRITON_BATCH_THRESHOLD = 256


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 256},  num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_N": 256},  num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_N": 256},  num_warps=8,  num_stages=3),
        triton.Config({"BLOCK_N": 256},  num_warps=16, num_stages=2),
        triton.Config({"BLOCK_N": 512},  num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_N": 512},  num_warps=16, num_stages=2),
        triton.Config({"BLOCK_N": 512},  num_warps=8,  num_stages=3),
        triton.Config({"BLOCK_N": 1024}, num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_N": 1024}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_N": 1024}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_N": 128},  num_warps=4,  num_stages=2),
    ],
    key=["M", "N"],
)
@triton.jit
def _e8d4_matvec_kernel(
    Q_ptr,          # (M, N) int16
    codes_ptr,      # (512, 4) fp16
    g_ptr,          # (M*8,) fp16
    xh_ptr,         # (B, N) fp32 — pre-multiplied x*h
    y_ptr,          # (B, M*8) fp16/bfloat16
    O_vals_ptr,     # (n_crit, 8) fp16
    O_col_ptr,      # (n_crit,) int32
    O_row_ptr,      # (M+1,) int32
    bias_ptr,       # (M*8,) fp16
    M, N, M8,
    stride_xh,
    stride_y,
    has_bias: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """
    Grid: (M, B).  Program (m, b) computes y[b, m*8 : m*8+8].

    xh[b, k] = x[b, k] * h[k] is precomputed before launch, saving 1 global
    load and 1 multiply per column vs the v4 design that loaded h and x separately.

    Uses (4, BLOCK_N) partial accumulators so that:
      - 2 vectorized (BLOCK_N, 4) codebook gathers replace 8 scalar gathers.
      - 2 reductions (axis=1) replace 8 scalar reductions.
      - Sparse correction and output use (4,) vector ops.
    """
    m      = tl.program_id(0)
    b      = tl.program_id(1)
    xh_off = b * stride_xh
    y_off  = b * stride_y
    q_row  = m * N

    d4 = tl.arange(0, 4)   # compile-time const (4,) range

    # (4, BLOCK_N) partial accumulators — ci = first D4 half, cj = second.
    pa_ci = tl.zeros((4, BLOCK_N), dtype=tl.float32)
    pa_cj = tl.zeros((4, BLOCK_N), dtype=tl.float32)

    # ── N-loop ────────────────────────────────────────────────────────────
    for k_base in range(0, N, BLOCK_N):
        k_offs = k_base + tl.arange(0, BLOCK_N)
        mask   = k_offs < N

        # Single load of pre-multiplied xh = x * h
        hx  = tl.load(xh_ptr + xh_off + k_offs, mask=mask, other=0.0).to(tl.float32)

        q   = tl.load(Q_ptr + q_row + k_offs, mask=mask, other=-32768).to(tl.int32)
        idx = (q + 32768) & 0xFFFF
        ok  = (idx < 65535) & mask
        hx  = tl.where(ok, hx, 0.0)

        ct  = (idx >> 14) & 3
        idi = (idx >> 7)  & 0x7F
        idj =  idx        & 0x7F
        ci  = (ct * 128 + idi) * 4   # (BLOCK_N,) flat codebook row start
        cj  = (ct * 128 + idj) * 4

        ok4  = ok[:, None]                                    # (BLOCK_N, 4) broadcast mask
        ci_v = tl.load(codes_ptr + ci[:, None] + d4[None, :],
                       mask=ok4, other=0.0).to(tl.float32)   # (BLOCK_N, 4)
        cj_v = tl.load(codes_ptr + cj[:, None] + d4[None, :],
                       mask=ok4, other=0.0).to(tl.float32)

        pa_ci += tl.trans(ci_v) * hx[None, :]  # (4, BLOCK_N)
        pa_cj += tl.trans(cj_v) * hx[None, :]

    # ── 2 reductions: (4, BLOCK_N) → (4,) ────────────────────────────────
    a_ci = tl.sum(pa_ci, axis=1)
    a_cj = tl.sum(pa_cj, axis=1)

    # ── parallel sparse correction ─────────────────────────────────────────
    # Process BLOCK_N correction entries at a time — same (4, BLOCK_N) pattern
    # as the main N-loop.  For ~1% masking and typical N, a single iteration
    # covers all entries (e.g. N=4096 → ~41 entries << BLOCK_N=256).
    csr_start = tl.load(O_row_ptr + m).to(tl.int32)
    csr_end   = tl.load(O_row_ptr + m + 1).to(tl.int32)

    for crit_base in range(csr_start, csr_end, BLOCK_N):
        crit_offs = crit_base + tl.arange(0, BLOCK_N)
        crit_mask = crit_offs < csr_end

        cols  = tl.load(O_col_ptr + crit_offs,
                        mask=crit_mask, other=0).to(tl.int32)
        hx_c  = tl.load(xh_ptr + xh_off + cols,
                        mask=crit_mask, other=0.0).to(tl.float32)   # (BLOCK_N,)

        ob    = crit_offs * 8
        ov_ci = tl.load(O_vals_ptr + ob[:, None] + d4[None, :],
                        mask=crit_mask[:, None], other=0.0).to(tl.float32)  # (BLOCK_N, 4)
        ov_cj = tl.load(O_vals_ptr + ob[:, None] + d4[None, :] + 4,
                        mask=crit_mask[:, None], other=0.0).to(tl.float32)  # (BLOCK_N, 4)

        a_ci = a_ci + tl.sum(tl.trans(ov_ci) * hx_c[None, :], axis=1)  # (4,)
        a_cj = a_cj + tl.sum(tl.trans(ov_cj) * hx_c[None, :], axis=1)

    # ── apply g scales, optional bias, write output ────────────────────────
    gbase  = m * 8
    res_ci = a_ci * tl.load(g_ptr + gbase + d4    ).to(tl.float32)
    res_cj = a_cj * tl.load(g_ptr + gbase + d4 + 4).to(tl.float32)

    if has_bias:
        res_ci = res_ci + tl.load(bias_ptr + gbase + d4    ).to(tl.float32)
        res_cj = res_cj + tl.load(bias_ptr + gbase + d4 + 4).to(tl.float32)

    out_ci = y_off + gbase + d4
    out_cj = y_off + gbase + d4 + 4
    tl.store(y_ptr + out_ci, res_ci.to(OUT_DTYPE))
    tl.store(y_ptr + out_cj, res_cj.to(OUT_DTYPE))


def e8d4_triton_forward(
    Q:         Tensor,
    codes:     Tensor,
    g:         Tensor,
    x_h:       Tensor,   # (B, N) fp32 — caller must precompute x * h
    O_vals:    Tensor,
    O_col:     Tensor,
    O_row_ptr: Tensor,
    bias:      Tensor,
    out_dtype: torch.dtype = torch.float16,
) -> Tensor:
    """E8D4 fused dequant-matvec via Triton (B < TRITON_BATCH_THRESHOLD).

    x_h must be precomputed by the caller as x * h (element-wise, same dtype as x).
    Returns (B, M*8) in out_dtype.
    """
    M, N     = Q.shape
    B        = x_h.shape[0]
    M8       = M * 8
    has_bias = bias.numel() > 0

    tl_out_dtype = tl.bfloat16 if out_dtype == torch.bfloat16 else tl.float16
    y = torch.empty(B, M8, dtype=out_dtype, device=x_h.device)

    _e8d4_matvec_kernel[(M, B)](
        Q, codes, g, x_h, y,
        O_vals, O_col, O_row_ptr, bias,
        M=M, N=N, M8=M8,
        stride_xh=x_h.stride(0),
        stride_y=M8,
        has_bias=has_bias,
        OUT_DTYPE=tl_out_dtype,
    )
    return y
