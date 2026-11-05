#pragma once
#include <torch/extension.h>

// Fused dequant + matvec + masked correction + fp16 output for E8 lattice.
// Q:         (M, N) int16  — stored as actual_index - 32768, sentinel 32767 = masked
// codebook:  (<=65535, 8) fp16
// g:         (M*8,) fp16  — row scale factors
// h:         (N,)   fp16  — column scale factors
// x:         (B, N) fp16, contiguous
// O_vals:    (n_masked, 8) fp16 — masked weight values, sorted by row
// O_col:     (n_masked,)   int32 — col indices, sorted by row
// O_row_ptr: (M+1,)        int32 — CSR row pointers
// bias:      (M*8,) fp16, or empty tensor if no bias
// Returns:   (B, M*8) fp16
torch::Tensor e8_forward(
    torch::Tensor Q,
    torch::Tensor codebook,
    torch::Tensor g,
    torch::Tensor h,
    torch::Tensor x,
    torch::Tensor O_vals,
    torch::Tensor O_col,
    torch::Tensor O_row_ptr,
    torch::Tensor bias
);

// Fused dequant + matvec + masked correction + fp16 output for E8-D4.
// codes:     (512, 4) fp16 flat — 4 types × 128 codes, 4 dims each
// Index packing: type=(idx>>14)&3, idi=(idx>>7)&0x7F, idj=idx&0x7F
// All other args same as e8_forward.
torch::Tensor e8d4_forward(
    torch::Tensor Q,
    torch::Tensor codes,
    torch::Tensor g,
    torch::Tensor h,
    torch::Tensor x,
    torch::Tensor O_vals,
    torch::Tensor O_col,
    torch::Tensor O_row_ptr,
    torch::Tensor bias
);
