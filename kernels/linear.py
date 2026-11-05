"""EPTQLinear: quantized linear layer using fused CUDA inference kernels."""

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import e8_forward, e8d4_forward
from .triton_e8d4 import e8d4_triton_forward, TRITON_BATCH_THRESHOLD

# Set EPTQ_CUDA_DECODE=1 to force CUDA kernel for decode (B<TRITON_BATCH_THRESHOLD).
# Default uses Triton (better under torch.compile, native bfloat16).
_CUDA_DECODE = os.environ.get("EPTQ_CUDA_DECODE", "0") == "1"
# Set EPTQ_FORCE_TRITON=1 to force Triton path for all batch sizes (bypasses WMMA).
# Useful for prefill/PPL eval where B>>1 but WMMA correctness is not guaranteed.
_FORCE_TRITON = os.environ.get("EPTQ_FORCE_TRITON", "0") == "1"

class EPTQLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that stores weights in EPTQ quantized
    format and performs inference via a single fused CUDA kernel.

    The fused kernel handles dequantization, matvec, masked-weight correction,
    and fp32→fp16 conversion in one pass — no intermediate fp32 buffer.

    Memory layout:
      - Q:         (M, N) int16  where M = out_features // 8
      - g:         (out_features,) fp16  row scales
      - h:         (in_features,)  fp16  col scales
      - O_vals:    (n_masked, 8)   fp16  critical weight values
      - O_col:     (n_masked,)     int32 col index per masked position (sorted by row)
      - O_row_ptr: (M+1,)          int32 CSR row pointers into O_vals/O_col
      - codebook:  shared fp16 tensor (not duplicated across layers)
    """

    def __init__(
        self,
        Q: torch.Tensor,
        codebook: torch.Tensor,
        g: torch.Tensor,
        h: torch.Tensor,
        O_vals: torch.Tensor,
        O_col: torch.Tensor,
        O_row_ptr: torch.Tensor,
        bias: Optional[torch.Tensor],
        quantizer_type: str,
        materialized_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.in_features  = Q.shape[1]
        self.out_features = Q.shape[0] * 8
        self.quantizer_type = quantizer_type

        self.register_buffer("Q",         Q.contiguous())
        self.register_buffer("g",         g.contiguous())
        self.register_buffer("h",         h.contiguous())
        self.register_buffer("O_vals",    O_vals.contiguous())
        self.register_buffer("O_col",     O_col.contiguous())
        self.register_buffer("O_row_ptr", O_row_ptr.contiguous())

        if bias is not None:
            self.register_buffer("bias", bias.contiguous())
        else:
            self.register_buffer("bias", None)

        # Pre-materialized fp16 weight for group-scale mode (None in normal mode).
        if materialized_weight is not None:
            self.register_buffer("_materialized_weight", materialized_weight.contiguous())
        else:
            self.register_buffer("_materialized_weight", None)

        # Shared codebook — not a buffer to avoid copies in state_dict.
        self._codebook = codebook

    def _apply(self, fn):
        super()._apply(fn)
        self._codebook = fn(self._codebook)
        return self

    @classmethod
    def from_quantized_data(
        cls,
        quant_data,
        codebook: torch.Tensor,
        quantizer_type: str,
    ) -> "EPTQLinear":
        Q = quant_data.Q          # int16 (M, N)
        M = Q.shape[0]

        cb = codebook.to(dtype=torch.float16)
        if quantizer_type == "e8_d4":
            cb = cb.reshape(512, 4).contiguous()

        # Group-scale mode: CUDA kernels don't support per-(row-group, col) scales,
        # so we pre-materialize the fp16 weight matrix once using dequantize_layer.
        materialized_weight = None
        if quant_data.S_group is not None:
            from ..config import QuantizedLayerData
            from ..algorithms.eptq import dequantize_layer
            from ..quantizers import E8LatticeQuantizer, E8D4Quantizer

            dev = codebook.device
            if quantizer_type == "e8_d4":
                e8q_tmp = E8D4Quantizer(codebook=codebook.float().reshape(4, 128, 4))
            else:
                e8q_tmp = E8LatticeQuantizer(codebook=codebook.float())

            qd_gpu = QuantizedLayerData(
                Q=quant_data.Q.to(dev),
                g=quant_data.g.to(dev),
                h=quant_data.h.to(dev),
                O=quant_data.O.to(dev),
                bias=quant_data.bias.to(dev) if quant_data.bias is not None else None,
                chosen_r=quant_data.chosen_r,
                S_group=quant_data.S_group.to(dev),
            )
            materialized_weight = dequantize_layer(qd_gpu, e8q_tmp)  # fp16, on dev

        # Masked positions — nonzero() on row-major Q returns in row-major order
        mask = Q == 32767
        O_row, O_col = mask.nonzero(as_tuple=True)
        O_row = O_row.to(torch.int32)
        O_col = O_col.to(torch.int32)

        n_masked = O_row.shape[0]
        assert quant_data.O.shape[0] == n_masked, (
            f"O shape mismatch: got {quant_data.O.shape[0]} values "
            f"but found {n_masked} masked positions in Q"
        )

        # CSR row pointers: O_row_ptr[m]..O_row_ptr[m+1] = masked positions in row m
        counts = torch.bincount(O_row.long(), minlength=M)
        O_row_ptr = torch.cat([
            torch.zeros(1, dtype=torch.int32, device=O_row.device),
            counts.cumsum(0).to(torch.int32),
        ])

        return cls(
            Q=Q,
            codebook=cb,
            g=quant_data.g.to(dtype=torch.float16),
            h=quant_data.h.to(dtype=torch.float16),
            O_vals=quant_data.O.to(dtype=torch.float16),
            O_col=O_col,
            O_row_ptr=O_row_ptr,
            bias=quant_data.bias,
            quantizer_type=quantizer_type,
            materialized_weight=materialized_weight,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        output_dtype = x.dtype

        # Group-scale mode: use pre-materialized fp16 weight (no CUDA kernel)
        if self._materialized_weight is not None:
            return F.linear(
                x.to(self._materialized_weight.dtype),
                self._materialized_weight,
                self.bias,
            ).reshape(*input_shape[:-1], self.out_features)

        x_2d = x.reshape(-1, self.in_features)

        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()

        bias_t = self.bias if self.bias is not None else \
            torch.empty(0, dtype=torch.float16, device=x.device)

        if self.quantizer_type == "e8_lattice":
            if x_2d.dtype != torch.float16:
                x_2d = x_2d.half()
            y = e8_forward(
                self.Q, self._codebook,
                self.g, self.h, x_2d,
                self.O_vals, self.O_col, self.O_row_ptr,
                bias_t,
            )
            if output_dtype != torch.float16:
                y = y.to(dtype=output_dtype)
        elif not _CUDA_DECODE and (_FORCE_TRITON or x_2d.shape[0] < TRITON_BATCH_THRESHOLD):
            # Decode path: Triton kernel handles fp16/bfloat16 natively — no cast needed.
            # Precompute x_h = x * h here so torch.compile sees the op and can fuse it,
            # and the Triton kernel only needs one global load per column instead of two.
            # Keep as fp32 (no .to(x.dtype) rounding) — exactly matches the v4 kernel's
            # in-kernel fp32 multiply, preserving numerical equivalence with the CUDA reference.
            x_h = x_2d.float() * self.h.float()  # (B, N) fp32
            y = e8d4_triton_forward(
                self.Q, self._codebook,
                self.g, x_h,
                self.O_vals, self.O_col, self.O_row_ptr,
                bias_t,
                out_dtype=output_dtype,
            )
        else:
            # CUDA path: SMEM-optimized decode kernel (or large-batch WMMA).
            if x_2d.dtype != torch.float16:
                x_2d = x_2d.half()
            y = e8d4_forward(
                self.Q, self._codebook,
                self.g, self.h, x_2d,
                self.O_vals, self.O_col, self.O_row_ptr,
                bias_t,
            )
            if output_dtype != torch.float16:
                y = y.to(dtype=output_dtype)

        return y.reshape(*input_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"quantizer={self.quantizer_type}, "
            f"n_masked={self.O_col.shape[0]}"
        )
