"""GPTQ-style uniform affine scalar quantizer."""

from typing import Tuple

import torch

from .base import BaseQuantizer


class GPTQQuantizer(BaseQuantizer):
    """
    GPTQ-style uniform affine scalar quantization.

    Each scalar weight is mapped independently to an integer in
    ``[0, 2**bits - 1]``:

        q = clamp(round(w / scale) + zero, 0, max_q)

    Dequantization:

        w_hat = (q - zero) * scale

    Unlike the lattice quantizers (E8, E8-D4) there is no codebook;
    the quantized representation is just the integer grid plus a
    per-group ``(scale, zero)`` pair.

    Parameters
    ----------
    bits : int, default 4
        Bits per weight. Common choices: 2, 3, 4, 8.
    symmetric : bool, default False
        If True, ``zero`` is fixed at ``(max_q + 1) / 2`` and not stored.
        Asymmetric (default) is what vanilla GPTQ uses.
    group_size : int, default -1
        Size of the axis-0 group sharing ``(scale, zero)``.
        ``-1`` → one group per column (the common GPTQ default).
    """

    def __init__(
        self,
        bits: int = 4,
        symmetric: bool = False,
        group_size: int = -1,
    ):
        if bits not in (2, 3, 4, 8):
            raise ValueError(f"Unsupported bits={bits}; choose from 2, 3, 4, 8")

        self.bits = bits
        self.symmetric = symmetric
        self.group_size = group_size
        self.max_q = (1 << bits) - 1   # e.g. 15 for int4

    # ------------------------------------------------------------------
    # (scale, zero) search
    # ------------------------------------------------------------------
    def find_params(
        self,
        w: torch.Tensor,
        per_channel: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute ``(scale, zero)`` minimising the naive round-to-nearest
        quant error over ``w``.

        Parameters
        ----------
        w : torch.Tensor
            Weight tensor. Shape depends on ``per_channel``:
              - ``per_channel=False``: any shape → scalar scale/zero.
              - ``per_channel=True``: 2-D ``(rows, cols)`` → one
                scale/zero per row, shape ``(rows, 1)``.  This is the
                standard GPTQ setting (rows = output channels).
        per_channel : bool
            If True, produce per-row params; otherwise scalar params.

        Returns
        -------
        scale : torch.Tensor (scalar or (rows, 1))
        zero  : torch.Tensor (same shape as scale)
        """
        zero_elem = torch.zeros((), device=w.device, dtype=w.dtype)

        if per_channel:
            assert w.dim() == 2, "per_channel requires 2-D input (rows, cols)"
            reduce_dim = (1,)
            keepdim = True
        else:
            reduce_dim = None
            keepdim = False

        if self.symmetric:
            if per_channel:
                max_abs = w.abs().amax(dim=reduce_dim, keepdim=keepdim).clamp(min=1e-8)
            else:
                max_abs = w.abs().max().clamp(min=1e-8)
            scale = max_abs / ((self.max_q - 1) / 2)
            zero = torch.full_like(scale, (self.max_q + 1) / 2)
        else:
            if per_channel:
                w_min = torch.minimum(
                    w.amin(dim=reduce_dim, keepdim=keepdim), zero_elem
                )
                w_max = torch.maximum(
                    w.amax(dim=reduce_dim, keepdim=keepdim), zero_elem
                )
            else:
                w_min = torch.minimum(w.min(), zero_elem)
                w_max = torch.maximum(w.max(), zero_elem)
            scale = ((w_max - w_min) / self.max_q).clamp(min=1e-8)
            zero = (-w_min / scale).round()
        return scale, zero

    # ------------------------------------------------------------------
    # quant / dequant with externally supplied (scale, zero)
    # ------------------------------------------------------------------
    def quant(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        zero: torch.Tensor = None,
        batchsize: int = 128,
    ) -> torch.Tensor:
        """
        Quantize ``weight`` with the given ``(scale, zero)``.

        Parameters
        ----------
        weight : torch.Tensor
            Any shape. Broadcasts against ``scale``/``zero``.
        scale, zero : torch.Tensor
            Scalars or broadcastable tensors. If ``zero`` is omitted it
            defaults to the symmetric value ``(max_q + 1) / 2``.
        batchsize : unused
            Kept for API compatibility with :class:`BaseQuantizer`.

        Returns
        -------
        q : torch.Tensor (int32, same shape as ``weight``)
        """
        if zero is None:
            zero = torch.full_like(scale, (self.max_q + 1) / 2)
        q = torch.round(weight / scale) + zero
        return q.clamp(0, self.max_q).to(torch.int32)

    def dequant(
        self,
        indices: torch.Tensor,
        scale: torch.Tensor,
        zero: torch.Tensor = None,
    ) -> torch.Tensor:
        """Inverse of :meth:`quant`."""
        if zero is None:
            zero = torch.full_like(scale, (self.max_q + 1) / 2)
        return (indices.to(scale.dtype) - zero) * scale

    # ------------------------------------------------------------------
    # Convenience: fake-quantize (quant → dequant) in one step
    # ------------------------------------------------------------------
    def quant_dequant(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor = None,
        zero: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Round-trip ``weight`` through the quant grid.

        If ``scale``/``zero`` are omitted they are found automatically
        via :meth:`find_params` on ``weight``.
        """
        if scale is None:
            scale, zero = self.find_params(weight)
        q = self.quant(weight, scale, zero)
        return self.dequant(q, scale, zero)

    def remove_last(self) -> None:
        """No-op (no codebook to truncate)."""
        pass
