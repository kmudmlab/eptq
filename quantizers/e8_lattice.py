"""E8 Lattice Quantizer implementation."""

from itertools import product
from typing import Optional
import torch

from .base import BaseQuantizer


class E8LatticeQuantizer(BaseQuantizer):
    """
    E8 Lattice-based quantizer.

    Uses the E8 lattice structure to quantize 8-dimensional vectors.
    The codebook consists of 65535 points from the first 6 shells of E8.
    """

    DEFAULT_RADIUS = 1.6

    def __init__(self, device: str = "cpu", codebook: Optional[torch.Tensor] = None,
                 radius: Optional[float] = None):
        """
        Initialize E8 Lattice Quantizer.

        Args:
            device: Device to store codebook on
            codebook: Optional pre-computed codebook tensor (already divided by radius)
            radius: Codebook divisor (default 1.6). Larger → finer grid, smaller range.
        """
        if codebook is not None:
            self.points = codebook
            self._raw_points = None  # not available from saved codebook
            self._radius = self.DEFAULT_RADIUS  # unknown; safe fallback
            return

        self.points = self._gen_e8_codebook(device=device)
        self._raw_points = self.points.clone()  # exact lattice points before any scaling
        self._radius = radius if radius is not None else self.DEFAULT_RADIUS
        self.points /= self._radius

    def set_radius(self, new_radius: float) -> None:
        """Rescale codebook so that the effective divisor is ``new_radius``.

        When raw points are available (fresh init), recomputes from raw via exact
        division — giving the same fp32 result as initialising with that radius
        directly. Falls back to in-place scaling for codebooks loaded from file.
        """
        if self._raw_points is not None:
            self.points = self._raw_points / new_radius
        else:
            self.points *= (self._radius / new_radius)
        self._radius = new_radius

    def _gen_e8_roots(self) -> torch.Tensor:
        """Generate E8 root system vectors."""
        indices = torch.combinations(torch.arange(8), 2)
        roots = torch.zeros((len(indices) * 4, 8))
        for i, (s1, s2) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
            roots[i::4].scatter_(1, indices[:, 0:1], s1)
            roots[i::4].scatter_(1, indices[:, 1:2], s2)

        base = torch.tensor(list(product([0.5, -0.5], repeat=7)))
        base = torch.cat([base, torch.zeros((base.shape[0], 1)) + 0.5], dim=1)
        base[:, -1] -= base.sum(dim=1) % 2
        roots = torch.cat([roots, base], dim=0)

        return roots

    def _gen_e8_codebook(self, device: str = "cpu") -> torch.Tensor:
        """Generate E8 codebook from shell expansion."""
        shells = [torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32, device=device)]
        e8_roots = self._gen_e8_roots().to(device)

        for i in range(6):
            curr_shell = shells[-1]
            cand = (curr_shell.unsqueeze(1) + e8_roots.unsqueeze(0)).view(-1, 8)
            next_shell = cand[cand.pow(2).sum(dim=1) == 2 * (i + 1)].unique(dim=0, sorted=False)
            shells.append(next_shell)

        codebook = torch.cat(shells, dim=0)[:65536]
        return codebook

    def _nn(self, p: torch.Tensor) -> torch.Tensor:
        """Find nearest neighbor in codebook."""
        dists = torch.cdist(p, self.points.float(), p=2.0)
        best_indices = torch.argmin(dists, dim=1)
        return best_indices.to(torch.int32)

    def quant(
        self, weight: torch.Tensor, scale: torch.Tensor, batchsize: int = 128
    ) -> torch.Tensor:
        """
        Quantize weight matrix.

        Args:
            weight: Weight matrix of shape (out_features, in_features)
            scale: Single scale value
            batchsize: Batch size for processing

        Returns:
            Quantized indices of shape (out_features//8, in_features)
        """
        indices = []
        W_reshaped = weight.T.reshape(-1, 8)
        for i in range(0, W_reshaped.shape[0], batchsize):
            w_vec_batch = W_reshaped[i : i + batchsize]
            best_indices = self._nn(w_vec_batch / scale)
            indices.append(best_indices)
        return torch.cat(indices, dim=0).reshape(weight.shape[1], -1).T

    def dequant(self, indices: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Dequantize indices.

        Args:
            indices: Quantized indices of shape (out_features//8, in_features)
            scale: Single scale value

        Returns:
            Dequantized weight matrix of shape (out_features, in_features)
        """
        return self.points[indices.T].reshape(indices.shape[1], -1).T * scale

    def remove_last(self) -> None:
        """Remove last codebook entry for masking purposes."""
        self.points = self.points[:-1]
        if self._raw_points is not None:
            self._raw_points = self._raw_points[:-1]
