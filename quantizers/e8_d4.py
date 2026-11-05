"""E8-D4 Quantizer implementation."""

import heapq as hq
from typing import Optional
import torch

from .base import BaseQuantizer


class E8D4Quantizer(BaseQuantizer):
    """
    E8-D4 Quantizer.

    Uses D4 lattice structure to quantize 4-dimensional vectors,
    combining two D4 vectors to form 8-dimensional representations.
    """

    DEFAULT_RADIUS = 1.875

    def __init__(self, device: str = "cpu", codebook: Optional[torch.Tensor] = None,
                 radius: Optional[float] = None):
        """
        Initialize E8-D4 Quantizer.

        Args:
            device: Device to store codebook on
            codebook: Optional pre-computed codebook tensor (already divided by radius)
            radius: Codebook divisor (default 1.875).
        """
        if codebook is not None:
            self.codes = codebook
            self._raw_codes = None  # not available from saved codebook
            self._radius = self.DEFAULT_RADIUS  # unknown; safe fallback
            self._setup_flat_codes()
            return

        d4root_system = [
            (1, 0, 0, 0), (-1, 0, 0, 0), (0, 1, 0, 0), (0, -1, 0, 0),
            (0, 0, 1, 0), (0, 0, -1, 0), (0, 0, 0, 1), (0, 0, 0, -1),
            (0.5, 0.5, 0.5, 0.5), (-0.5, 0.5, 0.5, 0.5),
            (0.5, -0.5, 0.5, 0.5), (0.5, 0.5, -0.5, 0.5),
            (0.5, 0.5, 0.5, -0.5), (-0.5, -0.5, 0.5, 0.5),
            (-0.5, 0.5, -0.5, 0.5), (-0.5, 0.5, 0.5, -0.5),
            (0.5, -0.5, -0.5, 0.5), (0.5, -0.5, 0.5, -0.5),
            (0.5, 0.5, -0.5, -0.5), (-0.5, -0.5, -0.5, 0.5),
            (-0.5, -0.5, 0.5, -0.5), (-0.5, 0.5, -0.5, -0.5),
            (0.5, -0.5, -0.5, -0.5), (-0.5, -0.5, -0.5, -0.5),
        ]

        candidate = [(0, (0, 0, 0, 0))]  # (distance squared, vector)
        visited = set()
        visited.add((0, 0, 0, 0))
        d4_of = []  # odd full
        d4_oh = []  # odd half
        d4_ef = []  # even full
        d4_eh = []  # even half

        while True:
            dist, top = hq.heappop(candidate)
            if dist > 7:
                break
            if sum(top) % 2 == 0:
                if all((x * 2) % 2 == 0 for x in top):
                    d4_ef.append(top)
                else:
                    d4_eh.append(top)
            else:
                if all((x * 2) % 2 == 0 for x in top):
                    d4_of.append(top)
                else:
                    d4_oh.append(top)

            for root in d4root_system:
                new_vec = tuple(top[i] + root[i] for i in range(4))
                if new_vec not in visited:
                    visited.add(new_vec)
                    hq.heappush(candidate, (sum(x * x for x in new_vec), new_vec))

        self.codes = torch.tensor(
            [d4_ef[:128], d4_eh[:128], d4_of[:128], d4_oh[:128]],
            dtype=torch.float32,
        )
        self.codes = self.codes.to(device)
        self._raw_codes = self.codes.clone()  # exact lattice points before any scaling
        self._radius = radius if radius is not None else self.DEFAULT_RADIUS
        self.codes /= self._radius

        self._setup_flat_codes()

    def set_radius(self, new_radius: float) -> None:
        """Rescale codebook so that the effective divisor is ``new_radius``.

        When raw codes are available (fresh init), recomputes from raw via exact
        division — giving the same fp32 result as initialising with that radius
        directly. Falls back to in-place scaling for codebooks loaded from file.
        """
        if self._raw_codes is not None:
            self.codes = self._raw_codes / new_radius
        else:
            self.codes *= (self._radius / new_radius)
        self._radius = new_radius
        self._setup_flat_codes()

    def _setup_flat_codes(self) -> None:
        """Setup flattened codes for efficient distance computation."""
        self.flat_codes_T = self.codes.view(-1, 4).T
        self.code_norms_T = (self.flat_codes_T**2).sum(dim=-2, keepdim=True)

    def _dist_proxy(self, a: torch.Tensor) -> torch.Tensor:
        """Compute proxy distances for efficient nearest neighbor search."""
        d = a @ self.flat_codes_T
        d *= -2
        d += self.code_norms_T
        return d

    def _encode_chunk(self, dists: torch.Tensor) -> torch.Tensor:
        """Encode proxy distances to packed 16-bit indices (no 65535).

        Args:
            dists: (n, 512) proxy distances; n must be even.
        Returns:
            indices: (n//2,) packed codes.
        """
        min_vals, min_inds = dists.view(-1, 4, 128).min(dim=-1)   # (n, 4)

        best_types = min_vals.view(-1, 2, 4).sum(dim=1).argmin(dim=-1)  # (n//2,)
        chosen_indices = min_inds.gather(
            1, best_types.repeat_interleave(2).unsqueeze(1)
        ).squeeze(1)
        types_packed = best_types << 14
        pairs = chosen_indices.view(-1, 2)
        indices = types_packed + (pairs[:, 0] << 7) + pairs[:, 1]

        mask = indices == 65535
        if mask.any():
            self._replace_65535(indices, mask, min_vals, min_inds)

        return indices

    def _replace_65535(
        self,
        indices: torch.Tensor,
        mask: torch.Tensor,
        min_vals: torch.Tensor,
        min_inds: torch.Tensor,
    ) -> None:
        """Replace reserved 65535 entries with the nearest valid code.

        65535 = (type=3, i=127, j=127).  By D4 Voronoi geometry, the
        second-nearest code to any point whose nearest type=3 code is
        c[3,127] is always in type=0/1/2.  So we only need to check
        those three types using the already-computed min_vals/min_inds.
        """
        prob = mask.nonzero(as_tuple=True)[0]   # pair indices
        v1 = prob * 2
        v2 = v1 + 1

        # Best (i, j) scores for types 0, 1, 2: (k, 3)
        scores = min_vals[v1, :3] + min_vals[v2, :3]
        best_t = scores.argmin(dim=1)                                    # (k,)
        best_i = min_inds[v1, :3].gather(1, best_t.unsqueeze(1)).squeeze(1).long()
        best_j = min_inds[v2, :3].gather(1, best_t.unsqueeze(1)).squeeze(1).long()

        indices[mask] = (best_t.long() << 14) + (best_i << 7) + best_j

    def quant(
        self, weights: torch.Tensor, scale: torch.Tensor, batchsize: int = 2097152
    ) -> torch.Tensor:
        """
        Quantize weight matrix.

        Args:
            weights: Weight matrix of shape (out_features, in_features)
            scale: Single scale value
            batchsize: Number of 4D rows per chunk (must be multiple of 8).
                       Default 2M rows ≈ 4 GB intermediate memory.

        Returns:
            Quantized indices of shape (out_features//8, in_features)
        """
        W_reshaped = (weights.T / scale).reshape(-1, 4).float()
        n = W_reshaped.shape[0]

        # Chunk must be a multiple of 8 (two groups of 4D = one 8D block pair).
        chunk = batchsize - (batchsize % 8) if batchsize % 8 != 0 else batchsize

        if n <= chunk:
            indices = self._encode_chunk(self._dist_proxy(W_reshaped))
        else:
            # Chunked path for large layers (e.g. 70B MLP projections).
            parts = []
            for start in range(0, n, chunk):
                dists = self._dist_proxy(W_reshaped[start:start + chunk])
                parts.append(self._encode_chunk(dists))
            indices = torch.cat(parts)

        return indices.view(weights.shape[1], -1).T

    def quant_block(
        self,
        weights: torch.Tensor,
        scales: torch.Tensor,
        qb_size: int,
        batchsize: int = 65536,
    ) -> torch.Tensor:
        """
        Quantize weight matrix with per-block scales.

        Args:
            weights: Weight matrix of shape (out_features, in_features)
            scales: Scale matrix of shape (out_features//qb_size, in_features)
            qb_size: Quantization block size
            batchsize: Batch size for processing

        Returns:
            Quantized indices of shape (out_features//8, in_features)
        """
        w_view = weights.view(-1, qb_size, weights.shape[1])
        s_view = scales.view(-1, 1, scales.shape[1])
        weights_scaled = (w_view / s_view).flatten(0, 1)
        W_reshaped = weights_scaled.T.reshape(-1, 4)

        batchsize *= 2
        indices_list = []
        for i in range(0, W_reshaped.shape[0], batchsize):
            W_reshaped_batch = W_reshaped[i : i + batchsize]
            dists = self._dist_proxy(W_reshaped_batch.float())
            indices_list.append(self._encode_chunk(dists))

        if len(indices_list) == 1:
            indices = indices_list[0]
        else:
            indices = torch.cat(indices_list, dim=0)

        return indices.view(weights.shape[1], -1).T

    def dequant(self, indices: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Dequantize indices.

        Args:
            indices: Quantized indices of shape (out_features//8, in_features)
            scale: Single scale value

        Returns:
            Dequantized weight matrix of shape (out_features, in_features)
        """
        code_types = (indices >> 14) & 0x03
        indices_i = (indices >> 7) & 0x7F
        indices_j = indices & 0x7F

        a1 = self.codes[code_types.T, indices_i.T]
        a2 = self.codes[code_types.T, indices_j.T]

        return torch.stack((a1, a2), dim=-2).reshape(indices_j.shape[-1], -1).T * scale

    def dequant_block(
        self, indices: torch.Tensor, scales: torch.Tensor, qb_size: int
    ) -> torch.Tensor:
        """
        Dequantize indices with per-block scales.

        Args:
            indices: Quantized indices of shape (out_features//8, in_features)
            scales: Scale matrix of shape (out_features//qb_size, in_features)
            qb_size: Quantization block size

        Returns:
            Dequantized weight matrix of shape (out_features, in_features)
        """
        code_types = (indices >> 14) & 0x03
        indices_i = (indices >> 7) & 0x7F
        indices_j = indices & 0x7F

        a1 = self.codes[code_types.T, indices_i.T]
        a2 = self.codes[code_types.T, indices_j.T]

        weight_dq = torch.stack((a1, a2), dim=-2).reshape(indices_j.shape[-1], -1).T
        weight_dq = weight_dq * scales.repeat_interleave(qb_size, dim=0)
        return weight_dq

    def warp_codebook(self, alpha: float) -> None:
        """Apply radial warping to 4D codes.

        New 4D norm: (||c||_4D / max_norm_4D)^alpha * max_norm_4D
        Direction unchanged; the max-norm point stays fixed.
        alpha=1.0 is a no-op. alpha>1.0 pulls points toward the origin.
        """
        codes_flat = self.codes.view(-1, 4)           # (512, 4)
        norms = codes_flat.norm(dim=1)                 # (512,)
        max_norm = norms.max()
        r_norm = norms / max_norm.clamp(min=1e-8)      # [0, 1]
        new_norms = r_norm.pow(alpha) * max_norm
        scale = torch.where(norms > 1e-8, new_norms / norms, torch.zeros_like(norms))
        self.codes = (codes_flat * scale.unsqueeze(1)).view(4, 128, 4)
        self._warp_alpha = alpha
        self._warp_max_norm = max_norm.item()
        self._setup_flat_codes()

    def remove_last(self) -> None:
        """Remove last codebook entry for masking purposes (no-op for E8D4)."""
        pass
