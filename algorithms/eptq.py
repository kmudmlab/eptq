"""EPTQ quantization algorithm."""

from typing import List, Optional, Tuple, Union
import sys
import torch

from ..config import QuantizedLayerData
from ..quantizers.base import BaseQuantizer


def comp_gh(W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute row and column scale factors using iterative normalization.

    This normalizes the weight matrix so that each row and column has
    approximately unit mean absolute value.

    Args:
        W: Weight matrix of shape (out_features, in_features)

    Returns:
        Tuple of (g, h) where:
            g: Row scale factors of shape (out_features,)
            h: Column scale factors of shape (in_features,)
    """
    Wabs = W.abs()

    g = torch.ones(Wabs.shape[0], device=Wabs.device)
    h = torch.ones(Wabs.shape[1], device=Wabs.device)

    for _ in range(10):
        Wabs_h = Wabs / h[None, :]
        g = Wabs_h.mean(dim=1)

        Wabs_g = Wabs / g[:, None]
        h = Wabs_g.mean(dim=0)

    return g, h


def _column_wise_quant(
    W: torch.Tensor,
    H_inv: torch.Tensor,
    e8q: BaseQuantizer,
    blocksize: int,
    mask: Optional[torch.Tensor] = None,
    mask_tau: Optional[float] = None,
    H_inv_diag: Optional[torch.Tensor] = None,
    qb_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Iterative column-wise quantization with error propagation.

    Mutates ``W`` in place via error propagation.

    Critical-weight handling — two modes (mutually exclusive):
    - ``mask`` (pre-computed): binary (out, in) tensor built before this call
      via an RTN pre-pass.  ``D_mask_2d`` is returned as None; the caller
      reconstructs it from ``D_mask``.
    - ``mask_tau`` (on-the-fly): after tentatively quantizing each 8D block,
      the Hessian-weighted error is compared against ``tau``.  Blocks that
      exceed the threshold are preserved at full precision without any RTN
      pre-pass.  ``D_mask_2d`` of shape ``(out//8, in)`` is returned.

    Without grouping (qb_size=None): S is 1-D (in_features,), folded into h
    by the caller.  With grouping: S_group is 2-D (out//qb_size, in).
    """
    device = W.device
    out_dim, in_dim = W.shape
    Q = torch.zeros((out_dim // 8, in_dim), dtype=torch.int32, device=device)
    W_q = torch.zeros_like(W)

    if qb_size is not None:
        assert out_dim % qb_size == 0, f"out_dim ({out_dim}) must be divisible by qb_size ({qb_size})"
        assert qb_size % 8 == 0, f"qb_size ({qb_size}) must be divisible by 8"
        n_groups = out_dim // qb_size
        S_group = torch.zeros(n_groups, in_dim, device=device, dtype=torch.float32)
    else:
        S = torch.zeros(in_dim, device=device, dtype=torch.float32)

    D_mask_2d: Optional[torch.Tensor] = None
    if mask_tau is not None:
        D_mask_2d = torch.zeros(out_dim // 8, in_dim, dtype=torch.bool, device=device)

    for i in range(0, in_dim, blocksize):
        i_end = min(i + blocksize, in_dim)
        W1 = W[:, i:i_end]
        E1 = torch.zeros_like(W1)
        Q1 = Q[:, i:i_end]
        H_inv1 = H_inv[i:i_end, i:i_end]

        for j in range(i_end - i):
            if qb_size is None:
                scale = W1[:, j].abs().quantile(0.95)
                S[i + j] = scale
                Q1[:, j : j + 1] = e8q.quant(W1[:, j : j + 1], scale)
                wq = e8q.dequant(Q1[:, j : j + 1], scale)
            else:
                # Vectorized: normalize each group by its scale, then one quant call.
                # quant(W/s, 1.0) == quant(W, s) because only W/scale enters the NN search.
                col = W1[:, j]  # (out_dim,)
                scale_g = col.reshape(n_groups, qb_size).abs().quantile(0.95, dim=1).clamp_min(1e-8)
                S_group[:, i + j] = scale_g
                scale_exp = scale_g.repeat_interleave(qb_size)  # (out_dim,)
                col_norm = (col / scale_exp).unsqueeze(1)        # (out_dim, 1)
                Q1[:, j : j + 1] = e8q.quant(col_norm, 1.0)
                wq = e8q.dequant(Q1[:, j : j + 1], 1.0) * scale_exp.unsqueeze(1)

            if mask_tau is not None:
                # On-the-fly: compare per-8D-block EPTQ error against threshold.
                # Uses the error-propagated W (W1[:, j]) for a more accurate signal
                # than an RTN pre-pass on the original W.
                err_j = ((W1[:, j] - wq[:, 0]) ** 2) / (H_inv_diag[i + j] + 1e-8)
                D_j = err_j.reshape(-1, 8).sum(dim=1)   # (out_dim // 8,)
                is_crit = D_j > mask_tau
                D_mask_2d[:, i + j] = is_crit
                crit_expand = is_crit.repeat_interleave(8).unsqueeze(1)  # (out_dim, 1)
                wq = wq + crit_expand.float() * (W1[:, j : j + 1] - wq)
            elif mask is not None:
                wq = wq + mask[:, i + j : i + j + 1] * (W1[:, j : j + 1] - wq)

            W_q[:, i + j : i + j + 1] = wq

            d = H_inv1[j, j]
            E1[:, j : j + 1] = (W1[:, j : j + 1] - W_q[:, i + j : i + j + 1]) / d
            W1[:, j:] -= torch.outer(E1[:, j], H_inv1[j, j:])

        W[:, i_end:] -= E1[:, :i_end] @ H_inv[i:i_end, i_end:]

    if qb_size is not None:
        return Q, S_group, W_q, D_mask_2d
    else:
        return Q, S, W_q, D_mask_2d


def _proxy_loss(
    W_ref: torch.Tensor,
    W_q: torch.Tensor,
    H_norm: torch.Tensor,
    g: torch.Tensor,
) -> float:
    """Hessian-weighted output reconstruction loss in normalized space.

    Equals ``(1/N) * ‖X (W - Ŵ)^T‖_F^2`` after substituting
    ``W = g · W_norm · h``, ``H_norm = h H h``:

        tr(R H R^T) = tr(g² · R_norm · H_norm · R_norm^T)
                    = Σ_i g_i² · (R_norm[i,:] H_norm R_norm[i,:]^T)

    so the per-row loss must be weighted by ``g²``.
    """
    R = W_ref - W_q
    per_row = (R * (R @ H_norm)).sum(dim=1)
    return ((g ** 2) * per_row).sum().item()


def eptq_quantize(
    H_orig: torch.Tensor,
    W_orig: torch.Tensor,
    e8q: BaseQuantizer,
    percdamp: float = 0.01,
    percmask: float = 0.01,
    mask_tau: Optional[float] = None,
    auto_mask: bool = False,
    auto_mask_search_pct: float = 0.10,
    blocksize: int = 128,
    weight_scale_norm: bool = True,
    radius_candidates: Optional[List[float]] = None,
    layer_name: str = "",
    qb_size: Optional[int] = None,
) -> QuantizedLayerData:
    """
    EPTQ quantization algorithm.

    Quantizes a weight matrix using E8 lattice quantization with
    critical weight masking to preserve the most important weights.
    Uses column-wise quantization with a single scale per column (default) or
    per-(row-group, column) scales when ``qb_size`` is set.

    Args:
        H_orig: Hessian matrix of shape (in_features, in_features)
        W_orig: Weight matrix of shape (out_features, in_features)
        e8q: E8 lattice quantizer instance
        percdamp: Damping percentage for Hessian
        percmask: Fraction of 8D blocks to preserve as critical (RTN pre-pass, top-k).
            Used when mask_tau is None and auto_mask is False.
        mask_tau: RTN pre-pass threshold.  Blocks with D_sum > tau are marked
            critical.  Variable count per layer.  Overrides percmask.
        auto_mask: Per-layer knee detection.  Finds the largest log-gap in the
            sorted D_sum distribution and uses it as the per-layer threshold.
            Overrides both percmask and mask_tau.
        blocksize: Block size for iterative quantization
        weight_scale_norm: Whether to apply row/column normalization
        qb_size: Row group size for group quantization (None = per-column, no grouping)

    Returns:
        QuantizedLayerData containing quantized representation
    """
    device = W_orig.device

    H = H_orig.detach().clone().to(device).float()
    W = W_orig.detach().clone().to(device).float()

    # Compute and apply weight scale normalization
    if weight_scale_norm:
        g, h = comp_gh(W)
        W /= g[:, None]
        W /= h[None, :]
        H *= h[:, None]
        H *= h[None, :]
    else:
        g = torch.ones(W.shape[0], device=device)
        h = torch.ones(W.shape[1], device=device)

    # Handle dead weights
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    # Snapshot h-scaled (un-damped, post-dead-handling) Hessian for proxy loss
    H_norm = H.clone()

    # Compute inverse Hessian with damping
    while True:
        try:
            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(H.shape[1], device=H.device)
            H[diag, diag] += damp
            H = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(H)
            H_inv_diag = torch.diag(H_inv)
            H_inv = torch.linalg.cholesky(H_inv, upper=True).to(device)
        except torch._C._LinAlgError:
            print(f"Cholesky failed, doubling damping percentage ({percdamp}x2)...")
            percdamp *= 2
            H = H_orig.detach().clone().to(device).float()
            if weight_scale_norm:
                H *= h[:, None]
                H *= h[None, :]
            H[dead, dead] = 1
        else:
            break

    default_radius = e8q._radius
    candidates = (
        list(radius_candidates) if radius_candidates else [default_radius]
    )

    # ------------------------------------------------------------------
    # RTN pre-pass: compute importance scores for all 8D blocks and build
    # a binary mask (top-k for percmask, threshold for mask_tau).
    # ------------------------------------------------------------------
    S_search = W.abs().quantile(0.95, dim=0).clamp_min(1e-8)
    W_scaled = W / S_search[None, :]
    one = torch.tensor(1.0, device=device, dtype=torch.float32)
    n_blocks = (W.shape[0] // 8) * W.shape[1]
    k_critical = int(n_blocks * percmask)

    Q_a = e8q.quant(W_scaled, one)
    W_q_a = e8q.dequant(Q_a, one) * S_search[None, :]
    err = ((W - W_q_a) ** 2) / (H_inv_diag + 1e-8)
    D_sum = err.T.reshape(-1, 8).sum(dim=1)

    if auto_mask:
        # Kneedle on top-X% of D_sum (descending order).
        # The curve drops steeply then flattens → it lies BELOW the diagonal
        # y = 1-x.  The knee maximises (1-x) - y (not y - (1-x)).
        D_sorted, _ = D_sum.sort(descending=True)
        n_search = max(2, int(n_blocks * auto_mask_search_pct))
        D_top = D_sorted[:n_search]
        x = torch.linspace(0.0, 1.0, n_search, device=D_top.device)
        y = torch.log(D_top.clamp_min(1e-10))
        y = (y - y[-1]) / (y[0] - y[-1] + 1e-10)  # normalize to [0, 1]
        distance = (1.0 - x) - y  # positive where curve is below diagonal
        knee_idx = int(distance.argmax().item())
        threshold = D_top[knee_idx].item()
        D_mask = D_sum > threshold
        k_actual = int(D_mask.sum().item())
        if layer_name:
            print(
                f"[auto-mask] {layer_name}: "
                f"min={D_sorted[-1]:.4f} p50={D_sorted[n_blocks // 2]:.4f} "
                f"p95={D_sorted[int(0.05 * n_blocks)]:.4f} max={D_sorted[0]:.4f} "
                f"knee_threshold={threshold:.4f} k={k_actual}/{n_blocks}",
                file=sys.stderr,
            )
    elif mask_tau is not None:
        D_mask = D_sum > mask_tau
        k_actual = int(D_mask.sum().item())
        if layer_name:
            print(
                f"[tau-scan] {layer_name}: "
                f"min={D_sum.min():.4f} p50={D_sum.sort().values[n_blocks // 2]:.4f} "
                f"p95={D_sum.sort().values[int(0.95 * n_blocks)]:.4f} max={D_sum.max():.4f} "
                f"tau={mask_tau:.4f} k={k_actual}/{n_blocks}",
                file=sys.stderr,
            )
    else:
        if k_critical > 0:
            threshold = torch.topk(D_sum, k_critical).values.min()
        else:
            threshold = D_sum.max()
        D_mask = D_sum > threshold
        if layer_name:
            print(
                f"[tau-scan] {layer_name}: "
                f"min={D_sum.min():.4f} p50={D_sum.median():.4f} "
                f"p95={D_sum.quantile(0.95):.4f} max={D_sum.max():.4f} "
                f"threshold={threshold:.4f} k={k_critical}/{n_blocks}",
                file=sys.stderr,
            )

    if len(candidates) == 1:
        # ----------------------------------------------------------------
        # Single candidate: fast path — no cloning, no extra EPTQ pass
        # ----------------------------------------------------------------
        r = candidates[0]
        e8q.set_radius(r)

        mask = D_mask.float().repeat_interleave(8).reshape(W.shape[1], -1).T
        Q, S, W_q, _ = _column_wise_quant(W, H_inv, e8q, blocksize, mask=mask, qb_size=qb_size)
        D_mask_2d = D_mask.reshape(Q.shape[1], Q.shape[0]).T

        chosen_r = r

    else:
        # ----------------------------------------------------------------
        # Multiple candidates: run full EPTQ for each candidate and select
        # by actual Hessian-weighted reconstruction loss (not RTN proxy).
        #
        # For each r:
        #   1. Use pre-computed RTN mask
        #   2. Clone W and run full _column_wise_quant (error propagation)
        #   3. Compute actual tr(R H_norm R^T) loss on the resulting W_q
        # Keep the best-loss result directly (no re-run needed).
        # ----------------------------------------------------------------
        best_r, best_loss = None, float("inf")
        best_Q = best_S = best_W_q = best_D_mask_2d = None

        mask = D_mask.float().repeat_interleave(8).reshape(W.shape[1], -1).T

        for r in candidates:
            e8q.set_radius(r)

            W_work = W.clone()
            Q_r, S_r, W_q_r, _ = _column_wise_quant(
                W_work, H_inv, e8q, blocksize, mask=mask, qb_size=qb_size,
            )
            D_mask_2d_r = D_mask.reshape(Q_r.shape[1], Q_r.shape[0]).T
            del W_work

            loss = _proxy_loss(W, W_q_r, H_norm, g)

            if loss < best_loss:
                best_loss = loss
                best_r = r
                best_Q, best_S, best_W_q = Q_r, S_r, W_q_r
                best_D_mask_2d = D_mask_2d_r

        chosen_r = best_r
        Q, S, W_q = best_Q, best_S, best_W_q
        D_mask_2d = best_D_mask_2d
        print(
            f"[eptq] {layer_name}: picked r={chosen_r:.4f} "
            f"(loss={best_loss:.6g}, candidates={candidates})",
            file=sys.stderr,
        )

    e8q.set_radius(default_radius)

    # S is either 1-D (no grouping) or 2-D S_group (with grouping)
    S_group_out = S if qb_size is not None else None
    S_flat = None if qb_size is not None else S

    # Extract critical weights before masking indices.
    # D_mask_2d shape: (out//8, in) — consistent between RTN and on-the-fly paths.
    W_view = W_q.view(-1, 8, W_q.shape[1])
    true_indices = D_mask_2d.nonzero(as_tuple=True)
    O = W_view[true_indices[0], :, true_indices[1]]

    if qb_size is None:
        # No grouping: divide O by S and fold S into h.
        # O shape: (num_masked, 8), true_indices[1] are column indices.
        S_masked = S_flat[true_indices[1]]  # shape: (num_masked,)
        O = O / S_masked[:, None]           # normalize O into "S-absorbed-in-h" space
        h *= S_flat
    # With grouping: O is already in g/h-normalized space (S_group NOT folded into h).
    # h stays unchanged; S_group is stored separately.

    # Mark masked positions in Q with special index
    Q[true_indices[0], true_indices[1]] = 65535

    # Convert Q to int16 (subtract 32768 to fit in signed int16 range)
    Q_int16 = (Q - 32768).to(torch.int16)

    return QuantizedLayerData(
        Q=Q_int16,
        g=g.half(),
        h=h.half(),
        O=O.half(),
        bias=None,  # Will be set by caller if needed
        chosen_r=chosen_r,
        S_group=S_group_out.half() if S_group_out is not None else None,
    )


def dequantize_layer(
    quant_data: QuantizedLayerData,
    e8q: BaseQuantizer,
) -> torch.Tensor:
    """
    Dequantize a layer from its quantized representation.

    Args:
        quant_data: Quantized layer data
        e8q: E8 lattice quantizer instance

    Returns:
        Dequantized weight matrix of shape (out_features, in_features)
    """
    # Convert Q back to int32 range
    Q = quant_data.Q.int() + 32768

    # Identify masked positions
    mask = Q == 65535

    # Set masked positions to 0 temporarily for dequantization
    Q[mask] = 0

    # Temporarily set the quantizer to the radius used during this layer's quantization.
    # This avoids encoding any radius factor into h/O and the fp16 precision loss
    # that would come from that approach.
    saved_r = e8q._radius
    if quant_data.chosen_r is not None and quant_data.chosen_r != saved_r:
        e8q.set_radius(quant_data.chosen_r)

    # Dequantize column-wise
    out_features = Q.shape[0] * 8
    W = torch.zeros((out_features, Q.shape[1]), dtype=torch.float32, device=Q.device)

    if quant_data.S_group is not None:
        # Group quantization: dequant with scale=1.0 then multiply by group scales.
        # dequant(Q_norm, 1.0) * scale_exp == dequant(Q, scale_g) per group.
        S_group = quant_data.S_group.float().to(Q.device)  # (n_groups, in_features)
        n_groups = S_group.shape[0]
        qb_size = out_features // n_groups
        for j in range(Q.shape[1]):
            raw = e8q.dequant(Q[:, j:j+1], 1.0)                      # (out_features, 1)
            scale_exp = S_group[:, j].repeat_interleave(qb_size)      # (out_features,)
            W[:, j:j+1] = raw * scale_exp.unsqueeze(1)
    else:
        # No grouping: S was folded into h, so use scale=1.0
        for j in range(Q.shape[1]):
            W[:, j:j+1] = e8q.dequant(Q[:, j:j+1], 1.0)

    # Restore the original radius
    if quant_data.chosen_r is not None and quant_data.chosen_r != saved_r:
        e8q.set_radius(saved_r)

    # Restore critical weights at masked positions
    # Find positions marked with 65535 in Q and restore from O sequentially
    W_view = W.view(-1, 8, W.shape[1])
    if quant_data.O.numel() > 0:
        # Find masked positions (where Q was 65535) - vectorized
        idx0, idx1 = mask.nonzero(as_tuple=True)
        # Restore critical weights from O in order (vectorized operation)
        W_view[idx0, :, idx1] = quant_data.O.float().to(W.device)

    # Apply scale factors
    W = W * quant_data.g.float().to(W.device)[:, None] * quant_data.h.float().to(W.device)[None, :]

    return W.half()
