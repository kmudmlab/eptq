"""GPTQ quantization algorithm.

Scalar (integer) counterpart of :func:`eptq_quantize`.  The Hessian
inversion and column-wise error-propagation loop are identical; only
the quantizer and storage layout change.

References
----------
Frantar et al., "GPTQ: Accurate Post-Training Quantization for
Generative Pre-trained Transformers", ICLR 2023.
"""

import math
from typing import Optional

import torch

from ..config import GPTQLayerData
from ..quantizers.gptq import GPTQQuantizer


def gptq_quantize(
    H_orig: torch.Tensor,
    W_orig: torch.Tensor,
    quantizer: GPTQQuantizer,
    percdamp: float = 0.01,
    blocksize: int = 128,
    actorder: bool = False,
) -> GPTQLayerData:
    """
    Run GPTQ on a single linear layer.

    Parameters
    ----------
    H_orig : torch.Tensor
        Calibration Hessian ``(in_features, in_features)``.
    W_orig : torch.Tensor
        Weight matrix ``(out_features, in_features)``.
    quantizer : GPTQQuantizer
        Scalar quantizer describing ``bits``, ``symmetric``, ``group_size``.
        Here ``group_size`` groups **input channels** (columns of ``W``).
    percdamp : float
        Percentage of mean(diag(H)) added to diagonal before Cholesky.
    blocksize : int
        Block size for the outer loop (only affects how often error is
        propagated to remaining columns; does not affect accuracy).
    actorder : bool
        If True, reorder columns by descending ``diag(H)`` before
        quantizing (AutoGPTQ ``act_order``).  Improves accuracy at
        essentially no cost.

    Returns
    -------
    GPTQLayerData
    """
    device = W_orig.device
    W = W_orig.detach().clone().to(device).float()
    H = H_orig.detach().clone().to(device).float()

    n_out, n_in = W.shape
    gs = quantizer.group_size if quantizer.group_size > 0 else n_in
    n_groups = math.ceil(n_in / gs)

    # --- 1. Handle dead input channels (zero activation) ---
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0

    # --- 2. Act-order: reorder columns by importance (diag(H)) ---
    if actorder:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
        inv_perm = torch.argsort(perm)
    else:
        perm = None
        inv_perm = None

    # --- 3. Damped inverse Hessian → upper-triangular Cholesky factor ---
    #     We need H_inv_upper such that H_inv_upper.T @ H_inv_upper = H_inv.
    #     The diagonal of H_inv_upper gives the "update denominator"
    #     d_j = H_inv_upper[j, j] for the error-propagation step.
    while True:
        try:
            damp = percdamp * torch.mean(torch.diag(H))
            diag_idx = torch.arange(n_in, device=device)
            H_d = H.clone()
            H_d[diag_idx, diag_idx] += damp
            L = torch.linalg.cholesky(H_d)
            H_inv = torch.cholesky_inverse(L)
            H_inv_upper = torch.linalg.cholesky(H_inv, upper=True)
        except torch._C._LinAlgError:
            print(f"[gptq] Cholesky failed, doubling damping "
                  f"({percdamp} → {percdamp * 2})")
            percdamp *= 2
            continue
        break
    Hinv = H_inv_upper  # shorter alias

    # --- 4. Storage ---
    Q = torch.zeros((n_out, n_in), dtype=torch.int32, device=device)
    scales = torch.zeros((n_out, n_groups), dtype=torch.float32, device=device)
    zeros  = torch.zeros((n_out, n_groups), dtype=torch.float32, device=device)

    # Current group params (updated as we cross group boundaries)
    cur_scale: Optional[torch.Tensor] = None
    cur_zero:  Optional[torch.Tensor] = None

    # --- 5. Column-wise quantization with block-wise error propagation ---
    for i1 in range(0, n_in, blocksize):
        i2 = min(i1 + blocksize, n_in)
        count = i2 - i1

        W1     = W[:, i1:i2].clone()
        Q1     = torch.zeros((n_out, count), dtype=torch.int32, device=device)
        Err1   = torch.zeros_like(W1)
        Hinv1  = Hinv[i1:i2, i1:i2]

        for j in range(count):
            col_idx = i1 + j
            w = W1[:, j]                 # (out,)

            # --- At each group boundary, re-fit per-row (scale, zero) ---
            # Reference GPTQ (Frantar et al.) uses the master ``W`` here,
            # which holds values updated only by PREVIOUS blocks'
            # error-propagation.  This gives a stable, pre-block view
            # of the next ``group_size`` columns.
            if col_idx % gs == 0:
                g_end = min(col_idx + gs, n_in)
                w_block = W[:, col_idx:g_end]
                s, z = quantizer.find_params(w_block, per_channel=True)
                cur_scale = s.squeeze(1)   # (out,)
                cur_zero  = z.squeeze(1)   # (out,)
                g_idx = col_idx // gs
                scales[:, g_idx] = cur_scale
                zeros[:, g_idx]  = cur_zero

            # --- Scalar quantize + dequantize this column ---
            q = quantizer.quant(w, cur_scale, cur_zero)          # (out,)
            wq = quantizer.dequant(q, cur_scale, cur_zero)        # (out,)

            # --- Error propagation (identical to GPTQ/EPTQ) ---
            d = Hinv1[j, j]
            err = (w - wq) / d
            Err1[:, j] = err
            Q1[:, j] = q
            if j + 1 < count:
                W1[:, j + 1:] -= torch.outer(err, Hinv1[j, j + 1:])

        Q[:, i1:i2] = Q1

        # Propagate block error to remaining columns outside this block
        if i2 < n_in:
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    # --- 6. Undo column reordering if actorder was used ---
    if perm is not None:
        Q = Q[:, inv_perm]
        # scales/zeros indexed by group; n_groups may not line up with inv_perm
        # when actorder is used (groups are in reordered column space).
        # For now we disallow actorder + group_size != -1.
        if quantizer.group_size > 0:
            raise NotImplementedError(
                "actorder with group_size > 0 requires per-column scale "
                "storage; not yet implemented."
            )

    # --- 7. Fit Q into the smallest reasonable integer dtype ---
    if quantizer.bits <= 8:
        Q = Q.to(torch.uint8)

    return GPTQLayerData(
        Q=Q,
        scales=scales.half(),
        zeros=zeros.half(),
        bits=quantizer.bits,
        group_size=quantizer.group_size,
        bias=None,
    )


def dequantize_gptq_layer(quant_data: GPTQLayerData) -> torch.Tensor:
    """
    Reconstruct ``(out_features, in_features)`` fp16 weight from GPTQ data.
    """
    Q = quant_data.Q.to(torch.float32)
    scales = quant_data.scales.to(torch.float32)
    zeros  = quant_data.zeros.to(torch.float32)
    gs = quant_data.group_size

    n_out, n_in = Q.shape

    if gs <= 0:
        # per-column (single group per output channel)
        W = (Q - zeros[:, 0:1]) * scales[:, 0:1]
    else:
        # broadcast-expand per-group (scale, zero) to per-column
        # group index of column j is j // gs
        group_ids = torch.arange(n_in, device=Q.device) // gs
        s = scales[:, group_ids]   # (out, in)
        z = zeros[:,  group_ids]   # (out, in)
        W = (Q - z) * s
    return W.half()
