"""Synthetic test for gptq_quantize: reconstruction error and RTN comparison."""
import torch

from eptq.algorithms.gptq import gptq_quantize, dequantize_gptq_layer
from eptq.quantizers.gptq import GPTQQuantizer


def rel_err(w_true: torch.Tensor, w_hat: torch.Tensor) -> float:
    return float((w_true - w_hat).pow(2).sum().sqrt() / w_true.pow(2).sum().sqrt())


def proxy_loss(w_true, w_hat, H):
    """
    Proxy of the calibration loss  ||X @ (W - W_hat)||^2.
    With H = X.T @ X this is  tr((W-W_hat) H (W-W_hat).T).
    Much better metric than naive ||W-W_hat|| for GPTQ-style methods.
    """
    dw = (w_true - w_hat).float()
    return float((dw @ H.float() * dw).sum())


def rtn_quant_dequant(W, quantizer, group_size):
    """Naive round-to-nearest (per-group) — no Hessian correction."""
    n_out, n_in = W.shape
    gs = group_size if group_size > 0 else n_in
    W_hat = torch.zeros_like(W)
    for g_start in range(0, n_in, gs):
        g_end = min(g_start + gs, n_in)
        block = W[:, g_start:g_end]
        s, z = quantizer.find_params(block, per_channel=True)
        q = quantizer.quant(block, s, z)
        W_hat[:, g_start:g_end] = quantizer.dequant(q, s, z)
    return W_hat


def run_test(bits, symmetric, group_size, actorder, seed=0, out=256, inp=512, device="cpu"):
    torch.manual_seed(seed)
    W = torch.randn(out, inp, dtype=torch.float32, device=device)

    # Build a realistic positive-definite Hessian: H = X.T @ X with n_samples > in
    n_samples = inp * 2
    X = torch.randn(n_samples, inp, dtype=torch.float32, device=device)
    H = X.T @ X / n_samples

    quantizer = GPTQQuantizer(bits=bits, symmetric=symmetric, group_size=group_size)

    # --- RTN baseline (same grid, no Hessian correction) ---
    W_rtn = rtn_quant_dequant(W, quantizer, group_size)

    # --- GPTQ ---
    q_data = gptq_quantize(H, W, quantizer, percdamp=0.01, blocksize=128, actorder=actorder)
    W_gptq = dequantize_gptq_layer(q_data).float()

    rtn_w  = rel_err(W, W_rtn)
    gptq_w = rel_err(W, W_gptq)

    rtn_p  = proxy_loss(W, W_rtn,  H)
    gptq_p = proxy_loss(W, W_gptq, H)

    label = f"bits={bits} {'sym' if symmetric else 'asym'} gs={group_size} act={actorder}"
    print(
        f"  {label:35s} | "
        f"||W-RTN||/||W|| {rtn_w:.4f}  ||W-GPTQ||/||W|| {gptq_w:.4f} | "
        f"proxy: RTN {rtn_p:.2f}  GPTQ {gptq_p:.2f}  "
        f"({gptq_p / rtn_p * 100:5.1f}%)"
    )

    # Shape sanity
    assert q_data.Q.shape == (out, inp)
    n_groups = (inp + max(group_size, 1) - 1) // max(group_size, 1) if group_size > 0 else 1
    assert q_data.scales.shape == (out, n_groups)
    assert q_data.zeros.shape == (out, n_groups)

    # GPTQ should beat RTN on the CALIBRATION-WEIGHTED proxy loss
    # (which is what GPTQ actually optimizes).  Raw ||W-W_hat|| can go
    # up a little; that is fine.
    assert gptq_p <= rtn_p * 1.05, (
        f"GPTQ proxy loss ({gptq_p:.2f}) should not exceed RTN ({rtn_p:.2f}) "
        f"by more than 5%"
    )


print("=" * 110)
print("gptq_quantize synthetic test (256×512, Hessian = X.T @ X with X ~ N(0,I))")
print("=" * 110)

print("\n[per-column  (group_size=-1)]")
for bits in (4, 8):
    for sym in (False, True):
        run_test(bits=bits, symmetric=sym, group_size=-1, actorder=False)

print("\n[per-group   (group_size=128)]")
for bits in (4, 8):
    for sym in (False, True):
        run_test(bits=bits, symmetric=sym, group_size=128, actorder=False)

print("\n[actorder    (per-column only for now)]")
for bits in (4, 8):
    run_test(bits=bits, symmetric=False, group_size=-1, actorder=True)

print("\nAll assertions passed ✓")
