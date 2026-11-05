"""Smoke test for GPTQQuantizer: round-trip error, index range, dtype."""
import torch

from eptq.quantizers import GPTQQuantizer


def test_config(bits: int, symmetric: bool, shape=(256, 512), seed=0):
    torch.manual_seed(seed)
    w = torch.randn(shape, dtype=torch.float32, device="cpu")

    q_obj = GPTQQuantizer(bits=bits, symmetric=symmetric)
    scale, zero = q_obj.find_params(w)

    q = q_obj.quant(w, scale, zero)
    w_hat = q_obj.dequant(q, scale, zero)

    # Round-trip error
    rel_err = (w - w_hat).pow(2).sum().sqrt() / w.pow(2).sum().sqrt()

    # Index sanity
    q_min, q_max = int(q.min()), int(q.max())

    label = f"bits={bits} {'sym' if symmetric else 'asym'}"
    print(
        f"  {label:15s} | "
        f"scale={float(scale):.5f} zero={float(zero):5.1f} | "
        f"q in [{q_min},{q_max}] ≤ {q_obj.max_q} | "
        f"rel_err={rel_err:.4f}"
    )

    # Assertions
    assert q.dtype == torch.int32
    assert 0 <= q_min and q_max <= q_obj.max_q, (
        f"indices out of range: [{q_min}, {q_max}] vs [0, {q_obj.max_q}]"
    )
    assert w_hat.shape == w.shape
    return float(rel_err)


print("=" * 70)
print("GPTQQuantizer smoke test (Gaussian weights 256×512)")
print("=" * 70)

for bits in (2, 3, 4, 8):
    for sym in (False, True):
        test_config(bits=bits, symmetric=sym)
    print()

print("-" * 70)
print("quant_dequant convenience (auto-finds params):")
torch.manual_seed(0)
w = torch.randn(128, 128)
for bits in (4, 8):
    q_obj = GPTQQuantizer(bits=bits, symmetric=False)
    w_hat = q_obj.quant_dequant(w)
    rel = (w - w_hat).pow(2).sum().sqrt() / w.pow(2).sum().sqrt()
    print(f"  bits={bits}  rel_err={rel:.4f}")

print("\nAll checks passed ✓")
