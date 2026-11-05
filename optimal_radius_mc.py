"""Monte Carlo: find MSE-optimal codebook radius for E8 and E8-D4
under N(0, σ²) per-dim 8D Gaussian source.

Source σ defaults to 0.51 (matches measured EPTQ-normalized weight std on Llama family).
"""

import argparse
import csv
from pathlib import Path

import torch

from eptq.quantizers import E8LatticeQuantizer, E8D4Quantizer


def mse_for_radius(quant_cls, radius: float, X: torch.Tensor, device: str) -> float:
    """Quantize X (N, 8) with a freshly built quantizer at given radius. Returns per-element MSE."""
    q = quant_cls(device=device, radius=radius)
    # API: weight shape (out, in), out must be divisible by 8.
    # Trick: make N the "in" dimension, 8 the "out".
    W = X.T.contiguous()                                 # (8, N)
    scale = torch.tensor(1.0, device=device)
    indices = q.quant(W, scale)
    Q = q.dequant(indices, scale)                        # (8, N)
    return ((X - Q.T) ** 2).mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", type=float, default=0.51)
    parser.add_argument("--n_samples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--r_min", type=float, default=0.8)
    parser.add_argument("--r_max", type=float, default=2.5)
    parser.add_argument("--r_step", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_csv", type=str,
                        default="/home/hmpark/eptq_new/results/optimal_radius_mc.csv")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    X = (torch.randn(args.n_samples, 8) * args.sigma).to(args.device)

    radii = torch.arange(args.r_min, args.r_max + 1e-9, args.r_step).tolist()
    rows = []
    print(f"Sigma = {args.sigma}, N = {args.n_samples}")
    print(f"{'radius':>8} {'mse_e8':>14} {'mse_d4':>14}")
    for r in radii:
        m_e8 = mse_for_radius(E8LatticeQuantizer, r, X, args.device)
        m_d4 = mse_for_radius(E8D4Quantizer, r, X, args.device)
        rows.append({"radius": round(r, 4), "mse_e8": m_e8, "mse_d4": m_d4})
        print(f"{r:>8.3f} {m_e8:>14.7f} {m_d4:>14.7f}")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["radius", "mse_e8", "mse_d4"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Find optima
    best_e8 = min(rows, key=lambda r: r["mse_e8"])
    best_d4 = min(rows, key=lambda r: r["mse_d4"])
    print()
    print(f"Optimal E8 radius:    {best_e8['radius']:.3f}  (MSE = {best_e8['mse_e8']:.7f})")
    print(f"Optimal E8-D4 radius: {best_d4['radius']:.3f}  (MSE = {best_d4['mse_d4']:.7f})")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
