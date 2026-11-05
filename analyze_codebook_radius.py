"""Diagnostic: measure per-layer weight distribution after EPTQ normalization.

For each Linear layer in a model:
  1. Compute g, h via comp_gh()  -> normalize W
  2. Per-column, S[j] = 95th-percentile of |W_norm[:, j]|
  3. Compute W_normalized = W_norm / S[j]   (this is what gets matched to the codebook)
  4. Report distribution stats across layers

If stats vary significantly across layers, a per-layer codebook radius
heuristic (Option 2) has potential. If they're stable, the global 1.6/1.875
constant is already near-optimal on average.
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from tqdm import tqdm

from eptq.algorithms.eptq import comp_gh


def stats_after_norm(W: torch.Tensor) -> dict:
    """Apply EPTQ-style normalization (g, h, S) and report distribution stats."""
    W = W.to(torch.float32)
    g, h = comp_gh(W)
    W_n = W / g[:, None] / h[None, :]

    S = W_n.abs().quantile(0.95, dim=0)              # (in,)  per-column
    W_n_n = W_n / S[None, :].clamp_min(1e-8)         # what reaches the codebook

    flat = W_n_n.flatten()
    abs_v = flat.abs()

    # torch.quantile has a hard cap (~16M elements). Use kthvalue on the abs tensor.
    n = abs_v.numel()
    abs_sorted = abs_v.sort().values
    return {
        "std":    flat.std().item(),
        "abs_max": abs_sorted[-1].item(),
        "p99":    abs_sorted[int(0.99 * (n - 1))].item(),
        "p999":   abs_sorted[int(0.999 * (n - 1))].item(),
        "kurt":   ((flat - flat.mean()) ** 4).mean().item() / (flat.var().item() ** 2 + 1e-12),
        "frac_gt_1.5": (abs_v > 1.5).float().mean().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--out_csv", type=str,
                        default="/home/hmpark/eptq_new/results/codebook_radius_diag.csv")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, use_safetensors=True)

    rows = []
    linear_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    # Exclude lm_head (not quantized in EPTQ flow)
    linear_layers = [(n, m) for n, m in linear_layers if "lm_head" not in n]

    for name, lin in tqdm(linear_layers, desc="Analyzing layers"):
        W = lin.weight.data.cuda()
        s = stats_after_norm(W)
        s["name"] = name
        s["out_features"] = lin.out_features
        s["in_features"]  = lin.in_features
        rows.append(s)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["name", "out_features", "in_features",
              "std", "abs_max", "p99", "p999", "kurt", "frac_gt_1.5"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(r[k], 6) if isinstance(r[k], float) else r[k]) for k in fields})
    print(f"Wrote {len(rows)} rows -> {out}")

    # Quick summary
    import statistics as st
    for key in ("std", "p99", "p999", "kurt", "frac_gt_1.5"):
        vals = [r[key] for r in rows]
        print(f"  {key:>14}: min={min(vals):.4f}  median={st.median(vals):.4f}  "
              f"max={max(vals):.4f}  ratio_max/min={max(vals)/max(min(vals),1e-12):.2f}")


if __name__ == "__main__":
    main()
