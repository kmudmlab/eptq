"""
Diagnostic script: collect RTN D_sum distributions per sublayer for a model,
then plot sorted D_sum curves with knee-point markers to visualize auto_mask detection.
Compares with vs. without weight_scale_norm (g/h normalization).
"""
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from eptq.algorithms.eptq import comp_gh
from eptq.quantizers import E8D4Quantizer
from eptq.models import LlamaHandler
from eptq.utils.calibration import get_c4
from eptq.utils.hessian import get_hessians, get_layer_output
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_knee(D_sum: torch.Tensor, search_pct: float = 0.10):
    """
    Same Kneedle algorithm as eptq_quantize() auto_mask branch.
    Returns (threshold, knee_idx_in_top_search, n_search, n_blocks).
    """
    n_blocks = D_sum.shape[0]
    D_sorted, _ = D_sum.sort(descending=True)
    n_search = max(2, int(n_blocks * search_pct))
    D_top = D_sorted[:n_search]
    x = torch.linspace(0.0, 1.0, n_search, device=D_top.device)
    y = torch.log(D_top.clamp_min(1e-10))
    y = (y - y[-1]) / (y[0] - y[-1] + 1e-10)
    distance = (1.0 - x) - y
    knee_idx = int(distance.argmax().item())
    threshold = D_top[knee_idx].item()
    return threshold, knee_idx, n_search, n_blocks


def compute_dsum(W_orig, H_orig, e8q, percdamp=0.01, weight_scale_norm=True, device="cuda"):
    """Run RTN pre-pass and return D_sum array (shape: n_blocks)."""
    H = H_orig.detach().clone().to(device).float()
    W = W_orig.detach().clone().to(device).float()

    if weight_scale_norm:
        g, h = comp_gh(W)
        W /= g[:, None]
        W /= h[None, :]
        H *= h[:, None]
        H *= h[None, :]
    else:
        h = torch.ones(W.shape[1], device=device)

    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    while True:
        try:
            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(H.shape[1], device=H.device)
            H[diag, diag] += damp
            H = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(H)
            H_inv_diag = torch.diag(H_inv)
            break
        except torch._C._LinAlgError:
            percdamp *= 2
            H = H_orig.detach().clone().to(device).float()
            if weight_scale_norm:
                H *= h[:, None]
                H *= h[None, :]
            H[dead, dead] = 1

    S_search = W.abs().quantile(0.95, dim=0).clamp_min(1e-8)
    W_scaled = W / S_search[None, :]
    one = torch.tensor(1.0, device=device, dtype=torch.float32)
    Q_a = e8q.quant(W_scaled, one)
    W_q_a = e8q.dequant(Q_a, one) * S_search[None, :]

    err = ((W - W_q_a) ** 2) / (H_inv_diag + 1e-8)
    D_sum = err.T.reshape(-1, 8).sum(dim=1)
    return D_sum.cpu()


def plot_dsums(dsums, knees, n_layers, title_prefix, out_path, top_pct=None):
    """
    Plot sorted D_sum distributions with knee-point markers.
    knees[name][lid] = (threshold, knee_pct) where knee_pct is the percentile
    position of the knee in the full distribution.
    """
    sublayer_names = list(dsums.keys())
    n_sub = len(sublayer_names)
    ncols = 3
    nrows = (n_sub + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i / max(n_layers - 1, 1)) for i in range(n_layers)]

    for idx, name in enumerate(sublayer_names):
        ax = axes[idx]
        for lid in range(n_layers):
            if lid not in dsums[name]:
                continue
            d = dsums[name][lid].numpy()
            d_sorted = np.sort(d)
            n = len(d_sorted)

            if top_pct is not None:
                top_k = max(10, int(n * top_pct / 100))
                d_plot = d_sorted[-top_k:]
                x = np.arange(top_k)
                xlabel = f"rank (0={(100-top_pct):.0f}th pct, right=max)"
            else:
                d_plot = d_sorted
                x = np.arange(n) / n * 100
                xlabel = "percentile (%)"

            ax.semilogy(x, d_plot + 1e-10, color=colors[lid], alpha=0.7, linewidth=0.8)

            # Overlay knee marker
            if name in knees and lid in knees[name]:
                threshold, knee_pct = knees[name][lid]
                if top_pct is not None:
                    # window: x=0 → (100-top_pct)th pct, x=top_k-1 → 100th pct
                    rank_in_top = (knee_pct - (100.0 - top_pct)) / top_pct * top_k
                    if 0 <= rank_in_top < top_k:
                        ax.plot(rank_in_top, threshold + 1e-10, 'x',
                                color=colors[lid], markersize=6, markeredgewidth=1.5)
                else:
                    ax.plot(knee_pct, threshold + 1e-10, 'x',
                            color=colors[lid], markersize=6, markeredgewidth=1.5)

        suffix = f" (top {top_pct}%)" if top_pct else ""
        ax.set_title(f"{name}{suffix}", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("D_sum (log scale)")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    handles = [plt.Line2D([0], [0], color=colors[i], label=f"L{i}", linewidth=1)
               for i in range(n_layers)]
    handles.append(plt.Line2D([0], [0], marker='x', color='gray',
                               label='knee', linewidth=0, markersize=7, markeredgewidth=1.5))
    axes[0].legend(handles=handles, fontsize=6, ncol=2, loc="upper left")
    for idx in range(n_sub, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title_prefix, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}", file=sys.stderr)


def plot_comparison(dsums_with, dsums_without, knees_with, knees_without,
                    n_layers, out_path, top_pct=None):
    """Side-by-side comparison: with scale (left) vs without scale (right) per sublayer."""
    sublayer_names = list(dsums_with.keys())
    n_sub = len(sublayer_names)
    ncols = 2

    fig, axes = plt.subplots(n_sub, ncols, figsize=(10, 4 * n_sub))
    if n_sub == 1:
        axes = axes[None, :]

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i / max(n_layers - 1, 1)) for i in range(n_layers)]

    for row, name in enumerate(sublayer_names):
        for col, (label, dsums, knees) in enumerate([
            ("with scale (g/h)", dsums_with, knees_with),
            ("without scale",    dsums_without, knees_without),
        ]):
            ax = axes[row, col]
            for lid in range(n_layers):
                if lid not in dsums[name]:
                    continue
                d = dsums[name][lid].numpy()
                d_sorted = np.sort(d)
                n = len(d_sorted)

                if top_pct is not None:
                    top_k = max(10, int(n * top_pct / 100))
                    d_plot = d_sorted[-top_k:]
                    x = np.arange(top_k)
                    xlabel = f"rank (top {top_pct}%)"
                else:
                    d_plot = d_sorted
                    x = np.arange(n) / n * 100
                    xlabel = "percentile (%)"

                ax.semilogy(x, d_plot + 1e-10, color=colors[lid], alpha=0.7, linewidth=0.8)

                if name in knees and lid in knees[name]:
                    threshold, knee_pct = knees[name][lid]
                    if top_pct is not None:
                        rank_in_top = (knee_pct - (100.0 - top_pct)) / top_pct * top_k
                        if 0 <= rank_in_top < top_k:
                            ax.plot(rank_in_top, threshold + 1e-10, 'x',
                                    color=colors[lid], markersize=6, markeredgewidth=1.5)
                    else:
                        ax.plot(knee_pct, threshold + 1e-10, 'x',
                                color=colors[lid], markersize=6, markeredgewidth=1.5)

            ax.set_title(f"{name} — {label}", fontsize=9)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("D_sum")
            ax.grid(True, which="both", linestyle="--", alpha=0.4)

    handles = [plt.Line2D([0], [0], color=colors[i], label=f"L{i}", linewidth=1)
               for i in range(n_layers)]
    handles.append(plt.Line2D([0], [0], marker='x', color='gray',
                               label='knee', linewidth=0, markersize=7, markeredgewidth=1.5))
    axes[0, 0].legend(handles=handles, fontsize=6, ncol=2, loc="upper left")

    suffix = f" (top {top_pct}%)" if top_pct else ""
    plt.suptitle(f"llama-3-8b: D_sum with vs without scale normalization{suffix}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}", file=sys.stderr)


def main():
    model_name = "meta-llama/Meta-Llama-3-8B"
    device = "cuda"
    out_dir = Path("/home/hmpark/eptq_new/results/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reuse cached D_sums if available
    cache_path = out_dir / "dsums_llama3_8b_both.pt"
    if cache_path.exists():
        print("Loading cached D_sums...", file=sys.stderr)
        saved = torch.load(cache_path)
        dsums_with    = defaultdict(dict, {k: dict(v) for k, v in saved["with_scale"].items()})
        dsums_without = defaultdict(dict, {k: dict(v) for k, v in saved["without_scale"].items()})
        n_layers = max(max(v.keys()) for v in dsums_with.values()) + 1
    else:
        print("Loading model...", file=sys.stderr)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", use_safetensors=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        handler = LlamaHandler()

        e8q = E8D4Quantizer(device=device)
        e8q.remove_last()

        print("Loading calibration data...", file=sys.stderr)
        calib = get_c4(128, 42, 2048, tokenizer)

        print("Computing first layer input...", file=sys.stderr)
        layer_input, cache = handler.get_first_layer_input(model, calib, batchsize=128)
        forward_kwargs = handler.get_layer_forward_kwargs(cache)
        for key, val in forward_kwargs.items():
            if isinstance(val, torch.Tensor):
                forward_kwargs[key] = val.to(device)
            elif isinstance(val, tuple):
                forward_kwargs[key] = tuple(
                    v.to(device) if isinstance(v, torch.Tensor) else v for v in val
                )

        layers = handler.get_layers(model)
        n_layers = len(layers)

        dsums_with    = defaultdict(dict)
        dsums_without = defaultdict(dict)

        for lid in range(n_layers):
            layer = layers[lid]
            layer.to(device)
            print(f"Layer {lid+1}/{n_layers}...", file=sys.stderr)

            hessians = get_hessians(layer, layer_input, batchsize=8, forward_kwargs=forward_kwargs)
            linear_layers = handler.get_linear_layers(layer)

            for name, linear in linear_layers.items():
                H = hessians[name]
                W = linear.weight.data
                d_with    = compute_dsum(W, H, e8q, weight_scale_norm=True,  device=device)
                d_without = compute_dsum(W, H, e8q, weight_scale_norm=False, device=device)
                dsums_with[name][lid]    = d_with
                dsums_without[name][lid] = d_without

            layer_input = get_layer_output(layer, layer_input, forward_kwargs=forward_kwargs)
            layer.cpu()

        torch.save({"with_scale": dict(dsums_with), "without_scale": dict(dsums_without)},
                   cache_path)

    # Compute knee thresholds for all sublayer×layer combinations
    print("Computing knee thresholds...", file=sys.stderr)
    knees_with    = defaultdict(dict)  # [name][lid] = (threshold, knee_pct)
    knees_without = defaultdict(dict)

    for name in dsums_with:
        for lid in dsums_with[name]:
            d = dsums_with[name][lid]
            threshold, knee_idx, n_search, n_blocks = compute_knee(d)
            # k = number of blocks above threshold → percentile of threshold
            k = int((d > threshold).sum().item())
            knee_pct = (1.0 - k / n_blocks) * 100.0
            knees_with[name][lid] = (threshold, knee_pct)

    for name in dsums_without:
        for lid in dsums_without[name]:
            d = dsums_without[name][lid]
            threshold, knee_idx, n_search, n_blocks = compute_knee(d)
            k = int((d > threshold).sum().item())
            knee_pct = (1.0 - k / n_blocks) * 100.0
            knees_without[name][lid] = (threshold, knee_pct)

    # Plots with knee markers (with scale)
    plot_dsums(dsums_with, knees_with, n_layers,
               "llama-3-8b D_sum (with scale) — full + knee",
               out_dir / "dsums_llama3_8b_with_scale_knee.png")
    plot_dsums(dsums_with, knees_with, n_layers,
               "llama-3-8b D_sum (with scale) — top 10% + knee",
               out_dir / "dsums_llama3_8b_with_scale_top10pct_knee.png", top_pct=10)
    plot_dsums(dsums_with, knees_with, n_layers,
               "llama-3-8b D_sum (with scale) — top 1% + knee",
               out_dir / "dsums_llama3_8b_with_scale_top1pct_knee.png", top_pct=1)

    # Plots with knee markers (without scale)
    plot_dsums(dsums_without, knees_without, n_layers,
               "llama-3-8b D_sum (no scale) — full + knee",
               out_dir / "dsums_llama3_8b_no_scale_knee.png")
    plot_dsums(dsums_without, knees_without, n_layers,
               "llama-3-8b D_sum (no scale) — top 10% + knee",
               out_dir / "dsums_llama3_8b_no_scale_top10pct_knee.png", top_pct=10)
    plot_dsums(dsums_without, knees_without, n_layers,
               "llama-3-8b D_sum (no scale) — top 1% + knee",
               out_dir / "dsums_llama3_8b_no_scale_top1pct_knee.png", top_pct=1)

    # Side-by-side comparison with knee markers
    plot_comparison(dsums_with, dsums_without, knees_with, knees_without,
                    n_layers, out_dir / "dsums_llama3_8b_compare_knee.png")
    plot_comparison(dsums_with, dsums_without, knees_with, knees_without,
                    n_layers, out_dir / "dsums_llama3_8b_compare_top10pct_knee.png", top_pct=10)
    plot_comparison(dsums_with, dsums_without, knees_with, knees_without,
                    n_layers, out_dir / "dsums_llama3_8b_compare_top1pct_knee.png", top_pct=1)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
