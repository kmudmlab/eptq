"""
Inference speed comparison:
  (A) Reconstruct path  — dequantize_layer() → fp16 weight → torch.mm
  (B) Kernel path       — EPTQLinear fused CUDA kernel (no weight materialisation)

Run from the repo parent directory:
    python3 eptq-new/benchmark_inference.py
"""

import sys
import time
from dataclasses import dataclass
from typing import List

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from eptq.algorithms.eptq import eptq_quantize, dequantize_layer
from eptq.quantizers.e8_lattice import E8LatticeQuantizer
from eptq.quantizers.e8_d4 import E8D4Quantizer
from eptq.kernels.linear import EPTQLinear

# ─── helpers ──────────────────────────────────────────────────────────────────

def make_qd(out_f: int, in_f: int, quantizer, device: str):
    """Quantize a random weight matrix (simple diagonal Hessian)."""
    W = torch.randn(out_f, in_f, device=device, dtype=torch.float32) * 0.02
    H = torch.eye(in_f, device=device)
    qd = eptq_quantize(H, W, quantizer, percmask=0.002)
    return qd


def timed(fn, warmup: int = 5, reps: int = 50):
    """Return median latency in ms over `reps` runs."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)

    times.sort()
    return times[len(times) // 2]   # median


def mem_of(tensor: torch.Tensor) -> float:
    """Tensor size in MB."""
    return tensor.element_size() * tensor.nelement() / 1e6


# ─── benchmark one configuration ──────────────────────────────────────────────

@dataclass
class Result:
    label: str
    out_f: int
    in_f: int
    batch: int
    reconstruct_ms: float
    kernel_ms: float
    speedup: float
    weight_mb_fp16: float
    weight_mb_quant: float
    mem_saving_pct: float


def bench(out_f: int, in_f: int, batch: int,
          quantizer, quantizer_type: str,
          codebook: torch.Tensor,
          device: str = "cuda") -> Result:

    qd = make_qd(out_f, in_f, quantizer, device)
    x  = torch.randn(batch, in_f, device=device, dtype=torch.float16)

    # ── (A) Reconstruct path ──────────────────────────────────────────────────
    W_fp16 = dequantize_layer(qd, quantizer)        # materialise fp16 weights

    def run_reconstruct():
        return torch.mm(x, W_fp16.T)

    ms_reconstruct = timed(run_reconstruct)

    # ── (B) Kernel path ───────────────────────────────────────────────────────
    lin = EPTQLinear.from_quantized_data(qd, codebook, quantizer_type)

    def run_kernel():
        return lin(x)

    ms_kernel = timed(run_kernel)

    # ── Memory ───────────────────────────────────────────────────────────────
    mb_fp16  = mem_of(W_fp16)
    mb_quant = (mem_of(qd.Q)      # int16 indices
              + mem_of(qd.g)      # fp16 row scales
              + mem_of(qd.h)      # fp16 col scales
              + mem_of(qd.O))     # fp16 masked values
    if qd.bias is not None:
        mb_quant += mem_of(qd.bias)

    return Result(
        label=quantizer_type,
        out_f=out_f, in_f=in_f, batch=batch,
        reconstruct_ms=ms_reconstruct,
        kernel_ms=ms_kernel,
        speedup=ms_reconstruct / ms_kernel,
        weight_mb_fp16=mb_fp16,
        weight_mb_quant=mb_quant,
        mem_saving_pct=(1 - mb_quant / mb_fp16) * 100,
    )


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    device = "cuda"
    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch {torch.__version__}\n")

    # Build quantizers once (codebook is shared)
    e8q = E8LatticeQuantizer(device=device)
    e8q.remove_last()
    e8d4q = E8D4Quantizer(device=device)

    # ── Layer shapes from representative LLMs ────────────────────────────────
    # (out_features, in_features, description)
    shapes = [
        (768,   768,  "OPT-125M  attn proj"),
        (3072,  768,  "OPT-125M  FFN"),
        (4096,  4096, "LLaMA-7B  attn proj"),
        (11008, 4096, "LLaMA-7B  FFN gate/up"),
        (4096,  11008,"LLaMA-7B  FFN down"),
    ]

    batch_sizes = [1, 8, 32]

    configs = [
        ("e8_lattice", e8q,   e8q.points),
        ("e8_d4",      e8d4q, e8d4q.codes),
    ]

    for q_type, quantizer, codebook in configs:
        cb_mb = mem_of(codebook)
        print("=" * 72)
        print(f"  Quantizer: {q_type}  (codebook {codebook.shape}, "
              f"{cb_mb:.2f} MB in memory)")
        print("=" * 72)
        print(f"{'Shape':>22}  {'Batch':>5}  "
              f"{'Reconstruct':>11}  {'Kernel':>9}  "
              f"{'Speedup':>7}  {'W_fp16':>7}  {'W_quant':>7}  {'Saving':>7}")
        print("-" * 72)

        for out_f, in_f, desc in shapes:
            for batch in batch_sizes:
                r = bench(out_f, in_f, batch, quantizer, q_type,
                          codebook, device)
                print(
                    f"  {desc:>20}  {batch:>5}  "
                    f"  {r.reconstruct_ms:>8.3f} ms"
                    f"  {r.kernel_ms:>7.3f} ms"
                    f"  {r.speedup:>6.2f}x"
                    f"  {r.weight_mb_fp16:>6.2f}MB"
                    f"  {r.weight_mb_quant:>6.2f}MB"
                    f"  {r.mem_saving_pct:>5.1f}%"
                )
        print()

    # ── Summary note on codebook overhead ────────────────────────────────────
    print("Note: kernel path stores weights as int16 indices + fp16 scales,")
    print("using ~2 bytes/weight vs 2 bytes/weight for fp16")
    print("(so raw weight memory is similar, but the quantized format")
    print(" additionally saves scales g/h per row/col instead of per-element).")
    print()
    print("Theoretical weight memory ratio (no masked):")
    print("  fp16:  out * in * 2 bytes")
    print("  quant: (out/8 * in) * 2  [Q int16]")
    print("       + out * 2           [g fp16]")
    print("       + in  * 2           [h fp16]")
    print("  ≈ same as fp16 for large layers (scales are negligible)")


if __name__ == "__main__":
    main()
