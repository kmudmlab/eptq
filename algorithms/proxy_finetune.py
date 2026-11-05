"""Layer-wise proxy fine-tuning and end-to-end fine-tuning for EPTQ.

Two fine-tuning modes are available:

1. **Proxy FT** (``proxy_finetune_layer``): Per-decoder-block MSE minimization
   between FP16 and quantized block outputs. Q is frozen; only g/h (and
   optionally O) are updated. Fast and memory-efficient.

2. **E2E FT** (``e2e_finetune_model``): Full-model end-to-end fine-tuning with
   cross-entropy loss on calibration tokens. All g/h across all layers are
   jointly optimized. Requires the full quantized model to fit in GPU memory.
"""

import copy
import sys
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..algorithms.eptq import dequantize_layer
from ..config import QuantizedLayerData
from ..quantizers.base import BaseQuantizer


class _ScaledLinear(nn.Module):
    """Linear layer with fixed normalized weights and learnable per-channel scales.

    Fast path (O frozen):
        forward: y = g * (W_base @ (h * x)) + bias

    O-learnable path:
        W is materialized each forward as W_base_nc * g * h, then critical
        positions are overwritten with g[out] * O[k] * h[in] via index_put.
        W_base_nc has zeros at critical positions.
    """

    def __init__(
        self,
        W_base: torch.Tensor,
        g: torch.Tensor,
        h: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        O_init: Optional[torch.Tensor] = None,   # (n_crit, 8), None = frozen O
        idx0_q: Optional[torch.Tensor] = None,   # (n_crit,) Q-row indices
        idx1_q: Optional[torch.Tensor] = None,   # (n_crit,) Q-col indices
    ):
        super().__init__()
        self.register_buffer("W_base", W_base)   # (out, in) float32, frozen
        self.g = nn.Parameter(g.float())          # (out,)
        self.h = nn.Parameter(h.float())          # (in,)
        if bias is not None:
            self.register_buffer("bias", bias.float())
        else:
            self.bias = None

        if O_init is not None and O_init.numel() > 0:
            self.O = nn.Parameter(O_init.float())  # (n_crit, 8)
            self.register_buffer("idx0_q", idx0_q)
            self.register_buffer("idx1_q", idx1_q)
            # Precompute flat W-space indices for index_put
            arange8 = torch.arange(8, device=idx0_q.device)
            idx0_flat = (idx0_q.unsqueeze(1) * 8 + arange8.unsqueeze(0)).reshape(-1)
            idx1_flat = idx1_q.unsqueeze(1).expand(-1, 8).reshape(-1).contiguous()
            self.register_buffer("idx0_flat", idx0_flat)  # (n_crit*8,) output indices
            self.register_buffer("idx1_flat", idx1_flat)  # (n_crit*8,) input indices
        else:
            self.O = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        # W_base dtype controls compute precision:
        #   float32 → proxy-FT (high precision, per-layer)
        #   float16 → e2e-FT   (memory-efficient, full-model)
        w_dtype = self.W_base.dtype

        if self.O is None:
            # Fast path: channelwise scaling only
            y = F.linear(x.to(w_dtype) * self.h.to(w_dtype), self.W_base) * self.g.to(w_dtype)
        else:
            # O-learnable path: always float32 (proxy-FT only)
            W = self.W_base * self.g[:, None] * self.h[None, :]
            g_rows = self.g[self.idx0_flat]
            h_cols = self.h[self.idx1_flat]
            O_scaled = self.O.reshape(-1) * g_rows * h_cols
            W = torch.index_put(W, (self.idx0_flat, self.idx1_flat), O_scaled)
            y = F.linear(x.float(), W)

        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y.to(orig_dtype)


def _set_submodule(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _compute_loss(
    y_q: torch.Tensor,
    y_ref: torch.Tensor,
    ft_loss: str,
    ft_cosine_w: float,
) -> torch.Tensor:
    """Compute proxy fine-tuning loss.

    ft_loss options:
      "mse"        : MSE only (default)
      "cosine"     : 1 - cosine_similarity, averaged over tokens
      "mse+cosine" : MSE + ft_cosine_w * cosine_loss
    """
    y_q = y_q.float()
    y_ref = y_ref.float()

    if ft_loss == "mse":
        return F.mse_loss(y_q, y_ref)

    # cosine_similarity over hidden dim (last), result shape: (batch, seq)
    cos_loss = (1.0 - F.cosine_similarity(y_q, y_ref, dim=-1)).mean()

    if ft_loss == "cosine":
        return cos_loss

    # mse+cosine
    return F.mse_loss(y_q, y_ref) + ft_cosine_w * cos_loss


@torch.no_grad()
def _eval_loss(
    layer: nn.Module,
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    batchsize: int,
    device: str,
    forward_kwargs: Dict[str, Any],
    ft_loss: str = "mse",
    ft_cosine_w: float = 1.0,
) -> float:
    total, steps = 0.0, 0
    for i in range(0, x_all.shape[0], batchsize):
        x = x_all[i : i + batchsize].to(device)
        y = y_all[i : i + batchsize].to(device)
        out = layer(x, **forward_kwargs)
        y_q = out[0] if isinstance(out, tuple) else out
        total += _compute_loss(y_q, y, ft_loss, ft_cosine_w).item()
        steps += 1
    return total / max(steps, 1)


def proxy_finetune_layer(
    layer: nn.Module,
    quant_data_map: Dict[str, QuantizedLayerData],
    e8q: BaseQuantizer,
    layer_input: torch.Tensor,         # (n_samples, seq_len, hidden), CPU
    layer_output_fp16: torch.Tensor,   # (n_samples, seq_len, hidden), CPU
    forward_kwargs: Dict[str, Any],
    device: str = "cuda",
    ft_epochs: int = 3,
    ft_lr: float = 1e-4,
    ft_batchsize: int = 8,
    ft_valid_size: int = 0,
    ft_early_stop: int = 5,
    ft_tune_O: bool = False,
    ft_loss: str = "mse",
    ft_cosine_w: float = 1.0,
    layer_name: str = "",
) -> Dict[str, QuantizedLayerData]:
    """Fine-tune g, h (and optionally O) for all quantized linear sublayers in one decoder block.

    When ft_valid_size > 0, the last ft_valid_size samples are held out as a
    validation set and early stopping is applied (patience = ft_early_stop epochs).
    The best-validation checkpoint is returned regardless of early stopping.

    When ft_tune_O=True, the full-precision critical-weight values O are also
    treated as learnable parameters (in addition to g and h).

    Returns an updated quant_data_map with new g, h (and O if ft_tune_O) tensors
    (fp16, on CPU).
    """
    # --- Train / validation split ---
    n_samples = layer_input.shape[0]
    n_valid = min(ft_valid_size, n_samples - 1) if ft_valid_size > 0 else 0
    n_train = n_samples - n_valid

    x_train = layer_input[:n_train]
    y_train = layer_output_fp16[:n_train]
    x_valid = layer_input[n_train:] if n_valid > 0 else None
    y_valid = layer_output_fp16[n_train:] if n_valid > 0 else None

    if layer_name:
        extras = []
        if ft_tune_O:
            extras.append("tune_O=True")
        if ft_loss != "mse":
            extras.append(f"loss={ft_loss}" + (f"(w={ft_cosine_w})" if ft_loss == "mse+cosine" else ""))
        suffix = ("  " + "  ".join(extras)) if extras else ""
        print(
            f"  [proxy-ft] {layer_name}: train={n_train} valid={n_valid}{suffix}",
            file=sys.stderr,
        )

    # --- Replace linear sublayers with _ScaledLinear ---
    scaled: Dict[str, _ScaledLinear] = {}
    saved: Dict[str, nn.Module] = {}

    for name, qd in quant_data_map.items():
        W_q = dequantize_layer(qd, e8q).float().to(device)
        g = qd.g.float().to(device)
        h = qd.h.float().to(device)
        W_base = W_q / g.abs().clamp(min=1e-8).unsqueeze(1) / h.abs().clamp(min=1e-8).unsqueeze(0)
        bias = qd.bias.float().to(device) if qd.bias is not None else None

        O_init, idx0_q, idx1_q = None, None, None
        if ft_tune_O and qd.O.numel() > 0:
            # Extract Q-space critical indices
            Q_int = qd.Q.int() + 32768
            mask = Q_int.to(device) == 65535
            idx0_q, idx1_q = mask.nonzero(as_tuple=True)

            # Zero out critical positions in W_base so they don't contribute
            # through the g/h path (they'll be handled via learnable O instead)
            arange8 = torch.arange(8, device=device)
            idx0_flat_init = (idx0_q.unsqueeze(1) * 8 + arange8.unsqueeze(0)).reshape(-1)
            idx1_flat_init = idx1_q.unsqueeze(1).expand(-1, 8).reshape(-1)
            W_base = W_base.clone()
            W_base[idx0_flat_init, idx1_flat_init] = 0.0

            O_init = qd.O.float().to(device)  # (n_crit, 8)

        sl = _ScaledLinear(W_base, g, h, bias, O_init=O_init, idx0_q=idx0_q, idx1_q=idx1_q)
        scaled[name] = sl
        saved[name] = layer.get_submodule(name)
        _set_submodule(layer, name, sl)

    # --- Optimiser: g, h, and optionally O ---
    all_params = []
    for sl in scaled.values():
        all_params.extend([sl.g, sl.h])
        if sl.O is not None:
            all_params.append(sl.O)
    optimizer = torch.optim.Adam(all_params, lr=ft_lr)

    best: Dict[str, tuple] = {
        n: (sl.g.data.clone(), sl.h.data.clone(), sl.O.data.clone() if sl.O is not None else None)
        for n, sl in scaled.items()
    }
    best_loss = float("inf")
    worse_ct = 0

    # --- Training loop ---
    layer.train()
    with torch.enable_grad():
        for epoch in range(ft_epochs):
            running, steps = 0.0, 0
            for i in range(0, n_train, ft_batchsize):
                x = x_train[i : i + ft_batchsize].to(device)
                y_ref = y_train[i : i + ft_batchsize].to(device)

                out = layer(x, **forward_kwargs)
                y_q = out[0] if isinstance(out, tuple) else out

                loss = _compute_loss(y_q, y_ref, ft_loss, ft_cosine_w)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running += loss.item()
                steps += 1
            train_loss = running / max(steps, 1)

            if n_valid > 0:
                layer.eval()
                val_loss = _eval_loss(layer, x_valid, y_valid, ft_batchsize, device, forward_kwargs, ft_loss, ft_cosine_w)
                layer.train()
                monitor_loss = val_loss
                print(
                    f"  [proxy-ft] {layer_name} epoch {epoch + 1}/{ft_epochs}"
                    f"  train={train_loss:.6g}  val={val_loss:.6g}",
                    file=sys.stderr,
                )
            else:
                monitor_loss = train_loss
                print(
                    f"  [proxy-ft] {layer_name} epoch {epoch + 1}/{ft_epochs}"
                    f"  train={train_loss:.6g}",
                    file=sys.stderr,
                )

            if monitor_loss < best_loss:
                best_loss = monitor_loss
                worse_ct = 0
                for n, sl in scaled.items():
                    best[n] = (
                        sl.g.data.clone(),
                        sl.h.data.clone(),
                        sl.O.data.clone() if sl.O is not None else None,
                    )
            else:
                worse_ct += 1
                if n_valid > 0 and worse_ct >= ft_early_stop:
                    print(
                        f"  [proxy-ft] {layer_name} early stop at epoch {epoch + 1}",
                        file=sys.stderr,
                    )
                    break

    layer.eval()

    # --- Restore original modules ---
    for name, orig in saved.items():
        _set_submodule(layer, name, orig)

    # --- Build updated quant_data with best g, h (and O if learned) ---
    updated: Dict[str, QuantizedLayerData] = {}
    for name, qd in quant_data_map.items():
        if name in best:
            g_new, h_new, O_new_flat = best[name]
            O_new = O_new_flat.reshape(-1, 8).cpu().half() if O_new_flat is not None else qd.O
            updated[name] = QuantizedLayerData(
                Q=qd.Q,
                g=g_new.cpu().half(),
                h=h_new.cpu().half(),
                O=O_new,
                bias=qd.bias,
                chosen_r=qd.chosen_r,
                S_group=qd.S_group,
            )
        else:
            updated[name] = qd

    return updated


def e2e_finetune_model(
    model: nn.Module,
    layers: List[nn.Module],
    all_quant_maps: Dict[int, Dict[str, "QuantizedLayerData"]],
    e8q: "BaseQuantizer",
    calib_tokens: torch.Tensor,          # (n_samples, seq_len) int64 token IDs
    device: str = "cuda",
    ft_epochs: int = 10,
    ft_lr: float = 1e-4,
    ft_batchsize: int = 1,
    ft_valid_size: int = 0,
    ft_early_stop: int = 5,
) -> Dict[int, Dict[str, "QuantizedLayerData"]]:
    """End-to-end fine-tuning of g/h across all layers jointly using CE loss.

    All decoder layers are loaded to GPU simultaneously. Q and O are frozen;
    only g/h are learnable. W_base is stored in fp16 to reduce memory usage.

    Returns updated all_quant_maps with new g/h values.
    """
    print("  [e2e-ft] Replacing linear sublayers with _ScaledLinear (fp16 W_base)...", file=sys.stderr)

    scaled_per_layer: Dict[int, Dict[str, _ScaledLinear]] = {}
    saved_per_layer: Dict[int, Dict[str, nn.Module]] = {}

    for lid, layer in enumerate(layers):
        if lid not in all_quant_maps:
            continue
        quant_map = all_quant_maps[lid]
        scaled_per_layer[lid] = {}
        saved_per_layer[lid] = {}
        for name, qd in quant_map.items():
            # Dequantize and normalize to get W_base, store as fp16 to save memory
            W_q = dequantize_layer(qd, e8q).float()
            g = qd.g.float()
            h = qd.h.float()
            W_base = (W_q / g.abs().clamp(min=1e-8).unsqueeze(1)
                         / h.abs().clamp(min=1e-8).unsqueeze(0)).half()
            bias = qd.bias.float() if qd.bias is not None else None
            sl = _ScaledLinear(W_base, g, h, bias)  # W_base=fp16, g/h=fp32
            scaled_per_layer[lid][name] = sl
            saved_per_layer[lid][name] = layer.get_submodule(name)
            _set_submodule(layer, name, sl)

    # Move full model to device (all layers including _ScaledLinear)
    print(f"  [e2e-ft] Moving model to {device}...", file=sys.stderr)
    model.to(device)

    # Freeze all params; unfreeze only g/h in _ScaledLinear
    for param in model.parameters():
        param.requires_grad_(False)
    all_params = []
    for lid, scaled in scaled_per_layer.items():
        for sl in scaled.values():
            sl.g.requires_grad_(True)
            sl.h.requires_grad_(True)
            all_params.extend([sl.g, sl.h])

    optimizer = torch.optim.Adam(all_params, lr=ft_lr)

    # Train / validation split
    n_samples = calib_tokens.shape[0]
    n_valid = min(ft_valid_size, n_samples - 1) if ft_valid_size > 0 else 0
    n_train = n_samples - n_valid
    train_tokens = calib_tokens[:n_train]
    valid_tokens = calib_tokens[n_train:] if n_valid > 0 else None

    print(f"  [e2e-ft] train={n_train} valid={n_valid}  epochs={ft_epochs}  lr={ft_lr}", file=sys.stderr)

    def _ce_loss(tokens_batch: torch.Tensor) -> torch.Tensor:
        input_ids = tokens_batch[:, :-1].to(device)
        labels = tokens_batch[:, 1:].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(input_ids).logits
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

    @torch.no_grad()
    def _eval_ce(tokens: torch.Tensor) -> float:
        model.eval()
        total, steps = 0.0, 0
        for i in range(0, tokens.shape[0], ft_batchsize):
            total += _ce_loss(tokens[i : i + ft_batchsize]).item()
            steps += 1
        model.train()
        return total / max(steps, 1)

    # Snapshot best parameters
    best_params = {
        lid: {name: (sl.g.data.clone(), sl.h.data.clone())
              for name, sl in scaled.items()}
        for lid, scaled in scaled_per_layer.items()
    }
    best_loss = _eval_ce(valid_tokens) if n_valid > 0 else float("inf")
    if n_valid > 0:
        print(f"  [e2e-ft] initial val={best_loss:.6g}", file=sys.stderr)
    worse_ct = 0

    model.train()
    with torch.enable_grad():
        for epoch in range(ft_epochs):
            perm = torch.randperm(n_train)
            running, steps = 0.0, 0
            for i in range(0, n_train, ft_batchsize):
                batch = train_tokens[perm[i : i + ft_batchsize]]
                loss = _ce_loss(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running += loss.item()
                steps += 1
            train_loss = running / max(steps, 1)

            if n_valid > 0:
                val_loss = _eval_ce(valid_tokens)
                monitor_loss = val_loss
                print(f"  [e2e-ft] epoch {epoch+1}/{ft_epochs}  train={train_loss:.6g}  val={val_loss:.6g}", file=sys.stderr)
            else:
                monitor_loss = train_loss
                print(f"  [e2e-ft] epoch {epoch+1}/{ft_epochs}  train={train_loss:.6g}", file=sys.stderr)

            if monitor_loss < best_loss:
                best_loss = monitor_loss
                worse_ct = 0
                for lid, scaled in scaled_per_layer.items():
                    for name, sl in scaled.items():
                        best_params[lid][name] = (sl.g.data.clone(), sl.h.data.clone())
            else:
                worse_ct += 1
                if n_valid > 0 and worse_ct >= ft_early_stop:
                    print(f"  [e2e-ft] early stop at epoch {epoch+1}", file=sys.stderr)
                    break

    model.eval()

    # Move model back to CPU and restore original modules
    model.cpu()
    for lid, saved in saved_per_layer.items():
        for name, orig in saved.items():
            _set_submodule(layers[lid], name, orig)

    # Build updated quant_maps with best g/h
    updated_all: Dict[int, Dict[str, "QuantizedLayerData"]] = {}
    for lid, quant_map in all_quant_maps.items():
        updated_all[lid] = {}
        for name, qd in quant_map.items():
            if lid in best_params and name in best_params[lid]:
                g_new, h_new = best_params[lid][name]
                updated_all[lid][name] = QuantizedLayerData(
                    Q=qd.Q,
                    g=g_new.cpu().half(),
                    h=h_new.cpu().half(),
                    O=qd.O,
                    bias=qd.bias,
                    chosen_r=qd.chosen_r,
                    S_group=qd.S_group,
                )
            else:
                updated_all[lid][name] = qd

    return updated_all
