"""Load a quantized model for direct inference using EPTQLinear layers."""

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ..config import QuantizedLayerData
from ..models import ModelHandler
from ..reconstruct import (
    get_model_handler_by_type,
    load_quantization_config,
    load_global_non_quantized_params,
    load_layer_non_quantized_params,
    load_quantized_layer,
)
from .linear import EPTQLinear


def load_quantized_model(
    quantized_dir: str,
    device: str = "cuda",
) -> Tuple[nn.Module, AutoTokenizer]:
    """
    Load a quantized model for inference using EPTQ CUDA kernels.

    Unlike reconstruct_model() which dequantizes weights back to fp16,
    this function keeps weights in quantized format (Q / g / h / O) and
    replaces each nn.Linear with an EPTQLinear that dequantizes on the fly.

    The codebook is shared across all layers (one copy in VRAM).

    Args:
        quantized_dir: Directory produced by eptq.quantize
        device:        Target device, e.g. "cuda" or "cuda:1"

    Returns:
        (model, tokenizer)
    """
    quantized_dir = Path(quantized_dir)

    # --- Config & quantizer type ---
    quant_config   = load_quantization_config(quantized_dir)
    quantizer_type = quant_config.get("quantizer_type", "e8_lattice")

    # --- Load shared codebook (one copy, moved to device once) ---
    print("Loading codebook...", file=sys.stderr)
    cb_data = torch.load(quantized_dir / "quantizer.pt", weights_only=True)
    if "raw_codes" in cb_data:
        # New format: codebook = raw_codes / radius
        codebook = (cb_data["raw_codes"] / float(cb_data["radius"])).to(device=device, dtype=torch.float16)
    else:
        # Legacy format: codebook already divided
        codebook = cb_data["codebook"].to(device=device, dtype=torch.float16)
    # E8D4 reshape to (512, 4) happens inside EPTQLinear.from_quantized_data()

    # --- Create empty model skeleton on meta device (no memory alloc / random init) ---
    print("Creating model structure...", file=sys.stderr)
    config = AutoConfig.from_pretrained(quantized_dir)
    model_dtype = getattr(config, "torch_dtype", torch.float16) or torch.float16
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=model_dtype)

    # --- Tokenizer ---
    tokenizer_dir = quantized_dir / "tokenizer"
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    except Exception:
        model_name = quant_config.get("model_name")
        if not model_name:
            raise ValueError(
                "Tokenizer not found in quantized_dir/tokenizer and "
                "'model_name' missing from quant_config.json."
            )
        print(
            f"Tokenizer not found in {tokenizer_dir}, "
            f"loading from {model_name}",
            file=sys.stderr,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- Model handler ---
    handler = get_model_handler_by_type(ModelHandler.detect_model_type(model))

    # --- Global non-quantized params (embeddings, final norm, lm_head) ---
    print("Loading non-quantized parameters...", file=sys.stderr)
    global_params = load_global_non_quantized_params(quantized_dir)

    # --- Layer-wise: replace nn.Linear with EPTQLinear ---
    layers   = handler.get_layers(model)
    n_layers = len(layers)
    print(f"Loading {n_layers} quantized layers...", file=sys.stderr)

    for lid in range(n_layers):
        layer     = layers[lid]
        layer_dir = quantized_dir / f"layer_{lid}"

        for name, _ in handler.get_linear_layers(layer).items():
            quant_data = load_quantized_layer(layer_dir, name)

            eptq_linear = EPTQLinear.from_quantized_data(
                quant_data=quant_data,
                codebook=codebook,   # shared reference
                quantizer_type=quantizer_type,
            )

            # Navigate the module hierarchy and replace the submodule
            parts, parent = name.split("."), layer
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], eptq_linear)

        # Layer norms and other non-quantized per-layer params
        layer_params = load_layer_non_quantized_params(layer_dir)
        for pname, param in layer_params.items():
            parts, target = pname.split("."), layer
            for part in parts[:-1]:
                target = getattr(target, part)
            setattr(target, parts[-1], nn.Parameter(param))

    # --- Global params ---
    for name, param in global_params.items():
        parts, target = name.split("."), model
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], nn.Parameter(param))

    # Warn if any meta tensors remain (means load logic missed some parameters)
    meta_params = [n for n, p in model.named_parameters() if p.is_meta]
    if meta_params:
        print(f"WARNING: {len(meta_params)} meta parameters not loaded: {meta_params[:5]}", file=sys.stderr)

    # Move full model to device; EPTQLinear._apply() handles the codebook ref
    model = model.to(device=device)
    model.eval()

    print("Quantized model loaded.", file=sys.stderr)
    return model, tokenizer
