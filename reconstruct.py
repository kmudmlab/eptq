"""Reconstruction script to convert quantized layers back to fp16 model."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .config import QuantizationConfig, QuantizedLayerData, GPTQLayerData
from .quantizers import E8LatticeQuantizer, E8D4Quantizer
from .quantizers.base import BaseQuantizer
from .algorithms.eptq import dequantize_layer
from .algorithms.gptq import dequantize_gptq_layer
from .models import ModelHandler, LlamaHandler, OPTHandler


def get_model_handler_by_type(model_type: str) -> ModelHandler:
    """Get the appropriate model handler based on model type string."""
    if model_type == "opt":
        return OPTHandler()
    elif model_type == "llama":
        return LlamaHandler()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def load_quantization_config(quantized_dir: Path) -> Dict:
    """Load quantization configuration from JSON file."""
    config_path = quantized_dir / "quant_config.json"
    with open(config_path, "r") as f:
        return json.load(f)


def load_quantizer(quantized_dir: Path, device: str = "cuda") -> BaseQuantizer:
    """Load quantizer from file.

    Handles three on-disk formats:
    - New format (raw_codes + radius): reconstruct exactly as raw_codes / radius.
    - Mid format (codebook + saved_radius): codebook at DEFAULT_RADIUS; _radius = saved_radius.
    - Old format (codebook only): codebook at actual radius; read from quant_config.json.
    """
    import json
    codebook_path = quantized_dir / "quantizer.pt"
    data = torch.load(codebook_path, weights_only=True)
    quantizer_type = data.get("quantizer_type", "e8_lattice")

    if "raw_codes" in data:
        # New format: reconstruct from raw codes + radius.
        raw = data["raw_codes"].float().to(device)
        radius = float(data["radius"])
        if quantizer_type == "e8_lattice":
            eq = E8LatticeQuantizer(device=device, codebook=raw / radius)
            eq._raw_points = raw
        else:
            eq = E8D4Quantizer(device=device, codebook=raw / radius)
            eq._raw_codes = raw
        eq._radius = radius
        warp_alpha = float(data.get("warp_alpha", 1.0))
        if warp_alpha != 1.0:
            eq.warp_codebook(warp_alpha)
        return eq

    # Legacy formats: codebook is pre-divided
    codebook = data["codebook"].float().to(device)
    if quantizer_type == "e8_lattice":
        eq = E8LatticeQuantizer(device=device, codebook=codebook)
    elif quantizer_type == "e8_d4":
        eq = E8D4Quantizer(device=device, codebook=codebook)
    else:
        raise ValueError(f"Unknown quantizer type: {quantizer_type}")

    if "saved_radius" in data:
        # Mid format: codebook stored at DEFAULT_RADIUS.
        eq._radius = float(data["saved_radius"])
    else:
        # Old format: codebook stored at actual radius; read from quant_config.json.
        quant_config_path = quantized_dir / "quant_config.json"
        if quant_config_path.exists():
            with open(quant_config_path) as f:
                quant_config = json.load(f)
            actual_radius = quant_config.get("codebook_radius")
            if actual_radius is not None:
                eq._radius = float(actual_radius)

    return eq


def load_quantized_layer(layer_dir: Path, layer_name: str) -> QuantizedLayerData:
    """Load quantized layer data from file."""
    layer_path = layer_dir / f"{layer_name}.pt"
    return QuantizedLayerData.load(layer_path)


def load_global_non_quantized_params(quantized_dir: Path) -> Dict[str, torch.Tensor]:
    """Load global non-quantized parameters (embeddings, final norm, lm_head) from file."""
    non_quantized_path = quantized_dir / "non_quantized_params.pt"
    return torch.load(non_quantized_path, weights_only=True)


def load_layer_non_quantized_params(layer_dir: Path) -> Dict[str, torch.Tensor]:
    """Load layer-specific non-quantized parameters from file."""
    non_quantized_path = layer_dir / "non_quantized.pt"
    return torch.load(non_quantized_path, weights_only=True)


def build_model_from_quantized(
    quantized_dir: str,
    device: str = "cuda",
    device_map: Optional[str] = None,
) -> tuple:
    """
    Dequantize all layers directly onto GPU(s) without allocating CPU RAM for weights.

    Uses accelerate's init_empty_weights so the model skeleton costs no RAM;
    each layer is dequantized and written straight to its target device.
    """
    from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
    from accelerate.utils import set_module_tensor_to_device

    quantized_dir = Path(quantized_dir)

    quant_config = load_quantization_config(quantized_dir)
    method = quant_config.get("method", "eptq")

    if method == "gptq":
        e8q = None
        print(
            f"GPTQ model: bits={quant_config.get('bits')} "
            f"group_size={quant_config.get('group_size')}",
            file=sys.stderr,
        )
    else:
        print("Loading EPTQ quantizer...", file=sys.stderr)
        e8q = load_quantizer(quantized_dir, device=device)

    print("Loading model config and creating model structure...", file=sys.stderr)
    config = AutoConfig.from_pretrained(quantized_dir)

    tokenizer_dir = quantized_dir / "tokenizer"
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    except Exception:
        if "model_name" not in quant_config:
            raise ValueError(
                f"Tokenizer not found in {tokenizer_dir} and 'model_name' not in quant_config.json."
            )
        model_name = quant_config["model_name"]
        print(f"Tokenizer not found in {tokenizer_dir}, loading from {model_name}", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create model skeleton with zero memory (meta tensors)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)

    handler = get_model_handler_by_type(ModelHandler.detect_model_type(model))
    layer_prefix = handler.get_layer_prefix()

    # Determine per-layer device assignment
    if device_map == "auto":
        n_gpus = torch.cuda.device_count()
        # Reserve 15% per GPU for activations / fragmentation
        max_memory = {
            i: int(torch.cuda.get_device_properties(i).total_memory * 0.85)
            for i in range(n_gpus)
        }
        no_split = [type(handler.get_layers(model)[0]).__name__]
        dm = infer_auto_device_map(
            model, max_memory=max_memory,
            no_split_module_classes=no_split,
            dtype=torch.float16,
        )
        from collections import Counter
        counts = Counter(str(v) for v in dm.values())
        print(f"Device map summary: {dict(counts)}", file=sys.stderr)
    else:
        dm = {}

    def resolve_device(param_path: str) -> str:
        """Find the target device by walking up the module path in dm."""
        parts = param_path.split(".")
        for i in range(len(parts), 0, -1):
            key = ".".join(parts[:i])
            if key in dm:
                val = dm[key]
                return f"cuda:{val}" if isinstance(val, int) else str(val)
        return device

    # Place global non-quantized params (embeddings, final norm, lm_head)
    print("Loading global non-quantized parameters...", file=sys.stderr)
    global_nq = load_global_non_quantized_params(quantized_dir)
    for name, param in tqdm(global_nq.items(), desc="Global params", leave=False):
        dev = resolve_device(name)
        set_module_tensor_to_device(model, name, dev, value=param.to(dev))

    # Dequantize and place each transformer layer directly on target device
    layers = handler.get_layers(model)
    n_layers = len(layers)
    print(f"Dequantizing {n_layers} layers directly to GPU...", file=sys.stderr)

    for lid in tqdm(range(n_layers), desc="Dequantizing layers"):
        layer = layers[lid]
        layer_dir = quantized_dir / f"layer_{lid}"
        layer_base = f"{layer_prefix}.{lid}"
        target_dev = resolve_device(layer_base)

        # For EPTQ the codebook lives on `device`; dequantize there then transfer if needed
        deq_dev = device if method == "eptq" else target_dev

        for lname, linear in handler.get_linear_layers(layer).items():
            layer_path = layer_dir / f"{lname}.pt"

            if method == "gptq":
                quant_data = GPTQLayerData.load(layer_path)
                quant_data.Q      = quant_data.Q.to(deq_dev)
                quant_data.scales = quant_data.scales.to(deq_dev)
                quant_data.zeros  = quant_data.zeros.to(deq_dev)
                W = dequantize_gptq_layer(quant_data)
            else:
                quant_data = QuantizedLayerData.load(layer_path)
                quant_data.Q = quant_data.Q.to(deq_dev)
                quant_data.g = quant_data.g.to(deq_dev)
                quant_data.h = quant_data.h.to(deq_dev)
                quant_data.O = quant_data.O.to(deq_dev)
                W = dequantize_layer(quant_data, e8q)

            if deq_dev != target_dev:
                W = W.to(target_dev)

            set_module_tensor_to_device(model, f"{layer_base}.{lname}.weight", target_dev, value=W)
            if quant_data.bias is not None:
                set_module_tensor_to_device(
                    model, f"{layer_base}.{lname}.bias", target_dev,
                    value=quant_data.bias.to(target_dev),
                )

        # Layer norms and other per-layer non-quantized params
        for name, param in load_layer_non_quantized_params(layer_dir).items():
            full_path = f"{layer_base}.{name}"
            dev = resolve_device(full_path)
            set_module_tensor_to_device(model, full_path, dev, value=param.to(dev))

    # Move any remaining CPU buffers (e.g. inv_freq from LlamaRotaryEmbedding,
    # which is non-persistent and not saved to disk) to the target device.
    for name, buf in list(model.named_buffers()):
        if buf.device.type == "cpu":
            set_module_tensor_to_device(model, name, device, value=buf.to(device))

    # Add inter-device routing hooks for multi-GPU
    if device_map == "auto" and dm:
        model = dispatch_model(model, device_map=dm)

    model.eval()
    return model, tokenizer


def reconstruct_model(
    quantized_dir: str,
    output_dir: str,
    device: str = "cuda",
) -> None:
    """Reconstruct a full fp16 model from quantized layer files and save to disk."""
    quantized_dir = Path(quantized_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reconstructing model from: {quantized_dir}", file=sys.stderr)
    print(f"Output directory: {output_dir}", file=sys.stderr)

    model, tokenizer = build_model_from_quantized(quantized_dir, device=device)

    print(f"Saving reconstructed model to: {output_dir}", file=sys.stderr)
    model.cpu()  # move to CPU before serialization
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Reconstruction complete!", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Reconstruct fp16 model from quantized layers")
    parser.add_argument(
        "--quantized_dir",
        type=str,
        required=True,
        help="Directory containing quantized layer files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for reconstructed model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )

    args = parser.parse_args()

    reconstruct_model(
        quantized_dir=args.quantized_dir,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
