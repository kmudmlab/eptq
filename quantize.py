"""Main quantization script with layer-wise saving."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import QuantizationConfig, QuantizedLayerData, GPTQLayerData
from .quantizers import E8LatticeQuantizer, E8D4Quantizer, GPTQQuantizer
from .algorithms.eptq import eptq_quantize, dequantize_layer
from .algorithms.gptq import gptq_quantize, dequantize_gptq_layer
from .algorithms.proxy_finetune import proxy_finetune_layer, e2e_finetune_model
from .models import ModelHandler, LlamaHandler, OPTHandler
from .utils.calibration import get_c4, get_wikitext2
from .utils.hessian import get_hessians, get_layer_output


def get_model_handler(model) -> ModelHandler:
    """Get the appropriate model handler based on model type."""
    model_type = ModelHandler.detect_model_type(model)
    if model_type == "opt":
        return OPTHandler()
    elif model_type == "llama":
        return LlamaHandler()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def save_quantization_config(config: QuantizationConfig, output_dir: Path) -> None:
    """Save quantization configuration to JSON file."""
    config_path = output_dir / "quant_config.json"
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)


def save_quantizer(e8q, output_dir: Path, quantizer_type: str) -> None:
    """Save quantizer as raw_codes + radius.

    Saving raw (unscaled) codes alongside the radius avoids any dependence
    on DEFAULT_RADIUS and is robust to future changes in code-generation order.
    load_quantizer reconstructs the exact codebook as raw_codes / radius.
    """
    codebook_path = output_dir / "quantizer.pt"
    if quantizer_type == "e8_lattice":
        raw = e8q._raw_points.float()
    else:  # e8_d4
        raw = e8q._raw_codes.float()
    torch.save({
        "quantizer_type": quantizer_type,
        "raw_codes": raw,
        "radius": e8q._radius,
        "warp_alpha": getattr(e8q, "_warp_alpha", 1.0),
    }, codebook_path)


def save_global_non_quantized_params(model, handler: ModelHandler, output_dir: Path) -> None:
    """
    Save global non-quantized parameters (embeddings, final layer norm, LM head).

    These parameters are not quantized and need to be saved separately
    for full model reconstruction without requiring the original model.
    Layer-specific parameters (layer norms) are saved per-layer.
    """
    non_quantized = {}

    # Save embedding parameters using handler
    embeddings = handler.get_embeddings(model)
    for path, module in embeddings.items():
        for name, param in module.named_parameters():
            non_quantized[f'{path}.{name}'] = param.data.cpu()

    # Save final layer norm
    final_norm = handler.get_final_norm(model)
    final_norm_path = handler.get_final_norm_path()
    if final_norm is not None:
        for name, param in final_norm.named_parameters():
            non_quantized[f'{final_norm_path}.{name}'] = param.data.cpu()

    # Save LM head
    if hasattr(model, 'lm_head'):
        non_quantized['lm_head.weight'] = model.lm_head.weight.data.cpu()
        if model.lm_head.bias is not None:
            non_quantized['lm_head.bias'] = model.lm_head.bias.data.cpu()

    # Save to file
    non_quantized_path = output_dir / "non_quantized_params.pt"
    torch.save(non_quantized, non_quantized_path)
    print(f"Saved {len(non_quantized)} global non-quantized parameters", file=sys.stderr)


def save_layer_non_quantized_params(layer: torch.nn.Module, layer_dir: Path) -> None:
    """
    Save layer-specific non-quantized parameters (layer norms) to the layer directory.

    Args:
        layer: The transformer layer module
        layer_dir: Directory for this layer's files
    """
    non_quantized = {}

    for name, module in layer.named_modules():
        # Save layer norm parameters
        if 'norm' in name.lower() or 'layernorm' in name.lower():
            for param_name, param in module.named_parameters(recurse=False):
                full_name = f'{name}.{param_name}' if name else param_name
                non_quantized[full_name] = param.data.cpu()

    # Save to file in layer directory
    non_quantized_path = layer_dir / "non_quantized.pt"
    torch.save(non_quantized, non_quantized_path)


def save_quantized_layer(
    layer_data: QuantizedLayerData,
    output_dir: Path,
    layer_name: str,
) -> None:
    """Save quantized layer data immediately after quantization."""
    layer_path = output_dir / f"{layer_name}.pt"
    layer_data.save(layer_path)


def quantize_model(config: QuantizationConfig, quant_results_file: str = None) -> None:
    """
    Main quantization entry point.

    Quantizes the model layer by layer, saving each layer's quantized
    data immediately to disk to minimize memory usage.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = config.device

    # Print configuration
    print(f"EPTQ Quantization with following parameters:", file=sys.stderr)
    print(f"  Model: {config.model_name}", file=sys.stderr)
    print(f"  Method: {config.method}", file=sys.stderr)
    print(f"  Output: {config.output_dir}", file=sys.stderr)
    for key, val in config.to_dict().items():
        if key not in ["model_name", "output_dir"]:
            print(f"  {key} = {val}", file=sys.stderr)

    # Load model and tokenizer
    print("Loading model and tokenizer...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype="auto", use_safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Get model handler
    handler = get_model_handler(model)

    # Initialize quantizer
    if config.method == "gptq":
        print(
            f"Initializing GPTQ quantizer "
            f"(bits={config.bits}, group_size={config.group_size}, "
            f"symmetric={config.symmetric}, actorder={config.actorder})...",
            file=sys.stderr,
        )
        e8q = GPTQQuantizer(
            bits=config.bits,
            symmetric=config.symmetric,
            group_size=config.group_size,
        )
    else:  # method == "eptq"
        print(
            f"Initializing EPTQ quantizer ({config.quantizer_type}, "
            f"radius={config.codebook_radius})...",
            file=sys.stderr,
        )
        if config.quantizer_type == "e8_lattice":
            e8q = E8LatticeQuantizer(device=device, radius=config.codebook_radius)
            e8q.remove_last()  # Reserve last index for masking
            print(f"Codebook shape: {e8q.points.shape}", file=sys.stderr)
        elif config.quantizer_type == "e8_d4":
            e8q = E8D4Quantizer(device=device, radius=config.codebook_radius)
            e8q.remove_last()  # No-op for E8D4, but keep for consistency
            if config.codebook_warp_alpha != 1.0:
                e8q.warp_codebook(config.codebook_warp_alpha)
                print(f"Codebook warped with alpha={config.codebook_warp_alpha}", file=sys.stderr)
            print(f"Codebook shape: {e8q.codes.shape}", file=sys.stderr)
        else:
            raise ValueError(f"Unknown quantizer type: {config.quantizer_type}")

    # Save config
    save_quantization_config(config, output_dir)
    # EPTQ uses a codebook that needs to be shared across layers; GPTQ doesn't.
    if config.method == "eptq":
        save_quantizer(e8q, output_dir, config.quantizer_type)

    # Prepare calibration data
    print("Loading calibration data...", file=sys.stderr)
    if config.calib_dataset == "wikitext2":
        calib = get_wikitext2(config.n_samples, config.seed, config.seqlen, tokenizer)
    else:
        calib = get_c4(config.n_samples, config.seed, config.seqlen, tokenizer)

    # Get first layer input
    print("Computing first layer input...", file=sys.stderr)
    layer_input, cache = handler.get_first_layer_input(model, calib, batchsize=128)
    forward_kwargs = handler.get_layer_forward_kwargs(cache)

    # Move forward_kwargs tensors to device
    for key, val in forward_kwargs.items():
        if isinstance(val, torch.Tensor):
            forward_kwargs[key] = val.to(device)
        elif isinstance(val, tuple):
            forward_kwargs[key] = tuple(
                v.to(device) if isinstance(v, torch.Tensor) else v for v in val
            )

    # Get layers
    layers = handler.get_layers(model)
    n_layers = len(layers)

    # Layer-wise quantization
    print(f"Quantizing {n_layers} layers...", file=sys.stderr)
    tqdm_iter = tqdm(range(n_layers))
    quant_start = time.perf_counter()
    total_crit = 0
    total_blocks = 0
    all_quant_maps: Dict[int, Dict[str, Any]] = {}  # {lid: {name: QuantizedLayerData}}

    # FP16 chain: propagate FP16 layer outputs separately from the quantized chain.
    # Used for mixed Hessians (--fp16_hessian: H = H_quant + H_fp16) and/or fine-tuning targets (--ft_fp16_chain).
    # layer_input_fp16[lid] = FP16_layer_0( ... FP16_layer_{lid-1}(original_input) ... )
    use_fp16_chain = config.method == "eptq" and config.ft_epochs > 0 and config.ft_fp16_chain
    use_fp16_hessian = config.method == "eptq" and getattr(config, "fp16_hessian", False)
    need_fp16_chain = use_fp16_chain or use_fp16_hessian
    layer_input_fp16 = layer_input.clone() if need_fp16_chain else None
    layer_input_fp16_cur = None  # current layer's FP16 input; set each iteration

    for lid in tqdm_iter:
        layer = layers[lid]
        layer.to(device)

        # Create output directory for this layer
        layer_dir = output_dir / f"layer_{lid}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        tqdm_iter.set_description(f"Layer {lid+1}/{n_layers} (hessian)")

        # FP16 chain forward pass: compute this layer's FP16 output while weights are still FP16.
        # Needed for: (a) fp16_hessian — mix FP16 + quant chain Hessians,
        #             (b) ft_fp16_chain — use FP16 trajectory as fine-tuning target.
        layer_output_fp16 = None
        if need_fp16_chain:
            tqdm_iter.set_description(f"Layer {lid+1}/{n_layers} (fp16 output)")
            layer_output_fp16 = get_layer_output(
                layer, layer_input_fp16, batchsize=8, forward_kwargs=forward_kwargs
            )
            # Save current FP16 input before advancing (used for Hessian if fp16_hessian=True).
            layer_input_fp16_cur = layer_input_fp16
            # Advance the FP16 chain for next layer (done here while weights are FP16).
            layer_input_fp16 = layer_output_fp16

        # Compute Hessians: quant chain only, or quant chain + FP16 chain (mixed).
        hessians = get_hessians(layer, layer_input, batchsize=8, forward_kwargs=forward_kwargs)
        if use_fp16_hessian:
            hessians_fp16 = get_hessians(
                layer, layer_input_fp16_cur, batchsize=8, forward_kwargs=forward_kwargs
            )
            hessians = {k: hessians[k] + hessians_fp16[k] for k in hessians}

        # Get linear layers
        linear_layers = handler.get_linear_layers(layer)

        # Accumulate quant_data per sublayer for proxy fine-tuning
        quant_data_map = {}

        # Quantize each linear layer
        for name, linear in linear_layers.items():
            tqdm_iter.set_description(f"Layer {lid+1}/{n_layers} ({name})")

            H = hessians[name]
            W = linear.weight.data

            # Quantize (method-specific path)
            if config.method == "gptq":
                quant_data = gptq_quantize(
                    H,
                    W,
                    e8q,
                    percdamp=config.percdamp,
                    blocksize=config.blocksize,
                    actorder=config.actorder,
                )
                if linear.bias is not None:
                    quant_data.bias = linear.bias.data.clone().half()
                save_quantized_layer(quant_data, layer_dir, name)
                W_dequant = dequantize_gptq_layer(quant_data)
            else:  # eptq
                quant_data = eptq_quantize(
                    H,
                    W,
                    e8q,
                    percdamp=config.percdamp,
                    percmask=config.percmask,
                    mask_tau=config.mask_tau,
                    auto_mask=config.auto_mask,
                    auto_mask_search_pct=config.auto_mask_search_pct,
                    blocksize=config.blocksize,
                    weight_scale_norm=config.weight_scale_norm,
                    radius_candidates=config.radius_candidates,
                    layer_name=f"L{lid}.{name}",
                    qb_size=config.qb_size,
                )
                if linear.bias is not None:
                    quant_data.bias = linear.bias.data.clone().half()
                n_crit = (quant_data.Q == 32767).sum().item()
                n_total = quant_data.Q.numel()
                total_crit += n_crit
                total_blocks += n_total
                print(
                    f"  {name}: {n_crit}/{n_total} critical blocks "
                    f"({100.0 * n_crit / n_total:.4f}%)",
                    file=sys.stderr,
                )
                quant_data_map[name] = quant_data
                save_quantized_layer(quant_data, layer_dir, name)
                W_dequant = dequantize_layer(quant_data, e8q)

            # Apply quantized weight to the model for next layer calibration
            # (ensures next layer's Hessian is computed on quantized activations)
            linear.weight.data.copy_(W_dequant)

        # Proxy fine-tuning: optimize g and h to minimize decoder-layer output MSE
        _layer_quant_map = quant_data_map  # tracks the most recent quant state for E2E FT
        if config.method == "eptq" and config.ft_epochs > 0 and quant_data_map:
            tqdm_iter.set_description(f"Layer {lid+1}/{n_layers} (proxy-ft)")
            updated_map = proxy_finetune_layer(
                layer=layer,
                quant_data_map=quant_data_map,
                e8q=e8q,
                layer_input=layer_input,
                layer_output_fp16=layer_output_fp16,
                forward_kwargs=forward_kwargs,
                device=device,
                ft_epochs=config.ft_epochs,
                ft_lr=config.ft_lr,
                ft_batchsize=config.ft_batchsize,
                ft_valid_size=config.ft_valid_size,
                ft_early_stop=config.ft_early_stop,
                ft_tune_O=config.ft_tune_O,
                ft_loss=config.ft_loss,
                ft_cosine_w=config.ft_cosine_w,
                layer_name=f"L{lid}",
            )
            _layer_quant_map = updated_map
            # Re-save and update linear weights with fine-tuned g, h
            for name, new_qd in updated_map.items():
                save_quantized_layer(new_qd, layer_dir, name)
                W_ft = dequantize_layer(new_qd, e8q)
                linear_layers[name].weight.data.copy_(W_ft)

        # Collect for E2E FT (post proxy-FT state, or initial quant state if no proxy FT)
        if config.ft_e2e and config.method == "eptq" and _layer_quant_map:
            all_quant_maps[lid] = _layer_quant_map

        # Save layer-specific non-quantized params (layer norms)
        save_layer_non_quantized_params(layer, layer_dir)

        # Compute layer output for next layer
        tqdm_iter.set_description(f"Layer {lid+1}/{n_layers} (output)")
        layer_input = get_layer_output(
            layer, layer_input, batchsize=8, forward_kwargs=forward_kwargs
        )

        # Free GPU memory
        layer.cpu()

    quant_elapsed = time.perf_counter() - quant_start
    print(f"Quantization time: {quant_elapsed:.1f} sec ({quant_elapsed / 60:.1f} min)", file=sys.stderr)
    if config.method == "eptq" and total_blocks > 0:
        print(
            f"Critical weights total: {total_crit}/{total_blocks} blocks "
            f"({100.0 * total_crit / total_blocks:.4f}%)",
            file=sys.stderr,
        )

    if quant_results_file:
        if config.method == "gptq":
            label = f"gptq-b{config.bits}-g{config.group_size}"
        else:
            label = config.quantizer_type
        crit_ratio = (100.0 * total_crit / total_blocks) if total_blocks > 0 else 0.0
        write_header = not Path(quant_results_file).exists()
        with open(quant_results_file, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["model", "quantizer_type", "quant_time_sec", "crit_ratio_pct"])
            writer.writerow([config.model_name, label, f"{quant_elapsed:.1f}", f"{crit_ratio:.4f}"])

    # E2E fine-tuning: jointly optimize all g/h with CE loss across all layers
    if config.ft_e2e and config.method == "eptq" and all_quant_maps:
        print("Starting end-to-end fine-tuning...", file=sys.stderr)
        e2e_start = time.perf_counter()
        updated_all = e2e_finetune_model(
            model=model,
            layers=layers,
            all_quant_maps=all_quant_maps,
            e8q=e8q,
            calib_tokens=calib,
            device=device,
            ft_epochs=config.ft_epochs,
            ft_lr=config.ft_lr,
            ft_batchsize=config.ft_batchsize,
            ft_valid_size=config.ft_valid_size,
            ft_early_stop=config.ft_early_stop,
        )
        for lid, updated_map in updated_all.items():
            layer_dir = output_dir / f"layer_{lid}"
            for name, new_qd in updated_map.items():
                save_quantized_layer(new_qd, layer_dir, name)
        e2e_elapsed = time.perf_counter() - e2e_start
        print(f"E2E fine-tuning time: {e2e_elapsed:.1f} sec ({e2e_elapsed / 60:.1f} min)", file=sys.stderr)

    # Save model config for reconstruction
    print("Saving model config...", file=sys.stderr)
    model.config.save_pretrained(output_dir)

    # Save tokenizer for reconstruction
    print("Saving tokenizer...", file=sys.stderr)
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)

    # Save global non-quantized parameters (embeddings, final norm, lm_head)
    print("Saving global non-quantized parameters...", file=sys.stderr)
    save_global_non_quantized_params(model, handler, output_dir)

    print(f"\nQuantization complete! Output saved to: {output_dir}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="EPTQ Model Quantization")
    parser.add_argument(
        "model",
        type=str,
        help="Model name or path (e.g., meta-llama/Llama-2-7b-hf)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for quantized model",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="eptq",
        choices=["eptq", "gptq"],
        help="Quantization method (default: eptq)",
    )
    parser.add_argument(
        "--quantizer",
        type=str,
        default="e8_lattice",
        choices=["e8_lattice", "e8_d4"],
        help="EPTQ quantizer type (default: e8_lattice). Ignored for GPTQ.",
    )
    # GPTQ-specific flags
    parser.add_argument("--bits", type=int, default=4,
                        help="GPTQ bits per weight (default: 4)")
    parser.add_argument("--group_size", type=int, default=-1,
                        help="GPTQ group size along input channels (-1 = per-column)")
    parser.add_argument("--symmetric", action="store_true",
                        help="GPTQ: use symmetric quantization (zero is fixed)")
    parser.add_argument("--actorder", action="store_true",
                        help="GPTQ: reorder columns by diag(H) (act_order)")
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Damping percentage for Hessian (default: 0.01)",
    )
    parser.add_argument(
        "--percmask",
        type=float,
        default=0.002,
        help="Percentage of critical weights to mask (default: 0.002)",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="Block size for quantization (default: 128)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=128,
        help="Number of calibration samples (default: 128)",
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        default=2048,
        help="Sequence length for calibration (default: 2048)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="c4",
        choices=["c4", "wikitext2"],
        help="Calibration dataset (default: c4)",
    )
    parser.add_argument(
        "--no_weight_scale_norm",
        action="store_true",
        help="Disable weight scale normalization",
    )
    parser.add_argument(
        "--codebook_radius",
        type=float,
        default=None,
        help="Override codebook radius (default: 1.6 for e8_lattice, 1.875 for e8_d4)",
    )
    parser.add_argument(
        "--codebook_warp_alpha",
        type=float,
        default=1.0,
        help="Radial warping exponent for E8D4 codebook (1.0 = no warp, >1.0 pulls points inward)",
    )
    parser.add_argument(
        "--radius_candidates",
        type=str,
        default=None,
        help="Comma-separated list of radii to search per layer "
             "(e.g. '1.5,1.6,1.7,1.8,1.9'). When set, each layer picks "
             "the candidate minimizing Hessian-weighted reconstruction loss.",
    )
    parser.add_argument(
        "--qb_size",
        type=int,
        default=None,
        help="EPTQ row group size for group quantization (e.g. 128). "
             "None = per-column scale only (default).",
    )
    parser.add_argument(
        "--mask_tau",
        type=float,
        default=None,
        help="EPTQ RTN pre-pass threshold. Blocks with D_sum > tau are marked "
             "critical (variable count per layer). Overrides --percmask.",
    )
    parser.add_argument(
        "--auto_mask",
        action="store_true",
        default=False,
        help="EPTQ per-layer knee detection. Finds the largest log-gap in each "
             "sublayer's RTN D_sum distribution as the natural threshold. "
             "Overrides --percmask and --mask_tau.",
    )
    parser.add_argument(
        "--auto_mask_search_pct",
        type=float,
        default=0.10,
        help="Top-X%% of D_sum used as Kneedle search window for --auto_mask (default: 0.10).",
    )
    parser.add_argument(
        "--ft_epochs",
        type=int,
        default=0,
        help="Proxy fine-tuning epochs per decoder layer (0 = disabled, default: 0). "
             "Fine-tunes per-channel scales g and h to minimize decoder-layer output MSE.",
    )
    parser.add_argument(
        "--ft_lr",
        type=float,
        default=1e-4,
        help="Adam learning rate for proxy fine-tuning (default: 1e-4)",
    )
    parser.add_argument(
        "--ft_batchsize",
        type=int,
        default=8,
        help="Batch size during proxy fine-tuning (default: 8)",
    )
    parser.add_argument(
        "--ft_valid_size",
        type=int,
        default=0,
        help="Number of calibration samples held out for validation and early stopping "
             "during proxy fine-tuning (0 = disabled, default: 0). "
             "QTiP uses 128 out of 384 total samples.",
    )
    parser.add_argument(
        "--ft_early_stop",
        type=int,
        default=5,
        help="Early stopping patience: stop fine-tuning if validation loss does not "
             "improve for this many epochs (default: 5). Only active when --ft_valid_size > 0.",
    )
    parser.add_argument(
        "--ft_tune_O",
        action="store_true",
        default=False,
        help="Also fine-tune the critical-weight values O (in addition to g and h).",
    )
    parser.add_argument(
        "--ft_loss",
        type=str,
        default="mse",
        choices=["mse", "cosine", "mse+cosine"],
        help="Loss function for proxy fine-tuning (default: mse).",
    )
    parser.add_argument(
        "--ft_cosine_w",
        type=float,
        default=1.0,
        help="Weight for cosine term when --ft_loss=mse+cosine (default: 1.0).",
    )
    parser.add_argument(
        "--fp16_hessian",
        action="store_true",
        default=False,
        help="Mix FP16 chain and quantized chain Hessians (H = H_quant + H_fp16). "
             "Doubles the effective calibration samples by combining both distributions.",
    )
    parser.add_argument(
        "--ft_fp16_chain",
        action="store_true",
        default=False,
        help="Use FP16 chain targets for proxy fine-tuning: each layer's target is the "
             "FP16 layer output computed from FP16 previous-layer outputs, tracking the "
             "true FP16 trajectory. Default (off): target = FP16 layer on quantized input.",
    )
    parser.add_argument(
        "--ft_e2e",
        action="store_true",
        default=False,
        help="Run end-to-end CE-loss fine-tuning after all layers are quantized. "
             "Jointly optimizes all g/h across all layers with next-token CE loss. "
             "Requires all layers to fit in GPU memory simultaneously. "
             "Uses --ft_epochs, --ft_lr, --ft_batchsize, --ft_valid_size, --ft_early_stop.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--quant_results_file",
        type=str,
        default=None,
        help="CSV file to append quantization timing results to",
    )

    args = parser.parse_args()

    config = QuantizationConfig(
        model_name=args.model,
        output_dir=args.output_dir,
        method=args.method,
        quantizer_type=args.quantizer,
        percdamp=args.percdamp,
        percmask=args.percmask,
        blocksize=args.blocksize,
        n_samples=args.n_samples,
        seqlen=args.seqlen,
        seed=args.seed,
        calib_dataset=args.calib_dataset,
        weight_scale_norm=not args.no_weight_scale_norm,
        codebook_radius=args.codebook_radius,
        codebook_warp_alpha=args.codebook_warp_alpha,
        radius_candidates=(
            [float(x) for x in args.radius_candidates.split(",")]
            if args.radius_candidates else None
        ),
        qb_size=args.qb_size,
        mask_tau=args.mask_tau,
        auto_mask=args.auto_mask,
        auto_mask_search_pct=args.auto_mask_search_pct,
        ft_epochs=args.ft_epochs,
        ft_lr=args.ft_lr,
        ft_batchsize=args.ft_batchsize,
        ft_valid_size=args.ft_valid_size,
        ft_early_stop=args.ft_early_stop,
        ft_tune_O=args.ft_tune_O,
        ft_loss=args.ft_loss,
        ft_cosine_w=args.ft_cosine_w,
        fp16_hessian=args.fp16_hessian,
        ft_fp16_chain=args.ft_fp16_chain,
        ft_e2e=args.ft_e2e,
        device=args.device,
        bits=args.bits,
        group_size=args.group_size,
        symmetric=args.symmetric,
        actorder=args.actorder,
    )

    quantize_model(config, quant_results_file=args.quant_results_file)


if __name__ == "__main__":
    main()
