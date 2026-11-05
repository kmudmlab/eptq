"""Standalone end-to-end fine-tuning for an already-quantized EPTQ model.

Loads the saved .pt files from a prior quantize.py run, jointly optimizes
all g/h across all layers with next-token CE loss, and writes the updated
.pt files back (in-place or to a new directory).

Usage
-----
# In-place (overwrites g/h in the existing quantized_dir)
python -m eptq.e2e_finetune --quantized_dir ./quantized_llama2_13b \\
    --ft_epochs 10 --ft_lr 1e-4 --ft_batchsize 1

# To a new directory (copies all files, then updates only the .pt layers)
python -m eptq.e2e_finetune --quantized_dir ./quantized_llama2_13b \\
    --output_dir ./quantized_llama2_13b_e2eft \\
    --ft_epochs 10 --ft_lr 1e-4 --ft_batchsize 1
"""

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .algorithms.proxy_finetune import e2e_finetune_model
from .config import QuantizedLayerData
from .models import ModelHandler
from .reconstruct import (
    get_model_handler_by_type,
    load_quantization_config,
    load_quantizer,
)
from .utils.calibration import get_c4, get_wikitext2


def _get_model_handler(model) -> ModelHandler:
    return get_model_handler_by_type(ModelHandler.detect_model_type(model))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apply end-to-end CE-loss fine-tuning to an already-quantized EPTQ model. "
            "Loads existing .pt files, optimizes all g/h jointly, and re-saves them."
        )
    )
    parser.add_argument(
        "--quantized_dir",
        type=str,
        required=True,
        help="Directory produced by eptq.quantize (contains layer_N/*.pt and quantizer.pt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="If given, copy quantized_dir here first then write updated .pt files there. "
             "Default: overwrite .pt files in-place inside --quantized_dir.",
    )
    parser.add_argument(
        "--ft_epochs",
        type=int,
        default=None,
        help="Number of E2E FT epochs (default: quant_config ft_epochs, else 10)",
    )
    parser.add_argument(
        "--ft_lr",
        type=float,
        default=None,
        help="Adam learning rate (default: quant_config ft_lr, else 1e-4)",
    )
    parser.add_argument(
        "--ft_batchsize",
        type=int,
        default=None,
        help="Batch size in sequences (default: quant_config ft_batchsize, else 1)",
    )
    parser.add_argument(
        "--ft_valid_size",
        type=int,
        default=None,
        help="Validation samples for early stopping; 0 = disabled "
             "(default: quant_config ft_valid_size, else 0)",
    )
    parser.add_argument(
        "--ft_early_stop",
        type=int,
        default=None,
        help="Early stopping patience in epochs "
             "(default: quant_config ft_early_stop, else 5)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Calibration sequences to use (default: from quant_config)",
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        default=None,
        help="Calibration sequence length (default: from quant_config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for calibration sampling (default: from quant_config)",
    )
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default=None,
        choices=["c4", "wikitext2"],
        help="Calibration dataset (default: from quant_config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (default: cuda)",
    )
    args = parser.parse_args()

    quantized_dir = Path(args.quantized_dir)

    # --- Load quant_config for defaults ---
    quant_config = load_quantization_config(quantized_dir)
    if quant_config.get("method", "eptq") != "eptq":
        print("ERROR: E2E FT is only supported for EPTQ models.", file=sys.stderr)
        sys.exit(1)

    def _pick(cli_val, config_key, fallback):
        if cli_val is not None:
            return cli_val
        return quant_config.get(config_key, fallback)

    ft_epochs     = _pick(args.ft_epochs,     "ft_epochs",     10)
    ft_lr         = _pick(args.ft_lr,         "ft_lr",         1e-4)
    ft_batchsize  = _pick(args.ft_batchsize,  "ft_batchsize",  1)
    ft_valid_size = _pick(args.ft_valid_size, "ft_valid_size", 0)
    ft_early_stop = _pick(args.ft_early_stop, "ft_early_stop", 5)
    n_samples     = _pick(args.n_samples,     "n_samples",     128)
    seqlen        = _pick(args.seqlen,        "seqlen",        2048)
    seed          = _pick(args.seed,          "seed",          42)
    calib_dataset = _pick(args.calib_dataset, "calib_dataset", "c4")
    model_name    = quant_config["model_name"]
    device        = args.device

    # --- Determine output directory ---
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        if output_dir.resolve() != quantized_dir.resolve():
            print(f"Copying {quantized_dir} → {output_dir} ...", file=sys.stderr)
            if output_dir.exists():
                shutil.rmtree(output_dir)
            shutil.copytree(quantized_dir, output_dir)
        else:
            output_dir = quantized_dir
    else:
        output_dir = quantized_dir

    print("E2E fine-tuning settings:", file=sys.stderr)
    print(f"  source        = {quantized_dir}", file=sys.stderr)
    print(f"  output        = {output_dir}", file=sys.stderr)
    print(f"  model         = {model_name}", file=sys.stderr)
    print(f"  ft_epochs     = {ft_epochs}", file=sys.stderr)
    print(f"  ft_lr         = {ft_lr}", file=sys.stderr)
    print(f"  ft_batchsize  = {ft_batchsize}", file=sys.stderr)
    print(f"  ft_valid_size = {ft_valid_size}", file=sys.stderr)
    print(f"  ft_early_stop = {ft_early_stop}", file=sys.stderr)
    print(f"  n_samples     = {n_samples}", file=sys.stderr)
    print(f"  seqlen        = {seqlen}", file=sys.stderr)
    print(f"  calib_dataset = {calib_dataset}", file=sys.stderr)

    # --- Load original FP16 model (architecture + non-linear weights) ---
    print(f"\nLoading model {model_name}...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", use_safetensors=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    handler = _get_model_handler(model)
    layers  = handler.get_layers(model)
    n_layers = len(layers)

    # --- Load quantizer ---
    print("Loading quantizer...", file=sys.stderr)
    e8q = load_quantizer(quantized_dir, device=device)

    # --- Build all_quant_maps from saved .pt files ---
    print(f"Loading quantized layer data ({n_layers} layers)...", file=sys.stderr)
    all_quant_maps: Dict[int, Dict[str, QuantizedLayerData]] = {}
    for lid, layer in enumerate(layers):
        layer_dir = quantized_dir / f"layer_{lid}"
        if not layer_dir.exists():
            continue
        quant_map = {}
        for name in handler.get_linear_layers(layer):
            pt_path = layer_dir / f"{name}.pt"
            if pt_path.exists():
                quant_map[name] = QuantizedLayerData.load(pt_path)
        if quant_map:
            all_quant_maps[lid] = quant_map
    print(f"  Loaded {len(all_quant_maps)}/{n_layers} layers.", file=sys.stderr)

    # --- Calibration data ---
    print("Loading calibration data...", file=sys.stderr)
    if calib_dataset == "wikitext2":
        calib = get_wikitext2(n_samples, seed, seqlen, tokenizer)
    else:
        calib = get_c4(n_samples, seed, seqlen, tokenizer)

    # --- Run E2E FT ---
    print("\nStarting E2E fine-tuning...", file=sys.stderr)
    start = time.perf_counter()
    updated_all = e2e_finetune_model(
        model=model,
        layers=layers,
        all_quant_maps=all_quant_maps,
        e8q=e8q,
        calib_tokens=calib,
        device=device,
        ft_epochs=ft_epochs,
        ft_lr=ft_lr,
        ft_batchsize=ft_batchsize,
        ft_valid_size=ft_valid_size,
        ft_early_stop=ft_early_stop,
    )
    elapsed = time.perf_counter() - start
    print(f"E2E FT time: {elapsed:.1f} sec ({elapsed / 60:.1f} min)", file=sys.stderr)

    # --- Save updated .pt files ---
    print("Saving updated layers...", file=sys.stderr)
    n_saved = 0
    for lid, updated_map in updated_all.items():
        layer_dir = output_dir / f"layer_{lid}"
        for name, new_qd in updated_map.items():
            new_qd.save(layer_dir / f"{name}.pt")
            n_saved += 1
    print(f"Saved {n_saved} layer files to {output_dir}", file=sys.stderr)

    # Record E2E FT settings in quant_config.json
    import json
    quant_config.update({
        "ft_e2e": True,
        "ft_epochs": ft_epochs,
        "ft_lr": ft_lr,
        "ft_batchsize": ft_batchsize,
        "ft_valid_size": ft_valid_size,
        "ft_early_stop": ft_early_stop,
    })
    with open(output_dir / "quant_config.json", "w") as f:
        json.dump(quant_config, f, indent=2)

    print(f"\nDone. Results saved to: {output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
