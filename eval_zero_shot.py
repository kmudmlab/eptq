"""Run lm-eval-harness zero-shot tasks on an EPTQ/GPTQ-quantized model.

Loads the quantized model directly into GPU memory via build_model_from_quantized
(no fp16 reconstruction to disk), wraps with lm_eval HFLM, and saves the results
in the same JSON shape as the standard lm_eval CLI output (so collect_lmeval.py
can pick it up).
"""

import argparse
import json
from pathlib import Path

import torch
import lm_eval
from lm_eval.models.huggingface import HFLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantized_dir", type=str, required=True,
                        help="Quantized model dir (EPTQ or GPTQ)")
    parser.add_argument("--tasks", type=str,
                        default="piqa,hellaswag,arc_easy,arc_challenge,winogrande")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device_map", type=str, default=None,
                        help="e.g. 'auto' for multi-GPU dispatch")
    parser.add_argument("--batch_size", type=str, default="auto")
    parser.add_argument("--output_path", type=str, required=True,
                        help="JSON file to save results to")
    parser.add_argument("--num_fewshot", type=int, default=0)
    args = parser.parse_args()

    from eptq.reconstruct import build_model_from_quantized
    print(f"Dequantizing model from {args.quantized_dir}...")
    model, tokenizer = build_model_from_quantized(
        args.quantized_dir, device=args.device, device_map=args.device_map
    )

    # batch_size: int if numeric, else "auto"
    bsz = int(args.batch_size) if args.batch_size.isdigit() else args.batch_size

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=bsz)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"Running tasks: {tasks}")

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        log_samples=False,
    )

    # Normalize to the same shape as standard lm_eval CLI output so that
    # collect_lmeval.py can read it.
    out_payload = {
        "results": results.get("results", {}),
        "model_name": args.quantized_dir,
        "n-shot": results.get("n-shot"),
        "config": results.get("config", {}),
        "versions": results.get("versions"),
    }

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(out_payload, f, indent=2, default=str)
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
