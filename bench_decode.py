"""Decode throughput benchmark matching qtip evaluation protocol.

Key design:
- Prefill is excluded from timing (only decode loop is measured)
- decode_one compiled with torch.compile(mode="max-autotune")
- StaticCache managed manually and reset between runs
- 4 prompts, results averaged
- Warmup: 2 short generation runs before compile, 2 more after compile
- max_new_tokens=256 per run
- temperature=0.6, top_k=5 sampling
- JSON output
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

torch.set_grad_enabled(False)

PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In the beginning, the universe was created.",
    "It was the best of times, it was the worst of times.",
    "To be or not to be, that is the question.",
]

WARMUP_PROMPT = "Once upon a time in a land far away,"


def sample(logits, temperature: float = 0.6, top_k: int = 5):
    logits = logits[:, -1] / max(temperature, 1e-5)
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits = torch.where(logits < v[:, [-1]], -float("Inf"), logits)
    probs = F.softmax(logits, dim=-1)
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


@torch.no_grad()
def decode_one(model, cur_token, past_kv, cache_position):
    out = model(cur_token, past_key_values=past_kv, cache_position=cache_position)
    return sample(out.logits)


@torch.no_grad()
def run_generation(model, tokenizer, text, max_new_tokens, past_kv, device):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    seq_length = inputs["input_ids"].shape[1]
    cache_position = torch.arange(seq_length, device=device)

    # Prefill (not timed)
    out = model(**inputs, past_key_values=past_kv, cache_position=cache_position)
    next_token = sample(out.logits)
    cache_position = torch.tensor([seq_length + 1], device=device)
    torch.cuda.synchronize()

    # Decode loop (timed)
    t0 = time.perf_counter()
    for _ in range(1, max_new_tokens):
        next_token = decode_one(model, next_token.clone(), past_kv, cache_position)
        cache_position += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    past_kv.reset()
    return (max_new_tokens - 1) / elapsed


def benchmark(model, tokenizer, past_kv, max_new_tokens, device):
    # Warmup: 2 short runs
    for prompt in [WARMUP_PROMPT, PROMPTS[0]]:
        run_generation(model, tokenizer, prompt, 32, past_kv, device)

    total_tps = 0.0
    for prompt in PROMPTS:
        tps = run_generation(model, tokenizer, prompt, max_new_tokens, past_kv, device)
        total_tps += tps
    return total_tps / len(PROMPTS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--quantized", action="store_true",
                        help="Load via EPTQ CUDA kernels")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--compile", action="store_true",
                        help="Apply torch.compile on decode_one")
    parser.add_argument("--fullgraph", action="store_true",
                        help="Use fullgraph=True in torch.compile (default: False)")
    parser.add_argument("--no_tf32", action="store_true")
    parser.add_argument("--results_file", type=str, default=None)
    parser.add_argument("--label", type=str, default=None)
    args = parser.parse_args()

    device = "cuda"

    if not args.no_tf32:
        torch.set_float32_matmul_precision("high")

    if args.quantized:
        from eptq.kernels.patch import load_quantized_model
        print(f"Loading quantized model from {args.model_dir}...")
        model, tokenizer = load_quantized_model(args.model_dir, device=device)
    else:
        print(f"Loading fp16 model from {args.model_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            torch_dtype="auto",
            attn_implementation="sdpa",
        ).to(device)

    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    # Detect model dtype for StaticCache
    model_dtype = next(
        (p.dtype for p in model.parameters() if p.dtype in (torch.float16, torch.bfloat16)),
        torch.float16,
    )

    past_kv = StaticCache(
        model.config,
        max_batch_size=1,
        max_cache_len=args.max_new_tokens * 2 + 64,
        device=device,
        dtype=model_dtype,
    )

    label = args.label or ("quantized" if args.quantized else "fp16")

    # No-compile run
    print(f"\n=== No-compile ===")
    tps_no_compile = benchmark(model, tokenizer, past_kv, args.max_new_tokens, device)
    print(f"  {tps_no_compile:.1f} tok/s")

    tps_compile = None
    if args.compile:
        global decode_one
        print(f"\nCompiling decode_one (max-autotune, fullgraph=False)...")
        decode_one = torch.compile(decode_one, mode="max-autotune", fullgraph=args.fullgraph)

        # Warmup after compile
        for prompt in [WARMUP_PROMPT, PROMPTS[0]]:
            run_generation(model, tokenizer, prompt, 32, past_kv, device)

        print(f"\n=== Compile ===")
        tps_compile = benchmark(model, tokenizer, past_kv, args.max_new_tokens, device)
        print(f"  {tps_compile:.1f} tok/s")

    best_tps = tps_compile if tps_compile is not None else tps_no_compile
    ms_per_tok = 1000 / best_tps

    print(f"\n=== Decode throughput ===")
    print(f"  Model:       {args.model_dir}")
    print(f"  Label:       {label}")
    print(f"  dtype:       {model_dtype}")
    print(f"  Compile:     {args.compile}")
    print(f"  No-compile:  {tps_no_compile:.1f} tok/s")
    if tps_compile:
        print(f"  Compile:     {tps_compile:.1f} tok/s")

    if args.results_file:
        out_path = Path(args.results_file)
        results = json.loads(out_path.read_text()) if out_path.exists() else {}
        entry = {
            "model_dir": args.model_dir,
            "dtype": str(model_dtype),
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "tokens_per_sec_no_compile": round(tps_no_compile, 2),
        }
        if tps_compile is not None:
            entry["tokens_per_sec_compile"] = round(tps_compile, 2)
        results[label] = entry
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  Saved to {args.results_file}")


if __name__ == "__main__":
    main()
