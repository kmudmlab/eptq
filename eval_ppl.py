"""Perplexity evaluation on WikiText-2 / PTB / C4 for a HuggingFace causal LM."""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DATASETS = ("wikitext2", "ptb", "c4")


def _get_test_tokens(dataset: str, tokenizer) -> torch.Tensor:
    """
    Tokenize the evaluation corpus following AWQ/GPTQ conventions.

    Returns an (1, N) tensor of token ids.
    """
    if dataset == "wikitext2":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(data["text"])
        return tokenizer(text, return_tensors="pt").input_ids

    if dataset == "ptb":
        data = load_dataset("allenai/paloma", "ptb", split="val")
        lines = [l for l in data[0]["text"].split("\n") if l.strip()]
        text = "\n\n".join(lines)
        return tokenizer(text, return_tensors="pt").input_ids

    if dataset == "c4":
        valdata = load_dataset(
            "allenai/c4", "en", split="validation",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            verification_mode="no_checks",
        )
        random.seed(0)
        chunks = []
        while len(chunks) < 256:
            i = random.randint(0, len(valdata) - 1)
            ids = tokenizer(valdata[i]["text"], return_tensors="pt").input_ids
            if ids.shape[1] >= 2048:
                start = random.randint(0, ids.shape[1] - 2048)
                chunks.append(ids[:, start:start + 2048])
        return torch.hstack(chunks)

    raise ValueError(f"Unknown dataset {dataset!r}; expected one of {DATASETS}")


def _load_model(model_dir, quantized, device, device_map, quantized_dir):
    """Load model and tokenizer once."""
    if quantized_dir is not None:
        from eptq.reconstruct import build_model_from_quantized
        print(f"Dequantizing model in memory from {quantized_dir}...")
        model, tokenizer = build_model_from_quantized(
            quantized_dir, device=device, device_map=device_map
        )
        eval_device = next(model.parameters()).device
    elif quantized:
        from eptq.kernels.patch import load_quantized_model
        print(f"Loading quantized model (CUDA kernels) from {model_dir}...")
        model, tokenizer = load_quantized_model(model_dir, device=device)
        eval_device = device
    elif device_map is not None:
        print(f"Loading model from {model_dir} (device_map={device_map})...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.float16, device_map=device_map
        )
        model.eval()
        eval_device = next(model.parameters()).device
    else:
        print(f"Loading model from {model_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.float16
        )
        model.eval().to(device)
        eval_device = device
    return model, tokenizer, eval_device


def _compute_ppl(model, tokenizer, eval_device, dataset, seq_len, stride,
                 results_file, label) -> float:
    """Compute PPL for one dataset on an already-loaded model."""
    print(f"\nLoading {dataset} evaluation tokens...")
    input_ids = _get_test_tokens(dataset, tokenizer)

    total_len = input_ids.size(1)
    total_nll = 0.0
    total_tokens = 0
    total_time = 0.0
    timed_chunks = 0

    for i, begin in enumerate(tqdm(range(0, total_len - seq_len, stride), desc=f"PPL [{dataset}]")):
        end = begin + seq_len
        chunk = input_ids[:, begin:end].to(eval_device)

        target_len = seq_len if begin == 0 else stride
        target_ids = chunk.clone()
        target_ids[:, :-target_len] = -100

        if i > 0:
            for d in range(torch.cuda.device_count()):
                torch.cuda.synchronize(d)
            t0 = time.perf_counter()

        with torch.no_grad():
            loss = model(chunk, labels=target_ids).loss

        if i > 0:
            for d in range(torch.cuda.device_count()):
                torch.cuda.synchronize(d)
            total_time += time.perf_counter() - t0
            timed_chunks += 1

        total_nll += loss.item() * target_len
        total_tokens += target_len

    ppl = torch.exp(torch.tensor(total_nll / total_tokens)).item()

    avg_latency = total_time / timed_chunks if timed_chunks > 0 else 0.0
    throughput = (timed_chunks * stride) / total_time if total_time > 0 else 0.0

    print(f"{dataset} Perplexity:   {ppl:.4f}")
    print(f"Throughput:            {throughput:.1f} tokens/sec")
    print(f"Avg latency per chunk: {avg_latency * 1000:.1f} ms  (stride={stride} tokens, warm-up excluded)")

    if results_file:
        record = {
            "model": label,
            "dataset": dataset,
            "seq_len": seq_len,
            "stride": stride,
            "ppl": round(ppl, 4),
            "throughput_tok_per_sec": round(throughput, 1),
            "avg_latency_ms": round(avg_latency * 1000, 1),
        }
        results_path = Path(results_file)
        if results_path.suffix == ".json":
            with open(results_path, "w") as f:
                json.dump(record, f, indent=2)
        else:
            write_header = not results_path.exists()
            with open(results_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(list(record.keys()))
                writer.writerow(list(record.values()))
        print(f"Result saved to {results_path}")

    return ppl


def evaluate_ppl(
    model_dir: str = None,
    seq_len: int = 2048,
    stride: int = 2048,
    device: str = "cuda",
    quantized: bool = False,
    device_map: str = None,
    results_file: str = None,
    quantized_dir: str = None,
    dataset: str = None,           # single dataset (legacy)
    datasets: tuple = None,        # multiple datasets (new)
) -> dict:
    """Load model once, evaluate PPL on all requested datasets.

    Returns dict of {dataset_name: ppl}.
    """
    if datasets is None:
        datasets = (dataset,) if dataset else DATASETS

    model, tokenizer, eval_device = _load_model(
        model_dir, quantized, device, device_map, quantized_dir
    )
    label = model_dir or quantized_dir

    results = {}
    for ds in datasets:
        results[ds] = _compute_ppl(
            model, tokenizer, eval_device, ds, seq_len, stride, results_file, label
        )

    if len(results) > 1:
        print("\n=== PPL Summary ===")
        for ds, ppl in results.items():
            print(f"  {ds:12s}: {ppl:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate perplexity on WikiText-2 / PTB / C4")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--quantized_dir", type=str, default=None)
    parser.add_argument("--results_file", type=str, default=None)
    parser.add_argument(
        "--dataset", type=str, nargs="+", choices=DATASETS, default=None,
        help="Dataset(s) to evaluate (default: all three — wikitext2 ptb c4)",
    )
    args = parser.parse_args()

    evaluate_ppl(
        args.model_dir, args.seq_len, args.stride, args.device,
        args.quantized, args.device_map, args.results_file,
        quantized_dir=args.quantized_dir,
        datasets=tuple(args.dataset) if args.dataset else None,
    )


if __name__ == "__main__":
    main()
