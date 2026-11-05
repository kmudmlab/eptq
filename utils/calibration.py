"""Calibration data loading utilities."""

import os
import random
import sys
from itertools import islice
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm


def get_wikitext2(
    nsamples: int,
    seed: int,
    seqlen: int,
    tokenizer,
    cache_dir: Optional[str] = None,
) -> torch.Tensor:
    """Load calibration data from the wikitext2 train split."""
    cache_path = f"calib_wikitext2_{nsamples}_{seqlen}_{seed}_v{tokenizer.vocab_size}.pt"
    if cache_dir:
        cache_path = os.path.join(cache_dir, cache_path)

    if os.path.exists(cache_path):
        print("Loading calib from file...", file=sys.stderr)
        return torch.load(cache_path, weights_only=True)

    print("Loading wikitext2 dataset...", file=sys.stderr)
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(traindata["text"])
    enc = tokenizer(text, return_tensors="pt").input_ids[0]

    random.seed(seed)
    calib = []
    for _ in tqdm(range(nsamples), desc="Sampling calibration data"):
        i = random.randint(0, enc.shape[0] - seqlen - 1)
        calib.append(enc[i : i + seqlen].unsqueeze(0))

    calib = torch.cat(calib, dim=0)
    torch.save(calib, cache_path)
    return calib


def get_c4(
    nsamples: int,
    seed: int,
    seqlen: int,
    tokenizer,
    cache_dir: Optional[str] = None,
) -> torch.Tensor:
    """
    Load calibration data from the C4 dataset.

    Args:
        nsamples: Number of samples to load
        seed: Random seed for reproducibility
        seqlen: Sequence length for each sample
        tokenizer: Tokenizer to use for encoding
        cache_dir: Directory to cache the calibration data

    Returns:
        Tensor of token ids of shape (nsamples, seqlen)
    """
    cache_path = f"calib_c4_{nsamples}_{seqlen}_{seed}_v{tokenizer.vocab_size}.pt"
    if cache_dir:
        cache_path = os.path.join(cache_dir, cache_path)

    if os.path.exists(cache_path):
        print("Loading calib from file...", file=sys.stderr)
        calib = torch.load(cache_path, weights_only=True)
        return calib

    print("Loading C4 dataset...", file=sys.stderr)
    traindata = load_dataset("allenai/c4", "en", split="train", streaming=True)
    traindata = list(islice(traindata, 356445))

    random.seed(seed)

    calib = []
    for _ in tqdm(range(nsamples), desc="Sampling calibration data"):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        calib.append(inp)

    calib = torch.cat(calib, dim=0)

    torch.save(calib, cache_path)

    return calib
