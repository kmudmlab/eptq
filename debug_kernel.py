"""Debug script: test real quantized data with CUDA_LAUNCH_BLOCKING=1."""
import os, sys
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from eptq.kernels.patch import load_quantized_model

device = "cuda"

for quant_dir, qtype in [
    ("./quantized/e8_lattice/opt-125m", "e8_lattice"),
    ("./quantized/e8_d4/opt-125m",      "e8_d4"),
]:
    print(f"\n=== {qtype} ===")
    model, tokenizer = load_quantized_model(quant_dir, device=device)

    # Inspect first EPTQLinear layer
    from eptq.kernels.linear import EPTQLinear
    for name, module in model.named_modules():
        if isinstance(module, EPTQLinear):
            print(f"  First layer: {name}")
            print(f"    Q:         {module.Q.shape}  device={module.Q.device}")
            print(f"    O_vals:    {module.O_vals.shape}  device={module.O_vals.device}")
            print(f"    O_col:     {module.O_col.shape}   device={module.O_col.device}")
            print(f"    O_row_ptr: {module.O_row_ptr.shape} device={module.O_row_ptr.device}")
            print(f"    codebook:  {module._codebook.shape} device={module._codebook.device}")

            # Single forward pass
            x = torch.randn(2, module.in_features, dtype=torch.float16, device=device)
            out = module(x)
            torch.cuda.synchronize()
            print(f"    forward OK: {out.shape}")
            break

    print(f"  Full model forward...")
    inputs = tokenizer("Hello world", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    torch.cuda.synchronize()
    print(f"  logits shape: {out.logits.shape}  OK")
