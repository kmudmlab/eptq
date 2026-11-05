"""Single forward pass for Nsight Compute profiling."""

import torch
from eptq.kernels.patch import load_quantized_model

QUANTIZED_DIR = "/home/hmpark/eptq_new/quantized/e8_d4/opt-125m"

model, tokenizer = load_quantized_model(QUANTIZED_DIR, device="cuda")

inputs = tokenizer("Hello, my name is", return_tensors="pt").to("cuda")

# warm-up
with torch.no_grad():
    for _ in range(3):
        model(**inputs)

torch.cuda.synchronize()

# profiling target
with torch.no_grad():
    model(**inputs)

torch.cuda.synchronize()
print("Done")
