# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EPTQ (E8 Lattice Post-Training Quantization) is a PyTorch-based library for quantizing large language models (LLMs) using E8 and D4 lattice-based vector quantization. It supports OPT and LLaMA model families and operates layer-wise to be memory-efficient.

## Commands

```bash
# Install in development mode
pip install -e ".[dev]" --no-build-isolation

# Build CUDA inference kernel (ahead-of-time; JIT fallback is automatic)
python3 setup_cuda.py build_ext --inplace

# Run quantization
python -m eptq.quantize facebook/opt-125m --output_dir ./quantized_opt-125m --percmask 0.002

# Run quantization with E8-D4 variant
python -m eptq.quantize facebook/opt-125m --output_dir ./quantized_opt-125m --quantizer e8_d4

# Reconstruct (dequantize) a model back to fp16
python -m eptq.reconstruct --quantized_dir ./quantized_opt-125m --output_dir ./reconstructed_opt-125m

# Format code
black eptq/ --line-length 100 --target-version py310
isort eptq/ --profile black --line-length 100
```

## Architecture

### Core Data Structures

- **`QuantizationConfig`** (`config.py`): All quantization hyperparameters — quantizer type, Hessian damping, critical-weight mask ratio, block size, calibration settings.
- **`QuantizedLayerData`** (`config.py`): Stores one quantized linear layer: `Q` (int16 indices, 65535 = masked critical weight), `g`/`h` (row/column scale factors), `O` (original values of masked weights), and optional `bias`.

### Quantization Pipeline (`quantize.py`)

1. Load model + C4 calibration data
2. For each transformer layer:
   - Compute Hessians from layer inputs via forward hooks (`utils/hessian.py`)
   - For each linear sublayer: call `eptq_quantize()` → save result
   - Compute next layer's input via `get_layer_output()`
3. Save per-layer quantized files, embeddings, final norm, LM head, codebook, config, and tokenizer

### Key Algorithm (`algorithms/eptq.py`)

- **`comp_gh()`**: Computes row (`g`) and column (`h`) scale factors via iterative normalization for numerical stability.
- **`eptq_quantize()`**: Main algorithm — normalizes weights using `g`/`h`, computes Hessian inverse via Cholesky decomposition (with damping), runs an initial quantization pass to identify critical weights, then does column-wise quantization with error propagation. Critical weights are marked with index 65535 and stored at full precision in `O`.
- **`dequantize_layer()`**: Reconstructs fp16 weights from `QuantizedLayerData` by codebook lookup then applying `g`/`h` scales and overwriting masked positions.

### Quantizers (`quantizers/`)

- **`E8LatticeQuantizer`**: Generates a 65535-point codebook from 6 shells of the E8 lattice; nearest-neighbor search for encoding, codebook lookup for decoding.
- **`E8D4Quantizer`**: Decomposes each 8D weight vector into two 4D D4 vectors with 4 code types × 128 codes each; uses a proxy-distance method for fast nearest-neighbor search.

### Model Handlers (`models/`)

Abstract base `ModelHandler` with architecture-specific subclasses. Key difference:

- **`OPTHandler`**: Layers at `model.model.decoder.layers`; passes token + position embeddings as first-layer input.
- **`LlamaHandler`**: Layers at `model.model.layers`; uses a "Catcher" hook to intercept the first layer's inputs (handles positional encoding internals).

To add a new model architecture, subclass `ModelHandler` and implement `get_layers()`, `get_first_layer_input()`, `get_layer_output()`, and `get_non_quantized_params()`.

### Inference Kernels (`kernels/`)

`EPTQLinear` (`kernels/linear.py`) is a drop-in replacement for `nn.Linear` that runs inference directly from quantized format without materializing the full fp16 weight matrix.

**CUDA kernels** (`csrc/eptq_kernels.cu`):
- **`e8_matvec_kernel`** / **`e8d4_matvec_kernel`**: Grid `(M, B)`, Block `(256)`. Each block computes 8 output elements for one batch item by iterating over all N input columns, looking up the codebook, and accumulating with warp+shared-mem reduction.
- **`masked_correction_kernel`**: Adds critical-weight contributions via `atomicAdd` on a float32 buffer.
- **`f32_to_f16_kernel`**: Final fp32→fp16 conversion with optional bias add.

**Usage**:
```python
from eptq.kernels.patch import load_quantized_model

model, tokenizer = load_quantized_model("./quantized_opt-125m", device="cuda")
# model now has EPTQLinear layers instead of nn.Linear; use normally
inputs = tokenizer("Hello", return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=50)
```

**Key detail**: `EPTQLinear._codebook` is a shared reference (not a buffer) so the codebook is stored only once in VRAM across all layers (~1 MB for E8, ~4 KB for E8D4). Moving with `model.to(device)` is handled by `_apply()` override.

**Build**: The extension JIT-compiles on first use. Run `python3 setup_cuda.py build_ext --inplace` for faster startup in production.

### Quantized Output Format

Saved under `<output_dir>/`:
```
layer_N/<layer_name>.pt   # QuantizedLayerData per linear layer
global_params.pt          # embeddings, final norm, LM head
layer_N/layer_params.pt   # layer norms and other non-quantized layer params
codebook.pt               # quantizer codebook
config.json               # QuantizationConfig + model name
tokenizer/                # HuggingFace tokenizer files
```
