"""EPTQ CUDA inference kernels — registered as torch.library.custom_op for compile compatibility."""

import os
from pathlib import Path

import torch

_ext = None
_CSRC = Path(__file__).parent.parent / "csrc"


def _load():
    global _ext
    if _ext is not None:
        return _ext

    # Try pre-built AOT extension (installed via setup_cuda.py)
    try:
        import _eptq_cuda
        _ext = _eptq_cuda
        return _ext
    except ImportError:
        pass

    # JIT compile with torch.utils.cpp_extension.load
    from torch.utils.cpp_extension import load

    _ext = load(
        name="_eptq_cuda_jit",
        sources=[
            str(_CSRC / "eptq_kernels.cu"),
            str(_CSRC / "binding.cpp"),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "--expt-relaxed-constexpr",
            "-std=c++17",
        ],
        verbose=False,
    )
    return _ext


# ---------------------------------------------------------------------------
# Register as torch.library.custom_op so torch.compile sees these as
# first-class ops (no graph break) and can wrap them in CUDA graphs.
# ---------------------------------------------------------------------------

@torch.library.custom_op("eptq::e8_forward", mutates_args=())
def e8_forward(
    Q: torch.Tensor,
    codebook: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    x: torch.Tensor,
    O_vals: torch.Tensor,
    O_col: torch.Tensor,
    O_row_ptr: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return _load().e8_forward(Q, codebook, g, h, x, O_vals, O_col, O_row_ptr, bias)


@e8_forward.register_fake
def _e8_forward_fake(Q, codebook, g, h, x, O_vals, O_col, O_row_ptr, bias):
    B = x.shape[0]
    M8 = Q.shape[0] * 8
    return x.new_empty(B, M8, dtype=torch.float16)


@torch.library.custom_op("eptq::e8d4_forward", mutates_args=())
def e8d4_forward(
    Q: torch.Tensor,
    codes: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    x: torch.Tensor,
    O_vals: torch.Tensor,
    O_col: torch.Tensor,
    O_row_ptr: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return _load().e8d4_forward(Q, codes, g, h, x, O_vals, O_col, O_row_ptr, bias)


@e8d4_forward.register_fake
def _e8d4_forward_fake(Q, codes, g, h, x, O_vals, O_col, O_row_ptr, bias):
    B = x.shape[0]
    M8 = Q.shape[0] * 8
    return x.new_empty(B, M8, dtype=torch.float16)
