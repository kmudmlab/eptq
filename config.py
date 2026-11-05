"""Configuration dataclasses for EPTQ."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import torch


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""

    model_name: str
    output_dir: str
    method: str = "eptq"                        # "eptq" or "gptq"
    quantizer_type: str = "e8_lattice"          # EPTQ: "e8_lattice" or "e8_d4"
    percdamp: float = 0.01
    percmask: float = 0.002                     # EPTQ only: critical-weight mask ratio
    blocksize: int = 128
    n_samples: int = 128
    seqlen: int = 2048
    seed: int = 42
    calib_dataset: str = "c4"                     # "c4" or "wikitext2"
    weight_scale_norm: bool = True              # EPTQ only
    codebook_radius: Optional[float] = None      # EPTQ only: override default codebook divisor
    codebook_warp_alpha: float = 1.0             # EPTQ E8D4 only: radial warping exponent (1.0 = no warp)
    radius_candidates: Optional[list] = None     # EPTQ only: per-layer radius search candidates
    qb_size: Optional[int] = None               # EPTQ only: row group size for group quantization
    mask_tau: Optional[float] = None            # EPTQ only: RTN pre-pass threshold (overrides percmask)
    auto_mask: bool = False                     # EPTQ only: per-layer knee detection (overrides percmask/mask_tau)
    auto_mask_search_pct: float = 0.10         # EPTQ only: top-X% of D_sum used as Kneedle search window
    device: str = "cuda"
    # Proxy fine-tuning (EPTQ only)
    ft_epochs: int = 0              # 0 = disabled; >0 enables per-layer g/h fine-tuning
    ft_lr: float = 1e-4
    ft_batchsize: int = 8
    ft_fp16_chain: bool = False     # True: target = FP16 chain output (FP16 inputs propagated separately)
                                    # False: target = FP16 layer on quantized input (local approach)
    fp16_hessian: bool = False      # True: compute Hessians from FP16 chain inputs (QTIP/VPTQ style)
                                    # False: compute Hessians from quantized chain inputs (default)
    ft_valid_size: int = 0          # number of samples held out for validation + early stopping (0 = disabled)
    ft_early_stop: int = 5          # stop if validation loss doesn't improve for this many epochs
    ft_tune_O: bool = False         # True: also fine-tune critical-weight values O (in addition to g/h)
    ft_loss: str = "mse"            # loss function: "mse" | "cosine" | "mse+cosine"
    ft_cosine_w: float = 1.0        # weight for cosine term when ft_loss="mse+cosine"
    ft_e2e: bool = False            # True: end-to-end CE-loss FT after all layers are quantized
    # GPTQ-specific
    bits: int = 4                               # GPTQ only: 2/3/4/8
    group_size: int = -1                        # GPTQ only: columns per group (-1 = per-column)
    symmetric: bool = False                     # GPTQ only
    actorder: bool = False                      # GPTQ only

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "output_dir": self.output_dir,
            "method": self.method,
            "quantizer_type": self.quantizer_type,
            "percdamp": self.percdamp,
            "percmask": self.percmask,
            "blocksize": self.blocksize,
            "n_samples": self.n_samples,
            "seqlen": self.seqlen,
            "seed": self.seed,
            "calib_dataset": self.calib_dataset,
            "weight_scale_norm": self.weight_scale_norm,
            "codebook_radius": self.codebook_radius,
            "codebook_warp_alpha": self.codebook_warp_alpha,
            "radius_candidates": self.radius_candidates,
            "qb_size": self.qb_size,
            "mask_tau": self.mask_tau,
            "auto_mask": self.auto_mask,
            "auto_mask_search_pct": self.auto_mask_search_pct,
            "ft_epochs": self.ft_epochs,
            "ft_lr": self.ft_lr,
            "ft_batchsize": self.ft_batchsize,
            "ft_fp16_chain": self.ft_fp16_chain,
            "fp16_hessian": self.fp16_hessian,
            "ft_valid_size": self.ft_valid_size,
            "ft_early_stop": self.ft_early_stop,
            "ft_tune_O": self.ft_tune_O,
            "ft_loss": self.ft_loss,
            "ft_cosine_w": self.ft_cosine_w,
            "ft_e2e": self.ft_e2e,
            "bits": self.bits,
            "group_size": self.group_size,
            "symmetric": self.symmetric,
            "actorder": self.actorder,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QuantizationConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class QuantizedLayerData:
    """
    Stores quantized layer data.

    This class holds the quantized representation of a linear layer's weights,
    including the quantized indices, scales, and critical weight information.
    Note: Per-column scales (S) are now integrated into h.
    Critical weight positions are marked with 65535 in Q (no separate mask_indices).
    """

    Q: torch.Tensor  # int16: quantized indices, shape: (out_features//8, in_features)
                     # Critical positions marked with 65535
    g: torch.Tensor  # fp32: row scale factors, shape: (out_features,)
    h: torch.Tensor  # fp32: column scale factors (integrated with S when no grouping),
                     # shape: (in_features,)
    O: torch.Tensor  # fp32: original values for masked weights, ordered sequentially.
                     # Normalized by S when S_group is None; raw g/h-space when S_group present.
    bias: Optional[torch.Tensor] = None   # fp16: original bias (if exists)
    chosen_r: Optional[float] = None      # codebook radius used for this layer's quantization
    S_group: Optional[torch.Tensor] = None  # fp16: per-(row-group, col) scales,
                                             # shape: (out_features//qb_size, in_features).
                                             # None = no group quantization (S folded into h).

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for torch.save()."""
        result = {
            "Q": self.Q,
            "g": self.g,
            "h": self.h,
            "O": self.O,
        }
        if self.bias is not None:
            result["bias"] = self.bias
        if self.chosen_r is not None:
            result["chosen_r"] = self.chosen_r
        if self.S_group is not None:
            result["S_group"] = self.S_group
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QuantizedLayerData":
        """Create from dictionary loaded via torch.load()."""
        return cls(
            Q=d["Q"],
            g=d["g"],
            h=d["h"],
            O=d["O"],
            bias=d.get("bias"),
            chosen_r=d.get("chosen_r"),
            S_group=d.get("S_group"),
        )

    def save(self, path: Path) -> None:
        """Save quantized layer data to file."""
        torch.save(self.to_dict(), path)

    @classmethod
    def load(cls, path: Path) -> "QuantizedLayerData":
        """Load quantized layer data from file."""
        data = torch.load(path, weights_only=True, map_location="cpu")
        return cls.from_dict(data)


@dataclass
class GPTQLayerData:
    """
    Stores quantized layer data for GPTQ-style scalar quantization.

    Unlike :class:`QuantizedLayerData` (vector quantization with codebook),
    GPTQ stores integer indices plus per-group ``(scale, zero)`` tables.

    Shapes
    ------
    Q       : int8/int32  (out_features, in_features)   unpacked indices
    scales  : fp16        (out_features, n_groups)      per (row, group)
    zeros   : fp16        (out_features, n_groups)      per (row, group)
              where n_groups = ceil(in_features / group_size), or 1 if
              group_size = -1 (per-column).
    """

    Q: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    bits: int = 4
    group_size: int = -1
    bias: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for torch.save()."""
        result = {
            "Q": self.Q,
            "scales": self.scales,
            "zeros": self.zeros,
            "bits": self.bits,
            "group_size": self.group_size,
        }
        if self.bias is not None:
            result["bias"] = self.bias
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GPTQLayerData":
        return cls(
            Q=d["Q"],
            scales=d["scales"],
            zeros=d["zeros"],
            bits=int(d.get("bits", 4)),
            group_size=int(d.get("group_size", -1)),
            bias=d.get("bias"),
        )

    def save(self, path: Path) -> None:
        torch.save(self.to_dict(), path)

    @classmethod
    def load(cls, path: Path) -> "GPTQLayerData":
        data = torch.load(path, weights_only=True)
        return cls.from_dict(data)
