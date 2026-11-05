"""Base quantizer interface."""

from abc import ABC, abstractmethod
from typing import Tuple
import torch


class BaseQuantizer(ABC):
    """Abstract base class for quantizers."""

    @abstractmethod
    def quant(
        self, weight: torch.Tensor, scale: torch.Tensor, batchsize: int = 128
    ) -> torch.Tensor:
        """Quantize weight matrix with given scale."""
        pass

    @abstractmethod
    def dequant(self, indices: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize indices with given scale."""
        pass

    def remove_last(self) -> None:
        """Remove last codebook entry for masking purposes."""
        pass
