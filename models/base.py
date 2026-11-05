"""Base model handler interface."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import torch
from torch import nn


class ModelHandler(ABC):
    """
    Abstract base class for model-specific handlers.

    Each model architecture (OPT, LLaMA, etc.) needs a handler that knows
    how to extract layers, compute first layer inputs, and identify linear layers.
    """

    @abstractmethod
    def get_layers(self, model: nn.Module) -> nn.ModuleList:
        """
        Get the list of transformer layers from the model.

        Args:
            model: The loaded model

        Returns:
            ModuleList of transformer layers
        """
        pass

    @abstractmethod
    def get_first_layer_input(
        self, model: nn.Module, calib: torch.Tensor, batchsize: int = 8
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute the input to the first transformer layer.

        Args:
            model: The loaded model
            calib: Calibration data (token ids)
            batchsize: Batch size for processing

        Returns:
            Tuple of (input tensor, cache dict with auxiliary inputs)
        """
        pass

    @abstractmethod
    def get_linear_layers(self, layer: nn.Module) -> Dict[str, nn.Linear]:
        """
        Get all linear layers from a transformer layer.

        Args:
            layer: A single transformer layer

        Returns:
            Dictionary mapping layer names to nn.Linear modules
        """
        pass

    @abstractmethod
    def get_layer_forward_kwargs(self, cache: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get keyword arguments needed for layer forward pass.

        Args:
            cache: Cache dict from get_first_layer_input

        Returns:
            Dictionary of keyword arguments
        """
        pass

    @abstractmethod
    def get_final_norm(self, model: nn.Module) -> nn.Module:
        """
        Get the final layer norm from the model.

        Args:
            model: The loaded model

        Returns:
            The final layer norm module
        """
        pass

    @abstractmethod
    def get_final_norm_path(self) -> str:
        """
        Get the parameter path prefix for the final layer norm.

        Returns:
            Path prefix string (e.g., 'model.norm' or 'model.decoder.final_layer_norm')
        """
        pass

    @abstractmethod
    def get_embeddings(self, model: nn.Module) -> Dict[str, nn.Module]:
        """
        Get embedding modules and their paths from the model.

        Args:
            model: The loaded model

        Returns:
            Dictionary mapping parameter path to embedding module
        """
        pass

    @abstractmethod
    def get_layer_prefix(self) -> str:
        """Return the named_modules path prefix for transformer layers (e.g. 'model.decoder.layers')."""
        pass

    def get_num_layers(self, model: nn.Module) -> int:
        """
        Get the number of transformer layers.

        Args:
            model: The loaded model

        Returns:
            Number of layers
        """
        return len(self.get_layers(model))

    @staticmethod
    def detect_model_type(model: nn.Module) -> str:
        """
        Detect the model type from the model architecture.

        Args:
            model: The loaded model

        Returns:
            Model type string ("opt", "llama", or "unknown")
        """
        model_class = model.__class__.__name__.lower()
        if "opt" in model_class:
            return "opt"
        elif "llama" in model_class:
            return "llama"
        else:
            return "unknown"
