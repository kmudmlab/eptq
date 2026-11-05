"""LLaMA model handler."""

from typing import Dict, Tuple, Any
import torch
from torch import nn

from .base import ModelHandler


class LlamaHandler(ModelHandler):
    """Handler for LLaMA model architecture."""

    def get_layers(self, model: nn.Module) -> nn.ModuleList:
        """Get transformer layers from LLaMA model."""
        return model.model.layers

    def get_first_layer_input(
        self, model: nn.Module, calib: torch.Tensor, batchsize: int = 8
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute the input to the first transformer layer for LLaMA.

        For LLaMA, we use a catcher module to capture the actual inputs
        to the first layer, including position_ids and position_embeddings.
        """
        layers = model.model.layers
        device = next(model.parameters()).device

        cache = {}
        input0_list = []

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                input0_list.append(inp.cpu())
                cache["attention_mask"] = kwargs.get("attention_mask")
                cache["position_ids"] = kwargs.get("position_ids")
                cache["position_embeddings"] = kwargs.get("position_embeddings")
                raise ValueError

        # Temporarily replace first layer with catcher
        layers[0] = Catcher(layers[0])

        with torch.no_grad():
            for i in range(0, len(calib), batchsize):
                batch = calib[i : i + batchsize]
                try:
                    model(batch.to(device))
                except ValueError:
                    pass

        # Restore original first layer
        layers[0] = layers[0].module

        input0 = torch.cat(input0_list, dim=0)

        return input0, cache

    def get_linear_layers(self, layer: nn.Module) -> Dict[str, nn.Linear]:
        """Get all linear layers from a LLaMA transformer layer."""
        linear_layers = {}
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers[name] = module
        return linear_layers

    def get_layer_forward_kwargs(self, cache: Dict[str, Any]) -> Dict[str, Any]:
        """Get keyword arguments for LLaMA layer forward pass."""
        kwargs = {}

        if "position_ids" in cache and cache["position_ids"] is not None:
            kwargs["position_ids"] = cache["position_ids"]

        if "position_embeddings" in cache and cache["position_embeddings"] is not None:
            pos_emb = cache["position_embeddings"]
            # Handle tuple of embeddings (cos, sin)
            if isinstance(pos_emb, tuple):
                kwargs["position_embeddings"] = pos_emb
            else:
                kwargs["position_embeddings"] = pos_emb

        return kwargs

    def get_final_norm(self, model: nn.Module) -> nn.Module:
        """Get the final layer norm from LLaMA model."""
        return model.model.norm

    def get_final_norm_path(self) -> str:
        """Get the parameter path prefix for the final layer norm."""
        return "model.norm"

    def get_layer_prefix(self) -> str:
        return "model.layers"

    def get_embeddings(self, model: nn.Module) -> Dict[str, nn.Module]:
        """Get embedding modules and their paths from LLaMA model."""
        embeddings = {}
        if hasattr(model.model, 'embed_tokens'):
            embeddings['model.embed_tokens'] = model.model.embed_tokens
        return embeddings
