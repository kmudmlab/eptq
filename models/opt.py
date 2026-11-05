"""OPT model handler."""

from typing import Dict, Tuple, Any
import torch
from torch import nn

from .base import ModelHandler


class OPTHandler(ModelHandler):
    """Handler for OPT model architecture."""

    def get_layers(self, model: nn.Module) -> nn.ModuleList:
        """Get transformer layers from OPT model."""
        return model.model.decoder.layers

    def get_first_layer_input(
        self, model: nn.Module, calib: torch.Tensor, batchsize: int = 8
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute the input to the first transformer layer for OPT.

        For OPT, we need to compute token embeddings + position embeddings.
        """
        device = model.device

        input0_list = []

        with torch.no_grad():
            for i in range(0, len(calib), batchsize):
                batch = calib[i : i + batchsize].to(device)

                token_embeds = model.model.decoder.embed_tokens(batch)
                attention_mask = torch.ones(batch.shape, dtype=torch.long, device=device)
                position_embeds = model.model.decoder.embed_positions(
                    attention_mask, past_key_values_length=0
                )
                hidden_states = token_embeds + position_embeds

                input0_list.append(hidden_states.cpu())

        input0 = torch.cat(input0_list, dim=0)

        # OPT doesn't need special cache for layer forward
        cache = {}

        return input0, cache

    def get_linear_layers(self, layer: nn.Module) -> Dict[str, nn.Linear]:
        """Get all linear layers from an OPT transformer layer."""
        linear_layers = {}
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers[name] = module
        return linear_layers

    def get_layer_forward_kwargs(self, cache: Dict[str, Any]) -> Dict[str, Any]:
        """Get keyword arguments for OPT layer forward pass."""
        # OPT layers don't need special kwargs
        return {}

    def get_final_norm(self, model: nn.Module) -> nn.Module:
        """Get the final layer norm from OPT model."""
        return model.model.decoder.final_layer_norm

    def get_final_norm_path(self) -> str:
        """Get the parameter path prefix for the final layer norm."""
        return "model.decoder.final_layer_norm"

    def get_layer_prefix(self) -> str:
        return "model.decoder.layers"

    def get_embeddings(self, model: nn.Module) -> Dict[str, nn.Module]:
        """Get embedding modules and their paths from OPT model."""
        embeddings = {}
        decoder = model.model.decoder
        if hasattr(decoder, 'embed_tokens'):
            embeddings['model.decoder.embed_tokens'] = decoder.embed_tokens
        if hasattr(decoder, 'embed_positions'):
            embeddings['model.decoder.embed_positions'] = decoder.embed_positions
        return embeddings
