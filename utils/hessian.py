"""Hessian computation and layer output utilities."""

from typing import Dict, Any
import torch
from torch import nn


@torch.no_grad()
def get_hessians(
    layer: nn.Module,
    layer_input: torch.Tensor,
    batchsize: int = 4,
    forward_kwargs: Dict[str, Any] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute Hessian matrices for all linear layers in a transformer layer.

    The Hessian H = X^T @ X / n approximates the second-order information
    needed for optimal quantization.

    Args:
        layer: Transformer layer module
        layer_input: Input tensor of shape (n_samples, seq_len, hidden_dim)
        batchsize: Batch size for processing
        forward_kwargs: Additional keyword arguments for layer forward pass

    Returns:
        Dictionary mapping layer names to Hessian matrices
    """
    if forward_kwargs is None:
        forward_kwargs = {}

    device = next(layer.parameters()).device

    # Hessian normalization term: divide by n/2 (following GPTQ)
    hnorm = torch.tensor(layer_input.shape[0] / 2).to(device)

    # Find all linear layers
    linear_layers = {}
    for name, module in layer.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers[name] = module

    # Initialize Hessian matrices
    hessians = {}
    for name, linear in linear_layers.items():
        hessians[name] = torch.zeros(
            (linear.in_features, linear.in_features),
            device=device,
            dtype=torch.float32,
        )

    def make_hook(mod_name):
        def hook(module, inputs, output):
            x = inputs[0]
            x = x.view((-1, x.shape[-1]))
            x = x.float()
            hessians[mod_name] += (x.T @ x) / hnorm

        return hook

    # Register hooks
    hooks = []
    for name, linear in linear_layers.items():
        hook = linear.register_forward_hook(make_hook(name))
        hooks.append(hook)

    # Forward pass to accumulate Hessians
    for i in range(0, layer_input.shape[0], batchsize):
        batch = layer_input[i : i + batchsize].to(device)
        layer(batch, **forward_kwargs)

    # Remove hooks
    for h in hooks:
        h.remove()

    return hessians


@torch.no_grad()
def get_layer_output(
    layer: nn.Module,
    layer_input: torch.Tensor,
    batchsize: int = 4,
    forward_kwargs: Dict[str, Any] = None,
) -> torch.Tensor:
    """
    Compute the output of a transformer layer.

    Args:
        layer: Transformer layer module
        layer_input: Input tensor of shape (n_samples, seq_len, hidden_dim)
        batchsize: Batch size for processing
        forward_kwargs: Additional keyword arguments for layer forward pass

    Returns:
        Output tensor of shape (n_samples, seq_len, hidden_dim)
    """
    if forward_kwargs is None:
        forward_kwargs = {}

    device = next(layer.parameters()).device

    outputs = []

    for i in range(0, layer_input.shape[0], batchsize):
        batch = layer_input[i : i + batchsize].to(device)
        out = layer(batch, **forward_kwargs)
        # Handle tuple output (some layers return (hidden_states, ...))
        if isinstance(out, tuple):
            out = out[0]
        outputs.append(out.detach().cpu())

    return torch.cat(outputs, dim=0)
