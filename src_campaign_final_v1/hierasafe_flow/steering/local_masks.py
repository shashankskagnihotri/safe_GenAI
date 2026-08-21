from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MaskConfig:
    enabled: bool = True
    mode: str = "max_normalized"
    threshold: float = 0.05
    percentile: float = 0.85
    eps: float = 1.0e-6


def activation_to_mask(activation: torch.Tensor, config: MaskConfig) -> torch.Tensor:
    if activation.numel() == 0:
        raise ValueError("Cannot build a mask from an empty activation tensor.")

    if not config.enabled or config.mode == "full":
        return torch.ones_like(activation)

    if config.mode == "max_normalized":
        flat = activation.detach().float().flatten(start_dim=1)
        denom = flat.amax(dim=1).reshape([-1] + [1] * (activation.ndim - 1)).clamp_min(config.eps)
        mask = activation.float() / denom
        if config.threshold > 0:
            mask = torch.where(mask >= config.threshold, mask, torch.zeros_like(mask))
        return mask.to(dtype=activation.dtype)

    if config.mode == "binary_threshold":
        return (activation.float() >= config.threshold).to(dtype=activation.dtype)

    if config.mode == "percentile":
        flat = activation.detach().float().flatten(start_dim=1)
        thresholds = torch.quantile(flat, q=config.percentile, dim=1, keepdim=True)
        thresholds = thresholds.reshape([-1] + [1] * (activation.ndim - 1))
        return (activation.float() >= thresholds).to(dtype=activation.dtype)

    if config.mode == "identity":
        return activation

    raise ValueError(
        f"Unknown mask mode '{config.mode}'. Valid modes are "
        "max_normalized, binary_threshold, percentile, identity, full."
    )


def broadcast_mask_to_vector_field(mask: torch.Tensor, vector_field: torch.Tensor) -> torch.Tensor:
    if mask.shape == vector_field.shape:
        return mask
    try:
        return torch.broadcast_to(mask, vector_field.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} cannot broadcast to vector field "
            f"shape {tuple(vector_field.shape)}."
        ) from exc
