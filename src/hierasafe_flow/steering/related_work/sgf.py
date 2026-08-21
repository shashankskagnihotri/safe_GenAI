"""SGF switch adaptation using the paper-sign RBF-MMD repulsive gradient."""

from __future__ import annotations

import torch

from ..canonical.contract import channel_tokens, restore_channel_tokens


def sgf_switch_x0(
    current_x0: torch.Tensor,
    unsafe_prototypes: torch.Tensor,
    *,
    model_id: str,
    layout: dict[str, object] | None = None,
    bandwidth: float,
    strength: float,
    max_relative_norm: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    tokens, metadata = channel_tokens(current_x0, model_id, layout)
    prototypes = unsafe_prototypes.to(device=tokens.device, dtype=torch.float32)
    work = tokens.float()
    if prototypes.ndim != 2 or prototypes.shape[-1] != work.shape[-1]:
        raise ValueError("SGF unsafe prototype width does not match latent channels")
    difference = work[:, None, :] - prototypes[None, :, :]
    distance2 = difference.square().sum(dim=-1)
    sigma2 = max(float(bandwidth) ** 2, 1.0e-8)
    kernel = torch.exp(-distance2 / (2.0 * sigma2))
    weighted_mean = (kernel[..., None] * prototypes[None, :, :]).sum(dim=1) / kernel.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    gradient = (2.0 / sigma2) * kernel.sum(dim=1, keepdim=True) * (work - weighted_mean)
    raw = restore_channel_tokens(gradient.to(tokens.dtype), metadata)
    raw_norm = raw.reshape(raw.shape[0], -1).norm(dim=1).clamp_min(1.0e-8)
    base_norm = current_x0.reshape(current_x0.shape[0], -1).norm(dim=1).clamp_min(1.0e-8)
    scale = torch.minimum(torch.ones_like(raw_norm), float(max_relative_norm) * base_norm / raw_norm)
    view = (raw.shape[0],) + (1,) * (raw.ndim - 1)
    correction = float(strength) * raw * scale.reshape(view)
    return current_x0 + correction, {
        "kernel_mean": float(kernel.mean().detach().cpu()),
        "raw_gradient_norm": float(raw_norm.mean().detach().cpu()),
        "applied_delta_norm": float(correction.reshape(correction.shape[0], -1).norm(dim=1).mean().detach().cpu()),
    }
