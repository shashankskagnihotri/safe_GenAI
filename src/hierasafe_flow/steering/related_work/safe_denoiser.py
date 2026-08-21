"""Safe Denoiser switch adaptation using a normalized unsafe KDE denoiser."""

from __future__ import annotations

import torch

from ..canonical.contract import channel_tokens, restore_channel_tokens


def safe_denoiser_switch_x0(
    current_x0: torch.Tensor,
    unsafe_prototypes: torch.Tensor,
    *,
    model_id: str,
    layout: dict[str, object] | None = None,
    bandwidth: float,
    eta: float,
    max_relative_norm: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    tokens, metadata = channel_tokens(current_x0, model_id, layout)
    prototypes = unsafe_prototypes.to(device=tokens.device, dtype=torch.float32)
    work = tokens.float()
    if prototypes.ndim != 2 or prototypes.shape[-1] != work.shape[-1]:
        raise ValueError("Safe Denoiser prototype width does not match latent channels")
    distance2 = (work[:, None, :] - prototypes[None, :, :]).square().sum(dim=-1)
    sigma2 = max(float(bandwidth) ** 2, 1.0e-8)
    affinity = torch.exp(-distance2 / (2.0 * sigma2))
    weights = affinity / affinity.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    unsafe_denoised = weights @ prototypes
    q_t = affinity.mean(dim=1)
    beta = float(eta) * q_t.mean()
    raw_tokens = float(eta) * beta * (work - unsafe_denoised)
    raw = restore_channel_tokens(raw_tokens.to(tokens.dtype), metadata)
    raw_norm = raw.reshape(raw.shape[0], -1).norm(dim=1).clamp_min(1.0e-8)
    base_norm = current_x0.reshape(current_x0.shape[0], -1).norm(dim=1).clamp_min(1.0e-8)
    scale = torch.minimum(torch.ones_like(raw_norm), float(max_relative_norm) * base_norm / raw_norm)
    view = (raw.shape[0],) + (1,) * (raw.ndim - 1)
    correction = raw * scale.reshape(view)
    return current_x0 + correction, {
        "mean_q_t": float(q_t.mean().detach().cpu()),
        "beta": float(beta.detach().cpu()),
        "unsafe_denoiser_norm": float(unsafe_denoised.norm(dim=1).mean().detach().cpu()),
        "applied_delta_norm": float(correction.reshape(correction.shape[0], -1).norm(dim=1).mean().detach().cpu()),
    }
