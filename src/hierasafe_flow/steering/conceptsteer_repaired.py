"""Repaired current ConceptSteer controller with an explicit clean-space budget."""

from __future__ import annotations

import torch


def window_weight(progress: float, start: float, end: float) -> float:
    if progress < start or progress > end:
        return 0.0
    span = max(end - start, 1.0e-8)
    phase = (progress - start) / span
    return float(torch.sin(torch.tensor(phase * torch.pi)).item())


def _batch_norm(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(value.shape[0], -1).norm(dim=1).clamp_min(1.0e-8)


def repaired_conceptsteer_x0(
    current_x0: torch.Tensor,
    unsafe_x0: list[torch.Tensor],
    safe_x0: list[torch.Tensor],
    *,
    strength: float,
    max_relative_norm: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not unsafe_x0 or len(unsafe_x0) != len(safe_x0):
        raise ValueError("ConceptSteer requires matched non-empty unsafe/safe branches")
    deltas = torch.stack([safe - unsafe for unsafe, safe in zip(unsafe_x0, safe_x0)], dim=0)
    raw = deltas.mean(dim=0)
    raw_norm = _batch_norm(raw)
    base_norm = _batch_norm(current_x0)
    cap = max_relative_norm * base_norm
    scale = torch.minimum(torch.ones_like(raw_norm), cap / raw_norm)
    view = (raw.shape[0],) + (1,) * (raw.ndim - 1)
    correction = float(strength) * raw * scale.reshape(view)
    return current_x0 + correction, {
        "raw_delta_norm": float(raw_norm.mean().detach().cpu()),
        "applied_delta_norm": float(_batch_norm(correction).mean().detach().cpu()),
        "mean_budget_scale": float(scale.mean().detach().cpu()),
    }
