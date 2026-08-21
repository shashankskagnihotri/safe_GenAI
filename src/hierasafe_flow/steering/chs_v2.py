"""Conflict-aware Hierarchical Safety steering v2.

CHS v2 composes independently estimated concept directions in clean space,
removing only anti-aligned components and enforcing one global correction
budget. It does not call the legacy bottleneck or Shapley implementations.
"""

from __future__ import annotations

import torch


def _flat(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(value.shape[0], -1)


def _norm(value: torch.Tensor) -> torch.Tensor:
    return _flat(value).norm(dim=1).clamp_min(1.0e-8)


def chs_v2_x0(
    current_x0: torch.Tensor,
    unsafe_x0: list[torch.Tensor],
    safe_x0: list[torch.Tensor],
    *,
    priorities: list[float],
    strength: float,
    max_relative_norm: float,
) -> tuple[torch.Tensor, dict[str, object]]:
    if not unsafe_x0 or len(unsafe_x0) != len(safe_x0):
        raise ValueError("CHS v2 requires matched non-empty unsafe/safe branches")
    if len(priorities) != len(unsafe_x0):
        raise ValueError("CHS v2 priority count must equal branch count")
    accepted: list[torch.Tensor] = []
    conflicts = 0
    for unsafe, safe in zip(unsafe_x0, safe_x0):
        candidate = safe - unsafe
        candidate_flat = _flat(candidate)
        for previous in accepted:
            previous_flat = _flat(previous)
            dot = (candidate_flat * previous_flat).sum(dim=1)
            conflict = dot < 0
            if conflict.any():
                denom = previous_flat.square().sum(dim=1).clamp_min(1.0e-8)
                coefficient = torch.where(conflict, dot / denom, torch.zeros_like(dot))
                shape = (candidate.shape[0],) + (1,) * (candidate.ndim - 1)
                candidate = candidate - coefficient.reshape(shape) * previous
                candidate_flat = _flat(candidate)
                conflicts += int(conflict.sum().detach().cpu())
        accepted.append(candidate)
    weights = torch.as_tensor(priorities, device=current_x0.device, dtype=current_x0.dtype)
    weights = weights
    combined = sum(weight * value for weight, value in zip(weights, accepted))
    raw_norm = _norm(combined)
    budget = float(max_relative_norm) * _norm(current_x0)
    scale = torch.minimum(torch.ones_like(raw_norm), budget / raw_norm)
    view = (combined.shape[0],) + (1,) * (combined.ndim - 1)
    correction = float(strength) * combined * scale.reshape(view)
    applied_directions = [
        float(strength) * weight * value * scale.reshape(view)
        for weight, value in zip(weights, accepted)
    ]
    return current_x0 + correction, {
        "conflict_projections": conflicts,
        "current_x0_norm": float(_norm(current_x0).mean().detach().cpu()),
        "budget_norm": float(budget.mean().detach().cpu()),
        "raw_delta_norm": float(raw_norm.mean().detach().cpu()),
        "applied_delta_norm": float(_norm(correction).mean().detach().cpu()),
        "mean_budget_scale": float(scale.mean().detach().cpu()),
        "budget_was_active": bool((scale < 1.0).any().detach().cpu()),
        "priority_coefficients": [float(x) for x in weights.detach().cpu()],
        "projected_direction_norms": [
            float(_norm(value).mean().detach().cpu()) for value in accepted
        ],
        "applied_contribution_norms": [
            float(_norm(value).mean().detach().cpu()) for value in applied_directions
        ],
    }
