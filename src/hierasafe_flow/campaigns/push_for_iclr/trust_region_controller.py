"""Context-routed, semantics-preserving trust-region steering for Stage 7 redesign."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from hierasafe_flow.campaigns.push_for_iclr.ablation_controller import (
    intervention_energy,
    normalize_direction_rms,
    tensor_rms,
    tensor_stats,
)
from hierasafe_flow.steering.local_masks import (
    MaskConfig,
    activation_to_mask,
    broadcast_mask_to_vector_field,
)
from hierasafe_flow.steering.vector_fields import local_unsafe_activation


class TrustRegionContractError(RuntimeError):
    """Raised instead of weakening a frozen trust-region contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrustRegionContractError(message)


@dataclass(frozen=True)
class TrustRegionArm:
    arm_id: str
    enabled: bool
    start_fraction: float
    end_fraction: float
    max_local_relative: float
    target_relative_rms: float
    cumulative_energy_budget: float
    semantic_projection: float
    top_k_pairs: int
    minimum_pair_score: float
    routing_temperature: float
    activation_top_fraction: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrustRegionArm":
        arm = cls(
            arm_id=str(value["id"]),
            enabled=bool(value["enabled"]),
            start_fraction=float(value["start_fraction"]),
            end_fraction=float(value["end_fraction"]),
            max_local_relative=float(value["max_local_relative"]),
            target_relative_rms=float(value["target_relative_rms"]),
            cumulative_energy_budget=float(value["cumulative_energy_budget"]),
            semantic_projection=float(value["semantic_projection"]),
            top_k_pairs=int(value["top_k_pairs"]),
            minimum_pair_score=float(value["minimum_pair_score"]),
            routing_temperature=float(value["routing_temperature"]),
            activation_top_fraction=float(value["activation_top_fraction"]),
        )
        _require(0.0 <= arm.start_fraction < arm.end_fraction <= 1.0, "Bad schedule window")
        _require(0.0 <= arm.semantic_projection <= 1.0, "Bad semantic projection")
        _require(arm.top_k_pairs >= 1, "top_k_pairs must be positive")
        _require(arm.minimum_pair_score >= 0.0, "minimum_pair_score must be non-negative")
        _require(arm.routing_temperature > 0.0, "routing_temperature must be positive")
        _require(
            0.0 < arm.activation_top_fraction <= 1.0,
            "activation_top_fraction must be in (0, 1]",
        )
        if arm.enabled:
            _require(0.0 < arm.max_local_relative <= 0.25, "Bad local trust-region cap")
            _require(
                0.0 < arm.target_relative_rms <= 0.25,
                "Bad target relative RMS",
            )
            _require(arm.cumulative_energy_budget > 0.0, "Energy budget must be positive")
        else:
            _require(arm.max_local_relative == 0.0, "Disabled arm must have zero local cap")
            _require(
                arm.target_relative_rms == 0.0,
                "Disabled arm must have zero target relative RMS",
            )
            _require(
                arm.cumulative_energy_budget == 0.0,
                "Disabled arm must have zero energy budget",
            )
        return arm


@dataclass(frozen=True)
class PairDirection:
    pair_id: str
    parent: str
    score: float
    direction: torch.Tensor
    metadata: dict[str, Any]


def contextualize_probe(original_prompt: str, concept: str) -> str:
    """Condition every probe on the requested scene while keeping the concept first."""
    _require(bool(original_prompt.strip()), "Original prompt is empty")
    _require(bool(concept.strip()), "Probe concept is empty")
    return (
        f"Safety concept: {concept.strip()}\n"
        f"Original requested scene: {original_prompt}"
    )


def raised_cosine_window(
    step_index: int,
    num_steps: int,
    *,
    start_fraction: float,
    end_fraction: float,
) -> float:
    _require(num_steps > 0, "num_steps must be positive")
    _require(0 <= step_index < num_steps, "step index out of range")
    progress = 1.0 if num_steps == 1 else step_index / float(num_steps - 1)
    if progress <= start_fraction or progress >= end_fraction:
        return 0.0
    phase = (progress - start_fraction) / (end_fraction - start_fraction)
    return float(math.sin(math.pi * phase) ** 2)


def matched_pair_direction(
    *,
    pair_id: str,
    parent: str,
    v_base: torch.Tensor,
    v_source: torch.Tensor,
    v_target: torch.Tensor,
    v_neutral: torch.Tensor,
    feature_dim: int,
    margin: float,
    mask_config: MaskConfig,
    activation_top_fraction: float,
) -> PairDirection:
    _require(
        v_base.shape == v_source.shape == v_target.shape == v_neutral.shape,
        f"Shape mismatch for pair {pair_id}",
    )
    base_residual = v_base - v_neutral
    source_basis = v_source - v_neutral
    target_basis = v_target - v_neutral
    activation = local_unsafe_activation(
        base_residual,
        source_basis,
        target_basis,
        feature_dim=feature_dim,
        margin=float(margin),
        eps=mask_config.eps,
    )
    mask = activation_to_mask(activation, mask_config)
    mask = broadcast_mask_to_vector_field(mask, v_base)
    direction = mask.to(v_base.dtype) * (target_basis - source_basis)
    flat = activation.detach().float().flatten(start_dim=1)
    top_count = max(1, int(math.ceil(flat.shape[1] * float(activation_top_fraction))))
    score = float(flat.topk(top_count, dim=1).values.mean().cpu())
    _require(math.isfinite(score), f"Non-finite routing score for {pair_id}")
    _require(bool(torch.isfinite(direction).all()), f"Non-finite direction for {pair_id}")
    return PairDirection(
        pair_id=pair_id,
        parent=parent,
        score=score,
        direction=direction,
        metadata={
            "activation": tensor_stats(activation),
            "mask": tensor_stats(mask),
            "base_residual": tensor_stats(base_residual),
            "source_basis": tensor_stats(source_basis),
            "target_basis": tensor_stats(target_basis),
            "raw_direction": tensor_stats(direction),
        },
    )


def route_pair_directions(
    pair_directions: Sequence[PairDirection],
    *,
    base: torch.Tensor,
    top_k_pairs: int,
    minimum_pair_score: float,
    routing_temperature: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    _require(bool(pair_directions), "No matched pair directions were supplied")
    eligible = [
        pair
        for pair in pair_directions
        if pair.score >= float(minimum_pair_score)
    ]
    eligible.sort(key=lambda pair: (-pair.score, pair.pair_id))
    selected = eligible[: int(top_k_pairs)]
    if not selected:
        return torch.zeros_like(base), {
            "selected_pairs": [],
            "all_pair_scores": {pair.pair_id: pair.score for pair in pair_directions},
            "weights": {},
            "reason": "no_pair_met_minimum_score",
        }
    logits = torch.tensor(
        [pair.score / float(routing_temperature) for pair in selected],
        dtype=torch.float64,
    )
    weights = torch.softmax(logits, dim=0).tolist()
    combined = torch.zeros_like(base, dtype=torch.float32)
    for pair, weight in zip(selected, weights):
        combined = combined + pair.direction.float() * float(weight)
    return combined.to(dtype=base.dtype), {
        "selected_pairs": [pair.pair_id for pair in selected],
        "all_pair_scores": {pair.pair_id: pair.score for pair in pair_directions},
        "weights": {
            pair.pair_id: float(weight) for pair, weight in zip(selected, weights)
        },
        "reason": "top_k_softmax_routing",
    }


def project_semantic_component(
    direction: torch.Tensor,
    semantic_residual: torch.Tensor,
    *,
    feature_dim: int,
    coefficient: float,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    _require(direction.shape == semantic_residual.shape, "Semantic projection shape mismatch")
    direction_f = direction.float()
    semantic_f = semantic_residual.float()
    dot = (direction_f * semantic_f).sum(dim=feature_dim, keepdim=True)
    semantic_sq = semantic_f.square().sum(dim=feature_dim, keepdim=True)
    projection = dot / semantic_sq.clamp_min(float(eps)) * semantic_f
    projected = direction_f - float(coefficient) * projection

    direction_norm = direction_f.square().sum(dim=feature_dim).sqrt()
    projected_norm = projected.square().sum(dim=feature_dim).sqrt()
    semantic_norm = semantic_f.square().sum(dim=feature_dim).sqrt()
    cosine_before = dot.squeeze(feature_dim) / (
        direction_norm * semantic_norm
    ).clamp_min(float(eps))
    dot_after = (projected * semantic_f).sum(dim=feature_dim)
    cosine_after = dot_after / (projected_norm * semantic_norm).clamp_min(float(eps))
    return projected.to(dtype=direction.dtype), {
        "coefficient": float(coefficient),
        "mean_abs_cosine_before": float(cosine_before.abs().mean().cpu()),
        "mean_abs_cosine_after": float(cosine_after.abs().mean().cpu()),
    }


def calibrate_relative_rms_direction(
    *,
    base: torch.Tensor,
    direction: torch.Tensor,
    target_relative_rms: float,
    eps: float = 1.0e-12,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Match historical relative-RMS strength before applying trust bounds."""

    _require(base.shape == direction.shape, "Relative-RMS calibration shape mismatch")
    _require(0.0 <= target_relative_rms <= 0.25, "Bad target relative RMS")
    _require(eps >= 0.0, "Negative RMS epsilon")
    _require(bool(torch.isfinite(base).all()), "Non-finite base during RMS calibration")
    _require(bool(torch.isfinite(direction).all()), "Non-finite direction during RMS calibration")

    base_rms_tensor = tensor_rms(base, eps=0.0)
    input_rms_tensor = tensor_rms(direction, eps=0.0)
    base_rms = float(base_rms_tensor.detach().cpu())
    input_rms = float(input_rms_tensor.detach().cpu())
    _require(math.isfinite(base_rms) and base_rms > eps, "Base RMS is too small")
    _require(math.isfinite(input_rms), "Non-finite input direction RMS")

    if target_relative_rms == 0.0 or input_rms <= eps:
        return torch.zeros_like(direction), {
            "policy": "normalize_projected_direction_then_scale_to_base_rms_v1",
            "reason": (
                "zero_target_relative_rms"
                if target_relative_rms == 0.0
                else "zero_projected_direction_rms"
            ),
            "requested_relative_rms": float(target_relative_rms),
            "base_rms": base_rms,
            "input_direction_rms": input_rms,
            "unit_direction_rms": 0.0,
            "calibration_scale": 0.0,
            "calibrated_direction_rms": 0.0,
            "achieved_pre_trust_relative_rms": 0.0,
        }

    unit_direction = normalize_direction_rms(direction, eps=0.0)
    target_rms_tensor = base_rms_tensor * float(target_relative_rms)
    calibrated = unit_direction * target_rms_tensor
    unit_rms = float(tensor_rms(unit_direction, eps=0.0).detach().cpu())
    calibrated_rms = float(tensor_rms(calibrated, eps=0.0).detach().cpu())
    achieved = calibrated_rms / base_rms
    calibration_scale = float((target_rms_tensor / input_rms_tensor).detach().cpu())
    _require(bool(torch.isfinite(calibrated).all()), "Non-finite calibrated direction")
    _require(
        abs(achieved - float(target_relative_rms)) <= 2.0e-5,
        f"Relative-RMS calibration mismatch: {achieved} != {target_relative_rms}",
    )
    return calibrated.to(dtype=direction.dtype), {
        "policy": "normalize_projected_direction_then_scale_to_base_rms_v1",
        "reason": "calibrated",
        "requested_relative_rms": float(target_relative_rms),
        "base_rms": base_rms,
        "input_direction_rms": input_rms,
        "unit_direction_rms": unit_rms,
        "calibration_scale": calibration_scale,
        "calibrated_direction_rms": calibrated_rms,
        "achieved_pre_trust_relative_rms": achieved,
    }


def bounded_trust_region_delta(
    *,
    base: torch.Tensor,
    direction: torch.Tensor,
    feature_dim: int,
    schedule_weight: float,
    max_local_relative: float,
    remaining_energy: float,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    _require(base.shape == direction.shape, "Trust-region shape mismatch")
    _require(0.0 <= schedule_weight <= 1.0 + 1.0e-9, "Bad schedule weight")
    _require(max_local_relative >= 0.0, "Bad local relative cap")
    _require(remaining_energy >= -1.0e-12, "Negative remaining energy")
    if schedule_weight <= 0.0 or max_local_relative == 0.0 or remaining_energy <= 0.0:
        return torch.zeros_like(base), {
            "schedule_weight": float(schedule_weight),
            "effective_local_cap": 0.0,
            "maximum_observed_local_relative": 0.0,
            "energy_before_budget_scale": 0.0,
            "energy_after_budget_scale": 0.0,
            "budget_scale": 0.0,
        }

    base_f = base.float()
    direction_f = direction.float()
    base_norm = base_f.square().sum(dim=feature_dim, keepdim=True).sqrt()
    direction_norm = direction_f.square().sum(dim=feature_dim, keepdim=True).sqrt()
    effective_cap = float(max_local_relative) * float(schedule_weight)
    maximum_delta_norm = effective_cap * base_norm
    local_scale = torch.minimum(
        torch.ones_like(direction_norm),
        maximum_delta_norm / direction_norm.clamp_min(float(eps)),
    )
    delta = direction_f * local_scale
    energy_before = intervention_energy(base_f, delta)
    budget_scale = 1.0
    if energy_before > float(remaining_energy):
        budget_scale = math.sqrt(float(remaining_energy) / max(energy_before, float(eps)))
        delta = delta * budget_scale
    energy_after = intervention_energy(base_f, delta)
    delta_norm = delta.square().sum(dim=feature_dim).sqrt()
    local_relative = delta_norm / base_norm.squeeze(feature_dim).clamp_min(float(eps))
    maximum_local = float(local_relative.max().cpu())
    _require(
        maximum_local <= effective_cap + 2.0e-5,
        f"Local trust-region cap violated: {maximum_local} > {effective_cap}",
    )
    _require(
        energy_after <= float(remaining_energy) + 2.0e-7,
        f"Cumulative energy budget violated: {energy_after} > {remaining_energy}",
    )
    return delta.to(dtype=base.dtype), {
        "schedule_weight": float(schedule_weight),
        "effective_local_cap": effective_cap,
        "maximum_observed_local_relative": maximum_local,
        "energy_before_budget_scale": float(energy_before),
        "energy_after_budget_scale": float(energy_after),
        "budget_scale": float(budget_scale),
    }
