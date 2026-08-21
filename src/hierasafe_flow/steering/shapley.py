"""Selective latent vector-field steering using local Shapley values.

Provenance and boundary
-----------------------
This implementation is inspired by the binary-coalition and affine Shapley
formulation in the official ShaplEIG repository at commit
``162ce44fe380c7c11b959fc85206b5dcdeddff58`` (MIT license), in particular its
``ShapleyApplication`` representation of Shapley values as ``A @ f(Z)``.
ShaplEIG studies Bayesian experimental design for scalar black-box games.  It
does not contain diffusion, latent-vector-field, token-local attribution, or
steering code.  The game, memory-bounded exact estimator, deterministic
antithetic permutation estimator, intervention, and trust region below are new
for this project; no GP/EIG estimator from ShaplEIG is claimed or copied here.

The resulting values are *functional attributions for the explicitly defined
latent vector-field game*.  They are not causal effects.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from hierasafe_flow.adapters.base import LatentLayout
from hierasafe_flow.steering.bottleneck import (
    BottleneckConfig,
    BottleneckTrace,
    ConceptTrace,
    HierarchicalVectorFieldBottleneck,
    _timestep_to_log_value,
    _validate_condition_calls,
    _validated_segment_record,
)
from hierasafe_flow.steering.concept_graph import ConceptHierarchy, ConceptPair
from hierasafe_flow.steering.schedules import step_is_enabled
from hierasafe_flow.steering.vector_fields import compute_concept_basis
from hierasafe_flow.utils.tensors import tensor_stats


SHAPLEY_TRACE_SCHEMA_VERSION = 2
SHAPLEY_INTERVENTION_IDENTITY = (
    "shapley_selected_attribution_weighted_current_to_safe_prediction_rollback"
)
DEFAULT_BACKTRACKING_SCALES = (1.0, 0.5, 0.25, 0.125, 0.0625)


@dataclass(frozen=True)
class ShapleyConfig:
    """Estimator and intervention controls from ``steering.shapley``."""

    estimator: str = "antithetic_permutation"
    score_tau: float = 0.10
    eps: float = 1.0e-6
    seed: int = 1234
    exact_max_players: int = 16
    exact_max_work: int = 2_000_000
    min_permutations: int = 8
    max_permutations: int = 32
    token_chunk_size: int = 65536
    coalition_chunk_size: int = 128
    confidence: float = 0.95
    relative_ci: float | None = 0.05
    absolute_ci: float | None = 0.005
    require_convergence: bool = True
    ci_quantile: float = 0.99
    ci_positive_only: bool = False
    positive_only: bool = True
    top_mass: float = 0.90
    per_coordinate_cap: float = 1.0
    trust_region_ratio: float = 0.25
    efficiency_tolerance: float = 1.0e-4
    score_skip_tolerance: float = 0.0
    backtracking_scales: tuple[float, ...] = DEFAULT_BACKTRACKING_SCALES
    interpolation_fraction_cap: float = 1.0

    def __post_init__(self) -> None:
        if self.estimator != "antithetic_permutation":
            raise ValueError(
                "steering.shapley.estimator must be 'antithetic_permutation'. "
                "Exact enumeration is selected automatically when its total-work gate permits it."
            )
        if self.score_tau <= 0:
            raise ValueError("steering.shapley.score_tau must be > 0.")
        if self.eps <= 0:
            raise ValueError("steering.shapley.eps must be > 0.")
        if self.exact_max_players < 0:
            raise ValueError("steering.shapley.exact_max_players must be >= 0.")
        if self.exact_max_work < 0:
            raise ValueError("steering.shapley.exact_max_work must be >= 0.")
        if self.max_permutations < 8 or self.max_permutations % 2:
            raise ValueError("steering.shapley.max_permutations must be an even integer >= 8.")
        if self.min_permutations < 8 or self.min_permutations % 2:
            raise ValueError(
                "steering.shapley.min_permutations must be an even integer >= 8 "
                "(four independent antithetic-pair observations)."
            )
        if self.min_permutations > self.max_permutations:
            raise ValueError(
                "steering.shapley.min_permutations cannot exceed max_permutations."
            )
        if self.token_chunk_size < 1 or self.coalition_chunk_size < 1:
            raise ValueError("Shapley token and coalition chunk sizes must be >= 1.")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("steering.shapley.confidence must be in (0, 1).")
        if self.relative_ci is not None and self.relative_ci < 0:
            raise ValueError("steering.shapley.relative_ci must be >= 0 when supplied.")
        if self.absolute_ci is not None and self.absolute_ci < 0:
            raise ValueError("steering.shapley.absolute_ci must be >= 0 when supplied.")
        if self.relative_ci is None and self.absolute_ci is None:
            raise ValueError("At least one of relative_ci or absolute_ci must be supplied.")
        if not 0.0 < self.ci_quantile <= 1.0:
            raise ValueError("steering.shapley.ci_quantile must be in (0, 1].")
        if not 0.0 < self.top_mass <= 1.0:
            raise ValueError("steering.shapley.top_mass must be in (0, 1].")
        if not self.positive_only:
            raise ValueError(
                "steering.shapley.positive_only=false is unsupported: intervention is "
                "defined only for positive unwanted-concept attribution."
            )
        if not 0.0 < self.per_coordinate_cap <= 1.0:
            raise ValueError("steering.shapley.per_coordinate_cap must be in (0, 1].")
        if self.trust_region_ratio <= 0:
            raise ValueError("steering.shapley.trust_region_ratio must be > 0.")
        if self.efficiency_tolerance <= 0:
            raise ValueError("steering.shapley.efficiency_tolerance must be > 0.")
        if self.score_skip_tolerance < 0:
            raise ValueError("steering.shapley.score_skip_tolerance must be >= 0.")
        object.__setattr__(
            self,
            "backtracking_scales",
            _validate_backtracking_scales(self.backtracking_scales),
        )
        if not 0.0 < self.interpolation_fraction_cap <= 1.0:
            raise ValueError(
                "steering.shapley.interpolation_fraction_cap must be in (0, 1]."
            )
        finite_controls = {
            "score_tau": self.score_tau,
            "eps": self.eps,
            "confidence": self.confidence,
            "ci_quantile": self.ci_quantile,
            "top_mass": self.top_mass,
            "per_coordinate_cap": self.per_coordinate_cap,
            "trust_region_ratio": self.trust_region_ratio,
            "efficiency_tolerance": self.efficiency_tolerance,
            "score_skip_tolerance": self.score_skip_tolerance,
            "interpolation_fraction_cap": self.interpolation_fraction_cap,
        }
        if self.relative_ci is not None:
            finite_controls["relative_ci"] = self.relative_ci
        if self.absolute_ci is not None:
            finite_controls["absolute_ci"] = self.absolute_ci
        nonfinite = [name for name, value in finite_controls.items() if not math.isfinite(value)]
        if nonfinite:
            raise ValueError(f"Non-finite steering.shapley controls are invalid: {nonfinite}.")

    @classmethod
    def from_dict(
        cls,
        steering: dict[str, Any] | None,
        *,
        default_seed: int = 1234,
    ) -> "ShapleyConfig":
        steering = steering or {}
        raw = steering.get("shapley") or {}
        if not isinstance(raw, dict):
            raise ValueError("steering.shapley must be a mapping.")
        known_keys = {
            # Canonical manifest-facing keys.
            "estimator",
            "seed",
            "min_permutations",
            "max_permutations",
            "confidence",
            "relative_ci",
            "absolute_ci",
            "require_convergence",
            "score_tau",
            "positive_only",
            "top_mass",
            "per_coordinate_cap",
            "trust_region_ratio",
            "efficiency_tolerance",
            # Explicit advanced controls, retained for reproducible tests and
            # workload tuning rather than silently accepting arbitrary keys.
            "eps",
            "exact_max_players",
            "exact_max_work",
            "token_chunk_size",
            "coalition_chunk_size",
            "ci_quantile",
            "ci_positive_only",
            "score_skip_tolerance",
            "backtracking_scales",
            "interpolation_fraction_cap",
        }
        unknown = sorted(set(raw) - known_keys)
        if unknown:
            raise ValueError(
                f"Unknown steering.shapley keys: {unknown}. Valid keys: {sorted(known_keys)}"
            )
        return cls(
            estimator=str(raw.get("estimator", cls.estimator)),
            score_tau=float(raw.get("score_tau", cls.score_tau)),
            eps=float(raw.get("eps", cls.eps)),
            seed=int(raw.get("seed", default_seed)),
            exact_max_players=int(raw.get("exact_max_players", cls.exact_max_players)),
            exact_max_work=int(raw.get("exact_max_work", cls.exact_max_work)),
            min_permutations=int(raw.get("min_permutations", cls.min_permutations)),
            max_permutations=int(raw.get("max_permutations", cls.max_permutations)),
            token_chunk_size=int(raw.get("token_chunk_size", cls.token_chunk_size)),
            coalition_chunk_size=int(raw.get("coalition_chunk_size", cls.coalition_chunk_size)),
            confidence=float(raw.get("confidence", cls.confidence)),
            relative_ci=(
                cls.relative_ci
                if "relative_ci" not in raw
                else (None if raw["relative_ci"] is None else float(raw["relative_ci"]))
            ),
            absolute_ci=(
                cls.absolute_ci
                if "absolute_ci" not in raw
                else (None if raw["absolute_ci"] is None else float(raw["absolute_ci"]))
            ),
            require_convergence=bool(raw.get("require_convergence", cls.require_convergence)),
            ci_quantile=float(raw.get("ci_quantile", cls.ci_quantile)),
            ci_positive_only=bool(raw.get("ci_positive_only", cls.ci_positive_only)),
            positive_only=bool(raw.get("positive_only", cls.positive_only)),
            top_mass=float(raw.get("top_mass", cls.top_mass)),
            per_coordinate_cap=float(
                raw.get("per_coordinate_cap", cls.per_coordinate_cap)
            ),
            trust_region_ratio=float(raw.get("trust_region_ratio", cls.trust_region_ratio)),
            efficiency_tolerance=float(
                raw.get("efficiency_tolerance", cls.efficiency_tolerance)
            ),
            score_skip_tolerance=float(
                raw.get("score_skip_tolerance", cls.score_skip_tolerance)
            ),
            backtracking_scales=_coerce_backtracking_scales(
                raw.get("backtracking_scales", cls.backtracking_scales)
            ),
            interpolation_fraction_cap=float(
                raw.get(
                    "interpolation_fraction_cap",
                    cls.interpolation_fraction_cap,
                )
            ),
        )

    @property
    def tau(self) -> float:
        return self.score_tau

    @property
    def max_mask_value(self) -> float:
        return self.per_coordinate_cap


@dataclass
class ShapleyEstimate:
    values: torch.Tensor
    variance: torch.Tensor
    ci_half_width: torch.Tensor
    metadata: dict[str, Any]


def local_unwanted_score(
    hybrid: torch.Tensor,
    unsafe_direction: torch.Tensor,
    safe_direction: torch.Tensor,
    *,
    margin: float,
    tau: float,
    eps: float,
) -> torch.Tensor:
    """Return ``softplus((cos(h,b_u)-cos(h,b_s)+margin)/tau)`` per token."""
    _require_finite_tensors(
        hybrid=hybrid,
        unsafe_direction=unsafe_direction,
        safe_direction=safe_direction,
    )
    hybrid_f = hybrid.float()
    unsafe_f = unsafe_direction.float()
    safe_f = safe_direction.float()
    unsafe_cos = _cosine_last_dim(hybrid_f, unsafe_f, eps=eps)
    safe_cos = _cosine_last_dim(hybrid_f, safe_f, eps=eps)
    score = F.softplus((unsafe_cos - safe_cos + float(margin)) / float(tau))
    _require_finite_tensors(local_unwanted_score=score)
    return score


def exact_shapley_from_coalition_values(coalition_values: torch.Tensor) -> torch.Tensor:
    """Apply the exact affine Shapley transform to values ordered by bit mask.

    The first axis must contain ``2**D`` coalitions.  Remaining axes may be
    arbitrary outputs, making the same transform valid for every local token
    game in parallel.
    """
    coalition_count = int(coalition_values.shape[0])
    if coalition_count < 2 or coalition_count & (coalition_count - 1):
        raise ValueError("The coalition axis must have power-of-two length >= 2.")
    players = coalition_count.bit_length() - 1
    transform = _exact_shapley_transform(players).to(
        device=coalition_values.device,
        dtype=torch.float32,
    )
    flattened = coalition_values.float().reshape(coalition_count, -1)
    result = transform @ flattened
    return result.reshape(players, *coalition_values.shape[1:])


def estimate_local_channel_shapley(
    h_empty: torch.Tensor,
    h_full: torch.Tensor,
    unsafe_direction: torch.Tensor,
    safe_direction: torch.Tensor,
    *,
    margin: float,
    config: ShapleyConfig,
    deterministic_seed: int,
) -> ShapleyEstimate:
    """Estimate channel-player Shapley values for every ``[B,T]`` local game."""
    _require_canonical_shapes(h_empty, h_full, unsafe_direction, safe_direction)
    _require_finite_tensors(
        h_empty=h_empty,
        h_full=h_full,
        unsafe_direction=unsafe_direction,
        safe_direction=safe_direction,
    )
    batch, tokens, players = h_full.shape
    positions = batch * tokens
    # Exact evaluation is selected by total arithmetic work, not player count
    # alone.  This prevents 16-channel million-token videos from accidentally
    # materializing or evaluating an impractical exhaustive game.
    exact_work = players * (2 ** max(players - 1, 0)) * positions
    use_exact = (
        players <= config.exact_max_players
        and exact_work <= config.exact_max_work
    )
    _synchronize(h_full.device)
    started = time.perf_counter()
    if use_exact:
        estimate = _estimate_exact(
            h_empty,
            h_full,
            unsafe_direction,
            safe_direction,
            margin=margin,
            config=config,
        )
        selection_reason = "within_exact_player_and_total_work_limits"
    else:
        estimate = _estimate_antithetic_permutations(
            h_empty,
            h_full,
            unsafe_direction,
            safe_direction,
            margin=margin,
            config=config,
            deterministic_seed=deterministic_seed,
        )
        if players > config.exact_max_players:
            selection_reason = "feature_count_exceeds_exact_max_players"
        else:
            selection_reason = "total_work_exceeds_exact_max_work"
    _synchronize(h_full.device)
    estimate.metadata.update(
        {
            "runtime_seconds": time.perf_counter() - started,
            "exact_total_work_estimate": int(exact_work),
            "exact_max_work": int(config.exact_max_work),
            "exact_max_players": int(config.exact_max_players),
            "selection_reason": selection_reason,
            "positions": int(positions),
            "feature_players": int(players),
            "deterministic_seed": int(deterministic_seed),
            "cuda_max_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(h_full.device))
                if h_full.is_cuda
                else None
            ),
        }
    )
    _require_finite_tensors(
        shapley_values=estimate.values,
        shapley_variance=estimate.variance,
        shapley_ci_half_width=estimate.ci_half_width,
    )
    return estimate


def attribution_to_mask(
    values: torch.Tensor,
    ci_half_width: torch.Tensor,
    q_empty: torch.Tensor,
    q_full: torch.Tensor,
    config: ShapleyConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build a normalized, positive-only, top-mass attribution mask."""
    if values.shape != ci_half_width.shape:
        raise ValueError("Shapley values and confidence intervals must have the same shape.")
    if values.shape[:-1] != q_empty.shape or q_empty.shape != q_full.shape:
        raise ValueError("Score tensors must match the Shapley batch/token axes.")
    _require_finite_tensors(
        shapley_values=values,
        ci_half_width=ci_half_width,
        q_empty=q_empty,
        q_full=q_full,
    )
    positive = values.float().clamp_min(0.0)
    if config.ci_positive_only:
        positive = torch.where(values.float() - ci_half_width.float() > 0, positive, 0.0)

    improves_over_target = q_full.float() > (
        q_empty.float() + config.score_skip_tolerance
    )
    positive = positive * improves_over_target.unsqueeze(-1)

    sorted_values, sorted_indices = torch.sort(positive, dim=-1, descending=True)
    total = sorted_values.sum(dim=-1, keepdim=True)
    cumulative_before = sorted_values.cumsum(dim=-1) - sorted_values
    keep_sorted = (
        (sorted_values > 0)
        & (cumulative_before < total * config.top_mass)
        & (total > config.eps)
    )
    selected_sorted = torch.where(keep_sorted, sorted_values, torch.zeros_like(sorted_values))
    selected = torch.zeros_like(selected_sorted).scatter(-1, sorted_indices, selected_sorted)
    denominator = selected.amax(dim=-1, keepdim=True).clamp_min(config.eps)
    mask = (selected / denominator).clamp(0.0, 1.0) * config.max_mask_value
    metadata = {
        "positive_coordinate_count": int((positive > 0).sum().item()),
        "selected_coordinate_count": int((mask > 0).sum().item()),
        "selected_token_count": int((mask > 0).any(dim=-1).sum().item()),
        "eligible_token_count": int(improves_over_target.sum().item()),
        "skipped_token_count": int((~improves_over_target).sum().item()),
        "top_mass": float(config.top_mass),
        "ci_positive_only": bool(config.ci_positive_only),
        "max_mask_value": float(config.max_mask_value),
    }
    mask = mask.to(dtype=values.dtype)
    _require_finite_tensors(attribution_mask=mask)
    return mask, metadata


def apply_attributed_intervention(
    current: torch.Tensor,
    safe_prediction: torch.Tensor,
    mask: torch.Tensor,
    *,
    unsafe_direction: torch.Tensor,
    safe_direction: torch.Tensor,
    lambda_t: float,
    trust_region_ratio: float,
    margin: float,
    tau: float,
    eps: float,
    backtracking_scales: tuple[float, ...] = DEFAULT_BACKTRACKING_SCALES,
    interpolation_fraction_cap: float = 1.0,
    token_chunk_size: int = 65536,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Roll selected coordinates from current toward the safe-game background.

    Candidate acceptance is token-local and is evaluated only after casting to
    ``current.dtype``.  The first (largest) deterministic scale whose actual
    representable update obeys the relative trust cap and strictly lowers the
    registered unwanted score is accepted.  Tokens for which every scale is
    rejected are returned bitwise unchanged.
    """
    scalar_controls = {
        "lambda_t": lambda_t,
        "trust_region_ratio": trust_region_ratio,
        "margin": margin,
        "tau": tau,
        "eps": eps,
        "interpolation_fraction_cap": interpolation_fraction_cap,
    }
    nonfinite = [
        name
        for name, value in scalar_controls.items()
        if isinstance(value, bool) or not math.isfinite(float(value))
    ]
    if nonfinite:
        raise ValueError(f"Non-finite Shapley intervention controls are invalid: {nonfinite}.")
    if float(lambda_t) < 0.0:
        raise ValueError("Shapley intervention lambda_t must be non-negative.")
    if float(trust_region_ratio) <= 0.0:
        raise ValueError("Shapley intervention trust_region_ratio must be > 0.")
    if float(tau) <= 0.0 or float(eps) <= 0.0:
        raise ValueError("Shapley intervention tau and eps must be > 0.")
    if not 0.0 < float(interpolation_fraction_cap) <= 1.0:
        raise ValueError("Shapley interpolation_fraction_cap must be in (0, 1].")
    if isinstance(token_chunk_size, bool) or not isinstance(token_chunk_size, int):
        raise ValueError("Shapley intervention token_chunk_size must be an integer >= 1.")
    if token_chunk_size < 1:
        raise ValueError("Shapley intervention token_chunk_size must be >= 1.")
    scales = _validate_backtracking_scales(backtracking_scales)

    tensors = (current, safe_prediction, mask, unsafe_direction, safe_direction)
    _require_canonical_shapes(*tensors)
    if any(not tensor.is_floating_point() for tensor in tensors):
        raise ValueError("Shapley intervention tensors must use floating-point dtypes.")
    if any(tensor.device != current.device for tensor in tensors[1:]):
        raise ValueError("Shapley intervention tensors must be on the same device.")
    _require_bitwise_supported_dtype(current.dtype)
    _require_finite_tensors(
        current=current,
        safe_prediction=safe_prediction,
        mask=mask,
        unsafe_direction=unsafe_direction,
        safe_direction=safe_direction,
    )

    shape = current.shape
    feature_count = int(shape[-1])
    current_flat = current.reshape(-1, feature_count)
    safe_flat = safe_prediction.reshape(-1, feature_count)
    mask_flat = mask.reshape(-1, feature_count)
    unsafe_flat = unsafe_direction.reshape(-1, feature_count)
    safe_direction_flat = safe_direction.reshape(-1, feature_count)
    steered_flat = current_flat.clone()

    scale_counts = {_scale_label(scale): 0 for scale in scales}
    scale_counts["0"] = 0
    integer_totals = {
        "selected_coordinate_count": 0,
        "selected_token_count": 0,
        "interpolation_saturated_coordinate_count": 0,
        "trust_capped_proposal_token_count": 0,
        "accepted_token_count": 0,
        "rejected_to_noop_token_count": 0,
        "changed_selected_coordinate_count": 0,
        "nonzero_delta_coordinate_count": 0,
        "score_decreased_token_count": 0,
        "post_quantization_trust_cap_violation_count": 0,
        "unselected_bit_change_count": 0,
        "nondecreasing_accepted_token_count": 0,
    }
    score_before_sum = 0.0
    score_after_sum = 0.0
    total_score_decrease = 0.0
    minimum_accepted_score_decrease: float | None = None
    maximum_accepted_score_decrease: float | None = None
    max_actual_ratio = 0.0

    for start in range(0, current_flat.shape[0], token_chunk_size):
        end = min(start + token_chunk_size, current_flat.shape[0])
        chunk, chunk_metadata = _apply_attributed_intervention_chunk(
            current_flat[start:end],
            safe_flat[start:end],
            mask_flat[start:end],
            unsafe_direction=unsafe_flat[start:end],
            safe_direction=safe_direction_flat[start:end],
            lambda_t=float(lambda_t),
            trust_region_ratio=float(trust_region_ratio),
            margin=float(margin),
            tau=float(tau),
            eps=float(eps),
            backtracking_scales=scales,
            interpolation_fraction_cap=float(interpolation_fraction_cap),
        )
        steered_flat[start:end].copy_(chunk)
        for key in integer_totals:
            integer_totals[key] += int(chunk_metadata[key])
        for key in scale_counts:
            scale_counts[key] += int(chunk_metadata["backtracking_scale_counts"][key])
        score_before_sum += float(chunk_metadata["score_before_sum"])
        score_after_sum += float(chunk_metadata["score_after_sum"])
        total_score_decrease += float(chunk_metadata["total_score_decrease"])
        chunk_minimum = chunk_metadata["minimum_accepted_score_decrease"]
        if chunk_minimum is not None:
            minimum_accepted_score_decrease = (
                float(chunk_minimum)
                if minimum_accepted_score_decrease is None
                else min(minimum_accepted_score_decrease, float(chunk_minimum))
            )
        chunk_maximum = chunk_metadata["maximum_accepted_score_decrease"]
        if chunk_maximum is not None:
            maximum_accepted_score_decrease = (
                float(chunk_maximum)
                if maximum_accepted_score_decrease is None
                else max(maximum_accepted_score_decrease, float(chunk_maximum))
            )
        max_actual_ratio = max(
            max_actual_ratio,
            float(chunk_metadata["max_actual_update_to_current_norm_ratio"]),
        )

    steered = steered_flat.reshape(shape)
    delta = steered - current
    _require_finite_tensors(steered=steered, applied_delta=delta)
    accepted_count = integer_totals["accepted_token_count"]
    metadata = {
        "schema_version": SHAPLEY_TRACE_SCHEMA_VERSION,
        "intervention_identity": SHAPLEY_INTERVENTION_IDENTITY,
        "endpoint": "safe_prompt_vector_field_prediction",
        "trust_region_ratio": float(trust_region_ratio),
        "interpolation_fraction_cap": float(interpolation_fraction_cap),
        "backtracking_scales": list(scales),
        "backtracking_scale_counts": scale_counts,
        "accepted_token_counts_by_scale": {
            _scale_label(scale): scale_counts[_scale_label(scale)] for scale in scales
        },
        "candidate_dtype": str(current.dtype).removeprefix("torch."),
        "score_dtype": "float32",
        "candidate_quantized_before_score": True,
        "coordinate_interpolation_capped_at_safe_endpoint": True,
        "post_quantization_trust_check": True,
        "strict_score_decrease_required": True,
        "token_count": int(current_flat.shape[0]),
        "token_chunk_size": int(token_chunk_size),
        **integer_totals,
        # Compatibility alias for the old pre-quantization cap counter name.
        "capped_token_count": integer_totals["trust_capped_proposal_token_count"],
        "score_before_sum": score_before_sum,
        "score_after_sum": score_after_sum,
        "total_score_decrease": total_score_decrease,
        "minimum_accepted_score_decrease": minimum_accepted_score_decrease,
        "maximum_accepted_score_decrease": maximum_accepted_score_decrease,
        "mean_accepted_score_decrease": (
            total_score_decrease / accepted_count if accepted_count else None
        ),
        "max_actual_update_to_current_norm_ratio": max_actual_ratio,
        # Compatibility alias retained for historical report consumers.
        "max_applied_to_current_norm_ratio": max_actual_ratio,
    }
    return steered, delta, metadata


def _apply_attributed_intervention_chunk(
    current: torch.Tensor,
    safe_prediction: torch.Tensor,
    mask: torch.Tensor,
    *,
    unsafe_direction: torch.Tensor,
    safe_direction: torch.Tensor,
    lambda_t: float,
    trust_region_ratio: float,
    margin: float,
    tau: float,
    eps: float,
    backtracking_scales: tuple[float, ...],
    interpolation_fraction_cap: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply the deterministic protocol to one flattened token chunk."""
    selected = mask > 0
    selected_coordinate_count = int(selected.sum().item())
    selected_token_count = int(selected.any(dim=-1).sum().item())
    current_f = current.float()
    unclamped_fraction = lambda_t * mask.float()
    saturated = unclamped_fraction > interpolation_fraction_cap
    fraction = unclamped_fraction.clamp(
        min=0.0,
        max=interpolation_fraction_cap,
    )
    raw_rollback = fraction * (safe_prediction.float() - current_f)
    current_norm = torch.linalg.vector_norm(current_f, dim=-1)
    raw_norm = torch.linalg.vector_norm(raw_rollback, dim=-1)
    trust_cap = trust_region_ratio * current_norm
    _require_finite_tensors(
        intervention_fraction=fraction,
        raw_safe_prediction_rollback=raw_rollback,
        current_token_norm=current_norm,
        raw_rollback_norm=raw_norm,
        trust_cap=trust_cap,
    )
    proposal_scale = torch.where(
        raw_norm > 0,
        torch.minimum(torch.ones_like(raw_norm), trust_cap / raw_norm),
        torch.zeros_like(raw_norm),
    )
    capped_rollback = raw_rollback * proposal_scale.unsqueeze(-1)
    _require_finite_tensors(
        proposal_trust_scale=proposal_scale,
        trust_capped_rollback=capped_rollback,
    )
    q_before = local_unwanted_score(
        current,
        unsafe_direction,
        safe_direction,
        margin=margin,
        tau=tau,
        eps=eps,
    )

    returned = current.clone()
    accepted = torch.zeros(current.shape[0], dtype=torch.bool, device=current.device)
    scale_counts = {_scale_label(scale): 0 for scale in backtracking_scales}
    for scale in backtracking_scales:
        # Only one model-dtype candidate and one candidate score are live here.
        quantized_candidate = (current_f + scale * capped_rollback).to(dtype=current.dtype)
        candidate = torch.where(selected, quantized_candidate, current)
        changed_selected = _bitwise_difference_mask(candidate, current) & selected
        changed_token = changed_selected.any(dim=-1)
        actual_delta = _actual_delta_float32(candidate, current)
        actual_norm = torch.linalg.vector_norm(actual_delta, dim=-1)
        _require_finite_tensors(
            quantized_candidate=candidate,
            actual_quantized_delta=actual_delta,
            actual_quantized_delta_norm=actual_norm,
        )
        within_trust_cap = actual_norm <= trust_cap
        q_candidate = local_unwanted_score(
            candidate,
            unsafe_direction,
            safe_direction,
            margin=margin,
            tau=tau,
            eps=eps,
        )
        accept = (
            (~accepted)
            & changed_token
            & within_trust_cap
            & (q_candidate < q_before)
        )
        if bool(accept.any().item()):
            returned[accept] = candidate[accept]
            accepted[accept] = True
            scale_counts[_scale_label(scale)] = int(accept.sum().item())

    rejected = ~accepted
    scale_counts["0"] = int(rejected.sum().item())
    returned_score = local_unwanted_score(
        returned,
        unsafe_direction,
        safe_direction,
        margin=margin,
        tau=tau,
        eps=eps,
    )
    returned_delta = _actual_delta_float32(returned, current)
    returned_norm = torch.linalg.vector_norm(returned_delta, dim=-1)
    actual_ratio = torch.where(
        current_norm > 0,
        returned_norm / current_norm,
        torch.zeros_like(returned_norm),
    )
    bit_changes = _bitwise_difference_mask(returned, current)
    changed_selected = bit_changes & selected
    unselected_bit_changes = bit_changes & (~selected)
    accepted_decrease = q_before[accepted] - returned_score[accepted]
    post_quantization_trust_violations = accepted & (returned_norm > trust_cap)
    nondecreasing_accepted = accepted & (returned_score >= q_before)
    score_decreased = returned_score < q_before
    return returned, {
        "selected_coordinate_count": selected_coordinate_count,
        "selected_token_count": selected_token_count,
        "interpolation_saturated_coordinate_count": int(saturated.sum().item()),
        "trust_capped_proposal_token_count": int((raw_norm > trust_cap).sum().item()),
        "accepted_token_count": int(accepted.sum().item()),
        "rejected_to_noop_token_count": int(rejected.sum().item()),
        "changed_selected_coordinate_count": int(changed_selected.sum().item()),
        "nonzero_delta_coordinate_count": int((returned_delta != 0).sum().item()),
        "score_decreased_token_count": int(score_decreased.sum().item()),
        "post_quantization_trust_cap_violation_count": int(
            post_quantization_trust_violations.sum().item()
        ),
        "unselected_bit_change_count": int(unselected_bit_changes.sum().item()),
        "nondecreasing_accepted_token_count": int(nondecreasing_accepted.sum().item()),
        "backtracking_scale_counts": scale_counts,
        "score_before_sum": float(q_before.double().sum().item()),
        "score_after_sum": float(returned_score.double().sum().item()),
        "total_score_decrease": float(accepted_decrease.double().sum().item()),
        "minimum_accepted_score_decrease": (
            float(accepted_decrease.min().item()) if accepted_decrease.numel() else None
        ),
        "maximum_accepted_score_decrease": (
            float(accepted_decrease.max().item()) if accepted_decrease.numel() else None
        ),
        "max_actual_update_to_current_norm_ratio": float(actual_ratio.max().item()),
    }


def _empty_trace_aggregate() -> dict[str, int | float]:
    return {
        "expected_step_count": 0,
        "observed_step_count": 0,
        "selected_coordinate_count": 0,
        "selected_token_count": 0,
        "accepted_token_count": 0,
        "score_decreased_token_count": 0,
        "nonzero_delta_coordinate_count": 0,
        "nonzero_delta_step_count": 0,
        "score_decrease_step_count": 0,
        "total_score_decrease": 0.0,
        "post_quantization_trust_cap_violation_count": 0,
        "unselected_bit_change_count": 0,
        "nondecreasing_accepted_token_count": 0,
        "final_active_pair_regression_count": 0,
    }


def _trace_aggregate_failures(
    aggregates: Mapping[str, Mapping[str, int | float]], *, scope: str
) -> list[str]:
    failures: list[str] = []
    for pair_id, aggregate in aggregates.items():
        prefix = f"{scope} pair '{pair_id}'"
        if aggregate["expected_step_count"] == 0:
            failures.append(f"{prefix} was not scheduled at any denoising step")
        if aggregate["observed_step_count"] != aggregate["expected_step_count"]:
            failures.append(
                f"{prefix} expected {aggregate['expected_step_count']} observations "
                f"but recorded {aggregate['observed_step_count']}"
            )
        if aggregate["selected_coordinate_count"] <= 0:
            failures.append(f"{prefix} selected zero coordinates")
        if aggregate["nonzero_delta_coordinate_count"] <= 0:
            failures.append(f"{prefix} applied zero delta")
        if aggregate["accepted_token_count"] <= 0:
            failures.append(f"{prefix} accepted zero tokens")
        if aggregate["nonzero_delta_step_count"] <= 0:
            failures.append(f"{prefix} has zero nonzero-delta steps")
        if aggregate["score_decrease_step_count"] <= 0:
            failures.append(f"{prefix} has zero score-decrease steps")
        if aggregate["total_score_decrease"] <= 0.0:
            failures.append(f"{prefix} has no positive score decrease")
        if any(
            aggregate[key] != 0
            for key in (
                "post_quantization_trust_cap_violation_count",
                "unselected_bit_change_count",
                "nondecreasing_accepted_token_count",
                "final_active_pair_regression_count",
            )
        ):
            failures.append(f"{prefix} has nonzero validation violations")
    return failures


class ShapleyConceptSteerer(HierarchicalVectorFieldBottleneck):
    """Sequential pair-local latent channel attribution and intervention."""

    def __init__(
        self,
        hierarchy: ConceptHierarchy,
        bottleneck_config: BottleneckConfig,
        shapley_config: ShapleyConfig,
    ) -> None:
        super().__init__(hierarchy=hierarchy, config=bottleneck_config)
        self.shapley_config = shapley_config

    def steer_step(
        self,
        adapter: Any,
        latents: torch.Tensor,
        timestep: Any,
        state: Any,
        prompt: str,
        step_index: int,
        num_steps: int,
    ) -> tuple[torch.Tensor, BottleneckTrace]:
        local_step_index, local_num_steps = self.schedule_coordinates(
            state, step_index, num_steps
        )
        enabled = self._shapley_step_enabled(local_step_index, local_num_steps)

        v_base = self._predict(
            adapter,
            latents,
            timestep,
            state,
            prompt,
            call_role="base_current",
        )
        _require_finite_tensors(base_vector_field=v_base)
        if not enabled:
            stats = tensor_stats(v_base)
            return v_base, BottleneckTrace(
                step_index=step_index,
                timestep=_timestep_to_log_value(timestep),
                enabled=False,
                concepts=[],
                base_stats=stats,
                steered_stats=stats,
            )

        lambda_t = self.config.lambda_schedule.value(local_step_index, local_num_steps)
        v_current = v_base
        traces: list[ConceptTrace] = []
        audited_pairs: list[tuple[ConceptPair, torch.Tensor]] = []
        neutral_prompt = self._compose_prompt(prompt, self.hierarchy.neutral_concept)
        v_neutral = self._predict(
            adapter,
            latents,
            timestep,
            state,
            neutral_prompt,
            call_role="neutral",
        )
        _require_finite_tensors(neutral_vector_field=v_neutral)

        # Pair order is the hierarchy/active_pair order.  Every later game uses
        # the already-updated v_current, so attribution is recomputed rather
        # than reusing a stale mask from the original base field.
        for pair_index, pair in enumerate(self.active_pairs):
            if not self._pair_enabled(pair.id, local_step_index, local_num_steps):
                continue
            v_current, trace, pair_post_score = self._apply_shapley_pair(
                adapter=adapter,
                latents=latents,
                timestep=timestep,
                state=state,
                prompt=prompt,
                pair=pair,
                pair_index=pair_index,
                step_index=step_index,
                v_current=v_current,
                v_neutral=v_neutral,
                lambda_t=self._pair_lambda(pair.id, lambda_t),
            )
            traces.append(trace)
            audited_pairs.append((pair, pair_post_score))

        self._audit_final_pair_non_regression(
            adapter=adapter,
            latents=latents,
            timestep=timestep,
            state=state,
            prompt=prompt,
            v_final=v_current,
            v_neutral=v_neutral,
            traces=traces,
            audited_pairs=audited_pairs,
        )

        _require_finite_tensors(steered_vector_field=v_current)
        return v_current, BottleneckTrace(
            step_index=step_index,
            timestep=_timestep_to_log_value(timestep),
            enabled=True,
            concepts=traces,
            base_stats=tensor_stats(v_base),
            steered_stats=tensor_stats(v_current),
        )

    def _audit_final_pair_non_regression(
        self,
        *,
        adapter: Any,
        latents: torch.Tensor,
        timestep: Any,
        state: Any,
        prompt: str,
        v_final: torch.Tensor,
        v_neutral: torch.Tensor,
        traces: list[ConceptTrace],
        audited_pairs: list[tuple[ConceptPair, torch.Tensor]],
    ) -> None:
        """Fail closed if a later sequential pair regresses an earlier game."""
        if len(traces) != len(audited_pairs):
            raise RuntimeError("Internal Shapley final-pair audit coverage mismatch.")
        layout: LatentLayout = adapter.latent_layout(v_final)
        final_c = _materialize_canonical(layout, v_final, label="final_vector_field")
        failures: list[str] = []
        for trace, (pair, pair_post_score) in zip(traces, audited_pairs):
            unsafe_prompt = self._compose_prompt(prompt, pair.unsafe_concept)
            safe_prompt = self._compose_prompt(prompt, pair.safe_sibling_concept)
            v_unsafe = self._predict(
                adapter,
                latents,
                timestep,
                state,
                unsafe_prompt,
                call_role=f"unsafe_source:{pair.id}",
            )
            v_safe = self._predict(
                adapter,
                latents,
                timestep,
                state,
                safe_prompt,
                call_role=f"safe_target:{pair.id}",
            )
            basis = compute_concept_basis(
                v_unsafe,
                v_safe,
                v_neutral,
                normalize=self.config.normalize_directions,
                eps=self.shapley_config.eps,
            )
            unsafe_direction_c = _materialize_canonical(
                layout,
                basis.unsafe,
                label=f"final_audit_{pair.id}_unsafe_direction",
            )
            safe_direction_c = _materialize_canonical(
                layout,
                basis.safe,
                label=f"final_audit_{pair.id}_safe_direction",
            )
            final_score = local_unwanted_score(
                final_c,
                unsafe_direction_c,
                safe_direction_c,
                margin=self.config.margin,
                tau=self.shapley_config.tau,
                eps=self.shapley_config.eps,
            )
            if final_score.shape != pair_post_score.shape:
                raise RuntimeError(
                    f"Final Shapley score shape changed for pair '{pair.id}'."
                )
            regression = final_score > pair_post_score
            regression_count = int(regression.sum().item())
            score_change = final_score.double() - pair_post_score.double()
            trace.shapley["final_active_pair_non_regression"] = {
                "schema_version": SHAPLEY_TRACE_SCHEMA_VERSION,
                "contract": (
                    "final_sequential_field_score_must_not_exceed_"
                    "pair_local_post_intervention_score_per_token"
                ),
                "audited": True,
                "regressed_token_count": regression_count,
                "nonregressed_token_count": int((~regression).sum().item()),
                "pair_local_post_score_sum": float(pair_post_score.double().sum().item()),
                "final_score_sum": float(final_score.double().sum().item()),
                "total_score_change": float(score_change.sum().item()),
                "maximum_score_regression": (
                    float(score_change[regression].max().item()) if regression_count else 0.0
                ),
                "passed": regression_count == 0,
            }
            if regression_count:
                failures.append(f"pair '{pair.id}' regressed at {regression_count} tokens")
        if failures:
            raise RuntimeError(
                "Final Shapley active-pair non-regression audit failed: "
                + "; ".join(failures)
                + "."
            )

    def expected_pair_ids_for_step(self, step_index: int, num_steps: int) -> tuple[str, ...]:
        """Return the exact ordered pair coverage required for one denoising step."""
        if not self._shapley_step_enabled(step_index, num_steps):
            return ()
        return tuple(
            pair.id
            for pair in self.active_pairs
            if self._pair_enabled(pair.id, step_index, num_steps)
        )

    def validate_run_trace(
        self,
        trace: list[dict[str, Any]],
        *,
        num_steps: int,
    ) -> dict[str, Any]:
        """Fail closed on missing pair/step coverage or an inert active intervention.

        This is intentionally called before decoding or saving media.  Per-step
        no-ops remain permissible because a concept can be absent at a particular
        noise level; the selected-coordinate and applied-delta requirements are
        aggregated over every scheduled step for each active pair.
        """
        if num_steps < 1:
            raise RuntimeError("Shapley trace validation requires at least one denoising step.")
        if len(trace) != num_steps:
            raise RuntimeError(
                "Shapley trace coverage mismatch: "
                f"expected {num_steps} step records, observed {len(trace)}."
            )

        aggregates = {pair.id: _empty_trace_aggregate() for pair in self.active_pairs}
        segment_aggregates: dict[int, dict[str, dict[str, int | float]]] = {}
        expected_scales = list(self.shapley_config.backtracking_scales)
        expected_scale_labels = [_scale_label(scale) for scale in expected_scales]
        expected_budget_stages = list(
            _permutation_budget_stages(
                self.shapley_config.min_permutations,
                self.shapley_config.max_permutations,
            )
        )
        segment_evidence_present = any(
            isinstance(step, dict) and step.get("segment") is not None for step in trace
        )
        for step_index, step in enumerate(trace):
            if not isinstance(step, dict):
                raise RuntimeError(
                    f"Shapley trace step {step_index} must be a mapping, got {type(step).__name__}."
                )
            if step.get("step_index") != step_index:
                raise RuntimeError(
                    "Shapley trace step order/index mismatch: "
                    f"position {step_index} reports step_index={step.get('step_index')!r}."
                )
            if segment_evidence_present:
                segment = _validated_segment_record(
                    step,
                    step_index=step_index,
                    num_steps=num_steps,
                )
                _validate_condition_calls(step, segment)
                local_step_index = int(segment["local_step_index"])
                local_num_steps = int(segment["local_num_steps"])
                segment_index = int(segment["segment_index"])
            else:
                local_step_index = step_index
                local_num_steps = num_steps
                segment_index = 0
            per_segment = segment_aggregates.setdefault(
                segment_index,
                {pair.id: _empty_trace_aggregate() for pair in self.active_pairs},
            )
            expected_enabled = self._shapley_step_enabled(
                local_step_index,
                local_num_steps,
            )
            if step.get("enabled") is not expected_enabled:
                raise RuntimeError(
                    f"Shapley trace enabled flag mismatch at step {step_index}: "
                    f"expected {expected_enabled}, observed {step.get('enabled')!r}."
                )
            expected_pair_ids = self.expected_pair_ids_for_step(
                local_step_index,
                local_num_steps,
            )
            concepts = step.get("concepts")
            if not isinstance(concepts, list):
                raise RuntimeError(f"Shapley trace concepts at step {step_index} must be a list.")
            observed_pair_ids = tuple(
                concept.get("concept_id") if isinstance(concept, dict) else None
                for concept in concepts
            )
            if observed_pair_ids != expected_pair_ids:
                raise RuntimeError(
                    f"Shapley pair coverage mismatch at step {step_index}: "
                    f"expected {list(expected_pair_ids)}, observed {list(observed_pair_ids)}."
                )

            for pair_id in expected_pair_ids:
                aggregates[pair_id]["expected_step_count"] += 1
                per_segment[pair_id]["expected_step_count"] += 1
            for concept in concepts:
                pair_id = str(concept["concept_id"])
                shapley = concept.get("shapley")
                if not isinstance(shapley, dict):
                    raise RuntimeError(
                        f"Shapley metadata is missing for pair '{pair_id}' at step {step_index}."
                    )
                if shapley.get("schema_version") != SHAPLEY_TRACE_SCHEMA_VERSION:
                    raise RuntimeError(
                        f"Shapley schema version mismatch for pair '{pair_id}' at step "
                        f"{step_index}."
                    )
                if shapley.get("intervention_identity") != SHAPLEY_INTERVENTION_IDENTITY:
                    raise RuntimeError(
                        f"Shapley intervention identity mismatch for pair '{pair_id}' at "
                        f"step {step_index}."
                    )
                estimator = shapley.get("estimator")
                if not isinstance(estimator, dict):
                    raise RuntimeError(
                        f"Estimator metadata is missing for pair '{pair_id}' at step {step_index}."
                    )
                exact = estimator.get("exact")
                if estimator.get("converged") is not True and (
                    exact is True or self.shapley_config.require_convergence
                ):
                    raise RuntimeError(
                        f"Unconverged Shapley estimate for pair '{pair_id}' at step "
                        f"{step_index}."
                    )
                if exact is True:
                    if (
                        estimator.get("name") != "exact_affine_enumeration"
                        or estimator.get("stopping_reason") != "exact_enumeration"
                    ):
                        raise RuntimeError(
                            f"Invalid exact Shapley estimator metadata for pair '{pair_id}' "
                            f"at step {step_index}."
                        )
                elif exact is False:
                    orders_used = _require_nonnegative_int(
                        estimator.get("permutation_orders_used"),
                        label=f"permutation_orders_used for pair '{pair_id}' at step {step_index}",
                    )
                    if (
                        estimator.get("name")
                        != "deterministic_staged_antithetic_permutation"
                        or estimator.get("permutation_orders_budget")
                        != self.shapley_config.max_permutations
                        or estimator.get("minimum_permutation_orders")
                        != self.shapley_config.min_permutations
                        or estimator.get("permutation_budget_stages") != expected_budget_stages
                        or orders_used not in expected_budget_stages
                        or estimator.get("completed_budget_stages")
                        != [stage for stage in expected_budget_stages if stage <= orders_used]
                        or estimator.get("confidence_distribution")
                        != "student_t_over_independent_antithetic_pair_averages"
                    ):
                        raise RuntimeError(
                            f"Invalid approximate Shapley estimator protocol for pair "
                            f"'{pair_id}' at step {step_index}."
                        )
                else:
                    raise RuntimeError(
                        f"Shapley estimator exact flag is invalid for pair '{pair_id}' at "
                        f"step {step_index}."
                    )
                common_random_numbers = shapley.get("common_random_numbers")
                estimator_seed = _require_nonnegative_int(
                    estimator.get("deterministic_seed"),
                    label=f"deterministic_seed for pair '{pair_id}' at step {step_index}",
                )
                if (
                    not isinstance(common_random_numbers, dict)
                    or common_random_numbers.get("configured_seed") != self.shapley_config.seed
                ):
                    raise RuntimeError(
                        f"Shapley deterministic seed provenance mismatch for pair '{pair_id}' "
                        f"at step {step_index}."
                    )
                common_seed = _require_nonnegative_int(
                    common_random_numbers.get("deterministic_seed"),
                    label=(
                        f"common-random-number deterministic_seed for pair '{pair_id}' "
                        f"at step {step_index}"
                    ),
                )
                if common_seed != estimator_seed:
                    raise RuntimeError(
                        f"Shapley deterministic seed provenance mismatch for pair '{pair_id}' "
                        f"at step {step_index}."
                    )
                selection = shapley.get("mask_selection")
                intervention = shapley.get("intervention_validation")
                if not isinstance(selection, dict) or not isinstance(intervention, dict):
                    raise RuntimeError(
                        f"Selection/intervention validation metadata is missing for pair "
                        f"'{pair_id}' at step {step_index}."
                    )
                selected_count = _require_nonnegative_int(
                    selection.get("selected_coordinate_count"),
                    label=(
                        f"selected_coordinate_count for pair '{pair_id}' at step {step_index}"
                    ),
                )
                intervention_selected_count = _require_nonnegative_int(
                    intervention.get("selected_coordinate_count"),
                    label=(
                        f"intervention selected_coordinate_count for pair '{pair_id}' "
                        f"at step {step_index}"
                    ),
                )
                if intervention_selected_count != selected_count:
                    raise RuntimeError(
                        f"Selected-coordinate count mismatch for pair '{pair_id}' at step "
                        f"{step_index}: selection reports {selected_count}, intervention "
                        f"reports {intervention_selected_count}."
                    )
                selected_token_count = _require_nonnegative_int(
                    selection.get("selected_token_count"),
                    label=f"selected_token_count for pair '{pair_id}' at step {step_index}",
                )
                if _require_nonnegative_int(
                    intervention.get("selected_token_count"),
                    label=(
                        f"intervention selected_token_count for pair '{pair_id}' "
                        f"at step {step_index}"
                    ),
                ) != selected_token_count:
                    raise RuntimeError(
                        f"Selected-token count mismatch for pair '{pair_id}' at step "
                        f"{step_index}."
                    )
                if (
                    intervention.get("schema_version") != SHAPLEY_TRACE_SCHEMA_VERSION
                    or intervention.get("intervention_identity")
                    != SHAPLEY_INTERVENTION_IDENTITY
                    or intervention.get("endpoint")
                    != "safe_prompt_vector_field_prediction"
                    or intervention.get("candidate_dtype")
                    not in {"float16", "bfloat16", "float32", "float64"}
                    or intervention.get("candidate_quantized_before_score") is not True
                    or intervention.get("score_dtype") != "float32"
                    or intervention.get("coordinate_interpolation_capped_at_safe_endpoint")
                    is not True
                    or intervention.get("post_quantization_trust_check") is not True
                    or intervention.get("strict_score_decrease_required") is not True
                    or intervention.get("backtracking_scales") != expected_scales
                    or intervention.get("interpolation_fraction_cap")
                    != self.shapley_config.interpolation_fraction_cap
                ):
                    raise RuntimeError(
                        f"Shapley intervention provenance is invalid for pair '{pair_id}' "
                        f"at step {step_index}."
                    )
                scale_counts = intervention.get("backtracking_scale_counts")
                if not isinstance(scale_counts, dict) or set(scale_counts) != {
                    *expected_scale_labels,
                    "0",
                }:
                    raise RuntimeError(
                        f"Backtracking scale counts are invalid for pair '{pair_id}' at "
                        f"step {step_index}."
                    )
                validated_scale_counts = {
                    label: _require_nonnegative_int(
                        scale_counts[label],
                        label=(
                            f"backtracking scale {label} count for pair '{pair_id}' "
                            f"at step {step_index}"
                        ),
                    )
                    for label in scale_counts
                }
                accepted_count = _require_nonnegative_int(
                    intervention.get("accepted_token_count"),
                    label=f"accepted_token_count for pair '{pair_id}' at step {step_index}",
                )
                if accepted_count != sum(
                    validated_scale_counts[label] for label in expected_scale_labels
                ):
                    raise RuntimeError(
                        f"Accepted-token/backtracking-scale mismatch for pair '{pair_id}' "
                        f"at step {step_index}."
                    )
                rejected_count = _require_nonnegative_int(
                    intervention.get("rejected_to_noop_token_count"),
                    label=(
                        f"rejected_to_noop_token_count for pair '{pair_id}' "
                        f"at step {step_index}"
                    ),
                )
                token_count = _require_nonnegative_int(
                    intervention.get("token_count"),
                    label=f"token_count for pair '{pair_id}' at step {step_index}",
                )
                if (
                    validated_scale_counts["0"] != rejected_count
                    or accepted_count + rejected_count != token_count
                ):
                    raise RuntimeError(
                        f"Rejected-token/backtracking-scale mismatch for pair '{pair_id}' "
                        f"at step {step_index}."
                    )
                score_decreased_count = _require_nonnegative_int(
                    intervention.get("score_decreased_token_count"),
                    label=(
                        f"score_decreased_token_count for pair '{pair_id}' "
                        f"at step {step_index}"
                    ),
                )
                if score_decreased_count != accepted_count:
                    raise RuntimeError(
                        f"Accepted tokens did not all strictly decrease score for pair "
                        f"'{pair_id}' at step {step_index}."
                    )
                delta_count = _require_nonnegative_int(
                    intervention.get("nonzero_delta_coordinate_count"),
                    label=(
                        f"nonzero_delta_coordinate_count for pair '{pair_id}' "
                        f"at step {step_index}"
                    ),
                )
                changed_selected_count = _require_nonnegative_int(
                    intervention.get("changed_selected_coordinate_count"),
                    label=(
                        f"changed_selected_coordinate_count for pair '{pair_id}' "
                        f"at step {step_index}"
                    ),
                )
                violation_values = {
                    key: _require_nonnegative_int(
                        intervention.get(key),
                        label=f"{key} for pair '{pair_id}' at step {step_index}",
                    )
                    for key in (
                        "post_quantization_trust_cap_violation_count",
                        "unselected_bit_change_count",
                        "nondecreasing_accepted_token_count",
                    )
                }
                if any(violation_values.values()):
                    raise RuntimeError(
                        f"Shapley intervention validation violation for pair '{pair_id}' "
                        f"at step {step_index}: {violation_values}."
                    )
                total_score_decrease = _require_nonnegative_finite_float(
                    intervention.get("total_score_decrease"),
                    label=f"total_score_decrease for pair '{pair_id}' at step {step_index}",
                )
                if (accepted_count > 0) != (total_score_decrease > 0.0):
                    raise RuntimeError(
                        f"Accepted-token/score-decrease mismatch for pair '{pair_id}' at "
                        f"step {step_index}."
                    )
                if (accepted_count > 0) != (delta_count > 0) or (
                    accepted_count > 0 and changed_selected_count <= 0
                ):
                    raise RuntimeError(
                        f"Accepted-token/nonzero-delta mismatch for pair '{pair_id}' at "
                        f"step {step_index}."
                    )
                minimum_decrease = intervention.get("minimum_accepted_score_decrease")
                if accepted_count:
                    if _require_nonnegative_finite_float(
                        minimum_decrease,
                        label=(
                            f"minimum_accepted_score_decrease for pair '{pair_id}' "
                            f"at step {step_index}"
                        ),
                    ) <= 0.0:
                        raise RuntimeError(
                            f"Accepted Shapley score decrease is not strict for pair "
                            f"'{pair_id}' at step {step_index}."
                        )
                elif minimum_decrease is not None:
                    raise RuntimeError(
                        f"No-op Shapley intervention reports an accepted decrease for pair "
                        f"'{pair_id}' at step {step_index}."
                    )
                final_audit = shapley.get("final_active_pair_non_regression")
                if not isinstance(final_audit, dict):
                    raise RuntimeError(
                        f"Final active-pair audit is missing for pair '{pair_id}' at step "
                        f"{step_index}."
                    )
                regression_count = _require_nonnegative_int(
                    final_audit.get("regressed_token_count"),
                    label=(
                        f"final regressed_token_count for pair '{pair_id}' "
                        f"at step {step_index}"
                    ),
                )
                if (
                    final_audit.get("schema_version") != SHAPLEY_TRACE_SCHEMA_VERSION
                    or final_audit.get("contract")
                    != (
                        "final_sequential_field_score_must_not_exceed_"
                        "pair_local_post_intervention_score_per_token"
                    )
                    or final_audit.get("audited") is not True
                    or final_audit.get("passed") is not True
                ):
                    raise RuntimeError(
                        f"Final active-pair audit did not pass for pair '{pair_id}' at step "
                        f"{step_index}."
                    )
                if regression_count:
                    raise RuntimeError(
                        f"Final active-pair score regressed for pair '{pair_id}' at step "
                        f"{step_index}."
                    )

                aggregate = aggregates[pair_id]
                aggregate["observed_step_count"] += 1
                aggregate["selected_coordinate_count"] += selected_count
                aggregate["selected_token_count"] += selected_token_count
                aggregate["accepted_token_count"] += accepted_count
                aggregate["score_decreased_token_count"] += score_decreased_count
                aggregate["nonzero_delta_coordinate_count"] += delta_count
                aggregate["nonzero_delta_step_count"] += int(delta_count > 0)
                aggregate["score_decrease_step_count"] += int(total_score_decrease > 0.0)
                aggregate["total_score_decrease"] += total_score_decrease
                for key, value in violation_values.items():
                    aggregate[key] += value
                aggregate["final_active_pair_regression_count"] += regression_count

                segment_aggregate = per_segment[pair_id]
                segment_aggregate["observed_step_count"] += 1
                segment_aggregate["selected_coordinate_count"] += selected_count
                segment_aggregate["selected_token_count"] += selected_token_count
                segment_aggregate["accepted_token_count"] += accepted_count
                segment_aggregate["score_decreased_token_count"] += score_decreased_count
                segment_aggregate["nonzero_delta_coordinate_count"] += delta_count
                segment_aggregate["nonzero_delta_step_count"] += int(delta_count > 0)
                segment_aggregate["score_decrease_step_count"] += int(
                    total_score_decrease > 0.0
                )
                segment_aggregate["total_score_decrease"] += total_score_decrease
                for key, value in violation_values.items():
                    segment_aggregate[key] += value
                segment_aggregate["final_active_pair_regression_count"] += regression_count

        failures = _trace_aggregate_failures(aggregates, scope="run")
        for segment_index, per_pair in sorted(segment_aggregates.items()):
            failures.extend(
                _trace_aggregate_failures(per_pair, scope=f"segment {segment_index}")
            )
        if failures:
            raise RuntimeError("Shapley trace validation failed: " + "; ".join(failures) + ".")

        aggregate_totals = {
            key: sum(aggregate[key] for aggregate in aggregates.values())
            for key in (
                "accepted_token_count",
                "score_decreased_token_count",
                "nonzero_delta_step_count",
                "score_decrease_step_count",
                "total_score_decrease",
                "post_quantization_trust_cap_violation_count",
                "unselected_bit_change_count",
                "nondecreasing_accepted_token_count",
                "final_active_pair_regression_count",
            )
        }
        per_segment_validation = []
        for segment_index, per_pair in sorted(segment_aggregates.items()):
            totals = {
                key: sum(aggregate[key] for aggregate in per_pair.values())
                for key in aggregate_totals
            }
            per_segment_validation.append(
                {
                    "segment_index": segment_index,
                    "status": "passed",
                    **totals,
                    "per_pair": per_pair,
                }
            )
        return {
            "schema_version": SHAPLEY_TRACE_SCHEMA_VERSION,
            "status": "passed",
            "intervention_identity": SHAPLEY_INTERVENTION_IDENTITY,
            "require_convergence": bool(self.shapley_config.require_convergence),
            "num_steps": int(num_steps),
            "segment_trace_bound": segment_evidence_present,
            "active_pair_ids": [pair.id for pair in self.active_pairs],
            **aggregate_totals,
            "per_pair": aggregates,
            "per_segment": per_segment_validation,
        }

    def _shapley_step_enabled(self, step_index: int, num_steps: int) -> bool:
        return self.config.enabled and step_is_enabled(
            step_index,
            num_steps,
            start_step=self.config.start_step,
            end_step=self.config.end_step,
            start_fraction=self.config.start_fraction,
            end_fraction=self.config.end_fraction,
        ) and self._stride_enabled(step_index)

    def _apply_shapley_pair(
        self,
        *,
        adapter: Any,
        latents: torch.Tensor,
        timestep: Any,
        state: Any,
        prompt: str,
        pair: ConceptPair,
        pair_index: int,
        step_index: int,
        v_current: torch.Tensor,
        v_neutral: torch.Tensor,
        lambda_t: float,
    ) -> tuple[torch.Tensor, ConceptTrace, torch.Tensor]:
        unsafe_prompt = self._compose_prompt(prompt, pair.unsafe_concept)
        safe_prompt = self._compose_prompt(prompt, pair.safe_sibling_concept)
        v_unsafe = self._predict(
            adapter,
            latents,
            timestep,
            state,
            unsafe_prompt,
            call_role=f"unsafe_source:{pair.id}",
        )
        v_safe = self._predict(
            adapter,
            latents,
            timestep,
            state,
            safe_prompt,
            call_role=f"safe_target:{pair.id}",
        )
        _require_finite_tensors(
            current_vector_field=v_current,
            neutral_vector_field=v_neutral,
            unsafe_vector_field=v_unsafe,
            safe_vector_field=v_safe,
        )
        basis = compute_concept_basis(
            v_unsafe,
            v_safe,
            v_neutral,
            normalize=self.config.normalize_directions,
            eps=self.shapley_config.eps,
        )

        layout: LatentLayout = adapter.latent_layout(v_current)
        current_c = _materialize_canonical(layout, v_current, label="current_vector_field")
        safe_prediction_c = _materialize_canonical(
            layout,
            v_safe,
            label="safe_prediction_vector_field",
        )
        unsafe_direction_c = _materialize_canonical(
            layout,
            basis.unsafe,
            label="unsafe_vector_field_direction",
        )
        safe_direction_c = _materialize_canonical(
            layout,
            basis.safe,
            label="safe_vector_field_direction",
        )
        q_empty = local_unwanted_score(
            safe_prediction_c,
            unsafe_direction_c,
            safe_direction_c,
            margin=self.config.margin,
            tau=self.shapley_config.tau,
            eps=self.shapley_config.eps,
        )
        q_full = local_unwanted_score(
            current_c,
            unsafe_direction_c,
            safe_direction_c,
            margin=self.config.margin,
            tau=self.shapley_config.tau,
            eps=self.shapley_config.eps,
        )
        deterministic_seed = _stable_seed(
            self.shapley_config.seed,
            adapter.adapter_name,
            pair.id,
            step_index,
            current_c.shape[-1],
        )
        estimate = estimate_local_channel_shapley(
            safe_prediction_c,
            current_c,
            unsafe_direction_c,
            safe_direction_c,
            margin=self.config.margin,
            config=self.shapley_config,
            deterministic_seed=deterministic_seed,
        )
        efficiency_residual = (
            estimate.values.float().sum(dim=-1) - (q_full.float() - q_empty.float())
        )
        _require_finite_tensors(efficiency_residual=efficiency_residual)
        max_abs_efficiency_residual = float(efficiency_residual.abs().amax().item())
        if max_abs_efficiency_residual > self.shapley_config.efficiency_tolerance:
            raise RuntimeError(
                "Shapley efficiency residual exceeded steering.shapley.efficiency_tolerance: "
                f"{max_abs_efficiency_residual:.6g} > "
                f"{self.shapley_config.efficiency_tolerance:.6g} for pair '{pair.id}'."
            )
        mask_c, mask_metadata = attribution_to_mask(
            estimate.values,
            estimate.ci_half_width,
            q_empty,
            q_full,
            self.shapley_config,
        )
        steered_c, delta_c, trust_metadata = apply_attributed_intervention(
            current_c,
            safe_prediction_c,
            mask_c,
            unsafe_direction=unsafe_direction_c,
            safe_direction=safe_direction_c,
            lambda_t=lambda_t,
            trust_region_ratio=self.shapley_config.trust_region_ratio,
            margin=self.config.margin,
            tau=self.shapley_config.tau,
            eps=self.shapley_config.eps,
            backtracking_scales=self.shapley_config.backtracking_scales,
            interpolation_fraction_cap=self.shapley_config.interpolation_fraction_cap,
            token_chunk_size=self.shapley_config.token_chunk_size,
        )
        q_after = local_unwanted_score(
            steered_c,
            unsafe_direction_c,
            safe_direction_c,
            margin=self.config.margin,
            tau=self.shapley_config.tau,
            eps=self.shapley_config.eps,
        )
        steered = _materialize_restored(
            layout,
            steered_c,
            dtype=v_current.dtype,
            label="steered_vector_field",
        )
        delta = _materialize_restored(
            layout,
            delta_c,
            dtype=v_current.dtype,
            label="applied_vector_field_delta",
        )
        shapley_metadata = {
            "schema_version": SHAPLEY_TRACE_SCHEMA_VERSION,
            "method": "latent_vector_field_feature_channel_shapley",
            "interpretation": "functional_attribution_not_causal",
            "intervention_identity": SHAPLEY_INTERVENTION_IDENTITY,
            "attribution_provenance": {
                "attributed_object": "denoiser_vector_field_prediction",
                "players": "feature_channel_coordinates_within_each_vector_field_token",
                "token_scope": "one_independent_coalition_game_per_flattened_vector_field_token",
                "not_text_embedding_attribution": True,
                "not_latent_state_causal_attribution": True,
            },
            "game": {
                "players": "denoiser_vector_field_feature_channels_scored_at_each_token",
                "empty_background": "safe_prompt_vector_field_prediction",
                "full_explicand": "current_sequential_vector_field_prediction",
                "value": "softplus((cos(h,b_unsafe)-cos(h,b_safe)+margin)/tau)",
                "margin": float(self.config.margin),
                "tau": float(self.shapley_config.tau),
            },
            "layout": layout.to_dict(),
            "estimator": estimate.metadata,
            "q_empty_stats": tensor_stats(q_empty),
            "q_full_stats": tensor_stats(q_full),
            "full_minus_empty_stats": tensor_stats(q_full - q_empty),
            "attribution_stats": tensor_stats(estimate.values),
            "ci_half_width_stats": tensor_stats(estimate.ci_half_width),
            "efficiency_residual_stats": tensor_stats(efficiency_residual),
            "max_abs_efficiency_residual": max_abs_efficiency_residual,
            "efficiency_tolerance": float(self.shapley_config.efficiency_tolerance),
            "mask_selection": mask_metadata,
            "trust_region": trust_metadata,
            "intervention_validation": {
                **trust_metadata,
                "canonical_vector_fields_contiguous": True,
                "restored_vector_fields_contiguous": True,
            },
            "intervention_provenance": {
                "identity": SHAPLEY_INTERVENTION_IDENTITY,
                "rollback_source": "current_sequential_vector_field_prediction",
                "rollback_endpoint": "safe_prompt_vector_field_prediction",
                "candidate_dtype": str(current_c.dtype).removeprefix("torch."),
                "score_dtype": "float32",
                "candidate_quantized_before_score": True,
                "coordinate_interpolation_capped_at_safe_endpoint": True,
                "interpolation_fraction_cap": float(
                    self.shapley_config.interpolation_fraction_cap
                ),
                "backtracking_scales": list(self.shapley_config.backtracking_scales),
                "strict_score_decrease_required": True,
                "post_quantization_trust_check": True,
            },
            "common_random_numbers": {
                "configured_seed": int(self.shapley_config.seed),
                "deterministic_seed": int(deterministic_seed),
                "seed_components": [
                    "configured_seed",
                    "adapter_name",
                    "stable_pair_id",
                    "denoising_step_index",
                    "feature_channel_count",
                ],
                "independent_of_active_pair_list_position": True,
            },
            "pair_order_index": int(pair_index),
        }
        trace = ConceptTrace(
            concept_id=pair.id,
            parent=pair.parent,
            lambda_t=float(lambda_t),
            activation=tensor_stats(q_full - q_empty),
            mask=tensor_stats(mask_c),
            unsafe_concept=pair.unsafe_concept,
            target_concept=pair.safe_sibling_concept,
            steering_delta_stats=tensor_stats(delta),
            unsafe_direction_stats=tensor_stats(basis.unsafe),
            safe_direction_stats=tensor_stats(basis.safe),
            shapley=shapley_metadata,
        )
        return steered, trace, q_after.detach()


def _estimate_exact(
    h_empty: torch.Tensor,
    h_full: torch.Tensor,
    unsafe_direction: torch.Tensor,
    safe_direction: torch.Tensor,
    *,
    margin: float,
    config: ShapleyConfig,
) -> ShapleyEstimate:
    batch, tokens, players = h_full.shape
    positions = batch * tokens
    empty = h_empty.float().reshape(positions, players)
    full = h_full.float().reshape(positions, players)
    unsafe = unsafe_direction.float().reshape(positions, players)
    safe = safe_direction.float().reshape(positions, players)
    coalition_count = 2**players
    transform = _exact_shapley_transform(players).to(device=full.device)
    values = torch.zeros((positions, players), device=full.device, dtype=torch.float32)

    for token_start in range(0, positions, config.token_chunk_size):
        token_end = min(token_start + config.token_chunk_size, positions)
        accumulator = torch.zeros(
            (players, token_end - token_start),
            device=full.device,
            dtype=torch.float32,
        )
        for coalition_start in range(0, coalition_count, config.coalition_chunk_size):
            coalition_end = min(
                coalition_start + config.coalition_chunk_size,
                coalition_count,
            )
            masks = _coalition_masks(
                coalition_start,
                coalition_end,
                players,
                device=full.device,
            )
            scores = _scores_for_masks(
                empty[token_start:token_end],
                full[token_start:token_end],
                unsafe[token_start:token_end],
                safe[token_start:token_end],
                masks,
                margin=margin,
                config=config,
            )
            accumulator.add_(
                transform[:, coalition_start:coalition_end] @ scores
            )
        values[token_start:token_end] = accumulator.transpose(0, 1)

    values = values.reshape(batch, tokens, players)
    zeros = torch.zeros_like(values)
    return ShapleyEstimate(
        values=values,
        variance=zeros,
        ci_half_width=zeros,
        metadata={
            "name": "exact_affine_enumeration",
            "exact": True,
            "coalition_evaluations": int(coalition_count),
            "permutation_orders_used": 0,
            "independent_antithetic_pairs": 0,
            "token_chunk_size": int(config.token_chunk_size),
            "coalition_chunk_size": int(config.coalition_chunk_size),
            "require_convergence": bool(config.require_convergence),
            "converged": True,
            "converged_early": False,
            "stopping_reason": "exact_enumeration",
            "confidence_distribution": None,
        },
    )


def _estimate_antithetic_permutations(
    h_empty: torch.Tensor,
    h_full: torch.Tensor,
    unsafe_direction: torch.Tensor,
    safe_direction: torch.Tensor,
    *,
    margin: float,
    config: ShapleyConfig,
    deterministic_seed: int,
) -> ShapleyEstimate:
    batch, tokens, players = h_full.shape
    positions = batch * tokens
    empty = h_empty.float().reshape(positions, players)
    full = h_full.float().reshape(positions, players)
    unsafe = unsafe_direction.float().reshape(positions, players)
    safe = safe_direction.float().reshape(positions, players)
    mean = torch.zeros((positions, players), device=full.device, dtype=torch.float32)
    m2 = torch.zeros_like(mean)
    orders = _antithetic_orders(players, config.max_permutations, deterministic_seed)
    budget_stages = _permutation_budget_stages(
        config.min_permutations,
        config.max_permutations,
    )
    stage_set = set(budget_stages)
    convergence_checks: list[dict[str, float | int | bool]] = []
    independent_pairs = 0
    orders_used = 0
    converged = False
    final_variance = torch.zeros_like(mean)
    final_ci_half_width = torch.full_like(mean, float("inf"))

    for order_index in range(0, len(orders), 2):
        independent_pairs += 1
        orders_used += 2
        forward_order = orders[order_index].tolist()
        reverse_order = orders[order_index + 1].tolist()
        for token_start in range(0, positions, config.token_chunk_size):
            token_end = min(token_start + config.token_chunk_size, positions)
            pair_average = 0.5 * (
                _permutation_marginals(
                    empty[token_start:token_end],
                    full[token_start:token_end],
                    unsafe[token_start:token_end],
                    safe[token_start:token_end],
                    forward_order,
                    margin=margin,
                    config=config,
                )
                + _permutation_marginals(
                    empty[token_start:token_end],
                    full[token_start:token_end],
                    unsafe[token_start:token_end],
                    safe[token_start:token_end],
                    reverse_order,
                    margin=margin,
                    config=config,
                )
            )
            old_mean = mean[token_start:token_end]
            difference = pair_average - old_mean
            new_mean = old_mean + difference / float(independent_pairs)
            mean[token_start:token_end] = new_mean
            m2[token_start:token_end].add_(
                difference * (pair_average - new_mean)
            )

        # A forward/reverse pair is one independent Monte Carlo observation.
        # Its two orders are deliberately *not* treated as IID when estimating
        # variance or confidence intervals.
        if orders_used not in stage_set:
            continue
        degrees_of_freedom = independent_pairs - 1
        student_t_critical = _student_t_critical(config.confidence, degrees_of_freedom)
        final_variance = m2 / float(degrees_of_freedom)
        final_ci_half_width = student_t_critical * torch.sqrt(
            final_variance.clamp_min(0.0) / float(independent_pairs)
        )
        _require_finite_tensors(
            stage_mean=mean,
            stage_variance=final_variance,
            stage_ci_half_width=final_ci_half_width,
        )
        ci_metric = float(
            torch.quantile(final_ci_half_width.flatten(), config.ci_quantile).item()
        )
        mean_scale = float(torch.quantile(mean.abs().flatten(), config.ci_quantile).item())
        threshold = (
            float(config.absolute_ci or 0.0)
            + float(config.relative_ci or 0.0) * mean_scale
        )
        converged = ci_metric <= threshold
        convergence_checks.append(
            {
                "permutation_orders": int(orders_used),
                "independent_pairs": int(independent_pairs),
                "degrees_of_freedom": int(degrees_of_freedom),
                "student_t_critical": float(student_t_critical),
                "ci_quantile_half_width": ci_metric,
                "attribution_scale_quantile": mean_scale,
                "stopping_threshold": threshold,
                "converged": converged,
            }
        )
        if converged:
            break

    if not convergence_checks:
        raise RuntimeError(
            "Internal Shapley estimator error: no deterministic permutation budget stage "
            "was evaluated."
        )

    converged_early = converged and orders_used < config.max_permutations
    stopping_reason = (
        "student_t_ci_threshold_met" if converged else "maximum_permutation_budget_exhausted"
    )
    metadata: dict[str, Any] = {
        "name": "deterministic_staged_antithetic_permutation",
        "exact": False,
        "coalition_evaluations": int(orders_used * (players + 1)),
        "permutation_incremental_coordinate_updates": int(
            orders_used * positions * players
        ),
        "marginal_evaluation": "incremental_cosine_sufficient_statistics",
        "marginal_arithmetic_complexity": (
            "O(permutation_orders * positions * feature_players)"
        ),
        "marginal_state_complexity": "O(positions * feature_players)",
        "permutation_orders_used": int(orders_used),
        "permutation_orders_budget": int(config.max_permutations),
        "permutation_budget_stages": list(budget_stages),
        "completed_budget_stages": [
            int(check["permutation_orders"]) for check in convergence_checks
        ],
        "independent_antithetic_pairs": int(independent_pairs),
        "minimum_permutation_orders": int(config.min_permutations),
        "token_chunk_size": int(config.token_chunk_size),
        "coalition_chunk_size": None,
        "confidence": float(config.confidence),
        "confidence_distribution": "student_t_over_independent_antithetic_pair_averages",
        "relative_ci": config.relative_ci,
        "absolute_ci": config.absolute_ci,
        "ci_quantile": float(config.ci_quantile),
        "require_convergence": bool(config.require_convergence),
        "converged": converged,
        "converged_early": converged_early,
        "stopping_reason": stopping_reason,
        "convergence_checks": convergence_checks,
        "variance_unit": "independent_forward_reverse_pair_average",
    }
    if not converged and config.require_convergence:
        final_check = convergence_checks[-1]
        raise RuntimeError(
            "Shapley attribution did not converge before the deterministic maximum "
            f"budget of {config.max_permutations} permutation orders: Student-t CI "
            f"quantile half-width={float(final_check['ci_quantile_half_width']):.6g}, "
            f"threshold={float(final_check['stopping_threshold']):.6g}, "
            f"independent antithetic pairs={independent_pairs}. "
            "Increase steering.shapley.max_permutations or relax the registered CI "
            "criterion; require_convergence=true prevents decode/save."
        )

    return ShapleyEstimate(
        values=mean.reshape(batch, tokens, players),
        variance=final_variance.reshape(batch, tokens, players),
        ci_half_width=final_ci_half_width.reshape(batch, tokens, players),
        metadata=metadata,
    )


def _permutation_marginals(
    empty: torch.Tensor,
    full: torch.Tensor,
    unsafe: torch.Tensor,
    safe: torch.Tensor,
    order: list[int],
    *,
    margin: float,
    config: ShapleyConfig,
) -> torch.Tensor:
    """Return one permutation's marginals using exact running statistics.

    Rebuilding the complete hybrid and evaluating two length-``D`` cosine
    similarities after every one of the ``D`` coordinate insertions costs
    ``O(positions * D**2)`` per order.  A coordinate insertion only changes
    three sufficient statistics of this game: the hybrid squared norm and its
    dot products with the unsafe and safe directions.  Updating those scalars
    gives the same coalition path in ``O(positions * D)`` arithmetic while
    preserving the supplied order exactly.

    Inputs and coordinate products use float32, matching
    :func:`local_unwanted_score`; the three running scalar statistics accumulate
    in float64 and are projected back to float32 before the softplus score.  The
    squared norm is clamped before ``sqrt`` to guard against a tiny negative
    value from accumulated floating-point cancellation.
    """
    empty_f = empty.float()
    full_f = full.float()
    unsafe_f = unsafe.float()
    safe_f = safe.float()

    # Accumulate the three scalar statistics in float64.  Float32 subtractive
    # updates can leave a spurious residual when a hybrid genuinely approaches
    # zero; division by the cosine epsilon would then amplify that residue.
    # Products remain float32 (the game's scoring precision) and only their
    # reductions/updates are accumulated in float64.
    hybrid_norm_squared = empty_f.square().sum(dim=-1, dtype=torch.float64)
    unsafe_dot = (empty_f * unsafe_f).sum(dim=-1, dtype=torch.float64)
    safe_dot = (empty_f * safe_f).sum(dim=-1, dtype=torch.float64)
    full_norm_squared = full_f.square().sum(dim=-1, dtype=torch.float64)
    full_unsafe_dot = (full_f * unsafe_f).sum(dim=-1, dtype=torch.float64)
    full_safe_dot = (full_f * safe_f).sum(dim=-1, dtype=torch.float64)
    unsafe_norm = torch.linalg.vector_norm(unsafe_f, dim=-1)
    safe_norm = torch.linalg.vector_norm(safe_f, dim=-1)
    previous_score = _score_from_running_statistics(
        hybrid_norm_squared,
        unsafe_dot,
        safe_dot,
        unsafe_norm,
        safe_norm,
        margin=margin,
        tau=config.tau,
        eps=config.eps,
    )
    marginals = torch.zeros_like(empty)
    for order_index, player in enumerate(order):
        empty_player = empty_f[:, player]
        full_player = full_f[:, player]
        delta = full_player - empty_player
        hybrid_norm_squared = (
            hybrid_norm_squared
            + (full_player.square() - empty_player.square()).double()
        ).clamp_min(0.0)
        unsafe_dot = unsafe_dot + (delta * unsafe_f[:, player]).double()
        safe_dot = safe_dot + (delta * safe_f[:, player]).double()
        # The terminal coalition is known exactly.  Reusing its direct
        # reduction prevents even a tiny accumulated residual from perturbing
        # efficiency when the full hybrid has (near-)zero norm.
        if order_index == len(order) - 1:
            hybrid_norm_squared = full_norm_squared
            unsafe_dot = full_unsafe_dot
            safe_dot = full_safe_dot
        next_score = _score_from_running_statistics(
            hybrid_norm_squared,
            unsafe_dot,
            safe_dot,
            unsafe_norm,
            safe_norm,
            margin=margin,
            tau=config.tau,
            eps=config.eps,
        )
        marginals[:, player] = next_score - previous_score
        previous_score = next_score
    return marginals


def _score_from_running_statistics(
    hybrid_norm_squared: torch.Tensor,
    unsafe_dot: torch.Tensor,
    safe_dot: torch.Tensor,
    unsafe_norm: torch.Tensor,
    safe_norm: torch.Tensor,
    *,
    margin: float,
    tau: float,
    eps: float,
) -> torch.Tensor:
    """Evaluate the local game from cosine sufficient statistics."""
    hybrid_norm = torch.sqrt(hybrid_norm_squared.clamp_min(0.0))
    # The sufficient statistics are accumulated more accurately above, then
    # projected back to float32 before softplus to retain the scalar game's
    # established scoring precision.
    unsafe_cos = (
        unsafe_dot / (hybrid_norm * unsafe_norm).clamp_min(eps)
    ).float()
    safe_cos = (safe_dot / (hybrid_norm * safe_norm).clamp_min(eps)).float()
    return F.softplus((unsafe_cos - safe_cos + float(margin)) / float(tau))


def _scores_for_masks(
    empty: torch.Tensor,
    full: torch.Tensor,
    unsafe: torch.Tensor,
    safe: torch.Tensor,
    masks: torch.Tensor,
    *,
    margin: float,
    config: ShapleyConfig,
) -> torch.Tensor:
    hybrid = empty.unsqueeze(0) + masks.unsqueeze(1) * (
        full - empty
    ).unsqueeze(0)
    return local_unwanted_score(
        hybrid,
        unsafe.unsqueeze(0),
        safe.unsqueeze(0),
        margin=margin,
        tau=config.tau,
        eps=config.eps,
    )


@lru_cache(maxsize=32)
def _exact_shapley_transform(players: int) -> torch.Tensor:
    if players < 1:
        raise ValueError("A Shapley game must contain at least one player.")
    coalition_count = 2**players
    transform = torch.zeros((players, coalition_count), dtype=torch.float32)
    for coalition in range(coalition_count):
        size = coalition.bit_count()
        for player in range(players):
            contains = bool(coalition & (1 << player))
            subset_size = size - 1 if contains else size
            weight = 1.0 / (players * math.comb(players - 1, subset_size))
            transform[player, coalition] = weight if contains else -weight
    return transform


def _coalition_masks(
    start: int,
    end: int,
    players: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    coalition_ids = torch.arange(start, end, device=device, dtype=torch.long)
    bits = torch.arange(players, device=device, dtype=torch.long)
    return ((coalition_ids.unsqueeze(1) >> bits.unsqueeze(0)) & 1).float()


@lru_cache(maxsize=128)
def _cached_antithetic_orders(
    players: int,
    samples: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    orders: list[tuple[int, ...]] = []
    for _ in range(samples // 2):
        permutation = torch.randperm(players, generator=generator).tolist()
        orders.append(tuple(permutation))
        orders.append(tuple(reversed(permutation)))
    return tuple(orders)


def _antithetic_orders(players: int, samples: int, seed: int) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.tensor(order, dtype=torch.long)
        for order in _cached_antithetic_orders(players, samples, seed)
    )


def _permutation_budget_stages(minimum: int, maximum: int) -> tuple[int, ...]:
    """Return deterministic geometric CI checkpoints including the exact maximum."""
    if minimum < 8 or maximum < minimum or minimum % 2 or maximum % 2:
        raise ValueError(
            "Permutation stages require even minimum/maximum budgets with 8 <= minimum <= maximum."
        )
    stages = [minimum]
    while stages[-1] < maximum:
        stages.append(min(maximum, stages[-1] * 2))
    return tuple(stages)


@lru_cache(maxsize=256)
def _student_t_critical(confidence: float, degrees_of_freedom: int) -> float:
    """Return the two-sided Student-t critical value without a SciPy dependency."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("Student-t confidence must be in (0, 1).")
    if degrees_of_freedom < 1:
        raise ValueError("Student-t confidence intervals require at least two observations.")
    target_cdf = (1.0 + confidence) / 2.0
    lower = 0.0
    upper = 1.0
    while _student_t_cdf(upper, degrees_of_freedom) < target_cdf:
        upper *= 2.0
        if not math.isfinite(upper):
            raise RuntimeError("Unable to bracket the Student-t critical value.")
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        if _student_t_cdf(midpoint, degrees_of_freedom) < target_cdf:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def _student_t_cdf(value: float, degrees_of_freedom: int) -> float:
    if value == 0.0:
        return 0.5
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    tail = 0.5 * _regularized_incomplete_beta(
        0.5 * degrees_of_freedom,
        0.5,
        x,
    )
    return 1.0 - tail if value > 0 else tail


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Numerically evaluate the regularized incomplete beta function."""
    if a <= 0.0 or b <= 0.0:
        raise ValueError("Incomplete-beta parameters must be positive.")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        value = front * _beta_continued_fraction(a, b, x) / a
    else:
        value = 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b
    return min(1.0, max(0.0, value))


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    max_iterations = 256
    tolerance = 3.0e-14
    floor = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    result = d
    for iteration in range(1, max_iterations + 1):
        even = 2 * iteration
        numerator = iteration * (b - iteration) * x / ((qam + even) * (a + even))
        d = 1.0 + numerator * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + numerator / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        result *= d * c

        numerator = -(a + iteration) * (qab + iteration) * x / (
            (a + even) * (qap + even)
        )
        d = 1.0 + numerator * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + numerator / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= tolerance:
            return result
    raise RuntimeError("Incomplete-beta continued fraction did not converge.")


def _cosine_last_dim(left: torch.Tensor, right: torch.Tensor, *, eps: float) -> torch.Tensor:
    numerator = (left * right).sum(dim=-1)
    denominator = (
        torch.linalg.vector_norm(left, dim=-1)
        * torch.linalg.vector_norm(right, dim=-1)
    ).clamp_min(eps)
    return numerator / denominator


def _coerce_backtracking_scales(value: Any) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            "steering.shapley.backtracking_scales must be a nonempty numeric sequence."
        )
    return tuple(value)


def _validate_backtracking_scales(scales: tuple[float, ...]) -> tuple[float, ...]:
    if not isinstance(scales, tuple) or not scales:
        raise ValueError(
            "steering.shapley.backtracking_scales must be a nonempty tuple."
        )
    if any(isinstance(scale, bool) for scale in scales):
        raise ValueError("steering.shapley.backtracking_scales cannot contain booleans.")
    try:
        normalized = tuple(float(scale) for scale in scales)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "steering.shapley.backtracking_scales must contain finite numbers."
        ) from error
    if any(not math.isfinite(scale) for scale in normalized):
        raise ValueError("steering.shapley.backtracking_scales must be finite.")
    if any(not 0.0 < scale <= 1.0 for scale in normalized):
        raise ValueError(
            "steering.shapley.backtracking_scales must each be in (0, 1]."
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("steering.shapley.backtracking_scales must be unique.")
    if any(left <= right for left, right in zip(normalized, normalized[1:])):
        raise ValueError(
            "steering.shapley.backtracking_scales must be strictly decreasing."
        )
    return normalized


def _scale_label(scale: float) -> str:
    return format(float(scale), ".10g")


def _require_bitwise_supported_dtype(dtype: torch.dtype) -> None:
    if dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        raise ValueError(
            "Shapley bitwise preservation supports fp16, bf16, fp32, and fp64; "
            f"got {dtype}."
        )


def _bitwise_difference_mask(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("Bitwise Shapley comparisons require identical shape and dtype.")
    _require_bitwise_supported_dtype(left.dtype)
    integer_dtype = {
        torch.float16: torch.int16,
        torch.bfloat16: torch.int16,
        torch.float32: torch.int32,
        torch.float64: torch.int64,
    }[left.dtype]
    return left.contiguous().view(integer_dtype) != right.contiguous().view(integer_dtype)


def _bitwise_difference_count(left: torch.Tensor, right: torch.Tensor) -> int:
    return int(_bitwise_difference_mask(left, right).sum().item())


def _actual_delta_float32(candidate: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    # For fp16/bf16/fp32, conversion to float32 is exact before subtraction.
    # Float64 subtraction is performed in float64 first so low-bit changes are
    # not erased merely by converting both endpoints to float32.
    if candidate.dtype == torch.float64:
        return (candidate - current).float()
    return candidate.float() - current.float()


def _require_canonical_shapes(*tensors: torch.Tensor) -> None:
    if not tensors:
        return
    expected = tensors[0].shape
    if len(expected) != 3:
        raise ValueError(f"Canonical vector fields must be [B,T,D], got {tuple(expected)}.")
    if any(int(size) < 1 for size in expected):
        raise ValueError(f"Canonical vector fields cannot have empty axes, got {tuple(expected)}.")
    for tensor in tensors[1:]:
        if tensor.shape != expected:
            raise ValueError(
                f"Canonical vector-field shape mismatch: expected {tuple(expected)}, "
                f"got {tuple(tensor.shape)}."
            )


def _materialize_canonical(
    layout: LatentLayout,
    tensor: torch.Tensor,
    *,
    label: str,
) -> torch.Tensor:
    canonical = layout.canonicalize(tensor).contiguous()
    _require_contiguous(canonical, label=f"canonical {label}")
    _require_finite_tensors(**{f"canonical_{label}": canonical})
    return canonical


def _materialize_restored(
    layout: LatentLayout,
    canonical: torch.Tensor,
    *,
    dtype: torch.dtype,
    label: str,
) -> torch.Tensor:
    restored = layout.restore(canonical).to(dtype=dtype).contiguous()
    _require_contiguous(restored, label=f"restored {label}")
    _require_finite_tensors(**{f"restored_{label}": restored})
    return restored


def _require_contiguous(tensor: torch.Tensor, *, label: str) -> None:
    if not tensor.is_contiguous():
        raise RuntimeError(f"Shapley {label} must be materialized as a contiguous tensor.")


def _require_finite_tensors(**tensors: torch.Tensor) -> None:
    failures: list[str] = []
    for label, tensor in tensors.items():
        if not (tensor.is_floating_point() or tensor.is_complex()):
            continue
        finite = torch.isfinite(tensor)
        if not bool(finite.all().item()):
            failures.append(f"{label} ({int((~finite).sum().item())} non-finite values)")
    if failures:
        raise RuntimeError("Non-finite Shapley tensors detected: " + ", ".join(failures) + ".")


def _require_nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"Shapley {label} must be a non-negative integer, got {value!r}.")
    return value


def _require_nonnegative_finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(
            f"Shapley {label} must be a finite non-negative number, got {value!r}."
        )
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise RuntimeError(
            f"Shapley {label} must be a finite non-negative number, got {value!r}."
        )
    return normalized


def _stable_seed(base_seed: int, *parts: Any) -> int:
    payload = ":".join([str(base_seed), *(str(part) for part in parts)]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
