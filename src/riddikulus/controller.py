"""Confidence-preserving local steering with explicit numerical contracts.

All detector, aggregation and budget arithmetic is FP32. Default thresholds are
declared experimental settings, not estimates calibrated from this one prompt.
The legacy flags exist solely for the registered cumulative component controls.
This controller accepts one generated sample at a time (including video frames).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch

from .diagnostics import DIAGNOSTIC_SCHEMA


@dataclass(frozen=True)
class ControllerConfig:
    center_residuals: bool = True
    threshold_gates: bool = True
    confidence_weighting: bool = True
    group_redundancy: bool = False
    conflict_handling: bool = True
    norm_budgets: bool = True
    lambda_max: float = 1.0
    unsafe_abs_threshold: float = 0.25
    unsafe_gap_threshold: float = 0.10
    unsafe_abs_ramp: float = 0.25
    unsafe_gap_ramp: float = 0.25
    residual_rms_floor: float = 1.0e-6
    residual_relative_floor: float = 1.0e-3
    minimum_support_fraction: float = 0.0
    minimum_support_positions: int = 1
    allow_suppression_only: bool = False
    legacy_mask_threshold: float = 0.05
    conflict_mu: float = 1.0
    conflict_rtol: float = 1.0e-5
    conflict_atol: float = 1.0e-8
    conflict_chunk_positions: int = 128
    global_relative_rms: float = 0.60
    prompt_residual_relative_rms: float | None = None
    local_relative_rms: float = 0.85
    cumulative_exposure_limit: float | None = None
    budget_rtol: float = 2.0e-6
    budget_atol: float = 1.0e-9
    quantization_max_iterations: int = 32
    aggregation_mode: str = "additive_leaf_v1"
    quantization_backtracking: str = "ratio_margin"

    def __post_init__(self) -> None:
        if self.aggregation_mode not in {"flat", "hierarchy_balanced", "additive_leaf_v1"}:
            raise ValueError(f"Unknown aggregation mode: {self.aggregation_mode}")
        if self.quantization_backtracking not in {"legacy_half", "ratio_margin"}:
            raise ValueError(f"Unknown quantization backtracking: {self.quantization_backtracking}")
        positive = ("unsafe_abs_ramp", "unsafe_gap_ramp", "residual_rms_floor",
                    "conflict_rtol", "conflict_atol")
        nonnegative = ("lambda_max", "residual_relative_floor", "conflict_mu",
                       "global_relative_rms", "local_relative_rms",
                       "budget_rtol", "budget_atol")
        for name in positive + nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0 or (name in positive and value == 0):
                raise ValueError(f"Invalid {name}: {value}")
        for name in ("unsafe_abs_threshold", "unsafe_gap_threshold", "legacy_mask_threshold"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"Nonfinite {name}")
        if not 0 <= self.minimum_support_fraction <= 1:
            raise ValueError("minimum_support_fraction must lie in [0, 1]")
        for name in ("minimum_support_positions", "conflict_chunk_positions", "quantization_max_iterations"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.prompt_residual_relative_rms is not None and (
            not math.isfinite(self.prompt_residual_relative_rms)
            or self.prompt_residual_relative_rms < 0
        ):
            raise ValueError("prompt_residual_relative_rms must be nonnegative or None")
        if self.cumulative_exposure_limit is not None and (
            not math.isfinite(self.cumulative_exposure_limit)
            or self.cumulative_exposure_limit < 0
        ):
            raise ValueError("cumulative_exposure_limit must be nonnegative or None")


@dataclass(frozen=True)
class Evidence:
    confidence: torch.Tensor
    active: bool
    diagnostics: dict[str, Any]
    joint: dict[str, torch.Tensor] | None = None


@dataclass(frozen=True)
class ConceptTerm:
    concept_id: str
    alias_key: str
    direction: torch.Tensor
    confidence: torch.Tensor
    # Ordered ancestors from the root's immediate child through the leaf parent.
    # Empty means a direct root term, including aliases spanning root branches.
    hierarchy_path: tuple[str, ...] = ()


@dataclass
class ExposureState:
    cumulative_exposure: float = 0.0
    steps: int = 0


@dataclass(frozen=True)
class AppliedCorrection:
    prediction: torch.Tensor
    delta_fp32: torch.Tensor
    diagnostics: dict[str, Any]


def _rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().square().mean().sqrt()


def _finite(value: torch.Tensor, label: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"Nonfinite {label}")


def _feature_dim(value: torch.Tensor, dim: int) -> int:
    dim %= value.ndim
    if value.ndim < 2 or value.shape[0] != 1 or dim == 0:
        raise ValueError("ECCV controller requires batch size one and a nonbatch feature axis")
    return dim


def _cosine(a: torch.Tensor, b: torch.Tensor, dim: int) -> torch.Tensor:
    # Computing norms separately avoids underflow in their product.
    an = a.square().sum(dim=dim, keepdim=True).sqrt()
    bn = b.square().sum(dim=dim, keepdim=True).sqrt()
    return ((a / an.clamp_min(1e-20)) * (b / bn.clamp_min(1e-20))).sum(
        dim=dim, keepdim=True
    ).clamp(-1, 1)


def _reliable(value: torch.Tensor, dim: int, config: ControllerConfig) -> torch.Tensor:
    local = value.square().mean(dim=dim, keepdim=True).sqrt()
    # Reference is a residual, so adding a common neutral field does not change it.
    floor = torch.maximum(
        local.new_tensor(config.residual_rms_floor),
        _rms(value) * config.residual_relative_floor,
    )
    return local > floor


def concept_multiplier(
    v_neutral: torch.Tensor, v_unsafe: torch.Tensor, v_safe: torch.Tensor,
    *, config: ControllerConfig,
) -> tuple[float, dict[str, Any]]:
    """One global centered cosine distance; unavailable geometry contributes zero.

    This is geometric reweighting, not calibrated harm confidence. It multiplies
    the raw safe-minus-unsafe direction, never the local confidence. Flattening
    includes every feature and spatial position of this single sample.
    """
    if v_unsafe.shape != v_neutral.shape or v_safe.shape != v_neutral.shape:
        raise ValueError("Concept multiplier prediction shapes differ")
    if v_neutral.ndim < 2 or v_neutral.shape[0] != 1:
        raise ValueError("Concept multiplier requires batch size one")
    for value in (v_neutral, v_unsafe, v_safe):
        _finite(value, "concept multiplier input")
    u = v_unsafe.float() - v_neutral.float()
    s = v_safe.float() - v_neutral.float()
    ru, rs = float(_rms(u)), float(_rms(s))
    reliable = ru > config.residual_rms_floor and rs > config.residual_rms_floor
    cosine = None
    multiplier = 0.0
    if reliable:
        uf, sf = u.reshape(-1), s.reshape(-1)
        cosine_tensor = ((uf / uf.square().sum().sqrt().clamp_min(1e-20)) *
                         (sf / sf.square().sum().sqrt().clamp_min(1e-20))).sum().clamp(-1, 1)
        cosine = float(cosine_tensor)
        multiplier = float((1.0 - cosine_tensor).clamp(0, 2))
    return multiplier, {
        "definition": "1-cosine(flatten(unsafe-neutral),flatten(safe-neutral))",
        "range": [0.0, 2.0], "arithmetic": "float32",
        "unsafe_residual_rms": ru, "safe_residual_rms": rs,
        "global_residual_rms_floor": config.residual_rms_floor,
        "geometry_reliable": reliable, "centered_cosine": cosine,
        "multiplier": multiplier,
        "zero_multiplier": multiplier == 0.0,
        "unavailable_reason": None if reliable else "global_residual_below_absolute_reliability_floor",
        "confidence_scaled": False,
    }


def sigmoid_concept_multiplier(
    v_neutral: torch.Tensor, v_unsafe: torch.Tensor, v_safe: torch.Tensor,
    *, config: ControllerConfig, temperature: float,
) -> tuple[float, dict[str, Any]]:
    """Sigmoid of one current global cosine distance, replacing the legacy weight."""
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Sigmoid temperature must be finite and positive")
    distance, geometry = concept_multiplier(v_neutral, v_unsafe, v_safe, config=config)
    reliable = geometry["geometry_reliable"]
    weight = 1.0 / (1.0 + math.exp(-distance / temperature)) if reliable else 0.0
    return weight, {
        **geometry, "definition": "sigmoid(global_centered_cosine_distance/temperature)",
        "distance_definition": geometry["definition"], "cosine_distance": distance if reliable else None,
        "sigmoid_temperature": float(temperature), "sigmoid_weight": weight,
        "multiplier": weight, "zero_multiplier": weight == 0.0, "range": [0.0, 1.0],
        "distance_multiplied_twice": False, "arithmetic": "float32_distance_float64_scalar_sigmoid",
    }


def _distribution(value: torch.Tensor, edges: Sequence[float]) -> dict[str, Any]:
    """Read-only distributions; never used by the scientific controller."""
    values = value.detach().float().reshape(-1).cpu()
    probabilities = DIAGNOSTIC_SCHEMA["quantile_probabilities"]
    if values.numel():
        quantiles = torch.quantile(values, values.new_tensor(probabilities)).tolist()
        counts = torch.histogram(values, bins=values.new_tensor(edges)).hist.to(torch.int64).tolist()
    else:
        quantiles = [None] * len(probabilities)
        counts = [0] * (len(edges) - 1)
    return {
        "count": values.numel(),
        "quantiles": {str(p): result for p, result in zip(probabilities, quantiles)},
        "histogram": {"edges": list(edges), "counts": counts},
    }


def _evidence_distribution_diagnostics(
    *, cu: torch.Tensor, cs: torch.Tensor, gap: torch.Tensor,
    confidence: torch.Tensor, eligible: torch.Tensor, reliable: torch.Tensor,
    reliable_r: torch.Tensor, reliable_u: torch.Tensor, reliable_s: torch.Tensor,
    suppression: torch.Tensor, residuals: Mapping[str, torch.Tensor],
    feature_dim: int, support: int, config: ControllerConfig,
) -> dict[str, Any]:
    observations = torch.stack([
        value.detach().float().reshape(-1)
        for value in (cu, cs, gap, confidence, eligible, reliable,
                      reliable_r, reliable_u, reliable_s, suppression)
    ]).cpu()
    unsafe_cosine, safe_cosine, cosine_gap, conf = observations[:4]
    eligible_cpu, reliable_cpu, rr, ru, rs, suppression_cpu = [
        value.bool() for value in observations[4:]
    ]
    count = conf.numel()
    active = conf > 0
    edges = DIAGNOSTIC_SCHEMA["histogram_edges"]
    hypothetical = []
    for margin in DIAGNOSTIC_SCHEMA["hypothetical_support"]["unsafe_gap_thresholds"]:
        supported = reliable_cpu & (unsafe_cosine > config.unsafe_abs_threshold) & (cosine_gap > margin)
        if config.allow_suppression_only:
            supported = torch.where(
                suppression_cpu, reliable_cpu & (unsafe_cosine > config.unsafe_abs_threshold), supported,
            )
        positions = int(supported.sum())
        for fraction in DIAGNOSTIC_SCHEMA["hypothetical_support"]["minimum_support_fractions"]:
            required = max(config.minimum_support_positions, math.ceil(fraction * count))
            hypothetical.append({
                "unsafe_abs_threshold": config.unsafe_abs_threshold,
                "unsafe_gap_threshold": margin, "minimum_support_fraction": fraction,
                "minimum_support_positions": config.minimum_support_positions,
                "support_positions": positions, "position_count": count,
                "support_fraction": positions / count,
                "required_support_positions": required, "support_passed": positions >= required,
            })
    zero_residuals = {}
    for name, residual in residuals.items():
        zero = (residual.detach() == 0).all(dim=feature_dim, keepdim=True)
        positions = int(zero.sum())
        zero_residuals[name] = {
            "entire_field_zero": positions == zero.numel(),
            "exact_zero_positions": positions, "position_count": zero.numel(),
            "exact_zero_fraction": positions / zero.numel(),
        }
    return {
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA["version"],
        "position_count": count, "support_fraction": support / count,
        "eligible_positions_after_support_gate": int(eligible_cpu.sum()),
        "active_positions": int(active.sum()), "active_position_fraction": float(active.float().mean()),
        "confidence_mass": float(conf.sum()),
        "confidence_mass_active_positions": float(conf[active].sum()),
        "confidence_mean_active_positions": float(conf[active].mean()) if bool(active.any()) else None,
        "confidence_mean_eligible_positions": float(conf[eligible_cpu].mean()) if bool(eligible_cpu.any()) else None,
        "unsafe_cosine_sum": float(unsafe_cosine.sum()),
        "safe_cosine_sum": float(safe_cosine.sum()),
        "cosine_gap_sum": float(cosine_gap.sum()),
        "distributions": {
            "unsafe_cosine": _distribution(unsafe_cosine, edges["cosine"]),
            "safe_cosine": _distribution(safe_cosine, edges["cosine"]),
            "cosine_gap": _distribution(cosine_gap, edges["cosine_gap"]),
            "unsafe_cosine_reliable_positions": _distribution(unsafe_cosine[rr & ru], edges["cosine"]),
            "safe_cosine_reliable_positions": _distribution(safe_cosine[rr & rs], edges["cosine"]),
            "cosine_gap_effectively_reliable_positions": _distribution(cosine_gap[reliable_cpu], edges["cosine_gap"]),
            "eligible_confidence_all_positions": _distribution(conf, edges["confidence"]),
            "eligible_confidence_eligible_positions": _distribution(conf[eligible_cpu], edges["confidence"]),
            "eligible_confidence_active_positions": _distribution(conf[active], edges["confidence"]),
        },
        "reliability": {
            name: {
                "positions": int(mask.sum()), "fraction": float(mask.float().mean()),
                "distribution": _distribution(mask, edges["reliability"]),
            }
            for name, mask in (
                ("reference_residual", rr), ("unsafe_residual", ru),
                ("safe_residual", rs), ("effective", reliable_cpu),
            )
        },
        "zero_residuals": zero_residuals,
        "hypothetical_support_scope": DIAGNOSTIC_SCHEMA["hypothetical_support"],
        "actual_threshold_gates_enabled": config.threshold_gates,
        "hypothetical_support": hypothetical,
    }


def _cancellation_diagnostics(
    terms: Sequence[ConceptTerm], *, reference: torch.Tensor, feature_dim: int,
) -> dict[str, Any]:
    """Triangle-inequality shortfalls before controller scaling, read-only."""
    total = torch.zeros_like(reference, dtype=torch.float32)
    local_shape = list(reference.shape)
    local_shape[feature_dim] = 1
    local_norm_sum = torch.zeros(local_shape, device=reference.device, dtype=torch.float32)
    term_rms_sum = 0.0
    confidence_mass = 0.0
    for term in terms:
        weighted = term.confidence.detach().float() * term.direction.detach().float()
        total = total + weighted
        term_rms_sum += float(_rms(weighted))
        confidence_mass += float(term.confidence.detach().float().sum())
        local_norm_sum = local_norm_sum + weighted.square().sum(dim=feature_dim, keepdim=True).sqrt()
    sum_rms = float(_rms(total))
    local_sum_norm = total.square().sum(dim=feature_dim, keepdim=True).sqrt()
    return {
        "term_count": len(terms), "confidence_mass": confidence_mass,
        "sum_term_rms": term_rms_sum, "summed_field_rms": sum_rms,
        "cancellation_magnitude_rms": max(0.0, term_rms_sum - sum_rms),
        "cancellation_fraction": max(0.0, min(1.0, 1.0 - sum_rms / term_rms_sum)) if term_rms_sum > 0 else 0.0,
        "mean_sum_term_local_l2": float(local_norm_sum.mean()),
        "mean_summed_field_local_l2": float(local_sum_norm.mean()),
        "mean_local_cancellation_l2": float((local_norm_sum - local_sum_norm).clamp_min(0).mean()),
        "scope": "before_overlap_normalization_lambda_and_conflict_solve",
    }


@torch.no_grad()
def _field_alignment_diagnostics(
    current: torch.Tensor, reference: torch.Tensor,
) -> dict[str, Any]:
    """Read-only global geometry, with FP64 reductions of the supplied fields.

    Zero fields have no observed direction. These measurements never supply a
    replacement direction, confidence, multiplier, or controller decision.
    """
    c, r = current.detach().double(), reference.detach().double()
    c2, r2, cross = torch.stack([
        c.square().mean(), r.square().mean(), (c * r).mean(),
    ]).cpu().tolist()
    finite = all(math.isfinite(value) for value in (c2, r2, cross))
    observed = finite and c2 > 0 and r2 > 0
    reason = None
    if not finite:
        reason = "nonfinite_diagnostic_moment"
    elif c2 == 0 or r2 == 0:
        reason = "zero_current_and_reference" if c2 == r2 == 0 else (
            "zero_current" if c2 == 0 else "zero_reference"
        )
    current_rms = math.sqrt(c2) if finite else None
    reference_rms = math.sqrt(r2) if finite else None
    return {
        "status": "observed" if observed else "unobserved",
        "unobserved_reason": reason,
        "current_rms": current_rms, "reference_rms": reference_rms,
        "mean_inner_product": cross if finite else None,
        "cosine": max(-1.0, min(1.0, cross / current_rms / reference_rms)) if observed else None,
        "current_over_reference_norm_ratio": current_rms / reference_rms if observed else None,
        "current_onto_reference_projection_coefficient": cross / r2 if observed else None,
        "arithmetic": "float64_reductions_of_detached_fields",
        "scope": "global_flattened_field; no_semantic_safety_interpretation",
    }


@torch.no_grad()
def _root_aggregation_diagnostics(
    terms: Sequence[ConceptTerm], *, total_confidence: torch.Tensor,
    reference: torch.Tensor, feature_dim: int, normalize_overlap: bool,
) -> dict[str, Any]:
    """Observe the existing root decomposition without feeding steering."""
    denominator = (total_confidence.detach().float().clamp_min(1.0)
                   if normalize_overlap else torch.ones_like(total_confidence))
    before = [term.confidence.detach().float() * term.direction.detach().float() for term in terms]
    after = [field / denominator for field in before]
    total = torch.zeros_like(reference, dtype=torch.float32)
    for field in before:
        total = total + field
    totals = (total, total / denominator)
    branch_records = {}
    for term, unnormalized, normalized in zip(terms, before, after):
        confidence = term.confidence.detach().double()
        original_mass = float(confidence.sum())
        normalized_mass = float((confidence / denominator.double()).sum())
        branch_records[term.concept_id] = {
            "coefficient_mass_before_root_normalization": original_mass,
            "coefficient_mass_after_root_normalization": normalized_mass,
            "coefficient_mass_attenuation_fraction": (
                1.0 - normalized_mass / original_mass if original_mass > 0 else None
            ),
            "coefficient_status": "observed" if original_mass > 0 else "unobserved_zero_confidence",
            "weighted_field_rms_before_root_normalization": float(unnormalized.double().square().mean().sqrt()),
            "weighted_field_rms_after_root_normalization": float(normalized.double().square().mean().sqrt()),
        }
    triangle = {}
    for name, fields, summed in zip(("before_root_normalization", "after_root_normalization"),
                                     (before, after), totals):
        norm_sum = sum(float(field.double().square().mean().sqrt()) for field in fields)
        summed_rms = float(summed.double().square().mean().sqrt())
        triangle[name] = {
            "sum_branch_rms": norm_sum, "summed_field_rms": summed_rms,
            "triangle_shortfall_fraction": max(0.0, min(1.0, 1.0 - summed_rms / norm_sum)) if norm_sum > 0 else None,
            "status": "observed" if norm_sum > 0 else "unobserved_zero_fields",
        }
    pairs = []
    for index, left in enumerate(terms):
        for right_index in range(index + 1, len(terms)):
            right = terms[right_index]
            pair = {"reference_term_id": left.concept_id, "current_term_id": right.concept_id}
            for name, fields in (("before_root_normalization", before), ("after_root_normalization", after)):
                pair[name] = _field_alignment_diagnostics(fields[right_index], fields[index])
            left_field, right_field = after[index].double(), after[right_index].double()
            jointly_nonzero = (left_field.square().sum(dim=feature_dim) > 0) & (
                right_field.square().sum(dim=feature_dim) > 0
            )
            negative_dot = ((left_field * right_field).sum(dim=feature_dim) < 0) & jointly_nonzero
            observed_positions = int(jointly_nonzero.sum())
            negative_positions = int(negative_dot.sum())
            pair["after_root_normalization_local_opposition"] = {
                "jointly_nonzero_positions": observed_positions,
                "negative_dot_positions": negative_positions,
                "negative_dot_fraction_of_jointly_nonzero_positions": (
                    negative_positions / observed_positions if observed_positions else None
                ),
            }
            pairs.append(pair)
    return {
        "schema": "root_aggregation_geometry_v1",
        "scope": "after_hierarchy_before_lambda_conflict_and_budgets",
        "normalization_enabled": normalize_overlap,
        "coefficient_definition": "a_b/max(1,sum_b(a_b)); a_b_when_normalization_disabled",
        "term_count": len(terms), "terms": branch_records, "pairs": pairs,
        "triangle_shortfall": triangle,
        "interpretation": "coefficient_attenuation_is_not_causal_safety_loss; triangle_shortfall_includes_orthogonality; projection_is_signed_not_a_fraction_removed",
        "feeds_steering": False,
    }


def concept_evidence(
    v_base: torch.Tensor,
    v_neutral: torch.Tensor,
    v_unsafe: torch.Tensor,
    v_safe: torch.Tensor,
    *,
    feature_dim: int,
    config: ControllerConfig,
    selection_gap_threshold: float | None = None,
    selection_abs_threshold: float = 0.25,
    capture_joint: bool = False,
) -> Evidence:
    """Evaluate absolute and contrastive alignment in a common neutral frame.

    A zero safe residual is unavailable by default. The optional suppression-only
    path deliberately uses only the absolute unsafe test and reports its use; it
    never presents an undefined safe cosine as contrastive evidence.
    """
    dim = _feature_dim(v_base, feature_dim)
    values = (v_base, v_neutral, v_unsafe, v_safe)
    if any(v.shape != v_base.shape for v in values):
        raise ValueError("Prediction shapes differ")
    for v in values:
        _finite(v, "concept prediction")
    base, neutral, unsafe, safe = (v.float() for v in values)
    r = base - neutral if config.center_residuals else base
    u, s = unsafe - neutral, safe - neutral
    cu, cs = _cosine(r, u, dim), _cosine(r, s, dim)
    gap = cu - cs
    reliable_r, reliable_u, reliable_s = (_reliable(v, dim, config) for v in (r, u, s))
    selection = None
    selection_counterfactuals = None
    if selection_gap_threshold is not None:
        if not config.center_residuals:
            raise ValueError("Selective acquisition requires neutral-centered residuals")
        selection = selection_support(
            cu, cs, reliable_r & reliable_u & reliable_s,
            absolute_threshold=selection_abs_threshold,
            gap_threshold=selection_gap_threshold,
        )
        selection_counterfactuals = [
            selection if threshold == selection_gap_threshold else selection_support(
                cu, cs, reliable_r & reliable_u & reliable_s,
                absolute_threshold=selection_abs_threshold, gap_threshold=threshold,
            ) for threshold in (0.50, 0.60, 0.70)
        ]
    suppression = (~reliable_s) & (s.square().sum(dim=dim, keepdim=True) == 0)
    reliable = reliable_r & reliable_u & reliable_s
    if config.allow_suppression_only:
        reliable = reliable | (reliable_r & reliable_u & suppression)
    if not config.center_residuals:
        # Preserve the historical unreliable-cosine behavior in the legacy arms.
        reliable = torch.ones_like(reliable)
    if config.threshold_gates:
        eligible = reliable & (cu > config.unsafe_abs_threshold) & (
            gap > config.unsafe_gap_threshold
        )
        if config.allow_suppression_only:
            eligible = torch.where(suppression, reliable & (cu > config.unsafe_abs_threshold), eligible)
        support = int(eligible.sum().item())
        required = max(config.minimum_support_positions,
                       math.ceil(config.minimum_support_fraction * eligible.numel()))
        support_ok = support >= required
        if not support_ok:
            eligible = torch.zeros_like(eligible)
    else:
        eligible = reliable & (gap > 0)
        support, required = int(eligible.sum().item()), 1
        support_ok = support > 0
    if config.confidence_weighting:
        absolute = ((cu - config.unsafe_abs_threshold) / config.unsafe_abs_ramp).clamp(0, 1)
        contrastive = ((gap - config.unsafe_gap_threshold) / config.unsafe_gap_ramp).clamp(0, 1)
        if config.allow_suppression_only:
            contrastive = torch.where(suppression, torch.ones_like(contrastive), contrastive)
        confidence = absolute * contrastive
    else:
        # Only cumulative legacy controls retain peak normalization.
        activation = torch.where(eligible, gap.relu(), torch.zeros_like(gap))
        confidence = activation / activation.amax().clamp_min(1e-6)
        confidence = torch.where(confidence >= config.legacy_mask_threshold,
                                 confidence, torch.zeros_like(confidence))
    confidence = torch.where(eligible, confidence, torch.zeros_like(confidence))
    _finite(confidence, "confidence")
    return Evidence(
        confidence=confidence,
        active=bool((confidence > 0).any()),
        joint=({"unsafe_cosine": cu.detach(), "cosine_gap": gap.detach(),
                "joint_reliability": (reliable_r & reliable_u & reliable_s).detach()}
               if capture_joint else None),
        diagnostics={
            **({"one_time_selection": selection,
                "one_time_selection_counterfactuals": selection_counterfactuals}
               if selection is not None else {}),
            "unsafe_cosine_mean": float(cu.mean()), "unsafe_cosine_max": float(cu.max()),
            "safe_cosine_mean": float(cs.mean()), "cosine_gap_mean": float(gap.mean()),
            "cosine_gap_max": float(gap.max()), "reliable_fraction": float(reliable.float().mean()),
            "support_positions": support, "required_support_positions": required,
            "support_passed": support_ok, "confidence_mean": float(confidence.mean()),
            "confidence_max": float(confidence.max()),
            "suppression_only_positions": int((suppression & eligible).sum()) if config.allow_suppression_only else 0,
            "undefined_safe_positions": int((~reliable_s).sum()),
            **_evidence_distribution_diagnostics(
                cu=cu, cs=cs, gap=gap, confidence=confidence, eligible=eligible,
                reliable=reliable, reliable_r=reliable_r, reliable_u=reliable_u,
                reliable_s=reliable_s, suppression=suppression,
                residuals={
                    "detector_reference": r, "prompt_minus_neutral": base - neutral,
                    "unsafe_minus_neutral": u, "safe_minus_neutral": s,
                    "safe_minus_unsafe": safe - unsafe,
                },
                feature_dim=dim, support=support, config=config,
            ),
        },
    )


def selection_support(
    unsafe_cosine: torch.Tensor, safe_cosine: torch.Tensor,
    jointly_reliable: torch.Tensor, *, absolute_threshold: float,
    gap_threshold: float,
) -> dict[str, Any]:
    """Select a leaf from one jointly qualifying position, independently of its mask.

    Both strict cosine comparisons and reliability must hold at the SAME position.
    This one-time inventory decision has no area gate or confidence-ramp test.
    """
    if not math.isfinite(absolute_threshold) or not -1 <= absolute_threshold <= 1:
        raise ValueError("Invalid selection absolute cosine threshold")
    if not math.isfinite(gap_threshold) or not 0 <= gap_threshold < 2:
        raise ValueError("Invalid selection cosine-gap threshold")
    if unsafe_cosine.shape != safe_cosine.shape or unsafe_cosine.shape != jointly_reliable.shape:
        raise ValueError("Selection evidence shapes differ")
    if jointly_reliable.dtype != torch.bool:
        raise ValueError("Selection reliability must be boolean")
    _finite(unsafe_cosine, "selection unsafe cosine")
    _finite(safe_cosine, "selection safe cosine")
    selected_positions = jointly_reliable & (unsafe_cosine > absolute_threshold) & (
        unsafe_cosine - safe_cosine > gap_threshold
    )
    count = int(selected_positions.sum())
    return {
        "rule": "any_reliable_position", "comparisons": "strict_greater_than",
        "absolute_threshold": float(absolute_threshold), "gap_threshold": float(gap_threshold),
        "support_positions": count, "selected": count > 0,
        "position_count": selected_positions.numel(), "area_gate_used": False,
        "independent_of_pointwise_confidence_threshold": True,
    }


def _group_terms(terms: Sequence[ConceptTerm], config: ControllerConfig) -> list[ConceptTerm]:
    ordered = sorted(terms, key=lambda t: (t.alias_key, t.concept_id))
    if not config.group_redundancy:
        return ordered
    groups: dict[str, list[ConceptTerm]] = {}
    for term in ordered:
        groups.setdefault(term.alias_key, []).append(term)
    result = []
    for alias, members in sorted(groups.items()):
        # Exact duplicate vectors/evidence are one observation, including when
        # multiple IDs expose the same text pair. Deduplicate before alias means.
        distinct: list[ConceptTerm] = []
        for member in members:
            if not any(torch.equal(member.direction, old.direction)
                       and torch.equal(member.confidence, old.confidence) for old in distinct):
                distinct.append(member)
        total = torch.zeros_like(distinct[0].confidence, dtype=torch.float32)
        confidence = torch.zeros_like(total)
        numerator = torch.zeros_like(distinct[0].direction, dtype=torch.float32)
        for member in distinct:
            a, d = member.confidence.float(), member.direction.float()
            total = total + a
            confidence = torch.maximum(confidence, a)
            numerator = numerator + a * d
        direction = numerator / total.clamp_min(1e-20)
        path = members[0].hierarchy_path
        for member in members[1:]:
            common = 0
            for left, right in zip(path, member.hierarchy_path):
                if left != right:
                    break
                common += 1
            path = path[:common]
        result.append(ConceptTerm(alias, alias, direction, confidence, path))
    return result


def _hierarchy_balanced_terms(groups: Sequence[ConceptTerm]) -> list[ConceptTerm]:
    """Summarize sibling evidence without introducing a leaf-count prior.

    At each non-root node, a=max(child a) and
    d=sum(child a * child d)/sum(child a), with d=0 on unsupported positions.
    Thus a never exceeds the strongest child confidence, a single weak leaf
    stays weak at any depth, and duplicate identical sibling observations do
    not amplify their common correction. Root combination retains the usual
    denominator max(1, sum a). Every active alias group participates exactly
    once; groups crossing hierarchy paths live at their common ancestor.

    This is an experimental aggregation rule, not calibrated harm probability.
    Ancestor concept predictions remain diagnostics and do not gate descendants.
    """
    def collect(members: Sequence[ConceptTerm], depth: int) -> list[ConceptTerm]:
        direct: list[ConceptTerm] = []
        children: dict[str, list[ConceptTerm]] = {}
        for member in members:
            if len(member.hierarchy_path) == depth:
                direct.append(member)
            else:
                children.setdefault(member.hierarchy_path[depth], []).append(member)
        for node_id, descendants in sorted(children.items()):
            child_terms = collect(descendants, depth + 1)
            confidence = torch.zeros_like(child_terms[0].confidence, dtype=torch.float32)
            total = torch.zeros_like(confidence)
            numerator = torch.zeros_like(child_terms[0].direction, dtype=torch.float32)
            for child in child_terms:
                a = child.confidence.float()
                confidence = torch.maximum(confidence, a)
                total = total + a
                numerator = numerator + a * child.direction.float()
            direction = numerator / total.clamp_min(1e-20)
            direct.append(ConceptTerm(
                "hierarchy:" + node_id, "hierarchy:" + node_id,
                direction, confidence, descendants[0].hierarchy_path[:depth],
            ))
        return direct

    return collect(groups, 0)


def _conflict_prox(
    raw: torch.Tensor, groups: Sequence[ConceptTerm], feature_dim: int, config: ControllerConfig,
    *, normalize_confidence: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Unique minimizer of .5||z-raw||² + mu Σ a_g relu(-z·unit(d_g))².

    Confidence coefficients are divided by max(1, sum a), giving a gradient
    Lipschitz constant <= 1+2mu. Fixed-step descent is globally convergent; every
    chunk must meet the reported first-order stationarity tolerance or fail.
    This is a soft geometric penalty, not a no-conflict safety guarantee.
    """
    dim = _feature_dim(raw, feature_dim)
    shape = raw.movedim(dim, -1).shape
    r = raw.movedim(dim, -1).reshape(-1, shape[-1])
    ds = [g.direction.float().movedim(dim, -1).reshape_as(r) for g in groups]
    cs = [g.confidence.float().movedim(dim, -1).reshape(-1) for g in groups]
    output = torch.empty_like(r)
    worst_residual, maximum_iterations = 0.0, 0
    before_penalty, after_penalty = 0.0, 0.0
    # Legacy normalized weights sum to <=1. The additive method retains each
    # leaf's own confidence, whose sum is <=number of admitted leaves.
    lipschitz_bound = 1.0 + 2.0 * config.conflict_mu * (1 if normalize_confidence else len(groups))
    alpha = 1.0 / lipschitz_bound
    for start in range(0, r.shape[0], config.conflict_chunk_positions):
        stop = min(r.shape[0], start + config.conflict_chunk_positions)
        target = r[start:stop]
        units = torch.stack([d[start:stop] for d in ds])
        norms = units.square().sum(-1, keepdim=True).sqrt()
        units = units / norms.clamp_min(1e-20)
        weights = torch.stack([a[start:stop] for a in cs])
        if normalize_confidence:
            weights = weights / weights.sum(0, keepdim=True).clamp_min(1.0)
        z = target.clone()
        scale = target.square().sum(-1).sqrt()
        tolerance = config.conflict_atol + config.conflict_rtol * scale
        initial_dots = (units * target.unsqueeze(0)).sum(-1)
        before_penalty += float((weights * initial_dots.clamp_max(0).square()).sum())
        iteration = 0
        while True:
            dots = (units * z.unsqueeze(0)).sum(-1)
            grad = z - target + 2.0 * config.conflict_mu * (
                (weights * dots.clamp_max(0)).unsqueeze(-1) * units
            ).sum(0)
            _finite(grad, "conflict gradient")
            residual = grad.square().sum(-1).sqrt()
            _finite(residual, "conflict stationarity residual")
            if bool((residual <= tolerance).all()):
                break
            next_z = z - alpha * grad
            _finite(next_z, "conflict iterate")
            stagnant = (next_z == z).all(-1) & (residual > tolerance)
            if bool(stagnant.any()):
                raise RuntimeError(
                    "Conflict solve reached floating-point stagnation before stationarity: "
                    f"iteration={iteration}, max residual/tolerance={float((residual[stagnant]/tolerance[stagnant]).max())}"
                )
            z = next_z
            iteration += 1
        _finite(z, "conflict solution")
        worst_residual = max(worst_residual, float((residual/tolerance).max()))
        maximum_iterations = max(maximum_iterations, iteration)
        after_penalty += float((weights * dots.clamp_max(0).square()).sum())
        output[start:stop] = z
    diagnostics = {
        "conflict_solver": "unique_local_hinge_squared_proximal_gradient",
        "conflict_converged": True, "conflict_iterations_max": maximum_iterations,
        "conflict_stationarity_over_tolerance_max": worst_residual,
        "opposition_penalty_before": before_penalty,
        "opposition_penalty_after": after_penalty,
    }
    if not normalize_confidence:
        diagnostics.update(
            conflict_weight_normalization="none_independent_leaf_confidences",
            conflict_step_size=alpha, conflict_gradient_lipschitz_bound=lipschitz_bound,
            conflict_term_count=len(groups),
        )
    return output.reshape(shape).movedim(-1, dim), diagnostics


def additive_leaf_effects(
    terms: Sequence[ConceptTerm], *, candidate: torch.Tensor,
    applied_delta: torch.Tensor,
) -> list[dict[str, Any]]:
    """Observed alignment with requested fields, not a causal allocation of output.

    Candidate is the post-conflict field; applied_delta is the verified actual
    native-prediction-dtype correction after all ceilings and quantization.
    """
    if candidate.shape != applied_delta.shape:
        raise ValueError("Candidate and applied correction shapes differ")
    rows = []
    for term in sorted(terms, key=lambda item: item.concept_id):
        if term.direction.shape != candidate.shape:
            raise ValueError("Leaf and combined correction shapes differ")
        requested = term.confidence.detach().float() * term.direction.detach().float()
        rows.append({
            "node_id": term.concept_id,
            "scope": "global_field_association_not_independent_causal_contribution",
            "requested_rms": float(_rms(requested)),
            "requested_nonzero_feature_fraction": float((requested != 0).float().mean()),
            "post_conflict_alignment": _field_alignment_diagnostics(candidate.detach(), requested),
            "post_budget_native_prediction_alignment": _field_alignment_diagnostics(applied_delta.detach(), requested),
            "feeds_steering": False,
        })
    return rows


def _combine_additive_leaves(
    terms: Sequence[ConceptTerm], *, feature_dim: int, config: ControllerConfig,
    effective_weight: float, reference: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Independent leaf sum, followed by an unnormalized leaf conflict objective."""
    if effective_weight != 1.0 or config.lambda_max != 1.0:
        raise ValueError("Additive sigmoid steering has no extra global strength or temporal multiplier")
    if len(terms) > 25:
        raise ValueError("Additive correction cannot contain more than25 leaves")
    if reference is None:
        if not terms:
            raise ValueError("Empty additive terms require a reference field")
        reference = terms[0].direction
    dim = _feature_dim(reference, feature_dim)
    expected = list(reference.shape)
    expected[dim] = 1
    raw = torch.zeros_like(reference, dtype=torch.float32)
    ids, pairs = set(), set()
    ordered = sorted(terms, key=lambda item: item.concept_id)
    for term in ordered:
        if term.concept_id in ids or term.alias_key in pairs:
            raise ValueError("Additive input contains duplicate IDs or undeduplicated exact pairs")
        ids.add(term.concept_id)
        pairs.add(term.alias_key)
        if term.direction.shape != reference.shape or list(term.confidence.shape) != expected:
            raise ValueError("Additive leaf fields have incompatible shapes")
        _finite(term.direction, "additive direction")
        _finite(term.confidence, "additive confidence")
        if bool(((term.confidence < 0) | (term.confidence > 1)).any()):
            raise ValueError("Additive leaf confidence must lie in [0,1]")
        raw = raw + term.confidence.float() * term.direction.float()
    raw_rms = float(_rms(raw))
    positive = [term for term in ordered if bool((term.confidence > 0).any())]
    diagnostics: dict[str, Any] = {
        "aggregation_mode": "additive_leaf_v1", "requested_aggregation_mode": config.aggregation_mode,
        "direction_averaging": False, "confidence_sum_normalization": False,
        "selected_count_normalization": False, "hierarchy_weighting": False,
        "alias_policy": "exact_text_pairs_deduplicated_before_admission; no_vector_averaging",
        "concept_count": len(ordered), "raw_leaf_count": len(ordered),
        "active_raw_leaf_count": len(positive), "group_count": len(positive),
        "grouped_leaf_count": len(positive), "aggregation_term_count": len(ordered),
        "aggregation_term_ids": [term.concept_id for term in ordered],
        "confidence_mass_raw_leaves": sum(float(term.confidence.float().sum()) for term in ordered),
        "raw_rms": raw_rms, "effective_weight": 1.0, "lambda_max": 1.0,
    }
    if config.conflict_handling and config.conflict_mu > 0 and len(positive) > 1:
        raw, conflict = _conflict_prox(raw, positive, dim, config, normalize_confidence=False)
        diagnostics.update(conflict)
    _finite(raw, "additive combined candidate")
    diagnostics["candidate_rms"] = float(_rms(raw))
    diagnostics["requested_leaf_fields"] = [
        {"node_id": term.concept_id,
         "requested_rms": float(_rms(term.confidence.float() * term.direction.float())),
         "post_conflict_alignment": _field_alignment_diagnostics(
             raw.detach(), (term.confidence.float() * term.direction.float()).detach()),
         "scope": "association_with_combined_field_not_causal_attribution"}
        for term in ordered
    ]
    return raw, diagnostics


def combine_terms(
    terms: Sequence[ConceptTerm], *, feature_dim: int, config: ControllerConfig,
    effective_weight: float = 1.0, reference: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Group aliases, preserve weak confidence, and resolve local conflicts."""
    if not math.isfinite(effective_weight) or effective_weight < 0:
        raise ValueError("effective_weight must be finite and nonnegative")
    if config.aggregation_mode == "additive_leaf_v1":
        return _combine_additive_leaves(
            terms, feature_dim=feature_dim, config=config,
            effective_weight=effective_weight, reference=reference,
        )
    # C06 introduces grouping/overlap normalization. Its hierarchical redesign
    # is disabled with that component, retaining earlier cumulative controls.
    aggregation_mode = config.aggregation_mode if config.group_redundancy else "flat"
    if not terms:
        if reference is None:
            raise ValueError("Empty term list requires a reference field")
        dim = _feature_dim(reference, feature_dim)
        empty = _cancellation_diagnostics([], reference=reference, feature_dim=dim)
        return torch.zeros_like(reference, dtype=torch.float32), {
            "concept_count": 0, "group_count": 0, "raw_leaf_count": 0,
            "active_raw_leaf_count": 0, "grouped_leaf_count": 0,
            "confidence_mass_raw_leaves": 0.0, "confidence_mass_groups": 0.0,
            "aggregation_mode": aggregation_mode,
            "requested_aggregation_mode": config.aggregation_mode,
            "aggregation_term_count": 0,
            "aggregation_term_ids": [], "aggregation_confidence_mass": {},
            "cancellation": {"before_group_aggregation": empty, "after_group_aggregation": dict(empty)},
        }
    reference = terms[0].direction if reference is None else reference
    dim = _feature_dim(reference, feature_dim)
    for term in terms:
        if term.direction.shape != reference.shape:
            raise ValueError("Direction shapes differ")
        expected = list(reference.shape)
        expected[dim] = 1
        if list(term.confidence.shape) != expected:
            raise ValueError("Confidence needs a singleton feature axis")
        _finite(term.direction, "direction")
        _finite(term.confidence, "confidence")
        if bool(((term.confidence < 0) | (term.confidence > 1)).any()):
            raise ValueError("Confidence must lie in [0, 1]")
    groups = _group_terms(terms, config)
    groups = [g for g in groups if bool((g.confidence > 0).any())]
    aggregation_terms = (
        _hierarchy_balanced_terms(groups)
        if aggregation_mode == "hierarchy_balanced" else groups
    )
    raw = torch.zeros_like(reference, dtype=torch.float32)
    weight = torch.zeros_like(terms[0].confidence, dtype=torch.float32)
    for group in aggregation_terms:
        raw = raw + group.confidence.float() * group.direction.float()
        weight = weight + group.confidence.float()
    if config.group_redundancy:
        raw = raw / weight.clamp_min(1.0)
    raw = raw * (config.lambda_max * effective_weight)
    diagnostics: dict[str, Any] = {
        "concept_count": len(terms), "group_count": len(groups),
        "overlap_confidence_max": float(weight.max()), "raw_rms": float(_rms(raw)),
        "lambda_max": config.lambda_max, "effective_weight": effective_weight,
        "aggregation_mode": aggregation_mode,
        "requested_aggregation_mode": config.aggregation_mode,
        "aggregation_term_count": len(aggregation_terms),
        "aggregation_term_ids": [term.concept_id for term in aggregation_terms],
        "aggregation_confidence_mass": {
            term.concept_id: float(term.confidence.float().sum()) for term in aggregation_terms
        },
        "raw_leaf_count": len(terms),
        "active_raw_leaf_count": sum(bool((term.confidence > 0).any()) for term in terms),
        "grouped_leaf_count": len(groups),
        "confidence_mass_raw_leaves": sum(float(term.confidence.float().sum()) for term in terms),
        "confidence_mass_groups": sum(float(group.confidence.float().sum()) for group in groups),
        "cancellation": {
            "before_group_aggregation": _cancellation_diagnostics(terms, reference=reference, feature_dim=dim),
            "after_group_aggregation": _cancellation_diagnostics(groups, reference=reference, feature_dim=dim),
        },
    }
    if config.conflict_handling and config.conflict_mu > 0 and len(aggregation_terms) > 1:
        raw, conflict_diagnostics = _conflict_prox(raw, aggregation_terms, dim, config)
        diagnostics.update(conflict_diagnostics)
    _finite(raw, "combined candidate")
    diagnostics["candidate_rms"] = float(_rms(raw))
    if aggregation_mode == "hierarchy_balanced":
        diagnostics["root_aggregation_geometry"] = _root_aggregation_diagnostics(
            aggregation_terms, total_confidence=weight, reference=reference,
            feature_dim=dim, normalize_overlap=config.group_redundancy,
        )
    return raw, diagnostics


def apply_correction(
    v_base: torch.Tensor, v_neutral: torch.Tensor, candidate: torch.Tensor, *,
    feature_dim: int, delta_sigma: float, exposure_state: ExposureState,
    config: ControllerConfig,
) -> AppliedCorrection:
    """Apply shrink-only ceilings, then verify the actual native-dtype update.

    Quantization may change both the magnitude and direction. Offending local
    cells are repeatedly shrunk before quantization; cells that cannot represent
    a compliant update revert exactly to base. Global and exposure ceilings are
    then rechecked. Diagnostics distinguish candidate and actual applied delta.
    Exposure is Σ abs(delta_sigma) * (RMS(actual delta)/RMS(base))², accumulated
    only after verification. It is a stated proxy, not a solver trajectory bound.
    """
    dim = _feature_dim(v_base, feature_dim)
    if candidate.shape != v_base.shape or v_neutral.shape != v_base.shape:
        raise ValueError("Correction, neutral and base shapes must agree")
    if not math.isfinite(delta_sigma):
        raise ValueError("Nonfinite delta_sigma")
    if not math.isfinite(exposure_state.cumulative_exposure) or exposure_state.cumulative_exposure < 0:
        raise ValueError("Invalid exposure state")
    for value in (v_base, v_neutral, candidate):
        _finite(value, "application input")
    base, neutral, work = v_base.float(), v_neutral.float(), candidate.float()
    base_rms, residual_rms = float(_rms(base)), float(_rms(base - neutral))
    ds = abs(float(delta_sigma))
    candidate_rms = float(_rms(work))
    global_cap = config.global_relative_rms * base_rms
    base_relative_cap = global_cap
    prompt_relative_cap = None
    if config.center_residuals and config.prompt_residual_relative_rms is not None:
        prompt_relative_cap = config.prompt_residual_relative_rms * residual_rms
        global_cap = min(global_cap, prompt_relative_cap)
    exposure_enabled = config.cumulative_exposure_limit is not None
    remaining = (max(0.0, config.cumulative_exposure_limit - exposure_state.cumulative_exposure)
                 if exposure_enabled else None)
    exposure_cap = (base_rms * math.sqrt(remaining / ds)
                    if remaining is not None and ds > 0 else math.inf)
    global_cap = min(global_cap, exposure_cap)
    local_cap = config.local_relative_rms * base.square().mean(dim=dim, keepdim=True).sqrt()
    local_scale = torch.ones_like(local_cap)
    global_scale = 1.0
    post_local_rms = candidate_rms
    if config.norm_budgets:
        local_rms = work.square().mean(dim=dim, keepdim=True).sqrt()
        local_scale = (local_cap / local_rms.clamp_min(1e-30)).clamp(max=1.0)
        work = work * local_scale
        bounded_rms = float(_rms(work))
        post_local_rms = bounded_rms
        global_scale = min(1.0, global_cap / max(bounded_rms, 1e-30))
        work = work * global_scale
    post_global_rms = float(_rms(work))
    prediction = (base + work).to(dtype=v_base.dtype)
    actual = prediction.float() - base
    quantization_iterations = 0
    reverted_positions = 0
    quantization_global_shrink = 1.0
    quantization_local_updates = 0
    if config.norm_budgets:
        # No additive tolerance at a zero cap: zero budgets mean exactly zero.
        local_tolerance = torch.where(local_cap > 0,
                                      config.budget_atol + config.budget_rtol * local_cap,
                                      torch.zeros_like(local_cap))
        global_tolerance = config.budget_atol + config.budget_rtol * global_cap if global_cap > 0 else 0.0
        for iteration in range(config.quantization_max_iterations + 1):
            actual_local = actual.square().mean(dim=dim, keepdim=True).sqrt()
            bad_local = actual_local > local_cap + local_tolerance
            actual_rms = float(_rms(actual))
            bad_global = actual_rms > min(global_cap + global_tolerance,
                                          exposure_cap * (1 + config.budget_rtol))
            if not bool(bad_local.any()) and not bad_global:
                break
            quantization_iterations = iteration + 1
            if iteration == config.quantization_max_iterations:
                # Explicit fail-closed representability fallback, recorded below.
                if bool(bad_local.any()):
                    reverted_positions += int(bad_local.sum())
                    work = torch.where(bad_local, torch.zeros_like(work), work)
                prediction = (base + work).to(dtype=v_base.dtype)
                actual = prediction.float() - base
                if float(_rms(actual)) > min(global_cap + global_tolerance,
                                             exposure_cap * (1 + config.budget_rtol)):
                    reverted_positions = local_cap.numel()
                    prediction, actual = v_base.clone(), torch.zeros_like(base)
                break
            if bool(bad_local.any()):
                if config.quantization_backtracking == "legacy_half":
                    local_backtrack = torch.full_like(local_cap, 0.5)
                else:
                    # The observed native-dtype norm determines the shrink.
                    # The 1% inward margin avoids repeated boundary violations;
                    # all resulting quantized fields are still verified below.
                    local_backtrack = (0.99 * local_cap / actual_local.clamp_min(1e-30)).clamp(max=0.99)
                quantization_local_updates += int(bad_local.sum())
                work = torch.where(bad_local, work * local_backtrack, work)
            if bad_global:
                maximum_shrink = 0.5 if config.quantization_backtracking == "legacy_half" else 0.99
                backtrack = min(maximum_shrink, global_cap / max(actual_rms, 1e-30) * 0.99)
                quantization_global_shrink *= backtrack
                work = work * backtrack
            prediction = (base + work).to(dtype=v_base.dtype)
            actual = prediction.float() - base
        actual_local = actual.square().mean(dim=dim, keepdim=True).sqrt()
        if bool((actual_local > local_cap + local_tolerance).any()) or float(_rms(actual)) > min(
            global_cap + global_tolerance, exposure_cap * (1 + config.budget_rtol)
        ):
            raise RuntimeError("Actual native-dtype correction violates norm ceilings")
    _finite(prediction, "applied prediction")
    applied_rms = float(_rms(actual))
    if base_rms == 0:
        relative_rms = 0.0 if applied_rms == 0 else math.inf
    else:
        relative_rms = applied_rms / base_rms
    added_exposure = ds * relative_rms**2 if ds > 0 else 0.0
    previous_exposure = exposure_state.cumulative_exposure
    next_exposure = previous_exposure + added_exposure
    if config.norm_budgets and exposure_enabled and next_exposure > config.cumulative_exposure_limit + max(
        1e-15, config.cumulative_exposure_limit * (3 * config.budget_rtol)
    ):
        raise RuntimeError("Actual native-dtype correction exceeds cumulative exposure")
    exposure_state.cumulative_exposure = next_exposure
    exposure_state.steps += 1
    # Zero-reference dose is undefined for a nonzero correction. Keep JSON finite
    # and expose that condition instead of inventing a normalized measurement.
    def dose(rms: float) -> float | None:
        if ds == 0 or rms == 0:
            return 0.0
        return ds * (rms / base_rms) ** 2 if base_rms > 0 else None
    candidate_dose, local_dose, global_dose = (
        dose(candidate_rms), dose(post_local_rms), dose(post_global_rms)
    )
    return AppliedCorrection(prediction, actual, {
        "candidate_rms": candidate_rms, "applied_delta_rms": applied_rms,
        "base_rms": base_rms, "prompt_residual_rms": residual_rms,
        "applied_relative_base_rms": relative_rms,
        "applied_relative_prompt_residual_rms": applied_rms / residual_rms if residual_rms > 0 else (0.0 if applied_rms == 0 else math.inf),
        "global_rms_cap": global_cap, "local_rms_cap_max": float(local_cap.max()),
        "base_relative_rms_cap": base_relative_cap,
        "prompt_relative_rms_cap": prompt_relative_cap,
        "remaining_exposure_rms_cap": exposure_cap if math.isfinite(exposure_cap) else None,
        "local_shrink_fraction": float((local_scale < 1).float().mean()),
        "global_shrink_factor": global_scale,
        "post_local_rms": post_local_rms, "post_global_rms": post_global_rms,
        "candidate_relative_squared_dose": candidate_dose,
        "post_local_relative_squared_dose": local_dose,
        "post_global_relative_squared_dose": global_dose,
        "local_removed_relative_squared_dose": (
            max(0.0, candidate_dose - local_dose)
            if candidate_dose is not None and local_dose is not None else None
        ),
        "global_removed_relative_squared_dose": (
            max(0.0, local_dose - global_dose)
            if local_dose is not None and global_dose is not None else None
        ),
        "local_cap_binding": bool((local_scale < 1).any()) if config.norm_budgets else False,
        "global_cap_binding": global_scale < 1.0 if config.norm_budgets else False,
        "dose_scope": "native_sigma_interval_times_squared_global_rms_relative_to_base; clipping_before_BF16",
        "quantization_backtrack_iterations": quantization_iterations,
        "quantization_reverted_positions": reverted_positions,
        "quantization_backtracking": config.quantization_backtracking,
        "quantization_global_shrink_factor": quantization_global_shrink,
        "quantization_local_update_count": quantization_local_updates,
        "prediction_dtype": str(v_base.dtype), "abs_delta_sigma": ds,
        "exposure_reference": "RMS(actual native-dtype prediction minus base) / RMS(base)",
        "exposure_before": previous_exposure, "exposure_increment": added_exposure,
        "exposure_after": next_exposure,
        "exposure_limit": config.cumulative_exposure_limit,
        "exposure_cap_enabled": exposure_enabled and config.norm_budgets,
        "prompt_residual_cap_enabled": (
            config.norm_budgets and config.center_residuals
            and config.prompt_residual_relative_rms is not None
        ),
        "norm_budgets_enabled": config.norm_budgets,
    })
