"""Paired-reference SGF and Safe Denoiser switch adaptations.

The pinned negative terms remain identifiable: SGF uses the published raw
kernel-sum repulsion and Safe Denoiser uses the released SD3 KDE subtraction.
Switching adds a separately traced target counterpart constructed by the same
kernel rule. Float32 is retained through conversion back to native model space.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from . import distribution_switch as _legacy


SGF_PAIRED_FORCE_POLICY = (
    "paper_negative_raw_kernel_sum_repulsive_plus_paired_target_attraction_v1"
)
SAFE_PAIRED_CORRECTION_POLICY = (
    "released_sd3_source_subtraction_plus_paired_target_addition_v1"
)


def __getattr__(name: str) -> Any:
    """Expose the sealed switch-reference artifact API from the original module."""

    return getattr(_legacy, name)


def _role_pairs(
    references: Mapping[str, Any] | _legacy.SwitchReferenceArtifact,
    model_role: str,
) -> Mapping[str, Any]:
    if isinstance(references, _legacy.SwitchReferenceArtifact):
        try:
            selected = references.references[model_role]
        except KeyError as exc:
            raise KeyError(
                f"No exact paired switch references for role {model_role!r}; "
                f"available roles are {sorted(references.references)!r}"
            ) from exc
        if not isinstance(selected, Mapping):
            raise ValueError(f"Malformed paired references for role {model_role!r}")
        return selected
    root: Mapping[str, Any] = references
    nested = root.get("references")
    if isinstance(nested, Mapping):
        root = nested
    selected = root.get(model_role)
    if not isinstance(selected, Mapping):
        raise KeyError(
            f"No exact paired switch references for role {model_role!r}; "
            f"available roles are {sorted(str(key) for key in root)!r}"
        )
    return selected


def _paired_banks(
    current_x0: torch.Tensor,
    references: Mapping[str, Any] | _legacy.SwitchReferenceArtifact,
    model_role: str,
    pair_ids: list[str] | tuple[str, ...] | None,
    *,
    method: str,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[str, int, int]]]:
    pairs = _role_pairs(references, model_role)
    available = [str(pair_id) for pair_id in pairs]
    selected = available if pair_ids is None else [str(pair_id) for pair_id in pair_ids]
    if len(selected) != len(set(selected)) or set(selected) != set(available):
        raise ValueError(
            f"{method} concept-pair provenance does not exactly match paired banks: "
            f"campaign={sorted(selected)!r}, artifact={sorted(available)!r}"
        )
    source_values: list[torch.Tensor] = []
    target_values: list[torch.Tensor] = []
    spans: list[tuple[str, int, int]] = []
    offset = 0
    expected_shape = tuple(current_x0.shape[1:])
    for pair_id in selected:
        sides = pairs.get(pair_id)
        if not isinstance(sides, Mapping) or set(map(str, sides)) != {
            "source",
            "target",
        }:
            raise ValueError(
                f"{method} pair {pair_id!r} must contain exactly source and target banks"
            )
        source = sides["source"]
        target = sides["target"]
        if not torch.is_tensor(source) or not torch.is_tensor(target):
            raise ValueError(f"{method} pair {pair_id!r} banks must be tensors")
        if (
            source.ndim < 2
            or target.ndim != source.ndim
            or tuple(source.shape[1:]) != expected_shape
            or tuple(target.shape[1:]) != expected_shape
            or int(source.shape[0]) < 2
            or int(target.shape[0]) != int(source.shape[0])
        ):
            raise ValueError(
                f"{method} pair {pair_id!r} has invalid paired shapes "
                f"{tuple(source.shape)} and {tuple(target.shape)} for current "
                f"{tuple(current_x0.shape)}"
            )
        if not torch.isfinite(source).all() or not torch.isfinite(target).all():
            raise ValueError(f"{method} pair {pair_id!r} banks contain non-finite values")
        source = source.to(device=current_x0.device, dtype=torch.float32)
        target = target.to(device=current_x0.device, dtype=torch.float32)
        source_values.append(source)
        target_values.append(target)
        end = offset + int(source.shape[0])
        spans.append((pair_id, offset, end))
        offset = end
    return (
        torch.cat(source_values, dim=0),
        torch.cat(target_values, dim=0),
        spans,
    )


def _feature_projection(data: torch.Tensor, feature_dim: int) -> torch.Tensor:
    dim = int(feature_dim)
    if dim < 0:
        dim += data.ndim
    if dim <= 0 or dim >= data.ndim:
        raise ValueError(
            f"Safe Denoiser feature_dim must identify a non-batch axis; "
            f"got {feature_dim} for shape {tuple(data.shape)}"
        )
    return F.normalize(data.to(torch.float32), p=2.0, dim=dim, eps=1.0e-12)


def _sgf_kernel_field(
    x: torch.Tensor,
    refs: torch.Tensor,
    *,
    top_k: int,
    epsilon: float,
    orientation: float,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    squared_distances = torch.cdist(x, refs, p=2).square()
    if refs.shape[0] < 2:
        raise ValueError("SGF requires at least two references per paired bank")
    neighbour_count = min(max(1, int(top_k)), int(refs.shape[0]) - 1)
    sorted_distances = torch.sort(squared_distances, dim=1).values
    nearest = sorted_distances[:, 1 : neighbour_count + 1]
    bandwidth_distance = max(float(nearest.mean().item()), 1.0e-12)
    gamma = -math.log(float(epsilon)) / bandwidth_distance
    kernels = torch.exp(-gamma * squared_distances)
    differences = x[:, None, :] - refs[None, :, :]
    field = float(orientation) * 2.0 * gamma * (
        kernels[..., None] * differences
    ).sum(dim=1)
    return field, kernels, gamma, bandwidth_distance


@torch.no_grad()
def sgf_switch_x0(
    current_x0: torch.Tensor,
    references: Mapping[str, Any] | _legacy.SwitchReferenceArtifact,
    *,
    model_role: str,
    pair_ids: list[str] | tuple[str, ...] | None = None,
    strength: float,
    force_policy: str,
    top_k: int = 3,
    epsilon: float = 0.05,
) -> tuple[torch.Tensor, dict[str, dict[str, float | int | bool | str]]]:
    """Apply published SGF source repulsion plus paired target attraction.

    Appendix D.1 calibrates the experimental scale against the unnormalised
    gradient of ``sum_j k(x, y_j)``.  Negative guidance negates that attractive
    gradient. The switch adaptation adds the sign-reversed field of an equally
    populated preregistered target bank.
    """

    if strength < 0:
        raise ValueError("SGF strength must be non-negative")
    if force_policy != SGF_PAIRED_FORCE_POLICY:
        raise ValueError(
            f"SGF paired switching requires force_policy={SGF_PAIRED_FORCE_POLICY!r}"
        )
    if top_k <= 0:
        raise ValueError("SGF top_k must be positive")
    if not 0 < epsilon < 1:
        raise ValueError("SGF epsilon must lie strictly between zero and one")

    x_shape = current_x0.shape
    x = current_x0.detach().to(torch.float32).reshape(current_x0.shape[0], -1)
    source_bank, target_bank, spans = _paired_banks(
        current_x0,
        references,
        model_role,
        pair_ids,
        method="SGF",
    )
    source_refs = source_bank.reshape(source_bank.shape[0], -1)
    target_refs = target_bank.reshape(target_bank.shape[0], -1)
    source_force, source_kernels, source_gamma, source_bandwidth = (
        _sgf_kernel_field(
            x,
            source_refs,
            top_k=top_k,
            epsilon=epsilon,
            orientation=1.0,
        )
    )
    target_force, target_kernels, target_gamma, target_bandwidth = (
        _sgf_kernel_field(
            x,
            target_refs,
            top_k=top_k,
            epsilon=epsilon,
            orientation=-1.0,
        )
    )
    total_force = source_force + target_force
    pair_forces: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for pair_id, start, end in spans:
        source_difference = x[:, None, :] - source_refs[None, start:end, :]
        target_difference = x[:, None, :] - target_refs[None, start:end, :]
        source_contribution = 2.0 * source_gamma * (
            source_kernels[:, start:end, None] * source_difference
        ).sum(dim=1)
        target_contribution = -2.0 * target_gamma * (
            target_kernels[:, start:end, None] * target_difference
        ).sum(dim=1)
        pair_forces[pair_id] = (source_contribution, target_contribution)

    delta = float(strength) * total_force
    steered = (x + delta).reshape(x_shape)
    total_delta_norm = float(delta.norm().item())
    details: dict[str, dict[str, float | int | bool | str]] = {}
    for pair_id, start, end in spans:
        source_contribution, target_contribution = pair_forces[pair_id]
        source_delta = float(strength) * source_contribution
        target_delta = float(strength) * target_contribution
        pair_delta = source_delta + target_delta
        details[pair_id] = {
            "delta_norm": float(pair_delta.norm().item()),
            "mathematical_x0_delta_norm": total_delta_norm,
            "source_repulsive_delta_norm": float(source_delta.norm().item()),
            "target_attractive_delta_norm": float(target_delta.norm().item()),
            "source_gamma": float(source_gamma),
            "target_gamma": float(target_gamma),
            "source_bandwidth_squared_distance": source_bandwidth,
            "target_bandwidth_squared_distance": target_bandwidth,
            "source_kernel_mean": float(
                source_kernels[:, start:end].mean().item()
            ),
            "target_kernel_mean": float(
                target_kernels[:, start:end].mean().item()
            ),
            "source_reference_count": int(end - start),
            "target_reference_count": int(end - start),
            "total_source_reference_count": int(source_refs.shape[0]),
            "total_target_reference_count": int(target_refs.shape[0]),
            "target_references_used": True,
            "gradient_convention": (
                "paper_source_repulsion_plus_paired_target_attraction"
            ),
            "force_policy": force_policy,
            "source_reference_count_normalization": (
                "none_published_raw_kernel_sum"
            ),
            "target_reference_count_normalization": (
                "none_symmetric_paired_adaptation"
            ),
            "source_force_coefficient": float(2.0 * source_gamma),
            "target_force_coefficient": float(-2.0 * target_gamma),
            "bandwidth_neighbour_policy": "paper_skip_first_then_next_k",
            "precision_policy": "float32_through_native_schedule_conversion",
            "adaptation": "sgf_paired_source_target_switch_v5",
        }
    return steered, details


@torch.no_grad()
def safe_denoiser_switch_x0(
    current_x0: torch.Tensor,
    references: Mapping[str, Any] | _legacy.SwitchReferenceArtifact,
    *,
    model_role: str,
    pair_ids: list[str] | tuple[str, ...] | None = None,
    sigma: float,
    scale: float,
    feature_dim: int,
    kernel_distance_policy: str,
    correction_policy: str,
    beta_threshold_margin: float,
    official_reference_population: int,
    threshold_quantile: float = 0.0,
) -> tuple[torch.Tensor, dict[str, dict[str, float | int | bool | str]]]:
    """Apply released source subtraction plus paired target addition.

    The released SD3 code L2-normalizes the latent channel axis before kernel
    distance evaluation.  Packed transformer latents require the same operation
    on their feature axis. Kernel weights are computed in projected space, while
    both kernel-regressed denoisers remain in original x0 space. The source term
    exactly preserves ``x0 -= scale * unsafe_denoiser``; switching adds the
    symmetric target-bank KDE estimate.
    """

    if sigma <= 0:
        raise ValueError("Safe Denoiser sigma must be positive")
    if scale < 0:
        raise ValueError("Safe Denoiser scale must be non-negative")
    if beta_threshold_margin < 0:
        raise ValueError("Safe Denoiser beta_threshold_margin must be non-negative")
    if official_reference_population <= 0:
        raise ValueError("Safe Denoiser official_reference_population must be positive")
    if not 0.0 <= threshold_quantile <= 1.0:
        raise ValueError("Safe Denoiser threshold_quantile must lie in [0, 1]")
    if correction_policy != SAFE_PAIRED_CORRECTION_POLICY:
        raise ValueError(
            "Safe Denoiser paired switching requires correction_policy="
            f"{SAFE_PAIRED_CORRECTION_POLICY!r}, got {correction_policy!r}"
        )

    x_shape = current_x0.shape
    x_tensor = current_x0.detach().to(torch.float32)
    x = x_tensor.reshape(current_x0.shape[0], -1)
    source_bank, target_bank, spans = _paired_banks(
        current_x0,
        references,
        model_role,
        pair_ids,
        method="Safe Denoiser",
    )
    source_refs = source_bank.reshape(source_bank.shape[0], -1)
    target_refs = target_bank.reshape(target_bank.shape[0], -1)
    if (
        int(source_refs.shape[0]) != int(official_reference_population)
        or int(target_refs.shape[0]) != int(official_reference_population)
    ):
        raise ValueError(
            "Safe Denoiser paired banks must each match "
            f"official_reference_population={official_reference_population}; got "
            f"source={source_refs.shape[0]}, target={target_refs.shape[0]}"
        )

    projected_x = _feature_projection(x_tensor, feature_dim).reshape(x.shape[0], -1)
    projected_source = _feature_projection(source_bank, feature_dim).reshape(
        source_bank.shape[0], -1
    )
    projected_target = _feature_projection(target_bank, feature_dim).reshape(
        target_bank.shape[0], -1
    )
    source_distances = torch.cdist(projected_x, projected_source, p=2)
    target_distances = torch.cdist(projected_x, projected_target, p=2)
    if kernel_distance_policy == "released_unsquared_l2":
        source_kernel_distances = source_distances
        target_kernel_distances = target_distances
    else:
        raise ValueError(
            "Safe Denoiser requires the explicit pinned-release kernel policy "
            f"'released_unsquared_l2', got {kernel_distance_policy!r}"
        )
    source_log_kernels = -source_kernel_distances / (2.0 * float(sigma) ** 2)
    target_log_kernels = -target_kernel_distances / (2.0 * float(sigma) ** 2)
    source_weights = torch.softmax(source_log_kernels, dim=1)
    target_weights = torch.softmax(target_log_kernels, dim=1)
    unsafe_denoiser = source_weights @ source_refs
    target_denoiser = target_weights @ target_refs
    source_log_beta = torch.logsumexp(source_log_kernels, dim=1) - math.log(
        float(source_refs.shape[0])
    )
    target_log_beta = torch.logsumexp(target_log_kernels, dim=1) - math.log(
        float(target_refs.shape[0])
    )
    source_beta = torch.exp(source_log_beta)
    target_beta = torch.exp(target_log_beta)

    # The released SD3 configuration disables beta thresholding. Density is
    # retained only for diagnostics and never scales or gates the correction.
    beta_threshold = 0.0
    reference_density_floor = 0.0
    normalized_margin = 0.0
    source_subtraction = -unsafe_denoiser
    target_addition = target_denoiser
    switch_direction = source_subtraction + target_addition
    delta = float(scale) * switch_direction
    steered = (x + delta).reshape(x_shape)

    total_delta_norm = float(delta.norm().item())
    direction_norm = float(switch_direction.norm().item())
    details: dict[str, dict[str, float | int | bool | str]] = {}
    for pair_id, start, end in spans:
        source_component = -(
            source_weights[:, start:end] @ source_refs[start:end]
        )
        target_component = target_weights[:, start:end] @ target_refs[start:end]
        pair_delta = float(scale) * (source_component + target_component)
        source_mass = source_weights[:, start:end].sum(dim=1).mean()
        target_mass = target_weights[:, start:end].sum(dim=1).mean()
        details[pair_id] = {
            "delta_norm": float(pair_delta.norm().item()),
            "mathematical_x0_delta_norm": total_delta_norm,
            "raw_switch_direction_norm": direction_norm,
            "source_subtraction_norm": float(
                (float(scale) * source_component).norm().item()
            ),
            "target_addition_norm": float(
                (float(scale) * target_component).norm().item()
            ),
            "source_posterior_mass": float(source_mass.item()),
            "target_posterior_mass": float(target_mass.item()),
            "unsafe_density_beta": float(source_beta.mean().item()),
            "target_density_beta": float(target_beta.mean().item()),
            "log_unsafe_density_beta": float(source_log_beta.mean().item()),
            "log_target_density_beta": float(target_log_beta.mean().item()),
            "beta_threshold": float(beta_threshold),
            "reference_density_quantile": float(reference_density_floor),
            "threshold_quantile": float(threshold_quantile),
            "normalized_beta_threshold_margin": float(normalized_margin),
            "beta_margin_reference_count": int(source_refs.shape[0]),
            "official_reference_population": int(official_reference_population),
            "intervention_applied": bool(
                float(scale) > 0.0 and float(pair_delta.norm().item()) > 0.0
            ),
            "is_negation": bool(float(scale) > 0.0),
            "is_negation_fraction": float(float(scale) > 0.0),
            "source_projected_l2_distance_mean": float(
                source_distances.mean().item()
            ),
            "target_projected_l2_distance_mean": float(
                target_distances.mean().item()
            ),
            "kernel_distance_policy": str(kernel_distance_policy),
            "correction_policy": str(correction_policy),
            "feature_dim": int(feature_dim),
            "source_reference_count": int(end - start),
            "target_reference_count": int(end - start),
            "total_source_reference_count": int(source_refs.shape[0]),
            "total_target_reference_count": int(target_refs.shape[0]),
            "target_references_used": True,
            "kernel_space": "l2_normalized_feature_axis_unsquared_l2_rbf",
            "direction_space": "raw_canonical_predicted_x0",
            "threshold_policy": "disabled_released_sd3_beta_threshold_false",
            "precision_policy": "float32_through_native_schedule_conversion",
            "adaptation": "safe_denoiser_paired_source_target_switch_v6",
        }
    return steered, details
