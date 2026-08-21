"""Paper-faithful SGF and Safe Denoiser concept-switch adaptations.

The source side of every concept pair is treated as the unsafe distribution.
Target references remain evaluation labels; neither SGF nor Safe Denoiser defines
positive guidance toward a target distribution.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from . import distribution_switch as _legacy


def __getattr__(name: str) -> Any:
    """Keep the existing artifact loader API while replacing only the math."""

    return getattr(_legacy, name)


def _role_pairs(
    references: Mapping[str, Any], model_role: str
) -> Mapping[str, Mapping[str, torch.Tensor]]:
    root: Mapping[str, Any] = references
    nested = root.get("references")
    if isinstance(nested, Mapping):
        root = nested
    selected = root.get(model_role)
    if isinstance(selected, Mapping):
        return selected
    if len(root) == 1:
        only = next(iter(root.values()))
        if isinstance(only, Mapping):
            return only
    raise KeyError(
        f"No switch references for model role {model_role!r}; "
        f"available roles are {sorted(str(key) for key in root)}"
    )


def _source_bank(
    current_x0: torch.Tensor,
    references: Mapping[str, Any] | _legacy.SwitchReferenceArtifact,
    model_role: str,
    pair_ids: list[str] | tuple[str, ...] | None = None,
) -> tuple[torch.Tensor, list[tuple[str, int, int]]]:
    if isinstance(references, _legacy.SwitchReferenceArtifact):
        role = model_role if model_role in references.references else "default"
        try:
            pairs = references.references[role]
        except KeyError as exc:
            raise KeyError(
                f"No switch references for model role {model_role!r}; "
                f"available roles are {sorted(references.references)!r}"
            ) from exc
    else:
        pairs = _role_pairs(references, model_role)

    available_pair_ids = [str(pair_id) for pair_id in pairs]
    requested_pair_ids = (
        available_pair_ids if pair_ids is None else [str(pair_id) for pair_id in pair_ids]
    )
    if set(requested_pair_ids) != set(available_pair_ids):
        raise ValueError(
            "Campaign concept-pair provenance does not match the calibrated source banks: "
            f"campaign={sorted(requested_pair_ids)!r}, "
            f"artifact={sorted(available_pair_ids)!r}"
        )

    banks: list[torch.Tensor] = []
    spans: list[tuple[str, int, int]] = []
    offset = 0
    for pair_id in requested_pair_ids:
        if isinstance(references, _legacy.SwitchReferenceArtifact):
            source = references.get(
                model_role, pair_id, "source", device=current_x0.device
            )
        else:
            pair = pairs[pair_id]
            source = pair.get("source")
        if not torch.is_tensor(source):
            raise TypeError(f"Reference pair {pair_id!r} has no tensor source bank")
        if source.ndim != current_x0.ndim or source.shape[1:] != current_x0.shape[1:]:
            raise ValueError(
                f"Reference pair {pair_id!r} has shape {tuple(source.shape)}, "
                f"incompatible with x0 shape {tuple(current_x0.shape)}"
            )
        converted = source.detach().to(device=current_x0.device, dtype=torch.float32)
        banks.append(converted)
        spans.append((str(pair_id), offset, offset + converted.shape[0]))
        offset += converted.shape[0]
    if not banks:
        raise ValueError(f"No source references found for model role {model_role!r}")
    return torch.cat(banks, dim=0), spans


@torch.no_grad()
def sgf_switch_x0(
    current_x0: torch.Tensor,
    references: Mapping[str, Any],
    *,
    model_role: str,
    pair_ids: list[str] | tuple[str, ...] | None = None,
    strength: float,
    top_k: int = 3,
    epsilon: float = 0.05,
) -> tuple[torch.Tensor, dict[str, dict[str, float | int | bool | str]]]:
    """Apply the SGF MMD-energy gradient away from source references.

    For a singleton current distribution and empirical unsafe distribution,
    grad MMD^2 = (4 gamma / N) sum_j k(x, y_j) (x - y_j).
    This is the repulsive energy gradient in the SGF paper. It intentionally has
    the opposite sign to the kernel-similarity gradient in the repository's
    appendix listing, whose sign is inconsistent with the paper's MMD equation.
    """

    if strength < 0:
        raise ValueError("SGF strength must be non-negative")
    if not 0 < epsilon < 1:
        raise ValueError("SGF epsilon must lie strictly between zero and one")

    original_dtype = current_x0.dtype
    x_shape = current_x0.shape
    x = current_x0.detach().to(torch.float32).reshape(current_x0.shape[0], -1)
    source_bank, spans = _source_bank(
        current_x0, references, model_role, pair_ids=pair_ids
    )
    artifact_pair_ids = {pair_id for pair_id, _, _ in spans}
    if pair_ids is not None and set(pair_ids) != artifact_pair_ids:
        raise ValueError(
            "SGF concept-pair provenance does not match the calibrated source banks: "
            f"campaign={sorted(pair_ids)!r}, artifact={sorted(artifact_pair_ids)!r}"
        )
    refs = source_bank.reshape(source_bank.shape[0], -1)

    squared_distances = torch.cdist(x, refs, p=2).square()
    neighbour_count = max(1, min(int(top_k), refs.shape[0]))
    nearest = torch.topk(
        squared_distances, k=neighbour_count, dim=1, largest=False
    ).values
    gamma = -math.log(float(epsilon)) / max(
        float(nearest.detach().mean().item()), 1.0e-12
    )
    kernels = torch.exp(-gamma * squared_distances)

    coefficient = 4.0 * gamma / float(refs.shape[0])
    total_force = torch.zeros_like(x)
    pair_forces: dict[str, torch.Tensor] = {}
    for pair_id, start, end in spans:
        differences = x[:, None, :] - refs[None, start:end, :]
        contribution = coefficient * (
            kernels[:, start:end, None] * differences
        ).sum(dim=1)
        pair_forces[pair_id] = contribution
        total_force.add_(contribution)

    delta = float(strength) * total_force
    steered = (x + delta).reshape(x_shape).to(original_dtype)
    total_delta_norm = float(delta.float().norm().item())
    details: dict[str, dict[str, float | int | bool | str]] = {}
    for pair_id, start, end in spans:
        pair_delta = float(strength) * pair_forces[pair_id]
        details[pair_id] = {
            "delta_norm": float(pair_delta.float().norm().item()),
            "total_delta_norm": total_delta_norm,
            "gamma": float(gamma),
            "kernel_mean": float(kernels[:, start:end].float().mean().item()),
            "source_reference_count": int(end - start),
            "total_source_reference_count": int(refs.shape[0]),
            "target_references_used": False,
            "adaptation": "sgf_source_negative_guidance",
        }
    return steered, details


@torch.no_grad()
def safe_denoiser_switch_x0(
    current_x0: torch.Tensor,
    references: Mapping[str, Any],
    *,
    model_role: str,
    pair_ids: list[str] | tuple[str, ...] | None = None,
    sigma: float,
    scale: float,
) -> tuple[torch.Tensor, dict[str, dict[str, float | int | bool | str]]]:
    """Apply the Training-Free Safe Denoiser clean-target update.

    The released image implementation uses exp(-||x-y||/(2 sigma^2)) for its
    practical unsafe-reference kernel. The paper's required denoiser update is
    x_safe = x_data + eta beta(x) (x_data - E_unsafe(x)).
    """

    if sigma <= 0:
        raise ValueError("Safe Denoiser sigma must be positive")
    if scale < 0:
        raise ValueError("Safe Denoiser scale must be non-negative")

    original_dtype = current_x0.dtype
    x_shape = current_x0.shape
    x = current_x0.detach().to(torch.float32).reshape(current_x0.shape[0], -1)
    source_bank, spans = _source_bank(
        current_x0, references, model_role, pair_ids=pair_ids
    )
    artifact_pair_ids = {pair_id for pair_id, _, _ in spans}
    if pair_ids is not None and set(pair_ids) != artifact_pair_ids:
        raise ValueError(
            "Safe Denoiser concept-pair provenance does not match the calibrated source banks: "
            f"campaign={sorted(pair_ids)!r}, artifact={sorted(artifact_pair_ids)!r}"
        )
    refs = source_bank.reshape(source_bank.shape[0], -1)

    distances = torch.cdist(x, refs, p=2)
    log_kernels = -distances / (2.0 * float(sigma) ** 2)
    posterior_weights = torch.softmax(log_kernels, dim=1)
    unsafe_denoiser = posterior_weights @ refs
    log_beta = torch.logsumexp(log_kernels, dim=1) - math.log(float(refs.shape[0]))
    beta = torch.exp(log_beta)
    safe_direction = x - unsafe_denoiser
    delta = float(scale) * beta[:, None] * safe_direction
    steered = (x + delta).reshape(x_shape).to(original_dtype)

    total_delta_norm = float(delta.float().norm().item())
    details: dict[str, dict[str, float | int | bool | str]] = {}
    for pair_id, start, end in spans:
        posterior_mass = posterior_weights[:, start:end].sum(dim=1).mean()
        details[pair_id] = {
            "delta_norm": float(total_delta_norm * posterior_mass.item()),
            "total_delta_norm": total_delta_norm,
            "posterior_mass": float(posterior_mass.item()),
            "unsafe_density_beta": float(beta.float().mean().item()),
            "log_unsafe_density_beta": float(log_beta.float().mean().item()),
            "source_reference_count": int(end - start),
            "total_source_reference_count": int(refs.shape[0]),
            "target_references_used": False,
            "adaptation": "safe_denoiser_source_negative_guidance",
        }
    return steered, details
