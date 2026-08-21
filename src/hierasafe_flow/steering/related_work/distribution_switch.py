"""Full-reference SGF and Safe Denoiser concept-switch adaptations."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import math

import torch


SGF_REVISION = "4bdd287475672c608eadfe32d466cf62b3a527fe"
SAFE_DENOISER_REVISION = "223415b2739de049969c8188e8c6c331ae3531b3"


class SwitchReferenceArtifact:
    def __init__(self, payload: dict[str, Any]) -> None:
        metadata = payload.get("metadata")
        references = payload.get("references")
        if not isinstance(metadata, dict) or not isinstance(references, dict) or not references:
            raise ValueError("Switch-reference artifact is malformed")
        if metadata.get("representation") != "canonical_predicted_x0":
            raise ValueError("Switch references are not canonical predicted-x0 samples")
        self.metadata = metadata
        self.references = references
        self._device_cache: dict[tuple[str, str, str, str], torch.Tensor] = {}

    @classmethod
    def load(cls, path: str | Path) -> "SwitchReferenceArtifact":
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        return cls(payload)

    @staticmethod
    def save(
        path: str | Path,
        *,
        metadata: dict[str, Any],
        references: dict[str, dict[str, dict[str, torch.Tensor]]],
    ) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "metadata": {
                    **metadata,
                    "representation": "canonical_predicted_x0",
                    "sgf_revision": SGF_REVISION,
                    "safe_denoiser_revision": SAFE_DENOISER_REVISION,
                },
                "references": references,
            },
            target,
        )

    def get(
        self,
        model_role: str,
        pair_id: str,
        side: str,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        role = model_role if model_role in self.references else "default"
        try:
            value = self.references[role][pair_id][side]
        except KeyError as exc:
            raise KeyError(
                f"Missing switch references role={model_role!r}, pair={pair_id!r}, side={side!r}"
            ) from exc
        if not isinstance(value, torch.Tensor) or value.ndim < 2 or value.shape[0] < 1:
            raise ValueError("Switch references must contain a non-empty sample dimension")
        key = (role, pair_id, side, str(device))
        if key not in self._device_cache:
            self._device_cache[key] = value.to(device=device, dtype=torch.float32)
        return self._device_cache[key]


def _flatten_samples(value: torch.Tensor) -> torch.Tensor:
    return value.to(dtype=torch.float32).reshape(value.shape[0], -1)


def _validate_shape(current: torch.Tensor, references: torch.Tensor, label: str) -> None:
    if tuple(references.shape[1:]) != tuple(current.shape[1:]):
        raise ValueError(
            f"{label} reference shape {tuple(references.shape[1:])} does not match "
            f"current x0 {tuple(current.shape[1:])}"
        )


def _sgf_gradient(
    current: torch.Tensor,
    references: torch.Tensor,
    *,
    top_k: int,
    epsilon: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    _validate_shape(current, references, "SGF")
    x = _flatten_samples(current)
    refs = _flatten_samples(references)
    difference = x[:, None, :] - refs[None, :, :]
    distances_sq = difference.square().sum(dim=-1)
    k = min(max(int(top_k), 1), int(refs.shape[0]))
    nearest = torch.topk(distances_sq, k=k, dim=1, largest=False).values
    mean_neighbor_distance_sq = nearest.mean().clamp_min(torch.finfo(torch.float32).eps)
    gamma = -math.log(float(epsilon)) / float(mean_neighbor_distance_sq.item())
    kernels = torch.exp(-gamma * distances_sq)
    gradient = -2.0 * gamma * (
        kernels[..., None] * difference
    ).sum(dim=1)
    return gradient.reshape_as(current), {
        "gamma": gamma,
        "mean_neighbor_distance_sq": float(mean_neighbor_distance_sq.item()),
        "kernel_max": float(kernels.max().item()),
        "kernel_mean": float(kernels.mean().item()),
        "gradient_norm": float(gradient.norm().item()),
    }


def sgf_switch_x0(
    current_x0: torch.Tensor,
    artifact: SwitchReferenceArtifact,
    *,
    model_role: str,
    pair_ids: list[str],
    strength: float,
    top_k: int = 3,
    epsilon: float = 0.05,
) -> tuple[torch.Tensor, dict[str, dict[str, float]]]:
    """Apply SGF source repulsion plus the preregistered target attraction."""

    if not 0.0 < epsilon < 1.0:
        raise ValueError("SGF epsilon must be in (0,1)")
    result = current_x0.to(dtype=torch.float32)
    pair_details: dict[str, dict[str, float]] = {}
    for pair_id in pair_ids:
        source = artifact.get(model_role, pair_id, "source", device=result.device)
        target = artifact.get(model_role, pair_id, "target", device=result.device)
        source_gradient, source_details = _sgf_gradient(
            result, source, top_k=top_k, epsilon=epsilon
        )
        target_gradient, target_details = _sgf_gradient(
            result, target, top_k=top_k, epsilon=epsilon
        )
        # The RBF gradient points toward its reference distribution.  Descending
        # the source term repels; ascending the target term completes the switch.
        delta = float(strength) * (target_gradient - source_gradient)
        if not torch.isfinite(delta).all() or float(delta.norm().item()) <= 0.0:
            raise RuntimeError(f"SGF pair {pair_id} produced a zero/non-finite update")
        result = result + delta
        pair_details[pair_id] = {
            "delta_norm": float(delta.norm().item()),
            "source_gamma": source_details["gamma"],
            "target_gamma": target_details["gamma"],
            "source_kernel_max": source_details["kernel_max"],
            "target_kernel_max": target_details["kernel_max"],
            "source_gradient_norm": source_details["gradient_norm"],
            "target_gradient_norm": target_details["gradient_norm"],
        }
    return result.to(dtype=current_x0.dtype), pair_details


def _normalized_kde_denoiser(
    current: torch.Tensor,
    references: torch.Tensor,
    *,
    sigma: float,
) -> tuple[torch.Tensor, float]:
    _validate_shape(current, references, "Safe Denoiser")
    x = _flatten_samples(current)
    refs = _flatten_samples(references)
    x_norm = x.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
    ref_norm = refs.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
    x_unit = x / x_norm
    refs_unit = refs / ref_norm
    distances_sq = torch.cdist(x_unit, refs_unit).square()
    log_weights = -distances_sq / (2.0 * float(sigma) ** 2)
    weights = torch.softmax(log_weights, dim=1)
    denoiser = weights @ refs_unit
    density = float(torch.exp(log_weights).mean().item())
    return denoiser, density


def safe_denoiser_switch_x0(
    current_x0: torch.Tensor,
    artifact: SwitchReferenceArtifact,
    *,
    model_role: str,
    pair_ids: list[str],
    sigma: float,
    scale: float,
) -> tuple[torch.Tensor, dict[str, dict[str, float]]]:
    """Subtract the unsafe KDE denoiser and add the target KDE counterpart."""

    if sigma <= 0.0 or scale <= 0.0:
        raise ValueError("Safe Denoiser sigma and scale must be positive")
    result = current_x0.to(dtype=torch.float32)
    pair_details: dict[str, dict[str, float]] = {}
    for pair_id in pair_ids:
        source = artifact.get(model_role, pair_id, "source", device=result.device)
        target = artifact.get(model_role, pair_id, "target", device=result.device)
        source_denoiser, source_density = _normalized_kde_denoiser(
            result, source, sigma=sigma
        )
        target_denoiser, target_density = _normalized_kde_denoiser(
            result, target, sigma=sigma
        )
        flat = _flatten_samples(result)
        magnitude = flat.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
        unsafe_likelihood = source_density / (
            source_density + target_density + 1.0e-12
        )
        beta = float(scale) * unsafe_likelihood
        unit_delta = beta * (target_denoiser - source_denoiser)
        delta = (magnitude * unit_delta).reshape_as(result)
        if not torch.isfinite(delta).all() or float(delta.norm().item()) <= 0.0:
            raise RuntimeError(
                f"Safe Denoiser pair {pair_id} produced a zero/non-finite update"
            )
        result = result + delta
        pair_details[pair_id] = {
            "delta_norm": float(delta.norm().item()),
            "source_density": source_density,
            "target_density": target_density,
            "unsafe_likelihood": unsafe_likelihood,
            "beta": beta,
            "sigma": float(sigma),
        }
    return result.to(dtype=current_x0.dtype), pair_details
