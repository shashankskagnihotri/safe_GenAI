from __future__ import annotations

from dataclasses import dataclass

import torch

from hierasafe_flow.utils.tensors import cosine_similarity_along, require_same_shape


@dataclass(frozen=True)
class ConceptBasis:
    unsafe: torch.Tensor
    safe: torch.Tensor
    neutral_prediction: torch.Tensor
    unsafe_prediction: torch.Tensor
    safe_prediction: torch.Tensor


def compute_concept_basis(
    unsafe_prediction: torch.Tensor,
    safe_prediction: torch.Tensor,
    neutral_prediction: torch.Tensor,
) -> ConceptBasis:
    require_same_shape(unsafe_prediction, safe_prediction, neutral_prediction)
    return ConceptBasis(
        unsafe=unsafe_prediction - neutral_prediction,
        safe=safe_prediction - neutral_prediction,
        neutral_prediction=neutral_prediction,
        unsafe_prediction=unsafe_prediction,
        safe_prediction=safe_prediction,
    )


def local_unsafe_activation(
    v_base: torch.Tensor,
    b_unsafe: torch.Tensor,
    b_safe: torch.Tensor,
    feature_dim: int = 1,
    margin: float = 0.0,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    require_same_shape(v_base, b_unsafe, b_safe)
    unsafe_cos = cosine_similarity_along(v_base, b_unsafe, dim=feature_dim, eps=eps)
    safe_cos = cosine_similarity_along(v_base, b_safe, dim=feature_dim, eps=eps)
    return torch.relu(unsafe_cos - safe_cos + margin)


def apply_vector_field_bottleneck(
    v_base: torch.Tensor,
    b_unsafe: torch.Tensor,
    b_safe: torch.Tensor,
    mask: torch.Tensor,
    lambda_t: float,
) -> torch.Tensor:
    require_same_shape(v_base, b_unsafe, b_safe)
    if mask.shape != v_base.shape:
        mask = torch.broadcast_to(mask, v_base.shape)
    return v_base + lambda_t * mask.to(dtype=v_base.dtype) * (b_safe - b_unsafe)

