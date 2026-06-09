from __future__ import annotations

import torch

from hierasafe_flow.steering.local_masks import MaskConfig, activation_to_mask
from hierasafe_flow.steering.vector_fields import (
    apply_vector_field_bottleneck,
    compute_concept_basis,
    local_unsafe_activation,
)


def test_local_activation_prefers_unsafe_alignment() -> None:
    v_base = torch.tensor([[[[1.0]], [[0.0]]]])
    b_unsafe = torch.tensor([[[[1.0]], [[0.0]]]])
    b_safe = torch.tensor([[[[0.0]], [[1.0]]]])
    activation = local_unsafe_activation(v_base, b_unsafe, b_safe, feature_dim=1, margin=0.0)
    assert activation.shape == (1, 1, 1, 1)
    assert torch.allclose(activation, torch.ones_like(activation))


def test_bottleneck_moves_toward_safe_basis() -> None:
    neutral = torch.zeros(1, 2, 1, 1)
    unsafe_pred = torch.tensor([[[[1.0]], [[0.0]]]])
    safe_pred = torch.tensor([[[[0.0]], [[1.0]]]])
    basis = compute_concept_basis(unsafe_pred, safe_pred, neutral)
    v_base = unsafe_pred.clone()
    activation = local_unsafe_activation(v_base, basis.unsafe, basis.safe, feature_dim=1)
    mask = activation_to_mask(activation, MaskConfig(mode="identity"))
    steered = apply_vector_field_bottleneck(v_base, basis.unsafe, basis.safe, mask, lambda_t=1.0)
    assert torch.allclose(steered, safe_pred)

