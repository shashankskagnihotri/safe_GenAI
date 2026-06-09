from __future__ import annotations

import torch

from hierasafe_flow.steering.local_masks import (
    MaskConfig,
    activation_to_mask,
    broadcast_mask_to_vector_field,
)


def test_max_normalized_mask_thresholds_small_values() -> None:
    activation = torch.tensor([[[[0.1, 0.2], [0.0, 0.4]]]])
    mask = activation_to_mask(activation, MaskConfig(mode="max_normalized", threshold=0.3))
    assert mask.max().item() == 1.0
    assert mask[0, 0, 1, 0].item() == 0.0


def test_broadcast_mask_to_vector_field() -> None:
    mask = torch.ones(1, 1, 2, 2)
    vector = torch.zeros(1, 4, 2, 2)
    result = broadcast_mask_to_vector_field(mask, vector)
    assert result.shape == vector.shape

