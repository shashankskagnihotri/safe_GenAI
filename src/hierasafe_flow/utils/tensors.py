from __future__ import annotations

import torch


def require_same_shape(*tensors: torch.Tensor) -> None:
    if not tensors:
        return
    shape = tensors[0].shape
    for tensor in tensors[1:]:
        if tensor.shape != shape:
            raise ValueError(f"Tensor shape mismatch: expected {shape}, got {tensor.shape}")


def normalize_along(tensor: torch.Tensor, dim: int, eps: float = 1.0e-6) -> torch.Tensor:
    norm = torch.linalg.vector_norm(tensor.float(), dim=dim, keepdim=True).clamp_min(eps)
    return tensor / norm.to(dtype=tensor.dtype)


def cosine_similarity_along(
    left: torch.Tensor,
    right: torch.Tensor,
    dim: int,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    require_same_shape(left, right)
    left_f = left.float()
    right_f = right.float()
    numerator = (left_f * right_f).sum(dim=dim, keepdim=True)
    left_norm = torch.linalg.vector_norm(left_f, dim=dim, keepdim=True)
    right_norm = torch.linalg.vector_norm(right_f, dim=dim, keepdim=True)
    denom = (left_norm * right_norm).clamp_min(eps)
    return (numerator / denom).to(dtype=left.dtype)


def tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    data = tensor.detach().float()
    return {
        "mean": float(data.mean().item()),
        "std": float(data.std(unbiased=False).item()),
        "min": float(data.min().item()),
        "max": float(data.max().item()),
    }

