"""Extracted numerical and conditioning helpers; no campaign dependencies."""
from __future__ import annotations
import dataclasses
import hashlib
from typing import Any
import torch
from .adapters.base import PromptCondition

def tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().contiguous().cpu()
    return tensor.view(torch.uint8).numpy().tobytes()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(tensor_bytes(value)).hexdigest()


def _tree_map(value: Any, fn) -> Any:
    if torch.is_tensor(value):
        return fn(value)
    if isinstance(value, dict):
        return {key: _tree_map(item, fn) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_map(item, fn) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_map(item, fn) for item in value)
    if dataclasses.is_dataclass(value):
        fields = {field.name: _tree_map(getattr(value, field.name), fn) for field in dataclasses.fields(value)}
        return type(value)(**fields)
    return value


def _condition_to_cpu(value: PromptCondition) -> PromptCondition:
    return PromptCondition(
        prompt=value.prompt,
        data=_tree_map(value.data, lambda tensor: tensor.detach().contiguous().cpu().clone()),
    )


def _condition_to_device(value: PromptCondition, device: torch.device) -> PromptCondition:
    return PromptCondition(
        prompt=value.prompt,
        data=_tree_map(value.data, lambda tensor: tensor.to(device=device)),
    )


def tensor_stats(value: torch.Tensor) -> dict[str, Any]:
    work = value.detach().float()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "minimum": float(work.amin().cpu()),
        "maximum": float(work.amax().cpu()),
        "mean": float(work.mean().cpu()),
        "rms": float(work.square().mean().sqrt().cpu()),
        "finite": bool(torch.isfinite(work).all()),
    }
