"""MidSteer rank-one hidden-state intervention.

This adaptation follows the pinned upstream controller's neutral centering,
covariance whitening, source/target mean construction, and gated rank-one
update. A diagonal covariance is used deliberately so the same method is
feasible for the largest video transformer hidden widths.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

import torch


@dataclass
class MidSteerArtifact:
    module_name: str
    neutral_mean: list[float]
    covariance_diagonal: list[float]
    source_mean: list[float]
    target_mean: list[float]
    source_revision: str = "0f3b31e15cdda6ad0d46167e10319e896d6f1541"
    covariance_adaptation: str = "diagonal"

    @classmethod
    def load(cls, path: str | Path) -> "MidSteerArtifact":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fit_midsteer_artifact(
    *,
    module_name: str,
    neutral: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
) -> MidSteerArtifact:
    for name, value in (("neutral", neutral), ("source", source), ("target", target)):
        if value.ndim != 2:
            raise ValueError(f"{name} activations must be [samples, hidden], got {tuple(value.shape)}")
    observations = torch.cat([neutral, source, target], dim=0).float()
    variance = observations.var(dim=0, unbiased=False).clamp_min(1.0e-6)
    return MidSteerArtifact(
        module_name=module_name,
        neutral_mean=neutral.float().mean(dim=0).tolist(),
        covariance_diagonal=variance.tolist(),
        source_mean=source.float().mean(dim=0).tolist(),
        target_mean=target.float().mean(dim=0).tolist(),
    )


def midsteer_transform(hidden: torch.Tensor, artifact: MidSteerArtifact, strength: float) -> torch.Tensor:
    dtype = hidden.dtype
    device = hidden.device
    mu = torch.as_tensor(artifact.neutral_mean, device=device, dtype=torch.float32)
    var = torch.as_tensor(artifact.covariance_diagonal, device=device, dtype=torch.float32).clamp_min(1.0e-6)
    source = torch.as_tensor(artifact.source_mean, device=device, dtype=torch.float32)
    target = torch.as_tensor(artifact.target_mean, device=device, dtype=torch.float32)
    if hidden.shape[-1] != mu.numel():
        raise ValueError(f"MidSteer hidden width {hidden.shape[-1]} != calibrated width {mu.numel()}")
    sqrt_var = var.sqrt()
    source_white = (source - mu) / sqrt_var
    target_white = (target - mu) / sqrt_var
    left = sqrt_var * (source_white - target_white)
    right = source_white / source_white.square().sum().clamp_min(1.0e-8) / sqrt_var
    work = hidden.float()
    gate = torch.relu(torch.einsum("...d,d->...", work - mu, right))
    transformed = work - float(strength) * gate.unsqueeze(-1) * left
    return transformed.to(dtype=dtype)


def _replace_first_tensor(output: Any, transform: Any) -> Any:
    if torch.is_tensor(output):
        return transform(output)
    if isinstance(output, tuple):
        values = list(output)
        for index, value in enumerate(values):
            if torch.is_tensor(value):
                values[index] = transform(value)
                return tuple(values)
        return output
    if isinstance(output, list):
        values = list(output)
        for index, value in enumerate(values):
            if torch.is_tensor(value):
                values[index] = transform(value)
                return values
        return output
    if hasattr(output, "sample") and torch.is_tensor(output.sample):
        output.sample = transform(output.sample)
    return output


def resolve_transformer_root(adapter: Any) -> torch.nn.Module:
    owners = [adapter, getattr(adapter, "pipeline", None), getattr(adapter, "pipe", None)]
    for owner in owners:
        if owner is None:
            continue
        for name in ("transformer", "unet", "model", "network", "generator", "denoiser"):
            value = getattr(owner, name, None)
            if isinstance(value, torch.nn.Module):
                return value
            if value is not None:
                for nested_name in ("transformer", "unet", "model", "network", "denoiser"):
                    nested = getattr(value, nested_name, None)
                    if isinstance(nested, torch.nn.Module):
                        return nested
    raise RuntimeError("Could not resolve a transformer root from the adapter")


def resolve_module(root: torch.nn.Module, module_name: str) -> torch.nn.Module:
    modules = dict(root.named_modules())
    if module_name not in modules:
        raise KeyError(f"Calibrated MidSteer module {module_name!r} is absent")
    return modules[module_name]


def choose_attention_module(root: torch.nn.Module) -> str:
    candidates = [
        name
        for name, module in root.named_modules()
        if name and "attention" in module.__class__.__name__.lower()
    ]
    if not candidates:
        candidates = [name for name, _ in root.named_modules() if name and (name.endswith("attn") or name.endswith("attn1"))]
    if not candidates:
        raise RuntimeError("No attention module is available for MidSteer calibration")
    return candidates[len(candidates) // 2]


class MidSteerHook(AbstractContextManager["MidSteerHook"]):
    def __init__(self, root: torch.nn.Module, artifact: MidSteerArtifact, strength: float) -> None:
        self.module = resolve_module(root, artifact.module_name)
        self.artifact = artifact
        self.strength = strength
        self.handle: Any | None = None

    def __enter__(self) -> "MidSteerHook":
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            return _replace_first_tensor(
                output,
                lambda hidden: midsteer_transform(hidden, self.artifact, self.strength),
            )

        self.handle = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
