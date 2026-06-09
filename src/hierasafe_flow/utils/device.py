from __future__ import annotations

import torch


def resolve_device(requested: str | None = None) -> torch.device:
    if requested in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def resolve_dtype(name: str | None) -> torch.dtype:
    if name in (None, "auto"):
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = str(name).lower()
    if key not in mapping:
        raise ValueError(f"Unsupported torch dtype '{name}'. Valid: {sorted(mapping)}")
    return mapping[key]


def configure_cuda(allow_tf32: bool = True) -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

