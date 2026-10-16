"""Fail closed when an allocated GPU pair cannot transfer tensors exactly."""
from __future__ import annotations

import os
from typing import Any

import torch


def require_exact_gpu_transport() -> dict[str, Any]:
    """Exercise direct copies only; never substitute host-staged inference.

    Deterministic CPU patterns avoid the architecture-dependent CUDA RNG and
    exactly represent the same values in FP32 and BF16. Every ordered pair is
    checked at two sizes, including the peer directions used by dispatch hooks.
    """
    count = torch.cuda.device_count()
    if count < 2:
        raise RuntimeError("The distributed placement transport guard requires at least two GPUs")
    report: dict[str, Any] = {
        "status": "PASS",
        "torch_version": str(torch.__version__),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(count)],
        "host_roundtrips_checked": 0,
        "ordered_peer_copies_checked": 0,
        "inference_host_staging": False,
    }
    for dtype in (torch.float32, torch.bfloat16):
        for size in (32768, 1048576):
            expected = ((torch.arange(size, dtype=torch.int64) % 251 - 125).float() / 8).to(dtype)
            expected_bytes = expected.view(torch.uint8)
            copies = [expected.to(torch.device("cuda", i)) for i in range(count)]
            for i, value in enumerate(copies):
                torch.cuda.synchronize(i)
                if not torch.equal(value.cpu().view(torch.uint8), expected_bytes):
                    raise RuntimeError(f"GPU transport corruption in CPU/CUDA roundtrip: gpu={i}, dtype={dtype}, size={size}")
                report["host_roundtrips_checked"] += 1
            for source in range(count):
                for target in range(count):
                    if source == target:
                        continue
                    received = copies[source].to(torch.device("cuda", target), non_blocking=False)
                    torch.cuda.synchronize(source)
                    torch.cuda.synchronize(target)
                    if not torch.equal(received.cpu().view(torch.uint8), expected_bytes):
                        raise RuntimeError(
                            f"GPU peer transport corruption before model loading: {source}->{target}, "
                            f"dtype={dtype}, size={size}. No CPU-staged fallback is permitted."
                        )
                    report["ordered_peer_copies_checked"] += 1
                    del received
            del copies, expected, expected_bytes, value
    torch.cuda.empty_cache()
    return report
