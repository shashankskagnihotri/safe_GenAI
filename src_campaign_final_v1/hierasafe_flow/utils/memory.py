from __future__ import annotations

import gc

import torch


def clear_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def cuda_memory_summary() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
        "max_allocated": int(torch.cuda.max_memory_allocated()),
        "max_reserved": int(torch.cuda.max_memory_reserved()),
    }
