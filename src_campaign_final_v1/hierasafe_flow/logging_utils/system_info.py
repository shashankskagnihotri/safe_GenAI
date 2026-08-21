from __future__ import annotations

import platform
from typing import Any

import torch


def collect_system_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda
        info["device_count"] = torch.cuda.device_count()
        info["devices"] = [
            {
                "index": idx,
                "name": torch.cuda.get_device_name(idx),
                "capability": torch.cuda.get_device_capability(idx),
            }
            for idx in range(torch.cuda.device_count())
        ]
    return info

