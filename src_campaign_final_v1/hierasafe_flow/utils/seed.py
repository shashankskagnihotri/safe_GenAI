from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_generator(seed: int | None, device: torch.device | str) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator(device=str(device))
    generator.manual_seed(seed)
    return generator

