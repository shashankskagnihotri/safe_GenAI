from __future__ import annotations

from pathlib import Path
from typing import Any


class TensorBoardLogger:
    def __init__(self, log_dir: str | Path | None, enabled: bool = True) -> None:
        self.enabled = enabled and log_dir is not None
        self.writer: Any | None = None
        if self.enabled:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=str(log_dir))

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def add_scalars(self, prefix: str, values: dict[str, float], step: int) -> None:
        for key, value in values.items():
            self.add_scalar(f"{prefix}/{key}", value, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()

    def __enter__(self) -> "TensorBoardLogger":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

