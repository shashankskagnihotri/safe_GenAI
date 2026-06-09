from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LambdaSchedule:
    kind: str = "constant"
    max_value: float = 1.0
    min_value: float = 0.0

    @classmethod
    def from_dict(cls, data: dict | None) -> "LambdaSchedule":
        data = data or {}
        return cls(
            kind=str(data.get("kind", "constant")),
            max_value=float(data.get("max_value", data.get("value", 1.0))),
            min_value=float(data.get("min_value", 0.0)),
        )

    def value(self, step_index: int, num_steps: int) -> float:
        if num_steps <= 1:
            progress = 1.0
        else:
            progress = step_index / float(num_steps - 1)

        if self.kind == "constant":
            return self.max_value
        if self.kind == "linear":
            return self.max_value + (self.min_value - self.max_value) * progress
        if self.kind == "cosine":
            weight = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_value + (self.max_value - self.min_value) * weight
        if self.kind == "warmup_cosine":
            warmup = min(progress / 0.2, 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_value + (self.max_value - self.min_value) * warmup * cosine
        raise ValueError(
            f"Unknown lambda schedule '{self.kind}'. Valid: constant, linear, cosine, warmup_cosine."
        )


def step_is_enabled(step_index: int, num_steps: int, start_step: int = 0, end_step: int | None = None) -> bool:
    del num_steps
    if step_index < start_step:
        return False
    if end_step is not None and step_index > end_step:
        return False
    return True

