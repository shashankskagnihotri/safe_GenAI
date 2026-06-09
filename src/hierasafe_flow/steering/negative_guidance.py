from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from hierasafe_flow.steering.bottleneck import BottleneckTrace, ConceptTrace
from hierasafe_flow.steering.concept_graph import compose_concept_prompt
from hierasafe_flow.steering.schedules import LambdaSchedule, step_is_enabled
from hierasafe_flow.utils.tensors import tensor_stats


@dataclass(frozen=True)
class NegativeGuidanceConfig:
    enabled: bool = True
    start_step: int = 0
    end_step: int | None = None
    lambda_schedule: LambdaSchedule = LambdaSchedule()
    negative_concept: str = (
        "explicit nudity, exposed intimate anatomy, pornographic framing, erotic pose, "
        "sexualized bed scene, bare full front body"
    )
    neutral_concept: str = "ordinary neutral non-explicit scene with context preserved"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "NegativeGuidanceConfig":
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", True)),
            start_step=int(data.get("start_step", 0)),
            end_step=data.get("end_step"),
            lambda_schedule=LambdaSchedule.from_dict(data.get("lambda_schedule")),
            negative_concept=str(data.get("negative_concept", cls.negative_concept)),
            neutral_concept=str(data.get("neutral_concept", cls.neutral_concept)),
        )


class NegativeConceptVectorGuidance:
    """Global negative-concept guidance baseline using only generator vector fields."""

    def __init__(self, config: NegativeGuidanceConfig) -> None:
        self.config = config
        self._condition_cache: dict[str, Any] = {}

    def _condition(self, adapter: Any, prompt: str) -> Any:
        cached = self._condition_cache.get(prompt)
        if cached is None:
            cached = adapter.prepare_prompt(prompt)
            self._condition_cache[prompt] = cached
        return cached

    def _predict(self, adapter: Any, latents: torch.Tensor, timestep: Any, state: Any, prompt: str) -> torch.Tensor:
        return adapter.predict_vector_field(
            latents=latents,
            timestep=timestep,
            condition=self._condition(adapter, prompt),
            state=state,
        )

    def steer_step(
        self,
        adapter: Any,
        latents: torch.Tensor,
        timestep: Any,
        state: Any,
        prompt: str,
        step_index: int,
        num_steps: int,
    ) -> tuple[torch.Tensor, BottleneckTrace]:
        v_base = self._predict(adapter, latents, timestep, state, prompt)
        if not self.config.enabled or not step_is_enabled(
            step_index,
            num_steps,
            start_step=self.config.start_step,
            end_step=self.config.end_step,
        ):
            stats = tensor_stats(v_base)
            return v_base, BottleneckTrace(
                step_index=step_index,
                timestep=_timestep_to_log_value(timestep),
                enabled=False,
                concepts=[],
                base_stats=stats,
                steered_stats=stats,
            )

        lambda_t = self.config.lambda_schedule.value(step_index, num_steps)
        negative_prompt = compose_concept_prompt(prompt, self.config.negative_concept)
        neutral_prompt = compose_concept_prompt(prompt, self.config.neutral_concept)
        v_negative = self._predict(adapter, latents, timestep, state, negative_prompt)
        v_neutral = self._predict(adapter, latents, timestep, state, neutral_prompt)
        negative_basis = v_negative - v_neutral
        guided = v_base - lambda_t * negative_basis
        return guided, BottleneckTrace(
            step_index=step_index,
            timestep=_timestep_to_log_value(timestep),
            enabled=True,
            concepts=[
                ConceptTrace(
                    concept_id="negative_guidance",
                    parent="global_negative_prompt_baseline",
                    lambda_t=float(lambda_t),
                    activation=tensor_stats(negative_basis),
                    mask={"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0},
                )
            ],
            base_stats=tensor_stats(v_base),
            steered_stats=tensor_stats(guided),
        )


def _timestep_to_log_value(timestep: Any) -> Any:
    if hasattr(timestep, "detach"):
        scalar = timestep.detach().flatten()[0].item()
        if isinstance(scalar, float):
            return float(scalar)
        return int(scalar)
    return timestep

