from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from hierasafe_flow.steering.concept_graph import ConceptHierarchy, ConceptPair, compose_concept_prompt
from hierasafe_flow.steering.local_masks import (
    MaskConfig,
    activation_to_mask,
    broadcast_mask_to_vector_field,
)
from hierasafe_flow.steering.schedules import LambdaSchedule, step_is_enabled
from hierasafe_flow.steering.vector_fields import (
    apply_vector_field_bottleneck,
    compute_concept_basis,
    local_unsafe_activation,
)
from hierasafe_flow.utils.tensors import tensor_stats


@dataclass(frozen=True)
class BottleneckConfig:
    enabled: bool = True
    start_step: int = 0
    end_step: int | None = None
    lambda_schedule: LambdaSchedule = LambdaSchedule()
    margin: float = 0.0
    feature_dim: int = 1
    mask: MaskConfig = MaskConfig()
    prompt_composition: str = "append"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BottleneckConfig":
        data = data or {}
        mask_data = data.get("mask") or {}
        return cls(
            enabled=bool(data.get("enabled", True)),
            start_step=int(data.get("start_step", 0)),
            end_step=data.get("end_step"),
            lambda_schedule=LambdaSchedule.from_dict(data.get("lambda_schedule")),
            margin=float(data.get("margin", 0.0)),
            feature_dim=int(data.get("feature_dim", 1)),
            prompt_composition=str(data.get("prompt_composition", "append")),
            mask=MaskConfig(
                mode=str(mask_data.get("mode", "max_normalized")),
                threshold=float(mask_data.get("threshold", 0.05)),
                percentile=float(mask_data.get("percentile", 0.85)),
                eps=float(mask_data.get("eps", 1.0e-6)),
            ),
        )


@dataclass
class ConceptTrace:
    concept_id: str
    parent: str
    lambda_t: float
    activation: dict[str, float]
    mask: dict[str, float]
    unsafe_concept: str = ""
    target_concept: str = ""
    steering_delta_stats: dict[str, float] | None = None
    unsafe_direction_stats: dict[str, float] | None = None
    safe_direction_stats: dict[str, float] | None = None


@dataclass
class BottleneckTrace:
    step_index: int
    timestep: Any
    enabled: bool
    concepts: list[ConceptTrace]
    base_stats: dict[str, float]
    steered_stats: dict[str, float]


class HierarchicalVectorFieldBottleneck:
    def __init__(self, hierarchy: ConceptHierarchy, config: BottleneckConfig) -> None:
        self.hierarchy = hierarchy
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
        v_current = v_base
        traces: list[ConceptTrace] = []
        neutral_prompt = self._compose_prompt(prompt, self.hierarchy.neutral_concept)
        v_neutral = self._predict(adapter, latents, timestep, state, neutral_prompt)

        for pair in self.hierarchy.pairs:
            v_current, trace = self._apply_pair(
                adapter=adapter,
                latents=latents,
                timestep=timestep,
                state=state,
                prompt=prompt,
                pair=pair,
                v_current=v_current,
                v_neutral=v_neutral,
                lambda_t=lambda_t,
            )
            traces.append(trace)

        return v_current, BottleneckTrace(
            step_index=step_index,
            timestep=_timestep_to_log_value(timestep),
            enabled=True,
            concepts=traces,
            base_stats=tensor_stats(v_base),
            steered_stats=tensor_stats(v_current),
        )

    def _apply_pair(
        self,
        adapter: Any,
        latents: torch.Tensor,
        timestep: Any,
        state: Any,
        prompt: str,
        pair: ConceptPair,
        v_current: torch.Tensor,
        v_neutral: torch.Tensor,
        lambda_t: float,
    ) -> tuple[torch.Tensor, ConceptTrace]:
        unsafe_prompt = self._compose_prompt(prompt, pair.unsafe_concept)
        safe_prompt = self._compose_prompt(prompt, pair.safe_sibling_concept)
        v_unsafe = self._predict(adapter, latents, timestep, state, unsafe_prompt)
        v_safe = self._predict(adapter, latents, timestep, state, safe_prompt)
        basis = compute_concept_basis(v_unsafe, v_safe, v_neutral)
        activation = local_unsafe_activation(
            v_current,
            basis.unsafe,
            basis.safe,
            feature_dim=self.config.feature_dim,
            margin=self.config.margin,
            eps=self.config.mask.eps,
        )
        mask = activation_to_mask(activation, self.config.mask)
        mask = broadcast_mask_to_vector_field(mask, v_current)
        steered = apply_vector_field_bottleneck(
            v_current,
            basis.unsafe,
            basis.safe,
            mask,
            lambda_t=lambda_t,
        )
        steering_delta = steered - v_current
        return steered, ConceptTrace(
            concept_id=pair.id,
            parent=pair.parent,
            lambda_t=float(lambda_t),
            activation=tensor_stats(activation),
            mask=tensor_stats(mask),
            unsafe_concept=pair.unsafe_concept,
            target_concept=pair.safe_sibling_concept,
            steering_delta_stats=tensor_stats(steering_delta),
            unsafe_direction_stats=tensor_stats(basis.unsafe),
            safe_direction_stats=tensor_stats(basis.safe),
        )

    def _compose_prompt(self, prompt: str, concept: str) -> str:
        mode = self.config.prompt_composition
        if mode == "append":
            return compose_concept_prompt(prompt, concept)
        if mode == "concept_only":
            return compose_concept_prompt("", concept)
        raise ValueError(
            f"Unknown steering.prompt_composition '{mode}'. Valid modes: append, concept_only."
        )


def _timestep_to_log_value(timestep: Any) -> Any:
    if hasattr(timestep, "detach"):
        scalar = timestep.detach().flatten()[0].item()
        if isinstance(scalar, float):
            return float(scalar)
        return int(scalar)
    return timestep
