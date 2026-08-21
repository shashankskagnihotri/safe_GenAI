from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from hierasafe_flow.steering.concept_graph import (
    ConceptHierarchy,
    ConceptPair,
    compose_concept_prompt,
)
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
from hierasafe_flow.generation.conditioning_cache import StateAwareConditionCache
from hierasafe_flow.utils.tensors import tensor_stats


@dataclass(frozen=True)
class PairOverride:
    weight: float = 1.0
    start_fraction: float | None = None
    end_fraction: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PairOverride":
        data = data or {}
        return cls(
            weight=float(data.get("weight", 1.0)),
            start_fraction=data.get("start_fraction"),
            end_fraction=data.get("end_fraction"),
        )


@dataclass(frozen=True)
class BottleneckConfig:
    enabled: bool = True
    start_step: int = 0
    end_step: int | None = None
    start_fraction: float | None = None
    end_fraction: float | None = None
    step_stride: int = 1
    lambda_schedule: LambdaSchedule = LambdaSchedule()
    margin: float = 0.0
    feature_dim: int = 1
    mask: MaskConfig = MaskConfig()
    prompt_composition: str = "append"
    active_pair_ids: tuple[str, ...] | None = None
    normalize_directions: bool = False
    pair_overrides: dict[str, PairOverride] | None = None

    def __post_init__(self) -> None:
        if self.step_stride < 1:
            raise ValueError("steering.step_stride must be an integer >= 1.")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BottleneckConfig":
        data = data or {}
        mask_data = data.get("mask") or {}
        active_pair_ids = data.get("active_pair_ids", data.get("pair_ids"))
        if active_pair_ids is not None:
            if isinstance(active_pair_ids, str):
                active_pair_ids = (active_pair_ids,)
            else:
                active_pair_ids = tuple(str(item) for item in active_pair_ids)
        return cls(
            enabled=bool(data.get("enabled", True)),
            start_step=int(data.get("start_step", 0)),
            end_step=data.get("end_step"),
            start_fraction=data.get("start_fraction"),
            end_fraction=data.get("end_fraction"),
            step_stride=int(data.get("step_stride", 1)),
            lambda_schedule=LambdaSchedule.from_dict(data.get("lambda_schedule")),
            margin=float(data.get("margin", 0.0)),
            feature_dim=int(data.get("feature_dim", 1)),
            prompt_composition=str(data.get("prompt_composition", "append")),
            active_pair_ids=active_pair_ids,
            normalize_directions=bool(data.get("normalize_directions", False)),
            pair_overrides={
                str(pair_id): PairOverride.from_dict(override)
                for pair_id, override in dict(data.get("pair_overrides") or {}).items()
            },
            mask=MaskConfig(
                enabled=bool(mask_data.get("enabled", True)),
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
    shapley: dict[str, Any] | None = None


@dataclass
class BottleneckTrace:
    step_index: int
    timestep: Any
    enabled: bool
    concepts: list[ConceptTrace]
    base_stats: dict[str, float]
    steered_stats: dict[str, float]
    segment: dict[str, Any] | None = None
    condition_calls: list[dict[str, Any]] = field(default_factory=list)
    dynamic_latent_fingerprint: str | None = None
    protected_state: dict[str, Any] = field(default_factory=dict)


class HierarchicalVectorFieldBottleneck:
    def __init__(
        self,
        hierarchy: ConceptHierarchy,
        config: BottleneckConfig,
        condition_cache: StateAwareConditionCache | None = None,
    ) -> None:
        self.hierarchy = hierarchy
        self.config = config
        self.condition_cache = condition_cache or StateAwareConditionCache(
            namespace="standalone_conceptsteer"
        )
        self.active_pairs = self._resolve_active_pairs()

    def set_condition_cache(self, cache: StateAwareConditionCache) -> None:
        if not isinstance(cache, StateAwareConditionCache):
            raise TypeError("Concept steerer requires StateAwareConditionCache.")
        self.condition_cache = cache

    def _condition(
        self,
        adapter: Any,
        prompt: str,
        state: Any,
        *,
        call_role: str = "condition",
        prompt_view: str = "registered",
    ) -> Any:
        return self.condition_cache.get_or_prepare_one(
            adapter=adapter,
            prompt=prompt,
            state=state,
            prompt_view=prompt_view,
            call_role=call_role,
        )

    def _warm_conditions(
        self,
        adapter: Any,
        prompts: list[str],
        state: Any | None = None,
        call_roles: list[str] | None = None,
    ) -> None:
        if any(not isinstance(prompt, str) for prompt in prompts):
            raise TypeError("Condition warm-up prompts must be strings.")
        if state is None:
            return
        roles = call_roles or [f"warm:{index}" for index in range(len(prompts))]
        self.condition_cache.get_or_prepare_many(
            adapter=adapter,
            prompts=prompts,
            state=state,
            prompt_views="registered",
            call_roles=roles,
        )

    def _predict(
        self,
        adapter: Any,
        latents: torch.Tensor,
        timestep: Any,
        state: Any,
        prompt: str,
        *,
        call_role: str = "condition",
        prompt_view: str = "registered",
    ) -> torch.Tensor:
        return adapter.predict_vector_field(
            latents=latents,
            timestep=timestep,
            condition=self._condition(
                adapter,
                prompt,
                state,
                call_role=call_role,
                prompt_view=prompt_view,
            ),
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
        local_step_index, local_num_steps = self.schedule_coordinates(
            state, step_index, num_steps
        )
        enabled = (
            self.config.enabled
            and step_is_enabled(
                local_step_index,
                local_num_steps,
                start_step=self.config.start_step,
                end_step=self.config.end_step,
                start_fraction=self.config.start_fraction,
                end_fraction=self.config.end_fraction,
            )
            and self._stride_enabled(local_step_index)
        )
        if enabled:
            warm_roles = ["base_current", "neutral"]
            for pair in self.active_pairs:
                warm_roles.extend(
                    [f"unsafe_source:{pair.id}", f"safe_target:{pair.id}"]
                )
            self._warm_conditions(
                adapter,
                self._step_prompts(prompt),
                state,
                warm_roles,
            )

        v_base = self._predict(
            adapter,
            latents,
            timestep,
            state,
            prompt,
            call_role="base_current",
        )
        if not enabled:
            stats = tensor_stats(v_base)
            return v_base, BottleneckTrace(
                step_index=step_index,
                timestep=_timestep_to_log_value(timestep),
                enabled=False,
                concepts=[],
                base_stats=stats,
                steered_stats=stats,
            )

        lambda_t = self.config.lambda_schedule.value(local_step_index, local_num_steps)
        v_current = v_base
        traces: list[ConceptTrace] = []
        neutral_prompt = self._compose_prompt(prompt, self.hierarchy.neutral_concept)
        v_neutral = self._predict(
            adapter,
            latents,
            timestep,
            state,
            neutral_prompt,
            call_role="neutral",
        )

        for pair in self.active_pairs:
            if not self._pair_enabled(pair.id, local_step_index, local_num_steps):
                continue
            v_current, trace = self._apply_pair(
                adapter=adapter,
                latents=latents,
                timestep=timestep,
                state=state,
                prompt=prompt,
                pair=pair,
                v_current=v_current,
                v_neutral=v_neutral,
                lambda_t=self._pair_lambda(pair.id, lambda_t),
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

    def expected_pair_ids_for_step(self, step_index: int, num_steps: int) -> tuple[str, ...]:
        """Return the exact ordered ConceptSteer pair coverage for one step."""

        if not self._bottleneck_step_enabled(step_index, num_steps):
            return ()
        return tuple(
            pair.id
            for pair in self.active_pairs
            if self._pair_enabled(pair.id, step_index, num_steps)
        )

    def validate_run_trace(
        self,
        trace: list[dict[str, Any]],
        *,
        num_steps: int,
    ) -> dict[str, Any]:
        """Fail closed on incomplete or inert ConceptSteer interventions.

        Exact pair-by-step coverage is checked in hierarchy order. A particular
        step may legitimately have a zero update when its source concept is not
        locally active, but every configured active pair must be scheduled and
        must apply a non-zero finite update on at least one scheduled step. This
        validation is intended to run before media decoding or saving.
        """

        if num_steps < 1:
            raise RuntimeError(
                "ConceptSteer trace validation requires at least one denoising step."
            )
        if len(trace) != num_steps:
            raise RuntimeError(
                "ConceptSteer trace coverage mismatch: "
                f"expected {num_steps} step records, observed {len(trace)}."
            )

        aggregates = _new_pair_aggregates(self.active_pairs)
        per_segment: dict[int, dict[str, Any]] = {}
        for step_index, step in enumerate(trace):
            if not isinstance(step, dict):
                raise RuntimeError(
                    f"ConceptSteer trace step {step_index} must be a mapping, "
                    f"got {type(step).__name__}."
                )
            if step.get("step_index") != step_index:
                raise RuntimeError(
                    "ConceptSteer trace step order/index mismatch: "
                    f"position {step_index} reports step_index={step.get('step_index')!r}."
                )
            segment = _validated_segment_record(step, step_index=step_index, num_steps=num_steps)
            segment_index = int(segment["segment_index"])
            local_step_index = int(segment["local_step_index"])
            local_num_steps = int(segment["local_num_steps"])
            segment_entry = per_segment.setdefault(
                segment_index,
                {
                    "segment": {
                        key: segment[key]
                        for key in (
                            "segment_index",
                            "segment_count",
                            "local_num_steps",
                            "model_role",
                            "model_id",
                            "model_revision",
                            "condition_epoch",
                            "anchor_sha256",
                            "segment_seed",
                        )
                    },
                    "local_steps": [],
                    "per_pair": _new_pair_aggregates(self.active_pairs),
                },
            )
            if segment_entry["segment"] != {
                key: segment[key]
                for key in segment_entry["segment"]
            }:
                raise RuntimeError("ConceptSteer segment identity changed within one segment.")
            segment_entry["local_steps"].append(local_step_index)
            _validate_condition_calls(step, segment)
            expected_enabled = self._bottleneck_step_enabled(
                local_step_index, local_num_steps
            )
            if step.get("enabled") is not expected_enabled:
                raise RuntimeError(
                    f"ConceptSteer trace enabled flag mismatch at step {step_index}: "
                    f"expected {expected_enabled}, observed {step.get('enabled')!r}."
                )
            expected_pair_ids = self.expected_pair_ids_for_step(
                local_step_index, local_num_steps
            )
            concepts = step.get("concepts")
            if not isinstance(concepts, list):
                raise RuntimeError(
                    f"ConceptSteer trace concepts at step {step_index} must be a list."
                )
            observed_pair_ids = tuple(
                concept.get("concept_id") if isinstance(concept, dict) else None
                for concept in concepts
            )
            if observed_pair_ids != expected_pair_ids:
                raise RuntimeError(
                    f"ConceptSteer pair coverage mismatch at step {step_index}: "
                    f"expected {list(expected_pair_ids)}, observed {list(observed_pair_ids)}."
                )

            for pair_id in expected_pair_ids:
                aggregates[pair_id]["expected_step_count"] += 1
                segment_entry["per_pair"][pair_id]["expected_step_count"] += 1
            for concept in concepts:
                pair_id = str(concept["concept_id"])
                delta_stats = concept.get("steering_delta_stats")
                if not isinstance(delta_stats, dict):
                    raise RuntimeError(
                        f"ConceptSteer delta statistics are missing for pair '{pair_id}' "
                        f"at step {step_index}."
                    )
                required_stats = ("mean", "std", "min", "max")
                parsed_stats: dict[str, float] = {}
                for stat_name in required_stats:
                    raw_value = delta_stats.get(stat_name)
                    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                        raise RuntimeError(
                            f"ConceptSteer delta statistic '{stat_name}' for pair '{pair_id}' "
                            f"at step {step_index} must be numeric."
                        )
                    value = float(raw_value)
                    if not math.isfinite(value):
                        raise RuntimeError(
                            f"ConceptSteer delta statistic '{stat_name}' for pair '{pair_id}' "
                            f"at step {step_index} must be finite."
                        )
                    parsed_stats[stat_name] = value
                if parsed_stats["std"] < 0.0:
                    raise RuntimeError(
                        f"ConceptSteer delta std for pair '{pair_id}' at step {step_index} "
                        "must be non-negative."
                    )
                if parsed_stats["min"] > parsed_stats["max"]:
                    raise RuntimeError(
                        f"ConceptSteer delta min/max are inconsistent for pair '{pair_id}' "
                        f"at step {step_index}."
                    )
                maximum_absolute = max(
                    abs(parsed_stats["mean"]),
                    abs(parsed_stats["min"]),
                    abs(parsed_stats["max"]),
                    parsed_stats["std"],
                )
                aggregates[pair_id]["observed_step_count"] += 1
                segment_entry["per_pair"][pair_id]["observed_step_count"] += 1
                if maximum_absolute > 0.0:
                    aggregates[pair_id]["nonzero_delta_step_count"] += 1
                    segment_entry["per_pair"][pair_id]["nonzero_delta_step_count"] += 1
                aggregates[pair_id]["maximum_absolute_delta_stat"] = max(
                    aggregates[pair_id]["maximum_absolute_delta_stat"],
                    maximum_absolute,
                )
                segment_entry["per_pair"][pair_id]["maximum_absolute_delta_stat"] = max(
                    segment_entry["per_pair"][pair_id]["maximum_absolute_delta_stat"],
                    maximum_absolute,
                )

        failures: list[str] = []
        if sorted(per_segment) != list(range(len(per_segment))):
            failures.append("segment indices are not contiguous from zero")
        for segment_index, segment_entry in per_segment.items():
            expected_steps = list(range(int(segment_entry["segment"]["local_num_steps"])))
            if segment_entry["local_steps"] != expected_steps:
                failures.append(
                    f"segment {segment_index} local coverage is "
                    f"{segment_entry['local_steps']}, expected {expected_steps}"
                )
            for pair_id, aggregate in segment_entry["per_pair"].items():
                _append_pair_failures(
                    failures,
                    pair_id,
                    aggregate,
                    prefix=f"segment {segment_index} ",
                )
        for pair_id, aggregate in aggregates.items():
            _append_pair_failures(failures, pair_id, aggregate)
        if failures:
            raise RuntimeError("ConceptSteer trace validation failed: " + "; ".join(failures) + ".")

        return {
            "schema_version": 1,
            "status": "passed",
            "segment_trace_schema_version": 1,
            "num_steps": int(num_steps),
            "active_pair_ids": [pair.id for pair in self.active_pairs],
            "per_pair": aggregates,
            "per_segment": per_segment,
        }

    def _bottleneck_step_enabled(self, step_index: int, num_steps: int) -> bool:
        return (
            self.config.enabled
            and step_is_enabled(
                step_index,
                num_steps,
                start_step=self.config.start_step,
                end_step=self.config.end_step,
                start_fraction=self.config.start_fraction,
                end_fraction=self.config.end_fraction,
            )
            and self._stride_enabled(step_index)
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
        v_unsafe = self._predict(
            adapter,
            latents,
            timestep,
            state,
            unsafe_prompt,
            call_role=f"unsafe_source:{pair.id}",
        )
        v_safe = self._predict(
            adapter,
            latents,
            timestep,
            state,
            safe_prompt,
            call_role=f"safe_target:{pair.id}",
        )
        basis = compute_concept_basis(
            v_unsafe,
            v_safe,
            v_neutral,
            normalize=self.config.normalize_directions,
            eps=self.config.mask.eps,
        )
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

    def _step_prompts(self, prompt: str) -> list[str]:
        prompts = [prompt, self._compose_prompt(prompt, self.hierarchy.neutral_concept)]
        for pair in self.active_pairs:
            prompts.append(self._compose_prompt(prompt, pair.unsafe_concept))
            prompts.append(self._compose_prompt(prompt, pair.safe_sibling_concept))
        return prompts

    def _pair_enabled(self, pair_id: str, step_index: int, num_steps: int) -> bool:
        override = (self.config.pair_overrides or {}).get(pair_id)
        if override is None:
            return True
        return step_is_enabled(
            step_index,
            num_steps,
            start_fraction=override.start_fraction,
            end_fraction=override.end_fraction,
        )

    def _pair_lambda(self, pair_id: str, global_lambda: float) -> float:
        override = (self.config.pair_overrides or {}).get(pair_id)
        if override is None:
            return global_lambda
        return global_lambda * override.weight

    def _stride_enabled(self, step_index: int) -> bool:
        return step_index % self.config.step_stride == 0

    def schedule_coordinates(
        self,
        state: Any,
        global_step_index: int,
        global_num_steps: int,
    ) -> tuple[int, int]:
        context = getattr(state, "extra", {}).get("_active_denoising_step_context")
        if context is None:
            return global_step_index, global_num_steps
        return int(context.local_step_index), int(context.local_num_steps)

    def _resolve_active_pairs(self) -> tuple[ConceptPair, ...]:
        by_id = {pair.id: pair for pair in self.hierarchy.pairs}
        override_ids = set((self.config.pair_overrides or {}).keys())
        unknown_overrides = sorted(pair_id for pair_id in override_ids if pair_id not in by_id)
        if unknown_overrides:
            raise ValueError(
                "steering.pair_overrides contains unknown concept pair IDs: "
                f"{unknown_overrides}. Available IDs: {sorted(by_id)}"
            )
        if not self.config.active_pair_ids:
            return self.hierarchy.pairs
        missing = [pair_id for pair_id in self.config.active_pair_ids if pair_id not in by_id]
        if missing:
            raise ValueError(
                "steering.active_pair_ids contains unknown concept pair IDs: "
                f"{missing}. Available IDs: {sorted(by_id)}"
            )
        inactive_overrides = sorted(
            pair_id for pair_id in override_ids if pair_id not in self.config.active_pair_ids
        )
        if inactive_overrides:
            raise ValueError(
                "steering.pair_overrides contains pair IDs that are not active: "
                f"{inactive_overrides}. Active IDs: {list(self.config.active_pair_ids)}"
            )
        return tuple(by_id[pair_id] for pair_id in self.config.active_pair_ids)


def _timestep_to_log_value(timestep: Any) -> Any:
    if hasattr(timestep, "detach"):
        scalar = timestep.detach().flatten()[0].item()
        if isinstance(scalar, float):
            return float(scalar)
        return int(scalar)
    return timestep


def _new_pair_aggregates(active_pairs: tuple[ConceptPair, ...]) -> dict[str, dict[str, Any]]:
    return {
        pair.id: {
            "expected_step_count": 0,
            "observed_step_count": 0,
            "nonzero_delta_step_count": 0,
            "maximum_absolute_delta_stat": 0.0,
        }
        for pair in active_pairs
    }


def _append_pair_failures(
    failures: list[str],
    pair_id: str,
    aggregate: dict[str, Any],
    *,
    prefix: str = "",
) -> None:
    if aggregate["expected_step_count"] == 0:
        failures.append(f"{prefix}pair '{pair_id}' was not scheduled at any denoising step")
    if aggregate["observed_step_count"] != aggregate["expected_step_count"]:
        failures.append(
            f"{prefix}pair '{pair_id}' expected {aggregate['expected_step_count']} "
            f"observations but recorded {aggregate['observed_step_count']}"
        )
    if aggregate["nonzero_delta_step_count"] <= 0:
        failures.append(f"{prefix}pair '{pair_id}' applied zero delta across all steps")


def _validated_segment_record(
    step: dict[str, Any],
    *,
    step_index: int,
    num_steps: int,
) -> dict[str, Any]:
    segment = step.get("segment")
    if not isinstance(segment, dict) or segment.get("schema_version") != 1:
        raise RuntimeError(
            f"ConceptSteer step {step_index} is missing segment-trace schema 1 evidence."
        )
    required_ints = (
        "global_step_index",
        "global_num_steps",
        "segment_index",
        "segment_count",
        "local_step_index",
        "local_num_steps",
        "condition_epoch",
        "segment_seed",
    )
    for key in required_ints:
        value = segment.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(f"Segment trace field {key} must be an integer.")
    if segment["global_step_index"] != step_index or segment["global_num_steps"] != num_steps:
        raise RuntimeError("Segment trace global ordering does not match the run trace.")
    if not 0 <= segment["local_step_index"] < segment["local_num_steps"]:
        raise RuntimeError("Segment trace local denoising coordinates are invalid.")
    if not 0 <= segment["segment_index"] < segment["segment_count"]:
        raise RuntimeError("Segment trace segment coordinates are invalid.")
    for key in ("model_role", "model_id", "model_revision"):
        if not isinstance(segment.get(key), str) or not segment[key]:
            raise RuntimeError(f"Segment trace field {key} must be non-empty.")
    anchor = segment.get("anchor_sha256")
    if anchor is not None and (
        not isinstance(anchor, str)
        or len(anchor) != 64
        or any(character not in "0123456789abcdef" for character in anchor)
    ):
        raise RuntimeError("Segment trace anchor_sha256 is malformed.")
    fingerprint = step.get("dynamic_latent_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise RuntimeError("Segment trace dynamic latent fingerprint is missing.")
    return segment


def _validate_condition_calls(step: dict[str, Any], segment: dict[str, Any]) -> None:
    calls = step.get("condition_calls")
    if not isinstance(calls, list) or not calls:
        raise RuntimeError("Segment trace has no ordered condition-call evidence.")
    for call in calls:
        if not isinstance(call, dict):
            raise RuntimeError("Condition-call evidence must contain mappings.")
        if call.get("sequence_index") is None:
            raise RuntimeError("Condition-call evidence is missing its global sequence index.")
        for key in ("identity_sha256", "encoding_fingerprint", "prompt_sha256"):
            value = call.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise RuntimeError(f"Condition-call evidence field {key} is malformed.")
        identity = call.get("identity")
        if not isinstance(identity, dict):
            raise RuntimeError("Condition-call cache identity is missing.")
        for identity_key, segment_key in (
            ("model_role", "model_role"),
            ("model_id", "model_id"),
            ("model_revision", "model_revision"),
            ("condition_epoch", "condition_epoch"),
            ("segment_seed", "segment_seed"),
            ("anchor_sha256", "anchor_sha256"),
        ):
            if identity.get(identity_key) != segment.get(segment_key):
                raise RuntimeError(
                    f"Condition-call identity field {identity_key} does not match segment state."
                )
    role_to_fingerprints: dict[str, set[str]] = {}
    for call in calls:
        role_to_fingerprints.setdefault(str(call["call_role"]), set()).add(
            str(call["encoding_fingerprint"])
        )
    unsafe = {
        fingerprint
        for role, fingerprints in role_to_fingerprints.items()
        if role.startswith("unsafe_source:")
        for fingerprint in fingerprints
    }
    safe = {
        fingerprint
        for role, fingerprints in role_to_fingerprints.items()
        if role.startswith("safe_target:")
        for fingerprint in fingerprints
    }
    if unsafe and safe and unsafe == safe:
        raise RuntimeError("Unsafe/source and safe/target encodings are identical.")
