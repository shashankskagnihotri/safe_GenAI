from __future__ import annotations

import hashlib
import inspect
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import torch


@dataclass
class PromptCondition:
    prompt: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterState:
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulerStepResult:
    latents: torch.Tensor
    state: AdapterState


@dataclass(frozen=True)
class AdapterCapabilities:
    task_type: str
    pipeline_class_name: str | None
    exposes_latents: bool
    exposes_timesteps: bool
    exposes_scheduler_step: bool
    exposes_vector_field: bool


class FrozenGeneratorAdapter(ABC):
    adapter_name = "base"
    task_type = "unknown"
    pipeline_class_name: str | None = None

    def __init__(
        self,
        model_id: str,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = torch.device(device)
        self.dtype = dtype
        self.config = dict(config or {})
        self.loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load and freeze model components."""

    @abstractmethod
    def prepare_prompt(self, prompt: str) -> PromptCondition:
        """Encode or otherwise prepare a prompt for vector-field prediction."""

    @abstractmethod
    def prepare_initial_latents(
        self,
        prompt: str,
        batch_size: int,
        generator: torch.Generator | None,
        **generation_kwargs: Any,
    ) -> tuple[torch.Tensor, AdapterState]:
        """Create initial latent state without invoking a full pipeline fallback."""

    @abstractmethod
    def set_timesteps(
        self,
        num_inference_steps: int,
        latents: torch.Tensor | None = None,
        state: AdapterState | None = None,
    ) -> list[Any]:
        """Prepare scheduler timesteps and return them in denoising order."""

    @abstractmethod
    def predict_vector_field(
        self,
        latents: torch.Tensor,
        timestep: Any,
        condition: PromptCondition,
        state: AdapterState,
    ) -> torch.Tensor:
        """Return velocity/noise prediction v_theta(z_t, t, condition)."""

    @abstractmethod
    def scheduler_step(
        self,
        model_prediction: torch.Tensor,
        timestep: Any,
        latents: torch.Tensor,
        state: AdapterState,
        generator: torch.Generator | None = None,
    ) -> SchedulerStepResult:
        """Advance the scheduler using the supplied steered model prediction."""

    @abstractmethod
    def decode_latents(self, latents: torch.Tensor, state: AdapterState) -> Any:
        """Decode final latents into media or return tensors for dummy adapters."""

    def inspect(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter_name,
            "model_id": self.model_id,
            "task_type": self.task_type,
            "pipeline_class_name": self.pipeline_class_name,
            "loaded": self.loaded,
            "capabilities": self.capabilities().__dict__,
        }

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            task_type=self.task_type,
            pipeline_class_name=self.pipeline_class_name,
            exposes_latents=True,
            exposes_timesteps=True,
            exposes_scheduler_step=True,
            exposes_vector_field=True,
        )

    def _require_loaded(self) -> None:
        if not self.loaded:
            raise RuntimeError(f"Adapter '{self.adapter_name}' has not been loaded.")


class DummyVectorFieldAdapter(FrozenGeneratorAdapter):
    """Small deterministic adapter for unit tests and full runner smoke checks."""

    adapter_name = "dummy"
    task_type = "dummy"
    pipeline_class_name = None

    def load(self) -> None:
        self.loaded = True

    def prepare_prompt(self, prompt: str) -> PromptCondition:
        scalar = _stable_prompt_scalar(prompt)
        return PromptCondition(prompt=prompt, data={"scalar": scalar})

    def prepare_initial_latents(
        self,
        prompt: str,
        batch_size: int,
        generator: torch.Generator | None,
        **generation_kwargs: Any,
    ) -> tuple[torch.Tensor, AdapterState]:
        self._require_loaded()
        del prompt
        task = generation_kwargs.get("task", "text_to_image")
        height = int(generation_kwargs.get("height", 32))
        width = int(generation_kwargs.get("width", 32))
        frames = int(generation_kwargs.get("num_frames", 5))
        latent_h = max(height // 8, 4)
        latent_w = max(width // 8, 4)
        if task == "text_to_video":
            shape = (batch_size, 4, frames, latent_h, latent_w)
        else:
            shape = (batch_size, 4, latent_h, latent_w)
        latents = torch.randn(shape, generator=generator, device=self.device, dtype=self.dtype)
        return latents, AdapterState(extra={"task": task, "step_size": 1.0})

    def set_timesteps(
        self,
        num_inference_steps: int,
        latents: torch.Tensor | None = None,
        state: AdapterState | None = None,
    ) -> list[torch.Tensor]:
        self._require_loaded()
        del latents, state
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive.")
        return list(torch.linspace(1.0, 0.0, steps=num_inference_steps, device=self.device))

    def predict_vector_field(
        self,
        latents: torch.Tensor,
        timestep: Any,
        condition: PromptCondition,
        state: AdapterState,
    ) -> torch.Tensor:
        self._require_loaded()
        del state
        time_value = float(timestep.detach().flatten()[0].item()) if hasattr(timestep, "detach") else float(timestep)
        scalar = float(condition.data["scalar"])
        pattern = _checker_pattern(latents)
        return (-0.20 * latents + 0.08 * scalar * pattern + 0.02 * time_value).to(dtype=latents.dtype)

    def scheduler_step(
        self,
        model_prediction: torch.Tensor,
        timestep: Any,
        latents: torch.Tensor,
        state: AdapterState,
        generator: torch.Generator | None = None,
    ) -> SchedulerStepResult:
        self._require_loaded()
        del timestep, generator
        step_size = float(state.extra.get("step_size", 1.0)) / max(int(state.extra.get("num_steps", 1)), 1)
        return SchedulerStepResult(latents=latents + step_size * model_prediction, state=state)

    def decode_latents(self, latents: torch.Tensor, state: AdapterState) -> torch.Tensor:
        self._require_loaded()
        del state
        return latents.detach().float().cpu()


class DiffusersFrozenAdapter(FrozenGeneratorAdapter):
    """Shared diffusers plumbing without using pipeline __call__ as a generation fallback."""

    required_components: tuple[str, ...] = ("transformer", "scheduler", "vae")
    encode_prompt_output_names: tuple[str, ...] = ("prompt_embeds",)

    def __init__(
        self,
        model_id: str,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(model_id=model_id, device=device, dtype=dtype, config=config)
        self.pipeline: Any | None = None
        self.timesteps: list[Any] = []

    def load(self) -> None:
        pipeline_cls = self._resolve_pipeline_class()
        kwargs = dict(self.config.get("load_kwargs", {}))
        if self.config.get("revision") is not None:
            kwargs["revision"] = self.config["revision"]
        if self.config.get("variant") is not None:
            kwargs["variant"] = self.config["variant"]
        kwargs.setdefault("torch_dtype", self.dtype)
        kwargs.setdefault("local_files_only", bool(self.config.get("local_files_only", False)))
        if os.environ.get("HF_TOKEN") and "token" not in kwargs:
            kwargs["token"] = os.environ["HF_TOKEN"]
        kwargs.setdefault("low_cpu_mem_usage", True)
        self.pipeline = pipeline_cls.from_pretrained(self.model_id, **kwargs)
        cpu_offload = self.config.get("cpu_offload", False)
        if cpu_offload == "sequential" and hasattr(self.pipeline, "enable_sequential_cpu_offload"):
            self.pipeline.enable_sequential_cpu_offload(device=self.device)
        elif bool(cpu_offload) and hasattr(self.pipeline, "enable_model_cpu_offload"):
            self.pipeline.enable_model_cpu_offload(device=self.device)
        elif hasattr(self.pipeline, "to"):
            self.pipeline.to(self.device)
        self._freeze_pipeline()
        self._validate_components()
        self.loaded = True

    def _resolve_pipeline_class(self) -> type:
        if not self.pipeline_class_name:
            raise NotImplementedError(
                f"Adapter '{self.adapter_name}' does not declare a diffusers pipeline class."
            )
        try:
            import diffusers
        except ModuleNotFoundError as exc:
            raise NotImplementedError(
                f"Adapter '{self.adapter_name}' requires diffusers with pipeline "
                f"'{self.pipeline_class_name}', but diffusers is not installed in this environment."
            ) from exc
        if not hasattr(diffusers, self.pipeline_class_name):
            raise NotImplementedError(
                f"diffusers is installed but does not expose '{self.pipeline_class_name}'. "
                f"Install a diffusers version that supports model '{self.model_id}' or update the adapter mapping."
            )
        return getattr(diffusers, self.pipeline_class_name)

    def _freeze_pipeline(self) -> None:
        assert self.pipeline is not None
        for component_name in self.required_components:
            component = getattr(self.pipeline, component_name, None)
            if component is not None and hasattr(component, "parameters"):
                component.eval()
                for parameter in component.parameters():
                    parameter.requires_grad_(False)

    def _validate_components(self) -> None:
        assert self.pipeline is not None
        missing = [name for name in self.required_components if not hasattr(self.pipeline, name)]
        if missing:
            raise NotImplementedError(
                f"Pipeline '{self.pipeline_class_name}' for '{self.model_id}' is missing required "
                f"components needed for vector-field steering: {missing}."
            )
        if not hasattr(self.pipeline, "scheduler"):
            raise NotImplementedError(
                f"Pipeline '{self.pipeline_class_name}' does not expose scheduler; cannot perform steered steps."
            )
        scheduler = self.pipeline.scheduler
        if not hasattr(scheduler, "set_timesteps") or not hasattr(scheduler, "step"):
            raise NotImplementedError(
                f"Scheduler for '{self.model_id}' must expose set_timesteps and step for steered inference."
            )

    def prepare_prompt(self, prompt: str) -> PromptCondition:
        self._require_loaded()
        output = self._call_encode_prompt(prompt)
        data = _map_encode_prompt_output(output, self.encode_prompt_output_names)
        return PromptCondition(prompt=prompt, data=data)

    def _call_encode_prompt(self, prompt: str) -> Any:
        assert self.pipeline is not None
        if not hasattr(self.pipeline, "encode_prompt"):
            raise NotImplementedError(
                f"Pipeline '{self.pipeline_class_name}' does not expose encode_prompt; "
                "the adapter cannot compute concept-conditioned vector fields."
            )
        guidance_scale = self._manual_guidance_scale()
        do_classifier_free_guidance = guidance_scale > 1.0
        candidates = {
            "prompt": prompt,
            "prompt_2": prompt,
            "prompt_3": prompt,
            "negative_prompt": "",
            "negative_prompt_2": "",
            "negative_prompt_3": "",
            "device": self.device,
            "dtype": self.dtype,
            "num_images_per_prompt": 1,
            "num_videos_per_prompt": 1,
            "do_classifier_free_guidance": do_classifier_free_guidance,
            "max_sequence_length": self.config.get("max_sequence_length"),
        }
        return _call_with_supported_kwargs(self.pipeline.encode_prompt, candidates)

    def _manual_guidance_scale(self) -> float:
        return float(self.config.get("manual_guidance_scale", self.config.get("guidance_scale") or 1.0))

    def prepare_initial_latents(
        self,
        prompt: str,
        batch_size: int,
        generator: torch.Generator | None,
        **generation_kwargs: Any,
    ) -> tuple[torch.Tensor, AdapterState]:
        self._require_loaded()
        del prompt
        assert self.pipeline is not None
        if not hasattr(self.pipeline, "prepare_latents"):
            raise NotImplementedError(
                f"Pipeline '{self.pipeline_class_name}' does not expose prepare_latents; "
                "the adapter cannot expose initial latent state for steering."
            )
        transformer = getattr(self.pipeline, "transformer", None)
        latent_channels = (
            getattr(getattr(transformer, "config", None), "in_channels", None)
            or getattr(getattr(self.pipeline.vae, "config", None), "latent_channels", None)
            or 4
        )
        candidates = {
            "batch_size": batch_size,
            "num_channels_latents": latent_channels,
            "height": generation_kwargs.get("height"),
            "width": generation_kwargs.get("width"),
            "num_frames": generation_kwargs.get("num_frames"),
            "dtype": self.dtype,
            "device": self.device,
            "generator": generator,
            "latents": None,
        }
        output = _call_with_supported_kwargs(self.pipeline.prepare_latents, candidates)
        if isinstance(output, torch.Tensor):
            return output, AdapterState(extra={})
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            state = AdapterState(extra={"prepare_latents_tail": output[1:]})
            _name_prepare_latents_tail(state, output[1:])
            return output[0], state
        raise NotImplementedError(
            f"prepare_latents for '{self.pipeline_class_name}' returned unsupported type "
            f"{type(output).__name__}; expected Tensor or tuple with Tensor first."
        )

    def set_timesteps(
        self,
        num_inference_steps: int,
        latents: torch.Tensor | None = None,
        state: AdapterState | None = None,
    ) -> list[Any]:
        self._require_loaded()
        del state
        assert self.pipeline is not None
        kwargs: dict[str, Any] = {"num_inference_steps": num_inference_steps, "device": self.device}
        if _scheduler_uses_dynamic_shifting(self.pipeline.scheduler):
            if latents is None:
                raise NotImplementedError(
                    f"Scheduler for '{self.model_id}' requires dynamic-shift mu, but latents were not "
                    "passed to set_timesteps."
                )
            from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps

            image_seq_len = latents.shape[1]
            scheduler_config = self.pipeline.scheduler.config
            mu = calculate_shift(
                image_seq_len,
                scheduler_config.get("base_image_seq_len", 256),
                scheduler_config.get("max_image_seq_len", 4096),
                scheduler_config.get("base_shift", 0.5),
                scheduler_config.get("max_shift", 1.15),
            )
            sigmas = None
            timesteps, _ = retrieve_timesteps(
                self.pipeline.scheduler,
                num_inference_steps,
                self.device,
                sigmas=sigmas,
                mu=mu,
            )
            if hasattr(self.pipeline.scheduler, "set_begin_index"):
                self.pipeline.scheduler.set_begin_index(0)
            self.timesteps = list(timesteps)
            return self.timesteps

        _call_with_supported_kwargs(self.pipeline.scheduler.set_timesteps, kwargs)
        if hasattr(self.pipeline.scheduler, "set_begin_index"):
            self.pipeline.scheduler.set_begin_index(0)
        if not hasattr(self.pipeline.scheduler, "timesteps"):
            raise NotImplementedError(
                f"Scheduler for '{self.model_id}' did not expose timesteps after set_timesteps."
            )
        self.timesteps = list(self.pipeline.scheduler.timesteps)
        return self.timesteps

    def scheduler_step(
        self,
        model_prediction: torch.Tensor,
        timestep: Any,
        latents: torch.Tensor,
        state: AdapterState,
        generator: torch.Generator | None = None,
    ) -> SchedulerStepResult:
        self._require_loaded()
        assert self.pipeline is not None
        output = _call_with_supported_kwargs(
            self.pipeline.scheduler.step,
            {
                "model_output": model_prediction,
                "model_prediction": model_prediction,
                "noise_pred": model_prediction,
                "timestep": timestep,
                "sample": latents,
                "latents": latents,
                "generator": generator,
            },
        )
        if hasattr(output, "prev_sample"):
            return SchedulerStepResult(latents=output.prev_sample, state=state)
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            return SchedulerStepResult(latents=output[0], state=state)
        if isinstance(output, torch.Tensor):
            return SchedulerStepResult(latents=output, state=state)
        raise NotImplementedError(
            f"Scheduler step for '{self.model_id}' returned unsupported type {type(output).__name__}; "
            "expected prev_sample, Tensor, or tuple with Tensor first."
        )

    def decode_latents(self, latents: torch.Tensor, state: AdapterState) -> Any:
        self._require_loaded()
        del state
        assert self.pipeline is not None
        if not hasattr(self.pipeline, "vae"):
            raise NotImplementedError(
                f"Pipeline '{self.pipeline_class_name}' does not expose a VAE for decoding."
            )
        vae = self.pipeline.vae
        scaling = getattr(getattr(vae, "config", None), "scaling_factor", None)
        latents_for_decode = latents
        if scaling:
            latents_for_decode = latents_for_decode / scaling
        decoded = vae.decode(latents_for_decode)
        sample = decoded.sample if hasattr(decoded, "sample") else decoded[0]
        if self.task_type == "text_to_image" and hasattr(self.pipeline, "image_processor"):
            return self.pipeline.image_processor.postprocess(sample, output_type="pil")
        if self.task_type == "text_to_video" and hasattr(self.pipeline, "video_processor"):
            return self.pipeline.video_processor.postprocess_video(sample, output_type="pil")
        return sample.detach().float().cpu()

    def _call_transformer(self, transformer: Callable[..., Any], candidates: dict[str, Any]) -> torch.Tensor:
        output = _call_with_supported_kwargs(transformer, candidates)
        if hasattr(output, "sample"):
            return output.sample
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            return output[0]
        if isinstance(output, torch.Tensor):
            return output
        raise NotImplementedError(
            f"Transformer for '{self.model_id}' returned unsupported type {type(output).__name__}; "
            "expected Tensor, tuple with Tensor first, or object with .sample."
        )

    def _timestep_batch(self, timestep: Any, batch_size: int) -> torch.Tensor:
        if isinstance(timestep, torch.Tensor):
            timestep_tensor = timestep.to(device=self.device)
        else:
            timestep_tensor = torch.tensor(timestep, device=self.device)
        if timestep_tensor.ndim == 0:
            timestep_tensor = timestep_tensor.repeat(batch_size)
        return timestep_tensor


def _call_with_supported_kwargs(callable_obj: Callable[..., Any], candidates: dict[str, Any]) -> Any:
    signature_source = callable_obj.forward if isinstance(callable_obj, torch.nn.Module) else callable_obj
    signature = inspect.signature(signature_source)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        kwargs = {key: value for key, value in candidates.items() if value is not None}
    else:
        kwargs = {
            key: value
            for key, value in candidates.items()
            if key in signature.parameters and value is not None
        }
    required = [
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.default is inspect.Parameter.empty
        and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ]
    missing_required = [name for name in required if name not in kwargs]
    if missing_required:
        raise NotImplementedError(
            f"Callable {callable_obj} requires arguments the adapter cannot supply: {missing_required}."
        )
    return callable_obj(**kwargs)


def _scheduler_uses_dynamic_shifting(scheduler: Any) -> bool:
    config = getattr(scheduler, "config", {})
    if hasattr(config, "get"):
        return bool(config.get("use_dynamic_shifting", False))
    return bool(getattr(config, "use_dynamic_shifting", False))


def _map_encode_prompt_output(output: Any, names: tuple[str, ...]) -> dict[str, Any]:
    if isinstance(output, dict):
        return dict(output)
    if isinstance(output, torch.Tensor):
        return {names[0]: output}
    if isinstance(output, tuple):
        if len(output) < len(names):
            raise NotImplementedError(
                f"encode_prompt returned {len(output)} values, but adapter expected at least {len(names)}: {names}."
            )
        return {name: value for name, value in zip(names, output)}
    raise NotImplementedError(
        f"encode_prompt returned unsupported type {type(output).__name__}; "
        "expected Tensor, tuple, or mapping."
    )


def _name_prepare_latents_tail(state: AdapterState, tail: tuple[Any, ...]) -> None:
    names = ["latent_image_ids", "latent_ids", "noise", "extra_0", "extra_1"]
    for name, value in zip(names, tail):
        state.extra[name] = value


def _stable_prompt_scalar(prompt: str) -> float:
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], byteorder="big", signed=False)
    return (value / 2**32) * 2.0 - 1.0


def _checker_pattern(latents: torch.Tensor) -> torch.Tensor:
    pattern = torch.ones_like(latents)
    if latents.ndim >= 3:
        grid = torch.arange(latents.shape[-1], device=latents.device, dtype=latents.dtype)
        pattern = pattern * torch.cos(grid).reshape([1] * (latents.ndim - 1) + [-1])
    return pattern
