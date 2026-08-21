from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping

import torch

if TYPE_CHECKING:
    from hierasafe_flow.generation.temporal_artifacts import TemporalEvidenceBundle


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass
class PromptCondition:
    prompt: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterState:
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DenoisingStepContext:
    """Authenticated coordinates for one model vector-field prediction.

    ``global_*`` coordinates are run-level ordering evidence.  Steering
    schedules must use ``local_*`` coordinates so each independently reset
    native segment receives the same registered intervention window.
    """

    global_step_index: int
    global_num_steps: int
    segment_index: int
    segment_count: int
    local_step_index: int
    local_num_steps: int
    model_role: str
    model_id: str
    model_revision: str
    condition_epoch: int
    anchor_sha256: str | None
    segment_seed: int

    def __post_init__(self) -> None:
        integer_fields = {
            "global_step_index": self.global_step_index,
            "global_num_steps": self.global_num_steps,
            "segment_index": self.segment_index,
            "segment_count": self.segment_count,
            "local_step_index": self.local_step_index,
            "local_num_steps": self.local_num_steps,
            "condition_epoch": self.condition_epoch,
            "segment_seed": self.segment_seed,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"DenoisingStepContext.{name} must be an integer.")
        if self.global_num_steps <= 0 or self.local_num_steps <= 0:
            raise ValueError("Global and local denoising step counts must be positive.")
        if self.segment_count <= 0:
            raise ValueError("DenoisingStepContext.segment_count must be positive.")
        if not 0 <= self.global_step_index < self.global_num_steps:
            raise ValueError("Global denoising step index is outside the run-level range.")
        if not 0 <= self.segment_index < self.segment_count:
            raise ValueError("Segment index is outside the declared segment range.")
        if not 0 <= self.local_step_index < self.local_num_steps:
            raise ValueError("Local denoising step index is outside the segment range.")
        if self.condition_epoch < 0 or self.segment_seed < 0:
            raise ValueError("Condition epoch and segment seed must be non-negative.")
        for name, value in {
            "model_role": self.model_role,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
        }.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"DenoisingStepContext.{name} must be a non-empty string.")
        if self.anchor_sha256 is not None and not _SHA256_RE.fullmatch(self.anchor_sha256):
            raise ValueError(
                "DenoisingStepContext.anchor_sha256 must be canonical lowercase SHA-256 or None."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "global_step_index": self.global_step_index,
            "global_num_steps": self.global_num_steps,
            "segment_index": self.segment_index,
            "segment_count": self.segment_count,
            "local_step_index": self.local_step_index,
            "local_num_steps": self.local_num_steps,
            "model_role": self.model_role,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "condition_epoch": self.condition_epoch,
            "anchor_sha256": self.anchor_sha256,
            "segment_seed": self.segment_seed,
        }


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


@dataclass(frozen=True)
class LatentLayout:
    """Invertible description of a model vector field as ``[B, T, D]``.

    ``D`` is the model's latent feature/channel axis and ``T`` is the
    flattened product of every non-batch, non-feature axis.  The original
    tensor is never copied merely to describe its layout; ``canonicalize``
    and ``restore`` perform the explicit permutation/reshape round trip used
    by model-independent steering methods.
    """

    adapter_name: str
    original_shape: tuple[int, ...]
    feature_dim: int
    token_dims: tuple[int, ...]
    token_shape: tuple[int, ...]
    canonical_shape: tuple[int, int, int]

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        *,
        adapter_name: str,
        feature_dim: int,
    ) -> "LatentLayout":
        if tensor.ndim < 2:
            raise ValueError(
                f"Adapter '{adapter_name}' vector fields must have a batch and feature axis; "
                f"got shape {tuple(tensor.shape)}."
            )
        normalized_feature_dim = feature_dim % tensor.ndim
        if normalized_feature_dim == 0:
            raise ValueError(
                f"Adapter '{adapter_name}' declares batch axis 0 as its latent feature axis."
            )
        token_dims = tuple(
            dim for dim in range(1, tensor.ndim) if dim != normalized_feature_dim
        )
        token_shape = tuple(int(tensor.shape[dim]) for dim in token_dims)
        token_count = 1
        for size in token_shape:
            token_count *= size
        return cls(
            adapter_name=adapter_name,
            original_shape=tuple(int(size) for size in tensor.shape),
            feature_dim=normalized_feature_dim,
            token_dims=token_dims,
            token_shape=token_shape,
            canonical_shape=(
                int(tensor.shape[0]),
                token_count,
                int(tensor.shape[normalized_feature_dim]),
            ),
        )

    def canonicalize(self, tensor: torch.Tensor) -> torch.Tensor:
        if tuple(tensor.shape) != self.original_shape:
            raise ValueError(
                f"Latent layout for '{self.adapter_name}' expects {self.original_shape}, "
                f"got {tuple(tensor.shape)}."
            )
        permutation = (0, *self.token_dims, self.feature_dim)
        return tensor.permute(permutation).reshape(self.canonical_shape)

    def restore(self, canonical: torch.Tensor) -> torch.Tensor:
        if tuple(canonical.shape) != self.canonical_shape:
            raise ValueError(
                f"Canonical layout for '{self.adapter_name}' expects {self.canonical_shape}, "
                f"got {tuple(canonical.shape)}."
            )
        permutation = (0, *self.token_dims, self.feature_dim)
        permuted_shape = (
            self.original_shape[0],
            *self.token_shape,
            self.original_shape[self.feature_dim],
        )
        inverse = [0] * len(permutation)
        for output_dim, input_dim in enumerate(permutation):
            inverse[input_dim] = output_dim
        return canonical.reshape(permuted_shape).permute(tuple(inverse))

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter_name,
            "original_shape": list(self.original_shape),
            "canonical_shape": list(self.canonical_shape),
            "feature_dim": self.feature_dim,
            "token_dims": list(self.token_dims),
            "token_shape": list(self.token_shape),
        }


class FrozenGeneratorAdapter(ABC):
    adapter_name = "base"
    task_type = "unknown"
    pipeline_class_name: str | None = None
    latent_feature_dim = 1

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

    def prepare_prompts(self, prompts: list[str]) -> list[PromptCondition]:
        """Encode multiple prompts, preserving the single-prompt contract for callers."""
        return [self.prepare_prompt(prompt) for prompt in prompts]

    def denoising_step_context(
        self,
        global_step_index: int,
        global_num_steps: int,
        state: AdapterState,
    ) -> DenoisingStepContext:
        """Return the default one-segment execution context.

        Segmented adapters override this method.  The default deliberately
        preserves existing adapters while still producing a complete trace.
        """

        return DenoisingStepContext(
            global_step_index=int(global_step_index),
            global_num_steps=int(global_num_steps),
            segment_index=0,
            segment_count=1,
            local_step_index=int(global_step_index),
            local_num_steps=int(global_num_steps),
            model_role=str(state.extra.get("model_role", "primary")),
            model_id=str(state.extra.get("model_id", self.model_id)),
            model_revision=str(
                state.extra.get("model_revision", self.config.get("revision") or "unversioned")
            ),
            condition_epoch=int(state.extra.get("condition_epoch", 0)),
            anchor_sha256=state.extra.get("anchor_sha256"),
            segment_seed=int(state.extra.get("segment_seed", state.extra.get("base_seed", 0))),
        )

    def conditioning_cache_identity(
        self,
        prompt: str,
        state: AdapterState,
        *,
        prompt_view: str,
        call_role: str,
    ) -> dict[str, Any]:
        """Return a canonical JSON identity for state-aware text conditioning."""

        if not isinstance(prompt, str):
            raise TypeError("Condition-cache prompts must be strings.")
        if not prompt_view or not call_role:
            raise ValueError("Condition-cache prompt_view and call_role must be non-empty.")
        context = state.extra.get("_active_denoising_step_context")
        if isinstance(context, DenoisingStepContext):
            model_role = context.model_role
            model_id = context.model_id
            model_revision = context.model_revision
            segment_index = context.segment_index
            condition_epoch = context.condition_epoch
            anchor_sha256 = context.anchor_sha256
            segment_seed = context.segment_seed
        else:
            model_role = str(state.extra.get("model_role", "primary"))
            model_id = str(state.extra.get("model_id", self.model_id))
            model_revision = str(
                state.extra.get("model_revision", self.config.get("revision") or "unversioned")
            )
            segment_index = int(state.extra.get("segment_index", 0))
            condition_epoch = int(state.extra.get("condition_epoch", 0))
            anchor_sha256 = state.extra.get("anchor_sha256")
            segment_seed = int(
                state.extra.get("segment_seed", state.extra.get("base_seed", 0))
            )
        identity = {
            "schema_version": 1,
            "namespace": "hierasafe_conditioning",
            "adapter": self.adapter_name,
            "model_role": model_role,
            "model_id": model_id,
            "model_revision": model_revision,
            "segment_index": segment_index,
            "condition_epoch": condition_epoch,
            "segment_seed": segment_seed,
            "anchor_sha256": anchor_sha256,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_view": str(prompt_view),
        }
        # Fail here rather than letting a cache implementation reinterpret a
        # non-serializable adapter identity.
        json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return identity

    def prepare_prompt_for_state(
        self,
        prompt: str,
        state: AdapterState,
        *,
        prompt_view: str,
        call_role: str,
    ) -> PromptCondition:
        del state, prompt_view, call_role
        return self.prepare_prompt(prompt)

    def prepare_prompts_for_state(
        self,
        prompts: list[str],
        state: AdapterState,
        *,
        prompt_view: str,
        call_roles: list[str],
    ) -> list[PromptCondition]:
        if len(prompts) != len(call_roles):
            raise ValueError("prompts and call_roles must have identical lengths.")
        del state, prompt_view, call_roles
        return self.prepare_prompts(prompts)

    def take_temporal_evidence(
        self,
        state: AdapterState,
    ) -> TemporalEvidenceBundle | None:
        del state
        return None

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

    def latent_layout(self, vector_field: torch.Tensor) -> LatentLayout:
        """Return this adapter's explicit, invertible latent feature layout."""
        return LatentLayout.from_tensor(
            vector_field,
            adapter_name=self.adapter_name,
            feature_dim=self.latent_feature_dim,
        )

    def begin_conditioning_provenance_scope(self) -> None:
        """Start one independently serialized sample-conditioning scope.

        The default is stateless. Adapters that cache model-native prompt
        transformations must reset only their per-sample provenance state here
        while retaining loaded model weights.
        """

    def conditioning_provenance(self) -> dict[str, Any]:
        """Return JSON-serializable model-native conditioning provenance.

        Most adapters encode the experiment prompt directly and therefore have
        nothing additional to report. Adapters that must transform text into a
        model-native representation (for example Ideogram 4 structured JSON)
        override this hook so the exact representation and its digest are saved
        with every sample report.
        """
        return {}

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
        single_file = kwargs.pop("single_file", None)
        single_file_components = kwargs.pop("single_file_components", None)
        if self.config.get("revision") is not None:
            kwargs["revision"] = self.config["revision"]
        if self.config.get("variant") is not None:
            kwargs["variant"] = self.config["variant"]
        kwargs.setdefault("torch_dtype", self.dtype)
        kwargs.setdefault("local_files_only", bool(self.config.get("local_files_only", False)))
        if os.environ.get("HF_TOKEN") and "token" not in kwargs:
            kwargs["token"] = os.environ["HF_TOKEN"]
        kwargs.setdefault("low_cpu_mem_usage", True)
        if single_file is not None:
            kwargs.update(
                load_single_file_companion_components(
                    single_file_components,
                    torch_dtype=kwargs.get("torch_dtype"),
                    local_files_only=bool(kwargs.get("local_files_only", False)),
                    token=kwargs.get("token"),
                    cache_dir=kwargs.get("cache_dir"),
                )
            )
            checkpoint_path = resolve_single_file_checkpoint(
                single_file,
                token=kwargs.get("token"),
                revision=kwargs.get("revision"),
                local_files_only=bool(kwargs.get("local_files_only", False)),
                cache_dir=kwargs.get("cache_dir"),
            )
            self.pipeline = pipeline_cls.from_single_file(checkpoint_path, **kwargs)
        else:
            self.pipeline = pipeline_cls.from_pretrained(self.model_id, **kwargs)
        configure_pipeline_vae_tiling(self.pipeline, self.config.get("vae_tiling"))
        cpu_offload = self.config.get("cpu_offload", False)
        offload_strategy = self._cpu_offload_strategy(cpu_offload)
        if offload_strategy == "group":
            self._enable_group_offload(cpu_offload)
        elif offload_strategy == "sequential" and hasattr(self.pipeline, "enable_sequential_cpu_offload"):
            self.pipeline.enable_sequential_cpu_offload(device=self.device)
        elif offload_strategy == "model" and hasattr(self.pipeline, "enable_model_cpu_offload"):
            self.pipeline.enable_model_cpu_offload(device=self.device)
        elif "device_map" not in kwargs and hasattr(self.pipeline, "to"):
            self.pipeline.to(self.device)
        self._freeze_pipeline()
        self._validate_components()
        self.loaded = True

    def _cpu_offload_strategy(self, cpu_offload: Any) -> str | None:
        if isinstance(cpu_offload, Mapping):
            strategy = cpu_offload.get("strategy", cpu_offload.get("mode", "model"))
            return str(strategy).lower() if strategy else None
        if isinstance(cpu_offload, str):
            value = cpu_offload.lower()
            if value in {"false", "none", "off"}:
                return None
            if value == "true":
                return "model"
            return value
        if cpu_offload is True:
            return "model"
        return None

    def _enable_group_offload(self, cpu_offload: Any) -> None:
        if self.pipeline is None:
            raise RuntimeError("Cannot enable group offload before the pipeline is loaded.")
        if not hasattr(self.pipeline, "enable_group_offload"):
            raise RuntimeError(
                f"Pipeline '{self.pipeline_class_name}' does not support group offload."
            )
        options = dict(cpu_offload) if isinstance(cpu_offload, Mapping) else {}
        kwargs = {
            "onload_device": self.device,
            "offload_device": torch.device(str(options.get("offload_device", "cpu"))),
            "offload_type": str(options.get("offload_type", "leaf_level")),
            "non_blocking": bool(options.get("non_blocking", False)),
            "use_stream": bool(options.get("use_stream", False)),
            "record_stream": bool(options.get("record_stream", False)),
            "low_cpu_mem_usage": bool(options.get("low_cpu_mem_usage", False)),
        }
        if options.get("num_blocks_per_group") is not None:
            kwargs["num_blocks_per_group"] = int(options["num_blocks_per_group"])
        if options.get("offload_to_disk_path") is not None:
            kwargs["offload_to_disk_path"] = str(options["offload_to_disk_path"])
        if options.get("exclude_modules") is not None:
            kwargs["exclude_modules"] = options["exclude_modules"]
        self.pipeline.enable_group_offload(**kwargs)

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

    def prepare_prompts(self, prompts: list[str]) -> list[PromptCondition]:
        self._require_loaded()
        if not prompts:
            return []
        output = self._call_encode_prompt(list(prompts))
        data = _map_encode_prompt_output(output, self.encode_prompt_output_names)
        return [
            PromptCondition(prompt=prompt, data=_slice_condition_data(data, index, len(prompts)))
            for index, prompt in enumerate(prompts)
        ]

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
            "negative_prompt": self.config.get("negative_prompt", ""),
            "negative_prompt_2": self.config.get("negative_prompt_2", ""),
            "negative_prompt_3": self.config.get("negative_prompt_3", ""),
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
    signature = _signature_for_callable(callable_obj)
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


def _signature_for_callable(callable_obj: Callable[..., Any]) -> inspect.Signature:
    if not isinstance(callable_obj, torch.nn.Module):
        return inspect.signature(callable_obj)
    signature = inspect.signature(callable_obj.forward)
    module_parameter = signature.parameters.get("module")
    if module_parameter is not None and module_parameter.default is inspect.Parameter.empty:
        return inspect.signature(type(callable_obj).forward)
    return signature


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


def _slice_condition_data(data: dict[str, Any], index: int, batch_size: int) -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
            sliced[key] = value[index : index + 1]
        elif isinstance(value, list) and len(value) == batch_size:
            sliced[key] = value[index]
        elif isinstance(value, tuple) and len(value) == batch_size:
            sliced[key] = value[index]
        else:
            sliced[key] = value
    return sliced


def _name_prepare_latents_tail(state: AdapterState, tail: tuple[Any, ...]) -> None:
    names = ["latent_image_ids", "latent_ids", "noise", "extra_0", "extra_1"]
    for name, value in zip(names, tail):
        state.extra[name] = value


def _stable_prompt_scalar(prompt: str) -> float:
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], byteorder="big", signed=False)
    return (value / 2**32) * 2.0 - 1.0


def resolve_single_file_checkpoint(
    single_file: Any,
    *,
    token: str | bool | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
    cache_dir: str | os.PathLike[str] | None = None,
) -> str:
    """Resolve a single-file checkpoint spec to a local path accepted by diffusers."""

    if not isinstance(single_file, Mapping):
        return str(single_file)

    if single_file.get("path") is not None:
        return str(single_file["path"])

    repo_id = single_file.get("repo_id")
    filename = single_file.get("filename")
    if not repo_id or not filename:
        raise ValueError(
            "Single-file checkpoint mappings must provide either 'path' or both "
            "'repo_id' and 'filename'."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Resolving a Hugging Face single-file checkpoint requires huggingface_hub."
        ) from exc

    download_kwargs: dict[str, Any] = {
        "repo_id": str(repo_id),
        "filename": str(filename),
        "local_files_only": bool(single_file.get("local_files_only", local_files_only)),
    }
    resolved_revision = single_file.get("revision", revision)
    if resolved_revision is not None:
        download_kwargs["revision"] = str(resolved_revision)
    resolved_token = single_file.get("token", token)
    if resolved_token is not None:
        download_kwargs["token"] = resolved_token
    resolved_cache_dir = single_file.get("cache_dir", cache_dir)
    if resolved_cache_dir is not None:
        download_kwargs["cache_dir"] = str(resolved_cache_dir)
    for optional_key in ("repo_type", "subfolder", "local_dir"):
        if single_file.get(optional_key) is not None:
            download_kwargs[optional_key] = str(single_file[optional_key])

    return hf_hub_download(**download_kwargs)


def load_single_file_companion_components(
    component_config: Any,
    *,
    torch_dtype: torch.dtype | None = None,
    local_files_only: bool = False,
    token: str | bool | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Load explicit pretrained components that are absent from a single-file checkpoint."""

    if component_config is None:
        return {}
    if not isinstance(component_config, Mapping):
        raise ValueError("single_file_components must be a mapping when provided.")

    default_repo_id = component_config.get("repo_id") or component_config.get("pretrained_model_name_or_path")
    component_specs = component_config.get("components", component_config)
    if not isinstance(component_specs, Mapping):
        raise ValueError("single_file_components.components must be a mapping.")

    loaded: dict[str, Any] = {}
    for component_name, raw_spec in component_specs.items():
        if component_name in {"repo_id", "pretrained_model_name_or_path"}:
            continue
        spec = dict(raw_spec or {})
        module_name = str(spec.pop("module", "diffusers"))
        class_name = spec.pop("class_name", None)
        if not class_name:
            raise ValueError(f"Companion component {component_name!r} is missing class_name.")
        pretrained_path = spec.pop("pretrained_model_name_or_path", spec.pop("repo_id", default_repo_id))
        if not pretrained_path:
            raise ValueError(
                f"Companion component {component_name!r} needs pretrained_model_name_or_path or a default repo_id."
            )

        class_obj = getattr(importlib.import_module(module_name), str(class_name))
        if not hasattr(class_obj, "from_pretrained"):
            raise ValueError(f"Companion component class {module_name}.{class_name} lacks from_pretrained().")

        load_kwargs = dict(spec.pop("kwargs", {}))
        load_kwargs.update(spec)
        load_kwargs.setdefault("pretrained_model_name_or_path", str(pretrained_path))
        load_kwargs.setdefault("local_files_only", local_files_only)
        if token is not None:
            load_kwargs.setdefault("token", token)
        if cache_dir is not None:
            load_kwargs.setdefault("cache_dir", str(cache_dir))
        if issubclass(class_obj, torch.nn.Module):
            load_kwargs.setdefault("torch_dtype", torch_dtype)

        loaded[str(component_name)] = class_obj.from_pretrained(**load_kwargs)

    return loaded


def configure_pipeline_vae_tiling(pipeline: Any, tiling_config: Any) -> None:
    """Enable configured VAE tiling for pipelines whose decoder needs spatial tiles."""

    if not tiling_config:
        return
    kwargs: dict[str, Any] = {}
    if isinstance(tiling_config, Mapping):
        if not bool(tiling_config.get("enabled", True)):
            return
        allowed = {
            "tile_sample_min_height",
            "tile_sample_min_width",
            "tile_sample_min_num_frames",
            "tile_sample_stride_height",
            "tile_sample_stride_width",
            "tile_sample_stride_num_frames",
        }
        kwargs = {key: value for key, value in tiling_config.items() if key in allowed and value is not None}
    elif tiling_config is not True:
        raise ValueError("model.vae_tiling must be true or a mapping with enabled/tile parameters.")

    vae = getattr(pipeline, "vae", None)
    if vae is None or not hasattr(vae, "enable_tiling"):
        raise NotImplementedError("model.vae_tiling is set, but the pipeline VAE does not expose enable_tiling().")
    vae.enable_tiling(**kwargs)


def _checker_pattern(latents: torch.Tensor) -> torch.Tensor:
    pattern = torch.ones_like(latents)
    if latents.ndim >= 3:
        grid = torch.arange(latents.shape[-1], device=latents.device, dtype=latents.dtype)
        pattern = pattern * torch.cos(grid).reshape([1] * (latents.ndim - 1) + [-1])
    return pattern
