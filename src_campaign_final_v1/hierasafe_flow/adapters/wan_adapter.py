from __future__ import annotations

import gc
import hashlib
from importlib import import_module, metadata
import json
import math
import os
import threading
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch

from hierasafe_flow.adapters.base import (
    AdapterState,
    DiffusersFrozenAdapter,
    PromptCondition,
    SchedulerStepResult,
    configure_pipeline_vae_tiling,
)
from hierasafe_flow.evaluation.temporal_qualification import (
    TEMPORAL_CRITERIA_BY_MODEL,
    validate_temporal_production_gate,
)


WAN_TEMPORAL_PROTOCOL_KEY = "wan_temporal_protocol"
WAN_T2V_MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
WAN_T2V_REVISION = "5be7df9619b54f4e2667b2755bc6a756675b5cd7"
WAN_I2V_MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
WAN_I2V_REVISION = "596658fd9ca6b7b71d5057529bbf319ecbc61d74"
WAN_TEMPORAL_PILOT_ENV = "HIERASAFE_WAN_TEMPORAL_PILOT"
WAN_SHARED_HASH_CACHE_ENV = "HIERASAFE_WAN_SHARED_HASH_CACHE_DIR"
WAN_FTFY_VERSION = "6.3.1"
WAN_DIFFUSERS_REPOSITORY = "https://github.com/huggingface/diffusers.git"
WAN_DIFFUSERS_REVISION = "577b28f8f5d30eabdd357d74944cd76568292faf"
_WAN_PRODUCTION_CRITERIA = TEMPORAL_CRITERIA_BY_MODEL["wan22_t2v_a14b"]

# Wan's pinned model card calls T2V-A14B a 5-second model and its official
# Diffusers example requests 81 frames and exports at 16 fps:
# https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers/blob/
# 5be7df9619b54f4e2667b2755bc6a756675b5cd7/README.md
# The continuation checkpoint is the official Wan2.2 A14B I2V conversion:
# https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers/tree/
# 596658fd9ca6b7b71d5057529bbf319ecbc61d74
_EXPECTED_TEMPORAL_PROTOCOL = {
    "schema_version": 1,
    "strategy": "t2v_5s_then_two_i2v_5s_continuations",
    "primary_model_id": WAN_T2V_MODEL_ID,
    "primary_revision": WAN_T2V_REVISION,
    "continuation_model_id": WAN_I2V_MODEL_ID,
    "continuation_revision": WAN_I2V_REVISION,
    "native_segment_frames": 81,
    "native_fps": 16,
    "segment_count": 3,
    "stitched_inclusive_frames": 241,
    "output_frames": 240,
    "output_fps": 16,
    "duration_seconds": 15.0,
    "boundary_deduplication": "segment_0 + segment_1[1:] + segment_2[1:]",
    "endpoint_crop": "[0:240]",
    "continuation_seed_derivation": "sha256(wan22_i2v_continuation_v1|base_seed|segment_index)[:8] mod (2**63-1)",
}

# The two pinned repositories publish byte-identical text encoder, tokenizer,
# and VAE weights. Prompt embeddings and decoded pixels therefore stay in one
# explicitly verified conditioning/latent space when the denoiser is swapped.
WAN_SHARED_COMPONENT_WEIGHT_SHA256 = {
    "text_encoder/model-00001-of-00003.safetensors": (
        "a8e861969c7433e707cc5a74065d795d36cca07ec96eb6763eb4083df7248f58"
    ),
    "text_encoder/model-00002-of-00003.safetensors": (
        "d57d948ece4837d850b7a859a4415121d57cacf8b9ee1d4db200c67f592902d7"
    ),
    "text_encoder/model-00003-of-00003.safetensors": (
        "0da9ee284e21d1406df708788db1d502d95d75f69faa25cd26151bf8829b7c5f"
    ),
    "tokenizer/spiece.model": ("e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458"),
    "tokenizer/tokenizer.json": (
        "20a46ac256746594ed7e1e3ef733b83fbc5a6f0922aa7480eda961743de080ef"
    ),
    "vae/diffusion_pytorch_model.safetensors": (
        "d6e524b3fffede1787a74e81b30976dce5400c4439ba64222168e607ed19e793"
    ),
}

_WAN_HASH_CACHE_SCHEMA_VERSION = 1
_WAN_HASH_MEMORY_CACHE: dict[tuple[Any, ...], str] = {}
_WAN_HASH_MEMORY_CACHE_LOCK = threading.Lock()


def _require_path_below_active_conda_prefix(path: Path, *, label: str) -> Path:
    raw_prefix = os.environ.get("CONDA_PREFIX")
    if not raw_prefix:
        raise RuntimeError(f"CONDA_PREFIX is absent while authenticating {label}.")
    prefix = Path(raw_prefix).resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(prefix)
    except ValueError as exc:
        raise RuntimeError(
            f"{label} resolves outside the active CONDA_PREFIX: {resolved} is not below {prefix}."
        ) from exc
    return resolved


def _validate_wan_diffusers_vcs_install() -> dict[str, Any]:
    """Bind the direct Wan cleaner to the exact production Diffusers checkout."""

    try:
        distribution = metadata.distribution("diffusers")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("Wan native negative prompting requires Diffusers.") from exc
    distribution_path = _require_path_below_active_conda_prefix(
        Path(distribution.locate_file("")),
        label="Diffusers distribution",
    )
    raw_direct_url = distribution.read_text("direct_url.json")
    if not raw_direct_url:
        raise RuntimeError(
            "Wan native negative prompting requires VCS-installed Diffusers direct_url.json."
        )
    try:
        direct_url = json.loads(raw_direct_url)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Diffusers direct_url.json is not valid JSON.") from exc
    if not isinstance(direct_url, dict):
        raise RuntimeError("Diffusers direct_url.json must contain an object.")
    vcs_info = direct_url.get("vcs_info")
    if not isinstance(vcs_info, dict) or vcs_info.get("vcs") != "git":
        raise RuntimeError("Wan native negative prompting requires a Git Diffusers install.")
    repository = direct_url.get("url")
    revision = vcs_info.get("commit_id")
    if repository != WAN_DIFFUSERS_REPOSITORY:
        raise RuntimeError(
            "Wan native negative prompting has the wrong Diffusers repository: "
            f"expected {WAN_DIFFUSERS_REPOSITORY!r}, got {repository!r}."
        )
    if revision != WAN_DIFFUSERS_REVISION:
        raise RuntimeError(
            "Wan native negative prompting has the wrong Diffusers revision: "
            f"expected {WAN_DIFFUSERS_REVISION}, got {revision!r}."
        )
    dir_info = direct_url.get("dir_info")
    if isinstance(dir_info, dict) and dir_info.get("editable") is True:
        raise RuntimeError("Wan native negative prompting rejects editable Diffusers installs.")
    return {
        "package_name": "diffusers",
        "package_version": distribution.version,
        "distribution_path": str(distribution_path),
        "repository": repository,
        "revision": revision,
        "editable": False,
    }


def validate_wan_native_negative_prompt_cleaner() -> dict[str, Any]:
    """Authenticate and execute Wan I2V's real pinned raw-prompt cleaner on CPU."""

    try:
        distribution = metadata.distribution("ftfy")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"Wan native negative prompting requires ftfy=={WAN_FTFY_VERSION}."
        ) from exc
    if distribution.version != WAN_FTFY_VERSION:
        raise RuntimeError(
            "Wan native negative prompting has the wrong ftfy version: "
            f"expected {WAN_FTFY_VERSION}, got {distribution.version!r}."
        )
    distribution_path = _require_path_below_active_conda_prefix(
        Path(distribution.locate_file("")),
        label="ftfy distribution",
    )
    diffusers_install = _validate_wan_diffusers_vcs_install()

    try:
        pipeline_module = import_module("diffusers.pipelines.wan.pipeline_wan_i2v")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("The authenticated Diffusers install lacks Wan I2V prompt cleaning.") from exc
    pipeline_module_file = getattr(pipeline_module, "__file__", None)
    if not isinstance(pipeline_module_file, str) or not pipeline_module_file:
        raise RuntimeError("Wan I2V prompt-cleaner module has no resolvable source path.")
    pipeline_module_path = _require_path_below_active_conda_prefix(
        Path(pipeline_module_file),
        label="Diffusers Wan I2V module",
    )

    ftfy_module = getattr(pipeline_module, "ftfy", None)
    fix_text = getattr(ftfy_module, "fix_text", None)
    if ftfy_module is None or not callable(fix_text):
        raise RuntimeError(
            "Diffusers Wan I2V did not bind callable module-level ftfy.fix_text; "
            "its conditional import is not usable."
        )
    ftfy_module_file = getattr(ftfy_module, "__file__", None)
    if not isinstance(ftfy_module_file, str) or not ftfy_module_file:
        raise RuntimeError("Diffusers Wan I2V bound ftfy without a resolvable module path.")
    ftfy_module_path = _require_path_below_active_conda_prefix(
        Path(ftfy_module_file),
        label="ftfy module",
    )

    prompt_clean = getattr(pipeline_module, "prompt_clean", None)
    if not callable(prompt_clean):
        raise RuntimeError("Diffusers Wan I2V does not expose callable prompt_clean.")
    sentinels = {
        "positive": "  Wan positive sentinel\twith   normalized whitespace.  ",
        "negative": "  Wan negative sentinel\nwith   normalized whitespace.  ",
    }
    sentinel_hashes: dict[str, dict[str, str]] = {}
    for name, sentinel in sentinels.items():
        try:
            first = prompt_clean(sentinel)
            second = prompt_clean(sentinel)
        except Exception as exc:
            raise RuntimeError(f"Diffusers Wan I2V prompt_clean failed for {name} sentinel.") from exc
        if not isinstance(first, str) or not first or first != second:
            raise RuntimeError(
                f"Diffusers Wan I2V prompt_clean returned a non-empty deterministic string "
                f"for neither invocation of the {name} sentinel."
            )
        if first != " ".join(first.split()):
            raise RuntimeError(
                f"Diffusers Wan I2V prompt_clean did not normalize {name} whitespace."
            )
        sentinel_hashes[name] = {
            "input_sha256": _text_sha256(sentinel),
            "output_sha256": _text_sha256(first),
        }

    return {
        "schema_version": 1,
        "status": "verified_before_native_negative_model_allocation",
        "package_name": "ftfy",
        "package_version": distribution.version,
        "distribution_path": str(distribution_path),
        "module_path": str(ftfy_module_path),
        "diffusers_module_path": str(pipeline_module_path),
        "diffusers_install": diffusers_install,
        "sentinel_sha256": sentinel_hashes,
    }


def _wan_prompt_cleaner_fingerprint(prompt: str) -> dict[str, str]:
    """Hash one raw/cleaned prompt using the authenticated official Wan cleaner."""

    if not isinstance(prompt, str):
        raise TypeError(f"Wan prompt-cleaner provenance requires str, got {type(prompt).__name__}.")
    try:
        pipeline_module = import_module("diffusers.pipelines.wan.pipeline_wan_i2v")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("The authenticated Diffusers install lacks Wan I2V prompt cleaning.") from exc
    pipeline_module_file = getattr(pipeline_module, "__file__", None)
    if not isinstance(pipeline_module_file, str) or not pipeline_module_file:
        raise RuntimeError("Wan I2V prompt-cleaner module has no resolvable source path.")
    _require_path_below_active_conda_prefix(
        Path(pipeline_module_file),
        label="Diffusers Wan I2V module",
    )
    prompt_clean = getattr(pipeline_module, "prompt_clean", None)
    if not callable(prompt_clean):
        raise RuntimeError("Diffusers Wan I2V does not expose callable prompt_clean.")
    try:
        first = prompt_clean(prompt)
        second = prompt_clean(prompt)
    except Exception as exc:
        raise RuntimeError("Diffusers Wan I2V prompt_clean failed during call provenance.") from exc
    if not isinstance(first, str) or not first or first != second:
        raise RuntimeError(
            "Diffusers Wan I2V prompt_clean did not return a deterministic non-empty string."
        )
    return {
        "raw_sha256": _text_sha256(prompt),
        "cleaned_sha256": _text_sha256(first),
    }


class WanAdapter(DiffusersFrozenAdapter):
    adapter_name = "wan"
    task_type = "text_to_video"
    pipeline_class_name = "WanPipeline"
    latent_feature_dim = 1  # [B, C, F, H, W]
    required_components = ("transformer", "scheduler", "vae")
    encode_prompt_output_names = ("prompt_embeds", "negative_prompt_embeds")

    def __init__(
        self,
        model_id: str,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(model_id=model_id, device=device, dtype=dtype, config=config)
        self._pipeline_role = "t2v"
        self._native_segment_timesteps: list[Any] = []
        self._completed_segments: list[list[list[Any]]] = []
        self._segment_seeds: list[int] = []
        self._segment_prediction_calls: list[int] = []
        self._segment_prompt_digests: list[set[str]] = []
        self._last_temporal_provenance: dict[str, Any] | None = None
        self._native_pipeline_temporal_configured = False
        self._native_negative_dependency_preflight: dict[str, Any] | None = None
        self._shared_component_artifact_verification: dict[str, dict[str, Any]] = {}

    def load(self) -> None:
        protocol = self._temporal_protocol()
        if protocol is not None:
            self._validate_temporal_protocol(protocol)
            self._require_temporal_qualification(protocol)
            self._validate_temporal_shared_component_load_contract()
            self._verify_pinned_shared_component_artifacts(
                role="primary",
                repo_id=WAN_T2V_MODEL_ID,
                revision=WAN_T2V_REVISION,
            )
        # The pinned official model-card recipe loads AutoencoderKLWan in
        # float32 even when both denoisers use bfloat16. Preserve that numerical
        # contract explicitly instead of allowing from_pretrained(torch_dtype=)
        # to silently downcast the VAE together with the experts.
        try:
            import diffusers
        except ModuleNotFoundError as exc:
            raise RuntimeError("Wan generation requires diffusers.") from exc
        vae_cls = getattr(diffusers, "AutoencoderKLWan", None)
        if vae_cls is None:
            raise RuntimeError("Installed diffusers does not expose AutoencoderKLWan.")
        original_load_kwargs = self.config.get("load_kwargs")
        load_kwargs = dict(original_load_kwargs or {})
        vae = load_kwargs.get("vae")
        if vae is None:
            vae_kwargs: dict[str, Any] = {
                "pretrained_model_name_or_path": self.model_id,
                "subfolder": "vae",
                "revision": WAN_T2V_REVISION,
                "torch_dtype": torch.float32,
                "local_files_only": bool(
                    load_kwargs.get(
                        "local_files_only",
                        self.config.get("local_files_only", False),
                    )
                ),
                "low_cpu_mem_usage": True,
            }
            if load_kwargs.get("cache_dir") is not None:
                vae_kwargs["cache_dir"] = load_kwargs["cache_dir"]
            if load_kwargs.get("token") is not None:
                vae_kwargs["token"] = load_kwargs["token"]
            elif os.environ.get("HF_TOKEN"):
                vae_kwargs["token"] = os.environ["HF_TOKEN"]
            vae = vae_cls.from_pretrained(**vae_kwargs)
            load_kwargs["vae"] = vae
        if getattr(vae, "dtype", None) != torch.float32:
            raise RuntimeError(
                "Wan AutoencoderKLWan must remain torch.float32 under the pinned official recipe."
            )
        self.config["load_kwargs"] = load_kwargs
        try:
            super().load()
        finally:
            if original_load_kwargs is None:
                self.config.pop("load_kwargs", None)
            else:
                self.config["load_kwargs"] = original_load_kwargs
        self._validate_primary_checkpoint_contract()

    def _validate_components(self) -> None:
        super()._validate_components()
        assert self.pipeline is not None
        if hasattr(self.pipeline, "transformer_2") and not hasattr(self.pipeline, "boundary_ratio"):
            raise NotImplementedError(
                "WanPipeline exposes transformer_2 but no boundary_ratio; cannot correctly select "
                "the high-noise/low-noise transformer for steered vector-field prediction."
            )

    def prepare_initial_latents(
        self,
        prompt: str,
        batch_size: int,
        generator: torch.Generator | None,
        **generation_kwargs: Any,
    ) -> tuple[torch.Tensor, AdapterState]:
        self._require_loaded()
        assert self.pipeline is not None
        if self._pipeline_role != "t2v":
            raise RuntimeError(
                "The Wan 15-second protocol is a one-way T2V-to-I2V model transition. "
                "Create a fresh adapter for another prompt."
            )
        if batch_size != 1 and self._temporal_protocol() is not None:
            raise ValueError("The validated Wan temporal protocol currently requires batch_size=1.")

        requested_num_frames = int(generation_kwargs.get("num_frames", 81))
        protocol = self._temporal_protocol()
        if protocol is None:
            _validate_native_frame_count(requested_num_frames)
            native_num_frames = requested_num_frames
        else:
            self._require_temporal_qualification(protocol)
            self._validate_requested_output_contract(generation_kwargs, protocol)
            native_num_frames = int(protocol["native_segment_frames"])

        latent_kwargs = dict(generation_kwargs)
        latent_kwargs["num_frames"] = native_num_frames
        # Match the pinned WanPipeline call path: noise latents remain float32
        # even when the active denoiser experts use bfloat16.
        latents = self.pipeline.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=int(self.pipeline.transformer.config.in_channels),
            height=int(latent_kwargs["height"]),
            width=int(latent_kwargs["width"]),
            num_frames=native_num_frames,
            dtype=torch.float32,
            device=self.device,
            generator=generator,
            latents=None,
        )
        if not isinstance(latents, torch.Tensor):
            raise RuntimeError("Wan T2V prepare_latents did not return a tensor.")
        state = AdapterState()
        if int(latents.shape[1]) != 16:
            raise RuntimeError(
                f"Wan T2V must expose 16 latent channels; got shape {tuple(latents.shape)}."
            )
        if latents.dtype != torch.float32:
            raise RuntimeError(f"Wan T2V noise latents must be float32; got {latents.dtype}.")
        state.extra.update(
            {
                "num_frames": native_num_frames,
                "output_num_frames": requested_num_frames,
                "height": int(generation_kwargs["height"]),
                "width": int(generation_kwargs["width"]),
                "batch_size": int(batch_size),
                "prompt": prompt,
                "base_seed": _generator_initial_seed(generator),
                "wan_segment_index": 0,
                "wan_segment_step_index": 0,
                "wan_pipeline_role": "t2v",
            }
        )
        self._completed_segments = []
        self._segment_seeds = [state.extra["base_seed"]]
        self._segment_prediction_calls = [0]
        self._segment_prompt_digests = [set()]
        self._last_temporal_provenance = None
        return latents, state

    def set_timesteps(
        self,
        num_inference_steps: int,
        latents: torch.Tensor | None = None,
        state: AdapterState | None = None,
    ) -> list[Any]:
        native_timesteps = super().set_timesteps(
            num_inference_steps,
            latents=latents,
            state=state,
        )
        if len(native_timesteps) != num_inference_steps:
            raise RuntimeError(
                "Wan temporal protocol requires exactly one scheduler timestep per configured "
                f"inference step; got {len(native_timesteps)} for {num_inference_steps}."
            )
        protocol = self._temporal_protocol()
        if protocol is None:
            return native_timesteps
        if state is None:
            raise RuntimeError("Wan temporal protocol requires AdapterState in set_timesteps().")
        self._native_segment_timesteps = list(native_timesteps)
        state.extra["wan_steps_per_segment"] = len(native_timesteps)
        state.extra["wan_total_denoising_steps"] = len(native_timesteps) * int(
            protocol["segment_count"]
        )
        # Each repeated segment is a real, independently scheduled denoising
        # trajectory. The configured finer-detailing steering schedule is
        # constant/full-window, so its stride repeats identically per segment.
        self.timesteps = list(native_timesteps) * int(protocol["segment_count"])
        return self.timesteps

    def predict_vector_field(
        self,
        latents: torch.Tensor,
        timestep: Any,
        condition: PromptCondition,
        state: AdapterState,
    ) -> torch.Tensor:
        self._require_loaded()
        assert self.pipeline is not None
        transformer = self._select_transformer(timestep)
        prompt_embeds = condition.data.get("prompt_embeds")
        if prompt_embeds is None:
            raise NotImplementedError("Wan encode_prompt did not expose prompt_embeds.")
        segment_index = int(state.extra.get("wan_segment_index", 0))
        self._record_segment_prediction(segment_index, condition.prompt)

        latent_model_input = latents
        if self._pipeline_role == "i2v":
            image_condition = state.extra.get("wan_i2v_condition")
            if not isinstance(image_condition, torch.Tensor):
                raise RuntimeError("Wan I2V segment is missing its encoded first-frame condition.")
            if (
                image_condition.shape[0] != latents.shape[0]
                or image_condition.shape[2:] != latents.shape[2:]
            ):
                raise RuntimeError(
                    "Wan I2V condition shape is incompatible with the active latent trajectory: "
                    f"condition={tuple(image_condition.shape)}, latents={tuple(latents.shape)}."
                )
            latent_model_input = torch.cat([latents, image_condition], dim=1)
            if int(latent_model_input.shape[1]) != 36:
                raise RuntimeError(
                    f"Wan2.2-I2V-A14B requires 36 input channels; got {latent_model_input.shape[1]}."
                )

        batch = latents.shape[0]
        timestep_batch = self._timestep_batch(timestep, batch)
        guidance_scale = self._manual_guidance_scale()
        candidates = {
            "hidden_states": latent_model_input.to(
                dtype=getattr(transformer, "dtype", latents.dtype)
            ),
            "sample": latent_model_input.to(dtype=getattr(transformer, "dtype", latents.dtype)),
            "timestep": timestep_batch,
            "encoder_hidden_states": prompt_embeds.to(
                dtype=getattr(transformer, "dtype", prompt_embeds.dtype)
            ),
            "encoder_hidden_states_image": None,
            "attention_kwargs": self.config.get("attention_kwargs"),
            "return_dict": True,
        }
        cache_context = (
            transformer.cache_context("cond")
            if hasattr(transformer, "cache_context")
            else nullcontext()
        )
        with cache_context:
            prediction = self._call_transformer(transformer, candidates)
        negative_prompt_embeds = condition.data.get("negative_prompt_embeds")
        if guidance_scale > 1.0 and negative_prompt_embeds is not None:
            uncond_candidates = dict(candidates)
            uncond_candidates["encoder_hidden_states"] = negative_prompt_embeds.to(
                dtype=getattr(transformer, "dtype", negative_prompt_embeds.dtype)
            )
            cache_context = (
                transformer.cache_context("uncond")
                if hasattr(transformer, "cache_context")
                else nullcontext()
            )
            with cache_context:
                uncond = self._call_transformer(transformer, uncond_candidates)
            prediction = uncond + guidance_scale * (prediction - uncond)
        if prediction.shape != latents.shape:
            raise RuntimeError(
                "Wan denoiser output must match the 16-channel latent shape: "
                f"prediction={tuple(prediction.shape)}, latents={tuple(latents.shape)}."
            )
        return prediction

    def scheduler_step(
        self,
        model_prediction: torch.Tensor,
        timestep: Any,
        latents: torch.Tensor,
        state: AdapterState,
        generator: torch.Generator | None = None,
    ) -> SchedulerStepResult:
        result = super().scheduler_step(
            model_prediction=model_prediction,
            timestep=timestep,
            latents=latents,
            state=state,
            generator=generator,
        )
        protocol = self._temporal_protocol()
        if protocol is None:
            return result
        steps_per_segment = int(state.extra["wan_steps_per_segment"])
        segment_step = int(state.extra.get("wan_segment_step_index", 0)) + 1
        state.extra["wan_segment_step_index"] = segment_step
        if segment_step < steps_per_segment:
            return result
        if segment_step > steps_per_segment:
            raise RuntimeError("Wan segment scheduler advanced beyond its frozen timestep count.")

        segment_index = int(state.extra["wan_segment_index"])
        segment_count = int(protocol["segment_count"])
        if segment_index >= segment_count - 1:
            return result

        completed = self._decode_native_segment(result.latents, protocol)
        self._append_completed_segment(completed, protocol)
        next_index = segment_index + 1
        next_latents = self._start_i2v_segment(
            anchors=[video[-1] for video in completed],
            state=state,
            segment_index=next_index,
        )
        state.extra["wan_segment_index"] = next_index
        state.extra["wan_segment_step_index"] = 0
        state.extra["wan_pipeline_role"] = "i2v"
        return SchedulerStepResult(latents=next_latents, state=state)

    def decode_latents(self, latents: torch.Tensor, state: AdapterState) -> Any:
        self._require_loaded()
        protocol = self._temporal_protocol()
        if protocol is None:
            frames = self._decode_native_segment(latents, None)
            expected = int(state.extra.get("output_num_frames", state.extra.get("num_frames", 0)))
            if any(len(video) != expected for video in frames):
                raise RuntimeError(
                    f"Wan native decode returned {[len(video) for video in frames]} frames; expected {expected}."
                )
            return frames

        final_segment = self._decode_native_segment(latents, protocol)
        self._append_completed_segment(final_segment, protocol)
        output, provenance = self._stitch_segments(self._completed_segments, protocol)
        provenance["generation_path"] = "adapter_vector_field_runner_three_segments"
        provenance["segment_seeds"] = list(self._segment_seeds)
        provenance["prediction_calls_per_segment"] = list(self._segment_prediction_calls)
        provenance["condition_prompt_sha256_per_segment"] = [
            sorted(items) for items in self._segment_prompt_digests
        ]
        if any(count <= 0 for count in self._segment_prediction_calls):
            raise RuntimeError(
                "Every Wan segment must receive vector-field predictions; got "
                f"{self._segment_prediction_calls}."
            )
        self._last_temporal_provenance = provenance
        return output

    def conditioning_provenance(self) -> dict[str, Any]:
        provenance = dict(super().conditioning_provenance())
        protocol = self._temporal_protocol()
        if protocol is not None:
            provenance["temporal_generation"] = (
                dict(self._last_temporal_provenance)
                if self._last_temporal_provenance is not None
                else self._configured_temporal_provenance(protocol)
            )
        return provenance

    def configure_native_pipeline_for_temporal_protocol(self) -> dict[str, Any]:
        """Return the only valid first call for the native-negative runner.

        The caller must invoke its already-loaded T2V pipeline once with the
        returned 81-frame contract, then pass that output to
        :meth:`complete_native_pipeline_temporal_protocol`. The completion hook
        performs two deterministic official I2V calls, checks both seams, and
        returns exact 240-frame videos. The hook is intentionally one-way: it
        releases the T2V denoisers before loading I2V to fit a 94-GB node.
        """

        self._require_loaded()
        protocol = self._temporal_protocol()
        if protocol is None:
            raise RuntimeError(
                f"Native Wan long-video generation requires {WAN_TEMPORAL_PROTOCOL_KEY!r}."
            )
        self._validate_temporal_protocol(protocol)
        self._require_temporal_qualification(protocol)
        self._validate_primary_checkpoint_contract()
        if self._native_negative_dependency_preflight is None:
            self.record_native_negative_dependency_preflight(
                validate_wan_native_negative_prompt_cleaner()
            )
        self._native_pipeline_temporal_configured = True
        return {
            "schema_version": 1,
            "num_frames": int(protocol["native_segment_frames"]),
            "native_fps": int(protocol["native_fps"]),
            "first_call": {
                "model_id": WAN_T2V_MODEL_ID,
                "revision": WAN_T2V_REVISION,
                "num_frames": int(protocol["native_segment_frames"]),
                "fps": int(protocol["native_fps"]),
            },
            "completion_hook": "WanAdapter.complete_native_pipeline_temporal_protocol",
            "completion_hook_required_kwargs": [
                "prompt",
                "negative_prompt",
                "generator",
                "num_inference_steps",
                "guidance_scale",
                "height",
                "width",
            ],
            "output_frames": int(protocol["output_frames"]),
            "output_fps": int(protocol["output_fps"]),
            "duration_seconds": float(protocol["duration_seconds"]),
            "one_way_model_transition": True,
        }

    def record_native_negative_dependency_preflight(self, record: Mapping[str, Any]) -> None:
        """Bind an authenticated CPU dependency record to native temporal provenance."""

        if record.get("package_name") != "ftfy" or record.get("package_version") != WAN_FTFY_VERSION:
            raise RuntimeError("Wan native-negative dependency provenance is not pinned ftfy.")
        if record.get("status") != "verified_before_native_negative_model_allocation":
            raise RuntimeError("Wan native-negative dependency provenance has the wrong status.")
        diffusers_install = record.get("diffusers_install")
        if not isinstance(diffusers_install, Mapping):
            raise RuntimeError("Wan native-negative provenance lacks Diffusers VCS identity.")
        if (
            diffusers_install.get("repository") != WAN_DIFFUSERS_REPOSITORY
            or diffusers_install.get("revision") != WAN_DIFFUSERS_REVISION
            or diffusers_install.get("editable") is not False
        ):
            raise RuntimeError("Wan native-negative provenance has untrusted Diffusers identity.")
        try:
            copied = json.loads(json.dumps(dict(record), sort_keys=True))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Wan native-negative dependency provenance is not JSON-safe.") from exc
        self._native_negative_dependency_preflight = copied

    def complete_native_pipeline_temporal_protocol(
        self,
        first_segment_videos: Any,
        *,
        prompt: str,
        negative_prompt: str,
        generator: torch.Generator | None,
        num_inference_steps: int,
        guidance_scale: float,
        height: int,
        width: int,
        guidance_scale_2: float | None = None,
    ) -> list[list[Any]]:
        """Complete a native-negative 81-frame Wan call to exact 15 seconds."""

        if not self._native_pipeline_temporal_configured:
            raise RuntimeError(
                "Call configure_native_pipeline_for_temporal_protocol before native completion."
            )
        if self._native_negative_dependency_preflight is None:
            raise RuntimeError("Wan native completion lacks dependency-preflight provenance.")
        protocol = self._temporal_protocol()
        assert protocol is not None
        segments = [
            _validate_video_batch(first_segment_videos, int(protocol["native_segment_frames"]))
        ]
        if len(segments[0]) != 1:
            raise ValueError("The validated Wan native temporal hook currently requires one video.")
        base_seed = _generator_initial_seed(generator)
        segment_seeds = [base_seed]
        continuation_prompt_fingerprints: list[dict[str, Any]] = []
        for segment_index in range(1, int(protocol["segment_count"])):
            if self._pipeline_role != "i2v":
                self._load_continuation_pipeline()
            assert self.pipeline is not None
            seed = _continuation_seed(base_seed, segment_index)
            segment_seeds.append(seed)
            continuation_prompt_fingerprints.append(
                {
                    "segment_index": segment_index,
                    "call_role": "official_wan_i2v_continuation",
                    "model_id": WAN_I2V_MODEL_ID,
                    "revision": WAN_I2V_REVISION,
                    "seed": seed,
                    "positive": _wan_prompt_cleaner_fingerprint(prompt),
                    "negative": _wan_prompt_cleaner_fingerprint(negative_prompt),
                }
            )
            segment_generator = _make_generator(seed, self.device)
            call_kwargs = {
                "image": segments[-1][0][-1],
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "height": int(height),
                "width": int(width),
                "num_frames": int(protocol["native_segment_frames"]),
                "num_inference_steps": int(num_inference_steps),
                "guidance_scale": float(guidance_scale),
                "guidance_scale_2": (
                    float(guidance_scale_2) if guidance_scale_2 is not None else None
                ),
                "num_videos_per_prompt": 1,
                "generator": segment_generator,
                "output_type": "pil",
                "return_dict": True,
            }
            with torch.inference_mode():
                output = self.pipeline(
                    **{key: value for key, value in call_kwargs.items() if value is not None}
                )
            media = output.frames if hasattr(output, "frames") else output[0]
            segments.append(_validate_video_batch(media, int(protocol["native_segment_frames"])))
        stitched, provenance = self._stitch_segments(segments, protocol)
        provenance["generation_path"] = "native_negative_t2v_then_two_native_i2v_calls"
        provenance["segment_seeds"] = segment_seeds
        provenance["negative_prompt_sha256"] = _text_sha256(negative_prompt)
        provenance["prompt_sha256"] = _text_sha256(prompt)
        provenance["continuation_prompt_cleaner_fingerprints"] = (
            continuation_prompt_fingerprints
        )
        provenance["native_negative_dependency_preflight"] = dict(
            self._native_negative_dependency_preflight
        )
        self._last_temporal_provenance = provenance
        return stitched

    def _start_i2v_segment(
        self,
        *,
        anchors: list[Any],
        state: AdapterState,
        segment_index: int,
    ) -> torch.Tensor:
        if len(anchors) != 1:
            raise ValueError(
                "The validated Wan continuation protocol currently requires one anchor."
            )
        if self._pipeline_role != "i2v":
            self._load_continuation_pipeline()
        assert self.pipeline is not None
        protocol = self._temporal_protocol()
        assert protocol is not None
        seed = _continuation_seed(int(state.extra["base_seed"]), segment_index)
        self._segment_seeds.append(seed)
        self._segment_prediction_calls.append(0)
        self._segment_prompt_digests.append(set())
        segment_generator = _make_generator(seed, self.device)
        image = self.pipeline.video_processor.preprocess(
            anchors[0],
            height=int(state.extra["height"]),
            width=int(state.extra["width"]),
        ).to(device=self.device, dtype=torch.float32)
        output = self.pipeline.prepare_latents(
            image=image,
            batch_size=1,
            num_channels_latents=int(self.pipeline.vae.config.z_dim),
            height=int(state.extra["height"]),
            width=int(state.extra["width"]),
            num_frames=int(protocol["native_segment_frames"]),
            dtype=torch.float32,
            device=self.device,
            generator=segment_generator,
            latents=None,
            last_image=None,
        )
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError(
                "Wan2.2-I2V-A14B prepare_latents must return (noise_latents, image_condition)."
            )
        latents, image_condition = output
        if int(latents.shape[1]) != 16 or int(image_condition.shape[1]) != 20:
            raise RuntimeError(
                "Unexpected Wan I2V latent/condition channels: "
                f"latents={tuple(latents.shape)}, condition={tuple(image_condition.shape)}."
            )
        state.extra["wan_i2v_condition"] = image_condition
        state.extra["wan_i2v_anchor_pixel_sha256"] = _frame_record(anchors[0])["pixel_sha256"]
        self._reset_continuation_scheduler()
        return latents

    def _load_continuation_pipeline(self) -> None:
        if self._pipeline_role == "i2v":
            return
        assert self.pipeline is not None
        self._verify_pinned_shared_component_artifacts(
            role="continuation",
            repo_id=WAN_I2V_MODEL_ID,
            revision=WAN_I2V_REVISION,
        )
        old_pipeline = self.pipeline
        shared = {
            "text_encoder": old_pipeline.text_encoder,
            "tokenizer": old_pipeline.tokenizer,
            "vae": old_pipeline.vae,
        }
        if hasattr(old_pipeline, "remove_all_hooks"):
            old_pipeline.remove_all_hooks()
        # The A14B T2V and I2V conversions each contain two ~14B experts. They
        # cannot coexist on a 94-GB GPU, so the transition deliberately drops
        # both completed T2V denoisers while preserving byte-identical shared
        # text/VAE components.
        old_transformer = getattr(old_pipeline, "transformer", None)
        old_transformer_2 = getattr(old_pipeline, "transformer_2", None)
        old_pipeline.transformer = None
        old_pipeline.transformer_2 = None
        del old_transformer, old_transformer_2
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            import diffusers
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Wan continuation requires Diffusers WanImageToVideoPipeline."
            ) from exc
        pipeline_cls = getattr(diffusers, "WanImageToVideoPipeline", None)
        if pipeline_cls is None:
            raise RuntimeError("Installed diffusers does not expose WanImageToVideoPipeline.")
        load_kwargs = dict(self.config.get("load_kwargs", {}))
        load_kwargs.pop("single_file", None)
        load_kwargs.pop("single_file_components", None)
        load_kwargs.pop("device_map", None)
        load_kwargs.update(shared)
        load_kwargs.update(
            {
                "revision": WAN_I2V_REVISION,
                "torch_dtype": self.dtype,
                "local_files_only": bool(self.config.get("local_files_only", False)),
                "low_cpu_mem_usage": True,
            }
        )
        self.pipeline = pipeline_cls.from_pretrained(WAN_I2V_MODEL_ID, **load_kwargs)
        for component_name, component in shared.items():
            if getattr(self.pipeline, component_name, None) is not component:
                raise RuntimeError(
                    "Wan continuation did not preserve the verified primary "
                    f"{component_name} object while swapping denoisers."
                )
        configure_pipeline_vae_tiling(self.pipeline, self.config.get("vae_tiling"))
        cpu_offload = self.config.get("cpu_offload", False)
        offload_strategy = self._cpu_offload_strategy(cpu_offload)
        if offload_strategy == "group":
            self._enable_group_offload(cpu_offload)
        elif offload_strategy == "sequential" and hasattr(
            self.pipeline, "enable_sequential_cpu_offload"
        ):
            self.pipeline.enable_sequential_cpu_offload(device=self.device)
        elif offload_strategy == "model" and hasattr(self.pipeline, "enable_model_cpu_offload"):
            self.pipeline.enable_model_cpu_offload(device=self.device)
        elif hasattr(self.pipeline, "to"):
            self.pipeline.to(self.device)
        self._freeze_pipeline()
        self._pipeline_role = "i2v"
        self._validate_continuation_checkpoint_contract()

    def _reset_continuation_scheduler(self) -> None:
        assert self.pipeline is not None
        steps = len(self._native_segment_timesteps)
        if steps <= 0:
            raise RuntimeError("Wan continuation scheduler has no frozen native timestep plan.")
        self.pipeline.scheduler.set_timesteps(steps, device=self.device)
        if hasattr(self.pipeline.scheduler, "set_begin_index"):
            self.pipeline.scheduler.set_begin_index(0)
        observed = list(self.pipeline.scheduler.timesteps)
        if len(observed) != steps or any(
            not math.isclose(_as_float(left), _as_float(right), rel_tol=0.0, abs_tol=1.0e-6)
            for left, right in zip(observed, self._native_segment_timesteps)
        ):
            raise RuntimeError(
                "Pinned Wan T2V and I2V schedulers did not produce identical timestep sequences."
            )

    def _decode_native_segment(
        self,
        latents: torch.Tensor,
        protocol: dict[str, Any] | None,
    ) -> list[list[Any]]:
        assert self.pipeline is not None
        latents = latents.to(self.pipeline.vae.dtype)
        latents_mean = (
            torch.tensor(self.pipeline.vae.config.latents_mean)
            .view(1, self.pipeline.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.pipeline.vae.config.latents_std).view(
            1, self.pipeline.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        decoded_latents = latents / latents_std + latents_mean
        video = self.pipeline.vae.decode(decoded_latents, return_dict=False)[0]
        if not hasattr(self.pipeline, "video_processor"):
            raise RuntimeError("Wan pipeline does not expose video postprocessing.")
        frames = self.pipeline.video_processor.postprocess_video(video, output_type="pil")
        if protocol is None:
            if not isinstance(frames, list) or any(not isinstance(item, list) for item in frames):
                raise RuntimeError("Wan native decode did not return a batch of frame lists.")
            return frames
        return _validate_video_batch(frames, int(protocol["native_segment_frames"]))

    def _append_completed_segment(
        self,
        segment: list[list[Any]],
        protocol: dict[str, Any],
    ) -> None:
        _validate_video_batch(segment, int(protocol["native_segment_frames"]))
        if self._completed_segments and len(segment) != len(self._completed_segments[0]):
            raise RuntimeError("Wan segment batch size changed during continuation.")
        self._completed_segments.append(segment)

    def _stitch_segments(
        self,
        segments: Sequence[list[list[Any]]],
        protocol: dict[str, Any],
    ) -> tuple[list[list[Any]], dict[str, Any]]:
        expected_segments = int(protocol["segment_count"])
        native_frames = int(protocol["native_segment_frames"])
        if len(segments) != expected_segments:
            raise RuntimeError(
                f"Wan temporal protocol requires {expected_segments} segments; got {len(segments)}."
            )
        validated = [_validate_video_batch(segment, native_frames) for segment in segments]
        batch_size = len(validated[0])
        if any(len(segment) != batch_size for segment in validated):
            raise RuntimeError("Wan segment batch sizes differ.")
        outputs: list[list[Any]] = []
        video_records: list[dict[str, Any]] = []
        for batch_index in range(batch_size):
            videos = [segment[batch_index] for segment in validated]
            stitched_inclusive = list(videos[0])
            for video in videos[1:]:
                stitched_inclusive.extend(video[1:])
            expected_inclusive = int(protocol["stitched_inclusive_frames"])
            if len(stitched_inclusive) != expected_inclusive:
                raise RuntimeError(
                    f"Wan stitch produced {len(stitched_inclusive)} frames; expected {expected_inclusive}."
                )
            output = stitched_inclusive[: int(protocol["output_frames"])]
            if len(output) != int(protocol["output_frames"]):
                raise RuntimeError("Wan endpoint crop did not produce exactly 240 frames.")
            seams = [
                _seam_record(videos[index], videos[index + 1], self._seam_gate_config())
                for index in range(len(videos) - 1)
            ]
            unique_ratios = [_unique_frame_ratio(video) for video in videos]
            minimum_unique = float(self._seam_gate_config()["min_unique_frame_ratio"])
            if any(ratio < minimum_unique for ratio in unique_ratios):
                raise RuntimeError(
                    "Wan pilot freeze gate failed: per-segment unique-frame ratios "
                    f"{unique_ratios} are below {minimum_unique}."
                )
            outputs.append(output)
            video_records.append(
                {
                    "batch_index": batch_index,
                    "native_segment_frame_counts": [len(video) for video in videos],
                    "native_segment_unique_frame_ratios": unique_ratios,
                    "seams": seams,
                    "inclusive_frame_count": len(stitched_inclusive),
                    "output_frame_count": len(output),
                    "output_frame_pixel_sha256": [
                        _frame_record(frame)["pixel_sha256"] for frame in output
                    ],
                }
            )
        provenance = self._configured_temporal_provenance(protocol)
        provenance.update(
            {
                "status": "completed_and_automated_pilot_gates_passed",
                "direct_240_frame_denoising_used": False,
                "native_denoised_frames_per_segment": native_frames,
                "total_native_denoising_segments": expected_segments,
                "video_records": video_records,
            }
        )
        return outputs, provenance

    def _seam_gate_config(self) -> dict[str, Any]:
        protocol = self._temporal_protocol() or {}
        raw = dict(protocol.get("pilot_gates") or {})
        return {
            "max_conditioning_reconstruction_normalized_mae": float(
                raw.get("max_conditioning_reconstruction_normalized_mae", 0.20)
            ),
            "min_conditioning_reconstruction_psnr_db": float(
                raw.get("min_conditioning_reconstruction_psnr_db", 12.0)
            ),
            "max_first_motion_step_normalized_mae": float(
                raw.get("max_first_motion_step_normalized_mae", 0.35)
            ),
            "min_unique_frame_ratio": float(raw.get("min_unique_frame_ratio", 0.95)),
            "manual_visual_review_required": bool(raw.get("manual_visual_review_required", True)),
        }

    def _validate_temporal_shared_component_load_contract(self) -> None:
        """Reject paths that would sever source-artifact-to-component provenance.

        The temporal protocol reuses the primary pipeline's text encoder,
        tokenizer, and VAE in the continuation pipeline.  User-supplied objects
        or a single-file load would make it impossible to prove that those
        in-memory components came from the exact pinned repository artifacts.
        """

        if self.model_id != WAN_T2V_MODEL_ID:
            raise RuntimeError(
                f"Wan adapter is pinned to {WAN_T2V_MODEL_ID!r}; got {self.model_id!r}."
            )
        if str(self.config.get("revision")) != WAN_T2V_REVISION:
            raise RuntimeError(
                f"Wan T2V revision must be pinned to {WAN_T2V_REVISION}; "
                f"got {self.config.get('revision')!r}."
            )
        load_kwargs = dict(self.config.get("load_kwargs") or {})
        forbidden = sorted(
            key
            for key in ("text_encoder", "tokenizer", "vae", "single_file", "single_file_components")
            if load_kwargs.get(key) is not None
        )
        if forbidden:
            raise RuntimeError(
                "Wan temporal shared-component provenance requires components loaded from "
                f"the exact pinned T2V snapshot; unsupported overrides: {forbidden}."
            )

    def _verify_pinned_shared_component_artifacts(
        self,
        *,
        role: str,
        repo_id: str,
        revision: str,
    ) -> None:
        expected_source = {
            "primary": (WAN_T2V_MODEL_ID, WAN_T2V_REVISION),
            "continuation": (WAN_I2V_MODEL_ID, WAN_I2V_REVISION),
        }.get(role)
        if expected_source is None:
            raise ValueError(f"Unknown Wan shared-component source role {role!r}.")
        # A failed re-verification must not leave a stale successful record
        # available to later provenance calls.
        self._shared_component_artifact_verification.pop(role, None)
        if (repo_id, revision) != expected_source:
            raise RuntimeError(
                f"Wan {role} shared-component verification source drifted: "
                f"expected {expected_source}, got {(repo_id, revision)}."
            )
        load_kwargs = dict(self.config.get("load_kwargs") or {})
        token = load_kwargs.get("token", os.environ.get("HF_TOKEN"))
        record = _verify_wan_shared_component_snapshot(
            repo_id=repo_id,
            revision=revision,
            expected_sha256=WAN_SHARED_COMPONENT_WEIGHT_SHA256,
            local_files_only=bool(
                load_kwargs.get(
                    "local_files_only",
                    self.config.get("local_files_only", False),
                )
            ),
            cache_dir=load_kwargs.get("cache_dir"),
            token=token,
        )
        # Publish the role only after every artifact has passed.  A partial
        # record must never look like verified shared-component identity.
        self._shared_component_artifact_verification[role] = record

    def _shared_component_identity_provenance(self) -> dict[str, Any]:
        roles = json.loads(json.dumps(self._shared_component_artifact_verification))
        verified_roles = sorted(roles)
        complete = verified_roles == ["continuation", "primary"]
        status = (
            "verified_both_pinned_sources_byte_identical"
            if complete
            else "primary_verified_continuation_pending"
            if verified_roles == ["primary"]
            else "not_verified"
        )
        provenance: dict[str, Any] = {
            "schema_version": 1,
            "status": status,
            "scope": (
                "SHA-256 identity of exact on-disk text-encoder, tokenizer, and VAE source "
                "artifacts; temporal loads reject component overrides, and the continuation "
                "pipeline must retain the primary in-memory component objects"
            ),
            "verification_method": (
                "streamed SHA-256 with immutable Hugging Face blob-name binding and a "
                "device/inode/size/mtime/ctime-bound persistent cache"
            ),
            "expected_sha256_by_artifact": dict(WAN_SHARED_COMPONENT_WEIGHT_SHA256),
            "required_roles": ["primary", "continuation"],
            "verified_roles": verified_roles,
            "sources": roles,
        }
        if complete:
            provenance["verified_shared_component_weight_sha256"] = dict(
                WAN_SHARED_COMPONENT_WEIGHT_SHA256
            )
        return provenance

    def _configured_temporal_provenance(self, protocol: dict[str, Any]) -> dict[str, Any]:
        qualification = validate_temporal_production_gate(
            protocol,
            model_name="wan22_t2v_a14b",
            model_revision=WAN_T2V_REVISION,
            criteria_names=_WAN_PRODUCTION_CRITERIA,
        )
        provenance = {
            "schema_version": 1,
            "status": "configured_not_yet_completed",
            "protocol": dict(protocol),
            "checkpoint_sources": {
                "primary": {"model_id": WAN_T2V_MODEL_ID, "revision": WAN_T2V_REVISION},
                "continuation": {"model_id": WAN_I2V_MODEL_ID, "revision": WAN_I2V_REVISION},
            },
            "shared_component_artifact_identity": self._shared_component_identity_provenance(),
            "seam_pilot_gates": self._seam_gate_config(),
            "scientific_label": (
                "composite official-family Wan2.2 T2V+I2V continuation; not a native "
                "15-second T2V-A14B trajectory"
            ),
            "manual_visual_review_required": bool(
                self._seam_gate_config()["manual_visual_review_required"]
            ),
            "qualification": {
                "mode": "production_approved" if qualification else "explicit_pilot_only",
                "pilot_environment_variable": WAN_TEMPORAL_PILOT_ENV,
                "production_gate_validation": qualification,
            },
        }
        if self._native_negative_dependency_preflight is not None:
            provenance["native_negative_dependency_preflight"] = dict(
                self._native_negative_dependency_preflight
            )
        return provenance

    def _validate_primary_checkpoint_contract(self) -> None:
        if self.model_id != WAN_T2V_MODEL_ID:
            raise RuntimeError(
                f"Wan adapter is pinned to {WAN_T2V_MODEL_ID!r}; got {self.model_id!r}."
            )
        if str(self.config.get("revision")) != WAN_T2V_REVISION:
            raise RuntimeError(
                f"Wan T2V revision must be pinned to {WAN_T2V_REVISION}; "
                f"got {self.config.get('revision')!r}."
            )
        assert self.pipeline is not None
        in_channels = int(self.pipeline.transformer.config.in_channels)
        if in_channels != 16:
            raise RuntimeError(
                f"Pinned Wan T2V transformer must have 16 input channels; got {in_channels}."
            )
        if getattr(self.pipeline.vae, "dtype", None) != torch.float32:
            raise RuntimeError("Pinned Wan generation requires a torch.float32 AutoencoderKLWan.")

    def _validate_continuation_checkpoint_contract(self) -> None:
        assert self.pipeline is not None
        if self.pipeline.__class__.__name__ != "WanImageToVideoPipeline":
            raise RuntimeError(
                "Wan continuation must load the official Diffusers WanImageToVideoPipeline."
            )
        in_channels = int(self.pipeline.transformer.config.in_channels)
        out_channels = int(self.pipeline.transformer.config.out_channels)
        image_dim = getattr(self.pipeline.transformer.config, "image_dim", None)
        if (in_channels, out_channels, image_dim) != (36, 16, None):
            raise RuntimeError(
                "Pinned Wan2.2 I2V transformer contract changed: expected "
                f"(in=36,out=16,image_dim=None), got {(in_channels, out_channels, image_dim)}."
            )
        if getattr(self.pipeline.vae, "dtype", None) != torch.float32:
            raise RuntimeError(
                "Pinned Wan continuation requires the shared torch.float32 AutoencoderKLWan."
            )

    def _validate_temporal_protocol(self, protocol: dict[str, Any]) -> None:
        allowed = {
            *_EXPECTED_TEMPORAL_PROTOCOL,
            "pilot_gates",
            "execution_phase",
            "production_gate",
        }
        unknown = sorted(set(protocol) - allowed)
        if unknown:
            raise ValueError(f"Unknown model.{WAN_TEMPORAL_PROTOCOL_KEY} keys: {unknown}.")
        for key, expected in _EXPECTED_TEMPORAL_PROTOCOL.items():
            if protocol.get(key) != expected:
                raise ValueError(
                    f"Invalid {WAN_TEMPORAL_PROTOCOL_KEY}.{key}: expected {expected!r}, "
                    f"got {protocol.get(key)!r}."
                )
        gates = self._seam_gate_config()
        if not 0.0 <= gates["max_conditioning_reconstruction_normalized_mae"] <= 1.0:
            raise ValueError("Wan reconstruction MAE gate must be in [0,1].")
        if gates["min_conditioning_reconstruction_psnr_db"] <= 0:
            raise ValueError("Wan reconstruction PSNR gate must be positive.")
        if not 0.0 <= gates["max_first_motion_step_normalized_mae"] <= 1.0:
            raise ValueError("Wan first-motion MAE gate must be in [0,1].")
        if not 0.0 < gates["min_unique_frame_ratio"] <= 1.0:
            raise ValueError("Wan unique-frame ratio gate must be in (0,1].")
        raw_gates = protocol.get("pilot_gates")
        if not isinstance(raw_gates, Mapping):
            raise ValueError("Wan temporal protocol requires a pilot_gates mapping.")
        allowed_gates = {
            "max_conditioning_reconstruction_normalized_mae",
            "min_conditioning_reconstruction_psnr_db",
            "max_first_motion_step_normalized_mae",
            "min_unique_frame_ratio",
            "manual_visual_review_required",
        }
        unknown_gates = sorted(set(raw_gates) - allowed_gates)
        if unknown_gates:
            raise ValueError(f"Unknown Wan pilot_gates keys: {unknown_gates}.")
        if not gates["manual_visual_review_required"]:
            raise ValueError("Wan temporal protocol cannot disable manual visual pilot review.")
        if protocol.get("execution_phase") not in {"pilot", "production"}:
            raise ValueError("Wan temporal execution_phase must be 'pilot' or 'production'.")
        validate_temporal_production_gate(
            protocol,
            model_name="wan22_t2v_a14b",
            model_revision=WAN_T2V_REVISION,
            criteria_names=_WAN_PRODUCTION_CRITERIA,
        )

    def _require_temporal_qualification(self, protocol: Mapping[str, Any]) -> None:
        qualification = validate_temporal_production_gate(
            protocol,
            model_name="wan22_t2v_a14b",
            model_revision=WAN_T2V_REVISION,
            criteria_names=_WAN_PRODUCTION_CRITERIA,
        )
        if qualification is not None:
            return
        if os.environ.get(WAN_TEMPORAL_PILOT_ENV) != "1":
            raise RuntimeError(
                "Wan 15-second continuation is pilot-only and not production-approved. "
                f"A single qualification job must explicitly set {WAN_TEMPORAL_PILOT_ENV}=1. "
                "Bulk manifests are refused until model.wan_temporal_protocol."
                "production_gate opens and validates the immutable 54-run evidence manifest."
            )

    def _validate_requested_output_contract(
        self,
        generation_kwargs: Mapping[str, Any],
        protocol: dict[str, Any],
    ) -> None:
        requested = {
            "num_frames": int(generation_kwargs.get("num_frames", -1)),
            "fps": int(generation_kwargs.get("fps", -1)),
            "duration_seconds": float(generation_kwargs.get("duration_seconds", -1.0)),
        }
        expected = {
            "num_frames": int(protocol["output_frames"]),
            "fps": int(protocol["output_fps"]),
            "duration_seconds": float(protocol["duration_seconds"]),
        }
        if requested != expected:
            raise ValueError(
                "Wan temporal protocol is frozen to exact 240 frames / 16 fps / 15 seconds; "
                f"got {requested}."
            )

    def _select_transformer(self, timestep: Any) -> Any:
        assert self.pipeline is not None
        if not hasattr(self.pipeline, "transformer_2"):
            return self.pipeline.transformer
        if not self.timesteps:
            raise NotImplementedError(
                "Wan dual-transformer selection requires scheduler timesteps; call set_timesteps first."
            )
        boundary_ratio = float(getattr(self.pipeline, "boundary_ratio"))
        current = _as_float(timestep)
        max_timestep = max(_as_float(item) for item in self.timesteps)
        normalized = current / max(max_timestep, 1.0e-6)
        return (
            self.pipeline.transformer
            if normalized >= boundary_ratio
            else self.pipeline.transformer_2
        )

    def _temporal_protocol(self) -> dict[str, Any] | None:
        raw = self.config.get(WAN_TEMPORAL_PROTOCOL_KEY)
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError(f"model.{WAN_TEMPORAL_PROTOCOL_KEY} must be a mapping.")
        return dict(raw)

    def _record_segment_prediction(self, segment_index: int, prompt: str) -> None:
        if segment_index >= len(self._segment_prediction_calls):
            raise RuntimeError(f"Wan prediction arrived for uninitialized segment {segment_index}.")
        self._segment_prediction_calls[segment_index] += 1
        self._segment_prompt_digests[segment_index].add(_text_sha256(prompt))


def _verify_wan_shared_component_snapshot(
    *,
    repo_id: str,
    revision: str,
    expected_sha256: Mapping[str, str],
    local_files_only: bool,
    cache_dir: str | os.PathLike[str] | None,
    token: str | bool | None,
) -> dict[str, Any]:
    """Resolve and verify every declared shared artifact in one pinned snapshot.

    ``hf_hub_download`` supplies repository/revision resolution, while the
    canonical Hugging Face cache lineage and streamed SHA-256 establish the
    exact bytes.  Merely observing an LFS-style blob name is never treated as
    the first verification of its content.
    """

    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise RuntimeError("Wan shared-component verification requires huggingface_hub.") from exc

    if not expected_sha256:
        raise RuntimeError("Wan shared-component verification requires at least one artifact.")
    artifacts: dict[str, dict[str, Any]] = {}
    snapshot_roots: set[str] = set()
    for relative_name, expected_digest in sorted(expected_sha256.items()):
        if not _is_sha256(expected_digest):
            raise RuntimeError(
                f"Invalid declared Wan SHA-256 for {relative_name!r}: {expected_digest!r}."
            )
        relative = PurePosixPath(relative_name)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise RuntimeError(f"Invalid Wan shared-component relative path {relative_name!r}.")
        download_kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "filename": relative_name,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        if cache_dir is not None:
            download_kwargs["cache_dir"] = str(cache_dir)
        if token is not None:
            download_kwargs["token"] = token
        try:
            downloaded = hf_hub_download(**download_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Could not resolve pinned Wan artifact {repo_id}@{revision}:{relative_name}."
            ) from exc
        snapshot_path, resolved_blob, snapshot_root = _validate_hf_snapshot_artifact_path(
            downloaded,
            repo_id=repo_id,
            revision=revision,
            relative_name=relative_name,
            expected_sha256=expected_digest,
        )
        observed_digest, verification_mode, fingerprint = _verified_file_sha256(
            resolved_blob,
            expected_sha256=expected_digest,
        )
        if observed_digest != expected_digest:
            raise RuntimeError(
                f"Wan shared-component SHA-256 mismatch for {repo_id}@{revision}:"
                f"{relative_name}: expected {expected_digest}, observed {observed_digest}."
            )
        snapshot_roots.add(str(snapshot_root))
        artifacts[relative_name] = {
            "snapshot_path": str(snapshot_path),
            "resolved_blob_path": str(resolved_blob),
            "expected_sha256": expected_digest,
            "observed_sha256": observed_digest,
            "size_bytes": fingerprint["size"],
            "verification_mode": verification_mode,
            "stat_fingerprint": fingerprint,
        }
    if len(snapshot_roots) != 1:
        raise RuntimeError(
            f"Wan shared artifacts resolved across multiple snapshot roots: {sorted(snapshot_roots)}."
        )
    aggregate = hashlib.sha256()
    for relative_name, record in sorted(artifacts.items()):
        aggregate.update(relative_name.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(record["observed_sha256"].encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(str(record["size_bytes"]).encode("ascii"))
        aggregate.update(b"\n")
    return {
        "status": "all_declared_artifacts_sha256_verified",
        "repo_id": repo_id,
        "revision": revision,
        "snapshot_root": next(iter(snapshot_roots)),
        "artifact_count": len(artifacts),
        "artifact_set_sha256": aggregate.hexdigest(),
        "artifacts": artifacts,
    }


def _validate_hf_snapshot_artifact_path(
    downloaded: str | os.PathLike[str],
    *,
    repo_id: str,
    revision: str,
    relative_name: str,
    expected_sha256: str,
) -> tuple[Path, Path, Path]:
    """Require ``models--<repo>/snapshots/<commit>/<file> -> blobs/<sha256>``."""

    snapshot_path = Path(downloaded).expanduser().absolute()
    relative_parts = PurePosixPath(relative_name).parts
    snapshot_root = snapshot_path
    for _ in relative_parts:
        snapshot_root = snapshot_root.parent
    expected_repo_cache_name = "models--" + "--".join(repo_id.split("/"))
    if (
        snapshot_root.name != revision
        or snapshot_root.parent.name != "snapshots"
        or snapshot_root.parent.parent.name != expected_repo_cache_name
        or tuple(snapshot_path.parts[-len(relative_parts) :]) != relative_parts
    ):
        raise RuntimeError(
            "Wan shared artifact was not resolved from the exact pinned Hugging Face snapshot: "
            f"expected .../{expected_repo_cache_name}/snapshots/{revision}/{relative_name}, "
            f"got {snapshot_path}."
        )
    if not snapshot_path.is_symlink():
        raise RuntimeError(
            "Wan shared artifact is not an immutable Hugging Face snapshot-to-blob symlink: "
            f"{snapshot_path}."
        )
    try:
        resolved_blob = snapshot_path.resolve(strict=True)
        resolved_repo_root = snapshot_root.parent.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"Wan shared artifact symlink is invalid: {snapshot_path}.") from exc
    if (
        resolved_blob.parent != resolved_repo_root / "blobs"
        or resolved_blob.name != expected_sha256
        or not resolved_blob.is_file()
    ):
        raise RuntimeError(
            "Wan shared artifact did not resolve to its declared content-addressed blob: "
            f"expected {resolved_repo_root / 'blobs' / expected_sha256}, got {resolved_blob}."
        )
    return snapshot_path, resolved_blob, snapshot_root.resolve(strict=True)


def _verified_file_sha256(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[str, str, dict[str, int]]:
    """Hash a large immutable blob once, with stat-bound process/disk caches.

    The cache key includes device, inode, size, nanosecond mtime, and nanosecond
    ctime.  A normal edit therefore invalidates both caches even when the byte
    length and mtime are deliberately restored.  Every cache miss is streamed
    in chunks, and a pre/post stat comparison rejects concurrent mutation.
    """

    canonical = path.resolve(strict=True)
    fingerprint = _file_stat_fingerprint(canonical)
    memory_key = _wan_hash_memory_key(canonical, fingerprint, expected_sha256)
    with _WAN_HASH_MEMORY_CACHE_LOCK:
        cached = _WAN_HASH_MEMORY_CACHE.get(memory_key)
    if cached is not None:
        return cached, "process_stat_bound_sha256_cache", fingerprint

    cache_root = _wan_hash_cache_root()
    cache_record_key = hashlib.sha256(str(canonical).encode("utf-8")).hexdigest()
    try:
        cache_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        return _verified_file_sha256_with_disk_cache(
            canonical,
            expected_sha256=expected_sha256,
            cache_root=cache_root,
            cache_record_key=cache_record_key,
        )
    except OSError:
        # Cache availability is an optimization, never evidence.  A read-only
        # home/cache still receives a full streamed verification.
        observed, stable_fingerprint = _stream_and_verify_stable_file(canonical)
        if observed != expected_sha256:
            return observed, "streamed_sha256_cache_unavailable", stable_fingerprint
        key = _wan_hash_memory_key(canonical, stable_fingerprint, expected_sha256)
        with _WAN_HASH_MEMORY_CACHE_LOCK:
            _WAN_HASH_MEMORY_CACHE[key] = observed
        return observed, "streamed_sha256_cache_unavailable", stable_fingerprint


def _verified_file_sha256_with_disk_cache(
    path: Path,
    *,
    expected_sha256: str,
    cache_root: Path,
    cache_record_key: str,
) -> tuple[str, str, dict[str, int]]:
    import fcntl

    record_path = cache_root / f"{cache_record_key}.json"
    lock_path = cache_root / f"{cache_record_key}.lock"
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        fingerprint = _file_stat_fingerprint(path)
        memory_key = _wan_hash_memory_key(path, fingerprint, expected_sha256)
        with _WAN_HASH_MEMORY_CACHE_LOCK:
            cached = _WAN_HASH_MEMORY_CACHE.get(memory_key)
        if cached is not None:
            return cached, "process_stat_bound_sha256_cache", fingerprint
        record = _read_wan_hash_cache_record(record_path)
        if _wan_hash_cache_record_matches(
            record,
            path=path,
            expected_sha256=expected_sha256,
            fingerprint=fingerprint,
        ):
            observed = str(record["observed_sha256"])
            with _WAN_HASH_MEMORY_CACHE_LOCK:
                _WAN_HASH_MEMORY_CACHE[memory_key] = observed
            return observed, "persistent_stat_bound_sha256_cache", fingerprint

        observed, stable_fingerprint = _stream_and_verify_stable_file(path)
        if observed != expected_sha256:
            return observed, "streamed_sha256", stable_fingerprint
        record = {
            "schema_version": _WAN_HASH_CACHE_SCHEMA_VERSION,
            "canonical_path": str(path),
            "expected_sha256": expected_sha256,
            "observed_sha256": observed,
            "stat_fingerprint": stable_fingerprint,
        }
        temporary = cache_root / (f".{cache_record_key}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(record, handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, record_path)
        finally:
            temporary.unlink(missing_ok=True)
        memory_key = _wan_hash_memory_key(path, stable_fingerprint, expected_sha256)
        with _WAN_HASH_MEMORY_CACHE_LOCK:
            _WAN_HASH_MEMORY_CACHE[memory_key] = observed
        return observed, "streamed_sha256", stable_fingerprint


def _stream_and_verify_stable_file(path: Path) -> tuple[str, dict[str, int]]:
    before = _file_stat_fingerprint(path)
    observed = _stream_file_sha256(path)
    after = _file_stat_fingerprint(path)
    if before != after:
        raise RuntimeError(
            f"Wan shared artifact changed while its SHA-256 was being computed: {path}."
        )
    return observed, after


def _stream_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_stat_fingerprint(path: Path) -> dict[str, int]:
    stat_result = path.stat()
    if not path.is_file():
        raise RuntimeError(f"Wan shared artifact is not a regular file: {path}.")
    return {
        "device": int(stat_result.st_dev),
        "inode": int(stat_result.st_ino),
        "size": int(stat_result.st_size),
        "mtime_ns": int(stat_result.st_mtime_ns),
        "ctime_ns": int(stat_result.st_ctime_ns),
    }


def _wan_hash_memory_key(
    path: Path,
    fingerprint: Mapping[str, int],
    expected_sha256: str,
) -> tuple[Any, ...]:
    return (
        str(path),
        expected_sha256,
        *(int(fingerprint[key]) for key in ("device", "inode", "size", "mtime_ns", "ctime_ns")),
    )


def _wan_hash_cache_root() -> Path:
    configured = os.environ.get(WAN_SHARED_HASH_CACHE_ENV)
    if configured:
        return Path(configured).expanduser().absolute()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return base / "hierasafe_flow" / "wan_shared_artifact_sha256" / "v1"


def _read_wan_hash_cache_record(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, ValueError, TypeError):
        return None
    return record if isinstance(record, dict) else None


def _wan_hash_cache_record_matches(
    record: Mapping[str, Any] | None,
    *,
    path: Path,
    expected_sha256: str,
    fingerprint: Mapping[str, int],
) -> bool:
    return bool(
        record
        and record.get("schema_version") == _WAN_HASH_CACHE_SCHEMA_VERSION
        and record.get("canonical_path") == str(path)
        and record.get("expected_sha256") == expected_sha256
        and record.get("observed_sha256") == expected_sha256
        and record.get("stat_fingerprint") == dict(fingerprint)
    )


def _validate_native_frame_count(num_frames: int) -> None:
    if num_frames <= 0 or num_frames > 81 or (num_frames - 1) % 4:
        raise ValueError(
            "Wan2.2-T2V-A14B direct generation is limited to positive 1+4*k frame counts "
            f"at or below the official 81-frame/5-second contract; got {num_frames}. "
            f"Configure {WAN_TEMPORAL_PROTOCOL_KEY!r} for exact 15-second output."
        )


def _validate_video_batch(videos: Any, expected_frames: int) -> list[list[Any]]:
    if (
        not isinstance(videos, list)
        or not videos
        or any(not isinstance(video, list) for video in videos)
    ):
        raise RuntimeError("Wan video output must be a non-empty batch of frame lists.")
    counts = [len(video) for video in videos]
    if any(count != expected_frames for count in counts):
        raise RuntimeError(
            f"Wan native segment must contain exactly {expected_frames} frames; got {counts}."
        )
    for video in videos:
        records = [_frame_record(frame) for frame in video]
        reference = (records[0]["height"], records[0]["width"])
        if any((record["height"], record["width"]) != reference for record in records):
            raise RuntimeError("Wan changed frame dimensions within a native segment.")
    return videos


def _seam_record(
    previous: Sequence[Any],
    continuation: Sequence[Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    if len(previous) < 1 or len(continuation) < 2:
        raise RuntimeError(
            "Wan seam validation requires a prior endpoint and two continuation frames."
        )
    anchor = _frame_rgb_uint8(previous[-1])
    reconstruction = _frame_rgb_uint8(continuation[0])
    first_motion = _frame_rgb_uint8(continuation[1])
    if anchor.shape != reconstruction.shape or anchor.shape != first_motion.shape:
        raise RuntimeError(
            "Wan continuation changed spatial dimensions at a seam: "
            f"anchor={anchor.shape}, reconstruction={reconstruction.shape}, next={first_motion.shape}."
        )
    reconstruction_mae = _normalized_mae(anchor, reconstruction)
    reconstruction_psnr = _psnr_db(anchor, reconstruction)
    first_motion_mae = _normalized_mae(anchor, first_motion)
    if reconstruction_mae > float(gates["max_conditioning_reconstruction_normalized_mae"]):
        raise RuntimeError(
            "Wan conditioning reconstruction seam gate failed: normalized MAE "
            f"{reconstruction_mae:.6f} exceeds "
            f"{gates['max_conditioning_reconstruction_normalized_mae']}."
        )
    if reconstruction_psnr < float(gates["min_conditioning_reconstruction_psnr_db"]):
        raise RuntimeError(
            "Wan conditioning reconstruction seam gate failed: PSNR "
            f"{reconstruction_psnr:.3f} dB is below "
            f"{gates['min_conditioning_reconstruction_psnr_db']} dB."
        )
    if first_motion_mae > float(gates["max_first_motion_step_normalized_mae"]):
        raise RuntimeError(
            "Wan first continuation motion-step seam gate failed: normalized MAE "
            f"{first_motion_mae:.6f} exceeds {gates['max_first_motion_step_normalized_mae']}."
        )
    return {
        "status": "automated_gates_passed",
        "conditioning_anchor": _frame_record(previous[-1]),
        "decoded_conditioning_frame": _frame_record(continuation[0]),
        "first_retained_continuation_frame": _frame_record(continuation[1]),
        "conditioning_reconstruction_normalized_mae": reconstruction_mae,
        "conditioning_reconstruction_psnr_db": reconstruction_psnr,
        "first_motion_step_normalized_mae": first_motion_mae,
        "dropped_duplicate_frame_index": 0,
    }


def _frame_rgb_uint8(frame: Any) -> np.ndarray:
    array = np.asarray(frame.convert("RGB") if hasattr(frame, "convert") else frame)
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise RuntimeError(f"Wan frame has unsupported RGB shape {array.shape}.")
    if array.shape[2] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        if not np.isfinite(array).all():
            raise RuntimeError("Wan frame contains non-finite pixels.")
        if np.issubdtype(array.dtype, np.floating) and array.max(initial=0.0) <= 1.0:
            array = np.rint(array * 255.0)
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _frame_record(frame: Any) -> dict[str, Any]:
    array = _frame_rgb_uint8(frame)
    return {
        "height": int(array.shape[0]),
        "width": int(array.shape[1]),
        "pixel_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def _unique_frame_ratio(video: Sequence[Any]) -> float:
    digests = {_frame_record(frame)["pixel_sha256"] for frame in video}
    return len(digests) / max(len(video), 1)


def _normalized_mae(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.abs(left.astype(np.float32) - right.astype(np.float32)).mean() / 255.0)


def _psnr_db(left: np.ndarray, right: np.ndarray) -> float:
    mse = float(np.square(left.astype(np.float32) - right.astype(np.float32)).mean())
    # Keep provenance strict-JSON finite even for an exact reconstruction.
    return 120.0 if mse == 0.0 else float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def _continuation_seed(base_seed: int, segment_index: int) -> int:
    if segment_index not in {1, 2}:
        raise ValueError(f"Wan continuation segment index must be 1 or 2; got {segment_index}.")
    payload = f"wan22_i2v_continuation_v1|{int(base_seed)}|{segment_index}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)
    return value or segment_index


def _make_generator(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _generator_initial_seed(generator: torch.Generator | None) -> int:
    if generator is None:
        raise ValueError("Wan temporal generation requires an explicit seeded torch.Generator.")
    return int(generator.initial_seed())


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _as_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().flatten()[0].item())
    return float(value)
