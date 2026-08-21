from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import torch

from hierasafe_flow.adapters.base import (
    AdapterState,
    DiffusersFrozenAdapter,
    PromptCondition,
    _signature_for_callable,
)


class FluxAdapter(DiffusersFrozenAdapter):
    adapter_name = "flux"
    task_type = "text_to_image"
    pipeline_class_name = "FluxPipeline"
    latent_feature_dim = -1  # packed [B, image_tokens, D]
    required_components = ("transformer", "scheduler", "vae")
    encode_prompt_output_names = ("prompt_embeds", "pooled_prompt_embeds", "text_ids")

    def set_timesteps(
        self,
        num_inference_steps: int,
        latents: torch.Tensor | None = None,
        state: AdapterState | None = None,
    ) -> list[Any]:
        """Reproduce the pinned ``FluxPipeline`` schedule exactly.

        FLUX.1-dev uses dynamic shifting, but its native pipeline does *not*
        call ``retrieve_timesteps`` with an unspecified sigma schedule.  It
        first constructs ``linspace(1, 1 / num_steps, num_steps)`` and only
        replaces it with ``None`` when the scheduler explicitly opts into
        ``use_flow_sigmas``.  The generic adapter path cannot encode that
        pipeline-specific contract and previously produced a different final
        timestep (and therefore a different image) at steering strength zero.
        """

        self._require_loaded()
        del state
        assert self.pipeline is not None
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive.")
        if latents is None:
            raise NotImplementedError(
                "Flux timestep setup requires packed latents to compute the native dynamic shift."
            )
        scheduler_config = getattr(self.pipeline.scheduler, "config", None)
        if scheduler_config is None or not hasattr(scheduler_config, "get"):
            raise NotImplementedError(
                "Flux native schedule equivalence requires a mapping-like scheduler config."
            )
        if not bool(scheduler_config.get("use_dynamic_shifting", False)):
            raise NotImplementedError(
                "FluxAdapter is qualified only for the pinned native dynamic-shifting schedule."
            )

        from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps

        sigmas: np.ndarray | None = np.linspace(
            1.0,
            1.0 / num_inference_steps,
            num_inference_steps,
        )
        if bool(scheduler_config.get("use_flow_sigmas", False)):
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            scheduler_config.get("base_image_seq_len", 256),
            scheduler_config.get("max_image_seq_len", 4096),
            scheduler_config.get("base_shift", 0.5),
            scheduler_config.get("max_shift", 1.15),
        )
        timesteps, returned_num_steps = retrieve_timesteps(
            self.pipeline.scheduler,
            num_inference_steps,
            self.device,
            sigmas=sigmas,
            mu=mu,
        )
        if int(returned_num_steps) != num_inference_steps or len(timesteps) != num_inference_steps:
            raise RuntimeError(
                "Pinned FluxPipeline schedule returned an unexpected number of timesteps."
            )
        if hasattr(self.pipeline.scheduler, "set_begin_index"):
            self.pipeline.scheduler.set_begin_index(0)
        self.timesteps = list(timesteps)
        return self.timesteps

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
        latent_channels = self.pipeline.transformer.config.in_channels // 4
        output = self.pipeline.prepare_latents(
            batch_size,
            latent_channels,
            int(generation_kwargs.get("height")),
            int(generation_kwargs.get("width")),
            self.dtype,
            self.device,
            generator,
            None,
        )
        if not (isinstance(output, tuple) and len(output) >= 2):
            raise NotImplementedError(
                "Flux prepare_latents must return packed latents and latent image ids."
            )
        latents, latent_image_ids = output[0], output[1]
        return latents, AdapterState(
            extra={
                "latent_image_ids": latent_image_ids,
                "height": int(generation_kwargs.get("height")),
                "width": int(generation_kwargs.get("width")),
            }
        )

    def predict_vector_field(
        self,
        latents: torch.Tensor,
        timestep: Any,
        condition: PromptCondition,
        state: AdapterState,
    ) -> torch.Tensor:
        self._require_loaded()
        assert self.pipeline is not None
        batch = latents.shape[0]
        model_latents = latents.to(dtype=self.dtype)
        guidance = float(self.config.get("guidance_scale", self.config.get("true_cfg_scale", 1.0)))
        guidance_tensor = torch.full((batch,), guidance, device=self.device, dtype=torch.float32)
        img_ids = state.extra.get("latent_image_ids")
        if img_ids is None:
            img_ids = state.extra.get("latent_ids")
        candidates = {
            "hidden_states": model_latents,
            "sample": model_latents,
            "timestep": self._timestep_batch(timestep, batch).to(dtype=self.dtype) / 1000,
            "img_ids": img_ids,
            "txt_ids": condition.data.get("text_ids"),
            "encoder_hidden_states": condition.data.get("prompt_embeds"),
            "pooled_projections": condition.data.get("pooled_prompt_embeds"),
            "guidance": guidance_tensor,
            "joint_attention_kwargs": None,
            "return_dict": True,
        }
        signature = _signature_for_callable(self.pipeline.transformer)
        required = [
            name
            for name, parameter in signature.parameters.items()
            if parameter.default is inspect.Parameter.empty
        ]
        if "img_ids" in required and candidates["img_ids"] is None:
            raise NotImplementedError(
                "Flux transformer requires img_ids, but prepare_latents did not expose latent image ids."
            )
        if "txt_ids" in required and candidates["txt_ids"] is None:
            raise NotImplementedError("Flux transformer requires txt_ids, but encode_prompt did not expose them.")
        return self._call_transformer(self.pipeline.transformer, candidates)

    def decode_latents(self, latents: torch.Tensor, state: AdapterState) -> Any:
        self._require_loaded()
        assert self.pipeline is not None
        height = state.extra.get("height")
        width = state.extra.get("width")
        if height is None or width is None:
            raise NotImplementedError("Flux decode requires original height and width in adapter state.")
        unpacked = self.pipeline._unpack_latents(latents, int(height), int(width), self.pipeline.vae_scale_factor)
        unpacked = (unpacked / self.pipeline.vae.config.scaling_factor) + self.pipeline.vae.config.shift_factor
        vae_dtype = next(self.pipeline.vae.parameters()).dtype
        unpacked = unpacked.to(dtype=vae_dtype)
        decoded = self.pipeline.vae.decode(unpacked, return_dict=False)[0]
        return self.pipeline.image_processor.postprocess(decoded, output_type="pil")


class Flux2Adapter(FluxAdapter):
    adapter_name = "flux2"
    pipeline_class_name = "Flux2Pipeline"
    latent_feature_dim = -1  # packed [B, image_tokens, D]
    encode_prompt_output_names = ("prompt_embeds", "text_ids")

    def set_timesteps(
        self,
        num_inference_steps: int,
        latents: torch.Tensor | None = None,
        state: AdapterState | None = None,
    ) -> list[Any]:
        self._require_loaded()
        del state
        assert self.pipeline is not None
        if latents is None:
            raise NotImplementedError("Flux2 timestep setup requires packed latents to compute empirical mu.")
        from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu, retrieve_timesteps

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        if getattr(getattr(self.pipeline.scheduler, "config", None), "use_flow_sigmas", False):
            sigmas = None
        mu = compute_empirical_mu(image_seq_len=latents.shape[1], num_steps=num_inference_steps)
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

    def predict_vector_field(
        self,
        latents: torch.Tensor,
        timestep: Any,
        condition: PromptCondition,
        state: AdapterState,
    ) -> torch.Tensor:
        self._require_loaded()
        assert self.pipeline is not None
        batch = latents.shape[0]
        model_latents = latents.to(dtype=self.dtype)
        img_ids = state.extra.get("latent_image_ids")
        if img_ids is None:
            img_ids = state.extra.get("latent_ids")
        guidance = float(self.config.get("guidance_scale", 4.0))
        guidance_tensor = torch.full((batch,), guidance, device=self.device, dtype=torch.float32)
        candidates = {
            "hidden_states": model_latents,
            "sample": model_latents,
            "timestep": self._timestep_batch(timestep, batch).to(dtype=self.dtype) / 1000,
            "encoder_hidden_states": condition.data.get("prompt_embeds"),
            "img_ids": img_ids,
            "txt_ids": condition.data.get("text_ids"),
            "guidance": guidance_tensor,
            "joint_attention_kwargs": None,
            "return_dict": True,
        }
        return self._call_transformer(self.pipeline.transformer, candidates)

    def decode_latents(self, latents: torch.Tensor, state: AdapterState) -> Any:
        self._require_loaded()
        assert self.pipeline is not None
        latent_ids = state.extra.get("latent_image_ids")
        if latent_ids is None:
            latent_ids = state.extra.get("latent_ids")
        if latent_ids is None:
            raise NotImplementedError("Flux2 decode requires latent ids from prepare_latents.")
        unpacked = self.pipeline._unpack_latents_with_ids(latents, latent_ids)
        bn_mean = self.pipeline.vae.bn.running_mean.view(1, -1, 1, 1).to(unpacked.device, unpacked.dtype)
        bn_std = torch.sqrt(
            self.pipeline.vae.bn.running_var.view(1, -1, 1, 1) + self.pipeline.vae.config.batch_norm_eps
        ).to(unpacked.device, unpacked.dtype)
        unpacked = unpacked * bn_std + bn_mean
        unpacked = self.pipeline._unpatchify_latents(unpacked)
        vae_dtype = next(self.pipeline.vae.parameters()).dtype
        unpacked = unpacked.to(dtype=vae_dtype)
        decoded = self.pipeline.vae.decode(unpacked, return_dict=False)[0]
        return self.pipeline.image_processor.postprocess(decoded, output_type="pil")
