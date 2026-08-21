from __future__ import annotations

from typing import Any

import numpy as np
import torch

from hierasafe_flow.adapters.base import AdapterState, DiffusersFrozenAdapter, PromptCondition


class QwenImageAdapter(DiffusersFrozenAdapter):
    adapter_name = "qwen_image"
    task_type = "text_to_image"
    pipeline_class_name = "QwenImagePipeline"
    latent_feature_dim = -1  # packed [B, image_tokens, D]
    required_components = ("transformer", "scheduler", "vae")
    encode_prompt_output_names = ("prompt_embeds", "prompt_embeds_mask")

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
        height = int(
            generation_kwargs.get("height")
            or self.pipeline.default_sample_size * self.pipeline.vae_scale_factor
        )
        width = int(
            generation_kwargs.get("width")
            or self.pipeline.default_sample_size * self.pipeline.vae_scale_factor
        )
        num_channels_latents = int(self.pipeline.transformer.config.in_channels) // 4
        latents = self.pipeline.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            self.dtype,
            self.device,
            generator,
            generation_kwargs.get("latents"),
        )
        state = AdapterState(extra={})
        state.extra["height"] = height
        state.extra["width"] = width
        state.extra["img_shapes"] = [
            [(1, height // self.pipeline.vae_scale_factor // 2, width // self.pipeline.vae_scale_factor // 2)]
        ] * batch_size
        state.extra["attention_kwargs"] = generation_kwargs.get("attention_kwargs") or {}
        return latents, state

    def set_timesteps(
        self,
        num_inference_steps: int,
        latents: torch.Tensor | None = None,
        state: AdapterState | None = None,
    ) -> list[Any]:
        self._require_loaded()
        del state
        if latents is None:
            raise NotImplementedError("Qwen Image set_timesteps requires packed latents to calculate shift.")
        assert self.pipeline is not None
        from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift, retrieve_timesteps

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        scheduler_config = self.pipeline.scheduler.config
        mu = calculate_shift(
            latents.shape[1],
            scheduler_config.get("base_image_seq_len", 256),
            scheduler_config.get("max_image_seq_len", 4096),
            scheduler_config.get("base_shift", 0.5),
            scheduler_config.get("max_shift", 1.15),
        )
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
        prompt_embeds = condition.data.get("prompt_embeds")
        if prompt_embeds is None:
            raise NotImplementedError("Qwen Image encode_prompt did not expose prompt_embeds.")
        guidance = None
        if getattr(self.pipeline.transformer.config, "guidance_embeds", False):
            guidance_scale = self.config.get("guidance_scale")
            if guidance_scale is None:
                raise ValueError("guidance_scale is required for guidance-distilled Qwen Image models.")
            guidance = torch.full([1], float(guidance_scale), device=self.device, dtype=torch.float32).expand(batch)
        candidates = {
            "hidden_states": latents,
            "sample": latents,
            "timestep": self._timestep_batch(timestep, batch).to(dtype=latents.dtype) / 1000,
            "guidance": guidance,
            "encoder_hidden_states": prompt_embeds,
            "encoder_hidden_states_mask": condition.data.get("prompt_embeds_mask"),
            "img_shapes": state.extra.get("img_shapes"),
            "attention_kwargs": state.extra.get("attention_kwargs", {}),
            "return_dict": True,
        }
        return self._call_transformer(self.pipeline.transformer, candidates)

    def decode_latents(self, latents: torch.Tensor, state: AdapterState) -> Any:
        self._require_loaded()
        assert self.pipeline is not None
        height = int(state.extra["height"])
        width = int(state.extra["width"])
        unpacked = self.pipeline._unpack_latents(latents, height, width, self.pipeline.vae_scale_factor)
        unpacked = unpacked.to(self.pipeline.vae.dtype)
        latents_mean = (
            torch.tensor(self.pipeline.vae.config.latents_mean)
            .view(1, self.pipeline.vae.config.z_dim, 1, 1, 1)
            .to(unpacked.device, unpacked.dtype)
        )
        latents_std = (
            1.0
            / torch.tensor(self.pipeline.vae.config.latents_std)
            .view(1, self.pipeline.vae.config.z_dim, 1, 1, 1)
            .to(unpacked.device, unpacked.dtype)
        )
        decoded_latents = unpacked / latents_std + latents_mean
        image = self.pipeline.vae.decode(decoded_latents, return_dict=False)[0][:, :, 0]
        return self.pipeline.image_processor.postprocess(image, output_type="pil")
