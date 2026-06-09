from __future__ import annotations

import inspect
from typing import Any

import torch

from hierasafe_flow.adapters.base import AdapterState, DiffusersFrozenAdapter, PromptCondition


class FluxAdapter(DiffusersFrozenAdapter):
    adapter_name = "flux"
    task_type = "text_to_image"
    pipeline_class_name = "FluxPipeline"
    required_components = ("transformer", "scheduler", "vae")
    encode_prompt_output_names = ("prompt_embeds", "pooled_prompt_embeds", "text_ids")

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
        guidance = float(self.config.get("guidance_scale", self.config.get("true_cfg_scale", 1.0)))
        guidance_tensor = torch.full((batch,), guidance, device=self.device, dtype=torch.float32)
        img_ids = state.extra.get("latent_image_ids")
        if img_ids is None:
            img_ids = state.extra.get("latent_ids")
        candidates = {
            "hidden_states": latents,
            "sample": latents,
            "timestep": self._timestep_batch(timestep, batch).to(dtype=latents.dtype) / 1000,
            "img_ids": img_ids,
            "txt_ids": condition.data.get("text_ids"),
            "encoder_hidden_states": condition.data.get("prompt_embeds"),
            "pooled_projections": condition.data.get("pooled_prompt_embeds"),
            "guidance": guidance_tensor,
            "joint_attention_kwargs": None,
            "return_dict": True,
        }
        signature = inspect.signature(getattr(self.pipeline.transformer, "forward"))
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
        decoded = self.pipeline.vae.decode(unpacked, return_dict=False)[0]
        return self.pipeline.image_processor.postprocess(decoded, output_type="pil")


class Flux2Adapter(FluxAdapter):
    adapter_name = "flux2"
    pipeline_class_name = "Flux2Pipeline"
    encode_prompt_output_names = ("prompt_embeds",)

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
        img_ids = state.extra.get("latent_image_ids")
        if img_ids is None:
            img_ids = state.extra.get("latent_ids")
        candidates = {
            "hidden_states": latents,
            "sample": latents,
            "timestep": self._timestep_batch(timestep, batch),
            "encoder_hidden_states": condition.data.get("prompt_embeds"),
            "img_ids": img_ids,
            "txt_ids": condition.data.get("text_ids"),
            "return_dict": True,
        }
        return self._call_transformer(self.pipeline.transformer, candidates)
