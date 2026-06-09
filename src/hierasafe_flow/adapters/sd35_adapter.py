from __future__ import annotations

from typing import Any

import torch

from hierasafe_flow.adapters.base import (
    AdapterState,
    DiffusersFrozenAdapter,
    PromptCondition,
)


class StableDiffusion35Adapter(DiffusersFrozenAdapter):
    adapter_name = "sd35"
    task_type = "text_to_image"
    pipeline_class_name = "StableDiffusion3Pipeline"
    required_components = ("transformer", "scheduler", "vae")
    encode_prompt_output_names = (
        "prompt_embeds",
        "negative_prompt_embeds",
        "pooled_prompt_embeds",
        "negative_pooled_prompt_embeds",
    )

    def predict_vector_field(
        self,
        latents: torch.Tensor,
        timestep: Any,
        condition: PromptCondition,
        state: AdapterState,
    ) -> torch.Tensor:
        self._require_loaded()
        del state
        assert self.pipeline is not None
        missing = [key for key in self.encode_prompt_output_names if key not in condition.data]
        if missing:
            raise NotImplementedError(f"SD3.5 condition is missing prompt fields: {missing}.")
        batch = latents.shape[0]
        guidance_scale = self._manual_guidance_scale()
        do_cfg = guidance_scale > 1.0 and condition.data.get("negative_prompt_embeds") is not None
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        prompt_embeds = condition.data["prompt_embeds"]
        pooled_prompt_embeds = condition.data["pooled_prompt_embeds"]
        if do_cfg:
            prompt_embeds = torch.cat([condition.data["negative_prompt_embeds"], prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(
                [condition.data["negative_pooled_prompt_embeds"], pooled_prompt_embeds],
                dim=0,
            )
        timestep_batch = self._timestep_batch(timestep, latent_model_input.shape[0])
        candidates = {
            "hidden_states": latent_model_input,
            "sample": latent_model_input,
            "timestep": timestep_batch,
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_prompt_embeds,
            "joint_attention_kwargs": None,
            "return_dict": True,
        }
        prediction = self._call_transformer(self.pipeline.transformer, candidates)
        if do_cfg:
            uncond, text = prediction.chunk(2)
            return uncond + guidance_scale * (text - uncond)
        return prediction
