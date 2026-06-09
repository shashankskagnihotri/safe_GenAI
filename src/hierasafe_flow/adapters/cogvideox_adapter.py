from __future__ import annotations

from typing import Any

import torch

from hierasafe_flow.adapters.base import AdapterState, DiffusersFrozenAdapter, PromptCondition


class CogVideoXAdapter(DiffusersFrozenAdapter):
    adapter_name = "cogvideox"
    task_type = "text_to_video"
    pipeline_class_name = "CogVideoXPipeline"
    required_components = ("transformer", "scheduler", "vae")
    encode_prompt_output_names = ("prompt_embeds", "negative_prompt_embeds")

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
        prompt_embeds = condition.data.get("prompt_embeds")
        if prompt_embeds is None:
            raise NotImplementedError("CogVideoX encode_prompt did not expose prompt_embeds.")
        batch = latents.shape[0]
        guidance_scale = self._manual_guidance_scale()
        do_cfg = guidance_scale > 1.0 and condition.data.get("negative_prompt_embeds") is not None
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        if do_cfg:
            prompt_embeds = torch.cat([condition.data["negative_prompt_embeds"], prompt_embeds], dim=0)
        candidates = {
            "hidden_states": latent_model_input,
            "sample": latent_model_input,
            "timestep": self._timestep_batch(timestep, latent_model_input.shape[0]),
            "encoder_hidden_states": prompt_embeds,
            "ofs": self.config.get("ofs"),
            "return_dict": True,
        }
        prediction = self._call_transformer(self.pipeline.transformer, candidates)
        if do_cfg:
            uncond, text = prediction.chunk(2)
            return uncond + guidance_scale * (text - uncond)
        return prediction

    def decode_latents(self, latents: torch.Tensor, state: AdapterState) -> Any:
        self._require_loaded()
        assert self.pipeline is not None
        if hasattr(self.pipeline, "decode_latents"):
            video = self.pipeline.decode_latents(latents)
            if hasattr(self.pipeline, "video_processor"):
                return self.pipeline.video_processor.postprocess_video(video=video, output_type="pil")
            return video.detach().float().cpu()
        return super().decode_latents(latents, state)
