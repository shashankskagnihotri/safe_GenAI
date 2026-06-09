from __future__ import annotations

from typing import Any

import torch

from hierasafe_flow.adapters.base import AdapterState, DiffusersFrozenAdapter, PromptCondition


class WanAdapter(DiffusersFrozenAdapter):
    adapter_name = "wan"
    task_type = "text_to_video"
    pipeline_class_name = "WanPipeline"
    required_components = ("transformer", "scheduler", "vae")
    encode_prompt_output_names = ("prompt_embeds", "negative_prompt_embeds")

    def _validate_components(self) -> None:
        super()._validate_components()
        assert self.pipeline is not None
        if hasattr(self.pipeline, "transformer_2") and not hasattr(self.pipeline, "boundary_ratio"):
            raise NotImplementedError(
                "WanPipeline exposes transformer_2 but no boundary_ratio; cannot correctly select "
                "the high-noise/low-noise transformer for steered vector-field prediction."
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
        transformer = self._select_transformer(timestep)
        prompt_embeds = condition.data.get("prompt_embeds")
        if prompt_embeds is None:
            raise NotImplementedError("Wan encode_prompt did not expose prompt_embeds.")
        batch = latents.shape[0]
        timestep_batch = self._timestep_batch(timestep, batch)
        guidance_scale = self._manual_guidance_scale()
        candidates = {
            "hidden_states": latents,
            "sample": latents,
            "timestep": timestep_batch,
            "encoder_hidden_states": prompt_embeds,
            "return_dict": True,
        }
        prediction = self._call_transformer(transformer, candidates)
        negative_prompt_embeds = condition.data.get("negative_prompt_embeds")
        if guidance_scale > 1.0 and negative_prompt_embeds is not None:
            uncond_candidates = dict(candidates)
            uncond_candidates["encoder_hidden_states"] = negative_prompt_embeds
            uncond = self._call_transformer(transformer, uncond_candidates)
            return uncond + guidance_scale * (prediction - uncond)
        return prediction

    def decode_latents(self, latents: torch.Tensor, state: AdapterState) -> Any:
        self._require_loaded()
        del state
        assert self.pipeline is not None
        latents = latents.to(self.pipeline.vae.dtype)
        latents_mean = (
            torch.tensor(self.pipeline.vae.config.latents_mean)
            .view(1, self.pipeline.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            1.0
            / torch.tensor(self.pipeline.vae.config.latents_std)
            .view(1, self.pipeline.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        decoded_latents = latents / latents_std + latents_mean
        video = self.pipeline.vae.decode(decoded_latents, return_dict=False)[0]
        if hasattr(self.pipeline, "video_processor"):
            return self.pipeline.video_processor.postprocess_video(video, output_type="pil")
        return video.detach().float().cpu()

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
        return self.pipeline.transformer if normalized >= boundary_ratio else self.pipeline.transformer_2


def _as_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().flatten()[0].item())
    return float(value)
