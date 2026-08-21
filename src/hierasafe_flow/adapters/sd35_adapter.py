from __future__ import annotations

from typing import Any

import torch

from hierasafe_flow.adapters.base import (
    AdapterState,
    DiffusersFrozenAdapter,
    PromptCondition,
)


def _trace_tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    work = tensor.detach().float()
    if not bool(torch.isfinite(work).all()):
        raise RuntimeError("SD3.5 native CFG produced a non-finite trace tensor.")
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "mean": float(work.mean().item()),
        "std": float(work.std(unbiased=False).item()),
        "min": float(work.min().item()),
        "max": float(work.max().item()),
        "norm": float(work.norm().item()),
    }


class StableDiffusion35Adapter(DiffusersFrozenAdapter):
    adapter_name = "sd35"
    task_type = "text_to_image"
    pipeline_class_name = "StableDiffusion3Pipeline"
    latent_feature_dim = 1  # [B, C, H, W]
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
        assert self.pipeline is not None
        missing = [key for key in self.encode_prompt_output_names if key not in condition.data]
        if missing:
            raise NotImplementedError(f"SD3.5 condition is missing prompt fields: {missing}.")
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
            combined = uncond + guidance_scale * (text - uncond)
            if bool(state.extra.get("_capture_sd35_native_cfg_trace", False)):
                cfg_delta = combined - text
                state.extra["_sd35_native_cfg_trace"] = {
                    "positive_prediction": _trace_tensor_stats(text),
                    "negative_prediction": _trace_tensor_stats(uncond),
                    "cfg_delta": _trace_tensor_stats(cfg_delta),
                    "combined_prediction": _trace_tensor_stats(combined),
                    "true_cfg_scale": guidance_scale,
                }
            else:
                state.extra.pop("_sd35_native_cfg_trace", None)
            return combined
        state.extra.pop("_sd35_native_cfg_trace", None)
        return prediction
