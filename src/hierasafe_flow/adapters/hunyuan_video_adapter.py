from __future__ import annotations

from typing import Any

import torch

from hierasafe_flow.adapters.base import AdapterState, DiffusersFrozenAdapter, PromptCondition


class HunyuanVideoAdapter(DiffusersFrozenAdapter):
    adapter_name = "hunyuan_video"
    task_type = "text_to_video"
    pipeline_class_name = "HunyuanVideoPipeline"
    required_components = ("transformer", "scheduler", "vae")
    encode_prompt_output_names = ("prompt_embeds",)

    def predict_vector_field(
        self,
        latents: torch.Tensor,
        timestep: Any,
        condition: PromptCondition,
        state: AdapterState,
    ) -> torch.Tensor:
        del latents, timestep, condition, state
        raise NotImplementedError(
            "tencent/HunyuanVideo did not expose a public diffusers model_index.json from this "
            "environment. Implement HunyuanVideo vector-field prediction only after inspecting the installed "
            "HunyuanVideoPipeline source and verifying latent layout, transformer forward arguments, and scheduler step."
        )

