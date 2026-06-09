from __future__ import annotations

from typing import Any

import torch

from hierasafe_flow.adapters.base import AdapterState, DiffusersFrozenAdapter, PromptCondition


class LTXAdapter(DiffusersFrozenAdapter):
    adapter_name = "ltx"
    task_type = "text_to_video"
    pipeline_class_name = "LTXPipeline"
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
            "Lightricks/LTX-2.3 did not expose a public diffusers model_index.json from this "
            "environment. Implement LTX vector-field prediction only after inspecting the installed "
            "LTXPipeline source and verifying latent layout, transformer forward arguments, and scheduler step."
        )

