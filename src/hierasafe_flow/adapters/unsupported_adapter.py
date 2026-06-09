from __future__ import annotations

from typing import Any

import torch

from hierasafe_flow.adapters.base import AdapterState, FrozenGeneratorAdapter, PromptCondition


class UnsupportedVectorFieldAdapter(FrozenGeneratorAdapter):
    adapter_name = "unsupported"
    task_type = "unknown"
    pipeline_class_name = None
    unsupported_reason = "This model does not yet expose a verified vector-field steering adapter."

    def load(self) -> None:
        raise NotImplementedError(self.unsupported_reason)

    def prepare_prompt(self, prompt: str) -> PromptCondition:
        raise NotImplementedError(self.unsupported_reason)

    def prepare_initial_latents(
        self,
        prompt: str,
        batch_size: int,
        generator: torch.Generator | None,
        **generation_kwargs: Any,
    ) -> tuple[torch.Tensor, AdapterState]:
        raise NotImplementedError(self.unsupported_reason)

    def set_timesteps(
        self,
        num_inference_steps: int,
        latents: torch.Tensor | None = None,
        state: AdapterState | None = None,
    ) -> list[Any]:
        raise NotImplementedError(self.unsupported_reason)

    def predict_vector_field(
        self,
        latents: torch.Tensor,
        timestep: Any,
        condition: PromptCondition,
        state: AdapterState,
    ) -> torch.Tensor:
        raise NotImplementedError(self.unsupported_reason)

    def scheduler_step(
        self,
        model_prediction: torch.Tensor,
        timestep: Any,
        latents: torch.Tensor,
        state: AdapterState,
        generator: torch.Generator | None = None,
    ) -> Any:
        raise NotImplementedError(self.unsupported_reason)

    def decode_latents(self, latents: torch.Tensor, state: AdapterState) -> Any:
        raise NotImplementedError(self.unsupported_reason)


class Cosmos3TextToImageAdapter(UnsupportedVectorFieldAdapter):
    adapter_name = "cosmos3_t2i"
    task_type = "text_to_image"
    pipeline_class_name = "DiffusionPipeline"
    unsupported_reason = (
        "nvidia/Cosmos3-Super-Text2Image is published for generic DiffusionPipeline/Cosmos use, "
        "but this installed diffusers build does not expose a verified Cosmos3 text-to-image "
        "pipeline class with vector-field steering hooks."
    )


class JoyAIEchoAdapter(UnsupportedVectorFieldAdapter):
    adapter_name = "joyai_echo"
    task_type = "text_to_video"
    pipeline_class_name = None
    unsupported_reason = (
        "jdopensource/JoyAI-Echo is an ltx-video checkpoint, but this repo does not yet have a "
        "verified JoyAI-Echo pipeline or vector-field steering adapter."
    )
