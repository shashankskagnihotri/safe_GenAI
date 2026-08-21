from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from hierasafe_flow.adapters.base import (
    AdapterState,
    DiffusersFrozenAdapter,
    PromptCondition,
    SchedulerStepResult,
)


class Cosmos3TextToImageAdapter(DiffusersFrozenAdapter):
    adapter_name = "cosmos3_t2i"
    task_type = "text_to_image"
    pipeline_class_name = "Cosmos3OmniPipeline"
    latent_feature_dim = 1  # [B, C, F, H, W]
    required_components = ("transformer", "scheduler", "vae")

    def load(self) -> None:
        ablation = self._local_ablation_metadata()
        if ablation is not None:
            self.model_id = str(ablation["base_model_id"])
        self.config.setdefault("load_kwargs", {})
        self.config["load_kwargs"].setdefault("enable_safety_checker", False)
        self.config["load_kwargs"].setdefault("device_map", os.environ.get("COSMOS3_DEVICE_MAP", "balanced"))
        pipeline_cls = self._resolve_pipeline_class()
        kwargs = dict(self.config.get("load_kwargs", {}))
        if self.config.get("revision") is not None:
            kwargs["revision"] = self.config["revision"]
        if self.config.get("variant") is not None:
            kwargs["variant"] = self.config["variant"]
        kwargs.setdefault("torch_dtype", self.dtype)
        kwargs.setdefault("local_files_only", bool(self.config.get("local_files_only", False)))
        if os.environ.get("HF_TOKEN") and "token" not in kwargs:
            kwargs["token"] = os.environ["HF_TOKEN"]
        kwargs.setdefault("low_cpu_mem_usage", True)
        self.pipeline = pipeline_cls.from_pretrained(self.model_id, **kwargs)
        if "device_map" not in kwargs and hasattr(self.pipeline, "to"):
            self.pipeline.to(self.device)
        self._freeze_pipeline()
        if ablation is not None:
            self._apply_local_ablation(ablation)
        self._validate_components()
        self.loaded = True

    def prepare_prompt(self, prompt: str) -> PromptCondition:
        self._require_loaded()
        assert self.pipeline is not None
        context = self._generation_context()
        cond_ids, uncond_ids = self.pipeline.tokenize_prompt(
            prompt,
            self.config.get("negative_prompt", None),
            num_frames=int(context["num_frames"]),
            height=int(context["height"]),
            width=int(context["width"]),
            fps=float(context["fps"]),
            use_system_prompt=bool(self.config.get("use_system_prompt", True)),
            add_resolution_template=bool(self.config.get("add_resolution_template", True)),
            add_duration_template=bool(self.config.get("add_duration_template", True)),
        )
        return PromptCondition(
            prompt=prompt,
            data={
                "cond_text_segment": self.pipeline._prepare_text_segment(cond_ids, device=self.device),
                "uncond_text_segment": self.pipeline._prepare_text_segment(uncond_ids, device=self.device),
            },
        )

    def prepare_prompts(self, prompts: list[str]) -> list[PromptCondition]:
        return [self.prepare_prompt(prompt) for prompt in prompts]

    def prepare_initial_latents(
        self,
        prompt: str,
        batch_size: int,
        generator: torch.Generator | None,
        **generation_kwargs: Any,
    ) -> tuple[torch.Tensor, AdapterState]:
        self._require_loaded()
        if batch_size != 1:
            raise ValueError("Cosmos3OmniPipeline supports one sample per call.")
        del prompt
        assert self.pipeline is not None
        height = int(generation_kwargs.get("height", 1024))
        width = int(generation_kwargs.get("width", 1024))
        fps = float(generation_kwargs.get("fps", 24.0))
        num_frames = int(generation_kwargs.get("num_frames", 1))
        context = {
            "height": height,
            "width": width,
            "fps": fps,
            "num_frames": num_frames,
        }
        self.config["_cosmos3_generation_context"] = dict(context)
        (
            latents,
            sound_latents,
            action_latents,
            fps_vision,
            fps_sound,
            vision_condition_mask,
            sound_condition_mask,
            action_condition_mask,
            action_domain_id,
            action_image_size,
            raw_action_dim_resolved,
            action_condition_frame_indexes,
        ) = self.pipeline.prepare_latents(
            image=None,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            latents=generation_kwargs.get("latents"),
            sound_latents=None,
            action_latents=None,
            generator=generator,
            device=self.device,
            dtype=self.pipeline.transformer.dtype,
            enable_sound=False,
            action=None,
        )
        if sound_latents is not None or action_latents is not None:
            raise RuntimeError("Cosmos3 text-to-image adapter received unexpected sound/action latents.")
        return latents, AdapterState(
            extra={
                **context,
                "fps_vision": fps_vision,
                "fps_sound": fps_sound,
                "vision_condition_mask": vision_condition_mask,
                "sound_condition_mask": sound_condition_mask,
                "action_condition_mask": action_condition_mask,
                "action_domain_id": action_domain_id,
                "action_image_size": action_image_size,
                "raw_action_dim_resolved": raw_action_dim_resolved,
                "action_condition_frame_indexes": action_condition_frame_indexes,
            }
        )

    def set_timesteps(
        self,
        num_inference_steps: int,
        latents: torch.Tensor | None = None,
        state: AdapterState | None = None,
    ) -> list[Any]:
        self._require_loaded()
        del latents, state
        assert self.pipeline is not None
        self.pipeline.scheduler.set_timesteps(num_inference_steps, device=self.device)
        self.timesteps = list(self.pipeline.scheduler.timesteps)
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
        cond = self._predict_for_text_segment(
            latents=latents,
            timestep=timestep,
            text_segment=condition.data["cond_text_segment"],
            state=state,
        )
        guidance_scale = self._manual_guidance_scale()
        if guidance_scale == 1.0:
            return cond
        uncond = self._predict_for_text_segment(
            latents=latents,
            timestep=timestep,
            text_segment=condition.data["uncond_text_segment"],
            state=state,
        )
        return uncond + guidance_scale * (cond - uncond)

    def scheduler_step(
        self,
        model_prediction: torch.Tensor,
        timestep: Any,
        latents: torch.Tensor,
        state: AdapterState,
        generator: torch.Generator | None = None,
    ) -> SchedulerStepResult:
        self._require_loaded()
        del generator
        assert self.pipeline is not None
        output = self.pipeline.scheduler.step(
            model_prediction.unsqueeze(0),
            timestep,
            latents.unsqueeze(0),
            return_dict=False,
        )[0].squeeze(0)
        return SchedulerStepResult(latents=output, state=state)

    def decode_latents(self, latents: torch.Tensor, state: AdapterState) -> Any:
        self._require_loaded()
        del state
        assert self.pipeline is not None
        in_dtype = latents.dtype
        dtype = self.pipeline.vae.dtype
        mean = self.pipeline._vae_latents_mean.to(device=latents.device, dtype=dtype)
        inv_std = self.pipeline._vae_latents_inv_std.to(device=latents.device, dtype=dtype)
        z_raw = latents.to(dtype) / inv_std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)
        decoded = self.pipeline.vae.decode(z_raw).sample.to(in_dtype)
        return self.pipeline.video_processor.postprocess_video(decoded, output_type="pil")[0]

    def _predict_for_text_segment(
        self,
        latents: torch.Tensor,
        timestep: Any,
        text_segment: dict[str, Any],
        state: AdapterState,
    ) -> torch.Tensor:
        assert self.pipeline is not None
        vision_condition_mask = state.extra["vision_condition_mask"]
        vision_condition_indexes = torch.nonzero(vision_condition_mask[:, 0, 0] > 0, as_tuple=False).flatten()
        vision_condition_indexes_for_pack = [int(idx.item()) for idx in vision_condition_indexes]
        has_image_condition = bool(vision_condition_indexes_for_pack)
        vision_segment = self.pipeline._prepare_vision_segment(
            input_vision_tokens=latents,
            has_image_condition=has_image_condition,
            mrope_offset=text_segment["vision_start_temporal_offset"],
            vision_fps=state.extra["fps_vision"],
            curr=text_segment["und_len"],
            device=self.device,
            condition_frame_indexes=vision_condition_indexes_for_pack,
        )
        packed = {
            **text_segment,
            **vision_segment,
            "position_ids": torch.cat(
                [text_segment["text_mrope_ids"], vision_segment["vision_mrope_ids"]],
                dim=1,
            ),
            "sequence_length": text_segment["und_len"] + vision_segment["num_vision_tokens"],
        }
        timestep_value = _timestep_value(timestep)
        vision_timesteps = torch.full(
            (vision_segment["num_noisy_vision_tokens"],),
            timestep_value,
            device=self.device,
        )
        preds_vision, preds_sound, preds_action = self.pipeline.transformer(
            input_ids=packed["input_ids"],
            text_indexes=packed["text_indexes"],
            position_ids=packed["position_ids"],
            und_len=packed["und_len"],
            sequence_length=packed["sequence_length"],
            vision_tokens=[latents.to(device=self.device, dtype=self.pipeline.transformer.dtype)],
            vision_token_shapes=packed["vision_token_shapes"],
            vision_sequence_indexes=packed["vision_sequence_indexes"],
            vision_mse_loss_indexes=packed["vision_mse_loss_indexes"],
            vision_timesteps=vision_timesteps,
            vision_noisy_frame_indexes=packed["vision_noisy_frame_indexes"],
        )
        velocity_vision, _, _ = self.pipeline._mask_velocity_predictions(
            preds_vision,
            preds_sound,
            vision_condition_mask=[vision_condition_mask],
            preds_action=preds_action,
        )
        return velocity_vision

    def _resolve_pipeline_class(self) -> type:
        try:
            import diffusers
        except ModuleNotFoundError as exc:
            raise NotImplementedError("Cosmos3 requires diffusers from GitHub main.") from exc
        if hasattr(diffusers, "Cosmos3OmniPipeline"):
            return diffusers.Cosmos3OmniPipeline
        if hasattr(diffusers, "Cosmos3OmniDiffusersPipeline"):
            return diffusers.Cosmos3OmniDiffusersPipeline
        raise NotImplementedError(
            "diffusers does not expose Cosmos3OmniPipeline. Install diffusers from GitHub main "
            "for nvidia/Cosmos3-Super-Text2Image support."
        )

    def _generation_context(self) -> dict[str, Any]:
        context = self.config.get("_cosmos3_generation_context")
        if context is None:
            context = {"height": 1024, "width": 1024, "fps": 24.0, "num_frames": 1}
        return dict(context)

    def _local_ablation_metadata(self) -> dict[str, Any] | None:
        model_path = Path(str(self.model_id)).expanduser()
        metadata_path = model_path / "cosmos3_abliteration.json"
        if not metadata_path.exists():
            return None
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata["_artifact_dir"] = str(model_path.resolve())
        return metadata

    def _apply_local_ablation(self, metadata: dict[str, Any]) -> None:
        assert self.pipeline is not None
        artifact_dir = Path(str(metadata["_artifact_dir"]))
        direction_path = artifact_dir / str(metadata.get("direction_file", "text_embedding_refusal_direction.pt"))
        direction = torch.load(direction_path, map_location="cpu")
        if isinstance(direction, dict):
            direction = direction["transformer.embed_tokens"]
        direction = direction.float()
        scale_factor = float(metadata.get("scale_factor", 1.0))
        embedding = self.pipeline.transformer.embed_tokens
        with torch.no_grad():
            weight = embedding.weight.data
            weight_fp32 = weight.float()
            direction = direction.to(weight_fp32.device)
            embedding.weight.data = (
                weight_fp32 - scale_factor * torch.outer(weight_fp32 @ direction, direction)
            ).to(weight.dtype)


def _timestep_value(timestep: Any) -> float:
    if isinstance(timestep, torch.Tensor):
        return float(timestep.detach().flatten()[0].item())
    return float(timestep)
