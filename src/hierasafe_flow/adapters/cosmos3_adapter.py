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
        load_kwargs = self.config.setdefault("load_kwargs", {})
        if load_kwargs.get("enable_safety_checker", True) is not True:
            raise ValueError(
                "Cosmos3 exact execution forbids disabling the official CosmosSafetyChecker."
            )
        load_kwargs["enable_safety_checker"] = True
        load_kwargs.setdefault("device_map", os.environ.get("COSMOS3_DEVICE_MAP", "balanced"))
        model_source, source_is_local = self._resolved_model_source()
        pipeline_cls = self._resolve_pipeline_class()
        kwargs = dict(self.config.get("load_kwargs", {}))
        prepartitioned_transformer = self._load_distributed_transformer(
            kwargs,
            model_source=model_source,
            source_is_local=source_is_local,
        )
        if self.config.get("revision") is not None and not source_is_local:
            kwargs["revision"] = self.config["revision"]
        if self.config.get("variant") is not None:
            kwargs["variant"] = self.config["variant"]
        kwargs.setdefault("torch_dtype", self.dtype)
        kwargs.setdefault(
            "local_files_only",
            source_is_local or bool(self.config.get("local_files_only", False)),
        )
        if os.environ.get("HF_TOKEN") and "token" not in kwargs:
            kwargs["token"] = os.environ["HF_TOKEN"]
        kwargs.setdefault("low_cpu_mem_usage", True)
        if prepartitioned_transformer is not None:
            kwargs["transformer"] = prepartitioned_transformer
        self.pipeline = pipeline_cls.from_pretrained(model_source, **kwargs)
        self._require_safety_checker()
        if prepartitioned_transformer is not None:
            self.pipeline.vae.to(self.device)
        elif "device_map" not in kwargs and hasattr(self.pipeline, "to"):
            self.pipeline.to(self.device)
        self._freeze_pipeline()
        if ablation is not None:
            self._apply_local_ablation(ablation)
        self._validate_components()
        self.config["_cosmos3_guardrail_contract"] = {
            "constructor_enabled": True,
            "text_pre_generation": True,
            "visual_post_decode": True,
            "disable_override_forbidden": True,
        }
        self.config["_cosmos3_model_source_contract"] = {
            "configured_model_id": str(self.model_id),
            "configured_revision": str(self.config.get("revision")),
            "resolved_source": model_source,
            "source_is_local": source_is_local,
            "local_files_only": bool(kwargs["local_files_only"]),
        }
        self.loaded = True

    def _require_safety_checker(self) -> Any:
        if self.pipeline is None:
            raise RuntimeError("Cosmos3 pipeline is not loaded.")
        checker = getattr(self.pipeline, "safety_checker", None)
        if checker is None:
            raise RuntimeError(
                "Cosmos3 exact execution requires the official CosmosSafetyChecker."
            )
        for method_name in ("to", "check_text_safety", "check_video_safety"):
            if not callable(getattr(checker, method_name, None)):
                raise RuntimeError(
                    "Cosmos3 safety checker is missing required method "
                    f"{method_name!r}."
                )
        return checker

    def _check_text_safety(self, prompt: str) -> None:
        assert self.pipeline is not None
        checker = self._require_safety_checker()
        execution_device = self.pipeline._get_execution_device()
        checker.to(execution_device)
        try:
            is_safe = checker.check_text_safety(prompt)
        finally:
            checker.to("cpu")
        if not bool(is_safe):
            raise ValueError(
                "Cosmos Guardrail detected unsafe text in the prompt. "
                "The sample is blocked by the model's mandatory safety contract."
            )

    def _load_distributed_transformer(
        self,
        pipeline_kwargs: dict[str, Any],
        *,
        model_source: str,
        source_is_local: bool,
    ) -> Any | None:
        device_map = pipeline_kwargs.get("device_map")
        visible_gpu_count = torch.cuda.device_count()
        if device_map not in {"auto", "balanced", "balanced_low_0", "sequential"}:
            return None
        if visible_gpu_count < 2:
            return None

        try:
            import diffusers
        except ModuleNotFoundError as exc:
            raise NotImplementedError("Cosmos3 requires diffusers from GitHub main.") from exc
        transformer_cls = getattr(diffusers, "Cosmos3OmniTransformer", None)
        if transformer_cls is None:
            raise NotImplementedError(
                "diffusers does not expose Cosmos3OmniTransformer for layer-wise dispatch."
            )

        reserve_gib = float(os.environ.get("COSMOS3_GPU_RESERVE_GIB", "12"))
        if reserve_gib <= 0:
            raise ValueError("COSMOS3_GPU_RESERVE_GIB must be positive.")
        reserve_bytes = int(reserve_gib * 1024**3)
        max_memory: dict[int | str, int] = {}
        for index in range(visible_gpu_count):
            total_bytes = int(torch.cuda.get_device_properties(index).total_memory)
            budget_bytes = total_bytes - reserve_bytes
            if budget_bytes <= 0:
                raise RuntimeError(
                    f"GPU {index} has {total_bytes} bytes, which is not greater than the "
                    f"required {reserve_bytes}-byte Cosmos3 runtime reserve."
                )
            max_memory[index] = budget_bytes
        # A CPU or disk placement is not an exact multi-GPU execution and is forbidden.
        max_memory["cpu"] = 0

        transformer_kwargs: dict[str, Any] = {
            "subfolder": "transformer",
            "device_map": device_map,
            "max_memory": max_memory,
            "torch_dtype": self.dtype,
            "local_files_only": source_is_local
            or bool(self.config.get("local_files_only", False)),
            "low_cpu_mem_usage": True,
        }
        if self.config.get("revision") is not None and not source_is_local:
            transformer_kwargs["revision"] = self.config["revision"]
        if self.config.get("variant") is not None:
            transformer_kwargs["variant"] = self.config["variant"]
        if os.environ.get("HF_TOKEN"):
            transformer_kwargs["token"] = os.environ["HF_TOKEN"]
        transformer_config_kwargs: dict[str, Any] = {
            "subfolder": "transformer",
            "local_files_only": source_is_local
            or bool(self.config.get("local_files_only", False)),
            "token": os.environ.get("HF_TOKEN"),
        }
        if self.config.get("revision") is not None and not source_is_local:
            transformer_config_kwargs["revision"] = self.config["revision"]
        transformer_config = transformer_cls.load_config(
            model_source,
            **transformer_config_kwargs,
        )
        explicit_device_map = _cosmos3_explicit_device_map(
            num_hidden_layers=int(transformer_config["num_hidden_layers"]),
            visible_gpu_count=visible_gpu_count,
            sound_gen=bool(transformer_config.get("sound_gen", False)),
            action_gen=bool(transformer_config.get("action_gen", False)),
        )
        transformer_kwargs["device_map"] = explicit_device_map

        transformer = transformer_cls.from_pretrained(model_source, **transformer_kwargs)
        placement = getattr(transformer, "hf_device_map", None)
        if not isinstance(placement, dict) or not placement:
            raise RuntimeError("Cosmos3 transformer did not expose a layer-wise hf_device_map.")

        normalized_devices = {_normalize_device_map_value(value) for value in placement.values()}
        forbidden = sorted(device for device in normalized_devices if device in {"cpu", "disk", "meta"})
        if forbidden:
            raise RuntimeError(
                f"Cosmos3 exact distributed load used forbidden placements: {forbidden}."
            )
        expected_devices = {f"cuda:{index}" for index in range(visible_gpu_count)}
        used_devices = {device for device in normalized_devices if device.startswith("cuda:")}
        if used_devices != expected_devices:
            raise RuntimeError(
                "Cosmos3 exact distributed load did not use every visible GPU: "
                f"expected={sorted(expected_devices)}, used={sorted(used_devices)}."
            )

        self.config["_cosmos3_transformer_device_map"] = {
            str(name): _normalize_device_map_value(value)
            for name, value in sorted(placement.items())
        }
        self.config["_cosmos3_transformer_max_memory_bytes"] = {
            str(device): int(value) for device, value in max_memory.items()
        }
        self.config["_cosmos3_transformer_gpu_reserve_bytes"] = reserve_bytes
        self.config["_cosmos3_transformer_requested_strategy"] = str(device_map)
        print(
            json.dumps(
                {
                    "cosmos3_exact_distributed_transformer": {
                        "device_map": self.config["_cosmos3_transformer_device_map"],
                        "max_memory_bytes": self.config[
                            "_cosmos3_transformer_max_memory_bytes"
                        ],
                        "reserve_bytes": reserve_bytes,
                    }
                },
                sort_keys=True,
            ),
            flush=True,
        )
        pipeline_kwargs.pop("device_map", None)
        return transformer

    def _resolved_model_source(self) -> tuple[str, bool]:
        configured = self.config.get("local_snapshot_path")
        if configured is None:
            if bool(self.config.get("local_files_only", False)) or os.environ.get(
                "HF_HUB_OFFLINE"
            ) == "1":
                raise RuntimeError(
                    "Cosmos3 offline execution requires model.local_snapshot_path; "
                    "a repository ID would trigger an unpinned Hub metadata request."
                )
            return str(self.model_id), False

        snapshot = Path(str(configured)).expanduser()
        if not snapshot.is_absolute():
            raise ValueError("Cosmos3 local_snapshot_path must be absolute.")
        if not snapshot.is_dir():
            raise FileNotFoundError(f"Cosmos3 local snapshot is missing: {snapshot}")
        revision = self.config.get("revision")
        if revision is None:
            raise ValueError("Cosmos3 local snapshot execution requires a frozen revision.")
        if snapshot.name != str(revision):
            raise ValueError(
                "Cosmos3 local snapshot/revision mismatch: "
                f"snapshot={snapshot.name!r}, revision={revision!r}."
            )
        required = (
            snapshot / "model_index.json",
            snapshot / "transformer" / "config.json",
            snapshot / "transformer" / "diffusion_pytorch_model.safetensors.index.json",
            snapshot / "vae" / "config.json",
            snapshot / "scheduler" / "scheduler_config.json",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Cosmos3 frozen local snapshot is incomplete: " + ", ".join(missing)
            )
        return str(snapshot), True

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
        self._check_text_safety(prompt)
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
        video = self.pipeline.video_processor.postprocess_video(decoded, output_type="pil")[0]
        self._require_safety_checker()
        safety_helper = getattr(self.pipeline, "_apply_video_safety_check", None)
        if not callable(safety_helper):
            raise RuntimeError(
                "Pinned Cosmos3 pipeline does not expose its required visual safety helper."
            )
        return safety_helper(
            video,
            output_type="pil",
            device=self.pipeline._get_execution_device(),
        )

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


def _normalize_device_map_value(value: Any) -> str:
    if isinstance(value, int):
        return f"cuda:{value}"
    device = str(value)
    if device.isdigit():
        return f"cuda:{device}"
    if device == "cuda":
        return "cuda:0"
    return device


def _cosmos3_explicit_device_map(
    *,
    num_hidden_layers: int,
    visible_gpu_count: int,
    sound_gen: bool,
    action_gen: bool,
) -> dict[str, int]:
    if visible_gpu_count < 2:
        raise ValueError("Cosmos3 distributed device map requires at least two GPUs.")
    if num_hidden_layers < visible_gpu_count:
        raise ValueError(
            f"Cannot distribute {num_hidden_layers} Cosmos3 layers over "
            f"{visible_gpu_count} GPUs."
        )

    layer_counts = [num_hidden_layers // visible_gpu_count] * visible_gpu_count
    for index in range(num_hidden_layers % visible_gpu_count):
        layer_counts[index] += 1
    if visible_gpu_count >= 4:
        layer_counts[0] -= 1
        layer_counts[1] += 1
        layer_counts[-1] -= 1
        layer_counts[-2] += 1

    device_map: dict[str, int] = {
        "embed_tokens": 0,
        "proj_in": 0,
        "time_embedder": 0,
        "norm": 0,
        "norm_moe_gen": 0,
        "proj_out": 0,
        # The language head is not used by text-to-image denoising and balances
        # the large token embedding table on the first device.
        "lm_head": visible_gpu_count - 1,
    }
    layer_index = 0
    for device_index, count in enumerate(layer_counts):
        for _ in range(count):
            device_map[f"layers.{layer_index}"] = device_index
            layer_index += 1
    if layer_index != num_hidden_layers:
        raise RuntimeError("Cosmos3 explicit layer map did not cover every decoder layer.")

    if sound_gen:
        device_map.update(
            {
                "audio_proj_in": 0,
                "audio_modality_embed": 0,
                "audio_proj_out": 0,
            }
        )
    if action_gen:
        device_map.update(
            {
                "action_proj_in": 0,
                "action_modality_embed": 0,
                "action_proj_out": 0,
            }
        )
    return device_map


def _timestep_value(timestep: Any) -> float:
    if isinstance(timestep, torch.Tensor):
        return float(timestep.detach().flatten()[0].item())
    return float(timestep)
