from __future__ import annotations

import copy
from contextlib import nullcontext
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from hierasafe_flow.adapters.base import (
    AdapterState,
    DiffusersFrozenAdapter,
    PromptCondition,
    SchedulerStepResult,
    _map_encode_prompt_output,
    _slice_condition_data,
)
from hierasafe_flow.evaluation.temporal_qualification import (
    TEMPORAL_CRITERIA_BY_MODEL,
    validate_temporal_production_gate,
)


LTX_TEMPORAL_PROTOCOL_KEY = "ltx_temporal_protocol"
LTX_CHECKPOINT_REVISION = "432e0d3c2d1769aaa4d295f9243f7062bf6b47ee"
LTX_DIFFUSERS_REVISION = "76e7d164a5eb92a8aeaa83f46b84c67e377591fb"
LTX_PIPELINE_SOURCE_SHA256 = "922f449e594423d833f86d4b4a92923b87e8e1b7c286bccf362c0cd32a25eb8b"
LTX_VIDEO_VAE_SOURCE_SHA256 = "631db22859ecd228785973ae0e4c38e80180fac3c86b66eddc0557dd7900a638"

_EXPECTED_LTX_TEMPORAL_PROTOCOL = {
    "schema_version": 1,
    "strategy": "single_241_frame_trajectory_terminal_crop",
    "checkpoint_family": "LTX-2.3-Distilled-Diffusers",
    "diffusers_revision": LTX_DIFFUSERS_REVISION,
    "pipeline_source_sha256": LTX_PIPELINE_SOURCE_SHA256,
    "video_vae_source_sha256": LTX_VIDEO_VAE_SOURCE_SHA256,
    "api_default_frames": 121,
    "api_default_fps": 24.0,
    "vae_temporal_compression_ratio": 8,
    "internal_generated_frames": 241,
    "internal_fps": 16.0,
    "output_frames": 240,
    "output_fps": 16,
    "duration_seconds": 15.0,
    "height": 768,
    "width": 1344,
    "terminal_endpoint_crop": "[0:240]",
    "frame_resampling": "none",
}
_LTX_TEMPORAL_DYNAMIC_FIELDS = {"execution_phase", "production_gate"}
_LTX_PRODUCTION_CRITERIA = TEMPORAL_CRITERIA_BY_MODEL["ltx_23"]


class LTXAdapter(DiffusersFrozenAdapter):
    adapter_name = "ltx"
    task_type = "text_to_video"
    pipeline_class_name = "LTX2Pipeline"
    latent_feature_dim = -1  # packed [B, audiovisual tokens, D]
    required_components = ("transformer", "scheduler", "vae", "audio_vae", "connectors")
    encode_prompt_output_names = (
        "prompt_embeds",
        "prompt_attention_mask",
        "negative_prompt_embeds",
        "negative_prompt_attention_mask",
    )

    def __init__(
        self,
        model_id: str,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(model_id=model_id, device=device, dtype=dtype, config=config)
        self._last_temporal_provenance: dict[str, Any] | None = None
        self._diffusers_source_provenance: dict[str, Any] | None = None

    def load(self) -> None:
        protocol = self._temporal_protocol()
        if protocol is not None:
            self._validate_temporal_execution_gate(protocol)
            self._diffusers_source_provenance = _validate_diffusers_source_contract(protocol)
        super().load()
        assert self.pipeline is not None
        if protocol is not None:
            actual_ratio = int(self.pipeline.vae_temporal_compression_ratio)
            expected_ratio = int(protocol["vae_temporal_compression_ratio"])
            if actual_ratio != expected_ratio:
                raise RuntimeError(
                    "Pinned LTX-2.3 VAE temporal contract changed: expected compression "
                    f"ratio {expected_ratio}, got {actual_ratio}."
                )

    def prepare_prompt(self, prompt: str) -> PromptCondition:
        self._require_loaded()
        self._validate_prompt_token_budget([prompt])
        data = super().prepare_prompt(prompt).data
        data = self._add_connector_outputs(data)
        return PromptCondition(prompt=prompt, data=data)

    def prepare_prompts(self, prompts: list[str]) -> list[PromptCondition]:
        self._require_loaded()
        if not prompts:
            return []
        self._validate_prompt_token_budget(prompts)
        output = self._call_encode_prompt(list(prompts))
        data = _map_encode_prompt_output(output, self.encode_prompt_output_names)
        data = self._add_connector_outputs(data)
        do_cfg = bool(data.get("do_classifier_free_guidance", False))
        batch_size = len(prompts)
        return [
            PromptCondition(
                prompt=prompt,
                data=_slice_ltx_condition_data(data, index, batch_size, do_cfg),
            )
            for index, prompt in enumerate(prompts)
        ]

    def _validate_prompt_token_budget(self, prompts: list[str]) -> None:
        assert self.pipeline is not None
        max_sequence_length = self.config.get("max_sequence_length")
        if max_sequence_length is None:
            return
        tokenizer = getattr(self.pipeline, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("LTX prompt token-budget validation requires its pinned tokenizer.")
        encoded = tokenizer(
            [prompt.strip() for prompt in prompts],
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        token_ids = encoded["input_ids"]
        longest = max(len(ids) for ids in token_ids)
        if longest > int(max_sequence_length):
            raise ValueError(
                "LTX prompt exceeds the frozen no-truncation token budget: "
                f"required {longest}, allowed {int(max_sequence_length)}."
            )

    def _add_connector_outputs(self, data: dict[str, Any]) -> dict[str, Any]:
        assert self.pipeline is not None
        data = dict(data)
        prompt_embeds = data.get("prompt_embeds")
        prompt_attention_mask = data.get("prompt_attention_mask")
        if prompt_embeds is None or prompt_attention_mask is None:
            raise NotImplementedError(
                "LTX2 encode_prompt did not expose prompt embeddings and attention mask."
            )
        guidance_scale = self._manual_guidance_scale()
        do_cfg = guidance_scale > 1.0 and data.get("negative_prompt_embeds") is not None
        if do_cfg:
            prompt_embeds = torch.cat([data["negative_prompt_embeds"], prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat(
                [data["negative_prompt_attention_mask"], prompt_attention_mask],
                dim=0,
            )
        tokenizer_padding_side = "left"
        if getattr(self.pipeline, "tokenizer", None) is not None:
            tokenizer_padding_side = getattr(self.pipeline.tokenizer, "padding_side", "left")
        connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask = (
            self.pipeline.connectors(
                prompt_embeds,
                prompt_attention_mask,
                padding_side=tokenizer_padding_side,
            )
        )
        data.update(
            {
                "connector_prompt_embeds": connector_prompt_embeds,
                "connector_audio_prompt_embeds": connector_audio_prompt_embeds,
                "connector_attention_mask": connector_attention_mask,
                "do_classifier_free_guidance": do_cfg,
            }
        )
        return data

    def prepare_initial_latents(
        self,
        prompt: str,
        batch_size: int,
        generator: torch.Generator | None,
        **generation_kwargs: Any,
    ) -> tuple[torch.Tensor, AdapterState]:
        self._require_loaded()
        assert self.pipeline is not None
        height = int(generation_kwargs.get("height", 512))
        width = int(generation_kwargs.get("width", 768))
        requested_num_frames = int(generation_kwargs.get("num_frames", 121))
        compression_ratio = int(self.pipeline.vae_temporal_compression_ratio)
        frame_rate = _finite_positive_float(
            generation_kwargs.get("frame_rate", generation_kwargs.get("fps", 24.0)),
            "LTX-2.3 requested frame_rate",
        )
        protocol = self._temporal_protocol()
        if protocol is None:
            if requested_num_frames <= 0 or (requested_num_frames - 1) % compression_ratio:
                raise ValueError(
                    "LTX-2.3 decoded frame counts must satisfy 1+8*k. A non-aligned "
                    "output requires an explicit terminal-crop temporal protocol; got "
                    f"{requested_num_frames}."
                )
            num_frames = requested_num_frames
            _validate_optional_duration(
                generation_kwargs,
                output_frames=requested_num_frames,
                output_fps=frame_rate,
            )
            temporal_provenance = _direct_temporal_provenance(
                internal_frames=num_frames,
                output_frames=requested_num_frames,
                fps=frame_rate,
            )
        else:
            self._validate_temporal_execution_gate(protocol)
            self._validate_requested_temporal_output(generation_kwargs, protocol)
            if compression_ratio != int(protocol["vae_temporal_compression_ratio"]):
                raise RuntimeError(
                    "LTX-2.3 VAE temporal compression changed after loading: expected "
                    f"{protocol['vae_temporal_compression_ratio']}, got {compression_ratio}."
                )
            num_frames = int(protocol["internal_generated_frames"])
            temporal_provenance = self._configured_temporal_provenance(protocol)
        noise_scale = float(generation_kwargs.get("noise_scale", 0.0))

        latent_num_frames = (num_frames - 1) // compression_ratio + 1
        latent_height = height // self.pipeline.vae_spatial_compression_ratio
        latent_width = width // self.pipeline.vae_spatial_compression_ratio
        num_channels_latents = int(self.pipeline.transformer.config.in_channels)
        latents = self.pipeline.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            noise_scale=noise_scale,
            dtype=torch.float32,
            device=self.device,
            generator=generator,
            latents=generation_kwargs.get("latents"),
        )
        expected_video_sequence_length = latent_num_frames * latent_height * latent_width
        if latents.ndim != 3 or int(latents.shape[1]) != expected_video_sequence_length:
            raise RuntimeError(
                "LTX-2.3 packed video latent arithmetic changed: expected sequence length "
                f"{expected_video_sequence_length}, got shape {tuple(latents.shape)}."
            )

        duration_seconds = num_frames / frame_rate
        audio_latents_per_second = (
            self.pipeline.audio_sampling_rate
            / self.pipeline.audio_hop_length
            / float(self.pipeline.audio_vae_temporal_compression_ratio)
        )
        audio_num_frames = round(duration_seconds * audio_latents_per_second)
        num_mel_bins = int(self.pipeline.audio_vae.config.mel_bins)
        latent_mel_bins = num_mel_bins // self.pipeline.audio_vae_mel_compression_ratio
        audio_channels = int(self.pipeline.audio_vae.config.latent_channels)
        audio_latents = self.pipeline.prepare_audio_latents(
            batch_size=batch_size,
            num_channels_latents=audio_channels,
            audio_latent_length=audio_num_frames,
            num_mel_bins=num_mel_bins,
            noise_scale=noise_scale,
            dtype=torch.float32,
            device=self.device,
            generator=generator,
            latents=generation_kwargs.get("audio_latents"),
        )
        if audio_latents.ndim != 3 or int(audio_latents.shape[1]) != audio_num_frames:
            raise RuntimeError(
                "LTX-2.3 packed audio latent arithmetic changed: expected sequence length "
                f"{audio_num_frames}, got shape {tuple(audio_latents.shape)}."
            )

        video_coords = self.pipeline.transformer.rope.prepare_video_coords(
            latents.shape[0],
            latent_num_frames,
            latent_height,
            latent_width,
            latents.device,
            fps=frame_rate,
        )
        audio_coords = self.pipeline.transformer.audio_rope.prepare_audio_coords(
            audio_latents.shape[0],
            audio_num_frames,
            audio_latents.device,
        )

        temporal_provenance["internal_trajectory"]["audio_latent_frames"] = audio_num_frames
        temporal_provenance["internal_trajectory"]["audio_latents_per_second"] = (
            audio_latents_per_second
        )
        temporal_provenance["internal_trajectory"]["audio_duration_seconds_on_latent_grid"] = (
            audio_num_frames / audio_latents_per_second
        )
        temporal_provenance["internal_trajectory"]["audio_video_clock_error_seconds"] = abs(
            audio_num_frames / audio_latents_per_second - duration_seconds
        )
        temporal_provenance["internal_trajectory"][
            "audio_video_clock_error_within_one_audio_latent_interval"
        ] = (
            abs(audio_num_frames / audio_latents_per_second - duration_seconds)
            <= 1.0 / audio_latents_per_second
        )
        state = AdapterState(
            extra={
                "base_prompt": prompt,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "output_num_frames": requested_num_frames,
                "frame_rate": frame_rate,
                "latent_num_frames": latent_num_frames,
                "latent_height": latent_height,
                "latent_width": latent_width,
                "video_sequence_length": latent_num_frames * latent_height * latent_width,
                "audio_latents": audio_latents,
                "audio_num_frames": audio_num_frames,
                "num_mel_bins": num_mel_bins,
                "latent_mel_bins": latent_mel_bins,
                "video_coords": video_coords,
                "audio_coords": audio_coords,
                "duration_seconds": requested_num_frames / frame_rate,
                "internal_duration_seconds": duration_seconds,
                "audio_latents_per_second": audio_latents_per_second,
                "temporal_provenance": temporal_provenance,
                "attention_kwargs": generation_kwargs.get("attention_kwargs"),
                "decode_timestep": generation_kwargs.get("decode_timestep", 0.0),
                "decode_noise_scale": generation_kwargs.get("decode_noise_scale"),
                "generator": generator,
                "step_index": 0,
                "use_cross_timestep": bool(generation_kwargs.get("use_cross_timestep", False)),
            }
        )
        self._last_temporal_provenance = temporal_provenance
        return latents, state

    def set_timesteps(
        self,
        num_inference_steps: int,
        latents: torch.Tensor | None = None,
        state: AdapterState | None = None,
    ) -> list[Any]:
        self._require_loaded()
        if latents is None or state is None:
            raise NotImplementedError("LTX2 set_timesteps requires latents and adapter state.")
        assert self.pipeline is not None
        from diffusers.pipelines.ltx2.pipeline_ltx2 import calculate_shift, retrieve_timesteps

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        scheduler_config = self.pipeline.scheduler.config
        mu = calculate_shift(
            scheduler_config.get("max_image_seq_len", 4096),
            scheduler_config.get("base_image_seq_len", 1024),
            scheduler_config.get("max_image_seq_len", 4096),
            scheduler_config.get("base_shift", 0.95),
            scheduler_config.get("max_shift", 2.05),
        )
        audio_scheduler = copy.deepcopy(self.pipeline.scheduler)
        retrieve_timesteps(
            audio_scheduler,
            num_inference_steps,
            self.device,
            sigmas=sigmas,
            mu=mu,
        )
        timesteps, _ = retrieve_timesteps(
            self.pipeline.scheduler,
            num_inference_steps,
            self.device,
            sigmas=sigmas,
            mu=mu,
        )
        if hasattr(self.pipeline.scheduler, "set_begin_index"):
            self.pipeline.scheduler.set_begin_index(0)
        if hasattr(audio_scheduler, "set_begin_index"):
            audio_scheduler.set_begin_index(0)
        state.extra["audio_scheduler"] = audio_scheduler
        self.timesteps = list(timesteps)
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
        audio_latents = state.extra.get("audio_latents")
        if audio_latents is None:
            raise NotImplementedError("LTX2 adapter state is missing audio_latents.")
        connector_prompt_embeds = condition.data.get("connector_prompt_embeds")
        connector_audio_prompt_embeds = condition.data.get("connector_audio_prompt_embeds")
        connector_attention_mask = condition.data.get("connector_attention_mask")
        if connector_prompt_embeds is None or connector_audio_prompt_embeds is None:
            raise NotImplementedError(
                "LTX2 prompt preparation did not expose connector embeddings."
            )
        audio_scheduler = state.extra.get("audio_scheduler")
        if audio_scheduler is None:
            raise NotImplementedError("LTX2 adapter state is missing its audio scheduler.")

        do_cfg = bool(condition.data.get("do_classifier_free_guidance", False))
        guidance_scale = self._manual_guidance_scale()
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        latent_model_input = latent_model_input.to(connector_prompt_embeds.dtype)
        audio_latent_model_input = torch.cat([audio_latents] * 2) if do_cfg else audio_latents
        audio_latent_model_input = audio_latent_model_input.to(connector_prompt_embeds.dtype)
        timestep_batch = self._timestep_batch(timestep, latent_model_input.shape[0])
        if timestep_batch.ndim == 0:
            timestep_batch = timestep_batch.expand(latent_model_input.shape[0])

        cache_context = (
            self.pipeline.transformer.cache_context("cond_uncond")
            if hasattr(self.pipeline.transformer, "cache_context")
            else nullcontext()
        )
        with cache_context:
            output = self.pipeline.transformer(
                hidden_states=latent_model_input,
                audio_hidden_states=audio_latent_model_input,
                encoder_hidden_states=connector_prompt_embeds,
                audio_encoder_hidden_states=connector_audio_prompt_embeds,
                timestep=timestep_batch,
                sigma=timestep_batch,
                encoder_attention_mask=connector_attention_mask,
                audio_encoder_attention_mask=connector_attention_mask,
                num_frames=int(state.extra["latent_num_frames"]),
                height=int(state.extra["latent_height"]),
                width=int(state.extra["latent_width"]),
                fps=float(state.extra["frame_rate"]),
                audio_num_frames=int(state.extra["audio_num_frames"]),
                video_coords=state.extra.get("video_coords"),
                audio_coords=state.extra.get("audio_coords"),
                isolate_modalities=False,
                spatio_temporal_guidance_blocks=None,
                perturbation_mask=None,
                use_cross_timestep=bool(state.extra.get("use_cross_timestep", False)),
                attention_kwargs=state.extra.get("attention_kwargs"),
                return_dict=False,
            )
        if not (isinstance(output, tuple) and len(output) >= 2):
            raise NotImplementedError("LTX2 transformer must return video and audio predictions.")
        noise_pred_video = output[0].float()
        noise_pred_audio = output[1].float()
        step_index = int(state.extra.get("step_index", 0))

        if do_cfg:
            video_uncond, video_text = noise_pred_video.chunk(2)
            audio_uncond, audio_text = noise_pred_audio.chunk(2)
            video_uncond = self.pipeline.convert_velocity_to_x0(
                latents, video_uncond, step_index, self.pipeline.scheduler
            )
            video_text = self.pipeline.convert_velocity_to_x0(
                latents, video_text, step_index, self.pipeline.scheduler
            )
            audio_uncond = self.pipeline.convert_velocity_to_x0(
                audio_latents,
                audio_uncond,
                step_index,
                audio_scheduler,
            )
            audio_text = self.pipeline.convert_velocity_to_x0(
                audio_latents,
                audio_text,
                step_index,
                audio_scheduler,
            )
            noise_pred_video = video_uncond + guidance_scale * (video_text - video_uncond)
            noise_pred_audio = audio_uncond + guidance_scale * (audio_text - audio_uncond)
            guidance_rescale = float(self.config.get("guidance_rescale", 0.0) or 0.0)
            if guidance_rescale > 0:
                from diffusers.pipelines.ltx2.pipeline_ltx2 import rescale_noise_cfg

                noise_pred_video = rescale_noise_cfg(
                    noise_pred_video,
                    video_text,
                    guidance_rescale=guidance_rescale,
                )
                noise_pred_audio = rescale_noise_cfg(
                    noise_pred_audio,
                    audio_text,
                    guidance_rescale=guidance_rescale,
                )
        else:
            noise_pred_video = self.pipeline.convert_velocity_to_x0(
                latents,
                noise_pred_video,
                step_index,
                self.pipeline.scheduler,
            )
            noise_pred_audio = self.pipeline.convert_velocity_to_x0(
                audio_latents,
                noise_pred_audio,
                step_index,
                audio_scheduler,
            )
        noise_pred_video = self.pipeline.convert_x0_to_velocity(
            latents,
            noise_pred_video,
            step_index,
            self.pipeline.scheduler,
        )
        noise_pred_audio = self.pipeline.convert_x0_to_velocity(
            audio_latents,
            noise_pred_audio,
            step_index,
            audio_scheduler,
        )

        if condition.prompt == state.extra.get("base_prompt"):
            state.extra["base_audio_prediction"] = noise_pred_audio
        state.extra["last_audio_prediction"] = noise_pred_audio
        return noise_pred_video

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
        video_output = self.pipeline.scheduler.step(
            model_prediction.float(),
            timestep,
            latents,
            return_dict=False,
        )
        audio_prediction = state.extra.pop(
            "base_audio_prediction", state.extra.get("last_audio_prediction")
        )
        if audio_prediction is None:
            raise NotImplementedError(
                "LTX2 scheduler_step needs an audio prediction from predict_vector_field."
            )
        audio_scheduler = state.extra.get("audio_scheduler")
        if audio_scheduler is None:
            raise NotImplementedError("LTX2 adapter state is missing its audio scheduler.")
        audio_output = audio_scheduler.step(
            audio_prediction.float(),
            timestep,
            state.extra["audio_latents"],
            return_dict=False,
        )
        state.extra["audio_latents"] = (
            audio_output[0] if isinstance(audio_output, tuple) else audio_output
        )
        state.extra["step_index"] = int(state.extra.get("step_index", 0)) + 1
        return SchedulerStepResult(
            latents=video_output[0] if isinstance(video_output, tuple) else video_output,
            state=state,
        )

    def decode_latents(self, latents: torch.Tensor, state: AdapterState) -> Any:
        self._require_loaded()
        assert self.pipeline is not None
        unpacked = self.pipeline._unpack_latents(
            latents,
            int(state.extra["latent_num_frames"]),
            int(state.extra["latent_height"]),
            int(state.extra["latent_width"]),
            self.pipeline.transformer_spatial_patch_size,
            self.pipeline.transformer_temporal_patch_size,
        )
        unpacked = unpacked.to(self.dtype)
        if not self.pipeline.vae.config.timestep_conditioning:
            decode_timestep = None
        else:
            generator = state.extra.get("generator")
            noise = torch.randn(
                unpacked.shape,
                generator=generator,
                device=unpacked.device,
                dtype=unpacked.dtype,
            )
            batch_size = unpacked.shape[0]
            timestep_value = state.extra.get("decode_timestep", 0.0)
            noise_scale_value = state.extra.get("decode_noise_scale")
            if not isinstance(timestep_value, list):
                timestep_value = [timestep_value] * batch_size
            if noise_scale_value is None:
                noise_scale_value = timestep_value
            elif not isinstance(noise_scale_value, list):
                noise_scale_value = [noise_scale_value] * batch_size
            decode_timestep = torch.tensor(
                timestep_value, device=unpacked.device, dtype=unpacked.dtype
            )
            decode_noise_scale = torch.tensor(
                noise_scale_value,
                device=unpacked.device,
                dtype=unpacked.dtype,
            )[:, None, None, None, None]
            unpacked = (1 - decode_noise_scale) * unpacked + decode_noise_scale * noise
        unpacked = self.pipeline._denormalize_latents(
            unpacked,
            self.pipeline.vae.latents_mean,
            self.pipeline.vae.latents_std,
            self.pipeline.vae.config.scaling_factor,
        )
        unpacked = unpacked.to(self.pipeline.vae.dtype)
        video = self.pipeline.vae.decode(unpacked, decode_timestep, return_dict=False)[0]
        frames = self.pipeline.video_processor.postprocess_video(video, output_type="pil")
        internal_num_frames = int(state.extra["num_frames"])
        output_num_frames = int(state.extra.get("output_num_frames", internal_num_frames))
        output = _terminal_crop_exact(
            frames,
            internal_frames=internal_num_frames,
            output_frames=output_num_frames,
        )
        completed = dict(state.extra["temporal_provenance"])
        completed.update(
            {
                "status": (
                    "pilot_decode_completed_and_frame_arithmetic_validated"
                    if completed.get("execution_phase") == "pilot"
                    else "decode_completed_and_frame_arithmetic_validated"
                ),
                "decoded_internal_frames": internal_num_frames,
                "saved_output_frames": output_num_frames,
                "joint_audio_latents_conditioned_video": True,
                "audio_decoded": False,
                "audio_saved_to_output_media": False,
            }
        )
        self._last_temporal_provenance = completed
        return output

    def conditioning_provenance(self) -> dict[str, Any]:
        provenance = dict(super().conditioning_provenance())
        if self._diffusers_source_provenance is not None:
            provenance["diffusers_source"] = dict(self._diffusers_source_provenance)
        protocol = self._temporal_protocol()
        if protocol is not None:
            provenance["temporal_generation"] = (
                dict(self._last_temporal_provenance)
                if self._last_temporal_provenance is not None
                else self._configured_temporal_provenance(protocol)
            )
        return provenance

    def _temporal_protocol(self) -> dict[str, Any] | None:
        raw = self.config.get(LTX_TEMPORAL_PROTOCOL_KEY)
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError(f"model.{LTX_TEMPORAL_PROTOCOL_KEY} must be a mapping.")
        protocol = dict(raw)
        required = set(_EXPECTED_LTX_TEMPORAL_PROTOCOL) | {"execution_phase"}
        missing = sorted(required - set(protocol))
        if missing:
            raise ValueError(f"{LTX_TEMPORAL_PROTOCOL_KEY} is missing frozen fields: {missing}.")
        unknown = sorted(
            set(protocol) - set(_EXPECTED_LTX_TEMPORAL_PROTOCOL) - _LTX_TEMPORAL_DYNAMIC_FIELDS
        )
        if unknown:
            raise ValueError(f"{LTX_TEMPORAL_PROTOCOL_KEY} has unknown fields: {unknown}.")
        mismatches = {
            key: {"expected": expected, "actual": protocol[key]}
            for key, expected in _EXPECTED_LTX_TEMPORAL_PROTOCOL.items()
            if protocol[key] != expected
        }
        if mismatches:
            raise ValueError(
                "LTX-2.3 temporal protocol differs from the frozen 15-second pilot "
                f"contract: {mismatches}."
            )
        if protocol["execution_phase"] not in {"pilot", "production"}:
            raise ValueError("LTX-2.3 temporal execution_phase must be 'pilot' or 'production'.")
        return protocol

    def _validate_temporal_execution_gate(self, protocol: Mapping[str, Any]) -> None:
        _validate_production_gate(protocol, criteria_names=_LTX_PRODUCTION_CRITERIA)

    def _validate_requested_temporal_output(
        self,
        generation_kwargs: Mapping[str, Any],
        protocol: Mapping[str, Any],
    ) -> None:
        actual = {
            "num_frames": int(generation_kwargs.get("num_frames", -1)),
            "fps": float(
                generation_kwargs.get(
                    "frame_rate",
                    generation_kwargs.get("fps", -1.0),
                )
            ),
            "duration_seconds": float(generation_kwargs.get("duration_seconds", -1.0)),
            "height": int(generation_kwargs.get("height", -1)),
            "width": int(generation_kwargs.get("width", -1)),
        }
        expected = {
            "num_frames": int(protocol["output_frames"]),
            "fps": float(protocol["output_fps"]),
            "duration_seconds": float(protocol["duration_seconds"]),
            "height": int(protocol["height"]),
            "width": int(protocol["width"]),
        }
        if actual != expected:
            raise ValueError(
                "LTX-2.3 temporal pilot requires the exact frozen output/resolution "
                f"contract; expected {expected}, got {actual}."
            )
        _validate_optional_duration(
            generation_kwargs,
            output_frames=int(protocol["output_frames"]),
            output_fps=float(protocol["output_fps"]),
        )

    def _configured_temporal_provenance(
        self,
        protocol: Mapping[str, Any],
    ) -> dict[str, Any]:
        internal_frames = int(protocol["internal_generated_frames"])
        internal_fps = float(protocol["internal_fps"])
        output_frames = int(protocol["output_frames"])
        return {
            "schema_version": 1,
            "status": "configured_not_yet_decoded",
            "execution_phase": protocol["execution_phase"],
            "strategy": protocol["strategy"],
            "scientific_classification": (
                "api_supported_non_default_fps_and_single_shot_length_pilot"
                if protocol["execution_phase"] == "pilot"
                else "api_supported_non_default_fps_and_single_shot_length_production_gated"
            ),
            "released_api_evidence": {
                "default_frames": int(protocol["api_default_frames"]),
                "default_fps": float(protocol["api_default_fps"]),
                "frame_rate_is_threaded_to_video_rope": True,
                "frame_rate_is_threaded_to_transformer": True,
                "audio_length_uses_num_frames_divided_by_frame_rate": True,
            },
            "source_pins": {
                "diffusers_revision": protocol["diffusers_revision"],
                "pipeline_source_sha256": protocol["pipeline_source_sha256"],
                "video_vae_source_sha256": protocol["video_vae_source_sha256"],
            },
            "internal_trajectory": {
                "frames": internal_frames,
                "fps": internal_fps,
                "duration_seconds_by_released_arithmetic": internal_frames / internal_fps,
                "video_latent_frames": (
                    1 + (internal_frames - 1) // int(protocol["vae_temporal_compression_ratio"])
                ),
            },
            "postprocessing": {
                "classification": "single_terminal_frame_crop",
                "endpoint_crop": protocol["terminal_endpoint_crop"],
                "dropped_frame_count": internal_frames - output_frames,
                "time_domain": "half_open_[0,duration_seconds)",
                "frame_pts_rule": "pts(frame_index)=frame_index/output_fps",
                "first_output_pts_seconds": 0.0,
                "last_output_pts_seconds": ((output_frames - 1) / float(protocol["output_fps"])),
                "dropped_terminal_frame_pts_seconds": (
                    output_frames / float(protocol["output_fps"])
                ),
                "frame_resampling": protocol["frame_resampling"],
                "output_frames": output_frames,
                "output_fps": int(protocol["output_fps"]),
                "duration_seconds": float(protocol["duration_seconds"]),
                "slow_motion_used": False,
                "duplicated_frames": 0,
                "synthesized_frames": 0,
            },
            "audio_output": {
                "joint_audio_latents_condition_video": True,
                "decoded_during_adapter_decode": False,
                "saved_to_mp4": False,
            },
            "production_gate": protocol.get("production_gate"),
        }


def _validate_diffusers_source_contract(protocol: Mapping[str, Any]) -> dict[str, Any]:
    try:
        distribution = metadata.distribution("diffusers")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("Pinned LTX-2.3 execution requires Diffusers to be installed.") from exc
    direct_url = distribution.read_text("direct_url.json")
    if not direct_url:
        raise RuntimeError(
            "Pinned LTX-2.3 execution requires a VCS-installed Diffusers direct_url.json."
        )
    try:
        direct_url_payload = json.loads(direct_url)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Diffusers direct_url.json is not valid JSON.") from exc
    installed_revision = direct_url_payload.get("vcs_info", {}).get("commit_id")
    expected_revision = protocol["diffusers_revision"]
    if installed_revision != expected_revision:
        raise RuntimeError(
            "Pinned LTX-2.3 Diffusers revision mismatch: expected "
            f"{expected_revision}, got {installed_revision!r}."
        )

    source_contract = {
        "pipeline_ltx2.py": (
            "diffusers/pipelines/ltx2/pipeline_ltx2.py",
            protocol["pipeline_source_sha256"],
        ),
        "autoencoder_kl_ltx2.py": (
            "diffusers/models/autoencoders/autoencoder_kl_ltx2.py",
            protocol["video_vae_source_sha256"],
        ),
    }
    records: list[dict[str, Any]] = []
    for label, (relative_path, expected_sha256) in source_contract.items():
        source_path = Path(distribution.locate_file(relative_path)).resolve()
        if not source_path.is_file():
            raise RuntimeError(f"Pinned LTX-2.3 Diffusers source file is missing: {source_path}.")
        actual_sha256 = _sha256_file(source_path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Pinned LTX-2.3 {label} SHA-256 mismatch: expected "
                f"{expected_sha256}, got {actual_sha256}."
            )
        records.append(
            {
                "file": label,
                "path": str(source_path),
                "sha256": actual_sha256,
            }
        )
    return {
        "schema_version": 1,
        "repository": direct_url_payload.get("url"),
        "revision": installed_revision,
        "files": records,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_positive_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite positive number; got {value!r}.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite positive number; got {value!r}.") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{field_name} must be a finite positive number; got {value!r}.")
    return parsed


def _validate_optional_duration(
    generation_kwargs: Mapping[str, Any],
    *,
    output_frames: int,
    output_fps: float,
) -> None:
    if output_frames <= 0:
        raise ValueError("LTX-2.3 output frame count must be positive.")
    expected_duration = output_frames / output_fps
    raw_duration = generation_kwargs.get("duration_seconds")
    if raw_duration is None:
        return
    duration = _finite_positive_float(raw_duration, "LTX-2.3 duration_seconds")
    if duration != expected_duration:
        raise ValueError(
            "LTX-2.3 requested duration/fps/frame arithmetic is inconsistent: "
            f"{output_frames}/{output_fps}={expected_duration}, got duration_seconds={duration}."
        )


def _validate_production_gate(
    protocol: Mapping[str, Any],
    *,
    criteria_names: tuple[str, ...],
) -> None:
    validate_temporal_production_gate(
        protocol,
        model_name="ltx_23",
        model_revision=LTX_CHECKPOINT_REVISION,
        criteria_names=criteria_names,
    )


def _direct_temporal_provenance(
    *,
    internal_frames: int,
    output_frames: int,
    fps: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "configured_not_yet_decoded",
        "strategy": "direct_aligned_generation",
        "internal_trajectory": {
            "frames": internal_frames,
            "fps": fps,
            "duration_seconds": internal_frames / fps,
        },
        "postprocessing": {
            "classification": "none",
            "output_frames": output_frames,
            "output_fps": fps,
            "duration_seconds": output_frames / fps,
        },
    }


def _terminal_crop_exact(
    videos: Any,
    *,
    internal_frames: int,
    output_frames: int,
) -> list[list[Any]]:
    if not isinstance(videos, (list, tuple)) or not videos:
        raise RuntimeError("LTX-2.3 decoder did not return a non-empty video batch.")
    if output_frames <= 0 or internal_frames < output_frames:
        raise RuntimeError(
            "LTX-2.3 invalid terminal-crop arithmetic: "
            f"internal={internal_frames}, output={output_frames}."
        )
    outputs: list[list[Any]] = []
    for video_index, frames in enumerate(videos):
        if not isinstance(frames, (list, tuple)):
            raise RuntimeError(f"LTX-2.3 decoded video {video_index} is not a frame sequence.")
        if len(frames) != internal_frames:
            raise RuntimeError(
                f"LTX-2.3 decoded {len(frames)} frames for video {video_index}; "
                f"expected exactly {internal_frames} before terminal crop."
            )
        cropped = list(frames[:output_frames])
        if len(cropped) != output_frames:
            raise RuntimeError(
                f"LTX-2.3 terminal crop produced {len(cropped)} frames; expected {output_frames}."
            )
        outputs.append(cropped)
    return outputs


def _slice_ltx_condition_data(
    data: dict[str, Any], index: int, batch_size: int, do_cfg: bool
) -> dict[str, Any]:
    sliced = _slice_condition_data(data, index, batch_size)
    if not do_cfg:
        return sliced

    for key in (
        "connector_prompt_embeds",
        "connector_audio_prompt_embeds",
        "connector_attention_mask",
    ):
        value = data.get(key)
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size * 2:
            sliced[key] = torch.cat(
                [
                    value[index : index + 1],
                    value[batch_size + index : batch_size + index + 1],
                ],
                dim=0,
            )
    return sliced
