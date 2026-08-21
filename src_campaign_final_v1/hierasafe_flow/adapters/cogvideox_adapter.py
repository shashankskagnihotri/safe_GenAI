from __future__ import annotations

import gc
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from hierasafe_flow.adapters.base import (
    AdapterState,
    DenoisingStepContext,
    DiffusersFrozenAdapter,
    PromptCondition,
    SchedulerStepResult,
    configure_pipeline_vae_tiling,
)
from hierasafe_flow.evaluation.temporal_qualification import (
    TEMPORAL_CRITERIA_BY_MODEL,
    validate_temporal_production_gate,
)
from hierasafe_flow.generation.conditioning_cache import (
    canonical_identity_sha256,
    encoding_fingerprint,
)
from hierasafe_flow.generation.rife_interpolation import (
    RifeMidpointInterpolator,
    rife_2x_and_crop_exact,
)
from hierasafe_flow.generation.temporal_artifacts import (
    TemporalEvidenceBundle,
    TemporalSegmentEvidence,
    frame_rgb_sha256,
)


COGVIDEOX_TEMPORAL_PROTOCOL_KEY = "cogvideox_temporal_protocol"
COGVIDEOX_T2V_MODEL_ID = "zai-org/CogVideoX-5b"
COGVIDEOX_I2V_MODEL_ID = "zai-org/CogVideoX-5b-I2V"
COGVIDEOX_CHECKPOINT_REVISION = "8fc5b281006c82b82d34fd2543d2f0ebb4e7e321"
COGVIDEOX_I2V_CHECKPOINT_REVISION = "a6f0f4858a8395e7429d82493864ce92bf73af11"
COGVIDEOX_SEGMENT_SEED_DOMAIN = "cogvideox_5b_i2v_continuation_v1"
COGVIDEOX_ARTIFACT_MANIFEST_SHA256 = (
    "bb65f9ea7607f2b75d63aeb342b7f3fb71d771f8e438d059441893feab6729bc"
)

_SCHEMA2_STRATEGY = "official_family_t2v_i2v_three_segment_composite"
_COGVIDEOX_TEMPORAL_DYNAMIC_FIELDS = {"execution_phase", "production_gate"}
_COGVIDEOX_PRODUCTION_CRITERIA = TEMPORAL_CRITERIA_BY_MODEL["cogvideox_5b"]
_EXPECTED_COGVIDEOX_TEMPORAL_PROTOCOL: Mapping[str, Any] = {
    "schema_version": 2,
    "native_temporal_call_schema_version": 2,
    "segment_trace_schema_version": 1,
    "temporal_evidence_schema_version": 1,
    "segmented_temporal_audit_schema_version": 1,
    "strategy": _SCHEMA2_STRATEGY,
    "scientific_label": (
        "CogVideoX-5b official-family T2V+I2V composite "
        "(not single-checkpoint native 15s)"
    ),
    "checkpoints": {
        "primary": {
            "model_id": COGVIDEOX_T2V_MODEL_ID,
            "revision": COGVIDEOX_CHECKPOINT_REVISION,
        },
        "continuation": {
            "model_id": COGVIDEOX_I2V_MODEL_ID,
            "revision": COGVIDEOX_I2V_CHECKPOINT_REVISION,
        },
    },
    "artifact_manifest": "configs/artifacts/cogvideox_5b_segmented_temporal_v2.json",
    "artifact_manifest_sha256": COGVIDEOX_ARTIFACT_MANIFEST_SHA256,
    "segments": [
        {"index": 0, "role": "t2v_primary", "frames": 49, "fps": 8, "steps": 50},
        {"index": 1, "role": "i2v_continuation", "frames": 49, "fps": 8, "steps": 50},
        {"index": 2, "role": "i2v_continuation", "frames": 49, "fps": 8, "steps": 50},
    ],
    "precision": "bfloat16",
    "scheduler": "CogVideoXDDIMScheduler",
    "guidance_scale": 6.0,
    "seed_domain": COGVIDEOX_SEGMENT_SEED_DOMAIN,
    "shared_components": {
        "tokenizer_sha256": (
            "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86"
        ),
        "text_encoder_sha256": (
            "bf396899ee29ab16d5151e02df3038938bdfde0f1306b7941ec000dd18fb41a7"
        ),
        "vae_sha256": (
            "a410e48d988c8224cef392b68db0654485cfd41f345f4a3a81d3e6b765bb995e"
        ),
        "preserve_runtime_object_identity": True,
    },
    "stitch": {
        "retained": ["segment_0[0:49]", "segment_1[1:49]", "segment_2[1:25]"],
        "native_frame_count": 121,
    },
    "postprocess": {
        "method": "pinned_rife_midpoint_once_over_complete_stitch",
        "code_revision": "5d8adbdd40e12c2c8f91930eff838aebe561c086",
        "weights_revision": "440cdec905de98e1d7e81f65d2c88a08da7cb4e2",
        "weights_sha256": (
            "fe854fc8996547c953f732aaa3b78cae76cc0a12833ae856ea0749c4c570d7d8"
        ),
        "scale": 1.0,
        "input_frames": 121,
        "inclusive_frames": 241,
        "output_slice": "[0:240]",
        "output_frames": 240,
        "output_fps": 16,
        "duration_seconds": 15.0,
    },
}
_OBSOLETE_PROTOCOL_KEYS = {
    "native_generated_frames",
    "riflex_k",
    "riflex_intrinsic_period_latent_frames",
    "single_trajectory",
    "timestep_spacing",
}
_EXPECTED_ARTIFACT_PATHS_BY_ROLE = {
    "primary": frozenset(
        {
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model-00001-of-00002.safetensors",
            "text_encoder/model-00002-of-00002.safetensors",
            "text_encoder/model.safetensors.index.json",
            "tokenizer/added_tokens.json",
            "tokenizer/special_tokens_map.json",
            "tokenizer/spiece.model",
            "tokenizer/tokenizer_config.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        }
    ),
    "continuation": frozenset(
        {
            ".gitattributes",
            "LICENSE",
            "README.md",
            "README_zh.md",
            "configuration.json",
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model-00001-of-00002.safetensors",
            "text_encoder/model-00002-of-00002.safetensors",
            "text_encoder/model.safetensors.index.json",
            "tokenizer/added_tokens.json",
            "tokenizer/special_tokens_map.json",
            "tokenizer/spiece.model",
            "tokenizer/tokenizer_config.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
            "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        }
    ),
}


class CogVideoXAdapter(DiffusersFrozenAdapter):
    """CogVideoX-5b 1.0 official-family T2V→I2V composite.

    Every native call remains at 49 frames, 8 fps, DDIM/50 steps/CFG 6.
    Only the 16-channel noise vector field is runner-visible in I2V segments;
    the 16-channel image condition is authenticated fixed state.
    """

    adapter_name = "cogvideox"
    task_type = "text_to_video"
    pipeline_class_name = "CogVideoXPipeline"
    latent_feature_dim = 2  # [B, F, C, H, W]
    required_components = ("transformer", "scheduler", "vae")
    encode_prompt_output_names = ("prompt_embeds", "negative_prompt_embeds")

    def __init__(
        self,
        model_id: str,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(model_id=model_id, device=device, dtype=dtype, config=config)
        self._pipeline_role = "t2v"
        self._native_segment_timesteps: list[Any] = []
        self._completed_segments: list[list[list[Any]]] = []
        self._segment_seeds: list[int] = []
        self._segment_conditioning: list[dict[str, Any]] = []
        self._segment_fixed_fingerprints: list[str | None] = []
        self._segment_protected_state: list[dict[str, Any]] = []
        self._temporal_evidence: TemporalEvidenceBundle | None = None
        self._last_temporal_provenance: dict[str, Any] | None = None
        self._rife_midpoint_interpolator: Any | None = None
        self._artifact_authentication: dict[str, Any] | None = None

    def load(self) -> None:
        protocol = self._temporal_protocol()
        if protocol is not None:
            self._validate_temporal_execution_gate(protocol)
            self._validate_preload_contract(protocol)
            self._artifact_authentication = self._authenticate_artifact_manifest(protocol)
        super().load()
        self._pipeline_role = "t2v"
        if protocol is not None:
            self._validate_primary_checkpoint_contract()

    def prepare_initial_latents(
        self,
        prompt: str,
        batch_size: int,
        generator: torch.Generator | None,
        **generation_kwargs: Any,
    ) -> tuple[torch.Tensor, AdapterState]:
        self._require_loaded()
        protocol = self._temporal_protocol()
        if protocol is None:
            requested = int(generation_kwargs.get("num_frames", 49))
            if requested <= 0 or requested > 49 or (requested - 1) % 8:
                raise ValueError("CogVideoX-5b native frame counts must be 8*N+1 and at most 49.")
            assert self.pipeline is not None
            compression = int(getattr(self.pipeline, "vae_scale_factor_temporal", 4))
            latent_frames = (requested - 1) // compression + 1
            patch_size_t = getattr(
                getattr(self.pipeline.transformer, "config", None),
                "patch_size_t",
                None,
            )
            additional_latent_frames = 0
            if patch_size_t is not None and latent_frames % int(patch_size_t):
                additional_latent_frames = int(patch_size_t) - latent_frames % int(patch_size_t)
            internal_num_frames = requested + additional_latent_frames * compression
            latent_kwargs = dict(generation_kwargs)
            latent_kwargs["num_frames"] = internal_num_frames
            latents, state = super().prepare_initial_latents(
                prompt, batch_size, generator, **latent_kwargs
            )
            state.extra.update(
                {
                    "num_frames": internal_num_frames,
                    "output_num_frames": requested,
                    "native_generated_num_frames": requested,
                    "native_latent_shape": list(latents.shape),
                    "native_latent_shape_observed": True,
                    "expected_native_latent_shape": list(latents.shape),
                    "additional_latent_frames": additional_latent_frames,
                    "height": int(generation_kwargs["height"]),
                    "width": int(generation_kwargs["width"]),
                }
            )
            if bool(
                getattr(
                    self.pipeline.transformer.config,
                    "use_rotary_positional_embeddings",
                    False,
                )
            ):
                state.extra["image_rotary_emb"] = (
                    self.pipeline._prepare_rotary_positional_embeddings(
                        int(generation_kwargs["height"]),
                        int(generation_kwargs["width"]),
                        latents.size(1),
                        self.device,
                    )
                )
            self._last_temporal_provenance = None
            return latents, state
        self._validate_temporal_execution_gate(protocol)
        self._validate_requested_output_contract(generation_kwargs, protocol)
        if batch_size != 1:
            raise RuntimeError("CogVideoX segmented protocol is qualified only at batch size 1.")
        native_kwargs = dict(generation_kwargs)
        native_kwargs.update(num_frames=49, height=480, width=720)
        latents, state = super().prepare_initial_latents(
            prompt,
            batch_size,
            generator,
            **native_kwargs,
        )
        if latents.ndim != 5 or int(latents.shape[2]) != 16:
            raise RuntimeError(
                "CogVideoX T2V segment must expose 16 dynamic/noise channels; "
                f"got {tuple(latents.shape)}."
            )
        base_seed = int(generator.initial_seed()) if generator is not None else 0
        state.extra.update(
            {
                "prompt": prompt,
                "base_seed": base_seed,
                "segment_seed": base_seed,
                "segment_index": 0,
                "segment_count": 3,
                "segment_step_index": 0,
                "condition_epoch": 0,
                "model_role": "t2v_primary",
                "model_id": COGVIDEOX_T2V_MODEL_ID,
                "model_revision": COGVIDEOX_CHECKPOINT_REVISION,
                "anchor_sha256": None,
                "height": 480,
                "width": 720,
                "native_num_frames": 49,
                "image_rotary_emb": self._prepare_rotary(latents, 480, 720),
            }
        )
        self._completed_segments = []
        self._segment_seeds = [base_seed]
        self._segment_conditioning = [
            {"schema_version": 1, "mode": "state_aware_text", "records": {}}
        ]
        self._segment_fixed_fingerprints = [None]
        self._segment_protected_state = [
            {
                "schema_version": 1,
                "model_role": "t2v_primary",
                "fixed_image_condition": "not_applicable",
                "dynamic_channels": 16,
                "branch_inputs": {},
            }
        ]
        self._temporal_evidence = None
        self._last_temporal_provenance = None
        return latents, state

    def set_timesteps(
        self,
        num_inference_steps: int,
        latents: torch.Tensor | None = None,
        state: AdapterState | None = None,
    ) -> list[Any]:
        protocol = self._temporal_protocol()
        if protocol is None:
            return super().set_timesteps(num_inference_steps, latents=latents, state=state)
        if num_inference_steps != 50:
            raise ValueError("CogVideoX schema-2 protocol requires exactly 50 steps per segment.")
        if state is None:
            raise RuntimeError("CogVideoX segmented scheduling requires AdapterState.")
        native = super().set_timesteps(50, latents=latents, state=state)
        if len(native) != 50:
            raise RuntimeError("CogVideoX native DDIM scheduler did not expose exactly 50 steps.")
        self._native_segment_timesteps = list(native)
        self.timesteps = list(native) * 3
        state.extra["local_num_steps"] = 50
        state.extra["global_num_steps"] = 150
        return self.timesteps

    def denoising_step_context(
        self,
        global_step_index: int,
        global_num_steps: int,
        state: AdapterState,
    ) -> DenoisingStepContext:
        if self._temporal_protocol() is None:
            return super().denoising_step_context(global_step_index, global_num_steps, state)
        if global_num_steps != 150:
            raise RuntimeError("CogVideoX schema-2 runner must execute exactly 150 global steps.")
        return DenoisingStepContext(
            global_step_index=global_step_index,
            global_num_steps=global_num_steps,
            segment_index=int(state.extra["segment_index"]),
            segment_count=3,
            local_step_index=int(state.extra["segment_step_index"]),
            local_num_steps=50,
            model_role=str(state.extra["model_role"]),
            model_id=str(state.extra["model_id"]),
            model_revision=str(state.extra["model_revision"]),
            condition_epoch=int(state.extra["condition_epoch"]),
            anchor_sha256=state.extra.get("anchor_sha256"),
            segment_seed=int(state.extra["segment_seed"]),
        )

    def conditioning_cache_identity(
        self,
        prompt: str,
        state: AdapterState,
        *,
        prompt_view: str,
        call_role: str,
    ) -> dict[str, Any]:
        identity = super().conditioning_cache_identity(
            prompt,
            state,
            prompt_view=prompt_view,
            call_role=call_role,
        )
        protocol = self._temporal_protocol()
        if protocol is not None:
            identity.update(
                {
                    "tokenizer_hash": protocol["shared_components"]["tokenizer_sha256"],
                    "text_encoder_hash": protocol["shared_components"][
                        "text_encoder_sha256"
                    ],
                }
            )
        return identity

    def prepare_prompt_for_state(
        self,
        prompt: str,
        state: AdapterState,
        *,
        prompt_view: str,
        call_role: str,
    ) -> PromptCondition:
        return self.prepare_prompts_for_state(
            [prompt],
            state,
            prompt_view=prompt_view,
            call_roles=[call_role],
        )[0]

    def prepare_prompts_for_state(
        self,
        prompts: list[str],
        state: AdapterState,
        *,
        prompt_view: str,
        call_roles: list[str],
    ) -> list[PromptCondition]:
        if len(prompts) != len(call_roles):
            raise ValueError("CogVideoX prompts and call roles must have identical lengths.")
        conditions = super().prepare_prompts_for_state(
            prompts,
            state,
            prompt_view=prompt_view,
            call_roles=call_roles,
        )
        if self._temporal_protocol() is None:
            return conditions
        segment_index = int(state.extra.get("segment_index", -1))
        if not 0 <= segment_index < len(self._segment_conditioning):
            raise RuntimeError("CogVideoX conditioning was prepared outside a registered segment.")
        records = self._segment_conditioning[segment_index]["records"]
        for prompt, call_role, condition in zip(
            prompts, call_roles, conditions, strict=True
        ):
            identity = self.conditioning_cache_identity(
                prompt,
                state,
                prompt_view=prompt_view,
                call_role=call_role,
            )
            record = {
                "call_role": call_role,
                "prompt_view": prompt_view,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "identity_sha256": canonical_identity_sha256(identity),
                "encoding_fingerprint": encoding_fingerprint(condition),
                "segment_index": segment_index,
                "condition_epoch": int(state.extra.get("condition_epoch", -1)),
                "anchor_sha256": state.extra.get("anchor_sha256"),
                "model_role": state.extra.get("model_role"),
            }
            record_id = hashlib.sha256(
                json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            prior = records.get(record_id)
            if prior is not None and prior != record:
                raise RuntimeError("CogVideoX condition-evidence digest collision.")
            records[record_id] = record
        return conditions

    def predict_vector_field(
        self,
        latents: torch.Tensor,
        timestep: Any,
        condition: PromptCondition,
        state: AdapterState,
    ) -> torch.Tensor:
        self._require_loaded()
        assert self.pipeline is not None
        prompt_embeds = condition.data.get("prompt_embeds")
        if not isinstance(prompt_embeds, torch.Tensor):
            raise RuntimeError("CogVideoX conditioning did not expose prompt embeddings.")
        state.extra["prompt_embeds_dtype"] = prompt_embeds.dtype
        guidance_scale = 6.0 if self._temporal_protocol() is not None else self._manual_guidance_scale()
        negative = condition.data.get("negative_prompt_embeds")
        do_cfg = guidance_scale > 1.0 and isinstance(negative, torch.Tensor)
        dynamic_input = torch.cat([latents] * 2) if do_cfg else latents
        if hasattr(self.pipeline.scheduler, "scale_model_input"):
            dynamic_input = self.pipeline.scheduler.scale_model_input(dynamic_input, timestep)
        if do_cfg:
            prompt_embeds = torch.cat([negative, prompt_embeds], dim=0)

        fixed_fingerprint: str | None = None
        expanded_fixed_fingerprint: str | None = None
        if self._pipeline_role == "i2v":
            fixed = state.extra.get("image_condition_latents")
            if not isinstance(fixed, torch.Tensor):
                raise RuntimeError("CogVideoX I2V segment is missing fixed image-condition latents.")
            fixed_fingerprint = encoding_fingerprint(fixed)
            expected = state.extra.get("fixed_image_condition_fingerprint")
            if fixed_fingerprint != expected:
                raise RuntimeError("CogVideoX fixed image condition changed between branch calls.")
            fixed_input = torch.cat([fixed] * 2) if do_cfg else fixed
            expanded_fixed_fingerprint = encoding_fingerprint(fixed_input)
            model_input = torch.cat([dynamic_input, fixed_input], dim=2)
            if int(model_input.shape[2]) != 32:
                raise RuntimeError("CogVideoX I2V transformer input must have 32 channels.")
        else:
            model_input = dynamic_input
            if int(model_input.shape[2]) != 16:
                raise RuntimeError("CogVideoX T2V transformer input must have 16 channels.")

        state.extra["protected_state_trace"] = {
            "dynamic_noise_fingerprint": encoding_fingerprint(dynamic_input),
            "fixed_image_condition_fingerprint": fixed_fingerprint,
            "cfg_expanded_fixed_image_condition_fingerprint": expanded_fixed_fingerprint,
            "transformer_input_fingerprint": encoding_fingerprint(model_input),
            "dynamic_channels": 16,
            "fixed_channels": 16 if self._pipeline_role == "i2v" else 0,
        }
        if self._temporal_protocol() is not None:
            segment_index = int(state.extra.get("segment_index", -1))
            if not 0 <= segment_index < len(self._segment_protected_state):
                raise RuntimeError("CogVideoX protected-state trace has an invalid segment.")
            condition_fingerprint = encoding_fingerprint(condition)
            branch_inputs = self._segment_protected_state[segment_index].setdefault(
                "branch_inputs", {}
            )
            branch_record = {
                "condition_encoding_fingerprint": condition_fingerprint,
                "dynamic_noise_fingerprint": state.extra["protected_state_trace"][
                    "dynamic_noise_fingerprint"
                ],
                "fixed_image_condition_fingerprint": fixed_fingerprint,
                "cfg_expanded_fixed_image_condition_fingerprint": expanded_fixed_fingerprint,
                "transformer_input_fingerprint": state.extra["protected_state_trace"][
                    "transformer_input_fingerprint"
                ],
            }
            prior_branch = branch_inputs.get(condition_fingerprint)
            if prior_branch is not None and prior_branch.get(
                "fixed_image_condition_fingerprint"
            ) != fixed_fingerprint:
                raise RuntimeError("CogVideoX branch changed its fixed image conditioning.")
            branch_inputs[condition_fingerprint] = branch_record
        candidates = {
            "hidden_states": model_input,
            "sample": model_input,
            "timestep": self._timestep_batch(timestep, model_input.shape[0]),
            "encoder_hidden_states": prompt_embeds,
            "ofs": self.config.get("ofs"),
            "image_rotary_emb": state.extra.get("image_rotary_emb"),
            "attention_kwargs": self.config.get("attention_kwargs"),
            "return_dict": True,
        }
        cache_context = (
            self.pipeline.transformer.cache_context("cond_uncond")
            if hasattr(self.pipeline.transformer, "cache_context")
            else nullcontext()
        )
        with cache_context:
            prediction = self._call_transformer(self.pipeline.transformer, candidates).float()
        if int(prediction.shape[2]) != 16:
            raise RuntimeError(
                "CogVideoX transformer must return only the 16-channel dynamic vector field."
            )
        if do_cfg:
            unconditional, conditional = prediction.chunk(2)
            prediction = unconditional + guidance_scale * (conditional - unconditional)
        if prediction.shape != latents.shape:
            raise RuntimeError(
                "CogVideoX dynamic prediction shape does not match runner-visible latents: "
                f"{tuple(prediction.shape)} != {tuple(latents.shape)}."
            )
        return prediction.to(dtype=latents.dtype)

    def scheduler_step(
        self,
        model_prediction: torch.Tensor,
        timestep: Any,
        latents: torch.Tensor,
        state: AdapterState,
        generator: torch.Generator | None = None,
    ) -> SchedulerStepResult:
        active_generator = state.extra.get("segment_generator", generator)
        result = super().scheduler_step(
            model_prediction,
            timestep,
            latents,
            state,
            generator=active_generator,
        )
        if self._temporal_protocol() is None:
            dtype = state.extra.get("prompt_embeds_dtype")
            native_latents = result.latents.to(dtype) if dtype is not None else result.latents
            return SchedulerStepResult(latents=native_latents, state=result.state)
        local_step = int(state.extra["segment_step_index"]) + 1
        state.extra["segment_step_index"] = local_step
        if local_step < 50:
            return result
        if local_step > 50:
            raise RuntimeError("CogVideoX segment scheduler advanced beyond 50 steps.")
        segment_index = int(state.extra["segment_index"])
        if segment_index == 2:
            return result
        segment = self._decode_native_segment(result.latents)
        self._append_segment(segment)
        anchor = segment[0][-1]
        anchor_sha = frame_rgb_sha256(anchor)
        if segment_index == 0:
            self._load_continuation_pipeline()
        next_index = segment_index + 1
        next_latents = self._start_i2v_segment(
            anchor=anchor,
            anchor_sha256=anchor_sha,
            segment_index=next_index,
            state=state,
        )
        return SchedulerStepResult(latents=next_latents, state=state)

    def decode_latents(self, latents: torch.Tensor, state: AdapterState) -> Any:
        if self._temporal_protocol() is None:
            self._require_loaded()
            assert self.pipeline is not None
            if latents.ndim != 5 or int(latents.shape[2]) != 16:
                raise RuntimeError(
                    "CogVideoX native decode requires runner latents shaped [B,F,16,H,W]."
                )
            additional_latent_frames = int(state.extra.get("additional_latent_frames", 0))
            if additional_latent_frames > 0:
                latents = latents[:, additional_latent_frames:]
            if not hasattr(self.pipeline, "decode_latents") or not hasattr(
                self.pipeline, "video_processor"
            ):
                raise RuntimeError("CogVideoX pipeline lacks its native video decode path.")
            video = self.pipeline.decode_latents(latents)
            frames = self.pipeline.video_processor.postprocess_video(
                video=video,
                output_type="pil",
            )
            requested = int(
                state.extra.get("output_num_frames", state.extra.get("num_frames", 0))
            )
            if requested > 0 and isinstance(frames, list):
                return [
                    list(item[:requested]) if isinstance(item, (list, tuple)) else item
                    for item in frames
                ]
            return frames
        if int(state.extra.get("segment_index", -1)) != 2 or int(
            state.extra.get("segment_step_index", -1)
        ) != 50:
            raise RuntimeError("CogVideoX decode requested before all three segments completed.")
        self._append_segment(self._decode_native_segment(latents))
        outputs = self._stitch_and_interpolate_segments(
            self._completed_segments,
            segment_seeds=self._segment_seeds,
            generation_path="adapter_vector_field_runner",
        )
        return outputs

    def take_temporal_evidence(
        self,
        state: AdapterState,
    ) -> TemporalEvidenceBundle | None:
        del state
        evidence = self._temporal_evidence
        self._temporal_evidence = None
        return evidence

    def configure_native_pipeline_for_temporal_protocol(self) -> dict[str, Any]:
        self._require_loaded()
        protocol = self._temporal_protocol()
        if protocol is None:
            raise RuntimeError("CogVideoX native temporal route requires schema-2 protocol.")
        self._validate_temporal_execution_gate(protocol)
        self._validate_primary_checkpoint_contract()
        return {
            "schema_version": 2,
            "strategy": _SCHEMA2_STRATEGY,
            "segment_count": 3,
            "first_call_role": "t2v_primary",
            "checkpoint_id": COGVIDEOX_T2V_MODEL_ID,
            "checkpoint_revision": COGVIDEOX_CHECKPOINT_REVISION,
            "num_frames": 49,
            "native_fps": 8,
            "height": 480,
            "width": 720,
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
            "scheduler": "CogVideoXDDIMScheduler",
            "output_frames": 240,
            "output_fps": 16,
            "completion_hook": "complete_native_pipeline_temporal_protocol",
        }

    def complete_native_pipeline_temporal_protocol(
        self,
        first_segment_media: Any,
        *,
        prompt: str,
        negative_prompt: str,
        generator: torch.Generator,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        height: int = 480,
        width: int = 720,
        **_: Any,
    ) -> list[list[Any]]:
        if not prompt or not negative_prompt:
            raise RuntimeError("CogVideoX native negative requires exact non-empty prompts.")
        if (num_inference_steps, guidance_scale, height, width) != (50, 6.0, 480, 720):
            raise RuntimeError("CogVideoX native temporal call contract drifted.")
        first = _validate_video_batch(first_segment_media, 49)
        if len(first) != 1:
            raise RuntimeError("CogVideoX native temporal completion requires batch size 1.")
        base_seed = int(generator.initial_seed())
        segments = [first]
        seeds = [base_seed]
        self._load_continuation_pipeline()
        assert self.pipeline is not None
        for segment_index in (1, 2):
            seed = _continuation_seed(base_seed, segment_index)
            seeds.append(seed)
            segment_generator = _make_generator(seed, self.device)
            kwargs = {
                "image": segments[-1][0][-1],
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "height": 480,
                "width": 720,
                "num_frames": 49,
                "num_inference_steps": 50,
                "guidance_scale": 6.0,
                "use_dynamic_cfg": False,
                "num_videos_per_prompt": 1,
                "generator": segment_generator,
                "output_type": "pil",
                "return_dict": True,
            }
            filtered = _validated_native_call_kwargs(self.pipeline, kwargs, role="i2v")
            output = self.pipeline(**filtered)
            media = output.frames if hasattr(output, "frames") else output[0]
            segments.append(_validate_video_batch(media, 49))
        return self._stitch_and_interpolate_segments(
            segments,
            segment_seeds=seeds,
            generation_path="official_native_negative_three_call",
            prompt=prompt,
            negative_prompt=negative_prompt,
        )

    def conditioning_provenance(self) -> dict[str, Any]:
        provenance = dict(super().conditioning_provenance())
        protocol = self._temporal_protocol()
        if protocol is not None:
            provenance["temporal_generation"] = (
                self._last_temporal_provenance
                or self._configured_temporal_provenance(protocol)
            )
        return provenance

    def _load_continuation_pipeline(self) -> None:
        if self._pipeline_role == "i2v":
            return
        assert self.pipeline is not None
        old_pipeline = self.pipeline
        shared = {
            "text_encoder": old_pipeline.text_encoder,
            "tokenizer": old_pipeline.tokenizer,
            "vae": old_pipeline.vae,
        }
        if hasattr(old_pipeline, "remove_all_hooks"):
            old_pipeline.remove_all_hooks()
        old_transformer = old_pipeline.transformer
        old_pipeline.transformer = None
        del old_transformer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        try:
            import diffusers
        except ModuleNotFoundError as exc:
            raise RuntimeError("CogVideoX continuation requires Diffusers.") from exc
        pipeline_cls = getattr(diffusers, "CogVideoXImageToVideoPipeline", None)
        if pipeline_cls is None:
            raise RuntimeError("Installed Diffusers lacks CogVideoXImageToVideoPipeline.")
        load_kwargs = dict(self.config.get("load_kwargs") or {})
        for key in (
            "single_file",
            "single_file_components",
            "text_encoder",
            "tokenizer",
            "vae",
        ):
            load_kwargs.pop(key, None)
        load_kwargs.update(
            {
                **shared,
                "revision": COGVIDEOX_I2V_CHECKPOINT_REVISION,
                "torch_dtype": self.dtype,
                "local_files_only": bool(self.config.get("local_files_only", False)),
                "low_cpu_mem_usage": True,
            }
        )
        self.pipeline = pipeline_cls.from_pretrained(COGVIDEOX_I2V_MODEL_ID, **load_kwargs)
        for name, component in shared.items():
            if getattr(self.pipeline, name, None) is not component:
                raise RuntimeError(
                    f"CogVideoX continuation did not preserve authenticated {name} object identity."
                )
        self._configure_pipeline_device_policy()
        self._freeze_pipeline()
        self._pipeline_role = "i2v"
        self._validate_continuation_checkpoint_contract()

    def _configure_pipeline_device_policy(self) -> None:
        assert self.pipeline is not None
        configure_pipeline_vae_tiling(self.pipeline, self.config.get("vae_tiling"))
        strategy = self._cpu_offload_strategy(self.config.get("cpu_offload", False))
        if strategy == "group":
            self._enable_group_offload(self.config.get("cpu_offload"))
        elif strategy == "sequential" and hasattr(
            self.pipeline, "enable_sequential_cpu_offload"
        ):
            self.pipeline.enable_sequential_cpu_offload(device=self.device)
        elif strategy == "model" and hasattr(self.pipeline, "enable_model_cpu_offload"):
            self.pipeline.enable_model_cpu_offload(device=self.device)
        elif hasattr(self.pipeline, "to"):
            self.pipeline.to(self.device)

    def _start_i2v_segment(
        self,
        *,
        anchor: Any,
        anchor_sha256: str,
        segment_index: int,
        state: AdapterState,
    ) -> torch.Tensor:
        if self._pipeline_role != "i2v":
            raise RuntimeError("CogVideoX I2V segment started before continuation load.")
        assert self.pipeline is not None
        seed = _continuation_seed(int(state.extra["base_seed"]), segment_index)
        segment_generator = _make_generator(seed, self.device)
        image = self.pipeline.video_processor.preprocess(anchor, height=480, width=720).to(
            self.device, dtype=self.dtype
        )
        output = self.pipeline.prepare_latents(
            image,
            1,
            16,
            49,
            480,
            720,
            self.dtype,
            self.device,
            segment_generator,
            None,
        )
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("CogVideoX I2V prepare_latents must return noise and image latents.")
        latents, image_condition = output
        if latents.shape != image_condition.shape or int(latents.shape[2]) != 16:
            raise RuntimeError("CogVideoX I2V dynamic/fixed latent shapes are invalid.")
        fixed_fingerprint = encoding_fingerprint(image_condition)
        self._reset_segment_scheduler()
        state.extra.update(
            {
                "segment_index": segment_index,
                "segment_step_index": 0,
                "condition_epoch": segment_index,
                "segment_seed": seed,
                "segment_generator": segment_generator,
                "model_role": "i2v_continuation",
                "model_id": COGVIDEOX_I2V_MODEL_ID,
                "model_revision": COGVIDEOX_I2V_CHECKPOINT_REVISION,
                "anchor_sha256": anchor_sha256,
                "image_condition_latents": image_condition,
                "fixed_image_condition_fingerprint": fixed_fingerprint,
                "image_rotary_emb": self._prepare_rotary(latents, 480, 720),
            }
        )
        self._segment_seeds.append(seed)
        self._segment_fixed_fingerprints.append(fixed_fingerprint)
        self._segment_conditioning.append(
            {"schema_version": 1, "mode": "state_aware_text", "records": {}}
        )
        self._segment_protected_state.append(
            {
                "schema_version": 1,
                "model_role": "i2v_continuation",
                "fixed_image_condition_fingerprint": fixed_fingerprint,
                "anchor_sha256": anchor_sha256,
                "dynamic_channels": 16,
                "fixed_channels": 16,
                "branch_inputs": {},
            }
        )
        return latents

    def _reset_segment_scheduler(self) -> None:
        assert self.pipeline is not None
        self.pipeline.scheduler.set_timesteps(50, device=self.device)
        if hasattr(self.pipeline.scheduler, "set_begin_index"):
            self.pipeline.scheduler.set_begin_index(0)
        observed = list(self.pipeline.scheduler.timesteps)
        if len(observed) != 50:
            raise RuntimeError("CogVideoX continuation scheduler reset did not produce 50 steps.")
        if self._native_segment_timesteps and any(
            not math.isclose(_as_float(a), _as_float(b), rel_tol=0.0, abs_tol=1.0e-6)
            for a, b in zip(observed, self._native_segment_timesteps)
        ):
            raise RuntimeError("CogVideoX T2V/I2V DDIM timestep schedules differ.")

    def _decode_native_segment(self, latents: torch.Tensor) -> list[list[Any]]:
        assert self.pipeline is not None
        if not hasattr(self.pipeline, "decode_latents"):
            raise RuntimeError("CogVideoX pipeline lacks native latent decoding.")
        video = self.pipeline.decode_latents(latents)
        frames = self.pipeline.video_processor.postprocess_video(video=video, output_type="pil")
        return _validate_video_batch(frames, 49)

    def _append_segment(self, segment: list[list[Any]]) -> None:
        validated = _validate_video_batch(segment, 49)
        if self._completed_segments and len(validated) != len(self._completed_segments[0]):
            raise RuntimeError("CogVideoX segment batch size changed.")
        self._completed_segments.append(validated)

    def _stitch_and_interpolate_segments(
        self,
        segments: Sequence[list[list[Any]]],
        *,
        segment_seeds: Sequence[int],
        generation_path: str,
        prompt: str | None = None,
        negative_prompt: str | None = None,
    ) -> list[list[Any]]:
        if len(segments) != 3 or len(segment_seeds) != 3:
            raise RuntimeError("CogVideoX composite requires exactly three complete segments.")
        validated = [_validate_video_batch(segment, 49) for segment in segments]
        if any(len(segment) != 1 for segment in validated):
            raise RuntimeError("CogVideoX evidence currently requires batch size 1.")
        source = [segment[0] for segment in validated]
        retained = [tuple(range(49)), tuple(range(1, 49)), tuple(range(1, 25))]
        discarded = [(), (0,), tuple([0, *range(25, 49)])]
        stitch_map = tuple(
            (segment_index, frame_index)
            for segment_index, indices in enumerate(retained)
            for frame_index in indices
        )
        stitched = [source[segment_index][frame_index] for segment_index, frame_index in stitch_map]
        if len(stitched) != 121:
            raise RuntimeError("CogVideoX stitch must be exactly 49+48+24=121 frames.")
        # The discarded tail was generated at full native length and is
        # authenticated through per-frame evidence below before omission.
        interpolator = self._get_rife_midpoint_interpolator()
        output, interpolation_record = rife_2x_and_crop_exact(
            stitched,
            midpoint_interpolator=interpolator,
            source_fps=8,
            output_fps=16,
            duration_seconds=15.0,
            output_frame_count=240,
        )
        if len(output) != 240:
            raise RuntimeError("CogVideoX interpolation/crop did not produce 240 frames.")
        prompt_sha = hashlib.sha256((prompt or "").encode()).hexdigest() if prompt else None
        negative_sha = (
            hashlib.sha256((negative_prompt or "").encode()).hexdigest()
            if negative_prompt
            else None
        )
        if generation_path == "adapter_vector_field_runner":
            if len(self._segment_conditioning) != 3 or len(
                self._segment_protected_state
            ) != 3:
                raise RuntimeError("CogVideoX runner route lacks complete segment evidence.")
            conditioning_evidence = tuple(
                json.loads(json.dumps(record)) for record in self._segment_conditioning
            )
            protected_evidence = tuple(
                json.loads(json.dumps(record)) for record in self._segment_protected_state
            )
            if any(not record.get("records") for record in conditioning_evidence):
                raise RuntimeError("CogVideoX segment lacks state-aware condition records.")
            if any(not record.get("branch_inputs") for record in protected_evidence):
                raise RuntimeError("CogVideoX segment lacks protected branch-input records.")
        else:
            if prompt_sha is None or negative_sha is None:
                raise RuntimeError("CogVideoX native route lacks positive/negative prompt evidence.")
            conditioning_evidence = tuple(
                {
                    "schema_version": 1,
                    "mode": "official_native_pipeline_prompt_pair",
                    "positive_prompt_sha256": prompt_sha,
                    "negative_prompt_sha256": negative_sha,
                    "call_roles": ["native_positive", "native_negative"],
                }
                for _ in range(3)
            )
            protected_evidence = tuple(
                {
                    "schema_version": 1,
                    "model_role": (
                        "t2v_primary" if index == 0 else "i2v_continuation"
                    ),
                    "fixed_image_condition": (
                        "not_applicable"
                        if index == 0
                        else "owned_by_official_i2v_pipeline"
                    ),
                    "anchor_sha256": (
                        None if index == 0 else frame_rgb_sha256(source[index - 1][-1])
                    ),
                    "steering_scope": "native_pipeline_cfg_without_manual_latent_intervention",
                }
                for index in range(3)
            )
        segment_evidence = tuple(
            TemporalSegmentEvidence(
                segment_index=index,
                model_role="t2v_primary" if index == 0 else "i2v_continuation",
                model_id=COGVIDEOX_T2V_MODEL_ID if index == 0 else COGVIDEOX_I2V_MODEL_ID,
                model_revision=(
                    COGVIDEOX_CHECKPOINT_REVISION
                    if index == 0
                    else COGVIDEOX_I2V_CHECKPOINT_REVISION
                ),
                segment_seed=int(segment_seeds[index]),
                native_fps=8,
                frames=tuple(source[index]),
                retained_indices=retained[index],
                discarded_indices=discarded[index],
                anchor_sha256=(frame_rgb_sha256(source[index - 1][-1]) if index else None),
                reconstruction_index=0 if index else None,
                first_motion_index=1 if index else None,
                scheduler={
                    "class": "CogVideoXDDIMScheduler",
                    "num_inference_steps": 50,
                    "guidance_scale": 6.0,
                    "precision": "bfloat16",
                },
                conditioning=conditioning_evidence[index],
                protected_state=protected_evidence[index],
            )
            for index in range(3)
        )
        protocol = self._temporal_protocol()
        assert protocol is not None
        self._temporal_evidence = TemporalEvidenceBundle(
            temporal_protocol=protocol,
            segments=segment_evidence,
            stitch_map=stitch_map,
            output_fps=16,
            output_frame_count=240,
            stitch_strategy="segment_0[0:49]+segment_1[1:49]+segment_2[1:25]",
            postprocess={
                "method": "pinned_rife_midpoint_once_over_complete_stitch",
                "input_frame_count": 121,
                "inclusive_output_frame_count": 241,
                "final_slice": "[0:240]",
                "seam_indices": [96, 97, 98, 192, 193, 194],
                "record": interpolation_record.to_dict(),
                "backend": (
                    interpolator.provenance()
                    if hasattr(interpolator, "provenance")
                    else {"backend": type(interpolator).__name__}
                ),
            },
            scientific_label=str(protocol["scientific_label"]),
            metadata={
                "generation_path": generation_path,
                "artifact_authentication": self._artifact_authentication,
            },
        )
        self._last_temporal_provenance = {
            **self._configured_temporal_provenance(protocol),
            "status": "completed",
            "generation_path": generation_path,
            "segment_seeds": list(segment_seeds),
            "native_segment_frame_counts": [49, 49, 49],
            "stitched_native_frame_count": 121,
            "inclusive_interpolated_frame_count": 241,
            "output_frame_count": 240,
        }
        return [output]

    def _get_rife_midpoint_interpolator(self) -> Any:
        if self._rife_midpoint_interpolator is None:
            protocol = self._temporal_protocol()
            assert protocol is not None
            self._rife_midpoint_interpolator = RifeMidpointInterpolator(
                device=self.device,
                weights_path=protocol["postprocess"].get("weights_path"),
                scale=float(protocol["postprocess"].get("scale", 1.0)),
            )
        return self._rife_midpoint_interpolator

    def _prepare_rotary(self, latents: torch.Tensor, height: int, width: int) -> Any:
        assert self.pipeline is not None
        if not bool(
            getattr(self.pipeline.transformer.config, "use_rotary_positional_embeddings", False)
        ):
            return None
        return self.pipeline._prepare_rotary_positional_embeddings(
            height, width, int(latents.shape[1]), self.device
        )

    def _configured_temporal_provenance(self, protocol: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "status": "configured_not_yet_completed",
            "execution_phase": protocol["execution_phase"],
            "strategy": _SCHEMA2_STRATEGY,
            "scientific_label": protocol["scientific_label"],
            "checkpoint_set": [
                {
                    "role": "primary",
                    "model_id": COGVIDEOX_T2V_MODEL_ID,
                    "revision": COGVIDEOX_CHECKPOINT_REVISION,
                },
                {
                    "role": "continuation",
                    "model_id": COGVIDEOX_I2V_MODEL_ID,
                    "revision": COGVIDEOX_I2V_CHECKPOINT_REVISION,
                },
            ],
            "artifact_manifest_sha256": protocol["artifact_manifest_sha256"],
            "production_gate": protocol.get("production_gate"),
            "segment_contract": {
                "count": 3,
                "native_frames": [49, 49, 49],
                "native_fps": 8,
                "steps": [50, 50, 50],
                "guidance_scale": 6.0,
                "scheduler": "CogVideoXDDIMScheduler",
            },
            "stitch": "49+48+24=121",
            "postprocess": {
                "method": protocol["postprocess"]["method"],
                "arithmetic": "121->241->[0:240]",
            },
            "segment_trace_schema_version": 1,
            "temporal_evidence_schema_version": 1,
        }

    def _temporal_protocol(self) -> dict[str, Any] | None:
        raw = self.config.get(COGVIDEOX_TEMPORAL_PROTOCOL_KEY)
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError(f"{COGVIDEOX_TEMPORAL_PROTOCOL_KEY} must be a mapping.")
        protocol = json.loads(json.dumps(raw))
        obsolete = sorted(_OBSOLETE_PROTOCOL_KEYS & set(protocol))
        if obsolete:
            raise ValueError(f"Obsolete CogVideoX temporal fields are forbidden: {obsolete}.")
        required = set(_EXPECTED_COGVIDEOX_TEMPORAL_PROTOCOL) | {"execution_phase"}
        missing = sorted(required - set(protocol))
        if missing:
            raise ValueError(f"CogVideoX temporal protocol is missing fields: {missing}.")
        unknown = sorted(
            set(protocol)
            - set(_EXPECTED_COGVIDEOX_TEMPORAL_PROTOCOL)
            - _COGVIDEOX_TEMPORAL_DYNAMIC_FIELDS
        )
        if unknown:
            raise ValueError(f"{COGVIDEOX_TEMPORAL_PROTOCOL_KEY} has unknown fields: {unknown}.")
        mismatches = {
            key: {"expected": expected, "actual": protocol[key]}
            for key, expected in _EXPECTED_COGVIDEOX_TEMPORAL_PROTOCOL.items()
            if protocol[key] != expected
        }
        if mismatches:
            raise ValueError(f"CogVideoX temporal schema-2 contract drifted: {mismatches}.")
        if protocol["execution_phase"] not in {"pilot", "production"}:
            raise ValueError("CogVideoX temporal execution_phase must be pilot or production.")
        return protocol

    def _validate_temporal_execution_gate(self, protocol: Mapping[str, Any]) -> None:
        validate_temporal_production_gate(
            protocol,
            model_name="cogvideox_5b",
            model_revision=COGVIDEOX_CHECKPOINT_REVISION,
            criteria_names=_COGVIDEOX_PRODUCTION_CRITERIA,
        )

    def _validate_preload_contract(self, protocol: Mapping[str, Any]) -> None:
        if self.model_id != COGVIDEOX_T2V_MODEL_ID:
            raise RuntimeError(
                f"CogVideoX schema-2 primary must be {COGVIDEOX_T2V_MODEL_ID}; got {self.model_id}."
            )
        if self.config.get("revision") != COGVIDEOX_CHECKPOINT_REVISION:
            raise RuntimeError("CogVideoX primary revision is not the sealed 40-hex revision.")
        if self.dtype != torch.bfloat16:
            raise RuntimeError("CogVideoX schema-2 protocol requires BF16 denoisers.")
        load_kwargs = dict(self.config.get("load_kwargs") or {})
        forbidden = sorted(
            key
            for key in ("single_file", "single_file_components", "text_encoder", "tokenizer", "vae")
            if load_kwargs.get(key) is not None
        )
        if forbidden:
            raise RuntimeError(
                "CogVideoX artifact provenance forbids component overrides: " + repr(forbidden)
            )
        if protocol["postprocess"].get("method") != (
            "pinned_rife_midpoint_once_over_complete_stitch"
        ):
            raise RuntimeError("CogVideoX postprocess must use the pinned one-pass midpoint route.")

    def _validate_primary_checkpoint_contract(self) -> None:
        assert self.pipeline is not None
        config = self.pipeline.transformer.config
        actual = (
            int(getattr(config, "in_channels", -1)),
            int(getattr(config, "out_channels", -1)),
            int(getattr(config, "sample_frames", -1)),
        )
        if actual != (16, 16, 49):
            raise RuntimeError(f"CogVideoX primary transformer contract drifted: {actual}.")
        if self.pipeline.scheduler.__class__.__name__ != "CogVideoXDDIMScheduler":
            raise RuntimeError("CogVideoX primary must retain the official DDIM scheduler.")

    def _validate_continuation_checkpoint_contract(self) -> None:
        assert self.pipeline is not None
        config = self.pipeline.transformer.config
        actual = (
            int(getattr(config, "in_channels", -1)),
            int(getattr(config, "out_channels", -1)),
            int(getattr(config, "sample_frames", -1)),
        )
        if actual != (32, 16, 49):
            raise RuntimeError(f"CogVideoX continuation transformer contract drifted: {actual}.")
        if self.pipeline.scheduler.__class__.__name__ != "CogVideoXDDIMScheduler":
            raise RuntimeError("CogVideoX continuation must retain the official DDIM scheduler.")

    def _validate_requested_output_contract(
        self,
        generation_kwargs: Mapping[str, Any],
        protocol: Mapping[str, Any],
    ) -> None:
        del protocol
        expected = {
            "num_frames": 240,
            "fps": 16,
            "duration_seconds": 15.0,
            "height": 480,
            "width": 720,
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
        }
        actual = {
            key: (
                float(generation_kwargs.get(key, -1))
                if key in {"duration_seconds", "guidance_scale"}
                else int(generation_kwargs.get(key, -1))
            )
            for key in expected
        }
        if actual != expected:
            raise ValueError(
                f"CogVideoX schema-2 output/generation contract must be {expected}; got {actual}."
            )

    def _authenticate_artifact_manifest(self, protocol: Mapping[str, Any]) -> dict[str, Any]:
        path = Path(str(protocol["artifact_manifest"]))
        if not path.is_absolute():
            project_root = Path(str(self.config.get("project_root", Path.cwd())))
            path = project_root / path
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != protocol["artifact_manifest_sha256"]:
            raise RuntimeError("CogVideoX artifact manifest digest mismatch.")
        document = json.loads(raw)
        if document.get("schema_version") != 1:
            raise RuntimeError("CogVideoX artifact manifest schema must be 1.")
        expected = [
            ("primary", COGVIDEOX_T2V_MODEL_ID, COGVIDEOX_CHECKPOINT_REVISION),
            ("continuation", COGVIDEOX_I2V_MODEL_ID, COGVIDEOX_I2V_CHECKPOINT_REVISION),
        ]
        checkpoints = document.get("checkpoints")
        if not isinstance(checkpoints, list) or [
            (entry.get("role"), entry.get("model_id"), entry.get("revision"))
            for entry in checkpoints
            if isinstance(entry, Mapping)
        ] != expected:
            raise RuntimeError("CogVideoX artifact manifest checkpoint set is incomplete.")
        try:
            from huggingface_hub import snapshot_download
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "CogVideoX artifact authentication requires huggingface_hub."
            ) from exc
        authenticated_checkpoints: list[dict[str, Any]] = []
        for checkpoint in checkpoints:
            snapshot = Path(
                snapshot_download(
                    repo_id=str(checkpoint["model_id"]),
                    revision=str(checkpoint["revision"]),
                    local_files_only=True,
                )
            ).expanduser()
            if not snapshot.is_dir() or snapshot.name != checkpoint["revision"]:
                raise RuntimeError(
                    "CogVideoX cached snapshot does not resolve to the exact revision."
                )
            files = checkpoint.get("files")
            if not isinstance(files, list) or not files:
                raise RuntimeError("CogVideoX artifact manifest has an empty file inventory.")
            declared_paths = {
                str(file_record.get("path", ""))
                for file_record in files
                if isinstance(file_record, Mapping)
            }
            if declared_paths != _EXPECTED_ARTIFACT_PATHS_BY_ROLE[checkpoint["role"]] or len(
                declared_paths
            ) != len(files):
                raise RuntimeError(
                    f"CogVideoX {checkpoint['role']} artifact inventory topology drifted."
                )
            authenticated_files: list[dict[str, Any]] = []
            seen: set[str] = set()
            for file_record in files:
                if not isinstance(file_record, Mapping):
                    raise RuntimeError("CogVideoX artifact file record is malformed.")
                relative = Path(str(file_record.get("path", "")))
                normalized = relative.as_posix()
                if (
                    not normalized
                    or relative.is_absolute()
                    or ".." in relative.parts
                    or normalized in seen
                ):
                    raise RuntimeError("CogVideoX artifact manifest path is unsafe or duplicated.")
                seen.add(normalized)
                candidate = snapshot / relative
                if not candidate.is_file():
                    raise RuntimeError(
                        f"CogVideoX cached artifact is missing: {checkpoint['role']}/{normalized}."
                    )
                stat = candidate.stat()
                if stat.st_size != file_record.get("size_bytes"):
                    raise RuntimeError(
                        f"CogVideoX cached artifact size drifted: {checkpoint['role']}/{normalized}."
                    )
                # Rehash every launch rather than trusting mutable cross-job
                # metadata.  This deliberately pays the full I/O cost to make
                # a same-size/inode timestamp race unable to reuse stale trust.
                actual_sha = _sha256_file(candidate)
                if actual_sha != file_record.get("sha256"):
                    raise RuntimeError(
                        f"CogVideoX cached artifact digest drifted: {checkpoint['role']}/{normalized}."
                    )
                authenticated_files.append(
                    {
                        "path": normalized,
                        "size_bytes": stat.st_size,
                        "sha256": actual_sha,
                    }
                )
            authenticated_checkpoints.append(
                {
                    "role": checkpoint["role"],
                    "model_id": checkpoint["model_id"],
                    "revision": checkpoint["revision"],
                    "snapshot_path": str(snapshot.resolve()),
                    "file_count": len(authenticated_files),
                    "file_inventory_sha256": hashlib.sha256(
                        json.dumps(
                            authenticated_files,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest(),
                }
            )
        return {
            "schema_version": 1,
            "path": str(path),
            "sha256": digest,
            "checkpoint_count": 2,
            "checkpoints": authenticated_checkpoints,
            "status": "all_manifest_files_authenticated_before_model_load",
        }


def _validate_video_batch(media: Any, expected_frames: int) -> list[list[Any]]:
    if not isinstance(media, list) or not media:
        raise RuntimeError("CogVideoX decode did not return a non-empty video batch.")
    if media and media[0] and not isinstance(media[0], list):
        media = [media]
    if any(not isinstance(video, list) or len(video) != expected_frames for video in media):
        raise RuntimeError(
            f"CogVideoX native segment must contain exactly {expected_frames} frames."
        )
    return media


def _continuation_seed(base_seed: int, segment_index: int) -> int:
    if segment_index not in {1, 2}:
        raise ValueError("CogVideoX continuation segment index must be 1 or 2.")
    payload = f"{COGVIDEOX_SEGMENT_SEED_DOMAIN}|{int(base_seed)}|{segment_index}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)
    return value or segment_index


def _make_generator(seed: int, device: torch.device) -> torch.Generator:
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    return torch.Generator(device=generator_device).manual_seed(seed)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().flatten()[0].item())
    return float(value)


def _validated_native_call_kwargs(
    pipeline: Any,
    kwargs: dict[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    import inspect

    signature = inspect.signature(pipeline.__call__)
    filtered = {key: value for key, value in kwargs.items() if key in signature.parameters}
    required = {
        "prompt",
        "negative_prompt",
        "num_frames",
        "num_inference_steps",
        "guidance_scale",
        "generator",
        "height",
        "width",
    }
    if role == "i2v":
        required.add("image")
    missing = sorted(required - set(filtered))
    if missing:
        raise RuntimeError(
            f"CogVideoX {role} call lost required kwargs after signature filtering: {missing}."
        )
    if filtered["num_frames"] != 49 or filtered["num_inference_steps"] != 50:
        raise RuntimeError("CogVideoX native call shape/step contract drifted.")
    return filtered
