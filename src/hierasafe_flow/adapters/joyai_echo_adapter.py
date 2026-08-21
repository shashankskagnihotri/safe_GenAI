from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from hierasafe_flow.adapters.base import (
    AdapterState,
    DenoisingStepContext,
    FrozenGeneratorAdapter,
    PromptCondition,
    SchedulerStepResult,
)
from hierasafe_flow.evaluation.temporal_qualification import (
    TEMPORAL_CRITERIA_BY_MODEL,
    validate_temporal_production_gate,
)
from hierasafe_flow.generation.conditioning_cache import (
    canonical_json_bytes,
    encoding_fingerprint,
)
from hierasafe_flow.generation.temporal_artifacts import (
    TemporalEvidenceBundle,
    TemporalSegmentEvidence,
)


JOYAI_ECHO_TEMPORAL_PROTOCOL_KEY = "joyai_echo_temporal_protocol"
JOYAI_ECHO_CHECKPOINT_REVISION = "4187f9a53c6eff3a76c51e79bd27f70d10f7591b"
JOYAI_ECHO_SEGMENT_SEED_DOMAIN = "joyai_echo_multishot_v1"
_SCHEMA2_STRATEGY = "reference_conditioned_multishot"

_EXPECTED_JOYAI_ECHO_TEMPORAL_PROTOCOL = {
    "schema_version": 2,
    "native_temporal_call_schema_version": 2,
    "segment_trace_schema_version": 1,
    "temporal_evidence_schema_version": 1,
    "segmented_temporal_audit_schema_version": 1,
    "strategy": _SCHEMA2_STRATEGY,
    "scientific_label": (
        "JoyAI-Echo reference-conditioned multishot (not hard I2V continuation)"
    ),
    "checkpoint_family": "JoyAI-Echo-release",
    "artifact_manifest": "configs/artifacts/joyai_echo_segmented_temporal_v2.json",
    "artifact_manifest_sha256": (
        "86982ab74eadf086fd8ad7b242f16b9b04500653e0d3e2520b02e1a1e031504b"
    ),
    "source_revision": "bdd3ec9ecad0bbbfc006cf5288709cb744c00b01",
    "source_tracked_diff_sha256": (
        "fa783e1f63be60d87014e5a0f8f4ef8cf020d27aa4583ab1cf7378d4c3129ae0"
    ),
    "native_video_fps": 24,
    "released_wrapper_position_fps": 24.0,
    "segments": [
        {
            "index": 0,
            "role": "base_t2av",
            "frames": 121,
            "fps": 24,
            "denoising_steps": 8,
            "memory": False,
        },
        {
            "index": 1,
            "role": "memory_t2av",
            "frames": 121,
            "fps": 24,
            "denoising_steps": 8,
            "memory": True,
        },
        {
            "index": 2,
            "role": "memory_t2av",
            "frames": 121,
            "fps": 24,
            "denoising_steps": 8,
            "memory": True,
        },
    ],
    "segment_video_latent_shape": [1, 16, 128, 23, 40],
    "segment_audio_latent_length": 126,
    "denoising_sigmas": [
        1.0,
        0.99375,
        0.9875,
        0.98125,
        0.975,
        0.909375,
        0.725,
        0.421875,
        0.0,
    ],
    "seed_domain": JOYAI_ECHO_SEGMENT_SEED_DOMAIN,
    "memory": {
        "video_tail_frames": 9,
        "audio_tail_latents": 96,
        "encoded_video_latents_per_entry": 1,
        "position_mode": "reference",
        "paired_audio_memory": True,
        "v2a_grad_scale": 1.0,
        "branch_byte_identity_required": True,
        "base_branch_audio_state_only": True,
    },
    "stitch": {
        "retained": [
            "segment_0[0:120]",
            "segment_1[0:120]",
            "segment_2[0:120]",
        ],
        "native_frame_count": 360,
        "overlap_semantics": "none",
    },
    "resampling": {
        "method": "nearest_timestamp_decimation_round_half_up",
        "source_frames": 360,
        "source_fps": 24,
        "output_frames": 240,
        "output_fps": 16,
        "unique_source_indices": True,
        "synthesized_frames": 0,
        "duplicated_frames": 0,
    },
    "output_frames": 240,
    "output_fps": 16,
    "duration_seconds": 15.0,
    "height": 736,
    "width": 1280,
}
_JOYAI_ECHO_TEMPORAL_DYNAMIC_FIELDS = {"execution_phase", "production_gate"}
_JOYAI_ECHO_OBSOLETE_PROTOCOL_FIELDS = {
    "effective_position_fps",
    "internal_generated_frames",
    "released_demo_fps",
    "released_demo_frames",
    "terminal_endpoint_crop",
    "frame_resampling",
}
_JOYAI_ECHO_PRODUCTION_CRITERIA = TEMPORAL_CRITERIA_BY_MODEL["joyai_echo"]


class JoyAIEchoAdapter(FrozenGeneratorAdapter):
    adapter_name = "joyai_echo"
    task_type = "text_to_video"
    pipeline_class_name = "InferenceEngine"
    latent_feature_dim = 2  # [B, F, C, H, W]

    default_sigmas = (
        1.0,
        0.99375,
        0.9875,
        0.98125,
        0.975,
        0.909375,
        0.725,
        0.421875,
        0.0,
    )

    checkpoint_filename = "JoyAI-Echo-release.safetensors"
    gemma_model_id = "google/gemma-3-12b-it"

    def __init__(
        self,
        model_id: str,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model_id=model_id, device=device, dtype=dtype, config=config)
        self.source_root: Path | None = None
        self.checkpoint_path: Path | None = None
        self.gemma_path: Path | None = None
        self.generator: Any | None = None
        self.video_vae: Any | None = None
        self.audio_vae: Any | None = None
        self.text_encoder: Any | None = None
        self.add_noise: Any | None = None
        self.compute_latent_shapes: Any | None = None
        self.decode_benchmark_sample: Any | None = None
        self.video_uint8_to_pil_frames: Any | None = None
        self.memory_bank_class: Any | None = None
        self.encode_memory_frames_batch: Any | None = None
        self._ablation_metadata: dict[str, Any] | None = None
        self._source_provenance: dict[str, Any] | None = None
        self._checkpoint_sha256: str | None = None
        self._released_wrapper_video_fps: float | None = None
        self._last_temporal_provenance: dict[str, Any] | None = None
        self._artifact_authentication: dict[str, Any] | None = None
        self._memory_bank: Any | None = None
        self._completed_segments: list[list[Any]] = []
        self._segment_seeds: list[int] = []
        self._segment_audio_hashes: list[str] = []
        self._segment_conditioning_records: dict[int, list[dict[str, Any]]] = {}
        self._segment_protected_records: dict[int, list[dict[str, Any]]] = {}
        self._memory_transition_records: dict[int, dict[str, Any]] = {}
        self._temporal_evidence: TemporalEvidenceBundle | None = None

    def load(self) -> None:
        # Authenticate the complete route before allocating model weights.  A
        # branch/tag, partial source inventory, or altered external checkout is
        # not an acceptable experiment input.
        self._require_commit_revision("revision")
        self._require_commit_revision("gemma_revision")
        self._require_commit_revision("source_revision")
        protocol = self._temporal_protocol()
        if protocol is not None:
            self._validate_temporal_execution_gate(protocol)
        self._ablation_metadata = self._local_ablation_metadata()
        if self._ablation_metadata is not None:
            self.model_id = str(self._ablation_metadata["base_model_id"])
        self.source_root = self._resolve_source_root()
        self._source_provenance = self._validate_source_checkout(self.source_root)
        if protocol is not None:
            self._artifact_authentication = self._authenticate_artifact_manifest(
                protocol,
                self.source_root,
            )
        self._add_source_tree(self.source_root)
        self.checkpoint_path = self._resolve_checkpoint_path()
        self._checkpoint_sha256 = self._validate_checkpoint(self.checkpoint_path)
        if protocol is not None and self._artifact_authentication is not None:
            expected_checkpoint = self._artifact_authentication["checkpoint"]["sha256"]
            if self._checkpoint_sha256 != expected_checkpoint:
                raise RuntimeError(
                    "JoyAI-Echo loaded checkpoint differs from the authenticated route manifest."
                )
        self.gemma_path = self._resolve_gemma_path()

        from ltx_distillation.models.ltx_wrapper import create_ltx2_wrapper
        from ltx_distillation.models.text_encoder_wrapper import create_text_encoder_wrapper
        from ltx_distillation.models.vae_wrapper import create_vae_wrappers
        from ltx_distillation.inference.memory_multishot import (
            PairedAudioVideoMemoryBank,
            video_uint8_to_pil_frames,
        )
        from ltx_distillation.utils import (
            add_noise,
            compute_latent_shapes,
            decode_benchmark_sample,
            encode_memory_frames_batch,
        )

        video_height = int(
            protocol["height"]
            if protocol is not None
            else self.config.get("height", self.config.get("video_height", 736))
        )
        video_width = int(
            protocol["width"]
            if protocol is not None
            else self.config.get("width", self.config.get("video_width", 1280))
        )

        self.text_encoder = create_text_encoder_wrapper(
            checkpoint_path=str(self.checkpoint_path),
            gemma_path=str(self.gemma_path),
            device=self.device,
            dtype=self.dtype,
        )
        if self._ablation_metadata is not None:
            artifact_dir = Path(str(self._ablation_metadata["_artifact_dir"]))
            state_path = artifact_dir / str(
                self._ablation_metadata.get(
                    "text_encoder_state_file", "joyai_echo_text_encoder_state.pt"
                )
            )
            self.text_encoder.load_state_dict(
                torch.load(state_path, map_location=self.device), strict=True
            )
        self.text_encoder.eval()

        self.generator = create_ltx2_wrapper(
            checkpoint_path=str(self.checkpoint_path),
            gemma_path=str(self.gemma_path),
            device=self.device,
            dtype=self.dtype,
            video_height=video_height,
            video_width=video_width,
        )
        self.generator.eval()
        self._released_wrapper_video_fps = _finite_positive_float(
            getattr(self.generator, "VIDEO_FPS", None),
            "released JoyAI-Echo wrapper VIDEO_FPS",
        )
        if protocol is not None:
            expected_wrapper_fps = float(protocol["released_wrapper_position_fps"])
            if self._released_wrapper_video_fps != expected_wrapper_fps:
                raise RuntimeError(
                    "Pinned JoyAI-Echo wrapper temporal contract changed: expected "
                    f"VIDEO_FPS={expected_wrapper_fps}, got {self._released_wrapper_video_fps}."
                )

        self.video_vae, self.audio_vae = create_vae_wrappers(
            checkpoint_path=str(self.checkpoint_path),
            device=torch.device("cpu"),
            dtype=self.dtype,
            with_video_encoder=True,
            with_audio_encoder=True,
            decoder_device=torch.device("cpu"),
        )
        self.video_vae.eval()
        self.audio_vae.eval()

        self.add_noise = add_noise
        self.compute_latent_shapes = compute_latent_shapes
        self.decode_benchmark_sample = decode_benchmark_sample
        self.video_uint8_to_pil_frames = video_uint8_to_pil_frames
        self.memory_bank_class = PairedAudioVideoMemoryBank
        self.encode_memory_frames_batch = encode_memory_frames_batch
        self._freeze_loaded_modules()
        self.loaded = True

    def prepare_prompt(self, prompt: str) -> PromptCondition:
        self._require_loaded()
        if self.text_encoder is None:
            raise RuntimeError("JoyAI-Echo text encoder was not loaded.")
        with torch.inference_mode():
            condition = self.text_encoder([prompt])
        return PromptCondition(prompt=prompt, data=dict(condition))

    def conditioning_cache_identity(
        self,
        prompt: str,
        state: AdapterState,
        *,
        prompt_view: str,
        call_role: str,
    ) -> dict[str, Any]:
        """Namespace only authenticated text identity, never dynamic memory/RNG state."""

        identity = super().conditioning_cache_identity(
            prompt,
            state,
            prompt_view=prompt_view,
            call_role=call_role,
        )
        identity.update(
            {
                "checkpoint_sha256": self._checkpoint_sha256
                or self.config.get("checkpoint_sha256"),
                "gemma_model_id": self.gemma_model_id,
                "gemma_revision": self.config.get("gemma_revision"),
                "source_revision": self.config.get("source_revision"),
                "source_tracked_diff_sha256": self.config.get(
                    "source_tracked_diff_sha256"
                ),
            }
        )
        # Memory, sigmas, noisy AV state, and RNG are deliberately absent.
        canonical_json_bytes(identity)
        return identity

    def prepare_initial_latents(
        self,
        prompt: str,
        batch_size: int,
        generator: torch.Generator | None,
        **generation_kwargs: Any,
    ) -> tuple[torch.Tensor, AdapterState]:
        self._require_loaded()
        if self.compute_latent_shapes is None:
            raise RuntimeError("JoyAI-Echo latent shape helper was not loaded.")
        protocol = self._temporal_protocol()
        if protocol is None:
            requested_num_frames = int(generation_kwargs.get("num_frames", 121))
            height = int(generation_kwargs.get("height", 736))
            width = int(generation_kwargs.get("width", 1280))
            fps = _finite_positive_float(
                generation_kwargs.get("fps", generation_kwargs.get("video_fps", 24)),
                "JoyAI-Echo requested fps",
            )
            if requested_num_frames <= 0 or (requested_num_frames - 1) % 8:
                raise ValueError(
                    "JoyAI-Echo direct frame counts must satisfy 1+8*k; got "
                    f"{requested_num_frames}."
                )
            num_frames = requested_num_frames
            released_fps = self._require_released_wrapper_video_fps()
            if fps != released_fps:
                raise ValueError(
                    "The released JoyAI-Echo wrapper hard-codes video RoPE at 24 fps. "
                    f"Refusing an unrecorded {fps}-fps direct request."
                )
            _validate_optional_duration(
                generation_kwargs,
                output_frames=requested_num_frames,
                output_fps=fps,
                adapter_name="JoyAI-Echo",
            )
            temporal_provenance = _direct_temporal_provenance(
                adapter_name="JoyAI-Echo",
                internal_frames=num_frames,
                output_frames=requested_num_frames,
                fps=fps,
            )
            segment_count = 1
        else:
            self._validate_temporal_execution_gate(protocol)
            self._validate_requested_temporal_output(generation_kwargs, protocol)
            self._assert_source_and_route_unchanged(protocol)
            if batch_size != 1:
                raise RuntimeError(
                    "JoyAI-Echo schema-2 multishot route is qualified only at batch size 1."
                )
            requested_num_frames = int(protocol["output_frames"])
            num_frames = 121
            height = int(protocol["height"])
            width = int(protocol["width"])
            fps = int(protocol["native_video_fps"])
            temporal_provenance = self._configured_temporal_provenance(protocol)
            segment_count = 3
        video_shape, audio_shape = self.compute_latent_shapes(
            num_frames=num_frames,
            video_height=height,
            video_width=width,
            batch_size=batch_size,
            video_fps=fps,
        )
        expected_video_shape = (
            tuple(int(value) for value in protocol["segment_video_latent_shape"])
            if protocol is not None
            else tuple(video_shape)
        )
        if tuple(video_shape) != expected_video_shape:
            raise RuntimeError(
                "JoyAI-Echo schema-2 video latent shape changed: expected "
                f"{expected_video_shape}, got {tuple(video_shape)}."
            )
        internal_duration_seconds = num_frames / fps
        audio_latents_per_second = 25.0
        expected_audio_frames = round(internal_duration_seconds * audio_latents_per_second)
        if protocol is not None and expected_audio_frames != int(
            protocol["segment_audio_latent_length"]
        ):
            raise RuntimeError("JoyAI-Echo frozen audio latent arithmetic changed.")
        if int(audio_shape[1]) != expected_audio_frames:
            raise RuntimeError(
                "JoyAI-Echo audio latent arithmetic changed: expected "
                f"{expected_audio_frames}, got {audio_shape[1]}."
            )

        base_seed = int(generator.initial_seed()) if generator is not None else 0
        segment_seeds = (
            [base_seed]
            if protocol is None
            else [
                base_seed,
                _derive_segment_seed(base_seed, 1),
                _derive_segment_seed(base_seed, 2),
            ]
        )
        segment_generator = _make_generator(segment_seeds[0], self.device)
        latents = torch.randn(
            tuple(video_shape),
            generator=segment_generator,
            device=self.device,
            dtype=self.dtype,
        )
        audio_latents = torch.randn(
            tuple(audio_shape),
            generator=segment_generator,
            device=self.device,
            dtype=self.dtype,
        )
        state = AdapterState(
            extra={
                "base_prompt": prompt,
                "base_seed": base_seed,
                "segment_seed": segment_seeds[0],
                "segment_seeds": segment_seeds,
                "segment_generator": segment_generator,
                "segment_index": 0,
                "segment_count": segment_count,
                "segment_step_index": 0,
                "condition_epoch": 0,
                "model_role": "base_t2av" if protocol is not None else "primary",
                "model_id": self.model_id,
                "model_revision": self._require_commit_revision("revision"),
                "anchor_sha256": None,
                "audio_latents": audio_latents,
                "video_shape": tuple(video_shape),
                "audio_shape": tuple(audio_shape),
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "output_num_frames": requested_num_frames,
                "fps": int(protocol["output_fps"]) if protocol is not None else fps,
                "native_fps": fps,
                "duration_seconds": (
                    float(protocol["duration_seconds"])
                    if protocol is not None
                    else requested_num_frames / fps
                ),
                "internal_duration_seconds": internal_duration_seconds,
                "audio_latents_per_second": audio_latents_per_second,
                "audio_num_frames": expected_audio_frames,
                "temporal_provenance": temporal_provenance,
                "protected_state_trace": {
                    "schema_version": 1,
                    "segment_index": 0,
                    "memory_present": False,
                    "video_noise_fingerprint": encoding_fingerprint(latents),
                    "audio_noise_fingerprint": encoding_fingerprint(audio_latents),
                    "audio_state_source": "base_current_branch_only",
                },
            }
        )
        if protocol is not None:
            if self.memory_bank_class is None:
                raise RuntimeError("JoyAI-Echo paired memory helper was not loaded.")
            self._memory_bank = self.memory_bank_class(
                max_size=1,
                save_mode="latest_tail_clip_only",
                num_fix_frames=0,
            )
            self._completed_segments = []
            self._segment_seeds = segment_seeds
            self._segment_audio_hashes = []
            self._segment_conditioning_records = {0: [], 1: [], 2: []}
            self._segment_protected_records = {0: [], 1: [], 2: []}
            self._memory_transition_records = {}
            self._temporal_evidence = None
        self._last_temporal_provenance = temporal_provenance
        return latents, state

    def set_timesteps(
        self,
        num_inference_steps: int,
        latents: torch.Tensor | None = None,
        state: AdapterState | None = None,
    ) -> list[Any]:
        self._require_loaded()
        del latents
        sigmas = self._sigmas()
        expected_steps = len(sigmas) - 1
        if num_inference_steps != expected_steps:
            raise ValueError(
                "JoyAI-Echo uses the released DMD sigma schedule with "
                f"{expected_steps} denoising steps; got {num_inference_steps}."
            )
        if state is not None:
            state.extra["next_sigmas"] = sigmas[1:]
            state.extra["segment_step_index"] = 0
            state.extra["local_num_steps"] = expected_steps
        native_timesteps = list(sigmas[:-1])
        if self._temporal_protocol() is not None:
            self.timesteps = [
                sigma.clone() if isinstance(sigma, torch.Tensor) else sigma
                for _segment_index in range(3)
                for sigma in native_timesteps
            ]
            if state is not None:
                state.extra["global_num_steps"] = 24
        else:
            self.timesteps = native_timesteps
        return self.timesteps

    def denoising_step_context(
        self,
        global_step_index: int,
        global_num_steps: int,
        state: AdapterState,
    ) -> DenoisingStepContext:
        if self._temporal_protocol() is None:
            return super().denoising_step_context(global_step_index, global_num_steps, state)
        if global_num_steps != 24:
            raise RuntimeError("JoyAI-Echo schema-2 route requires exactly 24 global steps.")
        return DenoisingStepContext(
            global_step_index=global_step_index,
            global_num_steps=global_num_steps,
            segment_index=int(state.extra["segment_index"]),
            segment_count=3,
            local_step_index=int(state.extra["segment_step_index"]),
            local_num_steps=8,
            model_role=str(state.extra["model_role"]),
            model_id=self.model_id,
            model_revision=self._require_commit_revision("revision"),
            condition_epoch=int(state.extra["condition_epoch"]),
            anchor_sha256=None,
            segment_seed=int(state.extra["segment_seed"]),
        )

    def predict_vector_field(
        self,
        latents: torch.Tensor,
        timestep: Any,
        condition: PromptCondition,
        state: AdapterState,
    ) -> torch.Tensor:
        self._require_loaded()
        if self.generator is None:
            raise RuntimeError("JoyAI-Echo generator was not loaded.")
        wrapper_fps = _finite_positive_float(
            getattr(self.generator, "VIDEO_FPS", None),
            "JoyAI-Echo wrapper VIDEO_FPS",
        )
        if wrapper_fps != 24.0:
            raise RuntimeError(
                "JoyAI-Echo schema-2 route must preserve released VIDEO_FPS=24; "
                f"got {wrapper_fps}."
            )
        audio_latents = state.extra.get("audio_latents")
        if audio_latents is None:
            raise RuntimeError("JoyAI-Echo adapter state is missing audio_latents.")
        cond = {
            key: (value.to(self.device) if isinstance(value, torch.Tensor) else value)
            for key, value in condition.data.items()
        }
        sigma = self._sigma_value(timestep)
        video_sigma = torch.full(
            (latents.shape[0], latents.shape[1]),
            sigma,
            device=self.device,
            dtype=torch.float32,
        )
        audio_sigma = torch.full(
            (audio_latents.shape[0], audio_latents.shape[1]),
            sigma,
            device=self.device,
            dtype=torch.float32,
        )
        memory_kwargs = self._memory_call_kwargs(state)
        protected_inputs_before = {
            "video_state_fingerprint": encoding_fingerprint(latents),
            "audio_state_fingerprint": encoding_fingerprint(audio_latents),
            "video_sigma_fingerprint": encoding_fingerprint(video_sigma),
            "audio_sigma_fingerprint": encoding_fingerprint(audio_sigma),
            "memory": _memory_fingerprint_record(memory_kwargs),
        }
        with torch.inference_mode():
            pred_video, pred_audio = self.generator(
                noisy_image_or_video=latents,
                conditional_dict=cond,
                timestep=video_sigma,
                noisy_audio=audio_latents,
                audio_timestep=audio_sigma,
                **memory_kwargs,
            )
        pred_video = pred_video.to(dtype=latents.dtype)
        pred_audio = pred_audio.to(dtype=audio_latents.dtype)
        if pred_video.shape != latents.shape or pred_audio.shape != audio_latents.shape:
            raise RuntimeError("JoyAI-Echo generator changed the released AV prediction shapes.")
        if not torch.isfinite(pred_video).all() or not torch.isfinite(pred_audio).all():
            raise RuntimeError("JoyAI-Echo generator returned a non-finite AV prediction.")
        protected_inputs_after = {
            "video_state_fingerprint": encoding_fingerprint(latents),
            "audio_state_fingerprint": encoding_fingerprint(audio_latents),
            "video_sigma_fingerprint": encoding_fingerprint(video_sigma),
            "audio_sigma_fingerprint": encoding_fingerprint(audio_sigma),
            "memory": _memory_fingerprint_record(memory_kwargs),
        }
        if protected_inputs_after != protected_inputs_before:
            raise RuntimeError(
                "JoyAI-Echo generator mutated noisy AV/sigma/memory branch inputs in place."
            )

        is_base_current = condition.prompt == state.extra.get("base_prompt")
        if is_base_current:
            state.extra["base_audio_prediction"] = pred_audio
            state.extra["base_audio_prediction_fingerprint"] = encoding_fingerprint(
                pred_audio
            )
        segment_index = int(state.extra.get("segment_index", 0))
        record = {
            "schema_version": 1,
            "segment_index": segment_index,
            "local_step_index": int(state.extra.get("segment_step_index", 0)),
            "prompt_sha256": hashlib.sha256(condition.prompt.encode("utf-8")).hexdigest(),
            "conditioning_fingerprint": encoding_fingerprint(condition.data),
            **protected_inputs_before,
            "base_current_audio_candidate": is_base_current,
            "video_prediction_fingerprint": encoding_fingerprint(pred_video),
            "audio_prediction_fingerprint": encoding_fingerprint(pred_audio),
        }
        if self._temporal_protocol() is not None:
            self._segment_conditioning_records[segment_index].append(record)
        protected = {
            "schema_version": 1,
            "segment_index": segment_index,
            "segment_seed": int(state.extra.get("segment_seed", 0)),
            "memory": record["memory"],
            "noisy_video_fingerprint": record["video_state_fingerprint"],
            "noisy_audio_fingerprint": record["audio_state_fingerprint"],
            "video_sigma_fingerprint": record["video_sigma_fingerprint"],
            "audio_sigma_fingerprint": record["audio_sigma_fingerprint"],
            "steering_target": "video_vector_field_only",
            "audio_state_source": "base_current_branch_only",
            "base_audio_prediction_fingerprint": state.extra.get(
                "base_audio_prediction_fingerprint"
            ),
        }
        state.extra["protected_state_trace"] = protected
        if self._temporal_protocol() is not None:
            self._segment_protected_records[segment_index].append(deepcopy(protected))
        return pred_video

    def scheduler_step(
        self,
        model_prediction: torch.Tensor,
        timestep: Any,
        latents: torch.Tensor,
        state: AdapterState,
        generator: torch.Generator | None = None,
    ) -> SchedulerStepResult:
        self._require_loaded()
        del timestep
        if self.add_noise is None:
            raise RuntimeError("JoyAI-Echo add_noise helper was not loaded.")
        audio_prediction = state.extra.pop("base_audio_prediction", None)
        if audio_prediction is None:
            raise RuntimeError(
                "JoyAI-Echo scheduler_step requires the current/base audio prediction; "
                "safe/unsafe audio may never advance state."
            )
        selected_audio_fingerprint = encoding_fingerprint(audio_prediction)
        if selected_audio_fingerprint != state.extra.pop(
            "base_audio_prediction_fingerprint", None
        ):
            raise RuntimeError("JoyAI-Echo current/base audio prediction binding changed.")
        audio_latents = state.extra["audio_latents"]
        step_index = int(state.extra.get("segment_step_index", 0))
        next_sigmas = state.extra.get("next_sigmas")
        if next_sigmas is None or step_index >= len(next_sigmas):
            raise RuntimeError("JoyAI-Echo adapter state is missing the next sigma schedule.")
        next_sigma = self._sigma_value(next_sigmas[step_index])
        segment_generator = state.extra.get("segment_generator", generator)
        if next_sigma > 0:
            fresh_video = torch.randn(
                latents.shape,
                generator=segment_generator,
                device=latents.device,
                dtype=latents.dtype,
            )
            fresh_audio = torch.randn(
                audio_latents.shape,
                generator=segment_generator,
                device=audio_latents.device,
                dtype=audio_latents.dtype,
            )
            next_video_sigma = torch.full(
                (latents.shape[0], latents.shape[1]),
                next_sigma,
                device=latents.device,
                dtype=torch.float32,
            )
            next_audio_sigma = torch.full(
                (audio_latents.shape[0], audio_latents.shape[1]),
                next_sigma,
                device=audio_latents.device,
                dtype=torch.float32,
            )
            next_latents = self.add_noise(
                model_prediction.flatten(0, 1),
                fresh_video.flatten(0, 1),
                next_video_sigma.flatten(0, 1),
            ).unflatten(0, (latents.shape[0], latents.shape[1]))
            next_audio = self.add_noise(audio_prediction, fresh_audio, next_audio_sigma)
        else:
            next_latents = model_prediction
            next_audio = audio_prediction
        state.extra["audio_latents"] = next_audio
        state.extra["segment_step_index"] = step_index + 1
        state.extra["protected_state_trace"] = {
            **dict(state.extra.get("protected_state_trace") or {}),
            "selected_audio_prediction_fingerprint": selected_audio_fingerprint,
            "selected_audio_branch": "base_current",
            "steered_video_prediction_fingerprint": encoding_fingerprint(
                model_prediction
            ),
        }
        protocol = self._temporal_protocol()
        if protocol is None or step_index + 1 < 8:
            return SchedulerStepResult(latents=next_latents, state=state)
        if step_index + 1 > 8:
            raise RuntimeError("JoyAI-Echo segment advanced beyond the released 8 sigmas.")
        segment_index = int(state.extra["segment_index"])
        if segment_index == 2:
            return SchedulerStepResult(latents=next_latents, state=state)
        self._assert_source_and_route_unchanged(protocol)
        frames, audio_hash = self._decode_native_segment(next_latents, next_audio)
        self._append_segment(frames, audio_hash)
        memory_record = self._build_paired_tail_memory(
            frames,
            next_audio,
            source_segment_index=segment_index,
            state=state,
        )
        next_latents, next_audio = self._start_memory_segment(
            segment_index=segment_index + 1,
            state=state,
            memory_record=memory_record,
        )
        state.extra["audio_latents"] = next_audio
        return SchedulerStepResult(latents=next_latents, state=state)

    def decode_latents(self, latents: torch.Tensor, state: AdapterState) -> Any:
        self._require_loaded()
        frames, audio_hash = self._decode_native_segment(
            latents,
            state.extra["audio_latents"],
        )
        protocol = self._temporal_protocol()
        if protocol is None:
            return [frames]
        if int(state.extra.get("segment_index", -1)) != 2 or int(
            state.extra.get("segment_step_index", -1)
        ) != 8:
            raise RuntimeError("JoyAI-Echo decode requested before all three shots completed.")
        self._append_segment(frames, audio_hash)
        self._assert_source_and_route_unchanged(protocol)
        return self._stitch_and_resample_segments(
            self._completed_segments,
            state=state,
            generation_path="adapter_vector_field_runner",
        )

    def take_temporal_evidence(
        self,
        state: AdapterState,
    ) -> TemporalEvidenceBundle | None:
        del state
        evidence = self._temporal_evidence
        self._temporal_evidence = None
        return evidence

    def configure_native_pipeline_for_temporal_protocol(self) -> dict[str, Any]:
        """Reject a false native-negative claim without loading any model."""

        raise RuntimeError(
            "JoyAI-Echo native negative prompting is unsupported: the released base and "
            "memory pipelines expose no negative-prompt/CFG surface. No emulation or prompt "
            "rewrite is permitted."
        )

    def complete_native_pipeline_temporal_protocol(self, *_: Any, **__: Any) -> Any:
        raise RuntimeError(
            "JoyAI-Echo has no native-negative completion route; use the authenticated "
            "manual reference-conditioned multishot adapter for baseline/steering."
        )

    def _temporal_protocol(self) -> dict[str, Any] | None:
        raw = self.config.get(JOYAI_ECHO_TEMPORAL_PROTOCOL_KEY)
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError(f"model.{JOYAI_ECHO_TEMPORAL_PROTOCOL_KEY} must be a mapping.")
        protocol = dict(raw)
        obsolete = sorted(_JOYAI_ECHO_OBSOLETE_PROTOCOL_FIELDS & set(protocol))
        if obsolete:
            raise ValueError(
                "JoyAI-Echo schema-2 protocol contains obsolete single-trajectory fields: "
                f"{obsolete}."
            )
        required = set(_EXPECTED_JOYAI_ECHO_TEMPORAL_PROTOCOL) | {"execution_phase"}
        missing = sorted(required - set(protocol))
        if missing:
            raise ValueError(
                f"{JOYAI_ECHO_TEMPORAL_PROTOCOL_KEY} is missing frozen fields: {missing}."
            )
        unknown = sorted(
            set(protocol)
            - set(_EXPECTED_JOYAI_ECHO_TEMPORAL_PROTOCOL)
            - _JOYAI_ECHO_TEMPORAL_DYNAMIC_FIELDS
        )
        if unknown:
            raise ValueError(f"{JOYAI_ECHO_TEMPORAL_PROTOCOL_KEY} has unknown fields: {unknown}.")
        mismatches = {
            key: {"expected": expected, "actual": protocol[key]}
            for key, expected in _EXPECTED_JOYAI_ECHO_TEMPORAL_PROTOCOL.items()
            if protocol[key] != expected
        }
        if mismatches:
            raise ValueError(
                "JoyAI-Echo temporal protocol differs from the frozen schema-2 multishot "
                f"contract: {mismatches}."
            )
        if protocol["execution_phase"] not in {"pilot", "production"}:
            raise ValueError("JoyAI-Echo temporal execution_phase must be 'pilot' or 'production'.")
        return protocol

    def _validate_temporal_execution_gate(self, protocol: Mapping[str, Any]) -> None:
        _validate_production_gate(
            protocol,
            adapter_name="JoyAI-Echo",
            criteria_names=_JOYAI_ECHO_PRODUCTION_CRITERIA,
        )

    def _validate_requested_temporal_output(
        self,
        generation_kwargs: Mapping[str, Any],
        protocol: Mapping[str, Any],
    ) -> None:
        actual = {
            "num_frames": int(generation_kwargs.get("num_frames", -1)),
            "fps": int(generation_kwargs.get("fps", -1)),
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
                "JoyAI-Echo schema-2 route requires the exact output/resolution "
                f"contract; expected {expected}, got {actual}."
            )
        requested_stitch = generation_kwargs.get("stitch_mode")
        if requested_stitch in {"361_overlap", "hard_anchor", "i2v_continuation"}:
            raise ValueError(
                "JoyAI-Echo 361_overlap/hard-anchor continuation is scientifically invalid: "
                "the released memory API provides reference attention, not a fixed anchor."
            )
        _validate_optional_duration(
            generation_kwargs,
            output_frames=int(protocol["output_frames"]),
            output_fps=float(protocol["output_fps"]),
            adapter_name="JoyAI-Echo",
        )

    def _require_released_wrapper_video_fps(self) -> float:
        if self._released_wrapper_video_fps is None:
            if self.generator is None:
                raise RuntimeError("JoyAI-Echo generator was not loaded.")
            self._released_wrapper_video_fps = _finite_positive_float(
                getattr(self.generator, "VIDEO_FPS", None),
                "released JoyAI-Echo wrapper VIDEO_FPS",
            )
        return self._released_wrapper_video_fps

    def _configured_temporal_provenance(
        self,
        protocol: Mapping[str, Any],
    ) -> dict[str, Any]:
        output_frames = int(protocol["output_frames"])
        native_frames = 121
        native_fps = int(protocol["native_video_fps"])
        audio_latent_fps = 25.0
        audio_frames = int(protocol["segment_audio_latent_length"])
        provenance = {
            "schema_version": 2,
            "status": "configured_not_yet_decoded",
            "execution_phase": protocol["execution_phase"],
            "strategy": protocol["strategy"],
            "scientific_classification": "reference_conditioned_multishot",
            "hard_anchor_guarantee": False,
            "native_clock": {
                "wrapper_video_fps": float(protocol["released_wrapper_position_fps"]),
                "preserved_without_mutation": True,
                "segment_frames": native_frames,
                "segment_count": 3,
                "denoising_steps_per_segment": 8,
            },
            "segment_trajectory": {
                "frames": native_frames,
                "fps": native_fps,
                "duration_seconds_inclusive_endpoint": native_frames / native_fps,
                "video_latent_shape": list(protocol["segment_video_latent_shape"]),
                "audio_latent_frames": audio_frames,
                "audio_latents_per_second": audio_latent_fps,
                "audio_duration_seconds_on_latent_grid": audio_frames / audio_latent_fps,
                "audio_video_clock_error_seconds": abs(
                    audio_frames / audio_latent_fps - native_frames / native_fps
                ),
                "audio_video_clock_error_within_one_audio_latent_interval": (
                    abs(audio_frames / audio_latent_fps - native_frames / native_fps)
                    <= 1.0 / audio_latent_fps
                ),
            },
            "postprocessing": {
                "classification": "honest_half_open_multishot_decimation",
                "endpoint_crop_per_segment": "[0:120]",
                "dropped_frame_count": 3,
                "stitched_native_frames": 360,
                "time_domain": "half_open_[0,duration_seconds)",
                "frame_resampling": protocol["resampling"]["method"],
                "output_frames": output_frames,
                "output_fps": int(protocol["output_fps"]),
                "duration_seconds": float(protocol["duration_seconds"]),
                "duplicated_frames": 0,
                "synthesized_frames": 0,
            },
            "memory": deepcopy(protocol["memory"]),
            "audio_output": {
                "joint_audio_latents_condition_video": True,
                "decoded_for_every_native_segment": True,
                "state_advanced_from_base_current_branch_only": True,
                "saved_to_mp4": False,
            },
            "artifact_authentication": self._artifact_authentication,
            "production_gate": protocol.get("production_gate"),
        }
        return provenance

    def _memory_call_kwargs(self, state: AdapterState) -> dict[str, Any]:
        segment_index = int(state.extra.get("segment_index", 0))
        if self._temporal_protocol() is None or segment_index == 0:
            if any(
                state.extra.get(key) is not None
                for key in (
                    "memory_video",
                    "memory_audio",
                    "memory_audio_timestep",
                    "memory_audio_segment_lengths",
                )
            ):
                raise RuntimeError("JoyAI-Echo segment 0 must not receive memory state.")
            return {}
        required = {
            "memory_video": state.extra.get("memory_video"),
            "memory_audio": state.extra.get("memory_audio"),
            "memory_audio_timestep": state.extra.get("memory_audio_timestep"),
            "memory_audio_segment_lengths": state.extra.get(
                "memory_audio_segment_lengths"
            ),
        }
        if not all(value is not None for value in required.values()):
            raise RuntimeError("JoyAI-Echo memory shot lacks paired AV memory arguments.")
        memory_video = required["memory_video"]
        memory_audio = required["memory_audio"]
        memory_timestep = required["memory_audio_timestep"]
        lengths = required["memory_audio_segment_lengths"]
        if not isinstance(memory_video, torch.Tensor) or tuple(memory_video.shape[:2]) != (
            1,
            1,
        ):
            raise RuntimeError("JoyAI-Echo must pass exactly one encoded video memory latent.")
        if not isinstance(memory_audio, torch.Tensor) or int(memory_audio.shape[1]) != 96:
            raise RuntimeError("JoyAI-Echo paired audio memory must contain exactly 96 latents.")
        if not isinstance(memory_timestep, torch.Tensor) or memory_timestep.shape != (
            memory_audio.shape[0],
            memory_audio.shape[1],
        ):
            raise RuntimeError("JoyAI-Echo paired audio memory timestep shape drifted.")
        if lengths != ((96,),):
            raise RuntimeError("JoyAI-Echo memory segment lengths must be exactly ((96,),).")
        kwargs = {
            **required,
            "paired_audio_memory": True,
            "v2a_grad_scale": 1.0,
            "memory_position_mode": "reference",
            "memory_downscale_factor": 1,
        }
        expected = state.extra.get("memory_argument_fingerprints")
        observed = _memory_fingerprint_record(kwargs)
        if expected != observed:
            raise RuntimeError(
                "JoyAI-Echo branch memory arguments changed after the segment transition."
            )
        # Return the original tensor objects. Every base/neutral/unsafe/safe
        # branch therefore receives byte-identical dynamic memory state.
        return kwargs

    def _decode_native_segment(
        self,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
    ) -> tuple[list[Any], str]:
        if self.decode_benchmark_sample is None or self.video_uint8_to_pil_frames is None:
            raise RuntimeError("JoyAI-Echo decode helpers were not loaded.")
        if self.generator is None or self.video_vae is None or self.audio_vae is None:
            raise RuntimeError("JoyAI-Echo native segment decoder is incomplete.")
        _move_module(self.generator, "cpu")
        _move_module(self.text_encoder, "cpu")
        _empty_cuda_cache(self.device)
        _move_module(getattr(self.video_vae, "decoder", None), self.device)
        _move_module(getattr(self.audio_vae, "decoder", None), self.device)
        _move_module(getattr(self.audio_vae, "vocoder", None), self.device)
        with torch.inference_mode():
            video_uint8, audio_waveform = self.decode_benchmark_sample(
                self.video_vae,
                self.audio_vae,
                video_latent,
                audio_latent,
            )
        frames = list(self.video_uint8_to_pil_frames(video_uint8))
        if len(frames) != 121:
            raise RuntimeError(
                "JoyAI-Echo native shot must decode exactly 121 complete frames; "
                f"got {len(frames)}."
            )
        audio_hash = encoding_fingerprint(audio_waveform)
        _move_module(getattr(self.video_vae, "decoder", None), "cpu")
        _move_module(getattr(self.audio_vae, "decoder", None), "cpu")
        _move_module(getattr(self.audio_vae, "vocoder", None), "cpu")
        _move_module(self.generator, self.device)
        _empty_cuda_cache(self.device)
        return frames, audio_hash

    def _append_segment(self, frames: Sequence[Any], audio_hash: str) -> None:
        complete = list(frames)
        if len(complete) != 121:
            raise RuntimeError("JoyAI-Echo evidence requires each complete 121-frame shot.")
        if len(self._completed_segments) >= 3:
            raise RuntimeError("JoyAI-Echo received more than three native shots.")
        self._completed_segments.append(complete)
        self._segment_audio_hashes.append(str(audio_hash))

    def _build_paired_tail_memory(
        self,
        frames: Sequence[Any],
        audio_latent: torch.Tensor,
        *,
        source_segment_index: int,
        state: AdapterState,
    ) -> dict[str, Any]:
        if self._memory_bank is None or self.encode_memory_frames_batch is None:
            raise RuntimeError("JoyAI-Echo paired-memory helpers were not initialized.")
        if len(frames) != 121 or int(audio_latent.shape[1]) != 126:
            raise RuntimeError("JoyAI-Echo memory extraction requires complete native AV state.")
        tail_frames = list(frames[-9:])
        tail_audio = audio_latent[:, -96:].detach().cpu().contiguous()
        if len(tail_frames) != 9 or int(tail_audio.shape[1]) != 96:
            raise RuntimeError("JoyAI-Echo tail crop did not produce exact 9/96 memory inputs.")
        metadata = self._memory_bank.save_memory_slot(
            tail_frames,
            tail_audio,
            audio_window_size=96,
            video_clip_num_frames=9,
            audio_waveform=None,
            audio_sample_rate=16000,
            video_fps=24.0,
            audio_window_selection_mode="center",
            video_frame_selection_mode="center",
            audio_memory_mel_bins=128,
            audio_memory_mel_hop_length=160,
            audio_memory_n_fft=1024,
            audio_memory_downsample_factor=4,
            audio_memory_is_causal=True,
        )
        if len(self._memory_bank) != 1:
            raise RuntimeError("JoyAI-Echo memory bank must retain exactly one latest entry.")
        required_metadata = {
            "audio_window_start": 0,
            "audio_window_end": 96,
            "audio_window_length": 96,
            "audio_total_frames": 96,
            "video_clip_start": 0,
            "video_clip_end": 9,
            "video_clip_length": 9,
            "video_total_frames": 9,
        }
        drift = {
            key: {"expected": value, "observed": metadata.get(key)}
            for key, value in required_metadata.items()
            if metadata.get(key) != value
        }
        if drift:
            raise RuntimeError(
                "JoyAI-Echo released memory helper drifted from exact tail selection: "
                f"{drift}."
            )

        _move_module(getattr(self.video_vae, "encoder", None), self.device)
        memory_video = self.encode_memory_frames_batch(
            video_vae=self.video_vae,
            batch_memory_frames=[self._memory_bank.get_memory_frames()],
            target_h=int(state.extra["height"]),
            target_w=int(state.extra["width"]),
            device=self.device,
            dtype=self.dtype,
        )
        _move_module(getattr(self.video_vae, "encoder", None), "cpu")
        _move_module(self.text_encoder, self.device)
        _empty_cuda_cache(self.device)
        expected_memory_shape = (
            1,
            1,
            int(state.extra["video_shape"][2]),
            int(state.extra["video_shape"][3]),
            int(state.extra["video_shape"][4]),
        )
        if tuple(memory_video.shape) != expected_memory_shape:
            raise RuntimeError(
                "JoyAI-Echo encode_memory_frames_batch must retain exactly one latent: "
                f"expected {expected_memory_shape}, got {tuple(memory_video.shape)}."
            )
        if memory_video.device != self.device or memory_video.dtype != self.dtype:
            raise RuntimeError("JoyAI-Echo encoded video memory device/dtype changed.")
        if not torch.isfinite(memory_video).all():
            raise RuntimeError("JoyAI-Echo encoded video memory contains non-finite values.")

        memory_audio = self._memory_bank.get_memory_audio()
        if not isinstance(memory_audio, torch.Tensor):
            raise RuntimeError("JoyAI-Echo paired memory bank did not expose audio latents.")
        memory_audio = memory_audio.to(device=self.device, dtype=self.dtype).contiguous()
        if tuple(memory_audio.shape) != (
            1,
            96,
            int(state.extra["audio_shape"][2]),
        ):
            raise RuntimeError("JoyAI-Echo memory audio shape differs from exact 96-tail route.")
        lengths = self._memory_bank.get_memory_audio_segment_lengths()
        if lengths != ((96,),):
            raise RuntimeError("JoyAI-Echo paired memory segment lengths changed.")
        memory_timestep = torch.zeros(
            memory_audio.shape[:2],
            device=self.device,
            dtype=torch.float32,
        )
        kwargs = {
            "memory_video": memory_video,
            "memory_audio": memory_audio,
            "memory_audio_timestep": memory_timestep,
            "memory_audio_segment_lengths": lengths,
            "paired_audio_memory": True,
            "v2a_grad_scale": 1.0,
            "memory_position_mode": "reference",
            "memory_downscale_factor": 1,
        }
        fingerprints = _memory_fingerprint_record(kwargs)
        record = {
            "schema_version": 1,
            "source_segment_index": source_segment_index,
            "target_segment_index": source_segment_index + 1,
            "tail_video_source_indices": list(range(112, 121)),
            "tail_audio_source_slice": [30, 126],
            "released_selection_metadata": dict(metadata),
            "memory_entry_count": 1,
            "encoded_video_latent_count": int(memory_video.shape[1]),
            "memory_argument_fingerprints": fingerprints,
            "memory_video": memory_video,
            "memory_audio": memory_audio,
            "memory_audio_timestep": memory_timestep,
            "memory_audio_segment_lengths": lengths,
        }
        self._memory_transition_records[source_segment_index + 1] = {
            key: value
            for key, value in record.items()
            if not isinstance(value, torch.Tensor)
        }
        return record

    def _start_memory_segment(
        self,
        *,
        segment_index: int,
        state: AdapterState,
        memory_record: Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if segment_index not in {1, 2}:
            raise RuntimeError("JoyAI-Echo memory segment index must be 1 or 2.")
        seed = int(state.extra["segment_seeds"][segment_index])
        segment_generator = _make_generator(seed, self.device)
        video = torch.randn(
            tuple(state.extra["video_shape"]),
            generator=segment_generator,
            device=self.device,
            dtype=self.dtype,
        )
        audio = torch.randn(
            tuple(state.extra["audio_shape"]),
            generator=segment_generator,
            device=self.device,
            dtype=self.dtype,
        )
        state.extra.update(
            {
                "segment_index": segment_index,
                "segment_step_index": 0,
                "condition_epoch": segment_index,
                "segment_seed": seed,
                "segment_generator": segment_generator,
                "model_role": "memory_t2av",
                "anchor_sha256": None,
                "next_sigmas": self._sigmas()[1:],
                "memory_video": memory_record["memory_video"],
                "memory_audio": memory_record["memory_audio"],
                "memory_audio_timestep": memory_record["memory_audio_timestep"],
                "memory_audio_segment_lengths": memory_record[
                    "memory_audio_segment_lengths"
                ],
                "memory_argument_fingerprints": memory_record[
                    "memory_argument_fingerprints"
                ],
                "protected_state_trace": {
                    "schema_version": 1,
                    "segment_index": segment_index,
                    "segment_seed": seed,
                    "memory": memory_record["memory_argument_fingerprints"],
                    "fresh_video_noise_fingerprint": encoding_fingerprint(video),
                    "fresh_audio_noise_fingerprint": encoding_fingerprint(audio),
                    "steering_target": "video_vector_field_only",
                    "audio_state_source": "base_current_branch_only",
                },
            }
        )
        state.extra.pop("base_audio_prediction", None)
        state.extra.pop("base_audio_prediction_fingerprint", None)
        return video, audio

    def _stitch_and_resample_segments(
        self,
        segments: Sequence[Sequence[Any]],
        *,
        state: AdapterState,
        generation_path: str,
    ) -> list[list[Any]]:
        if len(segments) != 3 or any(len(segment) != 121 for segment in segments):
            raise RuntimeError(
                "JoyAI-Echo reference-conditioned multishot requires three complete 121-frame shots."
            )
        retained = (tuple(range(120)), tuple(range(120)), tuple(range(120)))
        stitch_map = tuple(
            (segment_index, frame_index)
            for segment_index, indices in enumerate(retained)
            for frame_index in indices
        )
        stitched = [segments[s][f] for s, f in stitch_map]
        if len(stitched) != 360:
            raise RuntimeError("JoyAI-Echo honest half-open stitch must be 120+120+120=360.")
        selected_indices = _round_half_up_decimation_indices(
            source_frames=360,
            source_fps=24,
            output_frames=240,
            output_fps=16,
        )
        output = [stitched[index] for index in selected_indices]
        if len(output) != 240 or len(set(selected_indices)) != 240:
            raise RuntimeError("JoyAI-Echo exact decimation must select 240 unique frames.")

        prompt_sha256 = hashlib.sha256(
            str(state.extra["base_prompt"]).encode("utf-8")
        ).hexdigest()
        sigma_values = [float(value) for value in self._sigmas().detach().cpu().tolist()]
        scheduler_digest = hashlib.sha256(
            canonical_json_bytes({"sigmas": sigma_values, "resets": [0, 0, 0]})
        ).hexdigest()
        evidence_segments: list[TemporalSegmentEvidence] = []
        for segment_index, segment in enumerate(segments):
            conditioning_records = self._segment_conditioning_records.get(segment_index, [])
            protected_records = self._segment_protected_records.get(segment_index, [])
            if not conditioning_records or not protected_records:
                raise RuntimeError(
                    f"JoyAI-Echo segment {segment_index} lacks branch/protected-state evidence."
                )
            memory_record = self._memory_transition_records.get(segment_index)
            if segment_index == 0 and memory_record is not None:
                raise RuntimeError("JoyAI-Echo base shot unexpectedly contains memory evidence.")
            if segment_index > 0 and memory_record is None:
                raise RuntimeError("JoyAI-Echo memory shot lacks a released-helper transition.")
            per_step_branch_evidence: dict[str, Any] = {}
            for local_step in range(8):
                step_records = [
                    record
                    for record in conditioning_records
                    if record.get("local_step_index") == local_step
                ]
                if not step_records:
                    raise RuntimeError(
                        "JoyAI-Echo branch evidence does not cover every local sigma step."
                    )
                memory_records = {
                    hashlib.sha256(canonical_json_bytes(record["memory"])).hexdigest(): record[
                        "memory"
                    ]
                    for record in step_records
                }
                if len(memory_records) != 1:
                    raise RuntimeError(
                        "JoyAI-Echo prediction branches received different memory arguments."
                    )
                base_audio_candidates = sum(
                    bool(record.get("base_current_audio_candidate"))
                    for record in step_records
                )
                if base_audio_candidates != 1:
                    raise RuntimeError(
                        "JoyAI-Echo requires exactly one current/base audio candidate per step."
                    )
                per_step_branch_evidence[str(local_step)] = {
                    "prediction_branch_count": len(step_records),
                    "base_current_audio_candidate_count": base_audio_candidates,
                    "unique_memory_argument_record_count": len(memory_records),
                    "memory_argument_fingerprints": next(iter(memory_records.values())),
                    "ordered_branch_record_sha256": hashlib.sha256(
                        canonical_json_bytes(step_records)
                    ).hexdigest(),
                }
            evidence_segments.append(
                TemporalSegmentEvidence(
                    segment_index=segment_index,
                    model_role="base_t2av" if segment_index == 0 else "memory_t2av",
                    model_id=self.model_id,
                    model_revision=self._require_commit_revision("revision"),
                    segment_seed=int(self._segment_seeds[segment_index]),
                    native_fps=24,
                    frames=tuple(segment),
                    retained_indices=retained[segment_index],
                    discarded_indices=(120,),
                    anchor_sha256=None,
                    reconstruction_index=None,
                    first_motion_index=None,
                    scheduler={
                        "class": "released_DMD_sigma_schedule",
                        "denoising_steps": 8,
                        "sigma_values": sigma_values,
                        "reset_index": 0,
                        "schedule_sha256": scheduler_digest,
                    },
                    conditioning={
                        "positive_prompt_sha256": prompt_sha256,
                        "prediction_call_count": len(conditioning_records),
                        "text_encoding_fingerprints": sorted(
                            {
                                record["conditioning_fingerprint"]
                                for record in conditioning_records
                            }
                        ),
                        "ordered_call_record_sha256": hashlib.sha256(
                            canonical_json_bytes(conditioning_records)
                        ).hexdigest(),
                    },
                    protected_state={
                        "memory_present": segment_index > 0,
                        "memory_transition": memory_record,
                        "branch_memory_byte_identity_validated": True,
                        "per_step_branch_memory_validation": per_step_branch_evidence,
                        "steering_target": "video_vector_field_only",
                        "audio_state_source": "base_current_branch_only",
                        "decoded_audio_sha256": self._segment_audio_hashes[segment_index],
                        "ordered_protected_record_sha256": hashlib.sha256(
                            canonical_json_bytes(protected_records)
                        ).hexdigest(),
                    },
                )
            )

        protocol = self._temporal_protocol()
        assert protocol is not None
        self._temporal_evidence = TemporalEvidenceBundle(
            temporal_protocol=protocol,
            segments=tuple(evidence_segments),
            stitch_map=stitch_map,
            output_fps=16,
            output_frame_count=240,
            stitch_strategy=(
                "segment_0[0:120]+segment_1[0:120]+segment_2[0:120]"
            ),
            postprocess={
                "method": "nearest_timestamp_decimation_round_half_up",
                "source_frame_count": 360,
                "source_fps": 24,
                "output_frame_count": 240,
                "output_fps": 16,
                "selected_source_indices": selected_indices,
                "selected_source_index_prefix": selected_indices[:6],
                "last_selected_source_index": selected_indices[-1],
                "unique_source_indices": True,
                "synthesized_frames": 0,
                "duplicated_frames": 0,
                "source_seam_transitions": [[119, 120], [239, 240]],
                "output_seam_transitions": [[79, 80], [159, 160]],
            },
            scientific_label=str(protocol["scientific_label"]),
            metadata={
                "generation_path": generation_path,
                "artifact_authentication": self._artifact_authentication,
                "seed_domain": JOYAI_ECHO_SEGMENT_SEED_DOMAIN,
                "base_seed": int(state.extra["base_seed"]),
                "segment_seeds": list(self._segment_seeds),
                "memory_transitions": {
                    str(index): deepcopy(record)
                    for index, record in self._memory_transition_records.items()
                },
                "audio_saved_to_output_media": False,
            },
        )
        self._last_temporal_provenance = {
            **self._configured_temporal_provenance(protocol),
            "status": "completed",
            "generation_path": generation_path,
            "native_segment_frame_counts": [121, 121, 121],
            "retained_half_open_frame_counts": [120, 120, 120],
            "stitched_native_frame_count": 360,
            "output_frame_count": 240,
            "segment_seeds": list(self._segment_seeds),
        }
        return [output]

    def _authenticate_artifact_manifest(
        self,
        protocol: Mapping[str, Any],
        source_root: Path,
    ) -> dict[str, Any]:
        project_root = Path(
            str(self.config.get("project_root") or Path(__file__).resolve().parents[3])
        ).resolve()
        raw_path = Path(str(protocol["artifact_manifest"]))
        manifest_path = (
            raw_path.resolve() if raw_path.is_absolute() else (project_root / raw_path).resolve()
        )
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"JoyAI-Echo schema-2 artifact manifest is missing: {manifest_path}"
            )
        raw = manifest_path.read_bytes()
        actual_manifest_sha = hashlib.sha256(raw).hexdigest()
        expected_manifest_sha = str(protocol["artifact_manifest_sha256"])
        if actual_manifest_sha != expected_manifest_sha:
            raise RuntimeError(
                "JoyAI-Echo artifact manifest SHA-256 mismatch: "
                f"expected {expected_manifest_sha}, got {actual_manifest_sha}."
            )
        manifest = json.loads(raw)
        if (
            manifest.get("schema_version") != 1
            or manifest.get("scientific_identity")
            != "joyai_echo_reference_conditioned_multishot_segmented_temporal_v2"
            or manifest.get("scientific_label") != protocol["scientific_label"]
        ):
            raise RuntimeError("JoyAI-Echo artifact manifest scientific identity changed.")
        checkpoint = manifest.get("checkpoint")
        if not isinstance(checkpoint, Mapping) or {
            "model_id": checkpoint.get("model_id"),
            "revision": checkpoint.get("revision"),
            "filename": checkpoint.get("filename"),
            "sha256": checkpoint.get("sha256"),
        } != {
            "model_id": self.model_id,
            "revision": self._require_commit_revision("revision"),
            "filename": self.checkpoint_filename,
            "sha256": self._require_sha256("checkpoint_sha256"),
        }:
            raise RuntimeError("JoyAI-Echo artifact checkpoint identity differs from config.")
        text_encoder = manifest.get("text_encoder")
        if not isinstance(text_encoder, Mapping) or {
            "model_id": text_encoder.get("model_id"),
            "revision": text_encoder.get("revision"),
        } != {
            "model_id": self.gemma_model_id,
            "revision": self._require_commit_revision("gemma_revision"),
        }:
            raise RuntimeError("JoyAI-Echo artifact text-encoder identity differs from config.")
        route_source = manifest.get("route_source")
        if not isinstance(route_source, Mapping):
            raise RuntimeError("JoyAI-Echo artifact route-source record is missing.")
        expected_source = {
            "revision": self._require_commit_revision("source_revision"),
            "tracked_binary_diff_sha256": self._require_sha256(
                "source_tracked_diff_sha256"
            ),
        }
        if any(route_source.get(key) != value for key, value in expected_source.items()):
            raise RuntimeError("JoyAI-Echo artifact source identity differs from config.")
        if Path(str(route_source.get("root"))).resolve() != source_root.resolve():
            raise RuntimeError("JoyAI-Echo artifact source root differs from configured checkout.")
        if self._source_provenance is not None:
            for key, provenance_key in {
                "status_porcelain_v1_z_sha256": "status_porcelain_v1_z_sha256",
                "tracked_binary_diff_size_bytes": "tracked_binary_diff_size_bytes",
            }.items():
                if route_source.get(key) != self._source_provenance.get(provenance_key):
                    raise RuntimeError(
                        f"JoyAI-Echo artifact source provenance field changed: {key}."
                    )

        permitted_overrides = route_source.get("permitted_overrides")
        if not isinstance(permitted_overrides, list) or not permitted_overrides:
            raise RuntimeError("JoyAI-Echo artifact permitted-override inventory is missing.")
        configured_overrides = self._validated_source_overrides()
        authenticated_overrides: list[dict[str, Any]] = []
        if {str(record.get("path")) for record in permitted_overrides if isinstance(record, Mapping)} != set(
            configured_overrides
        ):
            raise RuntimeError("JoyAI-Echo permitted source overrides differ from config.")
        for record in permitted_overrides:
            if not isinstance(record, Mapping):
                raise RuntimeError("JoyAI-Echo permitted source override record is malformed.")
            relative = str(record.get("path", ""))
            candidate = _resolve_contained_file(source_root, relative)
            observed_size = candidate.stat().st_size
            observed_sha = _sha256_file(candidate)
            if (
                observed_sha != configured_overrides[relative]
                or observed_sha != record.get("sha256")
                or observed_size != record.get("size_bytes")
            ):
                raise RuntimeError(
                    f"JoyAI-Echo permitted source override changed: {relative!r}."
                )
            authenticated_overrides.append(
                {"path": relative, "size_bytes": observed_size, "sha256": observed_sha}
            )

        required_files = route_source.get("required_files")
        if not isinstance(required_files, list) or not required_files:
            raise RuntimeError("JoyAI-Echo artifact manifest has no complete source inventory.")
        authenticated_files: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        for record in required_files:
            if not isinstance(record, Mapping):
                raise RuntimeError("JoyAI-Echo source inventory contains a malformed record.")
            relative = str(record.get("path", ""))
            if relative in seen_paths:
                raise RuntimeError("JoyAI-Echo source inventory contains duplicate paths.")
            seen_paths.add(relative)
            candidate = _resolve_contained_file(source_root, relative)
            observed_size = candidate.stat().st_size
            observed_sha = _sha256_file(candidate)
            if observed_size != record.get("size_bytes") or observed_sha != record.get(
                "sha256"
            ):
                raise RuntimeError(
                    f"JoyAI-Echo authenticated source file changed: {relative!r}."
                )
            authenticated_files.append(
                {"path": relative, "size_bytes": observed_size, "sha256": observed_sha}
            )

        callables = route_source.get("callables")
        if not isinstance(callables, list) or not callables:
            raise RuntimeError("JoyAI-Echo artifact manifest has no callable signatures.")
        authenticated_callables: list[dict[str, str]] = []
        for record in callables:
            if not isinstance(record, Mapping):
                raise RuntimeError("JoyAI-Echo callable manifest record is malformed.")
            relative = str(record.get("source_path", ""))
            candidate = _resolve_contained_file(source_root, relative)
            qualified_name = str(record.get("qualified_name", ""))
            signature = _ast_callable_signature(candidate, qualified_name)
            signature_sha = hashlib.sha256(signature.encode("utf-8")).hexdigest()
            if signature != record.get("signature") or signature_sha != record.get(
                "signature_sha256"
            ):
                raise RuntimeError(
                    "JoyAI-Echo released callable signature changed: "
                    f"{qualified_name!r}."
                )
            authenticated_callables.append(
                {
                    "qualified_name": qualified_name,
                    "source_path": relative,
                    "signature": signature,
                    "signature_sha256": signature_sha,
                }
            )
        return {
            "schema_version": 1,
            "manifest_path": str(manifest_path),
            "manifest_sha256": actual_manifest_sha,
            "checkpoint": dict(checkpoint),
            "text_encoder": dict(text_encoder),
            "source_revision": expected_source["revision"],
            "source_tracked_diff_sha256": expected_source[
                "tracked_binary_diff_sha256"
            ],
            "authenticated_permitted_overrides": authenticated_overrides,
            "authenticated_required_files": authenticated_files,
            "authenticated_callables": authenticated_callables,
        }

    def _assert_source_and_route_unchanged(
        self,
        protocol: Mapping[str, Any],
    ) -> None:
        if self.source_root is None or self._source_provenance is None:
            # Unit doubles may mark the adapter loaded without a source tree.
            # Production load always sets both and therefore always rechecks.
            return
        current_source = self._validate_source_checkout(self.source_root)
        if current_source != self._source_provenance:
            raise RuntimeError("JoyAI-Echo external source checkout changed after load.")
        current_artifacts = self._authenticate_artifact_manifest(protocol, self.source_root)
        if self._artifact_authentication is None or current_artifacts != self._artifact_authentication:
            raise RuntimeError("JoyAI-Echo route artifacts changed after load.")

    def _resolve_source_root(self) -> Path:
        configured = self.config.get("source_root") or os.environ.get("JOYAI_ECHO_ROOT")
        if not configured:
            raise FileNotFoundError(
                "JoyAI-Echo source tree not configured. Set model.source_root in the model config "
                "or set JOYAI_ECHO_ROOT to the official inference repository path."
            )
        source_root = Path(str(configured)).expanduser()
        if not source_root.exists():
            raise FileNotFoundError(
                "JoyAI-Echo source tree not found. Clone the official inference repo or set "
                f"JOYAI_ECHO_ROOT. Tried: {source_root}"
            )
        return source_root.resolve()

    def _local_ablation_metadata(self) -> dict[str, Any] | None:
        model_path = Path(str(self.model_id)).expanduser()
        metadata_path = model_path / "joyai_echo_abliteration.json"
        if not metadata_path.exists():
            return None
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata["_artifact_dir"] = str(model_path.resolve())
        return metadata

    def _resolve_checkpoint_path(self) -> Path:
        revision = self._require_commit_revision("revision")
        configured = self.config.get("checkpoint_path") or os.environ.get("JOYAI_ECHO_CHECKPOINT")
        if configured:
            checkpoint = Path(str(configured)).expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"JoyAI-Echo checkpoint not found: {checkpoint}")
            return checkpoint
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=self.model_id,
                filename=self.checkpoint_filename,
                revision=revision,
            )
        ).resolve()

    def _resolve_gemma_path(self) -> Path:
        revision = self._require_commit_revision("gemma_revision")
        configured = self.config.get("gemma_path") or os.environ.get("JOYAI_GEMMA_PATH")
        if configured:
            gemma_path = Path(str(configured)).expanduser().resolve()
            if not gemma_path.is_dir():
                raise FileNotFoundError(
                    f"JoyAI-Echo Gemma text encoder path not found: {gemma_path}"
                )
            self._validate_local_snapshot_revision(
                gemma_path,
                revision=revision,
                artifact_name="JoyAI-Echo Gemma text encoder",
            )
            return gemma_path
        from huggingface_hub import snapshot_download

        gemma_path = Path(
            snapshot_download(
                repo_id=self.gemma_model_id,
                revision=revision,
            )
        ).resolve()
        self._validate_local_snapshot_revision(
            gemma_path,
            revision=revision,
            artifact_name="downloaded JoyAI-Echo Gemma text encoder",
        )
        return gemma_path

    def _validate_source_checkout(self, source_root: Path) -> dict[str, Any]:
        """Bind execution to one deliberately patched JoyAI source checkout.

        The approved checkout is not clean because one compatibility repair is
        needed for the installed Transformers version.  Consequently checking
        only ``git rev-parse HEAD`` is insufficient.  We additionally require
        the complete porcelain status, the SHA-256 of the complete binary diff,
        and the SHA-256 of every explicitly permitted modified file.
        """

        expected_revision = self._require_commit_revision("source_revision")
        expected_diff_sha256 = self._require_sha256("source_tracked_diff_sha256")
        overrides = self._validated_source_overrides()

        head_bytes = self._run_git(source_root, "rev-parse", "--verify", "HEAD")
        try:
            resolved_head = head_bytes.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise RuntimeError("JoyAI-Echo source HEAD was not ASCII.") from exc
        if not re.fullmatch(r"[0-9a-f]{40}", resolved_head):
            raise RuntimeError(
                "JoyAI-Echo source HEAD did not resolve to an exact lowercase 40-hex commit: "
                f"{resolved_head!r}."
            )
        if resolved_head != expected_revision:
            raise RuntimeError(
                "JoyAI-Echo source revision mismatch: "
                f"expected {expected_revision}, resolved {resolved_head}."
            )

        status_bytes = self._run_git(
            source_root,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        )
        expected_status = b"".join(
            f" M {relative_path}\0".encode("utf-8") for relative_path in sorted(overrides)
        )
        if status_bytes != expected_status:
            actual_records = _decode_porcelain_records(status_bytes)
            expected_records = _decode_porcelain_records(expected_status)
            raise RuntimeError(
                "JoyAI-Echo source checkout has changes outside the exact permitted override set: "
                f"expected {expected_records!r}, got {actual_records!r}."
            )

        diff_bytes = self._run_git(
            source_root,
            "diff",
            "--binary",
            "--no-ext-diff",
            "HEAD",
        )
        actual_diff_sha256 = hashlib.sha256(diff_bytes).hexdigest()
        if actual_diff_sha256 != expected_diff_sha256:
            raise RuntimeError(
                "JoyAI-Echo complete tracked source diff SHA-256 mismatch: "
                f"expected {expected_diff_sha256}, got {actual_diff_sha256}."
            )

        override_records: list[dict[str, str]] = []
        for relative_path, expected_sha256 in sorted(overrides.items()):
            candidate = _resolve_contained_file(source_root, relative_path)
            actual_sha256 = _sha256_file(candidate)
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"JoyAI-Echo source override SHA-256 mismatch for {relative_path!r}: "
                    f"expected {expected_sha256}, got {actual_sha256}."
                )
            override_records.append(
                {
                    "path": relative_path,
                    "sha256": actual_sha256,
                }
            )

        provenance = {
            "schema_version": 1,
            "source_root": str(source_root),
            "source_revision": resolved_head,
            "status_porcelain_v1_z_sha256": hashlib.sha256(status_bytes).hexdigest(),
            "status_records": _decode_porcelain_records(status_bytes),
            "tracked_binary_diff_sha256": actual_diff_sha256,
            "tracked_binary_diff_size_bytes": len(diff_bytes),
            "file_overrides": override_records,
        }
        return provenance

    def _validate_checkpoint(self, checkpoint_path: Path) -> str:
        expected_sha256 = self._require_sha256("checkpoint_sha256")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"JoyAI-Echo checkpoint not found: {checkpoint_path}")
        actual_sha256 = _sha256_file(checkpoint_path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                "JoyAI-Echo checkpoint SHA-256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}."
            )
        return actual_sha256

    def _validated_source_overrides(self) -> dict[str, str]:
        raw_overrides = self.config.get("source_file_overrides")
        if not isinstance(raw_overrides, Mapping):
            raise ValueError("model.source_file_overrides must be a path-to-SHA-256 mapping.")
        overrides: dict[str, str] = {}
        for raw_path, raw_sha256 in raw_overrides.items():
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(
                    "Every JoyAI-Echo source override path must be a non-empty string."
                )
            path = Path(raw_path)
            if path.is_absolute() or ".." in path.parts or path.as_posix() != raw_path:
                raise ValueError(
                    "JoyAI-Echo source override paths must be normalized relative POSIX paths; "
                    f"got {raw_path!r}."
                )
            if not isinstance(raw_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", raw_sha256):
                raise ValueError(
                    f"JoyAI-Echo source override {raw_path!r} needs an exact lowercase 64-hex SHA-256."
                )
            overrides[raw_path] = raw_sha256
        return overrides

    def _require_commit_revision(self, key: str) -> str:
        value = self.config.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError(
                f"JoyAI-Echo model.{key} must be an exact lowercase 40-hex commit; got {value!r}."
            )
        return value

    def _require_sha256(self, key: str) -> str:
        value = self.config.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(
                f"JoyAI-Echo model.{key} must be an exact lowercase 64-hex SHA-256; got {value!r}."
            )
        return value

    @staticmethod
    def _run_git(source_root: Path, *args: str) -> bytes:
        try:
            result = subprocess.run(
                ["git", "-c", "core.quotepath=false", "-C", str(source_root), *args],
                check=False,
                capture_output=True,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"Could not inspect JoyAI-Echo source checkout: {exc}") from exc
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"JoyAI-Echo source git command failed ({' '.join(args)}): {stderr or 'no stderr'}"
            )
        return result.stdout

    @staticmethod
    def _validate_local_snapshot_revision(
        snapshot_path: Path,
        *,
        revision: str,
        artifact_name: str,
    ) -> None:
        """Require the canonical Hugging Face ``snapshots/<commit>`` identity.

        Unlike the checkpoint, no compact upstream digest is published for the
        multi-file Gemma snapshot.  An arbitrary directory therefore cannot be
        proven equivalent and is rejected rather than silently trusted.
        """

        if snapshot_path.name != revision or snapshot_path.parent.name != "snapshots":
            raise RuntimeError(
                f"{artifact_name} path is not bound to the configured commit {revision}: "
                f"expected a .../snapshots/{revision} directory, got {snapshot_path}."
            )

    def conditioning_provenance(self) -> dict[str, Any]:
        self._require_loaded()
        if (
            self.source_root is None
            or self.checkpoint_path is None
            or self.gemma_path is None
            or self._source_provenance is None
            or self._checkpoint_sha256 is None
        ):
            raise RuntimeError("JoyAI-Echo artifact provenance is incomplete after model loading.")
        provenance = {
            "schema_version": 1,
            "adapter": self.adapter_name,
            "model_id": self.model_id,
            "model_revision": self._require_commit_revision("revision"),
            "checkpoint": {
                "path": str(self.checkpoint_path),
                "filename": self.checkpoint_path.name,
                "sha256": self._checkpoint_sha256,
            },
            "gemma": {
                "model_id": self.gemma_model_id,
                "path": str(self.gemma_path),
                "revision": self._require_commit_revision("gemma_revision"),
            },
            "source_checkout": dict(self._source_provenance),
            "segmented_route_artifacts": deepcopy(self._artifact_authentication),
        }
        temporal_protocol = self._temporal_protocol()
        if temporal_protocol is not None:
            provenance["temporal_generation"] = (
                dict(self._last_temporal_provenance)
                if self._last_temporal_provenance is not None
                else self._configured_temporal_provenance(temporal_protocol)
            )
        return provenance

    def _add_source_tree(self, source_root: Path) -> None:
        for relative in ("ltx-core/src", "ltx-pipelines/src", "ltx-distillation/src"):
            path = str(source_root / relative)
            if path not in sys.path:
                sys.path.insert(0, path)

    def _sigmas(self) -> torch.Tensor:
        protocol = self._temporal_protocol()
        values = (
            protocol["denoising_sigmas"]
            if protocol is not None
            else self.config.get("denoising_sigmas", self.default_sigmas)
        )
        normalized = tuple(float(value) for value in values)
        if protocol is not None and normalized != self.default_sigmas:
            raise RuntimeError("JoyAI-Echo schema-2 sigma schedule differs from release.")
        return torch.tensor(
            normalized, device=self.device, dtype=torch.float32
        )

    def _freeze_loaded_modules(self) -> None:
        for module in (self.text_encoder, self.generator, self.video_vae, self.audio_vae):
            if module is None:
                continue
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    @staticmethod
    def _sigma_value(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().flatten()[0].item())
        return float(value)


def _derive_segment_seed(base_seed: int, segment_index: int) -> int:
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("JoyAI-Echo base seed must be a non-negative integer.")
    if segment_index not in {1, 2}:
        raise ValueError("JoyAI-Echo derived segment index must be 1 or 2.")
    payload = (
        f"{JOYAI_ECHO_SEGMENT_SEED_DOMAIN}|{base_seed}|{segment_index}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _make_generator(seed: int, device: torch.device) -> torch.Generator:
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed))
    return generator


def _round_half_up_decimation_indices(
    *,
    source_frames: int,
    source_fps: int,
    output_frames: int,
    output_fps: int,
) -> list[int]:
    if (source_frames, source_fps, output_frames, output_fps) != (360, 24, 240, 16):
        raise ValueError("JoyAI-Echo exact decimator is sealed to 360@24 -> 240@16.")
    indices = [
        (output_index * source_fps * 2 + output_fps) // (2 * output_fps)
        for output_index in range(output_frames)
    ]
    if indices[:6] != [0, 2, 3, 5, 6, 8] or indices[-1] != 359:
        raise RuntimeError("JoyAI-Echo round-half-up index arithmetic changed.")
    if len(indices) != len(set(indices)) or any(
        not 0 <= index < source_frames for index in indices
    ):
        raise RuntimeError("JoyAI-Echo exact decimator duplicated/out-of-range a frame.")
    return indices


def _memory_fingerprint_record(memory_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    if not memory_kwargs:
        return {
            "present": False,
            "memory_video_sha256": None,
            "memory_audio_sha256": None,
            "memory_audio_timestep_sha256": None,
            "memory_audio_segment_lengths_sha256": None,
            "paired_audio_memory": False,
            "v2a_grad_scale": None,
            "memory_position_mode": None,
            "memory_downscale_factor": None,
        }
    return {
        "present": True,
        "memory_video_sha256": encoding_fingerprint(memory_kwargs["memory_video"]),
        "memory_audio_sha256": encoding_fingerprint(memory_kwargs["memory_audio"]),
        "memory_audio_timestep_sha256": encoding_fingerprint(
            memory_kwargs["memory_audio_timestep"]
        ),
        "memory_audio_segment_lengths_sha256": encoding_fingerprint(
            memory_kwargs["memory_audio_segment_lengths"]
        ),
        "paired_audio_memory": memory_kwargs.get("paired_audio_memory"),
        "v2a_grad_scale": memory_kwargs.get("v2a_grad_scale"),
        "memory_position_mode": memory_kwargs.get("memory_position_mode"),
        "memory_downscale_factor": memory_kwargs.get("memory_downscale_factor"),
    }


def _move_module(module: Any, device: torch.device | str) -> None:
    if module is not None and callable(getattr(module, "to", None)):
        module.to(device)


def _empty_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def _ast_callable_signature(path: Path, qualified_name: str) -> str:
    parts = qualified_name.split(".")
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    if qualified_name == "encode_memory_frames_batch":
        for node in module.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                node.name == qualified_name
            ):
                return ast.unparse(node.args)
    if len(parts) != 2 or not all(parts):
        raise RuntimeError(
            f"JoyAI-Echo callable name must be Class.method; got {qualified_name!r}."
        )
    class_name, function_name = parts
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                    child.name == function_name
                ):
                    return ast.unparse(child.args)
    raise RuntimeError(
        f"JoyAI-Echo authenticated callable was not found: {qualified_name!r}."
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_contained_file(source_root: Path, relative_path: str) -> Path:
    candidate = source_root / relative_path
    if candidate.is_symlink():
        raise RuntimeError(f"JoyAI-Echo source override may not be a symlink: {relative_path!r}.")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(source_root.resolve())
    except ValueError as exc:
        raise RuntimeError(
            f"JoyAI-Echo source override escapes the configured source root: {relative_path!r}."
        ) from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"JoyAI-Echo source override not found: {resolved}")
    return resolved


def _decode_porcelain_records(status_bytes: bytes) -> list[str]:
    try:
        return [record.decode("utf-8") for record in status_bytes.split(b"\0") if record]
    except UnicodeDecodeError as exc:
        raise RuntimeError("JoyAI-Echo git status contained a non-UTF-8 path.") from exc


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
    adapter_name: str,
) -> None:
    if output_frames <= 0:
        raise ValueError(f"{adapter_name} output frame count must be positive.")
    expected_duration = output_frames / output_fps
    raw_duration = generation_kwargs.get("duration_seconds")
    if raw_duration is None:
        return
    duration = _finite_positive_float(raw_duration, f"{adapter_name} duration_seconds")
    if duration != expected_duration:
        raise ValueError(
            f"{adapter_name} requested duration/fps/frame arithmetic is inconsistent: "
            f"{output_frames}/{output_fps}={expected_duration}, got duration_seconds={duration}."
        )


def _validate_production_gate(
    protocol: Mapping[str, Any],
    *,
    adapter_name: str,
    criteria_names: tuple[str, ...],
) -> None:
    del adapter_name
    validate_temporal_production_gate(
        protocol,
        model_name="joyai_echo",
        model_revision=JOYAI_ECHO_CHECKPOINT_REVISION,
        criteria_names=criteria_names,
    )


def _direct_temporal_provenance(
    *,
    adapter_name: str,
    internal_frames: int,
    output_frames: int,
    fps: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "configured_not_yet_decoded",
        "strategy": "direct_aligned_generation",
        "adapter": adapter_name,
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


def _compute_latent_shapes(
    *,
    num_frames: int,
    video_height: int,
    video_width: int,
    batch_size: int = 1,
    latent_channels: int = 128,
    vae_temporal_compression: int = 8,
    vae_spatial_compression: int = 32,
    video_fps: float = 24.0,
    audio_sample_rate: int = 16000,
    audio_hop_length: int = 160,
    audio_latent_downsample: int = 4,
) -> tuple[list[int], list[int]]:
    if (num_frames - 1) % vae_temporal_compression != 0:
        raise ValueError(f"num_frames must be 1 + 8*k, got {num_frames}")
    latent_frames = 1 + (num_frames - 1) // vae_temporal_compression
    latent_h = video_height // vae_spatial_compression
    latent_w = video_width // vae_spatial_compression
    video_duration = float(num_frames) / float(video_fps)
    audio_latent_fps = (
        float(audio_sample_rate) / float(audio_hop_length) / float(audio_latent_downsample)
    )
    audio_frames = round(video_duration * audio_latent_fps)
    return (
        [batch_size, latent_frames, latent_channels, latent_h, latent_w],
        [batch_size, audio_frames, latent_channels],
    )


def _add_noise(original: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    sigma = sigma.to(device=original.device, dtype=original.dtype)
    if sigma.dim() == 1:
        sigma = sigma.reshape(-1, *[1] * (original.dim() - 1))
    elif sigma.dim() == 2:
        sigma = sigma.reshape(*sigma.shape, *[1] * (original.dim() - 2))
    return (1 - sigma) * original + sigma * noise


@torch.no_grad()
def _decode_benchmark_sample(
    video_vae: Any, audio_vae: Any, video_latent: torch.Tensor, audio_latent: torch.Tensor
):
    video_pixel = video_vae.decode_to_pixel(video_latent)
    audio_waveform = (
        audio_vae.decode_to_waveform(audio_latent) if audio_latent is not None else None
    )
    video_uint8 = video_pixel[0]
    if video_uint8.shape[0] == 3:
        video_uint8 = video_uint8.permute(1, 0, 2, 3)
    video_uint8 = video_uint8.permute(0, 2, 3, 1)
    video_uint8 = (video_uint8.clamp(0, 1) * 255).cpu().to(torch.uint8).contiguous()
    return video_uint8, audio_waveform
