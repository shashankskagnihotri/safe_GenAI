"""Single-trajectory CogVideoX-5b RIFLEx benchmark adapter."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import torch
from diffusers import CogVideoXDPMScheduler
from diffusers.utils.torch_utils import randn_tensor

from .base import SchedulerStepResult
from .cogvideox_adapter import CogVideoXAdapter
from ..generation.cogvideox_riflex import (
    UPSTREAM_REPOSITORY,
    UPSTREAM_REVISION,
    prepare_cogvideox_1_0_riflex_rotary,
)
from ..generation.rife_interpolation import (
    RifeMidpointInterpolator,
    rife_2x_and_crop_exact,
)


def _find_mapping_key(value: Any, key: str, *, depth: int = 0) -> Any:
    if depth > 6:
        return None
    if isinstance(value, Mapping):
        if key in value:
            return value[key]
        for child in value.values():
            found = _find_mapping_key(child, key, depth=depth + 1)
            if found is not None:
                return found
    if dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            found = _find_mapping_key(
                getattr(value, field.name), key, depth=depth + 1
            )
            if found is not None:
                return found
    return None


class CogVideoXRIFLExAdapter(CogVideoXAdapter):
    """Generate one 121-frame CogVideoX trajectory, then RIFE it to 240."""

    def _riflex_protocol(self) -> dict[str, Any]:
        protocol = None
        for name in (
            "model_config",
            "_model_config",
            "config",
            "_config",
            "cfg",
            "_cfg",
            "spec",
            "_spec",
            "model_spec",
        ):
            if hasattr(self, name):
                protocol = _find_mapping_key(
                    getattr(self, name), "cogvideox_riflex_protocol"
                )
                if protocol is not None:
                    break
        if protocol is None:
            protocol = _find_mapping_key(
                vars(self), "cogvideox_riflex_protocol"
            )
        if not isinstance(protocol, Mapping):
            raise ValueError("Missing cogvideox_riflex_protocol mapping")
        result = dict(protocol)
        expected = {
            "native_num_frames": 121,
            "latent_num_frames": 31,
            "native_fps": 8,
            "final_num_frames": 240,
            "final_fps": 16,
            "height": 480,
            "width": 720,
            "k": 2,
            "N_k": 20,
            "upstream_revision": UPSTREAM_REVISION,
        }
        for key, expected_value in expected.items():
            if result.get(key) != expected_value:
                raise ValueError(
                    f"Invalid RIFLEx contract {key}={result.get(key)!r}; "
                    f"expected {expected_value!r}"
                )
        if float(result.get("duration_seconds", 0.0)) != 15.0:
            raise ValueError("RIFLEx duration_seconds must be 15.0")
        if result["latent_num_frames"] <= result["N_k"]:
            raise ValueError("RIFLEx must activate only beyond intrinsic period N_k")
        return result

    def _primary_pipeline(self) -> Any:
        for name in (
            "pipeline",
            "_pipeline",
            "pipe",
            "_pipe",
            "t2v_pipeline",
            "_t2v_pipeline",
        ):
            pipeline = getattr(self, name, None)
            if (
                pipeline is not None
                and hasattr(pipeline, "transformer")
                and hasattr(pipeline, "scheduler")
                and hasattr(pipeline, "vae")
            ):
                return pipeline
        raise RuntimeError("Could not locate the loaded CogVideoX T2V pipeline")

    def load(self, *args: Any, **kwargs: Any) -> Any:
        if self._temporal_protocol() is not None:
            raise ValueError(
                "RIFLEx and segmented CogVideoX temporal protocols are mutually exclusive"
            )
        result = super().load(*args, **kwargs)
        pipeline = self._primary_pipeline()
        pipeline.scheduler = CogVideoXDPMScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing="trailing"
        )
        pipeline.vae.enable_slicing()
        pipeline.vae.enable_tiling()
        self._riflex_decode_provenance: dict[str, Any] | None = None
        return result

    @staticmethod
    def _generator_snapshot(generator: Any) -> Any:
        if isinstance(generator, (list, tuple)):
            return [item.get_state() for item in generator]
        return generator.get_state()

    @staticmethod
    def _restore_generator(generator: Any, snapshot: Any) -> None:
        if isinstance(generator, (list, tuple)):
            for item, state in zip(generator, snapshot, strict=True):
                item.set_state(state)
            return
        generator.set_state(snapshot)

    def prepare_initial_latents(
        self,
        prompt: str,
        batch_size: int,
        generator: torch.Generator | None,
        **generation_kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        protocol = self._riflex_protocol()
        requested_frames = int(generation_kwargs.get("num_frames", 0))
        if requested_frames != protocol["final_num_frames"]:
            raise ValueError(
                f"Campaign requested {requested_frames} frames; exact contract "
                f"requires {protocol['final_num_frames']}"
            )
        for key in ("height", "width"):
            if int(generation_kwargs[key]) != protocol[key]:
                raise ValueError(
                    f"Campaign {key}={generation_kwargs[key]}; exact contract "
                    f"requires {protocol[key]}"
                )
        if generator is None:
            raise ValueError("Exact RIFLEx generation requires an explicit generator")
        if batch_size != 1:
            raise ValueError("Exact CogVideoX RIFLEx qualification requires batch size one")

        snapshot = self._generator_snapshot(generator)
        native_kwargs = dict(generation_kwargs)
        native_kwargs.update(num_frames=49, height=480, width=720)
        template, state = super().prepare_initial_latents(
            prompt,
            batch_size,
            generator,
            **native_kwargs,
        )
        self._restore_generator(generator, snapshot)

        if template.ndim != 5:
            raise TypeError("Expected five-dimensional CogVideoX latent state")
        if int(template.shape[1]) != 13 or int(template.shape[2]) != 16:
            raise ValueError(
                "49-frame CogVideoX template must have shape [B,13,16,H,W], got "
                f"shape={tuple(template.shape)}"
            )
        shape = list(template.shape)
        shape[1] = protocol["latent_num_frames"]
        pipeline = self._primary_pipeline()
        latents = randn_tensor(
            tuple(shape),
            generator=generator,
            device=template.device,
            dtype=template.dtype,
        )
        latents = latents * pipeline.scheduler.init_noise_sigma
        rotary = prepare_cogvideox_1_0_riflex_rotary(
            pipeline,
            height=protocol["height"],
            width=protocol["width"],
            latent_frames=protocol["latent_num_frames"],
            device=template.device,
            k=protocol["k"],
            L_test=protocol["latent_num_frames"],
        )
        state.extra.update(
            {
                "num_frames": protocol["native_num_frames"],
                "output_num_frames": protocol["native_num_frames"],
                "native_generated_num_frames": protocol["native_num_frames"],
                "native_latent_shape": list(latents.shape),
                "native_latent_shape_observed": True,
                "expected_native_latent_shape": list(latents.shape),
                "additional_latent_frames": 0,
                "height": protocol["height"],
                "width": protocol["width"],
                "image_rotary_emb": rotary,
                "cogvideox_riflex_protocol": {
                    "schema_version": 1,
                    "method": "single_trajectory_cogvideox_riflex_then_rife",
                    "upstream_repository": UPSTREAM_REPOSITORY,
                    **dict(protocol),
                },
                "riflex_dpm_old_pred_original_sample": None,
                "riflex_dpm_previous_timestep": None,
            }
        )
        self._last_temporal_provenance = None
        return latents, state

    def scheduler_step(
        self,
        model_prediction: torch.Tensor,
        timestep: Any,
        latents: torch.Tensor,
        state: Any,
        generator: torch.Generator | None = None,
    ) -> SchedulerStepResult:
        """Apply the official stateful CogVideoX DPM-solver++ recurrence."""

        self._require_loaded()
        pipeline = self._primary_pipeline()
        old_pred_original_sample = state.extra.get(
            "riflex_dpm_old_pred_original_sample"
        )
        timestep_back = state.extra.get("riflex_dpm_previous_timestep")
        output = pipeline.scheduler.step(
            model_output=model_prediction,
            old_pred_original_sample=old_pred_original_sample,
            timestep=timestep,
            timestep_back=timestep_back,
            sample=latents,
            generator=generator,
            return_dict=False,
        )
        if not isinstance(output, tuple) or len(output) != 2:
            raise TypeError(
                "CogVideoXDPMScheduler must return "
                "(prev_sample, pred_original_sample)"
            )
        next_latents, pred_original_sample = output
        if not isinstance(next_latents, torch.Tensor) or not isinstance(
            pred_original_sample, torch.Tensor
        ):
            raise TypeError("CogVideoXDPMScheduler returned non-tensor state")
        state.extra["riflex_dpm_old_pred_original_sample"] = pred_original_sample
        state.extra["riflex_dpm_previous_timestep"] = timestep
        dtype = state.extra.get("prompt_embeds_dtype")
        if dtype is not None:
            next_latents = next_latents.to(dtype)
        return SchedulerStepResult(latents=next_latents, state=state)

    def decode_latents(self, latents: torch.Tensor, state: Any) -> Any:
        protocol = self._riflex_protocol()
        if int(latents.shape[1]) != protocol["latent_num_frames"]:
            raise ValueError(
                f"Expected {protocol['latent_num_frames']} latent frames, got "
                f"shape={tuple(latents.shape)}"
            )
        native_videos = super().decode_latents(latents, state)
        if len(native_videos) != 1:
            raise ValueError(
                f"Exact diagnostic contract requires batch size one, got "
                f"{len(native_videos)}"
            )
        if len(native_videos[0]) != protocol["native_num_frames"]:
            raise ValueError(
                f"Decoded {len(native_videos[0])} native frames; expected "
                f"{protocol['native_num_frames']}"
            )

        interpolation = dict(protocol.get("interpolation", {}))
        midpoint = RifeMidpointInterpolator(
            device=latents.device,
            weights_path=interpolation.get("weights_path"),
            scale=float(interpolation.get("scale", 1.0)),
        )
        output, provenance = rife_2x_and_crop_exact(
            native_videos[0],
            midpoint_interpolator=midpoint,
            source_fps=protocol["native_fps"],
            output_fps=protocol["final_fps"],
            duration_seconds=float(protocol["duration_seconds"]),
            output_frame_count=protocol["final_num_frames"],
        )
        self._riflex_decode_provenance = {
            "protocol": dict(protocol),
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_revision": UPSTREAM_REVISION,
            "interpolation": dataclasses.asdict(provenance)
            if dataclasses.is_dataclass(provenance)
            else dict(provenance),
            "rife": midpoint.provenance(),
        }
        return [output]

    def conditioning_provenance(self) -> dict[str, Any]:
        provenance = dict(super().conditioning_provenance())
        provenance["cogvideox_riflex"] = {
            "protocol": self._riflex_protocol(),
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_revision": UPSTREAM_REVISION,
            "decode": self._riflex_decode_provenance,
        }
        return provenance
