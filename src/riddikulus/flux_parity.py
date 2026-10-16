"""One native FLUX.2 prediction alongside the first real denoising step.

The caller owns placement and persistence. Immediately after computing the raw
base prediction, before steering or scheduler.step(), call::

    diagnostics = check_flux_native_prediction(
        adapter, latents,
        {"base": base_condition, "state": state, "branches": branch_conditions},
        timestep, base_prediction,
    )

``base`` and optional ``branches`` values are the actual PromptCondition objects
used by inference, not prompt strings or rebuilt encodings. ``state`` is the
actual AdapterState carrying latent image IDs and output dimensions. Branch keys
are role names such as neutral, unsafe, and safe. Do not re-encode for this check.
Call once at step_index == 0, in the same autocast/inference context as the base
prediction. No second base prediction, decoding, scheduler step, or random draw
is performed here. CPU/CUDA default RNG states are restored even on exceptions.

Success returns a JSON-serializable dict. Persist FluxNativeParityError.diagnostics
on failure, then propagate the failure; do not silently relax equality or retry
a deterministic discrepancy. Native forward exceptions retain their own type.
This is prediction parity for supplied latents/conditioning, not independent
end-to-end validation of the checkpoint, prompt encoder, initial noise, or VAE.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from importlib.metadata import version
from typing import Any

import torch

from .adapters.base import AdapterState, PromptCondition


class FluxNativeParityError(RuntimeError):
    """A real-input contract or numerical parity failure with saveable evidence."""

    def __init__(self, reason: str, diagnostics: dict[str, Any]) -> None:
        diagnostics["status"] = "failed"
        diagnostics["reason"] = reason
        self.diagnostics = diagnostics
        super().__init__(
            "FLUX.2 native prediction parity failed: "
            + reason
            + "; diagnostics="
            + json.dumps(diagnostics, sort_keys=True, allow_nan=False)
        )


def _digest(value: Any) -> str:
    # Configs can contain torch.dtype; only the digest, never a raw load config,
    # is emitted. Prompt text, credentials, environment, and tensor data stay out.
    encoded = json.dumps(value, sort_keys=True, default=str, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _comparison(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    # Double-precision CPU reductions are diagnostic only; never alter inference.
    a = actual.detach().to(device="cpu", dtype=torch.float64)
    b = reference.detach().to(device="cpu", dtype=torch.float64)
    difference = a - b
    absolute = difference.abs()
    reference_l2 = float(b.norm().item())
    difference_l2 = float(difference.norm().item())
    return {
        "exact_equal": bool(torch.equal(actual, reference)),
        "element_count": actual.numel(),
        "unequal_element_count": int(torch.count_nonzero(a != b).item()),
        "max_abs_error": float(absolute.max().item()),
        "mean_abs_error": float(absolute.mean().item()),
        "rmse": float(difference.square().mean().sqrt().item()),
        "reference_l2": reference_l2,
        "difference_l2": difference_l2,
        "relative_l2_error": (
            difference_l2 / reference_l2
            if reference_l2 != 0.0
            else (0.0 if difference_l2 == 0.0 else None)
        ),
        "max_relative_error_reference_floor_1e-12": float(
            (absolute / b.abs().clamp_min(1.0e-12)).max().item()
        ),
    }


@torch.no_grad()
def check_flux_native_prediction(
    adapter: Any,
    latents: torch.Tensor,
    conditioning: Mapping[str, Any],
    timestep: torch.Tensor,
    base_prediction: torch.Tensor,
) -> dict[str, Any]:
    """Compare one existing real base prediction to the native transformer API.

    ``conditioning = {"base": PromptCondition, "state": AdapterState,
    "branches": Mapping[str, PromptCondition]}``; branches may be omitted.
    Native text-to-image geometry is derived from state.height/width. The 1024
    square, single-image campaign has packed shape [1, 4096, 128], with features
    on axis 2, not axis 1. Other geometry is reported explicitly, never reshaped.
    """
    diagnostics: dict[str, Any] = {
        "schema_version": 1,
        "status": "checking",
        "scope": "first_real_step_supplied_conditioning_native_transformer",
        "extra_transformer_predictions": 0,
        "comparison_policy": {
            "rtol": 0.0,
            "atol": 0.0,
            "require_same_dtype": True,
            "rationale": (
                "Same loaded transformer, inputs, device, and execution context; "
                "native return_dict=False versus adapter .sample only changes "
                "the output container. No hardware tolerance is assumed."
            ),
        },
        "inputs": {},
        "checked_invariants": {},
    }

    def require(ok: bool, reason: str) -> None:
        if not ok:
            raise FluxNativeParityError(reason, diagnostics)

    def record_tensor(name: str, value: Any) -> torch.Tensor:
        require(isinstance(value, torch.Tensor), f"{name} must be a Tensor.")
        diagnostics["inputs"][name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "stride": list(value.stride()),
        }
        require(value.device.type != "meta", f"{name} is an unmaterialized meta tensor.")
        require(value.numel() > 0, f"{name} is empty.")
        finite = bool(torch.isfinite(value).all().item())
        diagnostics["inputs"][name]["finite"] = finite
        require(finite, f"{name} contains non-finite values.")
        return value

    require(getattr(adapter, "adapter_name", None) == "flux2", "Expected registered Flux2Adapter.")
    require(bool(getattr(adapter, "loaded", False)), "Adapter is not loaded.")
    pipeline = adapter.pipeline
    require(pipeline is not None, "Adapter has no pipeline.")
    transformer = pipeline.transformer
    scheduler = pipeline.scheduler
    require(not transformer.training, "Transformer must already be in evaluation mode.")
    require(isinstance(conditioning, Mapping), "conditioning must contain base and state.")
    base = conditioning.get("base")
    state = conditioning.get("state")
    branches = conditioning.get("branches", {})
    require(isinstance(base, PromptCondition), "conditioning['base'] must be PromptCondition.")
    require(isinstance(state, AdapterState), "conditioning['state'] must be AdapterState.")
    require(isinstance(branches, Mapping), "conditioning['branches'] must be a role mapping.")

    latents = record_tensor("latents", latents)
    base_prediction = record_tensor("base_prediction", base_prediction)
    require(latents.device.type in {"cpu", "cuda"}, "This parity guard supports CPU/CUDA RNG only.")
    require(latents.ndim == 3 and latents.is_floating_point(), "Expected packed floating [B,T,D] latents.")
    batch, image_tokens, channels = latents.shape
    require(int(adapter.latent_feature_dim) % 3 == 2, "Packed FLUX.2 features must be on the last axis.")
    require(channels == int(transformer.config.in_channels) == 128, "Expected 128 packed latent features.")
    require(tuple(base_prediction.shape) == tuple(latents.shape), "Base prediction must retain packed latent shape.")
    require(base_prediction.device == latents.device, "Base prediction moved off the latent device.")
    require(
        adapter.dtype == transformer.dtype == latents.dtype,
        "Adapter/transformer/latent dtype mismatch: adapter casts hidden states and timestep "
        "to adapter.dtype, whereas native casts hidden states to transformer.dtype and "
        "timestep to latents.dtype. This configuration needs an explicit native audit.",
    )
    height = state.extra.get("height")
    width = state.extra.get("width")
    require(isinstance(height, int) and isinstance(width, int), "State must carry integer height and width.")
    multiple = int(pipeline.vae_scale_factor) * 2
    packed_height, packed_width = height // multiple, width // multiple
    require(packed_height > 0 and packed_width > 0, "Output dimensions are below one packed VAE patch.")
    require(image_tokens == packed_height * packed_width, "Packed token count disagrees with native spatial geometry.")
    img_ids = state.extra.get("latent_image_ids")
    if img_ids is None:
        img_ids = state.extra.get("latent_ids")
    img_ids = record_tensor("img_ids", img_ids)
    require(tuple(img_ids.shape) == (batch, image_tokens, 4), "Image IDs must have native shape [B,T,4].")
    require(img_ids.dtype == torch.int64 and img_ids.device == latents.device, "Image IDs must retain native int64/device.")
    positions = torch.arange(image_tokens, device=img_ids.device)
    require(
        bool((img_ids[..., 0] == 0).all())
        and bool((img_ids[..., 3] == 0).all())
        and bool((img_ids[..., 1] == positions // packed_width).all())
        and bool((img_ids[..., 2] == positions % packed_width).all()),
        "Image IDs are not native row-major (time=0, height, width, layer=0) coordinates.",
    )
    require(
        state.extra.get("image_latents") is None and state.extra.get("image_latent_ids") is None,
        "Reference-image conditioning is outside this text-to-image parity contract.",
    )

    config = adapter.config
    guidance_scale = config.get("guidance_scale", 4.0)
    require(
        isinstance(guidance_scale, (int, float)) and not isinstance(guidance_scale, bool)
        and guidance_scale == 4.0,
        "Campaign requires embedded distilled guidance_scale=4.0, not external CFG.",
    )
    require(not config.get("caption_upsample_temperature"), "Caption upsampling is not authorized for this campaign.")
    require(not config.get("attention_kwargs"), "Non-default attention kwargs require a separate native contract.")
    require(bool(transformer.config.guidance_embeds), "Loaded transformer lacks distilled guidance embeddings.")
    configured_layers = config.get("text_encoder_out_layers", (10, 20, 30))
    require(configured_layers is not None and tuple(configured_layers) == (10, 20, 30), "Expected native Mistral layers (10,20,30).")
    sequence_length = config.get("max_sequence_length")
    if sequence_length is None:
        sequence_length = 512
    joint_dimension = int(transformer.config.joint_attention_dim)

    condition_reports: dict[str, Any] = {}
    all_conditions = {"base": base}
    require("base" not in branches, "branches must not shadow the actual base condition.")
    all_conditions.update(branches)
    for role, condition in all_conditions.items():
        require(isinstance(role, str), "Condition branch roles must be strings.")
        require(isinstance(condition, PromptCondition), "Every branch must carry its actual PromptCondition.")
        embeds = record_tensor(f"{role}.prompt_embeds", condition.data.get("prompt_embeds"))
        txt_ids = record_tensor(f"{role}.text_ids", condition.data.get("text_ids"))
        require(tuple(embeds.shape) == (batch, sequence_length, joint_dimension), f"{role}: non-native embedding shape.")
        require(embeds.device == latents.device and embeds.dtype == latents.dtype, f"{role}: conditioning dtype/device mismatch.")
        require(tuple(txt_ids.shape) == (batch, sequence_length, 4), f"{role}: text IDs must be [B,L,4].")
        require(txt_ids.dtype == torch.int64 and txt_ids.device == latents.device, f"{role}: text IDs must retain native int64/device.")
        require(
            bool((txt_ids[..., :3] == 0).all())
            and bool((txt_ids[..., 3] == torch.arange(sequence_length, device=txt_ids.device)).all()),
            f"{role}: text IDs are not native (0,0,0,sequence_index) coordinates.",
        )
        require(condition.data.get("negative_prompt_embeds") is None, f"{role}: external CFG embeddings are not native FLUX.2 conditioning.")
        condition_reports[role] = {
            "prompt_sha256": hashlib.sha256(condition.prompt.encode("utf-8")).hexdigest(),
            "guidance_scale": float(guidance_scale),
            "same_latents_timestep_image_ids": "required_by_shared_adapter_call_contract",
        }

    timestep = record_tensor("scheduler_timestep", timestep)
    require(timestep.numel() == 1 and timestep.ndim <= 1, "Pass the actual scalar scheduler timestep, not a step index.")
    schedule_before = scheduler.timesteps.detach().clone()
    sigmas_before = scheduler.sigmas.detach().clone()
    step_index_before = scheduler.step_index
    begin_index_before = scheduler.begin_index
    require(step_index_before in (None, 0) and begin_index_before in (None, 0), "Call parity before the first scheduler step.")
    require(schedule_before.numel() > 0 and sigmas_before.numel() >= 2, "Scheduler has no initialized denoising schedule.")
    require(float(timestep.item()) == float(schedule_before[0].item()), "Timestep is not the first real scheduled timestep.")
    native_timestep = timestep.expand(batch).to(latents.dtype) / 1000
    guidance = torch.full([1], float(guidance_scale), device=latents.device, dtype=torch.float32).expand(batch)
    hidden_states = latents.to(transformer.dtype)
    record_tensor("native_timestep_div_1000", native_timestep)
    record_tensor("native_guidance", guidance)
    bn = pipeline.vae.bn
    require(bn.running_mean.numel() == channels and bn.running_var.numel() == channels, "VAE BN buffers must cover all 128 patchified features.")

    identity = {
        "model_id": str(adapter.model_id),
        "revision": config.get("revision"),
        "adapter_class": f"{type(adapter).__module__}.{type(adapter).__name__}",
        "pipeline_class": f"{type(pipeline).__module__}.{type(pipeline).__name__}",
        "transformer_class": f"{type(transformer).__module__}.{type(transformer).__name__}",
        "transformer_config_sha256": _digest(dict(transformer.config)),
        "scheduler_config_sha256": _digest(dict(scheduler.config)),
        "vae_config_sha256": _digest(dict(pipeline.vae.config)),
        "adapter_dtype": str(adapter.dtype),
        "guidance_scale": float(guidance_scale),
        "max_sequence_length": sequence_length,
        "text_encoder_out_layers": [10, 20, 30],
        "caption_upsampling": False,
        "height": height,
        "width": width,
        "schedule_sha256": _digest(schedule_before.cpu().tolist()),
        "sigmas_sha256": _digest(sigmas_before.cpu().tolist()),
    }
    diagnostics.update({
        "config_identity": identity,
        "config_identity_sha256": _digest(identity),
        "conditioning": condition_reports,
        "runtime": {
            "torch": torch.__version__,
            "diffusers": version("diffusers"),
            "transformers": version("transformers"),
            "cuda_runtime": torch.version.cuda,
            "device": str(latents.device),
            "device_name": torch.cuda.get_device_name(latents.device) if latents.is_cuda else "cpu",
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
        "timestep": {
            "scheduler": float(timestep.item()),
            "transformer": native_timestep.detach().cpu().tolist(),
            "sigma": float(sigmas_before[0].item()),
            "next_sigma": float(sigmas_before[1].item()),
            "step_index_before": step_index_before,
            "begin_index_before": begin_index_before,
        },
        "source_audit_not_independently_recomputed": {
            "text_encoding": "native encode_prompt; Mistral layers (10,20,30); apply_chat_template(add_generation_prompt=False)",
            "caption_upsampling": "adapter calls encode_prompt, not upsample_prompt or pipeline.__call__",
            "vae_decode": "unpack_with_ids; z * sqrt(running_var + batch_norm_eps) + running_mean; unpatchify; decode",
            "vae_batch_norm_eps": float(pipeline.vae.config.batch_norm_eps),
            "branch_guidance": "one shared Flux2Adapter call path uses embedded guidance_scale=4 for base/concept/neutral",
            "steering_sign": "x0=z-sigma*v; delta_v=lambda*(v_safe-v_unsafe) implies delta_x0=lambda*(x0_safe-x0_unsafe)",
            "clean_space_conversion": "an independently constructed delta_x0 converts to delta_v=-delta_x0/sigma for sigma>0",
            "limitation": "No branch-prediction, prompt-encoding, initial-noise, steering-output, or VAE-decode parity is claimed.",
        },
    })
    diagnostics["checked_invariants"].update({
        "packed_feature_axis": 2,
        "native_shape": [batch, packed_height * packed_width, channels],
        "matches_1024_single_image_shape": list(latents.shape) == [1, 4096, 128],
        "native_image_and_text_ids": True,
        "conditioning_shape_dtype_device_consistent": True,
        "embedded_guidance_scale": 4.0,
        "external_cfg_applied_by_helper": False,
        "native_timestep_divisor": 1000,
        "reference_image_tokens": 0,
        "scheduler_calls_by_helper": 0,
    })

    cuda_devices = [latents.device.index] if latents.is_cuda else []
    with torch.random.fork_rng(devices=cuda_devices):
        cpu_rng_before = torch.get_rng_state()
        cuda_rng_before = torch.cuda.get_rng_state(latents.device) if latents.is_cuda else None
        # Deliberately bypass adapter._call_transformer and its argument filtering.
        # This is the installed Flux2Pipeline text-to-image denoising invocation.
        diagnostics["extra_transformer_predictions"] = 1
        native_prediction = transformer(
            hidden_states=hidden_states,
            timestep=native_timestep,
            guidance=guidance,
            encoder_hidden_states=base.data["prompt_embeds"],
            txt_ids=base.data["text_ids"],
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        native_prediction = native_prediction[:, : latents.size(1)]
        cpu_rng_unchanged = torch.equal(cpu_rng_before, torch.get_rng_state())
        cuda_rng_unchanged = (
            torch.equal(cuda_rng_before, torch.cuda.get_rng_state(latents.device))
            if latents.is_cuda else True
        )

    scheduler_unchanged = (
        scheduler.step_index == step_index_before
        and scheduler.begin_index == begin_index_before
        and torch.equal(scheduler.timesteps, schedule_before)
        and torch.equal(scheduler.sigmas, sigmas_before)
    )
    diagnostics["checked_invariants"].update({
        "scheduler_unchanged": bool(scheduler_unchanged),
        "cpu_rng_not_consumed": bool(cpu_rng_unchanged),
        "cuda_rng_not_consumed": bool(cuda_rng_unchanged),
        "cpu_cuda_default_rng_restored": True,
    })
    native_prediction = record_tensor("native_prediction", native_prediction)
    require(native_prediction.shape == base_prediction.shape, "Native and adapter prediction shapes disagree.")
    require(native_prediction.dtype == base_prediction.dtype, "Native and adapter prediction dtypes disagree.")
    require(native_prediction.device == base_prediction.device, "Native and adapter prediction devices disagree.")
    diagnostics["comparison"] = _comparison(base_prediction, native_prediction)
    require(bool(scheduler_unchanged), "Scheduler state changed during the extra transformer prediction.")
    require(bool(cpu_rng_unchanged and cuda_rng_unchanged), "Native prediction unexpectedly consumed RNG; saved RNG state was restored.")
    require(
        diagnostics["comparison"]["exact_equal"],
        "Native prediction differs from the raw adapter base prediction. Stop before steering "
        "or scheduler.step; inspect argument/context/backend differences. Any tolerance "
        "requires a documented code/hardware explanation, not a silent relaxation.",
    )
    diagnostics["status"] = "passed"
    return diagnostics
