"""Native denoiser parity on the first real state of the five-model campaign.

One additional ordinary prediction is evaluated using the installed pipeline's
denoiser invocation, bypassing the adapter's argument filtering and CFG wrapper.
This is one forward for FLUX/SD3.5 and two forwards for Cosmos/Qwen true CFG.
It never encodes a prompt, advances a scheduler, decodes an image or draws noise.
The supplied condition is the actual generation condition, not reconstructed
text. Equality verifies the prediction API; it does not independently verify
prompt encoding, the checkpoint, initial noise, or decoding.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import torch


class NativePredictionParityError(RuntimeError):
    """Parity failure carrying a JSON-serializable record to persist."""

    def __init__(self, reason: str, diagnostics: dict[str, Any]) -> None:
        diagnostics.update(status="failed", reason=reason)
        self.diagnostics = diagnostics
        super().__init__(f"Native first-prediction parity failed: {reason}")


def _tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    contiguous = tensor.detach().cpu().contiguous()
    identity = json.dumps({"shape": list(tensor.shape), "dtype": str(tensor.dtype)}, sort_keys=True).encode()
    digest = hashlib.sha256(identity)
    digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "device": str(tensor.device), "finite": bool(torch.isfinite(tensor).all()),
            "sha256": digest.hexdigest()}


def _source_record(cls: type) -> dict[str, str]:
    filename = inspect.getsourcefile(cls)
    if filename is None or not Path(filename).is_file():
        raise RuntimeError(f"Loaded class has no inspectable source: {cls.__module__}.{cls.__name__}")
    return {"class": f"{cls.__module__}.{cls.__name__}", "path": filename,
            "sha256": hashlib.sha256(Path(filename).read_bytes()).hexdigest()}


def _compare(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    left = actual.detach().to(device="cpu", dtype=torch.float64)
    right = reference.detach().to(device="cpu", dtype=torch.float64)
    delta = left - right
    norm = float(right.norm())
    return {"exact_equal": bool(torch.equal(actual, reference)),
            "unequal_element_count": int(torch.count_nonzero(left != right)),
            "max_absolute_error": float(delta.abs().max()),
            "rms_error": float(delta.square().mean().sqrt()),
            "relative_l2_error": float(delta.norm()) / norm if norm else (0.0 if not bool(delta.any()) else None)}


def _flux1_native(adapter, latents, timestep, data, state, guidance):
    pipeline = adapter.pipeline
    if data.get("negative_prompt_embeds") is not None:
        raise RuntimeError("FLUX.1 campaign uses distilled guidance; external true CFG was not specified")
    if not bool(pipeline.transformer.config.guidance_embeds):
        raise RuntimeError("Requested FLUX.1-dev transformer lacks native guidance embeddings")
    img_ids = state.extra.get("latent_image_ids", state.extra.get("latent_ids"))
    if img_ids is None:
        raise RuntimeError("FLUX.1 native parity requires the actual latent image IDs")
    native_timestep = timestep.expand(latents.shape[0]).to(latents.dtype) / 1000
    native_guidance = torch.full([1], guidance, device=latents.device, dtype=torch.float32).expand(latents.shape[0])
    with pipeline.transformer.cache_context("cond"):
        output = pipeline.transformer(
            hidden_states=latents,
            timestep=native_timestep,
            guidance=native_guidance,
            pooled_projections=data["pooled_prompt_embeds"],
            encoder_hidden_states=data["prompt_embeds"],
            txt_ids=data["text_ids"],
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
    return output, 1, {"guidance": "distilled_embedding", "timestep_divisor": 1000}


def _sd35_native(adapter, latents, timestep, data, state, guidance):
    del state
    pipeline = adapter.pipeline
    use_cfg = guidance > 1.0
    if use_cfg and (data.get("negative_prompt_embeds") is None or data.get("negative_pooled_prompt_embeds") is None):
        raise RuntimeError("SD3.5 native true CFG requires both negative conditioning tensors")
    model_input = torch.cat([latents] * 2) if use_cfg else latents
    prompt_embeds = data["prompt_embeds"]
    pooled_embeds = data["pooled_prompt_embeds"]
    if use_cfg:
        prompt_embeds = torch.cat([data["negative_prompt_embeds"], prompt_embeds])
        pooled_embeds = torch.cat([data["negative_pooled_prompt_embeds"], pooled_embeds])
    output = pipeline.transformer(
        hidden_states=model_input,
        timestep=timestep.expand(model_input.shape[0]),
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_embeds,
        joint_attention_kwargs=None,
        return_dict=False,
    )[0]
    if use_cfg:
        negative, positive = output.chunk(2)
        output = negative + guidance * (positive - negative)
    return output, 1, {"guidance": "batched_true_cfg" if use_cfg else "conditional_only",
                       "cfg_scale": guidance, "prediction_normalization": "none", "timestep_divisor": 1}


def _cosmos_native(adapter, latents, timestep, data, state, guidance):
    pipeline = adapter.pipeline
    if not bool(adapter.config.get("offline_research_disable_guardrails", False)):
        raise RuntimeError("Cosmos3 campaign requires explicitly disabled offline-research guardrails")
    if getattr(pipeline, "safety_checker", None) is not None:
        raise RuntimeError("Cosmos3 still has a loaded safety checker")
    if adapter.config.get("load_kwargs", {}).get("enable_safety_checker") is not False:
        raise RuntimeError("Cosmos3 load receipt does not prove enable_safety_checker=False")
    if state.extra.get("sound_latents") is not None or state.extra.get("action_latents") is not None:
        raise RuntimeError("Cosmos3 parity contract is text-to-image without sound/action conditioning")
    mask = state.extra["vision_condition_mask"]
    indexes = torch.nonzero(mask[:, 0, 0] > 0, as_tuple=False).flatten()
    frame_indexes = [int(index.item()) for index in indexes]
    if frame_indexes:
        raise RuntimeError("Cosmos3 text-to-image parity received reference-frame conditioning")
    tokens = latents.to(device=adapter.device, dtype=pipeline.transformer.dtype)

    def forward(text):
        vision = pipeline._prepare_vision_segment(
            input_vision_tokens=latents, has_image_condition=False,
            mrope_offset=text["vision_start_temporal_offset"], vision_fps=state.extra["fps_vision"],
            curr=text["und_len"], device=adapter.device, condition_frame_indexes=frame_indexes,
        )
        packed = {**text, **vision,
                  "position_ids": torch.cat([text["text_mrope_ids"], vision["vision_mrope_ids"]], dim=1),
                  "sequence_length": text["und_len"] + vision["num_vision_tokens"]}
        result = pipeline.transformer(
            input_ids=packed["input_ids"], text_indexes=packed["text_indexes"],
            position_ids=packed["position_ids"], und_len=packed["und_len"],
            sequence_length=packed["sequence_length"], vision_tokens=[tokens],
            vision_token_shapes=packed["vision_token_shapes"],
            vision_sequence_indexes=packed["vision_sequence_indexes"],
            vision_mse_loss_indexes=packed["vision_mse_loss_indexes"],
            vision_timesteps=torch.full((vision["num_noisy_vision_tokens"],), timestep.item(), device=adapter.device),
            vision_noisy_frame_indexes=packed["vision_noisy_frame_indexes"],
            sound_tokens=None, sound_token_shapes=None, sound_sequence_indexes=None,
            sound_mse_loss_indexes=None, sound_timesteps=None, sound_noisy_frame_indexes=None,
            action_tokens=None, action_token_shapes=None, action_sequence_indexes=None,
            action_mse_loss_indexes=None, action_timesteps=None, action_noisy_frame_indexes=None,
            action_domain_ids=None,
        )
        velocity, _, _ = pipeline._mask_velocity_predictions(
            result[0], result[1], vision_condition_mask=[mask], sound_condition_mask=None,
            preds_action=result[2], action_condition_mask=None, raw_action_dim=None,
        )
        return velocity

    positive = forward(data["cond_text_segment"])
    count = 1
    if guidance != 1.0:
        negative = forward(data["uncond_text_segment"])
        positive = negative + guidance * (positive - negative)
        count += 1
    return positive, count, {"guidance": "separate_true_cfg" if count == 2 else "conditional_only",
                             "cfg_scale": guidance, "prediction_normalization": "none",
                             "timestep_divisor": 1, "guardrails_disabled": True,
                             "unconditional_cache_used_by_native_check": False}




@torch.no_grad()
def verify_first_prediction(adapter, latents, timestep, condition, state, prediction, generation_config) -> dict[str, Any]:
    """Verify native prediction parity before the real step-zero scheduler update.

    Call in the same inference/autocast context as the already computed ordinary
    prediction. A mismatch raises NativePredictionParityError with diagnostics.
    The initial policy is bitwise equality because the calls use the same loaded
    model, state, conditioning, backend, dtype and first-step cache mode. Any
    relaxation requires an explicit, evidenced numerical explanation.
    """
    if adapter.adapter_name == "flux3":
        hook = getattr(adapter, "verify_native_prediction", None)
        if not callable(hook):
            raise RuntimeError("FLUX3 adapter lacks its independent native parity implementation")
        record = hook(latents, timestep, condition, state, prediction, generation_config)
        if not isinstance(record, dict) or record.get("status") not in {"passed", "PASS"}:
            raise RuntimeError("FLUX3 native prediction parity did not pass")
        return record
    record: dict[str, Any] = {
        "schema": "nailing-method.native-first-prediction.v1", "status": "checking",
        "scope": "first_actual_state_supplied_condition_native_denoiser", "extra_ordinary_predictions": 1,
        "scheduler_updates": 0, "prompt_encoder_calls": 0,
        "comparison_policy": {"rtol": 0.0, "atol": 0.0, "require_same_dtype": True,
                              "rationale": "Same loaded model, inputs, backend, dtype and native first-step cache mode"},
    }

    def require(ok, reason):
        if not ok:
            raise NativePredictionParityError(reason, record)

    require(bool(adapter.loaded) and adapter.pipeline is not None, "Adapter is not loaded")
    require(isinstance(timestep, torch.Tensor) and timestep.numel() == 1, "Actual scalar native timestep is required")
    pipeline = adapter.pipeline
    scheduler = pipeline.scheduler
    require(not pipeline.transformer.training, "Transformer must be in evaluation mode")
    require(scheduler.step_index in (None, 0), "Parity must occur before the first scheduler update")
    require(len(scheduler.timesteps) == int(generation_config["num_inference_steps"]), "Native schedule length differs from configuration")
    require(float(timestep.item()) == float(scheduler.timesteps[0].item()), "Supplied timestep is not native index zero")
    record["model"] = {"model_id": str(adapter.model_id), "revision": adapter.config.get("revision"),
                       "adapter": str(adapter.adapter_name), "pipeline_source": _source_record(type(pipeline)),
                       "transformer_source": _source_record(type(pipeline.transformer)),
                       "scheduler_source": _source_record(type(scheduler))}
    record["latents"] = _tensor_record(latents)
    record["ordinary_prediction"] = _tensor_record(prediction)
    require(record["latents"]["finite"] and record["ordinary_prediction"]["finite"], "Non-finite actual input or prediction")
    require(prediction.shape == latents.shape and prediction.device == latents.device, "Ordinary prediction layout/device mismatch")
    guidance = float(generation_config.get("true_cfg_scale", generation_config.get("guidance_scale", 1.0)))
    configured = float(adapter.config.get("manual_guidance_scale", adapter.config.get("guidance_scale", 1.0)))
    require(guidance == configured, "Generation and adapter guidance values disagree")
    before_schedule = scheduler.timesteps.clone()
    before_sigmas = scheduler.sigmas.clone()
    before_index = scheduler.step_index
    before_begin = scheduler.begin_index
    devices = list(range(torch.cuda.device_count())) if latents.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        cpu_rng = torch.get_rng_state()
        cuda_rng = [torch.cuda.get_rng_state(index) for index in devices]
        if adapter.adapter_name == "flux2":
            from .flux_parity import check_flux_native_prediction

            nested = check_flux_native_prediction(adapter, latents, {"base": condition, "state": state}, timestep, prediction)
            record.update(native_api_audit=nested, extra_transformer_forwards=1,
                          status="passed", guidance={"guidance": "distilled_embedding", "timestep_divisor": 1000})
            record["native_prediction"] = dict(record["ordinary_prediction"])
            record["comparison"] = nested["comparison"]
        else:
            handlers = {"flux": _flux1_native, "sd35": _sd35_native,
                        "cosmos3_t2i": _cosmos_native}
            require(adapter.adapter_name in handlers, f"Unsupported model adapter: {adapter.adapter_name}")
            reference, count, semantics = handlers[adapter.adapter_name](adapter, latents, timestep, condition.data, state, guidance)
            record.update(extra_transformer_forwards=count, guidance=semantics, native_prediction=_tensor_record(reference))
            require(record["native_prediction"]["finite"], "Native denoiser returned non-finite values")
            require(reference.shape == prediction.shape and reference.device == prediction.device, "Native prediction layout/device mismatch")
            require(reference.dtype == prediction.dtype, "Native prediction dtype differs from ordinary prediction")
            record["comparison"] = _compare(prediction, reference)
            require(record["comparison"]["exact_equal"], "Native and adapter predictions differ; inspect saved diagnostics before execution continues")
        require(torch.equal(cpu_rng, torch.get_rng_state()), "Native prediction unexpectedly consumed CPU RNG")
        require(all(torch.equal(saved, torch.cuda.get_rng_state(index)) for index, saved in zip(devices, cuda_rng)), "Native prediction unexpectedly consumed CUDA RNG")
    require(scheduler.step_index == before_index and scheduler.begin_index == before_begin
            and torch.equal(before_schedule, scheduler.timesteps) and torch.equal(before_sigmas, scheduler.sigmas),
            "Native denoiser parity changed scheduler state")
    require(_tensor_record(latents)["sha256"] == record["latents"]["sha256"], "Native denoiser mutated actual latent input")
    record.update(status="passed", scheduler_unchanged=True, latent_input_unchanged=True,
                  cpu_cuda_rng_preserved=True,
                  limitation="Prediction API parity only; prompt encoding, initial noise, checkpoint and VAE decode are not independently recomputed")
    return record
