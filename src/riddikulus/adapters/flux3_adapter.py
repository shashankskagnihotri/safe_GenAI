"""Native FLUX.3 Klein flow-map adapter, using the pinned BFL implementation.

The distilled network conditions on both ends of a sampling interval. Its
BF16 velocity updates the native FP32 latent; it is not a FLUX.2 pipeline.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
from pathlib import Path
import sys
from typing import Any

import torch

from .base import AdapterState, FrozenGeneratorAdapter, PromptCondition, SchedulerStepResult


MODEL_ID = "black-forest-labs/flux-3-image-klein"
MODEL_REVISION = "337337621b8526e120f2275d250242f4a6bedd75"
REFERENCE_COMMIT = "e53f4ea697fe45db8f38fca91a90f5623f7f8f37"


class NativeFlowMapScheduler:
    """Checkpointable wrapper around BFL's exact deterministic sampling update."""

    order = 1

    def __init__(self, alpha: float = 6.93):
        self._internal_dict = {"alpha": float(alpha)}
        self.sigmas = torch.empty(0, dtype=torch.float32)
        self.timesteps = torch.empty(0, dtype=torch.float32)
        self._step_index = None
        self._begin_index = 0
        self.num_inference_steps = None

    @property
    def config(self):
        return self._internal_dict

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls(alpha=float(config["alpha"]))

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index=0):
        self._begin_index = int(begin_index)

    def set_timesteps(self, num_inference_steps, device=None):
        from flux3.sampling import timesteps
        if int(num_inference_steps) != 16:
            raise ValueError("The registered distilled FLUX.3 profile has 16 native updates")
        self.num_inference_steps = int(num_inference_steps)
        # BFL creates the schedule on the CPU and takes Python scalar deltas.
        self.sigmas = timesteps(self.num_inference_steps, self.config["alpha"])
        self.timesteps = self.sigmas[:-1].clone()
        self._step_index = None
        self._begin_index = 0

    def step(self, model_output, timestep, sample, return_dict=False, generator=None):
        index = self._begin_index if self._step_index is None else self._step_index
        if index >= len(self.timesteps):
            raise RuntimeError("Flow-map scheduler has already completed")
        if float(timestep) != float(self.timesteps[index]):
            raise RuntimeError("Current timestep differs from the native flow-map schedule")
        if sample.dtype != torch.float32 or model_output.dtype != torch.bfloat16:
            raise RuntimeError("FLUX.3 requires native FP32 state and BF16 velocity")
        dt = (self.sigmas[index + 1] - self.sigmas[index]).item()
        # Preserve BFL's order and precision: BF16 increment, then FP32 sum.
        updated = sample + dt * model_output
        self._step_index = index + 1
        if return_dict:
            return {"prev_sample": updated}
        return (updated,)


def _identity(tensor):
    value = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(tuple(value.shape)).encode())
    h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class Flux3Adapter(FrozenGeneratorAdapter):
    adapter_name = "flux3"
    task_type = "text_to_image"
    pipeline_class_name = "flux3.pipeline.Pipeline"
    latent_feature_dim = -1
    native_scheduler_class = NativeFlowMapScheduler

    def load(self):
        if self.model_id != MODEL_ID or self.config.get("revision") != MODEL_REVISION:
            raise ValueError("Only the exact pinned distilled FLUX.3 Klein is registered")
        if self.dtype != torch.bfloat16 or self.device.type != "cuda":
            raise ValueError("The registered native FLUX.3 profile requires CUDA and BF16 weights")
        if self.config.get("reference_source_commit") != REFERENCE_COMMIT:
            raise RuntimeError("The FLUX.3 reference source commit is not pinned")
        if self.config.get("prompt_upsampling") is not False or self.config.get("safety") is not False:
            raise ValueError("This experiment requires raw prompts and explicit safety=False")
        if self.config.get("cpu_offload", False) or float(self.config.get("guidance_scale", 1.0)) != 1.0:
            raise ValueError("The registered distilled FLUX.3 profile has no offload or CFG")
        source = Path(self.config["reference_source_path"]).resolve()
        if not (source / "flux3" / "pipeline.py").is_file():
            raise FileNotFoundError("Pinned FLUX.3 reference implementation is unavailable")
        source_hashes = self.config.get("reference_source_sha256")
        if not isinstance(source_hashes, dict) or not source_hashes:
            raise RuntimeError("Pinned FLUX.3 source file hashes are required")
        for relative, expected in source_hashes.items():
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError("Reference source path escapes its sealed root")
            filename = source / relative_path
            if not filename.is_file() or hashlib.sha256(filename.read_bytes()).hexdigest() != expected:
                raise RuntimeError("Pinned FLUX.3 reference source changed: " + relative)
        sys.path.insert(0, str(source))
        native = importlib.import_module("flux3.pipeline")
        if Path(inspect.getfile(native)).resolve() != source / "flux3" / "pipeline.py":
            raise RuntimeError("An unexpected FLUX.3 package shadows the pinned reference")
        snapshot = Path(self.config["local_snapshot_path"]).resolve()
        if snapshot.name != MODEL_REVISION or not (snapshot / "model.safetensors").is_file():
            raise FileNotFoundError("Exact FLUX.3 checkpoint is not present")
        self.pipeline = native.Pipeline(
            {"distilled": str(snapshot)}, str(snapshot / "text_encoder"),
            str(snapshot / "ae.safetensors"), device=str(self.device), offload=False,
        )
        self.pipeline.transformer = self.pipeline.dits["distilled"]
        self.pipeline.vae = self.pipeline.autoencoder
        self.pipeline.scheduler = NativeFlowMapScheduler(float(self.config.get("schedule_alpha", 6.93)))
        for module in (self.pipeline.transformer, self.pipeline.text_encoder.model, self.pipeline.vae):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        if not self.pipeline.transformer.flow_map:
            raise RuntimeError("Loaded checkpoint lacks required destination-time conditioning")
        self.config["_study_placement"] = {
            "policy": "native", "visible_gpus": torch.cuda.device_count(),
            "quantization": None, "cpu_offload": False,
        }
        self.config["_study_guardrail_proof"] = {
            "schema": "riddikulus.guardrail-proof.v1", "model_id": self.model_id,
            "model_revision": MODEL_REVISION, "adapter": self.adapter_name,
            "application_guardrails_active": False, "external_inference_provider": None,
            "learned_weight_safety_behavior_removed": False,
            "reference_generate_wrapper_called": False, "safety": False,
            "upsampler_loaded": False, "prompt_upsampling": False,
            "safety_classifier_loaded": False, "reference_source_commit": REFERENCE_COMMIT,
        }
        self.loaded = True

    @torch.inference_mode()
    def prepare_prompt(self, prompt):
        self._require_loaded()
        tokens, ids = self.pipeline.text_encoder.encode(prompt)
        return PromptCondition(prompt=prompt, data={"prompt_embeds": tokens, "text_ids": ids})

    def prepare_initial_latents(self, prompt, batch_size, generator, **generation_kwargs):
        from flux3.autoencoder import latent_ids, latent_shape
        from flux3.dit import IN_CHANNELS
        from flux3.pipeline import scaled_ids
        self._require_loaded()
        if batch_size != 1 or generation_kwargs.get("latents") is not None:
            raise ValueError("Registered FLUX.3 native preparation requires one fresh native sample")
        height, width = int(generation_kwargs["height"]), int(generation_kwargs["width"])
        if (height, width) != (1024, 1024):
            raise ValueError("Registered FLUX.3 benchmark resolution is 1024x1024")
        h, w = latent_shape(height, width)
        seed = int(generator.initial_seed())
        noise = torch.randn((h * w, IN_CHANNELS["image"]), generator=torch.Generator().manual_seed(seed))
        native = noise.to(torch.bfloat16).float().to(self.device)
        ids = latent_ids(h, w, self.device)
        return native.unsqueeze(0), AdapterState(extra={
            "height": height, "width": width, "image_ids": ids,
            "latent_image_ids": scaled_ids(ids), "base_seed": seed,
            "initial_noise_generator_device": "cpu", "initial_noise_dtype": "torch.float32",
            "latent_dtype": "torch.float32", "initial_bf16_roundtrip": True,
        })

    def set_timesteps(self, num_inference_steps, latents=None, state=None):
        self._require_loaded()
        self.pipeline.scheduler.set_timesteps(num_inference_steps)
        self.timesteps = list(self.pipeline.scheduler.timesteps)
        return self.timesteps

    def _packed(self, latents, timestep, condition, state):
        from flux3.pipeline import pack
        packed = pack([[(condition.data["prompt_embeds"], condition.data["text_ids"]),
                        None, (latents[0], state.extra["latent_image_ids"])]])
        packed.timesteps["image"].fill_(float(timestep))
        return packed

    @torch.inference_mode()
    def predict_vector_field(self, latents, timestep, condition, state):
        self._require_loaded()
        scheduler = self.pipeline.scheduler
        index = scheduler.begin_index if scheduler.step_index is None else scheduler.step_index
        if float(timestep) != float(scheduler.timesteps[index]):
            raise RuntimeError("Prediction is not anchored to the current native interval")
        packed = self._packed(latents, timestep, condition, state)
        target = scheduler.sigmas[index + 1].reshape(1).to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = self.pipeline.transformer(packed, timesteps_ref=target)
        return prediction.unsqueeze(0)

    def scheduler_step(self, model_prediction, timestep, latents, state, generator=None):
        updated = self.pipeline.scheduler.step(model_prediction, timestep, latents, generator=generator)[0]
        return SchedulerStepResult(latents=updated, state=state)

    @torch.inference_mode()
    def decode_latents(self, latents, state):
        from flux3.pipeline import to_pil
        return [to_pil(self.pipeline.vae.decode(latents[0], state.extra["image_ids"]))]

    @torch.inference_mode()
    def verify_native_prediction(self, latents, timestep, condition, state, prediction, generation_config):
        """Independent official first-step prediction in the real Baseline workload."""
        from flux3.pipeline import pack
        from flux3.sampling import timesteps
        from flux3 import sampling
        if int(generation_config["num_inference_steps"]) != 16:
            raise RuntimeError("Native parity profile differs from the registered 16 updates")
        schedule = timesteps(16, float(self.config.get("schedule_alpha", 6.93)))
        packed = pack([[(condition.data["prompt_embeds"], condition.data["text_ids"]),
                        None, (latents[0], state.extra["latent_image_ids"])]])
        packed.timesteps["image"].fill_(schedule[0].item())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reference = self.pipeline.transformer(packed, timesteps_ref=schedule[1].reshape(1).to(self.device))
        reference = reference.unsqueeze(0)
        exact = torch.equal(prediction, reference)
        record = {
            "schema": "riddikulus.native-first-prediction.v1", "status": "passed" if exact else "failed",
            "adapter": self.adapter_name, "exact_equal": exact,
            "comparison": {"exact_equal": exact}, "native_reference_model_forwards": 1,
            "prediction_sha256": _identity(prediction), "reference_sha256": _identity(reference),
            "native_semantics": {"current_timestep": float(schedule[0]),
                                 "destination_timestep": float(schedule[1]), "guidance": "none",
                                 "state_dtype": str(latents.dtype), "prediction_dtype": str(prediction.dtype)},
            "sampling_source_sha256": hashlib.sha256(Path(inspect.getfile(sampling)).read_bytes()).hexdigest(),
            "reference_source_commit": REFERENCE_COMMIT,
        }
        if not exact:
            raise RuntimeError("Native FLUX.3 first-prediction parity failed: " + str(record))
        return record
