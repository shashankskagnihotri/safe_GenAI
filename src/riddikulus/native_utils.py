"""Native image/noise helpers extracted from the completed method's runtime."""
from __future__ import annotations
from contextvars import ContextVar
import math
import os
from pathlib import Path
from typing import Any, Mapping
import numpy as np
from PIL import Image
import torch
from .artifacts import digest as sha256_file
from .conditioning_cache import encoding_fingerprint
from .numerics import tensor_sha256

_PROJECT_ROOT = ContextVar("riddikulus_project_root", default=Path.cwd())

def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _decoded_to_pil(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    if hasattr(value, "images"):
        value = value.images
    if isinstance(value, Mapping) and "images" in value:
        value = value["images"]
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise RuntimeError(f"Expected one decoded image, got {len(value)}")
        return _decoded_to_pil(value[0])
    if torch.is_tensor(value):
        tensor = value.detach().float().cpu()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4):
            tensor = tensor.permute(1, 2, 0)
        if float(tensor.min()) < 0:
            tensor = tensor.add(1).div(2)
        array = tensor.clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
        return Image.fromarray(array)
    if isinstance(value, np.ndarray):
        array = value[0] if value.ndim == 4 else value
        if np.issubdtype(array.dtype, np.floating):
            if float(array.min()) < 0:
                array = (array + 1) / 2
            array = np.clip(array, 0, 1) * 255
        return Image.fromarray(array.astype(np.uint8))
    raise TypeError(f"Unsupported decoded type: {type(value).__name__}")


def _atomic_image(path: Path, image: Image.Image) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    try:
        image.save(partial, format="PNG")
        with partial.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _scheduler_sigma(adapter: Any, step_index: int) -> tuple[float | None, str]:
    for attribute in ("pipeline", "_pipeline", "pipe", "_pipe"):
        pipeline = getattr(adapter, attribute, None)
        scheduler = getattr(pipeline, "scheduler", None)
        sigmas = getattr(scheduler, "sigmas", None)
        if sigmas is None or step_index >= len(sigmas):
            continue
        value = sigmas[step_index]
        if torch.is_tensor(value):
            if value.numel() != 1:
                continue
            value = value.detach().float().cpu().item()
        result = float(value)
        if math.isfinite(result):
            return result, f"{attribute}.scheduler.sigmas"
    return None, "adapter_scheduler_sigma_not_exposed"

def _prepare_initial_latents(
    adapter: Any, row: Mapping[str, Any], generation: Mapping[str, Any], generator: torch.Generator,
) -> tuple[torch.Tensor, Any]:
    """Replay pinned native noise when CUDA RNG hardware differs from its origin.

    Supplied latents go through the model's native preparation and packing API.
    Neither the legacy Cosmos hash nor same-input checks between arms are relaxed.
    """
    kwargs = dict(generation)
    reference = row.get("initial_latent_reference")
    if reference is not None:
        from safetensors.torch import load_file
        path = (_PROJECT_ROOT.get() / reference["path"]).resolve()
        if not path.is_relative_to((_PROJECT_ROOT.get() / "outputs").resolve()) or sha256_file(path) != reference["file_sha256"]:
            raise RuntimeError("Initial-latent artifact path/hash mismatch")
        if (reference["model"], reference["model_revision"], reference["seed"]) != (
            row["model"], row["model_revision"], int(row["seed"]),
        ):
            raise RuntimeError("Initial-latent artifact identity mismatch")
        values = load_file(str(path), device="cpu")
        if set(values) != {"native_latents", "prepared_latents", "generator_state_before", "generator_state_after"}:
            raise RuntimeError("Unexpected initial-latent artifact tensors")
        native, prepared = values["native_latents"], values["prepared_latents"]
        if native.dtype != torch.bfloat16 or prepared.dtype != torch.bfloat16 or reference["dtype"] != "torch.bfloat16":
            raise RuntimeError("Initial-latent artifact precision changed")
        if list(native.shape) != reference["native_shape"] or list(prepared.shape) != reference["prepared_shape"]:
            raise RuntimeError("Initial-latent artifact shape mismatch")
        if tensor_sha256(native) != reference["native_tensor_sha256"] or tensor_sha256(prepared) != reference["tensor_sha256"]:
            raise RuntimeError("Initial-latent tensor bytes changed")
        if reference["tensor_sha256"] != row.get("expected_initial_latent_sha256"):
            raise RuntimeError("Canonical noise and cell's initial-latent pin disagree")
        after = values["generator_state_after"]
        if after.dtype != torch.uint8 or tensor_sha256(after) != reference["generator_state_after_sha256"]:
            raise RuntimeError("Canonical CUDA generator state changed")
        if not torch.equal(generator.get_state(), values["generator_state_before"]):
            raise RuntimeError("Generator was not at the canonical pre-draw seed state")
        if "latents" in kwargs:
            raise RuntimeError("Two initial-latent sources were supplied")
        generator.set_state(after)
        kwargs["latents"] = native.to(device=adapter.device)
    latents, state = adapter.prepare_initial_latents(
        row["original_prompt"], batch_size=1, generator=generator, **kwargs,
    )
    if reference is not None and (list(latents.shape) != reference["prepared_shape"] or latents.dtype != torch.bfloat16):
        raise RuntimeError("Native preparation changed the canonical noise shape/dtype")
    return latents, state


def _sigma_intervals(adapter: Any, num_steps: int) -> list[tuple[float, float, float, str]]:
    values = [_scheduler_sigma(adapter, index) for index in range(num_steps + 1)]
    if any(value is None for value, _ in values):
        raise RuntimeError("Native scheduler must expose all sigmas, including the terminal sigma")
    output = []
    for index in range(num_steps):
        current, source = values[index]
        following, next_source = values[index + 1]
        assert current is not None and following is not None
        if source != next_source or not all(math.isfinite(value) for value in (current, following)):
            raise RuntimeError("Invalid native sigma sequence")
        if following > current + 1e-6:
            raise RuntimeError("Expected descending native sigma sequence")
        output.append((current, following, abs(following - current), source))
    return output


def zero_controller(v_base, exposure, reason):
    rms = float(v_base.float().square().mean().sqrt())
    return {'mode': 'native_no_concept_inference', 'reason': reason,
            'candidate_rms': 0.0, 'applied_delta_rms': 0.0, 'base_rms': rms,
            'applied_delta_relative_rms': 0.0, 'applied_relative_base_rms': 0.0, 'global_cap_binding': False,
            'post_local_rms': 0.0, 'post_global_rms': 0.0,
            'candidate_relative_squared_dose': 0.0, 'post_local_relative_squared_dose': 0.0,
            'post_global_relative_squared_dose': 0.0, 'local_removed_relative_squared_dose': 0.0,
            'global_removed_relative_squared_dose': 0.0, 'exposure_increment': 0.0,
            'exposure_before': exposure.cumulative_exposure, 'exposure_after': exposure.cumulative_exposure,
            'neutral_prediction_observed': False, 'prompt_residual_rms': None,
            'cumulative_exposure': exposure.cumulative_exposure,
            'actual_bounds_verified': True}
