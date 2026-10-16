"""Exact BF16 model placement for the ECCV experiment's allocated GPUs.

This changes placement only; prediction, conditioning and scheduler methods
remain those of the existing model adapters. No quantization or CPU offload.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch

from .adapters.registry import create_adapter
from .transport import require_exact_gpu_transport


def _returns_to_input_device(hook: Any) -> bool:
    """Accelerate must gather outputs needed by the native FLUX callers.

    Mistral returns hidden states from several shards which the pipeline stacks;
    the denoiser returns a field which the adapter combines with CUDA-0 latents.
    Accelerate's root AlignDevicesHook provides this contract, including when
    another integration has wrapped it in a SequentialHook.
    """
    return bool(getattr(hook, "io_same_device", False)) or any(
        _returns_to_input_device(child) for child in getattr(hook, "hooks", ())
    )


def _placement(module: Any, label: str) -> dict[str, str]:
    mapping = getattr(module, "hf_device_map", None)
    if not isinstance(mapping, dict) or not mapping:
        raise RuntimeError(f"{label} did not expose its dispatched device map")
    normalized = {str(key): str(value) for key, value in mapping.items()}
    if any(value in {"cpu", "disk", "meta"} for value in normalized.values()):
        raise RuntimeError(f"{label} requires GPU-only placement: {normalized}")
    if any(parameter.device.type != "cuda" for parameter in module.parameters()):
        raise RuntimeError(f"{label} retained a non-CUDA parameter")
    if len(set(normalized.values())) > 1 and not _returns_to_input_device(getattr(module, "_hf_hook", None)):
        raise RuntimeError(f"{label} lacks the Accelerate hook which returns sharded outputs to the input device")
    return normalized


def _free_budgets(reserve_gib: float) -> dict[int | str, int]:
    reserve = int(reserve_gib * 1024**3)
    budgets: dict[int | str, int] = {"cpu": 0}
    for index in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(index)
        available = int(free) - reserve
        if available <= 0:
            raise RuntimeError(f"Insufficient runtime reserve on allocated GPU {index}")
        budgets[index] = available
    return budgets


def load_adapter(model_runtime: dict[str, Any], *, device: torch.device, dtype: torch.dtype):
    runtime = dict(model_runtime)
    adapter = create_adapter(runtime, device=device, dtype=dtype)
    policy = runtime.get("eccv_placement", "native")
    if policy == "native":
        adapter.load()
        return adapter
    if policy not in {"four_gpu_bf16", "two_gpu_bf16"}:
        raise ValueError(f"Unknown ECCV placement: {policy}")
    gpu_count = 2 if policy == "two_gpu_bf16" else 4
    if dtype != torch.bfloat16 or torch.cuda.device_count() != gpu_count:
        raise RuntimeError(f"ECCV {policy} requires exactly {gpu_count} allocated CUDA GPUs and BF16")
    if gpu_count == 2 and any(torch.cuda.get_device_properties(i).total_memory < 80 * 1024**3 for i in range(gpu_count)):
        raise RuntimeError("Two-GPU BF16 placement requires at least 80 GiB per allocated GPU")
    transport = require_exact_gpu_transport()
    print(json.dumps({"eccv_gpu_transport": transport}, sort_keys=True), flush=True)
    if runtime["adapter"] == "cosmos3_t2i":
        adapter.load()
        adapter.config["_eccv_placement"] = {"policy": policy, "visible_gpus": gpu_count, "transport_preflight": transport, "quantization": None, "cpu_offload": False}
        return adapter
    if runtime["adapter"] != "flux2":
        raise ValueError("The ECCV placement policy supports only the two requested models")
    if runtime.get("cpu_offload", False):
        raise ValueError("GPU-resident ECCV FLUX placement cannot enable CPU offload")

    from diffusers import Flux2Pipeline, Flux2Transformer2DModel
    from transformers import Mistral3ForConditionalGeneration

    source = Path(str(runtime["local_snapshot_path"]))
    if not source.is_dir() or source.name != str(runtime["revision"]):
        raise RuntimeError("FLUX local snapshot is absent or differs from the sealed revision")

    # Reserve enough room for the denoiser on every GPU. Text encoder total
    # BF16 checkpoint is 44.73 GiB; a total 64-GiB cap leaves balancing slack.
    encoder_budgets = _free_budgets(12.0)
    for index in range(gpu_count):
        encoder_budgets[index] = min(encoder_budgets[index], (64 // gpu_count) * 1024**3)
    encoder = Mistral3ForConditionalGeneration.from_pretrained(
        str(source / "text_encoder"),
        torch_dtype=dtype,
        device_map="balanced",
        max_memory=encoder_budgets,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    encoder_map = _placement(encoder, "FLUX text encoder")
    gc.collect()
    torch.cuda.empty_cache()
    transformer_budgets = _free_budgets(12.0)
    if sum(transformer_budgets[i] for i in range(gpu_count)) < 64_446_562_304:
        raise RuntimeError("Allocated GPU headroom is insufficient for the pinned BF16 transformer")
    transformer = Flux2Transformer2DModel.from_pretrained(
        str(source),
        subfolder="transformer",
        torch_dtype=dtype,
        device_map="balanced",
        max_memory=transformer_budgets,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    transformer_map = _placement(transformer, "FLUX transformer")
    pipeline = Flux2Pipeline.from_pretrained(
        str(source),
        text_encoder=encoder,
        transformer=transformer,
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    if pipeline.text_encoder is not encoder or pipeline.transformer is not transformer:
        raise RuntimeError("FLUX pipeline replaced a component after its GPU placement was validated")
    # Do not call pipeline.to(): it would move the dispatched components and
    # invalidate their layer maps. Check that construction preserved the hooks.
    _placement(pipeline.text_encoder, "FLUX pipeline text encoder")
    _placement(pipeline.transformer, "FLUX pipeline transformer")
    pipeline.vae.to(device=device, dtype=dtype)
    adapter.pipeline = pipeline
    adapter._freeze_pipeline()
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    adapter._validate_components()
    adapter.loaded = True
    adapter.config["_eccv_placement"] = {
        "policy": policy,
        "visible_gpus": gpu_count,
        "transport_preflight": transport,
        "text_encoder_device_map": encoder_map,
        "transformer_device_map": transformer_map,
        "text_encoder_max_memory_bytes": {str(k): v for k, v in encoder_budgets.items()},
        "transformer_max_memory_bytes": {str(k): v for k, v in transformer_budgets.items()},
        "runtime_reserve_gib_per_gpu": 12.0,
        "quantization": None,
        "cpu_offload": False,
    }
    print(json.dumps({"eccv_placement": adapter.config["_eccv_placement"]}, sort_keys=True), flush=True)
    return adapter
