from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any

import torch

from hierasafe_flow.adapters.base import FrozenGeneratorAdapter


AdapterTarget = tuple[str, str]


_REGISTRY: dict[str, AdapterTarget] = {
    "dummy": ("hierasafe_flow.adapters.base", "DummyVectorFieldAdapter"),
    "sd35": (
        "hierasafe_flow.adapters.sd35_adapter",
        "StableDiffusion35Adapter",
    ),
    "stable_diffusion_3_5_large": (
        "hierasafe_flow.adapters.sd35_adapter",
        "StableDiffusion35Adapter",
    ),
    "flux": ("hierasafe_flow.adapters.flux_adapter", "FluxAdapter"),
    "flux1": ("hierasafe_flow.adapters.flux_adapter", "FluxAdapter"),
    "flux1_dev": ("hierasafe_flow.adapters.flux_adapter", "FluxAdapter"),
    "flux_dual_view": (
        "hierasafe_flow.adapters.flux_dual_view_adapter",
        "FluxDualViewAdapter",
    ),
    "flux2": ("hierasafe_flow.adapters.flux_adapter", "Flux2Adapter"),
    "flux2_dev": ("hierasafe_flow.adapters.flux_adapter", "Flux2Adapter"),
    "ideogram4": (
        "hierasafe_flow.adapters.ideogram4_adapter",
        "Ideogram4Adapter",
    ),
    "ideogram4_nf4": (
        "hierasafe_flow.adapters.ideogram4_adapter",
        "Ideogram4Adapter",
    ),
    "qwen_image": (
        "hierasafe_flow.adapters.qwen_image_adapter",
        "QwenImageAdapter",
    ),
    "qwen_image_2512": (
        "hierasafe_flow.adapters.qwen_image_adapter",
        "QwenImageAdapter",
    ),
    "cogvideox": (
        "hierasafe_flow.adapters.cogvideox_adapter",
        "CogVideoXAdapter",
    ),
    "cogvideox_5b": (
        "hierasafe_flow.adapters.cogvideox_adapter",
        "CogVideoXAdapter",
    ),
    "cogvideox_riflex": (
        "hierasafe_flow.adapters.cogvideox_riflex_adapter",
        "CogVideoXRIFLExAdapter",
    ),
    "wan": ("hierasafe_flow.adapters.wan_adapter", "WanAdapter"),
    "wan22": ("hierasafe_flow.adapters.wan_adapter", "WanAdapter"),
    "wan22_t2v_a14b": (
        "hierasafe_flow.adapters.wan_adapter",
        "WanAdapter",
    ),
    "ltx": ("hierasafe_flow.adapters.ltx_adapter", "LTXAdapter"),
    "ltx_23": ("hierasafe_flow.adapters.ltx_adapter", "LTXAdapter"),
    "hunyuan_video": (
        "hierasafe_flow.adapters.hunyuan_video_adapter",
        "HunyuanVideoAdapter",
    ),
    "cosmos3_t2i": (
        "hierasafe_flow.adapters.cosmos3_adapter",
        "Cosmos3TextToImageAdapter",
    ),
    "joyai_echo": (
        "hierasafe_flow.adapters.joyai_echo_adapter",
        "JoyAIEchoAdapter",
    ),
}


_MODEL_ID_TO_ADAPTER: dict[str, str] = {
    "stabilityai/stable-diffusion-3.5-large": "sd35",
    "black-forest-labs/FLUX.1-dev": "flux",
    "black-forest-labs/FLUX.2-dev": "flux2",
    "ideogram-ai/ideogram-4-nf4": "ideogram4",
    "ideogram-ai/ideogram-4-nf4-diffusers": "ideogram4",
    "Qwen/Qwen-Image": "qwen_image",
    "Qwen/Qwen-Image-2512": "qwen_image",
    "zai-org/CogVideoX-5b": "cogvideox",
    "THUDM/CogVideoX-5b": "cogvideox",
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": "wan",
    "Wan-AI/Wan2.2-T2V-A14B": "wan",
    "Lightricks/LTX-2.3": "ltx",
    "tencent/HunyuanVideo": "hunyuan_video",
    "nvidia/Cosmos3-Super-Text2Image": "cosmos3_t2i",
    "jdopensource/JoyAI-Echo": "joyai_echo",
}


def list_adapters() -> list[str]:
    return sorted(_REGISTRY)


@lru_cache(maxsize=None)
def _load_adapter_class(normalized: str) -> type[FrozenGeneratorAdapter]:
    module_name, class_name = _REGISTRY[normalized]
    module = importlib.import_module(module_name)
    adapter_class = getattr(module, class_name)
    if (
        not isinstance(adapter_class, type)
        or not issubclass(adapter_class, FrozenGeneratorAdapter)
    ):
        raise TypeError(
            f"Registered adapter {normalized!r} does not resolve to a "
            "FrozenGeneratorAdapter subclass."
        )
    return adapter_class


def get_adapter_class(
    adapter_name: str | None = None, model_id: str | None = None
) -> type[FrozenGeneratorAdapter]:
    key = adapter_name
    if key is None and model_id is not None:
        key = _MODEL_ID_TO_ADAPTER.get(model_id)
    if key is None:
        raise KeyError("No adapter name supplied and model_id is not recognized.")
    normalized = key.lower()
    if normalized not in _REGISTRY:
        raise KeyError(f"Unknown adapter '{key}'. Available adapters: {list_adapters()}")
    return _load_adapter_class(normalized)


def create_adapter(
    model_config: dict[str, Any],
    device: torch.device | str,
    dtype: torch.dtype,
) -> FrozenGeneratorAdapter:
    model_id = str(model_config.get("model_id", ""))
    adapter_name = model_config.get("adapter")
    adapter_cls = get_adapter_class(adapter_name=adapter_name, model_id=model_id)
    return adapter_cls(model_id=model_id, device=device, dtype=dtype, config=model_config)
