from __future__ import annotations

from typing import Any, Type

import torch

from hierasafe_flow.adapters.base import DummyVectorFieldAdapter, FrozenGeneratorAdapter
from hierasafe_flow.adapters.cogvideox_adapter import CogVideoXAdapter
from hierasafe_flow.adapters.cogvideox_riflex_adapter import CogVideoXRIFLExAdapter
from hierasafe_flow.adapters.cosmos3_adapter import Cosmos3TextToImageAdapter
from hierasafe_flow.adapters.flux_dual_view_adapter import FluxDualViewAdapter
from hierasafe_flow.adapters.flux_adapter import Flux2Adapter, FluxAdapter
from hierasafe_flow.adapters.hunyuan_video_adapter import HunyuanVideoAdapter
from hierasafe_flow.adapters.ideogram4_adapter import Ideogram4Adapter
from hierasafe_flow.adapters.joyai_echo_adapter import JoyAIEchoAdapter
from hierasafe_flow.adapters.ltx_adapter import LTXAdapter
from hierasafe_flow.adapters.qwen_image_adapter import QwenImageAdapter
from hierasafe_flow.adapters.sd35_adapter import StableDiffusion35Adapter
from hierasafe_flow.adapters.wan_adapter import WanAdapter


_REGISTRY: dict[str, Type[FrozenGeneratorAdapter]] = {
    "dummy": DummyVectorFieldAdapter,
    "sd35": StableDiffusion35Adapter,
    "stable_diffusion_3_5_large": StableDiffusion35Adapter,
    "flux": FluxAdapter,
    "flux1": FluxAdapter,
    "flux1_dev": FluxAdapter,
    "flux_dual_view": FluxDualViewAdapter,
    "flux2": Flux2Adapter,
    "flux2_dev": Flux2Adapter,
    "ideogram4": Ideogram4Adapter,
    "ideogram4_nf4": Ideogram4Adapter,
    "qwen_image": QwenImageAdapter,
    "qwen_image_2512": QwenImageAdapter,
    "cogvideox": CogVideoXAdapter,
    "cogvideox_5b": CogVideoXAdapter,
    "cogvideox_riflex": CogVideoXRIFLExAdapter,
    "wan": WanAdapter,
    "wan22": WanAdapter,
    "wan22_t2v_a14b": WanAdapter,
    "ltx": LTXAdapter,
    "ltx_23": LTXAdapter,
    "hunyuan_video": HunyuanVideoAdapter,
    "cosmos3_t2i": Cosmos3TextToImageAdapter,
    "joyai_echo": JoyAIEchoAdapter,
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


def get_adapter_class(
    adapter_name: str | None = None, model_id: str | None = None
) -> Type[FrozenGeneratorAdapter]:
    key = adapter_name
    if key is None and model_id is not None:
        key = _MODEL_ID_TO_ADAPTER.get(model_id)
    if key is None:
        raise KeyError("No adapter name supplied and model_id is not recognized.")
    normalized = key.lower()
    if normalized not in _REGISTRY:
        raise KeyError(f"Unknown adapter '{key}'. Available adapters: {list_adapters()}")
    return _REGISTRY[normalized]


def create_adapter(
    model_config: dict[str, Any],
    device: torch.device | str,
    dtype: torch.dtype,
) -> FrozenGeneratorAdapter:
    model_id = str(model_config.get("model_id", ""))
    adapter_name = model_config.get("adapter")
    adapter_cls = get_adapter_class(adapter_name=adapter_name, model_id=model_id)
    return adapter_cls(model_id=model_id, device=device, dtype=dtype, config=model_config)
