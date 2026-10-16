"""Explicit image-model dispatch, with lazy registration for native extensions."""
from __future__ import annotations
import importlib
from .base import FrozenGeneratorAdapter
_TARGETS = {
    "flux": ("riddikulus.adapters.flux_adapter", "FluxAdapter"),
    "flux1": ("riddikulus.adapters.flux_adapter", "FluxAdapter"),
    "flux1_dev": ("riddikulus.adapters.flux_adapter", "FluxAdapter"),
    "flux2": ("riddikulus.adapters.flux_adapter", "Flux2Adapter"),
    "flux2_dev": ("riddikulus.adapters.flux_adapter", "Flux2Adapter"),
    "cosmos3_t2i": ("riddikulus.adapters.cosmos3_adapter", "Cosmos3TextToImageAdapter"),
    "sd35": ("riddikulus.adapters.sd35_adapter", "StableDiffusion35Adapter"),
    "flux3": ("riddikulus.adapters.flux3_adapter", "Flux3Adapter"),
}
_MODEL_IDS = {
    "black-forest-labs/FLUX.1-dev": "flux",
    "black-forest-labs/FLUX.2-dev": "flux2",
    "nvidia/Cosmos3-Super-Text2Image": "cosmos3_t2i",
    "stabilityai/stable-diffusion-3.5-large": "sd35",
    "black-forest-labs/flux-3-image-klein": "flux3",
}

def create_adapter(model_config, device, dtype):
    model_id = str(model_config.get("model_id", ""))
    name = model_config.get("adapter", _MODEL_IDS.get(model_id))
    if name not in _TARGETS:
        raise ValueError("Unsupported Riddikulus image adapter: " + str(name))
    module, class_name = _TARGETS[name]
    cls = getattr(importlib.import_module(module), class_name)
    if not issubclass(cls, FrozenGeneratorAdapter):
        raise TypeError("Native adapter must implement the frozen generator contract")
    return cls(model_id=model_id, device=device, dtype=dtype, config=model_config)
