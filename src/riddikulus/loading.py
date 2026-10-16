"""Pinned native loading and runtime guardrail inspection for Riddikulus."""
from __future__ import annotations
from copy import deepcopy
from typing import Any
import torch
from .adapters.registry import create_adapter

def load_model(model_config, device="cuda", dtype=torch.bfloat16):
    config = deepcopy(model_config)
    runtime = config.get("model", config)
    generation = config.get("generation", {})
    for key in ("guidance_scale", "height", "width"):
        if key in generation:
            runtime[key] = generation[key]
    runtime.setdefault("negative_prompt", "")
    if dtype != torch.bfloat16 or runtime.get("cpu_offload", False):
        raise ValueError("Riddikulus uses resident BF16 weights, without quantization or CPU offload")
    device = torch.device(device)
    if runtime.get("adapter") == "flux3":
        adapter = create_adapter(runtime, device=device, dtype=dtype)
        adapter.load()
    else:
        from .placement import load_adapter
        if runtime.get("adapter") == "cosmos3_t2i":
            if runtime.get("offline_research_disable_guardrails") is not True:
                raise ValueError("Cosmos requires the explicit local-research guardrail flag")
            runtime.setdefault("load_kwargs", {})["enable_safety_checker"] = False
        adapter = load_adapter(runtime, device=device, dtype=dtype)
    adapter.config["_study_guardrail_proof"] = runtime_guardrail_proof(adapter)
    return adapter

def runtime_guardrail_proof(adapter: Any) -> dict[str, Any]:
    """Inspect runtime components rather than inferring checker state from text.

    This establishes absence of application/provider guardrails in this local
    execution. It cannot remove or rule out safety behavior learned in weights.
    """
    if not getattr(adapter, "loaded", False) or adapter.pipeline is None:
        raise RuntimeError("Guardrail proof requires a loaded native pipeline")
    pipeline = adapter.pipeline
    names = ("safety_checker", "text_safety_checker", "image_safety_checker",
             "video_safety_checker", "guardrail", "guardrails")
    observed = {name: None if getattr(pipeline, name, None) is None
                else type(getattr(pipeline, name)).__name__ for name in names}
    active = {name: value for name, value in observed.items() if value is not None}
    if active:
        raise RuntimeError(f"The study found active safety-checker components: {active}")
    cosmos = adapter.adapter_name == "cosmos3_t2i"
    if cosmos:
        contract = adapter.config.get("_cosmos3_guardrail_contract", {})
        if adapter.config.get("offline_research_disable_guardrails") is not True or any(
            contract.get(key) is not False for key in
            ("guardrails_enabled", "constructor_enable_safety_checker", "text_pre_generation",
             "visual_post_decode")
        ):
            raise RuntimeError("Cosmos guardrail-disable runtime contract is incomplete")
    else:
        contract = None
    result = {
        "schema": "nailing-method.guardrail-proof.v1",
        "model_id": adapter.model_id,
        "model_revision": adapter.config.get("revision"),
        "adapter": adapter.adapter_name,
        "pipeline_class": type(pipeline).__name__,
        "runtime_checker_components": observed,
        "cosmos_guardrail_contract": contract,
        "execution": "local_checkpoint_native_pipeline",
        "external_inference_provider": None,
        "application_guardrails_active": False,
        "learned_weight_safety_behavior_removed": False,
    }
    if adapter.adapter_name == "flux3":
        native = adapter.config.get("_study_guardrail_proof", {})
        if any(native.get(key) is not False for key in (
            "safety", "prompt_upsampling", "upsampler_loaded", "safety_classifier_loaded",
            "reference_generate_wrapper_called",
        )):
            raise RuntimeError("FLUX.3 raw-prompt and disabled-checker contract is incomplete")
        result["flow_map_native_contract"] = native
    return result
