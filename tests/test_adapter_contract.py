from __future__ import annotations

import torch

from hierasafe_flow.adapters.registry import create_adapter, get_adapter_class, list_adapters


def test_registry_contains_target_adapters() -> None:
    adapters = list_adapters()
    for name in [
        "dummy",
        "sd35",
        "flux",
        "flux2",
        "qwen_image",
        "cogvideox",
        "wan",
        "ltx",
        "hunyuan_video",
        "cosmos3_t2i",
        "joyai_echo",
    ]:
        assert name in adapters
    assert get_adapter_class(model_id="THUDM/CogVideoX-5b").adapter_name == "cogvideox"
    assert get_adapter_class(model_id="nvidia/Cosmos3-Super-Text2Image").adapter_name == "cosmos3_t2i"


def test_dummy_adapter_contract() -> None:
    adapter = create_adapter(
        {"adapter": "dummy", "model_id": "dummy/test"},
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    adapter.load()
    condition = adapter.prepare_prompt("a test prompt")
    latents, state = adapter.prepare_initial_latents(
        prompt="a test prompt",
        batch_size=1,
        generator=torch.Generator(device="cpu").manual_seed(0),
        task="text_to_image",
        height=32,
        width=32,
    )
    timesteps = adapter.set_timesteps(2)
    prediction = adapter.predict_vector_field(latents, timesteps[0], condition, state)
    result = adapter.scheduler_step(prediction, timesteps[0], latents, state)
    decoded = adapter.decode_latents(result.latents, result.state)
    assert prediction.shape == latents.shape
    assert decoded.shape == latents.shape
