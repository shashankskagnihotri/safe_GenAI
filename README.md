# Hierarchical Concept Vector-Field Bottleneck

Research code for inference-time safety steering of frozen text-to-image and text-to-video diffusion/flow generators. The method operates on the generator vector field itself: it compares base, unsafe-concept, neutral-concept, and safe-sibling vector fields at selected denoising steps, then replaces only locally unsafe vector components with safe sibling directions.

This project intentionally does not use external safety classifiers, VLMs, decoded-image detectors, or negative-prompt-only fallbacks inside the generation loop.

## Core Formula

```text
v_base = v_theta(z_t, t, prompt)
b_unsafe = v_theta(z_t, t, prompt + unsafe_concept) - v_theta(z_t, t, prompt + neutral_concept)
b_safe = v_theta(z_t, t, prompt + safe_sibling_concept) - v_theta(z_t, t, prompt + neutral_concept)
activation = relu(cosine(v_base, b_unsafe) - cosine(v_base, b_safe) + margin)
v_steered = v_base + lambda_t * mask * (b_safe - b_unsafe)
```

## Quick Start

```bash
conda env create -f environment.yml
conda activate hierasafe-flow
pip install -e .
pytest
```

Run the dummy smoke path, which exercises the full steering loop without downloading model weights:

```bash
hierasafe-smoke --config configs/experiments/smoke_t2i.yaml
```

Run a real model only after accepting the relevant model licenses and ensuring the adapter can inspect the installed diffusers pipeline:

```bash
hierasafe-generate \
  --config configs/experiments/main_nudity_t2i.yaml \
  --prompt "a documentary-style portrait in a studio"
```

## Adapter Mapping

Tiny public Hugging Face metadata was inspected for `model_index.json` where available. The current workspace did not have diffusers installed, so runtime adapter inspection is still performed before generation.

| Model | Adapter | Pipeline class | Metadata status |
| --- | --- | --- | --- |
| `stabilityai/stable-diffusion-3.5-large` | `sd35` | `StableDiffusion3Pipeline` | verified `model_index.json` |
| `black-forest-labs/FLUX.1-dev` | `flux` | `FluxPipeline` | verified `model_index.json` |
| `black-forest-labs/FLUX.2-dev` | `flux2` | `Flux2Pipeline` | verified `model_index.json` |
| `Qwen/Qwen-Image` | `qwen_image` | `QwenImagePipeline` | verified `model_index.json` |
| `Qwen/Qwen-Image-2512` | `qwen_image` | `QwenImagePipeline` | verified `model_index.json` |
| `zai-org/CogVideoX-5b` | `cogvideox` | `CogVideoXPipeline` | verified `model_index.json` |
| `THUDM/CogVideoX-5b` | `cogvideox` | `CogVideoXPipeline` | verified `model_index.json` |
| `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | `wan` | `WanPipeline` | verified `model_index.json` |
| `Wan-AI/Wan2.2-T2V-A14B` | `wan_native_alias` | n/a | native Wan repo, use diffusers repo for steering |
| `Lightricks/LTX-2.3` | `ltx` | `LTXPipeline` expected | no public `model_index.json` found |
| `tencent/HunyuanVideo` | `hunyuan_video` | `HunyuanVideoPipeline` expected | no public `model_index.json` found |

Adapters fail with `NotImplementedError` when they cannot expose latents, timesteps, scheduler stepping, and model predictions through diffusers internals. They never call the standard pipeline as a steering fallback.

