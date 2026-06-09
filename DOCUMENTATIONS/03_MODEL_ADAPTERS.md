# Model Adapters

Adapters must expose five operations without invoking a full pipeline fallback:

- initial latent preparation,
- scheduler timestep preparation,
- prompt-conditioned vector-field prediction,
- scheduler stepping with the steered prediction,
- final latent decoding.

Verified public `model_index.json` mappings:

| Model | Pipeline class |
| --- | --- |
| `stabilityai/stable-diffusion-3.5-large` | `StableDiffusion3Pipeline` |
| `black-forest-labs/FLUX.1-dev` | `FluxPipeline` |
| `black-forest-labs/FLUX.2-dev` | `Flux2Pipeline` |
| `Qwen/Qwen-Image` | `QwenImagePipeline` |
| `Qwen/Qwen-Image-2512` | `QwenImagePipeline` |
| `zai-org/CogVideoX-5b` | `CogVideoXPipeline` |
| `THUDM/CogVideoX-5b` | `CogVideoXPipeline` |
| `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | `WanPipeline` |

Repos without public `model_index.json` from this environment:

- `Lightricks/LTX-2.3`
- `tencent/HunyuanVideo`
- `Wan-AI/Wan2.2-T2V-A14B` native repo

Those adapters raise clear `NotImplementedError` until the installed diffusers source exposes the full internal contract.

