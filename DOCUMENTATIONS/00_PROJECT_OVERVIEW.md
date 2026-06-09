# Project Overview

HieraSafe-Flow implements inference-time steering for frozen text-to-image and text-to-video generators. It does not train model weights and does not query external safety models inside generation. The core loop repeatedly obtains the model's own vector-field prediction under the base prompt, neutral safety concept, unsafe concept, and safe sibling concept, then applies local replacement in latent token space.

The repository is organized around four contracts:

- `adapters`: expose latents, timesteps, model vector-field prediction, scheduler stepping, and decoding.
- `steering`: implements concept hierarchy loading, local masks, schedules, and vector-field bottleneck math.
- `generation`: runs prompt batches and saves traces, tensors, images, or videos.
- `configs`: records model mappings, concept hierarchies, experiment grids, and smoke-test prompts.

