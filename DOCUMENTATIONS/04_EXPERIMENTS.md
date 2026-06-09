# Experiments

Experiment YAMLs are under `configs/experiments/`. Each experiment can merge:

- `base_config`: common runtime and steering defaults,
- `model_config`: adapter and model metadata,
- `concept_config`: concept hierarchy path,
- local overrides for prompts, output directories, and ablations.

Smoke experiments use the dummy adapter and should pass before real model runs.

