# Logging And TensorBoard

Each run writes:

- `resolved_config.yaml`,
- `system_info.json`,
- `events.json`,
- `run.log`,
- per-sample `steering_trace.json`,
- optional TensorBoard scalars under `tensorboard/`.

TensorBoard logs base and steered vector-field stats plus per-concept activation and mask stats.

