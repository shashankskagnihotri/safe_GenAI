# Reproducibility

Reproducibility controls:

- all configs are resolved and copied into the output directory,
- random seeds are set through Python, NumPy, and PyTorch,
- CUDA TF32 behavior is configured from YAML,
- model adapters keep generator weights frozen,
- per-step steering traces are saved.

Real model reproducibility still depends on exact package versions, model revisions, CUDA kernels, and scheduler implementations.

