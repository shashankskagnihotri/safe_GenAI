# SLURM Usage

Smoke submission:

```bash
scripts/submit_smoke_tests.sh
```

Full experiment scripts are prepared but should be run only after smoke jobs pass:

```bash
scripts/submit_main_experiments.sh
scripts/submit_ablations.sh
```

The provided SBATCH files are set for this cluster's H100 NVL partition, `gpu-vram-94gb`, with `--gres=gpu:nvidia_h100_nvl:1`. Edit partition and environment activation lines for another cluster.
