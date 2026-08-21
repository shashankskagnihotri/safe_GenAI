# HieraSafe-Flow / ConceptSteer

Inference-time concept steering for frozen text-to-image and text-to-video diffusion or flow generators. The consolidated project root is:

```bash
cd /ceph/sagnihot/projects/safety_genAI
```

All commands below run from this repository only. The old overlay checkout is not a runtime dependency.

## Conda-Only Setup

Only conda is supported. Do not use `venv`, `virtualenv`, or `python -m venv`.

```bash
cd /ceph/sagnihot/projects/safety_genAI

conda env create -f environment.yml
conda activate safe_genai_conceptsteer

python -m pip install --no-deps -e .
python -m compileall .
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -m pytest
```

The environment name is `safe_genai_conceptsteer`. Slurm scripts activate the same environment with:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate safe_genai_conceptsteer
```

PyTorch is provided by conda through `environment.yml`; pip must not install or upgrade it.

## Repository Layout

- `configs/models/`: model adapter configs.
- `configs/concept_hierarchies/`: original safety concept hierarchies.
- `configs/concepts/`: benign benchmark concept trees.
- `configs/experiments/`: experiment, prompt-suite, and negative-prompt configs.
- `src/hierasafe_flow/`: package code for adapters, steering, generation, benchmarks, and evaluation.
- `scripts/`: local runner helpers.
- `scripts/slurm/`: Slurm entrypoints for consolidated benchmark jobs.
- `slurm/`: Slurm logs from repository-root jobs.
- `outputs/`: generated media and per-sample reports.
- `debugging/`: audits, environment notes, long-running reports, and result summaries.

## Run One Model For All Variants

The benign debugging benchmark is `benign_park_attribute_transfer_v1`. It uses seed `0` only.

Current July 1 result: for `qwen_image_2512`, the strongest usable Stage 0 setting is
`conceptsteer_full_strength_1p50` under
`outputs/TRYING_ALL_BENIGN_SAFETY/qwen_image_2512/conceptsteer_full_strength_1p50/P0_all_sources_main`.
It best transfers the stress-test prompt toward happy, red/blue clothing, walking beside the bench,
and holding a sandwich-like object. The output is not perfect because the sandwich placement is odd,
but it is the clearest completed ConceptSteer success in the final sweep.

For 15-second video models, the final July 1 sweeps completed technically but should not be treated
as semantic wins for this target. HunyuanVideo and Wan sparse steering produced valid 240-frame
videos, but the subject generally stayed seated/eating and artifacts increased at stronger settings.
LTX 2.3 stabilization variants also produced valid 240-frame videos, but low-strength variants stayed
seated/eating while stronger or unnormalized variants introduced visible artifacts.

```bash
cd /ceph/sagnihot/projects/safety_genAI
conda activate safe_genai_conceptsteer

python scripts/run_benign_park_benchmark.py \
  --model flux2_dev \
  --benchmark benign_park_attribute_transfer_v1 \
  --variants baseline,negativeprompt,negative_prompt_global,conceptsteer_color,conceptsteer_emotion,conceptsteer_pose,conceptsteer_action,conceptsteer_composition,conceptsteer_pose_action,conceptsteer_full \
  --seed 0 \
  --output-root outputs/benign_park_conceptsteer_debug
```

To prepare a Slurm array manifest instead of running locally:

```bash
cd /ceph/sagnihot/projects/safety_genAI
conda activate safe_genai_conceptsteer

python scripts/run_benign_park_benchmark.py \
  --model flux2_dev \
  --benchmark benign_park_attribute_transfer_v1 \
  --stage 0 \
  --seed 0 \
  --output-root outputs/benign_park_conceptsteer_debug \
  --write-manifest debugging/benign_stage0_manifest.json
```

## SLURM Usage

Default Stage 0 smoke submission:

```bash
cd /ceph/sagnihot/projects/safety_genAI
sbatch scripts/slurm/submit_benign_park_conceptsteer.sbatch
```

Array submission after manifest creation:

```bash
cd /ceph/sagnihot/projects/safety_genAI
sbatch --array=0-2 scripts/slurm/submit_benign_park_conceptsteer.sbatch debugging/benign_stage0_manifest.json
```

Monitor:

```bash
squeue -u $USER
tail -f slurm/safe_genai_benign_conceptsteer_<JOBID>_<ARRAYID>.out
tail -f slurm/safe_genai_benign_conceptsteer_<JOBID>_<ARRAYID>.err
```

## Evaluation

The evaluator uses CLIP image-text scores by default plus color statistics. It fails clearly if CLIP cannot load unless `--skip-clip` is explicitly passed.

```bash
cd /ceph/sagnihot/projects/safety_genAI
conda activate safe_genai_conceptsteer

python scripts/evaluate_benign_park_benchmark.py \
  --input-root outputs/benign_park_conceptsteer_debug \
  --benchmark benign_park_attribute_transfer_v1 \
  --output-json outputs/benign_park_conceptsteer_debug/evaluation_summary.json \
  --output-csv outputs/benign_park_conceptsteer_debug/evaluation_summary.csv
```

## Benchmark Files

- Prompt suite: `configs/experiments/benign_park_attribute_transfer_v1_prompts.yaml`
- Negative prompts: `configs/experiments/benign_park_attribute_transfer_v1_negative_prompts.yaml`
- Concept tree: `configs/concepts/benign_park_concept_tree.yaml`
- Stage/variant runner: `scripts/run_benign_park_benchmark.py`
- Evaluation runner: `scripts/evaluate_benign_park_benchmark.py`

Native negative-prompt variants run only when the selected diffusers pipeline exposes a real `negative_prompt` argument. Unsupported negative-prompt variants are recorded as `not_supported`; they are not faked with another method.
