#!/usr/bin/env bash
#SBATCH --job-name=iclr-a13
#SBATCH --partition=gpu-vram-94gb
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --account=ml-staff
#SBATCH --qos=walltime5d
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=1-00:00:00
#SBATCH --output=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/stage2_ablation_%A_%a.out
#SBATCH --error=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/stage2_ablation_%A_%a.err

set -euo pipefail

REPOSITORY_ROOT=/ceph/sagnihot/projects/safety_genAI
PYTHON=/ceph/sagnihot/miniconda3/envs/safe_genai_conceptsteer/bin/python
cd "$REPOSITORY_ROOT"

: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
: "${PUSH_FOR_ICLR_MANIFEST:?PUSH_FOR_ICLR_MANIFEST is required}"
: "${PUSH_FOR_ICLR_MANIFEST_FILE_SHA256:?PUSH_FOR_ICLR_MANIFEST_FILE_SHA256 is required}"
: "${PUSH_FOR_ICLR_CODE_COMMIT:?PUSH_FOR_ICLR_CODE_COMMIT is required}"

test -x "$PYTHON"
export PYTHONPATH="${REPOSITORY_ROOT}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "$PYTHON" scripts/push_for_iclr/run_ablation_cell.py \
  --manifest "$PUSH_FOR_ICLR_MANIFEST" \
  --manifest-file-sha256 "$PUSH_FOR_ICLR_MANIFEST_FILE_SHA256" \
  --index "$SLURM_ARRAY_TASK_ID"
