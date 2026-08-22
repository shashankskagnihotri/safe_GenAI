#!/usr/bin/env bash
#SBATCH --job-name=iclr-tr-dev
#SBATCH --partition=gpu-vram-94gb
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --account=ml-staff
#SBATCH --qos=walltime5d
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=1-00:00:00
#SBATCH --output=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/trust_region_dev_%A_%a.out
#SBATCH --error=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/trust_region_dev_%A_%a.err

set -euo pipefail
REPOSITORY_ROOT=/ceph/sagnihot/projects/safety_genAI
PYTHON=/ceph/sagnihot/miniconda3/envs/safe_genai_conceptsteer/bin/python
cd "$REPOSITORY_ROOT"

test -n "$SLURM_ARRAY_TASK_ID"
test -n "$PUSH_TR_MANIFEST"
test -n "$PUSH_TR_MANIFEST_FILE_SHA256"
test -n "$PUSH_TR_CODE_COMMIT"
test -x "$PYTHON"
export PYTHONPATH="$REPOSITORY_ROOT/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "$PYTHON" scripts/push_for_iclr/run_trust_region_cell.py \
  --manifest "$PUSH_TR_MANIFEST" \
  --manifest-file-sha256 "$PUSH_TR_MANIFEST_FILE_SHA256" \
  --index "$SLURM_ARRAY_TASK_ID"
