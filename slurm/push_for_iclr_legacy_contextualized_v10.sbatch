#!/usr/bin/env bash
#SBATCH --job-name=iclr-v10-context
#SBATCH --partition=gpu-vram-94gb
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --account=ml-staff
#SBATCH --qos=walltime5d
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=1-00:00:00
#SBATCH --output=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/v10_context_%A_%a.out
#SBATCH --error=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/v10_context_%A_%a.err

set -euo pipefail
: "${PUSH_V10_EXECUTION_ROOT:?PUSH_V10_EXECUTION_ROOT is required}"
: "${PUSH_V10_EXPECTED_COMMIT:?PUSH_V10_EXPECTED_COMMIT is required}"
: "${PUSH_V10_MANIFEST:?PUSH_V10_MANIFEST is required}"
: "${PUSH_V10_MANIFEST_FILE_SHA256:?PUSH_V10_MANIFEST_FILE_SHA256 is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

PYTHON=/ceph/sagnihot/miniconda3/envs/safe_genai_conceptsteer/bin/python
test -x "$PYTHON"
cd "$PUSH_V10_EXECUTION_ROOT"
test "$(git rev-parse HEAD)" = "$PUSH_V10_EXPECTED_COMMIT"
test -f "$PUSH_V10_MANIFEST"
export PYTHONPATH="$PUSH_V10_EXECUTION_ROOT/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "$PYTHON" scripts/push_for_iclr/run_legacy_bottleneck_cell.py \
  --manifest "$PUSH_V10_MANIFEST" \
  --manifest-file-sha256 "$PUSH_V10_MANIFEST_FILE_SHA256" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --expected-code-commit "$PUSH_V10_EXPECTED_COMMIT"
