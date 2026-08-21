#!/usr/bin/env bash
#SBATCH --job-name=iclr-gemini
#SBATCH --partition=cpu
#SBATCH --account=ml-staff
#SBATCH --qos=walltime5d
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=04:00:00
#SBATCH --output=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/stage2_gemini_%A_%a.out
#SBATCH --error=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/stage2_gemini_%A_%a.err

set -euo pipefail

EVAL_ROOT=/ceph/sagnihot/projects/safety_genAI_worktrees/push_for_iclr_evaluation
PYTHON=/ceph/sagnihot/miniconda3/envs/safe_genai_conceptsteer/bin/python
cd "$EVAL_ROOT"

: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
: "${PUSH_FOR_ICLR_MANIFEST:?PUSH_FOR_ICLR_MANIFEST is required}"
: "${PUSH_FOR_ICLR_MANIFEST_FILE_SHA256:?PUSH_FOR_ICLR_MANIFEST_FILE_SHA256 is required}"
: "${PUSH_FOR_ICLR_EVAL_COMMIT:?PUSH_FOR_ICLR_EVAL_COMMIT is required}"
: "${GEMINI_ForestAI_API_KEY:?GEMINI_ForestAI_API_KEY is required}"

export PYTHONPATH="${EVAL_ROOT}/src:${PYTHONPATH:-}"
exec "$PYTHON" scripts/push_for_iclr/evaluate_gemini_cell.py \
  --manifest "$PUSH_FOR_ICLR_MANIFEST" \
  --manifest-file-sha256 "$PUSH_FOR_ICLR_MANIFEST_FILE_SHA256" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --code-commit "$PUSH_FOR_ICLR_EVAL_COMMIT"
