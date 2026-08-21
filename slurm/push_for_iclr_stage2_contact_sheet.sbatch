#!/usr/bin/env bash
#SBATCH --job-name=iclr-sheet
#SBATCH --partition=cpu
#SBATCH --account=ml-staff
#SBATCH --qos=walltime5d
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/stage2_contact_%j.out
#SBATCH --error=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/stage2_contact_%j.err

set -euo pipefail

EVAL_ROOT=/ceph/sagnihot/projects/safety_genAI_worktrees/push_for_iclr_evaluation
PYTHON=/ceph/sagnihot/miniconda3/envs/safe_genai_conceptsteer/bin/python
cd "$EVAL_ROOT"

: "${PUSH_FOR_ICLR_MANIFEST:?PUSH_FOR_ICLR_MANIFEST is required}"
: "${PUSH_FOR_ICLR_EVAL_MODEL:?PUSH_FOR_ICLR_EVAL_MODEL is required}"
: "${PUSH_FOR_ICLR_EVAL_ABLATION:?PUSH_FOR_ICLR_EVAL_ABLATION is required}"
: "${PUSH_FOR_ICLR_EVAL_CATEGORY:?PUSH_FOR_ICLR_EVAL_CATEGORY is required}"
: "${PUSH_FOR_ICLR_EVAL_COMMIT:?PUSH_FOR_ICLR_EVAL_COMMIT is required}"

export PYTHONPATH="${EVAL_ROOT}/src:${PYTHONPATH:-}"
exec "$PYTHON" scripts/push_for_iclr/build_ablation_contact_sheet.py \
  --manifest "$PUSH_FOR_ICLR_MANIFEST" \
  --model "$PUSH_FOR_ICLR_EVAL_MODEL" \
  --ablation "$PUSH_FOR_ICLR_EVAL_ABLATION" \
  --category "$PUSH_FOR_ICLR_EVAL_CATEGORY" \
  --code-commit "$PUSH_FOR_ICLR_EVAL_COMMIT"
