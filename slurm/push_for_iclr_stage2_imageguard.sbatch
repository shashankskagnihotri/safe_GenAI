#!/usr/bin/env bash
#SBATCH --job-name=iclr-imguard
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:1
#SBATCH --account=ml-staff
#SBATCH --qos=walltime5d
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/stage2_imageguard_%j.out
#SBATCH --error=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/stage2_imageguard_%j.err

set -euo pipefail

EVAL_ROOT=/ceph/sagnihot/projects/safety_genAI_worktrees/push_for_iclr_evaluation
PYTHON=/ceph/sagnihot/miniconda3/envs/t2i_imageguard/bin/python
cd "$EVAL_ROOT"

: "${PUSH_FOR_ICLR_MANIFEST:?PUSH_FOR_ICLR_MANIFEST is required}"
: "${PUSH_FOR_ICLR_EVAL_MODEL:?PUSH_FOR_ICLR_EVAL_MODEL is required}"
: "${PUSH_FOR_ICLR_EVAL_ABLATION:?PUSH_FOR_ICLR_EVAL_ABLATION is required}"
: "${PUSH_FOR_ICLR_EVAL_COMMIT:?PUSH_FOR_ICLR_EVAL_COMMIT is required}"

export PYTHONPATH="${EVAL_ROOT}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
exec "$PYTHON" scripts/push_for_iclr/evaluate_imageguard_group.py \
  --manifest "$PUSH_FOR_ICLR_MANIFEST" \
  --model "$PUSH_FOR_ICLR_EVAL_MODEL" \
  --ablation "$PUSH_FOR_ICLR_EVAL_ABLATION" \
  --code-commit "$PUSH_FOR_ICLR_EVAL_COMMIT"
