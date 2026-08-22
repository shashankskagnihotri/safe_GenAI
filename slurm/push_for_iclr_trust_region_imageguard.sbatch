#!/usr/bin/env bash
#SBATCH --job-name=iclr-tr-imageguard
#SBATCH --partition=gpu-vram-32gb
#SBATCH --gres=gpu:1
#SBATCH --account=ml-staff
#SBATCH --qos=walltime5d
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/trust_region_imageguard_%A_%a.out
#SBATCH --error=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/trust_region_imageguard_%A_%a.err

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?}"
: "${TR_EVAL_WORKTREE:?}"
: "${TR_SOURCE_ROOT:?}"
: "${TR_MANIFEST:?}"
: "${TR_MANIFEST_SHA256:?}"
: "${TR_EVAL_CONFIG:?}"
: "${TR_SUMMARY_ROOT:?}"
: "${TR_EVAL_COMMIT:?}"
test "$(git -C "$TR_EVAL_WORKTREE" rev-parse HEAD)" = "$TR_EVAL_COMMIT"
export PYTHONPATH="${TR_EVAL_WORKTREE}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec /ceph/sagnihot/miniconda3/envs/t2i_imageguard/bin/python \
  "$TR_EVAL_WORKTREE/scripts/push_for_iclr/evaluate_trust_region_imageguard_group.py" \
  --manifest "$TR_MANIFEST" --manifest-file-sha256 "$TR_MANIFEST_SHA256" \
  --source-root "$TR_SOURCE_ROOT" --config "$TR_EVAL_CONFIG" \
  --group-index "$SLURM_ARRAY_TASK_ID" --summary-root "$TR_SUMMARY_ROOT"
