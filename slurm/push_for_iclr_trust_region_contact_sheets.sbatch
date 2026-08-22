#!/usr/bin/env bash
#SBATCH --job-name=iclr-tr-sheets
#SBATCH --partition=cpu
#SBATCH --account=ml-staff
#SBATCH --qos=walltime5d
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/trust_region_sheets_%A_%a.out
#SBATCH --error=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/trust_region_sheets_%A_%a.err

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?}"
: "${TR_EVAL_WORKTREE:?}"
: "${TR_SOURCE_ROOT:?}"
: "${TR_MANIFEST:?}"
: "${TR_MANIFEST_SHA256:?}"
: "${TR_SHEET_ROOT:?}"
: "${TR_EVAL_COMMIT:?}"
test "$(git -C "$TR_EVAL_WORKTREE" rev-parse HEAD)" = "$TR_EVAL_COMMIT"
export PYTHONPATH="${TR_EVAL_WORKTREE}/src:${PYTHONPATH:-}"
exec /ceph/sagnihot/miniconda3/envs/safe_genai_conceptsteer/bin/python \
  "$TR_EVAL_WORKTREE/scripts/push_for_iclr/build_trust_region_contact_sheet.py" \
  --manifest "$TR_MANIFEST" --manifest-file-sha256 "$TR_MANIFEST_SHA256" \
  --source-root "$TR_SOURCE_ROOT" --sheet-index "$SLURM_ARRAY_TASK_ID" \
  --sheet-root "$TR_SHEET_ROOT"
