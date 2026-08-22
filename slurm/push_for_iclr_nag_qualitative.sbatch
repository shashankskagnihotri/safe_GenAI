#!/usr/bin/env bash
#SBATCH --job-name=iclr-nag-qual
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --nice=10000
#SBATCH --output=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/%x_%A_%a.out
#SBATCH --error=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/%x_%A_%a.err

set -euo pipefail
: "${PUSH_FOR_ICLR_EXECUTION_ROOT:?missing PUSH_FOR_ICLR_EXECUTION_ROOT}"
: "${PUSH_FOR_ICLR_EXPECTED_COMMIT:?missing PUSH_FOR_ICLR_EXPECTED_COMMIT}"
: "${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"
test "$(git -C "$PUSH_FOR_ICLR_EXECUTION_ROOT" rev-parse HEAD)" = "$PUSH_FOR_ICLR_EXPECTED_COMMIT"
git -C "$PUSH_FOR_ICLR_EXECUTION_ROOT" diff --quiet
git -C "$PUSH_FOR_ICLR_EXECUTION_ROOT" diff --cached --quiet
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
ENV_ROOT=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/REPRODUCTIONS/ENVIRONMENTS/nag/8bb34a16e517/env
CONFIG="$PUSH_FOR_ICLR_EXECUTION_ROOT/configs/experiments/push_for_iclr/nag_qualitative_integrity_v1.json"
"$ENV_ROOT/bin/python" "$PUSH_FOR_ICLR_EXECUTION_ROOT/scripts/push_for_iclr/run_nag_qualitative_integrity.py" --config "$CONFIG" --arm-index "$SLURM_ARRAY_TASK_ID"
