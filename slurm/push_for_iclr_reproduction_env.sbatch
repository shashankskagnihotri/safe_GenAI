#!/usr/bin/env bash
#SBATCH --job-name=iclr-repro-env
#SBATCH --partition=cpu
#SBATCH --account=ml-staff
#SBATCH --qos=walltime5d
#SBATCH --array=0-9
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/reproduction_env_%A_%a.out
#SBATCH --error=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/reproduction_env_%A_%a.err

set -euo pipefail
METHODS_ROOT=/ceph/sagnihot/projects/safety_genAI_worktrees/push_for_iclr_methods
actual=$(git -C "$METHODS_ROOT" rev-parse HEAD)
test "$actual" = "$PUSH_FOR_ICLR_METHODS_COMMIT"
exec "$METHODS_ROOT/scripts/push_for_iclr/build_reproduction_env.sh" "$SLURM_ARRAY_TASK_ID"
