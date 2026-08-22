#!/usr/bin/env bash
#SBATCH --job-name=iclr-des-contacts
#SBATCH --partition=cpu
#SBATCH --array=0-193
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --nice=10000
#SBATCH --output=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/des_contacts_%A_%a.out
#SBATCH --error=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/des_contacts_%A_%a.err
set -euo pipefail
ROOT="${PUSH_FOR_ICLR_EXECUTION_ROOT:?PUSH_FOR_ICLR_EXECUTION_ROOT is required}"
PY=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/REPRODUCTIONS/ENVIRONMENTS/des/1d77e0a2a720/env/bin/python
cd "$ROOT"
test "$(git rev-parse HEAD)" = "${PUSH_FOR_ICLR_EXPECTED_COMMIT:?PUSH_FOR_ICLR_EXPECTED_COMMIT is required}"
exec "$PY" scripts/push_for_iclr/run_des_sd15_reproduction.py --config configs/experiments/push_for_iclr/des_sd15_paper_reproduction_v1.json render-contact --page-index "$SLURM_ARRAY_TASK_ID"
