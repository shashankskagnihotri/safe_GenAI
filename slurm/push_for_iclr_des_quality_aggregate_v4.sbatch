#!/usr/bin/env bash
#SBATCH --job-name=iclr-des-quality-aggregate-v4
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/des_quality_v4_aggregate_%j.out
#SBATCH --error=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/des_quality_v4_aggregate_%j.err

set -euo pipefail
ROOT="${PUSH_FOR_ICLR_EXECUTION_ROOT:?PUSH_FOR_ICLR_EXECUTION_ROOT is required}"
PY=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/REPRODUCTIONS/ENVIRONMENTS/des_quality/1d77e0a2a720_532229f679d7_dcba3cb2e282/env/bin/python
cd "$ROOT"
test "$(git rev-parse HEAD)" = "${PUSH_FOR_ICLR_EXPECTED_COMMIT:?PUSH_FOR_ICLR_EXPECTED_COMMIT is required}"
exec "$PY" scripts/push_for_iclr/run_des_sd15_quality.py \
  --config configs/experiments/push_for_iclr/des_sd15_quality_v4.json aggregate
