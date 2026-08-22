#!/usr/bin/env bash
#SBATCH --job-name=iclr-sld-sheets-ok
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --nice=10000
#SBATCH --output=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/sld_sheets_verify_%j.out
#SBATCH --error=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/sld_sheets_verify_%j.err

set -euo pipefail
REPO_ROOT="${PUSH_FOR_ICLR_EXECUTION_ROOT:-${SLURM_SUBMIT_DIR:-/ceph/sagnihot/projects/safety_genAI_worktrees/push_for_iclr_gpu_reproductions}}"
PYTHON=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/REPRODUCTIONS/ENVIRONMENTS/safree/b8b2c3fa9d7f_albucore0016_ortcpu1181_vendorednudenet_sldimport_v1/env/bin/python
cd "$REPO_ROOT"
: "${PUSH_FOR_ICLR_EXPECTED_COMMIT:?PUSH_FOR_ICLR_EXPECTED_COMMIT is required}"
test "$(git rev-parse HEAD)" = "$PUSH_FOR_ICLR_EXPECTED_COMMIT"
exec "$PYTHON" scripts/push_for_iclr/aggregate_i2p_paper_reproduction.py \
  --config configs/experiments/push_for_iclr/sld_table1_hypmax_v1.json \
  --method sld verify-contacts
