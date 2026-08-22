#!/usr/bin/env bash
#SBATCH --job-name=iclr-sld-sheets
#SBATCH --partition=cpu
#SBATCH --array=0-734
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --nice=10000
#SBATCH --output=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/sld_sheet_%A_%a.out
#SBATCH --error=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/sld_sheet_%A_%a.err

set -euo pipefail
REPO_ROOT="${PUSH_FOR_ICLR_EXECUTION_ROOT:-${SLURM_SUBMIT_DIR:-/ceph/sagnihot/projects/safety_genAI_worktrees/push_for_iclr_gpu_reproductions}}"
PYTHON=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/REPRODUCTIONS/ENVIRONMENTS/safree/b8b2c3fa9d7f_albucore0016_ortcpu1181_vendorednudenet_sldimport_v1/env/bin/python
cd "$REPO_ROOT"
: "${PUSH_FOR_ICLR_EXPECTED_COMMIT:?PUSH_FOR_ICLR_EXPECTED_COMMIT is required}"
test "$(git rev-parse HEAD)" = "$PUSH_FOR_ICLR_EXPECTED_COMMIT"
exec "$PYTHON" scripts/push_for_iclr/render_i2p_contact_sheet.py \
  --config configs/experiments/push_for_iclr/sld_table1_hypmax_v1.json \
  --page-index "$SLURM_ARRAY_TASK_ID"
