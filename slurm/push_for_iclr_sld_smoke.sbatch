#!/usr/bin/env bash
#SBATCH --job-name=iclr-sld-smoke
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --nice=10000
#SBATCH --output=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/sld_smoke_%j.out
#SBATCH --error=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/sld_smoke_%j.err

set -euo pipefail
REPO_ROOT="${PUSH_FOR_ICLR_EXECUTION_ROOT:-${SLURM_SUBMIT_DIR:-/ceph/sagnihot/projects/safety_genAI_worktrees/push_for_iclr_gpu_reproductions}}"
SLD_PY=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/REPRODUCTIONS/ENVIRONMENTS/sld/a42923c3de0e_numpy1264_hfhub0164_v1/env/bin/python
EVAL_PY=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/REPRODUCTIONS/ENVIRONMENTS/safree/b8b2c3fa9d7f_albucore0016_ortgpu1181_nudenet209_v1/env/bin/python
cd "$REPO_ROOT"
: "${PUSH_FOR_ICLR_EXPECTED_COMMIT:?PUSH_FOR_ICLR_EXPECTED_COMMIT is required}"
test "$(git rev-parse HEAD)" = "$PUSH_FOR_ICLR_EXPECTED_COMMIT"
export PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$SLD_PY" scripts/push_for_iclr/run_sld_table1_hypmax.py \
  --config configs/experiments/push_for_iclr/sld_table1_hypmax_v1.json runtime-check
"$EVAL_PY" scripts/push_for_iclr/evaluate_sld_table1_hypmax.py \
  --config configs/experiments/push_for_iclr/sld_table1_hypmax_v1.json runtime-check
