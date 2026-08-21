#!/usr/bin/env bash
#SBATCH --job-name=iclr-groce-asset
#SBATCH --partition=cpu
#SBATCH --account=ml-staff
#SBATCH --qos=walltime5d
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/groce_asset_%j.out
#SBATCH --error=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/groce_asset_%j.err

set -euo pipefail
test -n "$PUSH_FOR_ICLR_EXECUTION_ROOT"
test -n "$PUSH_FOR_ICLR_METHODS_COMMIT"
actual=$(git -C "$PUSH_FOR_ICLR_EXECUTION_ROOT" rev-parse HEAD)
test "$actual" = "$PUSH_FOR_ICLR_METHODS_COMMIT"
env_root=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/REPRODUCTIONS/ENVIRONMENTS/groce/970ea07eb7ed_readme_cu118_torch271_v2
contract=$PUSH_FOR_ICLR_EXECUTION_ROOT/configs/experiments/push_for_iclr/groce_table1_snoopy_v1.json
exec "$env_root/env/bin/python" "$PUSH_FOR_ICLR_EXECUTION_ROOT/scripts/push_for_iclr/run_groce_reproduction.py" --contract "$contract" fetch-metric-asset
