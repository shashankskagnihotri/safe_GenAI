#!/usr/bin/env bash
#SBATCH --job-name=iclr-groce-gen
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:1
#SBATCH --account=ml-staff
#SBATCH --qos=walltime5d
#SBATCH --array=0-1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/groce_generate_%A_%a.out
#SBATCH --error=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/groce_generate_%A_%a.err

set -euo pipefail
test -n "$PUSH_FOR_ICLR_EXECUTION_ROOT"
test -n "$PUSH_FOR_ICLR_METHODS_COMMIT"
actual=$(git -C "$PUSH_FOR_ICLR_EXECUTION_ROOT" rev-parse HEAD)
test "$actual" = "$PUSH_FOR_ICLR_METHODS_COMMIT"
env_root=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/REPRODUCTIONS/ENVIRONMENTS/groce/970ea07eb7ed_readme_cu118_torch271_v2
contract=$PUSH_FOR_ICLR_EXECUTION_ROOT/configs/experiments/push_for_iclr/groce_table1_snoopy_v1.json
case "$SLURM_ARRAY_TASK_ID" in
  0) mode=original ;;
  1) mode=erased ;;
  *) exit 64 ;;
esac
export PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec "$env_root/env/bin/python" "$PUSH_FOR_ICLR_EXECUTION_ROOT/scripts/push_for_iclr/run_groce_reproduction.py" --contract "$contract" generate --mode "$mode"
