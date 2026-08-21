#!/usr/bin/env bash
#SBATCH --job-name=iclr-ss-eval
#SBATCH --partition=cpu
#SBATCH --account=ml-staff
#SBATCH --qos=walltime5d
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/semantic_eval_%j.out
#SBATCH --error=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/semantic_eval_%j.err

set -euo pipefail
test -n "$PUSH_FOR_ICLR_EXECUTION_ROOT"
test -n "$PUSH_FOR_ICLR_METHODS_COMMIT"
actual=$(git -C "$PUSH_FOR_ICLR_EXECUTION_ROOT" rev-parse HEAD)
test "$actual" = "$PUSH_FOR_ICLR_METHODS_COMMIT"
env_root=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/REPRODUCTIONS/ENVIRONMENTS/semantic_surgery/816dc9e973e2
contract=$PUSH_FOR_ICLR_EXECUTION_ROOT/configs/experiments/push_for_iclr/semantic_surgery_table2_v1.json
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec "$env_root/env/bin/python" "$PUSH_FOR_ICLR_EXECUTION_ROOT/scripts/push_for_iclr/run_semantic_surgery_reproduction.py" --contract "$contract" evaluate
