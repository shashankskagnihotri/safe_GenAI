#!/usr/bin/env bash
#SBATCH --job-name=iclr-ss-i2p
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:1
#SBATCH --account=ml-staff
#SBATCH --qos=walltime5d
#SBATCH --array=0-15
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/semantic_i2p_%A_%a.out
#SBATCH --error=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/semantic_i2p_%A_%a.err

set -euo pipefail
test -n "$PUSH_FOR_ICLR_EXECUTION_ROOT"
test -n "$PUSH_FOR_ICLR_METHODS_COMMIT"
actual=$(git -C "$PUSH_FOR_ICLR_EXECUTION_ROOT" rev-parse HEAD)
test "$actual" = "$PUSH_FOR_ICLR_METHODS_COMMIT"
env_root=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/REPRODUCTIONS/ENVIRONMENTS/semantic_surgery/816dc9e973e2
contract=$PUSH_FOR_ICLR_EXECUTION_ROOT/configs/experiments/push_for_iclr/semantic_surgery_table2_v1.json
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec "$env_root/env/bin/python" "$PUSH_FOR_ICLR_EXECUTION_ROOT/scripts/push_for_iclr/run_semantic_surgery_reproduction.py" --contract "$contract" generate-shard --shard-id "$SLURM_ARRAY_TASK_ID"
