#!/usr/bin/env bash
set -euo pipefail
umask 077
cd /ceph/sagnihot/projects/safety_genAI
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPYCACHEPREFIX="/tmp/riddikulus_benchmark_${SLURM_JOB_ID:?Scheduler allocation required}"
execution=${1:?Immutable chunk execution required}
source_dir=${2:?Sealed source required}
study_python=${3:?Pinned environment required}
gpus=${4:?GPU count required}
if [[ -e outputs/FINALIZED_WORKING_METHOD_16_OCT/STOP ]]; then exit 1; fi
exec srun --mpi=none --kill-on-bad-exit=1 --exact --nodes=1 --ntasks=1 \
    --cpus-per-task=16 --gpus-per-task="nvidia_h100_nvl:$gpus" --mem-per-gpu=128G \
    "$study_python" "$source_dir/scripts/riddikulus_benchmark/worker.py" --execution "$execution"
