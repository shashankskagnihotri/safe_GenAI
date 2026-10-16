#!/usr/bin/env bash
set -euo pipefail
umask 077
cd /ceph/sagnihot/projects/safety_genAI
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
source_dir=${1:?Sealed benchmark source required}
exec /usr/bin/python3 "$source_dir/scripts/riddikulus_benchmark/campaign.py" monitor
