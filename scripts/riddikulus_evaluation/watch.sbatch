#!/bin/bash
#SBATCH --job-name=riddikulus-gemini
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --requeue
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ "$#" -lt 2 ]; then
  echo 'Usage: sbatch [site partition/account options] watch.sbatch PYTHON SEALED_WATCH_PY [watch arguments]' >&2
  exit 2
fi
evaluation_python="$1"
evaluation_watch="$2"
shift 2
exec "$evaluation_python" -u "$evaluation_watch" "$@"
