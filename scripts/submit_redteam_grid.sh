#!/usr/bin/env bash
set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set." >&2
  exit 2
fi

GRID_PATH="${1:?usage: scripts/submit_redteam_grid.sh GRID_PATH ATTEMPT_NAME REPORT_PATH}"
ATTEMPT_NAME="${2:?usage: scripts/submit_redteam_grid.sh GRID_PATH ATTEMPT_NAME REPORT_PATH}"
REPORT_PATH="${3:?usage: scripts/submit_redteam_grid.sh GRID_PATH ATTEMPT_NAME REPORT_PATH}"

sbatch --export=ALL,HF_TOKEN="${HF_TOKEN}",GRID_PATH="${GRID_PATH}",ATTEMPT_NAME="${ATTEMPT_NAME}",REPORT_PATH="${REPORT_PATH}" \
  slurm/redteam_four_models_h100.sbatch
